from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch

import fgaae
from audit_data import full_digest


ROOT = Path(__file__).resolve().parent


def normalized(values, parameters):
    result = ((np.asarray(values, dtype=np.float64) - parameters["mean"]) / parameters["std"]).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite normalized spectrum.")
    return result


def snapshot(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def parameter_changes(model, before):
    result = {}
    for prefix in ["encoder", "decoder", "teacher", "student", "critic"]:
        squared = sum(
            float((parameter.detach().cpu().double() - before[name].double()).square().sum())
            for name, parameter in model.named_parameters() if name.startswith(prefix + ".")
        )
        result[prefix] = squared ** 0.5
    return result


def encoder_buffer_changes(state, before):
    return {
        name: float((value.detach().cpu().double() - before[name].double()).abs().max())
        for name, value in state.items()
        if name.startswith("encoder.") and name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }


def prepare_inputs(arguments, output):
    dataset = Path(arguments.dataset).resolve()
    benchmark = Path(arguments.benchmark).resolve()
    station_run = Path(arguments.station_run).resolve()
    dataset_manifest = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    protocol = json.loads((benchmark / "protocol.json").read_text(encoding="utf-8"))
    old_station_manifest = json.loads((station_run / "inputs/manifest.json").read_text(encoding="utf-8"))
    if full_digest(dataset / "spectra.npy") != dataset_manifest["spectra_sha256"]:
        raise ValueError("Raw prepared spectra changed.")
    if full_digest(dataset / "partitions.csv") != dataset_manifest["partitions_sha256"]:
        raise ValueError("File-grouped split changed.")
    if full_digest(benchmark / "train.npy") != protocol["train_sha256"]:
        raise ValueError("Frozen synthetic training values changed.")
    if full_digest(station_run / "inputs/development.npy") != old_station_manifest["development_sha256"]:
        raise ValueError("Frozen development values changed.")
    partitions = pd.read_csv(dataset / "partitions.csv").set_index("row")
    station_rows = partitions[(partitions.station == arguments.station) & (partitions.split == "train")]
    groups = station_rows.groupby("source_file").ticks.min().sort_values()
    needed = arguments.reference_train_files + arguments.reference_validation_files
    if len(groups) <= needed:
        raise ValueError("Not enough independent training files for feature references and stage two.")
    reference_train_files = groups.index[:arguments.reference_train_files].tolist()
    reference_validation_files = groups.index[arguments.reference_train_files:needed].tolist()
    reference_train_rows = station_rows.index[station_rows.source_file.isin(reference_train_files)].to_numpy()
    reference_validation_rows = station_rows.index[station_rows.source_file.isin(reference_validation_files)].to_numpy()
    reference_files = set(reference_train_files + reference_validation_files)
    raw = np.load(dataset / "spectra.npy", mmap_mode="r", allow_pickle=False)
    reference_training = raw[reference_train_rows].astype(np.float32)
    reference_validation = raw[reference_validation_rows].astype(np.float32)
    normalization = {
        "mean": float(reference_training.astype(np.float64).mean()),
        "std": float(reference_training.astype(np.float64).std()),
        "fitted_on": "feature_reference_training_files_only",
    }
    if normalization["std"] < 1e-8:
        raise ValueError("Reference spectra have zero variance.")
    training_metadata = pd.read_csv(benchmark / "train_metadata.csv")
    joined = training_metadata.join(partitions[["source_file", "station", "split"]], on="source_row", validate="one_to_one")
    eligible = (joined.station == arguments.station) & (joined.split == "train") & ~joined.source_file.isin(reference_files)
    positions = np.flatnonzero(eligible.to_numpy())
    if arguments.train_size:
        if len(positions) < arguments.train_size:
            raise ValueError("Requested more training rows than remain after reference-file exclusion.")
        positions = positions[:arguments.train_size]
    selected_metadata = joined.iloc[positions].copy()
    training = np.load(benchmark / "train.npy", mmap_mode="r", allow_pickle=False)[positions].astype(np.float32)
    development_metadata = pd.read_csv(station_run / "inputs/development_metadata.csv")
    if not partitions.loc[development_metadata.source_row, "station"].eq(arguments.station).all():
        raise ValueError("Development station does not match the selected station.")
    if not partitions.loc[development_metadata.source_row, "split"].eq("validation").all():
        raise ValueError("Development accidentally contains test rows.")
    development_files = set(partitions.loc[development_metadata.source_row, "source_file"])
    if development_files.intersection(set(selected_metadata.source_file) | reference_files):
        raise ValueError("A file crosses reference/training/development boundaries.")
    development = np.load(station_run / "inputs/development.npy", allow_pickle=False)
    output.mkdir()
    selected_metadata.to_csv(output / "training_membership.csv", index=False, encoding="utf-8-sig")
    development_metadata.to_csv(output / "development_membership.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "station": arguments.station, "training_rows": len(training), "development_rows": len(development),
        "reference_training_rows": reference_train_rows.astype(int).tolist(),
        "reference_validation_rows": reference_validation_rows.astype(int).tolist(),
        "reference_training_files": reference_train_files, "reference_validation_files": reference_validation_files,
        "normalization": normalization,
        "dataset_sha256": dataset_manifest["spectra_sha256"],
        "frozen_training_sha256": protocol["train_sha256"],
        "development_sha256": old_station_manifest["development_sha256"],
        "protocol_id": protocol["protocol_id"], "synthetic_training_fraction": float(selected_metadata.label.mean()),
        "reference_normality": "Unmodified observed background, not human-confirmed normal ground truth.",
        "confirmation_or_test_accessed": False,
    }
    fgaae.write_json(output / "manifest.json", manifest)
    print(f"FULL MODEL DATA: feature_train={len(reference_training)}, feature_validation={len(reference_validation)}, stage2={len(training)}, development={len(development)}", flush=True)
    return {
        "manifest": manifest, "reference_training": reference_training,
        "reference_validation": reference_validation, "training": training,
        "development": development, "labels": development_metadata.label.to_numpy(dtype=np.int64),
        "source_rows": selected_metadata.source_row.to_numpy(dtype=np.int64),
        "development_rows": development_metadata.source_row.to_numpy(dtype=np.int64),
        "injected_labels": selected_metadata.label.to_numpy(dtype=np.int64),
    }


@torch.no_grad()
def teacher_metrics(model, loader, feature_names, device):
    model.eval()
    squared_errors, naive_errors = [], []
    for spectra, targets in loader:
        targets = targets.to(device)
        predicted = model.teacher(model.encoder(spectra.to(device)))
        squared_errors.append((predicted - targets).square().cpu().numpy())
        naive_errors.append(targets.square().cpu().numpy())
    learned = np.concatenate(squared_errors)
    naive = np.concatenate(naive_errors)
    frame = pd.DataFrame({
        "feature": feature_names, "teacher_normalized_mse": learned.mean(axis=0),
        "training_mean_predictor_mse": naive.mean(axis=0),
    })
    report = {
        "teacher_validation_mse": float(learned.mean()),
        "training_mean_predictor_validation_mse": float(naive.mean()),
        "relative_error_reduction": float(1 - learned.mean() / max(float(naive.mean()), 1e-12)),
        "teacher_outperforms_training_mean_predictor": bool(learned.mean() < naive.mean()),
    }
    return frame, report


def alternative_score_diagnostics(result, labels, config):
    weight = config["critic_score_weight"]
    rows = []
    for name, scores in [
        ("registered_literal_eq11", result["scores"]),
        ("literal_reconstruction_sum_only_diagnostic", result["reconstruction_score"]),
        ("semantic_negative_critic_diagnostic", result["reconstruction_score"] - weight * result["critic_value"]),
    ]:
        threshold = fgaae.select_threshold(scores, labels, {"threshold_method": "validation-f1"})
        rows.append({"interpretation": name, **fgaae.metrics(labels, scores, threshold)})
    return rows


def select_original_score(result, reference, labels, mask, relatives):
    """Joint original-score coefficient selection; check-file labels are never read."""
    ratio = float(reference["reconstruction_score"].std()) / max(float(reference["critic_value"].std()), 1e-8)
    if not np.isfinite(ratio) or ratio <= 0 or any(not np.isfinite(value) or value <= 0 for value in relatives):
        raise ValueError("Positive finite reference scales and coefficient candidates required.")
    best = None
    for relative, weight in [(0.0, 0.1), *[(value, value * ratio) for value in relatives]]:
        scores = (result["reconstruction_score"] + weight * result["critic_value"])[mask]
        threshold = fgaae.select_threshold(scores, labels[mask], {"threshold_method": "validation-f1"})
        measurement = fgaae.metrics(labels[mask], scores, threshold)
        if best is None or measurement["f1"] > best["score"] + 1e-12:
            best = {
                "score": measurement["f1"], "config_updates": {"critic_score_weight": weight},
                "record": {"selection_f1": measurement["f1"], "selection_precision": measurement["precision"],
                           "selection_recall": measurement["recall"], "selection_auc": measurement["roc_auc"],
                           "selection_critic_weight": weight, "selection_relative_weight": relative,
                           "selection_threshold": threshold},
            }
    return best


def train_seed(data, seed, arguments, output):
    output.mkdir(parents=True)
    device = torch.device(arguments.device)
    fgaae.seed_everything(seed, arguments.num_threads)
    config = vars(fgaae.build_parser().parse_args(["train"]))
    config.update({
        "data": str(Path(arguments.benchmark).resolve() / "train.npy"),
        "output": str(output.resolve()), "sequence_length": 840, "seed": seed,
        "device": str(device), "num_threads": arguments.num_threads,
        "stage1_epochs": arguments.stage1_epochs, "iterations": arguments.iterations,
        "epochs_per_iteration": arguments.epochs_per_iteration,
        "batch_size": arguments.batch_size, "critic_steps": arguments.critic_steps,
        "use_aqt": True, "warmup_iterations": 0,
        "reconstruction_weight": 1.0, "adversarial_weight": arguments.adversarial_weight, "distillation_weight": arguments.distillation_weight,
        "gp_weight": 10.0, "retention_start": 0.95, "retention_end": arguments.retention_end,
        "retention_alpha": 5.0, "retention_schedule": "exponential",
        "early_stopping_patience": 0, "minimum_epochs": 1,
        "critic_score_sign": 1, "critic_score_weight": 0.1,
        "reconstruction_score_reduction": "sum", "standardize_scores": False,
        "threshold_method": "validation-f1", "evaluation": "synthetic",
        "encoder_lr": arguments.encoder_lr, "freeze_encoder_bn": arguments.freeze_encoder_bn,
        "monitor_encoder_gradients": arguments.monitor_encoder_gradients,
        "training_profile": arguments.training_profile,
        "consistent_generator_forward": arguments.consistent_generator_forward,
        "checkpoint_selection_metric": arguments.checkpoint_selection_metric,
    })
    fgaae.validate_config(config)
    if config["adversarial_weight"] <= 0 or config["distillation_weight"] <= 0:
        raise ValueError("Complete FGAAE requires positive adversarial and distillation weights.")
    model = fgaae.FGAAE(config).to(device)
    initial = snapshot(model)
    parameters = data["manifest"]["normalization"]
    reference_training = normalized(data["reference_training"], parameters)
    reference_validation = normalized(data["reference_validation"], parameters)
    training_features = fgaae.expert_features(data["reference_training"], config).astype(np.float64)
    validation_features = fgaae.expert_features(data["reference_validation"], config).astype(np.float64)
    feature_mean = training_features.mean(axis=0)
    feature_std = training_features.std(axis=0)
    feature_std[feature_std < 1e-8] = 1
    stage1_training = fgaae.make_loader(reference_training, arguments.stage1_batch_size, (training_features - feature_mean) / feature_std, shuffle=True)
    stage1_validation = fgaae.make_loader(reference_validation, arguments.stage1_batch_size, (validation_features - feature_mean) / feature_std)
    fgaae.write_json(output / "config.json", config)
    pretraining_provenance = None
    if arguments.pretrained_run:
        previous = Path(arguments.pretrained_run).resolve()
        previous_manifest = json.loads((previous / "inputs/manifest.json").read_text(encoding="utf-8"))
        for key in ["reference_training_rows", "reference_validation_rows", "normalization", "dataset_sha256", "frozen_training_sha256", "development_sha256"]:
            if previous_manifest[key] != data["manifest"][key]:
                raise ValueError(f"Pretrained run has different data: {key}")
        previous_membership = pd.read_csv(previous / "inputs/training_membership.csv")
        if not np.array_equal(previous_membership.source_row, data["source_rows"]):
            raise ValueError("Pretrained comparison requires identical stage-two membership.")
        previous_seed = previous / f"seed_{seed}"
        previous_checkpoint = torch.load(previous_seed / "model.pt", map_location="cpu", weights_only=True)
        for key in ["sequence_length", "latent_dim", "strides", "level_scale", "units", "start_mhz", "end_mhz", "peak_prominence", "occupancy_margin", "stage1_epochs", "stage1_lr"]:
            if previous_checkpoint["config"][key] != config[key]:
                raise ValueError(f"Incompatible pretraining setting: {key}")
        if not np.allclose(feature_mean, previous_checkpoint["feature_mean"], rtol=1e-12, atol=1e-12) or not np.allclose(feature_std, previous_checkpoint["feature_std"], rtol=1e-12, atol=1e-12):
            raise ValueError("Expert feature definitions or normalization changed.")
        checkpoint_path = previous_seed / "stage1_best.pt"
        pretrained = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.encoder.load_state_dict(pretrained["encoder"])
        model.teacher.load_state_dict(pretrained["teacher"])
        model.freeze_teacher()
        for name in ["stage1_best.pt", "stage1_history.csv"]:
            shutil.copy2(previous_seed / name, output / name)
        pretraining_provenance = {"checkpoint": str(checkpoint_path), "sha256": full_digest(checkpoint_path)}
        print(f"REUSED FEATURE PRETRAINING seed={seed}: {checkpoint_path}", flush=True)
        del previous_checkpoint, pretrained
    else:
        fgaae.train_stage_one(model, stage1_training, stage1_validation, config, device, output)
    feature_frame, feature_report = teacher_metrics(model, stage1_validation, fgaae.FEATURE_NAMES, device)
    feature_frame.to_csv(output / "teacher_feature_diagnostics.csv", index=False)
    stage1_changes = parameter_changes(model, initial)
    feature_report.update({
        "stage1_batches_per_epoch": len(stage1_training),
        "stage1_optimizer_updates": len(stage1_training) * arguments.stage1_epochs,
        "parameter_changes": stage1_changes,
        "stage1_optimizer_updates_this_run": 0 if pretraining_provenance else len(stage1_training) * arguments.stage1_epochs,
        "pretraining_reused_from": pretraining_provenance,
    })
    fgaae.write_json(output / "teacher_health.json", feature_report)
    print(f"TEACHER HEALTH seed={seed}: {json.dumps(feature_report)}", flush=True)
    stage1_state = snapshot(model)
    training = normalized(data["training"], parameters)
    development = normalized(data["development"], parameters)
    selection_mask = np.ones(len(development), dtype=bool)
    selection_files = None
    if arguments.selection_earlier_files:
        metadata = pd.read_csv(output.parent / "inputs/development_membership.csv")
        files = metadata.groupby("source_file").ticks.min().sort_values().index.tolist()
        if len(files) < 2:
            raise ValueError("Grouped checkpoint selection requires multiple development files.")
        selection_files = files[:len(files) // 2]
        selection_mask = metadata.source_file.isin(selection_files).to_numpy()
        if len(np.unique(data["labels"][selection_mask])) != 2:
            raise ValueError("Checkpoint selection requires both classes in earlier files.")
    config["checkpoint_selection_files"] = selection_files
    selector = None
    if arguments.selection_relative_critic_weights:
        if arguments.checkpoint_selection_metric != "f1":
            raise ValueError("Joint coefficient/checkpoint selection uses F1.")
        config["selection_relative_critic_weights"] = arguments.selection_relative_critic_weights
        def selector(current_model, result):
            # Evaluation creates DataLoader iterators; preserve RNG for paired training.
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                reference_result = fgaae.evaluate(current_model, reference_training, config, device)
            return select_original_score(result, reference_result, data["labels"], selection_mask, arguments.selection_relative_critic_weights)
    elif arguments.selection_earlier_files:
        def selector(current_model, result):
            scores = result["scores"][selection_mask]
            labels = data["labels"][selection_mask]
            threshold = fgaae.select_threshold(scores, labels, config)
            values = fgaae.metrics(labels, scores, threshold)
            return {"score": values[arguments.checkpoint_selection_metric], "record": {"selection_f1": values["f1"]}}
    def monitor_reference(current_model, epoch):
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            _, health = teacher_metrics(current_model, stage1_validation, fgaae.FEATURE_NAMES, device)
        return {"reference_teacher_feature_mse": health["teacher_validation_mse"]}

    fgaae.seed_everything(seed + 10000, arguments.num_threads)
    selection = fgaae.train_stage_two(
        model, training, development, data["labels"], data["source_rows"], config, device, output,
        epoch_callback=monitor_reference if arguments.monitor_feature_alignment else None,
        checkpoint_selector=selector,
    )
    fgaae.write_json(output / "config.json", config)
    changes = parameter_changes(model, stage1_state)
    for prefix in ["encoder", "decoder", "student", "critic"]:
        if changes[prefix] <= 0:
            raise RuntimeError(f"Full-model component did not update: {prefix}")
    if changes["teacher"] != 0:
        raise RuntimeError("The teacher head was not frozen in stage two.")
    state = snapshot(model)
    selected_buffers = encoder_buffer_changes(state, stage1_state)
    latest_checkpoint = torch.load(output / "stage2_latest.pt", map_location="cpu", weights_only=True)
    latest_buffers = encoder_buffer_changes(latest_checkpoint["model_state"], stage1_state)
    if arguments.freeze_encoder_bn and any(value != 0 for value in [*selected_buffers.values(), *latest_buffers.values()]):
        raise RuntimeError("Frozen encoder BatchNorm statistics changed.")
    del latest_checkpoint
    result = fgaae.evaluate(model, development, config, device)
    threshold = fgaae.select_threshold(result["scores"][selection_mask], data["labels"][selection_mask], config)
    measurement = fgaae.metrics(data["labels"], result["scores"], threshold)
    fgaae.save_predictions(output / "development_scores.csv", data["development_rows"], data["labels"], None, result, threshold)
    pd.DataFrame(alternative_score_diagnostics(result, data["labels"], config)).to_csv(output / "eq11_diagnostic_interpretations.csv", index=False)
    pd.DataFrame({
        "source_row": data["development_rows"], "sum_squared_error": result["reconstruction_score"],
        "mean_squared_error": result["reconstruction_error"], "critic": result["critic_value"],
        "literal_eq11_score": result["scores"],
    }).to_csv(output / "score_components.csv", index=False)
    torch.save({
        "artifact_type": arguments.training_profile, "config": config, "model_state": state,
        "normalization": parameters, "threshold": threshold,
        "feature_mean": feature_mean.tolist(), "feature_std": feature_std.tolist(),
        "feature_names": fgaae.FEATURE_NAMES, "station": data["manifest"]["station"],
        "all_components_enabled": True, "primary_score": f"sum_squared_error + {config['critic_score_weight']:.17g} * critic(reconstruction)",
    }, output / "model.pt")
    retention = []
    truth = pd.DataFrame({"source_row": data["source_rows"], "injected": data["injected_labels"]})
    for path in sorted(output.glob("aqt_iteration_*.csv")):
        selected = pd.read_csv(path).merge(truth, on="source_row", validate="one_to_one")
        kept = selected[selected.retained == 1]
        retention.append({"iteration": int(path.stem.rsplit("_", 1)[1]), "retained": len(kept), "injected_fraction": float(kept.injected.mean()), "diagnostic_only": True})
    pd.DataFrame(retention).to_csv(output / "aqt_purity_diagnostics.csv", index=False)
    history = pd.read_csv(output / "stage2_history.csv")
    if not np.isfinite(history.select_dtypes(include="number").to_numpy()).all():
        raise FloatingPointError("Full training history contains non-finite values.")
    audit = {
        "all_components_enabled": True, "adversarial_weight": config["adversarial_weight"],
        "distillation_weight": config["distillation_weight"], "critic_steps": config["critic_steps"],
        "aqt_rounds": len(retention), "parameter_changes_stage2_selected_model": changes,
        "teacher_frozen": True, "score_reduction": "sum", "score_sign": 1,
        "pretraining_and_full_training_completed": True,
        "encoder_learning_rate": config.get("encoder_lr") or config["stage2_lr"],
        "decoder_student_learning_rate": config["stage2_lr"],
        "encoder_bn_statistics_frozen": arguments.freeze_encoder_bn,
        "encoder_bn_buffer_changes_selected": selected_buffers,
        "encoder_bn_buffer_changes_latest": latest_buffers,
    }
    fgaae.write_json(output / "component_audit.json", audit)
    fgaae.write_json(output / "report.json", {
        "seed": seed, "development": measurement, "selection": selection,
        "teacher_health": feature_report, "component_audit": audit,
        "confirmation_or_test_evaluated": False,
    })
    plot_history(output)
    del model, training, development
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"seed": seed, **measurement, "best_epoch": selection["best_epoch"], "teacher_relative_error_reduction": feature_report["relative_error_reduction"]}


def plot_history(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feature = pd.read_csv(output / "stage1_history.csv")
    joint = pd.read_csv(output / "stage2_history.csv")
    figure, axes = plt.subplots(1, 3, figsize=(17, 4))
    axes[0].plot(feature.epoch, feature.train_feature_loss, label="Train")
    axes[0].plot(feature.epoch, feature.validation_feature_loss, label="Internal validation")
    axes[0].set_title("Feature pretraining")
    axes[0].legend()
    for name in ["train_reconstruction_loss", "train_distillation_loss"]:
        axes[1].plot(joint.epoch, joint[name], label=name)
    axes[1].set_title("All-module joint training")
    axes[1].legend()
    axes[2].plot(joint.epoch, joint.validation_roc_auc, label="Registered Eq.11 score")
    axes[2].set_title("Development ROC-AUC (not test)")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output / "full_model_training.png", dpi=160)
    plt.close(figure)


def run(arguments):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    positive = [arguments.reference_train_files, arguments.reference_validation_files, arguments.stage1_epochs, arguments.stage1_batch_size, arguments.batch_size, arguments.iterations, arguments.epochs_per_iteration, arguments.critic_steps]
    if any(value < 1 for value in positive) or arguments.train_size < 0:
        raise ValueError("Training sizes and iterations must be positive (train-size 0 means all eligible rows).")
    if not arguments.seeds or len(set(arguments.seeds)) != len(arguments.seeds):
        raise ValueError("Use a nonempty set of distinct seeds.")
    output = Path(arguments.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    for name in ["fgaae.py", "full_model.py"]:
        shutil.copy2(ROOT / name, source / name)
    fgaae.seed_everything(arguments.seeds[0], arguments.num_threads)
    plan = {
        **vars(arguments), "scope": "Complete FGAAE only; no ablations, AE replacement, local top-k score or nonlearned detector selection.",
        "primary_score": "Equation 11 literal: sum squared error + positive lambda * C(reconstruction), on training-standardized spectra. Default lambda=0.1; optional joint F1 calibration is explicitly registered.",
        "score_caveat": "Paper's positive critic sign conflicts with its higher-score-is-more-real description. Literal sign is registered; the negative sign is diagnostic only, not selected post hoc.",
        "teacher_policy": "Freeze feature head only after training, keep shared encoder trainable as described by paper. Student independently initialized.",
        "architecture_caveat": "Table-1 strides [4,2,2]; text says [2,2,2]. Critic LayerNorm replaces text BatchNorm for per-sample GP.",
        "unknown_parameters": "Loss weights, critic-score weight, retention endpoints/decay and outer iteration count are explicit reconstruction assumptions, not recovered original configuration.",
        "feature_caveat": "25 explicit expert features are surrogate definitions, not a recovered original list.",
        "data_caveat": "Same frozen synthetic power-injection proxy; no claim of original channel-SNR or original anomaly labels.",
        "evaluation": "Existing development files only; no confirmation/test/external-test scoring.",
        "full_model_code_sha256": full_digest(Path(__file__)), "core_code_sha256": full_digest(ROOT / "fgaae.py"),
        "source_snapshot": "source/", "pretraining_reuse": arguments.pretrained_run,
        "gradient_monitoring": "Weighted encoder gradient norms and cosines on the first batch per epoch; diagnostic only.",
    }
    fgaae.write_json(output / "registered_plan.json", plan)
    fgaae.write_json(output / "environment.json", {"torch": str(torch.__version__), "numpy": np.__version__, "pandas": pd.__version__, "cuda": torch.version.cuda, "device": arguments.device})
    data = prepare_inputs(arguments, output / "inputs")
    rows = []
    for seed in arguments.seeds:
        rows.append(train_seed(data, seed, arguments, output / f"seed_{seed}"))
        pd.DataFrame(rows).to_csv(output / "development_runs.csv", index=False)
    frame = pd.DataFrame(rows)
    summary = {
        "model": "complete FGAAE (feature pretraining + distillation + WGAN-GP + AQT)",
        "seeds": arguments.seeds, "primary_score": plan["primary_score"],
        "development_auc_mean": float(frame.roc_auc.mean()),
        "development_auc_sample_std": float(frame.roc_auc.std(ddof=1)) if len(frame) > 1 else 0.0,
        "development_ap_mean": float(frame.average_precision.mean()),
        "development_f1_mean": float(frame.f1.mean()), "runs": rows,
        "all_components_verified": True, "confirmation_or_test_evaluated": False,
        "original_paper_performance_reproduced": False,
    }
    fgaae.write_json(output / "full_model_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"FULL FGAAE COMPLETE: {output}", flush=True)
    return output


def smoke_test(stabilized=False):
    from experiment_pipeline import power_interference

    with tempfile.TemporaryDirectory(prefix="full_fgaae_") as temporary:
        directory = Path(temporary)
        dataset, benchmark, old_station = directory / "dataset", directory / "benchmark", directory / "station"
        dataset.mkdir()
        benchmark.mkdir()
        (old_station / "inputs").mkdir(parents=True)
        generator = np.random.default_rng(2025)
        raw = generator.normal(200, 20, (256, 840)).astype(np.int16)
        np.save(dataset / "spectra.npy", raw)
        partitions = pd.DataFrame({
            "row": np.arange(256), "source_file": [f"file_{index // 32}" for index in range(256)],
            "station": "fixture", "ticks": np.arange(256),
            "split": ["train"] * 192 + ["validation"] * 64,
        })
        partitions.to_csv(dataset / "partitions.csv", index=False)
        fgaae.write_json(dataset / "dataset.json", {"spectra_sha256": full_digest(dataset / "spectra.npy"), "partitions_sha256": full_digest(dataset / "partitions.csv")})
        training, _, metadata = power_interference(raw[:192], np.arange(192), 0.05, 2025)
        np.save(benchmark / "train.npy", training)
        metadata.to_csv(benchmark / "train_metadata.csv", index=False)
        fgaae.write_json(benchmark / "protocol.json", {"protocol_id": "local-power-injection-v1", "train_sha256": full_digest(benchmark / "train.npy")})
        development, _, metadata = power_interference(raw[192:], np.arange(192, 256), 0.25, 2026)
        np.save(old_station / "inputs/development.npy", development)
        metadata.to_csv(old_station / "inputs/development_metadata.csv", index=False)
        fgaae.write_json(old_station / "inputs/manifest.json", {"development_sha256": full_digest(old_station / "inputs/development.npy")})
        arguments = build_parser().parse_args([
            "train", "--dataset", str(dataset), "--benchmark", str(benchmark),
            "--station-run", str(old_station), "--station", "fixture", "--output", str(directory / "run"),
            "--reference-train-files", "1", "--reference-validation-files", "1",
            "--train-size", "64", "--stage1-epochs", "1", "--stage1-batch-size", "16",
            "--batch-size", "16", "--iterations", "2", "--epochs-per-iteration", "1",
            "--critic-steps", "1", "--seeds", "2025", "--device", "cpu",
        ])
        output = run(arguments)
        if stabilized:
            arguments.pretrained_run = str(output)
            arguments.output = str(directory / "stable")
            arguments.encoder_lr = 2e-6
            arguments.freeze_encoder_bn = True
            arguments.retention_end = 0.85
            arguments.monitor_encoder_gradients = True
            arguments.monitor_feature_alignment = True
            arguments.training_profile = "complete_fgaae_v5_stabilized"
            output = run(arguments)
            audit = json.loads((output / "seed_2025/component_audit.json").read_text(encoding="utf-8"))
            assert all(value == 0 for value in audit["encoder_bn_buffer_changes_latest"].values())
            health = json.loads((output / "seed_2025/teacher_health.json").read_text(encoding="utf-8"))
            assert health["stage1_optimizer_updates_this_run"] == 0
            history = pd.read_csv(output / "seed_2025/stage2_history.csv")
            assert history.encoder_learning_rate.eq(2e-6).all()
            assert history.retention_rate.min() > 0.85
            assert history.encoder_gradient_reconstruction_norm.gt(0).all()
            assert np.isfinite(history.reference_teacher_feature_mse).all()
        checkpoint = torch.load(output / "seed_2025/model.pt", map_location="cpu", weights_only=True)
        assert checkpoint["all_components_enabled"] and checkpoint["config"]["use_aqt"]
        assert checkpoint["config"]["adversarial_weight"] > 0 and checkpoint["config"]["distillation_weight"] > 0
        model = fgaae.FGAAE(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state"])
        data = normalized(development[:4], checkpoint["normalization"])
        result = fgaae.evaluate(model, data, checkpoint["config"], torch.device("cpu"))
        assert np.allclose(result["reconstruction_score"], 840 * result["reconstruction_error"])
        assert np.allclose(result["scores"], result["reconstruction_score"] + 0.1 * result["critic_value"])
        assert not list(output.rglob("confirmation_scores.csv"))
        print("FULL MODEL SMOKE TEST PASSED: all modules active, teacher frozen, shared file isolation, exact Eq.11 sum/sign and checkpoint load.")


def build_parser(training_defaults=None):
    parser = argparse.ArgumentParser(description="Train the complete FGAAE only, with explicit paper-equation scoring and feature-pretraining health checks.")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--dataset", default=str(ROOT / "datasets/fm_87_108_v1"))
    training.add_argument("--benchmark", default=str(ROOT / "runs/grouped_pipeline_v1/benchmark"))
    training.add_argument("--station-run", default=str(ROOT / "runs/station_background_v2"))
    training.add_argument("--station", default="龙井三合海关30-3000")
    training.add_argument("--output", default=str(ROOT / "runs/full_fgaae_v4"))
    training.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    training.add_argument("--num-threads", type=int, default=2)
    training.add_argument("--reference-train-files", type=int, default=5)
    training.add_argument("--reference-validation-files", type=int, default=1)
    training.add_argument("--train-size", type=int, default=20000)
    training.add_argument("--stage1-epochs", type=int, default=100)
    training.add_argument("--stage1-batch-size", type=int, default=32)
    training.add_argument("--batch-size", type=int, default=256)
    training.add_argument("--iterations", type=int, default=3)
    training.add_argument("--epochs-per-iteration", type=int, default=20)
    training.add_argument("--critic-steps", type=int, default=5)
    training.add_argument("--seeds", type=int, nargs="+", default=[2025, 2026, 2027])
    training.add_argument("--encoder-lr", type=float)
    training.add_argument("--freeze-encoder-bn", action="store_true")
    training.add_argument("--retention-end", type=float, default=0.5)
    training.add_argument("--adversarial-weight", type=float, default=0.01)
    training.add_argument("--distillation-weight", type=float, default=1.0)
    training.add_argument("--monitor-encoder-gradients", action="store_true")
    training.add_argument("--monitor-feature-alignment", action="store_true")
    training.add_argument("--pretrained-run")
    training.add_argument("--training-profile", default="complete_fgaae_v4")
    training.add_argument("--consistent-generator-forward", action="store_true")
    training.add_argument("--checkpoint-selection-metric", choices=["roc_auc", "f1"], default="roc_auc")
    training.add_argument("--selection-earlier-files", action="store_true")
    training.add_argument("--selection-relative-critic-weights", type=float, nargs="+")
    if training_defaults:
        training.set_defaults(**training_defaults)
    commands.add_parser("smoke-test")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.command == "smoke-test":
        smoke_test()
    else:
        run(parsed)
