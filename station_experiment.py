from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch

import fgaae
from audit_data import full_digest


ROOT = Path(__file__).resolve().parent
METHODS = ["global_zscore_ae", "background_ae", "background_featureguided"]
BASELINES = ["mean_level", "total_power_db", "background_mean_z", "background_zero_mse", "background_top5_positive_z"]


def fit_transform(reference, mode, floor_db):
    levels = np.asarray(reference, dtype=np.float64) * 0.1
    if mode == "background":
        lower, center, upper = np.quantile(levels, [0.25, 0.5, 0.75], axis=0)
        scale = np.maximum((upper - lower) / 1.3489795003921634, floor_db)
    else:
        center = np.full(levels.shape[1], float(levels.mean()))
        scale = np.full(levels.shape[1], max(float(levels.std()), floor_db))
    return {
        "mode": mode, "center_db": center.tolist(), "scale_db": scale.tolist(),
        "input_level_scale": 0.1, "floor_db": floor_db, "fitted_on": "reference_training_only",
    }


def transform(values, parameters):
    center = np.asarray(parameters["center_db"], dtype=np.float32)
    scale = np.asarray(parameters["scale_db"], dtype=np.float32)
    result = (np.asarray(values, dtype=np.float32) * parameters["input_level_scale"] - center) / scale
    if not np.isfinite(result).all():
        raise ValueError("Background transformation produced non-finite values.")
    return np.ascontiguousarray(result, dtype=np.float32)


def baseline_scores(values, background):
    levels = np.asarray(values, dtype=np.float64) * 0.1
    maximum = levels.max(axis=1)
    total_power_db = maximum + 10 * np.log10((10 ** ((levels - maximum[:, None]) / 10)).sum(axis=1))
    residual = transform(values, background).astype(np.float64)
    positive = np.maximum(residual, 0)
    tail_count = max(1, round(values.shape[1] * 0.05))
    tail = np.partition(positive, positive.shape[1] - tail_count, axis=1)[:, -tail_count:]
    return {
        "mean_level": levels.mean(axis=1), "total_power_db": total_power_db,
        "background_mean_z": residual.mean(axis=1),
        "background_zero_mse": np.mean(residual ** 2, axis=1),
        "background_top5_positive_z": tail.mean(axis=1),
    }


@torch.no_grad()
def model_scores(model, spectra, batch_size, device):
    model.eval()
    scores, distillation = [], []
    for (batch,) in fgaae.make_loader(spectra, batch_size):
        batch = batch.to(device)
        reconstruction, latent = model(batch)
        scores.extend((reconstruction - batch).square().mean(dim=(1, 2)).cpu().tolist())
        distillation.extend((model.student(latent) - model.teacher(latent)).square().mean(dim=1).cpu().tolist())
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise FloatingPointError("Non-finite reconstruction score.")
    return scores, float(np.mean(distillation))


def prepare_inputs(arguments, output):
    dataset = Path(arguments.dataset).resolve()
    benchmark = Path(arguments.benchmark).resolve()
    dataset_report = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    protocol = json.loads((benchmark / "protocol.json").read_text(encoding="utf-8"))
    if full_digest(dataset / "spectra.npy") != dataset_report["spectra_sha256"]:
        raise ValueError("Dataset checksum changed.")
    if full_digest(dataset / "partitions.csv") != dataset_report["partitions_sha256"]:
        raise ValueError("Partition checksum changed.")
    for split_name in ["train", "validation"]:
        if full_digest(benchmark / f"{split_name}.npy") != protocol[f"{split_name}_sha256"]:
            raise ValueError("The inherited frozen training/development benchmark changed.")
    partitions = pd.read_csv(dataset / "partitions.csv")
    station = partitions[partitions.station == arguments.station].copy()
    source_training = station[station.split == "train"]
    source_validation = station[station.split == "validation"]
    if source_training.source_file.nunique() < 2 or source_validation.source_file.nunique() < 2:
        raise ValueError("Need at least two training files and two validation files for a station.")
    training_groups = source_training.groupby("source_file").ticks.min().sort_values()
    reference_file = str(training_groups.index[0])
    reference_rows = source_training.loc[source_training.source_file == reference_file, "row"].to_numpy()
    internal_validation_count = max(2, round(len(reference_rows) * 0.25))
    reference_training_rows = reference_rows[:-internal_validation_count]
    reference_validation_rows = reference_rows[-internal_validation_count:]
    source_values = np.load(dataset / "spectra.npy", mmap_mode="r", allow_pickle=False)
    reference_training = source_values[reference_training_rows].astype(np.float32)
    reference_validation = source_values[reference_validation_rows].astype(np.float32)
    global_parameters = fit_transform(reference_training, "global", arguments.scale_floor_db)
    background_parameters = fit_transform(reference_training, "background", arguments.scale_floor_db)
    train_metadata = pd.read_csv(benchmark / "train_metadata.csv")
    training_positions = np.flatnonzero(train_metadata.source_row.isin(
        source_training.loc[source_training.source_file != reference_file, "row"]
    ).to_numpy())
    if len(training_positions) < arguments.train_size:
        raise ValueError("Not enough frozen stage-two station rows for the requested pilot.")
    frozen_training = np.load(benchmark / "train.npy", mmap_mode="r", allow_pickle=False)
    training = frozen_training[training_positions].astype(np.float32)
    training_members = train_metadata.iloc[training_positions].copy()
    validation_metadata = pd.read_csv(benchmark / "validation_metadata.csv")
    enriched_validation = validation_metadata.join(
        partitions.set_index("row")[["source_file", "station", "ticks"]],
        on="source_row", how="left", validate="one_to_one",
    )
    station_validation = enriched_validation[enriched_validation.station == arguments.station]
    validation_groups = station_validation.groupby("source_file").ticks.min().sort_values()
    confirmation_count = max(1, round(len(validation_groups) * arguments.confirmation_fraction))
    confirmation_files = set(validation_groups.index[-confirmation_count:])
    development_files = set(validation_groups.index[:-confirmation_count])
    development_positions = station_validation.index[station_validation.source_file.isin(development_files)].to_numpy()
    confirmation_positions = station_validation.index[station_validation.source_file.isin(confirmation_files)].to_numpy()
    inherited_validation = np.load(benchmark / "validation.npy", mmap_mode="r", allow_pickle=False)
    development = inherited_validation[development_positions].astype(np.float32)
    confirmation = inherited_validation[confirmation_positions].astype(np.float32)
    development_metadata = enriched_validation.iloc[development_positions].copy()
    confirmation_metadata = enriched_validation.iloc[confirmation_positions].copy()
    if np.intersect1d(training_members.source_row, development_metadata.source_row).size:
        raise ValueError("Training rows overlap development rows.")
    if np.intersect1d(development_metadata.source_row, confirmation_metadata.source_row).size:
        raise ValueError("Development and confirmation rows overlap.")
    training_files = set(source_training.source_file) - {reference_file}
    if training_files.intersection(development_files | confirmation_files):
        raise ValueError("A source file crossed development boundaries.")
    if development_files.intersection(confirmation_files):
        raise ValueError("A source file crossed confirmation boundaries.")
    output.mkdir()
    np.save(output / "reference_train.npy", reference_training, allow_pickle=False)
    np.save(output / "reference_validation.npy", reference_validation, allow_pickle=False)
    np.save(output / "train.npy", training, allow_pickle=False)
    np.save(output / "development.npy", development, allow_pickle=False)
    np.save(output / "confirmation.npy", confirmation, allow_pickle=False)
    training_members.to_csv(output / "train_metadata.csv", index=False)
    development_metadata.to_csv(output / "development_metadata.csv", index=False)
    confirmation_metadata.to_csv(output / "confirmation_metadata.csv", index=False)
    manifest = {
        "station": arguments.station, "protocol_id": protocol["protocol_id"],
        "reference_source_file": reference_file,
        "reference_train_rows": reference_training_rows.astype(int).tolist(),
        "reference_validation_rows": reference_validation_rows.astype(int).tolist(),
        "train_pool_rows": len(training), "pilot_train_rows": arguments.train_size,
        "development_rows": len(development), "confirmation_rows": len(confirmation),
        "development_files": sorted(development_files), "confirmation_files": sorted(confirmation_files),
        "transforms": {"global": global_parameters, "background": background_parameters},
        "inherited_train_sha256": protocol["train_sha256"],
        "inherited_validation_sha256": protocol["validation_sha256"],
        "development_sha256": full_digest(output / "development.npy"),
        "confirmation_sha256": full_digest(output / "confirmation.npy"),
        "old_test_and_external_test_evaluated": False,
        "confirmation_caveat": "This is a file-disjoint confirmation slice of the previous validation pool, not a historically unseen independent test set.",
        "labels_caveat": "Background normality is assumed, not human verified; inherited power-injection proxy is not verified waveform/channel SNR.",
    }
    fgaae.write_json(output / "manifest.json", manifest)
    print(f"STATION DATA: reference={len(reference_rows)}, pilot={arguments.train_size}/{len(training)}, development={len(development)}, confirmation={len(confirmation)}", flush=True)
    return manifest


def run_model(output, method, seed, training, development, development_labels, reference_training, reference_validation, parameters, arguments):
    output.mkdir(parents=True)
    fgaae.seed_everything(seed, arguments.num_threads)
    device = torch.device(arguments.device)
    config = vars(fgaae.build_parser().parse_args(["train"]))
    config.update({
        "sequence_length": training.shape[1], "seed": seed,
        "batch_size": arguments.batch_size, "stage1_epochs": arguments.stage1_epochs,
        "num_threads": arguments.num_threads, "device": str(device),
    })
    model = fgaae.FGAAE(config).to(device)
    guided = method == "background_featureguided"
    training = transform(training, parameters)
    development = transform(development, parameters)
    feature_mean, feature_std = None, None
    if guided:
        feature_training = fgaae.expert_features(reference_training, config).astype(np.float64)
        feature_validation = fgaae.expert_features(reference_validation, config).astype(np.float64)
        feature_mean, feature_std = feature_training.mean(axis=0), feature_training.std(axis=0)
        feature_std[feature_std < 1e-8] = 1
        fgaae.train_stage_one(
            model,
            fgaae.make_loader(transform(reference_training, parameters), arguments.batch_size, (feature_training - feature_mean) / feature_std, shuffle=True),
            fgaae.make_loader(transform(reference_validation, parameters), arguments.batch_size, (feature_validation - feature_mean) / feature_std),
            config, device, output,
        )
    model.freeze_teacher()
    optimized_parameters = list(model.encoder.parameters()) + list(model.decoder.parameters())
    if guided:
        optimized_parameters += list(model.student.parameters())
    optimizer = torch.optim.AdamW(optimized_parameters, lr=arguments.learning_rate, weight_decay=1e-3)
    loader = fgaae.make_loader(training, arguments.batch_size, shuffle=True)
    fgaae.seed_everything(seed + 10000, arguments.num_threads)
    best_auc, best_epoch, stale = -float("inf"), 0, 0
    best_state, selected_threshold = None, None
    for epoch in range(arguments.epochs):
        model.train()
        model.teacher.eval()
        reconstruction_sum = distillation_sum = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, latent = model(batch)
            reconstruction_loss = (reconstruction - batch).square().mean()
            if guided:
                with torch.no_grad():
                    teacher_output = model.teacher(latent)
                distillation_loss = (model.student(latent) - teacher_output).square().mean()
            else:
                distillation_loss = torch.zeros((), device=device)
            loss = reconstruction_loss + arguments.distillation_weight * distillation_loss
            fgaae.check_loss(loss)
            loss.backward()
            optimizer.step()
            reconstruction_sum += reconstruction_loss.item() * len(batch)
            distillation_sum += distillation_loss.item() * len(batch)
        scores, validation_distillation = model_scores(model, development, arguments.batch_size, device)
        threshold = fgaae.select_threshold(scores, development_labels, {"threshold_method": "validation-f1"})
        measurement = fgaae.metrics(development_labels, scores, threshold)
        fgaae.append_record(output / "history.csv", {
            "epoch": epoch + 1, "train_reconstruction_loss": reconstruction_sum / len(training),
            "train_distillation_loss": distillation_sum / len(training),
            "development_auc": measurement["roc_auc"], "development_ap": measurement["average_precision"],
            "development_f1": measurement["f1"], "development_threshold": threshold,
            "development_distillation_loss": validation_distillation if guided else 0.0,
        })
        if measurement["roc_auc"] > best_auc + arguments.minimum_delta:
            best_auc, best_epoch, stale = measurement["roc_auc"], epoch + 1, 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            selected_threshold = threshold
        else:
            stale += 1
        print(f"{method} seed={seed} epoch={epoch + 1}: recon={reconstruction_sum / len(training):.5f}, dev_auc={measurement['roc_auc']:.6f}", flush=True)
        if stale >= arguments.patience and epoch + 1 >= arguments.minimum_epochs:
            break
    if best_state is None:
        raise RuntimeError("No model checkpoint selected.")
    model.load_state_dict(best_state)
    scores, _ = model_scores(model, development, arguments.batch_size, device)
    measurement = fgaae.metrics(development_labels, scores, selected_threshold)
    pd.DataFrame({"score": scores, "label": development_labels}).to_csv(output / "development_scores.csv", index=False)
    checkpoint = {
        "artifact_type": "station_background_autoencoder_v2", "station": arguments.station,
        "method": method, "seed": seed, "config": config, "model_state": best_state,
        "normalization": parameters, "threshold": selected_threshold,
        "feature_mean": feature_mean.tolist() if feature_mean is not None else None,
        "feature_std": feature_std.tolist() if feature_std is not None else None,
        "best_epoch": best_epoch, "epochs_ran": epoch + 1,
        "selection_partition": "development", "old_test_evaluated": False,
    }
    torch.save(checkpoint, output / "model.pt")
    report = {"method": method, "seed": seed, "training_rows": len(training), "best_epoch": best_epoch, "epochs_ran": epoch + 1, "development": measurement}
    fgaae.write_json(output / "report.json", report)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def confirmation_scores(path, values, arguments):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    device = torch.device(arguments.device)
    model = fgaae.FGAAE(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    scores, _ = model_scores(model, transform(values, checkpoint["normalization"]), arguments.batch_size, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return checkpoint, scores


def plot_results(output, development, confirmation):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    combined = development.merge(confirmation, on="method", suffixes=("_development", "_confirmation"))
    figure, axis = plt.subplots(figsize=(13, 5))
    positions = np.arange(len(combined))
    axis.bar(positions - 0.2, combined.auc_mean_development, 0.4, label="Development (selection)")
    axis.bar(positions + 0.2, combined.auc_mean_confirmation, 0.4, label="Confirmation (previous validation pool)")
    axis.set_xticks(positions, combined.method, rotation=30, ha="right")
    axis.set_ylabel("ROC-AUC")
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "station_comparison.png", dpi=150)
    plt.close(figure)


def summarize(rows, partition):
    frame = pd.DataFrame(rows)
    return frame.groupby("method", as_index=False).agg(
        auc_mean=("roc_auc", "mean"), auc_std=("roc_auc", lambda series: float(series.std(ddof=0))),
        ap_mean=("average_precision", "mean"), f1_mean=("f1", "mean"),
        runs=("roc_auc", "size"),
    ).assign(partition=partition)


def run(arguments):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not arguments.seeds or len(set(arguments.seeds)) != len(arguments.seeds):
        raise ValueError("Seeds must be a nonempty list of distinct integers.")
    if min(arguments.train_size, arguments.batch_size, arguments.epochs, arguments.stage1_epochs, arguments.patience, arguments.minimum_epochs) < 1:
        raise ValueError("Sizes, epoch limits and patience must be positive.")
    if not 0 < arguments.confirmation_fraction < 1 or arguments.scale_floor_db <= 0:
        raise ValueError("Invalid confirmation fraction or background-scale floor.")
    if arguments.minimum_delta < 0 or arguments.distillation_weight < 0 or arguments.learning_rate <= 0:
        raise ValueError("Invalid loss, stopping or learning-rate settings.")
    if arguments.scaling_margin < 0 or not np.isfinite([
        arguments.scale_floor_db, arguments.confirmation_fraction, arguments.minimum_delta,
        arguments.distillation_weight, arguments.learning_rate, arguments.scaling_margin,
    ]).all():
        raise ValueError("Numeric settings must be finite and the scaling margin nonnegative.")
    output = Path(arguments.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    fgaae.seed_everything(arguments.seeds[0], arguments.num_threads)
    fgaae.write_json(output / "environment.json", {
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "device": arguments.device,
        "gpu": torch.cuda.get_device_name(torch.device(arguments.device)) if torch.device(arguments.device).type == "cuda" else None,
        "numpy": np.__version__, "pandas": pd.__version__,
        "station_experiment_sha256": full_digest(Path(__file__)),
    })
    fgaae.write_json(output / "registered_plan.json", {
        **vars(arguments), "methods": METHODS, "baselines": BASELINES,
        "normalization_fit": "earliest training source file, internal training portion only",
        "validation_policy": "earlier validation files select epochs and thresholds; later validation files are confirmation, not a new independent test",
        "scaling_gate": "Best learned development mean AUC must exceed best nonlearned development AUC by the preregistered margin.",
        "old_test_access": False, "no_new_anomalies_generated": True,
    })
    manifest = prepare_inputs(arguments, output / "inputs")
    inputs = output / "inputs"
    training = np.load(inputs / "train.npy", mmap_mode="r", allow_pickle=False)
    development = np.load(inputs / "development.npy", allow_pickle=False)
    development_metadata = pd.read_csv(inputs / "development_metadata.csv")
    labels = development_metadata.label.to_numpy()
    reference_training = np.load(inputs / "reference_train.npy", allow_pickle=False)
    reference_validation = np.load(inputs / "reference_validation.npy", allow_pickle=False)
    normalization = manifest["transforms"]
    baseline_values = baseline_scores(development, normalization["background"])
    baseline_thresholds, development_rows = {}, []
    for name, scores in baseline_values.items():
        threshold = fgaae.select_threshold(scores, labels, {"threshold_method": "validation-f1"})
        baseline_thresholds[name] = threshold
        development_rows.append({"method": name, "seed": -1, **fgaae.metrics(labels, scores, threshold)})
    fgaae.write_json(output / "baseline_thresholds.json", baseline_thresholds)
    pd.DataFrame({"source_row": development_metadata.source_row, "label": labels, **baseline_values}).to_csv(output / "baseline_development_scores.csv", index=False)
    pilot_reports = []
    for method in METHODS:
        parameters = normalization["global" if method == "global_zscore_ae" else "background"]
        for seed in arguments.seeds:
            report = run_model(
                output / method / f"seed_{seed}", method, seed, training[:arguments.train_size],
                development, labels, reference_training, reference_validation,
                parameters, arguments,
            )
            pilot_reports.append(report)
            development_rows.append({"method": method, "seed": seed, **report["development"]})
            pd.DataFrame(development_rows).to_csv(output / "development_runs.csv", index=False)
    development_summary = summarize(development_rows, "development")
    development_summary.to_csv(output / "development_comparison.csv", index=False)
    best_baseline = development_summary[development_summary.method.isin(BASELINES)].sort_values(["auc_mean", "method"], ascending=[False, True]).iloc[0]
    best_learned = development_summary[development_summary.method.isin(METHODS)].sort_values(["auc_mean", "auc_std", "method"], ascending=[False, True, True]).iloc[0]
    scaling_passed = bool(best_learned.auc_mean > best_baseline.auc_mean + arguments.scaling_margin)
    locked_selection = {
        "selected_baseline": str(best_baseline.method), "baseline_development_auc": float(best_baseline.auc_mean),
        "selected_learned_method": str(best_learned.method), "learned_development_auc_mean": float(best_learned.auc_mean),
        "learned_development_auc_std": float(best_learned.auc_std),
        "scaling_margin": arguments.scaling_margin, "full_scale_gate_passed": scaling_passed,
        "confirmation_accessed_for_selection": False, "test_accessed_for_selection": False,
    }
    fgaae.write_json(output / "locked_development_selection.json", locked_selection)
    full_reports = []
    if scaling_passed and not arguments.skip_scaling:
        selected_method = str(best_learned.method)
        parameters = normalization["global" if selected_method == "global_zscore_ae" else "background"]
        for seed in arguments.seeds:
            full_reports.append(run_model(
                output / "selected_full" / f"seed_{seed}", selected_method, seed, training,
                development, labels, reference_training, reference_validation,
                parameters, arguments,
            ))
    confirmation = np.load(inputs / "confirmation.npy", allow_pickle=False)
    confirmation_metadata = pd.read_csv(inputs / "confirmation_metadata.csv")
    confirmation_labels = confirmation_metadata.label.to_numpy()
    confirmation_rows = []
    for name, scores in baseline_scores(confirmation, normalization["background"]).items():
        measurement = fgaae.metrics(confirmation_labels, scores, baseline_thresholds[name])
        confirmation_rows.append({"method": name, "seed": -1, **measurement})
    for report in pilot_reports:
        method, seed = report["method"], report["seed"]
        checkpoint, scores = confirmation_scores(output / method / f"seed_{seed}/model.pt", confirmation, arguments)
        measurement = fgaae.metrics(confirmation_labels, scores, checkpoint["threshold"])
        confirmation_rows.append({"method": method, "seed": seed, **measurement})
        pd.DataFrame({"source_row": confirmation_metadata.source_row, "label": confirmation_labels, "score": scores}).to_csv(output / method / f"seed_{seed}/confirmation_scores.csv", index=False)
    full_confirmation = []
    for report in full_reports:
        checkpoint, scores = confirmation_scores(output / "selected_full" / f"seed_{report['seed']}/model.pt", confirmation, arguments)
        full_confirmation.append({"method": report["method"], "seed": report["seed"], **fgaae.metrics(confirmation_labels, scores, checkpoint["threshold"])})
    pd.DataFrame(confirmation_rows).to_csv(output / "confirmation_runs.csv", index=False)
    confirmation_summary = summarize(confirmation_rows, "confirmation")
    confirmation_summary.to_csv(output / "confirmation_comparison.csv", index=False)
    if full_confirmation:
        pd.DataFrame(full_confirmation).to_csv(output / "full_scale_confirmation.csv", index=False)
    if full_digest(inputs / "development.npy") != manifest["development_sha256"] or full_digest(inputs / "confirmation.npy") != manifest["confirmation_sha256"]:
        raise RuntimeError("A frozen validation slice changed during training.")
    plot_results(output, development_summary, confirmation_summary)
    fgaae.write_json(output / "final_report.json", {
        "station": arguments.station, "protocol": manifest["protocol_id"],
        "selection": locked_selection,
        "development": development_summary.to_dict(orient="records"),
        "confirmation": confirmation_summary.to_dict(orient="records"),
        "full_scale_runs": full_reports, "full_scale_confirmation": full_confirmation,
        "raw_test_and_external_test_evaluated": False,
        "confirmation_is_new_independent_test": False,
        "original_paper_performance_reproduced": False,
        "feature_guidance_caveat": "Feature-guided AE includes additional pretraining; it is not a compute-budget-matched proof and does not use GAN or AQT.",
    })
    print("DEVELOPMENT\n" + development_summary.to_string(index=False), flush=True)
    print("CONFIRMATION (not a new independent test)\n" + confirmation_summary.to_string(index=False), flush=True)
    print("LOCKED SELECTION\n" + json.dumps(locked_selection, ensure_ascii=False, indent=2), flush=True)
    print(f"STATION EXPERIMENT COMPLETE: {output}", flush=True)
    return output


def predict(arguments):
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("artifact_type") != "station_background_autoencoder_v2":
        raise ValueError("Expected a station-background model checkpoint.")
    torch.set_num_threads(arguments.num_threads)
    values, _ = fgaae.read_spectra(Path(arguments.data))
    if values.shape[1] != checkpoint["config"]["sequence_length"]:
        raise ValueError("Input sequence length does not match the checkpoint.")
    device = torch.device(arguments.device)
    model = fgaae.FGAAE(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    scores, _ = model_scores(model, transform(values, checkpoint["normalization"]), arguments.batch_size, device)
    output = Path(arguments.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing predictions.")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"source_row": np.arange(len(values)), "score": scores, "predicted_label": (scores > checkpoint["threshold"]).astype(int)}).to_csv(output, index=False)
    print(f"Predictions for calibrated station {checkpoint['station']}; other stations are not validated: {output}")


def smoke_test():
    from experiment_pipeline import power_interference

    with tempfile.TemporaryDirectory(prefix="station_background_") as temporary:
        directory = Path(temporary)
        dataset, benchmark = directory / "dataset", directory / "benchmark"
        dataset.mkdir()
        benchmark.mkdir()
        generator = np.random.default_rng(2025)
        values = generator.normal(200, 10, (256, 840)).astype(np.int16)
        partitions = pd.DataFrame({
            "row": np.arange(256), "station": "fixture",
            "source_file": [f"source_{index // 32}" for index in range(256)],
            "ticks": np.arange(256), "split": ["train"] * 128 + ["validation"] * 128,
        })
        np.save(dataset / "spectra.npy", values)
        partitions.to_csv(dataset / "partitions.csv", index=False)
        fgaae.write_json(dataset / "dataset.json", {"spectra_sha256": full_digest(dataset / "spectra.npy"), "partitions_sha256": full_digest(dataset / "partitions.csv")})
        protocol = {"protocol_id": "local-power-injection-v1"}
        for name, indices, ratio in [("train", np.arange(128), 0.05), ("validation", np.arange(128, 256), 0.25)]:
            changed, _, metadata = power_interference(values[indices], indices, ratio, 2025)
            np.save(benchmark / f"{name}.npy", changed)
            metadata.to_csv(benchmark / f"{name}_metadata.csv", index=False)
            protocol[f"{name}_sha256"] = full_digest(benchmark / f"{name}.npy")
        fgaae.write_json(benchmark / "protocol.json", protocol)
        arguments = build_parser().parse_args([
            "run", "--dataset", str(dataset), "--benchmark", str(benchmark),
            "--station", "fixture", "--output", str(directory / "run"),
            "--device", "cpu", "--train-size", "64", "--seeds", "2025",
            "--stage1-epochs", "1", "--epochs", "2", "--patience", "1",
            "--minimum-epochs", "2", "--batch-size", "32", "--skip-scaling",
        ])
        output = run(arguments)
        report = json.loads((output / "final_report.json").read_text(encoding="utf-8"))
        assert len(report["confirmation"]) == len(METHODS) + len(BASELINES)
        assert report["raw_test_and_external_test_evaluated"] is False
        parameters = fit_transform(values[:24], "background", 1.0)
        assert np.isfinite(transform(np.full((2, 840), 200), parameters)).all()
        model_path = output / "background_ae/seed_2025/model.pt"
        predict(argparse.Namespace(checkpoint=str(model_path), data=str(output / "inputs/development.npy"), output=str(directory / "predictions.csv"), device="cpu", batch_size=32, num_threads=2))
        saved = pd.read_csv(output / "background_ae/seed_2025/development_scores.csv").score
        reloaded = pd.read_csv(directory / "predictions.csv").score
        assert np.allclose(saved, reloaded, rtol=1e-5, atol=1e-5)
        print("STATION SMOKE TEST PASSED: training-only robust transforms, frozen samples, disjoint source files, AE/feature training, locked development selection, confirmation and checkpoint inference.")


def build_parser():
    parser = argparse.ArgumentParser(description="Single-station background normalization and no-learning baseline experiment; no old test evaluation.")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("run")
    training.add_argument("--dataset", default=str(ROOT / "datasets/fm_87_108_v1"))
    training.add_argument("--benchmark", default=str(ROOT / "runs/grouped_pipeline_v1/benchmark"))
    training.add_argument("--station", default="龙井三合海关30-3000")
    training.add_argument("--output", default=str(ROOT / "runs/station_background_v2"))
    training.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    training.add_argument("--num-threads", type=int, default=2)
    training.add_argument("--train-size", type=int, default=20000)
    training.add_argument("--seeds", type=int, nargs="+", default=[2025, 2026, 2027])
    training.add_argument("--epochs", type=int, default=40)
    training.add_argument("--stage1-epochs", type=int, default=100)
    training.add_argument("--patience", type=int, default=6)
    training.add_argument("--minimum-epochs", type=int, default=10)
    training.add_argument("--minimum-delta", type=float, default=1e-4)
    training.add_argument("--batch-size", type=int, default=256)
    training.add_argument("--learning-rate", type=float, default=2e-5)
    training.add_argument("--distillation-weight", type=float, default=1.0)
    training.add_argument("--scale-floor-db", type=float, default=1.0)
    training.add_argument("--confirmation-fraction", type=float, default=1 / 3)
    training.add_argument("--scaling-margin", type=float, default=0.005)
    training.add_argument("--skip-scaling", action="store_true")
    inference = commands.add_parser("predict")
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--data", required=True)
    inference.add_argument("--output", required=True)
    inference.add_argument("--device", default="cpu")
    inference.add_argument("--batch-size", type=int, default=256)
    inference.add_argument("--num-threads", type=int, default=2)
    commands.add_parser("smoke-test")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if arguments.command == "run":
        run(arguments)
    elif arguments.command == "predict":
        predict(arguments)
    else:
        smoke_test()
