from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch
from torch import autograd
from torch.utils.data import DataLoader, TensorDataset

import fgaae
import station_experiment as station
from audit_data import full_digest


ROOT = Path(__file__).resolve().parent
SCORERS = ["mean_mse", "top5_mse", "window8_mse", "top5_positive_mse", "background_weighted_top5_mse"]
ARMS = [
    {"name": "clean_ae", "suite": "contamination", "injected": False, "features": False, "adversarial": False, "aqt": False},
    {"name": "injected_ae", "suite": "contamination", "injected": True, "features": False, "adversarial": False, "aqt": False},
    {"name": "injected_ae_aqt", "suite": "contamination", "injected": True, "features": False, "adversarial": False, "aqt": True},
    {"name": "full_training_modules", "suite": "modules", "injected": True, "features": True, "adversarial": True, "aqt": True},
    {"name": "without_feature_guidance", "suite": "modules", "injected": True, "features": False, "adversarial": True, "aqt": True},
    {"name": "without_adversarial", "suite": "modules", "injected": True, "features": True, "adversarial": False, "aqt": True},
    {"name": "without_aqt", "suite": "modules", "injected": True, "features": True, "adversarial": True, "aqt": False},
]


def state_copy(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def top_mean(values, count):
    return torch.topk(values, min(count, values.shape[1]), dim=1).values.mean(dim=1)


@torch.no_grad()
def score_model(model, normalized, normalization, background, batch_size, device, feature_targets=None, frozen_teacher=None):
    model.eval()
    outputs = {name: [] for name in SCORERS}
    if feature_targets is not None:
        outputs.update({"teacher_expert_error": [], "student_teacher_error": []})
        if frozen_teacher is not None:
            frozen_teacher.eval()
            outputs.update({"frozen_teacher_expert_error": [], "teacher_prediction_drift": []})
    native_scale = torch.as_tensor(normalization["scale_db"], dtype=torch.float32, device=device)
    background_scale = torch.as_tensor(background["scale_db"], dtype=torch.float32, device=device)
    tail_count = max(1, round(normalized.shape[1] * 0.05))
    loader = fgaae.make_loader(normalized, batch_size)
    start = 0
    for (batch,) in loader:
        batch = batch.to(device)
        reconstructed, latent = model(batch)
        residual = (batch - reconstructed).squeeze(1)
        squared = residual.square()
        cumulative = torch.nn.functional.pad(squared.cumsum(dim=1), (1, 0))
        window_size = min(8, squared.shape[1])
        window = (cumulative[:, window_size:] - cumulative[:, :-window_size]) / window_size
        weighted = (residual * native_scale / background_scale).square()
        batch_scores = {
            "mean_mse": squared.mean(dim=1), "top5_mse": top_mean(squared, tail_count),
            "window8_mse": window.max(dim=1).values,
            "top5_positive_mse": top_mean(residual.clamp_min(0).square(), tail_count),
            "background_weighted_top5_mse": top_mean(weighted, tail_count),
        }
        if feature_targets is not None:
            targets = torch.as_tensor(feature_targets[start:start + len(batch)], dtype=torch.float32, device=device)
            teacher_prediction = model.teacher(latent)
            batch_scores["teacher_expert_error"] = (teacher_prediction - targets).square().mean(dim=1)
            batch_scores["student_teacher_error"] = (model.student(latent) - teacher_prediction).square().mean(dim=1)
            if frozen_teacher is not None:
                frozen_prediction = frozen_teacher.teacher(frozen_teacher.encoder(batch))
                batch_scores["frozen_teacher_expert_error"] = (frozen_prediction - targets).square().mean(dim=1)
                batch_scores["teacher_prediction_drift"] = (teacher_prediction - frozen_prediction).square().mean(dim=1)
        for name, values in batch_scores.items():
            outputs[name].extend(values.cpu().tolist())
        start += len(batch)
    result = {name: np.asarray(values, dtype=np.float64) for name, values in outputs.items()}
    if any(not np.isfinite(values).all() for values in result.values()):
        raise FloatingPointError("Non-finite anomaly score in the ablation.")
    return result


def gradient_penalty(model, real, generated, generator):
    weight = torch.rand(real.shape[0], 1, 1, device=real.device, generator=generator)
    interpolated = (weight * real + (1 - weight) * generated.detach()).requires_grad_(True)
    predicted = model.critic(interpolated)
    gradients = autograd.grad(
        predicted, interpolated, grad_outputs=torch.ones_like(predicted), create_graph=True,
    )[0]
    return ((gradients.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def load_data(arguments, output):
    source = Path(arguments.station_run).resolve()
    inputs = source / "inputs"
    manifest = json.loads((inputs / "manifest.json").read_text(encoding="utf-8"))
    if full_digest(inputs / "development.npy") != manifest["development_sha256"]:
        raise ValueError("Frozen development spectra changed.")
    source_training = np.load(inputs / "train.npy", mmap_mode="r", allow_pickle=False)
    if len(source_training) < arguments.train_size:
        raise ValueError("Requested more training rows than the frozen pool contains.")
    metadata = pd.read_csv(inputs / "train_metadata.csv").iloc[:arguments.train_size].copy()
    development_metadata = pd.read_csv(inputs / "development_metadata.csv")
    partitions = pd.read_csv(Path(arguments.dataset) / "partitions.csv").set_index("row")
    source_rows = metadata.source_row.to_numpy(dtype=np.int64)
    development_rows = development_metadata.source_row.to_numpy(dtype=np.int64)
    if not partitions.loc[source_rows, "split"].eq("train").all():
        raise ValueError("A training row is not from the original training split.")
    if not partitions.loc[development_rows, "split"].eq("validation").all():
        raise ValueError("Development includes original test rows.")
    if set(partitions.loc[source_rows, "source_file"]).intersection(partitions.loc[development_rows, "source_file"]):
        raise ValueError("A source file crosses training/development.")
    reference_rows = set(manifest["reference_train_rows"] + manifest["reference_validation_rows"])
    if reference_rows.intersection(source_rows.tolist()):
        raise ValueError("Training includes background-reference rows.")
    raw = np.load(Path(arguments.dataset) / "spectra.npy", mmap_mode="r", allow_pickle=False)
    clean = raw[source_rows].astype(np.float32)
    injected = source_training[:arguments.train_size].astype(np.float32)
    labels = metadata.label.to_numpy(dtype=np.int64)
    if not np.array_equal(clean[labels == 0], injected[labels == 0]):
        raise ValueError("An inherited unmodified background does not match the original array.")
    changed = np.any(clean != injected, axis=1)
    if not np.array_equal(changed, labels.astype(bool)):
        raise ValueError("Inherited injection metadata does not match the actual changed rows.")
    reference_training = np.load(inputs / "reference_train.npy", allow_pickle=False)
    reference_validation = np.load(inputs / "reference_validation.npy", allow_pickle=False)
    for mode in ["global", "background"]:
        expected = station.fit_transform(reference_training, mode, manifest["transforms"][mode]["floor_db"])
        if expected != manifest["transforms"][mode]:
            raise ValueError("Stored normalization does not match the reference training data.")
    development = np.load(inputs / "development.npy", allow_pickle=False)
    output.mkdir()
    metadata.to_csv(output / "training_membership.csv", index=False)
    development_metadata.to_csv(output / "development_membership.csv", index=False)
    data_manifest = {
        "station": manifest["station"], "source_station_run": str(source),
        "training_rows": len(clean), "development_rows": len(development),
        "injected_fraction": float(labels.mean()),
        "source_row_order_sha256": hashlib.sha256(source_rows.tobytes()).hexdigest(),
        "frozen_training_file_sha256": full_digest(inputs / "train.npy"),
        "frozen_development_sha256": manifest["development_sha256"],
        "normalization": manifest["transforms"],
        "clean_definition": "Same source frames restored to raw observed values, with zero artificial injection; real normality is not verified.",
        "partitions_used": ["training reference", "stage2 training", "development"],
        "confirmation_or_test_used": False,
    }
    fgaae.write_json(output / "manifest.json", data_manifest)
    return {
        "source": source, "manifest": data_manifest, "clean": clean, "injected": injected,
        "injected_labels": labels, "source_rows": source_rows,
        "development": development, "development_rows": development_rows,
        "labels": development_metadata.label.to_numpy(dtype=np.int64),
        "reference_training": reference_training, "reference_validation": reference_validation,
        "normalization": manifest["transforms"]["global"],
        "background": manifest["transforms"]["background"],
    }


def measurements(labels, scores):
    threshold = fgaae.select_threshold(scores, labels, {"threshold_method": "validation-f1"})
    return fgaae.metrics(labels, scores, threshold)


def existing_model_diagnostics(data, arguments, output):
    output.mkdir()
    results = []
    device = torch.device(arguments.device)
    feature_raw = None
    for method in station.METHODS:
        for seed in arguments.seeds:
            directory = data["source"] / method / f"seed_{seed}"
            checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
            model = fgaae.FGAAE(checkpoint["config"]).to(device)
            model.load_state_dict(checkpoint["model_state"])
            targets, frozen = None, None
            if checkpoint.get("feature_mean") is not None:
                if feature_raw is None:
                    feature_raw = fgaae.expert_features(data["development"], checkpoint["config"])
                targets = (feature_raw - np.asarray(checkpoint["feature_mean"])) / np.asarray(checkpoint["feature_std"])
                stage1 = torch.load(directory / "stage1_best.pt", map_location="cpu", weights_only=True)
                frozen = fgaae.FGAAE(checkpoint["config"]).to(device)
                frozen.encoder.load_state_dict(stage1["encoder"])
                frozen.teacher.load_state_dict(stage1["teacher"])
                frozen.requires_grad_(False)
            scores = score_model(
                model, station.transform(data["development"], checkpoint["normalization"]),
                checkpoint["normalization"], data["background"], arguments.batch_size, device,
                feature_targets=targets, frozen_teacher=frozen,
            )
            pd.DataFrame({"source_row": data["development_rows"], "label": data["labels"], **scores}).to_csv(output / f"{method}_seed_{seed}_scores.csv", index=False)
            for scorer, values in scores.items():
                results.append({
                    "suite": "scoring" if scorer in SCORERS else "feature_diagnostic",
                    "model": method, "seed": seed, "scorer": scorer,
                    "checkpoint_selection": "old development mean-MSE optimum; identical checkpoint for every scorer",
                    **measurements(data["labels"], values),
                })
            del model, frozen
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"SCORE DIAGNOSTIC: {method} seed={seed}", flush=True)
    table = pd.DataFrame(results)
    table.to_csv(output / "scoring_and_feature_runs.csv", index=False)
    return table


def prepare_seed(data, seed, arguments, output, need_pretraining):
    fgaae.seed_everything(seed, arguments.num_threads)
    config = vars(fgaae.build_parser().parse_args(["train"]))
    config.update({"sequence_length": 840, "batch_size": arguments.batch_size, "stage1_epochs": arguments.stage1_epochs, "seed": seed})
    model = fgaae.FGAAE(config).to(torch.device(arguments.device))
    initial = state_copy(model)
    pretrained = None
    if need_pretraining:
        output.mkdir(parents=True)
        feature_training = fgaae.expert_features(data["reference_training"], config).astype(np.float64)
        feature_validation = fgaae.expert_features(data["reference_validation"], config).astype(np.float64)
        mean, scale = feature_training.mean(axis=0), feature_training.std(axis=0)
        scale[scale < 1e-8] = 1
        fgaae.train_stage_one(
            model,
            fgaae.make_loader(station.transform(data["reference_training"], data["normalization"]), arguments.batch_size, (feature_training - mean) / scale, shuffle=True),
            fgaae.make_loader(station.transform(data["reference_validation"], data["normalization"]), arguments.batch_size, (feature_validation - mean) / scale),
            config, torch.device(arguments.device), output,
        )
        pretrained = state_copy(model)
        torch.save({"initial_state": initial, "pretrained_state": pretrained, "config": config}, output / "initializations.pt")
    del model
    return config, initial, pretrained


def parameter_delta(model, initial, prefix):
    total = 0.0
    for name, parameter in model.named_parameters():
        if name.startswith(prefix + "."):
            total += float((parameter.detach().cpu().double() - initial[name].double()).square().sum())
    return math.sqrt(total)


def train_arm(arm, seed, config, initial, pretrained, data, arguments, output):
    output.mkdir(parents=True)
    device = torch.device(arguments.device)
    fgaae.seed_everything(seed + 10000, arguments.num_threads)
    model = fgaae.FGAAE(config).to(device)
    starting = pretrained if arm["features"] else initial
    if starting is None:
        raise ValueError("Feature guidance requires shared feature pretraining.")
    model.load_state_dict(starting)
    model.freeze_teacher()
    model.student.requires_grad_(arm["features"])
    model.critic.requires_grad_(arm["adversarial"])
    parameters = list(model.encoder.parameters()) + list(model.decoder.parameters())
    if arm["features"]:
        parameters += list(model.student.parameters())
    generator_optimizer = torch.optim.AdamW(parameters, lr=arguments.learning_rate, weight_decay=1e-3)
    critic_optimizer = torch.optim.AdamW(model.critic.parameters(), lr=arguments.learning_rate, weight_decay=1e-3, betas=(0.5, 0.9)) if arm["adversarial"] else None
    gp_generator = torch.Generator(device=device).manual_seed(seed + 20000)
    raw = data["injected"] if arm["injected"] else data["clean"]
    training = station.transform(raw, data["normalization"])
    development = station.transform(data["development"], data["normalization"])
    tensor = torch.from_numpy(training).unsqueeze(1)
    kept = np.arange(len(training))
    best_value, best_epoch, stale = -float("inf"), 0, 0
    best_state, best_threshold = None, None
    retention_records = []
    for epoch in range(arguments.epochs):
        if arm["aqt"] and epoch >= arguments.warmup_epochs and (epoch - arguments.warmup_epochs) % arguments.filter_every == 0:
            index = (epoch - arguments.warmup_epochs) // arguments.filter_every
            filter_count = max(1, math.ceil((arguments.epochs - arguments.warmup_epochs) / arguments.filter_every))
            progress = index / max(filter_count - 1, 1)
            rate = arguments.retention_end + (0.95 - arguments.retention_end) * math.exp(-5 * progress)
            ranking = score_model(model, training, data["normalization"], data["background"], arguments.batch_size, device)[arguments.primary_score]
            cutoff = float(np.quantile(ranking, rate))
            kept = np.flatnonzero(ranking <= cutoff)
            if len(kept) < 2:
                raise ValueError("AQT left fewer than two samples.")
            pd.DataFrame({"source_row": data["source_rows"], "score": ranking, "retained": np.isin(np.arange(len(training)), kept)}).to_csv(output / f"aqt_epoch_{epoch + 1:03d}.csv", index=False)
        generator = torch.Generator().manual_seed(seed + 100000 + epoch)
        loader = DataLoader(TensorDataset(tensor[kept]), batch_size=arguments.batch_size, shuffle=True, generator=generator, num_workers=0)
        model.train()
        model.teacher.eval()
        totals = {name: 0.0 for name in ["reconstruction", "distillation", "generator_adversarial", "critic", "gradient_penalty"]}
        seen = 0
        for (real,) in loader:
            real = real.to(device)
            critic_loss_value = penalty_value = 0.0
            if arm["adversarial"]:
                model.encoder.eval()
                model.decoder.eval()
                model.critic.requires_grad_(True)
                model.critic.train()
                with torch.no_grad():
                    generated, _ = model(real)
                for _ in range(arguments.critic_steps):
                    critic_optimizer.zero_grad(set_to_none=True)
                    penalty = gradient_penalty(model, real, generated, gp_generator)
                    critic_loss = model.critic(generated).mean() - model.critic(real).mean() + arguments.gp_weight * penalty
                    fgaae.check_loss(critic_loss)
                    critic_loss.backward()
                    critic_optimizer.step()
                    critic_loss_value += critic_loss.item() / arguments.critic_steps
                    penalty_value += penalty.item() / arguments.critic_steps
                model.critic.requires_grad_(False)
                model.critic.eval()
            model.encoder.train()
            model.decoder.train()
            generator_optimizer.zero_grad(set_to_none=True)
            generated, latent = model(real)
            reconstruction_loss = (generated - real).square().mean()
            if arm["features"]:
                with torch.no_grad():
                    target = model.teacher(latent)
                distillation_loss = (model.student(latent) - target).square().mean()
            else:
                distillation_loss = torch.zeros((), device=device)
            adversarial_loss = -model.critic(generated).mean() if arm["adversarial"] else torch.zeros((), device=device)
            total_loss = reconstruction_loss + arguments.distillation_weight * distillation_loss + arguments.adversarial_weight * adversarial_loss
            fgaae.check_loss(total_loss)
            total_loss.backward()
            generator_optimizer.step()
            for name, value in [
                ("reconstruction", reconstruction_loss.item()), ("distillation", distillation_loss.item()),
                ("generator_adversarial", adversarial_loss.item()), ("critic", critic_loss_value), ("gradient_penalty", penalty_value),
            ]:
                totals[name] += value * len(real)
            seen += len(real)
        scored = score_model(model, development, data["normalization"], data["background"], arguments.batch_size, device)
        measured = measurements(data["labels"], scored[arguments.primary_score])
        fgaae.append_record(output / "history.csv", {
            "epoch": epoch + 1, "retained": len(kept),
            **{f"train_{name}": value / seen for name, value in totals.items()},
            "development_auc": measured["roc_auc"], "development_ap": measured["average_precision"],
            "development_f1": measured["f1"], "threshold": measured["threshold"],
        })
        if measured["roc_auc"] > best_value + arguments.minimum_delta:
            best_value, best_epoch, stale = measured["roc_auc"], epoch + 1, 0
            best_state, best_threshold = state_copy(model), measured["threshold"]
        else:
            stale += 1
        retention_records.append({"epoch": epoch + 1, "retained": kept.copy()})
        print(f"{arm['name']} seed={seed} epoch={epoch + 1}: kept={len(kept)}, dev_auc={measured['roc_auc']:.6f}", flush=True)
        if arguments.patience > 0 and stale >= arguments.patience and epoch + 1 >= arguments.minimum_epochs:
            break
    deltas = {prefix: parameter_delta(model, starting, prefix) for prefix in ["encoder", "decoder", "teacher", "student", "critic"]}
    if deltas["teacher"] != 0 or (not arm["features"] and deltas["student"] != 0) or (not arm["adversarial"] and deltas["critic"] != 0):
        raise RuntimeError("An ablated/frozen module was unexpectedly updated.")
    fgaae.write_json(output / "module_update_checks.json", {"parameter_l2_change_last_epoch": deltas, "frozen_and_removed_modules_unchanged": True})
    if best_state is None:
        raise RuntimeError("No best checkpoint was selected.")
    model.load_state_dict(best_state)
    scored = score_model(model, development, data["normalization"], data["background"], arguments.batch_size, device)
    results = {name: measurements(data["labels"], values) for name, values in scored.items()}
    pd.DataFrame({"source_row": data["development_rows"], "label": data["labels"], **scored}).to_csv(output / "development_scores.csv", index=False)
    torch.save({
        "artifact_type": "controlled_ablation_v3", "model_state": best_state, "config": config,
        "arm": arm, "seed": seed, "normalization": data["normalization"], "background": data["background"],
        "primary_score": arguments.primary_score, "threshold": best_threshold,
        "best_epoch": best_epoch, "epochs_ran": epoch + 1,
        "station": data["manifest"]["station"], "selection_partition": "development only",
    }, output / "model.pt")
    fgaae.write_json(output / "report.json", {
        "arm": arm, "seed": seed, "primary_score": arguments.primary_score,
        "best_epoch": best_epoch, "epochs_ran": epoch + 1,
        "feature_pretraining_epochs": arguments.stage1_epochs if arm["features"] else 0,
        "scores_at_primary_selected_checkpoint": results,
        "confirmation_or_test_evaluated": False,
    })
    synthetic_labels = data["injected_labels"] if arm["injected"] else np.zeros(len(training), dtype=int)
    diagnostic_rows = [
        {"epoch": record["epoch"], "retained": len(record["retained"]),
         "injected_fraction": float(synthetic_labels[record["retained"]].mean()),
         "diagnostic_only_labels_not_used_by_aqt": True}
        for record in retention_records
    ]
    pd.DataFrame(diagnostic_rows).to_csv(output / "purity_diagnostic.csv", index=False)
    row = {
        **arm, "seed": seed, "best_epoch": best_epoch, "epochs_ran": epoch + 1,
        "primary_score": arguments.primary_score, "training_rows": len(training),
        "training_injected_fraction": float(synthetic_labels.mean()),
        **results[arguments.primary_score],
    }
    del model, generator_optimizer, critic_optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def summarize(frame, grouping):
    return frame.groupby(grouping, as_index=False).agg(
        auc_mean=("roc_auc", "mean"), auc_std=("roc_auc", lambda values: float(values.std(ddof=1)) if len(values) > 1 else 0.0),
        ap_mean=("average_precision", "mean"), ap_std=("average_precision", lambda values: float(values.std(ddof=1)) if len(values) > 1 else 0.0),
        f1_mean=("f1", "mean"), f1_std=("f1", lambda values: float(values.std(ddof=1)) if len(values) > 1 else 0.0),
        runs=("roc_auc", "size"),
    )


def export_tables(output, trained, diagnostics):
    sections = ["# Controlled ablations: synthetic development results", "", "Exploratory development-set measurements, not independent test results. Mean ± sample SD across seeds; not a confidence interval.", ""]
    if len(trained):
        summary = summarize(trained, ["suite", "name"])
        summary.to_csv(output / "training_ablation_summary.csv", index=False)
        module_rows = trained[trained.suite == "modules"]
        if len(module_rows):
            reference = module_rows[module_rows.name == "full_training_modules"][["seed", "roc_auc"]].rename(columns={"roc_auc": "full_auc"})
            paired = module_rows.merge(reference, on="seed", validate="many_to_one")
            paired["delta_auc_vs_full"] = paired.roc_auc - paired.full_auc
            paired.to_csv(output / "module_paired_deltas.csv", index=False)
        sections.extend(["## Training modules and contamination", "", "| Suite | Variant | ROC-AUC | AP | F1 | Seeds |", "|---|---|---:|---:|---:|---:|"])
        for row in summary.itertuples():
            sections.append(f"| {row.suite} | {row.name} | {row.auc_mean:.4f} ± {row.auc_std:.4f} | {row.ap_mean:.4f} ± {row.ap_std:.4f} | {row.f1_mean:.4f} ± {row.f1_std:.4f} | {row.runs} |")
        latex = [r"\begin{tabular}{llccc}", r"\hline", r"Suite & Variant & AUC & AP & F1 \\", r"\hline"]
        for row in summary.itertuples():
            suite = row.suite.replace("_", r"\_")
            name = row.name.replace("_", r"\_")
            latex.append(f"{suite} & {name} & ${row.auc_mean:.4f} \\pm {row.auc_std:.4f}$ & ${row.ap_mean:.4f} \\pm {row.ap_std:.4f}$ & ${row.f1_mean:.4f} \\pm {row.f1_std:.4f}$ " + r"\\")
        latex.extend([r"\hline", r"\end{tabular}"])
        (output / "development_ablation_table.tex").write_text("\n".join(latex), encoding="utf-8")
    if len(diagnostics):
        summary = summarize(diagnostics, ["suite", "model", "scorer"])
        summary.to_csv(output / "scoring_feature_summary.csv", index=False)
        sections.extend(["", "## Identical-checkpoint scoring and feature diagnostics", "", "| Model | Score | ROC-AUC | AP |", "|---|---|---:|---:|"])
        for row in summary.itertuples():
            sections.append(f"| {row.model} | {row.scorer} | {row.auc_mean:.4f} ± {row.auc_std:.4f} | {row.ap_mean:.4f} ± {row.ap_std:.4f} |")
    (output / "ABLATION_TABLES.md").write_text("\n".join(sections) + "\n", encoding="utf-8")


def plot_tables(output, trained, diagnostics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(trained):
        summary = summarize(trained, ["suite", "name"])
        suites = sorted(summary.suite.unique())
        figure, axes = plt.subplots(1, len(suites), figsize=(8 * len(suites), 5), squeeze=False)
        for axis, suite in zip(axes[0], suites):
            subset = summary[summary.suite == suite]
            axis.bar(subset.name, subset.auc_mean, yerr=subset.auc_std, capsize=4)
            axis.tick_params(axis="x", rotation=25)
            axis.set_ylabel("Development ROC-AUC (mean +/- sample SD)")
            axis.set_ylim(0, 1)
            axis.set_title(f"{suite}: synthetic development data")
        figure.tight_layout()
        figure.savefig(output / "training_ablations.png", dpi=160)
        plt.close(figure)
    if len(diagnostics):
        table = diagnostics[diagnostics.suite == "scoring"].pivot_table(index="model", columns="scorer", values="roc_auc", aggfunc="mean")
        figure, axis = plt.subplots(figsize=(11, 4))
        raster = axis.imshow(table.to_numpy(), vmin=0, vmax=1, aspect="auto")
        axis.set_xticks(np.arange(len(table.columns)), table.columns, rotation=20, ha="right")
        axis.set_yticks(np.arange(len(table.index)), table.index)
        for row in range(len(table)):
            for column in range(len(table.columns)):
                axis.text(column, row, f"{table.iloc[row, column]:.3f}", ha="center", va="center", color="black")
        axis.set_title("Identical-checkpoint scoring: synthetic development AUC")
        figure.colorbar(raster, ax=axis)
        figure.tight_layout()
        figure.savefig(output / "same_checkpoint_scoring.png", dpi=160)
        plt.close(figure)


def run(arguments):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not arguments.seeds or len(set(arguments.seeds)) != len(arguments.seeds):
        raise ValueError("Use distinct model seeds.")
    if min(arguments.train_size, arguments.batch_size, arguments.epochs, arguments.stage1_epochs, arguments.critic_steps, arguments.filter_every) < 1:
        raise ValueError("Sizes, epochs, critic steps and filtering interval must be positive.")
    if arguments.patience < 0 or arguments.warmup_epochs < 0 or not 0 < arguments.retention_end <= 0.95:
        raise ValueError("Invalid early stopping, warmup or retention settings.")
    if any(value < 0 or not math.isfinite(value) for value in [arguments.minimum_delta, arguments.adversarial_weight, arguments.distillation_weight]):
        raise ValueError("Loss weights and minimum improvement must be finite and nonnegative.")
    if arguments.learning_rate <= 0 or arguments.gp_weight <= 0 or not math.isfinite(arguments.learning_rate + arguments.gp_weight):
        raise ValueError("Learning rate and GP weight must be positive.")
    output = Path(arguments.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    fgaae.seed_everything(arguments.seeds[0], arguments.num_threads)
    arms = [arm for arm in ARMS if arm["suite"] in arguments.suites]
    plan = {
        **vars(arguments), "arms": arms, "fixed_scorers": SCORERS,
        "primary_checkpoint_criterion": f"development ROC-AUC using {arguments.primary_score}",
        "shared_controls": "Same source rows, normalization, seed, architecture, optimizer settings, epoch cap and scoring definition. Separate seeded data-order and GP generators.",
        "feature_removal": "without_feature_guidance removes both stage-one feature pretraining and stage-two distillation; other feature-enabled arms share one identical pretraining per seed.",
        "adversarial_removal": "without_adversarial disables both critic optimization and generator adversarial loss; critic scores never enter the common module-ablation decision score.",
        "feature_budget_caveat": "Removing the feature module also removes its 100-epoch pretraining cost. This is whole-module removal, not a compute-matched distillation-only ablation.",
        "evaluation_partition": "previously used development only; no confirmation, test or external-test scoring",
        "paper_scope": "Controlled single-station implementation ablations; not strict original-paper architecture/data/protocol replication.",
        "standard_deviation": "Sample SD across seeds, ddof=1, not a confidence interval.",
        "code_sha256": full_digest(Path(__file__)),
    }
    fgaae.write_json(output / "registered_plan.json", plan)
    data = load_data(arguments, output / "inputs")
    diagnostics = pd.DataFrame()
    if "scoring" in arguments.suites:
        diagnostics = existing_model_diagnostics(data, arguments, output / "same_checkpoint_diagnostics")
        export_tables(output, pd.DataFrame(), diagnostics)
    rows = []
    for seed in arguments.seeds:
        if not arms:
            break
        config, initial, pretrained = prepare_seed(
            data, seed, arguments, output / "shared_pretraining" / f"seed_{seed}",
            need_pretraining=any(arm["features"] for arm in arms),
        )
        for arm in arms:
            rows.append(train_arm(arm, seed, config, initial, pretrained, data, arguments, output / arm["name"] / f"seed_{seed}"))
            trained = pd.DataFrame(rows)
            trained.to_csv(output / "training_ablation_runs.csv", index=False)
            export_tables(output, trained, diagnostics)
    trained = pd.DataFrame(rows)
    export_tables(output, trained, diagnostics)
    plot_tables(output, trained, diagnostics)
    fgaae.write_json(output / "completion.json", {
        "completed_training_runs": len(trained), "existing_checkpoint_diagnostic_rows": len(diagnostics),
        "primary_score_fixed_before_execution": arguments.primary_score,
        "confirmation_or_test_evaluated": False, "independent_test_claim": False,
        "paper_performance_replication_claim": False,
    })
    if len(trained):
        print(summarize(trained, ["suite", "name"]).to_string(index=False), flush=True)
    print(f"ABLATION COMPLETE: {output}", flush=True)
    return output


def predict(arguments):
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("artifact_type") != "controlled_ablation_v3":
        raise ValueError("Expected a controlled-ablation checkpoint.")
    torch.set_num_threads(arguments.num_threads)
    raw, labels = fgaae.read_spectra(Path(arguments.data))
    if raw.shape[1] != checkpoint["config"]["sequence_length"]:
        raise ValueError("Input frequency grid length does not match the checkpoint.")
    model = fgaae.FGAAE(checkpoint["config"]).to(torch.device(arguments.device))
    model.load_state_dict(checkpoint["model_state"])
    scores = score_model(model, station.transform(raw, checkpoint["normalization"]), checkpoint["normalization"], checkpoint["background"], arguments.batch_size, torch.device(arguments.device))[checkpoint["primary_score"]]
    output = Path(arguments.output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite predictions.")
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({"row": np.arange(len(raw)), "score": scores, "predicted_label": (scores > checkpoint["threshold"]).astype(int)})
    if labels is not None:
        table["label"] = labels
        print(json.dumps(fgaae.metrics(labels, scores, checkpoint["threshold"]), indent=2))
    table.to_csv(output, index=False)
    print(f"Inference for calibrated station {checkpoint['station']}: {output}")


def smoke_test():
    from experiment_pipeline import power_interference

    with tempfile.TemporaryDirectory(prefix="ablation_v3_") as temporary:
        directory = Path(temporary)
        source, dataset = directory / "station", directory / "dataset"
        inputs = source / "inputs"
        inputs.mkdir(parents=True)
        dataset.mkdir()
        generator = np.random.default_rng(2025)
        raw = generator.normal(200, 10, (160, 840)).astype(np.int16)
        np.save(dataset / "spectra.npy", raw)
        partitions = pd.DataFrame({"row": np.arange(160), "source_file": [f"source_{index // 32}" for index in range(160)], "split": ["train"] * 128 + ["validation"] * 32})
        partitions.to_csv(dataset / "partitions.csv", index=False)
        training_rows = np.arange(32, 96)
        reference_training, reference_validation = raw[:24], raw[24:32]
        injected, _, metadata = power_interference(raw[training_rows], training_rows, 0.1, 2025)
        development, _, development_metadata = power_interference(raw[128:], np.arange(128, 160), 0.25, 2026)
        np.save(inputs / "train.npy", injected)
        metadata.to_csv(inputs / "train_metadata.csv", index=False)
        np.save(inputs / "development.npy", development)
        development_metadata.to_csv(inputs / "development_metadata.csv", index=False)
        np.save(inputs / "reference_train.npy", reference_training)
        np.save(inputs / "reference_validation.npy", reference_validation)
        fgaae.write_json(inputs / "manifest.json", {
            "station": "fixture", "reference_train_rows": list(range(24)),
            "reference_validation_rows": list(range(24, 32)),
            "development_sha256": full_digest(inputs / "development.npy"),
            "transforms": {mode: station.fit_transform(reference_training, mode, 1.0) for mode in ["global", "background"]},
        })
        args = build_parser().parse_args([
            "run", "--station-run", str(source), "--dataset", str(dataset), "--output", str(directory / "run"),
            "--suites", "contamination", "modules", "--seeds", "2025", "--device", "cpu",
            "--train-size", "64", "--epochs", "2", "--stage1-epochs", "1", "--warmup-epochs", "1", "--filter-every", "1", "--batch-size", "32",
        ])
        output = run(args)
        table = pd.read_csv(output / "training_ablation_runs.csv")
        assert len(table) == len(ARMS)
        assert table.loc[table.name == "clean_ae", "training_injected_fraction"].iloc[0] == 0
        assert (output / "full_training_modules/seed_2025/aqt_epoch_002.csv").exists()
        assert not (output / "without_aqt/seed_2025/aqt_epoch_002.csv").exists()
        residual = torch.tensor([[1.0, 2.0, 10.0]])
        assert float(top_mean(residual, 1)[0]) == 10.0
        checkpoint_path = output / "full_training_modules/seed_2025/model.pt"
        predict(argparse.Namespace(checkpoint=str(checkpoint_path), data=str(inputs / "development.npy"), output=str(directory / "predictions.csv"), device="cpu", batch_size=32, num_threads=2))
        expected = pd.read_csv(output / "full_training_modules/seed_2025/development_scores.csv")[args.primary_score]
        predicted = pd.read_csv(directory / "predictions.csv").score
        assert np.allclose(expected, predicted, rtol=1e-5, atol=1e-5)
        assert not list(output.rglob("confirmation_scores.csv"))
        print("ABLATION SMOKE TEST PASSED: paired rows, zero injection control, independent sampling, true module removal, AQT on/off, tables and checkpoint inference.")


def build_parser():
    parser = argparse.ArgumentParser(description="Development-only scoring, contamination and training-module ablations.")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("run")
    training.add_argument("--station-run", default=str(ROOT / "runs/station_background_v2"))
    training.add_argument("--dataset", default=str(ROOT / "datasets/fm_87_108_v1"))
    training.add_argument("--output", default=str(ROOT / "runs/ablation_v3"))
    training.add_argument("--suites", nargs="+", choices=["scoring", "contamination", "modules"], default=["scoring", "contamination", "modules"])
    training.add_argument("--seeds", nargs="+", type=int, default=[2025, 2026, 2027])
    training.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    training.add_argument("--num-threads", type=int, default=2)
    training.add_argument("--train-size", type=int, default=20000)
    training.add_argument("--epochs", type=int, default=20)
    training.add_argument("--stage1-epochs", type=int, default=100)
    training.add_argument("--batch-size", type=int, default=256)
    training.add_argument("--primary-score", choices=SCORERS, default="top5_mse")
    training.add_argument("--learning-rate", type=float, default=2e-5)
    training.add_argument("--adversarial-weight", type=float, default=0.01)
    training.add_argument("--distillation-weight", type=float, default=1.0)
    training.add_argument("--gp-weight", type=float, default=10.0)
    training.add_argument("--critic-steps", type=int, default=1)
    training.add_argument("--warmup-epochs", type=int, default=5)
    training.add_argument("--filter-every", type=int, default=5)
    training.add_argument("--retention-end", type=float, default=0.8)
    training.add_argument("--patience", type=int, default=0)
    training.add_argument("--minimum-epochs", type=int, default=10)
    training.add_argument("--minimum-delta", type=float, default=1e-4)
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
