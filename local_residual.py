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
from sklearn.metrics import precision_recall_curve, roc_auc_score

import fgaae
from audit_data import full_digest
from calibrate_full_model import coefficient_subsets
from full_model import normalized


ROOT = Path(__file__).resolve().parent
WINDOWS = [5, 11, 21, 41]
LOCAL_WEIGHTS = [0.5, 1.0, 2.0, 4.0]
BASELINE = "calibrated_global"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@torch.no_grad()
def infer(model, values, checkpoint, device, batch_size):
    model.eval()
    residuals, errors, critics = [], [], []
    for start in range(0, len(values), batch_size):
        inputs = torch.from_numpy(normalized(values[start:start + batch_size], checkpoint["normalization"])).unsqueeze(1).to(device)
        reconstruction, _ = model(inputs)
        difference = reconstruction - inputs
        residuals.append((-difference).squeeze(1).cpu().numpy())
        errors.extend(difference.square().mean(dim=(1, 2)).cpu().tolist())
        critics.extend(model.critic(reconstruction).cpu().tolist())
    error = np.asarray(errors, dtype=np.float64)
    critic = np.asarray(critics, dtype=np.float64)
    config = checkpoint["config"]
    if config.get("reconstruction_score_reduction") != "sum" or config.get("score_calibration"):
        raise ValueError("This enhancement expects the existing unstandardized sum-reconstruction checkpoint.")
    scores = error * values.shape[1] + config["critic_score_sign"] * config["critic_score_weight"] * critic
    result = {"residual": np.concatenate(residuals), "global_score": scores, "reconstruction_error": error, "critic_value": critic}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("Non-finite residual inference.")
    return result


def robust_scalar(values):
    center = float(np.median(values))
    lower, upper = np.quantile(values, [0.25, 0.75])
    scale = max(float((upper - lower) / 1.349), 0.1 * float(np.std(values)), 1e-6)
    return {"center": center, "scale": scale}


def fit_residual_scale(residual):
    lower, upper = np.quantile(residual, [0.25, 0.75], axis=0)
    scale = (upper - lower) / 1.349
    positive = scale[scale > 0]
    floor = max(0.1 * float(np.median(positive)) if len(positive) else 0, 1e-4)
    return {"center": np.median(residual, axis=0).tolist(), "scale": np.maximum(scale, floor).tolist(), "scale_floor": floor}


def standardized_residual(residual, parameters):
    return (residual.astype(np.float64) - np.asarray(parameters["center"])) / np.asarray(parameters["scale"])


def point_energy(residual, parameters):
    return standardized_residual(residual, parameters) ** 2


def local_feature(energy, family, width):
    if family in ["window", "signed_window", "absolute_window"]:
        cumulative = np.pad(np.cumsum(energy, axis=1, dtype=np.float64), ((0, 0), (1, 0)))
        averages = (cumulative[:, width:] - cumulative[:, :-width]) / width
        return (np.abs(averages) if family == "absolute_window" else averages).max(axis=1)
    if family in ["topk", "signed_topk"]:
        return np.partition(energy, energy.shape[1] - width, axis=1)[:, -width:].mean(axis=1)
    raise ValueError(f"Unknown local residual family: {family}")


def recipes(length, mode="two-sided"):
    if mode == "coherent":
        return [("absolute_window", width) for width in WINDOWS]
    prefix = "signed_" if mode == "signed" else ""
    return [(prefix + "window", width) for width in WINDOWS] + [(prefix + "topk", max(1, round(length * 0.05)))]


def transformed(values, parameters):
    return (values - parameters["center"]) / parameters["scale"]


def enhanced_score(inference, profile):
    if profile["family"] == "baseline":
        return inference["global_score"]
    energy = standardized_residual(inference["residual"], profile["residual_scale"]) if profile["family"].startswith(("signed_", "absolute_")) else point_energy(inference["residual"], profile["residual_scale"])
    local = local_feature(energy, profile["family"], profile["width"])
    return transformed(inference["global_score"], profile["global_scale"]) + profile["weight"] * transformed(local, profile["local_scale"])


def evaluation_rows(seed, name, scores, metadata, selection):
    labels = metadata.label.to_numpy(dtype=np.int64)
    threshold = fgaae.select_threshold(scores[selection], labels[selection], {"threshold_method": "validation-f1"})
    row = {"seed": seed, "candidate": name, "threshold": threshold}
    for prefix, mask in [("selection", selection), ("check", ~selection), ("development", np.ones(len(labels), dtype=bool))]:
        metrics = fgaae.metrics(labels[mask], scores[mask], threshold)
        for key in ["roc_auc", "average_precision", "f1", "precision", "recall", "accuracy"]:
            row[f"{prefix}_{key}"] = metrics[key]
        tn, fp, _, _ = metrics["confusion_matrix_tn_fp_fn_tp"]
        row[f"{prefix}_false_positive_rate"] = fp / (tn + fp)
    return row


def choose_candidate(rows):
    frame = pd.DataFrame(rows)
    aggregate = frame.groupby("candidate", sort=False).mean(numeric_only=True).drop(columns="seed")
    ordered = aggregate.sort_values(["selection_f1", "selection_average_precision", "selection_roc_auc"], ascending=False, kind="stable")
    chosen = ordered.index[0]
    if aggregate.loc[chosen, "selection_f1"] < aggregate.loc[BASELINE, "selection_f1"] + 0.005:
        chosen = BASELINE
    return chosen, aggregate


def acceptance(chosen, aggregate):
    selected = aggregate.loc[chosen]
    baseline = aggregate.loc[BASELINE]
    return bool(
        chosen != BASELINE
        and selected.check_f1 >= baseline.check_f1 + 0.01
        and selected.check_recall >= baseline.check_recall + 0.03
        and selected.check_precision >= 0.80
        and selected.development_f1 >= baseline.development_f1 + 0.01
        and selected.development_recall >= baseline.development_recall + 0.03
    )


def pair_diagnostics(seed, raw, clean, injected, metadata, normalization, residual_scale):
    rows = []
    frequency_indices = np.arange(raw.shape[1])
    z_energy = point_energy(injected["residual"], residual_scale)
    for index in np.flatnonzero(metadata.label.to_numpy() == 1):
        record = metadata.iloc[index]
        support = np.abs(frequency_indices - int(record.center_index)) <= 3 * int(record.gaussian_sigma_points)
        delta = normalized(raw[index:index + 1], normalization)[0] - normalized(clean["raw"][index:index + 1], normalization)[0]
        residual_delta = injected["residual"][index].astype(np.float64) - clean["residual"][index].astype(np.float64)
        delta = delta.astype(np.float64)
        denominator = float(np.dot(delta[support], delta[support]))
        if denominator <= 1e-12:
            raise ValueError("Injected sample has no measurable perturbation in its declared support.")
        response = delta - residual_delta
        top = np.argpartition(z_energy[index], -42)[-42:]
        rows.append({
            "seed": seed, "source_row": int(record.source_row),
            "nominal_interference_ratio_db": float(record.nominal_interference_ratio_db),
            "copy_projection_gain": float(np.dot(delta[support], response[support]) / denominator),
            "residual_response_norm_ratio": float(np.linalg.norm(residual_delta[support]) / np.sqrt(denominator)),
            "injected_support_fraction_of_top42_residual_points": float(support[top].mean()),
            "support_point_count": int(support.sum()),
            "diagnostic_only_location_and_labels": True,
        })
    return rows


def save_scores(path, metadata, inference, profile):
    scores = enhanced_score(inference, profile)
    result = {"scores": scores, "reconstruction_error": inference["reconstruction_error"], "critic_value": inference["critic_value"]}
    fgaae.save_predictions(path, metadata.source_row.to_numpy(), metadata.label.to_numpy(), None, result, profile["threshold"])


def power_metrics(seed, variant, metadata, scores, threshold, dataset):
    labels = metadata.label.to_numpy(dtype=np.int64)
    rows = []
    for ratio in sorted(metadata.loc[metadata.label == 1, "nominal_interference_ratio_db"].unique()):
        mask = (labels == 0) | (metadata.nominal_interference_ratio_db.to_numpy() == ratio)
        metrics = fgaae.metrics(labels[mask], scores[mask], threshold)
        rows.append({"seed": seed, "variant": variant, "dataset": dataset, "nominal_interference_ratio_db": ratio,
            **{key: metrics[key] for key in ["roc_auc", "f1", "precision", "recall", "average_precision"]}})
    return rows


def profile_artifact(checkpoint_path, profile, accepted):
    return {"artifact_type": "FGAAE_local_residual_scoring_enhancement", "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": full_digest(checkpoint_path), "accepted_on_development_check": accepted,
        "scoring": profile, "paper_original_score": False,
        "local_power_increase_prior": profile["family"].startswith("signed_"),
        "inference_uses_anomaly_labels_or_locations": False,
        "prediction_entry": "local_residual.py predict; scorer JSON is separate from the unchanged FGAAE checkpoint."}


def predict(arguments):
    artifact = read_json(Path(arguments.artifact))
    checkpoint_path = Path(arguments.checkpoint or artifact["checkpoint"])
    if full_digest(checkpoint_path) != artifact["checkpoint_sha256"]:
        raise ValueError("The scoring profile does not match this checkpoint.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    device = torch.device(arguments.device)
    fgaae.seed_everything(2025, arguments.num_threads)
    model = fgaae.FGAAE(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    values, labels = fgaae.read_spectra(Path(arguments.data), checkpoint["config"]["label_column"])
    if values.shape[1] != checkpoint["config"]["sequence_length"]:
        raise ValueError("Input frequency dimension does not match the checkpoint.")
    result = infer(model, values, checkpoint, device, arguments.batch_size)
    scores = enhanced_score(result, artifact["scoring"])
    output = Path(arguments.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fgaae.save_predictions(output, np.arange(len(values)), labels, None, {
        "scores": scores, "reconstruction_error": result["reconstruction_error"], "critic_value": result["critic_value"],
    }, artifact["scoring"]["threshold"])
    print(f"LOCAL RESIDUAL PREDICTIONS: {output.resolve()}", flush=True)
    return scores


def run(arguments):
    parent = Path(arguments.round_run).resolve()
    parent_report = read_json(parent / "round_report.json")
    training_run = Path(parent_report["selected_training_run"])
    plan = read_json(training_run / "registered_plan.json")
    manifest = read_json(training_run / "inputs/manifest.json")
    metadata = pd.read_csv(training_run / "inputs/development_membership.csv")
    selection, selection_files, check_files = coefficient_subsets(metadata)
    dataset = Path(plan["dataset"])
    station_inputs = Path(plan["station_run"]) / "inputs"
    station_manifest = read_json(station_inputs / "manifest.json")
    output = Path(arguments.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    for name in ["local_residual.py", "fgaae.py", "full_model.py"]:
        shutil.copy2(ROOT / name, source / name)
    registered = {
        "scope": "Frozen complete FGAAE plus local residual scoring enhancement; no retraining or component ablation.",
        "training_run": str(training_run), "dataset": str(dataset), "station_inputs": str(station_inputs),
        "seeds": plan["seeds"], "device": arguments.device, "batch_size": arguments.batch_size,
        "selection_files": selection_files, "check_files": check_files,
        "mode": arguments.mode, "local_families": recipes(840, arguments.mode), "local_weights": LOCAL_WEIGHTS,
        "residual_normalization": "Per-frequency signed-residual median and IQR/1.349 fitted on feature-training reference only; scale floor max(0.1*median-positive-scale,1e-4).",
        "score": "Training-reference robust-scaled original calibrated global score + positive weight * robust-scaled local residual statistic.",
        "direction_caveat": "Signed mode uses coherent positive residual shifts, matching this frozen power-increase injection protocol. It is not verified for negative notches, missing signals or other anomaly families.",
        "candidate_selection": "One shared recipe across seeds; earlier-file mean F1, AP then AUC as tie breakers; require selection F1 improvement >=0.005 or retain baseline.",
        "acceptance": "Later-file F1 +0.01, recall +0.03, precision >=0.80; whole-development F1 +0.01 and recall +0.03. Later files were previously used for checkpoint selection, not independent testing.",
        "paired_diagnostic": "Unmodified source-row counterpart; projection of reconstructed perturbation on injected perturbation within +/-3 Gaussian sigma. Location and labels are diagnostic only.",
        "historical_confirmation": "Evaluate once after selection and acceptance are frozen. Previously inspected temporal validation files, not a new independent or cross-station test.",
        "test_or_external_test_evaluated": False, "paper_original_experiment_reproduced": False,
        "code_sha256": full_digest(Path(__file__)), "core_code_sha256": full_digest(ROOT / "fgaae.py"),
    }
    fgaae.write_json(output / "registered_plan.json", registered)
    if full_digest(dataset / "spectra.npy") != manifest["dataset_sha256"] or full_digest(station_inputs / "development.npy") != manifest["development_sha256"]:
        raise ValueError("Frozen data changed.")
    if full_digest(ROOT / "fgaae.py") != plan["core_code_sha256"]:
        raise ValueError("The existing checkpoint core code changed.")
    partitions = pd.read_csv(dataset / "partitions.csv").set_index("row")
    if not partitions.loc[metadata.source_row, "split"].eq("validation").all():
        raise ValueError("Development contains non-validation rows.")
    original = np.load(dataset / "spectra.npy", mmap_mode="r", allow_pickle=False)
    reference = original[manifest["reference_training_rows"]].astype(np.float32)
    clean_raw = original[metadata.source_row.to_numpy()].astype(np.float32)
    values = np.load(station_inputs / "development.npy", allow_pickle=False)
    if not np.array_equal(values[metadata.label.to_numpy() == 0], clean_raw[metadata.label.to_numpy() == 0]):
        raise ValueError("Unmodified background does not match source rows.")
    fgaae.seed_everything(2025, arguments.num_threads)
    device = torch.device(arguments.device)
    rows, diagnostics, model_records, cached, examples = [], [], {}, {}, []
    for seed in plan["seeds"]:
        checkpoint_path = training_run / "score_calibration" / f"seed_{seed}_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not checkpoint["all_components_enabled"]:
            raise ValueError("Expected the complete trained FGAAE.")
        model = fgaae.FGAAE(checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
        reference_inference = infer(model, reference, checkpoint, device, arguments.batch_size)
        current = infer(model, values, checkpoint, device, arguments.batch_size)
        clean = infer(model, clean_raw, checkpoint, device, arguments.batch_size)
        clean["raw"] = clean_raw
        saved = pd.read_csv(training_run / "score_calibration" / f"seed_{seed}_development_scores.csv", float_precision="round_trip")
        if not np.array_equal(saved.source_row, metadata.source_row) or not np.allclose(current["global_score"], saved.anomaly_score, rtol=1e-5, atol=1e-3):
            raise ValueError("Frozen baseline inference or membership mismatch.")
        residual_scale = fit_residual_scale(reference_inference["residual"])
        global_scale = robust_scalar(reference_inference["global_score"])
        ref_energy = standardized_residual(reference_inference["residual"], residual_scale) if arguments.mode in ["signed", "coherent"] else point_energy(reference_inference["residual"], residual_scale)
        dev_energy = standardized_residual(current["residual"], residual_scale) if arguments.mode in ["signed", "coherent"] else point_energy(current["residual"], residual_scale)
        profiles = {BASELINE: {"family": "baseline", "width": 0, "weight": 0.0}}
        candidate_scores = {BASELINE: current["global_score"]}
        rows.append(evaluation_rows(seed, BASELINE, current["global_score"], metadata, selection))
        for family, width in recipes(values.shape[1], arguments.mode):
            ref_local = local_feature(ref_energy, family, width)
            dev_local = local_feature(dev_energy, family, width)
            local_scale = robust_scalar(ref_local)
            for weight in LOCAL_WEIGHTS:
                name = f"{family}_{width}_weight_{weight:g}"
                profile = {"family": family, "width": width, "weight": weight, "residual_scale": residual_scale, "global_scale": global_scale, "local_scale": local_scale}
                scores = transformed(current["global_score"], global_scale) + weight * transformed(dev_local, local_scale)
                profiles[name] = profile
                candidate_scores[name] = scores
                rows.append(evaluation_rows(seed, name, scores, metadata, selection))
        diagnostics.extend(pair_diagnostics(seed, values, clean, current, metadata, checkpoint["normalization"], residual_scale))
        if seed == plan["seeds"][len(plan["seeds"]) // 2]:
            for ratio in [-14, -10, -6, 0]:
                index = int(np.flatnonzero((metadata.label == 1) & (metadata.nominal_interference_ratio_db == ratio))[0])
                examples.append({"seed": seed, "ratio": ratio, "center": int(metadata.iloc[index].center_index), "source_row": int(metadata.iloc[index].source_row),
                    "clean_input": normalized(clean_raw[index:index + 1], checkpoint["normalization"])[0],
                    "injected_input": normalized(values[index:index + 1], checkpoint["normalization"])[0],
                    "clean_residual": clean["residual"][index], "injected_residual": current["residual"][index],
                    "residual_center": np.asarray(residual_scale["center"]), "residual_scale": np.asarray(residual_scale["scale"])})
        model_records[seed] = {"checkpoint_path": checkpoint_path, "checkpoint_sha256": full_digest(checkpoint_path), "profiles": profiles}
        cached[seed] = {"inference": current, "scores": candidate_scores}
        print(f"RESIDUAL INFERENCE seed={seed}: reference={len(reference)}, paired_development={len(values)}, baseline_max_difference={float(np.max(np.abs(current['global_score'] - saved.anomaly_score))):.8g}", flush=True)
        del model, reference_inference, clean, ref_energy, dev_energy
        if device.type == "cuda":
            torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "all_candidates.csv", index=False)
    chosen, aggregate = choose_candidate(rows)
    aggregate.to_csv(output / "candidate_summary.csv")
    accepted = acceptance(chosen, aggregate)
    recommended = chosen if accepted else BASELINE
    fgaae.write_json(output / "locked_selection.json", {"fit_selected_candidate": chosen, "accepted": accepted, "recommended_candidate": recommended, "historical_confirmation_not_yet_evaluated": True})
    diagnostic = pd.DataFrame(diagnostics)
    diagnostic.to_csv(output / "paired_diagnostics.csv", index=False)
    diagnostic.groupby("nominal_interference_ratio_db").agg(
        copy_gain_median=("copy_projection_gain", "median"), copy_gain_mean=("copy_projection_gain", "mean"),
        residual_response_norm_ratio_median=("residual_response_norm_ratio", "median"),
        top42_support_fraction_mean=("injected_support_fraction_of_top42_residual_points", "mean"),
    ).to_csv(output / "paired_diagnostics_by_ratio.csv")
    artifact_directory = output / "scorers"
    artifact_directory.mkdir()
    power_rows, verification = [], []
    for seed in plan["seeds"]:
        for name, suffix in [(BASELINE, "baseline"), (chosen, "fit_selected"), (recommended, "recommended")]:
            profile = {**model_records[seed]["profiles"][name]}
            row = frame[(frame.seed == seed) & (frame.candidate == name)].iloc[0]
            profile["threshold"] = float(row.threshold)
            artifact = profile_artifact(model_records[seed]["checkpoint_path"], profile, accepted and name == chosen)
            fgaae.write_json(artifact_directory / f"seed_{seed}_{suffix}.json", artifact)
            save_scores(output / f"seed_{seed}_{suffix}_development.csv", metadata, cached[seed]["inference"], profile)
            if suffix in ["baseline", "recommended"]:
                power_rows.extend(power_metrics(seed, suffix, metadata, cached[seed]["scores"][name], profile["threshold"], "development"))
        verification.append({"seed": seed, "checkpoint_unchanged": full_digest(model_records[seed]["checkpoint_path"]) == model_records[seed]["checkpoint_sha256"], "baseline_reproduced": True})
    historical_rows = historical_confirmation(output, recommended, model_records, frame, plan, manifest, station_manifest, metadata, partitions, arguments)
    power_rows.extend(historical_rows[1])
    pd.DataFrame(historical_rows[0]).to_csv(output / "historical_confirmation_metrics.csv", index=False)
    pd.DataFrame(power_rows).to_csv(output / "metrics_by_injection_ratio.csv", index=False)
    summary = {
        "fit_selected_candidate": chosen, "accepted_on_development_check": accepted, "recommended_candidate": recommended,
        "baseline": aggregate.loc[BASELINE].to_dict(), "fit_selected": aggregate.loc[chosen].to_dict(),
        "recommended": aggregate.loc[recommended].to_dict(),
        "development_f1_sample_std": float(frame[frame.candidate == recommended].development_f1.std(ddof=1)),
        "development_auc_sample_std": float(frame[frame.candidate == recommended].development_roc_auc.std(ddof=1)),
        "historical_confirmation": pd.DataFrame(historical_rows[0]).groupby("variant").mean(numeric_only=True).drop(columns="seed").to_dict(orient="index"),
        "independent_test": False, "cross_station_generalization_verified": False,
        "network_retrained": False, "original_model_checkpoints_modified": False,
        "paper_original_score": False, "scope": "Learned residual scoring enhancement to complete FGAAE, not strict original-paper reproduction.",
        "mode": arguments.mode, "local_power_increase_prior": arguments.mode == "signed",
    }
    fgaae.write_json(output / "report.json", summary)
    fgaae.write_json(output / "verification.json", {"runs": verification, "anomaly_locations_used_only_in_diagnostics": True, "selection_uses_earlier_development_only": True, "normalization_uses_training_reference_only": True})
    plot_results(output, metadata, plan["seeds"], examples)
    write_report(output, summary, plan["seeds"])
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"LOCAL RESIDUAL RUN COMPLETE: {output}", flush=True)
    return output


def historical_confirmation(output, recommended, records, frame, plan, manifest, station_manifest, development, partitions, arguments):
    inputs = Path(plan["station_run"]) / "inputs"
    path = inputs / "confirmation.npy"
    if full_digest(path) != station_manifest["confirmation_sha256"]:
        raise ValueError("Historical confirmation values changed.")
    metadata = pd.read_csv(inputs / "confirmation_metadata.csv")
    forbidden = set(development.source_file) | set(manifest["reference_training_files"]) | set(manifest["reference_validation_files"])
    training = pd.read_csv(Path(plan["output"]) / "inputs/training_membership.csv")
    forbidden.update(training.source_file)
    if not set(metadata.source_file).isdisjoint(forbidden):
        raise ValueError("Historical confirmation shares a source file with training or development.")
    if not partitions.loc[metadata.source_row, "split"].eq("validation").all():
        raise ValueError("Historical confirmation includes test rows.")
    values = np.load(path, allow_pickle=False)
    device = torch.device(arguments.device)
    rows, power = [], []
    for seed in plan["seeds"]:
        checkpoint = torch.load(records[seed]["checkpoint_path"], map_location="cpu", weights_only=True)
        model = fgaae.FGAAE(checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
        result = infer(model, values, checkpoint, device, arguments.batch_size)
        for name, variant in [(BASELINE, "baseline"), (recommended, "recommended")]:
            row = frame[(frame.seed == seed) & (frame.candidate == name)].iloc[0]
            profile = {**records[seed]["profiles"][name], "threshold": float(row.threshold)}
            scores = enhanced_score(result, profile)
            metrics = fgaae.metrics(metadata.label.to_numpy(dtype=np.int64), scores, profile["threshold"])
            rows.append({"seed": seed, "variant": variant, **{key: metrics[key] for key in ["roc_auc", "f1", "precision", "recall", "average_precision", "accuracy"]}})
            save_scores(output / f"seed_{seed}_{variant}_historical_confirmation.csv", metadata, result, profile)
            power.extend(power_metrics(seed, variant, metadata, scores, profile["threshold"], "historical_confirmation"))
        print(f"HISTORICAL CONFIRMATION seed={seed}: {len(values)} rows, locked scoring and thresholds; not an independent test", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows, power


def plot_results(output, metadata, seeds, examples):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    grid = np.linspace(0, 1, 501)
    power = pd.read_csv(output / "metrics_by_injection_ratio.csv")
    for variant in ["baseline", "recommended"]:
        precisions = []
        for seed in seeds:
            scores = pd.read_csv(output / f"seed_{seed}_{variant}_development.csv", float_precision="round_trip")
            precision, recall, _ = precision_recall_curve(scores.label, scores.anomaly_score)
            precisions.append(np.interp(grid, recall[::-1], precision[::-1]))
        axes[0].plot(grid, np.mean(precisions, axis=0), label=variant)
        group = power[(power.variant == variant) & (power.dataset == "development")].groupby("nominal_interference_ratio_db")
        for axis, column in [(axes[1], "roc_auc"), (axes[2], "recall")]:
            grouped = group[column].agg(["mean", "std"])
            axis.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"].fillna(0), marker="o", capsize=2, label=variant)
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Frozen development PR curves", xlim=(0, 1), ylim=(0, 1.02))
    axes[1].set(xlabel="Injected/background power ratio (dB)", ylabel="ROC-AUC", title="Not channel SNR")
    axes[2].set(xlabel="Injected/background power ratio (dB)", ylabel="Recall at frozen threshold", title="Threshold fitted on earlier files")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(len(examples), 2, figsize=(13, 3 * len(examples)), squeeze=False)
    frequencies = np.linspace(87, 108, 840, endpoint=False)
    for index, example in enumerate(examples):
        start, end = max(0, example["center"] - 35), min(840, example["center"] + 36)
        region = slice(start, end)
        axis = axes[index, 0]
        axis.plot(frequencies[region], example["clean_input"][region], label="Observed background")
        axis.plot(frequencies[region], example["injected_input"][region], label="Injected input")
        axis.plot(frequencies[region], (example["injected_input"] - example["injected_residual"])[region], label="Reconstruction")
        axis.set_title(f"Seed {example['seed']}, ratio {example['ratio']} dB, source row {example['source_row']}")
        for label, residual in [("Background residual", example["clean_residual"]), ("Injected residual", example["injected_residual"])]:
            standardized = (residual - example["residual_center"]) / example["residual_scale"]
            axes[index, 1].plot(frequencies[region], standardized[region], label=label)
        axes[index, 1].set_title("Signed residual / training reference scale")
        for axis in axes[index]:
            axis.set_xlabel("Frequency (MHz)")
            axis.grid(alpha=0.3)
            if index == 0:
                axis.legend()
    figure.tight_layout()
    figure.savefig(output / "paired_residual_examples.png", dpi=150)
    plt.close(figure)


def write_report(output, summary, seeds):
    baseline = summary["baseline"]
    selected = summary["recommended"]
    diagnostic = pd.read_csv(output / "paired_diagnostics_by_ratio.csv")
    weak = diagnostic[diagnostic.nominal_interference_ratio_db <= -8]
    lines = ["# 完整 FGAAE 局部残差诊断与评分增强", "",
        f"早期开发文件选中的方案：**{summary['fit_selected_candidate']}**；通过后期开发检查：**{summary['accepted_on_development_check']}**；建议使用：**{summary['recommended_candidate']}**。", "",
        "原 FGAAE 网络权重、全局评分及历史成绩均保留。本轮新增的是基于模型重构残差的局部评分，不是原论文公式的严格复现，也不是纯原始背景统计基线。所有候选均保留原全局评分中的重构与 Critic 项。", "",
        "| 开发集均值 | 原校准全局评分 | 建议评分 |", "| --- | --- | --- |",
    ]
    for label, column in [("AUC", "roc_auc"), ("AP", "average_precision"), ("F1", "f1"), ("Precision", "precision"), ("Recall", "recall")]:
        lines.append(f"| {label} | {baseline['development_' + column]:.6f} | {selected['development_' + column]:.6f} |")
    lines.extend(["", f"建议评分的 AUC 样本标准差：{summary['development_auc_sample_std']:.6f}；F1 样本标准差：{summary['development_f1_sample_std']:.6f}。三种子共用同一局部窗口/聚合方式和权重，频点及分数尺度只从训练参考拟合。", "",
        f"评分模式：{summary['mode']}。正向模式利用当前异常为局部功率增加这一先验，不能据此宣称负向缺失、凹口或其他异常类型也获得同样提升。",
        "## 诊断", "", f"各弱注入分组（≤−8 dB）的复制投影增益中位数范围为 {weak.copy_gain_median.min():.4f}～{weak.copy_gain_median.max():.4f}，残差响应范数比中位数范围为 {weak.residual_response_norm_ratio_median.min():.4f}～{weak.residual_response_norm_ratio_median.max():.4f}。",
        "", "复制投影增益约 0 表示模型很少沿注入扰动方向重构该扰动；约 1 表示扰动被较完整复制。残差响应范数比约 1 表示注入变化仍保留在残差中。该诊断使用已知注入位置，仅解释模型行为；预测器扫描全部频点，不读取异常位置或标签。",
        "", "## 较晚历史确认文件", "", "在候选和阈值锁定后才计算历史确认子集。该集合以前已查看过，且仍来自同站点、同合成协议，不能称为新的独立测试或跨站泛化证明。", "",
        "| 历史确认均值 | 原校准全局评分 | 建议评分 |", "| --- | --- | --- |",
    ])
    for column in ["roc_auc", "f1", "precision", "recall", "average_precision"]:
        lines.append(f"| {column} | {summary['historical_confirmation']['baseline'][column]:.6f} | {summary['historical_confirmation']['recommended'][column]:.6f} |")
    lines.extend(["", "开发后半文件也曾参与原模型检查点选择，所以本轮接受规则仍是开发调参。所有 dB 数值均为注入/背景功率比，不是论文信道 SNR。", "",
        "## 模型与推理", "", "神经网络检查点保持原路径。新增参数保存在 scorers/seed_种子_recommended.json，包含原检查点路径和哈希、残差尺度、局部规则、全局与局部分数尺度和早期开发阈值。使用 local_residual.py predict 加载该 JSON；旧 fgaae.py predict 仍计算原全局基准评分。", "",
        "```powershell", f"python local_residual.py predict --artifact runs/{output.name}/scorers/seed_{seeds[0]}_recommended.json --data runs/station_background_v2/inputs/development.npy --output local_predictions.csv --device cuda", "```", "",
        "[候选表](candidate_summary.csv)、[全部种子候选](all_candidates.csv)、[复制诊断](paired_diagnostics_by_ratio.csv)、[历史确认结果](historical_confirmation_metrics.csv)、[分组表现](metrics_by_injection_ratio.csv)、[对比图](comparison.png)、[残差示例](paired_residual_examples.png)。",
    ])
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def smoke_test():
    fgaae.seed_everything(7, 2)
    array = np.zeros((2, 20))
    array[0, 6:9] = 3
    assert np.allclose(local_feature(array, "window", 3), [3, 0])
    assert np.allclose(local_feature(array, "topk", 3), [3, 0])
    negative = -array
    assert np.allclose(local_feature(negative, "absolute_window", 3), [3, 0])
    parameters = fit_residual_scale(np.zeros((20, 20)))
    assert np.isfinite(point_energy(array, parameters)).all()
    generator = np.random.default_rng(7)
    metadata = pd.DataFrame({"source_file": np.repeat(["a", "b", "c", "d"], 32), "ticks": np.arange(128), "label": np.tile([0, 0, 0, 1], 32)})
    selection, _, _ = coefficient_subsets(metadata)
    scores = generator.normal(size=128) + metadata.label.to_numpy()
    first = [evaluation_rows(1, BASELINE, scores, metadata, selection), evaluation_rows(1, "other", scores + metadata.label.to_numpy(), metadata, selection)]
    modified = metadata.copy()
    modified.loc[~selection, "label"] = 1 - modified.loc[~selection, "label"]
    second = [evaluation_rows(1, BASELINE, scores, modified, selection), evaluation_rows(1, "other", scores + metadata.label.to_numpy(), modified, selection)]
    assert choose_candidate(first)[0] == choose_candidate(second)[0]
    with tempfile.TemporaryDirectory(prefix="fgaae_local_") as temporary:
        directory = Path(temporary)
        config = vars(fgaae.build_parser().parse_args(["train"]))
        config.update({"sequence_length": 840, "latent_dim": 32, "batch_size": 16, "reconstruction_score_reduction": "sum", "critic_score_sign": 1})
        model = fgaae.FGAAE(config)
        checkpoint = {"config": config, "model_state": model.state_dict(), "normalization": {"mean": 200.0, "std": 20.0}, "all_components_enabled": True}
        path = directory / "model.pt"
        torch.save(checkpoint, path)
        raw = generator.normal(200, 20, (64, 840)).astype(np.float32)
        result = infer(model, raw, checkpoint, torch.device("cpu"), 16)
        expected = fgaae.evaluate(model, normalized(raw, checkpoint["normalization"]), config, torch.device("cpu"))
        assert np.allclose(result["global_score"], expected["scores"], atol=1e-6)
        scale = fit_residual_scale(result["residual"])
        np.save(directory / "input.npy", raw)
        for family in ["window", "signed_window", "absolute_window"]:
            points = standardized_residual(result["residual"], scale) if family.startswith(("signed_", "absolute_")) else point_energy(result["residual"], scale)
            local = local_feature(points, family, 11)
            profile = {"family": family, "width": 11, "weight": 2.0, "residual_scale": scale, "global_scale": robust_scalar(result["global_score"]), "local_scale": robust_scalar(local), "threshold": 0.0}
            artifact = profile_artifact(path, profile, False)
            fgaae.write_json(directory / "scorer.json", artifact)
            actual = predict(argparse.Namespace(artifact=str(directory / "scorer.json"), checkpoint=None, data=str(directory / "input.npy"), output=str(directory / f"{family}_predictions.csv"), device="cpu", batch_size=16, num_threads=2))
            assert np.array_equal(actual, enhanced_score(result, profile))
            assert full_digest(path) == artifact["checkpoint_sha256"]
    print("LOCAL RESIDUAL SMOKE TEST PASSED: local aggregation, zero-variance floor, later labels excluded from selection, baseline parity and scorer serialization/prediction.")


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description="Diagnose frozen FGAAE residuals and test an explicitly enhanced local scoring rule.")
    commands = parser.add_subparsers(dest="command", required=True)
    experiment = commands.add_parser("run")
    experiment.add_argument("--round-run", default=str(ROOT / "runs/full_fgaae_improvement_round1"))
    experiment.add_argument("--output", default=str(ROOT / "runs/local_residual_v1"))
    experiment.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    experiment.add_argument("--batch-size", type=int, default=256)
    experiment.add_argument("--num-threads", type=int, default=2)
    experiment.add_argument("--mode", choices=["two-sided", "signed", "coherent"], default="two-sided")
    prediction = commands.add_parser("predict")
    prediction.add_argument("--artifact", required=True)
    prediction.add_argument("--checkpoint")
    prediction.add_argument("--data", required=True)
    prediction.add_argument("--output", required=True)
    prediction.add_argument("--device", default="cpu")
    prediction.add_argument("--batch-size", type=int, default=256)
    prediction.add_argument("--num-threads", type=int, default=2)
    commands.add_parser("smoke-test")
    arguments = parser.parse_args()
    if arguments.command == "run":
        run(arguments)
    elif arguments.command == "predict":
        predict(arguments)
    else:
        smoke_test()


if __name__ == "__main__":
    main()
