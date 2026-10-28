from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import fgaae
import local_residual
from audit_data import full_digest


ROOT = Path(__file__).resolve().parent


def verify(arguments):
    runs = [Path(path).resolve() for path in arguments.runs]
    output = Path(arguments.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    first_plan = local_residual.read_json(runs[0] / "registered_plan.json")
    training = Path(first_plan["training_run"])
    training_plan = local_residual.read_json(training / "registered_plan.json")
    dataset = Path(first_plan["dataset"])
    inputs = Path(first_plan["station_inputs"])
    seeds = first_plan["seeds"]
    policies = []
    for run in runs:
        plan = local_residual.read_json(run / "registered_plan.json")
        if plan["seeds"] != seeds or plan["training_run"] != str(training):
            raise ValueError("Scoring comparisons require the same models and seeds.")
        for seed in seeds:
            path = run / "scorers" / f"seed_{seed}_recommended.json"
            policies.append({"run": str(run), "seed": seed, "scorer_sha256": full_digest(path)})
    fgaae.write_json(output / "registered_plan.json", {
        "runs": [str(run) for run in runs], "scorers": policies,
        "deployment_check": "Reload scoring JSON and checkpoints through the public prediction entry on all frozen development and historical confirmation frames.",
        "direction_stress_enabled": arguments.direction_shift,
        "direction_stress": "Mirror existing positive level perturbation: negative = source_background - (frozen_positive - source_background). Same label positions, opposite raw-level direction; not a channel-SNR-equivalent physical interference protocol.",
        "selection": "No scorer weights, thresholds, checkpoints or recommendation are changed using stress results.",
        "independent_test": False, "cross_station_test": False,
        "code_sha256": full_digest(Path(__file__)),
    })
    verification = []
    for run in runs:
        for seed in seeds:
            for dataset_name, original_suffix in [("development", "development"), ("confirmation", "historical_confirmation")]:
                original = pd.read_csv(run / f"seed_{seed}_recommended_{original_suffix}.csv", float_precision="round_trip")
                predicted_path = output / f"{run.name}_seed_{seed}_{dataset_name}.csv"
                scores = local_residual.predict(argparse.Namespace(
                    artifact=str(run / "scorers" / f"seed_{seed}_recommended.json"), checkpoint=None,
                    data=str(inputs / f"{dataset_name}.npy"), output=str(predicted_path),
                    device=arguments.device, batch_size=arguments.batch_size, num_threads=arguments.num_threads,
                ))
                predicted = pd.read_csv(predicted_path, float_precision="round_trip")
                difference = float(np.max(np.abs(original.anomaly_score.to_numpy() - scores)))
                if not np.allclose(original.anomaly_score, scores, rtol=1e-6, atol=1e-5):
                    raise ValueError("Reloaded local score differs from the saved experiment.")
                if not np.array_equal(original.predicted_label, predicted.predicted_label):
                    raise ValueError("Reloaded threshold decisions differ.")
                verification.append({"run": run.name, "seed": seed, "dataset": dataset_name,
                    "rows": len(scores), "maximum_score_difference": difference, "all_decisions_equal": True})
    fgaae.write_json(output / "deployment_verification.json", {"total_predictions_checked": sum(row["rows"] for row in verification), "runs": verification})
    if arguments.direction_shift:
        stress_check(runs, output, dataset, inputs, seeds, arguments)
    for policy in policies:
        path = Path(policy["run"]) / "scorers" / f"seed_{policy['seed']}_recommended.json"
        if full_digest(path) != policy["scorer_sha256"]:
            raise ValueError("A scoring artifact changed during verification.")
    if full_digest(ROOT / "fgaae.py") != training_plan["core_code_sha256"]:
        raise ValueError("The base model code changed.")
    print(json.dumps({"all_prediction_checks_passed": True, "predictions_checked": sum(row["rows"] for row in verification), "scorers_unchanged": True, "output": str(output)}, ensure_ascii=False, indent=2), flush=True)


def stress_check(runs, output, dataset, inputs, seeds, arguments):
    metadata = pd.read_csv(inputs / "confirmation_metadata.csv")
    manifest = local_residual.read_json(inputs / "manifest.json")
    dataset_manifest = local_residual.read_json(dataset / "dataset.json")
    if full_digest(inputs / "confirmation.npy") != manifest["confirmation_sha256"] or full_digest(dataset / "spectra.npy") != dataset_manifest["spectra_sha256"]:
        raise ValueError("Stress source values changed.")
    clean = np.load(dataset / "spectra.npy", mmap_mode="r", allow_pickle=False)[metadata.source_row.to_numpy()].astype(np.float32)
    positive = np.load(inputs / "confirmation.npy", allow_pickle=False)
    negative = clean - (positive - clean)
    if not np.isfinite(negative).all():
        raise FloatingPointError("Non-finite mirrored perturbation.")
    if not np.array_equal(negative[metadata.label == 0], clean[metadata.label == 0]):
        raise ValueError("Stress changed nominal background rows.")
    negative_path = output / "mirror_negative_levels.npy"
    np.save(negative_path, negative)
    stress_metadata = metadata.rename(columns={"nominal_interference_ratio_db": "positive_reference_ratio_db"}).copy()
    stress_metadata.loc[stress_metadata.label == 1, "label_origin"] = "synthetic_mirrored_negative_levels_not_channel_snr"
    stress_metadata.to_csv(output / "mirror_negative_metadata.csv", index=False, encoding="utf-8-sig")
    fgaae.write_json(output / "mirror_negative_protocol.json", {
        "source_positive_sha256": manifest["confirmation_sha256"], "source_background_sha256": dataset_manifest["spectra_sha256"],
        "generated_sha256": full_digest(negative_path), "rows": len(negative),
        "rule": "source - (positive - source), in original spectrum level units",
        "physical_snr_equivalent": False, "used_for_model_or_scorer_selection": False,
        "historically_seen_background_files": True,
    })
    rows = []
    device = torch.device(arguments.device)
    fgaae.seed_everything(2025, arguments.num_threads)
    for seed in seeds:
        first = local_residual.read_json(runs[0] / "scorers" / f"seed_{seed}_recommended.json")
        checkpoint_path = Path(first["checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = fgaae.FGAAE(checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
        inference = local_residual.infer(model, negative, checkpoint, device, arguments.batch_size)
        profiles = [("baseline", local_residual.read_json(runs[0] / "scorers" / f"seed_{seed}_baseline.json"))]
        profiles.extend((run.name, local_residual.read_json(run / "scorers" / f"seed_{seed}_recommended.json")) for run in runs)
        for variant, artifact in profiles:
            if artifact["checkpoint_sha256"] != full_digest(checkpoint_path):
                raise ValueError("Stress variants do not share identical model weights.")
            profile = artifact["scoring"]
            scores = local_residual.enhanced_score(inference, profile)
            metrics = fgaae.metrics(metadata.label.to_numpy(dtype=np.int64), scores, profile["threshold"])
            rows.append({"seed": seed, "variant": variant, **{key: metrics[key] for key in ["roc_auc", "f1", "precision", "recall", "average_precision", "accuracy"]}})
            local_residual.save_scores(output / f"{variant}_seed_{seed}_mirror_negative.csv", metadata, inference, profile)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "mirror_negative_metrics.csv", index=False)
    summary = frame.groupby("variant").mean(numeric_only=True).drop(columns="seed")
    summary.to_csv(output / "mirror_negative_summary.csv")
    print("FROZEN DIRECTION-SHIFT STRESS (not independent or physical-SNR matched):", flush=True)
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description="Verify local scoring artifacts and diagnose frozen direction-shift robustness without retuning.")
    parser.add_argument("--runs", nargs="+", default=[str(ROOT / "runs/local_residual_v2_signed"), str(ROOT / "runs/local_residual_v3_coherent")])
    parser.add_argument("--output", default=str(ROOT / "runs/local_residual_checks"))
    parser.add_argument("--direction-shift", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-threads", type=int, default=2)
    verify(parser.parse_args())
