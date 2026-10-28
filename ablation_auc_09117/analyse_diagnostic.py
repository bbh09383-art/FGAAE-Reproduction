"""Verify completed diagnostic outputs and derive feature-conditioned AUC contrasts."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from freeze_baseline import ROOT, HERE, digest, read_json, write_json


def analyse(run):
    report = read_json(run / "report.json")
    if not report["complete"]:
        raise ValueError("Diagnostics have not completed.")
    plan = read_json(run / "registered_plan.json")
    for name, checksum in plan["code_sha256"].items():
        if digest(HERE / name) != checksum or digest(run / "source" / name) != checksum:
            raise ValueError(f"Diagnostic training source mismatch: {name}")
    for name, checksum in plan["previous_artifacts_sha256"].items():
        if digest(ROOT / name) != checksum:
            raise ValueError(f"Reused results changed: {name}")
    previous = Path(plan["previous_run"])
    old = pd.read_csv(previous / "auc_runs.csv")
    table = pd.read_csv(run / "auc_runs.csv")
    metadata = pd.read_csv(run / "inputs/development_membership.csv")
    verified, epochs, frames = [], 0, 0
    for arm in ["AQT_only", "GAN_only"]:
        replay = read_json(run / arm / "deployment_verification.json")
        for seed in [2025, 2026, 2027]:
            directory = run / arm / f"seed_{seed}"
            history = pd.read_csv(directory / "stage2_history.csv")
            assert len(history) == 60 and history.iteration.nunique() == 3
            assert np.isfinite(history.select_dtypes(include="number").to_numpy()).all()
            assert not any(name in history for name in ["validation_f1", "validation_recall", "validation_precision"])
            if arm == "GAN_only":
                assert history.retained_samples.eq(20000).all()
            initialization = read_json(directory / "initialization.json")
            assert initialization["expert_pretraining_loaded"] is False
            assert initialization["bn_calibration"]["parameter_updates"] == 0
            assert initialization["bn_calibration"]["expert_features_or_labels_used"] is False
            audit = read_json(directory / "component_audit.json")
            changes = audit["parameter_changes"]
            assert changes["teacher"] == changes["student"] == 0
            assert (changes["critic"] > 0) == (arm == "GAN_only")
            assert all(value == 0 for value in [*audit["encoder_bn_changes_selected"].values(), *audit["encoder_bn_changes_latest"].values()])
            scores = pd.read_csv(directory / "development_scores.csv", float_precision="round_trip")
            assert np.array_equal(scores.source_row, metadata.source_row) and np.array_equal(scores.label, metadata.label)
            expected = table[(table.arm == arm) & (table.seed == seed)].iloc[0]
            primary = float(roc_auc_score(scores.label, scores.score))
            reconstruction = float(roc_auc_score(scores.label, scores.reconstruction_sum))
            assert abs(primary - expected.auc_primary) < 1e-12
            assert abs(reconstruction - expected.auc_reconstruction) < 1e-12
            check = next(row for row in replay if row["seed"] == seed)
            assert check["maximum_score_difference"] == 0
            assert digest(directory / "model.pt") == check["model_sha256"]
            epochs += len(history)
            frames += len(scores)
            verified.append({"arm": arm, "seed": seed, "auc_recomputed": primary,
                             "cuda_replay_frames": len(scores), "cuda_replay_max_score_difference": 0.0})
    assert epochs == 360 and frames == 63552

    # Exploratory extension: AQT is on in all four cells. Expert guidance means
    # the complete pretraining/distillation process, not expert knowledge alone.
    aq = table[table.arm == "AQT_only"].copy()
    group = pd.concat([old[old.arm.isin(["A", "B", "Full"])], aq], ignore_index=True)
    effects = []
    for score in ["auc_primary", "auc_reconstruction"]:
        pivot = group.pivot(index="seed", columns="arm", values=score)
        for seed, row in pivot.iterrows():
            values = {"GAN_without_features_AQT_on": row.B-row.AQT_only,
                      "GAN_with_features_AQT_on": row.Full-row.A,
                      "features_without_GAN_AQT_on": row.A-row.AQT_only,
                      "features_with_GAN_AQT_on": row.Full-row.B,
                      "feature_GAN_interaction_AQT_on": row.Full-row.A-row.B+row.AQT_only}
            effects.extend({"score": score, "seed": int(seed), "contrast": name, "auc_difference": float(value)} for name, value in values.items())
    contrasts = pd.DataFrame(effects)
    contrasts.to_csv(run / "feature_conditioning_by_seed.csv", index=False)
    summary = contrasts.groupby(["score", "contrast"]).agg(mean=("auc_difference", "mean"),
              sample_std=("auc_difference", "std"), seeds_positive=("auc_difference", lambda x: int((x > 0).sum()))).reset_index()
    summary.to_csv(run / "feature_conditioning_summary.csv", index=False)
    write_json(run / "analysis_verification.json", {
        "complete": True, "new_runs": 6, "new_epochs": epochs,
        "new_model_cuda_replay_frames": frames, "new_model_cuda_replay_max_score_difference": 0.0,
        "auc_recomputed_from_exported_scores": True, "auc_only_histories": True,
        "old_results_and_models_unchanged": True, "base_b_retrained": False,
        "models": verified, "analysis_code_sha256": digest(Path(__file__)),
        "extra_analysis": "Post-hoc feature x GAN contrasts with AQT fixed on, reusing original A/B/Full and new AQT_only. Not a complete three-factor factorial or a causal mechanism proof.",
        "feature_contrasts": summary.to_dict("records"), "independent_test": False,
    })
    print("DIAGNOSTIC VERIFICATION PASSED: 360 epochs, 63552 exact CUDA replay predictions; original Base/B unchanged.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/ablation_auc_09117_diagnostic_v1")
    analyse(parser.parse_args().run.resolve())
