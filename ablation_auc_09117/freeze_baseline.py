"""Freeze and verify the historical FGAAE baseline without changing training files."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from xml.etree import ElementTree as ET
from zipfile import ZipFile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUN = ROOT / "runs/full_fgaae_improvement_round1/full_loss_verification"
FROZEN = HERE / "frozen"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def document_text(path):
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(path) as archive:
        body = ET.fromstring(archive.read("word/document.xml")).find("w:body", namespace)
    lines = []
    for child in body:
        if child.tag.endswith("}p"):
            lines.append("".join(node.text or "" for node in child.findall(".//w:t", namespace)))
        elif child.tag.endswith("}tbl"):
            for row in child.findall("w:tr", namespace):
                lines.append(" | ".join(" ".join(node.text or "" for node in cell.findall(".//w:t", namespace)) for cell in row.findall("w:tc", namespace)))
    return "\n".join(lines) + "\n"


def freeze(document, device_name):
    if FROZEN.exists():
        raise FileExistsError("Baseline is already frozen; use the verify command.")
    plan = read_json(RUN / "registered_plan.json")
    inputs = read_json(RUN / "inputs/manifest.json")
    calibration = read_json(RUN / "score_calibration/registered_plan.json")
    calibration_report = read_json(RUN / "score_calibration/report.json")
    sources = {
        "fgaae.py": RUN / "source/fgaae.py",
        "full_model.py": RUN / "source/full_model.py",
        "calibrate_full_model.py": RUN.parent / "source/calibrate_full_model.py",
    }
    expected = {"fgaae.py": plan["core_code_sha256"], "full_model.py": plan["full_model_code_sha256"], "calibrate_full_model.py": calibration["code_sha256"]}
    for name, path in sources.items():
        if digest(path) != expected[name]:
            raise ValueError(f"Historical source hash mismatch: {name}")
    spec = importlib.util.spec_from_file_location("fgaae_baseline_09117", sources["fgaae.py"])
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    core.seed_everything(2025, 2)
    device = torch.device(device_name)
    data_path = Path(plan["station_run"]) / "inputs/development.npy"
    dataset = Path(plan["dataset"])
    benchmark = Path(plan["benchmark"])
    for path, expected_digest in [(dataset / "spectra.npy", inputs["dataset_sha256"]), (benchmark / "train.npy", inputs["frozen_training_sha256"]), (data_path, inputs["development_sha256"])]:
        if digest(path) != expected_digest:
            raise ValueError(f"Historical data hash mismatch: {path}")
    raw = np.load(data_path, mmap_mode="r", allow_pickle=False)
    metadata = pd.read_csv(RUN / "inputs/development_membership.csv")
    rows, artifacts = [], list(sources.values())
    artifacts += [Path(__file__), RUN / "registered_plan.json", RUN / "inputs/manifest.json", RUN / "inputs/training_membership.csv", RUN / "inputs/development_membership.csv", dataset / "dataset.json", dataset / "spectra.npy", dataset / "partitions.csv", benchmark / "protocol.json", benchmark / "train.npy", benchmark / "train_metadata.csv", data_path, data_path.with_name("development_metadata.csv")]
    artifacts += [RUN / "score_calibration" / name for name in ["registered_plan.json", "report.json", "training_reference_scales.csv", "all_candidates.csv", "candidate_summary.csv"]]
    for seed in plan["seeds"]:
        directory = RUN / f"seed_{seed}"
        model_path = RUN / f"score_calibration/seed_{seed}_model.pt"
        scores_path = RUN / f"score_calibration/seed_{seed}_development_scores.csv"
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        trained = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        if not checkpoint["all_components_enabled"] or config["critic_score_sign"] != 1 or config["reconstruction_score_reduction"] != "sum" or config.get("score_calibration") or config.get("standardize_scores", False) or config["critic_score_weight"] <= 0:
            raise ValueError("Expected the historical complete FGAAE with a positive original-form score.")
        if not all(torch.equal(value, trained["model_state"][name]) for name, value in checkpoint["model_state"].items()):
            raise ValueError("Calibrated checkpoint weights differ from the original trained model.")
        model = core.FGAAE(config).to(device)
        model.load_state_dict(checkpoint["model_state"])
        parameters = checkpoint["normalization"]
        values = ((raw.astype(np.float64) - parameters["mean"]) / parameters["std"]).astype(np.float32)
        result = core.evaluate(model, values, config, device)
        saved = pd.read_csv(scores_path, float_precision="round_trip")
        if not np.array_equal(saved.source_row, metadata.source_row) or not np.array_equal(saved.label, metadata.label):
            raise ValueError("Historical score membership mismatch.")
        saved_auc = float(roc_auc_score(metadata.label, saved.anomaly_score))
        replay_auc = float(roc_auc_score(metadata.label, result["scores"]))
        maximum_difference = float(np.abs(saved.anomaly_score.to_numpy() - result["scores"]).max())
        if device.type == "cuda" and maximum_difference > 1e-8:
            raise ValueError("Historical CUDA predictions do not replay exactly.")
        if abs(saved_auc - replay_auc) > (1e-12 if device.type == "cuda" else 1e-4):
            raise ValueError("Historical baseline AUC changed.")
        selection = read_json(directory / "selection.json")
        history = pd.read_csv(directory / "stage2_history.csv")
        rows.append({"seed": seed, "auc_primary": saved_auc, "auc_reconstruction_same_checkpoint": float(roc_auc_score(metadata.label, result["reconstruction_score"])), "best_epoch": selection["best_epoch"], "critic_weight": config["critic_score_weight"], "generator_updates_completed": int(np.ceil(history.retained_samples / config["batch_size"]).sum()), "replay_frames": len(raw), "replay_device": str(device), "replay_max_score_difference": maximum_difference, "replay_auc": replay_auc})
        artifacts += [model_path, scores_path, *[directory / name for name in ["model.pt", "stage1_best.pt", "stage1_history.csv", "stage2_best.pt", "stage2_latest.pt", "stage2_history.csv", "config.json", "report.json", "selection.json", "score_components.csv", "component_audit.json"]]]
        del checkpoint, trained, model, values
    table = pd.DataFrame(rows)
    auc_mean = float(table.auc_primary.mean())
    auc_std = float(table.auc_primary.std(ddof=1))
    if abs(auc_mean - 0.9117440964193119) > 1e-12 or abs(auc_std - calibration_report["selected"]["full_development_auc_sample_std"]) > 1e-12:
        raise ValueError("The supplied artifacts do not represent the requested 0.9117 baseline.")
    FROZEN.mkdir()
    (FROZEN / "source").mkdir()
    for name, path in sources.items():
        shutil.copy2(path, FROZEN / "source" / name)
    shutil.copy2(document, FROZEN / "source_plan.docx")
    (FROZEN / "source_plan.txt").write_text(document_text(document), encoding="utf-8")
    table.to_csv(FROZEN / "baseline_auc.csv", index=False)
    artifacts += list((FROZEN / "source").iterdir()) + [FROZEN / "source_plan.docx", FROZEN / "source_plan.txt", FROZEN / "baseline_auc.csv"]
    manifest = {
        "baseline_run": str(RUN.relative_to(ROOT)), "auc_mean": auc_mean, "auc_sample_std": auc_std,
        "seeds": plan["seeds"], "training_rows": inputs["training_rows"], "development_rows": len(metadata),
        "normalization": inputs["normalization"], "reference_training_rows": inputs["reference_training_rows"],
        "reference_validation_rows": inputs["reference_validation_rows"], "historical_training_plan": plan,
        "checkpoint_selection": "All eight development files, maximum ROC-AUC at original lambda=0.1; archived training implementation.",
        "coefficient_selection": calibration, "reported_metrics": ["ROC-AUC"],
        "document_source": str(document), "historical_model_tensors_unchanged": True,
        "independent_test": False, "ablation_training_started": False,
        "artifacts_sha256": {path.resolve().relative_to(ROOT.resolve()).as_posix(): digest(path) for path in artifacts},
    }
    write_json(FROZEN / "baseline_manifest.json", manifest)
    print(table.to_string(index=False))
    print(f"FROZEN: AUC={auc_mean:.12f} +/- {auc_std:.12f}; no ablation training executed.", flush=True)


def verify():
    manifest = read_json(FROZEN / "baseline_manifest.json")
    for name, expected in manifest["artifacts_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen baseline artifact changed: {name}")
    print(f"VERIFIED: {len(manifest['artifacts_sha256'])} artifact hashes; AUC={manifest['auc_mean']:.12f}; frozen baseline unchanged.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "verify"])
    parser.add_argument("--document", type=Path, default=Path.home() / "Desktop/1023汇报(1).docx")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()
    if arguments.command == "freeze":
        freeze(arguments.document.resolve(), arguments.device)
    else:
        verify()
