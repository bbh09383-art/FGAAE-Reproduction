from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import fgaae
from audit_data import full_digest
from full_model import normalized


ROOT = Path(__file__).resolve().parent
CANDIDATES = [None, 0.01, 0.03, 0.1, 0.3, 1.0]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def coefficient_subsets(metadata):
    files = metadata.groupby("source_file").ticks.min().sort_values().index.tolist()
    if len(files) < 2:
        raise ValueError("Coefficient tuning requires at least two development source files.")
    selection_files = files[:len(files) // 2]
    selection = metadata.source_file.isin(selection_files).to_numpy()
    for mask in [selection, ~selection]:
        if metadata.loc[mask, "label"].nunique() != 2:
            raise ValueError("Both development subsets must contain both classes.")
    return selection, selection_files, files[len(files) // 2:]


def candidate_metrics(seed, metadata, reconstruction, critic, scales):
    selection, _, _ = coefficient_subsets(metadata)
    labels = metadata.label.to_numpy(dtype=np.int64)
    rows = []
    for relative_weight in CANDIDATES:
        name = "registered_0.1" if relative_weight is None else f"relative_{relative_weight:g}"
        weight = 0.1 if relative_weight is None else relative_weight * scales["reconstruction_std"] / scales["critic_std"]
        scores = reconstruction + weight * critic
        threshold = fgaae.select_threshold(scores[selection], labels[selection], {"threshold_method": "validation-f1"})
        row = {"seed": seed, "candidate": name, "critic_weight": weight, "threshold_from_selection_files": threshold}
        for prefix, mask in [("selection", selection), ("coefficient_check", ~selection), ("full_development", np.ones(len(labels), dtype=bool))]:
            measurement = fgaae.metrics(labels[mask], scores[mask], threshold)
            row.update({f"{prefix}_{key}": measurement[key] for key in ["roc_auc", "f1", "average_precision"]})
        rows.append(row)
    return rows


def choose_candidate(frame):
    aggregate = frame.groupby("candidate", sort=False).agg(
        selection_auc_mean=("selection_roc_auc", "mean"),
        coefficient_check_auc_mean=("coefficient_check_roc_auc", "mean"),
        full_development_auc_mean=("full_development_roc_auc", "mean"),
        full_development_auc_sample_std=("full_development_roc_auc", "std"),
        full_development_f1_mean=("full_development_f1", "mean"),
    ).fillna(0)
    best = aggregate.selection_auc_mean.idxmax()
    if aggregate.loc[best, "selection_auc_mean"] <= aggregate.loc["registered_0.1", "selection_auc_mean"] + 1e-4:
        best = "registered_0.1"
    return best, aggregate


def calibrate(run, device_name):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    plan = read_json(run / "registered_plan.json")
    manifest = read_json(run / "inputs/manifest.json")
    metadata = pd.read_csv(run / "inputs/development_membership.csv")
    selection, selection_files, check_files = coefficient_subsets(metadata)
    output = run / "score_calibration"
    output.mkdir(exist_ok=False)
    policy = {
        "scope": "Positive critic coefficient calibration only; complete FGAAE weights and literal Equation-11 sign are preserved.",
        "candidates_relative_to_reference_component_std": CANDIDATES,
        "reference_rows": manifest["reference_training_rows"],
        "selection_files": selection_files, "coefficient_check_files": check_files,
        "selection_rule": "One shared relative coefficient across all seeds, selected by mean AUC on earlier development files; ties within 1e-4 retain registered 0.1.",
        "threshold_rule": "Per-seed F1 threshold from earlier development files only.",
        "evaluation_caveat": "Later development files were previously used for checkpoint selection. They are a coefficient check, not an independent or untouched test.",
        "confirmation_or_test_accessed": False, "code_sha256": full_digest(Path(__file__)),
    }
    fgaae.write_json(output / "registered_plan.json", policy)
    dataset = Path(plan["dataset"]) / "spectra.npy"
    if full_digest(dataset) != manifest["dataset_sha256"]:
        raise ValueError("Reference data changed.")
    raw = np.load(dataset, mmap_mode="r", allow_pickle=False)[manifest["reference_training_rows"]].astype(np.float32)
    fgaae.seed_everything(plan["seeds"][0], 2)
    device = torch.device(device_name)
    rows, scale_records = [], []
    for seed in plan["seeds"]:
        directory = run / f"seed_{seed}"
        checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        if not checkpoint["all_components_enabled"] or checkpoint["config"]["critic_score_sign"] != 1 or checkpoint["config"].get("reconstruction_score_reduction") != "sum" or checkpoint["config"].get("standardize_scores", False):
            raise ValueError("Expected complete FGAAE with the registered positive critic sign.")
        model = fgaae.FGAAE(checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
        result = fgaae.evaluate(model, normalized(raw, checkpoint["normalization"]), checkpoint["config"], device)
        scales = {
            "reconstruction_std": float(result["reconstruction_score"].std()),
            "critic_std": float(result["critic_value"].std()),
        }
        if min(scales.values()) <= 1e-8:
            raise ValueError("Reference score components have insufficient variance for calibration.")
        scale_records.append({"seed": seed, **scales, "model_sha256": full_digest(directory / "model.pt")})
        components = pd.read_csv(directory / "score_components.csv", float_precision="round_trip")
        if not np.array_equal(components.source_row, metadata.source_row):
            raise ValueError("Development score membership mismatch.")
        rows.extend(candidate_metrics(seed, metadata, components.sum_squared_error.to_numpy(), components.critic.to_numpy(), scales))
        del model, checkpoint
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "all_candidates.csv", index=False)
    pd.DataFrame(scale_records).to_csv(output / "training_reference_scales.csv", index=False)
    best, aggregate = choose_candidate(frame)
    aggregate.to_csv(output / "candidate_summary.csv")
    selected = aggregate.loc[best]
    original = aggregate.loc["registered_0.1"]
    check_improved = bool(selected.coefficient_check_auc_mean > original.coefficient_check_auc_mean)
    full_improved = bool(selected.full_development_auc_mean > original.full_development_auc_mean)
    accepted = bool(best != "registered_0.1" and check_improved and full_improved)
    checkpoint_checks = []
    for seed in plan["seeds"]:
        directory = run / f"seed_{seed}"
        row = frame[(frame.seed == seed) & (frame.candidate == best)].iloc[0]
        checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        checkpoint["config"] = {**checkpoint["config"], "critic_score_weight": float(row.critic_weight), "score_calibration": None}
        checkpoint["threshold"] = float(row.threshold_from_selection_files)
        checkpoint["artifact_type"] = "complete_fgaae_development_calibrated"
        checkpoint["primary_score"] = f"sum_squared_error + {float(row.critic_weight):.17g} * critic(reconstruction)"
        checkpoint["calibration"] = {"candidate": best, "calibration_accepted": accepted, "independent_test": False}
        torch.save(checkpoint, output / f"seed_{seed}_model.pt")
        reloaded = torch.load(output / f"seed_{seed}_model.pt", map_location="cpu", weights_only=True)
        if not all(torch.equal(value, reloaded["model_state"][name]) for name, value in checkpoint["model_state"].items()):
            raise RuntimeError("Calibrated checkpoint changed model tensors.")
        if reloaded["config"]["critic_score_weight"] != float(row.critic_weight):
            raise RuntimeError("Calibrated coefficient did not survive checkpoint serialization.")
        components = pd.read_csv(directory / "score_components.csv", float_precision="round_trip")
        scores = components.sum_squared_error.to_numpy() + float(row.critic_weight) * components.critic.to_numpy()
        fgaae.save_predictions(output / f"seed_{seed}_development_scores.csv", metadata.source_row.to_numpy(), metadata.label.to_numpy(), None, {
            "scores": scores, "reconstruction_error": components.mean_squared_error.to_numpy(), "critic_value": components.critic.to_numpy(),
        }, checkpoint["threshold"])
        checkpoint_checks.append({"seed": seed, "model_tensors_bitwise_preserved": True, "coefficient_reload_verified": True, "coefficient": float(row.critic_weight)})
    summary = {
        "selected_shared_candidate": best, "coefficient_check_improved": check_improved,
        "full_development_improved": full_improved, "calibration_accepted": accepted,
        "registered": original.to_dict(), "selected": selected.to_dict(),
        "full_development_metrics_are_tuned": True, "independent_test": False,
        "confirmation_or_test_accessed": False, "source_models_modified": False,
    }
    fgaae.write_json(output / "report.json", summary)
    fgaae.write_json(output / "checkpoint_verification.json", checkpoint_checks)
    lines = [
        "# 完整 FGAAE 评分系数校准", "",
        "保持完整模型及论文公式的正号，只调整 Critic 项系数。尺度由特征训练参考帧拟合，所有种子选择同一相对系数。原始主实验与检查点保持可追溯。", "",
        f"选择系数使用 {int(selection.sum())} 帧、{len(selection_files)} 个较早开发文件；另 {int((~selection).sum())} 帧、{len(check_files)} 个较晚开发文件仅用于系数检查。所有开发文件都曾参与模型检查点选择，因此这里没有新的独立测试。", "",
        f"选中方案：**{best}**；较晚开发文件的平均 AUC 是否提高：**{check_improved}**；是否接受校准：**{accepted}**。", "",
        "| 方案 | 系数选择子集 AUC | 系数检查子集 AUC | 全开发集 AUC | 全开发集 F1 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, values in aggregate.iterrows():
        lines.append(f"| {name} | {values.selection_auc_mean:.6f} | {values.coefficient_check_auc_mean:.6f} | {values.full_development_auc_mean:.6f} | {values.full_development_f1_mean:.6f} |")
    lines.extend(["", "全开发集指标受系数选择影响，仅作为开发调参成绩。F1 阈值来自较早文件，因此 registered_0.1 的 F1 与使用全部开发标签选阈值的主报告可能不同。", "", "新增校准检查点保存于本目录；若系数检查没有改善，不将其作为性能提升。"])
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def smoke_test():
    generator = np.random.default_rng(7)
    metadata = pd.DataFrame({
        "source_file": np.repeat(["a", "b", "c", "d"], 32),
        "ticks": np.arange(128), "label": np.tile([0, 0, 0, 1], 32),
    })
    labels = metadata.label.to_numpy()
    reconstruction = generator.normal(size=128) + labels * 0.5
    critic = generator.normal(size=128) + labels
    scales = {"reconstruction_std": 1.0, "critic_std": 0.5}
    frame = pd.DataFrame(candidate_metrics(1, metadata, reconstruction, critic, scales))
    chosen, _ = choose_candidate(frame)
    selection, _, _ = coefficient_subsets(metadata)
    changed = metadata.copy()
    changed.loc[~selection, "label"] = 1 - changed.loc[~selection, "label"]
    changed_frame = pd.DataFrame(candidate_metrics(1, changed, reconstruction, critic, scales))
    assert choose_candidate(changed_frame)[0] == chosen
    assert np.array_equal(frame.selection_roc_auc, changed_frame.selection_roc_auc)
    assert frame.critic_weight.gt(0).all()
    print("CALIBRATION SMOKE TEST PASSED: positive coefficients, file grouping and later labels cannot affect coefficient selection.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate positive FGAAE score weights on explicitly grouped development subsets.")
    parser.add_argument("--run", type=Path, default=ROOT / "runs/full_fgaae_v5_stabilized")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.smoke_test:
        smoke_test()
    else:
        calibrate(arguments.run.resolve(), arguments.device)
