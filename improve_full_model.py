from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

import fgaae
import full_model
import summarize_full_model
import calibrate_full_model
from audit_data import full_digest


ROOT = Path(__file__).resolve().parent
PILOTS = [(0.003, 0.1), (0.01, 0.1), (0.03, 0.1)]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def training_arguments(plan, output, profile, **overrides):
    arguments = full_model.build_parser().parse_args(["train"])
    for key, value in plan.items():
        if hasattr(arguments, key):
            setattr(arguments, key, value)
    arguments.output = str(output)
    arguments.training_profile = profile
    for key, value in overrides.items():
        setattr(arguments, key, value)
    return arguments


def run_round(stable, output):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    plan = read_json(stable / "registered_plan.json")
    stable_report = read_json(stable / "full_model_report.json")
    baseline = Path(plan["pretrained_run"])
    baseline_report = read_json(baseline / "full_model_report.json")
    if stable_report["seeds"] != baseline_report["seeds"]:
        raise ValueError("Paired comparisons require identical seeds.")
    for name in ["training_membership.csv", "development_membership.csv"]:
        left = pd.read_csv(baseline / "inputs" / name)
        right = pd.read_csv(stable / "inputs" / name)
        if not np.array_equal(left[["source_row", "label"]], right[["source_row", "label"]]):
            raise ValueError("Data membership changed between full-model runs.")
    output.mkdir(parents=True, exist_ok=False)
    pilot_seed = plan["seeds"][len(plan["seeds"]) // 2]
    fgaae.write_json(output / "registered_plan.json", {
        "scope": "Complete FGAAE hyperparameter tuning; all modules remain enabled, no ablation study.",
        "baseline": str(baseline), "stabilized": str(stable), "seeds": plan["seeds"],
        "pilot_seed": pilot_seed, "pilot_loss_weights_adversarial_distillation": PILOTS,
        "pilot_epochs": 12, "pilot_aqt_rounds": 1,
        "pilot_trigger": "Stabilized three-seed mean AUC below 0.90.",
        "full_verification_gate": "Best pilot AUC exceeds the corresponding stabilized seed's full-run selected AUC by at least 0.002; run only the best passing weights for all seeds and 60 epochs.",
        "pilot_rationale": "Weighted distillation gradients exceeded reconstruction on monitored batches; probe lower distillation weight and three positive adversarial weights.",
        "score_calibration": "Fixed positive coefficient grid in calibrate_full_model.py; reference scales from training, coefficient selection on earlier development files.",
        "test_access": False, "development_tuning_is_adaptive": True,
        "code_sha256": full_digest(Path(__file__)), "calibration_code_sha256": full_digest(ROOT / "calibrate_full_model.py"),
    })
    summaries = [("v4_registered", baseline, baseline_report), ("v5_stabilized", stable, stable_report)]
    selected_run = stable if stable_report["development_auc_mean"] >= baseline_report["development_auc_mean"] else baseline
    pilot_rows = []
    if stable_report["development_auc_mean"] < 0.90:
        corresponding = next(row for row in stable_report["runs"] if row["seed"] == pilot_seed)
        threshold = corresponding["roc_auc"] + 0.002
        for index, (adversarial, distillation) in enumerate(PILOTS, 1):
            pilot = output / f"pilot_{index}"
            arguments = training_arguments(plan, pilot, f"complete_fgaae_loss_pilot_{index}",
                seeds=[pilot_seed], iterations=1, epochs_per_iteration=12,
                adversarial_weight=adversarial, distillation_weight=distillation)
            full_model.run(arguments)
            summarize_full_model.summarize(pilot)
            result = read_json(pilot / "full_model_report.json")
            pilot_rows.append({
                "pilot": index, "seed": pilot_seed, "adversarial_weight": adversarial,
                "distillation_weight": distillation, "development_auc": result["development_auc_mean"],
                "verification_gate_auc": threshold, "passes_gate": result["development_auc_mean"] > threshold,
            })
            pd.DataFrame(pilot_rows).to_csv(output / "loss_weight_pilots.csv", index=False)
        best = max(pilot_rows, key=lambda row: row["development_auc"])
        if best["passes_gate"]:
            verified = output / "full_loss_verification"
            arguments = training_arguments(plan, verified, "complete_fgaae_loss_verified",
                adversarial_weight=best["adversarial_weight"], distillation_weight=best["distillation_weight"])
            full_model.run(arguments)
            summarize_full_model.summarize(verified)
            report = read_json(verified / "full_model_report.json")
            summaries.append(("loss_weights_verified", verified, report))
            if report["development_auc_mean"] > read_json(selected_run / "full_model_report.json")["development_auc_mean"]:
                selected_run = verified
    calibrate_full_model.calibrate(selected_run, plan["device"])
    calibration = read_json(selected_run / "score_calibration/report.json")
    table = []
    for name, directory, report in summaries:
        diagnostic = pd.read_csv(directory / "full_model_diagnostics.csv")
        table.append({
            "variant": name, "auc_mean": report["development_auc_mean"],
            "auc_sample_std": report["development_auc_sample_std"], "f1_mean": report["development_f1_mean"],
            "last_epoch_auc_mean": float(diagnostic.last_epoch_auc.mean()),
            "last_teacher_mse_mean": float(diagnostic.last_epoch_teacher_validation_mse.mean()),
            "seeds": len(report["seeds"]), "directory": str(directory),
        })
    pd.DataFrame(table).to_csv(output / "training_comparison.csv", index=False)
    use_calibration = calibration["calibration_accepted"]
    selected_report = read_json(selected_run / "full_model_report.json")
    final_auc = calibration["selected"]["full_development_auc_mean"] if use_calibration else selected_report["development_auc_mean"]
    final_std = calibration["selected"]["full_development_auc_sample_std"] if use_calibration else selected_report["development_auc_sample_std"]
    final_f1 = calibration["selected"]["full_development_f1_mean"] if use_calibration else selected_report["development_f1_mean"]
    report = {
        "training_variants": table, "loss_weight_pilots": pilot_rows,
        "selected_training_run": str(selected_run), "score_calibration_accepted": use_calibration,
        "development_auc_mean": final_auc, "development_auc_sample_std": final_std,
        "development_f1_mean": final_f1, "auc_change_from_v4": final_auc - baseline_report["development_auc_mean"],
        "numerically_near_0_91_on_synthetic_development": bool(final_auc >= 0.90),
        "independent_test": False, "paper_performance_reproduced": False,
        "calibration": calibration,
    }
    fgaae.write_json(output / "round_report.json", report)
    plot_comparison(output, baseline, selected_run, plan["seeds"], use_calibration)
    lines = [
        "# 完整 FGAAE 性能提升试验", "",
        f"本轮最终开发 AUC **{final_auc:.6f} ± {final_std:.6f}**，F1 均值 **{final_f1:.6f}**，较 v4 的 AUC 均值变化 **{report['auc_change_from_v4']:+.6f}**。", "",
        "所有方案都保留完整 FGAAE。数据、原始异常注入、训练成员和三种子保持一致。本轮为自适应开发调参，未使用新的独立测试，因此数值接近 0.91 也不能视为论文原始实验已经复现。", "",
        "| 完整训练方案 | 种子数 | 开发 AUC 均值 | 样本标准差 | 开发 F1 均值 | 末轮 AUC 均值 | 末轮教师 MSE |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in table:
        lines.append(f"| {row['variant']} | {row['seeds']} | {row['auc_mean']:.6f} | {row['auc_sample_std']:.6f} | {row['f1_mean']:.6f} | {row['last_epoch_auc_mean']:.6f} | {row['last_teacher_mse_mean']:.6f} |")
    lines.extend(["", "## 损失权重短程试验", "", "短程试验固定使用中间位置的种子，每组 12 轮、1 次 AQT，旨在筛选联合损失权重。没有移除任何模型组件，不能将单种子试验当作三种子的稳定提升。", ""])
    for row in pilot_rows:
        lines.append(f"- adversarial={row['adversarial_weight']:g}, distillation={row['distillation_weight']:g}：开发 AUC {row['development_auc']:.6f}，通过三种子完整验证门槛：{row['passes_gate']}。")
    if not pilot_rows:
        lines.append("稳定训练已达到预定开发均值门槛，没有继续扩展损失权重搜索。")
    lines.extend([
        "", "## 评分校准", "",
        f"是否接受训练参考尺度校准后的正号 Critic 系数：**{use_calibration}**；选中候选为 **{calibration['selected_shared_candidate']}**。系数使用较早开发文件选择、较晚开发文件检查；较晚文件曾参与检查点选择，仍不是独立测试。",
        "", "校准后的 F1 阈值只由较早开发文件确定，原始完整训练报告的阈值由全部开发集确定，二者阈值协议不同，F1 数值应结合这一差异解读。",
        "", f"选中的完整训练目录：`{selected_run}`。" + ("校准模型位于其 score_calibration/seed_种子_model.pt。" if use_calibration else "使用各 seed_种子/model.pt。"),
        "", "训练评分保持论文公式字面正号；本轮校准只调整论文未明确的系数。25 维特征和原始异常协议仍未完全恢复，注入功率比也不等同于论文的信道 SNR。",
        "", "数据和检查点核验见各训练目录 verification.json；全部候选成绩见 loss_weight_pilots.csv 和 score_calibration/all_candidates.csv。曲线见 comparison.png，弱注入分组结果见 comparison_by_injection_ratio.csv。",
    ])
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def plot_comparison(output, baseline, selected, seeds, calibrated):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metadata = pd.read_csv(selected / "inputs/development_membership.csv")
    x = np.linspace(0, 1, 501)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    power_rows = []
    for name, directory, calibration in [("v4", baseline, False), ("Selected full FGAAE", selected, calibrated)]:
        curves = []
        for seed in seeds:
            path = directory / "score_calibration" / f"seed_{seed}_development_scores.csv" if calibration else directory / f"seed_{seed}/development_scores.csv"
            scores = pd.read_csv(path, float_precision="round_trip")
            if not np.array_equal(scores.source_row, metadata.source_row):
                raise ValueError("Cannot compare different development members.")
            labels = scores.label.to_numpy()
            values = scores.anomaly_score.to_numpy()
            fpr, tpr, _ = roc_curve(labels, values)
            curves.append(np.interp(x, fpr, tpr))
            for ratio in sorted(metadata.loc[metadata.label == 1, "nominal_interference_ratio_db"].unique()):
                mask = (labels == 0) | (metadata.nominal_interference_ratio_db.to_numpy() == ratio)
                power_rows.append({"variant": name, "seed": seed, "nominal_interference_ratio_db": ratio, "roc_auc": roc_auc_score(labels[mask], values[mask])})
        curves = np.asarray(curves)
        axes[0].plot(np.r_[0, x], np.r_[0, curves.mean(axis=0)], label=name)
        axes[0].fill_between(np.r_[0, x], np.r_[0, np.maximum(0, curves.mean(axis=0) - curves.std(axis=0))], np.r_[0, np.minimum(1, curves.mean(axis=0) + curves.std(axis=0))], alpha=0.15)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Development ROC: mean and seed variation", xlim=(0, 1), ylim=(0, 1))
    power = pd.DataFrame(power_rows)
    power.to_csv(output / "comparison_by_injection_ratio.csv", index=False)
    for name, group in power.groupby("variant", sort=False):
        summary = group.groupby("nominal_interference_ratio_db").roc_auc.agg(["mean", "std"])
        axes[1].errorbar(summary.index, summary["mean"], yerr=summary["std"].fillna(0), label=name, marker="o", capsize=2)
    axes[1].set(xlabel="Injected/background power ratio (dB; not channel SNR)", ylabel="ROC-AUC mean +/- sample SD", title="Same frozen synthetic development data")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Try prioritized full-FGAAE optimization with bounded loss tuning and grouped score calibration.")
    parser.add_argument("--stable-run", type=Path, default=ROOT / "runs/full_fgaae_v5_stabilized")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/full_fgaae_improvement_round1")
    arguments = parser.parse_args()
    run_round(arguments.stable_run.resolve(), arguments.output.resolve())
