from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

import fgaae
from audit_data import full_digest
from full_model import normalized, teacher_metrics


ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(run):
    plan = read_json(run / "registered_plan.json")
    summary = read_json(run / "full_model_report.json")
    manifest = read_json(run / "inputs/manifest.json")
    training = pd.read_csv(run / "inputs/training_membership.csv")
    development = pd.read_csv(run / "inputs/development_membership.csv")
    references = set(manifest["reference_training_files"] + manifest["reference_validation_files"])
    assert references.isdisjoint(set(training.source_file))
    assert references.isdisjoint(set(development.source_file))
    assert set(training.source_file).isdisjoint(set(development.source_file))
    assert set(training.source_row).isdisjoint(set(development.source_row))
    assert not training.source_row.duplicated().any()
    assert not development.source_row.duplicated().any()
    source = run / "source" if (run / "source").exists() else ROOT
    assert full_digest(source / "fgaae.py") == plan["core_code_sha256"]
    assert full_digest(source / "full_model.py") == plan["full_model_code_sha256"]
    assert full_digest(Path(plan["dataset"]) / "spectra.npy") == manifest["dataset_sha256"]
    assert full_digest(Path(plan["benchmark"]) / "train.npy") == manifest["frozen_training_sha256"]
    development_path = Path(plan["station_run"]) / "inputs/development.npy"
    assert full_digest(development_path) == manifest["development_sha256"]
    raw_development = np.load(development_path, mmap_mode="r", allow_pickle=False)
    raw_reference_validation = np.load(Path(plan["dataset"]) / "spectra.npy", mmap_mode="r", allow_pickle=False)[manifest["reference_validation_rows"]].astype(np.float32)
    assert len(raw_development) == len(development)
    fgaae.seed_everything(plan["seeds"][0], 2)
    rows, grouped, curves, verification = [], [], [], []
    for seed in plan["seeds"]:
        directory = run / f"seed_{seed}"
        report = read_json(directory / "report.json")
        audit = read_json(directory / "component_audit.json")
        health = read_json(directory / "teacher_health.json")
        history = pd.read_csv(directory / "stage2_history.csv")
        pretraining = pd.read_csv(directory / "stage1_history.csv")
        assert len(pretraining) == plan["stage1_epochs"]
        assert len(history) == plan["iterations"] * plan["epochs_per_iteration"]
        assert history.iteration.nunique() == plan["iterations"]
        assert len(list(directory.glob("aqt_iteration_*.csv"))) == plan["iterations"]
        assert audit["teacher_frozen"] and audit["all_components_enabled"]
        changes = audit["parameter_changes_stage2_selected_model"]
        assert changes["teacher"] == 0
        assert all(changes[name] > 0 for name in ["encoder", "decoder", "student", "critic"])
        if plan.get("freeze_encoder_bn", False):
            assert audit["encoder_bn_statistics_frozen"]
            assert all(value == 0 for value in audit["encoder_bn_buffer_changes_selected"].values())
            assert all(value == 0 for value in audit["encoder_bn_buffer_changes_latest"].values())
            assert history.encoder_learning_rate.eq(plan["encoder_lr"]).all()
        assert np.isfinite(history.select_dtypes(include="number").to_numpy()).all()
        scores = pd.read_csv(directory / "development_scores.csv", float_precision="round_trip")
        assert np.array_equal(scores.source_row, development.source_row)
        assert np.array_equal(scores.label, development.label)
        components = pd.read_csv(directory / "score_components.csv", float_precision="round_trip")
        assert np.allclose(components.sum_squared_error, 840 * components.mean_squared_error)
        checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        critic_weight = checkpoint["config"]["critic_score_weight"]
        assert critic_weight > 0
        assert np.allclose(scores.anomaly_score, components.sum_squared_error + critic_weight * components.critic)
        assert checkpoint["all_components_enabled"]
        configuration = {**checkpoint["config"], "batch_size": 64}
        assert configuration["use_aqt"]
        assert configuration["reconstruction_score_reduction"] == "sum"
        assert configuration["critic_score_sign"] == 1
        model = fgaae.FGAAE(configuration)
        model.load_state_dict(checkpoint["model_state"])
        count = min(256, len(development))
        values = normalized(raw_development[:count], checkpoint["normalization"])
        reloaded = fgaae.evaluate(model, values, configuration, torch.device("cpu"))
        difference = float(np.max(np.abs(reloaded["scores"] - scores.anomaly_score.to_numpy()[:count])))
        cpu_default_precision_close = bool(np.allclose(reloaded["scores"], scores.anomaly_score.to_numpy()[:count], rtol=5e-4, atol=1e-3))
        relative_difference = float(np.max(np.abs(reloaded["scores"] - scores.anomaly_score.to_numpy()[:count]) / np.maximum(np.abs(scores.anomaly_score.to_numpy()[:count]), 1)))
        cuda_difference = None
        fp32_cpu_gpu_difference = None
        if torch.cuda.is_available() and plan["device"].startswith("cuda"):
            model.to("cuda")
            gpu_configuration = {**configuration, "batch_size": checkpoint["config"]["batch_size"]}
            reloaded_cuda = fgaae.evaluate(model, values, gpu_configuration, torch.device("cuda"))
            cuda_difference = float(np.max(np.abs(reloaded_cuda["scores"] - scores.anomaly_score.to_numpy()[:count])))
            assert np.allclose(reloaded_cuda["scores"], scores.anomaly_score.to_numpy()[:count], rtol=1e-5, atol=1e-3)
            if not cpu_default_precision_close:
                # Compare CPU FP32 to CUDA FP32 explicitly. Registered CUDA TF32
                # predictions above must still replay at their original precision.
                with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=False):
                    reloaded_fp32 = fgaae.evaluate(model, values, configuration, torch.device("cuda"))
                fp32_cpu_gpu_difference = float(np.max(np.abs(reloaded_fp32["scores"] - reloaded["scores"])))
                assert np.allclose(reloaded_fp32["scores"], reloaded["scores"], rtol=5e-4, atol=1e-3)
            model.to("cpu")
        else:
            assert cpu_default_precision_close
        measurement = fgaae.metrics(scores.label.to_numpy(), scores.anomaly_score.to_numpy(), checkpoint["threshold"])
        assert abs(measurement["roc_auc"] - report["development"]["roc_auc"]) < 1e-12
        assert abs(measurement["f1"] - report["development"]["f1"]) < 1e-12
        targets = (fgaae.expert_features(raw_reference_validation, configuration).astype(np.float64) - np.asarray(checkpoint["feature_mean"])) / np.asarray(checkpoint["feature_std"])
        reference_loader = fgaae.make_loader(normalized(raw_reference_validation, checkpoint["normalization"]), 64, targets)
        selected_feature_frame, selected_feature_health = teacher_metrics(model, reference_loader, fgaae.FEATURE_NAMES, torch.device("cpu"))
        selected_feature_frame.to_csv(directory / "teacher_alignment_selected.csv", index=False)
        latest = torch.load(directory / "stage2_latest.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(latest["model_state"])
        latest_feature_frame, latest_feature_health = teacher_metrics(model, reference_loader, fgaae.FEATURE_NAMES, torch.device("cpu"))
        latest_feature_frame.to_csv(directory / "teacher_alignment_latest.csv", index=False)
        rows.append({
            "seed": seed, "development_auc": measurement["roc_auc"],
            "development_f1": measurement["f1"], "development_ap": measurement["average_precision"],
            "selected_epoch": report["selection"]["best_epoch"],
            "last_epoch_auc": float(history.validation_roc_auc.iloc[-1]),
            "teacher_relative_error_reduction": health["relative_error_reduction"],
            "pretrained_teacher_validation_mse": health["teacher_validation_mse"],
            "selected_teacher_validation_mse": selected_feature_health["teacher_validation_mse"],
            "last_epoch_teacher_validation_mse": latest_feature_health["teacher_validation_mse"],
            "critic_to_reconstruction_score_std_ratio": float(critic_weight * components.critic.std() / components.sum_squared_error.std()),
        })
        joined = development.merge(scores[["source_row", "anomaly_score"]], on="source_row", validate="one_to_one", sort=False)
        normal = joined[joined.label == 0]
        for ratio, anomalies in joined[joined.label == 1].groupby("nominal_interference_ratio_db"):
            subset = pd.concat([normal, anomalies], ignore_index=True)
            metrics = fgaae.metrics(subset.label.to_numpy(), subset.anomaly_score.to_numpy(), checkpoint["threshold"])
            grouped.append({"seed": seed, "nominal_interference_ratio_db": ratio, **metrics})
        fpr, tpr, _ = roc_curve(scores.label.to_numpy(), scores.anomaly_score.to_numpy())
        curves.append((seed, fpr, tpr, measurement["roc_auc"]))
        verification.append({
            "seed": seed, "cpu_reload_frames": count, "cpu_gpu_max_score_difference": difference,
            "cpu_gpu_max_relative_score_difference": relative_difference,
            "cpu_default_cuda_precision_score_tolerance_passed": cpu_default_precision_close,
            "cpu_cuda_fp32_max_score_difference_when_checked": fp32_cpu_gpu_difference,
            "cuda_reload_max_score_difference": cuda_difference,
            "cpu_comparison_rtol": 5e-4, "cpu_comparison_atol": 1e-3,
            "epochs_stage1": len(pretraining), "epochs_stage2": len(history),
            "aqt_rounds": audit["aqt_rounds"], "teacher_frozen": True,
            "all_trainable_components_updated": True, "saved_metrics_recomputed": True,
        })
        del model, checkpoint, latest
    frame = pd.DataFrame(rows)
    assert len(frame) == len(summary["seeds"])
    assert abs(float(frame.development_auc.mean()) - summary["development_auc_mean"]) < 1e-12
    frame.to_csv(run / "full_model_diagnostics.csv", index=False)
    by_power = pd.DataFrame(grouped)
    by_power.to_csv(run / "development_by_injection_ratio.csv", index=False)
    power_summary = by_power.groupby("nominal_interference_ratio_db").agg(
        roc_auc_mean=("roc_auc", "mean"), roc_auc_sample_std=("roc_auc", "std"),
        recall_mean=("recall", "mean"), precision_mean=("precision", "mean"),
    ).reset_index().fillna(0)
    power_summary.to_csv(run / "development_by_injection_ratio_summary.csv", index=False)
    fgaae.write_json(run / "verification.json", {
        "summary_code_sha256": full_digest(Path(__file__)),
        "code_and_data_hashes_unchanged": True, "source_file_and_row_isolation": True,
        "registered_budget_completed": True, "primary_equation_verified": True,
        "confirmation_or_test_evaluated": False, "runs": verification,
    })
    cli_note = ""
    cli_output = run / "reloaded_development.csv"
    if cli_output.exists():
        saved = pd.read_csv(run / f"seed_{plan['seeds'][0]}/development_scores.csv", float_precision="round_trip")
        inferred = pd.read_csv(cli_output, float_precision="round_trip")
        assert len(saved) == len(inferred) == len(development)
        assert np.allclose(saved.anomaly_score, inferred.anomaly_score, rtol=5e-4, atol=1e-3)
        threshold = read_json(run / f"seed_{plan['seeds'][0]}/report.json")["development"]["threshold"]
        inferred_metrics = fgaae.metrics(development.label.to_numpy(), inferred.anomaly_score.to_numpy(), threshold)
        saved_auc = float(frame.development_auc.iloc[0])
        assert abs(inferred_metrics["roc_auc"] - saved_auc) < 1e-4
        decision_differences = int((saved.predicted_label != inferred.predicted_label).sum())
        assert decision_differences / len(saved) < 0.001
        cli_verification = {
            "seed": plan["seeds"][0], "frames": len(saved),
            "maximum_score_difference": float(np.max(np.abs(saved.anomaly_score - inferred.anomaly_score))),
            "auc_absolute_difference": abs(inferred_metrics["roc_auc"] - saved_auc),
            "different_decisions": decision_differences, "cpu_development": inferred_metrics,
        }
        fgaae.write_json(run / "prediction_cli_verification.json", cli_verification)
        cli_note = f"第一个种子的 CPU 预测入口已额外检查全部 {len(saved)} 帧，AUC 绝对差 {cli_verification['auc_absolute_difference']:.8f}，{decision_differences} 帧因阈值附近的浮点差异改变判定，详见 prediction_cli_verification.json。"
    plot(run, curves, power_summary)
    lines = [
        f"# 完整 FGAAE 实验结果：{plan.get('training_profile', 'complete_fgaae_v4')}", "",
        f"已完成 {len(rows)} 个种子的特征预训练、教师/学生蒸馏、WGAN-GP 和 AQT。未开展消融。", "",
        f"开发集 AUC：**{summary['development_auc_mean']:.6f} ± {summary['development_auc_sample_std']:.6f}**（样本标准差）；F1 均值 **{summary['development_f1_mean']:.6f}**，AP 均值 **{summary['development_ap_mean']:.6f}**。", "",
        "该成绩来自既有合成异常开发集，开发集还用于检查点与阈值选择。它不是新独立测试成绩，也不证明论文原始实验性能已复现。", "",
        f"| 种子 | 开发 AUC | 开发 F1 | 开发 AP | 选中 epoch | 第 {plan['iterations'] * plan['epochs_per_iteration']} epoch AUC | 教师误差相对下降 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['seed']} | {row['development_auc']:.6f} | {row['development_f1']:.6f} | {row['development_ap']:.6f} | {row['selected_epoch']} | {row['last_epoch_auc']:.6f} | {row['teacher_relative_error_reduction']:.1%} |")
    lines.extend([
        "", "## 本轮设置与验证", "",
        f"- 阶段一：{len(manifest['reference_training_rows'])} 帧特征训练、{len(manifest['reference_validation_rows'])} 帧留出验证，按源文件隔离；每个种子预训练预算为 {plan['stage1_epochs']} 个 epoch、{health['stage1_optimizer_updates']} 次优化更新。" + ("本轮直接复用已完成的相同种子预训练检查点，没有重新执行预训练优化；来源哈希见 teacher_health.json。" if plan.get("pretrained_run") else ""),
        f"- 阶段二：{len(training)} 帧，初始合成注入比例 {float(training.label.mean()):.3%}，{plan['iterations']} 次 AQT、每次 {plan['epochs_per_iteration']} 个 epoch，Critic 每批更新 {plan['critic_steps']} 次；所有种子均跑满预算。",
        f"- 开发集：{len(development)} 帧、{development.source_file.nunique()} 个独立于训练的源文件，注入比例 {float(development.label.mean()):.3%}。数据和代码哈希验证通过。",
        "- 编码器、解码器、学生头、Critic 均有实际参数更新；教师头在阶段二未变化。所有模型均通过重新加载与评分核验，CPU/CUDA 数值差异单独记录。",
        "- 主评分为平方误差求和 + 正系数×Critic(重构)，默认系数 0.1；若启用检查点/系数联合选择，候选及开发文件在 registered_plan.json 中登记。负号解释只作诊断。",
        f"- 编码器学习率：{plan.get('encoder_lr') or 2e-5:g}；其余生成器模块学习率 2e-5；固定编码器 BatchNorm 统计量：{plan.get('freeze_encoder_bn', False)}；AQT 末端保留率参数：{plan.get('retention_end', 0.5):g}。",
        "", "## 如何解释结果", "",
        "完整训练跑满了预定 AQT 轮次，但导出的 model.pt 是开发集选择的最佳 epoch；若选中较早 epoch，不能据此声称后续 AQT 轮次带来提升。末轮成绩同时列出，用于观察继续训练是否改善了检测。",
        "", "教师误差下降表示特征预训练确有学习效果，不代表最终异常检测已达到同等提升。教师头冻结后，共享编码器仍更新，其输入和输出都可能变化。",
        "", f"同一特征留出集上的教师 MSE，各种子均值：预训练后 {float(frame.pretrained_teacher_validation_mse.mean()):.6f}，最佳检测检查点 {float(frame.selected_teacher_validation_mse.mean()):.6f}，末轮 {float(frame.last_epoch_teacher_validation_mse.mean()):.6f}。这项诊断检查共享编码器更新后特征映射是否漂移，不参与模型选择。",
        "", "原始背景没有经过人工正常性确认；异常是冻结的局部功率注入。图中的 dB 是注入/背景功率比，不能标作论文的信道 SNR。25 项专家特征、部分损失权重、AQT 参数、Critic 归一化及教师输出停止梯度仍是实现假设，详见 [FULL_MODEL.md](../../FULL_MODEL.md)。",
        "", "进一步逼近论文，需要先确定原始特征和异常生成/标注协议，再在开发集解决训练稳定性问题。已有确认集、测试集和外站集本轮均未评估。",
        "", "## 文件", "",
        "- [完整数值报告](full_model_report.json)、[验证记录](verification.json)、[训练诊断](full_model_diagnostics.csv)。",
        "- [开发集 ROC 与注入强度曲线](development_curves.png)、[按注入强度汇总](development_by_injection_ratio_summary.csv)。每种强度与同一批开发集背景比较，各组不是独立重复实验。",
        "- 各 seed 目录的 model.pt、训练历史、teacher_health.json、component_audit.json 和 AQT 逐帧记录。",
    ])
    if cli_note:
        lines.extend(["", cli_note])
    if float(frame.last_epoch_auc.mean()) < float(frame.development_auc.mean()) - 0.01:
        lines.extend(["", "本轮末期平均 AUC 低于保存的早期检查点，继续减少重构误差并未稳定改善异常排序；扩大训练规模前应优先检查完整模型的训练动态。"])
    (run / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verification_passed": True, "development_auc_mean": summary["development_auc_mean"], "development_auc_sample_std": summary["development_auc_sample_std"], "last_epoch_auc_mean": float(frame.last_epoch_auc.mean()), "report": str(run / "RESULTS.md")}, ensure_ascii=False, indent=2))


def plot(run, curves, power_summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for seed, fpr, tpr, auc in curves:
        axes[0].plot(fpr, tpr, label=f"Seed {seed}: AUC={auc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Development ROC (selected checkpoints)")
    axes[0].legend()
    axes[1].errorbar(power_summary.nominal_interference_ratio_db, power_summary.roc_auc_mean, yerr=power_summary.roc_auc_sample_std, marker="o", capsize=3)
    axes[1].set(xlabel="Injected/background power ratio (dB; not channel SNR)", ylabel="ROC-AUC mean +/- sample SD", title="Same frozen development backgrounds")
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(run / "development_curves.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify and summarize completed full FGAAE runs without evaluating test data.")
    parser.add_argument("--run", type=Path, default=ROOT / "runs/full_fgaae_v4")
    summarize(parser.parse_args().run.resolve())
