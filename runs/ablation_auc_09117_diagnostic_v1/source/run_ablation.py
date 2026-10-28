"""Run the approved five-group, AUC-only ablation against frozen FGAAE 0.9117."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import tempfile
import types

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

import ablation_core as implementation
from freeze_baseline import FROZEN, HERE, ROOT, RUN, digest, read_json, verify, write_json

CORE, FULL, DERIVED, PATCH = implementation.load_context()
SEEDS = [2025, 2026, 2027]
VARIANTS = ["Base", "A", "B", "C"]
CANDIDATES = [None, 0.01, 0.03, 0.1, 0.3, 1.0]


def code_hashes():
    return {name: digest(HERE / name) for name in ["run_ablation.py", "ablation_core.py"]}


def snapshot_sources(output):
    source = output / "source"
    source.mkdir()
    for name in code_hashes():
        shutil.copy2(HERE / name, source / name)
    (source / "fgaae_ablation.py").write_text(DERIVED, encoding="utf-8")
    (source / "historical_source_changes.diff").write_text(PATCH, encoding="utf-8")


def load_data(output):
    arguments = types.SimpleNamespace(**read_json(RUN / "registered_plan.json"))
    data = FULL.prepare_inputs(arguments, output / "inputs")
    for name in ["training_membership.csv", "development_membership.csv"]:
        if not pd.read_csv(output / "inputs" / name).equals(pd.read_csv(RUN / "inputs" / name)):
            raise ValueError(f"Frozen membership changed: {name}")
    parameters = data["manifest"]["normalization"]
    for source, target in [("training", "training_norm"), ("development", "development_norm"),
                           ("reference_training", "reference_norm"), ("reference_validation", "reference_validation_norm")]:
        data[target] = FULL.normalized(data[source], parameters)
    data["development_metadata"] = pd.read_csv(output / "inputs/development_membership.csv")
    return data


def prepare_model(arm, seed, data, device, output):
    CORE.seed_everything(seed, 2)
    original = torch.load(RUN / f"seed_{seed}/model.pt", map_location="cpu", weights_only=True)
    config = implementation.configure(original["config"], arm, seed, device, output)
    model = CORE.FGAAE(config).to(device)
    flags = implementation.ARMS[arm]
    provenance = {"expert_pretraining_loaded": flags["features"], "bn_calibration": None}
    if flags["features"]:
        path = RUN / f"seed_{seed}/stage1_best.pt"
        pretrained = torch.load(path, map_location="cpu", weights_only=True)
        model.encoder.load_state_dict(pretrained["encoder"])
        model.teacher.load_state_dict(pretrained["teacher"])
        provenance["pretraining_sha256"] = digest(path)
    else:
        provenance["bn_calibration"] = implementation.calibrate_encoder_bn(CORE, model, data["reference_norm"], config["batch_size"], device)
        model.student.requires_grad_(False)
    model.freeze_teacher()
    model.critic.requires_grad_(flags["adversarial"])
    callback = None
    if flags["features"]:
        expert = CORE.expert_features(data["reference_validation"], config).astype(np.float64)
        targets = (expert - np.asarray(original["feature_mean"])) / np.asarray(original["feature_std"])
        loader = CORE.make_loader(data["reference_validation_norm"], 32, targets)
        def callback(current_model, epoch):
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                _, health = FULL.teacher_metrics(current_model, loader, CORE.FEATURE_NAMES, device)
            return {"reference_teacher_feature_mse": health["teacher_validation_mse"]}
    metadata = {"normalization": data["manifest"]["normalization"], "arm": arm, "seed": seed,
                "active_modules": flags, "feature_mean": original["feature_mean"] if flags["features"] else None,
                "feature_std": original["feature_std"] if flags["features"] else None,
                "station": data["manifest"]["station"], "artifact_type": "fgaae_ablation_auc_09117"}
    return model, config, metadata, provenance, callback


def train_seed(arm, seed, data, device, output):
    output.mkdir()
    model, config, metadata, provenance, callback = prepare_model(arm, seed, data, device, output)
    before = FULL.snapshot(model)
    write_json(output / "config.json", config)
    write_json(output / "initialization.json", provenance)
    CORE.seed_everything(seed + 10000, 2)
    selection = CORE.train_stage_two(model, data["training_norm"], data["development_norm"], data["labels"],
                                    data["source_rows"], config, device, output, epoch_callback=callback)
    changes = FULL.parameter_changes(model, before)
    flags = implementation.ARMS[arm]
    expected = {"encoder": True, "decoder": True, "teacher": False, "student": flags["features"], "critic": flags["adversarial"]}
    for name, active in expected.items():
        if (changes[name] > 0) != active:
            raise AssertionError(f"Incorrect component update in {arm}/{seed}: {name}={changes[name]}")
    history = pd.read_csv(output / "stage2_history.csv")
    if len(history) != 60 or history.iteration.nunique() != 3:
        raise AssertionError("Incomplete registered training budget.")
    if not flags["aqt"] and not history.retained_samples.eq(len(data["training_norm"])).all():
        raise AssertionError("Disabled AQT removed training rows.")
    latest = torch.load(output / "stage2_latest.pt", map_location="cpu", weights_only=True)
    selected_buffers = FULL.encoder_buffer_changes(FULL.snapshot(model), before)
    latest_buffers = FULL.encoder_buffer_changes(latest["model_state"], before)
    if any(value != 0 for value in [*selected_buffers.values(), *latest_buffers.values()]):
        raise AssertionError("Frozen encoder BN changed during stage two.")
    state = FULL.snapshot(model)
    result = implementation.auc_scores(CORE, model, data["development_norm"], config, device)
    reference = implementation.auc_scores(CORE, model, data["reference_norm"], config, device)
    pd.DataFrame({"source_row": data["development_rows"], "label": data["labels"],
                  "reconstruction_sum": result["reconstruction_score"], "critic": result["critic_value"],
                  "uncalibrated_score": result["scores"]}).to_csv(output / "score_components.csv", index=False)
    scales = {"reconstruction_std": float(reference["reconstruction_score"].std()), "critic_std": float(reference["critic_value"].std())}
    if flags["adversarial"] and min(scales.values()) <= 1e-8:
        raise ValueError("Critic coefficient calibration has degenerate reference variance.")
    write_json(output / "reference_scales.json", scales)
    torch.save({**metadata, "config": config, "model_state": state}, output / "uncalibrated_model.pt")
    updates = int(np.ceil(history.retained_samples / config["batch_size"]).sum())
    audit = {"parameter_changes": changes, "encoder_bn_changes_selected": selected_buffers,
             "encoder_bn_changes_latest": latest_buffers, "generator_updates": updates,
             "critic_updates": updates * config["critic_steps"] if flags["adversarial"] else 0,
             "retained_per_iteration": history.groupby("iteration").retained_samples.first().astype(int).tolist(),
             "stage1_reused": flags["features"], "stage2_epochs": len(history), "active_modules": flags}
    write_json(output / "component_audit.json", audit)
    report = {"training_complete": True, "arm": arm, "seed": seed, "selection": selection,
              "uncalibrated_auc": float(roc_auc_score(data["labels"], result["scores"])),
              "reconstruction_auc": float(roc_auc_score(data["labels"], result["reconstruction_score"])),
              "uncalibrated_model_sha256": digest(output / "uncalibrated_model.pt"), "audit": audit}
    write_json(output / "training_report.json", report)
    print(f"TRAINED {arm} seed={seed}: raw AUC={report['uncalibrated_auc']:.6f}, R-only AUC={report['reconstruction_auc']:.6f}", flush=True)
    del model, latest, state
    if device.type == "cuda":
        torch.cuda.empty_cache()


def calibration_masks(metadata):
    policy = read_json(RUN / "score_calibration/registered_plan.json")
    selection = metadata.source_file.isin(policy["selection_files"]).to_numpy()
    check = metadata.source_file.isin(policy["coefficient_check_files"]).to_numpy()
    if not np.array_equal(~selection, check) or selection.sum() != 5296:
        raise AssertionError("Historical coefficient split changed.")
    return selection, check


def finalize_arm(arm, data, device, directory):
    flags = implementation.ARMS[arm]
    registered_name = "registered_0.1" if flags["adversarial"] else "reconstruction_only"
    selection, check = calibration_masks(data["development_metadata"])
    candidate_rows = []
    for seed in SEEDS:
        source = directory / f"seed_{seed}"
        components = pd.read_csv(source / "score_components.csv", float_precision="round_trip")
        scales = read_json(source / "reference_scales.json")
        for relative in CANDIDATES if flags["adversarial"] else [None]:
            name = registered_name if relative is None else f"relative_{relative:g}"
            weight = (0.1 if relative is None else relative * scales["reconstruction_std"] / scales["critic_std"]) if flags["adversarial"] else 0.0
            scores = components.reconstruction_sum.to_numpy() + weight * components.critic.to_numpy()
            candidate_rows.append({"seed": seed, "candidate": name, "critic_weight": weight,
                                   "selection_auc": float(roc_auc_score(data["labels"][selection], scores[selection])),
                                   "check_auc": float(roc_auc_score(data["labels"][check], scores[check])),
                                   "full_auc": float(roc_auc_score(data["labels"], scores))})
    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(directory / "calibration_candidates_auc.csv", index=False)
    means = candidates.groupby("candidate", sort=False)[["selection_auc", "check_auc", "full_auc"]].mean()
    best = means.selection_auc.idxmax()
    if means.loc[best, "selection_auc"] <= means.loc[registered_name, "selection_auc"] + 1e-4:
        best = registered_name
    accepted = bool(best != registered_name and means.loc[best, "check_auc"] > means.loc[registered_name, "check_auc"] and means.loc[best, "full_auc"] > means.loc[registered_name, "full_auc"])
    deployed = best if accepted else registered_name
    means.to_csv(directory / "calibration_candidate_summary.csv")
    write_json(directory / "calibration_policy_result.json", {"candidate_by_selection_auc": best, "calibration_accepted": accepted,
               "deployed_candidate": deployed, "shared_across_seeds": True,
               "later_and_full_development_used_for_acceptance": True, "independent_test": False})
    rows, replay = [], []
    for seed in SEEDS:
        source = directory / f"seed_{seed}"
        checkpoint = torch.load(source / "uncalibrated_model.pt", map_location="cpu", weights_only=True)
        weight = float(candidates[(candidates.seed == seed) & (candidates.candidate == deployed)].iloc[0].critic_weight)
        checkpoint["config"] = {**checkpoint["config"], "critic_score_weight": weight}
        checkpoint["calibration"] = {"candidate": deployed, "accepted": accepted, "reference_only_scales": True}
        checkpoint["score_formula"] = f"sum_squared_residual + {weight:.17g} * critic(reconstruction)" if flags["adversarial"] else "sum_squared_residual"
        torch.save(checkpoint, source / "model.pt")
        loaded = torch.load(source / "model.pt", map_location="cpu", weights_only=True)
        if not all(torch.equal(value, loaded["model_state"][name]) for name, value in checkpoint["model_state"].items()):
            raise AssertionError("Serialization changed tensors.")
        model = CORE.FGAAE(loaded["config"]).to(device)
        model.load_state_dict(loaded["model_state"])
        result = implementation.auc_scores(CORE, model, data["development_norm"], loaded["config"], device)
        components = pd.read_csv(source / "score_components.csv", float_precision="round_trip")
        expected = components.reconstruction_sum.to_numpy() + weight * components.critic.to_numpy()
        difference = float(np.abs(expected - result["scores"]).max())
        if difference > 1e-8:
            raise AssertionError(f"Exported checkpoint does not replay exactly: {difference}")
        pd.DataFrame({"source_row": data["development_rows"], "label": data["labels"], "score": result["scores"],
                      "reconstruction_sum": result["reconstruction_score"], "critic": result["critic_value"]}).to_csv(source / "development_scores.csv", index=False)
        training = read_json(source / "training_report.json")
        rows.append({"arm": arm, "seed": seed, "auc_primary": float(roc_auc_score(data["labels"], result["scores"])),
                     "auc_reconstruction": float(roc_auc_score(data["labels"], result["reconstruction_score"])),
                     "auc_uncalibrated": training["uncalibrated_auc"], "best_epoch": training["selection"]["best_epoch"],
                     "generator_updates": training["audit"]["generator_updates"], "critic_updates": training["audit"]["critic_updates"],
                     "critic_weight": weight, "coefficient_candidate": deployed})
        replay.append({"seed": seed, "frames": len(expected), "maximum_score_difference": difference, "model_sha256": digest(source / "model.pt")})
        del model, checkpoint, loaded
    pd.DataFrame(rows).to_csv(directory / "auc_runs.csv", index=False)
    write_json(directory / "deployment_verification.json", replay)
    write_json(directory / "arm_complete.json", {"arm": arm, "seeds": SEEDS, "complete": True})
    return rows


def baseline_rows():
    frozen = pd.read_csv(FROZEN / "baseline_auc.csv")
    rows = []
    for row in frozen.itertuples():
        rows.append({"arm": "Full", "seed": row.seed, "auc_primary": row.auc_primary,
                     "auc_reconstruction": row.auc_reconstruction_same_checkpoint, "best_epoch": row.best_epoch,
                     "generator_updates": row.generator_updates_completed, "critic_updates": row.generator_updates_completed * 5,
                     "critic_weight": row.critic_weight, "coefficient_candidate": "relative_1"})
    return rows


def summarize(output, data, rows):
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "auc_runs.csv", index=False)
    order = ["Base", "A", "B", "C", "Full"]
    summary = frame.groupby("arm").agg(auc_mean=("auc_primary", "mean"), auc_sample_std=("auc_primary", "std"),
              reconstruction_auc_mean=("auc_reconstruction", "mean"), reconstruction_auc_sample_std=("auc_reconstruction", "std")).reindex(order)
    full = frame[frame.arm == "Full"].set_index("seed")
    paired = []
    for row in frame[frame.arm != "Full"].itertuples():
        paired.append({"arm": row.arm, "seed": row.seed, "full_auc": float(full.loc[row.seed, "auc_primary"]),
                       "variant_auc": row.auc_primary, "delta_full_minus_variant": float(full.loc[row.seed, "auc_primary"] - row.auc_primary),
                       "reconstruction_delta_full_minus_variant": float(full.loc[row.seed, "auc_reconstruction"] - row.auc_reconstruction)})
    pair_frame = pd.DataFrame(paired)
    pair_frame.to_csv(output / "paired_auc_differences.csv", index=False)
    summary["delta_full_minus_variant"] = summary.loc["Full", "auc_mean"] - summary.auc_mean
    summary.to_csv(output / "auc_summary.csv")
    grouped = []
    curves = []
    for arm in order:
        for seed in SEEDS:
            if arm == "Full":
                scores = pd.read_csv(RUN / f"score_calibration/seed_{seed}_development_scores.csv", float_precision="round_trip").anomaly_score.to_numpy()
            else:
                scores = pd.read_csv(output / arm / f"seed_{seed}/development_scores.csv", float_precision="round_trip").score.to_numpy()
            fpr, tpr, _ = roc_curve(data["labels"], scores)
            curves.append((arm, seed, fpr, tpr))
            normal = np.flatnonzero(data["labels"] == 0)
            for ratio, subset in data["development_metadata"][data["labels"] == 1].groupby("nominal_interference_ratio_db"):
                indices = np.concatenate([normal, subset.index.to_numpy()])
                grouped.append({"arm": arm, "seed": seed, "nominal_interference_ratio_db": ratio,
                                "auc": float(roc_auc_score(data["labels"][indices], scores[indices]))})
    pd.DataFrame(grouped).to_csv(output / "auc_by_injection_ratio.csv", index=False)
    plot_results(output, summary, curves)
    conclusions = []
    for arm, title in [("A", "完整对抗模块"), ("B", "完整特征引导流程"), ("C", "AQT")]:
        differences = pair_frame[pair_frame.arm == arm].delta_full_minus_variant
        positives = int((differences > 0).sum())
        message = "三个种子均为 Full 更高" if positives == 3 else ("三个种子均未显示 Full 更高" if positives == 0 else "不同种子方向不一致")
        conclusions.append(f"- {title}（Full 对 {arm}）：平均差值 {differences.mean():+.6f}；{message}。")
    lines = ["# FGAAE 五组 AUC 消融结果", "", "基准为已冻结的历史 AUC 0.911744 版本。Base/A/B/C 各三个种子重新训练 60 epoch，Full 复用已核验的历史三个模型。指标仅使用 ROC-AUC。", "",
             "同一 20000 帧训练池、10592 帧开发集及冻结的异常协议；均值 ± 三个初始化种子的样本标准差。所有开发文件历史上都已参与模型开发，本轮不是独立测试。", "",
             "| 组别 | 主 AUC | 统一重构 AUC | Full − 本组 |", "| --- | --- | --- | --- |"]
    for arm, row in summary.iterrows():
        lines.append(f"| {arm} | {row.auc_mean:.6f} ± {row.auc_sample_std:.6f} | {row.reconstruction_auc_mean:.6f} ± {row.reconstruction_auc_sample_std:.6f} | {row.delta_full_minus_variant:+.6f} |")
    lines += ["", "正差值表示 Full 更高，负差值表示消融组更高。统一重构列使用相同已选检查点，不另行挑选模型。", "", *conclusions, "",
              "A 同时移除对抗训练、Critic 最终评分及 Critic 对 AQT 的影响，不能将主 AUC 的差异全部归为对抗训练。B 同时移除专家预训练和蒸馏，不能将差异全部归为专家知识。B/Base 使用训练参考背景进行无标签 BN 校准；没有加载专家预训练编码器。", "",
              "AQT 关闭组始终保留全部 20000 帧。相同 60 epoch 不等于相同优化更新数：Full 每种子 4200 次 G 更新，C/Base 每种子 4740 次；详见 auc_runs.csv。Full 历史选中的 epoch 为 2/9/10，均在第二次 AQT 前，不能据此证明后续动态过滤轮次有贡献。", "",
              "检查点、正评分系数候选、三个种子共享系数的选择和采用条件，沿用冻结方案。较晚四开发文件也参与采用校准的判定，不能当作独立测试。未使用局部残差增强、未调整异常强度或比例，也未根据结果删去某个预定组。", "",
              "每个变体均核验教师与关闭模块不更新、开启模块实际更新、编码器 BN 冻结、完整预算，以及模型导出后的全量 CUDA 重载评分一致性。代码补丁、源码快照和预检查记录均保留。", "",
              "文件：[逐种子 AUC](auc_runs.csv)、[配对差值](paired_auc_differences.csv)、[按名义功率比分组的 AUC](auc_by_injection_ratio.csv)、[ROC 曲线](roc_curves.png)、[登记方案](registered_plan.json)。名义注入功率比不是论文信道 SNR。AUC 结果支持的范围是此协议下的评分区分能力，不能直接证明原论文全部性能或跨站泛化。"]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "report.json", {"complete": True, "metrics": ["ROC-AUC"], "independent_test": False,
               "summary": summary.reset_index().to_dict("records"), "new_training_runs": 12, "epochs_per_run": 60,
               "paired_differences": paired, "full_reused": True, "all_prespecified_arms_reported": True})
    print(summary.to_string(), flush=True)


def plot_results(output, summary, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.linspace(0, 1, 1001)
    figure, axis = plt.subplots(figsize=(7, 6))
    interpolated = []
    for arm in summary.index:
        values = [np.interp(grid, fpr, tpr) for name, seed, fpr, tpr in curves if name == arm]
        mean = np.mean(values, axis=0)
        spread = np.std(values, axis=0, ddof=1)
        row = summary.loc[arm]
        axis.plot(grid, mean, label=f"{arm}: {row.auc_mean:.4f} +/- {row.auc_sample_std:.4f}")
        axis.fill_between(grid, np.maximum(0, mean-spread), np.minimum(1, mean+spread), alpha=0.08)
        interpolated.extend({"arm": arm, "fpr": float(x), "mean_tpr": float(y), "sample_std_tpr": float(s)} for x, y, s in zip(grid, mean, spread))
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate", title="Historical FGAAE ablation: same development protocol")
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output / "roc_curves.png", dpi=180)
    plt.close(figure)
    pd.DataFrame(interpolated).to_csv(output / "roc_interpolation.csv", index=False)


def small_config():
    config = read_json(RUN / "seed_2026/config.json")
    config.update(batch_size=8, iterations=2, epochs_per_iteration=1, critic_steps=1, monitor_encoder_gradients=False)
    return config


def smoke_test(device):
    original = implementation.load_module("original_core_for_parity", (implementation.SOURCE / "fgaae.py").read_text(encoding="utf-8"), implementation.SOURCE / "fgaae.py")
    rng = np.random.default_rng(17)
    training = rng.normal(size=(24, 840)).astype(np.float32)
    validation = rng.normal(size=(16, 840)).astype(np.float32)
    labels = np.tile([0, 0, 0, 1], 4)
    config = small_config()
    with tempfile.TemporaryDirectory(prefix="auc_ablation_smoke_") as temporary:
        directory = Path(temporary)
        def run_one(module, settings, path):
            path.mkdir()
            module.seed_everything(2026, 2)
            model = module.FGAAE(settings).to(device)
            model.freeze_teacher()
            before = FULL.snapshot(model)
            module.seed_everything(12026, 2)
            module.train_stage_two(model, training, validation, labels, np.arange(24), settings, device, path)
            return model, before
        legacy, _ = run_one(original, copy.deepcopy(config), directory / "legacy")
        derived, _ = run_one(CORE, copy.deepcopy(config), directory / "derived")
        if not all(torch.equal(value, derived.state_dict()[name]) for name, value in legacy.state_dict().items()):
            raise AssertionError("Full path differs from frozen core in synthetic parity.")
        for arm, flags in implementation.ARMS.items():
            if arm == "Full":
                continue
            settings = implementation.configure(small_config(), arm, 2026, device, directory / arm)
            settings["monitor_encoder_gradients"] = False
            CORE.seed_everything(2026, 2)
            model = CORE.FGAAE(settings).to(device)
            model.freeze_teacher()
            if not flags["features"]:
                implementation.calibrate_encoder_bn(CORE, model, training, 8, device)
                model.student.requires_grad_(False)
            model.critic.requires_grad_(flags["adversarial"])
            before = FULL.snapshot(model)
            path = directory / arm
            path.mkdir()
            CORE.seed_everything(12026, 2)
            CORE.train_stage_two(model, training, validation, labels, np.arange(24), settings, device, path)
            changes = FULL.parameter_changes(model, before)
            assert changes["teacher"] == 0
            assert (changes["student"] > 0) == flags["features"]
            assert (changes["critic"] > 0) == flags["adversarial"]
            assert changes["encoder"] > 0 and changes["decoder"] > 0
            assert all(value == 0 for value in FULL.encoder_buffer_changes(FULL.snapshot(model), before).values())
            result = implementation.auc_scores(CORE, model, validation, settings, device)
            history = pd.read_csv(path / "stage2_history.csv")
            if not flags["aqt"]:
                assert history.retained_samples.eq(len(training)).all()
            assert "validation_f1" not in history.columns and "validation_recall" not in history.columns
    print("SMOKE PASSED: Full tensor parity, all four switches, no removed-component updates, no Critic score leakage, unlabelled BN parameter isolation, AUC-only history.", flush=True)


def precheck(output, device):
    verify()
    output.mkdir(parents=True, exist_ok=False)
    snapshot_sources(output)
    smoke_test(device)
    data = load_data(output)
    calibration_test(data, device)
    directory = output / "Full_seed_2026_first_two_epochs"
    directory.mkdir()
    model, config, _, _, callback = prepare_model("Full", 2026, data, device, directory)
    config.update(iterations=1, epochs_per_iteration=2)
    CORE.seed_everything(12026, 2)
    CORE.train_stage_two(model, data["training_norm"], data["development_norm"], data["labels"], data["source_rows"], config, device, directory, epoch_callback=callback)
    current = pd.read_csv(directory / "stage2_history.csv")
    historical = pd.read_csv(RUN / "seed_2026/stage2_history.csv").iloc[:2]
    columns = [column for column in current.columns if column in historical.columns]
    maximum = float((current[columns] - historical[columns]).abs().max().max())
    if maximum != 0:
        raise AssertionError(f"Real Full trajectory differs from historical training: {maximum}")
    write_json(output / "precheck_report.json", {"passed": True, "smoke_test_passed": True, "calibration_pipeline_test_passed": True, "full_training_parity_epochs": 2,
               "shared_numeric_columns": len(columns), "maximum_numeric_difference": maximum,
               "code_sha256": code_hashes(), "frozen_manifest_sha256": digest(FROZEN / "baseline_manifest.json")})
    print("REAL FULL PARITY PASSED: identical first two epochs.", flush=True)


def calibration_test(data, device):
    """Replay the original calibration and Critic-free export using historical fixtures."""
    scales = pd.read_csv(RUN / "score_calibration/training_reference_scales.csv").set_index("seed")
    with tempfile.TemporaryDirectory(prefix="auc_calibration_fixture_") as temporary:
        for arm in ["Full", "A"]:
            directory = Path(temporary) / arm
            directory.mkdir()
            for seed in SEEDS:
                source = directory / f"seed_{seed}"
                source.mkdir()
                checkpoint = torch.load(RUN / f"seed_{seed}/model.pt", map_location="cpu", weights_only=True)
                checkpoint["config"] = implementation.configure(checkpoint["config"], arm, seed, device, source)
                torch.save(checkpoint, source / "uncalibrated_model.pt")
                components = pd.read_csv(RUN / f"seed_{seed}/score_components.csv", float_precision="round_trip")
                critic = components.critic.to_numpy() if arm == "Full" else np.zeros(len(components))
                reconstruction = components.sum_squared_error.to_numpy()
                uncalibrated = reconstruction + (0.1 if arm == "Full" else 0) * critic
                pd.DataFrame({"reconstruction_sum": reconstruction, "critic": critic}).to_csv(source / "score_components.csv", index=False)
                write_json(source / "reference_scales.json", {"reconstruction_std": float(scales.loc[seed, "reconstruction_std"]), "critic_std": float(scales.loc[seed, "critic_std"])})
                write_json(source / "training_report.json", {"uncalibrated_auc": float(roc_auc_score(data["labels"], uncalibrated)),
                           "selection": {"best_epoch": read_json(RUN / f"seed_{seed}/selection.json")["best_epoch"]},
                           "audit": {"generator_updates": 4200, "critic_updates": 21000 if arm == "Full" else 0}})
            rows = finalize_arm(arm, data, device, directory)
            expected = 0.9117440964193119 if arm == "Full" else 0.8918607919403739
            if abs(np.mean([row["auc_primary"] for row in rows]) - expected) > 1e-12:
                raise AssertionError("Calibration fixture did not recover historical AUC.")
            if arm == "A" and any(row["coefficient_candidate"] != "reconstruction_only" for row in rows):
                raise AssertionError("Critic-free models entered coefficient calibration.")
    print("CALIBRATION PIPELINE PASSED: historical 0.911744 recovered, Critic-free export verified; fixtures are not ablation results.", flush=True)


def run(output, device, prechecked, resume=False):
    verify()
    check = read_json(prechecked / "precheck_report.json")
    if not check["passed"] or check["code_sha256"] != code_hashes() or check["frozen_manifest_sha256"] != digest(FROZEN / "baseline_manifest.json"):
        raise ValueError("Precheck is missing or no longer matches the experiment code.")
    if resume:
        registered = read_json(output / "registered_plan.json")
        if registered["code_sha256"] != code_hashes():
            raise ValueError("Resume cannot change registered training code.")
    else:
        output.mkdir(parents=True, exist_ok=False)
        snapshot_sources(output)
        shutil.copy2(HERE / "PLAN.md", output / "PLAN.md")
        write_json(output / "registered_plan.json", {"arms": implementation.ARMS, "new_training_arms": VARIANTS,
                   "seeds": SEEDS, "epochs_per_seed": 60, "iterations": 3, "epochs_per_iteration": 20,
                   "metric": "ROC-AUC", "secondary_score": "reconstruction sum on identical selected checkpoint",
                   "full_reused_from": str(RUN), "full_auc_mean": 0.9117440964193119,
                   "precheck": str(prechecked), "code_sha256": code_hashes(),
                   "frozen_manifest_sha256": digest(FROZEN / "baseline_manifest.json"), "device": str(device),
                   "coefficient_candidates": CANDIDATES, "coefficient_selection": "Historical earlier-file mean AUC and later/full acceptance gate; one candidate per arm shared across seeds.",
                   "bn_without_features": "one unlabelled training-reference calibration pass with Dropout disabled; no parameter updates",
                   "checkpoint_selection": "all eight development files, maximum raw-score AUC, improvement >1e-4",
                   "all_development_previously_inspected": True, "independent_test": False,
                   "same_epochs_not_same_optimizer_updates": True, "extra_tuning_or_arm_selection": False})
        write_json(output / "environment.json", {"torch": str(torch.__version__), "numpy": np.__version__,
                   "pandas": pd.__version__, "cuda": torch.version.cuda, "device": str(device),
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                   "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                   "free_disk_bytes_at_start": shutil.disk_usage(output).free})
    if (output / "inputs").exists():
        # Regenerate nothing when resuming; inputs are reconstructed in a fresh temporary directory.
        with tempfile.TemporaryDirectory(prefix="auc_ablation_inputs_") as temporary:
            data = load_data(Path(temporary))
    else:
        data = load_data(output)
    rows = baseline_rows()
    for arm in VARIANTS:
        directory = output / arm
        directory.mkdir(exist_ok=resume)
        for seed in SEEDS:
            target = directory / f"seed_{seed}"
            report_path = target / "training_report.json"
            if resume and report_path.exists():
                report = read_json(report_path)
                if report.get("training_complete") and report["uncalibrated_model_sha256"] == digest(target / "uncalibrated_model.pt"):
                    print(f"REUSE COMPLETED {arm} seed={seed}", flush=True)
                    continue
            if target.exists():
                archived = directory / f"interrupted_seed_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if not target.resolve().is_relative_to(output.resolve()) or not archived.resolve().is_relative_to(output.resolve()):
                    raise ValueError("Interrupted outputs must remain inside the experiment directory.")
                target.rename(archived)
            print(f"START {arm} seed={seed}", flush=True)
            train_seed(arm, seed, data, device, target)
            write_json(output / "progress.json", {"last_completed_arm": arm, "last_completed_seed": seed, "updated": datetime.now().isoformat()})
        rows.extend(finalize_arm(arm, data, device, directory))
        pd.DataFrame(rows).to_csv(output / "auc_runs_so_far.csv", index=False)
    summarize(output, data, rows)
    verify()
    print(f"ALL ABLATIONS COMPLETE: {output / 'RESULTS.md'}", flush=True)


def predict(checkpoint_path, raw_path, output, device):
    if output.exists():
        raise FileExistsError(output)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = CORE.FGAAE(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    raw = np.load(raw_path, allow_pickle=False)
    values = FULL.normalized(raw, checkpoint["normalization"])
    result = implementation.auc_scores(CORE, model, values, checkpoint["config"], device)
    pd.DataFrame({"score": result["scores"], "reconstruction_sum": result["reconstruction_score"], "critic": result["critic_value"]}).to_csv(output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke-test", "precheck", "train", "predict"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/ablation_auc_09117_v1")
    parser.add_argument("--precheck", type=Path, default=ROOT / "runs/ablation_auc_09117_precheck_v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data", type=Path)
    arguments = parser.parse_args()
    selected_device = torch.device(arguments.device)
    CORE.seed_everything(2025, 2)
    if arguments.command == "smoke-test":
        smoke_test(selected_device)
    elif arguments.command == "precheck":
        precheck(arguments.output.resolve(), selected_device)
    elif arguments.command == "train":
        run(arguments.output.resolve(), selected_device, arguments.precheck.resolve(), arguments.resume)
    else:
        predict(arguments.checkpoint, arguments.data, arguments.output.resolve(), selected_device)
