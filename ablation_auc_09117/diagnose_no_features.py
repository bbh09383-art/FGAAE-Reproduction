"""Post-hoc 2x2 WGAN-GP/AQT diagnosis with expert feature guidance disabled."""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import tempfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import run_ablation as parent
from freeze_baseline import FROZEN, HERE, ROOT, digest, read_json, verify, write_json

PREVIOUS = ROOT / "runs/ablation_auc_09117_v1"
NEW_ARMS = {
    "AQT_only": {"adversarial": False, "features": False, "aqt": True},
    "GAN_only": {"adversarial": True, "features": False, "aqt": False},
}
parent.implementation.ARMS.update(NEW_ARMS)
CORE, FULL = parent.CORE, parent.FULL
ORDER = ["Base", "GAN_only", "AQT_only", "B"]


def code_hashes():
    return {**parent.code_hashes(), Path(__file__).name: digest(Path(__file__))}


def previous_artifacts():
    files = [PREVIOUS / name for name in ["registered_plan.json", "report.json", "auc_runs.csv", "auc_summary.csv", "RESULTS.md", "final_artifact_manifest.json", "inputs/training_membership.csv", "inputs/development_membership.csv", "inputs/manifest.json"]]
    for arm in ["Base", "B"]:
        files += [PREVIOUS / arm / name for name in ["auc_runs.csv", "deployment_verification.json", "calibration_policy_result.json"]]
        for seed in parent.SEEDS:
            directory = PREVIOUS / arm / f"seed_{seed}"
            files += [directory / name for name in ["model.pt", "uncalibrated_model.pt", "development_scores.csv", "score_components.csv", "stage2_history.csv", "training_report.json", "component_audit.json", "initialization.json", "config.json", "selection.json"]]
            files += sorted(directory.glob("aqt_iteration_*.csv"))
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in files}


def verify_previous(expected=None):
    registered = read_json(PREVIOUS / "registered_plan.json")
    for name, checksum in registered["code_sha256"].items():
        if digest(HERE / name) != checksum or digest(PREVIOUS / "source" / name) != checksum:
            raise ValueError(f"Original ablation training source changed: {name}")
    if not read_json(PREVIOUS / "report.json")["complete"]:
        raise ValueError("Original five-group experiment is incomplete.")
    final_manifest = read_json(PREVIOUS / "final_artifact_manifest.json")
    for name in ["auc_runs.csv", "auc_summary.csv", "report.json", "RESULTS.md"]:
        if digest(PREVIOUS / name) != final_manifest["artifacts_sha256"][name]:
            raise ValueError(f"Original result artifact changed since completion: {name}")
    for arm in ["Base", "B"]:
        for row in read_json(PREVIOUS / arm / "deployment_verification.json"):
            if digest(PREVIOUS / arm / f"seed_{row['seed']}/model.pt") != row["model_sha256"]:
                raise ValueError("Original Base/B exported model changed.")
    if expected is not None:
        for name, checksum in expected.items():
            if digest(ROOT / name) != checksum:
                raise ValueError(f"Original Base/B results changed: {name}")


def snapshot(output):
    parent.snapshot_sources(output)
    shutil.copy2(Path(__file__), output / "source" / Path(__file__).name)


def load_data(output):
    data = parent.load_data(output)
    for name in ["training_membership.csv", "development_membership.csv"]:
        if not pd.read_csv(output / "inputs" / name).equals(pd.read_csv(PREVIOUS / "inputs" / name)):
            raise AssertionError("Diagnostic data differs from reused Base/B data.")
    if data["manifest"]["normalization"] != read_json(PREVIOUS / "inputs/manifest.json")["normalization"]:
        raise AssertionError("Diagnostic normalization differs from Base/B.")
    return data


def smoke(device):
    rng = np.random.default_rng(43)
    training = rng.normal(size=(24, 840)).astype(np.float32)
    validation = rng.normal(size=(16, 840)).astype(np.float32)
    labels = np.tile([0, 0, 0, 1], 4)
    with tempfile.TemporaryDirectory(prefix="diagnostic_switches_") as temporary:
        for arm, flags in NEW_ARMS.items():
            directory = Path(temporary) / arm
            directory.mkdir()
            config = parent.implementation.configure(parent.small_config(), arm, 2026, device, directory)
            CORE.seed_everything(2026, 2)
            model = CORE.FGAAE(config).to(device)
            model.freeze_teacher()
            model.student.requires_grad_(False)
            model.critic.requires_grad_(flags["adversarial"])
            parent.implementation.calibrate_encoder_bn(CORE, model, training, 8, device)
            initial = FULL.snapshot(model)
            CORE.seed_everything(12026, 2)
            CORE.train_stage_two(model, training, validation, labels, np.arange(24), config, device, directory)
            changes = FULL.parameter_changes(model, initial)
            assert changes["encoder"] > 0 and changes["decoder"] > 0
            assert changes["teacher"] == changes["student"] == 0
            assert (changes["critic"] > 0) == flags["adversarial"]
            assert all(value == 0 for value in FULL.encoder_buffer_changes(FULL.snapshot(model), initial).values())
            history = pd.read_csv(directory / "stage2_history.csv")
            assert (history.retained_samples < len(training)).all() if flags["aqt"] else history.retained_samples.eq(len(training)).all()
            assert "validation_f1" not in history and "validation_recall" not in history
            parent.implementation.auc_scores(CORE, model, validation, config, device)
    print("DIAGNOSTIC SWITCH SMOKE PASSED: both new cases, component updates, frozen BN, AUC-only scoring.", flush=True)


def precheck(output, device):
    verify()
    verify_previous()
    output.mkdir(parents=True, exist_ok=False)
    snapshot(output)
    smoke(device)
    data = load_data(output)
    checks = []
    for old, new in [("Base", "AQT_only"), ("B", "GAN_only")]:
        old_model, old_config, _, old_provenance, _ = parent.prepare_model(old, 2026, data, device, output / f"init_{old}")
        new_model, new_config, metadata, new_provenance, _ = parent.prepare_model(new, 2026, data, device, output / f"init_{new}")
        assert all(torch.equal(value, new_model.state_dict()[name]) for name, value in old_model.state_dict().items())
        differing = [key for key in old_config if old_config[key] != new_config[key]]
        assert set(differing) == {"output", "use_aqt"}
        assert metadata["feature_mean"] is None and metadata["feature_std"] is None
        assert not old_provenance["expert_pretraining_loaded"] and not new_provenance["expert_pretraining_loaded"]
        directory = output / f"replay_{old}_2026_epoch1"
        directory.mkdir()
        old_config.update(iterations=1, epochs_per_iteration=1, output=str(directory))
        CORE.seed_everything(12026, 2)
        CORE.train_stage_two(old_model, data["training_norm"], data["development_norm"], data["labels"], data["source_rows"], old_config, device, directory)
        actual = pd.read_csv(directory / "stage2_history.csv")
        expected = pd.read_csv(PREVIOUS / old / "seed_2026/stage2_history.csv").iloc[:1]
        columns = [column for column in actual.columns if column in expected]
        difference = float((actual[columns] - expected[columns]).abs().max().max())
        if difference != 0:
            raise AssertionError(f"Existing {old} recipe does not replay: {difference}")
        checks.append({"reused_case": old, "new_case": new, "identical_initial_tensors": True,
                       "only_config_differences": differing, "original_epoch_replay_max_difference": difference})
        del old_model, new_model
    write_json(output / "precheck_report.json", {"passed": True, "smoke_passed": True,
               "reused_recipe_checks": checks, "code_sha256": code_hashes(),
               "frozen_manifest_sha256": digest(FROZEN / "baseline_manifest.json"),
               "previous_auc_runs_sha256": digest(PREVIOUS / "auc_runs.csv")})
    print("REAL BASE/B PARITY PASSED: original first epochs exact; new cases only toggle AQT against matched recipe.", flush=True)


def factorial_effects(table):
    rows = []
    for metric in ["auc_primary", "auc_reconstruction"]:
        pivot = table.pivot(index="seed", columns="arm", values=metric)
        for seed, values in pivot.iterrows():
            base, gan, aqt, both = (float(values[name]) for name in ORDER)
            effects = {"GAN_without_AQT": gan-base, "GAN_with_AQT": both-aqt,
                       "AQT_without_GAN": aqt-base, "AQT_with_GAN": both-gan,
                       "interaction": both-gan-aqt+base, "B_minus_Base": both-base}
            assert abs(effects["GAN_without_AQT"] + effects["AQT_without_GAN"] + effects["interaction"] - effects["B_minus_Base"]) < 1e-12
            rows.extend({"score": metric, "seed": int(seed), "effect": name, "auc_difference": value} for name, value in effects.items())
    return pd.DataFrame(rows)


def pool_diagnostics(output, data):
    labels = pd.read_csv(output / "inputs/training_membership.csv")[["source_row", "label"]]
    rows, overlaps = [], []
    for seed in parent.SEEDS:
        sets = {}
        for arm, location in [("AQT_only", output), ("B", PREVIOUS)]:
            for path in sorted((location / arm / f"seed_{seed}").glob("aqt_iteration_*.csv")):
                frame = pd.read_csv(path).merge(labels, on="source_row", validate="one_to_one")
                retained, removed = frame[frame.retained == 1], frame[frame.retained == 0]
                iteration = int(path.stem.rsplit("_", 1)[1])
                sets[arm, iteration] = set(retained.source_row)
                rows.append({"arm": arm, "seed": seed, "iteration": iteration, "retained": len(retained),
                             "original_injected_fraction": float(labels.label.mean()),
                             "retained_injected_fraction": float(retained.label.mean()),
                             "removed_injections": int(removed.label.sum()),
                             "removed_unmodified_background": int((removed.label == 0).sum())})
        for iteration in [1, 2, 3]:
            left, right = sets["AQT_only", iteration], sets["B", iteration]
            overlaps.append({"seed": seed, "iteration": iteration, "intersection": len(left & right),
                             "jaccard": len(left & right) / len(left | right), "different_members": len(left ^ right)})
    pd.DataFrame(rows).to_csv(output / "training_pool_diagnostics.csv", index=False)
    pd.DataFrame(overlaps).to_csv(output / "training_pool_overlap.csv", index=False)
    return pd.DataFrame(overlaps)


def summarize(output, data, rows):
    old = pd.read_csv(PREVIOUS / "auc_runs.csv")
    reused = old[old.arm.isin(["Base", "B"])].copy()
    reused["reused"] = True
    fresh = pd.DataFrame(rows)
    fresh["reused"] = False
    table = pd.concat([reused, fresh], ignore_index=True)
    table.to_csv(output / "auc_runs.csv", index=False)
    means = table.groupby("arm").agg(auc_mean=("auc_primary", "mean"), auc_sample_std=("auc_primary", "std"),
              reconstruction_auc_mean=("auc_reconstruction", "mean"), reconstruction_auc_sample_std=("auc_reconstruction", "std")).reindex(ORDER)
    means.to_csv(output / "auc_summary.csv")
    effects = factorial_effects(table)
    effects.to_csv(output / "conditional_effects_by_seed.csv", index=False)
    effect_summary = effects.groupby(["score", "effect"]).agg(mean=("auc_difference", "mean"),
                     sample_std=("auc_difference", "std"), seeds_positive=("auc_difference", lambda x: int((x > 0).sum()))).reset_index()
    effect_summary.to_csv(output / "conditional_effects_summary.csv", index=False)
    overlap = pool_diagnostics(output, data)
    primary = effect_summary[effect_summary.score == "auc_primary"].set_index("effect")
    lines = ["# B 与 Base 的补充诊断：无特征引导的 WGAN-GP × AQT", "",
             "这是观察到原五组实验 B 低于 Base 后追加的诊断实验。旧 Base/B 原样复用；只新增 AE＋WGAN-GP、AE＋AQT 各三个种子、每次 60 epoch。未为 B 单独调参，未重跑或替换旧结果。", "",
             "同一 20000 帧训练池、10592 帧既有开发集，种子 2025/2026/2027，沿用原编码器 LR、BN 校准/冻结、检查点和正评分系数校准规则。四组均无专家预训练、无教师/学生蒸馏；仅用 ROC-AUC 评价。", "",
             "| 组别 | WGAN-GP | AQT | 主 AUC | 统一重构 AUC | 来源 |", "| --- | --- | --- | --- | --- | --- |"]
    for arm, row in means.iterrows():
        flags = parent.implementation.ARMS[arm]
        lines.append(f"| {arm} | {'有' if flags['adversarial'] else '无'} | {'有' if flags['aqt'] else '无'} | {row.auc_mean:.6f} ± {row.auc_sample_std:.6f} | {row.reconstruction_auc_mean:.6f} ± {row.reconstruction_auc_sample_std:.6f} | {'旧结果复用' if arm in ['Base','B'] else '新增诊断'} |")
    lines += ["", "标准差描述三个初始化种子的波动，不是独立数据或跨站置信区间。", "",
              "| 条件效应 | 主 AUC 平均差值 | 配对差值样本标准差 | 正方向种子数 |", "| --- | --- | --- | --- |"]
    names = {"GAN_without_AQT": "无 AQT 时加入 WGAN-GP", "GAN_with_AQT": "有 AQT 时加入 WGAN-GP",
             "AQT_without_GAN": "无 WGAN-GP 时加入 AQT", "AQT_with_GAN": "有 WGAN-GP 时加入 AQT",
             "interaction": "AUC 差值尺度的组合交互项", "B_minus_Base": "B − Base"}
    for name in names:
        row = primary.loc[name]
        lines.append(f"| {names[name]} | {row['mean']:+.6f} | {row.sample_std:.6f} | {int(row.seeds_positive)}/3 |")
    lines += ["", "差值乘 100 为百分点。交互项定义为 B − GAN_only − AQT_only ＋ Base，只是当前 AUC 尺度的描述量，不直接证明损失冲突或网络机制。主 AUC 中 WGAN-GP 同时改变训练及 Critic 评分；统一重构列使用相同已选检查点，不另调模型。", "",
              "## 实现与训练设置排查", "",
              "旧 Base/B 的真实首轮训练已精确重放；两个新增组与对应旧组的全部初始张量一致，配置只在 AQT 开关和输出目录上不同。新组均通过关闭组件不更新、开启组件实际更新、无专家预训练残留、BN 校准不更新参数和阶段二 BN 冻结检查。未发现已检查路径上的实现不一致。", "",
              "相同 60 epoch 不等于相同更新次数：无 AQT 的 Base/GAN_only 每种子 4740 次 G 更新；有 AQT 的 AQT_only/B 为 4200 次。结果反映这个固定配方下启用模块的整体效应，不能把过滤质量、样本数量或优化预算的影响完全分开。", "",
              f"两个有 AQT 组首轮保留集合的 Jaccard：{overlap[overlap.iteration == 1].jaccard.tolist()}。训练池人工注入比例及未注入背景的移除量见 training_pool_diagnostics.csv。背景未经过人工正常性确认；这些数量不是测试集检出率。", "",
              "## 是否重跑 Base/B", "",
              "本轮只定位固定配置下的条件效应，不以 B 必须高于 Base 为验收条件。若仅观察到模块组合负收益而无实现错误，保留原结果；若后续有具体依据修改无特征引导训练配方，应以同一规则重新运行 Base/B，并把优化后的补充结果与原固定配置表分开。此次没有自动追加调参或替换主消融表。", "",
              "全部开发文件历史上均已查看，且原规则使用它们选择检查点及校准是否采用；本轮不是新独立测试，也不能从这四组推断含特征引导模型中同一模块必定有相同效果。", "",
              "文件：[逐种子 AUC](auc_runs.csv)、[条件效应](conditional_effects_by_seed.csv)、[配对效应图](conditional_effects.png)、[训练池统计](training_pool_diagnostics.csv)、[登记方案](registered_plan.json)。"]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot(output, effects)
    write_json(output / "report.json", {"complete": True, "post_hoc_diagnostic": True, "metric": "ROC-AUC",
               "new_runs": 6, "new_epochs": 360, "reused_cases": ["Base", "B"], "independent_test": False,
               "base_b_retrained": False, "summary": means.reset_index().to_dict("records"),
               "conditional_effects": effect_summary.to_dict("records")})
    print(means.to_string(), flush=True)
    print(effect_summary.to_string(index=False), flush=True)


def plot(output, effects):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["GAN_without_AQT", "GAN_with_AQT", "AQT_without_GAN", "AQT_with_GAN", "interaction"]
    titles = ["GAN\nwithout AQT", "GAN\nwith AQT", "AQT\nwithout GAN", "AQT\nwith GAN", "Interaction\non AUC scale"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, metric, title in zip(axes, ["auc_primary", "auc_reconstruction"], ["Primary ROC-AUC", "Reconstruction-only ROC-AUC"]):
        for index, effect in enumerate(labels):
            values = effects[(effects.score == metric) & (effects.effect == effect)].sort_values("seed").auc_difference.to_numpy()
            color = "#0072B2" if index < 2 else "#009E73" if index < 4 else "#777777"
            axis.bar(index, values.mean(), width=0.6, color=color, alpha=0.25)
            axis.errorbar(index, values.mean(), yerr=values.std(ddof=1), color=color, capsize=4)
            axis.scatter(index + np.array([-0.13, 0, 0.13]), values, color=color, edgecolors="white", zorder=3)
        axis.axhline(0, color="#555555", linestyle="--", linewidth=1)
        axis.set_xticks(range(5), titles, fontsize=8)
        axis.set(title=title, ylabel="Paired AUC difference")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("No feature guidance: three paired seeds; error bars = sample SD")
    figure.tight_layout()
    figure.savefig(output / "conditional_effects.png", dpi=200)
    figure.savefig(output / "conditional_effects.pdf")
    plt.close(figure)


def run(output, prechecked, device, resume):
    verify()
    verify_previous()
    check = read_json(prechecked / "precheck_report.json")
    if not check["passed"] or check["code_sha256"] != code_hashes() or check["previous_auc_runs_sha256"] != digest(PREVIOUS / "auc_runs.csv"):
        raise ValueError("Diagnostic precheck no longer matches source or reused results.")
    if resume:
        plan = read_json(output / "registered_plan.json")
        if plan["code_sha256"] != code_hashes():
            raise ValueError("Cannot resume with changed code.")
        verify_previous(plan["previous_artifacts_sha256"])
    else:
        output.mkdir(parents=True, exist_ok=False)
        snapshot(output)
        plan = {"scope": "Post-hoc diagnosis of B below Base; four no-feature cases, not an optimized replacement ablation table.",
                "new_arms": NEW_ARMS, "reused_arms": ["Base", "B"], "previous_run": str(PREVIOUS),
                "seeds": parent.SEEDS, "epochs_per_run": 60, "new_runs": 6, "metric": "ROC-AUC",
                "precheck": str(prechecked), "code_sha256": code_hashes(),
                "previous_artifacts_sha256": previous_artifacts(), "device": str(device),
                "same_training_recipe_and_calibration_as_original_ablation": True,
                "no_expert_pretraining": True, "bn_calibration_parameter_updates": 0,
                "conditional_effects": "GAN_only-Base, B-AQT_only, AQT_only-Base, B-GAN_only; interaction B-GAN_only-AQT_only+Base; repeat on R-only scores.",
                "rerun_policy": "No automatic Base/B reruns just to meet expected ranking. Preserve original results; report checked implementation consistency and fixed-recipe effects first.",
                "all_development_previously_inspected": True, "independent_test": False}
        write_json(output / "registered_plan.json", plan)
        write_json(output / "environment.json", {"torch": str(torch.__version__), "numpy": np.__version__, "pandas": pd.__version__,
                   "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")})
    if (output / "inputs").exists():
        with tempfile.TemporaryDirectory(prefix="diagnostic_inputs_") as temporary:
            data = load_data(Path(temporary))
    else:
        data = load_data(output)
    rows = []
    for arm in NEW_ARMS:
        directory = output / arm
        directory.mkdir(exist_ok=resume)
        for seed in parent.SEEDS:
            target = directory / f"seed_{seed}"
            if resume and (target / "training_report.json").exists():
                report = read_json(target / "training_report.json")
                if report.get("training_complete") and report["uncalibrated_model_sha256"] == digest(target / "uncalibrated_model.pt"):
                    print(f"REUSE COMPLETED {arm} seed={seed}", flush=True)
                    continue
            if target.exists():
                backup = directory / f"interrupted_seed_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if not target.resolve().is_relative_to(output.resolve()) or not backup.resolve().is_relative_to(output.resolve()):
                    raise ValueError("Interrupted files must remain within diagnostic output.")
                target.rename(backup)
            print(f"START DIAGNOSTIC {arm} seed={seed}", flush=True)
            parent.train_seed(arm, seed, data, device, target)
        rows.extend(parent.finalize_arm(arm, data, device, directory))
        pd.DataFrame(rows).to_csv(output / "new_auc_runs_so_far.csv", index=False)
    summarize(output, data, rows)
    verify_previous(plan["previous_artifacts_sha256"])
    verify()
    print(f"DIAGNOSTICS COMPLETE: {output / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["precheck", "train"])
    parser.add_argument("--output", type=Path, default=ROOT / "runs/ablation_auc_09117_diagnostic_v1")
    parser.add_argument("--precheck", type=Path, default=ROOT / "runs/ablation_auc_09117_diagnostic_precheck_v1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    device = torch.device(arguments.device)
    CORE.seed_everything(2025, 2)
    if arguments.command == "precheck":
        precheck(arguments.output.resolve(), device)
    else:
        run(arguments.output.resolve(), arguments.precheck.resolve(), device, arguments.resume)
