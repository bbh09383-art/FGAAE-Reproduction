"""Original FGAAE only: paired forward-state pilots and F1 checkpoint calibration."""
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

import fgaae
import full_model
import summarize_full_model
from audit_data import full_digest
from calibrate_full_model import coefficient_subsets

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "runs/full_fgaae_improvement_round1/full_loss_verification"
RELATIVES = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def measurements(run, calibrated=False):
    metadata = pd.read_csv(run / "inputs/development_membership.csv")
    mask, _, _ = coefficient_subsets(metadata)
    plan = read_json(run / "registered_plan.json")
    rows = []
    for seed in plan["seeds"]:
        model_path = run / (f"score_calibration/seed_{seed}_model.pt" if calibrated else f"seed_{seed}/model.pt")
        score_path = run / (f"score_calibration/seed_{seed}_development_scores.csv" if calibrated else f"seed_{seed}/development_scores.csv")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        frame = pd.read_csv(score_path, float_precision="round_trip")
        if not np.array_equal(frame.source_row, metadata.source_row):
            raise ValueError("Score membership mismatch.")
        for subset, keep in [("selection_earlier", mask), ("check_later", ~mask), ("full_development", np.ones(len(mask), bool))]:
            row = {"seed": seed, "subset": subset, "critic_weight": checkpoint["config"]["critic_score_weight"],
                   **fgaae.metrics(metadata.label.to_numpy()[keep], frame.anomaly_score.to_numpy()[keep], checkpoint["threshold"])}
            tn, fp, _, _ = row["confusion_matrix_tn_fp_fn_tp"]
            row["false_positive_rate"] = fp / max(tn + fp, 1)
            rows.append(row)
    return pd.DataFrame(rows)


def make_arguments(plan, output, consistent, seeds, epochs, iterations):
    arguments = full_model.build_parser().parse_args(["train"])
    for key, value in plan.items():
        if hasattr(arguments, key):
            setattr(arguments, key, value)
    arguments.output = str(output)
    arguments.pretrained_run = str(BASELINE)
    arguments.training_profile = "original_fgaae_f1_shared_forward" if consistent else "original_fgaae_f1_control"
    arguments.consistent_generator_forward = consistent
    arguments.checkpoint_selection_metric = "f1"
    arguments.selection_earlier_files = True
    arguments.selection_relative_critic_weights = RELATIVES
    arguments.seeds = seeds
    arguments.iterations = iterations
    arguments.epochs_per_iteration = epochs
    return arguments


def verify_deployment(run):
    plan = read_json(run / "registered_plan.json")
    raw = np.load(Path(plan["station_run"]) / "inputs/development.npy", mmap_mode="r")
    metadata = pd.read_csv(run / "inputs/development_membership.csv")
    records = []
    for seed in plan["seeds"]:
        checkpoint = torch.load(run / f"seed_{seed}/model.pt", map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        model = fgaae.FGAAE(config).to(plan["device"])
        model.load_state_dict(checkpoint["model_state"])
        result = fgaae.evaluate(model, full_model.normalized(raw, checkpoint["normalization"]), config, torch.device(plan["device"]))
        saved = pd.read_csv(run / f"seed_{seed}/development_scores.csv", float_precision="round_trip")
        difference = float(np.max(np.abs(saved.anomaly_score.to_numpy() - result["scores"])))
        if not np.array_equal(saved.predicted_label.to_numpy(), (result["scores"] > checkpoint["threshold"]).astype(int)):
            raise AssertionError("Reloaded decisions differ.")
        if difference > 1e-5:
            raise AssertionError(f"Reloaded scores differ: {difference}")
        selected_files = config["checkpoint_selection_files"]
        mask = metadata.source_file.isin(selected_files).to_numpy()
        threshold = fgaae.select_threshold(result["scores"][mask], metadata.label.to_numpy()[mask], config)
        if threshold != checkpoint["threshold"]:
            raise AssertionError("Threshold does not come from registered earlier files.")
        history = pd.read_csv(run / f"seed_{seed}/stage2_history.csv")
        selection = read_json(run / f"seed_{seed}/selection.json")
        best = history[history.epoch == selection["best_epoch"]].iloc[0]
        if abs(best.selection_critic_weight - config["critic_score_weight"]) > 1e-8:
            raise AssertionError("Wrong epoch's coefficient restored.")
        records.append({"seed": seed, "frames": len(raw), "max_score_difference": difference,
                        "decisions_identical": True, "threshold_from_earlier_files": True,
                        "model_sha256": full_digest(run / f"seed_{seed}/model.pt")})
    fgaae.write_json(run / "deployment_verification.json", records)


def run_round(output, pilot_epochs=20, resume=False):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    plan = read_json(BASELINE / "registered_plan.json")
    output.mkdir(parents=True, exist_ok=resume)
    source = output / "source"
    if not resume:
        source.mkdir()
        for name in ["fgaae.py", "full_model.py", "original_fidelity_round.py", "summarize_full_model.py"]:
            shutil.copy2(ROOT / name, source / name)
    else:
        previous_policy = read_json(output / "registered_plan.json")
        if previous_policy["pilot_epochs"] != pilot_epochs or previous_policy["relative_coefficients"] != RELATIVES:
            raise ValueError("Resume cannot change the registered experiment.")
        for name in ["fgaae.py", "full_model.py"]:
            if full_digest(ROOT / name) != previous_policy["source_sha256"][name]:
                raise ValueError("Resume cannot change training code.")
        resumed_source = output / "resume_source"
        resumed_source.mkdir(exist_ok=False)
        for name in ["original_fidelity_round.py", "summarize_full_model.py"]:
            shutil.copy2(ROOT / name, resumed_source / name)
        fgaae.write_json(output / "resume_provenance.json", {
            "reason": "Completed shared-forward pilot failed CPU-vs-default-CUDA score tolerance. Diagnosed cuDNN TF32: full CUDA replay is exact, all 10592 CPU decisions match. Add explicit CPU/CUDA FP32 parity check without changing original CUDA scores or training.",
            "training_code_unchanged": True, "registered_policy_unchanged": True,
            "original_source_snapshot_preserved": True,
            "resume_source_sha256": {path.name: full_digest(path) for path in resumed_source.iterdir()},
        })
    policy = {
        "scope": "Original complete FGAAE only; no residual enhancement, new modules, new losses or ablations.",
        "baseline": str(BASELINE), "pilot_seed": 2026, "pilot_epochs": pilot_epochs,
        "pilot_arms": ["control_eval_train", "shared_train_forward"],
        "score": "sum squared reconstruction residual + positive lambda * C(reconstruction)",
        "coefficients": "Registered lambda=0.1 plus k * training-reference std(reconstruction sum)/std(critic)",
        "relative_coefficients": RELATIVES,
        "selection": "Joint epoch and positive coefficient by maximum binary F1 on earlier four development files. Ties keep first candidate; checkpoint improvement must exceed 1e-4.",
        "threshold": "Earlier four development files only; fixed on later four files.",
        "aqt_score": "Fixed original lambda=0.1 during all training, unaffected by checkpoint coefficient selection.",
        "full_gate": "Pilot earlier-file F1 improves over baseline seed 2026 by >=0.01, recall does not decrease by more than 0.01; choose higher earlier-file F1 among eligible arms.",
        "full_budget": "Three original seeds, 3 AQT iterations x 20 epochs; reused identical feature pretraining.",
        "data": "Same 20000 stage-two training rows and frozen 10592 development rows; no changes to labels, injection, class ratio or nominal power distribution.",
        "caveat": "All development files have been inspected in historical work. This is adaptive development tuning, not a new independent test.",
        "confirmation_test_external_accessed": False,
        "source_sha256": {path.name: full_digest(path) for path in source.iterdir()},
    }
    if not resume:
        fgaae.write_json(output / "registered_plan.json", policy)
    baseline = measurements(BASELINE, calibrated=True)
    baseline.to_csv(output / "baseline_metrics.csv", index=False)
    reference = baseline[(baseline.seed == 2026) & (baseline.subset == "selection_earlier")].iloc[0]
    summaries, metrics = [], [baseline.assign(variant="previous_original_calibrated")]
    for name, consistent in [("control_eval_train", False), ("shared_train_forward", True)]:
        directory = output / name
        if not (resume and (directory / "full_model_report.json").exists()):
            full_model.run(make_arguments(plan, directory, consistent, [2026], pilot_epochs, 1))
        summarize_full_model.summarize(directory)
        verify_deployment(directory)
        frame = measurements(directory)
        metrics.append(frame.assign(variant=name))
        selected = frame[frame.subset == "selection_earlier"].iloc[0]
        passes = bool(selected.f1 >= reference.f1 + 0.01 and selected.recall >= reference.recall - 0.01)
        summaries.append({"variant": name, "consistent": consistent, "selection_f1": float(selected.f1),
                          "selection_recall": float(selected.recall), "passes_full_gate": passes})
        pd.DataFrame(summaries).to_csv(output / "pilot_summary.csv", index=False)
        pd.concat(metrics).to_csv(output / "all_metrics.csv", index=False)
    eligible = [row for row in summaries if row["passes_full_gate"]]
    verified = None
    if eligible:
        winner = max(eligible, key=lambda row: row["selection_f1"])
        verified = output / "full_verification"
        full_model.run(make_arguments(plan, verified, winner["consistent"], plan["seeds"], 20, 3))
        summarize_full_model.summarize(verified)
        verify_deployment(verified)
        metrics.append(measurements(verified).assign(variant="full_verification"))
    combined = pd.concat(metrics, ignore_index=True)
    combined.to_csv(output / "all_metrics.csv", index=False)
    numeric = ["roc_auc", "f1", "precision", "recall", "false_positive_rate", "average_precision"]
    aggregate = combined.groupby(["variant", "subset"], sort=False)[numeric].mean().reset_index()
    aggregate.to_csv(output / "mean_metrics.csv", index=False)
    report = {"pilots": summaries, "full_verification": str(verified) if verified else None,
              "adaptive_development_only": True, "independent_test": False,
              "original_architecture_loss_score_form_preserved": True,
              "paper_performance_reproduced": False, "means": aggregate.to_dict("records")}
    fgaae.write_json(output / "report.json", report)
    lines = ["# 原模型 F1 与召回率复现试验", "", "保留完整 FGAAE、原损失项和正号 Critic 论文评分形式；未采用局部残差增强。", "",
             "相同数据、异常分布及特征预训练。较早四个开发文件选择检查点、正评分系数及阈值；较晚四文件只作检查。所有文件历史上均已查看，本轮没有新的独立测试。", "",
             "| 方案 | 子集 | AUC | F1 | Precision | Recall |", "| --- | --- | --- | --- | --- | --- |"]
    for row in aggregate.itertuples():
        lines.append(f"| {row.variant} | {row.subset} | {row.roc_auc:.6f} | {row.f1:.6f} | {row.precision:.6f} | {row.recall:.6f} |")
    lines += ["", f"三种子完整验证：{'已执行，见 full_verification/' if verified else '短程试验未通过预先设定的扩展门槛，保留现有原模型基准。'}", "",
              "论文图中 F1/Recall 约 0.845/0.835，仅为读图值；数据、SNR 与统计口径不能直接等同。本地结果不能作为原论文原始指标。", "",
              "详细种子结果见 all_metrics.csv，训练日志与选中轮次见各 seed 目录，模型保存/加载核验见 deployment_verification.json。"]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def smoke_test():
    """Verify graph isolation, BN update counts, checkpoint restoration and label isolation."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    fgaae.seed_everything(29, 2)
    config = vars(fgaae.build_parser().parse_args(["train"]))
    config.update(sequence_length=840, batch_size=8, iterations=1, epochs_per_iteration=2,
                  critic_steps=2, early_stopping_patience=0, consistent_generator_forward=True,
                  checkpoint_selection_metric="f1", freeze_encoder_bn=True, encoder_lr=2e-6,
                  reconstruction_score_reduction="sum", critic_score_sign=1, critic_score_weight=0.1)
    model = fgaae.FGAAE(config)
    model.freeze_teacher()
    before = full_model.snapshot(model)
    counters, handles = {"generator_train_forwards": 0, "critic_grad_isolated": True}, []
    def decoder_hook(module, inputs, outputs):
        if module.training:
            counters["generator_train_forwards"] += 1
    def critic_hook(module, inputs):
        if module.training and torch.is_grad_enabled():
            counters["critic_grad_isolated"] &= all(p.grad is None for p in model.encoder.parameters())
            counters["critic_grad_isolated"] &= all(p.grad is None for p in model.decoder.parameters())
    handles.append(model.decoder.register_forward_hook(decoder_hook))
    handles.append(model.critic.register_forward_pre_hook(critic_hook))
    rng = np.random.default_rng(5)
    training = rng.normal(size=(16, 840)).astype(np.float32)
    labels = np.tile([0, 0, 0, 1], 4)
    call = 0
    def selector(current, result):
        nonlocal call
        call += 1
        return {"score": 1 / call, "config_updates": {"critic_score_weight": float(call)}, "record": {"selection_f1": 1 / call}}
    with tempfile.TemporaryDirectory(prefix="original_fgaae_smoke_") as temporary:
        output = Path(temporary)
        selection = fgaae.train_stage_two(model, training, training, labels, np.arange(16), config, torch.device("cpu"), output, checkpoint_selector=selector)
        latest = torch.load(output / "stage2_latest.pt", weights_only=True)
        for name, value in latest["model_state"].items():
            if name.startswith("decoder.") and name.endswith("num_batches_tracked"):
                assert int(value - before[name]) == 4
        assert selection["best_epoch"] == 1 and config["critic_score_weight"] == 1.0
        assert torch.load(output / "stage2_best.pt", weights_only=True)["config"]["critic_score_weight"] == 1.0
    for handle in handles:
        handle.remove()
    assert counters["generator_train_forwards"] == 4 and counters["critic_grad_isolated"]
    assert all(value == 0 for value in full_model.encoder_buffer_changes(full_model.snapshot(model), before).values())
    changes = full_model.parameter_changes(model, before)
    assert changes["teacher"] == 0 and all(changes[name] > 0 for name in ["encoder", "decoder", "student", "critic"])
    mask = np.arange(16) < 8
    components = {"reconstruction_score": rng.random(16), "critic_value": rng.random(16)}
    reference = {"reconstruction_score": rng.random(32), "critic_value": rng.random(32)}
    original = full_model.select_original_score(components, reference, labels, mask, RELATIVES)
    changed = labels.copy()
    changed[~mask] = 1 - changed[~mask]
    assert original == full_model.select_original_score(components, reference, changed, mask, RELATIVES)
    print("ORIGINAL FIDELITY SMOKE TEST PASSED: one train forward/batch, detached critic graph, all components updated, frozen teacher/encoder BN, F1 checkpoint/coefficient restoration, later-label isolation.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/original_fidelity_round1")
    parser.add_argument("--pilot-epochs", type=int, default=20)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed pilots with unchanged registered training code/policy.")
    arguments = parser.parse_args()
    if arguments.smoke_test:
        smoke_test()
    else:
        run_round(arguments.output.resolve(), arguments.pilot_epochs, arguments.resume)
