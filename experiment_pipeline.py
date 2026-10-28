from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch

import fgaae
from audit_data import IX_DTYPE, audit_fscan, full_digest, quality_flags
from prepare_dataset import grouped_partitions


ROOT = Path(__file__).resolve().parent
SNR_GRID = list(range(-14, 9, 2))
ARMS = [
    ("reconstruction_no_aqt", 0.0, False),
    ("reconstruction_aqt", 0.0, True),
    ("combined_no_aqt", 0.1, False),
    ("combined_aqt", 0.1, True),
]


def power_interference(values, source_rows, ratio, seed, fixed_snr=None):
    """Inject local power with a specified whole-band interference/background ratio, not waveform SNR."""
    generator = np.random.default_rng(seed)
    output = np.asarray(values, dtype=np.float32).copy()
    labels = np.zeros(len(output), dtype=np.int64)
    nominal_snr = np.full(len(output), np.nan)
    widths = np.zeros(len(output), dtype=np.int64)
    centers = np.full(len(output), -1, dtype=np.int64)
    count = min(len(output) - 1, max(1, round(len(output) * ratio))) if ratio else 0
    chosen = generator.permutation(len(output))[:count]
    assigned_snr = np.resize(np.asarray(SNR_GRID), count)
    generator.shuffle(assigned_snr)
    coordinates = np.arange(output.shape[1])
    for order, row_index in enumerate(chosen):
        snr = float(fixed_snr if fixed_snr is not None else assigned_snr[order])
        center = int(generator.integers(20, output.shape[1] - 20))
        width = int(generator.choice([2, 5, 10]))
        level_db = output[row_index].astype(np.float64) * 0.1
        reference_db = float(level_db.max())
        background = 10 ** np.clip((level_db - reference_db) / 10, -30, 0)
        shape = np.exp(-0.5 * ((coordinates - center) / width) ** 2)
        interference = shape * (background.sum() * 10 ** (snr / 10) / shape.sum())
        changed = (reference_db + 10 * np.log10(background + interference)) / 0.1
        output[row_index] = changed.astype(np.float32)
        labels[row_index] = 1
        nominal_snr[row_index], widths[row_index], centers[row_index] = snr, width, center
    metadata = pd.DataFrame({
        "source_row": source_rows, "label": labels, "nominal_interference_ratio_db": nominal_snr,
        "gaussian_sigma_points": widths, "center_index": centers,
        "label_origin": np.where(labels == 1, "synthetically_injected", "assumed_background_not_human_verified"),
    })
    return output, labels, metadata


def normalize(values, normalization):
    return ((np.asarray(values, dtype=np.float32) - normalization["mean"]) / normalization["std"]).astype(np.float32)


def create_benchmark(dataset: Path, output: Path, arguments):
    report = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    if full_digest(dataset / "spectra.npy") != report["spectra_sha256"]:
        raise ValueError("Prepared spectrum fingerprint changed.")
    if full_digest(dataset / "partitions.csv") != report["partitions_sha256"]:
        raise ValueError("Prepared split fingerprint changed.")
    values, _ = fgaae.read_spectra(dataset / "spectra.npy")
    partitions = pd.read_csv(dataset / "partitions.csv")
    if len(values) != len(partitions) or not np.array_equal(partitions.row, np.arange(len(values))):
        raise ValueError("Spectrum rows and partition metadata do not align.")
    if partitions.groupby("source_file").split.nunique().max() != 1:
        raise ValueError("A source file leaks across partitions.")
    if partitions.groupby("band_sha256").split.nunique().max() != 1:
        raise ValueError("An identical spectrum leaks across partitions.")
    generator = np.random.default_rng(arguments.seed)
    train_rows = generator.permutation(partitions.loc[partitions.split == "train", "row"].to_numpy())
    reference_count = arguments.reference_size
    if len(train_rows) < reference_count + arguments.pilot_size:
        raise ValueError("Insufficient training rows for reference data and the pilot experiment.")
    reference_rows, train_rows = train_rows[:reference_count], train_rows[reference_count:]
    reference_validation_count = max(2, round(reference_count * 0.25))
    feature_train_rows = reference_rows[reference_validation_count:]
    feature_validation_rows = reference_rows[:reference_validation_count]
    feature_training = values[feature_train_rows].astype(np.float32)
    normalization = {"mean": float(feature_training.astype(np.float64).mean()), "std": float(feature_training.astype(np.float64).std())}
    if normalization["std"] < 1e-8:
        raise ValueError("Reference data have zero variance.")
    output.mkdir()
    membership = partitions.copy()
    membership.loc[membership.row.isin(feature_train_rows), "split"] = "feature_train"
    membership.loc[membership.row.isin(feature_validation_rows), "split"] = "feature_validation"
    membership.to_csv(output / "membership.csv", index=False, encoding="utf-8-sig")
    np.save(output / "feature_train.npy", feature_training, allow_pickle=False)
    np.save(output / "feature_validation.npy", values[feature_validation_rows].astype(np.float32), allow_pickle=False)
    np.save(output / "training_order.npy", train_rows, allow_pickle=False)
    protocol = {
        "protocol_id": "local-power-injection-v1",
        "dataset": str(dataset.resolve()), "seed": arguments.seed,
        "dataset_sha256": report["spectra_sha256"], "partition_sha256": report["partitions_sha256"],
        "nominal_db_grid": SNR_GRID,
        "definition": "10*log10(sum(injected linear power)/sum(original linear power)) over the 840 frequency bins; Gaussian local power added in the linear domain.",
        "not_waveform_snr": True,
        "limitations": [
            "Raw spectra have no human normal/anomaly labels; unmodified backgrounds are assumed normal.",
            "The paper does not provide enough anomaly-generation detail to establish equivalence.",
            "This is a frequency-domain interference-ratio proxy, not an IQ/AWGN channel SNR reproduction.",
            "The benchmark intentionally differs from the previous seven-type synthetic experiment; their AUCs are not directly comparable.",
        ],
        "evaluation_ratio": arguments.anomaly_ratio, "training_ratio": arguments.train_anomaly_ratio,
        "normalization": normalization,
        "feature_training_rows": feature_train_rows.astype(int).tolist(),
        "feature_validation_rows": feature_validation_rows.astype(int).tolist(),
        "split_counts": membership.split.value_counts().to_dict(),
    }
    for split_name, source_rows, ratio, seed in [
        ("train", train_rows, arguments.train_anomaly_ratio, arguments.seed + 10),
        ("validation", partitions.loc[partitions.split == "validation", "row"].to_numpy(), arguments.anomaly_ratio, arguments.seed + 20),
        ("test", partitions.loc[partitions.split == "test", "row"].to_numpy(), arguments.anomaly_ratio, arguments.seed + 30),
        ("external_test", partitions.loc[partitions.split == "external_test", "row"].to_numpy(), arguments.anomaly_ratio, arguments.seed + 40),
    ]:
        if not len(source_rows):
            continue
        changed, labels, metadata = power_interference(values[source_rows], source_rows, ratio, seed)
        np.save(output / f"{split_name}.npy", changed, allow_pickle=False)
        np.save(output / f"{split_name}_labels.npy", labels, allow_pickle=False)
        metadata.to_csv(output / f"{split_name}_metadata.csv", index=False)
        protocol[f"{split_name}_sha256"] = full_digest(output / f"{split_name}.npy")
    fgaae.write_json(output / "protocol.json", protocol)
    return protocol


def pretrain(output, benchmark, config, normalization, device):
    output.mkdir()
    fgaae.seed_everything(config["seed"], config["num_threads"])
    model = fgaae.FGAAE(config).to(device)
    training = np.load(benchmark / "feature_train.npy", allow_pickle=False)
    validation = np.load(benchmark / "feature_validation.npy", allow_pickle=False)
    training_features = fgaae.expert_features(training, config).astype(np.float64)
    validation_features = fgaae.expert_features(validation, config).astype(np.float64)
    feature_mean = training_features.mean(axis=0)
    feature_std = training_features.std(axis=0)
    feature_std[feature_std < 1e-8] = 1
    train_loader = fgaae.make_loader(
        normalize(training, normalization), config["batch_size"],
        (training_features - feature_mean) / feature_std, shuffle=True,
    )
    validation_loader = fgaae.make_loader(
        normalize(validation, normalization), config["batch_size"],
        (validation_features - feature_mean) / feature_std,
    )
    fgaae.train_stage_one(model, train_loader, validation_loader, config, device, output)
    state = {
        "model_state": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "feature_mean": feature_mean.tolist(), "feature_std": feature_std.tolist(),
    }
    torch.save(state, output / "shared_initialization.pt")
    return state


def save_component_metrics(output, labels, result):
    rows = []
    for name, scores in [
        ("reconstruction", result["reconstruction_error"]),
        ("negative_critic", -result["critic_value"]),
        ("configured_combination", result["scores"]),
    ]:
        measurement = fgaae.metrics(labels, scores, float(np.quantile(scores[labels == 0], 0.99)))
        rows.append({"component": name, "roc_auc": measurement["roc_auc"], "average_precision": measurement["average_precision"]})
    pd.DataFrame(rows).to_csv(output / "validation_components.csv", index=False)


def run_arm(output, benchmark, shared_state, config, normalization, size, device):
    output.mkdir()
    fgaae.seed_everything(config["seed"], config["num_threads"])
    training_raw = np.load(benchmark / "train.npy", mmap_mode="r", allow_pickle=False)
    size = min(size, len(training_raw))
    training = normalize(training_raw[:size], normalization)
    validation = normalize(np.load(benchmark / "validation.npy", mmap_mode="r", allow_pickle=False), normalization)
    validation_labels = np.load(benchmark / "validation_labels.npy", allow_pickle=False)
    source_rows = np.load(benchmark / "training_order.npy", allow_pickle=False)[:size]
    reference = normalize(np.load(benchmark / "feature_validation.npy", allow_pickle=False), normalization)
    model = fgaae.FGAAE(config).to(device)
    model.load_state_dict(shared_state["model_state"])
    model.freeze_teacher()
    fgaae.write_json(output / "config.json", config)
    selection = fgaae.train_stage_two(
        model, training, validation, validation_labels, source_rows, config, device, output,
        score_reference=reference,
    )
    result = fgaae.evaluate(model, validation, config, device)
    threshold = fgaae.select_threshold(result["scores"], validation_labels, config)
    measurements = fgaae.metrics(validation_labels, result["scores"], threshold)
    metadata = pd.read_csv(benchmark / "validation_metadata.csv")
    fgaae.save_predictions(output / "validation_scores.csv", metadata.source_row.to_numpy(), validation_labels, None, result, threshold)
    save_component_metrics(output, validation_labels, result)
    report = {
        "training_samples": size, "selection": selection, "validation": measurements,
        "test_evaluated": False, "protocol": "local-power-injection-v1",
    }
    fgaae.write_json(output / "validation_report.json", report)
    fgaae.write_json(output / "selected_config.json", config)
    torch.save({
        "model_state": model.state_dict(), "config": config, "threshold": threshold,
        "normalization": normalization, "feature_names": fgaae.FEATURE_NAMES,
        "feature_mean": shared_state["feature_mean"], "feature_std": shared_state["feature_std"],
    }, output / "model.pt")
    training_labels = np.load(benchmark / "train_labels.npy", allow_pickle=False)[:size]
    truth = pd.DataFrame({"source_row": source_rows, "synthetic_label": training_labels})
    diagnostics = []
    for path in sorted(output.glob("aqt_iteration_*.csv")):
        selected = pd.read_csv(path).merge(truth, on="source_row", validate="one_to_one")
        retained = selected[selected.retained == 1]
        diagnostics.append({
            "iteration": int(path.stem.rsplit("_", 1)[1]), "original_anomaly_fraction": float(training_labels.mean()),
            "retained": len(retained), "retained_anomaly_fraction": float(retained.synthetic_label.mean()),
            "diagnostic_only_not_used_for_selection": True,
        })
    pd.DataFrame(diagnostics).to_csv(output / "purification_diagnostics.csv", index=False)
    del model, training, validation
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def final_evaluation(output, benchmark, dataset, normalization, device):
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = fgaae.FGAAE(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    report = {"protocol": "local-power-injection-v1", "not_original_paper_metrics": True}
    test_result = None
    for split_name in ["test", "external_test"]:
        path = benchmark / f"{split_name}.npy"
        if not path.exists():
            continue
        values = normalize(np.load(path, mmap_mode="r", allow_pickle=False), normalization)
        labels = np.load(benchmark / f"{split_name}_labels.npy", allow_pickle=False)
        metadata = pd.read_csv(benchmark / f"{split_name}_metadata.csv")
        result = fgaae.evaluate(model, values, config, device)
        report[split_name] = fgaae.metrics(labels, result["scores"], checkpoint["threshold"])
        fgaae.save_predictions(output / f"{split_name}_scores.csv", metadata.source_row.to_numpy(), labels, None, result, checkpoint["threshold"])
        if split_name == "test":
            test_result = (values, labels, result, metadata)
    if test_result is None:
        raise RuntimeError("The fixed test partition was not available.")
    values, labels, result, test_metadata = test_result
    training_history = output.parent / "shared_pretraining/stage1_history.csv"
    shutil.copyfile(training_history, output / "stage1_history.csv")
    model.eval()
    with torch.no_grad():
        reconstructed, _ = model(torch.from_numpy(values[:4]).unsqueeze(1).to(device))
    fgaae.plot_results(output, labels, result, values[:4], reconstructed.squeeze(1).cpu().numpy())
    partitions = pd.read_csv(dataset / "partitions.csv")
    source_values = np.load(dataset / "spectra.npy", mmap_mode="r", allow_pickle=False)
    source_rows = partitions.loc[partitions.split == "test", "row"].to_numpy()
    rows = []
    for nominal_db in SNR_GRID:
        changed, sweep_labels, _ = power_interference(
            source_values[source_rows], source_rows,
            float(json.loads((benchmark / "protocol.json").read_text(encoding="utf-8"))["evaluation_ratio"]),
            config["seed"] + 30, fixed_snr=nominal_db,
        )
        sweep_result = fgaae.evaluate(model, normalize(changed, normalization), config, device)
        rows.append({
            "nominal_interference_ratio_db": nominal_db,
            **fgaae.metrics(sweep_labels, sweep_result["scores"], checkpoint["threshold"]),
        })
    pd.DataFrame(rows).to_csv(output / "snr_proxy_sweep.csv", index=False)
    by_station = test_metadata.assign(score=result["scores"], label=labels).merge(
        partitions[["row", "station"]], left_on="source_row", right_on="row", validate="one_to_one",
    )
    station_rows = []
    for station, group in by_station.groupby("station"):
        station_rows.append({"station": station, **fgaae.metrics(group.label.to_numpy(), group.score.to_numpy(), checkpoint["threshold"])})
    pd.DataFrame(station_rows).to_csv(output / "test_station_metrics.csv", index=False, encoding="utf-8-sig")
    fgaae.write_json(output / "test_report.json", report)
    return report


def plot_comparison(output, ablations, scales):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(ablations.arm, ablations.validation_auc)
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("Validation ROC-AUC")
    axes[0].set_ylim(0, 1)
    axes[1].plot(scales.training_samples, scales.validation_auc, marker="o")
    axes[1].set_xlabel("Stage-two training samples")
    axes[1].set_ylabel("Validation ROC-AUC")
    figure.tight_layout()
    figure.savefig(output / "validation_comparison.png", dpi=150)
    plt.close(figure)


def run_pipeline(arguments):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    dataset = Path(arguments.dataset).resolve()
    output = Path(arguments.output).resolve()
    if min(arguments.pilot_size, arguments.middle_size, arguments.reference_size, arguments.batch_size) < 2:
        raise ValueError("Experiment sizes must be at least two.")
    if arguments.pilot_size > arguments.middle_size or arguments.reference_size < 8:
        raise ValueError("Require pilot <= middle size and at least eight reference rows.")
    if not 0 < arguments.anomaly_ratio < 1 or not 0 <= arguments.train_anomaly_ratio < 1:
        raise ValueError("Invalid synthetic anomaly ratios.")
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(arguments.device)
    fgaae.seed_everything(arguments.seed, arguments.num_threads)
    fgaae.write_json(output / "execution_plan.json", vars(arguments))
    protocol = create_benchmark(dataset, output / "benchmark", arguments)
    benchmark = output / "benchmark"
    defaults = vars(fgaae.build_parser().parse_args(["train"]))
    config = {
        **defaults, "command": "pipeline", "data": str(benchmark / "train.npy"),
        "sequence_length": 840, "seed": arguments.seed, "device": str(device),
        "num_threads": arguments.num_threads, "stage1_epochs": arguments.stage1_epochs,
        "iterations": arguments.iterations, "epochs_per_iteration": arguments.epochs_per_iteration,
        "batch_size": arguments.batch_size, "critic_steps": arguments.critic_steps,
        "early_stopping_patience": arguments.patience, "minimum_epochs": arguments.minimum_epochs,
        "warmup_iterations": 1, "standardize_scores": True,
        "retention_start": 0.95, "retention_end": 0.8,
        "threshold_method": "validation-f1", "evaluation": "synthetic",
    }
    fgaae.validate_config(config)
    fgaae.write_json(output / "environment.json", {
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "numpy": np.__version__, "pandas": pd.__version__,
    })
    shared_state = pretrain(output / "shared_pretraining", benchmark, config, protocol["normalization"], device)
    ablation_rows = []
    for name, critic_weight, use_aqt in ARMS:
        arm_config = {**config, "critic_score_weight": critic_weight, "use_aqt": use_aqt, "output": str(output / name)}
        print(f"START ABLATION: {name}", flush=True)
        result = run_arm(output / name, benchmark, shared_state, arm_config, protocol["normalization"], arguments.pilot_size, device)
        ablation_rows.append({
            "arm": name, "critic_score_weight": critic_weight, "use_aqt": use_aqt,
            "complexity": int(critic_weight > 0) + int(use_aqt),
            "validation_auc": result["validation"]["roc_auc"],
            "validation_f1": result["validation"]["f1"], "best_epoch": result["selection"]["best_epoch"],
            "epochs_ran": result["selection"]["epochs_ran"], "training_samples": result["training_samples"],
        })
        pd.DataFrame(ablation_rows).to_csv(output / "ablation_comparison.csv", index=False)
    ablations = pd.DataFrame(ablation_rows)
    winner = ablations.sort_values(["validation_auc", "validation_f1", "complexity", "arm"], ascending=[False, False, True, True]).iloc[0]
    fgaae.write_json(output / "validation_selected_arm.json", {
        "arm": str(winner.arm), "validation_auc": float(winner.validation_auc),
        "critic_score_weight": float(winner.critic_score_weight), "use_aqt": bool(winner.use_aqt),
        "selected_without_test_access": True,
    })
    print(f"SELECTED BY VALIDATION ONLY: {winner.arm}", flush=True)
    scale_rows = [{
        "scale": "pilot", "training_samples": int(winner.training_samples),
        "validation_auc": float(winner.validation_auc), "validation_f1": float(winner.validation_f1),
    }]
    for name, size in [
        ("scale_20000", arguments.middle_size),
        ("scale_full", len(np.load(benchmark / "training_order.npy", allow_pickle=False))),
    ]:
        scaled_config = {
            **config, "critic_score_weight": float(winner.critic_score_weight),
            "use_aqt": bool(winner.use_aqt), "output": str(output / name),
        }
        print(f"START SCALE: {name}, samples={size}", flush=True)
        report = run_arm(output / name, benchmark, shared_state, scaled_config, protocol["normalization"], size, device)
        scale_rows.append({"scale": name, "training_samples": report["training_samples"], "validation_auc": report["validation"]["roc_auc"], "validation_f1": report["validation"]["f1"]})
        pd.DataFrame(scale_rows).to_csv(output / "scale_comparison.csv", index=False)
    test_report = final_evaluation(output / "scale_full", benchmark, dataset, protocol["normalization"], device)
    scales = pd.DataFrame(scale_rows)
    plot_comparison(output, ablations, scales)
    fgaae.write_json(output / "final_report.json", {
        "evaluation": "synthetic local-power-injection-v1, not original paper metrics",
        "dataset_total_rows": sum(protocol["split_counts"].values()),
        "selected_arm": str(winner.arm), "ablation_validation_results": ablation_rows,
        "scale_validation_results": scale_rows, "full_scale_test": test_report,
        "test_evaluation_policy": "Only the full-size checkpoint selected by validation is evaluated on test; no pilot test scores used.",
        "snr_warning": "The -14..8 dB sweep is an injected-power/background proxy, not a validated reproduction of the paper's channel SNR.",
    })
    print(json.dumps(test_report, ensure_ascii=False, indent=2), flush=True)
    print(f"PIPELINE COMPLETE: {output}", flush=True)
    return output


def smoke_test():
    with tempfile.TemporaryDirectory(prefix="fgaae_pipeline_") as temporary:
        directory = Path(temporary)
        dataset = directory / "dataset"
        dataset.mkdir()
        generator = np.random.default_rng(2025)
        matrix = generator.normal(150, 12, (400, 840)).astype(np.int16)
        metadata_rows = []
        for index in range(len(matrix)):
            group = index // 16
            station = f"station_{group // 4}"
            metadata_rows.append({
                "row": index, "station": station, "source_file": f"file_{group:02d}",
                "ticks": 638839000000000000 + group * 10000000 + index,
                "band_sha256": hashlib.sha256(matrix[index].tobytes()).hexdigest(),
            })
        metadata = pd.DataFrame(metadata_rows)
        partitions = grouped_partitions(metadata)
        np.save(dataset / "spectra.npy", matrix)
        metadata.to_csv(dataset / "metadata.csv", index=False)
        partitions.to_csv(dataset / "partitions.csv", index=False)
        fgaae.write_json(dataset / "dataset.json", {
            "spectra_sha256": full_digest(dataset / "spectra.npy"),
            "partitions_sha256": full_digest(dataset / "partitions.csv"),
        })
        fixture = directory / "fixture"
        fixture.mkdir()
        records = np.zeros(2, dtype=IX_DTYPE)
        records["ticks"] = 638839000000000000
        records["declared_points"] = 840
        records["start"] = 87000000
        records["end"] = 107975000
        records["step"] = 25000
        records["count"] = [840, 100]
        records["offset_points"] = [0, 840]
        records["offset_bytes"] = [0, 1680]
        records.tofile(fixture / "example.ix")
        np.full(940, 100, dtype="<i2").tofile(fixture / "example.md")
        _, bands, _ = audit_fscan(fixture / "example.ix", directory)
        assert np.all(bands[1] == 0)
        assert quality_flags(bands)["all_zero"][1]
        arguments = build_parser().parse_args([
            "--dataset", str(dataset), "--output", str(directory / "run"),
            "--device", "cpu", "--num-threads", "2", "--reference-size", "40",
            "--pilot-size", "64", "--middle-size", "96", "--stage1-epochs", "1",
            "--iterations", "2", "--epochs-per-iteration", "1", "--patience", "1",
            "--minimum-epochs", "2", "--batch-size", "32", "--anomaly-ratio", "0.25",
        ])
        output = run_pipeline(arguments)
        assert len(pd.read_csv(output / "ablation_comparison.csv")) == 4
        assert len(pd.read_csv(output / "scale_full/snr_proxy_sweep.csv")) == len(SNR_GRID)
        for name, _, _ in ARMS:
            assert not (output / name / "test_scores.csv").exists()
        for name in ["scale_full", "scale_20000"]:
            selection = json.loads((output / name / "selection.json").read_text())
            history = pd.read_csv(output / name / "stage2_history.csv")
            best_value = history.loc[history.epoch == selection["best_epoch"], "validation_roc_auc"].iloc[0]
            assert abs(best_value - selection["best_validation_selection_score"]) < 1e-8
        print("PIPELINE SMOKE TEST PASSED: extraction guards, file-grouped splits, frozen benchmark, four ablations, score calibration, best checkpoints, scaling and final-only test evaluation.")


def build_parser():
    parser = argparse.ArgumentParser(description="Fixed grouped benchmark, four score/AQT ablations, nested scaling and final test evaluation.")
    parser.add_argument("--dataset", default=str(ROOT / "datasets/fm_87_108_v1"))
    parser.add_argument("--output", default=str(ROOT / "runs/grouped_pipeline_v1"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--pilot-size", type=int, default=5000)
    parser.add_argument("--middle-size", type=int, default=20000)
    parser.add_argument("--reference-size", type=int, default=800)
    parser.add_argument("--stage1-epochs", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--epochs-per-iteration", type=int, default=5)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--minimum-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--critic-steps", type=int, default=1)
    parser.add_argument("--anomaly-ratio", type=float, default=0.1)
    parser.add_argument("--train-anomaly-ratio", type=float, default=0.05)
    parser.add_argument("--smoke-test", action="store_true")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.smoke_test:
        smoke_test()
    else:
        run_pipeline(parsed)
