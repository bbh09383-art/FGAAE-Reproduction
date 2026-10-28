from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import random
import re
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
from scipy.signal import find_peaks
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import autograd, nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = (
    ROOT.parent
    / "feature_extraction&illegal_detect"
    / "data/cut_md_csv/all_md_files_cutted"
)
FEATURE_NAMES = [
    "mean_level", "std_level", "min_level", "max_level", "median_level",
    "rms_level", "level_range", "skewness", "excess_kurtosis",
    "power_centroid_mhz", "power_spread_mhz", "power_entropy", "power_flatness",
    "log_total_power", "peak_count", "dominant_peak_freq_mhz",
    "dominant_peak_level", "dominant_peak_prominence", "dominant_peak_width_mhz",
    "mean_peak_spacing_mhz", "std_peak_spacing_mhz", "occupancy",
    "lower_band_power_fraction", "middle_band_power_fraction",
    "upper_band_power_fraction",
]


def seed_everything(seed: int, num_threads: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if num_threads:
        torch.set_num_threads(num_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def read_spectra(path: Path, label_column: str = "label") -> tuple[np.ndarray, np.ndarray | None]:
    if path.suffix.lower() == ".npy":
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.ndim != 2 or values.shape[1] < 16:
            raise ValueError("Expected a two-dimensional numeric spectrum NPY array.")
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError("NPY spectra must have a numeric dtype.")
        for start in range(0, len(values), 4096):
            if not np.isfinite(values[start:start + 4096]).all():
                raise ValueError("NPY spectra contain non-finite values.")
        return values, None
    with path.open(encoding="utf-8-sig") as handle:
        first_line = next((line.strip() for line in handle if line.strip()), "")
    if not first_line:
        raise ValueError(f"Empty dataset: {path}")
    try:
        [float(value) for value in re.split(r"[,\s]+", first_line)]
        header = None
    except ValueError:
        header = 0
    frame = pd.read_csv(path, sep=r"[,\s]+", engine="python", header=header)
    labels = None
    if header == 0:
        frame.columns = [str(column).strip() for column in frame.columns]
        if label_column in frame:
            label_values = pd.to_numeric(frame.pop(label_column), errors="raise").to_numpy()
            if not np.isin(label_values, [0, 1]).all():
                raise ValueError("Labels must be 0 (normal) or 1 (anomaly).")
            labels = label_values.astype(np.int64)
        metadata_columns = [
            column for column in frame.columns
            if column.lower() in {"index", "original_index"}
            or column.lower().startswith("unnamed:")
        ]
        frame = frame.drop(columns=metadata_columns)
    values = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 16 or not np.isfinite(values).all():
        raise ValueError("Expected finite numeric spectra with at least 16 points per row.")
    return values, labels


def expert_features(spectra: np.ndarray, config: dict) -> np.ndarray:
    """Extract an explicit 25-feature surrogate; the paper does not specify its feature list."""
    sequence_length = spectra.shape[1]
    frequencies = np.linspace(
        config["start_mhz"], config["end_mhz"], sequence_length, endpoint=False
    )
    frequency_step = (config["end_mhz"] - config["start_mhz"]) / sequence_length
    feature_rows = []
    for spectrum in spectra:
        levels = spectrum.astype(np.float64) * config["level_scale"]
        centered = levels - levels.mean()
        standard_deviation = max(float(levels.std()), 1e-8)
        if config["units"] == "db":
            relative_power = 10.0 ** np.clip((levels - levels.max()) / 10.0, -30, 0)
            log_total_power = float(levels.max() + 10 * np.log10(relative_power.sum()))
            occupied = levels > np.median(levels) + config["occupancy_margin"]
        else:
            if np.any(levels < 0):
                raise ValueError("Linear power input cannot contain negative values.")
            relative_power = np.maximum(levels, 1e-12)
            log_total_power = float(10 * np.log10(relative_power.sum()))
            occupied = levels > np.median(levels) + 2 * standard_deviation
        probabilities = relative_power / relative_power.sum()
        centroid = float(np.dot(probabilities, frequencies))
        spread = float(np.sqrt(np.dot(probabilities, (frequencies - centroid) ** 2)))
        peaks, properties = find_peaks(levels, prominence=config["peak_prominence"])
        dominant_index = int(np.argmax(levels))
        dominant_prominence = 0.0
        if len(peaks):
            strongest = int(np.argmax(levels[peaks]))
            dominant_index = int(peaks[strongest])
            dominant_prominence = float(properties["prominences"][strongest])
        half_power_level = (
            levels[dominant_index] - 3.0
            if config["units"] == "db" else levels[dominant_index] * 0.5
        )
        left_index = right_index = dominant_index
        while left_index > 0 and levels[left_index] > half_power_level:
            left_index -= 1
        while right_index < sequence_length - 1 and levels[right_index] > half_power_level:
            right_index += 1
        spacings = np.diff(frequencies[peaks])
        subband_power = [float(part.sum()) for part in np.array_split(probabilities, 3)]
        feature_rows.append([
            float(levels.mean()), float(levels.std()), float(levels.min()),
            float(levels.max()), float(np.median(levels)), float(np.sqrt(np.mean(levels ** 2))),
            float(np.ptp(levels)), float(np.mean((centered / standard_deviation) ** 3)),
            float(np.mean((centered / standard_deviation) ** 4) - 3), centroid, spread,
            float(-np.sum(probabilities * np.log(probabilities.clip(1e-30)))),
            float(np.exp(np.log(relative_power.clip(1e-30)).mean()) / relative_power.mean()),
            log_total_power, float(len(peaks)), float(frequencies[dominant_index]),
            float(levels[dominant_index]), dominant_prominence,
            float((right_index - left_index) * frequency_step),
            float(spacings.mean()) if len(spacings) else 0.0,
            float(spacings.std()) if len(spacings) else 0.0,
            float(occupied.mean()), *subband_power,
        ])
    features = np.asarray(feature_rows, dtype=np.float32)
    if features.shape[1] != len(FEATURE_NAMES) or not np.isfinite(features).all():
        raise ValueError("Expert features contain invalid values.")
    return features


def inject_anomalies(
    spectra: np.ndarray, ratio: float, seed: int, severity: float, units: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    generator = np.random.default_rng(seed)
    modified = spectra.copy()
    labels = np.zeros(len(spectra), dtype=np.int64)
    descriptions = ["normal"] * len(spectra)
    if ratio == 0:
        return modified, labels, descriptions
    anomaly_count = min(len(spectra) - 1, max(1, round(len(spectra) * ratio)))
    anomaly_types = ["spike", "band_energy", "drift", "harmonic", "missing", "saturation", "step"]
    for row_index in generator.choice(len(spectra), anomaly_count, replace=False):
        original = spectra[row_index].astype(np.float64)
        signal = original.copy()
        deviation = max(float(signal.std()), float(np.ptp(signal)) * 0.05, 1e-3)
        amplitude = deviation * severity
        anomaly_type = str(generator.choice(anomaly_types))
        width = max(2, round(len(signal) * generator.uniform(0.03, 0.15)))
        start_index = int(generator.integers(0, len(signal) - width + 1))
        if anomaly_type == "spike":
            position = int(generator.integers(0, len(signal)))
            signal += 8 * amplitude * np.exp(-0.5 * ((np.arange(len(signal)) - position) / 2) ** 2)
        elif anomaly_type == "band_energy":
            signal[start_index:start_index + width] += 4 * amplitude
        elif anomaly_type == "drift":
            signal += np.linspace(0, 4 * amplitude, len(signal))
        elif anomaly_type == "harmonic":
            cycles = generator.uniform(2, 12)
            signal += 2 * amplitude * np.sin(np.linspace(0, 2 * np.pi * cycles, len(signal)))
        elif anomaly_type == "missing":
            signal[start_index:start_index + width] = np.median(signal)
        elif anomaly_type == "saturation":
            limit = float(np.quantile(signal, 1 - 0.3 * severity))
            signal = np.minimum(signal, limit)
        elif anomaly_type == "step":
            signal[start_index:] += 4 * amplitude
        if units == "linear":
            signal = np.maximum(signal, 0)
        if np.allclose(signal, original):
            signal[start_index] += 8 * amplitude
            anomaly_type += "+fallback_spike"
        modified[row_index] = signal.astype(np.float32)
        labels[row_index] = 1
        descriptions[row_index] = anomaly_type
    return modified, labels, descriptions


def split_indices(labels: np.ndarray, validation_fraction: float, test_fraction: float, seed: int):
    generator = np.random.default_rng(seed)
    splits = [[], [], []]
    for label in np.unique(labels):
        indices = generator.permutation(np.flatnonzero(labels == label))
        if len(indices) < 3:
            raise ValueError("Each class needs at least three rows for disjoint train/validation/test sets.")
        validation_count = max(1, round(len(indices) * validation_fraction))
        test_count = max(1, round(len(indices) * test_fraction))
        if validation_count + test_count >= len(indices):
            raise ValueError("Not enough samples for the requested split ratios.")
        splits[0].extend(indices[validation_count + test_count:])
        splits[1].extend(indices[:validation_count])
        splits[2].extend(indices[validation_count:validation_count + test_count])
    return tuple(generator.permutation(np.asarray(indices, dtype=np.int64)) for indices in splits)


def make_loader(spectra: np.ndarray, batch_size: int, features: np.ndarray | None = None, shuffle=False):
    tensors = [torch.from_numpy(np.ascontiguousarray(spectra, dtype=np.float32)).unsqueeze(1)]
    if features is not None:
        tensors.append(torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, num_workers=0)


class Encoder(nn.Module):
    def __init__(self, sequence_length: int, latent_dim: int, strides: list[int]):
        super().__init__()
        self.lengths = [sequence_length]
        layers = []
        input_channels = 1
        for layer_index, (output_channels, stride) in enumerate(zip([8, 16, 32], strides)):
            output_length = (self.lengths[-1] + 2 - 4) // stride + 1
            if output_length < 1:
                raise ValueError("Sequence is too short for the requested encoder strides.")
            self.lengths.append(output_length)
            layers.extend([
                nn.Conv1d(input_channels, output_channels, 4, stride=stride, padding=1),
                nn.BatchNorm1d(output_channels), nn.LeakyReLU(0.2),
            ])
            if layer_index == 1:
                layers.append(nn.Dropout(0.1))
            input_channels = output_channels
        self.convolutions = nn.Sequential(*layers)
        self.projection = nn.Linear(32 * self.lengths[-1], latent_dim)

    def forward(self, spectrum):
        return self.projection(self.convolutions(spectrum).flatten(1))


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, lengths: list[int], strides: list[int]):
        super().__init__()
        self.initial_length = lengths[-1]
        self.sequence_length = lengths[0]
        self.projection = nn.Linear(latent_dim, 32 * self.initial_length)
        layers = []
        input_channels = 32
        for layer_index, output_channels in zip([2, 1, 0], [16, 8, 1]):
            stride = strides[layer_index]
            base_length = (lengths[layer_index + 1] - 1) * stride - 2 + 4
            output_padding = lengths[layer_index] - base_length
            if not 0 <= output_padding < stride:
                raise ValueError("Invalid inverse convolution geometry.")
            layers.append(nn.ConvTranspose1d(
                input_channels, output_channels, 4, stride=stride,
                padding=1, output_padding=output_padding,
            ))
            if output_channels != 1:
                layers.extend([nn.BatchNorm1d(output_channels), nn.LeakyReLU(0.2)])
            input_channels = output_channels
        self.convolutions = nn.Sequential(*layers)

    def forward(self, latent):
        hidden = self.projection(latent).reshape(-1, 32, self.initial_length)
        return self.convolutions(hidden)[..., :self.sequence_length]


class Critic(nn.Module):
    def __init__(self, sequence_length: int, strides: list[int], normalization: str):
        super().__init__()
        layers = []
        input_channels = 1
        output_length = sequence_length
        for output_channels, stride in zip([64, 128, 256], strides):
            output_length = (output_length + 2 - 4) // stride + 1
            layers.append(nn.Conv1d(input_channels, output_channels, 4, stride=stride, padding=1))
            if normalization == "layer":
                layers.append(nn.LayerNorm([output_channels, output_length]))
            elif normalization == "batch":
                layers.append(nn.BatchNorm1d(output_channels))
            layers.append(nn.LeakyReLU(0.2))
            input_channels = output_channels
        self.convolutions = nn.Sequential(*layers)
        self.projection = nn.Linear(256 * output_length, 1)

    def forward(self, spectrum):
        return self.projection(self.convolutions(spectrum).flatten(1)).squeeze(1)


class FeatureHead(nn.Sequential):
    def __init__(self, latent_dim: int):
        super().__init__(
            nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, len(FEATURE_NAMES)),
        )


class FGAAE(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.encoder = Encoder(config["sequence_length"], config["latent_dim"], config["strides"])
        self.decoder = Decoder(config["latent_dim"], self.encoder.lengths, config["strides"])
        self.critic = Critic(config["sequence_length"], config["strides"], config["critic_normalization"])
        self.teacher = FeatureHead(config["latent_dim"])
        self.student = FeatureHead(config["latent_dim"])

    def forward(self, spectrum):
        latent = self.encoder(spectrum)
        return self.decoder(latent), latent

    def freeze_teacher(self):
        self.teacher.requires_grad_(False)
        self.teacher.eval()


def gradient_penalty(critic: Critic, real: torch.Tensor, generated: torch.Tensor):
    interpolation_weight = torch.rand(real.shape[0], 1, 1, device=real.device)
    interpolated = (interpolation_weight * real + (1 - interpolation_weight) * generated.detach())
    interpolated.requires_grad_(True)
    predictions = critic(interpolated)
    gradients = autograd.grad(
        predictions, interpolated, grad_outputs=torch.ones_like(predictions),
        create_graph=True, only_inputs=True,
    )[0]
    return ((gradients.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


@torch.no_grad()
def evaluate(model: FGAAE, spectra: np.ndarray, config: dict, device: torch.device):
    model.eval()
    reconstruction_errors, critic_values, distillation_errors = [], [], []
    for (batch,) in make_loader(spectra, config["batch_size"]):
        batch = batch.to(device)
        reconstructed, latent = model(batch)
        reconstruction_errors.extend((reconstructed - batch).square().mean(dim=(1, 2)).cpu().tolist())
        critic_values.extend(model.critic(reconstructed).cpu().tolist() if config.get("use_adversarial", True) else [0.0] * len(batch))
        distillation_errors.extend(
            (model.student(latent) - model.teacher(latent)).square().mean(dim=1).cpu().tolist() if config.get("use_feature_guidance", True) else [0.0] * len(batch)
        )
    reconstruction_errors = np.asarray(reconstruction_errors, dtype=np.float64)
    critic_values = np.asarray(critic_values, dtype=np.float64)
    reduction = config.get("reconstruction_score_reduction", "mean")
    if reduction not in {"mean", "sum"}:
        raise ValueError("Reconstruction score reduction must be mean or sum.")
    reconstruction_scores = reconstruction_errors * (spectra.shape[1] if reduction == "sum" else 1)
    calibration = config.get("score_calibration")
    if calibration:
        reconstruction_component = (reconstruction_scores - calibration["reconstruction_mean"]) / calibration["reconstruction_std"]
        critic_component = (critic_values - calibration["critic_mean"]) / calibration["critic_std"]
    else:
        reconstruction_component, critic_component = reconstruction_scores, critic_values
    scores = reconstruction_component + config["critic_score_sign"] * config["critic_score_weight"] * critic_component
    if not np.isfinite(scores).all():
        raise FloatingPointError("Non-finite anomaly scores; stop instead of reporting invalid metrics.")
    return {
        "scores": scores, "reconstruction_error": reconstruction_errors,
        "reconstruction_score": reconstruction_scores,
        "critic_value": critic_values,
        "distillation_error": np.asarray(distillation_errors, dtype=np.float64),
    }


def calibrate_score_components(model, reference, config, device):
    if not config.get("standardize_scores", False):
        return
    if reference is None or len(reference) < 2:
        raise ValueError("Standardized scores require a held-out training-reference subset.")
    result = evaluate(model, reference, {**config, "score_calibration": None}, device)
    config["score_calibration"] = {
        "reconstruction_mean": float(result["reconstruction_score"].mean()),
        "reconstruction_std": max(float(result["reconstruction_score"].std()), 1e-6),
        "critic_mean": float(result["critic_value"].mean()),
        "critic_std": max(float(result["critic_value"].std()), 1e-6),
        "fitted_on": "held_out_training_reference_only",
    }


def retention_rate(iteration: int, config: dict) -> float:
    progress = iteration / max(config["iterations"] - 1, 1)
    if config["retention_schedule"] == "linear":
        return config["retention_start"] + progress * (config["retention_end"] - config["retention_start"])
    return config["retention_end"] + (
        config["retention_start"] - config["retention_end"]
    ) * math.exp(-config["retention_alpha"] * progress)


def select_threshold(scores: np.ndarray, labels: np.ndarray, config: dict) -> float:
    if config["threshold_method"] == "validation-f1":
        if len(np.unique(labels)) != 2:
            raise ValueError("F1 threshold selection requires both classes in validation data.")
        precision_values, recall_values, thresholds = precision_recall_curve(labels, scores)
        f1_values = 2 * precision_values[:-1] * recall_values[:-1] / np.maximum(
            precision_values[:-1] + recall_values[:-1], 1e-12
        )
        best_threshold = float(thresholds[int(np.argmax(f1_values))])
        return float(np.nextafter(best_threshold, -np.inf))
    normal_scores = scores[labels == 0]
    if not len(normal_scores):
        raise ValueError("Quantile calibration requires normal validation samples.")
    return float(np.quantile(normal_scores, config["threshold_quantile"]))


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = (scores > threshold).astype(np.int64)
    both_classes = len(np.unique(labels)) == 2
    return {
        "samples": len(labels), "anomalies": int(labels.sum()),
        "predicted_anomalies": int(predicted.sum()), "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)) if both_classes else None,
        "average_precision": float(average_precision_score(labels, scores)) if both_classes else None,
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(labels, predicted, labels=[0, 1]).ravel().tolist(),
    }


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def append_record(path: Path, record: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def check_loss(loss: torch.Tensor) -> None:
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite training loss; aborting this run.")


def train_stage_one(model, train_loader, validation_loader, config, device, output):
    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.teacher.parameters()),
        lr=config["stage1_lr"], weight_decay=config["weight_decay"],
    )
    best_loss = float("inf")
    best_parameters = None
    for epoch in range(config["stage1_epochs"]):
        model.encoder.train()
        model.teacher.train()
        training_loss = 0.0
        for spectra, targets in train_loader:
            spectra, targets = spectra.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (model.teacher(model.encoder(spectra)) - targets).square().mean()
            check_loss(loss)
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * len(spectra)
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for spectra, targets in validation_loader:
                spectra, targets = spectra.to(device), targets.to(device)
                loss = (model.teacher(model.encoder(spectra)) - targets).square().mean()
                check_loss(loss)
                validation_loss += loss.item() * len(spectra)
        training_loss /= len(train_loader.dataset)
        validation_loss /= len(validation_loader.dataset)
        append_record(output / "stage1_history.csv", {
            "epoch": epoch + 1, "train_feature_loss": training_loss,
            "validation_feature_loss": validation_loss,
        })
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_parameters = {
                "encoder": copy.deepcopy(model.encoder.state_dict()),
                "teacher": copy.deepcopy(model.teacher.state_dict()),
            }
        print(f"Stage 1 {epoch + 1}/{config['stage1_epochs']}: train={training_loss:.6f}, val={validation_loss:.6f}", flush=True)
    model.encoder.load_state_dict(best_parameters["encoder"])
    model.teacher.load_state_dict(best_parameters["teacher"])
    model.freeze_teacher()
    torch.save(best_parameters, output / "stage1_best.pt")


def encoder_training_mode(model, config):
    model.encoder.train()
    if config.get("freeze_encoder_bn", False):
        for module in model.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def encoder_gradient_diagnostics(model, losses):
    parameters = tuple(model.encoder.parameters())
    gradients = {
        name: torch.cat([gradient.detach().flatten() for gradient in autograd.grad(loss, parameters, retain_graph=True)])
        for name, loss in losses.items()
    }
    if not all(torch.isfinite(value).all() for value in gradients.values()):
        raise FloatingPointError("Non-finite encoder gradient diagnostic.")
    result = {f"encoder_gradient_{name}_norm": float(value.norm()) for name, value in gradients.items()}
    for left, right in [("reconstruction", "adversarial"), ("reconstruction", "distillation")]:
        denominator = gradients[left].norm() * gradients[right].norm()
        result[f"encoder_gradient_{left}_{right}_cosine"] = float(
            gradients[left].dot(gradients[right]) / denominator.clamp_min(1e-12)
        )
    return result


def train_stage_two(model, training, validation, validation_labels, source_indices, config, device, output, score_reference=None, epoch_callback=None):
    generator_parameters = (
        list(model.encoder.parameters()) + list(model.decoder.parameters())
        + list(model.student.parameters())
    )
    if config.get("encoder_lr") is not None:
        generator_parameters = [
            {"params": list(model.encoder.parameters()), "lr": config["encoder_lr"]},
            {"params": list(model.decoder.parameters()) + list(model.student.parameters()), "lr": config["stage2_lr"]},
        ]
    generator_optimizer = torch.optim.AdamW(
        generator_parameters, lr=config["stage2_lr"], weight_decay=config["weight_decay"],
        betas=(0.5, 0.9),
    )
    critic_optimizer = torch.optim.AdamW(
        model.critic.parameters(), lr=config["critic_lr"],
        weight_decay=config["weight_decay"], betas=(0.5, 0.9),
    )
    global_epoch = 0
    best_score = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    best_state = None
    best_calibration = None
    stopped = False
    for iteration in range(config["iterations"]):
        calibrate_score_components(model, score_reference, config, device)
        inference = evaluate(model, training, config, device)
        filtering_active = config.get("use_aqt", True) and iteration >= config.get("warmup_iterations", 0)
        warmup = config.get("warmup_iterations", 0)
        schedule_config = {**config, "iterations": max(1, config["iterations"] - warmup)}
        rate = retention_rate(iteration - warmup, schedule_config) if filtering_active else 1.0
        screening_threshold = float(np.quantile(inference["scores"], rate))
        retained_indices = np.flatnonzero(inference["scores"] <= screening_threshold)
        if len(retained_indices) < 2:
            retained_indices = np.argsort(inference["scores"], kind="stable")[:2]
            screening_threshold = float(inference["scores"][retained_indices].max())
        pd.DataFrame({
            "source_row": source_indices,
            "score": inference["scores"],
            "retained": np.isin(np.arange(len(training)), retained_indices).astype(np.int64),
        }).to_csv(output / f"aqt_iteration_{iteration + 1:03d}.csv", index=False)
        loader = make_loader(training[retained_indices], config["batch_size"], shuffle=True)
        for epoch in range(config["epochs_per_iteration"]):
            model.train()
            model.teacher.eval()
            totals = {key: 0.0 for key in ["generator", "reconstruction", "adversarial", "distillation", "critic", "gradient_penalty"]}
            gradient_record = {}
            seen = 0
            for (real,) in loader:
                real = real.to(device)
                model.critic.requires_grad_(config.get("use_adversarial", True))
                model.encoder.eval()
                model.decoder.eval()
                with torch.no_grad():
                    generated, _ = model(real)
                critic_loss_sum = penalty_sum = 0.0
                for _ in range(config["critic_steps"] if config.get("use_adversarial", True) else 0):
                    critic_optimizer.zero_grad(set_to_none=True)
                    penalty = gradient_penalty(model.critic, real, generated)
                    critic_loss = (
                        model.critic(generated).mean() - model.critic(real).mean()
                        + config["gp_weight"] * penalty
                    )
                    check_loss(critic_loss)
                    critic_loss.backward()
                    critic_optimizer.step()
                    critic_loss_sum += critic_loss.item()
                    penalty_sum += penalty.item()
                model.critic.requires_grad_(False)
                model.critic.eval()
                encoder_training_mode(model, config)
                model.decoder.train()
                generator_optimizer.zero_grad(set_to_none=True)
                generated, latent = model(real)
                reconstruction_loss = (generated - real).square().mean()
                adversarial_loss = -model.critic(generated).mean() if config.get("use_adversarial", True) else generated.new_zeros(())
                if config.get("use_feature_guidance", True):
                    with torch.no_grad():
                        teacher_features = model.teacher(latent)
                    distillation_loss = (model.student(latent) - teacher_features).square().mean()
                else:
                    distillation_loss = generated.new_zeros(())
                generator_loss = (
                    config["reconstruction_weight"] * reconstruction_loss
                    + config["adversarial_weight"] * adversarial_loss
                    + config["distillation_weight"] * distillation_loss
                )
                check_loss(generator_loss)
                if config.get("monitor_encoder_gradients", False) and seen == 0:
                    gradient_record = encoder_gradient_diagnostics(model, {
                        "reconstruction": config["reconstruction_weight"] * reconstruction_loss,
                        "adversarial": config["adversarial_weight"] * adversarial_loss,
                        "distillation": config["distillation_weight"] * distillation_loss,
                    })
                generator_loss.backward()
                generator_optimizer.step()
                model.critic.requires_grad_(config.get("use_adversarial", True))
                model.critic.train()
                batch_losses = {
                    "generator": generator_loss.item(), "reconstruction": reconstruction_loss.item(),
                    "adversarial": adversarial_loss.item(), "distillation": distillation_loss.item(),
                    "critic": critic_loss_sum / config["critic_steps"],
                    "gradient_penalty": penalty_sum / config["critic_steps"],
                }
                for name, value in batch_losses.items():
                    totals[name] += value * len(real)
                seen += len(real)
            global_epoch += 1
            calibrate_score_components(model, score_reference, config, device)
            validation_result = evaluate(model, validation, config, device)
            validation_metrics = {"roc_auc": float(roc_auc_score(validation_labels, validation_result["scores"]))}
            record = {
                "epoch": global_epoch, "iteration": iteration + 1,
                "epoch_in_iteration": epoch + 1, "retention_rate": rate,
                "retained_samples": len(retained_indices), "aqt_threshold": screening_threshold,
                **{f"train_{name}_loss": value / seen for name, value in totals.items()},
                "validation_reconstruction_loss": float(validation_result["reconstruction_error"].mean()),
                "validation_distillation_loss": float(validation_result["distillation_error"].mean()),
                **{f"validation_{name}": validation_metrics[name] for name in ["roc_auc"]},
                **gradient_record,
            }
            if epoch_callback is not None:
                record.update(epoch_callback(model, global_epoch))
            if config.get("encoder_lr") is not None:
                record["encoder_learning_rate"] = generator_optimizer.param_groups[0]["lr"]
                record["decoder_student_learning_rate"] = generator_optimizer.param_groups[1]["lr"]
            append_record(output / "stage2_history.csv", record)
            validation_score = validation_metrics["roc_auc"]
            if validation_score is None:
                validation_score = -float(validation_result["reconstruction_error"].mean())
            if validation_score > best_score + config.get("early_stopping_delta", 1e-4):
                best_score = validation_score
                best_epoch = global_epoch
                stale_epochs = 0
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                best_calibration = copy.deepcopy(config.get("score_calibration"))
                torch.save({
                    "model_state": best_state, "config": copy.deepcopy(config),
                    "epoch": best_epoch, "validation_selection_score": best_score,
                }, output / "stage2_best.pt")
            else:
                stale_epochs += 1
            print(
                f"Stage 2 iteration {iteration + 1}/{config['iterations']}, epoch {epoch + 1}: "
                f"retained={len(retained_indices)}/{len(training)}, "
                f"recon={totals['reconstruction'] / seen:.6f}, val_auc={validation_metrics['roc_auc']}",
                flush=True,
            )
            patience = config.get("early_stopping_patience", 10)
            if (
                patience > 0 and stale_epochs >= patience
                and global_epoch >= config.get("minimum_epochs", 10)
            ):
                stopped = True
                print(f"Early stopping at epoch {global_epoch}; best validation epoch={best_epoch}.", flush=True)
                break
        torch.save({"model_state": model.state_dict(), "config": config, "iteration": iteration + 1}, output / "stage2_latest.pt")
        if stopped:
            break
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint was produced.")
    model.load_state_dict(best_state)
    if best_calibration is not None:
        config["score_calibration"] = best_calibration
    selection = {
        "selected_by": "validation ROC-AUC; negative reconstruction loss when only one class is available",
        "best_epoch": best_epoch, "best_validation_selection_score": best_score,
        "epochs_ran": global_epoch, "early_stopped": stopped,
        "test_used_for_selection": False,
    }
    write_json(output / "selection.json", selection)
    return selection


def save_predictions(path, source_indices, labels, descriptions, result, threshold):
    frame = pd.DataFrame({
        "source_row": source_indices, "anomaly_score": result["scores"],
        "reconstruction_error": result["reconstruction_error"],
        "critic_value": result["critic_value"],
        "predicted_label": (result["scores"] > threshold).astype(np.int64),
    })
    if labels is not None:
        frame["label"] = labels
    if descriptions is not None:
        frame["anomaly_type"] = descriptions
    frame.to_csv(path, index=False)


def plot_results(output, labels, result, examples, reconstructed):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for filename, pairs in [
        ("stage1_history.csv", [("train_feature_loss", "validation_feature_loss")]),
        ("stage2_history.csv", [
            ("train_reconstruction_loss", "validation_reconstruction_loss"),
            ("train_distillation_loss", "validation_distillation_loss"),
        ]),
    ]:
        history = pd.read_csv(output / filename)
        figure, axes = plt.subplots(1, len(pairs), figsize=(7 * len(pairs), 4), squeeze=False)
        for axis, columns in zip(axes[0], pairs):
            for column in columns:
                axis.plot(history["epoch"], history[column], label=column)
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Loss")
            axis.legend()
            axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(output / filename.replace(".csv", ".png"), dpi=150)
        plt.close(figure)
    if len(np.unique(labels)) == 2:
        false_positive, true_positive, _ = roc_curve(labels, result["scores"])
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.plot(false_positive, true_positive, label=f"FGAAE AUC={roc_auc_score(labels, result['scores']):.4f}")
        axis.plot([0, 1], [0, 1], "--", color="gray")
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / "test_roc.png", dpi=150)
        plt.close(figure)
    figure, axes = plt.subplots(len(examples), 1, figsize=(10, 3 * len(examples)), squeeze=False)
    for row, axis in enumerate(axes[:, 0]):
        axis.plot(examples[row], label="Input (normalized)")
        axis.plot(reconstructed[row], label="Reconstruction")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "reconstructions.png", dpi=150)
    plt.close(figure)


def validate_config(config: dict) -> None:
    if (
        config.get("early_stopping_patience", 10) < 0
        or config.get("early_stopping_delta", 1e-4) < 0
        or config.get("minimum_epochs", 10) < 1
    ):
        raise ValueError("Early stopping requires nonnegative patience/delta and positive minimum epochs.")
    positive_names = [
        "stage1_epochs", "iterations", "epochs_per_iteration", "batch_size", "latent_dim",
        "critic_steps", "stage1_lr", "stage2_lr", "critic_lr", "gp_weight", "level_scale",
        "retention_alpha", "anomaly_severity", "reconstruction_weight",
    ]
    if any(config[name] <= 0 for name in positive_names):
        raise ValueError("Epochs, sizes, learning rates, GP, scale and severity must be positive.")
    if config.get("encoder_lr") is not None and (not np.isfinite(config["encoder_lr"]) or config["encoder_lr"] <= 0):
        raise ValueError("Encoder learning rate must be finite and positive.")
    if config["num_threads"] < 0 or config["weight_decay"] < 0:
        raise ValueError("Thread count and weight decay cannot be negative.")
    if config["adversarial_weight"] < 0 or config["distillation_weight"] < 0 or config["critic_score_weight"] < 0:
        raise ValueError("Loss and critic score weights cannot be negative.")
    if not 0 < config["validation_fraction"] < 1 or not 0 < config["test_fraction"] < 1:
        raise ValueError("Validation/test fractions must be between zero and one.")
    if config["validation_fraction"] + config["test_fraction"] >= 1:
        raise ValueError("Validation and test fractions must leave a nonempty training set.")
    if not 0 < config["stage1_fraction"] < 1 or not 0 < config["stage1_validation_fraction"] < 1:
        raise ValueError("Stage-one fractions must be between zero and one.")
    if not 0 < config["retention_end"] <= config["retention_start"] <= 1:
        raise ValueError("Require 0 < retention_end <= retention_start <= 1.")
    if not 0 < config["anomaly_ratio"] < 1 or not 0 <= config["train_anomaly_ratio"] < 1:
        raise ValueError("Evaluation anomaly ratio must be in (0,1); train ratio in [0,1).")
    if config["anomaly_severity"] > 1 or not 0 < config["threshold_quantile"] < 1:
        raise ValueError("Severity must be <= 1 and threshold quantile must be in (0,1).")
    if config["start_mhz"] >= config["end_mhz"] or any(stride < 1 for stride in config["strides"]):
        raise ValueError("Invalid frequency bounds or convolution strides.")
    if config["peak_prominence"] < 0 or not all(
        np.isfinite(value) for value in config.values() if isinstance(value, float)
    ):
        raise ValueError("Configuration must be finite, with nonnegative peak prominence.")


def run_training(arguments: argparse.Namespace) -> Path:
    config = vars(arguments).copy()
    config.pop("command", None)
    config["data"] = str(Path(config["data"]).resolve())
    validate_config(config)
    seed_everything(config["seed"], config["num_threads"])
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available. Use --device cpu.")
    spectra, supplied_labels = read_spectra(Path(config["data"]), config["label_column"])
    if len(spectra) < 40:
        raise ValueError("At least 40 rows are needed for this two-stage experiment.")
    if supplied_labels is not None and config["evaluation"] == "synthetic":
        raise ValueError("For labeled input use --evaluation real; synthetic mode requires an unlabeled normal reference file.")
    if supplied_labels is None and config["evaluation"] == "real":
        raise ValueError("Real evaluation requires a CSV with a 0/1 label column.")
    synthetic = supplied_labels is None
    config["evaluation"] = "synthetic" if synthetic else "real"
    config["sequence_length"] = int(spectra.shape[1])
    if synthetic:
        print("WARNING: no real anomaly labels. This run uses SYNTHETIC anomalies and is NOT a reproduction of the paper's reported metrics.", flush=True)
        supplied_labels = np.zeros(len(spectra), dtype=np.int64)
    train_indices, validation_indices, test_indices = split_indices(
        supplied_labels, config["validation_fraction"], config["test_fraction"], config["seed"]
    )
    normal_training_indices = train_indices[supplied_labels[train_indices] == 0]
    stage1_count = max(8, round(len(train_indices) * config["stage1_fraction"]))
    if stage1_count > len(normal_training_indices) or stage1_count > len(train_indices) - 2:
        raise ValueError("Not enough normal training rows for stage one and a disjoint stage-two pool.")
    stage1_indices = normal_training_indices[:stage1_count]
    stage1_validation_count = min(stage1_count - 2, max(2, round(stage1_count * config["stage1_validation_fraction"])))
    stage1_validation_indices = stage1_indices[:stage1_validation_count]
    stage1_training_indices = stage1_indices[stage1_validation_count:]
    stage2_indices = train_indices[~np.isin(train_indices, stage1_indices)]
    training = spectra[stage2_indices].copy()
    validation = spectra[validation_indices].copy()
    testing = spectra[test_indices].copy()
    training_labels = supplied_labels[stage2_indices].copy()
    validation_labels = supplied_labels[validation_indices].copy()
    test_labels = supplied_labels[test_indices].copy()
    training_types = ["real"] * len(training)
    validation_types = ["real"] * len(validation)
    test_types = ["real"] * len(testing)
    if synthetic:
        training, training_labels, training_types = inject_anomalies(
            training, config["train_anomaly_ratio"], config["seed"] + 1,
            config["anomaly_severity"], config["units"],
        )
        validation, validation_labels, validation_types = inject_anomalies(
            validation, config["anomaly_ratio"], config["seed"] + 2,
            config["anomaly_severity"], config["units"],
        )
        testing, test_labels, test_types = inject_anomalies(
            testing, config["anomaly_ratio"], config["seed"] + 3,
            config["anomaly_severity"], config["units"],
        )
    normalization_reference = spectra[stage1_training_indices].astype(np.float64)
    spectrum_mean = float(normalization_reference.mean())
    spectrum_std = float(normalization_reference.std())
    if spectrum_std < 1e-8:
        raise ValueError("Normal reference spectra have zero variance.")
    normalizer = lambda values: ((values.astype(np.float64) - spectrum_mean) / spectrum_std).astype(np.float32)
    stage1_train_features = expert_features(spectra[stage1_training_indices], config)
    stage1_validation_features = expert_features(spectra[stage1_validation_indices], config)
    feature_mean = stage1_train_features.mean(axis=0, dtype=np.float64)
    feature_std = stage1_train_features.std(axis=0, dtype=np.float64)
    feature_std[feature_std < 1e-8] = 1.0
    stage1_train_loader = make_loader(
        normalizer(spectra[stage1_training_indices]), config["batch_size"],
        (stage1_train_features - feature_mean) / feature_std, shuffle=True,
    )
    stage1_validation_loader = make_loader(
        normalizer(spectra[stage1_validation_indices]), config["batch_size"],
        (stage1_validation_features - feature_mean) / feature_std,
    )
    output = Path(config["output"]) if config["output"] else ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    config["output"] = str(output.resolve())
    write_json(output / "config.json", config)
    write_json(output / "environment.json", {
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__, "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    })
    write_json(output / "normalization.json", {
        "spectrum_mean": spectrum_mean, "spectrum_std": spectrum_std,
        "feature_names": FEATURE_NAMES, "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(), "fitted_on": "stage1_training_only",
    })
    split_rows = []
    for name, indices, labels, descriptions in [
        ("stage1_train", stage1_training_indices, supplied_labels[stage1_training_indices], ["normal"] * len(stage1_training_indices)),
        ("stage1_validation", stage1_validation_indices, supplied_labels[stage1_validation_indices], ["normal"] * len(stage1_validation_indices)),
        ("stage2_train", stage2_indices, training_labels, training_types),
        ("validation", validation_indices, validation_labels, validation_types),
        ("test", test_indices, test_labels, test_types),
    ]:
        split_rows.extend({"split": name, "source_row": int(index), "label": int(label), "anomaly_type": description}
                          for index, label, description in zip(indices, labels, descriptions))
    pd.DataFrame(split_rows).to_csv(output / "splits.csv", index=False)
    print(f"Output: {output.resolve()}", flush=True)
    print(f"Data: {spectra.shape}, stage1={stage1_count}, stage2={len(training)}, val={len(validation)}, test={len(testing)}", flush=True)
    model = FGAAE(config).to(device)
    train_stage_one(model, stage1_train_loader, stage1_validation_loader, config, device, output)
    training, validation, testing = map(normalizer, [training, validation, testing])
    train_stage_two(model, training, validation, validation_labels, stage2_indices, config, device, output)
    validation_result = evaluate(model, validation, config, device)
    threshold = select_threshold(validation_result["scores"], validation_labels, config)
    test_result = evaluate(model, testing, config, device)
    report = {
        "evaluation": config["evaluation"],
        "claim": "Implementation experiment only; not the paper's original results.",
        "threshold_calibration": "validation only; test labels never used for threshold selection",
        "validation": metrics(validation_labels, validation_result["scores"], threshold),
        "test": metrics(test_labels, test_result["scores"], threshold),
    }
    write_json(output / "metrics.json", report)
    save_predictions(output / "validation_scores.csv", validation_indices, validation_labels, validation_types, validation_result, threshold)
    save_predictions(output / "test_scores.csv", test_indices, test_labels, test_types, test_result, threshold)
    torch.save({
        "model_state": model.state_dict(), "config": config, "threshold": threshold,
        "normalization": {"mean": spectrum_mean, "std": spectrum_std},
        "feature_names": FEATURE_NAMES, "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
    }, output / "model.pt")
    if not config["no_plots"]:
        examples = testing[:min(4, len(testing))]
        model.eval()
        with torch.no_grad():
            reconstructed, _ = model(torch.from_numpy(examples).unsqueeze(1).to(device))
        plot_results(output, test_labels, test_result, examples, reconstructed.squeeze(1).cpu().numpy())
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return output


def run_prediction(arguments):
    device = torch.device(arguments.device)
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    config["batch_size"] = arguments.batch_size
    spectra, labels = read_spectra(Path(arguments.data), config["label_column"])
    if spectra.shape[1] != config["sequence_length"]:
        raise ValueError(f"Expected {config['sequence_length']} points, received {spectra.shape[1]}.")
    model = FGAAE(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    normalization = checkpoint["normalization"]
    normalized = ((spectra.astype(np.float64) - normalization["mean"]) / normalization["std"]).astype(np.float32)
    result = evaluate(model, normalized, config, device)
    output = Path(arguments.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing predictions: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_predictions(output, np.arange(len(spectra)), labels, None, result, checkpoint["threshold"])
    if labels is not None:
        print(json.dumps(metrics(labels, result["scores"], checkpoint["threshold"]), indent=2))
    print(f"Predictions: {output.resolve()}")


def run_smoke_test():
    """Run a small CPU integration check in a temporary directory, not a scientific experiment."""
    with tempfile.TemporaryDirectory(prefix="fgaae_smoke_") as temporary:
        directory = Path(temporary)
        generator = np.random.default_rng(2025)
        frequencies = np.linspace(0, 1, 840)
        spectra = generator.normal(100, 8, (96, 840))
        for peak_center in [0.15, 0.4, 0.75]:
            spectra += 80 * np.exp(-0.5 * ((frequencies - peak_center) / 0.01) ** 2)
        np.savetxt(directory / "spectra.csv", spectra, delimiter=",")
        arguments = build_parser().parse_args([
            "train", "--data", str(directory / "spectra.csv"),
            "--output", str(directory / "run"), "--device", "cpu", "--num-threads", "2",
            "--stage1-epochs", "1", "--iterations", "1", "--epochs-per-iteration", "1",
            "--critic-steps", "1", "--batch-size", "16", "--no-plots",
        ])
        output = run_training(arguments)
        checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
        model = FGAAE(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state"])
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
        for length in [840, 841]:
            configuration = {**checkpoint["config"], "sequence_length": length}
            shape_model = FGAAE(configuration).eval()
            with torch.no_grad():
                reconstructed, latent = shape_model(torch.zeros(2, 1, length))
            assert reconstructed.shape == (2, 1, length)
            assert latent.shape == (2, 512)
        run_prediction(argparse.Namespace(
            checkpoint=str(output / "model.pt"), data=str(directory / "spectra.csv"),
            output=str(directory / "predictions.csv"), device="cpu", batch_size=16,
        ))
        predictions = pd.read_csv(directory / "predictions.csv")
        assert len(predictions) == 96 and np.isfinite(predictions["anomaly_score"]).all()
        split_frame = pd.read_csv(output / "splits.csv")
        assert split_frame["source_row"].is_unique
        history = pd.read_csv(output / "stage2_history.csv")
        assert history["train_gradient_penalty_loss"].iloc[0] >= 0
        print("Smoke test passed: training, WGAN-GP, AQT, disjoint splits, checkpoint, inference, exact decoder shapes.")


def build_parser():
    parser = argparse.ArgumentParser(description="Paper-oriented FGAAE reproduction with WGAN-GP and AQT.")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train and evaluate; unlabeled input uses explicitly synthetic anomalies.")
    training.add_argument("--data", default=str(DEFAULT_DATA))
    training.add_argument("--output", default=None)
    training.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    training.add_argument("--seed", type=int, default=2025)
    training.add_argument("--num-threads", type=int, default=0)
    training.add_argument("--evaluation", choices=["auto", "real", "synthetic"], default="auto")
    training.add_argument("--label-column", default="label")
    training.add_argument("--validation-fraction", type=float, default=0.15)
    training.add_argument("--test-fraction", type=float, default=0.15)
    training.add_argument("--stage1-fraction", type=float, default=0.1)
    training.add_argument("--stage1-validation-fraction", type=float, default=0.2)
    training.add_argument("--stage1-epochs", type=int, default=100)
    training.add_argument("--iterations", type=int, default=10)
    training.add_argument("--epochs-per-iteration", type=int, default=20)
    training.add_argument("--batch-size", type=int, default=32)
    training.add_argument("--latent-dim", type=int, default=512)
    training.add_argument("--strides", type=int, nargs=3, default=[4, 2, 2])
    training.add_argument("--critic-normalization", choices=["layer", "batch", "none"], default="layer")
    training.add_argument("--critic-steps", type=int, default=5)
    training.add_argument("--gp-weight", type=float, default=10.0)
    training.add_argument("--stage1-lr", type=float, default=1e-5)
    training.add_argument("--stage2-lr", type=float, default=2e-5)
    training.add_argument("--critic-lr", type=float, default=2e-5)
    training.add_argument("--weight-decay", type=float, default=1e-3)
    training.add_argument("--reconstruction-weight", type=float, default=1.0)
    training.add_argument("--adversarial-weight", type=float, default=0.01)
    training.add_argument("--distillation-weight", type=float, default=1.0)
    training.add_argument("--critic-score-weight", type=float, default=0.1)
    training.add_argument("--critic-score-sign", type=int, choices=[-1, 1], default=-1)
    training.add_argument("--reconstruction-score-reduction", choices=["mean", "sum"], default="mean")
    training.add_argument("--retention-start", type=float, default=0.95)
    training.add_argument("--retention-end", type=float, default=0.5)
    training.add_argument("--retention-alpha", type=float, default=5.0)
    training.add_argument("--retention-schedule", choices=["exponential", "linear"], default="exponential")
    training.add_argument("--threshold-method", choices=["quantile", "validation-f1"], default="quantile")
    training.add_argument("--threshold-quantile", type=float, default=0.99)
    training.add_argument("--anomaly-ratio", type=float, default=0.1)
    training.add_argument("--train-anomaly-ratio", type=float, default=0.05)
    training.add_argument("--anomaly-severity", type=float, default=0.7)
    training.add_argument("--units", choices=["db", "linear"], default="db")
    training.add_argument("--level-scale", type=float, default=0.1)
    training.add_argument("--start-mhz", type=float, default=87.0)
    training.add_argument("--end-mhz", type=float, default=108.0)
    training.add_argument("--peak-prominence", type=float, default=3.0)
    training.add_argument("--occupancy-margin", type=float, default=6.0)
    training.add_argument("--no-plots", action="store_true")
    training.add_argument("--early-stopping-patience", type=int, default=10)
    training.add_argument("--early-stopping-delta", type=float, default=1e-4)
    training.add_argument("--minimum-epochs", type=int, default=10)
    training.add_argument("--disable-aqt", dest="use_aqt", action="store_false", default=True)
    prediction = commands.add_parser("predict", help="Predict using a saved model and its training normalization.")
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--data", required=True)
    prediction.add_argument("--output", required=True)
    prediction.add_argument("--device", default="cpu")
    prediction.add_argument("--batch-size", type=int, default=32)
    commands.add_parser("smoke-test", help="Small CPU check; all artifacts go to a temporary directory.")
    return parser


def main():
    arguments = build_parser().parse_args()
    if arguments.command == "train":
        run_training(arguments)
    elif arguments.command == "predict":
        run_prediction(arguments)
    else:
        run_smoke_test()


if __name__ == "__main__":
    main()
