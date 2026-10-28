"""Render scientific AUC figures from completed, unchanged experiment tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from freeze_baseline import ROOT, digest, read_json, write_json

COLORS = {"Base": "#777777", "A": "#0072B2", "B": "#E69F00", "C": "#009E73", "Full": "#D55E00"}
LABELS = {"Base": "Base (AE)", "A": "A (without WGAN-GP)", "B": "B (without feature guidance)",
          "C": "C (without AQT)", "Full": "Full FGAAE"}


def render(run):
    if not read_json(run / "report.json")["complete"]:
        raise ValueError("Only completed experiments may be rendered.")
    names = ["auc_summary.csv", "auc_runs.csv", "paired_auc_differences.csv", "roc_interpolation.csv"]
    original_hashes = {name: digest(run / name) for name in names}
    summary = pd.read_csv(run / "auc_summary.csv").set_index("arm")
    interpolation = pd.read_csv(run / "roc_interpolation.csv")
    paired = pd.read_csv(run / "paired_auc_differences.csv")
    old_figure = digest(run / "roc_curves.png")
    figure, axis = plt.subplots(figsize=(8.0, 6.2))
    for arm in COLORS:
        curve = interpolation[interpolation.arm == arm].sort_values("fpr")
        x = curve.fpr.to_numpy()
        mean = curve.mean_tpr.to_numpy()
        spread = curve.sample_std_tpr.to_numpy()
        row = summary.loc[arm]
        axis.plot(x, mean, color=COLORS[arm], linewidth=2.2 if arm == "Full" else 1.6,
                  label=f"{LABELS[arm]}: {row.auc_mean:.4f} +/- {row.auc_sample_std:.4f}")
        axis.plot([0, 0], [0, mean[0]], color=COLORS[arm], linewidth=1.0)
        axis.fill_between(x, np.maximum(0, mean-spread), np.minimum(1, mean+spread), color=COLORS[arm], alpha=0.09)
    axis.plot([0, 1], [0, 1], "--", color="#AAAAAA", linewidth=1)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate",
             title="FGAAE ablation: frozen development protocol")
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right", fontsize=9)
    figure.tight_layout()
    figure.savefig(run / "roc_curves.png", dpi=200)
    figure.savefig(run / "roc_curves.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    variants = ["Base", "A", "B", "C"]
    for index, arm in enumerate(variants):
        rows = paired[paired.arm == arm].sort_values("seed")
        if len(rows) != 3:
            raise ValueError("Three paired seeds required for every variant.")
        values = rows.delta_full_minus_variant.to_numpy()
        axis.bar(index, values.mean(), width=0.55, color=COLORS[arm], alpha=0.30)
        axis.errorbar(index, values.mean(), yerr=values.std(ddof=1), color=COLORS[arm], capsize=5, linewidth=1.4)
        for offset, row in zip([-0.15, 0, 0.15], rows.itertuples()):
            axis.scatter(index + offset, row.delta_full_minus_variant, color=COLORS[arm], edgecolors="white", s=48, zorder=3)
            axis.annotate(str(row.seed), (index + offset, row.delta_full_minus_variant), xytext=(0, 6),
                          textcoords="offset points", ha="center", fontsize=8)
    axis.axhline(0, color="#444444", linestyle="--", linewidth=1)
    axis.set_xticks(range(4), [LABELS[name] for name in variants], fontsize=8)
    axis.set(ylabel="Paired ROC-AUC difference (Full - variant)", title="Three paired seeds; bars = mean, whiskers = sample SD")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(run / "paired_auc_differences.png", dpi=200)
    figure.savefig(run / "paired_auc_differences.pdf")
    plt.close(figure)
    if any(digest(run / name) != expected for name, expected in original_hashes.items()):
        raise AssertionError("Figure rendering changed experiment tables.")
    write_json(run / "figure_postprocessing.json", {
        "code_sha256": digest(Path(__file__)), "input_tables_sha256": original_hashes,
        "original_roc_png_sha256": old_figure, "roc_png_sha256": digest(run / "roc_curves.png"),
        "paired_png_sha256": digest(run / "paired_auc_differences.png"), "metrics_unchanged": True,
        "roc_method": "1001 fixed FPR points; linear interpolation per seed; mean and pointwise sample SD; matching line/band colors",
        "numeric_auc_method": "Experiment tables use roc_auc_score on original per-seed scores, not integration of interpolated display curves.",
        "paired_plot": "Full minus variant for each of three seeds; bars mean, whiskers sample SD; not confidence intervals.",
    })
    print("AUC figures exported to PNG/PDF; all numerical input tables unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/ablation_auc_09117_v1")
    render(parser.parse_args().run.resolve())
