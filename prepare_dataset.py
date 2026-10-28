from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_data import audit_fscan, full_digest, quality_flags


ROOT = Path(__file__).resolve().parent


def grouped_partitions(metadata: pd.DataFrame) -> pd.DataFrame:
    groups = metadata.groupby(["station", "source_file"], as_index=False).agg(first_ticks=("ticks", "min"))
    assignments = {}
    for _, station_groups in groups.groupby("station", sort=True):
        ordered = station_groups.sort_values(["first_ticks", "source_file"])
        source_files = ordered.source_file.tolist()
        if len(source_files) == 1:
            assignments[source_files[0]] = "external_test"
            continue
        if len(source_files) == 2:
            assignments[source_files[0]] = "train"
            assignments[source_files[1]] = "test"
            continue
        holdout_count = max(1, round(len(source_files) * 0.15))
        training_count = len(source_files) - 2 * holdout_count
        for index, name in enumerate(source_files):
            assignments[name] = (
                "train" if index < training_count else
                "validation" if index < training_count + holdout_count else "test"
            )
    result = metadata[["row", "source_file", "station", "ticks", "band_sha256"]].copy()
    result["split"] = result.source_file.map(assignments)
    if result.groupby("source_file").split.nunique().max() != 1:
        raise RuntimeError("A source file crossed split boundaries.")
    if result.groupby("band_sha256").split.nunique().max() != 1:
        raise RuntimeError("An identical spectrum crossed split boundaries.")
    if not {"train", "validation", "test"}.issubset(set(result.split)):
        raise ValueError("Not enough source files for the requested grouped experiment.")
    return result


def prepare(arguments):
    data_root = Path(arguments.data_root).resolve()
    output = Path(arguments.output).resolve()
    if output == data_root or data_root in output.parents:
        raise ValueError("Write prepared data outside the original data tree.")
    output.mkdir(parents=True, exist_ok=False)
    audited = pd.read_csv(arguments.audit)
    expected_hashes = pd.read_csv(arguments.copies).query("preferred == True").set_index("stem").ix_sha256
    candidates = audited[audited.status == "native_840_candidate"].sort_values(["station", "first_time", "stem"])
    all_bands, all_metadata, sources = [], [], []
    for number, candidate in enumerate(candidates.itertuples(), 1):
        path = data_root / candidate.ix_path
        if full_digest(path) != expected_hashes.loc[candidate.stem]:
            raise ValueError(f"IX changed since audit: {candidate.stem}")
        result, values, records = audit_fscan(path, data_root)
        if result["status"] != "native_840_candidate" or not result["md_length_matches_ix"]:
            raise ValueError(f"Source no longer passes structural checks: {candidate.stem}")
        clean = np.ones(len(values), dtype=bool)
        for mask in quality_flags(values).values():
            clean &= ~mask
        frame_indices = np.flatnonzero(clean)
        if len(frame_indices) != result["clean_band_frames"]:
            raise RuntimeError("Preparation disagrees with the audited target-band geometry.")
        frame_starts = np.flatnonzero(records["start_index"] == 0)
        frame_counts = np.add.reduceat(records["count"].astype(np.int64), frame_starts)
        selected = np.ascontiguousarray(values[frame_indices], dtype="<i2")
        for frame_index, band in zip(frame_indices, selected):
            record_index = frame_starts[frame_index]
            ticks = int(records["ticks"][record_index])
            all_metadata.append({
                "station": candidate.station, "source_file": candidate.stem,
                "source_ix": candidate.ix_path, "source_md": candidate.md_path,
                "frame_index": int(frame_index), "ticks": ticks,
                "source_time": str(datetime(1, 1, 1) + timedelta(microseconds=ticks // 10)),
                "step_hz": 25000, "start_hz": 87000000, "points": 840,
                "complete_wideband_scan": bool(frame_counts[frame_index] == records["declared_points"][record_index]),
                "ground_truth_label": -1,
                "band_sha256": hashlib.sha256(band.tobytes()).hexdigest(),
            })
        all_bands.append(selected)
        sources.append({"source_file": candidate.stem, "station": candidate.station, "frames_before_row_dedup": len(selected)})
        print(f"Prepared {number}/{len(candidates)}: {candidate.stem}, frames={len(selected)}", flush=True)
    matrix = np.concatenate(all_bands)
    metadata = pd.DataFrame(all_metadata)
    duplicate_mask = metadata.band_sha256.duplicated(keep="first")
    duplicates = metadata.loc[duplicate_mask].copy()
    first_sources = metadata.drop_duplicates("band_sha256").set_index("band_sha256").source_file
    duplicates["kept_source_file"] = duplicates.band_sha256.map(first_sources)
    duplicates.to_csv(output / "duplicate_frames.csv", index=False, encoding="utf-8-sig")
    matrix = np.ascontiguousarray(matrix[~duplicate_mask.to_numpy()])
    metadata = metadata.loc[~duplicate_mask].reset_index(drop=True)
    metadata.insert(0, "row", np.arange(len(metadata)))
    np.save(output / "spectra.npy", matrix, allow_pickle=False)
    metadata.to_csv(output / "metadata.csv", index=False, encoding="utf-8-sig")
    partitions = grouped_partitions(metadata)
    partitions.to_csv(output / "partitions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sources).to_csv(output / "sources.csv", index=False, encoding="utf-8-sig")
    report = {
        "shape": list(matrix.shape), "dtype": str(matrix.dtype),
        "raw_target_frames": int(len(all_metadata)), "identical_frames_removed": int(duplicate_mask.sum()),
        "source_files": int(metadata.source_file.nunique()),
        "split_counts": partitions.split.value_counts().to_dict(),
        "split_file_counts": partitions.groupby("split").source_file.nunique().to_dict(),
        "split_station_counts": partitions.groupby(["split", "station"]).size().reset_index(name="frames").to_dict(orient="records"),
        "spectra_sha256": full_digest(output / "spectra.npy"),
        "metadata_sha256": full_digest(output / "metadata.csv"),
        "partitions_sha256": full_digest(output / "partitions.csv"),
        "split_policy": "Within each multi-file station, whole files sorted by first timestamp: earlier train, middle validation, later test; single-file stations are external_test.",
        "ground_truth": "Unknown (-1). Quality checks do not establish normal/anomaly labels.",
        "original_data_modified": False,
    }
    (output / "dataset.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Rebuild deduplicated 87-108 MHz spectra with file-grouped chronological splits.")
    parser.add_argument("--data-root", default=str(ROOT.parents[1] / "数据"))
    parser.add_argument("--audit", default=str(ROOT / "data_audit_20261027/fscan_audit.csv"))
    parser.add_argument("--copies", default=str(ROOT / "data_audit_20261027/duplicate_copies.csv"))
    parser.add_argument("--output", required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
