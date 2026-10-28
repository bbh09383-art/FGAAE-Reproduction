from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd


IX_DTYPE = np.dtype([
    ("ticks", "<i8"), ("offset_points", "<i8"), ("offset_bytes", "<i8"),
    ("band", "u1"), ("declared_points", "<u4"), ("start", "<f8"),
    ("end", "<f8"), ("start_index", "<u4"), ("step", "<f4"),
    ("count", "<u4"), ("reserved", "i1", (3,)),
])
EXCLUDED = {"venv", ".venv", "__pycache__", "build", "dist", ".git", ".idea"}
START_HZ = 87_000_000
END_HZ = 108_000_000


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def full_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sampled_digest(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with path.open("rb") as handle:
        for offset in sorted({0, max(0, size // 2 - 32768), max(0, size - 65536)}):
            handle.seek(offset)
            digest.update(handle.read(65536))
    return digest.hexdigest()


def quality_flags(values: np.ndarray) -> dict[str, np.ndarray]:
    promoted = values.astype(np.int32)
    return {
        "all_zero": np.all(promoted == 0, axis=1),
        "constant": np.ptp(promoted, axis=1) == 0,
        "int16_limit": np.any((promoted == -32768) | (promoted == 32767), axis=1),
        "abs_above_1500": np.any(np.abs(promoted) > 1500, axis=1),
    }


def station_name(path: Path) -> str:
    if "AAA原始数据整理之旧数据+三合海关" in path.parts:
        return "旧数据_站点待确认"
    return path.parent.parent.name


def audit_fscan(path: Path, root: Path):
    body = path.with_suffix(".md")
    result = {
        "stem": path.stem, "station": station_name(path), "ix_path": relative(path, root),
        "md_path": relative(body, root), "ix_bytes": path.stat().st_size,
        "md_bytes": body.stat().st_size if body.exists() else 0,
        "frames": 0, "band_geometry_frames": 0, "clean_band_frames": 0,
    }
    if path.stat().st_size == 0 or path.stat().st_size % IX_DTYPE.itemsize:
        return {**result, "status": "invalid_ix_record_size"}, None, None
    records = np.fromfile(path, dtype=IX_DTYPE)
    frame_starts = np.flatnonzero(records["start_index"] == 0)
    if not len(frame_starts) or frame_starts[0] != 0:
        return {**result, "status": "invalid_frame_start"}, None, None
    frame_ids = np.cumsum(records["start_index"] == 0) - 1
    frame_count = len(frame_starts)
    frame_points = np.add.reduceat(records["count"].astype(np.int64), frame_starts)
    result.update({
        "frames": frame_count, "ix_records": len(records),
        "start_mhz": float(records["start"].min() / 1e6),
        "end_mhz": float(records["end"].max() / 1e6),
        "steps_hz": json.dumps(np.unique(records["step"]).astype(float).tolist()),
        "declared_points": json.dumps(np.unique(records["declared_points"]).astype(int).tolist()),
        "partial_scan_frames": int(np.sum(frame_points != records["declared_points"][frame_starts])),
        "expected_md_bytes": int(records["count"].sum()) * 2,
        "first_time": str(datetime(1, 1, 1) + timedelta(microseconds=int(records["ticks"][0]) // 10)),
        "byte_offsets_contiguous": bool(np.array_equal(
            records["offset_bytes"],
            np.r_[0, np.cumsum(records["count"].astype(np.int64)[:-1]) * 2],
        )),
    })
    if not body.exists():
        return {**result, "status": "missing_md"}, None, records
    steps = np.unique(records["step"])
    if len(steps) != 1 or steps[0] <= 0:
        return {**result, "status": "mixed_or_invalid_step"}, None, records
    step = float(steps[0])
    width = round((END_HZ - START_HZ) / step)
    if width < 1 or not np.isclose(width * step, END_HZ - START_HZ):
        return {**result, "status": "noninteger_target_grid"}, None, records
    result["step_hz"] = step
    result["target_points"] = width
    lower = np.clip(np.ceil((START_HZ - records["start"]) / step - 1e-7), 0, records["count"]).astype(np.int64)
    upper = np.clip(np.ceil((END_HZ - records["start"]) / step - 1e-7), 0, records["count"]).astype(np.int64)
    selected_records = np.flatnonzero(upper > lower)
    values = np.zeros((frame_count, width), dtype=np.int16)
    coverage = np.zeros((frame_count, width), dtype=np.uint8)
    bad_frames = np.zeros(frame_count, dtype=bool)
    with body.open("rb") as handle:
        for record_index in selected_records:
            record = records[record_index]
            frame_index = int(frame_ids[record_index])
            first_frequency = float(record["start"]) + lower[record_index] * step
            target_index_float = (first_frequency - START_HZ) / step
            target_index = round(target_index_float)
            count = int(upper[record_index] - lower[record_index])
            byte_offset = int(record["offset_bytes"]) + 2 * int(lower[record_index])
            if (
                not np.isclose(target_index, target_index_float, atol=1e-6)
                or target_index < 0 or target_index + count > width
                or byte_offset < 0 or byte_offset + 2 * count > result["md_bytes"]
            ):
                bad_frames[frame_index] = True
                continue
            handle.seek(byte_offset)
            packet = np.frombuffer(handle.read(2 * count), dtype="<i2")
            values[frame_index, target_index:target_index + count] = packet
            coverage[frame_index, target_index:target_index + count] += 1
    geometry = np.all(coverage == 1, axis=1) & ~bad_frames
    flags = quality_flags(values)
    clean = geometry.copy()
    for name, mask in flags.items():
        result[f"band_{name}_frames"] = int(np.sum(mask & geometry))
        clean &= ~mask
    result.update({
        "band_geometry_frames": int(geometry.sum()), "clean_band_frames": int(clean.sum()),
        "band_min_raw": int(values[geometry].min()) if geometry.any() else None,
        "band_max_raw": int(values[geometry].max()) if geometry.any() else None,
        "md_length_matches_ix": result["md_bytes"] == result["expected_md_bytes"],
        "status": "native_840_candidate" if width == 840 else "separate_grid_candidate",
    })
    values[~geometry] = 0
    return result, values, records


def count_text_rows(path: Path) -> int:
    count = 0
    last_byte = b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            count += block.count(b"\n")
            last_byte = block[-1:]
    return count + int(bool(last_byte) and last_byte != b"\n")


def audit_text(path: Path, root: Path, band: np.ndarray | None, raw_result: dict, records):
    with path.open("rb") as handle:
        first_lines = [handle.readline() for _ in range(3)]
        handle.seek(max(0, path.stat().st_size - 4 * 1024 * 1024))
        tail = handle.read().splitlines()
    row_count = count_text_rows(path)
    samples = [(index, line) for index, line in enumerate(first_lines) if line.strip()]
    if tail:
        samples.append((row_count - 1, tail[-1]))
    row = {
        "stem": path.stem, "path": relative(path, root), "bytes": path.stat().st_size,
        "rows": row_count, "raw_frames": raw_result.get("frames"),
        "sampled_rows": len(samples), "sampled_columns": [], "band_matches_raw": 0,
        "verification": "all bytes counted for row count; first three and final rows compared numerically",
    }
    if records is None or band is None:
        return {**row, "status": "raw_reference_unavailable"}
    lower = round((START_HZ - float(records["start"][0])) / float(records["step"][0]))
    expected_width = band.shape[1]
    mismatches = []
    for row_index, line in samples:
        values = np.fromstring(line.decode("ascii").strip(), sep=" ", dtype=np.float64)
        row["sampled_columns"].append(len(values))
        if (
            row_index < len(band) and len(values) >= lower + expected_width
            and np.array_equal(values[lower:lower + expected_width], band[row_index])
        ):
            row["band_matches_raw"] += 1
        else:
            mismatches.append(row_index)
    row["sampled_columns"] = json.dumps(row["sampled_columns"])
    row["mismatching_sample_rows"] = json.dumps(mismatches)
    row["status"] = "sample_matches_raw" if not mismatches else "do_not_use_before_reextracting"
    return row


def audit_road(path: Path, root: Path):
    size = path.stat().st_size
    row = {"path": relative(path, root), "bytes": size, "frames": 0, "clean_band_frames": 0}
    if size < 52:
        return {**row, "status": "empty_or_short"}
    with path.open("rb") as handle:
        header = handle.read(52)
    start, step, count = struct.unpack_from("<dfI", header, 36)
    frame_bytes = 52 + 2 * count
    row.update({"start_mhz": start / 1e6, "step_hz": step, "points": count})
    if not count or count > 2_000_000 or step <= 0 or not np.isfinite([start, step]).all() or size % frame_bytes:
        return {**row, "status": "unknown_or_variable_format"}
    frame_count = size // frame_bytes
    row["frames"] = frame_count
    dtype = np.dtype({
        "names": ["magic", "start", "step", "count"],
        "formats": ["<u4", "<f8", "<f4", "<u4"],
        "offsets": [0, 36, 44, 48], "itemsize": frame_bytes,
    })
    headers = np.memmap(path, mode="r", dtype=dtype, shape=(frame_count,))
    valid_headers = (
        (headers["magic"] == 0xABCDABCD) & (headers["start"] == start)
        & (headers["step"] == step) & (headers["count"] == count)
    )
    lower = round((START_HZ - start) / step)
    width = round((END_HZ - START_HZ) / step)
    row["bad_headers"] = int((~valid_headers).sum())
    row["end_mhz"] = (start + (count - 1) * step) / 1e6
    if lower < 0 or lower + width > count or not np.isclose(start + lower * step, START_HZ):
        return {**row, "status": "target_band_unavailable"}
    mapped = np.memmap(path, mode="r", dtype="u1")
    spectra = np.ndarray((frame_count, count), dtype="<i2", buffer=mapped, offset=52, strides=(frame_bytes, 2))
    selected = spectra[:, lower:lower + width].copy()
    flags = quality_flags(selected)
    clean = valid_headers.copy()
    for name, mask in flags.items():
        row[f"band_{name}_frames"] = int(np.sum(mask & valid_headers))
        clean &= ~mask
    row.update({
        "target_points": width, "clean_band_frames": int(clean.sum()),
        "band_min_raw": int(selected[valid_headers].min()) if valid_headers.any() else None,
        "band_max_raw": int(selected[valid_headers].max()) if valid_headers.any() else None,
        "status": "inferred_52byte_header_int16_needs_unit_confirmation",
    })
    return row


def audit_iq_metadata(path: Path, all_files: list[Path], root: Path, output: Path):
    namespace = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        strings = ["".join(node.itertext()) for node in ET.fromstring(
            archive.read("xl/sharedStrings.xml")
        ).findall("s:si", namespace)]
        rows = []
        for row in ET.fromstring(archive.read("xl/worksheets/sheet1.xml")).findall("s:sheetData/s:row", namespace):
            values = {}
            for cell in row.findall("s:c", namespace):
                value = cell.find("s:v", namespace)
                if value is not None:
                    column = "".join(character for character in cell.attrib["r"] if character.isalpha())
                    values[column] = strings[int(value.text)] if cell.get("t") == "s" else value.text
            rows.append(values)
    headings = rows[0]
    table = pd.DataFrame([{headings[column]: row.get(column, "") for column in headings} for row in rows[1:]])
    local_names = {item.name for item in all_files}
    for column in ["iqfileurl", "spmfileurl"]:
        table[column.replace("url", "_present_locally")] = table[column].map(
            lambda value: str(value).rsplit("/", 1)[-1] in local_names
        )
    table.to_csv(output / "iq_metadata_audit.csv", index=False, encoding="utf-8-sig")
    invalid = {"", "\\N", "nan"}
    return {
        "table_path": relative(path, root), "metadata_rows": len(table),
        "modulation_label_rows": int((~table["modulation"].isin(invalid)).sum()),
        "communication_system_label_rows": int((~table["comsystem"].isin(invalid)).sum()),
        "modulation_counts": dict(Counter(table["modulation"])),
        "label_warning": "modulation/system labels are NOT normal/anomaly labels; confidence columns may be machine predictions",
    }


def run_audit(arguments):
    root = Path(arguments.data_root).resolve()
    output = Path(arguments.output).resolve()
    if output == root or root in output.parents:
        raise ValueError("Audit output must be outside the original data directory.")
    output.mkdir(parents=True, exist_ok=False)
    all_files = sorted(path for path in root.rglob("*") if path.is_file() and not EXCLUDED.intersection(path.relative_to(root).parts))
    inventory = pd.DataFrame([
        {"path": relative(path, root), "extension": path.suffix.lower(), "bytes": path.stat().st_size}
        for path in all_files
    ])
    inventory.to_csv(output / "file_inventory.csv", index=False, encoding="utf-8-sig")
    archive_rows = []
    archived_names = defaultdict(list)
    for path in all_files:
        if path.suffix.lower() != ".zip":
            continue
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                if entry.is_dir():
                    continue
                archive_rows.append({
                    "archive": relative(path, root), "member": entry.filename,
                    "bytes": entry.file_size, "crc32": f"{entry.CRC:08x}",
                })
                archived_names[entry.filename.replace("\\", "/").rsplit("/", 1)[-1]].append(relative(path, root))
    pd.DataFrame(archive_rows).to_csv(output / "archive_inventory.csv", index=False, encoding="utf-8-sig")
    grouped = defaultdict(list)
    for path in all_files:
        if path.suffix.lower() == ".ix":
            grouped[path.stem].append(path)
    raw_rows, duplicate_rows, text_rows = [], [], []
    for number, (stem, paths) in enumerate(sorted(grouped.items()), 1):
        paths.sort(key=lambda path: (not path.with_suffix(".md").exists(), "test" in path.parts, len(str(path))))
        preferred = paths[0]
        ix_hashes = {str(path): full_digest(path) for path in paths}
        body_hashes = {str(path): sampled_digest(path.with_suffix(".md")) for path in paths if path.with_suffix(".md").exists()}
        row, values, records = audit_fscan(preferred, root)
        row.update({
            "ix_copies": len(paths), "ix_full_hash_variants": len(set(ix_hashes.values())),
            "md_sample_hash_variants": len(set(body_hashes.values())),
            "archived_md_locations": json.dumps(archived_names.get(stem + ".md", []), ensure_ascii=False),
        })
        if row["ix_full_hash_variants"] > 1 or row["md_sample_hash_variants"] > 1:
            row["status"] = "conflicting_copies_review_required"
        raw_rows.append(row)
        for path in paths:
            duplicate_rows.append({
                "stem": stem, "ix_path": relative(path, root), "preferred": path == preferred,
                "ix_sha256": ix_hashes[str(path)],
                "md_three_window_sha256": body_hashes.get(str(path), "MISSING"),
                "verification": "IX full hash; MD size and three 64-KiB windows only, not full-body proof",
            })
        text_path = root / "提取出的数据9.2/md" / f"{stem}.txt"
        if text_path.exists() and not arguments.skip_text:
            text_rows.append(audit_text(text_path, root, values, row, records))
        if number % 10 == 0 or number == len(grouped):
            print(f"FScan audit {number}/{len(grouped)}", flush=True)
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(output / "fscan_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(duplicate_rows).to_csv(output / "duplicate_copies.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(text_rows).to_csv(output / "extracted_text_audit.csv", index=False, encoding="utf-8-sig")
    road_rows = [audit_road(path, root) for path in all_files if path.suffix.lower() == ".bin"]
    road = pd.DataFrame(road_rows)
    road.to_csv(output / "road_bin_audit.csv", index=False, encoding="utf-8-sig")
    iq_tables = [path for path in all_files if path.name == "吉林省IQ数据列表.xlsx"]
    iq_summary = audit_iq_metadata(iq_tables[0], all_files, root, output) if iq_tables else {}
    group_summary = raw.groupby(["station", "step_hz"], dropna=False).agg(
        files=("stem", "size"), raw_frames=("frames", "sum"),
        band_geometry_frames=("band_geometry_frames", "sum"),
        clean_band_frames=("clean_band_frames", "sum"),
    ).reset_index()
    group_summary.to_csv(output / "station_summary.csv", index=False, encoding="utf-8-sig")
    native = raw[raw["status"] == "native_840_candidate"]
    grid_summary = raw.groupby("target_points", dropna=False).agg(
        files=("stem", "size"), geometry_frames=("band_geometry_frames", "sum"),
        clean_frames=("clean_band_frames", "sum"),
    ).reset_index()
    missing = raw[raw["status"] == "missing_md"]
    summary = {
        "data_root": str(root), "target_band_hz": [START_HZ, END_HZ],
        "files_excluding_environments_builds": len(all_files),
        "unique_fscan_stems": len(raw), "ix_copies": len(duplicate_rows),
        "native_840_files": len(native), "native_840_geometry_frames": int(native["band_geometry_frames"].sum()),
        "native_840_clean_frames": int(native["clean_band_frames"].sum()),
        "grid_summary": json.loads(grid_summary.to_json(orient="records")),
        "missing_md": missing[["stem", "archived_md_locations"]].to_dict(orient="records"),
        "text_rows_audited": len(text_rows), "text_status_counts": dict(Counter(row["status"] for row in text_rows)),
        "road_bin_files": len(road), "road_frames": int(road["frames"].sum()),
        "road_clean_band_frames_inferred_format": int(road["clean_band_frames"].sum()),
        "road_empty_files": road.loc[road["status"] == "empty_or_short", "path"].tolist(),
        "iq_spm_counts": dict(Counter(path.suffix.lower() for path in all_files if path.suffix.lower() in {".iqd", ".spm"})),
        "iq_metadata": iq_summary,
        "scope": [
            "Read-only audit of original data. No training, no bulk conversion, no archive extraction.",
            "FScan: all IX records and every recoverable 87-108 MHz frame checked; other frequencies not quality-screened.",
            "Clean means structural/heuristic quality checks passed, not a normal/anomaly ground-truth label.",
            "abs(raw)>1500 is a conservative legacy-unit heuristic, not a proven instrument saturation limit.",
            "Text row counts scan every byte, but numeric agreement is sampled at first three/final rows only.",
            "Duplicates grouped by stem, full IX hash and sampled MD fingerprint; no full 28-GB-body cryptographic deduplication.",
            "ZIP central directories inspected, payload CRC integrity not verified; nested ZIP and RAR payloads not decoded.",
            "Road BIN layout inferred from repeated headers; physical units and instrument invalid flags still need confirmation.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("STATIONS\n" + group_summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Read-only radio-data suitability audit for 87-108 MHz FGAAE input.")
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[2] / "数据"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-text", action="store_true")
    run_audit(parser.parse_args())


if __name__ == "__main__":
    main()
