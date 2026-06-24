#!/usr/bin/env python3
"""Analyze cross-band source-id presence patterns for one HSC patch."""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from astropy.table import Table


BANDS = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patch", default="4,5")
    parser.add_argument("--preprocessed-root", type=Path, default=Path("/nvme0/zc/scarlet/preprocessed"))
    parser.add_argument("--raw-root", type=Path, default=Path("/data1/czh23/Subaru"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/catalog_band_pattern_diagnostics_260611"))
    return parser.parse_args()


def read_table(path: Path) -> Table:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Table.read(path)


def choose_xy(table: Table) -> tuple[str | None, str | None]:
    for x_name, y_name in (
        ("base_SdssCentroid_x", "base_SdssCentroid_y"),
        ("base_SdssShape_x", "base_SdssShape_y"),
        ("slot_Centroid_x", "slot_Centroid_y"),
        ("base_NaiveCentroid_x", "base_NaiveCentroid_y"),
    ):
        if x_name in table.colnames and y_name in table.colnames:
            return x_name, y_name
    return None, None


def finite_array(values) -> np.ndarray:
    arr = np.asarray(values)
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    return arr


def load_band_sets(paths: dict[str, Path], *, leaf_only: bool = False) -> tuple[dict[str, set[int]], dict[int, dict[str, dict[str, float]]], dict[str, int]]:
    by_band: dict[str, set[int]] = {}
    info: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    n_rows: dict[str, int] = {}
    for band, path in paths.items():
        table = read_table(path)
        mask = np.ones(len(table), dtype=bool)
        if leaf_only and "deblend_nChild" in table.colnames:
            mask &= np.asarray(table["deblend_nChild"], dtype=int) == 0
        ids = np.asarray(table["id"][mask], dtype=np.int64)
        by_band[band] = set(int(x) for x in ids)
        n_rows[band] = int(len(ids))
        x_name, y_name = choose_xy(table)
        x = finite_array(table[x_name][mask]) if x_name else np.full(len(ids), np.nan)
        y = finite_array(table[y_name][mask]) if y_name else np.full(len(ids), np.nan)
        mag = finite_array(table["pu_mag"][mask]) if "pu_mag" in table.colnames else np.full(len(ids), np.nan)
        for source_id, xx, yy, mm in zip(ids, x, y, mag):
            info[int(source_id)][band] = {
                "x": float(xx) if np.isfinite(xx) else math.nan,
                "y": float(yy) if np.isfinite(yy) else math.nan,
                "mag": float(mm) if np.isfinite(mm) else math.nan,
            }
    return by_band, info, n_rows


def pattern_for(source_id: int, by_band: dict[str, set[int]]) -> str:
    return "".join("1" if source_id in by_band[band] else "0" for band in BANDS)


def has_hole(pattern: str) -> bool:
    first = pattern.find("1")
    last = pattern.rfind("1")
    return first >= 0 and "0" in pattern[first : last + 1]


def summarize(name: str, by_band: dict[str, set[int]], info: dict[int, dict[str, dict[str, float]]], n_rows: dict[str, int], out_dir: Path) -> dict[str, object]:
    all_ids = sorted(set().union(*by_band.values()))
    counts = Counter(pattern_for(source_id, by_band) for source_id in all_ids)
    hole_patterns = {pattern: count for pattern, count in counts.items() if has_hole(pattern)}

    rows = []
    for source_id in all_ids:
        pattern = pattern_for(source_id, by_band)
        if not has_hole(pattern):
            continue
        row = {
            "source_id": source_id,
            "pattern_GRIZY": pattern,
            "bands_present": ",".join(band for bit, band in zip(pattern, BANDS) if bit == "1"),
            "bands_missing": ",".join(band for bit, band in zip(pattern, BANDS) if bit == "0"),
        }
        for band in BANDS:
            band_info = info.get(source_id, {}).get(band, {})
            row[f"{band}_x"] = band_info.get("x", math.nan)
            row[f"{band}_y"] = band_info.get("y", math.nan)
            row[f"{band}_mag"] = band_info.get("mag", math.nan)
        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / f"{name}_non_monotonic_sources.csv"
    if rows:
        with detail_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    pattern_path = out_dir / f"{name}_pattern_counts.csv"
    with pattern_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pattern_GRIZY", "count", "has_hole", "bands_present", "bands_missing"])
        writer.writeheader()
        for pattern, count in counts.most_common():
            writer.writerow(
                {
                    "pattern_GRIZY": pattern,
                    "count": count,
                    "has_hole": int(has_hole(pattern)),
                    "bands_present": ",".join(band for bit, band in zip(pattern, BANDS) if bit == "1"),
                    "bands_missing": ",".join(band for bit, band in zip(pattern, BANDS) if bit == "0"),
                }
            )

    example_pattern = "10011"
    summary = {
        "name": name,
        "band_order": BANDS,
        "rows_by_band": n_rows,
        "unique_source_ids": len(all_ids),
        "all_five_bands": counts.get("11111", 0),
        "non_monotonic_sources": len(rows),
        "non_monotonic_fraction": float(len(rows) / len(all_ids)) if all_ids else math.nan,
        "pattern_GZY_without_RI_10011": counts.get(example_pattern, 0),
        "top_patterns": counts.most_common(12),
        "top_non_monotonic_patterns": sorted(hole_patterns.items(), key=lambda item: item[1], reverse=True)[:12],
        "pattern_counts_csv": str(pattern_path),
        "non_monotonic_sources_csv": str(detail_path) if rows else None,
    }
    (out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    return summary


def main() -> int:
    args = parse_args()
    tract = args.tract
    patch = args.patch
    clean_paths = {
        band: args.preprocessed_root / tract / patch / "band_reference_catalogs" / band / f"meas-{band}-{tract}-{patch}.fits"
        for band in BANDS
    }
    raw_paths = {
        band: args.raw_root / tract / band / patch / f"meas-{band}-{tract}-{patch}.fits"
        for band in BANDS
    }
    summaries = []
    clean_sets, clean_info, clean_n = load_band_sets(clean_paths)
    summaries.append(summarize("preprocessed_clean", clean_sets, clean_info, clean_n, args.out_dir))
    raw_sets, raw_info, raw_n = load_band_sets(raw_paths)
    summaries.append(summarize("raw_all", raw_sets, raw_info, raw_n, args.out_dir))
    raw_leaf_sets, raw_leaf_info, raw_leaf_n = load_band_sets(raw_paths, leaf_only=True)
    summaries.append(summarize("raw_leaf_nchild0", raw_leaf_sets, raw_leaf_info, raw_leaf_n, args.out_dir))
    print(json.dumps(summaries, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
