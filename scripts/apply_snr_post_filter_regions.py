#!/usr/bin/env python3
"""Apply AP2-SNR post filters to PU class CSVs and export DS9 regions."""

from __future__ import annotations

import argparse
import csv
import math
import os
import warnings
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")

import numpy as np
from astropy.table import Table


COLORS = {
    "clean": "green",
    "center_only": "orange",
    "ignore": "red",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--classes-csv", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("output/data_filter_0723/snr_post_filter_45"))
    p.add_argument("--prefix", required=True)
    p.add_argument("--band-kind", choices=("broad", "narrow"), required=True)
    p.add_argument("--catalog-hdu", type=int, default=1)
    p.add_argument("--ap2-flux-column", default="base_CircularApertureFlux_6_0_instFlux")
    p.add_argument("--ap2-err-column", default="base_CircularApertureFlux_6_0_instFluxErr")
    p.add_argument("--area-center-only-min", type=float, default=500.0)
    p.add_argument("--area-center-only-snr-max", type=float, default=8.0)
    p.add_argument("--broad-ignore-snr-max", type=float, default=3.0)
    p.add_argument("--broad-center-only-snr-max", type=float, default=5.0)
    p.add_argument("--narrow-ignore-snr-max", type=float, default=5.0)
    p.add_argument("--narrow-center-only-snr-max", type=float, default=8.0)
    p.add_argument("--dedup-close-center-arcsec", type=float, default=0.5)
    p.add_argument("--pixel-scale-arcsec", type=float, default=0.168)
    p.add_argument("--large-area-as-point", type=float, default=10000.0)
    p.add_argument("--dry-run", action="store_true", help="Compute and print summary without writing REG/CSV files.")
    return p.parse_args()


def _snr(flux: np.ndarray, err: np.ndarray) -> np.ndarray:
    out = np.full(len(flux), np.nan, dtype=np.float64)
    ok = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    out[ok] = flux[ok] / err[ok]
    return out


def _id_to_idx(table: Table) -> dict[int, int]:
    if "id" not in table.colnames:
        raise KeyError("catalog must contain id column")
    return {int(value): idx for idx, value in enumerate(np.asarray(table["id"]))}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _region_header(title: str) -> list[str]:
    return [
        "# Region file format: DS9 version 4.1",
        f"# {title}",
        'global color=green dashlist=8 3 width=2 font="helvetica 10 normal roman" '
        "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
        "physical",
    ]


def _region_line(row: dict[str, object], *, color: str, large_area_as_point: float) -> str:
    x = float(row["x_physical"])
    y = float(row["y_physical"])
    area = float(row.get("aperture_area") or "nan")
    sid = str(row.get("id", ""))
    snr = float(row.get("ap2_snr", "nan"))
    snr_label = "nan" if not np.isfinite(snr) else f"{snr:.2f}"
    reason = str(row.get("snr_post_reason", ""))
    comment = f"# color={color} width=2 tag={{id={sid}}} text={{AP2 SNR={snr_label}; {reason}}}"
    if np.isfinite(area) and area > large_area_as_point:
        return f"point({x:.6f},{y:.6f}) # point=circle color={color} width=2 tag={{id={sid}}} text={{AP2 SNR={snr_label}; {reason}}}"
    major = float(row["major_aperture"])
    minor = float(row["minor_aperture"])
    theta = float(row["theta_deg"])
    return f"ellipse({x:.6f},{y:.6f},{major:.6f},{minor:.6f},{theta:.6f}) {comment}"


def _classify_after_snr(row: dict[str, str], *, snr: float, args: argparse.Namespace) -> tuple[str, str]:
    old_class = row.get("class", "")
    if old_class not in {"clean", "center_only", "strict_center_only", "ignore"}:
        return old_class, "unchanged_non_pu_class"
    old_norm = "center_only" if old_class == "strict_center_only" else old_class

    if not np.isfinite(snr):
        return "ignore", "snr_invalid_to_ignore"

    if args.band_kind == "broad":
        ignore_snr_max = float(args.broad_ignore_snr_max)
        center_only_snr_max = float(args.broad_center_only_snr_max)
    else:
        ignore_snr_max = float(args.narrow_ignore_snr_max)
        center_only_snr_max = float(args.narrow_center_only_snr_max)

    if snr <= ignore_snr_max:
        return "ignore", f"snr_le_{ignore_snr_max:g}_to_ignore"

    if old_norm == "ignore":
        return "ignore", "kept_existing_ignore"

    try:
        area = float(row.get("aperture_area") or "nan")
    except ValueError:
        area = math.nan

    area_low_snr = (
        np.isfinite(area)
        and area > float(args.area_center_only_min)
        and snr <= float(args.area_center_only_snr_max)
    )
    transition_clean = old_norm == "clean" and snr < center_only_snr_max
    if area_low_snr:
        return "center_only", (
            f"area_gt_{float(args.area_center_only_min):g}_and_snr_le_"
            f"{float(args.area_center_only_snr_max):g}_to_center_only"
        )
    if transition_clean:
        return "center_only", f"clean_snr_lt_{center_only_snr_max:g}_to_center_only"

    return old_norm, "kept_existing_class"


def _float_from_row(row: dict[str, object], key: str, default: float = math.nan) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value


def _dedup_priority(row: dict[str, object]) -> tuple[int, float, float, float]:
    cls_rank = 2 if row.get("class") == "clean" else 1
    mag = _float_from_row(row, "pu_mag")
    if not np.isfinite(mag):
        mag = float("inf")
    snr = _float_from_row(row, "ap2_snr")
    if not np.isfinite(snr):
        snr = -float("inf")
    area = _float_from_row(row, "aperture_area")
    if not np.isfinite(area):
        area = float("inf")
    # Larger tuple wins: prefer clean, brighter source, higher SNR, then smaller aperture.
    return (cls_rank, -mag, snr, -area)


def _close_pairs(rows: list[dict[str, object]], radius_px: float) -> list[tuple[int, int, float]]:
    if len(rows) < 2 or radius_px <= 0.0:
        return []
    cell = max(radius_px, 1e-6)
    buckets: dict[tuple[int, int], list[int]] = {}
    xy: list[tuple[float, float]] = []
    for idx, row in enumerate(rows):
        x = _float_from_row(row, "x_physical")
        y = _float_from_row(row, "y_physical")
        xy.append((x, y))
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        key = (math.floor(x / cell), math.floor(y / cell))
        buckets.setdefault(key, []).append(idx)
    pairs: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()
    for idx, (x, y) in enumerate(xy):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        cx = math.floor(x / cell)
        cy = math.floor(y / cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for jdx in buckets.get((cx + dx, cy + dy), []):
                    if jdx <= idx:
                        continue
                    pair = (idx, jdx)
                    if pair in seen:
                        continue
                    seen.add(pair)
                    x2, y2 = xy[jdx]
                    dist = math.hypot(x - x2, y - y2)
                    if dist <= radius_px:
                        pairs.append((idx, jdx, dist))
    return pairs


def _apply_final_dedup(enriched: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    radius_px = float(args.dedup_close_center_arcsec) / max(float(args.pixel_scale_arcsec), 1e-12)
    candidate_indices = [
        idx for idx, row in enumerate(enriched) if row.get("class") in {"clean", "center_only"}
    ]
    candidate_rows = [enriched[idx] for idx in candidate_indices]
    pairs = _close_pairs(candidate_rows, radius_px)
    if not pairs:
        return {
            "dedup_radius_pix": radius_px,
            "dedup_close_pairs": 0,
            "dedup_demoted_to_ignore": 0,
            "dedup_demoted_ids": [],
        }

    neighbor_map: dict[int, list[tuple[int, float]]] = {idx: [] for idx in range(len(candidate_rows))}
    for i, j, dist in pairs:
        neighbor_map[i].append((j, dist))
        neighbor_map[j].append((i, dist))

    order = sorted(range(len(candidate_rows)), key=lambda idx: _dedup_priority(candidate_rows[idx]), reverse=True)
    removed: set[int] = set()
    keep_for: dict[int, tuple[int, float]] = {}
    for idx in order:
        if idx in removed:
            continue
        for jdx, dist in neighbor_map.get(idx, []):
            if jdx in removed:
                continue
            if _dedup_priority(candidate_rows[idx]) >= _dedup_priority(candidate_rows[jdx]):
                removed.add(jdx)
                keep_for[jdx] = (idx, dist)
    for local_idx in sorted(removed):
        global_idx = candidate_indices[local_idx]
        row = enriched[global_idx]
        keep_local, dist = keep_for[local_idx]
        keep_row = candidate_rows[keep_local]
        row["snr_post_pre_dedup_class"] = row.get("class", "")
        row["class"] = "ignore"
        old_reason = str(row.get("snr_post_reason", ""))
        row["snr_post_reason"] = (
            f"{old_reason};final_close_center_duplicate_to_ignore"
            if old_reason
            else "final_close_center_duplicate_to_ignore"
        )
        row["dedup_dropped"] = True
        row["dedup_keep_id"] = str(keep_row.get("id", ""))
        row["dedup_distance_pix"] = float(dist)
    for idx, row in enumerate(enriched):
        row.setdefault("snr_post_pre_dedup_class", row.get("class", ""))
        row.setdefault("dedup_dropped", False)
        row.setdefault("dedup_keep_id", "")
        row.setdefault("dedup_distance_pix", "")
    return {
        "dedup_radius_pix": radius_px,
        "dedup_close_pairs": len(pairs),
        "dedup_demoted_to_ignore": len(removed),
        "dedup_demoted_ids": [str(candidate_rows[idx].get("id", "")) for idx in sorted(removed)],
    }


def main() -> int:
    args = parse_args()
    if not bool(args.dry_run):
        args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(args.classes_csv)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*'second' did not parse as fits unit.*")
        catalog = Table.read(args.catalog, hdu=int(args.catalog_hdu))
    id_to_idx = _id_to_idx(catalog)
    snr_all = _snr(
        np.asarray(catalog[args.ap2_flux_column], dtype=np.float64),
        np.asarray(catalog[args.ap2_err_column], dtype=np.float64),
    )

    classes: dict[str, list[dict[str, object]]] = {"clean": [], "center_only": [], "ignore": []}
    enriched: list[dict[str, object]] = []
    for row in rows:
        sid = int(row["id"])
        idx = id_to_idx.get(sid)
        snr = float(snr_all[idx]) if idx is not None else math.nan
        new_class, reason = _classify_after_snr(row, snr=snr, args=args)
        out_row = {
            **row,
            "original_class": row.get("class", ""),
            "class": new_class,
            "ap2_snr": snr,
            "snr_post_reason": reason,
        }
        enriched.append(out_row)
    dedup_stats = _apply_final_dedup(enriched, args)

    classes: dict[str, list[dict[str, object]]] = {"clean": [], "center_only": [], "ignore": []}
    for out_row in enriched:
        if out_row.get("class") in classes:
            classes[str(out_row["class"])].append(out_row)

    if not bool(args.dry_run):
        for cls_name, cls_rows in classes.items():
            lines = _region_header(f"{args.prefix} {cls_name} after AP2-SNR post filter")
            lines.extend(
                _region_line(row, color=COLORS[cls_name], large_area_as_point=float(args.large_area_as_point))
                for row in cls_rows
            )
            (args.output_dir / f"{args.prefix}_snr_post_{cls_name}.reg").write_text("\n".join(lines) + "\n")

        fieldnames = list(enriched[0].keys()) if enriched else []
        csv_path = args.output_dir / f"{args.prefix}_snr_post_classes.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(enriched)

    before = Counter(row.get("original_class", "") for row in enriched)
    after = Counter(row.get("class", "") for row in enriched)
    transitions = Counter((row.get("original_class", ""), row.get("class", "")) for row in enriched)
    summary_lines = [
        f"prefix: {args.prefix}",
        f"band_kind: {args.band_kind}",
        f"dry_run: {bool(args.dry_run)}",
        f"before: {dict(before)}",
        f"after: {dict(after)}",
        f"dedup_radius_pix: {dedup_stats['dedup_radius_pix']:.6g}",
        f"dedup_close_pairs: {dedup_stats['dedup_close_pairs']}",
        f"dedup_demoted_to_ignore: {dedup_stats['dedup_demoted_to_ignore']}",
        f"dedup_demoted_ids_first10: {dedup_stats['dedup_demoted_ids'][:10]}",
        "transitions:",
    ]
    for (old, new), count in sorted(transitions.items()):
        summary_lines.append(f"  {old}->{new}: {count}")
    if not bool(args.dry_run):
        summary_path = args.output_dir / f"{args.prefix}_snr_post_summary.txt"
        summary_path.write_text("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
