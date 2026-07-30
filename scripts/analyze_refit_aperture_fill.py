#!/usr/bin/env python3
"""Diagnose refit aperture fill fractions and export DS9 regions."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")

import numpy as np


DEFAULT_BAND_DIRS = {
    "HSC-I": "45_I",
    "HSC-Z": "45_Z",
    "NB0816": "45_NB0816",
    "NB1010": "45_NB1010",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("output/data_filter_0723/refit_diagnostics"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/data_filter_0723/refit_aperture_fill_diagnostics"))
    parser.add_argument("--bands", nargs="+", default=["HSC-I", "HSC-Z", "NB1010", "NB0816"])
    parser.add_argument("--patch", default="4,5")
    parser.add_argument("--area-threshold", type=float, default=500.0)
    parser.add_argument("--low-threshold", type=float, default=0.3)
    parser.add_argument("--mid-threshold", type=float, default=0.5)
    parser.add_argument("--no-exclude-shape-flagged", action="store_true")
    parser.add_argument("--no-exclude-centroid-flagged", action="store_true")
    parser.add_argument("--large-area-as-point", type=float, default=10000.0)
    parser.add_argument("--hist-bins", type=int, default=60)
    return parser.parse_args()


def _float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, ""))
    except Exception:
        return default


def _bool(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "t", "yes", "y"}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _band_dir(input_root: Path, band: str) -> Path:
    candidates = []
    if band in DEFAULT_BAND_DIRS:
        candidates.append(input_root / DEFAULT_BAND_DIRS[band])
    short = band.replace("HSC-", "")
    candidates.extend(
        [
            input_root / f"45_{short}",
            input_root / f"45_{band}",
            input_root / band,
        ]
    )
    for candidate in candidates:
        if (candidate / "batch_heavyfp_kron_refit" / "batch_heavyfp_kron_refit.csv").exists():
            return candidate
    raise FileNotFoundError(f"no refit CSV found for {band} under {input_root}")


def _compute_row(row: dict[str, str]) -> dict[str, object] | None:
    target_radius = _float(row, "proxy_nan0_flux_aperture_radius")
    initial_radius = _float(row, "initial_determinant_radius")
    axis_a = _float(row, "axis_a")
    axis_b = _float(row, "axis_b")
    ap_pixels = _float(row, "aperture_pixel_count")
    if not all(np.isfinite(v) for v in (target_radius, initial_radius, axis_a, axis_b, ap_pixels)):
        return None
    if min(target_radius, initial_radius, axis_a, axis_b) <= 0:
        return None
    scale = target_radius / initial_radius
    major = axis_a * scale
    minor = axis_b * scale
    aperture_area = math.pi * major * minor
    if not np.isfinite(aperture_area) or aperture_area <= 0:
        return None
    ratio = ap_pixels / aperture_area
    official_radius = _float(row, "catalog_KronFlux_radius")
    official_aperture = 2.5 * official_radius if np.isfinite(official_radius) else math.nan
    return {
        "source_id": int(float(row["source_id"])),
        "x_image": _float(row, "x_image"),
        "y_image": _float(row, "y_image"),
        "major": major,
        "minor": minor,
        "theta_deg": _float(row, "theta_deg"),
        "aperture_area": aperture_area,
        "aperture_pixel_count": ap_pixels,
        "footprint_area": _float(row, "footprint_area"),
        "fill_ratio": ratio,
        "proxy_aperture_radius": target_radius,
        "official_aperture_radius": official_aperture,
        "proxy_gt_official_aperture": (
            np.isfinite(official_aperture) and target_radius > official_aperture * (1.0 + 1e-6)
        ),
        "official_psf_ab_mag": _float(row, "official_psf_ab_mag"),
        "measurement_surface": row.get("measurement_surface", ""),
        "base_sdss_shape_flag": _bool(row, "base_sdss_shape_flag"),
        "base_sdss_centroid_flag": _bool(row, "base_sdss_centroid_flag"),
        "status": row.get("status", ""),
    }


def _selected_rows(rows: Iterable[dict[str, str]], args: argparse.Namespace) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for row in rows:
        if row.get("status") != "ok" or not _bool(row, "proxy_nan0_good"):
            continue
        if not args.no_exclude_shape_flagged and _bool(row, "base_sdss_shape_flag"):
            continue
        if not args.no_exclude_centroid_flagged and _bool(row, "base_sdss_centroid_flag"):
            continue
        computed = _compute_row(row)
        if computed is not None:
            selected.append(computed)
    return selected


def _layer_name(ratio: float, args: argparse.Namespace) -> str:
    if ratio < float(args.low_threshold):
        return "fill_lt_0p3"
    if ratio < float(args.mid_threshold):
        return "fill_0p3_0p5"
    return "fill_ge_0p5"


def _layer_color(layer: str) -> str:
    return {
        "fill_lt_0p3": "red",
        "fill_0p3_0p5": "yellow",
        "fill_ge_0p5": "green",
    }[layer]


def _write_region(path: Path, band: str, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    title = (
        f"{band} patch {args.patch} proxy_flux_aperture fill ratio; "
        f"red < {args.low_threshold}, yellow {args.low_threshold}-{args.mid_threshold}, green >= {args.mid_threshold}"
    )
    lines = [
        "# Region file format: DS9 version 4.1",
        f"# {title}",
        'global color=green dashlist=8 3 width=2 font="helvetica 10 normal roman" '
        "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
        "image",
    ]
    # Draw larger ellipses first so smaller ones stay visible.
    for row in sorted(rows, key=lambda item: float(item["aperture_area"]), reverse=True):
        ratio = float(row["fill_ratio"])
        layer = _layer_name(ratio, args)
        color = _layer_color(layer)
        text = (
            f"id={row['source_id']} fill={ratio:.3f} area={float(row['aperture_area']):.1f} "
            f"apix={float(row['aperture_pixel_count']):.0f}"
        )
        x = float(row["x_image"])
        y = float(row["y_image"])
        area = float(row["aperture_area"])
        if area > float(args.large_area_as_point):
            lines.append(f"point({x:.3f},{y:.3f}) # point=circle color={color} text={{{text}}}")
        else:
            lines.append(
                f"ellipse({x:.3f},{y:.3f},{float(row['major']):.3f},{float(row['minor']):.3f},"
                f"{float(row['theta_deg']):.3f}) # color={color} text={{{text}}}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, band: str, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "band",
        "source_id",
        "x_image",
        "y_image",
        "fill_ratio",
        "ratio_bin",
        "aperture_area",
        "aperture_pixel_count",
        "footprint_area",
        "major",
        "minor",
        "theta_deg",
        "proxy_aperture_radius",
        "official_aperture_radius",
        "proxy_gt_official_aperture",
        "official_psf_ab_mag",
        "measurement_surface",
        "base_sdss_shape_flag",
        "base_sdss_centroid_flag",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, "") for field in fields}
            out["band"] = band
            out["ratio_bin"] = _layer_name(float(row["fill_ratio"]), args)
            writer.writerow(out)


def _write_histogram(path: Path, band: str, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray([float(row["fill_ratio"]) for row in rows if np.isfinite(float(row["fill_ratio"]))], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    if values.size:
        clipped = values[np.isfinite(values)]
        max_x = max(1.0, float(np.nanpercentile(clipped, 99.5))) if clipped.size else 1.0
        ax.hist(np.clip(clipped, 0.0, max_x), bins=int(args.hist_bins), color="#4477aa", alpha=0.85)
    ax.axvline(float(args.low_threshold), color="red", linestyle="--", linewidth=1.4, label=f"{args.low_threshold:g}")
    ax.axvline(float(args.mid_threshold), color="goldenrod", linestyle="--", linewidth=1.4, label=f"{args.mid_threshold:g}")
    ax.set_title(f"{band} patch {args.patch} aperture_pixel_count / aperture_area")
    ax.set_xlabel("fill ratio")
    ax.set_ylabel("source count")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _summary_row(band: str, rows: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    values = np.asarray([float(row["fill_ratio"]) for row in rows], dtype=float)
    area = np.asarray([float(row["aperture_area"]) for row in rows], dtype=float)
    area_gt = area > float(args.area_threshold)
    low = values < float(args.low_threshold)
    mid = (values >= float(args.low_threshold)) & (values < float(args.mid_threshold))
    high = values >= float(args.mid_threshold)
    return {
        "band": band,
        "selected_unflagged_proxy_good": int(len(rows)),
        "fill_lt_0p3": int(np.count_nonzero(low)),
        "fill_0p3_0p5": int(np.count_nonzero(mid)),
        "fill_ge_0p5": int(np.count_nonzero(high)),
        "area_gt_500": int(np.count_nonzero(area_gt)),
        "area_gt_500_fill_lt_0p3": int(np.count_nonzero(area_gt & low)),
        "area_gt_500_fill_0p3_0p5": int(np.count_nonzero(area_gt & mid)),
        "area_gt_500_fill_ge_0p5": int(np.count_nonzero(area_gt & high)),
        "proxy_gt_official_aperture": int(np.count_nonzero([bool(row["proxy_gt_official_aperture"]) for row in rows])),
        "fill_median": float(np.nanmedian(values)) if values.size else np.nan,
        "fill_p10": float(np.nanpercentile(values, 10)) if values.size else np.nan,
        "fill_p90": float(np.nanpercentile(values, 90)) if values.size else np.nan,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for band in args.bands:
        band_root = _band_dir(args.input_root, band)
        csv_path = band_root / "batch_heavyfp_kron_refit" / "batch_heavyfp_kron_refit.csv"
        rows = _selected_rows(_read_rows(csv_path), args)
        safe_band = band.replace("/", "_")
        _write_region(args.output_dir / f"{safe_band}_{args.patch.replace(',', '_')}_fill_ratio_proxy_flux_aperture.reg", band, rows, args)
        _write_csv(args.output_dir / f"{safe_band}_{args.patch.replace(',', '_')}_fill_ratio_proxy_flux_aperture.csv", band, rows, args)
        _write_histogram(args.output_dir / f"{safe_band}_{args.patch.replace(',', '_')}_fill_ratio_histogram.png", band, rows, args)
        summary_rows.append(_summary_row(band, rows, args))
        print(summary_rows[-1])

    fields = list(summary_rows[0].keys()) if summary_rows else []
    with (args.output_dir / "fill_ratio_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
