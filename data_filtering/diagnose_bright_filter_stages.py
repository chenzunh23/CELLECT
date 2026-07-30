#!/usr/bin/env python3
"""Visualize which bright sources are removed by early PU filter stages."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.table import Table
    from astropy.units import UnitsWarning
    from astropy.visualization import ZScaleInterval
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse, Patch
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires astropy and matplotlib.") from exc

from data_filtering.pu_source_filter import _kron_ellipse, attach_kron_refit_radius
from data_filtering.sam_input_scaling import build_bright_mask

warnings.filterwarnings("ignore", category=UnitsWarning)
warnings.filterwarnings("ignore", message="Warning: converting a masked element to nan.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/data/shared/Subaru"))
    parser.add_argument("--refit-root", type=Path, default=Path("/data/czh23/refit"))
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patches", nargs="+", default=["4,5", "6,1"])
    parser.add_argument("--bands", nargs="+", default=["HSC-I", "HSC-Y"])
    parser.add_argument("--out-dir", type=Path, default=Path("output/data_filter_0729/bright_filter_stage_diagnostics"))
    parser.add_argument("--mag-threshold", type=float, default=22.0)
    parser.add_argument("--zeropoint", type=float, default=27.0)
    parser.add_argument("--mag-column", default="ext_photometryKron_KronFlux_instFlux")
    parser.add_argument("--ap2-flux-column", default="base_CircularApertureFlux_6_0_instFlux")
    parser.add_argument("--kron-flux-column", default="ext_photometryKron_KronFlux_instFlux")
    parser.add_argument("--radius-column", default="proxy_nan0_flux_aperture_radius")
    parser.add_argument("--good-column", default="proxy_nan0_good")
    parser.add_argument("--ellipse-sigma", type=float, default=1.0)
    parser.add_argument("--min-axis", type=float, default=1.5)
    parser.add_argument("--a-area-max", type=float, default=10000.0)
    parser.add_argument("--a-faint-area-max", type=float, default=900.0)
    parser.add_argument("--a-faint-mag-min", type=float, default=28.0)
    parser.add_argument("--axis-ratio-max", type=float, default=5.0)
    parser.add_argument("--close-center-arcsec", type=float, default=0.5)
    parser.add_argument("--include-ap2-filter", action="store_true", help="After refit/A/axis/close, split by catalog abs(AP2-Kron) magnitude difference.")
    parser.add_argument("--refined-bright-ap2", action="store_true", help="Use bright-region-aware AP2/Kron rules for mag<threshold sources.")
    parser.add_argument("--ap2-kron-abs-max", type=float, default=1.0)
    parser.add_argument("--ap2-kron-mid-max", type=float, default=2.0)
    parser.add_argument("--large-bright-region-area-min", type=float, default=1000.0)
    parser.add_argument("--bright-mask-mode", choices=("log-lupton", "zscore-lupton-log", "anscombe", "raw", "none"), default="log-lupton")
    parser.add_argument("--bright-log-a", type=float, default=300.0)
    parser.add_argument("--bright-log-high-percentile", type=float, default=99.5)
    parser.add_argument("--bright-lupton-stretch", type=float, default=0.5)
    parser.add_argument("--bright-lupton-q", type=float, default=20.0)
    parser.add_argument("--bright-anscombe-scale", type=float, default=1000.0)
    parser.add_argument("--bright-z-threshold", type=float, default=3.0)
    parser.add_argument("--bright-mask-dilate", type=int, default=2)
    parser.add_argument("--pixel-scale-arcsec", type=float, default=0.168)
    parser.add_argument("--source-filter", choices=("all", "nchild0"), default="nchild0")
    parser.add_argument("--max-ellipse-area-to-draw", type=float, default=40000.0)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def calexp_path(data_root: Path, tract: str, band: str, patch: str) -> Path:
    return data_root / str(tract) / band / patch / f"calexp-{band}-{tract}-{patch}.fits"


def meas_path(data_root: Path, tract: str, band: str, patch: str) -> Path:
    return data_root / str(tract) / band / patch / f"meas-{band}-{tract}-{patch}.fits"


def refit_path(refit_root: Path, tract: str, band: str, patch: str) -> Path:
    return refit_root / str(tract) / band / patch / "batch_heavyfp_kron_refit" / "batch_heavyfp_kron_refit.csv"


def read_image_and_origin(path: Path) -> tuple[np.ndarray, tuple[float, float]]:
    with fits.open(path, memmap=True) as hdul:
        for idx in (1, 0):
            if idx < len(hdul) and hdul[idx].data is not None and np.asarray(hdul[idx].data).ndim == 2:
                data = np.asarray(hdul[idx].data, dtype=np.float32)
                header = hdul[idx].header
                return data, (float(header.get("LTV1", 0.0)), float(header.get("LTV2", 0.0)))
        for hdu in hdul:
            if hdu.data is not None and np.asarray(hdu.data).ndim == 2:
                data = np.asarray(hdu.data, dtype=np.float32)
                header = hdu.header
                return data, (float(header.get("LTV1", 0.0)), float(header.get("LTV2", 0.0)))
    raise ValueError(f"no 2D image found: {path}")


def finite_float_column(table: Table, names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        if name in table.colnames:
            values = np.asarray(table[name], dtype=np.float64)
            if np.any(np.isfinite(values)):
                return values
    return np.full(len(table), np.nan, dtype=np.float64)


def local_xy(table: Table, ltv: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    x = finite_float_column(table, ("base_SdssCentroid_x", "slot_Centroid_x", "x", "centroid_x"))
    y = finite_float_column(table, ("base_SdssCentroid_y", "slot_Centroid_y", "y", "centroid_y"))
    return x + float(ltv[0]), y + float(ltv[1])


def mag_from_table(table: Table, column: str, zeropoint: float) -> np.ndarray:
    flux = finite_float_column(
        table,
        (
            column,
            "ext_photometryKron_KronFlux_instFlux",
            "base_PsfFlux_instFlux",
            "modelfit_CModel_instFlux",
            "base_SdssShape_instFlux",
        ),
    )
    mag = np.full(len(table), np.nan, dtype=np.float64)
    valid = np.isfinite(flux) & (flux > 0.0)
    mag[valid] = float(zeropoint) - 2.5 * np.log10(flux[valid])
    return mag


def mag_from_exact_column(table: Table, column: str, zeropoint: float) -> np.ndarray:
    mag = np.full(len(table), np.nan, dtype=np.float64)
    if column not in table.colnames:
        return mag
    flux = np.asarray(table[column], dtype=np.float64)
    valid = np.isfinite(flux) & (flux > 0.0)
    mag[valid] = float(zeropoint) - 2.5 * np.log10(flux[valid])
    return mag


def source_mask(table: Table, mode: str) -> np.ndarray:
    mask = np.ones(len(table), dtype=bool)
    if mode == "nchild0" and "deblend_nChild" in table.colnames:
        mask &= np.asarray(table["deblend_nChild"], dtype=np.int64) == 0
    return mask


def candidate_pairs(x: np.ndarray, y: np.ndarray, radius: float) -> list[tuple[int, int]]:
    valid = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    if valid.size < 2:
        return []
    cell = max(float(radius), 1e-6)
    buckets: dict[tuple[int, int], list[int]] = {}
    for idx in valid:
        key = (int(math.floor(float(x[idx]) / cell)), int(math.floor(float(y[idx]) / cell)))
        buckets.setdefault(key, []).append(int(idx))
    out: list[tuple[int, int]] = []
    for key, items in buckets.items():
        kx, ky = key
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                other = buckets.get((kx + dx, ky + dy), [])
                for i in items:
                    for j in other:
                        if j <= i:
                            continue
                        if (float(x[i] - x[j]) ** 2 + float(y[i] - y[j]) ** 2) <= radius * radius:
                            out.append((i, j))
    return out


def component_area_image(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros(mask.shape, dtype=np.float32)
    if not np.any(mask):
        return out
    try:
        from scipy import ndimage

        labels, count = ndimage.label(mask)
        if count <= 0:
            return out
        areas = np.bincount(labels.ravel())
        out[mask] = areas[labels[mask]].astype(np.float32)
    except Exception:
        out[mask] = float(np.count_nonzero(mask))
    return out


def classify_stages(
    table: Table,
    args: argparse.Namespace,
    ltv: tuple[float, float],
    *,
    bright_region_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    x, y = local_xy(table, ltv)
    mag = mag_from_table(table, args.mag_column, args.zeropoint)
    a, b, theta, area = _kron_ellipse(
        table,
        sigma=float(args.ellipse_sigma),
        min_axis=float(args.min_axis),
        require_refit_match=True,
    )
    matched = (
        np.asarray(table["pu_refit_kron_radius_matched"], dtype=bool)
        if "pu_refit_kron_radius_matched" in table.colnames
        else np.zeros(len(table), dtype=bool)
    )
    base = np.isfinite(x) & np.isfinite(y) & source_mask(table, args.source_filter)
    bright = base & np.isfinite(mag) & (mag < float(args.mag_threshold))
    valid_shape = matched & np.isfinite(a) & np.isfinite(b) & np.isfinite(theta) & np.isfinite(area) & (a > 0) & (b > 0)
    refit_removed = bright & ~valid_shape

    eligible = base & valid_shape
    removed_a = eligible & (
        (np.isfinite(area) & (area > float(args.a_area_max)))
        | (
            np.isfinite(area)
            & (area > float(args.a_faint_area_max))
            & np.isfinite(mag)
            & (mag > float(args.a_faint_mag_min))
        )
    )
    after_a = eligible & ~removed_a

    axis_ratio = np.full(len(table), np.nan, dtype=np.float64)
    axis_min = np.minimum(np.abs(a), np.abs(b))
    axis_max = np.maximum(np.abs(a), np.abs(b))
    axis_valid = np.isfinite(axis_min) & np.isfinite(axis_max) & (axis_min > 0.0)
    axis_ratio[axis_valid] = axis_max[axis_valid] / axis_min[axis_valid]
    removed_axis = after_a & axis_valid & (axis_ratio > float(args.axis_ratio_max))
    after_axis = after_a & ~removed_axis

    removed_close = np.zeros(len(table), dtype=bool)
    close_radius_px = float(args.close_center_arcsec) / max(float(args.pixel_scale_arcsec), 1e-6)
    for i, j in candidate_pairs(x[after_axis], y[after_axis], close_radius_px):
        global_indices = np.flatnonzero(after_axis)
        ii = int(global_indices[i])
        jj = int(global_indices[j])
        mi = float(mag[ii]) if np.isfinite(mag[ii]) else float("inf")
        mj = float(mag[jj]) if np.isfinite(mag[jj]) else float("inf")
        if mi > mj:
            drop = ii
        elif mj > mi:
            drop = jj
        else:
            drop = ii if float(area[ii]) >= float(area[jj]) else jj
        removed_close[drop] = True
    after_close = after_axis & ~removed_close

    ap2_mag = mag_from_exact_column(table, args.ap2_flux_column, args.zeropoint)
    kron_mag = mag_from_exact_column(table, args.kron_flux_column, args.zeropoint)
    ap2_kron_diff = ap2_mag - kron_mag
    absdiff = np.abs(ap2_kron_diff)
    ap2_valid = np.isfinite(absdiff)
    center_in_bright_region = np.zeros(len(table), dtype=bool)
    bright_region_area = np.zeros(len(table), dtype=np.float32)
    if bright_region_mask is not None:
        bright_area_map = component_area_image(bright_region_mask)
        ix = np.rint(x).astype(np.int64)
        iy = np.rint(y).astype(np.int64)
        inside = (ix >= 0) & (iy >= 0) & (ix < bright_region_mask.shape[1]) & (iy < bright_region_mask.shape[0])
        center_in_bright_region[inside] = bright_region_mask[iy[inside], ix[inside]]
        bright_region_area[inside] = bright_area_map[iy[inside], ix[inside]]
    removed_ap2_invalid = np.zeros(len(table), dtype=bool)
    removed_ap2_1_2 = np.zeros(len(table), dtype=bool)
    removed_ap2_gt2 = np.zeros(len(table), dtype=bool)
    removed_ap2_outside_bright_ge1 = np.zeros(len(table), dtype=bool)
    removed_ap2_small_bright_ge2 = np.zeros(len(table), dtype=bool)
    remaining_ap2_normal = np.zeros(len(table), dtype=bool)
    remaining_ap2_outside_bright_invalid = np.zeros(len(table), dtype=bool)
    remaining_ap2_relaxed_small_bright = np.zeros(len(table), dtype=bool)
    remaining_ap2_skipped_large_bright = np.zeros(len(table), dtype=bool)
    if bool(args.include_ap2_filter):
        if bool(args.refined_bright_ap2):
            bright_source = bright
            small_bright_region = (
                bright_source
                & center_in_bright_region
                & (bright_region_area > 0)
                & (bright_region_area < float(args.large_bright_region_area_min))
            )
            large_bright_region = (
                bright_source
                & center_in_bright_region
                & (bright_region_area >= float(args.large_bright_region_area_min))
            )
            outside_bright_region = bright_source & ~small_bright_region & ~large_bright_region
            ordinary_nonbright = ~bright_source
            outside_or_nonbright = outside_bright_region | ordinary_nonbright
            removed_ap2_invalid = after_close & ordinary_nonbright & ~ap2_valid
            remaining_ap2_outside_bright_invalid = after_close & outside_bright_region & ~ap2_valid
            remaining_ap2_skipped_large_bright = after_close & large_bright_region
            remaining_ap2_relaxed_small_bright = (
                after_close
                & small_bright_region
                & ap2_valid
                & (absdiff < float(args.ap2_kron_mid_max))
            )
            removed_ap2_small_bright_ge2 = (
                after_close
                & small_bright_region
                & ap2_valid
                & (absdiff >= float(args.ap2_kron_mid_max))
            )
            remaining_ap2_normal = (
                after_close
                & outside_or_nonbright
                & ap2_valid
                & (absdiff < float(args.ap2_kron_abs_max))
            )
            removed_ap2_outside_bright_ge1 = (
                after_close
                & outside_bright_region
                & ap2_valid
                & (absdiff >= float(args.ap2_kron_abs_max))
            )
            remaining = (
                remaining_ap2_skipped_large_bright
                | remaining_ap2_relaxed_small_bright
                | remaining_ap2_normal
                | remaining_ap2_outside_bright_invalid
            )
        else:
            removed_ap2_invalid = after_close & ~ap2_valid
            removed_ap2_1_2 = (
                after_close
                & ap2_valid
                & (absdiff >= float(args.ap2_kron_abs_max))
                & (absdiff < float(args.ap2_kron_mid_max))
            )
            removed_ap2_gt2 = after_close & ap2_valid & (absdiff >= float(args.ap2_kron_mid_max))
            remaining = after_close & ap2_valid & (absdiff < float(args.ap2_kron_abs_max))
    else:
        remaining = after_close

    status = np.full(len(table), "not_bright_or_not_base", dtype=object)
    if bool(args.refined_bright_ap2):
        status[bright & remaining_ap2_normal] = "remaining_ap2_absdiff_lt1"
        status[bright & remaining_ap2_outside_bright_invalid] = "remaining_outside_bright_region_ap2_invalid"
        status[bright & remaining_ap2_relaxed_small_bright] = "remaining_small_bright_region_absdiff_lt2"
        status[bright & remaining_ap2_skipped_large_bright] = "remaining_large_bright_region_ap2_skipped"
        status[bright & removed_ap2_outside_bright_ge1] = "removed_outside_bright_region_absdiff_ge1"
        status[bright & removed_ap2_small_bright_ge2] = "removed_small_bright_region_absdiff_ge2"
    else:
        status[bright & remaining] = "remaining_after_A_axis_close_ap2" if bool(args.include_ap2_filter) else "remaining_after_A_axis_close"
    status[bright & removed_ap2_invalid] = "removed_ap2_kron_invalid"
    status[bright & removed_ap2_1_2] = "removed_ap2_kron_absdiff_1_2"
    status[bright & removed_ap2_gt2] = "removed_ap2_kron_absdiff_gt2"
    status[bright & removed_close] = "removed_close_pair_dimmer"
    status[bright & removed_axis] = "removed_axis_ratio"
    status[bright & removed_a] = "removed_A_filter"
    status[refit_removed] = "removed_refit_missing_or_bad"
    return {
        "x": x,
        "y": y,
        "mag": mag,
        "a": a,
        "b": b,
        "theta": theta,
        "area": area,
        "axis_ratio": axis_ratio,
        "ap2_mag": ap2_mag,
        "kron_mag": kron_mag,
        "ap2_kron_diff": ap2_kron_diff,
        "ap2_kron_absdiff": absdiff,
        "bright_region_center": center_in_bright_region,
        "bright_region_area": bright_region_area,
        "matched": matched,
        "bright": bright,
        "status": status,
    }


def write_csv(path: Path, table: Table, stage: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(table["id"], dtype=np.int64) if "id" in table.colnames else np.arange(len(table), dtype=np.int64)
    bright_idx = np.flatnonzero(stage["bright"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "source_id",
            "x",
            "y",
            "mag",
            "ap2_mag",
            "kron_mag",
            "ap2_minus_kron_mag",
            "abs_ap2_minus_kron_mag",
            "bright_region_center",
            "bright_region_area",
            "area",
            "major",
            "minor",
            "theta_rad",
            "axis_ratio",
            "refit_matched",
            "status",
        ])
        for idx in bright_idx:
            writer.writerow(
                [
                    int(ids[idx]),
                    f"{float(stage['x'][idx]):.6f}",
                    f"{float(stage['y'][idx]):.6f}",
                    f"{float(stage['mag'][idx]):.6f}",
                    f"{float(stage['ap2_mag'][idx]):.6f}",
                    f"{float(stage['kron_mag'][idx]):.6f}",
                    f"{float(stage['ap2_kron_diff'][idx]):.6f}",
                    f"{float(stage['ap2_kron_absdiff'][idx]):.6f}",
                    bool(stage["bright_region_center"][idx]),
                    f"{float(stage['bright_region_area'][idx]):.1f}",
                    f"{float(stage['area'][idx]):.6f}",
                    f"{float(stage['a'][idx]):.6f}",
                    f"{float(stage['b'][idx]):.6f}",
                    f"{float(stage['theta'][idx]):.8f}",
                    f"{float(stage['axis_ratio'][idx]):.6f}",
                    bool(stage["matched"][idx]),
                    str(stage["status"][idx]),
                ]
            )


def draw_ellipse(ax, x: float, y: float, a: float, b: float, theta: float, color: str, lw: float, alpha: float) -> None:
    if not all(np.isfinite(v) for v in (x, y, a, b, theta)) or a <= 0 or b <= 0:
        ax.plot([x], [y], marker="+", color=color, markersize=3.0, mew=lw, alpha=alpha)
        return
    patch = Ellipse(
        (x, y),
        width=2.0 * a,
        height=2.0 * b,
        angle=math.degrees(theta),
        fill=False,
        edgecolor=color,
        linewidth=lw,
        alpha=alpha,
    )
    ax.add_patch(patch)
    ax.plot([x], [y], marker="+", color=color, markersize=2.5, mew=lw, alpha=alpha)


def plot_png(path: Path, image: np.ndarray, stage: dict[str, np.ndarray], *, tract: str, patch: str, band: str, dpi: int, max_area: float) -> dict[str, int]:
    colors = {
        "removed_previous_stage": "#ff2a2a",
        "remaining_after_A_axis_close": "#24d35f",
        "remaining_after_A_axis_close_ap2": "#24d35f",
        "removed_A_filter": "#ff2a2a",
        "removed_axis_ratio": "#ff9800",
        "removed_close_pair_dimmer": "#b23aff",
        "removed_refit_missing_or_bad": "#00c8ff",
        "removed_ap2_kron_invalid": "#8a8a8a",
        "removed_ap2_kron_absdiff_1_2": "#ffd21f",
        "removed_ap2_kron_absdiff_gt2": "#ff4fb8",
        "remaining_ap2_absdiff_lt1": "#24d35f",
        "remaining_outside_bright_region_ap2_invalid": "#8fd6ff",
        "remaining_small_bright_region_absdiff_lt2": "#00d5ff",
        "remaining_large_bright_region_ap2_skipped": "#2459ff",
        "removed_outside_bright_region_absdiff_ge1": "#ff9800",
        "removed_small_bright_region_absdiff_ge2": "#ff4fb8",
    }
    labels = {
        "removed_previous_stage": "removed before AP2",
        "remaining_after_A_axis_close": "remaining",
        "remaining_after_A_axis_close_ap2": "remaining |dmag|<1",
        "removed_A_filter": "removed by A",
        "removed_axis_ratio": "removed by axis ratio",
        "removed_close_pair_dimmer": "removed by close pair",
        "removed_refit_missing_or_bad": "removed by refit",
        "removed_ap2_kron_invalid": "AP2/Kron invalid",
        "removed_ap2_kron_absdiff_1_2": "1<=|AP2-Kron|<2",
        "removed_ap2_kron_absdiff_gt2": "|AP2-Kron|>=2",
        "remaining_ap2_absdiff_lt1": "keep outside/small |dmag|<1",
        "remaining_outside_bright_region_ap2_invalid": "keep outside bright AP2 invalid",
        "remaining_small_bright_region_absdiff_lt2": "keep small bright |dmag|<2",
        "remaining_large_bright_region_ap2_skipped": "keep large bright AP2 skipped",
        "removed_outside_bright_region_absdiff_ge1": "remove outside bright |dmag|>=1",
        "removed_small_bright_region_absdiff_ge2": "remove small bright |dmag|>=2",
    }
    counts = {key: int(np.count_nonzero(stage["bright"] & (stage["status"] == key))) for key in colors}
    previous_statuses = {
        "removed_refit_missing_or_bad",
        "removed_A_filter",
        "removed_axis_ratio",
        "removed_close_pair_dimmer",
    }
    previous_mask = stage["bright"] & np.isin(stage["status"], list(previous_statuses))
    counts["removed_previous_stage"] = int(np.count_nonzero(previous_mask))
    finite = image[np.isfinite(image)]
    if finite.size:
        vmin, vmax = ZScaleInterval().get_limits(finite)
    else:
        vmin, vmax = 0.0, 1.0
    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
    ax.imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    refined_mode = bool(
        counts["remaining_ap2_absdiff_lt1"]
        or counts["remaining_small_bright_region_absdiff_lt2"]
        or counts["remaining_large_bright_region_ap2_skipped"]
        or counts["remaining_outside_bright_region_ap2_invalid"]
        or counts["removed_outside_bright_region_absdiff_ge1"]
        or counts["removed_small_bright_region_absdiff_ge2"]
    )
    ap2_mode = refined_mode or bool(
        counts["remaining_after_A_axis_close_ap2"]
        or counts["removed_ap2_kron_invalid"]
        or counts["removed_ap2_kron_absdiff_1_2"]
        or counts["removed_ap2_kron_absdiff_gt2"]
    )
    if refined_mode:
        order = [
            "removed_previous_stage",
            "removed_ap2_kron_invalid",
            "removed_outside_bright_region_absdiff_ge1",
            "removed_small_bright_region_absdiff_ge2",
            "remaining_ap2_absdiff_lt1",
            "remaining_outside_bright_region_ap2_invalid",
            "remaining_small_bright_region_absdiff_lt2",
            "remaining_large_bright_region_ap2_skipped",
        ]
    elif ap2_mode:
        order = [
            "removed_previous_stage",
            "removed_ap2_kron_invalid",
            "removed_ap2_kron_absdiff_1_2",
            "removed_ap2_kron_absdiff_gt2",
            "remaining_after_A_axis_close_ap2",
        ]
    else:
        order = [
            "removed_refit_missing_or_bad",
            "removed_A_filter",
            "removed_axis_ratio",
            "removed_close_pair_dimmer",
            "remaining_after_A_axis_close",
        ]
    for status in order:
        color = colors[status]
        if status == "removed_previous_stage":
            idxs = np.flatnonzero(previous_mask)
            ax.plot(
                stage["x"][idxs],
                stage["y"][idxs],
                linestyle="none",
                marker="+",
                color=color,
                markersize=3.2,
                mew=0.8,
                alpha=0.85,
            )
            continue
        idxs = np.flatnonzero(stage["bright"] & (stage["status"] == status))
        for idx in idxs:
            area = float(stage["area"][idx])
            if np.isfinite(area) and area <= max_area:
                draw_ellipse(
                    ax,
                    float(stage["x"][idx]),
                    float(stage["y"][idx]),
                    float(stage["a"][idx]),
                    float(stage["b"][idx]),
                    float(stage["theta"][idx]),
                    color,
                    lw=0.7 if status == "remaining_after_A_axis_close" else 1.0,
                    alpha=0.82,
                )
            else:
                ax.plot(float(stage["x"][idx]), float(stage["y"][idx]), marker="+", color=color, markersize=4.0, mew=1.0)
    legend = []
    for key in order:
        label = f"{labels[key]} n={counts[key]}"
        if key == "removed_previous_stage":
            legend.append(Line2D([], [], linestyle="none", marker="+", color=colors[key], label=label, markersize=6.0))
        else:
            legend.append(Patch(facecolor="none", edgecolor=colors[key], label=label, linewidth=2.0))
    ax.legend(handles=legend, loc="upper right", framealpha=0.86, fontsize=8)
    if refined_mode:
        title_suffix = "after refit/A/axis/close + refined bright AP2-Kron"
    elif ap2_mode:
        title_suffix = "after refit/A/axis/close + catalog AP2-Kron"
    else:
        title_suffix = "after refit/A/axis/close"
    ax.set_title(f"{band} {tract}/{patch}: mag < 22 {title_suffix}")
    ax.set_xlabel("image x")
    ax.set_ylabel("image y")
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return counts


def process_one(args: argparse.Namespace, patch: str, band: str) -> dict[str, object]:
    image_file = calexp_path(args.data_root, args.tract, band, patch)
    catalog_file = meas_path(args.data_root, args.tract, band, patch)
    refit_file = refit_path(args.refit_root, args.tract, band, patch)
    image, ltv = read_image_and_origin(image_file)
    table = Table.read(catalog_file, hdu=1, memmap=True)
    table = attach_kron_refit_radius(
        table,
        refit_file,
        radius_column=args.radius_column,
        good_column=args.good_column,
        output_column="pu_refit_kron_radius",
    )
    bright_mask = None
    if bool(args.refined_bright_ap2):
        bright_mask = build_bright_mask(
            image,
            mode=str(args.bright_mask_mode),
            threshold=float(args.bright_z_threshold),
            dilation=int(args.bright_mask_dilate),
            log_a=float(args.bright_log_a),
            log_high_percentile=float(args.bright_log_high_percentile),
            lupton_stretch=float(args.bright_lupton_stretch),
            lupton_q=float(args.bright_lupton_q),
            anscombe_scale=float(args.bright_anscombe_scale),
        )
    stage = classify_stages(table, args, ltv, bright_region_mask=bright_mask)
    stem = f"{args.tract}_{patch.replace(',', '_')}_{band}"
    out_patch = args.out_dir / args.tract / patch / band
    write_csv(out_patch / f"{stem}_bright_filter_stages.csv", table, stage)
    counts = plot_png(
        out_patch / f"{stem}_bright_filter_stages.png",
        image,
        stage,
        tract=args.tract,
        patch=patch,
        band=band,
        dpi=int(args.dpi),
        max_area=float(args.max_ellipse_area_to_draw),
    )
    counts["bright_total"] = int(np.count_nonzero(stage["bright"]))
    return {"patch": patch, "band": band, **counts}


def main() -> int:
    args = parse_args()
    if bool(args.refined_bright_ap2):
        args.include_ap2_filter = True
    rows = []
    for patch in args.patches:
        for band in args.bands:
            row = process_one(args, patch, band)
            rows.append(row)
            print(json_line(row), flush=True)
    summary_path = args.out_dir / "bright_filter_stage_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "patch",
        "band",
        "bright_total",
        "remaining_after_A_axis_close",
        "remaining_after_A_axis_close_ap2",
        "removed_previous_stage",
        "removed_refit_missing_or_bad",
        "removed_A_filter",
        "removed_axis_ratio",
        "removed_close_pair_dimmer",
        "removed_ap2_kron_invalid",
        "removed_ap2_kron_absdiff_1_2",
        "removed_ap2_kron_absdiff_gt2",
        "remaining_ap2_absdiff_lt1",
        "remaining_outside_bright_region_ap2_invalid",
        "remaining_small_bright_region_absdiff_lt2",
        "remaining_large_bright_region_ap2_skipped",
        "removed_outside_bright_region_absdiff_ge1",
        "removed_small_bright_region_absdiff_ge2",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, 0) for key in keys})
    print(f"wrote {summary_path}", flush=True)
    return 0


def json_line(row: dict[str, object]) -> str:
    return " ".join(f"{key}={value}" for key, value in row.items())


if __name__ == "__main__":
    raise SystemExit(main())
