#!/usr/bin/env python3
"""Preview JWST dense targets with capped clean/weak shape ellipses."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cellect")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy.wcs import FITSFixedWarning
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.labels import DenseLabel
from preprocessing.tests.visualize_jwst_psf_confidence_dense import (
    BRIGHT_Q,
    DEFAULT_BACKGROUND_ROOT,
    DEFAULT_RAW_ROOT,
    HSC_EMPTY_BRIGHT_COMPONENT_AREA_MIN,
    HSC_PIXSCALE_ARCSEC,
    PIPELINE,
    SCALING,
    SIZE,
    background_npz_path,
    bright_component_centers,
    build_gaia_context,
    dense_rgba,
    first_image_hdu,
    gaia_insert_centers,
    raw_fits_path,
    regions_for_shape,
    select_region_local,
    source_coverage_after_a_filter,
)
from preprocessing.utils.geometry import paint_ellipse


DEFAULT_OUT_DIR = Path("/home/czh23/analysis/2026-09/2026-09-06/jwst_truncated_dense_targets")
BANDS = ("f115w", "f444w")
POINTING = "0019"
MAX_MAJOR_PIX = 100.0
MAX_MAJOR_ARCSEC = 3.0


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def zscale_display(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.float32)
    sample = finite[:: max(1, finite.size // 500_000)]
    try:
        lo, hi = ZScaleInterval(contrast=0.25, krej=2.5, max_iterations=5).get_limits(sample)
    except Exception:
        lo, hi = np.nanpercentile(sample, [0.5, 99.7])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanpercentile(sample, [0.5, 99.7])
    safe = np.nan_to_num(image, nan=lo, posinf=hi, neginf=lo)
    return np.clip((safe - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def lupton_display(scale, flux_filled: np.ndarray, params: dict[str, float]) -> np.ndarray:
    lup = scale.lupton_map(flux_filled, params["zscore_median"], BRIGHT_Q)
    lup_z = (lup - np.float32(params["lupton_mean"])) / np.float32(params["lupton_std"])
    return ((np.clip(lup_z, -5.0, 5.0) + 5.0) / 10.0).astype(np.float32)


def capped_axes(a_pix: float, b_pix: float, pixscale: float) -> tuple[float, float, bool]:
    cap = min(float(MAX_MAJOR_PIX), float(MAX_MAJOR_ARCSEC) / float(pixscale))
    major = max(float(a_pix), float(b_pix))
    if not np.isfinite(major) or major <= 0:
        return a_pix, b_pix, False
    factor = min(1.0, cap / major)
    return float(a_pix) * factor, float(b_pix) * factor, factor < 1.0


def paint_truncated_sources(
    dense: np.ndarray,
    cache: dict[str, np.ndarray],
    sel: dict[str, np.ndarray],
    state: dict[str, np.ndarray],
    pixscale: float,
) -> dict[str, int]:
    idx = sel["idx"]
    final = np.asarray(state["final"]).astype(str)
    weak = (final != "ignore") & (final != "drop") & np.asarray(state["snr_center"], dtype=bool)
    clean = (final == "clean") & ~weak
    clipped_clean = 0
    clipped_weak = 0
    for mask, label, counter in (
        (weak, DenseLabel.WEAK_SHAPE, "weak"),
        (clean, DenseLabel.CLEAN, "clean"),
    ):
        for j in np.flatnonzero(mask):
            src_i = int(idx[j])
            a = float(cache["kron2_a"][src_i]) / float(pixscale)
            b = float(cache["kron2_b"][src_i]) / float(pixscale)
            theta = np.deg2rad(float(cache["theta_world"][src_i]))
            if not np.isfinite(a + b + theta) or a <= 0 or b <= 0:
                continue
            a_use, b_use, clipped = capped_axes(a, b, pixscale)
            if clipped and counter == "clean":
                clipped_clean += 1
            if clipped and counter == "weak":
                clipped_weak += 1
            paint_ellipse(dense, float(sel["x"][j]), float(sel["y"][j]), a_use, b_use, theta, int(label))
    return {
        "clean_sources": int(np.count_nonzero(clean)),
        "weak_shape_sources": int(np.count_nonzero(weak)),
        "clipped_clean_sources": clipped_clean,
        "clipped_weak_shape_sources": clipped_weak,
    }


def process_band(args: argparse.Namespace, pipe, scale, band: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw_path = raw_fits_path(args.raw_root, args.pointing, band)
    mask_path = background_npz_path(args.background_root, args.tract, args.pointing, band)
    with np.load(mask_path) as npz:
        full_background = np.asarray(npz["background_mask"], dtype=bool)

    pipe.POINTING = f"{int(args.pointing):04d}" if str(args.pointing).isdigit() else str(args.pointing)
    pipe.RAW_ROOT = pipe.RAW_BASE / f"Pointing_{pipe.POINTING}"
    pipe.BANDS = (band,)
    pipe.FULL_REG_BAND = band
    cache = pipe.load_catalog()

    panels: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    with fits.open(raw_path, memmap=True) as hdul:
        hdu = first_image_hdu(hdul)
        data = hdu.data
        header = hdu.header.copy()
        _image, wcs, pixscale = pipe.image_hdu(raw_path)
        ny, nx = data.shape
        source_x, source_y = wcs.all_world2pix(np.asarray(cache["ra"], dtype=float), np.asarray(cache["dec"], dtype=float), 0)
        cache["_x_pix"] = np.asarray(source_x, dtype=float)
        cache["_y_pix"] = np.asarray(source_y, dtype=float)
        gaia_ctx = build_gaia_context(pipe, cache, wcs, raw_path, pixscale, nx, ny)
        factor, _pix_area_ratio, jwst_pix_area = scale.jy_sr_to_zp27_hsc_pixel_factor(header)
        global_params = scale.global_scaling_params(data, factor, BRIGHT_Q)
        empty_bright_area_min = HSC_EMPTY_BRIGHT_COMPONENT_AREA_MIN * (HSC_PIXSCALE_ARCSEC / pixscale) ** 2

        for region, (x0, y0) in regions_for_shape(nx, ny).items():
            if not args.quiet:
                print(f"[{band}] {region}", flush=True)
            raw_cut = np.asarray(data[y0 : y0 + SIZE, x0 : x0 + SIZE], dtype=np.float32)
            flux_cut = raw_cut * np.float32(factor)
            flux_filled, nan_stats = scale.fill_nans_hybrid(flux_cut)
            bright, bright_stats = scale.bright_mask_from_params(flux_filled, global_params, BRIGHT_Q)
            background = full_background[y0 : y0 + SIZE, x0 : x0 + SIZE]

            sel = select_region_local(cache, x0, y0)
            nan_ignore = pipe.nan_component_center_mask(raw_cut, sel["x"], sel["y"])
            state = pipe.classify(
                cache,
                sel["idx"],
                band,
                use_psf=True,
                x_pix=sel["x"],
                y_pix=sel["y"],
                pixscale=pixscale,
                nan_component_ignore=nan_ignore,
            )
            coverage = source_coverage_after_a_filter(pipe, cache, sel, state, pixscale)
            gaia_centers_all = gaia_insert_centers(pipe, gaia_ctx, x0, y0)
            bright_centers, small_empty_bright, bright_component_stats = bright_component_centers(
                bright,
                coverage,
                min_area=empty_bright_area_min,
                protected_centers=gaia_centers_all,
            )
            dense = np.full((SIZE, SIZE), int(DenseLabel.ORDINARY_IGNORE), dtype=np.uint8)
            dense[background] = int(DenseLabel.BACKGROUND)
            dense[bright & ~small_empty_bright] = int(DenseLabel.RESTRICTED_BRIGHT_REGION)
            source_stats = paint_truncated_sources(dense, cache, sel, state, pixscale)

            zdisp = zscale_display(flux_filled)
            ldisp = lupton_display(scale, flux_filled, global_params)
            panels.append({"band": band, "region": region, "zscale": zdisp, "lupton": ldisp, "dense": dense})
            vals, counts = np.unique(dense, return_counts=True)
            dense_counts = {f"dense_{int(v)}": int(c) for v, c in zip(vals, counts, strict=False)}
            rows.append(
                {
                    "band": band,
                    "region": region,
                    "x0": x0,
                    "y0": y0,
                    "pixscale_arcsec": float(pixscale),
                    "max_major_pix": float(min(MAX_MAJOR_PIX, MAX_MAJOR_ARCSEC / pixscale)),
                    "max_major_arcsec": float(MAX_MAJOR_ARCSEC),
                    "jwst_pix_area_arcsec2": float(jwst_pix_area),
                    "sources": int(len(sel["idx"])),
                    "gaia_insert_candidates": int(len(gaia_centers_all)),
                    "bright_component_insert_conf_sources": int(len(bright_centers)),
                    "nan_pixels_interpolated": int(nan_stats["nan_pixels_interpolated"]),
                    **source_stats,
                    **bright_stats,
                    **bright_component_stats,
                    **dense_counts,
                }
            )
    return panels, rows


def plot_grid(out_dir: Path, panels: list[dict[str, object]], key: str, suffix: str) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=130)
    by_key = {(p["band"], p["region"]): p for p in panels}
    regions = ["left_top_lower_left_2048", "left_top_lower_right_2048", "top_row_col3_right_mid_2048"]
    for row, band in enumerate(BANDS):
        for col, region in enumerate(regions):
            ax = axes[row, col]
            panel = by_key[(band, region)]
            ax.imshow(panel[key], origin="lower", cmap="gray", interpolation="nearest", vmin=0, vmax=1)
            ax.imshow(dense_rgba(panel["dense"]), origin="lower", interpolation="nearest")
            ax.set_title(f"{band.upper()} {region}", fontsize=11)
            ax.set_xlim(0, SIZE)
            ax.set_ylim(0, SIZE)
            ax.set_axis_off()
    handles = [
        plt.Line2D([0], [0], color="#00d060", lw=5, label="clean capped shape"),
        plt.Line2D([0], [0], color="#00e5ff", lw=5, label="weak capped shape"),
        plt.Line2D([0], [0], color="#ffaa00", lw=5, label="bright region"),
        plt.Line2D([0], [0], color="#1a59ff", lw=5, label="background"),
        plt.Line2D([0], [0], color="#ff3030", lw=5, label="ignore"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10, framealpha=0.92)
    fig.suptitle(
        f"JWST dense targets with clean/weak semi-major capped at min({MAX_MAJOR_PIX:g} pix, {MAX_MAJOR_ARCSEC:g} arcsec)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    out = out_dir / f"pointing{POINTING}_f115w_f444w_truncated_dense_{suffix}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--background-root", type=Path, default=DEFAULT_BACKGROUND_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tract", default="default")
    parser.add_argument("--pointing", default=POINTING)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=FITSFixedWarning)
    warnings.filterwarnings("ignore", message="ERFA function")
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    args = parse_args()
    args.raw_root = args.raw_root.expanduser().resolve()
    args.background_root = args.background_root.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pipe = import_module(PIPELINE, "jwst_cosmos_snr_filter_and_regs")
    scale = import_module(SCALING, "jwst_log_lupton_local_global_q_compare")
    all_panels: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    for band in BANDS:
        panels, rows = process_band(args, pipe, scale, band)
        all_panels.extend(panels)
        all_rows.extend(rows)
    zscale_png = plot_grid(args.out_dir, all_panels, "zscale", "zscale_overlay")
    lupton_png = plot_grid(args.out_dir, all_panels, "lupton", "lupton_overlay")

    keys: list[str] = []
    for row in all_rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    csv_path = args.out_dir / f"pointing{POINTING}_f115w_f444w_truncated_dense_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_rows)
    meta = {
        "zscale_png": str(zscale_png),
        "lupton_png": str(lupton_png),
        "summary_csv": str(csv_path),
        "shape_cap": {
            "max_major_pix": MAX_MAJOR_PIX,
            "max_major_arcsec": MAX_MAJOR_ARCSEC,
            "rule": "scale both semi-axes by min(1, min(max_major_pix, max_major_arcsec/pixscale)/max(a,b))",
        },
    }
    (args.out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(zscale_png)
    print(lupton_png)
    print(csv_path)


if __name__ == "__main__":
    main()
