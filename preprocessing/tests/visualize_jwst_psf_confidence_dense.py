#!/usr/bin/env python3
"""Preview JWST dense regions and PSF-sized Euclidean confidence targets."""

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
from matplotlib.colors import ListedColormap
from scipy import ndimage
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.labels import DenseLabel
from preprocessing.utils.geometry import paint_ellipse


PIPELINE = Path(
    "/home/czh23/analysis/2026-09/2026-09-04/"
    "jwst_cosmos_selected_snr_full_regs/jwst_cosmos_snr_filter_and_regs.py"
)
SCALING = Path(
    "/home/czh23/analysis/2026-09/2026-09-05/"
    "jwst_hsc_scale_bright_visualization/log_lupton_local_global_q_compare/"
    "jwst_log_lupton_local_global_q_compare.py"
)
DEFAULT_RAW_ROOT = Path("/data/shared/jwst_foundation/raw/COSMOS_1727_1837_5893")
DEFAULT_BACKGROUND_ROOT = Path("/data/czh23/JWST/lsst_background_masks")
DEFAULT_OUT_DIR = Path("/home/czh23/analysis/2026-09/2026-09-06/jwst_psf_confidence_dense_preview")

POINTING = "0019"
BANDS = ("f444w", "f115w")
SIZE = 2048
DS = 1
BRIGHT_Q = 20.0
HSC_PIXSCALE_ARCSEC = 0.168
HSC_EMPTY_BRIGHT_COMPONENT_AREA_MIN = 1000.0


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_image_hdu(hdul: fits.HDUList):
    for hdu in hdul:
        if hdu.data is not None and getattr(hdu.data, "ndim", 0) == 2:
            return hdu
    raise ValueError("no 2-D image HDU found")


def raw_fits_path(raw_root: Path, pointing: str, band: str) -> Path:
    point = f"{int(pointing):04d}" if str(pointing).isdigit() else str(pointing)
    return raw_root / f"Pointing_{point}" / f"COSMOS_pointing_{point}_{band.upper()}_detector_p001.fits"


def background_npz_path(root: Path, tract: str, pointing: str, band: str) -> Path:
    point = f"{int(pointing):04d}" if str(pointing).isdigit() else str(pointing)
    return root / "jwst" / tract / point / "group_00" / band.lower() / "background_mask.npz"


def stretch(image: np.ndarray) -> np.ndarray:
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


def downsample_max(values: np.ndarray) -> np.ndarray:
    h = values.shape[0] // DS
    w = values.shape[1] // DS
    return values[: h * DS, : w * DS].reshape(h, DS, w, DS).max(axis=(1, 3))


def regions_for_shape(nx: int, ny: int) -> dict[str, tuple[int, int]]:
    full = 4096
    top_y0 = ny - full
    top_col3_x0 = 2 * full
    return {
        "left_top_lower_left_2048": (0, top_y0),
        "left_top_lower_right_2048": (SIZE, top_y0),
        "top_row_col3_right_mid_2048": (top_col3_x0 + full - SIZE, top_y0 + full // 2 - SIZE // 2),
    }


def select_region_local(cache: dict[str, np.ndarray], x0: int, y0: int) -> dict[str, np.ndarray]:
    x = np.asarray(cache["_x_pix"], dtype=float)
    y = np.asarray(cache["_y_pix"], dtype=float)
    inside = (x >= x0) & (x < x0 + SIZE) & (y >= y0) & (y < y0 + SIZE)
    idx = np.flatnonzero(inside)
    return {"idx": idx, "x": x[idx] - x0, "y": y[idx] - y0}


def build_gaia_context(pipe, cache: dict[str, np.ndarray], wcs, raw_path: Path, pixscale: float, nx: int, ny: int) -> dict[str, object]:
    gaia = pipe.load_gaia()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gaia_ra, gaia_dec, _gaia_pm_applied = pipe.propagated_gaia_radec(gaia, pipe.observation_time(raw_path))
    gmag = np.asarray(gaia["phot_g_mean_mag"], dtype=float)
    gx, gy = wcs.all_world2pix(gaia_ra, gaia_dec, 0)
    sx = np.asarray(cache["_x_pix"], dtype=float)
    sy = np.asarray(cache["_y_pix"], dtype=float)
    sok = np.isfinite(sx) & np.isfinite(sy) & (sx >= 0) & (sx < nx) & (sy >= 0) & (sy < ny)
    source_xy = np.column_stack([sx[sok], sy[sok]]).astype(np.float32)
    tree = cKDTree(source_xy) if len(source_xy) else None
    return {
        "gx": np.asarray(gx, dtype=float),
        "gy": np.asarray(gy, dtype=float),
        "gmag": gmag,
        "tree": tree,
        "match_radius_pix": float(pipe.GAIA_INSERT_RADIUS_ARCSEC) / float(pixscale),
    }


def gaia_insert_centers(pipe, gaia_ctx: dict[str, object], x0: int, y0: int) -> np.ndarray:
    gx = np.asarray(gaia_ctx["gx"], dtype=float)
    gy = np.asarray(gaia_ctx["gy"], dtype=float)
    gmag = np.asarray(gaia_ctx["gmag"], dtype=float)
    gok = (
        np.isfinite(gmag)
        & (gmag <= float(pipe.GAIA_MAG_MAX))
        & np.isfinite(gx)
        & np.isfinite(gy)
        & (gx >= x0)
        & (gx < x0 + SIZE)
        & (gy >= y0)
        & (gy < y0 + SIZE)
    )
    if not np.any(gok):
        return np.zeros((0, 2), dtype=np.float32)

    gaia_idx = np.flatnonzero(gok)
    tree = gaia_ctx.get("tree")
    if tree is None:
        keep = np.flatnonzero(gok)
        return np.column_stack([gx[keep] - x0, gy[keep] - y0]).astype(np.float32)

    matches = tree.query_ball_point(
        np.column_stack([gx[gaia_idx], gy[gaia_idx]]).astype(np.float32),
        r=float(gaia_ctx["match_radius_pix"]),
    )
    insert = gaia_idx[np.asarray([len(item) == 0 for item in matches], dtype=bool)]
    if insert.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.column_stack([gx[insert] - x0, gy[insert] - y0]).astype(np.float32)


def centers_inside_mask(centers: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int]:
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    if centers.size == 0:
        return centers, 0
    xi = np.rint(centers[:, 0]).astype(np.int64)
    yi = np.rint(centers[:, 1]).astype(np.int64)
    inside = (xi >= 0) & (xi < mask.shape[1]) & (yi >= 0) & (yi < mask.shape[0])
    keep = np.zeros(len(centers), dtype=bool)
    keep[inside] = np.asarray(mask, dtype=bool)[yi[inside], xi[inside]]
    return centers[keep], int(np.count_nonzero(~keep))


def paint_psf_confidence(conf: np.ndarray, centers: np.ndarray, fwhm_pix: float) -> None:
    if centers.size == 0:
        return
    h, w = conf.shape
    radius = int(np.ceil(1.25 * float(fwhm_pix))) + 1
    for cx, cy in np.asarray(centers, dtype=np.float32).reshape(-1, 2):
        cx_i = int(round(float(cx)))
        cy_i = int(round(float(cy)))
        if not (0 <= cx_i < w and 0 <= cy_i < h):
            continue
        y0 = max(0, cy_i - radius)
        y1 = min(h, cy_i + radius + 1)
        x0 = max(0, cx_i - radius)
        x1 = min(w, cx_i + radius + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.hypot(xx.astype(np.float32) - float(cx), yy.astype(np.float32) - float(cy))
        vals = np.zeros(dist.shape, dtype=np.uint8)
        vals[(dist > 0.0) & (dist <= 0.5 * fwhm_pix)] = 3
        vals[(dist > 0.5 * fwhm_pix) & (dist <= 0.75 * fwhm_pix)] = 2
        vals[(dist > 0.75 * fwhm_pix) & (dist <= 1.25 * fwhm_pix)] = 1
        if y0 <= cy_i < y1 and x0 <= cx_i < x1:
            vals[cy_i - y0, cx_i - x0] = 4
        patch = conf[y0:y1, x0:x1]
        np.maximum(patch, vals, out=patch)


def dense_rgba(dense: np.ndarray) -> np.ndarray:
    colors = {
        int(DenseLabel.CLEAN): (0.0, 0.82, 0.32, 0.46),
        int(DenseLabel.WEAK_SHAPE): (0.0, 0.85, 1.0, 0.42),
        int(DenseLabel.RESTRICTED_BRIGHT_REGION): (1.0, 0.68, 0.0, 0.34),
        int(DenseLabel.BACKGROUND): (0.10, 0.35, 1.0, 0.14),
        int(DenseLabel.ORDINARY_IGNORE): (1.0, 0.0, 0.0, 0.10),
        int(DenseLabel.STRICT_IGNORE): (0.65, 0.0, 0.85, 0.16),
    }
    rgba = np.zeros((*dense.shape, 4), dtype=np.float32)
    for label, color in colors.items():
        rgba[dense == label] = color
    return rgba


def confidence_rgba(conf: np.ndarray) -> np.ndarray:
    colors = {
        1: (0.12, 0.40, 1.0, 0.40),
        2: (0.00, 0.85, 0.25, 0.48),
        3: (1.00, 0.82, 0.00, 0.58),
        4: (1.00, 0.00, 0.55, 0.90),
    }
    rgba = np.zeros((*conf.shape, 4), dtype=np.float32)
    for level, color in colors.items():
        rgba[conf == level] = color
    return rgba


def paint_sources(
    dense: np.ndarray,
    pipe,
    cache: dict[str, np.ndarray],
    sel: dict[str, np.ndarray],
    state: dict[str, np.ndarray],
    pixscale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = sel["idx"]
    final = np.asarray(state["final"]).astype(str)
    weak = (final != "ignore") & (final != "drop") & np.asarray(state["snr_center"], dtype=bool)
    clean = (final == "clean") & ~weak
    center_only = (final != "ignore") & (final != "drop") & ~(clean | weak)

    for mask, label in ((weak, DenseLabel.WEAK_SHAPE), (clean, DenseLabel.CLEAN)):
        for j in np.flatnonzero(mask):
            src_i = int(idx[j])
            a = float(cache["kron2_a"][src_i]) / float(pixscale)
            b = float(cache["kron2_b"][src_i]) / float(pixscale)
            theta = np.deg2rad(float(cache["theta_world"][src_i]))
            if not np.isfinite(a + b + theta) or a <= 0 or b <= 0:
                continue
            paint_ellipse(dense, float(sel["x"][j]), float(sel["y"][j]), a, b, theta, int(label))
    return clean, weak, center_only


def source_coverage_after_a_filter(
    pipe,
    cache: dict[str, np.ndarray],
    sel: dict[str, np.ndarray],
    state: dict[str, np.ndarray],
    pixscale: float,
) -> np.ndarray:
    coverage = np.zeros((SIZE, SIZE), dtype=np.uint8)
    idx = sel["idx"]
    keep = ~np.asarray(state["a_ignore"], dtype=bool)
    for j in np.flatnonzero(keep):
        src_i = int(idx[j])
        a = float(cache["kron2_a"][src_i]) / float(pixscale)
        b = float(cache["kron2_b"][src_i]) / float(pixscale)
        theta = np.deg2rad(float(cache["theta_world"][src_i]))
        if not np.isfinite(a + b + theta) or a <= 0 or b <= 0:
            continue
        paint_ellipse(coverage, float(sel["x"][j]), float(sel["y"][j]), a, b, theta, 1)
    return coverage.astype(bool)


def bright_component_centers(
    bright: np.ndarray,
    source_coverage: np.ndarray,
    *,
    min_area: float,
    protected_centers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    labels, num = ndimage.label(np.asarray(bright, dtype=bool))
    if num <= 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(bright.shape, dtype=bool), {
            "bright_components": 0,
            "empty_bright_components": 0,
            "empty_bright_components_center_added": 0,
            "empty_bright_components_small_ignore": 0,
            "empty_bright_components_protected_by_external_center": 0,
        }
    protected_components: set[int] = set()
    if protected_centers is not None and np.asarray(protected_centers).size:
        for cx, cy in np.asarray(protected_centers, dtype=np.float32).reshape(-1, 2):
            xi = int(round(float(cx)))
            yi = int(round(float(cy)))
            if 0 <= xi < bright.shape[1] and 0 <= yi < bright.shape[0]:
                comp = int(labels[yi, xi])
                if comp > 0:
                    protected_components.add(comp)
    centers: list[tuple[float, float]] = []
    small_ignore = np.zeros(bright.shape, dtype=bool)
    empty = 0
    added = 0
    ignored = 0
    protected_empty = 0
    objects = ndimage.find_objects(labels)
    for comp_id, slices in enumerate(objects, 1):
        if slices is None:
            continue
        ys, xs = slices
        sub = labels[ys, xs] == comp_id
        if bool(np.any(sub & source_coverage[ys, xs])):
            continue
        if comp_id in protected_components:
            protected_empty += 1
            continue
        empty += 1
        area = int(np.count_nonzero(sub))
        if area < float(min_area):
            small_ignore[ys, xs] |= sub
            ignored += 1
            continue
        yy, xx = np.nonzero(sub)
        centers.append((float(xx.mean() + xs.start), float(yy.mean() + ys.start)))
        added += 1
    return np.asarray(centers, dtype=np.float32).reshape(-1, 2), small_ignore, {
        "bright_components": int(num),
        "empty_bright_components": empty,
        "empty_bright_components_center_added": added,
        "empty_bright_components_small_ignore": ignored,
        "empty_bright_components_protected_by_external_center": protected_empty,
    }


def source_centers(sel: dict[str, np.ndarray], keep: np.ndarray) -> np.ndarray:
    if not np.any(keep):
        return np.zeros((0, 2), dtype=np.float32)
    return np.column_stack([sel["x"][keep], sel["y"][keep]]).astype(np.float32)


def save_confidence_fullres(out_dir: Path, band: str, region: str, display: np.ndarray, conf: np.ndarray) -> tuple[Path, Path]:
    base_rgb = np.repeat(np.asarray(display, dtype=np.float32)[..., None], 3, axis=2)
    overlay = base_rgb.copy()
    colors = {
        1: np.asarray([0.08, 0.32, 1.00], dtype=np.float32),
        2: np.asarray([0.00, 0.90, 0.20], dtype=np.float32),
        3: np.asarray([1.00, 0.82, 0.00], dtype=np.float32),
        4: np.asarray([1.00, 0.00, 0.55], dtype=np.float32),
    }
    alphas = {1: 0.55, 2: 0.62, 3: 0.72, 4: 1.0}
    conf_only = np.zeros((*conf.shape, 3), dtype=np.float32)
    for level, color in colors.items():
        mask = conf == level
        if not np.any(mask):
            continue
        overlay[mask] = (1.0 - alphas[level]) * overlay[mask] + alphas[level] * color
        conf_only[mask] = color
    overlay_path = out_dir / f"pointing{POINTING}_{band}_{region}_confidence_overlay_fullres.png"
    only_path = out_dir / f"pointing{POINTING}_{band}_{region}_confidence_only_fullres.png"
    plt.imsave(overlay_path, np.clip(overlay, 0.0, 1.0), origin="lower")
    plt.imsave(only_path, np.clip(conf_only, 0.0, 1.0), origin="lower")
    return overlay_path, only_path


def plot_band(out_dir: Path, band: str, panels: list[dict[str, object]], fwhm_pix: float) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(16, 16), dpi=140)
    for row, panel in enumerate(panels):
        display = panel["display"]
        dense = downsample_max(panel["dense"])
        conf = downsample_max(panel["confidence"])
        centers = panel["centers"]

        axes[row, 0].imshow(display, origin="lower", cmap="gray", interpolation="nearest")
        axes[row, 0].imshow(dense_rgba(dense), origin="lower", interpolation="nearest")
        axes[row, 0].set_title(f"{panel['region']} dense", fontsize=11)

        axes[row, 1].imshow(display, origin="lower", cmap="gray", interpolation="nearest")
        axes[row, 1].imshow(confidence_rgba(conf), origin="lower", interpolation="nearest")
        axes[row, 1].set_title("PSF confidence", fontsize=11)

        axes[row, 2].imshow(display, origin="lower", cmap="gray", interpolation="nearest")
        for key, color, marker in [
            ("clean", "#00d060", "o"),
            ("weak", "#00e5ff", "o"),
            ("center_only", "#ffd400", "+"),
            ("gaia_insert", "#00ff80", "+"),
            ("bright_component_insert", "#ff8c00", "+"),
            ("ignore", "#ff3030", "x"),
        ]:
            pts = centers[key]
            if len(pts):
                axes[row, 2].scatter(pts[:, 0] / DS, pts[:, 1] / DS, s=8, c=color, marker=marker, linewidths=0.5)
        axes[row, 2].set_title("source centers", fontsize=11)

        for col in range(3):
            axes[row, col].set_xlim(0, SIZE // DS)
            axes[row, col].set_ylim(0, SIZE // DS)
            axes[row, col].set_axis_off()

    dense_handles = [
        plt.Line2D([0], [0], color="#00d060", lw=4, alpha=0.8, label="clean"),
        plt.Line2D([0], [0], color="#00e5ff", lw=4, alpha=0.8, label="weak shape"),
        plt.Line2D([0], [0], color="#ffaa00", lw=4, alpha=0.8, label="bright region"),
        plt.Line2D([0], [0], color="#1a59ff", lw=4, alpha=0.8, label="background"),
        plt.Line2D([0], [0], color="#ff3030", lw=4, alpha=0.8, label="ignore"),
    ]
    conf_handles = [
        plt.Line2D([0], [0], color="#1f66ff", lw=4, label="conf 1"),
        plt.Line2D([0], [0], color="#00d940", lw=4, label="conf 2"),
        plt.Line2D([0], [0], color="#ffd100", lw=4, label="conf 3"),
        plt.Line2D([0], [0], color="#ff0088", lw=4, label="conf 4"),
    ]
    fig.legend(handles=dense_handles + conf_handles, loc="lower center", ncol=9, fontsize=9, framealpha=0.92)
    fig.suptitle(
        f"Pointing {POINTING} {band.upper()}: dense priority clean > weak > bright > background > ignore; "
        f"confidence Euclidean FWHM={fwhm_pix:.2f} pix",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out_png = out_dir / f"pointing{POINTING}_{band}_three_blocks_psf_confidence_dense.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png


def process_band(args: argparse.Namespace, pipe, scale, band: str) -> tuple[Path, list[dict[str, object]]]:
    raw_path = raw_fits_path(args.raw_root, args.pointing, band)
    mask_path = background_npz_path(args.background_root, args.tract, args.pointing, band)
    with np.load(mask_path) as npz:
        full_background = np.asarray(npz["background_mask"], dtype=bool)

    pipe.POINTING = f"{int(args.pointing):04d}" if str(args.pointing).isdigit() else str(args.pointing)
    pipe.RAW_ROOT = pipe.RAW_BASE / f"Pointing_{pipe.POINTING}"
    pipe.BANDS = (band,)
    pipe.FULL_REG_BAND = band
    cache = pipe.load_catalog()

    with fits.open(raw_path, memmap=True) as hdul:
        hdu = first_image_hdu(hdul)
        data = hdu.data
        header = hdu.header.copy()
        _image, wcs, pixscale = pipe.image_hdu(raw_path)
        ny, nx = data.shape
        regions = regions_for_shape(nx, ny)
        source_x, source_y = wcs.all_world2pix(np.asarray(cache["ra"], dtype=float), np.asarray(cache["dec"], dtype=float), 0)
        cache["_x_pix"] = np.asarray(source_x, dtype=float)
        cache["_y_pix"] = np.asarray(source_y, dtype=float)
        gaia_ctx = build_gaia_context(pipe, cache, wcs, raw_path, pixscale, nx, ny)
        factor, _pix_area_ratio, jwst_pix_area = scale.jy_sr_to_zp27_hsc_pixel_factor(header)
        global_params = scale.global_scaling_params(data, factor, BRIGHT_Q)
        fwhm_pix = float(pipe.PSF_FWHM_ARCSEC[band] / pixscale)
        empty_bright_area_min = HSC_EMPTY_BRIGHT_COMPONENT_AREA_MIN * (HSC_PIXSCALE_ARCSEC / pixscale) ** 2

        panels: list[dict[str, object]] = []
        rows: list[dict[str, object]] = []
        for region, (x0, y0) in regions.items():
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

            dense = np.full((SIZE, SIZE), int(DenseLabel.ORDINARY_IGNORE), dtype=np.uint8)
            dense[background] = int(DenseLabel.BACKGROUND)
            coverage = source_coverage_after_a_filter(pipe, cache, sel, state, pixscale)
            gaia_centers_all = gaia_insert_centers(pipe, gaia_ctx, x0, y0)
            gaia_centers, gaia_skipped_no_bright = centers_inside_mask(gaia_centers_all, bright)
            bright_centers, small_empty_bright, bright_component_stats = bright_component_centers(
                bright,
                coverage,
                min_area=empty_bright_area_min,
                protected_centers=gaia_centers,
            )
            dense[bright & ~small_empty_bright] = int(DenseLabel.RESTRICTED_BRIGHT_REGION)
            clean, weak, center_only = paint_sources(dense, pipe, cache, sel, state, pixscale)

            final = np.asarray(state["final"]).astype(str)
            ignore = (final == "ignore")
            conf_keep = clean | weak | center_only
            conf = np.zeros((SIZE, SIZE), dtype=np.uint8)
            paint_psf_confidence(conf, source_centers(sel, conf_keep), fwhm_pix)
            paint_psf_confidence(conf, gaia_centers, fwhm_pix)
            paint_psf_confidence(conf, bright_centers, fwhm_pix)
            display = stretch(flux_filled[::DS, ::DS])
            if args.skip_fullres:
                conf_overlay_path = args.out_dir / f"pointing{POINTING}_{band}_{region}_confidence_overlay_fullres.png"
                conf_only_path = args.out_dir / f"pointing{POINTING}_{band}_{region}_confidence_only_fullres.png"
            else:
                conf_overlay_path, conf_only_path = save_confidence_fullres(args.out_dir, band, region, display, conf)

            panels.append(
                {
                    "region": region,
                    "display": display,
                    "dense": dense,
                    "confidence": conf,
                    "centers": {
                        "clean": source_centers(sel, clean),
                        "weak": source_centers(sel, weak),
                        "center_only": source_centers(sel, center_only),
                        "gaia_insert": gaia_centers,
                        "bright_component_insert": bright_centers,
                        "ignore": source_centers(sel, ignore),
                    },
                }
            )
            vals, counts = np.unique(dense, return_counts=True)
            dense_counts = {f"dense_{int(v)}": int(c) for v, c in zip(vals, counts, strict=False)}
            cvals, ccounts = np.unique(conf, return_counts=True)
            conf_counts = {f"conf_{int(v)}": int(c) for v, c in zip(cvals, ccounts, strict=False)}
            rows.append(
                {
                    "band": band,
                    "region": region,
                    "x0": x0,
                    "y0": y0,
                    "psf_fwhm_pix": fwhm_pix,
                    "sources": int(len(sel["idx"])),
                    "clean_sources": int(np.count_nonzero(clean)),
                    "weak_shape_sources": int(np.count_nonzero(weak)),
                    "center_only_conf_sources": int(np.count_nonzero(center_only)),
                    "gaia_insert_conf_sources": int(len(gaia_centers)),
                    "gaia_insert_candidates": int(len(gaia_centers_all)),
                    "gaia_insert_skipped_no_bright": int(gaia_skipped_no_bright),
                    "bright_component_insert_conf_sources": int(len(bright_centers)),
                    "ignore_sources": int(np.count_nonzero(ignore)),
                    "background_fraction": float(np.mean(background)),
                    "bright_fraction": float(np.mean(bright)),
                    "empty_bright_component_area_min": float(empty_bright_area_min),
                    "hsc_empty_bright_component_area_min": float(HSC_EMPTY_BRIGHT_COMPONENT_AREA_MIN),
                    "jwst_pix_area_arcsec2": float(jwst_pix_area),
                    "confidence_overlay_fullres": str(conf_overlay_path),
                    "confidence_only_fullres": str(conf_only_path),
                    "nan_pixels_interpolated": int(nan_stats["nan_pixels_interpolated"]),
                    **bright_stats,
                    **bright_component_stats,
                    **dense_counts,
                    **conf_counts,
                }
            )
    if args.skip_panel:
        out_png = args.out_dir / f"pointing{POINTING}_{band}_three_blocks_psf_confidence_dense.png"
    else:
        out_png = plot_band(args.out_dir, band, panels, fwhm_pix)
    return out_png, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--background-root", type=Path, default=DEFAULT_BACKGROUND_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tract", default="default")
    parser.add_argument("--pointing", default=POINTING)
    parser.add_argument("--bands", nargs="+", default=list(BANDS))
    parser.add_argument("--skip-panel", action="store_true", help="Do not rewrite the 3x3 overview panel PNGs.")
    parser.add_argument("--skip-fullres", action="store_true", help="Do not rewrite per-region full-resolution confidence PNGs.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-region progress lines.")
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
    all_rows: list[dict[str, object]] = []
    outputs: list[str] = []
    for band in [str(b).lower() for b in args.bands]:
        out_png, rows = process_band(args, pipe, scale, band)
        outputs.append(str(out_png))
        all_rows.extend(rows)
    all_keys: list[str] = []
    for row in all_rows:
        for key in row:
            if key not in all_keys:
                all_keys.append(key)
    out_csv = args.out_dir / f"pointing{args.pointing}_f115w_f444w_psf_confidence_dense_summary.csv"
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(all_rows)
    readme = args.out_dir / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "JWST dense/confidence preview for three 2048x2048 Pointing 0019 zoom cutouts.",
                "The zooms are left-top lower-left, left-top lower-right, and the right-middle part of top-row-col3.",
                "Dense priority: clean > weak_shape > bright_region > LSST background > ordinary_ignore.",
                "Weak_shape is assigned to otherwise usable sources downgraded by the SNR-center rule.",
                "Photometry/containment center-only sources are used as confidence centers only in this preview.",
                "Gaia sources with G<=22 are considered external confidence centers when no JWST catalog center is within 1 arcsec.",
                "Only Gaia external centers whose rounded pixel center lies inside the bright mask are inserted.",
                "Those inserted Gaia centers protect their bright component from empty-small -> ignore conversion.",
                "Empty bright components are handled like HSC: if no post-A-filter source ellipse covers a component,",
                f"  add a geometric-center confidence source only when area >= {HSC_EMPTY_BRIGHT_COMPONENT_AREA_MIN:g}",
                "  HSC pixels scaled by (0.168/JWST_pixscale)^2; smaller empty bright components are ordinary ignore.",
                "Confidence levels use Euclidean distance to source centers:",
                "  level 4: nearest center pixel only",
                "  level 3: 0 < r <= 0.5 FWHM",
                "  level 2: 0.5 FWHM < r <= 0.75 FWHM",
                "  level 1: 0.75 FWHM < r <= 1.25 FWHM",
                f"Bright region uses HSC-pixel surface-brightness scaling, global per-band statistics, Q={BRIGHT_Q:g}.",
                f"Catalog filter pipeline: {PIPELINE}",
                f"Scaling helper: {SCALING}",
                f"Background root: {args.background_root}",
            ]
        )
        + "\n"
    )
    meta = {
        "outputs": outputs,
        "summary_csv": str(out_csv),
        "dense_priority": ["clean", "weak_shape", "bright_region", "background", "ignore"],
        "empty_bright_component_area_min_hsc_pixels": HSC_EMPTY_BRIGHT_COMPONENT_AREA_MIN,
        "empty_bright_component_area_min_scaling": "(0.168 / jwst_pixscale_arcsec)^2",
        "gaia_external_rule": "insert only if G<=22, no JWST center within 1 arcsec, and center lies inside bright mask; protected bright component stays bright",
        "confidence_distance": "Euclidean",
        "confidence_levels": {"4": "center pixel", "3": "(0,0.5] FWHM", "2": "(0.5,0.75] FWHM", "1": "(0.75,1.25] FWHM"},
    }
    (args.out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    for path in outputs:
        print(path)
    print(out_csv)


if __name__ == "__main__":
    main()
