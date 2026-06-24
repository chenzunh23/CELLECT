#!/usr/bin/env python3
"""Visualize SAM-CELLECT instance masks on the Zangetsu demo cutout."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.visualization import ZScaleInterval
import astropy.units as u
from torch.utils.data import DataLoader


CELLECT_ROOT = Path("/home/czh23/CELLECT")
if str(CELLECT_ROOT) not in sys.path:
    sys.path.insert(0, str(CELLECT_ROOT))

from astro_train_data import AstroCutoutDataset, collate_cutouts, discover_cutout_records  # noqa: E402
from astro_train_ops import detect_centers, unwrap_model  # noqa: E402
from sam_backbone import build_sam_cellect2d  # noqa: E402


TRACT = "9813"
PATCH = "6,1"
TILE = "zangetsu_lower_right_x27366_y6453"
DEFAULT_BANDS = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y")
DEFAULT_BAND = "HSC-I"
DEFAULT_DATA_ROOT = (
    CELLECT_ROOT
    / "output/sam_cellect_combination_260611/preprocessing_diagnostics_260611/zangetsu_preprocessed_cutouts_260611"
)
DEFAULT_NATIVE_SAM_DIR = CELLECT_ROOT / "zangetsu_demo/output/native_sam_astro_vit_b_64_coadd_HSC-I"
MATCH_RADIUS_PIX = 0.5 / 0.168
REG_HEADER = [
    "# Region file format: DS9 version 4.1",
    'global color=green dashlist=8 3 width=2 font="helvetica 10 normal roman" '
    "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
    "image",
]


def _read_config(path: Path) -> dict:
    payload = json.loads(path.read_text())
    args = dict(payload.get("args", {}))
    args["_top"] = payload
    return args


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key[7:] if str(key).startswith("module.") else key: value for key, value in state.items()}


def _checkpoint_epoch(path: Path) -> object:
    ckpt = torch.load(path, map_location="cpu")
    return ckpt.get("epoch", "") if isinstance(ckpt, dict) else ""


def _checkpoint_variant(checkpoint: Path) -> str:
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if isinstance(ckpt, dict):
        if ckpt.get("model_variant"):
            return str(ckpt["model_variant"])
        if isinstance(ckpt.get("args"), dict) and ckpt["args"].get("model_variant"):
            return str(ckpt["args"]["model_variant"])
    if isinstance(state, dict):
        first_key = next(iter(state), "")
        if str(first_key).startswith(("encoder.", "module.encoder.")):
            return "sam_per_band"
    return ""


def _make_model(cfg: dict, checkpoint: Path, device: torch.device, bands: Sequence[str]) -> torch.nn.Module:
    top = cfg.get("_top", {})
    variant = _checkpoint_variant(checkpoint) or str(top.get("model_variant") or cfg.get("model_variant", "sam_per_band"))
    if variant != "sam_per_band":
        raise ValueError(f"{checkpoint} is model_variant={variant!r}; this visualizer expects sam_per_band checkpoints")

    base_channels = int(top.get("base_channels") or cfg.get("base_channels", 32))
    model = build_sam_cellect2d(
        str(top.get("sam_model_type") or cfg.get("sam_model_type", "vit_b")),
        checkpoint=None,
        num_bands=len(bands),
        image_size=512,
        patch_size=16,
        seg_classes=int(top.get("seg_classes") or cfg.get("seg_classes", 2)),
        confidence_levels=5,
        embedding_dim=int(cfg.get("embedding_dim", 64)),
        shape_channels=3,
        decoder_channels=(base_channels * 8, base_channels * 4, base_channels * 2, base_channels),
        use_cen=bool(top.get("sam_cen_enabled", not bool(cfg.get("disable_sam_cen", False)))),
        cen_input_image=True,
        cen_width=max(2, base_channels // 4),
        candidate_count=int(cfg.get("matcher_candidate_count", 5)),
        shape_feature_dim=6,
        enable_matchers=False,
        astro_preprocess_in_model=False,
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    incompatible = model.load_state_dict(_strip_module_prefix(state), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"[load] {checkpoint.name}: missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)}",
            flush=True,
        )
    model.eval()
    return model


def _dataset(root: Path, dataset_name: str, bands: Sequence[str], cfg: dict) -> DataLoader:
    tract_root = root / dataset_name / TRACT
    records = discover_cutout_records(tract_root, bands=bands)
    records = [rec for rec in records if rec.patch == PATCH and rec.tile_name == TILE]
    if len(records) != 1:
        raise RuntimeError(f"Expected one record for {dataset_name}/{PATCH}/{TILE}, got {len(records)} under {tract_root}")
    ds = AstroCutoutDataset(
        records,
        fits_hdu=int(cfg.get("fits_hdu", 1)),
        confidence_levels=5,
        ellipse_sigma=float(cfg.get("ellipse_sigma", 2.0)),
        core_radius=int(cfg.get("core_radius", 2)),
        shape_source=str(cfg.get("shape_source", "kron")),
        source_filter=str(cfg.get("source_filter", "nchild0")),
        load_eval_ignore_sources=True,
        augment=False,
    )
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_cutouts)


def _band_outputs(outputs: dict[str, torch.Tensor], band_idx: int) -> dict[str, torch.Tensor]:
    selected: dict[str, torch.Tensor] = {}
    for key, value in outputs.items():
        if not torch.is_tensor(value):
            continue
        if value.ndim >= 5:
            selected[key] = value[:, band_idx]
        else:
            selected[key] = value
    return selected


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _ellipse_line(
    x: float,
    y: float,
    major: float,
    minor: float,
    theta: float,
    *,
    color: str,
    width: int = 2,
    text: str = "",
) -> str:
    major = max(abs(_safe_float(major, 1.0)), 1.0)
    minor = max(abs(_safe_float(minor, major)), 1.0)
    theta = _safe_float(theta, 0.0)
    suffix = f" # color={color} width={width}"
    if text:
        suffix += f" text={{{text}}}"
    return f"ellipse({x + 1:.3f},{y + 1:.3f},{major:.3f},{minor:.3f},{math.degrees(theta):.3f}){suffix}"


def _circle_line(x: float, y: float, radius: float, *, color: str, width: int = 2, text: str = "") -> str:
    suffix = f" # color={color} width={width}"
    if text:
        suffix += f" text={{{text}}}"
    return f"circle({x + 1:.3f},{y + 1:.3f},{radius:.3f}){suffix}"


def _table(path: Path) -> Table:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Table.read(path)


def _tile_xy0(tile_name: str) -> tuple[int, int]:
    import re

    match = re.search(r"_x(-?\d+)_y(-?\d+)", tile_name)
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def _xy_from_table(table: Table, tile_name: str) -> tuple[np.ndarray, np.ndarray]:
    x0, y0 = _tile_xy0(tile_name)
    names = set(table.colnames)
    for x_name, y_name in (
        ("base_SdssShape_x", "base_SdssShape_y"),
        ("base_SdssCentroid_x", "base_SdssCentroid_y"),
        ("base_NaiveCentroid_x", "base_NaiveCentroid_y"),
        ("deblend_psfCenter_x", "deblend_psfCenter_y"),
        ("centroid_local_x", "centroid_local_y"),
    ):
        if x_name in names and y_name in names:
            x = np.asarray(table[x_name], dtype=np.float32)
            y = np.asarray(table[y_name], dtype=np.float32)
            if x_name == "centroid_local_x":
                return x, y
            return x - float(x0), y - float(y0)
    raise KeyError(f"No supported centroid columns in {table.colnames}")


def _local_rows(table: Table, tile_name: str, size: int = 512) -> tuple[Table, np.ndarray, np.ndarray]:
    x, y = _xy_from_table(table, tile_name)
    keep = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < size) & (y >= 0) & (y < size)
    return table[keep], x[keep], y[keep]


def _ellipse_from_row(row, x: float, y: float, color: str, width: int = 2) -> str:
    names = set(row.colnames) if hasattr(row, "colnames") else set()
    major = _safe_float(row["ellipse_major_sigma"], 4.0) if "ellipse_major_sigma" in names else 4.0
    minor = _safe_float(row["ellipse_minor_sigma"], 4.0) if "ellipse_minor_sigma" in names else 4.0
    theta = _safe_float(row["ellipse_theta"], 0.0) if "ellipse_theta" in names else 0.0
    return _ellipse_line(x, y, major, minor, theta, color=color, width=width)


def _greedy_match(pred_xy: np.ndarray, gt_xy: np.ndarray, radius: float) -> tuple[dict[int, int], set[int]]:
    if pred_xy.size == 0 or gt_xy.size == 0:
        return {}, set()
    pairs: list[tuple[float, int, int]] = []
    r2 = float(radius) ** 2
    for pi, pred in enumerate(pred_xy):
        d2 = np.sum((gt_xy - pred[None, :]) ** 2, axis=1)
        for gi in np.flatnonzero(d2 <= r2):
            pairs.append((float(d2[gi]), int(pi), int(gi)))
    pairs.sort(key=lambda item: item[0])
    pred_to_gt: dict[int, int] = {}
    used_gt: set[int] = set()
    for _dist, pi, gi in pairs:
        if pi in pred_to_gt or gi in used_gt:
            continue
        pred_to_gt[pi] = gi
        used_gt.add(gi)
    return pred_to_gt, used_gt


def _load_native_sam_centers(native_dir: Path) -> np.ndarray:
    """Load native SAM mask geometric centers in zero-based image coordinates."""

    native_dir = native_dir.expanduser().resolve()
    label_maps = sorted(native_dir.glob("*_sam_labelmap.fits"))
    if label_maps:
        label_map = np.asarray(fits.getdata(label_maps[-1]), dtype=np.int32)
        centers: list[tuple[float, float]] = []
        for label in sorted(int(value) for value in np.unique(label_map) if int(value) > 0):
            mask = label_map == label
            y, x = np.nonzero(mask)
            if y.size:
                centers.append((float(x.mean()), float(y.mean())))
        return np.asarray(centers, dtype=np.float32).reshape(-1, 2)

    center_regs = sorted(native_dir.glob("*_mask_centers.reg"))
    if not center_regs:
        raise FileNotFoundError(f"No native SAM *_sam_labelmap.fits or *_mask_centers.reg found in {native_dir}")

    import re

    centers = []
    pattern = re.compile(r"circle\(\s*([+-]?[0-9.]+)\s*,\s*([+-]?[0-9.]+)\s*,")
    for line in center_regs[-1].read_text().splitlines():
        match = pattern.search(line)
        if match:
            centers.append((float(match.group(1)) - 1.0, float(match.group(2)) - 1.0))
    return np.asarray(centers, dtype=np.float32).reshape(-1, 2)


def _load_clean_rows(root: Path, dataset_name: str, band: str, tile_name: str) -> tuple[Table, np.ndarray, np.ndarray]:
    path = root / dataset_name / TRACT / PATCH / "band_reference_catalogs" / band / f"meas-{band}-{TRACT}-{PATCH}.fits"
    rows, x, y = _local_rows(_table(path), tile_name)
    return rows, x, y


def _shape_rows(pred_xy: np.ndarray, shape: np.ndarray, scale: float) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    h, w = shape.shape[-2:]
    for pred_index, (x, y) in enumerate(pred_xy):
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            continue
        rows.append(
            {
                "pred_index": float(pred_index),
                "x": float(x),
                "y": float(y),
                "major": float(shape[0, yi, xi]) * float(scale),
                "minor": float(shape[1, yi, xi]) * float(scale),
                "theta": float(shape[2, yi, xi]) if shape.shape[0] > 2 else 0.0,
            }
        )
    return rows


def _boxes_from_shape_rows(rows: Sequence[dict[str, float]], image_size: int, scale: float) -> torch.Tensor:
    boxes: list[list[float]] = []
    for row in rows:
        a = max(abs(_safe_float(row["major"], 1.0)) * float(scale), 1.0)
        b = max(abs(_safe_float(row["minor"], a)) * float(scale), 1.0)
        theta = _safe_float(row.get("theta", 0.0), 0.0)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        dx = math.sqrt((a * cos_t) ** 2 + (b * sin_t) ** 2) + 1.0
        dy = math.sqrt((a * sin_t) ** 2 + (b * cos_t) ** 2) + 1.0
        x = _safe_float(row["x"], 0.0)
        y = _safe_float(row["y"], 0.0)
        boxes.append(
            [
                max(0.0, x - dx),
                max(0.0, y - dy),
                min(float(image_size - 1), x + dx),
                min(float(image_size - 1), y + dy),
            ]
        )
    return torch.tensor(boxes, dtype=torch.float32)


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return float("nan"), float("nan")
    return float(xs.mean()), float(ys.mean())


def _contours(mask: np.ndarray, max_vertices: int) -> Iterable[list[tuple[float, float]]]:
    try:
        from skimage import measure

        for contour in measure.find_contours(mask.astype(np.float32), 0.5):
            if contour.shape[0] < 3:
                continue
            step = max(1, int(math.ceil(contour.shape[0] / max_vertices)))
            pts = contour[::step]
            if pts.shape[0] >= 3:
                yield [(float(x + 1.0), float(y + 1.0)) for y, x in pts]
        return
    except Exception:
        pass

    ys, xs = np.where(mask)
    if ys.size:
        yield [
            (float(xs.min() + 1), float(ys.min() + 1)),
            (float(xs.max() + 2), float(ys.min() + 1)),
            (float(xs.max() + 2), float(ys.max() + 2)),
            (float(xs.min() + 1), float(ys.max() + 2)),
        ]


def _polygon_line(points: Sequence[tuple[float, float]], color: str, width: int, text: str = "") -> str:
    coords = ",".join(f"{x:.2f},{y:.2f}" for x, y in points)
    suffix = f" # color={color} width={width}"
    if text:
        suffix += f" text={{{text}}}"
    return f"polygon({coords}){suffix}"


def _write_text(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _instance_rgb(label: int) -> np.ndarray:
    palette = np.asarray(
        [
            (0.00, 0.76, 0.94),
            (0.95, 0.18, 0.65),
            (1.00, 0.82, 0.12),
            (0.16, 0.72, 0.33),
            (0.18, 0.38, 1.00),
            (0.95, 0.22, 0.14),
            (0.90, 0.56, 0.12),
            (0.58, 0.32, 0.90),
        ],
        dtype=np.float32,
    )
    return palette[(int(label) - 1) % len(palette)]


def _zscale_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not bool(finite.any()):
        return np.zeros_like(image, dtype=np.float32)

    finite_values = image[finite]
    try:
        interval = ZScaleInterval()
        lo, hi = interval.get_limits(finite_values)
    except Exception:
        lo, hi = np.nanpercentile(finite_values, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite_values)), float(np.nanmax(finite_values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def _calculate_stability_score(logits: torch.Tensor, *, mask_threshold: float = 0.0, threshold_offset: float = 1.0) -> torch.Tensor:
    intersections = (logits > (float(mask_threshold) + float(threshold_offset))).sum(dim=(-1, -2))
    unions = (logits > (float(mask_threshold) - float(threshold_offset))).sum(dim=(-1, -2))
    return intersections.to(dtype=logits.dtype) / unions.to(dtype=logits.dtype).clamp_min(1.0)


def _write_overlay(path: Path, image: np.ndarray, masks: Sequence[np.ndarray], alpha: float) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cellect")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = _zscale_image(np.asarray(image, dtype=np.float32))
    rgb = np.repeat(base[..., None], 3, axis=2)
    indexed = [(idx + 1, mask, int(mask.sum())) for idx, mask in enumerate(masks)]
    for label, mask, _area in sorted(indexed, key=lambda item: item[2], reverse=True):
        if not bool(mask.any()):
            continue
        color = _instance_rgb(label)
        rgb[mask] = (1.0 - float(alpha)) * rgb[mask] + float(alpha) * color

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 7.2), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    # Flip the rgb image upside down
    rgb = np.flip(rgb, axis=0)
    ax.imshow(rgb, origin="upper", interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_overlay_with_shapes(path: Path, image: np.ndarray, masks: Sequence[np.ndarray], shape_rows: Sequence[dict[str, float]], alpha: float) -> None:
    _write_overlay(path, image, masks, alpha)


def _raw_band_image_from_batch(batch: dict, band_idx: int, cfg: dict) -> tuple[np.ndarray | None, str]:
    image_paths = batch.get("image_paths") if isinstance(batch, dict) else None
    if not image_paths:
        return None, "batch_image"
    try:
        first = image_paths[0]
        sample_paths = image_paths if isinstance(first, (str, Path)) else first
        path = Path(sample_paths[int(band_idx)])
    except Exception:
        return None, "batch_image"
    if not path.exists() or path.name == "__zscale_cache_only__.fits":
        return None, "batch_image"
    hdu = int(cfg.get("fits_hdu", 1))
    try:
        raw = np.asarray(fits.getdata(path, hdu), dtype=np.float32)
    except Exception:
        try:
            raw = np.asarray(fits.getdata(path, "IMAGE"), dtype=np.float32)
        except Exception:
            return None, "batch_image"
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return raw, str(path)


PHOTOMETRY_COLUMNS = [
    "checkpoint_label",
    "dataset",
    "tile",
    "band",
    "pred_index",
    "x",
    "y",
    "major",
    "minor",
    "theta",
    "gt_index",
    "gt_x",
    "gt_y",
    "match_distance_pix",
    "gt_max_shape_iou",
    "tp_isolated",
    "gt_ap2_flux",
    "gt_ap2mag",
    "gt_kron_flux",
    "gt_kron_mag",
    "background_mode",
    "background2d_applied",
    "annulus_success",
    "annulus_median",
    "ap_radius",
    "ap_area",
    "ap_flux_raw",
    "ap_flux",
    "ap_flux_njy",
    "ap_abmag",
    "kron_area",
    "kron_flux_raw",
    "kron_flux",
    "kron_flux_njy",
    "kron_abmag",
    "mask_kept",
    "mask_area",
    "mask_flux_raw",
    "mask_flux",
    "mask_flux_njy",
    "mask_abmag",
    "mask_pred_iou",
]


def _nan() -> float:
    return float("nan")


def _mag_from_flux(flux: float, *, zero_point: float) -> float:
    try:
        flux_f = float(flux)
    except Exception:
        return _nan()
    if not math.isfinite(flux_f) or flux_f <= 0.0:
        return _nan()
    return float(float(zero_point) - 2.5 * math.log10(flux_f))


def _row_float(row, name: str, default: float = float("nan")) -> float:
    names = set(row.colnames) if hasattr(row, "colnames") else set()
    if name not in names:
        return default
    return _safe_float(row[name], default)


def _row_flux_to_mag(row, flux_candidates: Sequence[str], *, zero_point: float) -> tuple[float, float]:
    names = set(row.colnames) if hasattr(row, "colnames") else set()
    for name in flux_candidates:
        if name not in names:
            continue
        flux = _safe_float(row[name], _nan())
        mag = _mag_from_flux(flux, zero_point=zero_point)
        if math.isfinite(mag):
            return float(flux), mag
    return _nan(), _nan()


def _gt_ap2_flux_mag(row, *, zero_point: float) -> tuple[float, float]:
    # In the Zangetsu GT tables ap2 corresponds to the LSST 6 px circular aperture.
    return _row_flux_to_mag(
        row,
        (
            "base_CircularApertureFlux_6_0_instFlux",
            "ext_convolved_ConvolvedFlux_0_6_0_instFlux",
            "ext_convolved_ConvolvedFlux_1_6_0_instFlux",
            "ext_convolved_ConvolvedFlux_2_6_0_instFlux",
            "ext_convolved_ConvolvedFlux_3_6_0_instFlux",
        ),
        zero_point=zero_point,
    )


def _gt_kron_flux_mag(row, *, zero_point: float) -> tuple[float, float]:
    return _row_flux_to_mag(
        row,
        (
            "ext_photometryKron_KronFlux_instFlux",
            "ext_convolved_ConvolvedFlux_0_kron_instFlux",
            "ext_convolved_ConvolvedFlux_1_kron_instFlux",
            "ext_convolved_ConvolvedFlux_2_kron_instFlux",
            "ext_convolved_ConvolvedFlux_3_kron_instFlux",
        ),
        zero_point=zero_point,
    )


def _ellipse_bool_mask(
    x: float,
    y: float,
    major: float,
    minor: float,
    theta: float,
    image_shape: tuple[int, int],
) -> np.ndarray:
    h, w = int(image_shape[0]), int(image_shape[1])
    yy, xx = np.mgrid[0:h, 0:w]
    a = max(abs(float(major)), 1.0)
    b = max(abs(float(minor)), 1.0)
    cos_t = math.cos(float(theta))
    sin_t = math.sin(float(theta))
    dx = xx.astype(np.float32) - float(x)
    dy = yy.astype(np.float32) - float(y)
    xr = cos_t * dx + sin_t * dy
    yr = -sin_t * dx + cos_t * dy
    return (xr / a) ** 2 + (yr / b) ** 2 <= 1.0


def _gt_shape_max_ious(
    clean_rows: Table,
    clean_x: np.ndarray,
    clean_y: np.ndarray,
    *,
    image_shape: tuple[int, int],
) -> np.ndarray:
    n = len(clean_rows)
    max_ious = np.zeros((n,), dtype=np.float32)
    if n <= 1:
        return max_ious
    masks: list[np.ndarray] = []
    bboxes: list[tuple[int, int, int, int]] = []
    h, w = int(image_shape[0]), int(image_shape[1])
    for idx in range(n):
        row = clean_rows[idx]
        major = _row_float(row, "ellipse_major_sigma", 4.0)
        minor = _row_float(row, "ellipse_minor_sigma", 4.0)
        theta = _row_float(row, "ellipse_theta", 0.0)
        mask = _ellipse_bool_mask(float(clean_x[idx]), float(clean_y[idx]), major, minor, theta, image_shape)
        masks.append(mask)
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            bboxes.append((0, 0, -1, -1))
        else:
            bboxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    for i in range(n):
        x0_i, y0_i, x1_i, y1_i = bboxes[i]
        if x1_i < x0_i or y1_i < y0_i:
            continue
        area_i = int(masks[i].sum())
        if area_i == 0:
            continue
        for j in range(i + 1, n):
            x0_j, y0_j, x1_j, y1_j = bboxes[j]
            if x1_j < x0_j or y1_j < y0_j:
                continue
            x0 = max(x0_i, x0_j, 0)
            y0 = max(y0_i, y0_j, 0)
            x1 = min(x1_i, x1_j, w - 1)
            y1 = min(y1_i, y1_j, h - 1)
            if x1 < x0 or y1 < y0:
                continue
            inter = int(np.logical_and(masks[i][y0 : y1 + 1, x0 : x1 + 1], masks[j][y0 : y1 + 1, x0 : x1 + 1]).sum())
            if inter == 0:
                continue
            area_j = int(masks[j].sum())
            union = area_i + area_j - inter
            if union <= 0:
                continue
            iou = float(inter) / float(union)
            if iou > max_ious[i]:
                max_ious[i] = iou
            if iou > max_ious[j]:
                max_ious[j] = iou
    return max_ious


def _flux_to_njy_mag(flux: float, *, zero_point: float, psf_factor: float) -> tuple[float, float]:
    if not math.isfinite(float(flux)):
        return _nan(), _nan()
    corrected = float(flux) / max(float(psf_factor), 1e-12)
    flux_njy = corrected * (float(zero_point) * u.ABmag).to(u.nJy).value
    if not math.isfinite(flux_njy) or flux_njy <= 0.0:
        return float(flux_njy), _nan()
    return float(flux_njy), float((flux_njy * u.nJy).to(u.ABmag).value)


def _ellipse_mask_for_rows(shape_rows: Sequence[dict[str, float]], image_shape: tuple[int, int]) -> np.ndarray:
    h, w = int(image_shape[0]), int(image_shape[1])
    out = np.zeros((h, w), dtype=bool)
    if not shape_rows:
        return out
    yy, xx = np.mgrid[0:h, 0:w]
    for row in shape_rows:
        x = _safe_float(row.get("x", 0.0), 0.0)
        y = _safe_float(row.get("y", 0.0), 0.0)
        a = max(abs(_safe_float(row.get("major", 1.0), 1.0)), 1.0)
        b = max(abs(_safe_float(row.get("minor", a), a)), 1.0)
        theta = _safe_float(row.get("theta", 0.0), 0.0)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        dx = xx.astype(np.float32) - float(x)
        dy = yy.astype(np.float32) - float(y)
        xr = cos_t * dx + sin_t * dy
        yr = -sin_t * dx + cos_t * dy
        out |= (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
    return out


def _prepare_photometry_image(
    image: np.ndarray,
    *,
    dataset_name: str,
    bg_mode: str,
    source_mask: np.ndarray | None,
    box_size: int,
    filter_size: int,
    sigma_clip: float,
) -> tuple[np.ndarray, str, bool]:
    mode = str(bg_mode).lower().strip()
    if mode == "auto":
        mode = "photutils_annulus" if str(dataset_name).lower() == "denoised" else "none"
    image = np.asarray(image, dtype=np.float32)
    work = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=True)
    applied = False
    if mode in {"photutils", "photutils_annulus"}:
        try:
            from photutils.background import Background2D, MedianBackground

            bkg = Background2D(
                work,
                box_size=(int(box_size), int(box_size)),
                filter_size=(int(filter_size), int(filter_size)),
                bkg_estimator=MedianBackground(),
                mask=source_mask,
            )
            work = (work - np.asarray(bkg.background, dtype=np.float32)).astype(np.float32, copy=False)
            applied = True
        except Exception:
            valid = np.isfinite(work)
            if source_mask is not None:
                valid &= ~source_mask.astype(bool, copy=False)
            values = work[valid]
            values = values[np.isfinite(values)]
            median_value = 0.0
            if values.size:
                _mean, median_value, _std = sigma_clipped_stats(values, sigma=float(sigma_clip))
            work = (work - float(median_value)).astype(np.float32, copy=False)
            applied = True
    return work, mode, applied


def _annulus_medians(
    image: np.ndarray,
    positions: np.ndarray,
    *,
    r_in: float,
    r_out: float,
    source_mask: np.ndarray | None,
    sigma_clip: float,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    medians = np.zeros((positions.shape[0],), dtype=np.float64)
    success = np.zeros((positions.shape[0],), dtype=bool)
    if positions.shape[0] == 0:
        return medians, success
    try:
        from photutils.aperture import CircularAnnulus

        annuli = CircularAnnulus(positions, r_in=float(r_in), r_out=float(r_out))
        for idx, mask in enumerate(annuli.to_mask(method=str(method))):
            data = mask.multiply(image)
            if data is None:
                continue
            select = mask.data > 0
            if source_mask is not None:
                source_cutout = mask.cutout(source_mask.astype(bool, copy=False))
                if source_cutout is not None and source_cutout.shape == select.shape:
                    select &= ~source_cutout.astype(bool, copy=False)
            values = data[select]
            values = values[np.isfinite(values)]
            if values.size < 20:
                continue
            _mean, median, _std = sigma_clipped_stats(values, sigma=float(sigma_clip))
            medians[idx] = float(median)
            success[idx] = True
        return medians, success
    except ModuleNotFoundError:
        pass

    h, w = image.shape
    yy, xx = np.mgrid[0:h, 0:w]
    for idx, (x, y) in enumerate(positions):
        rr = (xx.astype(np.float32) - float(x)) ** 2 + (yy.astype(np.float32) - float(y)) ** 2
        select = (rr >= float(r_in) ** 2) & (rr <= float(r_out) ** 2)
        if source_mask is not None:
            select &= ~source_mask.astype(bool, copy=False)
        values = np.asarray(image, dtype=np.float32)[select]
        values = values[np.isfinite(values)]
        if values.size < 20:
            continue
        _mean, median, _std = sigma_clipped_stats(values, sigma=float(sigma_clip))
        medians[idx] = float(median)
        success[idx] = True
    return medians, success


def _aperture_sums(
    image: np.ndarray,
    shape_rows: Sequence[dict[str, float]],
    *,
    ap_radius: float,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(shape_rows)
    positions = np.asarray([[row["x"], row["y"]] for row in shape_rows], dtype=np.float64).reshape(n, 2)
    ap_flux = np.full((n,), np.nan, dtype=np.float64)
    kron_flux = np.full((n,), np.nan, dtype=np.float64)
    ap_area = np.full((n,), math.pi * float(ap_radius) ** 2, dtype=np.float64)
    kron_area = np.full((n,), np.nan, dtype=np.float64)
    if n == 0:
        return ap_flux, kron_flux, ap_area, kron_area

    try:
        from photutils.aperture import CircularAperture, EllipticalAperture, aperture_photometry

        circ = CircularAperture(positions, r=float(ap_radius))
        ap_flux[:] = np.asarray(aperture_photometry(image, circ, method=str(method))["aperture_sum"], dtype=np.float64)
        for idx, row in enumerate(shape_rows):
            a = max(abs(_safe_float(row.get("major", 1.0), 1.0)), 1.0)
            b = max(abs(_safe_float(row.get("minor", a), a)), 1.0)
            theta = _safe_float(row.get("theta", 0.0), 0.0)
            kron_area[idx] = math.pi * a * b
            aper = EllipticalAperture((float(row["x"]), float(row["y"])), a=a, b=b, theta=theta)
            kron_flux[idx] = float(np.asarray(aperture_photometry(image, aper, method=str(method))["aperture_sum"], dtype=np.float64)[0])
        return ap_flux, kron_flux, ap_area, kron_area
    except ModuleNotFoundError:
        pass

    h, w = image.shape
    yy, xx = np.mgrid[0:h, 0:w]
    image_f = np.asarray(image, dtype=np.float32)
    r2 = float(ap_radius) ** 2
    for idx, row in enumerate(shape_rows):
        x = float(row["x"])
        y = float(row["y"])
        circ = (xx.astype(np.float32) - x) ** 2 + (yy.astype(np.float32) - y) ** 2 <= r2
        ap_flux[idx] = float(image_f[circ].sum())
        a = max(abs(_safe_float(row.get("major", 1.0), 1.0)), 1.0)
        b = max(abs(_safe_float(row.get("minor", a), a)), 1.0)
        theta = _safe_float(row.get("theta", 0.0), 0.0)
        kron_area[idx] = math.pi * a * b
        dx = xx.astype(np.float32) - x
        dy = yy.astype(np.float32) - y
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        xr = cos_t * dx + sin_t * dy
        yr = -sin_t * dx + cos_t * dy
        ellipse = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
        kron_flux[idx] = float(image_f[ellipse].sum())
    return ap_flux, kron_flux, ap_area, kron_area


def _write_photometry_csv(
    path: Path,
    *,
    checkpoint_label: str,
    dataset_name: str,
    band: str,
    shape_rows: Sequence[dict[str, float]],
    image: np.ndarray,
    masks_by_pred_index: dict[int, np.ndarray],
    mask_iou_by_pred_index: dict[int, float],
    args: argparse.Namespace,
    pred_to_clean: dict[int, int] | None = None,
    pred_xy: np.ndarray | None = None,
    clean_rows: Table | None = None,
    clean_x: np.ndarray | None = None,
    clean_y: np.ndarray | None = None,
    gt_shape_max_ious: np.ndarray | None = None,
    isolated_path: Path | None = None,
    isolated_max_shape_iou: float = 0.05,
) -> tuple[str, int, int | None]:
    source_mask = _ellipse_mask_for_rows(shape_rows, image.shape)
    for mask in masks_by_pred_index.values():
        if mask.shape == source_mask.shape:
            source_mask |= mask.astype(bool, copy=False)
    phot_image, resolved_mode, background2d_applied = _prepare_photometry_image(
        image,
        dataset_name=dataset_name,
        bg_mode=str(args.photometry_bg_mode),
        source_mask=source_mask if bool(source_mask.any()) else None,
        box_size=int(args.photometry_bkg_box_size),
        filter_size=int(args.photometry_bkg_filter_size),
        sigma_clip=float(args.photometry_sigma_clip),
    )
    use_annulus = resolved_mode in {"annulus", "photutils_annulus"}
    n = len(shape_rows)
    positions = np.asarray([[row["x"], row["y"]] for row in shape_rows], dtype=np.float64).reshape(n, 2)
    ann_medians = np.zeros((n,), dtype=np.float64)
    ann_success = np.zeros((n,), dtype=bool)
    if use_annulus and n:
        ann_medians, ann_success = _annulus_medians(
            phot_image,
            positions,
            r_in=float(args.photometry_ann_r_in),
            r_out=float(args.photometry_ann_r_out),
            source_mask=source_mask if bool(source_mask.any()) else None,
            sigma_clip=float(args.photometry_sigma_clip),
            method=str(args.photometry_annulus_method),
        )

    ap_raw, kron_raw, ap_area, kron_area = _aperture_sums(
        phot_image,
        shape_rows,
        ap_radius=float(args.photometry_ap_radius),
        method=str(args.photometry_method),
    )

    rows: list[dict[str, object]] = []
    pred_to_clean = pred_to_clean or {}
    for idx, row in enumerate(shape_rows):
        pred_index = int(row["pred_index"])
        gt_index = pred_to_clean.get(pred_index)
        gt_x = gt_y = match_distance = gt_shape_iou = gt_ap2_flux = gt_ap2mag = gt_kron_flux = gt_kron_mag = _nan()
        tp_isolated = False
        if gt_index is not None and clean_rows is not None and clean_x is not None and clean_y is not None:
            gt_i = int(gt_index)
            gt_x = float(clean_x[gt_i])
            gt_y = float(clean_y[gt_i])
            if pred_xy is not None and pred_index < len(pred_xy):
                pred = np.asarray(pred_xy[pred_index], dtype=np.float64)
                match_distance = float(np.linalg.norm(pred - np.asarray([gt_x, gt_y], dtype=np.float64)))
            if gt_shape_max_ious is not None and gt_i < len(gt_shape_max_ious):
                gt_shape_iou = float(gt_shape_max_ious[gt_i])
                tp_isolated = math.isfinite(gt_shape_iou) and gt_shape_iou <= float(isolated_max_shape_iou)
            gt_ap2_flux, gt_ap2mag = _gt_ap2_flux_mag(clean_rows[gt_i], zero_point=float(args.gt_photometry_zero_point))
            gt_kron_flux, gt_kron_mag = _gt_kron_flux_mag(clean_rows[gt_i], zero_point=float(args.gt_photometry_zero_point))
        ann = float(ann_medians[idx]) if use_annulus and bool(ann_success[idx]) else 0.0
        ap_flux = float(ap_raw[idx] - ann * ap_area[idx]) if math.isfinite(float(ap_raw[idx])) else _nan()
        kron_flux = float(kron_raw[idx] - ann * kron_area[idx]) if math.isfinite(float(kron_raw[idx])) else _nan()
        ap_njy, ap_mag = _flux_to_njy_mag(
            ap_flux,
            zero_point=float(args.photometry_zero_point),
            psf_factor=float(args.photometry_psf_factor),
        )
        kron_njy, kron_mag = _flux_to_njy_mag(
            kron_flux,
            zero_point=float(args.photometry_zero_point),
            psf_factor=float(args.photometry_psf_factor),
        )

        mask = masks_by_pred_index.get(pred_index)
        mask_area = int(mask.sum()) if mask is not None else 0
        mask_raw = float(np.asarray(phot_image)[mask].sum()) if mask is not None and mask_area > 0 else _nan()
        mask_flux = float(mask_raw - ann * mask_area) if math.isfinite(mask_raw) else _nan()
        mask_njy, mask_mag = _flux_to_njy_mag(
            mask_flux,
            zero_point=float(args.photometry_zero_point),
            psf_factor=float(args.photometry_psf_factor),
        )
        rows.append(
            {
                "checkpoint_label": checkpoint_label,
                "dataset": dataset_name,
                "tile": TILE,
                "band": band,
                "pred_index": pred_index,
                "x": float(row["x"]),
                "y": float(row["y"]),
                "major": float(row["major"]),
                "minor": float(row["minor"]),
                "theta": float(row["theta"]),
                "gt_index": int(gt_index) if gt_index is not None else "",
                "gt_x": gt_x,
                "gt_y": gt_y,
                "match_distance_pix": match_distance,
                "gt_max_shape_iou": gt_shape_iou,
                "tp_isolated": bool(tp_isolated),
                "gt_ap2_flux": gt_ap2_flux,
                "gt_ap2mag": gt_ap2mag,
                "gt_kron_flux": gt_kron_flux,
                "gt_kron_mag": gt_kron_mag,
                "background_mode": resolved_mode,
                "background2d_applied": bool(background2d_applied),
                "annulus_success": bool(ann_success[idx]) if use_annulus else False,
                "annulus_median": ann,
                "ap_radius": float(args.photometry_ap_radius),
                "ap_area": float(ap_area[idx]),
                "ap_flux_raw": float(ap_raw[idx]) if math.isfinite(float(ap_raw[idx])) else _nan(),
                "ap_flux": ap_flux,
                "ap_flux_njy": ap_njy,
                "ap_abmag": ap_mag,
                "kron_area": float(kron_area[idx]) if math.isfinite(float(kron_area[idx])) else _nan(),
                "kron_flux_raw": float(kron_raw[idx]) if math.isfinite(float(kron_raw[idx])) else _nan(),
                "kron_flux": kron_flux,
                "kron_flux_njy": kron_njy,
                "kron_abmag": kron_mag,
                "mask_kept": mask is not None,
                "mask_area": mask_area,
                "mask_flux_raw": mask_raw,
                "mask_flux": mask_flux,
                "mask_flux_njy": mask_njy,
                "mask_abmag": mask_mag,
                "mask_pred_iou": mask_iou_by_pred_index.get(pred_index, _nan()),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PHOTOMETRY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    isolated_count: int | None = None
    if isolated_path is not None:
        isolated_rows = [row for row in rows if bool(row.get("tp_isolated", False))]
        isolated_path.parent.mkdir(parents=True, exist_ok=True)
        with isolated_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PHOTOMETRY_COLUMNS)
            writer.writeheader()
            writer.writerows(isolated_rows)
        isolated_count = len(isolated_rows)
    return resolved_mode, len(rows), isolated_count


def _write_native_diff_reg(
    *,
    path: Path,
    checkpoint_label: str,
    dataset_name: str,
    band: str,
    shape_rows: Sequence[dict[str, float]],
    native_xy: np.ndarray,
    match_radius: float,
    center_radius: float,
) -> tuple[int, int]:
    finetuned_xy = np.asarray([[row["x"], row["y"]] for row in shape_rows], dtype=np.float32).reshape(-1, 2)
    ft_to_native, native_used = _greedy_match(finetuned_xy, native_xy, match_radius)
    extra_ft = [idx for idx in range(len(shape_rows)) if idx not in ft_to_native]
    missed_native = [idx for idx in range(len(native_xy)) if idx not in native_used]

    lines = REG_HEADER + [
        f"# {checkpoint_label} {dataset_name} {PATCH}/{TILE} {band}: fine-tuned SAM vs native SAM detections",
        f"# magenta ellipses: fine-tuned detections not matched to native SAM within {match_radius:.3f} px",
        f"# red circles: native SAM detections not matched by fine-tuned SAM within {match_radius:.3f} px",
    ]
    for idx in extra_ft:
        row = shape_rows[idx]
        lines.append(
            _ellipse_line(
                row["x"],
                row["y"],
                row["major"],
                row["minor"],
                row["theta"],
                color="magenta",
                width=2,
                text=f"extra_ft={idx + 1}",
            )
        )
    for idx in missed_native:
        x, y = native_xy[idx]
        lines.append(
            _circle_line(
                float(x),
                float(y),
                float(center_radius),
                color="red",
                width=2,
                text=f"missed_native={idx + 1}",
            )
        )
    _write_text(path, lines)
    return len(extra_ft), len(missed_native)


@torch.no_grad()
def _run_one(
    *,
    model: torch.nn.Module,
    cfg: dict,
    dataset_root: Path,
    dataset_name: str,
    checkpoint_label: str,
    out_dir: Path,
    bands: Sequence[str],
    band: str,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, object]:
    loader = _dataset(dataset_root, dataset_name, bands, cfg)
    band_idx = list(bands).index(band)
    threshold = float(args.threshold if args.threshold is not None else cfg.get("confidence_threshold", 2.0))
    nms_radius = int(args.nms_radius if args.nms_radius is not None else cfg.get("nms_radius", 1))
    confidence_score = str(args.confidence_score or cfg.get("confidence_score", "cellect"))
    center_refinement = str(args.center_refinement or cfg.get("center_refinement", "softargmax"))
    center_refinement_radius = int(
        args.center_refinement_radius
        if args.center_refinement_radius is not None
        else cfg.get("center_refinement_radius", 1)
    )

    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32)
        outputs = model(image)
        outputs_i = _band_outputs(outputs, band_idx)
        pred_list = detect_centers(
            outputs_i,
            threshold=threshold,
            nms_radius=nms_radius,
            confidence_score=confidence_score,
            center_refinement=center_refinement,
            center_refinement_radius=center_refinement_radius,
        )
        pred_xy = np.asarray(pred_list[0], dtype=np.float32).reshape(-1, 2)
        shape = outputs_i["shape"][0].detach().float().cpu().numpy().astype(np.float32)
        image_band = image[0, band_idx].detach().cpu().numpy().astype(np.float32)
        raw_band, photometry_image_source = _raw_band_image_from_batch(batch, band_idx, cfg)
        photometry_image_band = raw_band if raw_band is not None else image_band
        image_embeddings = outputs["image_embeddings"]
        break

    shape_rows = _shape_rows(pred_xy, shape, float(args.shape_display_scale))
    masks: list[np.ndarray] = []
    mask_ious: list[float] = []
    raw_mask_count = 0
    mask_area_after_min = 0
    mask_area_after_max = 0
    mask_after_pred_iou = 0
    mask_after_stability = 0
    mask_area_median: float | None = None
    mask_iou_median: float | None = None
    mask_stability_median: float | None = None
    if shape_rows:
        points = torch.tensor([[row["x"], row["y"]] for row in shape_rows], device=device, dtype=torch.float32)
        boxes = None
        if not bool(args.mask_prompt_center_only):
            boxes = _boxes_from_shape_rows(shape_rows, int(args.image_size), float(args.prompt_box_scale)).to(device=device)
        batch_indices = torch.full((points.shape[0],), band_idx, device=device, dtype=torch.long)
        low_res_masks, iou_pred = unwrap_model(model).forward_sam_masks(
            image_embeddings,
            batch_indices,
            points,
            boxes,
            multimask_output=bool(args.multimask),
            chunk_size=int(args.mask_chunk_size),
        )
        if bool(args.multimask) and low_res_masks.shape[1] > 1:
            best = torch.argmax(iou_pred, dim=1)
            low_res_masks = low_res_masks[torch.arange(low_res_masks.shape[0], device=device), best][:, None]
            iou_pred = iou_pred[torch.arange(iou_pred.shape[0], device=device), best][:, None]
        stability = _calculate_stability_score(
            low_res_masks[:, 0].float(),
            mask_threshold=float(args.mask_threshold),
            threshold_offset=float(args.stability_score_offset),
        )
        full_masks = F.interpolate(
            low_res_masks.float(),
            size=(int(args.image_size), int(args.image_size)),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        bool_masks = (full_masks > float(args.mask_threshold)).detach().cpu().numpy()
        ious = iou_pred[:, 0].detach().float().cpu().numpy()
        stabilities = stability.detach().float().cpu().numpy()
        max_mask_area = float(args.image_size) * float(args.image_size) * float(args.max_mask_area_ratio)
        raw_mask_count = int(bool_masks.shape[0])
        areas = bool_masks.reshape(raw_mask_count, -1).sum(axis=1) if raw_mask_count else np.zeros((0,), dtype=np.int64)
        if raw_mask_count:
            mask_area_median = float(np.median(areas))
            mask_iou_median = float(np.median(ious))
            mask_stability_median = float(np.median(stabilities))
        keep = areas >= int(args.min_mask_area)
        mask_area_after_min = int(keep.sum())
        keep = keep & (areas <= max_mask_area)
        mask_area_after_max = int(keep.sum())
        if args.pred_iou_thresh is not None:
            keep = keep & (ious >= float(args.pred_iou_thresh))
        mask_after_pred_iou = int(keep.sum())
        if args.stability_score_thresh is not None:
            keep = keep & (stabilities >= float(args.stability_score_thresh))
        mask_after_stability = int(keep.sum())
        mask_source_indices: list[int] = []
        for source_idx in np.flatnonzero(keep):
            masks.append(bool_masks[int(source_idx)].astype(bool))
            mask_ious.append(float(ious[int(source_idx)]))
            mask_source_indices.append(int(source_idx))
    else:
        mask_source_indices = []

    masks_by_pred_index = {int(idx): mask for idx, mask in zip(mask_source_indices, masks)}
    mask_iou_by_pred_index = {int(idx): float(iou) for idx, iou in zip(mask_source_indices, mask_ious)}

    clean_rows, clean_x, clean_y = _load_clean_rows(dataset_root, dataset_name, band, TILE)
    clean_xy = np.column_stack([clean_x, clean_y]).astype(np.float32) if len(clean_x) else np.zeros((0, 2), np.float32)
    pred_to_clean, clean_used = _greedy_match(pred_xy, clean_xy, float(args.match_radius))

    tile_out = out_dir / checkpoint_label / dataset_name
    prefix = f"{checkpoint_label}_{dataset_name}_{PATCH.replace(',', '_')}_{TILE}_{band.replace('-', '_')}"
    mask_reg = tile_out / f"{prefix}_instance_masks.reg"
    center_reg = tile_out / f"{prefix}_centers.reg"
    shape_reg = tile_out / f"{prefix}_shape_kron.reg"
    fn_reg = tile_out / f"{prefix}_fn_shape.reg"
    native_diff_reg = tile_out / f"{prefix}_native_sam_diff.reg"
    overlay_png = tile_out / f"{prefix}_mask_overlay.png"
    photometry_csv = tile_out / f"{prefix}_photometry.csv"
    tp_isolated_csv = tile_out / f"{prefix}_tp_isolated_gt_photometry.csv"

    mask_lines = REG_HEADER + [f"# {checkpoint_label} {dataset_name} {PATCH}/{TILE} {band}: SAM instance mask contours"]
    for idx, (mask, iou) in enumerate(zip(masks, mask_ious), start=1):
        for contour in _contours(mask, int(args.max_contour_vertices)):
            mask_lines.append(_polygon_line(contour, "green", 1, text=f"id={idx} iou={iou:.3f} area={int(mask.sum())}"))

    center_lines = REG_HEADER + [f"# {checkpoint_label} {dataset_name} {PATCH}/{TILE} {band}: detected centers"]
    for idx, row in enumerate(shape_rows, start=1):
        center_lines.append(_circle_line(row["x"], row["y"], float(args.center_radius), color="cyan", width=2, text=f"id={idx}"))

    shape_lines = REG_HEADER + [f"# {checkpoint_label} {dataset_name} {PATCH}/{TILE} {band}: predicted Kron ellipses"]
    for idx, row in enumerate(shape_rows, start=1):
        color = "cyan" if int(row["pred_index"]) in pred_to_clean else "magenta"
        shape_lines.append(
            _ellipse_line(row["x"], row["y"], row["major"], row["minor"], row["theta"], color=color, width=2, text=f"id={idx}")
        )

    fn_lines = REG_HEADER + [f"# {checkpoint_label} {dataset_name} {PATCH}/{TILE} {band}: clean-source false negatives"]
    for gi in range(len(clean_rows)):
        if gi in clean_used:
            continue
        fn_lines.append(_ellipse_from_row(clean_rows[gi], float(clean_x[gi]), float(clean_y[gi]), "red", width=2))
        fn_lines.append(_circle_line(float(clean_x[gi]), float(clean_y[gi]), float(args.center_radius), color="red", width=1))

    _write_text(mask_reg, mask_lines)
    _write_text(center_reg, center_lines)
    _write_text(shape_reg, shape_lines)
    _write_text(fn_reg, fn_lines)
    _write_overlay(overlay_png, image_band, masks, float(args.overlay_alpha))

    gt_shape_max_ious = _gt_shape_max_ious(clean_rows, clean_x, clean_y, image_shape=image_band.shape)
    photometry_mode: str | None = None
    photometry_count: int | None = None
    tp_isolated_count: int | None = None
    if not bool(args.disable_photometry):
        photometry_mode, photometry_count, tp_isolated_count = _write_photometry_csv(
            photometry_csv,
            checkpoint_label=checkpoint_label,
            dataset_name=dataset_name,
            band=band,
            shape_rows=shape_rows,
            image=photometry_image_band,
            masks_by_pred_index=masks_by_pred_index,
            mask_iou_by_pred_index=mask_iou_by_pred_index,
            args=args,
            pred_to_clean=pred_to_clean,
            pred_xy=pred_xy,
            clean_rows=clean_rows,
            clean_x=clean_x,
            clean_y=clean_y,
            gt_shape_max_ious=gt_shape_max_ious,
            isolated_path=tp_isolated_csv,
            isolated_max_shape_iou=float(args.tp_isolated_max_shape_iou),
        )

    native_extra_count: int | None = None
    native_missed_count: int | None = None
    native_reg_path: str | None = None
    if args.native_sam_dir is not None and dataset_name == str(args.native_sam_dataset):
        native_xy = _load_native_sam_centers(Path(args.native_sam_dir))
        native_extra_count, native_missed_count = _write_native_diff_reg(
            path=native_diff_reg,
            checkpoint_label=checkpoint_label,
            dataset_name=dataset_name,
            band=band,
            shape_rows=shape_rows,
            native_xy=native_xy,
            match_radius=float(args.native_match_radius),
            center_radius=float(args.center_radius),
        )
        native_reg_path = str(native_diff_reg)

    return {
        "checkpoint_label": checkpoint_label,
        "dataset": dataset_name,
        "tile": TILE,
        "band": band,
        "detections": int(len(pred_xy)),
        "mask_prompts": int(len(shape_rows)),
        "raw_masks": int(raw_mask_count),
        "mask_after_min_area": int(mask_area_after_min),
        "mask_after_max_area": int(mask_area_after_max),
        "mask_after_pred_iou": int(mask_after_pred_iou),
        "mask_after_stability": int(mask_after_stability),
        "kept_masks": int(len(masks)),
        "pred_iou_thresh": None if args.pred_iou_thresh is None else float(args.pred_iou_thresh),
        "stability_score_thresh": None if args.stability_score_thresh is None else float(args.stability_score_thresh),
        "mask_area_median": mask_area_median,
        "mask_iou_median": mask_iou_median,
        "mask_stability_median": mask_stability_median,
        "clean_gt": int(len(clean_rows)),
        "clean_tp": int(len(clean_used)),
        "clean_fn": int(len(clean_rows) - len(clean_used)),
        "mask_reg": str(mask_reg),
        "center_reg": str(center_reg),
        "shape_reg": str(shape_reg),
        "fn_shape_reg": str(fn_reg),
        "native_sam_diff_reg": native_reg_path,
        "native_sam_extra_finetuned": native_extra_count,
        "native_sam_missed": native_missed_count,
        "overlay_png": str(overlay_png),
        "photometry_csv": None if bool(args.disable_photometry) else str(photometry_csv),
        "photometry_mode": photometry_mode,
        "photometry_count": photometry_count,
        "photometry_image_source": photometry_image_source,
        "tp_isolated_gt_photometry_csv": None if bool(args.disable_photometry) else str(tp_isolated_csv),
        "tp_isolated_gt_photometry_count": tp_isolated_count,
        "tp_isolated_max_shape_iou": float(args.tp_isolated_max_shape_iou),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, default=CELLECT_ROOT / "output/ckpts/SAM_per_band_debug_0612")
    parser.add_argument("--config", type=Path, default=None, help="Defaults to <ckpt-dir>/run_config.json")
    parser.add_argument("--checkpoint", "-c", type=Path, action="append", default=None, help="Can be passed multiple times")
    parser.add_argument("--checkpoint-label", "-l", action="append", default=None, help="One label per --checkpoint")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=CELLECT_ROOT / "zangetsu_demo/output/sam_cellect_visualization")
    parser.add_argument("--datasets", nargs="+", default=["coadd", "denoised"])
    parser.add_argument(
        "--native-sam-dir",
        type=Path,
        default=DEFAULT_NATIVE_SAM_DIR,
        help="Native SAM output directory used for fine-tuned-vs-native diff REG. Use 'none' to disable.",
    )
    parser.add_argument(
        "--native-sam-dataset",
        default="coadd",
        help="Dataset name to compare with --native-sam-dir. Default compares only coadd.",
    )
    parser.add_argument(
        "--native-match-radius",
        type=float,
        default=MATCH_RADIUS_PIX,
        help="Center match radius in pixels for fine-tuned-vs-native SAM comparison.",
    )
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    parser.add_argument("--band", default=DEFAULT_BAND)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--nms-radius", type=int, default=None)
    parser.add_argument("--confidence-score", choices=("cellect", "raw", "ordinal_prob"), default=None)
    parser.add_argument("--center-refinement", choices=("integer", "softargmax"), default=None)
    parser.add_argument("--center-refinement-radius", type=int, default=None)
    parser.add_argument("--match-radius", type=float, default=MATCH_RADIUS_PIX)
    parser.add_argument("--center-radius", type=float, default=7.0)
    parser.add_argument("--shape-display-scale", type=float, default=1.0)
    parser.add_argument("--prompt-box-scale", type=float, default=2.0)
    parser.add_argument(
        "--mask-prompt-center-only",
        action="store_true",
        default=None,
        help="Use center-only prompts for SAM mask decoding. Defaults to the training config.",
    )
    parser.add_argument(
        "--mask-prompt-center-bbox",
        dest="mask_prompt_center_only",
        action="store_false",
        default=None,
        help="Force center+bbox prompts for SAM mask decoding.",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--min-mask-area", type=int, default=15)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.5)
    parser.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=None,
        help=(
            "Filter masks below this SAM predicted-IoU. Defaults to the training config only "
            "when mask_pred_iou loss was enabled; otherwise no pred-IoU filter is applied."
        ),
    )
    parser.add_argument(
        "--stability-score-thresh",
        type=float,
        default=None,
        help=(
            "Filter masks below this SAM stability score. Defaults to the training config only "
            "when mask_stability loss was enabled; otherwise no stability filter is applied."
        ),
    )
    parser.add_argument("--stability-score-offset", type=float, default=1.0)
    parser.add_argument("--mask-chunk-size", type=int, default=128)
    parser.add_argument(
        "--multimask",
        action="store_true",
        default=None,
        help="Use SAM multimask output and keep the highest-IoU mask per prompt. Defaults to the training config.",
    )
    parser.add_argument(
        "--singlemask",
        dest="multimask",
        action="store_false",
        help="Force SAM single-mask output even if the training config used multimask.",
    )
    parser.add_argument("--disable-photometry", action="store_true", help="Do not write per-source SAM photometry CSV files.")
    parser.add_argument(
        "--photometry-bg-mode",
        choices=("auto", "none", "annulus", "photutils", "photutils_annulus"),
        default="auto",
        help="Background mode for photometry. auto uses photutils_annulus for denoised and none for coadd/noisy.",
    )
    parser.add_argument("--photometry-ap-radius", type=float, default=6.0, help="Fixed circular aperture radius in pixels; default ap2 (2 arcsec).")
    parser.add_argument("--photometry-ann-r-in", type=float, default=10.0)
    parser.add_argument("--photometry-ann-r-out", type=float, default=15.0)
    parser.add_argument("--photometry-bkg-box-size", type=int, default=64)
    parser.add_argument("--photometry-bkg-filter-size", type=int, default=3)
    parser.add_argument("--photometry-sigma-clip", type=float, default=3.0)
    parser.add_argument("--photometry-method", choices=("center", "exact", "subpixel"), default="exact")
    parser.add_argument("--photometry-annulus-method", choices=("center", "exact", "subpixel"), default="center")
    parser.add_argument("--photometry-zero-point", type=float, default=31.4, help="AB zero point for measured coadd/noisy/denoised image fluxes.")
    parser.add_argument("--gt-photometry-zero-point", type=float, default=27.0, help="AB zero point for GT catalog flux columns such as ap2 and Kron.")
    parser.add_argument("--photometry-psf-factor", type=float, default=1.0)
    parser.add_argument("--tp-isolated-max-shape-iou", type=float, default=0.05, help="Write an extra CSV for TP sources whose GT Kron ellipse max IoU with any other clean GT ellipse is at most this value.")
    parser.add_argument("--overlay-alpha", type=float, default=0.38)
    parser.add_argument("--max-contour-vertices", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if float(args.max_mask_area_ratio) <= 0.0:
        raise ValueError("--max-mask-area-ratio must be in (0, 1]; use >=1 to disable large-mask filtering.")
    args.ckpt_dir = args.ckpt_dir.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    if args.native_sam_dir is not None and str(args.native_sam_dir).lower() in {"", "none", "null"}:
        args.native_sam_dir = None
    elif args.native_sam_dir is not None:
        args.native_sam_dir = args.native_sam_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve() if args.config else args.ckpt_dir / "run_config.json"
    cfg = _read_config(config_path)
    loss_cfg = cfg.get("_top", {}).get("loss_weights", {})
    if args.multimask is None:
        args.multimask = bool(loss_cfg.get("mask_multimask", not bool(cfg.get("disable_mask_multimask", False))))
    if args.mask_prompt_center_only is None:
        args.mask_prompt_center_only = bool(cfg.get("mask_prompt_center_only", False))
    if args.pred_iou_thresh is None and float(loss_cfg.get("mask_pred_iou", 0.0)) > 0.0:
        args.pred_iou_thresh = float(loss_cfg.get("mask_pred_iou_thresh", 0.8))
    if args.stability_score_thresh is None and float(loss_cfg.get("mask_stability", 0.0)) > 0.0:
        args.stability_score_thresh = float(loss_cfg.get("mask_stability_score_thresh", 0.95))
    if args.band not in args.bands:
        raise ValueError(f"--band {args.band!r} is not present in --bands {args.bands}")

    if args.checkpoint:
        ckpt_items = [(Path(path).expanduser(), Path(path).stem) for path in args.checkpoint]
        if args.checkpoint_label:
            if len(args.checkpoint_label) != len(ckpt_items):
                raise ValueError("--checkpoint-label must be passed once per --checkpoint")
            ckpt_items = [(path, str(label)) for (path, _stem), label in zip(ckpt_items, args.checkpoint_label)]
    else:
        ckpt_items = [(args.ckpt_dir / "best.pt", "best"), (args.ckpt_dir / "last.pt", "latest")]

    device = torch.device(args.device)
    rows: list[dict[str, object]] = []
    for checkpoint, label in ckpt_items:
        if not checkpoint.is_absolute():
            checkpoint = args.ckpt_dir / checkpoint
        model = _make_model(cfg, checkpoint, device, args.bands)
        for dataset_name in args.datasets:
            row = _run_one(
                model=model,
                cfg=cfg,
                dataset_root=args.data_root,
                dataset_name=dataset_name,
                checkpoint_label=label,
                out_dir=args.out_dir,
                bands=args.bands,
                band=args.band,
                device=device,
                args=args,
            )
            row["checkpoint"] = str(checkpoint)
            row["checkpoint_epoch"] = _checkpoint_epoch(checkpoint)
            rows.append(row)
            print(
                f"{label} {dataset_name} {args.band}: det={row['detections']} "
                f"masks={row['kept_masks']} TP/FN/GT={row['clean_tp']}/{row['clean_fn']}/{row['clean_gt']}",
                flush=True,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with (args.out_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {args.out_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
