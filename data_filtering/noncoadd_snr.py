#!/usr/bin/env python3
"""Non-coadd image SNR measurement and visibility classification."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits
from astropy.table import Table


_POSITION_COLUMN_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("base_SdssCentroid_x", "base_SdssCentroid_y"),
    ("base_SdssShape_x", "base_SdssShape_y"),
    ("base_NaiveCentroid_x", "base_NaiveCentroid_y"),
    ("deblend_psfCenter_x", "deblend_psfCenter_y"),
    ("slot_Centroid_x", "slot_Centroid_y"),
    ("x", "y"),
)


def _find_image_hdu_index(hdul: fits.HDUList) -> int:
    if "IMAGE" in hdul:
        return hdul.index_of("IMAGE")
    for idx, hdu in enumerate(hdul):
        data = getattr(hdu, "data", None)
        if data is not None and getattr(data, "ndim", None) == 2:
            return idx
    raise KeyError("No 2D image HDU found; expected IMAGE or a 2D image extension")


def _plane_hdu_indices(hdul: fits.HDUList) -> Dict[str, int]:
    if all(plane in hdul for plane in ("IMAGE", "MASK", "VARIANCE")):
        return {plane: hdul.index_of(plane) for plane in ("IMAGE", "MASK", "VARIANCE")}

    image_idx = _find_image_hdu_index(hdul)
    indices = {"IMAGE": image_idx}
    image_shape = hdul[image_idx].data.shape
    for plane, idx in (("MASK", image_idx + 1), ("VARIANCE", image_idx + 2)):
        if idx < len(hdul):
            data = getattr(hdul[idx], "data", None)
            if data is not None and getattr(data, "ndim", None) == 2 and data.shape == image_shape:
                indices[plane] = idx
    return indices


def _mask_plane_bit_mapping(header: fits.Header) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for key, value in header.items():
        text_key = str(key).upper()
        if not text_key.startswith("MP_"):
            continue
        try:
            mapping[text_key[3:]] = int(value)
        except Exception:
            continue
    return mapping


def read_lsst_quality_mask(path: Path, mask_planes: Sequence[str]) -> Optional[np.ndarray]:
    """Return a boolean mask for selected LSST mask planes, or ``None``."""

    if not path.exists() or not mask_planes:
        return None
    try:
        with fits.open(path, memmap=False, ignore_missing_end=True) as hdul:
            plane_indices = _plane_hdu_indices(hdul)
            mask_idx = plane_indices.get("MASK")
            if mask_idx is None:
                return None
            mask_hdu = hdul[mask_idx]
            mask_data = np.asarray(mask_hdu.data)
            if mask_data.ndim != 2:
                return None
            bit_mapping = _mask_plane_bit_mapping(mask_hdu.header)
            selected = np.zeros(mask_data.shape, dtype=bool)
            mask_values = mask_data.astype(np.int64, copy=False)
            for plane in mask_planes:
                plane_name = str(plane).strip().upper()
                if not plane_name:
                    continue
                bit = bit_mapping.get(plane_name)
                if bit is not None:
                    selected |= (mask_values & (1 << int(bit))) != 0
            return selected
    except Exception as exc:
        print(f"WARNING: failed to read quality mask from {path}: {exc}", flush=True)
        return None


def _resolve_position_columns(table: Table, x_col: str, y_col: str) -> Tuple[str, str]:
    if x_col in table.colnames and y_col in table.colnames:
        return x_col, y_col
    for candidate_x, candidate_y in _POSITION_COLUMN_PAIRS:
        if candidate_x in table.colnames and candidate_y in table.colnames:
            return candidate_x, candidate_y
    available = ", ".join(table.colnames[:20])
    suffix = "..." if len(table.colnames) > 20 else ""
    raise KeyError(
        f"catalog must contain {x_col!r} and {y_col!r}, or one of "
        f"{_POSITION_COLUMN_PAIRS}; available columns: {available}{suffix}"
    )


def _require_position_columns(table: Table, x_col: str, y_col: str) -> Tuple[np.ndarray, np.ndarray]:
    resolved_x_col, resolved_y_col = _resolve_position_columns(table, x_col, y_col)
    return np.asarray(table[resolved_x_col], dtype=np.float32), np.asarray(table[resolved_y_col], dtype=np.float32)


def _paint_ellipse_mask(
    mask: np.ndarray,
    sources: Table,
    *,
    tile_origin: Tuple[int, int],
    x_col: str,
    y_col: str,
    ellipse_sigma: float,
) -> None:
    if len(sources) == 0:
        return
    h, w = mask.shape
    x_global, y_global = _require_position_columns(sources, x_col, y_col)
    major = np.asarray(sources["ellipse_major_sigma"], dtype=np.float32)
    minor = np.asarray(sources["ellipse_minor_sigma"], dtype=np.float32)
    angle = np.asarray(sources["ellipse_theta"], dtype=np.float32)
    yy_full, xx_full = np.mgrid[0:h, 0:w]
    for idx in range(len(sources)):
        cx = float(x_global[idx] - float(tile_origin[0]))
        cy = float(y_global[idx] - float(tile_origin[1]))
        if np.isfinite(major[idx]) and np.isfinite(minor[idx]) and np.isfinite(angle[idx]):
            a = float(max(major[idx] * ellipse_sigma, 1.5))
            b = float(max(minor[idx] * ellipse_sigma, 1.5))
            theta = float(angle[idx])
        else:
            a = b = float(max(2.0 * ellipse_sigma, 1.5))
            theta = 0.0
        radius = int(math.ceil(max(a, b))) + 2
        cx_i = int(round(cx))
        cy_i = int(round(cy))
        y0, y1 = max(0, cy_i - radius), min(h, cy_i + radius + 1)
        x0, x1 = max(0, cx_i - radius), min(w, cx_i + radius + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        dx = xx_full[y0:y1, x0:x1] - cx
        dy = yy_full[y0:y1, x0:x1] - cy
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        xr = cos_t * dx + sin_t * dy
        yr = -sin_t * dx + cos_t * dy
        ellipse = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
        mask[y0:y1, x0:x1][ellipse] = 1


def source_annulus_exclusion_mask(
    sources: Table,
    *,
    tile_size: int,
    tile_origin: Tuple[int, int],
    x_col: str,
    y_col: str,
    ellipse_sigma: float,
) -> np.ndarray:
    mask = np.zeros((int(tile_size), int(tile_size)), dtype=np.uint8)
    _paint_ellipse_mask(
        mask,
        sources,
        tile_origin=tile_origin,
        x_col=x_col,
        y_col=y_col,
        ellipse_sigma=ellipse_sigma,
    )
    return mask > 0


def aperture_annulus_snr(
    image: np.ndarray,
    centers: np.ndarray,
    *,
    ap_radius: float = 6.0,
    annulus_r_in: float = 10.0,
    annulus_r_out: float = 15.0,
    annulus_exclude_mask: Optional[np.ndarray] = None,
    annulus_hard_exclude_mask: Optional[np.ndarray] = None,
    annulus_self_ellipse_params: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    annulus_self_ellipse_sigma: float = 1.0,
    exclude_self_source: bool = True,
    min_annulus_pixels: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    snr = np.full((centers.shape[0],), np.nan, dtype=np.float32)
    annulus_counts = np.zeros((centers.shape[0],), dtype=np.int32)
    if image.ndim != 2 or centers.size == 0:
        return snr, annulus_counts

    exclude = None
    if annulus_exclude_mask is not None:
        exclude = np.asarray(annulus_exclude_mask, dtype=bool)
        if exclude.shape != image.shape:
            raise ValueError(f"annulus_exclude_mask shape {exclude.shape} != image shape {image.shape}")
    hard_exclude = None
    if annulus_hard_exclude_mask is not None:
        hard_exclude = np.asarray(annulus_hard_exclude_mask, dtype=bool)
        if hard_exclude.shape != image.shape:
            raise ValueError(f"annulus_hard_exclude_mask shape {hard_exclude.shape} != image shape {image.shape}")

    self_major = self_minor = self_theta = None
    if annulus_self_ellipse_params is not None:
        self_major, self_minor, self_theta = annulus_self_ellipse_params
        self_major = np.asarray(self_major, dtype=np.float32)
        self_minor = np.asarray(self_minor, dtype=np.float32)
        self_theta = np.asarray(self_theta, dtype=np.float32)
        if not (len(self_major) == len(self_minor) == len(self_theta) == centers.shape[0]):
            raise ValueError("annulus_self_ellipse_params must have one entry per center")

    h, w = image.shape
    rmax = float(max(ap_radius, annulus_r_out))
    min_ann = max(2, int(min_annulus_pixels))
    for idx, (cx, cy) in enumerate(centers):
        if not (np.isfinite(cx) and np.isfinite(cy)):
            continue
        x0 = max(0, int(math.floor(float(cx) - rmax - 1.0)))
        x1 = min(w, int(math.ceil(float(cx) + rmax + 2.0)))
        y0 = max(0, int(math.floor(float(cy) - rmax - 1.0)))
        y1 = min(h, int(math.ceil(float(cy) + rmax + 2.0)))
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rr = np.sqrt((xx.astype(np.float32) - float(cx)) ** 2 + (yy.astype(np.float32) - float(cy)) ** 2)
        patch = image[y0:y1, x0:x1]
        finite = np.isfinite(patch)
        ap_mask = (rr <= float(ap_radius)) & finite
        ann_mask = (rr >= float(annulus_r_in)) & (rr < float(annulus_r_out)) & finite
        if hard_exclude is not None:
            ann_mask &= ~hard_exclude[y0:y1, x0:x1]
        if exclude is not None:
            exclude_patch = exclude[y0:y1, x0:x1]
            if self_major is not None and not bool(exclude_self_source):
                if np.isfinite(self_major[idx]) and np.isfinite(self_minor[idx]) and np.isfinite(self_theta[idx]):
                    a = float(max(self_major[idx] * float(annulus_self_ellipse_sigma), 1.5))
                    b = float(max(self_minor[idx] * float(annulus_self_ellipse_sigma), 1.5))
                    theta = float(self_theta[idx])
                else:
                    a = b = float(max(2.0 * float(annulus_self_ellipse_sigma), 1.5))
                    theta = 0.0
                dx = xx.astype(np.float32) - float(cx)
                dy = yy.astype(np.float32) - float(cy)
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                xr = cos_t * dx + sin_t * dy
                yr = -sin_t * dx + cos_t * dy
                self_ellipse = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
                ann_mask &= ~(exclude_patch & ~self_ellipse)
            else:
                ann_mask &= ~exclude_patch
        ap_vals = patch[ap_mask]
        ann_vals = patch[ann_mask]
        annulus_counts[idx] = int(ann_vals.size)
        if ap_vals.size == 0 or ann_vals.size < min_ann:
            continue
        bkg = float(np.median(ann_vals))
        sigma = float(np.std(ann_vals.astype(np.float64), ddof=1))
        if not math.isfinite(sigma) or sigma <= 0.0:
            continue
        flux = float(np.sum(ap_vals.astype(np.float64))) - bkg * float(ap_vals.size)
        noise = sigma * math.sqrt(float(ap_vals.size))
        if noise > 0.0:
            snr[idx] = float(flux / noise)
    return snr, annulus_counts


def centers_for_image_snr(
    table: Table,
    *,
    image_origin: Tuple[int, int],
    x_col: str,
    y_col: str,
) -> np.ndarray:
    centers = np.zeros((len(table), 2), dtype=np.float32)
    if len(table) == 0:
        return centers
    x, y = _require_position_columns(table, x_col, y_col)
    centers[:, 0] = np.asarray(x, dtype=np.float32) - float(image_origin[0])
    centers[:, 1] = np.asarray(y, dtype=np.float32) - float(image_origin[1])
    return centers


def classify_clean_by_noncoadd_snr(
    clean_sources: Table,
    *,
    image: np.ndarray,
    image_origin: Tuple[int, int],
    args: argparse.Namespace,
    annulus_exclude_mask: Optional[np.ndarray] = None,
    annulus_hard_exclude_mask: Optional[np.ndarray] = None,
) -> Tuple[Table, Table, Table, np.ndarray]:
    if len(clean_sources) == 0:
        empty = clean_sources.copy(copy_data=True)
        return empty, empty.copy(copy_data=True), empty.copy(copy_data=True), np.zeros((0,), dtype=np.float32)
    centers = centers_for_image_snr(clean_sources, image_origin=image_origin, x_col=args.x_col, y_col=args.y_col)
    self_params = None
    if all(name in clean_sources.colnames for name in ("ellipse_major_sigma", "ellipse_minor_sigma", "ellipse_theta")):
        self_params = (
            np.asarray(clean_sources["ellipse_major_sigma"], dtype=np.float32),
            np.asarray(clean_sources["ellipse_minor_sigma"], dtype=np.float32),
            np.asarray(clean_sources["ellipse_theta"], dtype=np.float32),
        )
    snr, annulus_counts = aperture_annulus_snr(
        image,
        centers,
        ap_radius=float(args.noncoadd_snr_ap_radius),
        annulus_r_in=float(args.noncoadd_snr_annulus_r_in),
        annulus_r_out=float(args.noncoadd_snr_annulus_r_out),
        annulus_exclude_mask=annulus_exclude_mask,
        annulus_hard_exclude_mask=annulus_hard_exclude_mask,
        annulus_self_ellipse_params=self_params,
        annulus_self_ellipse_sigma=float(getattr(args, "noncoadd_snr_source_mask_ellipse_sigma", 1.0)),
        exclude_self_source=bool(getattr(args, "noncoadd_snr_exclude_self_source", True)),
        min_annulus_pixels=int(getattr(args, "noncoadd_snr_min_annulus_pixels", 2)),
    )
    finite_snr = np.isfinite(snr)
    insufficient_annulus = annulus_counts < int(getattr(args, "noncoadd_snr_min_annulus_pixels", 2))
    normal_keep = finite_snr & (snr >= float(args.noncoadd_snr_center_only_thresh))
    center_keep = insufficient_annulus | (
        finite_snr
        & (snr >= float(args.noncoadd_snr_ignore_thresh))
        & (snr < float(args.noncoadd_snr_center_only_thresh))
    )
    ignore_keep = (~insufficient_annulus) & ((~finite_snr) | (snr < float(args.noncoadd_snr_ignore_thresh)))

    annotated = clean_sources.copy(copy_data=True)
    visibility_class = np.full(len(annotated), "normal", dtype="U16")
    visibility_class[center_keep] = "center_only"
    visibility_class[ignore_keep] = "ignore"
    annotated["noncoadd_visibility_snr"] = snr.astype(np.float32)
    annotated["noncoadd_visibility_annulus_pixels"] = annulus_counts.astype(np.int32)
    annotated["noncoadd_visibility_class"] = visibility_class
    return (
        annotated[normal_keep],
        annotated[center_keep],
        annotated[ignore_keep],
        snr.astype(np.float32),
    )


def read_effective_count_map(path: Path) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Read a denoised/noisy effective-count map and its full-patch origin."""

    with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
        hdu = hdul[0]
        data = np.asarray(hdu.data, dtype=np.float32)
        origin = (-float(hdu.header.get("LTV1", 0.0)), -float(hdu.header.get("LTV2", 0.0)))
    return data, origin


def circular_aperture_offsets(radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return integer dy/dx offsets for a circular aperture."""

    r = int(math.ceil(float(radius)))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    keep = (xx.astype(np.float64) ** 2 + yy.astype(np.float64) ** 2) <= float(radius) ** 2
    return yy[keep].astype(np.int32), xx[keep].astype(np.int32)


def mean_map_value_at(
    image: np.ndarray,
    origin: Tuple[float, float],
    x: float,
    y: float,
    off_y: np.ndarray,
    off_x: np.ndarray,
) -> float:
    """Mean finite positive map value inside a circular aperture."""

    cx = int(round(float(x) - float(origin[0])))
    cy = int(round(float(y) - float(origin[1])))
    yy = cy + off_y
    xx = cx + off_x
    inside = (yy >= 0) & (yy < image.shape[0]) & (xx >= 0) & (xx < image.shape[1])
    if not bool(np.any(inside)):
        return float("nan")
    values = np.asarray(image[yy[inside], xx[inside]], dtype=np.float64)
    good = np.isfinite(values) & (values > 0.0)
    if not bool(np.any(good)):
        return float("nan")
    return float(np.mean(values[good]))


def read_coadd_weight_sum(path: Path, *, valid_only: bool = False) -> Tuple[float, int]:
    """Read a coadd warp-weight CSV and return summed finite positive weights."""

    total = 0.0
    count = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if valid_only:
                flag = str(row.get("has_valid_hsctile_weight", "")).strip().lower()
                if flag not in {"true", "1", "yes", "y"}:
                    continue
            try:
                weight = float(row["weight"])
            except Exception:
                continue
            if math.isfinite(weight) and weight > 0.0:
                total += weight
                count += 1
    return total, count


def read_noisy_group_weight_summary(meta_path: Path) -> Dict[str, float]:
    """Read noisy/denoised group metadata selected warp weights."""

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    weights = [float(value) for value in metadata.get("selected_weights", [])]
    weights = [value for value in weights if math.isfinite(value) and value > 0.0]
    return {
        "selected_weight_sum": float(sum(weights)),
        "selected_weight_count": float(len(weights)),
        "selected_weight_min": float(min(weights)) if weights else float("nan"),
        "selected_weight_max": float(max(weights)) if weights else float("nan"),
        "group_index": float(metadata.get("group_index", -1)),
    }


def predict_snr_from_weight_ratio(
    coadd_ap_snr: np.ndarray,
    *,
    coadd_weight_sum: float,
    selected_weight_sum: float,
    selected_weight_count: float,
    local_effective_count: np.ndarray,
    cap_t_max: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict non-coadd SNR from coadd AP SNR and noisy/coadd exposure ratio.

    ``selected_weight_sum / coadd_weight_sum`` gives the global inverse-variance
    exposure ratio for a noisy group.  The effective-count map is a local
    coverage correction: if only half of the selected warps contributed at a
    source position, the usable ratio is halved.
    """

    coadd_ap_snr = np.asarray(coadd_ap_snr, dtype=np.float64)
    local_effective_count = np.asarray(local_effective_count, dtype=np.float64)
    t_eff = np.full(coadd_ap_snr.shape, np.nan, dtype=np.float64)
    snr = np.full(coadd_ap_snr.shape, np.nan, dtype=np.float64)
    if (
        math.isfinite(float(coadd_weight_sum))
        and float(coadd_weight_sum) > 0.0
        and math.isfinite(float(selected_weight_sum))
        and float(selected_weight_sum) > 0.0
        and math.isfinite(float(selected_weight_count))
        and float(selected_weight_count) > 0.0
    ):
        global_t = float(selected_weight_sum) / float(coadd_weight_sum)
        coverage = local_effective_count / float(selected_weight_count)
        ok = np.isfinite(coverage) & (coverage > 0.0)
        t_eff[ok] = global_t * np.minimum(coverage[ok], 1.0)
        if float(cap_t_max) > 0.0:
            t_eff = np.minimum(t_eff, float(cap_t_max))
        snr_ok = ok & np.isfinite(coadd_ap_snr) & (coadd_ap_snr > 0.0) & np.isfinite(t_eff) & (t_eff >= 0.0)
        snr[snr_ok] = coadd_ap_snr[snr_ok] * np.sqrt(t_eff[snr_ok])
    return snr.astype(np.float32), t_eff.astype(np.float32)


def classify_snr_values(
    snr: np.ndarray,
    *,
    ignore_snr_max: float,
    center_only_snr_max: float,
) -> np.ndarray:
    """Classify predicted SNR values into clean/center_only/ignore strings."""

    values = np.asarray(snr, dtype=np.float64)
    classes = np.full(values.shape, "ignore", dtype="U16")
    finite = np.isfinite(values)
    classes[finite & (values > float(ignore_snr_max))] = "center_only"
    classes[finite & (values > float(center_only_snr_max))] = "clean"
    return classes
