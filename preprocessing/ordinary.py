"""Ordinary-source branch filters for preprocessing v3."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from astropy.table import Table

from .labels import SourceClass, SourceLabels
from .refit import RefitConfig, compute_kron_ellipse
from .snr import SnrConfig, classify_snr
from .utils.catalog import boolean_column, magnitude_from_flux
from .utils.geometry import EllipseGeometry, ellipse_contains_dict


@dataclass(frozen=True)
class OrdinaryConfig:
    ap2_flux_column: str = "base_CircularApertureFlux_6_0_instFlux"
    kron_flux_column: str = "ext_photometryKron_KronFlux_instFlux"
    ap2_kron_abs_max: float = 1.0
    zeropoint: float = 27.0
    b_flags: tuple[str, ...] = ("base_SdssShape_flag", "base_SdssCentroid_flag")
    containment_threshold: float = 0.80
    containment_max_sources: int = 0
    fill_area_min: float = 500.0
    fill_ratio_max: float = 0.3


@dataclass
class OrdinaryResult:
    labels: SourceLabels
    ap2_mag: np.ndarray
    kron_mag: np.ndarray
    ap2_minus_kron_mag: np.ndarray
    ap2_keep: np.ndarray
    b_flag_removed: np.ndarray
    containment_removed: np.ndarray
    fill_ratio: np.ndarray
    strict_center_by_fill: np.ndarray
    snr: np.ndarray
    snr_class: np.ndarray
    clean: np.ndarray
    weak_shape: np.ndarray
    strict_center_only: np.ndarray
    ordinary_ignore: np.ndarray


def ap2_kron_keep_mask(table: Table, *, config: OrdinaryConfig = OrdinaryConfig()) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ap2 = magnitude_from_flux(table, column=config.ap2_flux_column, zeropoint=config.zeropoint)
    kron = magnitude_from_flux(table, column=config.kron_flux_column, zeropoint=config.zeropoint)
    diff = ap2 - kron
    keep = np.isfinite(diff) & (np.abs(diff) <= config.ap2_kron_abs_max)
    return keep, diff, ap2, kron


def b_flag_removed_mask(table: Table, candidate: np.ndarray, *, config: OrdinaryConfig = OrdinaryConfig()) -> np.ndarray:
    out = np.zeros(len(table), dtype=bool)
    for name in config.b_flags:
        out |= boolean_column(table, name, default=False)
    return np.asarray(candidate, dtype=bool) & out


def aperture_fill_ratio(
    table: Table,
    geom: EllipseGeometry,
    *,
    config: OrdinaryConfig = OrdinaryConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    ratio = np.full(len(table), np.nan, dtype=np.float64)
    strict = np.zeros(len(table), dtype=bool)
    if "pu_refit_aperture_pixel_count" not in table.colnames:
        return ratio, strict
    aperture_pixels = np.asarray(table["pu_refit_aperture_pixel_count"], dtype=np.float64)
    valid = np.isfinite(aperture_pixels) & np.isfinite(geom.area) & (geom.area > 0.0)
    ratio[valid] = aperture_pixels[valid] / geom.area[valid]
    strict = (
        valid
        & (geom.area > float(config.fill_area_min))
        & (ratio < float(config.fill_ratio_max))
    )
    return ratio, strict


def _ellipse_dict(geom: EllipseGeometry, idx: int) -> dict[str, object]:
    return {
        "x": float(geom.x[idx]),
        "y": float(geom.y[idx]),
        "major": float(geom.major[idx]),
        "minor": float(geom.minor[idx]),
        "theta_deg": math.degrees(float(geom.theta[idx])),
        "area": float(geom.area[idx]),
    }


def _contained_fraction(inner: dict[str, object], outer: dict[str, object]) -> float:
    max_inner = max(float(inner["major"]), float(inner["minor"]))
    max_outer = max(float(outer["major"]), float(outer["minor"]))
    if math.hypot(float(inner["x"]) - float(outer["x"]), float(inner["y"]) - float(outer["y"])) > max_inner + max_outer:
        return 0.0
    x0 = math.floor(float(inner["x"]) - max_inner)
    x1 = math.ceil(float(inner["x"]) + max_inner)
    y0 = math.floor(float(inner["y"]) - max_inner)
    y1 = math.ceil(float(inner["y"]) + max_inner)
    if x1 < x0 or y1 < y0:
        return 0.0
    step = max(1, int(math.ceil(max(x1 - x0 + 1, y1 - y0 + 1) / 384.0)))
    yy, xx = np.mgrid[y0 : y1 + 1 : step, x0 : x1 + 1 : step]
    inside_inner = ellipse_contains_dict(inner, xx.astype(np.float32), yy.astype(np.float32))
    denom = int(np.count_nonzero(inside_inner))
    if denom == 0:
        return 0.0
    inside_outer = ellipse_contains_dict(outer, xx.astype(np.float32), yy.astype(np.float32))
    return float(np.count_nonzero(inside_inner & inside_outer) / denom)


def source_containment_removed_mask(
    geom: EllipseGeometry,
    candidate: np.ndarray,
    *,
    threshold: float = 0.80,
    max_sources: int = 0,
) -> np.ndarray:
    """Remove larger ordinary sources that contain >=threshold of a smaller one."""

    candidate = np.asarray(candidate, dtype=bool)
    idx = np.flatnonzero(candidate & geom.valid())
    removed = np.zeros(len(candidate), dtype=bool)
    if idx.size < 2:
        return removed
    # ``max_sources`` is retained for diagnostics/backward config
    # compatibility.  The implementation below prunes by x-position and
    # ellipse radius; silently skipping containment on dense patches would make
    # the labels depend on catalog density, so it is intentionally not a hard
    # guard.
    order = idx[np.argsort(geom.x[idx], kind="mergesort")]
    max_radius = np.maximum(geom.major, geom.minor)
    for pos, i in enumerate(order):
        if removed[i]:
            continue
        xi = float(geom.x[i])
        ri = float(max_radius[i])
        for j in order[pos + 1 :]:
            dx = float(geom.x[j]) - xi
            if dx > ri + float(max_radius[j]):
                break
            if removed[j]:
                continue
            if dx * dx + (float(geom.y[j]) - float(geom.y[i])) ** 2 > (ri + float(max_radius[j])) ** 2:
                continue
            if geom.area[i] <= geom.area[j]:
                small_idx, large_idx = int(i), int(j)
            else:
                small_idx, large_idx = int(j), int(i)
            small = _ellipse_dict(geom, small_idx)
            large = _ellipse_dict(geom, large_idx)
            if _contained_fraction(small, large) >= float(threshold):
                removed[large_idx] = True
    return removed


def label_ordinary_sources(
    table: Table,
    candidate: np.ndarray,
    labels: SourceLabels,
    *,
    is_narrow_band: bool,
    snr: np.ndarray | None = None,
    config: OrdinaryConfig = OrdinaryConfig(),
    snr_config: SnrConfig = SnrConfig(),
    refit_config: RefitConfig = RefitConfig(),
) -> OrdinaryResult:
    """Assign ordinary candidates to clean/weak_shape/strict_center/ignore."""

    candidate = np.asarray(candidate, dtype=bool)
    geom = compute_kron_ellipse(table, refit_config)
    ap2_keep, diff, ap2, kron = ap2_kron_keep_mask(table, config=config)

    labels.assign(candidate & ~ap2_keep, SourceClass.ORDINARY_IGNORE, "ordinary_ap2_kron")
    active = candidate & ap2_keep

    fill_ratio, strict_by_fill_all = aperture_fill_ratio(table, geom, config=config)
    strict_by_fill = active & strict_by_fill_all
    labels.assign(strict_by_fill, SourceClass.STRICT_CENTER_ONLY, "ordinary_fill_ratio_strict_center")
    active &= ~strict_by_fill

    b_removed = b_flag_removed_mask(table, active, config=config)
    labels.assign(b_removed, SourceClass.ORDINARY_IGNORE, "ordinary_b_flag")
    active &= ~b_removed

    contained = source_containment_removed_mask(
        geom,
        active,
        threshold=float(config.containment_threshold),
        max_sources=int(config.containment_max_sources),
    )
    labels.assign(contained, SourceClass.ORDINARY_IGNORE, "ordinary_source_containment")
    active &= ~contained

    snr_values = np.full(len(table), np.nan, dtype=np.float32) if snr is None else np.asarray(snr, dtype=np.float32)
    snr_class = np.full(len(table), "", dtype="U16")
    if snr is None:
        labels.assign(active, SourceClass.CLEAN, "ordinary_clean_no_snr")
    else:
        classes = classify_snr(snr_values, area=geom.area, is_narrow_band=is_narrow_band, config=snr_config)
        snr_class[:] = classes
        labels.assign(active & (classes == "ignore"), SourceClass.ORDINARY_IGNORE, "ordinary_snr_ignore")
        labels.assign(active & (classes == "weak_shape"), SourceClass.WEAK_SHAPE, "ordinary_snr_weak_shape")
        labels.assign(active & (classes == "clean"), SourceClass.CLEAN, "ordinary_snr_clean")

    return OrdinaryResult(
        labels=labels,
        ap2_mag=ap2,
        kron_mag=kron,
        ap2_minus_kron_mag=diff,
        ap2_keep=ap2_keep,
        b_flag_removed=b_removed,
        containment_removed=contained,
        fill_ratio=fill_ratio,
        strict_center_by_fill=strict_by_fill,
        snr=snr_values,
        snr_class=snr_class,
        clean=labels.mask(SourceClass.CLEAN) & candidate,
        weak_shape=labels.mask(SourceClass.WEAK_SHAPE) & candidate,
        strict_center_only=labels.mask(SourceClass.STRICT_CENTER_ONLY) & candidate,
        ordinary_ignore=labels.mask(SourceClass.ORDINARY_IGNORE) & candidate,
    )
