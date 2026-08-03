"""SNR filtering and non-coadd SNR scaling for preprocessing v3."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from astropy.io import fits
from astropy.table import Table

from .utils.catalog import magnitude_from_flux, source_xy


SnrClass = Literal["clean", "weak_shape", "ignore"]


@dataclass(frozen=True)
class SnrConfig:
    """SNR thresholds and metadata roots used by ordinary-source filtering."""

    enable: bool = True
    ap2_flux_column: str = "base_CircularApertureFlux_6_0_instFlux"
    ap2_err_column: str = "base_CircularApertureFlux_6_0_instFluxErr"
    zeropoint: float = 27.0
    broad_ignore_snr_max: float = 3.0
    broad_weak_shape_snr_max: float = 5.0
    narrow_ignore_snr_max: float = 5.0
    narrow_weak_shape_snr_max: float = 8.0
    area_weak_shape_min: float = 500.0
    area_weak_shape_snr_max: float = 8.0
    aperture_radius: float = 6.0
    cap_t_max: float = 1.0
    noncoadd_method: Literal["auto", "variance", "weight", "none"] = "auto"
    missing_noncoadd_policy: Literal["fallback_coadd", "none", "error"] = "fallback_coadd"
    variance_unit_scale: float = 1.0
    coadd_weight_root: Path = Path("/data/czh23/2026-06-21_171607_hsc_metadata_warp_n2n_epoch006_full-all-warp-weights")
    denoised_fits_root: Path = Path("/data/czh23/denoised_fits")
    valid_coadd_weights_only: bool = False


@dataclass
class SnrResult:
    snr: np.ndarray
    snr_class: np.ndarray
    t_eff: np.ndarray
    method: str
    details: dict[str, object]


def ap2_flux(table: Table, *, config: SnrConfig = SnrConfig()) -> np.ndarray:
    if config.ap2_flux_column not in table.colnames:
        return np.full(len(table), np.nan, dtype=np.float64)
    return np.asarray(table[config.ap2_flux_column], dtype=np.float64)


def ap2_flux_err(table: Table, *, config: SnrConfig = SnrConfig()) -> np.ndarray:
    if config.ap2_err_column not in table.colnames:
        return np.full(len(table), np.nan, dtype=np.float64)
    return np.asarray(table[config.ap2_err_column], dtype=np.float64)


def ap2_mag(table: Table, *, config: SnrConfig = SnrConfig()) -> np.ndarray:
    return magnitude_from_flux(table, column=config.ap2_flux_column, zeropoint=config.zeropoint)


def ap2_snr(table: Table, *, config: SnrConfig = SnrConfig()) -> np.ndarray:
    flux = ap2_flux(table, config=config)
    err = ap2_flux_err(table, config=config)
    snr = np.full(len(table), np.nan, dtype=np.float64)
    valid = np.isfinite(flux) & np.isfinite(err) & (err > 0.0)
    snr[valid] = flux[valid] / err[valid]
    return snr


def snr_thresholds(*, is_narrow_band: bool, config: SnrConfig = SnrConfig()) -> tuple[float, float]:
    if is_narrow_band:
        return float(config.narrow_ignore_snr_max), float(config.narrow_weak_shape_snr_max)
    return float(config.broad_ignore_snr_max), float(config.broad_weak_shape_snr_max)


def classify_snr(
    snr: np.ndarray,
    *,
    area: np.ndarray | None = None,
    is_narrow_band: bool,
    config: SnrConfig = SnrConfig(),
) -> np.ndarray:
    """Classify SNR into clean/weak_shape/ignore.

    ``weak_shape`` is the v3 name for the old SNR ``center_only`` shell: it
    still provides center and weak shape supervision.  The fill-ratio path that
    disables shape entirely is handled in ``ordinary.py``.
    """

    snr = np.asarray(snr, dtype=np.float64)
    ignore_thr, weak_thr = snr_thresholds(is_narrow_band=is_narrow_band, config=config)
    out = np.full(snr.shape, "ignore", dtype="U16")
    finite = np.isfinite(snr)
    out[finite & (snr > ignore_thr)] = "weak_shape"
    out[finite & (snr > weak_thr)] = "clean"
    if area is not None:
        area = np.asarray(area, dtype=np.float64)
        area_weak = (
            finite
            & (snr > ignore_thr)
            & (snr <= float(config.area_weak_shape_snr_max))
            & np.isfinite(area)
            & (area > float(config.area_weak_shape_min))
        )
        out[area_weak] = "weak_shape"
    return out


def circular_aperture_offsets(radius: float) -> tuple[np.ndarray, np.ndarray]:
    r = int(math.ceil(float(radius)))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    keep = (xx.astype(np.float64) ** 2 + yy.astype(np.float64) ** 2) <= float(radius) ** 2
    return yy[keep].astype(np.int32), xx[keep].astype(np.int32)


def mean_map_value_at(
    image: np.ndarray,
    origin: tuple[float, float],
    x: float,
    y: float,
    off_y: np.ndarray,
    off_x: np.ndarray,
) -> float:
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


def _find_image_hdu_index(hdul: fits.HDUList) -> int:
    if "IMAGE" in hdul:
        return int(hdul.index_of("IMAGE"))
    for idx, hdu in enumerate(hdul):
        data = getattr(hdu, "data", None)
        if data is not None and getattr(data, "ndim", None) == 2:
            return int(idx)
    raise KeyError("No 2D image HDU found")


def _plane_hdu_index(hdul: fits.HDUList, plane: str) -> int:
    plane = str(plane).upper()
    if plane in hdul:
        return int(hdul.index_of(plane))
    image_idx = _find_image_hdu_index(hdul)
    if plane == "IMAGE":
        return image_idx
    candidate = image_idx + {"MASK": 1, "VARIANCE": 2}.get(plane, 0)
    if candidate < len(hdul):
        return int(candidate)
    raise KeyError(f"{plane} plane not found")


def read_fits_plane(path: Path | str, plane: str) -> tuple[np.ndarray, tuple[float, float]]:
    """Read an LSST FITS plane and return full-patch pixel origin."""

    with fits.open(Path(path), memmap=True, ignore_missing_end=True) as hdul:
        idx = _plane_hdu_index(hdul, plane)
        hdu = hdul[idx]
        data = np.asarray(hdu.data, dtype=np.float32)
        origin = (-float(hdu.header.get("LTV1", 0.0)), -float(hdu.header.get("LTV2", 0.0)))
    return data, origin


def mean_variance_at_sources(
    variance: np.ndarray,
    origin: tuple[float, float],
    table: Table,
    *,
    config: SnrConfig = SnrConfig(),
) -> np.ndarray:
    x, y = source_xy(table)
    off_y, off_x = circular_aperture_offsets(float(config.aperture_radius))
    out = np.full(len(table), np.nan, dtype=np.float32)
    for idx in range(len(table)):
        if np.isfinite(x[idx]) and np.isfinite(y[idx]):
            out[idx] = mean_map_value_at(variance, origin, float(x[idx]), float(y[idx]), off_y, off_x)
    return out


def predicted_snr_from_variance(
    coadd_snr: np.ndarray,
    coadd_variance: np.ndarray,
    image_variance: np.ndarray,
    *,
    config: SnrConfig = SnrConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    coadd_snr = np.asarray(coadd_snr, dtype=np.float64)
    coadd_variance = np.asarray(coadd_variance, dtype=np.float64)
    image_variance = np.asarray(image_variance, dtype=np.float64)
    t_eff = np.full(coadd_snr.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(coadd_snr)
        & (coadd_snr > 0.0)
        & np.isfinite(coadd_variance)
        & (coadd_variance > 0.0)
        & np.isfinite(image_variance)
        & (image_variance > 0.0)
    )
    scale = float(config.variance_unit_scale)
    t_eff[valid] = coadd_variance[valid] * scale * scale / image_variance[valid]
    if float(config.cap_t_max) > 0.0:
        t_eff = np.minimum(t_eff, float(config.cap_t_max))
    snr = np.full(coadd_snr.shape, np.nan, dtype=np.float64)
    ok = valid & np.isfinite(t_eff) & (t_eff >= 0.0)
    snr[ok] = coadd_snr[ok] * np.sqrt(t_eff[ok])
    return snr.astype(np.float32), t_eff.astype(np.float32)


def resolve_coadd_weight_csv(root: Path | str, band: str, patch: str) -> Path:
    return Path(root) / str(band) / str(patch) / "weights.csv"


def read_coadd_weight_sum(path: Path | str, *, valid_only: bool = False) -> tuple[float, int]:
    total = 0.0
    count = 0
    with Path(path).open(newline="", encoding="utf-8") as handle:
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


def resolve_variant_group_dir(root: Path | str, patch: str, group: str | int, band: str) -> Path:
    root = Path(root)
    patch_dir = f"patch_{str(patch).replace(',', '_')}"
    group_text = str(group)
    candidates = [group_text]
    try:
        candidates.append(f"group_{int(group_text):02d}")
    except Exception:
        pass
    if not group_text.startswith("group_"):
        candidates.append(f"group_{group_text}")
    for candidate in dict.fromkeys(candidates):
        path = root / patch_dir / candidate / str(band)
        if path.exists():
            return path
    return root / patch_dir / group_text / str(band)


def read_group_weight_summary(meta_path: Path | str) -> dict[str, float]:
    metadata = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    weights = [float(value) for value in metadata.get("selected_weights", [])]
    weights = [value for value in weights if math.isfinite(value) and value > 0.0]
    return {
        "selected_weight_sum": float(sum(weights)),
        "selected_weight_count": float(len(weights)),
        "selected_weight_min": float(min(weights)) if weights else float("nan"),
        "selected_weight_max": float(max(weights)) if weights else float("nan"),
        "group_index": float(metadata.get("group_index", -1)),
    }


def predicted_snr_from_weight_ratio(
    coadd_snr: np.ndarray,
    *,
    coadd_weight_sum: float,
    selected_weight_sum: float,
    selected_weight_count: float,
    local_effective_count: np.ndarray,
    config: SnrConfig = SnrConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    coadd_snr = np.asarray(coadd_snr, dtype=np.float64)
    local_effective_count = np.asarray(local_effective_count, dtype=np.float64)
    t_eff = np.full(coadd_snr.shape, np.nan, dtype=np.float64)
    snr = np.full(coadd_snr.shape, np.nan, dtype=np.float64)
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
        if float(config.cap_t_max) > 0.0:
            t_eff = np.minimum(t_eff, float(config.cap_t_max))
        snr_ok = ok & np.isfinite(coadd_snr) & (coadd_snr > 0.0) & np.isfinite(t_eff) & (t_eff >= 0.0)
        snr[snr_ok] = coadd_snr[snr_ok] * np.sqrt(t_eff[snr_ok])
    return snr.astype(np.float32), t_eff.astype(np.float32)


def compute_weight_ratio_snr(
    table: Table,
    *,
    band: str,
    patch: str,
    group: str | int,
    config: SnrConfig = SnrConfig(),
) -> SnrResult:
    coadd = ap2_snr(table, config=config)
    weight_csv = resolve_coadd_weight_csv(config.coadd_weight_root, band, patch)
    group_dir = resolve_variant_group_dir(config.denoised_fits_root, patch, group, band)
    coadd_weight_sum, coadd_weight_count = read_coadd_weight_sum(weight_csv, valid_only=config.valid_coadd_weights_only)
    group_weights = read_group_weight_summary(group_dir / "meta.json")
    effective_count, effective_origin = read_fits_plane(group_dir / "effective_count.fits", "IMAGE")
    x, y = source_xy(table)
    off_y, off_x = circular_aperture_offsets(float(config.aperture_radius))
    local_effective = np.full(len(table), np.nan, dtype=np.float32)
    for idx in range(len(table)):
        if np.isfinite(x[idx]) and np.isfinite(y[idx]):
            local_effective[idx] = mean_map_value_at(effective_count, effective_origin, float(x[idx]), float(y[idx]), off_y, off_x)
    snr, t_eff = predicted_snr_from_weight_ratio(
        coadd,
        coadd_weight_sum=coadd_weight_sum,
        selected_weight_sum=float(group_weights["selected_weight_sum"]),
        selected_weight_count=float(group_weights["selected_weight_count"]),
        local_effective_count=local_effective,
        config=config,
    )
    return SnrResult(
        snr=snr,
        snr_class=np.asarray([], dtype="U16"),
        t_eff=t_eff,
        method="weight",
        details={
            "coadd_weight_csv": str(weight_csv),
            "coadd_weight_sum": float(coadd_weight_sum),
            "coadd_weight_count": int(coadd_weight_count),
            "group_dir": str(group_dir),
            **group_weights,
        },
    )


def compute_variance_scaled_snr(
    table: Table,
    *,
    image_fits: Path | str,
    coadd_image_fits: Path | str,
    config: SnrConfig = SnrConfig(),
) -> SnrResult:
    coadd = ap2_snr(table, config=config)
    coadd_var, coadd_origin = read_fits_plane(coadd_image_fits, "VARIANCE")
    image_var, image_origin = read_fits_plane(image_fits, "VARIANCE")
    coadd_v = mean_variance_at_sources(coadd_var, coadd_origin, table, config=config)
    image_v = mean_variance_at_sources(image_var, image_origin, table, config=config)
    snr, t_eff = predicted_snr_from_variance(coadd, coadd_v, image_v, config=config)
    return SnrResult(
        snr=snr,
        snr_class=np.asarray([], dtype="U16"),
        t_eff=t_eff,
        method="variance",
        details={"image_fits": str(image_fits), "coadd_image_fits": str(coadd_image_fits)},
    )


def compute_snr_for_sample(
    table: Table,
    *,
    dataset_source: str = "coadd",
    is_narrow_band: bool,
    band: str | None = None,
    patch: str | None = None,
    group: str | int | None = None,
    image_fits: Path | str | None = None,
    coadd_image_fits: Path | str | None = None,
    config: SnrConfig = SnrConfig(),
) -> SnrResult | None:
    if not bool(config.enable) or config.noncoadd_method == "none":
        return None
    source = str(dataset_source or "coadd").lower()
    base = ap2_snr(table, config=config)
    if source == "coadd":
        return SnrResult(
            snr=base.astype(np.float32),
            snr_class=classify_snr(base, is_narrow_band=is_narrow_band, config=config),
            t_eff=np.ones(len(table), dtype=np.float32),
            method="coadd_ap2",
            details={},
        )

    errors: list[str] = []
    if config.noncoadd_method in {"auto", "variance"} and image_fits is not None and coadd_image_fits is not None:
        try:
            result = compute_variance_scaled_snr(table, image_fits=image_fits, coadd_image_fits=coadd_image_fits, config=config)
            result.snr_class = classify_snr(result.snr, is_narrow_band=is_narrow_band, config=config)
            return result
        except Exception as exc:
            errors.append(f"variance: {exc}")
            if config.noncoadd_method == "variance":
                if config.missing_noncoadd_policy == "error":
                    raise

    if config.noncoadd_method in {"auto", "weight"} and band is not None and patch is not None and group is not None:
        try:
            result = compute_weight_ratio_snr(table, band=band, patch=patch, group=group, config=config)
            result.snr_class = classify_snr(result.snr, is_narrow_band=is_narrow_band, config=config)
            return result
        except Exception as exc:
            errors.append(f"weight: {exc}")
            if config.noncoadd_method == "weight":
                if config.missing_noncoadd_policy == "error":
                    raise

    if config.missing_noncoadd_policy == "error":
        raise RuntimeError("; ".join(errors) if errors else "noncoadd SNR inputs are missing")
    if config.missing_noncoadd_policy == "none":
        return None
    return SnrResult(
        snr=base.astype(np.float32),
        snr_class=classify_snr(base, is_narrow_band=is_narrow_band, config=config),
        t_eff=np.ones(len(table), dtype=np.float32),
        method="fallback_coadd_ap2",
        details={"errors": errors},
    )
