"""Build PU-style source quality labels from an HSC/LSST meas catalog.

The script is intentionally diagnostic: it mirrors the current A/B flag
experiments in ``catalog_flag_filter_diagnostics.ipynb`` and adds a conservative
Kron-ellipse ambiguity pass.  Outputs are DS9 region files plus a CSV summary.
"""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from astropy.table import Table


DEFAULT_A_FLAGS = (
)

DEFAULT_B_FLAGS = (
    "detect_isPrimary",
    "base_SdssShape_flag",
    "base_SdssCentroid_flag",
)

POSITION_COLUMNS_X = (
    "base_SdssCentroid_x",
    "base_SdssShape_x",
    "base_NaiveCentroid_x",
    "deblend_psfCenter_x",
)
POSITION_COLUMNS_Y = (
    "base_SdssCentroid_y",
    "base_SdssShape_y",
    "base_NaiveCentroid_y",
    "deblend_psfCenter_y",
)


def _first_finite_column(table: Table, names: Sequence[str]) -> np.ndarray:
    out = np.full(len(table), np.nan, dtype=np.float64)
    for name in names:
        if name in table.colnames:
            values = np.asarray(table[name], dtype=np.float64)
            take = ~np.isfinite(out) & np.isfinite(values)
            out[take] = values[take]
            if np.isfinite(out).all():
                break
    return out


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def attach_kron_refit_radius(
    table: Table,
    refit_csv: Path | str | None,
    *,
    radius_column: str = "proxy_nan0_flux_aperture_radius",
    good_column: str = "proxy_nan0_good",
    output_column: str = "pu_refit_kron_radius",
) -> Table:
    """Attach a finite Kron refit radius column matched by source id.

    The refit script writes source ids in ``source_id`` and a proxy Kron radius.
    Rows without a good finite proxy radius keep NaN so the classifier can move
    them to ordinary ignore when proxy-only masks are required.
    """

    if refit_csv is None:
        return table
    path = Path(refit_csv)
    if not path.exists():
        raise FileNotFoundError(f"kron refit CSV not found: {path}")
    id_column = "id" if "id" in table.colnames else "source_id" if "source_id" in table.colnames else None
    if id_column is None:
        raise KeyError("catalog must contain id or source_id to attach Kron refit radii")

    refit_by_id: dict[int, float] = {}
    refit_aperture_pixels_by_id: dict[int, float] = {}
    refit_footprint_area_by_id: dict[int, float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"kron refit CSV has no header: {path}")
        if "source_id" not in reader.fieldnames:
            raise KeyError(f"kron refit CSV missing column 'source_id': {path}")
        if radius_column not in reader.fieldnames:
            out = table.copy(copy_data=True)
            out[output_column] = np.full(len(out), np.nan, dtype=np.float32)
            out[f"{output_column}_matched"] = np.zeros(len(out), dtype=bool)
            out[f"{output_column}_missing_column"] = np.asarray([str(radius_column)] * len(out), dtype=str)
            print(f"WARNING: kron refit CSV missing {radius_column!r}; rows will be treated as unmatched: {path}", flush=True)
            return out
        has_good = good_column in reader.fieldnames
        for row in reader:
            if has_good and not _as_bool(row.get(good_column, False)):
                continue
            try:
                source_id = int(row["source_id"])
                radius = float(row[radius_column])
            except (TypeError, ValueError):
                continue
            if np.isfinite(radius) and radius > 0.0:
                refit_by_id[source_id] = radius
                for column, bucket in (
                    ("aperture_pixel_count", refit_aperture_pixels_by_id),
                    ("footprint_area", refit_footprint_area_by_id),
                ):
                    if column not in reader.fieldnames:
                        continue
                    try:
                        value = float(row[column])
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value):
                        bucket[source_id] = value

    out = table.copy(copy_data=True)
    radii = np.full(len(out), np.nan, dtype=np.float32)
    matched = np.zeros(len(out), dtype=bool)
    aperture_pixels = np.full(len(out), np.nan, dtype=np.float32)
    footprint_area = np.full(len(out), np.nan, dtype=np.float32)
    ids = np.asarray(out[id_column], dtype=np.int64)
    for idx, source_id in enumerate(ids):
        value = refit_by_id.get(int(source_id))
        if value is None:
            continue
        radii[idx] = float(value)
        matched[idx] = True
        aperture_pixel_value = refit_aperture_pixels_by_id.get(int(source_id))
        if aperture_pixel_value is not None:
            aperture_pixels[idx] = float(aperture_pixel_value)
        footprint_area_value = refit_footprint_area_by_id.get(int(source_id))
        if footprint_area_value is not None:
            footprint_area[idx] = float(footprint_area_value)
    out[output_column] = radii
    out[f"{output_column}_matched"] = matched
    if refit_aperture_pixels_by_id:
        out["pu_refit_aperture_pixel_count"] = aperture_pixels
    if refit_footprint_area_by_id:
        out["pu_refit_footprint_area"] = footprint_area
    return out


def _flag_name_to_bit(table: Table) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in table.meta.items():
        if isinstance(key, str) and key.startswith("TFLAG"):
            try:
                out[str(value)] = int(key[len("TFLAG") :]) - 1
            except ValueError:
                continue
    return out


def _flag_values(
    table: Table,
    name: str,
    *,
    flag_map: dict[str, int],
    packed_flags: Optional[np.ndarray],
    strict: bool,
) -> np.ndarray:
    if name in table.colnames:
        return np.asarray(table[name], dtype=bool)
    if packed_flags is not None and name in flag_map:
        return np.asarray(packed_flags[:, flag_map[name]], dtype=bool)
    if strict:
        raise KeyError(f"Flag {name!r} not found in table columns or packed flags")
    print(f"WARNING: missing flag {name!r}; treating as all False", flush=True)
    return np.zeros(len(table), dtype=bool)


def _packed_flags_array(table: Table) -> Optional[np.ndarray]:
    if "flags" not in table.colnames:
        return None
    flags = np.asarray(table["flags"], dtype=bool)
    if flags.ndim < 2:
        return None
    return flags


def _removed_by_b_flags(
    table: Table,
    flags: Sequence[str],
    *,
    mode: str,
    strict: bool,
) -> tuple[np.ndarray, list[list[str]]]:
    """Return rows removed by B flags.

    Normal LSST failure flags remove sources when the flag is true.  The
    selection flag ``detect_isPrimary`` is inverted: rows are removed when it is
    false, because non-primary detections should not be clean supervision.
    """

    reasons: list[list[str]] = [[] for _ in range(len(table))]
    if not flags:
        return np.zeros(len(table), dtype=bool), reasons
    flag_map = _flag_name_to_bit(table)
    packed_flags = _packed_flags_array(table)
    values: list[np.ndarray] = []
    for name in flags:
        raw = _flag_values(table, name, flag_map=flag_map, packed_flags=packed_flags, strict=strict)
        if name == "detect_isPrimary":
            remove = ~raw
            reason = "B_non_primary"
        else:
            remove = raw
            reason = f"B_flag_{name}"
        values.append(remove)
        for idx in np.flatnonzero(remove):
            reasons[int(idx)].append(reason)
    stacked = np.stack(values, axis=0)
    if mode == "any":
        return stacked.any(axis=0), reasons
    if mode == "all":
        return stacked.all(axis=0), reasons
    raise ValueError("flag mode must be 'any' or 'all'")


def _removed_by_flags(
    table: Table,
    flags: Sequence[str],
    *,
    mode: str,
    flag_map: dict[str, int],
    packed_flags: Optional[np.ndarray],
    strict: bool,
) -> np.ndarray:
    if not flags:
        return np.zeros(len(table), dtype=bool)
    values = [
        _flag_values(table, name, flag_map=flag_map, packed_flags=packed_flags, strict=strict)
        for name in flags
    ]
    stacked = np.stack(values, axis=0)
    if mode == "any":
        return stacked.any(axis=0)
    if mode == "all":
        return stacked.all(axis=0)
    raise ValueError("flag mode must be 'any' or 'all'")


def _source_filter_mask(table: Table, source_filter: str) -> np.ndarray:
    n = len(table)
    if source_filter == "all":
        return np.ones(n, dtype=bool)
    parent = np.asarray(table["parent"], dtype=np.int64) if "parent" in table.colnames else np.zeros(n, dtype=np.int64)
    child_col = "deblend_nChild" if "deblend_nChild" in table.colnames else "nChild" if "nChild" in table.colnames else None
    if source_filter == "parent":
        return parent == 0
    if child_col is None:
        raise KeyError(f"source_filter={source_filter!r} needs deblend_nChild or nChild")
    leaf = np.asarray(table[child_col], dtype=np.int64) == 0
    if source_filter == "nchild0":
        return leaf
    if source_filter == "leaf_child":
        return leaf & (parent != 0)
    raise ValueError(f"Unknown source_filter: {source_filter}")


def _sdss_ellipse(table: Table) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx = _first_finite_column(table, ("base_SdssShape_xx", "ext_shapeHSM_HsmSourceMoments_xx"))
    yy = _first_finite_column(table, ("base_SdssShape_yy", "ext_shapeHSM_HsmSourceMoments_yy"))
    xy = _first_finite_column(table, ("base_SdssShape_xy", "ext_shapeHSM_HsmSourceMoments_xy"))
    major = np.full(len(table), np.nan, dtype=np.float64)
    minor = np.full(len(table), np.nan, dtype=np.float64)
    theta = np.full(len(table), np.nan, dtype=np.float64)
    valid = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(xy)
    xx = np.maximum(xx, 0.25)
    yy = np.maximum(yy, 0.25)
    trace = xx + yy
    delta = np.sqrt(np.maximum((xx - yy) ** 2 + 4.0 * xy**2, 0.0))
    major[valid] = np.sqrt(np.maximum(0.5 * (trace[valid] + delta[valid]), 0.25))
    minor[valid] = np.sqrt(np.maximum(0.5 * (trace[valid] - delta[valid]), 0.25))
    theta[valid] = 0.5 * np.arctan2(2.0 * xy[valid], xx[valid] - yy[valid])
    return major, minor, theta


def _kron_ellipse(
    table: Table,
    *,
    sigma: float,
    min_axis: float,
    require_refit_match: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sdss_major, sdss_minor, theta = _sdss_ellipse(table)
    kron_columns = ("pu_refit_kron_radius",) if require_refit_match else (
        "pu_refit_kron_radius",
        "ext_photometryKron_KronFlux_radius",
        "ext_photometryKron_KronFlux_radius_for_radius",
    )
    kron = _first_finite_column(
        table,
        kron_columns,
    )
    refit_matched = (
        np.asarray(table["pu_refit_kron_radius_matched"], dtype=bool)
        if "pu_refit_kron_radius_matched" in table.colnames
        else np.zeros(len(table), dtype=bool)
    )
    determinant_radius = np.sqrt(np.maximum(sdss_major * sdss_minor, 0.0))
    valid = (
        np.isfinite(kron)
        & (kron > 0)
        & np.isfinite(sdss_major)
        & np.isfinite(sdss_minor)
        & np.isfinite(determinant_radius)
        & (determinant_radius > 0)
        & np.isfinite(theta)
    )
    if require_refit_match:
        valid &= refit_matched
    a = np.full(len(table), np.nan, dtype=np.float64)
    b = np.full(len(table), np.nan, dtype=np.float64)
    angle = np.full(len(table), np.nan, dtype=np.float64)
    scale = np.zeros(len(table), dtype=np.float64)
    scale[valid] = kron[valid] / determinant_radius[valid]
    a[valid] = np.maximum(sdss_major[valid] * scale[valid] * float(sigma), float(min_axis))
    b[valid] = np.maximum(sdss_minor[valid] * scale[valid] * float(sigma), float(min_axis))
    angle[valid] = theta[valid]
    area = math.pi * a * b
    return a, b, angle, area


def _magnitude_from_flux(table: Table, *, column: str, zeropoint: float) -> np.ndarray:
    if column in table.colnames:
        flux = np.asarray(table[column], dtype=np.float64)
    else:
        names = ("base_PsfFlux_instFlux", "modelfit_CModel_instFlux", "base_SdssShape_instFlux")
        flux = _first_finite_column(table, names)
    mag = np.full(len(table), np.nan, dtype=np.float64)
    valid = np.isfinite(flux) & (flux > 0.0)
    mag[valid] = float(zeropoint) - 2.5 * np.log10(flux[valid])
    return mag


def _magnitude_from_exact_flux_column(table: Table, *, column: str, zeropoint: float) -> np.ndarray:
    mag = np.full(len(table), np.nan, dtype=np.float64)
    if column not in table.colnames:
        return mag
    flux = np.asarray(table[column], dtype=np.float64)
    valid = np.isfinite(flux) & (flux > 0.0)
    mag[valid] = float(zeropoint) - 2.5 * np.log10(flux[valid])
    return mag


def _sample_unit_disk(grid: int = 9) -> np.ndarray:
    values = np.linspace(-1.0, 1.0, int(grid))
    pts = [(u, v) for u in values for v in values if u * u + v * v <= 1.0]
    return np.asarray(pts, dtype=np.float64)


def _coverage(points_unit: np.ndarray, i: int, j: int, x: np.ndarray, y: np.ndarray, a: np.ndarray, b: np.ndarray, angle: np.ndarray) -> float:
    ci = math.cos(float(angle[i]))
    si = math.sin(float(angle[i]))
    pts_x = x[i] + ci * (a[i] * points_unit[:, 0]) - si * (b[i] * points_unit[:, 1])
    pts_y = y[i] + si * (a[i] * points_unit[:, 0]) + ci * (b[i] * points_unit[:, 1])

    cj = math.cos(float(angle[j]))
    sj = math.sin(float(angle[j]))
    dx = pts_x - x[j]
    dy = pts_y - y[j]
    xr = cj * dx + sj * dy
    yr = -sj * dx + cj * dy
    inside = (xr / max(float(a[j]), 1e-6)) ** 2 + (yr / max(float(b[j]), 1e-6)) ** 2 <= 1.0
    return float(np.count_nonzero(inside)) / float(max(len(points_unit), 1))


def _ellipse_iou(
    points_unit: np.ndarray,
    i: int,
    j: int,
    x: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    angle: np.ndarray,
    area: np.ndarray,
) -> float:
    cov_ij = _coverage(points_unit, i, j, x, y, a, b, angle)
    cov_ji = _coverage(points_unit, j, i, x, y, a, b, angle)
    inter = 0.5 * (cov_ij * float(area[i]) + cov_ji * float(area[j]))
    union = float(area[i]) + float(area[j]) - inter
    if union <= 0.0:
        return 0.0
    return float(np.clip(inter / union, 0.0, 1.0))


def _candidate_pairs(xy: np.ndarray, radius: float) -> Iterable[Tuple[int, int]]:
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(xy)
        yield from tree.query_pairs(float(radius), output_type="set")
        return
    except Exception:
        pass

    radius2 = float(radius) ** 2
    n = xy.shape[0]
    for i in range(n):
        delta = xy[i + 1 :] - xy[i]
        close = np.flatnonzero((delta[:, 0] ** 2 + delta[:, 1] ** 2) <= radius2)
        for item in close:
            yield i, int(i + 1 + item)


def classify_sources(table: Table, args: argparse.Namespace) -> dict[str, object]:
    x = _first_finite_column(table, POSITION_COLUMNS_X)
    y = _first_finite_column(table, POSITION_COLUMNS_Y)
    a, b, angle, kron_area = _kron_ellipse(
        table,
        sigma=args.region_sigma,
        min_axis=args.min_axis,
        require_refit_match=bool(getattr(args, "require_kron_refit_match", False)),
    )
    mag = _magnitude_from_flux(
        table,
        column=getattr(args, "mag_column", "base_PsfFlux_instFlux"),
        zeropoint=float(getattr(args, "input_zeropoint", 27.0)),
    )
    ids = np.asarray(table["id"], dtype=np.int64) if "id" in table.colnames else np.arange(len(table), dtype=np.int64)

    finite_center = np.isfinite(x) & np.isfinite(y)
    source_mask = _source_filter_mask(table, args.source_filter)
    valid_kron = np.isfinite(a) & np.isfinite(b) & np.isfinite(angle) & (a > 0) & (b > 0)
    refit_matched = (
        np.asarray(table["pu_refit_kron_radius_matched"], dtype=bool)
        if "pu_refit_kron_radius_matched" in table.colnames
        else np.zeros(len(table), dtype=bool)
    )
    base = finite_center & source_mask
    eligible = base & valid_kron

    dropped_area = eligible & np.isfinite(kron_area) & (kron_area > float(args.a_area_max))
    removed_a = eligible & (
        dropped_area
        | (
            np.isfinite(kron_area)
            & (kron_area > float(args.a_faint_area_max))
            & np.isfinite(mag)
            & (mag > float(args.a_faint_mag_min))
        )
    )
    a_failed = eligible & removed_a
    a_candidate = eligible & ~removed_a

    fill_ratio = np.full(len(table), np.nan, dtype=np.float64)
    center_only_by_fill = np.zeros(len(table), dtype=bool)
    fill_area_min = float(getattr(args, "center_only_fill_area_min", -1.0))
    fill_ratio_max = float(getattr(args, "center_only_fill_ratio_max", -1.0))
    if fill_area_min >= 0.0 and fill_ratio_max >= 0.0 and "pu_refit_aperture_pixel_count" in table.colnames:
        aperture_pixels = np.asarray(table["pu_refit_aperture_pixel_count"], dtype=np.float64)
        fill_valid = np.isfinite(aperture_pixels) & np.isfinite(kron_area) & (kron_area > 0.0)
        fill_ratio[fill_valid] = aperture_pixels[fill_valid] / kron_area[fill_valid]
        center_only_by_fill = (
            a_candidate
            & np.isfinite(kron_area)
            & (kron_area > fill_area_min)
            & np.isfinite(fill_ratio)
            & (fill_ratio < fill_ratio_max)
        )
    ap2_kron_abs_max = getattr(args, "ap2_kron_abs_max", None)
    hard_small_bright_area_reject = np.zeros(len(table), dtype=bool)
    ap2_kron_diff = np.full(len(table), np.nan, dtype=np.float64)
    ap2_kron_limit = np.full(len(table), np.nan, dtype=np.float64)
    bright_mag_threshold = float(getattr(args, "ap2_kron_bright_mag_threshold", 22.0))
    ap2_bright_source = np.isfinite(mag) & (mag < bright_mag_threshold)
    ap2_bright_region_center = np.zeros(len(table), dtype=bool)
    ap2_bright_region_area = np.full(len(table), np.nan, dtype=np.float64)
    small_bright_region = np.zeros(len(table), dtype=bool)
    large_bright_region = np.zeros(len(table), dtype=bool)
    outside_bright_region = np.zeros(len(table), dtype=bool)
    hard_small_bright_area_enabled = bool(getattr(args, "ap2_kron_small_bright_area_reject", True))
    hard_small_bright_area_abs_min = float(getattr(args, "ap2_kron_small_bright_area_abs_min", 1.0))
    hard_small_bright_area_ratio_max = float(getattr(args, "ap2_kron_small_bright_area_ratio_max", 1.0))
    need_ap2_kron_diff = (
        (ap2_kron_abs_max is not None and float(ap2_kron_abs_max) >= 0.0)
        or (hard_small_bright_area_enabled and hard_small_bright_area_abs_min >= 0.0)
    )
    if need_ap2_kron_diff:
        ap2_mag = _magnitude_from_exact_flux_column(
            table,
            column=getattr(args, "ap2_flux_column", "base_CircularApertureFlux_6_0_instFlux"),
            zeropoint=float(getattr(args, "input_zeropoint", 27.0)),
        )
        kron_mag = _magnitude_from_exact_flux_column(
            table,
            column=getattr(args, "ap2_kron_flux_column", "ext_photometryKron_KronFlux_instFlux"),
            zeropoint=float(getattr(args, "input_zeropoint", 27.0)),
        )
        ap2_kron_diff = ap2_mag - kron_mag
        if ap2_kron_abs_max is not None and float(ap2_kron_abs_max) >= 0.0:
            ap2_kron_limit[:] = float(ap2_kron_abs_max)
            bright_abs_max = float(getattr(args, "ap2_kron_bright_abs_max", float(ap2_kron_abs_max)))
            bright_region_column = str(getattr(args, "ap2_kron_bright_region_column", "pu_bright_region_center"))
            bright_region_area_column = str(
                getattr(args, "ap2_kron_bright_region_area_column", "pu_bright_region_component_area")
            )
            large_bright_region_area_min = float(getattr(args, "ap2_kron_large_bright_region_area_min", 1000.0))
            if bright_region_column in table.colnames:
                ap2_bright_region_center = np.asarray(table[bright_region_column], dtype=bool)
            if bright_region_area_column in table.colnames:
                ap2_bright_region_area = np.asarray(table[bright_region_area_column], dtype=np.float64)
            small_bright_region = ap2_bright_region_center & np.isfinite(ap2_bright_region_area) & (
                ap2_bright_region_area < large_bright_region_area_min
            )
            large_bright_region = ap2_bright_region_center & np.isfinite(ap2_bright_region_area) & (
                ap2_bright_region_area >= large_bright_region_area_min
            )
            outside_bright_region = ap2_bright_source & ~small_bright_region & ~large_bright_region
            ap2_kron_limit[ap2_bright_source & small_bright_region] = bright_abs_max
            ap2_kron_limit[ap2_bright_source & large_bright_region] = np.inf
        if hard_small_bright_area_enabled and hard_small_bright_area_abs_min >= 0.0:
            outside_limit = (
                float(ap2_kron_abs_max)
                if ap2_kron_abs_max is not None and float(ap2_kron_abs_max) >= 0.0
                else float(hard_small_bright_area_abs_min)
            )
            small_limit = float(getattr(args, "ap2_kron_bright_abs_max", max(2.0, outside_limit)))
            ap2_valid = np.isfinite(ap2_kron_diff)
            hard_small_bright_area_reject = (
                a_candidate
                & ap2_bright_source
                & (
                    (outside_bright_region & ap2_valid & (np.abs(ap2_kron_diff) >= outside_limit))
                    | (small_bright_region & ap2_valid & (np.abs(ap2_kron_diff) >= small_limit))
                )
            )
            center_only_by_fill &= ~hard_small_bright_area_reject

    b_candidate = a_candidate & ~center_only_by_fill
    ordinary_b_candidate = b_candidate & ~ap2_bright_source

    mag_in_range = np.isfinite(mag) & (mag >= float(args.b_mag_min)) & (mag <= float(args.b_mag_max))
    removed_b = (ordinary_b_candidate & ~mag_in_range) | hard_small_bright_area_reject
    axis_ratio_max = getattr(args, "b_axis_ratio_max", None)
    axis_ratio = np.full(len(table), np.nan, dtype=np.float64)
    if axis_ratio_max is not None and float(axis_ratio_max) > 0.0:
        axis_min = np.minimum(np.abs(a), np.abs(b))
        axis_max = np.maximum(np.abs(a), np.abs(b))
        axis_valid = np.isfinite(axis_min) & np.isfinite(axis_max) & (axis_min > 0.0)
        axis_ratio[axis_valid] = axis_max[axis_valid] / axis_min[axis_valid]
        removed_b |= b_candidate & axis_valid & (axis_ratio > float(axis_ratio_max))
    if ap2_kron_abs_max is not None and float(ap2_kron_abs_max) >= 0.0:
        ap2_kron_valid = np.isfinite(ap2_kron_diff) & (ap2_kron_limit > 0.0) & (
            np.abs(ap2_kron_diff) < ap2_kron_limit
        )
        removed_b |= ordinary_b_candidate & ~ap2_kron_valid
    if bool(getattr(args, "require_kron_refit_match", False)):
        removed_b |= b_candidate & ~refit_matched
    flag_removed, flag_reasons = _removed_by_b_flags(
        table,
        getattr(args, "b_flags", ()),
        mode=getattr(args, "b_mode", "any"),
        strict=bool(getattr(args, "strict_flags", False)),
    )
    removed_b |= ordinary_b_candidate & flag_removed
    reasons: List[List[str]] = [[] for _ in range(len(table))]
    for idx in np.flatnonzero(removed_a):
        if np.isfinite(kron_area[idx]) and kron_area[idx] > float(args.a_area_max):
            reasons[int(idx)].append("A_area_gt_max")
        elif np.isfinite(kron_area[idx]) and np.isfinite(mag[idx]):
            reasons[int(idx)].append("A_faint_large")
        else:
            reasons[int(idx)].append("A_failed")
    for idx in np.flatnonzero(center_only_by_fill):
        reasons[int(idx)].append(
            f"center_only_aperture_fill_lt_{fill_ratio_max:.2f}_area_gt_{fill_area_min:g}"
        )
    for idx in np.flatnonzero(removed_b):
        if hard_small_bright_area_reject[int(idx)]:
            if outside_bright_region[int(idx)]:
                reasons[int(idx)].append("B_outside_bright_region_abs_ap2_minus_kron_mag_ge_limit")
            elif small_bright_region[int(idx)]:
                reasons[int(idx)].append("B_small_bright_region_abs_ap2_minus_kron_mag_ge_limit")
            else:
                reasons[int(idx)].append("B_bright_region_ap2_kron_hard_reject")
        if ordinary_b_candidate[int(idx)] and not mag_in_range[int(idx)]:
            if not np.isfinite(mag[int(idx)]):
                reasons[int(idx)].append("B_mag_invalid")
            else:
                reasons[int(idx)].append("B_mag_outside_range")
        if bool(getattr(args, "require_kron_refit_match", False)) and base[int(idx)] and not refit_matched[int(idx)]:
            reasons[int(idx)].append("B_missing_proxy_kron_refit")
        if ordinary_b_candidate[int(idx)] and ap2_kron_abs_max is not None and float(ap2_kron_abs_max) >= 0.0:
            if not np.isfinite(ap2_kron_diff[int(idx)]):
                reasons[int(idx)].append("B_ap2_kron_mag_invalid")
            elif np.isfinite(ap2_kron_limit[int(idx)]) and abs(float(ap2_kron_diff[int(idx)])) >= float(ap2_kron_limit[int(idx)]):
                suffix = "_bright_region" if bool(ap2_bright_source[int(idx)] and ap2_bright_region_center[int(idx)]) else ""
                reasons[int(idx)].append(f"B_abs_ap2_minus_kron_mag_ge_{float(ap2_kron_limit[int(idx)]):.2f}{suffix}")
        if axis_ratio_max is not None and float(axis_ratio_max) > 0.0:
            if np.isfinite(axis_ratio[int(idx)]) and float(axis_ratio[int(idx)]) > float(axis_ratio_max):
                reasons[int(idx)].append(f"B_axis_ratio_gt_{float(axis_ratio_max):.2f}")
        reasons[int(idx)].extend(flag_reasons[int(idx)])

    close_pair_count = 0
    close_radius_px = float(args.b_close_center_arcsec) / max(float(args.pixel_scale_arcsec), 1e-6)
    close_indices = np.flatnonzero(b_candidate & ~removed_b)
    if close_indices.size:
        close_xy = np.stack([x[close_indices], y[close_indices]], axis=1)
        for li, lj in _candidate_pairs(close_xy, close_radius_px):
            i = int(close_indices[li])
            j = int(close_indices[lj])
            dist = math.hypot(float(x[i] - x[j]), float(y[i] - y[j]))
            if dist > close_radius_px:
                continue
            close_pair_count += 1
            mi = float(mag[i]) if np.isfinite(mag[i]) else float("inf")
            mj = float(mag[j]) if np.isfinite(mag[j]) else float("inf")
            if mi > mj:
                drop = i
            elif mj > mi:
                drop = j
            else:
                drop = i if float(kron_area[i]) >= float(kron_area[j]) else j
            removed_b[drop] = True
            reasons[int(drop)].append("B_close_center_fainter")

    containment_pair_count = 0
    containment_threshold = float(getattr(args, "containment_threshold", 0.80))
    containment_indices = np.flatnonzero(ordinary_b_candidate & ~removed_b)
    if containment_indices.size:
        local_xy = np.stack([x[containment_indices], y[containment_indices]], axis=1)
        unit_points = _sample_unit_disk(args.overlap_sample_grid)
        max_axis = float(np.nanmax(np.maximum(a[containment_indices], b[containment_indices]))) if containment_indices.size else 0.0
        search_radius = max(float(args.neighbor_radius), 2.0 * max_axis)
        for li, lj in _candidate_pairs(local_xy, search_radius):
            i = int(containment_indices[li])
            j = int(containment_indices[lj])
            if removed_b[i] or removed_b[j]:
                continue
            if not (np.isfinite(kron_area[i]) and np.isfinite(kron_area[j]) and kron_area[i] > 0 and kron_area[j] > 0):
                continue
            if kron_area[i] <= kron_area[j]:
                small, large = i, j
            else:
                small, large = j, i
            if _coverage(unit_points, small, large, x, y, a, b, angle) >= containment_threshold:
                containment_pair_count += 1
                removed_b[large] = True
                reasons[int(large)].append(f"B_large_contains_small_ge_{containment_threshold:.2f}")

    b_class = b_candidate & removed_b
    ab_class = b_candidate & ~removed_b

    overlap_pair_count = 0
    ambiguous = np.zeros(len(table), dtype=bool)
    if not bool(getattr(args, "keep_all_ab_clean", True)):
        a_indices = np.flatnonzero(ab_class)
        local_xy = np.stack([x[a_indices], y[a_indices]], axis=1)
        unit_points = _sample_unit_disk(args.overlap_sample_grid)
        for li, lj in _candidate_pairs(local_xy, args.neighbor_radius):
            i = int(a_indices[li])
            j = int(a_indices[lj])
            iou = _ellipse_iou(unit_points, i, j, x, y, a, b, angle, kron_area)
            if iou >= float(args.overlap_iou_threshold):
                overlap_pair_count += 1
                if args.ambiguous_mark == "both":
                    mark = (i, j)
                elif args.ambiguous_mark == "smaller":
                    mark = (i if kron_area[i] <= kron_area[j] else j,)
                else:
                    mark = (i, j)
                for idx in mark:
                    ambiguous[idx] = True
                    reasons[int(idx)].append(f"overlap_iou_ge_{float(args.overlap_iou_threshold):.2f}")

    invalid_kron_drop = base & ~valid_kron
    for idx in np.flatnonzero(invalid_kron_drop):
        reasons[int(idx)].append("invalid_kron")

    clean = ab_class & ~ambiguous
    center_only = center_only_by_fill.copy()
    if not bool(getattr(args, "keep_all_ab_clean", True)):
        center_only |= ab_class & ambiguous
    ignore = b_class | (a_failed & ~dropped_area) | invalid_kron_drop

    class_name = np.full(len(table), "outside_base", dtype=object)
    class_name[b_class] = "b_ignore"
    class_name[a_failed] = "a_failed"
    class_name[dropped_area] = "dropped_area"
    class_name[invalid_kron_drop] = "b_ignore"
    class_name[center_only] = "center_only"
    class_name[clean] = "clean"
    for idx in np.flatnonzero(b_class):
        reasons[int(idx)].append("removed_by_B")

    return {
        "ids": ids,
        "x": x,
        "y": y,
        "a": a,
        "b": b,
        "angle": angle,
        "area": kron_area,
        "mag": mag,
        "aperture_fill_ratio": fill_ratio,
        "base": base,
        "eligible": eligible,
        "dropped_large_ellipse": dropped_area,
        "dropped_by_a": a_failed,
        "dropped_area": dropped_area,
        "dropped_invalid_kron": invalid_kron_drop,
        "a_class": ab_class,
        "a_failed": a_failed,
        "b_class": b_class,
        "ignore": ignore,
        "clean": clean,
        "center_only": center_only,
        "center_only_by_aperture_fill": center_only_by_fill,
        "no_shape_supervision": center_only_by_fill,
        "small_bright_area_ap2_kron_reject": hard_small_bright_area_reject,
        "skip_remeasure_ap2_kron": hard_small_bright_area_reject,
        "ap2_kron_limit": ap2_kron_limit,
        "ap2_bright_source": ap2_bright_source,
        "ap2_bright_region_center": ap2_bright_region_center,
        "ap2_bright_region_area": ap2_bright_region_area,
        "ap2_small_bright_region": small_bright_region,
        "ap2_large_bright_region": large_bright_region,
        "ap2_outside_bright_region": outside_bright_region,
        "class_name": class_name,
        "reasons": [";".join(sorted(set(item))) for item in reasons],
        "overlap_pair_count": overlap_pair_count,
        "containment_pair_count": containment_pair_count,
        "close_pair_count": close_pair_count,
    }


def _region_line(row: dict[str, object], *, color: str, point: bool, point_size: int) -> str:
    x = float(row["x"]) + 1.0
    y = float(row["y"]) + 1.0
    if point:
        return f"point({x:.3f},{y:.3f}) # point=circle {int(point_size)} color={color}\n"
    return (
        f"ellipse({x:.3f},{y:.3f},{float(row['a']):.3f},{float(row['b']):.3f},"
        f"{math.degrees(float(row['angle'])):.2f}) # color={color}\n"
    )


def write_region(path: Path, rows: Sequence[dict[str, object]], *, point_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Region file format: DS9 version 4.1\n")
        handle.write(
            'global color=green dashlist=8 3 width=2 font="helvetica 14 bold roman" '
            "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n"
        )
        handle.write("image\n")
        for row in rows:
            handle.write(_region_line(row, color=str(row["color"]), point=bool(row["point"]), point_size=point_size))


def _rows_for_mask(result: dict[str, object], mask: np.ndarray, *, color: str, point: bool, large_point_color: str, args: argparse.Namespace) -> List[dict[str, object]]:
    x = result["x"]
    y = result["y"]
    a = result["a"]
    b = result["b"]
    angle = result["angle"]
    area = result["area"]
    rows: List[dict[str, object]] = []
    for idx in np.flatnonzero(mask):
        if not (np.isfinite(x[idx]) and np.isfinite(y[idx])):
            continue
        row_point = point or not (np.isfinite(a[idx]) and np.isfinite(b[idx]) and np.isfinite(angle[idx]))
        row_color = color
        if not row_point and np.isfinite(area[idx]) and area[idx] > float(args.large_ellipse_as_point):
            row_point = True
            row_color = large_point_color
        rows.append(
            {
                "x": float(x[idx]),
                "y": float(y[idx]),
                "a": float(a[idx]) if np.isfinite(a[idx]) else args.min_axis,
                "b": float(b[idx]) if np.isfinite(b[idx]) else args.min_axis,
                "angle": float(angle[idx]) if np.isfinite(angle[idx]) else 0.0,
                "color": row_color,
                "point": row_point,
            }
        )
    return rows


def write_outputs(table: Table, result: dict[str, object], args: argparse.Namespace) -> List[Path]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.catalog).stem

    clean = result["clean"]
    center_only = result["center_only"]
    b_class = result.get("ignore", result["b_class"])

    clean_rows = _rows_for_mask(result, clean, color="green", point=False, large_point_color="cyan", args=args)
    center_rows = _rows_for_mask(result, center_only, color="orange", point=True, large_point_color="orange", args=args)
    b_rows = _rows_for_mask(result, b_class, color="red", point=False, large_point_color="magenta", args=args)

    paths: List[Path] = []
    outputs = {
        f"{stem}_pu_clean_labels.reg": clean_rows,
        f"{stem}_pu_center_only.reg": center_rows,
        f"{stem}_pu_b_ignore.reg": b_rows,
        f"{stem}_pu_true_sources_clean_plus_center.reg": clean_rows + center_rows,
        f"{stem}_pu_all_classes.reg": clean_rows + center_rows + b_rows,
    }
    for name, rows in outputs.items():
        path = out_dir / name
        write_region(path, rows, point_size=args.point_size)
        paths.append(path)

    csv_path = out_dir / f"{stem}_pu_source_quality.csv"
    ids = result["ids"]
    x = result["x"]
    y = result["y"]
    a = result["a"]
    b = result["b"]
    area = result["area"]
    class_name = result["class_name"]
    reasons = result["reasons"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "row",
                "id",
                "x",
                "y",
                "mag",
                "class",
                "reason",
                "kron_a",
                "kron_b",
                "kron_area",
                "aperture_fill_ratio",
                "small_bright_area_ap2_kron_reject",
                "no_shape_supervision",
            ),
        )
        writer.writeheader()
        for idx in range(len(table)):
            writer.writerow(
                {
                    "row": idx,
                    "id": int(ids[idx]),
                    "x": float(x[idx]) if np.isfinite(x[idx]) else "",
                    "y": float(y[idx]) if np.isfinite(y[idx]) else "",
                    "mag": float(result["mag"][idx]) if np.isfinite(result["mag"][idx]) else "",
                    "class": str(class_name[idx]),
                    "reason": reasons[idx],
                    "kron_a": float(a[idx]) if np.isfinite(a[idx]) else "",
                    "kron_b": float(b[idx]) if np.isfinite(b[idx]) else "",
                    "kron_area": float(area[idx]) if np.isfinite(area[idx]) else "",
                    "aperture_fill_ratio": (
                        float(result["aperture_fill_ratio"][idx])
                        if np.isfinite(result["aperture_fill_ratio"][idx])
                        else ""
                    ),
                    "small_bright_area_ap2_kron_reject": bool(
                        result["small_bright_area_ap2_kron_reject"][idx]
                    ),
                    "no_shape_supervision": bool(result["no_shape_supervision"][idx]),
                }
            )
    paths.append(csv_path)
    return paths


def _split_flags(value: Optional[str], default: Sequence[str]) -> Tuple[str, ...]:
    if value is None:
        return tuple(default)
    if not value.strip():
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("/data1/czh23/Subaru/9813/HSC-I/0,0/meas-HSC-I-9813-0,0.fits"))
    parser.add_argument("--hdu", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("output/catalog_flag_filter_diagnostics/pu_filter"))
    parser.add_argument("--source-filter", choices=("all", "nchild0", "parent", "leaf_child"), default="all")
    parser.add_argument("--a-flags", type=str, default=None, help="Comma-separated A removal flags. Default matches notebook.")
    parser.add_argument("--b-flags", type=str, default=None, help="Comma-separated B removal flags. Default matches notebook.")
    parser.add_argument("--a-mode", choices=("any", "all"), default="any")
    parser.add_argument("--b-mode", choices=("any", "all"), default="any")
    parser.add_argument("--strict-flags", action="store_true")
    parser.add_argument(
        "--region-sigma",
        type=float,
        default=1.0,
        help="Scale applied to Kron/refit ellipses. Default 1.0 matches batch-heavyfp-kron-refit DS9 regions.",
    )
    parser.add_argument("--min-axis", type=float, default=1.5)
    parser.add_argument("--mag-column", default="base_PsfFlux_instFlux")
    parser.add_argument("--input-zeropoint", type=float, default=27.0)
    parser.add_argument(
        "--ap2-kron-abs-max",
        type=float,
        default=None,
        help="Optional B filter: require abs(ap2_mag-kron_mag) to be below this threshold.",
    )
    parser.add_argument("--ap2-flux-column", default="base_CircularApertureFlux_6_0_instFlux")
    parser.add_argument("--ap2-kron-flux-column", default="ext_photometryKron_KronFlux_instFlux")
    parser.add_argument(
        "--ap2-kron-small-bright-area-reject",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Hard-reject AP2/Kron outliers whose refit aperture_pixel_count/ellipse_area is below "
            "--ap2-kron-small-bright-area-ratio-max."
        ),
    )
    parser.add_argument("--ap2-kron-small-bright-area-ratio-max", type=float, default=1.0)
    parser.add_argument("--ap2-kron-small-bright-area-abs-min", type=float, default=1.0)
    parser.add_argument(
        "--kron-refit-csv",
        type=Path,
        default=None,
        help=(
            "Optional batch-heavyfp-kron-refit CSV. When provided, proxy Kron radii matched by source_id "
            "are preferred over ext_photometryKron_KronFlux_radius."
        ),
    )
    parser.add_argument("--kron-refit-radius-column", default="proxy_nan0_flux_aperture_radius")
    parser.add_argument("--kron-refit-good-column", default="proxy_nan0_good")
    parser.add_argument(
        "--require-kron-refit-match",
        action="store_true",
        default=True,
        help="When --kron-refit-csv is provided, drop rows without a good matched proxy Kron radius instead of falling back.",
    )
    parser.add_argument("--a-area-max", type=float, default=10000.0)
    parser.add_argument("--a-faint-area-max", type=float, default=900.0)
    parser.add_argument("--a-faint-mag-min", type=float, default=28.0)
    parser.add_argument("--center-only-fill-area-min", type=float, default=500.0)
    parser.add_argument("--center-only-fill-ratio-max", type=float, default=0.3)
    parser.add_argument("--b-mag-min", type=float, default=18.0)
    parser.add_argument("--b-mag-max", type=float, default=30.0)
    parser.add_argument("--pixel-scale-arcsec", type=float, default=0.168)
    parser.add_argument("--b-close-center-arcsec", type=float, default=0.5)
    parser.add_argument("--overlap-iou-threshold", type=float, default=0.33)
    parser.add_argument("--b-ellipse-area-max", type=float, default=None)
    parser.add_argument("--b-footprint-area-max", type=float, default=None)
    parser.add_argument(
        "--b-axis-ratio-max",
        type=float,
        default=5.0,
        help="Optional B filter: move sources with max(axis)/min(axis) above this value to ignore. Use <=0 to disable.",
    )
    parser.add_argument("--b-kron-radius-lt-sdss-major-ratio", type=float, default=0.75)
    parser.add_argument(
        "--drop-ellipse-area-min",
        type=float,
        default=40000.0,
        help="Drop sources with Kron ellipse area above this threshold before clean/center/ignore assignment.",
    )
    parser.add_argument("--ambiguous-area-max", type=float, default=30000.0)
    parser.add_argument("--neighbor-radius", type=float, default=80.0)
    parser.add_argument("--center-distance-factor", type=float, default=0.75)
    parser.add_argument("--containment-threshold", type=float, default=0.80)
    parser.add_argument("--mutual-overlap-threshold", type=float, default=0.35)
    parser.add_argument("--overlap-sample-grid", type=int, default=9)
    parser.add_argument("--ambiguous-mark", choices=("both", "smaller"), default="both")
    parser.add_argument(
        "--keep-all-ab-clean",
        action="store_true",
        default=True,
        help="Do not move overlapping A+B sources to center_only; keep every A+B source as clean.",
    )
    parser.add_argument("--large-ellipse-as-point", type=float, default=30000.0)
    parser.add_argument("--point-size", type=int, default=8)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.a_flags = _split_flags(args.a_flags, DEFAULT_A_FLAGS)
    args.b_flags = _split_flags(args.b_flags, DEFAULT_B_FLAGS)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        table = Table.read(args.catalog, hdu=args.hdu)
    if any("truncated" in str(item.message).lower() for item in caught):
        print("WARNING: FITS file may be truncated, but requested HDU was readable.", flush=True)

    table = attach_kron_refit_radius(
        table,
        args.kron_refit_csv,
        radius_column=args.kron_refit_radius_column,
        good_column=args.kron_refit_good_column,
    )
    result = classify_sources(table, args)
    paths = write_outputs(table, result, args)
    counts = Counter(np.asarray(result["class_name"], dtype=object))
    base_count = int(np.count_nonzero(result["base"]))
    eligible_count = int(np.count_nonzero(result["eligible"]))
    print(f"catalog={args.catalog}")
    print(f"rows={len(table)} base={base_count} eligible={eligible_count} source_filter={args.source_filter}")
    print(
        "clean={clean} center_only={center_only} b_ignore={b_ignore} "
        "dropped_large_ellipse={dropped} dropped_by_A={dropped_a} dropped_invalid_kron={dropped_kron} "
        "outside_base={outside}".format(
            clean=counts["clean"],
            center_only=counts["center_only"],
            b_ignore=counts["b_ignore"],
            dropped=counts["dropped_large_ellipse"],
            dropped_a=int(np.count_nonzero(result["dropped_by_a"])),
            dropped_kron=int(np.count_nonzero(result["dropped_invalid_kron"])),
            outside=counts["outside_base"],
        )
    )
    if base_count:
        clean_frac = counts["clean"] / base_count
        print(f"clean_fraction_of_base={clean_frac:.3%}")
        if clean_frac < 0.10:
            print("WARNING: clean labels are below 10% of the base population; prefer center-only warmup/PU training.")
    print(f"ambiguous_overlap_pairs={result['overlap_pair_count']}")
    print("outputs:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
