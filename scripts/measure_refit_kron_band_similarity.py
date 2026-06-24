#!/usr/bin/env python3
"""Measure cross-band similarity of refit Kron-aperture ellipses.

The script compares sources with the same source id between bands.  Pairwise
"distance" summaries are computed only for pairs whose two ellipses intersect;
non-intersecting common-id pairs are counted separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.table import Table


DEFAULT_BANDS = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y")


@dataclass(frozen=True)
class EllipseCatalog:
    band: str
    path: Path
    source_id: np.ndarray
    x: np.ndarray
    y: np.ndarray
    major: np.ndarray
    minor: np.ndarray
    theta: np.ndarray
    area: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare refit Kron aperture ellipses across bands. Metrics include "
            "sampled ellipse IoU, center distance, angle difference, and a "
            "shape-loss-like distance."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("/nvme0/zc/scarlet/preprocessed/9813"))
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patches", nargs="+", default=["4,5"])
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    parser.add_argument(
        "--catalog-subdir",
        default="band_reference_catalogs",
        help="Patch-level subdir containing per-band catalogs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/refit_kron_band_similarity_260613"),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=512,
        help="Number of deterministic samples per ellipse for approximate IoU.",
    )
    parser.add_argument("--angle-weight", type=float, default=4.0)
    parser.add_argument(
        "--min-intersection-area",
        type=float,
        default=0.0,
        help="Minimum sampled intersection area in pixel^2 for inclusion in distance stats.",
    )
    parser.add_argument("--max-ellipse-area", type=float, default=40000.0)
    parser.add_argument("--write-pair-details", action="store_true")
    return parser.parse_args()


def read_table(path: Path) -> Table:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Table.read(path)


def _as_float(table: Table, name: str) -> np.ndarray:
    values = np.asarray(table[name])
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    return np.asarray(values, dtype=np.float64)


def _choose_xy(table: Table) -> tuple[str, str]:
    for x_name, y_name in (
        ("base_SdssCentroid_x", "base_SdssCentroid_y"),
        ("base_SdssShape_x", "base_SdssShape_y"),
        ("slot_Centroid_x", "slot_Centroid_y"),
        ("base_NaiveCentroid_x", "base_NaiveCentroid_y"),
    ):
        if x_name in table.colnames and y_name in table.colnames:
            return x_name, y_name
    raise KeyError("catalog must contain a supported centroid column pair")


def _ellipse_from_moments(table: Table) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx = _as_float(table, "base_SdssShape_xx")
    yy = _as_float(table, "base_SdssShape_yy")
    xy = _as_float(table, "base_SdssShape_xy")
    xx = np.maximum(xx, 0.25)
    yy = np.maximum(yy, 0.25)
    trace = xx + yy
    delta = np.sqrt(np.maximum((xx - yy) ** 2 + 4.0 * xy**2, 0.0))
    major = np.sqrt(np.maximum(0.5 * (trace + delta), 0.25))
    minor = np.sqrt(np.maximum(0.5 * (trace - delta), 0.25))
    theta = 0.5 * np.arctan2(2.0 * xy, xx - yy)
    return major, minor, theta


def _fallback_kron_ellipse(table: Table) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sdss_major, sdss_minor, theta = _ellipse_from_moments(table)
    kron = None
    for name in (
        "pu_refit_kron_radius",
        "proxy_nan0_flux_aperture_radius",
        "proxy_flux_aperture_radius",
        "proxy_kron_radius",
        "ext_photometryKron_KronFlux_radius",
        "ext_photometryKron_KronFlux_radius_for_radius",
    ):
        if name in table.colnames:
            kron = _as_float(table, name)
            break
    if kron is None:
        return sdss_major, sdss_minor, theta
    determinant_radius = np.sqrt(np.maximum(sdss_major * sdss_minor, 0.0))
    valid = np.isfinite(kron) & (kron > 0) & np.isfinite(determinant_radius) & (determinant_radius > 0)
    scale = np.ones(len(table), dtype=np.float64)
    scale[valid] = kron[valid] / determinant_radius[valid]
    return sdss_major * scale, sdss_minor * scale, theta


def _normalize_axes(major: np.ndarray, minor: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    swap = minor > major
    major2 = major.copy()
    minor2 = minor.copy()
    theta2 = theta.copy()
    major2[swap], minor2[swap] = minor[swap], major[swap]
    theta2[swap] = theta2[swap] + 0.5 * math.pi
    theta2 = np.mod(theta2, math.pi)
    return major2, minor2, theta2


def load_catalog(root: Path, tract: str, patch: str, band: str, catalog_subdir: str, max_area: float) -> EllipseCatalog:
    path = root / patch / catalog_subdir / band / f"meas-{band}-{tract}-{patch}.fits"
    if not path.exists():
        raise FileNotFoundError(path)
    table = read_table(path)
    x_name, y_name = _choose_xy(table)
    x = _as_float(table, x_name)
    y = _as_float(table, y_name)
    if all(name in table.colnames for name in ("ellipse_major_sigma", "ellipse_minor_sigma", "ellipse_theta")):
        major = _as_float(table, "ellipse_major_sigma")
        minor = _as_float(table, "ellipse_minor_sigma")
        theta = _as_float(table, "ellipse_theta")
    else:
        major, minor, theta = _fallback_kron_ellipse(table)
    major, minor, theta = _normalize_axes(major, minor, theta)
    source_id = np.asarray(table["id"], dtype=np.int64)
    area = math.pi * major * minor
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(major)
        & np.isfinite(minor)
        & np.isfinite(theta)
        & (major > 0)
        & (minor > 0)
        & np.isfinite(area)
        & (area > 0)
        & (area <= float(max_area))
    )
    return EllipseCatalog(
        band=band,
        path=path,
        source_id=source_id[valid],
        x=x[valid],
        y=y[valid],
        major=major[valid],
        minor=minor[valid],
        theta=theta[valid],
        area=area[valid],
    )


def unit_disk_samples(n: int) -> np.ndarray:
    n = max(int(n), 32)
    k = np.arange(n, dtype=np.float64)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    r = np.sqrt((k + 0.5) / n)
    phi = k * golden_angle
    return np.column_stack([r * np.cos(phi), r * np.sin(phi)])


def coverage(points_unit: np.ndarray, src: tuple[float, float, float, float, float], dst: tuple[float, float, float, float, float]) -> float:
    x1, y1, a1, b1, t1 = src
    x2, y2, a2, b2, t2 = dst
    c1 = math.cos(t1)
    s1 = math.sin(t1)
    px = x1 + c1 * (a1 * points_unit[:, 0]) - s1 * (b1 * points_unit[:, 1])
    py = y1 + s1 * (a1 * points_unit[:, 0]) + c1 * (b1 * points_unit[:, 1])
    c2 = math.cos(t2)
    s2 = math.sin(t2)
    dx = px - x2
    dy = py - y2
    xr = c2 * dx + s2 * dy
    yr = -s2 * dx + c2 * dy
    inside = (xr / max(a2, 1e-6)) ** 2 + (yr / max(b2, 1e-6)) ** 2 <= 1.0
    return float(np.count_nonzero(inside)) / float(len(points_unit))


def intersection_and_iou(points_unit: np.ndarray, e1: tuple[float, float, float, float, float], area1: float, e2: tuple[float, float, float, float, float], area2: float) -> tuple[float, float]:
    # Cheap rejection: two ellipses cannot intersect if their centers are farther
    # apart than the sum of their semi-major axes.
    if math.hypot(e1[0] - e2[0], e1[1] - e2[1]) > e1[2] + e2[2]:
        return 0.0, 0.0
    cov12 = coverage(points_unit, e1, e2)
    cov21 = coverage(points_unit, e2, e1)
    inter = 0.5 * (cov12 * area1 + cov21 * area2)
    union = area1 + area2 - inter
    if union <= 0:
        return inter, 0.0
    return inter, float(np.clip(inter / union, 0.0, 1.0))


def periodic_angle_delta(theta1: float, theta2: float) -> float:
    # Ellipse position angles are pi-periodic.
    delta = (theta1 - theta2 + 0.5 * math.pi) % math.pi - 0.5 * math.pi
    return abs(delta)


def smooth_l1(value: float) -> float:
    value = abs(float(value))
    if value < 1.0:
        return 0.5 * value * value
    return value - 0.5


def shape_distance(e1: tuple[float, float, float, float, float], area1: float, e2: tuple[float, float, float, float, float], area2: float, angle_weight: float) -> float:
    a1, b1, t1 = e1[2], e1[3], e1[4]
    a2, b2, t2 = e2[2], e2[3], e2[4]
    area_loss = smooth_l1(math.log(max(area1 / math.pi, 1e-6)) - math.log(max(area2 / math.pi, 1e-6)))
    ratio_loss = smooth_l1(math.log(max(a1 / b1, 1e-6)) - math.log(max(a2 / b2, 1e-6)))
    axes_loss = ratio_loss + area_loss
    angle_loss = 1.0 - math.cos(2.0 * periodic_angle_delta(t1, t2))
    weight = max(float(angle_weight), 0.0)
    return float((2.0 * axes_loss + weight * angle_loss) / (2.0 + weight))


def summarize_values(values: Iterable[float], prefix: str) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_p10": math.nan,
            f"{prefix}_p90": math.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p10": float(np.percentile(arr, 10)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
    }


def align_by_source_id(cat_a: EllipseCatalog, cat_b: EllipseCatalog) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx_a = {int(source_id): idx for idx, source_id in enumerate(cat_a.source_id)}
    idx_b = {int(source_id): idx for idx, source_id in enumerate(cat_b.source_id)}
    common = np.asarray(sorted(set(idx_a).intersection(idx_b)), dtype=np.int64)
    a_indices = np.asarray([idx_a[int(source_id)] for source_id in common], dtype=np.int64)
    b_indices = np.asarray([idx_b[int(source_id)] for source_id in common], dtype=np.int64)
    return common, a_indices, b_indices


def compare_pair(
    patch: str,
    cat_a: EllipseCatalog,
    cat_b: EllipseCatalog,
    points_unit: np.ndarray,
    min_intersection_area: float,
    angle_weight: float,
    write_details: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    source_ids, idx_a, idx_b = align_by_source_id(cat_a, cat_b)
    details: list[dict[str, object]] = []
    ious: list[float] = []
    inter_areas: list[float] = []
    center_dist: list[float] = []
    center_dist_norm: list[float] = []
    angle_deg: list[float] = []
    angle_loss: list[float] = []
    major_frac_diff: list[float] = []
    minor_frac_diff: list[float] = []
    area_log_abs_diff: list[float] = []
    shape_losses: list[float] = []
    non_intersect = 0
    for source_id, ia, ib in zip(source_ids, idx_a, idx_b):
        e1 = (float(cat_a.x[ia]), float(cat_a.y[ia]), float(cat_a.major[ia]), float(cat_a.minor[ia]), float(cat_a.theta[ia]))
        e2 = (float(cat_b.x[ib]), float(cat_b.y[ib]), float(cat_b.major[ib]), float(cat_b.minor[ib]), float(cat_b.theta[ib]))
        area1 = float(cat_a.area[ia])
        area2 = float(cat_b.area[ib])
        inter, iou = intersection_and_iou(points_unit, e1, area1, e2, area2)
        if inter <= float(min_intersection_area):
            non_intersect += 1
            continue
        dist = math.hypot(e1[0] - e2[0], e1[1] - e2[1])
        norm = dist / max(math.sqrt(0.5 * (area1 + area2) / math.pi), 1e-6)
        delta = periodic_angle_delta(e1[4], e2[4])
        shape_loss = shape_distance(e1, area1, e2, area2, angle_weight)
        ious.append(iou)
        inter_areas.append(inter)
        center_dist.append(dist)
        center_dist_norm.append(norm)
        angle_deg.append(math.degrees(delta))
        angle_loss.append(1.0 - math.cos(2.0 * delta))
        major_frac_diff.append(abs(e1[2] - e2[2]) / max(0.5 * (e1[2] + e2[2]), 1e-6))
        minor_frac_diff.append(abs(e1[3] - e2[3]) / max(0.5 * (e1[3] + e2[3]), 1e-6))
        area_log_abs_diff.append(abs(math.log(max(area1, 1e-6)) - math.log(max(area2, 1e-6))))
        shape_losses.append(shape_loss)
        if write_details:
            details.append(
                {
                    "patch": patch,
                    "band_a": cat_a.band,
                    "band_b": cat_b.band,
                    "source_id": int(source_id),
                    "intersection_area": inter,
                    "iou": iou,
                    "center_distance_pix": dist,
                    "center_distance_norm": norm,
                    "angle_delta_deg": math.degrees(delta),
                    "angle_loss_1_minus_cos2dtheta": 1.0 - math.cos(2.0 * delta),
                    "major_frac_diff": major_frac_diff[-1],
                    "minor_frac_diff": minor_frac_diff[-1],
                    "area_log_abs_diff": area_log_abs_diff[-1],
                    "shape_distance": shape_loss,
                    "a_major": e1[2],
                    "a_minor": e1[3],
                    "a_theta_deg": math.degrees(e1[4]),
                    "b_major": e2[2],
                    "b_minor": e2[3],
                    "b_theta_deg": math.degrees(e2[4]),
                }
            )
    row: dict[str, object] = {
        "patch": patch,
        "band_a": cat_a.band,
        "band_b": cat_b.band,
        "n_a": int(len(cat_a.source_id)),
        "n_b": int(len(cat_b.source_id)),
        "n_common_valid": int(len(source_ids)),
        "n_intersect": int(len(ious)),
        "n_non_intersect": int(non_intersect),
        "intersect_fraction": float(len(ious) / len(source_ids)) if len(source_ids) else math.nan,
    }
    for name, values in (
        ("iou", ious),
        ("intersection_area", inter_areas),
        ("center_distance_pix", center_dist),
        ("center_distance_norm", center_dist_norm),
        ("angle_delta_deg", angle_deg),
        ("angle_loss", angle_loss),
        ("major_frac_diff", major_frac_diff),
        ("minor_frac_diff", minor_frac_diff),
        ("area_log_abs_diff", area_log_abs_diff),
        ("shape_distance", shape_losses),
    ):
        row.update(summarize_values(values, name))
    return row, details


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    points_unit = unit_disk_samples(args.samples)
    all_summary_rows: list[dict[str, object]] = []
    all_detail_rows: list[dict[str, object]] = []
    for patch in args.patches:
        catalogs = {
            band: load_catalog(args.root.expanduser(), args.tract, patch, band, args.catalog_subdir, args.max_ellipse_area)
            for band in args.bands
        }
        for band_a, band_b in combinations(args.bands, 2):
            row, details = compare_pair(
                patch,
                catalogs[band_a],
                catalogs[band_b],
                points_unit,
                args.min_intersection_area,
                args.angle_weight,
                args.write_pair_details,
            )
            all_summary_rows.append(row)
            all_detail_rows.extend(details)
    write_csv(out_dir / "refit_kron_band_pair_summary.csv", all_summary_rows)
    if args.write_pair_details:
        write_csv(out_dir / "refit_kron_band_pair_details.csv", all_detail_rows)
    config = {
        "root": str(args.root),
        "tract": args.tract,
        "patches": args.patches,
        "bands": args.bands,
        "catalog_subdir": args.catalog_subdir,
        "samples": int(args.samples),
        "angle_weight": float(args.angle_weight),
        "min_intersection_area": float(args.min_intersection_area),
        "max_ellipse_area": float(args.max_ellipse_area),
    }
    payload = {"config": config, "summary": all_summary_rows}
    (out_dir / "refit_kron_band_pair_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    print(json.dumps(payload, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
