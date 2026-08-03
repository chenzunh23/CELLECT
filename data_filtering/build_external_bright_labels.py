#!/usr/bin/env python3
"""Build diagnostic external bright-source labels using Gaia and HSC masks."""

from __future__ import annotations

import argparse
import csv
import math
import os
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cellect")

try:
    import matplotlib.pyplot as plt
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.table import Table
    from astropy.units import UnitsWarning
    from astropy import units as u
    from astropy.visualization import ZScaleInterval, make_lupton_rgb
    from astropy.wcs import WCS
    from scipy import ndimage
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires astropy, scipy, and matplotlib.") from exc

warnings.filterwarnings("ignore", category=UnitsWarning)
warnings.filterwarnings("ignore", message="Warning: converting a masked element to nan.")

from data_filtering.sam_input_scaling import current_sam_zscore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/data/shared/Subaru"))
    parser.add_argument("--refit-root", type=Path, default=Path("/data/czh23/refit"))
    parser.add_argument("--preprocessed-root", type=Path, default=Path("/data/czh23/preprocessed"))
    parser.add_argument("--gaia-fits", type=Path, default=Path("output/gaia_dr3_cosmos.fits"))
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patch", default="4,5")
    parser.add_argument("--bands", nargs="+", default=["HSC-I", "HSC-Y"])
    parser.add_argument("--out-dir", type=Path, default=Path("output/data_filter_0728/external_bright_labels"))
    parser.add_argument("--bright-mag-threshold", type=float, default=22.0)
    parser.add_argument("--zeropoint", type=float, default=27.0)
    parser.add_argument("--match-radius-arcsec", type=float, default=1.0)
    parser.add_argument("--shape-max-area", type=float, default=10000.0)
    parser.add_argument("--shape-axis-ratio-max", type=float, default=5.0)
    parser.add_argument("--log-a", type=float, default=300.0)
    parser.add_argument("--log-high-percentile", type=float, default=99.5)
    parser.add_argument("--lupton-stretch", type=float, default=0.5)
    parser.add_argument("--lupton-q", type=float, default=20.0)
    parser.add_argument("--bright-z-threshold", type=float, default=2.99)
    parser.add_argument("--bright-mask-dilate", type=int, default=2)
    parser.add_argument("--component-search-radius", type=int, default=5)
    parser.add_argument("--mask-names", nargs="+", default=["SAT", "BAD", "EDGE"])
    parser.add_argument("--stellar-mask-names", nargs="+", default=["BAD", "NO_DATA", "EDGE", "UNMASKEDNAN"])
    parser.add_argument("--cluster-iou-threshold", type=float, default=1.0 / 3.0)
    parser.add_argument("--cluster-max-center-distance", type=float, default=50.0)
    parser.add_argument("--cluster-max-area", type=float, default=10000.0)
    parser.add_argument("--cluster-centroid-match-pixels", type=float, default=3.0)
    parser.add_argument("--make-png", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def finite_float(value: object, default: float = float("nan")) -> float:
    if np.ma.is_masked(value):
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "t", "yes", "y"}


def mag_from_flux(flux: float, zeropoint: float) -> float:
    if not math.isfinite(flux) or flux <= 0.0:
        return float("nan")
    return float(zeropoint) - 2.5 * math.log10(float(flux))


def image_path(data_root: Path, tract: str, band: str, patch: str) -> Path:
    return data_root / str(tract) / band / patch / f"calexp-{band}-{tract}-{patch}.fits"


def meas_path(data_root: Path, tract: str, band: str, patch: str) -> Path:
    return data_root / str(tract) / band / patch / f"meas-{band}-{tract}-{patch}.fits"


def refit_path(refit_root: Path, tract: str, band: str, patch: str) -> Path:
    return refit_root / str(tract) / band / patch / "batch_heavyfp_kron_refit" / "batch_heavyfp_kron_refit.csv"


def label_catalog_path(preprocessed_root: Path, tract: str, patch: str, label_dir: str, band: str) -> Path:
    return preprocessed_root / str(tract) / patch / label_dir / band / f"meas-{band}-{tract}-{patch}.fits"


def read_exposure(path: Path) -> tuple[np.ndarray, np.ndarray, fits.Header, fits.Header]:
    with fits.open(path, memmap=True) as hdul:
        image = np.asarray(hdul[1].data, dtype=np.float32)
        mask = np.asarray(hdul[2].data)
        return image, mask, hdul[1].header.copy(), hdul[2].header.copy()


def read_id_set(path: Path) -> set[int]:
    if not path.exists():
        return set()
    table = Table.read(path, hdu=1)
    if "id" in table.colnames:
        col = "id"
    elif "source_id" in table.colnames:
        col = "source_id"
    else:
        return set()
    return {int(value) for value in np.asarray(table[col])}


def load_label_sets(preprocessed_root: Path, tract: str, patch: str, band: str) -> dict[str, set[int]]:
    mapping = {
        "clean": "band_reference_catalogs",
        "center_only": "band_reference_center_only",
        "strict_center_only": "band_reference_strict_center_only",
        "ignore": "band_reference_ignore",
        "strict_ignore": "band_reference_strict_ignore",
        "rejected": "band_reference_rejected",
    }
    return {
        label: read_id_set(label_catalog_path(preprocessed_root, tract, patch, directory, band))
        for label, directory in mapping.items()
    }


def classify_existing_label(source_id: int, labels: dict[str, set[int]]) -> str:
    for label in ("clean", "center_only", "strict_center_only", "ignore", "strict_ignore", "rejected"):
        if source_id in labels.get(label, set()):
            return label
    return "other"


def source_class_from_extendedness(value: float) -> str:
    if not math.isfinite(value):
        return "unknown"
    return "star" if value <= 0.5 else "galaxy"


def load_bright_hsc_sources(
    *,
    meas: Table,
    refit_csv: Path,
    labels: dict[str, set[int]],
    mag_threshold: float,
    zeropoint: float,
) -> list[dict[str, object]]:
    if "base_ClassificationExtendedness_value" not in meas.colnames:
        raise KeyError("meas catalog lacks base_ClassificationExtendedness_value")
    rows: list[dict[str, object]] = []
    with refit_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            flux = finite_float(row.get("catalog_KronFlux_instFlux"))
            mag = mag_from_flux(flux, zeropoint)
            if not math.isfinite(mag) or mag >= mag_threshold:
                continue
            if row.get("status", "") != "ok":
                continue
            if "proxy_nan0_good" in row and not is_true(row.get("proxy_nan0_good")):
                continue
            row_index = int(round(finite_float(row.get("row_index"), -1.0)))
            if row_index < 0 or row_index >= len(meas):
                continue
            source_id = int(round(finite_float(row.get("source_id"), -1.0)))
            x = finite_float(row.get("x_image"))
            y = finite_float(row.get("y_image"))
            axis_a = finite_float(row.get("axis_a"))
            axis_b = finite_float(row.get("axis_b"))
            theta_deg = finite_float(row.get("theta_deg"))
            initial_radius = finite_float(row.get("initial_determinant_radius"))
            target_radius = finite_float(row.get("proxy_nan0_flux_aperture_radius"))
            if not all(math.isfinite(v) for v in (x, y, axis_a, axis_b, theta_deg, initial_radius, target_radius)):
                continue
            if axis_a <= 0 or axis_b <= 0 or initial_radius <= 0 or target_radius <= 0:
                continue
            scale = target_radius / initial_radius
            major = axis_a * scale
            minor = axis_b * scale
            ext = finite_float(meas["base_ClassificationExtendedness_value"][row_index])
            existing_label = classify_existing_label(source_id, labels)
            rows.append(
                {
                    "source_id": source_id,
                    "row_index": row_index,
                    "x": x,
                    "y": y,
                    "major": major,
                    "minor": minor,
                    "theta_deg": theta_deg,
                    "area": math.pi * major * minor,
                    "axis_ratio": max(major, minor) / max(min(major, minor), 1e-6),
                    "mag": mag,
                    "class": source_class_from_extendedness(ext),
                    "classification_extendedness": ext,
                    "existing_label": existing_label,
                    "measurement_surface": row.get("measurement_surface", ""),
                    "final_label": "",
                    "reason": "",
                    "component_id": 0,
                    "cluster_id": 0,
                    "cluster_size": 0,
                    "cluster_is_stellar_mask": False,
                    "gaia_source_id": "",
                    "gaia_g_mag": "",
                    "gaia_match_arcsec": "",
                    "gaia_match_pixels": "",
                    "gaia_match_mode": "",
                    "output_x": x,
                    "output_y": y,
                }
            )
    return rows


def finite_values(image: np.ndarray) -> np.ndarray:
    vals = np.asarray(image, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("image has no finite pixels")
    return vals.astype(np.float64, copy=False)


def standardize_clip(arr: np.ndarray) -> np.ndarray:
    vals = finite_values(arr)
    mean = float(np.median(vals))
    std = float(np.std(vals))
    if not math.isfinite(std) or std <= 0:
        std = 1.0
    z = (np.nan_to_num(arr, nan=mean, posinf=mean, neginf=mean).astype(np.float32) - mean) / std
    return np.clip(z, -3.0, 3.0).astype(np.float32)


def log_mapping(image: np.ndarray, *, a: float, high_percentile: float) -> np.ndarray:
    vals = finite_values(image)
    minimum = float(np.min(vals))
    high = float(np.percentile(vals, high_percentile))
    if not math.isfinite(high) or high <= minimum:
        high = float(np.max(vals))
    if not math.isfinite(high) or high <= minimum:
        high = minimum + 1.0
    safe = np.nan_to_num(image, nan=minimum, posinf=high, neginf=minimum).astype(np.float32)
    x = np.clip((safe - minimum) / (high - minimum), 0.0, 1.0)
    return (np.log1p(float(a) * x) / np.log(float(a))).astype(np.float32)


def lupton_mapping(image: np.ndarray, *, stretch: float, q: float) -> np.ndarray:
    _z, stats = current_sam_zscore(image)
    minimum = float(stats.get("zscore_median", stats["median"]))
    rgb = make_lupton_rgb(image, image, image, minimum=minimum, stretch=float(stretch), Q=float(q), output_dtype=float)
    return np.asarray(rgb[..., 0], dtype=np.float32)


def build_bright_components(
    image: np.ndarray,
    *,
    log_a: float,
    log_high_percentile: float,
    lupton_stretch: float,
    lupton_q: float,
    threshold: float,
    dilation: int,
) -> tuple[np.ndarray, np.ndarray]:
    log_z = standardize_clip(log_mapping(image, a=log_a, high_percentile=log_high_percentile))
    lupton_z = standardize_clip(lupton_mapping(image, stretch=lupton_stretch, q=lupton_q))
    bright = (log_z >= float(threshold)) & (lupton_z >= float(threshold))
    if dilation > 0:
        bright = ndimage.binary_dilation(bright, iterations=int(dilation))
    labels, _num = ndimage.label(bright)
    return bright.astype(np.uint8), labels.astype(np.int32)


def component_at(labels: np.ndarray, x: float, y: float, search_radius: int) -> int:
    xi = int(round(x))
    yi = int(round(y))
    if 0 <= xi < labels.shape[1] and 0 <= yi < labels.shape[0]:
        value = int(labels[yi, xi])
        if value > 0:
            return value
    r = int(search_radius)
    if r <= 0:
        return 0
    x0 = max(0, xi - r)
    x1 = min(labels.shape[1], xi + r + 1)
    y0 = max(0, yi - r)
    y1 = min(labels.shape[0], yi + r + 1)
    window = labels[y0:y1, x0:x1]
    values = window[window > 0]
    if values.size == 0:
        return 0
    return int(np.bincount(values.ravel()).argmax())


def mask_bits(header: fits.Header, names: list[str]) -> dict[str, int]:
    return {name: int(header[f"MP_{name}"]) for name in names if f"MP_{name}" in header}


def center_in_mask(mask: np.ndarray, bits: dict[str, int], x: float, y: float) -> tuple[bool, str]:
    xi = int(round(x))
    yi = int(round(y))
    if not (0 <= xi < mask.shape[1] and 0 <= yi < mask.shape[0]):
        return False, ""
    value = int(mask[yi, xi])
    names = [name for name, bit in bits.items() if value & (1 << int(bit))]
    return bool(names), ",".join(names)


def ellipse_contains(row: dict[str, object], xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    x0 = float(row["x"])
    y0 = float(row["y"])
    major = max(float(row["major"]), 1e-6)
    minor = max(float(row["minor"]), 1e-6)
    theta = math.radians(float(row["theta_deg"]))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = xs - x0
    dy = ys - y0
    xr = dx * cos_t + dy * sin_t
    yr = -dx * sin_t + dy * cos_t
    return (xr / major) ** 2 + (yr / minor) ** 2 <= 1.0


def approximate_ellipse_iou(a: dict[str, object], b: dict[str, object]) -> float:
    max_a = max(float(a["major"]), float(a["minor"]))
    max_b = max(float(b["major"]), float(b["minor"]))
    if math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"])) > max_a + max_b:
        return 0.0
    x0 = math.floor(min(float(a["x"]) - max_a, float(b["x"]) - max_b))
    x1 = math.ceil(max(float(a["x"]) + max_a, float(b["x"]) + max_b))
    y0 = math.floor(min(float(a["y"]) - max_a, float(b["y"]) - max_b))
    y1 = math.ceil(max(float(a["y"]) + max_a, float(b["y"]) + max_b))
    if x1 < x0 or y1 < y0:
        return 0.0
    # Keep pathological huge ellipses cheap; this is only a clustering diagnostic.
    step = max(1, int(math.ceil(max(x1 - x0 + 1, y1 - y0 + 1) / 512.0)))
    yy, xx = np.mgrid[y0 : y1 + 1 : step, x0 : x1 + 1 : step]
    ma = ellipse_contains(a, xx.astype(np.float32), yy.astype(np.float32))
    mb = ellipse_contains(b, xx.astype(np.float32), yy.astype(np.float32))
    union = int(np.count_nonzero(ma | mb))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(ma & mb) / union)


class UnionFind:
    def __init__(self, items: list[int]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_component_sources(
    indices: list[int],
    sources: list[dict[str, object]],
    *,
    iou_threshold: float,
    max_center_distance: float,
    max_area: float,
) -> list[list[int]]:
    if not indices:
        return []
    uf = UnionFind(indices)
    sorted_indices = sorted(indices, key=lambda idx: float(sources[idx]["x"]))
    area = {idx: max(float(sources[idx]["area"]), 1e-6) for idx in sorted_indices}
    for pos, idx_a in enumerate(sorted_indices):
        src_a = sources[idx_a]
        xa = float(src_a["x"])
        ya = float(src_a["y"])
        if area[idx_a] >= float(max_area):
            continue
        for idx_b in sorted_indices[pos + 1 :]:
            dx = float(sources[idx_b]["x"]) - xa
            if dx > float(max_center_distance):
                break
            if area[idx_b] >= float(max_area):
                continue
            dy = float(sources[idx_b]["y"]) - ya
            if dx * dx + dy * dy > float(max_center_distance) ** 2:
                continue
            if min(area[idx_a], area[idx_b]) / max(area[idx_a], area[idx_b]) < float(iou_threshold):
                continue
            if approximate_ellipse_iou(src_a, sources[idx_b]) >= float(iou_threshold):
                uf.union(idx_a, idx_b)
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in indices:
        groups[uf.find(idx)].append(idx)
    return sorted(groups.values(), key=lambda group: min(group))


def load_gaia_for_patch(gaia_path: Path, wcs: WCS, shape: tuple[int, int]) -> list[dict[str, object]]:
    table = Table.read(gaia_path)
    coords = SkyCoord(ra=np.asarray(table["ra"], dtype=float) * u.deg, dec=np.asarray(table["dec"], dtype=float) * u.deg)
    x, y = wcs.world_to_pixel(coords)
    good = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
    rows: list[dict[str, object]] = []
    for idx in np.nonzero(good)[0]:
        gmag = finite_float(table["phot_g_mean_mag"][idx])
        rows.append(
            {
                "source_id": int(table["source_id"][idx]),
                "ra": float(table["ra"][idx]),
                "dec": float(table["dec"][idx]),
                "phot_g_mean_mag": gmag,
                "x": float(x[idx]),
                "y": float(y[idx]),
            }
        )
    return rows


def nearest_gaia(source: dict[str, object], gaia_rows: list[dict[str, object]]) -> tuple[dict[str, object] | None, float]:
    if not gaia_rows:
        return None, float("inf")
    sx = float(source["x"])
    sy = float(source["y"])
    best = None
    best_dist = float("inf")
    for row in gaia_rows:
        dist_pix = math.hypot(float(row["x"]) - sx, float(row["y"]) - sy)
        if dist_pix < best_dist:
            best = row
            best_dist = dist_pix
    return best, best_dist * 0.168


def nearest_source_index_to_gaia(
    gaia: dict[str, object],
    indices: list[int],
    sources: list[dict[str, object]],
) -> tuple[int | None, float]:
    best_idx: int | None = None
    best_dist = float("inf")
    gx = float(gaia["x"])
    gy = float(gaia["y"])
    for idx in indices:
        source = sources[idx]
        dist_pix = math.hypot(float(source["x"]) - gx, float(source["y"]) - gy)
        if dist_pix < best_dist:
            best_idx = idx
            best_dist = dist_pix
    return best_idx, best_dist * 0.168


def brightest_gaia(gaia_rows: list[dict[str, object]]) -> dict[str, object] | None:
    if not gaia_rows:
        return None
    return min(
        gaia_rows,
        key=lambda row: finite_float(row.get("phot_g_mean_mag"), float("inf")),
    )


def nearest_gaia_to_cluster(
    indices: list[int],
    sources: list[dict[str, object]],
    gaia_rows: list[dict[str, object]],
    *,
    match_radius_arcsec: float,
    centroid_match_pixels: float,
    source_match_pixels: float | None = None,
) -> tuple[dict[str, object] | None, int | None, float, float, str]:
    if not indices:
        return None, None, float("inf"), float("inf"), ""
    centroid_x = float(np.mean([float(sources[idx]["x"]) for idx in indices]))
    centroid_y = float(np.mean([float(sources[idx]["y"]) for idx in indices]))
    candidates: list[tuple[float, float, float, dict[str, object], int, str]] = []
    for gaia in gaia_rows:
        source_idx, dist_arcsec = nearest_source_index_to_gaia(gaia, indices, sources)
        if source_idx is None:
            continue
        source_dist_pix = dist_arcsec / 0.168
        centroid_dist_pix = math.hypot(float(gaia["x"]) - centroid_x, float(gaia["y"]) - centroid_y)
        match_mode = ""
        if source_match_pixels is not None and source_dist_pix <= float(source_match_pixels):
            match_mode = "source_center"
            matched_dist_pix = source_dist_pix
        elif source_match_pixels is None and dist_arcsec <= float(match_radius_arcsec):
            match_mode = "source_center"
            matched_dist_pix = source_dist_pix
        elif centroid_dist_pix <= float(centroid_match_pixels):
            match_mode = "cluster_centroid"
            matched_dist_pix = centroid_dist_pix
        else:
            continue
        gmag = finite_float(gaia.get("phot_g_mean_mag"), float("inf"))
        candidates.append((gmag, dist_arcsec, matched_dist_pix, gaia, source_idx, match_mode))
    if not candidates:
        return None, None, float("inf"), float("inf"), ""
    _gmag, dist_arcsec, dist_pix, gaia, source_idx, match_mode = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    return gaia, source_idx, dist_arcsec, dist_pix, match_mode


def shape_ok(source: dict[str, object], *, max_area: float, max_axis_ratio: float) -> bool:
    return float(source["area"]) <= float(max_area) and float(source["axis_ratio"]) <= float(max_axis_ratio)


def assign_labels(
    *,
    sources: list[dict[str, object]],
    gaia_rows: list[dict[str, object]],
    component_labels: np.ndarray,
    mask: np.ndarray,
    mask_header: fits.Header,
    mask_names: list[str],
    stellar_mask_names: list[str],
    component_search_radius: int,
    match_radius_arcsec: float,
    cluster_iou_threshold: float,
    cluster_max_center_distance: float,
    cluster_max_area: float,
    cluster_centroid_match_pixels: float,
    shape_max_area: float,
    shape_axis_ratio_max: float,
) -> tuple[list[dict[str, object]], dict[int, dict[str, object]], list[dict[str, object]]]:
    bits = mask_bits(mask_header, mask_names)
    stellar_bits = mask_bits(mask_header, stellar_mask_names)
    by_component: dict[int, list[int]] = defaultdict(list)
    gaia_by_component: dict[int, list[dict[str, object]]] = defaultdict(list)
    for idx, source in enumerate(sources):
        comp = component_at(component_labels, float(source["x"]), float(source["y"]), component_search_radius)
        source["component_id"] = comp
        by_component[comp].append(idx)
    for gaia in gaia_rows:
        comp = component_at(component_labels, float(gaia["x"]), float(gaia["y"]), component_search_radius)
        if comp > 0:
            gaia_by_component[comp].append(gaia)

    component_meta: dict[int, dict[str, object]] = {}
    for comp, indices in by_component.items():
        if comp <= 0:
            continue
        component_meta[comp] = {
            "component_id": comp,
            "hsc_source_count": len(indices),
            "gaia_count": len(gaia_by_component.get(comp, [])),
        }

    pending_by_component: dict[int, list[int]] = defaultdict(list)
    for idx, source in enumerate(sources):
        existing = str(source["existing_label"])
        if existing in {"clean", "center_only", "strict_center_only"}:
            source["final_label"] = existing
            source["reason"] = "existing_supervised"
            continue
        comp = int(source["component_id"])
        in_bad_mask, mask_hit = center_in_mask(mask, bits, float(source["x"]), float(source["y"]))
        if comp <= 0 and in_bad_mask:
            source["final_label"] = "ignore"
            source["reason"] = f"outside_bright_center_in_{mask_hit}"
            continue
        if comp <= 0:
            source["final_label"] = "ignore"
            source["reason"] = "not_in_restricted_bright_region"
            continue
        pending_by_component[comp].append(idx)

    for comp, pending_indices in list(pending_by_component.items()):
        supervised = [
            idx
            for idx in by_component.get(comp, [])
            if str(sources[idx]["existing_label"]) in {"clean", "center_only", "strict_center_only"}
        ]
        if not supervised:
            continue
        kept = []
        for idx in pending_indices:
            overlaps_supervised = any(
                approximate_ellipse_iou(sources[idx], sources[sidx]) >= float(cluster_iou_threshold)
                for sidx in supervised
            )
            if overlaps_supervised:
                sources[idx]["final_label"] = "ignore"
                sources[idx]["reason"] = "overlaps_existing_supervised_cluster"
            else:
                kept.append(idx)
        pending_by_component[comp] = kept

    cluster_rows: list[dict[str, object]] = []
    next_cluster_id = 1
    for comp in sorted(pending_by_component):
        component_clusters = cluster_component_sources(
            pending_by_component[comp],
            sources,
            iou_threshold=float(cluster_iou_threshold),
            max_center_distance=float(cluster_max_center_distance),
            max_area=float(cluster_max_area),
        )
        component_meta.setdefault(comp, {})["cluster_count"] = len(component_clusters)
        component_meta.setdefault(comp, {})["candidate_source_count"] = len(pending_by_component[comp])
        for cluster in component_clusters:
            cluster_id = next_cluster_id
            next_cluster_id += 1
            stellar_hits = []
            for idx in cluster:
                hit, hit_names = center_in_mask(mask, stellar_bits, float(sources[idx]["x"]), float(sources[idx]["y"]))
                if hit:
                    stellar_hits.append(hit_names)
            cluster_is_stellar_mask = bool(stellar_hits)
            for idx in cluster:
                sources[idx]["cluster_id"] = cluster_id
                sources[idx]["cluster_size"] = len(cluster)
                sources[idx]["cluster_is_stellar_mask"] = cluster_is_stellar_mask
            chosen_idx: int | None = None
            chosen_gaia: dict[str, object] | None = None
            chosen_dist = float("inf")
            chosen_dist_pix = float("inf")
            chosen_match_mode = ""
            if cluster_is_stellar_mask:
                chosen_idx = min(cluster, key=lambda idx: float(sources[idx]["mag"]))
            else:
                chosen_gaia, chosen_idx, chosen_dist, chosen_dist_pix, chosen_match_mode = nearest_gaia_to_cluster(
                    cluster,
                    sources,
                    gaia_by_component.get(comp, []),
                    match_radius_arcsec=float(match_radius_arcsec),
                    centroid_match_pixels=float(cluster_centroid_match_pixels),
                )
                if chosen_idx is None:
                    chosen_idx = min(cluster, key=lambda idx: float(sources[idx]["mag"]))

            for idx in cluster:
                source = sources[idx]
                if idx == chosen_idx:
                    if chosen_gaia is not None:
                        source["gaia_source_id"] = chosen_gaia["source_id"]
                        source["gaia_g_mag"] = chosen_gaia["phot_g_mean_mag"]
                        source["gaia_match_arcsec"] = chosen_dist
                        source["gaia_match_pixels"] = chosen_dist_pix
                        source["gaia_match_mode"] = chosen_match_mode
                        source["output_x"] = chosen_gaia["x"]
                        source["output_y"] = chosen_gaia["y"]
                    if cluster_is_stellar_mask:
                        source["final_label"] = "strict_center_only_external"
                        source["reason"] = "cluster_center_in_stellar_mask_no_shape"
                    elif chosen_gaia is not None and str(source["class"]) == "galaxy" and shape_ok(
                        source, max_area=shape_max_area, max_axis_ratio=shape_axis_ratio_max
                    ):
                        source["final_label"] = "galaxy_shape_diagnostic"
                        source["reason"] = "cluster_gaia_matched_hsc_galaxy_shape_ok"
                    elif chosen_gaia is not None:
                        source["final_label"] = "strict_center_only_external"
                        source["reason"] = "cluster_gaia_matched_star_or_uncertain_no_shape"
                    else:
                        source["final_label"] = "dropped_bright_cluster"
                        source["reason"] = "cluster_no_gaia_chosen_brightest"
                else:
                    source["final_label"] = "restricted_bright_region"
                    if cluster_is_stellar_mask:
                        source["reason"] = "stellar_mask_cluster_member_not_chosen"
                    elif chosen_gaia is not None:
                        source["reason"] = "same_cluster_has_gaia_but_not_chosen"
                    else:
                        source["reason"] = "same_cluster_no_gaia_not_chosen"
            cluster_rows.append(
                {
                    "cluster_id": cluster_id,
                    "component_id": comp,
                    "cluster_size": len(cluster),
                    "cluster_is_stellar_mask": cluster_is_stellar_mask,
                    "chosen_source_id": sources[chosen_idx]["source_id"] if chosen_idx is not None else "",
                    "chosen_final_label": sources[chosen_idx]["final_label"] if chosen_idx is not None else "",
                    "gaia_source_id": chosen_gaia["source_id"] if chosen_gaia is not None else "",
                    "gaia_g_mag": chosen_gaia["phot_g_mean_mag"] if chosen_gaia is not None else "",
                    "gaia_match_arcsec": chosen_dist if chosen_gaia is not None else "",
                    "gaia_match_pixels": chosen_dist_pix if chosen_gaia is not None else "",
                    "gaia_match_mode": chosen_match_mode,
                    "source_ids": " ".join(str(sources[idx]["source_id"]) for idx in cluster),
                    "stellar_mask_hits": ";".join(sorted(set(stellar_hits))),
                }
            )
    return sources, component_meta, cluster_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "source_id",
        "row_index",
        "x",
        "y",
        "output_x",
        "output_y",
        "major",
        "minor",
        "theta_deg",
        "area",
        "axis_ratio",
        "mag",
        "class",
        "classification_extendedness",
        "existing_label",
        "final_label",
        "reason",
        "component_id",
        "cluster_id",
        "cluster_size",
        "cluster_is_stellar_mask",
        "gaia_source_id",
        "gaia_g_mag",
        "gaia_match_arcsec",
        "gaia_match_pixels",
        "gaia_match_mode",
        "measurement_surface",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def ellipse_line(row: dict[str, object], color: str, *, use_output_center: bool = False, text: str = "") -> str:
    x = float(row["output_x"] if use_output_center else row["x"])
    y = float(row["output_y"] if use_output_center else row["y"])
    if text:
        suffix = f" # color={color} text={{{text}}}\n"
    else:
        suffix = f" # color={color}\n"
    return (
        f"ellipse({x:.3f},{y:.3f},{float(row['major']):.3f},"
        f"{float(row['minor']):.3f},{float(row['theta_deg']):.3f}){suffix}"
    )


def point_line(row: dict[str, object], color: str, *, text: str = "") -> str:
    suffix = f" # point=circle color={color}"
    if text:
        suffix += f" text={{{text}}}"
    return f"point({float(row['output_x']):.3f},{float(row['output_y']):.3f}){suffix}\n"


def write_regions(out_dir: Path, stem: str, rows: list[dict[str, object]]) -> None:
    groups = {
        "strict_center_only_external": ("cyan", True, "point"),
        "restricted_bright_region": ("orange", False, "point"),
        "galaxy_shape_diagnostic": ("magenta", False, "ellipse"),
        "dropped_bright_cluster": ("red", False, "point"),
        "ignore": ("yellow", False, "point"),
    }
    for final_label, (color, output_center, kind) in groups.items():
        selected = [row for row in rows if row.get("final_label") == final_label]
        path = out_dir / f"{stem}_{final_label}.reg"
        with path.open("w", encoding="utf-8") as handle:
            handle.write("# Region file format: DS9 version 4.1\n")
            handle.write('global color=cyan dashlist=8 3 width=2 font="helvetica 10 normal roman"\n')
            handle.write("image\n")
            for row in selected:
                text = f"{row.get('class')} {row.get('reason')} mag={float(row['mag']):.2f}"
                if kind == "ellipse":
                    handle.write(ellipse_line(row, color, use_output_center=output_center, text=text))
                else:
                    handle.write(point_line(row, color, text=text))


def write_nonstellar_i_shapes(out_dir: Path, stem: str, rows: list[dict[str, object]], image: np.ndarray, dpi: int) -> None:
    gaia_strict_clusters = {
        int(row.get("cluster_id", 0))
        for row in rows
        if int(row.get("cluster_id", 0)) > 0
        and row.get("final_label") == "strict_center_only_external"
        and str(row.get("gaia_source_id", "")).strip()
    }
    selected = [
        row
        for row in rows
        if int(row.get("component_id", 0)) > 0
        and int(row.get("cluster_id", 0)) > 0
        and int(row.get("cluster_id", 0)) not in gaia_strict_clusters
        and not bool(row.get("cluster_is_stellar_mask", False))
        and row.get("existing_label") not in {"clean", "center_only", "strict_center_only"}
    ]
    reg_path = out_dir / f"{stem}_nonstellar_I_all_cluster_shapes.reg"
    with reg_path.open("w", encoding="utf-8") as handle:
        handle.write("# Region file format: DS9 version 4.1\n")
        handle.write('global color=magenta dashlist=8 3 width=2 font="helvetica 10 normal roman"\n')
        handle.write("image\n")
        for row in selected:
            text = (
                f"cluster={row.get('cluster_id')} final={row.get('final_label')} "
                f"class={row.get('class')} mag={float(row['mag']):.2f}"
            )
            handle.write(ellipse_line(row, "magenta", text=text))

    if not selected:
        return
    finite = image[np.isfinite(image)]
    vmin, vmax = ZScaleInterval().get_limits(finite)
    display = np.nan_to_num(image, nan=vmin, posinf=vmax, neginf=vmin)
    fig, ax = plt.subplots(figsize=(11, 11), constrained_layout=True)
    ax.imshow(display, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    from matplotlib.patches import Ellipse

    cluster_ids = sorted({int(row["cluster_id"]) for row in selected})
    cmap = plt.get_cmap("tab20", max(len(cluster_ids), 1))
    color_by_cluster = {cluster_id: cmap(pos % cmap.N) for pos, cluster_id in enumerate(cluster_ids)}
    for row in sorted(selected, key=lambda item: float(item["area"]), reverse=True):
        cluster_id = int(row["cluster_id"])
        color = color_by_cluster[cluster_id]
        patch = Ellipse(
            (float(row["x"]), float(row["y"])),
            width=2.0 * float(row["major"]),
            height=2.0 * float(row["minor"]),
            angle=float(row["theta_deg"]),
            fill=False,
            edgecolor=color,
            linewidth=0.7,
            alpha=0.9,
        )
        ax.add_patch(patch)
        ax.plot(float(row["x"]), float(row["y"]), marker="+", color=color, markersize=3, markeredgewidth=0.7)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.set_xlabel("image x")
    ax.set_ylabel("image y")
    ax.set_title(f"{stem}: non-stellar I cluster shapes (n={len(selected)}, clusters={len(cluster_ids)})")
    fig.savefig(out_dir / f"{stem}_nonstellar_I_all_cluster_shapes.png", dpi=dpi)
    plt.close(fig)


def write_component_products(
    out_dir: Path,
    stem: str,
    bright_mask: np.ndarray,
    component_labels: np.ndarray,
    sources: list[dict[str, object]],
    image_header: fits.Header,
) -> None:
    restricted_components = {
        int(row["component_id"])
        for row in sources
        if row.get("final_label") in {"strict_center_only_external", "restricted_bright_region", "dropped_bright_cluster", "galaxy_shape_diagnostic"}
        and int(row.get("component_id", 0)) > 0
    }
    restricted = np.isin(component_labels, list(restricted_components)).astype(np.uint8)
    fits.writeto(out_dir / f"{stem}_restricted_bright_region_mask.fits", restricted, header=image_header, overwrite=True)
    fits.writeto(out_dir / f"{stem}_log_lupton_intersection_mask.fits", bright_mask.astype(np.uint8), header=image_header, overwrite=True)
    reg_path = out_dir / f"{stem}_restricted_bright_region_bbox.reg"
    with reg_path.open("w", encoding="utf-8") as handle:
        handle.write("# Region file format: DS9 version 4.1\n")
        handle.write('global color=orange dashlist=8 3 width=2 font="helvetica 10 normal roman"\n')
        handle.write("image\n")
        for comp in sorted(restricted_components):
            ys, xs = np.nonzero(component_labels == comp)
            if xs.size == 0:
                continue
            x0 = float(xs.min())
            x1 = float(xs.max())
            y0 = float(ys.min())
            y1 = float(ys.max())
            handle.write(
                f"box({(x0+x1)/2:.3f},{(y0+y1)/2:.3f},{x1-x0+1:.3f},{y1-y0+1:.3f},0) "
                f"# color=orange text={{component={comp}}}\n"
            )


def write_gaia_coverage(out_dir: Path, stem: str, gaia_rows: list[dict[str, object]], shape: tuple[int, int], tile_size: int = 512) -> None:
    rows: list[dict[str, object]] = []
    for y0 in range(0, shape[0], tile_size):
        for x0 in range(0, shape[1], tile_size):
            x1 = min(shape[1], x0 + tile_size)
            y1 = min(shape[0], y0 + tile_size)
            count = sum(1 for row in gaia_rows if x0 <= float(row["x"]) < x1 and y0 <= float(row["y"]) < y1)
            rows.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "gaia_count": count})
    write_csv_generic(out_dir / f"{stem}_gaia_512_counts.csv", rows)


def write_csv_generic(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_overlay_png(path: Path, image: np.ndarray, rows: list[dict[str, object]], dpi: int) -> None:
    finite = image[np.isfinite(image)]
    vmin, vmax = ZScaleInterval().get_limits(finite)
    display = np.nan_to_num(image, nan=vmin, posinf=vmax, neginf=vmin)
    colors = {
        "strict_center_only_external": "#00d7ff",
        "restricted_bright_region": "#ff9d00",
        "galaxy_shape_diagnostic": "#ff3ecf",
        "dropped_bright_cluster": "#ff2a2a",
        "ignore": "#ffe45c",
        "clean": "#52d273",
        "center_only": "#8fd3ff",
        "strict_center_only": "#8fd3ff",
    }
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.imshow(display, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        label = str(row["final_label"])
        counts[label] += 1
        color = colors.get(label, "white")
        if label == "galaxy_shape_diagnostic":
            from matplotlib.patches import Ellipse

            patch = Ellipse(
                (float(row["x"]), float(row["y"])),
                width=2.0 * float(row["major"]),
                height=2.0 * float(row["minor"]),
                angle=float(row["theta_deg"]),
                fill=False,
                edgecolor=color,
                linewidth=0.7,
                alpha=0.9,
            )
            ax.add_patch(patch)
        else:
            x = float(row.get("output_x", row["x"]))
            y = float(row.get("output_y", row["y"]))
            ax.plot(x, y, marker="+", markersize=4, markeredgewidth=0.8, color=color)
    for label, color in colors.items():
        if counts[label] > 0:
            ax.plot([], [], color=color, label=f"{label} n={counts[label]}")
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.set_xlabel("image x")
    ax.set_ylabel("image y")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax.set_title(path.stem)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def process_band(args: argparse.Namespace, band: str) -> dict[str, object]:
    out_dir = args.out_dir / str(args.tract) / args.patch / band
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"processing {args.tract}/{args.patch} {band}", flush=True)
    image, mask, image_header, mask_header = read_exposure(image_path(args.data_root, args.tract, band, args.patch))
    meas = Table.read(meas_path(args.data_root, args.tract, band, args.patch), hdu=1)
    labels = load_label_sets(args.preprocessed_root, args.tract, args.patch, band)
    sources = load_bright_hsc_sources(
        meas=meas,
        refit_csv=refit_path(args.refit_root, args.tract, band, args.patch),
        labels=labels,
        mag_threshold=float(args.bright_mag_threshold),
        zeropoint=float(args.zeropoint),
    )
    wcs = WCS(image_header)
    gaia_rows = load_gaia_for_patch(args.gaia_fits, wcs, image.shape)
    bright_mask, component_labels = build_bright_components(
        image,
        log_a=float(args.log_a),
        log_high_percentile=float(args.log_high_percentile),
        lupton_stretch=float(args.lupton_stretch),
        lupton_q=float(args.lupton_q),
        threshold=float(args.bright_z_threshold),
        dilation=int(args.bright_mask_dilate),
    )
    print(f"{band}: loaded {len(sources)} bright HSC sources; bright components={int(component_labels.max())}", flush=True)
    assigned, component_meta, cluster_rows = assign_labels(
        sources=sources,
        gaia_rows=gaia_rows,
        component_labels=component_labels,
        mask=mask,
        mask_header=mask_header,
        mask_names=list(args.mask_names),
        stellar_mask_names=list(args.stellar_mask_names),
        component_search_radius=int(args.component_search_radius),
        match_radius_arcsec=float(args.match_radius_arcsec),
        cluster_iou_threshold=float(args.cluster_iou_threshold),
        cluster_max_center_distance=float(args.cluster_max_center_distance),
        cluster_max_area=float(args.cluster_max_area),
        cluster_centroid_match_pixels=float(args.cluster_centroid_match_pixels),
        shape_max_area=float(args.shape_max_area),
        shape_axis_ratio_max=float(args.shape_axis_ratio_max),
    )
    print(f"{band}: assigned {len(cluster_rows)} bright-source clusters", flush=True)
    stem = f"{args.tract}_{args.patch.replace(',', '_')}_{band}"
    write_csv(out_dir / f"{stem}_bright_reclassification.csv", assigned)
    write_regions(out_dir, stem, assigned)
    write_nonstellar_i_shapes(out_dir, stem, assigned, image, int(args.dpi))
    write_component_products(out_dir, stem, bright_mask, component_labels, assigned, image_header)
    write_gaia_coverage(out_dir, stem, gaia_rows, image.shape)
    write_csv_generic(out_dir / f"{stem}_component_summary.csv", list(component_meta.values()))
    write_csv_generic(out_dir / f"{stem}_cluster_summary.csv", cluster_rows)
    if bool(args.make_png):
        write_overlay_png(out_dir / f"{stem}_bright_reclassification_overlay.png", image, assigned, int(args.dpi))
    counts: dict[str, int] = defaultdict(int)
    for row in assigned:
        counts[str(row["final_label"])] += 1
    return {
        "band": band,
        "bright_hsc_sources": len(assigned),
        "gaia_patch_sources": len(gaia_rows),
        "bright_components": int(component_labels.max()),
        **counts,
    }


def main() -> int:
    args = parse_args()
    summaries = [process_band(args, band) for band in args.bands]
    write_csv_generic(args.out_dir / str(args.tract) / args.patch / "summary.csv", summaries)
    for summary in summaries:
        print(summary)
    print(f"wrote {args.out_dir / str(args.tract) / args.patch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
