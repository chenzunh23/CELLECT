#!/usr/bin/env python3
"""Build v2 external bright-source diagnostics and priority partitions."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cellect")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import build_external_bright_labels as base
from data_filtering import external_bright_labels_core as bright_core
from data_filtering.sam_input_scaling import build_bright_mask

try:
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.table import Table
    from astropy.visualization import ZScaleInterval
    from astropy.wcs import WCS
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Ellipse, Patch
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires astropy and matplotlib.") from exc


CENTER_LABELS = {
    "center_only",
    "strict_center_only",
    "center_only_external",
    "strict_center_only_external",
    "strict_center_only_added",
}
SUPERVISED_LABELS = {"clean", *CENTER_LABELS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/data/shared/Subaru"))
    parser.add_argument("--refit-root", type=Path, default=Path("/data/czh23/refit"))
    parser.add_argument("--preprocessed-root", type=Path, default=Path("/data/czh23/preprocessed"))
    parser.add_argument("--gaia-fits", type=Path, default=Path("output/gaia_dr3_cosmos.fits"))
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patches", nargs="+", default=["4,5", "6,1"])
    parser.add_argument("--bands", nargs="+", default=["HSC-I", "HSC-Y"])
    parser.add_argument("--out-dir", type=Path, default=Path("output/data_filter_0729/external_bright_labels_v2"))
    parser.add_argument(
        "--stage-filter-root",
        type=Path,
        default=Path("output/data_filter_0729/bright_filter_stage_diagnostics/A_B_refined"),
        help="Optional A/B refined diagnostics root; when present, only remaining bright sources enter this stage.",
    )
    parser.add_argument("--bright-mag-threshold", type=float, default=22.0)
    parser.add_argument("--gaia-bright-mag-threshold", type=float, default=18.0)
    parser.add_argument("--zeropoint", type=float, default=27.0)
    parser.add_argument("--match-radius-arcsec", type=float, default=1.0)
    parser.add_argument("--shape-max-area", type=float, default=10000.0)
    parser.add_argument(
        "--drop-area-max",
        type=float,
        default=10000.0,
        help="Sources with refit Kron aperture area >= this value are rejected before bright-source relabeling.",
    )
    parser.add_argument("--shape-axis-ratio-max", type=float, default=5.0)
    parser.add_argument("--log-a", type=float, default=300.0)
    parser.add_argument("--log-high-percentile", type=float, default=99.5)
    parser.add_argument("--lupton-stretch", type=float, default=0.5)
    parser.add_argument("--lupton-q", type=float, default=20.0)
    parser.add_argument(
        "--bright-mask-mode",
        default="log-lupton",
        choices=("log-lupton", "zscore", "zscore-no-upper", "zscore-unbounded", "anscombe", "raw", "none"),
        help=(
            "Bright-region scaling mode. log-lupton preserves the legacy path; zscore is the clipped-zscore "
            "reference with z>=threshold components; zscore-no-upper does not threshold image values and instead "
            "uses source clusters plus Gaia; anscombe uses standardized Anscombe >= threshold."
        ),
    )
    parser.add_argument("--bright-z-threshold", type=float, default=2.99)
    parser.add_argument("--bright-mask-dilate", type=int, default=2)
    parser.add_argument("--bright-anscombe-scale", type=float, default=1000.0)
    parser.add_argument("--component-search-radius", type=int, default=5)
    parser.add_argument("--mask-names", nargs="+", default=["SAT", "BAD", "EDGE"])
    parser.add_argument(
        "--use-bad-mask-first-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, non-galaxy sources centered in SAT/BAD/EDGE without a bright Gaia source go directly to ignore.",
    )
    parser.add_argument(
        "--use-bright-gaia-component-override",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, a bright component containing a bright Gaia star is handled as one bright-star region before HSC-fragment clustering.",
    )
    parser.add_argument(
        "--add-empty-large-bright-component-centers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, add a strict center-only label at the geometric center of large bright components that have no clean/center label.",
    )
    parser.add_argument("--empty-large-bright-component-area-min", type=float, default=1000.0)
    parser.add_argument("--cluster-iou-threshold", type=float, default=1.0 / 3.0)
    parser.add_argument("--cluster-max-center-distance", type=float, default=50.0)
    parser.add_argument("--cluster-max-area", type=float, default=10000.0)
    parser.add_argument("--cluster-source-match-pixels", type=float, default=6.0)
    parser.add_argument("--cluster-centroid-match-pixels", type=float, default=10.0)
    parser.add_argument(
        "--large-component-fast-center-only-source-min",
        type=int,
        default=256,
        help="For bright components with at least this many HSC candidate sources and no bright Gaia override, skip pairwise HSC clustering; fragments become restricted and the component can receive a geometric strict center.",
    )
    parser.add_argument("--external-center-radius", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def component_area_map(labels: np.ndarray) -> dict[int, int]:
    counts = np.bincount(labels.ravel())
    return {idx: int(value) for idx, value in enumerate(counts) if idx > 0 and value > 0}


def component_centroid_map(labels: np.ndarray) -> dict[int, tuple[float, float]]:
    flat = labels.ravel()
    if flat.size == 0:
        return {}
    yy, xx = np.indices(labels.shape, dtype=np.float64)
    counts = np.bincount(flat)
    sum_x = np.bincount(flat, weights=xx.ravel())
    sum_y = np.bincount(flat, weights=yy.ravel())
    out: dict[int, tuple[float, float]] = {}
    for idx in range(1, len(counts)):
        if counts[idx] > 0:
            out[idx] = (float(sum_x[idx] / counts[idx]), float(sum_y[idx] / counts[idx]))
    return out


def build_bright_components_v2(image: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    mode = str(args.bright_mask_mode).strip().lower().replace("_", "-")
    if mode == "log-lupton":
        return base.build_bright_components(
            image,
            log_a=float(args.log_a),
            log_high_percentile=float(args.log_high_percentile),
            lupton_stretch=float(args.lupton_stretch),
            lupton_q=float(args.lupton_q),
            threshold=float(args.bright_z_threshold),
            dilation=int(args.bright_mask_dilate),
        )
    bright = build_bright_mask(
        image,
        mode=mode,
        threshold=float(args.bright_z_threshold),
        dilation=int(args.bright_mask_dilate),
        log_a=float(args.log_a),
        log_high_percentile=float(args.log_high_percentile),
        lupton_stretch=float(args.lupton_stretch),
        lupton_q=float(args.lupton_q),
        anscombe_scale=float(args.bright_anscombe_scale),
    )
    labels, _num = base.ndimage.label(np.asarray(bright, dtype=bool))
    return np.asarray(bright, dtype=np.uint8), labels.astype(np.int32)


def component_has_bright_gaia(
    comp: int,
    gaia_by_component: dict[int, list[dict[str, object]]],
    *,
    gaia_bright_mag_threshold: float,
) -> bool:
    for row in gaia_by_component.get(comp, []):
        gmag = base.finite_float(row.get("phot_g_mean_mag"), float("inf"))
        if math.isfinite(gmag) and gmag <= float(gaia_bright_mag_threshold):
            return True
    return False


def source_shape_usable(source: dict[str, object], *, max_area: float, max_axis_ratio: float) -> bool:
    return float(source["area"]) < float(max_area) and float(source["axis_ratio"]) <= float(max_axis_ratio)


def det_path(data_root: Path, tract: str, band: str, patch: str) -> Path:
    return data_root / str(tract) / band / patch / f"det-{band}-{tract}-{patch}.fits"


def read_det_background_mask(path: Path, shape_yx: tuple[int, int], origin_xy: tuple[int, int]) -> np.ndarray:
    """Return True for pixels outside LSST detection footprints."""
    with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
        if len(hdul) <= 4 or hdul[4].data is None:
            raise ValueError(f"{path} does not contain LSST detection footprint spans in HDU 4")
        spans = hdul[4].data
        rows = [(int(row["y"]), int(row["x0"]), int(row["x1"])) for row in spans]

    def paint(*, subtract_origin: bool) -> tuple[np.ndarray, int]:
        footprint = np.zeros(shape_yx, dtype=bool)
        painted = 0
        ox, oy = origin_xy if subtract_origin else (0, 0)
        for raw_y, raw_x0, raw_x1 in rows:
            y = raw_y - int(oy)
            if y < 0 or y >= shape_yx[0]:
                continue
            x0 = max(0, raw_x0 - int(ox))
            x1 = min(shape_yx[1] - 1, raw_x1 - int(ox))
            if x1 >= x0:
                footprint[y, x0 : x1 + 1] = True
                painted += x1 - x0 + 1
        return footprint, painted

    footprint, painted = paint(subtract_origin=False)
    if painted == 0 and origin_xy != (0, 0):
        footprint, _painted = paint(subtract_origin=True)
    return ~footprint


def source_center_in_bad_mask(
    source: dict[str, object],
    mask: np.ndarray,
    mask_bits: dict[str, int],
) -> tuple[bool, str]:
    return base.center_in_mask(mask, mask_bits, float(source["x"]), float(source["y"]))


def stage_filter_path(root: Path, tract: str, patch: str, band: str) -> Path:
    return root / str(tract) / patch / band / f"{tract}_{patch.replace(',', '_')}_{band}_bright_filter_stages.csv"


def load_stage_rows(root: Path, tract: str, patch: str, band: str) -> list[dict[str, str]] | None:
    path = stage_filter_path(root, tract, patch, band)
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_stage_remaining_ids(root: Path, tract: str, patch: str, band: str) -> set[int] | None:
    rows = load_stage_rows(root, tract, patch, band)
    if rows is None:
        return None
    keep_prefixes = (
        "remaining_",
    )
    keep: set[int] = set()
    for row in rows:
        status = str(row.get("status", ""))
        if status.startswith(keep_prefixes):
            try:
                keep.add(int(row["source_id"]))
            except Exception:
                continue
    return keep


def source_id(source: dict[str, object]) -> int:
    return int(source["source_id"])


def meas_index_by_source_id(meas: Table) -> dict[int, int]:
    if "id" not in meas.colnames:
        return {}
    out: dict[int, int] = {}
    for idx, sid in enumerate(meas["id"]):
        try:
            out[int(sid)] = idx
        except Exception:
            continue
    return out


def load_sources_from_stage_rows(
    *,
    rows: list[dict[str, str]],
    meas: Table,
    labels: dict[str, set[int]],
) -> list[dict[str, object]]:
    """Use the A/B refined diagnostic CSV as the authoritative bright-source list."""
    id_to_index = meas_index_by_source_id(meas)
    sources: list[dict[str, object]] = []
    for row in rows:
        if not str(row.get("status", "")).startswith("remaining_"):
            continue
        sid = int(round(base.finite_float(row.get("source_id"), -1.0)))
        if sid < 0:
            continue
        row_index = id_to_index.get(sid, -1)
        ext = float("nan")
        if row_index >= 0 and "base_ClassificationExtendedness_value" in meas.colnames:
            ext = base.finite_float(meas["base_ClassificationExtendedness_value"][row_index])
        x = base.finite_float(row.get("x"))
        y = base.finite_float(row.get("y"))
        major = base.finite_float(row.get("major"))
        minor = base.finite_float(row.get("minor"))
        theta_rad = base.finite_float(row.get("theta_rad"))
        area = base.finite_float(row.get("area"), math.pi * major * minor)
        mag = base.finite_float(row.get("mag"))
        if not all(math.isfinite(v) for v in (x, y, major, minor, theta_rad, area, mag)):
            continue
        if major <= 0 or minor <= 0:
            continue
        existing_label = base.classify_existing_label(sid, labels)
        sources.append(
            {
                "source_id": sid,
                "row_index": row_index,
                "x": x,
                "y": y,
                "major": major,
                "minor": minor,
                "theta_deg": math.degrees(theta_rad),
                "area": area,
                "axis_ratio": base.finite_float(row.get("axis_ratio"), max(major, minor) / max(min(major, minor), 1e-6)),
                "mag": mag,
                "class": base.source_class_from_extendedness(ext),
                "classification_extendedness": ext,
                "existing_label": existing_label,
                "measurement_surface": "stage_A_B_refined",
                "stage_status": row.get("status", ""),
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
    return sources


def attach_gaia(
    source: dict[str, object],
    gaia: dict[str, object],
    dist_arcsec: float,
    dist_pix: float,
    mode: str,
) -> None:
    source["gaia_source_id"] = gaia["source_id"]
    source["gaia_g_mag"] = gaia["phot_g_mean_mag"]
    source["gaia_match_arcsec"] = dist_arcsec
    source["gaia_match_pixels"] = dist_pix
    source["gaia_match_mode"] = mode
    source["output_x"] = gaia["x"]
    source["output_y"] = gaia["y"]


def synthetic_gaia_source(
    gaia: dict[str, object],
    *,
    comp: int,
    component_area: int,
    cluster_id: int,
    cluster_size: int,
) -> dict[str, object]:
    gsid = int(round(base.finite_float(gaia.get("source_id"), 0.0)))
    gmag = base.finite_float(gaia.get("phot_g_mean_mag"), float("nan"))
    x = float(gaia["x"])
    y = float(gaia["y"])
    return {
        "source_id": -gsid,
        "row_index": -1,
        "x": x,
        "y": y,
        "output_x": x,
        "output_y": y,
        "major": 3.0,
        "minor": 3.0,
        "theta_deg": 0.0,
        "area": math.pi * 3.0 * 3.0,
        "axis_ratio": 1.0,
        "mag": gmag,
        "class": "gaia_star",
        "classification_extendedness": "",
        "existing_label": "external_gaia",
        "stage_status": "synthetic_bright_gaia",
        "final_label": "strict_center_only_external",
        "reason": "component_bright_gaia_direct_strict_center_only",
        "component_id": comp,
        "component_area": component_area,
        "cluster_id": cluster_id,
        "cluster_size": cluster_size,
        "center_in_bad_mask": "",
        "center_bad_mask": "",
        "cluster_has_bright_gaia": True,
        "gaia_source_id": gaia["source_id"],
        "gaia_g_mag": gaia["phot_g_mean_mag"],
        "gaia_match_arcsec": 0.0,
        "gaia_match_pixels": 0.0,
        "gaia_match_mode": "bright_gaia_component_direct",
        "measurement_surface": "gaia_direct",
    }


def synthetic_component_center_source(
    *,
    comp: int,
    component_area: int,
    x: float,
    y: float,
) -> dict[str, object] | None:
    return {
        "source_id": -(800000000000000000 + int(comp)),
        "row_index": -1,
        "x": x,
        "y": y,
        "output_x": x,
        "output_y": y,
        "major": 3.0,
        "minor": 3.0,
        "theta_deg": 0.0,
        "area": math.pi * 3.0 * 3.0,
        "axis_ratio": 1.0,
        "mag": "",
        "class": "added_bright_component_center",
        "classification_extendedness": "",
        "existing_label": "external_added",
        "stage_status": "synthetic_empty_large_bright_component",
        "final_label": "strict_center_only_added",
        "reason": "empty_large_bright_component_geometric_center",
        "component_id": comp,
        "component_area": component_area,
        "cluster_id": 0,
        "cluster_size": 1,
        "center_in_bad_mask": "",
        "center_bad_mask": "",
        "cluster_has_bright_gaia": False,
        "gaia_source_id": "",
        "gaia_g_mag": "",
        "gaia_match_arcsec": "",
        "gaia_match_pixels": "",
        "gaia_match_mode": "",
        "measurement_surface": "bright_component_geometric_center",
    }


def matching_gaia_rows_to_cluster(
    cluster: list[int],
    sources: list[dict[str, object]],
    gaia_rows: list[dict[str, object]],
    *,
    source_match_pixels: float,
    centroid_match_pixels: float,
) -> list[tuple[dict[str, object], str, float]]:
    if not cluster:
        return []
    centroid_x = float(np.mean([float(sources[idx]["x"]) for idx in cluster]))
    centroid_y = float(np.mean([float(sources[idx]["y"]) for idx in cluster]))
    matches: list[tuple[dict[str, object], str, float]] = []
    for gaia in gaia_rows:
        gx = float(gaia["x"])
        gy = float(gaia["y"])
        source_dist = min(math.hypot(float(sources[idx]["x"]) - gx, float(sources[idx]["y"]) - gy) for idx in cluster)
        centroid_dist = math.hypot(gx - centroid_x, gy - centroid_y)
        if source_dist <= float(source_match_pixels):
            matches.append((gaia, "source_center", source_dist))
        elif centroid_dist <= float(centroid_match_pixels):
            matches.append((gaia, "cluster_centroid", centroid_dist))
    return sorted(matches, key=lambda item: (base.finite_float(item[0].get("phot_g_mean_mag"), float("inf")), item[2]))


def assign_labels_no_upper_source_clusters(
    *,
    sources: list[dict[str, object]],
    gaia_rows: list[dict[str, object]],
    image_shape: tuple[int, int],
    mask: np.ndarray,
    mask_header: fits.Header,
    mask_names: list[str],
    cluster_iou_threshold: float,
    cluster_max_center_distance: float,
    cluster_max_area: float,
    cluster_source_match_pixels: float,
    cluster_centroid_match_pixels: float,
) -> tuple[list[dict[str, object]], dict[int, dict[str, object]], list[dict[str, object]], np.ndarray, np.ndarray]:
    bits = base.mask_bits(mask_header, mask_names)
    for source in sources:
        in_bad, mask_hit = source_center_in_bad_mask(source, mask, bits)
        source["center_bad_mask"] = mask_hit
        source["center_in_bad_mask"] = in_bad
    bright_mask = np.zeros(image_shape, dtype=bool)
    clusters = base.cluster_component_sources(
        list(range(len(sources))),
        sources,
        iou_threshold=float(cluster_iou_threshold),
        max_center_distance=float(cluster_max_center_distance),
        max_area=float(cluster_max_area),
    )
    component_meta: dict[int, dict[str, object]] = {}
    cluster_rows: list[dict[str, object]] = []
    next_synthetic = 1
    for cluster_id, cluster in enumerate(clusters, start=1):
        cluster_area_sum = float(sum(float(sources[idx].get("area", 0.0)) for idx in cluster))
        for idx in cluster:
            sources[idx]["component_id"] = cluster_id
            sources[idx]["component_area"] = cluster_area_sum
            sources[idx]["cluster_id"] = cluster_id
            sources[idx]["cluster_size"] = len(cluster)
            sources[idx]["cluster_has_bright_gaia"] = False
        component_meta[cluster_id] = {
            "component_id": cluster_id,
            "component_area": cluster_area_sum,
            "hsc_source_count": len(cluster),
            "gaia_count": 0,
            "has_bright_gaia": False,
            "source_cluster_no_upper": True,
        }
        if len(cluster) == 1:
            source = sources[cluster[0]]
            if float(source["area"]) < 1000.0:
                source["final_label"] = "clean"
                source["reason"] = "no_upper_isolated_small_aperture_clean"
            else:
                source["final_label"] = "center_only_external"
                source["reason"] = "no_upper_isolated_large_aperture_center_only"
            cluster_rows.append(
                {
                    "cluster_id": cluster_id,
                    "component_id": cluster_id,
                    "component_area": cluster_area_sum,
                    "cluster_size": len(cluster),
                    "cluster_in_bad_mask": bool(source.get("center_in_bad_mask", False)),
                    "component_has_bright_gaia": False,
                    "chosen_source_id": source["source_id"],
                    "chosen_final_label": source["final_label"],
                    "chosen_reason": source["reason"],
                    "gaia_source_id": "",
                    "gaia_g_mag": "",
                    "gaia_match_arcsec": "",
                    "gaia_match_pixels": "",
                    "gaia_match_mode": "",
                    "source_ids": str(source["source_id"]),
                }
            )
            continue
        gaia_matches = matching_gaia_rows_to_cluster(
            cluster,
            sources,
            gaia_rows,
            source_match_pixels=float(cluster_source_match_pixels),
            centroid_match_pixels=float(cluster_centroid_match_pixels),
        )
        cluster_in_bad = any(bool(sources[idx].get("center_in_bad_mask", False)) for idx in cluster)
        if gaia_matches:
            for idx in cluster:
                sources[idx]["final_label"] = "ignore"
                sources[idx]["reason"] = "no_upper_gaia_matched_cluster_hsc_fragment_ignore"
                draw_source_ellipse(bright_mask, sources[idx])
            synthetic_rows = []
            for gaia, mode, dist_pix in gaia_matches:
                synthetic = synthetic_gaia_source(
                    gaia,
                    comp=cluster_id,
                    component_area=int(round(cluster_area_sum)),
                    cluster_id=cluster_id,
                    cluster_size=len(cluster) + len(gaia_matches),
                )
                synthetic["source_id"] = int(synthetic["source_id"]) - next_synthetic
                next_synthetic += 1
                synthetic["gaia_match_pixels"] = float(dist_pix)
                synthetic["gaia_match_arcsec"] = float(dist_pix) * 0.168
                synthetic["gaia_match_mode"] = f"no_upper_{mode}"
                synthetic["reason"] = "no_upper_gaia_direct_strict_center_only"
                synthetic_rows.append(synthetic)
            sources.extend(synthetic_rows)
            component_meta[cluster_id]["gaia_count"] = len(gaia_matches)
            component_meta[cluster_id]["has_bright_gaia"] = True
            cluster_rows.append(
                {
                    "cluster_id": cluster_id,
                    "component_id": cluster_id,
                    "component_area": cluster_area_sum,
                    "cluster_size": len(cluster),
                    "cluster_in_bad_mask": cluster_in_bad,
                    "component_has_bright_gaia": True,
                    "chosen_source_id": " ".join(str(row["source_id"]) for row in synthetic_rows),
                    "chosen_final_label": "strict_center_only_external",
                    "chosen_reason": "no_upper_gaia_direct_strict_center_only",
                    "gaia_source_id": " ".join(str(row["gaia_source_id"]) for row in synthetic_rows),
                    "gaia_g_mag": " ".join(str(row["gaia_g_mag"]) for row in synthetic_rows),
                    "gaia_match_arcsec": " ".join(str(row["gaia_match_arcsec"]) for row in synthetic_rows),
                    "gaia_match_pixels": " ".join(str(row["gaia_match_pixels"]) for row in synthetic_rows),
                    "gaia_match_mode": "no_upper_gaia_direct",
                    "source_ids": " ".join(str(sources[idx]["source_id"]) for idx in cluster),
                }
            )
        else:
            for idx in cluster:
                sources[idx]["final_label"] = "ignore"
                sources[idx]["reason"] = "no_upper_unmatched_cluster_ignore"
            cluster_rows.append(
                {
                    "cluster_id": cluster_id,
                    "component_id": cluster_id,
                    "component_area": cluster_area_sum,
                    "cluster_size": len(cluster),
                    "cluster_in_bad_mask": cluster_in_bad,
                    "component_has_bright_gaia": False,
                    "chosen_source_id": "",
                    "chosen_final_label": "ignore",
                    "chosen_reason": "no_upper_unmatched_cluster_ignore",
                    "gaia_source_id": "",
                    "gaia_g_mag": "",
                    "gaia_match_arcsec": "",
                    "gaia_match_pixels": "",
                    "gaia_match_mode": "",
                    "source_ids": " ".join(str(sources[idx]["source_id"]) for idx in cluster),
                }
            )
    component_labels, _num = base.ndimage.label(bright_mask)
    return sources, component_meta, cluster_rows, bright_mask.astype(np.uint8), component_labels.astype(np.int32)


def assign_labels_v2(
    *,
    sources: list[dict[str, object]],
    gaia_rows: list[dict[str, object]],
    component_labels: np.ndarray,
    component_areas: dict[int, int],
    component_centroids: dict[int, tuple[float, float]],
    mask: np.ndarray,
    mask_header: fits.Header,
    mask_names: list[str],
    component_search_radius: int,
    match_radius_arcsec: float,
    gaia_bright_mag_threshold: float,
    cluster_iou_threshold: float,
    cluster_max_center_distance: float,
    cluster_max_area: float,
    cluster_source_match_pixels: float,
    cluster_centroid_match_pixels: float,
    shape_max_area: float,
    shape_axis_ratio_max: float,
    drop_area_max: float,
    use_bad_mask_first_step: bool,
    use_bright_gaia_component_override: bool,
    add_empty_large_bright_component_centers: bool,
    empty_large_bright_component_area_min: float,
    large_component_fast_center_only_source_min: int,
) -> tuple[list[dict[str, object]], dict[int, dict[str, object]], list[dict[str, object]]]:
    bad_bits = base.mask_bits(mask_header, mask_names)
    by_component: dict[int, list[int]] = defaultdict(list)
    gaia_by_component: dict[int, list[dict[str, object]]] = defaultdict(list)
    for idx, source in enumerate(sources):
        comp = base.component_at(component_labels, float(source["x"]), float(source["y"]), component_search_radius)
        source["component_id"] = comp
        source["component_area"] = component_areas.get(comp, 0)
        in_bad, mask_hit = source_center_in_bad_mask(source, mask, bad_bits)
        source["center_bad_mask"] = mask_hit
        source["center_in_bad_mask"] = in_bad
        by_component[comp].append(idx)
    for gaia in gaia_rows:
        comp = base.component_at(component_labels, float(gaia["x"]), float(gaia["y"]), component_search_radius)
        if comp > 0:
            gaia_by_component[comp].append(gaia)

    component_meta: dict[int, dict[str, object]] = {}
    for comp, indices in by_component.items():
        if comp <= 0:
            continue
        bright_gaia = component_has_bright_gaia(
            comp, gaia_by_component, gaia_bright_mag_threshold=gaia_bright_mag_threshold
        )
        component_meta[comp] = {
            "component_id": comp,
            "component_area": component_areas.get(comp, 0),
            "hsc_source_count": len(indices),
            "gaia_count": len(gaia_by_component.get(comp, [])),
            "has_bright_gaia": bright_gaia,
        }

    pending_by_component: dict[int, list[int]] = defaultdict(list)
    for idx, source in enumerate(sources):
        if float(source["area"]) >= float(drop_area_max):
            source["final_label"] = "ignore"
            source["reason"] = f"dropped_large_area_ge_{float(drop_area_max):g}"
            continue
        comp = int(source["component_id"])
        if comp <= 0:
            source["final_label"] = "ignore"
            if bool(source["center_in_bad_mask"]):
                source["reason"] = f"outside_bright_center_in_{source['center_bad_mask']}"
            else:
                source["reason"] = "outside_bright_region"
            continue
        source_is_galaxy = str(source.get("class")) == "galaxy"
        has_bright_gaia = component_has_bright_gaia(
            comp,
            gaia_by_component,
            gaia_bright_mag_threshold=gaia_bright_mag_threshold,
        )
        if (
            use_bad_mask_first_step
            and bool(source["center_in_bad_mask"])
            and not source_is_galaxy
            and not has_bright_gaia
        ):
            source["final_label"] = "ignore"
            source["reason"] = f"bad_mask_non_galaxy_no_bright_gaia:{source['center_bad_mask']}"
            continue
        pending_by_component[comp].append(idx)

    cluster_rows: list[dict[str, object]] = []
    next_cluster_id = 1
    for comp in sorted(pending_by_component):
        bright_gaia_rows = [
            gaia
            for gaia in gaia_by_component.get(comp, [])
            if math.isfinite(base.finite_float(gaia.get("phot_g_mean_mag"), float("inf")))
            and base.finite_float(gaia.get("phot_g_mean_mag"), float("inf")) <= float(gaia_bright_mag_threshold)
        ]
        if use_bright_gaia_component_override and bright_gaia_rows:
            cluster_id = next_cluster_id
            next_cluster_id += 1
            for idx in pending_by_component[comp]:
                sources[idx]["cluster_id"] = cluster_id
                sources[idx]["cluster_size"] = len(pending_by_component[comp])
                sources[idx]["cluster_has_bright_gaia"] = True
                sources[idx]["final_label"] = "restricted_bright_region"
                sources[idx]["reason"] = "same_bright_gaia_component_hsc_fragment"
            synthetic_rows = [
                synthetic_gaia_source(
                    gaia,
                    comp=comp,
                    component_area=component_areas.get(comp, 0),
                    cluster_id=cluster_id,
                    cluster_size=len(pending_by_component[comp]) + len(bright_gaia_rows),
                )
                for gaia in sorted(bright_gaia_rows, key=lambda row: base.finite_float(row.get("phot_g_mean_mag"), float("inf")))
            ]
            sources.extend(synthetic_rows)
            component_meta.setdefault(comp, {})["cluster_count"] = 1
            component_meta.setdefault(comp, {})["candidate_source_count"] = len(pending_by_component[comp])
            cluster_rows.append(
                {
                    "cluster_id": cluster_id,
                    "component_id": comp,
                    "component_area": component_areas.get(comp, 0),
                    "cluster_size": len(pending_by_component[comp]),
                    "cluster_in_bad_mask": any(bool(sources[idx]["center_in_bad_mask"]) for idx in pending_by_component[comp]),
                    "component_has_bright_gaia": True,
                    "chosen_source_id": " ".join(str(row["source_id"]) for row in synthetic_rows),
                    "chosen_final_label": "strict_center_only_external",
                    "chosen_reason": "component_bright_gaia_direct_strict_center_only",
                    "gaia_source_id": " ".join(str(row["gaia_source_id"]) for row in synthetic_rows),
                    "gaia_g_mag": " ".join(str(row["gaia_g_mag"]) for row in synthetic_rows),
                    "gaia_match_arcsec": " ".join("0.0000" for _row in synthetic_rows),
                    "gaia_match_pixels": " ".join("0.000" for _row in synthetic_rows),
                    "gaia_match_mode": "bright_gaia_component_direct",
                    "source_ids": " ".join(str(sources[idx]["source_id"]) for idx in pending_by_component[comp]),
                }
            )
            continue
        if (
            int(large_component_fast_center_only_source_min) > 0
            and len(pending_by_component[comp]) >= int(large_component_fast_center_only_source_min)
        ):
            cluster_id = next_cluster_id
            next_cluster_id += 1
            for idx in pending_by_component[comp]:
                sources[idx]["cluster_id"] = cluster_id
                sources[idx]["cluster_size"] = len(pending_by_component[comp])
                sources[idx]["cluster_has_bright_gaia"] = False
                sources[idx]["final_label"] = "restricted_bright_region"
                sources[idx]["reason"] = "large_bright_component_fast_restricted_fragment"
            component_meta.setdefault(comp, {})["cluster_count"] = 1
            component_meta.setdefault(comp, {})["candidate_source_count"] = len(pending_by_component[comp])
            component_meta.setdefault(comp, {})["large_component_fast_center_only"] = True
            cluster_rows.append(
                {
                    "cluster_id": cluster_id,
                    "component_id": comp,
                    "component_area": component_areas.get(comp, 0),
                    "cluster_size": len(pending_by_component[comp]),
                    "cluster_in_bad_mask": any(bool(sources[idx]["center_in_bad_mask"]) for idx in pending_by_component[comp]),
                    "component_has_bright_gaia": False,
                    "chosen_source_id": "",
                    "chosen_final_label": "restricted_bright_region",
                    "chosen_reason": "large_bright_component_fast_restricted_fragment",
                    "gaia_source_id": "",
                    "gaia_g_mag": "",
                    "gaia_match_arcsec": "",
                    "gaia_match_pixels": "",
                    "gaia_match_mode": "",
                    "source_ids": " ".join(str(sources[idx]["source_id"]) for idx in pending_by_component[comp]),
                }
            )
            continue
        clusters = base.cluster_component_sources(
            pending_by_component[comp],
            sources,
            iou_threshold=float(cluster_iou_threshold),
            max_center_distance=float(cluster_max_center_distance),
            max_area=float(cluster_max_area),
        )
        component_meta.setdefault(comp, {})["cluster_count"] = len(clusters)
        component_meta.setdefault(comp, {})["candidate_source_count"] = len(pending_by_component[comp])
        has_bright_gaia = component_has_bright_gaia(
            comp, gaia_by_component, gaia_bright_mag_threshold=gaia_bright_mag_threshold
        )
        for cluster in clusters:
            cluster_id = next_cluster_id
            next_cluster_id += 1
            chosen_idx = min(cluster, key=lambda idx: float(sources[idx]["mag"]))
            chosen = sources[chosen_idx]
            if float(chosen["area"]) >= float(drop_area_max):
                chosen_idx = min(cluster, key=lambda idx: (float(sources[idx]["area"]) >= float(drop_area_max), float(sources[idx]["mag"])))
                chosen = sources[chosen_idx]
            cluster_in_bad_mask = any(bool(sources[idx]["center_in_bad_mask"]) for idx in cluster)
            gaia, gaia_idx, dist_arcsec, dist_pix, match_mode = base.nearest_gaia_to_cluster(
                cluster,
                sources,
                gaia_by_component.get(comp, []),
                match_radius_arcsec=float(match_radius_arcsec),
                source_match_pixels=float(cluster_source_match_pixels),
                centroid_match_pixels=float(cluster_centroid_match_pixels),
            )
            if gaia_idx is not None:
                chosen_idx = gaia_idx
                chosen = sources[chosen_idx]

            for idx in cluster:
                sources[idx]["cluster_id"] = cluster_id
                sources[idx]["cluster_size"] = len(cluster)
                sources[idx]["cluster_has_bright_gaia"] = has_bright_gaia

            cluster_finalized = False
            if len(cluster) == 1:
                only = sources[cluster[0]]
                only_is_galaxy = str(only.get("class")) == "galaxy"
                only_is_unknown = str(only.get("class")) == "unknown"
                only_shape_ok = source_shape_usable(
                    only,
                    max_area=shape_max_area,
                    max_axis_ratio=shape_axis_ratio_max,
                )
                if int(component_areas.get(comp, 0)) < 1000 and only_shape_ok:
                    only["final_label"] = "clean"
                    only["reason"] = "isolated_small_bright_component_clean"
                    cluster_finalized = True
                elif only_is_unknown:
                    only["final_label"] = "center_only_external"
                    only["reason"] = "isolated_unknown_center_only"
                    cluster_finalized = True
                elif only_is_galaxy and only_shape_ok and int(component_areas.get(comp, 0)) >= 1000:
                    only["final_label"] = "center_only_external"
                    only["reason"] = "isolated_large_bright_component_galaxy_center_only"
                    cluster_finalized = True

            if not cluster_finalized and gaia is not None:
                attach_gaia(chosen, gaia, dist_arcsec, dist_pix, match_mode)
                gmag = base.finite_float(gaia.get("phot_g_mean_mag"), float("inf"))
                if math.isfinite(gmag) and gmag <= float(gaia_bright_mag_threshold):
                    chosen_label = "strict_center_only_external"
                    chosen_reason = "gaia_bright_star_strict_center_only"
                elif str(chosen["class"]) == "galaxy" and source_shape_usable(
                    chosen, max_area=shape_max_area, max_axis_ratio=shape_axis_ratio_max
                ):
                    chosen_label = "center_only_external"
                    chosen_reason = "gaia_matched_hsc_galaxy_center_only"
                else:
                    chosen_label = "strict_center_only_external"
                    chosen_reason = "gaia_matched_star_or_unknown_strict_center_only"
                for idx in cluster:
                    if idx == chosen_idx:
                        sources[idx]["final_label"] = chosen_label
                        sources[idx]["reason"] = chosen_reason
                    else:
                        sources[idx]["final_label"] = "restricted_bright_region"
                        sources[idx]["reason"] = "same_cluster_has_gaia_not_chosen"
                cluster_finalized = True

            if not cluster_finalized:
                chosen_is_galaxy = str(chosen["class"]) == "galaxy"
                chosen_shape_ok = source_shape_usable(chosen, max_area=shape_max_area, max_axis_ratio=shape_axis_ratio_max)
                if chosen_is_galaxy and chosen_shape_ok:
                    for idx in cluster:
                        if idx == chosen_idx:
                            sources[idx]["final_label"] = "center_only_external"
                            sources[idx]["reason"] = "brightest_hsc_galaxy_center_only"
                        else:
                            sources[idx]["final_label"] = "restricted_bright_region"
                            sources[idx]["reason"] = "same_cluster_galaxy_member_not_chosen"
                else:
                    for idx in cluster:
                        sources[idx]["final_label"] = "ignore"
                        sources[idx]["reason"] = "no_gaia_non_galaxy_or_bad_shape_ignore"

            cluster_rows.append(
                {
                    "cluster_id": cluster_id,
                    "component_id": comp,
                    "component_area": component_areas.get(comp, 0),
                    "cluster_size": len(cluster),
                    "cluster_in_bad_mask": cluster_in_bad_mask,
                    "component_has_bright_gaia": has_bright_gaia,
                    "chosen_source_id": sources[chosen_idx]["source_id"],
                    "chosen_final_label": sources[chosen_idx]["final_label"],
                    "chosen_reason": sources[chosen_idx]["reason"],
                    "gaia_source_id": gaia["source_id"] if gaia is not None else "",
                    "gaia_g_mag": gaia["phot_g_mean_mag"] if gaia is not None else "",
                    "gaia_match_arcsec": dist_arcsec if gaia is not None else "",
                    "gaia_match_pixels": dist_pix if gaia is not None else "",
                    "gaia_match_mode": match_mode,
                    "source_ids": " ".join(str(sources[idx]["source_id"]) for idx in cluster),
                }
            )
    if add_empty_large_bright_component_centers:
        supervised_component_labels = {
            "clean",
            "center_only_external",
            "strict_center_only_external",
            "strict_center_only_added",
        }
        supervised_components = {
            int(source.get("component_id", 0))
            for source in sources
            if str(source.get("final_label", "")) in supervised_component_labels
        }
        for comp, area in sorted(component_areas.items()):
            if int(comp) <= 0 or int(area) < float(empty_large_bright_component_area_min):
                continue
            if int(comp) in supervised_components:
                continue
            if int(comp) not in component_centroids:
                continue
            x, y = component_centroids[int(comp)]
            added = synthetic_component_center_source(
                comp=int(comp),
                component_area=int(area),
                x=float(x),
                y=float(y),
            )
            if added is None:
                continue
            sources.append(added)
            component_meta.setdefault(int(comp), {})["added_empty_large_center"] = True
            component_meta.setdefault(int(comp), {})["component_id"] = int(comp)
            component_meta.setdefault(int(comp), {})["component_area"] = int(area)
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
        "stage_status",
        "final_label",
        "reason",
        "component_id",
        "component_area",
        "cluster_id",
        "cluster_size",
        "center_in_bad_mask",
        "center_bad_mask",
        "cluster_has_bright_gaia",
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


def reg_text(row: dict[str, object]) -> str:
    mag = base.finite_float(row.get("mag"), float("nan"))
    mag_text = f"{mag:.2f}" if math.isfinite(mag) else "NA"
    text = (
        f"sid={row.get('source_id')} label={row.get('final_label')} reason={row.get('reason')} "
        f"class={row.get('class')} mag={mag_text} area={float(row['area']):.0f} "
        f"component={row.get('component_id')} component_area={row.get('component_area')} "
        f"cluster={row.get('cluster_id')} bad_mask={row.get('center_bad_mask') or 'NONE'}"
    )
    if str(row.get("gaia_source_id", "")).strip():
        text += f" gaia={row.get('gaia_source_id')} g={row.get('gaia_g_mag')} d={row.get('gaia_match_arcsec')}"
    return text.replace("{", "(").replace("}", ")")


def write_label_regions(out_dir: Path, stem: str, rows: list[dict[str, object]]) -> None:
    colors = {
        "clean": "green",
        "center_only": "blue",
        "strict_center_only": "cyan",
        "center_only_external": "cyan",
        "strict_center_only_external": "cyan",
        "strict_center_only_added": "magenta",
        "restricted_bright_region": "orange",
        "ignore": "yellow",
    }
    for label in sorted({str(row.get("final_label", "")) for row in rows if row.get("final_label")}):
        path = out_dir / f"{stem}_{label}.reg"
        color = colors.get(label, "white")
        selected = [row for row in rows if str(row.get("final_label")) == label]
        with path.open("w", encoding="utf-8") as handle:
            handle.write("# Region file format: DS9 version 4.1\n")
            handle.write('global color=cyan dashlist=8 3 width=2 font="helvetica 9 normal roman"\n')
            handle.write("image\n")
            for row in selected:
                text = reg_text(row)
                if label in {"clean", "center_only", "center_only_external"}:
                    handle.write(base.ellipse_line(row, color, text=text))
                else:
                    x = float(row.get("output_x", row["x"]))
                    y = float(row.get("output_y", row["y"]))
                    handle.write(f"point({x:.3f},{y:.3f}) # point=cross color={color} width=2 text={{{text}}}\n")


def stage_reason_group(status: str) -> str:
    if status.startswith("remaining_"):
        return "remaining"
    if "refit_missing_or_bad" in status:
        return "removed_refit_missing_or_bad"
    if "A_filter" in status:
        return "removed_A_filter"
    if "axis_ratio" in status:
        return "removed_axis_ratio"
    if "close_pair" in status:
        return "removed_close_pair"
    if "outside_bright_region_absdiff" in status:
        return "removed_ap2_outside_bright"
    if "small_bright_region_absdiff" in status:
        return "removed_ap2_small_bright"
    return status or "unknown_stage"


def write_filter_step_overlay_png(
    path: Path,
    image: np.ndarray,
    stage_rows: list[dict[str, str]] | None,
    assigned_rows: list[dict[str, object]],
    dpi: int,
    *,
    mode: str,
) -> None:
    stage_rows = stage_rows or []
    display, vmin, vmax, cmap = final_overlay_display(image, mode=mode)
    assigned_by_id = {int(row["source_id"]): row for row in assigned_rows}
    removed_colors = {
        "removed_refit_missing_or_bad": "#ff3333",
        "removed_A_filter": "#b000ff",
        "removed_axis_ratio": "#ff8c00",
        "removed_close_pair": "#ffd000",
        "removed_ap2_outside_bright": "#0066ff",
        "removed_ap2_small_bright": "#00b7ff",
    }
    final_colors = {
        "clean": "#31e35f",
        "center_only": "#00a6ff",
        "strict_center_only": "#00eaff",
        "center_only_external": "#00eaff",
        "strict_center_only_external": "#00eaff",
        "strict_center_only_added": "#ff33ff",
        "restricted_bright_region": "#ff9d00",
        "ignore": "#ff3333",
    }
    counts: dict[str, int] = defaultdict(int)
    plotted_assigned_ids: set[int] = set()
    fig, ax = plt.subplots(figsize=(11, 11), constrained_layout=True)
    ax.imshow(display, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    for row in stage_rows:
        sid = int(round(base.finite_float(row.get("source_id"), -1.0)))
        x = base.finite_float(row.get("x"))
        y = base.finite_float(row.get("y"))
        if sid < 0 or not math.isfinite(x) or not math.isfinite(y):
            continue
        assigned = assigned_by_id.get(sid)
        if assigned is None:
            group = stage_reason_group(str(row.get("status", "")))
            color = removed_colors.get(group, "#ffffff")
            counts[group] += 1
            ax.plot(x, y, marker="+", markersize=4, markeredgewidth=0.85, color=color, linestyle="None")
            continue
        label = str(assigned.get("final_label", ""))
        plotted_assigned_ids.add(int(assigned["source_id"]))
        reason = str(assigned.get("reason", ""))
        group = f"{label}:{reason}" if label in {"ignore", "restricted_bright_region"} else label
        color = final_colors.get(label, "#ffffff")
        counts[group] += 1
        if label in {"clean", "center_only", "center_only_external"}:
            ax.add_patch(
                Ellipse(
                    (float(assigned["x"]), float(assigned["y"])),
                    width=2.0 * float(assigned["major"]),
                    height=2.0 * float(assigned["minor"]),
                    angle=float(assigned["theta_deg"]),
                    fill=False,
                    edgecolor=color,
                    linewidth=0.65,
                    alpha=0.95,
                )
            )
            ax.plot(float(assigned["x"]), float(assigned["y"]), marker="+", markersize=3, markeredgewidth=0.75, color=color)
        else:
            ax.plot(float(assigned.get("output_x", assigned["x"])), float(assigned.get("output_y", assigned["y"])), marker="+", markersize=4, markeredgewidth=0.85, color=color)
    for assigned in assigned_rows:
        assigned_id = int(assigned["source_id"])
        if assigned_id in plotted_assigned_ids:
            continue
        label = str(assigned.get("final_label", ""))
        color = final_colors.get(label, "#ffffff")
        reason = str(assigned.get("reason", ""))
        group = f"{label}:{reason}" if label in {"ignore", "restricted_bright_region"} else label
        counts[group] += 1
        if label in {"clean", "center_only", "center_only_external"}:
            ax.add_patch(
                Ellipse(
                    (float(assigned["x"]), float(assigned["y"])),
                    width=2.0 * float(assigned["major"]),
                    height=2.0 * float(assigned["minor"]),
                    angle=float(assigned["theta_deg"]),
                    fill=False,
                    edgecolor=color,
                    linewidth=0.65,
                    alpha=0.95,
                )
            )
            ax.plot(float(assigned["x"]), float(assigned["y"]), marker="+", markersize=3, markeredgewidth=0.75, color=color)
        else:
            marker = "x" if int(assigned.get("row_index", 0)) == -1 else "+"
            ax.plot(
                float(assigned.get("output_x", assigned["x"])),
                float(assigned.get("output_y", assigned["y"])),
                marker=marker,
                markersize=5 if marker == "x" else 4,
                markeredgewidth=1.0 if marker == "x" else 0.85,
                color=color,
                linestyle="None",
            )
    handles = []
    for group, color in removed_colors.items():
        if counts[group] > 0:
            handles.append(Patch(facecolor=color, edgecolor="none", label=f"{group} n={counts[group]}"))
    for group in sorted(counts):
        if group in removed_colors:
            continue
        label = group.split(":", 1)[0]
        handles.append(Patch(facecolor=final_colors.get(label, "#ffffff"), edgecolor="none", label=f"{group} n={counts[group]}"))
    ax.legend(handles=handles, loc="upper right", fontsize=5.5, framealpha=0.86)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.set_xlabel("image x")
    ax.set_ylabel("image y")
    ax.set_title(path.stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def final_overlay_display(image: np.ndarray, *, mode: str) -> tuple[np.ndarray, float, float, str]:
    if mode == "lupton":
        display = base.lupton_mapping(image, stretch=0.5, q=20.0)
        finite = display[np.isfinite(display)]
        return np.nan_to_num(display, nan=0.0), float(np.nanmin(finite)), float(np.nanmax(finite)), "gray"
    finite = image[np.isfinite(image)]
    vmin, vmax = ZScaleInterval().get_limits(finite)
    return np.nan_to_num(image, nan=vmin, posinf=vmax, neginf=vmin), float(vmin), float(vmax), "gray"


def write_final_class_overlay_png(path: Path, image: np.ndarray, rows: list[dict[str, object]], dpi: int, *, mode: str) -> None:
    display, vmin, vmax, cmap = final_overlay_display(image, mode=mode)
    colors = {
        "clean": "#31e35f",
        "center_only": "#00a6ff",
        "strict_center_only": "#00eaff",
        "center_only_external": "#00eaff",
        "strict_center_only_external": "#00eaff",
        "strict_center_only_added": "#ff33ff",
        "restricted_bright_region": "#ff9d00",
        "ignore": "#ff3333",
    }
    fig, ax = plt.subplots(figsize=(11, 11), constrained_layout=True)
    ax.imshow(display, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    counts: dict[str, int] = defaultdict(int)
    for row in sorted(rows, key=lambda item: float(item.get("area", 0.0)), reverse=True):
        label = str(row.get("final_label", ""))
        if not label:
            continue
        counts[label] += 1
        color = colors.get(label, "white")
        x = float(row.get("output_x", row["x"]))
        y = float(row.get("output_y", row["y"]))
        if label in {"clean", "center_only", "center_only_external"}:
            patch = Ellipse(
                (float(row["x"]), float(row["y"])),
                width=2.0 * float(row["major"]),
                height=2.0 * float(row["minor"]),
                angle=float(row["theta_deg"]),
                fill=False,
                edgecolor=color,
                linewidth=0.75,
                alpha=0.92,
            )
            ax.add_patch(patch)
            ax.plot(float(row["x"]), float(row["y"]), marker="+", markersize=3, markeredgewidth=0.8, color=color)
        else:
            ax.plot(x, y, marker="+", markersize=4, markeredgewidth=0.9, color=color)
    for label, color in colors.items():
        if counts[label] > 0:
            ax.plot([], [], color=color, label=f"{label} n={counts[label]}")
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.set_xlabel("image x")
    ax.set_ylabel("image y")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax.set_title(path.stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def parse_tile_origin(path: Path) -> tuple[int, int] | None:
    match = re.search(r"_x(-?\d+)_y(-?\d+)", path.stem)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def patch_origin_from_header(header: fits.Header) -> tuple[int, int]:
    if "CRVAL1A" in header and "CRVAL2A" in header:
        return int(round(float(header["CRVAL1A"]))), int(round(float(header["CRVAL2A"])))
    if "LTV1" in header and "LTV2" in header:
        return int(round(-float(header["LTV1"]))), int(round(-float(header["LTV2"])))
    return 0, 0


def mosaic_preprocessed_masks(
    preprocessed_root: Path,
    tract: str,
    patch: str,
    band: str,
    shape: tuple[int, int],
    origin: tuple[int, int],
) -> dict[str, np.ndarray]:
    names = ["clean_mask", "center_only_mask", "strict_center_only_mask", "background_mask", "ignore_mask"]
    out = {name: np.zeros(shape, dtype=bool) for name in names}
    target_dir = preprocessed_root / str(tract) / patch / "band_targets" / band
    ox, oy = origin
    for path in target_dir.glob("*.npz"):
        tile_origin = parse_tile_origin(path)
        if tile_origin is None:
            continue
        x0 = tile_origin[0] - ox
        y0 = tile_origin[1] - oy
        if x0 >= shape[1] or y0 >= shape[0] or x0 + 1 <= 0 or y0 + 1 <= 0:
            continue
        data = np.load(path)
        for name in names:
            if name not in data.files:
                continue
            arr = np.asarray(data[name]).astype(bool)
            h, w = arr.shape
            sx0 = max(0, -x0)
            sy0 = max(0, -y0)
            sx1 = min(w, shape[1] - x0)
            sy1 = min(h, shape[0] - y0)
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            out[name][y0 + sy0 : y0 + sy1, x0 + sx0 : x0 + sx1] |= arr[sy0:sy1, sx0:sx1]
    return out


def official_lsst_background_mask(
    data_root: Path,
    tract: str,
    patch: str,
    band: str,
    shape: tuple[int, int],
    origin: tuple[int, int],
) -> np.ndarray:
    path = det_path(data_root, tract, band, patch)
    if not path.exists():
        raise FileNotFoundError(f"official LSST det footprint file not found: {path}")
    return read_det_background_mask(path, shape, origin)


def draw_center_disk(mask: np.ndarray, x: float, y: float, radius: int) -> None:
    xi = int(round(x))
    yi = int(round(y))
    r = int(radius)
    y0 = max(0, yi - r)
    y1 = min(mask.shape[0], yi + r + 1)
    x0 = max(0, xi - r)
    x1 = min(mask.shape[1], xi + r + 1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask[y0:y1, x0:x1] |= (xx - xi) ** 2 + (yy - yi) ** 2 <= r * r


def draw_source_ellipse(mask: np.ndarray, row: dict[str, object]) -> None:
    x0 = float(row["x"])
    y0 = float(row["y"])
    major = max(float(row["major"]), 1e-6)
    minor = max(float(row["minor"]), 1e-6)
    theta = math.radians(float(row["theta_deg"]))
    radius = int(math.ceil(max(major, minor)))
    xi = int(round(x0))
    yi = int(round(y0))
    x_min = max(0, xi - radius)
    x_max = min(mask.shape[1], xi + radius + 1)
    y_min = max(0, yi - radius)
    y_max = min(mask.shape[0], yi + radius + 1)
    if x_max <= x_min or y_max <= y_min:
        return
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    dx = xx.astype(np.float32) - x0
    dy = yy.astype(np.float32) - y0
    c = math.cos(theta)
    s = math.sin(theta)
    xr = dx * c + dy * s
    yr = -dx * s + dy * c
    inside = (xr / major) ** 2 + (yr / minor) ** 2 <= 1.0
    mask[y_min:y_max, x_min:x_max] |= inside


def build_priority_partition(
    *,
    pre_masks: dict[str, np.ndarray],
    lsst_background_mask: np.ndarray,
    bright_mask: np.ndarray,
    rows: list[dict[str, object]],
    external_center_radius: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    clean = pre_masks["clean_mask"].copy()
    center = pre_masks["center_only_mask"] | pre_masks["strict_center_only_mask"]
    background = np.asarray(lsst_background_mask, dtype=bool)
    ignore = pre_masks["ignore_mask"].copy()
    for row in rows:
        if str(row.get("final_label")) == "center_only_external":
            draw_source_ellipse(center, row)
        elif str(row.get("final_label")) in {"strict_center_only_external", "strict_center_only_added"}:
            draw_center_disk(center, float(row.get("output_x", row["x"])), float(row.get("output_y", row["y"])), external_center_radius)
    label = np.full(bright_mask.shape, 5, dtype=np.uint8)
    label[ignore] = 5
    label[background] = 4
    label[bright_mask.astype(bool)] = 3
    label[center] = 2
    label[clean] = 1
    exclusive = {
        "clean": label == 1,
        "center_only": label == 2,
        "bright": label == 3,
        "background": label == 4,
        "ignore": label == 5,
    }
    return label, exclusive


def write_partition_fits(out_dir: Path, stem: str, partition: np.ndarray, masks: dict[str, np.ndarray], header: fits.Header) -> None:
    fits.writeto(out_dir / f"{stem}_priority_partition.fits", partition.astype(np.uint8), header=header, overwrite=True)
    for name, mask in masks.items():
        fits.writeto(out_dir / f"{stem}_{name}_priority_mask.fits", mask.astype(np.uint8), header=header, overwrite=True)


def write_partition_png(path: Path, image: np.ndarray, partition: np.ndarray, rows: list[dict[str, object]], dpi: int) -> None:
    finite = image[np.isfinite(image)]
    vmin, vmax = ZScaleInterval().get_limits(finite)
    display = np.nan_to_num(image, nan=vmin, posinf=vmax, neginf=vmin)
    colors = {
        1: (0.05, 0.85, 0.25, 0.38),  # clean
        2: (0.0, 0.75, 1.0, 0.42),  # center only
        3: (1.0, 0.58, 0.0, 0.30),  # bright
        4: (0.35, 0.35, 1.0, 0.24),  # background
        5: (1.0, 0.92, 0.0, 0.22),  # ignore
    }
    rgba = np.zeros((*partition.shape, 4), dtype=np.float32)
    for value, color in colors.items():
        rgba[partition == value] = color
    fig, ax = plt.subplots(figsize=(11, 11), constrained_layout=True)
    ax.imshow(display, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.imshow(rgba, origin="lower", interpolation="nearest")
    marker_colors = {
        "clean": "#00ff5a",
        "center_only": "#00c8ff",
        "strict_center_only": "#00f7ff",
        "center_only_external": "#00f7ff",
        "strict_center_only_external": "#00f7ff",
        "strict_center_only_added": "#ff33ff",
        "restricted_bright_region": "#ff9500",
        "ignore": "#ffe45c",
    }
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        label = str(row.get("final_label", ""))
        counts[label] += 1
        color = marker_colors.get(label, "white")
        x = float(row.get("output_x", row["x"]))
        y = float(row.get("output_y", row["y"]))
        if label == "clean":
            patch = Ellipse(
                (float(row["x"]), float(row["y"])),
                width=2.0 * float(row["major"]),
                height=2.0 * float(row["minor"]),
                angle=float(row["theta_deg"]),
                fill=False,
                edgecolor=color,
                linewidth=0.6,
                alpha=0.8,
            )
            ax.add_patch(patch)
        ax.plot(x, y, marker="+", markersize=3, markeredgewidth=0.7, color=color)
    handles = [
        Patch(facecolor=colors[1], edgecolor="none", label="clean"),
        Patch(facecolor=colors[2], edgecolor="none", label="center only"),
        Patch(facecolor=colors[3], edgecolor="none", label="bright"),
        Patch(facecolor=colors[4], edgecolor="none", label="background"),
        Patch(facecolor=colors[5], edgecolor="none", label="ignore"),
    ]
    for label in sorted(counts):
        handles.append(Patch(facecolor=marker_colors.get(label, "white"), edgecolor="none", label=f"{label} src={counts[label]}"))
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.set_xlabel("image x")
    ax.set_ylabel("image y")
    ax.set_title(path.stem)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def process_one(args: argparse.Namespace, patch: str, band: str) -> dict[str, object]:
    out_dir = args.out_dir / str(args.tract) / patch / band
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"processing {args.tract}/{patch} {band}", flush=True)
    image, mask, image_header, mask_header = base.read_exposure(base.image_path(args.data_root, args.tract, band, patch))
    meas = Table.read(base.meas_path(args.data_root, args.tract, band, patch), hdu=1)
    labels = base.load_label_sets(args.preprocessed_root, args.tract, patch, band)
    stage_rows = load_stage_rows(args.stage_filter_root, args.tract, patch, band)
    if stage_rows is not None:
        sources = load_sources_from_stage_rows(rows=stage_rows, meas=meas, labels=labels)
        print(
            f"{band}: stage-filter loaded {len(sources)}/{len(stage_rows)} remaining bright sources from {args.stage_filter_root}",
            flush=True,
        )
    else:
        sources = base.load_bright_hsc_sources(
            meas=meas,
            refit_csv=base.refit_path(args.refit_root, args.tract, band, patch),
            labels=labels,
            mag_threshold=float(args.bright_mag_threshold),
            zeropoint=float(args.zeropoint),
        )
    wcs = WCS(image_header)
    gaia_rows = base.load_gaia_for_patch(args.gaia_fits, wcs, image.shape)
    bright_mode = str(args.bright_mask_mode).strip().lower().replace("_", "-")
    if bright_mode in {"zscore-no-upper", "zscore-unbounded"}:
        assigned, component_meta, cluster_rows, bright_mask, component_labels = bright_core.assign_no_upper_source_clusters(
            sources=sources,
            gaia_rows=gaia_rows,
            image_shape=image.shape,
            mask=mask,
            mask_header=mask_header,
            mask_names=list(args.mask_names),
            config=bright_core.NoUpperBrightLabelConfig(
                cluster_iou_threshold=float(args.cluster_iou_threshold),
                cluster_max_center_distance=float(args.cluster_max_center_distance),
                cluster_max_area=float(args.cluster_max_area),
                cluster_source_match_pixels=float(args.cluster_source_match_pixels),
                cluster_centroid_match_pixels=float(args.cluster_centroid_match_pixels),
                isolated_clean_area_max=1000.0,
            ),
        )
    else:
        bright_mask, component_labels = build_bright_components_v2(image, args)
        comp_areas = component_area_map(component_labels)
        comp_centroids = component_centroid_map(component_labels)
        assigned, component_meta, cluster_rows = assign_labels_v2(
            sources=sources,
            gaia_rows=gaia_rows,
            component_labels=component_labels,
            component_areas=comp_areas,
            component_centroids=comp_centroids,
            mask=mask,
            mask_header=mask_header,
            mask_names=list(args.mask_names),
            component_search_radius=int(args.component_search_radius),
            match_radius_arcsec=float(args.match_radius_arcsec),
            gaia_bright_mag_threshold=float(args.gaia_bright_mag_threshold),
            cluster_iou_threshold=float(args.cluster_iou_threshold),
            cluster_max_center_distance=float(args.cluster_max_center_distance),
            cluster_max_area=float(args.cluster_max_area),
            cluster_source_match_pixels=float(args.cluster_source_match_pixels),
            cluster_centroid_match_pixels=float(args.cluster_centroid_match_pixels),
            shape_max_area=float(args.shape_max_area),
            shape_axis_ratio_max=float(args.shape_axis_ratio_max),
            drop_area_max=float(args.drop_area_max),
            use_bad_mask_first_step=bool(args.use_bad_mask_first_step),
            use_bright_gaia_component_override=bool(args.use_bright_gaia_component_override),
            add_empty_large_bright_component_centers=bool(args.add_empty_large_bright_component_centers),
            empty_large_bright_component_area_min=float(args.empty_large_bright_component_area_min),
            large_component_fast_center_only_source_min=int(args.large_component_fast_center_only_source_min),
        )
    stem = f"{args.tract}_{patch.replace(',', '_')}_{band}"
    write_csv(out_dir / f"{stem}_bright_reclassification_v2.csv", assigned)
    base.write_csv_generic(out_dir / f"{stem}_component_summary_v2.csv", list(component_meta.values()))
    base.write_csv_generic(out_dir / f"{stem}_cluster_summary_v2.csv", cluster_rows)
    write_label_regions(out_dir, stem, assigned)
    fits.writeto(out_dir / f"{stem}_log_lupton_bright_mask.fits", bright_mask.astype(np.uint8), image_header, overwrite=True)
    fits.writeto(out_dir / f"{stem}_bright_mask.fits", bright_mask.astype(np.uint8), image_header, overwrite=True)
    write_final_class_overlay_png(
        out_dir / f"{stem}_final_class_overlay_zscale.png",
        image,
        assigned,
        int(args.dpi),
        mode="zscale",
    )
    write_final_class_overlay_png(
        out_dir / f"{stem}_final_class_overlay_lupton.png",
        image,
        assigned,
        int(args.dpi),
        mode="lupton",
    )
    write_filter_step_overlay_png(
        out_dir / f"{stem}_filter_step_diagnostics_zscale.png",
        image,
        stage_rows,
        assigned,
        int(args.dpi),
        mode="zscale",
    )
    write_filter_step_overlay_png(
        out_dir / f"{stem}_filter_step_diagnostics_lupton.png",
        image,
        stage_rows,
        assigned,
        int(args.dpi),
        mode="lupton",
    )
    counts: dict[str, int] = defaultdict(int)
    reason_counts: dict[str, int] = defaultdict(int)
    for row in assigned:
        counts[str(row["final_label"])] += 1
        reason_counts[str(row.get("reason", ""))] += 1
    base.write_csv_generic(
        out_dir / f"{stem}_reason_counts_v2.csv",
        [{"reason": reason, "count": count} for reason, count in sorted(reason_counts.items())],
    )
    first_step_removed = sum(
        count for reason, count in reason_counts.items() if reason.startswith("bad_mask_non_galaxy_no_bright_gaia")
    )
    summary = {
        "patch": patch,
        "band": band,
        "bright_hsc_sources": len(assigned),
        "gaia_patch_sources": len(gaia_rows),
        "bright_components": int(component_labels.max()),
        "bright_mask_components": int(component_labels.max()),
        "image_threshold_components": 0
        if bright_mode in {"zscore-no-upper", "zscore-unbounded"}
        else int(component_labels.max()),
        "bright_mask_mode": str(args.bright_mask_mode),
        "bright_z_threshold": float(args.bright_z_threshold),
        "bright_anscombe_scale": float(args.bright_anscombe_scale),
        "use_bad_mask_first_step": bool(args.use_bad_mask_first_step),
        "use_bright_gaia_component_override": bool(args.use_bright_gaia_component_override),
        "first_step_bad_mask_removed": int(first_step_removed),
        **counts,
    }
    print(summary, flush=True)
    return summary


def main() -> int:
    args = parse_args()
    summaries = []
    for patch in args.patches:
        for band in args.bands:
            summaries.append(process_one(args, patch, band))
    base.write_csv_generic(args.out_dir / str(args.tract) / "summary_v2.csv", summaries)
    print(f"wrote {args.out_dir / str(args.tract)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
