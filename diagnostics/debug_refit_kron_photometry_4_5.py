#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval


TRACT = 9813
PATCH = "4,5"
PATCH_X0 = 15900.0
PATCH_Y0 = 19900.0
ZP = 27.0

COLORS = {
    "new_diff_lt_1": (0, 255, 0),
    "new_diff_1_1p5": (255, 220, 0),
    "new_diff_gt_1p5": (255, 0, 255),
    "invalid_new_flux": (255, 0, 0),
    "final_ignore": (255, 0, 255),
}
DS9_COLORS = {
    "new_diff_lt_1": "green",
    "new_diff_1_1p5": "yellow",
    "new_diff_gt_1p5": "magenta",
    "invalid_new_flux": "red",
    "final_ignore": "magenta",
}

SMALL_FOOTPRINT_FILL_THRESHOLD = 0.2
CENTER_ONLY_APERTURE_AREA = 10000.0
FAINT_AP2_MAG_THRESHOLD = 28.0
FAINT_APERTURE_AREA_THRESHOLD = 900.0
AXIS_RATIO_THRESHOLD = 5.0
CONTAINMENT_FRACTION_THRESHOLD = 0.80


@dataclass(frozen=True)
class ArchiveLookup:
    row0: np.ndarray
    nrows: np.ndarray
    found: np.ndarray


@dataclass(frozen=True)
class ArchiveIndex:
    ids: np.ndarray
    archive_numbers: np.ndarray
    names: np.ndarray
    row0: np.ndarray
    nrows: np.ndarray

    @classmethod
    def from_archive(cls, archive: fits.FITS_rec) -> "ArchiveIndex":
        return cls(
            ids=np.asarray(archive["id"], dtype=np.int64),
            archive_numbers=np.asarray(archive["cat.archive"], dtype=np.int64),
            names=np.asarray([_decode_string(value) for value in archive["name"]]),
            row0=np.asarray(archive["row0"], dtype=np.int64),
            nrows=np.asarray(archive["nrows"], dtype=np.int64),
        )

    def lookup(self, target_ids: np.ndarray, *, archive_number: int, name: str | None) -> ArchiveLookup:
        target = np.asarray(target_ids, dtype=np.int64)
        mask = self.archive_numbers == int(archive_number)
        if name is not None:
            mask &= self.names == str(name)
        group_rows = np.flatnonzero(mask)
        group_ids = self.ids[group_rows]
        order = np.argsort(group_ids, kind="mergesort")
        sorted_ids = group_ids[order]
        sorted_rows = group_rows[order]

        found = np.zeros(target.shape, dtype=bool)
        row0 = np.full(target.shape, -1, dtype=np.int64)
        nrows = np.zeros(target.shape, dtype=np.int64)
        if sorted_ids.size:
            pos = np.searchsorted(sorted_ids, target)
            in_range = pos < sorted_ids.size
            matched = np.zeros(target.shape, dtype=bool)
            matched[in_range] = sorted_ids[pos[in_range]] == target[in_range]
            found = matched
            archive_rows = sorted_rows[pos[found]]
            row0[found] = self.row0[archive_rows]
            nrows[found] = self.nrows[archive_rows]
        return ArchiveLookup(row0=row0, nrows=nrows, found=found)


def _decode_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value).strip()


def magnitude(flux: np.ndarray | float, zp: float = ZP) -> np.ndarray | float:
    arr = np.asarray(flux, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(arr) & (arr > 0.0)
    out[valid] = float(zp) - 2.5 * np.log10(arr[valid])
    if np.isscalar(flux):
        return float(out)
    return out


def read_image(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
        for hdu in hdul:
            if hdu.data is not None and int(hdu.header.get("NAXIS", 0)) >= 2:
                return np.asarray(hdu.data, dtype=np.float32), hdu.header.copy()
    raise ValueError(f"no image HDU found: {path}")


def finite_float(table: fits.FITS_rec, column: str, default: float = np.nan) -> np.ndarray:
    if column not in table.columns.names:
        return np.full(len(table), default, dtype=np.float64)
    return np.asarray(table[column], dtype=np.float64)


def bool_column(table: fits.FITS_rec, column: str) -> np.ndarray:
    if column not in table.columns.names:
        return np.zeros(len(table), dtype=bool)
    return np.asarray(table[column], dtype=bool)


def row_axis_params(main: fits.FITS_rec, row: int, matched: bool) -> tuple[float, float, float, str]:
    major = float(main["ellipse_major_sigma"][row]) if "ellipse_major_sigma" in main.columns.names else np.nan
    minor = float(main["ellipse_minor_sigma"][row]) if "ellipse_minor_sigma" in main.columns.names else np.nan
    theta = float(main["ellipse_theta"][row]) if "ellipse_theta" in main.columns.names else 0.0
    if matched and np.isfinite(major) and np.isfinite(minor) and major > 0.0 and minor > 0.0:
        return major, minor, theta, "refit_ellipse"
    if np.isfinite(major) and np.isfinite(minor) and major > 0.0 and minor > 0.0:
        return 2.5 * major, 2.5 * minor, theta, "catalog_fallback_scaled_ellipse"
    radius = float(main["ext_photometryKron_KronFlux_radius"][row])
    if np.isfinite(radius) and radius > 0.0:
        aperture_radius = 2.5 * radius
        return aperture_radius, aperture_radius, 0.0, "catalog_fallback_circular"
    return np.nan, np.nan, np.nan, "invalid_shape"


def aperture_mask_values(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    cx: float,
    cy: float,
    major: float,
    minor: float,
    theta: float,
) -> tuple[float, int, int]:
    if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(major) and np.isfinite(minor)):
        return np.nan, 0, 0
    if major <= 0.0 or minor <= 0.0:
        return np.nan, 0, 0
    dx = x.astype(np.float64) - cx
    dy = y.astype(np.float64) - cy
    c = math.cos(theta)
    s = math.sin(theta)
    du = dx * c + dy * s
    dv = -dx * s + dy * c
    inside = (du / major) ** 2 + (dv / minor) ** 2 <= 1.0
    if not np.any(inside):
        return 0.0, 0, int(values.size)
    selected = np.asarray(values[inside], dtype=np.float64)
    finite = np.isfinite(selected)
    return float(np.sum(np.where(finite, selected, 0.0))), int(np.count_nonzero(inside)), int(values.size)


def measure_heavy(
    *,
    spans_table: fits.FITS_rec,
    heavy_table: fits.FITS_rec,
    span0: int,
    nspan: int,
    heavy_row: int,
    cx: float,
    cy: float,
    major: float,
    minor: float,
    theta: float,
) -> tuple[float, int, int]:
    if span0 < 0 or nspan <= 0 or heavy_row < 0:
        return np.nan, 0, 0
    spans = spans_table[span0 : span0 + nspan]
    widths = np.asarray(spans["x1"], dtype=np.int64) - np.asarray(spans["x0"], dtype=np.int64) + 1
    total_pixels = int(np.sum(widths))
    if total_pixels <= 0:
        return np.nan, 0, 0
    span_index = np.repeat(np.arange(nspan, dtype=np.int64), widths)
    starts = np.repeat(np.cumsum(widths) - widths, widths)
    x = np.asarray(spans["x0"], dtype=np.int64)[span_index] + (np.arange(total_pixels) - starts)
    y = np.asarray(spans["y"], dtype=np.int64)[span_index]
    values = np.asarray(heavy_table["image"][heavy_row], dtype=np.float64)
    if values.size != total_pixels:
        return np.nan, 0, total_pixels
    return aperture_mask_values(x, y, values, cx=cx, cy=cy, major=major, minor=minor, theta=theta)


def footprint_aperture_counts(
    *,
    spans_table: fits.FITS_rec,
    span0: int,
    nspan: int,
    cx: float,
    cy: float,
    major: float,
    minor: float,
    theta: float,
) -> tuple[int, int]:
    if span0 < 0 or nspan <= 0:
        return 0, 0
    spans = spans_table[span0 : span0 + nspan]
    widths = np.asarray(spans["x1"], dtype=np.int64) - np.asarray(spans["x0"], dtype=np.int64) + 1
    total_pixels = int(np.sum(widths))
    if total_pixels <= 0:
        return 0, 0
    span_index = np.repeat(np.arange(nspan, dtype=np.int64), widths)
    starts = np.repeat(np.cumsum(widths) - widths, widths)
    x = np.asarray(spans["x0"], dtype=np.int64)[span_index] + (np.arange(total_pixels) - starts)
    y = np.asarray(spans["y"], dtype=np.int64)[span_index]
    _, aperture_pixels, _ = aperture_mask_values(
        x,
        y,
        np.ones(total_pixels, dtype=np.float64),
        cx=cx,
        cy=cy,
        major=major,
        minor=minor,
        theta=theta,
    )
    return int(aperture_pixels), int(total_pixels)


def measure_direct(
    *,
    image: np.ndarray,
    x_ltv: float,
    y_ltv: float,
    cx: float,
    cy: float,
    major: float,
    minor: float,
    theta: float,
) -> tuple[float, int, int]:
    if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(major) and np.isfinite(minor)):
        return np.nan, 0, 0
    r = int(math.ceil(max(major, minor) + 2.0))
    local_cx = cx + x_ltv
    local_cy = cy + y_ltv
    x0 = max(0, int(math.floor(local_cx)) - r)
    x1 = min(image.shape[1] - 1, int(math.floor(local_cx)) + r)
    y0 = max(0, int(math.floor(local_cy)) - r)
    y1 = min(image.shape[0] - 1, int(math.floor(local_cy)) + r)
    if x1 < x0 or y1 < y0:
        return np.nan, 0, 0
    yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    global_x = xx.astype(np.float64) - x_ltv
    global_y = yy.astype(np.float64) - y_ltv
    return aperture_mask_values(
        global_x.ravel(),
        global_y.ravel(),
        np.asarray(image[y0 : y1 + 1, x0 : x1 + 1], dtype=np.float64).ravel(),
        cx=cx,
        cy=cy,
        major=major,
        minor=minor,
        theta=theta,
    )


def classify_new_diff(diff: float) -> str:
    if not np.isfinite(diff):
        return "invalid_new_flux"
    if diff < 1.0:
        return "new_diff_lt_1"
    if diff <= 1.5:
        return "new_diff_1_1p5"
    return "new_diff_gt_1p5"


def unit_disk_points(n: int = 15) -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, int(n), dtype=np.float64)
    uu, vv = np.meshgrid(grid, grid)
    pts = np.column_stack([uu.ravel(), vv.ravel()])
    return pts[np.sum(pts * pts, axis=1) <= 1.0]


def ellipse_contains_fraction(
    *,
    small: dict[str, float],
    big: dict[str, float],
    unit_points: np.ndarray,
) -> float:
    sx = small["x"]
    sy = small["y"]
    sa = small["a"]
    sb = small["b"]
    st = small["theta"]
    bx = big["x"]
    by = big["y"]
    ba = big["a"]
    bb = big["b"]
    bt = big["theta"]
    if min(sa, sb, ba, bb) <= 0.0:
        return 0.0
    cs = math.cos(st)
    ss = math.sin(st)
    u = unit_points[:, 0] * sa
    v = unit_points[:, 1] * sb
    px = sx + u * cs - v * ss
    py = sy + u * ss + v * cs
    cb = math.cos(bt)
    sbt = math.sin(bt)
    dx = px - bx
    dy = py - by
    du = dx * cb + dy * sbt
    dv = -dx * sbt + dy * cb
    inside = (du / ba) ** 2 + (dv / bb) ** 2 <= 1.0
    return float(np.mean(inside))


def table_string_column(main: fits.FITS_rec, column: str) -> np.ndarray:
    if column not in main.columns.names:
        return np.asarray([""] * len(main), dtype=object)
    return np.asarray([_decode_string(value) for value in main[column]], dtype=object)


def geometry_from_row(main: fits.FITS_rec, row: int) -> dict[str, float] | None:
    cx = float(main["base_SdssCentroid_x"][row])
    cy = float(main["base_SdssCentroid_y"][row])
    matched = bool(main["pu_refit_kron_radius_matched"][row]) if "pu_refit_kron_radius_matched" in main.columns.names else False
    major, minor, theta, _ = row_axis_params(main, row, matched)
    if not (
        np.isfinite(cx)
        and np.isfinite(cy)
        and np.isfinite(major)
        and np.isfinite(minor)
        and major > 0.0
        and minor > 0.0
    ):
        return None
    return {
        "source_id": int(main["id"][row]),
        "x": cx,
        "y": cy,
        "a": major,
        "b": minor,
        "theta": theta,
        "area": math.pi * major * minor,
        "rmax": max(major, minor),
    }


def add_final_classification(main: fits.FITS_rec, rows_out: list[dict[str, object]]) -> None:
    candidate_by_id = {int(row["source_id"]): row for row in rows_out}
    preignored_ids: set[int] = set()
    candidate_geometries: dict[int, dict[str, float]] = {}
    for row in rows_out:
        sid = int(row["source_id"])
        reasons: list[str] = []
        area = float(row["used_aperture_area"])
        ap2_mag = float(row["ap2_mag"])
        major = float(row["aperture_major"])
        minor = float(row["aperture_minor"])
        axis_ratio = max(major, minor) / min(major, minor) if min(major, minor) > 0 else np.inf
        new_diff = float(row["new_absdiff"]) if row["new_absdiff"] != "" else np.nan
        if bool(row.get("small_footprint_large_aperture", False)):
            reasons.append("small_footprint_large_aperture")
        if np.isfinite(area) and area > CENTER_ONLY_APERTURE_AREA:
            reasons.append("used_aperture_area_gt_10000")
        if (
            np.isfinite(ap2_mag)
            and ap2_mag > FAINT_AP2_MAG_THRESHOLD
            and np.isfinite(area)
            and area > FAINT_APERTURE_AREA_THRESHOLD
        ):
            reasons.append("ap2_mag_gt_28_and_area_gt_900")
        if np.isfinite(axis_ratio) and axis_ratio > AXIS_RATIO_THRESHOLD:
            reasons.append("axis_ratio_gt_5")
        if not np.isfinite(new_diff):
            reasons.append("invalid_new_diff")
        elif new_diff > 1.5:
            reasons.append("new_absdiff_gt_1.5")

        row["axis_ratio"] = float(axis_ratio) if np.isfinite(axis_ratio) else np.nan
        row["prelim_ignore"] = bool(reasons)
        row["final_ignore_reasons"] = ";".join(reasons)
        if reasons:
            preignored_ids.add(sid)
        candidate_geometries[sid] = {
            "source_id": sid,
            "x": float(row["x_global"]),
            "y": float(row["y_global"]),
            "a": major,
            "b": minor,
            "theta": float(row["theta"]),
            "area": area,
            "rmax": max(major, minor),
        }

    pu_class = table_string_column(main, "pu_class")
    nchild = finite_float(main, "deblend_nChild", default=1.0)
    clean_rows = np.flatnonzero((nchild == 0) & (pu_class == "clean"))
    small_pool: list[dict[str, float]] = []
    seen_small_ids: set[int] = set()
    for row in clean_rows:
        geom = geometry_from_row(main, int(row))
        if geom is not None:
            small_pool.append(geom)
            seen_small_ids.add(int(geom["source_id"]))
    for sid, geom in candidate_geometries.items():
        if sid not in preignored_ids and sid not in seen_small_ids:
            small_pool.append(geom)
            seen_small_ids.add(sid)

    unit_points = unit_disk_points()
    containment_counts = 0
    for sid, big in candidate_geometries.items():
        if sid in preignored_ids:
            continue
        possible = []
        for small in small_pool:
            if int(small["source_id"]) == sid:
                continue
            if small["area"] >= big["area"]:
                continue
            dx = small["x"] - big["x"]
            dy = small["y"] - big["y"]
            if dx * dx + dy * dy > (small["rmax"] + big["rmax"]) ** 2:
                continue
            possible.append(small)
        possible.sort(key=lambda item: item["area"])
        for small in possible:
            frac = ellipse_contains_fraction(small=small, big=big, unit_points=unit_points)
            if frac >= CONTAINMENT_FRACTION_THRESHOLD:
                row = candidate_by_id[sid]
                old_reason = str(row.get("final_ignore_reasons", ""))
                extra = (
                    f"contains_smaller_kron_ge_0.80:small_id={int(small['source_id'])},"
                    f"frac={frac:.3f}"
                )
                row["final_ignore_reasons"] = f"{old_reason};{extra}" if old_reason else extra
                row["prelim_ignore"] = True
                containment_counts += 1
                break

    for row in rows_out:
        reasons = str(row.get("final_ignore_reasons", ""))
        new_diff = float(row["new_absdiff"]) if row["new_absdiff"] != "" else np.nan
        if reasons:
            row["final_class"] = "ignore"
            row["final_training_class"] = "ignore"
            row["final_visual_category"] = "final_ignore"
            row["final_color"] = "magenta"
        elif np.isfinite(new_diff) and new_diff < 1.0:
            row["final_class"] = "clean_candidate"
            row["final_training_class"] = "clean"
            row["final_visual_category"] = "new_diff_lt_1"
            row["final_color"] = "green"
        elif np.isfinite(new_diff) and new_diff <= 1.5:
            row["final_class"] = "center_only_candidate"
            row["final_training_class"] = "center_only"
            row["final_visual_category"] = "new_diff_1_1p5"
            row["final_color"] = "yellow"
        else:
            row["final_class"] = "ignore"
            row["final_training_class"] = "ignore"
            row["final_visual_category"] = "final_ignore"
            row["final_color"] = "magenta"
            if not reasons:
                row["final_ignore_reasons"] = "new_diff_outside_keep_bins"
    for row in rows_out:
        row["containment_ignore_count_for_band"] = containment_counts


def process_band(args: argparse.Namespace, band: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    meas_path = args.preprocessed_root / str(TRACT) / PATCH / "band_reference_pu_all" / band / f"meas-{band}-{TRACT}-{PATCH}.fits"
    archive_meas_path = args.coadd_root / str(TRACT) / band / PATCH / f"meas-{band}-{TRACT}-{PATCH}.fits"
    image_path = args.coadd_root / str(TRACT) / band / PATCH / f"calexp-{band}-{TRACT}-{PATCH}.fits"
    image, image_header = read_image(image_path)
    x_ltv = float(image_header.get("LTV1", -PATCH_X0))
    y_ltv = float(image_header.get("LTV2", -PATCH_Y0))

    rows_out: list[dict[str, object]] = []
    with fits.open(meas_path, memmap=True, ignore_missing_end=True) as pu_hdul, fits.open(
        archive_meas_path, memmap=True, ignore_missing_end=True
    ) as archive_hdul:
        main = pu_hdul[1].data
        archive_main = archive_hdul[1].data
        archive = archive_hdul[2].data
        footprint_refs = archive_hdul[3].data
        spans_table = archive_hdul[4].data
        heavy_table = archive_hdul[6].data if len(archive_hdul) > 6 else None

        ap2_flux = finite_float(main, "base_CircularApertureFlux_6_0_instFlux")
        catalog_kron_flux = finite_float(main, "ext_photometryKron_KronFlux_instFlux")
        ap2_mag = magnitude(ap2_flux)
        catalog_kron_mag = magnitude(catalog_kron_flux)
        old_diff = np.abs(ap2_mag - catalog_kron_mag)
        nchild = finite_float(main, "deblend_nChild", default=1.0)
        finite_center = np.isfinite(finite_float(main, "base_SdssCentroid_x")) & np.isfinite(
            finite_float(main, "base_SdssCentroid_y")
        )
        selected = np.flatnonzero((nchild == 0) & finite_center & np.isfinite(old_diff) & (old_diff > 1.0))

        archive_ids = np.asarray(archive_main["id"], dtype=np.int64)
        archive_order = np.argsort(archive_ids, kind="mergesort")
        selected_ids = np.asarray(main["id"][selected], dtype=np.int64)
        id_pos = np.searchsorted(archive_ids[archive_order], selected_ids)
        in_range = id_pos < archive_order.size
        archive_rows = np.full(selected.shape, -1, dtype=np.int64)
        matched_ids = np.zeros(selected.shape, dtype=bool)
        matched_ids[in_range] = archive_ids[archive_order[id_pos[in_range]]] == selected_ids[in_range]
        archive_rows[matched_ids] = archive_order[id_pos[matched_ids]]

        archive_index = ArchiveIndex.from_archive(archive)
        footprint_ids = np.full(selected.shape, -1, dtype=np.int64)
        footprint_ids[matched_ids] = np.asarray(archive_main["footprint"][archive_rows[matched_ids]], dtype=np.int64)
        footprint_lookup = archive_index.lookup(footprint_ids, archive_number=1, name=None)
        spanset_ids = np.full(selected.shape, -1, dtype=np.int64)
        valid_fp = footprint_lookup.found & (footprint_lookup.row0 >= 0) & (footprint_lookup.row0 < len(footprint_refs))
        spanset_ids[valid_fp] = np.asarray(footprint_refs["id"], dtype=np.int64)[footprint_lookup.row0[valid_fp]]
        spanset_lookup = archive_index.lookup(spanset_ids, archive_number=2, name="SpanSet")
        if heavy_table is not None:
            heavy_lookup = archive_index.lookup(footprint_ids, archive_number=4, name="HeavyFootprintF")
        else:
            heavy_lookup = ArchiveLookup(
                row0=np.full(selected.shape, -1, dtype=np.int64),
                nrows=np.zeros(selected.shape, dtype=np.int64),
                found=np.zeros(selected.shape, dtype=bool),
            )

        refit_matched = bool_column(main, "pu_refit_kron_radius_matched")
        for local_index, row in enumerate(selected):
            sid = int(main["id"][row])
            cx = float(main["base_SdssCentroid_x"][row])
            cy = float(main["base_SdssCentroid_y"][row])
            matched = bool(refit_matched[row])
            major, minor, theta, axis_mode = row_axis_params(main, int(row), matched)
            has_archive_row = bool(archive_rows[local_index] >= 0)
            has_heavy = bool(has_archive_row and heavy_lookup.found[local_index] and heavy_lookup.row0[local_index] >= 0)
            use_heavy = matched and has_heavy and heavy_table is not None and axis_mode == "refit_ellipse"
            if use_heavy:
                flux, aperture_pixels, source_pixels = measure_heavy(
                    spans_table=spans_table,
                    heavy_table=heavy_table,
                    span0=int(spanset_lookup.row0[local_index]),
                    nspan=int(spanset_lookup.nrows[local_index]),
                    heavy_row=int(heavy_lookup.row0[local_index]),
                    cx=cx,
                    cy=cy,
                    major=major,
                    minor=minor,
                    theta=theta,
                )
                measurement_surface = "heavy_footprint"
                fallback = False
            else:
                flux, aperture_pixels, source_pixels = measure_direct(
                    image=image,
                    x_ltv=x_ltv,
                    y_ltv=y_ltv,
                    cx=cx,
                    cy=cy,
                    major=major,
                    minor=minor,
                    theta=theta,
                )
                measurement_surface = "direct_image_fallback"
                fallback = True

            new_mag = magnitude(flux)
            new_diff = abs(float(ap2_mag[row]) - float(new_mag)) if np.isfinite(new_mag) else np.nan
            category = classify_new_diff(new_diff)
            used_ap_area = math.pi * major * minor if np.isfinite(major) and np.isfinite(minor) else np.nan
            used_shape_area = used_ap_area / 6.25 if np.isfinite(used_ap_area) else np.nan
            footprint_aperture_pixels, footprint_total_pixels = footprint_aperture_counts(
                spans_table=spans_table,
                span0=int(spanset_lookup.row0[local_index]),
                nspan=int(spanset_lookup.nrows[local_index]),
                cx=cx,
                cy=cy,
                major=major,
                minor=minor,
                theta=theta,
            )
            footprint_fill_fraction = (
                float(footprint_aperture_pixels) / float(used_ap_area)
                if np.isfinite(used_ap_area) and used_ap_area > 0.0
                else np.nan
            )
            small_footprint_large_aperture = (
                np.isfinite(new_diff)
                and new_diff < 1.5
                and np.isfinite(footprint_fill_fraction)
                and footprint_fill_fraction < SMALL_FOOTPRINT_FILL_THRESHOLD
            )
            center_only_large_area = bool(np.isfinite(used_ap_area) and used_ap_area > CENTER_ONLY_APERTURE_AREA)
            rows_out.append(
                {
                    "band": band,
                    "row_index": int(row),
                    "source_id": sid,
                    "parent": int(main["parent"][row]),
                    "x_global": cx,
                    "y_global": cy,
                    "x_image0": cx + x_ltv,
                    "y_image0": cy + y_ltv,
                    "ap2_flux": float(ap2_flux[row]),
                    "ap2_mag": float(ap2_mag[row]),
                    "catalog_kron_flux": float(catalog_kron_flux[row]),
                    "catalog_kron_mag": float(catalog_kron_mag[row]),
                    "old_absdiff": float(old_diff[row]),
                    "remeasured_kron_flux": float(flux),
                    "remeasured_kron_mag": float(new_mag) if np.isfinite(new_mag) else np.nan,
                    "new_absdiff": float(new_diff) if np.isfinite(new_diff) else np.nan,
                    "heavy_new_diff": float(new_diff) if measurement_surface == "heavy_footprint" and np.isfinite(new_diff) else np.nan,
                    "category": category,
                    "measurement_surface": measurement_surface,
                    "fallback": bool(fallback),
                    "refit_matched": bool(matched),
                    "has_heavy": bool(has_heavy),
                    "has_archive_row": bool(has_archive_row),
                    "axis_mode": axis_mode,
                    "aperture_major": float(major),
                    "aperture_minor": float(minor),
                    "theta": float(theta),
                    "used_aperture_area": float(used_ap_area) if np.isfinite(used_ap_area) else np.nan,
                    "used_shape_area": float(used_shape_area) if np.isfinite(used_shape_area) else np.nan,
                    "aperture_pixels": int(aperture_pixels),
                    "source_pixels": int(source_pixels),
                    "footprint_aperture_pixels": int(footprint_aperture_pixels),
                    "footprint_total_pixels": int(footprint_total_pixels),
                    "footprint_fill_fraction": float(footprint_fill_fraction)
                    if np.isfinite(footprint_fill_fraction)
                    else np.nan,
                    "small_footprint_large_aperture": bool(small_footprint_large_aperture),
                    "center_only_large_area": bool(center_only_large_area),
                }
            )

        add_final_classification(main, rows_out)

    summary = {
        "band": band,
        "meas_path": str(meas_path),
        "archive_meas_path": str(archive_meas_path),
        "image_path": str(image_path),
        "selected_old_diff_gt_1": len(rows_out),
        "category_counts": count_values(row["category"] for row in rows_out),
        "final_class_counts": count_values(row["final_class"] for row in rows_out),
        "final_training_class_counts": count_values(row["final_training_class"] for row in rows_out),
        "final_visual_counts": count_values(row["final_visual_category"] for row in rows_out),
        "ignore_reason_counts": count_ignore_reasons(rows_out),
        "surface_counts": count_values(row["measurement_surface"] for row in rows_out),
        "fallback_count": int(sum(bool(row["fallback"]) for row in rows_out)),
        "small_footprint_large_aperture_count": int(
            sum(bool(row["small_footprint_large_aperture"]) for row in rows_out)
        ),
        "center_only_large_area_count": int(sum(bool(row["center_only_large_area"]) for row in rows_out)),
    }
    return rows_out, summary


def count_values(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def count_ignore_reasons(rows: list[dict[str, object]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        reasons = str(row.get("final_ignore_reasons", ""))
        for reason in reasons.split(";"):
            reason = reason.strip()
            if not reason:
                continue
            key = reason.split(":", 1)[0]
            out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_reg(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Region file format: DS9 version 4.1",
        'global font="helvetica 9 normal roman" edit=1 move=1 delete=1 include=1',
        "image",
    ]
    for row in rows:
        category = str(row.get("final_visual_category", row["category"]))
        color = DS9_COLORS.get(category, "white")
        x = float(row["x_image0"]) + 1.0
        y = float(row["y_image0"]) + 1.0
        major = float(row["aperture_major"])
        minor = float(row["aperture_minor"])
        theta_deg = math.degrees(float(row["theta"]))
        label = f'{row["source_id"]} {row.get("final_class", category)}'
        if bool(row["fallback"]):
            label += " fallback"
        reasons = str(row.get("final_ignore_reasons", ""))
        if reasons:
            label += f" {reasons}"
        dash = " dash=1" if bool(row["fallback"]) else ""
        center_only = bool(row.get("center_only_large_area", False))
        if center_only:
            lines.append(f"point({x:.3f},{y:.3f}) # point=cross color={color}{dash} text={{{label} center-only-large-area}}")
        elif np.isfinite(major) and np.isfinite(minor) and major > 0.0 and minor > 0.0:
            lines.append(
                f"ellipse({x:.3f},{y:.3f},{major:.3f},{minor:.3f},{theta_deg:.3f}) "
                f"# color={color}{dash} text={{{label}}}"
            )
        else:
            lines.append(f"point({x:.3f},{y:.3f}) # point=cross color={color}{dash} text={{{label}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def write_png(path: Path, rgb: np.ndarray) -> None:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("write_png expects uint8 RGB image")
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    data = b"\x89PNG\r\n\x1a\n"
    data += png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    data += png_chunk(b"IDAT", zlib.compress(raw, level=6))
    data += png_chunk(b"IEND", b"")
    path.write_bytes(data)


def zscale_rgb(image: np.ndarray, max_size: int) -> tuple[np.ndarray, float]:
    finite = image[np.isfinite(image)]
    if finite.size:
        vmin, vmax = ZScaleInterval(contrast=0.25).get_limits(finite)
    else:
        vmin, vmax = 0.0, 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = np.nanpercentile(image, [1, 99])
    scale = min(1.0, float(max_size) / float(max(image.shape)))
    if scale < 1.0:
        out_h = max(1, int(round(image.shape[0] * scale)))
        out_w = max(1, int(round(image.shape[1] * scale)))
        yy = np.minimum((np.arange(out_h) / scale).astype(int), image.shape[0] - 1)
        xx = np.minimum((np.arange(out_w) / scale).astype(int), image.shape[1] - 1)
        small = image[yy[:, None], xx[None, :]]
    else:
        small = image
    gray = np.clip((small.astype(np.float64) - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb = np.repeat((gray * 255.0).astype(np.uint8)[..., None], 3, axis=2)
    return rgb, scale


def draw_point(rgb: np.ndarray, x: float, y: float, color: tuple[int, int, int], radius: int = 2) -> None:
    xi = int(round(x))
    yi = int(round(y))
    h, w, _ = rgb.shape
    for yy in range(max(0, yi - radius), min(h, yi + radius + 1)):
        for xx in range(max(0, xi - radius), min(w, xi + radius + 1)):
            rgb[yy, xx] = color


def draw_cross(rgb: np.ndarray, x: float, y: float, color: tuple[int, int, int], radius: int = 5) -> None:
    xi = int(round(x))
    yi = int(round(y))
    h, w, _ = rgb.shape
    for dx in range(-radius, radius + 1):
        xx = xi + dx
        if 0 <= xx < w and 0 <= yi < h:
            rgb[yi, xx] = color
    for dy in range(-radius, radius + 1):
        yy = yi + dy
        if 0 <= xi < w and 0 <= yy < h:
            rgb[yy, xi] = color


def draw_ellipse(rgb: np.ndarray, row: dict[str, object], scale: float) -> None:
    category = str(row.get("final_visual_category", row["category"]))
    color = COLORS.get(category, (255, 255, 255))
    x0 = float(row["x_image0"]) * scale
    y0 = float(row["y_image0"]) * scale
    major = float(row["aperture_major"]) * scale
    minor = float(row["aperture_minor"]) * scale
    theta = float(row["theta"])
    center_only = bool(row.get("center_only_large_area", False))
    if center_only:
        draw_cross(rgb, x0, y0, color, radius=5)
    elif not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(major) and np.isfinite(minor)):
        draw_point(rgb, x0, y0, color, radius=3)
    elif major < 1.5 or minor < 1.5:
        draw_point(rgb, x0, y0, color, radius=3)
    else:
        c = math.cos(theta)
        s = math.sin(theta)
        previous: tuple[int, int] | None = None
        for t in np.linspace(0.0, 2.0 * math.pi, 96, endpoint=True):
            u = major * math.cos(float(t))
            v = minor * math.sin(float(t))
            x = x0 + u * c - v * s
            y = y0 + u * s + v * c
            point = (int(round(x)), int(round(y)))
            if previous is not None:
                draw_line(rgb, previous[0], previous[1], point[0], point[1], color)
            previous = point


def draw_line(rgb: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    h, w, _ = rgb.shape
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        if 0 <= x < w and 0 <= y < h:
            rgb[y, x] = color
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def write_png_overlay(path: Path, image_path: Path, rows: list[dict[str, object]], max_size: int) -> None:
    image, _ = read_image(image_path)
    rgb, scale = zscale_rgb(image, max_size=max_size)
    for row in rows:
        draw_ellipse(rgb, row, scale)
    write_png(path, rgb)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remeasure old ap2-Kron outliers using refit apertures.")
    parser.add_argument("--bands", nargs="+", default=["HSC-I", "HSC-Y"])
    parser.add_argument("--preprocessed-root", type=Path, default=Path("/nvme0/zc/scarlet/preprocessed"))
    parser.add_argument("--coadd-root", type=Path, default=Path("/nvme0/zc/scarlet"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/czh23/CELLECT/diagnostics/output/refit_kron_remeasured_photometry_4_5"),
    )
    parser.add_argument("--png-max-size", type=int, default=1800)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for band in args.bands:
        rows, summary = process_band(args, band)
        summaries.append(summary)
        stem = f"{TRACT}_{PATCH.replace(',', '_')}_{band}_remeasured_refit_kron"
        csv_path = args.output_dir / f"{stem}.csv"
        reg_path = args.output_dir / f"{stem}.reg"
        png_path = args.output_dir / f"{stem}.png"
        image_path = args.coadd_root / str(TRACT) / band / PATCH / f"calexp-{band}-{TRACT}-{PATCH}.fits"
        write_csv(csv_path, rows)
        write_reg(reg_path, rows)
        write_png_overlay(png_path, image_path, rows, max_size=int(args.png_max_size))
        summary["outputs"] = {"csv": str(csv_path), "reg": str(reg_path), "png": str(png_path)}
        print(json.dumps(summary, sort_keys=True))
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
