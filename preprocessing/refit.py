"""Kron refit attachment and ellipse construction for preprocessing v3."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from .utils.catalog import first_finite_column, source_ids, source_xy
from .utils.geometry import EllipseGeometry, ellipse_axes_from_moments


CENTRAL_PIXEL_ER = 0.38259771140356325
KRON_FLAGS = (
    "ext_photometryKron_KronFlux_flag_bad_radius",
    "ext_photometryKron_KronFlux_flag_used_psf_radius",
    "ext_photometryKron_KronFlux_flag_used_minimum_radius",
    "ext_photometryKron_KronFlux_flag_small_radius",
)


@dataclass(frozen=True)
class RefitConfig:
    radius_column: str = "proxy_nan0_flux_aperture_radius"
    good_column: str = "proxy_nan0_good"
    output_column: str = "pu_refit_kron_radius"
    ellipse_sigma: float = 1.0
    min_axis: float = 1.5
    require_match: bool = True
    use_smaller_official: bool = True
    official_aperture_scale: float = 2.5


@dataclass(frozen=True)
class DirectRefitConfig:
    """Configuration for measuring refit Kron radii directly from meas FITS."""

    mag_min: float = 10.0
    mag_max: float = 35.0
    input_zeropoint: float = 27.0
    output_zeropoint: float = 31.4
    n_sigma_for_radius: float = 6.0
    n_radius_for_flux: float = 2.5
    chunk_pixel_limit: int = 10_000_000
    include_sky: bool = False
    include_non_primary: bool = True
    leaf_only: bool = True
    include_shape_flagged: bool = False
    include_centroid_flagged: bool = False
    disable_heavy_footprints: bool = False
    allow_missing_heavy_footprints: bool = True
    allow_missing_ltv: bool = False


@dataclass(frozen=True)
class ArchiveLookup:
    row_index: np.ndarray
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
            names=np.asarray([decode_archive_string(value) for value in archive["name"]]),
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

        row_index = np.full(target.shape, -1, dtype=np.int64)
        found = np.zeros(target.shape, dtype=bool)
        if sorted_ids.size:
            pos = np.searchsorted(sorted_ids, target)
            in_range = pos < sorted_ids.size
            matched = np.zeros(target.shape, dtype=bool)
            matched[in_range] = sorted_ids[pos[in_range]] == target[in_range]
            found = matched
            row_index[found] = sorted_rows[pos[found]]

        row0 = np.full(target.shape, -1, dtype=np.int64)
        nrows = np.zeros(target.shape, dtype=np.int64)
        row0[found] = self.row0[row_index[found]]
        nrows[found] = self.nrows[row_index[found]]
        return ArchiveLookup(row_index=row_index, row0=row0, nrows=nrows, found=found)


@dataclass(frozen=True)
class ReferenceImageContext:
    path: Path
    image: np.ndarray
    header: fits.Header
    x_ltv: int
    y_ltv: int


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def decode_archive_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii", errors="ignore").strip()
    arr = np.asarray(value)
    if arr.dtype.kind in {"S", "U"}:
        return "".join(str(item.decode("ascii", errors="ignore") if isinstance(item, bytes) else item) for item in arr).strip()
    return str(value).strip()


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def flags_array(main: fits.FITS_rec) -> np.ndarray:
    return np.asarray(main["flags"], dtype=bool)


def flag_name_map(header: fits.Header) -> dict[str, int]:
    return {str(header[key]): int(key[5:]) - 1 for key in header if str(key).startswith("TFLAG")}


def magnitude_from_zp27_flux(flux_zp27: np.ndarray, *, input_zeropoint: float) -> np.ndarray:
    flux = np.asarray(flux_zp27, dtype=np.float64)
    mag = np.full(flux.shape, np.nan, dtype=np.float64)
    positive = np.isfinite(flux) & (flux > 0.0)
    mag[positive] = float(input_zeropoint) - 2.5 * np.log10(flux[positive])
    return mag


def magnitude_from_njy_flux(flux_njy: np.ndarray, *, output_zeropoint: float) -> np.ndarray:
    flux = np.asarray(flux_njy, dtype=np.float64)
    mag = np.full(flux.shape, np.nan, dtype=np.float64)
    positive = np.isfinite(flux) & (flux > 0.0)
    mag[positive] = float(output_zeropoint) - 2.5 * np.log10(flux[positive])
    return mag


def read_reference_image_array_and_header(path: Path | str) -> tuple[np.ndarray, fits.Header]:
    with fits.open(Path(path), memmap=True, ignore_missing_end=True) as hdul:
        candidates: list[int] = []
        if "IMAGE" in hdul:
            candidates.append(hdul.index_of("IMAGE"))
        for index, hdu in enumerate(hdul):
            class_name = hdu.__class__.__name__
            if class_name not in {"PrimaryHDU", "ImageHDU", "CompImageHDU"}:
                continue
            if int(hdu.header.get("NAXIS", 0)) < 2:
                continue
            if index not in candidates:
                candidates.append(index)
        for index in candidates:
            hdu = hdul[index]
            if hdu.data is None:
                continue
            return np.asarray(hdu.data, dtype=np.float32), hdu.header.copy()
    raise ValueError(f"reference image has no image data HDU: {path}")


def read_reference_image_context(path: Path | str, *, allow_missing_ltv: bool = False) -> ReferenceImageContext:
    image, header = read_reference_image_array_and_header(path)
    x_ltv = header.get("LTV1")
    y_ltv = header.get("LTV2")
    if x_ltv is None or y_ltv is None:
        if not allow_missing_ltv:
            raise ValueError(f"reference image must have LTV1/LTV2 for full-pixel mapping: {path}")
        x_ltv = 0
        y_ltv = 0
    return ReferenceImageContext(
        path=Path(path),
        image=image,
        header=header,
        x_ltv=int(round(float(x_ltv))),
        y_ltv=int(round(float(y_ltv))),
    )


def image_positions_from_reference(
    *,
    header: fits.Header,
    ra_rad: np.ndarray,
    dec_rad: np.ndarray,
    fallback_x: np.ndarray,
    fallback_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(ra_rad) & np.isfinite(dec_rad)
    x = np.asarray(fallback_x, dtype=np.float64).copy()
    y = np.asarray(fallback_y, dtype=np.float64).copy()
    try:
        wcs = WCS(header)
        if np.any(finite):
            x_pix, y_pix = wcs.all_world2pix(np.degrees(ra_rad[finite]), np.degrees(dec_rad[finite]), 1)
            x[finite] = x_pix
            y[finite] = y_pix
    except Exception:
        pass
    return x, y


def select_refit_rows(main: fits.FITS_rec, header: fits.Header, config: DirectRefitConfig) -> tuple[np.ndarray, dict[str, Any]]:
    flags = flags_array(main)
    flag_names = flag_name_map(header)

    def flag(name: str) -> np.ndarray:
        index = flag_names.get(name)
        if index is None:
            return np.zeros(len(main), dtype=bool)
        return flags[:, index]

    psf_flux = np.asarray(main["base_PsfFlux_instFlux"], dtype=np.float64)
    psf_mag = magnitude_from_zp27_flux(psf_flux, input_zeropoint=config.input_zeropoint)
    xx = np.asarray(main["base_SdssShape_xx"], dtype=np.float64)
    yy = np.asarray(main["base_SdssShape_yy"], dtype=np.float64)
    xy = np.asarray(main["base_SdssShape_xy"], dtype=np.float64)
    det = xx * yy - xy * xy
    shape_ok = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(xy) & (xx > 0.0) & (yy > 0.0) & (det > 0.0)
    mask = np.isfinite(psf_mag) & (psf_mag >= config.mag_min) & (psf_mag < config.mag_max)
    if not config.include_non_primary:
        mask &= flag("detect_isPrimary")
    if not config.include_shape_flagged:
        mask &= ~flag("base_SdssShape_flag") & shape_ok
    if not config.include_centroid_flagged:
        mask &= ~flag("base_SdssCentroid_flag")
    if not config.include_sky:
        mask &= ~(flag("merge_peak_sky") | flag("merge_footprint_sky"))
    if config.leaf_only:
        if "deblend_nChild" not in main.columns.names:
            raise KeyError("leaf_only requires deblend_nChild column")
        mask &= np.asarray(main["deblend_nChild"], dtype=np.int64) == 0
    rows = np.flatnonzero(mask).astype(np.int64)
    return rows, {
        "mode": "official_psf_mag",
        "mag_min": float(config.mag_min),
        "mag_max": float(config.mag_max),
        "selected_rows": int(rows.size),
        "include_non_primary": bool(config.include_non_primary),
        "leaf_only": bool(config.leaf_only),
        "include_shape_flagged": bool(config.include_shape_flagged),
        "include_centroid_flagged": bool(config.include_centroid_flagged),
        "include_sky": bool(config.include_sky),
    }


def build_refit_source_arrays(
    *,
    main: fits.FITS_rec,
    header: fits.Header,
    rows: np.ndarray,
    reference_context: ReferenceImageContext,
    config: DirectRefitConfig,
) -> dict[str, np.ndarray]:
    flux_scale_to_njy = float(10.0 ** ((config.output_zeropoint - config.input_zeropoint) / 2.5))
    flags = flags_array(main)
    flag_names = flag_name_map(header)

    def flag(name: str) -> np.ndarray:
        index = flag_names.get(name)
        if index is None:
            return np.zeros(len(main), dtype=bool)
        return flags[:, index]

    xx = np.asarray(main["base_SdssShape_xx"][rows], dtype=np.float64)
    yy = np.asarray(main["base_SdssShape_yy"][rows], dtype=np.float64)
    xy = np.asarray(main["base_SdssShape_xy"][rows], dtype=np.float64)
    axes = ellipse_axes_from_moments(xx, yy, xy)
    psf_flux_zp27 = np.asarray(main["base_PsfFlux_instFlux"][rows], dtype=np.float64)
    psf_flux_njy = psf_flux_zp27 * flux_scale_to_njy
    centroid_x = np.asarray(main["base_SdssCentroid_x"][rows], dtype=np.float64)
    centroid_y = np.asarray(main["base_SdssCentroid_y"][rows], dtype=np.float64)
    x_image, y_image = image_positions_from_reference(
        header=reference_context.header,
        ra_rad=np.asarray(main["coord_ra"][rows], dtype=np.float64),
        dec_rad=np.asarray(main["coord_dec"][rows], dtype=np.float64),
        fallback_x=centroid_x,
        fallback_y=centroid_y,
    )
    sources: dict[str, np.ndarray] = {
        "row_index": np.asarray(rows, dtype=np.int64),
        "source_id": np.asarray(main["id"][rows], dtype=np.int64),
        "parent": np.asarray(main["parent"][rows], dtype=np.int64),
        "n_child": np.asarray(main["deblend_nChild"][rows], dtype=np.int64),
        "footprint_id": np.asarray(main["footprint"][rows], dtype=np.int64),
        "footprint_area": np.asarray(main["base_FootprintArea_value"][rows], dtype=np.int64),
        "ra_deg": np.degrees(np.asarray(main["coord_ra"][rows], dtype=np.float64)),
        "dec_deg": np.degrees(np.asarray(main["coord_dec"][rows], dtype=np.float64)),
        "x_image": x_image,
        "y_image": y_image,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "shape_xx": xx,
        "shape_yy": yy,
        "shape_xy": xy,
        "axis_a": axes["a"],
        "axis_b": axes["b"],
        "theta_rad": axes["theta_rad"],
        "theta_deg": axes["theta_deg"],
        "initial_determinant_radius": axes["determinant_radius"],
        "shape_valid": axes["valid"],
        "catalog_kron_radius": np.asarray(main["ext_photometryKron_KronFlux_radius"][rows], dtype=np.float64),
        "catalog_kron_radius_for_radius": np.asarray(main["ext_photometryKron_KronFlux_radius_for_radius"][rows], dtype=np.float64),
        "catalog_kron_inst_flux": np.asarray(main["ext_photometryKron_KronFlux_instFlux"][rows], dtype=np.float64),
        "catalog_kron_inst_flux_err": np.asarray(main["ext_photometryKron_KronFlux_instFluxErr"][rows], dtype=np.float64),
        "catalog_psf_flux_zp27": psf_flux_zp27,
        "catalog_psf_flux_njy": psf_flux_njy,
        "official_psf_ab_mag": magnitude_from_njy_flux(psf_flux_njy, output_zeropoint=config.output_zeropoint),
        "detect_is_primary": flag("detect_isPrimary")[rows],
        "merge_peak_sky": flag("merge_peak_sky")[rows],
        "merge_footprint_sky": flag("merge_footprint_sky")[rows],
        "base_sdss_centroid_flag": flag("base_SdssCentroid_flag")[rows],
        "base_sdss_shape_flag": flag("base_SdssShape_flag")[rows],
        "base_psf_flux_flag": flag("base_PsfFlux_flag")[rows],
    }
    sources.update({name: flag(name)[rows] for name in KRON_FLAGS})
    n = len(rows)
    sources["footprint_archive_found"] = np.zeros(n, dtype=bool)
    sources["spanset_archive_found"] = np.zeros(n, dtype=bool)
    sources["heavy_archive_found"] = np.zeros(n, dtype=bool)
    sources["footprint_ref_row"] = np.full(n, -1, dtype=np.int64)
    sources["spanset_id"] = np.full(n, -1, dtype=np.int64)
    sources["span0"] = np.full(n, -1, dtype=np.int64)
    sources["nspan"] = np.zeros(n, dtype=np.int64)
    sources["heavy_row"] = np.full(n, -1, dtype=np.int64)
    sources["processable"] = np.zeros(n, dtype=bool)
    sources["measurement_surface"] = np.full(n, "none", dtype=object)
    sources["status"] = np.full(n, "pending", dtype=object)
    return sources


def prepare_heavy_lookups(
    *,
    sources: dict[str, np.ndarray],
    archive_index: ArchiveIndex,
    footprint_refs: fits.FITS_rec,
    use_heavy_footprints: bool,
) -> None:
    footprint_ids = sources["footprint_id"]
    footprint_lookup = archive_index.lookup(footprint_ids, archive_number=1, name=None)
    sources["footprint_archive_found"] = footprint_lookup.found
    sources["footprint_ref_row"] = footprint_lookup.row0
    valid_ref = footprint_lookup.found & (footprint_lookup.row0 >= 0) & (footprint_lookup.row0 < len(footprint_refs))
    spanset_ids = np.full(len(footprint_ids), -1, dtype=np.int64)
    spanset_ids[valid_ref] = np.asarray(footprint_refs["id"], dtype=np.int64)[footprint_lookup.row0[valid_ref]]
    sources["spanset_id"] = spanset_ids

    spanset_lookup = archive_index.lookup(spanset_ids, archive_number=2, name="SpanSet")
    sources["spanset_archive_found"] = spanset_lookup.found & valid_ref
    sources["span0"] = spanset_lookup.row0
    sources["nspan"] = spanset_lookup.nrows

    if use_heavy_footprints:
        heavy_lookup = archive_index.lookup(footprint_ids, archive_number=4, name="HeavyFootprintF")
        sources["heavy_archive_found"] = heavy_lookup.found
        sources["heavy_row"] = heavy_lookup.row0
    else:
        sources["heavy_archive_found"] = np.zeros(len(footprint_ids), dtype=bool)
        sources["heavy_row"] = np.full(len(footprint_ids), -1, dtype=np.int64)

    finite_centroid = np.isfinite(sources["centroid_x"]) & np.isfinite(sources["centroid_y"])
    processable = (
        sources["shape_valid"]
        & finite_centroid
        & sources["footprint_archive_found"]
        & sources["spanset_archive_found"]
        & (sources["nspan"] > 0)
    )
    has_heavy = sources["heavy_archive_found"] & (sources["heavy_row"] >= 0)
    sources["processable"] = processable
    sources["measurement_surface"][:] = "none"
    sources["measurement_surface"][processable & has_heavy] = "heavyfootprintf"
    sources["measurement_surface"][processable & ~has_heavy] = "direct_footprint_reference_image"
    status = sources["status"]
    status[:] = "ok"
    status[~sources["shape_valid"]] = "skip_invalid_shape"
    status[~finite_centroid] = "skip_invalid_centroid"
    status[~sources["footprint_archive_found"]] = "skip_no_footprint_archive"
    status[~sources["spanset_archive_found"]] = "skip_no_spanset_archive"
    status[~processable & (sources["nspan"] <= 0)] = "skip_empty_spanset"
    status[processable] = "ok"


def source_blocks(positions: np.ndarray, pixel_counts: np.ndarray, chunk_pixel_limit: int) -> list[np.ndarray]:
    if chunk_pixel_limit <= 0:
        return [positions]
    blocks: list[np.ndarray] = []
    start = 0
    running = 0
    for index, count in enumerate(pixel_counts):
        count_int = int(count)
        if index > start and running + count_int > chunk_pixel_limit:
            blocks.append(positions[start:index])
            start = index
            running = 0
        running += count_int
    if start < len(positions):
        blocks.append(positions[start:])
    return blocks


def init_measurement_arrays(n: int) -> dict[str, np.ndarray]:
    float_keys = [
        "heavyfp_nan0_radius_for_radius",
        "heavyfp_nan0_sum_i",
        "heavyfp_nan0_sum_radius_i",
        "heavyfp_nan0_positive_sum_i",
        "heavyfp_nan0_negative_sum_i",
        "heavyfp_nan0_raw_first_moment",
        "heavyfp_nan0_candidate_radius",
        "heavyfp_nan0_determine_radius_returned_radius",
        "heavyfp_nan0_flux_aperture_radius",
        "heavyfp_nan0_original_determine_radius_returned_radius",
        "heavyfp_nan0_original_flux_aperture_radius",
        "heavyfp_nan0_computed_span_area",
    ]
    int_keys = ["heavy_value_count", "finite_value_count", "nonfinite_value_count", "aperture_pixel_count", "central_pixel_count"]
    out: dict[str, np.ndarray] = {key: np.full(n, np.nan, dtype=np.float64) for key in float_keys}
    out.update({key: np.zeros(n, dtype=np.int64) for key in int_keys})
    out["heavyfp_nan0_candidate_le_initial_radius"] = np.zeros(n, dtype=bool)
    out["heavyfp_nan0_good"] = np.zeros(n, dtype=bool)
    out["heavyfp_nan0_fallback_large_aperture"] = np.zeros(n, dtype=bool)
    out["span_area_matches_catalog"] = np.zeros(n, dtype=bool)
    return out


def measure_refit_block(
    *,
    positions: np.ndarray,
    sources: dict[str, np.ndarray],
    spans_table: fits.FITS_rec,
    heavy_table: fits.FITS_rec | None,
    reference_context: ReferenceImageContext,
    value_source: str,
    flux_scale_to_njy: float,
    n_sigma_for_radius: float,
) -> dict[str, np.ndarray | int]:
    n_block = len(positions)
    nspan = np.asarray(sources["nspan"][positions], dtype=np.int64)
    span0 = np.asarray(sources["span0"][positions], dtype=np.int64)
    span_src = np.repeat(np.arange(n_block, dtype=np.int64), nspan)
    total_spans = int(np.sum(nspan))
    source_span_start = np.repeat(np.cumsum(nspan) - nspan, nspan)
    span_within_source = np.arange(total_spans, dtype=np.int64) - source_span_start
    span_indices = np.repeat(span0, nspan) + span_within_source
    spans = spans_table[span_indices]
    span_y = np.asarray(spans["y"], dtype=np.int64)
    span_x0 = np.asarray(spans["x0"], dtype=np.int64)
    span_x1 = np.asarray(spans["x1"], dtype=np.int64)
    span_width = span_x1 - span_x0 + 1

    pix_span = np.repeat(np.arange(total_spans, dtype=np.int64), span_width)
    total_pixels = int(np.sum(span_width))
    span_pixel_start = np.repeat(np.cumsum(span_width) - span_width, span_width)
    pix_within_span = np.arange(total_pixels, dtype=np.int64) - span_pixel_start
    all_x = span_x0[pix_span] + pix_within_span
    all_y = span_y[pix_span]
    all_src = span_src[pix_span]

    if value_source == "heavyfootprintf":
        if heavy_table is None:
            raise ValueError("HeavyFootprint measurement requested without heavy_table")
        heavy_rows = np.asarray(sources["heavy_row"][positions], dtype=np.int64)
        values_zp27 = np.concatenate(heavy_table["image"][heavy_rows]).astype(np.float64, copy=False)
        if values_zp27.size != total_pixels:
            raise ValueError(f"HeavyFootprint value count mismatch: values={values_zp27.size} pixels={total_pixels}")
        finite = np.isfinite(values_zp27)
        all_values = np.where(finite, values_zp27 * float(flux_scale_to_njy), 0.0)
    elif value_source == "direct_footprint_reference_image":
        local_x = all_x + int(reference_context.x_ltv)
        local_y = all_y + int(reference_context.y_ltv)
        in_reference = (
            (local_x >= 0)
            & (local_y >= 0)
            & (local_x < reference_context.image.shape[1])
            & (local_y < reference_context.image.shape[0])
        )
        sampled = np.full(total_pixels, np.nan, dtype=np.float64)
        if np.any(in_reference):
            sampled[in_reference] = np.asarray(reference_context.image[local_y[in_reference], local_x[in_reference]], dtype=np.float64)
        finite = np.isfinite(sampled)
        all_values = np.where(finite, sampled, 0.0)
    else:
        raise ValueError(f"unsupported value source: {value_source}")

    cx = sources["centroid_x"][positions]
    cy = sources["centroid_y"][positions]
    axis_a = sources["axis_a"][positions]
    axis_b = sources["axis_b"][positions]
    theta = sources["theta_rad"][positions]
    r_det = sources["initial_determinant_radius"][positions]

    dx = all_x.astype(np.float64) - cx[all_src]
    dy = all_y.astype(np.float64) - cy[all_src]
    cos_theta = np.cos(theta[all_src])
    sin_theta = np.sin(theta[all_src])
    du = dx * cos_theta + dy * sin_theta
    dv = -dx * sin_theta + dy * cos_theta

    scaled_a = float(n_sigma_for_radius) * axis_a[all_src]
    scaled_b = float(n_sigma_for_radius) * axis_b[all_src]
    inside = (du / scaled_a) ** 2 + (dv / scaled_b) ** 2 <= 1.0

    axis_ratio_a_over_b = axis_a[all_src] / axis_b[all_src]
    radius_major = np.hypot(du, dv * axis_ratio_a_over_b)
    center_distance = np.hypot(dx, dy)
    central = center_distance < 0.5
    if np.any(central):
        radius_major[central] = np.hypot(
            radius_major[central],
            CENTRAL_PIXEL_ER * (1.0 + center_distance[central] / math.sqrt(2.0)),
        )

    included_src = all_src[inside]
    included_values = all_values[inside]
    included_radius = radius_major[inside]
    sum_i = np.bincount(included_src, weights=included_values, minlength=n_block)
    sum_r_i = np.bincount(included_src, weights=included_radius * included_values, minlength=n_block)
    positive = np.where(included_values > 0.0, included_values, 0.0)
    negative = np.where(included_values < 0.0, included_values, 0.0)
    positive_sum = np.bincount(included_src, weights=positive, minlength=n_block)
    negative_sum = np.bincount(included_src, weights=negative, minlength=n_block)

    span_area = np.bincount(span_src, weights=span_width, minlength=n_block)
    finite_count = np.bincount(all_src, weights=finite.astype(np.int64), minlength=n_block)
    nonfinite_count = np.bincount(all_src, weights=(~finite).astype(np.int64), minlength=n_block)
    aperture_pixel_count = np.bincount(included_src, minlength=n_block)
    central_pixel_count = np.bincount(all_src[central], minlength=n_block)

    raw_first_moment = np.full(n_block, np.nan, dtype=np.float64)
    candidate_radius = np.full(n_block, np.nan, dtype=np.float64)
    good = (sum_i > 0.0) & (sum_r_i > 0.0)
    raw_first_moment[good] = sum_r_i[good] / sum_i[good]
    candidate_radius[good] = raw_first_moment[good] * np.sqrt(axis_b[good] / axis_a[good])

    radius_for_radius = float(n_sigma_for_radius) * r_det
    candidate_le_initial = good & (candidate_radius <= r_det)
    returned_radius = np.full(n_block, np.nan, dtype=np.float64)
    returned_radius[good] = np.where(candidate_le_initial[good], radius_for_radius[good], candidate_radius[good])

    return {
        "heavyfp_nan0_radius_for_radius": radius_for_radius,
        "heavyfp_nan0_sum_i": sum_i,
        "heavyfp_nan0_sum_radius_i": sum_r_i,
        "heavyfp_nan0_positive_sum_i": positive_sum,
        "heavyfp_nan0_negative_sum_i": negative_sum,
        "heavyfp_nan0_raw_first_moment": raw_first_moment,
        "heavyfp_nan0_candidate_radius": candidate_radius,
        "heavyfp_nan0_determine_radius_returned_radius": returned_radius,
        "heavyfp_nan0_flux_aperture_radius": np.full(n_block, np.nan, dtype=np.float64),
        "heavyfp_nan0_computed_span_area": span_area.astype(np.float64),
        "heavy_value_count": span_area.astype(np.int64),
        "finite_value_count": finite_count.astype(np.int64),
        "nonfinite_value_count": nonfinite_count.astype(np.int64),
        "aperture_pixel_count": aperture_pixel_count.astype(np.int64),
        "central_pixel_count": central_pixel_count.astype(np.int64),
        "heavyfp_nan0_candidate_le_initial_radius": candidate_le_initial,
        "heavyfp_nan0_good": good,
        "span_area_matches_catalog": span_area.astype(np.int64) == sources["footprint_area"][positions],
        "total_expanded_pixels": int(total_pixels),
        "max_pixels_per_source": int(np.max(span_area)) if span_area.size else 0,
    }


def measure_refit_sources(
    *,
    sources: dict[str, np.ndarray],
    spans_table: fits.FITS_rec,
    heavy_table: fits.FITS_rec | None,
    reference_context: ReferenceImageContext,
    config: DirectRefitConfig,
) -> dict[str, np.ndarray | int]:
    n = len(sources["row_index"])
    out = init_measurement_arrays(n)
    process_positions = np.flatnonzero(sources["processable"])
    if process_positions.size == 0:
        return {**out, "total_expanded_pixels": 0, "max_pixels_per_source": 0, "n_chunks": 0}

    blocks: list[tuple[str, np.ndarray]] = []
    for surface in ("heavyfootprintf", "direct_footprint_reference_image"):
        surface_positions = process_positions[np.asarray(sources["measurement_surface"][process_positions], dtype=object) == surface]
        if surface_positions.size == 0:
            continue
        pixel_counts = np.asarray(sources["footprint_area"][surface_positions], dtype=np.int64)
        blocks.extend((surface, block) for block in source_blocks(surface_positions, pixel_counts, config.chunk_pixel_limit))

    total_pixels = 0
    max_pixels = 0
    flux_scale_to_njy = float(10.0 ** ((config.output_zeropoint - config.input_zeropoint) / 2.5))
    for surface, block in blocks:
        block_result = measure_refit_block(
            positions=block,
            sources=sources,
            spans_table=spans_table,
            heavy_table=heavy_table,
            reference_context=reference_context,
            value_source=surface,
            flux_scale_to_njy=flux_scale_to_njy,
            n_sigma_for_radius=config.n_sigma_for_radius,
        )
        total_pixels += int(block_result["total_expanded_pixels"])
        max_pixels = max(max_pixels, int(block_result["max_pixels_per_source"]))
        for key, value in block_result.items():
            if key in out:
                out[key][block] = value

    original_returned_radius = out["heavyfp_nan0_determine_radius_returned_radius"].copy()
    out["heavyfp_nan0_flux_aperture_radius"] = out["heavyfp_nan0_determine_radius_returned_radius"] * float(config.n_radius_for_flux)
    original_flux_aperture_radius = out["heavyfp_nan0_flux_aperture_radius"].copy()
    official_radius = np.asarray(sources["catalog_kron_radius"], dtype=np.float64)
    fallback = (
        out["heavyfp_nan0_good"]
        & np.isfinite(official_radius)
        & (official_radius > 0.0)
        & np.isfinite(out["heavyfp_nan0_flux_aperture_radius"])
        & (out["heavyfp_nan0_flux_aperture_radius"] > config.n_radius_for_flux * official_radius)
    )
    if np.any(fallback):
        out["heavyfp_nan0_flux_aperture_radius"][fallback] = config.n_radius_for_flux * official_radius[fallback]
        out["heavyfp_nan0_determine_radius_returned_radius"][fallback] = out["heavyfp_nan0_flux_aperture_radius"][fallback] / float(config.n_radius_for_flux)
    out["heavyfp_nan0_original_determine_radius_returned_radius"] = original_returned_radius
    out["heavyfp_nan0_original_flux_aperture_radius"] = original_flux_aperture_radius
    out["heavyfp_nan0_fallback_large_aperture"] = fallback
    return {**out, "total_expanded_pixels": int(total_pixels), "max_pixels_per_source": int(max_pixels), "n_chunks": int(len(blocks))}


def ratio_or_none(left: Any, right: Any) -> float | None:
    left_value = finite_float(left)
    right_value = finite_float(right)
    if left_value is None or right_value is None or right_value == 0.0:
        return None
    return (left_value - right_value) / right_value


def build_refit_rows(sources: dict[str, np.ndarray], measurement: dict[str, np.ndarray | int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(len(sources["row_index"])):
        row: dict[str, Any] = {
            "row_index": int(sources["row_index"][index]),
            "source_id": int(sources["source_id"][index]),
            "parent": int(sources["parent"][index]),
            "n_child": int(sources["n_child"][index]),
            "status": str(sources["status"][index]),
            "measurement_surface": str(sources["measurement_surface"][index]),
            "detect_is_primary": bool(sources["detect_is_primary"][index]),
            "merge_peak_sky": bool(sources["merge_peak_sky"][index]),
            "merge_footprint_sky": bool(sources["merge_footprint_sky"][index]),
            "base_sdss_centroid_flag": bool(sources["base_sdss_centroid_flag"][index]),
            "base_sdss_shape_flag": bool(sources["base_sdss_shape_flag"][index]),
            "base_psf_flux_flag": bool(sources["base_psf_flux_flag"][index]),
            "ra_deg": finite_float(sources["ra_deg"][index]),
            "dec_deg": finite_float(sources["dec_deg"][index]),
            "x_image": finite_float(sources["x_image"][index]),
            "y_image": finite_float(sources["y_image"][index]),
            "centroid_x": finite_float(sources["centroid_x"][index]),
            "centroid_y": finite_float(sources["centroid_y"][index]),
            "official_psf_ab_mag": finite_float(sources["official_psf_ab_mag"][index]),
            "catalog_psf_flux_zp27": finite_float(sources["catalog_psf_flux_zp27"][index]),
            "catalog_psf_flux_njy": finite_float(sources["catalog_psf_flux_njy"][index]),
            "shape_xx": finite_float(sources["shape_xx"][index]),
            "shape_xy": finite_float(sources["shape_xy"][index]),
            "shape_yy": finite_float(sources["shape_yy"][index]),
            "axis_a": finite_float(sources["axis_a"][index]),
            "axis_b": finite_float(sources["axis_b"][index]),
            "theta_deg": finite_float(sources["theta_deg"][index]),
            "initial_determinant_radius": finite_float(sources["initial_determinant_radius"][index]),
            "catalog_KronFlux_radius": finite_float(sources["catalog_kron_radius"][index]),
            "catalog_KronFlux_radius_for_radius": finite_float(sources["catalog_kron_radius_for_radius"][index]),
            "catalog_KronFlux_instFlux": finite_float(sources["catalog_kron_inst_flux"][index]),
            "catalog_KronFlux_instFluxErr": finite_float(sources["catalog_kron_inst_flux_err"][index]),
            "footprint_id": int(sources["footprint_id"][index]),
            "footprint_area": int(sources["footprint_area"][index]),
            "heavy_row": int(sources["heavy_row"][index]),
        }
        row.update({flag_name: bool(sources[flag_name][index]) for flag_name in KRON_FLAGS})
        for key, value in measurement.items():
            if not isinstance(value, np.ndarray):
                continue
            item = value[index]
            if value.dtype.kind == "f":
                row[key] = finite_float(item)
            elif value.dtype.kind == "b":
                row[key] = bool(item)
            elif value.dtype.kind in {"i", "u"}:
                row[key] = int(item)
        row["proxy_nan0_radius_for_radius"] = row.get("heavyfp_nan0_radius_for_radius")
        row["proxy_nan0_sum_i"] = row.get("heavyfp_nan0_sum_i")
        row["proxy_nan0_sum_radius_i"] = row.get("heavyfp_nan0_sum_radius_i")
        row["proxy_nan0_raw_first_moment"] = row.get("heavyfp_nan0_raw_first_moment")
        row["proxy_nan0_candidate_radius"] = row.get("heavyfp_nan0_candidate_radius")
        row["proxy_nan0_determine_radius_returned_radius"] = row.get("heavyfp_nan0_determine_radius_returned_radius")
        row["proxy_nan0_flux_aperture_radius"] = row.get("heavyfp_nan0_flux_aperture_radius")
        row["proxy_nan0_original_determine_radius_returned_radius"] = row.get("heavyfp_nan0_original_determine_radius_returned_radius")
        row["proxy_nan0_original_flux_aperture_radius"] = row.get("heavyfp_nan0_original_flux_aperture_radius")
        row["proxy_nan0_fallback_large_aperture"] = row.get("heavyfp_nan0_fallback_large_aperture")
        row["proxy_nan0_candidate_le_initial_radius"] = row.get("heavyfp_nan0_candidate_le_initial_radius")
        row["proxy_nan0_good"] = row.get("heavyfp_nan0_good")
        row["pixel_value_count"] = row.get("heavy_value_count")
        row["official_minus_proxy_over_proxy"] = ratio_or_none(
            row.get("catalog_KronFlux_radius"),
            row.get("proxy_nan0_determine_radius_returned_radius"),
        )
        rows.append(row)
    return rows


def write_refit_csv(path: Path | str, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def refit_rows_to_table(rows: list[dict[str, Any]]) -> Table:
    return Table(rows=rows) if rows else Table()


def run_refit_from_meas(
    meas_catalog: Path | str,
    reference_image: Path | str,
    *,
    output_csv: Path | str | None = None,
    config: DirectRefitConfig = DirectRefitConfig(),
) -> Table:
    """Run the reusable HeavyFootprint/direct-footprint Kron refit flow.

    The output columns intentionally match ``batch_heavyfp_kron_refit.py``:
    ``proxy_nan0_determine_radius_returned_radius`` is the refit Kron radius,
    while ``proxy_nan0_flux_aperture_radius`` is the Kron flux aperture radius
    after multiplying by ``n_radius_for_flux`` and applying the large-refit
    fallback.
    """

    reference_context = read_reference_image_context(reference_image, allow_missing_ltv=config.allow_missing_ltv)
    heavy_footprints_enabled = not config.disable_heavy_footprints
    with fits.open(Path(meas_catalog), memmap=True, ignore_missing_end=True) as hdul:
        if len(hdul) <= 4:
            raise ValueError(
                "meas catalog must include LSST/HSC footprint archive HDUs "
                "(expected HDU 1 main table, 2 archive index, 3 footprint refs, 4 spans)"
            )
        main = hdul[1].data
        archive = hdul[2].data
        footprint_refs = hdul[3].data
        spans_table = hdul[4].data
        heavy_table = None
        if heavy_footprints_enabled:
            if len(hdul) <= 6:
                if not config.allow_missing_heavy_footprints:
                    raise ValueError("catalog has no HDU 6 HeavyFootprint table")
                heavy_footprints_enabled = False
            else:
                try:
                    heavy_table = hdul[6].data
                except Exception as exc:
                    if not config.allow_missing_heavy_footprints:
                        raise ValueError(f"failed to read HDU 6 HeavyFootprint table: {exc}") from exc
                    heavy_footprints_enabled = False

        selected_rows, _selection_meta = select_refit_rows(main, hdul[1].header, config)
        sources = build_refit_source_arrays(
            main=main,
            header=hdul[1].header,
            rows=selected_rows,
            reference_context=reference_context,
            config=config,
        )
        prepare_heavy_lookups(
            sources=sources,
            archive_index=ArchiveIndex.from_archive(archive),
            footprint_refs=footprint_refs,
            use_heavy_footprints=heavy_footprints_enabled,
        )
        measurement = measure_refit_sources(
            sources=sources,
            spans_table=spans_table,
            heavy_table=heavy_table,
            reference_context=reference_context,
            config=config,
        )

    rows = build_refit_rows(sources, measurement)
    if output_csv is not None:
        write_refit_csv(output_csv, rows)
    return refit_rows_to_table(rows)


def attach_refit_radius_from_table(table: Table, refit_table: Table, config: RefitConfig = RefitConfig()) -> Table:
    """Attach refit radius columns from an in-memory refit table."""

    if len(refit_table) == 0:
        out = table.copy(copy_data=True)
        out[config.output_column] = np.full(len(out), np.nan, dtype=np.float32)
        out[f"{config.output_column}_matched"] = np.zeros(len(out), dtype=bool)
        return out
    if "source_id" not in refit_table.colnames:
        raise KeyError("refit_table must contain source_id")
    if config.radius_column not in refit_table.colnames:
        raise KeyError(f"refit_table missing radius column {config.radius_column!r}")

    has_good = config.good_column in refit_table.colnames
    refit_ids = np.asarray(refit_table["source_id"], dtype=np.int64)
    refit_radius = np.asarray(refit_table[config.radius_column], dtype=np.float64)
    if has_good:
        good = np.asarray(refit_table[config.good_column], dtype=bool)
    else:
        good = np.ones(len(refit_table), dtype=bool)
    valid = good & np.isfinite(refit_radius) & (refit_radius > 0.0)
    radius_by_id = {int(sid): float(radius) for sid, radius in zip(refit_ids[valid], refit_radius[valid])}

    aperture_pixels_by_id: dict[int, float] = {}
    footprint_area_by_id: dict[int, float] = {}
    optional_by_column: dict[str, dict[int, float]] = {}
    if "aperture_pixel_count" in refit_table.colnames:
        values = np.asarray(refit_table["aperture_pixel_count"], dtype=np.float64)
        aperture_pixels_by_id = {int(sid): float(value) for sid, value in zip(refit_ids, values) if np.isfinite(value)}
    if "footprint_area" in refit_table.colnames:
        values = np.asarray(refit_table["footprint_area"], dtype=np.float64)
        footprint_area_by_id = {int(sid): float(value) for sid, value in zip(refit_ids, values) if np.isfinite(value)}
    for src_col, dst_col in (
        ("x_image", "pu_refit_x_image"),
        ("y_image", "pu_refit_y_image"),
        ("axis_a", "pu_refit_axis_a"),
        ("axis_b", "pu_refit_axis_b"),
        ("theta_deg", "pu_refit_theta_deg"),
        ("initial_determinant_radius", "pu_refit_initial_determinant_radius"),
    ):
        if src_col not in refit_table.colnames:
            continue
        values = np.asarray(refit_table[src_col], dtype=np.float64)
        optional_by_column[dst_col] = {int(sid): float(value) for sid, value in zip(refit_ids, values) if np.isfinite(value)}

    out = table.copy(copy_data=True)
    ids = source_ids(out)
    radii = np.full(len(out), np.nan, dtype=np.float32)
    matched = np.zeros(len(out), dtype=bool)
    aperture_pixels = np.full(len(out), np.nan, dtype=np.float32)
    footprint_area = np.full(len(out), np.nan, dtype=np.float32)
    for idx, sid in enumerate(ids):
        value = radius_by_id.get(int(sid))
        if value is None:
            continue
        radii[idx] = float(value)
        matched[idx] = True
        if int(sid) in aperture_pixels_by_id:
            aperture_pixels[idx] = aperture_pixels_by_id[int(sid)]
        if int(sid) in footprint_area_by_id:
            footprint_area[idx] = footprint_area_by_id[int(sid)]
    out[config.output_column] = radii
    out[f"{config.output_column}_matched"] = matched
    if aperture_pixels_by_id:
        out["pu_refit_aperture_pixel_count"] = aperture_pixels
    if footprint_area_by_id:
        out["pu_refit_footprint_area"] = footprint_area
    for dst_col, by_id in optional_by_column.items():
        values = np.full(len(out), np.nan, dtype=np.float32)
        for idx, sid in enumerate(ids):
            value = by_id.get(int(sid))
            if value is not None:
                values[idx] = float(value)
        out[dst_col] = values
    return out


def attach_refit_radius(table: Table, refit_csv: Path | str | None, config: RefitConfig = RefitConfig()) -> Table:
    """Attach a refit Kron aperture radius matched by source id.

    This is copied from the legacy PU filter flow, with the names kept stable so
    old diagnostics can be compared against v3 outputs.
    """

    if refit_csv is None:
        return table.copy(copy_data=True)
    path = Path(refit_csv)
    if not path.exists():
        raise FileNotFoundError(f"kron refit CSV not found: {path}")

    refit_by_id: dict[int, float] = {}
    aperture_pixels_by_id: dict[int, float] = {}
    footprint_area_by_id: dict[int, float] = {}
    optional_by_column: dict[str, dict[int, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"kron refit CSV has no header: {path}")
        if "source_id" not in reader.fieldnames:
            raise KeyError(f"kron refit CSV missing source_id: {path}")
        if config.radius_column not in reader.fieldnames:
            out = table.copy(copy_data=True)
            out[config.output_column] = np.full(len(out), np.nan, dtype=np.float32)
            out[f"{config.output_column}_matched"] = np.zeros(len(out), dtype=bool)
            out[f"{config.output_column}_missing_column"] = np.asarray([config.radius_column] * len(out), dtype=str)
            return out

        has_good = config.good_column in reader.fieldnames
        for row in reader:
            if has_good and not _as_bool(row.get(config.good_column, False)):
                continue
            try:
                sid = int(row["source_id"])
                radius = float(row[config.radius_column])
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(radius) and radius > 0.0):
                continue
            refit_by_id[sid] = radius
            for column, bucket in (
                ("aperture_pixel_count", aperture_pixels_by_id),
                ("footprint_area", footprint_area_by_id),
            ):
                if column not in reader.fieldnames:
                    continue
                try:
                    value = float(row[column])
                except (TypeError, ValueError):
                    continue
                    if np.isfinite(value):
                        bucket[sid] = value
            for src_col, dst_col in (
                ("x_image", "pu_refit_x_image"),
                ("y_image", "pu_refit_y_image"),
                ("axis_a", "pu_refit_axis_a"),
                ("axis_b", "pu_refit_axis_b"),
                ("theta_deg", "pu_refit_theta_deg"),
                ("initial_determinant_radius", "pu_refit_initial_determinant_radius"),
            ):
                if src_col not in reader.fieldnames:
                    continue
                try:
                    value = float(row[src_col])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    optional_by_column.setdefault(dst_col, {})[sid] = value

    out = table.copy(copy_data=True)
    ids = source_ids(out)
    radii = np.full(len(out), np.nan, dtype=np.float32)
    matched = np.zeros(len(out), dtype=bool)
    aperture_pixels = np.full(len(out), np.nan, dtype=np.float32)
    footprint_area = np.full(len(out), np.nan, dtype=np.float32)
    for idx, sid in enumerate(ids):
        value = refit_by_id.get(int(sid))
        if value is None:
            continue
        radii[idx] = float(value)
        matched[idx] = True
        if int(sid) in aperture_pixels_by_id:
            aperture_pixels[idx] = float(aperture_pixels_by_id[int(sid)])
        if int(sid) in footprint_area_by_id:
            footprint_area[idx] = float(footprint_area_by_id[int(sid)])

    out[config.output_column] = radii
    out[f"{config.output_column}_matched"] = matched
    if aperture_pixels_by_id:
        out["pu_refit_aperture_pixel_count"] = aperture_pixels
    if footprint_area_by_id:
        out["pu_refit_footprint_area"] = footprint_area
    for dst_col, by_id in optional_by_column.items():
        values = np.full(len(out), np.nan, dtype=np.float32)
        for idx, sid in enumerate(ids):
            value = by_id.get(int(sid))
            if value is not None:
                values[idx] = float(value)
        out[dst_col] = values
    return out


def sdss_ellipse(table: Table) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx = first_finite_column(table, ("base_SdssShape_xx", "ext_shapeHSM_HsmSourceMoments_xx"))
    yy = first_finite_column(table, ("base_SdssShape_yy", "ext_shapeHSM_HsmSourceMoments_yy"))
    xy = first_finite_column(table, ("base_SdssShape_xy", "ext_shapeHSM_HsmSourceMoments_xy"))
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


def compute_kron_ellipse(table: Table, config: RefitConfig = RefitConfig()) -> EllipseGeometry:
    """Compute refit Kron aperture ellipses using SDSS shape orientation."""

    x, y = source_xy(table)
    sdss_major, sdss_minor, theta = sdss_ellipse(table)
    if "pu_refit_x_image" in table.colnames and "pu_refit_y_image" in table.colnames:
        refit_x = np.asarray(table["pu_refit_x_image"], dtype=np.float64)
        refit_y = np.asarray(table["pu_refit_y_image"], dtype=np.float64)
        take = np.isfinite(refit_x) & np.isfinite(refit_y)
        x = x.copy()
        y = y.copy()
        x[take] = refit_x[take]
        y[take] = refit_y[take]
    kron_columns = (config.output_column,) if config.require_match else (
        config.output_column,
        "ext_photometryKron_KronFlux_radius",
        "ext_photometryKron_KronFlux_radius_for_radius",
    )
    kron = first_finite_column(table, kron_columns)
    if config.use_smaller_official and "ext_photometryKron_KronFlux_radius" in table.colnames:
        official = np.asarray(table["ext_photometryKron_KronFlux_radius"], dtype=np.float64)
        official_aperture = float(config.official_aperture_scale) * official
        use_official = np.isfinite(official_aperture) & (official_aperture > 0.0) & np.isfinite(kron) & (official_aperture < kron)
        kron = kron.copy()
        kron[use_official] = official_aperture[use_official]

    refit_matched = (
        np.asarray(table[f"{config.output_column}_matched"], dtype=bool)
        if f"{config.output_column}_matched" in table.colnames
        else np.zeros(len(table), dtype=bool)
    )
    determinant_radius = np.sqrt(np.maximum(sdss_major * sdss_minor, 0.0))
    valid = (
        np.isfinite(kron)
        & (kron > 0.0)
        & np.isfinite(sdss_major)
        & np.isfinite(sdss_minor)
        & np.isfinite(determinant_radius)
        & (determinant_radius > 0.0)
        & np.isfinite(theta)
        & np.isfinite(x)
        & np.isfinite(y)
    )
    if config.require_match:
        valid &= refit_matched

    major = np.full(len(table), np.nan, dtype=np.float64)
    minor = np.full(len(table), np.nan, dtype=np.float64)
    angle = np.full(len(table), np.nan, dtype=np.float64)
    if all(
        name in table.colnames
        for name in (
            "pu_refit_axis_a",
            "pu_refit_axis_b",
            "pu_refit_theta_deg",
            "pu_refit_initial_determinant_radius",
        )
    ):
        axis_a = np.asarray(table["pu_refit_axis_a"], dtype=np.float64)
        axis_b = np.asarray(table["pu_refit_axis_b"], dtype=np.float64)
        theta_deg = np.asarray(table["pu_refit_theta_deg"], dtype=np.float64)
        initial_radius = np.asarray(table["pu_refit_initial_determinant_radius"], dtype=np.float64)
        refit_valid = valid & np.isfinite(axis_a) & np.isfinite(axis_b) & np.isfinite(theta_deg) & np.isfinite(initial_radius) & (axis_a > 0.0) & (axis_b > 0.0) & (initial_radius > 0.0)
        scale = np.zeros(len(table), dtype=np.float64)
        scale[refit_valid] = kron[refit_valid] / initial_radius[refit_valid]
        major[refit_valid] = np.maximum(axis_a[refit_valid] * scale[refit_valid] * float(config.ellipse_sigma), float(config.min_axis))
        minor[refit_valid] = np.maximum(axis_b[refit_valid] * scale[refit_valid] * float(config.ellipse_sigma), float(config.min_axis))
        angle[refit_valid] = np.radians(theta_deg[refit_valid])
        fallback_valid = valid & ~refit_valid
    else:
        fallback_valid = valid
    scale = np.zeros(len(table), dtype=np.float64)
    scale[fallback_valid] = kron[fallback_valid] / determinant_radius[fallback_valid]
    major[fallback_valid] = np.maximum(sdss_major[fallback_valid] * scale[fallback_valid] * float(config.ellipse_sigma), float(config.min_axis))
    minor[fallback_valid] = np.maximum(sdss_minor[fallback_valid] * scale[fallback_valid] * float(config.ellipse_sigma), float(config.min_axis))
    angle[fallback_valid] = theta[fallback_valid]
    area = math.pi * major * minor
    return EllipseGeometry(x=x, y=y, major=major, minor=minor, theta=angle, area=area)


def attach_refit_geometry(table: Table, refit_csv: Path | str | None, config: RefitConfig = RefitConfig()) -> Table:
    out = attach_refit_radius(table, refit_csv, config)
    geom = compute_kron_ellipse(out, config)
    out["v3_x"] = geom.x.astype(np.float64)
    out["v3_y"] = geom.y.astype(np.float64)
    out["v3_major"] = geom.major.astype(np.float32)
    out["v3_minor"] = geom.minor.astype(np.float32)
    out["v3_theta"] = geom.theta.astype(np.float32)
    out["v3_area"] = geom.area.astype(np.float32)
    return out
