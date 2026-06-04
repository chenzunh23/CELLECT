from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

CENTRAL_PIXEL_ER = 0.38259771140356325


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
            names=np.asarray(
                [decode_archive_string(value) for value in archive["name"]]
            ),
            row0=np.asarray(archive["row0"], dtype=np.int64),
            nrows=np.asarray(archive["nrows"], dtype=np.int64),
        )

    def lookup(
        self, target_ids: np.ndarray, *, archive_number: int, name: str | None
    ) -> ArchiveLookup:
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
    x_ltv: int
    y_ltv: int


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.artifact_name:
        output_dir = output_dir / str(args.artifact_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_batch(args=args, output_dir=output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary_json": result["outputs"]["summary_json"],
            }
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone batch HeavyFootprint Kron proxy for HSC/LSST meas sources, "
            "using sparse FITS archive pixels directly. This is not an exact "
            "production KronFlux replay."
        )
    )
    parser.add_argument("--meas-catalog", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows-file", type=Path)
    parser.add_argument("--mag-min", type=float)
    parser.add_argument("--mag-max", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--artifact-name",
        default="batch_heavyfp_kron_refit",
        help="Optional subdirectory name under --output-dir. Use '' to write directly there.",
    )
    parser.add_argument("--input-zeropoint", type=float, default=27.0)
    parser.add_argument("--output-zeropoint", type=float, default=31.4)
    parser.add_argument("--n-sigma-for-radius", type=float, default=6.0)
    parser.add_argument("--n-radius-for-flux", type=float, default=2.5)
    parser.add_argument("--chunk-pixel-limit", type=int, default=10_000_000)
    parser.add_argument("--include-sky", action="store_true")
    parser.add_argument("--include-non-primary", action="store_true")
    parser.add_argument(
        "--leaf-only",
        action="store_true",
        help="Select only leaf deblend sources with deblend_nChild == 0.",
    )
    parser.add_argument("--include-shape-flagged", action="store_true")
    parser.add_argument("--include-centroid-flagged", action="store_true")
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help=(
            "Write only batch_heavyfp_kron_refit.csv and summary.json. "
            "This skips the row JSON dump, DS9 regions, and matplotlib histogram used for diagnostics."
        ),
    )
    parser.add_argument(
        "--disable-heavy-footprints",
        action="store_true",
        help=(
            "Do not read HDU 6 HeavyFootprint payloads. Measure all processable "
            "sources by sampling the reference image inside footprint spans."
        ),
    )
    parser.add_argument(
        "--allow-missing-heavy-footprints",
        action="store_true",
        help=(
            "If HDU 6 is missing or unreadable, continue in direct-reference-image "
            "fallback mode instead of failing."
        ),
    )
    return parser


def run_batch(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    flux_scale_to_njy = float(
        10.0 ** ((float(args.output_zeropoint) - float(args.input_zeropoint)) / 2.5)
    )
    reference_context = read_reference_image_context(args.reference_image)
    heavy_footprint_error = None
    heavy_footprints_enabled = not bool(args.disable_heavy_footprints)
    with fits.open(args.meas_catalog, memmap=True) as hdul:
        if len(hdul) <= 4:
            raise ValueError(
                "meas catalog must include LSST/HSC footprint archive HDUs "
                "(expected at least HDU 1 main table, 2 archive index, "
                "3 footprint refs, 4 spans). Got {} HDUs: {}".format(
                    len(hdul),
                    [
                        "{}:{}".format(index, hdu.name or "")
                        for index, hdu in enumerate(hdul)
                    ],
                )
            )
        main = hdul[1].data
        archive = hdul[2].data
        footprint_refs = hdul[3].data
        spans_table = hdul[4].data
        heavy_table = None
        if heavy_footprints_enabled:
            if len(hdul) <= 6:
                heavy_footprint_error = (
                    "catalog has no HDU 6 HeavyFootprint table; using direct "
                    "reference-image fallback"
                )
                if not args.allow_missing_heavy_footprints:
                    raise ValueError(
                        heavy_footprint_error
                        + " (pass --allow-missing-heavy-footprints or "
                        "--disable-heavy-footprints to continue)"
                    )
                heavy_footprints_enabled = False
            else:
                try:
                    heavy_table = hdul[6].data
                except Exception as exc:
                    heavy_footprint_error = "{}: {}".format(
                        type(exc).__name__, str(exc)
                    )
                    if not args.allow_missing_heavy_footprints:
                        raise ValueError(
                            "failed to read HDU 6 HeavyFootprint table: {} "
                            "(pass --allow-missing-heavy-footprints or "
                            "--disable-heavy-footprints to continue with direct "
                            "reference-image fallback)".format(heavy_footprint_error)
                        ) from exc
                    heavy_footprints_enabled = False
        selected_rows, selection_meta = select_rows(main, hdul[1].header, args)
        sources = build_source_arrays(
            main=main,
            header=hdul[1].header,
            rows=selected_rows,
            reference_image=args.reference_image,
            input_zeropoint=float(args.input_zeropoint),
            output_zeropoint=float(args.output_zeropoint),
            flux_scale_to_njy=flux_scale_to_njy,
        )
        prepare_heavy_lookups(
            sources=sources,
            archive_index=ArchiveIndex.from_archive(archive),
            footprint_refs=footprint_refs,
            use_heavy_footprints=heavy_footprints_enabled,
        )
        measurement = measure_sources(
            sources=sources,
            spans_table=spans_table,
            heavy_table=heavy_table,
            reference_context=reference_context,
            flux_scale_to_njy=flux_scale_to_njy,
            n_sigma_for_radius=float(args.n_sigma_for_radius),
            n_radius_for_flux=float(args.n_radius_for_flux),
            chunk_pixel_limit=int(args.chunk_pixel_limit),
        )

    rows = build_output_rows(sources, measurement)
    csv_path = output_dir / "batch_heavyfp_kron_refit.csv"
    json_path = output_dir / "batch_heavyfp_kron_refit.json"
    regions_dir = output_dir / "regions"
    regions_manifest_path = output_dir / "regions_manifest.json"
    histogram_png_path = output_dir / "official_minus_proxy_over_proxy_histogram.png"
    histogram_csv_path = output_dir / "official_minus_proxy_over_proxy_histogram.csv"
    ratio_stats_path = output_dir / "official_minus_proxy_over_proxy_stats.json"
    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"

    write_csv(csv_path, rows)
    if args.csv_only:
        region_outputs = {}
        ratio_summary = {"all": ratio_stats(ratio_values_for_rows(rows))}
    else:
        json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        region_outputs = write_split_ds9_regions(regions_dir, rows)
        regions_manifest_path.write_text(
            json.dumps(region_outputs, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        ratio_summary = write_ratio_histogram(
            rows=rows,
            png_path=histogram_png_path,
            csv_path=histogram_csv_path,
        )
        ratio_stats_path.write_text(
            json.dumps(ratio_summary, indent=2, sort_keys=True), encoding="utf-8"
        )

    summary = {
        "run_context": {
            "mode": "standalone",
            "cwd": str(Path.cwd()),
            "output_dir": str(output_dir),
        },
        "inputs": {
            "meas_catalog": str(args.meas_catalog),
            "reference_image": str(args.reference_image),
            "rows_file": str(args.rows_file) if args.rows_file else None,
            "mag_min": args.mag_min,
            "mag_max": args.mag_max,
            "input_zeropoint": float(args.input_zeropoint),
            "output_zeropoint": float(args.output_zeropoint),
            "flux_scale_to_njy": flux_scale_to_njy,
            "n_sigma_for_radius": float(args.n_sigma_for_radius),
            "n_radius_for_flux": float(args.n_radius_for_flux),
            "chunk_pixel_limit": int(args.chunk_pixel_limit),
            "heavy_footprints_enabled": bool(heavy_footprints_enabled),
            "heavy_footprint_error": heavy_footprint_error,
        },
        "selection": selection_meta,
        "processing": {
            "n_sources_selected": int(len(rows)),
            "n_sources_processed": int(np.count_nonzero(sources["processable"])),
            "n_sources_skipped": int(np.count_nonzero(~sources["processable"])),
            "status_counts": count_values([str(row["status"]) for row in rows]),
            "measurement_surface_counts": count_values(
                [
                    str(row["measurement_surface"])
                    for row in rows
                    if row["status"] == "ok"
                ]
            ),
            "total_expanded_pixels": int(measurement["total_expanded_pixels"]),
            "max_pixels_per_source": int(measurement["max_pixels_per_source"]),
            "n_chunks": int(measurement["n_chunks"]),
            "region_counts": {
                name: int(info["count"]) for name, info in region_outputs.items()
            },
            "official_minus_proxy_over_proxy": ratio_summary["all"],
        },
        "outputs": {
            "output_dir": str(output_dir),
            "csv": str(csv_path),
            "json": None if args.csv_only else str(json_path),
            "official_minus_proxy_over_proxy_histogram_png": None if args.csv_only else str(histogram_png_path),
            "official_minus_proxy_over_proxy_histogram_csv": None if args.csv_only else str(histogram_csv_path),
            "official_minus_proxy_over_proxy_stats_json": None if args.csv_only else str(ratio_stats_path),
            "ds9_regions_dir": None if args.csv_only else str(regions_dir),
            "ds9_regions": {
                name: str(info["path"]) for name, info in region_outputs.items()
            },
            "regions_manifest": None if args.csv_only else str(regions_manifest_path),
            "summary_json": str(summary_json_path),
            "summary_md": None if args.csv_only else str(summary_md_path),
        },
    }
    summary_json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not args.csv_only:
        write_summary_md(summary_md_path, summary, rows)
    return summary


def select_rows(
    main: fits.FITS_rec,
    header: fits.Header,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    if args.rows_file is not None:
        rows = read_rows_file(args.rows_file)
        if args.limit is not None:
            rows = rows[: int(args.limit)]
        return rows, {
            "mode": "rows_file",
            "rows_file": str(args.rows_file),
            "selected_rows": int(rows.size),
        }

    if args.mag_min is None or args.mag_max is None:
        raise ValueError("provide either --rows-file or both --mag-min/--mag-max")

    flags = flags_array(main)
    flag_names = flag_name_map(header)

    def flag(name: str) -> np.ndarray:
        index = flag_names.get(name)
        if index is None:
            return np.zeros(len(main), dtype=bool)
        return flags[:, index]

    psf_flux_zp27 = np.asarray(main["base_PsfFlux_instFlux"], dtype=np.float64)
    psf_mag = magnitude_from_zp27_flux(
        psf_flux_zp27, input_zeropoint=float(args.input_zeropoint)
    )
    xx = np.asarray(main["base_SdssShape_xx"], dtype=np.float64)
    yy = np.asarray(main["base_SdssShape_yy"], dtype=np.float64)
    xy = np.asarray(main["base_SdssShape_xy"], dtype=np.float64)
    det = xx * yy - xy * xy
    shape_ok = (
        np.isfinite(xx)
        & np.isfinite(yy)
        & np.isfinite(xy)
        & (xx > 0.0)
        & (yy > 0.0)
        & (det > 0.0)
    )
    mask = (
        np.isfinite(psf_mag)
        & (psf_mag >= float(args.mag_min))
        & (psf_mag < float(args.mag_max))
    )
    if not args.include_non_primary:
        mask &= flag("detect_isPrimary")
    if not args.include_shape_flagged:
        mask &= ~flag("base_SdssShape_flag") & shape_ok
    if not args.include_centroid_flagged:
        mask &= ~flag("base_SdssCentroid_flag")
    if not args.include_sky:
        mask &= ~(flag("merge_peak_sky") | flag("merge_footprint_sky"))
    if args.leaf_only:
        if "deblend_nChild" not in main.columns.names:
            raise KeyError("--leaf-only requires deblend_nChild column")
        mask &= np.asarray(main["deblend_nChild"], dtype=np.int64) == 0

    rows = np.flatnonzero(mask).astype(np.int64)
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    return rows, {
        "mode": "official_psf_mag",
        "mag_min": float(args.mag_min),
        "mag_max": float(args.mag_max),
        "mag_column": "base_PsfFlux_instFlux",
        "mag_formula": "input_zeropoint - 2.5*log10(base_PsfFlux_instFlux)",
        "input_zeropoint": float(args.input_zeropoint),
        "include_non_primary": bool(args.include_non_primary),
        "leaf_only": bool(args.leaf_only),
        "include_shape_flagged": bool(args.include_shape_flagged),
        "include_centroid_flagged": bool(args.include_centroid_flagged),
        "include_sky": bool(args.include_sky),
        "selected_rows": int(rows.size),
        "limit": int(args.limit) if args.limit is not None else None,
    }


def read_rows_file(path: Path) -> np.ndarray:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        return np.asarray([], dtype=np.int64)

    header = next(csv.reader([lines[0]]))
    header_names = {name.strip().lower(): name for name in header}
    row_column = header_names.get("row_index") or header_names.get("row")
    if row_column is not None:
        rows = []
        for record in csv.DictReader(lines):
            value = record.get(row_column)
            if value is None or not value.strip():
                continue
            rows.append(int(value))
        return np.asarray(rows, dtype=np.int64)

    rows: list[int] = []
    for line in lines:
        stripped = line.strip()
        for token in stripped.replace(",", " ").split():
            rows.append(int(token))
            break
    return np.asarray(rows, dtype=np.int64)


def build_source_arrays(
    *,
    main: fits.FITS_rec,
    header: fits.Header,
    rows: np.ndarray,
    reference_image: Path,
    input_zeropoint: float,
    output_zeropoint: float,
    flux_scale_to_njy: float,
) -> dict[str, np.ndarray]:
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
    axes = ellipse_axes_from_moments(xx=xx, yy=yy, xy=xy)
    psf_flux_zp27 = np.asarray(main["base_PsfFlux_instFlux"][rows], dtype=np.float64)
    psf_flux_njy = psf_flux_zp27 * float(flux_scale_to_njy)
    psf_mag = magnitude_from_njy_flux(psf_flux_njy, output_zeropoint=output_zeropoint)
    x_image, y_image = image_positions_from_reference(
        reference_image=reference_image,
        ra_rad=np.asarray(main["coord_ra"][rows], dtype=np.float64),
        dec_rad=np.asarray(main["coord_dec"][rows], dtype=np.float64),
    )
    sources: dict[str, np.ndarray] = {
        "row_index": np.asarray(rows, dtype=np.int64),
        "source_id": np.asarray(main["id"][rows], dtype=np.int64),
        "parent": np.asarray(main["parent"][rows], dtype=np.int64),
        "n_child": np.asarray(main["deblend_nChild"][rows], dtype=np.int64),
        "footprint_id": np.asarray(main["footprint"][rows], dtype=np.int64),
        "footprint_area": np.asarray(
            main["base_FootprintArea_value"][rows], dtype=np.int64
        ),
        "ra_deg": np.degrees(np.asarray(main["coord_ra"][rows], dtype=np.float64)),
        "dec_deg": np.degrees(np.asarray(main["coord_dec"][rows], dtype=np.float64)),
        "x_image": x_image,
        "y_image": y_image,
        "centroid_x": np.asarray(main["base_SdssCentroid_x"][rows], dtype=np.float64),
        "centroid_y": np.asarray(main["base_SdssCentroid_y"][rows], dtype=np.float64),
        "shape_xx": xx,
        "shape_yy": yy,
        "shape_xy": xy,
        "axis_a": axes["a"],
        "axis_b": axes["b"],
        "theta_rad": axes["theta_rad"],
        "theta_deg": axes["theta_deg"],
        "initial_determinant_radius": axes["determinant_radius"],
        "shape_valid": axes["valid"],
        "catalog_kron_radius": np.asarray(
            main["ext_photometryKron_KronFlux_radius"][rows], dtype=np.float64
        ),
        "catalog_kron_radius_for_radius": np.asarray(
            main["ext_photometryKron_KronFlux_radius_for_radius"][rows],
            dtype=np.float64,
        ),
        "catalog_kron_inst_flux": np.asarray(
            main["ext_photometryKron_KronFlux_instFlux"][rows], dtype=np.float64
        ),
        "catalog_kron_inst_flux_err": np.asarray(
            main["ext_photometryKron_KronFlux_instFluxErr"][rows], dtype=np.float64
        ),
        "catalog_psf_flux_zp27": psf_flux_zp27,
        "catalog_psf_flux_njy": psf_flux_njy,
        "official_psf_ab_mag": psf_mag,
        "detect_is_primary": flag("detect_isPrimary")[rows],
        "merge_peak_sky": flag("merge_peak_sky")[rows],
        "merge_footprint_sky": flag("merge_footprint_sky")[rows],
        "base_sdss_centroid_flag": flag("base_SdssCentroid_flag")[rows],
        "base_sdss_shape_flag": flag("base_SdssShape_flag")[rows],
        "base_psf_flux_flag": flag("base_PsfFlux_flag")[rows],
    }
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

    valid_ref = (
        footprint_lookup.found
        & (footprint_lookup.row0 >= 0)
        & (footprint_lookup.row0 < len(footprint_refs))
    )
    spanset_ids = np.full(len(footprint_ids), -1, dtype=np.int64)
    spanset_ids[valid_ref] = np.asarray(footprint_refs["id"], dtype=np.int64)[
        footprint_lookup.row0[valid_ref]
    ]
    sources["spanset_id"] = spanset_ids

    spanset_lookup = archive_index.lookup(spanset_ids, archive_number=2, name="SpanSet")
    sources["spanset_archive_found"] = spanset_lookup.found & valid_ref
    sources["span0"] = spanset_lookup.row0
    sources["nspan"] = spanset_lookup.nrows

    if use_heavy_footprints:
        heavy_lookup = archive_index.lookup(
            footprint_ids, archive_number=4, name="HeavyFootprintF"
        )
        sources["heavy_archive_found"] = heavy_lookup.found
        sources["heavy_row"] = heavy_lookup.row0
    else:
        sources["heavy_archive_found"] = np.zeros(len(footprint_ids), dtype=bool)
        sources["heavy_row"] = np.full(len(footprint_ids), -1, dtype=np.int64)

    finite_centroid = np.isfinite(sources["centroid_x"]) & np.isfinite(
        sources["centroid_y"]
    )
    base_processable = (
        sources["shape_valid"]
        & finite_centroid
        & sources["footprint_archive_found"]
        & sources["spanset_archive_found"]
        & (sources["nspan"] > 0)
    )
    has_heavy = sources["heavy_archive_found"] & (sources["heavy_row"] >= 0)
    processable = base_processable
    sources["processable"] = processable
    surface = sources["measurement_surface"]
    surface[:] = "none"
    surface[processable & has_heavy] = "heavyfootprintf"
    surface[processable & ~has_heavy] = "direct_footprint_reference_image"

    status = sources["status"]
    status[:] = "ok"
    status[~sources["shape_valid"]] = "skip_invalid_shape"
    status[~finite_centroid] = "skip_invalid_centroid"
    status[~sources["footprint_archive_found"]] = "skip_no_footprint_archive"
    status[~sources["spanset_archive_found"]] = "skip_no_spanset_archive"
    status[~processable & (sources["nspan"] <= 0)] = "skip_empty_spanset"
    status[processable] = "ok"


def measure_sources(
    *,
    sources: dict[str, np.ndarray],
    spans_table: fits.FITS_rec,
    heavy_table: fits.FITS_rec | None,
    reference_context: ReferenceImageContext,
    flux_scale_to_njy: float,
    n_sigma_for_radius: float,
    n_radius_for_flux: float,
    chunk_pixel_limit: int,
) -> dict[str, np.ndarray | int]:
    n = len(sources["row_index"])
    out = init_measurement_arrays(n)
    process_positions = np.flatnonzero(sources["processable"])
    if process_positions.size == 0:
        return {
            **out,
            "total_expanded_pixels": 0,
            "max_pixels_per_source": 0,
            "n_chunks": 0,
        }

    blocks: list[tuple[str, np.ndarray]] = []
    for surface in ("heavyfootprintf", "direct_footprint_reference_image"):
        surface_positions = process_positions[
            np.asarray(sources["measurement_surface"][process_positions], dtype=object)
            == surface
        ]
        if surface_positions.size == 0:
            continue
        pixel_counts = np.asarray(
            sources["footprint_area"][surface_positions], dtype=np.int64
        )
        blocks.extend(
            (surface, block)
            for block in source_blocks(
                surface_positions, pixel_counts, int(chunk_pixel_limit)
            )
        )
    total_pixels = 0
    max_pixels = 0

    for surface, block in blocks:
        block_result = measure_block(
            positions=block,
            sources=sources,
            spans_table=spans_table,
            heavy_table=heavy_table,
            reference_context=reference_context,
            value_source=surface,
            flux_scale_to_njy=flux_scale_to_njy,
            n_sigma_for_radius=n_sigma_for_radius,
        )
        total_pixels += int(block_result["total_expanded_pixels"])
        if int(block_result["max_pixels_per_source"]) > max_pixels:
            max_pixels = int(block_result["max_pixels_per_source"])
        for key in out:
            out[key][block] = block_result[key]

    out["heavyfp_nan0_flux_aperture_radius"] = out[
        "heavyfp_nan0_determine_radius_returned_radius"
    ] * float(n_radius_for_flux)
    return {
        **out,
        "total_expanded_pixels": int(total_pixels),
        "max_pixels_per_source": int(max_pixels),
        "n_chunks": int(len(blocks)),
    }


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
        "heavyfp_nan0_computed_span_area",
    ]
    int_keys = [
        "heavy_value_count",
        "finite_value_count",
        "nonfinite_value_count",
        "aperture_pixel_count",
        "central_pixel_count",
    ]
    out: dict[str, np.ndarray] = {
        key: np.full(n, np.nan, dtype=np.float64) for key in float_keys
    }
    out.update({key: np.zeros(n, dtype=np.int64) for key in int_keys})
    out["heavyfp_nan0_candidate_le_initial_radius"] = np.zeros(n, dtype=bool)
    out["heavyfp_nan0_good"] = np.zeros(n, dtype=bool)
    out["span_area_matches_catalog"] = np.zeros(n, dtype=bool)
    return out


def source_blocks(
    positions: np.ndarray,
    pixel_counts: np.ndarray,
    chunk_pixel_limit: int,
) -> list[np.ndarray]:
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


def measure_block(
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
        values_zp27 = np.concatenate(heavy_table["image"][heavy_rows]).astype(
            np.float64, copy=False
        )
        if values_zp27.size != total_pixels:
            raise ValueError(
                "HeavyFootprint value count mismatch: values={} pixels={}".format(
                    values_zp27.size, total_pixels
                )
            )
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
            sampled[in_reference] = np.asarray(
                reference_context.image[local_y[in_reference], local_x[in_reference]],
                dtype=np.float64,
            )
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
    sum_r_i = np.bincount(
        included_src,
        weights=included_radius * included_values,
        minlength=n_block,
    )
    positive = np.where(included_values > 0.0, included_values, 0.0)
    negative = np.where(included_values < 0.0, included_values, 0.0)
    positive_sum = np.bincount(included_src, weights=positive, minlength=n_block)
    negative_sum = np.bincount(included_src, weights=negative, minlength=n_block)

    span_area = np.bincount(span_src, weights=span_width, minlength=n_block)
    finite_count = np.bincount(
        all_src, weights=finite.astype(np.int64), minlength=n_block
    )
    nonfinite_count = np.bincount(
        all_src, weights=(~finite).astype(np.int64), minlength=n_block
    )
    aperture_pixel_count = np.bincount(included_src, minlength=n_block)
    central_pixel_count = np.bincount(all_src[central], minlength=n_block)

    raw_first_moment = np.full(n_block, np.nan, dtype=np.float64)
    candidate_radius = np.full(n_block, np.nan, dtype=np.float64)
    good = (sum_i > 0.0) & (sum_r_i > 0.0)
    raw_first_moment[good] = sum_r_i[good] / sum_i[good]
    candidate_radius[good] = raw_first_moment[good] * np.sqrt(
        axis_b[good] / axis_a[good]
    )

    radius_for_radius = float(n_sigma_for_radius) * r_det
    candidate_le_initial = good & (candidate_radius <= r_det)
    returned_radius = np.full(n_block, np.nan, dtype=np.float64)
    returned_radius[good] = np.where(
        candidate_le_initial[good],
        radius_for_radius[good],
        candidate_radius[good],
    )

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
        "span_area_matches_catalog": span_area.astype(np.int64)
        == sources["footprint_area"][positions],
        "total_expanded_pixels": int(total_pixels),
        "max_pixels_per_source": int(np.max(span_area)) if span_area.size else 0,
    }


def ellipse_axes_from_moments(
    *,
    xx: np.ndarray,
    yy: np.ndarray,
    xy: np.ndarray,
) -> dict[str, np.ndarray]:
    trace = xx + yy
    delta = np.sqrt((xx - yy) ** 2 + 4.0 * xy * xy)
    lambda_major = 0.5 * (trace + delta)
    lambda_minor = 0.5 * (trace - delta)
    valid = (
        np.isfinite(lambda_major)
        & np.isfinite(lambda_minor)
        & (lambda_major > 0.0)
        & (lambda_minor > 0.0)
    )
    a = np.full(xx.shape, np.nan, dtype=np.float64)
    b = np.full(xx.shape, np.nan, dtype=np.float64)
    theta_rad = np.full(xx.shape, np.nan, dtype=np.float64)
    determinant_radius = np.full(xx.shape, np.nan, dtype=np.float64)
    a[valid] = np.sqrt(lambda_major[valid])
    b[valid] = np.sqrt(lambda_minor[valid])
    theta_rad[valid] = 0.5 * np.arctan2(2.0 * xy[valid], xx[valid] - yy[valid])
    determinant_radius[valid] = np.sqrt(a[valid] * b[valid])
    return {
        "a": a,
        "b": b,
        "theta_rad": theta_rad,
        "theta_deg": np.degrees(theta_rad),
        "determinant_radius": determinant_radius,
        "valid": valid,
    }


def image_positions_from_reference(
    *,
    reference_image: Path,
    ra_rad: np.ndarray,
    dec_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _, header = read_reference_image_array_and_header(reference_image)
    wcs = WCS(header)
    finite = np.isfinite(ra_rad) & np.isfinite(dec_rad)
    x = np.full(ra_rad.shape, np.nan, dtype=np.float64)
    y = np.full(dec_rad.shape, np.nan, dtype=np.float64)
    if np.any(finite):
        x_pix, y_pix = wcs.all_world2pix(
            np.degrees(ra_rad[finite]), np.degrees(dec_rad[finite]), 1
        )
        x[finite] = x_pix
        y[finite] = y_pix
    return x, y


def read_reference_image_array_and_header(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
        candidates = []
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
            return np.asarray(hdu.data), hdu.header.copy()
    raise ValueError(f"reference image has no image data HDU: {path}")


def read_reference_image_context(path: Path) -> ReferenceImageContext:
    image, header = read_reference_image_array_and_header(path)
    x_ltv = header.get("LTV1")
    y_ltv = header.get("LTV2")
    if x_ltv is None or y_ltv is None:
        raise ValueError(
            f"reference image must have LTV1/LTV2 for full-pixel mapping: {path}"
        )
    return ReferenceImageContext(
        path=path,
        image=image,
        x_ltv=int(round(float(x_ltv))),
        y_ltv=int(round(float(y_ltv))),
    )


def build_output_rows(
    sources: dict[str, np.ndarray],
    measurement: dict[str, np.ndarray | int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = len(sources["row_index"])
    for index in range(n):
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
            "catalog_psf_flux_zp27": finite_float(
                sources["catalog_psf_flux_zp27"][index]
            ),
            "catalog_psf_flux_njy": finite_float(
                sources["catalog_psf_flux_njy"][index]
            ),
            "shape_xx": finite_float(sources["shape_xx"][index]),
            "shape_xy": finite_float(sources["shape_xy"][index]),
            "shape_yy": finite_float(sources["shape_yy"][index]),
            "axis_a": finite_float(sources["axis_a"][index]),
            "axis_b": finite_float(sources["axis_b"][index]),
            "theta_deg": finite_float(sources["theta_deg"][index]),
            "initial_determinant_radius": finite_float(
                sources["initial_determinant_radius"][index]
            ),
            "catalog_KronFlux_radius": finite_float(
                sources["catalog_kron_radius"][index]
            ),
            "catalog_KronFlux_radius_for_radius": finite_float(
                sources["catalog_kron_radius_for_radius"][index]
            ),
            "catalog_KronFlux_instFlux": finite_float(
                sources["catalog_kron_inst_flux"][index]
            ),
            "catalog_KronFlux_instFluxErr": finite_float(
                sources["catalog_kron_inst_flux_err"][index]
            ),
            "footprint_id": int(sources["footprint_id"][index]),
            "footprint_area": int(sources["footprint_area"][index]),
            "heavy_row": int(sources["heavy_row"][index]),
        }
        for key, value in measurement.items():
            if isinstance(value, np.ndarray):
                item = value[index]
                if value.dtype.kind in {"f"}:
                    row[key] = finite_float(item)
                elif value.dtype.kind in {"b"}:
                    row[key] = bool(item)
                elif value.dtype.kind in {"i", "u"}:
                    row[key] = int(item)
        row["proxy_nan0_radius_for_radius"] = row.get("heavyfp_nan0_radius_for_radius")
        row["proxy_nan0_sum_i"] = row.get("heavyfp_nan0_sum_i")
        row["proxy_nan0_sum_radius_i"] = row.get("heavyfp_nan0_sum_radius_i")
        row["proxy_nan0_raw_first_moment"] = row.get("heavyfp_nan0_raw_first_moment")
        row["proxy_nan0_candidate_radius"] = row.get("heavyfp_nan0_candidate_radius")
        row["proxy_nan0_determine_radius_returned_radius"] = row.get(
            "heavyfp_nan0_determine_radius_returned_radius"
        )
        row["proxy_nan0_flux_aperture_radius"] = row.get(
            "heavyfp_nan0_flux_aperture_radius"
        )
        row["proxy_nan0_candidate_le_initial_radius"] = row.get(
            "heavyfp_nan0_candidate_le_initial_radius"
        )
        row["proxy_nan0_good"] = row.get("heavyfp_nan0_good")
        row["pixel_value_count"] = row.get("heavy_value_count")
        row["official_minus_proxy_over_proxy"] = ratio_or_none(
            row.get("catalog_KronFlux_radius"),
            row.get("proxy_nan0_determine_radius_returned_radius"),
        )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_split_ds9_regions(
    output_dir: Path, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    specs = [
        {
            "name": "initial_shape",
            "filename": "initial_shape.reg",
            "target_radius_key": "initial_determinant_radius",
            "color": "yellow",
            "label": "initial_shape",
            "requires_heavy_good": False,
        },
        {
            "name": "official_radius_for_radius",
            "filename": "official_radius_for_radius.reg",
            "target_radius_key": "catalog_KronFlux_radius_for_radius",
            "color": "blue",
            "label": "official_radius_for_radius",
            "requires_heavy_good": False,
        },
        {
            "name": "official_kron_radius",
            "filename": "official_kron_radius.reg",
            "target_radius_key": "catalog_KronFlux_radius",
            "color": "red",
            "label": "official_kron_radius",
            "requires_heavy_good": False,
        },
        {
            "name": "proxy_radius_for_radius",
            "filename": "proxy_radius_for_radius.reg",
            "target_radius_key": "proxy_nan0_radius_for_radius",
            "color": "cyan",
            "label": "proxy_radius_for_radius",
            "requires_heavy_good": True,
        },
        {
            "name": "proxy_kron_radius",
            "filename": "proxy_kron_radius.reg",
            "target_radius_key": "proxy_nan0_determine_radius_returned_radius",
            "color": "green",
            "label": "proxy_kron_radius",
            "requires_heavy_good": True,
        },
        {
            "name": "proxy_flux_aperture",
            "filename": "proxy_flux_aperture.reg",
            "target_radius_key": "proxy_nan0_flux_aperture_radius",
            "color": "magenta",
            "label": "proxy_flux_aperture",
            "requires_heavy_good": True,
        },
    ]
    output: dict[str, Any] = {}
    for spec in specs:
        path = output_dir / str(spec["filename"])
        count = write_ds9_region_layer(
            path,
            rows,
            target_radius_key=str(spec["target_radius_key"]),
            color=str(spec["color"]),
            label=str(spec["label"]),
            requires_heavy_good=bool(spec["requires_heavy_good"]),
        )
        output[str(spec["name"])] = {
            "path": str(path),
            "count": int(count),
            "color": str(spec["color"]),
            "target_radius_key": str(spec["target_radius_key"]),
            "requires_heavy_good": bool(spec["requires_heavy_good"]),
        }
    return output


def write_ds9_region_layer(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    target_radius_key: str,
    color: str,
    label: str,
    requires_heavy_good: bool,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Region file format: DS9 version 4.1\n")
        handle.write(
            'global color={} dashlist=8 3 width=1 font="helvetica 10 normal roman"\n'.format(
                color
            )
        )
        handle.write("image\n")
        for row in rows:
            if requires_heavy_good and (
                row["status"] != "ok" or not bool(row.get("proxy_nan0_good", False))
            ):
                continue
            x = row["x_image"]
            y = row["y_image"]
            axis_a = row["axis_a"]
            axis_b = row["axis_b"]
            theta = row["theta_deg"]
            initial_radius = row["initial_determinant_radius"]
            target_radius = row.get(target_radius_key)
            if not all(
                is_number(value)
                for value in (
                    x,
                    y,
                    axis_a,
                    axis_b,
                    theta,
                    initial_radius,
                    target_radius,
                )
            ):
                continue
            if float(initial_radius) <= 0.0 or float(target_radius) <= 0.0:
                continue
            scale = float(target_radius) / float(initial_radius)
            text = "row={} mag={:.2f} {}".format(
                row["row_index"],
                row["official_psf_ab_mag"],
                "{} {}".format(label, row.get("measurement_surface", "unknown")),
            )
            write_ellipse(
                handle,
                x,
                y,
                axis_a * scale,
                axis_b * scale,
                theta,
                color=color,
                text=text,
            )
            count += 1
    return count


def write_ellipse(
    handle: Any,
    x: float,
    y: float,
    a: float,
    b: float,
    theta: float,
    *,
    color: str,
    text: str,
) -> None:
    if (
        not all(is_number(value) for value in (x, y, a, b, theta))
        or a <= 0.0
        or b <= 0.0
    ):
        return
    safe_text = str(text).replace("{", "(").replace("}", ")")
    handle.write(
        "ellipse({:.3f},{:.3f},{:.3f},{:.3f},{:.3f}) # color={} text={{{}}}\n".format(
            float(x),
            float(y),
            float(a),
            float(b),
            float(theta),
            color,
            safe_text,
        )
    )


def write_ratio_histogram(
    *,
    rows: list[dict[str, Any]],
    png_path: Path,
    csv_path: Path,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_values = ratio_values_for_rows(rows)
    stats = ratio_stats(all_values)
    if all_values.size == 0:
        bins = np.linspace(-1.0, 1.0, 81)
    else:
        q_low, q_high = np.nanpercentile(all_values, [0.5, 99.5])
        if not np.isfinite(q_low) or not np.isfinite(q_high) or q_low == q_high:
            center = float(np.nanmedian(all_values))
            q_low = center - 1.0
            q_high = center + 1.0
        pad = 0.05 * max(abs(float(q_high - q_low)), 1e-6)
        bins = np.linspace(float(q_low - pad), float(q_high + pad), 81)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    counts_all = (
        np.histogram(all_values, bins=bins)[0]
        if all_values.size
        else np.zeros(len(bins) - 1, dtype=int)
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "bin_lo",
                "bin_hi",
                "count_all",
            ],
        )
        writer.writeheader()
        for index in range(len(bins) - 1):
            writer.writerow(
                {
                    "bin_lo": float(bins[index]),
                    "bin_hi": float(bins[index + 1]),
                    "count_all": int(counts_all[index]),
                }
            )

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.hist(
        all_values,
        bins=bins,
        histtype="stepfilled",
        alpha=0.55,
        color="tab:blue",
        edgecolor="black",
        linewidth=0.3,
        label="all sources (n={})".format(stats["count"]),
    )
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    marker_specs = [
        ("p16", stats["p16"], "tab:orange", "--"),
        ("median", stats["median"], "tab:red", "-"),
        ("p84", stats["p84"], "tab:green", "--"),
    ]
    for label, value, color, linestyle in marker_specs:
        if value is None:
            continue
        ax.axvline(
            float(value),
            color=color,
            linewidth=1.8 if label == "median" else 1.4,
            linestyle=linestyle,
            label="{}={:.4g}".format(label, float(value)),
        )
    ax.set_xlabel("(official KronFlux radius - proxy Kron radius) / proxy Kron radius")
    ax.set_ylabel("source count")
    ax.set_title("Official minus proxy Kron-radius fractional difference")
    ax.legend(loc="best")
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    return {"all": stats} | {
        "histogram": {
            "bin_count": int(len(bins) - 1),
            "bin_min": float(bins[0]),
            "bin_max": float(bins[-1]),
            "csv": str(csv_path),
            "png": str(png_path),
        }
    }


def ratio_values_for_rows(rows: list[dict[str, Any]]) -> np.ndarray:
    values = [
        float(row["official_minus_proxy_over_proxy"])
        for row in rows
        if row.get("status") == "ok"
        and row.get("proxy_nan0_good")
        and row.get("official_minus_proxy_over_proxy") is not None
    ]
    return np.asarray(values, dtype=np.float64)


def ratio_stats(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "count": 0,
            "median": None,
            "mean": None,
            "std": None,
            "p16": None,
            "p84": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(finite.size),
        "median": float(np.nanmedian(finite)),
        "mean": float(np.nanmean(finite)),
        "std": float(np.nanstd(finite)),
        "p16": float(np.nanpercentile(finite, 16)),
        "p84": float(np.nanpercentile(finite, 84)),
        "min": float(np.nanmin(finite)),
        "max": float(np.nanmax(finite)),
    }


def write_summary_md(
    path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    processing = summary["processing"]
    lines = [
        "# Batch HeavyFootprint Kron Refit",
        "",
        "This is a sparse-pixel Kron proxy. It uses HeavyFootprintF payloads when "
        "available, and falls back to direct reference-image sampling inside the "
        "ordinary Footprint spans when no HeavyFootprintF exists. It does not "
        "rebuild NoiseReplacer and is not an exact production LSST Kron replay.",
        "",
        "## Counts",
        "",
        "- selected sources: `{}`".format(processing["n_sources_selected"]),
        "- processed sources: `{}`".format(processing["n_sources_processed"]),
        "- skipped sources: `{}`".format(processing["n_sources_skipped"]),
        "- total expanded pixels: `{}`".format(processing["total_expanded_pixels"]),
        "- max pixels per source: `{}`".format(processing["max_pixels_per_source"]),
        "- chunks: `{}`".format(processing["n_chunks"]),
        "",
        "## Status",
        "",
    ]
    for key, value in processing["status_counts"].items():
        lines.append("- {}: `{}`".format(key, value))
    lines.extend(["", "## Measurement Surfaces", ""])
    for key, value in processing["measurement_surface_counts"].items():
        lines.append("- {}: `{}`".format(key, value))
    good_rows = [
        row for row in rows if row["status"] == "ok" and row.get("proxy_nan0_good")
    ]
    if good_rows:
        returned = np.asarray(
            [row["proxy_nan0_determine_radius_returned_radius"] for row in good_rows],
            dtype=np.float64,
        )
        catalog = np.asarray(
            [row["catalog_KronFlux_radius"] for row in good_rows], dtype=np.float64
        )
        diff = returned - catalog
        lines.extend(
            [
                "",
                "## Radius Comparison",
                "",
                "- median proxy minus catalog radius: `{:.6f}` px".format(
                    float(np.nanmedian(diff))
                ),
                "- p16 proxy minus catalog radius: `{:.6f}` px".format(
                    float(np.nanpercentile(diff, 16))
                ),
                "- p84 proxy minus catalog radius: `{:.6f}` px".format(
                    float(np.nanpercentile(diff, 84))
                ),
            ]
        )
        ratio = summary["processing"]["official_minus_proxy_over_proxy"]
        lines.extend(
            [
                "",
                "## Fractional Radius Difference",
                "",
                "- definition: `(catalog_KronFlux_radius - proxy_radius) / proxy_radius`",
                "- count: `{}`".format(ratio["count"]),
                "- median: `{:.6f}`".format(ratio["median"]),
                "- p16: `{:.6f}`".format(ratio["p16"]),
                "- p84: `{:.6f}`".format(ratio["p84"]),
            ]
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- CSV: `{}`".format(summary["outputs"]["csv"]),
            "- JSON: `{}`".format(summary["outputs"]["json"]),
            "- fractional radius histogram PNG: `{}`".format(
                summary["outputs"]["official_minus_proxy_over_proxy_histogram_png"]
            ),
            "- fractional radius histogram CSV: `{}`".format(
                summary["outputs"]["official_minus_proxy_over_proxy_histogram_csv"]
            ),
            "- fractional radius stats JSON: `{}`".format(
                summary["outputs"]["official_minus_proxy_over_proxy_stats_json"]
            ),
            "- DS9 regions dir: `{}`".format(summary["outputs"]["ds9_regions_dir"]),
            "- DS9 regions manifest: `{}`".format(
                summary["outputs"]["regions_manifest"]
            ),
        ]
    )
    lines.append("")
    lines.append("## DS9 Regions")
    lines.append("")
    for name, path_value in summary["outputs"]["ds9_regions"].items():
        count = summary["processing"]["region_counts"].get(name, 0)
        lines.append("- {}: `{}` entries, `{}`".format(name, count, path_value))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def flags_array(main: fits.FITS_rec) -> np.ndarray:
    return np.asarray(main["flags"], dtype=bool)


def flag_name_map(header: fits.Header) -> dict[str, int]:
    return {header[key]: int(key[5:]) - 1 for key in header if key.startswith("TFLAG")}


def magnitude_from_zp27_flux(
    flux_zp27: np.ndarray, *, input_zeropoint: float
) -> np.ndarray:
    flux = np.asarray(flux_zp27, dtype=np.float64)
    mag = np.full(flux.shape, np.nan, dtype=np.float64)
    positive = np.isfinite(flux) & (flux > 0.0)
    mag[positive] = float(input_zeropoint) - 2.5 * np.log10(flux[positive])
    return mag


def magnitude_from_njy_flux(
    flux_njy: np.ndarray, *, output_zeropoint: float
) -> np.ndarray:
    flux = np.asarray(flux_njy, dtype=np.float64)
    mag = np.full(flux.shape, np.nan, dtype=np.float64)
    positive = np.isfinite(flux) & (flux > 0.0)
    mag[positive] = float(output_zeropoint) - 2.5 * np.log10(flux[positive])
    return mag


def decode_archive_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii", errors="ignore").strip()
    return str(value).strip()


def count_values(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def ratio_or_none(official: Any, proxy: Any) -> float | None:
    official_value = finite_float(official)
    proxy_value = finite_float(proxy)
    if official_value is None or proxy_value is None or proxy_value == 0.0:
        return None
    return float((official_value - proxy_value) / proxy_value)


def is_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
