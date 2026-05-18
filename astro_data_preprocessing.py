"""Preprocess HSC denoised coadds into AstroCELLECT training cutouts.

This script prepares fixed-size 512x512 tiles from denoised HSC projection
cutouts, crops a meas catalog into the same parent-patch coordinate boxes,
filters unreliable large 3-sigma ellipse sources, and writes precomputed dense
targets.  The output layout is intentionally compatible with
``astro_train_eval.py``:

    output_root/
      cutouts/<tile_name>/<band>/<source-fits-name>
      reference_catalogs/<tile_name>_meas.fits
      targets/<tile_name>.npz

The exposure-cropping logic follows the local
``segment-anything/lsst_pipeline/utils/run_cutout_magnitude_experiment.py``
helpers: LSST parent-patch coordinates are inferred from IMAGE LTV1/LTV2 and
IMAGE/MASK/VARIANCE are cropped together while keeping a 512x512 shape.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits
from astropy.table import Table


BANDS = ("HSC-G", "HSC-R", "HSC-I")
PIXEL_PLANES = ("IMAGE", "MASK", "VARIANCE")
DEFAULT_COADD_ROOT = Path("/home/chenzunhao/segment-anything/lsst_pipeline/fits/projection_cutout")
DEFAULT_CATALOG = Path("/home/chenzunhao/2026-05-01_scarlet_deblend_demo/catalog/HSC-I/meas-HSC-I-9813-4,5.fits")
DEFAULT_BAND_CATALOG_ROOT = Path("/home/chenzunhao/2026-05-01_scarlet_deblend_demo/catalog")
DEFAULT_OUTPUT_ROOT = Path("./output/hsc_astro_preprocessed")
DEFAULT_PARENT_ORIGIN = (15900, 19900)
DEFAULT_COMPARE_ORIGIN = (18204, 20924)


@dataclass(frozen=True)
class TileSpec:
    name: str
    x0: int
    y0: int
    size: int
    row: Optional[int] = None
    col: Optional[int] = None
    kind: str = "grid"

    @property
    def x1(self) -> int:
        return self.x0 + self.size

    @property
    def y1(self) -> int:
        return self.y0 + self.size


def _expand(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _origin_from_ltv(header: fits.Header) -> Tuple[int, int]:
    if "LTV1" not in header or "LTV2" not in header:
        raise KeyError("IMAGE header needs LTV1/LTV2 to infer parent-patch coordinates")
    return -int(round(float(header["LTV1"]))), -int(round(float(header["LTV2"])))


def _cropped_header(header: fits.Header, *, local_x0: int, local_y0: int) -> fits.Header:
    out = header.copy()
    if "LTV1" in out:
        out["LTV1"] = float(out["LTV1"]) - local_x0
    if "LTV2" in out:
        out["LTV2"] = float(out["LTV2"]) - local_y0
    if "CRPIX1" in out:
        out["CRPIX1"] = float(out["CRPIX1"]) - local_x0
    if "CRPIX2" in out:
        out["CRPIX2"] = float(out["CRPIX2"]) - local_y0
    if "CRVAL1A" in out:
        out["CRVAL1A"] = float(out["CRVAL1A"]) + local_x0
    if "CRVAL2A" in out:
        out["CRVAL2A"] = float(out["CRVAL2A"]) + local_y0
    return out


def _finite_replacement(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else 0.0


def _new_image_hdu_like(hdu, *, data: np.ndarray, header: fits.Header):
    if isinstance(hdu, fits.PrimaryHDU):
        return fits.PrimaryHDU(data=data, header=header)
    return fits.ImageHDU(data=data, header=header, name=hdu.name)


def crop_exposure_cutout(
    *,
    source_path: Path,
    output_path: Path,
    parent_x0: int,
    parent_y0: int,
    size: int,
    clean_nonfinite: bool = True,
    overwrite: bool = True,
) -> None:
    with fits.open(source_path, memmap=False) as hdul:
        if "IMAGE" not in hdul:
            raise KeyError(f"{source_path} has no IMAGE extension")
        source_origin = _origin_from_ltv(hdul["IMAGE"].header)
        local_x0 = int(parent_x0 - source_origin[0])
        local_y0 = int(parent_y0 - source_origin[1])

        for plane in PIXEL_PLANES:
            if plane not in hdul:
                raise KeyError(f"{source_path} has no {plane} extension")
            data = hdul[plane].data
            if data is None or data.ndim != 2:
                raise ValueError(f"{source_path}[{plane}] is not a 2D image")
            if local_x0 < 0 or local_y0 < 0 or local_x0 + size > data.shape[1] or local_y0 + size > data.shape[0]:
                raise ValueError(
                    f"{source_path}[{plane}] cannot cover parent cutout "
                    f"x={parent_x0}:{parent_x0 + size}, y={parent_y0}:{parent_y0 + size}; "
                    f"source origin={source_origin}, shape={data.shape}"
                )

        out_hdus = []
        for hdu in hdul:
            if hdu.name in PIXEL_PLANES:
                data = np.asarray(hdu.data[local_y0 : local_y0 + size, local_x0 : local_x0 + size]).copy()
                if clean_nonfinite and np.issubdtype(data.dtype, np.floating) and not np.all(np.isfinite(data)):
                    fill = _finite_replacement(data)
                    data = np.nan_to_num(data, nan=fill, posinf=fill, neginf=fill).astype(data.dtype, copy=False)
                header = _cropped_header(hdu.header, local_x0=local_x0, local_y0=local_y0)
                out_hdus.append(_new_image_hdu_like(hdu, data=data, header=header))
            else:
                out_hdus.append(hdu)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList(out_hdus).writeto(output_path, overwrite=overwrite)


def _band_fits_path(coadd_root: Path, band: str, tract: int, patch: str) -> Path:
    return coadd_root / band / f"deepCoadd-{band}-{tract}-{patch}.fits"


def _band_catalog_path(catalog_root: Path, band: str, tract: int, patch: str) -> Path:
    return catalog_root / band / f"meas-{band}-{tract}-{patch}.fits"


def edge_aligned_starts(length: int, tile_size: int, stride: int) -> List[int]:
    """Return floor-count stride starts with the final tile aligned to the edge.

    For length=4200, tile_size=512, stride=368 this returns 11 starts.  The last
    regular stride start is replaced by 4200-512=3688, matching the requested
    11x11 grid rather than adding a twelfth nearly duplicate edge tile.
    """

    if length <= tile_size:
        return [0]
    n = int(math.floor((length - tile_size) / stride)) + 1
    starts = [i * stride for i in range(max(n, 1))]
    starts[-1] = length - tile_size
    return starts


def make_tile_specs(
    *,
    parent_origin: Tuple[int, int],
    image_shape: Tuple[int, int],
    tile_size: int,
    stride: int,
    compare_origin: Optional[Tuple[int, int]],
) -> List[TileSpec]:
    width, height = image_shape
    x_starts = edge_aligned_starts(width, tile_size, stride)
    y_starts = edge_aligned_starts(height, tile_size, stride)
    specs: List[TileSpec] = []
    seen: set[Tuple[int, int]] = set()

    for row, y_local in enumerate(y_starts):
        for col, x_local in enumerate(x_starts):
            print(f"Adding tile spec for row={row}, col={col}, x_local={x_local}, y_local={y_local}")
            x0 = parent_origin[0] + x_local
            y0 = parent_origin[1] + y_local
            name = f"grid_r{row:02d}_c{col:02d}_x{x0}_y{y0}"
            specs.append(TileSpec(name=name, x0=x0, y0=y0, size=tile_size, row=row, col=col))
            seen.add((x0, y0))

    if compare_origin is not None and compare_origin not in seen:
        x0, y0 = compare_origin
        specs.append(TileSpec(name=f"sam_x{x0}_y{y0}", x0=x0, y0=y0, size=tile_size, kind="compare"))
    return specs


def _first_finite_column(table: Table, names: Sequence[str]) -> Optional[np.ndarray]:
    for name in names:
        if name in table.colnames:
            vals = np.asarray(table[name], dtype=np.float32)
            if np.isfinite(vals).any():
                return vals
    return None


def _require_position_columns(table: Table, x_col: str, y_col: str) -> Tuple[np.ndarray, np.ndarray]:
    if x_col not in table.colnames or y_col not in table.colnames:
        raise KeyError(f"catalog must contain {x_col!r} and {y_col!r}")
    return np.asarray(table[x_col], dtype=np.float32), np.asarray(table[y_col], dtype=np.float32)


def ellipse_parameters(
    table: Table,
    *,
    shape_source: str = "sdss",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if shape_source == "hsm" and {
        "ext_shapeHSM_HsmSourceMoments_xx",
        "ext_shapeHSM_HsmSourceMoments_yy",
        "ext_shapeHSM_HsmSourceMoments_xy",
    }.issubset(table.colnames):
        xx = np.asarray(table["ext_shapeHSM_HsmSourceMoments_xx"], dtype=np.float32)
        yy = np.asarray(table["ext_shapeHSM_HsmSourceMoments_yy"], dtype=np.float32)
        xy = np.asarray(table["ext_shapeHSM_HsmSourceMoments_xy"], dtype=np.float32)
    elif {"base_SdssShape_xx", "base_SdssShape_yy", "base_SdssShape_xy"}.issubset(table.colnames):
        xx = np.asarray(table["base_SdssShape_xx"], dtype=np.float32)
        yy = np.asarray(table["base_SdssShape_yy"], dtype=np.float32)
        xy = np.asarray(table["base_SdssShape_xy"], dtype=np.float32)
    else:
        n = len(table)
        xx = np.full(n, 4.0, dtype=np.float32)
        yy = np.full(n, 4.0, dtype=np.float32)
        xy = np.zeros(n, dtype=np.float32)

    xx = np.maximum(xx, 0.25)
    yy = np.maximum(yy, 0.25)
    trace = xx + yy
    delta = np.sqrt(np.maximum((xx - yy) ** 2 + 4.0 * xy ** 2, 0.0))
    major = np.sqrt(np.maximum(0.5 * (trace + delta), 0.25))
    minor = np.sqrt(np.maximum(0.5 * (trace - delta), 0.25))
    angle = 0.5 * np.arctan2(2.0 * xy, xx - yy)
    return major.astype(np.float32), minor.astype(np.float32), angle.astype(np.float32)


def add_ellipse_columns(table: Table, *, shape_source: str) -> Table:
    out = table.copy(copy_data=True)
    major, minor, angle = ellipse_parameters(out, shape_source=shape_source)
    out["ellipse_major_sigma"] = major
    out["ellipse_minor_sigma"] = minor
    out["ellipse_theta"] = angle
    out["ellipse_area_3sigma"] = (9.0 * math.pi * major * minor).astype(np.float32)
    return out


def filter_sources(
    table: Table,
    *,
    x_col: str,
    y_col: str,
    shape_source: str,
    max_area_3sigma: float,
    relaxed_area_3sigma: float,
    area_filter_policy: str,
    source_filter: str,
    drop_children: bool,
) -> Tuple[Table, Table]:
    table = add_ellipse_columns(table, shape_source=shape_source)
    x, y = _require_position_columns(table, x_col, y_col)
    valid = np.isfinite(x) & np.isfinite(y)
    area = np.asarray(table["ellipse_area_3sigma"], dtype=np.float32)
    valid &= np.isfinite(area)
    if area_filter_policy == "max_area":
        valid &= area <= float(max_area_3sigma)
    elif area_filter_policy == "area400_or_area900_not_bright":
        bright = bright_object_mask(table)
        # Keep all compact sources.  For moderately large sources, keep only
        # rows whose LSST/HSC bright-object flags do not mark the source or its
        # footprint as near a bright object.
        valid &= (area < float(max_area_3sigma)) | ((area < float(relaxed_area_3sigma)) & ~bright)
    else:
        raise ValueError(f"Unknown area_filter_policy: {area_filter_policy}")
    valid &= source_selection_mask(table, source_filter)
    if drop_children and "parent" in table.colnames:
        valid &= np.asarray(table["parent"], dtype=np.int64) == 0
    return table[valid], table[~valid]


def bright_object_mask(table: Table) -> np.ndarray:
    """Return rows flagged as on or near bright-object masked pixels."""

    bright = np.zeros(len(table), dtype=bool)
    for name in ("base_PixelFlags_flag_bright_object", "base_PixelFlags_flag_bright_objectCenter"):
        if name in table.colnames:
            bright |= np.asarray(table[name], dtype=bool)
    return bright


def source_selection_mask(table: Table, source_filter: str) -> np.ndarray:
    """Select catalog rows used as source-level supervision.

    HSC/LSST deblender catalogs contain parent rows and deblended child rows.
    For source detection/deblending we train on leaf sources by default:
    ``deblend_nChild == 0``. This keeps isolated sources and final children,
    while excluding parent blend containers.
    """

    n = len(table)
    if source_filter == "all":
        return np.ones(n, dtype=bool)
    if source_filter == "parent":
        if "parent" not in table.colnames:
            raise KeyError("source_filter='parent' requires a parent column")
        return np.asarray(table["parent"], dtype=np.int64) == 0

    child_col = "deblend_nChild" if "deblend_nChild" in table.colnames else "nChild" if "nChild" in table.colnames else None
    if child_col is None:
        raise KeyError(f"source_filter='{source_filter}' requires deblend_nChild or nChild column")
    leaf = np.asarray(table[child_col], dtype=np.int64) == 0
    if source_filter == "nchild0":
        return leaf
    if source_filter == "leaf_child":
        if "parent" not in table.colnames:
            raise KeyError("source_filter='leaf_child' requires a parent column")
        return leaf & (np.asarray(table["parent"], dtype=np.int64) != 0)
    raise ValueError(f"Unknown source_filter: {source_filter}")


def _contains(x: np.ndarray, y: np.ndarray, spec: TileSpec, *, margin: float = 0.0) -> np.ndarray:
    finite = np.isfinite(x) & np.isfinite(y)
    return (
        finite
        & (x >= spec.x0 - margin)
        & (x < spec.x1 + margin)
        & (y >= spec.y0 - margin)
        & (y < spec.y1 + margin)
    )


def crop_catalog_for_tile(
    table: Table,
    spec: TileSpec,
    *,
    x_col: str,
    y_col: str,
    margin: float,
) -> Table:
    x, y = _require_position_columns(table, x_col, y_col)
    cropped = table[_contains(x, y, spec, margin=margin)]
    out = cropped.copy(copy_data=True)
    out["centroid_local_x"] = np.asarray(out[x_col], dtype=np.float32) - float(spec.x0)
    out["centroid_local_y"] = np.asarray(out[y_col], dtype=np.float32) - float(spec.y0)
    return out


def _table_for_csv(table: Table) -> Table:
    out = Table()
    for name in table.colnames:
        arr = np.asarray(table[name])
        if arr.ndim <= 1:
            out[name] = table[name]
        else:
            out[name] = [json.dumps(np.asarray(row).tolist(), separators=(",", ":")) for row in arr]
    return out


def write_table_pair(table: Table, fits_path: Path, csv_path: Optional[Path] = None) -> None:
    fits_path.parent.mkdir(parents=True, exist_ok=True)
    table.write(fits_path, format="fits", overwrite=True)
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _table_for_csv(table).write(csv_path, format="ascii.csv", overwrite=True)


def make_dense_targets(
    sources: Table,
    spec: TileSpec,
    *,
    x_col: str,
    y_col: str,
    ellipse_sigma: float,
    confidence_levels: int,
    core_radius: int,
) -> Dict[str, np.ndarray]:
    h = w = spec.size
    seg = np.zeros((h, w), dtype=np.int16)
    confidence = np.zeros((h, w), dtype=np.int16)
    shape = np.zeros((3, h, w), dtype=np.float32)
    shape_weight = np.zeros((h, w), dtype=np.float32)
    overlap_count = np.zeros((h, w), dtype=np.int16)

    if len(sources) == 0:
        return {
            "seg": seg,
            "confidence": confidence,
            "shape": shape,
            "shape_weight": shape_weight,
            "overlap_count": overlap_count,
        }

    x_global, y_global = _require_position_columns(sources, x_col, y_col)
    major = np.asarray(sources["ellipse_major_sigma"], dtype=np.float32)
    minor = np.asarray(sources["ellipse_minor_sigma"], dtype=np.float32)
    angle = np.asarray(sources["ellipse_theta"], dtype=np.float32)
    yy_full, xx_full = np.mgrid[0:h, 0:w]

    for idx in range(len(sources)):
        cx = float(x_global[idx] - spec.x0)
        cy = float(y_global[idx] - spec.y0)
        a = float(max(major[idx] * ellipse_sigma, 1.5))
        b = float(max(minor[idx] * ellipse_sigma, 1.5))
        theta = float(angle[idx])
        radius = int(math.ceil(max(a, b))) + 2
        cx_i = int(round(cx))
        cy_i = int(round(cy))
        y0, y1 = max(0, cy_i - radius), min(h, cy_i + radius + 1)
        x0, x1 = max(0, cx_i - radius), min(w, cx_i + radius + 1)
        if y0 < y1 and x0 < x1:
            dx = xx_full[y0:y1, x0:x1] - cx
            dy = yy_full[y0:y1, x0:x1] - cy
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            xr = cos_t * dx + sin_t * dy
            yr = -sin_t * dx + cos_t * dy
            ellipse = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
            if ellipse.any():
                patch_seg = seg[y0:y1, x0:x1]
                patch_overlap = overlap_count[y0:y1, x0:x1]
                patch_weight = shape_weight[y0:y1, x0:x1]
                patch_seg[ellipse] = np.maximum(patch_seg[ellipse], 1)
                patch_overlap[ellipse] += 1
                shape[:, y0:y1, x0:x1][:, ellipse] = np.array([major[idx], minor[idx], theta], dtype=np.float32)[:, None]
                patch_weight[ellipse] = 1.0

        # Confidence and center core only supervise sources whose centers are in
        # the tile; sources outside the tile may still paint overlapping ellipses.
        if 0 <= cx_i < w and 0 <= cy_i < h:
            level_radius = confidence_levels - 1
            cy0, cy1 = max(0, cy_i - level_radius), min(h, cy_i + level_radius + 1)
            cx0, cx1 = max(0, cx_i - level_radius), min(w, cx_i + level_radius + 1)
            dist = np.abs(xx_full[cy0:cy1, cx0:cx1] - cx_i) + np.abs(yy_full[cy0:cy1, cx0:cx1] - cy_i)
            vals = np.clip(level_radius - dist, a_min=0, a_max=None).astype(np.int16)
            confidence[cy0:cy1, cx0:cx1] = np.maximum(confidence[cy0:cy1, cx0:cx1], vals)

            ky0, ky1 = max(0, cy_i - core_radius), min(h, cy_i + core_radius + 1)
            kx0, kx1 = max(0, cx_i - core_radius), min(w, cx_i + core_radius + 1)
            core = np.abs(xx_full[ky0:ky1, kx0:kx1] - cx_i) + np.abs(yy_full[ky0:ky1, kx0:kx1] - cy_i)
            seg[ky0:ky1, kx0:kx1][core <= core_radius] = 2

    return {
        "seg": seg,
        "confidence": confidence,
        "shape": shape,
        "shape_weight": shape_weight,
        "overlap_count": overlap_count,
    }


def write_targets(targets: Dict[str, np.ndarray], output_npz: Path, output_fits_prefix: Optional[Path]) -> None:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **targets)
    if output_fits_prefix is None:
        return
    output_fits_prefix.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(output_fits_prefix.with_name(output_fits_prefix.name + "_seg.fits"), targets["seg"], overwrite=True)
    fits.writeto(
        output_fits_prefix.with_name(output_fits_prefix.name + "_confidence.fits"),
        targets["confidence"],
        overwrite=True,
    )
    fits.writeto(
        output_fits_prefix.with_name(output_fits_prefix.name + "_overlap.fits"),
        targets["overlap_count"],
        overwrite=True,
    )


def _jsonable_args(args: argparse.Namespace) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def preprocess(args: argparse.Namespace) -> None:
    coadd_root = _expand(args.coadd_root)
    catalog_path = _expand(args.catalog)
    output_root = _expand(args.output_root)
    bands = tuple(args.bands)

    first_band = bands[0]
    first_path = _band_fits_path(coadd_root, first_band, args.tract, args.patch)
    with fits.open(first_path, memmap=True) as hdul:
        image_shape_yx = hdul["IMAGE"].data.shape
        image_origin = _origin_from_ltv(hdul["IMAGE"].header)
    width, height = int(image_shape_yx[1]), int(image_shape_yx[0])
    if (width, height) != (args.image_width, args.image_height):
        print(f"Detected image shape width={width}, height={height}; overriding CLI image size.")

    parent_origin = tuple(args.parent_origin)
    if image_origin != parent_origin:
        print(f"WARNING: FITS LTV origin {image_origin} differs from requested parent origin {parent_origin}.")

    compare_origin = tuple(args.compare_origin) if args.compare_origin else None
    specs = make_tile_specs(
        parent_origin=parent_origin,
        image_shape=(width, height),
        tile_size=args.tile_size,
        stride=args.stride,
        compare_origin=compare_origin,
    )
    if args.max_tiles is not None:
        specs = specs[: int(args.max_tiles)]

    table = Table.read(catalog_path, hdu=args.catalog_hdu)
    filtered, rejected = filter_sources(
        table,
        x_col=args.x_col,
        y_col=args.y_col,
        shape_source=args.shape_source,
        max_area_3sigma=args.max_area_3sigma,
        relaxed_area_3sigma=args.relaxed_area_3sigma,
        area_filter_policy=args.area_filter_policy,
        source_filter=args.source_filter,
        drop_children=args.drop_children,
    )

    sources_dir = output_root / "sources"
    if not args.dry_run:
        write_table_pair(filtered, sources_dir / "sources_filtered.fits", sources_dir / "sources_filtered.csv")
        write_table_pair(rejected, sources_dir / "sources_rejected.fits", sources_dir / "sources_rejected.csv")

    if args.band_catalog_root is not None and not args.dry_run:
        band_catalog_root = _expand(args.band_catalog_root)
        for band in bands:
            band_catalog_path = _band_catalog_path(band_catalog_root, band, args.tract, args.patch)
            band_table = Table.read(band_catalog_path, hdu=args.catalog_hdu)
            band_filtered, band_rejected = filter_sources(
                band_table,
                x_col=args.x_col,
                y_col=args.y_col,
                shape_source=args.shape_source,
                max_area_3sigma=args.max_area_3sigma,
                relaxed_area_3sigma=args.relaxed_area_3sigma,
                area_filter_policy=args.area_filter_policy,
                source_filter=args.source_filter,
                drop_children=args.drop_children,
            )
            band_ref_dir = output_root / "band_reference_catalogs" / band
            write_table_pair(
                band_filtered,
                band_ref_dir / f"meas-{band}-{args.tract}-{args.patch}.fits",
                band_ref_dir / f"meas-{band}-{args.tract}-{args.patch}.csv",
            )
            write_table_pair(
                band_rejected,
                output_root / "band_reference_rejected" / band / f"meas-{band}-{args.tract}-{args.patch}.fits",
                None,
            )

    manifest_rows = []
    cutout_paths: Dict[str, Dict[str, str]] = {}

    if args.dry_run:
        print(f"dry-run: would write {len(specs)} tiles to {output_root}")
    else:
        output_root.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        tile_catalog = crop_catalog_for_tile(
            filtered,
            spec,
            x_col=args.x_col,
            y_col=args.y_col,
            margin=0.0,
        )
        mask_sources = crop_catalog_for_tile(
            filtered,
            spec,
            x_col=args.x_col,
            y_col=args.y_col,
            margin=args.mask_margin,
        )

        band_paths: Dict[str, str] = {}
        for band in bands:
            src = _band_fits_path(coadd_root, band, args.tract, args.patch)
            dst = output_root / "cutouts" / spec.name / band / src.name
            band_paths[band] = str(dst)
            if not args.skip_cutouts and not args.dry_run and (args.overwrite or not dst.exists()):
                crop_exposure_cutout(
                    source_path=src,
                    output_path=dst,
                    parent_x0=spec.x0,
                    parent_y0=spec.y0,
                    size=spec.size,
                    clean_nonfinite=not args.no_clean_nonfinite,
                    overwrite=True,
                )

        targets = make_dense_targets(
            mask_sources,
            spec,
            x_col=args.x_col,
            y_col=args.y_col,
            ellipse_sigma=args.ellipse_sigma,
            confidence_levels=args.confidence_levels,
            core_radius=args.core_radius,
        )

        ref_path = output_root / "reference_catalogs" / f"{spec.name}_meas.fits"
        ref_csv = output_root / "reference_catalogs_csv" / f"{spec.name}_meas.csv"
        target_path = output_root / "targets" / f"{spec.name}.npz"
        target_fits_prefix = output_root / "target_fits" / spec.name if args.write_target_fits else None
        if not args.dry_run:
            write_table_pair(tile_catalog, ref_path, ref_csv)
            write_targets(targets, target_path, target_fits_prefix)

        cutout_paths[spec.name] = band_paths
        manifest_rows.append(
            {
                "name": spec.name,
                "kind": spec.kind,
                "row": spec.row,
                "col": spec.col,
                "x0": spec.x0,
                "y0": spec.y0,
                "x1": spec.x1,
                "y1": spec.y1,
                "size": spec.size,
                "n_sources_center_in_tile": len(tile_catalog),
                "n_sources_for_mask_with_margin": len(mask_sources),
                "seg_foreground_pixels": int((targets["seg"] > 0).sum()),
                "overlap_pixels": int((targets["overlap_count"] >= 2).sum()),
            }
        )

    metadata = {
        "coadd_root": str(coadd_root),
        "catalog": str(catalog_path),
        "output_root": str(output_root),
        "bands": bands,
        "tract": args.tract,
        "patch": args.patch,
        "parent_origin": parent_origin,
        "image_shape": {"width": width, "height": height},
        "tile_size": args.tile_size,
        "stride": args.stride,
        "compare_origin": compare_origin,
        "num_tiles": len(specs),
        "num_filtered_sources": len(filtered),
        "num_rejected_sources": len(rejected),
        "source_filter": args.source_filter,
        "max_area_3sigma": args.max_area_3sigma,
        "relaxed_area_3sigma": args.relaxed_area_3sigma,
        "area_filter_policy": args.area_filter_policy,
        "ellipse_sigma": args.ellipse_sigma,
        "mask_margin": args.mask_margin,
        "drop_children": args.drop_children,
        "skip_cutouts": args.skip_cutouts,
        "band_catalog_root": str(_expand(args.band_catalog_root)) if args.band_catalog_root is not None else None,
        "args": _jsonable_args(args),
    }

    if not args.dry_run:
        (output_root / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        with (output_root / "tiles.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
        (output_root / "cutout_paths.json").write_text(json.dumps(cutout_paths, indent=2), encoding="utf-8")

    print(
        f"prepared {len(specs)} tiles; filtered sources={len(filtered)}, rejected={len(rejected)}; "
        f"output={output_root}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess HSC denoised coadds for AstroCELLECT training.")
    parser.add_argument("--coadd-root", type=Path, default=DEFAULT_COADD_ROOT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--band-catalog-root",
        type=Path,
        default=DEFAULT_BAND_CATALOG_ROOT,
        help="Optional root containing per-band meas catalogs. Filtered copies are written for EX/EN training.",
    )
    parser.add_argument("--catalog-hdu", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bands", nargs="+", default=list(BANDS))
    parser.add_argument("--tract", type=int, default=9813)
    parser.add_argument("--patch", default="4,5")
    parser.add_argument("--parent-origin", nargs=2, type=int, default=list(DEFAULT_PARENT_ORIGIN), metavar=("X0", "Y0"))
    parser.add_argument("--image-width", type=int, default=4200)
    parser.add_argument("--image-height", type=int, default=4200)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=368)
    parser.add_argument(
        "--compare-origin",
        nargs=2,
        type=int,
        default=list(DEFAULT_COMPARE_ORIGIN),
        metavar=("X0", "Y0"),
        help="Extra 512x512 tile kept for comparison with segment-anything cutouts. Use --no-compare-origin to disable.",
    )
    parser.add_argument("--no-compare-origin", action="store_true")
    parser.add_argument("--x-col", default="base_SdssCentroid_x")
    parser.add_argument("--y-col", default="base_SdssCentroid_y")
    parser.add_argument("--shape-source", choices=("sdss", "hsm"), default="sdss")
    parser.add_argument("--max-area-3sigma", type=float, default=400.0)
    parser.add_argument("--relaxed-area-3sigma", type=float, default=900.0)
    parser.add_argument(
        "--area-filter-policy",
        choices=("max_area", "area400_or_area900_not_bright"),
        default="area400_or_area900_not_bright",
        help="Source quality filter. The default keeps area<400, plus area<900 only when bright-object flags are false.",
    )
    parser.add_argument(
        "--source-filter",
        choices=("nchild0", "all", "parent", "leaf_child"),
        default="nchild0",
        help="Catalog rows used for reference catalogs and dense targets. Default trains on deblend_nChild==0 leaf sources.",
    )
    parser.add_argument("--drop-children", action="store_true", help="Legacy extra filter: also drop rows with parent != 0.")
    parser.add_argument("--ellipse-sigma", type=float, default=3.0)
    parser.add_argument("--mask-margin", type=float, default=64.0)
    parser.add_argument("--confidence-levels", type=int, default=5)
    parser.add_argument("--core-radius", type=int, default=2)
    parser.add_argument("--write-target-fits", action="store_true")
    parser.add_argument("--max-tiles", type=int, default=None, help="Optional debug limit after tile generation.")
    parser.add_argument("--no-clean-nonfinite", action="store_true")
    parser.add_argument("--skip-cutouts", action="store_true", help="Skip FITS cutout creation; still write catalogs and targets.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.no_compare_origin:
        args.compare_origin = None
    preprocess(args)


if __name__ == "__main__":
    main()
