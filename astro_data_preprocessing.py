"""Preprocess HSC denoised or official coadds into AstroCELLECT training cutouts.

This script prepares fixed-size 512x512 tiles from denoised HSC projection
cutouts, crops a meas catalog into the same parent-patch coordinate boxes,
filters unreliable large 3-sigma ellipse sources, and writes precomputed dense
targets.  The output layout is intentionally compatible with
``astro_train_eval.py``:

    output_root/<tract>/<patch>/
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
import os
import shutil
import traceback
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
DEFAULT_CATALOG_BAND = "HSC-I"


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
        return 0, 0
    return -int(round(float(header["LTV1"]))), -int(round(float(header["LTV2"])))


def _find_image_hdu_index(hdul: fits.HDUList) -> int:
    if "IMAGE" in hdul:
        return hdul.index_of("IMAGE")
    for idx, hdu in enumerate(hdul):
        data = getattr(hdu, "data", None)
        if data is not None and getattr(data, "ndim", None) == 2:
            return idx
    raise KeyError("No 2D image HDU found; expected IMAGE or a 2D image extension")


def _plane_hdu_indices(hdul: fits.HDUList) -> Dict[str, int]:
    if all(plane in hdul for plane in PIXEL_PLANES):
        return {plane: hdul.index_of(plane) for plane in PIXEL_PLANES}

    image_idx = _find_image_hdu_index(hdul)
    indices = {"IMAGE": image_idx}
    image_shape = hdul[image_idx].data.shape
    for plane, idx in (("MASK", image_idx + 1), ("VARIANCE", image_idx + 2)):
        if idx < len(hdul):
            data = getattr(hdul[idx], "data", None)
            if data is not None and getattr(data, "ndim", None) == 2 and data.shape == image_shape:
                indices[plane] = idx
    return indices


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
        plane_indices = _plane_hdu_indices(hdul)
        source_origin = _origin_from_ltv(hdul[plane_indices["IMAGE"]].header)
        local_x0 = int(parent_x0 - source_origin[0])
        local_y0 = int(parent_y0 - source_origin[1])

        for plane, idx in plane_indices.items():
            data = hdul[idx].data
            if data is None or data.ndim != 2:
                raise ValueError(f"{source_path}[{plane}] is not a 2D image")
            if local_x0 < 0 or local_y0 < 0 or local_x0 + size > data.shape[1] or local_y0 + size > data.shape[0]:
                raise ValueError(
                    f"{source_path}[{plane}] cannot cover parent cutout "
                    f"x={parent_x0}:{parent_x0 + size}, y={parent_y0}:{parent_y0 + size}; "
                    f"source origin={source_origin}, shape={data.shape}"
                )

        out_hdus = [fits.PrimaryHDU(header=hdul[0].header if len(hdul) else None)]
        for plane in PIXEL_PLANES:
            idx = plane_indices.get(plane)
            if idx is None:
                continue
            hdu = hdul[idx]
            data = np.asarray(hdu.data[local_y0 : local_y0 + size, local_x0 : local_x0 + size]).copy()
            if clean_nonfinite and np.issubdtype(data.dtype, np.floating) and not np.all(np.isfinite(data)):
                fill = _finite_replacement(data)
                data = np.nan_to_num(data, nan=fill, posinf=fill, neginf=fill).astype(data.dtype, copy=False)
            header = _cropped_header(hdu.header, local_x0=local_x0, local_y0=local_y0)
            out_hdus.append(fits.ImageHDU(data=data, header=header, name=plane))

        if all(plane in hdul for plane in PIXEL_PLANES):
            named_indices = set(plane_indices.values())
            for idx, hdu in enumerate(hdul):
                if idx == 0 or idx in named_indices:
                    continue
                out_hdus.append(hdu)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList(out_hdus).writeto(output_path, overwrite=overwrite)


def _band_fits_path(coadd_root: Path, band: str, tract: int, patch: str) -> Path:
    filenames = [
        f"deepCoadd-{band}-{tract}-{patch}.fits",
        f"calexp-{band}-{tract}-{patch}.fits",
    ]
    candidates = [
        base / filename
        for base in (
            coadd_root / band,
            coadd_root / band / patch,
            coadd_root / str(tract) / band / patch,
            coadd_root / str(tract) / band,
        )
        for filename in filenames
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    search_dirs = [
        coadd_root / band,
        coadd_root / band / patch,
        coadd_root / str(tract) / band / patch,
        coadd_root / str(tract) / band,
    ]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        matches = sorted(
            path
            for path in search_dir.glob(f"*{band}*{tract}*{patch}*.fits")
            if not path.name.startswith(("meas-", "det-", "det_bkgd-"))
        )
        if matches:
            return matches[0]
    return candidates[0]


def _band_catalog_path(catalog_root: Path, band: str, tract: int, patch: str) -> Path:
    filename = f"meas-{band}-{tract}-{patch}.fits"
    candidates = [
        catalog_root / band / filename,
        catalog_root / band / patch / filename,
        catalog_root / str(tract) / band / patch / filename,
        catalog_root / str(tract) / band / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _existing_cutout_fits_path(tile_dir: Path, band: str) -> str:
    band_dir = tile_dir / band
    if not band_dir.exists():
        raise FileNotFoundError(f"Missing band directory: {band_dir}")
    matches = sorted(band_dir.glob("*.fits"))
    if not matches:
        raise FileNotFoundError(f"No FITS cutout found in {band_dir}")
    return str(matches[0])


def _split_patch_tokens(values: Iterable[str]) -> List[str]:
    patches: List[str] = []
    for value in values:
        for token in str(value).replace(";", " ").split():
            patch = token.strip()
            if patch:
                patches.append(patch)
    return patches


def _expand_patch_specs(values: Iterable[str], patch_file: Optional[Path]) -> List[str]:
    raw = _split_patch_tokens(values)
    if patch_file is not None:
        for line in patch_file.expanduser().read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                raw.extend(_split_patch_tokens([line]))
    if not raw:
        raise ValueError("No patch specified")

    patches: List[str] = []
    seen: set[str] = set()
    for patch in raw:
        expanded = [f"{x},{y}" for x in range(9) for y in range(9)] if patch.lower() == "all" else [patch]
        for item in expanded:
            if item in seen:
                continue
            patches.append(item)
            seen.add(item)
    return patches


def _patch_output_root(output_root: Path, tract: int, patch: str) -> Path:
    return output_root / str(tract) / patch


def _catalog_path_for_patch(args: argparse.Namespace, catalog_root: Path, patch: str, num_patches: int) -> Path:
    if args.catalog is not None:
        if num_patches > 1:
            raise ValueError("--catalog points to one FITS file; use --catalog-root for multi-patch preprocessing")
        return _expand(args.catalog)
    return _band_catalog_path(catalog_root, args.catalog_band, args.tract, patch)


def _worker_count(requested: int, num_patches: int) -> int:
    if num_patches <= 1:
        return 1
    if requested == 0:
        return max(1, min(num_patches, os.cpu_count() or 1))
    return max(1, min(int(requested), num_patches))


def _read_table(path: Path, *, hdu: int, role: str, patch: str, band: Optional[str] = None) -> Table:
    try:
        return Table.read(path, hdu=hdu)
    except Exception as exc:
        band_text = f", band={band}" if band else ""
        raise RuntimeError(f"Failed to read {role} catalog for patch={patch}{band_text}: {path}") from exc


def _filter_catalog(
    table: Table,
    args: argparse.Namespace,
) -> Tuple[Table, Table]:
    return filter_sources(
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


def _zscale_cache_path(
    zscale_root: Path,
    *,
    tract: int,
    patch: str,
    tile_name: str,
    bands: Sequence[str],
    fits_hdu: int,
) -> Path:
    band_key = "_".join(bands)
    return zscale_root / str(tract) / patch / "cutouts" / f"{tile_name}__{band_key}__hdu{fits_hdu}.pt"


def write_zscale_cache(
    image_paths: Sequence[str],
    output_path: Path,
    *,
    fits_hdu: int,
    overwrite: bool,
) -> None:
    try:
        import torch
        from astro_cellect2d import astro_zscale_preprocess, read_fits_bands
    except Exception as exc:
        raise RuntimeError("--zscale-root requires torch and astro_cellect2d dependencies") from exc

    if output_path.exists() and not overwrite:
        return
    image_np = read_fits_bands(tuple(image_paths), hdu=fits_hdu)
    image = astro_zscale_preprocess(image_np).to(dtype=torch.float32).cpu()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    torch.save(image, tmp_path)
    tmp_path.replace(output_path)


def mirror_fast_outputs(source_root: Path, fast_root: Path) -> None:
    dirs = (
        "targets",
        "reference_catalogs",
        "reference_catalogs_csv",
        "band_reference_catalogs",
        "band_reference_rejected",
        "sources",
    )
    files = ("manifest.json", "tiles.csv", "cutout_paths.json")
    fast_root.mkdir(parents=True, exist_ok=True)
    for dirname in dirs:
        src = source_root / dirname
        if src.exists():
            shutil.copytree(src, fast_root / dirname, dirs_exist_ok=True)
    for filename in files:
        src = source_root / filename
        if src.exists():
            shutil.copy2(src, fast_root / filename)


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


def _patch_failure_row(patch: str, catalog_path: Path, output_root: Path, exc: BaseException) -> Dict[str, str]:
    return {
        "patch": patch,
        "catalog": str(catalog_path),
        "output_root": str(output_root),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def _write_failed_patches(data_root: Path, failed_rows: Sequence[Dict[str, str]]) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "preprocess_failed_patches.json").write_text(json.dumps(list(failed_rows), indent=2), encoding="utf-8")
    with (data_root / "preprocess_failed_patches.csv").open("w", newline="") as handle:
        fieldnames = ["patch", "catalog", "output_root", "error_type", "error"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in failed_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def preprocess(args: argparse.Namespace) -> None:
    coadd_root = _expand(args.coadd_root)
    catalog_root = _expand(args.catalog_root)
    data_root = _expand(args.output_root)
    bands = tuple(args.bands)
    patch_values = args.patches if args.patches else [args.patch]
    patches = _expand_patch_specs(patch_values, args.patch_file)
    num_workers = _worker_count(args.num_workers, len(patches))

    tasks = [
        (
            patch,
            _catalog_path_for_patch(args, catalog_root, patch, len(patches)),
            _patch_output_root(data_root, args.tract, patch),
        )
        for patch in patches
    ]

    summaries_by_patch: Dict[str, Dict[str, object]] = {}
    failed_patch_rows: List[Dict[str, str]] = []
    worker_fn = _sync_existing_patch if args.reuse_existing_preprocessed else _preprocess_patch
    if num_workers == 1:
        for patch, catalog_path, patch_output_root in tasks:
            try:
                summaries_by_patch[patch] = worker_fn(
                    args,
                    coadd_root=coadd_root,
                    catalog_path=catalog_path,
                    output_root=patch_output_root,
                    bands=bands,
                    patch=patch,
                )
            except Exception as exc:
                failed_patch_rows.append(_patch_failure_row(patch, catalog_path, patch_output_root, exc))
                print(f"FAILED patch {patch}: {exc}", flush=True)
    else:
        if not args.dry_run:
            data_root.mkdir(parents=True, exist_ok=True)
        print(f"processing {len(tasks)} patch(es) with {num_workers} worker(s)")
        task_by_patch = {patch: (catalog_path, patch_output_root) for patch, catalog_path, patch_output_root in tasks}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_patch = {
                executor.submit(
                    worker_fn,
                    args,
                    coadd_root=coadd_root,
                    catalog_path=catalog_path,
                    output_root=patch_output_root,
                    bands=bands,
                    patch=patch,
                ): patch
                for patch, catalog_path, patch_output_root in tasks
            }
            completed = 0
            for future in as_completed(future_to_patch):
                patch = future_to_patch[future]
                try:
                    summaries_by_patch[patch] = future.result()
                except Exception as exc:
                    catalog_path, patch_output_root = task_by_patch[patch]
                    failed_patch_rows.append(_patch_failure_row(patch, catalog_path, patch_output_root, exc))
                    print(f"FAILED patch {patch}: {exc}", flush=True)
                completed += 1
                print(f"completed patch {patch} ({completed}/{len(tasks)})", flush=True)

    summaries = [summaries_by_patch[patch] for patch in patches if patch in summaries_by_patch]

    if not args.dry_run:
        data_root.mkdir(parents=True, exist_ok=True)
        (data_root / "preprocess_manifest.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        if failed_patch_rows:
            _write_failed_patches(data_root, failed_patch_rows)
        if args.fast_root is not None:
            fast_root = _expand(args.fast_root)
            fast_root.mkdir(parents=True, exist_ok=True)
            (fast_root / "preprocess_manifest.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
            if failed_patch_rows:
                _write_failed_patches(fast_root, failed_patch_rows)

    total_tiles = sum(int(item["num_tiles"]) for item in summaries)
    print(
        f"prepared {total_tiles} tiles across {len(summaries)}/{len(patches)} patch(es); "
        f"failed={len(failed_patch_rows)}; data_root={data_root}"
    )
    if failed_patch_rows:
        failed_names = " ".join(row["patch"] for row in failed_patch_rows)
        print(f"FAILED_PATCHES {failed_names}", flush=True)
        if not args.dry_run:
            print(
                f"failed patch details written to {data_root / 'preprocess_failed_patches.json'} "
                f"and {data_root / 'preprocess_failed_patches.csv'}",
                flush=True,
            )
        raise RuntimeError(f"{len(failed_patch_rows)} patch(es) failed during preprocessing")


def _preprocess_patch(
    args: argparse.Namespace,
    *,
    coadd_root: Path,
    catalog_path: Path,
    output_root: Path,
    bands: Tuple[str, ...],
    patch: str,
) -> Dict[str, object]:

    first_band = bands[0]
    first_path = _band_fits_path(coadd_root, first_band, args.tract, patch)
    with fits.open(first_path, memmap=True) as hdul:
        image_hdu = hdul[_find_image_hdu_index(hdul)]
        image_shape_yx = image_hdu.data.shape
        image_origin = _origin_from_ltv(image_hdu.header)
    width, height = int(image_shape_yx[1]), int(image_shape_yx[0])
    if (width, height) != (args.image_width, args.image_height):
        print(f"patch {patch}: detected image shape width={width}, height={height}; overriding CLI image size.", flush=True)

    parent_origin = tuple(args.parent_origin) if args.parent_origin is not None else image_origin
    if args.parent_origin is not None and image_origin != parent_origin:
        print(f"WARNING: FITS LTV origin {image_origin} differs from requested parent origin {parent_origin}.")

    compare_origin = tuple(args.compare_origin) if args.compare_origin else None
    if compare_origin is not None:
        cx, cy = compare_origin
        if (
            cx < parent_origin[0]
            or cy < parent_origin[1]
            or cx + args.tile_size > parent_origin[0] + width
            or cy + args.tile_size > parent_origin[1] + height
        ):
            compare_origin = None
    specs = make_tile_specs(
        parent_origin=parent_origin,
        image_shape=(width, height),
        tile_size=args.tile_size,
        stride=args.stride,
        compare_origin=compare_origin,
    )
    if args.max_tiles is not None:
        specs = specs[: int(args.max_tiles)]

    table = _read_table(catalog_path, hdu=args.catalog_hdu, role="primary", patch=patch)
    filtered, rejected = _filter_catalog(table, args)

    sources_dir = output_root / "sources"
    if not args.dry_run:
        write_table_pair(filtered, sources_dir / "sources_filtered.fits", sources_dir / "sources_filtered.csv")
        write_table_pair(rejected, sources_dir / "sources_rejected.fits", sources_dir / "sources_rejected.csv")

    band_catalog_warnings: List[Dict[str, str]] = []
    if not args.dry_run:
        band_catalog_root = _expand(args.band_catalog_root) if args.band_catalog_root is not None else _expand(args.catalog_root)
        for band in bands:
            band_catalog_path = _band_catalog_path(band_catalog_root, band, args.tract, patch)
            try:
                band_table = _read_table(band_catalog_path, hdu=args.catalog_hdu, role="band-reference", patch=patch, band=band)
                band_filtered, band_rejected = _filter_catalog(band_table, args)
            except Exception as exc:
                policy = str(args.bad_band_catalog_policy)
                warning = {
                    "band": band,
                    "path": str(band_catalog_path),
                    "policy": policy,
                    "error": str(exc),
                }
                band_catalog_warnings.append(warning)
                if policy == "error":
                    raise
                if policy == "skip":
                    print(
                        f"WARNING: skipping bad band catalog for patch={patch}, band={band}: {band_catalog_path}; {exc}",
                        flush=True,
                    )
                    continue
                if policy != "fallback-primary":
                    raise ValueError(f"Unknown bad_band_catalog_policy: {policy}")
                print(
                    f"WARNING: using primary catalog for patch={patch}, band={band} because band catalog is unreadable: "
                    f"{band_catalog_path}; {exc}",
                    flush=True,
                )
                band_filtered, band_rejected = filtered, rejected
            band_ref_dir = output_root / "band_reference_catalogs" / band
            write_table_pair(
                band_filtered,
                band_ref_dir / f"meas-{band}-{args.tract}-{patch}.fits",
                band_ref_dir / f"meas-{band}-{args.tract}-{patch}.csv",
            )
            write_table_pair(
                band_rejected,
                output_root / "band_reference_rejected" / band / f"meas-{band}-{args.tract}-{patch}.fits",
                None,
            )

    manifest_rows = []
    cutout_paths: Dict[str, Dict[str, str]] = {}

    if args.dry_run:
        print(f"dry-run: would write {len(specs)} tiles to {output_root}", flush=True)
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
            src = _band_fits_path(coadd_root, band, args.tract, patch)
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

        if args.zscale_root is not None and not args.dry_run:
            zscale_path = _zscale_cache_path(
                _expand(args.zscale_root),
                tract=args.tract,
                patch=patch,
                tile_name=spec.name,
                bands=bands,
                fits_hdu=args.zscale_fits_hdu,
            )
            write_zscale_cache(
                [band_paths[band] for band in bands],
                zscale_path,
                fits_hdu=args.zscale_fits_hdu,
                overwrite=args.overwrite_zscale,
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
        "patch": patch,
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
        "zscale_root": str(_expand(args.zscale_root)) if args.zscale_root is not None else None,
        "zscale_fits_hdu": args.zscale_fits_hdu,
        "fast_root": str(_expand(args.fast_root)) if args.fast_root is not None else None,
        "bad_band_catalog_policy": args.bad_band_catalog_policy,
        "band_catalog_warnings": band_catalog_warnings,
        "args": _jsonable_args(args),
    }

    if not args.dry_run:
        (output_root / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if manifest_rows:
            with (output_root / "tiles.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
                writer.writeheader()
                writer.writerows(manifest_rows)
        (output_root / "cutout_paths.json").write_text(json.dumps(cutout_paths, indent=2), encoding="utf-8")
        if args.fast_root is not None:
            mirror_fast_outputs(output_root, _patch_output_root(_expand(args.fast_root), args.tract, patch))

    print(
        f"patch {patch}: prepared {len(specs)} tiles; filtered sources={len(filtered)}, rejected={len(rejected)}; "
        f"output={output_root}",
        flush=True,
    )
    return metadata


def _sync_existing_patch(
    args: argparse.Namespace,
    *,
    coadd_root: Path,
    catalog_path: Path,
    output_root: Path,
    bands: Tuple[str, ...],
    patch: str,
) -> Dict[str, object]:
    del coadd_root, catalog_path
    if not output_root.exists():
        raise FileNotFoundError(f"Existing preprocessed patch root does not exist: {output_root}")

    ref_dir = output_root / "reference_catalogs"
    if not ref_dir.exists():
        raise FileNotFoundError(f"Existing reference_catalogs directory does not exist: {ref_dir}")

    manifest_path = output_root / "manifest.json"
    metadata: Dict[str, object] = {}
    if manifest_path.exists():
        try:
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    tile_names = sorted(path.name[: -len("_meas.fits")] for path in ref_dir.glob("*_meas.fits"))
    if not tile_names:
        raise RuntimeError(f"No existing tile reference catalogs found in {ref_dir}")

    cutout_paths: Dict[str, Dict[str, str]] = {}
    missing: List[str] = []
    for tile_name in tile_names:
        tile_dir = output_root / "cutouts" / tile_name
        band_paths: Dict[str, str] = {}
        for band in bands:
            try:
                band_paths[band] = _existing_cutout_fits_path(tile_dir, band)
            except Exception as exc:
                missing.append(f"{tile_name}/{band}: {exc}")
        if len(band_paths) == len(bands):
            cutout_paths[tile_name] = band_paths

    if missing:
        examples = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"Existing preprocessed patch {patch} is missing {len(missing)} requested band cutout(s). "
            f"First examples:\n{examples}"
        )

    zscale_written = 0
    zscale_skipped = 0
    if args.zscale_root is not None and not args.dry_run:
        zscale_root = _expand(args.zscale_root)
        for tile_name, band_paths in cutout_paths.items():
            zscale_path = _zscale_cache_path(
                zscale_root,
                tract=args.tract,
                patch=patch,
                tile_name=tile_name,
                bands=bands,
                fits_hdu=args.zscale_fits_hdu,
            )
            existed = zscale_path.exists()
            write_zscale_cache(
                [band_paths[band] for band in bands],
                zscale_path,
                fits_hdu=args.zscale_fits_hdu,
                overwrite=args.overwrite_zscale,
            )
            if existed and not args.overwrite_zscale:
                zscale_skipped += 1
            else:
                zscale_written += 1

    metadata.update(
        {
            "output_root": str(output_root),
            "bands": bands,
            "tract": args.tract,
            "patch": patch,
            "num_tiles": len(tile_names),
            "reuse_existing_preprocessed": True,
            "zscale_root": str(_expand(args.zscale_root)) if args.zscale_root is not None else None,
            "zscale_fits_hdu": args.zscale_fits_hdu,
            "zscale_written": zscale_written,
            "zscale_skipped": zscale_skipped,
            "fast_root": str(_expand(args.fast_root)) if args.fast_root is not None else None,
            "args": _jsonable_args(args),
        }
    )

    if not args.dry_run:
        (output_root / "cutout_paths.json").write_text(json.dumps(cutout_paths, indent=2), encoding="utf-8")
        manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if args.fast_root is not None:
            mirror_fast_outputs(output_root, _patch_output_root(_expand(args.fast_root), args.tract, patch))

    print(
        f"patch {patch}: reused {len(tile_names)} existing tiles; zscale_written={zscale_written}, "
        f"zscale_skipped={zscale_skipped}; output={output_root}",
        flush=True,
    )
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess HSC denoised or official coadds for AstroCELLECT training.")
    parser.add_argument("--coadd-root", type=Path, default=DEFAULT_COADD_ROOT)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="Single meas catalog FITS. Only valid when preprocessing one patch; otherwise use --catalog-root.",
    )
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=DEFAULT_BAND_CATALOG_ROOT,
        help="Root containing per-band meas catalogs for the primary training catalog.",
    )
    parser.add_argument(
        "--catalog-band",
        default=DEFAULT_CATALOG_BAND,
        help="Band used for the primary training catalog when --catalog is not provided.",
    )
    parser.add_argument(
        "--band-catalog-root",
        type=Path,
        default=None,
        help="Optional root containing per-band meas catalogs. Defaults to --catalog-root.",
    )
    parser.add_argument("--catalog-hdu", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bands", nargs="+", default=list(BANDS))
    parser.add_argument("--tract", type=int, default=9813)
    parser.add_argument("--patch", default="4,5")
    parser.add_argument(
        "--patches",
        nargs="*",
        default=None,
        help="Patch IDs to preprocess, e.g. 4,5 4,6, or 'all'. Semicolon-separated values are also accepted.",
    )
    parser.add_argument("--patch-file", type=Path, default=None, help="Optional text file with one patch ID per line.")
    parser.add_argument(
        "--parent-origin",
        nargs=2,
        type=int,
        default=None,
        metavar=("X0", "Y0"),
        help="Override parent-patch origin. By default each patch uses the FITS LTV1/LTV2 origin.",
    )
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
    parser.add_argument(
        "--reuse-existing-preprocessed",
        action="store_true",
        help=(
            "Reuse an existing <output-root>/<tract>/<patch> tree instead of recropping/rebuilding catalogs. "
            "This refreshes cutout_paths.json/manifest, optionally generates zscale for --bands, and mirrors "
            "metadata to --fast-root. Use this when adding bands after a previous preprocessing run."
        ),
    )
    parser.add_argument(
        "--zscale-root",
        type=Path,
        default=None,
        help=(
            "Optional root for precomputed zscale CHW tensors. Files are written as "
            "<zscale-root>/<tract>/<patch>/cutouts/<tile>__<bands>__hdu<N>.pt."
        ),
    )
    parser.add_argument("--zscale-fits-hdu", type=int, default=1, help="FITS HDU used when generating zscale cache.")
    parser.add_argument(
        "--overwrite-zscale",
        action="store_true",
        help="Regenerate zscale cache files that already exist. --overwrite only controls FITS cutouts.",
    )
    parser.add_argument(
        "--fast-root",
        type=Path,
        default=None,
        help=(
            "Optional SSD root that receives training metadata only: targets, reference catalogs, "
            "band reference catalogs, sources, manifest, tiles.csv, and cutout_paths.json. Cutout FITS are not copied."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--bad-band-catalog-policy",
        choices=("fallback-primary", "skip", "error"),
        default="fallback-primary",
        help=(
            "Policy when a per-band meas catalog is unreadable. fallback-primary writes the primary catalog "
            "for that band, skip omits that band reference, error aborts the patch."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of patch-level worker processes. Use 0 to auto-select up to the number of patches/CPU cores.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.no_compare_origin:
        args.compare_origin = None
    preprocess(args)


if __name__ == "__main__":
    main()
