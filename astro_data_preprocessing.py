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
import ast
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import traceback
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.units import UnitsWarning

try:
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover - tqdm is optional for preprocessing.
    _tqdm = None

from astro_pu_source_filter import (
    DEFAULT_A_FLAGS as DEFAULT_PU_A_FLAGS,
    DEFAULT_B_FLAGS as DEFAULT_PU_B_FLAGS,
    attach_kron_refit_radius,
    classify_sources as classify_pu_sources,
)


BANDS = ("HSC-G", "HSC-R", "HSC-I")
PIXEL_PLANES = ("IMAGE", "MASK", "VARIANCE")
DEFAULT_COADD_ROOT = Path("/home/chenzunhao/segment-anything/lsst_pipeline/fits/projection_cutout")
DEFAULT_CATALOG = Path("/home/chenzunhao/2026-05-01_scarlet_deblend_demo/catalog/HSC-I/meas-HSC-I-9813-4,5.fits")
DEFAULT_BAND_CATALOG_ROOT = Path("/home/chenzunhao/2026-05-01_scarlet_deblend_demo/catalog")
DEFAULT_OUTPUT_ROOT = Path("./output/hsc_astro_preprocessed")
DEFAULT_PARENT_ORIGIN = (15900, 19900)
DEFAULT_COMPARE_ORIGIN = (18204, 20924)
DEFAULT_CATALOG_BAND = "HSC-I"
_THREADPOOL_LIMITER = None


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


def _progress_iter(
    iterable,
    *,
    total: int,
    desc: str,
    unit: str,
    enabled: bool = True,
):
    if not enabled or _tqdm is None:
        return iterable
    return _tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True, leave=True)


def _configure_worker_threads(num_threads: int) -> None:
    global _THREADPOOL_LIMITER
    threads = max(1, int(num_threads))
    for name in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        os.environ[name] = str(threads)
    try:
        import torch

        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        pass
    try:
        from threadpoolctl import threadpool_limits

        if _THREADPOOL_LIMITER is not None:
            try:
                _THREADPOOL_LIMITER.__exit__(None, None, None)
            except Exception:
                pass
        _THREADPOOL_LIMITER = threadpool_limits(limits=threads)
        _THREADPOOL_LIMITER.__enter__()
    except Exception:
        pass
    try:
        import cv2

        cv2.setNumThreads(threads)
    except Exception:
        pass


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


def _band_det_path(root: Path, band: str, tract: int, patch: str) -> Optional[Path]:
    filename = f"det-{band}-{tract}-{patch}.fits"
    candidates = [
        root / band / filename,
        root / band / patch / filename,
        root / str(tract) / band / patch / filename,
        root / str(tract) / band / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _lsst_background_cache_root(args: argparse.Namespace, output_root: Path) -> Path:
    root = getattr(args, "lsst_background_cache_root", None)
    if root is not None:
        return _expand(root)
    return output_root / "lsst_detection_background"


def _cached_lsst_det_path(args: argparse.Namespace, output_root: Path, band: str, tract: int, patch: str) -> Path:
    return (
        _lsst_background_cache_root(args, output_root)
        / str(tract)
        / band
        / patch
        / f"det-{band}-{tract}-{patch}.fits"
    )


def _cached_lsst_detected_calexp_path(args: argparse.Namespace, output_root: Path, band: str, tract: int, patch: str) -> Path:
    return (
        _lsst_background_cache_root(args, output_root)
        / str(tract)
        / band
        / patch
        / f"calexp-detect-{band}-{tract}-{patch}.fits"
    )


def _run_lsst_detection_background(
    *,
    args: argparse.Namespace,
    output_root: Path,
    coadd_path: Path,
    band: str,
    tract: int,
    patch: str,
    output_calexp_path: Optional[Path] = None,
) -> Path:
    det_path = _cached_lsst_det_path(args, output_root, band, tract, patch)
    if det_path.exists() and (output_calexp_path is None or output_calexp_path.exists()) and not bool(
        getattr(args, "overwrite_lsst_background", False)
    ):
        return det_path
    helper = Path(__file__).resolve().with_name("lsst_detect_background.py")
    python_cmd = str(getattr(args, "lsst_detect_python", "") or sys.executable)
    cmd = [
        *shlex.split(python_cmd),
        str(helper),
        "--input",
        str(coadd_path),
        "--output-det",
        str(det_path),
    ]
    if output_calexp_path is not None or bool(getattr(args, "write_lsst_background_products", False)):
        product_root = det_path.parent
        output_calexp_path = output_calexp_path or product_root / f"calexp-detect-{band}-{tract}-{patch}.fits"
        cmd.extend(
            [
                "--output-calexp",
                str(output_calexp_path),
                "--output-background",
                str(product_root / f"det_bkgd-{band}-{tract}-{patch}.fits"),
            ]
        )
    print(
        "running LSST default detection for PU background: "
        f"patch={patch}, band={band}, input={coadd_path}, output={det_path}",
        flush=True,
    )
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"failed to run LSST detection command {cmd[0]!r}; set --lsst-detect-python to a Python executable "
            "with lsst_distrib loaded"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "LSST detection fallback failed for PU background. "
            "Run preprocessing in an LSST stack environment or pass --lsst-background-policy none/existing. "
            f"Command: {' '.join(shlex.quote(part) for part in cmd)}"
        ) from exc
    if not det_path.exists():
        raise RuntimeError(f"LSST detection finished but did not write expected det catalog: {det_path}")
    return det_path


def _resolve_lsst_background_det_path(
    *,
    args: argparse.Namespace,
    output_root: Path,
    coadd_root: Path,
    band: str,
    tract: int,
    patch: str,
) -> Optional[Path]:
    existing = _band_det_path(coadd_root, band, tract, patch)
    if existing is not None:
        return existing
    policy = str(getattr(args, "lsst_background_policy", "run-if-missing"))
    if policy in {"none", "existing"}:
        return None
    if policy != "run-if-missing":
        raise ValueError(f"Unknown lsst_background_policy: {policy}")
    coadd_path = _band_fits_path(coadd_root, band, tract, patch)
    if not coadd_path.exists():
        raise FileNotFoundError(f"cannot run LSST background fallback; coadd FITS not found: {coadd_path}")
    return _run_lsst_detection_background(
        args=args,
        output_root=output_root,
        coadd_path=coadd_path,
        band=band,
        tract=tract,
        patch=patch,
    )


def _read_det_background_mask(path: Path, shape_yx: Tuple[int, int], origin_xy: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """Return True for pixels outside LSST detection footprints."""
    with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
        if len(hdul) <= 4:
            return ~np.zeros(shape_yx, dtype=bool)
        spans = hdul[4].data
        if spans is None:
            return ~np.zeros(shape_yx, dtype=bool)
        rows = [(int(row["y"]), int(row["x0"]), int(row["x1"])) for row in spans]

    def _paint(*, subtract_origin: bool) -> tuple[np.ndarray, int]:
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

    footprint, painted = _paint(subtract_origin=False)
    if painted == 0 and origin_xy != (0, 0):
        footprint, _painted = _paint(subtract_origin=True)
    return ~footprint


def _crop_full_mask_for_tile(mask: Optional[np.ndarray], spec: TileSpec, parent_origin: Tuple[int, int]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    x0 = int(spec.x0 - parent_origin[0])
    y0 = int(spec.y0 - parent_origin[1])
    x1 = x0 + int(spec.size)
    y1 = y0 + int(spec.size)
    out = np.zeros((spec.size, spec.size), dtype=bool)
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(mask.shape[1], x1)
    src_y1 = min(mask.shape[0], y1)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out
    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    out[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = mask[src_y0:src_y1, src_x0:src_x1]
    return out


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


def _variant_patch_output_root(output_root: Path, variant: str, tract: int, patch: str) -> Path:
    return output_root / variant / str(tract) / patch


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
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*Unit 'second' not supported by the FITS standard.*",
                category=UnitsWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=".*'second' did not parse as fits unit.*",
                category=UnitsWarning,
            )
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


_DEFAULT_PU_BAND_LIMIT_MAGS = {
    "HSC-G": 27.4,
    "HSC-R": 27.1,
    "HSC-I": 26.9,
    "HSC-Z": 26.3,
    "HSC-Y": 25.3,
}

_DEFAULT_PU_STRICT_CENTER_ONLY_SATURATION_MAGS = {
    "HSC-G": 18.0,
    "HSC-R": 18.2,
    "HSC-I": 18.6,
    "HSC-Z": 17.7,
    "HSC-Y": 17.4,
}


@dataclass(frozen=True)
class _ArchiveLookup:
    row0: np.ndarray
    nrows: np.ndarray
    found: np.ndarray


@dataclass(frozen=True)
class _ArchiveIndex:
    ids: np.ndarray
    archive_numbers: np.ndarray
    names: np.ndarray
    row0: np.ndarray
    nrows: np.ndarray

    @classmethod
    def from_archive(cls, archive: fits.FITS_rec) -> "_ArchiveIndex":
        return cls(
            ids=np.asarray(archive["id"], dtype=np.int64),
            archive_numbers=np.asarray(archive["cat.archive"], dtype=np.int64),
            names=np.asarray([_decode_fits_string(value) for value in archive["name"]], dtype=object),
            row0=np.asarray(archive["row0"], dtype=np.int64),
            nrows=np.asarray(archive["nrows"], dtype=np.int64),
        )

    def lookup(self, target_ids: np.ndarray, *, archive_number: int, name: Optional[str]) -> _ArchiveLookup:
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
        return _ArchiveLookup(row0=row0, nrows=nrows, found=found)


def _decode_fits_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value).strip()


def _normalize_band_name(name: str) -> str:
    text = str(name).strip().upper()
    if not text:
        return text
    if not text.startswith("HSC-"):
        text = f"HSC-{text}"
    return text


def _parse_band_mags(values: Optional[Sequence[str]], defaults: Dict[str, float], *, label: str) -> Dict[str, float]:
    limits = dict(defaults)
    if not values:
        return limits
    for raw in values:
        for item in str(raw).replace(",", " ").split():
            if not item:
                continue
            if "=" not in item and ":" not in item:
                raise ValueError(f"{label} must be BAND=mag, got {item!r}")
            key, value = item.replace(":", "=", 1).split("=", 1)
            limits[_normalize_band_name(key)] = float(value)
    return limits


def _parse_band_limit_mags(values: Optional[Sequence[str]]) -> Dict[str, float]:
    return _parse_band_mags(values, _DEFAULT_PU_BAND_LIMIT_MAGS, label="band limit")


def _parse_strict_center_only_saturation_mags(values: Optional[Sequence[str]]) -> Dict[str, float]:
    return _parse_band_mags(
        values,
        _DEFAULT_PU_STRICT_CENTER_ONLY_SATURATION_MAGS,
        label="strict center-only saturation magnitude",
    )


def _strict_center_only_mag_threshold(args: argparse.Namespace, *, band: Optional[str]) -> float:
    override = getattr(args, "pu_strict_bright_center_only_mag_threshold", None)
    if override is None:
        override = getattr(args, "pu_strict_ignore_mag_threshold", None)
    if override is not None:
        return float(override)
    band_name = _normalize_band_name(band or getattr(args, "catalog_band", ""))
    limits = _parse_strict_center_only_saturation_mags(
        getattr(args, "pu_strict_bright_center_only_saturation_mags", None)
        or getattr(args, "pu_strict_ignore_saturation_mags", None)
    )
    if band_name not in limits:
        raise ValueError(f"No strict center-only saturation magnitude configured for {band_name!r}")
    return float(limits[band_name])


def _pu_args(args: argparse.Namespace, *, band: Optional[str] = None) -> argparse.Namespace:
    b_mag_min = float(args.pu_b_mag_min)
    b_mag_max = float(args.pu_b_mag_max)
    if bool(getattr(args, "pu_use_band_limit_b_filter", False)):
        band_name = _normalize_band_name(band or getattr(args, "catalog_band", ""))
        limits = _parse_band_limit_mags(getattr(args, "pu_band_limit_mags", None))
        if band_name not in limits:
            raise ValueError(f"No PU band limiting magnitude configured for {band_name!r}")
        limit = float(limits[band_name])
        b_mag_min = limit + float(args.pu_band_limit_b_min_offset)
        b_mag_max = limit + float(args.pu_band_limit_b_max_offset)
    return argparse.Namespace(
        source_filter=args.source_filter,
        a_flags=tuple(args.pu_a_flags),
        b_flags=tuple(args.pu_b_flags),
        a_mode=args.pu_a_mode,
        b_mode=args.pu_b_mode,
        strict_flags=args.pu_strict_flags,
        region_sigma=args.ellipse_sigma,
        min_axis=args.min_ellipse_axis,
        mag_column=args.pu_mag_column,
        input_zeropoint=args.pu_input_zeropoint,
        require_kron_refit_match=bool(args.pu_require_kron_refit_match),
        a_area_max=args.pu_a_area_max,
        a_faint_area_max=args.pu_a_faint_area_max,
        a_faint_mag_min=args.pu_a_faint_mag_min,
        b_mag_min=b_mag_min,
        b_mag_max=b_mag_max,
        ap2_kron_abs_max=args.pu_ap2_kron_abs_max,
        ap2_flux_column=args.pu_ap2_flux_column,
        ap2_kron_flux_column=args.pu_ap2_kron_flux_column,
        pixel_scale_arcsec=args.pixel_scale_arcsec,
        b_close_center_arcsec=args.pu_b_close_center_arcsec,
        overlap_iou_threshold=args.pu_overlap_iou_threshold,
        b_ellipse_area_max=args.pu_b_ellipse_area_max,
        b_footprint_area_max=args.pu_b_footprint_area_max,
        b_axis_ratio_max=args.pu_b_axis_ratio_max,
        b_kron_radius_lt_sdss_major_ratio=args.pu_b_kron_radius_lt_sdss_major_ratio,
        drop_ellipse_area_min=args.pu_drop_ellipse_area_min,
        ambiguous_area_max=args.pu_ambiguous_area_max,
        neighbor_radius=args.pu_neighbor_radius,
        center_distance_factor=args.pu_center_distance_factor,
        containment_threshold=args.pu_containment_threshold,
        mutual_overlap_threshold=args.pu_mutual_overlap_threshold,
        overlap_sample_grid=args.pu_overlap_sample_grid,
        ambiguous_mark=args.pu_ambiguous_mark,
        keep_all_ab_clean=bool(getattr(args, "pu_keep_all_ab_clean", False)),
    )


def _resolve_pu_kron_refit_csv(args: argparse.Namespace, *, band: Optional[str], patch: Optional[str]) -> Optional[Path]:
    raw = getattr(args, "pu_kron_refit_csv", None)
    if raw is None:
        return None
    text = str(raw)
    if any(token in text for token in ("{band}", "{tract}", "{patch}")):
        text = text.format(
            band=band or getattr(args, "band", "") or getattr(args, "catalog_band", ""),
            tract=getattr(args, "tract", ""),
            patch=patch or getattr(args, "patch", ""),
        )
    path = Path(text)
    if path.exists():
        return path
    if path.name == "kron_refit_rows.csv":
        current_name_path = path.with_name("batch_heavyfp_kron_refit.csv")
        if current_name_path.exists():
            return current_name_path
    return path


def _attach_pu_kron_refit(
    table: Table,
    args: argparse.Namespace,
    *,
    band: Optional[str] = None,
    patch: Optional[str] = None,
    radius_column: Optional[str] = None,
    output_column: str = "pu_refit_kron_radius",
) -> Table:
    path = _resolve_pu_kron_refit_csv(args, band=band, patch=patch)
    if path is None:
        return table
    return attach_kron_refit_radius(
        table,
        path,
        radius_column=radius_column or getattr(args, "pu_kron_refit_radius_column", "proxy_nan0_flux_aperture_radius"),
        good_column=getattr(args, "pu_kron_refit_good_column", "proxy_nan0_good"),
        output_column=output_column,
    )


def _magnitude_from_catalog(table: Table, *, mag_column: str, zeropoint: float) -> np.ndarray:
    flux = _first_finite_column(
        table,
        (
            mag_column,
            "ext_photometryKron_KronFlux_instFlux",
            "base_PsfFlux_instFlux",
            "modelfit_CModel_instFlux",
            "base_SdssShape_instFlux",
        ),
    )
    mag = np.full(len(table), np.nan, dtype=np.float32)
    if flux is None:
        return mag
    valid = np.isfinite(flux) & (flux > 0.0)
    mag[valid] = float(zeropoint) - 2.5 * np.log10(np.asarray(flux[valid], dtype=np.float64))
    return mag


def _magnitude_from_flux(flux: np.ndarray | float, *, zeropoint: float) -> np.ndarray | float:
    arr = np.asarray(flux, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(arr) & (arr > 0.0)
    out[valid] = float(zeropoint) - 2.5 * np.log10(arr[valid])
    if np.isscalar(flux):
        return float(out)
    return out


def _table_float_column(table: Table, column: str, default: float = np.nan) -> np.ndarray:
    if column not in table.colnames:
        return np.full(len(table), default, dtype=np.float64)
    return np.asarray(table[column], dtype=np.float64)


def _table_bool_column(table: Table, column: str) -> np.ndarray:
    if column not in table.colnames:
        return np.zeros(len(table), dtype=bool)
    return np.asarray(table[column], dtype=bool)


def _ellipse_aperture_sum(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    cx: float,
    cy: float,
    major: float,
    minor: float,
    theta: float,
) -> Tuple[float, int, int]:
    if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(major) and np.isfinite(minor)):
        return np.nan, 0, 0
    if major <= 0.0 or minor <= 0.0:
        return np.nan, 0, 0
    dx = np.asarray(x, dtype=np.float64) - float(cx)
    dy = np.asarray(y, dtype=np.float64) - float(cy)
    c = math.cos(float(theta))
    s = math.sin(float(theta))
    du = dx * c + dy * s
    dv = -dx * s + dy * c
    inside = (du / float(major)) ** 2 + (dv / float(minor)) ** 2 <= 1.0
    if not np.any(inside):
        return 0.0, 0, int(np.asarray(values).size)
    selected = np.asarray(values, dtype=np.float64)[inside]
    finite = np.isfinite(selected)
    return float(np.sum(np.where(finite, selected, 0.0))), int(np.count_nonzero(inside)), int(np.asarray(values).size)


def _spanset_pixels(spans_table: fits.FITS_rec, span0: int, nspan: int) -> Tuple[np.ndarray, np.ndarray]:
    if span0 < 0 or nspan <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    spans = spans_table[span0 : span0 + nspan]
    widths = np.asarray(spans["x1"], dtype=np.int64) - np.asarray(spans["x0"], dtype=np.int64) + 1
    total_pixels = int(np.sum(widths))
    if total_pixels <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    span_index = np.repeat(np.arange(nspan, dtype=np.int64), widths)
    starts = np.repeat(np.cumsum(widths) - widths, widths)
    x = np.asarray(spans["x0"], dtype=np.int64)[span_index] + (np.arange(total_pixels) - starts)
    y = np.asarray(spans["y"], dtype=np.int64)[span_index]
    return x, y


def _measure_heavy_ellipse(
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
) -> Tuple[float, int, int]:
    x, y = _spanset_pixels(spans_table, span0, nspan)
    if x.size == 0 or heavy_row < 0:
        return np.nan, 0, int(x.size)
    values = np.asarray(heavy_table["image"][heavy_row], dtype=np.float64)
    if values.size != x.size:
        return np.nan, 0, int(x.size)
    return _ellipse_aperture_sum(x, y, values, cx=cx, cy=cy, major=major, minor=minor, theta=theta)


def _measure_direct_ellipse(
    *,
    image: np.ndarray,
    image_origin: Tuple[int, int],
    cx: float,
    cy: float,
    major: float,
    minor: float,
    theta: float,
) -> Tuple[float, int, int]:
    if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(major) and np.isfinite(minor)):
        return np.nan, 0, 0
    if major <= 0.0 or minor <= 0.0:
        return np.nan, 0, 0
    local_cx = float(cx) - float(image_origin[0])
    local_cy = float(cy) - float(image_origin[1])
    r = int(math.ceil(max(float(major), float(minor)) + 2.0))
    x0 = max(0, int(math.floor(local_cx)) - r)
    x1 = min(image.shape[1] - 1, int(math.floor(local_cx)) + r)
    y0 = max(0, int(math.floor(local_cy)) - r)
    y1 = min(image.shape[0] - 1, int(math.floor(local_cy)) + r)
    if x1 < x0 or y1 < y0:
        return np.nan, 0, 0
    yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    global_x = xx.astype(np.float64) + float(image_origin[0])
    global_y = yy.astype(np.float64) + float(image_origin[1])
    return _ellipse_aperture_sum(
        global_x.ravel(),
        global_y.ravel(),
        np.asarray(image[y0 : y1 + 1, x0 : x1 + 1], dtype=np.float64).ravel(),
        cx=cx,
        cy=cy,
        major=major,
        minor=minor,
        theta=theta,
    )


def _footprint_ellipse_counts(
    *,
    spans_table: fits.FITS_rec,
    span0: int,
    nspan: int,
    cx: float,
    cy: float,
    major: float,
    minor: float,
    theta: float,
) -> Tuple[int, int]:
    x, y = _spanset_pixels(spans_table, span0, nspan)
    if x.size == 0:
        return 0, 0
    _, aperture_pixels, total_pixels = _ellipse_aperture_sum(
        x,
        y,
        np.ones(x.size, dtype=np.float64),
        cx=cx,
        cy=cy,
        major=major,
        minor=minor,
        theta=theta,
    )
    return int(aperture_pixels), int(total_pixels)


def _unit_disk_points(n: int = 15) -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, int(n), dtype=np.float64)
    uu, vv = np.meshgrid(grid, grid)
    pts = np.column_stack([uu.ravel(), vv.ravel()])
    return pts[np.sum(pts * pts, axis=1) <= 1.0]


def _ellipse_contains_fraction(small: Dict[str, float], big: Dict[str, float], unit_points: np.ndarray) -> float:
    if min(small["a"], small["b"], big["a"], big["b"]) <= 0.0:
        return 0.0
    cs = math.cos(small["theta"])
    ss = math.sin(small["theta"])
    u = unit_points[:, 0] * small["a"]
    v = unit_points[:, 1] * small["b"]
    px = small["x"] + u * cs - v * ss
    py = small["y"] + u * ss + v * cs
    cb = math.cos(big["theta"])
    sb = math.sin(big["theta"])
    dx = px - big["x"]
    dy = py - big["y"]
    du = dx * cb + dy * sb
    dv = -dx * sb + dy * cb
    inside = (du / big["a"]) ** 2 + (dv / big["b"]) ** 2 <= 1.0
    return float(np.mean(inside))


def _strict_bright_center_only_catalog(
    table: Table,
    args: argparse.Namespace,
    *,
    band: Optional[str] = None,
    patch: Optional[str] = None,
) -> Table:
    if not bool(getattr(args, "pu_enable_strict_bright_center_only", False)):
        return table[:0].copy(copy_data=True)
    table = _attach_pu_kron_refit(
        table,
        args,
        band=band,
        patch=patch,
        radius_column=getattr(args, "pu_strict_bright_center_only_radius_column", "proxy_nan0_flux_aperture_radius"),
        output_column="pu_refit_kron_radius",
    )
    shaped = add_ellipse_columns(table, shape_source=args.target_shape_source)
    mag = _magnitude_from_catalog(
        shaped,
        mag_column=getattr(args, "pu_mag_column", "ext_photometryKron_KronFlux_instFlux"),
        zeropoint=float(getattr(args, "pu_input_zeropoint", 27.0)),
    )
    x, y = _require_position_columns(shaped, args.x_col, args.y_col)
    major = np.asarray(shaped["ellipse_major_sigma"], dtype=np.float32)
    minor = np.asarray(shaped["ellipse_minor_sigma"], dtype=np.float32)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(mag) & np.isfinite(major) & np.isfinite(minor)
    valid &= (major > 0.0) & (minor > 0.0)
    valid &= source_selection_mask(shaped, getattr(args, "source_filter", "nchild0"))
    if bool(getattr(args, "pu_require_kron_refit_match", False)) and "pu_refit_kron_radius_matched" in shaped.colnames:
        valid &= np.asarray(shaped["pu_refit_kron_radius_matched"], dtype=bool)
    threshold = _strict_center_only_mag_threshold(args, band=band)
    valid &= mag < threshold
    out = shaped[valid]
    if len(out):
        out["pu_class"] = np.asarray(["strict_center_only"] * len(out), dtype=str)
        out["pu_reason"] = np.asarray(["strict_bright_center_only"] * len(out), dtype=str)
        out["pu_mag"] = mag[valid].astype(np.float32)
        out["pu_strict_center_only_mag_threshold"] = np.full(len(out), float(threshold), dtype=np.float32)
        out["pu_strict_center_only_radius_column"] = np.asarray(
            [str(getattr(args, "pu_strict_bright_center_only_radius_column", "proxy_nan0_flux_aperture_radius"))] * len(out),
            dtype=str,
        )
    return out


def _move_bright_clean_to_center_only(
    clean: Table,
    center_only: Table,
    args: argparse.Namespace,
    *,
    band: Optional[str] = None,
) -> Tuple[Table, Table, Table]:
    if not bool(getattr(args, "pu_enable_strict_bright_center_only", False)) or len(clean) == 0:
        return clean, center_only, clean[:0].copy(copy_data=True)
    threshold = _strict_center_only_mag_threshold(args, band=band)
    mag = _magnitude_from_catalog(
        clean,
        mag_column=getattr(args, "pu_mag_column", "ext_photometryKron_KronFlux_instFlux"),
        zeropoint=float(getattr(args, "pu_input_zeropoint", 27.0)),
    )
    bright_mask = np.isfinite(mag) & (mag < threshold)
    if not np.any(bright_mask):
        return clean, center_only, clean[:0].copy(copy_data=True)

    remaining_clean = clean[~bright_mask]
    bright_center = clean[bright_mask].copy(copy_data=True)
    bright_center["pu_class"] = np.asarray(["strict_center_only"] * len(bright_center), dtype=str)
    if "pu_reason" in bright_center.colnames:
        reasons = np.asarray(bright_center["pu_reason"], dtype=str)
        reasons = np.asarray(
            [
                "strict_bright_center_only" if not item or item == "--" else f"{item};strict_bright_center_only"
                for item in reasons
            ],
            dtype=str,
        )
    else:
        reasons = np.asarray(["strict_bright_center_only"] * len(bright_center), dtype=str)
    bright_center["pu_reason"] = reasons
    bright_center["pu_mag"] = mag[bright_mask].astype(np.float32)
    bright_center["pu_strict_center_only_mag_threshold"] = np.full(
        len(bright_center),
        float(threshold),
        dtype=np.float32,
    )
    if len(center_only):
        merged_center = vstack([center_only, bright_center], metadata_conflicts="silent")
    else:
        merged_center = bright_center
    return remaining_clean, merged_center, bright_center


def _source_geometry_from_shaped(shaped: Table, row: int) -> Optional[Dict[str, float]]:
    required = ("ellipse_major_sigma", "ellipse_minor_sigma", "ellipse_theta")
    if any(column not in shaped.colnames for column in required):
        return None
    x = _table_float_column(shaped, "base_SdssCentroid_x")
    y = _table_float_column(shaped, "base_SdssCentroid_y")
    major = _table_float_column(shaped, "ellipse_major_sigma")
    minor = _table_float_column(shaped, "ellipse_minor_sigma")
    theta = _table_float_column(shaped, "ellipse_theta", default=0.0)
    cx = float(x[row])
    cy = float(y[row])
    a = float(major[row])
    b = float(minor[row])
    t = float(theta[row])
    if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(a) and np.isfinite(b)):
        return None
    if a <= 0.0 or b <= 0.0:
        return None
    sid = int(shaped["id"][row]) if "id" in shaped.colnames else int(row)
    return {"source_id": sid, "x": cx, "y": cy, "a": a, "b": b, "theta": t, "area": math.pi * a * b, "rmax": max(a, b)}


def _remeasure_ap2_kron_outliers(
    shaped: Table,
    result: Dict[str, object],
    args: argparse.Namespace,
    *,
    band: Optional[str],
    patch: Optional[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, int]]:
    """Remeasure old ap2-Kron outliers with refit apertures and update PU classes.

    This ports the diagnostic logic from diagnostics/debug_refit_kron_photometry_4_5.py
    into preprocessing.  It only acts on leaf sources whose catalog ap2-vs-Kron
    difference failed the original B filter, then promotes reliable remeasured
    candidates out of ordinary ignore into clean or center_only.
    """

    n = len(shaped)
    empty_class = np.asarray([""] * n, dtype=object)
    if n == 0 or not bool(getattr(args, "pu_remeasure_ap2_kron_outliers", False)):
        return empty_class, {}, {"selected": 0}
    if band is None or patch is None:
        return empty_class, {}, {"selected": 0, "skipped_missing_band_patch": 1}
    if "id" not in shaped.colnames:
        return empty_class, {}, {"selected": 0, "skipped_missing_id": 1}

    ap2_flux = _table_float_column(shaped, getattr(args, "pu_ap2_flux_column", "base_CircularApertureFlux_6_0_instFlux"))
    kron_flux = _table_float_column(
        shaped,
        getattr(args, "pu_ap2_kron_flux_column", "ext_photometryKron_KronFlux_instFlux"),
    )
    zeropoint = float(getattr(args, "pu_input_zeropoint", 27.0))
    ap2_mag = np.asarray(_magnitude_from_flux(ap2_flux, zeropoint=zeropoint), dtype=np.float64)
    kron_mag = np.asarray(_magnitude_from_flux(kron_flux, zeropoint=zeropoint), dtype=np.float64)
    old_diff = np.abs(ap2_mag - kron_mag)
    threshold = float(getattr(args, "pu_remeasure_ap2_kron_threshold", np.nan))
    if not np.isfinite(threshold):
        threshold = float(getattr(args, "pu_ap2_kron_abs_max", 1.0))
    if threshold < 0.0:
        return empty_class, {}, {"selected": 0, "skipped_disabled_threshold": 1}

    nchild = _table_float_column(shaped, "deblend_nChild", default=1.0)
    x = _table_float_column(shaped, args.x_col)
    y = _table_float_column(shaped, args.y_col)
    finite_center = np.isfinite(x) & np.isfinite(y)
    leaf = nchild == 0
    old_ignore = np.asarray(result.get("ignore", result.get("b_class", np.zeros(n, dtype=bool))), dtype=bool)
    selected = np.flatnonzero(leaf & finite_center & np.isfinite(old_diff) & (old_diff > threshold) & old_ignore)
    class_override = np.asarray([""] * n, dtype=object)
    stats: Dict[str, int] = {"selected": int(selected.size)}

    diagnostics: Dict[str, np.ndarray] = {
        "pu_remeasure_ap2_mag": np.full(n, np.nan, dtype=np.float32),
        "pu_remeasure_catalog_kron_mag": np.full(n, np.nan, dtype=np.float32),
        "pu_remeasure_old_absdiff": np.full(n, np.nan, dtype=np.float32),
        "pu_remeasure_kron_flux": np.full(n, np.nan, dtype=np.float32),
        "pu_remeasure_kron_mag": np.full(n, np.nan, dtype=np.float32),
        "pu_remeasure_absdiff": np.full(n, np.nan, dtype=np.float32),
        "pu_remeasure_aperture_area": np.full(n, np.nan, dtype=np.float32),
        "pu_remeasure_footprint_fill_fraction": np.full(n, np.nan, dtype=np.float32),
        "pu_remeasure_axis_ratio": np.full(n, np.nan, dtype=np.float32),
        "pu_remeasure_surface": np.asarray([""] * n, dtype=object),
        "pu_remeasure_reason": np.asarray([""] * n, dtype=object),
    }
    if selected.size == 0:
        return class_override, diagnostics, stats

    try:
        image_path = _band_fits_path(_expand(args.coadd_root), band, int(args.tract), patch)
        archive_catalog_root = _expand(getattr(args, "band_catalog_root", None) or getattr(args, "catalog_root"))
        archive_path = _band_catalog_path(archive_catalog_root, band, int(args.tract), patch)
        image, image_origin = _read_exposure_image_plane(image_path, clean_nonfinite=not args.no_clean_nonfinite)
    except Exception as exc:
        stats["skipped_missing_inputs"] = int(selected.size)
        stats["error"] = 1
        diagnostics["pu_remeasure_reason"][selected] = f"remeasure_input_error:{exc}"
        return class_override, diagnostics, stats

    try:
        with fits.open(archive_path, memmap=True, ignore_missing_end=True) as archive_hdul:
            if len(archive_hdul) <= 4:
                raise RuntimeError("archive catalog lacks footprint archive HDUs")
            archive_main = archive_hdul[int(args.catalog_hdu)].data
            archive = archive_hdul[2].data
            footprint_refs = archive_hdul[3].data
            spans_table = archive_hdul[4].data
            heavy_table = archive_hdul[6].data if len(archive_hdul) > 6 else None

            archive_ids = np.asarray(archive_main["id"], dtype=np.int64)
            archive_order = np.argsort(archive_ids, kind="mergesort")
            selected_ids = np.asarray(shaped["id"][selected], dtype=np.int64)
            id_pos = np.searchsorted(archive_ids[archive_order], selected_ids)
            in_range = id_pos < archive_order.size
            archive_rows = np.full(selected.shape, -1, dtype=np.int64)
            matched_ids = np.zeros(selected.shape, dtype=bool)
            matched_ids[in_range] = archive_ids[archive_order[id_pos[in_range]]] == selected_ids[in_range]
            archive_rows[matched_ids] = archive_order[id_pos[matched_ids]]

            archive_index = _ArchiveIndex.from_archive(archive)
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
                heavy_lookup = _ArchiveLookup(
                    row0=np.full(selected.shape, -1, dtype=np.int64),
                    nrows=np.zeros(selected.shape, dtype=np.int64),
                    found=np.zeros(selected.shape, dtype=bool),
                )

            refit_matched = _table_bool_column(shaped, "pu_refit_kron_radius_matched")
            row_records: Dict[int, Dict[str, object]] = {}
            for local_index, row in enumerate(selected):
                geom = _source_geometry_from_shaped(shaped, int(row))
                if geom is None:
                    class_override[row] = "ignore"
                    diagnostics["pu_remeasure_reason"][row] = "invalid_refit_ellipse"
                    continue
                major = float(geom["a"])
                minor = float(geom["b"])
                theta = float(geom["theta"])
                has_heavy = bool(heavy_lookup.found[local_index] and heavy_lookup.row0[local_index] >= 0)
                use_heavy = bool(refit_matched[row] and has_heavy and heavy_table is not None)
                if use_heavy:
                    flux, ap_pixels, _source_pixels = _measure_heavy_ellipse(
                        spans_table=spans_table,
                        heavy_table=heavy_table,
                        span0=int(spanset_lookup.row0[local_index]),
                        nspan=int(spanset_lookup.nrows[local_index]),
                        heavy_row=int(heavy_lookup.row0[local_index]),
                        cx=float(geom["x"]),
                        cy=float(geom["y"]),
                        major=major,
                        minor=minor,
                        theta=theta,
                    )
                    surface = "heavy_footprint"
                else:
                    flux, ap_pixels, _source_pixels = _measure_direct_ellipse(
                        image=image,
                        image_origin=image_origin,
                        cx=float(geom["x"]),
                        cy=float(geom["y"]),
                        major=major,
                        minor=minor,
                        theta=theta,
                    )
                    surface = "direct_image_fallback"
                new_mag = float(_magnitude_from_flux(flux, zeropoint=zeropoint))
                new_diff = abs(float(ap2_mag[row]) - new_mag) if np.isfinite(new_mag) else np.nan
                aperture_area = math.pi * major * minor
                axis_ratio = max(major, minor) / min(major, minor) if min(major, minor) > 0.0 else np.inf
                fp_ap_pixels, _fp_total = _footprint_ellipse_counts(
                    spans_table=spans_table,
                    span0=int(spanset_lookup.row0[local_index]),
                    nspan=int(spanset_lookup.nrows[local_index]),
                    cx=float(geom["x"]),
                    cy=float(geom["y"]),
                    major=major,
                    minor=minor,
                    theta=theta,
                )
                fill_fraction = float(fp_ap_pixels) / float(aperture_area) if aperture_area > 0.0 else np.nan

                diagnostics["pu_remeasure_ap2_mag"][row] = float(ap2_mag[row])
                diagnostics["pu_remeasure_catalog_kron_mag"][row] = float(kron_mag[row])
                diagnostics["pu_remeasure_old_absdiff"][row] = float(old_diff[row])
                diagnostics["pu_remeasure_kron_flux"][row] = float(flux) if np.isfinite(flux) else np.nan
                diagnostics["pu_remeasure_kron_mag"][row] = new_mag if np.isfinite(new_mag) else np.nan
                diagnostics["pu_remeasure_absdiff"][row] = new_diff if np.isfinite(new_diff) else np.nan
                diagnostics["pu_remeasure_aperture_area"][row] = float(aperture_area)
                diagnostics["pu_remeasure_footprint_fill_fraction"][row] = fill_fraction if np.isfinite(fill_fraction) else np.nan
                diagnostics["pu_remeasure_axis_ratio"][row] = axis_ratio if np.isfinite(axis_ratio) else np.nan
                diagnostics["pu_remeasure_surface"][row] = surface

                row_records[int(row)] = {
                    "geom": geom,
                    "new_diff": new_diff,
                    "ap2_mag": float(ap2_mag[row]),
                    "area": aperture_area,
                    "axis_ratio": axis_ratio,
                    "fill_fraction": fill_fraction,
                    "reasons": [],
                }
                stats[surface] = stats.get(surface, 0) + 1
    except Exception as exc:
        stats["error"] = 1
        diagnostics["pu_remeasure_reason"][selected] = f"remeasure_archive_error:{exc}"
        return class_override, diagnostics, stats

    ignore_area_max = float(getattr(args, "pu_remeasure_ignore_area_max", 10000.0))
    faint_mag_min = float(getattr(args, "pu_remeasure_faint_mag_min", 28.0))
    faint_area_max = float(getattr(args, "pu_remeasure_faint_area_max", 900.0))
    axis_ratio_max = float(getattr(args, "pu_remeasure_axis_ratio_max", 5.0))
    small_fill_threshold = float(getattr(args, "pu_remeasure_small_footprint_fill_threshold", 0.2))
    keep_abs_max = float(getattr(args, "pu_remeasure_clean_abs_max", 1.0))
    center_abs_max = float(getattr(args, "pu_remeasure_center_only_abs_max", 1.5))
    containment_threshold = float(getattr(args, "pu_remeasure_containment_threshold", 0.80))

    for row, record in row_records.items():
        reasons: List[str] = []
        diff = float(record["new_diff"])
        area = float(record["area"])
        axis_ratio = float(record["axis_ratio"])
        fill_fraction = float(record["fill_fraction"])
        ap2_value = float(record["ap2_mag"])
        if np.isfinite(diff) and diff <= center_abs_max and np.isfinite(fill_fraction) and fill_fraction < small_fill_threshold:
            reasons.append("small_footprint_large_aperture")
        if np.isfinite(area) and area > ignore_area_max:
            reasons.append(f"used_aperture_area_gt_{ignore_area_max:g}")
        if np.isfinite(ap2_value) and ap2_value > faint_mag_min and np.isfinite(area) and area > faint_area_max:
            reasons.append(f"ap2_mag_gt_{faint_mag_min:g}_and_area_gt_{faint_area_max:g}")
        if np.isfinite(axis_ratio) and axis_ratio > axis_ratio_max:
            reasons.append(f"axis_ratio_gt_{axis_ratio_max:g}")
        if not np.isfinite(diff):
            reasons.append("invalid_new_diff")
        elif diff > center_abs_max:
            reasons.append(f"new_absdiff_gt_{center_abs_max:g}")
        record["reasons"] = reasons

    unit_points = _unit_disk_points()
    clean_mask = np.asarray(result["clean"], dtype=bool)
    small_pool: List[Dict[str, float]] = []
    seen: set[int] = set()
    for row in np.flatnonzero(clean_mask):
        geom = _source_geometry_from_shaped(shaped, int(row))
        if geom is not None:
            small_pool.append(geom)
            seen.add(int(geom["source_id"]))
    for row, record in row_records.items():
        if record["reasons"]:
            continue
        geom = record["geom"]
        sid = int(geom["source_id"])
        if sid not in seen:
            small_pool.append(geom)
            seen.add(sid)

    for row, record in row_records.items():
        if record["reasons"]:
            continue
        big = record["geom"]
        possible = []
        for small in small_pool:
            if int(small["source_id"]) == int(big["source_id"]):
                continue
            if small["area"] >= big["area"]:
                continue
            dx = float(small["x"]) - float(big["x"])
            dy = float(small["y"]) - float(big["y"])
            if dx * dx + dy * dy > (float(small["rmax"]) + float(big["rmax"])) ** 2:
                continue
            possible.append(small)
        possible.sort(key=lambda item: float(item["area"]))
        for small in possible:
            frac = _ellipse_contains_fraction(small, big, unit_points)
            if frac >= containment_threshold:
                record["reasons"].append(
                    f"contains_smaller_kron_ge_{containment_threshold:.2f}:small_id={int(small['source_id'])},frac={frac:.3f}"
                )
                break

    for row, record in row_records.items():
        reasons = list(record["reasons"])
        diff = float(record["new_diff"])
        if reasons:
            label = "ignore"
        elif np.isfinite(diff) and diff < keep_abs_max:
            label = "clean"
        elif np.isfinite(diff) and diff <= center_abs_max:
            label = "center_only"
        else:
            label = "ignore"
            reasons.append("new_diff_outside_keep_bins")
        class_override[row] = label
        reason = ";".join(reasons) if reasons else f"remeasured_ap2_kron_{label}"
        diagnostics["pu_remeasure_reason"][row] = reason
        stats[label] = stats.get(label, 0) + 1
    return class_override, diagnostics, stats


def _exclude_catalog_rows_by_id(table: Table, excluded: Table) -> Table:
    if len(table) == 0 or len(excluded) == 0:
        return table
    id_column = "id" if "id" in table.colnames and "id" in excluded.colnames else (
        "source_id" if "source_id" in table.colnames and "source_id" in excluded.colnames else None
    )
    if id_column is None:
        return table
    excluded_ids = set(int(value) for value in np.asarray(excluded[id_column], dtype=np.int64))
    keep = np.asarray([int(value) not in excluded_ids for value in np.asarray(table[id_column], dtype=np.int64)], dtype=bool)
    return table[keep]


def _classify_pu_catalog(
    table: Table,
    args: argparse.Namespace,
    *,
    band: Optional[str] = None,
    patch: Optional[str] = None,
) -> Tuple[Table, Table, Table, Table, Dict[str, object]]:
    table = _attach_pu_kron_refit(table, args, band=band, patch=patch)
    shaped = add_ellipse_columns(table, shape_source=args.target_shape_source)
    result = classify_pu_sources(table, _pu_args(args, band=band))
    class_name = np.asarray(result["class_name"], dtype=object)
    reasons = np.asarray(result["reasons"], dtype=object)
    area = np.asarray(result["area"], dtype=np.float32)
    mag = np.asarray(result.get("mag", np.full(len(shaped), np.nan)), dtype=np.float32)
    shaped["pu_class"] = class_name.astype(str)
    shaped["pu_reason"] = reasons.astype(str)
    shaped["pu_kron_area"] = area
    shaped["pu_mag"] = mag
    clean_mask = np.asarray(result["clean"], dtype=bool).copy()
    center_only_mask = np.asarray(result["center_only"], dtype=bool).copy()
    ignore_mask = np.asarray(result.get("ignore", result["b_class"]), dtype=bool).copy()
    remeasure_class, remeasure_diagnostics, remeasure_stats = _remeasure_ap2_kron_outliers(
        shaped,
        result,
        args,
        band=band,
        patch=patch,
    )
    if remeasure_diagnostics:
        for key, value in remeasure_diagnostics.items():
            shaped[key] = value
    override_mask = np.asarray([bool(str(value)) for value in remeasure_class], dtype=bool)
    if np.any(override_mask):
        remeasure_class_str = np.asarray(remeasure_class, dtype=str)
        clean_mask[override_mask] = remeasure_class_str[override_mask] == "clean"
        center_only_mask[override_mask] = remeasure_class_str[override_mask] == "center_only"
        ignore_mask[override_mask] = remeasure_class_str[override_mask] == "ignore"
        class_name = np.asarray(shaped["pu_class"], dtype=object)
        class_name[override_mask] = remeasure_class_str[override_mask]
        shaped["pu_class"] = class_name.astype(str)
        if "pu_remeasure_reason" in shaped.colnames:
            old_reasons = np.asarray(shaped["pu_reason"], dtype=object)
            new_reasons = np.asarray(shaped["pu_remeasure_reason"], dtype=object)
            merged_reasons = old_reasons.copy()
            for idx in np.flatnonzero(override_mask):
                old = str(old_reasons[idx])
                new = str(new_reasons[idx])
                if new:
                    merged_reasons[idx] = f"{old};{new}" if old and old != "--" else new
            shaped["pu_reason"] = merged_reasons.astype(str)
    result["remeasure_ap2_kron"] = remeasure_stats
    result["clean"] = clean_mask
    result["center_only"] = center_only_mask
    result["ignore"] = ignore_mask
    clean = shaped[clean_mask]
    center_only = shaped[center_only_mask]
    ignore = shaped[ignore_mask]
    return clean, center_only, ignore, shaped, result


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
    out = np.full(len(table), np.nan, dtype=np.float32)
    found = False
    for name in names:
        if name in table.colnames:
            vals = np.asarray(table[name], dtype=np.float32)
            take = ~np.isfinite(out) & np.isfinite(vals)
            if np.any(take):
                out[take] = vals[take]
                found = True
            if np.isfinite(out).all():
                break
    return out if found else None


_POSITION_COLUMN_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("base_SdssCentroid_x", "base_SdssCentroid_y"),
    ("base_SdssShape_x", "base_SdssShape_y"),
    ("base_NaiveCentroid_x", "base_NaiveCentroid_y"),
    ("deblend_psfCenter_x", "deblend_psfCenter_y"),
    ("slot_Centroid_x", "slot_Centroid_y"),
    ("x", "y"),
)


def _resolve_position_columns(table: Table, x_col: str, y_col: str) -> Tuple[str, str]:
    if x_col in table.colnames and y_col in table.colnames:
        return x_col, y_col
    for candidate_x, candidate_y in _POSITION_COLUMN_PAIRS:
        if candidate_x in table.colnames and candidate_y in table.colnames:
            return candidate_x, candidate_y
    available = ", ".join(table.colnames[:20])
    suffix = "..." if len(table.colnames) > 20 else ""
    raise KeyError(
        f"catalog must contain {x_col!r} and {y_col!r}, or one of "
        f"{_POSITION_COLUMN_PAIRS}; available columns: {available}{suffix}"
    )


def _require_position_columns(table: Table, x_col: str, y_col: str) -> Tuple[np.ndarray, np.ndarray]:
    resolved_x_col, resolved_y_col = _resolve_position_columns(table, x_col, y_col)
    return np.asarray(table[resolved_x_col], dtype=np.float32), np.asarray(table[resolved_y_col], dtype=np.float32)


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

    if shape_source in {"kron", "circular_kron"}:
        kron = _first_finite_column(
            table,
            (
                "pu_refit_kron_radius",
                "ext_photometryKron_KronFlux_radius",
                "ext_photometryKron_KronFlux_radius_for_radius",
            ),
        )
        if kron is not None:
            kron = np.asarray(kron, dtype=np.float32)
            determinant_radius = np.sqrt(np.maximum(major * minor, 0.0))
            valid = np.isfinite(kron) & (kron > 0) & np.isfinite(determinant_radius) & (determinant_radius > 0)
            if shape_source == "kron":
                scale = np.ones_like(kron, dtype=np.float32)
                scale[valid] = kron[valid] / determinant_radius[valid]
                major = np.where(valid, major * scale, major)
                minor = np.where(valid, minor * scale, minor)
            else:
                major = np.where(valid, kron, major)
                minor = np.where(valid, kron, minor)
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
    if len(table) == 0:
        out = table.copy(copy_data=True)
        out["centroid_local_x"] = np.asarray([], dtype=np.float32)
        out["centroid_local_y"] = np.asarray([], dtype=np.float32)
        return out
    resolved_x_col, resolved_y_col = _resolve_position_columns(table, x_col, y_col)
    x, y = _require_position_columns(table, resolved_x_col, resolved_y_col)
    cropped = table[_contains(x, y, spec, margin=margin)]
    out = cropped.copy(copy_data=True)
    out["centroid_local_x"] = np.asarray(out[resolved_x_col], dtype=np.float32) - float(spec.x0)
    out["centroid_local_y"] = np.asarray(out[resolved_y_col], dtype=np.float32) - float(spec.y0)
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
        if np.isfinite(major[idx]) and np.isfinite(minor[idx]) and np.isfinite(angle[idx]):
            a = float(max(major[idx] * ellipse_sigma, 1.5))
            b = float(max(minor[idx] * ellipse_sigma, 1.5))
            theta = float(angle[idx])
        else:
            a = b = float(max(2.0 * ellipse_sigma, 1.5))
            theta = 0.0
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
            dist = np.abs(xx_full[cy0:cy1, cx0:cx1] - cx) + np.abs(yy_full[cy0:cy1, cx0:cx1] - cy)
            vals = np.ceil(np.clip(level_radius - dist, a_min=0, a_max=None)).astype(np.int16)
            confidence[cy0:cy1, cx0:cx1] = np.maximum(confidence[cy0:cy1, cx0:cx1], vals)

    return {
        "seg": seg,
        "confidence": confidence,
        "shape": shape,
        "shape_weight": shape_weight,
        "overlap_count": overlap_count,
    }


def _paint_ellipse_mask(
    mask: np.ndarray,
    sources: Table,
    spec: TileSpec,
    *,
    x_col: str,
    y_col: str,
    ellipse_sigma: float,
) -> None:
    if len(sources) == 0:
        return
    h, w = mask.shape
    x_global, y_global = _require_position_columns(sources, x_col, y_col)
    major = np.asarray(sources["ellipse_major_sigma"], dtype=np.float32)
    minor = np.asarray(sources["ellipse_minor_sigma"], dtype=np.float32)
    angle = np.asarray(sources["ellipse_theta"], dtype=np.float32)
    yy_full, xx_full = np.mgrid[0:h, 0:w]
    for idx in range(len(sources)):
        cx = float(x_global[idx] - spec.x0)
        cy = float(y_global[idx] - spec.y0)
        if np.isfinite(major[idx]) and np.isfinite(minor[idx]) and np.isfinite(angle[idx]):
            a = float(max(major[idx] * ellipse_sigma, 1.5))
            b = float(max(minor[idx] * ellipse_sigma, 1.5))
            theta = float(angle[idx])
        else:
            a = b = float(max(2.0 * ellipse_sigma, 1.5))
            theta = 0.0
        radius = int(math.ceil(max(a, b))) + 2
        cx_i = int(round(cx))
        cy_i = int(round(cy))
        y0, y1 = max(0, cy_i - radius), min(h, cy_i + radius + 1)
        x0, x1 = max(0, cx_i - radius), min(w, cx_i + radius + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        dx = xx_full[y0:y1, x0:x1] - cx
        dy = yy_full[y0:y1, x0:x1] - cy
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        xr = cos_t * dx + sin_t * dy
        yr = -sin_t * dx + cos_t * dy
        ellipse = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
        mask[y0:y1, x0:x1][ellipse] = 1


def _paint_confidence(
    confidence: np.ndarray,
    confidence_weight: np.ndarray,
    sources: Table,
    spec: TileSpec,
    *,
    x_col: str,
    y_col: str,
    confidence_levels: int,
    weight: float,
) -> None:
    if len(sources) == 0:
        return
    h, w = confidence.shape
    x_global, y_global = _require_position_columns(sources, x_col, y_col)
    yy_full, xx_full = np.mgrid[0:h, 0:w]
    for idx in range(len(sources)):
        cx = float(x_global[idx] - spec.x0)
        cy = float(y_global[idx] - spec.y0)
        cx_i = int(round(cx))
        cy_i = int(round(cy))
        if not (0 <= cx_i < w and 0 <= cy_i < h):
            continue
        level_radius = confidence_levels - 1
        cy0, cy1 = max(0, cy_i - level_radius), min(h, cy_i + level_radius + 1)
        cx0, cx1 = max(0, cx_i - level_radius), min(w, cx_i + level_radius + 1)
        dist = np.abs(xx_full[cy0:cy1, cx0:cx1] - cx) + np.abs(yy_full[cy0:cy1, cx0:cx1] - cy)
        vals = np.ceil(np.clip(level_radius - dist, a_min=0, a_max=None)).astype(np.int16)
        region = vals > 0
        patch_conf = confidence[cy0:cy1, cx0:cx1]
        patch_weight = confidence_weight[cy0:cy1, cx0:cx1]
        patch_conf[region] = np.maximum(patch_conf[region], vals[region])
        patch_weight[region] = np.maximum(patch_weight[region], float(weight))


def make_pu_dense_targets(
    clean_sources: Table,
    center_only_sources: Table,
    ignore_sources: Table,
    spec: TileSpec,
    *,
    x_col: str,
    y_col: str,
    ellipse_sigma: float,
    confidence_levels: int,
    core_radius: int,
    center_only_weight: float,
    lsst_background_mask: Optional[np.ndarray] = None,
    strict_center_only_sources: Optional[Table] = None,
    strict_center_only_ellipse_sigma: float = 1.0,
) -> Dict[str, np.ndarray]:
    targets = make_dense_targets(
        clean_sources,
        spec,
        x_col=x_col,
        y_col=y_col,
        ellipse_sigma=ellipse_sigma,
        confidence_levels=confidence_levels,
        core_radius=core_radius,
    )
    h = w = spec.size
    clean_mask = (targets["seg"] > 0).astype(np.uint8)
    center_only_mask = np.zeros((h, w), dtype=np.uint8)
    ignore_mask = np.zeros((h, w), dtype=np.uint8)
    strict_center_only_mask = np.zeros((h, w), dtype=np.uint8)
    legacy_strict_ignore_mask = np.zeros((h, w), dtype=np.uint8)
    a_failed_mask = np.zeros((h, w), dtype=np.uint8)
    if "pu_class" in ignore_sources.colnames:
        classes = np.asarray(ignore_sources["pu_class"], dtype=str)
        a_failed_sources = ignore_sources[classes == "a_failed"]
        ordinary_ignore_sources = ignore_sources[classes != "a_failed"]
    else:
        a_failed_sources = Table()
        ordinary_ignore_sources = ignore_sources
    _paint_ellipse_mask(center_only_mask, center_only_sources, spec, x_col=x_col, y_col=y_col, ellipse_sigma=ellipse_sigma)
    _paint_ellipse_mask(ignore_mask, ordinary_ignore_sources, spec, x_col=x_col, y_col=y_col, ellipse_sigma=ellipse_sigma)
    _paint_ellipse_mask(a_failed_mask, a_failed_sources, spec, x_col=x_col, y_col=y_col, ellipse_sigma=ellipse_sigma)
    if strict_center_only_sources is not None and len(strict_center_only_sources):
        _paint_ellipse_mask(
            strict_center_only_mask,
            strict_center_only_sources,
            spec,
            x_col=x_col,
            y_col=y_col,
            ellipse_sigma=strict_center_only_ellipse_sigma,
        )
    clean_bool = clean_mask > 0
    strict_center_only_bool = strict_center_only_mask > 0
    center_bool = ((center_only_mask > 0) | strict_center_only_bool) & ~clean_bool
    ignore_bool = (ignore_mask > 0) & ~clean_bool & ~center_bool
    a_failed_bool = (a_failed_mask > 0) & ~clean_bool & ~center_bool & ~ignore_bool
    if lsst_background_mask is not None:
        lsst_background_bool = np.asarray(lsst_background_mask, dtype=bool)
        if lsst_background_bool.shape != (h, w):
            raise ValueError(f"lsst_background_mask shape {lsst_background_bool.shape} != {(h, w)}")
        background_bool = lsst_background_bool & ~clean_bool & ~center_bool & ~ignore_bool
        ignore_bool = ignore_bool | (~background_bool & ~clean_bool & ~center_bool)
    else:
        background_bool = np.zeros((h, w), dtype=bool)
        ignore_bool = ignore_bool | (~clean_bool & ~center_bool)
    clean_mask = clean_bool.astype(np.uint8)
    center_only_mask = center_bool.astype(np.uint8)
    strict_center_only_mask = (strict_center_only_bool & center_bool).astype(np.uint8)
    ignore_mask = ignore_bool.astype(np.uint8)

    source_union_mask = (clean_bool | center_bool | ignore_bool).astype(np.uint8)
    background_mask = background_bool.astype(np.uint8)
    pu_class_mask = np.zeros((h, w), dtype=np.uint8)
    pu_class_mask[clean_bool] = 1
    pu_class_mask[center_bool] = 2
    pu_class_mask[ignore_bool] = 3
    pu_class_mask[background_bool] = 4
    pu_class_mask[strict_center_only_mask > 0] = 5

    confidence_weight = np.ones((h, w), dtype=np.float32)
    seg_loss_weight = np.ones((h, w), dtype=np.float32)
    uncertain = center_bool | ignore_bool
    confidence_weight[uncertain] = 0.0
    seg_loss_weight[uncertain] = 0.0
    targets["shape_weight"][uncertain] = 0.0
    _paint_confidence(
        targets["confidence"],
        confidence_weight,
        center_only_sources,
        spec,
        x_col=x_col,
        y_col=y_col,
        confidence_levels=confidence_levels,
        weight=center_only_weight,
    )
    _paint_confidence(
        targets["confidence"],
        confidence_weight,
        clean_sources,
        spec,
        x_col=x_col,
        y_col=y_col,
        confidence_levels=confidence_levels,
        weight=1.0,
    )

    targets.update(
        {
            "clean_mask": clean_mask,
            "center_only_mask": center_only_mask,
            "ignore_mask": ignore_mask,
            "strict_center_only_mask": strict_center_only_mask,
            "strict_ignore_mask": legacy_strict_ignore_mask,
            "source_union_mask": source_union_mask,
            "background_mask": background_mask,
            "pu_class_mask": pu_class_mask,
            "confidence_weight": confidence_weight,
            "seg_loss_weight": seg_loss_weight,
            "center_only_weight_value": np.asarray(float(center_only_weight), dtype=np.float32),
        }
    )
    return targets


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
    for key in (
        "clean_mask",
        "center_only_mask",
        "ignore_mask",
        "strict_center_only_mask",
        "strict_ignore_mask",
        "source_union_mask",
        "background_mask",
        "pu_class_mask",
    ):
        if key in targets:
            fits.writeto(output_fits_prefix.with_name(output_fits_prefix.name + f"_{key}.fits"), targets[key], overwrite=True)


def _first_present_float_column(table: Table, names: Sequence[str], default: float = np.nan) -> np.ndarray:
    out = np.full(len(table), float(default), dtype=np.float32)
    filled = np.zeros(len(table), dtype=bool)
    for name in names:
        if name in table.colnames:
            values = np.asarray(table[name], dtype=np.float32)
            take = ~filled & np.isfinite(values)
            out[take] = values[take]
            filled[take] = True
            if filled.all():
                break
    return out


def _metadata_from_catalog(table: Table) -> Dict[str, np.ndarray]:
    centers = np.zeros((len(table), 2), dtype=np.float32)
    if len(table):
        centers[:, 0] = np.asarray(table["centroid_local_x"], dtype=np.float32)
        centers[:, 1] = np.asarray(table["centroid_local_y"], dtype=np.float32)
    ids = np.asarray(table["id"], dtype=np.int64) if "id" in table.colnames else np.arange(len(table), dtype=np.int64)
    xx = _first_present_float_column(table, ("base_SdssShape_xx", "ext_shapeHSM_HsmSourceMoments_xx"), default=4.0)
    yy = _first_present_float_column(table, ("base_SdssShape_yy", "ext_shapeHSM_HsmSourceMoments_yy"), default=4.0)
    xy = _first_present_float_column(table, ("base_SdssShape_xy", "ext_shapeHSM_HsmSourceMoments_xy"), default=0.0)
    kron_radius = _first_present_float_column(
        table,
        ("pu_refit_kron_radius", "ext_photometryKron_KronFlux_radius", "ext_photometryKron_KronFlux_radius_for_radius"),
        default=np.nan,
    )
    footprint = _first_present_float_column(table, ("base_FootprintArea_value",), default=0.0)
    return {
        "centers": centers.astype(np.float32),
        "ids": ids.astype(np.int64),
        "moments": np.stack([xx, yy, xy], axis=1).astype(np.float32),
        "kron_radius": kron_radius.astype(np.float32),
        "footprint": footprint.astype(np.float32),
    }


def write_catalog_metadata(table: Table, output_npz: Path) -> None:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **_metadata_from_catalog(table))


def write_ids_metadata(table: Table, output_npz: Path) -> None:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(table["id"], dtype=np.int64) if "id" in table.colnames else np.arange(len(table), dtype=np.int64)
    np.savez_compressed(output_npz, ids=ids.astype(np.int64))


def _read_exposure_image_plane(source_path: Path, *, clean_nonfinite: bool = True) -> Tuple[np.ndarray, Tuple[int, int]]:
    with fits.open(source_path, memmap=False) as hdul:
        plane_indices = _plane_hdu_indices(hdul)
        hdu = hdul[plane_indices["IMAGE"]]
        data = np.asarray(hdu.data, dtype=np.float32).copy()
        if clean_nonfinite and not np.all(np.isfinite(data)):
            fill = _finite_replacement(data)
            data = np.nan_to_num(data, nan=fill, posinf=fill, neginf=fill).astype(np.float32, copy=False)
        return data, _origin_from_ltv(hdu.header)


def _aperture_annulus_snr(
    image: np.ndarray,
    centers: np.ndarray,
    *,
    ap_radius: float = 6.0,
    annulus_r_in: float = 10.0,
    annulus_r_out: float = 15.0,
    exclude_centers: Optional[np.ndarray] = None,
    annulus_exclude_radius: float = 0.0,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    snr = np.full((centers.shape[0],), np.nan, dtype=np.float32)
    if image.ndim != 2 or centers.size == 0:
        return snr
    exclude_xy = np.asarray(exclude_centers, dtype=np.float32).reshape(-1, 2) if exclude_centers is not None else centers
    exclude_radius = float(annulus_exclude_radius)
    h, w = image.shape
    rmax = float(max(ap_radius, annulus_r_out))
    for idx, (cx, cy) in enumerate(centers):
        if not (np.isfinite(cx) and np.isfinite(cy)):
            continue
        x0 = max(0, int(math.floor(float(cx) - rmax - 1.0)))
        x1 = min(w, int(math.ceil(float(cx) + rmax + 2.0)))
        y0 = max(0, int(math.floor(float(cy) - rmax - 1.0)))
        y1 = min(h, int(math.ceil(float(cy) + rmax + 2.0)))
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rr = np.sqrt((xx.astype(np.float32) - float(cx)) ** 2 + (yy.astype(np.float32) - float(cy)) ** 2)
        patch = image[y0:y1, x0:x1]
        finite = np.isfinite(patch)
        ap_mask = (rr <= float(ap_radius)) & finite
        ann_mask = (rr >= float(annulus_r_in)) & (rr < float(annulus_r_out)) & finite
        if exclude_radius > 0.0 and exclude_xy.size:
            nearby = (
                (exclude_xy[:, 0] >= x0 - exclude_radius)
                & (exclude_xy[:, 0] < x1 + exclude_radius)
                & (exclude_xy[:, 1] >= y0 - exclude_radius)
                & (exclude_xy[:, 1] < y1 + exclude_radius)
            )
            for ex, ey in exclude_xy[nearby]:
                if not (np.isfinite(ex) and np.isfinite(ey)):
                    continue
                ann_mask &= ((xx.astype(np.float32) - float(ex)) ** 2 + (yy.astype(np.float32) - float(ey)) ** 2) > exclude_radius**2
        ap_vals = patch[ap_mask]
        ann_vals = patch[ann_mask]
        if ap_vals.size == 0 or ann_vals.size < 2:
            continue
        bkg = float(np.median(ann_vals))
        sigma = float(np.std(ann_vals.astype(np.float64), ddof=1))
        if not math.isfinite(sigma) or sigma <= 0.0:
            continue
        flux = float(np.sum(ap_vals.astype(np.float64))) - bkg * float(ap_vals.size)
        noise = sigma * math.sqrt(float(ap_vals.size))
        if noise > 0.0:
            snr[idx] = float(flux / noise)
    return snr


def _centers_for_image_snr(
    table: Table,
    *,
    image_origin: Tuple[int, int],
    x_col: str,
    y_col: str,
) -> np.ndarray:
    centers = np.zeros((len(table), 2), dtype=np.float32)
    if len(table) == 0:
        return centers
    x, y = _require_position_columns(table, x_col, y_col)
    centers[:, 0] = np.asarray(x, dtype=np.float32) - float(image_origin[0])
    centers[:, 1] = np.asarray(y, dtype=np.float32) - float(image_origin[1])
    return centers


def _classify_clean_by_noncoadd_snr(
    clean_sources: Table,
    *,
    image: np.ndarray,
    image_origin: Tuple[int, int],
    args: argparse.Namespace,
) -> Tuple[Table, Table, Table, np.ndarray]:
    if len(clean_sources) == 0:
        empty = clean_sources.copy(copy_data=True)
        return empty, empty.copy(copy_data=True), empty.copy(copy_data=True), np.zeros((0,), dtype=np.float32)
    centers = _centers_for_image_snr(clean_sources, image_origin=image_origin, x_col=args.x_col, y_col=args.y_col)
    snr = _aperture_annulus_snr(
        image,
        centers,
        ap_radius=float(args.noncoadd_snr_ap_radius),
        annulus_r_in=float(args.noncoadd_snr_annulus_r_in),
        annulus_r_out=float(args.noncoadd_snr_annulus_r_out),
        exclude_centers=centers,
        annulus_exclude_radius=float(getattr(args, "noncoadd_snr_annulus_exclude_radius", 0.0)),
    )
    finite_snr = np.isfinite(snr)
    normal_keep = finite_snr & (snr >= float(args.noncoadd_snr_center_only_thresh))
    center_keep = finite_snr & (snr >= float(args.noncoadd_snr_ignore_thresh)) & (snr < float(args.noncoadd_snr_center_only_thresh))
    ignore_keep = (~finite_snr) | (snr < float(args.noncoadd_snr_ignore_thresh))

    annotated = clean_sources.copy(copy_data=True)
    visibility_class = np.full(len(annotated), "normal", dtype="U16")
    visibility_class[center_keep] = "center_only"
    visibility_class[ignore_keep] = "ignore"
    annotated["noncoadd_visibility_snr"] = snr.astype(np.float32)
    annotated["noncoadd_visibility_class"] = visibility_class
    return (
        annotated[normal_keep],
        annotated[center_keep],
        annotated[ignore_keep],
        snr.astype(np.float32),
    )


def _vstack_nonempty(parts: Sequence[Table]) -> Table:
    nonempty = [part for part in parts if len(part)]
    if not nonempty:
        return Table()
    if len(nonempty) == 1:
        return nonempty[0].copy(copy_data=True)
    return vstack(nonempty, join_type="outer", metadata_conflicts="silent")


def _local_centers_from_catalog(table: Table) -> np.ndarray:
    centers = np.zeros((len(table), 2), dtype=np.float32)
    if len(table) == 0:
        return centers
    if "centroid_local_x" not in table.colnames or "centroid_local_y" not in table.colnames:
        raise KeyError("visibility catalog must contain centroid_local_x/centroid_local_y")
    centers[:, 0] = np.asarray(table["centroid_local_x"], dtype=np.float32)
    centers[:, 1] = np.asarray(table["centroid_local_y"], dtype=np.float32)
    return centers


def _restore_center_only_shape_targets(
    targets: Dict[str, np.ndarray],
    center_only_sources: Table,
    spec: TileSpec,
    *,
    x_col: str,
    y_col: str,
    ellipse_sigma: float,
    confidence_levels: int,
    core_radius: int,
) -> None:
    if len(center_only_sources) == 0:
        return
    center_targets = make_dense_targets(
        center_only_sources,
        spec,
        x_col=x_col,
        y_col=y_col,
        ellipse_sigma=ellipse_sigma,
        confidence_levels=confidence_levels,
        core_radius=core_radius,
    )
    mask = center_targets["shape_weight"] > 0
    if not np.any(mask):
        return
    targets["shape"][:, mask] = center_targets["shape"][:, mask]
    targets["shape_weight"][mask] = np.maximum(targets["shape_weight"][mask], center_targets["shape_weight"][mask])


def _coadd_target_background_mask(coadd_patch_root: Path, band: str, tile_name: str) -> Optional[np.ndarray]:
    target_path = coadd_patch_root / "band_targets" / band / f"{tile_name}.npz"
    if not target_path.exists():
        target_path = coadd_patch_root / "targets" / f"{tile_name}.npz"
    if not target_path.exists():
        return None
    try:
        with np.load(target_path) as data:
            if "background_mask" in data:
                return np.asarray(data["background_mask"], dtype=bool)
    except Exception:
        return None
    return None


def _variant_lsst_background_root(args: argparse.Namespace) -> Optional[Path]:
    root = getattr(args, "variant_lsst_background_root", None)
    return _expand(root) if root is not None else None


def _read_cached_variant_background_mask(path: Path) -> Tuple[np.ndarray, Tuple[int, int]]:
    with np.load(path, allow_pickle=False) as data:
        if "background_mask" not in data:
            raise KeyError(f"{path} has no background_mask array")
        mask = np.asarray(data["background_mask"], dtype=bool)
        origin = data["origin_xy"] if "origin_xy" in data else np.asarray([0, 0], dtype=np.int32)
        if mask.ndim != 2:
            raise ValueError(f"{path} background_mask is not 2D: shape={mask.shape}")
        return mask, (int(origin[0]), int(origin[1]))


def _variant_lsst_background_mask(
    args: argparse.Namespace,
    *,
    variant: str,
    tract: int,
    patch: str,
    group: str,
    band: str,
    shape_yx: Tuple[int, int],
    image_origin: Tuple[int, int],
) -> Optional[Tuple[np.ndarray, Tuple[int, int]]]:
    root = _variant_lsst_background_root(args)
    if root is None:
        return None
    candidates = (
        root / variant / str(tract) / patch / group / band / "background_mask.npz",
        root / variant / str(tract) / patch / group / band / f"background_mask-{variant}-{band}-{tract}-{patch}-{group}.npz",
        root / variant / f"patch_{patch.replace(',', '_')}" / group / band / "background_mask.npz",
    )
    for candidate in candidates:
        if candidate.exists():
            return _read_cached_variant_background_mask(candidate)
    det_candidates = (
        root / variant / str(tract) / patch / group / band / f"det-{variant}-{band}-{tract}-{patch}-{group}.fits",
        root / variant / str(tract) / patch / group / band / "det.fits",
    )
    for candidate in det_candidates:
        if candidate.exists():
            return _read_det_background_mask(candidate, shape_yx, origin_xy=image_origin), image_origin
    print(
        f"WARNING: no variant LSST background found for {variant}/{tract}/{patch}/{group}/{band} under {root}; "
        "falling back to coadd target background if available.",
        flush=True,
    )
    return None


def _read_variant_label_sources(
    coadd_patch_root: Path,
    *,
    band: str,
    tract: int,
    patch: str,
    args: argparse.Namespace,
) -> Tuple[Table, Table, Table, Table]:
    filename = f"meas-{band}-{tract}-{patch}.fits"
    clean_path = coadd_patch_root / "band_reference_catalogs" / band / filename
    if not clean_path.exists():
        raise FileNotFoundError(f"missing coadd band reference catalog for variant labels: {clean_path}")
    clean = _read_table(clean_path, hdu=1, role="variant-band-clean", patch=patch, band=band)

    def _optional(dirname: str) -> Table:
        candidate = coadd_patch_root / dirname / band / filename
        if candidate.exists():
            return _read_table(candidate, hdu=1, role=f"variant-{dirname}", patch=patch, band=band)
        return Table()

    center_only = _optional("band_reference_center_only") if args.label_mode == "pu" else Table()
    ignore = _optional("band_reference_ignore") if args.label_mode == "pu" else Table()
    strict_center_only = _optional("band_reference_strict_center_only") if args.label_mode == "pu" else Table()
    return clean, center_only, ignore, strict_center_only


def _zscale_cache_path(
    zscale_root: Path,
    *,
    tract: int,
    patch: str,
    tile_name: str,
    bands: Sequence[str],
    fits_hdu: int,
    relative_root: Optional[str] = None,
) -> Path:
    band_key = "_".join(bands)
    if relative_root:
        return zscale_root / relative_root / "cutouts" / f"{tile_name}__{band_key}__hdu{fits_hdu}.pt"
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
    source_root = source_root.resolve()
    fast_root = fast_root.resolve()
    if source_root == fast_root:
        return
    dirs = (
        "targets",
        "tile_metadata",
        "reference_catalogs",
        "reference_catalogs_csv",
        "center_only_catalogs",
        "ignore_catalogs",
        "strict_center_only_catalogs",
        "strict_ignore_catalogs",
        "band_reference_catalogs",
        "band_reference_rejected",
        "band_reference_center_only",
        "band_reference_ignore",
        "band_reference_strict_center_only",
        "band_reference_strict_ignore",
        "band_reference_pu_all",
        "band_targets",
        "band_tile_metadata",
        "band_rejected_ids",
        "sources",
    )
    files = ("manifest.json", "tiles.csv", "cutout_paths.json")
    fast_root.mkdir(parents=True, exist_ok=True)
    for dirname in dirs:
        src = source_root / dirname
        dst = fast_root / dirname
        if src.exists() and src.resolve() != dst.resolve():
            shutil.copytree(src, dst, dirs_exist_ok=True)
    for filename in files:
        src = source_root / filename
        dst = fast_root / filename
        if src.exists() and src.resolve() != dst.resolve():
            shutil.copy2(src, dst)


def _read_existing_tile_specs(patch_root: Path) -> List[TileSpec]:
    tiles_csv = patch_root / "tiles.csv"
    if not tiles_csv.exists():
        raise FileNotFoundError(f"Existing coadd preprocessed patch has no tiles.csv: {tiles_csv}")

    def _int_field(row: Dict[str, str], key: str, default: int) -> int:
        text = str(row.get(key, "")).strip()
        return int(float(text)) if text not in {"", "None"} else int(default)

    specs: List[TileSpec] = []
    with tiles_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            specs.append(
                TileSpec(
                    name=name,
                    x0=_int_field(row, "x0", 0),
                    y0=_int_field(row, "y0", 0),
                    size=_int_field(row, "size", 512),
                    row=_int_field(row, "row", 0) if str(row.get("row", "")).strip() not in {"", "None"} else None,
                    col=_int_field(row, "col", 0) if str(row.get("col", "")).strip() not in {"", "None"} else None,
                    kind=str(row.get("kind", "grid") or "grid"),
                )
            )
    if not specs:
        raise RuntimeError(f"No tile specs found in {tiles_csv}")
    return specs


def _find_denoised_patch_dir(denoised_root: Path, patch: str) -> Optional[Path]:
    x_str, y_str = patch.split(",", 1)
    candidates = (
        denoised_root / f"patch_{x_str}_{y_str}",
        denoised_root / patch,
        denoised_root / patch.replace(",", "_"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _write_variant_tiles_csv(variant_root: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = ["name", "kind", "row", "col", "x0", "y0", "x1", "y1", "size", "variant_group", "base_tile_name"]
    variant_root.mkdir(parents=True, exist_ok=True)
    with (variant_root / "tiles.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _preprocess_image_variant_patch(
    args: argparse.Namespace,
    *,
    coadd_patch_root: Path,
    denoised_patch_dir: Path,
    variant: str,
    output_root: Path,
    bands: Sequence[str],
    patch: str,
) -> Dict[str, object]:
    specs = _read_existing_tile_specs(coadd_patch_root)
    requested_tiles = tuple(str(tile) for tile in getattr(args, "image_variant_tiles", ()) or ())
    if requested_tiles:
        tile_names = {tile.split("/", 1)[-1] for tile in requested_tiles}
        specs = [
            spec
            for spec in specs
            if spec.name in tile_names or any(name.endswith(f"_{spec.name}") for name in tile_names)
        ]
        if not specs:
            raise FileNotFoundError(f"No requested image variant tiles found under {coadd_patch_root}: {sorted(tile_names)}")
    groups = sorted(path for path in denoised_patch_dir.iterdir() if path.is_dir() and path.name.startswith("group_"))
    if not groups:
        groups = sorted(path for path in denoised_patch_dir.iterdir() if path.is_dir())
    requested_groups = tuple(str(group) for group in getattr(args, "image_variant_groups", ()) or ())
    if requested_groups:
        normalized_groups = {
            group if group.startswith("group_") else f"group_{int(group):02d}" for group in requested_groups
        }
        groups = [group for group in groups if group.name in normalized_groups]
    if not groups:
        raise FileNotFoundError(f"No denoised/noisy group directories found in {denoised_patch_dir}")

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    cutout_paths: Dict[str, Dict[str, str]] = {}
    effective_count_paths: Dict[str, Dict[str, str]] = {}
    zscale_written = 0
    zscale_skipped = 0
    missing: List[str] = []
    write_variant_labels = (
        bool(getattr(args, "noncoadd_snr_filter", True))
        and str(getattr(args, "label_mode", "legacy")) == "pu"
        and not bool(getattr(args, "skip_band_targets", False))
    )
    label_sources_by_band: Dict[str, Tuple[Table, Table, Table, Table]] = {}
    visibility_counts: Dict[str, Dict[str, int]] = {
        band: {"normal": 0, "center_only": 0, "ignore": 0} for band in bands
    }
    if write_variant_labels:
        for band in bands:
            label_sources_by_band[band] = _read_variant_label_sources(
                coadd_patch_root,
                band=band,
                tract=args.tract,
                patch=patch,
                args=args,
            )

    for group_dir in groups:
        group_images: Dict[str, Tuple[np.ndarray, Tuple[int, int]]] = {}
        if write_variant_labels:
            for band in bands:
                src = group_dir / band / f"{variant}.fits"
                if src.exists():
                    group_images[band] = _read_exposure_image_plane(src, clean_nonfinite=not args.no_clean_nonfinite)
        group_name = group_dir.name
        group_lsst_background_masks: Dict[str, Tuple[np.ndarray, Tuple[int, int]]] = {}
        if write_variant_labels and _variant_lsst_background_root(args) is not None:
            for band, (image, image_origin) in group_images.items():
                try:
                    cached = _variant_lsst_background_mask(
                        args,
                        variant=variant,
                        tract=args.tract,
                        patch=patch,
                        group=group_name,
                        band=band,
                        shape_yx=(int(image.shape[0]), int(image.shape[1])),
                        image_origin=image_origin,
                    )
                    if cached is not None:
                        group_lsst_background_masks[band] = cached
                except Exception as exc:
                    print(
                        f"WARNING: failed to read variant LSST background for "
                        f"{variant}/{args.tract}/{patch}/{group_name}/{band}: {exc}",
                        flush=True,
                    )
        for spec in specs:
            tile_name = f"{group_name}_{spec.name}"
            band_paths: Dict[str, str] = {}
            effective_paths: Dict[str, str] = {}
            for band in bands:
                src = group_dir / band / f"{variant}.fits"
                if not src.exists():
                    missing.append(str(src))
                    continue
                dst = output_root / "cutouts" / tile_name / band / f"{variant}-{band}-{args.tract}-{patch}-{group_name}.fits"
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
                eff_src = group_dir / band / "effective_count.fits"
                if eff_src.exists():
                    eff_dst = (
                        output_root
                        / "effective_count"
                        / "cutouts"
                        / tile_name
                        / band
                        / f"effective_count-{band}-{args.tract}-{patch}-{group_name}.fits"
                    )
                    effective_paths[band] = str(eff_dst)
                    if not args.skip_cutouts and not args.dry_run and (args.overwrite or not eff_dst.exists()):
                        crop_exposure_cutout(
                            source_path=eff_src,
                            output_path=eff_dst,
                            parent_x0=spec.x0,
                            parent_y0=spec.y0,
                            size=spec.size,
                            clean_nonfinite=False,
                            overwrite=True,
                        )
            if len(band_paths) == len(bands):
                cutout_paths[tile_name] = band_paths
                if effective_paths:
                    effective_count_paths[tile_name] = effective_paths
                if args.zscale_root is not None and not args.dry_run:
                    zscale_path = _zscale_cache_path(
                        _expand(args.zscale_root),
                        tract=args.tract,
                        patch=patch,
                        tile_name=tile_name,
                        bands=bands,
                        fits_hdu=args.zscale_fits_hdu,
                        relative_root=f"{variant}/{args.tract}/{patch}",
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
                if write_variant_labels and not args.dry_run:
                    primary_written = False
                    for band in bands:
                        if band not in group_images:
                            continue
                        image, image_origin = group_images[band]
                        clean_full, center_full, ignore_full, strict_center_full = label_sources_by_band[band]
                        clean_tile_all = crop_catalog_for_tile(
                            clean_full,
                            spec,
                            x_col=args.x_col,
                            y_col=args.y_col,
                            margin=0.0,
                        )
                        clean_mask_all = crop_catalog_for_tile(
                            clean_full,
                            spec,
                            x_col=args.x_col,
                            y_col=args.y_col,
                            margin=args.mask_margin,
                        )
                        center_tile = crop_catalog_for_tile(
                            center_full,
                            spec,
                            x_col=args.x_col,
                            y_col=args.y_col,
                            margin=0.0,
                        )
                        center_mask = crop_catalog_for_tile(
                            center_full,
                            spec,
                            x_col=args.x_col,
                            y_col=args.y_col,
                            margin=args.mask_margin,
                        )
                        ignore_tile = crop_catalog_for_tile(
                            ignore_full,
                            spec,
                            x_col=args.x_col,
                            y_col=args.y_col,
                            margin=0.0,
                        )
                        ignore_mask = crop_catalog_for_tile(
                            ignore_full,
                            spec,
                            x_col=args.x_col,
                            y_col=args.y_col,
                            margin=args.mask_margin,
                        )
                        strict_center_mask = crop_catalog_for_tile(
                            strict_center_full,
                            spec,
                            x_col=args.x_col,
                            y_col=args.y_col,
                            margin=args.mask_margin,
                        )

                        normal_tile, snr_center_tile, snr_ignore_tile, _tile_snr = _classify_clean_by_noncoadd_snr(
                            clean_tile_all,
                            image=image,
                            image_origin=image_origin,
                            args=args,
                        )
                        normal_mask, snr_center_mask, snr_ignore_mask, _mask_snr = _classify_clean_by_noncoadd_snr(
                            clean_mask_all,
                            image=image,
                            image_origin=image_origin,
                            args=args,
                        )
                        combined_center_tile = _vstack_nonempty([center_tile, snr_center_tile])
                        combined_ignore_tile = _vstack_nonempty([ignore_tile, snr_ignore_tile])
                        combined_center_mask = _vstack_nonempty([center_mask, snr_center_mask])
                        combined_ignore_mask = _vstack_nonempty([ignore_mask, snr_ignore_mask])

                        band_target = make_pu_dense_targets(
                            normal_mask,
                            combined_center_mask,
                            combined_ignore_mask,
                            spec,
                            x_col=args.x_col,
                            y_col=args.y_col,
                            ellipse_sigma=args.ellipse_sigma,
                            confidence_levels=args.confidence_levels,
                            core_radius=args.core_radius,
                            center_only_weight=args.pu_center_only_weight,
                            lsst_background_mask=(
                                _crop_full_mask_for_tile(
                                    group_lsst_background_masks[band][0],
                                    spec,
                                    group_lsst_background_masks[band][1],
                                )
                                if band in group_lsst_background_masks
                                else _coadd_target_background_mask(coadd_patch_root, band, spec.name)
                            ),
                            strict_center_only_sources=strict_center_mask,
                            strict_center_only_ellipse_sigma=args.pu_strict_bright_center_only_ellipse_sigma,
                        )
                        _restore_center_only_shape_targets(
                            band_target,
                            combined_center_mask,
                            spec,
                            x_col=args.x_col,
                            y_col=args.y_col,
                            ellipse_sigma=args.ellipse_sigma,
                            confidence_levels=args.confidence_levels,
                            core_radius=args.core_radius,
                        )
                        band_target["visibility_center_only_centers"] = _local_centers_from_catalog(snr_center_tile)
                        band_target["visibility_ignore_centers"] = _local_centers_from_catalog(snr_ignore_tile)
                        band_target["noncoadd_snr_thresholds"] = np.asarray(
                            [float(args.noncoadd_snr_ignore_thresh), float(args.noncoadd_snr_center_only_thresh)],
                            dtype=np.float32,
                        )
                        write_targets(band_target, output_root / "band_targets" / band / f"{tile_name}.npz", None)
                        write_catalog_metadata(normal_tile, output_root / "band_tile_metadata" / band / f"{tile_name}.npz")
                        visibility_counts[band]["normal"] += int(len(normal_tile))
                        visibility_counts[band]["center_only"] += int(len(snr_center_tile))
                        visibility_counts[band]["ignore"] += int(len(snr_ignore_tile))

                        if not primary_written:
                            write_targets(band_target, output_root / "targets" / f"{tile_name}.npz", None)
                            write_catalog_metadata(normal_tile, output_root / "tile_metadata" / f"{tile_name}.npz")
                            primary_written = True

                rows.append(
                    {
                        "name": tile_name,
                        "kind": spec.kind,
                        "row": spec.row,
                        "col": spec.col,
                        "x0": spec.x0,
                        "y0": spec.y0,
                        "x1": spec.x1,
                        "y1": spec.y1,
                        "size": spec.size,
                        "variant_group": group_name,
                        "base_tile_name": spec.name,
                    }
                )

    if missing:
        examples = "\n".join(missing[:10])
        raise FileNotFoundError(f"{variant} patch {patch} is missing {len(missing)} band image(s). First examples:\n{examples}")

    metadata: Dict[str, object] = {}
    source_manifest = coadd_patch_root / "manifest.json"
    if source_manifest.exists():
        try:
            metadata = json.loads(source_manifest.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    metadata.update(
        {
            "dataset_source": variant,
            "variant_source_root": str(denoised_patch_dir),
            "output_root": str(output_root),
            "bands": list(bands),
            "tract": args.tract,
            "patch": patch,
            "num_tiles": len(cutout_paths),
            "variant_groups": [group.name for group in groups],
            "base_patch_root": str(coadd_patch_root),
            "shared_label_root": str(coadd_patch_root),
            "shared_label_policy": "coadd_preprocessed",
            "variant_lsst_background_root": str(_variant_lsst_background_root(args))
            if _variant_lsst_background_root(args) is not None
            else None,
            "noncoadd_snr_filter": write_variant_labels,
            "noncoadd_snr_ignore_thresh": float(getattr(args, "noncoadd_snr_ignore_thresh", 2.0)),
            "noncoadd_snr_center_only_thresh": float(getattr(args, "noncoadd_snr_center_only_thresh", 3.0)),
            "noncoadd_visibility_counts": visibility_counts if write_variant_labels else None,
            "zscale_root": str(_expand(args.zscale_root)) if args.zscale_root is not None else None,
            "zscale_written": zscale_written,
            "zscale_skipped": zscale_skipped,
            "args": _jsonable_args(args),
        }
    )
    if not args.dry_run:
        _write_variant_tiles_csv(output_root, rows)
        (output_root / "cutout_paths.json").write_text(json.dumps(cutout_paths, indent=2), encoding="utf-8")
        if effective_count_paths:
            (output_root / "effective_count_paths.json").write_text(
                json.dumps(effective_count_paths, indent=2),
                encoding="utf-8",
            )
        (output_root / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if args.fast_root is not None:
            fast_root = _variant_patch_output_root(_expand(args.fast_root), variant, args.tract, patch)
            fast_root.mkdir(parents=True, exist_ok=True)
            for filename in ("manifest.json", "tiles.csv", "cutout_paths.json"):
                src = output_root / filename
                if src.exists():
                    shutil.copy2(src, fast_root / filename)
            if (output_root / "effective_count_paths.json").exists():
                shutil.copy2(output_root / "effective_count_paths.json", fast_root / "effective_count_paths.json")
            for dirname in ("targets", "tile_metadata", "band_targets", "band_tile_metadata"):
                src_dir = output_root / dirname
                dst_dir = fast_root / dirname
                if src_dir.exists() and src_dir.resolve() != dst_dir.resolve():
                    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    print(
        f"patch {patch}: prepared {len(cutout_paths)} {variant} tiles from {len(groups)} group(s); "
        f"zscale_written={zscale_written}, zscale_skipped={zscale_skipped}; output={output_root}",
        flush=True,
    )
    return metadata


def _sync_existing_variant_patch(
    args: argparse.Namespace,
    *,
    coadd_patch_root: Path,
    denoised_patch_dir: Optional[Path],
    variant: str,
    output_root: Path,
    bands: Sequence[str],
    patch: str,
) -> Dict[str, object]:
    del coadd_patch_root, denoised_patch_dir
    if not output_root.exists():
        raise FileNotFoundError(f"Existing image-variant patch root does not exist: {output_root}")

    manifest_path = output_root / "manifest.json"
    metadata: Dict[str, object] = {}
    if manifest_path.exists():
        try:
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    specs = _read_existing_tile_specs(output_root)
    tile_names = [spec.name for spec in specs]
    if not tile_names:
        raise RuntimeError(f"No existing image-variant tile specs found in {output_root / 'tiles.csv'}")

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
            f"Existing image-variant patch {variant}:{patch} is missing {len(missing)} requested band cutout(s). "
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
                relative_root=f"{variant}/{args.tract}/{patch}",
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
            "dataset_source": variant,
            "output_root": str(output_root),
            "bands": list(bands),
            "tract": args.tract,
            "patch": patch,
            "num_tiles": len(tile_names),
            "reuse_existing_preprocessed": True,
            "zscale_root": str(_expand(args.zscale_root)) if args.zscale_root is not None else None,
            "zscale_written": zscale_written,
            "zscale_skipped": zscale_skipped,
            "args": _jsonable_args(args),
        }
    )

    if not args.dry_run:
        (output_root / "cutout_paths.json").write_text(json.dumps(cutout_paths, indent=2), encoding="utf-8")
        manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if args.fast_root is not None:
            fast_root = _variant_patch_output_root(_expand(args.fast_root), variant, args.tract, patch)
            fast_root.mkdir(parents=True, exist_ok=True)
            for filename in ("manifest.json", "tiles.csv", "cutout_paths.json"):
                src = output_root / filename
                if src.exists():
                    shutil.copy2(src, fast_root / filename)
            if (output_root / "effective_count_paths.json").exists():
                shutil.copy2(output_root / "effective_count_paths.json", fast_root / "effective_count_paths.json")
            for dirname in ("targets", "tile_metadata", "band_targets", "band_tile_metadata"):
                src_dir = output_root / dirname
                dst_dir = fast_root / dirname
                if src_dir.exists() and src_dir.resolve() != dst_dir.resolve():
                    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    print(
        f"patch {patch}: reused {len(tile_names)} existing {variant} tiles; zscale_written={zscale_written}, "
        f"zscale_skipped={zscale_skipped}; output={output_root}",
        flush=True,
    )
    return metadata


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


def _compact_exception_message(exc: BaseException, *, limit: int = 2000) -> str:
    if isinstance(exc, shutil.Error) and exc.args:
        payload = exc.args[0]
        if isinstance(payload, list):
            if all(isinstance(item, str) and len(item) == 1 for item in payload):
                message = "".join(payload)
            else:
                parts = []
                for item in payload[:5]:
                    if isinstance(item, tuple) and len(item) >= 3:
                        parts.append(f"{item[0]} -> {item[1]}: {item[2]}")
                    else:
                        parts.append(str(item))
                suffix = f"; ... {len(payload) - 5} more" if len(payload) > 5 else ""
                message = "; ".join(parts) + suffix
        else:
            message = str(payload)
    else:
        message = str(exc)

    if message.startswith("['") and "'," in message:
        try:
            parsed = ast.literal_eval(message)
        except Exception:
            parsed = None
        if isinstance(parsed, list) and all(isinstance(item, str) and len(item) == 1 for item in parsed):
            message = "".join(parsed)
    if len(message) > limit:
        return message[:limit] + f"... [truncated {len(message) - limit} chars]"
    return message


def _patch_failure_row(patch: str, catalog_path: Path, output_root: Path, exc: BaseException) -> Dict[str, str]:
    error = _compact_exception_message(exc)
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    if len(tb) > 20000:
        tb = tb[:20000] + f"... [traceback truncated {len(tb) - 20000} chars]"
    return {
        "patch": patch,
        "catalog": str(catalog_path),
        "output_root": str(output_root),
        "error_type": type(exc).__name__,
        "error": error,
        "traceback": tb,
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
    denoised_fits_root = _expand(args.denoised_fits_root) if args.denoised_fits_root is not None else None
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
    show_progress = not bool(getattr(args, "no_progress", False))
    _configure_worker_threads(int(getattr(args, "worker_threads", 1)))
    if num_workers == 1:
        for patch, catalog_path, patch_output_root in _progress_iter(
            tasks,
            total=len(tasks),
            desc="preprocess patches",
            unit="patch",
            enabled=show_progress,
        ):
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
                print(f"FAILED patch {patch}: {_compact_exception_message(exc, limit=500)}", flush=True)
    else:
        if not args.dry_run:
            data_root.mkdir(parents=True, exist_ok=True)
        print(f"processing {len(tasks)} patch(es) with {num_workers} worker(s)")
        task_by_patch = {patch: (catalog_path, patch_output_root) for patch, catalog_path, patch_output_root in tasks}
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_configure_worker_threads,
            initargs=(int(getattr(args, "worker_threads", 1)),),
        ) as executor:
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
            for future in _progress_iter(
                as_completed(future_to_patch),
                total=len(future_to_patch),
                desc="preprocess patches",
                unit="patch",
                enabled=show_progress,
            ):
                patch = future_to_patch[future]
                try:
                    summaries_by_patch[patch] = future.result()
                except Exception as exc:
                    catalog_path, patch_output_root = task_by_patch[patch]
                    failed_patch_rows.append(_patch_failure_row(patch, catalog_path, patch_output_root, exc))
                    print(f"FAILED patch {patch}: {_compact_exception_message(exc, limit=500)}", flush=True)
                completed += 1
                print(f"completed patch {patch} ({completed}/{len(tasks)})", flush=True)

    summaries = [summaries_by_patch[patch] for patch in patches if patch in summaries_by_patch]

    variant_summaries: List[Dict[str, object]] = []
    variant_failed_rows: List[Dict[str, str]] = []
    if denoised_fits_root is not None:
        variants = tuple(str(item).strip() for item in args.image_variants if str(item).strip())
        variant_tasks: List[Tuple[str, str, Optional[Path], Path]] = []
        skipped_variant_patches: List[str] = []
        rebuild_variants = bool(getattr(args, "rebuild_image_variants", False))
        variant_worker_fn = (
            _sync_existing_variant_patch
            if args.reuse_existing_preprocessed and not rebuild_variants
            else _preprocess_image_variant_patch
        )
        for patch in patches:
            if patch not in summaries_by_patch:
                continue
            coadd_patch_root = _patch_output_root(data_root, args.tract, patch)
            denoised_patch_dir = _find_denoised_patch_dir(denoised_fits_root, patch)
            for variant in variants:
                variant_output_root = _variant_patch_output_root(data_root, variant, args.tract, patch)
                if denoised_patch_dir is None and not (args.reuse_existing_preprocessed and variant_output_root.exists()):
                    skipped_variant_patches.append(f"{variant}:{patch}")
                    print(
                        f"WARNING: skipping image variant {variant}:{patch}; "
                        f"patch directory not found under {denoised_fits_root}",
                        flush=True,
                    )
                    continue
                variant_tasks.append(
                    (
                        patch,
                        variant,
                        denoised_patch_dir,
                        variant_output_root,
                    )
                )

        variant_workers = _worker_count(
            int(getattr(args, "variant_num_workers", 0)),
            len(variant_tasks),
        )
        if variant_tasks:
            print(
                f"processing {len(variant_tasks)} image-variant patch task(s) "
                f"from {denoised_fits_root}: variants={list(variants)} with {variant_workers} worker(s)",
                flush=True,
            )
        if variant_workers == 1:
            for patch, variant, denoised_patch_dir, variant_output_root in _progress_iter(
                variant_tasks,
                total=len(variant_tasks),
                desc="image variants",
                unit="task",
                enabled=show_progress,
            ):
                try:
                    variant_summaries.append(
                        variant_worker_fn(
                            args,
                            coadd_patch_root=_patch_output_root(data_root, args.tract, patch),
                            denoised_patch_dir=denoised_patch_dir,
                            variant=variant,
                            output_root=variant_output_root,
                            bands=bands,
                            patch=patch,
                        )
                    )
                except Exception as exc:
                    variant_failed_rows.append(
                        _patch_failure_row(
                            f"{variant}:{patch}",
                            _catalog_path_for_patch(args, catalog_root, patch, len(patches)),
                            variant_output_root,
                            exc,
                        )
                    )
                    print(f"FAILED patch {variant}:{patch}: {_compact_exception_message(exc, limit=500)}", flush=True)
        elif variant_tasks:
            task_by_key = {
                f"{variant}:{patch}": (patch, variant, variant_output_root)
                for patch, variant, _denoised_patch_dir, variant_output_root in variant_tasks
            }
            with ProcessPoolExecutor(
                max_workers=variant_workers,
                initializer=_configure_worker_threads,
                initargs=(int(getattr(args, "worker_threads", 1)),),
            ) as executor:
                future_to_key = {
                    executor.submit(
                        variant_worker_fn,
                        args,
                        coadd_patch_root=_patch_output_root(data_root, args.tract, patch),
                        denoised_patch_dir=denoised_patch_dir,
                        variant=variant,
                        output_root=variant_output_root,
                        bands=bands,
                        patch=patch,
                    ): f"{variant}:{patch}"
                    for patch, variant, denoised_patch_dir, variant_output_root in variant_tasks
                }
                completed = 0
                for future in _progress_iter(
                    as_completed(future_to_key),
                    total=len(future_to_key),
                    desc="image variants",
                    unit="task",
                    enabled=show_progress,
                ):
                    key = future_to_key[future]
                    patch, variant, variant_output_root = task_by_key[key]
                    try:
                        variant_summaries.append(future.result())
                    except Exception as exc:
                        variant_failed_rows.append(
                            _patch_failure_row(
                                key,
                                _catalog_path_for_patch(args, catalog_root, patch, len(patches)),
                                variant_output_root,
                                exc,
                            )
                        )
                        print(f"FAILED patch {key}: {_compact_exception_message(exc, limit=500)}", flush=True)
                    completed += 1
                    print(f"completed image variant {key} ({completed}/{len(variant_tasks)})", flush=True)
        if skipped_variant_patches:
            skipped_names = " ".join(skipped_variant_patches)
            print(
                f"SKIPPED_IMAGE_VARIANT_PATCHES {skipped_names}",
                flush=True,
            )
        summaries.extend(variant_summaries)
        failed_patch_rows.extend(variant_failed_rows)

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
    pu_result: Optional[Dict[str, object]] = None
    center_only = Table()
    ignore_sources = Table()
    strict_center_only_sources = Table()
    if args.label_mode == "pu":
        filtered, center_only, ignore_sources, pu_all, pu_result = _classify_pu_catalog(
            table,
            args,
            band=args.catalog_band,
            patch=patch,
        )
        filtered, center_only, strict_center_only_sources = _move_bright_clean_to_center_only(
            filtered,
            center_only,
            args,
            band=args.catalog_band,
        )
        rejected_parts = [part for part in (center_only, ignore_sources) if len(part)]
        rejected = vstack(rejected_parts) if len(rejected_parts) >= 2 else (rejected_parts[0] if rejected_parts else Table())
    else:
        filtered, rejected = _filter_catalog(table, args)
        if args.target_shape_source != args.shape_source:
            filtered = add_ellipse_columns(filtered, shape_source=args.target_shape_source)
            rejected = add_ellipse_columns(rejected, shape_source=args.target_shape_source)

    sources_dir = output_root / "sources"
    if not args.dry_run:
        write_table_pair(filtered, sources_dir / "sources_filtered.fits", sources_dir / "sources_filtered.csv")
        write_table_pair(
            rejected,
            sources_dir / "sources_rejected.fits",
            None if args.label_mode == "pu" else sources_dir / "sources_rejected.csv",
        )
        if args.label_mode == "pu":
            write_table_pair(filtered, sources_dir / "sources_pu_clean.fits", None)
            write_table_pair(center_only, sources_dir / "sources_pu_center_only.fits", None)
            write_table_pair(ignore_sources, sources_dir / "sources_pu_ignore.fits", None)
            write_table_pair(strict_center_only_sources, sources_dir / "sources_pu_strict_center_only.fits", None)
            write_table_pair(pu_all, sources_dir / "sources_pu_all.fits", None)

    band_catalog_warnings: List[Dict[str, str]] = []
    band_target_sources: Dict[str, Tuple[Table, Table, Table, Table]] = {}
    if not args.dry_run:
        band_catalog_root = _expand(args.band_catalog_root) if args.band_catalog_root is not None else _expand(args.catalog_root)
        for band in bands:
            band_catalog_path = _band_catalog_path(band_catalog_root, band, args.tract, patch)
            try:
                band_table = _read_table(band_catalog_path, hdu=args.catalog_hdu, role="band-reference", patch=patch, band=band)
                if args.label_mode == "pu":
                    band_filtered, band_center_only, band_ignore, band_pu_all, _band_pu_result = _classify_pu_catalog(
                        band_table,
                        args,
                        band=band,
                        patch=patch,
                    )
                    band_strict_center_only = Table()
                    band_filtered, band_center_only, band_strict_center_only = _move_bright_clean_to_center_only(
                        band_filtered,
                        band_center_only,
                        args,
                        band=band,
                    )
                    band_rejected_parts = [part for part in (band_center_only, band_ignore) if len(part)]
                    band_rejected = (
                        vstack(band_rejected_parts)
                        if len(band_rejected_parts) >= 2
                        else (band_rejected_parts[0] if band_rejected_parts else Table())
                    )
                else:
                    band_filtered, band_rejected = _filter_catalog(band_table, args)
                    if args.target_shape_source != args.shape_source:
                        band_filtered = add_ellipse_columns(band_filtered, shape_source=args.target_shape_source)
                        band_rejected = add_ellipse_columns(band_rejected, shape_source=args.target_shape_source)
                    band_center_only = Table()
                    band_ignore = Table()
                    band_strict_center_only = Table()
                    band_pu_all = Table()
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
                band_center_only = center_only
                band_ignore = ignore_sources
                band_strict_center_only = strict_center_only_sources
                band_pu_all = pu_all if args.label_mode == "pu" else Table()
            band_target_sources[band] = (band_filtered, band_center_only, band_ignore, band_strict_center_only)
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
            write_ids_metadata(band_rejected, output_root / "band_rejected_ids" / f"{band}.npz")
            if args.label_mode == "pu":
                write_table_pair(
                    band_center_only,
                    output_root / "band_reference_center_only" / band / f"meas-{band}-{args.tract}-{patch}.fits",
                    None,
                )
                write_table_pair(
                    band_ignore,
                    output_root / "band_reference_ignore" / band / f"meas-{band}-{args.tract}-{patch}.fits",
                    None,
                )
                write_table_pair(
                    band_strict_center_only,
                    output_root / "band_reference_strict_center_only" / band / f"meas-{band}-{args.tract}-{patch}.fits",
                    None,
                )
                write_table_pair(
                    band_pu_all,
                    output_root / "band_reference_pu_all" / band / f"meas-{band}-{args.tract}-{patch}.fits",
                    None,
                )

    manifest_rows = []
    cutout_paths: Dict[str, Dict[str, str]] = {}
    lsst_background_masks: Dict[str, np.ndarray] = {}
    if args.label_mode == "pu":
        for band in bands:
            if bool(getattr(args, "lsst_background_detect_cutouts", False)):
                det_path = _band_det_path(coadd_root, band, args.tract, patch)
                if det_path is None:
                    continue
                try:
                    lsst_background_masks[band] = _read_det_background_mask(det_path, image_shape_yx, origin_xy=parent_origin)
                except Exception as exc:
                    print(
                        f"WARNING: failed to read LSST det footprints for patch={patch}, band={band}: {det_path}; {exc}",
                        flush=True,
                    )
                continue
            det_path = _resolve_lsst_background_det_path(
                args=args,
                output_root=output_root,
                coadd_root=coadd_root,
                band=band,
                tract=args.tract,
                patch=patch,
            )
            if det_path is None:
                print(
                    f"WARNING: no LSST det footprint file found for patch={patch}, band={band}; "
                    "PU background mask will be empty and unlabelled pixels will be ignored.",
                    flush=True,
                )
                continue
            try:
                lsst_background_masks[band] = _read_det_background_mask(det_path, image_shape_yx, origin_xy=parent_origin)
            except Exception as exc:
                print(f"WARNING: failed to read LSST det footprints for patch={patch}, band={band}: {det_path}; {exc}", flush=True)

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
        if args.label_mode == "pu":
            tile_center_only = crop_catalog_for_tile(
                center_only,
                spec,
                x_col=args.x_col,
                y_col=args.y_col,
                margin=0.0,
            )
            tile_ignore = crop_catalog_for_tile(
                ignore_sources,
                spec,
                x_col=args.x_col,
                y_col=args.y_col,
                margin=0.0,
            )
            tile_strict_center_only = crop_catalog_for_tile(
                strict_center_only_sources,
                spec,
                x_col=args.x_col,
                y_col=args.y_col,
                margin=0.0,
            )
            center_only_mask_sources = crop_catalog_for_tile(
                center_only,
                spec,
                x_col=args.x_col,
                y_col=args.y_col,
                margin=args.mask_margin,
            )
            ignore_mask_sources = crop_catalog_for_tile(
                ignore_sources,
                spec,
                x_col=args.x_col,
                y_col=args.y_col,
                margin=args.mask_margin,
            )
            strict_center_only_mask_sources = crop_catalog_for_tile(
                strict_center_only_sources,
                spec,
                x_col=args.x_col,
                y_col=args.y_col,
                margin=args.mask_margin,
            )
        else:
            tile_center_only = Table()
            tile_ignore = Table()
            tile_strict_center_only = Table()
            center_only_mask_sources = Table()
            ignore_mask_sources = Table()
            strict_center_only_mask_sources = Table()

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

        tile_lsst_background_masks: Dict[str, np.ndarray] = {}
        if args.label_mode == "pu" and bool(getattr(args, "lsst_background_detect_cutouts", False)):
            policy = str(getattr(args, "lsst_background_policy", "run-if-missing"))
            for band in bands:
                if band in lsst_background_masks or policy in {"none", "existing"}:
                    continue
                cutout_path = Path(band_paths[band])
                if args.dry_run:
                    continue
                if not cutout_path.exists():
                    raise FileNotFoundError(
                        f"--lsst-background-detect-cutouts requires an existing cutout for {patch}/{spec.name}/{band}: "
                        f"{cutout_path}"
                    )
                tile_patch_key = f"{patch}_{spec.name}"
                detected_calexp_path = (
                    cutout_path.with_name(f"lsst-detect-{cutout_path.name}")
                    if bool(getattr(args, "use_lsst_detection_calexp_cutouts", False))
                    else None
                )
                det_path = _run_lsst_detection_background(
                    args=args,
                    output_root=output_root,
                    coadd_path=cutout_path,
                    band=band,
                    tract=args.tract,
                    patch=tile_patch_key,
                    output_calexp_path=detected_calexp_path,
                )
                tile_lsst_background_masks[band] = _read_det_background_mask(
                    det_path,
                    (spec.size, spec.size),
                    origin_xy=(spec.x0, spec.y0),
                )
                if detected_calexp_path is not None and detected_calexp_path.exists():
                    band_paths[band] = str(detected_calexp_path)

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

        if args.label_mode == "pu":
            targets = make_pu_dense_targets(
                mask_sources,
                center_only_mask_sources,
                ignore_mask_sources,
                spec,
                x_col=args.x_col,
                y_col=args.y_col,
                ellipse_sigma=args.ellipse_sigma,
                confidence_levels=args.confidence_levels,
                core_radius=args.core_radius,
                center_only_weight=args.pu_center_only_weight,
                lsst_background_mask=_crop_full_mask_for_tile(
                    lsst_background_masks.get(first_band), spec, parent_origin
                )
                if first_band in lsst_background_masks
                else tile_lsst_background_masks.get(first_band),
                strict_center_only_sources=strict_center_only_mask_sources,
                strict_center_only_ellipse_sigma=args.pu_strict_bright_center_only_ellipse_sigma,
            )
        else:
            targets = make_dense_targets(
                mask_sources,
                spec,
                x_col=args.x_col,
                y_col=args.y_col,
                ellipse_sigma=args.ellipse_sigma,
                confidence_levels=args.confidence_levels,
                core_radius=args.core_radius,
            )

        band_targets: Dict[str, Dict[str, np.ndarray]] = {}
        band_tile_catalogs: Dict[str, Table] = {}
        for band, (band_filtered, band_center_only, band_ignore, band_strict_center_only) in band_target_sources.items():
            band_tile_catalogs[band] = crop_catalog_for_tile(
                band_filtered,
                spec,
                x_col=args.x_col,
                y_col=args.y_col,
                margin=0.0,
            )
            if args.skip_band_targets:
                continue
            band_mask_sources = crop_catalog_for_tile(
                band_filtered,
                spec,
                x_col=args.x_col,
                y_col=args.y_col,
                margin=args.mask_margin,
            )
            if args.label_mode == "pu":
                band_center_mask_sources = crop_catalog_for_tile(
                    band_center_only,
                    spec,
                    x_col=args.x_col,
                    y_col=args.y_col,
                    margin=args.mask_margin,
                )
                band_ignore_mask_sources = crop_catalog_for_tile(
                    band_ignore,
                    spec,
                    x_col=args.x_col,
                    y_col=args.y_col,
                    margin=args.mask_margin,
                )
                band_strict_center_only_mask_sources = crop_catalog_for_tile(
                    band_strict_center_only,
                    spec,
                    x_col=args.x_col,
                    y_col=args.y_col,
                    margin=args.mask_margin,
                )
                band_targets[band] = make_pu_dense_targets(
                    band_mask_sources,
                    band_center_mask_sources,
                    band_ignore_mask_sources,
                    spec,
                    x_col=args.x_col,
                    y_col=args.y_col,
                    ellipse_sigma=args.ellipse_sigma,
                    confidence_levels=args.confidence_levels,
                    core_radius=args.core_radius,
                    center_only_weight=args.pu_center_only_weight,
                    lsst_background_mask=_crop_full_mask_for_tile(
                        lsst_background_masks.get(band), spec, parent_origin
                    )
                    if band in lsst_background_masks
                    else tile_lsst_background_masks.get(band),
                    strict_center_only_sources=band_strict_center_only_mask_sources,
                    strict_center_only_ellipse_sigma=args.pu_strict_bright_center_only_ellipse_sigma,
                )
            else:
                band_targets[band] = make_dense_targets(
                    band_mask_sources,
                    spec,
                    x_col=args.x_col,
                    y_col=args.y_col,
                    ellipse_sigma=args.ellipse_sigma,
                    confidence_levels=args.confidence_levels,
                    core_radius=args.core_radius,
                )

        ref_path = output_root / "reference_catalogs" / f"{spec.name}_meas.fits"
        ref_csv = output_root / "reference_catalogs_csv" / f"{spec.name}_meas.csv"
        center_ref_path = output_root / "center_only_catalogs" / f"{spec.name}_meas.fits"
        ignore_ref_path = output_root / "ignore_catalogs" / f"{spec.name}_meas.fits"
        strict_center_only_ref_path = output_root / "strict_center_only_catalogs" / f"{spec.name}_meas.fits"
        target_path = output_root / "targets" / f"{spec.name}.npz"
        target_fits_prefix = output_root / "target_fits" / spec.name if args.write_target_fits else None
        if not args.dry_run:
            write_table_pair(tile_catalog, ref_path, ref_csv)
            if args.label_mode == "pu":
                write_table_pair(tile_center_only, center_ref_path, None)
                write_table_pair(tile_ignore, ignore_ref_path, None)
                write_table_pair(tile_strict_center_only, strict_center_only_ref_path, None)
            write_targets(targets, target_path, target_fits_prefix)
            write_catalog_metadata(tile_catalog, output_root / "tile_metadata" / f"{spec.name}.npz")
            for band, band_tile_catalog in band_tile_catalogs.items():
                write_catalog_metadata(
                    band_tile_catalog,
                    output_root / "band_tile_metadata" / band / f"{spec.name}.npz",
                )
            for band, band_target in band_targets.items():
                write_targets(band_target, output_root / "band_targets" / band / f"{spec.name}.npz", None)

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
                "n_center_only_center_in_tile": len(tile_center_only),
                "n_center_only_for_mask_with_margin": len(center_only_mask_sources),
                "n_ignore_center_in_tile": len(tile_ignore),
                "n_ignore_for_mask_with_margin": len(ignore_mask_sources),
                "n_strict_center_only_center_in_tile": len(tile_strict_center_only),
                "n_strict_center_only_for_mask_with_margin": len(strict_center_only_mask_sources),
                "seg_foreground_pixels": int((targets["seg"] > 0).sum()),
                "overlap_pixels": int((targets["overlap_count"] >= 2).sum()),
                "source_union_pixels": int(targets["source_union_mask"].sum()) if "source_union_mask" in targets else None,
                "strict_center_only_pixels": int(targets["strict_center_only_mask"].sum()) if "strict_center_only_mask" in targets else None,
                "background_pixels": int(targets["background_mask"].sum()) if "background_mask" in targets else None,
                "band_target_paths": {
                    band: str(output_root / "band_targets" / band / f"{spec.name}.npz") for band in sorted(band_targets)
                },
                "tile_metadata_path": str(output_root / "tile_metadata" / f"{spec.name}.npz"),
                "band_tile_metadata_paths": {
                    band: str(output_root / "band_tile_metadata" / band / f"{spec.name}.npz")
                    for band in sorted(band_tile_catalogs)
                },
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
        "label_mode": args.label_mode,
        "target_shape_source": args.target_shape_source,
        "num_pu_clean_sources": len(filtered) if args.label_mode == "pu" else None,
        "num_pu_center_only_sources": len(center_only) if args.label_mode == "pu" else None,
        "num_pu_ignore_sources": len(ignore_sources) if args.label_mode == "pu" else None,
        "num_pu_dropped_large_ellipse_sources": int(np.count_nonzero(pu_result["dropped_large_ellipse"]))
        if pu_result is not None
        else None,
        "num_pu_dropped_by_a_sources": int(np.count_nonzero(pu_result["dropped_by_a"])) if pu_result is not None else None,
        "num_pu_dropped_invalid_kron_sources": int(np.count_nonzero(pu_result["dropped_invalid_kron"]))
        if pu_result is not None
        else None,
        "pu_drop_ellipse_area_min": args.pu_drop_ellipse_area_min if args.label_mode == "pu" else None,
        "pu_ambiguous_overlap_pairs": int(pu_result["overlap_pair_count"]) if pu_result is not None else None,
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
    parser.add_argument("--shape-source", choices=("sdss", "hsm", "kron", "circular_kron"), default="sdss")
    parser.add_argument(
        "--target-shape-source",
        choices=("sdss", "hsm", "kron", "circular_kron"),
        default="kron",
        help=(
            "Ellipse source used for dense target masks. PU mode defaults to Kron radius plus SDSS axis ratio/angle. "
            "--shape-source is kept for legacy area filtering."
        ),
    )
    parser.add_argument(
        "--label-mode",
        choices=("legacy", "pu"),
        default="legacy",
        help="legacy uses the old filtered/rejected catalog split; pu writes clean/center_only/ignore masks and catalogs.",
    )
    parser.add_argument(
        "--skip-band-targets",
        action="store_true",
        help="Do not write per-band dense target npz files. By default they are written for faster training.",
    )
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
    parser.add_argument(
        "--ellipse-sigma",
        type=float,
        default=1.0,
        help="Scale applied to target ellipses. Default 1.0 matches Kron/refit aperture radii; use 3.0 for legacy SDSS-moment masks.",
    )
    parser.add_argument("--min-ellipse-axis", type=float, default=1.5)
    parser.add_argument("--mask-margin", type=float, default=64.0)
    parser.add_argument("--confidence-levels", type=int, default=5)
    parser.add_argument("--core-radius", type=int, default=2)
    parser.add_argument("--pu-center-only-weight", type=float, default=0.25)
    parser.add_argument(
        "--noncoadd-snr-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For noisy/denoised image variants, build variant-local targets by moving clean coadd labels "
            "with raw-image SNR below the thresholds to ignore/center_only during preprocessing."
        ),
    )
    parser.add_argument("--noncoadd-snr-ignore-thresh", type=float, default=2.0)
    parser.add_argument("--noncoadd-snr-center-only-thresh", type=float, default=3.0)
    parser.add_argument("--noncoadd-snr-ap-radius", type=float, default=6.0)
    parser.add_argument("--noncoadd-snr-annulus-r-in", type=float, default=10.0)
    parser.add_argument("--noncoadd-snr-annulus-r-out", type=float, default=15.0)
    parser.add_argument(
        "--noncoadd-snr-annulus-exclude-radius",
        type=float,
        default=6.0,
        help="Exclude pixels within this radius of any clean source center from non-coadd SNR annuli. Use <=0 to disable.",
    )
    parser.add_argument("--pu-a-flags", nargs="*", default=list(DEFAULT_PU_A_FLAGS))
    parser.add_argument("--pu-b-flags", nargs="*", default=list(DEFAULT_PU_B_FLAGS))
    parser.add_argument("--pu-a-mode", choices=("any", "all"), default="any")
    parser.add_argument("--pu-b-mode", choices=("any", "all"), default="any")
    parser.add_argument("--pu-strict-flags", action="store_true")
    parser.add_argument("--pu-mag-column", default="ext_photometryKron_KronFlux_instFlux")
    parser.add_argument("--pu-input-zeropoint", type=float, default=27.0)
    parser.add_argument(
        "--pu-kron-refit-csv",
        type=Path,
        default=None,
        help=(
            "Optional batch-heavyfp-kron-refit CSV. The path may contain {tract}, {patch}, and {band}; "
            "matched proxy Kron radii are preferred when target-shape-source is kron/circular_kron."
        ),
    )
    parser.add_argument("--pu-kron-refit-radius-column", default="proxy_nan0_flux_aperture_radius")
    parser.add_argument("--pu-kron-refit-good-column", default="proxy_nan0_good")
    parser.add_argument(
        "--pu-require-kron-refit-match",
        action="store_true",
        default=True,
        help=(
            "In PU mode, if --pu-kron-refit-csv is provided, only sources with a good matched refit radius "
            "can become clean/center_only/ignore source ellipses."
        ),
    )
    parser.add_argument("--pixel-scale-arcsec", type=float, default=0.168)
    parser.add_argument("--pu-a-area-max", type=float, default=10000.0)
    parser.add_argument("--pu-a-faint-area-max", type=float, default=900.0)
    parser.add_argument("--pu-a-faint-mag-min", type=float, default=28.0)
    parser.add_argument("--pu-b-mag-min", type=float, default=18.0)
    parser.add_argument("--pu-b-mag-max", type=float, default=30.0)
    parser.add_argument(
        "--pu-ap2-kron-abs-max",
        type=float,
        default=1.0,
        help=(
            "PU B filter for direct training catalogs: require abs(ap2_mag-kron_mag) to be below "
            "this threshold. Use a negative value to disable."
        ),
    )
    parser.add_argument("--pu-ap2-flux-column", default="base_CircularApertureFlux_6_0_instFlux")
    parser.add_argument("--pu-ap2-kron-flux-column", default="ext_photometryKron_KronFlux_instFlux")
    parser.add_argument(
        "--pu-remeasure-ap2-kron-outliers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For sources rejected by the catalog ap2-vs-Kron B filter, remeasure Kron flux with "
            "heavy footprints plus refit Kron apertures and promote reliable candidates to clean/center_only."
        ),
    )
    parser.add_argument(
        "--pu-remeasure-ap2-kron-threshold",
        type=float,
        default=np.nan,
        help="Old abs(ap2_mag-kron_mag) threshold selecting rows for remeasurement. Default: --pu-ap2-kron-abs-max.",
    )
    parser.add_argument("--pu-remeasure-clean-abs-max", type=float, default=1.0)
    parser.add_argument("--pu-remeasure-center-only-abs-max", type=float, default=1.5)
    parser.add_argument("--pu-remeasure-small-footprint-fill-threshold", type=float, default=0.2)
    parser.add_argument("--pu-remeasure-ignore-area-max", type=float, default=10000.0)
    parser.add_argument("--pu-remeasure-faint-mag-min", type=float, default=28.0)
    parser.add_argument("--pu-remeasure-faint-area-max", type=float, default=900.0)
    parser.add_argument("--pu-remeasure-axis-ratio-max", type=float, default=5.0)
    parser.add_argument("--pu-remeasure-containment-threshold", type=float, default=0.80)
    parser.add_argument(
        "--pu-use-band-limit-b-filter",
        action="store_true",
        help="Use per-band limiting magnitudes for the PU B filter instead of --pu-b-mag-min/max.",
    )
    parser.add_argument(
        "--pu-band-limit-mags",
        nargs="*",
        default=None,
        help="Per-band limits such as HSC-G=27.4 HSC-R=27.1 HSC-I=26.9 HSC-Z=26.3 HSC-Y=25.3.",
    )
    parser.add_argument("--pu-band-limit-b-min-offset", type=float, default=-5.0)
    parser.add_argument("--pu-band-limit-b-max-offset", type=float, default=0.0)
    parser.add_argument(
        "--pu-enable-strict-bright-center-only",
        action="store_true",
        help=(
            "Move clean sources brighter than the strict bright threshold to center_only. "
            "They keep low-weight center supervision instead of becoming strict ignore regions."
        ),
    )
    parser.add_argument(
        "--pu-strict-bright-center-only-mag-threshold",
        type=float,
        default=None,
        help=(
            "Optional global magnitude threshold for strict bright-source center_only labels. "
            "When unset, per-band HSC saturation magnitudes are used."
        ),
    )
    parser.add_argument(
        "--pu-strict-ignore-mag-threshold",
        dest="pu_strict_ignore_mag_threshold",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pu-strict-bright-center-only-saturation-mags",
        nargs="*",
        default=None,
        help="Per-band strict center_only saturation magnitudes, e.g. HSC-G=18.0 HSC-R=18.2 HSC-I=18.6 HSC-Z=17.7 HSC-Y=17.4.",
    )
    parser.add_argument(
        "--pu-strict-ignore-saturation-mags",
        dest="pu_strict_ignore_saturation_mags",
        nargs="*",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pu-strict-bright-center-only-radius-column",
        default="proxy_nan0_flux_aperture_radius",
        help="Radius column from --pu-kron-refit-csv used for strict bright-source center_only apertures.",
    )
    parser.add_argument(
        "--pu-strict-ignore-radius-column",
        dest="pu_strict_bright_center_only_radius_column",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pu-strict-bright-center-only-ellipse-sigma",
        type=float,
        default=1.0,
        help="Scale applied when rasterizing strict bright-source center_only apertures.",
    )
    parser.add_argument(
        "--pu-strict-ignore-ellipse-sigma",
        dest="pu_strict_bright_center_only_ellipse_sigma",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--pu-b-close-center-arcsec", type=float, default=0.5)
    parser.add_argument("--pu-overlap-iou-threshold", type=float, default=0.33)
    parser.add_argument("--pu-b-ellipse-area-max", type=float, default=None)
    parser.add_argument("--pu-b-footprint-area-max", type=float, default=None)
    parser.add_argument(
        "--pu-b-axis-ratio-max",
        type=float,
        default=5.0,
        help="PU B filter: move sources with max(axis)/min(axis) above this value to ordinary ignore. Use <=0 to disable.",
    )
    parser.add_argument("--pu-b-kron-radius-lt-sdss-major-ratio", type=float, default=0.5)
    parser.add_argument(
        "--pu-drop-ellipse-area-min",
        type=float,
        default=40000.0,
        help=(
            "In PU mode, discard sources with target Kron ellipse area above this threshold before assigning "
            "clean/center_only/ignore labels. Use a negative value to disable."
        ),
    )
    parser.add_argument("--pu-ambiguous-area-max", type=float, default=40000.0)
    parser.add_argument("--pu-neighbor-radius", type=float, default=80.0)
    parser.add_argument("--pu-center-distance-factor", type=float, default=0.75)
    parser.add_argument("--pu-containment-threshold", type=float, default=0.80)
    parser.add_argument("--pu-mutual-overlap-threshold", type=float, default=0.35)
    parser.add_argument("--pu-overlap-sample-grid", type=int, default=9)
    parser.add_argument("--pu-ambiguous-mark", choices=("both", "smaller"), default="both")
    parser.add_argument(
        "--pu-keep-all-ab-clean",
        action="store_true",
        default=True,
        help="Do not make center_only labels from overlapping A+B sources; every A+B source is clean.",
    )
    parser.add_argument(
        "--lsst-background-policy",
        choices=("run-if-missing", "existing", "none"),
        default="run-if-missing",
        help=(
            "PU background source. Use existing official det footprints when present; with run-if-missing, "
            "run LSST default DetectCoaddSourcesTask on the input coadd when det footprints are absent. "
            "existing preserves the old no-run behavior, and none disables LSST background masks."
        ),
    )
    parser.add_argument(
        "--lsst-background-cache-root",
        type=Path,
        default=None,
        help="Cache root for det catalogs generated by --lsst-background-policy run-if-missing. Default: <patch output>/lsst_detection_background.",
    )
    parser.add_argument(
        "--lsst-detect-python",
        default="",
        help=(
            "Python command used for lsst_detect_background.py. Defaults to the current interpreter; set this to a "
            "Python executable or wrapper with lsst_distrib loaded when preprocessing runs outside an LSST environment."
        ),
    )
    parser.add_argument(
        "--overwrite-lsst-background",
        action="store_true",
        help="Regenerate cached LSST default detection catalogs used for PU background masks.",
    )
    parser.add_argument(
        "--write-lsst-background-products",
        action="store_true",
        help="Also write post-detection exposure and LSST BackgroundList products when running the LSST background fallback.",
    )
    parser.add_argument(
        "--lsst-background-detect-cutouts",
        action="store_true",
        help=(
            "When official det footprints are absent, run LSST default detection independently on each 512x512 cutout "
            "instead of on the full patch. This is optional because it changes preprocessing I/O and requires an LSST stack."
        ),
    )
    parser.add_argument(
        "--use-lsst-detection-calexp-cutouts",
        action="store_true",
        help=(
            "With --lsst-background-detect-cutouts, use the LSST detection outputExposure FITS as the cutout path "
            "for zscale/training metadata. This keeps no-background inputs aligned with the coadd training convention."
        ),
    )
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
        "--rebuild-image-variants",
        action="store_true",
        help=(
            "With --reuse-existing-preprocessed, reuse the coadd patch tree but rebuild denoised/noisy image-variant "
            "cutout metadata and targets. Use this to refresh variant PU targets after changing noncoadd SNR or "
            "variant LSST background masks."
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
        "--denoised-fits-root",
        type=Path,
        default=None,
        help=(
            "Optional root containing denoised/noisy full-patch FITS, e.g. "
            "<root>/patch_8_4/group_00/HSC-I/denoised.fits. When set, existing coadd "
            "preprocessed labels are mirrored into <output-root>/<variant>/<tract>/<patch>."
        ),
    )
    parser.add_argument(
        "--variant-lsst-background-root",
        type=Path,
        default=None,
        help=(
            "Optional cache root containing variant/group LSST detection background masks generated by "
            "lsst_pipeline/batch_detect_background.py. Layout: "
            "<root>/<variant>/<tract>/<patch>/<group>/<band>/background_mask.npz. "
            "When present, denoised/noisy PU targets use these masks before falling back to coadd target backgrounds."
        ),
    )
    parser.add_argument(
        "--image-variants",
        nargs="+",
        default=("denoised", "noisy"),
        help="Image variant FITS basenames to crop from --denoised-fits-root. Default: denoised noisy.",
    )
    parser.add_argument(
        "--image-variant-groups",
        nargs="+",
        default=(),
        help="Optional subset of denoised/noisy group directories to process, e.g. group_01 or 01.",
    )
    parser.add_argument(
        "--image-variant-tiles",
        nargs="+",
        default=(),
        help="Optional subset of base coadd tile names to process for denoised/noisy variants.",
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
    parser.add_argument(
        "--variant-num-workers",
        type=int,
        default=0,
        help=(
            "Number of worker processes for --denoised-fits-root image variants. "
            "Use 0 to auto-select up to the number of variant tasks/CPU cores."
        ),
    )
    parser.add_argument(
        "--worker-threads",
        type=int,
        default=1,
        help=(
            "Intra-process CPU threads used by torch/OpenMP/BLAS inside each preprocessing worker. "
            "Default 1 avoids oversubscription when many patch workers are active."
        ),
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.no_compare_origin:
        args.compare_origin = None
    if args.use_lsst_detection_calexp_cutouts and not args.lsst_background_detect_cutouts:
        parser.error("--use-lsst-detection-calexp-cutouts requires --lsst-background-detect-cutouts")
    preprocess(args)


if __name__ == "__main__":
    main()
