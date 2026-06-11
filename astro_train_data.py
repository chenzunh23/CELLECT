"""Dataset discovery and target construction helpers for AstroCELLECT training."""

from __future__ import annotations

import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from astro_cellect2d import astro_zscale_preprocess, read_fits_bands

_CATALOG_IDS_CACHE: Dict[str, np.ndarray] = {}
_PSEUDO_LABELS_EMPTY: Dict[str, Dict[str, List[dict]]] = {}


@dataclass(frozen=True)
class CutoutRecord:
    name: str
    image_paths: Tuple[str, ...]
    meas_path: str
    x0: int
    y0: int
    band_meas_paths: Tuple[str, ...] = ()
    band_rejected_paths: Tuple[str, ...] = ()
    band_ignore_paths: Tuple[str, ...] = ()
    band_strict_center_only_paths: Tuple[str, ...] = ()
    band_strict_ignore_paths: Tuple[str, ...] = ()
    band_target_paths: Tuple[str, ...] = ()
    band_metadata_paths: Tuple[str, ...] = ()
    band_rejected_id_paths: Tuple[str, ...] = ()
    tile_name: str = ""
    tract: str = ""
    patch: str = ""
    relative_root: str = ""
    target_path: str = ""
    metadata_path: str = ""
    ignore_path: str = ""
    strict_center_only_path: str = ""
    strict_ignore_path: str = ""


def _expand_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _parse_tile_origin(name: str) -> Tuple[int, int]:
    match = re.search(r"_x(-?\d+)_y(-?\d+)", name)
    if match is None:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def _default_band_order(cutout_dir: Path) -> List[str]:
    bands = [p.name for p in cutout_dir.iterdir() if p.is_dir()]
    return sorted(bands)


def _find_band_fits(tile_dir: Path, band: str) -> str:
    band_dir = tile_dir / band
    if not band_dir.exists():
        raise FileNotFoundError(f"Missing band directory: {band_dir}")
    matches = sorted(band_dir.glob("*.fits"))
    if not matches:
        raise FileNotFoundError(f"No FITS image found in {band_dir}")
    return str(matches[0])


def _pseudo_band_fits(tile_dir: Path, band: str) -> str:
    return str(tile_dir / band / "__zscale_cache_only__.fits")


def _manifest_bands(patch_root: Path) -> List[str]:
    manifest_path = patch_root / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    bands = data.get("bands", [])
    return [str(band) for band in bands] if isinstance(bands, list) else []


def _find_band_meas(reference_root: Path, band: str, *, tract: str = "", patch: str = "") -> str:
    band_dir = reference_root / band
    candidates: List[Path] = []
    if tract and patch:
        exact_name = f"meas-{band}-{tract}-{patch}.fits"
        exact_candidates = [band_dir / exact_name] if band_dir.exists() else []
        exact_candidates.extend(sorted(reference_root.glob(f"**/{exact_name}")))
        candidates = [path for path in exact_candidates if path.exists()]
    if not candidates:
        candidates = sorted(band_dir.glob(f"meas-{band}-*.fits")) if band_dir.exists() else []
    if not candidates:
        candidates = sorted(reference_root.glob(f"**/meas-{band}-*.fits"))
    if not candidates:
        raise FileNotFoundError(f"No meas catalog found for band {band} under {reference_root}")
    return str(candidates[0])


def _find_optional_band_meas(reference_root: Path, band: str, *, tract: str = "", patch: str = "") -> str:
    try:
        return _find_band_meas(reference_root, band, tract=tract, patch=patch)
    except FileNotFoundError:
        return ""


def load_catalog_ids(path: str) -> np.ndarray:
    if not path:
        return np.zeros((0,), dtype=np.int64)
    resolved = Path(path)
    if not resolved.exists():
        return np.zeros((0,), dtype=np.int64)
    cache_key = str(resolved)
    cached = _CATALOG_IDS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if resolved.suffix == ".npz":
        with np.load(resolved) as data:
            ids = np.asarray(data["ids"], dtype=np.int64) if "ids" in data else np.zeros((0,), dtype=np.int64)
        _CATALOG_IDS_CACHE[cache_key] = ids
        return ids
    try:
        from astropy.io import fits
    except Exception as exc:
        raise RuntimeError("astro_train_eval.py requires astropy.") from exc
    with fits.open(resolved, memmap=True) as hdul:
        table = hdul[1].data
        cols = set(table.columns.names)
        if "id" not in cols:
            ids = np.zeros((0,), dtype=np.int64)
            _CATALOG_IDS_CACHE[cache_key] = ids
            return ids
        ids = np.asarray(table["id"], dtype=np.int64)
    _CATALOG_IDS_CACHE[cache_key] = ids
    return ids


def _record_name(tile_name: str, relative_root: str) -> str:
    return f"{relative_root}/{tile_name}" if relative_root else tile_name


def _relative_root_parts(relative_root: str) -> Tuple[str, str]:
    parts = Path(relative_root).parts
    if len(parts) >= 2:
        return str(parts[0]), str(parts[1])
    if len(parts) == 1 and str(parts[0]) not in ("", "."):
        return "", str(parts[0])
    return "", ""


def _discover_dataset_roots(
    root: Path,
    *,
    reference_dir: Optional[Path],
    cutout_dir: Optional[Path],
) -> List[Tuple[Path, Path, Path, str]]:
    if reference_dir is not None or cutout_dir is not None:
        ref_dir = reference_dir or (root / "reference_catalogs")
        img_dir = cutout_dir or (root / "cutouts")
        patch_root = ref_dir.parent if ref_dir.name == "reference_catalogs" else root
        try:
            relative_root = str(patch_root.relative_to(root))
        except ValueError:
            relative_root = ""
        if relative_root == ".":
            relative_root = ""
        return [(patch_root, ref_dir, img_dir, relative_root)]

    dataset_roots: List[Tuple[Path, Path, Path, str]] = []
    seen: set[Path] = set()

    flat_ref = root / "reference_catalogs"
    flat_cutouts = root / "cutouts"
    if flat_ref.exists():
        dataset_roots.append((root, flat_ref, flat_cutouts, ""))
        seen.add(flat_ref.resolve())

    for ref_dir in sorted(root.rglob("reference_catalogs")):
        resolved = ref_dir.resolve()
        if resolved in seen:
            continue
        patch_root = ref_dir.parent
        img_dir = patch_root / "cutouts"
        try:
            relative_root = str(patch_root.relative_to(root))
        except ValueError:
            relative_root = ""
        if relative_root == ".":
            relative_root = ""
        dataset_roots.append((patch_root, ref_dir, img_dir, relative_root))
        seen.add(resolved)

    return dataset_roots


def discover_cutout_records(
    root: Path,
    *,
    reference_dir: Optional[Path] = None,
    cutout_dir: Optional[Path] = None,
    band_reference_root: Optional[Path] = None,
    bands: Optional[Sequence[str]] = None,
    max_records: Optional[int] = None,
) -> List[CutoutRecord]:
    """Pair each 512x512 FITS cutout directory with its reference meas catalog."""

    dataset_roots = _discover_dataset_roots(root, reference_dir=reference_dir, cutout_dir=cutout_dir)
    if not dataset_roots:
        raise FileNotFoundError(
            f"No reference dataset found under {root}. Expected either reference_catalogs/ "
            f"or <tract>/<patch>/reference_catalogs."
        )

    records: List[CutoutRecord] = []
    for patch_root, ref_dir, img_dir, relative_root in dataset_roots:
        if not ref_dir.exists():
            raise FileNotFoundError(f"Reference catalog directory does not exist: {ref_dir}")

        local_band_reference_root = band_reference_root
        if local_band_reference_root is None and (patch_root / "band_reference_catalogs").exists():
            local_band_reference_root = patch_root / "band_reference_catalogs"
        local_band_rejected_root = patch_root / "band_reference_rejected"
        if not local_band_rejected_root.exists():
            local_band_rejected_root = None
        local_band_ignore_root = patch_root / "band_reference_ignore"
        if not local_band_ignore_root.exists():
            local_band_ignore_root = None
        local_band_strict_center_only_root = patch_root / "band_reference_strict_center_only"
        if not local_band_strict_center_only_root.exists():
            local_band_strict_center_only_root = None
        local_band_strict_ignore_root = patch_root / "band_reference_strict_ignore"
        if not local_band_strict_ignore_root.exists():
            local_band_strict_ignore_root = None
        tract, patch = _relative_root_parts(relative_root)

        for meas_path in sorted(ref_dir.glob("*_meas.fits")):
            tile_name = meas_path.name[: -len("_meas.fits")]
            tile_dir = img_dir / tile_name
            has_cutout_tile = tile_dir.exists()
            if has_cutout_tile:
                band_order = list(bands) if bands else _default_band_order(tile_dir)
            else:
                band_order = list(bands) if bands else _manifest_bands(patch_root)
            if not band_order:
                continue
            image_paths = (
                tuple(_find_band_fits(tile_dir, band) for band in band_order)
                if has_cutout_tile
                else tuple(_pseudo_band_fits(tile_dir, band) for band in band_order)
            )
            band_meas_paths = (
                tuple(_find_band_meas(local_band_reference_root, band, tract=tract, patch=patch) for band in band_order)
                if local_band_reference_root
                else ()
            )
            band_rejected_paths = (
                tuple(_find_optional_band_meas(local_band_rejected_root, band, tract=tract, patch=patch) for band in band_order)
                if local_band_rejected_root is not None
                else ()
            )
            band_ignore_paths = (
                tuple(_find_optional_band_meas(local_band_ignore_root, band, tract=tract, patch=patch) for band in band_order)
                if local_band_ignore_root is not None
                else ()
            )
            band_strict_ignore_paths = (
                tuple(
                    _find_optional_band_meas(local_band_strict_ignore_root, band, tract=tract, patch=patch)
                    for band in band_order
                )
                if local_band_strict_ignore_root is not None
                else ()
            )
            band_strict_center_only_paths = (
                tuple(
                    _find_optional_band_meas(local_band_strict_center_only_root, band, tract=tract, patch=patch)
                    for band in band_order
                )
                if local_band_strict_center_only_root is not None
                else band_strict_ignore_paths
            )
            band_target_paths = tuple(
                str(patch_root / "band_targets" / band / f"{tile_name}.npz")
                if (patch_root / "band_targets" / band / f"{tile_name}.npz").exists()
                else ""
                for band in band_order
            )
            band_metadata_paths = tuple(
                str(patch_root / "band_tile_metadata" / band / f"{tile_name}.npz")
                if (patch_root / "band_tile_metadata" / band / f"{tile_name}.npz").exists()
                else ""
                for band in band_order
            )
            band_rejected_id_paths = tuple(
                str(patch_root / "band_rejected_ids" / f"{band}.npz")
                if (patch_root / "band_rejected_ids" / f"{band}.npz").exists()
                else ""
                for band in band_order
            )
            x0, y0 = _parse_tile_origin(tile_name)
            target_path = patch_root / "targets" / f"{tile_name}.npz"
            metadata_path = patch_root / "tile_metadata" / f"{tile_name}.npz"
            ignore_path = patch_root / "ignore_catalogs" / f"{tile_name}_meas.fits"
            strict_center_only_path = patch_root / "strict_center_only_catalogs" / f"{tile_name}_meas.fits"
            strict_ignore_path = patch_root / "strict_ignore_catalogs" / f"{tile_name}_meas.fits"
            records.append(
                CutoutRecord(
                    name=_record_name(tile_name, relative_root),
                    image_paths=image_paths,
                    meas_path=str(meas_path),
                    x0=x0,
                    y0=y0,
                    band_meas_paths=band_meas_paths,
                    band_rejected_paths=band_rejected_paths,
                    band_ignore_paths=band_ignore_paths,
                    band_strict_center_only_paths=band_strict_center_only_paths,
                    band_strict_ignore_paths=band_strict_ignore_paths,
                    band_target_paths=band_target_paths,
                    band_metadata_paths=band_metadata_paths,
                    band_rejected_id_paths=band_rejected_id_paths,
                    tile_name=tile_name,
                    tract=tract,
                    patch=patch,
                    relative_root=relative_root,
                    target_path=str(target_path) if target_path.exists() else "",
                    metadata_path=str(metadata_path) if metadata_path.exists() else "",
                    ignore_path=str(ignore_path) if ignore_path.exists() else "",
                    strict_center_only_path=(
                        str(strict_center_only_path)
                        if strict_center_only_path.exists()
                        else (str(strict_ignore_path) if strict_ignore_path.exists() else "")
                    ),
                    strict_ignore_path=str(strict_ignore_path) if strict_ignore_path.exists() else "",
                )
            )
            if max_records is not None and len(records) >= max_records:
                break
        if max_records is not None and len(records) >= max_records:
            break
    if not records:
        raise RuntimeError(f"No paired cutout/meas records found under {root}")
    return records


def _finite_column(table, names: Sequence[str]) -> Optional[np.ndarray]:
    cols = set(table.columns.names)
    for name in names:
        if name in cols:
            vals = np.asarray(table[name], dtype=np.float32)
            if np.isfinite(vals).any():
                return vals
    return None


def load_meas_catalog(
    path: str,
    *,
    x0: int,
    y0: int,
    image_shape: Tuple[int, int],
    source_filter: str = "nchild0",
) -> Dict[str, np.ndarray]:
    """Read centers and approximate shape labels from an LSST meas FITS table.

    The center label is treated as the reliable supervision. Shape is less
    reliable and is only used to build a CELLECT-style pseudo segmentation mask:
    moments provide orientation/axis ratio, and Kron radius can optionally
    provide the aperture scale.
    """

    try:
        from astropy.io import fits
    except Exception as exc:
        raise RuntimeError("astro_train_eval.py requires astropy.") from exc

    with fits.open(path, memmap=True) as hdul:
        table = hdul[1].data
        cols = set(table.columns.names)

        x = _finite_column(
            table,
            (
                "centroid_local_x",
                "base_SdssShape_x",
                "base_SdssCentroid_x",
                "base_NaiveCentroid_x",
                "deblend_psfCenter_x",
            ),
        )
        y = _finite_column(
            table,
            (
                "centroid_local_y",
                "base_SdssShape_y",
                "base_SdssCentroid_y",
                "base_NaiveCentroid_y",
                "deblend_psfCenter_y",
            ),
        )
        if x is None or y is None:
            raise ValueError(f"No usable center columns found in {path}")

        if "centroid_local_x" not in cols:
            x = x - float(x0)
        if "centroid_local_y" not in cols:
            y = y - float(y0)

        ids = np.asarray(table["id"], dtype=np.int64) if "id" in cols else np.arange(len(x), dtype=np.int64)
        parent = np.asarray(table["parent"], dtype=np.int64) if "parent" in cols else np.zeros(len(x), dtype=np.int64)
        nchild = _child_count_column(table, cols, source_filter)
        xx = _finite_column(table, ("base_SdssShape_xx", "ext_shapeHSM_HsmSourceMoments_xx"))
        yy = _finite_column(table, ("base_SdssShape_yy", "ext_shapeHSM_HsmSourceMoments_yy"))
        xy = _finite_column(table, ("base_SdssShape_xy", "ext_shapeHSM_HsmSourceMoments_xy"))
        kron_radius = _finite_column(
            table,
            (
                "ext_photometryKron_KronFlux_radius",
                "ext_photometryKron_KronFlux_radius_for_radius",
            ),
        )
        footprint = _finite_column(table, ("base_FootprintArea_value",))

    h, w = image_shape
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    valid &= _source_filter_mask(parent=parent, nchild=nchild, source_filter=source_filter)
    if xx is not None and yy is not None and xy is not None:
        valid &= np.isfinite(xx) & np.isfinite(yy) & np.isfinite(xy)

    x = x[valid]
    y = y[valid]
    ids = ids[valid]
    if xx is None or yy is None or xy is None:
        xx = np.full_like(x, 4.0, dtype=np.float32)
        yy = np.full_like(x, 4.0, dtype=np.float32)
        xy = np.zeros_like(x, dtype=np.float32)
    else:
        xx = np.asarray(xx[valid], dtype=np.float32)
        yy = np.asarray(yy[valid], dtype=np.float32)
        xy = np.asarray(xy[valid], dtype=np.float32)
    if footprint is None:
        footprint = np.zeros_like(x, dtype=np.float32)
    else:
        footprint = np.asarray(footprint[valid], dtype=np.float32)
    if kron_radius is None:
        kron_radius = np.full_like(x, np.nan, dtype=np.float32)
    else:
        kron_radius = np.asarray(kron_radius[valid], dtype=np.float32)

    centers = np.stack([x, y], axis=1).astype(np.float32)
    return {
        "centers": centers,
        "ids": ids.astype(np.int64),
        "moments": np.stack([xx, yy, xy], axis=1).astype(np.float32),
        "kron_radius": kron_radius.astype(np.float32),
        "footprint": footprint.astype(np.float32),
    }


def load_catalog_centers(path: str, *, x0: int, y0: int, image_shape: Tuple[int, int]) -> np.ndarray:
    if not path:
        return np.zeros((0, 2), dtype=np.float32)
    resolved = Path(path)
    if not resolved.exists():
        return np.zeros((0, 2), dtype=np.float32)
    try:
        from astropy.io import fits
    except Exception as exc:
        raise RuntimeError("astro_train_eval.py requires astropy.") from exc

    with fits.open(resolved, memmap=True) as hdul:
        table = hdul[1].data
        cols = set(table.columns.names)
        x = _finite_column(
            table,
            (
                "centroid_local_x",
                "base_SdssShape_x",
                "base_SdssCentroid_x",
                "base_NaiveCentroid_x",
                "deblend_psfCenter_x",
            ),
        )
        y = _finite_column(
            table,
            (
                "centroid_local_y",
                "base_SdssShape_y",
                "base_SdssCentroid_y",
                "base_NaiveCentroid_y",
                "deblend_psfCenter_y",
            ),
        )
    if x is None or y is None:
        return np.zeros((0, 2), dtype=np.float32)
    if "centroid_local_x" not in cols:
        x = x - float(x0)
    if "centroid_local_y" not in cols:
        y = y - float(y0)
    h, w = image_shape
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if not bool(valid.any()):
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack([x[valid], y[valid]], axis=1).astype(np.float32)


def load_catalog_metadata_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(Path(path)) as data:
        centers = np.asarray(data["centers"], dtype=np.float32)
        ids = np.asarray(data["ids"], dtype=np.int64) if "ids" in data else np.arange(len(centers), dtype=np.int64)
        if "moments" in data:
            moments = np.asarray(data["moments"], dtype=np.float32)
        else:
            moments = np.zeros((len(centers), 3), dtype=np.float32)
            moments[:, 0] = 4.0
            moments[:, 1] = 4.0
        kron_radius = (
            np.asarray(data["kron_radius"], dtype=np.float32)
            if "kron_radius" in data
            else np.full(len(centers), np.nan, dtype=np.float32)
        )
        footprint = (
            np.asarray(data["footprint"], dtype=np.float32)
            if "footprint" in data
            else np.zeros(len(centers), dtype=np.float32)
        )
    return {
        "centers": centers.astype(np.float32),
        "ids": ids.astype(np.int64),
        "moments": moments.astype(np.float32),
        "kron_radius": kron_radius.astype(np.float32),
        "footprint": footprint.astype(np.float32),
    }


def _child_count_column(table, cols: set[str], source_filter: str) -> Optional[np.ndarray]:
    if source_filter in ("all", "parent"):
        return None
    if "deblend_nChild" in cols:
        return np.asarray(table["deblend_nChild"], dtype=np.int64)
    if "nChild" in cols:
        return np.asarray(table["nChild"], dtype=np.int64)
    raise KeyError(f"source_filter='{source_filter}' requires deblend_nChild or nChild column")


def _source_filter_mask(
    *,
    parent: np.ndarray,
    nchild: Optional[np.ndarray],
    source_filter: str,
) -> np.ndarray:
    if source_filter == "all":
        return np.ones(parent.shape[0], dtype=bool)
    if source_filter == "parent":
        return parent == 0
    if nchild is None:
        raise KeyError(f"source_filter='{source_filter}' requires child-count labels")
    leaf = nchild == 0
    if source_filter == "nchild0":
        return leaf
    if source_filter == "leaf_child":
        return leaf & (parent != 0)
    raise ValueError(f"Unknown source_filter: {source_filter}")


def _ellipse_parameters(
    moments: np.ndarray,
    *,
    kron_radius: Optional[np.ndarray] = None,
    shape_source: str = "kron",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx = np.maximum(moments[:, 0], 0.25)
    yy = np.maximum(moments[:, 1], 0.25)
    xy = moments[:, 2]
    trace = xx + yy
    delta = np.sqrt(np.maximum((xx - yy) ** 2 + 4.0 * xy ** 2, 0.0))
    major_var = np.maximum(0.5 * (trace + delta), 0.25)
    minor_var = np.maximum(0.5 * (trace - delta), 0.25)
    major = np.clip(np.sqrt(major_var), 1.0, 32.0)
    minor = np.clip(np.sqrt(minor_var), 1.0, 32.0)
    angle = 0.5 * np.arctan2(2.0 * xy, xx - yy)

    if shape_source == "kron" and kron_radius is not None:
        kron = np.asarray(kron_radius, dtype=np.float32)
        valid = np.isfinite(kron) & (kron > 0)
        # LSST KronFlux stores the Kron aperture radius as a scalar. The local
        # moments still provide orientation and axis ratio; Kron supplies scale.
        axis_ratio = np.clip(minor / np.maximum(major, 1e-3), 0.15, 1.0)
        major = np.where(valid, kron, major)
        minor = np.where(valid, kron * axis_ratio, minor)

    if shape_source == "circular_kron" and kron_radius is not None:
        kron = np.asarray(kron_radius, dtype=np.float32)
        valid = np.isfinite(kron) & (kron > 0)
        major = np.where(valid, kron, major)
        minor = np.where(valid, kron, minor)

    major = np.clip(major, 1.0, 64.0)
    minor = np.clip(minor, 1.0, 64.0)
    return major.astype(np.float32), minor.astype(np.float32), angle.astype(np.float32)


def make_targets(
    *,
    image_shape: Tuple[int, int],
    centers: np.ndarray,
    moments: np.ndarray,
    kron_radius: Optional[np.ndarray] = None,
    confidence_levels: int = 5,
    ellipse_sigma: float = 2.0,
    core_radius: int = 2,
    shape_source: str = "kron",
) -> Dict[str, Tensor]:
    """Build CELLECT-style dense labels from accurate centers and noisy shapes."""

    h, w = image_shape
    seg = torch.zeros((h, w), dtype=torch.long)
    conf = torch.zeros((h, w), dtype=torch.long)
    shape = torch.zeros((3, h, w), dtype=torch.float32)
    shape_weight = torch.zeros((h, w), dtype=torch.float32)
    if centers.size == 0:
        return {"seg": seg, "confidence": conf, "shape": shape, "shape_weight": shape_weight}

    major, minor, angle = _ellipse_parameters(
        moments,
        kron_radius=kron_radius,
        shape_source=shape_source,
    )
    yy_full, xx_full = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    xx_np = xx_full.numpy()
    yy_np = yy_full.numpy()

    for idx, (cx, cy) in enumerate(centers):
        cx_i = int(round(float(cx)))
        cy_i = int(round(float(cy)))
        if cx_i < 0 or cx_i >= w or cy_i < 0 or cy_i >= h:
            continue

        a = float(max(major[idx] * ellipse_sigma, 1.5))
        b = float(max(minor[idx] * ellipse_sigma, 1.5))
        theta = float(angle[idx])
        radius = int(math.ceil(max(a, b))) + 2
        y0, y1 = max(0, cy_i - radius), min(h, cy_i + radius + 1)
        x0, x1 = max(0, cx_i - radius), min(w, cx_i + radius + 1)

        dx = xx_np[y0:y1, x0:x1] - float(cx)
        dy = yy_np[y0:y1, x0:x1] - float(cy)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        xr = cos_t * dx + sin_t * dy
        yr = -sin_t * dx + cos_t * dy
        ellipse = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
        if ellipse.any():
            ellipse_t = torch.from_numpy(ellipse)
            patch_seg = seg[y0:y1, x0:x1]
            patch_seg[ellipse_t] = torch.maximum(patch_seg[ellipse_t], torch.ones_like(patch_seg[ellipse_t]))
            shape[:, y0:y1, x0:x1][:, ellipse_t] = torch.tensor(
                [float(major[idx]), float(minor[idx]), theta], dtype=torch.float32
            )[:, None]
            shape_weight[y0:y1, x0:x1][ellipse_t] = 1.0

        level_radius = confidence_levels - 1
        cy0, cy1 = max(0, cy_i - level_radius), min(h, cy_i + level_radius + 1)
        cx0, cx1 = max(0, cx_i - level_radius), min(w, cx_i + level_radius + 1)
        dist = torch.abs(xx_full[cy0:cy1, cx0:cx1] - float(cx)) + torch.abs(yy_full[cy0:cy1, cx0:cx1] - float(cy))
        vals = torch.ceil(torch.clamp(level_radius - dist, min=0))
        conf[cy0:cy1, cx0:cx1] = torch.maximum(conf[cy0:cy1, cx0:cx1], vals.long())

    return {"seg": seg, "confidence": conf, "shape": shape, "shape_weight": shape_weight}


_SPATIAL_TARGET_KEYS = (
    "seg",
    "confidence",
    "shape",
    "shape_weight",
    "clean_mask",
    "center_only_mask",
    "ignore_mask",
    "strict_center_only_mask",
    "strict_ignore_mask",
    "source_union_mask",
    "background_mask",
    "pu_class_mask",
    "confidence_weight",
    "seg_loss_weight",
    "pseudo_mask",
)


def _target_defaults(targets: Dict[str, Tensor]) -> Dict[str, Tensor]:
    h, w = int(targets["seg"].shape[-2]), int(targets["seg"].shape[-1])
    device = targets["seg"].device
    has_confidence_weight = "confidence_weight" in targets
    has_seg_loss_weight = "seg_loss_weight" in targets
    targets.setdefault("clean_mask", (targets["shape_weight"] > 0).to(dtype=torch.uint8))
    targets.setdefault("center_only_mask", torch.zeros((h, w), dtype=torch.uint8, device=device))
    targets.setdefault("ignore_mask", torch.zeros((h, w), dtype=torch.uint8, device=device))
    targets.setdefault("strict_center_only_mask", torch.zeros((h, w), dtype=torch.uint8, device=device))
    targets.setdefault("strict_ignore_mask", torch.zeros((h, w), dtype=torch.uint8, device=device))
    targets["ignore_mask"] = ((targets["ignore_mask"] > 0) | (targets["strict_ignore_mask"] > 0)).to(dtype=torch.uint8)
    targets.setdefault("source_union_mask", (targets["clean_mask"] > 0).to(dtype=torch.uint8))
    targets.setdefault("background_mask", (targets["source_union_mask"] == 0).to(dtype=torch.uint8))
    targets.setdefault("pu_class_mask", targets["clean_mask"].to(dtype=torch.uint8))
    targets.setdefault("pseudo_mask", torch.zeros((h, w), dtype=torch.uint8, device=device))
    if not has_confidence_weight:
        uncertain = (targets["ignore_mask"] > 0).to(dtype=torch.bool)
        targets["confidence_weight"] = (~uncertain).to(dtype=torch.float32)
    if not has_seg_loss_weight:
        reliable = ((targets["clean_mask"] > 0) | (targets["background_mask"] > 0)).to(dtype=torch.bool)
        targets["seg_loss_weight"] = reliable.to(dtype=torch.float32)
    return targets


def _read_target_npz(path: Path) -> Dict[str, Tensor]:
    with np.load(path) as data:
        targets: Dict[str, Tensor] = {
            "seg": torch.from_numpy(np.asarray(data["seg"], dtype=np.int64)),
            "confidence": torch.from_numpy(np.asarray(data["confidence"], dtype=np.int64)),
            "shape": torch.from_numpy(np.asarray(data["shape"], dtype=np.float32)),
            "shape_weight": torch.from_numpy(np.asarray(data["shape_weight"], dtype=np.float32)),
        }
        for key in (
            "clean_mask",
            "center_only_mask",
            "ignore_mask",
            "strict_center_only_mask",
            "strict_ignore_mask",
            "source_union_mask",
            "background_mask",
            "pu_class_mask",
            "pseudo_mask",
        ):
            if key in data:
                targets[key] = torch.from_numpy(np.asarray(data[key], dtype=np.uint8))
        for key in ("confidence_weight", "seg_loss_weight"):
            if key in data:
                targets[key] = torch.from_numpy(np.asarray(data[key], dtype=np.float32))
    return _target_defaults(targets)


def load_pseudo_labels(path: Optional[Path]) -> Dict[str, Dict[str, List[dict]]]:
    if path is None or not Path(path).exists():
        return _PSEUDO_LABELS_EMPTY
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    by_record = data.get("by_record", {})
    if not isinstance(by_record, dict):
        return _PSEUDO_LABELS_EMPTY
    out: Dict[str, Dict[str, List[dict]]] = {}
    for record_name, bands in by_record.items():
        if not isinstance(bands, dict):
            continue
        record_bucket: Dict[str, List[dict]] = {}
        for band_name, rows in bands.items():
            if isinstance(rows, list):
                record_bucket[str(band_name)] = [row for row in rows if isinstance(row, dict)]
        out[str(record_name)] = record_bucket
    return out


def _pseudo_record_bands(pseudo_labels: Dict[str, Dict[str, List[dict]]], rec: CutoutRecord) -> Dict[str, List[dict]]:
    for key in (rec.name, rec.tile_name, Path(rec.name).name):
        if key and key in pseudo_labels:
            return pseudo_labels[key]
    return {}


def _paint_pseudo_labels(
    targets: Dict[str, Tensor],
    labels: Sequence[dict],
    *,
    confidence_levels: int,
    ellipse_sigma: float,
    confidence_weight: float,
    seg_weight: float,
    shape_weight: float,
) -> None:
    if not labels:
        return
    h, w = int(targets["seg"].shape[-2]), int(targets["seg"].shape[-1])
    yy_full, xx_full = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    for row in labels:
        try:
            cx = float(row["x"])
            cy = float(row["y"])
        except Exception:
            continue
        cx_i = int(round(cx))
        cy_i = int(round(cy))
        if cx_i < 0 or cx_i >= w or cy_i < 0 or cy_i >= h:
            continue
        major = float(row.get("major", 1.5))
        minor = float(row.get("minor", major))
        theta = float(row.get("theta", 0.0))
        if not (math.isfinite(major) and math.isfinite(minor) and math.isfinite(theta)):
            continue
        a = max(abs(major) * float(ellipse_sigma), 1.5)
        b = max(abs(minor) * float(ellipse_sigma), 1.5)
        radius = int(math.ceil(max(a, b))) + 2
        y0, y1 = max(0, cy_i - radius), min(h, cy_i + radius + 1)
        x0, x1 = max(0, cx_i - radius), min(w, cx_i + radius + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        dx = xx_full[y0:y1, x0:x1].numpy() - cx
        dy = yy_full[y0:y1, x0:x1].numpy() - cy
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        xr = cos_t * dx + sin_t * dy
        yr = -sin_t * dx + cos_t * dy
        ellipse_np = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
        ellipse = torch.from_numpy(ellipse_np)
        if bool(ellipse.any()):
            targets["seg"][y0:y1, x0:x1][ellipse] = 1
            targets["shape"][:, y0:y1, x0:x1][:, ellipse] = torch.tensor(
                [major, minor, theta], dtype=torch.float32
            )[:, None]
            targets["shape_weight"][y0:y1, x0:x1][ellipse] = torch.maximum(
                targets["shape_weight"][y0:y1, x0:x1][ellipse],
                torch.full_like(targets["shape_weight"][y0:y1, x0:x1][ellipse], float(shape_weight)),
            )
            targets["seg_loss_weight"][y0:y1, x0:x1][ellipse] = torch.maximum(
                targets["seg_loss_weight"][y0:y1, x0:x1][ellipse],
                torch.full_like(targets["seg_loss_weight"][y0:y1, x0:x1][ellipse], float(seg_weight)),
            )
            targets["pseudo_mask"][y0:y1, x0:x1][ellipse] = 1
            targets["source_union_mask"][y0:y1, x0:x1][ellipse] = 1
            targets["background_mask"][y0:y1, x0:x1][ellipse] = 0
            targets["pu_class_mask"][y0:y1, x0:x1][ellipse] = 4

        level_radius = int(confidence_levels) - 1
        cy0, cy1 = max(0, cy_i - level_radius), min(h, cy_i + level_radius + 1)
        cx0, cx1 = max(0, cx_i - level_radius), min(w, cx_i + level_radius + 1)
        dist = torch.abs(xx_full[cy0:cy1, cx0:cx1] - float(cx)) + torch.abs(yy_full[cy0:cy1, cx0:cx1] - float(cy))
        vals = torch.ceil(torch.clamp(level_radius - dist, min=0)).long()
        region = vals > 0
        targets["confidence"][cy0:cy1, cx0:cx1][region] = torch.maximum(
            targets["confidence"][cy0:cy1, cx0:cx1][region],
            vals[region],
        )
        targets["confidence_weight"][cy0:cy1, cx0:cx1][region] = torch.maximum(
            targets["confidence_weight"][cy0:cy1, cx0:cx1][region],
            torch.full_like(targets["confidence_weight"][cy0:cy1, cx0:cx1][region], float(confidence_weight)),
        )


class AstroCutoutDataset(Dataset):
    def __init__(
        self,
        records: Sequence[CutoutRecord],
        *,
        fits_hdu: int = 1,
        confidence_levels: int = 5,
        ellipse_sigma: float = 2.0,
        core_radius: int = 2,
        shape_source: str = "kron",
        source_filter: str = "nchild0",
        targets_dir: Optional[Path] = None,
        image_cache_dir: Optional[Path] = None,
        pseudo_label_path: Optional[Path] = None,
        pseudo_confidence_weight: float = 0.35,
        pseudo_seg_weight: float = 0.25,
        pseudo_shape_weight: float = 0.15,
        load_eval_ignore_sources: bool = False,
        augment: bool = False,
    ) -> None:
        self.records = list(records)
        self.fits_hdu = int(fits_hdu)
        self.confidence_levels = int(confidence_levels)
        self.ellipse_sigma = float(ellipse_sigma)
        self.core_radius = int(core_radius)
        self.shape_source = str(shape_source)
        self.source_filter = str(source_filter)
        self.targets_dir = Path(targets_dir).expanduser().resolve() if targets_dir is not None else None
        self.image_cache_dir = Path(image_cache_dir).expanduser().resolve() if image_cache_dir is not None else None
        self.pseudo_label_path = Path(pseudo_label_path).expanduser().resolve() if pseudo_label_path is not None else None
        self.pseudo_confidence_weight = float(pseudo_confidence_weight)
        self.pseudo_seg_weight = float(pseudo_seg_weight)
        self.pseudo_shape_weight = float(pseudo_shape_weight)
        self.pseudo_labels = load_pseudo_labels(self.pseudo_label_path)
        self.load_eval_ignore_sources = bool(load_eval_ignore_sources)
        self.augment = bool(augment)

    def reload_pseudo_labels(self, pseudo_label_path: Optional[Path] = None) -> None:
        if pseudo_label_path is not None:
            self.pseudo_label_path = Path(pseudo_label_path).expanduser().resolve()
        self.pseudo_labels = load_pseudo_labels(self.pseudo_label_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        rec = self.records[idx]
        image = self._load_or_make_image(rec)
        h, w = int(image.shape[-2]), int(image.shape[-1])
        catalog = self._load_catalog(rec.metadata_path, rec.meas_path, rec, image_shape=(h, w))
        targets = self._load_or_make_targets(rec, image_shape=(h, w), catalog=catalog)
        record_pseudo = _pseudo_record_bands(self.pseudo_labels, rec)
        primary_pseudo: List[dict] = []
        for rows in record_pseudo.values():
            primary_pseudo.extend(rows)
        _paint_pseudo_labels(
            targets,
            primary_pseudo,
            confidence_levels=self.confidence_levels,
            ellipse_sigma=self.ellipse_sigma,
            confidence_weight=self.pseudo_confidence_weight,
            seg_weight=self.pseudo_seg_weight,
            shape_weight=self.pseudo_shape_weight,
        )
        band_catalogs: List[Dict[str, np.ndarray]] = []
        band_targets: List[Dict[str, Tensor]] = []
        band_meas_paths = rec.band_meas_paths if rec.band_meas_paths else tuple(rec.meas_path for _ in range(image.shape[0]))
        for band_idx, meas_path in enumerate(band_meas_paths):
            band_metadata_path = rec.band_metadata_paths[band_idx] if band_idx < len(rec.band_metadata_paths) else ""
            band_catalog = self._load_catalog(
                band_metadata_path,
                meas_path,
                rec,
                image_shape=(h, w),
            )
            band_catalogs.append(band_catalog)
            band_target_path = rec.band_target_paths[band_idx] if band_idx < len(rec.band_target_paths) else ""
            if band_target_path and Path(band_target_path).exists():
                band_targets.append(_read_target_npz(Path(band_target_path)))
            else:
                band_targets.append(
                    _target_defaults(
                        make_targets(
                            image_shape=(h, w),
                            centers=band_catalog["centers"],
                            moments=band_catalog["moments"],
                            kron_radius=band_catalog["kron_radius"],
                            confidence_levels=self.confidence_levels,
                            ellipse_sigma=self.ellipse_sigma,
                            core_radius=self.core_radius,
                            shape_source=self.shape_source,
                        )
                    )
                )
            band_name = Path(meas_path).parts[-2] if len(Path(meas_path).parts) >= 2 else ""
            pseudo_rows = record_pseudo.get(band_name, [])
            if not pseudo_rows and band_idx < image.shape[0]:
                # Fall back to the band order implied by image paths.
                image_band = Path(rec.image_paths[band_idx]).parts[-2] if band_idx < len(rec.image_paths) else ""
                pseudo_rows = record_pseudo.get(image_band, [])
            if pseudo_rows:
                _paint_pseudo_labels(
                    band_targets[-1],
                    pseudo_rows,
                    confidence_levels=self.confidence_levels,
                    ellipse_sigma=self.ellipse_sigma,
                    confidence_weight=self.pseudo_confidence_weight,
                    seg_weight=self.pseudo_seg_weight,
                    shape_weight=self.pseudo_shape_weight,
                )
        centers = torch.from_numpy(catalog["centers"])
        band_centers = [torch.from_numpy(item["centers"]) for item in band_catalogs]
        band_ids = [torch.from_numpy(item["ids"]) for item in band_catalogs]
        ignore_centers = torch.from_numpy(
            load_catalog_centers(rec.ignore_path, x0=rec.x0, y0=rec.y0, image_shape=(h, w))
            if self.load_eval_ignore_sources
            else np.zeros((0, 2), dtype=np.float32)
        )
        strict_ignore_centers = torch.from_numpy(
            load_catalog_centers(rec.strict_ignore_path, x0=rec.x0, y0=rec.y0, image_shape=(h, w))
            if self.load_eval_ignore_sources
            else np.zeros((0, 2), dtype=np.float32)
        )
        strict_center_only_centers = torch.from_numpy(
            load_catalog_centers(rec.strict_center_only_path, x0=rec.x0, y0=rec.y0, image_shape=(h, w))
            if self.load_eval_ignore_sources
            else np.zeros((0, 2), dtype=np.float32)
        )
        if self.load_eval_ignore_sources:
            band_ignore_centers = [
                torch.from_numpy(load_catalog_centers(path, x0=rec.x0, y0=rec.y0, image_shape=(h, w)))
                for path in rec.band_ignore_paths
            ]
            band_strict_center_only_centers = [
                torch.from_numpy(load_catalog_centers(path, x0=rec.x0, y0=rec.y0, image_shape=(h, w)))
                for path in rec.band_strict_center_only_paths
            ]
            band_strict_ignore_centers = [
                torch.from_numpy(load_catalog_centers(path, x0=rec.x0, y0=rec.y0, image_shape=(h, w)))
                for path in rec.band_strict_ignore_paths
            ]
        else:
            band_ignore_centers = []
            band_strict_center_only_centers = []
            band_strict_ignore_centers = []
        if len(band_ignore_centers) < len(band_meas_paths):
            band_ignore_centers.extend(torch.empty((0, 2), dtype=torch.float32) for _ in range(len(band_meas_paths) - len(band_ignore_centers)))
        if len(band_strict_ignore_centers) < len(band_meas_paths):
            band_strict_ignore_centers.extend(
                torch.empty((0, 2), dtype=torch.float32) for _ in range(len(band_meas_paths) - len(band_strict_ignore_centers))
            )
        if len(band_strict_center_only_centers) < len(band_meas_paths):
            band_strict_center_only_centers.extend(
                torch.empty((0, 2), dtype=torch.float32)
                for _ in range(len(band_meas_paths) - len(band_strict_center_only_centers))
            )
        rejected_id_paths = rec.band_rejected_id_paths if any(rec.band_rejected_id_paths) else rec.band_rejected_paths
        if rejected_id_paths:
            band_rejected_ids = [torch.from_numpy(load_catalog_ids(path)) for path in rejected_id_paths]
        else:
            band_rejected_ids = [torch.empty((0,), dtype=torch.long) for _ in band_meas_paths]
        if self.augment and random.random() < 0.5:
            image = torch.flip(image, dims=(-1,))
            targets["seg"] = torch.flip(targets["seg"], dims=(-1,))
            targets["confidence"] = torch.flip(targets["confidence"], dims=(-1,))
            targets["shape"] = torch.flip(targets["shape"], dims=(-1,))
            targets["shape"][2] = -targets["shape"][2]
            targets["shape_weight"] = torch.flip(targets["shape_weight"], dims=(-1,))
            for key in _SPATIAL_TARGET_KEYS:
                if key in ("seg", "confidence", "shape", "shape_weight"):
                    continue
                targets[key] = torch.flip(targets[key], dims=(-1,))
            for target in band_targets:
                target["seg"] = torch.flip(target["seg"], dims=(-1,))
                target["confidence"] = torch.flip(target["confidence"], dims=(-1,))
                target["shape"] = torch.flip(target["shape"], dims=(-1,))
                target["shape"][2] = -target["shape"][2]
                target["shape_weight"] = torch.flip(target["shape_weight"], dims=(-1,))
                for key in _SPATIAL_TARGET_KEYS:
                    if key in ("seg", "confidence", "shape", "shape_weight"):
                        continue
                    target[key] = torch.flip(target[key], dims=(-1,))
            centers = centers.clone()
            centers[:, 0] = float(w - 1) - centers[:, 0]
            ignore_centers = ignore_centers.clone()
            if ignore_centers.numel():
                ignore_centers[:, 0] = float(w - 1) - ignore_centers[:, 0]
            strict_ignore_centers = strict_ignore_centers.clone()
            if strict_ignore_centers.numel():
                strict_ignore_centers[:, 0] = float(w - 1) - strict_ignore_centers[:, 0]
            strict_center_only_centers = strict_center_only_centers.clone()
            if strict_center_only_centers.numel():
                strict_center_only_centers[:, 0] = float(w - 1) - strict_center_only_centers[:, 0]
            band_centers = [center.clone() for center in band_centers]
            for center in band_centers:
                center[:, 0] = float(w - 1) - center[:, 0]
            band_ignore_centers = [center.clone() for center in band_ignore_centers]
            for center in band_ignore_centers:
                if center.numel():
                    center[:, 0] = float(w - 1) - center[:, 0]
            band_strict_ignore_centers = [center.clone() for center in band_strict_ignore_centers]
            for center in band_strict_ignore_centers:
                if center.numel():
                    center[:, 0] = float(w - 1) - center[:, 0]
            band_strict_center_only_centers = [center.clone() for center in band_strict_center_only_centers]
            for center in band_strict_center_only_centers:
                if center.numel():
                    center[:, 0] = float(w - 1) - center[:, 0]

        return {
            "image": image,
            "seg": targets["seg"],
            "confidence": targets["confidence"],
            "shape": targets["shape"],
            "shape_weight": targets["shape_weight"],
            "confidence_weight": targets["confidence_weight"],
            "seg_loss_weight": targets["seg_loss_weight"],
            "clean_mask": targets["clean_mask"],
            "center_only_mask": targets["center_only_mask"],
            "ignore_mask": targets["ignore_mask"],
            "strict_center_only_mask": targets["strict_center_only_mask"],
            "strict_ignore_mask": targets["strict_ignore_mask"],
            "source_union_mask": targets["source_union_mask"],
            "background_mask": targets["background_mask"],
            "pu_class_mask": targets["pu_class_mask"],
            "pseudo_mask": targets["pseudo_mask"],
            "band_seg": torch.stack([target["seg"] for target in band_targets]),
            "band_confidence": torch.stack([target["confidence"] for target in band_targets]),
            "band_shape": torch.stack([target["shape"] for target in band_targets]),
            "band_shape_weight": torch.stack([target["shape_weight"] for target in band_targets]),
            "band_confidence_weight": torch.stack([target["confidence_weight"] for target in band_targets]),
            "band_seg_loss_weight": torch.stack([target["seg_loss_weight"] for target in band_targets]),
            "band_clean_mask": torch.stack([target["clean_mask"] for target in band_targets]),
            "band_center_only_mask": torch.stack([target["center_only_mask"] for target in band_targets]),
            "band_ignore_mask": torch.stack([target["ignore_mask"] for target in band_targets]),
            "band_strict_center_only_mask": torch.stack([target["strict_center_only_mask"] for target in band_targets]),
            "band_strict_ignore_mask": torch.stack([target["strict_ignore_mask"] for target in band_targets]),
            "band_source_union_mask": torch.stack([target["source_union_mask"] for target in band_targets]),
            "band_background_mask": torch.stack([target["background_mask"] for target in band_targets]),
            "band_pu_class_mask": torch.stack([target["pu_class_mask"] for target in band_targets]),
            "band_pseudo_mask": torch.stack([target["pseudo_mask"] for target in band_targets]),
            "centers": centers,
            "ids": torch.from_numpy(catalog["ids"]),
            "ignore_centers": ignore_centers,
            "strict_center_only_centers": strict_center_only_centers,
            "strict_ignore_centers": strict_ignore_centers,
            "band_centers": band_centers,
            "band_ids": band_ids,
            "band_ignore_centers": band_ignore_centers,
            "band_strict_center_only_centers": band_strict_center_only_centers,
            "band_strict_ignore_centers": band_strict_ignore_centers,
            "band_rejected_ids": band_rejected_ids,
            "name": rec.name,
            "tile_name": rec.tile_name,
            "tract": rec.tract,
            "patch": rec.patch,
            "relative_root": rec.relative_root,
            "x0": rec.x0,
            "y0": rec.y0,
            "image_paths": rec.image_paths,
        }

    def _load_catalog(
        self,
        metadata_path: str,
        fallback_meas_path: str,
        rec: CutoutRecord,
        *,
        image_shape: Tuple[int, int],
    ) -> Dict[str, np.ndarray]:
        if metadata_path and Path(metadata_path).exists():
            return load_catalog_metadata_npz(metadata_path)
        return load_meas_catalog(
            fallback_meas_path,
            x0=rec.x0,
            y0=rec.y0,
            image_shape=image_shape,
            source_filter=self.source_filter,
        )

    def _image_cache_path(self, rec: CutoutRecord) -> Optional[Path]:
        if self.image_cache_dir is None:
            return None
        band_key = "_".join(Path(path).parent.name for path in rec.image_paths)
        tile_name = rec.tile_name or Path(rec.name).name
        cache_dir = self.image_cache_dir
        if rec.relative_root:
            cache_dir = cache_dir / rec.relative_root / "cutouts"
        return cache_dir / f"{tile_name}__{band_key}__hdu{self.fits_hdu}.pt"

    def _load_single_band_image_caches(self, rec: CutoutRecord) -> Optional[Tensor]:
        if self.image_cache_dir is None:
            return None
        tile_name = rec.tile_name or Path(rec.name).name
        cache_dir = self.image_cache_dir
        if rec.relative_root:
            cache_dir = cache_dir / rec.relative_root / "cutouts"
        bands = [Path(path).parent.name for path in rec.image_paths]
        tensors: List[Tensor] = []
        missing: List[str] = []
        for band in bands:
            path = cache_dir / f"{tile_name}__{band}__hdu{self.fits_hdu}.pt"
            if not path.exists():
                missing.append(str(path))
                continue
            tensor = torch.load(path, map_location="cpu").to(dtype=torch.float32)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 3 or tensor.shape[0] != 1:
                raise ValueError(f"single-band zscale cache must have shape [1,H,W] or [H,W]: {path} has {tuple(tensor.shape)}")
            tensors.append(tensor[0])
        if missing:
            return None
        if not tensors:
            return None
        return torch.stack(tensors, dim=0).to(dtype=torch.float32)

    def _load_or_make_image(self, rec: CutoutRecord) -> Tensor:
        cache_path = self._image_cache_path(rec)
        if cache_path is not None and cache_path.exists():
            return torch.load(cache_path, map_location="cpu").to(dtype=torch.float32)
        single_band_cache = self._load_single_band_image_caches(rec)
        if single_band_cache is not None:
            return single_band_cache

        if any(Path(path).name == "__zscale_cache_only__.fits" for path in rec.image_paths):
            raise FileNotFoundError(
                f"No FITS cutouts are present for {rec.name}; expected precomputed zscale cache at {cache_path}. "
                "Set --image-cache-dir to the zscale root generated by astro_data_preprocessing.py."
            )

        image_np = read_fits_bands(rec.image_paths, hdu=self.fits_hdu)
        image = astro_zscale_preprocess(image_np).to(dtype=torch.float32)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
            torch.save(image.cpu(), tmp_path)
            try:
                tmp_path.replace(cache_path)
            except FileExistsError:
                tmp_path.unlink(missing_ok=True)
        return image

    def _load_or_make_targets(
        self,
        rec: CutoutRecord,
        *,
        image_shape: Tuple[int, int],
        catalog: Dict[str, np.ndarray],
        target_path: str = "",
    ) -> Dict[str, Tensor]:
        if target_path:
            path = Path(target_path)
            if path.exists():
                return _read_target_npz(path)
        if self.targets_dir is not None:
            for target_path in self._target_path_candidates(rec):
                if not target_path.exists():
                    continue
                return _read_target_npz(target_path)
        elif rec.target_path:
            target_path = Path(rec.target_path)
            if target_path.exists():
                return _read_target_npz(target_path)

        return _target_defaults(make_targets(
            image_shape=image_shape,
            centers=catalog["centers"],
            moments=catalog["moments"],
            kron_radius=catalog["kron_radius"],
            confidence_levels=self.confidence_levels,
            ellipse_sigma=self.ellipse_sigma,
            core_radius=self.core_radius,
            shape_source=self.shape_source,
        ))

    def _target_path_candidates(self, rec: CutoutRecord) -> List[Path]:
        if self.targets_dir is None:
            return []
        candidates: List[Path] = []
        tile_name = rec.tile_name or Path(rec.name).name
        if rec.target_path:
            candidates.append(Path(rec.target_path))
        if rec.relative_root:
            candidates.append(self.targets_dir / rec.relative_root / "targets" / f"{tile_name}.npz")
            candidates.append(self.targets_dir / rec.relative_root / f"{tile_name}.npz")
        candidates.append(self.targets_dir / f"{rec.name}.npz")
        candidates.append(self.targets_dir / f"{tile_name}.npz")

        out: List[Path] = []
        seen: set[Path] = set()
        for path in candidates:
            resolved = path.expanduser()
            if resolved in seen:
                continue
            out.append(resolved)
            seen.add(resolved)
        return out


def collate_cutouts(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "image": torch.stack([item["image"] for item in batch]),  # type: ignore[index]
        "seg": torch.stack([item["seg"] for item in batch]),  # type: ignore[index]
        "confidence": torch.stack([item["confidence"] for item in batch]),  # type: ignore[index]
        "shape": torch.stack([item["shape"] for item in batch]),  # type: ignore[index]
        "shape_weight": torch.stack([item["shape_weight"] for item in batch]),  # type: ignore[index]
        "confidence_weight": torch.stack([item["confidence_weight"] for item in batch]),  # type: ignore[index]
        "seg_loss_weight": torch.stack([item["seg_loss_weight"] for item in batch]),  # type: ignore[index]
        "clean_mask": torch.stack([item["clean_mask"] for item in batch]),  # type: ignore[index]
        "center_only_mask": torch.stack([item["center_only_mask"] for item in batch]),  # type: ignore[index]
        "ignore_mask": torch.stack([item["ignore_mask"] for item in batch]),  # type: ignore[index]
        "strict_center_only_mask": torch.stack([item["strict_center_only_mask"] for item in batch]),  # type: ignore[index]
        "strict_ignore_mask": torch.stack([item["strict_ignore_mask"] for item in batch]),  # type: ignore[index]
        "source_union_mask": torch.stack([item["source_union_mask"] for item in batch]),  # type: ignore[index]
        "background_mask": torch.stack([item["background_mask"] for item in batch]),  # type: ignore[index]
        "pu_class_mask": torch.stack([item["pu_class_mask"] for item in batch]),  # type: ignore[index]
        "pseudo_mask": torch.stack([item["pseudo_mask"] for item in batch]),  # type: ignore[index]
        "band_seg": torch.stack([item["band_seg"] for item in batch]),  # type: ignore[index]
        "band_confidence": torch.stack([item["band_confidence"] for item in batch]),  # type: ignore[index]
        "band_shape": torch.stack([item["band_shape"] for item in batch]),  # type: ignore[index]
        "band_shape_weight": torch.stack([item["band_shape_weight"] for item in batch]),  # type: ignore[index]
        "band_confidence_weight": torch.stack([item["band_confidence_weight"] for item in batch]),  # type: ignore[index]
        "band_seg_loss_weight": torch.stack([item["band_seg_loss_weight"] for item in batch]),  # type: ignore[index]
        "band_clean_mask": torch.stack([item["band_clean_mask"] for item in batch]),  # type: ignore[index]
        "band_center_only_mask": torch.stack([item["band_center_only_mask"] for item in batch]),  # type: ignore[index]
        "band_ignore_mask": torch.stack([item["band_ignore_mask"] for item in batch]),  # type: ignore[index]
        "band_strict_center_only_mask": torch.stack([item["band_strict_center_only_mask"] for item in batch]),  # type: ignore[index]
        "band_strict_ignore_mask": torch.stack([item["band_strict_ignore_mask"] for item in batch]),  # type: ignore[index]
        "band_source_union_mask": torch.stack([item["band_source_union_mask"] for item in batch]),  # type: ignore[index]
        "band_background_mask": torch.stack([item["band_background_mask"] for item in batch]),  # type: ignore[index]
        "band_pu_class_mask": torch.stack([item["band_pu_class_mask"] for item in batch]),  # type: ignore[index]
        "band_pseudo_mask": torch.stack([item["band_pseudo_mask"] for item in batch]),  # type: ignore[index]
        "centers": [item["centers"] for item in batch],
        "ids": [item["ids"] for item in batch],
        "ignore_centers": [item["ignore_centers"] for item in batch],
        "strict_center_only_centers": [item["strict_center_only_centers"] for item in batch],
        "strict_ignore_centers": [item["strict_ignore_centers"] for item in batch],
        "band_centers": [item["band_centers"] for item in batch],
        "band_ids": [item["band_ids"] for item in batch],
        "band_ignore_centers": [item["band_ignore_centers"] for item in batch],
        "band_strict_center_only_centers": [item["band_strict_center_only_centers"] for item in batch],
        "band_strict_ignore_centers": [item["band_strict_ignore_centers"] for item in batch],
        "band_rejected_ids": [item["band_rejected_ids"] for item in batch],
        "name": [item["name"] for item in batch],
        "tile_name": [item["tile_name"] for item in batch],
        "tract": [item["tract"] for item in batch],
        "patch": [item["patch"] for item in batch],
        "relative_root": [item["relative_root"] for item in batch],
        "x0": [item["x0"] for item in batch],
        "y0": [item["y0"] for item in batch],
        "image_paths": [item["image_paths"] for item in batch],
    }
def split_records(
    records: Sequence[CutoutRecord],
    val_fraction: float,
    seed: int,
    *,
    fixed_val_names: Sequence[str] = (),
    patch_val: bool = False,
) -> Tuple[List[CutoutRecord], List[CutoutRecord]]:
    rng = random.Random(seed)
    if patch_val:
        patch_to_records: Dict[Tuple[str, str], List[CutoutRecord]] = {}
        for rec in records:
            key = (rec.tract, rec.patch)
            patch_to_records.setdefault(key, []).append(rec)
        patches = list(patch_to_records.keys())
        rng.shuffle(patches)
        target_n_val = max(1, int(round(len(records) * val_fraction))) if len(records) > 1 else 0
        n_val_patches = max(1, int(round(len(patches) * val_fraction))) if len(patches) > 1 else 0
        val_patches = set(patches[:n_val_patches])
        selected = [f"{tract}/{patch}" if tract else patch for tract, patch in patches[:n_val_patches]]
        print(f"Patches selected for validation: {selected}")
        train = [rec for rec in records if (rec.tract, rec.patch) not in val_patches]
        val = [rec for rec in records if (rec.tract, rec.patch) in val_patches]
        return train, val
    fixed_names = {name for name in fixed_val_names if name}
    fixed_val = [rec for rec in records if _record_matches_any_name(rec, fixed_names)]
    remaining = [rec for rec in records if not _record_matches_any_name(rec, fixed_names)]
    shuffled = list(remaining)
    rng.shuffle(shuffled)
    target_n_val = max(1, int(round(len(records) * val_fraction))) if len(records) > 1 else 0
    n_random_val = max(0, target_n_val - len(fixed_val))
    random_val = shuffled[:n_random_val]
    train = shuffled[n_random_val:]
    return train, fixed_val + random_val


def _record_name_aliases(rec: CutoutRecord) -> set[str]:
    aliases = {rec.name}
    if rec.tile_name:
        aliases.add(rec.tile_name)
    basename = Path(rec.name).name
    if basename:
        aliases.add(basename)
    return aliases


def _record_matches_any_name(rec: CutoutRecord, names: set[str]) -> bool:
    return bool(_record_name_aliases(rec) & names)
