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


@dataclass(frozen=True)
class CutoutRecord:
    name: str
    image_paths: Tuple[str, ...]
    meas_path: str
    x0: int
    y0: int
    band_meas_paths: Tuple[str, ...] = ()
    band_rejected_paths: Tuple[str, ...] = ()
    tile_name: str = ""
    tract: str = ""
    patch: str = ""
    relative_root: str = ""
    target_path: str = ""


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
            x0, y0 = _parse_tile_origin(tile_name)
            target_path = patch_root / "targets" / f"{tile_name}.npz"
            records.append(
                CutoutRecord(
                    name=_record_name(tile_name, relative_root),
                    image_paths=image_paths,
                    meas_path=str(meas_path),
                    x0=x0,
                    y0=y0,
                    band_meas_paths=band_meas_paths,
                    band_rejected_paths=band_rejected_paths,
                    tile_name=tile_name,
                    tract=tract,
                    patch=patch,
                    relative_root=relative_root,
                    target_path=str(target_path) if target_path.exists() else "",
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
        dist = torch.abs(xx_full[cy0:cy1, cx0:cx1] - cx_i) + torch.abs(yy_full[cy0:cy1, cx0:cx1] - cy_i)
        vals = torch.clamp(level_radius - dist, min=0)
        conf[cy0:cy1, cx0:cx1] = torch.maximum(conf[cy0:cy1, cx0:cx1], vals.long())

        ky0, ky1 = max(0, cy_i - core_radius), min(h, cy_i + core_radius + 1)
        kx0, kx1 = max(0, cx_i - core_radius), min(w, cx_i + core_radius + 1)
        core = (xx_full[ky0:ky1, kx0:kx1] - cx_i).abs() + (yy_full[ky0:ky1, kx0:kx1] - cy_i).abs()
        seg[ky0:ky1, kx0:kx1][core <= core_radius] = 2

    return {"seg": seg, "confidence": conf, "shape": shape, "shape_weight": shape_weight}


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
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        rec = self.records[idx]
        image = self._load_or_make_image(rec)
        h, w = int(image.shape[-2]), int(image.shape[-1])
        catalog = load_meas_catalog(
            rec.meas_path,
            x0=rec.x0,
            y0=rec.y0,
            image_shape=(h, w),
            source_filter=self.source_filter,
        )
        targets = self._load_or_make_targets(rec, image_shape=(h, w), catalog=catalog)
        band_catalogs: List[Dict[str, np.ndarray]] = []
        band_targets: List[Dict[str, Tensor]] = []
        band_meas_paths = rec.band_meas_paths if rec.band_meas_paths else tuple(rec.meas_path for _ in range(image.shape[0]))
        for band_idx, meas_path in enumerate(band_meas_paths):
            band_catalog = load_meas_catalog(
                meas_path,
                x0=rec.x0,
                y0=rec.y0,
                image_shape=(h, w),
                source_filter=self.source_filter,
            )
            band_catalogs.append(band_catalog)
            band_targets.append(
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
        centers = torch.from_numpy(catalog["centers"])
        band_centers = [torch.from_numpy(item["centers"]) for item in band_catalogs]
        band_ids = [torch.from_numpy(item["ids"]) for item in band_catalogs]
        if rec.band_rejected_paths:
            band_rejected_ids = [torch.from_numpy(load_catalog_ids(path)) for path in rec.band_rejected_paths]
        else:
            band_rejected_ids = [torch.empty((0,), dtype=torch.long) for _ in band_meas_paths]
        if self.augment and random.random() < 0.5:
            image = torch.flip(image, dims=(-1,))
            targets["seg"] = torch.flip(targets["seg"], dims=(-1,))
            targets["confidence"] = torch.flip(targets["confidence"], dims=(-1,))
            targets["shape"] = torch.flip(targets["shape"], dims=(-1,))
            targets["shape_weight"] = torch.flip(targets["shape_weight"], dims=(-1,))
            for target in band_targets:
                target["seg"] = torch.flip(target["seg"], dims=(-1,))
                target["confidence"] = torch.flip(target["confidence"], dims=(-1,))
                target["shape"] = torch.flip(target["shape"], dims=(-1,))
                target["shape_weight"] = torch.flip(target["shape_weight"], dims=(-1,))
            centers = centers.clone()
            centers[:, 0] = float(w - 1) - centers[:, 0]
            band_centers = [center.clone() for center in band_centers]
            for center in band_centers:
                center[:, 0] = float(w - 1) - center[:, 0]

        return {
            "image": image,
            "seg": targets["seg"],
            "confidence": targets["confidence"],
            "shape": targets["shape"],
            "shape_weight": targets["shape_weight"],
            "band_seg": torch.stack([target["seg"] for target in band_targets]),
            "band_confidence": torch.stack([target["confidence"] for target in band_targets]),
            "band_shape": torch.stack([target["shape"] for target in band_targets]),
            "band_shape_weight": torch.stack([target["shape_weight"] for target in band_targets]),
            "centers": centers,
            "ids": torch.from_numpy(catalog["ids"]),
            "band_centers": band_centers,
            "band_ids": band_ids,
            "band_rejected_ids": band_rejected_ids,
            "name": rec.name,
        }

    def _image_cache_path(self, rec: CutoutRecord) -> Optional[Path]:
        if self.image_cache_dir is None:
            return None
        band_key = "_".join(Path(path).parent.name for path in rec.image_paths)
        tile_name = rec.tile_name or Path(rec.name).name
        cache_dir = self.image_cache_dir
        if rec.relative_root:
            cache_dir = cache_dir / rec.relative_root / "cutouts"
        return cache_dir / f"{tile_name}__{band_key}__hdu{self.fits_hdu}.pt"

    def _load_or_make_image(self, rec: CutoutRecord) -> Tensor:
        cache_path = self._image_cache_path(rec)
        if cache_path is not None and cache_path.exists():
            return torch.load(cache_path, map_location="cpu").to(dtype=torch.float32)

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
    ) -> Dict[str, Tensor]:
        if self.targets_dir is not None:
            for target_path in self._target_path_candidates(rec):
                if not target_path.exists():
                    continue
                with np.load(target_path) as data:
                    seg = torch.from_numpy(np.asarray(data["seg"], dtype=np.int64))
                    confidence = torch.from_numpy(np.asarray(data["confidence"], dtype=np.int64))
                    shape = torch.from_numpy(np.asarray(data["shape"], dtype=np.float32))
                    shape_weight = torch.from_numpy(np.asarray(data["shape_weight"], dtype=np.float32))
                return {
                    "seg": seg,
                    "confidence": confidence,
                    "shape": shape,
                    "shape_weight": shape_weight,
                }
        elif rec.target_path:
            target_path = Path(rec.target_path)
            if target_path.exists():
                with np.load(target_path) as data:
                    seg = torch.from_numpy(np.asarray(data["seg"], dtype=np.int64))
                    confidence = torch.from_numpy(np.asarray(data["confidence"], dtype=np.int64))
                    shape = torch.from_numpy(np.asarray(data["shape"], dtype=np.float32))
                    shape_weight = torch.from_numpy(np.asarray(data["shape_weight"], dtype=np.float32))
                return {
                    "seg": seg,
                    "confidence": confidence,
                    "shape": shape,
                    "shape_weight": shape_weight,
                }

        return make_targets(
            image_shape=image_shape,
            centers=catalog["centers"],
            moments=catalog["moments"],
            kron_radius=catalog["kron_radius"],
            confidence_levels=self.confidence_levels,
            ellipse_sigma=self.ellipse_sigma,
            core_radius=self.core_radius,
            shape_source=self.shape_source,
        )

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
        "band_seg": torch.stack([item["band_seg"] for item in batch]),  # type: ignore[index]
        "band_confidence": torch.stack([item["band_confidence"] for item in batch]),  # type: ignore[index]
        "band_shape": torch.stack([item["band_shape"] for item in batch]),  # type: ignore[index]
        "band_shape_weight": torch.stack([item["band_shape_weight"] for item in batch]),  # type: ignore[index]
        "centers": [item["centers"] for item in batch],
        "ids": [item["ids"] for item in batch],
        "band_centers": [item["band_centers"] for item in batch],
        "band_ids": [item["band_ids"] for item in batch],
        "band_rejected_ids": [item["band_rejected_ids"] for item in batch],
        "name": [item["name"] for item in batch],
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
