"""
Train and evaluate the 2D CELLECT-style astronomy model on LSST/HSC FITS cutouts.

The dense losses intentionally keep CELLECT's original constants:
  - segmentation class weights [1, 32, 1];
  - ordinal confidence positive class weight 32;
  - astronomy-specific center localization is normalized by the astrometric
    tolerance, default 0.5 arcsec / 0.168 arcsec per pixel;
  - optional embedding/matcher losses keep the original outer multiplier 10.

Division supervision is omitted because AstroUNet2D does not output a division
branch. EX/EN modules are saved with the checkpoint, but they are only trained
when pair labels are available; the provided single-epoch cutout catalogs mainly
supervise segmentation, confidence, shape, and embeddings.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# For debugging
import time

from astro_cellect2d import (
    AstroMatchNet2D,
    AstroUNet2D,
    FusedEncoderMultiBandAstroCELLECT2D,
    MultiBandAstroCELLECT2D,
    astro_zscale_preprocess,
    ordinal_confidence_loss,
    read_fits_bands,
)
from astro_match_eval import (
    matcher_classification_loss as vectorized_matcher_classification_loss,
    parse_ex_band_pairs as parse_matcher_ex_band_pairs,
)


@dataclass(frozen=True)
class LossWeights:
    """CELLECT constants carried into the 2D astronomy training loop."""

    segmentation_3cls: Tuple[float, float, float] = (1.0, 32.0, 1.0)
    confidence_pos_weight: float = 32.0
    center_position: float = 1.0
    triplet_margin: float = 0.3
    triplet_outer_weight: float = 10.0
    matcher_outer_weight: float = 10.0


@dataclass(frozen=True)
class CutoutRecord:
    name: str
    image_paths: Tuple[str, ...]
    meas_path: str
    x0: int
    y0: int
    band_meas_paths: Tuple[str, ...] = ()


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


def _find_band_meas(reference_root: Path, band: str) -> str:
    band_dir = reference_root / band
    candidates = sorted(band_dir.glob(f"meas-{band}-*.fits")) if band_dir.exists() else []
    if not candidates:
        candidates = sorted(reference_root.glob(f"**/meas-{band}-*.fits"))
    if not candidates:
        raise FileNotFoundError(f"No meas catalog found for band {band} under {reference_root}")
    return str(candidates[0])


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

    reference_dir = reference_dir or (root / "reference_catalogs")
    cutout_dir = cutout_dir or (root / "cutouts")
    if not reference_dir.exists():
        raise FileNotFoundError(f"Reference catalog directory does not exist: {reference_dir}")
    if not cutout_dir.exists():
        raise FileNotFoundError(f"Cutout image directory does not exist: {cutout_dir}")

    records: List[CutoutRecord] = []
    for meas_path in sorted(reference_dir.glob("*_meas.fits")):
        tile_name = meas_path.name[: -len("_meas.fits")]
        tile_dir = cutout_dir / tile_name
        if not tile_dir.exists():
            continue
        band_order = list(bands) if bands else _default_band_order(tile_dir)
        if not band_order:
            continue
        image_paths = tuple(_find_band_fits(tile_dir, band) for band in band_order)
        band_meas_paths = tuple(_find_band_meas(band_reference_root, band) for band in band_order) if band_reference_root else ()
        x0, y0 = _parse_tile_origin(tile_name)
        records.append(
            CutoutRecord(
                name=tile_name,
                image_paths=image_paths,
                meas_path=str(meas_path),
                x0=x0,
                y0=y0,
                band_meas_paths=band_meas_paths,
            )
        )
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
            "name": rec.name,
        }

    def _image_cache_path(self, rec: CutoutRecord) -> Optional[Path]:
        if self.image_cache_dir is None:
            return None
        band_key = "_".join(Path(path).parent.name for path in rec.image_paths)
        return self.image_cache_dir / f"{rec.name}__{band_key}__hdu{self.fits_hdu}.pt"

    def _load_or_make_image(self, rec: CutoutRecord) -> Tensor:
        cache_path = self._image_cache_path(rec)
        if cache_path is not None and cache_path.exists():
            return torch.load(cache_path, map_location="cpu").to(dtype=torch.float32)

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
            target_path = self.targets_dir / f"{rec.name}.npz"
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
        "name": [item["name"] for item in batch],
    }


class HardTripletLoss(nn.Module):
    """CELLECT/Open-ReID style hard triplet loss with margin 0.3.

    Positives are repeated object IDs, usually the same source appearing in
    overlapping tiles. Negatives can be constrained with group labels; for the
    HSC single-band setup this keeps negatives inside the same 512x512 image.
    """

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, features: Tensor, labels: Tensor, groups: Optional[Tensor] = None) -> Tensor:
        labels = labels.long()
        valid = labels > 0
        if groups is not None:
            groups = groups.long()
            groups = groups[valid]
        features = features[valid]
        labels = labels[valid]
        if features.shape[0] < 4 or torch.unique(labels).numel() < 2:
            return features.sum() * 0.0
        dist = torch.cdist(features, features, p=2)
        same = labels[:, None] == labels[None, :]
        eye = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
        pos_mask = same & ~eye
        neg_mask = ~same
        if groups is not None:
            same_group = groups[:, None] == groups[None, :]
            neg_mask = neg_mask & same_group
        has_pos = pos_mask.any(dim=1)
        has_neg = neg_mask.any(dim=1)
        keep = has_pos & has_neg
        if not bool(keep.any()):
            return features.sum() * 0.0
        dist_ap = dist.masked_fill(~pos_mask, -1.0).max(dim=1).values[keep]
        dist_an = dist.masked_fill(~neg_mask, float("inf")).min(dim=1).values[keep]
        target = torch.ones_like(dist_an)
        return self.margin_loss(dist_an, dist_ap, target)


def dense_losses(
    outputs: Dict[str, Tensor],
    batch: Dict[str, object],
    *,
    weights: LossWeights,
    device: torch.device,
    center_radius_px: float,
) -> Dict[str, Tensor]:
    # Debug
    # torch.cuda.synchronize()
    # start_time = time.time()
    seg_target = batch["seg"].to(device=device, dtype=torch.long)  # type: ignore[union-attr]
    conf_target = batch["confidence"].to(device=device, dtype=torch.long)  # type: ignore[union-attr]
    shape_target = batch["shape"].to(device=device, dtype=torch.float32)  # type: ignore[union-attr]
    shape_weight = batch["shape_weight"].to(device=device, dtype=torch.float32)  # type: ignore[union-attr]

    seg_weight = torch.tensor(weights.segmentation_3cls, device=device, dtype=torch.float32)
    seg_loss = F.cross_entropy(outputs["seg_logits"], seg_target, weight=seg_weight)
    # torch.cuda.synchronize()
    # seg_time = time.time()
    # print(f'[DEBUG] Segmentation loss computed in {seg_time - start_time:.3f} seconds.')
    conf_loss = ordinal_confidence_loss(
        outputs["confidence"],
        conf_target,
        pos_weight=weights.confidence_pos_weight,
    )
    # torch.cuda.synchronize()
    # conf_time = time.time()
    # print(f'[DEBUG] Confidence loss computed in {conf_time - seg_time:.3f} seconds.')
    per_pixel_shape = F.mse_loss(outputs["shape"], shape_target, reduction="none").mean(dim=1)
    if bool((shape_weight > 0).any()):
        shape_loss = (per_pixel_shape * shape_weight).sum() / shape_weight.sum().clamp_min(1.0)
    else:
        shape_loss = per_pixel_shape.mean() * 0.0
    # torch.cuda.synchronize()
    # shape_time = time.time()
    # print(f'[DEBUG] Shape loss computed in {shape_time - conf_time:.3f} seconds.')
    if weights.center_position > 0.0:
        center_loss = center_localization_loss(outputs, batch["centers"], radius_px=center_radius_px)
    else:
        center_loss = torch.tensor(0.0, device=device)
    # torch.cuda.synchronize()
    # center_time = time.time()
    # print(f'[DEBUG] Center localization loss computed in {center_time - shape_time:.3f} seconds.')
    total = seg_loss + conf_loss + shape_loss + weights.center_position * center_loss
    return {"total": total, "seg": seg_loss, "confidence": conf_loss, "shape": shape_loss, "center": center_loss}


def _flatten_per_band_outputs(outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Convert [B, C, ...] per-band outputs to [B*C, ...] dense-loss layout."""

    if outputs["seg_logits"].ndim != 5:
        return outputs
    return {key: value.reshape(value.shape[0] * value.shape[1], *value.shape[2:]) for key, value in outputs.items()}


def _flatten_band_centers(nested: Sequence[Sequence[Tensor]]) -> List[Tensor]:
    return [centers for item in nested for centers in item]


def dense_losses_any(
    outputs: Dict[str, Tensor],
    batch: Dict[str, object],
    *,
    weights: LossWeights,
    device: torch.device,
    center_radius_px: float,
) -> Dict[str, Tensor]:
    """Dense loss for both fused BCHW outputs and per-band B,C,CHW outputs."""

    if outputs["seg_logits"].ndim != 5:
        return dense_losses(outputs, batch, weights=weights, device=device, center_radius_px=center_radius_px)

    flat_outputs = _flatten_per_band_outputs(outputs)
    flat_batch = {
        "seg": batch["band_seg"].reshape(-1, *batch["band_seg"].shape[2:]),  # type: ignore[union-attr]
        "confidence": batch["band_confidence"].reshape(-1, *batch["band_confidence"].shape[2:]),  # type: ignore[union-attr]
        "shape": batch["band_shape"].reshape(-1, *batch["band_shape"].shape[2:]),  # type: ignore[union-attr]
        "shape_weight": batch["band_shape_weight"].reshape(-1, *batch["band_shape_weight"].shape[2:]),  # type: ignore[union-attr]
        "centers": _flatten_band_centers(batch["band_centers"]),  # type: ignore[arg-type]
    }
    return dense_losses(flat_outputs, flat_batch, weights=weights, device=device, center_radius_px=center_radius_px)


def center_localization_loss(outputs: Dict[str, Tensor], centers_list: Sequence[Tensor], *, radius_px: float) -> Tensor:
    """Differentiable center-position loss on the top confidence channel.

    The astronomy data has one 2D catalog center shared by all input bands. This
    differs from CELLECT microscopy, where the center is a point in 3D space. For
    each catalog center we take a local window on the highest confidence channel,
    compute a soft-argmax position, and penalize the offset normalized by the
    astrometric matching tolerance, by default 0.5 arcsec / 0.168 arcsec per px.
    """

    conf = outputs["confidence"][:, -1]
    h, w = conf.shape[-2:]
    radius = max(float(radius_px), 1e-6)
    window = max(1, int(math.ceil(radius)))
    losses: List[Tensor] = []
    for b, centers in enumerate(centers_list):
        if centers.numel() == 0:
            continue
        centers_f = centers.to(device=conf.device, dtype=conf.dtype)
        for center in centers_f:
            cx, cy = center[0], center[1]
            xi = int(round(float(cx.detach().cpu())))
            yi = int(round(float(cy.detach().cpu())))
            x0, x1 = max(0, xi - window), min(w, xi + window + 1)
            y0, y1 = max(0, yi - window), min(h, yi + window + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            patch = conf[b, y0:y1, x0:x1]
            prob = F.softmax(patch.reshape(-1), dim=0)
            xs = torch.arange(x0, x1, device=conf.device, dtype=conf.dtype)
            ys = torch.arange(y0, y1, device=conf.device, dtype=conf.dtype)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            pred_x = (prob * xx.reshape(-1)).sum()
            pred_y = (prob * yy.reshape(-1)).sum()
            pred_xy = torch.stack((pred_x, pred_y)) / radius
            target_xy = torch.stack((cx, cy)) / radius
            losses.append(F.smooth_l1_loss(pred_xy, target_xy, reduction="sum"))
    if not losses:
        return conf.sum() * 0.0
    return torch.stack(losses).mean()


def _sample_embeddings_at_centers(outputs: Dict[str, Tensor], centers_list: Sequence[Tensor]) -> Tuple[Tensor, Tensor]:
    feats: List[Tensor] = []
    batch_ids: List[Tensor] = []
    emb = outputs["embedding"]
    h, w = emb.shape[-2:]
    for b, centers in enumerate(centers_list):
        if centers.numel() == 0:
            continue
        xy = centers.to(device=emb.device, dtype=torch.long)
        x = xy[:, 0].clamp(0, w - 1)
        y = xy[:, 1].clamp(0, h - 1)
        feats.append(emb[b, :, y, x].transpose(0, 1))
        batch_ids.append(torch.full((xy.shape[0],), b, dtype=torch.long, device=emb.device))
    if not feats:
        return emb.new_zeros((0, emb.shape[1])), emb.new_zeros((0,), dtype=torch.long)
    return torch.cat(feats, dim=0), torch.cat(batch_ids, dim=0)


def _sample_map_at_centers(map_tensor: Tensor, centers: Tensor) -> Tensor:
    """Sample [C,H,W] tensor at x,y center coordinates -> [N,C]."""

    if centers.numel() == 0:
        return map_tensor.new_zeros((0, map_tensor.shape[0]))
    h, w = map_tensor.shape[-2:]
    xy = centers.to(device=map_tensor.device, dtype=torch.long)
    x = xy[:, 0].clamp(0, w - 1)
    y = xy[:, 1].clamp(0, h - 1)
    return map_tensor[:, y, x].transpose(0, 1)


def _confidence_score_at_centers(outputs: Dict[str, Tensor], centers: Tensor, *, batch_index: int = 0) -> Tensor:
    """Sample CELLECT-smoothed top confidence channel at detected centers."""

    if centers.numel() == 0:
        return outputs["confidence"].new_zeros((0,))
    confidence = outputs["confidence"]
    if confidence.ndim == 5:
        raise ValueError("_confidence_score_at_centers expects one dense band output")
    score_map = _cellect_confidence_smooth_2d(confidence)[batch_index, -1]
    h, w = score_map.shape[-2:]
    xy = centers.to(device=score_map.device, dtype=torch.long)
    x = xy[:, 0].clamp(0, w - 1)
    y = xy[:, 1].clamp(0, h - 1)
    return score_map[y, x]


@torch.no_grad()
def en_deduplicate_centers(
    en_net: AstroMatchNet2D,
    outputs: Dict[str, Tensor],
    centers_xy: np.ndarray,
    *,
    batch_index: int = 0,
    candidate_count: int = 5,
    offset_scale: float = 1.0,
    same_threshold: float = 0.6,
    strong_threshold: float = 0.9999,
    use_group_mean: bool = True,
) -> np.ndarray:
    """CELLECT-style intra-frame EN post-processing for detected centers.

    CELLECT does not stop at raw confidence local maxima.  During inference it
    samples features at detected centers, finds five nearest same-frame
    neighbors, applies EN, and groups centers when EN says they are the same
    object.  This helper implements the same idea for one 2D band.
    """

    centers = torch.as_tensor(np.asarray(centers_xy, dtype=np.float32), device=outputs["embedding"].device)
    if centers.ndim != 2 or centers.shape[0] <= 1:
        return np.asarray(centers_xy, dtype=np.float32)
    n = centers.shape[0]
    k = min(int(candidate_count), max(n - 1, 1))

    emb_map = outputs["embedding"][batch_index]
    shape_map = outputs["shape"][batch_index]
    features = _sample_map_at_centers(emb_map, centers)
    shapes = _sample_map_at_centers(shape_map, centers)
    center_scores = _confidence_score_at_centers(outputs, centers, batch_index=batch_index)

    dist = torch.cdist(centers, centers, p=2)
    dist.fill_diagonal_(float("inf"))
    nn_dist, nn_idx = torch.topk(dist, k=k, largest=False)
    if k < int(candidate_count):
        pad = int(candidate_count) - k
        nn_idx = torch.cat([nn_idx, nn_idx[:, -1:].repeat(1, pad)], dim=1)
        nn_dist = torch.cat([nn_dist, nn_dist[:, -1:].repeat(1, pad)], dim=1)

    cand_features = features[nn_idx]
    cand_xy = centers[nn_idx]
    offsets = (cand_xy - centers[:, None, :]) / max(float(offset_scale), 1e-6)
    cand_shapes = shapes[nn_idx]
    shape_features = torch.cat([shapes[:, None, :].expand(-1, int(candidate_count), -1), cand_shapes], dim=-1)
    logits = en_net(features, cand_features, offsets, shape_features)
    prob = F.softmax(logits, dim=1)
    cand_prob = prob[:, :-1]
    none_prob = prob[:, -1:]

    suppressed: set[int] = set()
    groups: List[List[int]] = []
    for i in range(n):
        if i in suppressed:
            continue
        group = [i]
        major = float(shapes[i, 0].detach().cpu()) if shapes.shape[1] > 0 else 1.0
        local_radius = max(5.0, 0.6 * max(abs(major), 1.0))
        for j in range(int(candidate_count)):
            candidate = int(nn_idx[i, j].item())
            if candidate == i or candidate in suppressed:
                continue
            same = bool(cand_prob[i, j] > none_prob[i, 0])
            confident = bool(cand_prob[i, j] > strong_threshold)
            moderate_close = bool(cand_prob[i, j] > same_threshold and nn_dist[i, min(j, nn_dist.shape[1] - 1)] < local_radius)
            if same and (confident or moderate_close):
                group.append(candidate)
                suppressed.add(candidate)
        groups.append(group)

    dedup: List[np.ndarray] = []
    centers_cpu = centers.detach().cpu().numpy().astype(np.float32)
    scores_cpu = center_scores.detach().cpu().numpy()
    for group in groups:
        if use_group_mean and len(group) > 1:
            dedup.append(centers_cpu[group].mean(axis=0))
        else:
            best = max(group, key=lambda idx: float(scores_cpu[idx]))
            dedup.append(centers_cpu[best])
    return np.stack(dedup, axis=0).astype(np.float32) if dedup else np.zeros((0, 2), dtype=np.float32)


@torch.no_grad()
def detect_centers_with_en(
    model: nn.Module,
    outputs: Dict[str, Tensor],
    *,
    threshold: float,
    nms_radius: int,
    confidence_score: str,
    match_radius: float,
    candidate_count: int,
    en_threshold: float,
) -> List[np.ndarray]:
    """Detect centers and optionally apply CELLECT-style EN deduplication."""

    if outputs["seg_logits"].ndim == 5:
        batch, bands = outputs["seg_logits"].shape[:2]
        flat_outputs = _flatten_per_band_outputs(outputs)
        raw = detect_centers(
            flat_outputs,
            threshold=threshold,
            nms_radius=nms_radius,
            confidence_score=confidence_score,
        )
        if not hasattr(model, "EN"):
            return raw
        result: List[np.ndarray] = []
        for b in range(batch):
            for band in range(bands):
                idx = b * bands + band
                one = {key: value[b, band].unsqueeze(0) for key, value in outputs.items()}
                result.append(
                    en_deduplicate_centers(
                        model.EN,
                        one,
                        raw[idx],
                        batch_index=0,
                        candidate_count=candidate_count,
                        offset_scale=match_radius,
                        same_threshold=en_threshold,
                    )
                )
        return result

    raw = detect_centers(
        outputs,
        threshold=threshold,
        nms_radius=nms_radius,
        confidence_score=confidence_score,
    )
    if not hasattr(model, "EN"):
        return raw
    return [
        en_deduplicate_centers(
            model.EN,
            outputs,
            centers,
            batch_index=batch_idx,
            candidate_count=candidate_count,
            offset_scale=match_radius,
            same_threshold=en_threshold,
        )
        for batch_idx, centers in enumerate(raw)
    ]


@torch.no_grad()
def detect_centers_with_ex_link(
    model: nn.Module,
    outputs: Dict[str, Tensor],
    *,
    threshold: float,
    nms_radius: int,
    confidence_score: str,
    match_radius: float,
    candidate_count: int,
    ex_threshold: float,
    use_en_postprocess: bool = False,
    en_threshold: float = 0.6,
    max_distance_factor: float = 6.0,
    band_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[List[np.ndarray], List[List[Dict[str, object]]]]:
    """CELLECT-style cross-band linking after per-band center detection.

    CELLECT uses EX after center extraction: each anchor detection searches its
    five nearest detections in the next frame and the MLP scores candidate links
    using center embedding, position offset, and size/division features.  The
    astronomy variant has no division branch, so EX links detections across
    bands using embedding, position offset, and predicted shape/size.  The
    returned predictions are object-level fused centers, while ``components``
    keeps the per-band members for diagnostics and catalog construction.
    """

    if outputs["seg_logits"].ndim != 5 or not hasattr(model, "EX"):
        if outputs["seg_logits"].ndim == 5:
            flat_outputs = _flatten_per_band_outputs(outputs)
            raw = detect_centers(
                flat_outputs,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
            )
            return raw, []
        if use_en_postprocess and hasattr(model, "EN"):
            return (
                detect_centers_with_en(
                    model,
                    outputs,
                    threshold=threshold,
                    nms_radius=nms_radius,
                    confidence_score=confidence_score,
                    match_radius=match_radius,
                    candidate_count=candidate_count,
                    en_threshold=en_threshold,
                ),
                [],
            )
        return (
            detect_centers(
                outputs,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
            ),
            [],
        )

    batch, bands = outputs["seg_logits"].shape[:2]
    flat_outputs = _flatten_per_band_outputs(outputs)
    raw = detect_centers(
        flat_outputs,
        threshold=threshold,
        nms_radius=nms_radius,
        confidence_score=confidence_score,
    )
    per_batch: List[List[np.ndarray]] = []
    for batch_idx in range(batch):
        item: List[np.ndarray] = []
        for band_idx in range(bands):
            flat_idx = batch_idx * bands + band_idx
            centers = raw[flat_idx]
            if use_en_postprocess and hasattr(model, "EN"):
                one = {key: value[batch_idx, band_idx].unsqueeze(0) for key, value in outputs.items()}
                centers = en_deduplicate_centers(
                    model.EN,
                    one,
                    centers,
                    batch_index=0,
                    candidate_count=candidate_count,
                    offset_scale=match_radius,
                    same_threshold=en_threshold,
                )
            item.append(np.asarray(centers, dtype=np.float32))
        per_batch.append(item)

    merged: List[np.ndarray] = []
    components_all: List[List[Dict[str, object]]] = []
    for batch_idx, centers_by_band in enumerate(per_batch):
        centers, components = _link_one_multiband_item(
            model.EX,
            outputs,
            centers_by_band,
            batch_index=batch_idx,
            candidate_count=candidate_count,
            offset_scale=match_radius,
            ex_threshold=ex_threshold,
            max_distance_factor=max_distance_factor,
            band_pairs=band_pairs,
        )
        merged.append(centers)
        components_all.append(components)
    return merged, components_all


@torch.no_grad()
def _link_one_multiband_item(
    ex_net: AstroMatchNet2D,
    outputs: Dict[str, Tensor],
    centers_by_band: Sequence[np.ndarray],
    *,
    batch_index: int,
    candidate_count: int,
    offset_scale: float,
    ex_threshold: float,
    max_distance_factor: float,
    band_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    device = outputs["embedding"].device
    bands = len(centers_by_band)
    node_band: List[int] = []
    node_index: List[int] = []
    node_xy: List[Tensor] = []
    node_feature: List[Tensor] = []
    node_shape: List[Tensor] = []
    node_score: List[Tensor] = []

    for band_idx, centers_np in enumerate(centers_by_band):
        centers = torch.as_tensor(np.asarray(centers_np, dtype=np.float32), device=device)
        if centers.ndim != 2 or centers.shape[0] == 0:
            continue
        emb_map = outputs["embedding"][batch_index, band_idx]
        shape_map = outputs["shape"][batch_index, band_idx]
        one = {key: value[batch_index, band_idx].unsqueeze(0) for key, value in outputs.items()}
        features = _sample_map_at_centers(emb_map, centers)
        shapes = _sample_map_at_centers(shape_map, centers)
        scores = _confidence_score_at_centers(one, centers, batch_index=0)
        for det_idx in range(centers.shape[0]):
            node_band.append(band_idx)
            node_index.append(det_idx)
            node_xy.append(centers[det_idx])
            node_feature.append(features[det_idx])
            node_shape.append(shapes[det_idx])
            node_score.append(scores[det_idx])

    if not node_xy:
        return np.zeros((0, 2), dtype=np.float32), []
    if len(node_xy) == 1 or bands <= 1:
        center = node_xy[0].detach().cpu().numpy()[None].astype(np.float32)
        return center, [{"members": [(node_band[0], node_index[0])], "score": float(node_score[0].detach().cpu())}]

    xy = torch.stack(node_xy)
    features = torch.stack(node_feature)
    shapes = torch.stack(node_shape)
    scores = torch.stack(node_score)
    node_band_t = torch.tensor(node_band, dtype=torch.long, device=device)
    k = int(candidate_count)
    edges: List[Tuple[float, float, int, int]] = []
    allowed_pairs = {(int(src), int(dst)) for src, dst in band_pairs} if band_pairs is not None else None

    for anchor in range(xy.shape[0]):
        other = node_band_t != node_band_t[anchor]
        if allowed_pairs is not None:
            allowed_dst = torch.tensor(
                [dst for src, dst in allowed_pairs if src == int(node_band_t[anchor].item())],
                dtype=torch.long,
                device=device,
            )
            if allowed_dst.numel() == 0:
                continue
            other = other & torch.isin(node_band_t, allowed_dst)
        if not bool(other.any()):
            continue
        candidate_ids = torch.nonzero(other, as_tuple=False).flatten()
        dist = torch.linalg.norm(xy[candidate_ids] - xy[anchor][None, :], dim=1)
        order = torch.argsort(dist)[: min(k, candidate_ids.numel())]
        selected = candidate_ids[order]
        selected_dist = dist[order]
        if selected.numel() == 0:
            continue
        if selected.numel() < k:
            pad = k - int(selected.numel())
            selected = torch.cat([selected, selected[-1:].repeat(pad)], dim=0)
            selected_dist = torch.cat([selected_dist, selected_dist[-1:].repeat(pad)], dim=0)

        cand_features = features[selected].unsqueeze(0)
        cand_xy = xy[selected].unsqueeze(0)
        offsets = (cand_xy - xy[anchor].view(1, 1, 2)) / max(float(offset_scale), 1e-6)
        cand_shapes = shapes[selected].unsqueeze(0)
        shape_features = torch.cat([shapes[anchor].view(1, 1, -1).expand(1, k, -1), cand_shapes], dim=-1)
        ex_was_training = ex_net.training
        if ex_was_training:
            ex_net.eval()
        logits = ex_net(features[anchor].view(1, -1), cand_features, offsets, shape_features)
        if ex_was_training:
            ex_net.train()
        prob = F.softmax(logits, dim=1)[0]
        cand_prob = prob[:k]
        none_prob = prob[k]
        best = int(torch.argmax(cand_prob).item())
        best_node = int(selected[best].item())
        best_score = float(cand_prob[best].detach().cpu())
        best_dist = float(selected_dist[best].detach().cpu())
        major = max(float(abs(shapes[anchor, 0].detach().cpu())) if shapes.shape[1] > 0 else 1.0, 1.0)
        distance_limit = max(float(offset_scale), max_distance_factor * max(major, 1.0))
        if best_score > float(ex_threshold) and cand_prob[best] > none_prob and best_dist <= distance_limit:
            lo, hi = sorted((anchor, best_node))
            edges.append((best_score, best_dist, lo, hi))

    parent = list(range(len(node_xy)))
    component_bands: List[set[int]] = [{band} for band in node_band]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for score, dist, a, b in sorted(edges, key=lambda item: (-item[0], item[1])):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if component_bands[ra] & component_bands[rb]:
            continue
        parent[rb] = ra
        component_bands[ra] |= component_bands[rb]

    grouped: Dict[int, List[int]] = {}
    for idx in range(len(node_xy)):
        grouped.setdefault(find(idx), []).append(idx)

    components: List[Dict[str, object]] = []
    fused_centers: List[np.ndarray] = []
    for members in grouped.values():
        member_scores = scores[members].detach()
        weights = torch.clamp(member_scores, min=0)
        if not bool((weights > 0).any()):
            weights = torch.ones_like(weights)
        member_xy = xy[members]
        fused = (member_xy * weights[:, None]).sum(dim=0) / weights.sum().clamp_min(1e-6)
        fused_centers.append(fused.detach().cpu().numpy().astype(np.float32))
        components.append(
            {
                "members": [(int(node_band[i]), int(node_index[i])) for i in members],
                "center": fused.detach().cpu().numpy().astype(np.float32).tolist(),
                "score": float(member_scores.max().detach().cpu()),
            }
        )

    order = np.argsort([item["score"] for item in components])[::-1] if components else []
    components = [components[int(i)] for i in order]
    fused_array = (
        np.stack([fused_centers[int(i)] for i in order], axis=0).astype(np.float32)
        if len(order) > 0
        else np.zeros((0, 2), dtype=np.float32)
    )
    return fused_array, components


def _sample_multiband_sources(
    outputs: Dict[str, Tensor],
    batch: Dict[str, object],
) -> List[List[Dict[str, Tensor]]]:
    """Sample per-source features for every item and band.

    Returns a nested list [batch][band] with ``xy``, ``ids``, ``features`` and
    ``shape`` tensors.  This is the training-time bridge from dense UNet maps to
    CELLECT-style EX/EN candidate classification.
    """

    emb = outputs["embedding"]
    shape = outputs["shape"]
    if emb.ndim != 5:
        raise ValueError("multiband source sampling requires outputs with shape [B,C,...]")
    nested_centers: Sequence[Sequence[Tensor]] = batch["band_centers"]  # type: ignore[assignment]
    nested_ids: Sequence[Sequence[Tensor]] = batch["band_ids"]  # type: ignore[assignment]
    sampled: List[List[Dict[str, Tensor]]] = []
    for b in range(emb.shape[0]):
        per_band: List[Dict[str, Tensor]] = []
        for band in range(emb.shape[1]):
            centers = nested_centers[b][band].to(device=emb.device, dtype=torch.float32)
            ids = nested_ids[b][band].to(device=emb.device, dtype=torch.long)
            per_band.append(
                {
                    "xy": centers,
                    "ids": ids,
                    "features": _sample_map_at_centers(emb[b, band], centers),
                    "shape": _sample_map_at_centers(shape[b, band], centers),
                }
            )
        sampled.append(per_band)
    return sampled


def _nearest_candidate_indices(
    anchor_xy: Tensor,
    candidate_xy: Tensor,
    *,
    candidate_count: int,
    exclude_index: Optional[int] = None,
    positive_mask: Optional[Tensor] = None,
) -> Tensor:
    """Return K candidate indices, forcing a positive inside K when present."""

    if candidate_xy.numel() == 0:
        return candidate_xy.new_empty((0,), dtype=torch.long)
    dist = torch.linalg.norm(candidate_xy - anchor_xy[None, :], dim=1)
    if exclude_index is not None and 0 <= exclude_index < dist.shape[0]:
        dist = dist.clone()
        dist[int(exclude_index)] = float("inf")
    finite = torch.isfinite(dist)
    if not bool(finite.any()):
        return candidate_xy.new_empty((0,), dtype=torch.long)
    order = torch.argsort(dist)
    order = order[torch.isfinite(dist[order])]
    selected = order[: int(candidate_count)]
    if positive_mask is not None and bool(positive_mask.any()):
        pos_order = torch.argsort(dist.masked_fill(~positive_mask, float("inf")))
        pos_order = pos_order[torch.isfinite(dist[pos_order])]
        if pos_order.numel() > 0 and not bool((selected == pos_order[0]).any()):
            if selected.numel() >= int(candidate_count):
                selected = torch.cat([selected[:-1], pos_order[:1]], dim=0)
            else:
                selected = torch.cat([selected, pos_order[:1]], dim=0)
    return selected


def _resolve_band_index(bands: Sequence[str], name: str) -> int:
    """Resolve full band names like HSC-I and short names like I."""

    query = str(name).strip()
    if not query:
        raise ValueError("empty band name")
    for idx, band in enumerate(bands):
        if str(band) == query:
            return idx
    query_upper = query.upper()
    for idx, band in enumerate(bands):
        band_upper = str(band).upper()
        if band_upper == query_upper or band_upper.split("-")[-1] == query_upper:
            return idx
    raise ValueError(f"band {name!r} is not in configured bands {list(bands)}")


def parse_ex_band_pairs(
    bands: Sequence[str],
    *,
    core_band: str,
    pair_specs: Optional[Sequence[str]] = None,
) -> Tuple[Tuple[int, int], ...]:
    """Build directed EX training pairs.

    By default EX is trained from one core band to every other band, which keeps
    the CELLECT idea of one anchor frame while avoiding all pair permutations.
    Explicit specs use ``src:dst`` or ``src->dst``; ``all`` restores every
    directed cross-band pair.
    """

    if len(bands) <= 1:
        return tuple()
    specs = [part.strip() for item in (pair_specs or ()) for part in str(item).split(",") if part.strip()]
    if any(spec.lower() == "all" for spec in specs):
        return tuple((src, dst) for src in range(len(bands)) for dst in range(len(bands)) if src != dst)
    pairs: List[Tuple[int, int]] = []
    if specs:
        for spec in specs:
            if "->" in spec:
                src_name, dst_name = spec.split("->", 1)
            elif ":" in spec:
                src_name, dst_name = spec.split(":", 1)
            else:
                raise ValueError(f"EX band pair {spec!r} must use src:dst or src->dst")
            src = _resolve_band_index(bands, src_name)
            dst = _resolve_band_index(bands, dst_name)
            if src == dst:
                raise ValueError(f"EX band pair {spec!r} links a band to itself")
            pair = (src, dst)
            if pair not in pairs:
                pairs.append(pair)
        return tuple(pairs)

    core_idx = _resolve_band_index(bands, core_band)
    return tuple((core_idx, dst) for dst in range(len(bands)) if dst != core_idx)


def matcher_classification_loss(
    matcher: AstroMatchNet2D,
    outputs: Dict[str, Tensor],
    batch: Dict[str, object],
    *,
    mode: str,
    candidate_count: int,
    offset_scale: float,
    band_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tensor:
    """Cross-entropy loss for EX/EN candidate classification.

    ``mode="ex"`` samples candidates from other bands in the same cutout; the
    target is the candidate with the same source ``id``.  This trains EX to
    merge detections of the same astronomical source across bands.

    ``mode="en"`` samples candidates from the same band, excluding the anchor;
    positives are repeated ids if they exist, otherwise the none-of-above logit
    is the target.  This trains EN as an intra-band duplicate suppressor while
    remaining usable for single-band data.
    """

    if outputs["embedding"].ndim != 5:
        return outputs["embedding"].sum() * 0.0
    sampled = _sample_multiband_sources(outputs, batch)
    anchor_features: List[Tensor] = []
    candidate_features: List[Tensor] = []
    candidate_offsets: List[Tensor] = []
    candidate_shapes: List[Tensor] = []
    targets: List[int] = []
    k = int(candidate_count)
    scale = max(float(offset_scale), 1e-6)
    pair_map: Optional[Dict[int, List[int]]] = None
    if mode == "ex" and band_pairs is not None:
        pair_map = {}
        for src, dst in band_pairs:
            pair_map.setdefault(int(src), []).append(int(dst))

    for per_item in sampled:
        for band_idx, anchor_band in enumerate(per_item):
            if anchor_band["xy"].numel() == 0:
                continue
            if mode == "ex":
                if pair_map is None:
                    candidate_band_indices = [idx for idx in range(len(per_item)) if idx != band_idx]
                else:
                    candidate_band_indices = [idx for idx in pair_map.get(band_idx, []) if 0 <= idx < len(per_item)]
            elif mode == "en":
                candidate_band_indices = [band_idx]
            else:
                raise ValueError(f"unknown matcher mode: {mode}")
            if not candidate_band_indices:
                continue

            for anchor_idx in range(anchor_band["xy"].shape[0]):
                anchor_id = anchor_band["ids"][anchor_idx]
                anchor_xy = anchor_band["xy"][anchor_idx]
                anchor_shape = anchor_band["shape"][anchor_idx]
                for cand_band_idx in candidate_band_indices:
                    cand_band = per_item[cand_band_idx]
                    if cand_band["xy"].numel() == 0:
                        continue
                    positive = cand_band["ids"] == anchor_id
                    exclude_index = anchor_idx if mode == "en" and cand_band_idx == band_idx else None
                    if exclude_index is not None:
                        positive = positive.clone()
                        if 0 <= exclude_index < positive.shape[0]:
                            positive[exclude_index] = False
                    selected = _nearest_candidate_indices(
                        anchor_xy,
                        cand_band["xy"],
                        candidate_count=k,
                        exclude_index=exclude_index,
                        positive_mask=positive,
                    )
                    if selected.numel() == 0:
                        continue
                    pad = k - int(selected.numel())
                    if pad > 0:
                        selected = torch.cat([selected, selected[-1:].repeat(pad)], dim=0)
                    else:
                        selected = selected[:k]

                    cand_features = cand_band["features"][selected]
                    cand_xy = cand_band["xy"][selected]
                    cand_shape = cand_band["shape"][selected]
                    target_idx = k
                    same_selected = cand_band["ids"][selected] == anchor_id
                    if bool(same_selected.any()):
                        target_idx = int(torch.nonzero(same_selected, as_tuple=False)[0, 0].item())

                    anchor_features.append(anchor_band["features"][anchor_idx])
                    candidate_features.append(cand_features)
                    candidate_offsets.append((cand_xy - anchor_xy[None, :]) / scale)
                    candidate_shapes.append(torch.cat([anchor_shape[None, :].expand(k, -1), cand_shape], dim=1))
                    targets.append(target_idx)

    if not anchor_features:
        return outputs["embedding"].sum() * 0.0

    anchor_tensor = torch.stack(anchor_features)
    candidate_tensor = torch.stack(candidate_features)
    offset_tensor = torch.stack(candidate_offsets)
    shape_tensor = torch.stack(candidate_shapes)
    target_tensor = torch.tensor(targets, dtype=torch.long, device=anchor_tensor.device)
    logits = matcher(anchor_tensor, candidate_tensor, offset_tensor, shape_tensor)
    return F.cross_entropy(logits, target_tensor)


def embedding_triplet_loss(
    outputs: Dict[str, Tensor],
    batch: Dict[str, object],
    *,
    loss_fn: HardTripletLoss,
    max_sources_per_group: int = 0,
) -> Tensor:
    def _loss_with_optional_source_filter(features: Tensor, labels: Tensor, groups: Tensor) -> Tensor:
        if max_sources_per_group > 0:
            features, labels, groups = _filter_triplet_sources(
                features,
                labels,
                groups,
                max_sources_per_group=max_sources_per_group,
            )
            if features.shape[0] == 0:
                return outputs["embedding"].sum() * 0.0
        return loss_fn(features, labels, groups=groups)

    if outputs["embedding"].ndim == 5:
        sampled = _sample_multiband_sources(outputs, batch)
        features: List[Tensor] = []
        labels: List[Tensor] = []
        groups: List[Tensor] = []
        for batch_idx, per_item in enumerate(sampled):
            for band in per_item:
                if band["features"].numel() == 0:
                    continue
                features.append(band["features"])
                labels.append(band["ids"])
                groups.append(torch.full((band["ids"].shape[0],), batch_idx, dtype=torch.long, device=band["ids"].device))
        if not features:
            return outputs["embedding"].sum() * 0.0
        return _loss_with_optional_source_filter(
            torch.cat(features, dim=0),
            torch.cat(labels, dim=0),
            torch.cat(groups, dim=0),
        )

    features, batch_ids = _sample_embeddings_at_centers(outputs, batch["centers"])  # type: ignore[arg-type]
    labels_list = [ids.to(device=features.device, dtype=torch.long) for ids in batch["ids"]]  # type: ignore[index]
    if not labels_list:
        return outputs["embedding"].sum() * 0.0
    labels = torch.cat(labels_list, dim=0)
    if labels.shape[0] != features.shape[0]:
        return outputs["embedding"].sum() * 0.0
    return _loss_with_optional_source_filter(features, labels, batch_ids)


def _filter_triplet_sources(
    features: Tensor,
    labels: Tensor,
    groups: Tensor,
    *,
    max_sources_per_group: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Limit triplet mining to a random subset of source IDs per image/group.

    The expensive part of hard triplet mining is the all-pairs ``cdist``.  This
    sampler keeps at most ``max_sources_per_group`` unique positive labels inside
    each group, while retaining every feature row for the selected labels.  For
    multi-band data that means G/R/I embeddings of the same source remain
    together, so cross-band positive pairs are not accidentally split.
    """

    if features.shape[0] == 0 or max_sources_per_group <= 0:
        return features, labels, groups
    labels = labels.long()
    groups = groups.long()
    keep = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
    valid = labels > 0
    all_labels, all_counts = torch.unique(labels[valid], return_counts=True) if bool(valid.any()) else (labels[:0], labels[:0])
    repeated_labels = all_labels[all_counts > 1]
    for group in torch.unique(groups):
        in_group = valid & (groups == group)
        if not bool(in_group.any()):
            continue
        unique_labels = torch.unique(labels[in_group])
        limit = int(max_sources_per_group)
        if unique_labels.numel() > limit:
            priority = unique_labels[torch.isin(unique_labels, repeated_labels)]
            other = unique_labels[~torch.isin(unique_labels, repeated_labels)]
            if priority.numel() >= limit:
                order = torch.randperm(priority.numel(), device=labels.device)[:limit]
                unique_labels = priority[order]
            else:
                take_other = limit - int(priority.numel())
                if other.numel() > take_other:
                    other = other[torch.randperm(other.numel(), device=labels.device)[:take_other]]
                unique_labels = torch.cat([priority, other], dim=0)
        keep |= in_group & torch.isin(labels, unique_labels)
    return features[keep], labels[keep], groups[keep]


@torch.no_grad()
def _confidence_detection_score(outputs: Dict[str, Tensor], mode: str) -> Tensor:
    logits = outputs["confidence"]
    if mode == "raw":
        return logits[:, -1]
    if mode == "ordinal_prob":
        prev = logits[:, :-1].max(dim=1).values
        curr = logits[:, -1]
        return torch.softmax(torch.stack([prev, curr], dim=1), dim=1)[:, 1]
    raise ValueError(f"Unknown confidence score mode: {mode}")


def _cellect_confidence_smooth_2d(logits: Tensor) -> Tensor:
    """Apply CELLECT's DK1 confidence smoothing kernel in 2D.

    Original CELLECT applies a grouped 3D convolution with the 7x7x1 DK1
    diamond kernel before candidate extraction. For HSC 2D images we use the
    same XY kernel without normalization, matching the original score scale.
    """

    kernel = logits.new_tensor(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 2, 1, 0, 0],
            [0, 1, 2, 3, 2, 1, 0],
            [1, 2, 3, 4, 3, 2, 1],
            [0, 1, 2, 3, 2, 1, 0],
            [0, 0, 1, 2, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ]
    ).reshape(1, 1, 7, 7)
    weight = kernel.expand(logits.shape[1], 1, 7, 7)
    return F.conv2d(logits.float(), weight.float(), padding=3, groups=logits.shape[1]).to(dtype=logits.dtype)


def _cellect_foreground_gate_2d(seg_logits: Tensor) -> Tensor:
    """Reject candidates touching predicted background, matching CELLECT's kflb gate."""

    background = seg_logits.argmax(dim=1) == 0
    background_near = F.max_pool2d(background.float().unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1) > 0
    return ~background_near


def detect_centers(
    outputs: Dict[str, Tensor],
    *,
    threshold: float = 0.0,
    nms_radius: int = 1,
    confidence_score: str = "cellect",
) -> List[np.ndarray]:
    """Detect center candidates from confidence maps.

    ``confidence_score="cellect"`` follows the original CELLECT extraction:
    smooth all confidence channels with DK1, max-pool the channel-wise maximum
    with kernel size 3, and keep voxels/pixels where that local maximum equals
    channel 4 while the local segmentation neighborhood is non-background.
    """

    if confidence_score == "cellect":
        smoothed = _cellect_confidence_smooth_2d(outputs["confidence"])
        local_score = smoothed.max(dim=1).values
        center_score = smoothed[:, -1]
        pooled = F.max_pool2d(local_score.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        peaks = (pooled == center_score) & _cellect_foreground_gate_2d(outputs["seg_logits"]) & (center_score > threshold)
        conf = center_score
    else:
        conf = _confidence_detection_score(outputs, confidence_score)
        fg = outputs["seg_logits"].argmax(dim=1) > 0
        pooled = F.max_pool2d(
            conf.unsqueeze(1),
            kernel_size=2 * nms_radius + 1,
            stride=1,
            padding=nms_radius,
        ).squeeze(1)
        peaks = (conf == pooled) & fg & (conf > threshold)

    result: List[np.ndarray] = []
    for b in range(conf.shape[0]):
        y, x = torch.where(peaks[b])
        coords = torch.stack([x, y], dim=1).detach().cpu().numpy().astype(np.float32)
        result.append(coords)
    return result


@torch.no_grad()
def build_cellect_style_segmentation(
    outputs: Dict[str, Tensor],
    centers_list: Sequence[np.ndarray | Tensor],
    *,
    ellipse_sigma: float = 3.0,
    min_axis: float = 1.5,
) -> List[Dict[str, np.ndarray]]:
    """Build CELLECT-style dense ``seg`` and per-center ``seg_mask`` outputs.

    CELLECT first detects centers, samples a size estimate at each center, fills
    one instance region per center, and finally intersects that instance map
    with the coarse foreground segmentation.  This is the 2D astronomy version:

    - ``seg`` is the dense argmax class map from ``seg_logits``.
    - ``seg_mask`` stores instance ids 1..N filled from detected centers and
      predicted ellipses, then filtered by ``seg > 0``.

    The model's shape target stores unscaled semi-major/semi-minor axes, while
    the pseudo segmentation label uses ``ellipse_sigma`` times those axes.  The
    same scale is therefore used here to produce instance masks comparable to
    training-time masks.
    """

    seg = outputs["seg_logits"].argmax(dim=1).detach().cpu().numpy().astype(np.int16)
    shape = outputs["shape"].detach().cpu().numpy().astype(np.float32)
    results: List[Dict[str, np.ndarray]] = []

    for batch_idx, centers in enumerate(centers_list):
        if isinstance(centers, Tensor):
            centers_np = centers.detach().cpu().numpy().astype(np.float32)
        else:
            centers_np = np.asarray(centers, dtype=np.float32)
        if centers_np.ndim == 1:
            centers_np = centers_np.reshape((-1, 2)) if centers_np.size else np.zeros((0, 2), dtype=np.float32)

        seg_map = seg[batch_idx]
        h, w = seg_map.shape
        instance = np.zeros((h, w), dtype=np.int32)

        for instance_id, (cx, cy) in enumerate(centers_np[:, :2], start=1):
            if not np.isfinite(cx) or not np.isfinite(cy):
                continue
            xi = int(round(float(cx)))
            yi = int(round(float(cy)))
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                continue

            major = float(shape[batch_idx, 0, yi, xi]) if shape.shape[1] >= 1 else min_axis
            minor = float(shape[batch_idx, 1, yi, xi]) if shape.shape[1] >= 2 else major
            theta = float(shape[batch_idx, 2, yi, xi]) if shape.shape[1] >= 3 else 0.0
            if not np.isfinite(major):
                major = min_axis
            if not np.isfinite(minor):
                minor = major
            if not np.isfinite(theta):
                theta = 0.0

            a = max(abs(major) * float(ellipse_sigma), float(min_axis))
            b = max(abs(minor) * float(ellipse_sigma), float(min_axis))
            radius = int(math.ceil(max(a, b))) + 2
            y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
            x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
            if x0 >= x1 or y0 >= y1:
                continue

            yy, xx = np.mgrid[y0:y1, x0:x1]
            dx = xx.astype(np.float32) - float(cx)
            dy = yy.astype(np.float32) - float(cy)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            xr = cos_t * dx + sin_t * dy
            yr = -sin_t * dx + cos_t * dy
            ellipse = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
            patch = instance[y0:y1, x0:x1]
            patch[ellipse] = instance_id

        instance[seg_map <= 0] = 0
        results.append({"seg": seg_map.copy(), "seg_mask": instance})

    return results


def match_points(pred_xy: np.ndarray, gt_xy: np.ndarray, radius: float) -> Tuple[int, int, int]:
    if pred_xy.size == 0:
        return 0, 0, int(len(gt_xy))
    if gt_xy.size == 0:
        return 0, int(len(pred_xy)), 0
    dist = np.sqrt(((pred_xy[:, None, :] - gt_xy[None, :, :]) ** 2).sum(axis=2))
    pairs = []
    for i in range(dist.shape[0]):
        j = int(np.argmin(dist[i]))
        if dist[i, j] <= radius:
            pairs.append((float(dist[i, j]), i, j))
    pairs.sort()
    used_pred = set()
    used_gt = set()
    tp = 0
    for _d, i, j in pairs:
        if i in used_pred or j in used_gt:
            continue
        used_pred.add(i)
        used_gt.add(j)
        tp += 1
    fp = len(pred_xy) - tp
    fn = len(gt_xy) - tp
    return tp, fp, fn


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    weights: LossWeights,
    triplet_loss_fn: HardTripletLoss,
    triplet_enabled: bool,
    ex_enabled: bool,
    en_enabled: bool,
    matcher_candidate_count: int,
    matcher_max_anchors_per_band: int,
    triplet_max_sources_per_group: int,
    ex_band_pairs: Optional[Sequence[Tuple[int, int]]],
    center_radius_px: float,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums: Dict[str, float] = {
        "total": 0.0,
        "seg": 0.0,
        "confidence": 0.0,
        "shape": 0.0,
        "center": 0.0,
        "triplet": 0.0,
        "ex_class": 0.0,
        "en_class": 0.0,
    }
    count = 0
    t_epoch_start = time.time()
    # torch.cuda.synchronize(device) if device.type == "cuda" else None
    # print('[DEBUG] Starting epoch, running initial evaluation loop to prime caches...')
    # t_batch = time.time()
    for batch in tqdm(loader, desc="train" if training else "eval", leave=False):
        image = batch["image"].to(device=device, dtype=torch.float32)
        outputs = model(image)
        # torch.cuda.synchronize(device) if device.type == "cuda" else None
        # t_batch_model = time.time()
        # print(f'[DEBUG] Model forward pass done, time {t_batch_model - t_batch:.3f} seconds.')
        losses = dense_losses_any(outputs, batch, weights=weights, device=device, center_radius_px=center_radius_px)
        total = losses["total"]
        # torch.cuda.synchronize(device) if device.type == "cuda" else None
        # dense_time = time.time()
        # print(f'[DEBUG] Dense loss computation done, time {dense_time - t_batch_model:.3f} seconds.')

        triplet = total.new_tensor(0.0)
        if triplet_enabled:
            triplet = embedding_triplet_loss(
                outputs,
                batch,
                loss_fn=triplet_loss_fn,
                max_sources_per_group=triplet_max_sources_per_group,
            )
            total = total + weights.triplet_outer_weight * triplet
            # torch.cuda.synchronize(device) if device.type == "cuda" else None
            # t_triplet = time.time()
            # print(f'[DEBUG] Triplet loss computation done, time {t_triplet - dense_time:.3f} seconds.')
            # dense_time = t_triplet

        ex_class = total.new_tensor(0.0)
        if ex_enabled and hasattr(model, "EX") and image.shape[1] > 1:
            ex_class = vectorized_matcher_classification_loss(
                model.EX,
                outputs,
                batch,
                mode="ex",
                candidate_count=matcher_candidate_count,
                offset_scale=center_radius_px,
                band_pairs=ex_band_pairs,
                max_anchors_per_band=matcher_max_anchors_per_band,
            )
            total = total + weights.matcher_outer_weight * ex_class
            # torch.cuda.synchronize(device) if device.type == "cuda" else None
            # t_ex = time.time()
            # print(f'[DEBUG] EX classification loss computation done, time {t_ex - dense_time:.3f} seconds.')
            # dense_time = t_ex

        en_class = total.new_tensor(0.0)
        if en_enabled and hasattr(model, "EN"):
            en_class = vectorized_matcher_classification_loss(
                model.EN,
                outputs,
                batch,
                mode="en",
                candidate_count=matcher_candidate_count,
                offset_scale=center_radius_px,
                max_anchors_per_band=matcher_max_anchors_per_band,
            )
            total = total + weights.matcher_outer_weight * en_class
            # torch.cuda.synchronize(device) if device.type == "cuda" else None
            # t_en = time.time()
            # print(f'[DEBUG] EN classification loss computation done, time {t_en - dense_time:.3f} seconds.')
            # dense_time = t_en

        # t_backward = dense_time
        if training:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            # torch.cuda.synchronize(device) if device.type == "cuda" else None
            # t_backward = time.time()
            # print(f'[DEBUG] Optimizer step done, time {t_backward - dense_time:.3f} seconds.')

        batch_size = int(image.shape[0])
        count += batch_size
        sums["total"] += float(total.detach()) * batch_size
        sums["seg"] += float(losses["seg"].detach()) * batch_size
        sums["confidence"] += float(losses["confidence"].detach()) * batch_size
        sums["shape"] += float(losses["shape"].detach()) * batch_size
        sums["center"] += float(losses["center"].detach()) * batch_size
        sums["triplet"] += float(triplet.detach()) * batch_size
        sums["ex_class"] += float(ex_class.detach()) * batch_size
        sums["en_class"] += float(en_class.detach()) * batch_size
        t_batch = time.time()
    # print(f"[DEBUG] Epoch completed in {time.time() - t_epoch_start:.3f} seconds.")
    return {key: val / max(count, 1) for key, val in sums.items()}


@torch.no_grad()
def evaluate_detection(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    threshold: float,
    nms_radius: int,
    confidence_score: str,
    match_radius: float,
    use_en_postprocess: bool = False,
    en_candidate_count: int = 5,
    en_threshold: float = 0.6,
    use_ex_link_postprocess: bool = False,
    ex_link_threshold: float = 0.5,
    ex_band_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    band_names: Sequence[str] = (),
) -> Dict[str, object]:
    model.eval()
    tp = fp = fn = 0
    linked_tp = linked_fp = linked_fn = 0
    per_band_counts: Dict[str, Dict[str, int]] = {
        str(name): {"tp": 0, "fp": 0, "fn": 0} for name in band_names
    }
    for batch in tqdm(loader, desc="detect", leave=False):
        image = batch["image"].to(device=device, dtype=torch.float32)
        outputs = model(image)
        band_count = int(outputs["seg_logits"].shape[1]) if outputs["seg_logits"].ndim == 5 else 0
        if use_en_postprocess and hasattr(model, "EN"):
            pred_list = detect_centers_with_en(
                model,
                outputs,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
                match_radius=match_radius,
                candidate_count=en_candidate_count,
                en_threshold=en_threshold,
            )
            gt_list = _flatten_band_centers(batch["band_centers"]) if outputs["seg_logits"].ndim == 5 else batch["centers"]  # type: ignore[arg-type]
            band_indices = [idx % band_count for idx in range(len(pred_list))] if band_count else [None for _ in pred_list]
        elif outputs["seg_logits"].ndim == 5:
            flat_outputs = _flatten_per_band_outputs(outputs)
            pred_list = detect_centers(
                flat_outputs,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
            )
            gt_list = _flatten_band_centers(batch["band_centers"])  # type: ignore[arg-type]
            band_indices = [idx % band_count for idx in range(len(pred_list))]
        else:
            pred_list = detect_centers(
                outputs,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
            )
            gt_list = batch["centers"]  # type: ignore[assignment]
            band_indices = [None for _ in pred_list]
        for pred_xy, gt_xy, band_idx in zip(pred_list, gt_list, band_indices):
            t, f, n = match_points(pred_xy, gt_xy.numpy().astype(np.float32), match_radius)
            tp += t
            fp += f
            fn += n
            if band_idx is not None and band_names:
                band_name = str(band_names[int(band_idx)])
                counts = per_band_counts.setdefault(band_name, {"tp": 0, "fp": 0, "fn": 0})
                counts["tp"] += int(t)
                counts["fp"] += int(f)
                counts["fn"] += int(n)

        if use_ex_link_postprocess and outputs["seg_logits"].ndim == 5 and hasattr(model, "EX"):
            linked_pred_list, _components = detect_centers_with_ex_link(
                model,
                outputs,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
                match_radius=match_radius,
                candidate_count=en_candidate_count,
                ex_threshold=ex_link_threshold,
                use_en_postprocess=use_en_postprocess,
                en_threshold=en_threshold,
                band_pairs=ex_band_pairs,
            )
            linked_gt_list = batch["centers"]  # type: ignore[assignment]
            for pred_xy, gt_xy in zip(linked_pred_list, linked_gt_list):
                t, f, n = match_points(pred_xy, gt_xy.numpy().astype(np.float32), match_radius)
                linked_tp += t
                linked_fp += f
                linked_fn += n
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    result: Dict[str, object] = {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "purity": precision,
        "recall": recall,
        "completeness": recall,
        "f1": f1,
    }
    if per_band_counts:
        per_band: Dict[str, Dict[str, float]] = {}
        for band_name, counts in per_band_counts.items():
            btp = int(counts["tp"])
            bfp = int(counts["fp"])
            bfn = int(counts["fn"])
            b_precision = btp / max(btp + bfp, 1)
            b_recall = btp / max(btp + bfn, 1)
            per_band[band_name] = {
                "tp": float(btp),
                "fp": float(bfp),
                "fn": float(bfn),
                "precision": b_precision,
                "purity": b_precision,
                "recall": b_recall,
                "completeness": b_recall,
                "f1": 2.0 * b_precision * b_recall / max(b_precision + b_recall, 1e-12),
            }
        result["per_band"] = per_band
    if use_ex_link_postprocess:
        linked_precision = linked_tp / max(linked_tp + linked_fp, 1)
        linked_recall = linked_tp / max(linked_tp + linked_fn, 1)
        result["linked"] = {
            "tp": float(linked_tp),
            "fp": float(linked_fp),
            "fn": float(linked_fn),
            "precision": linked_precision,
            "purity": linked_precision,
            "recall": linked_recall,
            "completeness": linked_recall,
            "f1": 2.0 * linked_precision * linked_recall / max(linked_precision + linked_recall, 1e-12),
        }
    return result


def split_records(
    records: Sequence[CutoutRecord],
    val_fraction: float,
    seed: int,
    *,
    fixed_val_names: Sequence[str] = (),
) -> Tuple[List[CutoutRecord], List[CutoutRecord]]:
    rng = random.Random(seed)
    fixed_names = {name for name in fixed_val_names if name}
    fixed_val = [rec for rec in records if rec.name in fixed_names]
    remaining = [rec for rec in records if rec.name not in fixed_names]
    shuffled = list(remaining)
    rng.shuffle(shuffled)
    target_n_val = max(1, int(round(len(records) * val_fraction))) if len(records) > 1 else 0
    n_random_val = max(0, target_n_val - len(fixed_val))
    random_val = shuffled[:n_random_val]
    train = shuffled[n_random_val:]
    return train, fixed_val + random_val


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate AstroCELLECT2D on LSST/HSC FITS cutouts.")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument(
        "--root",
        default="~/segment-anything/lsst_pipeline/output/cutout_magnitude_experiment_grid",
        help="Root containing cutouts/ and reference_catalogs/.",
    )
    parser.add_argument("--reference-dir", default=None)
    parser.add_argument("--cutout-dir", default=None)
    parser.add_argument(
        "--band-reference-root",
        default=None,
        help="Optional root containing per-band meas catalogs, e.g. catalog/HSC-I/meas-HSC-I-*.fits. "
        "When set, each band uses its own catalog ids/centers for EX/EN training.",
    )
    parser.add_argument(
        "--targets-dir",
        default=None,
        help="Optional directory containing precomputed <tile>.npz dense targets from astro_data_preprocessing.py.",
    )
    parser.add_argument("--bands", nargs="+", default=("HSC-G", "HSC-R", "HSC-I"))
    parser.add_argument("--fits-hdu", type=int, default=1)
    parser.add_argument("--out-dir", default="./output/astro_cellect2d")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument(
        "--model-variant",
        choices=("auto", "fused", "per_band", "fused_encoder"),
        default="auto",
        help="fused treats bands as channels and outputs one map. per_band runs one shared single-band backbone per band. "
        "fused_encoder runs one multi-band encoder and lightweight per-band heads for EX/EN. "
        "auto uses fused_encoder for multi-band data or when EN is enabled.",
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--fixed-val-names",
        nargs="*",
        default=("sam_x18204_y20924",),
        help="Tile names that must be placed in the validation set. Default keeps the SAM comparison cutout in validation.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ellipse-sigma", type=float, default=2.0)
    parser.add_argument("--core-radius", type=int, default=2)
    parser.add_argument(
        "--shape-source",
        choices=("kron", "sdss", "circular_kron"),
        default="kron",
        help="Pseudo-mask ellipse source: Kron radius plus moment axis ratio, pure Sdss moments, or circular Kron.",
    )
    parser.add_argument(
        "--source-filter",
        choices=("nchild0", "all", "parent", "leaf_child"),
        default="nchild0",
        help="Catalog rows used as center/embedding/eval GT. Default is deblend_nChild==0 leaf sources.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument(
        "--nms-radius",
        type=int,
        default=1,
        help="Local-max suppression radius. CELLECT uses kernel_size=3, equivalent to radius=1.",
    )
    parser.add_argument(
        "--confidence-score",
        choices=("cellect", "raw", "ordinal_prob"),
        default="cellect",
        help="Score used for center detection. cellect applies DK1 smoothing plus kernel_size=3 local-max logic from CELLECT.",
    )
    parser.add_argument("--center-tolerance-arcsec", type=float, default=0.5)
    parser.add_argument("--pixel-scale-arcsec", type=float, default=0.168)
    parser.add_argument(
        "--match-radius",
        type=float,
        default=None,
        help="Center matching radius in pixels. Defaults to center_tolerance_arcsec / pixel_scale_arcsec.",
    )
    # Shut down the time consuming center loss by default since confidence map supervision is usually sufficient for good center detection, and the center loss can be a significant bottleneck when training with many small sources.
    parser.add_argument("--center-loss-weight", type=float, default=0.0)
    parser.add_argument("--segmentation-class-weights", type=float, nargs=3, default=(1.0, 32.0, 1.0))
    parser.add_argument("--confidence-pos-weight", type=float, default=32.0)
    parser.add_argument("--triplet-outer-weight", type=float, default=10.0)
    parser.add_argument("--enable-triplet", action="store_true")
    parser.add_argument(
        "--triplet-max-sources-per-group",
        type=int,
        default=256,
        help="Maximum unique source IDs per image/group used by hard triplet mining. "
        "Set <=0 to use all sources. Dense losses and EX/EN losses are unchanged.",
    )
    parser.add_argument(
        "--disable-ex-loss",
        action="store_true",
        help="Disable EX cross-band classification loss. By default EX is trained for per_band multi-band runs.",
    )
    parser.add_argument(
        "--ex-core-band",
        default="HSC-I",
        help="Core band used as EX anchor when --ex-band-pairs is not set. Short names like I are accepted.",
    )
    parser.add_argument(
        "--ex-band-pairs",
        nargs="*",
        default=None,
        help="Directed EX training pairs such as HSC-I:HSC-G HSC-I:HSC-R. "
        "Use 'all' to restore all directed cross-band pairs.",
    )
    parser.add_argument(
        "--enable-en-loss",
        action="store_true",
        help="Train EN same-band duplicate/none-of-above classification loss. Useful for single-band duplicate suppression too.",
    )
    parser.add_argument(
        "--use-en-postprocess",
        action="store_true",
        help="Apply CELLECT-style EN same-band deduplication after detect_centers during evaluation/inference.",
    )
    parser.add_argument("--en-postprocess-threshold", type=float, default=0.6)
    parser.add_argument(
        "--use-ex-link-postprocess",
        action="store_true",
        help="Apply CELLECT-style EX cross-band linking after per-band center detection. "
        "When enabled, detection metrics are object-level instead of band-level.",
    )
    parser.add_argument("--ex-link-threshold", type=float, default=0.5)
    parser.add_argument("--matcher-candidate-count", type=int, default=5)
    parser.add_argument(
        "--matcher-max-anchors-per-band",
        type=int,
        default=128,
        help="Maximum source anchors per batch item and band for EX/EN classification loss. "
        "Set <=0 to use all anchors.",
    )
    parser.add_argument("--matcher-outer-weight", type=float, default=10.0)
    parser.add_argument(
        "--image-cache-dir",
        default=None,
        help="Optional directory for zscale-preprocessed CHW tensors. This avoids FITS/zscale work after the first epoch.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    # Debug: output the time of each step
    T=time.time()
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = _expand_path(args.root)
    reference_dir = _expand_path(args.reference_dir) if args.reference_dir else None
    cutout_dir = _expand_path(args.cutout_dir) if args.cutout_dir else None
    band_reference_root = _expand_path(args.band_reference_root) if args.band_reference_root else None
    targets_dir = _expand_path(args.targets_dir) if args.targets_dir else (root / "targets")
    if not targets_dir.exists():
        targets_dir = None
    image_cache_dir = _expand_path(args.image_cache_dir) if args.image_cache_dir else None
    out_dir = _expand_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = discover_cutout_records(
        root,
        reference_dir=reference_dir,
        cutout_dir=cutout_dir,
        band_reference_root=band_reference_root,
        bands=args.bands,
        max_records=args.max_records,
    )
    train_records, val_records = split_records(
        records,
        args.val_fraction,
        args.seed,
        fixed_val_names=args.fixed_val_names,
    )
    missing_fixed_val = sorted(set(args.fixed_val_names) - {rec.name for rec in records})
    if missing_fixed_val:
        print(f"WARNING: fixed validation tile(s) not found and cannot be forced into val: {missing_fixed_val}")
    if args.mode == "eval":
        val_records = records

    center_radius_px = (
        float(args.match_radius)
        if args.match_radius is not None
        else float(args.center_tolerance_arcsec) / max(float(args.pixel_scale_arcsec), 1e-12)
    )
    weights = LossWeights(
        segmentation_3cls=tuple(float(v) for v in args.segmentation_class_weights),
        confidence_pos_weight=float(args.confidence_pos_weight),
        center_position=float(args.center_loss_weight),
        triplet_outer_weight=float(args.triplet_outer_weight),
        matcher_outer_weight=float(args.matcher_outer_weight),
    )
    common_ds = dict(
        fits_hdu=args.fits_hdu,
        confidence_levels=5,
        ellipse_sigma=args.ellipse_sigma,
        core_radius=args.core_radius,
        shape_source=args.shape_source,
        source_filter=args.source_filter,
        targets_dir=targets_dir,
        image_cache_dir=image_cache_dir,
    )
    train_ds = AstroCutoutDataset(train_records, augment=True, **common_ds)
    val_ds = AstroCutoutDataset(val_records, augment=False, **common_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_cutouts,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_cutouts,
    )
    # t_preprocessing = time.time()
    # print(f'[DEBUG] Preprocessing done, time {t_preprocessing - T:.3f} seconds. Starting model setup...')
    device = torch.device(args.device)
    if args.model_variant == "auto":
        model_variant = "fused_encoder" if len(args.bands) > 1 or args.enable_en_loss else "fused"
    else:
        model_variant = args.model_variant
    matcher_variant = model_variant in ("per_band", "fused_encoder")
    ex_enabled = matcher_variant and len(args.bands) > 1 and not args.disable_ex_loss
    en_enabled = matcher_variant and bool(args.enable_en_loss)
    en_postprocess_enabled = matcher_variant and (bool(args.use_en_postprocess) or en_enabled)
    ex_link_postprocess_enabled = matcher_variant and len(args.bands) > 1 and bool(args.use_ex_link_postprocess)
    ex_band_pairs = (
        parse_matcher_ex_band_pairs(args.bands, core_band=args.ex_core_band, pair_specs=args.ex_band_pairs)
        if ex_enabled
        else tuple()
    )

    if model_variant == "per_band":
        model = MultiBandAstroCELLECT2D(
            num_bands=len(args.bands),
            seg_classes=3,
            confidence_levels=5,
            embedding_dim=args.embedding_dim,
            base_channels=args.base_channels,
            shape_channels=3,
            candidate_count=args.matcher_candidate_count,
            shape_feature_dim=6,
        ).to(device)
    elif model_variant == "fused_encoder":
        model = FusedEncoderMultiBandAstroCELLECT2D(
            num_bands=len(args.bands),
            seg_classes=3,
            confidence_levels=5,
            embedding_dim=args.embedding_dim,
            base_channels=args.base_channels,
            shape_channels=3,
            candidate_count=args.matcher_candidate_count,
            shape_feature_dim=6,
        ).to(device)
    else:
        model = AstroUNet2D(
            in_channels=len(args.bands),
            seg_classes=3,
            confidence_levels=5,
            embedding_dim=args.embedding_dim,
            base_channels=args.base_channels,
            shape_channels=3,
        ).to(device)

    if args.checkpoint:
        ckpt = torch.load(_expand_path(args.checkpoint), map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
        if hasattr(model, "EX") and hasattr(model, "EN"):
            if isinstance(ckpt, dict) and ckpt.get("EX") is not None:
                model.EX.load_state_dict(ckpt["EX"])
            if isinstance(ckpt, dict) and ckpt.get("EN") is not None:
                model.EN.load_state_dict(ckpt["EN"])

    if args.mode == "eval":
        dense = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            weights=weights,
            triplet_loss_fn=HardTripletLoss(weights.triplet_margin),
            triplet_enabled=args.enable_triplet,
            ex_enabled=ex_enabled,
            en_enabled=en_enabled,
            matcher_candidate_count=args.matcher_candidate_count,
            matcher_max_anchors_per_band=args.matcher_max_anchors_per_band,
            triplet_max_sources_per_group=args.triplet_max_sources_per_group,
            ex_band_pairs=ex_band_pairs,
            center_radius_px=center_radius_px,
        )
        det = evaluate_detection(
            model,
            val_loader,
            device=device,
            threshold=args.confidence_threshold,
            nms_radius=args.nms_radius,
            confidence_score=args.confidence_score,
            match_radius=center_radius_px,
            use_en_postprocess=en_postprocess_enabled,
            en_candidate_count=args.matcher_candidate_count,
            en_threshold=args.en_postprocess_threshold,
            use_ex_link_postprocess=ex_link_postprocess_enabled,
            ex_link_threshold=args.ex_link_threshold,
            ex_band_pairs=ex_band_pairs,
            band_names=args.bands,
        )
        print(json.dumps({"dense": dense, "detection": det}, indent=2))
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    triplet_loss_fn = HardTripletLoss(weights.triplet_margin)

    metadata = {
        "args": vars(args),
        "loss_weights": asdict(weights),
        "num_records": len(records),
        "num_train": len(train_records),
        "num_val": len(val_records),
        "fixed_val_names": list(args.fixed_val_names),
        "val_record_names": [rec.name for rec in val_records],
        "center_radius_px": center_radius_px,
        "targets_dir": str(targets_dir) if targets_dir is not None else None,
        "image_cache_dir": str(image_cache_dir) if image_cache_dir is not None else None,
        "band_reference_root": str(band_reference_root) if band_reference_root is not None else None,
        "model_variant": model_variant,
        "ex_enabled": ex_enabled,
        "ex_band_pairs": [[str(args.bands[src]), str(args.bands[dst])] for src, dst in ex_band_pairs],
        "matcher_max_anchors_per_band": int(args.matcher_max_anchors_per_band),
        "en_enabled": en_enabled,
        "en_postprocess_enabled": en_postprocess_enabled,
        "ex_link_postprocess_enabled": ex_link_postprocess_enabled,
        "ex_link_threshold": float(args.ex_link_threshold),
    }
    (out_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # t_model = time.time()
    # print(f'[DEBUG] Model setup done, time {t_model - t_preprocessing:.3f} seconds. Starting training loop...')

    best_val = float("inf")
    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            weights=weights,
            triplet_loss_fn=triplet_loss_fn,
            triplet_enabled=args.enable_triplet,
            ex_enabled=ex_enabled,
            en_enabled=en_enabled,
            matcher_candidate_count=args.matcher_candidate_count,
            matcher_max_anchors_per_band=args.matcher_max_anchors_per_band,
            triplet_max_sources_per_group=args.triplet_max_sources_per_group,
            ex_band_pairs=ex_band_pairs,
            center_radius_px=center_radius_px,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            weights=weights,
            triplet_loss_fn=triplet_loss_fn,
            triplet_enabled=args.enable_triplet,
            ex_enabled=ex_enabled,
            en_enabled=en_enabled,
            matcher_candidate_count=args.matcher_candidate_count,
            matcher_max_anchors_per_band=args.matcher_max_anchors_per_band,
            triplet_max_sources_per_group=args.triplet_max_sources_per_group,
            ex_band_pairs=ex_band_pairs,
            center_radius_px=center_radius_px,
        )
        det_metrics = evaluate_detection(
            model,
            val_loader,
            device=device,
            threshold=args.confidence_threshold,
            nms_radius=args.nms_radius,
            confidence_score=args.confidence_score,
            match_radius=center_radius_px,
            use_en_postprocess=en_postprocess_enabled,
            en_candidate_count=args.matcher_candidate_count,
            en_threshold=args.en_postprocess_threshold,
            use_ex_link_postprocess=ex_link_postprocess_enabled,
            ex_link_threshold=args.ex_link_threshold,
            ex_band_pairs=ex_band_pairs,
            band_names=args.bands,
        )
        scheduler.step(val_metrics["total"])
        log_line = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "detection": det_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        print(json.dumps(log_line, indent=2))

        ckpt = {
            "model": model.state_dict(),
            "EX": model.EX.state_dict() if hasattr(model, "EX") else None,
            "EN": model.EN.state_dict() if hasattr(model, "EN") else None,
            "model_variant": model_variant,
            "epoch": epoch,
            "args": vars(args),
            "loss_weights": asdict(weights),
            "center_radius_px": center_radius_px,
            "val": val_metrics,
            "detection": det_metrics,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save(ckpt, out_dir / "best.pt")


if __name__ == "__main__":
    main()
