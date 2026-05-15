"""
Train and evaluate the 2D CELLECT-style astronomy model on LSST/HSC FITS cutouts.

The dense losses intentionally keep CELLECT's original constants:
  - segmentation class weights [1, 32, 1];
  - ordinal confidence positive class weight 32;
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

from astro_cellect2d import (
    AstroMatchNet2D,
    AstroUNet2D,
    astro_zscale_preprocess,
    ordinal_confidence_loss,
    read_fits_bands,
)


@dataclass(frozen=True)
class LossWeights:
    """CELLECT constants carried into the 2D astronomy training loop."""

    segmentation_3cls: Tuple[float, float, float] = (1.0, 32.0, 1.0)
    confidence_pos_weight: float = 32.0
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


def discover_cutout_records(
    root: Path,
    *,
    reference_dir: Optional[Path] = None,
    cutout_dir: Optional[Path] = None,
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
        x0, y0 = _parse_tile_origin(tile_name)
        records.append(
            CutoutRecord(
                name=tile_name,
                image_paths=image_paths,
                meas_path=str(meas_path),
                x0=x0,
                y0=y0,
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
) -> Dict[str, np.ndarray]:
    """Read centers and approximate shape labels from an LSST meas FITS table."""

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
        xx = _finite_column(table, ("base_SdssShape_xx", "ext_shapeHSM_HsmSourceMoments_xx"))
        yy = _finite_column(table, ("base_SdssShape_yy", "ext_shapeHSM_HsmSourceMoments_yy"))
        xy = _finite_column(table, ("base_SdssShape_xy", "ext_shapeHSM_HsmSourceMoments_xy"))
        footprint = _finite_column(table, ("base_FootprintArea_value",))

    h, w = image_shape
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    valid &= parent == 0
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

    centers = np.stack([x, y], axis=1).astype(np.float32)
    return {
        "centers": centers,
        "ids": ids.astype(np.int64),
        "moments": np.stack([xx, yy, xy], axis=1).astype(np.float32),
        "footprint": footprint.astype(np.float32),
    }


def _ellipse_parameters(moments: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return major.astype(np.float32), minor.astype(np.float32), angle.astype(np.float32)


def make_targets(
    *,
    image_shape: Tuple[int, int],
    centers: np.ndarray,
    moments: np.ndarray,
    confidence_levels: int = 5,
    ellipse_sigma: float = 2.0,
    core_radius: int = 2,
) -> Dict[str, Tensor]:
    """Build CELLECT-style dense labels from accurate centers and noisy shapes."""

    h, w = image_shape
    seg = torch.zeros((h, w), dtype=torch.long)
    conf = torch.zeros((h, w), dtype=torch.long)
    shape = torch.zeros((3, h, w), dtype=torch.float32)
    shape_weight = torch.zeros((h, w), dtype=torch.float32)
    if centers.size == 0:
        return {"seg": seg, "confidence": conf, "shape": shape, "shape_weight": shape_weight}

    major, minor, angle = _ellipse_parameters(moments)
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
        augment: bool = False,
    ) -> None:
        self.records = list(records)
        self.fits_hdu = int(fits_hdu)
        self.confidence_levels = int(confidence_levels)
        self.ellipse_sigma = float(ellipse_sigma)
        self.core_radius = int(core_radius)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        rec = self.records[idx]
        image_np = read_fits_bands(rec.image_paths, hdu=self.fits_hdu)
        image = astro_zscale_preprocess(image_np)
        h, w = int(image.shape[-2]), int(image.shape[-1])
        catalog = load_meas_catalog(rec.meas_path, x0=rec.x0, y0=rec.y0, image_shape=(h, w))
        targets = make_targets(
            image_shape=(h, w),
            centers=catalog["centers"],
            moments=catalog["moments"],
            confidence_levels=self.confidence_levels,
            ellipse_sigma=self.ellipse_sigma,
            core_radius=self.core_radius,
        )
        centers = torch.from_numpy(catalog["centers"])
        if self.augment and random.random() < 0.5:
            image = torch.flip(image, dims=(-1,))
            targets["seg"] = torch.flip(targets["seg"], dims=(-1,))
            targets["confidence"] = torch.flip(targets["confidence"], dims=(-1,))
            targets["shape"] = torch.flip(targets["shape"], dims=(-1,))
            targets["shape_weight"] = torch.flip(targets["shape_weight"], dims=(-1,))
            centers = centers.clone()
            centers[:, 0] = float(w - 1) - centers[:, 0]

        return {
            "image": image,
            "seg": targets["seg"],
            "confidence": targets["confidence"],
            "shape": targets["shape"],
            "shape_weight": targets["shape_weight"],
            "centers": centers,
            "ids": torch.from_numpy(catalog["ids"]),
            "name": rec.name,
        }


def collate_cutouts(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "image": torch.stack([item["image"] for item in batch]),  # type: ignore[index]
        "seg": torch.stack([item["seg"] for item in batch]),  # type: ignore[index]
        "confidence": torch.stack([item["confidence"] for item in batch]),  # type: ignore[index]
        "shape": torch.stack([item["shape"] for item in batch]),  # type: ignore[index]
        "shape_weight": torch.stack([item["shape_weight"] for item in batch]),  # type: ignore[index]
        "centers": [item["centers"] for item in batch],
        "ids": [item["ids"] for item in batch],
        "name": [item["name"] for item in batch],
    }


class HardTripletLoss(nn.Module):
    """CELLECT/Open-ReID style hard triplet loss with margin 0.3."""

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        labels = labels.long()
        valid = labels > 0
        features = features[valid]
        labels = labels[valid]
        if features.shape[0] < 4 or torch.unique(labels).numel() < 2:
            return features.sum() * 0.0
        dist = torch.cdist(features, features, p=2)
        same = labels[:, None] == labels[None, :]
        eye = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
        pos_mask = same & ~eye
        neg_mask = ~same
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
) -> Dict[str, Tensor]:
    seg_target = batch["seg"].to(device=device, dtype=torch.long)  # type: ignore[union-attr]
    conf_target = batch["confidence"].to(device=device, dtype=torch.long)  # type: ignore[union-attr]
    shape_target = batch["shape"].to(device=device, dtype=torch.float32)  # type: ignore[union-attr]
    shape_weight = batch["shape_weight"].to(device=device, dtype=torch.float32)  # type: ignore[union-attr]

    seg_weight = torch.tensor(weights.segmentation_3cls, device=device, dtype=torch.float32)
    seg_loss = F.cross_entropy(outputs["seg_logits"], seg_target, weight=seg_weight)
    conf_loss = ordinal_confidence_loss(
        outputs["confidence"],
        conf_target,
        pos_weight=weights.confidence_pos_weight,
    )
    per_pixel_shape = F.mse_loss(outputs["shape"], shape_target, reduction="none").mean(dim=1)
    if bool((shape_weight > 0).any()):
        shape_loss = (per_pixel_shape * shape_weight).sum() / shape_weight.sum().clamp_min(1.0)
    else:
        shape_loss = per_pixel_shape.mean() * 0.0
    total = seg_loss + conf_loss + shape_loss
    return {"total": total, "seg": seg_loss, "confidence": conf_loss, "shape": shape_loss}


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


@torch.no_grad()
def detect_centers(outputs: Dict[str, Tensor], *, threshold: float = 0.0, nms_radius: int = 2) -> List[np.ndarray]:
    """Local maxima of the top confidence channel inside predicted foreground."""

    conf = outputs["confidence"][:, -1]
    fg = outputs["seg_logits"].argmax(dim=1) > 0
    pooled = F.max_pool2d(conf.unsqueeze(1), kernel_size=2 * nms_radius + 1, stride=1, padding=nms_radius).squeeze(1)
    peaks = (conf == pooled) & fg & (conf > threshold)
    result: List[np.ndarray] = []
    for b in range(conf.shape[0]):
        y, x = torch.where(peaks[b])
        coords = torch.stack([x, y], dim=1).detach().cpu().numpy().astype(np.float32)
        result.append(coords)
    return result


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
    model: AstroUNet2D,
    loader: DataLoader,
    *,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    weights: LossWeights,
    triplet_loss_fn: HardTripletLoss,
    triplet_enabled: bool,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums: Dict[str, float] = {"total": 0.0, "seg": 0.0, "confidence": 0.0, "shape": 0.0, "triplet": 0.0}
    count = 0
    for batch in tqdm(loader, desc="train" if training else "eval", leave=False):
        image = batch["image"].to(device=device, dtype=torch.float32)
        outputs = model(image)
        losses = dense_losses(outputs, batch, weights=weights, device=device)
        total = losses["total"]

        triplet = total.new_tensor(0.0)
        if triplet_enabled:
            features, batch_ids = _sample_embeddings_at_centers(outputs, batch["centers"])
            # IDs are unique inside one catalog in most cutouts; this hook keeps
            # CELLECT's triplet term available when overlapping catalogs produce
            # repeated object IDs in the same batch.
            labels_list = [ids.to(device=device, dtype=torch.long) for ids in batch["ids"]]
            if labels_list:
                labels = torch.cat(labels_list, dim=0)
                if labels.shape[0] == features.shape[0] and torch.unique(labels).numel() < labels.numel():
                    triplet = triplet_loss_fn(features, labels)
                    total = total + weights.triplet_outer_weight * triplet

        if training:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()

        batch_size = int(image.shape[0])
        count += batch_size
        sums["total"] += float(total.detach()) * batch_size
        sums["seg"] += float(losses["seg"].detach()) * batch_size
        sums["confidence"] += float(losses["confidence"].detach()) * batch_size
        sums["shape"] += float(losses["shape"].detach()) * batch_size
        sums["triplet"] += float(triplet.detach()) * batch_size
    return {key: val / max(count, 1) for key, val in sums.items()}


@torch.no_grad()
def evaluate_detection(
    model: AstroUNet2D,
    loader: DataLoader,
    *,
    device: torch.device,
    threshold: float,
    match_radius: float,
) -> Dict[str, float]:
    model.eval()
    tp = fp = fn = 0
    for batch in tqdm(loader, desc="detect", leave=False):
        image = batch["image"].to(device=device, dtype=torch.float32)
        outputs = model(image)
        pred_list = detect_centers(outputs, threshold=threshold)
        for pred_xy, gt_xy in zip(pred_list, batch["centers"]):
            t, f, n = match_points(pred_xy, gt_xy.numpy().astype(np.float32), match_radius)
            tp += t
            fp += f
            fn += n
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {"tp": float(tp), "fp": float(fp), "fn": float(fn), "precision": precision, "recall": recall, "f1": f1}


def split_records(records: Sequence[CutoutRecord], val_fraction: float, seed: int) -> Tuple[List[CutoutRecord], List[CutoutRecord]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
    return shuffled[n_val:], shuffled[:n_val]


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
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ellipse-sigma", type=float, default=2.0)
    parser.add_argument("--core-radius", type=int, default=2)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--match-radius", type=float, default=4.0)
    parser.add_argument("--enable-triplet", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = _expand_path(args.root)
    reference_dir = _expand_path(args.reference_dir) if args.reference_dir else None
    cutout_dir = _expand_path(args.cutout_dir) if args.cutout_dir else None
    out_dir = _expand_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = discover_cutout_records(
        root,
        reference_dir=reference_dir,
        cutout_dir=cutout_dir,
        bands=args.bands,
        max_records=args.max_records,
    )
    train_records, val_records = split_records(records, args.val_fraction, args.seed)
    if args.mode == "eval":
        val_records = records

    weights = LossWeights()
    common_ds = dict(
        fits_hdu=args.fits_hdu,
        confidence_levels=5,
        ellipse_sigma=args.ellipse_sigma,
        core_radius=args.core_radius,
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

    device = torch.device(args.device)
    model = AstroUNet2D(
        in_channels=len(args.bands),
        seg_classes=3,
        confidence_levels=5,
        embedding_dim=args.embedding_dim,
        base_channels=args.base_channels,
        shape_channels=3,
    ).to(device)
    ex_net = AstroMatchNet2D(feature_dim=args.embedding_dim).to(device)
    en_net = AstroMatchNet2D(feature_dim=args.embedding_dim).to(device)

    if args.checkpoint:
        ckpt = torch.load(_expand_path(args.checkpoint), map_location=device)
        model.load_state_dict(ckpt["model"])
        if "EX" in ckpt:
            ex_net.load_state_dict(ckpt["EX"])
        if "EN" in ckpt:
            en_net.load_state_dict(ckpt["EN"])

    if args.mode == "eval":
        dense = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            weights=weights,
            triplet_loss_fn=HardTripletLoss(weights.triplet_margin),
            triplet_enabled=args.enable_triplet,
        )
        det = evaluate_detection(
            model,
            val_loader,
            device=device,
            threshold=args.confidence_threshold,
            match_radius=args.match_radius,
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
    }
    (out_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

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
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            weights=weights,
            triplet_loss_fn=triplet_loss_fn,
            triplet_enabled=args.enable_triplet,
        )
        det_metrics = evaluate_detection(
            model,
            val_loader,
            device=device,
            threshold=args.confidence_threshold,
            match_radius=args.match_radius,
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
            "EX": ex_net.state_dict(),
            "EN": en_net.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "loss_weights": asdict(weights),
            "val": val_metrics,
            "detection": det_metrics,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save(ckpt, out_dir / "best.pt")


if __name__ == "__main__":
    main()
