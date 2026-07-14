#!/usr/bin/env python3
"""Export GT and predicted confidence-map overlays for the Zangetsu SAM cutout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from torch.utils.data import DataLoader

CELLECT_ROOT = Path("/home/czh23/CELLECT")
if str(CELLECT_ROOT) not in sys.path:
    sys.path.insert(0, str(CELLECT_ROOT))

from astro_train_data import AstroCutoutDataset, collate_cutouts, discover_cutout_records  # noqa: E402
from astro_train_ops import model_forward_with_batch_context  # noqa: E402
from sam_backbone import build_sam_cellect2d  # noqa: E402


DEFAULT_DATA_ROOT = CELLECT_ROOT / "zangetsu_demo/data/sam_x18204_y20924"
DEFAULT_OUT_DIR = CELLECT_ROOT / "zangetsu_demo/output/confidence_map_overlays_0703"
DEFAULT_CKPT = Path("/nvme0/zc/scarlet/ckpts/sam_new_bkgd_cdn_0702/epoch_0016.pt") # /nvme0/zc/scarlet/ckpts/sam_detector_cdn_0628/epoch_0018.pt
DEFAULT_CONFIG = Path("/nvme0/zc/scarlet/ckpts/sam_new_bkgd_cdn_0702/run_config.json")
DEFAULT_BANDS = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y")
TRACT = "9813"
PATCH = "4,5"
BASE_TILE = "sam_x18204_y20924"
VARIANT_GROUP = "group_01"

LEVEL_COLORS = {
    1: np.array([0.10, 0.45, 1.00], dtype=np.float32),
    2: np.array([0.00, 0.85, 0.35], dtype=np.float32),
    3: np.array([1.00, 0.85, 0.05], dtype=np.float32),
    4: np.array([1.00, 0.05, 0.05], dtype=np.float32),
}
CLEAN_COLOR = np.array([0.0, 0.95, 1.0], dtype=np.float32)
CENTER_ONLY_COLOR = np.array([1.0, 0.2, 1.0], dtype=np.float32)
IGNORE_COLOR = np.array([0.8, 0.1, 0.1], dtype=np.float32)


def _read_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    args = dict(payload.get("args", {}))
    args["_top"] = payload
    return args


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key[7:] if str(key).startswith("module.") else key: value for key, value in state.items()}


def _checkpoint_variant(checkpoint: Path) -> str:
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if isinstance(ckpt, dict):
        if ckpt.get("model_variant"):
            return str(ckpt["model_variant"])
        if isinstance(ckpt.get("args"), dict) and ckpt["args"].get("model_variant"):
            return str(ckpt["args"]["model_variant"])
    if isinstance(state, dict):
        first_key = next(iter(state), "")
        if str(first_key).startswith(("encoder.", "module.encoder.")):
            return "sam_per_band"
    return ""


def _make_model(cfg: dict, checkpoint: Path, device: torch.device, bands: Sequence[str]) -> torch.nn.Module:
    top = cfg.get("_top", {})
    variant = _checkpoint_variant(checkpoint) or str(top.get("model_variant") or cfg.get("model_variant", "sam_per_band"))
    if variant != "sam_per_band":
        raise ValueError(f"{checkpoint} is model_variant={variant!r}; expected sam_per_band")

    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    film_from_state = isinstance(state, dict) and any(
        "decoder.denoised_film." in str(key) for key in state
    )
    style_from_state = isinstance(state, dict) and any(
        "encoder.style_router." in str(key) or "encoder.image_encoder.style_adapters." in str(key)
        for key in state
    )
    style_enabled = bool(
        top.get("sam_encoder_style_prompt", cfg.get("sam_encoder_style_prompt", False))
        or style_from_state
    )
    base_channels = int(top.get("base_channels") or cfg.get("base_channels", 32))
    model = build_sam_cellect2d(
        str(top.get("sam_model_type") or cfg.get("sam_model_type", "vit_b")),
        checkpoint=None,
        num_bands=len(bands),
        image_size=512,
        patch_size=16,
        seg_classes=int(top.get("seg_classes") or cfg.get("seg_classes", 2)),
        confidence_levels=5,
        embedding_dim=int(cfg.get("embedding_dim", 64)),
        shape_channels=3,
        decoder_channels=(base_channels * 8, base_channels * 4, base_channels * 2, base_channels),
        use_cen=bool(top.get("sam_cen_enabled", not bool(cfg.get("disable_sam_cen", False)))),
        cen_input_image=True,
        cen_width=max(2, base_channels // 4),
        decoder_denoised_film=bool(
            top.get("sam_decoder_film", cfg.get("sam_decoder_film", False)) or film_from_state
        ),
        encoder_style_prompt=style_enabled,
        style_prompt_dim=int(top.get("style_prompt_dim", cfg.get("style_prompt_dim", 32))),
        style_prompt_layers=tuple(top.get("style_prompt_layers", cfg.get("style_prompt_layers", (2, 5, 8)))),
        style_adapter_dim=int(top.get("style_adapter_dim", cfg.get("style_adapter_dim", 32))),
        style_router_temperature=float(
            top.get("style_router_temperature", cfg.get("style_router_temperature", 1.0))
        ),
        candidate_count=int(cfg.get("matcher_candidate_count", 5)),
        shape_feature_dim=6,
        enable_matchers=False,
        astro_preprocess_in_model=False,
    ).to(device)

    incompatible = model.load_state_dict(_strip_module_prefix(state), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"[load] {checkpoint.name}: missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)}",
            flush=True,
        )
    model.eval()
    return model


def _tile_for_dataset(dataset: str, variant_group: str) -> str:
    if dataset == "coadd":
        return BASE_TILE
    return f"{variant_group}_{BASE_TILE}" if variant_group else BASE_TILE


def _dataset(root: Path, dataset: str, bands: Sequence[str], cfg: dict, image_cache_dir: Path | None, tile_name: str) -> DataLoader:
    records = discover_cutout_records(root, bands=bands)
    records = [
        rec
        for rec in records
        if str(rec.dataset_source) == dataset and rec.patch == PATCH and rec.tile_name == tile_name
    ]
    if len(records) != 1:
        raise RuntimeError(f"Expected one record for {dataset}/{PATCH}/{tile_name}, got {len(records)} under {root}")
    ds = AstroCutoutDataset(
        records,
        fits_hdu=int(cfg.get("fits_hdu", 1)),
        confidence_levels=5,
        ellipse_sigma=float(cfg.get("ellipse_sigma", 2.0)),
        core_radius=int(cfg.get("core_radius", 2)),
        shape_source=str(cfg.get("shape_source", "kron")),
        source_filter=str(cfg.get("source_filter", "nchild0")),
        image_cache_dir=None if dataset == "coadd" else image_cache_dir,
        load_eval_ignore_sources=True,
        augment=False,
    )
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_cutouts)


def _raw_band_image(batch: dict, band_idx: int, hdu: int) -> np.ndarray:
    image_paths = batch.get("image_paths")
    if not image_paths:
        raise KeyError("batch has no image_paths")
    if len(image_paths) == 1 and isinstance(image_paths[0], (list, tuple)):
        path = Path(image_paths[0][band_idx])
    elif len(image_paths) > band_idx and isinstance(image_paths[band_idx], (list, tuple)):
        path = Path(image_paths[band_idx][0])
    else:
        path = Path(image_paths[band_idx])
    data = fits.getdata(path, ext=int(hdu))
    return np.asarray(data, dtype=np.float32)


def _zscale_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not bool(finite.any()):
        scaled = np.zeros_like(image, dtype=np.float32)
    else:
        interval = ZScaleInterval()
        try:
            vmin, vmax = interval.get_limits(image[finite])
        except Exception:
            vmin, vmax = float(np.nanpercentile(image[finite], 1.0)), float(np.nanpercentile(image[finite], 99.0))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = float(np.nanmin(image[finite])), float(np.nanmax(image[finite]))
        scaled = np.clip((image - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    return np.repeat(scaled[..., None], 3, axis=2)


def _alpha_blend(rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> None:
    mask = np.asarray(mask, dtype=bool)
    if not bool(mask.any()):
        return
    rgb[mask] = (1.0 - float(alpha)) * rgb[mask] + float(alpha) * color[None, :]


def _gt_overlay(image: np.ndarray, targets: dict[str, torch.Tensor]) -> np.ndarray:
    rgb = _zscale_rgb(image)
    clean = targets["band_clean_mask"].detach().cpu().numpy().astype(bool)
    center_only = targets["band_center_only_mask"].detach().cpu().numpy().astype(bool)
    ignored = targets["band_ignore_mask"].detach().cpu().numpy().astype(bool)
    confidence = targets["band_confidence"].detach().cpu().numpy().astype(np.int16)

    _alpha_blend(rgb, clean, CLEAN_COLOR, 0.16)
    _alpha_blend(rgb, center_only, CENTER_ONLY_COLOR, 0.22)
    _alpha_blend(rgb, ignored, IGNORE_COLOR, 0.18)
    for level in (1, 2, 3, 4):
        _alpha_blend(rgb, confidence == level, LEVEL_COLORS[level], 0.82)
    return np.clip(rgb, 0.0, 1.0)


def _pred_overlay(image: np.ndarray, logits: torch.Tensor, *, min_prob: float) -> np.ndarray:
    rgb = _zscale_rgb(image)
    prob = torch.softmax(logits.float(), dim=0)
    pred_level = torch.argmax(prob, dim=0).detach().cpu().numpy().astype(np.int16)
    pred_prob = torch.max(prob, dim=0).values.detach().cpu().numpy().astype(np.float32)
    for level in (1, 2, 3, 4):
        mask = (pred_level == level) & (pred_prob >= float(min_prob))
        alpha = np.clip((pred_prob - float(min_prob)) / max(1.0 - float(min_prob), 1e-6), 0.0, 1.0)
        level_rgb = rgb[mask]
        if level_rgb.size:
            a = (0.18 + 0.58 * alpha[mask])[:, None]
            rgb[mask] = (1.0 - a) * level_rgb + a * LEVEL_COLORS[level][None, :]
    return np.clip(rgb, 0.0, 1.0)


def _ordinal_expectation(logits: torch.Tensor) -> np.ndarray:
    prob = torch.softmax(logits.float(), dim=0)
    levels = torch.arange(logits.shape[0], device=logits.device, dtype=prob.dtype).view(-1, 1, 1)
    return (prob * levels).sum(dim=0).detach().cpu().numpy().astype(np.float32)


def _save_image(path: Path, rgb: np.ndarray, title: str, legend: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    ax.imshow(rgb, origin="lower", interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if legend:
        patches = [
            mpatches.Patch(color=LEVEL_COLORS[i], label=f"level {i}") for i in (1, 2, 3, 4)
        ]
        patches.extend(
            [
                mpatches.Patch(color=CLEAN_COLOR, label="clean region"),
                mpatches.Patch(color=CENTER_ONLY_COLOR, label="center-only"),
                mpatches.Patch(color=IGNORE_COLOR, label="ignore"),
            ]
        )
        ax.legend(handles=patches, loc="lower right", fontsize=7, framealpha=0.72)
    fig.tight_layout(pad=0.2)
    fig.savefig(path)
    plt.close(fig)


def _save_score_heatmap(path: Path, image: np.ndarray, score: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    ax.imshow(_zscale_rgb(image), origin="lower", interpolation="nearest")
    im = ax.imshow(score, origin="lower", interpolation="nearest", cmap="magma", alpha=0.52, vmin=0.0, vmax=4.0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="ordinal expectation")
    fig.tight_layout(pad=0.2)
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--datasets", nargs="+", default=["coadd", "denoised", "noisy"])
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    parser.add_argument("--band", default="HSC-I")
    parser.add_argument("--variant-group", default=VARIANT_GROUP)
    parser.add_argument("--image-cache-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pred-min-prob", type=float, default=0.35)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    cfg = _read_config(args.config)
    if args.image_cache_dir is None:
        cache_from_cfg = cfg.get("image_cache_dir") or cfg.get("_top", {}).get("image_cache_dir")
        args.image_cache_dir = Path(str(cache_from_cfg)).expanduser().resolve() if cache_from_cfg else None
    else:
        args.image_cache_dir = args.image_cache_dir.expanduser().resolve()

    bands = tuple(str(b) for b in args.bands)
    if args.band not in bands:
        raise ValueError(f"--band {args.band!r} must be one of {bands}")
    band_idx = bands.index(args.band)
    device = torch.device(args.device)
    model = _make_model(cfg, args.checkpoint, device, bands)

    summary: list[dict[str, object]] = []
    for dataset in args.datasets:
        tile_name = _tile_for_dataset(str(dataset), str(args.variant_group))
        loader = _dataset(args.data_root, str(dataset), bands, cfg, args.image_cache_dir, tile_name)
        batch = next(iter(loader))
        image = batch["image"].to(device=device, dtype=torch.float32)
        raw = _raw_band_image(batch, band_idx, int(cfg.get("fits_hdu", 1)))
        with torch.no_grad():
            outputs = model_forward_with_batch_context(model, image, batch)
        logits = outputs["confidence"][0, band_idx]
        targets = {
            "band_confidence": batch["band_confidence"][0, band_idx],
            "band_clean_mask": batch["band_clean_mask"][0, band_idx],
            "band_center_only_mask": batch["band_center_only_mask"][0, band_idx],
            "band_ignore_mask": batch["band_ignore_mask"][0, band_idx],
        }

        gt_rgb = _gt_overlay(raw, targets)
        pred_rgb = _pred_overlay(raw, logits, min_prob=float(args.pred_min_prob))
        expectation = _ordinal_expectation(logits)

        prefix = f"{dataset}_{PATCH.replace(',', '_')}_{tile_name}_{args.band.replace('-', '_')}"
        gt_path = args.out_dir / f"{prefix}_gt_confidence_overlay.png"
        pred_path = args.out_dir / f"{prefix}_pred_confidence_argmax_overlay.png"
        score_path = args.out_dir / f"{prefix}_pred_ordinal_expectation_overlay.png"
        _save_image(gt_path, gt_rgb, f"{dataset} {PATCH}/{tile_name} {args.band}: GT confidence levels")
        _save_image(pred_path, pred_rgb, f"{dataset} {PATCH}/{tile_name} {args.band}: predicted confidence argmax")
        _save_score_heatmap(score_path, raw, expectation, f"{dataset} {PATCH}/{tile_name} {args.band}: predicted ordinal expectation")

        gt_conf = targets["band_confidence"].detach().cpu().numpy()
        pred_argmax = torch.argmax(torch.softmax(logits.float(), dim=0), dim=0).detach().cpu().numpy()
        summary.append(
            {
                "dataset": str(dataset),
                "tile": tile_name,
                "gt_overlay": str(gt_path),
                "pred_argmax_overlay": str(pred_path),
                "pred_ordinal_expectation_overlay": str(score_path),
                "gt_level_pixels": {str(level): int(np.count_nonzero(gt_conf == level)) for level in range(5)},
                "pred_argmax_pixels": {str(level): int(np.count_nonzero(pred_argmax == level)) for level in range(5)},
                "pred_expectation_min": float(np.nanmin(expectation)),
                "pred_expectation_max": float(np.nanmax(expectation)),
                "pred_expectation_mean": float(np.nanmean(expectation)),
            }
        )
        print(f"wrote {gt_path}")
        print(f"wrote {pred_path}")
        print(f"wrote {score_path}")

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
