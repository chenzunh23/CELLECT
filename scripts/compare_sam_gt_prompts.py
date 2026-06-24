#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

CELLECT_ROOT = Path("/home/czh23/CELLECT")
SAM_ROOT = Path("/home/czh23/SAM-astro")
for root in (CELLECT_ROOT, SAM_ROOT):
    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
sys.path.insert(0, str(CELLECT_ROOT))
sys.path.insert(0, str(SAM_ROOT))

from segment_anything import SamPredictor, sam_model_registry  # noqa: E402
from astro_train_data import AstroCutoutDataset, collate_cutouts, discover_cutout_records  # noqa: E402
from sam_backbone.losses import _boxes_from_centers_shapes  # noqa: E402


DEFAULT_DATA_ROOT = (
    CELLECT_ROOT
    / "output/sam_cellect_combination_260611/preprocessing_diagnostics_260611/zangetsu_preprocessed_cutouts_260611"
)
DEFAULT_BANDS = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--datasets", nargs="+", default=["coadd", "denoised"])
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patch", default="6,1")
    parser.add_argument("--tile", default="zangetsu_lower_right_x27366_y6453")
    parser.add_argument("--band", default="HSC-I")
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    parser.add_argument("--model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--checkpoint", default="/home/czh23/sam_ckpts/sam_vit_b_01ec64.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hdu", type=int, default=1)
    parser.add_argument("--ellipse-sigma", type=float, default=2.0)
    parser.add_argument("--shape-source", default="kron", choices=["kron", "sdss", "circular_kron"])
    parser.add_argument(
        "--prompt-source",
        default="clean",
        choices=["clean", "all"],
        help="Use only centers inside band_clean_mask by default; use all to reproduce the older diagnostic.",
    )
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", type=Path, default=Path("/tmp/sam_gt_prompt_comparison.json"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for PNG overlays. Defaults to <out-json parent>/sam_gt_prompt_overlays.",
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.35)
    parser.add_argument("--min-mask-area", type=float, default=15.0)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.15)
    parser.add_argument("--no-overlays", action="store_true", help="Disable PNG overlay writing.")
    return parser.parse_args()


def load_one_batch(args: argparse.Namespace, dataset_name: str) -> dict[str, object]:
    tract_root = args.data_root / dataset_name / args.tract
    records = discover_cutout_records(tract_root, bands=args.bands)
    records = [rec for rec in records if rec.patch == args.patch and rec.tile_name == args.tile]
    if len(records) != 1:
        raise RuntimeError(f"expected one record, got {len(records)} for {dataset_name}/{args.patch}/{args.tile}")
    ds = AstroCutoutDataset(
        records,
        fits_hdu=args.hdu,
        confidence_levels=5,
        ellipse_sigma=args.ellipse_sigma,
        core_radius=2,
        shape_source=args.shape_source,
        source_filter="nchild0",
        load_eval_ignore_sources=True,
        augment=False,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_cutouts)
    return next(iter(loader))


def sample_shapes(shape_chw: torch.Tensor, centers_xy: torch.Tensor) -> torch.Tensor:
    h, w = int(shape_chw.shape[-2]), int(shape_chw.shape[-1])
    x = centers_xy[:, 0].round().long().clamp(0, w - 1)
    y = centers_xy[:, 1].round().long().clamp(0, h - 1)
    return shape_chw[:, y, x].transpose(0, 1).contiguous()


def sample_hw(mask_hw: torch.Tensor, centers_xy: torch.Tensor) -> torch.Tensor:
    if centers_xy.numel() == 0:
        return mask_hw.new_zeros((0,), dtype=mask_hw.dtype)
    h, w = int(mask_hw.shape[-2]), int(mask_hw.shape[-1])
    x = centers_xy[:, 0].round().long().clamp(0, w - 1)
    y = centers_xy[:, 1].round().long().clamp(0, h - 1)
    return mask_hw[y, x]


def filter_centers_by_prompt_source(
    batch: dict[str, object],
    centers: torch.Tensor,
    *,
    band_idx: int,
    prompt_source: str,
) -> torch.Tensor:
    if prompt_source == "all":
        return centers
    clean_mask = batch["band_clean_mask"][0, band_idx].to(dtype=torch.bool)  # type: ignore[index]
    keep = sample_hw(clean_mask, centers).to(dtype=torch.bool)
    return centers[keep]


def filter_mask_outputs_by_area(
    masks: np.ndarray,
    iou: np.ndarray,
    lowres: np.ndarray,
    centers: np.ndarray,
    *,
    min_area: float,
    max_area_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if masks.shape[0] == 0:
        return masks, iou[:0], lowres[:0], centers[:0]
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(np.float32)
    max_area = float(max_area_ratio) * float(masks.shape[-2] * masks.shape[-1])
    keep = np.isfinite(areas) & (areas >= float(min_area))
    if max_area_ratio > 0:
        keep &= areas <= max_area
    return masks[keep], iou[keep], lowres[keep], centers[keep]

def run_prompts(
    predictor: SamPredictor,
    centers: torch.Tensor,
    boxes: torch.Tensor | None,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = predictor.device
    centers_np = centers.detach().cpu().numpy().astype(np.float32)
    transformed_centers = predictor.transform.apply_coords(centers_np[:, None, :], predictor.original_size)
    point_coords = torch.as_tensor(transformed_centers, dtype=torch.float32, device=device)
    point_labels = torch.ones((centers.shape[0], 1), dtype=torch.int64, device=device)
    transformed_boxes = None
    if boxes is not None:
        box_np = boxes.detach().cpu().numpy().astype(np.float32)
        transformed_boxes = torch.as_tensor(
            predictor.transform.apply_boxes(box_np, predictor.original_size),
            dtype=torch.float32,
            device=device,
        )

    masks_out = []
    ious_out = []
    lowres_out = []
    for start in range(0, centers.shape[0], chunk_size):
        end = min(start + chunk_size, centers.shape[0])
        masks, iou_pred, low_res = predictor.predict_torch(
            point_coords=point_coords[start:end],
            point_labels=point_labels[start:end],
            boxes=None if transformed_boxes is None else transformed_boxes[start:end],
            mask_input=None,
            multimask_output=True,
            return_logits=True,
        )
        best = torch.argmax(iou_pred, dim=1)
        row = torch.arange(best.shape[0], device=device)
        masks_out.append((masks[row, best] > 0).detach().cpu())
        ious_out.append(iou_pred[row, best].detach().cpu())
        lowres_out.append(low_res[row, best].detach().cpu())
    masks_np = torch.cat(masks_out if masks_out else [torch.zeros((1, 512, 512), dtype=torch.bool)], dim=0).numpy().astype(bool)
    ious_np = torch.cat(ious_out if ious_out else [torch.zeros((1, 1), dtype=torch.float32)], dim=0).numpy().astype(np.float32)
    lowres_np = torch.cat(lowres_out if lowres_out else [torch.zeros((1, 512, 512), dtype=torch.float32)], dim=0).numpy().astype(np.float32)
    return masks_np, ious_np, lowres_np


def stability_from_lowres(lowres: np.ndarray, offset: float = 1.0) -> np.ndarray:
    high = lowres > offset
    low = lowres > -offset
    inter = high.reshape(high.shape[0], -1).sum(axis=1)
    union = low.reshape(low.shape[0], -1).sum(axis=1)
    return inter.astype(np.float32) / np.maximum(union.astype(np.float32), 1.0)


def zscale_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not bool(finite.any()):
        return np.zeros_like(image, dtype=np.float32)
    vals = image[finite]
    try:
        from astropy.visualization import ZScaleInterval

        lo, hi = ZScaleInterval().get_limits(vals)
    except Exception:
        lo, hi = np.nanpercentile(vals, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def instance_color(index: int) -> np.ndarray:
    palette = np.asarray(
        [
            (0.00, 0.76, 0.94),
            (0.95, 0.18, 0.65),
            (1.00, 0.82, 0.12),
            (0.16, 0.72, 0.33),
            (0.18, 0.38, 1.00),
            (0.95, 0.22, 0.14),
            (0.90, 0.56, 0.12),
            (0.58, 0.32, 0.90),
        ],
        dtype=np.float32,
    )
    return palette[int(index) % len(palette)]


def write_mask_overlay(
    path: Path,
    image: np.ndarray,
    masks: np.ndarray,
    *,
    alpha: float,
    title: str,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cellect")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = zscale_image(image)
    rgb = np.repeat(base[..., None], 3, axis=2)
    if masks.size:
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
        # Draw large masks first so small masks remain visible on top.
        order = np.argsort(-areas)
        for color_index, mask_index in enumerate(order):
            mask = masks[int(mask_index)]
            if not bool(mask.any()):
                continue
            color = instance_color(color_index)
            rgb[mask] = (1.0 - float(alpha)) * rgb[mask] + float(alpha) * color

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 7.2), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.flipud(rgb), origin="upper", interpolation="nearest")
    ax.set_axis_off()
    if title:
        ax.text(
            0.012,
            0.988,
            title,
            transform=ax.transAxes,
            va="top",
            ha="left",
            color="white",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.45, "linewidth": 0, "pad": 2},
        )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def component_count(mask_pairs: list[tuple[int, int]], n: int) -> tuple[int, int]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in mask_pairs:
        union(a, b)
    groups: dict[int, int] = {}
    for i in range(n):
        r = find(i)
        groups[r] = groups.get(r, 0) + 1
    merged = [v for v in groups.values() if v > 1]
    return len(merged), int(sum(merged))


def summarize_masks(name: str, masks: np.ndarray, pred_iou: np.ndarray, lowres: np.ndarray, centers: np.ndarray) -> dict[str, object]:
    n = int(masks.shape[0])
    areas = masks.reshape(n, -1).sum(axis=1).astype(np.float32)
    stability = stability_from_lowres(lowres)
    center_x = np.rint(centers[:, 0]).astype(np.int64).clip(0, masks.shape[2] - 1)
    center_y = np.rint(centers[:, 1]).astype(np.int64).clip(0, masks.shape[1] - 1)
    contains = masks[:, center_y, center_x]
    own_center_inside = np.diag(contains)
    centers_covered_by_any = contains.any(axis=0)
    centers_covered_by_multiple = contains.sum(axis=0) >= 2
    masks_covering_other_center = (contains.sum(axis=1) >= 2)

    boxes = np.zeros((n, 4), dtype=np.int32)
    for i in range(n):
        ys, xs = np.nonzero(masks[i])
        if ys.size == 0:
            boxes[i] = 0
        else:
            boxes[i] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

    overlap_any_pairs = []
    overlap_iou_01_pairs = []
    overlap_iou_05_pairs = []
    overlap_iomin_05_pairs = []
    for i in range(n):
        ai = float(areas[i])
        if ai <= 0:
            continue
        x0i, y0i, x1i, y1i = boxes[i]
        for j in range(i + 1, n):
            aj = float(areas[j])
            if aj <= 0:
                continue
            x0j, y0j, x1j, y1j = boxes[j]
            x0, y0 = max(x0i, x0j), max(y0i, y0j)
            x1, y1 = min(x1i, x1j), min(y1i, y1j)
            if x0 >= x1 or y0 >= y1:
                continue
            inter = int(np.logical_and(masks[i, y0:y1, x0:x1], masks[j, y0:y1, x0:x1]).sum())
            if inter <= 0:
                continue
            union = ai + aj - inter
            iou = float(inter) / max(union, 1.0)
            iomin = float(inter) / max(min(ai, aj), 1.0)
            overlap_any_pairs.append((i, j))
            if iou >= 0.1:
                overlap_iou_01_pairs.append((i, j))
            if iou >= 0.5:
                overlap_iou_05_pairs.append((i, j))
            if iomin >= 0.5:
                overlap_iomin_05_pairs.append((i, j))

    merged_components, merged_prompts = component_count(overlap_iomin_05_pairs, n)
    return {
        "name": name,
        "prompts": n,
        "pred_iou_mean": float(np.mean(pred_iou)) if n else 0.0,
        "pred_iou_median": float(np.median(pred_iou)) if n else 0.0,
        "stability_mean": float(np.mean(stability)) if n else 0.0,
        "stability_median": float(np.median(stability)) if n else 0.0,
        "area_mean": float(np.mean(areas)) if n else 0.0,
        "area_median": float(np.median(areas)) if n else 0.0,
        "area_p10": float(np.percentile(areas, 10)) if n else 0.0,
        "area_p90": float(np.percentile(areas, 90)) if n else 0.0,
        "area_lt_15": int(np.count_nonzero(areas < 15)),
        "own_center_inside": int(np.count_nonzero(own_center_inside)),
        "gt_centers_covered_by_any_mask": int(np.count_nonzero(centers_covered_by_any)),
        "gt_centers_covered_by_multiple_masks": int(np.count_nonzero(centers_covered_by_multiple)),
        "masks_covering_multiple_gt_centers": int(np.count_nonzero(masks_covering_other_center)),
        "overlap_pair_any": len(overlap_any_pairs),
        "overlap_pair_iou_ge_0p1": len(overlap_iou_01_pairs),
        "overlap_pair_iou_ge_0p5": len(overlap_iou_05_pairs),
        "overlap_pair_iomin_ge_0p5": len(overlap_iomin_05_pairs),
        "overlap_iomin_ge_0p5_components": merged_components,
        "overlap_iomin_ge_0p5_prompts": merged_prompts,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    sam = sam_model_registry[args.model_type](
        checkpoint=args.checkpoint,
        scaling_mode="astro_rgb",
        astro_rgb_mode="none",
        astro_preprocess_in_model=True,
        astro_preprocess_clip_sigma=3.0,
        astro_preprocess_sigma_iters=-1,
        astro_preprocess_z_clip=(-3.0, 3.0),
    ).to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    out_dir = args.out_dir if args.out_dir is not None else args.out_json.parent / "sam_gt_prompt_overlays"

    all_results = {}
    for dataset_name in args.datasets:
        print(f"[dataset] {dataset_name}: loading batch", flush=True)
        batch = load_one_batch(args, dataset_name)
        band_idx = list(args.bands).index(args.band)
        image = batch["image"][0, band_idx].detach().cpu().numpy().astype(np.float32)
        sam_input = np.repeat(image[..., None], 3, axis=2).astype(np.float32)
        centers = batch["band_centers"][0][band_idx].to(dtype=torch.float32)
        original_center_count = int(centers.shape[0])
        centers = filter_centers_by_prompt_source(
            batch,
            centers,
            band_idx=band_idx,
            prompt_source=str(args.prompt_source),
        )
        if args.max_prompts > 0 and centers.shape[0] > args.max_prompts:
            keep = torch.randperm(centers.shape[0])[: args.max_prompts]
            centers = centers[keep]
        shape_map = batch["band_shape"][0, band_idx].to(dtype=torch.float32)
        shapes = sample_shapes(shape_map, centers)
        boxes = _boxes_from_centers_shapes(
            centers,
            shapes,
            image_size=(int(image.shape[0]), int(image.shape[1])),
            ellipse_sigma=float(args.ellipse_sigma),
        )

        print(
            f"[dataset] {dataset_name}: prompts={centers.shape[0]} "
            f"(source={args.prompt_source}, original={original_center_count}) set_image",
            flush=True,
        )
        predictor.set_image(sam_input, image_format="RGB")
        print(f"[dataset] {dataset_name}: center-only prompts", flush=True)
        masks_center, iou_center, lowres_center = run_prompts(
            predictor,
            centers.to(device=device),
            None,
            chunk_size=args.chunk_size,
        )
        print(f"[dataset] {dataset_name}: center+bbox prompts", flush=True)
        masks_box, iou_box, lowres_box = run_prompts(
            predictor,
            centers.to(device=device),
            boxes.to(device=device),
            chunk_size=args.chunk_size,
        )

        centers_np = centers.detach().cpu().numpy().astype(np.float32)
        masks_center, iou_center, lowres_center, centers_center_np = filter_mask_outputs_by_area(
            masks_center,
            iou_center,
            lowres_center,
            centers_np,
            min_area=float(args.min_mask_area),
            max_area_ratio=float(args.max_mask_area_ratio),
        )
        masks_box, iou_box, lowres_box, centers_box_np = filter_mask_outputs_by_area(
            masks_box,
            iou_box,
            lowres_box,
            centers_np,
            min_area=float(args.min_mask_area),
            max_area_ratio=float(args.max_mask_area_ratio),
        )
        print(f"[dataset] {dataset_name}: summarize center-only", flush=True)
        center_stats = summarize_masks("center_only", masks_center, iou_center, lowres_center, centers_center_np)
        print(f"[dataset] {dataset_name}: summarize center+bbox", flush=True)
        box_stats = summarize_masks("center_plus_bbox", masks_box, iou_box, lowres_box, centers_box_np)
        overlay_paths: dict[str, str] = {}
        if not args.no_overlays:
            center_png = out_dir / f"{dataset_name}_{args.band}_center_only_overlay.png"
            box_png = out_dir / f"{dataset_name}_{args.band}_center_plus_bbox_overlay.png"
            write_mask_overlay(
                center_png,
                image,
                masks_center,
                alpha=float(args.overlay_alpha),
                title=f"{dataset_name} {args.band} center only",
            )
            write_mask_overlay(
                box_png,
                image,
                masks_box,
                alpha=float(args.overlay_alpha),
                title=f"{dataset_name} {args.band} center + bbox",
            )
            overlay_paths = {
                "center_only": str(center_png),
                "center_plus_bbox": str(box_png),
            }
            print(f"[dataset] {dataset_name}: wrote overlays to {out_dir}", flush=True)
        diff = {
            key: box_stats[key] - center_stats[key]
            for key in center_stats
            if key != "name" and isinstance(center_stats[key], (int, float)) and isinstance(box_stats.get(key), (int, float))
        }
        all_results[dataset_name] = {
            "band": args.band,
            "gt_prompts": int(centers.shape[0]),
            "original_centers": original_center_count,
            "prompt_source": str(args.prompt_source),
            "center_only": center_stats,
            "center_plus_bbox": box_stats,
            "box_minus_center": diff,
            "overlay_png": overlay_paths,
        }
        predictor.reset_image()

    args.out_json.write_text(json.dumps(all_results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(all_results, indent=2, sort_keys=True))
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
