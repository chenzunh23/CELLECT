#!/usr/bin/env python3
"""Run a CELLECT sam_per_band checkpoint and visualize detector outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")

import numpy as np
import torch
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_utils import (
    build_prompt_masks,
    build_scaled_tensor_from_fits,
    confidence_overlay,
    detection_rows,
    draw_ellipses,
    infer_cellect,
    load_cellect_model,
    mask_overlay,
    read_zarr_sample,
    resolve_zarr_sample,
    rows_to_reg,
    save_heatmap,
    save_png,
    scaled_rgb_for_display,
    select_band_outputs,
    shifted_crop_header,
    write_mask_reg,
    write_reg,
    write_sources_csv,
    zarr_sample_group,
    zscale_rgb,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", choices=("none", "bf16"), default="bf16")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, action="append", help="Input FITS file. Repeat for multiple bands.")
    src.add_argument("--zarr-store", type=Path, help="Direct-zarr store to visualize.")
    src.add_argument("--root", type=Path, help="Direct-zarr root used to select coadd/noisy/denoised samples.")
    p.add_argument("--band", action="append", default=None, help="Band labels for FITS input; repeat with --input.")
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--zarr-band", default=None, help="Band inside zarr store.")
    p.add_argument("--patch", default=None)
    p.add_argument("--tile-name", default=None)
    p.add_argument(
        "--dataset-source",
        nargs="*",
        default=None,
        help="Dataset source(s) for --root: coadd, noisy, denoised. Use 'all' for coadd noisy denoised.",
    )
    p.add_argument("--group", default=None, help="Validate zarr sample group, e.g. group_00 or 0.")
    p.add_argument("--image-level", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--hdu", type=int, default=None)
    p.add_argument("--x0", type=int, default=0)
    p.add_argument("--y0", type=int, default=0)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument(
        "--scaling-mode",
        default="zscore_clip",
        choices=("zscore_clip", "zscore_no_clip", "zscore_no_upper", "log_lupton", "anscombe"),
        help="Scaling used when --input FITS is provided.",
    )
    p.add_argument("--log-a", type=float, default=300.0)
    p.add_argument("--anscombe-scale", type=float, default=1000.0)

    p.add_argument("--confidence-threshold", type=float, default=2.0)
    p.add_argument("--confidence-score", default="ordinal_expectation")
    p.add_argument("--nms-radius", type=int, default=3)
    p.add_argument("--center-refinement", choices=("integer", "weighted_centroid", "softargmax"), default="softargmax")
    p.add_argument("--center-refinement-radius", type=int, default=1)
    p.add_argument("--make-masks", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mask-threshold", type=float, default=0.0)
    p.add_argument("--mask-box-scale", type=float, default=2.0)
    p.add_argument("--mask-chunk-size", type=int, default=512)
    p.add_argument("--mask-multimask", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--out-dir", type=Path, default=Path("output/eval_visualizations/cellect_outputs"))
    return p.parse_args()


def _dataset_sources(values: list[str] | None) -> list[str | None]:
    if not values:
        return [None]
    normalized = [str(value).strip() for value in values if str(value).strip()]
    if any(value.lower() == "all" for value in normalized):
        return ["coadd", "noisy", "denoised"]
    return normalized


def _score_map(outputs: dict[str, torch.Tensor]) -> np.ndarray:
    logits = outputs["confidence"][0].detach().float().cpu()
    prob = torch.softmax(logits, dim=0)
    levels = torch.arange(logits.shape[0], dtype=prob.dtype).view(-1, 1, 1)
    return (prob * levels).sum(dim=0).numpy().astype(np.float32)


def _pred_conf_overlay(image: np.ndarray, outputs: dict[str, torch.Tensor]) -> np.ndarray:
    score = _score_map(outputs)
    levels = np.clip(np.rint(score), 0, 4).astype(np.uint8)
    return confidence_overlay(image, levels)


def _prepare_from_zarr(args: argparse.Namespace, dataset_source: str | None = None):
    effective_group = None if dataset_source == "coadd" else args.group
    reader, sample_idx, band_idx, attrs = resolve_zarr_sample(
        zarr_store=args.zarr_store.expanduser().resolve() if args.zarr_store else None,
        sample_index=int(args.sample_index),
        root=args.root.expanduser().resolve() if args.root else None,
        patch=args.patch,
        tile_name=args.tile_name,
        band=args.zarr_band,
        dataset_source=dataset_source,
        group=effective_group,
        image_level=bool(args.image_level),
    )
    sample = read_zarr_sample(reader, sample_idx, band_idx)
    image = np.asarray(sample["image"], dtype=np.float32)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=0)
    tensor = torch.from_numpy(image[None, None].astype(np.float32, copy=False))
    display = np.asarray(sample["display_image"], dtype=np.float32)
    band = (attrs.get("bands") or [args.zarr_band or "band0"])[band_idx]
    dataset = str(attrs.get("dataset_source", dataset_source or "zarr"))
    group_name = zarr_sample_group(reader, sample_idx)
    stem_base = Path(args.zarr_store).stem if args.zarr_store else f"{dataset}_{attrs.get('patch', args.patch or 'patch')}"
    group_part = f"_{group_name}" if group_name else ""
    stem = f"{stem_base}{group_part}_sample{sample_idx:05d}_{band}"
    x0 = int(reader.read_full_small("tile_x0")[sample_idx]) if reader.has_array("tile_x0") else 0
    y0 = int(reader.read_full_small("tile_y0")[sample_idx]) if reader.has_array("tile_y0") else 0
    return tensor, [display], [image], [None], [str(band)], stem, [x0], [y0]


def _prepare_from_fits(args: argparse.Namespace):
    paths = [path.expanduser().resolve() for path in args.input]
    width = int(args.width or args.size)
    height = int(args.height or args.size)
    tensor, raw_images, scaled_images, headers = build_scaled_tensor_from_fits(
        paths,
        hdu=args.hdu,
        x0=int(args.x0),
        y0=int(args.y0),
        width=width,
        height=height,
        scaling_mode=str(args.scaling_mode),
        log_a=float(args.log_a),
        anscombe_scale=float(args.anscombe_scale),
    )
    bands = args.band or [path.stem for path in paths]
    if len(bands) != len(paths):
        raise ValueError("--band must be repeated once per --input")
    stem = f"x{args.x0}_y{args.y0}_{str(args.scaling_mode).replace('_', '-')}"
    return tensor, raw_images, scaled_images, headers, bands, stem, [int(args.x0)] * len(paths), [int(args.y0)] * len(paths)


def _run_one(args: argparse.Namespace, dataset_source: str | None = None) -> Path:
    width = int(args.width or args.size)
    height = int(args.height or args.size)
    if args.input:
        tensor, raw_images, scaled_images, headers, bands, stem, image_x0, image_y0 = _prepare_from_fits(args)
    else:
        tensor, raw_images, scaled_images, headers, bands, stem, image_x0, image_y0 = _prepare_from_zarr(args, dataset_source)
        height, width = int(tensor.shape[-2]), int(tensor.shape[-1])

    device = torch.device(args.device)
    model, cfg = load_cellect_model(
        args.checkpoint.expanduser().resolve(),
        args.config.expanduser().resolve() if args.config else None,
        device,
        bands,
    )
    outputs = infer_cellect(model=model, image_tensor=tensor, device=device, amp=str(args.amp))
    out_dir = args.out_dir.expanduser().resolve() / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for band_idx, band in enumerate(bands):
        band_outputs = select_band_outputs(outputs, band_idx)
        image = raw_images[band_idx]
        scaled = scaled_images[band_idx]
        header = headers[band_idx]
        band_dir = out_dir / str(band)
        band_dir.mkdir(parents=True, exist_ok=True)
        rows = detection_rows(
            band_outputs,
            threshold=float(args.confidence_threshold),
            nms_radius=int(args.nms_radius),
            confidence_score=str(args.confidence_score),
            center_refinement=str(args.center_refinement),
            center_refinement_radius=int(args.center_refinement_radius),
            width=width,
            height=height,
        )
        score = _score_map(band_outputs)[:height, :width]
        save_heatmap(band_dir / f"{band}_confidence_score.png", score, title=f"{band} confidence score")
        save_png(band_dir / f"{band}_confidence_overlay.png", _pred_conf_overlay(image, band_outputs), title=f"{band} confidence")
        save_png(band_dir / f"{band}_input_rgb.png", scaled_rgb_for_display(scaled), title=f"{band} model input")
        for channel in range(min(3, scaled.shape[0])):
            save_heatmap(band_dir / f"{band}_input_channel{channel}.png", scaled[channel], title=f"{band} input channel {channel}")
        write_reg(band_dir / f"{band}_centers.reg", rows_to_reg(rows, shape=False, color="yellow"))
        write_reg(band_dir / f"{band}_shapes.reg", rows_to_reg(rows, shape=True, color="cyan"))
        write_sources_csv(band_dir / f"{band}_sources.csv", rows, header=header, x0=int(image_x0[band_idx]), y0=int(image_y0[band_idx]))
        save_png(band_dir / f"{band}_shape_overlay.png", draw_ellipses(image, rows), title=f"{band} predicted shapes")

        mask_count = 0
        if bool(args.make_masks):
            label_map = build_prompt_masks(
                model,
                band_outputs,
                rows,
                device=device,
                width=width,
                height=height,
                threshold=float(args.mask_threshold),
                box_scale=float(args.mask_box_scale),
                chunk_size=int(args.mask_chunk_size),
                multimask=bool(args.mask_multimask),
            )
            mask_count = int(np.count_nonzero(np.unique(label_map) > 0))
            mask_header = (
                shifted_crop_header(header, x0=int(image_x0[band_idx]), y0=int(image_y0[band_idx]))
                if header is not None
                else fits.Header()
            )
            fits.PrimaryHDU(label_map.astype(np.int32), header=mask_header).writeto(
                band_dir / f"{band}_mask_labelmap.fits",
                overwrite=True,
            )
            write_mask_reg(band_dir / f"{band}_mask_contours.reg", label_map)
            save_png(band_dir / f"{band}_mask_overlay.png", mask_overlay(image, label_map), title=f"{band} masks")

        summaries.append({"band": str(band), "detections": len(rows), "masks": mask_count})

    (out_dir / "summary.json").write_text(json.dumps({"summary": summaries, "config": cfg}, indent=2, default=str) + "\n", encoding="utf-8")
    return out_dir


def main() -> int:
    args = parse_args()
    if args.input or args.zarr_store:
        if args.dataset_source:
            raise ValueError("--dataset-source is only used with --root")
        out_dirs = [_run_one(args, None)]
    else:
        out_dirs = [_run_one(args, dataset_source) for dataset_source in _dataset_sources(args.dataset_source)]
    for out_dir in out_dirs:
        print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
