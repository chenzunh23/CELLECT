#!/usr/bin/env python3
"""Run native SAM AMG on HSC FITS crops with CELLECT data-filtering scalings."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")

import numpy as np
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_utils import (
    SamAutomaticMaskGenerator,
    crop_or_pad,
    labelmap_from_amg,
    load_native_sam,
    make_training_rgb,
    mask_overlay,
    read_fits_image,
    sam_uint8_from_scaled,
    save_heatmap,
    save_png,
    scaled_rgb_for_display,
    shifted_crop_header,
    write_mask_reg,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, action="append", required=True, help="Input FITS file. Repeat for multiple files.")
    p.add_argument("--band", action="append", default=None, help="Optional band labels, one per --input.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model-type", default="vit_b", choices=("vit_b", "vit_l", "vit_h"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--hdu", type=int, default=None)
    p.add_argument("--x0", type=int, default=0)
    p.add_argument("--y0", type=int, default=0)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument(
        "--scaling-mode",
        action="append",
        default=None,
        choices=("zscore_clip", "zscore_no_clip", "zscore_no_upper", "log_lupton", "anscombe"),
        help="Repeat to run selected scalings. Defaults to zscore_clip, zscore_no_clip, log_lupton, anscombe.",
    )
    p.add_argument("--log-a", type=float, default=300.0)
    p.add_argument("--anscombe-scale", type=float, default=1000.0)
    p.add_argument("--points-per-side", type=int, default=64)
    p.add_argument("--points-per-batch", type=int, default=64)
    p.add_argument("--pred-iou-thresh", type=float, default=0.8)
    p.add_argument("--stability-score-thresh", type=float, default=0.95)
    p.add_argument("--crop-n-layers", type=int, default=1)
    p.add_argument("--min-mask-region-area", type=int, default=15)
    p.add_argument("--out-dir", type=Path, default=Path("output/eval_visualizations/native_sam"))
    return p.parse_args()


def _write_ann_csv(path: Path, annotations: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("id", "area", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "predicted_iou", "stability_score")
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, ann in enumerate(annotations, start=1):
            bbox = ann.get("bbox", [np.nan, np.nan, np.nan, np.nan])
            writer.writerow(
                {
                    "id": idx,
                    "area": int(ann.get("area", 0)),
                    "bbox_x": float(bbox[0]),
                    "bbox_y": float(bbox[1]),
                    "bbox_w": float(bbox[2]),
                    "bbox_h": float(bbox[3]),
                    "predicted_iou": float(ann.get("predicted_iou", np.nan)),
                    "stability_score": float(ann.get("stability_score", np.nan)),
                }
            )


def main() -> int:
    args = parse_args()
    modes = args.scaling_mode or ["zscore_clip", "zscore_no_clip", "log_lupton", "anscombe"]
    width = int(args.width or args.size)
    height = int(args.height or args.size)
    if width > 512 or height > 512:
        raise ValueError("native_sam_baseline currently expects one <=512 crop")
    paths = [path.expanduser().resolve() for path in args.input]
    bands = args.band or [path.stem for path in paths]
    if len(bands) != len(paths):
        raise ValueError("--band must be repeated once per --input")

    model = load_native_sam(str(args.model_type), args.checkpoint.expanduser().resolve(), str(args.device))
    generator = SamAutomaticMaskGenerator(
        model,
        points_per_side=int(args.points_per_side),
        points_per_batch=int(args.points_per_batch),
        pred_iou_thresh=float(args.pred_iou_thresh),
        stability_score_thresh=float(args.stability_score_thresh),
        crop_n_layers=int(args.crop_n_layers),
        min_mask_region_area=int(args.min_mask_region_area),
        output_mode="binary_mask",
    )

    summaries = []
    for path, band in zip(paths, bands):
        image, header, hdu = read_fits_image(path, hdu=args.hdu)
        crop, valid = crop_or_pad(image, x0=int(args.x0), y0=int(args.y0), width=width, height=height)
        crop = np.where(valid, crop, np.nan).astype(np.float32)
        clean_crop = np.nan_to_num(crop, nan=0.0, posinf=0.0, neginf=0.0)[:height, :width]
        for mode in modes:
            scaled = make_training_rgb(crop, mode=str(mode), log_a=float(args.log_a), anscombe_scale=float(args.anscombe_scale))
            sam_input = sam_uint8_from_scaled(scaled)
            annotations = generator.generate(sam_input)
            label_map = labelmap_from_amg(annotations, height=height, width=width)
            mode_name = str(mode).replace("_", "-")
            out_dir = args.out_dir.expanduser().resolve() / f"{Path(path).stem}_x{args.x0}_y{args.y0}" / str(band) / mode_name
            out_dir.mkdir(parents=True, exist_ok=True)

            fits.PrimaryHDU(
                label_map.astype(np.int32),
                header=shifted_crop_header(header, x0=int(args.x0), y0=int(args.y0)),
            ).writeto(out_dir / f"{band}_{mode_name}_mask_labelmap.fits", overwrite=True)
            write_mask_reg(out_dir / f"{band}_{mode_name}_mask_contours.reg", label_map)
            _write_ann_csv(out_dir / f"{band}_{mode_name}_masks.csv", annotations)
            save_png(out_dir / f"{band}_{mode_name}_input_rgb.png", scaled_rgb_for_display(scaled), title=f"{band} {mode_name} input")
            for channel in range(min(3, scaled.shape[0])):
                save_heatmap(
                    out_dir / f"{band}_{mode_name}_input_channel{channel}.png",
                    scaled[channel, :height, :width],
                    title=f"{band} {mode_name} input channel {channel}",
                )
            save_png(out_dir / f"{band}_{mode_name}_mask_overlay.png", mask_overlay(clean_crop, label_map), title=f"{band} {mode_name} native SAM")
            summaries.append(
                {
                    "input": str(path),
                    "band": str(band),
                    "hdu": int(hdu),
                    "scaling_mode": str(mode),
                    "masks": int(np.count_nonzero(np.unique(label_map) > 0)),
                    "out_dir": str(out_dir),
                }
            )
            print(f"[done] {band} {mode_name}: masks={summaries[-1]['masks']} out={out_dir}", flush=True)

    args.out_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    (args.out_dir.expanduser().resolve() / "manifest.json").write_text(json.dumps({"summary": summaries}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
