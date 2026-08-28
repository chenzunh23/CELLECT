#!/usr/bin/env python3
"""Run a CELLECT sam_per_band checkpoint and visualize detector outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
    decode_fixed_utf8,
    detection_rows,
    draw_ellipses,
    infer_cellect,
    inverse_ellipse_overlay,
    input_channel_display_limits,
    load_cellect_model,
    mask_overlay,
    read_zarr_sample,
    resolve_zarr_sample,
    rows_to_reg,
    save_heatmap,
    save_pixel_png,
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
from eval.matching import ground_truth_rows_from_zarr, write_matching_diagnostics


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
    p.add_argument("--clip-threshold", type=float, default=3.0)
    p.add_argument("--log-high-percentile", type=float, default=99.5)
    p.add_argument("--lupton-stretch", type=float, default=0.5)
    p.add_argument("--lupton-q", type=float, default=20.0)
    p.add_argument("--anscombe-clip", action=argparse.BooleanOptionalAction, default=False)
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
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--shape-overlay-centers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--input-shape-overlay", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--inverse-shape-overlay", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--inverse-shape-overlay-centers", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--inverse-shape-overlay-color", default="#0066ff")
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


def _slug(text: object) -> str:
    value = str(text).strip().lower().replace("_", "-").replace("/", "-")
    while "--" in value:
        value = value.replace("--", "-")
    return value.strip("-")


def _zarr_scaling_slug(attrs: dict, *, root: Path | None, store: Path | None) -> str:
    mode = attrs.get("image_scaling_mode") or attrs.get("scaling_mode")
    if mode:
        slug = _slug(mode)
        if slug.endswith("-rgb"):
            slug = slug[:-4]
        return slug or "zarr"
    haystacks = [str(path) for path in (store, root) if path is not None]
    for text in haystacks:
        normalized = _slug(text)
        if "zscore-no-upper" in normalized:
            return "zscore-no-upper"
        if "log-lupton" in normalized or "lupton" in normalized:
            return "zscore-log-lupton"
        if "anscombe" in normalized:
            return "anscombe"
        if "zscore" in normalized:
            return "zscore"
    return "zarr"


def _checkpoint_epoch_label(path: Path) -> str:
    stem = Path(path).name.rsplit(".", 1)[0]
    value = stem.split("_")[-1]
    return str(int(value)) if value.isdigit() else value


def _pred_conf_overlay(image: np.ndarray, outputs: dict[str, torch.Tensor]) -> np.ndarray:
    score = _score_map(outputs)
    levels = np.clip(np.rint(score), 0, 4).astype(np.uint8)
    return confidence_overlay(image, levels)


def _input_overlay_channel(scaled: np.ndarray, scaling: str) -> tuple[int, np.ndarray] | None:
    label = str(scaling).replace("_", "-").lower()
    arr = np.asarray(scaled, dtype=np.float32)
    if "log-lupton" in label or "lupton" in label:
        idx = min(2, arr.shape[0] - 1)
        return idx, arr[idx]
    if "anscombe" in label or "zscore" in label:
        return 0, arr[0]
    return None


def _visual_options(args: argparse.Namespace) -> dict[str, object]:
    return {
        "shape_overlay_centers": bool(args.shape_overlay_centers),
        "input_shape_overlay": bool(args.input_shape_overlay),
        "inverse_shape_overlay": bool(args.inverse_shape_overlay),
        "inverse_shape_overlay_centers": bool(args.inverse_shape_overlay_centers),
        "inverse_shape_overlay_color": str(args.inverse_shape_overlay_color),
        "make_masks": bool(args.make_masks),
        "confidence_threshold": float(args.confidence_threshold),
        "confidence_score": str(args.confidence_score),
        "nms_radius": int(args.nms_radius),
        "center_refinement": str(args.center_refinement),
        "center_refinement_radius": int(args.center_refinement_radius),
        "scaling_mode": str(args.scaling_mode),
        "clip_threshold": float(args.clip_threshold),
        "log_a": float(args.log_a),
        "log_high_percentile": float(args.log_high_percentile),
        "lupton_stretch": float(args.lupton_stretch),
        "lupton_q": float(args.lupton_q),
        "anscombe_clip": bool(args.anscombe_clip),
        "anscombe_scale": float(args.anscombe_scale),
    }


def _band_outputs_current(band_dir: Path, band: str, options: dict[str, object]) -> bool:
    if not (band_dir / "matching" / f"{band}_matching_summary.json").exists():
        return False
    if not (band_dir / f"{band}_shape_overlay.png").exists():
        return False
    if bool(options.get("input_shape_overlay")) and not (band_dir / f"{band}_input_shape_overlay.png").exists():
        return False
    if bool(options.get("inverse_shape_overlay")) and not (band_dir / f"{band}_inverse_shape_overlay.png").exists():
        return False
    if bool(options.get("make_masks")) and not (band_dir / f"{band}_mask_overlay.png").exists():
        return False
    options_path = band_dir / f"{band}_visual_options.json"
    if not options_path.exists():
        return False
    try:
        previous = json.loads(options_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return previous == options


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
    zarr_clip_threshold = float(attrs.get("clip_threshold", attrs.get("image_clip_threshold", 3.0)))
    finite_image = image[np.isfinite(image)]
    if (
        finite_image.size
        and zarr_clip_threshold > 3.0
        and "no-upper" not in _zarr_scaling_slug(
            attrs,
            root=args.root.expanduser().resolve() if args.root else None,
            store=args.zarr_store.expanduser().resolve() if args.zarr_store else None,
        )
        and float(np.max(finite_image)) <= 3.0001
    ):
        print(
            "[warn] zarr attrs report clip_threshold="
            f"{zarr_clip_threshold:g}, but stored image max={float(np.max(finite_image)):.4g}; "
            "this store was likely generated before image clip-threshold was wired into scaling.",
            flush=True,
        )
    tensor = torch.from_numpy(image[None, None].astype(np.float32, copy=False))
    display = np.asarray(sample["display_image"], dtype=np.float32)
    band = (attrs.get("bands") or [args.zarr_band or "band0"])[band_idx]
    dataset = str(attrs.get("dataset_source", dataset_source or "zarr"))
    if reader.has_array("dataset_source"):
        dataset_sources = decode_fixed_utf8(reader.read_full_small("dataset_source"))
        if int(sample_idx) < len(dataset_sources) and dataset_sources[int(sample_idx)]:
            dataset = dataset_sources[int(sample_idx)]
    group_name = zarr_sample_group(reader, sample_idx)
    ckpt_root = str(args.checkpoint.expanduser().resolve().parent).split(os.sep)[-1]
    epoch_num = _checkpoint_epoch_label(args.checkpoint)
    if reader.has_array("tile_name"):
        tile_names = decode_fixed_utf8(reader.read_full_small("tile_name"))
        tile_name = tile_names[int(sample_idx)] if int(sample_idx) < len(tile_names) else f"sample{sample_idx:05d}"
    else:
        tile_name = f"sample{sample_idx:05d}"
    patch_label = str(attrs.get("patch", args.patch or "patch")).replace(",", "_")
    scaling_label = _zarr_scaling_slug(
        attrs,
        root=args.root.expanduser().resolve() if args.root else None,
        store=args.zarr_store.expanduser().resolve() if args.zarr_store else None,
    )
    stem = f"{patch_label}_{tile_name}_{scaling_label}_{ckpt_root}_epoch{epoch_num}"
    x0 = int(reader.read_full_small("tile_x0")[sample_idx]) if reader.has_array("tile_x0") else 0
    y0 = int(reader.read_full_small("tile_y0")[sample_idx]) if reader.has_array("tile_y0") else 0
    context = {
        "reader": reader,
        "sample_idx": int(sample_idx),
        "band_idx": int(band_idx),
        "dataset_source": dataset,
        "group": group_name,
        "tile_name": tile_name,
        "scaling": scaling_label,
        "clip_threshold": zarr_clip_threshold,
    }
    return tensor, [display], [image], [None], [str(band)], stem, [x0], [y0], context


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
        clip_threshold=float(args.clip_threshold),
        log_a=float(args.log_a),
        log_high_percentile=float(args.log_high_percentile),
        lupton_stretch=float(args.lupton_stretch),
        lupton_q=float(args.lupton_q),
        anscombe_clip=bool(args.anscombe_clip),
        anscombe_scale=float(args.anscombe_scale),
    )
    bands = args.band or [path.stem for path in paths]
    if len(bands) != len(paths):
        raise ValueError("--band must be repeated once per --input")
    ckpt_root = str(args.checkpoint.expanduser().resolve().parent).split(os.sep)[-1]
    epoch_num = _checkpoint_epoch_label(args.checkpoint)
    stem = f"x{args.x0}_y{args.y0}_{str(args.scaling_mode).replace('_', '-')}_{ckpt_root}_epoch{epoch_num}"
    return tensor, raw_images, scaled_images, headers, bands, stem, [int(args.x0)] * len(paths), [int(args.y0)] * len(paths), None


def _run_one(args: argparse.Namespace, dataset_source: str | None = None) -> Path:
    width = int(args.width or args.size)
    height = int(args.height or args.size)
    if args.input:
        tensor, raw_images, scaled_images, headers, bands, stem, image_x0, image_y0, zarr_context = _prepare_from_fits(args)
    else:
        tensor, raw_images, scaled_images, headers, bands, stem, image_x0, image_y0, zarr_context = _prepare_from_zarr(args, dataset_source)
        height, width = int(tensor.shape[-2]), int(tensor.shape[-1])

    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()).split(" ")[0]
    year, month, day = date.split("-")
    out_dir = args.out_dir.expanduser().resolve() / f"{year}-{month}" / date / stem
    if zarr_context is not None:
        variant = str(zarr_context.get("dataset_source", "coadd"))
        if variant not in {"", "coadd", "zarr"}:
            out_dir = out_dir / variant
            group_name = str(zarr_context.get("group", ""))
            if group_name:
                out_dir = out_dir / group_name
    visual_options = _visual_options(args)
    if bool(args.skip_existing) and all(
        _band_outputs_current(out_dir / str(band), str(band), visual_options)
        for band in bands
    ):
        print(f"[skip] existing outputs for {bands}: {out_dir}", flush=True)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, cfg = load_cellect_model(
        args.checkpoint.expanduser().resolve(),
        args.config.expanduser().resolve() if args.config else None,
        device,
        bands,
    )
    outputs = infer_cellect(model=model, image_tensor=tensor, device=device, amp=str(args.amp))

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
        save_pixel_png(band_dir / f"{band}_confidence_overlay.png", _pred_conf_overlay(image, band_outputs))
        save_png(band_dir / f"{band}_input_rgb.png", scaled_rgb_for_display(scaled), title=f"{band} model input")
        for channel in range(min(3, scaled.shape[0])):
            scaling_for_display = str(zarr_context.get("scaling", args.scaling_mode) if zarr_context is not None else args.scaling_mode)
            clip_threshold_for_display = float(
                zarr_context.get("clip_threshold", args.clip_threshold)
                if zarr_context is not None
                else args.clip_threshold
            )
            vmin, vmax = input_channel_display_limits(
                scaled[channel],
                scaling=scaling_for_display,
                channel_index=channel,
                clip_threshold=clip_threshold_for_display,
            )
            save_heatmap(
                band_dir / f"{band}_input_channel{channel}.png",
                scaled[channel],
                title=f"{band} input channel {channel}",
                vmin=vmin,
                vmax=vmax,
            )
        write_reg(band_dir / f"{band}_centers.reg", rows_to_reg(rows, shape=False, color="yellow"))
        write_reg(band_dir / f"{band}_shapes.reg", rows_to_reg(rows, shape=True, color="cyan"))
        write_sources_csv(band_dir / f"{band}_sources.csv", rows, header=header, x0=int(image_x0[band_idx]), y0=int(image_y0[band_idx]))
        save_pixel_png(
            band_dir / f"{band}_shape_overlay.png",
            draw_ellipses(image, rows, draw_centers=bool(args.shape_overlay_centers)),
        )
        if bool(args.inverse_shape_overlay):
            save_pixel_png(
                band_dir / f"{band}_inverse_shape_overlay.png",
                inverse_ellipse_overlay(
                    image,
                    rows,
                    color=str(args.inverse_shape_overlay_color),
                    line_width=1.8,
                    draw_centers=bool(args.inverse_shape_overlay_centers),
                ),
            )
        if bool(args.input_shape_overlay):
            scaling_name = str(zarr_context.get("scaling", args.scaling_mode) if zarr_context is not None else args.scaling_mode)
            clip_threshold_for_display = float(
                zarr_context.get("clip_threshold", args.clip_threshold)
                if zarr_context is not None
                else args.clip_threshold
            )
            input_selection = _input_overlay_channel(scaled, scaling_name)
            if input_selection is not None:
                input_channel_idx, input_channel = input_selection
                save_pixel_png(
                    band_dir / f"{band}_input_shape_overlay.png",
                    draw_ellipses(
                        input_channel,
                        rows,
                        draw_centers=bool(args.shape_overlay_centers),
                        input_scaled_background=True,
                        input_scaling=scaling_name,
                        input_channel_index=input_channel_idx,
                        input_clip_threshold=clip_threshold_for_display,
                    ),
                )
        matching_summary = None
        if zarr_context is not None:
            gt_rows = ground_truth_rows_from_zarr(
                zarr_context["reader"],
                int(zarr_context["sample_idx"]),
                int(zarr_context["band_idx"]),
            )
            matching_summary = write_matching_diagnostics(
                out_dir=band_dir / "matching",
                pred_rows=rows,
                gt_rows=gt_rows,
                image=image,
                band=str(band),
                match_radius=3.0,
            )

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
            save_pixel_png(band_dir / f"{band}_mask_overlay.png", mask_overlay(image, label_map))

        summary_row = {"band": str(band), "detections": len(rows), "masks": mask_count}
        if matching_summary is not None:
            summary_row["matching"] = matching_summary
        summaries.append(summary_row)
        (band_dir / f"{band}_visual_options.json").write_text(
            json.dumps(visual_options, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

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
