#!/usr/bin/env python3
"""Run a single-band SAM CELLECT checkpoint on old patch-level zarr samples.

The old zarr layout stores all broad bands in one patch-level sample.  The
model is still ``sam_per_band``: this script extracts one requested band at a
time, runs it as ``[B, 1, H, W]`` (or rebuilt single-band RGB), and matches
against the corresponding band labels from the same old zarr sample.
"""

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

from astro_train_zarr_data import PatchZarrReader  # noqa: E402
from eval.eval_utils import (  # noqa: E402
    build_prompt_masks,
    confidence_overlay,
    crop_or_pad,
    crop_origin_for_image,
    decode_fixed_utf8,
    detection_rows,
    draw_ellipses,
    infer_cellect,
    input_channel_display_limits,
    load_cellect_model,
    make_training_rgb,
    mask_overlay,
    read_fits_image,
    rows_to_reg,
    save_heatmap,
    save_pixel_png,
    save_png,
    scaled_rgb_for_display,
    select_band_outputs,
    shifted_crop_header,
    tile_name_matches,
    write_mask_reg,
    write_reg,
    write_sources_csv,
)
from eval.matching import ground_truth_rows_from_zarr, write_matching_diagnostics  # noqa: E402
from eval.visualize_cellect_outputs import _checkpoint_epoch_label, _score_map  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", choices=("none", "bf16"), default="bf16")
    p.add_argument("--zarr-root", type=Path, default=Path("/data/czh23/direct_zarr_v2_0711"))
    p.add_argument("--tract", default="9813")
    p.add_argument("--patch", required=True)
    p.add_argument("--dataset-source", choices=("coadd", "noisy", "denoised"), default="coadd")
    p.add_argument("--group", default=None, help="For noisy/denoised stores, e.g. group_01. Sample index is group-relative.")
    p.add_argument("--sample-index", type=int, default=None, help="Coadd/group-relative sample index.")
    p.add_argument("--tile-name", default=None, help="Tile name, e.g. grid_r04_c06_x18108_y21372.")
    p.add_argument("--band", action="append", required=True, help="Output band. Repeat for multiple bands.")
    p.add_argument("--skip-missing-band", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--input-scaling-mode",
        choices=("zarr", "zscore_clip", "zscore_no_upper"),
        default="zarr",
        help="zarr uses the stored old zarr image. zscore_* rebuilds the multi-band tensor from FITS.",
    )
    p.add_argument("--fits-root", type=Path, default=None, help="Root like /data/shared/Subaru; required for zscore_no_upper.")
    p.add_argument("--fits-hdu", type=int, default=None)
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
    p.add_argument("--out-dir", type=Path, default=Path("output/eval_visualizations/cellect_outputs"))
    return p.parse_args()


def _slug(text: object) -> str:
    value = str(text).strip().lower().replace("_", "-").replace("/", "-")
    while "--" in value:
        value = value.replace("--", "-")
    return value.strip("-")


def _normalize_group(group: str | None) -> str | None:
    if group is None or not str(group).strip():
        return None
    text = str(group).strip()
    if text.startswith("group_"):
        return text
    if text.isdigit():
        return f"group_{int(text):02d}"
    return text


def _store_path(args: argparse.Namespace) -> Path:
    return args.zarr_root.expanduser().resolve() / str(args.tract) / str(args.dataset_source) / f"{args.patch}.zarr"


def _select_sample(reader: PatchZarrReader, *, sample_index: int | None, tile_name: str | None, group: str | None) -> int:
    n = int(reader.meta("images").shape[0])
    groups = decode_fixed_utf8(reader.read_full_small("group")) if reader.has_array("group") else [""] * n
    tiles = decode_fixed_utf8(reader.read_full_small("tile_name")) if reader.has_array("tile_name") else [f"sample_{idx:06d}" for idx in range(n)]
    group = _normalize_group(group)
    candidates = []
    for idx in range(n):
        if group is not None and groups[idx] != group:
            continue
        if tile_name is not None and not tile_name_matches(tiles[idx], tile_name):
            continue
        candidates.append(idx)
    if tile_name is not None:
        if not candidates:
            raise RuntimeError(f"no sample matches tile={tile_name!r} group={group!r}")
        if sample_index is not None and group is not None:
            # Treat sample-index as an optional selector within the matching
            # tile candidates only when a duplicated tile is unexpectedly found.
            rel = int(sample_index)
            if rel < len(candidates):
                return int(candidates[rel])
        return int(candidates[0])
    if sample_index is None:
        raise ValueError("provide --sample-index or --tile-name")
    if group is not None:
        group_candidates = [idx for idx in range(n) if groups[idx] == group]
        if int(sample_index) >= len(group_candidates):
            raise IndexError(f"group-relative sample-index {sample_index} exceeds {len(group_candidates)} samples for {group}")
        return int(group_candidates[int(sample_index)])
    if int(sample_index) >= n:
        raise IndexError(f"sample-index {sample_index} exceeds {n} samples")
    return int(sample_index)


def _sample_group(reader: PatchZarrReader, sample_idx: int) -> str:
    if not reader.has_array("group"):
        return ""
    groups = decode_fixed_utf8(reader.read_full_small("group"))
    return groups[int(sample_idx)] if int(sample_idx) < len(groups) else ""


def _sample_tile(reader: PatchZarrReader, sample_idx: int) -> str:
    if not reader.has_array("tile_name"):
        return f"sample{sample_idx:05d}"
    tiles = decode_fixed_utf8(reader.read_full_small("tile_name"))
    tile = tiles[int(sample_idx)] if int(sample_idx) < len(tiles) else f"sample{sample_idx:05d}"
    group = _sample_group(reader, sample_idx)
    prefix = f"{group}_"
    return tile[len(prefix) :] if group and tile.startswith(prefix) else tile


def _scaling_label(args: argparse.Namespace) -> str:
    if args.input_scaling_mode == "zscore_no_upper":
        return "zscore-no-upper"
    if args.input_scaling_mode == "zscore_clip":
        return "zscore-clip"
    return "astro-zscore"


def _fits_path(fits_root: Path, tract: str, band: str, patch: str) -> Path:
    return fits_root.expanduser().resolve() / str(tract) / str(band) / str(patch) / f"calexp-{band}-{tract}-{patch}.fits"


def _read_fits_band(
    *,
    fits_root: Path,
    tract: str,
    patch: str,
    band: str,
    x0: int,
    y0: int,
    width: int,
    height: int,
    mode: str,
    hdu: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, fits.Header]:
    path = _fits_path(fits_root, tract, band, patch)
    image, header, _ = read_fits_image(path, hdu=hdu)
    local_x0, local_y0 = crop_origin_for_image(image, header, x0=x0, y0=y0, width=width, height=height)
    crop, valid = crop_or_pad(image, x0=local_x0, y0=local_y0, width=width, height=height)
    crop = np.where(valid, crop, np.nan).astype(np.float32)
    scaled_rgb = make_training_rgb(crop, mode=str(mode), log_a=300.0, anscombe_scale=1000.0)
    scaled = np.asarray(scaled_rgb, dtype=np.float32)[:, :height, :width]
    tensor_image = scaled[0]
    raw = np.nan_to_num(crop, nan=0.0, posinf=0.0, neginf=0.0)[:height, :width]
    return tensor_image.astype(np.float32, copy=False), raw, scaled, header


def _zarr_band(
    reader: PatchZarrReader,
    sample_idx: int,
    band_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, fits.Header | None]:
    images = reader.read_first_axis("images", sample_idx).astype(np.float32, copy=False)
    if images.ndim == 4 and images.shape[1] == 3:
        scaled = images[band_idx]
        tensor_image = scaled[0]
        raw = scaled[0]
    elif images.ndim == 3:
        tensor_image = images[band_idx]
        raw = images[band_idx]
        scaled = np.stack([images[band_idx], images[band_idx], images[band_idx]], axis=0)
    else:
        raise ValueError(f"unexpected images shape in old zarr: {images.shape}")
    return (
        tensor_image.astype(np.float32, copy=False),
        raw.astype(np.float32, copy=False),
        scaled.astype(np.float32, copy=False),
        None,
    )


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
    if "anscombe" in label or "zscore" in label or "astro-zscore" in label:
        return 0, arr[0]
    return None


def _visual_options(args: argparse.Namespace) -> dict[str, object]:
    return {
        "shape_overlay_centers": bool(args.shape_overlay_centers),
        "input_shape_overlay": bool(args.input_shape_overlay),
        "make_masks": bool(args.make_masks),
        "confidence_threshold": float(args.confidence_threshold),
        "confidence_score": str(args.confidence_score),
        "nms_radius": int(args.nms_radius),
        "center_refinement": str(args.center_refinement),
        "center_refinement_radius": int(args.center_refinement_radius),
        "input_scaling_mode": str(args.input_scaling_mode),
    }


def _band_outputs_current(band_dir: Path, band: str, options: dict[str, object]) -> bool:
    if not (band_dir / "matching" / f"{band}_matching_summary.json").exists():
        return False
    if not (band_dir / f"{band}_shape_overlay.png").exists():
        return False
    if bool(options.get("input_shape_overlay")) and not (band_dir / f"{band}_input_shape_overlay.png").exists():
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


def _output_root(args: argparse.Namespace, reader: PatchZarrReader, sample_idx: int) -> Path:
    date = time.strftime("%Y-%m-%d", time.localtime())
    year, month, _day = date.split("-")
    ckpt_root = str(args.checkpoint.expanduser().resolve().parent).split(os.sep)[-1]
    epoch = _checkpoint_epoch_label(args.checkpoint)
    patch_label = str(reader.attrs.get("patch", args.patch)).replace(",", "_")
    stem = f"{patch_label}_{_sample_tile(reader, sample_idx)}_{_scaling_label(args)}_{ckpt_root}_epoch{epoch}"
    out_dir = args.out_dir.expanduser().resolve() / f"{year}-{month}" / date / stem
    if str(args.dataset_source) != "coadd":
        out_dir = out_dir / str(args.dataset_source)
        group = _sample_group(reader, sample_idx)
        if group:
            out_dir = out_dir / group
    return out_dir


def main() -> int:
    args = parse_args()
    store = _store_path(args)
    if not store.exists():
        raise FileNotFoundError(store)
    reader = PatchZarrReader(store)
    sample_idx = _select_sample(reader, sample_index=args.sample_index, tile_name=args.tile_name, group=args.group)
    all_bands = [str(band) for band in reader.attrs.get("bands", [])]
    if not all_bands:
        raise RuntimeError(f"store has no band metadata: {store}")
    requested = [str(band) for band in args.band]
    missing = [band for band in requested if band not in all_bands]
    if missing and not bool(args.skip_missing_band):
        raise ValueError(f"requested band(s) not in store {store}: {missing}; available={all_bands}")
    requested = [band for band in requested if band in all_bands]
    if not requested:
        print(f"[skip] no requested bands present in {store}: {args.band}", flush=True)
        return 0

    out_root = _output_root(args, reader, sample_idx)
    visual_options = _visual_options(args)
    pending = [
        band
        for band in requested
        if not (bool(args.skip_existing) and _band_outputs_current(out_root / band, band, visual_options))
    ]
    if not pending:
        print(f"[skip] existing outputs for {requested}: {out_root}", flush=True)
        return 0

    width = int(reader.attrs.get("tile_size", 512))
    height = width
    tile_x0 = int(reader.read_full_small("tile_x0")[sample_idx]) if reader.has_array("tile_x0") else 0
    tile_y0 = int(reader.read_full_small("tile_y0")[sample_idx]) if reader.has_array("tile_y0") else 0
    device = torch.device(args.device)
    model, cfg = load_cellect_model(
        args.checkpoint.expanduser().resolve(),
        args.config.expanduser().resolve() if args.config else None,
        device,
        ["single_band"],
    )
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for band in pending:
        band_idx = all_bands.index(band)
        if args.input_scaling_mode == "zarr":
            tensor_image, image, scaled, header = _zarr_band(reader, sample_idx, band_idx)
        else:
            if args.fits_root is None:
                raise ValueError("--fits-root is required when --input-scaling-mode is not 'zarr'")
            tensor_image, image, scaled, header = _read_fits_band(
                fits_root=args.fits_root,
                tract=str(args.tract),
                patch=str(args.patch),
                band=band,
                x0=tile_x0,
                y0=tile_y0,
                width=width,
                height=height,
                mode=str(args.input_scaling_mode),
                hdu=args.fits_hdu,
            )
        tensor = torch.from_numpy(tensor_image[None, None].astype(np.float32, copy=False))
        band_outputs = select_band_outputs(
            infer_cellect(model=model, image_tensor=tensor, device=device, amp=str(args.amp)),
            0,
        )
        band_dir = out_root / band
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
            vmin, vmax = input_channel_display_limits(scaled[channel], scaling=_scaling_label(args), channel_index=channel)
            save_heatmap(
                band_dir / f"{band}_input_channel{channel}.png",
                scaled[channel],
                title=f"{band} input channel {channel}",
                vmin=vmin,
                vmax=vmax,
            )
        write_reg(band_dir / f"{band}_centers.reg", rows_to_reg(rows, shape=False, color="yellow"))
        write_reg(band_dir / f"{band}_shapes.reg", rows_to_reg(rows, shape=True, color="cyan"))
        write_sources_csv(band_dir / f"{band}_sources.csv", rows, header=header, x0=tile_x0, y0=tile_y0)
        save_pixel_png(
            band_dir / f"{band}_shape_overlay.png",
            draw_ellipses(image, rows, draw_centers=bool(args.shape_overlay_centers)),
        )
        if bool(args.input_shape_overlay):
            scaling_name = _scaling_label(args)
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
                    ),
                )
        gt_rows = ground_truth_rows_from_zarr(reader, sample_idx, band_idx)
        matching_summary = write_matching_diagnostics(
            out_dir=band_dir / "matching",
            pred_rows=rows,
            gt_rows=gt_rows,
            image=image,
            band=band,
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
            mask_header = shifted_crop_header(header, x0=tile_x0, y0=tile_y0) if header is not None else fits.Header()
            fits.PrimaryHDU(label_map.astype(np.int32), header=mask_header).writeto(
                band_dir / f"{band}_mask_labelmap.fits",
                overwrite=True,
            )
            write_mask_reg(band_dir / f"{band}_mask_contours.reg", label_map)
            save_pixel_png(band_dir / f"{band}_mask_overlay.png", mask_overlay(image, label_map))
        summaries.append({"band": band, "detections": len(rows), "masks": mask_count, "matching": matching_summary})
        (band_dir / f"{band}_visual_options.json").write_text(
            json.dumps(visual_options, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[done] {args.dataset_source} {args.patch} sample={sample_idx} {band}: out={band_dir}", flush=True)

    (out_root / "summary.json").write_text(
        json.dumps(
            {
                "summary": summaries,
                "config": cfg,
                "store": str(store),
                "sample_idx": int(sample_idx),
                "tile_name": _sample_tile(reader, sample_idx),
                "group": _sample_group(reader, sample_idx),
                "all_bands": all_bands,
                "input_scaling_mode": str(args.input_scaling_mode),
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
