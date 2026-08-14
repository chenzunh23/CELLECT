#!/usr/bin/env python3
"""Evaluate a CELLECT SAM checkpoint on raw 256x256 HSC pack tiles.

The HSC pack stores raw zp27-scaled single-exposure frames in ragged Zarr
groups.  This script supports two lightweight inference modes:

- ``single256``: evaluate one 256x256 frame directly with dynamic image-size
  inference, so the SAM ViT positional embedding is interpolated to 16x16.
- ``mosaic2x2``: stitch four neighboring 256x256 frames into one 512x512 image
  and evaluate with the native 512x512 model geometry.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_utils import (  # noqa: E402
    build_prompt_masks,
    detection_rows,
    draw_ellipses,
    infer_cellect,
    input_channel_display_limits,
    load_cellect_model,
    mask_overlay,
    rows_to_reg,
    save_heatmap,
    save_pixel_png,
    save_png,
    scaled_rgb_for_display,
    select_band_outputs,
    write_mask_reg,
    write_reg,
)
from eval.visualize_cellect_outputs import _pred_conf_overlay, _score_map  # noqa: E402
from eval.eval_utils import make_training_rgb, write_sources_csv  # noqa: E402


DEFAULT_ROOT = Path("/data/zc/Subaru/data/hsctile/pack_full_9813_256/9813")
_METADATA_WARNED: set[tuple[str, str]] = set()


@dataclass(frozen=True)
class TileRow:
    tile_index: int
    tile_id: str
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class FrameRef:
    band: str
    patch: str
    tile_id: str
    tile_index: int
    frame_rank: int
    frame_index: int
    tile_length: int
    visit: int | None
    weight: float | None
    scale: float | None
    x0: int
    y0: int
    x1: int
    y1: int
    pack_path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", choices=("none", "bf16"), default="bf16")

    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--index", type=Path, default=None)
    p.add_argument("--sample-index", type=int, default=None, help="Global sample index in hsc_pack_index JSON.")
    p.add_argument("--sample-id", default=None, help="sample_id in hsc_pack_index JSON.")
    p.add_argument("--band", action="append", default=None, help="Band to evaluate. Repeat for multi-band input.")
    p.add_argument("--patch", default=None)
    p.add_argument("--tile-id", default=None, help="Tile id such as x070_y083.")
    p.add_argument("--tile-index", type=int, default=None)
    p.add_argument("--frame-rank", type=int, default=0, help="Frame rank inside each tile span.")
    p.add_argument("--visit", type=int, default=None, help="Prefer a specific visit/frame when present.")
    p.add_argument("--strict-visit", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--mosaic-frame-policy",
        choices=("same_visit", "same_rank"),
        default="same_visit",
        help="For mosaic2x2, how to choose neighboring tile frames when --visit is not given.",
    )

    p.add_argument("--mode", choices=("single256", "mosaic2x2"), default="single256")
    p.add_argument("--scaling-mode", choices=("zscore_clip", "zscore_no_clip", "zscore_no_upper", "log_lupton", "anscombe"), default="zscore_clip")
    p.add_argument("--log-a", type=float, default=300.0)
    p.add_argument("--clip-threshold", type=float, default=3.0)
    p.add_argument("--log-high-percentile", type=float, default=99.5)
    p.add_argument("--lupton-stretch", type=float, default=0.5)
    p.add_argument("--lupton-q", type=float, default=20.0)
    p.add_argument("--anscombe-clip", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--anscombe-scale", type=float, default=1000.0)
    p.add_argument("--dynamic-image-size", action=argparse.BooleanOptionalAction, default=None)

    p.add_argument("--confidence-threshold", type=float, default=2.0)
    p.add_argument("--confidence-score", default="ordinal_expectation")
    p.add_argument("--nms-radius", type=int, default=3)
    p.add_argument("--center-refinement", choices=("integer", "weighted_centroid", "softargmax"), default="softargmax")
    p.add_argument("--center-refinement-radius", type=int, default=1)
    p.add_argument("--make-masks", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--mask-threshold", type=float, default=0.0)
    p.add_argument("--mask-box-scale", type=float, default=2.0)
    p.add_argument("--mask-chunk-size", type=int, default=512)
    p.add_argument("--mask-multimask", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--shape-overlay-centers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw center crosses on the shape overlay PNG. Use --no-shape-overlay-centers for ellipse-only overlays.",
    )

    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out-dir", type=Path, default=Path("output/eval_visualizations/hsctile_pack"))
    return p.parse_args()


def _import_zarr():
    try:
        import zarr  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("zarr is required to read the HSC pack; run in the cellect environment") from exc
    return zarr


def _read_index(root: Path, index: Path | None) -> dict[str, Any]:
    path = index.expanduser() if index else root / "hsc_pack_index_9813_allbands_256.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sample_from_index(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any] | None:
    samples = list(payload.get("samples", []))
    if args.sample_index is not None:
        if int(args.sample_index) < 0 or int(args.sample_index) >= len(samples):
            raise IndexError(f"--sample-index {args.sample_index} outside [0, {len(samples)})")
        return dict(samples[int(args.sample_index)])
    if args.sample_id:
        for row in samples:
            if str(row.get("sample_id")) == str(args.sample_id):
                return dict(row)
        raise KeyError(f"sample_id not found: {args.sample_id}")
    return None


def _read_tiles_csv(path: Path) -> list[TileRow]:
    rows: list[TileRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                TileRow(
                    tile_index=int(row["tile_index"]),
                    tile_id=str(row["tile_id"]),
                    x0=int(row["x0"]),
                    y0=int(row["y0"]),
                    x1=int(row["x1"]),
                    y1=int(row["y1"]),
                )
            )
    return rows


def _pack_dir(root: Path, band: str, patch: str) -> Path:
    return root / str(band) / str(patch) / "256"


def _open_pack(root: Path, band: str, patch: str):
    zarr = _import_zarr()
    path = _pack_dir(root, band, patch) / "pack.zarr"
    if not path.is_dir():
        raise FileNotFoundError(f"pack not found: {path}")
    return zarr.open_group(str(path), mode="r")


def _tile_by_id(root: Path, band: str, patch: str) -> dict[str, TileRow]:
    return {row.tile_id: row for row in _read_tiles_csv(_pack_dir(root, band, patch) / "tiles.csv")}


def _tile_by_index(root: Path, band: str, patch: str) -> dict[int, TileRow]:
    return {row.tile_index: row for row in _read_tiles_csv(_pack_dir(root, band, patch) / "tiles.csv")}


def _parse_tile_id(tile_id: str) -> tuple[int, int]:
    match = re.fullmatch(r"x(\d+)_y(\d+)", str(tile_id))
    if not match:
        raise ValueError(f"cannot parse tile id {tile_id!r}; expected xNNN_yNNN")
    return int(match.group(1)), int(match.group(2))


def _format_tile_id(x: int, y: int) -> str:
    return f"x{x:03d}_y{y:03d}"


def _scalar(group: Any, name: str, index: int) -> Any:
    if name not in group:
        return None
    try:
        value = group[name][int(index)]
    except Exception as exc:
        key = (str(getattr(group, "store", "")), str(name))
        if key not in _METADATA_WARNED:
            _METADATA_WARNED.add(key)
            print(
                f"WARNING: failed to read hsctile metadata array {name!r}; "
                f"continuing with None values: {exc}",
                file=sys.stderr,
                flush=True,
            )
        return None
    return value.item() if hasattr(value, "item") else value


def _choose_frame_index(
    group: Any,
    tile: TileRow,
    *,
    frame_rank: int,
    visit: int | None,
    strict_visit: bool,
) -> tuple[int, int]:
    start = int(group["tile_offsets"][tile.tile_index])
    length = int(group["tile_lengths"][tile.tile_index])
    if length <= 0:
        raise RuntimeError(f"tile {tile.tile_id} has no frames")
    if visit is not None and "visits" in group:
        try:
            visits = np.asarray(group["visits"][start : start + length])
            where = np.where(visits == int(visit))[0]
            if where.size:
                rank = int(where[0])
                return start + rank, rank
            if strict_visit:
                raise RuntimeError(f"visit {visit} is not present in tile {tile.tile_id}")
        except Exception as exc:
            if strict_visit:
                raise RuntimeError(f"failed to read visits for strict visit selection in tile {tile.tile_id}") from exc
            print(
                f"WARNING: failed to read visits for tile {tile.tile_id}; falling back to --frame-rank: {exc}",
                file=sys.stderr,
                flush=True,
            )
    rank = int(frame_rank)
    if rank < 0:
        rank = length + rank
    rank = max(0, min(rank, length - 1))
    return start + rank, rank


def _frame_ref(
    root: Path,
    band: str,
    patch: str,
    tile: TileRow,
    *,
    frame_rank: int,
    visit: int | None,
    strict_visit: bool,
) -> FrameRef:
    group = _open_pack(root, band, patch)
    frame_index, actual_rank = _choose_frame_index(
        group,
        tile,
        frame_rank=frame_rank,
        visit=visit,
        strict_visit=strict_visit,
    )
    length = int(group["tile_lengths"][tile.tile_index])
    visit_value = _scalar(group, "visits", frame_index)
    weight_value = _scalar(group, "weights", frame_index)
    scale_value = _scalar(group, "scales", frame_index)
    return FrameRef(
        band=str(band),
        patch=str(patch),
        tile_id=str(tile.tile_id),
        tile_index=int(tile.tile_index),
        frame_rank=int(actual_rank),
        frame_index=int(frame_index),
        tile_length=int(length),
        visit=int(visit_value) if visit_value is not None else None,
        weight=float(weight_value) if weight_value is not None else None,
        scale=float(scale_value) if scale_value is not None else None,
        x0=int(tile.x0),
        y0=int(tile.y0),
        x1=int(tile.x1),
        y1=int(tile.y1),
        pack_path=_pack_dir(root, band, patch) / "pack.zarr",
    )


def _resolve_base(args: argparse.Namespace) -> tuple[list[str], str, str]:
    root = args.root.expanduser().resolve()
    index = _read_index(root, args.index)
    sample = _sample_from_index(args, index)
    bands = [str(v) for v in args.band] if args.band else []
    patch = args.patch
    tile_id = args.tile_id
    if sample is not None:
        patch = patch or str(sample["patch_id"])
        tile_id = tile_id or str(sample["tile_id"])
        if not bands:
            bands = [str(sample["band"])]
    if not bands:
        raise ValueError("provide --band or --sample-index/--sample-id")
    if not patch:
        raise ValueError("provide --patch or --sample-index/--sample-id")
    if not tile_id and args.tile_index is None:
        raise ValueError("provide --tile-id, --tile-index, or --sample-index/--sample-id")
    if tile_id is None:
        by_index = _tile_by_index(root, bands[0], patch)
        if int(args.tile_index) not in by_index:
            raise KeyError(f"tile index not found for {bands[0]} {patch}: {args.tile_index}")
        tile_id = by_index[int(args.tile_index)].tile_id
    return bands, str(patch), str(tile_id)


def _read_frame(root: Path, ref: FrameRef) -> np.ndarray:
    group = _open_pack(root, ref.band, ref.patch)
    return np.asarray(group["images"][ref.frame_index], dtype=np.float32)


def _select_refs_for_band(args: argparse.Namespace, band: str, patch: str, tile_id: str) -> list[FrameRef]:
    root = args.root.expanduser().resolve()
    by_id = _tile_by_id(root, band, patch)
    if tile_id not in by_id:
        raise KeyError(f"tile {tile_id!r} not found for {band} {patch}")
    base_tile = by_id[tile_id]
    base_ref = _frame_ref(
        root,
        band,
        patch,
        base_tile,
        frame_rank=int(args.frame_rank),
        visit=args.visit,
        strict_visit=bool(args.strict_visit),
    )
    if args.mode == "single256":
        return [base_ref]

    base_x, base_y = _parse_tile_id(tile_id)
    preferred_visit = args.visit
    if preferred_visit is None and str(args.mosaic_frame_policy) == "same_visit":
        preferred_visit = base_ref.visit
    refs = []
    for dy in (0, 1):
        for dx in (0, 1):
            neighbor_id = _format_tile_id(base_x + dx, base_y + dy)
            if neighbor_id not in by_id:
                raise KeyError(f"neighbor tile {neighbor_id!r} not found for {band} {patch}")
            refs.append(
                _frame_ref(
                    root,
                    band,
                    patch,
                    by_id[neighbor_id],
                    frame_rank=base_ref.frame_rank,
                    visit=preferred_visit,
                    strict_visit=bool(args.strict_visit),
                )
            )
    return refs


def _make_band_image(root: Path, refs: Sequence[FrameRef], mode: str) -> tuple[np.ndarray, FrameRef]:
    if mode == "single256":
        return _read_frame(root, refs[0]), refs[0]
    out = np.full((512, 512), np.nan, dtype=np.float32)
    for idx, ref in enumerate(refs):
        dx = idx % 2
        dy = idx // 2
        out[dy * 256 : (dy + 1) * 256, dx * 256 : (dx + 1) * 256] = _read_frame(root, ref)
    return out, refs[0]


def _scale_stack(
    images: Sequence[np.ndarray],
    *,
    scaling_mode: str,
    clip_threshold: float,
    log_a: float,
    log_high_percentile: float,
    lupton_stretch: float,
    lupton_q: float,
    anscombe_clip: bool,
    anscombe_scale: float,
) -> list[np.ndarray]:
    return [
        make_training_rgb(
            image,
            mode=scaling_mode,
            clip_threshold=float(clip_threshold),
            log_a=float(log_a),
            log_high_percentile=float(log_high_percentile),
            lupton_stretch=float(lupton_stretch),
            lupton_q=float(lupton_q),
            anscombe_clip=bool(anscombe_clip),
            anscombe_scale=float(anscombe_scale),
        )
        for image in images
    ]


def _slug(text: object) -> str:
    return str(text).strip().replace("_", "-").replace("/", "-").replace(",", "_")


def _checkpoint_epoch_label(path: Path) -> str:
    stem = Path(path).name.rsplit(".", 1)[0]
    value = stem.split("_")[-1]
    return str(int(value)) if value.isdigit() else value


def _write_manifest(path: Path, refs_by_band: dict[str, list[FrameRef]], args: argparse.Namespace) -> None:
    payload = {
        "mode": args.mode,
        "root": str(args.root.expanduser().resolve()),
        "bands": list(refs_by_band),
        "scaling_mode": args.scaling_mode,
        "clip_threshold": float(args.clip_threshold),
        "log_a": float(args.log_a),
        "log_high_percentile": float(args.log_high_percentile),
        "lupton_stretch": float(args.lupton_stretch),
        "lupton_q": float(args.lupton_q),
        "anscombe_clip": bool(args.anscombe_clip),
        "anscombe_scale": float(args.anscombe_scale),
        "frames": {
            band: [
                {
                    "patch": ref.patch,
                    "tile_id": ref.tile_id,
                    "tile_index": ref.tile_index,
                    "frame_rank": ref.frame_rank,
                    "frame_index": ref.frame_index,
                    "tile_length": ref.tile_length,
                    "visit": ref.visit,
                    "weight": ref.weight,
                    "scale": ref.scale,
                    "x0": ref.x0,
                    "y0": ref.y0,
                    "x1": ref.x1,
                    "y1": ref.y1,
                    "pack_path": str(ref.pack_path),
                }
                for ref in refs
            ]
            for band, refs in refs_by_band.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_sources_csv_with_refs(path: Path, rows: Sequence[dict[str, float]], *, ref: FrameRef) -> None:
    write_sources_csv(path, rows, header=None, x0=ref.x0, y0=ref.y0)


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    bands, patch, tile_id = _resolve_base(args)
    dynamic = bool(args.dynamic_image_size) if args.dynamic_image_size is not None else args.mode == "single256"
    width = 256 if args.mode == "single256" else 512
    height = width

    refs_by_band: dict[str, list[FrameRef]] = {}
    raw_images: list[np.ndarray] = []
    first_refs: list[FrameRef] = []
    for band in bands:
        refs = _select_refs_for_band(args, band, patch, tile_id)
        image, first_ref = _make_band_image(root, refs, str(args.mode))
        refs_by_band[str(band)] = refs
        raw_images.append(np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False))
        first_refs.append(first_ref)

    scaled_images = _scale_stack(
        raw_images,
        scaling_mode=str(args.scaling_mode),
        clip_threshold=float(args.clip_threshold),
        log_a=float(args.log_a),
        log_high_percentile=float(args.log_high_percentile),
        lupton_stretch=float(args.lupton_stretch),
        lupton_q=float(args.lupton_q),
        anscombe_clip=bool(args.anscombe_clip),
        anscombe_scale=float(args.anscombe_scale),
    )
    tensor = torch.from_numpy(np.stack(scaled_images, axis=0).astype(np.float32, copy=False))[None]

    ckpt_root = str(args.checkpoint.expanduser().resolve().parent).split(os.sep)[-1]
    epoch_num = _checkpoint_epoch_label(args.checkpoint)
    visit_label = first_refs[0].visit if first_refs and first_refs[0].visit is not None else f"rank{first_refs[0].frame_rank}"
    stem = (
        f"{_slug(patch)}_{_slug(tile_id)}_{args.mode}_{_slug(args.scaling_mode)}_"
        f"{ckpt_root}_epoch{epoch_num}_v{visit_label}"
    )
    date = time.strftime("%Y-%m-%d", time.localtime())
    year, month, _day = date.split("-")
    out_dir = args.out_dir.expanduser().resolve() / f"{year}-{month}" / date / stem
    if bool(args.skip_existing) and all((out_dir / band / f"{band}_sources.csv").exists() for band in bands):
        print(f"[skip] existing outputs: {out_dir}", flush=True)
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, cfg = load_cellect_model(
        args.checkpoint.expanduser().resolve(),
        args.config.expanduser().resolve() if args.config else None,
        device,
        bands,
        dynamic_image_size=dynamic,
    )
    outputs = infer_cellect(model=model, image_tensor=tensor, device=device, amp=str(args.amp))
    _write_manifest(out_dir / "hsctile_eval_manifest.json", refs_by_band, args)

    summaries = []
    for band_idx, band in enumerate(bands):
        band_outputs = select_band_outputs(outputs, band_idx)
        image = raw_images[band_idx]
        scaled = scaled_images[band_idx]
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
        save_pixel_png(
            band_dir / f"{band}_shape_overlay.png",
            draw_ellipses(image, rows, draw_centers=bool(args.shape_overlay_centers)),
        )
        for channel in range(min(3, scaled.shape[0])):
            vmin, vmax = input_channel_display_limits(
                scaled[channel],
                scaling=str(args.scaling_mode),
                channel_index=channel,
                clip_threshold=float(args.clip_threshold),
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
        _write_sources_csv_with_refs(band_dir / f"{band}_sources.csv", rows, ref=first_refs[band_idx])

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
            write_mask_reg(band_dir / f"{band}_mask_contours.reg", label_map)
            save_pixel_png(band_dir / f"{band}_mask_overlay.png", mask_overlay(image, label_map))
        summaries.append({"band": str(band), "detections": len(rows), "masks": mask_count})

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "summary": summaries,
                "config": cfg,
                "dynamic_image_size": dynamic,
                "input_shape": list(tensor.shape),
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
