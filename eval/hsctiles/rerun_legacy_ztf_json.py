#!/usr/bin/env python3
"""Rerun CELLECT detection on previously exported ZTF JSON cutouts.

This intentionally bypasses the current ZTF dataset loader.  The input JSON
files already define the old export semantics (raw FITS path, x0/y0, tile id,
visit), so rerunning from them gives a strict visual comparison against an
older ``ztf_catalog_match_diagnostics`` tree.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")

import numpy as np
import torch
from astropy.io import fits
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_utils import (  # noqa: E402
    detection_rows,
    draw_ellipses,
    infer_cellect,
    input_channel_display_limits,
    load_cellect_model,
    make_training_rgb,
    rows_to_reg,
    save_heatmap,
    save_pixel_png,
    scaled_rgb_for_display,
    select_band_outputs,
    write_reg,
    write_sources_csv,
    zscale_gray,
)
from eval.visualize_cellect_outputs import _pred_conf_overlay, _score_map  # noqa: E402


DEFAULT_LEGACY_ROOT = Path(
    "/home/czh23/analysis/2026-08/2026-08-19/"
    "ztf_catalog_match_diagnostics/interactive_selected_20260818_215030"
)
DEFAULT_OUT_ROOT = Path("output/ztf_catalog_match_diagnostics_log_lupton_0810_epoch20/interactive_selected_20260818_215030")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=Path("/data/czh23/ckpts/sam_log_lupton_0810/epoch_0020.pt"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("none", "bf16"), default="bf16")
    parser.add_argument("--json", action="append", default=None, help="Specific legacy JSON file. Repeatable.")
    parser.add_argument("--band", action="append", default=None, help="Only rerun these bands.")
    parser.add_argument("--patch", action="append", default=None, help="Only rerun these ZTF q patches.")
    parser.add_argument("--tile-id", action="append", default=None, help="Only rerun these tile ids.")
    parser.add_argument("--visit", action="append", default=None, help="Only rerun these visits/obsids.")
    parser.add_argument("--scaling-mode", choices=("zscore_clip", "zscore_no_clip", "zscore_no_upper", "log_lupton", "anscombe"), default="log_lupton")
    parser.add_argument("--clip-threshold", type=float, default=3.0)
    parser.add_argument("--log-a", type=float, default=300.0)
    parser.add_argument("--log-high-percentile", type=float, default=99.5)
    parser.add_argument("--lupton-stretch", type=float, default=0.5)
    parser.add_argument("--lupton-q", type=float, default=20.0)
    parser.add_argument("--anscombe-clip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--anscombe-scale", type=float, default=1000.0)
    parser.add_argument("--confidence-threshold", type=float, default=2.0)
    parser.add_argument("--confidence-score", default="ordinal_expectation")
    parser.add_argument("--nms-radius", type=int, default=3)
    parser.add_argument("--center-refinement", choices=("integer", "weighted_centroid", "softargmax"), default="softargmax")
    parser.add_argument("--center-refinement-radius", type=int, default=1)
    parser.add_argument("--shape-overlay-centers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _legacy_json_paths(args: argparse.Namespace) -> list[Path]:
    if args.json:
        paths = [Path(path).expanduser().resolve() for path in args.json]
    else:
        paths = sorted(args.legacy_root.expanduser().resolve().glob("*/*/*/*/*.json"))
    out = []
    bands = set(args.band or [])
    patches = set(args.patch or [])
    tile_ids = set(args.tile_id or [])
    visits = set(str(v) for v in (args.visit or []))
    for path in paths:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if bands and str(row.get("band")) not in bands:
            continue
        if patches and str(row.get("patch")) not in patches:
            continue
        if tile_ids and str(row.get("tile_id")) not in tile_ids:
            continue
        if visits and str(row.get("visit")) not in visits:
            continue
        out.append(path)
    return out


def _read_crop(path: Path, x0: int, y0: int, width: int, height: int) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
        if hdul[0].data is None:
            raise ValueError(f"no image in primary HDU: {path}")
        data = np.asarray(hdul[0].data, dtype=np.float32)
        header = hdul[0].header.copy()
    crop = np.full((height, width), np.nan, dtype=np.float32)
    sx0 = max(0, int(x0))
    sy0 = max(0, int(y0))
    sx1 = min(data.shape[1], int(x0) + width)
    sy1 = min(data.shape[0], int(y0) + height)
    if sx1 > sx0 and sy1 > sy0:
        dx0 = sx0 - int(x0)
        dy0 = sy0 - int(y0)
        crop[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = data[sy0:sy1, sx0:sx1]
    return crop, header


def _input_channel_for_overlay(scaled: np.ndarray, scaling: str) -> tuple[int, np.ndarray]:
    label = str(scaling).replace("_", "-").lower()
    if "log-lupton" in label or "lupton" in label:
        idx = min(2, scaled.shape[0] - 1)
    else:
        idx = 0
    return idx, np.asarray(scaled[idx], dtype=np.float32)


def _input_shape_overlay(
    scaled: np.ndarray,
    rows: Sequence[dict[str, float]],
    *,
    scaling: str,
    clip_threshold: float,
    draw_centers: bool,
) -> np.ndarray:
    idx, channel = _input_channel_for_overlay(scaled, scaling)
    return draw_ellipses(
        channel,
        rows,
        color="cyan",
        point_color="blue",
        draw_centers=bool(draw_centers),
        input_scaled_background=True,
        input_scaling=scaling,
        input_channel_index=idx,
        input_clip_threshold=float(clip_threshold),
    )


def _display_gray_uint8(image: np.ndarray) -> np.ndarray:
    gray = zscale_gray(image)
    return np.clip(np.rint(np.flipud(gray) * 255.0), 0, 255).astype(np.uint8)


def _save_titled_rgb(path: Path, image: np.ndarray, title: str, *, min_image_size: int = 512) -> None:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        base = Image.fromarray(arr, mode="L").convert("RGB")
    else:
        base = Image.fromarray(arr, mode="RGB")
    scale = max(1, int(np.ceil(float(min_image_size) / max(base.width, base.height))))
    if scale > 1:
        base = base.resize((base.width * scale, base.height * scale), resample=Image.Resampling.NEAREST)
    title_h = 28
    out = Image.new("RGB", (base.width, base.height + title_h), "white")
    out.paste(base, (0, title_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    draw.text((4, 7), title, fill=(0, 0, 0), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def _write_detection_csv(path: Path, rows: Sequence[dict[str, float]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["x", "y", "score", "major", "minor", "theta"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _relative_output_dir(out_root: Path, row: dict[str, Any]) -> Path:
    return out_root / str(row["tract"]) / str(row["patch"]) / str(row["band"]) / str(row["tile_id"])


def _manifest_row(old: dict[str, Any], args: argparse.Namespace, out_dir: Path, rows: Sequence[dict[str, float]]) -> dict[str, Any]:
    stem = str(old.get("candidate_id") or Path(str(old.get("pack_path", "candidate"))).stem)
    payload = dict(old)
    payload.update(
        {
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "scaling_mode": str(args.scaling_mode),
            "clip_threshold": float(args.clip_threshold),
            "log_a": float(args.log_a),
            "log_high_percentile": float(args.log_high_percentile),
            "lupton_stretch": float(args.lupton_stretch),
            "lupton_q": float(args.lupton_q),
            "confidence_threshold": float(args.confidence_threshold),
            "nms_radius": int(args.nms_radius),
            "shape_overlay_centers": bool(args.shape_overlay_centers),
            "npz_path": str(out_dir / f"{stem}.npz"),
            "png_path": str(out_dir / f"{stem}.png"),
            "detect_png_path": str(out_dir / f"{stem}_detect_overlay.png"),
            "input_shape_png_path": str(out_dir / f"{stem}_input_shape_overlay.png"),
            "detect_csv_path": str(out_dir / f"{stem}_detections.csv"),
            "centers_reg_path": str(out_dir / f"{stem}_centers.reg"),
            "shapes_reg_path": str(out_dir / f"{stem}_shapes.reg"),
            "export_subdir": str(out_dir),
            "n_detections": len(rows),
        }
    )
    return payload


def main() -> int:
    args = parse_args()
    paths = _legacy_json_paths(args)
    if not paths:
        raise RuntimeError("no legacy JSON files selected")

    device = torch.device(args.device)
    model_by_band: dict[str, tuple[torch.nn.Module, dict]] = {}
    out_root = args.out_root.expanduser().resolve()
    manifest: list[dict[str, Any]] = []

    for json_path in paths:
        old = json.loads(json_path.read_text(encoding="utf-8"))
        band = str(old["band"])
        if band not in model_by_band:
            model_by_band[band] = load_cellect_model(
                args.checkpoint.expanduser().resolve(),
                args.config.expanduser().resolve() if args.config else None,
                device,
                [band],
                dynamic_image_size=False,
            )
        model, _cfg = model_by_band[band]
        stem = str(old.get("candidate_id") or Path(str(old["pack_path"])).stem)
        out_dir = _relative_output_dir(out_root, old)
        if bool(args.skip_existing) and (out_dir / f"{stem}_detections.csv").exists():
            print(f"[skip] {out_dir}", flush=True)
            manifest.append(json.loads((out_dir / f"{stem}.json").read_text(encoding="utf-8")))
            continue

        width = int(old.get("width", old.get("x1", 512) - old.get("x0", 0)) or 512)
        height = int(old.get("height", old.get("y1", 512) - old.get("y0", 0)) or 512)
        image, header = _read_crop(Path(str(old["pack_path"])), int(old["x0"]), int(old["y0"]), width, height)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        scaled = make_training_rgb(
            image,
            mode=str(args.scaling_mode),
            clip_threshold=float(args.clip_threshold),
            log_a=float(args.log_a),
            log_high_percentile=float(args.log_high_percentile),
            lupton_stretch=float(args.lupton_stretch),
            lupton_q=float(args.lupton_q),
            anscombe_clip=bool(args.anscombe_clip),
            anscombe_scale=float(args.anscombe_scale),
        )
        tensor = torch.from_numpy(scaled[None, None].astype(np.float32, copy=False))
        outputs = infer_cellect(model=model, image_tensor=tensor, device=device, amp=str(args.amp))
        band_outputs = select_band_outputs(outputs, 0)
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

        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f"{stem}.npz", image=image, scaled=scaled)
        _save_titled_rgb(out_dir / f"{stem}.png", _display_gray_uint8(image), stem)
        detect_overlay = draw_ellipses(image, rows, color="cyan", draw_centers=bool(args.shape_overlay_centers))
        save_pixel_png(out_dir / f"{stem}_detect_overlay.png", detect_overlay)
        _save_titled_rgb(
            out_dir / f"{stem}_detect_overlay_titled.png",
            np.clip(np.rint(np.flipud(detect_overlay) * 255.0), 0, 255).astype(np.uint8),
            f"{stem} detections={len(rows)}",
        )
        input_overlay = _input_shape_overlay(
            scaled,
            rows,
            scaling=str(args.scaling_mode),
            clip_threshold=float(args.clip_threshold),
            draw_centers=bool(args.shape_overlay_centers),
        )
        save_pixel_png(out_dir / f"{stem}_input_shape_overlay.png", input_overlay)
        _save_titled_rgb(
            out_dir / f"{stem}_input_shape_overlay_titled.png",
            np.clip(np.rint(np.flipud(input_overlay) * 255.0), 0, 255).astype(np.uint8),
            f"{stem} input shape detections={len(rows)}",
        )

        score = _score_map(band_outputs)[:height, :width]
        save_heatmap(out_dir / f"{stem}_confidence_score.png", score, title=f"{stem} confidence score")
        save_pixel_png(out_dir / f"{stem}_confidence_overlay.png", _pred_conf_overlay(image, band_outputs))
        save_pixel_png(out_dir / f"{stem}_input_rgb.png", scaled_rgb_for_display(scaled))
        for channel in range(min(3, scaled.shape[0])):
            vmin, vmax = input_channel_display_limits(
                scaled[channel],
                scaling=str(args.scaling_mode),
                channel_index=channel,
                clip_threshold=float(args.clip_threshold),
            )
            save_heatmap(
                out_dir / f"{stem}_input_channel{channel}.png",
                scaled[channel],
                title=f"{stem} input channel {channel}",
                vmin=vmin,
                vmax=vmax,
            )

        write_reg(out_dir / f"{stem}_centers.reg", rows_to_reg(rows, shape=False, color="yellow"))
        write_reg(out_dir / f"{stem}_shapes.reg", rows_to_reg(rows, shape=True, color="cyan"))
        write_sources_csv(out_dir / f"{stem}_sources.csv", rows, header=header, x0=int(old["x0"]), y0=int(old["y0"]))
        _write_detection_csv(out_dir / f"{stem}_detections.csv", rows)
        payload = _manifest_row(old, args, out_dir, rows)
        (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest.append(payload)
        print(f"[done] {stem}: detections={len(rows)} -> {out_dir}", flush=True)

    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "export_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
