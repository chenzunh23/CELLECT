#!/usr/bin/env python3
"""Visualize SAM encoder/decoder feature flow for selected HSC cutouts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_utils import (  # noqa: E402
    crop_origin_for_image,
    crop_or_pad,
    infer_cellect,
    load_cellect_model,
    make_training_rgb,
    read_fits_image,
    zscale_gray,
)


DEFAULT_CHECKPOINT = Path("/data/czh23/ckpts/sam_log_lupton_0810/epoch_0020.pt")
DEFAULT_IMAGE = Path("/data/shared/Subaru/9813/HSC-I/4,5/calexp-HSC-I-9813-4,5.fits")
DEFAULT_OUT_DIR = Path("/home/czh23/analysis/2026-08/2026-08-21/feature_flow_sam_log_lupton_0810_epoch20")

R5C4_SOURCES = (
    (17532.0, 22201.0),
    (17511.0, 21975.0),
    (17542.0, 21980.0),
    (17840.0, 21838.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--band", default="HSC-I")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", choices=("none", "bf16"), default="none")
    parser.add_argument("--hdu", type=int, default=None)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument(
        "--crop",
        action="append",
        default=None,
        help="Crop spec name:x0:y0 in physical coordinates. Defaults to r5c4 and r7c10.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Source spec crop_name:x:y in physical coordinates. Defaults to four r5c4 sources.",
    )
    parser.add_argument(
        "--r7c10-local-source",
        action="append",
        default=None,
        help="Manual r7c10 source spec label:x:y in local crop pixels.",
    )
    parser.add_argument("--scaling-mode", default="log_lupton")
    parser.add_argument("--clip-threshold", type=float, default=5.0)
    parser.add_argument("--log-a", type=float, default=300.0)
    parser.add_argument("--log-high-percentile", type=float, default=99.5)
    parser.add_argument("--lupton-stretch", type=float, default=0.5)
    parser.add_argument("--lupton-q", type=float, default=20.0)
    parser.add_argument("--anscombe-clip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--anscombe-scale", type=float, default=1000.0)
    parser.add_argument("--source-window", type=int, default=64)
    return parser.parse_args()


def _parse_crop(spec: str) -> tuple[str, int, int]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"crop must be name:x0:y0, got {spec!r}")
    return parts[0], int(float(parts[1])), int(float(parts[2]))


def _default_crops() -> list[tuple[str, int, int]]:
    # r7c10 is derived from r5c4 by the preprocessing stride: dx=6*368, dy=2*368.
    return [("r5c4", 17372, 21740), ("r7c10", 19580, 22476)]


def _default_sources() -> dict[str, list[tuple[str, float, float]]]:
    out: dict[str, list[tuple[str, float, float]]] = {"r5c4": [], "r7c10": []}
    for idx, (x, y) in enumerate(R5C4_SOURCES, start=1):
        out["r5c4"].append((f"s{idx}", x, y))
    r7_x0, r7_y0 = 19580, 22476
    for label, lx, ly in (
        ("left_lower", 80.0, 80.0),
        ("right_lower", 440.0, 80.0),
        ("upper_center_1", 220.0, 330.0),
        ("upper_center_2", 310.0, 355.0),
    ):
        out["r7c10"].append((label, r7_x0 + lx, r7_y0 + ly))
    return out


def _parse_sources(args: argparse.Namespace) -> dict[str, list[tuple[str, float, float]]]:
    out = _default_sources()
    if args.source:
        out = {}
        for i, spec in enumerate(args.source, start=1):
            parts = spec.split(":")
            if len(parts) != 3:
                raise ValueError(f"source must be crop_name:x:y, got {spec!r}")
            out.setdefault(parts[0], []).append((f"s{i}", float(parts[1]), float(parts[2])))
    if args.r7c10_local_source:
        out.setdefault("r7c10", [])
        for spec in args.r7c10_local_source:
            parts = spec.split(":")
            if len(parts) != 3:
                raise ValueError(f"r7c10 local source must be label:x:y, got {spec!r}")
            out["r7c10"].append((parts[0], 19580 + float(parts[1]), 22476 + float(parts[2])))
    return out


def _capture_tensor(value: object, *, channel_last: bool) -> np.ndarray | None:
    if not torch.is_tensor(value):
        return None
    tensor = value.detach().float().cpu()
    if tensor.ndim == 4 and channel_last:
        # SAM ViT internals: [B, H, W, C].
        tensor = tensor[0].permute(2, 0, 1)
    elif tensor.ndim == 4:
        # CNN-like internals: [B, C, H, W].
        tensor = tensor[0]
    elif tensor.ndim == 5:
        tensor = tensor[0, 0]
    else:
        return None
    return tensor.numpy().astype(np.float32, copy=False)


def _register_hooks(model: torch.nn.Module, captures: OrderedDict[str, np.ndarray]) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    modules: list[tuple[str, torch.nn.Module, bool]] = [
        ("patch_embed", model.encoder.image_encoder.patch_embed, True),
        ("block0", model.encoder.image_encoder.blocks[0], True),
        ("block2", model.encoder.image_encoder.blocks[2], True),
        ("block5", model.encoder.image_encoder.blocks[5], True),
        ("block8", model.encoder.image_encoder.blocks[8], True),
        ("block11", model.encoder.image_encoder.blocks[11], True),
        ("neck", model.encoder.image_encoder.neck, False),
        ("dec_stem", model.decoder.stem, False),
        ("dec_up1", model.decoder.up1, False),
        ("dec_up2", model.decoder.up2, False),
        ("dec_up3", model.decoder.up3, False),
        ("dec_up4", model.decoder.up4, False),
        ("dec_refine", model.decoder.refine, False),
        ("confidence_logits", model.decoder.confidence_head, False),
    ]

    def make_hook(name: str, channel_last: bool):
        def hook(_module, _inputs, output):
            arr = _capture_tensor(output, channel_last=channel_last)
            if arr is not None:
                captures[name] = arr

        return hook

    for name, module, channel_last in modules:
        handles.append(module.register_forward_hook(make_hook(name, channel_last)))
    return handles


def _score_map(outputs: dict[str, torch.Tensor]) -> np.ndarray:
    logits = outputs["confidence"][0, 0].detach().float().cpu()
    prob = torch.softmax(logits, dim=0)
    levels = torch.arange(logits.shape[0], dtype=prob.dtype).view(-1, 1, 1)
    return (prob * levels).sum(dim=0).numpy().astype(np.float32)


def _feature_norm(feature: np.ndarray) -> np.ndarray:
    arr = np.asarray(feature, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    return np.sqrt(np.nanmean(arr * arr, axis=0))


def _robust_unit(image: np.ndarray, pct: tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    if not bool(finite.any()):
        return np.zeros(arr.shape, dtype=np.float32)
    lo, hi = np.nanpercentile(arr[finite], pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(arr[finite])), float(np.nanmax(arr[finite]))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def _upsample_to(image: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.shape[-2:] == (size, size):
        return arr
    sy = max(1, size // arr.shape[-2])
    sx = max(1, size // arr.shape[-1])
    out = np.repeat(np.repeat(arr, sy, axis=-2), sx, axis=-1)
    return out[:size, :size]


def _feature_panel_image(feature: np.ndarray, size: int) -> np.ndarray:
    return _robust_unit(_upsample_to(_feature_norm(feature), size))


def _pca_rgb(feature: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(feature, dtype=np.float32)
    if arr.ndim == 2:
        gray = _robust_unit(_upsample_to(arr, size))
        return np.repeat(gray[..., None], 3, axis=2)
    c, h, w = arr.shape
    x = arr.reshape(c, h * w).T
    x = x - np.nanmean(x, axis=0, keepdims=True)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        _u, _s, vt = np.linalg.svd(x, full_matrices=False)
        rgb_low = (x @ vt[:3].T).reshape(h, w, min(3, vt.shape[0]))
    except np.linalg.LinAlgError:
        rgb_low = np.zeros((h, w, 3), dtype=np.float32)
    if rgb_low.shape[2] < 3:
        rgb_low = np.pad(rgb_low, ((0, 0), (0, 0), (0, 3 - rgb_low.shape[2])))
    channels = [_robust_unit(_upsample_to(rgb_low[..., i], size), (1.0, 99.0)) for i in range(3)]
    return np.stack(channels, axis=2)


def _plot_grid(
    path: Path,
    panels: list[tuple[str, np.ndarray, str]],
    *,
    title: str,
    cols: int,
    cmap_default: str = "viridis",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), dpi=150, constrained_layout=True)
    flat_axes = np.asarray(axes).reshape(-1)
    for ax, (label, image, cmap) in zip(flat_axes, panels):
        arr = np.asarray(image)
        if arr.ndim == 3:
            ax.imshow(np.clip(arr, 0.0, 1.0), origin="lower", interpolation="nearest")
        else:
            ax.imshow(arr, origin="lower", interpolation="nearest", cmap=cmap or cmap_default, vmin=0.0, vmax=1.0)
        ax.set_title(label, fontsize=10)
        ax.set_axis_off()
    for ax in flat_axes[len(panels) :]:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=13)
    fig.savefig(path)
    plt.close(fig)


def _crop_raw_image(image: np.ndarray, header, *, x0_phys: int, y0_phys: int, size: int) -> tuple[np.ndarray, int, int]:
    local_x0, local_y0 = crop_origin_for_image(image, header, x0=x0_phys, y0=y0_phys, width=size, height=size)
    crop, valid = crop_or_pad(image, x0=local_x0, y0=local_y0, width=size, height=size, inference_size=size)
    crop = np.where(valid, crop, np.nan).astype(np.float32)
    return crop, local_x0, local_y0


def _local_sources(
    sources: list[tuple[str, float, float]],
    *,
    crop_x0: int,
    crop_y0: int,
    size: int,
) -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = []
    for label, x, y in sources:
        lx = float(x) - float(crop_x0)
        ly = float(y) - float(crop_y0)
        if 0.0 <= lx < size and 0.0 <= ly < size:
            out.append((label, lx, ly))
    return out


def _draw_source_overview(ax, raw_crop: np.ndarray, sources: list[tuple[str, float, float]]) -> None:
    ax.imshow(zscale_gray(raw_crop), origin="lower", cmap="gray", interpolation="nearest")
    for idx, (label, x, y) in enumerate(sources, start=1):
        ax.scatter([x], [y], s=80, facecolors="none", edgecolors="cyan", linewidths=1.4)
        ax.text(x + 5, y + 5, f"{idx}:{label}", color="yellow", fontsize=8, weight="bold")
    ax.set_title("raw crop + diagnostic points")
    ax.set_axis_off()


def _cosine_patch(feature: np.ndarray, x: float, y: float, *, token_size: int = 16, radius: int = 4) -> np.ndarray:
    arr = np.asarray(feature, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    c, h, w = arr.shape
    tx = int(np.clip(np.floor(float(x) / token_size), 0, w - 1))
    ty = int(np.clip(np.floor(float(y) / token_size), 0, h - 1))
    vector = arr[:, ty, tx]
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    y0 = max(0, ty - radius)
    y1 = min(h, ty + radius + 1)
    x0 = max(0, tx - radius)
    x1 = min(w, tx + radius + 1)
    patch = arr[:, y0:y1, x0:x1].reshape(c, -1)
    denom = np.linalg.norm(patch, axis=0) * norm
    vals = (vector[None] @ patch / np.maximum(denom[None], 1e-12)).reshape(y1 - y0, x1 - x0)
    out = np.full((2 * radius + 1, 2 * radius + 1), np.nan, dtype=np.float32)
    oy = y0 - (ty - radius)
    ox = x0 - (tx - radius)
    out[oy : oy + vals.shape[0], ox : ox + vals.shape[1]] = vals
    return out


def _confidence_crop(score: np.ndarray, x: float, y: float, *, width: int) -> np.ndarray:
    half = int(width) // 2
    x0 = int(round(x)) - half
    y0 = int(round(y)) - half
    out = np.full((width, width), np.nan, dtype=np.float32)
    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(score.shape[1], x0 + width)
    sy1 = min(score.shape[0], y0 + width)
    if sx1 > sx0 and sy1 > sy0:
        dx0 = sx0 - x0
        dy0 = sy0 - y0
        out[dy0 : dy0 + sy1 - sy0, dx0 : dx0 + sx1 - sx0] = score[sy0:sy1, sx0:sx1]
    return out


def _plot_source_diagnostics(
    path: Path,
    raw_crop: np.ndarray,
    captures: OrderedDict[str, np.ndarray],
    score: np.ndarray,
    sources: list[tuple[str, float, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not sources:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
        _draw_source_overview(ax, raw_crop, [])
        ax.text(
            0.5,
            0.5,
            "No requested diagnostic source falls inside this crop.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="yellow",
            fontsize=11,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 8},
        )
        fig.savefig(path)
        plt.close(fig)
        return

    stages = ["patch_embed", "block2", "block5", "block8", "block11", "neck"]
    cols = len(stages) + 2
    rows = len(sources)
    fig, axes = plt.subplots(rows, cols, figsize=(2.55 * cols, 2.4 * rows), dpi=160, constrained_layout=True)
    axes = np.asarray(axes).reshape(rows, cols)
    for row, (label, x, y) in enumerate(sources):
        ax0 = axes[row, 0]
        _draw_source_overview(ax0, raw_crop, [(label, x, y)])
        ax0.set_title(f"{label} ({x:.0f},{y:.0f})")
        for col, stage in enumerate(stages, start=1):
            ax = axes[row, col]
            if stage not in captures:
                ax.set_axis_off()
                continue
            cos = _cosine_patch(captures[stage], x, y)
            ax.imshow(cos, origin="lower", interpolation="nearest", cmap="coolwarm", vmin=-1.0, vmax=1.0)
            ax.scatter([4], [4], s=20, c="yellow", marker="+")
            ax.set_title(stage, fontsize=9)
            ax.set_axis_off()
        ax = axes[row, -1]
        conf = _confidence_crop(score, x, y, width=64)
        ax.imshow(conf, origin="lower", interpolation="nearest", cmap="magma", vmin=0.0, vmax=max(3.0, float(np.nanmax(score))))
        ax.scatter([32], [32], s=32, facecolors="none", edgecolors="cyan", linewidths=1.0)
        ax.set_title("confidence 64x64", fontsize=9)
        ax.set_axis_off()
    fig.suptitle("Source-token cosine maps and local confidence")
    fig.savefig(path)
    plt.close(fig)


def _run_crop(
    *,
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    image: np.ndarray,
    header,
    crop_name: str,
    x0_phys: int,
    y0_phys: int,
    sources_by_crop: dict[str, list[tuple[str, float, float]]],
) -> dict:
    out_dir = args.out_dir.expanduser() / crop_name
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_crop, local_x0, local_y0 = _crop_raw_image(image, header, x0_phys=x0_phys, y0_phys=y0_phys, size=int(args.size))
    scaled = make_training_rgb(
        raw_crop,
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

    captures: OrderedDict[str, np.ndarray] = OrderedDict()
    handles = _register_hooks(model, captures)
    try:
        outputs = infer_cellect(model=model, image_tensor=tensor, device=device, amp=str(args.amp))
    finally:
        for handle in handles:
            handle.remove()
    score = _score_map(outputs)

    size = int(args.size)
    raw_panel = zscale_gray(raw_crop)
    encoder_stages = ["patch_embed", "block0", "block2", "block5", "block8", "block11", "neck"]
    encoder_panels = [("raw zscale", raw_panel, "gray")]
    encoder_panels.extend((stage, _feature_panel_image(captures[stage], size), "viridis") for stage in encoder_stages if stage in captures)
    _plot_grid(
        out_dir / "encoder_stage_norms.png",
        encoder_panels,
        title=f"{crop_name}: encoder feature L2 norms",
        cols=4,
    )

    pca_panels = [("raw zscale", np.repeat(raw_panel[..., None], 3, axis=2), "")]
    pca_panels.extend((stage, _pca_rgb(captures[stage], size), "") for stage in encoder_stages if stage in captures)
    _plot_grid(
        out_dir / "encoder_stage_pca_rgb.png",
        pca_panels,
        title=f"{crop_name}: per-stage PCA RGB",
        cols=4,
    )

    decoder_stages = ["neck", "dec_stem", "dec_up1", "dec_up2", "dec_up3", "dec_up4", "dec_refine"]
    decoder_panels = [("raw zscale", raw_panel, "gray")]
    decoder_panels.extend((stage, _feature_panel_image(captures[stage], size), "viridis") for stage in decoder_stages if stage in captures)
    decoder_panels.append(("confidence score", _robust_unit(score, (0.0, 99.5)), "magma"))
    _plot_grid(
        out_dir / "decoder_upsample_path.png",
        decoder_panels,
        title=f"{crop_name}: neck and decoder path",
        cols=3,
    )

    local_sources = _local_sources(sources_by_crop.get(crop_name, []), crop_x0=x0_phys, crop_y0=y0_phys, size=size)
    _plot_source_diagnostics(
        out_dir / "source_token_diagnostics.png",
        raw_crop,
        captures,
        score,
        local_sources,
    )

    summary = {
        "crop": crop_name,
        "physical_origin": [int(x0_phys), int(y0_phys)],
        "local_origin": [int(local_x0), int(local_y0)],
        "size": int(size),
        "sources_local": [{"label": label, "x": float(x), "y": float(y)} for label, x, y in local_sources],
        "captured_shapes": {name: list(arr.shape) for name, arr in captures.items()},
        "confidence_score_range": [float(np.nanmin(score)), float(np.nanmax(score))],
        "outputs": [
            "encoder_stage_norms.png",
            "encoder_stage_pca_rgb.png",
            "decoder_upsample_path.png",
            "source_token_diagnostics.png",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    crops = [_parse_crop(spec) for spec in args.crop] if args.crop else _default_crops()
    sources_by_crop = _parse_sources(args)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, _cfg = load_cellect_model(args.checkpoint.expanduser(), None, device, [args.band], dynamic_image_size=False)
    image, header, hdu = read_fits_image(args.image.expanduser(), hdu=args.hdu)

    all_summaries = []
    for crop_name, x0_phys, y0_phys in crops:
        print(f"[feature-flow] running {crop_name} at physical ({x0_phys},{y0_phys})", flush=True)
        all_summaries.append(
            _run_crop(
                model=model,
                device=device,
                args=args,
                image=image,
                header=header,
                crop_name=crop_name,
                x0_phys=x0_phys,
                y0_phys=y0_phys,
                sources_by_crop=sources_by_crop,
            )
        )
    args.out_dir.expanduser().mkdir(parents=True, exist_ok=True)
    (args.out_dir.expanduser() / "summary.json").write_text(
        json.dumps({"image": str(args.image), "hdu": hdu, "checkpoint": str(args.checkpoint), "crops": all_summaries}, indent=2),
        encoding="utf-8",
    )
    print(f"[feature-flow] wrote {args.out_dir.expanduser()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
