#!/usr/bin/env python3
"""Diagnose the SAM image encoder patch embedding."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_utils import make_training_rgb, zscale_gray  # noqa: E402


DEFAULT_CHECKPOINT = Path("/data/czh23/ckpts/sam_log_lupton_0810/epoch_0020.pt")
DEFAULT_IMAGE = Path("/data/shared/Subaru/9813/HSC-I/4,5/calexp-HSC-I-9813-4,5.fits")
DEFAULT_OUT_DIR = Path("/home/czh23/analysis/2026-08/2026-08-21/patch_embed_svd_sam_log_lupton_0810_epoch20")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--hdu", type=int, default=None)
    p.add_argument("--x0", type=int, default=1472)
    p.add_argument("--y0", type=int, default=1840)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--scaling-mode", default="log_lupton")
    p.add_argument("--clip-threshold", type=float, default=5.0)
    p.add_argument("--log-a", type=float, default=300.0)
    p.add_argument("--log-high-percentile", type=float, default=99.5)
    p.add_argument("--lupton-stretch", type=float, default=0.5)
    p.add_argument("--lupton-q", type=float, default=20.0)
    p.add_argument("--low-ranks", type=int, nargs="+", default=[64, 128, 256, 512, 768])
    return p.parse_args()


def _state_dict(path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path.expanduser(), map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    return {str(k).removeprefix("module."): v for k, v in state.items() if torch.is_tensor(v)}


def _patch_embed_weight(state: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    weight_key = next((k for k in state if k.endswith("patch_embed.proj.weight")), None)
    if weight_key is None:
        raise KeyError("checkpoint has no patch_embed.proj.weight")
    bias_key = weight_key.removesuffix("weight") + "bias"
    weight = state[weight_key].detach().float().cpu().numpy()
    bias = state[bias_key].detach().float().cpu().numpy() if bias_key in state else np.zeros(weight.shape[0], dtype=np.float32)
    return weight, bias


def _read_image(path: Path, hdu: int | None) -> np.ndarray:
    with fits.open(path.expanduser(), memmap=True) as hdul:
        if hdu is not None:
            data = hdul[int(hdu)].data
        else:
            data = None
            for item in hdul:
                if item.data is not None and np.asarray(item.data).ndim == 2:
                    data = item.data
                    break
        if data is None:
            raise ValueError(f"no 2D image HDU found in {path}")
        arr = np.asarray(data, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _crop(image: np.ndarray, x0: int, y0: int, size: int) -> np.ndarray:
    h, w = image.shape
    x0 = max(0, min(int(x0), max(w - int(size), 0)))
    y0 = max(0, min(int(y0), max(h - int(size), 0)))
    return image[y0 : y0 + int(size), x0 : x0 + int(size)].astype(np.float32, copy=False)


def _patchify(chw: np.ndarray, patch: int = 16) -> np.ndarray:
    c, h, w = chw.shape
    if h % patch or w % patch:
        raise ValueError(f"image shape {(h, w)} is not divisible by patch={patch}")
    view = chw.reshape(c, h // patch, patch, w // patch, patch)
    view = view.transpose(1, 3, 0, 2, 4)
    return view.reshape((h // patch) * (w // patch), c * patch * patch)


def _unpatchify(patches: np.ndarray, *, channels: int = 3, height: int = 512, width: int = 512, patch: int = 16) -> np.ndarray:
    hp, wp = height // patch, width // patch
    view = patches.reshape(hp, wp, channels, patch, patch)
    return view.transpose(2, 0, 3, 1, 4).reshape(channels, height, width)


def _normalize_signed_rgb(chw: np.ndarray, *, pct: float = 99.0) -> np.ndarray:
    arr = np.asarray(chw, dtype=np.float32)
    scale = np.nanpercentile(np.abs(arr[np.isfinite(arr)]), pct) if np.isfinite(arr).any() else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    rgb = np.clip(arr / scale * 0.5 + 0.5, 0.0, 1.0)
    return np.moveaxis(rgb, 0, -1)


def _display_rgb(chw: np.ndarray, *, vmin: float = -5.0, vmax: float = 5.0) -> np.ndarray:
    arr = np.clip((np.asarray(chw, dtype=np.float32) - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    return np.moveaxis(arr, 0, -1)


def _radial_power(filters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # filters: [n, c, 16, 16]
    f = np.fft.fftshift(np.fft.fft2(filters, axes=(-2, -1)), axes=(-2, -1))
    power = np.mean(np.abs(f) ** 2, axis=(0, 1))
    h, w = power.shape
    yy, xx = np.mgrid[:h, :w]
    rr = np.sqrt((xx - (w - 1) / 2.0) ** 2 + (yy - (h - 1) / 2.0) ** 2)
    bins = np.arange(0, int(math.ceil(rr.max())) + 2)
    centers = 0.5 * (bins[:-1] + bins[1:])
    values = np.zeros_like(centers, dtype=np.float64)
    for i in range(len(centers)):
        mask = (rr >= bins[i]) & (rr < bins[i + 1])
        values[i] = float(np.mean(power[mask])) if np.any(mask) else np.nan
    return centers, values


def _high_frequency_fraction(patterns: np.ndarray, cutoff: float = 0.35) -> np.ndarray:
    # patterns: [n, c, 16, 16]
    f = np.fft.fftshift(np.fft.fft2(patterns, axes=(-2, -1)), axes=(-2, -1))
    power = np.sum(np.abs(f) ** 2, axis=1)
    h, w = power.shape[-2:]
    yy, xx = np.mgrid[:h, :w]
    rr = np.sqrt((xx - (w - 1) / 2.0) ** 2 + (yy - (h - 1) / 2.0) ** 2)
    rr = rr / max(float(rr.max()), 1.0)
    high = rr >= float(cutoff)
    total = np.sum(power, axis=(-2, -1))
    return np.sum(power[:, high], axis=1) / np.maximum(total, 1e-12)


def plot_svd(out_dir: Path, singular: np.ndarray, hf_frac: np.ndarray) -> dict[str, float | int]:
    energy = singular**2
    frac = energy / max(float(np.sum(energy)), 1e-12)
    cumulative = np.cumsum(frac)
    effective_rank = float(np.exp(-np.sum(frac * np.log(np.maximum(frac, 1e-12)))))
    rank_tol = int(np.sum(singular > singular.max() * 1e-6))
    summary = {
        "rank_tol_1e-6": rank_tol,
        "condition_number": float(singular.max() / max(singular.min(), 1e-12)),
        "effective_rank_entropy": effective_rank,
        "components_90pct_energy": int(np.searchsorted(cumulative, 0.90) + 1),
        "components_95pct_energy": int(np.searchsorted(cumulative, 0.95) + 1),
        "components_99pct_energy": int(np.searchsorted(cumulative, 0.99) + 1),
        "top256_energy_fraction": float(cumulative[min(255, len(cumulative) - 1)]),
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), dpi=170, constrained_layout=True)
    axes[0].plot(singular, lw=1.6)
    axes[0].set_title("PatchEmbed singular values")
    axes[0].set_xlabel("index")
    axes[0].set_ylabel("singular value")
    axes[0].set_yscale("log")
    axes[1].plot(cumulative, lw=1.6)
    axes[1].axhline(0.90, color="gray", ls="--", lw=0.8)
    axes[1].axhline(0.99, color="gray", ls=":", lw=0.8)
    axes[1].set_ylim(0, 1.01)
    axes[1].set_title("Cumulative energy")
    axes[1].set_xlabel("top-k components")
    axes[1].set_ylabel("fraction")
    axes[2].scatter(np.arange(len(hf_frac)), hf_frac, s=5, alpha=0.55)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Right-singular-vector high-frequency fraction")
    axes[2].set_xlabel("singular vector index")
    axes[2].set_ylabel("FFT power fraction")
    fig.savefig(out_dir / "patch_embed_svd_spectrum.png")
    plt.close(fig)
    return summary


def plot_patterns(out_dir: Path, vt: np.ndarray, filters: np.ndarray) -> None:
    top = vt[:24].reshape(24, 3, 16, 16)
    fig, axes = plt.subplots(4, 6, figsize=(12, 8), dpi=180, constrained_layout=True)
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(_normalize_signed_rgb(top[i]), origin="lower", interpolation="nearest")
        ax.set_title(f"V{i+1}", fontsize=8)
        ax.set_axis_off()
    fig.suptitle("Top right singular vectors reshaped to 3x16x16")
    fig.savefig(out_dir / "patch_embed_top_right_singular_vectors.png")
    plt.close(fig)

    centers, values = _radial_power(filters)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=170, constrained_layout=True)
    ax.plot(centers, values / max(float(np.nanmax(values)), 1e-12), marker="o", lw=1.4)
    ax.set_title("Mean radial Fourier power of learned patch filters")
    ax.set_xlabel("FFT radius on 16x16 kernel")
    ax.set_ylabel("normalized mean power")
    fig.savefig(out_dir / "patch_embed_filter_radial_power.png")
    plt.close(fig)


def plot_reconstruction(
    out_dir: Path,
    raw_crop: np.ndarray,
    scaled: np.ndarray,
    w: np.ndarray,
    bias: np.ndarray,
    u: np.ndarray,
    singular: np.ndarray,
    vt: np.ndarray,
    ranks: list[int],
) -> dict[str, float]:
    patches = _patchify(scaled)
    tokens = patches @ w.T + bias[None, :]
    pinv = np.linalg.pinv(w)
    full = (tokens - bias[None, :]) @ pinv.T
    full_img = _unpatchify(full, height=scaled.shape[1], width=scaled.shape[2])

    metrics: dict[str, float] = {
        "full_pinv_rmse": float(np.sqrt(np.mean((full_img - scaled) ** 2))),
        "full_pinv_mae": float(np.mean(np.abs(full_img - scaled))),
    }

    rank_imgs: list[tuple[int, np.ndarray, float]] = []
    for rank in ranks:
        k = min(int(rank), len(singular))
        wk = (u[:, :k] * singular[:k]) @ vt[:k]
        tokens_k = patches @ wk.T + bias[None, :]
        patch_k = (tokens_k - bias[None, :]) @ np.linalg.pinv(wk).T
        img_k = _unpatchify(patch_k, height=scaled.shape[1], width=scaled.shape[2])
        rmse = float(np.sqrt(np.mean((img_k - scaled) ** 2)))
        metrics[f"rank_{k}_pinv_rmse"] = rmse
        rank_imgs.append((k, img_k, rmse))

    token_norm = np.linalg.norm(tokens, axis=1).reshape(scaled.shape[1] // 16, scaled.shape[2] // 16)
    token_norm = np.repeat(np.repeat(token_norm, 16, axis=0), 16, axis=1)
    patch_mean = patches.reshape(-1, 3, 16, 16).mean(axis=(2, 3)).reshape(scaled.shape[1] // 16, scaled.shape[2] // 16, 3)
    patch_mean = np.repeat(np.repeat(patch_mean, 16, axis=0), 16, axis=1)

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), dpi=170, constrained_layout=True)
    axes[0, 0].imshow(zscale_gray(raw_crop), origin="lower", cmap="gray", interpolation="nearest")
    axes[0, 0].set_title("raw zscale crop")
    axes[0, 1].imshow(_display_rgb(scaled, vmin=-5, vmax=5), origin="lower", interpolation="nearest")
    axes[0, 1].set_title("model input: z/log/lupton")
    axes[0, 2].imshow(_display_rgb(full_img, vmin=-5, vmax=5), origin="lower", interpolation="nearest")
    axes[0, 2].set_title(f"PatchEmbed pinv recon\nRMSE={metrics['full_pinv_rmse']:.2e}")
    axes[0, 3].imshow(np.mean(np.abs(full_img - scaled), axis=0), origin="lower", cmap="magma", interpolation="nearest")
    axes[0, 3].set_title("absolute recon error")
    axes[1, 0].imshow(token_norm, origin="lower", cmap="viridis", interpolation="nearest")
    axes[1, 0].set_title("token L2 norm map\n32x32 upsampled")
    axes[1, 1].imshow(np.clip((patch_mean - np.nanmin(patch_mean)) / max(np.nanmax(patch_mean) - np.nanmin(patch_mean), 1e-6), 0, 1), origin="lower", interpolation="nearest")
    axes[1, 1].set_title("per-patch mean input\nblocky reference")
    if rank_imgs:
        k, img_k, rmse = rank_imgs[min(2, len(rank_imgs) - 1)]
        axes[1, 2].imshow(_display_rgb(img_k, vmin=-5, vmax=5), origin="lower", interpolation="nearest")
        axes[1, 2].set_title(f"rank-{k} low-rank recon\nRMSE={rmse:.3f}")
        axes[1, 3].imshow(np.mean(np.abs(img_k - scaled), axis=0), origin="lower", cmap="magma", interpolation="nearest")
        axes[1, 3].set_title(f"rank-{k} abs error")
    for ax in axes.ravel():
        ax.set_axis_off()
    fig.savefig(out_dir / "patch_embed_actual_crop_reconstruction.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(rank_imgs) + 1, figsize=(4 * (len(rank_imgs) + 1), 4), dpi=150, constrained_layout=True)
    axes[0].imshow(_display_rgb(scaled, vmin=-5, vmax=5), origin="lower", interpolation="nearest")
    axes[0].set_title("input")
    axes[0].set_axis_off()
    for ax, (k, img_k, rmse) in zip(axes[1:], rank_imgs, strict=True):
        ax.imshow(_display_rgb(img_k, vmin=-5, vmax=5), origin="lower", interpolation="nearest")
        ax.set_title(f"rank {k}\nRMSE={rmse:.3f}", fontsize=9)
        ax.set_axis_off()
    fig.savefig(out_dir / "patch_embed_low_rank_reconstruction_grid.png")
    plt.close(fig)
    return metrics


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    state = _state_dict(args.checkpoint)
    weight4, bias = _patch_embed_weight(state)
    out_chans, in_chans, patch_h, patch_w = weight4.shape
    if (in_chans, patch_h, patch_w) != (3, 16, 16):
        raise ValueError(f"expected PatchEmbed weight [out,3,16,16], got {weight4.shape}")
    w = weight4.reshape(out_chans, -1).astype(np.float64)
    bias = bias.astype(np.float64)
    u, singular, vt = np.linalg.svd(w, full_matrices=False)
    filters = weight4.astype(np.float64)
    hf_frac = _high_frequency_fraction(vt.reshape(vt.shape[0], 3, 16, 16))

    summary = plot_svd(out_dir, singular, hf_frac)
    plot_patterns(out_dir, vt, filters)

    image = _read_image(args.image, args.hdu)
    raw_crop = _crop(image, args.x0, args.y0, args.size)
    scaled = make_training_rgb(
        raw_crop,
        mode=str(args.scaling_mode),
        clip_threshold=float(args.clip_threshold),
        log_a=float(args.log_a),
        log_high_percentile=float(args.log_high_percentile),
        lupton_stretch=float(args.lupton_stretch),
        lupton_q=float(args.lupton_q),
    )
    recon_metrics = plot_reconstruction(
        out_dir,
        raw_crop,
        scaled,
        w,
        bias,
        u,
        singular,
        vt,
        [int(v) for v in args.low_ranks],
    )
    summary.update(recon_metrics)
    summary.update(
        {
            "checkpoint": str(args.checkpoint.expanduser()),
            "image": str(args.image.expanduser()),
            "crop_x0": int(args.x0),
            "crop_y0": int(args.y0),
            "crop_size": int(args.size),
            "scaling_mode": str(args.scaling_mode),
            "clip_threshold": float(args.clip_threshold),
            "patch_embed_weight_shape": list(weight4.shape),
            "bias_norm": float(np.linalg.norm(bias)),
            "weight_frobenius_norm": float(np.linalg.norm(w)),
        }
    )
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
