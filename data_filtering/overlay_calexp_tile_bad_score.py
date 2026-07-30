#!/usr/bin/env python3
"""Overlay tile-level calexp bad-score bins on Astropy-zscale images."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_filtering.calexp_quality import (  # noqa: E402
    DEFAULT_BAD_SCORE_WEIGHTS,
    find_calexp,
    normalize_band_dir,
    parse_patches,
    parse_score_weights,
    read_calexp,
    score_regular_tiles,
)


COLORS = {
    "5_10": (0.05, 1.0, 0.0, 0.22),
    "10_20": (1.0, 0.95, 0.0, 0.32),
    "20_50": (1.0, 0.45, 0.0, 0.42),
    "gt_50": (1.0, 0.0, 0.0, 0.55),
}


def zscale_limits(image: np.ndarray) -> tuple[float, float]:
    from astropy.visualization import ZScaleInterval

    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    if finite.size > 500_000:
        finite = finite[:: int(np.ceil(finite.size / 500_000))]
    try:
        lo, hi = ZScaleInterval().get_limits(finite)
    except Exception as e:
        print(f'[WARNING] ZScale failed: {e}. Falling back to 1st/99th percentile.', file=sys.stderr)
        lo, hi = np.nanpercentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        med = float(np.nanmedian(finite))
        sigma = float(np.nanstd(finite))
        if not np.isfinite(sigma) or sigma <= 0.0:
            sigma = 1.0
        lo, hi = med - sigma, med + sigma
    return float(lo), float(hi)


def downsample_mean(image: np.ndarray, factor: int) -> np.ndarray:
    factor = max(1, int(factor))
    if factor <= 1:
        return image
    h, w = image.shape
    out_h = max(1, h // factor)
    out_w = max(1, w // factor)
    trimmed = image[: out_h * factor, : out_w * factor]
    with np.errstate(invalid="ignore"):
        return np.nanmean(trimmed.reshape(out_h, factor, out_w, factor), axis=(1, 3)).astype(np.float32, copy=False)


def score_tiles(mask: np.ndarray, planes: dict[str, int], weights: dict[str, float], tile_size: int, stride: int) -> list[dict[str, object]]:
    return score_regular_tiles(mask, planes, weights, tile_size=int(tile_size), stride=int(stride))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["band", "patch", "x0", "y0", "x1", "y1", "score", "score_percent", "category"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot_overlay(
    image: np.ndarray,
    rows: list[dict[str, object]],
    out_path: Path,
    *,
    title: str,
    downsample: int,
    label_tiles: bool,
) -> None:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lo, hi = zscale_limits(image)
    show = downsample_mean(image, downsample)
    fig_w = max(7.0, show.shape[1] / 180.0)
    fig_h = max(7.0, show.shape[0] / 180.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    ax.imshow(show, cmap="gray", origin="lower", interpolation="nearest", vmin=lo, vmax=hi)

    for row in rows:
        category = str(row["category"])
        color = COLORS.get(category)
        if color is None:
            continue
        x0 = float(row["x0"]) / downsample
        y0 = float(row["y0"]) / downsample
        width = (float(row["x1"]) - float(row["x0"])) / downsample
        height = (float(row["y1"]) - float(row["y0"])) / downsample
        ax.add_patch(
            patches.Rectangle(
                (x0, y0),
                width,
                height,
                facecolor=color,
                edgecolor=(1.0, 1.0, 1.0, 0.55),
                linewidth=0.45,
            )
        )
        if label_tiles:
            ax.text(
                x0 + width * 0.5,
                y0 + height * 0.5,
                f"{float(row['score_percent']):.0f}%",
                ha="center",
                va="center",
                fontsize=5,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.35, "pad": 0.3, "edgecolor": "none"},
            )

    legend_handles = [
        patches.Patch(facecolor=COLORS["10_20"], label="10-20%"),
        patches.Patch(facecolor=COLORS["20_50"], label="20-50%"),
        patches.Patch(facecolor=COLORS["gt_50"], label=">50%"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.75)
    ax.set_title(f"{title} | tile bad score overlay | zscale=({lo:.3g},{hi:.3g})", fontsize=9)
    ax.set_xlabel("x pixel / downsample")
    ax.set_ylabel("y pixel / downsample")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay tile-level calexp bad-score bins on zscale images.")
    parser.add_argument("--data-root", type=Path, default=Path("/data/shared/Subaru"))
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--band", required=True)
    parser.add_argument("--patches", nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("output/calexp_tile_bad_score_overlays"))
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=368, help="Tile stride. Default: 368, matching training cutouts.")
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--bad-score-weights", nargs="*", default=None)
    parser.add_argument("--label-tiles", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    band = normalize_band_dir(args.band)
    patches = parse_patches(args.patches)
    weights = parse_score_weights(args.bad_score_weights)
    stride = int(args.stride)

    for patch in patches:
        patch_dir = args.data_root / str(args.tract) / band / patch
        calexp = find_calexp(patch_dir)
        if calexp is None:
            print(f"{band} {patch}: missing calexp")
            continue
        image, mask, planes = read_calexp(calexp)
        rows = score_tiles(mask, planes, weights, int(args.tile_size), stride)
        for row in rows:
            row["band"] = band
            row["patch"] = patch
        safe_patch = patch.replace(",", "_")
        out_base = args.out_dir / str(args.tract) / band
        png_path = out_base / f"{band}_{args.tract}_{safe_patch}_tile_bad_score_overlay.png"
        csv_path = out_base / f"{band}_{args.tract}_{safe_patch}_tile_bad_score.csv"
        write_csv(csv_path, rows)
        plot_overlay(
            image,
            rows,
            png_path,
            title=f"{band} {args.tract} {patch}",
            downsample=max(1, int(args.downsample)),
            label_tiles=bool(args.label_tiles),
        )
        counts = {key: sum(1 for row in rows if row["category"] == key) for key in ("5_10", "10_20", "20_50", "gt_50")}
        print(f"{band} {patch}: tiles={len(rows)} {counts} -> {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
