#!/usr/bin/env python3
"""Run broadband patch/tile bad-score diagnostics and visualize top patches."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_calexp_mask_quality import (  # noqa: E402
    find_calexp,
    mask_planes_from_header,
    normalize_band_dir,
    parse_score_weights,
)
from scripts.overlay_calexp_tile_bad_score import (  # noqa: E402
    plot_overlay,
    read_calexp,
    score_tiles,
    tile_bad_score,
)


def parse_patches(values: Iterable[str]) -> list[str]:
    patches: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in str(value).split(";"):
            patch = item.strip()
            if not patch:
                continue
            expanded = [f"{x},{y}" for x in range(9) for y in range(9)] if patch.lower() == "all" else [patch]
            for candidate in expanded:
                if candidate not in seen:
                    patches.append(candidate)
                    seen.add(candidate)
    return patches


def read_calexp_mask(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    from astropy.io import fits

    with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
        mask = np.asarray(hdul[2].data, dtype=np.int64)
        planes = mask_planes_from_header(hdul[2].header)
    return mask, planes


def analyze_patch_task(
    *,
    data_root: str,
    tract: str,
    band: str,
    patch: str,
    weights: dict[str, float],
    tile_size: int,
    stride: int,
    threshold: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    patch_dir = Path(data_root) / str(tract) / band / patch
    calexp = find_calexp(patch_dir)
    patch_row: dict[str, object] = {
        "band": band,
        "patch": patch,
        "status": "missing",
        "path": "",
        "patch_score": "",
        "patch_score_percent": "",
    }
    if calexp is None:
        return patch_row, []
    try:
        mask, planes = read_calexp_mask(calexp)
        tiles = score_tiles(mask, planes, weights, int(tile_size), int(stride))
        patch_score = tile_bad_score(mask, planes, weights)
        scores = np.asarray([float(row["score"]) for row in tiles], dtype=np.float32)
        patch_row.update(
            {
                "status": "ok",
                "path": str(calexp),
                "height": int(mask.shape[0]),
                "width": int(mask.shape[1]),
                "patch_score": float(patch_score),
                "patch_score_percent": float(patch_score * 100.0),
                "tile_count": int(len(tiles)),
                "tile_score_mean": float(np.mean(scores)) if scores.size else "",
                "tile_score_p50": float(np.percentile(scores, 50)) if scores.size else "",
                "tile_score_p90": float(np.percentile(scores, 90)) if scores.size else "",
                "tile_score_p95": float(np.percentile(scores, 95)) if scores.size else "",
                "tile_score_max": float(np.max(scores)) if scores.size else "",
                "tile_gt_threshold_count": int(np.sum(scores > float(threshold))) if scores.size else 0,
                "tile_gt_threshold_fraction": float(np.mean(scores > float(threshold))) if scores.size else 0.0,
                "tile_gt_20_count": int(np.sum(scores >= 0.20)) if scores.size else 0,
                "tile_gt_50_count": int(np.sum(scores >= 0.50)) if scores.size else 0,
            }
        )
        tile_rows: list[dict[str, object]] = []
        for row in tiles:
            tile_rows.append(
                {
                    "band": band,
                    "patch": patch,
                    "x0": int(row["x0"]),
                    "y0": int(row["y0"]),
                    "x1": int(row["x1"]),
                    "y1": int(row["y1"]),
                    "score": float(row["score"]),
                    "score_percent": float(row["score_percent"]),
                    "category": str(row["category"]),
                }
            )
        return patch_row, tile_rows
    except Exception as exc:  # noqa: BLE001
        patch_row.update({"status": "error", "path": str(calexp), "error": f"{type(exc).__name__}: {exc}"})
        return patch_row, []


def patch_sort_key(row: dict[str, object]) -> tuple[str, int, int]:
    band = str(row.get("band", ""))
    patch = str(row.get("patch", ""))
    try:
        x_str, y_str = patch.split(",", 1)
        return band, int(x_str), int(y_str)
    except Exception:
        return band, 999, 999


def write_dict_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    preferred = [
        "band",
        "patch",
        "status",
        "path",
        "patch_score",
        "patch_score_percent",
        "tile_count",
        "tile_score_mean",
        "tile_score_p50",
        "tile_score_p90",
        "tile_score_p95",
        "tile_score_max",
        "tile_gt_threshold_count",
        "tile_gt_threshold_fraction",
        "tile_gt_20_count",
        "tile_gt_50_count",
        "x0",
        "y0",
        "x1",
        "y1",
        "score",
        "score_percent",
        "category",
    ]
    for key in preferred:
        fields.append(key)
        seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot_histograms(
    patch_rows: list[dict[str, object]],
    tile_rows: list[dict[str, object]],
    bands: list[str],
    out_dir: Path,
    *,
    threshold: float,
) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(bands), figsize=(4.0 * len(bands), 3.4), dpi=180, sharey=True)
    if len(bands) == 1:
        axes = [axes]
    bins = np.arange(0.0, 31.0, 1.0)
    for ax, band in zip(axes, bands):
        values = [
            float(row["patch_score_percent"])
            for row in patch_rows
            if row.get("band") == band and row.get("status") == "ok" and row.get("patch_score_percent") != ""
        ]
        ax.hist(values, bins=bins, color="#4c78a8", edgecolor="white")
        ax.axvline(threshold * 100.0, color="red", linestyle="--", linewidth=1.2, label=f"{threshold*100:.0f}%")
        ax.set_title(f"{band} patch score")
        ax.set_xlabel("score (%)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("patch count")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "broadband_patch_score_histograms.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(bands), figsize=(4.0 * len(bands), 3.4), dpi=180, sharey=True)
    if len(bands) == 1:
        axes = [axes]
    bins = np.arange(0.0, 101.0, 2.0)
    for ax, band in zip(axes, bands):
        values = [float(row["score_percent"]) for row in tile_rows if row.get("band") == band]
        ax.hist(values, bins=bins, color="#f58518", edgecolor="white")
        ax.axvline(threshold * 100.0, color="red", linestyle="--", linewidth=1.2, label=f"{threshold*100:.0f}%")
        ax.set_title(f"{band} tile score")
        ax.set_xlabel("score (%)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("tile count")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "broadband_tile_score_histograms.png")
    plt.close(fig)


def write_top_overlays(
    top_rows: list[dict[str, object]],
    *,
    data_root: Path,
    tract: str,
    weights: dict[str, float],
    tile_size: int,
    stride: int,
    out_dir: Path,
    downsample: int,
    label_tiles: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(top_rows, start=1):
        band = str(row["band"])
        patch = str(row["patch"])
        calexp = find_calexp(data_root / str(tract) / band / patch)
        if calexp is None:
            continue
        image, mask, planes = read_calexp(calexp)
        tiles = score_tiles(mask, planes, weights, int(tile_size), int(stride))
        safe_patch = patch.replace(",", "_")
        out_path = out_dir / f"rank{rank:02d}_{band}_{tract}_{safe_patch}_score{float(row['patch_score_percent']):.1f}_overlay.png"
        plot_overlay(
            image,
            tiles,
            out_path,
            title=f"rank {rank} | {band} {tract} {patch} | patch score={float(row['patch_score_percent']):.2f}%",
            downsample=max(1, int(downsample)),
            label_tiles=label_tiles,
        )
        print(f"top {rank}: {band} {patch} score={float(row['patch_score_percent']):.2f}% -> {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Broadband patch/tile bad-score experiment.")
    parser.add_argument("--data-root", type=Path, default=Path("/data/shared/Subaru"))
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--bands", nargs="+", default=["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"])
    parser.add_argument("--patches", nargs="+", default=["all"])
    parser.add_argument("--out-dir", type=Path, default=Path("output/broadband_bad_score_experiment"))
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=368)
    parser.add_argument("--threshold", type=float, default=0.11)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--label-tiles", action="store_true")
    parser.add_argument("--bad-score-weights", nargs="*", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bands = [normalize_band_dir(band) for band in args.bands]
    patches = parse_patches(args.patches)
    weights = parse_score_weights(args.bad_score_weights)
    tasks = [
        {
            "data_root": str(args.data_root),
            "tract": str(args.tract),
            "band": band,
            "patch": patch,
            "weights": weights,
            "tile_size": int(args.tile_size),
            "stride": int(args.stride),
            "threshold": float(args.threshold),
        }
        for band in bands
        for patch in patches
    ]
    patch_rows: list[dict[str, object]] = []
    tile_rows: list[dict[str, object]] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        iterator = ((index, analyze_patch_task(**task)) for index, task in enumerate(tasks, start=1))
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        futures = [executor.submit(analyze_patch_task, **task) for task in tasks]
        iterator = ((index, future.result()) for index, future in enumerate(as_completed(futures), start=1))

    try:
        for index, (patch_row, patch_tile_rows) in iterator:
            patch_rows.append(patch_row)
            tile_rows.extend(patch_tile_rows)
            if patch_row.get("status") == "ok":
                print(
                    f"[{index}/{len(tasks)}] {patch_row['band']} {patch_row['patch']}: "
                    f"patch={float(patch_row['patch_score_percent']):.2f}% "
                    f"tiles>{float(args.threshold)*100:.0f}%={patch_row['tile_gt_threshold_count']}",
                    flush=True,
                )
            else:
                print(f"[{index}/{len(tasks)}] {patch_row['band']} {patch_row['patch']}: {patch_row['status']}", flush=True)
    finally:
        if workers != 1:
            executor.shutdown(wait=True, cancel_futures=False)  # type: ignore[possibly-undefined]

    patch_rows.sort(key=patch_sort_key)
    tile_rows.sort(key=patch_sort_key)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_dict_csv(args.out_dir / "broadband_patch_scores.csv", patch_rows)
    write_dict_csv(args.out_dir / "broadband_tile_scores.csv", tile_rows)
    plot_histograms(patch_rows, tile_rows, bands, args.out_dir, threshold=float(args.threshold))

    valid_rows = [row for row in patch_rows if row.get("status") == "ok" and row.get("patch_score") != ""]
    top_rows = sorted(valid_rows, key=lambda row: float(row["patch_score"]), reverse=True)[: max(0, int(args.top_k))]
    write_dict_csv(args.out_dir / f"top{int(args.top_k)}_patch_scores.csv", top_rows)
    write_top_overlays(
        top_rows,
        data_root=args.data_root,
        tract=str(args.tract),
        weights=weights,
        tile_size=int(args.tile_size),
        stride=int(args.stride),
        out_dir=args.out_dir / "top_overlays",
        downsample=int(args.downsample),
        label_tiles=bool(args.label_tiles),
    )
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
