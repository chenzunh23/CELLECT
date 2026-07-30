#!/usr/bin/env python3
"""Summarize LSST/HSC calexp mask-plane quality by patch."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy import ndimage

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_filtering.calexp_quality import (  # noqa: E402
    DEFAULT_BAD_PLANES,
    DEFAULT_BAD_SCORE_WEIGHTS,
    bad_score_map,
    find_calexp,
    mask_planes_from_header,
    normalize_band_dir,
    parse_patches,
    parse_score_weights,
)


def _sample_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    factor = max(1, int(factor))
    if factor <= 1:
        return mask
    return mask[::factor, ::factor]


def largest_component_fraction(mask: np.ndarray, *, sample_factor: int) -> float:
    mask = _sample_mask(mask, sample_factor)
    if not bool(mask.any()):
        return 0.0
    labels, count = ndimage.label(mask)
    if count <= 0:
        return 0.0
    sizes = np.bincount(labels.ravel())
    if sizes.size <= 1:
        return 0.0
    return float(sizes[1:].max() / mask.size)


def edge_touch_fraction(mask: np.ndarray, *, sample_factor: int) -> float:
    mask = _sample_mask(mask, sample_factor)
    if not bool(mask.any()):
        return 0.0
    edge = np.zeros_like(mask, dtype=bool)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True
    grown = ndimage.binary_dilation(edge, iterations=max(1, int(round(64 / max(1, sample_factor)))))
    return float((mask & grown).sum() / mask.size)


def summarize_calexp(
    path: Path,
    bad_planes: tuple[str, ...],
    bad_score_weights: dict[str, float],
    *,
    component_downsample: int,
    include_image_stats: bool,
) -> dict[str, object]:
    from astropy.io import fits

    with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
        mask = np.asarray(hdul[2].data, dtype=np.int64)
        planes = mask_planes_from_header(hdul[2].header)
        image = np.asarray(hdul[1].data, dtype=np.float32) if include_image_stats else None
        variance = np.asarray(hdul[3].data, dtype=np.float32) if include_image_stats else None

    total = int(mask.size)
    row: dict[str, object] = {
        "path": str(path),
        "height": int(mask.shape[0]),
        "width": int(mask.shape[1]),
        "finite_image_fraction": "",
        "finite_variance_fraction": "",
        "positive_variance_fraction": "",
    }
    if include_image_stats and image is not None and variance is not None:
        row.update(
            {
                "finite_image_fraction": float(np.isfinite(image).sum() / max(image.size, 1)),
                "finite_variance_fraction": float(np.isfinite(variance).sum() / max(variance.size, 1)),
                "positive_variance_fraction": float((np.isfinite(variance) & (variance > 0)).sum() / max(variance.size, 1)),
            }
        )

    bad_union = np.zeros(mask.shape, dtype=bool)
    bad_score = bad_score_map(mask, planes, bad_score_weights)
    for name, bit in planes.items():
        plane_mask = (mask & (1 << int(bit))) != 0
        frac = float(plane_mask.sum() / max(total, 1))
        row[f"mask_{name.lower()}_fraction"] = frac
        if name in bad_planes:
            bad_union |= plane_mask

    row["bad_union_fraction"] = float(bad_union.sum() / max(total, 1))
    row["bad_score_fraction"] = float(np.nanmean(bad_score)) if bad_score.size else 0.0
    row["bad_union_largest_component_fraction"] = largest_component_fraction(
        bad_union,
        sample_factor=component_downsample,
    )
    row["bad_union_edge_touch_fraction"] = edge_touch_fraction(
        bad_union,
        sample_factor=component_downsample,
    )

    detected_bit = planes.get("DETECTED")
    if detected_bit is not None:
        detected = (mask & (1 << int(detected_bit))) != 0
        row["detected_fraction"] = float(detected.sum() / max(total, 1))
    else:
        row["detected_fraction"] = ""

    finite = np.isfinite(image) if image is not None else np.zeros((), dtype=bool)
    if image is not None and bool(finite.any()):
        values = image[finite]
        if values.size > 1_000_000:
            values = values[:: int(math.ceil(values.size / 1_000_000))]
        row.update(
            {
                "image_p01": float(np.nanpercentile(values, 1.0)),
                "image_p50": float(np.nanpercentile(values, 50.0)),
                "image_p99": float(np.nanpercentile(values, 99.0)),
                "image_max": float(np.nanmax(values)),
            }
        )
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    preferred = [
        "band",
        "patch",
        "status",
        "path",
        "height",
        "width",
        "bad_union_fraction",
        "bad_score_fraction",
        "bad_union_largest_component_fraction",
        "bad_union_edge_touch_fraction",
        "finite_image_fraction",
        "finite_variance_fraction",
        "positive_variance_fraction",
        "detected_fraction",
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
            writer.writerow({key: row.get(key, "") for key in fields})


def plot_heatmaps(rows: list[dict[str, object]], out_dir: Path, band: str, metrics: tuple[str, ...]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        grid = np.full((9, 9), np.nan, dtype=float)
        for row in rows:
            if row.get("status") != "ok":
                continue
            patch = str(row.get("patch", ""))
            try:
                x_str, y_str = patch.split(",", 1)
                x, y = int(x_str), int(y_str)
                value = row.get(metric)
                if value != "":
                    grid[y, x] = float(value)
            except Exception:
                continue
        fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=180)
        im = ax.imshow(grid * 100.0, origin="lower", cmap="magma", vmin=0.0, vmax=100.0)
        for y in range(9):
            for x in range(9):
                text = "missing" if not np.isfinite(grid[y, x]) else f"{grid[y, x] * 100.0:.0f}%"
                ax.text(x, y, f"{x},{y}\n{text}", ha="center", va="center", fontsize=5, color="white")
        ax.set_xlabel("patch x")
        ax.set_ylabel("patch y")
        ax.set_title(f"{band} {metric} (%)")
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
        fig.tight_layout()
        fig.savefig(out_dir / f"{band}_{metric}_heatmap.png")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze HSC calexp mask-plane quality by patch.")
    parser.add_argument("--data-root", type=Path, default=Path("/data/shared/Subaru"))
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--bands", nargs="+", required=True)
    parser.add_argument("--patches", nargs="+", default=["all"])
    parser.add_argument("--out-dir", type=Path, default=Path("output/calexp_mask_quality"))
    parser.add_argument("--bad-planes", nargs="+", default=list(DEFAULT_BAD_PLANES))
    parser.add_argument(
        "--bad-score-weights",
        nargs="*",
        default=None,
        help=(
            "Per-plane severity weights for bad_score_fraction, e.g. NO_DATA=1 INTRP=0.9 BAD=0.5. "
            f"Defaults: {' '.join(f'{key}={value:g}' for key, value in DEFAULT_BAD_SCORE_WEIGHTS.items())}."
        ),
    )
    parser.add_argument(
        "--component-downsample",
        type=int,
        default=8,
        help="Stride used for connected-component diagnostics. Area fractions are still measured at full resolution.",
    )
    parser.add_argument("--include-image-stats", action="store_true", help="Also read image/variance HDUs and write image percentiles.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Patch-level worker processes. Use 4-16 depending on disk throughput. Default: 1.",
    )
    return parser.parse_args()


def analyze_one_task(
    *,
    data_root: str,
    tract: str,
    band: str,
    patch: str,
    bad_planes: tuple[str, ...],
    bad_score_weights: dict[str, float],
    component_downsample: int,
    include_image_stats: bool,
) -> dict[str, object]:
    patch_dir = Path(data_root) / str(tract) / band / patch
    calexp = find_calexp(patch_dir)
    row: dict[str, object] = {"band": band, "patch": patch, "status": "missing", "path": ""}
    if calexp is None:
        return row
    try:
        row.update(
            summarize_calexp(
                calexp,
                bad_planes,
                bad_score_weights,
                component_downsample=int(component_downsample),
                include_image_stats=bool(include_image_stats),
            )
        )
        row.update({"band": band, "patch": patch, "status": "ok"})
    except Exception as exc:  # noqa: BLE001
        row.update({"status": "error", "path": str(calexp), "error": f"{type(exc).__name__}: {exc}"})
    return row


def patch_sort_key(row: dict[str, object]) -> tuple[str, int, int]:
    band = str(row.get("band", ""))
    patch = str(row.get("patch", ""))
    try:
        x_str, y_str = patch.split(",", 1)
        return band, int(x_str), int(y_str)
    except Exception:
        return band, 999, 999


def main() -> int:
    args = parse_args()
    patches = parse_patches(args.patches)
    bad_planes = tuple(str(name).upper() for name in args.bad_planes)
    bad_score_weights = parse_score_weights(args.bad_score_weights)
    bands = [normalize_band_dir(band_arg) for band_arg in args.bands]
    tasks = [
        {
            "data_root": str(args.data_root),
            "tract": str(args.tract),
            "band": band,
            "patch": patch,
            "bad_planes": bad_planes,
            "bad_score_weights": bad_score_weights,
            "component_downsample": int(args.component_downsample),
            "include_image_stats": bool(args.include_image_stats),
        }
        for band in bands
        for patch in patches
    ]
    all_rows: list[dict[str, object]] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for index, task in enumerate(tasks, start=1):
            row = analyze_one_task(**task)
            all_rows.append(row)
            if row.get("status") == "ok":
                print(
                    f"[{index}/{len(tasks)}] {row['band']} {row['patch']}: "
                    f"bad={float(row['bad_union_fraction']):.3f} "
                    f"score={float(row['bad_score_fraction']):.3f} "
                    f"largest={float(row['bad_union_largest_component_fraction']):.3f}",
                    flush=True,
                )
            else:
                print(f"[{index}/{len(tasks)}] {row['band']} {row['patch']}: {row['status']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(analyze_one_task, **task) for task in tasks]
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                all_rows.append(row)
                if row.get("status") == "ok":
                    print(
                        f"[{index}/{len(tasks)}] {row['band']} {row['patch']}: "
                        f"bad={float(row['bad_union_fraction']):.3f} "
                        f"score={float(row['bad_score_fraction']):.3f} "
                        f"largest={float(row['bad_union_largest_component_fraction']):.3f}",
                        flush=True,
                    )
                else:
                    message = str(row.get("error", "")) if row.get("status") == "error" else str(row.get("status", ""))
                    print(f"[{index}/{len(tasks)}] {row['band']} {row['patch']}: {message}", flush=True)

    all_rows.sort(key=patch_sort_key)
    for band in bands:
        rows = [row for row in all_rows if row.get("band") == band]
        band_out = args.out_dir / str(args.tract) / band
        write_csv(band_out / f"{band}_{args.tract}_calexp_mask_quality.csv", rows)
        plot_heatmaps(
            rows,
            band_out,
            band,
            metrics=(
                "bad_union_fraction",
                "bad_score_fraction",
                "bad_union_largest_component_fraction",
                "mask_no_data_fraction",
                "mask_intrp_fraction",
                "mask_bright_object_fraction",
                "mask_cr_fraction",
                "mask_crosstalk_fraction",
                "mask_rejected_fraction",
                "mask_sensor_edge_fraction",
                "mask_clipped_fraction",
                "mask_suspect_fraction",
                "mask_bad_fraction",
                "mask_edge_fraction",
                "mask_sat_fraction",
                "mask_unmaskednan_fraction",
            ),
        )
    write_csv(args.out_dir / str(args.tract) / f"{args.tract}_calexp_mask_quality.csv", all_rows)
    print(args.out_dir / str(args.tract))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
