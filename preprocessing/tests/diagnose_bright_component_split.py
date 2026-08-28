#!/usr/bin/env python3
"""Diagnose bright-component splitting around one full-patch tile.

Example:
    conda run -n cellect python preprocessing/tests/diagnose_bright_component_split.py \
      --coadd-fits-root /data/czh23/Subaru_products/half_coadd \
      --dataset-source coadd --patch 4,5 --band HSC-Y --tile-index 59 \
      --bright-mask-mode anscombe --bright-threshold 5 --clip-threshold 5 \
      --split-shrink 3 --out-dir output/preprocessing_tests/bright_split_59
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cellect")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ZScaleInterval
from scipy import ndimage

from astro_data_preprocessing import make_tile_specs
from data_filtering.sam_input_scaling import build_bright_mask
from preprocessing.build_image_level_zarr import _coadd_image_path, _read_image_header_origin, _variant_image_path


@dataclass(frozen=True)
class ComponentDiagnostic:
    component_id: int
    area: int
    bbox_y0: int
    bbox_y1: int
    bbox_x0: int
    bbox_x1: int
    eroded_area: int
    stability: float
    seed_count: int
    valid_seed_count: int
    invalid_seed_count: int
    split_count: int
    reason: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, default=None, help="Direct FITS path. Overrides root-based resolution.")
    p.add_argument("--data-root", type=Path, default=Path("/data/shared/Subaru"))
    p.add_argument("--coadd-fits-root", type=Path, default=Path("/data/czh23/Subaru_products/half_coadd"))
    p.add_argument("--denoised-fits-root", type=Path, default=Path("/data/czh23/Subaru_products/noisy"))
    p.add_argument("--dataset-source", default="coadd", choices=("coadd", "noisy", "denoised"))
    p.add_argument("--group", default="0")
    p.add_argument("--tract", type=int, default=9813)
    p.add_argument("--patch", default="4,5")
    p.add_argument("--band", default="HSC-Y")
    p.add_argument("--hdu", type=int, default=1)
    p.add_argument("--tile-index", type=int, default=59, help="0-based tile index from make_tile_specs.")
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--stride", type=int, default=368)
    p.add_argument("--context", type=int, default=96, help="Extra pixels around the selected tile in diagnostic crops.")
    p.add_argument("--bright-mask-mode", default="anscombe")
    p.add_argument("--bright-threshold", type=float, default=5.0)
    p.add_argument("--bright-dilation", type=int, default=2)
    p.add_argument("--clip-threshold", type=float, default=5.0)
    p.add_argument("--log-a", type=float, default=1000.0)
    p.add_argument("--log-high-percentile", type=float, default=99.5)
    p.add_argument("--lupton-stretch", type=float, default=0.5)
    p.add_argument("--lupton-q", type=float, default=20.0)
    p.add_argument("--anscombe-scale", type=float, default=1000.0)
    p.add_argument("--split-shrink", type=int, default=3)
    p.add_argument("--split-stability-min", type=float, default=0.2)
    p.add_argument("--split-min-seed-area", type=int, default=4)
    p.add_argument("--out-dir", type=Path, default=Path("output/preprocessing_tests/bright_component_split"))
    return p.parse_args()


def _resolve_image(args: argparse.Namespace) -> Path:
    if args.image is not None:
        return args.image.expanduser().resolve()
    if args.dataset_source == "coadd":
        return _coadd_image_path(args.coadd_fits_root.expanduser().resolve(), args.band, int(args.tract), args.patch)
    return _variant_image_path(
        args.denoised_fits_root.expanduser().resolve(),
        args.patch,
        str(args.group),
        args.band,
        args.dataset_source,
        int(args.tract),
    )


def _zscale(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.float32)
    try:
        vmin, vmax = ZScaleInterval().get_limits(finite)
    except Exception:
        vmin, vmax = np.nanpercentile(finite, [1.0, 99.0])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = np.nanpercentile(finite, [1.0, 99.0])
    return np.clip((np.nan_to_num(arr, nan=vmin) - float(vmin)) / max(float(vmax - vmin), 1e-6), 0.0, 1.0)


def _component_slices(labels: np.ndarray) -> dict[int, tuple[slice, slice]]:
    return {idx: slc for idx, slc in enumerate(ndimage.find_objects(labels), start=1) if slc is not None}


def _bbox_intersects_tile(slc: tuple[slice, slice], tile_bounds: tuple[int, int, int, int]) -> bool:
    y0, y1 = int(slc[0].start), int(slc[0].stop)
    x0, x1 = int(slc[1].start), int(slc[1].stop)
    tx0, ty0, tx1, ty1 = tile_bounds
    return x1 > tx0 and x0 < tx1 and y1 > ty0 and y0 < ty1


def _run_split_diagnostic(
    labels: np.ndarray,
    *,
    shrink: int,
    stability_min: float,
    min_seed_area: int,
    keep_components: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[ComponentDiagnostic], list[dict[str, object]]]:
    labels = np.asarray(labels, dtype=np.int32)
    structure = ndimage.generate_binary_structure(2, 1)
    split = np.zeros(labels.shape, dtype=np.int32)
    valid_seed_mask = np.zeros(labels.shape, dtype=np.int32)
    invalid_seed_mask = np.zeros(labels.shape, dtype=np.int32)
    diagnostics: list[ComponentDiagnostic] = []
    seed_rows: list[dict[str, object]] = []
    next_label = 1
    min_seed_area = max(1, int(min_seed_area))

    for comp, slc in _component_slices(labels).items():
        comp_mask = labels[slc] == comp
        area = int(np.count_nonzero(comp_mask))
        if area <= 0:
            continue
        eroded = ndimage.binary_erosion(comp_mask, structure=structure, iterations=int(shrink), border_value=0) if shrink > 0 else comp_mask.copy()
        eroded_area = int(np.count_nonzero(eroded))
        stability = float(eroded_area) / float(area) if area > 0 else 0.0
        seed_labels, seed_count = ndimage.label(eroded, structure=structure)
        seed_areas = np.bincount(seed_labels.ravel(), minlength=seed_count + 1)
        valid = np.zeros(seed_count + 1, dtype=bool)
        if seed_count:
            valid[1:] = seed_areas[1:] >= min_seed_area
        valid_count = int(np.count_nonzero(valid))
        invalid_count = int(seed_count - valid_count)

        yy, xx = np.indices(comp_mask.shape)
        for seed in range(1, seed_count + 1):
            mask = seed_labels == seed
            if not bool(np.any(mask)):
                continue
            ys = yy[mask] + int(slc[0].start)
            xs = xx[mask] + int(slc[1].start)
            is_valid = bool(valid[seed])
            row = {
                "component_id": comp,
                "seed_id": seed,
                "seed_area": int(seed_areas[seed]),
                "valid_seed": is_valid,
                "centroid_x": float(np.mean(xs)),
                "centroid_y": float(np.mean(ys)),
                "bbox_x0": int(np.min(xs)),
                "bbox_x1": int(np.max(xs) + 1),
                "bbox_y0": int(np.min(ys)),
                "bbox_y1": int(np.max(ys) + 1),
            }
            seed_rows.append(row)
            target = valid_seed_mask if is_valid else invalid_seed_mask
            target_view = target[slc]
            target_view[mask] = int(seed)

        split_this = seed_count > 1 and eroded_area > 0 and stability >= float(stability_min) and valid_count > 1
        if split_this:
            valid_seed_labels = np.where(valid[seed_labels], seed_labels, 0).astype(np.int32, copy=False)
            seed_mask = valid_seed_labels > 0
            nearest = ndimage.distance_transform_edt(~seed_mask, return_distances=False, return_indices=True)
            assigned = valid_seed_labels[tuple(nearest)]
            split_count = 0
            split_view = split[slc]
            for seed in np.flatnonzero(valid):
                part = comp_mask & (assigned == int(seed))
                if bool(np.any(part)):
                    split_view[part] = next_label
                    next_label += 1
                    split_count += 1
            reason = "split"
        else:
            split_view = split[slc]
            split_view[comp_mask] = next_label
            next_label += 1
            split_count = 1
            if seed_count <= 1:
                reason = "not_split_seed_count_le1"
            elif eroded_area <= 0:
                reason = "not_split_empty_after_erosion"
            elif stability < float(stability_min):
                reason = "not_split_low_stability"
            elif valid_count <= 1:
                reason = "not_split_valid_seed_count_le1"
            else:
                reason = "not_split"

        if keep_components is None or comp in keep_components:
            diagnostics.append(
                ComponentDiagnostic(
                    component_id=int(comp),
                    area=area,
                    bbox_y0=int(slc[0].start),
                    bbox_y1=int(slc[0].stop),
                    bbox_x0=int(slc[1].start),
                    bbox_x1=int(slc[1].stop),
                    eroded_area=eroded_area,
                    stability=stability,
                    seed_count=int(seed_count),
                    valid_seed_count=valid_count,
                    invalid_seed_count=invalid_count,
                    split_count=int(split_count),
                    reason=reason,
                )
            )

    if keep_components is not None:
        seed_rows = [row for row in seed_rows if int(row["component_id"]) in keep_components]
    return split, valid_seed_mask, invalid_seed_mask, diagnostics, seed_rows


def _crop_bounds(shape: tuple[int, int], tile_bounds: tuple[int, int, int, int], context: int) -> tuple[int, int, int, int]:
    tx0, ty0, tx1, ty1 = tile_bounds
    h, w = shape
    x0 = max(0, int(tx0) - int(context))
    y0 = max(0, int(ty0) - int(context))
    x1 = min(w, int(tx1) + int(context))
    y1 = min(h, int(ty1) + int(context))
    return x0, y0, x1, y1


def _draw_contours(ax: plt.Axes, mask: np.ndarray, *, color: str, linewidth: float, label: str) -> None:
    if bool(np.any(mask)):
        ax.contour(mask.astype(np.uint8), levels=[0.5], colors=[color], linewidths=linewidth)
        ax.plot([], [], color=color, linewidth=linewidth, label=label)


def _annotate_labels(ax: plt.Axes, labels: np.ndarray, *, color: str, prefix: str = "", min_area: int = 1) -> None:
    labels = np.asarray(labels, dtype=np.int32)
    ids = [int(value) for value in np.unique(labels) if int(value) > 0]
    for label_id in ids:
        mask = labels == label_id
        area = int(np.count_nonzero(mask))
        if area < int(min_area):
            continue
        cy, cx = ndimage.center_of_mass(mask)
        if not (math.isfinite(cx) and math.isfinite(cy)):
            continue
        ax.text(
            float(cx),
            float(cy),
            f"{prefix}{label_id}",
            color=color,
            fontsize=7,
            ha="center",
            va="center",
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 1.0},
        )


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    image_path = _resolve_image(args)
    image, _header, origin = _read_image_header_origin(image_path, hdu=args.hdu)
    specs = make_tile_specs(
        parent_origin=origin,
        image_shape=(int(image.shape[1]), int(image.shape[0])),
        tile_size=int(args.tile_size),
        stride=int(args.stride),
        compare_origin=None,
    )
    if int(args.tile_index) < 0 or int(args.tile_index) >= len(specs):
        raise IndexError(f"tile index {args.tile_index} outside [0, {len(specs) - 1}]")
    spec = specs[int(args.tile_index)]
    tile_x0 = int(spec.x0) - int(origin[0])
    tile_y0 = int(spec.y0) - int(origin[1])
    tile_bounds = (tile_x0, tile_y0, tile_x0 + int(spec.size), tile_y0 + int(spec.size))

    bright = build_bright_mask(
        image,
        mode=str(args.bright_mask_mode),
        threshold=float(args.bright_threshold),
        clip_threshold=float(args.clip_threshold),
        dilation=int(args.bright_dilation),
        log_a=float(args.log_a),
        log_high_percentile=float(args.log_high_percentile),
        lupton_stretch=float(args.lupton_stretch),
        lupton_q=float(args.lupton_q),
        anscombe_clip=False,
        anscombe_scale=float(args.anscombe_scale),
    )
    raw_labels, _ = ndimage.label(np.asarray(bright, dtype=bool))
    slices = _component_slices(raw_labels)
    keep_components = {comp for comp, slc in slices.items() if _bbox_intersects_tile(slc, tile_bounds)}
    split, valid_seed_mask, invalid_seed_mask, diagnostics, seed_rows = _run_split_diagnostic(
        raw_labels,
        shrink=int(args.split_shrink),
        stability_min=float(args.split_stability_min),
        min_seed_area=int(args.split_min_seed_area),
        keep_components=keep_components,
    )

    x0, y0, x1, y1 = _crop_bounds(image.shape, tile_bounds, int(args.context))
    slc = np.s_[y0:y1, x0:x1]
    display = _zscale(image[slc])
    raw_crop = raw_labels[slc]
    split_crop = split[slc]
    valid_crop = valid_seed_mask[slc]
    invalid_crop = invalid_seed_mask[slc]
    tile_rect = (
        tile_bounds[0] - x0,
        tile_bounds[1] - y0,
        int(args.tile_size),
        int(args.tile_size),
    )

    out_dir = args.out_dir.expanduser().resolve()
    stem = (
        f"{args.dataset_source}_{args.patch.replace(',', '_')}_{args.band}_"
        f"tile{int(args.tile_index):03d}_shrink{int(args.split_shrink)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14), constrained_layout=True)
    panels = [
        ("raw bright components", raw_crop, "yellow"),
        ("split components", split_crop, "lime"),
        ("valid erosion seeds", valid_crop, "cyan"),
        ("invalid/small erosion seeds", invalid_crop, "red"),
    ]
    for ax, (title, arr, color) in zip(axes.ravel(), panels):
        ax.imshow(display, origin="lower", cmap="gray", interpolation="nearest")
        _draw_contours(ax, arr > 0, color=color, linewidth=1.0, label=title)
        if title == "raw bright components":
            _annotate_labels(ax, raw_crop, color="yellow", prefix="c", min_area=1)
        elif title == "split components":
            _annotate_labels(ax, split_crop, color="lime", prefix="s", min_area=1)
        elif title == "valid erosion seeds":
            _annotate_labels(ax, valid_crop, color="cyan", prefix="v", min_area=1)
        elif title == "invalid/small erosion seeds":
            _annotate_labels(ax, invalid_crop, color="red", prefix="x", min_area=1)
        rect = plt.Rectangle((tile_rect[0], tile_rect[1]), tile_rect[2], tile_rect[3], fill=False, edgecolor="white", linewidth=1.0)
        ax.add_patch(rect)
        ax.set_title(title)
        ax.set_xlim(0, display.shape[1] - 1)
        ax.set_ylim(0, display.shape[0] - 1)
        ax.legend(loc="upper right", fontsize=8)
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    comp_rows = [diag.__dict__ for diag in diagnostics]
    comp_csv = out_dir / f"{stem}_components.csv"
    seed_csv = out_dir / f"{stem}_seeds.csv"
    _write_csv(
        comp_csv,
        comp_rows,
        [
            "component_id",
            "area",
            "bbox_y0",
            "bbox_y1",
            "bbox_x0",
            "bbox_x1",
            "eroded_area",
            "stability",
            "seed_count",
            "valid_seed_count",
            "invalid_seed_count",
            "split_count",
            "reason",
        ],
    )
    _write_csv(
        seed_csv,
        seed_rows,
        ["component_id", "seed_id", "seed_area", "valid_seed", "centroid_x", "centroid_y", "bbox_x0", "bbox_x1", "bbox_y0", "bbox_y1"],
    )
    npz_path = out_dir / f"{stem}_masks.npz"
    np.savez_compressed(
        npz_path,
        crop_bounds=np.asarray([x0, y0, x1, y1], dtype=np.int32),
        tile_bounds=np.asarray(tile_bounds, dtype=np.int32),
        raw_labels=raw_crop.astype(np.int32),
        split_labels=split_crop.astype(np.int32),
        valid_seed_mask=valid_crop.astype(np.int32),
        invalid_seed_mask=invalid_crop.astype(np.int32),
    )
    summary = {
        "image": str(image_path),
        "dataset_source": str(args.dataset_source),
        "group": str(args.group),
        "tract": int(args.tract),
        "patch": str(args.patch),
        "band": str(args.band),
        "origin": [int(origin[0]), int(origin[1])],
        "tile_index": int(args.tile_index),
        "tile_name": str(spec.name),
        "tile_bounds_local_xyxy": list(map(int, tile_bounds)),
        "crop_bounds_local_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "bright_mask_mode": str(args.bright_mask_mode),
        "bright_threshold": float(args.bright_threshold),
        "bright_dilation": int(args.bright_dilation),
        "clip_threshold": float(args.clip_threshold),
        "split_shrink": int(args.split_shrink),
        "split_stability_min": float(args.split_stability_min),
        "split_min_seed_area": int(args.split_min_seed_area),
        "raw_components_in_tile": len(keep_components),
        "diagnostic_components": len(comp_rows),
        "seed_rows": len(seed_rows),
        "invalid_seed_rows": int(sum(1 for row in seed_rows if not bool(row["valid_seed"]))),
        "outputs": {
            "png": str(png_path),
            "components_csv": str(comp_csv),
            "seeds_csv": str(seed_csv),
            "masks_npz": str(npz_path),
        },
    }
    summary_path = out_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
