#!/usr/bin/env python3
"""Visualize direct-Zarr CELLECT labels for one tile/sample.

This script does not re-run filtering. It reads the training products already
stored in zarr and visualizes exactly what training sees: PU regions, GT
confidence map, source centers, and shape-supervised ellipses.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_utils import (
    CONF_COLORS,
    PU_CLASS_NAMES,
    PU_COLORS,
    centers_from_zarr,
    confidence_overlay,
    crop_or_pad,
    draw_ellipses,
    draw_points,
    input_channel_display_limits,
    label_mask_overlay,
    normalize_group_name,
    read_fits_image,
    read_zarr_sample,
    resolve_zarr_sample,
    rows_to_reg,
    save_heatmap,
    save_pixel_png,
    save_png,
    source_rows_from_zarr,
    strict_centers_from_zarr,
    tile_name_matches,
    write_reg,
    zarr_sample_group,
    zscale_rgb,
)
from astro_train_zarr_data import PatchZarrReader, discover_zarr_image_records


EXTERNAL_SOURCE_COLOR = "#a020f0"
EXTERNAL_SOURCE_REG_COLOR = "magenta"
SOURCE_CLASS_LABELS = {
    1: "clean",
    2: "weak_shape",
    4: "strict_center_only",
    5: "strict_center_only",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_argument_group("zarr selection")
    src.add_argument("--zarr-store", type=Path, default=None, help="Direct path to one .zarr store.")
    src.add_argument("--sample-index", type=int, default=0, help="Sample index within --zarr-store, or match index under --root.")
    src.add_argument("--root", type=Path, default=None, help="Direct-zarr root used when --zarr-store is not given.")
    src.add_argument("--patch", default=None)
    src.add_argument("--tile-name", default=None)
    src.add_argument("--band", default=None, help="Band to visualize. Defaults to the first band in the zarr store.")
    src.add_argument(
        "--dataset-source",
        nargs="*",
        default=None,
        help="Dataset source(s): coadd, noisy, denoised. Use 'all' for coadd noisy denoised.",
    )
    src.add_argument("--group", default=None, help="Variant group such as group_00. Numeric values like 0 become group_00.")
    src.add_argument("--image-level", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--fits-image", type=Path, default=None, help="Optional raw FITS background instead of zarr input channel.")
    p.add_argument("--fits-hdu", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("output/eval_visualizations/labels"))
    p.add_argument("--png-scale", type=int, default=1)
    p.add_argument(
        "--confidence-negative-overlay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also visualize pixels used as confidence negatives: confidence == 0 and confidence_weight > 0.",
    )
    p.add_argument(
        "--confidence-negative-clear-overlay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Visualize confidence negatives by leaving them clear and covering all other pixels.",
    )
    p.add_argument(
        "--all-samples",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="With --confidence-negative-overlay and --root, write one confidence-negative PNG for every matching sample.",
    )
    p.add_argument("--negative-alpha", type=float, default=0.70)
    return p.parse_args()


def _dataset_sources(values: list[str] | None) -> list[str | None]:
    if not values:
        return [None]
    normalized = [str(value).strip() for value in values if str(value).strip()]
    if any(value.lower() == "all" for value in normalized):
        return ["coadd", "noisy", "denoised"]
    return normalized


def _source_color(row: dict[str, float]) -> str:
    class_id = int(row.get("class_id", 1))
    if class_id == 1:
        return "green"
    if class_id == 2:
        return "cyan"
    if class_id in {4, 5}:
        return EXTERNAL_SOURCE_COLOR
    return "yellow"


def _draw_external_centers(ax, centers: list[dict[str, float]], *, label: str | None = None) -> None:
    if not centers:
        return
    xs = [float(row["x"]) for row in centers]
    ys = [float(row["y"]) for row in centers]
    ax.scatter(
        xs,
        ys,
        marker="o",
        s=54,
        facecolors="none",
        edgecolors=EXTERNAL_SOURCE_COLOR,
        linewidths=1.1,
        label=label,
        zorder=5,
    )
    ax.scatter(
        xs,
        ys,
        marker="+",
        s=34,
        c=EXTERNAL_SOURCE_COLOR,
        linewidths=1.0,
        zorder=6,
    )


def _draw_source_legend(ax, sources: list[dict[str, float]], external_centers: list[dict[str, float]]) -> None:
    handles = []
    present = sorted({int(row.get("class_id", 1)) for row in sources})
    for class_id in present:
        handles.append(
            mpatches.Patch(
                color=_source_color({"class_id": class_id}),
                label=SOURCE_CLASS_LABELS.get(class_id, PU_CLASS_NAMES.get(class_id, f"class_{class_id}")),
            )
        )
    if external_centers:
        handles.append(mpatches.Patch(color=EXTERNAL_SOURCE_COLOR, label="strict center point"))
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.75)


def _source_shapes_overlay(
    image: np.ndarray,
    sources: list[dict[str, float]],
    external_centers: list[dict[str, float]],
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Ellipse

    fig, ax = plt.subplots(figsize=(image.shape[1] / 100.0, image.shape[0] / 100.0), dpi=100)
    ax.imshow(zscale_rgb(image), origin="lower", interpolation="nearest")
    for row in sorted(sources, key=lambda r: abs(float(r["major"]) * float(r["minor"])), reverse=True):
        color = _source_color(row)
        ax.add_patch(
            Ellipse(
                (float(row["x"]), float(row["y"])),
                width=2.0 * max(abs(float(row["major"])), 1.0),
                height=2.0 * max(abs(float(row["minor"])), 1.0),
                angle=np.degrees(float(row.get("theta", 0.0))),
                fill=False,
                edgecolor=color,
                linewidth=0.9,
                alpha=0.95,
            )
        )
    _draw_external_centers(ax, external_centers)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    plt.close(fig)
    return np.flipud(rgba[..., :3]).astype(np.float32) / 255.0


def _confidence_negative_mask(sample: dict[str, np.ndarray]) -> np.ndarray:
    confidence = np.asarray(sample["confidence"], dtype=np.uint8)
    weight = np.asarray(sample["confidence_weight"], dtype=np.float32)
    return (confidence == 0) & np.isfinite(weight) & (weight > 0.0)


def _confidence_negative_overlay(image: np.ndarray, negative: np.ndarray, *, alpha: float) -> np.ndarray:
    rgb = zscale_rgb(image)
    mask = np.asarray(negative, dtype=bool)
    color = np.asarray((1.0, 0.58, 0.0), dtype=np.float32)
    if bool(mask.any()):
        rgb[mask] = (1.0 - float(alpha)) * rgb[mask] + float(alpha) * color
    return rgb


def _confidence_negative_clear_overlay(image: np.ndarray, negative: np.ndarray, *, alpha: float) -> np.ndarray:
    rgb = zscale_rgb(image)
    cover = ~np.asarray(negative, dtype=bool)
    color = np.asarray((0.10, 0.36, 0.90), dtype=np.float32)
    if bool(cover.any()):
        rgb[cover] = (1.0 - float(alpha)) * rgb[cover] + float(alpha) * color
    return rgb


def _save_confidence_negative_plot(
    path: Path,
    image: np.ndarray,
    sample: dict[str, np.ndarray],
    *,
    title: str,
    alpha: float,
    clear_negative: bool = False,
) -> None:
    negative = _confidence_negative_mask(sample)
    overlay = (
        _confidence_negative_clear_overlay(image, negative, alpha=alpha)
        if bool(clear_negative)
        else _confidence_negative_overlay(image, negative, alpha=alpha)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=150)
    ax.imshow(overlay, origin="lower", interpolation="nearest")
    total = int(negative.size)
    count = int(np.count_nonzero(negative))
    frac = 100.0 * float(count) / max(total, 1)
    mode = "negative clear; others covered" if bool(clear_negative) else "negative overlay"
    ax.set_title(f"{title}\n{mode}: {count}/{total} ({frac:.1f}%)", fontsize=9)
    patch = mpatches.Patch(
        color=(0.10, 0.36, 0.90) if bool(clear_negative) else (1.0, 0.58, 0.0),
        label="non-negative covered" if bool(clear_negative) else "confidence negative",
    )
    ax.legend(handles=[patch], loc="lower right", fontsize=7, framealpha=0.75)
    ax.set_axis_off()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _parse_zarr_uri(uri: str) -> tuple[Path, int]:
    if not str(uri).startswith("zarr://") or "#" not in str(uri):
        raise ValueError(f"invalid zarr URI: {uri}")
    store_s, idx_s = str(uri)[len("zarr://") :].rsplit("#", 1)
    return Path(store_s), int(idx_s)


def _safe_stem(text: object) -> str:
    return str(text).replace(",", "_").replace("/", "_").replace(" ", "_")


def _run_all_confidence_negative_overlays(args: argparse.Namespace, dataset_source: str | None) -> list[Path]:
    if args.root is None:
        raise ValueError("--all-samples requires --root")
    if not args.band:
        raise ValueError("--all-samples requires --band")
    effective_group = None if dataset_source == "coadd" else normalize_group_name(args.group)
    records = discover_zarr_image_records(args.root.expanduser().resolve(), bands=[args.band])
    outputs: list[Path] = []
    readers: dict[Path, PatchZarrReader] = {}
    for rec in records:
        rec_patch = str(rec.patch).split("__", 1)[0]
        if args.patch and rec.patch != args.patch and rec_patch != args.patch:
            continue
        if dataset_source and rec.dataset_source != dataset_source:
            continue
        if args.tile_name and not tile_name_matches(rec.tile_name, args.tile_name):
            continue
        store, sample_idx = _parse_zarr_uri(rec.image_paths[0])
        reader = readers.get(store)
        if reader is None:
            reader = PatchZarrReader(store)
            readers[store] = reader
        group_name = zarr_sample_group(reader, sample_idx)
        if effective_group is not None and group_name != effective_group:
            continue
        attrs = dict(reader.attrs)
        bands = list(attrs.get("bands", []))
        band_idx = bands.index(args.band) if args.band in bands else 0
        sample = read_zarr_sample(reader, sample_idx, band_idx)
        image = np.asarray(sample["display_image"], dtype=np.float32)
        dataset = str(rec.dataset_source or attrs.get("dataset_source", dataset_source or "zarr"))
        patch_name = str(rec.patch or attrs.get("patch", args.patch or store.stem)).split("__", 1)[0]
        group_part = f"_{group_name}" if group_name else ""
        out_dir = args.out_dir.expanduser().resolve() / "confidence_negative_samples" / patch_name / str(args.band)
        stem = (
            f"{_safe_stem(dataset)}{_safe_stem(group_part)}_"
            f"{_safe_stem(patch_name)}_{_safe_stem(args.band)}_"
            f"{_safe_stem(rec.tile_name)}_sample{sample_idx:05d}"
        )
        suffix = "confidence_negative_clear_overlay" if bool(args.confidence_negative_clear_overlay) else "confidence_negative_overlay"
        out_path = out_dir / f"{stem}_{suffix}.png"
        title = f"{dataset}{group_part} {patch_name} {args.band} {rec.tile_name}"
        _save_confidence_negative_plot(
            out_path,
            image,
            sample,
            title=title,
            alpha=float(args.negative_alpha),
            clear_negative=bool(args.confidence_negative_clear_overlay),
        )
        outputs.append(out_path)
    if not outputs:
        raise RuntimeError("no matching samples for confidence-negative visualization")
    return outputs


def _panel(
    path: Path,
    image: np.ndarray,
    pu: np.ndarray,
    conf: np.ndarray,
    sources: list[dict[str, float]],
    external_centers: list[dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    axes[0].imshow(label_mask_overlay(image, pu), origin="lower", interpolation="nearest")
    axes[0].set_title("PU regions")
    axes[1].imshow(confidence_overlay(image, conf), origin="lower", interpolation="nearest")
    axes[1].set_title("GT confidence")
    axes[2].imshow(zscale_rgb(image), origin="lower", interpolation="nearest")
    axes[2].set_title("source ellipses")
    for row in sorted(sources, key=lambda r: abs(float(r["major"]) * float(r["minor"])), reverse=True):
        from matplotlib.patches import Ellipse

        color = _source_color(row)
        axes[2].add_patch(
            Ellipse(
                (float(row["x"]), float(row["y"])),
                width=2.0 * max(abs(float(row["major"])), 1.0),
                height=2.0 * max(abs(float(row["minor"])), 1.0),
                angle=np.degrees(float(row.get("theta", 0.0))),
                fill=False,
                edgecolor=color,
                linewidth=0.9,
                alpha=0.95,
            )
        )
        axes[2].plot(float(row["x"]), float(row["y"]), marker="+", color=color, ms=3.0, mew=0.8)
    _draw_external_centers(axes[2], external_centers, label="external strict center")
    for ax in axes:
        ax.set_axis_off()
    pu_patches = [mpatches.Patch(color=PU_COLORS[k], label=PU_CLASS_NAMES[k]) for k in (1, 2, 5, 6, 7, 3, 4)]
    conf_patches = [mpatches.Patch(color=CONF_COLORS[k], label=f"conf {k}") for k in (1, 2, 3, 4)]
    axes[0].legend(handles=pu_patches, loc="lower right", fontsize=7, framealpha=0.75)
    axes[1].legend(handles=conf_patches, loc="lower right", fontsize=7, framealpha=0.75)
    _draw_source_legend(axes[2], sources, external_centers)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _write_counts_csv(
    path: Path,
    *,
    pu: np.ndarray,
    conf: np.ndarray,
    sources: list[dict[str, float]],
    centers: list[dict[str, float]],
    strict_centers: list[dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dense_values, dense_counts = np.unique(np.asarray(pu, dtype=np.uint8), return_counts=True)
    conf_values, conf_counts = np.unique(np.asarray(conf, dtype=np.uint8), return_counts=True)
    source_class_values, source_class_counts = (
        np.unique(np.asarray([int(row.get("class_id", 1)) for row in sources], dtype=np.uint8), return_counts=True)
        if sources
        else (np.asarray([], dtype=np.uint8), np.asarray([], dtype=np.int64))
    )
    center_class_values, center_class_counts = (
        np.unique(np.asarray([int(row.get("class_id", 1)) for row in centers], dtype=np.uint8), return_counts=True)
        if centers
        else (np.asarray([], dtype=np.uint8), np.asarray([], dtype=np.int64))
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kind", "class_id", "name", "count"])
        writer.writeheader()
        dense_map = {int(value): int(count) for value, count in zip(dense_values, dense_counts)}
        for class_id in sorted(PU_CLASS_NAMES):
            writer.writerow(
                {
                    "kind": "dense_pixels",
                    "class_id": class_id,
                    "name": PU_CLASS_NAMES[class_id],
                    "count": dense_map.get(class_id, 0),
                }
            )
        for value, count in zip(conf_values, conf_counts):
            writer.writerow({"kind": "confidence_pixels", "class_id": int(value), "name": f"conf_{int(value)}", "count": int(count)})
        for value, count in zip(source_class_values, source_class_counts):
            writer.writerow(
                {
                    "kind": "shape_sources",
                    "class_id": int(value),
                    "name": SOURCE_CLASS_LABELS.get(int(value), PU_CLASS_NAMES.get(int(value), f"class_{int(value)}")),
                    "count": int(count),
                }
            )
        for value, count in zip(center_class_values, center_class_counts):
            writer.writerow(
                {
                    "kind": "source_centers",
                    "class_id": int(value),
                    "name": SOURCE_CLASS_LABELS.get(int(value), PU_CLASS_NAMES.get(int(value), f"class_{int(value)}")),
                    "count": int(count),
                }
            )
        writer.writerow({"kind": "strict_centers", "class_id": 5, "name": "strict_center_only", "count": len(strict_centers)})


def _run_one(args: argparse.Namespace, dataset_source: str | None) -> Path:
    effective_group = None if dataset_source == "coadd" else args.group
    reader, sample_idx, band_idx, attrs = resolve_zarr_sample(
        zarr_store=args.zarr_store.expanduser().resolve() if args.zarr_store else None,
        sample_index=int(args.sample_index),
        root=args.root.expanduser().resolve() if args.root else None,
        patch=args.patch,
        tile_name=args.tile_name,
        band=args.band,
        dataset_source=dataset_source,
        group=effective_group,
        image_level=bool(args.image_level),
    )
    sample = read_zarr_sample(reader, sample_idx, band_idx)
    band_name = (attrs.get("bands") or [args.band or f"band{band_idx}"])[band_idx]
    patch_name = str(attrs.get("patch", args.patch or reader.root.stem))
    dataset = str(attrs.get("dataset_source", dataset_source or "zarr"))
    group_name = zarr_sample_group(reader, sample_idx)
    tile_x0 = int(reader.read_full_small("tile_x0")[sample_idx]) if reader.has_array("tile_x0") else 0
    tile_y0 = int(reader.read_full_small("tile_y0")[sample_idx]) if reader.has_array("tile_y0") else 0
    group_part = f"_{group_name}" if group_name else ""
    out_stem = f"{dataset}{group_part}_{patch_name.replace(',', '_')}_{band_name}_sample{sample_idx:05d}"
    out_dir = args.out_dir.expanduser().resolve() / patch_name / str(band_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = np.asarray(sample["display_image"], dtype=np.float32)
    if args.fits_image is not None:
        raw, _header, _hdu = read_fits_image(args.fits_image.expanduser().resolve(), hdu=args.fits_hdu)
        image, _valid = crop_or_pad(raw, x0=tile_x0, y0=tile_y0, width=image.shape[1], height=image.shape[0])
        image = image[: sample["display_image"].shape[0], : sample["display_image"].shape[1]]

    pu = np.asarray(sample["pu_class"], dtype=np.uint8)
    conf = np.asarray(sample["confidence"], dtype=np.uint8)
    sources = source_rows_from_zarr(reader, sample_idx, band_idx)
    centers = centers_from_zarr(reader, sample_idx, band_idx)
    strict_centers = strict_centers_from_zarr(reader, sample_idx, band_idx)
    all_centers = [*centers, *strict_centers]

    save_pixel_png(out_dir / f"{out_stem}_pu_overlay.png", label_mask_overlay(image, pu), scale=args.png_scale)
    save_pixel_png(
        out_dir / f"{out_stem}_confidence_overlay.png",
        confidence_overlay(image, conf),
        scale=args.png_scale,
    )
    save_pixel_png(
        out_dir / f"{out_stem}_source_shapes_overlay.png",
        _source_shapes_overlay(image, sources, strict_centers),
        scale=args.png_scale,
    )
    save_pixel_png(
        out_dir / f"{out_stem}_centers_overlay.png",
        draw_points(image, all_centers),
        scale=args.png_scale,
    )
    if bool(args.confidence_negative_overlay):
        _save_confidence_negative_plot(
            out_dir / f"{out_stem}_confidence_negative_overlay.png",
            image,
            sample,
            title=f"{dataset}{group_part} {patch_name} {band_name} sample {sample_idx}",
            alpha=float(args.negative_alpha),
        )
    if bool(args.confidence_negative_clear_overlay):
        _save_confidence_negative_plot(
            out_dir / f"{out_stem}_confidence_negative_clear_overlay.png",
            image,
            sample,
            title=f"{dataset}{group_part} {patch_name} {band_name} sample {sample_idx}",
            alpha=float(args.negative_alpha),
            clear_negative=True,
        )
    _panel(out_dir / f"{out_stem}_panel.png", image, pu, conf, sources, strict_centers)
    _write_counts_csv(
        out_dir / f"{out_stem}_label_counts.csv",
        pu=pu,
        conf=conf,
        sources=sources,
        centers=centers,
        strict_centers=strict_centers,
    )

    write_reg(out_dir / f"{out_stem}_source_shapes.reg", rows_to_reg(sources, shape=True, color=None))
    write_reg(out_dir / f"{out_stem}_source_centers.reg", rows_to_reg(centers, shape=False, color=None))
    write_reg(
        out_dir / f"{out_stem}_strict_center_only_centers.reg",
        rows_to_reg(strict_centers, shape=False, color=EXTERNAL_SOURCE_REG_COLOR),
    )
    write_reg(
        out_dir / f"{out_stem}_external_sources.reg",
        rows_to_reg(strict_centers, shape=False, color=EXTERNAL_SOURCE_REG_COLOR),
    )
    write_reg(out_dir / f"{out_stem}_all_supervised_centers.reg", rows_to_reg(all_centers, shape=False, color=None))

    image_stack = np.asarray(sample["image"], dtype=np.float32)
    if image_stack.ndim == 3:
        scaling_mode = str(attrs.get("image_scaling_mode") or attrs.get("scaling_mode") or "zscore")
        clip_threshold = float(attrs.get("clip_threshold", attrs.get("image_clip_threshold", 3.0)))
        for channel in range(min(3, image_stack.shape[0])):
            vmin, vmax = input_channel_display_limits(
                image_stack[channel],
                scaling=scaling_mode,
                channel_index=channel,
                clip_threshold=clip_threshold,
            )
            save_heatmap(
                out_dir / f"{out_stem}_input_channel{channel}.png",
                image_stack[channel],
                title=f"{out_stem} input channel {channel}",
                vmin=vmin,
                vmax=vmax,
            )
    return out_dir


def main() -> int:
    args = parse_args()
    out_dirs = []
    if args.zarr_store is not None and args.dataset_source:
        raise ValueError("--dataset-source selection is only used with --root; direct --zarr-store already selects one dataset")
    for dataset_source in _dataset_sources(args.dataset_source):
        if bool(args.all_samples):
            if not (bool(args.confidence_negative_overlay) or bool(args.confidence_negative_clear_overlay)):
                raise ValueError("--all-samples currently requires --confidence-negative-overlay or --confidence-negative-clear-overlay")
            paths = _run_all_confidence_negative_overlays(args, dataset_source)
            out_dirs.extend(path.parent for path in paths)
        else:
            out_dirs.append(_run_one(args, dataset_source))
    for out_dir in sorted(set(out_dirs)):
        print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
