#!/usr/bin/env python3
"""Shared utilities for lightweight CELLECT/SAM evaluation visualizers."""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from astro_train_ops import detect_centers_with_scores, model_forward_with_batch_context  # noqa: E402
from astro_train_zarr_data import PatchZarrReader, discover_zarr_image_records, discover_zarr_records  # noqa: E402
from data_filtering.sam_input_scaling import (  # noqa: E402
    current_sam_zscore,
    no_first_clip_zscore,
    scale_training_image,
)
from sam_backbone import SamAutomaticMaskGenerator, build_sam_cellect2d, sam_model_registry  # noqa: E402


PU_CLASS_NAMES = {
    0: "none",
    1: "clean",
    2: "weak_shape",
    3: "ordinary_ignore",
    4: "background",
    5: "strict_center_only",
    6: "restricted_bright_region",
    7: "strict_ignore",
}

PU_COLORS = {
    1: np.asarray((0.0, 0.85, 0.25), dtype=np.float32),
    2: np.asarray((0.20, 0.70, 1.00), dtype=np.float32),
    3: np.asarray((1.00, 0.18, 0.12), dtype=np.float32),
    4: np.asarray((0.45, 0.45, 0.45), dtype=np.float32),
    5: np.asarray((0.78, 0.22, 1.00), dtype=np.float32),
    6: np.asarray((1.00, 0.72, 0.00), dtype=np.float32),
    7: np.asarray((0.48, 0.32, 1.00), dtype=np.float32),
}

CONF_COLORS = {
    1: np.asarray((0.15, 0.45, 1.00), dtype=np.float32),
    2: np.asarray((0.00, 0.95, 0.35), dtype=np.float32),
    3: np.asarray((1.00, 0.85, 0.05), dtype=np.float32),
    4: np.asarray((1.00, 0.05, 0.10), dtype=np.float32),
    5: np.asarray((1.00, 0.05, 0.10), dtype=np.float32),
    6: np.asarray((1.00, 0.25, 0.12), dtype=np.float32),
    7: np.asarray((1.00, 0.25, 0.12), dtype=np.float32),
}

SOURCE_CLASS_COLORS = {
    1: "cyan", # green
    2: "blue", # cyan
    4: "magenta",
    5: "magenta",
    6: "orange",
    7: "violet",
}

ELLIPSE_REGION_RE = re.compile(
    r"ellipse\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)"
)

REG_HEADER_IMAGE = (
    "# Region file format: DS9 version 4.1",
    'global color=cyan dashlist=8 3 width=2 font="helvetica 10 normal roman" '
    "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
    "image",
)


def read_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    args = dict(payload.get("args", {}))
    args["_top"] = payload
    return args


def strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key[7:] if str(key).startswith("module.") else key: value for key, value in state.items()}


def read_fits_image(path: Path, *, hdu: int | None = None) -> tuple[np.ndarray, fits.Header, int]:
    with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
        indices = range(len(hdul)) if hdu is None else (int(hdu),)
        for index in indices:
            data = hdul[index].data
            if data is None:
                continue
            arr = np.asarray(data).squeeze()
            if arr.ndim == 2:
                return np.asarray(arr, dtype=np.float32), hdul[index].header.copy(), int(index)
    raise ValueError(f"no 2D image HDU found: {path}")


def ltv_image_origin(header: fits.Header | None) -> tuple[int, int]:
    """Return the parent physical-coordinate origin encoded by LSST LTV."""
    if header is None or "LTV1" not in header or "LTV2" not in header:
        return (0, 0)
    return (-int(round(float(header["LTV1"]))), -int(round(float(header["LTV2"]))))


def crop_origin_for_image(
    image: np.ndarray,
    header: fits.Header | None,
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> tuple[int, int]:
    """Choose local array crop coordinates from either local or physical input."""
    x0_i = int(x0)
    y0_i = int(y0)
    direct = (x0_i, y0_i)
    origin_x, origin_y = ltv_image_origin(header)
    physical = (x0_i - origin_x, y0_i - origin_y)

    def overlap(candidate: tuple[int, int]) -> int:
        cx, cy = candidate
        sx0 = max(0, cx)
        sy0 = max(0, cy)
        sx1 = min(int(image.shape[1]), cx + int(width))
        sy1 = min(int(image.shape[0]), cy + int(height))
        return max(0, sx1 - sx0) * max(0, sy1 - sy0)

    direct_overlap = overlap(direct)
    physical_overlap = overlap(physical)
    if physical_overlap > direct_overlap:
        return physical
    return direct


def crop_or_pad(
    image: np.ndarray,
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
    inference_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    if width > inference_size or height > inference_size:
        raise ValueError(f"crop {width}x{height} exceeds inference size {inference_size}")
    out = np.full((inference_size, inference_size), np.nan, dtype=np.float32)
    valid = np.zeros((inference_size, inference_size), dtype=bool)
    sx0 = max(0, int(x0))
    sy0 = max(0, int(y0))
    sx1 = min(image.shape[1], int(x0) + int(width))
    sy1 = min(image.shape[0], int(y0) + int(height))
    if sx1 > sx0 and sy1 > sy0:
        dx0 = sx0 - int(x0)
        dy0 = sy0 - int(y0)
        view = image[sy0:sy1, sx0:sx1]
        out[dy0 : dy0 + view.shape[0], dx0 : dx0 + view.shape[1]] = view
        valid[dy0 : dy0 + view.shape[0], dx0 : dx0 + view.shape[1]] = np.isfinite(view)
    return out, valid


def zscale_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    if not bool(finite.any()):
        return np.zeros(arr.shape, dtype=np.float32)
    try:
        lo, hi = ZScaleInterval().get_limits(arr[finite])
    except Exception:
        lo, hi = np.nanpercentile(arr[finite], [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(arr[finite])), float(np.nanmax(arr[finite]))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def zscale_rgb(image: np.ndarray) -> np.ndarray:
    return np.repeat(zscale_gray(image)[..., None], 3, axis=2)


def alpha_blend(rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> None:
    mask = np.asarray(mask, dtype=bool)
    if bool(mask.any()):
        rgb[mask] = (1.0 - float(alpha)) * rgb[mask] + float(alpha) * color[None, :]


def save_png(path: Path, rgb: np.ndarray, *, title: str = "", scale: int = 1) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = rgb.shape[:2]
    fig_w = max(4.0, min(10.0, w / 100.0))
    fig_h = max(4.0, min(10.0, h / 100.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140 * max(1, int(scale)))
    ax.imshow(np.clip(rgb, 0.0, 1.0), origin="lower", interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_pixel_png(path: Path, rgb: np.ndarray, *, scale: int = 1) -> None:
    """Save an image-coordinate RGB array without axes, title, or padding.

    The visualization helpers in this module draw in image coordinates with
    y=0 at the lower edge.  PNG files are row-major with y=0 at the upper edge,
    so this function flips before writing.  This keeps a 512x512 cutout exactly
    512x512 on disk, which is important when overlay coordinates are inspected.
    """
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rgb, dtype=np.float32)
    arr = np.clip(np.rint(np.flipud(arr) * 255.0), 0, 255).astype(np.uint8)
    image = Image.fromarray(arr, mode="RGB")
    scale_i = max(1, int(scale))
    if scale_i > 1:
        image = image.resize((image.width * scale_i, image.height * scale_i), resample=Image.Resampling.NEAREST)
    image.save(path)


def save_heatmap(path: Path, plane: np.ndarray, *, title: str = "", vmin: float | None = None, vmax: float | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 6.2), dpi=150)
    im = ax.imshow(plane, origin="lower", cmap="magma", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def input_channel_display_limits(
    channel: np.ndarray,
    *,
    scaling: str,
    channel_index: int | None = None,
    clip_threshold: float = 3.0,
) -> tuple[float, float]:
    """Return the display interval used for model-input channel plots.

    Most training scalings store standardized channels clipped to [-3, 3].
    ``zscore-no-upper`` deliberately has no finite upper clip, so for that
    mode the exact finite min/max of the input channel is the displayed range,
    matching matplotlib's default heatmap semantics.
    """
    label = str(scaling).strip().lower().replace("_", "-")
    arr = np.asarray(channel, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (0.0, 1.0)
    if "no-upper" in label or "unbounded" in label:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    else:
        threshold = float(clip_threshold)
        if not np.isfinite(threshold) or threshold <= 0.0:
            threshold = 3.0
        lo, hi = -threshold, threshold
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return (0.0, 1.0)
    return (lo, hi)


def input_channel_to_rgb(
    channel: np.ndarray,
    *,
    scaling: str,
    channel_index: int | None = None,
    clip_threshold: float = 3.0,
) -> np.ndarray:
    lo, hi = input_channel_display_limits(
        channel,
        scaling=scaling,
        channel_index=channel_index,
        clip_threshold=clip_threshold,
    )
    gray = np.clip((np.asarray(channel, dtype=np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    gray = np.nan_to_num(gray, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    return np.repeat(gray[..., None], 3, axis=2)


def ellipse_line(
    x: float,
    y: float,
    major: float,
    minor: float,
    theta: float,
    *,
    color: str,
    width: int = 2,
    text: str = "",
) -> str:
    suffix = f" # color={color} width={int(width)}"
    if text:
        suffix += f" text={{{text}}}"
    return (
        f"ellipse({x + 1.0:.3f},{y + 1.0:.3f},"
        f"{max(abs(major), 1.0):.3f},{max(abs(minor), 1.0):.3f},{math.degrees(theta):.3f}){suffix}"
    )


def point_line(x: float, y: float, *, color: str, width: int = 2, text: str = "") -> str:
    suffix = f" # point=cross color={color} width={int(width)}"
    if text:
        suffix += f" text={{{text}}}"
    return f"point({x + 1.0:.3f},{y + 1.0:.3f}){suffix}"


def polygon_line(points: Sequence[tuple[float, float]], *, color: str, width: int = 2, text: str = "") -> str:
    coords = ",".join(f"{x + 1.0:.2f},{y + 1.0:.2f}" for x, y in points)
    suffix = f" # color={color} width={int(width)}"
    if text:
        suffix += f" text={{{text}}}"
    return f"polygon({coords}){suffix}"


def write_reg(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([*REG_HEADER_IMAGE, *lines]) + "\n", encoding="ascii")


def contours(mask: np.ndarray, *, max_vertices: int = 180) -> Iterable[list[tuple[float, float]]]:
    try:
        from skimage import measure

        for contour in measure.find_contours(mask.astype(np.float32), 0.5):
            if contour.shape[0] < 3:
                continue
            step = max(1, int(math.ceil(contour.shape[0] / int(max_vertices))))
            pts = contour[::step]
            if pts.shape[0] >= 3:
                yield [(float(x), float(y)) for y, x in pts]
        return
    except Exception:
        pass
    ys, xs = np.where(mask)
    if ys.size:
        yield [
            (float(xs.min()), float(ys.min())),
            (float(xs.max() + 1), float(ys.min())),
            (float(xs.max() + 1), float(ys.max() + 1)),
            (float(xs.min()), float(ys.max() + 1)),
        ]


def draw_ellipses(
    image: np.ndarray,
    rows: Sequence[dict[str, float]],
    *,
    color: str | None = None,
    point_color: str | None = None,
    draw_centers: bool = True,
    invert_background: bool = False,
    line_width: float = 1.0,
    input_scaled_background: bool = False,
    input_scaling: str | None = None,
    input_channel_index: int | None = None,
    input_clip_threshold: float = 3.0,
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Ellipse

    fig, ax = plt.subplots(figsize=(image.shape[1] / 100.0, image.shape[0] / 100.0), dpi=100)
    if input_scaled_background:
        rgb = input_channel_to_rgb(
            image,
            scaling=input_scaling or "zscore",
            channel_index=input_channel_index,
            clip_threshold=float(input_clip_threshold),
        )
        gray = rgb[..., 0]
    else:
        gray = zscale_gray(image)
    if bool(invert_background):
        gray = 1.0 - gray
    ax.imshow(gray, origin="lower", cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    for row in sorted(rows, key=lambda r: abs(float(r["major"]) * float(r["minor"])), reverse=True):
        row_color = color or SOURCE_CLASS_COLORS.get(int(row.get("class_id", 1)), "yellow")
        row_point_color = point_color or row_color
        ax.add_patch(
            Ellipse(
                (float(row["x"]), float(row["y"])),
                width=2.0 * max(abs(float(row["major"])), 1.0),
                height=2.0 * max(abs(float(row["minor"])), 1.0),
                angle=math.degrees(float(row.get("theta", 0.0))),
                fill=False,
                edgecolor=row_color,
                linewidth=float(line_width),
                alpha=0.95,
            )
        )
        if draw_centers:
            ax.plot(float(row["x"]), float(row["y"]), marker="+", color=row_point_color, markersize=1.5, mew=0.8)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    plt.close(fig)
    # Agg buffers are top-row first, while downstream save_png uses
    # origin="lower" for image-coordinate displays.
    return np.flipud(rgba[..., :3]).astype(np.float32) / 255.0


def read_ds9_ellipse_regions(path: Path) -> list[dict[str, float]]:
    """Read DS9 image-coordinate ellipse rows into ``draw_ellipses`` format."""
    rows: list[dict[str, float]] = []
    for line in Path(path).read_text(encoding="ascii", errors="ignore").splitlines():
        match = ELLIPSE_REGION_RE.search(line)
        if match is None:
            continue
        x, y, major, minor, theta_deg = (float(value) for value in match.groups())
        rows.append(
            {
                "x": x,
                "y": y,
                "major": major,
                "minor": minor,
                "theta": math.radians(theta_deg),
            }
        )
    return rows


def inverse_ellipse_overlay(
    image: np.ndarray,
    rows: Sequence[dict[str, float]],
    *,
    color: str = "#0066ff",
    line_width: float = 1.6,
    draw_centers: bool = False,
) -> np.ndarray:
    """Draw ellipses on an inverted zscale background."""
    return draw_ellipses(
        image,
        rows,
        color=color,
        point_color=color,
        draw_centers=bool(draw_centers),
        invert_background=True,
        line_width=float(line_width),
    )


def draw_points(
    image: np.ndarray,
    rows: Sequence[dict[str, float]],
    *,
    color: str | None = None,
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig, ax = plt.subplots(figsize=(image.shape[1] / 100.0, image.shape[0] / 100.0), dpi=100)
    ax.imshow(zscale_gray(image), origin="lower", cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    for row in rows:
        row_color = color or SOURCE_CLASS_COLORS.get(int(row.get("class_id", 1)), "yellow")
        ax.plot(float(row["x"]), float(row["y"]), marker="+", color=row_color, markersize=4.0, mew=1.0)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    plt.close(fig)
    return np.flipud(rgba[..., :3]).astype(np.float32) / 255.0


def label_mask_overlay(image: np.ndarray, pu_class: np.ndarray, *, alpha: float = 0.36) -> np.ndarray:
    rgb = zscale_rgb(image)
    # The zarr target stores one mutually-exclusive class id per pixel. This
    # visualization intentionally does not reconstruct labels from component
    # masks or impose any extra priority beyond those stored class ids.
    classes = np.asarray(pu_class, dtype=np.uint8)
    color = np.zeros_like(rgb)
    visible = np.zeros(classes.shape, dtype=bool)
    for class_id, class_color in PU_COLORS.items():
        mask = classes == int(class_id)
        if bool(mask.any()):
            color[mask] = class_color
            visible |= mask
    if bool(visible.any()):
        rgb[visible] = (1.0 - float(alpha)) * rgb[visible] + float(alpha) * color[visible]
    return rgb


def confidence_overlay(image: np.ndarray, confidence: np.ndarray, *, alpha: float = 0.72) -> np.ndarray:
    rgb = zscale_rgb(image)
    for level in (1, 2, 3, 4, 5, 6, 7):
        alpha_blend(rgb, np.asarray(confidence) == level, CONF_COLORS[level], alpha)
    return rgb


def source_rows_from_zarr(reader: PatchZarrReader, sample_idx: int, band_idx: int) -> list[dict[str, float]]:
    if not reader.has_array("shape_source_offsets"):
        return []
    offsets = reader.read_full_small("shape_source_offsets").astype(np.int64, copy=False)
    centers = reader.read_full_small("shape_source_centers").astype(np.float32, copy=False)
    values = reader.read_full_small("shape_source_values").astype(np.float32, copy=False)
    classes = reader.read_full_small("shape_source_classes").astype(np.uint8, copy=False)
    ids = reader.read_full_small("shape_source_ids").astype(np.int64, copy=False)
    s0, s1 = int(offsets[sample_idx, band_idx]), int(offsets[sample_idx, band_idx + 1])
    rows = []
    for center, value, class_id, source_id in zip(centers[s0:s1], values[s0:s1], classes[s0:s1], ids[s0:s1]):
        rows.append(
            {
                "x": float(center[0]),
                "y": float(center[1]),
                "major": float(value[0]),
                "minor": float(value[1]),
                "theta": float(value[2]),
                "class_id": int(class_id),
                "source_id": int(source_id),
            }
        )
    return rows


def centers_from_zarr(reader: PatchZarrReader, sample_idx: int, band_idx: int) -> list[dict[str, float]]:
    if not reader.has_array("source_offsets"):
        return []
    offsets = reader.read_full_small("source_offsets").astype(np.int64, copy=False)
    centers = reader.read_full_small("source_centers").astype(np.float32, copy=False)
    ids = reader.read_full_small("source_ids").astype(np.int64, copy=False)
    class_by_id: dict[int, int] = {}
    if reader.has_array("shape_source_offsets") and reader.has_array("shape_source_ids") and reader.has_array("shape_source_classes"):
        shape_offsets = reader.read_full_small("shape_source_offsets").astype(np.int64, copy=False)
        shape_ids = reader.read_full_small("shape_source_ids").astype(np.int64, copy=False)
        shape_classes = reader.read_full_small("shape_source_classes").astype(np.uint8, copy=False)
        t0, t1 = int(shape_offsets[sample_idx, band_idx]), int(shape_offsets[sample_idx, band_idx + 1])
        class_by_id = {
            int(source_id): int(class_id)
            for source_id, class_id in zip(shape_ids[t0:t1], shape_classes[t0:t1])
        }
    s0, s1 = int(offsets[sample_idx, band_idx]), int(offsets[sample_idx, band_idx + 1])
    return [
        {"x": float(center[0]), "y": float(center[1]), "source_id": int(source_id), "class_id": class_by_id.get(int(source_id), 1)}
        for center, source_id in zip(centers[s0:s1], ids[s0:s1])
    ]


def strict_centers_from_zarr(reader: PatchZarrReader, sample_idx: int, band_idx: int) -> list[dict[str, float]]:
    if not reader.has_array("strict_center_only_offsets"):
        return []
    offsets = reader.read_full_small("strict_center_only_offsets").astype(np.int64, copy=False)
    centers = reader.read_full_small("strict_center_only_centers").astype(np.float32, copy=False)
    ids = (
        reader.read_full_small("strict_center_only_ids").astype(np.int64, copy=False)
        if reader.has_array("strict_center_only_ids")
        else np.full((len(centers),), -1, dtype=np.int64)
    )
    s0, s1 = int(offsets[sample_idx, band_idx]), int(offsets[sample_idx, band_idx + 1])
    return [
        {"x": float(center[0]), "y": float(center[1]), "source_id": int(source_id), "class_id": 5}
        for center, source_id in zip(centers[s0:s1], ids[s0:s1])
    ]


def resolve_zarr_sample(
    *,
    zarr_store: Path | None,
    sample_index: int,
    root: Path | None,
    patch: str | None,
    tile_name: str | None,
    band: str | None,
    dataset_source: str | None,
    group: str | None,
    image_level: bool,
) -> tuple[PatchZarrReader, int, int, dict]:
    group = normalize_group_name(group)
    if zarr_store is not None:
        reader = PatchZarrReader(zarr_store)
        attrs = dict(reader.attrs)
        if group is not None:
            sample_group = zarr_sample_group(reader, int(sample_index))
            if sample_group != group:
                raise ValueError(f"sample {sample_index} is group={sample_group!r}, not requested group={group!r}")
        bands = list(attrs.get("bands", []))
        band_idx = bands.index(band) if band in bands else 0
        return reader, int(sample_index), int(band_idx), attrs
    if root is None:
        raise ValueError("provide --zarr-store or --root")
    records = (discover_zarr_image_records if image_level else discover_zarr_records)(root, bands=[band] if band else [])
    matches = []
    readers_by_store: dict[str, PatchZarrReader] = {}
    for rec in records:
        rec_patch = str(rec.patch).split("__", 1)[0]
        if patch and rec.patch != patch and rec_patch != patch:
            continue
        if tile_name and not tile_name_matches(rec.tile_name, tile_name):
            continue
        if dataset_source and rec.dataset_source != dataset_source:
            continue
        if group is not None:
            store_s, idx_s = str(rec.image_paths[0])[len("zarr://") :].split("#", 1)
            reader = readers_by_store.get(store_s)
            if reader is None:
                reader = PatchZarrReader(Path(store_s))
                readers_by_store[store_s] = reader
            if zarr_sample_group(reader, int(idx_s)) != group:
                continue
        matches.append(rec)
    if not matches:
        raise RuntimeError("no matching zarr sample found")
    if int(sample_index) >= len(matches):
        raise IndexError(f"sample-index {sample_index} exceeds {len(matches)} matching records")
    rec = matches[int(sample_index)]
    uri = rec.image_paths[0]
    store_s, idx_s = str(uri)[len("zarr://") :].split("#", 1)
    reader = PatchZarrReader(Path(store_s))
    attrs = dict(reader.attrs)
    bands = list(attrs.get("bands", []))
    band_idx = bands.index(band) if band in bands else 0
    return reader, int(idx_s), int(band_idx), attrs


def decode_fixed_utf8(arr: np.ndarray) -> list[str]:
    out = []
    raw = np.asarray(arr)
    for row in raw:
        item = np.asarray(row)
        if item.dtype.kind == "S":
            text = bytes(item.reshape(-1)[0]).split(b"\0", 1)[0].decode("utf-8", errors="replace")
        elif item.dtype.kind == "U":
            text = str(item.reshape(-1)[0]).split("\0", 1)[0]
        else:
            text = bytes(int(v) for v in item.astype(np.uint8, copy=False).ravel()).split(b"\0", 1)[0].decode(
                "utf-8",
                errors="replace",
            )
        out.append(text)
    return out


def normalize_group_name(group: str | None) -> str | None:
    if group is None:
        return None
    text = str(group).strip()
    if not text:
        return None
    if text.startswith("group_"):
        return text
    if text.isdigit():
        return f"group_{int(text):02d}"
    return text


def zarr_sample_group(reader: PatchZarrReader, sample_idx: int) -> str:
    if reader.has_array("group"):
        groups = decode_fixed_utf8(reader.read_full_small("group"))
        if int(sample_idx) < len(groups) and groups[int(sample_idx)]:
            return groups[int(sample_idx)]
    stem = reader.root.stem
    if "__" not in stem:
        return ""
    suffix = stem.split("__", 1)[1]
    return suffix if suffix.startswith("group_") else ""


def tile_name_matches(actual: str, requested: str) -> bool:
    actual = str(actual)
    requested = str(requested)
    if actual == requested:
        return True
    if actual.endswith("_" + requested):
        return True
    actual_parts = actual.split("_")
    for start in range(len(actual_parts)):
        if "_".join(actual_parts[start:]) == requested:
            return True
    import re

    match = re.fullmatch(r"r(\d+)c(\d+)", requested.lower())
    if match:
        row = int(match.group(1))
        col = int(match.group(2))
        return f"grid_r{row:02d}_c{col:02d}_" in actual
    return False


def read_zarr_sample(reader: PatchZarrReader, sample_idx: int, band_idx: int) -> dict[str, np.ndarray]:
    image = reader.read_first_axis("images", sample_idx).astype(np.float32, copy=False)
    confidence = reader.read_first_axis("band_confidence", sample_idx).astype(np.uint8, copy=False)
    conf_weight = reader.read_first_axis("band_conf_weight", sample_idx).astype(np.float32, copy=False)
    shape = reader.read_first_axis("band_shape", sample_idx).astype(np.float32, copy=False)
    shape_weight = reader.read_first_axis("band_shape_weight", sample_idx).astype(np.float32, copy=False)
    pu_class = reader.read_first_axis("band_pu_class_mask", sample_idx).astype(np.uint8, copy=False)
    if image.ndim == 5:
        raise ValueError("unexpected image ndim after first-axis read")
    if image.ndim == 4:
        image_band = image[band_idx]
        display = image_band[0] if image_band.ndim == 3 else image_band
    else:
        image_band = image[band_idx]
        display = image_band
    return {
        "image": image_band,
        "display_image": np.asarray(display, dtype=np.float32),
        "confidence": confidence[band_idx],
        "confidence_weight": conf_weight[band_idx],
        "shape": shape[band_idx],
        "shape_weight": shape_weight[band_idx],
        "pu_class": pu_class[band_idx],
    }


def build_scaled_tensor_from_fits(
    paths: Sequence[Path],
    *,
    hdu: int | None,
    x0: int,
    y0: int,
    width: int,
    height: int,
    scaling_mode: str,
    clip_threshold: float,
    log_a: float,
    log_high_percentile: float,
    lupton_stretch: float,
    lupton_q: float,
    anscombe_clip: bool,
    anscombe_scale: float,
) -> tuple[torch.Tensor, list[np.ndarray], list[np.ndarray], list[fits.Header]]:
    raw_images: list[np.ndarray] = []
    scaled_images: list[np.ndarray] = []
    headers: list[fits.Header] = []
    for path in paths:
        full, header, _ = read_fits_image(path, hdu=hdu)
        local_x0, local_y0 = crop_origin_for_image(full, header, x0=x0, y0=y0, width=width, height=height)
        crop, valid = crop_or_pad(full, x0=local_x0, y0=local_y0, width=width, height=height)
        crop = np.where(valid, crop, np.nan).astype(np.float32)
        scaled = make_training_rgb(
            crop,
            mode=scaling_mode,
            clip_threshold=clip_threshold,
            log_a=log_a,
            log_high_percentile=log_high_percentile,
            lupton_stretch=lupton_stretch,
            lupton_q=lupton_q,
            anscombe_clip=anscombe_clip,
            anscombe_scale=anscombe_scale,
        )
        raw_images.append(np.nan_to_num(crop, nan=0.0, posinf=0.0, neginf=0.0)[:height, :width])
        scaled_images.append(scaled[:, :height, :width])
        headers.append(header)
    tensor = torch.from_numpy(np.stack(scaled_images, axis=0).astype(np.float32, copy=False))[None]
    return tensor, raw_images, scaled_images, headers


def make_training_rgb(
    image: np.ndarray,
    *,
    mode: str,
    clip_threshold: float = 3.0,
    log_a: float = 300.0,
    log_high_percentile: float = 99.5,
    lupton_stretch: float = 0.5,
    lupton_q: float = 20.0,
    anscombe_clip: bool = False,
    anscombe_scale: float = 1000.0,
) -> np.ndarray:
    normalized = str(mode).strip().lower().replace("_", "-")
    if normalized in {"zscore-clip", "zscore", "clip"}:
        return scale_training_image(image, mode="zscore-rgb", clip_threshold=float(clip_threshold))
    if normalized in {"zscore-no-clip", "zscore-noclip"}:
        z, _stats = no_first_clip_zscore(image, z_clip=(-float(clip_threshold), float(clip_threshold)))
        return np.stack([z, z, z], axis=0).astype(np.float32)
    if normalized in {"zscore-no-upper", "zscore-unbounded"}:
        return scale_training_image(image, mode="zscore-no-upper-rgb", clip_threshold=float(clip_threshold))
    if normalized in {"log-lupton", "lupton-log", "zscore-log-lupton"}:
        return scale_training_image(
            image,
            mode="zscore-log-lupton-rgb",
            clip_threshold=float(clip_threshold),
            log_a=log_a,
            log_high_percentile=float(log_high_percentile),
            lupton_stretch=float(lupton_stretch),
            lupton_q=float(lupton_q),
        )
    if normalized == "anscombe":
        return scale_training_image(
            image,
            mode="anscombe-rgb",
            clip_threshold=float(clip_threshold),
            anscombe_clip=bool(anscombe_clip),
            anscombe_scale=anscombe_scale,
        )
    raise ValueError(f"unknown scaling mode: {mode}")


def scaled_rgb_for_display(chw: np.ndarray) -> np.ndarray:
    arr = np.asarray(chw, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=0)
    out = []
    for channel in arr[:3]:
        finite = np.isfinite(channel)
        if not bool(finite.any()):
            out.append(np.zeros(channel.shape, dtype=np.float32))
            continue
        lo, hi = np.nanpercentile(channel[finite], [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(channel[finite])), float(np.nanmax(channel[finite]))
        out.append(np.clip((channel - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32))
    return np.stack(out[:3], axis=2)


def sam_uint8_from_scaled(chw: np.ndarray) -> np.ndarray:
    rgb = scaled_rgb_for_display(chw)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def load_cellect_model(
    checkpoint: Path,
    config: Path | None,
    device: torch.device,
    bands: Sequence[str],
    *,
    dynamic_image_size: bool | None = None,
) -> tuple[torch.nn.Module, dict]:
    cfg = read_json(config if config is not None else checkpoint.parent / "run_config.json")
    top = cfg.get("_top", {})
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if isinstance(ckpt, dict) and ckpt.get("args"):
        cfg.update({k: v for k, v in ckpt["args"].items() if k not in cfg})
    style_from_state = isinstance(state, dict) and any(
        "encoder.style_router." in str(k) or "encoder.image_encoder.style_adapters." in str(k) for k in state
    )
    film_from_state = isinstance(state, dict) and any("decoder.denoised_film." in str(k) for k in state)
    base_channels = int(top.get("base_channels") or cfg.get("base_channels", 32))
    if dynamic_image_size is None:
        dynamic_image_size = bool(top.get("sam_dynamic_image_size", cfg.get("sam_dynamic_image_size", False)))
    model = build_sam_cellect2d(
        str(top.get("sam_model_type") or cfg.get("sam_model_type", "vit_b")),
        checkpoint=None,
        num_bands=len(bands),
        image_size=512,
        patch_size=16,
        seg_classes=int(top.get("seg_classes") or cfg.get("seg_classes", 2)),
        confidence_levels=5,
        embedding_dim=int(cfg.get("embedding_dim", 64)),
        shape_channels=3,
        decoder_channels=(base_channels * 8, base_channels * 4, base_channels * 2, base_channels),
        use_cen=bool(top.get("sam_cen_enabled", not bool(cfg.get("disable_sam_cen", False)))),
        cen_input_image=True,
        cen_width=max(2, base_channels // 4),
        decoder_denoised_film=bool(top.get("sam_decoder_film", cfg.get("sam_decoder_film", False)) or film_from_state),
        encoder_style_prompt=bool(top.get("sam_encoder_style_prompt", cfg.get("sam_encoder_style_prompt", False)) or style_from_state),
        style_prompt_dim=int(top.get("style_prompt_dim", cfg.get("style_prompt_dim", 32))),
        style_prompt_layers=tuple(top.get("style_prompt_layers", cfg.get("style_prompt_layers", (2, 5, 8)))),
        style_adapter_dim=int(top.get("style_adapter_dim", cfg.get("style_adapter_dim", 32))),
        style_router_temperature=float(top.get("style_router_temperature", cfg.get("style_router_temperature", 1.0))),
        candidate_count=int(cfg.get("matcher_candidate_count", 5)),
        shape_feature_dim=6,
        enable_matchers=False,
        astro_preprocess_in_model=False,
        dynamic_image_size=bool(dynamic_image_size),
    ).to(device)
    incompatible = model.load_state_dict(strip_module_prefix(state), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"[load] missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}",
            flush=True,
        )
    model.eval()
    return model, cfg


def select_band_outputs(outputs: dict[str, torch.Tensor], band_idx: int) -> dict[str, torch.Tensor]:
    selected = {}
    for key, value in outputs.items():
        if not torch.is_tensor(value):
            continue
        selected[key] = value[:, band_idx] if value.ndim >= 5 else value
    return selected


def infer_cellect(
    *,
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    amp: str = "bf16",
) -> dict[str, torch.Tensor]:
    context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" and amp == "bf16" else nullcontext()
    with torch.inference_mode(), context:
        batch = image_tensor.to(device=device, dtype=torch.float32)
        context_batch = {"processing_id": torch.zeros((batch.shape[0],), dtype=torch.long, device=device)}
        outputs = model_forward_with_batch_context(model, batch, context_batch)
    return outputs


def detection_rows(
    outputs: dict[str, torch.Tensor],
    *,
    threshold: float,
    nms_radius: int,
    confidence_score: str,
    center_refinement: str,
    center_refinement_radius: int,
    width: int,
    height: int,
) -> list[dict[str, float]]:
    detect_outputs = {
        key: value.float() if torch.is_tensor(value) and value.dtype in {torch.float16, torch.bfloat16} else value
        for key, value in outputs.items()
    }
    found = detect_centers_with_scores(
        detect_outputs,
        threshold=float(threshold),
        nms_radius=int(nms_radius),
        confidence_score=str(confidence_score),
        center_refinement=str(center_refinement),
        center_refinement_radius=int(center_refinement_radius),
    )[0]
    xy = np.asarray(found["xy"], dtype=np.float32).reshape(-1, 2)
    scores = np.asarray(found["score"], dtype=np.float32).reshape(-1)
    shape = detect_outputs["shape"][0].detach().float().cpu().numpy()
    rows = []
    for idx, (x, y) in enumerate(xy):
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if xi < 0 or yi < 0 or xi >= width or yi >= height:
            continue
        rows.append(
            {
                "x": float(x),
                "y": float(y),
                "score": float(scores[idx]) if idx < len(scores) else float("nan"),
                "major": float(shape[0, yi, xi]),
                "minor": float(shape[1, yi, xi]),
                "theta": float(shape[2, yi, xi]) if shape.shape[0] > 2 else 0.0,
            }
        )
    return rows


def rows_to_reg(rows: Sequence[dict[str, float]], *, shape: bool = True, color: str | None = "cyan") -> list[str]:
    lines = []
    for idx, row in enumerate(rows, start=1):
        row_color = color or SOURCE_CLASS_COLORS.get(int(row.get("class_id", 1)), "yellow")
        label = PU_CLASS_NAMES.get(int(row.get("class_id", 0)), "source")
        if shape:
            lines.append(
                ellipse_line(
                    float(row["x"]),
                    float(row["y"]),
                    float(row.get("major", 1.0)),
                    float(row.get("minor", 1.0)),
                    float(row.get("theta", 0.0)),
                    color=row_color,
                    text=f"id={idx} class={label} score={float(row.get('score', float('nan'))):.3g}",
                )
            )
        else:
            lines.append(point_line(float(row["x"]), float(row["y"]), color=row_color, text=f"id={idx} class={label}"))
    return lines


def prompt_boxes(rows: Sequence[dict[str, float]], *, image_size: int = 512, scale: float = 2.0) -> torch.Tensor:
    boxes = []
    for row in rows:
        a = max(abs(float(row.get("major", 1.0))) * float(scale), 1.0)
        b = max(abs(float(row.get("minor", 1.0))) * float(scale), 1.0)
        theta = float(row.get("theta", 0.0))
        ct, st = math.cos(theta), math.sin(theta)
        dx = math.sqrt((a * ct) ** 2 + (b * st) ** 2) + 1.0
        dy = math.sqrt((a * st) ** 2 + (b * ct) ** 2) + 1.0
        x, y = float(row["x"]), float(row["y"])
        boxes.append([max(0.0, x - dx), max(0.0, y - dy), min(image_size - 1.0, x + dx), min(image_size - 1.0, y + dy)])
    return torch.as_tensor(boxes, dtype=torch.float32)


def build_prompt_masks(
    model: torch.nn.Module,
    outputs: dict[str, torch.Tensor],
    rows: Sequence[dict[str, float]],
    *,
    device: torch.device,
    width: int,
    height: int,
    threshold: float = 0.0,
    box_scale: float = 2.0,
    chunk_size: int = 256,
    multimask: bool = False,
) -> np.ndarray:
    label_map = np.zeros((height, width), dtype=np.int32)
    if not rows or not hasattr(model, "forward_sam_masks"):
        return label_map
    points = torch.as_tensor([[row["x"], row["y"]] for row in rows], dtype=torch.float32, device=device)
    boxes = prompt_boxes(rows, image_size=512, scale=box_scale).to(device=device)
    batch_indices = torch.zeros((len(rows),), dtype=torch.long, device=device)
    with torch.inference_mode():
        low_res, iou = model.forward_sam_masks(
            outputs["image_embeddings"],
            batch_indices,
            points,
            boxes,
            multimask_output=bool(multimask),
            chunk_size=int(chunk_size),
        )
        if bool(multimask) and low_res.shape[1] > 1:
            best = torch.argmax(iou, dim=1)
            idx = torch.arange(low_res.shape[0], device=device)
            low_res = low_res[idx, best][:, None]
        masks = F.interpolate(low_res.float(), size=(512, 512), mode="bilinear", align_corners=False)[:, 0] > float(threshold)
    masks_np = masks.detach().cpu().numpy()[:, :height, :width].astype(bool)
    areas = masks_np.reshape(masks_np.shape[0], -1).sum(axis=1) if len(masks_np) else np.zeros((0,), dtype=np.int64)
    for label, idx in enumerate(np.argsort(-areas), start=1):
        if areas[idx] <= 0:
            continue
        label_map[masks_np[idx]] = label
    return label_map


def mask_overlay(image: np.ndarray, label_map: np.ndarray, *, alpha: float = 0.40, edge_width: int = 1) -> np.ndarray:
    rgb = zscale_rgb(image)
    palette = np.asarray(
        [
            (0.00, 0.82, 1.00),
            (1.00, 0.15, 0.55),
            (1.00, 0.85, 0.05),
            (0.20, 0.85, 0.20),
            (0.65, 0.40, 1.00),
            (1.00, 0.45, 0.10),
        ],
        dtype=np.float32,
    )
    for label in sorted([int(v) for v in np.unique(label_map) if int(v) > 0], key=lambda lab: int((label_map == lab).sum()), reverse=True):
        color = palette[(label - 1) % len(palette)]
        mask = label_map == label
        rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * color
        boundary = mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
        if edge_width > 1:
            boundary = ndimage.binary_dilation(boundary, iterations=int(edge_width) - 1)
        rgb[boundary] = np.asarray((1.0, 1.0, 1.0), dtype=np.float32)
    return rgb


def write_mask_reg(path: Path, label_map: np.ndarray) -> None:
    lines = []
    for label in [int(v) for v in np.unique(label_map) if int(v) > 0]:
        mask = label_map == label
        for contour in contours(mask):
            lines.append(polygon_line(contour, color="green", width=2, text=f"id={label} area={int(mask.sum())}"))
    write_reg(path, lines)


def write_sources_csv(path: Path, rows: Sequence[dict[str, float]], *, header: fits.Header | None = None, x0: int = 0, y0: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    world = np.full((len(rows), 2), np.nan, dtype=np.float64)
    if header is not None and header.get("CTYPE1") and header.get("CTYPE2") and len(rows):
        try:
            xy = np.asarray([[x0 + float(row["x"]), y0 + float(row["y"])] for row in rows], dtype=np.float64)
            world = WCS(header).celestial.all_pix2world(xy, 0)
        except Exception:
            pass
    fields = ("id", "x", "y", "x_image", "y_image", "ra_deg", "dec_deg", "score", "major", "minor", "theta")
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "id": idx,
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "x_image": x0 + float(row["x"]),
                    "y_image": y0 + float(row["y"]),
                    "ra_deg": float(world[idx - 1, 0]),
                    "dec_deg": float(world[idx - 1, 1]),
                    "score": float(row.get("score", float("nan"))),
                    "major": float(row.get("major", float("nan"))),
                    "minor": float(row.get("minor", float("nan"))),
                    "theta": float(row.get("theta", float("nan"))),
                }
            )


def shifted_crop_header(header: fits.Header | None, *, x0: int, y0: int) -> fits.Header:
    out = fits.Header() if header is None else header.copy()
    if "CRPIX1" in out:
        out["CRPIX1"] = float(out["CRPIX1"]) - float(x0)
    if "CRPIX2" in out:
        out["CRPIX2"] = float(out["CRPIX2"]) - float(y0)
    if "LTV1" in out:
        out["LTV1"] = float(out["LTV1"]) + float(x0)
    if "LTV2" in out:
        out["LTV2"] = float(out["LTV2"]) + float(y0)
    return out


def load_native_sam(model_type: str, checkpoint: Path, device: str):
    model = sam_model_registry[str(model_type)](checkpoint=str(checkpoint))
    model.to(device=device)
    model.eval()
    return model


def run_native_sam_amg(
    image_uint8: np.ndarray,
    *,
    model_type: str,
    checkpoint: Path,
    device: str,
    points_per_side: int,
    points_per_batch: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    crop_n_layers: int,
    min_mask_region_area: int,
) -> list[dict]:
    model = load_native_sam(model_type, checkpoint, device)
    generator = SamAutomaticMaskGenerator(
        model,
        points_per_side=int(points_per_side),
        points_per_batch=int(points_per_batch),
        pred_iou_thresh=float(pred_iou_thresh),
        stability_score_thresh=float(stability_score_thresh),
        crop_n_layers=int(crop_n_layers),
        min_mask_region_area=int(min_mask_region_area),
        output_mode="binary_mask",
    )
    return generator.generate(image_uint8)


def labelmap_from_amg(annotations: Sequence[dict], *, height: int, width: int) -> np.ndarray:
    label_map = np.zeros((height, width), dtype=np.int32)
    ordered = sorted(annotations, key=lambda ann: int(ann.get("area", 0)), reverse=True)
    for label, ann in enumerate(ordered, start=1):
        mask = np.asarray(ann["segmentation"], dtype=bool)[:height, :width]
        if bool(mask.any()):
            label_map[mask] = label
    return label_map
