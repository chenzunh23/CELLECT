#!/usr/bin/env python3
"""Run a trained CELLECT sam_per_band checkpoint on arbitrary HSC FITS cutouts.

The script crops one region from one or more FITS files, runs the detector,
decodes SAM instance masks from predicted centers/shapes, and writes:

- zscale mask-overlay PNG with explicit mask boundaries
- zscale Kron-shape overlay PNG
- DS9 mask contour REG
- DS9 Kron-shape REG in crop-local image coordinates
- DS9 Kron-shape REG in original-HSC physical coordinates
- source CSV with pixel and optional WCS coordinates
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Sequence

os.environ["MPLCONFIGDIR"] = "/tmp/cellect_mplconfig"

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

from astro_cellect2d import astro_zscale_preprocess  # noqa: E402
from astro_train_ops import detect_centers_with_scores, model_forward_with_batch_context  # noqa: E402
from zangetsu_demo.visualize_sam_cellect import _band_outputs, _make_model  # noqa: E402


REG_HEADER_IMAGE = (
    "# Region file format: DS9 version 4.1",
    'global color=cyan width=2 font="helvetica 10 normal roman" '
    "select=1 highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 source=1",
    "image",
)
REG_HEADER_PHYSICAL = (
    "# Region file format: DS9 version 4.1",
    'global color=cyan width=2 font="helvetica 10 normal roman" '
    "select=1 highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 source=1",
    "physical",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, action="append", required=True, help="Input FITS file. Repeat for multiple bands.")
    p.add_argument("--band", action="append", default=None, help="Band label for each --input; defaults to FITS stem.")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trained CELLECT sam_per_band checkpoint. Required unless --dump-cutout-only is set.",
    )
    p.add_argument("--config", type=Path, default=None, help="run_config.json. Defaults to CHECKPOINT_PARENT/run_config.json if present.")
    p.add_argument("--out-dir", type=Path, default=Path("output/data_filter_0723/hsc_cutout_eval"))
    p.add_argument(
        "--hdu",
        default=None,
        help=(
            "Image HDU index or EXTNAME. Default auto-detects LSST/HSC calexp "
            "IMAGE when present, otherwise the first 2D image-like HDU."
        ),
    )
    p.add_argument(
        "--image-ext",
        default=None,
        help="Alias for --hdu when selecting by EXTNAME, e.g. IMAGE. Useful for calexp files.",
    )
    p.add_argument("--x0", type=float, required=True, help="Crop origin x, 0-index image pixels unless --crop-coordinates=physical.")
    p.add_argument("--y0", type=float, required=True, help="Crop origin y, 0-index image pixels unless --crop-coordinates=physical.")
    p.add_argument("--size", type=int, default=512, help="Square crop size. Values below 512 are padded before inference.")
    p.add_argument("--width", type=int, default=None, help="Optional crop width; defaults to --size.")
    p.add_argument("--height", type=int, default=None, help="Optional crop height; defaults to --size.")
    p.add_argument("--crop-coordinates", choices=("image", "physical"), default="image")
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", choices=("none", "bf16"), default="bf16")
    p.add_argument("--confidence-threshold", type=float, default=None)
    p.add_argument("--confidence-score", choices=("cellect", "raw", "ordinal_prob", "ordinal_expectation"), default=None)
    p.add_argument("--nms-radius", type=int, default=None)
    p.add_argument("--center-refinement", choices=("integer", "weighted_centroid", "softargmax"), default=None)
    p.add_argument("--center-refinement-radius", type=int, default=None)
    p.add_argument("--shape-display-scale", type=float, default=1.0)
    p.add_argument("--prompt-box-scale", type=float, default=2.0)
    p.add_argument("--mask-threshold", type=float, default=0.0)
    p.add_argument("--mask-chunk-size", type=int, default=128)
    p.add_argument("--mask-multimask", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--mask-pred-iou-threshold", type=float, default=None)
    p.add_argument("--min-mask-area", type=int, default=15)
    p.add_argument("--max-mask-area-ratio", type=float, default=0.5)
    p.add_argument("--overlay-alpha", type=float, default=0.38)
    p.add_argument("--edge-color", default="white", help="Mask boundary color name for PNG/REG.")
    p.add_argument("--edge-width", type=float, default=0.45)
    p.add_argument("--png-scale", type=int, default=1, help="Nearest-neighbor PNG scale factor.")
    p.add_argument("--max-contour-vertices", type=int, default=160)
    p.add_argument("--shape-overlay-centers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--dump-cutout-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only read/crop/write cutout FITS files, without loading the model or running inference.",
    )
    return p.parse_args()


def read_config(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text())
    args = dict(payload.get("args", {}))
    args["_top"] = payload
    return args


def _hdu_label(index: int, hdu: fits.hdu.base.ExtensionHDU | fits.PrimaryHDU) -> str:
    name = str(getattr(hdu, "name", "") or hdu.header.get("EXTNAME", "") or "").strip()
    return f"{index}:{name or hdu.__class__.__name__}"


def _parse_hdu_selector(selector: object | None) -> int | str | None:
    if selector is None:
        return None
    text = str(selector).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text.upper()


def _safe_2d_hdu_data(hdu: fits.hdu.base.ExtensionHDU | fits.PrimaryHDU) -> np.ndarray | None:
    """Return a 2D float image if this HDU is image-like, otherwise ``None``.

    Some HSC/LSST FITS products do not expose a named ``IMAGE`` extension.  In
    those cases a calexp image may be the primary HDU or an unnamed
    ImageHDU/CompImageHDU.  We avoid table HDUs here because touching truncated
    table data can raise buffer/reshape errors unrelated to the image plane.
    """

    class_name = hdu.__class__.__name__
    if class_name not in {"PrimaryHDU", "ImageHDU", "CompImageHDU"}:
        return None
    shape = getattr(hdu, "shape", None)
    if shape is not None and len(tuple(shape)) != 2:
        return None
    try:
        data = hdu.data
    except Exception:
        return None
    if data is None:
        return None
    arr = np.asarray(data).squeeze()
    if arr.ndim != 2:
        return None
    return np.asarray(arr, dtype=np.float32)


def _candidate_image_indices(hdul: fits.HDUList, selector: int | str | None) -> list[int]:
    if isinstance(selector, int):
        return [int(selector)]
    if isinstance(selector, str):
        if selector in hdul:
            return [int(hdul.index_of(selector))]
        lowered = selector.lower()
        matches = [
            idx
            for idx, hdu in enumerate(hdul)
            if str(getattr(hdu, "name", "") or hdu.header.get("EXTNAME", "") or "").strip().lower() == lowered
        ]
        if matches:
            return matches
        raise KeyError(f"Extension {selector!r} not found. Available HDUs: {', '.join(_hdu_label(i, h) for i, h in enumerate(hdul))}")

    preferred: list[int] = []
    for name in ("IMAGE", "SCI", "PRIMARY"):
        try:
            if name == "PRIMARY":
                preferred.append(0)
            elif name in hdul:
                preferred.append(int(hdul.index_of(name)))
        except Exception:
            pass
    seen = set()
    ordered = []
    for idx in [*preferred, *range(len(hdul))]:
        if idx not in seen:
            ordered.append(int(idx))
            seen.add(int(idx))
    return ordered


def first_image_hdu(path: Path, hdu: object | None) -> tuple[np.ndarray, fits.Header, int]:
    selector = _parse_hdu_selector(hdu)
    errors: list[str] = []
    with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
        candidates = _candidate_image_indices(hdul, selector)
        for index in candidates:
            if index < 0 or index >= len(hdul):
                errors.append(f"{index}: out of range")
                continue
            arr = _safe_2d_hdu_data(hdul[index])
            if arr is None:
                errors.append(f"{_hdu_label(index, hdul[index])}: not a readable 2D image HDU")
                continue
            return arr, hdul[index].header.copy(), int(index)
        available = ", ".join(_hdu_label(i, h) for i, h in enumerate(hdul))
    detail = "; ".join(errors[:8])
    raise ValueError(f"no readable 2D image HDU found in {path}; available HDUs: {available}; tried: {detail}")


def physical_origin_from_header(header: fits.Header) -> tuple[float, float]:
    return -float(header.get("LTV1", 0.0)), -float(header.get("LTV2", 0.0))


def crop_image(
    image: np.ndarray,
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
    inference_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    if width > inference_size or height > inference_size:
        raise ValueError(f"crop width/height must be <= {inference_size}; got {width}x{height}")
    out = np.full((inference_size, inference_size), np.nan, dtype=np.float32)
    invalid = np.ones((inference_size, inference_size), dtype=bool)
    src_x0 = max(0, int(x0))
    src_y0 = max(0, int(y0))
    src_x1 = min(image.shape[1], int(x0) + int(width))
    src_y1 = min(image.shape[0], int(y0) + int(height))
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0 = src_x0 - int(x0)
        dst_y0 = src_y0 - int(y0)
        view = image[src_y0:src_y1, src_x0:src_x1]
        out[dst_y0 : dst_y0 + view.shape[0], dst_x0 : dst_x0 + view.shape[1]] = view
        invalid[dst_y0 : dst_y0 + view.shape[0], dst_x0 : dst_x0 + view.shape[1]] = ~np.isfinite(view)
    return out, invalid, (src_x0, src_y0, src_x1, src_y1)


def shifted_crop_header(header: fits.Header, *, crop_x0: int, crop_y0: int) -> fits.Header:
    out = header.copy()
    if "CRPIX1" in out:
        out["CRPIX1"] = float(out["CRPIX1"]) - float(crop_x0)
    if "CRPIX2" in out:
        out["CRPIX2"] = float(out["CRPIX2"]) - float(crop_y0)
    if "LTV1" in out:
        out["LTV1"] = float(out["LTV1"]) + float(crop_x0)
    if "LTV2" in out:
        out["LTV2"] = float(out["LTV2"]) + float(crop_y0)
    return out


def zscale_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not bool(finite.any()):
        return np.zeros_like(image, dtype=np.float32)
    try:
        lo, hi = ZScaleInterval().get_limits(image[finite])
    except Exception:
        lo, hi = np.nanpercentile(image[finite], [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def instance_rgb(label: int) -> np.ndarray:
    palette = np.asarray(
        [
            (0.00, 0.76, 0.94),
            (0.95, 0.18, 0.65),
            (1.00, 0.82, 0.12),
            (0.16, 0.72, 0.33),
            (0.18, 0.38, 1.00),
            (0.95, 0.22, 0.14),
            (0.90, 0.56, 0.12),
            (0.58, 0.32, 0.90),
        ],
        dtype=np.float32,
    )
    return palette[(int(label) - 1) % len(palette)]


def color_rgb(name: str) -> np.ndarray:
    table = {
        "white": (1.0, 1.0, 1.0),
        "yellow": (1.0, 0.90, 0.05),
        "cyan": (0.0, 0.9, 1.0),
        "green": (0.1, 1.0, 0.2),
        "red": (1.0, 0.1, 0.05),
    }
    return np.asarray(table.get(str(name).lower(), table["white"]), dtype=np.float32)


def png_from_rgb(path: Path, rgb: np.ndarray, *, scale: int = 2) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.clip(np.rint(np.flip(rgb, axis=0) * 255.0), 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels, mode="RGB")
    if scale > 1:
        img = img.resize((pixels.shape[1] * scale, pixels.shape[0] * scale), resample=Image.Resampling.NEAREST)
    img.save(path, format="PNG", compress_level=3)


def shape_rows(pred_xy: np.ndarray, shape: np.ndarray, scale: float, valid_width: int, valid_height: int) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for pred_index, (x, y) in enumerate(np.asarray(pred_xy, dtype=np.float32)):
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if xi < 0 or yi < 0 or xi >= valid_width or yi >= valid_height:
            continue
        rows.append(
            {
                "pred_index": float(pred_index),
                "x": float(x),
                "y": float(y),
                "major": float(shape[0, yi, xi]) * float(scale),
                "minor": float(shape[1, yi, xi]) * float(scale),
                "theta": float(shape[2, yi, xi]) if shape.shape[0] > 2 else 0.0,
            }
        )
    return rows


def prompt_boxes(rows: Sequence[dict[str, float]], image_size: int, scale: float) -> torch.Tensor:
    boxes: list[list[float]] = []
    for row in rows:
        a = max(abs(float(row["major"])) * float(scale), 1.0)
        b = max(abs(float(row["minor"])) * float(scale), 1.0)
        theta = float(row.get("theta", 0.0))
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        dx = math.sqrt((a * cos_t) ** 2 + (b * sin_t) ** 2) + 1.0
        dy = math.sqrt((a * sin_t) ** 2 + (b * cos_t) ** 2) + 1.0
        x, y = float(row["x"]), float(row["y"])
        boxes.append([max(0.0, x - dx), max(0.0, y - dy), min(image_size - 1.0, x + dx), min(image_size - 1.0, y + dy)])
    return torch.as_tensor(boxes, dtype=torch.float32)


def contours(mask: np.ndarray, max_vertices: int) -> Iterable[list[tuple[float, float]]]:
    try:
        from skimage import measure

        for contour in measure.find_contours(mask.astype(np.float32), 0.5):
            if contour.shape[0] < 3:
                continue
            step = max(1, int(math.ceil(contour.shape[0] / max_vertices)))
            pts = contour[::step]
            if pts.shape[0] >= 3:
                yield [(float(x + 1.0), float(y + 1.0)) for y, x in pts]
        return
    except Exception:
        pass
    ys, xs = np.where(mask)
    if ys.size:
        yield [
            (float(xs.min() + 1), float(ys.min() + 1)),
            (float(xs.max() + 2), float(ys.min() + 1)),
            (float(xs.max() + 2), float(ys.max() + 2)),
            (float(xs.min() + 1), float(ys.max() + 2)),
        ]


def polygon_line(points: Sequence[tuple[float, float]], color: str, width: int, text: str = "") -> str:
    coords = ",".join(f"{x:.2f},{y:.2f}" for x, y in points)
    suffix = f" # color={color} width={width}"
    if text:
        suffix += f" text={{{text}}}"
    return f"polygon({coords}){suffix}"


def ellipse_line(x: float, y: float, major: float, minor: float, theta: float, color: str, width: int, text: str = "") -> str:
    suffix = f" # color={color} width={width}"
    if text:
        suffix += f" text={{{text}}}"
    return f"ellipse({x:.3f},{y:.3f},{major:.3f},{minor:.3f},{math.degrees(theta):.3f}){suffix}"


def draw_shape_overlay(path: Path, image: np.ndarray, rows: Sequence[dict[str, float]], *, draw_centers: bool = True) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    base = zscale_image(image)
    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=160)
    ax.imshow(base, origin="lower", cmap="gray", vmin=0.0, vmax=1.0)
    order = sorted(rows, key=lambda row: abs(float(row["major"]) * float(row["minor"])), reverse=True)
    for idx, row in enumerate(order, start=1):
        ax.add_patch(
            Ellipse(
                (float(row["x"]), float(row["y"])),
                width=2.0 * max(abs(float(row["major"])), 1.0),
                height=2.0 * max(abs(float(row["minor"])), 1.0),
                angle=math.degrees(float(row["theta"])),
                fill=False,
                edgecolor="cyan",
                linewidth=1.1,
                alpha=0.9,
            )
        )
        if draw_centers:
            ax.plot([float(row["x"])], [float(row["y"])], marker="+", color="yellow", ms=3.5, mew=0.8)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.set_axis_off()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def write_regs(
    prefix: Path,
    rows: Sequence[dict[str, float]],
    *,
    crop_x0_image: int,
    crop_y0_image: int,
    physical_origin: tuple[float, float],
) -> tuple[Path, Path]:
    image_lines = list(REG_HEADER_IMAGE) + ["# crop-local image coordinates"]
    physical_lines = list(REG_HEADER_PHYSICAL) + ["# original-image physical coordinates"]
    ox, oy = physical_origin
    for idx, row in enumerate(rows, start=1):
        local_x = float(row["x"]) + 1.0
        local_y = float(row["y"]) + 1.0
        phys_x = ox + crop_x0_image + float(row["x"])
        phys_y = oy + crop_y0_image + float(row["y"])
        major = max(abs(float(row["major"])), 1.0)
        minor = max(abs(float(row["minor"])), 1.0)
        theta = float(row["theta"])
        image_lines.append(ellipse_line(local_x, local_y, major, minor, theta, "cyan", 2, text=f"id={idx}"))
        physical_lines.append(ellipse_line(phys_x, phys_y, major, minor, theta, "cyan", 2, text=f"id={idx}"))
    image_path = prefix.with_name(prefix.name + "_shape_kron_image.reg")
    physical_path = prefix.with_name(prefix.name + "_shape_kron_physical.reg")
    image_path.write_text("\n".join(image_lines) + "\n", encoding="ascii")
    physical_path.write_text("\n".join(physical_lines) + "\n", encoding="ascii")
    return image_path, physical_path


def write_mask_reg(path: Path, label_map: np.ndarray, *, max_vertices: int, edge_width: float) -> None:
    lines = list(REG_HEADER_IMAGE) + ["# crop-local mask contours"]
    for label in [int(value) for value in np.unique(label_map) if int(value) > 0]:
        mask = label_map == label
        area = int(mask.sum())
        for contour in contours(mask, max_vertices):
            lines.append(polygon_line(contour, "green", max(1, int(round(float(edge_width)))), text=f"id={label} area={area}"))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def mask_overlay_rgb(image: np.ndarray, label_map: np.ndarray, alpha: float) -> np.ndarray:
    base = zscale_image(image)
    rgb = np.repeat(base[..., None], 3, axis=2)
    for label in sorted([int(value) for value in np.unique(label_map) if int(value) > 0], key=lambda lab: int((label_map == lab).sum()), reverse=True):
        mask = label_map == label
        rgb[mask] = (1.0 - float(alpha)) * rgb[mask] + float(alpha) * instance_rgb(label)
    return rgb


def save_mask_overlay_png(
    path: Path,
    image: np.ndarray,
    label_map: np.ndarray,
    *,
    alpha: float,
    edge_color: str,
    edge_width: float,
    png_scale: int,
) -> None:
    rgb = mask_overlay_rgb(image, label_map, alpha)
    boundary = np.zeros(label_map.shape, dtype=bool)
    for label in [int(value) for value in np.unique(label_map) if int(value) > 0]:
        mask = label_map == label
        boundary |= mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    extra = max(0, int(round(float(edge_width))) - 1)
    if extra > 0:
        boundary = ndimage.binary_dilation(boundary, iterations=extra)
    rgb[boundary] = color_rgb(edge_color)
    png_from_rgb(path, rgb, scale=max(1, int(png_scale)))


def write_csv(path: Path, rows: Sequence[dict[str, float]], *, crop_x0: int, crop_y0: int, origin: tuple[float, float], header: fits.Header) -> None:
    physical = np.asarray([[origin[0] + crop_x0 + row["x"], origin[1] + crop_y0 + row["y"]] for row in rows], dtype=np.float64)
    world = np.full((len(rows), 2), np.nan, dtype=np.float64)
    if len(rows) and header.get("CTYPE1") and header.get("CTYPE2"):
        try:
            # WCS wants local image pixels in the FITS array coordinate system.
            image_xy = np.asarray([[crop_x0 + row["x"], crop_y0 + row["y"]] for row in rows], dtype=np.float64)
            world = WCS(header).celestial.all_pix2world(image_xy, 0)
        except Exception:
            pass
    fields = ("id", "x_crop", "y_crop", "x_image0", "y_image0", "x_physical", "y_physical", "ra_deg", "dec_deg", "score", "major", "minor", "theta_rad")
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "id": idx,
                    "x_crop": float(row["x"]),
                    "y_crop": float(row["y"]),
                    "x_image0": crop_x0 + float(row["x"]),
                    "y_image0": crop_y0 + float(row["y"]),
                    "x_physical": float(physical[idx - 1, 0]),
                    "y_physical": float(physical[idx - 1, 1]),
                    "ra_deg": float(world[idx - 1, 0]),
                    "dec_deg": float(world[idx - 1, 1]),
                    "score": float(row["score"]),
                    "major": float(row["major"]),
                    "minor": float(row["minor"]),
                    "theta_rad": float(row["theta"]),
                }
            )


def infer_one(
    *,
    model: torch.nn.Module | None,
    device: torch.device,
    cfg: dict,
    args: argparse.Namespace,
    fits_path: Path,
    band: str,
) -> dict[str, object]:
    image, header, hdu_index = first_image_hdu(fits_path, args.image_ext or args.hdu)
    origin = physical_origin_from_header(header)
    width = int(args.width or args.size)
    height = int(args.height or args.size)
    if args.crop_coordinates == "physical":
        crop_x0 = int(round(float(args.x0) - origin[0]))
        crop_y0 = int(round(float(args.y0) - origin[1]))
    else:
        crop_x0 = int(round(float(args.x0)))
        crop_y0 = int(round(float(args.y0)))
    crop, invalid, source_bounds = crop_image(image, x0=crop_x0, y0=crop_y0, width=width, height=height)
    cleaned = np.nan_to_num(crop, nan=0.0, posinf=0.0, neginf=0.0)
    out_dir = args.out_dir / f"{fits_path.stem}_x{crop_x0}_y{crop_y0}_{band}"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"{fits_path.stem}_x{crop_x0}_y{crop_y0}_{band}"
    crop_header = shifted_crop_header(header, crop_x0=crop_x0, crop_y0=crop_y0)
    fits.PrimaryHDU(cleaned[:height, :width], header=crop_header).writeto(prefix.with_name(prefix.name + "_cutout.fits"), overwrite=True)
    if args.dump_cutout_only:
        summary = {
            "input": str(fits_path),
            "band": band,
            "hdu": hdu_index,
            "crop_x0_image": crop_x0,
            "crop_y0_image": crop_y0,
            "physical_origin": list(origin),
            "source_bounds": list(source_bounds),
            "detections": 0,
            "mask_instances": 0,
            "cutout_only": True,
            "out_dir": str(out_dir),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
        print(f"[cutout] {band} {fits_path.name}: hdu={hdu_index} out={out_dir}", flush=True)
        return summary
    if model is None:
        raise ValueError("model is required unless --dump-cutout-only is set")
    work = crop.copy()
    work[invalid] = np.nan
    normalized = astro_zscale_preprocess(work[None]).to(dtype=torch.float32)
    normalized[0][torch.from_numpy(invalid)] = 0.0
    batch = normalized[None].to(device=device, dtype=torch.float32)
    context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" and args.amp == "bf16" else nullcontext()
    with torch.inference_mode(), context:
        outputs = model_forward_with_batch_context(model, batch, {"processing_id": torch.zeros(1, dtype=torch.long, device=device)})
        one_band = _band_outputs(outputs, 0)
        detection_outputs = {key: value.float() if key in {"confidence", "seg_logits"} else value for key, value in one_band.items()}
        found = detect_centers_with_scores(
            detection_outputs,
            threshold=float(args.confidence_threshold if args.confidence_threshold is not None else cfg.get("confidence_threshold", 2.0)),
            nms_radius=int(args.nms_radius if args.nms_radius is not None else cfg.get("nms_radius", 1)),
            confidence_score=str(args.confidence_score or cfg.get("confidence_score", "ordinal_expectation")),
            center_refinement=str(args.center_refinement or cfg.get("center_refinement", "softargmax")),
            center_refinement_radius=int(args.center_refinement_radius if args.center_refinement_radius is not None else cfg.get("center_refinement_radius", 1)),
        )
    xy = found[0]["xy"].detach().float().cpu().numpy() if torch.is_tensor(found[0]["xy"]) else np.asarray(found[0]["xy"], dtype=np.float32)
    score = found[0]["score"].detach().float().cpu().numpy() if torch.is_tensor(found[0]["score"]) else np.asarray(found[0]["score"], dtype=np.float32)
    shape_map = one_band["shape"][0].detach().float().cpu().numpy()
    rows = shape_rows(xy, shape_map, float(args.shape_display_scale), width, height)
    for row in rows:
        pred_index = int(row["pred_index"])
        row["score"] = float(score[pred_index]) if 0 <= pred_index < len(score) else float("nan")

    label_map = np.zeros((512, 512), dtype=np.int32)
    mask_ious = np.zeros((0,), dtype=np.float32)
    if rows:
        if not hasattr(model, "forward_sam_masks"):
            raise RuntimeError("checkpoint/model has no forward_sam_masks; instance mask output requires sam_per_band")
        points = torch.as_tensor([[row["x"], row["y"]] for row in rows], device=device, dtype=torch.float32)
        boxes = prompt_boxes(rows, 512, float(args.prompt_box_scale)).to(device=device)
        batch_indices = torch.zeros((len(rows),), device=device, dtype=torch.long)
        with torch.inference_mode(), context:
            low_res, predicted_iou = model.forward_sam_masks(
                outputs["image_embeddings"],
                batch_indices,
                points,
                boxes,
                multimask_output=bool(args.mask_multimask),
                chunk_size=int(args.mask_chunk_size),
            )
            if bool(args.mask_multimask) and low_res.shape[1] > 1:
                best = torch.argmax(predicted_iou, dim=1)
                idx = torch.arange(low_res.shape[0], device=device)
                low_res = low_res[idx, best][:, None]
                predicted_iou = predicted_iou[idx, best][:, None]
            full_res = F.interpolate(low_res.float(), size=(512, 512), mode="bilinear", align_corners=False)[:, 0]
        masks = (full_res > float(args.mask_threshold)).detach().cpu().numpy()
        mask_ious = predicted_iou[:, 0].detach().float().cpu().numpy()
        max_area = float(args.max_mask_area_ratio) * float(width * height)
        candidates: list[tuple[int, np.ndarray, float, int]] = []
        valid_area_region = np.zeros((512, 512), dtype=bool)
        valid_area_region[:height, :width] = True
        for index, mask in enumerate(masks, start=1):
            mask = np.asarray(mask, dtype=bool) & valid_area_region & ~invalid
            area = int(mask.sum())
            if area < int(args.min_mask_area) or area > max_area:
                continue
            iou = float(mask_ious[index - 1])
            if args.mask_pred_iou_threshold is not None and iou < float(args.mask_pred_iou_threshold):
                continue
            candidates.append((index, mask, iou, area))
        for new_label, (_old_label, mask, _iou, _area) in enumerate(sorted(candidates, key=lambda item: item[3], reverse=True), start=1):
            label_map[mask] = new_label

    fits.PrimaryHDU(label_map[:height, :width], header=crop_header).writeto(prefix.with_name(prefix.name + "_mask_labelmap.fits"), overwrite=True)
    write_mask_reg(
        prefix.with_name(prefix.name + "_mask_contours_image.reg"),
        label_map[:height, :width],
        max_vertices=int(args.max_contour_vertices),
        edge_width=float(args.edge_width),
    )
    write_regs(prefix, rows, crop_x0_image=crop_x0, crop_y0_image=crop_y0, physical_origin=origin)
    write_csv(prefix.with_name(prefix.name + "_sources.csv"), rows, crop_x0=crop_x0, crop_y0=crop_y0, origin=origin, header=header)
    save_mask_overlay_png(
        prefix.with_name(prefix.name + "_mask_overlay.png"),
        cleaned[:height, :width],
        label_map[:height, :width],
        alpha=float(args.overlay_alpha),
        edge_color=str(args.edge_color),
        edge_width=float(args.edge_width),
        png_scale=int(args.png_scale),
    )
    draw_shape_overlay(
        prefix.with_name(prefix.name + "_kron_shape_overlay.png"),
        cleaned[:height, :width],
        rows,
        draw_centers=bool(args.shape_overlay_centers),
    )
    summary = {
        "input": str(fits_path),
        "band": band,
        "hdu": hdu_index,
        "crop_x0_image": crop_x0,
        "crop_y0_image": crop_y0,
        "physical_origin": list(origin),
        "source_bounds": list(source_bounds),
        "detections": len(rows),
        "mask_instances": int(len([v for v in np.unique(label_map) if int(v) > 0])),
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(f"[done] {band} {fits_path.name}: detections={summary['detections']} masks={summary['mask_instances']} out={out_dir}", flush=True)
    return summary


def main() -> int:
    args = parse_args()
    if args.width is None:
        args.width = int(args.size)
    if args.height is None:
        args.height = int(args.size)
    if args.width > 512 or args.height > 512:
        raise ValueError("This script evaluates one SAM-sized crop; use width/height <= 512.")
    args.input = [path.expanduser().resolve() for path in args.input]
    args.out_dir = args.out_dir.expanduser().resolve()
    if args.dump_cutout_only:
        cfg = {}
        model = None
        device = torch.device("cpu")
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required unless --dump-cutout-only is set")
        args.checkpoint = args.checkpoint.expanduser().resolve()
        config = args.config.expanduser().resolve() if args.config else args.checkpoint.parent / "run_config.json"
        cfg = read_config(config if config.exists() else None)
        device = torch.device(args.device)
        model = _make_model(cfg, args.checkpoint, device, ("HSC",))
    bands = args.band or [path.stem for path in args.input]
    if len(bands) != len(args.input):
        raise ValueError("--band must be provided once per --input")
    summaries = [infer_one(model=model, device=device, cfg=cfg, args=args, fits_path=path, band=band) for path, band in zip(args.input, bands)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps({"summaries": summaries}, indent=2) + "\n", encoding="ascii")
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
