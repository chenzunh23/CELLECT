#!/usr/bin/env python3
"""Visualize HSC bright-source and source-obscuring MASK planes on calexp images."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from matplotlib.colors import ListedColormap


DEFAULT_BANDS = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y")
DEFAULT_POINTS = ((19185.0, 20720.0), (18442.0, 21902.0))
BRIGHT_PLANES = ("BRIGHT_OBJECT",)
SOURCE_OBSCURING_PLANES = ("BRIGHT_OBJECT", "SAT", "BAD", "NO_DATA", "EDGE", "UNMASKEDNAN")
DIAGNOSTIC_PLANES = (
    "BRIGHT_OBJECT",
    "SAT",
    "BAD",
    "NO_DATA",
    "EDGE",
    "UNMASKEDNAN",
    "SUSPECT",
    "CLIPPED",
    "REJECTED",
    "SENSOR_EDGE",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coadd-root", type=Path, default=Path("/data/shared/Subaru"))
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patch", default="4,5")
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    parser.add_argument("--out-dir", type=Path, default=Path("output/data_filter_0725/bright_source_masks_patch45"))
    parser.add_argument("--points", nargs="*", default=[f"{x},{y}" for x, y in DEFAULT_POINTS])
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=512)
    return parser.parse_args()


def _find_image_hdu_index(hdul: fits.HDUList) -> int:
    if "IMAGE" in hdul:
        return hdul.index_of("IMAGE")
    for idx, hdu in enumerate(hdul):
        data = getattr(hdu, "data", None)
        if data is not None and getattr(data, "ndim", None) == 2:
            return idx
    raise KeyError("no 2D image HDU found")


def _plane_indices(hdul: fits.HDUList) -> dict[str, int]:
    if all(name in hdul for name in ("IMAGE", "MASK", "VARIANCE")):
        return {name: hdul.index_of(name) for name in ("IMAGE", "MASK", "VARIANCE")}
    image_idx = _find_image_hdu_index(hdul)
    out = {"IMAGE": image_idx}
    shape = hdul[image_idx].data.shape
    for name, idx in (("MASK", image_idx + 1), ("VARIANCE", image_idx + 2)):
        if idx < len(hdul):
            data = getattr(hdul[idx], "data", None)
            if data is not None and getattr(data, "ndim", None) == 2 and data.shape == shape:
                out[name] = idx
    return out


def _mask_plane_bits(header: fits.Header) -> dict[str, int]:
    bits: dict[str, int] = {}
    for key, value in header.items():
        text = str(key).upper()
        if not text.startswith("MP_"):
            continue
        try:
            bits[text[3:]] = int(value)
        except Exception:
            continue
    return bits


def _parse_points(values: list[str]) -> list[tuple[float, float]]:
    points = []
    for value in values:
        if not str(value).strip():
            continue
        x_text, y_text = str(value).split(",", 1)
        points.append((float(x_text), float(y_text)))
    return points


def _mask_for_planes(mask: np.ndarray, bits: dict[str, int], planes: tuple[str, ...]) -> np.ndarray:
    values = mask.astype(np.int64, copy=False)
    out = np.zeros(mask.shape, dtype=bool)
    for plane in planes:
        bit = bits.get(plane)
        if bit is not None:
            out |= (values & (1 << int(bit))) != 0
    return out


def _downsample_bool(mask: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return mask
    h, w = mask.shape
    hh = h // factor
    ww = w // factor
    trimmed = mask[: hh * factor, : ww * factor]
    return trimmed.reshape(hh, factor, ww, factor).any(axis=(1, 3))


def _downsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return image
    h, w = image.shape
    hh = h // factor
    ww = w // factor
    trimmed = image[: hh * factor, : ww * factor]
    return np.nanmean(trimmed.reshape(hh, factor, ww, factor), axis=(1, 3))


def _local_points(points: list[tuple[float, float]], origin: tuple[float, float]) -> list[tuple[float, float]]:
    return [(float(x) - float(origin[0]) + 1.0, float(y) - float(origin[1]) + 1.0) for x, y in points]


def _show_points(ax, points: list[tuple[float, float]], *, downsample: int, x0: float = 0.0, y0: float = 0.0) -> None:
    for idx, (x, y) in enumerate(points, start=1):
        px = (float(x) - float(x0)) / float(downsample)
        py = (float(y) - float(y0)) / float(downsample)
        ax.plot(px, py, marker="o", ms=8, mfc="none", mec="cyan", mew=1.5)
        ax.text(px + 6, py + 6, f"P{idx}", color="cyan", fontsize=9, weight="bold")


def _zscale_image(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    finite = np.asarray(image[np.isfinite(image)], dtype=np.float32)
    if finite.size == 0:
        return image, 0.0, 1.0
    vmin, vmax = ZScaleInterval().get_limits(finite)
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanpercentile(finite, 1)), float(np.nanpercentile(finite, 99))
    return image, float(vmin), float(vmax)


def _plot_full(
    *,
    band: str,
    image: np.ndarray,
    mask: np.ndarray,
    bits: dict[str, int],
    local_points: list[tuple[float, float]],
    out_path: Path,
    downsample: int,
) -> None:
    image_ds = _downsample_image(image, downsample)
    _, vmin, vmax = _zscale_image(image_ds)
    bright = _downsample_bool(_mask_for_planes(mask, bits, BRIGHT_PLANES), downsample)
    source_obscuring = _downsample_bool(_mask_for_planes(mask, bits, SOURCE_OBSCURING_PLANES), downsample)
    sat = _downsample_bool(_mask_for_planes(mask, bits, ("SAT",)), downsample)
    bad_union = _downsample_bool(_mask_for_planes(mask, bits, ("BAD", "NO_DATA", "EDGE", "UNMASKEDNAN")), downsample)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), constrained_layout=True)
    panels = [
        ("zscale + BRIGHT_OBJECT", bright, "Reds"),
        ("zscale + source-obscuring union", source_obscuring, "autumn"),
        ("BRIGHT(red) SAT(cyan) bad/edge(magenta)", None, None),
    ]
    for ax, (title, overlay, cmap) in zip(axes, panels):
        ax.imshow(image_ds, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        if overlay is not None:
            ax.imshow(np.ma.masked_where(~overlay, overlay), origin="lower", cmap=cmap, alpha=0.45)
        else:
            ax.imshow(np.ma.masked_where(~bright, bright), origin="lower", cmap=ListedColormap(["red"]), alpha=0.42)
            ax.imshow(np.ma.masked_where(~sat, sat), origin="lower", cmap=ListedColormap(["cyan"]), alpha=0.42)
            ax.imshow(np.ma.masked_where(~bad_union, bad_union), origin="lower", cmap=ListedColormap(["magenta"]), alpha=0.42)
        _show_points(ax, local_points, downsample=downsample)
        ax.set_title(f"{band} {title}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_crops(
    *,
    band: str,
    image: np.ndarray,
    mask: np.ndarray,
    bits: dict[str, int],
    local_points: list[tuple[float, float]],
    out_dir: Path,
    crop_size: int,
) -> None:
    source_obscuring = _mask_for_planes(mask, bits, SOURCE_OBSCURING_PLANES)
    bright = _mask_for_planes(mask, bits, BRIGHT_PLANES)
    sat = _mask_for_planes(mask, bits, ("SAT",))
    bad_union = _mask_for_planes(mask, bits, ("BAD", "NO_DATA", "EDGE", "UNMASKEDNAN"))
    h, w = image.shape
    half = int(crop_size) // 2
    for idx, (px, py) in enumerate(local_points, start=1):
        cx = int(round(px - 1.0))
        cy = int(round(py - 1.0))
        x0 = max(0, min(w - int(crop_size), cx - half))
        y0 = max(0, min(h - int(crop_size), cy - half))
        x1 = min(w, x0 + int(crop_size))
        y1 = min(h, y0 + int(crop_size))
        img = image[y0:y1, x0:x1]
        _, vmin, vmax = _zscale_image(img)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        overlays = [
            ("BRIGHT_OBJECT", bright[y0:y1, x0:x1], "Reds"),
            ("source-obscuring union", source_obscuring[y0:y1, x0:x1], "autumn"),
            ("BRIGHT/SAT/bad", None, None),
        ]
        for ax, (title, overlay, cmap) in zip(axes, overlays):
            ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
            if overlay is not None:
                ax.imshow(np.ma.masked_where(~overlay, overlay), origin="lower", cmap=cmap, alpha=0.45)
            else:
                b = bright[y0:y1, x0:x1]
                s = sat[y0:y1, x0:x1]
                q = bad_union[y0:y1, x0:x1]
                ax.imshow(np.ma.masked_where(~b, b), origin="lower", cmap=ListedColormap(["red"]), alpha=0.42)
                ax.imshow(np.ma.masked_where(~s, s), origin="lower", cmap=ListedColormap(["cyan"]), alpha=0.42)
                ax.imshow(np.ma.masked_where(~q, q), origin="lower", cmap=ListedColormap(["magenta"]), alpha=0.42)
            _show_points(ax, [(px, py)], downsample=1, x0=x0, y0=y0)
            ax.set_title(f"{band} P{idx} {title}")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.savefig(out_dir / f"{band}_point{idx}_bright_mask_crop.png", dpi=180)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    points = _parse_points(args.points)
    summary_rows = []

    for band in args.bands:
        path = args.coadd_root / str(args.tract) / band / str(args.patch) / f"calexp-{band}-{args.tract}-{args.patch}.fits"
        if not path.exists():
            raise FileNotFoundError(path)
        with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
            planes = _plane_indices(hdul)
            image_hdu = hdul[planes["IMAGE"]]
            mask_hdu = hdul[planes["MASK"]]
            image = np.asarray(image_hdu.data, dtype=np.float32)
            mask = np.asarray(mask_hdu.data)
            origin = (-float(mask_hdu.header.get("LTV1", image_hdu.header.get("LTV1", 0.0))),
                      -float(mask_hdu.header.get("LTV2", image_hdu.header.get("LTV2", 0.0))))
            bits = _mask_plane_bits(mask_hdu.header)

        local = _local_points(points, origin)
        _plot_full(
            band=band,
            image=image,
            mask=mask,
            bits=bits,
            local_points=local,
            out_path=args.out_dir / f"{band}_{args.tract}_{args.patch.replace(',', '_')}_bright_source_mask.png",
            downsample=max(1, int(args.downsample)),
        )
        _plot_crops(
            band=band,
            image=image,
            mask=mask,
            bits=bits,
            local_points=local,
            out_dir=args.out_dir,
            crop_size=int(args.crop_size),
        )

        values = mask.astype(np.int64, copy=False)
        total = float(mask.size)
        row = {"band": band, "path": str(path), "origin_x": origin[0], "origin_y": origin[1]}
        for plane in DIAGNOSTIC_PLANES:
            bit = bits.get(plane)
            frac = float(np.count_nonzero((values & (1 << int(bit))) != 0) / total) if bit is not None else float("nan")
            row[f"{plane}_frac"] = frac
        for idx, (px, py) in enumerate(local, start=1):
            x_i = int(round(px - 1.0))
            y_i = int(round(py - 1.0))
            if 0 <= y_i < mask.shape[0] and 0 <= x_i < mask.shape[1]:
                pixel = int(values[y_i, x_i])
                active = [plane for plane, bit in bits.items() if (pixel & (1 << int(bit))) != 0]
            else:
                pixel = -1
                active = []
            row[f"point{idx}_x_image"] = px
            row[f"point{idx}_y_image"] = py
            row[f"point{idx}_mask_value"] = pixel
            row[f"point{idx}_planes"] = "|".join(sorted(active))
        summary_rows.append(row)

    fields = sorted({key for row in summary_rows for key in row})
    with (args.out_dir / f"{args.tract}_{args.patch.replace(',', '_')}_bright_mask_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
