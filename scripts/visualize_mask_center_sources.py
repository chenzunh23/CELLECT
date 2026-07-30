#!/usr/bin/env python3
"""Visualize preprocessed sources whose centers fall on selected calexp MASK planes."""

from __future__ import annotations

import argparse
import csv
import math
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import ZScaleInterval
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse


DEFAULT_BANDS = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y")
DEFAULT_POINTS = ((19185.0, 21720.0), (18442.0, 21902.0))
CLASS_DIRS = (
    ("clean", "band_reference_catalogs", "red"),
    ("center_only", "band_reference_center_only", "yellow"),
    ("strict_center_only", "band_reference_strict_center_only", "orange"),
    ("ignore", "band_reference_ignore", "gray"),
)
SOURCE_OBSCURING_PLANES = ("SAT", "BAD", "NO_DATA", "EDGE", "UNMASKEDNAN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coadd-root", type=Path, default=Path("/data/shared/Subaru"))
    parser.add_argument("--preprocessed-root", type=Path, default=Path("/data/czh23/preprocessed"))
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patch", default="4,5")
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    parser.add_argument("--out-dir", type=Path, default=Path("output/data_filter_0725/bright_source_masks_patch45/sources_in_mask"))
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


def _mask_for_planes(mask: np.ndarray, bits: dict[str, int], planes: tuple[str, ...]) -> np.ndarray:
    values = mask.astype(np.int64, copy=False)
    out = np.zeros(mask.shape, dtype=bool)
    for plane in planes:
        bit = bits.get(str(plane).upper())
        if bit is not None:
            out |= (values & (1 << int(bit))) != 0
    return out


def _read_table(path: Path) -> Table:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*'second' did not parse as fits unit.*")
        return Table.read(path, hdu=1, memmap=True)


def _load_sources(preprocessed_root: Path, *, tract: str, patch: str, band: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for class_name, dirname, color in CLASS_DIRS:
        path = preprocessed_root / str(tract) / str(patch) / dirname / band / f"meas-{band}-{tract}-{patch}.fits"
        if not path.exists():
            continue
        table = _read_table(path)
        required = [
            "id",
            "base_SdssCentroid_x",
            "base_SdssCentroid_y",
            "ellipse_major_sigma",
            "ellipse_minor_sigma",
            "ellipse_theta",
        ]
        missing = [name for name in required if name not in table.colnames]
        if missing:
            raise KeyError(f"{path} missing columns: {missing}")
        for row in table:
            rows.append(
                {
                    "id": int(row["id"]),
                    "class": class_name,
                    "color": color,
                    "x": float(row["base_SdssCentroid_x"]),
                    "y": float(row["base_SdssCentroid_y"]),
                    "major": float(row["ellipse_major_sigma"]),
                    "minor": float(row["ellipse_minor_sigma"]),
                    "theta": float(row["ellipse_theta"]),
                }
            )
    return rows


def _select_center_in_mask(
    rows: list[dict[str, object]],
    selected_mask: np.ndarray,
    *,
    origin: tuple[float, float],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    h, w = selected_mask.shape
    for row in rows:
        x_image = float(row["x"]) - float(origin[0]) + 1.0
        y_image = float(row["y"]) - float(origin[1]) + 1.0
        x_idx = int(round(x_image - 1.0))
        y_idx = int(round(y_image - 1.0))
        if 0 <= y_idx < h and 0 <= x_idx < w and bool(selected_mask[y_idx, x_idx]):
            annotated = dict(row)
            annotated["x_image"] = x_image
            annotated["y_image"] = y_image
            out.append(annotated)
    return out


def _zscale_limits(image: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(image[np.isfinite(image)], dtype=np.float32)
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = ZScaleInterval().get_limits(finite)
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanpercentile(finite, 1)), float(np.nanpercentile(finite, 99))
    return float(vmin), float(vmax)


def _downsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return image
    h, w = image.shape
    hh = h // factor
    ww = w // factor
    return np.nanmean(image[: hh * factor, : ww * factor].reshape(hh, factor, ww, factor), axis=(1, 3))


def _downsample_bool(mask: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return mask
    h, w = mask.shape
    hh = h // factor
    ww = w // factor
    return mask[: hh * factor, : ww * factor].reshape(hh, factor, ww, factor).any(axis=(1, 3))


def _draw_ellipses(ax, rows: list[dict[str, object]], *, downsample: int, x0: float = 0.0, y0: float = 0.0) -> None:
    for row in sorted(rows, key=lambda r: float(r["major"]) * float(r["minor"]), reverse=True):
        x = (float(row["x_image"]) - float(x0)) / float(downsample)
        y = (float(row["y_image"]) - float(y0)) / float(downsample)
        major = float(row["major"]) / float(downsample)
        minor = float(row["minor"]) / float(downsample)
        area = math.pi * major * minor
        color = str(row["color"])
        if not np.isfinite(area) or area > 15000.0 / (float(downsample) ** 2):
            ax.plot(x, y, marker="o", ms=5, mfc="none", mec=color, mew=1.0)
            continue
        ellipse = Ellipse(
            (x, y),
            width=2.0 * major,
            height=2.0 * minor,
            angle=math.degrees(float(row["theta"])),
            facecolor="none",
            edgecolor=color,
            linewidth=0.8,
            alpha=0.95,
        )
        ax.add_patch(ellipse)


def _write_reg(path: Path, rows: list[dict[str, object]], *, region_name: str) -> None:
    header = [
        "# Region file format: DS9 version 4.1",
        'global color=red dashlist=8 3 width=2 font="helvetica 10 normal roman" '
        "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
        "image",
    ]
    lines = []
    for row in sorted(rows, key=lambda r: float(r["major"]) * float(r["minor"]), reverse=True):
        x = float(row["x_image"])
        y = float(row["y_image"])
        major = float(row["major"])
        minor = float(row["minor"])
        theta = math.degrees(float(row["theta"]))
        color = str(row["color"])
        text = f"{row['id']} {row['class']} center_in_{region_name}"
        area = math.pi * major * minor
        if not np.isfinite(area) or area > 40000.0:
            lines.append(f"point({x:.3f},{y:.3f}) # point=circle color={color} width=2 text={{{text}}}")
        else:
            lines.append(
                f"ellipse({x:.3f},{y:.3f},{major:.3f},{minor:.3f},{theta:.3f}) "
                f"# color={color} width=2 text={{{text}}}"
            )
    path.write_text("\n".join(header + lines) + "\n", encoding="utf-8")


def _parse_points(values: list[str]) -> list[tuple[float, float]]:
    points = []
    for value in values:
        if not str(value).strip():
            continue
        x_text, y_text = str(value).split(",", 1)
        points.append((float(x_text), float(y_text)))
    return points


def _plot_band(
    *,
    band: str,
    image: np.ndarray,
    sat_mask: np.ndarray,
    obscuring_mask: np.ndarray,
    sat_rows: list[dict[str, object]],
    obscuring_rows: list[dict[str, object]],
    out_path: Path,
    downsample: int,
) -> None:
    image_ds = _downsample_image(image, downsample)
    sat_ds = _downsample_bool(sat_mask, downsample)
    obscuring_ds = _downsample_bool(obscuring_mask, downsample)
    vmin, vmax = _zscale_limits(image_ds)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    panels = [
        ("centers in SAT", sat_ds, sat_rows, "cyan"),
        ("centers in SAT|bad/edge/no-data", obscuring_ds, obscuring_rows, "magenta"),
    ]
    for ax, (title, mask_ds, rows, color) in zip(axes, panels):
        ax.imshow(image_ds, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        ax.imshow(np.ma.masked_where(~mask_ds, mask_ds), origin="lower", cmap=ListedColormap([color]), alpha=0.35)
        _draw_ellipses(ax, rows, downsample=downsample)
        ax.set_title(f"{band} {title} n={len(rows)}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_crops(
    *,
    band: str,
    image: np.ndarray,
    sat_mask: np.ndarray,
    obscuring_mask: np.ndarray,
    sat_rows: list[dict[str, object]],
    obscuring_rows: list[dict[str, object]],
    points: list[tuple[float, float]],
    origin: tuple[float, float],
    out_dir: Path,
    crop_size: int,
) -> None:
    local_points = [(float(x) - float(origin[0]) + 1.0, float(y) - float(origin[1]) + 1.0) for x, y in points]
    h, w = image.shape
    half = int(crop_size) // 2
    all_rows = sat_rows + [row for row in obscuring_rows if row not in sat_rows]
    for idx, (px, py) in enumerate(local_points, start=1):
        cx = int(round(px - 1.0))
        cy = int(round(py - 1.0))
        x0 = max(0, min(w - int(crop_size), cx - half))
        y0 = max(0, min(h - int(crop_size), cy - half))
        x1 = min(w, x0 + int(crop_size))
        y1 = min(h, y0 + int(crop_size))
        crop = image[y0:y1, x0:x1]
        vmin, vmax = _zscale_limits(crop)
        nearby = [
            row for row in all_rows
            if x0 <= float(row["x_image"]) - 1.0 < x1 and y0 <= float(row["y_image"]) - 1.0 < y1
        ]
        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        ax.imshow(crop, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        sat_crop = sat_mask[y0:y1, x0:x1]
        obs_crop = obscuring_mask[y0:y1, x0:x1]
        ax.imshow(np.ma.masked_where(~sat_crop, sat_crop), origin="lower", cmap=ListedColormap(["cyan"]), alpha=0.35)
        ax.imshow(np.ma.masked_where(~(obs_crop & ~sat_crop), obs_crop & ~sat_crop), origin="lower", cmap=ListedColormap(["magenta"]), alpha=0.35)
        _draw_ellipses(ax, nearby, downsample=1, x0=x0 + 1.0, y0=y0 + 1.0)
        ax.plot(px - x0 - 1.0, py - y0 - 1.0, marker="+", ms=12, mec="white", mew=1.5)
        ax.set_title(f"{band} point{idx}: cyan=SAT magenta=bad/edge/no-data")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(out_dir / f"{band}_point{idx}_sources_center_in_mask_crop.png", dpi=180)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    points = _parse_points(args.points)
    summary_rows = []

    for band in args.bands:
        calexp_path = args.coadd_root / str(args.tract) / band / str(args.patch) / f"calexp-{band}-{args.tract}-{args.patch}.fits"
        with fits.open(calexp_path, memmap=True, ignore_missing_end=True) as hdul:
            planes = _plane_indices(hdul)
            image_hdu = hdul[planes["IMAGE"]]
            mask_hdu = hdul[planes["MASK"]]
            image = np.asarray(image_hdu.data, dtype=np.float32)
            mask = np.asarray(mask_hdu.data)
            origin = (
                -float(mask_hdu.header.get("LTV1", image_hdu.header.get("LTV1", 0.0))),
                -float(mask_hdu.header.get("LTV2", image_hdu.header.get("LTV2", 0.0))),
            )
            bits = _mask_plane_bits(mask_hdu.header)
        sat_mask = _mask_for_planes(mask, bits, ("SAT",))
        obscuring_mask = _mask_for_planes(mask, bits, SOURCE_OBSCURING_PLANES)
        rows = _load_sources(args.preprocessed_root, tract=str(args.tract), patch=str(args.patch), band=band)
        sat_rows = _select_center_in_mask(rows, sat_mask, origin=origin)
        obscuring_rows = _select_center_in_mask(rows, obscuring_mask, origin=origin)
        _write_reg(args.out_dir / f"{band}_{args.tract}_{args.patch.replace(',', '_')}_center_in_SAT.reg", sat_rows, region_name="SAT")
        _write_reg(
            args.out_dir / f"{band}_{args.tract}_{args.patch.replace(',', '_')}_center_in_source_obscuring.reg",
            obscuring_rows,
            region_name="source_obscuring",
        )
        _plot_band(
            band=band,
            image=image,
            sat_mask=sat_mask,
            obscuring_mask=obscuring_mask,
            sat_rows=sat_rows,
            obscuring_rows=obscuring_rows,
            out_path=args.out_dir / f"{band}_{args.tract}_{args.patch.replace(',', '_')}_sources_center_in_mask.png",
            downsample=max(1, int(args.downsample)),
        )
        _plot_crops(
            band=band,
            image=image,
            sat_mask=sat_mask,
            obscuring_mask=obscuring_mask,
            sat_rows=sat_rows,
            obscuring_rows=obscuring_rows,
            points=points,
            origin=origin,
            out_dir=args.out_dir,
            crop_size=int(args.crop_size),
        )

        def class_counts(selected: list[dict[str, object]], prefix: str) -> dict[str, int]:
            out: dict[str, int] = {}
            for class_name, _, _ in CLASS_DIRS:
                out[f"{prefix}_{class_name}"] = sum(1 for row in selected if row["class"] == class_name)
            out[f"{prefix}_total"] = len(selected)
            return out

        row = {
            "band": band,
            "sat_pixel_fraction": float(np.count_nonzero(sat_mask) / sat_mask.size),
            "source_obscuring_pixel_fraction": float(np.count_nonzero(obscuring_mask) / obscuring_mask.size),
        }
        row.update(class_counts(sat_rows, "center_in_sat"))
        row.update(class_counts(obscuring_rows, "center_in_source_obscuring"))
        summary_rows.append(row)
        print(row, flush=True)

    fields = sorted({key for row in summary_rows for key in row})
    with (args.out_dir / f"{args.tract}_{args.patch.replace(',', '_')}_sources_center_in_mask_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
