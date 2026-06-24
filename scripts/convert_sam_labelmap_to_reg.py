#!/usr/bin/env python3
"""Convert a SAM AMG label-map FITS file to DS9 region files.

The AMG metadata field point_input_x/y is a prompt point, not the final mask
centroid.  This script writes both prompt points and true labelmap centroids.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from astropy.io import fits


COLORS = ("green", "cyan", "magenta", "yellow", "red", "blue", "orange", "white")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labelmap", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stem", default=None)
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--max-boundary-lines", type=int, default=250000)
    parser.add_argument(
        "--overwrite-mask-centers",
        action="store_true",
        help="Also write *_mask_centers.reg using true labelmap centroids.",
    )
    return parser.parse_args()


def read_labelmap(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data)
    if data.ndim != 2:
        raise ValueError(f"labelmap must be 2D, got shape {data.shape}")
    return data.astype(np.int64, copy=False)


def read_metadata(path: Path | None) -> dict[int, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {int(row["id"]): row for row in csv.DictReader(handle) if row.get("id")}


def ds9_header(title: str) -> list[str]:
    return [
        "# Region file format: DS9 version 4.1",
        f"# {title}",
        "global color=green dashlist=8 3 width=2 font='helvetica 10 normal roman' "
        "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
        "image",
    ]


def color_for(label: int) -> str:
    return COLORS[(int(label) - 1) % len(COLORS)]


def point_line(x_zero_based: float, y_zero_based: float, color: str, text: str, radius: float = 3.0) -> str:
    return f"circle({x_zero_based + 1.0:.3f},{y_zero_based + 1.0:.3f},{radius:.1f}) # color={color} width=2 text={{{text}}}"


def mask_boundary_lines(mask: np.ndarray, color: str) -> list[str]:
    ys, xs = np.nonzero(mask)
    lines: list[str] = []
    height, width = mask.shape
    for y, x in zip(ys.tolist(), xs.tolist()):
        x0 = x + 0.5
        x1 = x + 1.5
        y0 = y + 0.5
        y1 = y + 1.5
        if x == 0 or not mask[y, x - 1]:
            lines.append(f"line({x0:.1f},{y0:.1f},{x0:.1f},{y1:.1f}) # color={color} width=2")
        if x == width - 1 or not mask[y, x + 1]:
            lines.append(f"line({x1:.1f},{y0:.1f},{x1:.1f},{y1:.1f}) # color={color} width=2")
        if y == 0 or not mask[y - 1, x]:
            lines.append(f"line({x0:.1f},{y0:.1f},{x1:.1f},{y0:.1f}) # color={color} width=2")
        if y == height - 1 or not mask[y + 1, x]:
            lines.append(f"line({x0:.1f},{y1:.1f},{x1:.1f},{y1:.1f}) # color={color} width=2")
    return lines


def bbox_line(label: int, row: dict[str, str], color: str) -> str | None:
    try:
        x0 = float(row["bbox_x0"]) + 1.0
        y0 = float(row["bbox_y0"]) + 1.0
        w = float(row["bbox_w"])
        h = float(row["bbox_h"])
    except Exception:
        return None
    xc = x0 + 0.5 * (w - 1.0)
    yc = y0 + 0.5 * (h - 1.0)
    text = f"id={label}"
    if row.get("predicted_iou"):
        text += f" iou={float(row['predicted_iou']):.3f}"
    if row.get("stability_score"):
        text += f" stab={float(row['stability_score']):.3f}"
    return f"box({xc:.3f},{yc:.3f},{w:.3f},{h:.3f},0) # color={color} width=2 text={{{text}}}"


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    labelmap = read_labelmap(args.labelmap.expanduser())
    metadata = read_metadata(args.metadata_csv.expanduser() if args.metadata_csv else None)
    stem = args.stem or args.labelmap.stem
    out_dir = args.out_dir.expanduser()

    labels, counts = np.unique(labelmap[labelmap > 0], return_counts=True)
    label_counts = {int(label): int(count) for label, count in zip(labels, counts)}
    labels = np.asarray([label for label in labels if label_counts[int(label)] >= args.min_area], dtype=np.int64)

    boundary = ds9_header(f"SAM labelmap boundaries from {args.labelmap}")
    bbox = ds9_header(f"SAM labelmap bboxes from {args.labelmap}")
    centroids = ds9_header(f"SAM final labelmap centroids from {args.labelmap}")
    prompts = ds9_header(f"SAM AMG prompt points from {args.labelmap}")

    n_boundary_lines = 0
    prompt_inside = 0
    n_prompt = 0
    for label in labels.tolist():
        color = color_for(label)
        mask = labelmap == label
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        lines = mask_boundary_lines(mask, color)
        if n_boundary_lines + len(lines) <= args.max_boundary_lines:
            boundary.extend(lines)
            n_boundary_lines += len(lines)
        else:
            boundary.append(
                f"# skipped boundary for label {label}: line budget exceeded "
                f"({n_boundary_lines}+{len(lines)}>{args.max_boundary_lines})"
            )

        row = metadata.get(label, {})
        bline = bbox_line(label, row, color)
        if bline is not None:
            bbox.append(bline)

        cx = float(xs.mean())
        cy = float(ys.mean())
        centroids.append(point_line(cx, cy, color, f"id={label} centroid"))

        try:
            px = float(row["point_input_x"])
            py = float(row["point_input_y"])
            prompts.append(point_line(px, py, color, f"id={label} prompt", radius=2.5))
            n_prompt += 1
            xi = int(round(px))
            yi = int(round(py))
            if 0 <= yi < labelmap.shape[0] and 0 <= xi < labelmap.shape[1] and int(labelmap[yi, xi]) == int(label):
                prompt_inside += 1
        except Exception:
            pass

    write_text(out_dir / f"{stem}_mask_boundaries.reg", boundary)
    write_text(out_dir / f"{stem}_mask_bboxes.reg", bbox)
    write_text(out_dir / f"{stem}_mask_centroids.reg", centroids)
    write_text(out_dir / f"{stem}_mask_prompt_points.reg", prompts)
    if args.overwrite_mask_centers:
        write_text(out_dir / f"{stem}_mask_centers.reg", centroids)

    summary = [
        "label,n_pixels",
        *[f"{label},{label_counts[int(label)]}" for label in labels.tolist()],
    ]
    write_text(out_dir / f"{stem}_mask_summary.csv", summary)
    print(
        {
            "labels": int(len(labels)),
            "boundary_lines": int(n_boundary_lines),
            "prompt_inside_fraction": float(prompt_inside / n_prompt) if n_prompt else None,
            "out_dir": str(out_dir),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
