#!/usr/bin/env python3
"""Export source and ignore-mask REG files from direct patch zarr outputs."""

from __future__ import annotations

import argparse
import math
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")

import numpy as np
import zarr


CLASS_COLORS = {
    "clean": "green",
    "center_only": "yellow",
    "ignore": "red",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zarr-root", type=Path, required=True)
    p.add_argument("--tract", default="9813")
    p.add_argument("--variant", default="noisy")
    p.add_argument("--patch", default="4,5")
    p.add_argument("--group", default="group_01")
    p.add_argument("--bands", nargs="+", default=["HSC-I", "HSC-Y"])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--ignore-min-area", type=int, default=16)
    p.add_argument("--ignore-box-grid", type=int, default=32)
    return p.parse_args()


def _decode_fixed(row: np.ndarray) -> str:
    return bytes(np.asarray(row, dtype=np.uint8)).split(b"\0", 1)[0].decode("utf-8", "ignore")


def _reg_header(coord: str = "physical") -> list[str]:
    return [
        "# Region file format: DS9 version 4.1",
        'global color=green dashlist=8 3 width=2 font="helvetica 10 normal roman" '
        "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
        coord,
    ]


def _ellipse_line(x: float, y: float, major: float, minor: float, theta: float, color: str, sid: int, class_name: str) -> str:
    area = math.pi * major * minor if np.isfinite(major * minor) else float("inf")
    # DS9 physical/image coordinates are 1-indexed.
    xp = x + 1.0
    yp = y + 1.0
    if not np.isfinite(area) or area <= 0.0 or area > 40000.0:
        return f"point({xp:.3f},{yp:.3f}) # point=circle color={color} width=2 text={{{sid} {class_name}}}"
    return (
        f"ellipse({xp:.3f},{yp:.3f},{major:.3f},{minor:.3f},{math.degrees(theta):.3f}) "
        f"# color={color} width=2 text={{{sid} {class_name}}}"
    )


def _source_rows(root, *, band_idx: int, group: str) -> tuple[list[dict[str, object]], Counter]:
    rows_by_key: dict[tuple[int, int], dict[str, object]] = {}
    counts = Counter()
    n_samples = root["shape_source_offsets"].shape[0]
    for sample_idx in range(n_samples):
        if _decode_fixed(root["group"][sample_idx]) != group:
            continue
        x0 = int(root["tile_x0"][sample_idx])
        y0 = int(root["tile_y0"][sample_idx])
        start = int(root["shape_source_offsets"][sample_idx, band_idx])
        stop = int(root["shape_source_offsets"][sample_idx, band_idx + 1])
        if stop <= start:
            continue
        centers = np.asarray(root["shape_source_centers"][start:stop], dtype=np.float64)
        values = np.asarray(root["shape_source_values"][start:stop], dtype=np.float64)
        classes = np.asarray(root["shape_source_classes"][start:stop], dtype=np.uint8)
        ids = np.asarray(root["shape_source_ids"][start:stop], dtype=np.int64)
        for center, value, cls, sid in zip(centers, values, classes, ids):
            if int(cls) == 1:
                class_name = "clean"
            elif int(cls) == 2:
                class_name = "center_only"
            else:
                continue
            abs_x = float(center[0]) + float(x0)
            abs_y = float(center[1]) + float(y0)
            source_id = int(sid)
            # One source can appear in multiple overlapping tiles. Prefer clean over center_only
            # and otherwise keep the first deterministic occurrence.
            key = (source_id, 0 if class_name == "clean" else 1)
            if source_id >= 0:
                existing = rows_by_key.get((source_id, 0)) or rows_by_key.get((source_id, 1))
                if existing is not None:
                    if existing["class"] == "clean" or class_name != "clean":
                        continue
                    rows_by_key.pop((source_id, 1), None)
                    key = (source_id, 0)
            else:
                key = (int(round(abs_x * 1000.0)), int(round(abs_y * 1000.0)))
            rows_by_key[key] = {
                "class": class_name,
                "x": abs_x,
                "y": abs_y,
                "major": float(value[0]),
                "minor": float(value[1]),
                "theta": float(value[2]),
                "id": source_id,
            }
    rows = list(rows_by_key.values())
    counts.update(row["class"] for row in rows)
    return rows, counts


def _full_patch_mask(root, *, band_idx: int, group: str, class_value: int) -> np.ndarray:
    x0s = []
    y0s = []
    samples = []
    n_samples = root["band_pu_class_mask"].shape[0]
    for sample_idx in range(n_samples):
        if _decode_fixed(root["group"][sample_idx]) != group:
            continue
        x0 = int(root["tile_x0"][sample_idx])
        y0 = int(root["tile_y0"][sample_idx])
        x0s.append(x0)
        y0s.append(y0)
        samples.append((sample_idx, x0, y0))
    if not samples:
        return np.zeros((0, 0), dtype=bool)
    tile_h, tile_w = root["band_pu_class_mask"].shape[-2:]
    min_x = min(x0s)
    min_y = min(y0s)
    max_x = max(x0 + tile_w for _idx, x0, _y0 in samples)
    max_y = max(y0 + tile_h for _idx, _x0, y0 in samples)
    mask = np.zeros((max_y - min_y, max_x - min_x), dtype=bool)
    for sample_idx, x0, y0 in samples:
        tile_mask = np.asarray(root["band_pu_class_mask"][sample_idx, band_idx] == int(class_value), dtype=bool)
        yy = y0 - min_y
        xx = x0 - min_x
        mask[yy : yy + tile_h, xx : xx + tile_w] |= tile_mask
    return mask, min_x, min_y


def _ignore_box_lines(mask: np.ndarray, *, origin_x: int, origin_y: int, min_area: int, grid: int) -> tuple[list[str], int]:
    # Aggregate dense ignore masks into fixed-grid boxes. This keeps REG files usable
    # without pretending zarr contains source-level ignore ellipses.
    lines: list[str] = []
    h, w = mask.shape
    n_boxes = 0
    for y0 in range(0, h, grid):
        for x0 in range(0, w, grid):
            y1 = min(h, y0 + grid)
            x1 = min(w, x0 + grid)
            block = mask[y0:y1, x0:x1]
            area = int(block.sum())
            if area < int(min_area):
                continue
            ys, xs = np.where(block)
            if ys.size == 0:
                continue
            bx0 = origin_x + x0 + int(xs.min())
            bx1 = origin_x + x0 + int(xs.max()) + 1
            by0 = origin_y + y0 + int(ys.min())
            by1 = origin_y + y0 + int(ys.max()) + 1
            cx = 0.5 * (bx0 + bx1) + 1.0
            cy = 0.5 * (by0 + by1) + 1.0
            bw = max(1, bx1 - bx0)
            bh = max(1, by1 - by0)
            lines.append(f"box({cx:.3f},{cy:.3f},{bw:.3f},{bh:.3f},0) # color=red width=1 text={{ignore area={area}}}")
            n_boxes += 1
    return lines, n_boxes


def _write_source_regs(out_dir: Path, *, prefix: str, rows: list[dict[str, object]]) -> Counter:
    counts = Counter(row["class"] for row in rows)
    for class_name in ("clean", "center_only"):
        lines = _reg_header("physical")
        for row in sorted((r for r in rows if r["class"] == class_name), key=lambda r: float(r["major"]) * float(r["minor"]), reverse=True):
            lines.append(
                _ellipse_line(
                    float(row["x"]),
                    float(row["y"]),
                    float(row["major"]),
                    float(row["minor"]),
                    float(row["theta"]),
                    CLASS_COLORS[class_name],
                    int(row["id"]),
                    class_name,
                )
            )
        (out_dir / f"{prefix}_{class_name}.reg").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return counts


def main() -> int:
    args = parse_args()
    zarr_path = args.zarr_root / str(args.tract) / args.variant / f"{args.patch}.zarr"
    root = zarr.open_group(str(zarr_path), mode="r")
    bands = list(root.attrs["bands"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = ["band,clean,center_only,ignore_boxes,ignore_pixels"]
    for band in args.bands:
        if band not in bands:
            raise KeyError(f"band {band!r} not in zarr bands {bands}")
        band_idx = bands.index(band)
        prefix = f"{args.variant}_{args.group}_{band}_{args.tract}_{args.patch.replace(',', '_')}"
        rows, counts = _source_rows(root, band_idx=band_idx, group=args.group)
        _write_source_regs(args.out_dir, prefix=prefix, rows=rows)
        ignore_mask, origin_x, origin_y = _full_patch_mask(root, band_idx=band_idx, group=args.group, class_value=3)
        ignore_lines, ignore_boxes = _ignore_box_lines(
            ignore_mask,
            origin_x=origin_x,
            origin_y=origin_y,
            min_area=int(args.ignore_min_area),
            grid=int(args.ignore_box_grid),
        )
        (args.out_dir / f"{prefix}_ignore.reg").write_text("\n".join(_reg_header("physical") + ignore_lines) + "\n", encoding="utf-8")
        summary_lines.append(
            f"{band},{int(counts['clean'])},{int(counts['center_only'])},{ignore_boxes},{int(ignore_mask.sum())}"
        )
        print(summary_lines[-1])
    (args.out_dir / f"{args.variant}_{args.group}_{args.tract}_{args.patch.replace(',', '_')}_summary.csv").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
