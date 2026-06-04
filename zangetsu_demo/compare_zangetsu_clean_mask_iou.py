#!/usr/bin/env python3
"""Compare Zangetsu clean masks against refit proxy Kron masks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table


TRACT = "9813"
PATCH = "6,1"
BAND = "HSC-I"
TILES = (
    "zangetsu_upper_left_x26342_y7477",
    "zangetsu_lower_right_x27366_y6453",
)


def parse_tile_origin(tile: str) -> tuple[int, int]:
    parts = tile.split("_")
    x_part = next(part for part in parts if part.startswith("x"))
    y_part = next(part for part in parts if part.startswith("y"))
    return int(x_part[1:]), int(y_part[1:])


def read_refit_rows(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("status", "")).lower() != "ok":
                continue
            if str(row.get("proxy_nan0_good", "")).lower() not in {"1", "true", "t", "yes", "y"}:
                continue
            try:
                source_id = int(row["source_id"])
                axis_a = float(row["axis_a"])
                axis_b = float(row["axis_b"])
                theta = math.radians(float(row["theta_deg"]))
                initial_radius = float(row["initial_determinant_radius"])
                proxy_radius = float(row["proxy_nan0_determine_radius_returned_radius"])
                x = float(row["centroid_x"])
                y = float(row["centroid_y"])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(np.isfinite(v) and v > 0.0 for v in (axis_a, axis_b, initial_radius, proxy_radius)):
                continue
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(theta)):
                continue
            scale = proxy_radius / initial_radius
            rows[source_id] = {
                "x": x,
                "y": y,
                "a": axis_a * scale,
                "b": axis_b * scale,
                "theta": theta,
            }
    return rows


def rasterize_ellipses(rows: Iterable[dict[str, float]], *, x0: int, y0: int, size: int = 512) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    yy_full, xx_full = np.mgrid[0:size, 0:size]
    for row in rows:
        cx = float(row["x"] - x0)
        cy = float(row["y"] - y0)
        a = float(max(row["a"], 1.5))
        b = float(max(row["b"], 1.5))
        theta = float(row["theta"])
        if not all(np.isfinite(v) for v in (cx, cy, a, b, theta)):
            continue
        radius = int(math.ceil(max(a, b))) + 2
        cx_i = int(round(cx))
        cy_i = int(round(cy))
        yy0, yy1 = max(0, cy_i - radius), min(size, cy_i + radius + 1)
        xx0, xx1 = max(0, cx_i - radius), min(size, cx_i + radius + 1)
        if yy0 >= yy1 or xx0 >= xx1:
            continue
        dx = xx_full[yy0:yy1, xx0:xx1] - cx
        dy = yy_full[yy0:yy1, xx0:xx1] - cy
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        xr = cos_t * dx + sin_t * dy
        yr = -sin_t * dx + cos_t * dy
        mask[yy0:yy1, xx0:xx1] |= (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
    return mask


def current_rows_from_table(table: Table, *, x_col: str = "base_SdssShape_x", y_col: str = "base_SdssShape_y") -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for row in table:
        try:
            x = float(row[x_col])
            y = float(row[y_col])
            a = float(row["ellipse_major_sigma"])
            b = float(row["ellipse_minor_sigma"])
            theta = float(row["ellipse_theta"])
        except (KeyError, TypeError, ValueError):
            continue
        if all(np.isfinite(v) and (v > 0.0 if name in {"a", "b"} else True) for name, v in {"x": x, "y": y, "a": a, "b": b, "theta": theta}.items()):
            out.append({"x": x, "y": y, "a": a, "b": b, "theta": theta})
    return out


def proxy_rows_for_table(table: Table, refit_by_id: dict[int, dict[str, float]]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if "id" not in table.colnames:
        return rows
    for row in table:
        proxy = refit_by_id.get(int(row["id"]))
        if proxy is not None:
            rows.append(proxy)
    return rows


def proxy_rows_for_tile(refit_by_id: dict[int, dict[str, float]], *, x0: int, y0: int, size: int = 512, margin: float = 64.0) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for row in refit_by_id.values():
        if x0 - margin <= row["x"] < x0 + size + margin and y0 - margin <= row["y"] < y0 + size + margin:
            rows.append(row)
    return rows


def iou(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int, int]:
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return (float(inter / union) if union else 1.0), inter, union, int(np.count_nonzero(a ^ b))


def overlay_image(current: np.ndarray, proxy: np.ndarray) -> np.ndarray:
    img = np.zeros((*current.shape, 3), dtype=np.float32)
    only_current = current & ~proxy
    only_proxy = proxy & ~current
    both = current & proxy
    img[only_current] = (0.1, 0.55, 1.0)
    img[only_proxy] = (1.0, 0.15, 0.75)
    img[both] = (1.0, 1.0, 1.0)
    return img


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("zangetsu_demo/preprocessed"))
    parser.add_argument("--dataset", default="coadd")
    parser.add_argument("--refit-csv", type=Path, default=Path("/nvme0/zc/scarlet/refit/9813/HSC-I/6,1/batch_heavyfp_kron_refit/batch_heavyfp_kron_refit.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("zangetsu_demo/clean_mask_iou"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    refit_by_id = read_refit_rows(args.refit_csv)
    rows_out: list[dict[str, object]] = []

    for tile in TILES:
        x0, y0 = parse_tile_origin(tile)
        targets_path = args.root / args.dataset / TRACT / PATCH / "band_targets" / BAND / f"{tile}.npz"
        table_path = args.root / args.dataset / TRACT / PATCH / "band_reference_catalogs" / BAND / f"meas-{BAND}-{TRACT}-{PATCH}.fits"
        with np.load(targets_path) as data:
            target_clean = np.asarray(data["clean_mask"], dtype=bool)
        table = Table.read(table_path)
        table = table[
            (np.asarray(table["base_SdssShape_x"], dtype=float) >= x0 - 64)
            & (np.asarray(table["base_SdssShape_x"], dtype=float) < x0 + 512 + 64)
            & (np.asarray(table["base_SdssShape_y"], dtype=float) >= y0 - 64)
            & (np.asarray(table["base_SdssShape_y"], dtype=float) < y0 + 512 + 64)
        ]

        current_mask = rasterize_ellipses(current_rows_from_table(table), x0=x0, y0=y0)
        proxy_same_sources_mask = rasterize_ellipses(proxy_rows_for_table(table, refit_by_id), x0=x0, y0=y0)
        proxy_all_good_mask = rasterize_ellipses(proxy_rows_for_tile(refit_by_id, x0=x0, y0=y0), x0=x0, y0=y0)

        comparisons = {
            "target_vs_current_table": (target_clean, current_mask),
            "target_vs_proxy_same_clean_sources": (target_clean, proxy_same_sources_mask),
            "current_table_vs_proxy_same_clean_sources": (current_mask, proxy_same_sources_mask),
            "target_vs_proxy_all_good_refit_sources": (target_clean, proxy_all_good_mask),
        }
        for name, (left, right) in comparisons.items():
            score, inter, union, xor = iou(left, right)
            rows_out.append(
                {
                    "dataset": args.dataset,
                    "tile": tile,
                    "comparison": name,
                    "iou": score,
                    "intersection_px": inter,
                    "union_px": union,
                    "xor_px": xor,
                    "left_px": int(np.count_nonzero(left)),
                    "right_px": int(np.count_nonzero(right)),
                    "clean_rows_in_tile": int(len(table)),
                    "clean_rows_with_proxy_refit": int(len(proxy_rows_for_table(table, refit_by_id))),
                    "proxy_all_good_rows_in_tile": int(len(proxy_rows_for_tile(refit_by_id, x0=x0, y0=y0))),
                }
            )

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        panels = [
            ("target clean vs proxy same clean", target_clean, proxy_same_sources_mask),
            ("current table vs proxy same clean", current_mask, proxy_same_sources_mask),
            ("target clean vs proxy all good", target_clean, proxy_all_good_mask),
        ]
        for ax, (title, left, right) in zip(axes, panels):
            score, _, _, _ = iou(left, right)
            ax.imshow(overlay_image(left, right), origin="lower")
            ax.set_title(f"{title}\nIoU={score:.3f}")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{args.dataset} {BAND} {PATCH} {tile}: blue=current, magenta=proxy, white=overlap")
        fig.savefig(args.out_dir / f"{args.dataset}_{BAND}_{PATCH.replace(',', '_')}_{tile}_clean_mask_iou.png", dpi=180)
        plt.close(fig)

    csv_path = args.out_dir / f"{args.dataset}_{BAND}_{PATCH.replace(',', '_')}_clean_mask_iou.csv"
    json_path = args.out_dir / f"{args.dataset}_{BAND}_{PATCH.replace(',', '_')}_clean_mask_iou.json"
    fieldnames = list(rows_out[0].keys()) if rows_out else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    json_path.write_text(json.dumps(rows_out, indent=2) + "\n")
    print(csv_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
