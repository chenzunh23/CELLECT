#!/usr/bin/env python
"""Summarize patch 4,5 preprocessing outputs and export diagnostic REG/overlays."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import warnings
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cellect")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from astropy.table import Table, vstack
from astropy.units import UnitsWarning
from astropy.visualization import ZScaleInterval

warnings.filterwarnings("ignore", category=UnitsWarning)
warnings.filterwarnings("ignore", category=VerifyWarning)

BANDS = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y")
CLASS_COLORS = {
    "clean": "green",
    "center_only": "yellow",
    "ignore": "red",
}
RGB_CLASS_COLORS = {
    "clean": np.array([0.0, 0.95, 1.0], dtype=np.float32),
    "center_only": np.array([1.0, 0.2, 1.0], dtype=np.float32),
    "ignore": np.array([0.95, 0.05, 0.05], dtype=np.float32),
}
LEVEL_COLORS = {
    1: np.array([0.10, 0.45, 1.00], dtype=np.float32),
    2: np.array([0.00, 0.85, 0.35], dtype=np.float32),
    3: np.array([1.00, 0.85, 0.05], dtype=np.float32),
    4: np.array([1.00, 0.05, 0.05], dtype=np.float32),
}


def parse_tile_origin(name: str) -> tuple[int, int]:
    match = re.search(r"_x(-?\d+)_y(-?\d+)", name)
    if match is None:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def strip_group(name: str) -> str:
    return re.sub(r"^group_\d+_", "", name)


def read_table(path: Path) -> Table:
    if not path.exists():
        return Table()
    return Table.read(path)


def first_column(table: Table, names: Iterable[str], default: float = np.nan) -> np.ndarray:
    out = np.full(len(table), float(default), dtype=np.float64)
    filled = np.zeros(len(table), dtype=bool)
    for name in names:
        if name not in table.colnames:
            continue
        values = np.asarray(table[name], dtype=np.float64)
        take = ~filled & np.isfinite(values)
        out[take] = values[take]
        filled[take] = True
    return out


def table_xy(table: Table) -> tuple[np.ndarray, np.ndarray]:
    x = first_column(table, ("base_SdssCentroid_x", "base_SdssShape_x", "slot_Centroid_x", "x"))
    y = first_column(table, ("base_SdssCentroid_y", "base_SdssShape_y", "slot_Centroid_y", "y"))
    return x, y


def ellipse_from_moments(xx: np.ndarray, yy: np.ndarray, xy: np.ndarray, kron: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx = np.asarray(xx, dtype=np.float64)
    yy = np.asarray(yy, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    kron = np.asarray(kron, dtype=np.float64)
    major = np.full(xx.shape, np.nan, dtype=np.float64)
    minor = np.full(xx.shape, np.nan, dtype=np.float64)
    theta = np.full(xx.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(xy) & np.isfinite(kron) & (kron > 0)
    xx_safe = np.maximum(xx, 0.25)
    yy_safe = np.maximum(yy, 0.25)
    trace = xx_safe + yy_safe
    delta = np.sqrt(np.maximum((xx_safe - yy_safe) ** 2 + 4.0 * xy**2, 0.0))
    sdss_major = np.sqrt(np.maximum(0.5 * (trace + delta), 0.25))
    sdss_minor = np.sqrt(np.maximum(0.5 * (trace - delta), 0.25))
    det_radius = np.sqrt(np.maximum(sdss_major * sdss_minor, 0.0))
    valid &= np.isfinite(det_radius) & (det_radius > 0)
    scale = np.zeros_like(kron)
    scale[valid] = kron[valid] / det_radius[valid]
    major[valid] = np.maximum(sdss_major[valid] * scale[valid], 0.25)
    minor[valid] = np.maximum(sdss_minor[valid] * scale[valid], 0.25)
    theta[valid] = 0.5 * np.arctan2(2.0 * xy[valid], xx_safe[valid] - yy_safe[valid])
    return major, minor, theta


def ellipse_from_table(table: Table) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx = first_column(table, ("base_SdssShape_xx", "ext_shapeHSM_HsmSourceMoments_xx"), default=np.nan)
    yy = first_column(table, ("base_SdssShape_yy", "ext_shapeHSM_HsmSourceMoments_yy"), default=np.nan)
    xy = first_column(table, ("base_SdssShape_xy", "ext_shapeHSM_HsmSourceMoments_xy"), default=np.nan)
    kron = first_column(
        table,
        ("pu_refit_kron_radius", "ext_photometryKron_KronFlux_radius", "ext_photometryKron_KronFlux_radius_for_radius"),
        default=np.nan,
    )
    return ellipse_from_moments(xx, yy, xy, kron)


def ellipse_from_metadata(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        centers = np.asarray(data["centers"], dtype=np.float64)
        moments = np.asarray(data["moments"], dtype=np.float64)
        kron = np.asarray(data["kron_radius"], dtype=np.float64)
        ids = np.asarray(data["ids"], dtype=np.int64)
    major, minor, theta = ellipse_from_moments(moments[:, 0], moments[:, 1], moments[:, 2], kron)
    return {"centers": centers, "ids": ids, "major": major, "minor": minor, "theta": theta}


def reg_line(x: float, y: float, major: float, minor: float, theta: float, color: str, *, width: int = 2) -> str:
    area = math.pi * major * minor if np.isfinite(major * minor) else np.inf
    if not np.isfinite(area) or area <= 0 or area > 40000:
        return f"point({x + 1:.3f},{y + 1:.3f}) # point=circle 5 color={color} width={width}\n"
    return f"ellipse({x + 1:.3f},{y + 1:.3f},{major:.3f},{minor:.3f},{math.degrees(theta):.3f}) # color={color} width={width}\n"


def write_reg(path: Path, rows: list[dict[str, object]], *, coordinate: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: float(r.get("area", 0.0)), reverse=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Region file format: DS9 version 4.1\n")
        handle.write('global color=green dashlist=8 3 width=2 font="helvetica 12 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        handle.write(f"{coordinate}\n")
        for row in rows:
            handle.write(
                reg_line(
                    float(row["x"]),
                    float(row["y"]),
                    float(row["major"]),
                    float(row["minor"]),
                    float(row["theta"]),
                    str(row["color"]),
                )
            )


def coadd_class_catalog(root: Path, tract: str, patch: str, band: str, class_name: str) -> Table:
    dirname = {
        "clean": "band_reference_catalogs",
        "center_only": "band_reference_center_only",
        "ignore": "band_reference_ignore",
    }[class_name]
    return read_table(root / tract / patch / dirname / band / f"meas-{band}-{tract}-{patch}.fits")


def rows_from_table(table: Table, class_name: str, *, local_origin: tuple[int, int] | None = None) -> list[dict[str, object]]:
    if len(table) == 0:
        return []
    x, y = table_xy(table)
    major, minor, theta = ellipse_from_table(table)
    if local_origin is not None:
        x = x - float(local_origin[0])
        y = y - float(local_origin[1])
    color = CLASS_COLORS[class_name]
    rows = []
    for i in range(len(table)):
        if not np.isfinite(x[i]) or not np.isfinite(y[i]):
            continue
        rows.append(
            {
                "x": float(x[i]),
                "y": float(y[i]),
                "major": float(major[i]),
                "minor": float(minor[i]),
                "theta": float(theta[i]),
                "color": color,
                "class": class_name,
                "area": float(math.pi * major[i] * minor[i]) if np.isfinite(major[i] * minor[i]) else 0.0,
            }
        )
    return rows


def nearest_shape_lookup(clean_table: Table) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = table_xy(clean_table)
    major, minor, theta = ellipse_from_table(clean_table)
    return x, y, major, minor, theta


def nearest_shape(abs_x: float, abs_y: float, lookup: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> tuple[float, float, float]:
    x, y, major, minor, theta = lookup
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    dist2 = (x - abs_x) ** 2 + (y - abs_y) ** 2
    idx = int(np.nanargmin(dist2))
    if not np.isfinite(dist2[idx]) or dist2[idx] > 4.0:
        return np.nan, np.nan, np.nan
    return float(major[idx]), float(minor[idx]), float(theta[idx])


def target_path(patch_root: Path, band: str, tile: str) -> Path:
    return patch_root / "band_targets" / band / f"{tile}.npz"


def metadata_path(patch_root: Path, band: str, tile: str) -> Path:
    return patch_root / "band_tile_metadata" / band / f"{tile}.npz"


def variant_tiles(patch_root: Path, group: str | None) -> list[str]:
    tiles = []
    target_dir = patch_root / "targets"
    for path in sorted(target_dir.glob("*.npz")):
        name = path.stem
        if group is not None and not name.startswith(f"{group}_"):
            continue
        tiles.append(name)
    return tiles


def variant_rows_for_band(
    patch_root: Path,
    coadd_root: Path,
    tract: str,
    patch: str,
    band: str,
    *,
    group: str,
    local_tile: str | None = None,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    clean_table = coadd_class_catalog(coadd_root, tract, patch, band, "clean")
    lookup = nearest_shape_lookup(clean_table)
    rows: list[dict[str, object]] = []
    clean_ids: set[int] = set()
    center_keys: set[tuple[int, int]] = set()
    ignore_keys: set[tuple[int, int]] = set()
    tiles = variant_tiles(patch_root, group)
    if local_tile is not None:
        tiles = [tile for tile in tiles if tile == local_tile]
    for tile in tiles:
        base = strip_group(tile)
        x0, y0 = parse_tile_origin(base)
        meta = metadata_path(patch_root, band, tile)
        if meta.exists():
            md = ellipse_from_metadata(meta)
            centers = md["centers"]
            ids = md["ids"]
            for i, source_id in enumerate(ids):
                key = int(source_id)
                if key in clean_ids and local_tile is None:
                    continue
                clean_ids.add(key)
                abs_x = float(centers[i, 0] + x0)
                abs_y = float(centers[i, 1] + y0)
                rx = float(centers[i, 0]) if local_tile is not None else abs_x
                ry = float(centers[i, 1]) if local_tile is not None else abs_y
                rows.append(
                    {
                        "x": rx,
                        "y": ry,
                        "major": float(md["major"][i]),
                        "minor": float(md["minor"][i]),
                        "theta": float(md["theta"][i]),
                        "color": CLASS_COLORS["clean"],
                        "class": "clean",
                        "area": float(math.pi * md["major"][i] * md["minor"][i]) if np.isfinite(md["major"][i] * md["minor"][i]) else 0.0,
                    }
                )
        tgt = target_path(patch_root, band, tile)
        if not tgt.exists():
            continue
        with np.load(tgt) as data:
            for class_name, key_name, seen in (
                ("center_only", "visibility_center_only_centers", center_keys),
                ("ignore", "visibility_ignore_centers", ignore_keys),
            ):
                if key_name not in data:
                    continue
                centers = np.asarray(data[key_name], dtype=np.float64).reshape(-1, 2)
                for center in centers:
                    abs_x = float(center[0] + x0)
                    abs_y = float(center[1] + y0)
                    key = (int(round(abs_x * 1000.0)), int(round(abs_y * 1000.0)))
                    if key in seen and local_tile is None:
                        continue
                    seen.add(key)
                    major, minor, theta = nearest_shape(abs_x, abs_y, lookup)
                    rx = float(center[0]) if local_tile is not None else abs_x
                    ry = float(center[1]) if local_tile is not None else abs_y
                    rows.append(
                        {
                            "x": rx,
                            "y": ry,
                            "major": major,
                            "minor": minor,
                            "theta": theta,
                            "color": CLASS_COLORS[class_name],
                            "class": class_name,
                            "area": float(math.pi * major * minor) if np.isfinite(major * minor) else 0.0,
                        }
                    )
    return rows, {"clean": len(clean_ids), "center_only": len(center_keys), "ignore": len(ignore_keys)}


def coadd_stats(root: Path, tract: str, patch: str) -> list[dict[str, object]]:
    rows = []
    for band in BANDS:
        counts = {}
        for class_name in ("clean", "center_only", "ignore"):
            counts[class_name] = len(coadd_class_catalog(root, tract, patch, band, class_name))
        rows.append({"dataset": "coadd", "group": "", "band": band, "total": sum(counts.values()), **counts})
    return rows


def variant_stats(root: Path, coadd_root: Path, tract: str, patch: str, dataset: str, *, group: str | None) -> list[dict[str, object]]:
    patch_root = root / dataset / tract / patch
    groups = [group] if group else sorted({path.stem.split("_", 2)[0] + "_" + path.stem.split("_", 2)[1] for path in (patch_root / "targets").glob("group_*_*.npz")})
    out = []
    for group_name in groups:
        for band in BANDS:
            _rows, counts = variant_rows_for_band(patch_root, coadd_root, tract, patch, band, group=group_name)
            out.append({"dataset": dataset, "group": group_name, "band": band, "total": sum(counts.values()), **counts})
    return out


def write_stats(out_dir: Path, rows: list[dict[str, object]], manifests: dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "source_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "group", "band", "total", "clean", "center_only", "ignore"))
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, dict[str, int]] = {}
    for row in rows:
        key = f"{row['dataset']}:{row['group'] or 'patch'}"
        bucket = summary.setdefault(key, {"total": 0, "clean": 0, "center_only": 0, "ignore": 0})
        for name in ("total", "clean", "center_only", "ignore"):
            bucket[name] += int(row[name])
    payload = {"dedup_by_id_or_center": {"per_band": rows, "summary": summary}, "manifest_source_instances": manifests}
    (out_dir / "source_counts.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def zscale_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        scaled = np.zeros_like(image, dtype=np.float32)
    else:
        try:
            vmin, vmax = ZScaleInterval().get_limits(image[finite])
        except Exception:
            vmin, vmax = np.nanpercentile(image[finite], [1, 99])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = float(np.nanmin(image[finite])), float(np.nanmax(image[finite]))
        scaled = np.clip((image - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    return np.repeat(scaled[:, :, None], 3, axis=2)


def alpha_blend(rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> None:
    mask = np.asarray(mask, dtype=bool)
    if mask.any():
        rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * color[None, :]


def image_for_tile(patch_root: Path, tile: str, band: str) -> Path:
    band_dir = patch_root / "cutouts" / tile / band
    matches = sorted(band_dir.glob("*.fits"))
    if not matches:
        raise FileNotFoundError(band_dir)
    return matches[0]


def save_overlay(patch_root: Path, tile: str, band: str, out_path: Path, *, title: str) -> None:
    image = np.asarray(fits.getdata(image_for_tile(patch_root, tile, band), ext=1), dtype=np.float32)
    tgt = target_path(patch_root, band, tile)
    with np.load(tgt) as data:
        clean = np.asarray(data["clean_mask"], dtype=bool)
        center = np.asarray(data["center_only_mask"], dtype=bool) | np.asarray(data.get("strict_center_only_mask", 0), dtype=bool)
        ignored = np.asarray(data["ignore_mask"], dtype=bool)
        confidence = np.asarray(data["confidence"], dtype=np.int16)
    gt = zscale_rgb(image)
    alpha_blend(gt, clean, RGB_CLASS_COLORS["clean"], 0.18)
    alpha_blend(gt, center, RGB_CLASS_COLORS["center_only"], 0.24)
    alpha_blend(gt, ignored, RGB_CLASS_COLORS["ignore"], 0.18)
    conf = zscale_rgb(image)
    for level, color in LEVEL_COLORS.items():
        alpha_blend(conf, confidence == level, color, 0.78)
    gt = np.clip(gt, 0.0, 1.0)
    conf = np.clip(conf, 0.0, 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), dpi=180)
    axes[0].imshow(zscale_rgb(image), origin="lower", interpolation="nearest")
    axes[0].set_title("zscale")
    axes[1].imshow(gt, origin="lower", interpolation="nearest")
    axes[1].set_title("GT regions")
    axes[2].imshow(conf, origin="lower", interpolation="nearest")
    axes[2].set_title("GT confidence")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    patches = [
        mpatches.Patch(color=RGB_CLASS_COLORS["clean"], label="clean"),
        mpatches.Patch(color=RGB_CLASS_COLORS["center_only"], label="center only"),
        mpatches.Patch(color=RGB_CLASS_COLORS["ignore"], label="ignore"),
    ] + [mpatches.Patch(color=color, label=f"conf {level}") for level, color in LEVEL_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=7, fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/nvme0/zc/scarlet/debug_patch45_preprocess/preprocessed"))
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patch", default="4,5")
    parser.add_argument("--group", default="group_01")
    parser.add_argument("--sam-tile", default="sam_x18204_y20924")
    parser.add_argument(
        "--sam-reg-margin",
        type=float,
        default=0.0,
        help="Margin in pixels for sam-tile REG center selection. Default 0 keeps only centers inside 512x512.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("output/patch45_preprocess_summary_260708"))
    args = parser.parse_args()

    root = args.root
    tract = str(args.tract)
    patch = str(args.patch)
    out_dir = args.out_dir
    coadd_root = root
    rows = coadd_stats(coadd_root, tract, patch)
    rows.extend(variant_stats(root, coadd_root, tract, patch, "denoised", group=args.group))
    rows.extend(variant_stats(root, coadd_root, tract, patch, "noisy", group=args.group))
    manifests = {}
    for dataset in ("denoised", "noisy"):
        path = root / dataset / tract / patch / "manifest.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            manifests[dataset] = data.get("noncoadd_visibility_counts", {})
    write_stats(out_dir, rows, manifests)

    reg_dir = out_dir / "regions"
    overlay_dir = out_dir / "overlays"
    for band in BANDS:
        coadd_reg_rows = []
        for class_name in ("clean", "center_only", "ignore"):
            coadd_reg_rows.extend(rows_from_table(coadd_class_catalog(coadd_root, tract, patch, band, class_name), class_name))
        write_reg(reg_dir / f"coadd_{band}_summary_physical.reg", coadd_reg_rows, coordinate="physical")

        for dataset in ("denoised", "noisy"):
            patch_root = root / dataset / tract / patch
            vrows, _counts = variant_rows_for_band(patch_root, coadd_root, tract, patch, band, group=args.group)
            write_reg(reg_dir / f"{dataset}_{args.group}_{band}_summary_physical.reg", vrows, coordinate="physical")

    for dataset, patch_root, tile in (
        ("coadd", root / tract / patch, args.sam_tile),
        ("denoised", root / "denoised" / tract / patch, f"{args.group}_{args.sam_tile}"),
        ("noisy", root / "noisy" / tract / patch, f"{args.group}_{args.sam_tile}"),
    ):
        for band in BANDS:
            if dataset == "coadd":
                x0, y0 = parse_tile_origin(tile)
                local_rows = []
                for class_name in ("clean", "center_only", "ignore"):
                    table = coadd_class_catalog(coadd_root, tract, patch, band, class_name)
                    local_rows.extend(rows_from_table(table, class_name, local_origin=(x0, y0)))
                margin = float(args.sam_reg_margin)
                local_rows = [
                    row
                    for row in local_rows
                    if -margin <= float(row["x"]) < 512.0 + margin
                    and -margin <= float(row["y"]) < 512.0 + margin
                ]
            else:
                local_rows, _counts = variant_rows_for_band(
                    patch_root,
                    coadd_root,
                    tract,
                    patch,
                    band,
                    group=args.group,
                    local_tile=tile,
                )
            write_reg(reg_dir / "sam_tile" / f"{dataset}_{band}_{tile}_image.reg", local_rows, coordinate="image")
            save_overlay(
                patch_root,
                tile,
                band,
                overlay_dir / f"{dataset}_{band}_{tile}_gt_confidence.png",
                title=f"{dataset} {band} {tile}",
            )

    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
