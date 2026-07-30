#!/usr/bin/env python3
"""Count NB0387 sources in patch/tile regions that pass bad-score cuts."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np

warnings.filterwarnings("ignore", message=".*'second' did not parse as fits unit.*")

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_calexp_mask_quality import find_calexp, normalize_band_dir, parse_score_weights  # noqa: E402
from scripts.overlay_calexp_tile_bad_score import read_calexp, score_tiles, tile_bad_score  # noqa: E402


def parse_patches(values: Iterable[str]) -> list[str]:
    patches: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in str(value).split(";"):
            patch = item.strip()
            if not patch:
                continue
            expanded = [f"{x},{y}" for x in range(9) for y in range(9)] if patch.lower() == "all" else [patch]
            for candidate in expanded:
                if candidate not in seen:
                    patches.append(candidate)
                    seen.add(candidate)
    return patches


def find_meas(patch_dir: Path, band: str, tract: str, patch: str) -> Path | None:
    preferred = patch_dir / f"meas-{band}-{tract}-{patch}.fits"
    if preferred.exists():
        return preferred
    files = sorted(patch_dir.glob("meas-*.fits")) + sorted(patch_dir.glob("meas-*.fits.gz"))
    return files[0] if files else None


def table_column(table, names: tuple[str, ...]):
    for name in names:
        if name in table.colnames:
            return np.asarray(table[name])
    raise KeyError(f"catalog is missing all columns: {', '.join(names)}")


def calexp_physical_origin(calexp_path: Path) -> tuple[float, float]:
    from astropy.io import fits

    with fits.open(calexp_path, memmap=True, ignore_missing_end=True) as hdul:
        header = hdul[1].header
        return -float(header.get("LTV1", 0.0)), -float(header.get("LTV2", 0.0))


def load_sources(meas_path: Path, *, source_filter: str, origin: tuple[float, float]) -> dict[str, np.ndarray]:
    from astropy.table import Table

    table = Table.read(meas_path, hdu=1)
    x_physical = table_column(table, ("base_SdssCentroid_x", "base_NaiveCentroid_x", "slot_Centroid_x")).astype(float)
    y_physical = table_column(table, ("base_SdssCentroid_y", "base_NaiveCentroid_y", "slot_Centroid_y")).astype(float)
    x = x_physical - float(origin[0])
    y = y_physical - float(origin[1])
    radius = table_column(
        table,
        (
            "ext_photometryKron_KronFlux_radius",
            "ext_photometryKron_KronFlux_radius_for_radius",
        ),
    ).astype(float)
    xx = table_column(table, ("base_SdssShape_xx",)).astype(float)
    yy = table_column(table, ("base_SdssShape_yy",)).astype(float)
    xy = table_column(table, ("base_SdssShape_xy",)).astype(float)
    trace = xx + yy
    delta = np.sqrt(np.maximum((xx - yy) ** 2 + 4.0 * xy * xy, 0.0))
    lambda_major = 0.5 * (trace + delta)
    lambda_minor = 0.5 * (trace - delta)
    shape_valid = np.isfinite(lambda_major) & np.isfinite(lambda_minor) & (lambda_major > 0.0) & (lambda_minor > 0.0)
    shape_a = np.full(xx.shape, np.nan, dtype=float)
    shape_b = np.full(xx.shape, np.nan, dtype=float)
    theta_deg = np.full(xx.shape, np.nan, dtype=float)
    determinant_radius = np.full(xx.shape, np.nan, dtype=float)
    shape_a[shape_valid] = np.sqrt(lambda_major[shape_valid])
    shape_b[shape_valid] = np.sqrt(lambda_minor[shape_valid])
    theta_deg[shape_valid] = np.degrees(0.5 * np.arctan2(2.0 * xy[shape_valid], xx[shape_valid] - yy[shape_valid]))
    determinant_radius[shape_valid] = np.sqrt(shape_a[shape_valid] * shape_b[shape_valid])
    scale = radius / determinant_radius
    major = shape_a * scale
    minor = shape_b * scale
    ellipse_area = np.pi * major * minor
    keep = (
        np.isfinite(x_physical)
        & np.isfinite(y_physical)
        & np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(radius)
        & np.isfinite(major)
        & np.isfinite(minor)
        & np.isfinite(theta_deg)
        & (radius > 0.0)
        & (major > 0.0)
        & (minor > 0.0)
    )
    if source_filter == "nchild0" and "deblend_nChild" in table.colnames:
        keep &= np.asarray(table["deblend_nChild"]) == 0
    elif source_filter == "primary" and "detect_isPrimary" in table.colnames:
        keep &= np.asarray(table["detect_isPrimary"]).astype(bool)
    elif source_filter == "nchild0_primary":
        if "deblend_nChild" in table.colnames:
            keep &= np.asarray(table["deblend_nChild"]) == 0
        if "detect_isPrimary" in table.colnames:
            keep &= np.asarray(table["detect_isPrimary"]).astype(bool)
    elif source_filter != "all":
        raise ValueError(f"unknown --source-filter {source_filter!r}")
    ids = np.asarray(table["id"]) if "id" in table.colnames else np.arange(len(table), dtype=np.int64)
    return {
        "id": ids[keep],
        "x": x[keep],
        "y": y[keep],
        "x_physical": x_physical[keep],
        "y_physical": y_physical[keep],
        "radius": radius[keep],
        "major": major[keep],
        "minor": minor[keep],
        "theta_deg": theta_deg[keep],
        "ellipse_area": ellipse_area[keep],
    }


def count_sources_in_tile(sources: dict[str, np.ndarray], x0: int, y0: int, x1: int, y1: int) -> int:
    x = sources["x"]
    y = sources["y"]
    keep = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    return int(keep.sum())


def ds9_header(title: str) -> list[str]:
    return [
        "# Region file format: DS9 version 4.1",
        f"# {title}",
        "global color=green dashlist=8 3 width=2 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
        "physical",
    ]


def write_kron_ellipse_reg(path: Path, sources: dict[str, np.ndarray], *, title: str, point_area_min: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ds9_header(title)
    x_values = sources.get("x_physical", sources["x"])
    y_values = sources.get("y_physical", sources["y"])
    for sid, x, y, major, minor, theta, area in zip(
        sources["id"],
        x_values,
        y_values,
        sources["major"],
        sources["minor"],
        sources["theta_deg"],
        sources["ellipse_area"],
    ):
        if float(area) > float(point_area_min):
            lines.append(f"point({float(x):.3f},{float(y):.3f}) # point=circle color=red text={{{sid} area={float(area):.1f}}}")
        else:
            lines.append(
                f"ellipse({float(x):.3f},{float(y):.3f},{float(major):.3f},{float(minor):.3f},{float(theta):.3f}) "
                f"# color=green text={{{sid} area={float(area):.1f}}}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    preferred = [
        "band",
        "patch",
        "status",
        "pass_patch_score",
        "pass_tile_score",
        "pass_clean_region",
        "patch_score",
        "patch_score_percent",
        "max_tile_score",
        "max_tile_score_percent",
        "tile_count",
        "tile_pass_count",
        "source_count",
        "tile_source_mean_all",
        "tile_source_mean_pass",
        "tile_source_median_all",
        "tile_source_max_all",
        "path_calexp",
        "path_meas",
        "error",
    ]
    for key in preferred:
        fields.append(key)
        seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def analyze_patch(
    *,
    data_root: Path,
    tract: str,
    band: str,
    patch: str,
    weights: dict[str, float],
    tile_size: int,
    stride: int,
    patch_score_max: float,
    tile_score_max: float,
    source_filter: str,
) -> tuple[dict[str, object], dict[str, np.ndarray] | None]:
    patch_dir = data_root / str(tract) / band / patch
    row: dict[str, object] = {"band": band, "patch": patch, "status": "missing"}
    calexp = find_calexp(patch_dir)
    meas = find_meas(patch_dir, band, tract, patch)
    if calexp is None or meas is None:
        row.update(
            {
                "path_calexp": str(calexp) if calexp is not None else "",
                "path_meas": str(meas) if meas is not None else "",
                "error": "missing calexp or meas",
            }
        )
        return row, None
    row.update({"path_calexp": str(calexp), "path_meas": str(meas)})
    try:
        _image, mask, planes = read_calexp(calexp)
        tiles = score_tiles(mask, planes, weights, int(tile_size), int(stride))
        patch_score = tile_bad_score(mask, planes, weights)
        tile_scores = np.asarray([float(tile["score"]) for tile in tiles], dtype=np.float32)
        origin = calexp_physical_origin(calexp)
        sources = load_sources(meas, source_filter=source_filter, origin=origin)
        tile_counts = np.asarray(
            [count_sources_in_tile(sources, int(tile["x0"]), int(tile["y0"]), int(tile["x1"]), int(tile["y1"])) for tile in tiles],
            dtype=np.int64,
        )
        tile_pass = tile_scores < float(tile_score_max)
        pass_patch = patch_score < float(patch_score_max)
        pass_tiles = bool(tile_scores.size and np.all(tile_pass))
        row.update(
            {
                "status": "ok",
                "physical_origin_x": float(origin[0]),
                "physical_origin_y": float(origin[1]),
                "pass_patch_score": bool(pass_patch),
                "pass_tile_score": bool(pass_tiles),
                "pass_clean_region": bool(pass_patch and pass_tiles),
                "patch_score": float(patch_score),
                "patch_score_percent": float(patch_score * 100.0),
                "max_tile_score": float(np.max(tile_scores)) if tile_scores.size else "",
                "max_tile_score_percent": float(np.max(tile_scores) * 100.0) if tile_scores.size else "",
                "tile_count": int(len(tiles)),
                "tile_pass_count": int(np.sum(tile_pass)),
                "source_count": int(len(sources["x"])),
                "tile_source_mean_all": float(np.mean(tile_counts)) if tile_counts.size else "",
                "tile_source_mean_pass": float(np.mean(tile_counts[tile_pass])) if bool(np.any(tile_pass)) else "",
                "tile_source_median_all": float(np.median(tile_counts)) if tile_counts.size else "",
                "tile_source_max_all": int(np.max(tile_counts)) if tile_counts.size else "",
            }
        )
        return row, sources
    except Exception as exc:  # noqa: BLE001
        row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        return row, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count NB0387 sources in clean patch/tile bad-score regions.")
    parser.add_argument("--data-root", type=Path, default=Path("/data/shared/Subaru"))
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--band", default="NB0387")
    parser.add_argument("--patches", nargs="+", default=["all"])
    parser.add_argument("--out-dir", type=Path, default=Path("output/nb0387_clean_source_density"))
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=368)
    parser.add_argument("--patch-score-max", type=float, default=0.11)
    parser.add_argument("--tile-score-max", type=float, default=0.13)
    parser.add_argument("--source-filter", choices=("nchild0", "primary", "nchild0_primary", "all"), default="nchild0")
    parser.add_argument("--random-seed", type=int, default=23)
    parser.add_argument("--random-reg-count", type=int, default=3)
    parser.add_argument("--large-area-as-point", type=float, default=10000.0, help="Draw sources with Kron ellipse area above this value as points.")
    parser.add_argument("--bad-score-weights", nargs="*", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    band = normalize_band_dir(args.band)
    patches = parse_patches(args.patches)
    weights = parse_score_weights(args.bad_score_weights)
    rows: list[dict[str, object]] = []
    sources_by_patch: dict[str, dict[str, np.ndarray]] = {}
    for index, patch in enumerate(patches, start=1):
        row, sources = analyze_patch(
            data_root=args.data_root,
            tract=str(args.tract),
            band=band,
            patch=patch,
            weights=weights,
            tile_size=int(args.tile_size),
            stride=int(args.stride),
            patch_score_max=float(args.patch_score_max),
            tile_score_max=float(args.tile_score_max),
            source_filter=str(args.source_filter),
        )
        rows.append(row)
        if sources is not None and row.get("pass_clean_region") is True:
            sources_by_patch[patch] = sources
        if row.get("status") == "ok":
            print(
                f"[{index}/{len(patches)}] {band} {patch}: pass={row['pass_clean_region']} "
                f"patch={float(row['patch_score_percent']):.2f}% max_tile={float(row['max_tile_score_percent']):.2f}% "
                f"sources={row['source_count']} tile_mean={float(row['tile_source_mean_all']):.1f}",
                flush=True,
            )
        else:
            print(f"[{index}/{len(patches)}] {band} {patch}: {row['status']} {row.get('error', '')}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: tuple(int(part) for part in str(row["patch"]).split(",")) if "," in str(row["patch"]) else (99, 99))
    write_csv(args.out_dir / f"{band}_{args.tract}_clean_source_density.csv", rows)
    clean_rows = [row for row in rows if row.get("pass_clean_region") is True]
    write_csv(args.out_dir / f"{band}_{args.tract}_clean_pass_patches.csv", clean_rows)

    rng = random.Random(int(args.random_seed))
    sample_rows = rng.sample(clean_rows, k=min(int(args.random_reg_count), len(clean_rows))) if clean_rows else []
    reg_dir = args.out_dir / "random_clean_patch_kron_regs"
    for row in sample_rows:
        patch = str(row["patch"])
        sources = sources_by_patch.get(patch)
        if sources is None:
            _row, sources = analyze_patch(
                data_root=args.data_root,
                tract=str(args.tract),
                band=band,
                patch=patch,
                weights=weights,
                tile_size=int(args.tile_size),
                stride=int(args.stride),
                patch_score_max=float(args.patch_score_max),
                tile_score_max=float(args.tile_score_max),
                source_filter=str(args.source_filter),
            )
        if sources is None:
            continue
        safe_patch = patch.replace(",", "_")
        reg_path = reg_dir / f"{band}_{args.tract}_{safe_patch}_gt_kron_sdss_ellipses.reg"
        write_kron_ellipse_reg(
            reg_path,
            sources,
            title=(
                f"{band} {args.tract} {patch} GT Kron+Sdss ellipses; "
                f"area=pi*kron_radius^2; source_filter={args.source_filter}; n={len(sources['x'])}"
            ),
            point_area_min=float(args.large_area_as_point),
        )
        print(f"wrote {reg_path}")

    if clean_rows:
        source_counts = np.asarray([int(row["source_count"]) for row in clean_rows], dtype=float)
        tile_means = np.asarray([float(row["tile_source_mean_all"]) for row in clean_rows], dtype=float)
        print(
            f"clean patches={len(clean_rows)}/{len(rows)}; "
            f"source_count mean/median={source_counts.mean():.1f}/{np.median(source_counts):.1f}; "
            f"tile_mean mean/median={tile_means.mean():.1f}/{np.median(tile_means):.1f}"
        )
    else:
        print("no clean patches matched the score cuts")
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
