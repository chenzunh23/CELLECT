#!/usr/bin/env python3
"""Visualize denoised false-negative GT shapes on selected cutouts."""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cellect")

import numpy as np
import torch
from astropy.table import Table
from torch.utils.data import DataLoader


CELLECT_ROOT = Path("/home/czh23/CELLECT")
if str(CELLECT_ROOT) not in sys.path:
    sys.path.insert(0, str(CELLECT_ROOT))

from astro_train_data import AstroCutoutDataset, CutoutRecord, collate_cutouts, discover_cutout_records  # noqa: E402
from astro_train_ops import detect_centers  # noqa: E402
from zangetsu_demo import visualize_sam_cellect as vis  # noqa: E402


DEFAULT_BANDS = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y")
DEFAULT_CHECKPOINT = Path("/nvme0/zc/scarlet/ckpts/sam_new_bkgd_cdn_0702/epoch_0018.pt")
DEFAULT_CONFIG = Path("/nvme0/zc/scarlet/ckpts/sam_new_bkgd_cdn_0702/run_config.json")
DEFAULT_DATA_ROOT = Path("/nvme0/zc/scarlet/preprocessed")
DEFAULT_IMAGE_CACHE = Path("/nvme0/zc/scarlet/cellect_zscale_cache")
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "output/denoised_fn_shapes_0702_epoch0018"
DEFAULT_SAM_TILE = "group_01_sam_x18204_y20924"
DEFAULT_REUSE_SAM_DIR = (
    CELLECT_ROOT
    / "zangetsu_demo/output/sam_new_bkgd_cdn_0702/sam/epoch_0018/denoised"
)


def _base_tile_name(tile_name: str) -> str:
    parts = str(tile_name).split("_", 2)
    if len(parts) == 3 and parts[0] == "group" and parts[1].isdigit():
        return parts[2]
    return str(tile_name)


def _select_records(
    records: Sequence[CutoutRecord],
    *,
    patches: Sequence[str],
    group: str,
    extra_tiles: int,
    sam_tile: str,
) -> list[CutoutRecord]:
    denoised = [
        rec
        for rec in records
        if rec.dataset_source == "denoised"
        and rec.patch in set(patches)
        and (not group or rec.tile_name.startswith(f"{group}_"))
    ]
    selected: list[CutoutRecord] = []
    seen: set[tuple[str, str]] = set()

    for rec in denoised:
        if rec.patch == "4,5" and rec.tile_name == sam_tile:
            selected.append(rec)
            seen.add((rec.patch, rec.tile_name))
            break

    remaining_budget = max(0, int(extra_tiles))
    per_patch = {patch: remaining_budget // max(1, len(patches)) for patch in patches}
    for patch in patches[: remaining_budget % max(1, len(patches))]:
        per_patch[patch] += 1

    for patch in patches:
        candidates = [
            rec
            for rec in denoised
            if rec.patch == patch
            and (rec.patch, rec.tile_name) not in seen
            and _base_tile_name(rec.tile_name).startswith("grid_")
        ]
        if not candidates:
            candidates = [rec for rec in denoised if rec.patch == patch and (rec.patch, rec.tile_name) not in seen]
        if not candidates:
            continue
        med_x = float(np.median([rec.x0 for rec in candidates]))
        med_y = float(np.median([rec.y0 for rec in candidates]))
        candidates.sort(key=lambda rec: ((rec.x0 - med_x) ** 2 + (rec.y0 - med_y) ** 2, rec.tile_name))
        for rec in candidates[: per_patch.get(patch, 0)]:
            selected.append(rec)
            seen.add((rec.patch, rec.tile_name))

    if len(selected) < 1 + remaining_budget:
        for rec in denoised:
            key = (rec.patch, rec.tile_name)
            if key in seen:
                continue
            selected.append(rec)
            seen.add(key)
            if len(selected) >= 1 + remaining_budget:
                break
    return selected


def _exclude_key_values(values: Sequence[str]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if "/" in text:
            patch, tile = text.split("/", 1)
            out.add((patch.strip(), tile.strip()))
        else:
            out.add(("", text))
    return out


def _is_excluded(rec: CutoutRecord, excluded: set[tuple[str, str]]) -> bool:
    return (rec.patch, rec.tile_name) in excluded or ("", rec.tile_name) in excluded


def _select_random_records(
    records: Sequence[CutoutRecord],
    *,
    patches: Sequence[str],
    group: str,
    total: int,
    seed: int,
    exclude_tiles: Sequence[str],
) -> list[CutoutRecord]:
    rng = random.Random(int(seed))
    excluded = _exclude_key_values(exclude_tiles)
    patch_list = [str(patch) for patch in patches]
    denoised = [
        rec
        for rec in records
        if rec.dataset_source == "denoised"
        and rec.patch in set(patch_list)
        and (not group or rec.tile_name.startswith(f"{group}_"))
        and _base_tile_name(rec.tile_name).startswith("grid_")
        and not _is_excluded(rec, excluded)
    ]
    if int(total) <= 0:
        return []
    per_patch = {patch: int(total) // max(1, len(patch_list)) for patch in patch_list}
    for patch in patch_list[: int(total) % max(1, len(patch_list))]:
        per_patch[patch] += 1
    selected: list[CutoutRecord] = []
    seen: set[tuple[str, str]] = set()
    for patch in patch_list:
        candidates = [rec for rec in denoised if rec.patch == patch]
        rng.shuffle(candidates)
        for rec in candidates[: per_patch.get(patch, 0)]:
            selected.append(rec)
            seen.add((rec.patch, rec.tile_name))
    if len(selected) < int(total):
        remaining = [rec for rec in denoised if (rec.patch, rec.tile_name) not in seen]
        rng.shuffle(remaining)
        selected.extend(remaining[: int(total) - len(selected)])
    selected.sort(key=lambda rec: (rec.patch, rec.tile_name))
    return selected


def _dataset_for_record(rec: CutoutRecord, cfg: dict, image_cache_dir: Path | None) -> DataLoader:
    ds = AstroCutoutDataset(
        [rec],
        fits_hdu=int(cfg.get("fits_hdu", 1)),
        confidence_levels=5,
        ellipse_sigma=float(cfg.get("ellipse_sigma", 2.0)),
        core_radius=int(cfg.get("core_radius", 2)),
        shape_source=str(cfg.get("shape_source", "kron")),
        source_filter=str(cfg.get("source_filter", "nchild0")),
        image_cache_dir=image_cache_dir,
        load_eval_ignore_sources=True,
        augment=False,
    )
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_cutouts)


def _read_raw_or_batch_image(batch: dict, band_idx: int, cfg: dict) -> np.ndarray:
    raw, _source = vis._raw_band_image_from_batch(batch, band_idx, cfg)
    if raw is not None:
        return raw
    return batch["image"][0, band_idx].detach().cpu().numpy().astype(np.float32)


def _target_visibility_centers(rec: CutoutRecord, band_idx: int) -> tuple[np.ndarray, np.ndarray]:
    path = Path(rec.band_target_paths[band_idx]) if band_idx < len(rec.band_target_paths) else Path()
    if not path.exists():
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    with np.load(path) as data:
        center_only = (
            np.asarray(data["visibility_center_only_centers"], dtype=np.float32).reshape(-1, 2)
            if "visibility_center_only_centers" in data
            else np.zeros((0, 2), dtype=np.float32)
        )
        ignored = (
            np.asarray(data["visibility_ignore_centers"], dtype=np.float32).reshape(-1, 2)
            if "visibility_ignore_centers" in data
            else np.zeros((0, 2), dtype=np.float32)
        )
    return center_only, ignored


def _visibility_keep(cls: str, mode: str) -> bool:
    if mode == "raw":
        return True
    if mode == "snr_ge2":
        return cls in {"clean", "center_only"}
    if mode == "snr_ge3":
        return cls == "clean"
    raise ValueError(f"unknown visibility mode: {mode}")


def _load_gt_rows(
    rec: CutoutRecord,
    band_idx: int,
    *,
    visibility_filter: str,
    visibility_match_radius: float,
) -> tuple[Table, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    if band_idx >= len(rec.band_meas_paths) or not rec.band_meas_paths[band_idx]:
        raise FileNotFoundError(f"No band meas catalog path for {rec.name} band index {band_idx}")
    rows, x, y = vis._local_rows(vis._table(Path(rec.band_meas_paths[band_idx])), rec.tile_name)
    points = np.column_stack([x, y]).astype(np.float32) if len(x) else np.zeros((0, 2), dtype=np.float32)
    center_only, ignored = _target_visibility_centers(rec, band_idx)
    ignored_match = vis._within_any_radius(points, ignored, float(visibility_match_radius))
    center_only_match = vis._within_any_radius(points, center_only, float(visibility_match_radius)) & ~ignored_match
    classes = np.full((len(rows),), "clean", dtype=object)
    classes[center_only_match] = "center_only"
    classes[ignored_match] = "ignore"
    keep = np.asarray([_visibility_keep(str(cls), visibility_filter) for cls in classes], dtype=bool)
    stats = {
        "raw_gt": int(len(rows)),
        "visibility_clean_gt": int(np.count_nonzero(classes == "clean")),
        "visibility_center_only_gt": int(np.count_nonzero(classes == "center_only")),
        "visibility_ignore_gt": int(np.count_nonzero(classes == "ignore")),
        "filtered_gt": int(np.count_nonzero(keep)),
    }
    return rows[keep], x[keep], y[keep], classes[keep], stats


def _row_shape(row, x: float, y: float) -> tuple[float, float, float, float, float]:
    names = set(row.colnames) if hasattr(row, "colnames") else set()
    major = vis._safe_float(row["ellipse_major_sigma"], 4.0) if "ellipse_major_sigma" in names else 4.0
    minor = vis._safe_float(row["ellipse_minor_sigma"], 4.0) if "ellipse_minor_sigma" in names else 4.0
    theta = vis._safe_float(row["ellipse_theta"], 0.0) if "ellipse_theta" in names else 0.0
    return float(x), float(y), max(abs(float(major)), 1.0), max(abs(float(minor)), 1.0), float(theta)


def _row_mag(row, *, zero_point: float) -> float:
    names = set(row.colnames) if hasattr(row, "colnames") else set()
    if "pu_mag" in names:
        mag = vis._safe_float(row["pu_mag"], float("nan"))
        if math.isfinite(mag):
            return float(mag)
    _flux, mag = vis._gt_ap2_flux_mag(row, zero_point=float(zero_point))
    return float(mag) if math.isfinite(float(mag)) else float("nan")


def _mag_text(mag: float) -> str:
    return f"{float(mag):.2f}" if math.isfinite(float(mag)) else "mag=nan"


def _draw_shapes_png(
    path: Path,
    image: np.ndarray,
    reference_shapes: Sequence[tuple[tuple[float, float, float, float, float], str, bool, float]],
    *,
    title_lines: Sequence[str],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cellect")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    base = vis._zscale_image(np.asarray(image, dtype=np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 7.2), dpi=160)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.88])
    ax.imshow(base, origin="lower", cmap="gray", interpolation="nearest")

    for (x, y, major, minor, theta), cls, missed, _mag in reference_shapes:
        if missed:
            continue
        color = "#1de6ff" if cls == "clean" else "#ffd84d"
        ax.add_patch(
            Ellipse(
                (x, y),
                width=2.0 * major,
                height=2.0 * minor,
                angle=math.degrees(theta),
                fill=False,
                edgecolor=color,
                linewidth=0.9,
                alpha=0.75,
            )
        )
    for (x, y, major, minor, theta), cls, missed, mag in reference_shapes:
        if not missed:
            continue
        color = "#ff3030" if cls == "clean" else "#ff45e6"
        ax.add_patch(
            Ellipse(
                (x, y),
                width=2.0 * major,
                height=2.0 * minor,
                angle=math.degrees(theta),
                fill=False,
                edgecolor=color,
                linewidth=1.8,
                alpha=0.95,
            )
        )
        ax.text(
            min(max(x + major + 2.0, 2.0), image.shape[1] - 2.0),
            min(max(y + minor + 2.0, 2.0), image.shape[0] - 2.0),
            _mag_text(mag),
            color="white",
            fontsize=6.5,
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "black", "edgecolor": "none", "alpha": 0.62},
        )

    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.set_axis_off()
    fig.suptitle("\n".join(str(line) for line in title_lines), fontsize=10, y=0.985, linespacing=1.15)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_shape_reg(
    path: Path,
    header: str,
    clean_rows: Table,
    clean_x: np.ndarray,
    clean_y: np.ndarray,
    classes: np.ndarray,
    missed: set[int],
    mags: np.ndarray,
) -> None:
    lines = vis.REG_HEADER + [
        f"# {header}",
        "# cyan=matched clean; yellow=matched center-only; red=FN clean; magenta=FN center-only",
    ]
    for gi in range(len(clean_rows)):
        cls = str(classes[gi]) if gi < len(classes) else "clean"
        if gi in missed:
            color = "red" if cls == "clean" else "magenta"
        else:
            color = "cyan" if cls == "clean" else "yellow"
        width = 2 if gi in missed else 1
        if gi in missed:
            x, y, major, minor, theta = _row_shape(clean_rows[gi], float(clean_x[gi]), float(clean_y[gi]))
            lines.append(
                vis._ellipse_line(
                    x,
                    y,
                    major,
                    minor,
                    theta,
                    color=color,
                    width=width,
                    text=f"mag={_mag_text(float(mags[gi]))}",
                )
            )
        else:
            lines.append(vis._ellipse_from_row(clean_rows[gi], float(clean_x[gi]), float(clean_y[gi]), color, width=width))
    vis._write_text(path, lines)


def _coadd_record_for(records: Sequence[CutoutRecord], rec: CutoutRecord) -> CutoutRecord | None:
    base_tile = _base_tile_name(rec.tile_name)
    preferred_root = f"{rec.tract}/{rec.patch}" if rec.tract and rec.patch else ""
    candidates = [
        item
        for item in records
        if item.dataset_source == "coadd" and item.patch == rec.patch and item.tile_name == base_tile
    ]
    for item in candidates:
        if item.relative_root == preferred_root:
            return item
    return candidates[0] if candidates else None


def _copy_input_fits(
    *,
    records: Sequence[CutoutRecord],
    selected: Sequence[CutoutRecord],
    bands: Sequence[str],
    band: str,
    out_dir: Path,
) -> Path:
    band_idx = list(bands).index(band)
    root = out_dir / "input_fits"
    entries: list[tuple[str, str, str, str, str, str]] = []
    for rec in selected:
        patch_key = rec.patch.replace(",", "_")
        denoised_src = Path(rec.image_paths[band_idx])
        denoised_dst = root / "denoised" / patch_key / f"{rec.tile_name}_{band}.fits"
        entries.append(("denoised_image", rec.patch, rec.tile_name, band, str(denoised_src), str(denoised_dst)))
        coadd_rec = _coadd_record_for(records, rec)
        if coadd_rec is not None and band_idx < len(coadd_rec.image_paths):
            coadd_src = Path(coadd_rec.image_paths[band_idx])
            coadd_dst = root / "coadd" / patch_key / f"{_base_tile_name(rec.tile_name)}_{band}.fits"
            entries.append(("coadd_image", rec.patch, _base_tile_name(rec.tile_name), band, str(coadd_src), str(coadd_dst)))
    for _kind, _patch, _tile, _band, src, dst in entries:
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_path)
    manifest = root / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "patch", "tile", "band", "source_path", "copied_path"])
        writer.writerows(entries)
    return manifest


def _region_ellipse_points(path: Path) -> np.ndarray:
    pattern = re.compile(r"ellipse\(\s*([+-]?[0-9.]+)\s*,\s*([+-]?[0-9.]+)\s*,")
    points: list[tuple[float, float]] = []
    if not path.exists():
        return np.zeros((0, 2), dtype=np.float32)
    for line in path.read_text().splitlines():
        if not line.startswith("ellipse("):
            continue
        match = pattern.search(line)
        if match:
            points.append((float(match.group(1)) - 1.0, float(match.group(2)) - 1.0))
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def _region_circle_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.startswith("circle("))


def _legacy_sam_output_paths(args: argparse.Namespace, rec: CutoutRecord, band: str) -> tuple[Path, Path]:
    safe_patch = rec.patch.replace(",", "_")
    safe_band = band.replace("-", "_")
    prefix = f"epoch_0018_denoised_{safe_patch}_{rec.tile_name}_{safe_band}"
    root = Path(args.reuse_sam_output_dir).expanduser()
    return root / f"{prefix}_fn_shape.reg", root / f"{prefix}_centers.reg"


def _legacy_missed_set(clean_xy: np.ndarray, fn_points: np.ndarray, radius: float) -> set[int]:
    missed: set[int] = set()
    if clean_xy.size == 0 or fn_points.size == 0:
        return missed
    radius2 = float(radius) * float(radius)
    for point in fn_points:
        d2 = np.sum((clean_xy - point[None, :]) ** 2, axis=1)
        if d2.size == 0:
            continue
        idx = int(np.argmin(d2))
        if float(d2[idx]) <= radius2:
            missed.add(idx)
    return missed


@torch.no_grad()
def _run_record(
    *,
    rec: CutoutRecord,
    model: torch.nn.Module,
    cfg: dict,
    bands: Sequence[str],
    band: str,
    device: torch.device,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    band_idx = list(bands).index(band)
    loader = _dataset_for_record(rec, cfg, args.image_cache_dir)
    batch = next(iter(loader))
    raw_image = _read_raw_or_batch_image(batch, band_idx, cfg)
    clean_rows, clean_x, clean_y, gt_classes, gt_stats = _load_gt_rows(
        rec,
        band_idx,
        visibility_filter=str(args.gt_visibility_filter),
        visibility_match_radius=float(args.gt_visibility_match_radius),
    )
    clean_xy = np.column_stack([clean_x, clean_y]).astype(np.float32) if len(clean_x) else np.zeros((0, 2), np.float32)

    use_legacy = (
        bool(args.reuse_sam_output)
        and rec.patch == "4,5"
        and rec.tile_name == str(args.sam_tile)
        and str(band) == "HSC-I"
    )
    pred_count: int | None = None
    if use_legacy:
        legacy_fn_reg, legacy_center_reg = _legacy_sam_output_paths(args, rec, band)
        fn_points = _region_ellipse_points(legacy_fn_reg)
        if fn_points.size > 0:
            missed = _legacy_missed_set(clean_xy, fn_points, float(args.legacy_fn_match_radius))
            pred_count = _region_circle_count(legacy_center_reg)
            if pred_count <= 0:
                pred_count = None
        else:
            missed = set()
            use_legacy = False
    else:
        missed = set()

    if not use_legacy:
        if model is None:
            raise RuntimeError("model is required when not reusing legacy SAM output")
        image = batch["image"].to(device=device, dtype=torch.float32)
        outputs = model(image)
        outputs_i = vis._band_outputs(outputs, band_idx)
        threshold = float(args.threshold if args.threshold is not None else cfg.get("confidence_threshold", 2.0))
        nms_radius = int(args.nms_radius if args.nms_radius is not None else cfg.get("nms_radius", 1))
        confidence_score = str(args.confidence_score or cfg.get("confidence_score", "cellect"))
        center_refinement = str(args.center_refinement or cfg.get("center_refinement", "softargmax"))
        center_refinement_radius = int(
            args.center_refinement_radius
            if args.center_refinement_radius is not None
            else cfg.get("center_refinement_radius", 1)
        )
        pred_list = detect_centers(
            outputs_i,
            threshold=threshold,
            nms_radius=nms_radius,
            confidence_score=confidence_score,
            center_refinement=center_refinement,
            center_refinement_radius=center_refinement_radius,
        )
        pred_xy = np.asarray(pred_list[0], dtype=np.float32).reshape(-1, 2)
        _pred_to_gt, used_gt = vis._greedy_match(pred_xy, clean_xy, float(args.match_radius))
        missed = {idx for idx in range(len(clean_rows)) if idx not in used_gt}
        pred_count = int(len(pred_xy))

    reference_shapes = [
        (
            _row_shape(clean_rows[i], clean_x[i], clean_y[i]),
            str(gt_classes[i]),
            i in missed,
            _row_mag(clean_rows[i], zero_point=float(args.gt_photometry_zero_point)),
        )
        for i in range(len(clean_rows))
    ]
    mags = np.asarray([item[3] for item in reference_shapes], dtype=np.float32)
    fn_clean = int(sum(1 for idx in missed if str(gt_classes[idx]) == "clean"))
    fn_center_only = int(sum(1 for idx in missed if str(gt_classes[idx]) == "center_only"))
    fn_ignore = int(sum(1 for idx in missed if str(gt_classes[idx]) == "ignore"))

    safe_patch = rec.patch.replace(",", "_")
    safe_band = band.replace("-", "_")
    prefix = f"epoch_0018_denoised_{safe_patch}_{rec.tile_name}_{safe_band}"
    patch_out = out_dir / rec.patch
    png_path = patch_out / f"{prefix}_clean_gt_fn_shapes.png"
    reg_path = patch_out / f"{prefix}_clean_gt_fn_shapes.reg"
    title_line1 = f"denoised {rec.patch}/{rec.tile_name} {band}"
    title_line2 = (
        f"reference GT={len(clean_rows)} FN={len(missed)} "
        f"cleanFN={fn_clean} centerOnlyFN={fn_center_only} pred={pred_count}"
    )
    if use_legacy:
        title_line2 += " legacy-zangetsu"
    title = f"{title_line1}: {title_line2}"
    _draw_shapes_png(png_path, raw_image, reference_shapes, title_lines=(title_line1, title_line2))
    _write_shape_reg(reg_path, title, clean_rows, clean_x, clean_y, gt_classes, missed, mags)

    row = {
        "patch": rec.patch,
        "tile": rec.tile_name,
        "band": band,
        "pred": int(pred_count or 0),
        "reference_gt": int(len(clean_rows)),
        "fn": int(len(missed)),
        "fn_clean": fn_clean,
        "fn_center_only": fn_center_only,
        "fn_ignore": fn_ignore,
        "legacy_sam_output": bool(use_legacy),
        "png": str(png_path),
        "reg": str(reg_path),
        **gt_stats,
    }
    print(
        f"[done] {rec.patch}/{rec.tile_name} {band}: pred={row['pred']} reference_gt={row['reference_gt']} fn={row['fn']}",
        flush=True,
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--image-cache-dir", type=Path, default=DEFAULT_IMAGE_CACHE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--patches", nargs="+", default=["4,5", "8,4"])
    parser.add_argument("--group", default="group_01")
    parser.add_argument("--sam-tile", default=DEFAULT_SAM_TILE)
    parser.add_argument("--extra-tiles", type=int, default=4)
    parser.add_argument("--random-tiles-total", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=704)
    parser.add_argument("--exclude-tiles", nargs="*", default=())
    parser.add_argument("--band", default="HSC-I", choices=DEFAULT_BANDS)
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--nms-radius", type=int, default=None)
    parser.add_argument("--confidence-score", choices=("cellect", "raw", "ordinal_prob", "ordinal_expectation"), default=None)
    parser.add_argument("--center-refinement", choices=("integer", "softargmax"), default=None)
    parser.add_argument("--center-refinement-radius", type=int, default=None)
    parser.add_argument("--match-radius", type=float, default=0.5 / 0.168)
    parser.add_argument("--gt-visibility-filter", default="snr_ge2", choices=("raw", "snr_ge2", "snr_ge3"))
    parser.add_argument("--gt-visibility-match-radius", type=float, default=1.0)
    parser.add_argument("--reuse-sam-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-sam-output-dir", type=Path, default=DEFAULT_REUSE_SAM_DIR)
    parser.add_argument("--legacy-fn-match-radius", type=float, default=1.0)
    parser.add_argument("--gt-photometry-zero-point", type=float, default=27.0)
    parser.add_argument("--copy-input-fits", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.image_cache_dir = args.image_cache_dir.expanduser().resolve() if args.image_cache_dir else None
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()

    cfg = vis._read_config(args.config)
    bands = tuple(str(band) for band in args.bands)
    device = torch.device(args.device)
    print(f"[load] records from {args.data_root}", flush=True)
    records = discover_cutout_records(args.data_root, bands=bands)
    if int(args.random_tiles_total) > 0:
        selected = _select_random_records(
            records,
            patches=[str(patch) for patch in args.patches],
            group=str(args.group),
            total=int(args.random_tiles_total),
            seed=int(args.random_seed),
            exclude_tiles=tuple(str(item) for item in args.exclude_tiles),
        )
    else:
        selected = _select_records(
            records,
            patches=[str(patch) for patch in args.patches],
            group=str(args.group),
            extra_tiles=int(args.extra_tiles),
            sam_tile=str(args.sam_tile),
        )
    if not selected:
        raise RuntimeError("No denoised records selected")
    print("[select] " + ", ".join(f"{rec.patch}/{rec.tile_name}" for rec in selected), flush=True)

    print(f"[load] model {args.checkpoint}", flush=True)
    model = vis._make_model(cfg, args.checkpoint, device, bands)
    rows = [
        _run_record(
            rec=rec,
            model=model,
            cfg=cfg,
            bands=bands,
            band=str(args.band),
            device=device,
            out_dir=args.out_dir,
            args=args,
        )
        for rec in selected
    ]

    summary_path = args.out_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summary] {summary_path}", flush=True)
    if bool(args.copy_input_fits):
        manifest = _copy_input_fits(records=records, selected=selected, bands=bands, band=str(args.band), out_dir=args.out_dir)
        print(f"[input-fits] {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
