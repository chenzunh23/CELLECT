#!/usr/bin/env python3
"""Batch SAM-CELLECT detection and photometry evaluation on CELLECT cutouts.

The script reuses ``CELLECT/zangetsu_demo/visualize_sam_cellect.py`` for model
inference, mask generation, center matching, and per-source photometry.  It then
aggregates all per-tile CSVs into magnitude-binned completeness/purity plots and
flux-ratio diagnostics.

Usage example:
python evaluate_sam_cellect_photometry.py \
    --ckpt-dir /path/to/ckpt_dir \
    --data-root /nvme0/zc/scarlet/preprocessed \
    --patch 4,5 --group 01 \
    --datasets coadd denoised
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/cellect_matplotlib_{os.getuid()}")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


CELLECT_ROOT = Path("/home/czh23/CELLECT")
CELLECT_DEMO = CELLECT_ROOT / "zangetsu_demo"
for _path in (CELLECT_DEMO, CELLECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import zangetsu_demo.visualize_sam_cellect as vis  # noqa: E402
from astro_train_data import astro_zscale_preprocess, read_fits_bands  # noqa: E402


def _safe_tile_for_dataset(dataset_name: str, args: argparse.Namespace) -> str:
    tile_name = getattr(args, "tile_name", None)
    if tile_name:
        return str(tile_name)
    if str(dataset_name) == "coadd":
        return vis.TILE
    variant_group = getattr(args, "variant_group", None)
    if variant_group:
        return f"{variant_group}_{vis.TILE}"
    return vis.TILE


vis._tile_for_dataset = _safe_tile_for_dataset


DEFAULT_DATA_ROOT = Path("/nvme0/zc/scarlet/preprocessed")
DEFAULT_OUT_DIR = CELLECT_ROOT / "output/sam_cellect_patch45_group1_photometry_eval"
DEFAULT_BANDS = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y")
DEFAULT_DATASETS = ("coadd", "denoised")
DEFAULT_IMAGE_PHOTOMETRY_ZERO_POINT = 31.4
DEFAULT_GT_PHOTOMETRY_ZERO_POINT = 27.0
DEFAULT_GT_FLUX_SCALE_FOR_RATIOS =  1 # 10.0 ** (0.4 * (31.4 - 27.0))
RATIO_SPECS = (
    ("pred_ap2_over_gt_ap2", "ap_flux", "gt_ap2_flux", "predicted ap2 / scaled GT ap2"),
    ("pred_kron_over_gt_kron", "kron_flux", "gt_kron_flux", "predicted kron / scaled GT kron"),
    ("mask_over_pred_ap2", "mask_flux", "ap_flux", "mask / predicted ap2"),
    ("mask_over_pred_kron", "mask_flux", "kron_flux", "mask / predicted kron"),
    ("mask_over_gt_ap2", "mask_flux", "gt_ap2_flux", "mask / scaled GT ap2"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, default=CELLECT_ROOT / "output/ckpts/SAM_per_band_debug_0612")
    parser.add_argument("--config", type=Path, default=None, help="Defaults to <ckpt-dir>/run_config.json")
    parser.add_argument("--checkpoint", "-c", type=Path, action="append", default=None, help="Can be passed multiple times")
    parser.add_argument("--checkpoint-label", "-l", action="append", default=None, help="One label per --checkpoint")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--data-format", choices=("cutout", "zarr"), default="cutout")
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help="Optional legacy preprocessed root containing GT catalogs when --data-format=zarr.",
    )
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=None,
        help="Unfiltered source catalog root joined to authoritative Zarr source IDs for magnitude metrics.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patch", default="4,5")
    parser.add_argument("--group", default="01", help="Cutout group to run, e.g. 01. Use 'all' to disable group filtering.")
    parser.add_argument("--tile-name", default=None, help="Compatibility with visualize_sam_cellect; normally set per discovered tile.")
    parser.add_argument("--variant-group", default=None, help="Compatibility with visualize_sam_cellect; inferred from --group when omitted.")
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    parser.add_argument("--band", default="HSC-I")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-tiles", type=int, default=None, help="Optional debug cap after filtering.")
    parser.add_argument(
        "--tile-workers",
        type=int,
        default=1,
        help="Number of tile chunks to process in parallel per checkpoint/dataset. Each worker loads one model copy.",
    )
    parser.add_argument(
        "--tile-worker-devices",
        nargs="*",
        default=None,
        help="Optional devices assigned round-robin to tile workers, e.g. cuda:0 cuda:1. Defaults to --device for every worker.",
    )

    parser.add_argument("--match-radius", type=float, default=vis.MATCH_RADIUS_PIX)
    parser.add_argument("--mag-min", type=float, default=22.0)
    parser.add_argument("--mag-max", type=float, default=30.0)
    parser.add_argument("--bin-size", type=float, default=0.5)
    parser.add_argument(
        "--reverse-mag-axis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Plot magnitude axes right-to-left by default, so fainter larger magnitudes appear on the left.",
    )
    parser.add_argument("--curve-gt-mag-col", default="gt_ap2mag")
    parser.add_argument("--curve-pred-mag-col", default="ap_abmag")
    parser.add_argument(
        "--gt-visibility-filter",
        choices=("raw", "snr_ge2", "snr_ge3"),
        default="snr_ge2",
        help="GT denominator for completeness. raw keeps the full catalog; snr_ge2 keeps clean+center-only sources; snr_ge3 keeps clean sources only.",
    )
    parser.add_argument(
        "--gt-visibility-match-radius",
        type=float,
        default=1.0,
        help="Pixel radius for matching preprocessed visibility-filter centers back to photometry GT rows.",
    )
    parser.add_argument("--ratio-source", choices=("isolated", "matched"), default="isolated")
    parser.add_argument("--ratio-min", type=float, default=0.0)
    parser.add_argument("--ratio-max", type=float, default=3.0)
    parser.add_argument("--ratio-bins", type=int, default=80)
    parser.add_argument(
        "--gt-flux-scale-for-ratios",
        type=float,
        default=DEFAULT_GT_FLUX_SCALE_FOR_RATIOS,
        help="Multiply GT flux denominators by this factor before flux-ratio plots; default is 10**(0.4*(31.4-27.0)) for ZP 27 -> 31.4.",
    )

    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--nms-radius", type=int, default=None)
    parser.add_argument("--confidence-score", default=None)
    parser.add_argument("--center-refinement", default=None)
    parser.add_argument("--center-refinement-radius", type=int, default=None)
    parser.add_argument("--shape-display-scale", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--prompt-box-scale", type=float, default=1.0)
    parser.add_argument("--mask-prompt-center-only", action="store_true", default=None)
    parser.add_argument("--multimask", action="store_true", default=None)
    parser.add_argument("--singlemask", dest="multimask", action="store_false")
    parser.add_argument("--mask-chunk-size", type=int, default=128)
    parser.add_argument(
        "--skip-mask-decoder",
        action="store_true",
        help="Skip SAM mask decoding; completeness/purity and fixed-aperture magnitudes remain available.",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--min-mask-area", type=int, default=15)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.50)
    parser.add_argument("--pred-iou-thresh", type=float, default=None)
    parser.add_argument("--stability-score-thresh", type=float, default=None)
    parser.add_argument("--stability-score-offset", type=float, default=1.0)

    parser.add_argument("--disable-photometry", action="store_true")
    parser.add_argument(
        "--photometry-bg-mode",
        choices=("auto", "none", "annulus", "photutils", "photutils_annulus"),
        default="auto",
    )
    parser.add_argument("--photometry-ap-radius", type=float, default=6.0)
    parser.add_argument("--photometry-ann-r-in", type=float, default=10.0)
    parser.add_argument("--photometry-ann-r-out", type=float, default=15.0)
    parser.add_argument("--photometry-bkg-box-size", type=int, default=64)
    parser.add_argument("--photometry-bkg-filter-size", type=int, default=3)
    parser.add_argument("--photometry-sigma-clip", type=float, default=3.0)
    parser.add_argument("--photometry-method", choices=("center", "exact", "subpixel"), default="exact")
    parser.add_argument("--photometry-annulus-method", choices=("center", "exact", "subpixel"), default="center")
    parser.add_argument(
        "--photometry-zero-point",
        type=float,
        default=DEFAULT_IMAGE_PHOTOMETRY_ZERO_POINT,
        help="AB zero point for measured coadd/noisy/denoised image fluxes. Old LSST FITS use 31.4 for nJy-scale pixels.",
    )
    parser.add_argument(
        "--gt-photometry-zero-point",
        type=float,
        default=DEFAULT_GT_PHOTOMETRY_ZERO_POINT,
        help="AB zero point for GT catalog flux columns such as ap2 and Kron.",
    )
    parser.add_argument("--photometry-psf-factor", type=float, default=1.0)
    parser.add_argument("--tp-isolated-max-shape-iou", type=float, default=0.05)

    parser.add_argument("--center-radius", type=float, default=7.0)
    parser.add_argument("--overlay-alpha", type=float, default=0.38)
    parser.add_argument("--max-contour-vertices", type=int, default=128)
    parser.add_argument("--skip-visual-products", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-only", action="store_true", help="Skip inference and plot from existing summary CSVs.")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mag_bins(mag_min: float, mag_max: float, bin_size: float) -> list[tuple[float, float, float]]:
    edges = np.arange(float(mag_min), float(mag_max) + float(bin_size) * 0.5, float(bin_size))
    return [(float(lo), float(hi), float((lo + hi) * 0.5)) for lo, hi in zip(edges[:-1], edges[1:])]


def _mag_in(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.isfinite(values) & (values >= lo) & (values < hi)


def _discover_tiles(args: argparse.Namespace, dataset_name: str) -> list[str]:
    if str(getattr(args, "data_format", "cutout")) == "zarr":
        records = vis.discover_zarr_records(args.data_root, bands=args.bands)
        group_prefix = f"group_{int(args.group):02d}_" if str(args.group).lower() != "all" else ""
        if dataset_name == "coadd":
            group_prefix = ""
        tiles = sorted(
            {
                rec.tile_name
                for rec in records
                if str(rec.dataset_source) == str(dataset_name)
                and rec.patch == args.patch
                and (not group_prefix or rec.tile_name.startswith(group_prefix))
            }
        )
        return tiles[: int(args.max_tiles)] if args.max_tiles is not None else tiles
    if dataset_name == "coadd":
        tract_root = args.data_root / args.tract
    else:
        tract_root = args.data_root / dataset_name / args.tract
    if not tract_root.exists():
        print(f"[warn] missing dataset root: {tract_root}", flush=True)
        return []
    tiles: list[str] = []
    group_prefix = f"group_{int(args.group):02d}_" if str(args.group).lower() != "all" else ""
    if dataset_name == "coadd":
        group_prefix = ""  # coadd tiles are not grouped
    try:
        records = vis.discover_cutout_records(tract_root, bands=args.bands)
        for rec in records:
            if rec.patch != args.patch:
                continue
            if group_prefix and not rec.tile_name.startswith(group_prefix):
                continue
            tiles.append(rec.tile_name)
    except FileNotFoundError:
        cutout_dir = tract_root / args.patch / "cutouts"
        if cutout_dir.exists():
            for path in cutout_dir.iterdir():
                if not path.is_dir():
                    continue
                if group_prefix and not path.name.startswith(group_prefix):
                    continue
                tiles.append(path.name)
    tiles = sorted(set(tiles))
    if args.max_tiles is not None:
        tiles = tiles[: int(args.max_tiles)]
    return tiles


def _base_tile_name(tile_name: str) -> str:
    parts = tile_name.split("_", 2)
    if len(parts) == 3 and parts[0] == "group" and parts[1].isdigit():
        return parts[2]
    return tile_name


def _table(path: Path) -> "vis.Table":
    if path.suffix.lower() == ".csv":
        return vis.Table.read(path, format="ascii.csv")
    return vis._table(path)


def _clean_catalog_candidates(root: Path, dataset_name: str, band: str, tile_name: str) -> list[Path]:
    base_tile = _base_tile_name(tile_name)
    return [
        root / f"meas-{band}-{vis.TRACT}-{vis.PATCH}.fits",
        root / band / f"meas-{band}-{vis.TRACT}-{vis.PATCH}.fits",
        root / dataset_name / vis.TRACT / vis.PATCH / "reference_catalogs" / f"{base_tile}_meas.fits",
        root / vis.TRACT / vis.PATCH / "reference_catalogs" / f"{base_tile}_meas.fits",
        root / dataset_name / vis.TRACT / vis.PATCH / "reference_catalogs_csv" / f"{base_tile}_meas.csv",
        root / vis.TRACT / vis.PATCH / "reference_catalogs_csv" / f"{base_tile}_meas.csv",
        root / dataset_name / vis.TRACT / vis.PATCH / "band_reference_catalogs" / band / f"meas-{band}-{vis.TRACT}-{vis.PATCH}.fits",
        root / vis.TRACT / vis.PATCH / "band_reference_catalogs" / band / f"meas-{band}-{vis.TRACT}-{vis.PATCH}.fits",
    ]


def _load_clean_rows_compat(
    root: Path,
    dataset_name: str,
    band: str,
    tile_name: str,
    args: argparse.Namespace | None = None,
):
    catalog_root = (
        Path(args.reference_root)
        if args is not None and getattr(args, "reference_root", None) is not None
        else root
    )
    for path in _clean_catalog_candidates(catalog_root, dataset_name, band, tile_name):
        if path.exists():
            rows, x, y = vis._local_rows(_table(path), tile_name)
            if args is None:
                stats = {
                    "raw_gt": int(len(rows)),
                    "visibility_clean_gt": int(len(rows)),
                    "visibility_center_only_gt": 0,
                    "visibility_ignore_gt": 0,
                    "filtered_gt": int(len(rows)),
                }
                return rows, x, y, stats
            points = np.column_stack([x, y]).astype(np.float32) if len(x) else np.zeros((0, 2), dtype=np.float32)
            center_only, ignored = _visibility_centers(catalog_root, dataset_name, band, tile_name)
            radius = float(getattr(args, "gt_visibility_match_radius", 1.0))
            ignored_match = _within_any_radius(points, ignored, radius)
            center_only_match = _within_any_radius(points, center_only, radius) & ~ignored_match
            visibility_class = np.full((len(rows),), "clean", dtype=object)
            visibility_class[center_only_match] = "center_only"
            visibility_class[ignored_match] = "ignore"
            mode = str(getattr(args, "gt_visibility_filter", "snr_ge2"))
            keep = np.asarray([_visibility_keep(str(cls), mode) for cls in visibility_class], dtype=bool)
            stats = {
                "raw_gt": int(len(rows)),
                "visibility_clean_gt": int(np.count_nonzero(visibility_class == "clean")),
                "visibility_center_only_gt": int(np.count_nonzero(visibility_class == "center_only")),
                "visibility_ignore_gt": int(np.count_nonzero(visibility_class == "ignore")),
                "filtered_gt": int(np.count_nonzero(keep)),
            }
            # Keep raw row order so photometry gt_index matches gt_photometry.csv.
            # Visibility filtering is applied later in metric aggregation.
            return rows, x, y, stats
    searched = "\n  ".join(str(path) for path in _clean_catalog_candidates(catalog_root, dataset_name, band, tile_name))
    raise FileNotFoundError(f"No GT catalog found for {dataset_name}/{vis.PATCH}/{tile_name}/{band}. Searched:\n  {searched}")


def _visibility_target_candidates(root: Path, dataset_name: str, band: str, tile_name: str) -> list[Path]:
    base_tile = _base_tile_name(tile_name)
    return [
        root / dataset_name / vis.TRACT / vis.PATCH / "band_targets" / band / f"{tile_name}.npz",
        root / dataset_name / vis.TRACT / vis.PATCH / "band_targets" / band / f"{base_tile}.npz",
        root / vis.TRACT / vis.PATCH / "band_targets" / band / f"{base_tile}.npz",
        root / vis.TRACT / vis.PATCH / "band_targets" / band / f"{tile_name}.npz",
    ]


def _visibility_centers(root: Path, dataset_name: str, band: str, tile_name: str) -> tuple[np.ndarray, np.ndarray]:
    if str(dataset_name).lower() == "coadd":
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    for path in _visibility_target_candidates(root, dataset_name, band, tile_name):
        if not path.exists():
            continue
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
    return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)


def _within_any_radius(points: np.ndarray, centers: np.ndarray, radius: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    matched = np.zeros((points.shape[0],), dtype=bool)
    if points.size == 0 or centers.size == 0 or float(radius) < 0.0:
        return matched
    radius2 = float(radius) * float(radius)
    for center in centers:
        matched |= np.sum((points - center[None, :]) ** 2, axis=1) <= radius2
    return matched


def _visibility_keep(cls: str, mode: str) -> bool:
    cls = str(cls)
    mode = str(mode)
    if mode == "raw":
        return True
    if mode == "snr_ge2":
        return cls in {"clean", "center_only"}
    if mode == "snr_ge3":
        return cls == "clean"
    raise ValueError(f"unknown GT visibility filter: {mode!r}")


def _dataset_compat(
    root: Path,
    dataset_name: str,
    bands: Sequence[str],
    cfg: dict,
    image_cache_dir: Path | None = None,
    tile_name: str | None = None,
    data_format: str = "cutout",
):
    tile = str(tile_name) if tile_name else str(vis.TILE)
    if str(data_format) == "zarr":
        return vis._ORIGINAL_DATASET(
            root,
            dataset_name,
            bands,
            cfg,
            image_cache_dir=image_cache_dir,
            tile_name=tile,
            data_format="zarr",
        )
    tile_dirs = [
        root / dataset_name / vis.TRACT / vis.PATCH / "cutouts" / tile,
        root / vis.TRACT / vis.PATCH / "cutouts" / tile,
    ]
    tile_dir = next((path for path in tile_dirs if path.exists()), None)
    if tile_dir is None:
        return vis._ORIGINAL_DATASET(
            root,
            dataset_name,
            bands,
            cfg,
            image_cache_dir=image_cache_dir,
            tile_name=tile,
            data_format="cutout",
        )
    image_paths: list[str] = []
    for band in bands:
        band_dir = tile_dir / band
        matches = sorted(band_dir.glob("*.fits"))
        if not matches:
            searched = ", ".join(str(path / band) for path in tile_dirs)
            raise FileNotFoundError(f"No FITS image for {dataset_name}/{vis.PATCH}/{tile}/{band}; searched {searched}")
        image_paths.append(str(matches[0]))
    image_np = read_fits_bands(tuple(image_paths), hdu=int(cfg.get("fits_hdu", 1)))
    image = astro_zscale_preprocess(image_np).to(dtype=torch.float32)
    return [
        {
            "image": image.unsqueeze(0),
            "image_paths": [tuple(image_paths)],
            "dataset_source": [str(dataset_name)],
            "processing_id": torch.tensor(
                [1 if str(dataset_name).lower() == "denoised" else 0],
                dtype=torch.long,
            ),
        }
    ]


def _install_visualizer_io_patch() -> None:
    if not hasattr(vis, "_ORIGINAL_DATASET"):
        vis._ORIGINAL_DATASET = vis._dataset
    vis._dataset = _dataset_compat
    vis._load_clean_rows = _load_clean_rows_compat


def _checkpoint_items(args: argparse.Namespace) -> list[tuple[Path, str]]:
    if args.checkpoint:
        items = [(Path(args.ckpt_dir / path).expanduser(), Path(args.ckpt_dir / path).stem) for path in args.checkpoint]
        if args.checkpoint_label:
            if len(args.checkpoint_label) != len(items):
                raise ValueError("--checkpoint-label count must match --checkpoint count")
            items = [(path, label) for (path, _), label in zip(items, args.checkpoint_label)]
        return items
    ckpt_dir = args.ckpt_dir.expanduser().resolve()
    candidates = [(ckpt_dir / "best.pt", "best"), (ckpt_dir / "last.pt", "latest")]
    return [(path, label) for path, label in candidates if path.exists()]


def _resolve_config(args: argparse.Namespace) -> dict:
    ckpt_dir = args.ckpt_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve() if args.config else ckpt_dir / "run_config.json"
    return vis._read_config(config_path)


def _resolve_visual_defaults(args: argparse.Namespace, cfg: dict) -> None:
    loss_cfg = cfg.get("_top", {}).get("loss_weights", {})
    if not hasattr(args, "tile_name"):
        args.tile_name = None
    if not hasattr(args, "variant_group"):
        group = str(getattr(args, "group", "")).strip()
        if not group or group.lower() in {"all", "none"}:
            args.variant_group = ""
        elif group.startswith("group_"):
            args.variant_group = group
        else:
            args.variant_group = f"group_{int(group):02d}"
    if args.multimask is None:
        args.multimask = bool(loss_cfg.get("mask_multimask", not bool(cfg.get("disable_mask_multimask", False))))
    if args.mask_prompt_center_only is None:
        args.mask_prompt_center_only = bool(cfg.get("mask_prompt_center_only", False))
    if args.pred_iou_thresh is None and float(loss_cfg.get("mask_pred_iou", 0.0)) > 0.0:
        args.pred_iou_thresh = float(loss_cfg.get("mask_pred_iou_thresh", 0.8))
    if args.stability_score_thresh is None and float(loss_cfg.get("mask_stability", 0.0)) > 0.0:
        args.stability_score_thresh = float(loss_cfg.get("mask_stability_score_thresh", 0.95))
    args.native_sam_dir = None
    args.native_sam_dataset = "coadd"
    args.native_match_radius = args.match_radius
    args.return_gt_metric_rows = True


def _disable_visual_products() -> None:
    def _noop_text(_path: Path, _lines: Sequence[str]) -> None:
        return None

    def _noop_overlay(_path: Path, _image: np.ndarray, _masks: Sequence[np.ndarray], _alpha: float) -> None:
        return None

    vis._write_text = _noop_text
    vis._write_overlay = _noop_overlay


def _gt_mag_rows(args: argparse.Namespace, dataset_name: str, tile_name: str) -> list[dict[str, object]]:
    vis.TRACT = args.tract
    vis.PATCH = args.patch
    vis.TILE = tile_name
    reference_root = args.reference_root if getattr(args, "reference_root", None) is not None else args.data_root
    rows, x, y, _stats = vis._load_clean_rows(reference_root, dataset_name, args.band, tile_name)
    gt_xy = np.column_stack([x, y]).astype(np.float32) if len(x) else np.zeros((0, 2), dtype=np.float32)
    center_only_xy, ignore_xy = _visibility_centers(reference_root, dataset_name, args.band, tile_name)
    ignore_mask = _within_any_radius(gt_xy, ignore_xy, float(args.gt_visibility_match_radius))
    center_only_mask = _within_any_radius(gt_xy, center_only_xy, float(args.gt_visibility_match_radius)) & ~ignore_mask
    out: list[dict[str, object]] = []
    for idx in range(len(rows)):
        ap_flux, ap_mag = vis._gt_ap2_flux_mag(rows[idx], zero_point=float(args.gt_photometry_zero_point))
        kron_flux, kron_mag = vis._gt_kron_flux_mag(rows[idx], zero_point=float(args.gt_photometry_zero_point))
        visibility_class = "ignore" if bool(ignore_mask[idx]) else ("center_only" if bool(center_only_mask[idx]) else "clean")
        out.append(
            {
                "dataset": dataset_name,
                "tile": tile_name,
                "gt_index": idx,
                "gt_ap2_flux": ap_flux,
                "gt_ap2mag": ap_mag,
                "gt_kron_flux": kron_flux,
                "gt_kron_mag": kron_mag,
                "visibility_class": visibility_class,
                "visibility_keep_snr_ge2": int(_visibility_keep(visibility_class, "snr_ge2")),
                "visibility_keep_snr_ge3": int(_visibility_keep(visibility_class, "snr_ge3")),
            }
        )
    return out


def _run_tile_local(
    *,
    args: argparse.Namespace,
    cfg: dict,
    model: torch.nn.Module,
    checkpoint: Path,
    label: str,
    dataset_name: str,
    tile_name: str,
    device: torch.device,
    include_gt_rows: bool,
) -> tuple[dict, list[dict], list[dict]]:
    vis.TRACT = args.tract
    vis.PATCH = args.patch
    vis.TILE = tile_name
    print(f"[run] {label} {dataset_name} {args.patch}/{tile_name} {args.band}", flush=True)
    previous_tile_name = getattr(args, "tile_name", None)
    args.tile_name = tile_name
    try:
        row = vis._run_one(
            model=model,
            cfg=cfg,
            dataset_root=args.data_root,
            dataset_name=dataset_name,
            checkpoint_label=label,
            out_dir=args.out_dir,
            bands=args.bands,
            band=args.band,
            device=device,
            args=args,
        )
    finally:
        args.tile_name = previous_tile_name
    embedded_gt_rows = row.pop("_gt_metric_rows", None)
    row["checkpoint"] = str(checkpoint)
    row["checkpoint_epoch"] = vis._checkpoint_epoch(checkpoint)
    tile_gt_rows = embedded_gt_rows if embedded_gt_rows is not None else _gt_mag_rows(args, dataset_name, tile_name)
    if embedded_gt_rows is not None:
        gt_mag_col = str(getattr(args, "curve_gt_mag_col", "gt_ap2mag"))
        has_finite_gt_mag = any(math.isfinite(_float(row.get(gt_mag_col))) for row in tile_gt_rows)
        if tile_gt_rows and not has_finite_gt_mag:
            tile_gt_rows = _gt_mag_rows(args, dataset_name, tile_name)
    raw_gt_count = len(tile_gt_rows)
    filtered_gt_count = sum(
        1
        for gt_row in tile_gt_rows
        if _visibility_keep(str(gt_row.get("visibility_class", "clean")), str(args.gt_visibility_filter))
    )
    kept_gt_indices = {
        str(gt_row.get("gt_index", ""))
        for gt_row in tile_gt_rows
        if _visibility_keep(str(gt_row.get("visibility_class", "clean")), str(args.gt_visibility_filter))
    }
    phot_path = Path(str(row.get("photometry_csv", "")))
    tile_phot_rows = _read_csv(phot_path)
    matched_kept = {
        str(phot.get("gt_index", "")).strip()
        for phot in tile_phot_rows
        if str(phot.get("gt_index", "")).strip() in kept_gt_indices
    }
    row["raw_gt"] = int(raw_gt_count)
    row["filtered_gt"] = int(filtered_gt_count)
    row["gt_visibility_filter"] = str(args.gt_visibility_filter)
    row["clean_gt"] = int(filtered_gt_count)
    row["clean_tp"] = int(len(matched_kept))
    row["clean_fn"] = max(0, int(filtered_gt_count) - int(len(matched_kept)))

    for phot in tile_phot_rows:
        phot["checkpoint_label"] = label
        phot["checkpoint"] = str(checkpoint)
    return row, tile_phot_rows, (tile_gt_rows if include_gt_rows else [])


def _tile_chunks(tiles: Sequence[str], workers: int) -> list[list[str]]:
    worker_count = max(1, min(int(workers), len(tiles)))
    chunks: list[list[str]] = [[] for _ in range(worker_count)]
    for idx, tile in enumerate(tiles):
        chunks[idx % worker_count].append(str(tile))
    return [chunk for chunk in chunks if chunk]


def _worker_device(args: argparse.Namespace, worker_index: int) -> str:
    devices = getattr(args, "tile_worker_devices", None)
    if devices:
        return str(devices[int(worker_index) % len(devices)])
    return str(args.device)


def _run_tile_batch_worker(payload: dict) -> tuple[list[dict], list[dict], list[dict]]:
    args: argparse.Namespace = payload["args"]
    cfg: dict = payload["cfg"]
    checkpoint = Path(payload["checkpoint"]).expanduser().resolve()
    label = str(payload["label"])
    dataset_name = str(payload["dataset_name"])
    tiles = [str(tile) for tile in payload["tiles"]]
    include_gt_rows = bool(payload["include_gt_rows"])
    worker_index = int(payload["worker_index"])
    args.device = _worker_device(args, worker_index)
    device = torch.device(args.device)
    _install_visualizer_io_patch()
    if args.skip_visual_products:
        _disable_visual_products()
    model = vis._make_model(cfg, checkpoint, device, args.bands)
    summaries: list[dict] = []
    phot_rows: list[dict] = []
    gt_rows: list[dict] = []
    try:
        for tile_name in tiles:
            row, tile_phot, tile_gt = _run_tile_local(
                args=args,
                cfg=cfg,
                model=model,
                checkpoint=checkpoint,
                label=label,
                dataset_name=dataset_name,
                tile_name=tile_name,
                device=device,
                include_gt_rows=include_gt_rows,
            )
            summaries.append(row)
            phot_rows.extend(tile_phot)
            gt_rows.extend(tile_gt)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return summaries, phot_rows, gt_rows


def _run_inference(args: argparse.Namespace, cfg: dict) -> tuple[list[dict], list[dict], list[dict]]:
    if args.skip_visual_products:
        _disable_visual_products()
    ckpt_items = _checkpoint_items(args)
    if not ckpt_items:
        raise FileNotFoundError("No checkpoints found. Pass --checkpoint or provide best.pt/last.pt in --ckpt-dir.")

    summaries: list[dict] = []
    phot_rows: list[dict] = []
    gt_rows: list[dict] = []
    tile_workers = max(1, int(getattr(args, "tile_workers", 1)))
    for checkpoint, label in ckpt_items:
        checkpoint = checkpoint.expanduser().resolve()
        if tile_workers > 1 and not getattr(args, "tile_worker_devices", None):
            print(
                f"[parallel] --tile-workers={tile_workers} will run all worker model copies on {args.device}; "
                "use --tile-worker-devices to spread work across multiple GPUs.",
                flush=True,
            )
        if tile_workers <= 1:
            device = torch.device(args.device)
            model = vis._make_model(cfg, checkpoint, device, args.bands)
            try:
                for dataset_name in args.datasets:
                    tiles = _discover_tiles(args, dataset_name)
                    if not tiles:
                        print(f"[warn] no tiles for {dataset_name} patch={args.patch} group={args.group}", flush=True)
                        continue
                    for tile_name in tiles:
                        row, tile_phot, tile_gt = _run_tile_local(
                            args=args,
                            cfg=cfg,
                            model=model,
                            checkpoint=checkpoint,
                            label=label,
                            dataset_name=dataset_name,
                            tile_name=tile_name,
                            device=device,
                            include_gt_rows=label == ckpt_items[0][1],
                        )
                        summaries.append(row)
                        phot_rows.extend(tile_phot)
                        gt_rows.extend(tile_gt)
            finally:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            continue

        for dataset_name in args.datasets:
            tiles = _discover_tiles(args, dataset_name)
            if not tiles:
                print(f"[warn] no tiles for {dataset_name} patch={args.patch} group={args.group}", flush=True)
                continue
            chunks = _tile_chunks(tiles, tile_workers)
            print(
                f"[parallel] {label} {dataset_name}: {len(tiles)} tile(s), {len(chunks)} worker chunk(s)",
                flush=True,
            )
            payloads = [
                {
                    "args": args,
                    "cfg": cfg,
                    "checkpoint": str(checkpoint),
                    "label": label,
                    "dataset_name": dataset_name,
                    "tiles": chunk,
                    "include_gt_rows": label == ckpt_items[0][1],
                    "worker_index": idx,
                }
                for idx, chunk in enumerate(chunks)
            ]
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=len(payloads),
                mp_context=mp.get_context("spawn"),
            ) as executor:
                futures = [executor.submit(_run_tile_batch_worker, payload) for payload in payloads]
                for future in concurrent.futures.as_completed(futures):
                    worker_summaries, worker_phot, worker_gt = future.result()
                    summaries.extend(worker_summaries)
                    phot_rows.extend(worker_phot)
                    gt_rows.extend(worker_gt)
    return summaries, phot_rows, gt_rows


def _requested_checkpoint_map(args: argparse.Namespace) -> dict[str, Path]:
    return {label: path for path, label in _checkpoint_items(args)}


def _photometry_csvs_for_pair(args: argparse.Namespace, label: str, dataset: str) -> list[Path]:
    pair_dir = args.out_dir / str(label) / str(dataset)
    if not pair_dir.exists():
        return []
    paths = []
    for path in sorted(pair_dir.glob("*_photometry.csv")):
        if path.name.endswith("_tp_isolated_gt_photometry.csv"):
            continue
        paths.append(path)
    return paths


def _infer_tile_from_photometry_path(path: Path, label: str, dataset: str, args: argparse.Namespace) -> str:
    stem = path.name.removesuffix("_photometry.csv")
    patch_token = str(args.patch).replace(",", "_")
    band_token = str(args.band).replace("-", "_")
    prefix = f"{label}_{dataset}_{patch_token}_"
    suffix = f"_{band_token}"
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


def _summary_from_photometry_rows(
    args: argparse.Namespace,
    *,
    label: str,
    checkpoint: Path | None,
    dataset: str,
    tile: str,
    phot_path: Path,
    rows: Sequence[dict],
    gt_count: int,
    kept_gt_indices: set[str] | None = None,
) -> dict:
    matched = {
        str(row.get("gt_index", "")).strip()
        for row in rows
        if str(row.get("gt_index", "")).strip() not in {"", "nan", "None"}
    }
    if kept_gt_indices is not None:
        matched = {gt_index for gt_index in matched if gt_index in kept_gt_indices}
    kept_masks = sum(1 for row in rows if _is_true(row.get("mask_kept", False)))
    mask_areas = np.asarray([_float(row.get("mask_area")) for row in rows if _is_true(row.get("mask_kept", False))], dtype=float)
    mask_ious = np.asarray([_float(row.get("mask_pred_iou")) for row in rows if _is_true(row.get("mask_kept", False))], dtype=float)
    return {
        "checkpoint_label": label,
        "dataset": dataset,
        "tile": tile,
        "band": args.band,
        "detections": len(rows),
        "mask_prompts": len(rows),
        "raw_masks": len(rows),
        "kept_masks": kept_masks,
        "mask_area_median": float(np.nanmedian(mask_areas)) if mask_areas.size else math.nan,
        "mask_iou_median": float(np.nanmedian(mask_ious)) if mask_ious.size else math.nan,
        "gt_visibility_filter": str(args.gt_visibility_filter),
        "clean_gt": int(gt_count),
        "clean_tp": len(matched),
        "clean_fn": max(0, int(gt_count) - len(matched)),
        "photometry_csv": str(phot_path),
        "photometry_count": len(rows),
        "checkpoint": str(checkpoint) if checkpoint is not None else "",
        "checkpoint_epoch": vis._checkpoint_epoch(checkpoint) if checkpoint is not None and checkpoint.exists() else "",
    }


def _ensure_gt_visibility_rows(args: argparse.Namespace, gt_rows: Sequence[dict]) -> list[dict]:
    rows = [dict(row) for row in gt_rows]
    if not rows or all("visibility_class" in row for row in rows):
        return rows
    rebuilt_by_key: dict[tuple[str, str, str], dict] = {}
    for dataset, tile in sorted({(str(row.get("dataset", "")), str(row.get("tile", ""))) for row in rows}):
        if not dataset or not tile:
            continue
        try:
            for fresh in _gt_mag_rows(args, dataset, tile):
                rebuilt_by_key[(dataset, tile, str(fresh.get("gt_index", "")))] = fresh
        except Exception as exc:
            print(f"[warn] could not add GT visibility labels for {dataset}/{tile}: {exc}", flush=True)
    out: list[dict] = []
    for row in rows:
        key = (str(row.get("dataset", "")), str(row.get("tile", "")), str(row.get("gt_index", "")))
        fresh = rebuilt_by_key.get(key)
        if fresh is not None:
            merged = dict(row)
            for name in ("visibility_class", "visibility_keep_snr_ge2", "visibility_keep_snr_ge3"):
                merged[name] = fresh.get(name, merged.get(name, ""))
            out.append(merged)
        else:
            merged = dict(row)
            merged.setdefault("visibility_class", "clean")
            merged.setdefault("visibility_keep_snr_ge2", 1)
            merged.setdefault("visibility_keep_snr_ge3", 1)
            out.append(merged)
    return out


def _recalibrate_prediction_photometry_rows(args: argparse.Namespace, rows: Sequence[dict]) -> list[dict]:
    """Recompute predicted-source magnitudes from stored image fluxes.

    Older output CSVs may contain ``*_abmag`` values written with a different
    image zero point.  The raw aperture/mask fluxes are still valid, so derive
    the nJy and AB magnitude columns from the current ``--photometry-zero-point``
    before binning by predicted magnitude.
    """
    out: list[dict] = []
    zp = float(args.photometry_zero_point)
    psf_factor = float(args.photometry_psf_factor)
    specs = (
        ("ap_flux", "ap_flux_njy", "ap_abmag"),
        ("kron_flux", "kron_flux_njy", "kron_abmag"),
        ("mask_flux", "mask_flux_njy", "mask_abmag"),
    )
    for row in rows:
        merged = dict(row)
        for flux_col, njy_col, mag_col in specs:
            flux = _float(merged.get(flux_col))
            if not math.isfinite(flux):
                continue
            njy, mag = vis._flux_to_njy_mag(flux, zero_point=zp, psf_factor=psf_factor)
            merged[njy_col] = njy
            merged[mag_col] = mag
        out.append(merged)
    return out


def _rebuild_existing_from_raw_csv(args: argparse.Namespace) -> tuple[list[dict], list[dict], list[dict], set[tuple[str, str]]]:
    checkpoint_map = _requested_checkpoint_map(args)
    requested_labels = sorted(checkpoint_map) if checkpoint_map else []
    requested_datasets = [str(name) for name in args.datasets]
    summaries: list[dict] = []
    phot_rows: list[dict] = []
    gt_rows: list[dict] = []
    rebuilt_pairs: set[tuple[str, str]] = set()
    gt_seen: set[tuple[str, str]] = set()

    for label in requested_labels:
        checkpoint = checkpoint_map.get(label)
        for dataset in requested_datasets:
            paths = _photometry_csvs_for_pair(args, label, dataset)
            if not paths:
                continue
            rebuilt_pairs.add((label, dataset))
            for phot_path in paths:
                rows = _read_csv(phot_path)
                if rows:
                    tile = str(rows[0].get("tile", "")) or _infer_tile_from_photometry_path(phot_path, label, dataset, args)
                else:
                    tile = _infer_tile_from_photometry_path(phot_path, label, dataset, args)
                for row in rows:
                    row.setdefault("checkpoint_label", label)
                    row.setdefault("dataset", dataset)
                    row.setdefault("tile", tile)
                    row.setdefault("band", args.band)
                    if checkpoint is not None:
                        row.setdefault("checkpoint", str(checkpoint))
                phot_rows.extend(rows)

                gt_key = (dataset, tile)
                tile_gt_rows: list[dict] = []
                if gt_key not in gt_seen:
                    tile_gt_rows = _gt_mag_rows(args, dataset, tile)
                    gt_rows.extend(tile_gt_rows)
                    gt_seen.add(gt_key)
                else:
                    # Count from existing rows for this tile if GT rows were already collected.
                    tile_gt_rows = [row for row in gt_rows if str(row.get("dataset", "")) == dataset and str(row.get("tile", "")) == tile]

                summaries.append(
                    _summary_from_photometry_rows(
                        args,
                        label=label,
                        checkpoint=checkpoint,
                        dataset=dataset,
                        tile=tile,
                        phot_path=phot_path,
                        rows=rows,
                        gt_count=sum(
                            1
                            for gt_row in tile_gt_rows
                            if _visibility_keep(str(gt_row.get("visibility_class", "clean")), str(args.gt_visibility_filter))
                        ),
                        kept_gt_indices={
                            str(gt_row.get("gt_index", ""))
                            for gt_row in tile_gt_rows
                            if _visibility_keep(str(gt_row.get("visibility_class", "clean")), str(args.gt_visibility_filter))
                        },
                    )
                )
    return summaries, phot_rows, gt_rows, rebuilt_pairs


def _merge_rebuilt_rows(
    existing_summaries: Sequence[dict],
    existing_phot: Sequence[dict],
    existing_gt: Sequence[dict],
    rebuilt_summaries: Sequence[dict],
    rebuilt_phot: Sequence[dict],
    rebuilt_gt: Sequence[dict],
    rebuilt_pairs: set[tuple[str, str]],
) -> tuple[list[dict], list[dict], list[dict]]:
    summary_by_key = {
        (str(row.get("checkpoint_label", "")), str(row.get("dataset", "")), str(row.get("tile", ""))): dict(row)
        for row in existing_summaries
        if (str(row.get("checkpoint_label", "")), str(row.get("dataset", ""))) not in rebuilt_pairs
    }
    for row in rebuilt_summaries:
        summary_by_key[(str(row.get("checkpoint_label", "")), str(row.get("dataset", "")), str(row.get("tile", "")))] = dict(row)

    phot_rows = [
        dict(row)
        for row in existing_phot
        if (str(row.get("checkpoint_label", "")), str(row.get("dataset", ""))) not in rebuilt_pairs
    ]
    phot_rows.extend(dict(row) for row in rebuilt_phot)

    rebuilt_gt_keys = {(str(row.get("dataset", "")), str(row.get("tile", ""))) for row in rebuilt_gt}
    gt_rows = [
        dict(row)
        for row in existing_gt
        if (str(row.get("dataset", "")), str(row.get("tile", ""))) not in rebuilt_gt_keys
    ]
    gt_rows.extend(dict(row) for row in rebuilt_gt)
    return list(summary_by_key.values()), phot_rows, gt_rows


def _load_existing(args: argparse.Namespace) -> tuple[list[dict], list[dict], list[dict]]:
    existing_summaries = _read_csv(args.out_dir / "tile_summary.csv")
    existing_phot = _read_csv(args.out_dir / "per_source_photometry.csv")
    existing_gt = _read_csv(args.out_dir / "gt_photometry.csv")
    if not existing_summaries or not existing_phot or not existing_gt:
        print("[plot-only] summary CSVs are missing or empty; trying to rebuild from per-tile photometry CSVs", flush=True)

    requested_datasets = {str(name) for name in args.datasets}
    checkpoint_map = _requested_checkpoint_map(args)
    requested_labels = set(checkpoint_map)
    requested_pairs = {(label, dataset) for label in requested_labels for dataset in requested_datasets}

    existing_pairs = {
        (str(row.get("checkpoint_label", "")), str(row.get("dataset", "")))
        for row in existing_summaries
        if str(row.get("dataset", "")) in requested_datasets
        and (not requested_labels or str(row.get("checkpoint_label", "")) in requested_labels)
    }
    need_rebuild = bool(requested_pairs - existing_pairs) or not existing_summaries or not existing_phot or not existing_gt
    if need_rebuild:
        rebuilt_summaries, rebuilt_phot, rebuilt_gt, rebuilt_pairs = _rebuild_existing_from_raw_csv(args)
        if rebuilt_pairs:
            all_summaries, all_phot, all_gt = _merge_rebuilt_rows(
                existing_summaries,
                existing_phot,
                existing_gt,
                rebuilt_summaries,
                rebuilt_phot,
                rebuilt_gt,
                rebuilt_pairs,
            )
            _write_csv(args.out_dir / "tile_summary.csv", all_summaries)
            _write_csv(args.out_dir / "per_source_photometry.csv", all_phot)
            _write_csv(args.out_dir / "gt_photometry.csv", all_gt)
            print(
                f"[plot-only] rebuilt summary rows from raw per-tile CSVs for pairs: {sorted(rebuilt_pairs)}",
                flush=True,
            )
            existing_summaries, existing_phot, existing_gt = all_summaries, all_phot, all_gt

    def keep_common(row: dict) -> bool:
        if str(row.get("dataset", "")) not in requested_datasets:
            return False
        if requested_labels and str(row.get("checkpoint_label", "")) not in requested_labels:
            return False
        return True

    summaries = [row for row in existing_summaries if keep_common(row)]
    phot_rows = [row for row in existing_phot if keep_common(row)]
    gt_dataset_tiles = {(str(row.get("dataset", "")), str(row.get("tile", ""))) for row in summaries}
    gt_rows = [
        row for row in existing_gt
        if (str(row.get("dataset", "")), str(row.get("tile", ""))) in gt_dataset_tiles
    ]
    gt_rows = _ensure_gt_visibility_rows(args, gt_rows)

    available_pairs = sorted({(str(row.get("checkpoint_label", "")), str(row.get("dataset", ""))) for row in existing_summaries})
    missing_pairs = sorted(requested_pairs - {(str(row.get("checkpoint_label", "")), str(row.get("dataset", ""))) for row in summaries})
    if missing_pairs:
        raise ValueError(
            "plot-only requested checkpoint/dataset pairs are not present in summary CSVs or raw per-tile CSVs: "
            f"{missing_pairs}. Available pairs in {args.out_dir}: {available_pairs}. "
            "Run inference without --plot-only for the missing pairs."
        )
    if not summaries or not phot_rows or not gt_rows:
        raise FileNotFoundError("plot-only filtering left no rows to plot")
    return summaries, phot_rows, gt_rows


def _aggregate_bins(args: argparse.Namespace, phot_rows: Sequence[dict], gt_rows: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    bins = _mag_bins(args.mag_min, args.mag_max, args.bin_size)
    grouped: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for row in phot_rows:
        key = (str(row.get("checkpoint_label", "")), str(row.get("dataset", "")))
        grouped.setdefault(key, {"phot": [], "gt": []})["phot"].append(row)
    for row in gt_rows:
        for checkpoint_label in {key[0] for key in grouped if key[1] == str(row.get("dataset", ""))}:
            key = (checkpoint_label, str(row.get("dataset", "")))
            grouped.setdefault(key, {"phot": [], "gt": []})["gt"].append(row)

    detail: list[dict] = []
    aggregate_groups: dict[tuple[str, float, float], dict[str, object]] = {}
    for (label, dataset), payload in sorted(grouped.items()):
        gt_payload = payload["gt"]
        phot_payload = payload["phot"]
        gt_mag = np.asarray([_float(row.get(args.curve_gt_mag_col)) for row in gt_payload], dtype=float)
        gt_keys = [(str(row.get("tile", "")), str(row.get("gt_index", ""))) for row in gt_payload]
        gt_keep = np.asarray(
            [
                _visibility_keep(str(row.get("visibility_class", "clean")), str(args.gt_visibility_filter))
                for row in gt_payload
            ],
            dtype=bool,
        )
        kept_gt_keys = {key for key, keep in zip(gt_keys, gt_keep) if bool(keep)}
        matched_keys = {
            (str(row.get("tile", "")), str(row.get("gt_index", "")))
            for row in phot_payload
            if str(row.get("gt_index", "")).strip() != ""
        }
        gt_matched = np.asarray([key in matched_keys for key in gt_keys], dtype=bool)

        pred_mag = np.asarray([_float(row.get(args.curve_pred_mag_col)) for row in phot_payload], dtype=float)
        pred_matched = np.asarray(
            [
                (str(row.get("tile", "")), str(row.get("gt_index", ""))) in kept_gt_keys
                for row in phot_payload
            ],
            dtype=bool,
        )

        for lo, hi, center in bins:
            ref_in = _mag_in(gt_mag, lo, hi)
            ref_keep_in = ref_in & gt_keep
            pred_in = _mag_in(pred_mag, lo, hi)
            raw_ref_total = int(ref_in.sum())
            raw_ref_matched = int(np.count_nonzero(ref_in & gt_matched))
            ref_total = int(ref_keep_in.sum())
            ref_matched = int(np.count_nonzero(ref_keep_in & gt_matched))
            pred_total = int(pred_in.sum())
            pred_tp = int(np.count_nonzero(pred_in & pred_matched))
            row = {
                "checkpoint_label": label,
                "dataset": dataset,
                "method": f"{label}:{dataset}",
                "gt_visibility_filter": str(args.gt_visibility_filter),
                "mag_left": lo,
                "mag_right": hi,
                "mag_center": center,
                "raw_reference_total": raw_ref_total,
                "raw_reference_matched": raw_ref_matched,
                "reference_total": ref_total,
                "reference_matched": ref_matched,
                "completeness": ref_matched / ref_total if ref_total else math.nan,
                "prediction_total": pred_total,
                "prediction_matched": pred_tp,
                "purity": pred_tp / pred_total if pred_total else math.nan,
            }
            detail.append(row)
            agg_key = (label, lo, hi)
            agg = aggregate_groups.setdefault(
                agg_key,
                {
                    "checkpoint_label": label,
                    "dataset": "all",
                    "method": label,
                    "gt_visibility_filter": str(args.gt_visibility_filter),
                    "mag_left": lo,
                    "mag_right": hi,
                    "mag_center": center,
                    "raw_reference_total": 0,
                    "raw_reference_matched": 0,
                    "reference_total": 0,
                    "reference_matched": 0,
                    "prediction_total": 0,
                    "prediction_matched": 0,
                },
            )
            agg["raw_reference_total"] = int(agg["raw_reference_total"]) + raw_ref_total
            agg["raw_reference_matched"] = int(agg["raw_reference_matched"]) + raw_ref_matched
            agg["reference_total"] = int(agg["reference_total"]) + ref_total
            agg["reference_matched"] = int(agg["reference_matched"]) + ref_matched
            agg["prediction_total"] = int(agg["prediction_total"]) + pred_total
            agg["prediction_matched"] = int(agg["prediction_matched"]) + pred_tp

    aggregate: list[dict] = []
    for row in aggregate_groups.values():
        ref_total = int(row["reference_total"])
        pred_total = int(row["prediction_total"])
        row["completeness"] = int(row["reference_matched"]) / ref_total if ref_total else math.nan
        row["purity"] = int(row["prediction_matched"]) / pred_total if pred_total else math.nan
        aggregate.append(row)
    return detail, sorted(aggregate, key=lambda r: (str(r["method"]), float(r["mag_left"])))


def _plot_curves(path: Path, rows: Sequence[dict], *, title_suffix: str, reverse_mag_axis: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = sorted({str(row["method"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for method in methods:
        selected = [row for row in rows if str(row["method"]) == method]
        x = np.asarray([_float(row["mag_center"]) for row in selected], dtype=float)
        order = np.argsort(x)
        comp = np.asarray([_float(row["completeness"]) for row in selected], dtype=float)[order]
        pur = np.asarray([_float(row["purity"]) for row in selected], dtype=float)[order]
        x = x[order]
        axes[0].plot(x, comp, marker="o", linewidth=1.8, label=method)
        axes[1].plot(x, pur, marker="o", linewidth=1.8, label=method)
    for ax, title in zip(axes, ("Completeness", "Purity")):
        ax.set_title(f"{title} {title_suffix}".strip())
        ax.set_xlabel("AB magnitude")
        ax.set_ylabel(title.lower())
        ax.set_ylim(-0.03, 1.03)
        if reverse_mag_axis:
            ax.invert_xaxis()
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    if methods:
        axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _bin_label(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g}"


def _plot_count_bars(completeness_path: Path, purity_path: Path, rows: Sequence[dict], *, reverse_mag_axis: bool = False) -> None:
    methods = sorted({str(row["method"]) for row in rows})
    bins = sorted({(_float(row["mag_left"]), _float(row["mag_right"])) for row in rows}, reverse=reverse_mag_axis)
    if not methods or not bins:
        return
    by_key = {(str(row["method"]), _float(row["mag_left"]), _float(row["mag_right"])): row for row in rows}
    labels = [_bin_label(lo, hi) for lo, hi in bins]
    x = np.arange(len(bins), dtype=float)
    width = min(0.30, 0.78 / max(1, len(methods)))
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
    figsize = (max(11.0, 0.55 * len(bins)), 5.2)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ref = [max(int(row.get("reference_total", 0)) for row in rows if _float(row["mag_left"]) == lo and _float(row["mag_right"]) == hi) for lo, hi in bins]
    ax.bar(x, ref, width=0.86, color="0.82", edgecolor="0.55", linewidth=0.6, label="GT sources")
    for offset, method in zip(offsets, methods):
        tp = [int(by_key.get((method, lo, hi), {}).get("reference_matched", 0)) for lo, hi in bins]
        ax.bar(x + offset, tp, width=width, label=f"{method} TP")
    ax.set_title("GT sources and matched detections by GT magnitude")
    ax.set_xlabel("GT magnitude bin")
    ax.set_ylabel("source count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(completeness_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for offset, method in zip(offsets, methods):
        tp = [int(by_key.get((method, lo, hi), {}).get("prediction_matched", 0)) for lo, hi in bins]
        total = [int(by_key.get((method, lo, hi), {}).get("prediction_total", 0)) for lo, hi in bins]
        fp = [max(0, t - m) for t, m in zip(total, tp)]
        ax.bar(x + offset, tp, width=width, label=f"{method} TP")
        ax.bar(x + offset, fp, width=width, bottom=tp, label=f"{method} FP")
    ax.set_title("Predicted detections by predicted magnitude")
    ax.set_xlabel("predicted magnitude bin")
    ax.set_ylabel("source count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(purity_path, dpi=180)
    plt.close(fig)


def _plot_detection_total_bars(path: Path, rows: Sequence[dict], *, reverse_mag_axis: bool = False, aggregate: bool = False) -> None:
    methods = sorted({str(row["method"]) for row in rows})
    bins = sorted({(_float(row["mag_left"]), _float(row["mag_right"])) for row in rows}, reverse=reverse_mag_axis)
    if not methods or not bins:
        return
    by_key = {(str(row["method"]), _float(row["mag_left"]), _float(row["mag_right"])): row for row in rows}
    labels = [_bin_label(lo, hi) for lo, hi in bins]
    x = np.arange(len(bins), dtype=float)
    width = min(0.30, 0.78 / max(1, len(methods)))
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
    figsize = (max(11.0, 0.55 * len(bins)), 5.4)

    raw_gt: list[int] = []
    filtered_gt: list[int] = []
    for lo, hi in bins:
        bin_rows = [row for row in rows if _float(row["mag_left"]) == lo and _float(row["mag_right"]) == hi]
        raw_gt.append(max((int(row.get("raw_reference_total", row.get("reference_total", 0))) for row in bin_rows), default=0))
        filtered_gt.append(max((int(row.get("reference_total", 0)) for row in bin_rows), default=0))

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.bar(x, raw_gt, width=0.88, color="0.88", edgecolor="0.55", linewidth=0.6, label="raw GT")
    if not aggregate:
        ax.bar(x, filtered_gt, width=0.58, color="0.62", edgecolor="0.35", linewidth=0.6, label="filtered GT")
    for offset, method in zip(offsets, methods):
        totals = [int(by_key.get((method, lo, hi), {}).get("prediction_total", 0)) for lo, hi in bins]
        ax.bar(x + offset, totals, width=width, alpha=0.82, label=f"{method} detections")
    ax.set_title("Detections and GT source counts by magnitude")
    ax.set_xlabel("magnitude bin")
    ax.set_ylabel("source count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _finite_positive_ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or num <= 0.0 or den <= 0.0:
        return math.nan
    return num / den


def _ratio_rows(args: argparse.Namespace, phot_rows: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    detail_rows: list[dict] = []
    stats_rows: list[dict] = []
    groups = sorted({(str(row.get("checkpoint_label", "")), str(row.get("dataset", ""))) for row in phot_rows})
    for label, dataset in groups:
        selected = [
            row
            for row in phot_rows
            if str(row.get("checkpoint_label", "")) == label
            and str(row.get("dataset", "")) == dataset
            and str(row.get("gt_index", "")).strip() != ""
            and (args.ratio_source != "isolated" or _is_true(row.get("tp_isolated", "")))
        ]
        for ratio_name, num_col, den_col, label_text in RATIO_SPECS:
            values: list[float] = []
            denominator_scale = float(args.gt_flux_scale_for_ratios) if str(den_col).startswith("gt_") else 1.0
            for row in selected:
                ratio = _finite_positive_ratio(_float(row.get(num_col)), _float(row.get(den_col)) * denominator_scale)
                if math.isfinite(ratio):
                    values.append(ratio)
                    detail_rows.append(
                        {
                            "checkpoint_label": label,
                            "dataset": dataset,
                            "tile": row.get("tile", ""),
                            "pred_index": row.get("pred_index", ""),
                            "gt_index": row.get("gt_index", ""),
                            "ratio_name": ratio_name,
                            "ratio": ratio,
                            "denominator_scale": denominator_scale,
                        }
                    )
            arr = np.asarray(values, dtype=float)
            stats_rows.append(
                {
                    "checkpoint_label": label,
                    "dataset": dataset,
                    "ratio_name": ratio_name,
                    "label": label_text,
                    "denominator_scale": denominator_scale,
                    "count": int(arr.size),
                    "mean": float(np.mean(arr)) if arr.size else math.nan,
                    "median": float(np.median(arr)) if arr.size else math.nan,
                    "p16": float(np.percentile(arr, 16)) if arr.size else math.nan,
                    "p84": float(np.percentile(arr, 84)) if arr.size else math.nan,
                }
            )
    return detail_rows, stats_rows


def _plot_ratio_histograms(path: Path, ratio_detail: Sequence[dict], ratio_stats: Sequence[dict], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = sorted({(str(row["checkpoint_label"]), str(row["dataset"])) for row in ratio_detail})
    if not groups:
        return
    stats_by_key = {
        (str(row["checkpoint_label"]), str(row["dataset"]), str(row["ratio_name"])): row for row in ratio_stats
    }
    bins = np.linspace(float(args.ratio_min), float(args.ratio_max), int(args.ratio_bins) + 1)
    for label, dataset in groups:
        fig, axes = plt.subplots(len(RATIO_SPECS), 1, figsize=(9, 2.5 * len(RATIO_SPECS)), constrained_layout=True)
        if len(RATIO_SPECS) == 1:
            axes = [axes]
        for ax, (ratio_name, _num, _den, label_text) in zip(axes, RATIO_SPECS):
            values = np.asarray(
                [
                    _float(row["ratio"])
                    for row in ratio_detail
                    if str(row["checkpoint_label"]) == label
                    and str(row["dataset"]) == dataset
                    and str(row["ratio_name"]) == ratio_name
                ],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            plot_values = values[(values >= float(args.ratio_min)) & (values <= float(args.ratio_max))]
            ax.hist(plot_values, bins=bins, color="#4c78a8", alpha=0.82, edgecolor="white", linewidth=0.4)
            plot_stat = {
                "count": int(plot_values.size),
                "mean": float(np.mean(plot_values)) if plot_values.size else math.nan,
                "median": float(np.median(plot_values)) if plot_values.size else math.nan,
                "p16": float(np.percentile(plot_values, 16)) if plot_values.size else math.nan,
                "p84": float(np.percentile(plot_values, 84)) if plot_values.size else math.nan,
            }
            for key, color, style in (("mean", "#e45756", "-"), ("median", "#111111", "--"), ("p16", "#72b7b2", ":"), ("p84", "#72b7b2", ":")):
                value = _float(plot_stat.get(key))
                if math.isfinite(value):
                    ax.axvline(value, color=color, linestyle=style, linewidth=1.2)
            text = (
                f"shown N={int(plot_stat['count'])} / {int(values.size)}\n"
                f"mean={_float(plot_stat.get('mean')):.3g}\n"
                f"med={_float(plot_stat.get('median')):.3g}\n"
                f"p16/p84={_float(plot_stat.get('p16')):.3g}/{_float(plot_stat.get('p84')):.3g}"
            )
            ax.text(0.98, 0.95, text, transform=ax.transAxes, ha="right", va="top", fontsize=8, bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "0.75", "linewidth": 0.5})
            ax.set_title(label_text)
            ax.set_xlabel("flux ratio")
            ax.set_ylabel("count")
            ax.grid(axis="y", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{label} {dataset}: {args.ratio_source} TP flux ratios", fontsize=13)
        fig.savefig(path.parent / f"flux_ratio_histograms_{label}_{dataset}.png", dpi=180)
        plt.close(fig)


def main() -> int:
    args = _parse_args()
    args.ckpt_dir = args.ckpt_dir.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    if args.reference_root is not None:
        args.reference_root = args.reference_root.expanduser().resolve()
    if args.catalog_root is not None:
        args.catalog_root = args.catalog_root.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if os.environ.get("CELLECT_DEBUG_IMPORT"):
        print(
            "[debug-import] "
            f"eval_script={Path(__file__).resolve()} "
            f"visualizer={Path(getattr(vis, '__file__', '')).resolve()} "
            f"visualizer_cached={getattr(vis, '__cached__', '')}",
            flush=True,
        )
    if args.band not in args.bands:
        raise ValueError(f"--band {args.band!r} is not present in --bands")
    _install_visualizer_io_patch()

    if args.plot_only:
        summaries, phot_rows, gt_rows = _load_existing(args)
    else:
        cfg = _resolve_config(args)
        _resolve_visual_defaults(args, cfg)
        summaries, phot_rows, gt_rows = _run_inference(args, cfg)
        _write_csv(args.out_dir / "tile_summary.csv", summaries)
        _write_csv(args.out_dir / "per_source_photometry.csv", phot_rows)
        _write_csv(args.out_dir / "gt_photometry.csv", gt_rows)

    phot_rows = _recalibrate_prediction_photometry_rows(args, phot_rows)
    metric_rows, aggregate_rows = _aggregate_bins(args, phot_rows, gt_rows)
    _write_csv(args.out_dir / "magnitude_bin_metrics.csv", metric_rows)
    _write_csv(args.out_dir / "magnitude_bin_metrics_aggregate.csv", aggregate_rows)
    _plot_curves(
        args.out_dir / "magnitude_completeness_purity_curves.png",
        metric_rows,
        title_suffix=f"{args.patch} group {args.group}",
        reverse_mag_axis=bool(args.reverse_mag_axis),
    )
    _plot_curves(
        args.out_dir / "magnitude_completeness_purity_curves_aggregate.png",
        aggregate_rows,
        title_suffix=f"{args.patch} group {args.group} aggregate",
        reverse_mag_axis=bool(args.reverse_mag_axis),
    )
    _plot_count_bars(
        args.out_dir / "magnitude_completeness_counts.png",
        args.out_dir / "magnitude_purity_fp_counts.png",
        metric_rows,
        reverse_mag_axis=bool(args.reverse_mag_axis),
    )
    _plot_detection_total_bars(
        args.out_dir / "magnitude_detection_total_counts.png",
        metric_rows,
        reverse_mag_axis=bool(args.reverse_mag_axis),
    )
    _plot_count_bars(
        args.out_dir / "magnitude_completeness_counts_aggregate.png",
        args.out_dir / "magnitude_purity_fp_counts_aggregate.png",
        aggregate_rows,
        reverse_mag_axis=bool(args.reverse_mag_axis),
    )
    _plot_detection_total_bars(
        args.out_dir / "magnitude_detection_total_counts_aggregate.png",
        aggregate_rows,
        reverse_mag_axis=bool(args.reverse_mag_axis),
        aggregate=True,
    )
    
    ratio_detail, ratio_stats = _ratio_rows(args, phot_rows)
    _write_csv(args.out_dir / "flux_ratio_details.csv", ratio_detail)
    _write_csv(args.out_dir / "flux_ratio_summary.csv", ratio_stats)
    _plot_ratio_histograms(args.out_dir / "flux_ratio_histograms.png", ratio_detail, ratio_stats, args)

    manifest = {
        "data_root": str(args.data_root),
        "data_format": str(args.data_format),
        "reference_root": None if args.reference_root is None else str(args.reference_root),
        "catalog_root": None if args.catalog_root is None else str(args.catalog_root),
        "gt_classification_source": (
            "zarr_shape_source_classes" if str(args.data_format) == "zarr" and str(args.gt_visibility_filter) != "raw"
            else "reference_catalog_visibility"
        ),
        "out_dir": str(args.out_dir),
        "datasets": list(args.datasets),
        "tract": args.tract,
        "patch": args.patch,
        "group": args.group,
        "band": args.band,
        "checkpoint_count": len(_checkpoint_items(args)),
        "tile_summary_rows": len(summaries),
        "photometry_rows": len(phot_rows),
        "gt_rows": len(gt_rows),
        "metric_rows": len(metric_rows),
        "ratio_rows": len(ratio_detail),
        "ratio_source": args.ratio_source,
        "photometry_zero_point": float(args.photometry_zero_point),
        "gt_photometry_zero_point": float(args.gt_photometry_zero_point),
        "curve_pred_mag_col": str(args.curve_pred_mag_col),
        "curve_gt_mag_col": str(args.curve_gt_mag_col),
        "prediction_photometry_recomputed_from_flux": True,
        "gt_visibility_filter": str(args.gt_visibility_filter),
        "gt_visibility_match_radius": float(args.gt_visibility_match_radius),
        "tile_workers": int(args.tile_workers),
        "tile_worker_devices": list(args.tile_worker_devices or []),
        "gt_flux_scale_for_ratios": float(args.gt_flux_scale_for_ratios),
        "reverse_mag_axis": bool(args.reverse_mag_axis),
        "skip_mask_decoder": bool(args.skip_mask_decoder),
        "mask_photometry_available": not bool(args.skip_mask_decoder),
    }
    (args.out_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
