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
import csv
import json
import math
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


DEFAULT_DATA_ROOT = Path("/nvme0/zc/scarlet/preprocessed")
DEFAULT_OUT_DIR = CELLECT_ROOT / "output/sam_cellect_patch45_group1_photometry_eval"
DEFAULT_BANDS = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y")
DEFAULT_DATASETS = ("coadd", "denoised")
DEFAULT_GT_FLUX_SCALE_FOR_RATIOS = 10.0 ** (0.4 * (31.4 - 27.0))
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
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patch", default="4,5")
    parser.add_argument("--group", default="01", help="Cutout group to run, e.g. 01. Use 'all' to disable group filtering.")
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    parser.add_argument("--band", default="HSC-I")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-tiles", type=int, default=None, help="Optional debug cap after filtering.")

    parser.add_argument("--match-radius", type=float, default=vis.MATCH_RADIUS_PIX)
    parser.add_argument("--mag-min", type=float, default=22.0)
    parser.add_argument("--mag-max", type=float, default=30.0)
    parser.add_argument("--bin-size", type=float, default=0.5)
    parser.add_argument("--curve-gt-mag-col", default="gt_ap2mag")
    parser.add_argument("--curve-pred-mag-col", default="ap_abmag")
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
    parser.add_argument("--photometry-zero-point", type=float, default=31.4)
    parser.add_argument("--gt-photometry-zero-point", type=float, default=27.0)
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
        root / dataset_name / vis.TRACT / vis.PATCH / "band_reference_catalogs" / band / f"meas-{band}-{vis.TRACT}-{vis.PATCH}.fits",
        root / vis.TRACT / vis.PATCH / "band_reference_catalogs" / band / f"meas-{band}-{vis.TRACT}-{vis.PATCH}.fits",
        root / dataset_name / vis.TRACT / vis.PATCH / "reference_catalogs" / f"{base_tile}_meas.fits",
        root / vis.TRACT / vis.PATCH / "reference_catalogs" / f"{base_tile}_meas.fits",
        root / dataset_name / vis.TRACT / vis.PATCH / "reference_catalogs_csv" / f"{base_tile}_meas.csv",
        root / vis.TRACT / vis.PATCH / "reference_catalogs_csv" / f"{base_tile}_meas.csv",
    ]


def _load_clean_rows_compat(root: Path, dataset_name: str, band: str, tile_name: str):
    for path in _clean_catalog_candidates(root, dataset_name, band, tile_name):
        if path.exists():
            rows, x, y = vis._local_rows(_table(path), tile_name)
            return rows, x, y
    searched = "\n  ".join(str(path) for path in _clean_catalog_candidates(root, dataset_name, band, tile_name))
    raise FileNotFoundError(f"No GT catalog found for {dataset_name}/{vis.PATCH}/{tile_name}/{band}. Searched:\n  {searched}")


def _dataset_compat(root: Path, dataset_name: str, bands: Sequence[str], cfg: dict):
    tile_dirs = [
        root / dataset_name / vis.TRACT / vis.PATCH / "cutouts" / vis.TILE,
        root / vis.TRACT / vis.PATCH / "cutouts" / vis.TILE,
    ]
    tile_dir = next((path for path in tile_dirs if path.exists()), None)
    if tile_dir is None:
        return vis._ORIGINAL_DATASET(root, dataset_name, bands, cfg)
    image_paths: list[str] = []
    for band in bands:
        band_dir = tile_dir / band
        matches = sorted(band_dir.glob("*.fits"))
        if not matches:
            searched = ", ".join(str(path / band) for path in tile_dirs)
            raise FileNotFoundError(f"No FITS image for {dataset_name}/{vis.PATCH}/{vis.TILE}/{band}; searched {searched}")
        image_paths.append(str(matches[0]))
    image_np = read_fits_bands(tuple(image_paths), hdu=int(cfg.get("fits_hdu", 1)))
    image = astro_zscale_preprocess(image_np).to(dtype=torch.float32)
    return [{"image": image.unsqueeze(0), "image_paths": tuple(image_paths)}]


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
    rows, _x, _y = vis._load_clean_rows(args.data_root, dataset_name, args.band, tile_name)
    out: list[dict[str, object]] = []
    for idx in range(len(rows)):
        ap_flux, ap_mag = vis._gt_ap2_flux_mag(rows[idx], zero_point=float(args.gt_photometry_zero_point))
        kron_flux, kron_mag = vis._gt_kron_flux_mag(rows[idx], zero_point=float(args.gt_photometry_zero_point))
        out.append(
            {
                "dataset": dataset_name,
                "tile": tile_name,
                "gt_index": idx,
                "gt_ap2_flux": ap_flux,
                "gt_ap2mag": ap_mag,
                "gt_kron_flux": kron_flux,
                "gt_kron_mag": kron_mag,
            }
        )
    return out


def _run_inference(args: argparse.Namespace, cfg: dict) -> tuple[list[dict], list[dict], list[dict]]:
    if args.skip_visual_products:
        _disable_visual_products()
    device = torch.device(args.device)
    ckpt_items = _checkpoint_items(args)
    if not ckpt_items:
        raise FileNotFoundError("No checkpoints found. Pass --checkpoint or provide best.pt/last.pt in --ckpt-dir.")

    summaries: list[dict] = []
    phot_rows: list[dict] = []
    gt_rows: list[dict] = []
    for checkpoint, label in ckpt_items:
        checkpoint = checkpoint.expanduser().resolve()
        model = vis._make_model(cfg, checkpoint, device, args.bands)
        for dataset_name in args.datasets:
            tiles = _discover_tiles(args, dataset_name)
            if not tiles:
                print(f"[warn] no tiles for {dataset_name} patch={args.patch} group={args.group}", flush=True)
                continue
            for tile_name in tiles:
                vis.TRACT = args.tract
                vis.PATCH = args.patch
                vis.TILE = tile_name
                print(f"[run] {label} {dataset_name} {args.patch}/{tile_name} {args.band}", flush=True)
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
                row["checkpoint"] = str(checkpoint)
                row["checkpoint_epoch"] = vis._checkpoint_epoch(checkpoint)
                summaries.append(row)
                phot_path = Path(str(row.get("photometry_csv", "")))
                for phot in _read_csv(phot_path):
                    phot["checkpoint_label"] = label
                    phot["checkpoint"] = str(checkpoint)
                    phot_rows.append(phot)
                if label == ckpt_items[0][1]:
                    gt_rows.extend(_gt_mag_rows(args, dataset_name, tile_name))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return summaries, phot_rows, gt_rows


def _load_existing(args: argparse.Namespace) -> tuple[list[dict], list[dict], list[dict]]:
    summaries = _read_csv(args.out_dir / "tile_summary.csv")
    phot_rows = _read_csv(args.out_dir / "per_source_photometry.csv")
    gt_rows = _read_csv(args.out_dir / "gt_photometry.csv")
    if not summaries or not phot_rows or not gt_rows:
        raise FileNotFoundError("plot-only requires tile_summary.csv, per_source_photometry.csv, and gt_photometry.csv")
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
        matched_keys = {
            (str(row.get("tile", "")), str(row.get("gt_index", "")))
            for row in phot_payload
            if str(row.get("gt_index", "")).strip() != ""
        }
        gt_matched = np.asarray([key in matched_keys for key in gt_keys], dtype=bool)

        pred_mag = np.asarray([_float(row.get(args.curve_pred_mag_col)) for row in phot_payload], dtype=float)
        pred_matched = np.asarray([str(row.get("gt_index", "")).strip() != "" for row in phot_payload], dtype=bool)

        for lo, hi, center in bins:
            ref_in = _mag_in(gt_mag, lo, hi)
            pred_in = _mag_in(pred_mag, lo, hi)
            ref_total = int(ref_in.sum())
            ref_matched = int(np.count_nonzero(ref_in & gt_matched))
            pred_total = int(pred_in.sum())
            pred_tp = int(np.count_nonzero(pred_in & pred_matched))
            row = {
                "checkpoint_label": label,
                "dataset": dataset,
                "method": f"{label}:{dataset}",
                "mag_left": lo,
                "mag_right": hi,
                "mag_center": center,
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
                    "mag_left": lo,
                    "mag_right": hi,
                    "mag_center": center,
                    "reference_total": 0,
                    "reference_matched": 0,
                    "prediction_total": 0,
                    "prediction_matched": 0,
                },
            )
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


def _plot_curves(path: Path, rows: Sequence[dict], *, title_suffix: str) -> None:
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
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    if methods:
        axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _bin_label(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g}"


def _plot_count_bars(completeness_path: Path, purity_path: Path, rows: Sequence[dict]) -> None:
    methods = sorted({str(row["method"]) for row in rows})
    bins = sorted({(_float(row["mag_left"]), _float(row["mag_right"])) for row in rows})
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
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
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

    metric_rows, aggregate_rows = _aggregate_bins(args, phot_rows, gt_rows)
    _write_csv(args.out_dir / "magnitude_bin_metrics.csv", metric_rows)
    _write_csv(args.out_dir / "magnitude_bin_metrics_aggregate.csv", aggregate_rows)
    _plot_curves(args.out_dir / "magnitude_completeness_purity_curves.png", metric_rows, title_suffix=f"{args.patch} group {args.group}")
    _plot_curves(args.out_dir / "magnitude_completeness_purity_curves_aggregate.png", aggregate_rows, title_suffix=f"{args.patch} group {args.group} aggregate")
    _plot_count_bars(args.out_dir / "magnitude_completeness_counts.png", args.out_dir / "magnitude_purity_fp_counts.png", metric_rows)
    _plot_count_bars(args.out_dir / "magnitude_completeness_counts_aggregate.png", args.out_dir / "magnitude_purity_fp_counts_aggregate.png", aggregate_rows)

    ratio_detail, ratio_stats = _ratio_rows(args, phot_rows)
    _write_csv(args.out_dir / "flux_ratio_details.csv", ratio_detail)
    _write_csv(args.out_dir / "flux_ratio_summary.csv", ratio_stats)
    _plot_ratio_histograms(args.out_dir / "flux_ratio_histograms.png", ratio_detail, ratio_stats, args)

    manifest = {
        "data_root": str(args.data_root),
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
        "gt_flux_scale_for_ratios": float(args.gt_flux_scale_for_ratios),
    }
    (args.out_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
