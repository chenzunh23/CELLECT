#!/usr/bin/env python3
"""Evaluate cached, Photutils-subtracted, and LSST-detection SAM inputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astro_cellect2d import astro_zscale_preprocess  # noqa: E402
from zangetsu_demo import visualize_sam_cellect as vis  # noqa: E402


DEFAULT_RUNS = {
    "prompt0714": Path("/data/czh23/ckpts/sam_prompt_0714"),
    "shape0712": Path("/data/czh23/ckpts/sam_shape_0712"),
    "zarr0709": Path("/data/czh23/ckpts/sam_zarr_0709"),
}


def _read_image(path: str | Path) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul["IMAGE"] if "IMAGE" in hdul else next(
            item for item in hdul if getattr(item, "data", None) is not None and item.data.ndim == 2
        )
        return np.asarray(hdu.data, dtype=np.float32).copy()


def _raw_chw(batch: dict) -> np.ndarray:
    paths = batch["image_paths"][0]
    return np.stack([_read_image(path) for path in paths], axis=0)


def _photutils_chw(
    raw: np.ndarray,
    box_size: int,
    filter_size: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    result = np.empty_like(raw)
    models = np.empty_like(raw)
    stats = []
    for idx, image in enumerate(raw):
        model = Background2D(
            image,
            box_size=(box_size, box_size),
            filter_size=(filter_size, filter_size),
            sigma_clip=SigmaClip(sigma=3.0, maxiters=10),
            bkg_estimator=MedianBackground(),
        ).background.astype(np.float32, copy=False)
        models[idx] = model
        result[idx] = image - model
        stats.append({"band": vis.DEFAULT_BANDS[idx], "median": float(np.nanmedian(model)), "std": float(np.nanstd(model))})
    return result, models, stats


def _write_photutils_visualizations(
    data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_dir: Path,
    *,
    band: str,
    box_size: int,
    filter_size: int,
) -> None:
    import matplotlib.pyplot as plt
    from astropy.visualization import ZScaleInterval

    band_idx = list(vis.DEFAULT_BANDS).index(band)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = ("coadd", "denoised", "noisy")
    fig, axes = plt.subplots(len(rows), 4, figsize=(17, 13), constrained_layout=True)
    for row_idx, dataset in enumerate(rows):
        raw, models, subtracted = data[dataset]
        image = raw[band_idx]
        model = models[band_idx]
        sub = subtracted[band_idx]
        z_raw = astro_zscale_preprocess(image[None])[0].numpy()
        z_sub = astro_zscale_preprocess(sub[None])[0].numpy()
        raw_limits = ZScaleInterval().get_limits(image[np.isfinite(image)])
        sub_limits = ZScaleInterval().get_limits(sub[np.isfinite(sub)])
        model_limits = tuple(float(value) for value in np.nanpercentile(model, [1, 99]))
        panels = (
            (image, "Original FITS", raw_limits, "gray"),
            (model, "Photutils background model", model_limits, "viridis"),
            (sub, "Background subtracted", sub_limits, "gray"),
            (z_sub - z_raw, "Model-input difference\n(zscale(subtracted) - zscale(original))", (-1.0, 1.0), "coolwarm"),
        )
        for col_idx, (array, title, limits, cmap) in enumerate(panels):
            axis = axes[row_idx, col_idx]
            image_artist = axis.imshow(
                array,
                origin="lower",
                cmap=cmap,
                vmin=limits[0],
                vmax=limits[1],
                interpolation="nearest",
            )
            axis.set_title(f"{dataset}: {title}", fontsize=11)
            axis.set_xticks([])
            axis.set_yticks([])
            if col_idx in (1, 3):
                fig.colorbar(image_artist, ax=axis, fraction=0.046, pad=0.03)

        fits.PrimaryHDU(model).writeto(out_dir / f"{dataset}_{band}_photutils_background_model.fits", overwrite=True)
        fits.PrimaryHDU(sub).writeto(out_dir / f"{dataset}_{band}_photutils_background_subtracted.fits", overwrite=True)

        single_fig, single_axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        for axis, (array, title, limits, cmap) in zip(single_axes, panels[:3]):
            image_artist = axis.imshow(
                array,
                origin="lower",
                cmap=cmap,
                vmin=limits[0],
                vmax=limits[1],
                interpolation="nearest",
            )
            axis.set_title(title)
            axis.set_xticks([])
            axis.set_yticks([])
            if title == "Photutils background model":
                single_fig.colorbar(image_artist, ax=axis, fraction=0.046, pad=0.03)
        single_fig.suptitle(f"{dataset} {band} SAM cutout | Background2D box={box_size} filter={filter_size}", fontsize=14)
        single_fig.savefig(out_dir / f"{dataset}_{band}_photutils_background_subtraction.png", dpi=180)
        plt.close(single_fig)

    fig.suptitle(
        f"SAM {band} cutout: Photutils Background2D (box={box_size}, filter={filter_size}, sigma=3)",
        fontsize=16,
    )
    fig.savefig(out_dir / f"sam_{band}_photutils_background_subtraction_comparison.png", dpi=180)
    plt.close(fig)


def _lsst_chw(root: Path, dataset: str) -> np.ndarray:
    group = "coadd" if dataset == "coadd" else "group_01"
    paths = [root / dataset / band / f"calexp-detect-{dataset}-{band}-9813-4,5-{group}.fits" for band in vis.DEFAULT_BANDS]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing LSST outputExposure products:\n" + "\n".join(missing))
    return np.stack([_read_image(path) for path in paths], axis=0)


def _visualizer_args(data_root: Path, out_dir: Path, confidence_score: str) -> argparse.Namespace:
    saved = sys.argv
    try:
        sys.argv = [saved[0], "--data-root", str(data_root), "--out-dir", str(out_dir), "--disable-photometry",
                    "--native-sam-dir", "none", "--confidence-score", confidence_score]
        args = vis.parse_args()
    finally:
        sys.argv = saved
    args.data_root = data_root
    args.out_dir = out_dir
    args.image_cache_dir = None
    args.native_sam_dir = None
    args.variant_group = "group_01"
    args.tile_name = None
    args.disable_photometry = True
    args.multimask = False
    args.mask_prompt_center_only = False
    args.pred_iou_thresh = None
    args.stability_score_thresh = None
    return args


def _load_batches(data_root: Path, cfg: dict) -> dict[str, dict]:
    batches = {}
    helper_args = argparse.Namespace(tile_name=None, variant_group="group_01")
    for dataset in ("coadd", "denoised", "noisy"):
        tile = vis._tile_for_dataset(dataset, helper_args)
        batches[dataset] = next(iter(vis._dataset(data_root, dataset, vis.DEFAULT_BANDS, cfg, None, tile_name=tile)))
    return batches


def _replace_image(batch: dict, image: torch.Tensor) -> dict:
    changed = dict(batch)
    changed["image"] = image.unsqueeze(0).to(dtype=torch.float32)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "zangetsu_demo/data/sam_x18204_y20924")
    parser.add_argument("--lsst-products-root", type=Path, default=Path("/data/czh23/diagnostics/lsst_sam_cutout_products_20260715"))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "zangetsu_demo/output/background_input_ablation_20260715")
    parser.add_argument("--epochs", type=int, nargs="+", default=[18, 30])
    parser.add_argument("--modes", nargs="+", choices=("cache", "photutils", "lsst_calexp"), default=["cache", "photutils", "lsst_calexp"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence-score", default="ordinal_expectation", choices=("cellect", "raw", "ordinal_prob", "ordinal_expectation"))
    parser.add_argument("--photutils-box-size", type=int, default=64)
    parser.add_argument("--photutils-filter-size", type=int, default=3)
    parser.add_argument(
        "--visualize-photutils-background",
        action="store_true",
        help="Write Photutils background-model, background-subtracted, and zscale-input comparison images.",
    )
    parser.add_argument(
        "--photutils-visualization-only",
        action="store_true",
        help="Write Photutils visualizations and exit without loading checkpoints or running inference.",
    )
    parser.add_argument("--photutils-visualization-band", choices=vis.DEFAULT_BANDS, default="HSC-I")
    parser.add_argument(
        "--photutils-visualization-dir",
        type=Path,
        default=None,
        help="Visualization output directory; defaults to <out-dir>/photutils_visualization.",
    )
    args = parser.parse_args()
    if args.photutils_visualization_only:
        args.visualize_photutils_background = True
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows = []
    input_stats = []
    visualization_written = False

    for run_name, ckpt_dir in DEFAULT_RUNS.items():
        cfg = vis._read_config(ckpt_dir / "run_config.json")
        batches = _load_batches(args.data_root, cfg)
        prepared: dict[tuple[str, str], torch.Tensor] = {}
        photutils_visualization_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for dataset, batch in batches.items():
            raw = _raw_chw(batch)
            prepared[(dataset, "cache")] = batch["image"][0].clone()
            phot, phot_models, stats = _photutils_chw(raw, args.photutils_box_size, args.photutils_filter_size)
            photutils_visualization_data[dataset] = (raw, phot_models, phot)
            prepared[(dataset, "photutils")] = astro_zscale_preprocess(phot)
            if "lsst_calexp" in args.modes:
                lsst = _lsst_chw(args.lsst_products_root, dataset)
                prepared[(dataset, "lsst_calexp")] = astro_zscale_preprocess(lsst)
            cache_recomputed = astro_zscale_preprocess(raw)
            for mode in ("photutils", "lsst_calexp"):
                if (dataset, mode) not in prepared:
                    continue
                delta = prepared[(dataset, mode)] - prepared[(dataset, "cache")]
                input_stats.append({"run": run_name, "dataset": dataset, "mode": mode,
                                    "zscale_mae_vs_cache": float(delta.abs().mean()),
                                    "zscale_max_abs_vs_cache": float(delta.abs().max())})
            input_stats.append({"run": run_name, "dataset": dataset, "mode": "raw_recomputed",
                                "zscale_mae_vs_cache": float((cache_recomputed - batch["image"][0]).abs().mean()),
                                "zscale_max_abs_vs_cache": float((cache_recomputed - batch["image"][0]).abs().max()),
                                "photutils_models": stats})

        if args.visualize_photutils_background and not visualization_written:
            visualization_dir = args.photutils_visualization_dir or args.out_dir / "photutils_visualization"
            _write_photutils_visualizations(
                photutils_visualization_data,
                visualization_dir,
                band=args.photutils_visualization_band,
                box_size=args.photutils_box_size,
                filter_size=args.photutils_filter_size,
            )
            visualization_written = True
            print(f"wrote Photutils visualizations to {visualization_dir}", flush=True)
        if args.photutils_visualization_only:
            return 0

        run_args = _visualizer_args(args.data_root, args.out_dir, args.confidence_score)
        loss_cfg = cfg.get("_top", {}).get("loss_weights", {})
        run_args.multimask = bool(loss_cfg.get("mask_multimask", not bool(cfg.get("disable_mask_multimask", False))))
        run_args.mask_prompt_center_only = bool(cfg.get("mask_prompt_center_only", False))
        for epoch in args.epochs:
            checkpoint = ckpt_dir / f"epoch_{epoch:04d}.pt"
            if not checkpoint.exists():
                print(f"[missing] {checkpoint}", flush=True)
                continue
            model = vis._make_model(cfg, checkpoint, device, vis.DEFAULT_BANDS)
            for mode in args.modes:
                for dataset, baseline_batch in batches.items():
                    batch = _replace_image(baseline_batch, prepared[(dataset, mode)])
                    original_dataset = vis._dataset
                    vis._dataset = lambda *_a, **_kw: [batch]
                    try:
                        label = f"{run_name}_epoch_{epoch:04d}_{mode}"
                        row = vis._run_one(model=model, cfg=cfg, dataset_root=args.data_root, dataset_name=dataset,
                                           checkpoint_label=label, out_dir=args.out_dir, bands=vis.DEFAULT_BANDS,
                                           band="HSC-I", device=device, args=run_args)
                    finally:
                        vis._dataset = original_dataset
                    row.update({"run": run_name, "epoch": epoch, "input_mode": mode, "checkpoint": str(checkpoint)})
                    rows.append(row)
                    print(f"{run_name} e{epoch} {mode} {dataset}: det={row['detections']} TP/FN/GT={row['clean_tp']}/{row['clean_fn']}/{row['clean_gt']}", flush=True)
            del model
            torch.cuda.empty_cache()

    with (args.out_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader(); writer.writerows(rows)
    (args.out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.out_dir / "input_stats.json").write_text(json.dumps(input_stats, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
