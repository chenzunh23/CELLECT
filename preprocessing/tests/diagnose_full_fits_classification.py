#!/usr/bin/env python3
"""Run preprocessing-v3 classification on full FITS images and export diagnostics.

This script does not write zarr.  It runs the same catalog/filter/bright/SNR
classification path used by ``preprocessing/build_image_level_zarr.py`` and
writes a full-patch PNG plus one DS9 REG file per final source class.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cellect")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ZScaleInterval
from matplotlib.patches import Ellipse, Patch

from astro_data_preprocessing import _band_fits_path
from preprocessing.build_image_level_zarr import StoreTask, _classify_patch, _read_image_header_origin
from preprocessing.labels import SourceClass
from preprocessing.refit import RefitConfig, compute_kron_ellipse


CLASS_COLORS = {
    SourceClass.CLEAN: "#00d15c",
    SourceClass.WEAK_SHAPE: "#009eff",
    SourceClass.STRICT_CENTER_ONLY: "#c738ff",
    SourceClass.RESTRICTED_BRIGHT_REGION: "#ffb000",
    SourceClass.ORDINARY_IGNORE: "#ff3b30",
    SourceClass.STRICT_IGNORE: "#7a52ff",
    SourceClass.DROPPED: "#777777",
}

REG_COLORS = {
    SourceClass.CLEAN: "green",
    SourceClass.WEAK_SHAPE: "blue",
    SourceClass.STRICT_CENTER_ONLY: "magenta",
    SourceClass.RESTRICTED_BRIGHT_REGION: "yellow",
    SourceClass.ORDINARY_IGNORE: "red",
    SourceClass.STRICT_IGNORE: "magenta",
    SourceClass.DROPPED: "gray",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("/data/shared/Subaru"))
    p.add_argument("--refit-root", type=Path, default=Path("/data/czh23/refit"))
    p.add_argument("--denoised-fits-root", type=Path, default=Path("/data/czh23/denoised_fits"))
    p.add_argument("--variant-lsst-background-root", type=Path, default=Path("/data/czh23/lsst_background_masks"))
    p.add_argument("--gaia-fits", type=Path, default=Path("output/gaia_dr3_cosmos.fits"))
    p.add_argument("--out-dir", type=Path, default=Path("output/preprocessing_v3_diagnostics_0803/full_fits_classification"))
    p.add_argument("--tract", type=int, default=9813)
    p.add_argument("--patch", default="4,5")
    p.add_argument("--bands", nargs="+", default=["HSC-I", "HSC-Y", "NB1010"])
    p.add_argument("--dataset-source", default="coadd", choices=["coadd", "denoised", "noisy"])
    p.add_argument("--group", default="group_00")
    p.add_argument("--image-scaling-mode", default="zscore-log-lupton-rgb")
    p.add_argument("--image-scaling-scope", default="patch", choices=["patch", "tile"])
    p.add_argument("--bright-mask-mode", default="log-lupton")
    p.add_argument("--bright-threshold", type=float, default=2.99)
    p.add_argument("--bright-dilation", type=int, default=2)
    p.add_argument("--log-a", type=float, default=float("nan"))
    p.add_argument("--image-log-a", type=float, default=float("nan"))
    p.add_argument("--bright-log-a", type=float, default=float("nan"))
    p.add_argument("--image-log-high-percentile", type=float, default=99.5)
    p.add_argument("--bright-log-high-percentile", type=float, default=99.5)
    p.add_argument("--lupton-stretch", type=float, default=None)
    p.add_argument("--lupton-q", type=float, default=None)
    p.add_argument("--image-lupton-stretch", type=float, default=0.5)
    p.add_argument("--image-lupton-q", type=float, default=20.0)
    p.add_argument("--bright-lupton-stretch", type=float, default=0.5)
    p.add_argument("--bright-lupton-q", type=float, default=20.0)
    p.add_argument("--image-anscombe-scale", type=float, default=1000.0)
    p.add_argument("--bright-anscombe-scale", type=float, default=1000.0)
    p.add_argument("--cluster-source-match-pixels", type=float, default=6.0)
    p.add_argument("--cluster-centroid-match-pixels", type=float, default=10.0)
    p.add_argument("--gaia-bright-mag-threshold", type=float, default=18.0)
    p.add_argument("--large-area-point-only", type=float, default=10000.0)
    p.add_argument("--max-ellipses-per-class", type=int, default=20000)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--image-variant-background-source", default="auto", choices=["auto", "coadd-target", "variant-lsst", "none"])
    p.add_argument("--missing-variant-background-policy", default="fallback_coadd", choices=["fallback_coadd", "none", "error"])
    return p.parse_args()


def zscale(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.float32)
    vmin, vmax = ZScaleInterval().get_limits(finite)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = np.nanpercentile(finite, [1.0, 99.0])
    return np.clip((np.nan_to_num(values, nan=vmin) - vmin) / max(float(vmax - vmin), 1e-6), 0.0, 1.0)


def reg_header(color: str) -> list[str]:
    return [
        "# Region file format: DS9 version 4.1",
        f'global color={color} dashlist=8 3 width=2 font="helvetica 10 normal roman" '
        "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
        "image",
    ]


def ellipse_reg_line(x: float, y: float, major: float, minor: float, theta: float, *, color: str, text: str = "") -> str:
    suffix = f" # color={color}"
    if text:
        suffix += f" text={{{text}}}"
    return f"ellipse({x:.3f},{y:.3f},{major:.3f},{minor:.3f},{math.degrees(theta):.3f}){suffix}"


def point_reg_line(x: float, y: float, *, color: str, text: str = "") -> str:
    suffix = f" # point=cross 12 color={color}"
    if text:
        suffix += f" text={{{text}}}"
    return f"point({x:.3f},{y:.3f}){suffix}"


def write_reg(path: Path, lines: list[str], *, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([*reg_header(color), *lines, ""]) , encoding="utf-8")


def draw_class(ax: plt.Axes, geom, indices: np.ndarray, *, source_class: SourceClass, point_only_area: float, max_ellipses: int) -> int:
    color = CLASS_COLORS[source_class]
    valid = indices[geom.valid()[indices]]
    if valid.size > max_ellipses:
        valid = valid[:max_ellipses]
    order = valid[np.argsort(geom.area[valid])[::-1]]
    for idx in order:
        if float(geom.area[idx]) > float(point_only_area):
            ax.plot(float(geom.x[idx]), float(geom.y[idx]), marker="+", color=color, ms=3.5, mew=0.9)
            continue
        ax.add_patch(
            Ellipse(
                (float(geom.x[idx]), float(geom.y[idx])),
                width=2.0 * float(geom.major[idx]),
                height=2.0 * float(geom.minor[idx]),
                angle=math.degrees(float(geom.theta[idx])),
                edgecolor=color,
                facecolor="none",
                linewidth=0.75,
                alpha=0.9,
            )
        )
        ax.plot(float(geom.x[idx]), float(geom.y[idx]), marker="+", color=color, ms=2.0, mew=0.7)
    return int(valid.size)


def make_task(args: argparse.Namespace, band: str) -> StoreTask:
    image_log_a = float(args.log_a) if np.isfinite(float(args.log_a)) else float(args.image_log_a)
    bright_log_a = float(args.log_a) if np.isfinite(float(args.log_a)) else float(args.bright_log_a)
    image_lupton_stretch = float(args.lupton_stretch) if args.lupton_stretch is not None else float(args.image_lupton_stretch)
    bright_lupton_stretch = float(args.lupton_stretch) if args.lupton_stretch is not None else float(args.bright_lupton_stretch)
    image_lupton_q = float(args.lupton_q) if args.lupton_q is not None else float(args.image_lupton_q)
    bright_lupton_q = float(args.lupton_q) if args.lupton_q is not None else float(args.bright_lupton_q)
    return StoreTask(
        data_root=args.data_root.expanduser().resolve(),
        output_root=args.out_dir.expanduser().resolve(),
        refit_root=args.refit_root.expanduser().resolve(),
        denoised_fits_root=args.denoised_fits_root.expanduser().resolve(),
        variant_lsst_background_root=args.variant_lsst_background_root.expanduser().resolve() if args.variant_lsst_background_root else None,
        gaia_fits=args.gaia_fits.expanduser().resolve() if args.gaia_fits else None,
        tract=int(args.tract),
        patch=str(args.patch),
        band=str(band),
        dataset_source=str(args.dataset_source),
        group="" if args.dataset_source == "coadd" else str(args.group),
        tile_size=512,
        stride=368,
        max_tiles=0,
        overwrite=bool(args.overwrite),
        chunk_tiles=16,
        image_scaling_mode=str(args.image_scaling_mode),
        image_scaling_scope=str(args.image_scaling_scope),
        bright_mask_mode=str(args.bright_mask_mode),
        bright_threshold=float(args.bright_threshold),
        bright_dilation=int(args.bright_dilation),
        image_log_a=image_log_a,
        image_log_high_percentile=float(args.image_log_high_percentile),
        image_lupton_stretch=image_lupton_stretch,
        image_lupton_q=image_lupton_q,
        image_anscombe_scale=float(args.image_anscombe_scale),
        bright_log_a=bright_log_a,
        bright_log_high_percentile=float(args.bright_log_high_percentile),
        bright_lupton_stretch=bright_lupton_stretch,
        bright_lupton_q=bright_lupton_q,
        bright_anscombe_scale=float(args.bright_anscombe_scale),
        cluster_source_match_pixels=float(args.cluster_source_match_pixels),
        cluster_centroid_match_pixels=float(args.cluster_centroid_match_pixels),
        gaia_bright_mag_threshold=float(args.gaia_bright_mag_threshold),
        snr_method="auto",
        missing_noncoadd_policy="fallback_coadd",
        image_variant_background_source=str(args.image_variant_background_source),
        missing_variant_background_policy=str(args.missing_variant_background_policy),
    )


def run_one(args: argparse.Namespace, band: str) -> Path:
    task = make_task(args, band)
    image_path = _band_fits_path(task.data_root, task.band, task.tract, task.patch)
    image, header, origin = _read_image_header_origin(image_path)
    labels = _classify_patch(task, image, header, origin, image_path)
    geom = compute_kron_ellipse(labels.table, RefitConfig())
    class_ids = labels.label_classes
    out_dir = args.out_dir.expanduser().resolve() / str(args.patch) / str(band)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.tract}_{str(args.patch).replace(',', '_')}_{band}"

    display = zscale(image)
    fig, ax = plt.subplots(figsize=(11, 11), constrained_layout=True)
    ax.imshow(display, cmap="gray", origin="lower", interpolation="nearest")
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.set_xlabel("image x")
    ax.set_ylabel("image y")
    ax.set_title(f"{stem}: non-ignore final source classes")
    handles: list[Patch] = []
    for source_class in (SourceClass.CLEAN, SourceClass.WEAK_SHAPE, SourceClass.STRICT_CENTER_ONLY, SourceClass.RESTRICTED_BRIGHT_REGION):
        idx = np.flatnonzero(class_ids == int(source_class))
        count = draw_class(
            ax,
            geom,
            idx,
            source_class=source_class,
            point_only_area=float(args.large_area_point_only),
            max_ellipses=int(args.max_ellipses_per_class),
        )
        if count:
            handles.append(Patch(color=CLASS_COLORS[source_class], label=f"{source_class.name.lower()} {count}"))
    if labels.strict_x.size:
        ax.scatter(labels.strict_x, labels.strict_y, c=CLASS_COLORS[SourceClass.STRICT_CENTER_ONLY], marker="+", s=32, linewidths=1.2)
        handles.append(Patch(color=CLASS_COLORS[SourceClass.STRICT_CENTER_ONLY], label=f"synthetic strict centers {labels.strict_x.size}"))
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.75)
    png_path = out_dir / f"{stem}_non_ignore_sources.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    csv_path = out_dir / f"{stem}_source_classes.csv"
    source_id_col = "id" if "id" in labels.table.colnames else "source_id"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_id", "x", "y", "major", "minor", "theta_deg", "area", "source_class"])
        writer.writeheader()
        for idx in range(len(labels.table)):
            cls = SourceClass(int(class_ids[idx])).name.lower()
            writer.writerow(
                {
                    "source_id": int(labels.table[source_id_col][idx]) if source_id_col in labels.table.colnames else int(idx),
                    "x": float(geom.x[idx]) if np.isfinite(geom.x[idx]) else "",
                    "y": float(geom.y[idx]) if np.isfinite(geom.y[idx]) else "",
                    "major": float(geom.major[idx]) if np.isfinite(geom.major[idx]) else "",
                    "minor": float(geom.minor[idx]) if np.isfinite(geom.minor[idx]) else "",
                    "theta_deg": math.degrees(float(geom.theta[idx])) if np.isfinite(geom.theta[idx]) else "",
                    "area": float(geom.area[idx]) if np.isfinite(geom.area[idx]) else "",
                    "source_class": cls,
                }
            )
        for x, y, sid in zip(labels.strict_x, labels.strict_y, labels.strict_ids):
            writer.writerow(
                {
                    "source_id": int(sid),
                    "x": float(x),
                    "y": float(y),
                    "major": "",
                    "minor": "",
                    "theta_deg": "",
                    "area": "",
                    "source_class": "strict_center_only_synthetic",
                }
            )

    for source_class in SourceClass:
        if source_class == SourceClass.DROPPED:
            continue
        color = REG_COLORS[source_class]
        lines: list[str] = []
        for idx in np.flatnonzero(class_ids == int(source_class)):
            if not bool(geom.valid()[idx]):
                continue
            sid = int(labels.table[source_id_col][idx]) if source_id_col in labels.table.colnames else int(idx)
            if float(geom.area[idx]) > float(args.large_area_point_only):
                lines.append(point_reg_line(float(geom.x[idx]), float(geom.y[idx]), color=color, text=f"{source_class.name.lower()} sid={sid} area={float(geom.area[idx]):.1f}"))
            else:
                lines.append(
                    ellipse_reg_line(
                        float(geom.x[idx]),
                        float(geom.y[idx]),
                        float(geom.major[idx]),
                        float(geom.minor[idx]),
                        float(geom.theta[idx]),
                        color=color,
                        text=f"{source_class.name.lower()} sid={sid}",
                    )
                )
        if source_class == SourceClass.STRICT_CENTER_ONLY:
            for x, y, sid in zip(labels.strict_x, labels.strict_y, labels.strict_ids):
                lines.append(point_reg_line(float(x), float(y), color=color, text=f"synthetic_strict sid={int(sid)}"))
        write_reg(out_dir / f"{stem}_{source_class.name.lower()}.reg", lines, color=color)

    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")
    return out_dir


def main() -> int:
    args = parse_args()
    for band in args.bands:
        run_one(args, band)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
