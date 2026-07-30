#!/usr/bin/env python3
"""Visualize shrunken HSC BRIGHT_OBJECT masks and unlabeled bright clusters."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.visualization import ZScaleInterval
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Ellipse, Patch
from scipy import ndimage as ndi

from data_filtering.run_prompt_sam_bright_kron_clusters import (
    Cluster,
    SourceEllipse,
    cluster_sources,
    read_refit_sources,
    scaled_ellipse_bbox,
    scaled_ellipse_mask,
)


DEFAULT_BANDS = ("HSC-I", "HSC-Y")
DEFAULT_PATCHES = ("4,5", "6,1")
LABEL_FILES = (
    "sources_pu_clean.fits",
    "sources_pu_center_only.fits",
    "sources_pu_strict_center_only.fits",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coadd-root", type=Path, default=Path("/data/shared/Subaru"))
    parser.add_argument("--preprocessed-root", type=Path, default=Path("/data/czh23/preprocessed"))
    parser.add_argument("--refit-root", type=Path, default=Path("/data/czh23/refit"))
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patches", nargs="+", default=list(DEFAULT_PATCHES))
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    parser.add_argument("--out-dir", type=Path, default=Path("output/data_filter_0727/shrunken_bright_star_masks"))
    parser.add_argument("--scales", nargs="+", type=float, default=[0.5, 1.0 / 3.0])
    parser.add_argument("--mag-max", type=float, default=22.0)
    parser.add_argument("--hsc-bright-mag-max", type=float, default=18.0)
    parser.add_argument("--zeropoint", type=float, default=27.0)
    parser.add_argument(
        "--bright-radius-unit",
        choices=["arcsec", "pixel"],
        default="arcsec",
        help="Unit of the HSC bright-star-mask radius formula. The documentation radius is treated as arcsec by default.",
    )
    parser.add_argument("--pixel-scale-arcsec", type=float, default=0.168)
    parser.add_argument("--cluster-radius", type=float, default=50.0)
    parser.add_argument("--cluster-iou-threshold", type=float, default=1.0 / 3.0)
    parser.add_argument("--label-match-radius-pix", type=float, default=3.0)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--min-component-area", type=int, default=16)
    parser.add_argument(
        "--cluster-mask-scale",
        type=float,
        default=1.0,
        help="Scale member Kron apertures when rasterizing unlabeled cluster union masks.",
    )
    parser.add_argument(
        "--include-official-bright-mask",
        action="store_true",
        help="Also show the original BRIGHT_OBJECT union mask from the calexp MASK plane.",
    )
    parser.add_argument("--radius-column", default="proxy_nan0_flux_aperture_radius")
    parser.add_argument(
        "--official-radius-columns",
        nargs="+",
        default=["catalog_KronFlux_radius_for_radius", "catalog_KronFlux_radius"],
    )
    return parser.parse_args()


def calexp_path(root: Path, tract: str, band: str, patch: str) -> Path:
    return root / str(tract) / band / patch / f"calexp-{band}-{tract}-{patch}.fits"


def refit_csv_path(root: Path, tract: str, band: str, patch: str) -> Path:
    base = root / str(tract) / band / patch / "batch_heavyfp_kron_refit"
    preferred = base / "batch_heavyfp_kron_refit.csv"
    if preferred.exists():
        return preferred
    fallback = base / "kron_refit_rows.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(preferred)


def image_mask_hdus(hdul: fits.HDUList) -> tuple[int, int]:
    if "IMAGE" in hdul and "MASK" in hdul:
        return hdul.index_of("IMAGE"), hdul.index_of("MASK")
    image_idx = -1
    for idx, hdu in enumerate(hdul):
        data = getattr(hdu, "data", None)
        if data is not None and getattr(data, "ndim", None) == 2:
            image_idx = idx
            break
    if image_idx < 0:
        raise KeyError("no 2D IMAGE HDU found")
    image_shape = hdul[image_idx].data.shape
    for idx in range(image_idx + 1, len(hdul)):
        data = getattr(hdul[idx], "data", None)
        if data is not None and getattr(data, "ndim", None) == 2 and data.shape == image_shape:
            return image_idx, idx
    raise KeyError("no 2D MASK HDU found")


def mask_bits(header: fits.Header) -> dict[str, int]:
    bits: dict[str, int] = {}
    for key, value in header.items():
        key_text = str(key).upper()
        if not key_text.startswith("MP_"):
            continue
        try:
            bits[key_text[3:]] = int(value)
        except Exception:
            continue
    return bits


def read_image_and_bright_mask(path: Path) -> tuple[np.ndarray, np.ndarray, fits.Header, tuple[float, float]]:
    with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
        image_idx, mask_idx = image_mask_hdus(hdul)
        image_hdu = hdul[image_idx]
        mask_hdu = hdul[mask_idx]
        image = np.asarray(image_hdu.data, dtype=np.float32)
        mask = np.asarray(mask_hdu.data)
        bits = mask_bits(mask_hdu.header)
        bright_bit = bits.get("BRIGHT_OBJECT")
        if bright_bit is None:
            bright = np.zeros(mask.shape, dtype=bool)
        else:
            bright = (mask.astype(np.int64, copy=False) & (1 << int(bright_bit))) != 0
        origin = (
            -float(image_hdu.header.get("LTV1", mask_hdu.header.get("LTV1", 0.0))),
            -float(image_hdu.header.get("LTV2", mask_hdu.header.get("LTV2", 0.0))),
        )
        header = image_hdu.header.copy()
    return image, bright, header, origin


def zscale_limits(image: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(image[np.isfinite(image)], dtype=np.float32)
    if finite.size == 0:
        return 0.0, 1.0
    if finite.size > 2_000_000:
        finite = finite[:: int(math.ceil(finite.size / 2_000_000))]
    try:
        vmin, vmax = ZScaleInterval().get_limits(finite)
    except Exception:
        vmin, vmax = np.nanpercentile(finite, [1, 99])
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanpercentile(finite, 1)), float(np.nanpercentile(finite, 99))
    return float(vmin), float(vmax)


def downsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return image
    h, w = image.shape
    hh = h // factor
    ww = w // factor
    trimmed = image[: hh * factor, : ww * factor]
    return np.nanmean(trimmed.reshape(hh, factor, ww, factor), axis=(1, 3))


def downsample_bool(mask: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return mask
    h, w = mask.shape
    hh = h // factor
    ww = w // factor
    trimmed = mask[: hh * factor, : ww * factor]
    return trimmed.reshape(hh, factor, ww, factor).any(axis=(1, 3))


def shrink_connected_mask(mask: np.ndarray, scale: float, min_area: int) -> np.ndarray:
    if not np.any(mask) or scale >= 0.999:
        return mask.copy()
    labels, n_label = ndi.label(mask)
    out = np.zeros(mask.shape, dtype=bool)
    objects = ndi.find_objects(labels)
    for label_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        comp = labels[slc] == label_id
        area = int(np.count_nonzero(comp))
        if area < min_area:
            continue
        ys, xs = np.nonzero(comp)
        if ys.size == 0:
            continue
        crop_h, crop_w = comp.shape
        cy = float(ys.mean())
        cx = float(xs.mean())
        small = ndi.zoom(comp.astype(np.uint8), zoom=float(scale), order=0) > 0
        small = ndi.binary_fill_holes(small)
        if small.size == 0:
            continue
        sh, sw = small.shape
        y0 = int(round(slc[0].start + cy - sh / 2.0))
        x0 = int(round(slc[1].start + cx - sw / 2.0))
        y1 = y0 + sh
        x1 = x0 + sw
        sy0 = max(0, -y0)
        sx0 = max(0, -x0)
        sy1 = sh - max(0, y1 - mask.shape[0])
        sx1 = sw - max(0, x1 - mask.shape[1])
        py0 = max(0, y0)
        px0 = max(0, x0)
        py1 = py0 + max(0, sy1 - sy0)
        px1 = px0 + max(0, sx1 - sx0)
        if py1 > py0 and px1 > px0:
            out[py0:py1, px0:px1] |= small[sy0:sy1, sx0:sx1]
    return out


def hsc_bright_star_radius(mag: float) -> float:
    """HSC bright-star-mask radius formula in documented units.

    From the HSC SSP bright-star-mask documentation:
    A0=200, B0=0.25, C0=7.0, A1=12.0, B1=0.05, C1=16.0.
    """

    return 200.0 * 10.0 ** (0.25 * (7.0 - float(mag))) + 12.0 * 10.0 ** (0.05 * (16.0 - float(mag)))


def hsc_bright_star_radius_pix(mag: float, args: argparse.Namespace) -> float:
    radius = hsc_bright_star_radius(mag)
    if args.bright_radius_unit == "arcsec":
        return radius / float(args.pixel_scale_arcsec)
    return radius


def rasterize_catalog_bright_mask(
    sources: list[SourceEllipse],
    shape: tuple[int, int],
    scale: float,
    args: argparse.Namespace,
) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    height, width = shape
    for src in sources:
        radius = hsc_bright_star_radius_pix(src.mag, args) * float(scale)
        if not math.isfinite(radius) or radius <= 0:
            continue
        x0 = max(0, int(math.floor(src.x - radius)))
        x1 = min(width - 1, int(math.ceil(src.x + radius)))
        y0 = max(0, int(math.floor(src.y - radius)))
        y1 = min(height - 1, int(math.ceil(src.y + radius)))
        if x1 < x0 or y1 < y0:
            continue
        yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        out[y0 : y1 + 1, x0 : x1 + 1] |= (xx - src.x) ** 2 + (yy - src.y) ** 2 <= radius**2
    return out


def load_labeled_sources(preprocessed_root: Path, tract: str, patch: str, origin: tuple[float, float]) -> tuple[set[int], np.ndarray]:
    base = preprocessed_root / str(tract) / patch / "sources"
    tables = []
    for name in LABEL_FILES:
        path = base / name
        if path.exists():
            tables.append(Table.read(path))
    if not tables:
        return set(), np.zeros((0, 2), dtype=np.float32)
    table = vstack(tables, metadata_conflicts="silent") if len(tables) > 1 else tables[0]
    id_col = "id" if "id" in table.colnames else "source_id"
    ids = set(int(v) for v in table[id_col])
    x_col = "base_SdssCentroid_x" if "base_SdssCentroid_x" in table.colnames else "x"
    y_col = "base_SdssCentroid_y" if "base_SdssCentroid_y" in table.colnames else "y"
    centers = np.column_stack(
        [
            np.asarray(table[x_col], dtype=np.float64) - float(origin[0]),
            np.asarray(table[y_col], dtype=np.float64) - float(origin[1]),
        ]
    )
    finite = np.isfinite(centers).all(axis=1)
    return ids, centers[finite].astype(np.float32)


def cluster_is_labeled(cluster: Cluster, labeled_ids: set[int], labeled_centers: np.ndarray, radius_pix: float) -> bool:
    for member in cluster.members:
        try:
            if int(float(member.source_id)) in labeled_ids:
                return True
        except Exception:
            pass
    if labeled_centers.size == 0:
        return False
    points = np.asarray([[m.x, m.y] for m in cluster.members], dtype=np.float32)
    # Bright clusters are few; a dense distance check is clearer than pulling in a tree.
    d2 = (
        (points[:, None, 0] - labeled_centers[None, :, 0]) ** 2
        + (points[:, None, 1] - labeled_centers[None, :, 1]) ** 2
    )
    return bool(np.any(d2 <= float(radius_pix) ** 2))


def load_bright_clusters(
    csv_path: Path,
    args: argparse.Namespace,
    image_shape: tuple[int, int],
) -> list[Cluster]:
    read_args = argparse.Namespace(
        radius_column=args.radius_column,
        official_radius_columns=args.official_radius_columns,
        mag_min=-np.inf,
        mag_max=args.mag_max,
        zeropoint=args.zeropoint,
    )
    sources = read_refit_sources(csv_path, read_args)
    return cluster_sources(
        sources,
        radius=args.cluster_radius,
        iou_threshold=args.cluster_iou_threshold,
        image_shape=image_shape,
        padding=0,
        min_crop_size=1,
        max_crop_size=max(image_shape),
        prompt_kron_scale=1.0,
    )


def load_hsc_bright_sources(
    csv_path: Path,
    args: argparse.Namespace,
    mag_max: float,
) -> list[SourceEllipse]:
    read_args = argparse.Namespace(
        radius_column=args.radius_column,
        official_radius_columns=args.official_radius_columns,
        mag_min=-np.inf,
        mag_max=mag_max,
        zeropoint=args.zeropoint,
    )
    return read_refit_sources(csv_path, read_args)


def write_cluster_csv(path: Path, clusters: list[Cluster], bright_mask_by_scale: dict[float, np.ndarray]) -> None:
    fieldnames = [
        "cluster_id",
        "n_members",
        "x",
        "y",
        "member_source_ids",
        "member_mags",
        "member_areas",
    ] + [f"center_in_bright_scale_{scale:g}" for scale in bright_mask_by_scale]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cluster in clusters:
            row = {
                "cluster_id": cluster.cluster_id,
                "n_members": len(cluster.members),
                "x": cluster.x,
                "y": cluster.y,
                "member_source_ids": ";".join(m.source_id for m in cluster.members),
                "member_mags": ";".join(f"{m.mag:.4f}" for m in cluster.members),
                "member_areas": ";".join(f"{m.area:.2f}" for m in cluster.members),
            }
            for scale, mask in bright_mask_by_scale.items():
                x = int(round(cluster.x))
                y = int(round(cluster.y))
                row[f"center_in_bright_scale_{scale:g}"] = int(0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x])
            writer.writerow(row)


def write_cluster_reg(path: Path, clusters: list[Cluster]) -> None:
    with path.open("w") as handle:
        handle.write("# Region file format: DS9 version 4.1\n")
        handle.write("physical\n")
        for cluster in clusters:
            handle.write(f"circle({cluster.x + 1:.3f},{cluster.y + 1:.3f},7) # color=orange width=2 text={{n={len(cluster.members)}}}\n")
            for member in sorted(cluster.members, key=lambda m: m.area, reverse=True):
                angle = math.degrees(member.theta)
                handle.write(
                    f"ellipse({member.x + 1:.3f},{member.y + 1:.3f},{member.major:.3f},{member.minor:.3f},{angle:.3f}) "
                    "# color=yellow width=1\n"
                )


def rasterize_cluster_union_mask(
    clusters: list[Cluster],
    shape: tuple[int, int],
    scale: float,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    height, width = shape
    for cluster in clusters:
        for member in cluster.members:
            x0, y0, x1, y1 = scaled_ellipse_bbox(member, scale=scale)
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(width - 1, x1)
            y1 = min(height - 1, y1)
            if x1 < x0 or y1 < y0:
                continue
            mask[y0 : y1 + 1, x0 : x1 + 1] |= scaled_ellipse_mask(member, (x0, y0, x1, y1), scale=scale)
    return mask


def draw_clusters(ax, clusters: list[Cluster], downsample: int, color: str, label: str, max_ellipses: int = 400) -> None:
    for cluster in clusters:
        ax.plot(cluster.x / downsample, cluster.y / downsample, "+", color=color, markersize=4, mew=0.9)
    drawn = 0
    for cluster in sorted(clusters, key=lambda c: max(m.area for m in c.members), reverse=True):
        for member in sorted(cluster.members, key=lambda m: m.area, reverse=True):
            if drawn >= max_ellipses:
                return
            ax.add_patch(
                Ellipse(
                    (member.x / downsample, member.y / downsample),
                    width=2 * member.major / downsample,
                    height=2 * member.minor / downsample,
                    angle=math.degrees(member.theta),
                    fill=False,
                    color=color,
                    linewidth=0.6,
                    alpha=0.8,
                )
            )
            drawn += 1


def plot_panel(
    path: Path,
    image: np.ndarray,
    official_bright: np.ndarray | None,
    catalog_masks: dict[float, np.ndarray],
    hsc_bright_sources: list[SourceEllipse],
    unlabeled_clusters: list[Cluster],
    unlabeled_cluster_mask: np.ndarray,
    title: str,
    downsample: int,
    args: argparse.Namespace,
) -> None:
    image_ds = downsample_image(image, downsample)
    vmin, vmax = zscale_limits(image_ds)
    panels: list[tuple[str, np.ndarray, float | None]] = []
    if official_bright is not None:
        panels.append(("official BRIGHT_OBJECT union", official_bright, None))
    panels.extend((f"HSC mag<18 formula {scale:g}x", mask, scale) for scale, mask in catalog_masks.items())
    n_panel = len(panels)
    fig, axes = plt.subplots(1, n_panel, figsize=(6.2 * n_panel, 6.2), constrained_layout=True)
    if n_panel == 1:
        axes = [axes]
    for ax, (mask_title, mask, scale) in zip(axes, panels):
        ax.imshow(image_ds, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        mask_ds = downsample_bool(mask, downsample)
        ax.imshow(np.ma.masked_where(~mask_ds, mask_ds), origin="lower", cmap=ListedColormap(["red"]), alpha=0.55)
        if np.any(mask_ds):
            ax.contour(mask_ds.astype(np.uint8), levels=[0.5], colors=["red"], linewidths=0.8)
        cluster_ds = downsample_bool(unlabeled_cluster_mask, downsample)
        ax.imshow(np.ma.masked_where(~cluster_ds, cluster_ds), origin="lower", cmap=ListedColormap(["orange"]), alpha=0.36)
        if np.any(cluster_ds):
            ax.contour(cluster_ds.astype(np.uint8), levels=[0.5], colors=["orange"], linewidths=0.7)
        for src in hsc_bright_sources:
            ax.plot(src.x / downsample, src.y / downsample, marker="x", color="cyan", markersize=3, mew=0.7)
            if scale is not None:
                radius = hsc_bright_star_radius_pix(src.mag, args) * float(scale) / float(downsample)
                ax.add_patch(
                    Circle(
                        (src.x / downsample, src.y / downsample),
                        radius=radius,
                        fill=False,
                        edgecolor="red",
                        linewidth=0.8,
                        alpha=0.9,
                    )
                )
        draw_clusters(ax, unlabeled_clusters, downsample, color="orange", label="unlabeled")
        ax.set_title(f"{title}\n{mask_title}; unlabeled bright clusters={len(unlabeled_clusters)}")
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [
        Patch(facecolor="red", alpha=0.36, label="HSC mag<18 bright-star formula mask"),
        Patch(facecolor="cyan", alpha=0.9, label="HSC mag<18 center"),
        Patch(facecolor="orange", alpha=0.9, label="unlabeled mag<=22 cluster"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def process_one(args: argparse.Namespace, patch: str, band: str) -> dict[str, object]:
    image_path = calexp_path(args.coadd_root, args.tract, band, patch)
    refit_path = refit_csv_path(args.refit_root, args.tract, band, patch)
    image, bright, header, origin = read_image_and_bright_mask(image_path)
    hsc_bright_sources = load_hsc_bright_sources(refit_path, args, args.hsc_bright_mag_max)
    catalog_masks = {
        float(scale): rasterize_catalog_bright_mask(hsc_bright_sources, image.shape, float(scale), args)
        for scale in args.scales
    }
    labeled_ids, labeled_centers = load_labeled_sources(args.preprocessed_root, args.tract, patch, origin)
    clusters = load_bright_clusters(refit_path, args, image.shape)
    unlabeled = [
        cluster
        for cluster in clusters
        if not cluster_is_labeled(cluster, labeled_ids, labeled_centers, args.label_match_radius_pix)
    ]
    unlabeled_cluster_mask = rasterize_cluster_union_mask(unlabeled, image.shape, float(args.cluster_mask_scale))
    out_dir = args.out_dir / str(args.tract) / patch / band
    out_dir.mkdir(parents=True, exist_ok=True)
    for scale, mask in catalog_masks.items():
        scale_text = str(scale).replace(".", "p")
        out_header = header.copy()
        out_header["BSCALE"] = float(scale)
        out_header["BMAGMAX"] = float(args.hsc_bright_mag_max)
        fits.writeto(
            out_dir / f"{band}_{args.tract}_{patch.replace(',', '_')}_hsc_mag_lt_{args.hsc_bright_mag_max:g}_bright_formula_scaled_{scale_text}.fits",
            mask.astype(np.uint8),
            out_header,
            overwrite=True,
        )
    write_cluster_csv(out_dir / f"{band}_{args.tract}_{patch.replace(',', '_')}_unlabeled_mag_le_{args.mag_max:g}_clusters.csv", unlabeled, catalog_masks)
    write_cluster_reg(out_dir / f"{band}_{args.tract}_{patch.replace(',', '_')}_unlabeled_mag_le_{args.mag_max:g}_clusters.reg", unlabeled)
    plot_panel(
        out_dir / f"{band}_{args.tract}_{patch.replace(',', '_')}_hsc_bright_formula_mask_unlabeled_clusters.png",
        image,
        bright if args.include_official_bright_mask else None,
        catalog_masks,
        hsc_bright_sources,
        unlabeled,
        unlabeled_cluster_mask,
        title=f"{band} {args.tract}/{patch}",
        downsample=args.downsample,
        args=args,
    )
    out_header = header.copy()
    out_header["CMSCALE"] = float(args.cluster_mask_scale)
    fits.writeto(
        out_dir / f"{band}_{args.tract}_{patch.replace(',', '_')}_unlabeled_mag_le_{args.mag_max:g}_cluster_union_mask.fits",
        unlabeled_cluster_mask.astype(np.uint8),
        out_header,
        overwrite=True,
    )
    return {
        "patch": patch,
        "band": band,
        "bright_pixels": int(np.count_nonzero(bright)),
        "hsc_bright_sources_mag_lt": len(hsc_bright_sources),
        "clusters_mag_le": len(clusters),
        "unlabeled_clusters": len(unlabeled),
        "unlabeled_cluster_mask_pixels": int(np.count_nonzero(unlabeled_cluster_mask)),
        "labeled_source_ids": len(labeled_ids),
        **{f"hsc_bright_pixels_scale_{scale:g}": int(np.count_nonzero(mask)) for scale, mask in catalog_masks.items()},
    }


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for patch in args.patches:
        for band in args.bands:
            row = process_one(args, patch, band)
            rows.append(row)
            print(row, flush=True)
    summary = args.out_dir / f"{args.tract}_shrunken_bright_mask_unlabeled_summary.csv"
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["patch", "band"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
