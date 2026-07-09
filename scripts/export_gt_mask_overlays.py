#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import ZScaleInterval

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from astro_data_preprocessing import _find_image_hdu_index


COLORS = {
    "clean": (0.0, 0.78, 0.22),
    "center_only": (1.0, 0.62, 0.0),
    "ignore": (1.0, 0.05, 0.05),
}


def _read_image(path: Path) -> Tuple[np.ndarray, Tuple[int, int]]:
    with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
        hdu = hdul[_find_image_hdu_index(hdul)]
        image = np.asarray(hdu.data, dtype=np.float32)
        header = hdu.header
        if "CRVAL1A" in header and "CRVAL2A" in header:
            origin = (int(round(float(header["CRVAL1A"]))), int(round(float(header["CRVAL2A"]))))
        else:
            origin = (
                int(round(-float(header.get("LTV1", 0.0)))),
                int(round(-float(header.get("LTV2", 0.0)))),
            )
    return image, origin


def _zscale_limits(image: np.ndarray) -> Tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    try:
        lo, hi = ZScaleInterval().get_limits(finite)
    except Exception:
        lo, hi = np.nanpercentile(finite, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanpercentile(finite, [1, 99])
    return float(lo), float(hi)


def _parse_xy(stem: str) -> Tuple[int, int]:
    match = re.search(r"_x(-?\d+)_y(-?\d+)$", stem)
    if match is None:
        raise ValueError(f"Cannot parse x/y from target name: {stem}")
    return int(match.group(1)), int(match.group(2))


def _place_mask(dst: np.ndarray, tile_mask: np.ndarray, *, x0: int, y0: int, origin: Tuple[int, int]) -> None:
    local_x0 = int(x0 - origin[0])
    local_y0 = int(y0 - origin[1])
    h, w = tile_mask.shape
    src_x0 = max(0, -local_x0)
    src_y0 = max(0, -local_y0)
    src_x1 = min(w, dst.shape[1] - local_x0)
    src_y1 = min(h, dst.shape[0] - local_y0)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return
    dst[
        local_y0 + src_y0 : local_y0 + src_y1,
        local_x0 + src_x0 : local_x0 + src_x1,
    ] |= tile_mask[src_y0:src_y1, src_x0:src_x1]


def _assemble_masks(
    target_dir: Path,
    *,
    prefix: Optional[str],
    origin: Tuple[int, int],
    shape: Tuple[int, int],
) -> Tuple[Dict[str, np.ndarray], list[Path]]:
    masks = {
        "clean": np.zeros(shape, dtype=bool),
        "center_only": np.zeros(shape, dtype=bool),
        "ignore": np.zeros(shape, dtype=bool),
    }
    paths = sorted(target_dir.glob("*.npz"))
    if prefix:
        paths = [path for path in paths if path.stem.startswith(prefix)]
    for path in paths:
        x0, y0 = _parse_xy(path.stem)
        data = np.load(path)
        clean = np.asarray(data["clean_mask"], dtype=bool) if "clean_mask" in data else np.zeros((512, 512), bool)
        center = (
            np.asarray(data["center_only_mask"], dtype=bool)
            if "center_only_mask" in data
            else np.zeros_like(clean)
        )
        ignore = np.asarray(data["ignore_mask"], dtype=bool) if "ignore_mask" in data else np.zeros_like(clean)
        if "strict_center_only_mask" in data:
            center |= np.asarray(data["strict_center_only_mask"], dtype=bool)
        if "strict_ignore_mask" in data:
            center |= np.asarray(data["strict_ignore_mask"], dtype=bool)

        center &= ~clean
        ignore &= ~clean & ~center
        _place_mask(masks["clean"], clean, x0=x0, y0=y0, origin=origin)
        _place_mask(masks["center_only"], center, x0=x0, y0=y0, origin=origin)
        _place_mask(masks["ignore"], ignore, x0=x0, y0=y0, origin=origin)

    masks["center_only"] &= ~masks["clean"]
    masks["ignore"] &= ~masks["clean"] & ~masks["center_only"]
    return masks, paths


def _catalog_count(path: Path) -> int:
    return len(Table.read(path, hdu=1)) if path.exists() else 0


def _unique_clean_ids(meta_dir: Path, *, prefix: Optional[str]) -> int:
    ids: list[int] = []
    for path in sorted(meta_dir.glob("*.npz")):
        if prefix and not path.stem.startswith(prefix):
            continue
        data = np.load(path)
        if "ids" in data:
            ids.extend(int(value) for value in np.asarray(data["ids"]).ravel())
    return len(set(ids))


def _unique_visibility_centers(target_dir: Path, key: str, *, prefix: Optional[str]) -> int:
    points: set[Tuple[float, float]] = set()
    for path in sorted(target_dir.glob("*.npz")):
        if prefix and not path.stem.startswith(prefix):
            continue
        data = np.load(path)
        if key not in data:
            continue
        x0, y0 = _parse_xy(path.stem)
        centers = np.asarray(data[key], dtype=np.float64).reshape(-1, 2)
        for x, y in centers:
            points.add((round(float(x + x0), 3), round(float(y + y0), 3)))
    return len(points)


def _rgba(mask: np.ndarray, color: Tuple[float, float, float], alpha: float) -> np.ndarray:
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[mask] = (color[0], color[1], color[2], alpha)
    return rgba


def _downsample(array: np.ndarray, factor: int) -> np.ndarray:
    return array[::factor, ::factor] if array.ndim == 2 else array[::factor, ::factor, :]


def _write_overlay(
    *,
    name: str,
    image_path: Path,
    masks: Dict[str, np.ndarray],
    counts: Dict[str, int],
    target_count: int,
    output: Path,
    downsample: int,
) -> None:
    image, _origin = _read_image(image_path)
    lo, hi = _zscale_limits(image)
    image_show = _downsample(image, downsample)
    mask_show = {key: _downsample(value, downsample) for key, value in masks.items()}

    combined = np.zeros((*image_show.shape, 4), dtype=np.float32)
    for key, alpha in (("clean", 0.42), ("center_only", 0.48), ("ignore", 0.42)):
        layer = _rgba(mask_show[key], COLORS[key], alpha)
        replace = layer[..., 3] > 0
        combined[replace] = layer[replace]

    overlays = [
        None,
        combined,
        _rgba(mask_show["clean"], COLORS["clean"], 0.55),
        _rgba(mask_show["center_only"], COLORS["center_only"], 0.60),
        _rgba(mask_show["ignore"], COLORS["ignore"], 0.55),
    ]
    titles = [
        "zscale",
        "combined GT",
        f"clean\nn={counts.get('clean', 0)} px={int(masks['clean'].sum())}",
        f"center_only\nn={counts.get('center_only', 0)} px={int(masks['center_only'].sum())}",
        f"ignore\nn={counts.get('ignore', 0)} px={int(masks['ignore'].sum())}",
    ]

    fig, axes = plt.subplots(1, 5, figsize=(24, 5.2), constrained_layout=True)
    for ax, title, overlay in zip(axes, titles, overlays):
        ax.imshow(image_show, origin="lower", cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        if overlay is not None:
            ax.imshow(overlay, origin="lower", interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    axes[1].legend(
        handles=[Patch(facecolor=COLORS[key], alpha=0.65, label=key) for key in ("clean", "center_only", "ignore")],
        loc="lower left",
        fontsize=8,
        framealpha=0.8,
    )
    fig.suptitle(f"{name} | targets={target_count} | downsample={downsample}x", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export coadd/variant GT mask zscale overlays from precomputed targets.")
    parser.add_argument("--preprocessed-root", type=Path, default=Path("/nvme0/zc/scarlet/preprocessed"))
    parser.add_argument("--image-root", type=Path, default=Path("/nvme0/zc/scarlet"))
    parser.add_argument("--denoised-fits-root", type=Path, default=Path("/nvme0/zc/scarlet/denoised_fits"))
    parser.add_argument("--tract", type=int, default=9813)
    parser.add_argument("--patch", default="4,5")
    parser.add_argument("--band", default="HSC-I")
    parser.add_argument("--variant", default="denoised")
    parser.add_argument("--group", default="group_01")
    parser.add_argument("--out-dir", type=Path, default=Path("output/gt_overlays_260708"))
    parser.add_argument("--downsample", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    coadd_patch_root = args.preprocessed_root / str(args.tract) / args.patch
    variant_patch_root = args.preprocessed_root / args.variant / str(args.tract) / args.patch
    coadd_image = args.image_root / str(args.tract) / args.band / args.patch / f"calexp-{args.band}-{args.tract}-{args.patch}.fits"
    variant_image = args.denoised_fits_root / f"patch_{args.patch.replace(',', '_')}" / args.group / args.band / f"{args.variant}.fits"

    image, origin = _read_image(coadd_image)
    shape = tuple(int(v) for v in image.shape)

    coadd_masks, coadd_paths = _assemble_masks(
        coadd_patch_root / "band_targets" / args.band,
        prefix=None,
        origin=origin,
        shape=shape,
    )
    variant_masks, variant_paths = _assemble_masks(
        variant_patch_root / "band_targets" / args.band,
        prefix=f"{args.group}_",
        origin=origin,
        shape=shape,
    )

    band_filename = f"meas-{args.band}-{args.tract}-{args.patch}.fits"
    coadd_counts = {
        "clean": _catalog_count(coadd_patch_root / "band_reference_catalogs" / args.band / band_filename),
        "center_only": _catalog_count(coadd_patch_root / "band_reference_center_only" / args.band / band_filename),
        "ignore": _catalog_count(coadd_patch_root / "band_reference_ignore" / args.band / band_filename),
    }
    variant_counts = {
        "clean": _unique_clean_ids(variant_patch_root / "band_tile_metadata" / args.band, prefix=f"{args.group}_"),
        "center_only": _unique_visibility_centers(
            variant_patch_root / "band_targets" / args.band,
            "visibility_center_only_centers",
            prefix=f"{args.group}_",
        ),
        "ignore": _unique_visibility_centers(
            variant_patch_root / "band_targets" / args.band,
            "visibility_ignore_centers",
            prefix=f"{args.group}_",
        ),
    }

    coadd_output = args.out_dir / f"{args.band}_{args.tract}_{args.patch.replace(',', '_')}_coadd_gt_clean_center_ignore.png"
    variant_output = (
        args.out_dir
        / f"{args.band}_{args.tract}_{args.patch.replace(',', '_')}_{args.variant}_{args.group}_gt_clean_center_ignore.png"
    )
    _write_overlay(
        name=f"coadd GT {args.band} {args.tract}/{args.patch}",
        image_path=coadd_image,
        masks=coadd_masks,
        counts=coadd_counts,
        target_count=len(coadd_paths),
        output=coadd_output,
        downsample=max(1, int(args.downsample)),
    )
    _write_overlay(
        name=f"{args.variant} {args.group} GT {args.band} {args.tract}/{args.patch}",
        image_path=variant_image,
        masks=variant_masks,
        counts=variant_counts,
        target_count=len(variant_paths),
        output=variant_output,
        downsample=max(1, int(args.downsample)),
    )

    summary = []
    for dataset, masks, counts, paths, image_path, output in (
        ("coadd", coadd_masks, coadd_counts, coadd_paths, coadd_image, coadd_output),
        (f"{args.variant}_{args.group}", variant_masks, variant_counts, variant_paths, variant_image, variant_output),
    ):
        row = {
            "dataset": dataset,
            "band": args.band,
            "tract": args.tract,
            "patch": args.patch,
            "image": str(image_path),
            "overlay": str(output),
            "target_tiles": len(paths),
            "clean_count": int(counts.get("clean", 0)),
            "center_only_count": int(counts.get("center_only", 0)),
            "ignore_count": int(counts.get("ignore", 0)),
            "clean_pixels": int(masks["clean"].sum()),
            "center_only_pixels": int(masks["center_only"].sum()),
            "ignore_pixels": int(masks["ignore"].sum()),
        }
        summary.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.out_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
