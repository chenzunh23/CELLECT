#!/usr/bin/env python3
"""Visualize dense targets from existing image-level zarr stores.

Unlike ``diagnose_full_fits_classification.py``, this script does not rerun
preprocessing.  It reads exactly the dense labels, confidence map and source
metadata stored in zarr, i.e. the targets seen by training.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cellect")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import Ellipse

from astro_train_zarr_data import PatchZarrReader, discover_zarr_image_records
from eval.eval_utils import (
    decode_fixed_utf8,
    read_zarr_sample,
    source_rows_from_zarr,
    strict_centers_from_zarr,
    zscale_rgb,
)

V3_DENSE_NAMES = {
    0: "none",
    1: "clean",
    2: "weak_shape",
    3: "ordinary_ignore",
    4: "background",
    5: "strict_center_only",
    6: "restricted_bright_region",
    7: "strict_ignore",
}

V3_DENSE_COLORS = {
    1: np.asarray((0.00, 0.82, 0.36), dtype=np.float32),  # green
    2: np.asarray((0.00, 0.62, 1.00), dtype=np.float32),  # blue-cyan
    3: np.asarray((1.00, 0.18, 0.16), dtype=np.float32),  # red
    4: np.asarray((0.48, 0.48, 0.48), dtype=np.float32),  # gray
    5: np.asarray((0.78, 0.22, 1.00), dtype=np.float32),  # purple/magenta
    6: np.asarray((1.00, 0.69, 0.00), dtype=np.float32),  # amber
    7: np.asarray((0.48, 0.32, 1.00), dtype=np.float32),  # violet
}

CONFIDENCE_COLORS = {
    1: np.asarray((0.10, 0.42, 1.00), dtype=np.float32),
    2: np.asarray((0.00, 0.78, 0.55), dtype=np.float32),
    3: np.asarray((1.00, 0.86, 0.05), dtype=np.float32),
    4: np.asarray((1.00, 0.48, 0.00), dtype=np.float32),
    5: np.asarray((1.00, 0.05, 0.05), dtype=np.float32),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/data/czh23/lupton_zarr_test_0802"))
    p.add_argument("--zarr-store", type=Path, default=None)
    p.add_argument("--patch", default="4,5")
    p.add_argument("--band", default="HSC-I")
    p.add_argument("--dataset-source", default="coadd", choices=["coadd", "denoised", "noisy"])
    p.add_argument("--group", default=None)
    p.add_argument("--tiles", nargs="*", default=None, help="Tile names or short forms such as r4c6.")
    p.add_argument("--num-random", type=int, default=3)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-dir", type=Path, default=Path("output/preprocessing_v3_diagnostics_0803/dense_target_tiles"))
    return p.parse_args()


def normalize_group(group: str | None) -> str | None:
    if group is None or not str(group).strip():
        return None
    text = str(group).strip()
    if text.startswith("group_"):
        return text
    if text.isdigit():
        return f"group_{int(text):02d}"
    return text


def tile_matches(actual: str, requested: str) -> bool:
    actual = str(actual)
    requested = str(requested)
    if actual == requested or actual.endswith("_" + requested):
        return True
    match = re.fullmatch(r"r(\d+)c(\d+)", requested.lower())
    if match:
        row = int(match.group(1))
        col = int(match.group(2))
        return f"grid_r{row:02d}_c{col:02d}_" in actual
    return requested in actual


def zarr_group(reader: PatchZarrReader, idx: int) -> str:
    if not reader.has_array("group"):
        return ""
    groups = decode_fixed_utf8(reader.read_full_small("group"))
    return groups[int(idx)] if int(idx) < len(groups) else ""


def records_for_args(args: argparse.Namespace) -> list[tuple[PatchZarrReader, int, int, str]]:
    if args.zarr_store is not None:
        reader = PatchZarrReader(args.zarr_store.expanduser().resolve())
        attrs = reader.attrs
        bands = list(attrs.get("bands", []))
        band_idx = bands.index(args.band) if args.band in bands else 0
        tile_names = decode_fixed_utf8(reader.read_full_small("tile_name"))
        candidates = [(reader, idx, band_idx, tile_names[idx]) for idx in range(len(tile_names))]
    else:
        root = args.root.expanduser().resolve()
        records = discover_zarr_image_records(root, bands=[args.band])
        candidates = []
        for rec in records:
            if args.patch and rec.patch != args.patch:
                continue
            if args.dataset_source and rec.dataset_source != args.dataset_source:
                continue
            store_s, idx_s = str(rec.image_paths[0])[len("zarr://") :].split("#", 1)
            reader = PatchZarrReader(Path(store_s))
            attrs = reader.attrs
            bands = list(attrs.get("bands", []))
            band_idx = bands.index(args.band) if args.band in bands else 0
            candidates.append((reader, int(idx_s), band_idx, rec.tile_name))
    group = normalize_group(args.group)
    if group is not None:
        candidates = [item for item in candidates if zarr_group(item[0], item[1]) == group]
    if args.tiles:
        selected: list[tuple[PatchZarrReader, int, int, str]] = []
        for requested in args.tiles:
            matches = [item for item in candidates if tile_matches(item[3], requested)]
            if not matches:
                raise RuntimeError(f"no zarr sample matches tile {requested!r}")
            selected.append(matches[0])
        return selected
    rng = random.Random(int(args.seed))
    if len(candidates) <= int(args.num_random):
        return candidates
    return rng.sample(candidates, int(args.num_random))


def source_color(class_id: int) -> str:
    if int(class_id) == 1:
        return "#00d15c"
    if int(class_id) == 2:
        return "#009eff"
    if int(class_id) == 5:
        return "#c738ff"
    return "#ffb000"


def dense_overlay(image: np.ndarray, pu_class: np.ndarray, *, alpha: float = 0.34) -> np.ndarray:
    rgb = zscale_rgb(image)
    classes = np.asarray(pu_class, dtype=np.uint8)
    color = np.zeros_like(rgb)
    visible = np.zeros(classes.shape, dtype=bool)
    for class_id, class_color in V3_DENSE_COLORS.items():
        mask = classes == int(class_id)
        if bool(mask.any()):
            color[mask] = class_color
            visible |= mask
    if bool(visible.any()):
        rgb[visible] = (1.0 - float(alpha)) * rgb[visible] + float(alpha) * color[visible]
    return rgb


def draw_sources(ax: plt.Axes, sources: list[dict[str, float]], strict_centers: list[dict[str, float]]) -> None:
    for row in sorted(sources, key=lambda item: float(item.get("major", 1.0)) * float(item.get("minor", 1.0)), reverse=True):
        color = source_color(int(row.get("class_id", 1)))
        ax.add_patch(
            Ellipse(
                (float(row["x"]), float(row["y"])),
                width=2.0 * max(abs(float(row.get("major", 1.0))), 1.0),
                height=2.0 * max(abs(float(row.get("minor", 1.0))), 1.0),
                angle=np.degrees(float(row.get("theta", 0.0))),
                fill=False,
                edgecolor=color,
                linewidth=0.8,
                alpha=0.9,
            )
        )
        ax.plot(float(row["x"]), float(row["y"]), marker="+", color=color, ms=3.0, mew=0.8)
    if strict_centers:
        ax.scatter(
            [float(row["x"]) for row in strict_centers],
            [float(row["y"]) for row in strict_centers],
            marker="o",
            s=46,
            facecolors="none",
            edgecolors="#c738ff",
            linewidths=1.0,
        )
        ax.scatter(
            [float(row["x"]) for row in strict_centers],
            [float(row["y"]) for row in strict_centers],
            marker="+",
            s=28,
            c="#c738ff",
            linewidths=1.0,
        )


def local_confidence_overlay(image: np.ndarray, confidence: np.ndarray, *, alpha: float = 0.72) -> np.ndarray:
    rgb = zscale_rgb(image)
    for level in sorted(int(v) for v in np.unique(confidence) if int(v) > 0):
        mask = np.asarray(confidence) == level
        color = CONFIDENCE_COLORS.get(level, np.asarray((1.0, 1.0, 1.0), dtype=np.float32))
        rgb[mask] = (1.0 - float(alpha)) * rgb[mask] + float(alpha) * color
    return rgb


def write_counts_csv(path: Path, pu: np.ndarray, conf: np.ndarray, sources: list[dict[str, float]], strict_centers: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values, counts = np.unique(np.asarray(pu, dtype=np.uint8), return_counts=True)
    pixel_counts = {int(v): int(c) for v, c in zip(values, counts)}
    conf_values, conf_counts = np.unique(np.asarray(conf, dtype=np.uint8), return_counts=True)
    confidence_counts = {int(v): int(c) for v, c in zip(conf_values, conf_counts)}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kind", "class_id", "name", "count"])
        writer.writeheader()
        for class_id in sorted(V3_DENSE_NAMES):
            writer.writerow({"kind": "dense_pixels", "class_id": class_id, "name": V3_DENSE_NAMES[class_id], "count": pixel_counts.get(class_id, 0)})
        for class_id in sorted(confidence_counts):
            writer.writerow({"kind": "confidence_pixels", "class_id": class_id, "name": f"conf_{class_id}", "count": confidence_counts[class_id]})
        writer.writerow({"kind": "sources", "class_id": 1, "name": "shape_sources", "count": len(sources)})
        writer.writerow({"kind": "sources", "class_id": 5, "name": "strict_center_only_centers", "count": len(strict_centers)})


def run_one(args: argparse.Namespace, reader: PatchZarrReader, sample_idx: int, band_idx: int, tile_name: str) -> Path:
    attrs = reader.attrs
    patch = str(attrs.get("patch", args.patch))
    bands = list(attrs.get("bands", [args.band]))
    band = bands[band_idx] if band_idx < len(bands) else args.band
    dataset = str(attrs.get("dataset_source", args.dataset_source))
    group = zarr_group(reader, sample_idx)
    sample = read_zarr_sample(reader, sample_idx, band_idx)
    image = np.asarray(sample["display_image"], dtype=np.float32)
    pu = np.asarray(sample["pu_class"], dtype=np.uint8)
    conf = np.asarray(sample["confidence"], dtype=np.uint8)
    sources = source_rows_from_zarr(reader, sample_idx, band_idx)
    strict_centers = strict_centers_from_zarr(reader, sample_idx, band_idx)
    out_dir = args.out_dir.expanduser().resolve() / patch / str(band)
    out_dir.mkdir(parents=True, exist_ok=True)
    group_part = f"_{group}" if group else ""
    stem = f"{dataset}{group_part}_{patch.replace(',', '_')}_{band}_{tile_name}"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150, constrained_layout=True)
    axes[0].imshow(dense_overlay(image, pu), origin="lower", interpolation="nearest")
    axes[0].set_title("dense target classes")
    axes[1].imshow(local_confidence_overlay(image, conf), origin="lower", interpolation="nearest")
    axes[1].set_title("GT confidence")
    axes[2].imshow(zscale_rgb(image), origin="lower", interpolation="nearest")
    axes[2].set_title("shape / strict centers")
    draw_sources(axes[2], sources, strict_centers)
    for ax in axes:
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(0, image.shape[0])
        ax.set_axis_off()
    pu_handles = [mpatches.Patch(color=V3_DENSE_COLORS[k], label=f"{V3_DENSE_NAMES[k]}") for k in (1, 2, 5, 6, 3, 7, 4)]
    conf_handles = [mpatches.Patch(color=CONFIDENCE_COLORS[k], label=f"conf {k}") for k in (1, 2, 3, 4, 5)]
    axes[0].legend(handles=pu_handles, loc="lower right", fontsize=7, framealpha=0.75)
    axes[1].legend(handles=conf_handles, loc="lower right", fontsize=7, framealpha=0.75)
    png_path = out_dir / f"{stem}_dense_target_panel.png"
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    write_counts_csv(out_dir / f"{stem}_dense_target_counts.csv", pu, conf, sources, strict_centers)
    print(f"wrote {png_path}")
    return png_path


def main() -> int:
    args = parse_args()
    selected = records_for_args(args)
    for reader, sample_idx, band_idx, tile_name in selected:
        run_one(args, reader, sample_idx, band_idx, tile_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
