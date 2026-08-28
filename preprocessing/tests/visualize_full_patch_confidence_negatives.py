#!/usr/bin/env python3
"""Stitch image-level Zarr tiles into full-patch confidence-negative overlays."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cellect")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image

from astro_train_zarr_data import PatchZarrReader, discover_zarr_image_records
from eval.eval_utils import read_zarr_sample, zscale_gray, zarr_sample_group


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--patch", default="4,5")
    p.add_argument("--bands", nargs="+", required=True)
    p.add_argument("--dataset-source", default="coadd", choices=("coadd", "noisy", "denoised"))
    p.add_argument("--group", default=None, help="Variant group for noisy/denoised, e.g. group_00 or 0.")
    p.add_argument("--patch-size", type=int, default=4200)
    p.add_argument("--origin-x", type=int, default=None, help="Physical x origin to subtract. Defaults to minimum tile_x0.")
    p.add_argument("--origin-y", type=int, default=None, help="Physical y origin to subtract. Defaults to minimum tile_y0.")
    p.add_argument("--alpha", type=float, default=0.70, help="Opacity used to cover non-negative pixels.")
    p.add_argument("--out-dir", type=Path, default=Path("output/eval_0815/confidence_negative_full_patch"))
    return p.parse_args()


def _parse_zarr_uri(uri: str) -> tuple[Path, int]:
    if not str(uri).startswith("zarr://") or "#" not in str(uri):
        raise ValueError(f"invalid zarr URI: {uri}")
    store_s, idx_s = str(uri)[len("zarr://") :].rsplit("#", 1)
    return Path(store_s), int(idx_s)


def _normalize_group(group: str | None) -> str | None:
    if group is None or not str(group).strip():
        return None
    text = str(group).strip()
    if text.startswith("group_"):
        return text
    if text.isdigit():
        return f"group_{int(text):02d}"
    return text


def _records_for_band(args: argparse.Namespace, band: str):
    group = _normalize_group(args.group)
    records = []
    for rec in discover_zarr_image_records(args.root.expanduser().resolve(), bands=[band]):
        rec_patch = str(rec.patch).split("__", 1)[0]
        if rec_patch != str(args.patch):
            continue
        if str(rec.dataset_source) != str(args.dataset_source):
            continue
        if group is not None:
            store, idx = _parse_zarr_uri(rec.image_paths[0])
            if zarr_sample_group(PatchZarrReader(store), idx) != group:
                continue
        records.append(rec)
    if not records:
        raise RuntimeError(f"no records found for {args.dataset_source} {args.patch} {band}")
    return sorted(records, key=lambda r: (int(r.y0), int(r.x0), str(r.tile_name)))


def _overlay_clear_negative(gray: np.ndarray, negative: np.ndarray, *, alpha: float) -> np.ndarray:
    rgb = np.repeat(np.asarray(gray, dtype=np.float32)[..., None], 3, axis=2)
    cover = ~np.asarray(negative, dtype=bool)
    cover_color = np.asarray((0.10, 0.36, 0.90), dtype=np.float32)
    if bool(cover.any()):
        rgb[cover] = (1.0 - float(alpha)) * rgb[cover] + float(alpha) * cover_color
    return np.clip(rgb, 0.0, 1.0)


def _save_pixel_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.rint(np.flipud(np.asarray(rgb, dtype=np.float32)) * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def stitch_band(args: argparse.Namespace, band: str) -> dict[str, object]:
    patch_size = int(args.patch_size)
    image_sum = np.zeros((patch_size, patch_size), dtype=np.float32)
    image_count = np.zeros((patch_size, patch_size), dtype=np.float32)
    negative_sum = np.zeros((patch_size, patch_size), dtype=np.float32)
    negative_count = np.zeros((patch_size, patch_size), dtype=np.float32)
    readers: dict[Path, PatchZarrReader] = {}
    records = _records_for_band(args, band)
    origin_x = int(args.origin_x) if args.origin_x is not None else min(int(rec.x0) for rec in records)
    origin_y = int(args.origin_y) if args.origin_y is not None else min(int(rec.y0) for rec in records)
    for rec in records:
        store, idx = _parse_zarr_uri(rec.image_paths[0])
        reader = readers.get(store)
        if reader is None:
            reader = PatchZarrReader(store)
            readers[store] = reader
        bands = list(reader.attrs.get("bands", []))
        band_idx = bands.index(band) if band in bands else 0
        sample = read_zarr_sample(reader, idx, band_idx)
        image = np.asarray(sample["display_image"], dtype=np.float32)
        negative = (np.asarray(sample["confidence"], dtype=np.uint8) == 0) & (
            np.asarray(sample["confidence_weight"], dtype=np.float32) > 0.0
        )
        x0 = max(0, int(rec.x0) - origin_x)
        y0 = max(0, int(rec.y0) - origin_y)
        x1 = min(patch_size, x0 + image.shape[1])
        y1 = min(patch_size, y0 + image.shape[0])
        if x1 <= x0 or y1 <= y0:
            continue
        view = image[: y1 - y0, : x1 - x0]
        neg_view = negative[: y1 - y0, : x1 - x0]
        finite = np.isfinite(view)
        image_sum_view = image_sum[y0:y1, x0:x1]
        image_count_view = image_count[y0:y1, x0:x1]
        image_sum_view[finite] += view[finite]
        image_count_view[finite] += 1.0
        image_sum[y0:y1, x0:x1] = image_sum_view
        image_count[y0:y1, x0:x1] = image_count_view
        negative_sum[y0:y1, x0:x1] += neg_view.astype(np.float32)
        negative_count[y0:y1, x0:x1] += 1.0

    covered = image_count > 0
    full_image = np.zeros((patch_size, patch_size), dtype=np.float32)
    full_image[covered] = image_sum[covered] / image_count[covered]
    full_negative = np.zeros((patch_size, patch_size), dtype=bool)
    voted = negative_count > 0
    full_negative[voted] = (negative_sum[voted] / negative_count[voted]) >= 0.5
    display_image = full_image.copy()
    display_image[~covered] = np.nan
    gray = zscale_gray(display_image)
    rgb = _overlay_clear_negative(gray, full_negative, alpha=float(args.alpha))
    out_dir = args.out_dir.expanduser().resolve() / str(args.patch) / band
    group_part = f"_{_normalize_group(args.group)}" if _normalize_group(args.group) else ""
    stem = f"{args.dataset_source}{group_part}_{str(args.patch).replace(',', '_')}_{band}_confidence_negative_clear_full_patch"
    out_path = out_dir / f"{stem}.png"
    _save_pixel_png(out_path, rgb)
    summary = {
        "band": band,
        "patch": str(args.patch),
        "dataset_source": str(args.dataset_source),
        "group": _normalize_group(args.group),
        "records": len(records),
        "patch_size": patch_size,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "covered_pixels": int(np.count_nonzero(covered)),
        "negative_pixels": int(np.count_nonzero(full_negative)),
        "negative_fraction_covered": float(np.count_nonzero(full_negative) / max(np.count_nonzero(covered), 1)),
        "output": str(out_path),
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    summaries = [stitch_band(args, band) for band in args.bands]
    for row in summaries:
        print(row["output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
