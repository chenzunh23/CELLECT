#!/usr/bin/env python
"""Pack one legacy CELLECT preprocessed patch into a patch-level Zarr store."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zarr_preprocessing.schema import (  # noqa: E402
    DEFAULT_BANDS,
    band_metadata_path,
    band_target_path,
    infer_dataset_source,
    infer_tract_patch,
    read_tiles,
    zscale_cache_path,
)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")


class ManualZarrArray:
    """Small Zarr v2 directory-array writer.

    The zarr 3.1 synchronous API deadlocks in this environment, so preprocessing
    writes standard v2 metadata and raw uncompressed chunks directly.
    """

    def __init__(self, path: Path, *, shape: Sequence[int], chunks: Sequence[int], dtype: np.dtype):
        self.path = path
        self.shape = tuple(int(v) for v in shape)
        self.chunks = tuple(max(1, int(v)) for v in chunks)
        self.dtype = np.dtype(dtype)
        self.path.mkdir(parents=True, exist_ok=True)
        _write_json(
            self.path / ".zarray",
            {
                "zarr_format": 2,
                "shape": list(self.shape),
                "chunks": list(self.chunks),
                "dtype": self.dtype.str,
                "compressor": None,
                "filters": None,
                "fill_value": 0,
                "order": "C",
            },
        )
        _write_json(self.path / ".zattrs", {})

    def _chunk_name(self, chunk_index: Sequence[int]) -> str:
        return ".".join(str(int(v)) for v in chunk_index)

    def write_chunk(self, chunk_index: Sequence[int], data: np.ndarray) -> None:
        arr = np.asarray(data, dtype=self.dtype)
        (self.path / self._chunk_name(chunk_index)).write_bytes(arr.tobytes(order="C"))

    def write_full(self, data: np.ndarray) -> None:
        arr = np.asarray(data, dtype=self.dtype)
        if arr.shape != self.shape:
            raise ValueError(f"{self.path.name}: expected {self.shape}, got {arr.shape}")
        if 0 in self.shape:
            return
        ranges = [range((dim + chunk - 1) // chunk) for dim, chunk in zip(self.shape, self.chunks)]
        for chunk_index in product(*ranges):
            src = []
            for axis, ci in enumerate(chunk_index):
                start = ci * self.chunks[axis]
                end = min(self.shape[axis], start + self.chunks[axis])
                src.append(slice(start, end))
            self.write_chunk(chunk_index, arr[tuple(src)])


def _encode_fixed_utf8(values: Sequence[str], width: int) -> np.ndarray:
    out = np.zeros((len(values), width), dtype=np.uint8)
    for i, value in enumerate(values):
        raw = str(value).encode("utf-8")[:width]
        out[i, : len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return out


def read_image_tensor(path: Path, *, dtype: np.dtype) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"zscale cache not found: {path}")
    tensor = torch.load(path, map_location="cpu")
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)
    if arr.ndim != 3:
        raise ValueError(f"Expected zscale image [B,H,W], got {arr.shape}: {path}")
    return np.asarray(arr, dtype=dtype)


def read_target(path: Path, *, include_shape: bool, target_float_dtype: np.dtype) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"band target not found: {path}")
    with np.load(path) as data:
        out = {
            "confidence": np.asarray(data["confidence"], dtype=np.uint8),
            "confidence_weight": np.asarray(data.get("confidence_weight", data["confidence"] > 0), dtype=target_float_dtype),
        }
        if include_shape:
            out["shape"] = np.asarray(data["shape"], dtype=target_float_dtype)
            out["shape_weight"] = np.asarray(data["shape_weight"], dtype=target_float_dtype)
    return out


def read_metadata(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    with np.load(path) as data:
        centers = np.asarray(data.get("centers", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32).reshape(-1, 2)
        ids = np.asarray(data.get("ids", np.zeros((centers.shape[0],), dtype=np.int64)), dtype=np.int64).reshape(-1)
    if ids.shape[0] != centers.shape[0]:
        ids = np.resize(ids, (centers.shape[0],)).astype(np.int64, copy=False)
    return centers, ids


def count_sources(patch_root: Path, tiles, bands: Sequence[str]) -> tuple[np.ndarray, int]:
    offsets = np.zeros((len(tiles), len(bands) + 1), dtype=np.int64)
    total = 0
    for i, tile in enumerate(tiles):
        offsets[i, 0] = total
        for b, band in enumerate(bands):
            centers, _ids = read_metadata(band_metadata_path(patch_root, band, tile.name))
            total += int(centers.shape[0])
            offsets[i, b + 1] = total
    return offsets, total


def load_tile_package(
    *,
    patch_root: Path,
    zscale_root: Path,
    dataset_source: str,
    tract: str,
    patch: str,
    tile,
    bands: Sequence[str],
    fits_hdu: int,
    image_dtype: np.dtype,
    target_float_dtype: np.dtype,
    include_shape: bool,
) -> dict[str, object]:
    image_path = zscale_cache_path(
        zscale_root,
        dataset_source=dataset_source,
        tract=tract,
        patch=patch,
        tile_name=tile.name,
        bands=bands,
        fits_hdu=fits_hdu,
    )
    image = read_image_tensor(image_path, dtype=image_dtype)
    if image.shape[0] != len(bands):
        raise ValueError(f"{image_path} has {image.shape[0]} bands, expected {len(bands)}")

    confidence = []
    conf_weight = []
    shape = []
    shape_weight = []
    centers_by_band = []
    ids_by_band = []
    for band in bands:
        target = read_target(
            band_target_path(patch_root, band, tile.name),
            include_shape=include_shape,
            target_float_dtype=target_float_dtype,
        )
        confidence.append(target["confidence"])
        conf_weight.append(target["confidence_weight"])
        if include_shape:
            shape.append(target["shape"])
            shape_weight.append(target["shape_weight"])
        centers, ids = read_metadata(band_metadata_path(patch_root, band, tile.name))
        centers_by_band.append(centers)
        ids_by_band.append(ids)

    out: dict[str, object] = {
        "image": image,
        "confidence": np.stack(confidence, axis=0).astype(np.uint8, copy=False),
        "confidence_weight": np.stack(conf_weight, axis=0).astype(target_float_dtype, copy=False),
        "centers_by_band": centers_by_band,
        "ids_by_band": ids_by_band,
    }
    if include_shape:
        out["shape"] = np.stack(shape, axis=0).astype(target_float_dtype, copy=False)
        out["shape_weight"] = np.stack(shape_weight, axis=0).astype(target_float_dtype, copy=False)
    return out


def create_array(root: Path, name: str, *, shape, chunks, dtype):
    return ManualZarrArray(root / name, shape=shape, chunks=chunks, dtype=np.dtype(dtype))


def pack_patch(args: argparse.Namespace) -> Path:
    patch_root = args.patch_root.expanduser().resolve()
    zscale_root = args.zscale_root.expanduser().resolve()
    bands = tuple(args.bands)
    dataset_source = infer_dataset_source(patch_root, args.dataset_source)
    tract, patch = infer_tract_patch(patch_root, args.tract, args.patch)
    if not tract or not patch:
        raise ValueError("--tract and --patch are required when they cannot be inferred from patch_root")

    tiles = read_tiles(patch_root)
    if args.tile_filter:
        wanted = set(args.tile_filter)
        tiles = [tile for tile in tiles if tile.name in wanted]
    if not tiles:
        raise RuntimeError(f"No tiles selected under {patch_root}")

    image_dtype = np.dtype(args.image_dtype)
    target_float_dtype = np.dtype(args.target_float_dtype)
    include_shape = bool(args.include_shape)
    h = w = int(args.tile_size)
    n = len(tiles)
    b = len(bands)
    chunk_tiles = max(1, int(args.chunk_tiles))

    output = args.output.expanduser().resolve()
    if output.exists():
        if args.overwrite:
            shutil.rmtree(output)
        else:
            raise FileExistsError(f"Output exists; use --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    print(f"[zarr] first pass: source offsets for {n} tile(s)", flush=True)
    source_offsets, total_sources = count_sources(patch_root, tiles, bands)
    print(f"[zarr] first pass done: total_sources={total_sources}", flush=True)

    _write_json(output / ".zgroup", {"zarr_format": 2})
    _write_json(
        output / ".zattrs",
        {
            "format": "cellect_patch_zarr",
            "format_version": 1,
            "patch_root": str(patch_root),
            "zscale_root": str(zscale_root),
            "dataset_source": dataset_source,
            "tract": tract,
            "patch": patch,
            "bands": list(bands),
            "tile_size": h,
            "include_shape": include_shape,
            "image_dtype": str(image_dtype),
            "target_float_dtype": str(target_float_dtype),
            "source_offsets_semantics": "source_offsets[i,b]: start offset for sample i band b; source_offsets[i,b+1]: end",
            "created_unix_time": time.time(),
        },
    )

    print("[zarr] creating arrays", flush=True)
    images = create_array(output, "images", shape=(n, b, h, w), chunks=(chunk_tiles, b, h, w), dtype=image_dtype)
    confidence = create_array(output, "band_confidence", shape=(n, b, h, w), chunks=(chunk_tiles, b, h, w), dtype=np.uint8)
    conf_weight = create_array(
        output,
        "band_conf_weight",
        shape=(n, b, h, w),
        chunks=(chunk_tiles, b, h, w),
        dtype=target_float_dtype,
    )
    if include_shape:
        shape = create_array(
            output,
            "band_shape",
            shape=(n, b, 3, h, w),
            chunks=(chunk_tiles, b, 3, h, w),
            dtype=target_float_dtype,
        )
        shape_weight = create_array(
            output,
            "band_shape_weight",
            shape=(n, b, h, w),
            chunks=(chunk_tiles, b, h, w),
            dtype=target_float_dtype,
        )
    else:
        shape = None
        shape_weight = None

    centers = create_array(output, "source_centers", shape=(total_sources, 2), chunks=(max(1, total_sources), 2), dtype=np.float32)
    ids = create_array(output, "source_ids", shape=(total_sources,), chunks=(max(1, total_sources),), dtype=np.int64)
    source_centers_flat = np.zeros((total_sources, 2), dtype=np.float32)
    source_ids_flat = np.zeros((total_sources,), dtype=np.int64)

    create_array(output, "source_offsets", shape=source_offsets.shape, chunks=(max(1, n), b + 1), dtype=np.int64).write_full(source_offsets)
    create_array(output, "tile_x0", shape=(n,), chunks=(max(1, n),), dtype=np.int32).write_full(np.asarray([tile.x0 for tile in tiles], dtype=np.int32))
    create_array(output, "tile_y0", shape=(n,), chunks=(max(1, n),), dtype=np.int32).write_full(np.asarray([tile.y0 for tile in tiles], dtype=np.int32))
    create_array(output, "tile_name", shape=(n, 192), chunks=(max(1, n), 192), dtype=np.uint8).write_full(_encode_fixed_utf8(
        [tile.name for tile in tiles],
        192,
    ))
    create_array(output, "group", shape=(n, 32), chunks=(max(1, n), 32), dtype=np.uint8).write_full(_encode_fixed_utf8(
        [tile.group for tile in tiles],
        32,
    ))
    create_array(output, "dataset_source", shape=(n, 32), chunks=(max(1, n), 32), dtype=np.uint8).write_full(_encode_fixed_utf8(
        [dataset_source] * n,
        32,
    ))
    print("[zarr] arrays ready", flush=True)

    samples_csv = output.parent / f"{output.name}_samples.csv"
    with samples_csv.open("w", encoding="utf-8") as handle:
        handle.write("index,tile_name,x0,y0,group,dataset_source\n")
        for i, tile in enumerate(tiles):
            handle.write(f"{i},{tile.name},{tile.x0},{tile.y0},{tile.group},{dataset_source}\n")

    print(f"[zarr] writing {output} with {n} tile(s), total_sources={total_sources}", flush=True)
    max_workers = max(1, int(args.workers))
    for start in range(0, n, chunk_tiles):
        end = min(n, start + chunk_tiles)
        batch_tiles = tiles[start:end]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            packages = list(
                executor.map(
                    lambda tile: load_tile_package(
                        patch_root=patch_root,
                        zscale_root=zscale_root,
                        dataset_source=dataset_source,
                        tract=tract,
                        patch=patch,
                        tile=tile,
                        bands=bands,
                        fits_hdu=int(args.fits_hdu),
                        image_dtype=image_dtype,
                        target_float_dtype=target_float_dtype,
                        include_shape=include_shape,
                    ),
                    batch_tiles,
                )
            )
        chunk_index = (start // chunk_tiles, 0, 0, 0)
        images.write_chunk(chunk_index, np.stack([pkg["image"] for pkg in packages], axis=0))
        confidence.write_chunk(chunk_index, np.stack([pkg["confidence"] for pkg in packages], axis=0))
        conf_weight.write_chunk(chunk_index, np.stack([pkg["confidence_weight"] for pkg in packages], axis=0))
        if include_shape and shape is not None and shape_weight is not None:
            shape.write_chunk((start // chunk_tiles, 0, 0, 0, 0), np.stack([pkg["shape"] for pkg in packages], axis=0))
            shape_weight.write_chunk(chunk_index, np.stack([pkg["shape_weight"] for pkg in packages], axis=0))
        for local_idx, pkg in enumerate(packages):
            sample_idx = start + local_idx
            for band_idx in range(b):
                s0 = int(source_offsets[sample_idx, band_idx])
                s1 = int(source_offsets[sample_idx, band_idx + 1])
                if s1 <= s0:
                    continue
                source_centers_flat[s0:s1] = pkg["centers_by_band"][band_idx]
                source_ids_flat[s0:s1] = pkg["ids_by_band"][band_idx]
        print(f"[zarr] wrote tiles {start + 1}-{end}/{n}", flush=True)

    centers.write_full(source_centers_flat)
    ids.write_full(source_ids_flat)

    metadata = {
        "output": str(output),
        "samples_csv": str(samples_csv),
        "patch_root": str(patch_root),
        "dataset_source": dataset_source,
        "tract": tract,
        "patch": patch,
        "bands": list(bands),
        "num_tiles": n,
        "total_sources": int(total_sources),
        "arrays": {
            "images": list(images.shape),
            "band_confidence": list(confidence.shape),
            "band_conf_weight": list(conf_weight.shape),
            "source_centers": list(centers.shape),
            "source_ids": list(ids.shape),
            "source_offsets": list(source_offsets.shape),
        },
        "tile_records": [asdict(tile) for tile in tiles],
    }
    if include_shape and shape is not None and shape_weight is not None:
        metadata["arrays"]["band_shape"] = list(shape.shape)
        metadata["arrays"]["band_shape_weight"] = list(shape_weight.shape)
    (output.parent / f"{output.name}_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-root", type=Path, required=True, help="Legacy preprocessed patch root.")
    parser.add_argument("--zscale-root", type=Path, required=True, help="Root containing precomputed zscale .pt files.")
    parser.add_argument("--output", type=Path, required=True, help="Output patch .zarr directory.")
    parser.add_argument("--tract", default="", help="Tract, inferred from patch_root/manifest when omitted.")
    parser.add_argument("--patch", default="", help="Patch, inferred from patch_root/manifest when omitted.")
    parser.add_argument("--dataset-source", default="auto", help="coadd, denoised, noisy, or auto.")
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    parser.add_argument("--fits-hdu", type=int, default=1)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--image-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--target-float-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--include-shape", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--chunk-tiles", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4, help="Threads used to read tile packages inside each write batch.")
    parser.add_argument("--tile-filter", nargs="*", default=())
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    pack_patch(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
