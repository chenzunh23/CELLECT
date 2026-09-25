"""Zarr writing helpers for preprocessing v3.

The writer intentionally keeps the old direct-zarr v2 layout primitives, but
the high-level sample schema is owned by preprocessing v3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from direct_zarr_preprocessing.zarr_writer import ZarrGroupWriter, encode_fixed_utf8, write_json


@dataclass(frozen=True)
class ZarrSampleBatch:
    images: np.ndarray
    dense_labels: np.ndarray
    names: Sequence[str]
    attrs: Mapping[str, object]
    source_centers: np.ndarray | None = None
    source_ids: np.ndarray | None = None
    source_offsets: np.ndarray | None = None
    strict_center_only_centers: np.ndarray | None = None
    strict_center_only_ids: np.ndarray | None = None
    strict_center_only_offsets: np.ndarray | None = None
    shape_source_centers: np.ndarray | None = None
    shape_source_values: np.ndarray | None = None
    shape_source_classes: np.ndarray | None = None
    shape_source_ids: np.ndarray | None = None
    shape_source_offsets: np.ndarray | None = None
    diagnostic_source_rows: Sequence[Mapping[str, object]] | None = None


@dataclass(frozen=True)
class ImageLevelTrainingBatch:
    """Batch matching the direct-zarr image-level training schema."""

    images: np.ndarray
    band_confidence: np.ndarray
    band_conf_weight: np.ndarray
    band_shape: np.ndarray
    band_shape_weight: np.ndarray
    band_pu_class_mask: np.ndarray
    sample_names: Sequence[str]
    tile_x0: np.ndarray
    tile_y0: np.ndarray
    tile_names: Sequence[str]
    groups: Sequence[str]
    dataset_sources: Sequence[str]
    attrs: Mapping[str, object]
    source_centers: np.ndarray
    source_ids: np.ndarray
    source_offsets: np.ndarray
    strict_center_only_centers: np.ndarray
    strict_center_only_ids: np.ndarray
    strict_center_only_offsets: np.ndarray
    shape_source_centers: np.ndarray
    shape_source_values: np.ndarray
    shape_source_classes: np.ndarray
    shape_source_ids: np.ndarray
    shape_source_offsets: np.ndarray
    band_segmentation_ids: np.ndarray | None = None
    band_segmentation_weight: np.ndarray | None = None
    segmentation_overlap: dict | None = None
    band_valid_mask: np.ndarray | None = None


def write_image_level_zarr(
    output: Path | str,
    batch: ZarrSampleBatch,
    *,
    overwrite: bool,
    image_chunks: tuple[int, ...] | None = None,
) -> Path:
    output = Path(output)
    images = np.asarray(batch.images, dtype=np.float32)
    dense = np.asarray(batch.dense_labels, dtype=np.uint8)
    if images.shape[0] != dense.shape[0]:
        raise ValueError(f"image/label sample mismatch: {images.shape[0]} != {dense.shape[0]}")
    attrs = dict(batch.attrs)
    attrs["schema"] = "preprocessing_v3_image_level"
    attrs["num_samples"] = int(images.shape[0])
    group = ZarrGroupWriter(output, overwrite=overwrite, attrs=attrs)
    if image_chunks is None:
        image_chunks = (1,) + tuple(images.shape[1:])
    group.array("images", shape=images.shape, chunks=image_chunks, dtype=np.float32).write_full(images)
    group.array("dense_labels", shape=dense.shape, chunks=(1,) + tuple(dense.shape[1:]), dtype=np.uint8).write_full(dense)
    names = encode_fixed_utf8(list(batch.names), 192)
    group.array("sample_names", shape=names.shape, chunks=(max(1, min(len(names), 1024)), names.shape[1]), dtype=np.uint8).write_full(names)
    if batch.source_centers is not None and batch.source_ids is not None and batch.source_offsets is not None:
        centers = np.asarray(batch.source_centers, dtype=np.float32).reshape(-1, 2)
        ids = np.asarray(batch.source_ids, dtype=np.int64).reshape(-1)
        offsets = np.asarray(batch.source_offsets, dtype=np.int64)
        group.array("source_centers", shape=centers.shape, chunks=(max(1, len(centers)), 2), dtype=np.float32).write_full(centers)
        group.array("source_ids", shape=ids.shape, chunks=(max(1, len(ids)),), dtype=np.int64).write_full(ids)
        group.array("source_offsets", shape=offsets.shape, chunks=(max(1, offsets.shape[0]), offsets.shape[1]), dtype=np.int64).write_full(offsets)
    if (
        batch.strict_center_only_centers is not None
        and batch.strict_center_only_ids is not None
        and batch.strict_center_only_offsets is not None
    ):
        centers = np.asarray(batch.strict_center_only_centers, dtype=np.float32).reshape(-1, 2)
        ids = np.asarray(batch.strict_center_only_ids, dtype=np.int64).reshape(-1)
        offsets = np.asarray(batch.strict_center_only_offsets, dtype=np.int64)
        group.array("strict_center_only_centers", shape=centers.shape, chunks=(max(1, len(centers)), 2), dtype=np.float32).write_full(centers)
        group.array("strict_center_only_ids", shape=ids.shape, chunks=(max(1, len(ids)),), dtype=np.int64).write_full(ids)
        group.array("strict_center_only_offsets", shape=offsets.shape, chunks=(max(1, offsets.shape[0]), offsets.shape[1]), dtype=np.int64).write_full(offsets)
    if (
        batch.shape_source_centers is not None
        and batch.shape_source_values is not None
        and batch.shape_source_classes is not None
        and batch.shape_source_ids is not None
        and batch.shape_source_offsets is not None
    ):
        centers = np.asarray(batch.shape_source_centers, dtype=np.float32).reshape(-1, 2)
        values = np.asarray(batch.shape_source_values, dtype=np.float32).reshape(-1, 3)
        classes = np.asarray(batch.shape_source_classes, dtype=np.uint8).reshape(-1)
        ids = np.asarray(batch.shape_source_ids, dtype=np.int64).reshape(-1)
        offsets = np.asarray(batch.shape_source_offsets, dtype=np.int64)
        group.array("shape_source_centers", shape=centers.shape, chunks=(max(1, len(centers)), 2), dtype=np.float32).write_full(centers)
        group.array("shape_source_values", shape=values.shape, chunks=(max(1, len(values)), 3), dtype=np.float32).write_full(values)
        group.array("shape_source_classes", shape=classes.shape, chunks=(max(1, len(classes)),), dtype=np.uint8).write_full(classes)
        group.array("shape_source_ids", shape=ids.shape, chunks=(max(1, len(ids)),), dtype=np.int64).write_full(ids)
        group.array("shape_source_offsets", shape=offsets.shape, chunks=(max(1, offsets.shape[0]), offsets.shape[1]), dtype=np.int64).write_full(offsets)
    if batch.diagnostic_source_rows is not None:
        write_json(output / "diagnostic_source_rows.json", {"rows": list(batch.diagnostic_source_rows)})
    return output


def write_training_image_level_zarr(
    output: Path | str,
    batch: ImageLevelTrainingBatch,
    *,
    overwrite: bool,
    chunk_tiles: int = 16,
) -> Path:
    """Write zarr arrays consumed by ``discover_zarr_image_records``."""

    output = Path(output)
    images = np.asarray(batch.images, dtype=np.float32)
    if images.ndim not in (4, 5):
        raise ValueError(f"images must be (N,B,H,W) or (N,B,C,H,W), got {images.shape}")
    n = int(images.shape[0])
    bands = int(images.shape[1])
    h, w = int(images.shape[-2]), int(images.shape[-1])
    if batch.band_valid_mask is not None and np.shape(batch.band_valid_mask)!=(n,bands,h,w):
        raise ValueError('validity/image shape mismatch')
    seg_ids = batch.band_segmentation_ids
    seg_weight = batch.band_segmentation_weight
    if (seg_ids is None) != (seg_weight is None):
        raise ValueError("segmentation IDs and weights must be supplied together")
    if seg_ids is not None:
        seg_ids = np.asarray(seg_ids)
        seg_weight = np.asarray(seg_weight)
        expected = (n, bands, h, w)
        if seg_ids.shape != expected or seg_weight.shape != expected:
            raise ValueError(f"segmentation arrays must have shape {expected}")
        if (not np.issubdtype(seg_ids.dtype, np.integer) or np.any(seg_ids < 0)
                or np.any(seg_ids > np.iinfo(np.int32).max)):
            raise ValueError("segmentation IDs must be nonnegative int32-compatible integers")
        if (not np.isfinite(seg_weight).all() or np.any(seg_weight < 0)
                or np.any(seg_weight > 1) or np.any(seg_weight[seg_ids == 0] != 0)):
            raise ValueError("segmentation weights must be finite in [0,1], zero outside instances")
    overlap = batch.segmentation_overlap
    if overlap is not None:
        from .utils.segmentation_storage import validate_overlap_masks
        if seg_ids is None:
            raise ValueError('overlap sidecars require the base segmentation ID plane')
        validate_overlap_masks(overlap, n, bands, h, w)
    attrs = dict(batch.attrs)
    attrs.setdefault("format", "cellect_direct_patch_zarr")
    attrs["schema"] = "preprocessing_v3_image_level"
    attrs["image_level_training"] = True
    attrs["num_samples"] = n
    attrs["segmentation_supervision"] = "positive_only" if seg_ids is not None else "none"
    if seg_ids is not None:
        attrs["segmentation_id_namespace"] = "cosmos_catalog_id"
    if overlap is not None:
        from .utils.segmentation_storage import MASK_COLUMNS, MASK_ENCODING
        attrs['segmentation_overlap_encoding'] = MASK_ENCODING
        attrs['segmentation_overlap_columns'] = MASK_COLUMNS
        attrs['segmentation_instance_reader'] = 'preprocessing.utils.segmentation_storage.iter_training_segmentation_masks'
    writer = ZarrGroupWriter(output, overwrite=overwrite, attrs=attrs)
    chunk_tiles = max(1, int(chunk_tiles))
    if images.ndim == 5:
        image_chunks = (min(chunk_tiles, max(1, n)), bands, images.shape[2], h, w)
    else:
        image_chunks = (min(chunk_tiles, max(1, n)), bands, h, w)
    writer.array("images", shape=images.shape, chunks=image_chunks, dtype=np.float32).write_full(images)
    if batch.band_valid_mask is not None:
        valid=np.asarray(batch.band_valid_mask,bool)
        writer.array('band_valid_mask',shape=valid.shape,chunks=(1,bands,h,w),dtype=np.uint8).write_full(valid.astype(np.uint8))

    def _write_band_array(name: str, values: np.ndarray, dtype, extra: tuple[int, ...] = ()) -> None:
        arr = np.asarray(values, dtype=dtype)
        expected = (n, bands) + extra + (h, w)
        if arr.shape != expected:
            raise ValueError(f"{name} shape {arr.shape} != {expected}")
        chunks = (min(chunk_tiles, max(1, n)), bands) + extra + (h, w)
        writer.array(name, shape=arr.shape, chunks=chunks, dtype=dtype).write_full(arr)

    _write_band_array("band_confidence", batch.band_confidence, np.uint8)
    _write_band_array("band_conf_weight", batch.band_conf_weight, np.float32)
    _write_band_array("band_shape", batch.band_shape, np.float32, extra=(3,))
    _write_band_array("band_shape_weight", batch.band_shape_weight, np.float32)
    _write_band_array("band_pu_class_mask", batch.band_pu_class_mask, np.uint8)
    if seg_ids is not None:
        _write_band_array("band_segmentation_ids", seg_ids, np.int32)
        _write_band_array("band_segmentation_weight", seg_weight, np.float32)

    writer.array("tile_x0", shape=(n,), chunks=(max(1, n),), dtype=np.int32).write_full(np.asarray(batch.tile_x0, dtype=np.int32))
    writer.array("tile_y0", shape=(n,), chunks=(max(1, n),), dtype=np.int32).write_full(np.asarray(batch.tile_y0, dtype=np.int32))
    writer.array("tile_name", shape=(n, 192), chunks=(max(1, n), 192), dtype=np.uint8).write_full(encode_fixed_utf8(list(batch.tile_names), 192))
    writer.array("sample_names", shape=(n, 192), chunks=(max(1, n), 192), dtype=np.uint8).write_full(encode_fixed_utf8(list(batch.sample_names), 192))
    writer.array("group", shape=(n, 32), chunks=(max(1, n), 32), dtype=np.uint8).write_full(encode_fixed_utf8(list(batch.groups), 32))
    writer.array("dataset_source", shape=(n, 32), chunks=(max(1, n), 32), dtype=np.uint8).write_full(encode_fixed_utf8(list(batch.dataset_sources), 32))

    def _write_flat(name: str, values: np.ndarray, dtype, width: int | None = None) -> None:
        arr = np.asarray(values, dtype=dtype)
        if width is not None:
            arr = arr.reshape(-1, width)
            chunks = (max(1, len(arr)), width)
        else:
            arr = arr.reshape(-1)
            chunks = (max(1, len(arr)),)
        writer.array(name, shape=arr.shape, chunks=chunks, dtype=dtype).write_full(arr)

    if overlap is not None:
        _write_flat('segmentation_overlap_meta', overlap['meta'], np.int64, width=8)
        _write_flat('segmentation_overlap_data', overlap['data'], np.uint8)
        _write_flat('segmentation_overlap_offsets', overlap['offsets'], np.int64)
        _write_flat('segmentation_overlap_weights', overlap['weights'], np.float32)
    _write_flat("source_centers", batch.source_centers, np.float32, width=2)
    _write_flat("source_ids", batch.source_ids, np.int64)
    writer.array("source_offsets", shape=np.asarray(batch.source_offsets).shape, chunks=(max(1, n), bands + 1), dtype=np.int64).write_full(
        np.asarray(batch.source_offsets, dtype=np.int64)
    )
    _write_flat("strict_center_only_centers", batch.strict_center_only_centers, np.float32, width=2)
    _write_flat("strict_center_only_ids", batch.strict_center_only_ids, np.int64)
    writer.array(
        "strict_center_only_offsets",
        shape=np.asarray(batch.strict_center_only_offsets).shape,
        chunks=(max(1, n), bands + 1),
        dtype=np.int64,
    ).write_full(np.asarray(batch.strict_center_only_offsets, dtype=np.int64))
    _write_flat("shape_source_centers", batch.shape_source_centers, np.float32, width=2)
    _write_flat("shape_source_values", batch.shape_source_values, np.float32, width=3)
    _write_flat("shape_source_classes", batch.shape_source_classes, np.uint8)
    _write_flat("shape_source_ids", batch.shape_source_ids, np.int64)
    writer.array(
        "shape_source_offsets",
        shape=np.asarray(batch.shape_source_offsets).shape,
        chunks=(max(1, n), bands + 1),
        dtype=np.int64,
    ).write_full(np.asarray(batch.shape_source_offsets, dtype=np.int64))
    write_json(
        output.parent / f"{output.name}_manifest.json",
        {
            "output": str(output),
            "num_samples": n,
            "bands": list(attrs.get("bands", [])),
            "schema": attrs["schema"],
        },
    )
    return output
