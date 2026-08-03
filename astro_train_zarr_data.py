"""Zarr-backed Dataset adapter for AstroCELLECT training/evaluation."""

from __future__ import annotations

import json
import multiprocessing as mp
import random
import weakref
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset, Sampler, get_worker_info

from astro_train_data import CutoutRecord, collate_cutouts


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _decode_fixed_utf8(arr: np.ndarray) -> list[str]:
    out = []
    for row in np.asarray(arr, dtype=np.uint8):
        out.append(bytes(row).split(b"\0", 1)[0].decode("utf-8", errors="replace"))
    return out


@dataclass(frozen=True)
class _ArrayMeta:
    shape: tuple[int, ...]
    chunks: tuple[int, ...]
    dtype: np.dtype


class PatchZarrReader:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.attrs = _read_json(self.root / ".zattrs")
        self._meta: Dict[str, _ArrayMeta] = {}
        self._logical_chunk0: int | None = None
        self._chunk_cache: Dict[str, np.ndarray] = {}
        self._full_cache: Dict[str, np.ndarray] = {}

    def has_array(self, name: str) -> bool:
        return (self.root / name / ".zarray").is_file()

    def meta(self, name: str) -> _ArrayMeta:
        cached = self._meta.get(name)
        if cached is not None:
            return cached
        raw = _read_json(self.root / name / ".zarray")
        meta = _ArrayMeta(
            shape=tuple(int(v) for v in raw["shape"]),
            chunks=tuple(int(v) for v in raw["chunks"]),
            dtype=np.dtype(raw["dtype"]),
        )
        self._meta[name] = meta
        return meta

    def _chunk_path(self, name: str, index: Sequence[int]) -> Path:
        return self.root / name / ".".join(str(int(v)) for v in index)

    def read_first_axis(self, name: str, idx: int) -> np.ndarray:
        meta = self.meta(name)
        chunk0 = int(idx) // meta.chunks[0]
        local = int(idx) - chunk0 * meta.chunks[0]
        if self._logical_chunk0 != chunk0:
            self._logical_chunk0 = chunk0
            self._chunk_cache.clear()
        data = self._chunk_cache.get(name)
        if data is None:
            chunk_index = (chunk0,) + tuple(0 for _ in meta.shape[1:])
            first = min(meta.chunks[0], meta.shape[0] - chunk0 * meta.chunks[0])
            chunk_shape = (first,) + meta.shape[1:]
            data = np.frombuffer(self._chunk_path(name, chunk_index).read_bytes(), dtype=meta.dtype).reshape(chunk_shape)
            self._chunk_cache[name] = data
        return np.asarray(data[local]).copy()

    def clear_chunk_cache(self) -> None:
        self._logical_chunk0 = None
        self._chunk_cache.clear()

    def read_full_small(self, name: str) -> np.ndarray:
        cached = self._full_cache.get(name)
        if cached is not None:
            return cached
        meta = self.meta(name)
        if 0 in meta.shape:
            return np.zeros(meta.shape, dtype=meta.dtype)
        path = self._chunk_path(name, tuple(0 for _ in meta.shape))
        chunk = np.frombuffer(path.read_bytes(), dtype=meta.dtype).reshape(meta.shape)
        out = np.asarray(chunk).copy()
        self._full_cache[name] = out
        return out


_READER_REGISTRY: "weakref.WeakValueDictionary[str, PatchZarrReader]" = weakref.WeakValueDictionary()


@lru_cache(maxsize=512)
def _reader(path: str) -> PatchZarrReader:
    reader = PatchZarrReader(Path(path))
    _READER_REGISTRY[str(Path(path).expanduser().resolve())] = reader
    return reader


def _clear_all_zarr_chunk_caches() -> None:
    for reader in list(_READER_REGISTRY.values()):
        reader.clear_chunk_cache()


def _zarr_uri(path: Path, idx: int) -> str:
    return f"zarr://{path.resolve()}#{int(idx)}"


def _parse_zarr_uri(uri: str) -> tuple[Path, int]:
    if not uri.startswith("zarr://") or "#" not in uri:
        raise ValueError(f"Invalid zarr URI: {uri}")
    path, idx = uri[len("zarr://") :].rsplit("#", 1)
    return Path(path), int(idx)


def _build_zarr_record_chunks(records: Sequence[CutoutRecord]) -> list[list[int]]:
    buckets: dict[tuple[str, int], list[int]] = {}
    for dataset_idx, rec in enumerate(records):
        store, sample_idx = _parse_zarr_uri(rec.image_paths[0])
        reader = _reader(str(store))
        chunk_tiles = int(reader.meta("images").chunks[0])
        chunk0 = int(sample_idx) // max(1, chunk_tiles)
        buckets.setdefault((str(store), chunk0), []).append(dataset_idx)
    return [sorted(values) for _key, values in sorted(buckets.items())]


def zarr_passthrough_batch(batch: dict[str, object]) -> dict[str, object]:
    return batch


class ZarrChunkLocalBatchSampler(Sampler[list[int]]):
    """Yield batches that stay inside one patch-store chunk.

    The chunk order is randomized per epoch; optional DDP support partitions
    chunks across ranks so every batch remains chunk-local.
    """

    def __init__(
        self,
        records: Sequence[CutoutRecord],
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
        num_replicas: int = 1,
        rank: int = 0,
        drop_last: bool = False,
        shuffle_within_chunk: bool = True,
        equalize_replicas: bool = False,
    ) -> None:
        self.records = list(records)
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        self.shuffle_within_chunk = bool(shuffle_within_chunk)
        self.equalize_replicas = bool(equalize_replicas)
        self.epoch = 0
        self._chunks = _build_zarr_record_chunks(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rank_chunks_for_rank(self, rank: int) -> list[list[int]]:
        chunks = [list(chunk) for chunk in self._chunks]
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(chunks)
        chunks = chunks[int(rank) :: self.num_replicas]
        if self.shuffle and self.shuffle_within_chunk:
            for chunk in chunks:
                rng.shuffle(chunk)
        return chunks

    def _batch_count(self, chunks: Sequence[Sequence[int]]) -> int:
        total = 0
        for chunk in chunks:
            if self.drop_last:
                total += len(chunk) // self.batch_size
            else:
                total += (len(chunk) + self.batch_size - 1) // self.batch_size
        return total

    def _truncate_to_batches(self, chunks: Sequence[Sequence[int]], target_batches: int) -> list[list[int]]:
        out: list[list[int]] = []
        remaining = max(0, int(target_batches))
        for chunk in chunks:
            if remaining <= 0:
                break
            n_batches = self._batch_count([chunk])
            if n_batches <= remaining:
                out.append(list(chunk))
                remaining -= n_batches
                continue
            keep = remaining * self.batch_size
            if keep > 0:
                out.append(list(chunk[:keep]))
            break
        return out

    def _rank_chunks(self) -> list[list[int]]:
        chunks = self._rank_chunks_for_rank(self.rank)
        if self.equalize_replicas and self.num_replicas > 1:
            target_batches = min(
                self._batch_count(self._rank_chunks_for_rank(replica_rank))
                for replica_rank in range(self.num_replicas)
            )
            chunks = self._truncate_to_batches(chunks, target_batches)
        return chunks

    def __iter__(self) -> Iterator[list[int]]:
        for chunk in self._rank_chunks():
            for start in range(0, len(chunk), self.batch_size):
                batch = chunk[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        total = 0
        for chunk in self._rank_chunks():
            if self.drop_last:
                total += len(chunk) // self.batch_size
            else:
                total += (len(chunk) + self.batch_size - 1) // self.batch_size
        return total

    def max_local_chunk_size(self) -> int:
        chunks = self._rank_chunks()
        return max((len(chunk) for chunk in chunks), default=0)


class ZarrChunkBatchIterableDataset(IterableDataset):
    """Worker-owned Zarr chunk iterator that yields already-collated batches."""

    def __init__(
        self,
        records: Sequence[CutoutRecord],
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
        num_replicas: int = 1,
        rank: int = 0,
        drop_last: bool = False,
        shuffle_within_chunk: bool = True,
        equalize_replicas: bool = False,
        augment: bool = False,
        **dataset_kwargs,
    ) -> None:
        super().__init__()
        self.records = list(records)
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        self.shuffle_within_chunk = bool(shuffle_within_chunk)
        self.equalize_replicas = bool(equalize_replicas)
        self.sample_dataset = ZarrCutoutDataset(self.records, augment=augment, **dataset_kwargs)
        self._chunks = _build_zarr_record_chunks(self.records)
        self._epoch = mp.Value("i", 0)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def _epoch_value(self) -> int:
        with self._epoch.get_lock():
            return int(self._epoch.value)

    def _rank_chunks_for_rank(self, *, epoch: int, rank: int) -> list[list[int]]:
        chunks = [list(chunk) for chunk in self._chunks]
        rng = random.Random(self.seed + int(epoch))
        if self.shuffle:
            rng.shuffle(chunks)
        chunks = chunks[int(rank) :: self.num_replicas]
        if self.shuffle and self.shuffle_within_chunk:
            for chunk in chunks:
                rng.shuffle(chunk)
        return chunks

    def _batch_count(self, chunks: Sequence[Sequence[int]]) -> int:
        total = 0
        for chunk in chunks:
            if self.drop_last:
                total += len(chunk) // self.batch_size
            else:
                total += (len(chunk) + self.batch_size - 1) // self.batch_size
        return total

    def _truncate_to_batches(self, chunks: Sequence[Sequence[int]], target_batches: int) -> list[list[int]]:
        out: list[list[int]] = []
        remaining = max(0, int(target_batches))
        for chunk in chunks:
            if remaining <= 0:
                break
            n_batches = self._batch_count([chunk])
            if n_batches <= remaining:
                out.append(list(chunk))
                remaining -= n_batches
                continue
            keep = remaining * self.batch_size
            if keep > 0:
                out.append(list(chunk[:keep]))
            break
        return out

    def _rank_chunks(self, *, epoch: int) -> list[list[int]]:
        chunks = self._rank_chunks_for_rank(epoch=epoch, rank=self.rank)
        if self.equalize_replicas and self.num_replicas > 1:
            target_batches = min(
                self._batch_count(self._rank_chunks_for_rank(epoch=epoch, rank=replica_rank))
                for replica_rank in range(self.num_replicas)
            )
            chunks = self._truncate_to_batches(chunks, target_batches)
        return chunks

    def _worker_chunks(self, *, epoch: int) -> list[list[int]]:
        chunks = self._rank_chunks(epoch=epoch)
        worker = get_worker_info()
        if worker is None:
            return chunks
        return chunks[int(worker.id) :: max(1, int(worker.num_workers))]

    def __iter__(self) -> Iterator[dict[str, object]]:
        for chunk in self._worker_chunks(epoch=self._epoch_value()):
            _clear_all_zarr_chunk_caches()
            for start in range(0, len(chunk), self.batch_size):
                indices = chunk[start : start + self.batch_size]
                if len(indices) < self.batch_size and self.drop_last:
                    continue
                yield collate_cutouts([self.sample_dataset[int(idx)] for idx in indices])

    def __len__(self) -> int:
        total = 0
        for chunk in self._rank_chunks(epoch=self._epoch_value()):
            if self.drop_last:
                total += len(chunk) // self.batch_size
            else:
                total += (len(chunk) + self.batch_size - 1) // self.batch_size
        return total

    def incomplete_batch_count(self) -> int:
        if self.drop_last:
            return 0
        return sum(1 for chunk in self._rank_chunks(epoch=self._epoch_value()) if len(chunk) % self.batch_size)

    def local_sample_count(self) -> int:
        return sum(len(chunk) for chunk in self._rank_chunks(epoch=self._epoch_value()))

    def max_local_chunk_size(self) -> int:
        chunks = self._rank_chunks(epoch=self._epoch_value())
        return max((len(chunk) for chunk in chunks), default=0)


def discover_zarr_records(
    root: Path,
    *,
    bands: Sequence[str],
    max_records: int | None = None,
) -> list[CutoutRecord]:
    root = Path(root).expanduser().resolve()
    stores = sorted(root.rglob("*.zarr"))
    records: list[CutoutRecord] = []
    wanted_bands = tuple(str(band) for band in bands)
    for store in stores:
        manifest_path = store.parent / f"{store.name}_manifest.json"
        try:
            reader = PatchZarrReader(store)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            if not manifest_path.exists():
                print(f"[zarr] skip incomplete store without manifest: {store} ({exc})", flush=True)
                continue
            raise
        attrs = reader.attrs
        if attrs.get("format") == "cellect_direct_patch_zarr" and not manifest_path.exists():
            print(f"[zarr] skip incomplete direct store without manifest: {store}", flush=True)
            continue
        if bool(attrs.get("image_level_training", False)):
            continue
        store_bands = tuple(str(b) for b in attrs.get("bands", []))
        if wanted_bands and store_bands and tuple(wanted_bands) != store_bands:
            continue
        n = int(reader.meta("images").shape[0])
        tile_x0 = reader.read_full_small("tile_x0").astype(np.int32, copy=False)
        tile_y0 = reader.read_full_small("tile_y0").astype(np.int32, copy=False)
        tile_names = _decode_fixed_utf8(reader.read_full_small("tile_name"))
        dataset_sources = _decode_fixed_utf8(reader.read_full_small("dataset_source"))
        tract = str(attrs.get("tract", ""))
        patch = str(attrs.get("patch", store.stem))
        try:
            rel_root = str(store.parent.relative_to(root))
        except ValueError:
            rel_root = ""
        for i in range(n):
            dataset_source = dataset_sources[i] if i < len(dataset_sources) else str(attrs.get("dataset_source", "zarr"))
            tile_name = tile_names[i] if i < len(tile_names) else f"sample_{i:06d}"
            name = f"{tract}/{dataset_source}/{patch}/{tile_name}" if tract else f"{dataset_source}/{patch}/{tile_name}"
            records.append(
                CutoutRecord(
                    name=name,
                    image_paths=(_zarr_uri(store, i),),
                    meas_path="",
                    x0=int(tile_x0[i]),
                    y0=int(tile_y0[i]),
                    tile_name=tile_name,
                    tract=tract,
                    patch=patch,
                    relative_root=rel_root,
                    dataset_source=dataset_source or "zarr",
                )
            )
            if max_records is not None and len(records) >= int(max_records):
                return records
    if not records:
        raise RuntimeError(f"No Zarr records found under {root}")
    return records


def discover_zarr_image_records(
    root: Path,
    *,
    bands: Sequence[str],
    max_records: int | None = None,
) -> list[CutoutRecord]:
    """Discover single-band image-level Zarr samples for SAM detector training."""

    root = Path(root).expanduser().resolve()
    stores = sorted(root.rglob("*.zarr"))
    records: list[CutoutRecord] = []
    wanted_bands = {str(band) for band in bands}
    for store in stores:
        manifest_path = store.parent / f"{store.name}_manifest.json"
        try:
            reader = PatchZarrReader(store)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            if not manifest_path.exists():
                print(f"[zarr] skip incomplete store without manifest: {store} ({exc})", flush=True)
                continue
            raise
        attrs = reader.attrs
        if attrs.get("format") == "cellect_direct_patch_zarr" and not manifest_path.exists():
            print(f"[zarr] skip incomplete direct store without manifest: {store}", flush=True)
            continue
        if not bool(attrs.get("image_level_training", False)):
            continue
        store_bands = tuple(str(b) for b in attrs.get("bands", []))
        if len(store_bands) != 1:
            continue
        band = store_bands[0]
        if wanted_bands and band not in wanted_bands:
            continue
        n = int(reader.meta("images").shape[0])
        tile_x0 = reader.read_full_small("tile_x0").astype(np.int32, copy=False)
        tile_y0 = reader.read_full_small("tile_y0").astype(np.int32, copy=False)
        tile_names = _decode_fixed_utf8(reader.read_full_small("tile_name"))
        groups = _decode_fixed_utf8(reader.read_full_small("group"))
        dataset_sources = _decode_fixed_utf8(reader.read_full_small("dataset_source"))
        tract = str(attrs.get("tract", ""))
        patch = str(attrs.get("patch", store.stem))
        try:
            rel_root = str(store.parent.relative_to(root))
        except ValueError:
            rel_root = ""
        for i in range(n):
            dataset_source = dataset_sources[i] if i < len(dataset_sources) else str(attrs.get("dataset_source", "zarr"))
            group = groups[i] if i < len(groups) else ""
            tile_name = tile_names[i] if i < len(tile_names) else f"sample_{i:06d}"
            parts = [value for value in (tract, dataset_source, patch, group, band, tile_name) if value]
            records.append(
                CutoutRecord(
                    name="/".join(parts),
                    image_paths=(_zarr_uri(store, i),),
                    meas_path="",
                    x0=int(tile_x0[i]),
                    y0=int(tile_y0[i]),
                    tile_name=tile_name,
                    tract=tract,
                    patch=patch,
                    relative_root=rel_root,
                    dataset_source=dataset_source or "zarr",
                )
            )
            if max_records is not None and len(records) >= int(max_records):
                return records
    if not records:
        raise RuntimeError(f"No image-level Zarr records found under {root}")
    return records


def _target_defaults_from_pu(
    *,
    confidence: Tensor,
    shape: Tensor,
    shape_weight: Tensor,
    conf_weight: Tensor,
    pu: Tensor,
) -> dict[str, Tensor]:
    clean = pu == 1
    center = pu == 2
    ignore = pu == 3
    background = pu == 4
    strict_center = pu == 5
    bright = pu == 6
    source_union = clean | center | ignore
    return {
        "seg": clean.to(dtype=torch.long),
        "confidence": confidence.to(dtype=torch.long),
        "shape": shape.to(dtype=torch.float32),
        "shape_weight": shape_weight.to(dtype=torch.float32),
        "confidence_weight": conf_weight.to(dtype=torch.float32),
        "seg_loss_weight": torch.where(
            clean | background | strict_center | bright,
            conf_weight.to(dtype=torch.float32),
            torch.zeros_like(conf_weight, dtype=torch.float32),
        ),
        "clean_mask": clean.to(dtype=torch.uint8),
        "center_only_mask": center.to(dtype=torch.uint8),
        "ignore_mask": ignore.to(dtype=torch.uint8),
        "strict_center_only_mask": strict_center.to(dtype=torch.uint8),
        "strict_ignore_mask": torch.zeros_like(pu, dtype=torch.uint8),
        "source_union_mask": source_union.to(dtype=torch.uint8),
        "background_mask": background.to(dtype=torch.uint8),
        "bright_mask": bright.to(dtype=torch.uint8),
        "pu_class_mask": pu.to(dtype=torch.uint8),
        "pseudo_mask": torch.zeros_like(pu, dtype=torch.uint8),
    }


class ZarrCutoutDataset(Dataset):
    def __init__(self, records: Sequence[CutoutRecord], *, augment: bool = False, **_unused) -> None:
        self.records = list(records)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, object]:
        rec = self.records[idx]
        store, sample_idx = _parse_zarr_uri(rec.image_paths[0])
        reader = _reader(str(store))
        image = torch.from_numpy(reader.read_first_axis("images", sample_idx).astype(np.float32, copy=False))
        band_conf = torch.from_numpy(reader.read_first_axis("band_confidence", sample_idx).astype(np.int64, copy=False))
        band_conf_weight = torch.from_numpy(reader.read_first_axis("band_conf_weight", sample_idx).astype(np.float32, copy=False))
        band_shape = torch.from_numpy(reader.read_first_axis("band_shape", sample_idx).astype(np.float32, copy=False))
        band_shape_weight = torch.from_numpy(reader.read_first_axis("band_shape_weight", sample_idx).astype(np.float32, copy=False))
        band_pu = torch.from_numpy(reader.read_first_axis("band_pu_class_mask", sample_idx).astype(np.uint8, copy=False))

        band_targets = [
            _target_defaults_from_pu(
                confidence=band_conf[b],
                shape=band_shape[b],
                shape_weight=band_shape_weight[b],
                conf_weight=band_conf_weight[b],
                pu=band_pu[b],
            )
            for b in range(int(image.shape[0]))
        ]
        offsets = reader.read_full_small("source_offsets").astype(np.int64, copy=False)
        centers_flat = reader.read_full_small("source_centers").astype(np.float32, copy=False)
        ids_flat = reader.read_full_small("source_ids").astype(np.int64, copy=False)
        band_centers = []
        band_ids = []
        for b in range(int(image.shape[0])):
            s0, s1 = int(offsets[sample_idx, b]), int(offsets[sample_idx, b + 1])
            band_centers.append(torch.from_numpy(centers_flat[s0:s1]))
            band_ids.append(torch.from_numpy(ids_flat[s0:s1]))
        centers = band_centers[0] if band_centers else torch.empty((0, 2), dtype=torch.float32)
        ids = band_ids[0] if band_ids else torch.empty((0,), dtype=torch.long)
        empty_centers = [torch.empty((0, 2), dtype=torch.float32) for _ in band_targets]
        empty_ids = [torch.empty((0,), dtype=torch.long) for _ in band_targets]
        band_strict_center_only_centers = list(empty_centers)
        band_strict_center_only_ids = list(empty_ids)
        if reader.has_array("strict_center_only_offsets"):
            strict_offsets = reader.read_full_small("strict_center_only_offsets").astype(np.int64, copy=False)
            strict_centers_flat = reader.read_full_small("strict_center_only_centers").astype(np.float32, copy=False)
            strict_ids_flat = (
                reader.read_full_small("strict_center_only_ids").astype(np.int64, copy=False)
                if reader.has_array("strict_center_only_ids")
                else np.full((len(strict_centers_flat),), -1, dtype=np.int64)
            )
            band_strict_center_only_centers = []
            band_strict_center_only_ids = []
            for b in range(int(image.shape[0])):
                s0, s1 = int(strict_offsets[sample_idx, b]), int(strict_offsets[sample_idx, b + 1])
                band_strict_center_only_centers.append(torch.from_numpy(strict_centers_flat[s0:s1]))
                band_strict_center_only_ids.append(torch.from_numpy(strict_ids_flat[s0:s1]))

        band_shape_source_centers = []
        band_shape_source_values = []
        band_shape_source_classes = []
        band_shape_source_ids = []
        if reader.has_array("shape_source_offsets"):
            shape_offsets = reader.read_full_small("shape_source_offsets").astype(np.int64, copy=False)
            shape_centers_flat = reader.read_full_small("shape_source_centers").astype(np.float32, copy=False)
            shape_values_flat = reader.read_full_small("shape_source_values").astype(np.float32, copy=False)
            shape_classes_flat = reader.read_full_small("shape_source_classes").astype(np.uint8, copy=False)
            shape_ids_flat = reader.read_full_small("shape_source_ids").astype(np.int64, copy=False)
            for b in range(int(image.shape[0])):
                s0, s1 = int(shape_offsets[sample_idx, b]), int(shape_offsets[sample_idx, b + 1])
                band_shape_source_centers.append(torch.from_numpy(shape_centers_flat[s0:s1]))
                band_shape_source_values.append(torch.from_numpy(shape_values_flat[s0:s1]))
                band_shape_source_classes.append(torch.from_numpy(shape_classes_flat[s0:s1]))
                band_shape_source_ids.append(torch.from_numpy(shape_ids_flat[s0:s1]))
        else:
            # Backward compatibility for stores created before source-level
            # shape metadata was added. Only clean centers are recoverable.
            h, w = int(image.shape[-2]), int(image.shape[-1])
            for b, source_centers in enumerate(band_centers):
                rounded = source_centers.round().to(dtype=torch.long)
                valid = (
                    torch.isfinite(source_centers).all(dim=1)
                    & (source_centers[:, 0] >= 0.0)
                    & (source_centers[:, 0] < float(w))
                    & (source_centers[:, 1] >= 0.0)
                    & (source_centers[:, 1] < float(h))
                )
                source_centers = source_centers[valid]
                rounded = rounded[valid]
                if rounded.numel():
                    rounded[:, 0].clamp_(0, w - 1)
                    rounded[:, 1].clamp_(0, h - 1)
                source_ids = band_ids[b][valid]
                values = (
                    band_shape[b, :, rounded[:, 1], rounded[:, 0]].transpose(0, 1).contiguous()
                    if rounded.numel()
                    else torch.empty((0, 3), dtype=torch.float32)
                )
                band_shape_source_centers.append(source_centers)
                band_shape_source_values.append(values)
                band_shape_source_classes.append(torch.ones((len(source_centers),), dtype=torch.uint8))
                band_shape_source_ids.append(source_ids)

        if self.augment and random.random() < 0.5:
            width = int(image.shape[-1])
            image = torch.flip(image, dims=(-1,))
            flipped_targets = []
            for target in band_targets:
                target = {key: value.clone() for key, value in target.items()}
                for key, value in list(target.items()):
                    if value.ndim >= 2:
                        target[key] = torch.flip(value, dims=(-1,))
                target["shape"][2] = -target["shape"][2]
                flipped_targets.append(target)
            band_targets = flipped_targets
            band_centers = [c.clone() for c in band_centers]
            for c in band_centers:
                if c.numel():
                    c[:, 0] = float(width - 1) - c[:, 0]
            centers = band_centers[0] if band_centers else centers
            band_shape_source_centers = [c.clone() for c in band_shape_source_centers]
            band_shape_source_values = [v.clone() for v in band_shape_source_values]
            for source_centers, source_values in zip(band_shape_source_centers, band_shape_source_values):
                if source_centers.numel():
                    source_centers[:, 0] = float(width - 1) - source_centers[:, 0]
                    source_values[:, 2] = -source_values[:, 2]

        primary = band_targets[0]
        shape_source_centers = band_shape_source_centers[0]
        shape_source_values = band_shape_source_values[0]
        shape_source_classes = band_shape_source_classes[0]
        shape_source_ids = band_shape_source_ids[0]
        return {
            "image": image,
            "seg": primary["seg"],
            "confidence": primary["confidence"],
            "shape": primary["shape"],
            "shape_weight": primary["shape_weight"],
            "confidence_weight": primary["confidence_weight"],
            "seg_loss_weight": primary["seg_loss_weight"],
            "clean_mask": primary["clean_mask"],
            "center_only_mask": primary["center_only_mask"],
            "ignore_mask": primary["ignore_mask"],
            "strict_center_only_mask": primary["strict_center_only_mask"],
            "strict_ignore_mask": primary["strict_ignore_mask"],
            "source_union_mask": primary["source_union_mask"],
            "background_mask": primary["background_mask"],
            "bright_mask": primary["bright_mask"],
            "pu_class_mask": primary["pu_class_mask"],
            "pseudo_mask": primary["pseudo_mask"],
            "band_seg": torch.stack([target["seg"] for target in band_targets]),
            "band_confidence": torch.stack([target["confidence"] for target in band_targets]),
            "band_shape": torch.stack([target["shape"] for target in band_targets]),
            "band_shape_weight": torch.stack([target["shape_weight"] for target in band_targets]),
            "band_confidence_weight": torch.stack([target["confidence_weight"] for target in band_targets]),
            "band_seg_loss_weight": torch.stack([target["seg_loss_weight"] for target in band_targets]),
            "band_clean_mask": torch.stack([target["clean_mask"] for target in band_targets]),
            "band_center_only_mask": torch.stack([target["center_only_mask"] for target in band_targets]),
            "band_ignore_mask": torch.stack([target["ignore_mask"] for target in band_targets]),
            "band_strict_center_only_mask": torch.stack([target["strict_center_only_mask"] for target in band_targets]),
            "band_strict_ignore_mask": torch.stack([target["strict_ignore_mask"] for target in band_targets]),
            "band_source_union_mask": torch.stack([target["source_union_mask"] for target in band_targets]),
            "band_background_mask": torch.stack([target["background_mask"] for target in band_targets]),
            "band_bright_mask": torch.stack([target["bright_mask"] for target in band_targets]),
            "band_pu_class_mask": torch.stack([target["pu_class_mask"] for target in band_targets]),
            "band_pseudo_mask": torch.stack([target["pseudo_mask"] for target in band_targets]),
            "centers": centers,
            "ids": ids,
            "ignore_centers": torch.empty((0, 2), dtype=torch.float32),
            "strict_center_only_centers": band_strict_center_only_centers[0] if band_strict_center_only_centers else torch.empty((0, 2), dtype=torch.float32),
            "strict_ignore_centers": torch.empty((0, 2), dtype=torch.float32),
            "band_centers": band_centers,
            "band_ids": band_ids,
            "band_ignore_centers": list(empty_centers),
            "band_strict_center_only_centers": band_strict_center_only_centers,
            "band_strict_ignore_centers": list(empty_centers),
            "band_strict_center_only_ids": band_strict_center_only_ids,
            "band_rejected_ids": list(empty_ids),
            "shape_source_centers": shape_source_centers,
            "shape_source_values": shape_source_values,
            "shape_source_classes": shape_source_classes,
            "shape_source_ids": shape_source_ids,
            "band_shape_source_centers": band_shape_source_centers,
            "band_shape_source_values": band_shape_source_values,
            "band_shape_source_classes": band_shape_source_classes,
            "band_shape_source_ids": band_shape_source_ids,
            "name": rec.name,
            "tile_name": rec.tile_name,
            "tract": rec.tract,
            "patch": rec.patch,
            "relative_root": rec.relative_root,
            "dataset_source": rec.dataset_source,
            "x0": rec.x0,
            "y0": rec.y0,
            "image_paths": rec.image_paths,
        }
