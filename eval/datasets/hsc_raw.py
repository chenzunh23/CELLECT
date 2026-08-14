from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from .base import FrameRef, TileRow, patch_sort_key


DEFAULT_HSC_RAW_ROOT = Path("/data/zc/Subaru/data/hsctile/pack_full_9813_256/9813")
DEFAULT_HSC_RAW_BANDS = ("HSC-I", "HSC-G", "HSC-Y", "NB1010", "NB0816")


def read_tiles_csv(path: Path) -> list[TileRow]:
    rows: list[TileRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                TileRow(
                    tile_index=int(row["tile_index"]),
                    tile_id=row["tile_id"],
                    x0=int(row["x0"]),
                    y0=int(row["y0"]),
                    x1=int(row["x1"]),
                    y1=int(row["y1"]),
                )
            )
    return rows


class HscRawAccess:
    dataset = "hsc_raw"
    display_name = "HSC raw tiles"
    default_tract = "9813"
    default_bands = DEFAULT_HSC_RAW_BANDS
    tile_size = 256

    def __init__(self, root: Path, tract: str = "9813") -> None:
        self.requested_root = Path(root).expanduser().resolve()
        self.tract = str(tract)
        if any((self.requested_root / band).is_dir() for band in DEFAULT_HSC_RAW_BANDS):
            self.root = self.requested_root
        elif (self.requested_root / self.tract).is_dir():
            self.root = self.requested_root / self.tract
        else:
            self.root = self.requested_root
        self._groups: dict[str, Any] = {}
        self._tiles: dict[str, list[TileRow]] = {}

    def pack_dir(self, band: str, patch: str) -> Path:
        path = self.root / band / patch / "256"
        if not (path / "pack.zarr").is_dir():
            raise FileNotFoundError(f"pack not found: {path / 'pack.zarr'}")
        return path

    def group(self, band: str, patch: str):
        path = self.pack_dir(band, patch) / "pack.zarr"
        key = str(path)
        if key not in self._groups:
            self._groups[key] = zarr.open_group(key, mode="r")
        return self._groups[key]

    def tiles(self, band: str, patch: str) -> list[TileRow]:
        path = self.pack_dir(band, patch) / "tiles.csv"
        key = str(path)
        if key not in self._tiles:
            self._tiles[key] = read_tiles_csv(path)
        return self._tiles[key]

    def tile_by_id(self, band: str, patch: str) -> dict[str, TileRow]:
        return {row.tile_id: row for row in self.tiles(band, patch)}

    def frame_span(self, band: str, patch: str, tile_index: int) -> tuple[int, int]:
        group = self.group(band, patch)
        start = int(group["tile_offsets"][tile_index])
        length = int(group["tile_lengths"][tile_index])
        return start, length

    def readable_frame_stop(self, band: str, patch: str) -> int:
        images = self.group(band, patch)["images"]
        chunk0 = int(images.chunks[0])
        return int(images.shape[0] // chunk0 * chunk0)

    def safe_scalar(self, band: str, patch: str, array_name: str, index: int) -> Any:
        try:
            value = self.group(band, patch)[array_name][index]
        except Exception:
            return None
        if hasattr(value, "item"):
            return value.item()
        return value

    def read_frame(self, ref: FrameRef) -> np.ndarray:
        group = self.group(ref.band, ref.patch)
        return np.asarray(group["images"][ref.frame_index], dtype=np.float32)

    def available_bands(self) -> list[str]:
        if not self.root.is_dir():
            return []
        out = []
        for path in sorted(self.root.iterdir()):
            if path.is_dir() and any((path / patch / "256" / "pack.zarr").is_dir() for patch in self._safe_iterdir_names(path)):
                out.append(path.name)
        return out

    def available_patches(self, bands: list[str] | None = None) -> list[str]:
        bands = bands or self.available_bands()
        patches: set[str] = set()
        for band in bands:
            band_dir = self.root / band
            if not band_dir.is_dir():
                continue
            for patch in self._safe_iterdir_names(band_dir):
                if (band_dir / patch / "256" / "pack.zarr").is_dir():
                    patches.add(patch)
        return sorted(patches, key=patch_sort_key)

    def valid_tiles_for_band(self, band: str, patch: str) -> set[str]:
        group = self.group(band, patch)
        readable_stop = self.readable_frame_stop(band, patch)
        valid = set()
        for tile in self.tiles(band, patch):
            start = int(group["tile_offsets"][tile.tile_index])
            length = int(group["tile_lengths"][tile.tile_index])
            if min(start + length, readable_stop) > start:
                valid.add(tile.tile_id)
        return valid

    def choose_tiles(self, patch: str, bands: list[str], *, n_tiles: int | None, all_tiles: bool, seed: int, mode: str = "default") -> list[str]:
        sets = []
        usable_bands = []
        for band in bands:
            try:
                valid = self.valid_tiles_for_band(band, patch)
            except Exception as exc:
                print(f"WARNING: skip missing/unreadable band {band} for patch {patch}: {exc}", flush=True)
                continue
            if not valid:
                print(f"WARNING: skip empty band {band} for patch {patch}", flush=True)
                continue
            sets.append(valid)
            usable_bands.append(band)
        common = sorted(set.intersection(*sets), key=lambda text: tuple(int(v) for v in re.findall(r"\d+", text))) if sets else []
        skipped = [band for band in bands if band not in usable_bands]
        if skipped:
            print(f"WARNING: {patch}: dropped unavailable bands for tile intersection: {', '.join(skipped)}", flush=True)
        print(f"[DEBUG] {patch}: found {len(common)} common non-empty tile IDs across usable bands {usable_bands}", flush=True)
        if not common:
            print(f"WARNING: no common non-empty 256x256 tile IDs for patch {patch} across requested bands {bands}", flush=True)
            return []
        if all_tiles or n_tiles is None:
            return common
        count = min(max(1, int(n_tiles)), len(common))
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(np.asarray(common, dtype=object), size=count, replace=False).tolist())

    def usable_frame_count(self, band: str, patch: str, tile_id: str) -> int:
        tile = self.tile_by_id(band, patch)[tile_id]
        start, length = self.frame_span(band, patch, tile.tile_index)
        readable_stop = self.readable_frame_stop(band, patch)
        return max(0, min(length, readable_stop - start))

    def tile_slot_count(
        self,
        patch: str,
        tile_id: str,
        bands: list[str],
        *,
        frames_per_tile: int,
        visit: int | None,
    ) -> int:
        counts = []
        for band in bands:
            try:
                count = self.usable_frame_count(band, patch, tile_id)
            except Exception as exc:
                print(f"WARNING: skip missing/unreadable band {band} for {patch} {tile_id}: {exc}", flush=True)
                continue
            if count > 0:
                counts.append(count)
        if not counts or min(counts) <= 0:
            return 0
        if visit is not None:
            return 1
        return max(0, min(int(frames_per_tile), min(counts)))

    def choose_frame(
        self,
        band: str,
        patch: str,
        tile: TileRow,
        *,
        frame_slot: int,
        frame_rank: int,
        frames_per_tile: int,
        visit: int | None,
        strict_visit: bool,
    ) -> tuple[int, int, int]:
        start, length = self.frame_span(band, patch, tile.tile_index)
        readable_stop = self.readable_frame_stop(band, patch)
        usable_length = max(0, min(length, readable_stop - start))
        if usable_length <= 0:
            raise RuntimeError(f"tile {tile.tile_id} has no readable frames for {band} {patch}")
        if visit is not None:
            try:
                visits = np.asarray(self.group(band, patch)["visits"][start : start + usable_length])
                where = np.where(visits == int(visit))[0]
                if where.size:
                    rank = int(where[min(frame_slot, where.size - 1)])
                    return start + rank, rank, length
                if strict_visit:
                    raise RuntimeError(f"visit {visit} is not present in {band} {patch} {tile.tile_id}")
            except Exception as exc:
                if strict_visit:
                    raise
                print(f"WARNING: failed to use visit={visit} for {band} {patch} {tile.tile_id}; using frame rank: {exc}", flush=True)
        rank = int(frame_rank)
        if rank < 0:
            rank = usable_length + rank
        if frames_per_tile > 1 and visit is None:
            rank = frame_slot
        rank = max(0, min(rank, usable_length - 1))
        return start + rank, rank, length

    def make_ref(
        self,
        *,
        token: str,
        patch: str,
        band: str,
        tile_id: str,
        frame_slot: int,
        frame_rank: int,
        frames_per_tile: int,
        visit: int | None,
        strict_visit: bool,
    ) -> FrameRef:
        tile = self.tile_by_id(band, patch)[tile_id]
        frame_index, actual_rank, length = self.choose_frame(
            band,
            patch,
            tile,
            frame_slot=frame_slot,
            frame_rank=frame_rank,
            frames_per_tile=frames_per_tile,
            visit=visit,
            strict_visit=strict_visit,
        )
        actual_visit = self.safe_scalar(band, patch, "visits", frame_index)
        weight = self.safe_scalar(band, patch, "weights", frame_index)
        scale = self.safe_scalar(band, patch, "scales", frame_index)
        return FrameRef(
            token=token,
            root=str(self.root),
            tract=self.tract,
            patch=patch,
            band=band,
            pack_path=str(self.pack_dir(band, patch) / "pack.zarr"),
            tile_index=tile.tile_index,
            tile_id=tile.tile_id,
            x0=tile.x0,
            y0=tile.y0,
            x1=tile.x1,
            y1=tile.y1,
            frame_slot=frame_slot,
            frame_rank=actual_rank,
            frame_index=frame_index,
            tile_length=length,
            visit=int(actual_visit) if actual_visit is not None else None,
            weight=float(weight) if weight is not None else None,
            scale=float(scale) if scale is not None else None,
            dataset=self.dataset,
        )

    @staticmethod
    def _safe_iterdir_names(path: Path) -> list[str]:
        try:
            return [child.name for child in path.iterdir() if child.is_dir()]
        except Exception:
            return []
