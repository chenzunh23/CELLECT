from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from astro_train_zarr_data import PatchZarrReader

from eval.eval_utils import crop_or_pad, crop_origin_for_image, read_fits_image

from .base import FrameRef, TileRow, patch_sort_key


DEFAULT_HSC_IMAGE_ROOT = Path("/data/czh23/direct_zarr_v4_anscombe")
DEFAULT_HSC_WEIGHT_ROOT = Path("/data/czh23/Subaru_products/warp_weights")
DEFAULT_HSC_COADD_FITS_ROOT = Path("/data/czh23/Subaru_products/half_coadd")
DEFAULT_HSC_NOISY_FITS_ROOT = Path("/data/czh23/Subaru_products/noisy")
DEFAULT_HSC_DENOISED_FITS_ROOT = Path("/data/czh23/denoised_fits")
DEFAULT_HSC_OFFICIAL_COADD_FITS_ROOT = Path("/data/shared/Subaru")
DEFAULT_HSC_OFFICIAL_WEIGHT_ROOT = Path("/data/czh23/2026-06-21_171607_hsc_metadata_warp_n2n_epoch006_full-all-warp-weights")
DEFAULT_HSC_IMAGE_BANDS = (
    "HSC-G",
    "HSC-R",
    "HSC-I",
    "HSC-Z",
    "HSC-Y",
    "NB0387",
    "NB0816",
    "NB0921",
    "NB1010",
)
SOURCE_ORDER = ("coadd", "denoised", "noisy")


def _decode_fixed_utf8(arr: np.ndarray) -> list[str]:
    out = []
    for row in np.asarray(arr, dtype=np.uint8):
        out.append(bytes(row).split(b"\0", 1)[0].decode("utf-8", errors="replace"))
    return out


def _group_sort_key(value: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", str(value))
    return (int(match.group(1)) if match else 10**9, str(value))


def _group_label(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return "0"
    match = re.search(r"(\d+)", text)
    if match:
        return str(int(match.group(1)))
    return text


def _tile_id_from_name(name: str, x0: int, y0: int, index: int) -> str:
    match = re.search(r"(?:^|_)r(\d+)_c(\d+)(?:_|$)", str(name))
    if match:
        return f"x{int(match.group(2)):03d}_y{int(match.group(1)):03d}"
    return f"x{int(x0):05d}_y{int(y0):05d}_{int(index):04d}"


@dataclass(frozen=True)
class _Store:
    source: str
    band: str
    patch: str
    group: str
    path: Path


class HscImageAccess:
    dataset = "hsc_image"
    display_name = "HSC coadd/noisy/denoised"
    default_tract = "9813"
    default_bands = DEFAULT_HSC_IMAGE_BANDS
    tile_size = 512

    def __init__(
        self,
        root: Path,
        tract: str = "9813",
        weight_root: Path | None = DEFAULT_HSC_WEIGHT_ROOT,
        coadd_weight_root: Path | None = None,
        variant_weight_root: Path | None = None,
        coadd_fits_root: Path = DEFAULT_HSC_COADD_FITS_ROOT,
        noisy_fits_root: Path = DEFAULT_HSC_NOISY_FITS_ROOT,
        denoised_fits_root: Path = DEFAULT_HSC_DENOISED_FITS_ROOT,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.tract = str(tract)
        self.weight_root = Path(weight_root).expanduser().resolve() if weight_root is not None else None
        self.coadd_weight_root = Path(coadd_weight_root).expanduser().resolve() if coadd_weight_root is not None else self.weight_root
        self.variant_weight_root = Path(variant_weight_root).expanduser().resolve() if variant_weight_root is not None else self.weight_root
        self.coadd_fits_root = Path(coadd_fits_root).expanduser().resolve()
        self.noisy_fits_root = Path(noisy_fits_root).expanduser().resolve()
        self.denoised_fits_root = Path(denoised_fits_root).expanduser().resolve()
        self._index: dict[tuple[str, str, str, str], _Store] | None = None
        self._readers: dict[str, PatchZarrReader] = {}
        self._tiles: dict[tuple[str, str], list[TileRow]] = {}
        self._sample_by_tile: dict[tuple[str, str, str, str], dict[str, int]] = {}
        self._browser_slots: dict[tuple[str, int, str], list[tuple[str, str]]] = {}
        self._weight_loaded: set[tuple[str, str]] = set()
        self._weights: dict[tuple[str, str, str, str], tuple[float, int]] = {}

    def _scan(self) -> dict[tuple[str, str, str, str], _Store]:
        if self._index is not None:
            return self._index
        index: dict[tuple[str, str, str, str], _Store] = {}
        base = self.root / "image_level"
        if not base.is_dir():
            self._index = index
            return index
        for path in base.glob("*/*/*.zarr"):
            rel = path.relative_to(base).parts
            if len(rel) != 3:
                continue
            source, band, filename = rel
            if source not in SOURCE_ORDER:
                continue
            stem = Path(filename).stem
            patch = stem.split("__", 1)[0]
            group = ""
            if "__" in stem:
                group = _group_label(stem.split("__", 1)[1])
            try:
                attrs = PatchZarrReader(path).attrs
                group = _group_label(attrs.get("group") or group) if source != "coadd" else ""
                patch = str(attrs.get("patch") or patch)
            except Exception:
                pass
            index[(source, band, patch, group)] = _Store(source, band, patch, group, path)
        self._index = index
        return index

    def _reader(self, store: _Store) -> PatchZarrReader:
        key = str(store.path)
        if key not in self._readers:
            self._readers[key] = PatchZarrReader(store.path)
        return self._readers[key]

    def _stores(self, *, band: str | None = None, patch: str | None = None, source: str | None = None) -> list[_Store]:
        rows = []
        for (src, b, p, _group), store in self._scan().items():
            if source is not None and src != source:
                continue
            if band is not None and b != band:
                continue
            if patch is not None and p != patch:
                continue
            rows.append(store)
        return sorted(rows, key=lambda item: (SOURCE_ORDER.index(item.source), _group_sort_key(item.group), item.band, patch_sort_key(item.patch)))

    def available_bands(self) -> list[str]:
        bands = {store.band for store in self._stores() if store.band in DEFAULT_HSC_IMAGE_BANDS}
        return [band for band in DEFAULT_HSC_IMAGE_BANDS if band in bands]

    def available_patches(self, bands: list[str] | None = None) -> list[str]:
        wanted = set(bands or self.available_bands())
        patches = {store.patch for store in self._stores() if store.band in wanted}
        return sorted(patches, key=patch_sort_key)

    def available_variant_patches(self, bands: list[str] | None = None) -> list[str]:
        wanted = set(bands or self.available_bands())
        patches = {
            store.patch
            for store in self._stores()
            if store.band in wanted and store.source in {"noisy", "denoised"}
        }
        return sorted(patches, key=patch_sort_key)

    def _store_for_slot(self, band: str, patch: str, source: str, group: str) -> _Store:
        key = (source, band, patch, "" if source == "coadd" else _group_label(group))
        store = self._scan().get(key)
        if store is None:
            raise FileNotFoundError(f"missing {source} zarr for {band} {patch} {group}")
        return store

    def _resolve_weight_csv(self, root: Path | None, band: str, patch: str) -> Path | None:
        if root is None:
            return None
        candidates = [
            root / str(band) / str(patch) / "weights.csv",
            root / self.tract / str(band) / str(patch) / "weights.csv",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        if root.exists():
            for tract_dir in sorted(path for path in root.iterdir() if path.is_dir()):
                candidate = tract_dir / str(band) / str(patch) / "weights.csv"
                if candidate.is_file():
                    return candidate
        return None

    def _load_coadd_weights(self, band: str, patch: str) -> None:
        path = self._resolve_weight_csv(self.coadd_weight_root, band, patch)
        if path is None:
            return
        total = 0.0
        count = 0
        try:
            with path.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    collection = str(row.get("product_collection", "")).strip()
                    if collection and collection not in {"half_coadd", "coadd"}:
                        continue
                    try:
                        weight = float(row.get("weight", "nan"))
                    except Exception:
                        continue
                    if np.isfinite(weight) and weight > 0.0:
                        total += weight
                        count += 1
        except Exception:
            return
        if count > 0:
            self._weights[("coadd", band, patch, "")] = (float(total), int(count))

    def _load_variant_weights(self, band: str, patch: str) -> None:
        path = self._resolve_weight_csv(self.variant_weight_root, band, patch)
        if path is None:
            return
        sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        try:
            with path.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    collection = str(row.get("product_collection", "")).strip()
                    if collection and collection != "noisy":
                        continue
                    if not collection and "group_num" not in row:
                        continue
                    try:
                        weight = float(row.get("weight", "nan"))
                    except Exception:
                        continue
                    if np.isfinite(weight) and weight > 0.0:
                        group = _group_label(row.get("group_num"))
                        sums[group][0] += weight
                        sums[group][1] += 1.0
        except Exception:
            return
        for group, (total, count) in sums.items():
            if count <= 0:
                continue
            self._weights[("noisy", band, patch, group)] = (float(total), int(count))
            self._weights[("denoised", band, patch, group)] = (float(total), int(count))

    def _load_weights(self, band: str, patch: str) -> None:
        key = (band, patch)
        if key in self._weight_loaded:
            return
        self._weight_loaded.add(key)
        self._load_coadd_weights(band, patch)
        self._load_variant_weights(band, patch)

    def weight_for(self, source: str, band: str, patch: str, group: str) -> float | None:
        self._load_weights(band, patch)
        key = (source, band, patch, "" if source == "coadd" else _group_label(group))
        value = self._weights.get(key)
        return None if value is None else float(value[0])

    def _groups_for_patch(self, patch: str, bands: list[str]) -> list[str]:
        groups = {
            store.group
            for store in self._stores(patch=patch)
            if store.band in bands and store.source in {"noisy", "denoised"} and store.group
        }
        return sorted(groups, key=_group_sort_key)

    def _slot_specs(self, patch: str, bands: list[str], frames_per_tile: int) -> list[tuple[str, str]]:
        groups = self._groups_for_patch(patch, bands)
        if groups:
            groups = groups[: max(1, int(frames_per_tile))]
        else:
            groups = ["0"]
        specs: list[tuple[str, str]] = []
        if any(self._scan().get(("coadd", band, patch, "")) is not None for band in bands):
            specs.append(("coadd", ""))
        for group in groups:
            for source in SOURCE_ORDER:
                if source == "coadd":
                    continue
                if any(self._scan().get((source, band, patch, group)) is not None for band in bands):
                    specs.append((source, group))
        return specs

    def _tile_table(self, band: str, patch: str) -> list[TileRow]:
        key = (band, patch)
        cached = self._tiles.get(key)
        if cached is not None:
            return cached
        stores = self._stores(band=band, patch=patch)
        if not stores:
            raise FileNotFoundError(f"no image-level zarr for {band} {patch}")
        store = stores[0]
        reader = self._reader(store)
        x0 = reader.read_full_small("tile_x0").astype(np.int32, copy=False)
        y0 = reader.read_full_small("tile_y0").astype(np.int32, copy=False)
        names = _decode_fixed_utf8(reader.read_full_small("tile_name")) if reader.has_array("tile_name") else ["" for _ in range(len(x0))]
        rows = [
            TileRow(
                tile_index=i,
                tile_id=_tile_id_from_name(names[i] if i < len(names) else "", int(x0[i]), int(y0[i]), i),
                x0=int(x0[i]),
                y0=int(y0[i]),
                x1=int(x0[i]) + self.tile_size,
                y1=int(y0[i]) + self.tile_size,
            )
            for i in range(len(x0))
        ]
        self._tiles[key] = rows
        return rows

    def _sample_index(self, store: _Store, tile_id: str) -> int:
        cache_key = (store.source, store.band, store.patch, store.group)
        mapping = self._sample_by_tile.get(cache_key)
        if mapping is None:
            rows = self._tile_table(store.band, store.patch)
            mapping = {row.tile_id: row.tile_index for row in rows}
            self._sample_by_tile[cache_key] = mapping
        if tile_id not in mapping:
            raise KeyError(f"{tile_id} not present in {store.source} {store.band} {store.patch} {store.group}")
        return int(mapping[tile_id])

    def tile_by_id(self, band: str, patch: str) -> dict[str, TileRow]:
        return {row.tile_id: row for row in self._tile_table(band, patch)}

    def valid_tiles_for_band(self, band: str, patch: str) -> set[str]:
        if not self._stores(band=band, patch=patch):
            return set()
        return {row.tile_id for row in self._tile_table(band, patch)}

    def choose_tiles(self, patch: str, bands: list[str], *, n_tiles: int | None, all_tiles: bool, seed: int, mode: str = "default") -> list[str]:
        sets = []
        for band in bands:
            valid = self.valid_tiles_for_band(band, patch)
            if valid:
                sets.append(valid)
        common = sorted(set.intersection(*sets), key=lambda text: tuple(int(v) for v in re.findall(r"\d+", text))) if sets else []
        if all_tiles or n_tiles is None:
            return common
        count = min(max(1, int(n_tiles)), len(common))
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(np.asarray(common, dtype=object), size=count, replace=False).tolist())

    def tile_slot_count(
        self,
        patch: str,
        tile_id: str,
        bands: list[str],
        *,
        frames_per_tile: int,
        visit: int | None,
    ) -> int:
        specs = self._slot_specs(patch, bands, frames_per_tile)
        if visit is not None:
            target = _group_label(str(visit))
            specs = [spec for spec in specs if spec[0] == "coadd" or spec[1] == target]
        self._browser_slots[(patch, int(frames_per_tile), str(visit))] = list(specs)
        count = 0
        for source, group in specs:
            if any(self._scan().get((source, band, patch, "" if source == "coadd" else group)) is not None for band in bands):
                count += 1
        return count

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
        specs = self._browser_slots.get((patch, int(frames_per_tile), str(visit)))
        if specs is None:
            specs = self._slot_specs(patch, [band], frames_per_tile)
            if visit is not None:
                target = _group_label(str(visit))
                specs = [spec for spec in specs if spec[0] == "coadd" or spec[1] == target]
        if not specs:
            raise FileNotFoundError(f"no zarr slots for {band} {patch}")
        if int(frame_slot) < 0 or int(frame_slot) >= len(specs):
            raise FileNotFoundError(f"slot {frame_slot} is not present for {band} {patch}")
        source, group = specs[int(frame_slot)]
        store = self._store_for_slot(band, patch, source, group)
        tile = self.tile_by_id(band, patch)[tile_id]
        sample_idx = self._sample_index(store, tile_id)
        group_num = None
        match = re.search(r"(\d+)", group)
        if match:
            group_num = int(match.group(1))
        weight = self.weight_for(source, band, patch, group)
        return FrameRef(
            token=token,
            root=str(self.root),
            tract=self.tract,
            patch=patch,
            band=band,
            pack_path=str(store.path),
            tile_index=tile.tile_index,
            tile_id=tile.tile_id,
            x0=tile.x0,
            y0=tile.y0,
            x1=tile.x1,
            y1=tile.y1,
            frame_slot=int(frame_slot),
            frame_rank=int(frame_slot),
            frame_index=sample_idx,
            tile_length=len(specs),
            visit=group_num,
            weight=weight,
            scale=None,
            dataset=f"hsc_image:{source}",
        )

    def _read_zarr_frame(self, ref: FrameRef) -> np.ndarray:
        source = str(ref.dataset).split(":", 1)[-1] if ":" in str(ref.dataset) else "coadd"
        group = _group_label(str(ref.visit or 0))
        store = _Store(source, ref.band, ref.patch, "" if source == "coadd" else group, Path(ref.pack_path))
        data = self._reader(store).read_first_axis("images", int(ref.frame_index))
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3:
            return np.asarray(arr[0], dtype=np.float32)
        return np.asarray(arr, dtype=np.float32)

    def read_frame(self, ref: FrameRef) -> np.ndarray:
        try:
            return self.read_detection_frame(ref)
        except Exception:
            return self._read_zarr_frame(ref)

    def input_fits_path(self, ref: FrameRef) -> Path:
        source = str(ref.dataset).split(":", 1)[-1] if ":" in str(ref.dataset) else "coadd"
        group = _group_label(str(ref.visit or 0))
        band = str(ref.band)
        patch = str(ref.patch)
        def first_existing(candidates: list[Path]) -> Path:
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            return candidates[0]

        if source == "coadd":
            return first_existing([
                self.coadd_fits_root / self.tract / band / patch / f"warp_half-{band}-{self.tract}-{patch}-0.fits",
                self.coadd_fits_root / self.tract / band / patch / f"calexp-{band}-{self.tract}-{patch}.fits",
                self.coadd_fits_root / band / patch / f"warp_half-{band}-{self.tract}-{patch}-0.fits",
                self.coadd_fits_root / band / patch / f"calexp-{band}-{self.tract}-{patch}.fits",
            ])
        if source == "noisy":
            return first_existing([
                self.noisy_fits_root / self.tract / band / patch / f"warp_8-{band}-{self.tract}-{patch}-{group}.fits",
                self.noisy_fits_root / band / patch / f"warp_8-{band}-{self.tract}-{patch}-{group}.fits",
            ])
        if source == "denoised":
            group_dir = f"group_{int(group):02d}" if str(group).isdigit() else f"group_{group}"
            denoised = self.denoised_fits_root / f"patch_{patch.replace(',', '_')}" / group_dir / band / "denoised.fits"
            if denoised.exists():
                return denoised
            return self.denoised_fits_root / f"patch_{patch.replace(',', '_')}" / group_dir / band / "noisy.fits"
        raise ValueError(f"unsupported hsc_image source: {source}")

    def read_detection_frame(self, ref: FrameRef) -> np.ndarray:
        path = self.input_fits_path(ref)
        full, header, _ = read_fits_image(path)
        width = int(ref.width)
        height = int(ref.height)
        local_x0, local_y0 = crop_origin_for_image(full, header, x0=int(ref.x0), y0=int(ref.y0), width=width, height=height)
        crop, valid = crop_or_pad(full, x0=local_x0, y0=local_y0, width=width, height=height, inference_size=self.tile_size)
        crop = np.where(valid, crop, np.nan).astype(np.float32, copy=False)
        return crop[:height, :width]
