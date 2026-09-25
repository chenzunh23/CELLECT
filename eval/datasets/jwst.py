from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from .base import FrameRef, TileRow, patch_sort_key


DEFAULT_JWST_NIRCAM_ROOT = Path("/data/czh23/JWST/NIRCam")
DEFAULT_JWST_NIRCAM_BANDS = ("f090w", "f187n", "f300m", "f356w", "f444w")
DEFAULT_JWST_NIRCAM_TILE_SIZE = 512
JWST_MAX_NO_DATA_FRACTION = 0.10


def _require_astropy() -> tuple[Any, Any]:
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
    except Exception as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("JWST NIRCam browsing requires astropy in the active Python environment") from exc
    return fits, WCS


def _band_from_name(path: Path) -> str | None:
    text = path.name.lower()
    for band in DEFAULT_JWST_NIRCAM_BANDS:
        if f"_{band}_" in text or text.endswith(f"_{band}.fits"):
            return band
    return None


def _tile_sort_key(tile_id: str) -> tuple[int, int, str]:
    found = [int(value) for value in re.findall(r"\d+", str(tile_id))]
    if len(found) >= 2:
        return found[0], found[1], str(tile_id)
    return 10**9, 10**9, str(tile_id)


class JwstNircamAccess:
    dataset = "jwst"
    display_name = "JWST NIRCam"
    default_tract = "default"
    default_bands = DEFAULT_JWST_NIRCAM_BANDS
    tile_size = DEFAULT_JWST_NIRCAM_TILE_SIZE
    default_nms_radius = 2
    pixel_scale_arcsec = 0.06
    median_seeing_pixels = 2.0

    def __init__(self, root: Path, tract: str = "default", *, tile_size: int | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.tract = "default" if str(tract or "default") == "default" else str(tract)
        self.tile_size = int(tile_size or self.tile_size)
        self._files_by_patch: dict[str, dict[str, Path]] | None = None
        self._shape_cache: dict[Path, tuple[int, int]] = {}
        self._wcs_cache: dict[Path, Any] = {}
        self._headers: dict[Path, Any] = {}
        self._hdul_cache: dict[Path, Any] = {}
        self._data_cache: dict[Path, np.ndarray] = {}
        self._tiles_by_key: dict[tuple[str, tuple[str, ...]], list[TileRow]] = {}
        self._selected_tiles_by_key: dict[tuple[str, tuple[str, ...]], list[TileRow]] = {}
        self._common_mask_cache: dict[tuple[str, str, tuple[str, ...]], np.ndarray] = {}
        self._active_bands_by_patch: dict[str, tuple[str, ...]] = {}
        self._ref_band_by_patch: dict[tuple[str, tuple[str, ...]], str] = {}

    def _discover(self) -> dict[str, dict[str, Path]]:
        if self._files_by_patch is not None:
            return self._files_by_patch
        out: dict[str, dict[str, Path]] = {}
        if not self.root.is_dir():
            self._files_by_patch = out
            return out
        for patch_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            files: dict[str, Path] = {}
            for path in sorted(patch_dir.glob("*.fits")) + sorted(patch_dir.glob("*.fits.gz")):
                band = _band_from_name(path)
                if band is not None and band not in files:
                    files[band] = path
            if files:
                out[patch_dir.name] = files
        self._files_by_patch = out
        return out

    def available_bands(self) -> list[str]:
        bands: set[str] = set()
        for files in self._discover().values():
            bands.update(files)
        return [band for band in DEFAULT_JWST_NIRCAM_BANDS if band in bands]

    def available_patches(self, bands: list[str] | None = None) -> list[str]:
        requested = set(str(band).lower() for band in (bands or DEFAULT_JWST_NIRCAM_BANDS))
        patches = []
        for patch, files in self._discover().items():
            if requested.intersection(files):
                patches.append(patch)
        return sorted(patches, key=patch_sort_key)

    def image_file(self, band: str, patch: str) -> Path:
        files = self._discover().get(str(patch), {})
        path = files.get(str(band).lower())
        if path is None:
            raise FileNotFoundError(f"no JWST NIRCam {band} FITS for patch {patch} under {self.root}")
        return path

    def _image_hdu_index(self, hdul: Any) -> int:
        for idx, hdu in enumerate(hdul):
            name = str(getattr(hdu, "name", "") or "").upper()
            if hdu.data is not None and (name == "SCI" or idx == 0):
                return idx
        for idx, hdu in enumerate(hdul):
            if hdu.data is not None:
                return idx
        raise ValueError("no image HDU found")

    def _shape_wcs_header(self, path: Path) -> tuple[tuple[int, int], Any, Any]:
        path = Path(path)
        if path not in self._shape_cache:
            fits, WCS = _require_astropy()
            with fits.open(path, memmap=True) as hdul:
                idx = self._image_hdu_index(hdul)
                data = hdul[idx].data
                if data is None:
                    raise ValueError(f"no image HDU found in {path}")
                self._shape_cache[path] = (int(data.shape[-2]), int(data.shape[-1]))
                self._headers[path] = hdul[idx].header.copy()
                self._wcs_cache[path] = WCS(self._headers[path]).celestial
        return self._shape_cache[path], self._wcs_cache[path], self._headers[path]

    def _shape(self, band: str, patch: str) -> tuple[int, int]:
        shape, _wcs, _header = self._shape_wcs_header(self.image_file(band, patch))
        return shape

    def _wcs_for(self, band: str, patch: str) -> Any:
        _shape, wcs, _header = self._shape_wcs_header(self.image_file(band, patch))
        return wcs

    def _image_data(self, path: Path) -> np.ndarray:
        path = Path(path)
        if path in self._data_cache:
            return self._data_cache[path]
        fits, _WCS = _require_astropy()
        hdul = fits.open(path, memmap=True)
        idx = self._image_hdu_index(hdul)
        data = hdul[idx].data
        if data is None:
            hdul.close()
            raise ValueError(f"no image HDU found in {path}")
        self._hdul_cache[path] = hdul
        self._data_cache[path] = data
        return data

    def _present_bands(self, patch: str, bands: list[str]) -> tuple[str, ...]:
        files = self._discover().get(str(patch), {})
        return tuple(str(band).lower() for band in bands if str(band).lower() in files)

    def _reference_band(self, patch: str, bands: tuple[str, ...]) -> str:
        key = (str(patch), tuple(bands))
        if key not in self._ref_band_by_patch:
            for band in DEFAULT_JWST_NIRCAM_BANDS:
                if band in bands:
                    self._ref_band_by_patch[key] = band
                    break
            else:
                raise FileNotFoundError(f"no requested JWST bands are present for patch {patch}")
        return self._ref_band_by_patch[key]

    def _same_grid(self, ref_band: str, band: str, patch: str) -> bool:
        if ref_band == band:
            return True
        try:
            ref_shape, _ref_wcs, ref_header = self._shape_wcs_header(self.image_file(ref_band, patch))
            shape, _wcs, header = self._shape_wcs_header(self.image_file(band, patch))
        except Exception:
            return False
        if ref_shape != shape:
            return False
        keys = ("CTYPE1", "CTYPE2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CD1_1", "CD1_2", "CD2_1", "CD2_2", "CDELT1", "CDELT2")
        for key in keys:
            if key in ref_header or key in header:
                if str(ref_header.get(key, "")) != str(header.get(key, "")):
                    return False
        return True

    def _world_to_band_pixels(self, ref_band: str, band: str, patch: str, tile: TileRow) -> tuple[np.ndarray, np.ndarray]:
        yy, xx = np.indices((int(tile.y1 - tile.y0), int(tile.x1 - tile.x0)), dtype=np.float64)
        ref_x = xx + float(tile.x0)
        ref_y = yy + float(tile.y0)
        ref_wcs = self._wcs_for(ref_band, patch)
        band_wcs = self._wcs_for(band, patch)
        ra, dec = ref_wcs.pixel_to_world_values(ref_x, ref_y)
        sx, sy = band_wcs.world_to_pixel_values(ra, dec)
        return np.asarray(sx, dtype=np.float64), np.asarray(sy, dtype=np.float64)

    @staticmethod
    def _bilinear_sample(data: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        arr = np.asarray(data)
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        x1 = x0 + 1
        y1 = y0 + 1
        valid = (x0 >= 0) & (y0 >= 0) & (x1 < arr.shape[1]) & (y1 < arr.shape[0])
        out = np.full(x.shape, np.nan, dtype=np.float32)
        if not np.any(valid):
            return out
        wx = (x - x0).astype(np.float32)
        wy = (y - y0).astype(np.float32)
        v00 = arr[y0[valid], x0[valid]]
        v10 = arr[y0[valid], x1[valid]]
        v01 = arr[y1[valid], x0[valid]]
        v11 = arr[y1[valid], x1[valid]]
        finite = np.isfinite(v00) & np.isfinite(v10) & np.isfinite(v01) & np.isfinite(v11)
        if np.any(finite):
            vv = np.flatnonzero(valid)[finite]
            out.flat[vv] = (
                (1.0 - wx[valid][finite]) * (1.0 - wy[valid][finite]) * v00[finite]
                + wx[valid][finite] * (1.0 - wy[valid][finite]) * v10[finite]
                + (1.0 - wx[valid][finite]) * wy[valid][finite] * v01[finite]
                + wx[valid][finite] * wy[valid][finite] * v11[finite]
            )
        return out

    def _valid_mask_for_band(self, band: str, patch: str, ref_band: str, tile: TileRow) -> np.ndarray:
        data = self._image_data(self.image_file(band, patch))
        if self._same_grid(ref_band, band, patch):
            crop = data[int(tile.y0) : int(tile.y1), int(tile.x0) : int(tile.x1)]
            return np.isfinite(crop)
        sx, sy = self._world_to_band_pixels(ref_band, band, patch, tile)
        sampled = self._bilinear_sample(data, sx, sy)
        return np.isfinite(sampled)

    def _tile_valid_fraction(self, patch: str, bands: tuple[str, ...], tile: TileRow) -> float:
        ref_band = self._reference_band(patch, bands)
        common: np.ndarray | None = None
        for band in bands:
            mask = self._valid_mask_for_band(band, patch, ref_band, tile)
            common = mask if common is None else (common & mask)
            if common is not None and float(np.mean(common)) < 1.0 - JWST_MAX_NO_DATA_FRACTION:
                # Keep evaluating no further once the intersection already fails.
                return float(np.mean(common))
        return float(np.mean(common)) if common is not None else 0.0

    def _tiles_for_bands(self, patch: str, bands: tuple[str, ...]) -> list[TileRow]:
        key = (str(patch), tuple(bands))
        if key in self._tiles_by_key:
            return self._tiles_by_key[key]
        if not bands:
            self._tiles_by_key[key] = []
            return []
        ref_band = self._reference_band(patch, bands)
        height, width = self._shape(ref_band, patch)
        tiles: list[TileRow] = []
        tile_index = 0
        for ty, y0 in enumerate(range(0, max(0, height - self.tile_size + 1), self.tile_size)):
            for tx, x0 in enumerate(range(0, max(0, width - self.tile_size + 1), self.tile_size)):
                tile = TileRow(tile_index, f"x{tx:03d}_y{ty:03d}", int(x0), int(y0), int(x0 + self.tile_size), int(y0 + self.tile_size))
                if self._tile_valid_fraction(patch, bands, tile) >= 1.0 - JWST_MAX_NO_DATA_FRACTION:
                    tiles.append(TileRow(len(tiles), tile.tile_id, tile.x0, tile.y0, tile.x1, tile.y1))
                tile_index += 1
        self._tiles_by_key[key] = tiles
        return tiles

    def _candidate_tiles_for_bands(self, patch: str, bands: tuple[str, ...]) -> list[TileRow]:
        if not bands:
            return []
        ref_band = self._reference_band(patch, bands)
        height, width = self._shape(ref_band, patch)
        tiles: list[TileRow] = []
        tile_index = 0
        for ty, y0 in enumerate(range(0, max(0, height - self.tile_size + 1), self.tile_size)):
            for tx, x0 in enumerate(range(0, max(0, width - self.tile_size + 1), self.tile_size)):
                tiles.append(TileRow(tile_index, f"x{tx:03d}_y{ty:03d}", int(x0), int(y0), int(x0 + self.tile_size), int(y0 + self.tile_size)))
                tile_index += 1
        return tiles

    def valid_tiles_for_band(self, band: str, patch: str) -> set[str]:
        present = self._present_bands(patch, [band])
        if not present:
            return set()
        return {tile.tile_id for tile in self._tiles_for_bands(patch, present)}

    def choose_tiles(self, patch: str, bands: list[str], *, n_tiles: int | None, all_tiles: bool, seed: int, mode: str = "default") -> list[str]:
        present = self._present_bands(patch, bands)
        self._active_bands_by_patch[str(patch)] = present
        key = (str(patch), tuple(present))
        if all_tiles or n_tiles is None:
            tiles = self._tiles_for_bands(patch, present)
            ids = sorted([tile.tile_id for tile in tiles], key=_tile_sort_key)
            return ids
        candidates = self._candidate_tiles_for_bands(patch, present)
        if not candidates:
            return []
        count = min(max(1, int(n_tiles)), len(candidates))
        rng = np.random.default_rng(seed)
        selected: list[TileRow] = []
        for idx in rng.permutation(len(candidates)).tolist():
            tile = candidates[int(idx)]
            if self._tile_valid_fraction(patch, present, tile) >= 1.0 - JWST_MAX_NO_DATA_FRACTION:
                selected.append(TileRow(len(selected), tile.tile_id, tile.x0, tile.y0, tile.x1, tile.y1))
                if len(selected) >= count:
                    break
        self._selected_tiles_by_key[key] = sorted(selected, key=lambda row: _tile_sort_key(row.tile_id))
        return [tile.tile_id for tile in self._selected_tiles_by_key[key]]

    def tiles(self, band: str, patch: str) -> list[TileRow]:
        active = self._active_bands_by_patch.get(str(patch), self._present_bands(patch, [band]))
        key = (str(patch), tuple(active))
        return self._selected_tiles_by_key.get(key) or self._tiles_for_bands(patch, active)

    def tile_by_id(self, band: str, patch: str) -> dict[str, TileRow]:
        return {row.tile_id: row for row in self.tiles(band, patch)}

    def tile_slot_count(self, patch: str, tile_id: str, bands: list[str], *, frames_per_tile: int, visit: int | None) -> int:
        present = self._present_bands(patch, bands)
        if not present:
            return 0
        self._active_bands_by_patch[str(patch)] = present
        return 1 if tile_id in self.tile_by_id(present[0], patch) else 0

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
        band = str(band).lower()
        tile = self.tile_by_id(band, patch)[tile_id]
        path = self.image_file(band, patch)
        return FrameRef(
            token=token,
            root=str(self.root),
            tract="default",
            patch=str(patch),
            band=band,
            pack_path=str(path),
            tile_index=int(tile.tile_index),
            tile_id=str(tile.tile_id),
            x0=int(tile.x0),
            y0=int(tile.y0),
            x1=int(tile.x1),
            y1=int(tile.y1),
            frame_slot=0,
            frame_rank=0,
            frame_index=0,
            tile_length=1,
            visit=None,
            weight=None,
            scale=None,
            dataset=self.dataset,
        )

    def read_frame(self, ref: FrameRef) -> np.ndarray:
        band = str(ref.band).lower()
        patch = str(ref.patch)
        active = self._active_bands_by_patch.get(patch, self._present_bands(patch, [band]))
        ref_band = self._reference_band(patch, active)
        tile = self.tile_by_id(band, patch)[ref.tile_id]
        data = self._image_data(self.image_file(band, patch))
        if self._same_grid(ref_band, band, patch):
            out = np.array(data[int(tile.y0) : int(tile.y1), int(tile.x0) : int(tile.x1)], dtype=np.float32, copy=True)
        else:
            sx, sy = self._world_to_band_pixels(ref_band, band, patch, tile)
            out = self._bilinear_sample(data, sx, sy)
        common_key = (patch, tile.tile_id, tuple(active))
        common = self._common_mask_cache.get(common_key)
        if common is None:
            for present_band in active:
                mask = self._valid_mask_for_band(present_band, patch, ref_band, tile)
                common = mask if common is None else (common & mask)
            if common is not None:
                self._common_mask_cache[common_key] = common
        if common is not None:
            out = np.asarray(out, dtype=np.float32)
            out[~common] = np.nan
        return out

    def manifest(self, patch: str) -> dict[str, Any]:
        return {
            "tract": "default",
            "patch": str(patch),
            "bands": {band: str(path) for band, path in self._discover().get(str(patch), {}).items()},
            "tile_size": int(self.tile_size),
            "max_no_data_fraction": JWST_MAX_NO_DATA_FRACTION,
        }
