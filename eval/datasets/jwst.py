from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .base import FrameRef, TileRow, patch_sort_key
from .intensity import hsc_intensity_frame
from .array_cache import ArrayCache


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
        self._tile_pixels = ArrayCache(128 * 1024**2)
        self._grid_relation = {}
        self._path_locks = {}
        self._path_registry_lock = threading.Lock()
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

    def close(self):
        for hdul in getattr(self, '_hdul_cache', {}).values():
            hdul.close()
        self._hdul_cache.clear()
        self._data_cache.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

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
        from preprocessing.dataset_inputs import image_hdu
        return image_hdu(hdul)

    def _file_lock(self, path):
        with self._path_registry_lock:
            return self._path_locks.setdefault(Path(path), threading.RLock())

    def _shape_wcs_header(self, path: Path) -> tuple[tuple[int, int], Any, Any]:
        with self._file_lock(path):
            path = Path(path)
            if path not in self._shape_cache:
                fits, WCS = _require_astropy()
                with fits.open(path, memmap=True) as hdul:
                    idx = self._image_hdu_index(hdul)
                    self._shape_cache[path] = tuple(hdul[idx].shape[-2:])
                    header = hdul[0].header.copy()
                    header.update(hdul[idx].header)
                    self._headers[path] = header
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

    def _affine_grid_transform(self, ref_band, band, patch):
        """Exact linear pixel transform only when celestial projections coincide."""
        key = (ref_band, band, patch)
        if key not in self._grid_relation:
            a, b = self._wcs_for(ref_band, patch), self._wcs_for(band, patch)
            same_projection = (not a.has_distortion and not b.has_distortion
                and list(a.wcs.ctype) == list(b.wcs.ctype)
                and list(a.wcs.cunit) == list(b.wcs.cunit)
                and np.array_equal(a.wcs.crval, b.wcs.crval)
                and a.wcs.lonpole == b.wcs.lonpole and a.wcs.latpole == b.wcs.latpole
                and a.wcs.radesys == b.wcs.radesys
                and np.allclose(a.wcs.equinox, b.wcs.equinox, equal_nan=True)
                and a.wcs.get_pv() == b.wcs.get_pv() and a.wcs.get_ps() == b.wcs.get_ps())
            if same_projection:
                matrix = np.linalg.solve(b.pixel_scale_matrix, a.pixel_scale_matrix)
                offset = b.wcs.crpix - 1 + matrix @ (1 - a.wcs.crpix)
                self._grid_relation[key] = (matrix, offset)
            else:
                self._grid_relation[key] = None
        return self._grid_relation[key]

    def _tan_grid_transform(self, ref_band, band, patch):
        # TAN-to-TAN is an exact projective transform. Restrict to ordinary
        # equatorial TAN without distortion/PV; other WCS use Astropy below.
        key = ("tan", ref_band, band, patch)
        if key not in self._grid_relation:
            a, b = self._wcs_for(ref_band, patch), self._wcs_for(band, patch)
            eligible = (not a.has_distortion and not b.has_distortion
                and list(a.wcs.ctype) == list(b.wcs.ctype) == ['RA---TAN', 'DEC--TAN']
                and all(str(u) == 'deg' for u in list(a.wcs.cunit) + list(b.wcs.cunit))
                and a.wcs.lonpole == b.wcs.lonpole == 180.
                and a.wcs.radesys == b.wcs.radesys
                and np.allclose(a.wcs.equinox, b.wcs.equinox, equal_nan=True)
                and not a.wcs.get_pv() and not b.wcs.get_pv()
                and not a.wcs.get_ps() and not b.wcs.get_ps())
            result = None
            if eligible:
                def basis(w):
                    ra, dec = np.deg2rad(w.wcs.crval)
                    return np.array([[-np.sin(ra), -np.sin(dec)*np.cos(ra), np.cos(dec)*np.cos(ra)],
                        [np.cos(ra), -np.sin(dec)*np.sin(ra), np.cos(dec)*np.sin(ra)],
                        [0, np.cos(dec), np.sin(dec)]])
                src = np.eye(3)
                src[:2,:2] = np.deg2rad(a.pixel_scale_matrix)
                src[:2,2] = src[:2,:2] @ (1-a.wcs.crpix)
                dst = np.eye(3)
                dst[:2,:2] = np.linalg.inv(np.deg2rad(b.pixel_scale_matrix))
                dst[:2,2] = b.wcs.crpix-1
                result = dst @ basis(b).T @ basis(a) @ src
            self._grid_relation[key] = result
        return self._grid_relation[key]

    def _same_grid(self, ref_band: str, band: str, patch: str) -> bool:
        transform = self._affine_grid_transform(ref_band, band, patch)
        return (transform is not None and self._shape(ref_band, patch) == self._shape(band, patch)
                and np.allclose(transform[0], np.eye(2), rtol=0, atol=1e-12)
                and np.allclose(transform[1], 0, rtol=0, atol=1e-7))

    def _world_to_band_pixels(self, ref_band: str, band: str, patch: str, tile: TileRow):
        yy, xx = np.indices((tile.height, tile.width), dtype=np.float64)
        xx += tile.x0
        yy += tile.y0
        affine = self._affine_grid_transform(ref_band, band, patch)
        if affine is not None:
            m, t = affine
            return m[0,0]*xx+m[0,1]*yy+t[0], m[1,0]*xx+m[1,1]*yy+t[1]
        matrix = self._tan_grid_transform(ref_band, band, patch)
        if matrix is not None:
            den = matrix[2,0]*xx + matrix[2,1]*yy + matrix[2,2]
            with np.errstate(divide='ignore', invalid='ignore'):
                sx = (matrix[0,0]*xx + matrix[0,1]*yy + matrix[0,2]) / den
                sy = (matrix[1,0]*xx + matrix[1,1]*yy + matrix[1,2]) / den
            return np.where(den > 0, sx, np.nan), np.where(den > 0, sy, np.nan)
        ra, dec = self._wcs_for(ref_band, patch).pixel_to_world_values(xx, yy)
        return self._wcs_for(band, patch).world_to_pixel_values(ra, dec)

    def _read_section(self, path, rows, columns):
        with self._file_lock(path):
            # Unscaled FITS SCI uses a zero-copy mmap: only the requested rectangle
            # is copied/read. Astropy.section otherwise performs hundreds of Python
            # row reads. Scaled integer/compressed HDUs retain safe section access.
            fits, _ = _require_astropy()
            if path not in self._hdul_cache:
                self._hdul_cache[path] = fits.open(path, memmap=True)
            hdu = self._hdul_cache[path][self._image_hdu_index(self._hdul_cache[path])]
            scaled = any(k in hdu.header for k in ('BZERO', 'BSCALE', 'BLANK'))
            if scaled or isinstance(hdu, fits.CompImageHDU):
                with fits.open(path, memmap=False) as hd:
                    data = hd[self._image_hdu_index(hd)].section[rows, columns]
                    return np.array(data, dtype=np.float32, copy=True)
            return np.array(hdu.data[rows, columns], dtype=np.float32, copy=True)

    def _sample_tile(self, band, patch, ref_band, tile):
        key = (band, patch, ref_band, tile.x0, tile.y0, tile.x1, tile.y1)
        cached = self._tile_pixels.get(key)
        if cached is not None:
            return cached
        path = self.image_file(band, patch)
        fits, _ = _require_astropy()
        ny, nx = self._shape(band, patch)
        if self._same_grid(ref_band, band, patch):
            out = self._read_section(path, slice(tile.y0,tile.y1), slice(tile.x0,tile.x1))
        else:
            sx, sy = self._world_to_band_pixels(ref_band, band, patch, tile)
            finite = np.isfinite(sx) & np.isfinite(sy)
            out = np.full((tile.height, tile.width), np.nan, np.float32)
            if finite.any():
                xa, ya = max(0, int(np.floor(sx[finite].min()))), max(0, int(np.floor(sy[finite].min())))
                xb, yb = min(nx, int(np.floor(sx[finite].max()))+2), min(ny, int(np.floor(sy[finite].max()))+2)
                if xb > xa and yb > ya:
                    cut = self._read_section(path, slice(ya,yb), slice(xa,xb))
                    out = self._bilinear_sample(cut, np.where(finite, sx-xa, -1), np.where(finite, sy-ya, -1))
                    out[~finite] = np.nan
        return self._tile_pixels.put(key, out)

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
        return np.isfinite(self._sample_tile(band, patch, ref_band, tile))

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
        return self._selected_tiles_by_key[key] if key in self._selected_tiles_by_key else self._tiles_for_bands(patch, active)

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
        path = self.image_file(band, patch)
        out = self._sample_tile(band, patch, ref_band, tile).copy()
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
        _shape, _wcs, header = self._shape_wcs_header(path)
        return hsc_intensity_frame(out, header, input_unit="auto", nan_policy="hybrid")

    def manifest(self, patch: str) -> dict[str, Any]:
        return {
            "tract": "default",
            "patch": str(patch),
            "bands": {band: str(path) for band, path in self._discover().get(str(patch), {}).items()},
            "tile_size": int(self.tile_size),
            "max_no_data_fraction": JWST_MAX_NO_DATA_FRACTION,
        }
