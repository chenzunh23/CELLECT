"""COSMOS / Abell half-coadd adapters shared by the CLI and tile browser.

Coordinates are zero-based: full pointing pixels for COSMOS, common-WCS
4096-parent pixels for Abell. Only bounded FITS sections are read. Full Abell
coadds supply the grid header, never the inference science pixels.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import json
import hashlib
import tempfile
import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from preprocessing.dataset_inputs import ImageInput, discover_cosmos, header_info, load_image, load_paths
from preprocessing.image_processing import prepare_image
from preprocessing.utils.parent_scaling import fit_parent_rgb, apply_parent_rgb
from preprocessing.utils.sky_tiles import SkyTile, tile_header, reproject_tile
from .parallel_load import ParallelLoad
from .jwst import JwstNircamAccess

DEFAULT_COSMOS_ROOT = Path('/data/czh23/JWST/COSMOS_1727_5893')
DEFAULT_ABELL_PLAN = Path('/data/czh23/JWST/Abell2744_preprocessing')
JWST_DATASETS = {'jwst', 'jwst_cosmos', 'jwst_abell'}


def window_header(header, x, y, width, height):
    h = header.copy()
    h['CRPIX1'] = float(h['CRPIX1']) - x
    h['CRPIX2'] = float(h['CRPIX2']) - y
    h['NAXIS1'], h['NAXIS2'] = width, height
    return h


def fits_window(source, x, y, width, height):
    """Return SCI and validity separately; pad out-of-image pixels with NaN."""
    header, (ny, nx), _ = header_info(source.image_fits)
    out = np.full((height, width), np.nan, np.float32)
    valid = np.zeros(out.shape, bool)
    xa, ya, xb, yb = max(0, x), max(0, y), min(nx, x+width), min(ny, y+height)
    if xb > xa and yb > ya:
        loaded = load_image(source, origin=(xa, ya), shape=(yb-ya, xb-xa))
        sl = np.s_[ya-y:yb-y, xa-x:xb-x]
        out[sl] = loaded.image
        valid[sl] = np.isfinite(loaded.image) & ~loaded.bad
    return out, valid, window_header(header, x, y, width, height)


class JwstFieldAccess(JwstNircamAccess):
    """Reuse browser sampling/FrameRef while keeping training units and scaling."""
    pixel_scale_arcsec = 0.03

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-instance caches: a global method LRU would retain old browser
        # sessions and their pixel caches after a dataset switch.
        self._parent_load = ParallelLoad(2, "parent-scaling")
        cached_parameters = lru_cache(maxsize=64)(self.scaling_parameters)
        # lru_cache alone permits concurrent misses to fit the same parent twice.
        def parameters(band, patch, px, py):
            key = (band, patch, px, py)
            return self._parent_load.run(key, lambda: cached_parameters(*key))
        self.scaling_parameters = parameters
        self.grid_header = lru_cache(maxsize=256)(self.grid_header)

    def available_bands(self):
        return sorted({b.upper() for files in self._discover().values() for b in files})

    def available_patches(self, bands=None):
        requested = {str(b).lower() for b in (bands or self.available_bands())}
        return sorted(p for p, files in self._discover().items() if requested.intersection(files))

    def _reference_band(self, patch, bands):
        if not bands:
            raise FileNotFoundError(f'No requested bands for {patch}')
        return bands[0]

    def make_ref(self, **kwargs):
        # New training bands use uppercase IDs (the legacy NIRCam adapter stays unchanged).
        ref = super().make_ref(**kwargs)
        return replace(ref, band=ref.band.upper())

    def _shape_wcs_header(self, path):
        with self._file_lock(path):
            path = Path(path)
            if path not in self._shape_cache:
                header, shape, _ = header_info(path)
                self._shape_cache[path] = shape
                self._headers[path] = header
                self._wcs_cache[path] = WCS(header).celestial
            return self._shape_cache[path], self._wcs_cache[path], self._headers[path]

    def grid_header(self, band, patch):
        return self._shape_wcs_header(self.image_file(band, patch))[2].copy()

    def _shape(self, band, patch):
        h = self.grid_header(band, patch)
        return int(h['NAXIS2']), int(h['NAXIS1'])

    def _wcs_for(self, band, patch):
        return WCS(self.grid_header(band, patch)).celestial

    def _valid_mask_for_band(self, band, patch, ref_band, tile):
        return self.read_window(band, patch, tile.x0, tile.y0, tile.x1-tile.x0, tile.y1-tile.y0)[1]

    def read_frame(self, ref):
        raw, valid, h = self.read_window(ref.band, ref.patch, ref.x0, ref.y0, ref.width, ref.height)
        raw[~valid] = np.nan
        return prepare_image(raw, header=h).image.astype(np.float32)

    def read_window(self, band, patch, x, y, width, height):
        key = ("field", band.lower(), patch, x, y, width, height)
        value = self._tile_pixels.get(key)
        if value is None:
            value = self._read_window(band, patch, x, y, width, height)
            if width * height <= 1024**2:
                self._tile_pixels.put(key, value)
        # Consumers mark invalid pixels; cached SCI must remain unchanged.
        return value[0].copy(), value[1], value[2].copy()

    def _scaling_cache_file(self, band, patch, px, py):
        paths = [self.image_file(band, patch), Path(__file__),
                 Path(__file__).parents[2]/'preprocessing/image_processing.py',
                 Path(__file__).parents[2]/'preprocessing/utils/parent_scaling.py',
                 Path(__file__).parents[2]/'preprocessing/utils/image.py',
                 Path(__file__).parents[2]/'data_filtering/sam_input_scaling.py']
        source = getattr(self, '_sources', {}).get((patch, band.lower()))
        if source is not None and source.quality_mask_npz is not None:
            paths.append(source.quality_mask_npz)
        if self.dataset == 'jwst_abell':
            paths += [p for p in (self.root/'grid_plan.json', self.root/'manifests'/f'{band.upper()}.json') if p.exists()]
        stamps = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
        key = json.dumps([1, self.dataset, patch, band, px, py, stamps], sort_keys=True)
        return Path.home()/'.cache/cellect/jwst_parent_scaling'/f'{hashlib.sha256(key.encode()).hexdigest()}.json'

    def scaling_parameters(self, band, patch, px, py):
        self._discover()
        cache_file = self._scaling_cache_file(band, patch, px, py)
        try:
            cached = json.loads(cache_file.read_text())
            if all(k in cached for k in ('z', 'log', 'log_stats', 'lupton_stats', 'q', 'stretch', 'clip')):
                return cached
        except (OSError, ValueError, TypeError):
            pass
        raw, _, h = self.read_window(band, patch, px, py, 4096, 4096)
        if not np.isfinite(raw).any():
            raise ValueError(f'No finite pixels in scaling parent: {patch}/{band} ({px},{py})')
        params = fit_parent_rgb(prepare_image(raw, header=h).image)
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile('w', dir=cache_file.parent, delete=False) as f:
                json.dump(params, f)
                temporary = f.name
            os.replace(temporary, cache_file)
        except OSError:
            pass  # Read-only home is fine; in-memory caching still works.
        return params

    def scaled_window(self, band, patch, x, y, width=512, height=512):
        raw, valid, h = self.read_window(band, patch, x, y, width, height)
        raw[~valid] = np.nan
        prepared = prepare_image(raw, header=h).image
        # Center chooses the owning parent, including a crop straddling a seam.
        px, py = (int((x+width/2)//4096)*4096, int((y+height/2)//4096)*4096)
        if self.dataset == 'jwst_abell':
            px = py = 0
        params = self.scaling_parameters(str(band).lower(), patch, px, py)
        return prepared, apply_parent_rgb(prepared, params), self.grid_header(band, patch), valid

    def read_scaled_frame(self, ref):
        return self.scaled_window(ref.band, ref.patch, ref.x0, ref.y0, ref.width, ref.height)[1]

    def manifest(self, patch):
        result = super().manifest(patch)
        result.update(dataset=self.dataset, scaling='training parent RGB (4096)',
                      coordinates='zero-based full pointing' if self.dataset == 'jwst_cosmos' else 'zero-based common-WCS parent')
        return result


class CosmosAccess(JwstFieldAccess):
    dataset = 'jwst_cosmos'
    display_name = 'COSMOS 1727 / 5893'

    def _discover(self):
        if self._files_by_patch is None:
            self._files_by_patch = {}
            self._sources = {}
            manifest = self.root / 'manifest.json'
            if not manifest.is_file():
                return self._files_by_patch
            config = dict(load_paths()['cosmos'], manifest=str(manifest))
            for source in discover_cosmos(config):
                patch = f'p{source.proposal}_P{int(source.patch):04d}'
                band = source.band.lower()
                self._files_by_patch.setdefault(patch, {})[band] = source.image_fits
                self._sources[(patch, band)] = source
        return self._files_by_patch

    def _read_window(self, band, patch, x, y, width, height):
        self._discover()
        return fits_window(self._sources[(patch, band.lower())], x, y, width, height)


class AbellAccess(JwstFieldAccess):
    dataset = 'jwst_abell'
    display_name = 'Abell 2744 half coadd'

    def _discover(self):
        if self._files_by_patch is None:
            self._files_by_patch, self._rows = {}, {}
            for manifest in sorted((self.root / 'manifests').glob('*.json')):
                for row in json.loads(manifest.read_text()):
                    if row.get('status') != 'kept':
                        continue
                    path = Path(row['training_fits'])
                    if not path.is_file():
                        path = Path(row['original_training'])
                    if not path.is_file():
                        continue
                    patch, band = row['patch'], row['band'].lower()
                    self._files_by_patch.setdefault(patch, {})[band] = path
                    self._rows[(patch, band)] = row
        return self._files_by_patch

    def grid_header(self, band, patch):
        self._discover()
        row = self._rows[(patch, band.lower())]
        parent = Path(row['training_fits'])
        if parent.is_file():
            return header_info(parent)[0]
        plan = json.loads((self.root / 'grid_plan.json').read_text())
        reference = header_info(plan['reference_fits'])[0]
        native = header_info(row['original_training'])[0]
        return tile_header(reference, SkyTile(row['ix'], row['iy'], row['x0'], row['y0'], row['size']), native)

    def _read_window(self, band, patch, x, y, width, height):
        self._discover()
        row = self._rows[(patch, band.lower())]
        path = self.image_file(band, patch)
        if Path(row['training_fits']).is_file():
            source = ImageInput('abell', patch, band, patch, path, path, dataset_source='half_coadd')
            return fits_window(source, x, y, width, height)
        # Same reprojection as preprocessing; FITS.section avoids loading a whole mosaic.
        h = window_header(self.grid_header(band, patch), x, y, width, height)
        native, shape, idx = header_info(path)
        with fits.open(path, memmap=False) as hd:
            class Section:
                def __init__(self): self.shape = shape
                def __getitem__(self, key): return hd[idx].section[key]
            out = reproject_tile(Section(), WCS(native).celestial, h)
        return out, np.isfinite(out), h
