"""Metadata-first selection of mixed-survey Zarr samples without image decoding."""
from pathlib import Path
import json
import os
import re
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astro_train_zarr_data import PatchZarrReader


def zarr_header(attrs):
    value = attrs.get('sky_wcs_header')
    if not value:
        return None
    return fits.Header.fromstring(value, sep='\n') if isinstance(value, str) else fits.Header(value)


def resolve_field_zarr(*, root=None, zarr_store=None, sample_index=0, band=None,
                       dataset='auto', proposal=None, patch=None, tile_name=None,
                       xy=None, radec=None, dataset_source=None, group=None):
    from eval.eval_utils import decode_fixed_utf8, tile_name_matches, zarr_sample_group, normalize_group_name
    if sample_index < 0:
        raise ValueError('sample-index must be nonnegative')
    if xy is not None and radec is not None:
        raise ValueError('Choose pixel coordinates or RA/Dec, not both')
    if zarr_store:
        stores = [Path(zarr_store).expanduser()]
    else:
        if root is None:
            raise ValueError('Provide a Zarr root or store')
        stores = []
        # A store can contain thousands of chunks; only visit the outer directory tree.
        for current, dirs, files in os.walk(Path(root).expanduser()):
            dirs.sort()
            if current.endswith('.zarr'):
                dirs[:] = []
                if '.zattrs' in files:
                    stores.append(Path(current))
        stores.sort()
    remaining = int(sample_index)
    for store in stores:
        attrs = json.loads((store / '.zattrs').read_text())
        if dataset != 'auto' and str(attrs.get('dataset', '')).lower() != dataset:
            continue
        if proposal is not None and str(attrs.get('proposal')) != str(proposal):
            continue
        actual_patch = str(attrs.get('patch', store.stem))
        if patch and patch != actual_patch and not (
            re.fullmatch(r'P\d{4}', patch, re.I) and actual_patch.upper().startswith(patch.upper() + '_')
        ):
            continue
        bands = list(attrs.get('bands', []))
        matches = [i for i, b in enumerate(bands) if band is None or str(b).upper() == band.upper()]
        if not matches:
            continue
        reader = PatchZarrReader(store)
        count = reader.meta('images').shape[0]
        indices = np.arange(count)
        if xy is not None or radec is not None:
            point = xy
            if radec is not None:
                header = zarr_header(attrs)
                if header is None:
                    continue
                point = WCS(header).celestial.world_to_pixel_values(*radec)
            if not reader.has_array('tile_x0') or not reader.has_array('tile_y0'):
                continue
            xs, ys = reader.read_full_small('tile_x0'), reader.read_full_small('tile_y0')
            h, w = reader.meta('images').shape[-2:]
            x, y = point
            indices = indices[(xs <= x) & (x < xs+w) & (ys <= y) & (y < ys+h)]
            # For overlapping samples choose the closest center first, stable for ties.
            distance = (xs[indices]+(w-1)/2-x)**2 + (ys[indices]+(h-1)/2-y)**2
            indices = indices[np.argsort(distance, kind='stable')]
        if tile_name:
            names = decode_fixed_utf8(reader.read_full_small('tile_name')) if reader.has_array('tile_name') else []
            indices = [i for i in indices if i < len(names) and tile_name_matches(names[i], tile_name)]
        if dataset_source:
            if reader.has_array('dataset_source'):
                names = decode_fixed_utf8(reader.read_full_small('dataset_source'))
                indices = [i for i in indices if names[i] == dataset_source]
            elif attrs.get('dataset_source') != dataset_source:
                continue
        if group is not None:
            group = normalize_group_name(group)
            indices = [i for i in indices if zarr_sample_group(reader, int(i)) == group]
        if remaining < len(indices):
            return reader, int(indices[remaining]), matches[0], attrs
        remaining -= len(indices)
    raise LookupError(f'No matching Zarr sample at index {sample_index}: dataset={dataset}, proposal={proposal}, patch={patch}, band={band}, xy={xy}, radec={radec}')
