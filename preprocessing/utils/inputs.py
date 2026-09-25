"""Small FITS/path/tiling primitives owned by preprocessing.

Migrated from astro_data_preprocessing without importing its CLI or pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import math
import numpy as np
from astropy.io import fits

@dataclass(frozen=True)
class TileSpec:
    name: str
    x0: int
    y0: int
    size: int
    row: Optional[int] = None
    col: Optional[int] = None
    kind: str = "grid"

    @property
    def x1(self) -> int:
        return self.x0 + self.size

    @property
    def y1(self) -> int:
        return self.y0 + self.size

def _origin_from_ltv(header: fits.Header) -> Tuple[int, int]:
    if "LTV1" not in header or "LTV2" not in header:
        return 0, 0
    return -int(round(float(header["LTV1"]))), -int(round(float(header["LTV2"])))

def _find_image_hdu_index(hdul: fits.HDUList) -> int:
    if "IMAGE" in hdul:
        return hdul.index_of("IMAGE")
    for idx, hdu in enumerate(hdul):
        data = getattr(hdu, "data", None)
        if data is not None and getattr(data, "ndim", None) == 2:
            return idx
    raise KeyError("No 2D image HDU found; expected IMAGE or a 2D image extension")

def _band_fits_path(coadd_root: Path, band: str, tract: int, patch: str) -> Path:
    filenames = [
        f"deepCoadd-{band}-{tract}-{patch}.fits",
        f"calexp-{band}-{tract}-{patch}.fits",
    ]
    candidates = [
        base / filename
        for base in (
            coadd_root / band,
            coadd_root / band / patch,
            coadd_root / str(tract) / band / patch,
            coadd_root / str(tract) / band,
        )
        for filename in filenames
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    search_dirs = [
        coadd_root / band,
        coadd_root / band / patch,
        coadd_root / str(tract) / band / patch,
        coadd_root / str(tract) / band,
    ]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        matches = sorted(
            path
            for path in search_dir.glob(f"*{band}*{tract}*{patch}*.fits")
            if not path.name.startswith(("meas-", "det-", "det_bkgd-"))
        )
        if matches:
            return matches[0]
    return candidates[0]

def _band_catalog_path(catalog_root: Path, band: str, tract: int, patch: str) -> Path:
    filename = f"meas-{band}-{tract}-{patch}.fits"
    candidates = [
        catalog_root / band / filename,
        catalog_root / band / patch / filename,
        catalog_root / str(tract) / band / patch / filename,
        catalog_root / str(tract) / band / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

def _band_det_path(root: Path, band: str, tract: int, patch: str) -> Optional[Path]:
    filename = f"det-{band}-{tract}-{patch}.fits"
    candidates = [
        root / band / filename,
        root / band / patch / filename,
        root / str(tract) / band / patch / filename,
        root / str(tract) / band / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None

def _read_det_background_mask(path: Path, shape_yx: Tuple[int, int], origin_xy: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """Return True for pixels outside LSST detection footprints."""
    with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
        if len(hdul) <= 4:
            return ~np.zeros(shape_yx, dtype=bool)
        spans = hdul[4].data
        if spans is None:
            return ~np.zeros(shape_yx, dtype=bool)
        rows = [(int(row["y"]), int(row["x0"]), int(row["x1"])) for row in spans]

    def _paint(*, subtract_origin: bool) -> tuple[np.ndarray, int]:
        footprint = np.zeros(shape_yx, dtype=bool)
        painted = 0
        ox, oy = origin_xy if subtract_origin else (0, 0)
        for raw_y, raw_x0, raw_x1 in rows:
            y = raw_y - int(oy)
            if y < 0 or y >= shape_yx[0]:
                continue
            x0 = max(0, raw_x0 - int(ox))
            x1 = min(shape_yx[1] - 1, raw_x1 - int(ox))
            if x1 >= x0:
                footprint[y, x0 : x1 + 1] = True
                painted += x1 - x0 + 1
        return footprint, painted

    footprint, painted = _paint(subtract_origin=False)
    if painted == 0 and origin_xy != (0, 0):
        footprint, _painted = _paint(subtract_origin=True)
    return ~footprint

def edge_aligned_starts(length: int, tile_size: int, stride: int) -> List[int]:
    """Return floor-count stride starts with the final tile aligned to the edge.

    For length=4200, tile_size=512, stride=368 this returns 11 starts.  The last
    regular stride start is replaced by 4200-512=3688, matching the requested
    11x11 grid rather than adding a twelfth nearly duplicate edge tile.
    """

    if length <= tile_size:
        return [0]
    n = int(math.floor((length - tile_size) / stride)) + 1
    starts = [i * stride for i in range(max(n, 1))]
    starts[-1] = length - tile_size
    return starts

def make_tile_specs(
    *,
    parent_origin: Tuple[int, int],
    image_shape: Tuple[int, int],
    tile_size: int,
    stride: int,
    compare_origin: Optional[Tuple[int, int]],
) -> List[TileSpec]:
    width, height = image_shape
    x_starts = edge_aligned_starts(width, tile_size, stride)
    y_starts = edge_aligned_starts(height, tile_size, stride)
    specs: List[TileSpec] = []
    seen: set[Tuple[int, int]] = set()

    for row, y_local in enumerate(y_starts):
        for col, x_local in enumerate(x_starts):
            x0 = parent_origin[0] + x_local
            y0 = parent_origin[1] + y_local
            name = f"grid_r{row:02d}_c{col:02d}_x{x0}_y{y0}"
            specs.append(TileSpec(name=name, x0=x0, y0=y0, size=tile_size, row=row, col=col))
            seen.add((x0, y0))

    if compare_origin is not None and compare_origin not in seen:
        x0, y0 = compare_origin
        specs.append(TileSpec(name=f"sam_x{x0}_y{y0}", x0=x0, y0=y0, size=tile_size, kind="compare"))
    return specs
