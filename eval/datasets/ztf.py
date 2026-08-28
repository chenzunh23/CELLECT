from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from .base import FrameRef, TileRow


DEFAULT_ZTF_ROOT = Path("/data/shared/ZTF/quadrants")
DEFAULT_ZTF_FIELD = "468"
DEFAULT_ZTF_CCD = "c03"
DEFAULT_ZTF_BANDS = ("zg", "zr", "zi")
DEFAULT_ZTF_PATCHES = ("q1", "q2")
DEFAULT_ZTF_TILE_SIZE = 512
DEFAULT_ZTF_CUT_ORIGIN_DIR = Path("/home/czh23/analysis/2026-08/2026-08-19/ztf_exposure_cut_origins")


def _field_dir_name(field: str | int) -> str:
    text = str(field).strip()
    if text.startswith("field"):
        text = text.removeprefix("field")
    return f"field{int(text):06d}"


def _visit_from_path(path: Path) -> str:
    return path.parent.name


@dataclass(frozen=True)
class CutOriginRow:
    qid: str
    band: str
    obsid: str
    source_path: Path
    tile_size: int
    origin_common_x0: float
    origin_common_y0: float
    tiles_nx: int
    tiles_ny: int
    source_origin_x0: float
    source_origin_y0: float
    source_naxis1: int
    source_naxis2: int
    d_source_x_d_common_x: float
    d_source_x_d_common_y: float
    d_source_y_d_common_x: float
    d_source_y_d_common_y: float


class ZtfAccess:
    dataset = "ztf"
    display_name = "ZTF"
    default_tract = DEFAULT_ZTF_FIELD
    default_bands = DEFAULT_ZTF_BANDS
    tile_size = DEFAULT_ZTF_TILE_SIZE
    default_nms_radius = 2
    pixel_scale_arcsec = 1.013
    median_seeing_pixels = 2.0

    def __init__(
        self,
        root: Path,
        tract: str = DEFAULT_ZTF_FIELD,
        *,
        ccd: str = DEFAULT_ZTF_CCD,
        tile_size: int | None = None,
        cut_origin_dir: Path | None = DEFAULT_ZTF_CUT_ORIGIN_DIR,
    ) -> None:
        self.requested_root = Path(root).expanduser().resolve()
        self.tract = str(tract or DEFAULT_ZTF_FIELD)
        self.ccd = str(ccd or DEFAULT_ZTF_CCD)
        self.tile_size = int(tile_size or self.tile_size)
        self.cut_origin_dir = Path(cut_origin_dir).expanduser().resolve() if cut_origin_dir else None
        self.root = self._resolve_root()
        self._files: dict[tuple[str, str], list[Path]] = {}
        self._tiles: dict[tuple[str, str], list[TileRow]] = {}
        self._shape: dict[Path, tuple[int, int]] = {}
        self._cut_origins_by_patch: dict[str, dict[str, CutOriginRow]] = {}
        self._cut_origin_summary: dict[str, Any] | None = None

    def _resolve_root(self) -> Path:
        field_dir = _field_dir_name(self.tract)
        candidates = [
            self.requested_root / field_dir / self.ccd,
            self.requested_root / "quadrants" / field_dir / self.ccd,
            self.requested_root if self.requested_root.name == self.ccd else self.requested_root / self.ccd,
        ]
        for path in candidates:
            if any((path / patch / "science").is_dir() for patch in DEFAULT_ZTF_PATCHES):
                return path
        return candidates[0]

    def _science_dir(self, band: str, patch: str) -> Path:
        return self.root / str(patch) / "science" / str(band)

    def image_files(self, band: str, patch: str) -> list[Path]:
        key = (str(band), str(patch))
        if key not in self._files:
            parent = self._science_dir(*key)
            files = sorted(
                path
                for path in parent.glob("*/*_sciimg.fits")
                if path.is_file() and f"_{key[0]}_" in path.name and f"_{key[1]}_" in path.name
            )
            self._files[key] = files
        return self._files[key]

    def available_bands(self) -> list[str]:
        bands: set[str] = set()
        if not self.root.is_dir():
            return []
        for patch_dir in sorted(self.root.glob("q*")):
            science = patch_dir / "science"
            if not science.is_dir():
                continue
            for band_dir in sorted(science.iterdir()):
                if band_dir.is_dir() and any(band_dir.glob("*/*_sciimg.fits")):
                    bands.add(band_dir.name)
        return [band for band in DEFAULT_ZTF_BANDS if band in bands] + sorted(bands - set(DEFAULT_ZTF_BANDS))

    def available_patches(self, bands: list[str] | None = None) -> list[str]:
        bands = bands or self.available_bands()
        patches = []
        for patch in DEFAULT_ZTF_PATCHES:
            if any(self._science_dir(band, patch).is_dir() and self.image_files(band, patch) for band in bands):
                patches.append(patch)
        for patch_dir in sorted(self.root.glob("q*")):
            if patch_dir.name not in patches and (patch_dir / "science").is_dir():
                if any(self._science_dir(band, patch_dir.name).is_dir() and self.image_files(band, patch_dir.name) for band in bands):
                    patches.append(patch_dir.name)
        return patches

    def _first_image(self, band: str, patch: str) -> Path:
        files = self.image_files(band, patch)
        if not files:
            raise FileNotFoundError(f"no ZTF science images for field {self.tract} {patch} {band} under {self.root}")
        return files[0]

    def _image_shape(self, path: Path) -> tuple[int, int]:
        if path not in self._shape:
            with fits.open(path, memmap=True) as hdul:
                if hdul[0].data is None:
                    raise ValueError(f"no image in primary HDU: {path}")
                height, width = hdul[0].data.shape[:2]
            self._shape[path] = (int(height), int(width))
        return self._shape[path]

    def _load_cut_origins(self, patch: str) -> dict[str, CutOriginRow]:
        patch = str(patch)
        if patch in self._cut_origins_by_patch:
            return self._cut_origins_by_patch[patch]
        rows: dict[str, CutOriginRow] = {}
        if self.cut_origin_dir is None:
            self._cut_origins_by_patch[patch] = rows
            return rows
        path = self.cut_origin_dir / f"{patch}_cut_origins_{self.tile_size}.csv"
        if not path.exists():
            self._cut_origins_by_patch[patch] = rows
            return rows
        with path.open(newline="", encoding="utf-8") as handle:
            for item in csv.DictReader(handle):
                try:
                    source_path = Path(str(item["source_path"])).expanduser().resolve()
                    row = CutOriginRow(
                        qid=str(item["qid"]),
                        band=str(item["band"]),
                        obsid=str(item["obsid"]),
                        source_path=source_path,
                        tile_size=int(float(item["tile_size"])),
                        origin_common_x0=float(item["origin_common_x0"]),
                        origin_common_y0=float(item["origin_common_y0"]),
                        tiles_nx=int(float(item["tiles_nx"])),
                        tiles_ny=int(float(item["tiles_ny"])),
                        source_origin_x0=float(item["source_origin_x0"]),
                        source_origin_y0=float(item["source_origin_y0"]),
                        source_naxis1=int(float(item["source_naxis1"])),
                        source_naxis2=int(float(item["source_naxis2"])),
                        d_source_x_d_common_x=float(item["d_source_x_d_common_x"]),
                        d_source_x_d_common_y=float(item["d_source_x_d_common_y"]),
                        d_source_y_d_common_x=float(item["d_source_y_d_common_x"]),
                        d_source_y_d_common_y=float(item["d_source_y_d_common_y"]),
                    )
                except Exception:
                    continue
                rows[str(source_path)] = row
        self._cut_origins_by_patch[patch] = rows
        return rows

    def _cut_origin_for_path(self, path: Path, patch: str) -> CutOriginRow | None:
        origins = self._load_cut_origins(patch)
        if not origins:
            return None
        return origins.get(str(Path(path).expanduser().resolve()))

    def _cut_origin_summary_data(self) -> dict[str, Any]:
        if self._cut_origin_summary is not None:
            return self._cut_origin_summary
        summary: dict[str, Any] = {}
        if self.cut_origin_dir is not None:
            path = self.cut_origin_dir / "summary.json"
            if path.exists():
                try:
                    summary = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    summary = {}
        self._cut_origin_summary = summary
        return summary

    def _common_grid_shape_yx(self, patch: str) -> tuple[int, int] | None:
        summary = self._cut_origin_summary_data()
        try:
            shape = summary["q"][str(patch)]["common_grid_shape_yx"]
            return int(shape[0]), int(shape[1])
        except Exception:
            return None

    def tiles(self, band: str, patch: str) -> list[TileRow]:
        key = (str(band), str(patch))
        if key not in self._tiles:
            tiles: list[TileRow] = []
            tile_index = 0
            origins = self._load_cut_origins(key[1])
            origin_rows = [row for row in origins.values() if row.band == key[0]]
            if origin_rows:
                # The cut-origin CSV defines the usable common-grid tile
                # lattice.  Do not recompute a denser grid from the full common
                # bounding box: the extra margin can map outside individual ZTF
                # exposures and makes x000_y000 refer to a different sky area.
                row = origin_rows[0]
                nx, ny = int(row.tiles_nx), int(row.tiles_ny)
                x_origin = int(round(row.origin_common_x0))
                y_origin = int(round(row.origin_common_y0))
                x_values = [x_origin + tx * self.tile_size for tx in range(nx)]
                y_values = [y_origin + ty * self.tile_size for ty in range(ny)]
            else:
                height, width = self._image_shape(self._first_image(*key))
                x_values = list(range(0, width - self.tile_size + 1, self.tile_size))
                y_values = list(range(0, height - self.tile_size + 1, self.tile_size))
            for ty, y0 in enumerate(y_values):
                for tx, x0 in enumerate(x_values):
                    tiles.append(
                        TileRow(
                            tile_index=tile_index,
                            tile_id=f"x{tx:03d}_y{ty:03d}",
                            x0=int(x0),
                            y0=int(y0),
                            x1=int(x0 + self.tile_size),
                            y1=int(y0 + self.tile_size),
                        )
                    )
                    tile_index += 1
            self._tiles[key] = tiles
        return self._tiles[key]

    def tile_by_id(self, band: str, patch: str) -> dict[str, TileRow]:
        return {row.tile_id: row for row in self.tiles(band, patch)}

    def valid_tiles_for_band(self, band: str, patch: str) -> set[str]:
        if not self.image_files(band, patch):
            return set()
        return {tile.tile_id for tile in self.tiles(band, patch)}

    def choose_tiles(self, patch: str, bands: list[str], *, n_tiles: int | None, all_tiles: bool, seed: int, mode: str = "default") -> list[str]:
        sets = []
        usable_bands = []
        for band in bands:
            try:
                valid = self.valid_tiles_for_band(band, patch)
            except Exception as exc:
                print(f"WARNING: skip unavailable ZTF band {band} for {patch}: {exc}", flush=True)
                continue
            if valid:
                sets.append(valid)
                usable_bands.append(band)
        common = sorted(set.intersection(*sets), key=lambda text: tuple(int(v) for v in re.findall(r"\d+", text))) if sets else []
        if not common:
            print(f"WARNING: no common ZTF tiles for {patch} across requested bands {bands}", flush=True)
            return []
        if all_tiles or n_tiles is None:
            return common
        count = min(max(1, int(n_tiles)), len(common))
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(np.asarray(common, dtype=object), size=count, replace=False).tolist())

    def usable_frame_count(self, band: str, patch: str, tile_id: str) -> int:
        return len(self.image_files(band, patch)) if tile_id in self.tile_by_id(band, patch) else 0

    def tile_slot_count(
        self,
        patch: str,
        tile_id: str,
        bands: list[str],
        *,
        frames_per_tile: int,
        visit: int | None,
    ) -> int:
        if visit is not None:
            return 1
        counts = [self.usable_frame_count(band, patch, tile_id) for band in bands]
        return max(0, min(int(frames_per_tile), min(counts))) if counts else 0

    def _choose_file(
        self,
        band: str,
        patch: str,
        *,
        frame_slot: int,
        frame_rank: int,
        frames_per_tile: int,
        visit: int | None,
        strict_visit: bool,
    ) -> tuple[Path, int, int]:
        files = self.image_files(band, patch)
        if not files:
            raise FileNotFoundError(f"no ZTF exposures for {patch} {band}")
        if visit is not None:
            target = str(visit)
            for idx, path in enumerate(files):
                if _visit_from_path(path) == target:
                    return path, idx, len(files)
            if strict_visit:
                raise RuntimeError(f"visit {visit} is not present in ZTF {patch} {band}")
        rank = int(frame_rank)
        if rank < 0:
            rank = len(files) + rank
        if frames_per_tile > 1 and visit is None:
            rank = int(frame_slot)
        rank = max(0, min(rank, len(files) - 1))
        return files[rank], rank, len(files)

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
        path, actual_rank, length = self._choose_file(
            band,
            patch,
            frame_slot=frame_slot,
            frame_rank=frame_rank,
            frames_per_tile=frames_per_tile,
            visit=visit,
            strict_visit=strict_visit,
        )
        obsid = _visit_from_path(path)
        weight: float | None = None
        scale: float | None = None
        try:
            with fits.open(path, memmap=True) as hdul:
                header = hdul[0].header
                weight = float(header["SEEING"]) if "SEEING" in header else None
                scale = float(header["MAGZP"]) if "MAGZP" in header else None
        except Exception:
            pass
        return FrameRef(
            token=token,
            root=str(self.root),
            tract=str(self.tract),
            patch=str(patch),
            band=str(band),
            pack_path=str(path),
            tile_index=int(tile.tile_index),
            tile_id=str(tile.tile_id),
            x0=int(tile.x0),
            y0=int(tile.y0),
            x1=int(tile.x1),
            y1=int(tile.y1),
            frame_slot=int(frame_slot),
            frame_rank=int(actual_rank),
            frame_index=int(actual_rank),
            tile_length=int(length),
            visit=int(obsid) if obsid.isdigit() else None,
            weight=weight,
            scale=scale,
            dataset=self.dataset,
        )

    def _read_aligned_frame(self, ref: FrameRef, data: np.ndarray, origin: CutOriginRow) -> np.ndarray:
        height = int(ref.height)
        width = int(ref.width)
        yy, xx = np.indices((height, width), dtype=np.float32)
        common_x = float(ref.x0) + xx
        common_y = float(ref.y0) + yy
        dx = common_x - float(origin.origin_common_x0)
        dy = common_y - float(origin.origin_common_y0)
        source_x = float(origin.source_origin_x0) + float(origin.d_source_x_d_common_x) * dx + float(origin.d_source_x_d_common_y) * dy
        source_y = float(origin.source_origin_y0) + float(origin.d_source_y_d_common_x) * dx + float(origin.d_source_y_d_common_y) * dy

        x0 = np.floor(source_x).astype(np.int64)
        y0 = np.floor(source_y).astype(np.int64)
        x1 = x0 + 1
        y1 = y0 + 1
        valid = (x0 >= 0) & (y0 >= 0) & (x1 < data.shape[1]) & (y1 < data.shape[0])
        out = np.full((height, width), np.nan, dtype=np.float32)
        if not np.any(valid):
            return out
        wx = (source_x - x0).astype(np.float32)
        wy = (source_y - y0).astype(np.float32)
        v00 = np.asarray(data[y0[valid], x0[valid]], dtype=np.float32)
        v10 = np.asarray(data[y0[valid], x1[valid]], dtype=np.float32)
        v01 = np.asarray(data[y1[valid], x0[valid]], dtype=np.float32)
        v11 = np.asarray(data[y1[valid], x1[valid]], dtype=np.float32)
        wxv = wx[valid]
        wyv = wy[valid]
        out[valid] = (
            (1.0 - wxv) * (1.0 - wyv) * v00
            + wxv * (1.0 - wyv) * v10
            + (1.0 - wxv) * wyv * v01
            + wxv * wyv * v11
        )
        return out

    def read_frame(self, ref: FrameRef) -> np.ndarray:
        with fits.open(Path(ref.pack_path), memmap=True) as hdul:
            data = hdul[0].data
            if data is None:
                raise ValueError(f"no image in primary HDU: {ref.pack_path}")
            origin = self._cut_origin_for_path(Path(ref.pack_path), ref.patch)
            if origin is not None:
                return self._read_aligned_frame(ref, np.asarray(data), origin)
            if self._load_cut_origins(ref.patch):
                raise KeyError(f"ZTF cut-origin row not found for {ref.pack_path}")
            return np.array(data[int(ref.y0) : int(ref.y1), int(ref.x0) : int(ref.x1)], dtype=np.float32, copy=True)

    def manifest(self, patch: str) -> dict[str, Any]:
        rows = []
        for band in self.available_bands():
            for idx, path in enumerate(self.image_files(band, patch)):
                rows.append({"band": band, "frame_index": idx, "visit": _visit_from_path(path), "path": str(path)})
        return {"field": self.tract, "patch": patch, "frames": rows}
