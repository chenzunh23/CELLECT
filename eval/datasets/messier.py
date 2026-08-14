from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .base import FrameRef, TileRow


DEFAULT_MESSIER_ROOT = Path("/data/czh23/Messier")
MESSIER_EXTENSIONS = {".fits", ".fit", ".fts", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy", ".npz"}
MESSIER_TIFF_EXTENSIONS = {".tif", ".tiff"}
PEAK_SEARCH_MAX_SIDE = 2048


def _read_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        data = np.load(path)
        arr = data[data.files[0]] if suffix == ".npz" else data
    elif suffix in {".fits", ".fit", ".fts"}:
        from astropy.io import fits

        with fits.open(path, memmap=False) as hdul:
            arr = None
            for hdu in hdul:
                if hdu.data is not None:
                    arr = np.asarray(hdu.data)
                    break
            if arr is None:
                raise ValueError(f"no image HDU found in {path}")
    else:
        with Image.open(path) as img:
            arr = np.asarray(img)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[-1] >= 3:
            arr = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
        else:
            arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"expected 2D image, got shape={arr.shape} from {path}")
    return arr


def _uint8_from_percentiles(image: np.ndarray, low: float = 1.0, high: float = 99.8) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    fill = float(np.nanmedian(arr[finite])) if np.any(finite) else 0.0
    arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
    lo, hi = np.percentile(arr, [float(low), float(high)])
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return np.rint(arr * 255.0).astype(np.uint8)


def _subsample_for_peak_search(image: np.ndarray, max_side: int = PEAK_SEARCH_MAX_SIDE) -> tuple[np.ndarray, float, float]:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return np.asarray(image, dtype=np.float32), 1.0, 1.0
    stride = max(1, int(np.ceil(float(longest) / float(max_side))))
    small = np.asarray(image[::stride, ::stride], dtype=np.float32)
    return small, float(width) / float(small.shape[1]), float(height) / float(small.shape[0])


def _source_density_for_peak_search(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Score cluster-like fields on a downsampled image to avoid slow full-res filtering."""
    small, scale_x, scale_y = _subsample_for_peak_search(image)
    base_u8 = _uint8_from_percentiles(small)
    radius_scale = max(scale_x, scale_y)
    background_radius = max(2.0, 32.0 / radius_scale)
    density_radius = max(4.0, 96.0 / radius_scale)
    broad_radius = max(4.0, 128.0 / radius_scale)
    base = base_u8.astype(np.float32) / 255.0
    background = np.asarray(
        Image.fromarray(base_u8, mode="L").filter(ImageFilter.GaussianBlur(radius=background_radius)),
        dtype=np.float32,
    ) / 255.0
    highpass = np.clip(base - background, 0.0, 1.0)
    finite = highpass[np.isfinite(highpass)]
    if finite.size == 0:
        return np.zeros_like(highpass, dtype=np.float32)
    threshold = max(float(np.percentile(finite, 97.5)), float(np.median(finite) + 3.0 * np.std(finite)))
    source_mask = (highpass >= threshold).astype(np.uint8) * 255
    density = np.asarray(
        Image.fromarray(source_mask, mode="L").filter(ImageFilter.GaussianBlur(radius=density_radius)),
        dtype=np.float32,
    )
    # Mild intensity prior breaks ties inside equally dense fields without letting one
    # isolated saturated source dominate the selection.
    broad_light = np.asarray(
        Image.fromarray(base_u8, mode="L").filter(ImageFilter.GaussianBlur(radius=broad_radius)),
        dtype=np.float32,
    )
    return density + 0.15 * broad_light, scale_x, scale_y


class MessierAccess:
    dataset = "sitian"
    display_name = "Sitian"
    default_tract = "default"
    default_bands = ("default",)
    tile_size = 512

    def __init__(self, root: Path, tract: str = "default", *, selection_mode: str = "brightest") -> None:
        self.root = Path(root).expanduser().resolve()
        self.tract = str(tract or "default")
        self.selection_mode = str(selection_mode)
        self._patch_files: dict[str, list[Path]] | None = None
        self._images: dict[str, np.ndarray] = {}
        self._tiles_by_patch: dict[str, list[TileRow]] = {}

    def _discover(self) -> dict[str, list[Path]]:
        if self._patch_files is not None:
            return self._patch_files
        out: dict[str, list[Path]] = {}
        if not self.root.is_dir():
            self._patch_files = out
            return out
        for obj_dir in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            all_dir = obj_dir / "all"
            all_tiffs = [
                path
                for path in sorted(all_dir.glob("*"))
                if path.is_file() and path.suffix.lower() in MESSIER_TIFF_EXTENSIONS and not path.name.startswith(".")
            ]
            files = all_tiffs
            if not files:
                files = [
                    path
                    for path in sorted(obj_dir.rglob("*"))
                    if path.is_file() and path.suffix.lower() in MESSIER_EXTENSIONS and not path.name.startswith(".")
                ]
            if files:
                files = sorted(files, key=lambda p: p.as_posix())
                out[obj_dir.name] = files
        self._patch_files = out
        return out

    def available_bands(self) -> list[str]:
        return ["default"]

    def available_patches(self, bands: list[str] | None = None) -> list[str]:
        return sorted(self._discover())

    def image_files(self, patch: str) -> list[Path]:
        files = self._discover().get(str(patch), [])
        if not files:
            raise FileNotFoundError(f"no Messier image files found for {patch} under {self.root}")
        return files

    def _image_key(self, patch: str, frame_index: int) -> str:
        return f"{patch}:{frame_index}"

    def read_image_file(self, patch: str, frame_index: int = 0) -> np.ndarray:
        key = self._image_key(patch, frame_index)
        if key not in self._images:
            self._images[key] = _read_image(self.image_files(patch)[frame_index])
        return self._images[key]

    def _crop_tile(self, image: np.ndarray, tile_index: int, cx: int, cy: int) -> TileRow:
        height, width = image.shape[:2]
        size = self.tile_size
        x0 = int(np.clip(int(round(cx)) - size // 2, 0, max(0, width - size)))
        y0 = int(np.clip(int(round(cy)) - size // 2, 0, max(0, height - size)))
        return TileRow(tile_index=tile_index, tile_id=f"x{x0}_y{y0}", x0=x0, y0=y0, x1=x0 + size, y1=y0 + size)

    def _brightest_tiles(self, patch: str, count: int = 4) -> list[TileRow]:
        image = self.read_image_file(patch, 0)
        height, width = image.shape[:2]
        if height < self.tile_size or width < self.tile_size:
            raise ValueError(f"Messier image {patch} is smaller than {self.tile_size}x{self.tile_size}: {image.shape}")
        score, scale_x, scale_y = _source_density_for_peak_search(image)
        work = score.copy()
        score_height, score_width = work.shape[:2]
        margin_x = max(1, int(np.ceil(float(self.tile_size) / float(scale_x))))
        margin_y = max(1, int(np.ceil(float(self.tile_size) / float(scale_y))))
        if score_height > 2 * margin_y and score_width > 2 * margin_x:
            work[:margin_y, :] = -np.inf
            work[-margin_y:, :] = -np.inf
            work[:, :margin_x] = -np.inf
            work[:, -margin_x:] = -np.inf
        tiles: list[TileRow] = []
        yy, xx = np.ogrid[:score_height, :score_width]
        dedupe_radius = max(2, int(round(256.0 / max(scale_x, scale_y))))
        for _ in range(max(1, int(count))):
            if not np.isfinite(work).any():
                break
            cy_small, cx_small = np.unravel_index(int(np.nanargmax(work)), work.shape)
            cx = int(round((float(cx_small) + 0.5) * scale_x - 0.5))
            cy = int(round((float(cy_small) + 0.5) * scale_y - 0.5))
            tile = self._crop_tile(image, len(tiles), cx, cy)
            if all(abs(tile.x0 - old.x0) > 8 or abs(tile.y0 - old.y0) > 8 for old in tiles):
                tiles.append(tile)
            mask = (xx - cx_small) ** 2 + (yy - cy_small) ** 2 <= dedupe_radius**2
            work[mask] = -np.inf
        return tiles

    def _grid_tiles(self, patch: str) -> list[TileRow]:
        image = self.read_image_file(patch, 0)
        height, width = image.shape[:2]
        tiles: list[TileRow] = []
        size = self.tile_size
        for y0 in range(0, max(1, height - size + 1), size):
            for x0 in range(0, max(1, width - size + 1), size):
                if x0 + size <= width and y0 + size <= height:
                    tiles.append(TileRow(len(tiles), f"x{x0}_y{y0}", x0, y0, x0 + size, y0 + size))
        return tiles

    def choose_tiles(self, patch: str, bands: list[str], *, n_tiles: int | None, all_tiles: bool, seed: int, mode: str = "brightest") -> list[str]:
        mode = str(mode or self.selection_mode)
        if mode == "random_grid":
            tiles = self._grid_tiles(patch)
            if not all_tiles:
                count = min(max(1, int(n_tiles or 4)), len(tiles))
                rng = np.random.default_rng(seed)
                tiles = [tiles[int(i)] for i in sorted(rng.choice(np.arange(len(tiles)), size=count, replace=False).tolist())]
        else:
            tiles = self._brightest_tiles(patch, count=int(n_tiles or 4))
            if all_tiles:
                tiles = self._grid_tiles(patch)
        self._tiles_by_patch[patch] = tiles
        return [tile.tile_id for tile in tiles]

    def tiles(self, band: str, patch: str) -> list[TileRow]:
        if patch not in self._tiles_by_patch:
            self._tiles_by_patch[patch] = self._brightest_tiles(patch, count=4)
        return self._tiles_by_patch[patch]

    def tile_by_id(self, band: str, patch: str) -> dict[str, TileRow]:
        return {row.tile_id: row for row in self.tiles(band, patch)}

    def tile_slot_count(
        self,
        patch: str,
        tile_id: str,
        bands: list[str],
        *,
        frames_per_tile: int,
        visit: int | None,
    ) -> int:
        return min(max(1, int(frames_per_tile)), min(2, len(self.image_files(patch))))

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
        files = self.image_files(patch)
        frame_index = max(0, min(int(frame_slot), len(files) - 1))
        return FrameRef(
            token=token,
            root=str(self.root),
            tract=self.tract,
            patch=patch,
            band="default",
            pack_path=str(files[frame_index]),
            tile_index=tile.tile_index,
            tile_id=tile.tile_id,
            x0=tile.x0,
            y0=tile.y0,
            x1=tile.x1,
            y1=tile.y1,
            frame_slot=frame_slot,
            frame_rank=frame_index,
            frame_index=frame_index,
            tile_length=len(files),
            visit=frame_index + 1,
            weight=None,
            scale=None,
            dataset=self.dataset,
        )

    def read_frame(self, ref: FrameRef) -> np.ndarray:
        image = self.read_image_file(ref.patch, ref.frame_index)
        return np.asarray(image[ref.y0 : ref.y1, ref.x0 : ref.x1], dtype=np.float32)

    def manifest(self, patch: str) -> dict[str, Any]:
        rows = []
        for idx, path in enumerate(self.image_files(patch)):
            manifest_path = path.with_name("stack_manifest.json")
            manifest = {}
            if manifest_path.is_file():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    manifest = {}
            rows.append({"frame_index": idx, "path": str(path), "manifest": manifest})
        return {"patch": patch, "frames": rows}
