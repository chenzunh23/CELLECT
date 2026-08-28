from __future__ import annotations

import io
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/{os.environ.get('USER', str(os.getuid()))}_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from PIL import Image, ImageDraw, ImageFont


DEFAULT_DATA_QUALITY_BANDS = (
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
DEFAULT_WEIGHTS = {"NO_DATA": 1.0, "UNMASKEDNAN": 1.0, "INTRP": 0.3, "EDGE": 0.1, "BAD": 0.5}
SOURCE_ORDER = ("coadd", "noisy", "denoised")
PATCH_COUNT = 9
PAGE_PATCHES = 3
TILE_OVERLAY_COLORS = {
    "13_20": (245, 220, 35, 76),
    "20_50": (255, 125, 20, 90),
    "gt_50": (115, 10, 24, 108),
}


@dataclass(frozen=True)
class DataQualitySource:
    key: str
    label: str
    root: Path
    pattern: str
    uses_group: bool


@dataclass(frozen=True)
class PatchScore:
    patch: str
    source: str
    path: str | None
    status: str
    score: float | None
    shape: tuple[int, int] | None


def patch_sort_key(patch: str) -> tuple[int, int]:
    x, y = patch.split(",", 1)
    return int(x), int(y)


def mask_planes(header: fits.Header) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in header.items():
        name = str(key).upper()
        if re.fullmatch(r"MP_\d+", name):
            try:
                out[str(value).strip().upper()] = int(name.split("_", 1)[1])
            except Exception:
                pass
        elif name.startswith("MP_") and not name[3:].isdigit():
            try:
                out[name[3:].strip().upper()] = int(value)
            except Exception:
                pass
    return out


def fits_image_mask(path: Path) -> tuple[np.ndarray, np.ndarray | None, dict[str, int]]:
    with fits.open(path, memmap=False) as hdul:
        image_hdu = hdul[1] if len(hdul) > 1 and getattr(hdul[1], "data", None) is not None else hdul[0]
        image = np.asarray(image_hdu.data, dtype=np.float32)
        if "MASK" in hdul:
            mask_hdu = hdul["MASK"]
        elif len(hdul) > 2 and getattr(hdul[2], "data", None) is not None:
            mask_hdu = hdul[2]
        else:
            return image, None, {}
        mask = np.asarray(mask_hdu.data, dtype=np.int64)
        planes = mask_planes(mask_hdu.header)
    return image, mask, planes


def score_map(mask: np.ndarray, planes: dict[str, int], edge_weight: float) -> np.ndarray:
    weights = dict(DEFAULT_WEIGHTS)
    weights["EDGE"] = float(edge_weight)
    out = np.zeros(mask.shape, dtype=np.float32)
    for plane, weight in weights.items():
        bit = planes.get(plane)
        if bit is None:
            continue
        selected = (mask & (1 << int(bit))) != 0
        out[selected] = np.maximum(out[selected], float(weight))
    return out


def source_path(source: DataQualitySource, tract: str, band: str, patch: str, group: str) -> Path:
    try:
        group_index = int(str(group).replace("group_", ""))
    except Exception:
        group_index = 0
    return source.root / source.pattern.format(
        tract=tract,
        band=band,
        patch=patch,
        patch_us=str(patch).replace(",", "_"),
        group=group_index,
        group2=f"{group_index:02d}",
    )


def safe_zscale(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    try:
        lo, hi = ZScaleInterval().get_limits(finite)
    except Exception:
        lo, hi = float(np.nanpercentile(finite, 0.5)), float(np.nanpercentile(finite, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.nan_to_num(arr, nan=lo, posinf=hi, neginf=lo)
    scaled = np.rint(255.0 * np.clip((arr - lo) / (hi - lo), 0.0, 1.0)).astype(np.uint8)
    return np.flipud(scaled)


def resize_gray(arr: np.ndarray, size: int) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="L").resize((size, size), Image.Resampling.BILINEAR)


def tile_overlay_color(score: float, threshold: float) -> tuple[int, int, int, int] | None:
    value = float(score)
    if value < float(threshold):
        return None
    if value < 0.2:
        return TILE_OVERLAY_COLORS["13_20"]
    if value < 0.5:
        return TILE_OVERLAY_COLORS["20_50"]
    return TILE_OVERLAY_COLORS["gt_50"]


def score_regular_tiles_from_map(smap: np.ndarray, *, tile_size: int, stride: int) -> list[dict[str, float]]:
    h, w = np.asarray(smap).shape
    rows: list[dict[str, float]] = []
    if h < tile_size or w < tile_size:
        return rows
    for y0 in range(0, h - tile_size + 1, stride):
        for x0 in range(0, w - tile_size + 1, stride):
            rows.append({
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x0 + tile_size),
                "y1": float(y0 + tile_size),
                "score": float(np.nanmean(smap[y0 : y0 + tile_size, x0 : x0 + tile_size])),
            })
    return rows


def draw_tile_overlays(image: Image.Image, tile_rows: list[dict[str, float]], *, source_shape: tuple[int, int], threshold: float) -> None:
    h, w = int(source_shape[0]), int(source_shape[1])
    if h <= 0 or w <= 0:
        return
    draw = ImageDraw.Draw(image, "RGBA")
    out_w, out_h = image.size
    for row in tile_rows:
        color = tile_overlay_color(float(row["score"]), threshold)
        if color is None:
            continue
        x0 = int(round(float(row["x0"]) / w * out_w))
        x1 = int(round(float(row["x1"]) / w * out_w))
        y0 = int(round((h - float(row["y1"])) / h * out_h))
        y1 = int(round((h - float(row["y0"])) / h * out_h))
        draw.rectangle((x0, y0, x1, y1), fill=color, outline=(255, 255, 255, 92), width=1)


def color_for_score(score: float | None, threshold: float) -> tuple[int, int, int]:
    if score is None or not np.isfinite(score):
        return (120, 124, 130)
    t = min(1.0, max(0.0, float(score) / max(threshold * 2.0, 1e-6)))
    if t < 0.5:
        u = t / 0.5
        a = np.array([40, 145, 80], dtype=np.float32)
        b = np.array([230, 205, 55], dtype=np.float32)
    else:
        u = (t - 0.5) / 0.5
        a = np.array([230, 205, 55], dtype=np.float32)
        b = np.array([210, 45, 50], dtype=np.float32)
    c = np.rint((1.0 - u) * a + u * b).astype(int)
    return int(c[0]), int(c[1]), int(c[2])


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int]) -> None:
    font = ImageFont.load_default()
    bbox = draw.textbbox(xy, text, font=font)
    box = (bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 4)
    draw.rectangle(box, fill=(0, 0, 0, 180))
    draw.text(xy, text, font=font, fill=fill)


class DataQualityPreview:
    def __init__(
        self,
        *,
        tract: str,
        bands: list[str],
        threshold: float,
        edge_weight: float,
        panel_size: int,
        tile_size: int,
        tile_stride: int,
        coadd_root: Path,
        noisy_root: Path,
        denoised_fits_root: Path,
        groups: list[str],
    ) -> None:
        self.tract = str(tract)
        self.bands = [band for band in bands if band in DEFAULT_DATA_QUALITY_BANDS]
        self.threshold = float(threshold)
        self.edge_weight = float(edge_weight)
        self.panel_size = int(panel_size)
        self.tile_size = int(tile_size)
        self.tile_stride = int(tile_stride)
        self.groups = [str(group) for group in groups]
        self.sources = {
            "coadd": DataQualitySource("coadd", "coadd", Path(coadd_root).expanduser(), "{tract}/{band}/{patch}/warp_half-{band}-{tract}-{patch}-0.fits", False),
            "noisy": DataQualitySource("noisy", "noisy", Path(noisy_root).expanduser(), "{tract}/{band}/{patch}/warp_8-{band}-{tract}-{patch}-{group}.fits", True),
            "denoised": DataQualitySource("denoised", "denoised", Path(denoised_fits_root).expanduser(), "patch_{patch_us}/group_{group2}/{band}/noisy.fits", True),
        }
        self.score_cache: dict[tuple[str, str, str], list[PatchScore]] = {}
        self.image_cache: dict[tuple[Any, ...], bytes] = {}
        self.lock = threading.Lock()

    def patches(self) -> list[str]:
        return [f"{x},{y}" for x in range(PATCH_COUNT) for y in range(PATCH_COUNT)]

    @staticmethod
    def page_for_patch(patch: str) -> int:
        x, y = patch_sort_key(patch)
        return int(y // PAGE_PATCHES) * PAGE_PATCHES + int(x // PAGE_PATCHES)

    def available_bands(self, source_key: str) -> list[str]:
        source = self.sources[source_key]
        out = []
        for band in self.bands:
            groups = self.groups if source.uses_group else ["0"]
            if any(source_path(source, self.tract, band, patch, group).exists() for patch in self.patches() for group in groups):
                out.append(band)
        return out

    def scores(self, source_key: str, band: str, group: str) -> list[PatchScore]:
        source = self.sources[source_key]
        if not source.uses_group:
            group = "0"
        key = (source_key, band, group)
        with self.lock:
            cached = self.score_cache.get(key)
        if cached is not None:
            return cached
        rows: list[PatchScore] = []
        for patch in self.patches():
            path = source_path(source, self.tract, band, patch, group)
            if not path.exists():
                rows.append(PatchScore(patch, source_key, str(path), "missing", None, None))
                continue
            try:
                image, mask, planes = fits_image_mask(path)
                if mask is None:
                    rows.append(PatchScore(patch, source_key, str(path), "ok", None, tuple(image.shape)))
                else:
                    smap = score_map(mask, planes, self.edge_weight)
                    rows.append(PatchScore(patch, source_key, str(path), "ok", float(np.mean(smap)), tuple(mask.shape)))
            except Exception as exc:
                rows.append(PatchScore(patch, source_key, str(path), f"error: {exc}", None, None))
        with self.lock:
            self.score_cache[key] = rows
        return rows

    def patch_panel(self, source_key: str, band: str, group: str, patch: str, overlay: bool) -> Image.Image:
        source = self.sources[source_key]
        if not source.uses_group:
            group = "0"
        size = self.panel_size
        path = source_path(source, self.tract, band, patch, group)
        if not path.exists():
            img = Image.new("RGB", (size, size), (35, 38, 42))
            draw_label(ImageDraw.Draw(img, "RGBA"), (10, 10), f"{patch}\nmissing", (230, 230, 230))
            return img
        try:
            image, mask, planes = fits_image_mask(path)
            base = resize_gray(safe_zscale(image), size).convert("RGB")
            score = None
            dropped = False
            if mask is not None:
                smap = score_map(mask, planes, self.edge_weight)
                score = float(np.mean(smap))
                dropped = score >= self.threshold
                if overlay and not dropped:
                    draw_tile_overlays(base, score_regular_tiles_from_map(smap, tile_size=self.tile_size, stride=self.tile_stride), source_shape=smap.shape, threshold=self.threshold)
            draw = ImageDraw.Draw(base, "RGBA")
            draw_label(draw, (10, 10), f"{patch}\n{source_key}\n{'n/a' if score is None else f'{score:.4f}'}", (255, 80, 80) if dropped else (180, 255, 190))
            if dropped:
                draw.rectangle((2, 2, size - 3, size - 3), outline=(255, 35, 35, 255), width=5)
            return base
        except Exception as exc:
            img = Image.new("RGB", (size, size), (44, 36, 36))
            draw = ImageDraw.Draw(img, "RGBA")
            draw_label(draw, (10, 10), f"{patch}\nerror", (255, 150, 150))
            draw.text((10, 50), str(exc)[:80], fill=(255, 150, 150), font=ImageFont.load_default())
            return img

    def page_png(self, source_key: str, band: str, group: str, page: int, overlay: bool) -> bytes:
        key = ("page", source_key, band, group, int(page), bool(overlay), self.edge_weight, self.panel_size, self.threshold, self.tile_size, self.tile_stride)
        with self.lock:
            cached = self.image_cache.get(key)
        if cached is not None:
            return cached
        px = int(page) % 3
        py = int(page) // 3
        size = self.panel_size
        gutter = 8
        title_h = 34
        out = Image.new("RGB", (PAGE_PATCHES * size + (PAGE_PATCHES + 1) * gutter, PAGE_PATCHES * size + (PAGE_PATCHES + 1) * gutter + title_h), (18, 20, 23))
        draw = ImageDraw.Draw(out, "RGBA")
        suffix = f" group {group}" if self.sources[source_key].uses_group else ""
        title = f"{source_key} {band}{suffix} page {page + 1}/9 EDGE={self.edge_weight:g} threshold={self.threshold:g}"
        draw.text((gutter, 9), title, fill=(235, 238, 242), font=ImageFont.load_default())
        for row in range(PAGE_PATCHES):
            for col in range(PAGE_PATCHES):
                x = px * PAGE_PATCHES + col
                y = py * PAGE_PATCHES + (PAGE_PATCHES - 1 - row)
                patch = f"{x},{y}"
                out.paste(self.patch_panel(source_key, band, group, patch, overlay), (gutter + col * (size + gutter), title_h + gutter + row * (size + gutter)))
        buf = io.BytesIO()
        out.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()
        with self.lock:
            self.image_cache[key] = data
        return data

    def overview_png(self, source_key: str, band: str, group: str) -> bytes:
        key = ("overview", source_key, band, group, self.edge_weight, self.threshold)
        with self.lock:
            cached = self.image_cache.get(key)
        if cached is not None:
            return cached
        rows = {row.patch: row for row in self.scores(source_key, band, group)}
        cell = 110
        pad = 42
        title_h = 42
        img = Image.new("RGB", (pad * 2 + PATCH_COUNT * cell, title_h + pad + PATCH_COUNT * cell + pad), (18, 20, 23))
        draw = ImageDraw.Draw(img, "RGBA")
        suffix = f" group {group}" if self.sources[source_key].uses_group else ""
        draw.text((pad, 14), f"{source_key} {band}{suffix} EDGE={self.edge_weight:g} threshold={self.threshold:g}", fill=(235, 238, 242), font=ImageFont.load_default())
        for y in range(PATCH_COUNT - 1, -1, -1):
            for x in range(PATCH_COUNT):
                patch = f"{x},{y}"
                row = rows.get(patch)
                score = row.score if row else None
                left = pad + x * cell
                top = title_h + (PATCH_COUNT - 1 - y) * cell
                draw.rectangle((left, top, left + cell - 3, top + cell - 3), fill=color_for_score(score, self.threshold) + (255,), outline=(42, 47, 54, 255), width=2)
                text = f"{patch}\n" + ("missing" if row is None else ("n/a" if row.score is None else f"{row.score:.4f}"))
                draw.multiline_text((left + 10, top + 10), text, fill=(255, 255, 255), font=ImageFont.load_default(), spacing=4)
                if score is not None and score >= self.threshold:
                    draw.rectangle((left + 3, top + 3, left + cell - 6, top + cell - 6), outline=(255, 255, 255, 255), width=4)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()
        with self.lock:
            self.image_cache[key] = data
        return data

    def drops_json(self, source_key: str, band: str, group: str) -> dict[str, Any]:
        rows = self.scores(source_key, band, group)
        ok = [row for row in rows if row.status == "ok"]
        drops = [row for row in ok if row.score is not None and float(row.score) >= self.threshold]
        drops.sort(key=lambda row: patch_sort_key(row.patch))
        pages_with_data = sorted({self.page_for_patch(row.patch) for row in ok})
        return {
            "source": source_key,
            "band": band,
            "group": group,
            "group_note": f" group {group}" if self.sources[source_key].uses_group else "",
            "threshold": self.threshold,
            "edge_weight": self.edge_weight,
            "ok_count": len(ok),
            "drop_count": len(drops),
            "pages_with_data": pages_with_data,
            "drops": [{"patch": row.patch, "score": row.score, "path": row.path} for row in drops],
            "errors": [{"patch": row.patch, "status": row.status, "path": row.path} for row in rows if row.status != "ok"],
        }

    def meta_json(self) -> dict[str, Any]:
        return {
            "sources": [{"id": key, "label": source.label, "uses_group": source.uses_group} for key, source in self.sources.items()],
            "bands": list(self.bands),
            "source_bands": {key: self.available_bands(key) for key in self.sources},
            "groups": list(self.groups),
            "threshold": self.threshold,
            "edge_weight": self.edge_weight,
            "tile_size": self.tile_size,
            "tile_stride": self.tile_stride,
            "pages": 9,
        }
