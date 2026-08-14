from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
from astropy.visualization import ZScaleInterval
from PIL import Image, ImageDraw, ImageFont
import torch


DEFAULT_CELLECT_ROOT = Path(__file__).resolve().parents[2]
CELLECT_ROOT = Path(os.environ.get("CELLECT_ROOT", str(DEFAULT_CELLECT_ROOT))).expanduser().resolve()
if str(CELLECT_ROOT) not in sys.path:
    sys.path.insert(0, str(CELLECT_ROOT))

from eval.datasets import (  # noqa: E402
    DEFAULT_HSC_RAW_BANDS,
    DEFAULT_HSC_RAW_ROOT,
    DEFAULT_MESSIER_ROOT,
    FrameRef,
    HscRawAccess,
    MessierAccess,
)
from eval.datasets.base import patch_sort_key  # noqa: E402
from eval.eval_utils import (  # noqa: E402
    detection_rows,
    draw_ellipses,
    infer_cellect,
    load_cellect_model,
    make_training_rgb,
    select_band_outputs,
)


DEFAULT_ROOT = DEFAULT_HSC_RAW_ROOT
DEFAULT_BANDS = DEFAULT_HSC_RAW_BANDS
DEFAULT_CHECKPOINT = Path("/data/czh23/ckpts/sam_anscombe_0803/epoch_0030.pt")


PAGES_DIR = Path(__file__).with_name("pages")
HTML_PATH = PAGES_DIR / "index.html"
ASSETS_DIR = CELLECT_ROOT / "eval" / "assets"


def _slug_text(text: str, *, fallback: str = "run") -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text).strip())
    clean = "_".join(part for part in clean.split("_") if part)
    return clean or fallback


def _session_name(run_name: str, stamp: str) -> str:
    run_name = str(run_name or "").strip()
    if not run_name:
        return stamp
    return f"{_slug_text(run_name)}_{stamp}"


def _parse_tile_xy(tile_id: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"x(\d+)_y(\d+)", str(tile_id))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _tile_id_candidates(text: str, y: int | None = None) -> list[str]:
    raw = str(text or "").strip()
    values: list[tuple[int, int]] = []
    match = re.fullmatch(r"x(\d+)_y(\d+)", raw)
    if match:
        values.append((int(match.group(1)), int(match.group(2))))
    elif "," in raw:
        parts = [part.strip() for part in raw.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            values.append((int(parts[0]), int(parts[1])))
    elif y is not None and raw:
        values.append((int(raw), int(y)))
    out: list[str] = []
    for x_value, y_value in values:
        for candidate in (f"x{x_value}_y{y_value}", f"x{x_value:03d}_y{y_value:03d}"):
            if candidate not in out:
                out.append(candidate)
    if raw and raw not in out:
        out.insert(0, raw)
    return out


def _tile_heat_color(count: int) -> tuple[int, int, int]:
    count = int(count)
    if count <= 0:
        return (178, 181, 176)
    level = min(15, max(1, count))
    t = (level - 1) / 14.0 if level > 1 else 0.0
    if t <= 0.5:
        u = t / 0.5
        c0 = np.array([35, 165, 95], dtype=np.float32)
        c1 = np.array([245, 225, 40], dtype=np.float32)
    else:
        u = (t - 0.5) / 0.5
        c0 = np.array([245, 225, 40], dtype=np.float32)
        c1 = np.array([215, 55, 45], dtype=np.float32)
    color = np.rint((1.0 - u) * c0 + u * c1).astype(int)
    return int(color[0]), int(color[1]), int(color[2])


def load_html() -> bytes:
    return HTML_PATH.read_bytes()


def display_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    try:
        lo, hi = ZScaleInterval().get_limits(finite)
    except Exception:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = np.clip((np.nan_to_num(arr, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
    return np.flipud(np.rint(255.0 * scaled).astype(np.uint8))


def _smooth_float_image(image: np.ndarray, sigma: float) -> np.ndarray:
    sigma = float(sigma)
    arr = np.asarray(image, dtype=np.float32)
    if sigma <= 0.0:
        return arr
    finite = np.isfinite(arr)
    fill = float(np.nanmedian(arr[finite])) if np.any(finite) else 0.0
    arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill).astype(np.float32, copy=False)
    radius = max(1, int(np.ceil(2.0 * sigma)))
    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (axis / sigma) ** 2)
    kernel /= np.sum(kernel)

    def convolve_axis(values: np.ndarray, axis_index: int) -> np.ndarray:
        pad = [(0, 0)] * values.ndim
        pad[axis_index] = (radius, radius)
        padded = np.pad(values, pad, mode="reflect")
        return np.apply_along_axis(lambda line: np.convolve(line, kernel, mode="valid"), axis_index, padded)

    out = convolve_axis(arr, 1)
    out = convolve_axis(out, 0)
    return out.astype(np.float32, copy=False)


def _boxcar_float_image(image: np.ndarray, radius: int) -> np.ndarray:
    radius = int(radius)
    arr = np.asarray(image, dtype=np.float32)
    if radius <= 0:
        return arr
    finite = np.isfinite(arr)
    fill = float(np.nanmedian(arr[finite])) if np.any(finite) else 0.0
    arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill).astype(np.float32, copy=False)
    size = 2 * radius + 1
    padded = np.pad(arr, ((radius, radius), (radius, radius)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    total = integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]
    return (total / float(size * size)).astype(np.float32, copy=False)


def _tophat_float_image(image: np.ndarray, radius: int) -> np.ndarray:
    radius = int(radius)
    arr = np.asarray(image, dtype=np.float32)
    if radius <= 0:
        return arr
    finite = np.isfinite(arr)
    fill = float(np.nanmedian(arr[finite])) if np.any(finite) else 0.0
    arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill).astype(np.float32, copy=False)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = ((xx * xx + yy * yy) <= radius * radius).astype(np.float32)
    denom = float(mask.sum())
    padded = np.pad(arr, ((radius, radius), (radius, radius)), mode="reflect")
    out = np.zeros_like(arr, dtype=np.float32)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            weight = float(mask[dy, dx])
            if weight:
                out += weight * padded[dy : dy + arr.shape[0], dx : dx + arr.shape[1]]
    return (out / denom).astype(np.float32, copy=False)


def _display_filter_image(
    image: np.ndarray,
    *,
    smooth_mode: str = "none",
    smooth_sigma: float = 1.0,
    smooth_radius: int = 1,
) -> np.ndarray:
    mode = str(smooth_mode or "none").replace("_", "-").lower()
    if mode in {"", "none", "off"}:
        return np.asarray(image, dtype=np.float32)
    if mode == "gaussian":
        return _smooth_float_image(image, float(smooth_sigma))
    if mode == "boxcar":
        return _boxcar_float_image(image, int(smooth_radius))
    if mode == "tophat":
        return _tophat_float_image(image, int(smooth_radius))
    raise ValueError(f"unknown smoothing mode: {smooth_mode}")


def _png_bytes(rgb_or_gray: np.ndarray) -> bytes:
    arr = np.asarray(rgb_or_gray)
    handle = io.BytesIO()
    if arr.ndim == 2:
        image = Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
    else:
        image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
    image.save(handle, format="PNG", optimize=True)
    return handle.getvalue()


def _overlay_png_bytes(image: np.ndarray, rows: list[dict[str, float]], *, draw_centers: bool) -> bytes:
    rgb = draw_ellipses(image, rows, draw_centers=draw_centers, point_color="yellow")
    arr = np.clip(np.rint(np.flipud(rgb) * 255.0), 0, 255).astype(np.uint8)
    return _png_bytes(arr)


def _overlay_uint8(image: np.ndarray, rows: list[dict[str, float]], *, draw_centers: bool) -> np.ndarray:
    rgb = draw_ellipses(image, rows, draw_centers=draw_centers, point_color="yellow")
    return np.clip(np.rint(np.flipud(rgb) * 255.0), 0, 255).astype(np.uint8)


def _input_overlay_channel(scaled: np.ndarray, scaling: str) -> tuple[int, np.ndarray] | None:
    label = str(scaling).replace("_", "-").lower()
    arr = np.asarray(scaled, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] <= 0:
        return None
    if "log-lupton" in label or "lupton" in label:
        idx = min(2, arr.shape[0] - 1)
        return idx, arr[idx]
    if "anscombe" in label or "zscore" in label:
        return 0, arr[0]
    return None


def _input_shape_overlay_png_bytes(
    scaled: np.ndarray,
    rows: list[dict[str, float]],
    *,
    scaling: str,
    clip_threshold: float,
    draw_centers: bool,
) -> bytes:
    selection = _input_overlay_channel(scaled, scaling)
    if selection is None:
        return _png_bytes(display_gray(np.asarray(scaled[0] if np.asarray(scaled).ndim == 3 else scaled)))
    channel_idx, channel = selection
    rgb = draw_ellipses(
        channel,
        rows,
        draw_centers=draw_centers,
        point_color="yellow",
        input_scaled_background=True,
        input_scaling=scaling,
        input_channel_index=channel_idx,
        input_clip_threshold=float(clip_threshold),
    )
    arr = np.clip(np.rint(np.flipud(rgb) * 255.0), 0, 255).astype(np.uint8)
    return _png_bytes(arr)


def _input_display_png_bytes(
    scaled: np.ndarray,
    *,
    scaling: str,
    clip_threshold: float,
) -> bytes:
    return _input_shape_overlay_png_bytes(
        scaled,
        [],
        scaling=scaling,
        clip_threshold=float(clip_threshold),
        draw_centers=False,
    )


def _input_shape_overlay_uint8(
    scaled: np.ndarray,
    rows: list[dict[str, float]],
    *,
    scaling: str,
    clip_threshold: float,
    draw_centers: bool,
    smooth_mode: str = "none",
    smooth_sigma: float = 1.0,
    smooth_radius: int = 1,
) -> np.ndarray:
    selection = _input_overlay_channel(scaled, scaling)
    if selection is None:
        return np.repeat(display_gray(np.asarray(scaled[0] if np.asarray(scaled).ndim == 3 else scaled))[..., None], 3, axis=2)
    channel_idx, channel = selection
    channel = _display_filter_image(
        channel,
        smooth_mode=smooth_mode,
        smooth_sigma=smooth_sigma,
        smooth_radius=smooth_radius,
    )
    rgb = draw_ellipses(
        channel,
        rows,
        draw_centers=draw_centers,
        point_color="yellow",
        input_scaled_background=True,
        input_scaling=scaling,
        input_channel_index=channel_idx,
        input_clip_threshold=float(clip_threshold),
    )
    return np.clip(np.rint(np.flipud(rgb) * 255.0), 0, 255).astype(np.uint8)


def _input_display_uint8(
    scaled: np.ndarray,
    *,
    scaling: str,
    clip_threshold: float,
    smooth_mode: str = "none",
    smooth_sigma: float = 1.0,
    smooth_radius: int = 1,
) -> np.ndarray:
    return _input_shape_overlay_uint8(
        scaled,
        [],
        scaling=scaling,
        clip_threshold=float(clip_threshold),
        draw_centers=False,
        smooth_mode=smooth_mode,
        smooth_sigma=smooth_sigma,
        smooth_radius=smooth_radius,
    )


def _draw_centers_on_uint8(image_rgb: np.ndarray, rows: list[dict[str, float]], *, radius: int = 5) -> np.ndarray:
    arr = np.asarray(image_rgb).copy()
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    height, width = arr.shape[:2]
    pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(pil)
    color = (255, 230, 0)
    for row in rows:
        x = int(round(float(row.get("x", 0.0))))
        y = height - 1 - int(round(float(row.get("y", 0.0))))
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        draw.line((x - radius, y, x + radius, y), fill=color, width=1)
        draw.line((x, y - radius, x, y + radius), fill=color, width=1)
    return np.asarray(pil, dtype=np.uint8)


def _save_titled_png(path: Path, image_rgb: np.ndarray, title: str, *, min_image_size: int = 512) -> None:
    arr = np.asarray(image_rgb)
    if arr.ndim == 2:
        pil = Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
    else:
        pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
    scale = max(1, int(np.ceil(float(min_image_size) / float(max(pil.size)))))
    if scale > 1:
        pil = pil.resize((pil.width * scale, pil.height * scale), Image.Resampling.NEAREST)
    title_suffix = f" ({scale}x)" if scale > 1 else ""
    font = ImageFont.load_default()
    title_text = f"{title}{title_suffix}"
    title_h = 30
    canvas = Image.new("RGB", (pil.width, pil.height + title_h), "white")
    canvas.paste(pil, (0, title_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title_text, fill=(20, 20, 20), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def dataset_label(dataset_id: str) -> str:
    return {
        "hsc_raw": "HSC raw tiles",
        "sitian": "Sitian",
        "hsc_image": "HSC coadd/noisy/denoised",
        "ztf": "ZTF",
    }.get(str(dataset_id), str(dataset_id))


def make_access(dataset_id: str, args: argparse.Namespace, tract: str):
    dataset_id = str(dataset_id)
    if dataset_id == "hsc_raw":
        return HscRawAccess(Path(args.root), tract)
    if dataset_id == "sitian":
        return MessierAccess(Path(args.messier_root), tract, selection_mode=str(args.messier_tile_mode))
    if dataset_id in {"hsc_image", "ztf"}:
        raise NotImplementedError(f"{dataset_label(dataset_id)} dataset is a placeholder in this browser")
    raise KeyError(f"unknown dataset: {dataset_id}")


class BrowserState:
    def __init__(
        self,
        args: argparse.Namespace,
        *,
        tract: str,
        patches: list[str],
        bands: list[str],
        n_tiles: int | None,
        all_tiles: bool,
        frames_per_tile: int | None = None,
        tiles_per_page: int | None = None,
        run_name: str = "",
        stamp: str | None = None,
    ) -> None:
        self.run_name = str(run_name or "")
        self.stamp = str(stamp or time.strftime("%Y%m%d_%H%M%S"))
        self.session_name = _session_name(self.run_name, self.stamp)
        self.dataset_id = str(getattr(args, "dataset", "hsc_raw"))
        self.args_tile_selection_mode = str(args.messier_tile_mode) if self.dataset_id == "sitian" else "default"
        self.access = make_access(self.dataset_id, args, tract)
        self.root = Path(getattr(self.access, "root", args.root)).expanduser().resolve()
        self.tract = str(tract)
        self.patches = sorted([str(value) for value in patches], key=patch_sort_key)
        self.patch = self.patches[0]
        self.bands = [str(value) for value in bands]
        default_n_tiles = int(args.messier_n_tiles) if self.dataset_id == "sitian" else int(args.n_tiles)
        self.n_tiles_requested = None if all_tiles else int(n_tiles if n_tiles is not None else default_n_tiles)
        self.all_tiles = bool(all_tiles)
        self.frames_per_tile = max(1, int(frames_per_tile if frames_per_tile is not None else args.frames_per_tile))
        self.tiles_per_page = max(1, int(tiles_per_page if tiles_per_page is not None else args.tiles_per_page))
        self.detect_batch_size = max(1, int(args.detect_batch_size))
        self.seed = int(args.seed)
        self.visit = int(args.visit) if args.visit is not None else None
        self.frame_rank = int(args.frame_rank)
        self.strict_visit = bool(args.strict_visit)
        self.checkpoint = Path(args.checkpoint).expanduser().resolve()
        self.config = Path(args.config).expanduser().resolve() if args.config else None
        self.device_name = str(args.device)
        self.amp = str(args.amp)
        self.scaling_mode = str(args.scaling_mode)
        self.clip_threshold = float(args.clip_threshold)
        self.log_a = float(args.log_a)
        self.log_high_percentile = float(args.log_high_percentile)
        self.lupton_stretch = float(args.lupton_stretch)
        self.lupton_q = float(args.lupton_q)
        self.anscombe_clip = bool(args.anscombe_clip)
        self.anscombe_scale = float(args.anscombe_scale)
        self.confidence_threshold = float(args.confidence_threshold)
        self.confidence_score = str(args.confidence_score)
        self.nms_radius = int(args.nms_radius)
        self.center_refinement = str(args.center_refinement)
        self.center_refinement_radius = int(args.center_refinement_radius)
        self.shape_overlay_centers = bool(args.shape_overlay_centers)
        self.session_dir = Path(args.session_dir).expanduser().resolve()
        self.export_dir = Path(args.export_dir).expanduser().resolve()
        self.selected: set[str] = set()
        self.warnings: list[str] = []
        self.bands_by_patch: dict[str, list[str]] = {}
        self.selected_tiles_by_patch: dict[str, list[str]] = {}
        self.frame_slots_by_patch: dict[str, dict[str, int]] = {}
        self.refs_by_patch: dict[str, list[FrameRef]] = {}
        self.ref_by_token: dict[str, FrameRef] = {}
        self.input_png_by_token: dict[str, bytes] = {}
        self.detect_png_by_token: dict[str, bytes] = {}
        self.input_shape_png_by_token: dict[str, bytes] = {}
        self.detect_rows_by_token: dict[str, list[dict[str, float]]] = {}
        self.tile_map_png_by_patch: dict[str, bytes] = {}
        self._model_by_bands: dict[tuple[str, ...], tuple[torch.nn.Module, dict[str, Any]]] = {}
        self._model_lock = threading.Lock()
        for patch in self.patches:
            self.selected_tiles_by_patch[patch] = self._choose_tiles(patch, self.seed + 1009 * len(self.selected_tiles_by_patch))
        self._build_refs()
        self._build_tile_maps()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

    def _choose_tiles(self, patch: str, seed: int) -> list[str]:
        bands = self._usable_bands_for_patch(patch)
        self.bands_by_patch[patch] = bands
        if not bands:
            self.warnings.append(f"{patch}: no requested bands are available; page will be empty")
            return []
        try:
            tiles = self.access.choose_tiles(
                patch,
                bands,
                n_tiles=self.n_tiles_requested,
                all_tiles=self.all_tiles,
                seed=seed,
                mode=self.args_tile_selection_mode,
            )
        except Exception as exc:
            self.warnings.append(f"{patch}: tile selection failed: {exc}")
            print(f"WARNING: {patch}: tile selection failed: {exc}", flush=True)
            return []
        if not tiles:
            self.warnings.append(f"{patch}: no common tiles for available bands {bands}; page will be empty")
        return tiles

    def _usable_bands_for_patch(self, patch: str) -> list[str]:
        if self.dataset_id == "hsc_raw" and isinstance(self.access, HscRawAccess):
            usable = []
            for band in self.bands:
                try:
                    valid = self.access.valid_tiles_for_band(band, patch)
                except Exception as exc:
                    self.warnings.append(f"{patch}: skip unavailable {band}: {exc}")
                    print(f"WARNING: {patch}: skip unavailable {band}: {exc}", flush=True)
                    continue
                if valid:
                    usable.append(band)
                else:
                    self.warnings.append(f"{patch}: skip empty {band}")
                    print(f"WARNING: {patch}: skip empty {band}", flush=True)
            return usable
        return list(self.bands)

    def _bands_for_patch(self, patch: str | None = None) -> list[str]:
        return self.bands_by_patch.get(str(patch or self.patch), list(self.bands))

    def _tile_slot_count(self, patch: str, tile_id: str) -> int:
        return self.access.tile_slot_count(
            patch,
            tile_id,
            self._bands_for_patch(patch),
            frames_per_tile=self.frames_per_tile,
            visit=self.visit,
        )

    def _build_refs(self) -> None:
        token = 0
        for patch in self.patches:
            refs: list[FrameRef] = []
            self.frame_slots_by_patch[patch] = {}
            bands = self._bands_for_patch(patch)
            for tile_id in self.selected_tiles_by_patch[patch]:
                slot_count = self._tile_slot_count(patch, tile_id)
                self.frame_slots_by_patch[patch][tile_id] = slot_count
                for frame_slot in range(slot_count):
                    for band in bands:
                        try:
                            ref = self.access.make_ref(
                                token=f"P{token:05d}",
                                patch=patch,
                                band=band,
                                tile_id=tile_id,
                                frame_slot=frame_slot,
                                frame_rank=self.frame_rank,
                                frames_per_tile=self.frames_per_tile,
                                visit=self.visit,
                                strict_visit=self.strict_visit,
                            )
                        except Exception as exc:
                            message = f"{patch} {tile_id} slot {frame_slot + 1}: skip {band}: {exc}"
                            self.warnings.append(message)
                            print(f"WARNING: {message}", flush=True)
                            continue
                        refs.append(ref)
                        self.ref_by_token[ref.token] = ref
                        token += 1
            self.refs_by_patch[patch] = refs

    def _all_map_tile_ids(self, patch: str) -> list[str]:
        if self.dataset_id == "hsc_raw" and isinstance(self.access, HscRawAccess):
            try:
                sets = [self.access.valid_tiles_for_band(band, patch) for band in self._bands_for_patch(patch)]
                if sets:
                    return sorted(
                        set.intersection(*sets),
                        key=lambda text: tuple(int(v) for v in re.findall(r"\d+", text)),
                    )
            except Exception as exc:
                print(f"WARNING: failed to collect full tile map for {patch}; using selected tiles only: {exc}", flush=True)
        return list(self.selected_tiles_by_patch.get(patch, []))

    def _tile_map_counts(self, patch: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for tile_id in self._all_map_tile_ids(patch):
            counts[tile_id] = 0
        for tile_id, slot_count in self.frame_slots_by_patch.get(patch, {}).items():
            counts[tile_id] = int(slot_count)
        return counts

    def _make_tile_map_png(self, patch: str) -> bytes:
        counts = self._tile_map_counts(patch)
        parsed = {tile_id: xy for tile_id, xy in ((tile_id, _parse_tile_xy(tile_id)) for tile_id in counts) if xy is not None}
        if not parsed:
            return (ASSETS_DIR / "blank.png").read_bytes()

        xs = sorted({xy[0] for xy in parsed.values()})
        ys = sorted({xy[1] for xy in parsed.values()})
        x_index = {value: idx for idx, value in enumerate(xs)}
        y_index = {value: idx for idx, value in enumerate(ys)}
        nx = len(xs)
        ny = len(ys)
        cell = max(34, min(58, int(900 / max(nx, ny, 1))))
        left = 58
        right = 78
        top = 56
        bottom = 54
        width = left + nx * cell + right
        height = top + ny * cell + bottom
        image = Image.new("RGB", (width, height), (246, 246, 243))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        title = f"{dataset_label(self.dataset_id)} {self.tract}/{patch}: selected groups per tile"
        draw.text((left, 16), title, fill=(25, 25, 25), font=font)

        for tile_id, (tx, ty) in parsed.items():
            ix = x_index[tx]
            iy = y_index[ty]
            # Show tile y increasing upward, matching astronomical tile-map convention.
            py = ny - 1 - iy
            x0 = left + ix * cell
            y0 = top + py * cell
            count = int(counts.get(tile_id, 0))
            color = _tile_heat_color(count)
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color, outline=(92, 92, 88), width=1)
            text = str(count)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((x0 + (cell - tw) / 2, y0 + (cell - th) / 2), text, fill=(10, 10, 10), font=font)

        for tx in xs:
            ix = x_index[tx]
            x = left + ix * cell + cell / 2
            label = str(tx)
            bbox = draw.textbbox((0, 0), label, font=font)
            draw.text((x - (bbox[2] - bbox[0]) / 2, top + ny * cell + 8), label, fill=(35, 35, 35), font=font)
        for ty in ys:
            iy = y_index[ty]
            py = ny - 1 - iy
            y = top + py * cell + cell / 2
            label = str(ty)
            bbox = draw.textbbox((0, 0), label, font=font)
            draw.text((left - 10 - (bbox[2] - bbox[0]), y - (bbox[3] - bbox[1]) / 2), label, fill=(35, 35, 35), font=font)
        draw.text((left + nx * cell / 2 - 26, height - 24), "tile x", fill=(35, 35, 35), font=font)
        draw.text((10, top + ny * cell / 2 - 8), "tile y", fill=(35, 35, 35), font=font)

        cb_x0 = left + nx * cell + 26
        cb_y0 = top
        cb_w = 22
        cb_h = ny * cell
        for j in range(cb_h):
            t = 1.0 - j / max(1, cb_h - 1)
            level = int(round(t * 15))
            color = _tile_heat_color(level)
            draw.line((cb_x0, cb_y0 + j, cb_x0 + cb_w, cb_y0 + j), fill=color)
        draw.rectangle((cb_x0, cb_y0, cb_x0 + cb_w, cb_y0 + cb_h), outline=(60, 60, 60), width=1)
        draw.text((cb_x0 + cb_w + 6, cb_y0 - 2), "15+", fill=(35, 35, 35), font=font)
        draw.text((cb_x0 + cb_w + 6, cb_y0 + cb_h - 10), "0", fill=(35, 35, 35), font=font)

        path = self.session_dir / f"tile_map_{_slug_text(str(patch), fallback='patch')}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return path.read_bytes()

    def _build_tile_maps(self) -> None:
        for patch in self.patches:
            self.tile_map_png_by_patch[patch] = self._make_tile_map_png(patch)

    @property
    def refs(self) -> list[FrameRef]:
        return self.refs_by_patch.get(self.patch, [])

    @property
    def selected_tiles(self) -> list[str]:
        return self.selected_tiles_by_patch.get(self.patch, [])

    @property
    def n_pages(self) -> int:
        return max(1, int(np.ceil(len(self.selected_tiles) / self.tiles_per_page)))

    def _write_manifest(self) -> None:
        payload = {
            "root": str(self.root),
            "dataset": self.dataset_id,
            "dataset_label": dataset_label(self.dataset_id),
            "run_name": self.run_name,
            "session_name": self.session_name,
            "tract": self.tract,
            "patches": self.patches,
            "bands": self.bands,
            "bands_by_patch": self.bands_by_patch,
            "tile_size": int(getattr(self.access, "tile_size", 256)),
            "n_tiles_by_patch": {patch: len(values) for patch, values in self.selected_tiles_by_patch.items()},
            "frames_per_tile": self.frames_per_tile,
            "tiles_per_page": self.tiles_per_page,
            "detect_batch_size": self.detect_batch_size,
            "n_candidates_by_patch": {patch: len(values) for patch, values in self.refs_by_patch.items()},
            "selected_tile_ids_by_patch": self.selected_tiles_by_patch,
            "checkpoint": str(self.checkpoint),
            "scaling_mode": self.scaling_mode,
            "visit": self.visit,
            "note": f"{dataset_label(self.dataset_id)} frames; detection uses browser scaling={self.scaling_mode}.",
        }
        (self.session_dir / "browser_manifest.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def state_payload(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "dataset": self.dataset_id,
            "dataset_label": dataset_label(self.dataset_id),
            "run_name": self.run_name,
            "session_name": self.session_name,
            "tract": self.tract,
            "started": True,
            "patches": self.patches,
            "patch": self.patch,
            "bands": self._bands_for_patch(self.patch),
            "requested_bands": self.bands,
            "tile_size": int(getattr(self.access, "tile_size", 256)),
            "n_tiles": len(self.selected_tiles),
            "frames_per_tile": self.frames_per_tile,
            "tiles_per_page": self.tiles_per_page,
            "detect_batch_size": self.detect_batch_size,
            "n_candidates": len(self.refs),
            "n_selected": len(self.selected),
            "n_pages": self.n_pages,
            "checkpoint_name": self.checkpoint.name,
            "scaling_mode": self.scaling_mode,
            "visit": self.visit,
            "session_dir": str(self.session_dir),
            "export_dir": str(self.export_dir),
            "warnings": self.warnings[-20:],
        }

    def set_patch(self, patch: str) -> None:
        if patch not in self.refs_by_patch:
            raise KeyError(f"unknown patch: {patch}")
        self.patch = patch

    def _tile_summary(self, tile_id: str) -> dict[str, Any]:
        refs = [ref for ref in self.refs if ref.tile_id == tile_id]
        if not refs:
            raise KeyError(f"tile is not loaded in current patch: {tile_id}")
        ref = refs[0]
        index = self.selected_tiles.index(tile_id)
        return {
            "tile_id": tile_id,
            "tile_index": int(ref.tile_index),
            "x0": int(ref.x0),
            "y0": int(ref.y0),
            "x1": int(ref.x1),
            "y1": int(ref.y1),
            "page": int(index // self.tiles_per_page),
            "page_display": int(index // self.tiles_per_page + 1),
            "n_pages": int(self.n_pages),
        }

    def find_tile(self, mode: str, x: Any, y: Any = None) -> dict[str, Any]:
        mode = str(mode)
        if mode == "tile_xy":
            candidates = _tile_id_candidates(str(x), int(y) if y not in (None, "") else None)
            tile_id = next((value for value in candidates if value in self.selected_tiles), candidates[0])
            if tile_id not in self.selected_tiles:
                raise KeyError(
                    f"{' or '.join(candidates)} is not loaded in current patch; use max/all tiles if it was not sampled"
                )
            return self._tile_summary(tile_id)
        if mode == "pixel_xy":
            x = int(x)
            y = int(y)
            matches = []
            seen: set[str] = set()
            for ref in self.refs:
                if ref.tile_id in seen:
                    continue
                seen.add(ref.tile_id)
                if int(ref.x0) <= x < int(ref.x1) and int(ref.y0) <= y < int(ref.y1):
                    matches.append(ref.tile_id)
            if not matches:
                raise KeyError(
                    f"no loaded tile contains pixel ({x}, {y}) in current patch; use max/all tiles if it was not sampled"
                )
            matches = sorted(matches, key=lambda tile_id: self.selected_tiles.index(tile_id))
            return self._tile_summary(matches[0])
        raise ValueError(f"unknown search mode: {mode}")

    def page_payload(self, page_index: int) -> dict[str, Any]:
        page_index = max(0, min(page_index, self.n_pages - 1))
        start = page_index * self.tiles_per_page
        tile_ids = set(self.selected_tiles[start : start + self.tiles_per_page])
        refs = []
        for ref in self.refs:
            if ref.tile_id in tile_ids:
                row = ref.to_dict()
                row["selected"] = ref.token in self.selected
                row["image_url"] = f"/image/{ref.token}.png"
                row["input_image_url"] = f"/image/{ref.token}.png?input=1&v={len(self.input_png_by_token)}"
                row["detected"] = ref.token in self.detect_png_by_token
                row["detect_image_url"] = f"/image/{ref.token}.png?detect=1&v={len(self.detect_png_by_token)}"
                row["input_shape_detected"] = ref.token in self.input_shape_png_by_token
                row["input_shape_image_url"] = f"/image/{ref.token}.png?input_shape=1&detect=1&v={len(self.input_shape_png_by_token)}"
                row["n_detections"] = len(self.detect_rows_by_token.get(ref.token, ()))
                refs.append(row)
        return {"page": page_index, "n_pages": self.n_pages, "bands": self._bands_for_patch(self.patch), "tile_ids": sorted(tile_ids), "candidates": refs}

    def set_selected(self, tokens: list[str], selected: bool) -> None:
        for token in tokens:
            if token not in self.ref_by_token:
                raise KeyError(f"unknown token: {token}")
            if selected:
                self.selected.add(token)
            else:
                self.selected.discard(token)

    def selected_refs(self) -> list[FrameRef]:
        return [ref for ref in self.refs if ref.token in self.selected]

    def _scaled_input_for_token(self, token: str) -> np.ndarray:
        image = self.access.read_frame(self.ref_by_token[token])
        return make_training_rgb(
            image,
            mode=self.scaling_mode,
            clip_threshold=self.clip_threshold,
            log_a=self.log_a,
            log_high_percentile=self.log_high_percentile,
            lupton_stretch=self.lupton_stretch,
            lupton_q=self.lupton_q,
            anscombe_clip=self.anscombe_clip,
            anscombe_scale=self.anscombe_scale,
        )

    def image_png(
        self,
        token: str,
        *,
        detect: bool = False,
        input_image: bool = False,
        input_shape: bool = False,
        show_shape: bool = True,
        show_center: bool = False,
        smooth_mode: str = "none",
        smooth_sigma: float = 1.0,
        smooth_radius: int = 1,
    ) -> bytes:
        use_input = bool(input_image or input_shape)
        if detect and token in self.detect_rows_by_token:
            rows = self.detect_rows_by_token[token]
            if use_input:
                scaled = self._scaled_input_for_token(token)
                if show_shape:
                    arr = _input_shape_overlay_uint8(
                        scaled,
                        rows,
                        scaling=self.scaling_mode,
                        clip_threshold=self.clip_threshold,
                        draw_centers=show_center,
                        smooth_mode=smooth_mode,
                        smooth_sigma=smooth_sigma,
                        smooth_radius=smooth_radius,
                    )
                else:
                    arr = _input_display_uint8(
                        scaled,
                        scaling=self.scaling_mode,
                        clip_threshold=self.clip_threshold,
                        smooth_mode=smooth_mode,
                        smooth_sigma=smooth_sigma,
                        smooth_radius=smooth_radius,
                    )
                    if show_center:
                        arr = _draw_centers_on_uint8(arr, rows)
                return _png_bytes(arr)
            image = _display_filter_image(
                self.access.read_frame(self.ref_by_token[token]),
                smooth_mode=smooth_mode,
                smooth_sigma=smooth_sigma,
                smooth_radius=smooth_radius,
            )
            if show_shape:
                arr = _overlay_uint8(image, rows, draw_centers=show_center)
            else:
                arr = Image.fromarray(display_gray(image), mode="L").convert("RGB")
                arr = np.asarray(arr, dtype=np.uint8)
                if show_center:
                    arr = _draw_centers_on_uint8(arr, rows)
            return _png_bytes(arr)
        if input_image or input_shape:
            if smooth_mode in {"", "none", "off"} and token not in self.input_png_by_token:
                self.input_png_by_token[token] = _input_display_png_bytes(
                    self._scaled_input_for_token(token),
                    scaling=self.scaling_mode,
                    clip_threshold=self.clip_threshold,
                )
            if smooth_mode in {"", "none", "off"}:
                return self.input_png_by_token[token]
            return _png_bytes(
                _input_display_uint8(
                    self._scaled_input_for_token(token),
                    scaling=self.scaling_mode,
                    clip_threshold=self.clip_threshold,
                    smooth_mode=smooth_mode,
                    smooth_sigma=smooth_sigma,
                    smooth_radius=smooth_radius,
                )
            )
        image = _display_filter_image(
            self.access.read_frame(self.ref_by_token[token]),
            smooth_mode=smooth_mode,
            smooth_sigma=smooth_sigma,
            smooth_radius=smooth_radius,
        )
        return _png_bytes(display_gray(image))

    def raw_image_png(self, token: str) -> bytes:
        image = self.access.read_frame(self.ref_by_token[token])
        return _png_bytes(display_gray(image))

    def _load_model(self, bands: list[str] | None = None) -> tuple[torch.nn.Module, dict[str, Any]]:
        key = tuple(str(band) for band in (bands or self._bands_for_patch(self.patch)))
        with self._model_lock:
            if key not in self._model_by_bands:
                device = torch.device(self.device_name)
                model, cfg = load_cellect_model(
                    self.checkpoint,
                    self.config,
                    device,
                    list(key),
                    dynamic_image_size=True,
                )
                self._model_by_bands[key] = (model, dict(cfg or {}))
            model, cfg = self._model_by_bands[key]
            return model, dict(cfg or {})

    def _page_tile_slots(self, page_index: int) -> list[tuple[str, int]]:
        page_index = max(0, min(page_index, self.n_pages - 1))
        start = page_index * self.tiles_per_page
        tile_ids = self.selected_tiles[start : start + self.tiles_per_page]
        slot_counts = self.frame_slots_by_patch.get(self.patch, {})
        return [(tile_id, frame_slot) for tile_id in tile_ids for frame_slot in range(int(slot_counts.get(tile_id, 0)))]

    def _refs_by_tile_slot(self) -> dict[tuple[str, int], dict[str, FrameRef]]:
        refs_by_key: dict[tuple[str, int], dict[str, FrameRef]] = {}
        for ref in self.refs:
            refs_by_key.setdefault((ref.tile_id, ref.frame_slot), {})[ref.band] = ref
        return refs_by_key

    def _detect_tile_slot(
        self,
        tile_id: str,
        frame_slot: int,
        refs_by_key: dict[tuple[str, int], dict[str, FrameRef]] | None = None,
    ) -> dict[str, int]:
        refs_by_key = refs_by_key or self._refs_by_tile_slot()
        per_band = refs_by_key.get((tile_id, frame_slot), {})
        bands = [band for band in self._bands_for_patch(self.patch) if band in per_band]
        if not bands:
            return {"n_images": 0, "n_detections": 0}
        if all(per_band[band].token in self.detect_rows_by_token for band in bands):
            return {
                "n_images": len(bands),
                "n_detections": sum(len(self.detect_rows_by_token.get(per_band[band].token, ())) for band in bands),
            }
        try:
            model, _cfg = self._load_model(bands)
        except Exception as exc:
            message = f"{tile_id} slot {frame_slot + 1}: detection skipped for bands {bands}: {exc}"
            self.warnings.append(message)
            print(f"WARNING: {message}", flush=True)
            return {"n_images": 0, "n_detections": 0}
        device = torch.device(self.device_name)
        raw_images = [self.access.read_frame(per_band[band]) for band in bands]
        scaled = [
            make_training_rgb(
                image,
                mode=self.scaling_mode,
                clip_threshold=self.clip_threshold,
                log_a=self.log_a,
                log_high_percentile=self.log_high_percentile,
                lupton_stretch=self.lupton_stretch,
                lupton_q=self.lupton_q,
                anscombe_clip=self.anscombe_clip,
                anscombe_scale=self.anscombe_scale,
            )
            for image in raw_images
        ]
        tensor = torch.from_numpy(np.stack(scaled, axis=0).astype(np.float32, copy=False))[None]
        outputs = infer_cellect(model=model, image_tensor=tensor, device=device, amp=self.amp)
        total_images = 0
        total_detections = 0
        for band_idx, band in enumerate(bands):
            ref = per_band[band]
            rows = detection_rows(
                select_band_outputs(outputs, band_idx),
                threshold=self.confidence_threshold,
                nms_radius=self.nms_radius,
                confidence_score=self.confidence_score,
                center_refinement=self.center_refinement,
                center_refinement_radius=self.center_refinement_radius,
                width=per_band[band].width,
                height=per_band[band].height,
            )
            self.detect_rows_by_token[ref.token] = rows
            self.detect_png_by_token[ref.token] = _overlay_png_bytes(
                raw_images[band_idx],
                rows,
                draw_centers=self.shape_overlay_centers,
            )
            self.input_shape_png_by_token[ref.token] = _input_shape_overlay_png_bytes(
                scaled[band_idx],
                rows,
                scaling=self.scaling_mode,
                clip_threshold=self.clip_threshold,
                draw_centers=self.shape_overlay_centers,
            )
            total_images += 1
            total_detections += len(rows)
        return {"n_images": total_images, "n_detections": total_detections}

    @staticmethod
    def _select_sample_outputs(outputs: dict[str, torch.Tensor], sample_idx: int) -> dict[str, torch.Tensor]:
        selected = {}
        for key, value in outputs.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] > sample_idx:
                selected[key] = value[sample_idx : sample_idx + 1]
            else:
                selected[key] = value
        return selected

    def _detect_tile_slots(
        self,
        slots: list[tuple[str, int]],
        refs_by_key: dict[tuple[str, int], dict[str, FrameRef]] | None = None,
    ) -> dict[str, int]:
        refs_by_key = refs_by_key or self._refs_by_tile_slot()
        pending: list[tuple[str, int, dict[str, FrameRef]]] = []
        for tile_id, frame_slot in slots:
            per_band = refs_by_key.get((tile_id, frame_slot), {})
            bands = [band for band in self._bands_for_patch(self.patch) if band in per_band]
            if not bands:
                continue
            if all(per_band[band].token in self.detect_rows_by_token for band in bands):
                continue
            pending.append((tile_id, frame_slot, {band: per_band[band] for band in bands}))
        if pending:
            device = torch.device(self.device_name)
            grouped: dict[tuple[str, ...], list[tuple[str, int, dict[str, FrameRef]]]] = {}
            for item in pending:
                grouped.setdefault(tuple(item[2]), []).append(item)
            for bands_key, group_items in grouped.items():
                bands = list(bands_key)
                try:
                    model, _cfg = self._load_model(bands)
                except Exception as exc:
                    message = f"detection skipped for bands {bands}: {exc}"
                    self.warnings.append(message)
                    print(f"WARNING: {message}", flush=True)
                    continue
                for start in range(0, len(group_items), self.detect_batch_size):
                    chunk = group_items[start : start + self.detect_batch_size]
                    raw_by_sample: list[list[np.ndarray]] = []
                    scaled_by_sample = []
                    for _tile_id, _frame_slot, per_band in chunk:
                        raw_images = [self.access.read_frame(per_band[band]) for band in bands]
                        raw_by_sample.append(raw_images)
                        scaled_by_sample.append(
                            [
                                make_training_rgb(
                                    image,
                                    mode=self.scaling_mode,
                                    clip_threshold=self.clip_threshold,
                                    log_a=self.log_a,
                                    log_high_percentile=self.log_high_percentile,
                                    lupton_stretch=self.lupton_stretch,
                                    lupton_q=self.lupton_q,
                                    anscombe_clip=self.anscombe_clip,
                                    anscombe_scale=self.anscombe_scale,
                                )
                                for image in raw_images
                            ]
                        )
                    tensor = torch.from_numpy(np.stack(scaled_by_sample, axis=0).astype(np.float32, copy=False))
                    outputs = infer_cellect(model=model, image_tensor=tensor, device=device, amp=self.amp)
                    for sample_idx, (_tile_id, _frame_slot, per_band) in enumerate(chunk):
                        sample_outputs = self._select_sample_outputs(outputs, sample_idx)
                        for band_idx, band in enumerate(bands):
                            ref = per_band[band]
                            rows = detection_rows(
                                select_band_outputs(sample_outputs, band_idx),
                                threshold=self.confidence_threshold,
                                nms_radius=self.nms_radius,
                                confidence_score=self.confidence_score,
                                center_refinement=self.center_refinement,
                                center_refinement_radius=self.center_refinement_radius,
                                width=ref.width,
                                height=ref.height,
                            )
                            self.detect_rows_by_token[ref.token] = rows
                            self.detect_png_by_token[ref.token] = _overlay_png_bytes(
                                raw_by_sample[sample_idx][band_idx],
                                rows,
                                draw_centers=self.shape_overlay_centers,
                            )
                            self.input_shape_png_by_token[ref.token] = _input_shape_overlay_png_bytes(
                                scaled_by_sample[sample_idx][band_idx],
                                rows,
                                scaling=self.scaling_mode,
                                clip_threshold=self.clip_threshold,
                                draw_centers=self.shape_overlay_centers,
                            )
        total_images = 0
        total_detections = 0
        for tile_id, frame_slot in slots:
            per_band = refs_by_key.get((tile_id, frame_slot), {})
            bands = [band for band in self._bands_for_patch(self.patch) if band in per_band]
            if not bands:
                continue
            total_images += len(bands)
            total_detections += sum(len(self.detect_rows_by_token.get(per_band[band].token, ())) for band in bands)
        return {"n_images": total_images, "n_detections": total_detections}

    def detect_page(self, page_index: int) -> dict[str, Any]:
        refs_by_key = self._refs_by_tile_slot()
        return self._detect_tile_slots(self._page_tile_slots(page_index), refs_by_key)

    def write_selection_csv(self, out_dir: Path | None = None) -> Path:
        refs = self.selected_refs()
        if not refs:
            raise RuntimeError("no selected candidates")
        target_dir = (out_dir or self.session_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / "selection.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            rows = [ref.to_dict() for ref in refs]
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def export_selected(self, write_png: bool = True) -> dict[str, Any]:
        refs = self.selected_refs()
        if not refs:
            raise RuntimeError("no selected candidates")
        self.export_dir.mkdir(parents=True, exist_ok=True)
        selection_csv = self.write_selection_csv(self.export_dir)
        refs_by_key = self._refs_by_tile_slot()
        selected_slots = sorted({(ref.tile_id, ref.frame_slot) for ref in refs})
        self._detect_tile_slots(selected_slots, refs_by_key)
        manifest = []
        for ref in refs:
            image = self.access.read_frame(ref)
            out_dir = self.export_dir / ref.tract / ref.patch / ref.band / ref.tile_id
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = ref.candidate_id
            npz_path = out_dir / f"{stem}.npz"
            np.savez_compressed(npz_path, image=image)
            png_path = out_dir / f"{stem}.png"
            detect_png_path = out_dir / f"{stem}_detect_overlay.png"
            input_shape_png_path = out_dir / f"{stem}_input_shape_overlay.png"
            detect_csv_path = out_dir / f"{stem}_detections.csv"
            if write_png:
                Image.fromarray(display_gray(image), mode="L").convert("RGB").save(png_path)
            detect_rows = self.detect_rows_by_token.get(ref.token, [])
            fieldnames = sorted({key for item in detect_rows for key in item}) if detect_rows else [
                "x",
                "y",
                "score",
                "major",
                "minor",
                "theta",
            ]
            with detect_csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(detect_rows)
            if write_png:
                overlay = _overlay_uint8(image, detect_rows, draw_centers=self.shape_overlay_centers)
                _save_titled_png(
                    detect_png_path,
                    overlay,
                    f"{ref.candidate_id} detections={len(detect_rows)}",
                    min_image_size=512,
                )
                scaled = make_training_rgb(
                    image,
                    mode=self.scaling_mode,
                    clip_threshold=self.clip_threshold,
                    log_a=self.log_a,
                    log_high_percentile=self.log_high_percentile,
                    lupton_stretch=self.lupton_stretch,
                    lupton_q=self.lupton_q,
                    anscombe_clip=self.anscombe_clip,
                    anscombe_scale=self.anscombe_scale,
                )
                input_overlay = _input_shape_overlay_uint8(
                    scaled,
                    detect_rows,
                    scaling=self.scaling_mode,
                    clip_threshold=self.clip_threshold,
                    draw_centers=self.shape_overlay_centers,
                )
                _save_titled_png(
                    input_shape_png_path,
                    input_overlay,
                    f"{ref.candidate_id} input shape detections={len(detect_rows)}",
                    min_image_size=512,
                )
            row = {
                **ref.to_dict(),
                "npz_path": str(npz_path),
                "png_path": str(png_path) if write_png else "",
                "detect_png_path": str(detect_png_path) if write_png else "",
                "input_shape_png_path": str(input_shape_png_path) if write_png else "",
                "detect_csv_path": str(detect_csv_path),
                "export_subdir": str(out_dir),
                "n_detections": len(detect_rows),
            }
            (out_dir / f"{stem}.json").write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            manifest.append(row)
        with (self.export_dir / "export_manifest.jsonl").open("w", encoding="utf-8") as handle:
            for row in manifest:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {
            "n_exported": len(manifest),
            "out_dir": str(self.export_dir),
            "selection_csv": str(selection_csv),
            "n_detections": sum(int(row["n_detections"]) for row in manifest),
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "HscPackBrowser/1.0"

    @property
    def state(self) -> BrowserState:
        state = self.server.browser_state  # type: ignore[attr-defined]
        if state is None:
            raise RuntimeError("browser has not been started; use the menu first")
        return state

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        return json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                body = load_html()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path.startswith("/pages/"):
                rel = parsed.path.removeprefix("/pages/").strip("/")
                path = (PAGES_DIR / rel).resolve()
                if PAGES_DIR.resolve() not in path.parents or not path.is_file():
                    self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                    return
                body = path.read_bytes()
                content_type = "text/plain"
                if path.suffix == ".js":
                    content_type = "application/javascript; charset=utf-8"
                elif path.suffix == ".css":
                    content_type = "text/css; charset=utf-8"
                elif path.suffix == ".html":
                    content_type = "text/html; charset=utf-8"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path.startswith("/assets/"):
                rel = parsed.path.removeprefix("/assets/").strip("/")
                path = (ASSETS_DIR / rel).resolve()
                if ASSETS_DIR.resolve() not in path.parents or not path.is_file():
                    self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                    return
                body = path.read_bytes()
                content_type = "application/octet-stream"
                if path.suffix.lower() == ".png":
                    content_type = "image/png"
                elif path.suffix.lower() in {".jpg", ".jpeg"}:
                    content_type = "image/jpeg"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path == "/api/options":
                self._send_json(self.server.options_payload())  # type: ignore[attr-defined]
            elif parsed.path == "/api/state":
                state = self.server.browser_state  # type: ignore[attr-defined]
                self._send_json(state.state_payload() if state is not None else {"started": False})
            elif parsed.path == "/api/page":
                page = int(parse_qs(parsed.query).get("page", ["0"])[0])
                self._send_json(self.state.page_payload(page))
            elif parsed.path.startswith("/image/") and parsed.path.endswith(".png"):
                token = Path(parsed.path).name.removesuffix(".png")
                query = parse_qs(parsed.query)
                detect = query.get("detect", ["0"])[0] in {"1", "true", "yes"}
                input_image = query.get("input", ["0"])[0] in {"1", "true", "yes"}
                input_shape = query.get("input_shape", ["0"])[0] in {"1", "true", "yes"}
                show_shape = query.get("shape", ["1"])[0] in {"1", "true", "yes"}
                show_center = query.get("center", ["0"])[0] in {"1", "true", "yes"}
                smooth_mode = query.get("smooth_mode", ["none"])[0]
                try:
                    smooth_sigma = float(query.get("smooth_sigma", ["1.0"])[0])
                except Exception:
                    smooth_sigma = 1.0
                try:
                    smooth_radius = int(float(query.get("smooth_radius", ["1"])[0]))
                except Exception:
                    smooth_radius = 1
                body = self.state.image_png(
                    token,
                    detect=detect,
                    input_image=input_image,
                    input_shape=input_shape,
                    show_shape=show_shape,
                    show_center=show_center,
                    smooth_mode=smooth_mode,
                    smooth_sigma=smooth_sigma,
                    smooth_radius=smooth_radius,
                )
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path.startswith("/tile_map/") and parsed.path.endswith(".png"):
                patch = Path(parsed.path).name.removesuffix(".png")
                query = parse_qs(parsed.query)
                patch = query.get("patch", [patch])[0]
                body = self.state.tile_map_png_by_patch.get(str(patch))
                if body is None:
                    self._send_json({"error": f"tile map not found for patch {patch}"}, HTTPStatus.NOT_FOUND)
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/start":
                self.server.start_browser(payload)  # type: ignore[attr-defined]
                self._send_json(self.state.state_payload())
            elif parsed.path == "/api/set_patch":
                self.state.set_patch(str(payload["patch"]))
                self._send_json(self.state.state_payload())
            elif parsed.path == "/api/select":
                tokens = payload.get("tokens")
                if tokens is None:
                    tokens = [payload["token"]]
                self.state.set_selected([str(token) for token in tokens], bool(payload.get("selected", True)))
                self._send_json(self.state.state_payload())
            elif parsed.path == "/api/detect_page":
                page = int(payload.get("page", 0))
                self._send_json(self.state.detect_page(page))
            elif parsed.path == "/api/find_tile":
                mode = str(payload.get("mode", ""))
                self._send_json(
                    self.state.find_tile(
                        mode,
                        payload.get("tile_id") if mode == "tile_xy" else payload.get("x", 0),
                        payload.get("y", None),
                    )
                )
            elif parsed.path == "/api/save_selection_csv":
                path = self.state.write_selection_csv()
                self._send_json({"selection_csv": str(path), "n_selected": len(self.state.selected)})
            elif parsed.path == "/api/export":
                result = self.state.export_selected(write_png=bool(payload.get("write_png", True)))
                self._send_json(result)
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


class Server(ThreadingHTTPServer):
    def __init__(self, addr: tuple[str, int], args: argparse.Namespace) -> None:
        super().__init__(addr, Handler)
        self.args = args
        self.browser_state: BrowserState | None = None

    def options_payload(self) -> dict[str, Any]:
        by_dataset: dict[str, dict[str, Any]] = {}
        hsc_access = HscRawAccess(Path(self.args.root), str(self.args.tract))
        hsc_bands = hsc_access.available_bands() or list(DEFAULT_BANDS)
        hsc_default_bands = [band for band in self.args.bands if band in hsc_bands] or [band for band in DEFAULT_BANDS if band in hsc_bands]
        hsc_patches = hsc_access.available_patches(hsc_default_bands or hsc_bands)
        hsc_default_patches = [patch for patch in self.args.patches if patch in hsc_patches] or (
            [self.args.patch] if getattr(self.args, "patch", None) in hsc_patches else hsc_patches[:1]
        )
        by_dataset["hsc_raw"] = {
            "id": "hsc_raw",
            "label": "HSC raw tiles",
            "enabled": True,
            "tract": str(self.args.tract),
            "bands": hsc_bands,
            "patches": hsc_patches,
            "default_bands": hsc_default_bands,
            "default_patches": hsc_default_patches,
            "default_n_tiles": int(self.args.n_tiles),
            "default_frames_per_tile": int(self.args.frames_per_tile),
            "default_tiles_per_page": int(self.args.tiles_per_page),
            "tile_size": 256,
        }
        messier_access = MessierAccess(Path(self.args.messier_root), "default", selection_mode=str(self.args.messier_tile_mode))
        messier_patches = messier_access.available_patches()
        messier_defaults = [value for value in self.args.messier_patches if value in messier_patches] or messier_patches[:1]
        by_dataset["sitian"] = {
            "id": "sitian",
            "label": "Sitian",
            "enabled": bool(messier_patches),
            "tract": "default",
            "bands": ["default"],
            "patches": messier_patches,
            "default_bands": ["default"],
            "default_patches": messier_defaults,
            "default_n_tiles": int(self.args.messier_n_tiles),
            "default_frames_per_tile": 1,
            "default_tiles_per_page": int(self.args.tiles_per_page),
            "tile_size": 512,
        }
        by_dataset["hsc_image"] = {
            "id": "hsc_image",
            "label": "HSC coadd/noisy/denoised",
            "enabled": False,
            "reason": "placeholder",
            "tract": str(self.args.tract),
            "bands": list(DEFAULT_BANDS),
            "patches": [],
            "default_bands": list(DEFAULT_BANDS),
            "default_patches": [],
            "default_n_tiles": int(self.args.n_tiles),
            "default_frames_per_tile": 1,
            "default_tiles_per_page": int(self.args.tiles_per_page),
            "tile_size": 512,
        }
        by_dataset["ztf"] = {
            "id": "ztf",
            "label": "ZTF",
            "enabled": False,
            "reason": "placeholder",
            "tract": "default",
            "bands": [],
            "patches": [],
            "default_bands": [],
            "default_patches": [],
            "default_n_tiles": 4,
            "default_frames_per_tile": 1,
            "default_tiles_per_page": int(self.args.tiles_per_page),
            "tile_size": 512,
        }
        return {
            "dataset": str(self.args.dataset),
            "datasets": [
                {"id": "hsc_raw", "label": "HSC raw tiles", "enabled": True},
                {"id": "sitian", "label": "Sitian", "enabled": bool(messier_patches)},
                {"id": "hsc_image", "label": "HSC coadd/noisy/denoised", "enabled": False, "reason": "placeholder"},
                {"id": "ztf", "label": "ZTF", "enabled": False, "reason": "placeholder"},
            ],
            "by_dataset": by_dataset,
        }

    def start_browser(self, payload: dict[str, Any]) -> None:
        dataset = str(payload.get("dataset") or self.args.dataset)
        if dataset not in {"hsc_raw", "sitian"}:
            raise NotImplementedError(f"{dataset_label(dataset)} is a placeholder")
        self.args.dataset = dataset
        tract = str(payload.get("tract") or self.args.tract)
        patches = [str(v) for v in payload.get("patches", []) if str(v)]
        bands = [str(v) for v in payload.get("bands", []) if str(v)]
        if not patches:
            patches = [str(v) for v in (self.args.messier_patches if dataset == "sitian" else self.args.patches)]
        if not bands:
            bands = ["default"] if dataset == "sitian" else [str(v) for v in self.args.bands]
        tract = "default" if dataset == "sitian" else tract
        all_tiles = bool(payload.get("all_tiles", False))
        n_tiles_raw = payload.get("n_tiles", None)
        n_tiles = None if n_tiles_raw in (None, "", 0) else int(n_tiles_raw)
        frames_raw = payload.get("frames_per_tile", None)
        tiles_page_raw = payload.get("tiles_per_page", None)
        frames_per_tile = None if frames_raw in (None, "", 0) else int(frames_raw)
        tiles_per_page = None if tiles_page_raw in (None, "", 0) else int(tiles_page_raw)
        run_name = str(payload.get("run_name") or self.args.run_name or "")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        session_name = _session_name(run_name, stamp)
        base = Path(__file__).resolve().parent
        self.args.session_dir = base / "interactive_sessions" / session_name
        self.args.export_dir = base / "interactive_selected" / session_name
        self.browser_state = BrowserState(
            self.args,
            tract=tract,
            patches=patches,
            bands=bands,
            n_tiles=n_tiles,
            all_tiles=all_tiles,
            frames_per_tile=frames_per_tile,
            tiles_per_page=tiles_per_page,
            run_name=run_name,
            stamp=stamp,
        )


def main() -> None:
    base = Path(__file__).resolve().parent
    stamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Serve interactive CELLECT/SAM QC browser for multiple astronomy image datasets.")
    parser.add_argument("--dataset", choices=("hsc_raw", "sitian", "hsc_image", "ztf"), default="hsc_raw")
    parser.add_argument("--root", "--data-root", dest="root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--messier-root", type=Path, default=DEFAULT_MESSIER_ROOT)
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patch", default="4,5", help="Backward-compatible single default patch.")
    parser.add_argument("--patches", nargs="+", default=None, help="Default selected patches for the menu.")
    parser.add_argument("--messier-patches", nargs="+", default=None, help="Default selected Messier/Sitian objects for the menu.")
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    parser.add_argument("--n-tiles", type=int, default=60)
    parser.add_argument("--messier-n-tiles", type=int, default=4)
    parser.add_argument("--messier-tile-mode", choices=("brightest", "random_grid"), default="brightest")
    parser.add_argument("--frames-per-tile", type=int, default=1)
    parser.add_argument("--tiles-per-page", type=int, default=2)
    parser.add_argument("--detect-batch-size", type=int, default=80, help="Maximum tile-slots per model forward pass.")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("none", "bf16"), default="bf16")
    parser.add_argument("--visit", type=int, default=None)
    parser.add_argument("--frame-rank", type=int, default=0)
    parser.add_argument("--strict-visit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scaling-mode", choices=("zscore_clip", "zscore_no_clip", "zscore_no_upper", "log_lupton", "anscombe"), default="anscombe")
    parser.add_argument("--clip-threshold", type=float, default=3.0)
    parser.add_argument("--log-a", type=float, default=300.0)
    parser.add_argument("--log-high-percentile", type=float, default=99.5)
    parser.add_argument("--lupton-stretch", type=float, default=0.5)
    parser.add_argument("--lupton-q", type=float, default=20.0)
    parser.add_argument("--anscombe-clip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--anscombe-scale", type=float, default=1000.0)
    parser.add_argument("--confidence-threshold", type=float, default=2.0)
    parser.add_argument("--confidence-score", default="ordinal_expectation")
    parser.add_argument("--nms-radius", type=int, default=3)
    parser.add_argument("--center-refinement", choices=("integer", "weighted_centroid", "softargmax"), default="softargmax")
    parser.add_argument("--center-refinement-radius", type=int, default=1)
    parser.add_argument("--shape-overlay-centers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--session-dir", type=Path, default=base / "interactive_sessions" / stamp)
    parser.add_argument("--export-dir", type=Path, default=base / "interactive_selected" / stamp)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--run-name", type=str, default=f"")
    args = parser.parse_args()
    if args.patches is None:
        args.patches = [args.patch]
    if args.messier_patches is None:
        args.messier_patches = []

    server = Server((args.host, args.port), args)
    host, port = server.server_address[:2]
    print(
        json.dumps(
            {
                "url": f"http://{host}:{port}/",
                "dataset": str(args.dataset),
                "data_root": str(Path(args.root).expanduser().resolve()),
                "messier_root": str(Path(args.messier_root).expanduser().resolve()),
                "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
                "session_dir": str(Path(args.session_dir).expanduser().resolve()),
                "export_dir": str(Path(args.export_dir).expanduser().resolve()),
            },
            indent=2,
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
