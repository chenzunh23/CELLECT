"""Image loading, scaling, and background-mask helpers for preprocessing v3."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
from astropy.io import fits

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cellect")

from data_filtering.sam_input_scaling import build_bright_mask, scale_training_image


@dataclass(frozen=True)
class ImageProcessingConfig:
    scaling_mode: str = "zscore-log-lupton-rgb"
    hdu: int | str = 1
    clip_threshold: float = 3.0
    log_a: float | None = None
    log_high_percentile: float = 99.5
    lupton_stretch: float = 0.5
    lupton_q: float = 20.0
    anscombe_clip: bool = False
    anscombe_scale: float = 1.0


@dataclass(frozen=True)
class BrightRegionConfig:
    mode: str = "log-lupton"
    threshold: float = 2.99
    clip_threshold: float = 3.0
    dilation: int = 2
    log_a: float = 1000.0
    log_high_percentile: float = 99.5
    lupton_stretch: float = 0.5
    lupton_q: float = 20.0
    anscombe_clip: bool = False
    anscombe_scale: float = 1000.0


def read_fits_image(path: Path | str, *, hdu: int | str = 1) -> tuple[np.ndarray, fits.Header]:
    with fits.open(Path(path), memmap=True) as hdul:
        data = np.asarray(hdul[hdu].data, dtype=np.float32)
        header = hdul[hdu].header.copy()
    return data, header


def scale_image_for_training(image: np.ndarray, *, config: ImageProcessingConfig) -> np.ndarray:
    kwargs = {"clip_threshold": float(config.clip_threshold)}
    if config.log_a is not None:
        kwargs["log_a"] = config.log_a
    kwargs["log_high_percentile"] = float(config.log_high_percentile)
    kwargs["lupton_stretch"] = float(config.lupton_stretch)
    kwargs["lupton_q"] = float(config.lupton_q)
    if "anscombe" in config.scaling_mode:
        kwargs["anscombe_clip"] = config.anscombe_clip
        kwargs["anscombe_scale"] = config.anscombe_scale
    return scale_training_image(image, mode=config.scaling_mode, **kwargs)


def component_area_map(labels: np.ndarray) -> dict[int, int]:
    counts = np.bincount(np.asarray(labels, dtype=np.int32).ravel())
    return {idx: int(value) for idx, value in enumerate(counts) if idx > 0 and value > 0}


def component_centroid_map(labels: np.ndarray) -> dict[int, tuple[float, float]]:
    labels = np.asarray(labels, dtype=np.int32)
    flat = labels.ravel()
    if flat.size == 0:
        return {}
    yy, xx = np.indices(labels.shape, dtype=np.float64)
    counts = np.bincount(flat)
    sum_x = np.bincount(flat, weights=xx.ravel())
    sum_y = np.bincount(flat, weights=yy.ravel())
    out: dict[int, tuple[float, float]] = {}
    for idx in range(1, len(counts)):
        if counts[idx] > 0:
            out[idx] = (float(sum_x[idx] / counts[idx]), float(sum_y[idx] / counts[idx]))
    return out


def build_bright_components(image: np.ndarray, *, config: BrightRegionConfig = BrightRegionConfig()) -> tuple[np.ndarray, np.ndarray]:
    """Return bright-region mask and connected-component labels.

    This is the non-plotting version of ``build_external_bright_labels_v2``.
    Modes ``zscore-no-upper``, ``zscore-unbounded``, ``raw`` and ``none`` have
    no pixel-level bright threshold by design; their bright-source branch is
    handled by source clustering and Gaia matching.
    """

    bright = build_bright_mask(
        image,
        mode=config.mode,
        threshold=float(config.threshold),
        clip_threshold=float(config.clip_threshold),
        dilation=int(config.dilation),
        log_a=float(config.log_a),
        log_high_percentile=float(config.log_high_percentile),
        lupton_stretch=float(config.lupton_stretch),
        lupton_q=float(config.lupton_q),
        anscombe_clip=bool(config.anscombe_clip),
        anscombe_scale=float(config.anscombe_scale),
    )
    try:
        from scipy import ndimage

        labels, _num = ndimage.label(np.asarray(bright, dtype=bool))
    except Exception:
        labels = np.zeros(np.asarray(bright).shape, dtype=np.int32)
    return np.asarray(bright, dtype=np.uint8), np.asarray(labels, dtype=np.int32)


def read_background_mask(path: Path | str | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(shape, dtype=bool)
    path = Path(path)
    if not path.exists():
        return np.zeros(shape, dtype=bool)
    data = np.load(path)
    for key in ("background", "background_mask", "lsst_background"):
        if key in data:
            mask = np.asarray(data[key], dtype=bool)
            if mask.shape != shape:
                raise ValueError(f"background mask shape mismatch for {path}: {mask.shape} != {shape}")
            return mask
    return np.zeros(shape, dtype=bool)


def read_quality_mask(path: Path | str | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None:
        return np.zeros(shape, dtype=bool)
    path = Path(path)
    if not path.exists():
        return np.zeros(shape, dtype=bool)
    data = np.load(path)
    masks = []
    for key in ("sat", "bad", "edge", "SAT", "BAD", "EDGE"):
        if key in data:
            arr = np.asarray(data[key], dtype=bool)
            if arr.shape == shape:
                masks.append(arr)
    return np.logical_or.reduce(masks) if masks else np.zeros(shape, dtype=bool)
