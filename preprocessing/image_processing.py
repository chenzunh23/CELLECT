"""Image loading, scaling, and background-mask helpers for preprocessing v3."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
from astropy.io import fits

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cellect")

from data_filtering.sam_input_scaling import build_bright_mask, scale_training_image
from .utils.image import fill_nonfinite_hybrid, hsc_surface_brightness_factor


@dataclass(frozen=True)
class ImagePreparationConfig:
    input_unit: str = "auto"
    pixel_scale_arcsec: float | None = None
    input_zeropoint: float | None = None
    nan_policy: str = "hybrid"


@dataclass(frozen=True)
class PreparedImage:
    image: np.ndarray
    finite_mask: np.ndarray
    metadata: dict[str, object]

    @property
    def input_valid_mask(self) -> np.ndarray:
        """Usable model context, including repaired holes; not observed science."""
        return np.isfinite(self.image)


def prepare_image(image: np.ndarray, *, header: fits.Header | None = None,
                  config: ImagePreparationConfig = ImagePreparationConfig(),
                  source_path: Path | str | None = None,
                  protected_invalid_mask: np.ndarray | None = None) -> PreparedImage:
    """Convert raw intensities once, preserving the grid and original validity.

    Pass the result to both scaling and bright-component construction. Statistics
    remain local to the supplied array; callers choose the patch/tile extent.
    """
    if isinstance(image, PreparedImage):
        raise ValueError("image is already prepared; do not convert units twice")
    original = np.asarray(image)
    with np.errstate(over="ignore", invalid="ignore"):
        raw = np.asarray(original, dtype=np.float32)
    if raw.ndim != 2 or raw.size == 0:
        raise ValueError("image preparation requires a nonempty 2-D image")
    if config.nan_policy not in {"hybrid", "none"}:
        raise ValueError(f"unknown nan_policy: {config.nan_policy!r}")
    finite = np.isfinite(original)
    if protected_invalid_mask is not None:
        protected_invalid_mask = np.asarray(protected_invalid_mask, bool)
        if protected_invalid_mask.shape != raw.shape:
            raise ValueError('protected invalid mask/image shape mismatch')
        finite &= ~protected_invalid_mask
        raw = np.where(protected_invalid_mask, np.nan, raw)
    if np.any(finite & ~np.isfinite(raw)):
        raise ValueError("input image overflows float32")
    factor, metadata = hsc_surface_brightness_factor(
        header, input_unit=config.input_unit, pixel_scale_arcsec=config.pixel_scale_arcsec,
        input_zeropoint=config.input_zeropoint,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        factor32 = np.float32(factor)
        converted = raw * factor32
    if not np.isfinite(factor32) or factor32 <= 0:
        raise ValueError("unit conversion factor is not representable in float32")
    if np.any(finite & ~np.isfinite(converted)):
        raise ValueError("unit conversion overflows float32")
    if config.nan_policy == "hybrid":
        boundary = None
        background_stats = None
        source = source_path or (header.get('ORIGFILE') if header is not None else None)
        if finite.any() and not finite.all() and source is not None:
            from .utils.nan_boundaries import original_fits_edge_mask,edge_connected_nonfinite,source_nan_background
            # Enclosed crop-local holes cannot connect to any original edge.
            # Avoid scanning a mosaic for those common, unambiguous cases.
            local_open=edge_connected_nonfinite(raw)
            boundary=original_fits_edge_mask(raw,header,source) if local_open.any() else local_open
            native_stats = source_nan_background(source)
            if native_stats is not None:
                background_stats = tuple(v*float(factor32) for v in native_stats)
        if protected_invalid_mask is not None and protected_invalid_mask.any():
            from .utils.nan_boundaries import edge_connected_nonfinite
            if boundary is None:
                boundary = edge_connected_nonfinite(converted)
            boundary = boundary | protected_invalid_mask
        converted, stats = fill_nonfinite_hybrid(converted, edge_connected_mask=boundary,
                                               background_stats=background_stats)
        metadata.update(stats)
        from .utils.image import NAN_FILL_POLICY, NAN_SOURCE_RING_PIXELS, NAN_SOURCE_MEDIAN_SIGMA, NAN_SOURCE_P90_SIGMA
        metadata.update(nan_fill_policy=NAN_FILL_POLICY, nan_source_ring_pixels=NAN_SOURCE_RING_PIXELS,
                        nan_source_median_sigma=NAN_SOURCE_MEDIAN_SIGMA, nan_source_p90_sigma=NAN_SOURCE_P90_SIGMA,
                        nan_threshold_reference=('original_fits_sampled_finite' if background_stats is not None else 'supplied_array_finite'),
                        nan_boundary_policy='original-fits-edge-4connected-v1',
                        nan_boundary_source=str(source) if source is not None else 'supplied full image')
    # A finite SCI sample rejected by DQ/TRAIN_BAD is not a repairable NaN hole.
    if protected_invalid_mask is not None:
        converted[protected_invalid_mask] = np.nan
    metadata.update(nan_policy=config.nan_policy, finite_pixels=int(finite.sum()), total_pixels=int(raw.size),
                    input_valid_pixels=int(np.isfinite(converted).sum()))
    return PreparedImage(converted, finite, metadata)


def _prepared_input(image, header, preparation):
    if preparation is not None:
        image = prepare_image(image, header=header, config=preparation)
    if isinstance(image, PreparedImage):
        return image.image, bool(image.finite_mask.any())
    return image, True


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
    statistics_clip_sigma: float | None = None


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
    statistics_clip_sigma: float | None = None


def read_fits_image(path: Path | str, *, hdu: int | str = 1) -> tuple[np.ndarray, fits.Header]:
    from .dataset_inputs import image_hdu
    with fits.open(Path(path), memmap=None) as hdul:
        idx = image_hdu(hdul, hdu)
        data = np.array(hdul[idx].data, dtype=np.float32, copy=True)
        header = hdul[0].header.copy()
        header.update(hdul[idx].header)
    return data, header


def scale_image_for_training(image: np.ndarray | PreparedImage, *, config: ImageProcessingConfig,
                             header: fits.Header | None = None,
                             preparation: ImagePreparationConfig | None = None) -> np.ndarray:
    image, _valid = _prepared_input(image, header, preparation)
    kwargs = {"clip_threshold": float(config.clip_threshold)}
    if config.statistics_clip_sigma is not None:
        kwargs["statistics_clip_sigma"] = float(config.statistics_clip_sigma)
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


def build_bright_components(image: np.ndarray | PreparedImage, *, config: BrightRegionConfig = BrightRegionConfig(),
                            header: fits.Header | None = None,
                            preparation: ImagePreparationConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return bright-region mask and connected-component labels.

    This is the non-plotting version of ``build_external_bright_labels_v2``.
    Modes ``zscore-no-upper``, ``zscore-unbounded``, ``raw`` and ``none`` have
    no pixel-level bright threshold by design; their bright-source branch is
    handled by source clustering and Gaia matching.
    """

    image, valid = _prepared_input(image, header, preparation)
    if not valid:
        return np.zeros(image.shape, dtype=np.uint8), np.zeros(image.shape, dtype=np.int32)
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
        statistics_clip_sigma=config.statistics_clip_sigma,
    )
    # A morphology dilation must not turn an unfilled coverage boundary bright.
    bright = np.asarray(bright,bool) & np.isfinite(image)
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
