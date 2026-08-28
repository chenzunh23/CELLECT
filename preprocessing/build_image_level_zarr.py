#!/usr/bin/env python
"""Build v3 image-level Zarr stores for SAM-style training.

The output layout and array schema intentionally match the historical
``direct_zarr_preprocessing`` image-level stores so ``astro_train_eval.py`` can
keep using ``--data-format zarr --zarr-random-image-batches``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy.io import fits
from astropy.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cellect")

from astro_data_preprocessing import (
    _band_catalog_path,
    _band_det_path,
    _band_fits_path,
    _find_image_hdu_index,
    _origin_from_ltv,
    _read_det_background_mask,
    make_tile_specs,
)

from preprocessing.bright_ap2 import BrightAp2Config, classify_bright_ap2
from preprocessing.bright_label import (
    BrightLabelConfig,
    label_bright_sources,
    unsupervised_seeded_component_centers,
)
from preprocessing.image_processing import (
    BrightRegionConfig,
    ImageProcessingConfig,
    build_bright_components,
    read_background_mask,
    scale_image_for_training,
)
from preprocessing.labels import DenseLabel, LabelWeights, SourceClass
from preprocessing.meas_processing import MeasProcessingConfig, classify_meas_basics
from preprocessing.ordinary import OrdinaryConfig, label_ordinary_sources
from preprocessing.refit import RefitConfig, attach_refit_geometry, compute_kron_ellipse
from preprocessing.region_filling import fill_dense_regions
from preprocessing.snr import SnrConfig, compute_snr_for_sample
from preprocessing.utils.catalog import source_ids
from preprocessing.utils.geometry import paint_ellipse
from preprocessing.zarr_writing import ImageLevelTrainingBatch, write_training_image_level_zarr


DEFAULT_BANDS = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y", "NB0387", "NB0816", "NB0921", "NB1010")
MASK_PLANES_FOR_STRICT_IGNORE = ("SAT", "BAD", "EDGE", "NO_DATA", "UNMASKEDNAN")


@dataclass(frozen=True)
class StoreTask:
    data_root: Path
    coadd_fits_root: Path
    output_root: Path
    refit_root: Path
    denoised_fits_root: Path
    coadd_weight_root: Path
    coadd_lsst_background_root: Path | None
    variant_lsst_background_root: Path | None
    gaia_fits: Path | None
    tract: int
    patch: str
    band: str
    dataset_source: str
    group: str
    tile_size: int
    stride: int
    max_tiles: int
    overwrite: bool
    chunk_tiles: int
    image_scaling_mode: str
    image_scaling_scope: str
    bright_mask_mode: str
    bright_threshold: float
    bright_dilation: int
    clip_threshold: float
    image_log_a: float
    image_log_high_percentile: float
    image_lupton_stretch: float
    image_lupton_q: float
    image_anscombe_clip: bool
    image_anscombe_scale: float
    bright_log_a: float
    bright_log_high_percentile: float
    bright_lupton_stretch: float
    bright_lupton_q: float
    bright_anscombe_scale: float
    cluster_source_match_pixels: float
    cluster_centroid_match_pixels: float
    gaia_bright_mag_threshold: float
    snr_method: str
    missing_noncoadd_policy: str
    image_variant_background_source: str = "auto"
    missing_variant_background_policy: str = "fallback_coadd"


@dataclass
class PatchLabels:
    table: Table
    dense: np.ndarray
    label_classes: np.ndarray
    geom_x: np.ndarray
    geom_y: np.ndarray
    geom_major: np.ndarray
    geom_minor: np.ndarray
    geom_theta: np.ndarray
    source_ids: np.ndarray
    strict_x: np.ndarray
    strict_y: np.ndarray
    strict_ids: np.ndarray


def _is_narrow_band(band: str) -> bool:
    return str(band).upper().startswith("NB")


def _band_log_a(band: str) -> float:
    band = str(band).upper()
    if band == "NB1010":
        return 100.0
    if band == "NB0387":
        return 3000.0
    return 1000.0


def _image_log_a(task: StoreTask) -> float:
    value = float(task.image_log_a)
    return value if math.isfinite(value) and value > 0.0 else _band_log_a(task.band)


def _bright_log_a(task: StoreTask) -> float:
    value = float(task.bright_log_a)
    return value if math.isfinite(value) and value > 0.0 else _band_log_a(task.band)


def _group_number(group: str | int | None) -> int | None:
    if group is None:
        return None
    text = str(group).strip()
    if text.startswith("group_"):
        text = text[6:]
    try:
        return int(text)
    except Exception:
        return None


def _product_patch_dir(root: Path, dataset_source: str, tract: int, band: str, patch: str) -> Path:
    candidates = [
        root / str(tract) / band / patch,
        root / dataset_source / str(tract) / band / patch,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _variant_groups(root: Path, patch: str, *, band: str | None = None, tract: int | None = None, dataset_source: str | None = None) -> list[str]:
    patch_dir = root / f"patch_{patch.replace(',', '_')}"
    if not patch_dir.exists():
        old_groups: list[str] = []
    else:
        old_groups = [path.name for path in sorted(patch_dir.glob("group_*")) if path.is_dir()]
    if old_groups or band is None or tract is None or dataset_source is None:
        return old_groups
    product_dir = _product_patch_dir(root, str(dataset_source), int(tract), str(band), patch)
    if not product_dir.exists():
        return []
    groups: set[int] = set()
    for path in product_dir.glob(f"warp*{band}*{tract}*{patch}*.fits"):
        group = _group_number(path.stem.rsplit("-", 1)[-1])
        if group is not None:
            groups.add(group)
    return [str(group) for group in sorted(groups)]


def _variant_image_path(root: Path, patch: str, group: str, band: str, dataset_source: str, tract: int | None = None) -> Path:
    old = root / f"patch_{patch.replace(',', '_')}" / group / band / f"{dataset_source}.fits"
    if old.exists() or tract is None:
        return old
    product_dir = _product_patch_dir(root, dataset_source, int(tract), band, patch)
    group_num = _group_number(group)
    if product_dir.exists():
        matches = sorted(path for path in product_dir.glob(f"warp*{band}*{tract}*{patch}*.fits") if not path.name.startswith("effective_count"))
        if group_num is not None:
            matches = [path for path in matches if path.stem.rsplit("-", 1)[-1] == str(group_num)]
        if matches:
            return matches[0]
    return old


def _coadd_image_path(root: Path, band: str, tract: int, patch: str) -> Path:
    official = _band_fits_path(root, band, tract, patch)
    if official.exists() and not official.name.startswith("effective_count"):
        return official
    product_dirs = [
        root / str(tract) / band / patch,
        root / "half_coadd" / str(tract) / band / patch,
    ]
    for product_dir in product_dirs:
        if not product_dir.exists():
            continue
        matches = sorted(path for path in product_dir.glob(f"warp_half*{band}*{tract}*{patch}*.fits") if not path.name.startswith("effective_count"))
        if matches:
            return matches[0]
    return official


def _store_output_path(output_root: Path, patch: str, band: str, dataset_source: str, group: str) -> Path:
    if dataset_source == "coadd":
        return output_root / "image_level" / "coadd" / band / f"{patch}.zarr"
    return output_root / "image_level" / dataset_source / band / f"{patch}__{group}.zarr"


def _refit_csv_path(refit_root: Path, tract: int, band: str, patch: str) -> Path:
    return refit_root / str(tract) / band / patch / "batch_heavyfp_kron_refit" / "batch_heavyfp_kron_refit.csv"


def _read_image_header_origin(path: Path, hdu: int | str = 1) -> tuple[np.ndarray, fits.Header, tuple[int, int]]:
    with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
        if isinstance(hdu, str):
            idx = hdul.index_of(hdu)
        else:
            idx = int(hdu)
            if idx >= len(hdul) or getattr(hdul[idx], "data", None) is None:
                idx = _find_image_hdu_index(hdul)
        data = np.asarray(hdul[idx].data, dtype=np.float32)
        header = hdul[idx].header.copy()
        origin = _origin_from_ltv(header)
    return data, header, origin


def _mask_plane_bits(header: fits.Header) -> dict[str, int]:
    bits: dict[str, int] = {}
    for key, value in header.items():
        if not str(key).startswith("MP_"):
            continue
        name = str(key)[3:].upper()
        try:
            bits[name] = int(value)
        except Exception:
            continue
    return bits


def _read_fits_quality_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros(shape, dtype=bool)
    try:
        with fits.open(path, memmap=False, ignore_missing_end=True) as hdul:
            image_idx = _find_image_hdu_index(hdul)
            image = np.asarray(hdul[image_idx].data)
            image_all_finite = bool(np.isfinite(image).all()) if image.shape == shape else False
            if "MASK" in hdul:
                hdu = hdul["MASK"]
            else:
                mask_idx = image_idx + 1
                if mask_idx >= len(hdul) or getattr(hdul[mask_idx], "data", None) is None:
                    return np.zeros(shape, dtype=bool)
                hdu = hdul[mask_idx]
            mask = np.asarray(hdu.data, dtype=np.int64)
            if mask.shape != shape:
                return np.zeros(shape, dtype=bool)
            bits = _mask_plane_bits(hdu.header)
            out = np.zeros(shape, dtype=bool)
            for plane in MASK_PLANES_FOR_STRICT_IGNORE:
                bit = bits.get(plane)
                if bit is not None:
                    plane_mask = (mask & (1 << int(bit))) != 0
                    if plane == "BAD" and image_all_finite and bool(plane_mask.all()):
                        continue
                    out |= plane_mask
            return out
    except Exception:
        return np.zeros(shape, dtype=bool)


def _read_background_from_det(data_root: Path, band: str, tract: int, patch: str, shape: tuple[int, int], origin: tuple[int, int]) -> np.ndarray:
    det = _band_det_path(data_root, band, tract, patch)
    if det is None or not det.exists():
        return np.zeros(shape, dtype=bool)
    try:
        return _read_det_background_mask(det, shape, origin)
    except Exception:
        return np.zeros(shape, dtype=bool)


def _background_group_candidates(group: str | int) -> list[str]:
    text = str(group)
    candidates = [text]
    number = _group_number(text)
    if number is not None:
        candidates.extend([f"group_{number:02d}", f"group_{number}", str(number)])
    elif text.startswith("group_"):
        number = _group_number(text)
        if number is not None:
            candidates.extend([str(number), f"group_{number:02d}"])
    return list(dict.fromkeys(candidates))


def _variant_background_dirs(root: Path, variant: str, tract: int, patch: str, group: str, band: str) -> list[Path]:
    return [root / variant / str(tract) / patch / candidate / band for candidate in _background_group_candidates(group)]


def _variant_background_dir(root: Path, variant: str, tract: int, patch: str, group: str, band: str) -> Path:
    return _variant_background_dirs(root, variant, tract, patch, group, band)[0]


def _read_variant_background(
    *,
    root: Path | None,
    variant: str,
    tract: int,
    patch: str,
    group: str,
    band: str,
    shape: tuple[int, int],
    origin: tuple[int, int],
) -> np.ndarray | None:
    if root is None:
        return None
    for base in _variant_background_dirs(root, variant, tract, patch, group, band):
        if not base.exists():
            continue
        npz = base / "background_mask.npz"
        if npz.exists():
            mask = read_background_mask(npz, shape)
            return np.asarray(mask, dtype=bool)
        for det in sorted(base.glob("det-*.fits")):
            try:
                return _read_det_background_mask(det, shape, origin)
            except Exception:
                continue
    return None


def _read_coadd_lsst_background(
    *,
    root: Path | None,
    tract: int,
    patch: str,
    band: str,
    shape: tuple[int, int],
    origin: tuple[int, int],
) -> np.ndarray | None:
    if root is None:
        return None
    for variant in ("coadd", "half_coadd"):
        background = _read_variant_background(
            root=root,
            variant=variant,
            tract=tract,
            patch=patch,
            group="coadd",
            band=band,
            shape=shape,
            origin=origin,
        )
        if background is not None:
            return background
    return None


def _background_for_task(task: StoreTask, shape: tuple[int, int], origin: tuple[int, int]) -> np.ndarray:
    coadd = _read_background_from_det(task.data_root, task.band, task.tract, task.patch, shape, origin)
    if task.dataset_source == "coadd":
        coadd_lsst = _read_coadd_lsst_background(
            root=task.coadd_lsst_background_root,
            tract=task.tract,
            patch=task.patch,
            band=task.band,
            shape=shape,
            origin=origin,
        )
        if coadd_lsst is not None:
            return coadd_lsst
        if task.coadd_lsst_background_root is not None:
            tried = []
            for variant in ("coadd", "half_coadd"):
                tried.extend(
                    str(path / "background_mask.npz")
                    for path in _variant_background_dirs(
                        task.coadd_lsst_background_root,
                        variant,
                        task.tract,
                        task.patch,
                        "coadd",
                        task.band,
                    )
                )
            raise FileNotFoundError(
                "coadd LSST background not found; tried: "
                + ", ".join(tried)
            )
        return coadd

    source = str(task.image_variant_background_source).strip().lower()
    if source not in {"auto", "coadd-target", "variant-lsst", "none"}:
        raise ValueError(f"unknown image variant background source: {task.image_variant_background_source}")
    if source == "none":
        return np.zeros(shape, dtype=bool)
    if source == "coadd-target":
        return coadd

    variant = _read_variant_background(
        root=task.variant_lsst_background_root,
        variant=task.dataset_source,
        tract=task.tract,
        patch=task.patch,
        group=task.group,
        band=task.band,
        shape=shape,
        origin=origin,
    )
    if variant is not None:
        return variant
    if source == "variant-lsst" or str(task.missing_variant_background_policy) == "error":
        tried = [
            str(path / "background_mask.npz")
            for path in _variant_background_dirs(
                task.variant_lsst_background_root or Path("<variant-lsst-background-root>"),
                task.dataset_source,
                task.tract,
                task.patch,
                task.group,
                task.band,
            )
        ]
        raise FileNotFoundError(
            "variant LSST background not found; tried: "
            + ", ".join(tried)
        )
    if str(task.missing_variant_background_policy) == "none":
        return np.zeros(shape, dtype=bool)
    return coadd


def _crop(arr: np.ndarray, x0: int, y0: int, origin: tuple[int, int], size: int) -> np.ndarray:
    lx0 = int(x0) - int(origin[0])
    ly0 = int(y0) - int(origin[1])
    if arr.ndim == 2:
        return np.asarray(arr[ly0 : ly0 + size, lx0 : lx0 + size])
    if arr.ndim == 3:
        return np.asarray(arr[:, ly0 : ly0 + size, lx0 : lx0 + size])
    raise ValueError(f"cannot crop array with shape {arr.shape}")


def _scale_image_chw(raw_image: np.ndarray, task: StoreTask) -> np.ndarray:
    scaled = scale_image_for_training(
        raw_image,
        config=ImageProcessingConfig(
            scaling_mode=task.image_scaling_mode,
            clip_threshold=float(task.clip_threshold),
            log_a=_image_log_a(task),
            log_high_percentile=float(task.image_log_high_percentile),
            lupton_stretch=float(task.image_lupton_stretch),
            lupton_q=float(task.image_lupton_q),
            anscombe_clip=bool(task.image_anscombe_clip),
            anscombe_scale=float(task.image_anscombe_scale),
        ),
    )
    if scaled.ndim == 2:
        return scaled[None, :, :].astype(np.float32, copy=False)
    if scaled.ndim == 3 and scaled.shape[-1] in (1, 3):
        return np.moveaxis(scaled, -1, 0).astype(np.float32, copy=False)
    if scaled.ndim == 3 and scaled.shape[0] in (1, 3):
        return scaled.astype(np.float32, copy=False)
    raise ValueError(f"unsupported scaled image shape: {scaled.shape}")


def _classify_patch(task: StoreTask, image: np.ndarray, image_header: fits.Header, origin: tuple[int, int], coadd_image_fits: Path) -> PatchLabels:
    meas_path = _band_catalog_path(task.data_root, task.band, task.tract, task.patch)
    refit_csv = _refit_csv_path(task.refit_root, task.tract, task.band, task.patch)
    if not meas_path.exists():
        raise FileNotFoundError(f"meas catalog not found: {meas_path}")
    if not refit_csv.exists():
        raise FileNotFoundError(f"refit CSV not found: {refit_csv}")
    table = attach_refit_geometry(Table.read(meas_path), refit_csv, RefitConfig())
    bright_config = BrightRegionConfig(
        mode=task.bright_mask_mode,
        threshold=float(task.bright_threshold),
        clip_threshold=float(task.clip_threshold),
        dilation=int(task.bright_dilation),
        log_a=_bright_log_a(task),
        log_high_percentile=float(task.bright_log_high_percentile),
        lupton_stretch=float(task.bright_lupton_stretch),
        lupton_q=float(task.bright_lupton_q),
        anscombe_clip=bool(task.image_anscombe_clip),
        anscombe_scale=float(task.bright_anscombe_scale),
    )
    bright_region, components = build_bright_components(image, config=bright_config)
    quality = _read_fits_quality_mask(
        _variant_image_path(task.denoised_fits_root, task.patch, task.group, task.band, task.dataset_source, task.tract)
        if task.dataset_source != "coadd"
        else coadd_image_fits,
        image.shape,
    )
    if not bool(np.any(quality)) and task.dataset_source != "coadd":
        quality = _read_fits_quality_mask(coadd_image_fits, image.shape)
    stage = classify_meas_basics(table, config=MeasProcessingConfig(), refit_config=RefitConfig())
    snr = compute_snr_for_sample(
        table,
        dataset_source=task.dataset_source,
        is_narrow_band=_is_narrow_band(task.band),
        band=task.band,
        patch=task.patch,
        group=task.group if task.dataset_source != "coadd" else None,
        image_fits=(coadd_image_fits if task.dataset_source == "coadd" else _variant_image_path(task.denoised_fits_root, task.patch, task.group, task.band, task.dataset_source, task.tract)),
        coadd_image_fits=coadd_image_fits,
        config=SnrConfig(
            noncoadd_method=task.snr_method,
            missing_noncoadd_policy=task.missing_noncoadd_policy,  # type: ignore[arg-type]
            denoised_fits_root=task.denoised_fits_root,
            coadd_weight_root=task.coadd_weight_root,
        ),
    )
    ordinary = label_ordinary_sources(
        table,
        stage.ordinary_candidate,
        stage.labels,
        is_narrow_band=_is_narrow_band(task.band),
        snr=snr.snr if snr is not None else None,
        config=OrdinaryConfig(),
        snr_config=SnrConfig(),
        refit_config=RefitConfig(),
    )
    stage.labels = ordinary.labels
    bright_ap2 = classify_bright_ap2(
        table,
        stage.bright_candidate,
        stage.labels,
        component_labels=components,
        config=BrightAp2Config(),
        refit_config=RefitConfig(),
    )
    gaia = Table.read(task.gaia_fits) if task.gaia_fits is not None and task.gaia_fits.exists() else None
    bright = label_bright_sources(
        table,
        bright_ap2.candidate,
        bright_ap2.labels,
        bright_region=bright_region,
        component_labels=components,
        gaia_table=gaia,
        image_header=image_header,
        quality_mask=quality,
        mag=stage.mag,
        config=BrightLabelConfig(
            cluster_source_match_pixels=float(task.cluster_source_match_pixels),
            cluster_centroid_match_pixels=float(task.cluster_centroid_match_pixels),
            gaia_bright_mag_threshold=float(task.gaia_bright_mag_threshold),
        ),
        refit_config=RefitConfig(),
    )
    seed_components = bright_ap2.component_id[np.asarray(stage.bright_candidate, dtype=bool)]
    fallback_x, fallback_y, fallback_ids, fallback_component_ids = unsupervised_seeded_component_centers(
        table,
        bright.labels,
        components,
        seed_component_ids=seed_components,
        catalog_component_ids=seed_components,
        existing_strict_component_ids=bright.strict_center_component_id,
        min_area=float(BrightLabelConfig().empty_seeded_bright_component_area_min),
        component_search_radius=int(BrightAp2Config().component_search_radius),
        refit_config=RefitConfig(),
    )
    if fallback_x.size:
        bright.strict_center_x = np.concatenate([bright.strict_center_x, fallback_x]).astype(np.float64)
        bright.strict_center_y = np.concatenate([bright.strict_center_y, fallback_y]).astype(np.float64)
        bright.strict_center_source_id = np.concatenate([bright.strict_center_source_id, fallback_ids]).astype(np.int64)
        bright.strict_center_reason = np.concatenate(
            [
                bright.strict_center_reason,
                np.full(fallback_x.shape, "seeded_bright_component_no_supervised_center", dtype=object),
            ]
        )
        bright.strict_center_component_id = np.concatenate([bright.strict_center_component_id, fallback_component_ids]).astype(np.int32)
        bright.restricted_fallback_component_ids = np.unique(
            np.concatenate([bright.restricted_fallback_component_ids, fallback_component_ids]).astype(np.int32)
        )
        bright.ordinary_ignore_component_ids = np.setdiff1d(
            np.asarray(bright.ordinary_ignore_component_ids, dtype=np.int32),
            np.asarray(fallback_component_ids, dtype=np.int32),
            assume_unique=False,
        ).astype(np.int32)
    background = _background_for_task(task, image.shape, origin)
    restricted_fallback_mask = None
    if bright.restricted_fallback_component_ids.size and components is not None and np.asarray(components).size:
        component_ids = np.asarray(bright.restricted_fallback_component_ids, dtype=np.int32)
        component_ids = component_ids[component_ids > 0]
        if component_ids.size:
            restricted_fallback_mask = np.isin(np.asarray(components, dtype=np.int32), component_ids)
    ordinary_ignore_mask = None
    if bright.ordinary_ignore_component_ids.size and components is not None and np.asarray(components).size:
        component_ids = np.asarray(bright.ordinary_ignore_component_ids, dtype=np.int32)
        if bright.restricted_fallback_component_ids.size:
            component_ids = np.setdiff1d(
                component_ids,
                np.asarray(bright.restricted_fallback_component_ids, dtype=np.int32),
                assume_unique=False,
            ).astype(np.int32)
        component_ids = component_ids[component_ids > 0]
        if component_ids.size:
            ordinary_ignore_mask = np.isin(np.asarray(components, dtype=np.int32), component_ids)
    dense = fill_dense_regions(
        table,
        bright.labels,
        image.shape,
        background_mask=background,
        quality_ignore_mask=quality,
        restricted_fallback_mask=restricted_fallback_mask,
        ordinary_ignore_mask=ordinary_ignore_mask,
        ordinary_ignore_source_mask=bright.ordinary_ignore_source_mask,
        refit_config=RefitConfig(),
    )
    geom = compute_kron_ellipse(table, RefitConfig())
    return PatchLabels(
        table=table,
        dense=dense,
        label_classes=bright.labels.source_class.copy(),
        geom_x=geom.x.astype(np.float32),
        geom_y=geom.y.astype(np.float32),
        geom_major=geom.major.astype(np.float32),
        geom_minor=geom.minor.astype(np.float32),
        geom_theta=geom.theta.astype(np.float32),
        source_ids=np.asarray(source_ids(table), dtype=np.int64),
        strict_x=bright.strict_center_x.astype(np.float32),
        strict_y=bright.strict_center_y.astype(np.float32),
        strict_ids=bright.strict_center_source_id.astype(np.int64),
    )


def _paint_confidence(conf: np.ndarray, weight: np.ndarray, centers: np.ndarray, *, levels: int = 5, value_weight: float = 1.0) -> None:
    if centers.size == 0:
        return
    h, w = conf.shape
    yy, xx = np.mgrid[0:h, 0:w]
    radius = int(levels) - 1
    for cx, cy in np.asarray(centers, dtype=np.float32).reshape(-1, 2):
        cx_i = int(round(float(cx)))
        cy_i = int(round(float(cy)))
        if not (0 <= cx_i < w and 0 <= cy_i < h):
            continue
        y0 = max(0, cy_i - radius)
        y1 = min(h, cy_i + radius + 1)
        x0 = max(0, cx_i - radius)
        x1 = min(w, cx_i + radius + 1)
        dist = np.abs(xx[y0:y1, x0:x1] - float(cx)) + np.abs(yy[y0:y1, x0:x1] - float(cy))
        vals = np.ceil(np.clip(radius - dist, 0, None)).astype(np.uint8)
        keep = vals > 0
        patch_conf = conf[y0:y1, x0:x1]
        patch_weight = weight[y0:y1, x0:x1]
        patch_conf[keep] = np.maximum(patch_conf[keep], vals[keep])
        patch_weight[keep] = np.maximum(patch_weight[keep], float(value_weight))


def _source_indices_in_tile(labels: PatchLabels, mask: np.ndarray, spec, origin: tuple[int, int]) -> np.ndarray:
    tile_x0 = float(spec.x0) - float(origin[0])
    tile_y0 = float(spec.y0) - float(origin[1])
    local_x = labels.geom_x - tile_x0
    local_y = labels.geom_y - tile_y0
    return np.flatnonzero(
        np.asarray(mask, dtype=bool)
        & np.isfinite(local_x)
        & np.isfinite(local_y)
        & (local_x >= 0.0)
        & (local_x < float(spec.size))
        & (local_y >= 0.0)
        & (local_y < float(spec.size))
    )


def _strict_centers_in_tile(labels: PatchLabels, spec, origin: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    tile_x0 = float(spec.x0) - float(origin[0])
    tile_y0 = float(spec.y0) - float(origin[1])
    centers = np.column_stack([labels.strict_x - tile_x0, labels.strict_y - tile_y0]).astype(np.float32)
    keep = (
        np.isfinite(centers[:, 0])
        & np.isfinite(centers[:, 1])
        & (centers[:, 0] >= 0.0)
        & (centers[:, 0] < float(spec.size))
        & (centers[:, 1] >= 0.0)
        & (centers[:, 1] < float(spec.size))
    )
    return centers[keep], labels.strict_ids[keep]


def _tile_targets(labels: PatchLabels, spec, origin: tuple[int, int]) -> dict[str, np.ndarray]:
    size = int(spec.size)
    dense = _crop(labels.dense, spec.x0, spec.y0, origin, size).astype(np.uint8, copy=False)
    tile_x0 = float(spec.x0) - float(origin[0])
    tile_y0 = float(spec.y0) - float(origin[1])
    weights = LabelWeights().as_array()
    conf = np.zeros((size, size), dtype=np.uint8)
    conf_weight = weights[np.clip(dense, 0, len(weights) - 1)].astype(np.float32, copy=False)
    shape = np.zeros((3, size, size), dtype=np.float32)
    shape_weight = np.zeros((size, size), dtype=np.float32)

    clean_or_weak = (labels.label_classes == int(SourceClass.CLEAN)) | (labels.label_classes == int(SourceClass.WEAK_SHAPE))
    strict_table = labels.label_classes == int(SourceClass.STRICT_CENTER_ONLY)
    train_center = clean_or_weak | strict_table
    center_idx = _source_indices_in_tile(labels, train_center, spec, origin)
    center_xy = np.column_stack([labels.geom_x[center_idx] - tile_x0, labels.geom_y[center_idx] - tile_y0]).astype(np.float32)
    strict_extra_xy, strict_extra_ids = _strict_centers_in_tile(labels, spec, origin)
    all_conf_centers = center_xy
    if len(strict_extra_xy):
        all_conf_centers = np.concatenate([all_conf_centers, strict_extra_xy], axis=0)
    _paint_confidence(conf, conf_weight, all_conf_centers, levels=5, value_weight=1.0)

    shape_idx = _source_indices_in_tile(labels, clean_or_weak, spec, origin)
    for idx in shape_idx:
        cx = float(labels.geom_x[idx] - tile_x0)
        cy = float(labels.geom_y[idx] - tile_y0)
        major = float(labels.geom_major[idx])
        minor = float(labels.geom_minor[idx])
        theta = float(labels.geom_theta[idx])
        if not (math.isfinite(major) and math.isfinite(minor) and major > 0 and minor > 0):
            continue
        region = np.zeros((size, size), dtype=np.uint8)
        paint_ellipse(region, cx, cy, major, minor, theta, 1)
        inside = region > 0
        shape[0][inside] = major
        shape[1][inside] = minor
        shape[2][inside] = theta
        shape_weight[inside] = 1.0

    source_idx = _source_indices_in_tile(labels, clean_or_weak, spec, origin)
    source_centers = np.column_stack([labels.geom_x[source_idx] - tile_x0, labels.geom_y[source_idx] - tile_y0]).astype(np.float32)
    source_ids_arr = labels.source_ids[source_idx].astype(np.int64, copy=False)
    strict_table_idx = _source_indices_in_tile(labels, strict_table, spec, origin)
    strict_table_centers = np.column_stack([labels.geom_x[strict_table_idx] - tile_x0, labels.geom_y[strict_table_idx] - tile_y0]).astype(np.float32)
    strict_table_ids = labels.source_ids[strict_table_idx].astype(np.int64, copy=False)
    strict_centers = np.concatenate([strict_table_centers, strict_extra_xy], axis=0).astype(np.float32)
    strict_ids = np.concatenate([strict_table_ids, strict_extra_ids], axis=0).astype(np.int64)
    shape_centers = source_centers.astype(np.float32)
    shape_values = np.column_stack([labels.geom_major[source_idx], labels.geom_minor[source_idx], labels.geom_theta[source_idx]]).astype(np.float32)
    shape_classes = labels.label_classes[source_idx].astype(np.uint8, copy=False)
    shape_ids = source_ids_arr
    return {
        "confidence": conf,
        "conf_weight": conf_weight,
        "shape": shape,
        "shape_weight": shape_weight,
        "pu": dense,
        "source_centers": source_centers,
        "source_ids": source_ids_arr,
        "strict_centers": strict_centers,
        "strict_ids": strict_ids,
        "shape_centers": shape_centers,
        "shape_values": shape_values,
        "shape_classes": shape_classes,
        "shape_ids": shape_ids,
    }


def _build_store(task: StoreTask) -> dict[str, object]:
    coadd_image_fits = _coadd_image_path(task.coadd_fits_root, task.band, task.tract, task.patch)
    if task.dataset_source == "coadd":
        image_fits = coadd_image_fits
    else:
        image_fits = _variant_image_path(task.denoised_fits_root, task.patch, task.group, task.band, task.dataset_source, task.tract)
    if not image_fits.exists():
        raise FileNotFoundError(f"image FITS not found: {image_fits}")
    image, header, origin = _read_image_header_origin(image_fits)
    labels = _classify_patch(task, image, header, origin, coadd_image_fits)
    specs = make_tile_specs(
        parent_origin=origin,
        image_shape=(int(image.shape[1]), int(image.shape[0])),
        tile_size=task.tile_size,
        stride=task.stride,
        compare_origin=None,
    )
    if task.max_tiles > 0:
        specs = specs[: int(task.max_tiles)]
    n = len(specs)
    if n == 0:
        raise RuntimeError(f"no tile specs generated for {task.patch} {task.band} {task.dataset_source} {task.group}")
    h = w = int(task.tile_size)
    scaling_scope = str(task.image_scaling_scope).strip().lower().replace("_", "-")
    if scaling_scope not in {"patch", "tile"}:
        raise ValueError(f"unknown image scaling scope: {task.image_scaling_scope}")
    if scaling_scope == "patch":
        full_scaled_chw = _scale_image_chw(image, task)
        c = int(full_scaled_chw.shape[0])
        first_scaled_chw = None
    else:
        full_scaled_chw = None
        first_scaled_chw = _scale_image_chw(_crop(image, specs[0].x0, specs[0].y0, origin, h), task)
        c = int(first_scaled_chw.shape[0])
    images = np.zeros((n, 1, c, h, w), dtype=np.float32)
    band_conf = np.zeros((n, 1, h, w), dtype=np.uint8)
    band_conf_weight = np.zeros((n, 1, h, w), dtype=np.float32)
    band_shape = np.zeros((n, 1, 3, h, w), dtype=np.float32)
    band_shape_weight = np.zeros((n, 1, h, w), dtype=np.float32)
    band_pu = np.zeros((n, 1, h, w), dtype=np.uint8)
    sample_names: list[str] = []
    tile_names: list[str] = []
    groups: list[str] = []
    sources: list[str] = []
    tile_x0 = np.zeros(n, dtype=np.int32)
    tile_y0 = np.zeros(n, dtype=np.int32)
    centers_flat: list[np.ndarray] = []
    ids_flat: list[np.ndarray] = []
    offsets = np.zeros((n, 2), dtype=np.int64)
    strict_centers_flat: list[np.ndarray] = []
    strict_ids_flat: list[np.ndarray] = []
    strict_offsets = np.zeros((n, 2), dtype=np.int64)
    shape_centers_flat: list[np.ndarray] = []
    shape_values_flat: list[np.ndarray] = []
    shape_classes_flat: list[np.ndarray] = []
    shape_ids_flat: list[np.ndarray] = []
    shape_offsets = np.zeros((n, 2), dtype=np.int64)
    source_cursor = 0
    strict_cursor = 0
    shape_cursor = 0

    for i, spec in enumerate(specs):
        if full_scaled_chw is not None:
            scaled_chw = _crop(full_scaled_chw, spec.x0, spec.y0, origin, h)
        elif i == 0:
            scaled_chw = first_scaled_chw
        else:
            raw_tile = _crop(image, spec.x0, spec.y0, origin, h)
            scaled_chw = _scale_image_chw(raw_tile, task)
        images[i, 0] = scaled_chw
        target = _tile_targets(labels, spec, origin)
        band_conf[i, 0] = target["confidence"]
        band_conf_weight[i, 0] = target["conf_weight"]
        band_shape[i, 0] = target["shape"]
        band_shape_weight[i, 0] = target["shape_weight"]
        band_pu[i, 0] = target["pu"]
        tile_x0[i] = int(spec.x0)
        tile_y0[i] = int(spec.y0)
        tile_names.append(spec.name)
        groups.append("" if task.dataset_source == "coadd" else task.group)
        sources.append(task.dataset_source)
        prefix = task.band if task.dataset_source == "coadd" else f"{task.group}_{task.band}"
        sample_names.append(f"{prefix}_{spec.name}")

        offsets[i, 0] = source_cursor
        centers_flat.append(target["source_centers"])
        ids_flat.append(target["source_ids"])
        source_cursor += len(target["source_centers"])
        offsets[i, 1] = source_cursor

        strict_offsets[i, 0] = strict_cursor
        strict_centers_flat.append(target["strict_centers"])
        strict_ids_flat.append(target["strict_ids"])
        strict_cursor += len(target["strict_centers"])
        strict_offsets[i, 1] = strict_cursor

        shape_offsets[i, 0] = shape_cursor
        shape_centers_flat.append(target["shape_centers"])
        shape_values_flat.append(target["shape_values"])
        shape_classes_flat.append(target["shape_classes"])
        shape_ids_flat.append(target["shape_ids"])
        shape_cursor += len(target["shape_centers"])
        shape_offsets[i, 1] = shape_cursor

    def _cat(parts: list[np.ndarray], width: int | None, dtype) -> np.ndarray:
        if not parts:
            return np.empty((0, width), dtype=dtype) if width is not None else np.empty((0,), dtype=dtype)
        nonempty = [part for part in parts if len(part)]
        if not nonempty:
            return np.empty((0, width), dtype=dtype) if width is not None else np.empty((0,), dtype=dtype)
        return np.concatenate(nonempty, axis=0).astype(dtype, copy=False)

    output = _store_output_path(task.output_root, task.patch, task.band, task.dataset_source, task.group)
    batch = ImageLevelTrainingBatch(
        images=images,
        band_confidence=band_conf,
        band_conf_weight=band_conf_weight,
        band_shape=band_shape,
        band_shape_weight=band_shape_weight,
        band_pu_class_mask=band_pu,
        sample_names=sample_names,
        tile_x0=tile_x0,
        tile_y0=tile_y0,
        tile_names=tile_names,
        groups=groups,
        dataset_sources=sources,
        attrs={
            "tract": str(task.tract),
            "patch": task.patch,
            "bands": [task.band],
            "dataset_source": task.dataset_source,
            "group": task.group,
            "image_scaling_mode": task.image_scaling_mode,
            "image_scaling_scope": task.image_scaling_scope,
            "bright_mask_mode": task.bright_mask_mode,
            "bright_threshold": float(task.bright_threshold),
            "bright_dilation": int(task.bright_dilation),
            "clip_threshold": float(task.clip_threshold),
            "image_clip_threshold": float(task.clip_threshold),
            "image_log_a": float(_image_log_a(task)),
            "image_log_high_percentile": float(task.image_log_high_percentile),
            "image_lupton_stretch": float(task.image_lupton_stretch),
            "image_lupton_q": float(task.image_lupton_q),
            "image_anscombe_clip": bool(task.image_anscombe_clip),
            "image_anscombe_scale": float(task.image_anscombe_scale),
            "bright_log_a": float(_bright_log_a(task)),
            "bright_log_high_percentile": float(task.bright_log_high_percentile),
            "bright_lupton_stretch": float(task.bright_lupton_stretch),
            "bright_lupton_q": float(task.bright_lupton_q),
            "bright_anscombe_scale": float(task.bright_anscombe_scale),
            "cluster_source_match_pixels": float(task.cluster_source_match_pixels),
            "cluster_centroid_match_pixels": float(task.cluster_centroid_match_pixels),
            "gaia_bright_mag_threshold": float(task.gaia_bright_mag_threshold),
            "image_variant_background_source": task.image_variant_background_source,
            "coadd_lsst_background_root": str(task.coadd_lsst_background_root) if task.coadd_lsst_background_root is not None else "",
            "variant_lsst_background_root": str(task.variant_lsst_background_root) if task.variant_lsst_background_root is not None else "",
            "missing_variant_background_policy": task.missing_variant_background_policy,
            "tile_size": int(task.tile_size),
            "stride": int(task.stride),
            "source_export_mode": "preprocessing_v3",
            "image_fits": str(image_fits),
            "coadd_image_fits": str(coadd_image_fits),
            "refit_csv": str(_refit_csv_path(task.refit_root, task.tract, task.band, task.patch)),
        },
        source_centers=_cat(centers_flat, 2, np.float32),
        source_ids=_cat(ids_flat, None, np.int64),
        source_offsets=offsets,
        strict_center_only_centers=_cat(strict_centers_flat, 2, np.float32),
        strict_center_only_ids=_cat(strict_ids_flat, None, np.int64),
        strict_center_only_offsets=strict_offsets,
        shape_source_centers=_cat(shape_centers_flat, 2, np.float32),
        shape_source_values=_cat(shape_values_flat, 3, np.float32),
        shape_source_classes=_cat(shape_classes_flat, None, np.uint8),
        shape_source_ids=_cat(shape_ids_flat, None, np.int64),
        shape_source_offsets=shape_offsets,
    )
    write_training_image_level_zarr(output, batch, overwrite=task.overwrite, chunk_tiles=task.chunk_tiles)
    return {"output": str(output), "samples": n, "patch": task.patch, "band": task.band, "dataset_source": task.dataset_source, "group": task.group}


def _parse_patches(values: Sequence[str]) -> list[str]:
    if len(values) == 1 and str(values[0]).lower() == "all":
        return [f"{x},{y}" for x in range(9) for y in range(9)]
    out: list[str] = []
    for value in values:
        out.extend(part for part in str(value).split() if part)
    return out


def _make_tasks(args: argparse.Namespace) -> list[StoreTask]:
    patches = _parse_patches(args.patches)
    data_root = Path(args.data_root).expanduser().resolve()
    coadd_fits_root = Path(args.coadd_fits_root).expanduser().resolve() if args.coadd_fits_root else data_root
    output_root = Path(args.output_root).expanduser().resolve()
    refit_root = Path(args.refit_root).expanduser().resolve()
    denoised_root = Path(args.denoised_fits_root).expanduser().resolve()
    coadd_weight_root = Path(args.coadd_weight_root).expanduser().resolve()
    coadd_background_root = (
        Path(args.coadd_lsst_background_root).expanduser().resolve()
        if args.coadd_lsst_background_root
        else None
    )
    variant_background_root = (
        Path(args.variant_lsst_background_root).expanduser().resolve()
        if args.variant_lsst_background_root
        else None
    )
    gaia = Path(args.gaia_fits).expanduser().resolve() if args.gaia_fits else None
    tasks: list[StoreTask] = []
    for patch in patches:
        for band in args.bands:
            image_log_a = float(args.log_a) if math.isfinite(float(args.log_a)) else float(args.image_log_a)
            bright_log_a = float(args.log_a) if math.isfinite(float(args.log_a)) else float(args.bright_log_a)
            image_lupton_stretch = float(args.lupton_stretch) if args.lupton_stretch is not None else float(args.image_lupton_stretch)
            bright_lupton_stretch = float(args.lupton_stretch) if args.lupton_stretch is not None else float(args.bright_lupton_stretch)
            image_lupton_q = float(args.lupton_q) if args.lupton_q is not None else float(args.image_lupton_q)
            bright_lupton_q = float(args.lupton_q) if args.lupton_q is not None else float(args.bright_lupton_q)
            if "coadd" in args.dataset_sources:
                coadd_path = _coadd_image_path(coadd_fits_root, band, int(args.tract), patch)
                if not coadd_path.exists():
                    if args.missing_image_policy == "error":
                        raise FileNotFoundError(f"coadd image missing: {coadd_path}")
                    print(f"[preprocessing-v3] skip missing coadd image: patch={patch} band={band} path={coadd_path}", flush=True)
                else:
                    tasks.append(
                        StoreTask(
                            data_root=data_root,
                            coadd_fits_root=coadd_fits_root,
                            output_root=output_root,
                            refit_root=refit_root,
                            denoised_fits_root=denoised_root,
                            coadd_weight_root=coadd_weight_root,
                            coadd_lsst_background_root=coadd_background_root,
                            variant_lsst_background_root=variant_background_root,
                            gaia_fits=gaia,
                            tract=int(args.tract),
                            patch=patch,
                            band=band,
                            dataset_source="coadd",
                            group="",
                            tile_size=int(args.tile_size),
                            stride=int(args.stride),
                            max_tiles=int(args.max_tiles),
                            overwrite=bool(args.overwrite),
                            chunk_tiles=int(args.chunk_tiles),
                            image_scaling_mode=str(args.image_scaling_mode),
                            image_scaling_scope=str(args.image_scaling_scope),
                            bright_mask_mode=str(args.bright_mask_mode),
                            bright_threshold=float(args.bright_threshold),
                            bright_dilation=int(args.bright_dilation),
                            clip_threshold=float(args.clip_threshold),
                            image_log_a=image_log_a,
                            image_log_high_percentile=float(args.image_log_high_percentile),
                            image_lupton_stretch=image_lupton_stretch,
                            image_lupton_q=image_lupton_q,
                            image_anscombe_clip=bool(args.image_anscombe_clip),
                            image_anscombe_scale=float(args.image_anscombe_scale),
                            bright_log_a=bright_log_a,
                            bright_log_high_percentile=float(args.bright_log_high_percentile),
                            bright_lupton_stretch=bright_lupton_stretch,
                            bright_lupton_q=bright_lupton_q,
                            bright_anscombe_scale=float(args.bright_anscombe_scale),
                            cluster_source_match_pixels=float(args.cluster_source_match_pixels),
                            cluster_centroid_match_pixels=float(args.cluster_centroid_match_pixels),
                            gaia_bright_mag_threshold=float(args.gaia_bright_mag_threshold),
                            snr_method=str(args.snr_method),
                            missing_noncoadd_policy=str(args.missing_noncoadd_policy),
                            image_variant_background_source=str(args.image_variant_background_source),
                            missing_variant_background_policy=str(args.missing_variant_background_policy),
                        )
                    )
            for dataset_source in args.dataset_sources:
                if dataset_source == "coadd":
                    continue
                variant_groups = list(args.groups)
                if not variant_groups or variant_groups == ["all"]:
                    variant_groups = _variant_groups(denoised_root, patch, band=band, tract=int(args.tract), dataset_source=dataset_source)
                for group in variant_groups:
                    image_path = _variant_image_path(denoised_root, patch, group, band, dataset_source, int(args.tract))
                    if not image_path.exists():
                        if args.missing_image_policy == "error":
                            raise FileNotFoundError(f"variant image missing: {image_path}")
                        print(
                            f"[preprocessing-v3] skip missing {dataset_source} image: patch={patch} group={group} band={band} path={image_path}",
                            flush=True,
                        )
                        continue
                    tasks.append(
                        StoreTask(
                            data_root=data_root,
                            coadd_fits_root=coadd_fits_root,
                            output_root=output_root,
                            refit_root=refit_root,
                            denoised_fits_root=denoised_root,
                            coadd_weight_root=coadd_weight_root,
                            coadd_lsst_background_root=coadd_background_root,
                            variant_lsst_background_root=variant_background_root,
                            gaia_fits=gaia,
                            tract=int(args.tract),
                            patch=patch,
                            band=band,
                            dataset_source=dataset_source,
                            group=group,
                            tile_size=int(args.tile_size),
                            stride=int(args.stride),
                            max_tiles=int(args.max_tiles),
                            overwrite=bool(args.overwrite),
                            chunk_tiles=int(args.chunk_tiles),
                            image_scaling_mode=str(args.image_scaling_mode),
                            image_scaling_scope=str(args.image_scaling_scope),
                            bright_mask_mode=str(args.bright_mask_mode),
                            bright_threshold=float(args.bright_threshold),
                            bright_dilation=int(args.bright_dilation),
                            clip_threshold=float(args.clip_threshold),
                            image_log_a=image_log_a,
                            image_log_high_percentile=float(args.image_log_high_percentile),
                            image_lupton_stretch=image_lupton_stretch,
                            image_lupton_q=image_lupton_q,
                            image_anscombe_clip=bool(args.image_anscombe_clip),
                            image_anscombe_scale=float(args.image_anscombe_scale),
                            bright_log_a=bright_log_a,
                            bright_log_high_percentile=float(args.bright_log_high_percentile),
                            bright_lupton_stretch=bright_lupton_stretch,
                            bright_lupton_q=bright_lupton_q,
                            bright_anscombe_scale=float(args.bright_anscombe_scale),
                            cluster_source_match_pixels=float(args.cluster_source_match_pixels),
                            cluster_centroid_match_pixels=float(args.cluster_centroid_match_pixels),
                            gaia_bright_mag_threshold=float(args.gaia_bright_mag_threshold),
                            snr_method=str(args.snr_method),
                            missing_noncoadd_policy=str(args.missing_noncoadd_policy),
                            image_variant_background_source=str(args.image_variant_background_source),
                            missing_variant_background_policy=str(args.missing_variant_background_policy),
                        )
                    )
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/data/shared/Subaru")
    parser.add_argument(
        "--coadd-fits-root",
        default=None,
        help="Optional FITS root for coadd images. Catalogs/backgrounds still come from --data-root.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--refit-root", default="/data/czh23/refit")
    parser.add_argument("--denoised-fits-root", default="/data/czh23/denoised_fits")
    parser.add_argument("--coadd-weight-root", default=str(SnrConfig().coadd_weight_root))
    parser.add_argument(
        "--coadd-lsst-background-root",
        default=None,
        help=(
            "Optional root for coadd/half-coadd LSST detection backgrounds. "
            "Supported layout: <root>/{coadd,half_coadd}/<tract>/<patch>/coadd/<band>/background_mask.npz."
        ),
    )
    parser.add_argument(
        "--variant-lsst-background-root",
        default=None,
        help=(
            "Optional root for separately generated denoised/noisy LSST backgrounds. "
            "Expected layout: <root>/<variant>/<tract>/<patch>/<group>/<band>/background_mask.npz."
        ),
    )
    parser.add_argument("--gaia-fits", default="output/gaia_dr3_cosmos.fits")
    parser.add_argument("--tract", type=int, default=9813)
    parser.add_argument("--patches", nargs="+", default=["all"])
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    parser.add_argument("--dataset-sources", nargs="+", default=["coadd", "denoised", "noisy"], choices=["coadd", "denoised", "noisy"])
    parser.add_argument("--groups", nargs="*", default=["all"])
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=368)
    parser.add_argument("--chunk-tiles", type=int, default=16)
    parser.add_argument("--max-tiles", type=int, default=0, help="debug limit per output store; 0 means all tiles")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--image-scaling-mode", default="zscore-log-lupton-rgb")
    parser.add_argument(
        "--image-scaling-scope",
        default="patch",
        choices=["patch", "tile"],
        help="patch keeps image scaling consistent with full-patch bright labels; tile reproduces old direct-zarr local scaling for diagnostics.",
    )
    parser.add_argument("--bright-mask-mode", default="log-lupton")
    parser.add_argument("--bright-threshold", type=float, default=2.99)
    parser.add_argument("--bright-dilation", type=int, default=2)
    parser.add_argument("--clip-threshold", type=float, default=3.0, help="Clip image after scaling.")
    parser.add_argument("--log-a", type=float, default=float("nan"), help="Compatibility override for both --image-log-a and --bright-log-a.")
    parser.add_argument("--image-log-a", type=float, default=float("nan"), help="Log exponent for the RGB image written to zarr; default is per-band broad=1000, NB1010=100, NB0387=3000.")
    parser.add_argument("--bright-log-a", type=float, default=float("nan"), help="Log exponent for bright-region labels; default is broad=1000, NB1010=100, NB0387=3000.")
    parser.add_argument("--image-log-high-percentile", type=float, default=99.5)
    parser.add_argument("--bright-log-high-percentile", type=float, default=99.5)
    parser.add_argument("--lupton-stretch", type=float, default=None, help="Compatibility override for both image and bright Lupton stretch.")
    parser.add_argument("--lupton-q", type=float, default=None, help="Compatibility override for both image and bright Lupton Q.")
    parser.add_argument("--image-lupton-stretch", type=float, default=0.5)
    parser.add_argument("--image-lupton-q", type=float, default=20.0)
    parser.add_argument("--bright-lupton-stretch", type=float, default=0.5)
    parser.add_argument("--bright-lupton-q", type=float, default=20.0)
    parser.add_argument("--image-anscombe-clip", action="store_true", help="Clip image by removing pixels more than 3 raw standard deviations from the raw mean.")
    parser.add_argument("--image-anscombe-scale", type=float, default=1000.0)
    parser.add_argument("--bright-anscombe-scale", type=float, default=1000.0)
    parser.add_argument("--cluster-source-match-pixels", type=float, default=6.0)
    parser.add_argument("--cluster-centroid-match-pixels", type=float, default=10.0)
    parser.add_argument("--gaia-bright-mag-threshold", type=float, default=18.0)
    parser.add_argument("--snr-method", default="auto", choices=["auto", "variance", "weight", "none"])
    parser.add_argument("--missing-noncoadd-policy", default="fallback_coadd", choices=["fallback_coadd", "none", "error"])
    parser.add_argument(
        "--image-variant-background-source",
        default="auto",
        choices=["auto", "coadd-target", "variant-lsst", "none"],
        help=(
            "Background source for noisy/denoised dense labels. auto prefers variant LSST backgrounds "
            "and falls back according to --missing-variant-background-policy."
        ),
    )
    parser.add_argument(
        "--missing-variant-background-policy",
        default="fallback_coadd",
        choices=["fallback_coadd", "none", "error"],
        help="Fallback when a noisy/denoised background is missing in auto mode.",
    )
    parser.add_argument("--missing-image-policy", default="skip", choices=["skip", "error"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    tasks = _make_tasks(args)
    print(f"[preprocessing-v3] writing {len(tasks)} image-level store(s) to {args.output_root}", flush=True)
    results = []
    failures = []
    if int(args.workers) <= 1:
        for task in tasks:
            try:
                result = _build_store(task)
                results.append(result)
                print(
                    f"[preprocessing-v3] wrote {result['dataset_source']} {result['patch']} {result['band']} {result['group']} "
                    f"samples={result['samples']}",
                    flush=True,
                )
            except Exception as exc:
                failures.append({"patch": task.patch, "band": task.band, "dataset_source": task.dataset_source, "group": task.group, "error": str(exc)})
                print(f"[preprocessing-v3] FAILED {task.dataset_source} {task.patch} {task.band} {task.group}: {exc}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            future_by_task = {executor.submit(_build_store, task): task for task in tasks}
            for future in as_completed(future_by_task):
                task = future_by_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(
                        f"[preprocessing-v3] wrote {result['dataset_source']} {result['patch']} {result['band']} {result['group']} "
                        f"samples={result['samples']}",
                        flush=True,
                    )
                except Exception as exc:
                    failures.append({"patch": task.patch, "band": task.band, "dataset_source": task.dataset_source, "group": task.group, "error": str(exc)})
                    print(f"[preprocessing-v3] FAILED {task.dataset_source} {task.patch} {task.band} {task.group}: {exc}", flush=True)
    summary = {"results": results, "failures": failures}
    summary_path = Path(args.output_root).expanduser().resolve() / "preprocessing_v3_image_level_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)} image-level store(s) failed; see {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
