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
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy.io import fits
from astropy.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cellect")

from preprocessing.utils.inputs import (
    _band_catalog_path,
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
    build_bright_components,
)
from preprocessing.labels import SourceClass
from preprocessing.dataset_inputs import (
    ImageInput, load_paths, discover_cosmos, discover_abell, hsc_input,
    header_info, load_image as load_registered_image, bind_task,
)
from preprocessing.meas_processing import MeasProcessingConfig, classify_meas_basics
from preprocessing.ordinary import OrdinaryConfig, label_ordinary_sources
from preprocessing.refit import RefitConfig, attach_refit_geometry, compute_kron_ellipse
from preprocessing.region_filling import fill_dense_regions
from preprocessing.snr import SnrConfig, compute_snr_for_sample
from preprocessing.utils.catalog import source_ids
from preprocessing.zarr_writing import ImageLevelTrainingBatch, write_training_image_level_zarr
from preprocessing.utils.image_level import (
    _scale_image_chw,
    attach_cosmos_segmentation,
    StoreTask,
    PatchLabels,
    _is_narrow_band,
    _band_log_a,
    _image_log_a,
    _bright_log_a,
    _group_number,
    _product_patch_dir,
    _variant_groups,
    _variant_image_path,
    _coadd_image_path,
    _store_output_path,
    _refit_csv_path,
    _read_image_header_origin,
    _mask_plane_bits,
    _mask_plane_bit,
    _read_fits_quality_mask,
    _read_bright_object_mask,
    _read_background_from_det,
    _background_group_candidates,
    _variant_background_dirs,
    _variant_background_dir,
    _read_variant_background,
    _read_coadd_lsst_background,
    _subtract_bright_object_background,
    _background_for_task,
    _crop,
    _paint_confidence,
    _source_indices_in_tile,
    _strict_centers_in_tile,
    _tile_targets,
)


DEFAULT_BANDS = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y", "NB0387", "NB0816", "NB0921", "NB1010")
MASK_PLANES_FOR_STRICT_IGNORE = ("SAT", "BAD", "NO_DATA", "UNMASKEDNAN")




def _classify_patch(task: StoreTask, image: np.ndarray, image_header: fits.Header, origin: tuple[int, int], coadd_image_fits: Path, *, background_override=None) -> PatchLabels:
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
    stage = classify_meas_basics(table, config=MeasProcessingConfig(), refit_config=RefitConfig(),
                                 # Refitted x_image/y_image are already local.
                                 # The FITS origin is only used for global tile IDs;
                                 # subtracting it here marks every refit as no-data.
                                 image=image, origin=(0, 0))
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
    background = (_background_for_task(task, image.shape, origin)
                  if background_override is None else np.asarray(background_override, bool))
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
        bright_region_mask=bright_region,
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
        strict_is_gaia=np.array(["gaia" in str(r) for r in bright.strict_center_reason], bool),
    )


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
    return write_classified_patch(task, image, labels, origin, provenance={
        "image_fits": str(image_fits), "coadd_image_fits": str(coadd_image_fits),
        "refit_csv": str(_refit_csv_path(task.refit_root, task.tract, task.band, task.patch)),
    })


def write_classified_patch(task: StoreTask, image: np.ndarray, labels: PatchLabels,
                           origin=(0, 0), *, cosmos_catalog=None, image_wcs=None,
                           provenance=None, input_source: ImageInput | None = None,
                           tile_specs=None, valid_mask=None, max_invalid_fraction=None,
                           scaled_image_chw=None, include_large_sources=True) -> dict[str, object]:
    """Write already classified sources without rerunning HSC source filters.

    COSMOS callers supply the catalog and full-image WCS to generate isolated
    segmentation targets before tiling. Other datasets leave them unset.
    Scaling is performed here: image must be linear and already converted to
    HSC surface-brightness units for JWST, not RGB or previously scaled values.
    input_source namespaces JWST outputs by dataset/proposal and records both
    training-image and catalog-selection reference paths. Labels must already
    be on the training-image grid; this function does not rerun source filters.
    """
    if input_source is not None:
        task = bind_task(task, input_source)
        provenance = {**(provenance or {}), **input_source.to_dict()}
        if input_source.dataset in ('abell','cosmos') and max_invalid_fraction is None:
            max_invalid_fraction=.10
    if image.shape != labels.dense.shape:
        raise ValueError("image and dense target shapes must match")
    if cosmos_catalog is not None:
        if image_wcs is None:
            raise ValueError("COSMOS segmentation requires image_wcs")
        segmentation_valid = np.isfinite(image)
        if valid_mask is not None:
            if np.shape(valid_mask) != image.shape:
                raise ValueError("valid mask/image shape mismatch")
            segmentation_valid &= np.asarray(valid_mask, bool)
        attach_cosmos_segmentation(labels, image_wcs, cosmos_catalog,
                                   origin=origin, valid_mask=segmentation_valid)
    if (labels.segmentation_ids is None) != (labels.segmentation_weight is None):
        raise ValueError("segmentation IDs and weights must be supplied together")
    specs = list(tile_specs) if tile_specs is not None else make_tile_specs(
        parent_origin=origin,
        image_shape=(int(image.shape[1]), int(image.shape[0])),
        tile_size=task.tile_size,
        stride=task.stride,
        compare_origin=None,
    )
    if tile_specs is None and include_large_sources and input_source is not None and input_source.dataset in ('abell','cosmos'):
        from preprocessing.utils.large_sources import large_source_specs
        specs.extend(large_source_specs(labels,input_source.dataset,origin=origin,size=task.tile_size))
    if valid_mask is not None:
        valid_mask=np.asarray(valid_mask,bool)
        if valid_mask.shape != image.shape:raise ValueError('valid mask/image shape mismatch')
    if max_invalid_fraction is not None:
        from preprocessing.utils.large_sources import valid_tile_specs
        valid_mask=np.isfinite(image) if valid_mask is None else valid_mask
        specs=valid_tile_specs(specs,valid_mask,origin,max_invalid_fraction)
    dataset = input_source.dataset if input_source is not None else (provenance or {}).get('dataset')
    if dataset in ('cosmos', 'abell'):
        from preprocessing.utils.truncated_kron import filter_truncated_kron_tiles
        specs, truncation_audit = filter_truncated_kron_tiles(labels, specs, origin)
        provenance = {**(provenance or {}), 'truncated_kron_filter': truncation_audit}
        audit_output = _store_output_path(task.output_root, task.patch, task.band, task.dataset_source, task.group)
        audit_output.parent.mkdir(parents=True, exist_ok=True)
        Path(str(audit_output) + '_tile_filter.json').write_text(json.dumps(truncation_audit, indent=2))
    if task.max_tiles > 0:
        specs = specs[: int(task.max_tiles)]
    n = len(specs)
    if n == 0:
        if max_invalid_fraction is not None:
            return {'samples':0,'status':'no_valid_tiles','patch':task.patch,'band':task.band}
        raise RuntimeError(f"no tile specs generated for {task.patch} {task.band} {task.dataset_source} {task.group}")
    from preprocessing.utils.confidence import resolve_confidence
    confidence_config = resolve_confidence(
        task, image_wcs=image_wcs,
        image_path=(input_source.image_fits if input_source is not None else (provenance or {}).get('image_fits')),
    )
    h = w = int(task.tile_size)
    scaling_scope = str(task.image_scaling_scope).strip().lower().replace("_", "-")
    if scaling_scope not in {"patch", "tile"}:
        raise ValueError(f"unknown image scaling scope: {task.image_scaling_scope}")
    if scaling_scope == "patch":
        full_scaled_chw = _scale_image_chw(image, task) if scaled_image_chw is None else np.asarray(scaled_image_chw)
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
    band_valid = np.zeros((n,1,h,w),bool) if valid_mask is not None else None
    band_seg_ids = band_seg_weight = None
    overlap_rows = []
    if labels.segmentation_ids is not None:
        band_seg_ids = np.zeros((n, 1, h, w), dtype=np.int32)
        band_seg_weight = np.zeros((n, 1, h, w), dtype=np.float32)
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
        target = _tile_targets(labels, spec, origin, confidence=confidence_config)
        if valid_mask is not None:
            v=_crop(valid_mask,spec.x0,spec.y0,origin,h)
            band_valid[i,0]=v
            images[i,0,...,~v]=0
            target['confidence'][~v]=0
            target['conf_weight'][~v]=0
            target['shape_weight'][~v]=0
            from preprocessing.labels import DenseLabel
            target['pu'][~v]=int(DenseLabel.STRICT_IGNORE)
            if 'segmentation_weight' in target:target['segmentation_weight'][~v]=0
            for center_key,others in [('source_centers',['source_ids']),('strict_centers',['strict_ids']),
                                      ('shape_centers',['shape_values','shape_classes','shape_ids'])]:
                centers=target[center_key]
                xy=np.clip(np.floor(centers+.5).astype(int),0,h-1)
                keep=v[xy[:,1],xy[:,0]]
                if center_key == 'strict_centers':
                    keep |= target['strict_is_gaia']
                for key in [center_key]+others:target[key]=target[key][keep]
        band_conf[i, 0] = target["confidence"]
        band_conf_weight[i, 0] = target["conf_weight"]
        band_shape[i, 0] = target["shape"]
        band_shape_weight[i, 0] = target["shape_weight"]
        band_pu[i, 0] = target["pu"]
        if band_seg_ids is not None:
            band_seg_ids[i, 0] = target["segmentation_ids"]
            band_seg_weight[i, 0] = target["segmentation_weight"]
        for row in target.get('segmentation_overlap_masks', []):
            if valid_mask is not None:
                y, x = row['y0'], row['x0']; mh, mw = row['mask'].shape
                row['mask'] &= v[y:y+mh, x:x+mw]
            if row['mask'].any():
                overlap_rows.append(dict(row, sample=i, band=0))
        tile_x0[i] = int(spec.x0)
        tile_y0[i] = int(spec.y0)
        tile_names.append(spec.name)
        groups.append("" if task.dataset_source == "coadd" else task.group)
        sources.append(task.dataset_source)
        prefix = task.band if task.dataset_source == "coadd" else f"{task.group}_{task.band}"
        if input_source is not None:
            prefix = input_source.sample_name
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
    from preprocessing.utils.segmentation_storage import pack_overlap_masks
    batch = ImageLevelTrainingBatch(
        images=images,
        band_confidence=band_conf,
        band_conf_weight=band_conf_weight,
        band_shape=band_shape,
        band_shape_weight=band_shape_weight,
        band_pu_class_mask=band_pu,
        band_valid_mask=band_valid,
        band_segmentation_ids=band_seg_ids,
        band_segmentation_weight=band_seg_weight,
        segmentation_overlap=(pack_overlap_masks(overlap_rows)
                              if labels.segmentation_overlap_masks is not None else None),
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
            "subtract_bright_object_from_background": bool(task.subtract_bright_object_from_background),
            "bright_object_mask_root": str(task.bright_object_mask_root) if task.bright_object_mask_root is not None else "",
            "tile_size": int(task.tile_size),
            "max_invalid_fraction": max_invalid_fraction,
            "stride": int(task.stride),
            "source_export_mode": "preprocessing_v3",
            **(provenance or {}),
            "confidence_config": confidence_config,
            "segmentation_policy": labels.segmentation_policy,
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
    bright_object_mask_root = (
        Path(args.bright_object_mask_root).expanduser().resolve()
        if args.bright_object_mask_root
        else data_root
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
                            bright_object_mask_root=bright_object_mask_root,
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
                            subtract_bright_object_from_background=bool(args.subtract_bright_object_from_background),
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
                            bright_object_mask_root=bright_object_mask_root,
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
                            subtract_bright_object_from_background=bool(args.subtract_bright_object_from_background),
                            image_variant_background_source=str(args.image_variant_background_source),
                            missing_variant_background_policy=str(args.missing_variant_background_policy),
                        )
                    )
    return [replace(task, confidence_mode=args.confidence_mode,
                    confidence_config_path=args.confidence_config_path,
                    confidence_fwhm_min=args.confidence_fwhm_min,
                    confidence_fwhm_max=args.confidence_fwhm_max,
                    confidence_fwhm_pixels=args.confidence_fwhm_pixels) for task in tasks]


def parse_args(argv=None) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--paths-config')
    preliminary, _ = bootstrap.parse_known_args(argv)
    paths = load_paths(preliminary.paths_config)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths-config', help='Partial overrides of preprocessing/dataset_paths.json')
    parser.add_argument('--datasets', nargs='+', choices=['hsc','cosmos','abell'], default=['hsc'])
    parser.add_argument('--training-batch', action='store_true', help='Four-family production runner with precomputed SExtractor masks and independent centered labels')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--background-root', default='/data/czh23/analysis/2026-09/2026-09-25/batch_aggressive_background')
    parser.add_argument('--hsc-products-root', default='/data/czh23/Subaru_products')
    parser.add_argument('--training-kinds', nargs='+', choices=['hsc_half','hsc_noisy','cosmos','abell'], default=['hsc_half','hsc_noisy','cosmos','abell'])
    parser.add_argument('--job-limit', type=int, default=0, help='Pilot job limit; 0 means all')
    parser.add_argument('--large-only', action='store_true', help='Pilot: write centered large sources without ordinary grid')
    parser.add_argument('--cross-boundary-only', action='store_true', help='Pilot: retain large stamps crossing a parent boundary')
    parser.add_argument('--large-limit', type=int, default=0, help='Pilot centered-stamp limit per parent; 0 means all')
    parser.add_argument('--parent-xy', nargs=2, type=int, help='COSMOS pilot parent lower-left x y')
    parser.add_argument('--diagnostic-dir', help='Optional two-panel centered-stamp diagnostics')
    parser.add_argument('--list-inputs', action='store_true', help='Discover/check images and references; write inventory only, no labels/Zarr')
    parser.add_argument('--input-manifest-out', help='Inventory JSON path (default: output-root/input_inventory.json)')
    parser.add_argument('--jwst-bands', nargs='+', help='Effective JWST bands; default all available')
    parser.add_argument('--cosmos-proposals', type=int, nargs='+', choices=[1727,5893])
    parser.add_argument('--cosmos-pointings', type=int, nargs='+')
    parser.add_argument('--abell-stage', choices=['cutouts','zarr','all'], default='all')
    parser.add_argument('--abell-plan-only', action='store_true')
    parser.add_argument('--abell-parent', nargs='+', help='Optional aligned parent IDs for pilots, e.g. x+00_y+00')
    parser.add_argument('--abell-anchor', type=int, nargs=2, default=[8064,19102], metavar=('X0','Y0'))
    parser.add_argument('--parent-size', type=int, default=4096)
    parser.add_argument('--parent-overlap', type=int, default=128)
    parser.add_argument('--jwst-max-invalid-fraction', type=float, default=0.10)
    parser.add_argument('--confidence-mode', choices=['auto', 'manhattan', 'psf-matched', 'psf-ee'], default='auto',
                        help='auto: JWST asset EE rings; HSC original Manhattan. psf-matched selects legacy FWHM rings.')
    parser.add_argument('--confidence-config-path', help='Optional EE or legacy FWHM asset JSON; default bundled asset for selected mode')
    parser.add_argument('--confidence-fwhm-min', type=float, default=1.6, help='PSF confidence FWHM lower limit in output pixels')
    parser.add_argument('--confidence-fwhm-max', type=float, default=8.0, help='PSF confidence FWHM upper limit in output pixels')
    parser.add_argument('--confidence-fwhm-pixels', type=float, default=None,
                        help='Optional measured FWHM override in output pixels; otherwise use nominal JWST FWHM / WCS scale')
    parser.add_argument('--sex-detect-thresh', type=float, default=1.5)
    parser.add_argument('--sex-minarea', type=int, default=5)
    parser.add_argument('--sex-back-size', type=int, default=64)
    parser.add_argument('--sex-grow', type=int, default=0)
    parser.add_argument('--abell-catalog', default='/data/shared/jwst_foundation/catalog/detect/field_ra3p573_dec-30p376_det_cat.fits')
    parser.add_argument('--abell-gaia', default='/home/czh23/CELLECT/output/gaia_dr3_abell2744.fits')
    parser.add_argument('--abell-background-root', default='/data/czh23/JWST/lsst_background_masks/jwst/default/Abell2744/group_00')
    parser.add_argument("--data-root", default=paths['hsc']['data_root'])
    parser.add_argument(
        "--coadd-fits-root",
        default=paths['hsc']['coadd_fits_root'],
        help="Optional FITS root for coadd images. Catalogs/backgrounds still come from --data-root.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--refit-root", default=paths['hsc']['refit_root'])
    parser.add_argument("--denoised-fits-root", default=paths['hsc']['denoised_fits_root'])
    parser.add_argument("--coadd-weight-root", default=paths['hsc']['coadd_weight_root'])
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
    parser.add_argument(
        "--subtract-bright-object-from-background",
        action="store_true",
        help=(
            "Remove official coadd BRIGHT_OBJECT pixels from any selected background mask. "
            "This is off by default."
        ),
    )
    parser.add_argument(
        "--bright-object-mask-root",
        default=None,
        help=(
            "Root of the official coadd FITS files used for BRIGHT_OBJECT subtraction. "
            "Defaults to --data-root."
        ),
    )
    parser.add_argument("--gaia-fits", default=paths['hsc']['gaia_fits'])
    parser.add_argument("--tract", type=int, default=9813)
    parser.add_argument("--patches", nargs="+", default=["all"])
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    parser.add_argument("--dataset-sources", nargs="+", default=["coadd", "denoised", "noisy"], choices=["coadd", "denoised", "noisy"])
    parser.add_argument("--groups", nargs="*", default=["all"])
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=368)
    parser.add_argument("--chunk-tiles", type=int, default=20)
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
    args = parser.parse_args(argv)
    from preprocessing.utils.confidence import validate_confidence
    try:
        validate_confidence(args.confidence_mode, args.confidence_fwhm_min,
                            args.confidence_fwhm_max, args.confidence_fwhm_pixels)
    except ValueError as exc:
        parser.error(str(exc))
    args.paths = paths
    return args


def discover_inputs(args) -> list[ImageInput]:
    """One registry for HSC variants, curated COSMOS and Abell half/full pairs."""
    sources = []
    if 'hsc' in args.datasets:
        sources.extend(hsc_input(task) for task in _make_tasks(args))
    if 'cosmos' in args.datasets:
        sources.extend(discover_cosmos(args.paths['cosmos'], proposals=args.cosmos_proposals,
            bands=args.jwst_bands, pointings=args.cosmos_pointings))
    if 'abell' in args.datasets:
        sources.extend(discover_abell(args.paths['abell'], bands=args.jwst_bands))
    if not sources:
        raise ValueError('No images matched the selected datasets/filters')
    names = [s.sample_name for s in sources]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate input sample identifiers')
    return sources


def write_input_inventory(args):
    sources = discover_inputs(args)
    rows = []
    for source in sources:
        header, shape, idx = header_info(source.image_fits)
        rh, rshape, ridx = header_info(source.reference_fits)
        rows.append({**source.to_dict(), 'shape': list(shape), 'image_hdu': idx,
                     'bunit': header.get('BUNIT'), 'reference_shape': list(rshape),
                     'reference_hdu': ridx, 'reference_bunit': rh.get('BUNIT')})
    output = Path(args.input_manifest_out or Path(args.output_root)/'input_inventory.json').expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(schema=1, paths=args.paths, items=rows,
                  counts={name:sum(s.dataset==name for s in sources) for name in args.datasets})
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(f'[preprocessing-v3] inputs only: {result["counts"]}; inventory={output}',flush=True)
    return result


def main() -> int:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if args.training_batch:
        from preprocessing.training_batch import run
        return run(args)
    if args.list_inputs:
        write_input_inventory(args)
        return 0
    if args.datasets == ['abell']:
        from preprocessing.abell_cutouts import run_cutouts
        if args.abell_stage in ('cutouts','all') or args.abell_plan_only:
            run_cutouts(args)
        if args.abell_stage in ('zarr','all') and not args.abell_plan_only:
            from preprocessing.abell_zarr import run_abell_zarr
            run_abell_zarr(args)
        return 0
    if args.datasets != ['hsc']:
        raise ValueError('JWST input loading is available via --list-inputs and load_registered_image. '
                         'For Zarr, supply final dataset-specific PatchLabels to write_classified_patch(input_source=...). '
                         'HSC catalog filtering must not be applied to JWST images.')
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
