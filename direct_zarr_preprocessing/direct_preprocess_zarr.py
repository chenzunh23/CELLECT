#!/usr/bin/env python
"""Direct CELLECT preprocessing to patch-level Zarr stores.

This script intentionally bypasses the legacy cutout/NPZ/FITS output tree.
It reuses the established catalog filtering and target painting functions, but
writes only compact Zarr v2 stores for training and visualization.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from astropy.table import Table, vstack

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astro_cellect2d import astro_zscale_preprocess  # noqa: E402
from astro_data_preprocessing import (  # noqa: E402
    TileSpec,
    _band_catalog_path,
    _band_det_path,
    _band_fits_path,
    _apply_external_bright_labels,
    _classify_pu_catalog,
    _configure_worker_threads,
    _crop_full_mask_for_tile,
    _crop_image_for_tile,
    _find_denoised_patch_dir,
    _fits_open_with_path_warnings,
    _find_image_hdu_index,
    _metadata_from_catalog,
    _move_bright_clean_to_center_only,
    _origin_from_ltv,
    _paint_ellipse_mask,
    _pu_dropped_sources,
    _read_det_background_mask,
    _read_external_bright_mask,
    _read_exposure_image_plane,
    _restore_center_only_shape_targets,
    _read_table,
    _variant_lsst_background_mask,
    _vstack_nonempty,
    add_ellipse_columns,
    build_pu_runtime_config,
    crop_catalog_for_tile,
    make_pu_dense_targets,
    make_tile_specs,
)
from data_filtering.noncoadd_snr import (  # noqa: E402
    classify_clean_by_noncoadd_snr as _classify_clean_by_noncoadd_snr,
    read_lsst_quality_mask as _quality_mask_from_lsst_fits,
    source_annulus_exclusion_mask as _source_annulus_exclusion_mask,
)
from data_filtering.calexp_quality import (  # noqa: E402
    patch_and_tile_scores,
    parse_score_weights,
)
from data_filtering.sam_input_scaling import build_bright_mask, scale_training_image  # noqa: E402
from direct_zarr_preprocessing.zarr_writer import ZarrGroupWriter, encode_fixed_utf8, write_json  # noqa: E402

DEFAULT_BANDS = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y")


@dataclass(frozen=True)
class Sample:
    name: str
    spec: TileSpec
    dataset_source: str
    group: str
    image_key: str
    band: str = ""


def _expand_patch_tokens(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for raw in values:
        for token in str(raw).replace(";", " ").split():
            if token.lower() == "all":
                out.extend(f"{x},{y}" for x in range(9) for y in range(9))
            elif token:
                out.append(token)
    seen = set()
    unique = []
    for patch in out:
        if patch not in seen:
            seen.add(patch)
            unique.append(patch)
    return unique


def _read_patch_image_meta(coadd_root: Path, band: str, tract: int, patch: str) -> tuple[tuple[int, int], tuple[int, int]]:
    path = _band_fits_path(coadd_root, band, tract, patch)
    with _fits_open_with_path_warnings(path, memmap=True, ignore_missing_end=True) as hdul:
        hdu = hdul[_find_image_hdu_index(hdul)]
        origin = _origin_from_ltv(hdu.header)
        shape = tuple(int(v) for v in hdu.data.shape)
    return shape, origin


def _classify_all_bands(args: argparse.Namespace, runtime: argparse.Namespace, patch: str) -> dict[str, tuple[Table, Table, Table, Table]]:
    catalog_root = Path(args.band_catalog_root or args.catalog_root).expanduser()
    image_root = Path(args.coadd_root).expanduser()
    out = {}
    for band in args.bands:
        catalog_path = _band_catalog_path(catalog_root, band, int(args.tract), patch)
        if not catalog_path.exists() and str(getattr(args, "missing_band_policy", "error")) == "skip":
            print(f"[direct-zarr] skip missing band catalog for {patch} {band}: {catalog_path}", flush=True)
            continue
        table = _read_table(catalog_path, hdu=int(args.catalog_hdu), role="direct-zarr-band", patch=patch, band=band)
        bright_mask = None
        bright_origin = None
        if bool(getattr(args, "pu_enable_bright_background_mask", False)):
            try:
                image, bright_origin = _read_exposure_image_plane(
                    _band_fits_path(image_root, band, int(args.tract), patch),
                    clean_nonfinite=True,
                )
                bright_mask = build_bright_mask(
                    image,
                    mode=str(getattr(args, "pu_bright_mask_mode", "log-lupton")),
                    threshold=float(getattr(args, "pu_bright_z_threshold", 3.0)),
                    dilation=int(getattr(args, "pu_bright_mask_dilate", 2)),
                    log_a=float(getattr(args, "pu_bright_log_a", 300.0)),
                    log_high_percentile=float(getattr(args, "pu_bright_log_high_percentile", 99.5)),
                    lupton_stretch=float(getattr(args, "pu_bright_lupton_stretch", 0.5)),
                    lupton_q=float(getattr(args, "pu_bright_lupton_q", 20.0)),
                    anscombe_scale=float(getattr(args, "pu_bright_anscombe_scale", 1000.0)),
                )
            except Exception as exc:
                print(
                    f"WARNING: failed to build coadd bright mask for catalog filtering {patch} {band}: {exc}",
                    flush=True,
                )
                bright_mask = None
                bright_origin = None
        if getattr(args, "external_bright_label_root", None) is not None:
            try:
                external_mask = _read_external_bright_mask(args, band=band, patch=patch)
            except Exception as exc:
                print(f"WARNING: failed to read external bright mask for {patch} {band}: {exc}", flush=True)
                external_mask = None
            if external_mask is not None:
                if bright_mask is not None and bright_mask.shape != external_mask.shape:
                    print(
                        f"WARNING: external bright mask shape {external_mask.shape} differs from coadd bright mask "
                        f"{bright_mask.shape} for {patch} {band}; ignoring external mask",
                        flush=True,
                    )
                else:
                    bright_mask = external_mask if bright_mask is None else (np.asarray(bright_mask, dtype=bool) | external_mask)
                    bright_origin = bright_origin or _read_patch_image_meta(image_root, band, int(args.tract), patch)[1]
        clean, center, ignore, pu_all, _result = _classify_pu_catalog(
            table,
            runtime,
            band=band,
            patch=patch,
            bright_region_mask=bright_mask,
            bright_region_origin=bright_origin,
        )
        clean, center, strict_center = _move_bright_clean_to_center_only(clean, center, runtime, band=band)
        clean, center, ignore, strict_center, _external_stats = _apply_external_bright_labels(
            clean,
            center,
            ignore,
            strict_center,
            pu_all,
            runtime,
            band=band,
            patch=patch,
        )
        out[band] = (clean, center, ignore, strict_center)
    if not out:
        raise RuntimeError(f"No usable band catalogs found for patch {patch}")
    return out


def _read_quality_masks(
    args: argparse.Namespace,
    patch: str,
) -> dict[str, tuple[np.ndarray, tuple[int, int]]]:
    if not bool(args.noncoadd_snr_use_quality_mask):
        return {}
    root = Path(args.coadd_root).expanduser()
    planes = tuple(str(plane) for plane in args.noncoadd_snr_mask_planes if str(plane).strip())
    out: dict[str, tuple[np.ndarray, tuple[int, int]]] = {}
    for band in args.bands:
        path = _band_fits_path(root, band, int(args.tract), patch)
        mask = _quality_mask_from_lsst_fits(path, planes)
        if mask is None:
            print(f"WARNING: no usable quality mask for {patch} {band}: {path}", flush=True)
            continue
        shape_yx, origin_xy = _read_patch_image_meta(root, band, int(args.tract), patch)
        if tuple(mask.shape) != tuple(shape_yx):
            print(
                f"WARNING: quality mask shape {mask.shape} != image shape {shape_yx} for {path}; ignoring mask",
                flush=True,
            )
            continue
        out[band] = (mask, origin_xy)
    return out


def _read_backgrounds(
    args: argparse.Namespace,
    patch: str,
    shape_yx: tuple[int, int],
    origin_xy: tuple[int, int],
) -> dict[str, tuple[np.ndarray, tuple[int, int]]]:
    out = {}
    if args.background_policy == "none":
        return out
    root = Path(args.coadd_root).expanduser()
    for band in args.bands:
        det_path = _band_det_path(root, band, int(args.tract), patch)
        if det_path is None:
            print(f"WARNING: no det background for {patch} {band}; non-source pixels will be ignore", flush=True)
            continue
        out[band] = (_read_det_background_mask(det_path, shape_yx, origin_xy=origin_xy), origin_xy)
    return out


def _variant_backgrounds_for_images(
    args: argparse.Namespace,
    *,
    variant: str,
    patch: str,
    variant_images: dict[str, dict[str, tuple[np.ndarray, tuple[int, int]]]],
    coadd_backgrounds: dict[str, tuple[np.ndarray, tuple[int, int]]],
) -> dict[str, dict[str, tuple[np.ndarray, tuple[int, int]]]]:
    source = str(args.image_variant_background_source).strip().lower()
    if source not in {"auto", "coadd-target", "variant-lsst", "none"}:
        raise ValueError(f"Unknown image_variant_background_source: {source}")
    out: dict[str, dict[str, tuple[np.ndarray, tuple[int, int]]]] = {}
    for group, images_by_band in variant_images.items():
        group_backgrounds: dict[str, tuple[np.ndarray, tuple[int, int]]] = {}
        if source == "none":
            out[group] = group_backgrounds
            continue
        if source in {"auto", "variant-lsst"}:
            for band, (image, image_origin) in images_by_band.items():
                try:
                    cached = _variant_lsst_background_mask(
                        args,
                        variant=variant,
                        tract=int(args.tract),
                        patch=patch,
                        group=group,
                        band=band,
                        shape_yx=(int(image.shape[0]), int(image.shape[1])),
                        image_origin=image_origin,
                        image_path=_variant_fits_path(args, patch, variant, group, band),
                    )
                except Exception as exc:
                    print(
                        f"WARNING: failed to read variant LSST background for "
                        f"{variant}/{args.tract}/{patch}/{group}/{band}: {exc}",
                        flush=True,
                    )
                    cached = None
                if cached is not None:
                    group_backgrounds[band] = cached
        if source == "coadd-target" or (source == "auto" and len(group_backgrounds) < len(args.bands)):
            for band in args.bands:
                if band not in group_backgrounds and band in coadd_backgrounds:
                    group_backgrounds[band] = coadd_backgrounds[band]
        out[group] = group_backgrounds
    return out


def _bright_backgrounds_for_images(
    args: argparse.Namespace,
    images_by_key: dict[str, dict[str, tuple[np.ndarray, tuple[int, int]]]],
    *,
    patch: str,
) -> dict[str, dict[str, tuple[np.ndarray, tuple[int, int]]]]:
    out: dict[str, dict[str, tuple[np.ndarray, tuple[int, int]]]] = {}
    use_image_bright = bool(getattr(args, "pu_enable_bright_background_mask", False))
    use_external_bright = getattr(args, "external_bright_label_root", None) is not None
    if not use_image_bright and not use_external_bright:
        return {key: {} for key in images_by_key}
    for key, band_images in images_by_key.items():
        per_band: dict[str, tuple[np.ndarray, tuple[int, int]]] = {}
        for band, (image, origin) in band_images.items():
            bright_mask = None
            if use_image_bright:
                bright_mask = build_bright_mask(
                    image,
                    mode=str(getattr(args, "pu_bright_mask_mode", "log-lupton")),
                    threshold=float(getattr(args, "pu_bright_z_threshold", 3.0)),
                    dilation=int(getattr(args, "pu_bright_mask_dilate", 2)),
                    log_a=float(getattr(args, "pu_bright_log_a", 300.0)),
                    log_high_percentile=float(getattr(args, "pu_bright_log_high_percentile", 99.5)),
                    lupton_stretch=float(getattr(args, "pu_bright_lupton_stretch", 0.5)),
                    lupton_q=float(getattr(args, "pu_bright_lupton_q", 20.0)),
                    anscombe_scale=float(getattr(args, "pu_bright_anscombe_scale", 1000.0)),
                )
            if use_external_bright:
                try:
                    external_mask = _read_external_bright_mask(args, band=band, patch=patch)
                except Exception as exc:
                    print(f"WARNING: failed to read external bright mask for zarr target {patch} {band}: {exc}", flush=True)
                    external_mask = None
                if external_mask is not None:
                    if tuple(external_mask.shape) != tuple(image.shape):
                        print(
                            f"WARNING: external bright mask shape {external_mask.shape} differs from image shape "
                            f"{image.shape} for {patch} {key} {band}; ignoring external mask",
                            flush=True,
                        )
                    else:
                        bright_mask = external_mask if bright_mask is None else (np.asarray(bright_mask, dtype=bool) | external_mask)
            if bright_mask is not None:
                per_band[band] = (np.asarray(bright_mask, dtype=bool), origin)
        out[key] = per_band
    return out


def _replicate_coadd_bright_backgrounds(
    coadd_bright_backgrounds: dict[str, dict[str, tuple[np.ndarray, tuple[int, int]]]],
    image_sources: dict[str, dict[str, tuple[np.ndarray, tuple[int, int]]]],
) -> dict[str, dict[str, tuple[np.ndarray, tuple[int, int]]]]:
    coadd_entry = coadd_bright_backgrounds.get("coadd", {})
    return {
        key: {band: coadd_entry[band] for band in band_images if band in coadd_entry}
        for key, band_images in image_sources.items()
    }


def _read_coadd_images(args: argparse.Namespace, patch: str) -> dict[str, tuple[np.ndarray, tuple[int, int]]]:
    root = Path(args.coadd_root).expanduser()
    out = {}
    for band in args.bands:
        path = _band_fits_path(root, band, int(args.tract), patch)
        if not path.exists() and str(getattr(args, "missing_band_policy", "error")) == "skip":
            print(f"[direct-zarr] skip missing coadd image for {patch} {band}: {path}", flush=True)
            continue
        out[band] = _read_exposure_image_plane(path, clean_nonfinite=True)
    if not out:
        raise FileNotFoundError(f"No usable coadd images found for patch {patch}")
    return out


def _read_variant_images(args: argparse.Namespace, patch: str, variant: str) -> dict[str, dict[str, tuple[np.ndarray, tuple[int, int]]]]:
    root = Path(args.denoised_fits_root).expanduser()
    patch_dir = _find_denoised_patch_dir(root, patch)
    if patch_dir is None:
        raise FileNotFoundError(f"variant patch directory not found for {patch} under {root}")
    groups = sorted(path for path in patch_dir.iterdir() if path.is_dir() and path.name.startswith("group_"))
    if not groups:
        groups = sorted(path for path in patch_dir.iterdir() if path.is_dir())
    if args.image_variant_groups:
        wanted = {g if g.startswith("group_") else f"group_{int(g):02d}" for g in args.image_variant_groups}
        groups = [g for g in groups if g.name in wanted]
    if not groups:
        raise FileNotFoundError(f"No requested image variant group directories found in {patch_dir}")
    out = {}
    for group in groups:
        band_images = {}
        for band in args.bands:
            path = group / band / f"{variant}.fits"
            if not path.exists():
                if str(getattr(args, "missing_band_policy", "error")) == "skip":
                    print(f"[direct-zarr] skip missing {variant} FITS for {patch} {group.name} {band}: {path}", flush=True)
                    continue
                raise FileNotFoundError(f"missing {variant} FITS: {path}")
            band_images[band] = _read_exposure_image_plane(path, clean_nonfinite=True)
        if band_images:
            out[group.name] = band_images
    if not out:
        raise FileNotFoundError(f"No usable {variant} images found for patch {patch}")
    return out


def _variant_fits_path(args: argparse.Namespace, patch: str, variant: str, group: str, band: str) -> Optional[Path]:
    if args.denoised_fits_root is None:
        return None
    patch_dir = _find_denoised_patch_dir(Path(args.denoised_fits_root).expanduser(), patch)
    if patch_dir is None:
        return None
    path = patch_dir / group / band / f"{variant}.fits"
    return path if path.exists() else None


def _make_samples(dataset: str, specs: Sequence[TileSpec], image_keys: Sequence[str]) -> list[Sample]:
    samples = []
    for key in image_keys:
        for spec in specs:
            name = spec.name if dataset == "coadd" else f"{key}_{spec.name}"
            samples.append(Sample(name=name, spec=spec, dataset_source=dataset, group="" if dataset == "coadd" else key, image_key=key))
    return samples


def _make_image_level_samples(dataset: str, specs: Sequence[TileSpec], image_key: str, band: str) -> list[Sample]:
    samples = []
    for spec in specs:
        prefix = f"{band}_" if dataset == "coadd" else f"{image_key}_{band}_"
        samples.append(
            Sample(
                name=f"{prefix}{spec.name}",
                spec=spec,
                dataset_source=dataset,
                group="" if dataset == "coadd" else image_key,
                image_key=image_key,
                band=band,
            )
        )
    return samples


def _quality_allowed_tiles(
    args: argparse.Namespace,
    patch: str,
    specs: Sequence[TileSpec],
) -> dict[str, dict[str, object]]:
    if not bool(getattr(args, "quality_filter", False)):
        return {}
    threshold = float(getattr(args, "quality_bad_score_threshold", 0.13))
    weights = parse_score_weights(getattr(args, "quality_bad_score_weights", None))
    root = Path(args.coadd_root).expanduser()
    out: dict[str, dict[str, object]] = {}
    for band in args.bands:
        try:
            shape_yx, origin = _read_patch_image_meta(root, band, int(args.tract), patch)
            starts = [(int(spec.x0 - origin[0]), int(spec.y0 - origin[1])) for spec in specs]
            patch_score, tile_scores = patch_and_tile_scores(
                _band_fits_path(root, band, int(args.tract), patch),
                starts=starts,
                tile_size=int(args.tile_size),
                weights=weights,
            )
        except Exception as exc:
            policy = str(getattr(args, "quality_filter_missing_policy", "keep")).strip().lower()
            if policy == "error":
                raise
            keep = policy != "drop"
            print(
                f"[direct-zarr] quality filter {policy} for {patch} {band}: {exc}",
                flush=True,
            )
            out[band] = {
                "patch_score": float("nan"),
                "patch_allowed": keep,
                "tile_scores": {},
                "allowed_tile_names": {spec.name for spec in specs} if keep else set(),
            }
            continue
        allowed = set()
        patch_allowed = bool(patch_score < threshold)
        for spec in specs:
            local_key = (int(spec.x0 - origin[0]), int(spec.y0 - origin[1]))
            score = float(tile_scores.get(local_key, float("inf")))
            if patch_allowed and score < threshold:
                allowed.add(spec.name)
        out[band] = {
            "patch_score": float(patch_score),
            "patch_allowed": patch_allowed,
            "tile_scores": {spec.name: float(tile_scores.get((int(spec.x0 - origin[0]), int(spec.y0 - origin[1])), float("nan"))) for spec in specs},
            "allowed_tile_names": allowed,
        }
    return out


def _quality_allows(quality: dict[str, dict[str, object]], band: str, spec: TileSpec) -> bool:
    if not quality:
        return True
    entry = quality.get(band)
    if entry is None:
        return True
    allowed = entry.get("allowed_tile_names", set())
    return str(spec.name) in allowed


def _quality_filter_specs(
    specs: Sequence[TileSpec],
    quality: dict[str, dict[str, object]],
    bands: Sequence[str],
) -> list[TileSpec]:
    if not quality:
        return list(specs)
    return [spec for spec in specs if all(_quality_allows(quality, band, spec) for band in bands)]


def _crop_sources(source_tuple, spec: TileSpec, args: argparse.Namespace, *, margin: float) -> tuple[Table, Table, Table, Table]:
    return tuple(
        crop_catalog_for_tile(part, spec, x_col=args.x_col, y_col=args.y_col, margin=margin)
        for part in source_tuple
    )


def _shape_source_metadata(
    clean_sources: Table,
    center_only_sources: Table,
    target: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers_parts = []
    values_parts = []
    classes_parts = []
    ids_parts = []
    h, w = np.asarray(target["pu_class_mask"]).shape
    for table, class_id in ((clean_sources, 1), (center_only_sources, 2)):
        if len(table) == 0:
            continue
        if class_id == 2 and "pu_no_shape_supervision" in table.colnames:
            usable_shape = ~np.asarray(table["pu_no_shape_supervision"], dtype=bool)
            table = table[usable_shape]
            if len(table) == 0:
                continue
        meta = _metadata_from_catalog(table)
        centers = np.asarray(meta["centers"], dtype=np.float32).reshape(-1, 2)
        ids = np.asarray(meta["ids"], dtype=np.int64).reshape(-1)
        values = np.stack(
            [
                np.asarray(table["ellipse_major_sigma"], dtype=np.float32),
                np.asarray(table["ellipse_minor_sigma"], dtype=np.float32),
                np.asarray(table["ellipse_theta"], dtype=np.float32),
            ],
            axis=1,
        )
        finite_centers = np.isfinite(centers).all(axis=1)
        valid = (
            finite_centers
            & np.isfinite(values).all(axis=1)
            & (centers[:, 0] >= 0.0)
            & (centers[:, 0] < float(w))
            & (centers[:, 1] >= 0.0)
            & (centers[:, 1] < float(h))
            & (values[:, 0] > 0.0)
            & (values[:, 1] > 0.0)
        )
        if not np.any(valid):
            continue
        centers_parts.append(centers[valid])
        values_parts.append(values[valid])
        classes_parts.append(np.full(int(valid.sum()), int(class_id), dtype=np.uint8))
        ids_parts.append(ids[valid])
    if not centers_parts:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.uint8),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.concatenate(centers_parts, axis=0),
        np.concatenate(values_parts, axis=0),
        np.concatenate(classes_parts, axis=0),
        np.concatenate(ids_parts, axis=0),
    )


def _local_image_stack(images_by_band: dict[str, tuple[np.ndarray, tuple[int, int]]], sample: Sample, bands: Sequence[str]) -> np.ndarray:
    planes = []
    for band in bands:
        image, origin = images_by_band[band]
        tile, _tile_origin = _crop_image_for_tile(image, sample.spec, origin)
        planes.append(tile)
    return np.stack(planes, axis=0).astype(np.float32, copy=False)


def _scaled_image_stack(
    images_by_band: dict[str, tuple[np.ndarray, tuple[int, int]]],
    sample: Sample,
    bands: Sequence[str],
    args: argparse.Namespace,
) -> np.ndarray:
    raw_stack = _local_image_stack(images_by_band, sample, bands)
    mode = str(getattr(args, "image_scaling_mode", "astro-zscore")).strip().lower().replace("_", "-")
    if mode in {"astro-zscore", "legacy", "zscale"}:
        return astro_zscale_preprocess(raw_stack, z_clip=args.z_clip).cpu().numpy().astype(args.image_dtype, copy=False)
    planes = [
        scale_training_image(
            raw_stack[band_idx],
            mode=mode,
            z_clip=tuple(args.z_clip) if args.z_clip is not None else None,
            log_a=float(getattr(args, "image_log_a", 300.0)),
            log_high_percentile=float(getattr(args, "image_log_high_percentile", 99.5)),
            lupton_stretch=float(getattr(args, "image_lupton_stretch", 0.5)),
            lupton_q=float(getattr(args, "image_lupton_q", 20.0)),
            anscombe_scale=float(getattr(args, "image_anscombe_scale", 1000.0)),
        )
        for band_idx in range(raw_stack.shape[0])
    ]
    return np.stack(planes, axis=0).astype(args.image_dtype, copy=False)


def _image_has_rgb_axis(args: argparse.Namespace) -> bool:
    mode = str(getattr(args, "image_scaling_mode", "astro-zscore")).strip().lower().replace("_", "-")
    return mode.endswith("-rgb")


def _target_for_sample(
    sample: Sample,
    *,
    bands: Sequence[str],
    labels_by_band,
    images_by_band,
    backgrounds,
    bright_backgrounds,
    quality_masks,
    args,
    runtime,
) -> dict[str, object]:
    raw_image_stack = _local_image_stack(images_by_band, sample, bands)
    image = _scaled_image_stack(images_by_band, sample, bands, args)
    conf = []
    conf_weight = []
    shape = []
    shape_weight = []
    pu_class = []
    centers_by_band = []
    ids_by_band = []
    shape_source_centers_by_band = []
    shape_source_values_by_band = []
    shape_source_classes_by_band = []
    shape_source_ids_by_band = []
    for band_idx, band in enumerate(bands):
        clean_mask, center_mask, ignore_mask, strict_mask = _crop_sources(
            labels_by_band[band],
            sample.spec,
            args,
            margin=float(args.mask_margin),
        )
        clean_tile, center_tile, _ignore_tile, _strict_tile = _crop_sources(
            labels_by_band[band],
            sample.spec,
            args,
            margin=0.0,
        )
        if sample.dataset_source != "coadd" and bool(args.noncoadd_snr_filter):
            source_mask = None
            if bool(args.noncoadd_snr_use_source_mask):
                source_mask = _source_annulus_exclusion_mask(
                    _vstack_nonempty([clean_mask, center_mask, strict_mask]),
                    tile_size=sample.spec.size,
                    tile_origin=(sample.spec.x0, sample.spec.y0),
                    x_col=args.x_col,
                    y_col=args.y_col,
                    ellipse_sigma=float(args.noncoadd_snr_source_mask_ellipse_sigma),
                )
            hard_exclude_mask = None
            if band in quality_masks:
                quality_mask, quality_origin = quality_masks[band]
                hard_exclude_mask = _crop_full_mask_for_tile(quality_mask, sample.spec, quality_origin)
            normal_tile, snr_center_tile, snr_ignore_tile, _snr = _classify_clean_by_noncoadd_snr(
                clean_tile,
                image=raw_image_stack[band_idx],
                image_origin=(sample.spec.x0, sample.spec.y0),
                args=runtime,
                annulus_exclude_mask=source_mask,
                annulus_hard_exclude_mask=hard_exclude_mask,
            )
            normal_mask, snr_center_mask, snr_ignore_mask, _snr = _classify_clean_by_noncoadd_snr(
                clean_mask,
                image=raw_image_stack[band_idx],
                image_origin=(sample.spec.x0, sample.spec.y0),
                args=runtime,
                annulus_exclude_mask=source_mask,
                annulus_hard_exclude_mask=hard_exclude_mask,
            )
            clean_tile = normal_tile
            center_tile = _vstack_nonempty([center_tile, snr_center_tile])
            clean_mask = normal_mask
            center_mask = _vstack_nonempty([center_mask, snr_center_mask])
            ignore_mask = _vstack_nonempty([ignore_mask, snr_ignore_mask])
        lsst_background = None
        if band in backgrounds:
            background_mask, background_origin = backgrounds[band]
            lsst_background = _crop_full_mask_for_tile(background_mask, sample.spec, background_origin)
        bright_background = None
        if band in bright_backgrounds:
            bright_mask, bright_origin = bright_backgrounds[band]
            bright_background = _crop_full_mask_for_tile(bright_mask, sample.spec, bright_origin)
        target = make_pu_dense_targets(
            clean_mask,
            center_mask,
            ignore_mask,
            sample.spec,
            x_col=args.x_col,
            y_col=args.y_col,
            ellipse_sigma=float(args.ellipse_sigma),
            confidence_levels=int(args.confidence_levels),
            core_radius=int(args.core_radius),
            center_only_weight=float(args.center_only_weight),
            lsst_background_mask=lsst_background,
            bright_background_mask=bright_background,
            strict_center_only_sources=strict_mask,
            strict_center_only_ellipse_sigma=float(args.strict_bright_center_only_ellipse_sigma),
        )
        _restore_center_only_shape_targets(
            target,
            center_mask,
            sample.spec,
            x_col=args.x_col,
            y_col=args.y_col,
            ellipse_sigma=float(args.ellipse_sigma),
            confidence_levels=int(args.confidence_levels),
            core_radius=int(args.core_radius),
        )
        conf.append(np.asarray(target["confidence"], dtype=np.uint8))
        conf_weight.append(np.asarray(target["confidence_weight"], dtype=args.target_float_dtype))
        shape.append(np.asarray(target["shape"], dtype=args.target_float_dtype))
        shape_weight.append(np.asarray(target["shape_weight"], dtype=args.target_float_dtype))
        pu_class.append(np.asarray(target["pu_class_mask"], dtype=np.uint8))
        meta = _metadata_from_catalog(clean_tile)
        centers_by_band.append(meta["centers"])
        ids_by_band.append(meta["ids"])
        shape_centers, shape_values, shape_classes, shape_ids = _shape_source_metadata(
            clean_tile,
            center_tile,
            target,
        )
        shape_source_centers_by_band.append(shape_centers)
        shape_source_values_by_band.append(shape_values)
        shape_source_classes_by_band.append(shape_classes)
        shape_source_ids_by_band.append(shape_ids)
    return {
        "image": image,
        "confidence": np.stack(conf, axis=0),
        "confidence_weight": np.stack(conf_weight, axis=0),
        "shape": np.stack(shape, axis=0),
        "shape_weight": np.stack(shape_weight, axis=0),
        "pu_class": np.stack(pu_class, axis=0),
        "centers_by_band": centers_by_band,
        "ids_by_band": ids_by_band,
        "shape_source_centers_by_band": shape_source_centers_by_band,
        "shape_source_values_by_band": shape_source_values_by_band,
        "shape_source_classes_by_band": shape_source_classes_by_band,
        "shape_source_ids_by_band": shape_source_ids_by_band,
    }


def _copy_target_package(pkg: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in pkg.items():
        if isinstance(value, np.ndarray):
            out[key] = value.copy()
        elif isinstance(value, list):
            out[key] = [np.asarray(item).copy() for item in value]
        else:
            out[key] = value
    return out


def _source_rows_by_key(pkg: dict[str, object], band_idx: int) -> dict[tuple[str, object], tuple[np.ndarray, np.ndarray, int, int]]:
    centers = np.asarray(pkg["shape_source_centers_by_band"][band_idx], dtype=np.float32).reshape(-1, 2)
    values = np.asarray(pkg["shape_source_values_by_band"][band_idx], dtype=np.float32).reshape(-1, 3)
    classes = np.asarray(pkg["shape_source_classes_by_band"][band_idx], dtype=np.uint8).reshape(-1)
    ids = np.asarray(pkg["shape_source_ids_by_band"][band_idx], dtype=np.int64).reshape(-1)
    rows: dict[tuple[str, object], tuple[np.ndarray, np.ndarray, int, int]] = {}
    for idx, (center, value, class_id, source_id) in enumerate(zip(centers, values, classes, ids)):
        if int(source_id) >= 0:
            key = ("id", int(source_id))
        else:
            rounded = tuple(np.round(center.astype(np.float64), 3).tolist())
            key = ("xy", rounded)
        rows.setdefault(key, (center, value, int(class_id), int(source_id)))
    return rows


def _select_source_rows(
    *,
    primary_rows: dict[tuple[str, object], tuple[np.ndarray, np.ndarray, int, int]],
    secondary_rows: dict[tuple[str, object], tuple[np.ndarray, np.ndarray, int, int]],
    keys: Sequence[tuple[str, object]],
    class_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = []
    values = []
    classes = []
    ids = []
    for key in keys:
        row = primary_rows.get(key) or secondary_rows.get(key)
        if row is None:
            continue
        center, value, _old_class, source_id = row
        centers.append(center)
        values.append(value)
        classes.append(int(class_id))
        ids.append(int(source_id))
    if not centers:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.uint8),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.asarray(centers, dtype=np.float32).reshape(-1, 2),
        np.asarray(values, dtype=np.float32).reshape(-1, 3),
        np.asarray(classes, dtype=np.uint8),
        np.asarray(ids, dtype=np.int64),
    )


def _align_denoised_noisy_snr_packages(
    denoised_pkg: dict[str, object],
    noisy_pkg: dict[str, object],
    *,
    primary: str,
) -> dict[str, object]:
    if primary not in {"denoised", "noisy"}:
        raise ValueError(f"primary must be denoised or noisy, got {primary!r}")
    primary_pkg = denoised_pkg if primary == "denoised" else noisy_pkg
    secondary_pkg = noisy_pkg if primary == "denoised" else denoised_pkg
    out = _copy_target_package(primary_pkg)

    primary_pu = np.asarray(primary_pkg["pu_class"], dtype=np.uint8)
    secondary_pu = np.asarray(secondary_pkg["pu_class"], dtype=np.uint8)
    if primary_pu.shape != secondary_pu.shape:
        raise ValueError(f"cannot align PU masks with shapes {primary_pu.shape} and {secondary_pu.shape}")

    out_pu = np.asarray(out["pu_class"], dtype=np.uint8)
    out_conf = np.asarray(out["confidence"], dtype=np.uint8)
    out_conf_weight = np.asarray(out["confidence_weight"])
    out_shape = np.asarray(out["shape"])
    out_shape_weight = np.asarray(out["shape_weight"])
    primary_conf = np.asarray(primary_pkg["confidence"], dtype=np.uint8)
    secondary_conf = np.asarray(secondary_pkg["confidence"], dtype=np.uint8)
    primary_shape = np.asarray(primary_pkg["shape"])
    secondary_shape = np.asarray(secondary_pkg["shape"])
    primary_shape_weight = np.asarray(primary_pkg["shape_weight"])
    secondary_shape_weight = np.asarray(secondary_pkg["shape_weight"])

    centers_by_band = []
    ids_by_band = []
    shape_centers_by_band = []
    shape_values_by_band = []
    shape_classes_by_band = []
    shape_ids_by_band = []

    for band_idx in range(primary_pu.shape[0]):
        den_pu = primary_pu[band_idx] if primary == "denoised" else secondary_pu[band_idx]
        noi_pu = secondary_pu[band_idx] if primary == "denoised" else primary_pu[band_idx]
        clean = (den_pu == 1) & (noi_pu == 1)
        center = ((den_pu == 2) | (den_pu == 5) | (noi_pu == 2) | (noi_pu == 5)) & ~clean
        bright = ((den_pu == 6) | (noi_pu == 6)) & ~clean & ~center
        background = (den_pu == 4) & (noi_pu == 4) & ~clean & ~center & ~bright
        merged = np.full(den_pu.shape, 3, dtype=np.uint8)
        merged[background] = 4
        merged[bright] = 6
        merged[center] = 2
        merged[clean] = 1
        out_pu[band_idx] = merged

        supervised = clean | background
        out_conf_weight[band_idx] = supervised.astype(out_conf_weight.dtype, copy=False)
        out_conf[band_idx] = np.maximum(primary_conf[band_idx], secondary_conf[band_idx])

        take_secondary = secondary_shape_weight[band_idx] > primary_shape_weight[band_idx]
        out_shape[band_idx] = primary_shape[band_idx]
        if np.any(take_secondary):
            out_shape[band_idx, :, take_secondary] = secondary_shape[band_idx, :, take_secondary]
        merged_shape_weight = np.maximum(primary_shape_weight[band_idx], secondary_shape_weight[band_idx])
        merged_shape_weight = np.where(clean | center, merged_shape_weight, 0).astype(out_shape_weight.dtype, copy=False)
        out_shape_weight[band_idx] = merged_shape_weight

        den_rows = _source_rows_by_key(denoised_pkg, band_idx)
        noi_rows = _source_rows_by_key(noisy_pkg, band_idx)
        den_clean = {key for key, row in den_rows.items() if row[2] == 1}
        noi_clean = {key for key, row in noi_rows.items() if row[2] == 1}
        clean_keys = sorted(den_clean & noi_clean, key=lambda item: (item[0], item[1]))
        den_center = {key for key, row in den_rows.items() if row[2] in (2, 5)}
        noi_center = {key for key, row in noi_rows.items() if row[2] in (2, 5)}
        center_keys = sorted((den_center | noi_center) - set(clean_keys), key=lambda item: (item[0], item[1]))

        primary_rows = den_rows if primary == "denoised" else noi_rows
        secondary_rows = noi_rows if primary == "denoised" else den_rows
        clean_centers, clean_values, clean_classes, clean_ids = _select_source_rows(
            primary_rows=primary_rows,
            secondary_rows=secondary_rows,
            keys=clean_keys,
            class_id=1,
        )
        center_centers, center_values, center_classes, center_ids = _select_source_rows(
            primary_rows=primary_rows,
            secondary_rows=secondary_rows,
            keys=center_keys,
            class_id=2,
        )
        centers_by_band.append(clean_centers)
        ids_by_band.append(clean_ids)
        shape_centers_by_band.append(np.concatenate([clean_centers, center_centers], axis=0))
        shape_values_by_band.append(np.concatenate([clean_values, center_values], axis=0))
        shape_classes_by_band.append(np.concatenate([clean_classes, center_classes], axis=0))
        shape_ids_by_band.append(np.concatenate([clean_ids, center_ids], axis=0))

    out["pu_class"] = out_pu
    out["confidence"] = out_conf
    out["confidence_weight"] = out_conf_weight
    out["shape"] = out_shape
    out["shape_weight"] = out_shape_weight
    out["centers_by_band"] = centers_by_band
    out["ids_by_band"] = ids_by_band
    out["shape_source_centers_by_band"] = shape_centers_by_band
    out["shape_source_values_by_band"] = shape_values_by_band
    out["shape_source_classes_by_band"] = shape_classes_by_band
    out["shape_source_ids_by_band"] = shape_ids_by_band
    return out


def _write_dataset_zarr(
    *,
    output: Path,
    samples: Sequence[Sample],
    bands: Sequence[str],
    labels_by_band,
    image_sources,
    background_sources,
    bright_background_sources,
    quality_masks,
    args,
    runtime,
    attrs: dict,
    package_loader=None,
) -> dict:
    manifest_path = output.parent / f"{output.name}_manifest.json"
    staging = output.with_name(f".{output.name}.inprogress")
    if manifest_path.exists() and output.exists() and not bool(args.overwrite):
        print(f"[direct-zarr] skip complete existing store: {output}", flush=True)
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if output.exists() and not manifest_path.exists() and not bool(args.overwrite):
        print(f"[direct-zarr] removing incomplete existing store before rebuild: {output}", flush=True)
        shutil.rmtree(output)
    if staging.exists():
        print(f"[direct-zarr] removing stale staging store: {staging}", flush=True)
        shutil.rmtree(staging)
    n = len(samples)
    b = len(bands)
    h = w = int(args.tile_size)
    chunk_tiles = max(1, int(args.chunk_tiles))
    rgb_axis = _image_has_rgb_axis(args)
    # Write under a non-*.zarr staging name so concurrent training discovery
    # cannot observe arrays before the completion manifest is ready.
    writer = ZarrGroupWriter(staging, overwrite=True, attrs=attrs)
    if rgb_axis:
        images = writer.array("images", shape=(n, b, 3, h, w), chunks=(chunk_tiles, b, 3, h, w), dtype=args.image_dtype)
    else:
        images = writer.array("images", shape=(n, b, h, w), chunks=(chunk_tiles, b, h, w), dtype=args.image_dtype)
    confidence = writer.array("band_confidence", shape=(n, b, h, w), chunks=(chunk_tiles, b, h, w), dtype=np.uint8)
    conf_weight = writer.array("band_conf_weight", shape=(n, b, h, w), chunks=(chunk_tiles, b, h, w), dtype=args.target_float_dtype)
    shape = writer.array("band_shape", shape=(n, b, 3, h, w), chunks=(chunk_tiles, b, 3, h, w), dtype=args.target_float_dtype)
    shape_weight = writer.array("band_shape_weight", shape=(n, b, h, w), chunks=(chunk_tiles, b, h, w), dtype=args.target_float_dtype)
    pu_class = writer.array("band_pu_class_mask", shape=(n, b, h, w), chunks=(chunk_tiles, b, h, w), dtype=np.uint8)
    centers_flat = []
    ids_flat = []
    offsets = np.zeros((n, b + 1), dtype=np.int64)
    shape_centers_flat = []
    shape_values_flat = []
    shape_classes_flat = []
    shape_ids_flat = []
    shape_offsets = np.zeros((n, b + 1), dtype=np.int64)

    def load_one(sample: Sample):
        if package_loader is not None:
            return package_loader(sample)
        images_by_band = image_sources[sample.image_key]
        backgrounds = background_sources.get(sample.image_key, {})
        bright_backgrounds = bright_background_sources.get(sample.image_key, {})
        return _target_for_sample(
            sample,
            bands=bands,
            labels_by_band=labels_by_band,
            images_by_band=images_by_band,
            backgrounds=backgrounds,
            bright_backgrounds=bright_backgrounds,
            quality_masks=quality_masks,
            args=args,
            runtime=runtime,
        )

    for start in range(0, n, chunk_tiles):
        end = min(n, start + chunk_tiles)
        batch = list(samples[start:end])
        with ThreadPoolExecutor(max_workers=max(1, int(args.tile_workers))) as executor:
            packages = list(executor.map(load_one, batch))
        chunk_index = (start // chunk_tiles, 0, 0, 0)
        image_chunk_index = (start // chunk_tiles, 0, 0, 0, 0) if rgb_axis else chunk_index
        images.write_chunk(image_chunk_index, np.stack([pkg["image"] for pkg in packages], axis=0))
        confidence.write_chunk(chunk_index, np.stack([pkg["confidence"] for pkg in packages], axis=0))
        conf_weight.write_chunk(chunk_index, np.stack([pkg["confidence_weight"] for pkg in packages], axis=0))
        shape.write_chunk((start // chunk_tiles, 0, 0, 0, 0), np.stack([pkg["shape"] for pkg in packages], axis=0))
        shape_weight.write_chunk(chunk_index, np.stack([pkg["shape_weight"] for pkg in packages], axis=0))
        pu_class.write_chunk(chunk_index, np.stack([pkg["pu_class"] for pkg in packages], axis=0))
        for local_idx, pkg in enumerate(packages):
            sample_idx = start + local_idx
            offsets[sample_idx, 0] = len(centers_flat)
            shape_offsets[sample_idx, 0] = len(shape_centers_flat)
            for band_idx in range(b):
                centers = np.asarray(pkg["centers_by_band"][band_idx], dtype=np.float32).reshape(-1, 2)
                ids = np.asarray(pkg["ids_by_band"][band_idx], dtype=np.int64).reshape(-1)
                if len(centers):
                    centers_flat.extend(centers)
                    ids_flat.extend(ids)
                offsets[sample_idx, band_idx + 1] = len(centers_flat)
                source_centers = np.asarray(pkg["shape_source_centers_by_band"][band_idx], dtype=np.float32).reshape(-1, 2)
                source_values = np.asarray(pkg["shape_source_values_by_band"][band_idx], dtype=np.float32).reshape(-1, 3)
                source_classes = np.asarray(pkg["shape_source_classes_by_band"][band_idx], dtype=np.uint8).reshape(-1)
                source_ids = np.asarray(pkg["shape_source_ids_by_band"][band_idx], dtype=np.int64).reshape(-1)
                if len(source_centers):
                    shape_centers_flat.extend(source_centers)
                    shape_values_flat.extend(source_values)
                    shape_classes_flat.extend(source_classes)
                    shape_ids_flat.extend(source_ids)
                shape_offsets[sample_idx, band_idx + 1] = len(shape_centers_flat)
        print(f"[direct-zarr] {output.name}: wrote samples {start + 1}-{end}/{n}", flush=True)

    centers_arr = np.asarray(centers_flat, dtype=np.float32).reshape(-1, 2)
    ids_arr = np.asarray(ids_flat, dtype=np.int64).reshape(-1)
    writer.array("source_centers", shape=centers_arr.shape, chunks=(max(1, len(centers_arr)), 2), dtype=np.float32).write_full(centers_arr)
    writer.array("source_ids", shape=ids_arr.shape, chunks=(max(1, len(ids_arr)),), dtype=np.int64).write_full(ids_arr)
    writer.array("source_offsets", shape=offsets.shape, chunks=(max(1, n), b + 1), dtype=np.int64).write_full(offsets)
    shape_centers_arr = np.asarray(shape_centers_flat, dtype=np.float32).reshape(-1, 2)
    shape_values_arr = np.asarray(shape_values_flat, dtype=np.float32).reshape(-1, 3)
    shape_classes_arr = np.asarray(shape_classes_flat, dtype=np.uint8).reshape(-1)
    shape_ids_arr = np.asarray(shape_ids_flat, dtype=np.int64).reshape(-1)
    writer.array(
        "shape_source_centers",
        shape=shape_centers_arr.shape,
        chunks=(max(1, len(shape_centers_arr)), 2),
        dtype=np.float32,
    ).write_full(shape_centers_arr)
    writer.array(
        "shape_source_values",
        shape=shape_values_arr.shape,
        chunks=(max(1, len(shape_values_arr)), 3),
        dtype=np.float32,
    ).write_full(shape_values_arr)
    writer.array(
        "shape_source_classes",
        shape=shape_classes_arr.shape,
        chunks=(max(1, len(shape_classes_arr)),),
        dtype=np.uint8,
    ).write_full(shape_classes_arr)
    writer.array(
        "shape_source_ids",
        shape=shape_ids_arr.shape,
        chunks=(max(1, len(shape_ids_arr)),),
        dtype=np.int64,
    ).write_full(shape_ids_arr)
    writer.array(
        "shape_source_offsets",
        shape=shape_offsets.shape,
        chunks=(max(1, n), b + 1),
        dtype=np.int64,
    ).write_full(shape_offsets)
    writer.array("tile_x0", shape=(n,), chunks=(max(1, n),), dtype=np.int32).write_full(np.asarray([s.spec.x0 for s in samples], dtype=np.int32))
    writer.array("tile_y0", shape=(n,), chunks=(max(1, n),), dtype=np.int32).write_full(np.asarray([s.spec.y0 for s in samples], dtype=np.int32))
    writer.array("tile_name", shape=(n, 192), chunks=(max(1, n), 192), dtype=np.uint8).write_full(encode_fixed_utf8([s.name for s in samples], 192))
    writer.array("group", shape=(n, 32), chunks=(max(1, n), 32), dtype=np.uint8).write_full(encode_fixed_utf8([s.group for s in samples], 32))
    writer.array("dataset_source", shape=(n, 32), chunks=(max(1, n), 32), dtype=np.uint8).write_full(encode_fixed_utf8([s.dataset_source for s in samples], 32))
    manifest = {
        "output": str(output),
        "num_samples": n,
        "total_sources": int(len(ids_arr)),
        "bands": list(bands),
        "samples": [{"name": s.name, "x0": s.spec.x0, "y0": s.spec.y0, "group": s.group, "dataset_source": s.dataset_source} for s in samples],
    }
    backup = output.with_name(f".{output.name}.replaced")
    if backup.exists():
        shutil.rmtree(backup)
    if output.exists():
        output.rename(backup)
    try:
        staging.rename(output)
        write_json(manifest_path, manifest)
    except Exception:
        if not output.exists() and backup.exists():
            backup.rename(output)
        raise
    if backup.exists():
        shutil.rmtree(backup)
    return manifest


def _write_aligned_denoised_noisy_zarrs(
    *,
    output_root: Path,
    patch: str,
    specs: Sequence[TileSpec],
    labels,
    variant_images_by_name,
    variant_backgrounds_by_name,
    variant_bright_backgrounds_by_name,
    quality_masks,
    args,
    runtime,
) -> list[dict]:
    missing = [variant for variant in ("denoised", "noisy") if variant not in variant_images_by_name]
    if missing:
        raise FileNotFoundError(f"cannot align denoised/noisy labels; missing variants: {', '.join(missing)}")
    common_groups = sorted(set(variant_images_by_name["denoised"]) & set(variant_images_by_name["noisy"]))
    if not common_groups:
        raise FileNotFoundError("cannot align denoised/noisy labels; no common image groups")
    common_bands = [
        band
        for band in args.bands
        if band in labels
        and all(
            band in variant_images_by_name[pair_variant].get(group, {})
            for pair_variant in ("denoised", "noisy")
            for group in common_groups
        )
    ]
    if not common_bands:
        raise FileNotFoundError("cannot align denoised/noisy labels; no common bands across common image groups")
    summaries = []
    for variant in ("denoised", "noisy"):
        samples = _make_samples(variant, specs, common_groups)

        def load_aligned(sample: Sample, *, primary_variant=variant):
            paired = {}
            for pair_variant in ("denoised", "noisy"):
                pair_sample = Sample(
                    name=f"{sample.image_key}_{sample.spec.name}",
                    spec=sample.spec,
                    dataset_source=pair_variant,
                    group=sample.image_key,
                    image_key=sample.image_key,
                )
                paired[pair_variant] = _target_for_sample(
                    pair_sample,
                    bands=common_bands,
                    labels_by_band=labels,
                    images_by_band=variant_images_by_name[pair_variant][sample.image_key],
                    backgrounds=variant_backgrounds_by_name[pair_variant].get(sample.image_key, {}),
                    bright_backgrounds=variant_bright_backgrounds_by_name[pair_variant].get(sample.image_key, {}),
                    quality_masks=quality_masks,
                    args=args,
                    runtime=runtime,
                )
            return _align_denoised_noisy_snr_packages(
                paired["denoised"],
                paired["noisy"],
                primary=primary_variant,
            )

        out = output_root / variant / f"{patch}.zarr"
        summaries.append(
            _write_dataset_zarr(
                output=out,
                samples=samples,
                bands=common_bands,
                labels_by_band=labels,
                image_sources=variant_images_by_name[variant],
                background_sources=variant_backgrounds_by_name[variant],
                bright_background_sources=variant_bright_backgrounds_by_name[variant],
                quality_masks=quality_masks,
                args=args,
                runtime=runtime,
                attrs={**_attrs(args, runtime, patch, variant, len(samples)), "bands": list(common_bands), "source_bands_requested": list(args.bands)},
                package_loader=load_aligned,
            )
        )
    return summaries


def _single_band_sources(
    sources_by_key: dict[str, dict[str, tuple[np.ndarray, tuple[int, int]]]],
    key: str,
    band: str,
) -> dict[str, dict[str, tuple[np.ndarray, tuple[int, int]]]]:
    value = sources_by_key.get(key, {}).get(band)
    return {key: {band: value}} if value is not None else {key: {}}


def _write_aligned_denoised_noisy_image_level_zarrs(
    *,
    output_root: Path,
    patch: str,
    specs: Sequence[TileSpec],
    labels,
    variant_images_by_name,
    variant_backgrounds_by_name,
    variant_bright_backgrounds_by_name,
    quality,
    quality_masks,
    args,
    runtime,
) -> list[dict]:
    missing = [variant for variant in ("denoised", "noisy") if variant not in variant_images_by_name]
    if missing:
        raise FileNotFoundError(f"cannot align denoised/noisy image-level labels; missing variants: {', '.join(missing)}")
    common_groups = sorted(set(variant_images_by_name["denoised"]) & set(variant_images_by_name["noisy"]))
    if not common_groups:
        raise FileNotFoundError("cannot align denoised/noisy image-level labels; no common image groups")
    summaries = []
    for variant in ("denoised", "noisy"):
        for image_key in common_groups:
            den_images = variant_images_by_name["denoised"].get(image_key, {})
            noi_images = variant_images_by_name["noisy"].get(image_key, {})
            for band in args.bands:
                if band not in labels or band not in den_images or band not in noi_images:
                    continue
                band_specs = [spec for spec in specs if _quality_allows(quality, band, spec)]
                if not band_specs:
                    q = quality.get(band, {}) if quality else {}
                    print(
                        f"[direct-zarr] skip aligned image-level {variant}/{patch}/{image_key}/{band}: "
                        f"0 tiles passed quality filter (patch_score={q.get('patch_score', 'n/a')})",
                        flush=True,
                    )
                    continue
                samples = _make_image_level_samples(variant, band_specs, image_key, band)

                def load_aligned(sample: Sample, *, primary_variant=variant, sample_band=band):
                    paired = {}
                    for pair_variant in ("denoised", "noisy"):
                        pair_sample = Sample(
                            name=f"{sample.image_key}_{sample_band}_{sample.spec.name}",
                            spec=sample.spec,
                            dataset_source=pair_variant,
                            group=sample.image_key,
                            image_key=sample.image_key,
                            band=sample_band,
                        )
                        paired[pair_variant] = _target_for_sample(
                            pair_sample,
                            bands=[sample_band],
                            labels_by_band=labels,
                            images_by_band={sample_band: variant_images_by_name[pair_variant][sample.image_key][sample_band]},
                            backgrounds=variant_backgrounds_by_name[pair_variant].get(sample.image_key, {}),
                            bright_backgrounds=variant_bright_backgrounds_by_name[pair_variant].get(sample.image_key, {}),
                            quality_masks=quality_masks,
                            args=args,
                            runtime=runtime,
                        )
                    return _align_denoised_noisy_snr_packages(
                        paired["denoised"],
                        paired["noisy"],
                        primary=primary_variant,
                    )

                suffix = f"{patch}__{image_key}"
                out = output_root / "image_level" / variant / band / f"{suffix}.zarr"
                attrs = _attrs(args, runtime, patch, variant, len(samples))
                attrs.update(
                    {
                        "image_level_training": True,
                        "image_level_band": str(band),
                        "bands": [str(band)],
                        "source_bands_requested": list(args.bands),
                        "quality_filter": bool(getattr(args, "quality_filter", False)),
                        "quality_bad_score_threshold": float(getattr(args, "quality_bad_score_threshold", 0.13)),
                        "quality_patch_score": (
                            float(quality.get(band, {}).get("patch_score", float("nan"))) if quality else None
                        ),
                        "quality_num_tiles_kept": int(len(band_specs)),
                        "quality_num_tiles_total": int(len(specs)),
                    }
                )
                summaries.append(
                    _write_dataset_zarr(
                        output=out,
                        samples=samples,
                        bands=[band],
                        labels_by_band=labels,
                        image_sources={image_key: {band: variant_images_by_name[variant][image_key][band]}},
                        background_sources=_single_band_sources(variant_backgrounds_by_name[variant], image_key, band),
                        bright_background_sources=_single_band_sources(
                            variant_bright_backgrounds_by_name[variant],
                            image_key,
                            band,
                        ),
                        quality_masks=quality_masks,
                        args=args,
                        runtime=runtime,
                        attrs=attrs,
                        package_loader=load_aligned,
                    )
                )
    return summaries


def _write_image_level_zarrs(
    *,
    output_root: Path,
    patch: str,
    specs: Sequence[TileSpec],
    dataset: str,
    labels,
    image_sources,
    background_sources,
    bright_background_sources,
    quality,
    quality_masks,
    args,
    runtime,
) -> list[dict]:
    summaries = []
    for image_key, images_by_band in sorted(image_sources.items()):
        for band in args.bands:
            if band not in images_by_band or band not in labels:
                continue
            band_specs = [spec for spec in specs if _quality_allows(quality, band, spec)]
            if not band_specs:
                q = quality.get(band, {}) if quality else {}
                print(
                    f"[direct-zarr] skip image-level {dataset}/{patch}/{image_key}/{band}: "
                    f"0 tiles passed quality filter (patch_score={q.get('patch_score', 'n/a')})",
                    flush=True,
                )
                continue
            samples = _make_image_level_samples(dataset, band_specs, image_key, band)
            suffix = patch if dataset == "coadd" else f"{patch}__{image_key}"
            out = output_root / "image_level" / dataset / band / f"{suffix}.zarr"
            attrs = _attrs(args, runtime, patch, dataset, len(samples))
            attrs.update(
                {
                    "image_level_training": True,
                    "image_level_band": str(band),
                    "bands": [str(band)],
                    "source_bands_requested": list(args.bands),
                    "quality_filter": bool(getattr(args, "quality_filter", False)),
                    "quality_bad_score_threshold": float(getattr(args, "quality_bad_score_threshold", 0.13)),
                    "quality_patch_score": (
                        float(quality.get(band, {}).get("patch_score", float("nan"))) if quality else None
                    ),
                    "quality_num_tiles_kept": int(len(band_specs)),
                    "quality_num_tiles_total": int(len(specs)),
                }
            )
            summaries.append(
                _write_dataset_zarr(
                    output=out,
                    samples=samples,
                    bands=[band],
                    labels_by_band=labels,
                    image_sources={image_key: {band: images_by_band[band]}},
                    background_sources=_single_band_sources(background_sources, image_key, band),
                    bright_background_sources=_single_band_sources(bright_background_sources, image_key, band),
                    quality_masks=quality_masks,
                    args=args,
                    runtime=runtime,
                    attrs=attrs,
                )
            )
    return summaries


def preprocess_patch(args: argparse.Namespace, patch: str) -> dict:
    output_root = Path(args.output_root).expanduser() / str(args.tract)
    expected_manifests = []
    if args.include_coadd and not bool(args.image_level_only):
        expected_manifests.append(output_root / "coadd" / f"{patch}.zarr_manifest.json")
    variant_patch_dir = None
    if args.denoised_fits_root is not None and not bool(args.image_level_only):
        variant_patch_dir = _find_denoised_patch_dir(Path(args.denoised_fits_root).expanduser(), patch)
        if variant_patch_dir is not None:
            for variant in args.image_variants:
                expected_manifests.append(output_root / variant / f"{patch}.zarr_manifest.json")
    if (
        expected_manifests
        and not bool(args.write_image_level_zarr)
        and not bool(args.overwrite)
        and all(path.exists() for path in expected_manifests)
    ):
        print(f"[direct-zarr] skip complete patch {patch}", flush=True)
        return {
            "patch": patch,
            "summaries": [json.loads(path.read_text(encoding="utf-8")) for path in expected_manifests],
            "skipped_existing": True,
        }

    runtime = build_pu_runtime_config(args)
    shape_yx, parent_origin = _read_patch_image_meta(Path(args.coadd_root).expanduser(), args.bands[0], int(args.tract), patch)
    args.parent_origin_xy = parent_origin
    specs = make_tile_specs(
        parent_origin=parent_origin,
        image_shape=(int(shape_yx[1]), int(shape_yx[0])),
        tile_size=int(args.tile_size),
        stride=int(args.stride),
        compare_origin=tuple(args.compare_origin) if args.compare_origin else None,
    )
    if args.tile_filter:
        wanted = set(args.tile_filter)
        specs = [spec for spec in specs if spec.name in wanted]
        if not specs:
            raise RuntimeError(f"No tile matched --tile-filter {sorted(wanted)} for patch {patch}")
    if args.max_tiles is not None:
        specs = specs[: int(args.max_tiles)]
    labels = _classify_all_bands(args, runtime, patch)
    backgrounds = _read_backgrounds(args, patch, shape_yx, parent_origin)
    quality_masks = _read_quality_masks(args, patch)
    quality = _quality_allowed_tiles(args, patch, specs)
    summaries = []
    coadd_images_for_bright = None
    coadd_bright_backgrounds = {"coadd": {}}

    if args.include_coadd:
        coadd_images = {"coadd": _read_coadd_images(args, patch)}
        coadd_images_for_bright = coadd_images
        coadd_bands = [band for band in args.bands if band in labels and band in coadd_images["coadd"]]
        if not coadd_bands:
            raise FileNotFoundError(f"No usable coadd bands found for patch {patch}")
        coadd_bright_backgrounds = _bright_backgrounds_for_images(args, coadd_images, patch=patch)
        if bool(args.write_image_level_zarr):
            summaries.extend(
                _write_image_level_zarrs(
                    output_root=output_root,
                    patch=patch,
                    specs=specs,
                    dataset="coadd",
                    labels=labels,
                    image_sources=coadd_images,
                    background_sources={"coadd": backgrounds},
                    bright_background_sources=coadd_bright_backgrounds,
                    quality=quality,
                    quality_masks=quality_masks,
                    args=args,
                    runtime=runtime,
                )
            )
        if not bool(args.image_level_only):
            samples = _make_samples("coadd", specs, ["coadd"])
            out = output_root / "coadd" / f"{patch}.zarr"
            summaries.append(
                _write_dataset_zarr(
                    output=out,
                    samples=samples,
                    bands=coadd_bands,
                    labels_by_band=labels,
                    image_sources=coadd_images,
                    background_sources={"coadd": backgrounds},
                    bright_background_sources=coadd_bright_backgrounds,
                    quality_masks=quality_masks,
                    args=args,
                    runtime=runtime,
                    attrs={
                        **_attrs(args, runtime, patch, "coadd", len(samples)),
                        "bands": list(coadd_bands),
                        "source_bands_requested": list(args.bands),
                    },
                )
            )

    if args.denoised_fits_root is not None:
        if coadd_images_for_bright is None and (
            bool(getattr(args, "pu_enable_bright_background_mask", False))
            or getattr(args, "external_bright_label_root", None) is not None
        ):
            coadd_images_for_bright = {"coadd": _read_coadd_images(args, patch)}
            coadd_bright_backgrounds = _bright_backgrounds_for_images(args, coadd_images_for_bright, patch=patch)
        if bool(args.align_denoised_noisy_snr_labels) and {"denoised", "noisy"}.issubset(set(args.image_variants)):
            variant_images_by_name = {}
            variant_backgrounds_by_name = {}
            variant_bright_backgrounds_by_name = {}
            for variant in args.image_variants:
                try:
                    variant_images = _read_variant_images(args, patch, variant)
                except FileNotFoundError as exc:
                    if args.missing_variant_policy == "error":
                        raise
                    print(f"[direct-zarr] skip {variant} patch {patch}: {exc}", flush=True)
                    continue
                variant_images_by_name[variant] = variant_images
                variant_backgrounds_by_name[variant] = _variant_backgrounds_for_images(
                    args,
                    variant=variant,
                    patch=patch,
                    variant_images=variant_images,
                    coadd_backgrounds=backgrounds,
                )
                variant_bright_backgrounds_by_name[variant] = _replicate_coadd_bright_backgrounds(
                    coadd_bright_backgrounds,
                    variant_images,
                )
            missing_aligned_variants = [
                variant for variant in ("denoised", "noisy") if variant not in variant_images_by_name
            ]
            if missing_aligned_variants:
                print(
                    f"[direct-zarr] skip aligned denoised/noisy outputs for patch {patch}: "
                    f"missing variants {', '.join(missing_aligned_variants)}",
                    flush=True,
                )
                return {
                    "patch": patch,
                    "summaries": summaries,
                    "skipped_missing_aligned_variants": missing_aligned_variants,
                }
            if bool(args.write_image_level_zarr):
                summaries.extend(
                    _write_aligned_denoised_noisy_image_level_zarrs(
                        output_root=output_root,
                        patch=patch,
                        specs=specs,
                        labels=labels,
                        variant_images_by_name=variant_images_by_name,
                        variant_backgrounds_by_name=variant_backgrounds_by_name,
                        variant_bright_backgrounds_by_name=variant_bright_backgrounds_by_name,
                        quality=quality,
                        quality_masks=quality_masks,
                        args=args,
                        runtime=runtime,
                    )
                )
            if not bool(args.image_level_only):
                summaries.extend(
                    _write_aligned_denoised_noisy_zarrs(
                        output_root=output_root,
                        patch=patch,
                        specs=specs,
                        labels=labels,
                        variant_images_by_name=variant_images_by_name,
                        variant_backgrounds_by_name=variant_backgrounds_by_name,
                        variant_bright_backgrounds_by_name=variant_bright_backgrounds_by_name,
                        quality_masks=quality_masks,
                        args=args,
                        runtime=runtime,
                    )
                )
            return {"patch": patch, "summaries": summaries}
        for variant in args.image_variants:
            try:
                variant_images = _read_variant_images(args, patch, variant)
            except FileNotFoundError as exc:
                if args.missing_variant_policy == "error":
                    raise
                print(f"[direct-zarr] skip {variant} patch {patch}: {exc}", flush=True)
                continue
            variant_backgrounds = _variant_backgrounds_for_images(
                args,
                variant=variant,
                patch=patch,
                variant_images=variant_images,
                coadd_backgrounds=backgrounds,
            )
            variant_bright_backgrounds = _replicate_coadd_bright_backgrounds(coadd_bright_backgrounds, variant_images)
            if bool(args.write_image_level_zarr):
                summaries.extend(
                    _write_image_level_zarrs(
                        output_root=output_root,
                        patch=patch,
                        specs=specs,
                        dataset=variant,
                        labels=labels,
                        image_sources=variant_images,
                        background_sources=variant_backgrounds,
                        bright_background_sources=variant_bright_backgrounds,
                        quality=quality,
                        quality_masks=quality_masks,
                        args=args,
                        runtime=runtime,
                    )
                )
            if not bool(args.image_level_only):
                variant_bands = [
                    band
                    for band in args.bands
                    if band in labels and all(band in images_by_band for images_by_band in variant_images.values())
                ]
                if not variant_bands:
                    print(f"[direct-zarr] skip {variant} patch-level {patch}: no common usable bands", flush=True)
                    continue
                samples = _make_samples(variant, specs, sorted(variant_images))
                out = output_root / variant / f"{patch}.zarr"
                summaries.append(
                    _write_dataset_zarr(
                        output=out,
                        samples=samples,
                        bands=variant_bands,
                        labels_by_band=labels,
                        image_sources=variant_images,
                        background_sources=variant_backgrounds,
                        bright_background_sources=variant_bright_backgrounds,
                        quality_masks=quality_masks,
                        args=args,
                        runtime=runtime,
                        attrs={
                            **_attrs(args, runtime, patch, variant, len(samples)),
                            "bands": list(variant_bands),
                            "source_bands_requested": list(args.bands),
                        },
                    )
                )
    return {"patch": patch, "summaries": summaries}


def _attrs(
    args: argparse.Namespace,
    runtime: argparse.Namespace,
    patch: str,
    dataset_source: str,
    n: int,
) -> dict:
    return {
        "format": "cellect_direct_patch_zarr",
        "format_version": 2,
        "tract": str(args.tract),
        "patch": patch,
        "dataset_source": dataset_source,
        "bands": list(args.bands),
        "num_samples": int(n),
        "tile_size": int(args.tile_size),
        "filter_contract": "PU broad fixed B magnitude range; no band-limit B filter switch",
        "b_mag_min": float(args.b_mag_min),
        "b_mag_max": float(args.b_mag_max),
        "image_variant_background_source": str(args.image_variant_background_source),
        "variant_lsst_background_root": str(args.variant_lsst_background_root) if args.variant_lsst_background_root is not None else None,
        "center_only_shape_restored": True,
        "noncoadd_snr_use_source_mask": bool(args.noncoadd_snr_use_source_mask),
        "noncoadd_snr_use_quality_mask": bool(args.noncoadd_snr_use_quality_mask),
        "noncoadd_snr_mask_planes": list(args.noncoadd_snr_mask_planes),
        "noncoadd_snr_exclude_self_source": bool(args.noncoadd_snr_exclude_self_source),
        "align_denoised_noisy_snr_labels": bool(args.align_denoised_noisy_snr_labels),
        "pu_enable_bright_background_mask": bool(args.pu_enable_bright_background_mask),
        "pu_bright_mask_mode": str(args.pu_bright_mask_mode),
        "external_bright_label_root": str(args.external_bright_label_root) if args.external_bright_label_root is not None else None,
        "image_scaling_mode": str(args.image_scaling_mode),
        "image_rgb_channels": 3 if _image_has_rgb_axis(args) else 1,
        "aligned_variant_label_contract": (
            "clean=denoised_clean AND noisy_clean; center_only=denoised/noisy center-only union; "
            "bright=denoised/noisy bright union after clean/center priority; "
            "background=denoised_background AND noisy_background after clean/center/bright priority; remaining=ignore"
            if bool(args.align_denoised_noisy_snr_labels)
            else None
        ),
        "pu_runtime_config": {
            "b_mag_min": float(runtime.pu_b_mag_min),
            "b_mag_max": float(runtime.pu_b_mag_max),
            "use_band_limit_b_filter": bool(runtime.pu_use_band_limit_b_filter),
            "band_limit_mags": runtime.pu_band_limit_mags,
            "ap2_kron_abs_max": float(runtime.pu_ap2_kron_abs_max),
            "center_only_fill_area_min": float(runtime.pu_center_only_fill_area_min),
            "center_only_fill_ratio_max": float(runtime.pu_center_only_fill_ratio_max),
            "require_kron_refit_match": bool(runtime.pu_require_kron_refit_match),
            "remeasure_ap2_kron_outliers": bool(runtime.pu_remeasure_ap2_kron_outliers),
            "enable_strict_bright_center_only": bool(runtime.pu_enable_strict_bright_center_only),
            "target_shape_source": str(runtime.target_shape_source),
            "noncoadd_snr_use_source_mask": bool(runtime.noncoadd_snr_use_source_mask),
            "noncoadd_snr_use_quality_mask": bool(runtime.noncoadd_snr_use_quality_mask),
            "noncoadd_snr_mask_planes": list(runtime.noncoadd_snr_mask_planes),
            "noncoadd_snr_exclude_self_source": bool(runtime.noncoadd_snr_exclude_self_source),
        },
        "created_unix_time": time.time(),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coadd-root", type=Path, required=True)
    p.add_argument("--catalog-root", type=Path, required=True)
    p.add_argument("--band-catalog-root", type=Path, default=None)
    p.add_argument("--denoised-fits-root", type=Path, default=None)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--tract", type=int, default=9813)
    p.add_argument("--patches", nargs="+", default=["all"])
    p.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    p.add_argument("--catalog-band", default="HSC-I")
    p.add_argument("--catalog-hdu", type=int, default=1)
    p.add_argument("--image-variants", nargs="+", default=["denoised", "noisy"])
    p.add_argument("--image-variant-groups", nargs="*", default=())
    p.add_argument(
        "--missing-band-policy",
        choices=("error", "skip"),
        default="error",
        help="How to handle absent per-band coadd/catalog files. Use skip for mixed broadband/narrowband image-level Zarrs.",
    )
    p.add_argument(
        "--missing-variant-policy",
        choices=("skip", "error"),
        default="skip",
        help="How to handle requested denoised/noisy patch/group directories or FITS files that are absent.",
    )
    p.add_argument("--include-coadd", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--write-image-level-zarr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also write single-band image-level Zarr stores for SAM detector training.",
    )
    p.add_argument(
        "--image-level-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only write image-level Zarr stores; skip legacy multiband patch stores.",
    )
    p.add_argument(
        "--quality-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Filter image-level training tiles by calexp mask bad-score.",
    )
    p.add_argument("--quality-bad-score-threshold", type=float, default=0.13)
    p.add_argument("--quality-bad-score-weights", nargs="*", default=None)
    p.add_argument(
        "--quality-filter-missing-policy",
        choices=("keep", "drop", "error"),
        default="keep",
        help="How to handle bands whose calexp mask cannot be scored.",
    )
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--stride", type=int, default=368)
    p.add_argument("--compare-origin", nargs=2, type=int, default=None)
    p.add_argument("--tile-filter", nargs="*", default=())
    p.add_argument("--max-tiles", type=int, default=None)
    p.add_argument("--x-col", default="base_SdssCentroid_x")
    p.add_argument("--y-col", default="base_SdssCentroid_y")
    p.add_argument("--source-filter", default="nchild0")
    p.add_argument("--target-shape-source", choices=("sdss", "kron", "circular_kron", "hsm"), default="kron")
    p.add_argument("--kron-refit-csv", default="/nvme0/zc/scarlet/refit/{tract}/{band}/{patch}/batch_heavyfp_kron_refit/batch_heavyfp_kron_refit.csv")
    p.add_argument("--kron-refit-radius-column", default="proxy_nan0_flux_aperture_radius")
    p.add_argument("--kron-refit-good-column", default="proxy_nan0_good")
    p.add_argument("--require-kron-refit-match", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mag-column", default="ext_photometryKron_KronFlux_instFlux")
    p.add_argument("--zeropoint", type=float, default=27.0)
    p.add_argument("--b-mag-min", type=float, default=15.0)
    p.add_argument("--b-mag-max", type=float, default=35.0)
    p.add_argument("--a-area-max", type=float, default=10000.0)
    p.add_argument("--a-faint-area-max", type=float, default=900.0)
    p.add_argument("--a-faint-mag-min", type=float, default=28.0)
    p.add_argument("--center-only-fill-area-min", type=float, default=500.0)
    p.add_argument("--center-only-fill-ratio-max", type=float, default=0.3)
    p.add_argument("--ap2-kron-abs-max", type=float, default=1.0)
    p.add_argument("--ap2-kron-bright-mag-threshold", type=float, default=22.0)
    p.add_argument("--ap2-kron-bright-abs-max", type=float, default=2.0)
    p.add_argument("--ap2-kron-bright-region-column", default="pu_bright_region_center")
    p.add_argument("--ap2-kron-bright-region-area-column", default="pu_bright_region_component_area")
    p.add_argument("--ap2-kron-large-bright-region-area-min", type=float, default=1000.0)
    p.add_argument("--ap2-flux-column", default="base_CircularApertureFlux_6_0_instFlux")
    p.add_argument("--ap2-kron-flux-column", default="ext_photometryKron_KronFlux_instFlux")
    p.add_argument("--b-flags", nargs="*", default=("base_SdssShape_flag", "base_SdssCentroid_flag"))
    p.add_argument("--close-center-arcsec", type=float, default=0.5)
    p.add_argument("--axis-ratio-max", type=float, default=5.0)
    p.add_argument("--containment-threshold", type=float, default=0.80)
    p.add_argument("--drop-ellipse-area-min", type=float, default=40000.0)
    p.add_argument("--remeasure-ap2-kron-outliers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--enable-strict-bright-center-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--strict-bright-center-only-mag-threshold", type=float, default=None)
    p.add_argument("--strict-bright-center-only-saturation-mags", nargs="*", default=None)
    p.add_argument("--strict-bright-center-only-radius-column", default="proxy_nan0_flux_aperture_radius")
    p.add_argument("--strict-bright-center-only-ellipse-sigma", type=float, default=1.0)
    p.add_argument("--pu-enable-bright-background-mask", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--pu-bright-mask-mode",
        choices=("log-lupton", "zscore", "zscore-lupton-log", "zscore-no-upper", "zscore-unbounded", "anscombe", "raw", "none"),
        default="log-lupton",
        help=(
            "Bright-region mask scaling. zscore-no-upper/zscore-unbounded intentionally produce no image bright "
            "mask; external bright labels should be supplied by the source-cluster/Gaia flow."
        ),
    )
    p.add_argument("--pu-bright-log-a", type=float, default=300.0)
    p.add_argument("--pu-bright-log-high-percentile", type=float, default=99.5)
    p.add_argument("--pu-bright-lupton-stretch", type=float, default=0.5)
    p.add_argument("--pu-bright-lupton-q", type=float, default=20.0)
    p.add_argument("--pu-bright-anscombe-scale", type=float, default=1000.0)
    p.add_argument("--pu-bright-z-threshold", type=float, default=3.0)
    p.add_argument("--pu-bright-mask-dilate", type=int, default=2)
    p.add_argument(
        "--external-bright-label-root",
        "--pu-external-bright-label-root",
        dest="external_bright_label_root",
        type=Path,
        default=None,
        help=(
            "Optional root produced by data_filtering/build_external_bright_labels_v2.py. "
            "The CSV labels override bright-source clean/center/ignore classes; bright_mask.fits is ORed "
            "into the dense-target bright region."
        ),
    )
    p.add_argument("--ellipse-sigma", type=float, default=1.0)
    p.add_argument("--mask-margin", type=float, default=64.0)
    p.add_argument("--confidence-levels", type=int, default=5)
    p.add_argument("--core-radius", type=int, default=2)
    p.add_argument("--center-only-weight", type=float, default=0.25)
    p.add_argument("--background-policy", choices=("existing", "none"), default="existing")
    p.add_argument(
        "--variant-lsst-background-root",
        type=Path,
        default=None,
        help=(
            "Optional root containing denoised/noisy LSST background masks. Supported layout: "
            "<root>/<variant>/<tract>/<patch>/<group>/<band>/background_mask.npz or det-*.fits."
        ),
    )
    p.add_argument(
        "--variant-lsst-background-policy",
        choices=("run-if-missing", "existing", "none"),
        default="run-if-missing",
    )
    p.add_argument(
        "--lsst-detect-python",
        default=None,
        help="Python executable in an LSST stack environment for generating missing variant backgrounds.",
    )
    p.add_argument(
        "--image-variant-background-source",
        choices=("auto", "coadd-target", "variant-lsst", "none"),
        default="variant-lsst",
        help=(
            "Background source for denoised/noisy labels. auto prefers --variant-lsst-background-root "
            "and falls back to the coadd det background; coadd-target always reuses coadd; "
            "variant-lsst uses only variant/group masks; none disables variant backgrounds."
        ),
    )
    p.add_argument("--pixel-scale-arcsec", type=float, default=0.168)
    p.add_argument("--noncoadd-snr-filter", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--noncoadd-snr-ignore-thresh", type=float, default=2.0)
    p.add_argument("--noncoadd-snr-center-only-thresh", type=float, default=3.0)
    p.add_argument("--noncoadd-snr-ap-radius", type=float, default=6.0)
    p.add_argument("--noncoadd-snr-annulus-r-in", type=float, default=10.0)
    p.add_argument("--noncoadd-snr-annulus-r-out", type=float, default=15.0)
    p.add_argument("--noncoadd-snr-source-mask-ellipse-sigma", type=float, default=1.0)
    p.add_argument("--noncoadd-snr-min-annulus-pixels", type=int, default=50)
    p.add_argument("--noncoadd-snr-use-source-mask", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--noncoadd-snr-use-quality-mask", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--noncoadd-snr-mask-planes",
        nargs="*",
        default=["BRIGHT_OBJECT", "SAT", "BAD", "NO_DATA", "EDGE", "UNMASKEDNAN"],
    )
    p.add_argument("--noncoadd-snr-exclude-self-source", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--align-denoised-noisy-snr-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For denoised/noisy variant Zarrs, align PU labels after the per-variant SNR/background pass: "
            "clean is the denoised/noisy clean intersection, center-only is the union of either center-only class, "
            "bright is the denoised/noisy bright union, background is the denoised/noisy background intersection, "
            "and all remaining pixels are ignore."
        ),
    )
    p.add_argument("--z-clip", nargs=2, type=float, default=None)
    p.add_argument(
        "--image-scaling-mode",
        choices=(
            "astro-zscore",
            "zscore",
            "zscore-rgb",
            "zscore-no-upper",
            "zscore-no-upper-rgb",
            "zscore-unbounded",
            "zscore-unbounded-rgb",
            "zscore-log-lupton-rgb",
            "zscore-lupton-log-rgb",
            "log-lupton-rgb",
            "anscombe",
            "anscombe-rgb",
        ),
        default="astro-zscore",
        help=(
            "Preprocessing written into the Zarr images array. Modes ending in -rgb write "
            "[sample, band, 3, H, W] tensors for native SAM RGB input; astro-zscore preserves the legacy "
            "[sample, band, H, W] layout."
        ),
    )
    p.add_argument("--image-log-a", type=float, default=300.0)
    p.add_argument("--image-log-high-percentile", type=float, default=99.5)
    p.add_argument("--image-lupton-stretch", type=float, default=0.5)
    p.add_argument("--image-lupton-q", type=float, default=20.0)
    p.add_argument("--image-anscombe-scale", type=float, default=1000.0)
    p.add_argument("--image-dtype", choices=("float16", "float32"), default="float16")
    p.add_argument("--target-float-dtype", choices=("float16", "float32"), default="float16")
    p.add_argument("--chunk-tiles", type=int, default=8)
    p.add_argument("--tile-workers", type=int, default=2)
    p.add_argument("--patch-workers", type=int, default=1)
    p.add_argument(
        "--worker-threads",
        type=int,
        default=1,
        help="Torch/BLAS/OpenMP threads per patch process; tile parallelism is controlled separately.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    args.bands = tuple(args.bands)
    args.image_dtype = np.dtype(args.image_dtype)
    args.target_float_dtype = np.dtype(args.target_float_dtype)
    args.z_clip = tuple(args.z_clip) if args.z_clip is not None else None
    patches = _expand_patch_tokens(args.patches)
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    if int(args.patch_workers) <= 1 or len(patches) <= 1:
        _configure_worker_threads(int(args.worker_threads))
        summaries = [preprocess_patch(args, patch) for patch in patches]
    else:
        summaries = []
        with ProcessPoolExecutor(
            max_workers=min(int(args.patch_workers), len(patches)),
            initializer=_configure_worker_threads,
            initargs=(int(args.worker_threads),),
        ) as ex:
            futs = {ex.submit(preprocess_patch, args, patch): patch for patch in patches}
            for fut in as_completed(futs):
                summaries.append(fut.result())
                print(f"[direct-zarr] completed patch {futs[fut]}", flush=True)
    write_json(output_root / "direct_preprocess_manifest.json", {"patches": summaries, "args": vars(args)})
    print(f"[direct-zarr] wrote {len(summaries)} patch summary record(s) to {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
