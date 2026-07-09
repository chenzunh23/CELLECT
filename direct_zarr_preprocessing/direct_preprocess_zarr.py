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
from types import SimpleNamespace
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
    _classify_clean_by_noncoadd_snr,
    _classify_pu_catalog,
    _crop_full_mask_for_tile,
    _crop_image_for_tile,
    _find_denoised_patch_dir,
    _find_image_hdu_index,
    _metadata_from_catalog,
    _move_bright_clean_to_center_only,
    _origin_from_ltv,
    _paint_ellipse_mask,
    _pu_dropped_sources,
    _read_det_background_mask,
    _read_exposure_image_plane,
    _read_table,
    _source_annulus_exclusion_mask,
    _variant_lsst_background_mask,
    _vstack_nonempty,
    add_ellipse_columns,
    crop_catalog_for_tile,
    make_pu_dense_targets,
    make_tile_specs,
)
from direct_zarr_preprocessing.zarr_writer import ZarrGroupWriter, encode_fixed_utf8, write_json  # noqa: E402

DEFAULT_BANDS = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y")


@dataclass(frozen=True)
class Sample:
    name: str
    spec: TileSpec
    dataset_source: str
    group: str
    image_key: str


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


def _legacy_args(args: argparse.Namespace) -> SimpleNamespace:
    """Build the small namespace required by imported filtering functions."""

    return SimpleNamespace(
        coadd_root=str(args.coadd_root),
        catalog_root=str(args.catalog_root),
        band_catalog_root=str(args.band_catalog_root or args.catalog_root),
        tract=int(args.tract),
        catalog_hdu=int(args.catalog_hdu),
        catalog_band=str(args.catalog_band),
        x_col=args.x_col,
        y_col=args.y_col,
        source_filter=args.source_filter,
        shape_source=args.target_shape_source,
        target_shape_source=args.target_shape_source,
        max_area_3sigma=400.0,
        relaxed_area_3sigma=900.0,
        area_filter_policy="max_area",
        drop_children=False,
        label_mode="pu",
        ellipse_sigma=float(args.ellipse_sigma),
        min_ellipse_axis=1.5,
        pixel_scale_arcsec=float(args.pixel_scale_arcsec),
        no_clean_nonfinite=False,
        pu_a_flags=(),
        pu_b_flags=tuple(args.b_flags),
        pu_a_mode="any",
        pu_b_mode="any",
        pu_strict_flags=(),
        pu_mag_column=args.mag_column,
        pu_input_zeropoint=float(args.zeropoint),
        pu_require_kron_refit_match=bool(args.require_kron_refit_match),
        pu_kron_refit_csv=str(args.kron_refit_csv),
        pu_kron_refit_radius_column=args.kron_refit_radius_column,
        pu_kron_refit_good_column=args.kron_refit_good_column,
        pu_a_area_max=float(args.a_area_max),
        pu_a_faint_area_max=float(args.a_faint_area_max),
        pu_a_faint_mag_min=float(args.a_faint_mag_min),
        pu_b_mag_min=float(args.b_mag_min),
        pu_b_mag_max=float(args.b_mag_max),
        pu_use_band_limit_b_filter=False,
        pu_band_limit_mags=None,
        pu_band_limit_b_min_offset=-5.0,
        pu_band_limit_b_max_offset=0.0,
        pu_ap2_kron_abs_max=float(args.ap2_kron_abs_max),
        pu_ap2_flux_column=args.ap2_flux_column,
        pu_ap2_kron_flux_column=args.ap2_kron_flux_column,
        pu_b_close_center_arcsec=float(args.close_center_arcsec),
        pu_overlap_iou_threshold=0.33,
        pu_b_ellipse_area_max=None,
        pu_b_footprint_area_max=None,
        pu_b_axis_ratio_max=float(args.axis_ratio_max),
        pu_b_kron_radius_lt_sdss_major_ratio=0.5,
        pu_drop_ellipse_area_min=float(args.drop_ellipse_area_min),
        pu_ambiguous_area_max=None,
        pu_neighbor_radius=0.0,
        pu_center_distance_factor=0.0,
        pu_containment_threshold=float(args.containment_threshold),
        pu_mutual_overlap_threshold=0.0,
        pu_overlap_sample_grid=16,
        pu_ambiguous_mark="center_only",
        pu_keep_all_ab_clean=True,
        pu_enable_strict_bright_center_only=bool(args.enable_strict_bright_center_only),
        pu_strict_bright_center_only_mag_threshold=args.strict_bright_center_only_mag_threshold,
        pu_strict_ignore_mag_threshold=None,
        pu_strict_bright_center_only_saturation_mags=args.strict_bright_center_only_saturation_mags,
        pu_strict_ignore_saturation_mags=None,
        pu_strict_bright_center_only_radius_column=args.strict_bright_center_only_radius_column,
        pu_strict_bright_center_only_ellipse_sigma=float(args.strict_bright_center_only_ellipse_sigma),
        pu_remeasure_ap2_kron_outliers=bool(args.remeasure_ap2_kron_outliers),
        pu_remeasure_ap2_kron_threshold=np.nan,
        pu_remeasure_clean_abs_max=float(args.ap2_kron_abs_max),
        pu_remeasure_center_only_abs_max=1.5,
        pu_remeasure_small_footprint_fill_threshold=0.2,
        pu_remeasure_ignore_area_max=10000.0,
        pu_remeasure_faint_mag_min=28.0,
        pu_remeasure_faint_area_max=900.0,
        pu_remeasure_axis_ratio_max=float(args.axis_ratio_max),
        pu_remeasure_containment_threshold=float(args.containment_threshold),
        noncoadd_snr_ap_radius=float(args.noncoadd_snr_ap_radius),
        noncoadd_snr_annulus_r_in=float(args.noncoadd_snr_annulus_r_in),
        noncoadd_snr_annulus_r_out=float(args.noncoadd_snr_annulus_r_out),
        noncoadd_snr_ignore_thresh=float(args.noncoadd_snr_ignore_thresh),
        noncoadd_snr_center_only_thresh=float(args.noncoadd_snr_center_only_thresh),
        noncoadd_snr_source_mask_ellipse_sigma=float(args.noncoadd_snr_source_mask_ellipse_sigma),
        noncoadd_snr_min_annulus_pixels=int(args.noncoadd_snr_min_annulus_pixels),
        noncoadd_snr_use_source_mask=True,
        noncoadd_snr_use_quality_mask=False,
    )


def _read_patch_image_meta(coadd_root: Path, band: str, tract: int, patch: str) -> tuple[tuple[int, int], tuple[int, int]]:
    from astropy.io import fits

    path = _band_fits_path(coadd_root, band, tract, patch)
    with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
        hdu = hdul[_find_image_hdu_index(hdul)]
        origin = _origin_from_ltv(hdu.header)
        shape = tuple(int(v) for v in hdu.data.shape)
    return shape, origin


def _classify_all_bands(args: argparse.Namespace, legacy: SimpleNamespace, patch: str) -> dict[str, tuple[Table, Table, Table, Table]]:
    catalog_root = Path(args.band_catalog_root or args.catalog_root).expanduser()
    out = {}
    for band in args.bands:
        table = _read_table(
            _band_catalog_path(catalog_root, band, int(args.tract), patch),
            hdu=int(args.catalog_hdu),
            role="direct-zarr-band",
            patch=patch,
            band=band,
        )
        clean, center, ignore, _all, _result = _classify_pu_catalog(table, legacy, band=band, patch=patch)
        clean, center, strict_center = _move_bright_clean_to_center_only(clean, center, legacy, band=band)
        out[band] = (clean, center, ignore, strict_center)
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


def _read_coadd_images(args: argparse.Namespace, patch: str) -> dict[str, tuple[np.ndarray, tuple[int, int]]]:
    root = Path(args.coadd_root).expanduser()
    return {
        band: _read_exposure_image_plane(_band_fits_path(root, band, int(args.tract), patch), clean_nonfinite=True)
        for band in args.bands
    }


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
    out = {}
    for group in groups:
        band_images = {}
        for band in args.bands:
            path = group / band / f"{variant}.fits"
            if not path.exists():
                raise FileNotFoundError(f"missing {variant} FITS: {path}")
            band_images[band] = _read_exposure_image_plane(path, clean_nonfinite=True)
        out[group.name] = band_images
    return out


def _make_samples(dataset: str, specs: Sequence[TileSpec], image_keys: Sequence[str]) -> list[Sample]:
    samples = []
    for key in image_keys:
        for spec in specs:
            name = spec.name if dataset == "coadd" else f"{key}_{spec.name}"
            samples.append(Sample(name=name, spec=spec, dataset_source=dataset, group="" if dataset == "coadd" else key, image_key=key))
    return samples


def _crop_sources(source_tuple, spec: TileSpec, args: argparse.Namespace, *, margin: float) -> tuple[Table, Table, Table, Table]:
    return tuple(
        crop_catalog_for_tile(part, spec, x_col=args.x_col, y_col=args.y_col, margin=margin)
        for part in source_tuple
    )


def _local_image_stack(images_by_band: dict[str, tuple[np.ndarray, tuple[int, int]]], sample: Sample, bands: Sequence[str]) -> np.ndarray:
    planes = []
    for band in bands:
        image, origin = images_by_band[band]
        tile, _tile_origin = _crop_image_for_tile(image, sample.spec, origin)
        planes.append(tile)
    return np.stack(planes, axis=0).astype(np.float32, copy=False)


def _target_for_sample(
    sample: Sample,
    *,
    bands: Sequence[str],
    labels_by_band,
    images_by_band,
    backgrounds,
    args,
    legacy,
) -> dict[str, object]:
    image_stack = _local_image_stack(images_by_band, sample, bands)
    image = astro_zscale_preprocess(image_stack, z_clip=args.z_clip).cpu().numpy().astype(args.image_dtype, copy=False)
    conf = []
    conf_weight = []
    shape = []
    shape_weight = []
    pu_class = []
    centers_by_band = []
    ids_by_band = []
    for band_idx, band in enumerate(bands):
        clean_mask, center_mask, ignore_mask, strict_mask = _crop_sources(
            labels_by_band[band],
            sample.spec,
            args,
            margin=float(args.mask_margin),
        )
        clean_tile, _center_tile, _ignore_tile, _strict_tile = _crop_sources(
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
                    sample.spec,
                    x_col=args.x_col,
                    y_col=args.y_col,
                    ellipse_sigma=float(args.noncoadd_snr_source_mask_ellipse_sigma),
                )
            normal_tile, snr_center_tile, snr_ignore_tile, _snr = _classify_clean_by_noncoadd_snr(
                clean_tile,
                image=image_stack[band_idx],
                image_origin=(sample.spec.x0, sample.spec.y0),
                args=legacy,
                annulus_exclude_mask=source_mask,
                annulus_hard_exclude_mask=None,
            )
            normal_mask, snr_center_mask, snr_ignore_mask, _snr = _classify_clean_by_noncoadd_snr(
                clean_mask,
                image=image_stack[band_idx],
                image_origin=(sample.spec.x0, sample.spec.y0),
                args=legacy,
                annulus_exclude_mask=source_mask,
                annulus_hard_exclude_mask=None,
            )
            clean_tile = normal_tile
            clean_mask = normal_mask
            center_mask = _vstack_nonempty([center_mask, snr_center_mask])
            ignore_mask = _vstack_nonempty([ignore_mask, snr_ignore_mask])
        lsst_background = None
        if band in backgrounds:
            background_mask, background_origin = backgrounds[band]
            lsst_background = _crop_full_mask_for_tile(background_mask, sample.spec, background_origin)
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
            strict_center_only_sources=strict_mask,
            strict_center_only_ellipse_sigma=float(args.strict_bright_center_only_ellipse_sigma),
        )
        conf.append(np.asarray(target["confidence"], dtype=np.uint8))
        conf_weight.append(np.asarray(target["confidence_weight"], dtype=args.target_float_dtype))
        shape.append(np.asarray(target["shape"], dtype=args.target_float_dtype))
        shape_weight.append(np.asarray(target["shape_weight"], dtype=args.target_float_dtype))
        pu_class.append(np.asarray(target["pu_class_mask"], dtype=np.uint8))
        meta = _metadata_from_catalog(clean_tile)
        centers_by_band.append(meta["centers"])
        ids_by_band.append(meta["ids"])
    return {
        "image": image,
        "confidence": np.stack(conf, axis=0),
        "confidence_weight": np.stack(conf_weight, axis=0),
        "shape": np.stack(shape, axis=0),
        "shape_weight": np.stack(shape_weight, axis=0),
        "pu_class": np.stack(pu_class, axis=0),
        "centers_by_band": centers_by_band,
        "ids_by_band": ids_by_band,
    }


def _write_dataset_zarr(
    *,
    output: Path,
    samples: Sequence[Sample],
    bands: Sequence[str],
    labels_by_band,
    image_sources,
    background_sources,
    args,
    legacy,
    attrs: dict,
) -> dict:
    manifest_path = output.parent / f"{output.name}_manifest.json"
    if manifest_path.exists() and output.exists() and not bool(args.overwrite):
        print(f"[direct-zarr] skip complete existing store: {output}", flush=True)
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if output.exists() and not manifest_path.exists() and not bool(args.overwrite):
        print(f"[direct-zarr] removing incomplete existing store before rebuild: {output}", flush=True)
        shutil.rmtree(output)
    n = len(samples)
    b = len(bands)
    h = w = int(args.tile_size)
    chunk_tiles = max(1, int(args.chunk_tiles))
    writer = ZarrGroupWriter(output, overwrite=bool(args.overwrite), attrs=attrs)
    images = writer.array("images", shape=(n, b, h, w), chunks=(chunk_tiles, b, h, w), dtype=args.image_dtype)
    confidence = writer.array("band_confidence", shape=(n, b, h, w), chunks=(chunk_tiles, b, h, w), dtype=np.uint8)
    conf_weight = writer.array("band_conf_weight", shape=(n, b, h, w), chunks=(chunk_tiles, b, h, w), dtype=args.target_float_dtype)
    shape = writer.array("band_shape", shape=(n, b, 3, h, w), chunks=(chunk_tiles, b, 3, h, w), dtype=args.target_float_dtype)
    shape_weight = writer.array("band_shape_weight", shape=(n, b, h, w), chunks=(chunk_tiles, b, h, w), dtype=args.target_float_dtype)
    pu_class = writer.array("band_pu_class_mask", shape=(n, b, h, w), chunks=(chunk_tiles, b, h, w), dtype=np.uint8)
    centers_flat = []
    ids_flat = []
    offsets = np.zeros((n, b + 1), dtype=np.int64)

    def load_one(sample: Sample):
        images_by_band = image_sources[sample.image_key]
        backgrounds = background_sources.get(sample.image_key, {})
        return _target_for_sample(
            sample,
            bands=bands,
            labels_by_band=labels_by_band,
            images_by_band=images_by_band,
            backgrounds=backgrounds,
            args=args,
            legacy=legacy,
        )

    for start in range(0, n, chunk_tiles):
        end = min(n, start + chunk_tiles)
        batch = list(samples[start:end])
        with ThreadPoolExecutor(max_workers=max(1, int(args.tile_workers))) as executor:
            packages = list(executor.map(load_one, batch))
        chunk_index = (start // chunk_tiles, 0, 0, 0)
        images.write_chunk(chunk_index, np.stack([pkg["image"] for pkg in packages], axis=0))
        confidence.write_chunk(chunk_index, np.stack([pkg["confidence"] for pkg in packages], axis=0))
        conf_weight.write_chunk(chunk_index, np.stack([pkg["confidence_weight"] for pkg in packages], axis=0))
        shape.write_chunk((start // chunk_tiles, 0, 0, 0, 0), np.stack([pkg["shape"] for pkg in packages], axis=0))
        shape_weight.write_chunk(chunk_index, np.stack([pkg["shape_weight"] for pkg in packages], axis=0))
        pu_class.write_chunk(chunk_index, np.stack([pkg["pu_class"] for pkg in packages], axis=0))
        for local_idx, pkg in enumerate(packages):
            sample_idx = start + local_idx
            offsets[sample_idx, 0] = len(centers_flat)
            for band_idx in range(b):
                centers = np.asarray(pkg["centers_by_band"][band_idx], dtype=np.float32).reshape(-1, 2)
                ids = np.asarray(pkg["ids_by_band"][band_idx], dtype=np.int64).reshape(-1)
                if len(centers):
                    centers_flat.extend(centers)
                    ids_flat.extend(ids)
                offsets[sample_idx, band_idx + 1] = len(centers_flat)
        print(f"[direct-zarr] {output.name}: wrote samples {start + 1}-{end}/{n}", flush=True)

    centers_arr = np.asarray(centers_flat, dtype=np.float32).reshape(-1, 2)
    ids_arr = np.asarray(ids_flat, dtype=np.int64).reshape(-1)
    writer.array("source_centers", shape=centers_arr.shape, chunks=(max(1, len(centers_arr)), 2), dtype=np.float32).write_full(centers_arr)
    writer.array("source_ids", shape=ids_arr.shape, chunks=(max(1, len(ids_arr)),), dtype=np.int64).write_full(ids_arr)
    writer.array("source_offsets", shape=offsets.shape, chunks=(max(1, n), b + 1), dtype=np.int64).write_full(offsets)
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
    write_json(manifest_path, manifest)
    return manifest


def preprocess_patch(args: argparse.Namespace, patch: str) -> dict:
    output_root = Path(args.output_root).expanduser() / str(args.tract)
    expected_manifests = []
    if args.include_coadd:
        expected_manifests.append(output_root / "coadd" / f"{patch}.zarr_manifest.json")
    variant_patch_dir = None
    if args.denoised_fits_root is not None:
        variant_patch_dir = _find_denoised_patch_dir(Path(args.denoised_fits_root).expanduser(), patch)
        if variant_patch_dir is not None:
            for variant in args.image_variants:
                expected_manifests.append(output_root / variant / f"{patch}.zarr_manifest.json")
    if expected_manifests and not bool(args.overwrite) and all(path.exists() for path in expected_manifests):
        print(f"[direct-zarr] skip complete patch {patch}", flush=True)
        return {
            "patch": patch,
            "summaries": [json.loads(path.read_text(encoding="utf-8")) for path in expected_manifests],
            "skipped_existing": True,
        }

    legacy = _legacy_args(args)
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
    labels = _classify_all_bands(args, legacy, patch)
    backgrounds = _read_backgrounds(args, patch, shape_yx, parent_origin)
    summaries = []

    if args.include_coadd:
        coadd_images = {"coadd": _read_coadd_images(args, patch)}
        samples = _make_samples("coadd", specs, ["coadd"])
        out = output_root / "coadd" / f"{patch}.zarr"
        summaries.append(
            _write_dataset_zarr(
                output=out,
                samples=samples,
                bands=args.bands,
                labels_by_band=labels,
                image_sources=coadd_images,
                background_sources={"coadd": backgrounds},
                args=args,
                legacy=legacy,
                attrs=_attrs(args, patch, "coadd", len(samples)),
            )
        )

    if args.denoised_fits_root is not None:
        for variant in args.image_variants:
            variant_images = _read_variant_images(args, patch, variant)
            variant_backgrounds = _variant_backgrounds_for_images(
                args,
                variant=variant,
                patch=patch,
                variant_images=variant_images,
                coadd_backgrounds=backgrounds,
            )
            samples = _make_samples(variant, specs, sorted(variant_images))
            out = output_root / variant / f"{patch}.zarr"
            summaries.append(
                _write_dataset_zarr(
                    output=out,
                    samples=samples,
                    bands=args.bands,
                    labels_by_band=labels,
                    image_sources=variant_images,
                    background_sources=variant_backgrounds,
                    args=args,
                    legacy=legacy,
                    attrs=_attrs(args, patch, variant, len(samples)),
                )
            )
    return {"patch": patch, "summaries": summaries}


def _attrs(args: argparse.Namespace, patch: str, dataset_source: str, n: int) -> dict:
    return {
        "format": "cellect_direct_patch_zarr",
        "format_version": 1,
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
    p.add_argument("--include-coadd", action=argparse.BooleanOptionalAction, default=True)
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
    p.add_argument("--ap2-kron-abs-max", type=float, default=1.0)
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
        "--image-variant-background-source",
        choices=("auto", "coadd-target", "variant-lsst", "none"),
        default="auto",
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
    p.add_argument("--z-clip", nargs=2, type=float, default=None)
    p.add_argument("--image-dtype", choices=("float16", "float32"), default="float16")
    p.add_argument("--target-float-dtype", choices=("float16", "float32"), default="float16")
    p.add_argument("--chunk-tiles", type=int, default=8)
    p.add_argument("--tile-workers", type=int, default=2)
    p.add_argument("--patch-workers", type=int, default=1)
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
        summaries = [preprocess_patch(args, patch) for patch in patches]
    else:
        summaries = []
        with ProcessPoolExecutor(max_workers=min(int(args.patch_workers), len(patches))) as ex:
            futs = {ex.submit(preprocess_patch, args, patch): patch for patch in patches}
            for fut in as_completed(futs):
                summaries.append(fut.result())
                print(f"[direct-zarr] completed patch {futs[fut]}", flush=True)
    write_json(output_root / "direct_preprocess_manifest.json", {"patches": summaries, "args": vars(args)})
    print(f"[direct-zarr] wrote {len(summaries)} patch summary record(s) to {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
