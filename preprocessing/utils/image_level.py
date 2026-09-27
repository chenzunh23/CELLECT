"""Image-level task records, input resolution and tile-target adapters."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
from astropy.io import fits
from astropy.table import Table
from preprocessing.utils.inputs import (_band_fits_path, _band_det_path,
    _find_image_hdu_index, _origin_from_ltv, _read_det_background_mask)
from preprocessing.image_processing import read_background_mask, ImageProcessingConfig, scale_image_for_training
from preprocessing.labels import LabelWeights, SourceClass
from preprocessing.utils.geometry import paint_ellipse

MASK_PLANES_FOR_STRICT_IGNORE = ("SAT", "BAD", "NO_DATA", "UNMASKEDNAN")


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
    bright_object_mask_root: Path | None
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
    subtract_bright_object_from_background: bool = False
    image_variant_background_source: str = "auto"
    missing_variant_background_policy: str = "fallback_coadd"
    confidence_mode: str = "auto"
    confidence_config_path: str | None = None
    confidence_fwhm_min: float = 1.6
    confidence_fwhm_max: float = 8.0
    confidence_fwhm_pixels: float | None = None


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
    segmentation_ids: np.ndarray | None = None
    segmentation_weight: np.ndarray | None = None
    strict_is_gaia: np.ndarray | None = None
    segmentation_policy: dict | None = None
    segmentation_overlap_masks: list[dict] | None = None
    truncation_geometry: dict | None = None  # Raw large Krons, including off-window centers.


def attach_cosmos_segmentation(labels: PatchLabels, wcs, catalog_path, *,
                               origin=(0, 0), valid_mask=None, clearance=0,
                               weight=0.25, allow_nested=True,
                               main_fraction_min=0.95, closing_radius=0,
                               gaussian_sigma=1.0, gaussian_min_raw_area=100) -> None:
    """Attach positive-only masks after final COSMOS source classification.

    WCS describes the full image; origin locates this patch in that image.
    Run before tiling so artificial tile edges do not change isolation.
    """
    from preprocessing.labels import SourceLabels
    from preprocessing.utils.segmentation import (
        read_cosmos_segment_cutout, isolated_segmentation_targets, SEGMENTATION_POLICY_VERSION,
    )
    seg, stars, valid = read_cosmos_segment_cutout(
        wcs, labels.dense.shape, catalog_path, origin=origin)
    if valid_mask is not None:
        if np.shape(valid_mask) != valid.shape:
            raise ValueError("segmentation valid-mask shape mismatch")
        valid &= np.asarray(valid_mask, bool)
    source_labels = SourceLabels(labels.label_classes, np.full(len(labels.source_ids), "", object))
    target = isolated_segmentation_targets(seg, labels.source_ids, source_labels,
        star_mask=stars, valid_mask=valid, clearance=clearance, weight=weight,
        allow_nested=allow_nested, main_fraction_min=main_fraction_min,
        closing_radius=closing_radius, gaussian_sigma=gaussian_sigma,
        gaussian_min_raw_area=gaussian_min_raw_area)
    labels.segmentation_overlap_masks = target.overlap_masks
    labels.segmentation_ids = target.instance_ids
    labels.segmentation_weight = target.positive_weight
    labels.segmentation_policy = dict(
        version=SEGMENTATION_POLICY_VERSION, allow_nested=bool(allow_nested),
        main_fraction_min=float(main_fraction_min), closing_radius=int(closing_radius),
        gaussian_sigma_pixels=float(gaussian_sigma), gaussian_min_raw_area=int(gaussian_min_raw_area),
        gaussian_area_comparison='raw_area > threshold', gaussian_binary_threshold=0.5,
        area_basis='raw pixel count before component cleanup',
        overlap_basis='processed independent masks; touching allowed at clearance=0',
        isolation_blockers='all non-dropped IDs; unknown IDs block',
        overlap_sidecar_count=len(target.overlap_masks),
        clearance_pixels=int(clearance), positive_weight=float(weight),
        parent_selected_counts={kind: sum(r['mask_type'] == kind for r in target.sources)
                                for kind in ('isolated', 'nested_isolated')},
        containment='all RAW child pixels inside RAW parent outer boundary; 8-connected exterior',
        acceptance='parent requires all descendants; child independent of parent eligibility',
        holes='all processable non-dropped IDs fill holes; overlapping parent sidecars',
    )


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
    from preprocessing.dataset_inputs import image_hdu
    with fits.open(path, memmap=None, ignore_missing_end=True) as hdul:
        idx = image_hdu(hdul, hdu)
        data = np.asarray(hdul[idx].data, dtype=np.float32)
        header = hdul[0].header.copy()
        header.update(hdul[idx].header)
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


def _mask_plane_bit(header: fits.Header, plane: str) -> int | None:
    """Return one mask-plane bit for both HSC header encodings."""

    plane = str(plane).upper()
    direct = header.get(f"MP_{plane}")
    if direct is not None:
        try:
            return int(direct)
        except Exception:
            return None
    for key, value in header.items():
        text = str(key).upper()
        if not text.startswith("MP_"):
            continue
        suffix = text[3:]
        if str(value).upper() != plane or not suffix.isdigit():
            continue
        try:
            return int(suffix)
        except Exception:
            return None
    return None


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


def _read_bright_object_mask(root: Path | None, band: str, tract: int, patch: str, shape: tuple[int, int]) -> np.ndarray:
    if root is None:
        raise FileNotFoundError("BRIGHT_OBJECT mask root is not set")
    path = _band_fits_path(root, band, tract, patch)
    if not path.exists():
        raise FileNotFoundError(f"BRIGHT_OBJECT mask FITS missing: {path}")
    with fits.open(path, memmap=False, ignore_missing_end=True) as hdul:
        image_idx = _find_image_hdu_index(hdul)
        if "MASK" in hdul:
            hdu = hdul["MASK"]
        else:
            mask_idx = image_idx + 1
            if mask_idx >= len(hdul) or getattr(hdul[mask_idx], "data", None) is None:
                raise KeyError(f"MASK HDU missing in {path}")
            hdu = hdul[mask_idx]
        mask = np.asarray(hdu.data, dtype=np.int64)
        if mask.shape != shape:
            raise ValueError(f"BRIGHT_OBJECT mask shape mismatch for {path}: {mask.shape} != {shape}")
        bit = _mask_plane_bit(hdu.header, "BRIGHT_OBJECT")
        if bit is None:
            raise KeyError(f"BRIGHT_OBJECT mask plane missing in {path}")
        return (mask & (1 << int(bit))) != 0


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


def _subtract_bright_object_background(task: StoreTask, background: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if not bool(task.subtract_bright_object_from_background):
        return np.asarray(background, dtype=bool)
    bright_object = _read_bright_object_mask(task.bright_object_mask_root, task.band, task.tract, task.patch, shape)
    return np.asarray(background, dtype=bool) & ~np.asarray(bright_object, dtype=bool)


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
            return _subtract_bright_object_background(task, coadd_lsst, shape)
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
        return _subtract_bright_object_background(task, coadd, shape)

    source = str(task.image_variant_background_source).strip().lower()
    if source not in {"auto", "coadd-target", "variant-lsst", "none"}:
        raise ValueError(f"unknown image variant background source: {task.image_variant_background_source}")
    if source == "none":
        return np.zeros(shape, dtype=bool)
    if source == "coadd-target":
        return _subtract_bright_object_background(task, coadd, shape)

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
        return _subtract_bright_object_background(task, variant, shape)
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
    return _subtract_bright_object_background(task, coadd, shape)


def _crop(arr: np.ndarray, x0: int, y0: int, origin: tuple[int, int], size: int) -> np.ndarray:
    lx0 = int(x0) - int(origin[0])
    ly0 = int(y0) - int(origin[1])
    if lx0<0 or ly0<0 or lx0+size>arr.shape[-1] or ly0+size>arr.shape[-2]:
        from .large_sources import crop_padded
        return crop_padded(arr,x0,y0,origin,size,fill=0)
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


def _strict_centers_in_tile(labels: PatchLabels, spec, origin: tuple[int, int], *, with_gaia=False):
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
    result = (centers[keep], labels.strict_ids[keep])
    if with_gaia:
        gaia = np.zeros(len(centers), bool) if labels.strict_is_gaia is None else np.asarray(labels.strict_is_gaia, bool)
        if gaia.shape != (len(centers),):
            raise ValueError('strict_is_gaia must have one flag per extra center')
        return (*result, gaia[keep])
    return result


def _tile_targets(labels: PatchLabels, spec, origin: tuple[int, int], *, confidence=None) -> dict[str, np.ndarray]:
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
    strict_extra_xy, strict_extra_ids, strict_extra_gaia = _strict_centers_in_tile(labels, spec, origin, with_gaia=True)
    all_conf_centers = center_xy
    if len(strict_extra_xy):
        all_conf_centers = np.concatenate([all_conf_centers, strict_extra_xy], axis=0)
    if confidence is not None and confidence['mode'] == 'psf-ee':
        from .confidence import paint_ee_confidence
        paint_ee_confidence(conf, conf_weight, all_conf_centers,
                            level_radii_pixels=confidence['level_radii_pixels'])
    elif confidence is not None and confidence['mode'] == 'psf-matched':
        from .confidence import paint_psf_confidence
        minimum, maximum = confidence['fwhm_clip_pixels']
        paint_psf_confidence(conf, conf_weight, all_conf_centers,
                             fwhm_pixels=confidence['fwhm_pixels_used'],
                             minimum=minimum, maximum=maximum)
    else:
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
    target = {
        "confidence": conf,
        "conf_weight": conf_weight,
        "shape": shape,
        "shape_weight": shape_weight,
        "pu": dense,
        "source_centers": source_centers,
        "source_ids": source_ids_arr,
        "strict_centers": strict_centers,
        "strict_ids": strict_ids,
        "strict_is_gaia": np.concatenate([np.zeros(len(strict_table_ids), bool), strict_extra_gaia]),
        "shape_centers": shape_centers,
        "shape_values": shape_values,
        "shape_classes": shape_classes,
        "shape_ids": shape_ids,
    }
    if (labels.segmentation_ids is None) != (labels.segmentation_weight is None):
        raise ValueError("segmentation IDs and weights must be supplied together")
    if labels.segmentation_ids is not None:
        if (labels.segmentation_ids.shape != labels.dense.shape
                or labels.segmentation_weight.shape != labels.dense.shape):
            raise ValueError("segmentation/dense shape mismatch")
        target["segmentation_ids"] = _crop(labels.segmentation_ids, spec.x0, spec.y0, origin, size)
        target["segmentation_weight"] = _crop(labels.segmentation_weight, spec.x0, spec.y0, origin, size)
    if labels.segmentation_overlap_masks is not None:
        from .segmentation_storage import crop_overlap_masks
        target['segmentation_overlap_masks'] = crop_overlap_masks(
            labels.segmentation_overlap_masks, spec.x0-origin[0], spec.y0-origin[1], size)
    return target
