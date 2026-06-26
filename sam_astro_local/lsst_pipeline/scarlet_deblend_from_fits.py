from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

CELLECT_ROOT = Path(__file__).resolve().parents[2]
if str(CELLECT_ROOT) not in sys.path:
    sys.path.insert(0, str(CELLECT_ROOT))

try:
    from .pipeline_catalog_filters import (
        DEFAULT_DEBLEND_HARD_FAILURE_FLAGS,
        DEFAULT_MEASUREMENT_FLUX_COLUMN,
        catalog_summary as _catalog_summary,
        filter_deblend_hard_failures,
        filter_measurement_failures,
        measurement_failure_flags,
    )
except ImportError:
    from pipeline_catalog_filters import (
        DEFAULT_DEBLEND_HARD_FAILURE_FLAGS,
        DEFAULT_MEASUREMENT_FLUX_COLUMN,
        catalog_summary as _catalog_summary,
        filter_deblend_hard_failures,
        filter_measurement_failures,
        measurement_failure_flags,
    )


def _parse_band_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected BAND=/path/to/deepCoadd.fits")
    band, path = value.split("=", 1)
    band = band.strip()
    if not band:
        raise argparse.ArgumentTypeError("band name is empty")
    return band, Path(path).expanduser()


def _parse_patch(value: str) -> tuple[int, int]:
    pieces = [piece.strip() for piece in str(value).split(",")]
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("expected patch in 'x,y' form")
    return int(pieces[0]), int(pieces[1])


def _stable_int(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & ((1 << 63) - 1)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _persist_scarlet_model(output_path: Path, scarlet_model_data) -> dict[str, str]:
    """Persist scarlet model data across old and new LSST scarlet IO APIs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from lsst.meas.extensions.scarlet.io import write_scarlet_model
    except ImportError:
        write_scarlet_model = None

    if write_scarlet_model is not None:
        write_scarlet_model(str(output_path), scarlet_model_data)
        return {
            "scarlet_model_path": str(output_path),
            "scarlet_model_format": "lsst_meas_extensions_scarlet_zip",
        }

    pickle_path = output_path.with_suffix(".pickle")
    with pickle_path.open("wb") as handle:
        pickle.dump(scarlet_model_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "scarlet_model_path": str(pickle_path),
        "scarlet_model_format": "python_pickle_scarlet_model_data",
    }


def _load_sky_info(*, repo: Path, tract: int, patch: str):
    from lsst.pipe.base import Struct

    sky_map_path = repo / "deepCoadd" / "skyMap.pickle"
    if not sky_map_path.exists():
        raise FileNotFoundError(f"skyMap.pickle not found: {sky_map_path}")
    with sky_map_path.open("rb") as handle:
        sky_map = pickle.load(handle)
    tract_info = sky_map[int(tract)]
    patch_info = tract_info.getPatchInfo(_parse_patch(patch))
    return Struct(
        skyMap=sky_map,
        tractInfo=tract_info,
        patchInfo=patch_info,
        wcs=tract_info.getWcs(),
        bbox=patch_info.getOuterBBox(),
    )


def _filter_band_label(exposure) -> str:
    label = exposure.getInfo().getFilter()
    band_label = getattr(label, "bandLabel", None)
    if not band_label:
        raise RuntimeError("input exposure has no filter bandLabel")
    return str(band_label)


def _run_measurement_stage(
    *,
    deblended_catalog,
    scarlet_model_data,
    exposures_by_label: dict[str, Any],
    band_label_by_name: dict[str, str],
    ordered_band_names: list[str],
    sky_info,
    output_dir: Path,
    measurement_filter: str,
    measurement_failure_flag_names: list[str] | tuple[str, ...],
    measurement_flux_column: str,
    measurement_require_positive_flux: bool,
) -> dict[str, Any]:
    """Run per-band coadd measurement on scarlet deblended sources."""
    import lsst.afw.table as afwTable
    from lsst.meas.extensions.scarlet.io import updateCatalogFootprints
    from lsst.pipe.tasks.multiBand import MeasureMergedCoaddSourcesTask

    measure_root = output_dir / "measure"
    measure_root.mkdir(parents=True, exist_ok=True)

    measure_config = MeasureMergedCoaddSourcesTask.ConfigClass()
    measure_config.doAddFootprints = True
    measure_config.doConserveFlux = True
    measure_config.doPropagateFlags = False
    measure_config.checkUnitsParseStrict = "warn"
    measure_task = MeasureMergedCoaddSourcesTask(
        config=measure_config,
        initInputs={"inputSchema": deblended_catalog},
    )

    outputs: dict[str, Any] = {}
    for band_name in ordered_band_names:
        band_label = band_label_by_name[band_name]
        exposure = exposures_by_label[band_label]
        try:
            exposure.getPsf().setCacheCapacity(measure_config.psfCache)
        except Exception:
            pass

        table = afwTable.SourceTable.make(
            measure_task.schema,
            afwTable.IdFactory.makeSimple(),
        )
        sources = afwTable.SourceCatalog(table)
        sources.extend(deblended_catalog, measure_task.schemaMapper)

        updateCatalogFootprints(
            modelData=scarlet_model_data,
            catalog=sources,
            band=band_label,
            imageForRedistribution=exposure,
            removeScarletData=False,
            updateFluxColumns=True,
        )
        result = measure_task.run(
            exposure=exposure,
            sources=sources,
            skyInfo=sky_info,
            exposureId=_stable_int("measure", band_name),
        )

        band_root = measure_root / band_name
        band_root.mkdir(parents=True, exist_ok=True)
        meas_fits = band_root / "deepCoadd_meas.fits"
        raw_meas_fits = band_root / "deepCoadd_meas_raw.fits"
        output_sources = result.outputSources
        raw_summary = _catalog_summary(output_sources)
        filter_summary: dict[str, Any] = {
            "enabled": False,
            "policy": "none",
            "input_rows": int(len(output_sources)),
            "output_rows": int(len(output_sources)),
            "removed_rows": 0,
        }
        if measurement_filter != "none":
            output_sources.writeFits(str(raw_meas_fits))
            output_sources, filter_summary = filter_measurement_failures(
                output_sources,
                policy=measurement_filter,
                failure_flags=measurement_failure_flag_names,
                flux_column=measurement_flux_column,
                require_positive_flux=measurement_require_positive_flux,
            )
        output_sources.writeFits(str(meas_fits))
        outputs[band_name] = {
            "band_label": band_label,
            "meas_fits": str(meas_fits),
            "raw_meas_fits": str(raw_meas_fits) if measurement_filter != "none" else None,
            "raw_summary": raw_summary,
            "filter": filter_summary,
            "summary": _catalog_summary(output_sources),
        }

    return outputs


def _get_exposure_image_array(exposure):
    image = exposure.image if hasattr(exposure, "image") else exposure.getImage()
    return image.array


def _bbox_xyxy(bbox) -> list[int]:
    return [
        int(bbox.getMinX()),
        int(bbox.getMinY()),
        int(bbox.getMaxX()),
        int(bbox.getMaxY()),
    ]


def _load_sam_core():
    from sam_astro_local.scripts import amg_fits_core as sam_core

    return sam_core


def _sam_generator_args(config: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        model_type=config["model_type"],
        checkpoint=config["checkpoint"],
        scaling_mode="astro_rgb",
        astro_rgb_low_sigma=None,
        astro_preprocess_in_model=config["astro_preprocess_in_model"],
        astro_preprocess_clip_sigma=config["astro_preprocess_clip_sigma"],
        astro_preprocess_sigma_iters=config["astro_preprocess_sigma_iters"],
        astro_preprocess_z_clip=config["astro_preprocess_z_clip"],
        device=config["device"],
        points_per_side=config["points_per_side"],
        points_per_batch=config["points_per_batch"],
        pred_iou_thresh=config["pred_iou_thresh"],
        stability_score_thresh=config["stability_score_thresh"],
        box_nms_thresh=config["box_nms_thresh"],
        crop_n_layers=config["crop_n_layers"],
        crop_nms_thresh=config["crop_nms_thresh"],
        crop_overlap_ratio=config["crop_overlap_ratio"],
        crop_n_points_downscale_factor=config["crop_n_points_downscale_factor"],
        min_mask_region_area=config["min_mask_region_area"],
    )


def _validate_sam_triplet(exposures_by_label: dict[str, Any], ordered_labels: list[str]) -> None:
    first = exposures_by_label[ordered_labels[0]]
    first_shape = tuple(_get_exposure_image_array(first).shape)
    first_bbox = _bbox_xyxy(first.getBBox())
    for label in ordered_labels[1:]:
        exposure = exposures_by_label[label]
        shape = tuple(_get_exposure_image_array(exposure).shape)
        bbox = _bbox_xyxy(exposure.getBBox())
        if shape != first_shape or bbox != first_bbox:
            raise RuntimeError(
                "SAM detection requires all coadds to be on the same pixel grid; "
                f"{ordered_labels[0]} shape={first_shape} bbox={first_bbox}, "
                f"{label} shape={shape} bbox={bbox}"
            )


def _write_sam_label_map(path: Path, label_map, reference_fits: Path) -> None:
    from astropy.io import fits

    header = None
    with fits.open(reference_fits, memmap=False) as hdul:
        if "IMAGE" in hdul:
            header = hdul["IMAGE"].header.copy()
        elif len(hdul) > 1:
            header = hdul[1].header.copy()
        else:
            header = hdul[0].header.copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(path, label_map.astype("int32"), header=header, overwrite=True)


def _make_sam_label_map(
    *,
    arrays: list[Any],
    output_dir: Path,
    reference_fits: Path,
    stem: str,
    config: dict[str, Any],
):
    import numpy as np

    sam_core = _load_sam_core()
    r, g, b = [np.asarray(array, dtype=np.float32) for array in arrays]
    if r.shape != g.shape or r.shape != b.shape:
        raise RuntimeError(f"SAM input arrays must have identical shapes, got {[a.shape for a in (r, g, b)]}")

    crop_size = max(r.shape) if int(config["astro_crop_size"]) <= 0 else int(config["astro_crop_size"])
    astro_input = sam_core.build_astro_input(
        r,
        g,
        b,
        mode="none",
        stats_mode=config["astro_stats_mode"],
        low_sigma_override=None,
        crop_size=crop_size,
        low_pct=config["low_percentile"],
        high_pct=config["high_percentile"],
        preprocess_in_model=config["astro_preprocess_in_model"],
    )
    generator = sam_core.build_generator(_sam_generator_args(config), "none")
    masks = sam_core.run_generator(generator, astro_input.sam_input)
    sam_core.expand_crop_masks(masks, r.shape, astro_input.sam_input.shape[:2], astro_input.crop_y0, astro_input.crop_x0)
    masks, removed_small = sam_core.filter_small_masks(masks, r.shape[0], r.shape[1], config["min_mask_region_area"])
    masks, removed_large = sam_core.filter_large_masks(masks, r.shape[0], r.shape[1], config["max_mask_area_ratio"])
    label_map, masks, removed_label_small = sam_core.make_filtered_label_map(
        masks, r.shape[0], r.shape[1], config["min_mask_region_area"]
    )
    removed_small += removed_label_small

    sam_root = output_dir / "sam"
    label_map_fits = sam_root / f"{stem}_sam_labelmap.fits"
    metadata_csv = sam_root / f"{stem}_sam_metadata.csv"
    _write_sam_label_map(label_map_fits, label_map, reference_fits)
    sam_core.masks_to_csv(masks, metadata_csv)
    return label_map, {
        "label_map_fits": str(label_map_fits),
        "metadata_csv": str(metadata_csv),
        "kept_mask_count": int(len(masks)),
        "removed_small_count": int(removed_small),
        "removed_large_count": int(removed_large),
    }


def _combined_detection_image(arrays: list[Any]):
    import numpy as np

    stack = np.stack([np.nan_to_num(np.asarray(array, dtype=np.float32), nan=0.0) for array in arrays], axis=0)
    return np.sum(np.clip(stack, 0.0, None), axis=0)


def _robust_background_sigma(values):
    import numpy as np

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    background = float(np.median(finite))
    mad = float(np.median(np.abs(finite - background)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(finite))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return background, sigma


def _peaks_for_label(
    label_map,
    label: int,
    detection_image,
    *,
    max_peaks: int,
    threshold_sigma: float,
    min_distance: int,
    smooth_sigma: float,
):
    import numpy as np

    mask = label_map == label
    y, x = np.nonzero(mask)
    if y.size == 0:
        return []

    work = np.asarray(detection_image, dtype=np.float64)
    fill = float(np.nanmedian(work[np.isfinite(work)])) if np.any(np.isfinite(work)) else 0.0
    work = np.nan_to_num(work, nan=fill, posinf=fill, neginf=fill)
    if smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter

            work = gaussian_filter(work, sigma=float(smooth_sigma))
        except Exception:
            pass

    outside = work[~mask]
    background, sigma = _robust_background_sigma(outside if outside.size else work)
    threshold = background + float(threshold_sigma) * sigma
    masked = np.where(mask, work, -np.inf)

    try:
        from scipy.ndimage import maximum_filter

        size = max(3, 2 * int(min_distance) + 1)
        local_max = masked == maximum_filter(masked, size=size, mode="constant", cval=-np.inf)
        candidates = np.argwhere(local_max & mask & (masked >= threshold))
    except Exception:
        candidates = np.argwhere(mask & (masked >= threshold))

    if candidates.size == 0:
        values = masked[y, x]
        if np.all(~np.isfinite(values)):
            return [(int(round(float(np.mean(y)))), int(round(float(np.mean(x)))))]
        idx = int(np.argmax(values))
        return [(int(y[idx]), int(x[idx]))]

    values = masked[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(values)[::-1]
    peaks: list[tuple[int, int]] = []
    min_dist2 = max(0, int(min_distance)) ** 2
    for idx in order:
        py, px = int(candidates[idx, 0]), int(candidates[idx, 1])
        if all((py - old_y) ** 2 + (px - old_x) ** 2 >= min_dist2 for old_y, old_x in peaks):
            peaks.append((py, px))
        if len(peaks) >= int(max_peaks):
            break
    return peaks


def _source_catalog_peak_summary(catalog) -> dict[str, Any]:
    from collections import Counter

    peak_counts = []
    for record in catalog:
        footprint = record.getFootprint()
        peak_counts.append(len(footprint.getPeaks()) if footprint is not None else 0)
    hist = Counter(peak_counts)
    return {
        "total_peak_count": int(sum(peak_counts)),
        "peak_count_histogram": {str(key): int(value) for key, value in sorted(hist.items())},
    }


def _label_map_to_source_catalog(
    *,
    label_map,
    exposure,
    schema,
    detection_image,
    id_prefix: object,
    config: dict[str, Any],
):
    import numpy as np
    import lsst.afw.detection as afwDetect
    import lsst.afw.geom as afwGeom
    import lsst.afw.image as afwImage
    import lsst.afw.table as afwTable
    import lsst.geom as geom

    image_array = np.asarray(_get_exposure_image_array(exposure), dtype=np.float32)
    if tuple(label_map.shape) != tuple(image_array.shape):
        raise RuntimeError(f"SAM label map shape {label_map.shape} does not match exposure shape {image_array.shape}")

    bbox = exposure.getBBox()
    x0 = int(bbox.getMinX())
    y0 = int(bbox.getMinY())
    wcs = exposure.getWcs()
    if wcs is None:
        raise RuntimeError("SAM-to-LSST catalog conversion requires an exposure WCS")

    catalog = afwTable.SourceCatalog(schema)
    labels = [int(label) for label in np.unique(label_map) if int(label) > 0]
    work_mask = afwImage.Mask(bbox)
    for label in labels:
        peaks = _peaks_for_label(
            label_map,
            label,
            detection_image,
            max_peaks=config["max_peaks_per_mask"],
            threshold_sigma=config["peak_threshold_sigma"],
            min_distance=config["peak_min_distance"],
            smooth_sigma=config["peak_smooth_sigma"],
        )
        if not peaks:
            continue
        local_y, local_x = peaks[0]
        binary = label_map == label
        if not np.any(binary):
            continue

        work_mask.array[:, :] = binary.astype(work_mask.array.dtype)
        spans = afwGeom.SpanSet.fromMask(work_mask, 1)
        if spans.getArea() <= 0:
            continue

        global_x = x0 + int(local_x)
        global_y = y0 + int(local_y)
        peak_value = float(np.nan_to_num(image_array[local_y, local_x], nan=0.0, posinf=0.0, neginf=0.0))
        footprint = afwDetect.Footprint(spans)
        footprint.addPeak(float(global_x), float(global_y), peak_value)
        for local_y, local_x in peaks[1:]:
            global_x = x0 + int(local_x)
            global_y = y0 + int(local_y)
            peak_value = float(np.nan_to_num(image_array[local_y, local_x], nan=0.0, posinf=0.0, neginf=0.0))
            footprint.addPeak(float(global_x), float(global_y), peak_value)

        record = catalog.addNew()
        record.setId(_stable_int("sam", id_prefix, label))
        record.setParent(0)
        record.setCoord(wcs.pixelToSky(geom.Point2D(float(global_x), float(global_y))))
        record.setFootprint(footprint)

    return catalog


def _mark_detected_mask(exposure, catalog) -> None:
    mask = exposure.mask if hasattr(exposure, "mask") else exposure.getMaskedImage().getMask()
    detected = mask.getPlaneBitMask("DETECTED")
    for record in catalog:
        footprint = record.getFootprint()
        if footprint is not None:
            footprint.spans.setMask(mask, detected)


def _run_sam_detection(
    *,
    coadds: list[tuple[str, Path]],
    output_dir: Path,
    config: dict[str, Any],
):
    import lsst.afw.image as afwImage
    import lsst.afw.table as afwTable

    exposures_by_name = {band_name: afwImage.ExposureF(str(exposure_path)) for band_name, exposure_path in coadds}
    band_label_by_name = {band_name: _filter_band_label(exposure) for band_name, exposure in exposures_by_name.items()}
    ordered_labels = [band_label_by_name[band_name] for band_name, _ in coadds]
    exposures_by_label = {band_label_by_name[band_name]: exposure for band_name, exposure in exposures_by_name.items()}
    _validate_sam_triplet(exposures_by_label, ordered_labels)

    arrays_by_label = {label: _get_exposure_image_array(exposures_by_label[label]) for label in ordered_labels}
    reference_fits_by_label = {band_label_by_name[band_name]: exposure_path for band_name, exposure_path in coadds}
    schema = afwTable.SourceTable.makeMinimalSchema()
    detect_catalogs_by_label: dict[str, Any] = {}
    detect_outputs: dict[str, Any] = {}

    if config["sam_detection_scope"] == "multiband":
        arrays = [arrays_by_label[label] for label in ordered_labels]
        label_map, sam_summary = _make_sam_label_map(
            arrays=arrays,
            output_dir=output_dir,
            reference_fits=reference_fits_by_label[ordered_labels[0]],
            stem="multiband",
            config=config,
        )
        detection_image = _combined_detection_image(arrays)
        label_maps_by_label = {label: label_map for label in ordered_labels}
        detection_images_by_label = {label: detection_image for label in ordered_labels}
        sam_summaries_by_label = {label: sam_summary for label in ordered_labels}
    elif config["sam_detection_scope"] == "per-band":
        label_maps_by_label = {}
        detection_images_by_label = {}
        sam_summaries_by_label = {}
        for label in ordered_labels:
            array = arrays_by_label[label]
            label_map, sam_summary = _make_sam_label_map(
                arrays=[array, array, array],
                output_dir=output_dir,
                reference_fits=reference_fits_by_label[label],
                stem=f"band_{label}",
                config=config,
            )
            label_maps_by_label[label] = label_map
            detection_images_by_label[label] = array
            sam_summaries_by_label[label] = sam_summary
    else:
        raise RuntimeError(f"unknown SAM detection scope: {config['sam_detection_scope']}")

    for band_name, exposure_path in coadds:
        band_label = band_label_by_name[band_name]
        exposure = exposures_by_label[band_label]
        catalog = _label_map_to_source_catalog(
            label_map=label_maps_by_label[band_label],
            exposure=exposure,
            schema=schema,
            detection_image=detection_images_by_label[band_label],
            id_prefix=band_label,
            config=config,
        )
        _mark_detected_mask(exposure, catalog)

        band_root = output_dir / "detect" / band_name
        band_root.mkdir(parents=True, exist_ok=True)
        det_fits = band_root / "deepCoadd_det.fits"
        calexp_fits = band_root / "deepCoadd_calexp.fits"
        catalog.writeFits(str(det_fits))
        exposure.writeFits(str(calexp_fits))

        detect_catalogs_by_label[band_label] = catalog
        detect_outputs[band_name] = {
            "input_fits": str(exposure_path),
            "band_label": band_label,
            "det_fits": str(det_fits),
            "post_detect_calexp_fits": str(calexp_fits),
            "background_fits": None,
            "sam": sam_summaries_by_label[band_label],
            "peak_summary": _source_catalog_peak_summary(catalog),
            "summary": _catalog_summary(catalog),
        }

    return detect_catalogs_by_label, exposures_by_label, band_label_by_name, detect_outputs


def _clip_merge_sky_sources_to_bbox(merge_task, *, bbox):
    """Cutout demo helper: keep LSST sky-source injection inside a cutout bbox."""
    import lsst.afw.detection as afwDetect
    import lsst.afw.image as afwImage

    def _clipped(merged_list, sky_info, seed):
        mask = afwImage.Mask(bbox)
        detected = mask.getPlaneBitMask("DETECTED")
        for source in merged_list:
            source.getFootprint().spans.setMask(mask, detected)

        footprints = merge_task.skyObjects.run(mask, seed)
        if not footprints:
            return footprints

        schema = merge_task.merged.getPeakSchema()
        merge_key = schema.find(f"merge_peak_{merge_task.config.skyFilterName}").key
        converted = []
        for old_footprint in footprints:
            peak = old_footprint.getPeaks()[0]
            new_footprint = afwDetect.Footprint(old_footprint.spans, schema)
            new_footprint.addPeak(peak.getFx(), peak.getFy(), peak.getPeakValue())
            new_footprint.getPeaks()[0].set(merge_key, True)
            converted.append(new_footprint)
        return converted

    return _clipped


def run_demo(
    *,
    coadds: list[tuple[str, Path]],
    repo: Path,
    tract: int,
    patch: str,
    output_dir: Path,
    clip_sky_sources_to_exposure_bbox: bool,
    detection_mode: str,
    sam_config: dict[str, Any],
    deblend_filter: str,
    deblend_hard_failure_flags: list[str] | tuple[str, ...],
    measurement_filter: str,
    measurement_failure_flag_names: list[str] | tuple[str, ...],
    measurement_flux_column: str,
    measurement_require_positive_flux: bool,
) -> dict[str, Any]:
    try:
        import lsst.afw.image as afwImage
        import lsst.afw.table as afwTable
        from lsst.pipe.tasks.deblendCoaddSourcesPipeline import DeblendCoaddSourcesMultiTask
        from lsst.pipe.tasks.mergeDetections import MergeDetectionsTask
        from lsst.pipe.tasks.multiBand import DetectCoaddSourcesTask
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This demo requires the full LSST stack environment. "
            "Run it through ./run_demo_in_lsst_container.sh, or source the stack "
            "first (for example: source /opt/lsst/software/stack/loadLSST.bash && setup lsst_distrib)."
        ) from exc

    output_dir.mkdir(parents=True, exist_oks=True)
    sky_info = _load_sky_info(repo=repo, tract=tract, patch=patch)

    if detection_mode == "sam":
        detect_catalogs_by_label, detect_exposures_by_label, band_label_by_name, detect_outputs = _run_sam_detection(
            coadds=coadds,
            output_dir=output_dir,
            config=sam_config,
        )
    elif detection_mode == "lsst":
        detect_catalogs_by_label: dict[str, Any] = {}
        detect_exposures_by_label: dict[str, Any] = {}
        band_label_by_name: dict[str, str] = {}
        detect_outputs: dict[str, Any] = {}

        lsst_config = DetectCoaddSourcesTask.ConfigClass()
        # lsst_config.detection.thresholdValue = 8.0 # Default 5.0
        lsst_config.detection.minPixels = 15 # Accounting for seeing and resampling, default is 1
        # In lsst-scipipe-10.1.0, combinedGrow=False enters a Python loop over
        # FootprintSet, but that binding is not iterable.  Use no grow here:
        # it also avoids reconnecting nearby detections on denoised cutouts.
        # lsst_config.detection.nSigmaToGrow = 0.0 # Default 2.4
        # lsst_config.detection.returnOriginalFootprints = True

        for band_name, exposure_path in coadds:
            exposure = afwImage.ExposureF(str(exposure_path))
            detect_task = DetectCoaddSourcesTask(config=lsst_config)
            detect_result = detect_task.run(
                exposure=exposure,
                idFactory=afwTable.IdFactory.makeSimple(),
                expId=_stable_int("detect", band_name),
            )
            band_label = _filter_band_label(detect_result.outputExposure)
            if band_label in detect_catalogs_by_label:
                raise RuntimeError(f"duplicate filter bandLabel after detect: {band_label}")

            band_root = output_dir / "detect" / band_name
            band_root.mkdir(parents=True, exist_ok=True)
            det_fits = band_root / "deepCoadd_det.fits"
            calexp_fits = band_root / "deepCoadd_calexp.fits"
            background_fits = band_root / "deepCoadd_calexp_background.fits"
            detect_result.outputSources.writeFits(str(det_fits))
            detect_result.outputExposure.writeFits(str(calexp_fits))
            detect_result.outputBackgrounds.writeFits(str(background_fits))

            detect_catalogs_by_label[band_label] = detect_result.outputSources
            detect_exposures_by_label[band_label] = detect_result.outputExposure
            band_label_by_name[band_name] = band_label
            detect_outputs[band_name] = {
                "input_fits": str(exposure_path),
                "band_label": band_label,
                "det_fits": str(det_fits),
                "post_detect_calexp_fits": str(calexp_fits),
                "background_fits": str(background_fits),
                "summary": _catalog_summary(detect_result.outputSources),
            }
    else:
        raise RuntimeError(f"unknown detection mode: {detection_mode}")

    merge_root = output_dir / "merge"
    merge_root.mkdir(parents=True, exist_ok=True)
    ordered_labels = [band_label_by_name[band_name] for band_name, _ in coadds]
    first_label = ordered_labels[0]
    merge_config = MergeDetectionsTask.ConfigClass()
    merge_config.priorityList = list(ordered_labels)
    merge_task = MergeDetectionsTask(
        config=merge_config,
        schema=detect_catalogs_by_label[first_label].schema,
    )
    clipped_bbox = None
    if clip_sky_sources_to_exposure_bbox:
        bbox = detect_exposures_by_label[first_label].getBBox()
        clipped_bbox = [
            int(bbox.getMinX()),
            int(bbox.getMinY()),
            int(bbox.getMaxX()),
            int(bbox.getMaxY()),
        ]
        merge_task.getSkySourceFootprints = _clip_merge_sky_sources_to_bbox(merge_task, bbox=bbox)

    merge_result = merge_task.run(
        catalogs=detect_catalogs_by_label,
        skyInfo=sky_info,
        idFactory=afwTable.IdFactory.makeSimple(),
        skySeed=int(_stable_int("merge", tract, patch) & 0x7FFFFFFF),
    )
    merge_catalog = merge_result.outputCatalog
    merge_fits = merge_root / "deepCoadd_mergeDet.fits"
    peak_schema_fits = merge_root / "deepCoadd_peak_schema.fits"
    merge_catalog.writeFits(str(merge_fits))
    merge_task.outputPeakSchema.writeFits(str(peak_schema_fits))

    deblend_root = output_dir / "deblend"
    deblend_root.mkdir(parents=True, exist_ok=True)
    deblend_task = DeblendCoaddSourcesMultiTask(
        initInputs={
            "inputSchema": merge_catalog,
            "peakSchema": merge_task.outputPeakSchema,
        }
    )
    deblend_result = deblend_task.run(
        coadds=[detect_exposures_by_label[label] for label in ordered_labels],
        bands=ordered_labels,
        mergedDetections=merge_catalog,
        idFactory=afwTable.IdFactory.makeSimple(),
    )
    deblended_fits = deblend_root / "deepCoadd_deblendedFlux.fits"
    raw_deblended_fits = deblend_root / "deepCoadd_deblendedFlux_raw.fits"
    scarlet_model_zip = deblend_root / "deepCoadd_scarletModelData.zip"
    deblended_catalog = deblend_result.deblendedCatalog
    raw_deblend_summary = _catalog_summary(deblended_catalog)
    deblend_filter_summary: dict[str, Any] = {
        "enabled": False,
        "policy": "none",
        "input_rows": int(len(deblended_catalog)),
        "output_rows": int(len(deblended_catalog)),
        "removed_rows": 0,
    }
    if deblend_filter == "hard":
        deblended_catalog.writeFits(str(raw_deblended_fits))
        deblended_catalog, deblend_filter_summary = filter_deblend_hard_failures(
            deblended_catalog,
            hard_failure_flags=deblend_hard_failure_flags,
        )
    elif deblend_filter != "none":
        raise RuntimeError(f"unknown deblend filter policy: {deblend_filter}")

    deblended_catalog.writeFits(str(deblended_fits))
    scarlet_model_manifest = _persist_scarlet_model(scarlet_model_zip, deblend_result.scarletModelData)
    measure_outputs = _run_measurement_stage(
        deblended_catalog=deblended_catalog,
        scarlet_model_data=deblend_result.scarletModelData,
        exposures_by_label=detect_exposures_by_label,
        band_label_by_name=band_label_by_name,
        ordered_band_names=[band_name for band_name, _ in coadds],
        sky_info=sky_info,
        output_dir=output_dir,
        measurement_filter=measurement_filter,
        measurement_failure_flag_names=measurement_failure_flag_names,
        measurement_flux_column=measurement_flux_column,
        measurement_require_positive_flux=measurement_require_positive_flux,
    )

    manifest = {
        "inputs": {
            "repo": str(repo),
            "tract": int(tract),
            "patch": str(patch),
            "coadds": {band: str(path) for band, path in coadds},
        },
        "detection_mode": detection_mode,
        "sam_config": sam_config if detection_mode == "sam" else None,
        "band_labels": band_label_by_name,
        "clip_sky_sources_to_exposure_bbox": bool(clip_sky_sources_to_exposure_bbox),
        "clipped_sky_bbox_xyxy": clipped_bbox,
        "detect": detect_outputs,
        "merge": {
            "merge_det_fits": str(merge_fits),
            "peak_schema_fits": str(peak_schema_fits),
            "summary": _catalog_summary(merge_catalog),
        },
        "deblend": {
            "deblended_catalog_fits": str(deblended_fits),
            "raw_deblended_catalog_fits": str(raw_deblended_fits) if deblend_filter == "hard" else None,
            **scarlet_model_manifest,
            "raw_summary": raw_deblend_summary,
            "filter": deblend_filter_summary,
            "summary": _catalog_summary(deblended_catalog),
        },
        "measure": measure_outputs,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Minimal official LSST FITS-to-FITS Scarlet deblend demo."
    )
    parser.add_argument(
        "--coadd",
        action="append",
        required=True,
        type=_parse_band_arg,
        help="Input coadd exposure FITS, as BAND=/path/to/deepCoadd.fits. Repeat once per band.",
    )
    parser.add_argument("--repo", required=True, type=Path, help="Butler-style repo root containing deepCoadd/skyMap.pickle.")
    parser.add_argument("--tract", required=True, type=int)
    parser.add_argument("--patch", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--clip-sky-sources-to-exposure-bbox",
        action="store_true",
        help="Use for cutout FITS so merge-time sky-source footprints stay inside the cutout exposure bbox.",
    )
    parser.add_argument(
        "--detection-mode",
        choices=["lsst", "sam"],
        default="lsst",
        help="Use official LSST DetectCoaddSourcesTask or SAM-generated footprints before LSST merge/deblend.",
    )
    parser.add_argument(
        "--deblend-filter",
        choices=["none", "hard"],
        default="none",
        help=(
            "Optional post-scarlet deblend catalog filter before measurement. "
            "'hard' removes rows with hard deblend failure flags and descendants of failed parents."
        ),
    )
    parser.add_argument(
        "--deblend-hard-failure-flags",
        nargs="+",
        default=list(DEFAULT_DEBLEND_HARD_FAILURE_FLAGS),
        help=(
            "Flag fields removed when --deblend-filter hard is enabled. "
            "Defaults to the scarlet deblend hard-failure flags."
        ),
    )
    parser.add_argument(
        "--measurement-filter",
        choices=["none", "basic", "strict"],
        default="none",
        help=(
            "Optional post-measurement source filter. 'basic' removes clearly invalid "
            "leaf-source measurements; 'strict' also removes shape/blendedness/clipped-center failures."
        ),
    )
    parser.add_argument(
        "--measurement-filter-flags",
        nargs="+",
        default=None,
        help=(
            "Override the flag fields removed by --measurement-filter basic/strict. "
            "Only applies when --measurement-filter is not none."
        ),
    )
    parser.add_argument(
        "--measurement-flux-column",
        default=DEFAULT_MEASUREMENT_FLUX_COLUMN,
        help="Flux field used for finite/positive measurement filtering.",
    )
    parser.add_argument(
        "--measurement-require-positive-flux",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require --measurement-flux-column to be finite and positive when measurement filtering is enabled.",
    )
    parser.add_argument(
        "--sam-detection-scope",
        choices=["multiband", "per-band"],
        default="multiband",
        help="multiband runs SAM once on the coadd triplet; per-band repeats each band three times and merges band-specific masks.",
    )
    parser.add_argument("--sam-model-type", default="vit_h", choices=["default", "vit_h", "vit_l", "vit_b"])
    parser.add_argument("--sam-checkpoint", default="/home/chenzunhao/sam_vit_h_4b8939.pth")
    parser.add_argument("--sam-device", default="cuda")
    parser.add_argument("--sam-points-per-side", type=int, default=32)
    parser.add_argument("--sam-points-per-batch", type=int, default=128)
    parser.add_argument("--sam-pred-iou-thresh", type=float, default=0.8)
    parser.add_argument("--sam-stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--sam-box-nms-thresh", type=float, default=0.7)
    parser.add_argument("--sam-crop-n-layers", type=int, default=1)
    parser.add_argument("--sam-crop-nms-thresh", type=float, default=0.7)
    parser.add_argument("--sam-crop-overlap-ratio", type=float, default=512 / 1500)
    parser.add_argument("--sam-crop-n-points-downscale-factor", type=int, default=1)
    parser.add_argument("--sam-min-mask-region-area", type=int, default=15)
    parser.add_argument("--sam-max-mask-area-ratio", type=float, default=0.5)
    parser.add_argument("--sam-overlay-alpha", type=float, default=0.35)
    parser.add_argument("--sam-max-peaks-per-mask", type=int, default=8)
    parser.add_argument("--sam-peak-threshold-sigma", type=float, default=2.5)
    parser.add_argument("--sam-peak-min-distance", type=int, default=4)
    parser.add_argument("--sam-peak-smooth-sigma", type=float, default=1.0)
    parser.add_argument("--sam-astro-preprocess-in-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sam-astro-preprocess-clip-sigma", type=float, default=3.0)
    parser.add_argument("--sam-astro-preprocess-sigma-iters", type=int, default=-1)
    parser.add_argument("--sam-astro-preprocess-z-clip", type=float, nargs=2, default=[-3.0, 3.0])
    parser.add_argument("--sam-astro-stats-mode", default="sigmaclip", choices=["bgd", "sigmaclip"])
    parser.add_argument("--sam-astro-crop-size", type=int, default=0, help="0 means use the full frame.")
    parser.add_argument("--sam-low-percentile", type=float, default=0.1)
    parser.add_argument("--sam-high-percentile", type=float, default=99.5)
    args = parser.parse_args()

    sam_config = {
        "sam_detection_scope": args.sam_detection_scope,
        "model_type": args.sam_model_type,
        "checkpoint": args.sam_checkpoint,
        "device": args.sam_device,
        "points_per_side": int(args.sam_points_per_side),
        "points_per_batch": int(args.sam_points_per_batch),
        "pred_iou_thresh": float(args.sam_pred_iou_thresh),
        "stability_score_thresh": float(args.sam_stability_score_thresh),
        "box_nms_thresh": float(args.sam_box_nms_thresh),
        "crop_n_layers": int(args.sam_crop_n_layers),
        "crop_nms_thresh": float(args.sam_crop_nms_thresh),
        "crop_overlap_ratio": float(args.sam_crop_overlap_ratio),
        "crop_n_points_downscale_factor": int(args.sam_crop_n_points_downscale_factor),
        "min_mask_region_area": int(args.sam_min_mask_region_area),
        "max_mask_area_ratio": float(args.sam_max_mask_area_ratio),
        "overlay_alpha": float(args.sam_overlay_alpha),
        "max_peaks_per_mask": int(args.sam_max_peaks_per_mask),
        "peak_threshold_sigma": float(args.sam_peak_threshold_sigma),
        "peak_min_distance": int(args.sam_peak_min_distance),
        "peak_smooth_sigma": float(args.sam_peak_smooth_sigma),
        "astro_preprocess_in_model": bool(args.sam_astro_preprocess_in_model),
        "astro_preprocess_clip_sigma": float(args.sam_astro_preprocess_clip_sigma),
        "astro_preprocess_sigma_iters": int(args.sam_astro_preprocess_sigma_iters),
        "astro_preprocess_z_clip": [float(v) for v in args.sam_astro_preprocess_z_clip],
        "astro_stats_mode": args.sam_astro_stats_mode,
        "astro_crop_size": int(args.sam_astro_crop_size),
        "low_percentile": float(args.sam_low_percentile),
        "high_percentile": float(args.sam_high_percentile),
    }

    measurement_flags = measurement_failure_flags(args.measurement_filter, args.measurement_filter_flags)
    manifest = run_demo(
        coadds=args.coadd,
        repo=args.repo.expanduser(),
        tract=int(args.tract),
        patch=str(args.patch),
        output_dir=args.output_dir.expanduser(),
        clip_sky_sources_to_exposure_bbox=bool(args.clip_sky_sources_to_exposure_bbox),
        detection_mode=args.detection_mode,
        sam_config=sam_config,
        deblend_filter=args.deblend_filter,
        deblend_hard_failure_flags=list(args.deblend_hard_failure_flags),
        measurement_filter=args.measurement_filter,
        measurement_failure_flag_names=measurement_flags,
        measurement_flux_column=str(args.measurement_flux_column),
        measurement_require_positive_flux=bool(args.measurement_require_positive_flux),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
