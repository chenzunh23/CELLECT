"""Main orchestration for preprocessing v3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.table import Table

from .bright_ap2 import BrightAp2Config, classify_bright_ap2
from .bright_label import BrightLabelConfig, label_bright_sources
from .image_processing import BrightRegionConfig, ImageProcessingConfig, build_bright_components, read_background_mask, read_fits_image, read_quality_mask, scale_image_for_training
from .labels import SourceClass, SourceLabels
from .meas_processing import MeasProcessingConfig, classify_meas_basics
from .ordinary import OrdinaryConfig, label_ordinary_sources
from .refit import DirectRefitConfig, RefitConfig, attach_refit_geometry, attach_refit_radius_from_table, compute_kron_ellipse, run_refit_from_meas
from .region_filling import fill_dense_regions
from .snr import SnrConfig, compute_snr_for_sample
from .utils.catalog import source_ids
from .zarr_writing import ZarrSampleBatch, write_image_level_zarr


@dataclass(frozen=True)
class PipelineConfig:
    refit: RefitConfig = RefitConfig()
    direct_refit: DirectRefitConfig = DirectRefitConfig()
    run_direct_refit_if_missing: bool = True
    meas: MeasProcessingConfig = MeasProcessingConfig()
    ordinary: OrdinaryConfig = OrdinaryConfig()
    snr: SnrConfig = SnrConfig()
    bright_ap2: BrightAp2Config = BrightAp2Config()
    bright: BrightLabelConfig = BrightLabelConfig()
    image: ImageProcessingConfig = ImageProcessingConfig()
    bright_region: BrightRegionConfig = BrightRegionConfig()
    overwrite: bool = False
    write_training_source_arrays: bool = True
    write_diagnostic_source_rows: bool = False


@dataclass(frozen=True)
class PatchBandInputs:
    meas_catalog: Path
    image_fits: Path
    output_zarr: Path
    refit_csv: Path | None = None
    background_npz: Path | None = None
    quality_mask_npz: Path | None = None
    gaia_catalog: Path | None = None
    sample_name: str | None = None
    is_narrow_band: bool = False
    band: str | None = None
    patch: str | None = None
    dataset_source: str = "coadd"
    group: str | int | None = None
    coadd_image_fits: Path | None = None


class PreprocessingPipeline:
    """Small, explicit orchestrator for one patch-band-image source."""

    def __init__(self, config: PipelineConfig = PipelineConfig()) -> None:
        self.config = config

    def _training_source_arrays(self, table: Table, labels: SourceLabels, bright_result) -> dict[str, np.ndarray]:
        """Build zarr source arrays using the v3 training-label vocabulary only."""

        geom = compute_kron_ellipse(table, self.config.refit)
        ids = source_ids(table)
        finite = np.isfinite(geom.x) & np.isfinite(geom.y)
        center_mask = finite & (
            labels.mask(SourceClass.CLEAN)
            | labels.mask(SourceClass.WEAK_SHAPE)
        )
        shape_mask = center_mask & np.isfinite(geom.major) & np.isfinite(geom.minor) & np.isfinite(geom.theta)
        strict_table_mask = finite & labels.mask(SourceClass.STRICT_CENTER_ONLY)

        centers = np.column_stack([geom.x[center_mask], geom.y[center_mask]]).astype(np.float32)
        center_ids = np.asarray(ids[center_mask], dtype=np.int64)
        source_offsets = np.asarray([[0, len(centers)]], dtype=np.int64)

        strict_centers_parts = []
        strict_ids_parts = []
        if np.any(strict_table_mask):
            strict_centers_parts.append(np.column_stack([geom.x[strict_table_mask], geom.y[strict_table_mask]]).astype(np.float32))
            strict_ids_parts.append(np.asarray(ids[strict_table_mask], dtype=np.int64))
        if getattr(bright_result, "strict_center_x", np.asarray([])).size:
            strict_centers_parts.append(
                np.column_stack([bright_result.strict_center_x, bright_result.strict_center_y]).astype(np.float32)
            )
            strict_ids_parts.append(np.asarray(bright_result.strict_center_source_id, dtype=np.int64))
        strict_centers = (
            np.concatenate(strict_centers_parts, axis=0).astype(np.float32)
            if strict_centers_parts
            else np.empty((0, 2), dtype=np.float32)
        )
        strict_ids = (
            np.concatenate(strict_ids_parts, axis=0).astype(np.int64)
            if strict_ids_parts
            else np.empty((0,), dtype=np.int64)
        )
        strict_offsets = np.asarray([[0, len(strict_centers)]], dtype=np.int64)

        shape_centers = np.column_stack([geom.x[shape_mask], geom.y[shape_mask]]).astype(np.float32)
        shape_values = np.column_stack([geom.major[shape_mask], geom.minor[shape_mask], geom.theta[shape_mask]]).astype(np.float32)
        shape_classes = labels.source_class[shape_mask].astype(np.uint8, copy=False)
        shape_ids = np.asarray(ids[shape_mask], dtype=np.int64)
        shape_offsets = np.asarray([[0, len(shape_centers)]], dtype=np.int64)
        return {
            "source_centers": centers,
            "source_ids": center_ids,
            "source_offsets": source_offsets,
            "strict_center_only_centers": strict_centers,
            "strict_center_only_ids": strict_ids,
            "strict_center_only_offsets": strict_offsets,
            "shape_source_centers": shape_centers,
            "shape_source_values": shape_values,
            "shape_source_classes": shape_classes,
            "shape_source_ids": shape_ids,
            "shape_source_offsets": shape_offsets,
        }

    def run_patch_band(self, inputs: PatchBandInputs) -> Path:
        table = Table.read(inputs.meas_catalog)
        if inputs.refit_csv is not None:
            table = attach_refit_geometry(table, inputs.refit_csv, self.config.refit)
        elif self.config.run_direct_refit_if_missing:
            refit_table = run_refit_from_meas(
                inputs.meas_catalog,
                inputs.image_fits,
                config=self.config.direct_refit,
            )
            table = attach_refit_radius_from_table(table, refit_table, self.config.refit)
            geom = compute_kron_ellipse(table, self.config.refit)
            table["v3_x"] = geom.x.astype(np.float64)
            table["v3_y"] = geom.y.astype(np.float64)
            table["v3_major"] = geom.major.astype(np.float32)
            table["v3_minor"] = geom.minor.astype(np.float32)
            table["v3_theta"] = geom.theta.astype(np.float32)
            table["v3_area"] = geom.area.astype(np.float32)
        else:
            table = attach_refit_geometry(table, None, self.config.refit)

        image, image_header = read_fits_image(inputs.image_fits, hdu=self.config.image.hdu)
        scaled = scale_image_for_training(image, config=self.config.image)
        if scaled.ndim == 2:
            scaled = scaled[None, ...]
        elif scaled.ndim == 3 and scaled.shape[-1] in (1, 3):
            scaled = np.moveaxis(scaled, -1, 0)

        stage = classify_meas_basics(table, config=self.config.meas, refit_config=self.config.refit)
        bright_mask, bright_components = build_bright_components(image, config=self.config.bright_region)
        quality_ignore = read_quality_mask(inputs.quality_mask_npz, image.shape)
        snr_result = compute_snr_for_sample(
            table,
            dataset_source=inputs.dataset_source,
            is_narrow_band=inputs.is_narrow_band,
            band=inputs.band,
            patch=inputs.patch,
            group=inputs.group,
            image_fits=inputs.image_fits,
            coadd_image_fits=inputs.coadd_image_fits,
            config=self.config.snr,
        )
        ordinary = label_ordinary_sources(
            table,
            stage.ordinary_candidate,
            stage.labels,
            is_narrow_band=inputs.is_narrow_band,
            snr=snr_result.snr if snr_result is not None else None,
            config=self.config.ordinary,
            snr_config=self.config.snr,
            refit_config=self.config.refit,
        )
        stage.labels = ordinary.labels
        bright_ap2 = classify_bright_ap2(
            table,
            stage.bright_candidate,
            stage.labels,
            component_labels=bright_components,
            config=self.config.bright_ap2,
            refit_config=self.config.refit,
        )
        gaia = Table.read(inputs.gaia_catalog) if inputs.gaia_catalog is not None else None
        bright = label_bright_sources(
            table,
            bright_ap2.candidate,
            bright_ap2.labels,
            bright_region=bright_mask,
            component_labels=bright_components,
            gaia_table=gaia,
            image_header=image_header,
            quality_mask=quality_ignore,
            mag=stage.mag,
            config=self.config.bright,
            refit_config=self.config.refit,
        )

        background = read_background_mask(inputs.background_npz, image.shape)
        restricted_fallback_mask = None
        if bright.restricted_fallback_component_ids.size and bright_components is not None and np.asarray(bright_components).size:
            component_ids = np.asarray(bright.restricted_fallback_component_ids, dtype=np.int32)
            component_ids = component_ids[component_ids > 0]
            if component_ids.size:
                restricted_fallback_mask = np.isin(np.asarray(bright_components, dtype=np.int32), component_ids)
        dense = fill_dense_regions(
            table,
            bright.labels,
            image.shape,
            background_mask=background,
            quality_ignore_mask=quality_ignore,
            restricted_fallback_mask=restricted_fallback_mask,
            refit_config=self.config.refit,
        )
        sample_name = inputs.sample_name or inputs.image_fits.stem
        batch = ZarrSampleBatch(
            images=scaled[None, ...].astype(np.float32),
            dense_labels=dense[None, ...].astype(np.uint8),
            names=[sample_name],
            attrs={
                "sample_name": sample_name,
                "meas_catalog": str(inputs.meas_catalog),
                "image_fits": str(inputs.image_fits),
                "refit_csv": str(inputs.refit_csv) if inputs.refit_csv is not None else "",
                "source_export_mode": "training_v3",
                "dataset_source": str(inputs.dataset_source),
                "band": str(inputs.band or ""),
                "patch": str(inputs.patch or ""),
                "snr_method": str(snr_result.method if snr_result is not None else "none"),
            },
            **(self._training_source_arrays(table, bright.labels, bright) if self.config.write_training_source_arrays else {}),
            diagnostic_source_rows=bright.source_rows if self.config.write_diagnostic_source_rows else None,
        )
        return write_image_level_zarr(inputs.output_zarr, batch, overwrite=self.config.overwrite)
