from __future__ import annotations

import math
from typing import Any

DEFAULT_DEBLEND_HARD_FAILURE_FLAGS = (
    "deblend_failed",
    "deblend_skipped",
    "deblend_tooManyPeaks",
    "deblend_parentTooBig",
    "deblend_masked",
    "deblend_incompleteData",
    "deblend_zeroFlux",
    # "deblend_blendConvergenceFailedFlag"
)

DEFAULT_MEASUREMENT_BASIC_FAILURE_FLAGS = (
    "base_PsfFlux_flag",
    # "base_PsfFlux_flag_noGoodPixels",
    # "base_PsfFlux_flag_edge",
    # "base_SdssCentroid_flag",
    # "base_PixelFlags_flag_edge",
    # "base_PixelFlags_flag_bad",
    # "base_PixelFlags_flag_saturatedCenter",
    # "base_PixelFlags_flag_interpolatedCenter",
    # "base_PixelFlags_flag_crCenter",
)

DEFAULT_MEASUREMENT_STRICT_EXTRA_FLAGS = (
    "base_SdssShape_flag",
    "base_PixelFlags_flag_clippedCenter",
    "base_Blendedness_flag_noCentroid",
    "base_Blendedness_flag_noShape",
)

DEFAULT_MEASUREMENT_FLUX_COLUMN = "base_PsfFlux_instFlux"


def catalog_summary(catalog) -> dict[str, int]:
    parent = catalog["parent"] if "parent" in catalog.schema else []
    if len(parent) == 0:
        return {"row_count": int(len(catalog)), "parent_zero_count": 0, "child_count": 0}

    is_sky = [_is_sky_source(record) for record in catalog]
    science = [not value for value in is_sky]
    positive_flux = [record_int(record, "parent", 0) == 0 and _is_flux_positive(record, DEFAULT_MEASUREMENT_FLUX_COLUMN) for record in catalog]
    return {
        "row_count": int(sum(science)),
        "sky_count": int(sum(is_sky)),
        "parent_zero_count": int(sum(int(p) == 0 and keep for p, keep in zip(parent, science))),
        "positive_flux_parent_zero_count": int(sum(int(p) == 0 and keep and flux for p, keep, flux in zip(parent, science, positive_flux))),
        "child_count": int(sum(int(p) != 0 and keep for p, keep in zip(parent, science))),
    }


def catalog_has_field(catalog, name: str) -> bool:
    try:
        catalog.schema.find(name)
    except Exception:
        return False
    return True


def record_bool(record, name: str) -> bool:
    try:
        return bool(record[name])
    except Exception:
        return False


def record_int(record, name: str, default: int = 0) -> int:
    try:
        return int(record[name])
    except Exception:
        return int(default)


def record_float(record, name: str, default: float = math.nan) -> float:
    try:
        return float(record[name])
    except Exception:
        return float(default)


def _source_catalog_like(catalog):
    import lsst.afw.table as afwTable

    table = catalog.table if hasattr(catalog, "table") else catalog.getTable()
    return afwTable.SourceCatalog(table)


def _is_sky_source(record) -> bool:
    return record_bool(record, "merge_footprint_sky") or record_bool(record, "sky_source")


def _is_structural_parent(record) -> bool:
    return record_int(record, "parent", 0) == 0 and record_int(record, "deblend_nChild", 0) > 0

def _is_flux_positive(record, flux_column: str) -> bool:
    try:
        flux = float(record[flux_column])
        return math.isfinite(flux) and flux > 0
    except Exception:
        return False


def _kept_child_counts(catalog, remove_ids: set[int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for record in catalog:
        source_id = int(record.getId())
        if source_id in remove_ids:
            continue
        parent = record_int(record, "parent", 0)
        if parent != 0:
            counts[parent] = counts.get(parent, 0) + 1
    return counts


def _remove_empty_parents(catalog, remove_ids: set[int], kept_child_counts: dict[int, int]) -> int:
    removed = 0
    for record in catalog:
        source_id = int(record.getId())
        if source_id in remove_ids:
            continue
        if _is_structural_parent(record) and kept_child_counts.get(source_id, 0) == 0:
            remove_ids.add(source_id)
            removed += 1
    return removed


def _append_filtered_records(catalog, remove_ids: set[int]):
    filtered = _source_catalog_like(catalog)
    kept_child_counts = _kept_child_counts(catalog, remove_ids)
    for record in catalog:
        source_id = int(record.getId())
        if source_id in remove_ids:
            continue
        parent = record_int(record, "parent", 0)
        if parent == 0 and catalog_has_field(catalog, "deblend_nChild"):
            n_child = record_int(record, "deblend_nChild", 0)
            if n_child > 0:
                record["deblend_nChild"] = kept_child_counts.get(source_id, 0)
        elif parent != 0 and catalog_has_field(catalog, "deblend_parentNChild"):
            record["deblend_parentNChild"] = kept_child_counts.get(parent, 0)
        filtered.append(record)
    return filtered


def filter_deblend_hard_failures(
    catalog,
    *,
    hard_failure_flags: list[str] | tuple[str, ...],
) -> tuple[Any, dict[str, Any]]:
    """Drop scarlet deblend rows that are certainly unusable."""
    available_flags = [name for name in hard_failure_flags if catalog_has_field(catalog, name)]
    missing_flags = [name for name in hard_failure_flags if name not in available_flags]

    hard_failed_ids: set[int] = set()
    flag_counts = {name: 0 for name in available_flags}
    for record in catalog:
        record_failed = False
        for name in available_flags:
            if record_bool(record, name):
                flag_counts[name] += 1
                record_failed = True
        if record_failed:
            hard_failed_ids.add(int(record.getId()))

    parent_failed_ids = {
        int(record.getId())
        for record in catalog
        if int(record.getId()) in hard_failed_ids and record_int(record, "parent", 0) == 0
    }

    remove_ids = set(hard_failed_ids)
    descendant_removed = 0
    for record in catalog:
        parent = record_int(record, "parent", 0)
        if parent in parent_failed_ids and int(record.getId()) not in remove_ids:
            remove_ids.add(int(record.getId()))
            descendant_removed += 1

    kept_child_counts = _kept_child_counts(catalog, remove_ids)
    empty_parent_removed = _remove_empty_parents(catalog, remove_ids, kept_child_counts)
    filtered = _append_filtered_records(catalog, remove_ids)

    return filtered, {
        "enabled": True,
        "policy": "hard",
        "input_rows": int(len(catalog)),
        "output_rows": int(len(filtered)),
        "removed_rows": int(len(catalog) - len(filtered)),
        "hard_failed_rows": int(len(hard_failed_ids)),
        "descendants_removed_from_failed_parents": int(descendant_removed),
        "empty_parents_removed": int(empty_parent_removed),
        "available_flags": available_flags,
        "missing_flags": missing_flags,
        "flag_counts": flag_counts,
    }


def measurement_failure_flags(policy: str, override: list[str] | tuple[str, ...] | None = None) -> list[str]:
    if override is not None:
        return list(override)
    if policy == "basic":
        return list(DEFAULT_MEASUREMENT_BASIC_FAILURE_FLAGS)
    if policy == "strict":
        return list(DEFAULT_MEASUREMENT_BASIC_FAILURE_FLAGS + DEFAULT_MEASUREMENT_STRICT_EXTRA_FLAGS)
    if policy == "none":
        return []
    raise ValueError(f"unknown measurement filter policy: {policy}")


def filter_measurement_failures(
    catalog,
    *,
    policy: str,
    failure_flags: list[str] | tuple[str, ...],
    flux_column: str = DEFAULT_MEASUREMENT_FLUX_COLUMN,
    require_positive_flux: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Filter measurement rows with clearly invalid source measurements.

    Quality cuts are applied to science leaf rows. Structural parent rows are
    kept unless all of their children are removed, because child rows still use
    parent ids to preserve blend family relationships.
    """
    available_flags = [name for name in failure_flags if catalog_has_field(catalog, name)]
    missing_flags = [name for name in failure_flags if name not in available_flags]
    has_flux = catalog_has_field(catalog, flux_column)

    remove_ids: set[int] = set()
    flagged_ids: set[int] = set()
    invalid_centroid_ids: set[int] = set()
    invalid_flux_ids: set[int] = set()
    sky_ids: set[int] = set()
    flag_counts = {name: 0 for name in available_flags}

    for record in catalog:
        source_id = int(record.getId())
        if _is_sky_source(record):
            remove_ids.add(source_id)
            sky_ids.add(source_id)
            continue

        if _is_structural_parent(record):
            continue
        
        record_failed = False
        for name in available_flags:
            if record_bool(record, name):
                flag_counts[name] += 1
                record_failed = True
        if record_failed:
            remove_ids.add(source_id)
            flagged_ids.add(source_id)

        try:
            x = float(record.getX())
            y = float(record.getY())
        except Exception:
            x = math.nan
            y = math.nan
        if not (math.isfinite(x) and math.isfinite(y)):
            remove_ids.add(source_id)
            invalid_centroid_ids.add(source_id)

        if has_flux:
            flux = record_float(record, flux_column)
            if require_positive_flux:
                flux_is_valid = _is_flux_positive(record, flux_column)
            else:
                flux_is_valid = math.isfinite(flux)
            if not flux_is_valid:
                remove_ids.add(source_id)
                invalid_flux_ids.add(source_id)

    removed_parent_ids = {
        int(record.getId())
        for record in catalog
        if int(record.getId()) in remove_ids and record_int(record, "parent", 0) == 0
    }
    descendant_removed = 0
    for record in catalog:
        parent = record_int(record, "parent", 0)
        if parent in removed_parent_ids and int(record.getId()) not in remove_ids:
            remove_ids.add(int(record.getId()))
            descendant_removed += 1

    kept_child_counts = _kept_child_counts(catalog, remove_ids)
    empty_parent_removed = _remove_empty_parents(catalog, remove_ids, kept_child_counts)
    filtered = _append_filtered_records(catalog, remove_ids)

    return filtered, {
        "enabled": True,
        "policy": policy,
        "input_rows": int(len(catalog)),
        "output_rows": int(len(filtered)),
        "removed_rows": int(len(catalog) - len(filtered)),
        "sky_rows_removed": int(len(sky_ids)),
        "flagged_rows_removed": int(len(flagged_ids)),
        "invalid_centroid_rows_removed": int(len(invalid_centroid_ids)),
        "invalid_flux_rows_removed": int(len(invalid_flux_ids)),
        "descendants_removed_from_removed_parents": int(descendant_removed),
        "empty_parents_removed": int(empty_parent_removed),
        "flux_column": flux_column,
        "flux_column_found": bool(has_flux),
        "require_positive_flux": bool(require_positive_flux),
        "available_flags": available_flags,
        "missing_flags": missing_flags,
        "flag_counts": flag_counts,
    }
