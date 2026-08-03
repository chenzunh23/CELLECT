"""Catalog helpers shared by preprocessing v3."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from astropy.table import Table


POSITION_COLUMNS_X = (
    "base_SdssCentroid_x",
    "base_SdssShape_x",
    "base_NaiveCentroid_x",
    "deblend_psfCenter_x",
)
POSITION_COLUMNS_Y = (
    "base_SdssCentroid_y",
    "base_SdssShape_y",
    "base_NaiveCentroid_y",
    "deblend_psfCenter_y",
)


def first_finite_column(table: Table, names: Sequence[str], *, default: float = np.nan) -> np.ndarray:
    out = np.full(len(table), default, dtype=np.float64)
    for name in names:
        if name not in table.colnames:
            continue
        values = np.asarray(table[name], dtype=np.float64)
        take = ~np.isfinite(out) & np.isfinite(values)
        out[take] = values[take]
        if np.isfinite(out).all():
            break
    return out


def source_ids(table: Table) -> np.ndarray:
    if "id" in table.colnames:
        return np.asarray(table["id"], dtype=np.int64)
    if "source_id" in table.colnames:
        return np.asarray(table["source_id"], dtype=np.int64)
    raise KeyError("catalog must contain id or source_id")


def source_xy(table: Table) -> tuple[np.ndarray, np.ndarray]:
    return first_finite_column(table, POSITION_COLUMNS_X), first_finite_column(table, POSITION_COLUMNS_Y)


def magnitude_from_flux(table: Table, *, column: str, zeropoint: float) -> np.ndarray:
    if column not in table.colnames:
        return np.full(len(table), np.nan, dtype=np.float64)
    flux = np.asarray(table[column], dtype=np.float64)
    mag = np.full(len(table), np.nan, dtype=np.float64)
    valid = np.isfinite(flux) & (flux > 0.0)
    mag[valid] = float(zeropoint) - 2.5 * np.log10(flux[valid])
    return mag


def source_filter_mask(table: Table, source_filter: str) -> np.ndarray:
    n = len(table)
    if source_filter == "all":
        return np.ones(n, dtype=bool)
    parent = np.asarray(table["parent"], dtype=np.int64) if "parent" in table.colnames else np.zeros(n, dtype=np.int64)
    child_col = "deblend_nChild" if "deblend_nChild" in table.colnames else "nChild" if "nChild" in table.colnames else None
    if source_filter == "parent":
        return parent == 0
    if child_col is None:
        raise KeyError(f"source_filter={source_filter!r} needs deblend_nChild or nChild")
    leaf = np.asarray(table[child_col], dtype=np.int64) == 0
    if source_filter == "nchild0":
        return leaf
    if source_filter == "leaf_child":
        return leaf & (parent != 0)
    raise ValueError(f"unknown source_filter: {source_filter}")


def boolean_column(table: Table, name: str, *, default: bool = False) -> np.ndarray:
    if name not in table.colnames:
        return np.full(len(table), bool(default), dtype=bool)
    return np.asarray(table[name], dtype=bool)

