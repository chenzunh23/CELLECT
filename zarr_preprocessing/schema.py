#!/usr/bin/env python
"""Shared schema and path helpers for CELLECT patch Zarr stores."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

DEFAULT_BANDS = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y")
DEFAULT_TILE_SIZE = 512


@dataclass(frozen=True)
class TileRecord:
    name: str
    x0: int
    y0: int
    size: int
    group: str
    base_tile_name: str


def parse_tile_origin(name: str) -> tuple[int, int]:
    match = re.search(r"_x(-?\d+)_y(-?\d+)", str(name))
    if match is None:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def infer_group(tile_name: str) -> str:
    match = re.match(r"^(group_\d+)_", str(tile_name))
    return match.group(1) if match else ""


def strip_group(tile_name: str) -> str:
    return re.sub(r"^group_\d+_", "", str(tile_name))


def read_manifest(patch_root: Path) -> dict:
    path = patch_root / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_tiles(patch_root: Path) -> list[TileRecord]:
    path = patch_root / "tiles.csv"
    if not path.exists():
        raise FileNotFoundError(f"tiles.csv not found: {path}")
    out: list[TileRecord] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            base = str(row.get("base_tile_name", "")).strip() or strip_group(name)
            x_text = str(row.get("x0", "")).strip()
            y_text = str(row.get("y0", "")).strip()
            s_text = str(row.get("size", "")).strip()
            if x_text and y_text:
                x0, y0 = int(float(x_text)), int(float(y_text))
            else:
                x0, y0 = parse_tile_origin(base)
            size = int(float(s_text)) if s_text else DEFAULT_TILE_SIZE
            group = str(row.get("variant_group", "")).strip() or infer_group(name)
            out.append(TileRecord(name=name, x0=x0, y0=y0, size=size, group=group, base_tile_name=base))
    if not out:
        raise RuntimeError(f"No tile records found in {path}")
    return out


def infer_dataset_source(patch_root: Path, explicit: str = "auto") -> str:
    if explicit and explicit != "auto":
        return explicit
    manifest = read_manifest(patch_root)
    if manifest.get("dataset_source"):
        return str(manifest["dataset_source"])
    parts = patch_root.parts
    for item in ("denoised", "noisy", "coadd"):
        if item in parts:
            return item
    return "coadd"


def infer_tract_patch(patch_root: Path, tract: str = "", patch: str = "") -> tuple[str, str]:
    if tract and patch:
        return str(tract), str(patch)
    manifest = read_manifest(patch_root)
    out_tract = str(tract or manifest.get("tract", ""))
    out_patch = str(patch or manifest.get("patch", ""))
    if out_tract and out_patch:
        return out_tract, out_patch
    parts = patch_root.parts
    for idx in range(len(parts) - 1):
        if parts[idx].isdigit() and "," in parts[idx + 1]:
            return str(parts[idx]), str(parts[idx + 1])
    return out_tract, out_patch


def zscale_cache_path(
    zscale_root: Path,
    *,
    dataset_source: str,
    tract: str,
    patch: str,
    tile_name: str,
    bands: Sequence[str],
    fits_hdu: int,
) -> Path:
    band_key = "_".join(bands)
    if dataset_source and dataset_source != "coadd":
        return zscale_root / dataset_source / tract / patch / "cutouts" / f"{tile_name}__{band_key}__hdu{fits_hdu}.pt"
    return zscale_root / tract / patch / "cutouts" / f"{tile_name}__{band_key}__hdu{fits_hdu}.pt"


def band_target_path(patch_root: Path, band: str, tile_name: str) -> Path:
    return patch_root / "band_targets" / band / f"{tile_name}.npz"


def band_metadata_path(patch_root: Path, band: str, tile_name: str) -> Path:
    return patch_root / "band_tile_metadata" / band / f"{tile_name}.npz"

