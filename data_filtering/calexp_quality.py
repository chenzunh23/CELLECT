"""Calexp mask-plane quality scoring used by preprocessing and diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from astropy.io import fits


DEFAULT_BAD_PLANES = (
    "BAD",
    "INTRP",
    "EDGE",
    "NO_DATA",
    "UNMASKEDNAN",
)

DEFAULT_BAD_SCORE_WEIGHTS = {
    "NO_DATA": 1.0,
    "UNMASKEDNAN": 1.0,
    "INTRP": 0.3,
    "EDGE": 0.7,
    "BAD": 0.5,
}


def mask_planes_from_header(header: fits.Header) -> dict[str, int]:
    planes: dict[str, int] = {}
    for key, value in header.items():
        if str(key).upper().startswith("MP_"):
            try:
                planes[str(key)[3:].upper()] = int(value)
            except Exception:
                continue
    return planes


def parse_score_weights(values: Iterable[str] | None) -> dict[str, float]:
    weights = dict(DEFAULT_BAD_SCORE_WEIGHTS)
    if not values:
        return weights
    for item in values:
        name, sep, raw_value = str(item).partition("=")
        if not sep:
            raise ValueError(f"invalid bad-score weight {item!r}, expected PLANE=WEIGHT")
        weights[name.strip().upper()] = float(raw_value)
    return weights


def normalize_band_dir(band: str) -> str:
    text = str(band).strip().upper()
    if text.startswith("HSC-") or text.startswith("NB"):
        return text
    return f"HSC-{text}"


def parse_patches(values: Iterable[str]) -> list[str]:
    patches: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in str(value).replace(";", " ").split():
            patch = item.strip()
            if not patch:
                continue
            expanded = [f"{x},{y}" for x in range(9) for y in range(9)] if patch.lower() == "all" else [patch]
            for candidate in expanded:
                if candidate not in seen:
                    patches.append(candidate)
                    seen.add(candidate)
    return patches


def find_calexp(patch_dir: Path) -> Path | None:
    files = sorted(Path(patch_dir).glob("calexp-*.fits")) + sorted(Path(patch_dir).glob("calexp-*.fits.gz"))
    return files[0] if files else None


def read_calexp(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    with fits.open(path, memmap=False, ignore_missing_end=True) as hdul:
        image = np.asarray(hdul[1].data, dtype=np.float32)
        if "MASK" in hdul:
            mask_hdu = hdul["MASK"]
        else:
            mask_hdu = hdul[2]
        mask = np.asarray(mask_hdu.data, dtype=np.int64)
        planes = mask_planes_from_header(mask_hdu.header)
    return image, mask, planes


def read_mask_plane(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    with fits.open(path, memmap=False, ignore_missing_end=True) as hdul:
        if "MASK" in hdul:
            hdu = hdul["MASK"]
        elif len(hdul) > 2 and getattr(hdul[2], "data", None) is not None:
            hdu = hdul[2]
        else:
            raise KeyError(f"no LSST MASK plane found in {path}")
        return np.asarray(hdu.data, dtype=np.int64), mask_planes_from_header(hdu.header)


def read_mask_plane_for_score(path: Path) -> tuple[np.ndarray, dict[str, int], set[str]]:
    with fits.open(path, memmap=False, ignore_missing_end=True) as hdul:
        image = np.asarray(hdul[1].data, dtype=np.float32)
        if "MASK" in hdul:
            hdu = hdul["MASK"]
        elif len(hdul) > 2 and getattr(hdul[2], "data", None) is not None:
            hdu = hdul[2]
        else:
            raise KeyError(f"no LSST MASK plane found in {path}")
        mask = np.asarray(hdu.data, dtype=np.int64)
        planes = mask_planes_from_header(hdu.header)
    ignored: set[str] = set()
    bad_bit = planes.get("BAD")
    if bad_bit is not None and mask.shape == image.shape:
        bad = (mask & (1 << int(bad_bit))) != 0
        if bool(bad.all()) and bool(np.isfinite(image).all()):
            ignored.add("BAD")
    return mask, planes, ignored


def bad_score_map(
    mask: np.ndarray,
    planes: dict[str, int],
    weights: dict[str, float],
    *,
    ignored_planes: Iterable[str] | None = None,
) -> np.ndarray:
    score = np.zeros(np.asarray(mask).shape, dtype=np.float32)
    mask_values = np.asarray(mask, dtype=np.int64)
    ignored = {str(name).upper() for name in (ignored_planes or ())}
    for name, weight in weights.items():
        plane = str(name).upper()
        if plane in ignored:
            continue
        bit = planes.get(plane)
        if bit is None or float(weight) <= 0.0:
            continue
        plane_mask = (mask_values & (1 << int(bit))) != 0
        if bool(plane_mask.any()):
            score = np.maximum(score, plane_mask.astype(np.float32) * float(weight))
    return score


def bad_score_fraction(
    mask: np.ndarray,
    planes: dict[str, int],
    weights: dict[str, float],
    *,
    ignored_planes: Iterable[str] | None = None,
) -> float:
    score = bad_score_map(mask, planes, weights, ignored_planes=ignored_planes)
    return float(np.nanmean(score)) if score.size else 0.0


def tile_bad_score(
    mask_tile: np.ndarray,
    planes: dict[str, int],
    weights: dict[str, float],
    *,
    ignored_planes: Iterable[str] | None = None,
) -> float:
    if np.asarray(mask_tile).size == 0:
        return 0.0
    return bad_score_fraction(mask_tile, planes, weights, ignored_planes=ignored_planes)


def tile_score_category(score: float) -> str:
    value = float(score)
    if value >= 0.5:
        return "gt_50"
    if value >= 0.2:
        return "20_50"
    if value >= 0.1:
        return "10_20"
    if value >= 0.05:
        return "5_10"
    return "lt_5"


def score_regular_tiles(
    mask: np.ndarray,
    planes: dict[str, int],
    weights: dict[str, float],
    *,
    tile_size: int,
    stride: int,
) -> list[dict[str, object]]:
    h, w = np.asarray(mask).shape
    rows: list[dict[str, object]] = []
    if h < tile_size or w < tile_size:
        return rows
    for y0 in range(0, h - int(tile_size) + 1, int(stride)):
        for x0 in range(0, w - int(tile_size) + 1, int(stride)):
            x1 = int(x0) + int(tile_size)
            y1 = int(y0) + int(tile_size)
            score = tile_bad_score(mask[y0:y1, x0:x1], planes, weights)
            rows.append(
                {
                    "x0": int(x0),
                    "y0": int(y0),
                    "x1": int(x1),
                    "y1": int(y1),
                    "score": score,
                    "score_percent": score * 100.0,
                    "category": tile_score_category(score),
                }
            )
    return rows


def score_tile_grid(
    mask: np.ndarray,
    planes: dict[str, int],
    weights: dict[str, float],
    *,
    starts: Sequence[tuple[int, int]],
    tile_size: int,
    ignored_planes: Iterable[str] | None = None,
) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    h, w = np.asarray(mask).shape
    for x0, y0 in starts:
        x1 = int(x0) + int(tile_size)
        y1 = int(y0) + int(tile_size)
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            continue
        out[(int(x0), int(y0))] = tile_bad_score(
            mask[y0:y1, x0:x1],
            planes,
            weights,
            ignored_planes=ignored_planes,
        )
    return out


def patch_and_tile_scores(
    calexp_path: Path,
    *,
    starts: Sequence[tuple[int, int]],
    tile_size: int,
    weights: dict[str, float],
) -> tuple[float, dict[tuple[int, int], float]]:
    mask, planes, ignored_planes = read_mask_plane_for_score(calexp_path)
    return (
        bad_score_fraction(mask, planes, weights, ignored_planes=ignored_planes),
        score_tile_grid(mask, planes, weights, starts=starts, tile_size=tile_size, ignored_planes=ignored_planes),
    )
