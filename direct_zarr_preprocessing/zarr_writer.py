#!/usr/bin/env python
"""Small Zarr v2 writer used by direct preprocessing."""

from __future__ import annotations

import json
import shutil
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.dtype):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")


def encode_fixed_utf8(values: Sequence[str], width: int) -> np.ndarray:
    out = np.zeros((len(values), width), dtype=np.uint8)
    for i, value in enumerate(values):
        raw = str(value).encode("utf-8")[:width]
        out[i, : len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return out


class ZarrArrayWriter:
    def __init__(self, root: Path, name: str, *, shape: Sequence[int], chunks: Sequence[int], dtype) -> None:
        self.path = root / name
        self.name = name
        self.shape = tuple(int(v) for v in shape)
        self.chunks = tuple(max(1, int(v)) for v in chunks)
        self.dtype = np.dtype(dtype)
        self.path.mkdir(parents=True, exist_ok=True)
        write_json(
            self.path / ".zarray",
            {
                "zarr_format": 2,
                "shape": list(self.shape),
                "chunks": list(self.chunks),
                "dtype": self.dtype.str,
                "compressor": None,
                "filters": None,
                "fill_value": 0,
                "order": "C",
            },
        )
        write_json(self.path / ".zattrs", {})

    def write_chunk(self, chunk_index: Sequence[int], data: np.ndarray) -> None:
        arr = np.asarray(data, dtype=self.dtype)
        (self.path / ".".join(str(int(v)) for v in chunk_index)).write_bytes(arr.tobytes(order="C"))

    def write_full(self, data: np.ndarray) -> None:
        arr = np.asarray(data, dtype=self.dtype)
        if arr.shape != self.shape:
            raise ValueError(f"{self.name}: expected shape {self.shape}, got {arr.shape}")
        if 0 in self.shape:
            return
        ranges = [range((dim + chunk - 1) // chunk) for dim, chunk in zip(self.shape, self.chunks)]
        for chunk_index in product(*ranges):
            src = []
            for axis, ci in enumerate(chunk_index):
                start = ci * self.chunks[axis]
                end = min(self.shape[axis], start + self.chunks[axis])
                src.append(slice(start, end))
            self.write_chunk(chunk_index, arr[tuple(src)])


class ZarrGroupWriter:
    def __init__(self, root: Path, *, overwrite: bool, attrs: dict) -> None:
        self.root = root.expanduser().resolve()
        if self.root.exists():
            if not overwrite:
                raise FileExistsError(f"Output exists; use --overwrite: {self.root}")
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.root / ".zgroup", {"zarr_format": 2})
        write_json(self.root / ".zattrs", attrs)

    def array(self, name: str, *, shape: Sequence[int], chunks: Sequence[int], dtype) -> ZarrArrayWriter:
        return ZarrArrayWriter(self.root, name, shape=shape, chunks=chunks, dtype=dtype)
