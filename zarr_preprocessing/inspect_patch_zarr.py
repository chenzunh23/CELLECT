#!/usr/bin/env python
"""Inspect a CELLECT patch Zarr store without importing zarr."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def decode_fixed_utf8(path: Path, shape: tuple[int, int], chunks: tuple[int, int], limit: int) -> list[str]:
    chunk = path / "0.0"
    if not chunk.exists():
        return []
    raw = np.frombuffer(chunk.read_bytes(), dtype=np.uint8).reshape(chunks)
    raw = raw[: shape[0], : shape[1]]
    out = []
    for row in raw[:limit]:
        out.append(bytes(row).split(b"\0", 1)[0].decode("utf-8", errors="replace"))
    return out


def array_summary(array_dir: Path) -> dict:
    meta = read_json(array_dir / ".zarray")
    chunk_files = [path for path in array_dir.iterdir() if not path.name.startswith(".")]
    total_bytes = sum(path.stat().st_size for path in chunk_files)
    return {
        "name": array_dir.name,
        "shape": meta["shape"],
        "chunks": meta["chunks"],
        "dtype": meta["dtype"],
        "num_chunks": len(chunk_files),
        "chunk_bytes": total_bytes,
    }


def inspect_store(root: Path, *, sample_limit: int) -> dict:
    attrs = read_json(root / ".zattrs") if (root / ".zattrs").exists() else {}
    arrays = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / ".zarray").exists():
            arrays.append(array_summary(child))
    samples = {}
    for name in ("tile_name", "group", "dataset_source"):
        path = root / name
        if not (path / ".zarray").exists():
            continue
        meta = read_json(path / ".zarray")
        samples[name] = decode_fixed_utf8(path, tuple(meta["shape"]), tuple(meta["chunks"]), sample_limit)
    return {"path": str(root), "attrs": attrs, "arrays": arrays, "samples": samples}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_root", type=Path)
    parser.add_argument("--sample-limit", type=int, default=5)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = inspect_store(args.zarr_root.expanduser().resolve(), sample_limit=args.sample_limit)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
