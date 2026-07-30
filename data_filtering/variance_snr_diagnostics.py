#!/usr/bin/env python3
"""Run the variance-scaled SNR diagnostic without building training products.

This is a thin entry point for the current diagnostic implementation in
``scripts/export_legacy_noisy_variance_snr_regs.py``.  It exports REG/CSV
diagnostics only; it does not create target NPZ, PT caches, or zarr stores.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from scripts.export_legacy_noisy_variance_snr_regs import main as diagnostic_main

    return int(diagnostic_main())


if __name__ == "__main__":
    raise SystemExit(main())
