#!/usr/bin/env python3
"""Compatibility wrapper for ``data_filtering.analyze_calexp_mask_quality``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_filtering.analyze_calexp_mask_quality import *  # noqa: F401,F403
from data_filtering.analyze_calexp_mask_quality import main


if __name__ == "__main__":
    raise SystemExit(main())
