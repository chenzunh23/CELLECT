#!/usr/bin/env python3
"""Compatibility wrapper for the PU source-filtering CLI.

The implementation lives in ``data_filtering.pu_source_filter``.
"""

from data_filtering.pu_source_filter import *  # noqa: F401,F403
from data_filtering.pu_source_filter import main


if __name__ == "__main__":
    main()
