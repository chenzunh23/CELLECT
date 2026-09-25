#!/usr/bin/env python3
"""Download Gaia sources around HSC tract 9813.

The original POLYGON/CONTAINS query can trigger Gaia TAP server-side 500
errors. A rectangular RA/Dec query is enough here and is more robust.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from astroquery.gaia import Gaia


HSC_9813_RA_MIN = 3.0 # 149.40687637
HSC_9813_RA_MAX = 4.0 # 151.08894468
HSC_9813_DEC_MIN = -31.0 # 1.39136227
HSC_9813_DEC_MAX = -30.0 # 3.07134110

TABLES = {
    "dr2": "gaiadr2.gaia_source",
    "dr3": "gaiadr3.gaia_source",
}

COMMON_COLUMNS = [
    "source_id",
    "ra",
    "dec",
    "parallax",
    "parallax_error",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
]

EXTRA_COLUMNS = {
    "dr2": [
        "astrometric_excess_noise",
        "phot_bp_rp_excess_factor",
    ],
    "dr3": [
        "ruwe",
    ],
}


def build_query(args: argparse.Namespace) -> str:
    ra_min = float(args.ra_min) - float(args.margin_deg)
    ra_max = float(args.ra_max) + float(args.margin_deg)
    dec_min = float(args.dec_min) - float(args.margin_deg)
    dec_max = float(args.dec_max) + float(args.margin_deg)
    table = TABLES[str(args.release)]
    columns = COMMON_COLUMNS + EXTRA_COLUMNS[str(args.release)]
    select_list = ",\n    ".join(columns)
    return f"""
SELECT
    {select_list}
FROM {table}
WHERE ra BETWEEN {ra_min:.8f} AND {ra_max:.8f}
  AND dec BETWEEN {dec_min:.8f} AND {dec_max:.8f}
"""


def run_query(query: str, *, mode: str, retries: int, sleep_s: float):
    last_error: Exception | None = None
    for attempt in range(1, int(retries) + 1):
        try:
            print(f"[gaia] launch {mode} query attempt {attempt}/{retries}", flush=True)
            if mode == "async":
                job = Gaia.launch_job_async(query)
            else:
                job = Gaia.launch_job(query)
            return job.get_results()
        except Exception as exc:  # Gaia TAP often returns transient HTTP 500.
            last_error = exc
            print(f"[gaia] attempt {attempt} failed: {exc}", flush=True)
            if attempt < retries:
                time.sleep(float(sleep_s))
    assert last_error is not None
    raise last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", choices=sorted(TABLES), default="dr2")
    parser.add_argument("--ra-min", type=float, default=HSC_9813_RA_MIN)
    parser.add_argument("--ra-max", type=float, default=HSC_9813_RA_MAX)
    parser.add_argument("--dec-min", type=float, default=HSC_9813_DEC_MIN)
    parser.add_argument("--dec-max", type=float, default=HSC_9813_DEC_MAX)
    parser.add_argument(
        "--margin-deg",
        type=float,
        default=0.05,
        help="Expand each side of the HSC 9813 bounding box; 0.05 deg is useful for bright-star masks.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=["async", "sync"],
        default="async",
        help="Use async by default; sync Gaia TAP jobs can silently cap results at 2000 rows.",
    )
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    Gaia.ROW_LIMIT = -1
    output = args.output
    if output is None:
        output = Path(f"output/gaia_{args.release}_cosmos.fits")
    query = build_query(args)
    print(query.strip(), flush=True)
    if args.dry_run:
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    table = run_query(query, mode=str(args.mode), retries=int(args.retries), sleep_s=float(args.retry_sleep))
    print(f"[gaia] rows={len(table)}", flush=True)
    table.write(output, overwrite=True)
    print(f"[gaia] wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
