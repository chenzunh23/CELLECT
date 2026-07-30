#!/usr/bin/env python3
"""Count bright ClassificationExtendedness-unknown sources on SAT/BAD/EDGE masks."""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    from astropy.io import fits
    from astropy.table import Table
    from astropy.units import UnitsWarning
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires astropy.") from exc

warnings.filterwarnings("ignore", category=UnitsWarning)
warnings.filterwarnings("ignore", message="Warning: converting a masked element to nan.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/data/shared/Subaru"))
    parser.add_argument("--refit-root", type=Path, default=Path("/data/czh23/refit"))
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--bands", nargs="+", default=["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"])
    parser.add_argument("--mag-threshold", type=float, default=22.0)
    parser.add_argument("--zeropoint", type=float, default=27.0)
    parser.add_argument("--radius-column", default="proxy_nan0_flux_aperture_radius")
    parser.add_argument("--good-column", default="proxy_nan0_good")
    parser.add_argument("--mask-names", nargs="+", default=["SAT", "BAD", "EDGE"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("output/data_filter_0727/star_galaxy_maglt22/unknown_sat_bad_edge_summary.csv"),
    )
    return parser.parse_args()


def finite_float(value: object, default: float = float("nan")) -> float:
    if np.ma.is_masked(value):
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "t", "yes", "y"}


def mag_from_flux(flux: float, zeropoint: float) -> float:
    if not math.isfinite(flux) or flux <= 0.0:
        return float("nan")
    return float(zeropoint) - 2.5 * math.log10(float(flux))


def origin_from_ltv(header: fits.Header) -> tuple[int, int]:
    return -int(round(float(header.get("LTV1", 0.0)))), -int(round(float(header.get("LTV2", 0.0))))


def mask_bit_map(header: fits.Header, mask_names: list[str]) -> dict[str, int]:
    bits: dict[str, int] = {}
    for name in mask_names:
        key = f"MP_{name}"
        if key in header:
            bits[name] = int(header[key])
    return bits


def meas_path(data_root: Path, tract: str, band: str, patch: str) -> Path:
    return data_root / str(tract) / band / patch / f"meas-{band}-{tract}-{patch}.fits"


def calexp_path(data_root: Path, tract: str, band: str, patch: str) -> Path:
    return data_root / str(tract) / band / patch / f"calexp-{band}-{tract}-{patch}.fits"


def count_one(
    *,
    data_root: Path,
    refit_csv: Path,
    tract: str,
    band: str,
    patch: str,
    mag_threshold: float,
    zeropoint: float,
    radius_column: str,
    good_column: str,
    mask_names: list[str],
) -> dict[str, object]:
    row: dict[str, object] = {"band": band, "patch": patch, "status": "ok"}
    cat_path = meas_path(data_root, tract, band, patch)
    exp_path = calexp_path(data_root, tract, band, patch)
    missing = [str(path) for path in (cat_path, exp_path, refit_csv) if not path.exists()]
    if missing:
        row.update({"status": "missing", "missing": ";".join(missing)})
        return row

    meas = Table.read(cat_path, hdu=1)
    if "base_ClassificationExtendedness_value" not in meas.colnames:
        row.update({"status": "missing_classification", "missing": "base_ClassificationExtendedness_value"})
        return row

    counts: dict[str, int] = defaultdict(int)
    union_key = "_".join(mask_names) + "_union"
    unknown_centers: list[tuple[float, float]] = []
    with refit_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for csv_row in reader:
            counts["refit_rows"] += 1
            flux = finite_float(csv_row.get("catalog_KronFlux_instFlux"))
            mag = mag_from_flux(flux, zeropoint)
            if not math.isfinite(mag) or mag >= mag_threshold:
                continue
            counts["maglt22"] += 1
            if csv_row.get("status", "") != "ok":
                counts["bad_refit_status"] += 1
                continue
            if good_column in csv_row and not is_true(csv_row.get(good_column)):
                counts["bad_refit_good_column"] += 1
                continue
            if radius_column not in csv_row:
                counts["missing_radius_column"] += 1
                continue
            row_index = int(round(finite_float(csv_row.get("row_index"), -1.0)))
            if row_index < 0 or row_index >= len(meas):
                counts["bad_row_index"] += 1
                continue
            ext = finite_float(meas["base_ClassificationExtendedness_value"][row_index])
            if math.isfinite(ext):
                continue
            counts["unknown"] += 1
            x = finite_float(csv_row.get("x_image"))
            y = finite_float(csv_row.get("y_image"))
            if math.isfinite(x) and math.isfinite(y):
                unknown_centers.append((x, y))
            else:
                counts["bad_unknown_center"] += 1
    if unknown_centers:
        with fits.open(exp_path, memmap=True) as hdul:
            mask_hdu = hdul[2]
            mask = np.asarray(mask_hdu.data)
            bits = mask_bit_map(mask_hdu.header, mask_names)
        if not bits:
            row.update({"status": "missing_mask_bits", "missing": ",".join(mask_names)})
            return row
        for x, y in unknown_centers:
            local_x = int(round(x))
            local_y = int(round(y))
            if not (0 <= local_x < mask.shape[1] and 0 <= local_y < mask.shape[0]):
                counts["center_out_of_bounds"] += 1
                continue
            mask_value = int(mask[local_y, local_x])
            hit_any = False
            for name, bit in bits.items():
                if mask_value & (1 << int(bit)):
                    counts[name] += 1
                    hit_any = True
            if hit_any:
                counts[union_key] += 1
            else:
                counts["none_of_" + "_".join(mask_names)] += 1
    row.update(counts)
    row["union_key"] = union_key
    return row


def main() -> int:
    args = parse_args()
    tasks: list[dict[str, object]] = []
    for band in args.bands:
        band_refit_root = args.refit_root / str(args.tract) / band
        for refit_csv in sorted(band_refit_root.glob("*/batch_heavyfp_kron_refit/batch_heavyfp_kron_refit.csv")):
            patch = refit_csv.parts[-3]
            tasks.append(
                {
                    "data_root": args.data_root,
                    "refit_csv": refit_csv,
                    "tract": str(args.tract),
                    "band": band,
                    "patch": patch,
                    "mag_threshold": float(args.mag_threshold),
                    "zeropoint": float(args.zeropoint),
                    "radius_column": args.radius_column,
                    "good_column": args.good_column,
                    "mask_names": list(args.mask_names),
                }
            )
    rows: list[dict[str, object]] = []
    if int(args.workers) > 1:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            future_to_task = {executor.submit(count_one, **task): task for task in tasks}
            completed = 0
            for future in as_completed(future_to_task):
                rows.append(future.result())
                completed += 1
                if completed % 25 == 0 or completed == len(tasks):
                    print(f"processed {completed}/{len(tasks)} patch-band jobs", flush=True)
    else:
        for idx, task in enumerate(tasks, start=1):
            rows.append(count_one(**task))
            if idx % 25 == 0 or idx == len(tasks):
                print(f"processed {idx}/{len(tasks)} patch-band jobs", flush=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    union_key = "_".join(args.mask_names) + "_union"
    print(f"wrote {args.out_csv}")
    print("band maglt22 unknown " + " ".join(args.mask_names) + f" {union_key} none out_of_bounds")
    grand: dict[str, int] = defaultdict(int)
    for band in args.bands:
        selected = [row for row in rows if row.get("band") == band and row.get("status") == "ok"]
        totals: dict[str, int] = defaultdict(int)
        for row in selected:
            for key in ["maglt22", "unknown", union_key, "none_of_" + "_".join(args.mask_names), "center_out_of_bounds", *args.mask_names]:
                totals[key] += int(row.get(key, 0) or 0)
        for key, value in totals.items():
            grand[key] += value
        print(
            band,
            totals["maglt22"],
            totals["unknown"],
            *(totals[name] for name in args.mask_names),
            totals[union_key],
            totals["none_of_" + "_".join(args.mask_names)],
            totals["center_out_of_bounds"],
        )
    print(
        "ALL",
        grand["maglt22"],
        grand["unknown"],
        *(grand[name] for name in args.mask_names),
        grand[union_key],
        grand["none_of_" + "_".join(args.mask_names)],
        grand["center_out_of_bounds"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
