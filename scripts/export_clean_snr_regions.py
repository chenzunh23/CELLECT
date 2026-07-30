#!/usr/bin/env python3
"""Export clean-source DS9 regions colored by official AP2 SNR."""

from __future__ import annotations

import argparse
import csv
import math
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--classes-csv", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--catalog-hdu", type=int, default=1)
    p.add_argument("--ap2-flux-column", default="base_CircularApertureFlux_6_0_instFlux")
    p.add_argument("--ap2-err-column", default="base_CircularApertureFlux_6_0_instFluxErr")
    p.add_argument("--kron-flux-column", default="ext_photometryKron_KronFlux_instFlux")
    p.add_argument("--kron-err-column", default="ext_photometryKron_KronFlux_instFluxErr")
    p.add_argument("--large-area-as-point", type=float, default=10000.0)
    p.add_argument("--hist-max-percentile", type=float, default=99.0)
    return p.parse_args()


def _snr(flux: np.ndarray, err: np.ndarray) -> np.ndarray:
    out = np.full(len(flux), np.nan, dtype=np.float64)
    ok = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    out[ok] = flux[ok] / err[ok]
    return out


def _region_header(title: str) -> list[str]:
    return [
        "# Region file format: DS9 version 4.1",
        f"# {title}",
        'global color=green dashlist=8 3 width=2 font="helvetica 10 normal roman" '
        "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
        "physical",
    ]


def _snr_color(ap2_snr: float) -> str:
    if not np.isfinite(ap2_snr):
        return "red"
    if ap2_snr <= 3.0:
        return "blue"
    if ap2_snr < 5.0:
        return "cyan"
    return "green"


def _ellipse_or_point(row: dict[str, str], *, color: str, large_area_as_point: float) -> str:
    x = float(row["x_physical"])
    y = float(row["y_physical"])
    area = float(row.get("aperture_area") or "nan")
    sid = row.get("id", "")
    ap2_snr = float(row.get("ap2_snr", "nan"))
    snr_label = "nan" if not np.isfinite(ap2_snr) else f"{ap2_snr:.2f}"
    comment = f"# color={color} width=2 tag={{id={sid}}} text={{AP2 SNR={snr_label}}}"
    if np.isfinite(area) and area > large_area_as_point:
        return f"point({x:.6f},{y:.6f}) # point=circle color={color} width=2 tag={{id={sid}}} text={{AP2 SNR={snr_label}}}"
    major = float(row["major_aperture"])
    minor = float(row["minor_aperture"])
    theta = float(row["theta_deg"])
    return f"ellipse({x:.6f},{y:.6f},{major:.6f},{minor:.6f},{theta:.6f}) {comment}"


def _read_clean_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("class") == "clean"]


def _table_id_map(table: Table) -> dict[int, int]:
    if "id" not in table.colnames:
        raise KeyError("catalog must contain id column")
    return {int(value): idx for idx, value in enumerate(np.asarray(table["id"]))}


def _finite_summary(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0}
    qs = np.quantile(finite, [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1])
    return {
        "n": int(finite.size),
        "min": float(qs[0]),
        "p25": float(qs[1]),
        "median": float(qs[2]),
        "p75": float(qs[3]),
        "p90": float(qs[4]),
        "p99": float(qs[5]),
        "max": float(qs[6]),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*'second' did not parse as fits unit.*")
        catalog = Table.read(args.catalog, hdu=int(args.catalog_hdu))
    id_to_idx = _table_id_map(catalog)
    clean_rows = _read_clean_rows(args.classes_csv)

    ap2_snr_all = _snr(
        np.asarray(catalog[args.ap2_flux_column], dtype=np.float64),
        np.asarray(catalog[args.ap2_err_column], dtype=np.float64),
    )
    kron_snr_all = _snr(
        np.asarray(catalog[args.kron_flux_column], dtype=np.float64),
        np.asarray(catalog[args.kron_err_column], dtype=np.float64),
    )

    enriched: list[dict[str, object]] = []
    region_lines = _region_header(f"{args.prefix} clean sources colored by AP2 SNR")
    for row in clean_rows:
        sid = int(row["id"])
        idx = id_to_idx.get(sid)
        if idx is None:
            ap2_snr = math.nan
            kron_snr = math.nan
        else:
            ap2_snr = float(ap2_snr_all[idx])
            kron_snr = float(kron_snr_all[idx])
        color = _snr_color(ap2_snr)
        enriched.append(
            {
                **row,
                "ap2_snr": ap2_snr,
                "kron_snr": kron_snr,
                "snr_color": color,
            }
        )
        region_lines.append(_ellipse_or_point(enriched[-1], color=color, large_area_as_point=args.large_area_as_point))

    reg_path = args.output_dir / f"{args.prefix}_clean_ap2_snr_colored.reg"
    reg_path.write_text("\n".join(region_lines) + "\n")

    csv_path = args.output_dir / f"{args.prefix}_clean_ap2_kron_snr.csv"
    if enriched:
        fieldnames = list(enriched[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(enriched)
    else:
        csv_path.write_text("")

    ap2 = np.asarray([float(row["ap2_snr"]) for row in enriched], dtype=np.float64)
    kron = np.asarray([float(row["kron_snr"]) for row in enriched], dtype=np.float64)
    finite_for_range = np.concatenate([ap2[np.isfinite(ap2)], kron[np.isfinite(kron)]])
    hist_max = 10.0
    if finite_for_range.size:
        hist_max = max(10.0, float(np.nanpercentile(finite_for_range, args.hist_max_percentile)))
    bins = np.linspace(0.0, hist_max, 80)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(ap2[np.isfinite(ap2)], bins=bins, histtype="step", linewidth=2.0, label="official AP2 SNR")
    ax.hist(kron[np.isfinite(kron)], bins=bins, histtype="step", linewidth=2.0, label="official Kron SNR")
    ax.axvline(3.0, color="blue", linestyle="--", linewidth=1.5, label="SNR=3")
    ax.axvline(5.0, color="green", linestyle="--", linewidth=1.5, label="SNR=5")
    ax.set_xlabel("SNR")
    ax.set_ylabel("Clean source count")
    ax.set_title(f"{args.prefix} clean-source SNR")
    ax.legend()
    fig.savefig(args.output_dir / f"{args.prefix}_clean_snr_hist.png", dpi=180)
    plt.close(fig)

    zoom_bins = np.linspace(0.0, 20.0, 81)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(ap2[np.isfinite(ap2)], bins=zoom_bins, histtype="step", linewidth=2.0, label="official AP2 SNR")
    ax.hist(kron[np.isfinite(kron)], bins=zoom_bins, histtype="step", linewidth=2.0, label="official Kron SNR")
    ax.axvline(3.0, color="blue", linestyle="--", linewidth=1.5, label="SNR=3")
    ax.axvline(5.0, color="green", linestyle="--", linewidth=1.5, label="SNR=5")
    ax.set_xlim(0.0, 20.0)
    ax.set_xlabel("SNR")
    ax.set_ylabel("Clean source count")
    ax.set_title(f"{args.prefix} clean-source SNR, zoom 0-20")
    ax.legend()
    fig.savefig(args.output_dir / f"{args.prefix}_clean_snr_hist_zoom_0_20.png", dpi=180)
    plt.close(fig)

    counts = {
        "ap2_snr_le_3": int(np.sum(np.isfinite(ap2) & (ap2 <= 3.0))),
        "ap2_snr_3_5": int(np.sum(np.isfinite(ap2) & (ap2 > 3.0) & (ap2 < 5.0))),
        "ap2_snr_ge_5": int(np.sum(np.isfinite(ap2) & (ap2 >= 5.0))),
        "ap2_snr_invalid": int(np.sum(~np.isfinite(ap2))),
    }
    summary = {
        "n_clean": len(enriched),
        **counts,
        "ap2_snr": _finite_summary(ap2),
        "kron_snr": _finite_summary(kron),
        "region": str(reg_path),
        "csv": str(csv_path),
        "histogram": str(args.output_dir / f"{args.prefix}_clean_snr_hist.png"),
        "histogram_zoom_0_20": str(args.output_dir / f"{args.prefix}_clean_snr_hist_zoom_0_20.png"),
    }
    summary_path = args.output_dir / f"{args.prefix}_clean_snr_summary.txt"
    summary_path.write_text("\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
