#!/usr/bin/env python3
"""Export REG diagnostics for noisy/denoised groups using warp-weight SNR scaling."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import warnings
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from astropy.table import Table

from data_filtering.noncoadd_snr import (
    circular_aperture_offsets,
    classify_snr_values,
    mean_map_value_at,
    predict_snr_from_weight_ratio,
    read_coadd_weight_sum,
    read_effective_count_map,
    read_noisy_group_weight_summary,
)


COLORS = {
    "clean": "green",
    "center_only": "yellow",
    "ignore": "red",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed-root", type=Path, default=Path("/data/czh23/preprocessed"))
    parser.add_argument("--catalog-root", type=Path, default=Path("/data/shared/Subaru"))
    parser.add_argument(
        "--coadd-weight-root",
        type=Path,
        default=Path("/data/shared/handoff/2026-06-21_171607_hsc_metadata_warp_n2n_epoch006_full-all-warp-weights"),
    )
    parser.add_argument("--denoised-fits-root", type=Path, default=Path("/data/czh23/denoised_fits"))
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patch", default="4,5")
    parser.add_argument("--groups", nargs="+", default=["group_00", "group_01"])
    parser.add_argument("--bands", nargs="+", default=["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"])
    parser.add_argument("--out-dir", type=Path, default=Path("output/data_filter_0725/weight_ratio_snr_patch45"))
    parser.add_argument("--ap-radius", type=float, default=6.0)
    parser.add_argument("--ignore-snr-max", type=float, default=3.0)
    parser.add_argument("--center-only-snr-max", type=float, default=5.0)
    parser.add_argument("--cap-t-max", type=float, default=1.0)
    parser.add_argument("--zeropoint", type=float, default=27.0)
    parser.add_argument("--valid-coadd-weights-only", action="store_true")
    return parser.parse_args()


def _read_table(path: Path) -> Table:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*'second' did not parse as fits unit.*")
        return Table.read(path, hdu=1, memmap=True)


def _catalog_photometry_by_id(path: Path, *, zeropoint: float) -> dict[int, dict[str, float]]:
    table = _read_table(path)
    required = [
        "id",
        "base_CircularApertureFlux_6_0_instFlux",
        "base_CircularApertureFlux_6_0_instFluxErr",
    ]
    missing = [name for name in required if name not in table.colnames]
    if missing:
        raise KeyError(f"{path} missing columns: {missing}")
    ids = np.asarray(table["id"], dtype=np.int64)
    flux = np.asarray(table["base_CircularApertureFlux_6_0_instFlux"], dtype=np.float64)
    err = np.asarray(table["base_CircularApertureFlux_6_0_instFluxErr"], dtype=np.float64)
    snr = np.full(len(table), np.nan, dtype=np.float64)
    ok = np.isfinite(flux) & np.isfinite(err) & (err > 0.0)
    snr[ok] = flux[ok] / err[ok]
    mag = np.full(len(table), np.nan, dtype=np.float64)
    pos = np.isfinite(flux) & (flux > 0.0)
    mag[pos] = float(zeropoint) - 2.5 * np.log10(flux[pos])
    return {
        int(sid): {
            "ap2_flux": float(ap_flux),
            "ap2_flux_err": float(ap_err),
            "ap2_snr_coadd": float(ap_snr),
            "ap2_mag": float(ap_mag),
        }
        for sid, ap_flux, ap_err, ap_snr, ap_mag in zip(ids, flux, err, snr, mag)
    }


def _coadd_clean_rows(preprocessed_root: Path, *, tract: str, patch: str, band: str) -> list[dict[str, object]]:
    path = preprocessed_root / str(tract) / str(patch) / "band_reference_catalogs" / band / f"meas-{band}-{tract}-{patch}.fits"
    table = _read_table(path)
    required = [
        "id",
        "base_SdssCentroid_x",
        "base_SdssCentroid_y",
        "ellipse_major_sigma",
        "ellipse_minor_sigma",
        "ellipse_theta",
    ]
    missing = [name for name in required if name not in table.colnames]
    if missing:
        raise KeyError(f"{path} missing columns: {missing}")
    rows: list[dict[str, object]] = []
    for row in table:
        rows.append(
            {
                "id": int(row["id"]),
                "x": float(row["base_SdssCentroid_x"]),
                "y": float(row["base_SdssCentroid_y"]),
                "major": float(row["ellipse_major_sigma"]),
                "minor": float(row["ellipse_minor_sigma"]),
                "theta": float(row["ellipse_theta"]),
            }
        )
    return rows


def _line(row: dict[str, object], *, coord: str) -> str:
    if coord == "image":
        x = float(row["x_image"])
        y = float(row["y_image"])
    else:
        x = float(row["x"])
        y = float(row["y"])
    major = float(row["major"])
    minor = float(row["minor"])
    theta = float(row["theta"])
    class_name = str(row["weight_snr_class"])
    color = COLORS.get(class_name, "white")
    sid = int(row["id"])
    snr = float(row.get("snr_weight_pred", float("nan")))
    t_eff = float(row.get("t_eff", float("nan")))
    eff = float(row.get("local_effective_count", float("nan")))
    text = f"{sid} {class_name} SNRw={snr:.2f} T={t_eff:.3f} eff={eff:.2f}"
    area = math.pi * major * minor if np.isfinite(major * minor) else float("inf")
    if not np.isfinite(area) or area <= 0.0 or area > 40000.0:
        return f"point({x:.3f},{y:.3f}) # point=circle color={color} width=2 text={{{text}}}"
    return (
        f"ellipse({x:.3f},{y:.3f},{major:.3f},{minor:.3f},{math.degrees(theta):.3f}) "
        f"# color={color} width=2 text={{{text}}}"
    )


def _write_regs(out_dir: Path, *, prefix: str, rows: list[dict[str, object]], coord: str = "image") -> None:
    header = [
        "# Region file format: DS9 version 4.1",
        'global color=green dashlist=8 3 width=2 font="helvetica 10 normal roman" '
        "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
        coord,
    ]
    all_rows = sorted(rows, key=lambda r: float(r["major"]) * float(r["minor"]), reverse=True)
    (out_dir / f"{prefix}_weight_snr_all.reg").write_text(
        "\n".join(header + [_line(row, coord=coord) for row in all_rows]) + "\n",
        encoding="utf-8",
    )
    for class_name in ("clean", "center_only", "ignore"):
        selected = [row for row in all_rows if row["weight_snr_class"] == class_name]
        (out_dir / f"{prefix}_weight_snr_{class_name}.reg").write_text(
            "\n".join(header + [_line(row, coord=coord) for row in selected]) + "\n",
            encoding="utf-8",
        )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "id",
        "weight_snr_class",
        "x",
        "y",
        "x_image",
        "y_image",
        "major",
        "minor",
        "theta",
        "ap2_flux",
        "ap2_flux_err",
        "ap2_snr_coadd",
        "ap2_mag",
        "coadd_weight_sum",
        "coadd_weight_count",
        "selected_weight_sum",
        "selected_weight_count",
        "global_t",
        "local_effective_count",
        "local_coverage_fraction",
        "t_eff",
        "snr_weight_pred",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    off_y, off_x = circular_aperture_offsets(float(args.ap_radius))
    summary_rows: list[dict[str, object]] = []

    for group in args.groups:
        for band in args.bands:
            clean_rows = _coadd_clean_rows(args.preprocessed_root, tract=str(args.tract), patch=str(args.patch), band=band)
            catalog_path = args.catalog_root / str(args.tract) / band / str(args.patch) / f"meas-{band}-{args.tract}-{args.patch}.fits"
            phot_by_id = _catalog_photometry_by_id(catalog_path, zeropoint=float(args.zeropoint))

            weight_csv = args.coadd_weight_root / band / str(args.patch) / "weights.csv"
            coadd_weight_sum, coadd_weight_count = read_coadd_weight_sum(weight_csv, valid_only=bool(args.valid_coadd_weights_only))
            group_dir = args.denoised_fits_root / f"patch_{str(args.patch).replace(',', '_')}" / str(group) / band
            group_weights = read_noisy_group_weight_summary(group_dir / "meta.json")
            effective_count, effective_origin = read_effective_count_map(group_dir / "effective_count.fits")

            ap_snr = np.full(len(clean_rows), np.nan, dtype=np.float32)
            local_effective = np.full(len(clean_rows), np.nan, dtype=np.float32)
            for idx, row in enumerate(clean_rows):
                phot = phot_by_id.get(int(row["id"]), {})
                row.update(phot)
                ap_snr[idx] = float(phot.get("ap2_snr_coadd", float("nan")))
                local_effective[idx] = mean_map_value_at(
                    effective_count,
                    effective_origin,
                    float(row["x"]),
                    float(row["y"]),
                    off_y,
                    off_x,
                )
                row["local_effective_count"] = float(local_effective[idx])
                row["x_image"] = float(row["x"]) - float(effective_origin[0]) + 1.0
                row["y_image"] = float(row["y"]) - float(effective_origin[1]) + 1.0

            snr_pred, t_eff = predict_snr_from_weight_ratio(
                ap_snr,
                coadd_weight_sum=coadd_weight_sum,
                selected_weight_sum=float(group_weights["selected_weight_sum"]),
                selected_weight_count=float(group_weights["selected_weight_count"]),
                local_effective_count=local_effective,
                cap_t_max=float(args.cap_t_max),
            )
            classes = classify_snr_values(
                snr_pred,
                ignore_snr_max=float(args.ignore_snr_max),
                center_only_snr_max=float(args.center_only_snr_max),
            )
            global_t = (
                float(group_weights["selected_weight_sum"]) / float(coadd_weight_sum)
                if coadd_weight_sum > 0.0
                else float("nan")
            )
            for idx, row in enumerate(clean_rows):
                row["weight_snr_class"] = str(classes[idx])
                row["coadd_weight_sum"] = float(coadd_weight_sum)
                row["coadd_weight_count"] = int(coadd_weight_count)
                row["selected_weight_sum"] = float(group_weights["selected_weight_sum"])
                row["selected_weight_count"] = int(group_weights["selected_weight_count"])
                row["global_t"] = float(global_t)
                row["local_coverage_fraction"] = (
                    float(local_effective[idx]) / float(group_weights["selected_weight_count"])
                    if float(group_weights["selected_weight_count"]) > 0.0 and np.isfinite(local_effective[idx])
                    else float("nan")
                )
                row["t_eff"] = float(t_eff[idx])
                row["snr_weight_pred"] = float(snr_pred[idx])

            prefix = f"{group}_{band}_{args.tract}_{str(args.patch).replace(',', '_')}"
            _write_regs(args.out_dir, prefix=prefix, rows=clean_rows, coord="image")
            _write_csv(args.out_dir / f"{prefix}_weight_snr_sources.csv", clean_rows)

            counts = Counter(str(row["weight_snr_class"]) for row in clean_rows)
            finite_t = np.asarray([float(row["t_eff"]) for row in clean_rows], dtype=np.float64)
            finite_t = finite_t[np.isfinite(finite_t)]
            finite_snr = np.asarray([float(row["snr_weight_pred"]) for row in clean_rows], dtype=np.float64)
            finite_snr = finite_snr[np.isfinite(finite_snr)]
            summary = {
                "group": group,
                "band": band,
                "coadd_clean_input": len(clean_rows),
                "weight_clean": int(counts["clean"]),
                "weight_center_only": int(counts["center_only"]),
                "weight_ignore": int(counts["ignore"]),
                "coadd_weight_sum": float(coadd_weight_sum),
                "coadd_weight_count": int(coadd_weight_count),
                "selected_weight_sum": float(group_weights["selected_weight_sum"]),
                "selected_weight_count": int(group_weights["selected_weight_count"]),
                "global_t": float(global_t),
                "t_eff_median": float(np.nanmedian(finite_t)) if finite_t.size else float("nan"),
                "t_eff_p10": float(np.nanpercentile(finite_t, 10)) if finite_t.size else float("nan"),
                "t_eff_p90": float(np.nanpercentile(finite_t, 90)) if finite_t.size else float("nan"),
                "snr_pred_median": float(np.nanmedian(finite_snr)) if finite_snr.size else float("nan"),
            }
            summary_rows.append(summary)
            print(summary, flush=True)

    if summary_rows:
        fields = list(summary_rows[0].keys())
        with (args.out_dir / f"{args.tract}_{str(args.patch).replace(',', '_')}_weight_snr_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"wrote {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
