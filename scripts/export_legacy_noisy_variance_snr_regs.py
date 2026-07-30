#!/usr/bin/env python3
"""Export legacy noisy REGs with per-source variance-scaled SNR filtering."""

from __future__ import annotations

import argparse
import csv
import math
import os
import warnings
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")

import numpy as np
import zarr
from astropy.io import fits
from astropy.table import Table


COLORS = {
    "clean": "green",
    "center_only": "yellow",
    "ignore": "red",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zarr-root", type=Path, default=Path("/data/czh23/legacy_zarr/legacy_zarr"))
    p.add_argument("--preprocessed-root", type=Path, default=Path("/data/czh23/preprocessed"))
    p.add_argument("--noisy-fits-root", type=Path, default=Path("/home/czh23/fits/noisy"))
    p.add_argument("--coadd-root", type=Path, default=Path("/data/shared/Subaru"))
    p.add_argument("--catalog-root", type=Path, default=Path("/data/shared/Subaru"))
    p.add_argument("--tract", default="9813")
    p.add_argument("--patch", default="4,5")
    p.add_argument("--variant", default="noisy")
    p.add_argument("--group", default="group_00")
    p.add_argument("--bands", nargs="+", default=["HSC-G", "HSC-I"])
    p.add_argument("--out-dir", type=Path, default=Path("output/data_filter_0725/legacy_noisy_variance_snr_regs"))
    p.add_argument("--ap-radius", type=float, default=6.0)
    p.add_argument("--scale-max-sources", type=int, default=5000)
    p.add_argument("--ignore-snr-max", type=float, default=3.0)
    p.add_argument("--center-only-snr-max", type=float, default=5.0)
    p.add_argument("--cap-t-max", type=float, default=1.0, help="Cap effective exposure ratio T. Use <=0 to disable.")
    p.add_argument("--zeropoint", type=float, default=31.4)
    return p.parse_args()


def _decode_fixed(row: np.ndarray) -> str:
    return bytes(np.asarray(row, dtype=np.uint8)).split(b"\0", 1)[0].decode("utf-8", "ignore")


def _read_fits_image(path: Path, extname: str) -> tuple[np.ndarray, tuple[float, float]]:
    with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
        if extname in hdul:
            hdu = hdul[extname]
        else:
            # LSST calexp fallback: IMAGE/MASK/VARIANCE are usually HDU 1/2/3.
            index = {"IMAGE": 1, "MASK": 2, "VARIANCE": 3}[extname]
            hdu = hdul[index]
        data = np.asarray(hdu.data, dtype=np.float32)
        origin = (-float(hdu.header.get("LTV1", 0.0)), -float(hdu.header.get("LTV2", 0.0)))
    return data, origin


def _catalog_snr_map(catalog_path: Path, *, zeropoint: float) -> dict[int, dict[str, float]]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*'second' did not parse as fits unit.*")
        table = Table.read(catalog_path, hdu=1, memmap=True)
    required = [
        "id",
        "base_CircularApertureFlux_6_0_instFlux",
        "base_CircularApertureFlux_6_0_instFluxErr",
    ]
    missing = [name for name in required if name not in table.colnames]
    if missing:
        raise KeyError(f"{catalog_path} missing columns: {missing}")
    ids = np.asarray(table["id"], dtype=np.int64)
    flux = np.asarray(table["base_CircularApertureFlux_6_0_instFlux"], dtype=np.float64)
    err = np.asarray(table["base_CircularApertureFlux_6_0_instFluxErr"], dtype=np.float64)
    snr = np.full(len(table), np.nan, dtype=np.float64)
    ok = np.isfinite(flux) & np.isfinite(err) & (err > 0.0)
    snr[ok] = flux[ok] / err[ok]
    mag = np.full(len(table), np.nan, dtype=np.float64)
    pos = np.isfinite(flux) & (flux > 0.0)
    mag[pos] = float(zeropoint) - 2.5 * np.log10(flux[pos])
    return {int(sid): {"ap2_flux": float(f), "ap2_snr_coadd": float(s), "ap2_mag": float(m)} for sid, f, s, m in zip(ids, flux, snr, mag)}


def _source_rows_from_zarr(root, *, band_idx: int, group: str) -> list[dict[str, object]]:
    rows_by_id: dict[int, dict[str, object]] = {}
    n_samples = root["shape_source_offsets"].shape[0]
    for sample_idx in range(n_samples):
        if _decode_fixed(root["group"][sample_idx]) != group:
            continue
        x0 = int(root["tile_x0"][sample_idx])
        y0 = int(root["tile_y0"][sample_idx])
        start = int(root["shape_source_offsets"][sample_idx, band_idx])
        stop = int(root["shape_source_offsets"][sample_idx, band_idx + 1])
        if stop <= start:
            continue
        centers = np.asarray(root["shape_source_centers"][start:stop], dtype=np.float64)
        values = np.asarray(root["shape_source_values"][start:stop], dtype=np.float64)
        classes = np.asarray(root["shape_source_classes"][start:stop], dtype=np.uint8)
        ids = np.asarray(root["shape_source_ids"][start:stop], dtype=np.int64)
        for center, value, cls, sid in zip(centers, values, classes, ids):
            if int(cls) == 1:
                legacy_class = "clean"
            elif int(cls) == 2:
                legacy_class = "center_only"
            else:
                continue
            source_id = int(sid)
            row = {
                "id": source_id,
                "legacy_class": legacy_class,
                "x": float(center[0]) + float(x0),
                "y": float(center[1]) + float(y0),
                "major": float(value[0]),
                "minor": float(value[1]),
                "theta": float(value[2]),
            }
            old = rows_by_id.get(source_id)
            if old is None:
                rows_by_id[source_id] = row
            elif old["legacy_class"] != "clean" and legacy_class == "clean":
                rows_by_id[source_id] = row
    return list(rows_by_id.values())


def _coadd_clean_rows(preprocessed_root: Path, *, tract: str, patch: str, band: str) -> list[dict[str, object]]:
    path = (
        preprocessed_root
        / str(tract)
        / str(patch)
        / "band_reference_catalogs"
        / band
        / f"meas-{band}-{tract}-{patch}.fits"
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*'second' did not parse as fits unit.*")
        table = Table.read(path, hdu=1, memmap=True)
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
                "legacy_class": "coadd_clean",
                "x": float(row["base_SdssCentroid_x"]),
                "y": float(row["base_SdssCentroid_y"]),
                "major": float(row["ellipse_major_sigma"]),
                "minor": float(row["ellipse_minor_sigma"]),
                "theta": float(row["ellipse_theta"]),
            }
        )
    return rows


def _aperture_offsets(radius: float) -> tuple[np.ndarray, np.ndarray]:
    r = int(math.ceil(float(radius)))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    keep = (xx.astype(np.float64) ** 2 + yy.astype(np.float64) ** 2) <= float(radius) ** 2
    return yy[keep].astype(np.int32), xx[keep].astype(np.int32)


def _mean_variance_at(var: np.ndarray, origin: tuple[float, float], x: float, y: float, off_y: np.ndarray, off_x: np.ndarray) -> float:
    cx = int(round(float(x) - float(origin[0])))
    cy = int(round(float(y) - float(origin[1])))
    yy = cy + off_y
    xx = cx + off_x
    inside = (yy >= 0) & (yy < var.shape[0]) & (xx >= 0) & (xx < var.shape[1])
    if not bool(np.any(inside)):
        return float("nan")
    values = np.asarray(var[yy[inside], xx[inside]], dtype=np.float64)
    good = np.isfinite(values) & (values > 0.0)
    if not bool(np.any(good)):
        return float("nan")
    return float(np.mean(values[good]))


def _aperture_sum_at(image: np.ndarray, origin: tuple[float, float], x: float, y: float, off_y: np.ndarray, off_x: np.ndarray) -> float:
    cx = int(round(float(x) - float(origin[0])))
    cy = int(round(float(y) - float(origin[1])))
    yy = cy + off_y
    xx = cx + off_x
    inside = (yy >= 0) & (yy < image.shape[0]) & (xx >= 0) & (xx < image.shape[1])
    if not bool(np.any(inside)):
        return float("nan")
    values = np.asarray(image[yy[inside], xx[inside]], dtype=np.float64)
    good = np.isfinite(values)
    if not bool(np.any(good)):
        return float("nan")
    return float(np.sum(values[good]))


def _estimate_noisy_to_coadd_scale(
    rows: list[dict[str, object]],
    *,
    coadd_image: np.ndarray,
    coadd_origin: tuple[float, float],
    noisy_image: np.ndarray,
    noisy_origin: tuple[float, float],
    off_y: np.ndarray,
    off_x: np.ndarray,
    max_sources: int,
) -> tuple[float, dict[str, float]]:
    ratios = []
    for row in rows[: max(0, int(max_sources))]:
        coadd_sum = _aperture_sum_at(coadd_image, coadd_origin, float(row["x"]), float(row["y"]), off_y, off_x)
        noisy_sum = _aperture_sum_at(noisy_image, noisy_origin, float(row["x"]), float(row["y"]), off_y, off_x)
        if np.isfinite(coadd_sum) and np.isfinite(noisy_sum) and coadd_sum > 0.0 and noisy_sum > 0.0:
            ratios.append(noisy_sum / coadd_sum)
    values = np.asarray(ratios, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return 1.0, {"n": 0, "p10": float("nan"), "p25": float("nan"), "median": 1.0, "p75": float("nan"), "p90": float("nan")}
    q = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])
    return float(q[2]), {"n": int(values.size), "p10": float(q[0]), "p25": float(q[1]), "median": float(q[2]), "p75": float(q[3]), "p90": float(q[4])}


def _class_from_snr(snr: float, args: argparse.Namespace) -> str:
    if not np.isfinite(snr):
        return "ignore"
    if snr <= float(args.ignore_snr_max):
        return "ignore"
    if snr <= float(args.center_only_snr_max):
        return "center_only"
    return "clean"


def _line(row: dict[str, object]) -> str:
    x = float(row["x"]) + 1.0
    y = float(row["y"]) + 1.0
    major = float(row["major"])
    minor = float(row["minor"])
    theta = float(row["theta"])
    class_name = str(row["variance_snr_class"])
    color = COLORS[class_name]
    sid = int(row["id"])
    snr_noisy = float(row.get("snr_noisy_pred", float("nan")))
    t_eff = float(row.get("t_eff", float("nan")))
    text = (
        f"{sid} {class_name} "
        f"SNRn={snr_noisy:.2f} "
        f"T={t_eff:.3f} "
        f"old={row['legacy_class']}"
    )
    area = math.pi * major * minor if np.isfinite(major * minor) else float("inf")
    if not np.isfinite(area) or area <= 0.0 or area > 40000.0:
        return f"point({x:.3f},{y:.3f}) # point=circle color={color} width=2 text={{{text}}}"
    return (
        f"ellipse({x:.3f},{y:.3f},{major:.3f},{minor:.3f},{math.degrees(theta):.3f}) "
        f"# color={color} width=2 text={{{text}}}"
    )


def _write_regs(out_dir: Path, *, prefix: str, rows: list[dict[str, object]]) -> None:
    header = [
        "# Region file format: DS9 version 4.1",
        'global color=green dashlist=8 3 width=2 font="helvetica 10 normal roman" '
        "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
        "physical",
    ]
    for class_name in ("clean", "center_only", "ignore"):
        selected = [row for row in rows if row["variance_snr_class"] == class_name]
        selected.sort(key=lambda r: float(r["major"]) * float(r["minor"]), reverse=True)
        path = out_dir / f"{prefix}_variance_snr_{class_name}.reg"
        path.write_text("\n".join(header + [_line(row) for row in selected]) + "\n", encoding="utf-8")


def _write_legacy_regs(out_dir: Path, *, prefix: str, rows: list[dict[str, object]]) -> None:
    header = [
        "# Region file format: DS9 version 4.1",
        'global color=green dashlist=8 3 width=2 font="helvetica 10 normal roman" '
        "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
        "physical",
    ]
    for class_name in ("clean", "center_only"):
        selected = [dict(row, variance_snr_class=class_name) for row in rows if row["legacy_class"] == class_name]
        selected.sort(key=lambda r: float(r["major"]) * float(r["minor"]), reverse=True)
        path = out_dir / f"{prefix}_legacy_noisy_{class_name}.reg"
        path.write_text("\n".join(header + [_line(row) for row in selected]) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "id",
        "legacy_class",
        "variance_snr_class",
        "x",
        "y",
        "major",
        "minor",
        "theta",
        "ap2_flux",
        "ap2_mag",
        "ap2_snr_coadd",
        "mean_var_coadd",
        "mean_var_noisy",
        "noisy_to_coadd_scale",
        "t_eff",
        "snr_noisy_pred",
        "m5_coadd",
        "m5_noisy",
        "legacy_noisy_class",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    args = parse_args()
    zarr_path = args.zarr_root / str(args.tract) / args.variant / f"{args.patch}.zarr"
    root = zarr.open_group(str(zarr_path), mode="r")
    bands = list(root.attrs["bands"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    off_y, off_x = _aperture_offsets(float(args.ap_radius))

    for band in args.bands:
        if band not in bands:
            raise KeyError(f"{band} not in {zarr_path}; bands={bands}")
        band_idx = bands.index(band)
        noisy_path = args.noisy_fits_root / band / f"deepCoadd-{band}-{args.tract}-{args.patch}.fits"
        coadd_path = args.coadd_root / str(args.tract) / band / args.patch / f"calexp-{band}-{args.tract}-{args.patch}.fits"
        catalog_path = args.catalog_root / str(args.tract) / band / args.patch / f"meas-{band}-{args.tract}-{args.patch}.fits"

        noisy_var, noisy_origin = _read_fits_image(noisy_path, "VARIANCE")
        noisy_image, noisy_image_origin = _read_fits_image(noisy_path, "IMAGE")
        coadd_var, coadd_origin = _read_fits_image(coadd_path, "VARIANCE")
        coadd_image, coadd_image_origin = _read_fits_image(coadd_path, "IMAGE")
        snr_map = _catalog_snr_map(catalog_path, zeropoint=float(args.zeropoint))
        legacy_rows = _source_rows_from_zarr(root, band_idx=band_idx, group=str(args.group))
        legacy_by_id = {int(row["id"]): str(row["legacy_class"]) for row in legacy_rows}
        rows = _coadd_clean_rows(args.preprocessed_root, tract=str(args.tract), patch=str(args.patch), band=band)
        scale, scale_stats = _estimate_noisy_to_coadd_scale(
            rows,
            coadd_image=coadd_image,
            coadd_origin=coadd_image_origin,
            noisy_image=noisy_image,
            noisy_origin=noisy_image_origin,
            off_y=off_y,
            off_x=off_x,
            max_sources=int(args.scale_max_sources),
        )
        for row in rows:
            sid = int(row["id"])
            phot = snr_map.get(sid, {})
            row.update(phot)
            v_coadd = _mean_variance_at(coadd_var, coadd_origin, float(row["x"]), float(row["y"]), off_y, off_x)
            v_noisy = _mean_variance_at(noisy_var, noisy_origin, float(row["x"]), float(row["y"]), off_y, off_x)
            t_eff = float("nan")
            if np.isfinite(v_coadd) and np.isfinite(v_noisy) and v_coadd > 0.0 and v_noisy > 0.0:
                t_eff = (v_coadd * scale * scale) / v_noisy
                if float(args.cap_t_max) > 0.0:
                    t_eff = min(t_eff, float(args.cap_t_max))
            snr_coadd = float(row.get("ap2_snr_coadd", float("nan")))
            snr_noisy = snr_coadd * math.sqrt(t_eff) if np.isfinite(snr_coadd) and np.isfinite(t_eff) and t_eff >= 0.0 else float("nan")
            mag = float(row.get("ap2_mag", float("nan")))
            m5_coadd = mag + 2.5 * math.log10(snr_coadd / 5.0) if np.isfinite(mag) and snr_coadd > 0.0 else float("nan")
            m5_noisy = m5_coadd + 1.25 * math.log10(t_eff) if np.isfinite(m5_coadd) and t_eff > 0.0 else float("nan")
            row.update(
                {
                    "mean_var_coadd": v_coadd,
                    "mean_var_noisy": v_noisy,
                    "noisy_to_coadd_scale": scale,
                    "t_eff": t_eff,
                    "snr_noisy_pred": snr_noisy,
                    "m5_coadd": m5_coadd,
                    "m5_noisy": m5_noisy,
                    "variance_snr_class": _class_from_snr(snr_noisy, args),
                    "legacy_noisy_class": legacy_by_id.get(int(row["id"]), "legacy_absent_or_ignore"),
                }
            )
        prefix = f"{args.variant}_{args.group}_{band}_{args.tract}_{args.patch.replace(',', '_')}"
        _write_legacy_regs(args.out_dir, prefix=prefix, rows=legacy_rows)
        _write_regs(args.out_dir, prefix=prefix, rows=rows)
        _write_csv(args.out_dir / f"{prefix}_variance_snr_sources.csv", rows)
        legacy_before = Counter(str(row["legacy_class"]) for row in legacy_rows)
        after = Counter(str(row["variance_snr_class"]) for row in rows)
        transitions = Counter(
            (str(row["legacy_noisy_class"]), str(row["variance_snr_class"]))
            for row in rows
        )
        t_values = np.asarray([float(row["t_eff"]) for row in rows], dtype=np.float64)
        snr_values = np.asarray([float(row["snr_noisy_pred"]) for row in rows], dtype=np.float64)
        finite_t = t_values[np.isfinite(t_values)]
        finite_snr = snr_values[np.isfinite(snr_values)]
        summary = {
            "band": band,
            "coadd_clean_input": len(rows),
            "legacy_clean": int(legacy_before["clean"]),
            "legacy_center_only": int(legacy_before["center_only"]),
            "legacy_source_total": int(sum(legacy_before.values())),
            "variance_clean": int(after["clean"]),
            "variance_center_only": int(after["center_only"]),
            "variance_ignore": int(after["ignore"]),
            "noisy_to_coadd_scale": float(scale),
            "scale_n": int(scale_stats["n"]),
            "scale_p10": float(scale_stats["p10"]),
            "scale_p90": float(scale_stats["p90"]),
            "t_eff_median": float(np.nanmedian(finite_t)) if finite_t.size else float("nan"),
            "t_eff_p10": float(np.nanpercentile(finite_t, 10)) if finite_t.size else float("nan"),
            "t_eff_p90": float(np.nanpercentile(finite_t, 90)) if finite_t.size else float("nan"),
            "snr_noisy_median": float(np.nanmedian(finite_snr)) if finite_snr.size else float("nan"),
        }
        summary_rows.append(summary)
        print(summary)
        transition_path = args.out_dir / f"{prefix}_legacy_vs_variance_transition.csv"
        with transition_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["legacy_noisy_class", "variance_snr_class", "count"])
            writer.writeheader()
            for (old, new), count in sorted(transitions.items()):
                writer.writerow({"legacy_noisy_class": old, "variance_snr_class": new, "count": int(count)})

    with (args.out_dir / f"{args.variant}_{args.group}_{args.tract}_{args.patch.replace(',', '_')}_variance_snr_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = list(summary_rows[0].keys()) if summary_rows else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
