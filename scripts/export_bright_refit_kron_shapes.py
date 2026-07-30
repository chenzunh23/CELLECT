#!/usr/bin/env python3
"""Export bright refit Kron-aperture shape REGs and quick-look overlays."""

from __future__ import annotations

import argparse
import csv
import math
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from matplotlib.patches import Ellipse


DEFAULT_BANDS = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refit-root", type=Path, default=Path("/data/czh23/refit"))
    parser.add_argument("--coadd-root", type=Path, default=Path("/data/shared/Subaru"))
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patch", default="4,5")
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    parser.add_argument("--mag-threshold", type=float, default=18.0)
    parser.add_argument("--zeropoint", type=float, default=27.0)
    parser.add_argument("--radius-column", default="proxy_nan0_flux_aperture_radius")
    parser.add_argument("--good-column", default="proxy_nan0_good")
    parser.add_argument("--out-dir", type=Path, default=Path("output/data_filter_0725/bright_refit_kron_patch45"))
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--plot-downsample", type=int, default=4)
    return parser.parse_args()


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _mag_from_flux(flux: float, zeropoint: float) -> float:
    if not np.isfinite(flux) or flux <= 0.0:
        return math.nan
    return float(zeropoint) - 2.5 * math.log10(float(flux))


def _read_refit_rows(path: Path, *, args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, int]]:
    rows: list[dict[str, object]] = []
    stats = {
        "csv_rows": 0,
        "bright_mag_rows": 0,
        "selected_good_rows": 0,
        "skipped_bad_refit": 0,
        "skipped_bad_geometry": 0,
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        required = {
            "source_id",
            "x_image",
            "y_image",
            "axis_a",
            "axis_b",
            "theta_deg",
            "initial_determinant_radius",
            "catalog_KronFlux_instFlux",
            args.radius_column,
        }
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise KeyError(f"{path} missing columns: {missing}")
        for row in reader:
            stats["csv_rows"] += 1
            flux = _float(row, "catalog_KronFlux_instFlux")
            mag = _mag_from_flux(flux, float(args.zeropoint))
            if not np.isfinite(mag) or mag >= float(args.mag_threshold):
                continue
            stats["bright_mag_rows"] += 1
            if row.get("status", "") != "ok" or (args.good_column in row and not _is_true(row.get(args.good_column))):
                stats["skipped_bad_refit"] += 1
                continue
            x = _float(row, "x_image")
            y = _float(row, "y_image")
            axis_a = _float(row, "axis_a")
            axis_b = _float(row, "axis_b")
            theta = _float(row, "theta_deg")
            initial_radius = _float(row, "initial_determinant_radius")
            target_radius = _float(row, args.radius_column)
            if not all(np.isfinite(v) for v in (x, y, axis_a, axis_b, theta, initial_radius, target_radius)):
                stats["skipped_bad_geometry"] += 1
                continue
            if axis_a <= 0.0 or axis_b <= 0.0 or initial_radius <= 0.0 or target_radius <= 0.0:
                stats["skipped_bad_geometry"] += 1
                continue
            scale = target_radius / initial_radius
            major = axis_a * scale
            minor = axis_b * scale
            rows.append(
                {
                    "source_id": int(float(row["source_id"])),
                    "row_index": int(float(row.get("row_index", -1))),
                    "x": x,
                    "y": y,
                    "major": major,
                    "minor": minor,
                    "theta_deg": theta,
                    "mag": mag,
                    "flux": flux,
                    "area": math.pi * major * minor,
                    "measurement_surface": row.get("measurement_surface", ""),
                }
            )
            stats["selected_good_rows"] += 1
    return rows, stats


def _read_image(path: Path) -> np.ndarray:
    with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
        if "IMAGE" in hdul:
            hdu = hdul["IMAGE"]
        else:
            hdu = next(h for h in hdul if getattr(h, "data", None) is not None and getattr(h.data, "ndim", None) == 2)
        return np.asarray(hdu.data, dtype=np.float32)


def _zscale_image(image: np.ndarray) -> np.ndarray:
    finite = np.asarray(image[np.isfinite(image)], dtype=np.float32)
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vmin, vmax = ZScaleInterval().get_limits(finite)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = np.nanpercentile(finite, [1, 99])
    return np.clip((image - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)


def _write_reg(path: Path, rows: list[dict[str, object]], *, band: str, args: argparse.Namespace) -> None:
    lines = [
        "# Region file format: DS9 version 4.1",
        f"# {band} tract={args.tract} patch={args.patch} refit {args.radius_column} Kron mag < {args.mag_threshold}",
        'global color=red dashlist=8 3 width=2 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',
        "image",
    ]
    for row in sorted(rows, key=lambda item: float(item["area"]), reverse=True):
        text = (
            f"id={row['source_id']} mag={float(row['mag']):.2f} "
            f"area={float(row['area']):.1f} {row['measurement_surface']}"
        )
        lines.append(
            "ellipse({x:.3f},{y:.3f},{major:.3f},{minor:.3f},{theta:.3f}) "
            "# color=red width=2 text={{{text}}}".format(
                x=float(row["x"]),
                y=float(row["y"]),
                major=float(row["major"]),
                minor=float(row["minor"]),
                theta=float(row["theta_deg"]),
                text=text,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _downsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    factor = max(1, int(factor))
    if factor == 1:
        return image
    return np.asarray(image[::factor, ::factor], dtype=np.float32)


def _plot_band(path: Path, image: np.ndarray, rows: list[dict[str, object]], *, band: str, args: argparse.Namespace) -> None:
    ds = max(1, int(args.plot_downsample))
    plot_image = _downsample_image(image, ds)
    scaled = _zscale_image(plot_image)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=args.dpi)
    ax.imshow(scaled, cmap="gray", origin="lower", interpolation="nearest")
    for row in sorted(rows, key=lambda item: float(item["area"]), reverse=True):
        ell = Ellipse(
            (float(row["x"]) / ds, float(row["y"]) / ds),
            width=2.0 * float(row["major"]) / ds,
            height=2.0 * float(row["minor"]) / ds,
            angle=float(row["theta_deg"]),
            fill=False,
            edgecolor="red",
            linewidth=1.4,
            alpha=0.95,
        )
        ax.add_patch(ell)
        ax.plot(float(row["x"]) / ds, float(row["y"]) / ds, marker="+", color="cyan", markersize=4, markeredgewidth=0.8)
    ax.set_xlim(0, plot_image.shape[1])
    ax.set_ylim(0, plot_image.shape[0])
    ax.set_title(f"{band} 9813/{args.patch}: refit Kron mag < {args.mag_threshold:g} (n={len(rows)})")
    ax.set_xlabel("image x")
    ax.set_ylabel("image y")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_contact(path: Path, panels: list[tuple[str, np.ndarray, list[dict[str, object]]]], *, args: argparse.Namespace) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4), dpi=args.dpi, squeeze=False)
    for ax, (band, image, rows) in zip(axes[0], panels):
        ds = max(1, int(args.plot_downsample))
        plot_image = _downsample_image(image, ds)
        ax.imshow(_zscale_image(plot_image), cmap="gray", origin="lower", interpolation="nearest")
        for row in sorted(rows, key=lambda item: float(item["area"]), reverse=True):
            ax.add_patch(
                Ellipse(
                    (float(row["x"]) / ds, float(row["y"]) / ds),
                    width=2.0 * float(row["major"]) / ds,
                    height=2.0 * float(row["minor"]) / ds,
                    angle=float(row["theta_deg"]),
                    fill=False,
                    edgecolor="red",
                    linewidth=1.1,
                    alpha=0.95,
                )
            )
        ax.set_title(f"{band}\nn={len(rows)}")
        ax.set_xlim(0, plot_image.shape[1])
        ax.set_ylim(0, plot_image.shape[0])
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"9813/{args.patch} refit Kron aperture, catalog Kron mag < {args.mag_threshold:g}", y=0.99)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panels: list[tuple[str, np.ndarray, list[dict[str, object]]]] = []
    summary_rows = []
    for band in args.bands:
        refit_csv = args.refit_root / str(args.tract) / band / args.patch / "batch_heavyfp_kron_refit" / "batch_heavyfp_kron_refit.csv"
        image_path = args.coadd_root / str(args.tract) / band / args.patch / f"calexp-{band}-{args.tract}-{args.patch}.fits"
        rows, stats = _read_refit_rows(refit_csv, args=args)
        image = _read_image(image_path)
        safe_patch = args.patch.replace(",", "_")
        _write_reg(args.out_dir / f"{band}_{args.tract}_{safe_patch}_kron_mag_lt_{args.mag_threshold:g}_refit_aperture.reg", rows, band=band, args=args)
        _plot_band(args.out_dir / f"{band}_{args.tract}_{safe_patch}_kron_mag_lt_{args.mag_threshold:g}_refit_aperture.png", image, rows, band=band, args=args)
        panels.append((band, image, rows))
        stats.update({"band": band})
        summary_rows.append(stats)
        print(f"{band}: {stats}")
    _plot_contact(args.out_dir / f"{args.tract}_{args.patch.replace(',', '_')}_all_bands_kron_mag_lt_{args.mag_threshold:g}_refit_aperture.png", panels, args=args)
    with (args.out_dir / f"{args.tract}_{args.patch.replace(',', '_')}_kron_mag_lt_{args.mag_threshold:g}_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = ["band", "csv_rows", "bright_mag_rows", "selected_good_rows", "skipped_bad_refit", "skipped_bad_geometry"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
