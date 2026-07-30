#!/usr/bin/env python3
"""Export PU clean/center-only/ignore Kron aperture regions for one patch."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import warnings
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/cellect_mplconfig"

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import ZScaleInterval

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from astro_data_preprocessing import _classify_pu_catalog, _move_bright_clean_to_center_only  # noqa: E402
from data_filtering.pu_source_filter import attach_kron_refit_radius  # noqa: E402


COLORS = {
    "clean": "green",
    "center_only": "orange",
    "strict_center_only": "cyan",
    "ignore": "red",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--calexp", type=Path, required=True)
    p.add_argument("--refit-csv", type=Path, required=True)
    p.add_argument("--tract", default="9813")
    p.add_argument("--band", default="NB1010")
    p.add_argument("--patch", default="4,5")
    p.add_argument("--output-dir", type=Path, default=Path("output/data_filter_0722/nb1010_45_pu_kron_aperture"))
    p.add_argument("--catalog-hdu", type=int, default=1)
    p.add_argument("--source-filter", default="nchild0")
    p.add_argument(
        "--aperture-scale",
        type=float,
        default=1.0,
        help=(
            "Extra display scale applied to the refit aperture. Keep at 1.0 for "
            "batch-heavyfp-kron-refit proxy_flux_aperture.reg parity; that radius "
            "is already n_radius_for_flux times the Kron radius."
        ),
    )
    p.add_argument("--large-area-as-point", type=float, default=10000.0)
    p.add_argument("--max-overlay-per-class", type=int, default=2500)
    p.add_argument(
        "--strict-bright-center-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Move clean sources brighter than the per-band saturation threshold to center_only.",
    )
    p.add_argument(
        "--include-unmatched-refit-as-ignore",
        action="store_true",
        help=(
            "Keep catalog rows without a matched refit radius and let PU classification put them in ignore. "
            "By default only rows matched to the refit CSV are classified."
        ),
    )
    p.add_argument(
        "--cap-refit-aperture-to-official",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cap refit flux-aperture radius at 2.5 * catalog_KronFlux_radius before PU classification.",
    )
    p.add_argument("--official-aperture-scale", type=float, default=2.5)
    return p.parse_args()


def ds9_header(title: str) -> list[str]:
    return [
        "# Region file format: DS9 version 4.1",
        f"# {title}",
        'global color=green dashlist=8 3 width=2 font="helvetica 10 normal roman" '
        "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
        "physical",
    ]


def _source_id(table: Table, row: int) -> str:
    if "id" in table.colnames:
        return str(int(table["id"][row]))
    if "source_id" in table.colnames:
        return str(int(table["source_id"][row]))
    return str(row)


def table_to_region_lines(
    table: Table,
    *,
    color: str,
    aperture_scale: float,
    large_area_as_point: float,
) -> list[str]:
    if len(table) == 0:
        return []
    x = np.asarray(table["base_SdssCentroid_x"], dtype=float)
    y = np.asarray(table["base_SdssCentroid_y"], dtype=float)
    base_major = np.asarray(table["ellipse_major_sigma"], dtype=float)
    base_minor = np.asarray(table["ellipse_minor_sigma"], dtype=float)
    major = base_major * float(aperture_scale)
    minor = base_minor * float(aperture_scale)
    theta = np.asarray(table["ellipse_theta"], dtype=float)
    area = math.pi * major * minor
    base_area = math.pi * base_major * base_minor
    lines: list[str] = []
    for idx in range(len(table)):
        if not (np.isfinite(x[idx]) and np.isfinite(y[idx])):
            continue
        sid = _source_id(table, idx)
        if (
            not (np.isfinite(major[idx]) and np.isfinite(minor[idx]) and np.isfinite(theta[idx]))
            or major[idx] <= 0.0
            or minor[idx] <= 0.0
            or (np.isfinite(base_area[idx]) and base_area[idx] > float(large_area_as_point))
        ):
            lines.append(f"point({x[idx]:.3f},{y[idx]:.3f}) # point=circle color={color} text={{{sid}}}")
            continue
        lines.append(
            f"ellipse({x[idx]:.3f},{y[idx]:.3f},{major[idx]:.3f},{minor[idx]:.3f},"
            f"{math.degrees(theta[idx]):.3f}) # color={color} text={{{sid} area={area[idx]:.1f}}}"
        )
    return lines


def write_region(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ds9_header(title) + lines) + "\n", encoding="utf-8")


def write_csv(path: Path, classes: dict[str, Table], *, aperture_scale: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "class",
        "id",
        "x_physical",
        "y_physical",
        "major_aperture",
        "minor_aperture",
        "theta_deg",
        "aperture_area",
        "pu_mag",
        "pu_reason",
        "pu_aperture_fill_ratio",
        "pu_no_shape_supervision",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, table in classes.items():
            for idx in range(len(table)):
                major = float(table["ellipse_major_sigma"][idx]) * float(aperture_scale)
                minor = float(table["ellipse_minor_sigma"][idx]) * float(aperture_scale)
                writer.writerow(
                    {
                        "class": name,
                        "id": _source_id(table, idx),
                        "x_physical": float(table["base_SdssCentroid_x"][idx]),
                        "y_physical": float(table["base_SdssCentroid_y"][idx]),
                        "major_aperture": major,
                        "minor_aperture": minor,
                        "theta_deg": math.degrees(float(table["ellipse_theta"][idx])),
                        "aperture_area": math.pi * major * minor,
                        "pu_mag": float(table["pu_mag"][idx]) if "pu_mag" in table.colnames else "",
                        "pu_reason": str(table["pu_reason"][idx]) if "pu_reason" in table.colnames else "",
                        "pu_aperture_fill_ratio": (
                            float(table["pu_aperture_fill_ratio"][idx])
                            if "pu_aperture_fill_ratio" in table.colnames
                            else ""
                        ),
                        "pu_no_shape_supervision": (
                            bool(table["pu_no_shape_supervision"][idx])
                            if "pu_no_shape_supervision" in table.colnames
                            else False
                        ),
                    }
                )


def attach_and_prepare_refit_table(table: Table, args: argparse.Namespace) -> tuple[Table, dict[str, int]]:
    """Attach refit aperture radii and apply the preprocessing aperture cap."""

    table = attach_kron_refit_radius(
        table,
        args.refit_csv,
        radius_column="proxy_nan0_flux_aperture_radius",
        good_column="proxy_nan0_good",
    )
    stats = {
        "original_rows": int(len(table)),
        "matched_rows": int(np.count_nonzero(np.asarray(table["pu_refit_kron_radius_matched"], dtype=bool))),
        "aperture_capped_to_official": 0,
    }
    if bool(args.cap_refit_aperture_to_official) and "catalog_KronFlux_radius" in table.colnames:
        radius = np.asarray(table["pu_refit_kron_radius"], dtype=np.float32).copy()
        official_kron = np.asarray(table["catalog_KronFlux_radius"], dtype=np.float32)
        official_aperture = official_kron * float(args.official_aperture_scale)
        cap = (
            np.asarray(table["pu_refit_kron_radius_matched"], dtype=bool)
            & np.isfinite(radius)
            & np.isfinite(official_aperture)
            & (official_aperture > 0.0)
            & (radius > official_aperture)
        )
        stats["aperture_capped_to_official"] = int(np.count_nonzero(cap))
        radius[cap] = official_aperture[cap]
        table["pu_refit_kron_radius"] = radius
    return table, stats


def read_image_and_origin(path: Path) -> tuple[np.ndarray, tuple[float, float]]:
    with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
        hdu = None
        for item in hdul:
            if getattr(item, "data", None) is not None and np.asarray(item.data).ndim == 2:
                hdu = item
                break
        if hdu is None:
            raise ValueError(f"no 2D image HDU found in {path}")
        image = np.asarray(hdu.data, dtype=np.float32)
        origin = (-float(hdu.header.get("LTV1", 0.0)), -float(hdu.header.get("LTV2", 0.0)))
    return image, origin


def plot_overlay(path: Path, image: np.ndarray, origin: tuple[float, float], classes: dict[str, Table], args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Patch

    finite = image[np.isfinite(image)]
    vmin, vmax = ZScaleInterval().get_limits(finite if finite.size else image)
    fig, ax = plt.subplots(figsize=(11, 11), dpi=160)
    ax.imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    rng = np.random.default_rng(23)
    for name, table in classes.items():
        if len(table) == 0:
            continue
        order = np.arange(len(table))
        if len(order) > int(args.max_overlay_per_class):
            order = rng.choice(order, size=int(args.max_overlay_per_class), replace=False)
        # Draw large apertures first so small ones remain visible.
        area = (
            np.asarray(table["ellipse_major_sigma"], dtype=float)
            * np.asarray(table["ellipse_minor_sigma"], dtype=float)
        )
        order = order[np.argsort(area[order])[::-1]]
        color = COLORS.get(name, "white")
        for idx in order:
            x = float(table["base_SdssCentroid_x"][idx]) - float(origin[0])
            y = float(table["base_SdssCentroid_y"][idx]) - float(origin[1])
            major = float(table["ellipse_major_sigma"][idx]) * float(args.aperture_scale)
            minor = float(table["ellipse_minor_sigma"][idx]) * float(args.aperture_scale)
            theta = math.degrees(float(table["ellipse_theta"][idx]))
            base_area = math.pi * float(table["ellipse_major_sigma"][idx]) * float(table["ellipse_minor_sigma"][idx])
            if base_area > float(args.large_area_as_point):
                ax.plot([x], [y], marker="o", ms=2.8, color=color, alpha=0.75)
            else:
                ax.add_patch(
                    Ellipse(
                        (x, y),
                        width=2.0 * major,
                        height=2.0 * minor,
                        angle=theta,
                        fill=False,
                        edgecolor=color,
                        linewidth=0.55,
                        alpha=0.65,
                    )
                )
    ax.set_title(f"{args.band} tract {args.tract} patch {args.patch} PU Kron aperture x{args.aperture_scale}")
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.set_xlabel("x local pixel")
    ax.set_ylabel("y local pixel")
    ax.legend(
        handles=[Patch(facecolor=COLORS[name], label=f"{name} n={len(table)}") for name, table in classes.items()],
        loc="upper right",
        framealpha=0.85,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*'second' did not parse as fits unit.*")
        table = Table.read(args.catalog, hdu=int(args.catalog_hdu))
    table, refit_stats = attach_and_prepare_refit_table(table, args)
    if not bool(args.include_unmatched_refit_as_ignore):
        matched = np.asarray(table["pu_refit_kron_radius_matched"], dtype=bool)
        table = table[matched]

    runtime = argparse.Namespace(
        coadd_root=str(args.calexp.parents[3]) if False else str(args.calexp.parents[3]),
        catalog_root=str(args.catalog.parents[3]) if False else str(args.catalog.parents[3]),
        band_catalog_root=str(args.catalog.parents[3]) if False else str(args.catalog.parents[3]),
        tract=str(args.tract),
        catalog_hdu=int(args.catalog_hdu),
        x_col="base_SdssCentroid_x",
        y_col="base_SdssCentroid_y",
        target_shape_source="kron",
        source_filter=args.source_filter,
        kron_refit_csv=None,
        kron_refit_radius_column="proxy_nan0_flux_aperture_radius",
        kron_refit_good_column="proxy_nan0_good",
        require_kron_refit_match=True,
        mag_column="ext_photometryKron_KronFlux_instFlux",
        zeropoint=27.0,
        b_mag_min=15.0,
        b_mag_max=35.0,
        a_area_max=10000.0,
        a_faint_area_max=900.0,
        a_faint_mag_min=28.0,
        ap2_kron_abs_max=1.0,
        ap2_flux_column="base_CircularApertureFlux_6_0_instFlux",
        ap2_kron_flux_column="ext_photometryKron_KronFlux_instFlux",
        b_flags=("base_SdssShape_flag", "base_SdssCentroid_flag"),
        close_center_arcsec=0.5,
        axis_ratio_max=5.0,
        containment_threshold=0.80,
        drop_ellipse_area_min=40000.0,
        remeasure_ap2_kron_outliers=True,
        enable_strict_bright_center_only=bool(args.strict_bright_center_only),
        strict_bright_center_only_saturation_mags=None,
        ellipse_sigma=1.0,
        min_ellipse_axis=1.5,
        pixel_scale_arcsec=0.168,
        no_clean_nonfinite=False,
    )
    # Derive roots for the remeasurement path expected by astro_data_preprocessing.
    runtime.coadd_root = str(args.calexp.parents[3])
    runtime.catalog_root = str(args.catalog.parents[3])
    runtime.band_catalog_root = str(args.catalog.parents[3])

    clean, center, ignore, _all, result = _classify_pu_catalog(table, runtime, band=args.band, patch=args.patch)
    if bool(args.strict_bright_center_only):
        clean, center, strict_center = _move_bright_clean_to_center_only(clean, center, runtime, band=args.band)
    else:
        strict_center = clean[:0].copy(copy_data=True)

    classes = {
        "clean": clean,
        "center_only": center,
        "strict_center_only": strict_center,
        "ignore": ignore,
    }
    all_lines: list[str] = []
    for name, cls_table in classes.items():
        lines = table_to_region_lines(
            cls_table,
            color=COLORS[name],
            aperture_scale=float(args.aperture_scale),
            large_area_as_point=float(args.large_area_as_point),
        )
        write_region(
            out_dir / f"{args.band}_{args.tract}_{args.patch.replace(',', '_')}_pu_{name}_kron_aperture_x{args.aperture_scale:g}.reg",
            f"{args.band} {args.tract} {args.patch} PU {name} Kron aperture x{args.aperture_scale:g}",
            lines,
        )
        all_lines.extend(lines)

    write_region(
        out_dir / f"{args.band}_{args.tract}_{args.patch.replace(',', '_')}_pu_all_classes_kron_aperture_x{args.aperture_scale:g}.reg",
        f"{args.band} {args.tract} {args.patch} PU all classes Kron aperture x{args.aperture_scale:g}",
        all_lines,
    )
    write_csv(
        out_dir / f"{args.band}_{args.tract}_{args.patch.replace(',', '_')}_pu_classes_kron_aperture_x{args.aperture_scale:g}.csv",
        classes,
        aperture_scale=float(args.aperture_scale),
    )
    image, origin = read_image_and_origin(args.calexp)
    plot_overlay(
        out_dir / f"{args.band}_{args.tract}_{args.patch.replace(',', '_')}_pu_kron_aperture_x{args.aperture_scale:g}_overlay.png",
        image,
        origin,
        classes,
        args,
    )

    print(
        "counts "
        + " ".join(f"{name}={len(cls_table)}" for name, cls_table in classes.items())
        + f" original_rows={refit_stats['original_rows']} classified_refit_rows={len(table)}"
        + f" matched_rows={refit_stats['matched_rows']}"
        + f" aperture_capped_to_official={refit_stats['aperture_capped_to_official']}"
        + f" remeasure={result.get('remeasure_ap2_kron', {})}"
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
