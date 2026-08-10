#!/usr/bin/env python3
"""Visualize preprocessing v3 refit/A/B/bright-label stages.

Example:
    conda run -n cellect python preprocessing/tests/visualize_filter_stages.py \
      --meas-catalog /data/shared/Subaru/9813/HSC-I/4,5/meas-HSC-I-9813-4,5.fits \
      --refit-csv /data/czh23/refit/9813/HSC-I/4,5/batch_heavyfp_kron_refit/batch_heavyfp_kron_refit.csv \
      --image /data/shared/Subaru/9813/HSC-I/4,5/calexp-HSC-I-9813-4,5.fits \
      --gaia-fits output/gaia_dr3_cosmos.fits \
      --band HSC-I --patch 4,5
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cellect")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.visualization import ZScaleInterval
from matplotlib.patches import Ellipse, Patch

from preprocessing.bright_ap2 import BrightAp2Config, classify_bright_ap2
from preprocessing.bright_label import BrightLabelConfig, label_bright_sources
from preprocessing.image_processing import (
    BrightRegionConfig,
    build_bright_components,
    read_fits_image,
    read_quality_mask,
)
from preprocessing.labels import SourceClass, SourceLabels
from preprocessing.meas_processing import MeasProcessingConfig, classify_meas_basics
from preprocessing.refit import (
    DirectRefitConfig,
    RefitConfig,
    attach_refit_geometry,
    attach_refit_radius_from_table,
    compute_kron_ellipse,
    run_refit_from_meas,
)
from preprocessing.utils.geometry import EllipseGeometry


CLASS_NAMES = {int(item): item.name.lower() for item in SourceClass}
CLASS_COLORS = {
    SourceClass.CLEAN: "#19d45a",
    SourceClass.WEAK_SHAPE: "#00c8ff",
    SourceClass.STRICT_CENTER_ONLY: "#ff33ff",
    SourceClass.RESTRICTED_BRIGHT_REGION: "#ff9900",
    SourceClass.ORDINARY_IGNORE: "#ff3030",
    SourceClass.STRICT_IGNORE: "#8a2be2",
    SourceClass.DROPPED: "#777777",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meas-catalog", type=Path, required=True)
    parser.add_argument("--refit-csv", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--gaia-fits", type=Path)
    parser.add_argument("--quality-mask-npz", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("output/preprocessing_tests/filter_stages"))
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patch", default="")
    parser.add_argument("--band", default="")
    parser.add_argument("--source-filter", default="nchild0")
    parser.add_argument("--zeropoint", type=float, default=27.0)
    parser.add_argument("--bright-mag-threshold", type=float, default=22.0)
    parser.add_argument("--run-refit-if-missing", action="store_true")
    parser.add_argument("--bright-mask-mode", default="log-lupton")
    parser.add_argument("--bright-threshold", type=float, default=2.99)
    parser.add_argument("--bright-dilate", type=int, default=2)
    parser.add_argument("--clip-threshold", type=float, default=3.0)
    parser.add_argument("--log-a", type=float, default=None)
    parser.add_argument("--anscombe-scale", type=float, default=1000.0)
    parser.add_argument("--origin", choices=("lower", "upper"), default="lower")
    parser.add_argument("--max-ellipses", type=int, default=6000)
    parser.add_argument("--large-area-point-only", type=float, default=10000.0)
    return parser.parse_args()


def validate_band_paths(args: argparse.Namespace) -> None:
    if not args.band:
        return
    mismatches: list[str] = []
    for label, path in (
        ("meas-catalog", args.meas_catalog),
        ("refit-csv", args.refit_csv),
        ("image", args.image),
    ):
        if path is None:
            continue
        path_text = str(path)
        if args.band not in path_text:
            mismatches.append(f"{label}={path_text}")
    if mismatches:
        joined = "\n  ".join(mismatches)
        raise ValueError(
            f"--band {args.band!r} does not match these input path(s):\n  {joined}\n"
            "Use the meas catalog, refit CSV, and image from the same band."
        )


def zscale_image(image: np.ndarray) -> np.ndarray:
    interval = ZScaleInterval()
    finite = np.asarray(image, dtype=np.float32)
    values = finite[np.isfinite(finite)]
    if values.size == 0:
        return np.zeros_like(finite, dtype=np.float32)
    vmin, vmax = interval.get_limits(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanpercentile(values, 1.0)), float(np.nanpercentile(values, 99.0))
    return np.clip((np.nan_to_num(finite, nan=vmin) - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)


def draw_sources(
    ax: plt.Axes,
    geom,
    mask: np.ndarray,
    *,
    color: str,
    label: str,
    max_ellipses: int,
    point_only_area: float,
    linewidth: float = 0.9,
    alpha: float = 0.95,
) -> int:
    indices = np.flatnonzero(np.asarray(mask, dtype=bool) & geom.valid())
    if indices.size > max_ellipses:
        indices = indices[:max_ellipses]
    for idx in indices:
        if float(geom.area[idx]) > float(point_only_area):
            ax.plot(float(geom.x[idx]), float(geom.y[idx]), marker="+", color=color, markersize=4, mew=linewidth)
            continue
        patch = Ellipse(
            (float(geom.x[idx]), float(geom.y[idx])),
            width=2.0 * float(geom.major[idx]),
            height=2.0 * float(geom.minor[idx]),
            angle=math.degrees(float(geom.theta[idx])),
            edgecolor=color,
            facecolor="none",
            linewidth=linewidth,
            alpha=alpha,
        )
        ax.add_patch(patch)
    return int(indices.size)


def draw_points(
    ax: plt.Axes,
    geom,
    mask: np.ndarray,
    *,
    color: str,
    max_points: int,
    linewidth: float = 1.0,
    markersize: float = 4.0,
) -> int:
    indices = np.flatnonzero(np.asarray(mask, dtype=bool) & np.isfinite(geom.x) & np.isfinite(geom.y))
    if indices.size > max_points:
        indices = indices[:max_points]
    if indices.size:
        ax.plot(geom.x[indices], geom.y[indices], linestyle="none", marker="+", color=color, markersize=markersize, mew=linewidth)
    return int(indices.size)


def refit_csv_visual_geometry(table: Table, refit_csv: Path | None, fallback_geom: EllipseGeometry) -> EllipseGeometry:
    """Build display geometry from batch refit CSV when available.

    The refit CSV stores ``x_image/y_image`` and the exact aperture scale used
    by the historical diagnostics.  Using these columns avoids empty overlays
    when meas centroid coordinates and the plotted FITS image coordinates are
    not in the same frame.
    """

    if refit_csv is None or not refit_csv.exists():
        return fallback_geom
    source_id_col = "id" if "id" in table.colnames else "source_id"
    id_to_index = {int(sid): idx for idx, sid in enumerate(table[source_id_col])}
    x = fallback_geom.x.copy()
    y = fallback_geom.y.copy()
    major = fallback_geom.major.copy()
    minor = fallback_geom.minor.copy()
    theta = fallback_geom.theta.copy()
    area = fallback_geom.area.copy()
    with refit_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                sid = int(row["source_id"])
                idx = id_to_index[sid]
                x_image = float(row["x_image"])
                y_image = float(row["y_image"])
                axis_a = float(row["axis_a"])
                axis_b = float(row["axis_b"])
                theta_deg = float(row["theta_deg"])
                initial_radius = float(row["initial_determinant_radius"])
                target_radius = float(row["proxy_nan0_flux_aperture_radius"])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(np.isfinite(v) for v in (x_image, y_image, axis_a, axis_b, theta_deg, initial_radius, target_radius)):
                continue
            if axis_a <= 0.0 or axis_b <= 0.0 or initial_radius <= 0.0 or target_radius <= 0.0:
                continue
            scale = target_radius / initial_radius
            x[idx] = x_image
            y[idx] = y_image
            major[idx] = axis_a * scale
            minor[idx] = axis_b * scale
            theta[idx] = math.radians(theta_deg)
            area[idx] = math.pi * major[idx] * minor[idx]
    return EllipseGeometry(x=x, y=y, major=major, minor=minor, theta=theta, area=area)


def setup_axis(ax: plt.Axes, display: np.ndarray, title: str, origin: str) -> None:
    ax.imshow(display, cmap="gray", origin=origin, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlim(0, display.shape[1])
    if origin == "lower":
        ax.set_ylim(0, display.shape[0])
    else:
        ax.set_ylim(display.shape[0], 0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def write_summary_csv(path: Path, table: Table, geom, stage, bright_ap2, bright_result, final_labels: SourceLabels, bright_scope: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source_id_col = "id" if "id" in table.colnames else "source_id"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "x",
                "y",
                "mag",
                "area",
                "axis_ratio",
                "source_ok",
                "refit_valid",
                "after_a",
                "after_b_basic",
                "bright_ap2_candidate",
                "bright_component_area",
                "ap2_mag",
                "kron_mag",
                "ap2_minus_kron_mag",
                "abs_ap2_minus_kron_mag",
                "bright_candidate",
                "ordinary_candidate",
                "final_class",
                "reason",
            ],
        )
        writer.writeheader()
        axis_ratio = geom.axis_ratio()
        for idx in range(len(table)):
            if not bool(bright_scope[idx]):
                continue
            writer.writerow(
                {
                    "source_id": int(table[source_id_col][idx]),
                    "x": float(geom.x[idx]) if np.isfinite(geom.x[idx]) else "",
                    "y": float(geom.y[idx]) if np.isfinite(geom.y[idx]) else "",
                    "mag": float(stage.mag[idx]) if np.isfinite(stage.mag[idx]) else "",
                    "area": float(geom.area[idx]) if np.isfinite(geom.area[idx]) else "",
                    "axis_ratio": float(axis_ratio[idx]) if np.isfinite(axis_ratio[idx]) else "",
                    "source_ok": bool(stage.source_ok[idx]),
                    "refit_valid": bool(stage.refit_valid[idx]),
                    "after_a": bool(stage.after_a[idx]),
                    "after_b_basic": bool(stage.after_b_basic[idx]),
                    "bright_ap2_candidate": bool(bright_ap2.candidate[idx]),
                    "bright_component_area": float(bright_ap2.component_area[idx]) if np.isfinite(bright_ap2.component_area[idx]) else "",
                    "ap2_mag": float(bright_ap2.ap2_mag[idx]) if np.isfinite(bright_ap2.ap2_mag[idx]) else "",
                    "kron_mag": float(bright_ap2.kron_mag[idx]) if np.isfinite(bright_ap2.kron_mag[idx]) else "",
                    "ap2_minus_kron_mag": float(bright_ap2.diff[idx]) if np.isfinite(bright_ap2.diff[idx]) else "",
                    "abs_ap2_minus_kron_mag": float(bright_ap2.absdiff[idx]) if np.isfinite(bright_ap2.absdiff[idx]) else "",
                    "bright_candidate": bool(stage.bright_candidate[idx]),
                    "ordinary_candidate": bool(stage.ordinary_candidate[idx]),
                    "final_class": CLASS_NAMES.get(int(final_labels.source_class[idx]), str(final_labels.source_class[idx])),
                    "reason": str(final_labels.reason[idx]),
                }
            )
        for x, y, sid, reason in zip(
            bright_result.strict_center_x,
            bright_result.strict_center_y,
            bright_result.strict_center_source_id,
            bright_result.strict_center_reason,
        ):
            writer.writerow(
                {
                    "source_id": int(sid),
                    "x": float(x),
                    "y": float(y),
                    "mag": "",
                    "area": "",
                    "axis_ratio": "",
                    "source_ok": "",
                    "refit_valid": "",
                    "after_a": "",
                    "after_b_basic": "",
                    "bright_ap2_candidate": "",
                    "bright_component_area": "",
                    "ap2_mag": "",
                    "kron_mag": "",
                    "ap2_minus_kron_mag": "",
                    "abs_ap2_minus_kron_mag": "",
                    "bright_candidate": True,
                    "ordinary_candidate": False,
                    "final_class": "strict_center_only",
                    "reason": str(reason),
                }
            )


def main() -> int:
    args = parse_args()
    validate_band_paths(args)
    stem_bits = [bit for bit in (args.tract, args.patch.replace(",", "_"), args.band) if bit]
    stem = "_".join(stem_bits) if stem_bits else args.image.stem
    out_dir = args.out_dir / (args.patch or "patch") / (args.band or "band")
    out_dir.mkdir(parents=True, exist_ok=True)

    table = Table.read(args.meas_catalog)
    refit_config = RefitConfig()
    if args.refit_csv is not None:
        table = attach_refit_geometry(table, args.refit_csv, refit_config)
    elif args.run_refit_if_missing:
        refit_table = run_refit_from_meas(args.meas_catalog, args.image, config=DirectRefitConfig(allow_missing_ltv=True))
        table = attach_refit_radius_from_table(table, refit_table, refit_config)
    else:
        raise ValueError("provide --refit-csv or pass --run-refit-if-missing")

    image, header = read_fits_image(args.image)
    display = zscale_image(image)
    geom = compute_kron_ellipse(table, refit_config)
    meas_config = MeasProcessingConfig(
        source_filter=args.source_filter,
        zeropoint=args.zeropoint,
        bright_mag_threshold=args.bright_mag_threshold,
    )
    stage = classify_meas_basics(table, config=meas_config, refit_config=refit_config)
    bright_scope = np.isfinite(stage.mag) & (stage.mag <= float(args.bright_mag_threshold))

    if args.log_a is not None:
        log_a = float(args.log_a)
    elif args.band == "NB1010":
        log_a = 100.0
    elif args.band == "NB0387":
        log_a = 3000.0
    else:
        log_a = 1000.0
    bright_config = BrightRegionConfig(
        mode=args.bright_mask_mode,
        threshold=args.bright_threshold,
        clip_threshold=args.clip_threshold,
        dilation=args.bright_dilate,
        log_a=log_a,
        anscombe_scale=args.anscombe_scale,
    )
    bright_mask, components = build_bright_components(image, config=bright_config)
    quality_mask = read_quality_mask(args.quality_mask_npz, image.shape)
    bright_ap2 = classify_bright_ap2(
        table,
        stage.bright_candidate,
        stage.labels,
        component_labels=components,
        config=BrightAp2Config(zeropoint=args.zeropoint),
        refit_config=refit_config,
    )
    gaia = Table.read(args.gaia_fits) if args.gaia_fits is not None else None
    bright_result = label_bright_sources(
        table,
        bright_ap2.candidate,
        bright_ap2.labels,
        bright_region=bright_mask,
        component_labels=components,
        gaia_table=gaia,
        image_header=header,
        quality_mask=quality_mask,
        mag=stage.mag,
        config=BrightLabelConfig(),
        refit_config=refit_config,
    )

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)
    for ax, title in zip(
        axes,
        ("refit valid", "A filter", "B basic + bright AP2", "bright label final"),
    ):
        setup_axis(ax, display, f"{stem}: {title}", args.origin)

    draw_sources(axes[0], geom, bright_scope & stage.source_ok & ~stage.refit_valid, color="#ff3030", label="invalid", max_ellipses=args.max_ellipses, point_only_area=args.large_area_point_only)
    draw_sources(axes[0], geom, bright_scope & stage.refit_valid, color="#19d45a", label="refit valid", max_ellipses=args.max_ellipses, point_only_area=args.large_area_point_only)
    axes[0].legend(handles=[Patch(color="#19d45a", label=f"valid {int((bright_scope & stage.refit_valid).sum())}"), Patch(color="#ff3030", label=f"invalid {int((bright_scope & stage.source_ok & ~stage.refit_valid).sum())}")], loc="upper right")

    draw_sources(axes[1], geom, bright_scope & stage.a_large, color="#ff3030", label="A large", max_ellipses=args.max_ellipses, point_only_area=args.large_area_point_only)
    draw_sources(axes[1], geom, bright_scope & stage.a_faint_large, color="#ff9900", label="A faint large", max_ellipses=args.max_ellipses, point_only_area=args.large_area_point_only)
    draw_sources(axes[1], geom, bright_scope & stage.after_a, color="#19d45a", label="after A", max_ellipses=args.max_ellipses, point_only_area=args.large_area_point_only)
    axes[1].legend(handles=[Patch(color="#19d45a", label=f"after A {int((bright_scope & stage.after_a).sum())}"), Patch(color="#ff3030", label=f"area>{meas_config.a_area_max:g} {int((bright_scope & stage.a_large).sum())}"), Patch(color="#ff9900", label=f"faint large {int((bright_scope & stage.a_faint_large).sum())}")], loc="upper right")

    draw_sources(axes[2], geom, bright_scope & stage.b_too_faint, color="#ff3030", label="too faint", max_ellipses=args.max_ellipses, point_only_area=args.large_area_point_only)
    draw_sources(axes[2], geom, bright_scope & stage.b_bad_axis, color="#8a2be2", label="axis", max_ellipses=args.max_ellipses, point_only_area=args.large_area_point_only)
    draw_sources(axes[2], geom, bright_scope & stage.b_close_dimmer, color="#ff9900", label="close dimmer", max_ellipses=args.max_ellipses, point_only_area=args.large_area_point_only)
    draw_sources(axes[2], geom, bright_ap2.removed_invalid, color="#777777", label="AP2 invalid", max_ellipses=args.max_ellipses, point_only_area=args.large_area_point_only, linewidth=1.1)
    draw_sources(axes[2], geom, bright_ap2.removed_outside, color="#ff9800", label="outside AP2", max_ellipses=args.max_ellipses, point_only_area=args.large_area_point_only, linewidth=1.1)
    draw_sources(axes[2], geom, bright_ap2.removed_small, color="#ff4fb8", label="small AP2", max_ellipses=args.max_ellipses, point_only_area=args.large_area_point_only, linewidth=1.1)
    draw_sources(axes[2], geom, bright_ap2.large_bright_region, color="#2459ff", label="large skip", max_ellipses=args.max_ellipses, point_only_area=args.large_area_point_only, linewidth=1.2)
    draw_sources(axes[2], geom, bright_ap2.candidate, color="#00eaff", label="bright AP2 kept", max_ellipses=args.max_ellipses, point_only_area=args.large_area_point_only, linewidth=1.2)
    axes[2].legend(handles=[
        Patch(color="#00eaff", label=f"bright AP2 kept {int(bright_ap2.candidate.sum())}"),
        Patch(color="#2459ff", label=f"large bright skip {int(bright_ap2.large_bright_region.sum())}"),
        Patch(color="#ff9800", label=f"outside |dmag|>=1 {int(bright_ap2.removed_outside.sum())}"),
        Patch(color="#ff4fb8", label=f"small |dmag|>=2 {int(bright_ap2.removed_small.sum())}"),
        Patch(color="#777777", label=f"AP2 invalid {int(bright_ap2.removed_invalid.sum())}"),
        Patch(color="#ff9900", label=f"close {int((bright_scope & stage.b_close_dimmer).sum())}"),
    ], loc="upper right")

    for cls, color in CLASS_COLORS.items():
        if cls == SourceClass.DROPPED:
            continue
        mask = bright_scope & bright_result.labels.mask(cls)
        if np.any(mask):
            if cls in {SourceClass.ORDINARY_IGNORE, SourceClass.STRICT_IGNORE, SourceClass.RESTRICTED_BRIGHT_REGION}:
                draw_points(axes[3], geom, mask, color=color, max_points=args.max_ellipses)
            else:
                draw_sources(axes[3], geom, mask, color=color, label=cls.name, max_ellipses=args.max_ellipses, point_only_area=args.large_area_point_only)
    if bright_result.strict_center_x.size:
        axes[3].scatter(bright_result.strict_center_x, bright_result.strict_center_y, c="#ff33ff", marker="+", s=30, linewidths=1.2)
    handles = []
    for cls, color in CLASS_COLORS.items():
        if cls == SourceClass.DROPPED:
            continue
        count = int(np.count_nonzero(bright_scope & bright_result.labels.mask(cls)))
        if cls == SourceClass.STRICT_CENTER_ONLY:
            count += int(bright_result.strict_center_x.size)
        if count:
            handles.append(Patch(color=color, label=f"{cls.name.lower()} {count}"))
    axes[3].legend(handles=handles, loc="upper right")

    panel_path = out_dir / f"{stem}_filter_stage_panel.png"
    fig.savefig(panel_path, dpi=180)
    plt.close(fig)

    final_path = out_dir / f"{stem}_final_class_overlay_zscale.png"
    fig_final, ax_final = plt.subplots(figsize=(10, 10), constrained_layout=True)
    setup_axis(ax_final, display, f"{stem}: final bright-source classes (Kron mag <= {args.bright_mag_threshold:g})", args.origin)
    final_handles = []
    for cls, color in CLASS_COLORS.items():
        if cls == SourceClass.DROPPED:
            continue
        mask = bright_scope & bright_result.labels.mask(cls)
        if not np.any(mask):
            continue
        if cls in {SourceClass.ORDINARY_IGNORE, SourceClass.STRICT_IGNORE, SourceClass.RESTRICTED_BRIGHT_REGION}:
            count = draw_points(
                ax_final,
                geom,
                mask,
                color=color,
                max_points=args.max_ellipses,
                linewidth=1.0,
                markersize=4.0,
            )
        else:
            count = draw_sources(
                ax_final,
                geom,
                mask,
                color=color,
                label=cls.name,
                max_ellipses=args.max_ellipses,
                point_only_area=args.large_area_point_only,
                linewidth=1.0,
            )
        if count:
            final_handles.append(Patch(color=color, label=f"{cls.name.lower()} {count}"))
    if bright_result.strict_center_x.size:
        ax_final.scatter(
            bright_result.strict_center_x,
            bright_result.strict_center_y,
            c=CLASS_COLORS[SourceClass.STRICT_CENTER_ONLY],
            marker="+",
            s=36,
            linewidths=1.4,
        )
        final_handles.append(Patch(color=CLASS_COLORS[SourceClass.STRICT_CENTER_ONLY], label=f"strict_center_only_external {bright_result.strict_center_x.size}"))
    ax_final.legend(handles=final_handles, loc="upper right")
    fig_final.savefig(final_path, dpi=180)
    plt.close(fig_final)

    summary_csv = out_dir / f"{stem}_filter_stage_summary.csv"
    write_summary_csv(summary_csv, table, geom, stage, bright_ap2, bright_result, bright_result.labels, bright_scope)

    bright_mask_path = out_dir / f"{stem}_bright_mask_components.png"
    fig2, ax2 = plt.subplots(figsize=(8, 8), constrained_layout=True)
    setup_axis(ax2, display, f"{stem}: bright mask components", args.origin)
    if np.any(bright_mask):
        overlay = np.ma.masked_where(~np.asarray(bright_mask, dtype=bool), bright_mask)
        ax2.imshow(overlay, cmap="autumn", alpha=0.35, origin=args.origin, interpolation="nearest")
    ax2.set_title(f"{stem}: bright mask mode={args.bright_mask_mode}, components={int(components.max())}")
    fig2.savefig(bright_mask_path, dpi=180)
    plt.close(fig2)

    print(f"wrote {panel_path}")
    print(f"wrote {final_path}")
    print(f"wrote {summary_csv}")
    print(f"wrote {bright_mask_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
