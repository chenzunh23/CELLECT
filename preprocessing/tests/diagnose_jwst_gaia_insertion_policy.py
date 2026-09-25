#!/usr/bin/env python3
"""Diagnose a stricter Gaia insertion policy for JWST catalog labels."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
from pathlib import Path
import sys
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cellect")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ZScaleInterval
from astropy.wcs import FITSFixedWarning
from matplotlib.patches import Ellipse
from scipy.spatial import cKDTree
import astropy.units as u


PIPELINE = Path(
    "/home/czh23/analysis/2026-09/2026-09-04/"
    "jwst_cosmos_selected_snr_full_regs/jwst_cosmos_snr_filter_and_regs.py"
)
DEFAULT_GAIA = Path("/home/czh23/CELLECT/output/gaia_dr3_cosmos_full.fits")
DEFAULT_OUT = Path("/home/czh23/analysis/2026-09/2026-09-07/jwst_gaia_insertion_policy")
DS = 4
GAIA_MAG_MAX = 22.0
MATCH_RADIUS_ARCSEC = 1.0

JOBS = (("0019", "f444w"), ("0004", "f277w"))
LABEL_COLORS = {
    "clean": "#00d060",
    "center_only": "#ffd400",
    "strict_center_only": "#00e5ff",
    "ignore": "#ff3030",
    "drop": "#b000ff",
    "unknown": "#ffffff",
}


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_pipe(pipe, pointing: str, band: str, out_dir: Path) -> None:
    point = str(pointing).zfill(4)
    pipe.POINTING = point
    pipe.RAW_ROOT = pipe.RAW_BASE / f"Pointing_{point}"
    pipe.BANDS = (band.lower(),)
    pipe.FULL_REG_BAND = band.lower()
    pipe.OUT_DIR = out_dir


def stretch_ds(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.float32), 0.0, 1.0
    sample = finite[:: max(1, finite.size // 1_000_000)]
    try:
        lo, hi = ZScaleInterval(contrast=0.25, krej=2.5, max_iterations=5).get_limits(sample)
    except Exception:
        lo, hi = np.nanpercentile(sample, [0.5, 99.7])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanpercentile(sample, [0.5, 99.7])
    safe = np.nan_to_num(image, nan=lo, posinf=hi, neginf=lo)
    return np.clip((safe - lo) / (hi - lo), 0.0, 1.0).astype(np.float32), float(lo), float(hi)


def load_gaia(path: Path) -> dict[str, np.ndarray]:
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        names = set(data.columns.names)
        out: dict[str, np.ndarray] = {}
        for name in [
            "source_id",
            "ra",
            "dec",
            "phot_g_mean_mag",
            "pmra",
            "pmra_error",
            "pmdec",
            "pmdec_error",
            "parallax",
            "parallax_error",
            "ruwe",
            "ref_epoch",
        ]:
            if name in names:
                out[name] = np.asarray(data[name])
        return out


def propagated_gaia_radec(gaia: dict[str, np.ndarray], obstime: Time) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ra = np.asarray(gaia["ra"], dtype=float).copy()
    dec = np.asarray(gaia["dec"], dtype=float).copy()
    pmra = np.asarray(gaia.get("pmra", np.full(ra.shape, np.nan)), dtype=float)
    pmdec = np.asarray(gaia.get("pmdec", np.full(ra.shape, np.nan)), dtype=float)
    ref_epoch = np.asarray(gaia.get("ref_epoch", np.full(ra.shape, 2016.0)), dtype=float)
    movable = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(pmra) & np.isfinite(pmdec)
    if np.any(movable):
        coords = SkyCoord(
            ra=ra[movable] * u.deg,
            dec=dec[movable] * u.deg,
            pm_ra_cosdec=pmra[movable] * u.mas / u.yr,
            pm_dec=pmdec[movable] * u.mas / u.yr,
            obstime=Time(ref_epoch[movable], format="jyear"),
        )
        moved = coords.apply_space_motion(new_obstime=obstime)
        ra[movable] = moved.ra.deg
        dec[movable] = moved.dec.deg
    return ra, dec, movable


def astrometric_evidence(gaia: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(gaia["ra"])
    parallax = np.asarray(gaia.get("parallax", np.full(n, np.nan)), dtype=float)
    parallax_error = np.asarray(gaia.get("parallax_error", np.full(n, np.nan)), dtype=float)
    pmra = np.asarray(gaia.get("pmra", np.full(n, np.nan)), dtype=float)
    pmra_error = np.asarray(gaia.get("pmra_error", np.full(n, np.nan)), dtype=float)
    pmdec = np.asarray(gaia.get("pmdec", np.full(n, np.nan)), dtype=float)
    pmdec_error = np.asarray(gaia.get("pmdec_error", np.full(n, np.nan)), dtype=float)

    parallax_snr = np.full(n, np.nan, dtype=float)
    okp = np.isfinite(parallax) & np.isfinite(parallax_error) & (parallax_error > 0.0)
    parallax_snr[okp] = parallax[okp] / parallax_error[okp]

    pm_snr = np.full(n, np.nan, dtype=float)
    okra = np.isfinite(pmra) & np.isfinite(pmra_error) & (pmra_error > 0.0)
    okde = np.isfinite(pmdec) & np.isfinite(pmdec_error) & (pmdec_error > 0.0)
    both = okra & okde
    pm_snr[both] = np.hypot(pmra[both] / pmra_error[both], pmdec[both] / pmdec_error[both])

    pm_abs = np.full(n, np.nan, dtype=float)
    okpm = np.isfinite(pmra) & np.isfinite(pmdec)
    pm_abs[okpm] = np.hypot(pmra[okpm], pmdec[okpm])
    evidence = (parallax_snr >= 3.0) | (pm_snr >= 3.0) | (pm_abs >= 3.0)
    return evidence, parallax_snr, pm_snr, pm_abs


def classify_full_image(pipe, cache: dict[str, np.ndarray], band: str, data: np.ndarray, wcs, pixscale: float):
    ny, nx = data.shape
    x, y = wcs.all_world2pix(np.asarray(cache["ra"], dtype=float), np.asarray(cache["dec"], dtype=float), 0)
    inside = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
    idx = np.flatnonzero(inside)
    nan_ignore = pipe.nan_component_center_mask(data, x[idx], y[idx])
    state = pipe.classify(
        cache,
        idx,
        band,
        use_psf=True,
        x_pix=x[idx],
        y_pix=y[idx],
        pixscale=pixscale,
        nan_component_ignore=nan_ignore,
    )
    final = np.full(len(cache["id"]), "outside", dtype=object)
    final[idx] = state["final"]
    return idx, x, y, state, final


def write_reg(path: Path, rows: list[str], color: str = "white") -> None:
    with path.open("w") as fh:
        fh.write("# Region file format: DS9 version 4.1\n")
        fh.write(
            f"global color={color} dashlist=8 3 width=1 font='helvetica 10 normal roman' "
            "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n"
        )
        fh.write("image\n")
        fh.writelines(rows)


def circle_row(x: float, y: float, radius: float, color: str, text: str) -> str:
    return f"circle({x + 1.0:.3f},{y + 1.0:.3f},{radius:.3f}) # color={color} text={{{text}}}\n"


def ellipse_row(
    x: float,
    y: float,
    a_arcsec: float,
    b_arcsec: float,
    theta: float,
    pixscale: float,
    color: str,
    text: str,
) -> str:
    return (
        f"ellipse({x + 1.0:.3f},{y + 1.0:.3f},{a_arcsec / pixscale:.3f},"
        f"{b_arcsec / pixscale:.3f},{theta:.3f}) # color={color} text={{{text}}}\n"
    )


def draw_open(ax, x: np.ndarray, y: np.ndarray, color: str, size: float, label: str, lw: float = 1.2) -> None:
    if len(x) == 0:
        return
    ax.scatter(x / DS, y / DS, s=size, marker="o", facecolors="none", edgecolors=color, linewidths=lw, label=label)


def draw_jwst_ellipse(ax, x: float, y: float, a_arcsec: float, b_arcsec: float, theta: float, pixscale: float, color: str, alpha: float = 0.85) -> None:
    ax.add_patch(
        Ellipse(
            (x / DS, y / DS),
            width=2.0 * a_arcsec / pixscale / DS,
            height=2.0 * b_arcsec / pixscale / DS,
            angle=theta,
            fill=False,
            ec=color,
            lw=0.75,
            alpha=alpha,
            zorder=6,
        )
    )


def process_job(pipe, gaia: dict[str, np.ndarray], out_dir: Path, pointing: str, band: str) -> dict[str, int | float | str]:
    configure_pipe(pipe, pointing, band, out_dir)
    raw_path = pipe.raw_fits_path(band)
    data, wcs, pixscale = pipe.image_hdu(raw_path)
    ny, nx = data.shape
    cache = pipe.load_catalog()
    idx, sx, sy, state, final = classify_full_image(pipe, cache, band, data, wcs, pixscale)
    idx_pos = {int(src_i): local_i for local_i, src_i in enumerate(idx)}

    g_ra, g_dec, pm_applied = propagated_gaia_radec(gaia, pipe.observation_time(raw_path))
    gmag = np.asarray(gaia["phot_g_mean_mag"], dtype=float)
    gid = np.asarray(gaia["source_id"])
    evidence, parallax_snr, pm_snr, pm_abs = astrometric_evidence(gaia)
    gx, gy = wcs.all_world2pix(g_ra, g_dec, 0)
    g_lt_22_in_image = np.isfinite(gmag) & (gmag < GAIA_MAG_MAX) & np.isfinite(gx) & np.isfinite(gy)
    g_lt_22_in_image &= (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
    gaia_idx = np.flatnonzero(g_lt_22_in_image)

    cat_xy_arcsec = np.column_stack([sx[idx] * pixscale, sy[idx] * pixscale]).astype(np.float32)
    tree = cKDTree(cat_xy_arcsec) if len(cat_xy_arcsec) else None
    psf_fwhm = float(pipe.PSF_FWHM_ARCSEC[band])

    accepted_rows: list[str] = []
    inserted_rows: list[str] = []
    matched_rows: list[str] = []
    complete_rows: list[str] = []
    policy_csv: list[dict[str, object]] = []
    pair_csv: list[dict[str, object]] = []
    accepted_mask = np.zeros(len(gaia_idx), dtype=bool)
    inserted_mask = np.zeros(len(gaia_idx), dtype=bool)
    skipped_matched_no_evidence = np.zeros(len(gaia_idx), dtype=bool)
    matched_any = np.zeros(len(gaia_idx), dtype=bool)
    matched_source_indices: set[int] = set()

    for pos, gi in enumerate(gaia_idx):
        matches_local: list[int] = []
        seps_arcsec: list[float] = []
        if tree is not None:
            query = np.asarray([gx[gi] * pixscale, gy[gi] * pixscale], dtype=np.float32)
            matches_local = [int(item) for item in tree.query_ball_point(query, r=MATCH_RADIUS_ARCSEC)]
            for local_i in matches_local:
                src_i = int(idx[local_i])
                seps_arcsec.append(float(math.hypot((sx[src_i] - gx[gi]) * pixscale, (sy[src_i] - gy[gi]) * pixscale)))
        order = np.argsort(seps_arcsec) if seps_arcsec else np.asarray([], dtype=int)
        matches_local = [matches_local[int(k)] for k in order]
        seps_arcsec = [seps_arcsec[int(k)] for k in order]
        if matches_local:
            matched_any[pos] = True
        has_astrometric_evidence = bool(evidence[gi])
        isolated_close = has_astrometric_evidence and len(matches_local) == 1 and seps_arcsec[0] < psf_fwhm
        nearest_src = int(idx[matches_local[0]]) if matches_local else -1
        nearest_label = str(final[nearest_src]) if nearest_src >= 0 else ""
        nearest_id = int(cache["id"][nearest_src]) if nearest_src >= 0 else -1
        nearest_sep = float(seps_arcsec[0]) if seps_arcsec else float("nan")

        gaia_text = (
            f"gaia_id={int(gid[gi])} G={float(gmag[gi]):.2f} "
            f"pm={float(pm_abs[gi]):.2f} pm_snr={float(pm_snr[gi]):.2f} "
            f"plx_snr={float(parallax_snr[gi]):.2f}"
        )
        if isolated_close:
            accepted_mask[pos] = True
            src_i = nearest_src
            color = LABEL_COLORS.get(nearest_label, LABEL_COLORS["unknown"])
            row = ellipse_row(
                float(sx[src_i]),
                float(sy[src_i]),
                float(cache["kron2_a"][src_i]),
                float(cache["kron2_b"][src_i]),
                float(cache["theta_world"][src_i]),
                pixscale,
                "green",
                f"accept_jwst_kron {gaia_text} jwst_id={nearest_id} label={nearest_label} sep={nearest_sep:.3f}arcsec",
            )
            accepted_rows.append(row)
            complete_rows.append(row)
            matched_source_indices.add(src_i)
        elif has_astrometric_evidence or not matches_local:
            inserted_mask[pos] = True
            row = circle_row(
                float(gx[gi]),
                float(gy[gi]),
                6.0,
                "cyan",
                f"gaia_strict_center_only {gaia_text} nearest_jwst_id={nearest_id} nearest_label={nearest_label} nearest_sep={nearest_sep:.3f}arcsec n_match={len(matches_local)}",
            )
            inserted_rows.append(row)
            complete_rows.append(row)
        else:
            skipped_matched_no_evidence[pos] = True

        for local_i, sep in zip(matches_local, seps_arcsec, strict=False):
            src_i = int(idx[local_i])
            matched_source_indices.add(src_i)
            label = str(final[src_i])
            local_state_i = idx_pos[src_i]
            color = color_name(label)
            row = ellipse_row(
                float(sx[src_i]),
                float(sy[src_i]),
                float(cache["kron2_a"][src_i]),
                float(cache["kron2_b"][src_i]),
                float(cache["theta_world"][src_i]),
                pixscale,
                color,
                f"gaia_match jwst_id={int(cache['id'][src_i])} label={label} sep={sep:.3f}arcsec G={float(gmag[gi]):.2f} gaia_id={int(gid[gi])}",
            )
            matched_rows.append(row)
            pair_csv.append(
                {
                    "pointing": pointing,
                    "band": band,
                    "gaia_source_id": int(gid[gi]),
                    "gaia_g": float(gmag[gi]),
                    "gaia_x": float(gx[gi]),
                    "gaia_y": float(gy[gi]),
                    "jwst_id": int(cache["id"][src_i]),
                    "jwst_x": float(sx[src_i]),
                    "jwst_y": float(sy[src_i]),
                    "sep_arcsec": float(sep),
                    "jwst_final": label,
                    "jwst_mag_auto": float(state["mag_auto"][local_state_i]),
                    "jwst_snr": float(state["snr"][local_state_i]),
                    "jwst_warn_flag": int(cache["warn_flag"][src_i]),
                    "accepted_official_kron": bool(isolated_close and src_i == nearest_src),
                }
            )

        policy_csv.append(
            {
                "pointing": pointing,
                "band": band,
                "gaia_source_id": int(gid[gi]),
                "gaia_g": float(gmag[gi]),
                "gaia_x": float(gx[gi]),
                "gaia_y": float(gy[gi]),
                "pm_applied": bool(pm_applied[gi]),
                "pm_abs_masyr": float(pm_abs[gi]),
                "pm_snr": float(pm_snr[gi]),
                "parallax_snr": float(parallax_snr[gi]),
                "astrometric_evidence": bool(evidence[gi]),
                "n_jwst_matches_1arcsec": int(len(matches_local)),
                "nearest_jwst_id": nearest_id,
                "nearest_jwst_final": nearest_label,
                "nearest_sep_arcsec": nearest_sep,
                "psf_fwhm_arcsec": psf_fwhm,
                "accept_official_kron": bool(isolated_close),
                "insert_gaia_strict_center": bool(inserted_mask[pos]),
                "skip_matched_no_astrometric_evidence": bool(skipped_matched_no_evidence[pos]),
            }
        )

    prefix = out_dir / f"pointing{pointing}_{band}_gaia_policy"
    write_reg(prefix.with_name(prefix.name + "_accepted_jwst_kron.reg"), accepted_rows, "green")
    write_reg(prefix.with_name(prefix.name + "_inserted_gaia_strict_center.reg"), inserted_rows, "cyan")
    write_reg(prefix.with_name(prefix.name + "_matched_jwst_labels.reg"), matched_rows, "white")
    write_reg(prefix.with_name(prefix.name + "_complete.reg"), complete_rows + matched_rows, "white")
    write_csv(prefix.with_name(prefix.name + "_gaia_policy.csv"), policy_csv)
    write_csv(prefix.with_name(prefix.name + "_matched_pairs.csv"), pair_csv)

    png_path = plot_png(
        out_dir,
        pointing,
        band,
        data,
        pixscale,
        gx[gaia_idx],
        gy[gaia_idx],
        gmag[gaia_idx],
        evidence[gaia_idx],
        accepted_mask,
        inserted_mask,
        matched_any,
        skipped_matched_no_evidence,
        cache,
        sx,
        sy,
        final,
        sorted(matched_source_indices),
    )
    summary = {
        "pointing": pointing,
        "band": band,
        "raw_fits": str(raw_path),
        "image_nx": int(nx),
        "image_ny": int(ny),
        "pixscale_arcsec": float(pixscale),
        "psf_fwhm_arcsec": psf_fwhm,
        "gaia_g_lt_22_in_image": int(len(gaia_idx)),
        "gaia_g_lt_22_stellar_candidates": int(np.count_nonzero(evidence[gaia_idx])),
        "gaia_g_lt_22_no_astrometric_evidence": int(np.count_nonzero(~evidence[gaia_idx])),
        "gaia_with_astrometric_evidence": int(np.count_nonzero(evidence[gaia_idx])),
        "gaia_pm_applied": int(np.count_nonzero(pm_applied[gaia_idx])),
        "gaia_matched_within_1arcsec": int(np.count_nonzero(matched_any)),
        "accepted_official_jwst_kron": int(np.count_nonzero(accepted_mask)),
        "inserted_gaia_strict_center": int(np.count_nonzero(inserted_mask)),
        "skipped_matched_no_astrometric_evidence": int(np.count_nonzero(skipped_matched_no_evidence)),
        "matched_jwst_sources_unique": int(len(matched_source_indices)),
        "png": str(png_path),
    }
    with prefix.with_name(prefix.name + "_summary.txt").open("w") as fh:
        for key, value in summary.items():
            fh.write(f"{key}: {value}\n")
        fh.write(f"rule: accept official JWST Kron only when exactly one JWST source is within {MATCH_RADIUS_ARCSEC:.1f} arcsec and its center is closer than the band PSF FWHM; otherwise insert Gaia as strict center only.\n")
        fh.write("gaia_candidate: Gaia DR3 source with finite G<22 inside the image. With astrometric evidence, accept isolated PSF-close JWST Kron or insert Gaia; without astrometric evidence, skip insertion when a JWST match exists and insert only when unmatched.\n")
    return summary


def color_name(label: str) -> str:
    return {
        "clean": "green",
        "center_only": "yellow",
        "strict_center_only": "cyan",
        "ignore": "red",
        "drop": "magenta",
    }.get(label, "white")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_png(
    out_dir: Path,
    pointing: str,
    band: str,
    data: np.ndarray,
    pixscale: float,
    gx: np.ndarray,
    gy: np.ndarray,
    gmag: np.ndarray,
    evidence: np.ndarray,
    accepted: np.ndarray,
    inserted: np.ndarray,
    matched_any: np.ndarray,
    skipped_matched_no_evidence: np.ndarray,
    cache: dict[str, np.ndarray],
    sx: np.ndarray,
    sy: np.ndarray,
    final: np.ndarray,
    matched_sources: list[int],
) -> Path:
    display, zlo, zhi = stretch_ds(np.asarray(data[::DS, ::DS], dtype=np.float32))
    ny, nx = data.shape
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=120)
    for ax in axes:
        ax.imshow(display, origin="lower", cmap="gray", interpolation="nearest")
        ax.set_xlim(0, nx / DS)
        ax.set_ylim(0, ny / DS)
        ax.set_axis_off()

    draw_open(axes[0], gx[matched_any], gy[matched_any], "#ffffff", 26, 'Gaia matched <=1"', 0.7)
    draw_open(axes[0], gx[evidence], gy[evidence], "#ff00ff", 62, "Gaia astrometric signal", 1.05)
    draw_open(axes[0], gx[accepted], gy[accepted], "#00d060", 96, "accept JWST Kron", 1.4)
    draw_open(axes[0], gx[inserted], gy[inserted], "#00e5ff", 86, "insert Gaia center", 1.25)
    draw_open(axes[0], gx[skipped_matched_no_evidence], gy[skipped_matched_no_evidence], "#ff8c00", 74, "matched, no astrometric signal: skip", 1.15)
    bright = gmag < 18.0
    mid = (gmag >= 18.0) & (gmag < 20.0)
    faint = (gmag >= 20.0) & (gmag < GAIA_MAG_MAX)
    axes[0].scatter(gx[faint] / DS, gy[faint] / DS, s=8, c="#d8d8d8", alpha=0.75, linewidths=0, label="20<=G<22")
    axes[0].scatter(gx[mid] / DS, gy[mid] / DS, s=14, c="#ffd400", alpha=0.80, linewidths=0, label="18<=G<20")
    axes[0].scatter(gx[bright] / DS, gy[bright] / DS, s=22, c="#ff3030", alpha=0.90, linewidths=0, label="G<18")
    axes[0].set_title(f"Pointing {pointing} {band.upper()}: Gaia G<22 stellar policy", fontsize=12, pad=12)
    axes[0].legend(loc="lower right", fontsize=9, framealpha=0.86)

    label_counts = {label: 0 for label in LABEL_COLORS}
    for src_i in matched_sources:
        label = str(final[src_i])
        label_counts[label] = label_counts.get(label, 0) + 1
        a = float(cache["kron2_a"][src_i])
        b = float(cache["kron2_b"][src_i])
        theta = float(cache["theta_world"][src_i])
        if not np.isfinite(a + b + theta) or a <= 0 or b <= 0:
            continue
        draw_jwst_ellipse(
            axes[1],
            float(sx[src_i]),
            float(sy[src_i]),
            a,
            b,
            theta,
            pixscale,
            LABEL_COLORS.get(label, LABEL_COLORS["unknown"]),
        )
    axes[1].scatter(gx / DS, gy / DS, s=18, marker="+", c="#00ff80", linewidths=0.8, alpha=0.88, label="Gaia G<22")
    handles = [
        plt.Line2D([0], [0], color=color, lw=2, label=f"{label} ({label_counts.get(label, 0)})")
        for label, color in LABEL_COLORS.items()
        if label_counts.get(label, 0)
    ]
    handles.append(plt.Line2D([0], [0], marker="+", color="#00ff80", lw=0, label="Gaia G<22"))
    axes[1].legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.86)
    axes[1].set_title(f'JWST source labels for Gaia matches within {MATCH_RADIUS_ARCSEC:.1f}"', fontsize=12, pad=12)
    fig.suptitle(
        f"Gaia DR3 G<22 stellar matching, DS={DS}, zscale=[{zlo:.3g},{zhi:.3g}], "
        f"accept JWST Kron only for isolated match closer than PSF FWHM",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), w_pad=1.8)
    out = out_dir / f"pointing{pointing}_{band}_gaia_policy_matching_ds{DS}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaia", type=Path, default=DEFAULT_GAIA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--jobs", nargs="*", default=[f"{p}:{b}" for p, b in JOBS], help="Jobs as POINTING:BAND.")
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=FITSFixedWarning)
    warnings.filterwarnings("ignore", message="ERFA function")
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = import_module(PIPELINE, "jwst_cosmos_snr_filter_and_regs")
    gaia = load_gaia(args.gaia.expanduser().resolve())
    summaries = []
    for item in args.jobs:
        pointing, band = item.split(":", 1)
        pointing = pointing.zfill(4)
        band = band.lower()
        print(f"[job] pointing={pointing} band={band}", flush=True)
        summaries.append(process_job(pipe, gaia, out_dir, pointing, band))
    write_csv(out_dir / "gaia_policy_job_summary.csv", summaries)
    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
