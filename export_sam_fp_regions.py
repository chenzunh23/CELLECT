"""Export LSST and Astro false-positive ellipses for the SAM 512x512 cutout.

The output is a DS9 region file in image coordinates.  False positives are
defined by the same greedy one-to-one center matching used elsewhere in this
workspace: predictions unmatched to leaf reference sources within 0.5 arcsec
are false positives.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from astropy.io import fits

LSST_UTILS = Path("/home/chenzunhao/segment-anything/lsst_pipeline/utils")
if str(LSST_UTILS) not in sys.path:
    sys.path.insert(0, str(LSST_UTILS))

from evaluate_centroid_matches import (  # noqa: E402
    DEFAULT_PIXEL_SCALE,
    DEFAULT_RADIUS_ARCSEC,
    _load_points,
    match_nearest_unique,
)

from astro_cellect2d import AstroUNet2D  # noqa: E402
from astro_train_eval import (  # noqa: E402
    AstroCutoutDataset,
    CutoutRecord,
    collate_cutouts,
    detect_centers,
    discover_cutout_records,
)


SAM_ASTRO_RECORD = "sam_x18204_y20924"
SAM_LSST_RECORD = "grid_r02_c04_x18204_y20924"


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _flux_to_mag(flux: np.ndarray, zeropoint: float) -> np.ndarray:
    mag = np.full(flux.shape, np.nan, dtype=float)
    good = np.isfinite(flux) & (flux > 0)
    mag[good] = float(zeropoint) - 2.5 * np.log10(flux[good])
    return mag


def _origin_from_fits(path: Path, *, hdu: str | int = "IMAGE") -> Tuple[float, float]:
    with fits.open(path) as hdul:
        if hdu in hdul:
            header = hdul[hdu].header
        else:
            header = hdul[0].header
        if "LTV1" not in header or "LTV2" not in header:
            return 0.0, 0.0
        return -float(header["LTV1"]), -float(header["LTV2"])


def _shape_prefixes(table) -> List[str]:
    prefixes: List[str] = []
    for prefix in (
        "base_SdssShape",
        "ext_shapeHSM_HsmSourceMoments",
        "ext_shapeHSM_HsmSourceMomentsRound",
        "modelfit_CModel_ellipse",
        "base_SdssShape_psf",
    ):
        if all(f"{prefix}_{suffix}" in table.colnames for suffix in ("xx", "yy", "xy")):
            prefixes.append(prefix)
    return prefixes


def _shape_prefix(table) -> Optional[str]:
    prefixes = _shape_prefixes(table)
    return prefixes[0] if prefixes else None


def _ellipse_from_moments(table, row_index: int, *, sigma: float) -> Tuple[float, float, float, str] | None:
    for prefix in _shape_prefixes(table):
        try:
            values = [table[f"{prefix}_{suffix}"][row_index] for suffix in ("xx", "yy", "xy")]
            if any(np.ma.is_masked(value) for value in values):
                continue
            xx, yy, xy = [float(value) for value in values]
        except Exception:
            continue
        cov = np.array([[xx, xy], [xy, yy]], dtype=float)
        if not np.all(np.isfinite(cov)):
            continue
        vals, vecs = np.linalg.eigh(cov)
        if np.any(vals <= 0) or not np.all(np.isfinite(vals)):
            continue
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        a = float(sigma * np.sqrt(vals[0]))
        b = float(sigma * np.sqrt(vals[1]))
        angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
        if np.all(np.isfinite([a, b, angle])) and a > 0 and b > 0:
            return a, b, angle, prefix
    return None


def _build_astro_model(args: argparse.Namespace, device: torch.device) -> AstroUNet2D:
    checkpoint = torch.load(args.astro_checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    embedding_dim = int(ckpt_args.get("embedding_dim", args.embedding_dim))
    base_channels = int(ckpt_args.get("base_channels", args.base_channels))
    model = AstroUNet2D(
        in_channels=len(args.astro_bands),
        seg_classes=3,
        confidence_levels=5,
        embedding_dim=embedding_dim,
        base_channels=base_channels,
        shape_channels=3,
    ).to(device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


def _match_local(ref_xy: np.ndarray, pred_xy: np.ndarray, *, radius_pix: float) -> np.ndarray:
    pred_used = np.zeros(pred_xy.shape[0], dtype=bool)
    if ref_xy.size == 0 or pred_xy.size == 0:
        return pred_used
    dist = np.sqrt(((ref_xy[:, None, :] - pred_xy[None, :, :]) ** 2).sum(axis=2))
    ref_idx, pred_idx = np.nonzero(dist <= float(radius_pix))
    order = np.argsort(dist[ref_idx, pred_idx], kind="stable")
    ref_used = np.zeros(ref_xy.shape[0], dtype=bool)
    for item in order:
        ri = int(ref_idx[item])
        pi = int(pred_idx[item])
        if ref_used[ri] or pred_used[pi]:
            continue
        ref_used[ri] = True
        pred_used[pi] = True
    return pred_used


def _astro_reference_centers(record: CutoutRecord, *, image_shape: Tuple[int, int], source_filter: str) -> np.ndarray:
    from astro_magnitude_completeness import load_reference_sources

    ref = load_reference_sources(
        record,
        image_shape=image_shape,
        source_filter=source_filter,
        flux_col="base_PsfFlux_instFlux",
        mag_zero_point=27.0,
        require_finite_shape=False,
    )
    return np.asarray(ref["centers"], dtype=np.float32)


def _reference_centers_local_from_lsst_crop(args: argparse.Namespace) -> np.ndarray:
    """Load the same cropped leaf-reference catalog used for LSST evaluation."""

    origin_x, origin_y = _origin_from_fits(args.lsst_background, hdu="IMAGE")
    ref_points = _load_points(
        args.astro_match_reference,
        x_col=args.lsst_ref_x,
        y_col=args.lsst_ref_y,
        role="ref",
        hdu=1,
        leaf_only=True,
        drop_flagged_centroids=True,
        require_science_model=False,
    )
    return np.stack([ref_points.x - origin_x, ref_points.y - origin_y], axis=1).astype(np.float32)


@torch.no_grad()
def collect_astro_fp_rows(args: argparse.Namespace) -> Tuple[List[dict], dict]:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = _build_astro_model(args, device)
    records = discover_cutout_records(
        args.astro_root,
        reference_dir=args.astro_reference_dir,
        cutout_dir=args.astro_cutout_dir,
        bands=args.astro_bands,
    )
    records = [record for record in records if record.name == args.astro_record]
    if len(records) != 1:
        raise RuntimeError(f"Expected one Astro record named {args.astro_record!r}, found {len(records)}")

    dataset = AstroCutoutDataset(
        records,
        fits_hdu=args.astro_fits_hdu,
        confidence_levels=5,
        ellipse_sigma=2.0,
        core_radius=2,
        shape_source=args.astro_shape_source,
        source_filter=args.source_filter,
        targets_dir=args.astro_root / "targets",
        augment=False,
    )
    batch = collate_cutouts([dataset[0]])
    image = batch["image"].to(device=device, dtype=torch.float32)
    outputs = model(image)
    pred_xy = detect_centers(
        outputs,
        threshold=args.astro_confidence_threshold,
        nms_radius=args.astro_nms_radius,
        confidence_score=args.astro_confidence_score,
    )[0]
    h, w = int(image.shape[-2]), int(image.shape[-1])
    if args.astro_match_reference is not None:
        ref_xy = _reference_centers_local_from_lsst_crop(args)
    else:
        ref_xy = _astro_reference_centers(records[0], image_shape=(h, w), source_filter=args.source_filter)
    pred_used = _match_local(ref_xy, pred_xy, radius_pix=args.match_radius_arcsec / args.pixel_scale)

    shape = outputs["shape"][0].detach().cpu().numpy().astype(np.float32)
    rows: List[dict] = []
    for fp_idx in np.flatnonzero(~pred_used):
        x, y = pred_xy[int(fp_idx)]
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            continue
        major = float(shape[0, yi, xi])
        minor = float(shape[1, yi, xi]) if shape.shape[0] > 1 else major
        theta = float(shape[2, yi, xi]) if shape.shape[0] > 2 else 0.0
        if not np.isfinite(major):
            major = 1.5
        if not np.isfinite(minor):
            minor = major
        if not np.isfinite(theta):
            theta = 0.0
        rows.append(
            {
                "method": "astro",
                "group": "astro_all",
                "color": args.astro_color,
                "source_id": f"astro_fp_{int(fp_idx)}",
                "table_index": "",
                "parent": "",
                "mag": "",
                "flux": "",
                "global_x": "",
                "global_y": "",
                "image_x": float(x) + 1.0,
                "image_y": float(y) + 1.0,
                "ellipse_a_pix": max(abs(major) * args.ellipse_sigma, args.min_ellipse_axis),
                "ellipse_b_pix": max(abs(minor) * args.ellipse_sigma, args.min_ellipse_axis),
                "ellipse_angle_deg": math.degrees(theta),
                "ellipse_source": "AstroUNet2D.shape",
            }
        )
    summary = {
        "reference_count": int(ref_xy.shape[0]),
        "prediction_count": int(pred_xy.shape[0]),
        "matched_count": int(pred_used.sum()),
        "false_positive_count": int((~pred_used).sum()),
    }
    return rows, summary


def _lsst_prediction_path(run_dir: Path, band: str) -> Path:
    meas = run_dir / "measure" / band / "deepCoadd_meas.fits"
    if meas.exists():
        return meas
    deblend = run_dir / "deblend" / "deepCoadd_deblendedFlux.fits"
    if deblend.exists():
        return deblend
    raise FileNotFoundError(f"no measurement/deblend catalog found under {run_dir}")


def _lsst_group(mag: float) -> Tuple[str, str]:
    if np.isfinite(mag) and mag < 27.5:
        return "lsst_mag_lt_27p5", "orange"
    if np.isfinite(mag) and mag < 29.0:
        return "lsst_mag_27p5_29", "gold"
    return "lsst_mag_ge_29_or_nan", "yellow"


def collect_lsst_fp_rows(args: argparse.Namespace) -> Tuple[List[dict], dict]:
    prediction = args.lsst_prediction or _lsst_prediction_path(args.lsst_run_dir, args.lsst_band)
    radius_pix = float(args.match_radius_arcsec) / float(args.pixel_scale)
    ref_points = _load_points(
        args.lsst_reference,
        x_col=args.lsst_ref_x,
        y_col=args.lsst_ref_y,
        role="ref",
        hdu=1,
        leaf_only=True,
        drop_flagged_centroids=True,
        require_science_model=False,
    )
    pred_points = _load_points(
        prediction,
        x_col=args.lsst_pred_x,
        y_col=args.lsst_pred_y,
        role="pred",
        hdu=1,
        leaf_only=True,
        drop_flagged_centroids=True,
        require_science_model=True,
    )
    matches, _, pred_used = match_nearest_unique(ref_points, pred_points, radius_pix)

    table = pred_points.table
    if args.lsst_flux_col not in table.colnames:
        raise KeyError(f"{prediction} missing flux column {args.lsst_flux_col!r}")
    flux = np.asarray(table[args.lsst_flux_col], dtype=float)[pred_points.table_indices]
    mag = _flux_to_mag(flux, args.lsst_pred_mag_zero_point)
    origin_x, origin_y = _origin_from_fits(args.lsst_background, hdu="IMAGE")

    rows: List[dict] = []
    missing_ellipse = 0
    for filtered_idx in np.flatnonzero(~pred_used):
        table_idx = int(pred_points.table_indices[int(filtered_idx)])
        ellipse = _ellipse_from_moments(table, table_idx, sigma=args.ellipse_sigma)
        if ellipse is None:
            missing_ellipse += 1
            if not args.fallback_point_ellipse:
                continue
            a = b = float(args.min_ellipse_axis)
            angle = 0.0
            ellipse_source = "fallback_point_ellipse"
        else:
            a, b, angle, ellipse_source = ellipse
        group, color = _lsst_group(float(mag[int(filtered_idx)]))
        gx = float(pred_points.x[int(filtered_idx)])
        gy = float(pred_points.y[int(filtered_idx)])
        source_id = int(pred_points.ids[int(filtered_idx)])
        parent = int(table["parent"][table_idx]) if "parent" in table.colnames else ""
        rows.append(
            {
                "method": "lsst",
                "group": group,
                "color": color,
                "source_id": source_id,
                "table_index": table_idx,
                "parent": parent,
                "mag": "" if not np.isfinite(mag[int(filtered_idx)]) else float(mag[int(filtered_idx)]),
                "flux": "" if not np.isfinite(flux[int(filtered_idx)]) else float(flux[int(filtered_idx)]),
                "global_x": gx,
                "global_y": gy,
                "image_x": gx - origin_x + 1.0,
                "image_y": gy - origin_y + 1.0,
                "ellipse_a_pix": max(float(a), args.min_ellipse_axis),
                "ellipse_b_pix": max(float(b), args.min_ellipse_axis),
                "ellipse_angle_deg": float(angle),
                "ellipse_source": ellipse_source,
            }
        )
    summary = {
        "reference_count": int(ref_points.n),
        "prediction_count": int(pred_points.n),
        "matched_count": int(len(matches)),
        "false_positive_count": int(np.count_nonzero(~pred_used)),
        "exported_false_positive_count": len(rows),
        "missing_ellipse_count": int(missing_ellipse),
        "prediction": str(prediction),
    }
    return rows, summary


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "group",
        "color",
        "source_id",
        "table_index",
        "parent",
        "mag",
        "flux",
        "global_x",
        "global_y",
        "image_x",
        "image_y",
        "ellipse_a_pix",
        "ellipse_b_pix",
        "ellipse_angle_deg",
        "ellipse_source",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_reg(path: Path, rows: Sequence[dict], *, show_text: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("# Region file format: DS9 version 4.1\n")
        handle.write('global dashlist=8 3 width=2 font="helvetica 12 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        handle.write("image\n")
        for row in rows:
            text = ""
            if show_text:
                mag = row["mag"] if row["mag"] != "" else "nan"
                text = f" text={{{row['method']} {row['source_id']} mag={mag}}}"
            handle.write(
                f"ellipse({float(row['image_x']):.3f},{float(row['image_y']):.3f},"
                f"{float(row['ellipse_a_pix']):.3f},{float(row['ellipse_b_pix']):.3f},"
                f"{float(row['ellipse_angle_deg']):.2f}) "
                f"# color={row['color']} tag={{{row['method']}}} tag={{{row['group']}}}{text}\n"
            )


def write_outputs(args: argparse.Namespace, rows: Sequence[dict], summary: dict) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_reg = args.output_dir / "sam_512_fp_all_lsst_astro.reg"
    all_csv = args.output_dir / "sam_512_fp_all_lsst_astro.csv"
    _write_reg(all_reg, rows, show_text=args.show_text)
    _write_csv(all_csv, rows)

    for group in sorted({str(row["group"]) for row in rows}):
        group_rows = [row for row in rows if str(row["group"]) == group]
        _write_reg(args.output_dir / f"{group}.reg", group_rows, show_text=args.show_text)
        _write_csv(args.output_dir / f"{group}.csv", group_rows)

    summary = dict(summary)
    summary["all_reg"] = str(all_reg)
    summary["all_csv"] = str(all_csv)
    summary["group_counts"] = {
        group: int(sum(1 for row in rows if str(row["group"]) == group))
        for group in sorted({str(row["group"]) for row in rows})
    }
    summary_path = args.output_dir / "sam_512_fp_region_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export SAM cutout LSST/Astro false-positive 3sigma ellipses to DS9 regions.")
    parser.add_argument("--output-dir", type=Path, default=Path("output/sam_fp_regions"))
    parser.add_argument("--source-filter", choices=("nchild0", "all", "parent", "leaf_child"), default="nchild0")
    parser.add_argument("--pixel-scale", type=float, default=DEFAULT_PIXEL_SCALE)
    parser.add_argument("--match-radius-arcsec", type=float, default=DEFAULT_RADIUS_ARCSEC)
    parser.add_argument("--ellipse-sigma", type=float, default=3.0)
    parser.add_argument("--min-ellipse-axis", type=float, default=1.5)
    parser.add_argument("--fallback-point-ellipse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-text", action="store_true")

    parser.add_argument("--astro-root", type=Path, default=Path("output/hsc_astro_preprocessed"))
    parser.add_argument("--astro-record", default=SAM_ASTRO_RECORD)
    parser.add_argument("--astro-reference-dir", type=Path, default=None)
    parser.add_argument("--astro-cutout-dir", type=Path, default=None)
    parser.add_argument("--astro-checkpoint", type=Path, default=Path("output/astro_hsc_i_cellect_conf/best.pt"))
    parser.add_argument("--astro-bands", nargs="+", default=("HSC-I",))
    parser.add_argument("--astro-fits-hdu", type=int, default=1)
    parser.add_argument("--astro-shape-source", choices=("kron", "sdss", "circular_kron"), default="sdss")
    parser.add_argument("--astro-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--astro-confidence-score", choices=("cellect", "raw", "ordinal_prob"), default="cellect")
    parser.add_argument("--astro-nms-radius", type=int, default=1)
    parser.add_argument(
        "--astro-match-reference",
        type=Path,
        default=None,
        help="Reference catalog for Astro FP matching. Defaults to --lsst-reference so LSST and Astro use the same GT denominator.",
    )
    parser.add_argument("--astro-color", default="cyan")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument(
        "--lsst-run-dir",
        type=Path,
        default=Path(
            "/home/chenzunhao/segment-anything/lsst_pipeline/output/"
            "sam_cutout_lsst_fp_rerun/runs/grid_r02_c04_x18204_y20924/lsst"
        ),
    )
    parser.add_argument("--lsst-prediction", type=Path, default=None)
    parser.add_argument(
        "--lsst-reference",
        type=Path,
        default=Path(
            "/home/chenzunhao/segment-anything/lsst_pipeline/output/"
            "sam_cutout_lsst_fp_rerun/reference_catalogs/grid_r02_c04_x18204_y20924_meas.fits"
        ),
    )
    parser.add_argument(
        "--lsst-background",
        type=Path,
        default=Path(
            "/home/chenzunhao/segment-anything/lsst_pipeline/output/"
            "sam_cutout_lsst_fp_rerun/cutouts/grid_r02_c04_x18204_y20924/HSC-I/"
            "deepCoadd-HSC-I-9813-4,5.fits"
        ),
    )
    parser.add_argument("--lsst-band", default="HSC-I")
    parser.add_argument("--lsst-ref-x", default=None)
    parser.add_argument("--lsst-ref-y", default=None)
    parser.add_argument("--lsst-pred-x", default=None)
    parser.add_argument("--lsst-pred-y", default=None)
    parser.add_argument("--lsst-flux-col", default="base_PsfFlux_instFlux")
    parser.add_argument("--lsst-pred-mag-zero-point", type=float, default=31.4)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir = _expand(args.output_dir)
    args.astro_root = _expand(args.astro_root)
    args.astro_checkpoint = _expand(args.astro_checkpoint)
    args.astro_reference_dir = _expand(args.astro_reference_dir) if args.astro_reference_dir else None
    args.astro_cutout_dir = _expand(args.astro_cutout_dir) if args.astro_cutout_dir else None
    args.lsst_run_dir = _expand(args.lsst_run_dir)
    args.lsst_prediction = _expand(args.lsst_prediction) if args.lsst_prediction else None
    args.lsst_reference = _expand(args.lsst_reference)
    args.lsst_background = _expand(args.lsst_background)
    args.astro_match_reference = _expand(args.astro_match_reference) if args.astro_match_reference else args.lsst_reference

    lsst_rows, lsst_summary = collect_lsst_fp_rows(args)
    astro_rows, astro_summary = collect_astro_fp_rows(args)
    rows = [*lsst_rows, *astro_rows]
    summary = {
        "lsst": lsst_summary,
        "astro": astro_summary,
        "ellipse_sigma": float(args.ellipse_sigma),
        "match_radius_arcsec": float(args.match_radius_arcsec),
        "pixel_scale": float(args.pixel_scale),
    }
    write_outputs(args, rows, summary)


if __name__ == "__main__":
    main()
