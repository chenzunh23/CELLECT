"""Evaluate source detection quality by centroid matching.

The metric is a one-to-one nearest-neighbor match between a reference catalog and
a prediction catalog.  By default it uses a 0.5 arcsec radius and a 0.168
arcsec/pixel scale, matching the HSC cutouts used in this experiment.  Optional
diagnostic outputs include FP/FN/GT CSV files, per-source stamps, and parent-mask
overlays for debugging mismatches.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy.table import Table


DEFAULT_SDSS_X = "base_SdssCentroid_x"
DEFAULT_SDSS_Y = "base_SdssCentroid_y"
DEFAULT_PIXEL_SCALE = 0.168
DEFAULT_RADIUS_ARCSEC = 0.5


@dataclass(frozen=True)
class SourcePoints:
    """Filtered source positions plus their original table indices."""

    ids: np.ndarray
    x: np.ndarray
    y: np.ndarray
    table_indices: np.ndarray
    x_col: str
    y_col: str
    table: Table

    @property
    def n(self) -> int:
        return int(self.x.size)


def _resolve_prediction_path(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / "deblend" / "deepCoadd_deblendedFlux.fits"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"{path} is not a catalog FITS and {candidate} does not exist")


def _choose_position_columns(table: Table, x_col: str | None, y_col: str | None, *, role: str) -> tuple[str, str]:
    """Pick centroid columns, preferring SDSS centroids and then deblend peaks."""
    if x_col is not None or y_col is not None:
        if x_col is None or y_col is None:
            raise ValueError(f"pass both --{role}-x and --{role}-y, or neither")
        return x_col, y_col

    candidates = [
        (DEFAULT_SDSS_X, DEFAULT_SDSS_Y),
        ("deblend_peak_center_x", "deblend_peak_center_y"),
        ("deblend_psfCenter_x", "deblend_psfCenter_y"),
        ("base_NaiveCentroid_x", "base_NaiveCentroid_y"),
    ]
    for x_name, y_name in candidates:
        if x_name in table.colnames and y_name in table.colnames:
            return x_name, y_name
    raise KeyError(f"no usable centroid columns found in {role} catalog")


def _centroid_flag_column(x_col: str) -> str | None:
    if x_col.endswith("_x"):
        return f"{x_col[:-2]}_flag"
    return None


def _is_sky_source(table: Table) -> np.ndarray:
    if "merge_footprint_sky" not in table.colnames:
        return np.zeros(len(table), dtype=bool)
    return np.asarray(table["merge_footprint_sky"], dtype=bool)


def _leaf_mask(table: Table) -> np.ndarray:
    if "deblend_nChild" in table.colnames:
        return np.asarray(table["deblend_nChild"], dtype=int) == 0
    if "parent" in table.colnames:
        return np.asarray(table["parent"], dtype=int) != 0
    return np.ones(len(table), dtype=bool)


def _science_model_mask(table: Table) -> np.ndarray:
    if "deblend_modelType" not in table.colnames:
        return np.ones(len(table), dtype=bool)
    values = np.asarray(table["deblend_modelType"])
    return np.array([str(value.decode() if isinstance(value, bytes) else value).strip() != "" for value in values])


def _load_points(
    path: Path,
    *,
    x_col: str | None,
    y_col: str | None,
    role: str,
    hdu: int,
    leaf_only: bool,
    drop_flagged_centroids: bool,
    require_science_model: bool,
) -> SourcePoints:
    table = Table.read(path, hdu=hdu)
    x_name, y_name = _choose_position_columns(table, x_col, y_col, role=role)
    for name in (x_name, y_name):
        if name not in table.colnames:
            raise KeyError(f"{path} missing required column {name!r}")

    x = np.asarray(table[x_name], dtype=float)
    y = np.asarray(table[y_name], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & ~_is_sky_source(table)

    if leaf_only:
        mask &= _leaf_mask(table)
    if require_science_model:
        mask &= _science_model_mask(table)
    if drop_flagged_centroids:
        flag_col = _centroid_flag_column(x_name)
        if flag_col is not None and flag_col in table.colnames:
            mask &= ~np.asarray(table[flag_col], dtype=bool)

    indices = np.flatnonzero(mask)
    ids = np.asarray(table["id"], dtype=np.int64)[indices] if "id" in table.colnames else indices.astype(np.int64)
    return SourcePoints(
        ids=ids,
        x=x[indices],
        y=y[indices],
        table_indices=indices,
        x_col=x_name,
        y_col=y_name,
        table=table,
    )


def match_nearest_unique(ref: SourcePoints, pred: SourcePoints, radius_pix: float) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Return greedy one-to-one centroid matches within ``radius_pix``.

    Candidate pairs are sorted by distance, so the closest remaining pair is
    kept first.  The returned masks are in filtered SourcePoints order and are
    later used for recall/completeness and precision/purity accounting.
    """
    if ref.n == 0 or pred.n == 0:
        return [], np.zeros(ref.n, dtype=bool), np.zeros(pred.n, dtype=bool)

    dx = ref.x[:, None] - pred.x[None, :]
    dy = ref.y[:, None] - pred.y[None, :]
    dist = np.hypot(dx, dy)
    ref_idx, pred_idx = np.nonzero(dist <= radius_pix)
    order = np.argsort(dist[ref_idx, pred_idx], kind="stable")

    ref_used = np.zeros(ref.n, dtype=bool)
    pred_used = np.zeros(pred.n, dtype=bool)
    matches: list[dict] = []
    for item in order:
        i = int(ref_idx[item])
        j = int(pred_idx[item])
        if ref_used[i] or pred_used[j]:
            continue
        ref_used[i] = True
        pred_used[j] = True
        matches.append(
            {
                "ref_id": int(ref.ids[i]),
                "pred_id": int(pred.ids[j]),
                "ref_x": float(ref.x[i]),
                "ref_y": float(ref.y[i]),
                "pred_x": float(pred.x[j]),
                "pred_y": float(pred.y[j]),
                "distance_pix": float(dist[i, j]),
                "ref_table_index": int(ref.table_indices[i]),
                "pred_table_index": int(pred.table_indices[j]),
            }
        )
    return matches, ref_used, pred_used


def _write_matches(path: Path, matches: list[dict], *, pixel_scale: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ref_id",
        "pred_id",
        "ref_x",
        "ref_y",
        "pred_x",
        "pred_y",
        "distance_pix",
        "distance_arcsec",
        "ref_table_index",
        "pred_table_index",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for match in matches:
            row = dict(match)
            row["distance_arcsec"] = float(match["distance_pix"]) * pixel_scale
            writer.writerow(row)


def _table_value(table: Table, row_index: int, name: str, default=""):
    if name not in table.colnames:
        return default
    value = table[name][row_index]
    if isinstance(value, bytes):
        return value.decode(errors="replace").strip()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parent_id(points: SourcePoints, point_index: int) -> int:
    row_index = int(points.table_indices[point_index])
    parent = _table_value(points.table, row_index, "parent", 0)
    source_id = int(points.ids[point_index])
    try:
        parent = int(parent)
    except Exception:
        parent = 0
    return parent if parent != 0 else source_id


def _shape_prefix(table: Table) -> str | None:
    for prefix in (
        "base_SdssShape",
        "ext_shapeHSM_HsmSourceMoments",
        "ext_shapeHSM_HsmSourceMomentsRound",
        "modelfit_CModel_ellipse",
    ):
        if all(f"{prefix}_{suffix}" in table.colnames for suffix in ("xx", "yy", "xy")):
            return prefix
    return None


def _ellipse_params(table: Table, row_index: int, *, scale: float = 2.0) -> tuple[float, float, float] | None:
    prefix = _shape_prefix(table)
    if prefix is None:
        return None
    try:
        values = [table[f"{prefix}_{suffix}"][row_index] for suffix in ("xx", "yy", "xy")]
        if any(np.ma.is_masked(value) for value in values):
            return None
        xx, yy, xy = [float(value) for value in values]
    except Exception:
        return None
    cov = np.array([[xx, xy], [xy, yy]], dtype=float)
    if not np.all(np.isfinite(cov)):
        return None
    vals, vecs = np.linalg.eigh(cov)
    if np.any(vals <= 0) or not np.all(np.isfinite(vals)):
        return None
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    width = float(2.0 * scale * np.sqrt(vals[0]))
    height = float(2.0 * scale * np.sqrt(vals[1]))
    return width, height, angle


def _source_rows(
    points: SourcePoints,
    *,
    status: str,
    used: np.ndarray,
    matches_by_point: dict[int, dict],
    role: str,
    pixel_scale: float,
    origin: tuple[int, int] | None,
) -> list[dict]:
    rows = []
    shape_prefix = _shape_prefix(points.table)
    ox, oy = origin if origin is not None else (0, 0)
    for i in range(points.n):
        row_index = int(points.table_indices[i])
        match = matches_by_point.get(i)
        parent = _table_value(points.table, row_index, "parent", 0)
        n_child = _table_value(points.table, row_index, "deblend_nChild", "")
        model_type = _table_value(points.table, row_index, "deblend_modelType", "")
        ellipse = _ellipse_params(points.table, row_index)
        rows.append(
            {
                "role": role,
                "status": status if not used[i] else "matched",
                "id": int(points.ids[i]),
                "parent": parent,
                "deblend_nChild": n_child,
                "deblend_modelType": model_type,
                "x": float(points.x[i]),
                "y": float(points.y[i]),
                "local_x": float(points.x[i] - ox),
                "local_y": float(points.y[i] - oy),
                "matched_id": match["pred_id"] if role == "reference" and match else match["ref_id"] if match else "",
                "distance_pix": match["distance_pix"] if match else "",
                "distance_arcsec": match["distance_pix"] * pixel_scale if match else "",
                "shape_prefix": shape_prefix or "",
                "ellipse_width_pix": ellipse[0] if ellipse else "",
                "ellipse_height_pix": ellipse[1] if ellipse else "",
                "ellipse_angle_deg": ellipse[2] if ellipse else "",
                "table_index": row_index,
            }
        )
    return rows


def _write_dict_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0])
    else:
        fieldnames = [
            "role",
            "status",
            "id",
            "parent",
            "deblend_nChild",
            "deblend_modelType",
            "x",
            "y",
            "local_x",
            "local_y",
            "matched_id",
            "distance_pix",
            "distance_arcsec",
            "shape_prefix",
            "ellipse_width_pix",
            "ellipse_height_pix",
            "ellipse_angle_deg",
            "table_index",
        ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_background(path: Path, *, hdu: str | int = "IMAGE") -> tuple[np.ndarray, tuple[int, int]]:
    with fits.open(path, memmap=False) as hdul:
        image_hdu = hdul[hdu]
        data = np.asarray(image_hdu.data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"{path}[{hdu}] is not a 2D image")
        header = image_hdu.header
        if "LTV1" in header and "LTV2" in header:
            origin = (-int(round(float(header["LTV1"]))), -int(round(float(header["LTV2"]))))
        else:
            origin = (0, 0)
    finite = data[np.isfinite(data)]
    if finite.size:
        try:
            lo, hi = ZScaleInterval().get_limits(finite)
        except Exception:
            lo, hi = np.percentile(finite, [1.0, 99.5])
        if not np.isfinite(hi - lo) or hi <= lo:
            hi = lo + 1.0
        data = np.clip(np.nan_to_num((data - lo) / (hi - lo)), 0, 1).astype(np.float32)
    else:
        data = np.zeros_like(data, dtype=np.float32)
    return data, origin


def _candidate_mask_dirs(prediction_arg: Path, pred_catalog: Path, explicit: Path | None) -> list[Path]:
    candidates = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    roots = []
    if prediction_arg.expanduser().is_dir():
        roots.append(prediction_arg.expanduser())
    roots.extend([pred_catalog.parent.parent, pred_catalog.parent.parent / "vis"])
    for root in roots:
        candidates.extend(
            [
                root / "merge_regions",
                root / "vis" / "merge_regions",
            ]
        )
    seen = set()
    out = []
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_dir():
            out.append(path)
    return out


def _mask_path_for_parent(mask_dirs: list[Path], parent_id: int) -> Path | None:
    name = f"merge_parent_{int(parent_id):08d}_mask.fits"
    for mask_dir in mask_dirs:
        path = mask_dir / name
        if path.exists():
            return path
    return None


def _read_mask(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                data = np.asarray(hdu.data)
                if data.ndim >= 2:
                    if data.ndim > 2:
                        data = data.reshape((-1, data.shape[-2], data.shape[-1]))[0]
                    return data != 0
    raise ValueError(f"no image mask found in {path}")


def _color_for_id(value: int) -> np.ndarray:
    cmap = plt.get_cmap("tab20", 20)
    return np.asarray(cmap(abs(int(value)) % 20)[:3], dtype=np.float32)


def _overlay_masks(rgb: np.ndarray, masks: list[tuple[int, np.ndarray]], *, alpha: float = 0.35) -> np.ndarray:
    out = np.array(rgb, copy=True)
    for parent_id, mask in masks:
        if mask.shape != out.shape[:2]:
            continue
        color = _color_for_id(parent_id)
        m = mask.astype(bool)
        out[m] = (1 - alpha) * out[m] + alpha * color
    return np.clip(out, 0, 1)


def _draw_points_and_ellipses(
    ax,
    points: SourcePoints,
    indices: np.ndarray,
    *,
    origin: tuple[int, int],
    color: str,
    marker: str,
    label: str,
    annotate: bool = False,
    draw_ellipses: bool = True,
    ellipse_max_major: float = 20.0,
    ellipse_max_axis_ratio: float = 8.0,
) -> None:
    if indices.size == 0:
        return
    ox, oy = origin
    xs = points.x[indices] - ox
    ys = points.y[indices] - oy
    ax.scatter(xs, ys, s=22, c=color, marker=marker, linewidths=0.8, label=label)
    for i, x, y in zip(indices, xs, ys):
        row_index = int(points.table_indices[int(i)])
        ellipse = _ellipse_params(points.table, row_index) if draw_ellipses else None
        if ellipse is not None:
            width, height, angle = ellipse
            major = max(width, height)
            minor = max(min(width, height), 1e-6)
            if major <= ellipse_max_major and major / minor <= ellipse_max_axis_ratio:
                patch = Ellipse(
                    (x, y),
                    width,
                    height,
                    angle=angle,
                    fill=False,
                    edgecolor=color,
                    linewidth=0.7,
                    alpha=0.85,
                    clip_on=True,
                )
                patch.set_clip_path(ax.patch)
                ax.add_patch(patch)
        if annotate:
            ax.text(x + 2, y + 2, str(int(points.ids[int(i)])), color=color, fontsize=6)


def _save_overview(
    path: Path,
    background: np.ndarray,
    *,
    origin: tuple[int, int],
    ref: SourcePoints,
    pred: SourcePoints,
    ref_indices: np.ndarray,
    pred_indices: np.ndarray,
    pred_mask_dirs: list[Path],
    title: str,
    draw_ellipses: bool,
    ellipse_max_major: float,
    ellipse_max_axis_ratio: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.dstack([background, background, background])
    masks = []
    for idx in pred_indices:
        parent_id = _parent_id(pred, int(idx))
        mask_path = _mask_path_for_parent(pred_mask_dirs, parent_id)
        if mask_path is not None:
            masks.append((parent_id, _read_mask(mask_path)))
    if masks:
        rgb = _overlay_masks(rgb, masks)

    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.imshow(rgb, origin="lower")
    _draw_points_and_ellipses(
        ax,
        ref,
        ref_indices,
        origin=origin,
        color="#ff4f5e",
        marker="x",
        label="reference",
        draw_ellipses=draw_ellipses,
        ellipse_max_major=ellipse_max_major,
        ellipse_max_axis_ratio=ellipse_max_axis_ratio,
    )
    _draw_points_and_ellipses(
        ax,
        pred,
        pred_indices,
        origin=origin,
        color="#55d8ff",
        marker="+",
        label="prediction",
        draw_ellipses=draw_ellipses,
        ellipse_max_major=ellipse_max_major,
        ellipse_max_axis_ratio=ellipse_max_axis_ratio,
    )
    ax.set_title(title)
    ax.set_xlabel("local x")
    ax.set_ylabel("local y")
    ax.set_xlim(0, background.shape[1])
    ax.set_ylim(0, background.shape[0])
    ax.legend(loc="upper right", framealpha=0.75)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_stamps(
    outdir: Path,
    background: np.ndarray,
    *,
    origin: tuple[int, int],
    points: SourcePoints,
    indices: np.ndarray,
    role: str,
    pred_mask_dirs: list[Path],
    max_stamps: int,
    stamp_size: int,
    draw_ellipses: bool,
    ellipse_max_major: float,
    ellipse_max_axis_ratio: float,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    half = max(4, int(stamp_size) // 2)
    height, width = background.shape
    for count, idx in enumerate(indices[:max_stamps], start=1):
        x = float(points.x[int(idx)] - origin[0])
        y = float(points.y[int(idx)] - origin[1])
        x0 = max(0, int(round(x)) - half)
        x1 = min(width, int(round(x)) + half)
        y0 = max(0, int(round(y)) - half)
        y1 = min(height, int(round(y)) + half)
        crop = background[y0:y1, x0:x1]
        rgb = np.dstack([crop, crop, crop])
        if role == "fp":
            parent_id = _parent_id(points, int(idx))
            mask_path = _mask_path_for_parent(pred_mask_dirs, parent_id)
            if mask_path is not None:
                mask = _read_mask(mask_path)[y0:y1, x0:x1]
                rgb = _overlay_masks(rgb, [(parent_id, mask)])

        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
        ax.imshow(rgb, origin="lower", extent=[x0, x1, y0, y1])
        _draw_points_and_ellipses(
            ax,
            points,
            np.array([idx], dtype=int),
            origin=origin,
            color="#55d8ff" if role == "fp" else "#ff4f5e",
            marker="+" if role == "fp" else "x",
            label=role.upper(),
            annotate=True,
            draw_ellipses=draw_ellipses,
            ellipse_max_major=ellipse_max_major,
            ellipse_max_axis_ratio=ellipse_max_axis_ratio,
        )
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_title(f"{role.upper()} source {int(points.ids[int(idx)])}")
        fig.savefig(outdir / f"{role}_{count:04d}_source_{int(points.ids[int(idx)]):08d}.png", dpi=140)
        plt.close(fig)


def _write_diagnostics(
    args: argparse.Namespace,
    *,
    ref: SourcePoints,
    pred: SourcePoints,
    matches: list[dict],
    ref_used: np.ndarray,
    pred_used: np.ndarray,
    pred_path: Path,
) -> dict:
    if args.diagnostic_dir is None:
        return {}

    outdir = args.diagnostic_dir.expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    ref_match_by_index = {int(match["ref_table_index"]): match for match in matches}
    pred_match_by_index = {int(match["pred_table_index"]): match for match in matches}
    ref_match_by_point = {
        i: ref_match_by_index[int(ref.table_indices[i])]
        for i in range(ref.n)
        if int(ref.table_indices[i]) in ref_match_by_index
    }
    pred_match_by_point = {
        i: pred_match_by_index[int(pred.table_indices[i])]
        for i in range(pred.n)
        if int(pred.table_indices[i]) in pred_match_by_index
    }

    ref_rows = _source_rows(
        ref,
        status="false_negative",
        used=ref_used,
        matches_by_point=ref_match_by_point,
        role="reference",
        pixel_scale=float(args.pixel_scale),
        origin=None,
    )
    pred_rows = _source_rows(
        pred,
        status="false_positive",
        used=pred_used,
        matches_by_point=pred_match_by_point,
        role="prediction",
        pixel_scale=float(args.pixel_scale),
        origin=None,
    )
    _write_dict_csv(outdir / "ground_truth_sources.csv", ref_rows)
    _write_dict_csv(outdir / "prediction_sources.csv", pred_rows)
    _write_dict_csv(outdir / "false_negatives.csv", [row for i, row in enumerate(ref_rows) if not ref_used[i]])
    _write_dict_csv(outdir / "false_positives.csv", [row for i, row in enumerate(pred_rows) if not pred_used[i]])

    outputs = {
        "ground_truth_csv": str(outdir / "ground_truth_sources.csv"),
        "prediction_csv": str(outdir / "prediction_sources.csv"),
        "false_negative_csv": str(outdir / "false_negatives.csv"),
        "false_positive_csv": str(outdir / "false_positives.csv"),
    }

    if args.background is None:
        return outputs

    background, origin = _read_background(args.background.expanduser(), hdu=args.background_hdu)
    ref_rows = _source_rows(
        ref,
        status="false_negative",
        used=ref_used,
        matches_by_point=ref_match_by_point,
        role="reference",
        pixel_scale=float(args.pixel_scale),
        origin=origin,
    )
    pred_rows = _source_rows(
        pred,
        status="false_positive",
        used=pred_used,
        matches_by_point=pred_match_by_point,
        role="prediction",
        pixel_scale=float(args.pixel_scale),
        origin=origin,
    )
    _write_dict_csv(outdir / "ground_truth_sources.csv", ref_rows)
    _write_dict_csv(outdir / "prediction_sources.csv", pred_rows)
    _write_dict_csv(outdir / "false_negatives.csv", [row for i, row in enumerate(ref_rows) if not ref_used[i]])
    _write_dict_csv(outdir / "false_positives.csv", [row for i, row in enumerate(pred_rows) if not pred_used[i]])

    mask_dirs = _candidate_mask_dirs(args.prediction, pred_path, args.pred_mask_dir)
    fn_indices = np.flatnonzero(~ref_used)
    fp_indices = np.flatnonzero(~pred_used)
    all_ref_indices = np.arange(ref.n, dtype=int)

    _save_overview(
        outdir / "ground_truth_sources.png",
        background,
        origin=origin,
        ref=ref,
        pred=pred,
        ref_indices=all_ref_indices,
        pred_indices=np.array([], dtype=int),
        pred_mask_dirs=[],
        title="Ground truth sources with ellipses",
        draw_ellipses=bool(args.draw_ellipses),
        ellipse_max_major=float(args.ellipse_max_major),
        ellipse_max_axis_ratio=float(args.ellipse_max_axis_ratio),
    )
    _save_overview(
        outdir / "false_negatives.png",
        background,
        origin=origin,
        ref=ref,
        pred=pred,
        ref_indices=fn_indices,
        pred_indices=np.array([], dtype=int),
        pred_mask_dirs=[],
        title="False negatives: unmatched ground truth",
        draw_ellipses=bool(args.draw_ellipses),
        ellipse_max_major=float(args.ellipse_max_major),
        ellipse_max_axis_ratio=float(args.ellipse_max_axis_ratio),
    )
    _save_overview(
        outdir / "false_positives.png",
        background,
        origin=origin,
        ref=ref,
        pred=pred,
        ref_indices=np.array([], dtype=int),
        pred_indices=fp_indices,
        pred_mask_dirs=mask_dirs,
        title="False positives: unmatched predictions with parent masks",
        draw_ellipses=bool(args.draw_ellipses),
        ellipse_max_major=float(args.ellipse_max_major),
        ellipse_max_axis_ratio=float(args.ellipse_max_axis_ratio),
    )
    _save_stamps(
        outdir / "false_negative_stamps",
        background,
        origin=origin,
        points=ref,
        indices=fn_indices,
        role="fn",
        pred_mask_dirs=[],
        max_stamps=int(args.max_diagnostic_stamps),
        stamp_size=int(args.stamp_size),
        draw_ellipses=bool(args.draw_ellipses),
        ellipse_max_major=float(args.ellipse_max_major),
        ellipse_max_axis_ratio=float(args.ellipse_max_axis_ratio),
    )
    _save_stamps(
        outdir / "false_positive_stamps",
        background,
        origin=origin,
        points=pred,
        indices=fp_indices,
        role="fp",
        pred_mask_dirs=mask_dirs,
        max_stamps=int(args.max_diagnostic_stamps),
        stamp_size=int(args.stamp_size),
        draw_ellipses=bool(args.draw_ellipses),
        ellipse_max_major=float(args.ellipse_max_major),
        ellipse_max_axis_ratio=float(args.ellipse_max_axis_ratio),
    )
    outputs.update(
        {
            "ground_truth_png": str(outdir / "ground_truth_sources.png"),
            "false_negative_png": str(outdir / "false_negatives.png"),
            "false_positive_png": str(outdir / "false_positives.png"),
            "false_negative_stamp_dir": str(outdir / "false_negative_stamps"),
            "false_positive_stamp_dir": str(outdir / "false_positive_stamps"),
            "prediction_mask_dirs": [str(path) for path in mask_dirs],
            "background_origin_xy": [int(origin[0]), int(origin[1])],
        }
    )
    return outputs


def evaluate(args: argparse.Namespace) -> dict:
    ref_path = args.reference.expanduser()
    pred_path = _resolve_prediction_path(args.prediction.expanduser())
    radius_pix = float(args.radius_arcsec) / float(args.pixel_scale)

    ref = _load_points(
        ref_path,
        x_col=args.ref_x,
        y_col=args.ref_y,
        role="ref",
        hdu=int(args.ref_hdu),
        leaf_only=bool(args.ref_leaf_only),
        drop_flagged_centroids=bool(args.drop_flagged_centroids),
        require_science_model=False,
    )
    pred = _load_points(
        pred_path,
        x_col=args.pred_x,
        y_col=args.pred_y,
        role="pred",
        hdu=int(args.pred_hdu),
        leaf_only=bool(args.pred_leaf_only),
        drop_flagged_centroids=bool(args.drop_flagged_centroids),
        require_science_model=bool(args.pred_require_science_model),
    )

    matches, ref_used, pred_used = match_nearest_unique(ref, pred, radius_pix)
    n_match = len(matches)
    recall = n_match / ref.n if ref.n else 0.0
    precision = n_match / pred.n if pred.n else 0.0

    distances = np.array([match["distance_pix"] for match in matches], dtype=float)
    result = {
        "reference_catalog": str(ref_path),
        "prediction_catalog": str(pred_path),
        "reference_xy_columns": [ref.x_col, ref.y_col],
        "prediction_xy_columns": [pred.x_col, pred.y_col],
        "pixel_scale_arcsec": float(args.pixel_scale),
        "match_radius_arcsec": float(args.radius_arcsec),
        "match_radius_pix": radius_pix,
        "reference_count": ref.n,
        "prediction_count": pred.n,
        "matched_count": n_match,
        "unmatched_reference_count": int(np.count_nonzero(~ref_used)),
        "unmatched_prediction_count": int(np.count_nonzero(~pred_used)),
        "recall": recall,
        "precision": precision,
        "f1": (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0,
        "distance_pix_median": float(np.median(distances)) if distances.size else None,
        "distance_pix_p90": float(np.percentile(distances, 90)) if distances.size else None,
        "distance_arcsec_median": float(np.median(distances) * args.pixel_scale) if distances.size else None,
        "distance_arcsec_p90": float(np.percentile(distances, 90) * args.pixel_scale) if distances.size else None,
        "filters": {
            "ref_leaf_only": bool(args.ref_leaf_only),
            "pred_leaf_only": bool(args.pred_leaf_only),
            "drop_flagged_centroids": bool(args.drop_flagged_centroids),
            "pred_require_science_model": bool(args.pred_require_science_model),
        },
    }

    if args.matches_csv is not None:
        _write_matches(args.matches_csv.expanduser(), matches, pixel_scale=float(args.pixel_scale))
        result["matches_csv"] = str(args.matches_csv.expanduser())

    diagnostic_outputs = _write_diagnostics(
        args,
        ref=ref,
        pred=pred,
        matches=matches,
        ref_used=ref_used,
        pred_used=pred_used,
        pred_path=pred_path,
    )
    if diagnostic_outputs:
        result["diagnostic_outputs"] = diagnostic_outputs

    if args.output is not None:
        args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.output.expanduser().write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate source-position recall/precision by one-to-one nearest-neighbor centroid matching. "
            "Default radius is 0.5 arcsec and default pixel scale is 0.168 arcsec/pixel."
        )
    )
    parser.add_argument("--reference", required=True, type=Path, help="Reference I-band meas catalog FITS.")
    parser.add_argument(
        "--prediction",
        required=True,
        type=Path,
        help="Prediction catalog FITS, or a run output root containing deblend/deepCoadd_deblendedFlux.fits.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON metrics output path.")
    parser.add_argument("--matches-csv", type=Path, default=None, help="Optional per-match CSV output path.")
    parser.add_argument("--diagnostic-dir", type=Path, default=None, help="Write FP/FN/GT CSVs and optional PNG diagnostics here.")
    parser.add_argument("--background", type=Path, default=None, help="Cutout FITS used as PNG background for diagnostics.")
    parser.add_argument("--background-hdu", default="IMAGE", help="Image HDU in --background, default IMAGE.")
    parser.add_argument("--pred-mask-dir", type=Path, default=None, help="Optional merge_regions directory for prediction parent masks.")
    parser.add_argument("--stamp-size", type=int, default=64, help="Stamp size in pixels for FP/FN diagnostic PNGs.")
    parser.add_argument("--max-diagnostic-stamps", type=int, default=80, help="Maximum FP and FN stamps to write.")
    parser.add_argument("--draw-ellipses", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ellipse-max-major", type=float, default=20.0, help="Do not draw ellipses with major axis above this many pixels.")
    parser.add_argument("--ellipse-max-axis-ratio", type=float, default=8.0, help="Do not draw extremely elongated ellipses above this axis ratio.")
    parser.add_argument("--ref-x", default=None, help=f"Reference x column, default auto preferring {DEFAULT_SDSS_X}.")
    parser.add_argument("--ref-y", default=None, help=f"Reference y column, default auto preferring {DEFAULT_SDSS_Y}.")
    parser.add_argument("--pred-x", default=None, help="Prediction x column; auto uses SDSS centroid then deblend peak center.")
    parser.add_argument("--pred-y", default=None, help="Prediction y column; auto uses SDSS centroid then deblend peak center.")
    parser.add_argument("--ref-hdu", type=int, default=1)
    parser.add_argument("--pred-hdu", type=int, default=1)
    parser.add_argument("--radius-arcsec", type=float, default=DEFAULT_RADIUS_ARCSEC)
    parser.add_argument("--pixel-scale", type=float, default=DEFAULT_PIXEL_SCALE)
    parser.add_argument("--ref-leaf-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pred-leaf-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-flagged-centroids", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--pred-require-science-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For deblendedFlux catalogs, drop rows with empty deblend_modelType.",
    )
    args = parser.parse_args()

    result = evaluate(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
