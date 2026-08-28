#!/usr/bin/env python3
"""Compare CELLECT detections between two raw HSC tile visits.

The script reads the 256x256 HSC tile pack, runs a checkpoint on paired visits,
matches detections by mutual nearest neighbors, and measures fixed-aperture SNR
for both common and visit-specific detections.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Ellipse  # noqa: E402
from scipy import ndimage  # noqa: E402

from eval.eval_utils import (  # noqa: E402
    detection_rows,
    infer_cellect,
    load_cellect_model,
    make_training_rgb,
    select_band_outputs,
    zscale_gray,
)
from eval.hsctiles.eval_hsctile_pack import (  # noqa: E402
    DEFAULT_ROOT,
    FrameRef,
    _frame_ref,
    _open_pack,
    _read_frame,
    _tile_by_id,
)
from eval.visualize_cellect_outputs import _score_map  # noqa: E402
from utils.aperture_snr import (  # noqa: E402
    ApertureNoiseFields,
    ApertureSnrResult as SnrResult,
    build_aperture_noise_fields,
    ellipse_union_mask,
)


DEFAULT_COMPARISONS = (
    ("4,5", "x000_y006", "HSC-I", 100752, 100762),
    ("4,5", "x000_y006", "HSC-Y", 102128, 102140),
    ("4,5", "x000_y006", "NB0816", 145600, 163710),
    ("6,1", "x000_y003", "HSC-G", 101424, 101434),
    ("6,1", "x000_y003", "HSC-I", 100752, 100758),
    ("6,1", "x000_y003", "HSC-Y", 102128, 102136),
    ("6,1", "x000_y008", "HSC-G", 101418, 101424),
    ("6,1", "x000_y008", "HSC-I", 100770, 103660),
)


@dataclass(frozen=True)
class VisitResult:
    label: str
    ref: FrameRef
    image: np.ndarray
    scaled: np.ndarray
    score_map: np.ndarray
    rows: list[dict[str, float]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("/data/czh23/ckpts/sam_log_lupton_0810/epoch_0020.pt"))
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out-dir", type=Path, default=Path("output/eval_0815"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", choices=("none", "bf16"), default="bf16")
    p.add_argument("--comparison", action="append", default=None, help="patch,tile,band,visit_a,visit_b")
    p.add_argument("--scaling-mode", default="log_lupton")
    p.add_argument("--clip-threshold", type=float, default=5.0)
    p.add_argument("--log-a", type=float, default=300.0)
    p.add_argument("--log-high-percentile", type=float, default=99.5)
    p.add_argument("--lupton-stretch", type=float, default=0.5)
    p.add_argument("--lupton-q", type=float, default=20.0)
    p.add_argument("--confidence-threshold", type=float, default=2.0)
    p.add_argument("--confidence-score", default="ordinal_expectation")
    p.add_argument("--nms-radius", type=int, default=3)
    p.add_argument("--center-refinement", choices=("integer", "weighted_centroid", "softargmax"), default="softargmax")
    p.add_argument("--center-refinement-radius", type=int, default=1)
    p.add_argument("--match-radius", type=float, default=3.0)
    p.add_argument("--aperture-radius", type=float, default=5.0)
    p.add_argument("--snr-clip-rounds", type=int, default=2)
    p.add_argument("--snr-clip-sigma", type=float, default=3.0)
    p.add_argument("--min-sky-apertures", type=int, default=16)
    p.add_argument("--source-mask-scale", type=float, default=1.0)
    p.add_argument("--source-only-scale", type=float, default=1.2)
    p.add_argument("--snr-background-box", type=int, default=65)
    p.add_argument("--snr-background-method", choices=("median", "poly"), default="median")
    p.add_argument("--snr-background-poly-degree", type=int, default=2)
    p.add_argument("--snr-background-poly-clip-rounds", type=int, default=2)
    p.add_argument("--snr-background-poly-clip-sigma", type=float, default=3.0)
    p.add_argument("--snr-high-threshold-sigma", type=float, default=3.0)
    p.add_argument("--snr-high-dilation-radius", type=int, default=5)
    p.add_argument("--annulus-inner-radius", type=float, default=10.0)
    p.add_argument("--annulus-outer-radius", type=float, default=15.0)
    p.add_argument("--min-annulus-pixels", type=int, default=100)
    p.add_argument("--local-background-radius", type=float, default=48.0)
    p.add_argument("--min-local-sky-apertures", type=int, default=8)
    p.add_argument("--invert-background", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--profile-crop", type=int, default=64)
    p.add_argument("--profiles-per-page", type=int, default=12)
    return p.parse_args()


def _parse_comparison(text: str) -> tuple[str, str, str, int, int]:
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) == 6:
        patch = f"{parts[0]},{parts[1]}"
        tile, band, visit_a, visit_b = parts[2:]
    elif len(parts) == 5:
        patch, tile, band, visit_a, visit_b = parts
    else:
        raise ValueError(f"bad --comparison {text!r}; expected patch,tile,band,visit_a,visit_b")
    return patch, tile, band, int(visit_a), int(visit_b)


def _comparisons(args: argparse.Namespace) -> list[tuple[str, str, str, int, int]]:
    if not args.comparison:
        return list(DEFAULT_COMPARISONS)
    return [_parse_comparison(value) for value in args.comparison]


def _slug(text: object) -> str:
    return str(text).replace(",", "_").replace("/", "_")


def _resolve_ref(root: Path, patch: str, tile_id: str, band: str, visit: int) -> FrameRef:
    by_id = _tile_by_id(root, band, patch)
    if tile_id not in by_id:
        raise KeyError(f"tile {tile_id!r} not found for {band} {patch}")
    return _frame_ref(root, band, patch, by_id[tile_id], frame_rank=0, visit=int(visit), strict_visit=True)


def _run_visit(
    *,
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    root: Path,
    patch: str,
    tile_id: str,
    band: str,
    visit: int,
) -> VisitResult:
    ref = _resolve_ref(root, patch, tile_id, band, visit)
    image = np.nan_to_num(_read_frame(root, ref), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    scaled = make_training_rgb(
        image,
        mode=str(args.scaling_mode),
        clip_threshold=float(args.clip_threshold),
        log_a=float(args.log_a),
        log_high_percentile=float(args.log_high_percentile),
        lupton_stretch=float(args.lupton_stretch),
        lupton_q=float(args.lupton_q),
    )
    tensor = torch.from_numpy(scaled[None, None].astype(np.float32, copy=False))
    outputs = infer_cellect(model=model, image_tensor=tensor, device=device, amp=str(args.amp))
    band_outputs = select_band_outputs(outputs, 0)
    rows = detection_rows(
        band_outputs,
        threshold=float(args.confidence_threshold),
        nms_radius=int(args.nms_radius),
        confidence_score=str(args.confidence_score),
        center_refinement=str(args.center_refinement),
        center_refinement_radius=int(args.center_refinement_radius),
        width=image.shape[1],
        height=image.shape[0],
    )
    for idx, row in enumerate(rows):
        row["id"] = float(idx + 1)
    score = _score_map(band_outputs)[: image.shape[0], : image.shape[1]].astype(np.float32, copy=False)
    return VisitResult(label=str(visit), ref=ref, image=image, scaled=scaled, score_map=score, rows=rows)


def _xy(rows: Sequence[dict[str, float]]) -> np.ndarray:
    if not rows:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray([[float(row["x"]), float(row["y"])] for row in rows], dtype=np.float32)


def _mutual_matches(rows_a: Sequence[dict[str, float]], rows_b: Sequence[dict[str, float]], radius: float) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    xy_a = _xy(rows_a)
    xy_b = _xy(rows_b)
    if len(xy_a) == 0 or len(xy_b) == 0:
        return [], list(range(len(rows_a))), list(range(len(rows_b)))
    dist = np.sqrt(((xy_a[:, None, :] - xy_b[None, :, :]) ** 2).sum(axis=2))
    nearest_b = np.argmin(dist, axis=1)
    nearest_a = np.argmin(dist, axis=0)
    pairs: list[tuple[int, int, float]] = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    for ia, ib in enumerate(nearest_b):
        if int(nearest_a[int(ib)]) == ia and float(dist[ia, ib]) <= float(radius):
            pairs.append((ia, int(ib), float(dist[ia, ib])))
            used_a.add(ia)
            used_b.add(int(ib))
    unmatched_a = [idx for idx in range(len(rows_a)) if idx not in used_a]
    unmatched_b = [idx for idx in range(len(rows_b)) if idx not in used_b]
    return pairs, unmatched_a, unmatched_b


def _ellipse_params(row: dict[str, float]) -> tuple[float, float, float, float, float]:
    x = float(row.get("x", 0.0))
    y = float(row.get("y", 0.0))
    major = max(abs(float(row.get("major", 1.0))), 1.0)
    minor = max(abs(float(row.get("minor", 1.0))), 1.0)
    theta = float(row.get("theta", 0.0))
    if abs(theta) > 2.0 * math.pi:
        theta = math.radians(theta)
    return x, y, major, minor, theta


def _ellipse_contains(xx: np.ndarray, yy: np.ndarray, row: dict[str, float], *, scale: float = 1.0) -> np.ndarray:
    x, y, major, minor, theta = _ellipse_params(row)
    major = max(major * float(scale), 1.0)
    minor = max(minor * float(scale), 1.0)
    ct, st = math.cos(theta), math.sin(theta)
    dx = xx - x
    dy = yy - y
    xp = dx * ct + dy * st
    yp = -dx * st + dy * ct
    return (xp / major) ** 2 + (yp / minor) ** 2 <= 1.0


def _internal(row: dict[str, float], shape: tuple[int, int], margin: float = 0.0) -> bool:
    h, w = shape
    x, y, major, minor, _theta = _ellipse_params(row)
    radius = max(major, minor, 15.0) + float(margin)
    return x - radius >= 0 and y - radius >= 0 and x + radius < w and y + radius < h


def _add_ellipse(ax: plt.Axes, row: dict[str, float], *, color: str, lw: float = 1.1, alpha: float = 0.95, scale: float = 1.0) -> None:
    x, y, major, minor, theta = _ellipse_params(row)
    patch = Ellipse(
        (x, y),
        width=2.0 * major * float(scale),
        height=2.0 * minor * float(scale),
        angle=math.degrees(theta),
        fill=False,
        edgecolor=color,
        linewidth=lw,
        alpha=alpha,
    )
    ax.add_patch(patch)


def _display_gray(image: np.ndarray, *, invert_background: bool = False) -> np.ndarray:
    gray = zscale_gray(image)
    if bool(invert_background):
        gray = 1.0 - gray
    return gray


def _setup_image_axis(ax: plt.Axes, image: np.ndarray, title: str, *, invert_background: bool = False) -> None:
    ax.imshow(
        _display_gray(image, invert_background=bool(invert_background)),
        origin="lower",
        cmap="gray",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, image.shape[1] - 1)
    ax.set_ylim(0, image.shape[0] - 1)
    ax.set_aspect("equal")


def plot_all_detections(out_path: Path, a: VisitResult, b: VisitResult, *, invert_background: bool = False) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=160, constrained_layout=True)
    for ax, result in zip(axes, (a, b), strict=True):
        _setup_image_axis(
            ax,
            result.image,
            f"visit {result.label}: all detections n={len(result.rows)}",
            invert_background=bool(invert_background),
        )
        for row in result.rows:
            _add_ellipse(ax, row, color="cyan", lw=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_all_snr(
    out_path: Path,
    a: VisitResult,
    b: VisitResult,
    snr_by_label: dict[str, list[SnrResult]],
    *,
    invert_background: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), dpi=180, constrained_layout=True)
    for ax, result in zip(axes, (a, b), strict=True):
        _setup_image_axis(
            ax,
            result.image,
            f"visit {result.label}: all-source aperture SNR",
            invert_background=bool(invert_background),
        )
        snrs = snr_by_label[result.label]
        for row, entry in zip(result.rows, snrs, strict=True):
            x, y = float(row["x"]), float(row["y"])
            if not entry.trusted or not np.isfinite(entry.snr):
                color = "white"
                label = "bad"
                lw = 1.5
            elif entry.snr < 5.0:
                color = "red"
                label = f"{entry.snr:.1f}"
                lw = 1.7
            else:
                color = "lime"
                label = f"{entry.snr:.1f}"
                lw = 1.25
            ax.add_patch(Circle((x, y), radius=5.0, fill=False, edgecolor=color, linewidth=lw))
            ax.text(
                x + 5.8,
                y + 4.8,
                label,
                color=color,
                fontsize=8.5,
                fontweight="bold",
                path_effects=[],
            )
        handles = [
            Circle((0, 0), radius=1, fill=False, edgecolor="lime", linewidth=1.4, label="SNR >= 5"),
            Circle((0, 0), radius=1, fill=False, edgecolor="red", linewidth=1.7, label="SNR < 5"),
            Circle((0, 0), radius=1, fill=False, edgecolor="white", linewidth=1.5, label="untrusted"),
        ]
        ax.legend(handles=handles, fontsize=7, loc="upper right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_difference_snr(
    out_path: Path,
    a: VisitResult,
    b: VisitResult,
    unmatched_a: Sequence[int],
    unmatched_b: Sequence[int],
    snr_a_at_diff: dict[str, list[SnrResult]],
    snr_b_at_diff: dict[str, list[SnrResult]],
    *,
    invert_background: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5), dpi=180, constrained_layout=True)
    diff_rows = {
        "A_only": [a.rows[idx] for idx in unmatched_a],
        "B_only": [b.rows[idx] for idx in unmatched_b],
    }
    palette = {"A_only": "magenta", "B_only": "yellow"}
    labels = {"A_only": f"only {a.label}", "B_only": f"only {b.label}"}
    for ax, result, snr_lookup in (
        (axes[0], a, snr_a_at_diff),
        (axes[1], b, snr_b_at_diff),
    ):
        _setup_image_axis(
            ax,
            result.image,
            f"visit {result.label}: difference-source SNR",
            invert_background=bool(invert_background),
        )
        for key, rows in diff_rows.items():
            for idx, row in enumerate(rows):
                x, y = float(row["x"]), float(row["y"])
                snr = snr_lookup[key][idx]
                ax.add_patch(Circle((x, y), radius=5.0, fill=False, edgecolor=palette[key], linewidth=1.8))
                text = "bad" if not snr.trusted or not np.isfinite(snr.snr) else f"{snr.snr:.1f}"
                ax.text(x + 6.2, y + 5.5, text, color=palette[key], fontsize=11, fontweight="bold")
        handles = [
            Circle((0, 0), radius=1, fill=False, edgecolor=palette[key], linewidth=1.2, label=labels[key])
            for key in ("A_only", "B_only")
        ]
        ax.legend(handles=handles, fontsize=7, loc="upper right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_confidence_detections(
    out_path: Path,
    a: VisitResult,
    b: VisitResult,
    unmatched_a: Sequence[int],
    unmatched_b: Sequence[int],
    *,
    invert_background: bool = False,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 11.2), dpi=220, constrained_layout=True)
    confidence_vmin = 0.0
    confidence_vmax = 2.5
    diff_sets = {
        a.label: {"A_only": [a.rows[idx] for idx in unmatched_a], "B_only": [b.rows[idx] for idx in unmatched_b]},
        b.label: {"A_only": [a.rows[idx] for idx in unmatched_a], "B_only": [b.rows[idx] for idx in unmatched_b]},
    }
    conf_images = []
    for col, result in enumerate((a, b)):
        finite = np.isfinite(result.score_map)
        vmax = float(np.nanpercentile(result.score_map[finite], 99.7)) if bool(finite.any()) else 1.0
        _unused_data_vmax = max(vmax, 1.0)
        conf = np.asarray(result.score_map, dtype=np.float32)
        positive = np.where(conf > 0.0, conf, np.nan)
        for row_idx, (ax, plane, title, cmap, vmin, vmax_use) in enumerate(
            (
                (axes[0, col], positive, f"visit {result.label}: confidence score > 0", "magma", confidence_vmin, confidence_vmax),
                (
                    axes[1, col],
                    _display_gray(result.image, invert_background=bool(invert_background)),
                    f"visit {result.label}: raw tile",
                    "gray",
                    0.0,
                    1.0,
                ),
            )
        ):
            im = ax.imshow(plane, origin="lower", cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax_use)
            if row_idx == 0:
                conf_images.append(im)
            ax.set_title(title, fontsize=10)
            ax.set_xlim(0, result.score_map.shape[1] - 1)
            ax.set_ylim(0, result.score_map.shape[0] - 1)
            ax.set_aspect("equal")
            for row in result.rows:
                if row not in diff_sets[result.label]["A_only"] and row not in diff_sets[result.label]["B_only"]:
                    ax.add_patch(Circle((float(row["x"]), float(row["y"])), radius=4.0, fill=False, edgecolor="cyan", linewidth=0.9, alpha=0.85))
            for row in diff_sets[result.label]["A_only"]:
                ax.add_patch(Circle((float(row["x"]), float(row["y"])), radius=8.0, fill=False, edgecolor="magenta", linewidth=2.2))
            for row in diff_sets[result.label]["B_only"]:
                ax.add_patch(Circle((float(row["x"]), float(row["y"])), radius=8.0, fill=False, edgecolor="yellow", linewidth=2.2))
            if row_idx == 0:
                handles = [
                    Circle((0, 0), radius=1, fill=False, edgecolor="cyan", linewidth=1.0, label="all detections"),
                    Circle((0, 0), radius=1, fill=False, edgecolor="magenta", linewidth=2.0, label=f"only {a.label}"),
                    Circle((0, 0), radius=1, fill=False, edgecolor="yellow", linewidth=2.0, label=f"only {b.label}"),
                ]
                ax.legend(handles=handles, fontsize=7, loc="upper right")
    for ax, im in zip(axes[0, :], conf_images, strict=True):
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.015)
        cbar.set_label("confidence score", fontsize=7)
        cbar.ax.tick_params(labelsize=6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _write_detection_csv(path: Path, result: VisitResult, snrs: Sequence[SnrResult]) -> None:
    fields = (
        "visit",
        "row_id",
        "x",
        "y",
        "score",
        "major",
        "minor",
        "theta",
        "snr",
        "snr_trusted",
        "snr_internal",
        "flux",
        "aperture_background",
        "background_method",
        "aperture_sigma",
        "aperture_pixels",
        "sky_aperture_count",
        "local_sky_aperture_count",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(result.rows):
            snr = snrs[idx]
            writer.writerow(
                {
                    "visit": result.label,
                    "row_id": idx + 1,
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "score": float(row.get("score", float("nan"))),
                    "major": float(row.get("major", float("nan"))),
                    "minor": float(row.get("minor", float("nan"))),
                    "theta": float(row.get("theta", float("nan"))),
                    "snr": snr.snr,
                    "snr_trusted": int(snr.trusted),
                    "snr_internal": int(snr.internal),
                    "flux": snr.flux,
                    "aperture_background": snr.background,
                    "background_method": snr.background_method,
                    "aperture_sigma": snr.sigma,
                    "aperture_pixels": snr.aperture_pixels,
                    "sky_aperture_count": snr.sky_aperture_count,
                    "local_sky_aperture_count": snr.local_sky_aperture_count,
                }
            )


def _write_match_csv(
    path: Path,
    a: VisitResult,
    b: VisitResult,
    pairs: Sequence[tuple[int, int, float]],
    unmatched_a: Sequence[int],
    unmatched_b: Sequence[int],
    snr_a_at_diff: dict[str, list[SnrResult]],
    snr_b_at_diff: dict[str, list[SnrResult]],
) -> None:
    fields = (
        "kind",
        "visit_a_row",
        "visit_b_row",
        "distance",
        "x",
        "y",
        "snr_visit_a",
        "snr_visit_a_trusted",
        "snr_visit_b",
        "snr_visit_b_trusted",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for ia, ib, dist in pairs:
            row = a.rows[ia]
            writer.writerow(
                {
                    "kind": "matched",
                    "visit_a_row": ia + 1,
                    "visit_b_row": ib + 1,
                    "distance": dist,
                    "x": row["x"],
                    "y": row["y"],
                    "snr_visit_a": "",
                    "snr_visit_a_trusted": "",
                    "snr_visit_b": "",
                    "snr_visit_b_trusted": "",
                }
            )
        for local_idx, ia in enumerate(unmatched_a):
            row = a.rows[ia]
            sa = snr_a_at_diff["A_only"][local_idx]
            sb = snr_b_at_diff["A_only"][local_idx]
            writer.writerow(
                {
                    "kind": "A_only",
                    "visit_a_row": ia + 1,
                    "visit_b_row": "",
                    "distance": "",
                    "x": row["x"],
                    "y": row["y"],
                    "snr_visit_a": sa.snr,
                    "snr_visit_a_trusted": int(sa.trusted),
                    "snr_visit_b": sb.snr,
                    "snr_visit_b_trusted": int(sb.trusted),
                }
            )
        for local_idx, ib in enumerate(unmatched_b):
            row = b.rows[ib]
            sa = snr_a_at_diff["B_only"][local_idx]
            sb = snr_b_at_diff["B_only"][local_idx]
            writer.writerow(
                {
                    "kind": "B_only",
                    "visit_a_row": "",
                    "visit_b_row": ib + 1,
                    "distance": "",
                    "x": row["x"],
                    "y": row["y"],
                    "snr_visit_a": sa.snr,
                    "snr_visit_a_trusted": int(sa.trusted),
                    "snr_visit_b": sb.snr,
                    "snr_visit_b_trusted": int(sb.trusted),
                }
            )


def _robust_sigma(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    med = float(np.median(arr))
    return float(1.4826 * np.median(np.abs(arr - med)))


def _disk_structure(radius: int) -> np.ndarray:
    r = int(max(radius, 0))
    if r <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return (xx * xx + yy * yy) <= r * r


def _source_mask(rows: Sequence[dict[str, float]], shape: tuple[int, int], *, scale: float) -> np.ndarray:
    mask_rows = []
    for row in rows:
        if "x" not in row or "y" not in row:
            continue
        mask_rows.append(
            {
                "x": float(row["x"]),
                "y": float(row["y"]),
                "major": max(float(row.get("major", 5.0)), 1.0),
                "minor": max(float(row.get("minor", 5.0)), 1.0),
                "theta": float(row.get("theta", 0.0)),
            }
        )
    return ellipse_union_mask(shape, mask_rows, scale=float(scale))


def _poly_design(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    terms = []
    deg = int(max(degree, 0))
    for total in range(deg + 1):
        for x_power in range(total + 1):
            y_power = total - x_power
            terms.append((x ** x_power) * (y ** y_power))
    return np.stack(terms, axis=1)


def _fit_polynomial_background(
    args: argparse.Namespace,
    image: np.ndarray,
    *,
    exclude_mask: np.ndarray,
) -> np.ndarray | None:
    arr = np.asarray(image, dtype=np.float32)
    height, width = arr.shape
    yy, xx = np.mgrid[:height, :width]
    x_norm = (xx.astype(np.float64) / max(width - 1, 1)) * 2.0 - 1.0
    y_norm = (yy.astype(np.float64) / max(height - 1, 1)) * 2.0 - 1.0
    finite = np.isfinite(arr)
    keep = finite & (~np.asarray(exclude_mask, dtype=bool))
    min_points = max(30, (int(args.snr_background_poly_degree) + 1) * (int(args.snr_background_poly_degree) + 2))
    if int(np.sum(keep)) < min_points:
        return None

    z = arr.astype(np.float64, copy=False)
    for _round in range(max(int(args.snr_background_poly_clip_rounds), 0) + 1):
        design = _poly_design(x_norm[keep], y_norm[keep], int(args.snr_background_poly_degree))
        values = z[keep]
        if values.size < min_points:
            return None
        try:
            coeff, *_unused = np.linalg.lstsq(design, values, rcond=None)
        except np.linalg.LinAlgError:
            return None
        if _round >= int(args.snr_background_poly_clip_rounds):
            break
        predicted = design @ coeff
        resid = values - predicted
        scale = _robust_sigma(resid)
        if not np.isfinite(scale) or scale <= 0.0:
            break
        center = float(np.median(resid))
        clipped_keep_values = np.abs(resid - center) <= float(args.snr_background_poly_clip_sigma) * scale
        old_indices = np.flatnonzero(keep)
        new_keep = np.zeros_like(keep)
        new_keep.ravel()[old_indices[clipped_keep_values]] = True
        if int(np.sum(new_keep)) == int(np.sum(keep)):
            keep = new_keep
            break
        keep = new_keep

    full_design = _poly_design(x_norm.ravel(), y_norm.ravel(), int(args.snr_background_poly_degree))
    background = (full_design @ coeff).reshape(arr.shape)
    return background.astype(np.float32, copy=False)


def _background_residual(
    args: argparse.Namespace,
    image: np.ndarray,
    source_rows: Sequence[dict[str, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    box = int(max(args.snr_background_box, 3))
    if box % 2 == 0:
        box += 1
    arr = np.asarray(image, dtype=np.float32)
    median_background = ndimage.median_filter(arr, size=box, mode="reflect")
    background = median_background
    if str(args.snr_background_method) == "poly":
        initial_residual = arr - median_background.astype(np.float32, copy=False)
        finite = initial_residual[np.isfinite(initial_residual)]
        center = float(np.median(finite)) if finite.size else 0.0
        scale = _robust_sigma(finite)
        if np.isfinite(scale) and scale > 0.0:
            high_mask = initial_residual > center + float(args.snr_high_threshold_sigma) * scale
            high_mask = ndimage.binary_dilation(high_mask, structure=_disk_structure(int(args.snr_high_dilation_radius)))
        else:
            high_mask = np.zeros_like(initial_residual, dtype=bool)
        if source_rows is None:
            source_mask = np.zeros_like(initial_residual, dtype=bool)
        else:
            source_mask = _source_mask(source_rows, initial_residual.shape, scale=float(args.source_only_scale))
        poly_background = _fit_polynomial_background(args, arr, exclude_mask=high_mask | source_mask)
        if poly_background is not None:
            background = poly_background
    residual = arr - background.astype(np.float32, copy=False)
    return residual.astype(np.float32, copy=False), background.astype(np.float32, copy=False)


def _snr_noise_fields(args: argparse.Namespace, result: VisitResult) -> ApertureNoiseFields:
    residual, _background = _background_residual(args, result.image, result.rows)
    finite = residual[np.isfinite(residual)]
    center = float(np.median(finite)) if finite.size else 0.0
    sigma = _robust_sigma(finite)
    if np.isfinite(sigma) and sigma > 0.0:
        high_mask = residual > center + float(args.snr_high_threshold_sigma) * sigma
    else:
        high_mask = np.zeros_like(residual, dtype=bool)
    high_mask = ndimage.binary_dilation(high_mask, structure=_disk_structure(int(args.snr_high_dilation_radius)))
    source_mask = _source_mask(result.rows, result.image.shape, scale=float(args.source_only_scale))
    total_mask = high_mask | source_mask
    return build_aperture_noise_fields(
        residual,
        source_mask=total_mask,
        aperture_radius=float(args.aperture_radius),
        clip_rounds=int(args.snr_clip_rounds),
        clip_sigma=float(args.snr_clip_sigma),
        min_sky_apertures=int(args.min_sky_apertures),
    )


def _annulus_flux(
    args: argparse.Namespace,
    image: np.ndarray,
    row: dict[str, float],
    source_rows: Sequence[dict[str, float]],
) -> tuple[float, float, int, int]:
    height, width = image.shape
    x = float(row["x"])
    y = float(row["y"])
    r_ap = float(args.aperture_radius)
    r_in = float(args.annulus_inner_radius)
    r_out = float(args.annulus_outer_radius)
    pad = int(math.ceil(r_out)) + 2
    x0 = max(0, int(math.floor(x)) - pad)
    x1 = min(width, int(math.floor(x)) + pad + 1)
    y0 = max(0, int(math.floor(y)) - pad)
    y1 = min(height, int(math.floor(y)) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return float("nan"), float("nan"), 0, 0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr2 = (xx - x) ** 2 + (yy - y) ** 2
    aperture = rr2 <= r_ap * r_ap
    annulus = (rr2 >= r_in * r_in) & (rr2 <= r_out * r_out)
    other_rows = [other for other in source_rows if other is not row]
    other_mask = _source_mask(other_rows, (height, width), scale=1.0)[y0:y1, x0:x1]
    local = np.asarray(image[y0:y1, x0:x1], dtype=np.float32)
    finite_ap = aperture & np.isfinite(local)
    finite_ann = annulus & np.isfinite(local) & (~other_mask)
    aperture_pixels = int(np.sum(finite_ap))
    annulus_pixels = int(np.sum(finite_ann))
    if aperture_pixels <= 0:
        return float("nan"), float("nan"), annulus_pixels, aperture_pixels
    background_per_pixel = (
        float(np.median(local[finite_ann])) if annulus_pixels >= int(args.min_annulus_pixels) else float("nan")
    )
    aperture_sum = float(np.sum(local[finite_ap], dtype=np.float64))
    flux = aperture_sum - background_per_pixel * aperture_pixels if np.isfinite(background_per_pixel) else float("nan")
    return flux, background_per_pixel * aperture_pixels if np.isfinite(background_per_pixel) else float("nan"), annulus_pixels, aperture_pixels


def _measure_snr(args: argparse.Namespace, result: VisitResult, fields: ApertureNoiseFields, row: dict[str, float]) -> SnrResult:
    flux, aperture_background, annulus_pixels, aperture_pixels = _annulus_flux(args, result.image, row, result.rows)
    area_fraction = aperture_pixels / max(float(fields.model.aperture_area_pixels), 1.0)
    sigma = float(fields.model.sigma) * math.sqrt(max(area_fraction, 0.0))
    snr = flux / sigma if np.isfinite(flux) and np.isfinite(sigma) and sigma > 0.0 else float("nan")
    x = float(row.get("x", row.get("X_IMAGE", 0.0)))
    y = float(row.get("y", row.get("Y_IMAGE", 0.0)))
    major = max(abs(float(row.get("major", row.get("a", 1.0)))), 1.0)
    minor = max(abs(float(row.get("minor", row.get("b", 1.0)))), 1.0)
    edge_radius = max(major, minor, float(args.aperture_radius))
    height, width = result.image.shape
    internal = bool(
        x - edge_radius >= 0.0
        and y - edge_radius >= 0.0
        and x + edge_radius < width
        and y + edge_radius < height
    )
    trusted = bool(
        fields.model.trusted
        and aperture_pixels > 0
        and annulus_pixels >= int(args.min_annulus_pixels)
        and internal
        and np.isfinite(snr)
    )
    return SnrResult(
        float(snr),
        float(flux),
        float(aperture_background),
        float(sigma),
        int(aperture_pixels),
        int(fields.model.sky_aperture_count),
        trusted,
        internal,
        "annulus_flux_masked_aperture_rms",
        int(annulus_pixels),
    )


def analyze_one(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    root: Path,
    device: torch.device,
    patch: str,
    tile_id: str,
    band: str,
    visit_a: int,
    visit_b: int,
) -> dict[str, object]:
    out_dir = args.out_dir / f"{_slug(patch)}_{tile_id}_{band}_{visit_a}_vs_{visit_b}"
    out_dir.mkdir(parents=True, exist_ok=True)
    a = _run_visit(model=model, device=device, args=args, root=root, patch=patch, tile_id=tile_id, band=band, visit=visit_a)
    b = _run_visit(model=model, device=device, args=args, root=root, patch=patch, tile_id=tile_id, band=band, visit=visit_b)
    pairs, unmatched_a, unmatched_b = _mutual_matches(a.rows, b.rows, float(args.match_radius))

    noise_a = _snr_noise_fields(args, a)
    noise_b = _snr_noise_fields(args, b)

    snr_a = [_measure_snr(args, a, noise_a, row) for row in a.rows]
    snr_b = [_measure_snr(args, b, noise_b, row) for row in b.rows]

    a_only_rows = [a.rows[idx] for idx in unmatched_a]
    b_only_rows = [b.rows[idx] for idx in unmatched_b]
    snr_a_at_diff = {
        "A_only": [_measure_snr(args, a, noise_a, row) for row in a_only_rows],
        "B_only": [_measure_snr(args, a, noise_a, row) for row in b_only_rows],
    }
    snr_b_at_diff = {
        "A_only": [_measure_snr(args, b, noise_b, row) for row in a_only_rows],
        "B_only": [_measure_snr(args, b, noise_b, row) for row in b_only_rows],
    }

    plot_all_detections(out_dir / "all_detections.png", a, b, invert_background=bool(args.invert_background))
    plot_difference_snr(
        out_dir / "difference_source_snr.png",
        a,
        b,
        unmatched_a,
        unmatched_b,
        snr_a_at_diff,
        snr_b_at_diff,
        invert_background=bool(args.invert_background),
    )
    plot_all_snr(
        out_dir / "all_detection_snr.png",
        a,
        b,
        {a.label: snr_a, b.label: snr_b},
        invert_background=bool(args.invert_background),
    )
    old_profile_dir = out_dir / "confidence_profiles"
    if old_profile_dir.exists():
        shutil.rmtree(old_profile_dir)
    plot_confidence_detections(
        out_dir / "confidence_detections.png",
        a,
        b,
        unmatched_a,
        unmatched_b,
        invert_background=bool(args.invert_background),
    )
    _write_detection_csv(out_dir / f"detections_visit_{visit_a}.csv", a, snr_a)
    _write_detection_csv(out_dir / f"detections_visit_{visit_b}.csv", b, snr_b)
    _write_match_csv(out_dir / "match_and_difference_snr.csv", a, b, pairs, unmatched_a, unmatched_b, snr_a_at_diff, snr_b_at_diff)
    manifest = {
        "patch": patch,
        "tile_id": tile_id,
        "band": band,
        "visit_a": visit_a,
        "visit_b": visit_b,
        "frame_a": a.ref.__dict__,
        "frame_b": b.ref.__dict__,
        "n_a": len(a.rows),
        "n_b": len(b.rows),
        "matched": len(pairs),
        "a_only": len(unmatched_a),
        "b_only": len(unmatched_b),
        "match_radius": float(args.match_radius),
        "aperture_radius": float(args.aperture_radius),
        "snr_method": "annulus_flux_high_dilated_source12_aperture_rms",
        "snr_clip_rounds": int(args.snr_clip_rounds),
        "snr_clip_sigma": float(args.snr_clip_sigma),
        "source_only_scale": float(args.source_only_scale),
        "snr_background_method": str(args.snr_background_method),
        "snr_background_box": int(args.snr_background_box),
        "snr_background_poly_degree": int(args.snr_background_poly_degree),
        "snr_background_poly_clip_rounds": int(args.snr_background_poly_clip_rounds),
        "snr_background_poly_clip_sigma": float(args.snr_background_poly_clip_sigma),
        "snr_high_threshold_sigma": float(args.snr_high_threshold_sigma),
        "snr_high_dilation_radius": int(args.snr_high_dilation_radius),
        "annulus_inner_radius": float(args.annulus_inner_radius),
        "annulus_outer_radius": float(args.annulus_outer_radius),
        "min_annulus_pixels": int(args.min_annulus_pixels),
        "source_mask_scale_legacy_unused": float(args.source_mask_scale),
        "local_background_radius_legacy_unused": float(args.local_background_radius),
        "min_local_sky_apertures": int(args.min_local_sky_apertures),
        "invert_background": bool(args.invert_background),
        "noise_a": noise_a.model.__dict__,
        "noise_b": noise_b.model.__dict__,
    }
    (out_dir / "summary.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(
        f"[done] {patch} {tile_id} {band} {visit_a} vs {visit_b}: "
        f"n=({len(a.rows)},{len(b.rows)}) matched={len(pairs)} only=({len(unmatched_a)},{len(unmatched_b)}) -> {out_dir}",
        flush=True,
    )
    return manifest


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    out_root = args.out_dir.expanduser().resolve()
    args.out_dir = out_root
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    model_cache: dict[str, torch.nn.Module] = {}
    summaries = []
    for patch, tile_id, band, visit_a, visit_b in _comparisons(args):
        try:
            if band not in model_cache:
                model_cache[band], _cfg = load_cellect_model(
                    args.checkpoint.expanduser().resolve(),
                    args.config.expanduser().resolve() if args.config else None,
                    device,
                    [band],
                    dynamic_image_size=True,
                )
            summaries.append(
                analyze_one(
                    args=args,
                    model=model_cache[band],
                    root=root,
                    device=device,
                    patch=patch,
                    tile_id=tile_id,
                    band=band,
                    visit_a=visit_a,
                    visit_b=visit_b,
                )
            )
        except Exception as exc:
            failure = {
                "patch": patch,
                "tile_id": tile_id,
                "band": band,
                "visit_a": int(visit_a),
                "visit_b": int(visit_b),
                "failed": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
            summaries.append(failure)
            print(
                f"[failed] {patch} {tile_id} {band} {visit_a} vs {visit_b}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
    (out_root / "visit_difference_summary.json").write_text(json.dumps(summaries, indent=2, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
