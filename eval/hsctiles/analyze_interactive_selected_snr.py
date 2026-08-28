#!/usr/bin/env python3
"""Build confidence/SNR panels for exported interactive selected cutouts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Ellipse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_utils import (  # noqa: E402
    detection_rows,
    infer_cellect,
    load_cellect_model,
    make_training_rgb,
    select_band_outputs,
    zscale_gray,
)
from eval.hsctiles.analyze_visit_differences import (  # noqa: E402
    VisitResult,
    _measure_snr,
    _score_map,
    _snr_noise_fields,
)


DEFAULT_SELECTED_ROOT = Path("eval/hsctiles/interactive_selected/20260818_215030/468")
DEFAULT_CHECKPOINT = Path("/data/czh23/ckpts/sam_anscombe_0803/epoch_0030.pt")
DEFAULT_OUT_DIR = Path("/home/czh23/analysis/2026-08/2026-08-19/interactive_selected_468_confidence_snr")
DEFAULT_BAND_ORDER = ("zg", "zr", "zi")


@dataclass(frozen=True)
class SelectedCutout:
    meta: dict[str, Any]
    leaf_dir: Path
    json_path: Path
    npz_path: Path

    @property
    def patch(self) -> str:
        return str(self.meta.get("patch", ""))

    @property
    def tile_id(self) -> str:
        return str(self.meta.get("tile_id", ""))

    @property
    def band(self) -> str:
        return str(self.meta.get("band", ""))

    @property
    def visit(self) -> str:
        return str(self.meta.get("visit", ""))

    @property
    def candidate_id(self) -> str:
        return str(self.meta.get("candidate_id") or self.json_path.stem)

    @property
    def group_key(self) -> tuple[str, str, int]:
        return (self.patch, self.tile_id, int(self.meta.get("frame_slot", 0) or 0))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selected-root", type=Path, default=DEFAULT_SELECTED_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", choices=("none", "bf16"), default="bf16")
    p.add_argument("--band-order", nargs="+", default=list(DEFAULT_BAND_ORDER))
    p.add_argument("--scaling-mode", default="anscombe")
    p.add_argument("--clip-threshold", type=float, default=3.0)
    p.add_argument("--log-a", type=float, default=300.0)
    p.add_argument("--log-high-percentile", type=float, default=99.5)
    p.add_argument("--lupton-stretch", type=float, default=0.5)
    p.add_argument("--lupton-q", type=float, default=20.0)
    p.add_argument("--anscombe-clip", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--anscombe-scale", type=float, default=1000.0)
    p.add_argument("--confidence-threshold", type=float, default=2.0)
    p.add_argument("--confidence-score", default="ordinal_expectation")
    p.add_argument("--nms-radius", type=int, default=2)
    p.add_argument("--center-refinement", choices=("integer", "weighted_centroid", "softargmax"), default="softargmax")
    p.add_argument("--center-refinement-radius", type=int, default=1)
    p.add_argument("--confidence-vmax", type=float, default=2.5)
    p.add_argument("--confidence-marker-radius", type=float, default=8.0)
    p.add_argument("--panel-size", type=int, default=1024)

    p.add_argument("--aperture-radius", type=float, default=5.0)
    p.add_argument("--snr-clip-rounds", type=int, default=2)
    p.add_argument("--snr-clip-sigma", type=float, default=3.0)
    p.add_argument("--min-sky-apertures", type=int, default=16)
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
    p.add_argument("--min-local-sky-apertures", type=int, default=8)
    return p.parse_args()


def _selected_cutouts(root: Path) -> list[SelectedCutout]:
    items: list[SelectedCutout] = []
    for path in sorted(root.rglob("*.json")):
        if path.name == "browser_manifest.json":
            continue
        with path.open(encoding="utf-8") as handle:
            meta = json.load(handle)
        npz_path = Path(meta.get("npz_path") or path.with_suffix(".npz"))
        if not npz_path.is_absolute():
            npz_path = path.parent / npz_path
        if not npz_path.is_file():
            raise FileNotFoundError(f"NPZ not found for {path}: {npz_path}")
        items.append(SelectedCutout(meta=meta, leaf_dir=path.parent, json_path=path, npz_path=npz_path))
    if not items:
        raise RuntimeError(f"no selected JSON cutouts found under {root}")
    return items


def _load_image(item: SelectedCutout) -> np.ndarray:
    with np.load(item.npz_path) as data:
        if "image" not in data:
            raise KeyError(f"{item.npz_path} does not contain 'image'")
        image = np.asarray(data["image"], dtype=np.float32)
    return np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)


def _upscale_nearest(image: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(image)
    if arr.shape[0] == int(size) and arr.shape[1] == int(size):
        return arr
    fy = max(1, int(round(float(size) / max(float(arr.shape[0]), 1.0))))
    fx = max(1, int(round(float(size) / max(float(arr.shape[1]), 1.0))))
    return np.repeat(np.repeat(arr, fy, axis=0), fx, axis=1)[:size, :size]


def _display(image: np.ndarray, *, size: int, invert: bool) -> np.ndarray:
    gray = zscale_gray(image)
    if invert:
        gray = 1.0 - gray
    return _upscale_nearest(gray, size)


def _ellipse_params(row: dict[str, float], scale: float) -> tuple[float, float, float, float, float]:
    x = float(row.get("x", 0.0)) * scale
    y = float(row.get("y", 0.0)) * scale
    major = max(abs(float(row.get("major", 1.0))), 1.0) * scale
    minor = max(abs(float(row.get("minor", 1.0))), 1.0) * scale
    theta = float(row.get("theta", 0.0))
    if abs(theta) > 2.0 * math.pi:
        theta = math.radians(theta)
    return x, y, major, minor, theta


def _draw_rows(ax: plt.Axes, rows: Sequence[dict[str, float]], *, scale: float, color: str = "cyan") -> None:
    for row in sorted(rows, key=lambda r: abs(float(r.get("major", 1.0)) * float(r.get("minor", 1.0))), reverse=True):
        x, y, major, minor, theta = _ellipse_params(row, scale)
        ax.add_patch(
            Ellipse(
                (x, y),
                width=2.0 * major,
                height=2.0 * minor,
                angle=math.degrees(theta),
                fill=False,
                edgecolor=color,
                linewidth=1.2,
                alpha=0.9,
            )
        )


def _draw_snr(ax: plt.Axes, rows: Sequence[dict[str, float]], snrs: Sequence[Any], *, scale: float) -> None:
    for row, entry in zip(rows, snrs, strict=True):
        x = float(row["x"]) * scale
        y = float(row["y"]) * scale
        if not entry.trusted or not np.isfinite(entry.snr):
            color = "white"
            label = "bad"
            lw = 2.4
        elif entry.snr < 5.0:
            color = "red"
            label = f"{entry.snr:.1f}"
            lw = 2.6
        else:
            color = "lime"
            label = f"{entry.snr:.1f}"
            lw = 1.9
        ax.add_patch(Circle((x, y), radius=5.0 * scale, fill=False, edgecolor=color, linewidth=lw))
        ax.text(
            x + 6.2 * scale,
            y + 4.8 * scale,
            label,
            color=color,
            fontsize=8.5,
            fontweight="bold",
        )


def _plot_panel(out_path: Path, result: VisitResult, snrs: Sequence[Any], meta: dict[str, Any], args: argparse.Namespace) -> None:
    size = int(args.panel_size)
    scale = size / float(result.image.shape[1])
    conf = _upscale_nearest(np.where(result.score_map > 0.0, result.score_map, np.nan), size)
    inv = _display(result.image, size=size, invert=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), dpi=170, constrained_layout=True)
    im = axes[0].imshow(conf, origin="lower", cmap="magma", vmin=0.0, vmax=float(args.confidence_vmax), interpolation="nearest")
    axes[0].set_title(f"confidence score > 0 ({size}x{size})", fontsize=11)
    for row in result.rows:
        axes[0].add_patch(
            Circle(
                (float(row["x"]) * scale, float(row["y"]) * scale),
                radius=float(args.confidence_marker_radius) * scale,
                fill=False,
                edgecolor="cyan",
                linewidth=1.4,
            )
        )
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02)

    axes[1].imshow(inv, origin="lower", cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    axes[1].set_title(f"all detections n={len(result.rows)}", fontsize=11)
    _draw_rows(axes[1], result.rows, scale=scale, color="cyan")

    axes[2].imshow(inv, origin="lower", cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    axes[2].set_title("aperture SNR overlay", fontsize=11)
    _draw_snr(axes[2], result.rows, snrs, scale=scale)

    title = (
        f"{meta.get('candidate_id', result.label)} | "
        f"{meta.get('dataset', '')} {meta.get('tract', '')}/{meta.get('patch', '')} {meta.get('band', '')} "
        f"{meta.get('tile_id', '')} visit={meta.get('visit', '')}"
    )
    fig.suptitle(title, fontsize=12)
    for ax in axes:
        ax.set_xlim(0, size - 1)
        ax.set_ylim(0, size - 1)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _write_csv(path: Path, result: VisitResult, snrs: Sequence[Any], meta: dict[str, Any]) -> None:
    fields = [
        "candidate_id",
        "patch",
        "band",
        "tile_id",
        "visit",
        "id",
        "x",
        "y",
        "score",
        "major",
        "minor",
        "theta",
        "snr",
        "snr_trusted",
        "snr_flux",
        "snr_sigma",
        "aperture_background",
        "aperture_pixels",
        "sky_aperture_count",
        "annulus_pixels",
        "internal",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, entry in zip(result.rows, snrs, strict=True):
            writer.writerow(
                {
                    "candidate_id": meta.get("candidate_id", ""),
                    "patch": meta.get("patch", ""),
                    "band": meta.get("band", ""),
                    "tile_id": meta.get("tile_id", ""),
                    "visit": meta.get("visit", ""),
                    "id": int(row.get("id", 0)),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "score": float(row.get("score", float("nan"))),
                    "major": float(row.get("major", float("nan"))),
                    "minor": float(row.get("minor", float("nan"))),
                    "theta": float(row.get("theta", float("nan"))),
                    "snr": float(entry.snr),
                    "snr_trusted": bool(entry.trusted),
                    "snr_flux": float(entry.flux),
                    "snr_sigma": float(entry.sigma),
                    "aperture_background": float(entry.background),
                    "aperture_pixels": int(entry.aperture_pixels),
                    "sky_aperture_count": int(entry.sky_aperture_count),
                    "annulus_pixels": int(entry.local_sky_aperture_count),
                    "internal": bool(entry.internal),
                }
            )


def _run_group(
    items: Sequence[SelectedCutout],
    *,
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    by_band = {item.band: item for item in items}
    bands = [band for band in args.band_order if band in by_band]
    bands.extend(sorted(set(by_band) - set(bands)))
    images = [_load_image(by_band[band]) for band in bands]
    scaled = [
        make_training_rgb(
            image,
            mode=str(args.scaling_mode),
            clip_threshold=float(args.clip_threshold),
            log_a=float(args.log_a),
            log_high_percentile=float(args.log_high_percentile),
            lupton_stretch=float(args.lupton_stretch),
            lupton_q=float(args.lupton_q),
            anscombe_clip=bool(args.anscombe_clip),
            anscombe_scale=float(args.anscombe_scale),
        )
        for image in images
    ]
    tensor = torch.from_numpy(np.stack(scaled, axis=0).astype(np.float32, copy=False))[None]
    outputs = infer_cellect(model=model, image_tensor=tensor, device=device, amp=str(args.amp))
    summaries: list[dict[str, Any]] = []
    for band_idx, band in enumerate(bands):
        item = by_band[band]
        band_outputs = select_band_outputs(outputs, band_idx)
        image = images[band_idx]
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
        score_map = _score_map(band_outputs)[: image.shape[0], : image.shape[1]].astype(np.float32, copy=False)
        result = VisitResult(label=item.candidate_id, ref=None, image=image, scaled=scaled[band_idx], score_map=score_map, rows=rows)  # type: ignore[arg-type]
        fields = _snr_noise_fields(args, result)
        snrs = [_measure_snr(args, result, fields, row) for row in rows]
        stem = f"{item.patch}_{band}_{item.tile_id}_{item.visit}".replace(",", "_")
        out_subdir = args.out_dir / str(item.patch) / str(band) / str(item.tile_id)
        _plot_panel(out_subdir / f"{stem}_confidence_snr_panel.png", result, snrs, item.meta, args)
        _write_csv(out_subdir / f"{stem}_detections_snr.csv", result, snrs, item.meta)
        summaries.append(
            {
                "candidate_id": item.candidate_id,
                "patch": item.patch,
                "band": band,
                "tile_id": item.tile_id,
                "visit": item.visit,
                "n_detections": len(rows),
                "n_snr_trusted": int(sum(bool(entry.trusted) for entry in snrs)),
                "n_snr_ge5": int(sum(bool(entry.trusted) and np.isfinite(entry.snr) and entry.snr >= 5.0 for entry in snrs)),
                "panel": str(out_subdir / f"{stem}_confidence_snr_panel.png"),
                "csv": str(out_subdir / f"{stem}_detections_snr.csv"),
                "aperture_noise_sigma": float(fields.model.sigma),
                "sky_aperture_count": int(fields.model.sky_aperture_count),
            }
        )
    return summaries


def main() -> int:
    args = parse_args()
    args.selected_root = args.selected_root.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    items = _selected_cutouts(args.selected_root)
    grouped: dict[tuple[str, str, int], list[SelectedCutout]] = {}
    for item in items:
        grouped.setdefault(item.group_key, []).append(item)
    device = torch.device(str(args.device) if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    model, _cfg = load_cellect_model(args.checkpoint.expanduser(), args.config, device, list(args.band_order), dynamic_image_size=True)
    summaries: list[dict[str, Any]] = []
    for key in sorted(grouped):
        summaries.extend(_run_group(grouped[key], model=model, device=device, args=args))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "selected_root": str(args.selected_root),
                "checkpoint": str(args.checkpoint),
                "scaling_mode": str(args.scaling_mode),
                "confidence_threshold": float(args.confidence_threshold),
                "nms_radius": int(args.nms_radius),
                "snr_background_box": int(args.snr_background_box),
                "snr_background_method": str(args.snr_background_method),
                "confidence_marker_radius": float(args.confidence_marker_radius),
                "items": summaries,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")
    print(f"wrote {len(summaries)} panels to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
