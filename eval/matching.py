#!/usr/bin/env python3
"""Match CELLECT detections against zarr source labels and write diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_utils import (  # noqa: E402
    PU_CLASS_NAMES,
    ellipse_line,
    read_zarr_sample,
    resolve_zarr_sample,
    rows_to_reg,
    save_png,
    source_rows_from_zarr,
    strict_centers_from_zarr,
    write_reg,
    zscale_gray,
)


MATCH_TP_COLOR = "cyan"
MATCH_FN_COLOR = "red"
MATCH_FP_COLOR = "magenta"
MATCH_CENTER_RADIUS = 7.0


def _with_tag(line: str, tag: str) -> str:
    return f"{line} tag={{{tag}}}"


def circle_line(x: float, y: float, *, radius: float = MATCH_CENTER_RADIUS, color: str, width: int = 2, text: str = "", tag: str = "") -> str:
    suffix = f" # color={color} width={int(width)}"
    if text:
        suffix += f" text={{{text}}}"
    if tag:
        suffix += f" tag={{{tag}}}"
    return f"circle({x + 1.0:.3f},{y + 1.0:.3f},{float(radius):.3f}){suffix}"


def load_prediction_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with Path(path).expanduser().open(newline="", encoding="ascii") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            try:
                x = float(raw.get("x", raw.get("x_image", "nan")))
                y = float(raw.get("y", raw.get("y_image", "nan")))
            except ValueError:
                continue
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            row: dict[str, float] = {"x": x, "y": y}
            for key in ("id", "score", "major", "minor", "theta", "x_image", "y_image", "ra_deg", "dec_deg"):
                value = raw.get(key)
                if value is None or value == "":
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
            rows.append(row)
    return rows


def find_prediction_csv(output_stem_dir: Path, band: str, dataset_source: str | None = None) -> Path:
    root = Path(output_stem_dir).expanduser()
    direct = root / str(band) / f"{band}_sources.csv"
    if direct.exists():
        return direct
    candidates = sorted(root.rglob(f"{band}_sources.csv"))
    if dataset_source:
        candidates = [path for path in candidates if str(dataset_source) in path.parts]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"no {band}_sources.csv found under {root}")
    raise RuntimeError(f"multiple {band}_sources.csv files found under {root}; pass --pred-csv")


def ground_truth_rows_from_zarr(reader, sample_idx: int, band_idx: int) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    seen: set[tuple[int, int, int] | tuple[str, int]] = set()

    def key_for(row: dict[str, float]) -> tuple[int, int, int] | tuple[str, int]:
        source_id = int(row.get("source_id", -1))
        if source_id >= 0:
            return ("id", source_id)
        return ("xy", int(round(float(row["x"]) * 1000.0)), int(round(float(row["y"]) * 1000.0)))

    def append_unique(row: dict[str, float]) -> None:
        key = key_for(row)
        if key in seen:
            return
        seen.add(key)
        rows.append(row)

    for row in source_rows_from_zarr(reader, sample_idx, band_idx):
        class_id = int(row.get("class_id", 1))
        if class_id in {1, 2, 5}:
            out = dict(row)
            out["has_shape"] = 1.0
            append_unique(out)
    for row in strict_centers_from_zarr(reader, sample_idx, band_idx):
        out = dict(row)
        out.setdefault("major", 1.5)
        out.setdefault("minor", 1.5)
        out.setdefault("theta", 0.0)
        out["class_id"] = 5
        out["has_shape"] = 0.0
        append_unique(out)
    return rows


def greedy_match(
    pred_rows: Sequence[dict[str, float]],
    gt_rows: Sequence[dict[str, float]],
    *,
    radius: float = 3.0,
) -> tuple[dict[int, int], dict[int, int], list[tuple[int, int, float]]]:
    pairs: list[tuple[float, int, int]] = []
    r2 = float(radius) * float(radius)
    for pred_idx, pred in enumerate(pred_rows):
        px, py = float(pred["x"]), float(pred["y"])
        for gt_idx, gt in enumerate(gt_rows):
            dx = px - float(gt["x"])
            dy = py - float(gt["y"])
            d2 = dx * dx + dy * dy
            if d2 <= r2:
                pairs.append((math.sqrt(d2), pred_idx, gt_idx))
    pred_to_gt: dict[int, int] = {}
    gt_to_pred: dict[int, int] = {}
    matched: list[tuple[int, int, float]] = []
    for dist, pred_idx, gt_idx in sorted(pairs, key=lambda item: item[0]):
        if pred_idx in pred_to_gt or gt_idx in gt_to_pred:
            continue
        pred_to_gt[pred_idx] = gt_idx
        gt_to_pred[gt_idx] = pred_idx
        matched.append((pred_idx, gt_idx, dist))
    return pred_to_gt, gt_to_pred, matched


def _summary(pred_rows: Sequence[dict[str, float]], gt_rows: Sequence[dict[str, float]], matches: Sequence[tuple[int, int, float]], radius: float) -> dict:
    tp = len(matches)
    fp = max(0, len(pred_rows) - tp)
    fn = max(0, len(gt_rows) - tp)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    gt_by_class: dict[str, int] = {}
    fn_by_class: dict[str, int] = {}
    matched_gt = {gt_idx for _, gt_idx, _ in matches}
    for gt_idx, gt in enumerate(gt_rows):
        label = PU_CLASS_NAMES.get(int(gt.get("class_id", 0)), str(int(gt.get("class_id", 0))))
        gt_by_class[label] = gt_by_class.get(label, 0) + 1
        if gt_idx not in matched_gt:
            fn_by_class[label] = fn_by_class.get(label, 0) + 1
    return {
        "match_radius_px": float(radius),
        "gt": len(gt_rows),
        "pred": len(pred_rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "purity": precision,
        "recall": recall,
        "completeness": recall,
        "f1": f1,
        "gt_by_class": gt_by_class,
        "fn_by_class": fn_by_class,
    }


def _matching_reg_lines(
    pred_rows: Sequence[dict[str, float]],
    gt_rows: Sequence[dict[str, float]],
    pred_to_gt: dict[int, int],
    gt_to_pred: dict[int, int],
    *,
    shape: bool,
) -> list[str]:
    lines: list[str] = []
    for pred_idx, pred in enumerate(pred_rows):
        color = MATCH_TP_COLOR if pred_idx in pred_to_gt else MATCH_FP_COLOR
        label = "TP" if pred_idx in pred_to_gt else "FP"
        tag = f"{label} {'shape' if shape else 'center'}"
        text = f"{label} pred={pred_idx + 1} score={float(pred.get('score', float('nan'))):.3g}"
        if pred_idx in pred_to_gt:
            text += f" gt={pred_to_gt[pred_idx] + 1}"
        if shape:
            lines.append(
                _with_tag(
                    ellipse_line(
                        float(pred["x"]),
                        float(pred["y"]),
                        float(pred.get("major", 1.0)),
                        float(pred.get("minor", 1.0)),
                        float(pred.get("theta", 0.0)),
                        color=color,
                        width=2,
                        text=text,
                    ),
                    tag,
                )
            )
        else:
            lines.append(
                circle_line(
                    float(pred["x"]),
                    float(pred["y"]),
                    color=color,
                    width=2,
                    text=text,
                    tag=tag,
                )
            )
    for gt_idx, gt in enumerate(gt_rows):
        if gt_idx in gt_to_pred:
            continue
        label = PU_CLASS_NAMES.get(int(gt.get("class_id", 0)), "gt")
        text = f"FN gt={gt_idx + 1} class={label} source={int(gt.get('source_id', -1))}"
        if shape and bool(gt.get("has_shape", 0.0)):
            lines.append(
                _with_tag(
                    ellipse_line(
                        float(gt["x"]),
                        float(gt["y"]),
                        float(gt.get("major", 1.0)),
                        float(gt.get("minor", 1.0)),
                        float(gt.get("theta", 0.0)),
                        color=MATCH_FN_COLOR,
                        width=2,
                        text=text,
                    ),
                    "FN shape",
                )
            )
        else:
            lines.append(
                circle_line(
                    float(gt["x"]),
                    float(gt["y"]),
                    color=MATCH_FN_COLOR,
                    width=2,
                    text=text,
                    tag="FN center",
                )
            )
    return lines


def _write_match_csv(
    path: Path,
    pred_rows: Sequence[dict[str, float]],
    gt_rows: Sequence[dict[str, float]],
    pred_to_gt: dict[int, int],
    gt_to_pred: dict[int, int],
    matched: Sequence[tuple[int, int, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dist_by_pair = {(pred_idx, gt_idx): dist for pred_idx, gt_idx, dist in matched}
    fields = (
        "kind",
        "pred_index",
        "gt_index",
        "distance_px",
        "x",
        "y",
        "score",
        "major",
        "minor",
        "theta",
        "gt_x",
        "gt_y",
        "gt_source_id",
        "gt_class",
    )
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pred_idx, pred in enumerate(pred_rows):
            gt_idx = pred_to_gt.get(pred_idx)
            gt = gt_rows[gt_idx] if gt_idx is not None else {}
            writer.writerow(
                {
                    "kind": "tp" if gt_idx is not None else "fp",
                    "pred_index": pred_idx + 1,
                    "gt_index": "" if gt_idx is None else gt_idx + 1,
                    "distance_px": "" if gt_idx is None else dist_by_pair[(pred_idx, gt_idx)],
                    "x": pred.get("x", ""),
                    "y": pred.get("y", ""),
                    "score": pred.get("score", ""),
                    "major": pred.get("major", ""),
                    "minor": pred.get("minor", ""),
                    "theta": pred.get("theta", ""),
                    "gt_x": gt.get("x", ""),
                    "gt_y": gt.get("y", ""),
                    "gt_source_id": gt.get("source_id", ""),
                    "gt_class": PU_CLASS_NAMES.get(int(gt.get("class_id", 0)), "") if gt_idx is not None else "",
                }
            )
        for gt_idx, gt in enumerate(gt_rows):
            if gt_idx in gt_to_pred:
                continue
            writer.writerow(
                {
                    "kind": "fn",
                    "pred_index": "",
                    "gt_index": gt_idx + 1,
                    "distance_px": "",
                    "x": "",
                    "y": "",
                    "score": "",
                    "major": "",
                    "minor": "",
                    "theta": "",
                    "gt_x": gt.get("x", ""),
                    "gt_y": gt.get("y", ""),
                    "gt_source_id": gt.get("source_id", ""),
                    "gt_class": PU_CLASS_NAMES.get(int(gt.get("class_id", 0)), ""),
                }
            )


def _draw_match_overlay(
    image: np.ndarray,
    pred_rows: Sequence[dict[str, float]],
    gt_rows: Sequence[dict[str, float]],
    pred_to_gt: dict[int, int],
    gt_to_pred: dict[int, int],
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Ellipse

    fig, ax = plt.subplots(figsize=(image.shape[1] / 100.0, image.shape[0] / 100.0), dpi=100)
    ax.imshow(zscale_gray(image), origin="lower", cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")

    def draw_row(row: dict[str, float], *, color: str, marker_only: bool = False) -> None:
        x = float(row["x"])
        y = float(row["y"])
        if not marker_only:
            ax.add_patch(
                Ellipse(
                    (x, y),
                    width=2.0 * max(abs(float(row.get("major", 1.0))), 1.0),
                    height=2.0 * max(abs(float(row.get("minor", 1.0))), 1.0),
                    angle=math.degrees(float(row.get("theta", 0.0))),
                    fill=False,
                    edgecolor=color,
                    linewidth=1.2,
                    alpha=0.95,
                )
            )
        ax.plot(x, y, marker="+", color=color, markersize=4.0, mew=1.0)

    for pred_idx, pred in enumerate(pred_rows):
        draw_row(dict(pred), color=MATCH_TP_COLOR if pred_idx in pred_to_gt else MATCH_FP_COLOR)
    for gt_idx, gt in enumerate(gt_rows):
        if gt_idx in gt_to_pred:
            continue
        draw_row(dict(gt), color=MATCH_FN_COLOR, marker_only=not bool(gt.get("has_shape", 0.0)))

    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    plt.close(fig)
    return np.flipud(rgba[..., :3]).astype(np.float32) / 255.0


def write_matching_diagnostics(
    *,
    out_dir: Path,
    pred_rows: Sequence[dict[str, float]],
    gt_rows: Sequence[dict[str, float]],
    image: np.ndarray | None = None,
    band: str = "band",
    match_radius: float = 3.0,
) -> dict:
    pred_to_gt, gt_to_pred, matched = greedy_match(pred_rows, gt_rows, radius=float(match_radius))
    summary = _summary(pred_rows, gt_rows, matched, float(match_radius))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{band}_matching_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_match_csv(out_dir / f"{band}_matching_sources.csv", pred_rows, gt_rows, pred_to_gt, gt_to_pred, matched)
    write_reg(
        out_dir / f"{band}_matching_centers.reg",
        _matching_reg_lines(pred_rows, gt_rows, pred_to_gt, gt_to_pred, shape=False),
    )
    write_reg(
        out_dir / f"{band}_matching_shapes.reg",
        _matching_reg_lines(pred_rows, gt_rows, pred_to_gt, gt_to_pred, shape=True),
    )
    write_reg(out_dir / f"{band}_gt_sources.reg", rows_to_reg(gt_rows, shape=True, color=None))
    if image is not None:
        save_png(
            out_dir / f"{band}_matching_overlay.png",
            _draw_match_overlay(np.asarray(image, dtype=np.float32), pred_rows, gt_rows, pred_to_gt, gt_to_pred),
            title=f"{band} match: TP cyan, FN red, other detections magenta",
        )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-csv", type=Path, default=None, help="Prediction CSV, usually <band>_sources.csv.")
    p.add_argument("--output-stem-dir", type=Path, default=None, help="Existing cellect output stem directory.")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--band", required=True)
    p.add_argument("--match-radius", type=float, default=3.0)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--zarr-store", type=Path)
    src.add_argument("--root", type=Path)
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--patch", default=None)
    p.add_argument("--tile-name", default=None)
    p.add_argument("--dataset-source", default=None)
    p.add_argument("--group", default=None)
    p.add_argument("--image-level", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pred_csv = args.pred_csv
    if pred_csv is None:
        if args.output_stem_dir is None:
            raise ValueError("provide --pred-csv or --output-stem-dir")
        pred_csv = find_prediction_csv(args.output_stem_dir, args.band, args.dataset_source)
    pred_rows = load_prediction_csv(pred_csv)
    reader, sample_idx, band_idx, _attrs = resolve_zarr_sample(
        zarr_store=args.zarr_store.expanduser().resolve() if args.zarr_store else None,
        sample_index=int(args.sample_index),
        root=args.root.expanduser().resolve() if args.root else None,
        patch=args.patch,
        tile_name=args.tile_name,
        band=args.band,
        dataset_source=args.dataset_source,
        group=args.group,
        image_level=bool(args.image_level),
    )
    gt_rows = ground_truth_rows_from_zarr(reader, sample_idx, band_idx)
    sample = read_zarr_sample(reader, sample_idx, band_idx)
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = pred_csv.parent / "matching"
    summary = write_matching_diagnostics(
        out_dir=out_dir,
        pred_rows=pred_rows,
        gt_rows=gt_rows,
        image=sample["display_image"],
        band=args.band,
        match_radius=float(args.match_radius),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
