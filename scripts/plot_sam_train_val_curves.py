#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_LOG = Path("/home/czh23/CELLECT/output/training_logs/SAM_per_band_debug_0612.log")
DEFAULT_OUT_DIR = Path("/home/czh23/CELLECT/output/training_logs/sam_per_band_debug_0612_curves")


def _json_objects_from_log(path: Path) -> list[dict]:
    text = path.read_text(errors="ignore")
    rows: list[dict] = []
    depth = 0
    start: int | None = None
    in_string = False
    escape = False

    for idx, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                blob = text[start : idx + 1]
                start = None
                try:
                    obj = json.loads(blob)
                except Exception:
                    continue
                if isinstance(obj, dict) and "epoch" in obj and "train" in obj and "val" in obj:
                    rows.append(obj)
    return rows


def _series(rows: list[dict], split: str, key: str) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    values: list[float] = []
    for row in rows:
        metrics = row.get(split, {})
        if key not in metrics:
            continue
        epochs.append(int(row["epoch"]) + 1)
        values.append(float(metrics[key]))
    return epochs, values


def _plot_metric(ax, rows: list[dict], metric: str, title: str) -> None:
    for split, linestyle, color in (("train", "-", "#1f77b4"), ("val", "--", "#d62728")):
        x, y = _series(rows, split, metric)
        if x:
            ax.plot(x, y, linestyle=linestyle, linewidth=2.0, color=color, label=split)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot train/val curves from SAM per-band debug log.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rows = _json_objects_from_log(args.log.expanduser())
    if not rows:
        raise SystemExit(f"No epoch JSON blocks found in {args.log}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "log": str(args.log.expanduser()),
        "num_epochs": len(rows),
        "first_epoch": int(rows[0]["epoch"]) + 1,
        "last_epoch": int(rows[-1]["epoch"]) + 1,
    }
    (args.out_dir / "parse_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Main figure: total loss train/val curve.
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    _plot_metric(ax, rows, "total", "Train/Val Total Loss")
    fig.savefig(args.out_dir / "train_val_total_loss.png", dpi=180)
    plt.close(fig)

    # Auxiliary figure: component curves.
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    metric_specs = [
        ("total", "Total Loss"),
        ("seg", "Segmentation Loss"),
        ("confidence", "Confidence Loss"),
        ("shape", "Shape Loss"),
    ]
    for ax, (metric, title) in zip(axes.flat, metric_specs):
        _plot_metric(ax, rows, metric, title)
    fig.savefig(args.out_dir / "train_val_loss_components.png", dpi=180)
    plt.close(fig)

    print(args.out_dir / "train_val_total_loss.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())