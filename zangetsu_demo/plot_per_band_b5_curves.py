#!/usr/bin/env python3
"""Plot training curves for per_band_b5 model variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


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


def _plot_metric(ax, all_rows: dict[str, list[dict]], metric: str, title: str) -> None:
    for label, rows in all_rows.items():
        for split, linestyle in (("train", "-"), ("val", "--")):
            x, y = _series(rows, split, metric)
            if x:
                ax.plot(x, y, linestyle=linestyle, linewidth=1.8, label=f"{label} {split}")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.25)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-log", type=Path, default=Path("output/per_band_b5_new.log"))
    parser.add_argument("--optimized-log", type=Path, default=Path("output/per_band_b5_optimized.log"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/per_band_b5_compare"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = {
        "new": _json_objects_from_log(args.new_log),
        "optimized": _json_objects_from_log(args.optimized_log),
    }
    summary = {name: len(rows) for name, rows in all_rows.items()}
    (args.out_dir / "curve_parse_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    _plot_metric(axes[0, 0], all_rows, "total", "Total loss")
    _plot_metric(axes[0, 1], all_rows, "shape", "Shape loss")
    _plot_metric(axes[1, 0], all_rows, "confidence", "Confidence loss")
    _plot_metric(axes[1, 1], all_rows, "seg", "Segmentation loss")
    for ax in axes.flat:
        ax.legend(fontsize=8)
    fig.savefig(args.out_dir / "per_band_b5_new_vs_optimized_losses.png", dpi=180)
    plt.close(fig)

    for metric in ("total", "shape", "confidence", "seg"):
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        _plot_metric(ax, all_rows, metric, f"{metric} loss")
        ax.legend(fontsize=8)
        fig.savefig(args.out_dir / f"{metric}_loss.png", dpi=180)
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
