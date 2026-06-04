#!/usr/bin/env python3
"""Plot per-model shape loss curves with independent y-axis limits."""

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


def _set_tight_ylim(ax, values: list[float]) -> None:
    if not values:
        return
    ymin = min(values)
    ymax = max(values)
    if ymin == ymax:
        pad = max(abs(ymin) * 0.05, 1.0)
    else:
        pad = (ymax - ymin) * 0.12
    ax.set_ylim(ymin - pad, ymax + pad)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-log", type=Path, default=Path("output/per_band_b5_new.log"))
    parser.add_argument("--optimized-log", type=Path, default=Path("output/per_band_b5_optimized.log"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/per_band_b5_compare"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_model = {
        "per_band_b5_new": _json_objects_from_log(args.new_log),
        "per_band_b5_optimized": _json_objects_from_log(args.optimized_log),
    }

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True, sharex=False)
    colors = {"train": "#1f77b4", "val": "#d62728"}
    for ax, (name, rows) in zip(axes, rows_by_model.items()):
        all_values: list[float] = []
        for split, linestyle in (("train", "-"), ("val", "--")):
            x, y = _series(rows, split, "shape")
            all_values.extend(y)
            ax.plot(x, y, linestyle=linestyle, linewidth=2.0, color=colors[split], label=f"{split} shape")
        _set_tight_ylim(ax, all_values)
        ax.set_title(f"{name} shape loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("shape loss")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
        ax.text(
            0.99,
            0.96,
            f"parsed epochs: {len(rows)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )

    out_path = args.out_dir / "shape_loss_new_optimized_separate_axes.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
