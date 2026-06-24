from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_METRICS = Path("/home/czh23/CELLECT/zangetsu_demo/eval_old_ckpt/reg_diagnostics_by_band/zangetsu_reg_metrics.json")
DEFAULT_OUTPUT = Path("/home/czh23/CELLECT/zangetsu_demo/eval_old_ckpt/reg_diagnostics_by_band/zangetsu_lower_right_x27366_y6453_HSC-I_stacked_bar.png")
DEFAULT_TILE = "zangetsu_lower_right_x27366_y6453"
DEFAULT_BAND = "HSC-I"


def build_plot(metrics_path: Path, output_path: Path, *, tile: str, band: str) -> None:
    rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    selected = [
        row
        for row in rows
        if row.get("level") == "band"
        and row.get("tile") == tile
        and row.get("band") == band
        and row.get("dataset") in {"coadd", "noisy", "denoised"}
    ]
    order = {"coadd": 0, "noisy": 1, "denoised": 2}
    selected.sort(key=lambda row: order[str(row["dataset"])])

    if len(selected) != 3:
        raise SystemExit(f"Expected 3 rows for coadd/noisy/denoised, found {len(selected)}")

    datasets = [str(row["dataset"]).capitalize() for row in selected]
    tp = [int(row["TP"]) for row in selected]
    fp = [int(row["FP"]) for row in selected]
    gt = int(selected[0]["GT"])

    x = range(len(datasets))
    width = 0.62

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=160)
    colors = {"tp": "#4C78A8", "fp": "#F58518"}

    tp_bars = ax.bar(x, tp, width=width, color=colors["tp"], label="TP")
    fp_bars = ax.bar(x, fp, width=width, bottom=tp, color=colors["fp"], label="FP")

    ax.axhline(gt, color="#D62728", linewidth=2.2, linestyle="-", label=f"GT = {gt}")
    ax.set_xticks(list(x), datasets)
    ax.set_ylabel("Source count")
    ax.set_title(f"{tile} / {band}: stacked TP + FP by dataset")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    ymax = max(max(t + f for t, f in zip(tp, fp)), gt)
    ax.set_ylim(0, int(ymax * 1.18) + 5)

    for bar, value in zip(tp_bars, tp):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value / 2,
            str(value),
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )

    for bar, base, value in zip(fp_bars, tp, fp):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            base + value / 2,
            str(value),
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )

    for idx, total in enumerate(t + f for t, f in zip(tp, fp)):
        ax.text(idx, total + 8, f"{total}", ha="center", va="bottom", fontsize=10)

    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot stacked TP/FP bars for zangetsu metrics.")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--tile", default=DEFAULT_TILE)
    parser.add_argument("--band", default=DEFAULT_BAND)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    build_plot(args.metrics.expanduser(), args.output.expanduser(), tile=args.tile, band=args.band)
    print(args.output.expanduser())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())