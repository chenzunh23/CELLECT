"""
Summarize CELLECT extradata annotation sizes and per-image cell counts.

Outputs are written under output/extradata_annotation_stats by default:
  - sequence_summary.csv
  - per_frame_counts.csv
  - track_radius_annotations.csv
  - nuclei_size_annotations.csv
  - several PNG visualizations

The main CELLECT training annotations are tracks.txt. The tracks/nuclei files are
also parsed because they include raw nuclei entries; these can include unnamed
candidate nuclei that are not part of the tracked-cell table.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _float(value: str) -> float:
    value = value.strip()
    if value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _int(value: str) -> int:
    value = value.strip()
    if value == "":
        return -1
    try:
        return int(float(value))
    except ValueError:
        return -1


def _stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_tracks(path: Path, sequence: str, annotation_type: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            radius = _float(row.get("radius", ""))
            frame = _int(row.get("t", ""))
            rows.append(
                {
                    "sequence": sequence,
                    "annotation_type": annotation_type,
                    "frame": frame,
                    "z": _float(row.get("z", "")),
                    "y": _float(row.get("y", "")),
                    "x": _float(row.get("x", "")),
                    "cell_id": _int(row.get("cell_id", "")),
                    "parent_id": _int(row.get("parent_id", "")),
                    "track_id": _int(row.get("track_id", "")),
                    "radius": radius,
                    "diameter": radius * 2.0 if np.isfinite(radius) else float("nan"),
                    "name": row.get("name", ""),
                    "div_state": _int(row.get("div_state", "")),
                }
            )
    return rows


def parse_nuclei_file(path: Path, sequence: str) -> List[Dict[str, object]]:
    match = re.search(r"t(\d+)-nuclei$", path.name)
    if match is None:
        return []
    # StarryNite-style files are 1-indexed while tracks.txt uses t=0 for t001.
    frame = int(match.group(1)) - 1
    rows: List[Dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for raw in reader:
            cols = [c.strip() for c in raw]
            if len(cols) < 11:
                continue
            diameter = _float(cols[8])
            rows.append(
                {
                    "sequence": sequence,
                    "frame": frame,
                    "nucleus_id": _int(cols[0]),
                    "status": _int(cols[1]),
                    "x": _float(cols[5]),
                    "y": _float(cols[6]),
                    "z_index": _float(cols[7]),
                    "diameter": diameter,
                    "radius": diameter / 2.0 if np.isfinite(diameter) else float("nan"),
                    "name": cols[9],
                    "raw_volume_like": _float(cols[10]),
                    "is_named": cols[9] != "",
                }
            )
    return rows


def count_image_files(sequence_dir: Path) -> Dict[int, str]:
    images: Dict[int, str] = {}
    for path in sorted((sequence_dir / "images").glob("*.tif*")):
        match = re.search(r"_t(\d+)", path.name)
        if match is not None:
            images[int(match.group(1))] = str(path)
    return images


def summarize_sequence(
    sequence_dir: Path,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    sequence = sequence_dir.name
    images = count_image_files(sequence_dir)
    tracks = read_tracks(sequence_dir / "tracks" / "tracks.txt", sequence, "cell")
    polar = read_tracks(sequence_dir / "tracks" / "tracks_polar_bodies.txt", sequence, "polar_body")

    nuclei_rows: List[Dict[str, object]] = []
    nuclei_dir = sequence_dir / "tracks" / "nuclei"
    if nuclei_dir.exists():
        for path in sorted(nuclei_dir.glob("t*-nuclei")):
            nuclei_rows.extend(parse_nuclei_file(path, sequence))

    frames = sorted(set(images) | {int(r["frame"]) for r in tracks} | {int(r["frame"]) for r in nuclei_rows})
    track_by_frame: Dict[int, int] = defaultdict(int)
    polar_by_frame: Dict[int, int] = defaultdict(int)
    nuclei_by_frame: Dict[int, int] = defaultdict(int)
    named_nuclei_by_frame: Dict[int, int] = defaultdict(int)
    for r in tracks:
        track_by_frame[int(r["frame"])] += 1
    for r in polar:
        polar_by_frame[int(r["frame"])] += 1
    for r in nuclei_rows:
        nuclei_by_frame[int(r["frame"])] += 1
        if bool(r["is_named"]):
            named_nuclei_by_frame[int(r["frame"])] += 1

    per_frame_rows: List[Dict[str, object]] = []
    for frame in frames:
        per_frame_rows.append(
            {
                "sequence": sequence,
                "frame": frame,
                "image_file": images.get(frame, ""),
                "has_image": frame in images,
                "tracked_cell_count": track_by_frame.get(frame, 0),
                "polar_body_count": polar_by_frame.get(frame, 0),
                "nuclei_entry_count": nuclei_by_frame.get(frame, 0),
                "named_nuclei_count": named_nuclei_by_frame.get(frame, 0),
            }
        )

    track_radii = [float(r["radius"]) for r in tracks]
    nuclei_radii = [float(r["radius"]) for r in nuclei_rows]
    counts = [int(r["tracked_cell_count"]) for r in per_frame_rows if bool(r["has_image"])]
    raw_counts = [int(r["nuclei_entry_count"]) for r in per_frame_rows if bool(r["has_image"])]
    named_counts = [int(r["named_nuclei_count"]) for r in per_frame_rows if bool(r["has_image"])]
    radius_stats = _stats(track_radii)
    nuclei_radius_stats = _stats(nuclei_radii)
    count_stats = _stats(counts)
    raw_count_stats = _stats(raw_counts)
    named_count_stats = _stats(named_counts)

    summary = {
        "sequence": sequence,
        "image_count": len(images),
        "track_annotation_count": len(tracks),
        "polar_body_annotation_count": len(polar),
        "nuclei_entry_count": len(nuclei_rows),
        "tracked_cell_count_mean": count_stats["mean"],
        "tracked_cell_count_min": count_stats["min"],
        "tracked_cell_count_median": count_stats["median"],
        "tracked_cell_count_max": count_stats["max"],
        "nuclei_entry_count_mean": raw_count_stats["mean"],
        "nuclei_entry_count_min": raw_count_stats["min"],
        "nuclei_entry_count_median": raw_count_stats["median"],
        "nuclei_entry_count_max": raw_count_stats["max"],
        "named_nuclei_count_mean": named_count_stats["mean"],
        "named_nuclei_count_min": named_count_stats["min"],
        "named_nuclei_count_median": named_count_stats["median"],
        "named_nuclei_count_max": named_count_stats["max"],
        "radius_mean": radius_stats["mean"],
        "radius_std": radius_stats["std"],
        "radius_min": radius_stats["min"],
        "radius_p05": radius_stats["p05"],
        "radius_p25": radius_stats["p25"],
        "radius_median": radius_stats["median"],
        "radius_p75": radius_stats["p75"],
        "radius_p95": radius_stats["p95"],
        "radius_max": radius_stats["max"],
        "nuclei_radius_mean": nuclei_radius_stats["mean"],
        "nuclei_radius_median": nuclei_radius_stats["median"],
        "nuclei_radius_min": nuclei_radius_stats["min"],
        "nuclei_radius_max": nuclei_radius_stats["max"],
    }
    return summary, per_frame_rows, tracks + polar, nuclei_rows


def plot_counts(per_frame: Sequence[Dict[str, object]], out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, sequence in zip(axes, sorted({str(r["sequence"]) for r in per_frame})):
        rows = sorted([r for r in per_frame if r["sequence"] == sequence], key=lambda r: int(r["frame"]))
        frames = np.asarray([int(r["frame"]) for r in rows])
        tracked = np.asarray([int(r["tracked_cell_count"]) for r in rows])
        nuclei = np.asarray([int(r["nuclei_entry_count"]) for r in rows])
        named = np.asarray([int(r["named_nuclei_count"]) for r in rows])
        ax.plot(frames, tracked, label="tracks.txt cells", linewidth=1.6)
        ax.plot(frames, nuclei, label="raw nuclei entries", linewidth=1.0, alpha=0.65)
        ax.plot(frames, named, label="named nuclei", linewidth=1.0, alpha=0.8)
        ax.set_title(sequence)
        ax.set_ylabel("count/image")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_radius_distribution(track_rows: Sequence[Dict[str, object]], nuclei_rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    sequences = sorted({str(r["sequence"]) for r in track_rows if r["annotation_type"] == "cell"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    box_data = []
    labels = []
    for sequence in sequences:
        vals = [float(r["radius"]) for r in track_rows if r["sequence"] == sequence and r["annotation_type"] == "cell"]
        vals = [v for v in vals if np.isfinite(v)]
        axes[0].hist(vals, bins=35, alpha=0.45, density=True, label=sequence)
        box_data.append(vals)
        labels.append(sequence)
    # Add vertical lines for median, 5th and 95th percentiles across all sequences.
    all_vals = [float(r["radius"]) for r in track_rows if r["annotation_type"] == "cell"]
    all_vals = [v for v in all_vals if np.isfinite(v)]
    if all_vals:
        median = np.median(all_vals)
        p05 = np.percentile(all_vals, 5)
        p95 = np.percentile(all_vals, 95)
        axes[0].axvline(median, color="black", linestyle="--", linewidth=1.0, label="median")
        axes[0].axvline(p05, color="red", linestyle=":", linewidth=1.0, label="5th percentile")
        axes[0].axvline(p95, color="blue", linestyle=":", linewidth=1.0, label="95th percentile")
    axes[0].set_title("Tracked-cell radius distribution")
    axes[0].set_xlabel("radius")
    axes[0].set_ylabel("density")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].boxplot(box_data, labels=labels, showfliers=False)
    axes[1].set_title("Tracked-cell radius by sequence")
    axes[1].set_ylabel("radius")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    if nuclei_rows:
        fig, ax = plt.subplots(figsize=(8, 5))
        for sequence in sequences:
            vals = [float(r["radius"]) for r in nuclei_rows if r["sequence"] == sequence]
            vals = [v for v in vals if np.isfinite(v)]
            ax.hist(vals, bins=35, alpha=0.45, density=True, label=sequence)
        ax.set_title("Raw nuclei radius distribution")
        ax.set_xlabel("radius = nuclei diameter / 2")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path.with_name("nuclei_radius_distribution.png"), dpi=180)
        plt.close(fig)


def plot_radius_over_time(track_rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    sequences = sorted({str(r["sequence"]) for r in track_rows if r["annotation_type"] == "cell"})
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, sequence in zip(axes, sequences):
        rows = [r for r in track_rows if r["sequence"] == sequence and r["annotation_type"] == "cell"]
        by_frame: Dict[int, List[float]] = defaultdict(list)
        for r in rows:
            radius = float(r["radius"])
            if np.isfinite(radius):
                by_frame[int(r["frame"])].append(radius)
        frames = np.asarray(sorted(by_frame))
        med = np.asarray([np.median(by_frame[f]) for f in frames])
        q25 = np.asarray([np.percentile(by_frame[f], 25) for f in frames])
        q75 = np.asarray([np.percentile(by_frame[f], 75) for f in frames])
        ax.plot(frames, med, label="median radius", linewidth=1.5)
        ax.fill_between(frames, q25, q75, alpha=0.25, label="IQR")
        ax.set_title(sequence)
        ax.set_ylabel("radius")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_count_hist(per_frame: Sequence[Dict[str, object]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sequences = sorted({str(r["sequence"]) for r in per_frame})
    for sequence in sequences:
        rows = [r for r in per_frame if r["sequence"] == sequence and bool(r["has_image"])]
        tracked = [int(r["tracked_cell_count"]) for r in rows]
        nuclei = [int(r["nuclei_entry_count"]) for r in rows]
        axes[0].hist(tracked, bins=30, alpha=0.45, label=sequence)
        axes[1].hist(nuclei, bins=30, alpha=0.45, label=sequence)
    axes[0].set_title("Tracked cells per image")
    axes[1].set_title("Raw nuclei entries per image")
    for ax in axes:
        ax.set_xlabel("count")
        ax.set_ylabel("frames")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze extradata annotation size/count distributions.")
    parser.add_argument("--data-root", default="extradata/mskcc-confocal")
    parser.add_argument("--out-dir", default="output/extradata_annotation_stats")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, object]] = []
    all_per_frame: List[Dict[str, object]] = []
    all_track_rows: List[Dict[str, object]] = []
    all_nuclei_rows: List[Dict[str, object]] = []
    for sequence_dir in sorted(data_root.glob("mskcc_confocal_s*")):
        if not sequence_dir.is_dir():
            continue
        summary, per_frame, track_rows, nuclei_rows = summarize_sequence(sequence_dir)
        summaries.append(summary)
        all_per_frame.extend(per_frame)
        all_track_rows.extend(track_rows)
        all_nuclei_rows.extend(nuclei_rows)

    summary_fields = list(summaries[0].keys()) if summaries else []
    per_frame_fields = [
        "sequence",
        "frame",
        "image_file",
        "has_image",
        "tracked_cell_count",
        "polar_body_count",
        "nuclei_entry_count",
        "named_nuclei_count",
    ]
    track_fields = [
        "sequence",
        "annotation_type",
        "frame",
        "z",
        "y",
        "x",
        "cell_id",
        "parent_id",
        "track_id",
        "radius",
        "diameter",
        "name",
        "div_state",
    ]
    nuclei_fields = [
        "sequence",
        "frame",
        "nucleus_id",
        "status",
        "x",
        "y",
        "z_index",
        "diameter",
        "radius",
        "name",
        "raw_volume_like",
        "is_named",
    ]

    write_csv(out_dir / "sequence_summary.csv", summaries, summary_fields)
    write_csv(out_dir / "per_frame_counts.csv", all_per_frame, per_frame_fields)
    write_csv(out_dir / "track_radius_annotations.csv", all_track_rows, track_fields)
    write_csv(out_dir / "nuclei_size_annotations.csv", all_nuclei_rows, nuclei_fields)

    plot_counts(all_per_frame, out_dir / "cell_counts_per_frame.png")
    plot_count_hist(all_per_frame, out_dir / "cell_count_histograms.png")
    plot_radius_distribution(all_track_rows, all_nuclei_rows, out_dir / "track_radius_distribution.png")
    plot_radius_over_time(all_track_rows, out_dir / "track_radius_over_time.png")

    total = {
        "sequence": "ALL",
        "image_count": sum(int(s["image_count"]) for s in summaries),
        "track_annotation_count": sum(int(s["track_annotation_count"]) for s in summaries),
        "polar_body_annotation_count": sum(int(s["polar_body_annotation_count"]) for s in summaries),
        "nuclei_entry_count": sum(int(s["nuclei_entry_count"]) for s in summaries),
    }
    print("Wrote:", out_dir)
    print("Total:", total)
    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
