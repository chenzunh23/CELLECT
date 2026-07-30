#!/usr/bin/env python3
"""Export DS9 regions for non-huge dropped bright-cluster diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics-csv",
        type=Path,
        default=Path(
            "output/data_filter_0728/external_bright_labels/9813/4,5/"
            "9813_4_5_large_component_dropped_diagnostics.csv"
        ),
    )
    parser.add_argument(
        "--reclassification-root",
        type=Path,
        default=Path("output/data_filter_0728/external_bright_labels/9813/4,5"),
    )
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patch", default="4,5")
    parser.add_argument("--max-area", type=float, default=10000.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def region_text(row: dict[str, str]) -> str:
    parts = [
        f"sid={row.get('source_id', '')}",
        f"class={row.get('class', '')}",
        f"mag={safe_float(row.get('mag')):.2f}",
        f"area={safe_float(row.get('area')):.0f}",
        f"component_area={safe_float(row.get('component_area')):.0f}",
        f"mask={row.get('center_mask') or 'NONE'}",
        f"gaia={safe_float(row.get('nearest_gaia_arcsec')):.2f}arcsec",
        f"near_clean={row.get('nearby_supervised_count_50px', '')}",
        f"near_drop={row.get('nearby_dropped_count_50px', '')}",
        f"reason={row.get('diagnostic_categories', '')}",
    ]
    flags = row.get("true_flags", "")
    if flags:
        parts.append(f"flags={flags}")
    text = " ".join(parts)
    return text.replace("{", "(").replace("}", ")").replace("\n", " ")


def source_color(row: dict[str, str]) -> str:
    categories = row.get("diagnostic_categories", "")
    center_mask = row.get("center_mask", "")
    if "SAT" in center_mask or "BAD" in center_mask or "EDGE" in center_mask:
        return "magenta"
    if center_mask:
        return "orange"
    if "nearby_supervised_sources_within_50px" in categories:
        return "yellow"
    if "nearby_dropped_sources_within_50px" in categories:
        return "green"
    if row.get("true_flags", ""):
        return "red"
    return "cyan"


def load_reclassification(root: Path, tract: str, patch: str, band: str) -> dict[tuple[str, str], dict[str, str]]:
    patch_token = patch.replace(",", "_")
    path = root / band / f"{tract}_{patch_token}_{band}_bright_reclassification.csv"
    if not path.exists():
        return {}
    rows = read_rows(path)
    return {(row.get("source_id", ""), row.get("row_index", "")): row for row in rows}


def ellipse_params(row: dict[str, str], rec: dict[str, str] | None) -> tuple[float, float, float, float, float]:
    x = safe_float(row.get("x"))
    y = safe_float(row.get("y"))
    major = minor = theta = float("nan")
    if rec:
        x = safe_float(rec.get("output_x"), x)
        y = safe_float(rec.get("output_y"), y)
        major = safe_float(rec.get("major"))
        minor = safe_float(rec.get("minor"))
        theta = safe_float(rec.get("theta_deg"), 0.0)
    if not (math.isfinite(major) and math.isfinite(minor) and major > 0 and minor > 0):
        area = safe_float(row.get("area"))
        ratio = max(safe_float(row.get("axis_ratio"), 1.0), 1.0)
        if math.isfinite(area) and area > 0:
            major = math.sqrt(area * ratio / math.pi)
            minor = math.sqrt(area / (math.pi * ratio))
            theta = 0.0
    return x, y, major, minor, theta


def write_band_reg(
    *,
    rows: list[dict[str, str]],
    rec_by_key: dict[tuple[str, str], dict[str, str]],
    path: Path,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with path.open("w") as handle:
        handle.write("# Region file format: DS9 version 4.1\n")
        handle.write('global color=cyan dashlist=8 3 width=2 font="helvetica 9 normal roman"\n')
        handle.write("# color meaning: magenta=SAT/BAD/EDGE center, orange=other mask center, "
                     "yellow=near supervised, green=near dropped, red=bad flags, cyan=no obvious flag\n")
        handle.write("image\n")
        for row in rows:
            if safe_float(row.get("area"), 0.0) >= 10000.0:
                continue
            rec = rec_by_key.get((row.get("source_id", ""), row.get("row_index", "")))
            x, y, major, minor, theta = ellipse_params(row, rec)
            if not all(math.isfinite(v) for v in (x, y, major, minor, theta)):
                continue
            color = source_color(row)
            text = region_text(row)
            handle.write(f"ellipse({x:.3f},{y:.3f},{major:.3f},{minor:.3f},{theta:.3f}) "
                         f"# color={color} width=2 text={{{text}}}\n")
            handle.write(f"point({x:.3f},{y:.3f}) # point=cross color={color} width=2\n")
            kept += 1
    return kept


def main() -> int:
    args = parse_args()
    rows = read_rows(args.diagnostics_csv)
    out_dir = args.out_dir or args.diagnostics_csv.parent
    prefix = f"{args.tract}_{args.patch.replace(',', '_')}"
    total = 0
    for band in sorted({row["band"] for row in rows}):
        band_rows = [row for row in rows if row["band"] == band]
        rec_by_key = load_reclassification(args.reclassification_root, args.tract, args.patch, band)
        out_path = out_dir / band / f"{prefix}_{band}_dropped_large_component_area_lt_{int(args.max_area)}_reasons.reg"
        kept = write_band_reg(rows=band_rows, rec_by_key=rec_by_key, path=out_path)
        total += kept
        print(f"{band}: wrote {kept} region(s) to {out_path}")
    print(f"total regions written: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
