#!/usr/bin/env python3
"""Create a dated index of SAM/CELLECT experiment outputs."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


UTC8 = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Entry:
    section: str
    title: str
    source: Path
    note: str


def utc8_date_suffix() -> str:
    return datetime.now(tz=UTC8).strftime("%y%m%d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=utc8_date_suffix(), help="YYMMDD suffix in UTC+8.")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Default: output/sam_cellect_combination_<date>",
    )
    parser.add_argument(
        "--mode",
        choices=("link", "copy", "move"),
        default="link",
        help="Organize entries by symlink, copy, or direct move.",
    )
    return parser.parse_args()


def entries(date: str) -> list[Entry]:
    return [
        Entry(
            "amg_masks",
            f"sam_amg_coadd_irg_lower_right_masks_{date}",
            Path("/home/czh23/SAM-astro/output/test_zangetsu"),
            "SAM automatic mask generator on coadd lower-right Zangetsu cutout, RGB=I,R,G.",
        ),
        Entry(
            "amg_masks",
            f"sam_amg_denoised_irg_lower_right_masks_{date}",
            Path("/home/czh23/SAM-astro/output/test_zangetsu_denoised"),
            "SAM automatic mask generator on denoised lower-right Zangetsu cutout, RGB=I,R,G.",
        ),
        Entry(
            "amg_masks",
            f"sam_amg_noisy_irg_lower_right_masks_{date}",
            Path("/home/czh23/SAM-astro/output/test_zangetsu_noisy"),
            "SAM automatic mask generator on noisy lower-right Zangetsu cutout, RGB=I,R,G.",
        ),
        Entry(
            "amg_masks",
            f"sam_amg_coadd_gri_reference_masks_{date}",
            Path("/home/czh23/SAM-astro/output/zangetsu/zangestu_coadd"),
            "Older SAM automatic mask generator coadd run, RGB=G,R,I.",
        ),
        Entry(
            "amg_masks",
            f"sam_amg_denoised_gri_reference_masks_{date}",
            Path("/home/czh23/SAM-astro/output/zangetsu_denoised"),
            "Older SAM automatic mask generator denoised run, RGB=G,R,I.",
        ),
        Entry(
            "amg_masks",
            f"sam_amg_noisy_gri_reference_masks_{date}",
            Path("/home/czh23/SAM-astro/output/zangetsu_noisy"),
            "Older SAM automatic mask generator noisy run, RGB=G,R,I.",
        ),
        Entry(
            "scarlet_products",
            f"sam_scarlet_irg_zangetsu_products_{date}",
            Path("zangetsu_demo/sam_irg"),
            "SAM detection plus LSST scarlet deblend products for Zangetsu cutouts.",
        ),
        Entry(
            "scarlet_products",
            f"sam0_current_gt_regions_{date}",
            Path("zangetsu_demo/sam0_current_gt_regs"),
            "Existing sam0 denoised run converted to REG using current preprocessed GT.",
        ),
        Entry(
            "cellect_detector_diagnostics",
            f"cellect_per_band_filtered_sam0_current_gt_diagnostics_{date}",
            Path("zangetsu_demo/cellect_sam0_eval/per_band_data_filtered_0609"),
            "CELLECT per-band filtered checkpoint diagnostics on the sam0 cutout against current GT.",
        ),
        Entry(
            "cellect_detector_diagnostics",
            f"cellect_i_band_noisy_denoised_coadd_comparison_{date}",
            Path("zangetsu_demo/eval_per_band_data_filtered_0609_i"),
            "I-band CELLECT comparison across coadd, noisy, and denoised Zangetsu cutouts.",
        ),
        Entry(
            "cellect_detector_diagnostics",
            f"cellect_i_band_shape_region_comparison_{date}",
            Path("zangetsu_demo/eval_i_band_shape_regs"),
            "I-band predicted shape region diagnostics for multiple CELLECT checkpoints.",
        ),
        Entry(
            "cellect_detector_diagnostics",
            f"cellect_per_band_b5_zangetsu_detections_{date}",
            Path("zangetsu_demo/eval_per_band_b5"),
            "Five-band per-band checkpoint detector outputs on Zangetsu cutouts.",
        ),
        Entry(
            "cellect_detector_diagnostics",
            f"cellect_old_checkpoint_zangetsu_detections_{date}",
            Path("zangetsu_demo/eval_old_ckpt"),
            "Older five-band checkpoint detector outputs on Zangetsu cutouts.",
        ),
        Entry(
            "cellect_detector_diagnostics",
            f"cellect_detection_regions_by_band_zangetsu_{date}",
            Path("zangetsu_demo/reg_diagnostics_by_band"),
            "Per-band DS9 regions for clean GT, clean TP, FN, and clean/background FP.",
        ),
        Entry(
            "preprocessing_diagnostics",
            f"zangetsu_preprocessed_cutouts_{date}",
            Path("zangetsu_demo/preprocessed"),
            "Prepared coadd, noisy, and denoised Zangetsu cutouts and catalogs.",
        ),
        Entry(
            "preprocessing_diagnostics",
            f"zangetsu_pu_partition_panels_{date}",
            Path("zangetsu_demo/pu_partition_overlays"),
            "PU partition panels using the available background setup.",
        ),
        Entry(
            "preprocessing_diagnostics",
            f"zangetsu_pu_partition_panels_official_background_{date}",
            Path("zangetsu_demo/pu_partition_overlays_official_background"),
            "PU partition panels using official-background-derived masks.",
        ),
        Entry(
            "preprocessing_diagnostics",
            f"zangetsu_clean_mask_iou_checks_{date}",
            Path("zangetsu_demo/clean_mask_iou"),
            "Clean-mask IoU checks between current preprocessing and reference products.",
        ),
        Entry(
            "preprocessing_diagnostics",
            f"zangetsu_cmodel_ellipse_overlays_{date}",
            Path("zangetsu_demo/cmodel_ellipse_overlays"),
            "CModel ellipse overlay diagnostics on Zangetsu cutouts.",
        ),
    ]


def script_entries(date: str) -> list[Entry]:
    return [
        Entry(
            "scripts_used",
            f"prepare_zangetsu_cutouts_{date}.py",
            Path("zangetsu_demo/prepare_zangetsu_demo.py"),
            "Prepare Zangetsu coadd/noisy/denoised cutouts for CELLECT diagnostics.",
        ),
        Entry(
            "scripts_used",
            f"run_sam_scarlet_irg_zangetsu_{date}.py",
            Path("zangetsu_demo/run_sam_irg_zangetsu.py"),
            "Run SAM detection with RGB=I,R,G and LSST scarlet deblend.",
        ),
        Entry(
            "scripts_used",
            f"convert_sam_labelmap_to_ds9_regions_{date}.py",
            Path("zangetsu_demo/convert_sam_labelmap_to_reg.py"),
            "Convert SAM AMG labelmap FITS to DS9 region files.",
        ),
        Entry(
            "scripts_used",
            f"export_sam0_regions_against_current_gt_{date}.py",
            Path("zangetsu_demo/export_existing_sam_regs.py"),
            "Export existing SAM/LSST run regions against current preprocessed GT.",
        ),
        Entry(
            "scripts_used",
            f"export_cellect_regions_from_source_csv_{date}.py",
            Path("zangetsu_demo/export_cellect_eval_regs.py"),
            "Export CELLECT source CSV diagnostics into DS9 regions.",
        ),
        Entry(
            "scripts_used",
            f"make_zangetsu_detection_regions_by_band_{date}.py",
            Path("zangetsu_demo/make_zangetsu_reg_diagnostics.py"),
            "Generate by-band clean/TP/FN/FP regions for Zangetsu detector diagnostics.",
        ),
        Entry(
            "scripts_used",
            f"visualize_zangetsu_pu_partitions_{date}.py",
            Path("zangetsu_demo/visualize_zangetsu_pu_partition_panel.py"),
            "Draw PU partition panels for Zangetsu cutouts.",
        ),
        Entry(
            "scripts_used",
            f"compare_zangetsu_clean_mask_iou_{date}.py",
            Path("zangetsu_demo/compare_zangetsu_clean_mask_iou.py"),
            "Compare clean-mask overlap between preprocessing/reference products.",
        ),
    ]


def _resolved(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _clear_existing_link_or_empty_target(target: Path) -> str | None:
    if target.is_symlink():
        target.unlink()
        return None
    if not target.exists():
        return None
    return "occupied"


def organize_entry(source: Path, target: Path, mode: str) -> str:
    source = source.expanduser()
    if not source.is_absolute():
        source = (Path.cwd() / source).resolve()
    else:
        source = source.resolve()
    if not source.exists():
        return "missing"
    target.parent.mkdir(parents=True, exist_ok=True)

    if mode == "link":
        if target.exists() or target.is_symlink():
            if target.is_symlink() and Path(os.readlink(target)) == source:
                return "exists"
            return "occupied"
        target.symlink_to(source, target_is_directory=source.is_dir())
        return "linked"

    occupied = _clear_existing_link_or_empty_target(target)
    if occupied is not None:
        return "occupied"

    if mode == "copy":
        if source.is_dir():
            shutil.copytree(source, target, symlinks=True)
        else:
            shutil.copy2(source, target)
        return "copied"

    if mode == "move":
        shutil.move(str(source), str(target))
        return "moved"

    raise ValueError(f"unsupported mode: {mode}")


def write_readme(out_root: Path, rows: list[dict[str, str]], date: str) -> None:
    lines = [
        f"# SAM and CELLECT Combination Outputs {date}",
        "",
        "Created with UTC+8 date suffix.",
        "",
        "Naming convention: `<method>_<data>_<content>_<YYMMDD>`.",
        "",
        "Ambiguous labels such as `eval` and `train` are avoided in the organized titles. Original paths are kept in `MANIFEST.csv`.",
        "",
        "## Sections",
        "",
        f"- `amg_masks_{date}`: SAM automatic mask generator labelmaps, overlays, metadata, and DS9 regions.",
        f"- `scarlet_products_{date}`: SAM detection plus LSST scarlet products and current-GT region diagnostics.",
        f"- `cellect_detector_diagnostics_{date}`: CELLECT detector outputs, source CSVs, metrics, and regions.",
        f"- `preprocessing_diagnostics_{date}`: Zangetsu preprocessing panels, cutouts, and mask quality checks.",
        f"- `scripts_used_{date}`: symlinked scripts with clearer dated names.",
        "",
        "## Manifest Preview",
        "",
        "| Section | Title | Status | Original Path |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['section']}` | `{row['title']}` | `{row['status']}` | `{row['source']}` |"
        )
    lines.append("")
    out_root.joinpath("README.md").write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    date = str(args.date)
    out_root = args.out_root or Path("output") / f"sam_cellect_combination_{date}"
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    all_entries = entries(date) + script_entries(date)
    for entry in all_entries:
        section_dir = out_root / f"{entry.section}_{date}"
        link = section_dir / entry.title
        status = organize_entry(entry.source, link, args.mode)
        rows.append(
            {
                "section": entry.section,
                "title": entry.title,
                "link": str(link),
                "source": str(entry.source),
                "status": status,
                "note": entry.note,
            }
        )

    with out_root.joinpath("MANIFEST.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "title", "link", "source", "status", "note"])
        writer.writeheader()
        writer.writerows(rows)
    write_readme(out_root, rows, date)
    print(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
