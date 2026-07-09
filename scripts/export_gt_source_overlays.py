#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Tuple

import matplotlib
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.visualization import ZScaleInterval

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, Patch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from astro_data_preprocessing import (  # noqa: E402
    TileSpec,
    _classify_clean_by_noncoadd_snr,
    _find_image_hdu_index,
    _quality_mask_from_lsst_fits,
    _source_annulus_exclusion_mask,
    _vstack_nonempty,
)


CLASS_COLORS = {
    "clean": "green",
    "center_only": "yellow",
    "ignore": "magenta",
}


def _parse_xy(stem: str) -> Tuple[int, int]:
    import re

    match = re.search(r"_x(-?\d+)_y(-?\d+)$", stem)
    if match is None:
        raise ValueError(f"Cannot parse x/y from target name: {stem}")
    return int(match.group(1)), int(match.group(2))


def _read_image(path: Path) -> Tuple[np.ndarray, Tuple[int, int]]:
    with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
        hdu = hdul[_find_image_hdu_index(hdul)]
        image = np.asarray(hdu.data, dtype=np.float32)
        header = hdu.header
        if "CRVAL1A" in header and "CRVAL2A" in header:
            origin = (int(round(float(header["CRVAL1A"]))), int(round(float(header["CRVAL2A"]))))
        else:
            origin = (
                int(round(-float(header.get("LTV1", 0.0)))),
                int(round(-float(header.get("LTV2", 0.0)))),
            )
    return image, origin


def _read_table(path: Path) -> Table:
    if not path.exists():
        return Table()
    return Table.read(path, hdu=1)


def _source_id_column(table: Table) -> str:
    if "id" in table.colnames:
        return "id"
    if "source_id" in table.colnames:
        return "source_id"
    raise KeyError("source table must contain id or source_id")


def _copy_rows(table: Table, mask: np.ndarray) -> Table:
    if len(table) == 0:
        return table.copy(copy_data=True)
    return table[np.asarray(mask, dtype=bool)].copy(copy_data=True)


def _select_ids(table: Table, ids: set[int]) -> Table:
    if len(table) == 0 or not ids:
        return table[:0].copy(copy_data=True)
    source_col = _source_id_column(table)
    values = np.asarray(table[source_col], dtype=np.int64)
    return table[np.asarray([int(value) in ids for value in values], dtype=bool)].copy(copy_data=True)


def _exclude_ids(table: Table, ids: set[int]) -> Table:
    if len(table) == 0 or not ids:
        return table.copy(copy_data=True)
    source_col = _source_id_column(table)
    values = np.asarray(table[source_col], dtype=np.int64)
    return table[np.asarray([int(value) not in ids for value in values], dtype=bool)].copy(copy_data=True)


def _ids_from_table(table: Table) -> set[int]:
    if len(table) == 0:
        return set()
    source_col = _source_id_column(table)
    return set(int(value) for value in np.asarray(table[source_col], dtype=np.int64))


def _load_remeasured_overrides(path: Path) -> Dict[int, dict[str, object]]:
    if not path.exists():
        return {}
    rows: Dict[int, dict[str, object]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                source_id = int(row["source_id"])
            except Exception:
                continue
            rows[source_id] = row
    return rows


def _float_value(row: dict[str, object], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def _apply_remeasured_classes(table: Table, *, strict_center: Table, remeasured_csv: Path) -> Table:
    out = table.copy(copy_data=True)
    source_col = _source_id_column(out)
    class_values = np.asarray(["ignore"] * len(out), dtype=object)
    if "pu_class" in out.colnames:
        class_values[np.asarray(out["pu_class"], dtype=str) == "clean"] = "clean"

    if len(strict_center):
        strict_col = _source_id_column(strict_center)
        strict_ids = set(int(value) for value in np.asarray(strict_center[strict_col], dtype=np.int64))
        source_ids = np.asarray(out[source_col], dtype=np.int64)
        class_values[np.asarray([int(value) in strict_ids for value in source_ids], dtype=bool)] = "center_only"

    overrides = _load_remeasured_overrides(remeasured_csv)
    if overrides:
        source_ids = np.asarray(out[source_col], dtype=np.int64)
        for idx, source_id in enumerate(source_ids):
            row = overrides.get(int(source_id))
            if row is None:
                continue
            area = _float_value(row, "used_aperture_area")
            if np.isfinite(area) and area > 10000.0:
                class_values[idx] = "drop"
                continue
            final_class = str(row.get("final_training_class", "")).strip()
            if final_class in {"clean", "center_only", "ignore"}:
                class_values[idx] = final_class
            major = _float_value(row, "aperture_major")
            minor = _float_value(row, "aperture_minor")
            theta = _float_value(row, "theta")
            if np.isfinite(major) and np.isfinite(minor) and major > 0.0 and minor > 0.0:
                out["ellipse_major_sigma"][idx] = major
                out["ellipse_minor_sigma"][idx] = minor
                out["ellipse_theta"][idx] = theta if np.isfinite(theta) else 0.0
    out["gt_training_class"] = np.asarray(class_values, dtype=str)
    return out


def _split_by_gt_class(table: Table) -> Tuple[Table, Table, Table]:
    labels = np.asarray(table["gt_training_class"], dtype=str) if "gt_training_class" in table.colnames else np.asarray([], dtype=str)
    return (
        _copy_rows(table, labels == "clean"),
        _copy_rows(table, labels == "center_only"),
        _copy_rows(table, labels == "ignore"),
    )


def _variant_clean_ids_from_metadata(meta_dir: Path, *, prefix: str) -> set[int]:
    ids: set[int] = set()
    for path in sorted(meta_dir.glob("*.npz")):
        if not path.stem.startswith(prefix):
            continue
        data = np.load(path)
        if "ids" not in data:
            continue
        ids.update(int(value) for value in np.asarray(data["ids"], dtype=np.int64).ravel())
    return ids


def _variant_visibility_centers(target_dir: Path, key: str, *, prefix: str) -> np.ndarray:
    centers: list[Tuple[float, float]] = []
    for path in sorted(target_dir.glob("*.npz")):
        if not path.stem.startswith(prefix):
            continue
        data = np.load(path)
        if key not in data:
            continue
        x0, y0 = _parse_xy(path.stem)
        local = np.asarray(data[key], dtype=np.float64).reshape(-1, 2)
        for x, y in local:
            centers.append((float(x + x0), float(y + y0)))
    if not centers:
        return np.zeros((0, 2), dtype=np.float64)
    rounded = sorted(set((round(x, 3), round(y, 3)) for x, y in centers))
    return np.asarray(rounded, dtype=np.float64)


def _match_centers_to_table(table: Table, centers: np.ndarray, *, tolerance: float = 0.05) -> Table:
    if len(table) == 0 or centers.size == 0:
        return table[:0].copy(copy_data=True)
    x = np.asarray(table["base_SdssCentroid_x"], dtype=np.float64)
    y = np.asarray(table["base_SdssCentroid_y"], dtype=np.float64)
    matched_rows: set[int] = set()
    tol2 = float(tolerance) ** 2
    for center in np.asarray(centers, dtype=np.float64).reshape(-1, 2):
        dx = x - center[0]
        dy = y - center[1]
        dist2 = dx * dx + dy * dy
        idx = int(np.argmin(dist2))
        if float(dist2[idx]) <= tol2:
            matched_rows.add(idx)
    if not matched_rows:
        return table[:0].copy(copy_data=True)
    return table[sorted(matched_rows)].copy(copy_data=True)


def _variant_classes_from_existing_targets(
    *,
    source_table: Table,
    coadd_center: Table,
    coadd_ignore: Table,
    variant_patch_root: Path,
    band: str,
    group: str,
) -> Tuple[Table, Table, Table]:
    prefix = f"{group}_"
    clean_ids = _variant_clean_ids_from_metadata(variant_patch_root / "band_tile_metadata" / band, prefix=prefix)
    clean = _select_ids(source_table, clean_ids)
    used = _ids_from_table(clean)

    center_centers = _variant_visibility_centers(
        variant_patch_root / "band_targets" / band,
        "visibility_center_only_centers",
        prefix=prefix,
    )
    ignore_centers = _variant_visibility_centers(
        variant_patch_root / "band_targets" / band,
        "visibility_ignore_centers",
        prefix=prefix,
    )
    snr_center = _exclude_ids(_match_centers_to_table(source_table, center_centers), used)
    center = _vstack_nonempty([_exclude_ids(coadd_center, used), snr_center])
    used |= _ids_from_table(center)

    snr_ignore = _exclude_ids(_match_centers_to_table(source_table, ignore_centers), used)
    ignore = _vstack_nonempty([_exclude_ids(coadd_ignore, used), snr_ignore])
    used |= _ids_from_table(ignore)

    clean["gt_training_class"] = np.asarray(["clean"] * len(clean), dtype=str)
    center["gt_training_class"] = np.asarray(["center_only"] * len(center), dtype=str)
    ignore["gt_training_class"] = np.asarray(["ignore"] * len(ignore), dtype=str)
    return clean, center, ignore


def _snr_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        x_col="base_SdssCentroid_x",
        y_col="base_SdssCentroid_y",
        noncoadd_snr_ap_radius=float(args.noncoadd_snr_ap_radius),
        noncoadd_snr_annulus_r_in=float(args.noncoadd_snr_annulus_r_in),
        noncoadd_snr_annulus_r_out=float(args.noncoadd_snr_annulus_r_out),
        noncoadd_snr_ignore_thresh=float(args.noncoadd_snr_ignore_thresh),
        noncoadd_snr_center_only_thresh=float(args.noncoadd_snr_center_only_thresh),
        noncoadd_snr_source_mask_ellipse_sigma=float(args.noncoadd_snr_source_mask_ellipse_sigma),
        noncoadd_snr_min_annulus_pixels=int(args.noncoadd_snr_min_annulus_pixels),
    )


def _classify_variant_sources(
    *,
    clean: Table,
    center: Table,
    ignore: Table,
    strict_center: Table,
    image: np.ndarray,
    image_origin: Tuple[int, int],
    quality_mask: np.ndarray | None,
    args: argparse.Namespace,
) -> Tuple[Table, Table, Table]:
    if len(clean) == 0:
        return clean, center, ignore
    spec = TileSpec(name="full_patch", x0=image_origin[0], y0=image_origin[1], size=int(image.shape[0]))
    exclusion_sources = _vstack_nonempty([clean, center, strict_center])
    annulus_exclude = _source_annulus_exclusion_mask(
        exclusion_sources,
        spec,
        x_col="base_SdssCentroid_x",
        y_col="base_SdssCentroid_y",
        ellipse_sigma=float(args.noncoadd_snr_source_mask_ellipse_sigma),
    )
    if quality_mask is not None and quality_mask.shape != image.shape:
        quality_mask = None
    normal, snr_center, snr_ignore, _snr = _classify_clean_by_noncoadd_snr(
        clean,
        image=image,
        image_origin=image_origin,
        args=_snr_args(args),
        annulus_exclude_mask=annulus_exclude,
        annulus_hard_exclude_mask=quality_mask,
    )
    normal["gt_training_class"] = np.asarray(["clean"] * len(normal), dtype=str)
    snr_center["gt_training_class"] = np.asarray(["center_only"] * len(snr_center), dtype=str)
    snr_ignore["gt_training_class"] = np.asarray(["ignore"] * len(snr_ignore), dtype=str)
    center = _vstack_nonempty([center, snr_center])
    ignore = _vstack_nonempty([ignore, snr_ignore])
    return normal, center, ignore


def _local_xy(row, origin: Tuple[int, int]) -> Tuple[float, float]:
    return (
        float(row["base_SdssCentroid_x"]) - float(origin[0]),
        float(row["base_SdssCentroid_y"]) - float(origin[1]),
    )


def _iter_class_rows(classes: Dict[str, Table]) -> Iterable[Tuple[str, object]]:
    for class_name in ("clean", "center_only", "ignore"):
        for row in classes[class_name]:
            yield class_name, row


def _write_reg(path: Path, classes: Dict[str, Table], origin: Tuple[int, int]) -> None:
    lines = [
        "# Region file format: DS9 version 4.1",
        'global font="helvetica 9 normal roman" edit=1 move=1 delete=1 include=1',
        "image",
    ]
    for class_name, row in _iter_class_rows(classes):
        color = CLASS_COLORS[class_name]
        x, y = _local_xy(row, origin)
        x += 1.0
        y += 1.0
        sid = int(row[_source_id_column(row)]) if hasattr(row, "colnames") else int(row["id"])
        try:
            major = float(row["ellipse_major_sigma"])
            minor = float(row["ellipse_minor_sigma"])
            theta = math.degrees(float(row["ellipse_theta"]))
        except Exception:
            major = minor = theta = float("nan")
        if np.isfinite(major) and np.isfinite(minor) and major > 0.0 and minor > 0.0:
            lines.append(
                f"ellipse({x:.3f},{y:.3f},{major:.3f},{minor:.3f},{theta:.3f}) "
                f"# color={color} width=2 text={{{sid} {class_name}}}"
            )
        else:
            lines.append(f"point({x:.3f},{y:.3f}) # point=cross color={color} width=2 text={{{sid} {class_name}}}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _zscale_limits(image: np.ndarray) -> Tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    try:
        lo, hi = ZScaleInterval(contrast=0.25).get_limits(finite)
    except Exception:
        lo, hi = np.nanpercentile(finite, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanpercentile(finite, [1, 99])
    return float(lo), float(hi)


def _write_crop_overlay(
    path: Path,
    *,
    image: np.ndarray,
    image_origin: Tuple[int, int],
    classes: Dict[str, Table],
    crop_x0: int,
    crop_y0: int,
    crop_size: int,
    title: str,
) -> None:
    local_x0 = int(crop_x0 - image_origin[0])
    local_y0 = int(crop_y0 - image_origin[1])
    crop = image[local_y0 : local_y0 + crop_size, local_x0 : local_x0 + crop_size]
    lo, hi = _zscale_limits(crop)
    fig, ax = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
    ax.imshow(crop, origin="lower", cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    counts = {key: 0 for key in ("clean", "center_only", "ignore")}
    for class_name, row in _iter_class_rows(classes):
        x = float(row["base_SdssCentroid_x"]) - float(crop_x0)
        y = float(row["base_SdssCentroid_y"]) - float(crop_y0)
        try:
            major = float(row["ellipse_major_sigma"])
            minor = float(row["ellipse_minor_sigma"])
            theta = math.degrees(float(row["ellipse_theta"]))
        except Exception:
            major = minor = theta = float("nan")
        if x < -max(major, 10.0) or x > crop_size + max(major, 10.0):
            continue
        if y < -max(major, 10.0) or y > crop_size + max(major, 10.0):
            continue
        counts[class_name] += 1
        color = CLASS_COLORS[class_name]
        if np.isfinite(major) and np.isfinite(minor) and major > 0.0 and minor > 0.0:
            ax.add_patch(
                Ellipse(
                    (x, y),
                    width=2.0 * major,
                    height=2.0 * minor,
                    angle=theta,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.1 if class_name != "ignore" else 0.9,
                    alpha=0.9 if class_name != "ignore" else 0.65,
                )
            )
        else:
            ax.plot(x, y, "+", color=color, markersize=5)
    ax.set_xlim(-0.5, crop_size - 0.5)
    ax.set_ylim(-0.5, crop_size - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{title}\nclean={counts['clean']} center={counts['center_only']} ignore={counts['ignore']}", fontsize=10)
    ax.legend(
        handles=[Patch(facecolor=CLASS_COLORS[key], label=key) for key in ("clean", "center_only", "ignore")],
        loc="lower left",
        fontsize=8,
        framealpha=0.8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _summary_row(dataset: str, classes: Dict[str, Table], reg: Path, overlay: Path) -> Dict[str, object]:
    return {
        "dataset": dataset,
        "clean": len(classes["clean"]),
        "center_only": len(classes["center_only"]),
        "ignore": len(classes["ignore"]),
        "reg": str(reg),
        "overlay": str(overlay),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export source-level GT REG and crop overlays.")
    parser.add_argument("--preprocessed-root", type=Path, default=Path("/nvme0/zc/scarlet/preprocessed"))
    parser.add_argument("--coadd-root", type=Path, default=Path("/nvme0/zc/scarlet"))
    parser.add_argument("--denoised-fits-root", type=Path, default=Path("/nvme0/zc/scarlet/denoised_fits"))
    parser.add_argument("--remeasured-csv", type=Path, default=Path("diagnostics/output/refit_kron_remeasured_photometry_4_5/9813_4_5_HSC-I_remeasured_refit_kron.csv"))
    parser.add_argument("--tract", type=int, default=9813)
    parser.add_argument("--patch", default="4,5")
    parser.add_argument("--band", default="HSC-I")
    parser.add_argument("--variant", default="denoised")
    parser.add_argument("--group", default="group_01")
    parser.add_argument("--crop-x0", type=int, default=18204)
    parser.add_argument("--crop-y0", type=int, default=20924)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--out-dir", type=Path, default=Path("output/gt_source_overlays_260708"))
    parser.add_argument(
        "--variant-source-mode",
        choices=("targets", "snr-full-image"),
        default="targets",
        help="Use existing variant targets for exact training counts, or recompute SNR on the full image.",
    )
    parser.add_argument("--noncoadd-snr-ignore-thresh", type=float, default=2.0)
    parser.add_argument("--noncoadd-snr-center-only-thresh", type=float, default=3.0)
    parser.add_argument("--noncoadd-snr-ap-radius", type=float, default=6.0)
    parser.add_argument("--noncoadd-snr-annulus-r-in", type=float, default=10.0)
    parser.add_argument("--noncoadd-snr-annulus-r-out", type=float, default=15.0)
    parser.add_argument("--noncoadd-snr-source-mask-ellipse-sigma", type=float, default=1.0)
    parser.add_argument("--noncoadd-snr-min-annulus-pixels", type=int, default=50)
    parser.add_argument(
        "--noncoadd-snr-mask-planes",
        nargs="*",
        default=["BRIGHT_OBJECT", "SAT", "BAD", "NO_DATA", "EDGE", "UNMASKEDNAN"],
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    band_filename = f"meas-{args.band}-{args.tract}-{args.patch}.fits"
    coadd_patch_root = args.preprocessed_root / str(args.tract) / args.patch
    pu_all = _read_table(coadd_patch_root / "band_reference_pu_all" / args.band / band_filename)
    strict_center = _read_table(coadd_patch_root / "band_reference_strict_center_only" / args.band / band_filename)
    coadd_table = _apply_remeasured_classes(pu_all, strict_center=strict_center, remeasured_csv=args.remeasured_csv)
    coadd_clean, coadd_center, coadd_ignore = _split_by_gt_class(coadd_table)

    coadd_image_path = args.coadd_root / str(args.tract) / args.band / args.patch / f"calexp-{args.band}-{args.tract}-{args.patch}.fits"
    coadd_image, image_origin = _read_image(coadd_image_path)

    variant_image_path = (
        args.denoised_fits_root
        / f"patch_{args.patch.replace(',', '_')}"
        / args.group
        / args.band
        / f"{args.variant}.fits"
    )
    variant_image, variant_origin = _read_image(variant_image_path)
    quality_mask = _quality_mask_from_lsst_fits(coadd_image_path, tuple(args.noncoadd_snr_mask_planes))
    variant_patch_root = args.preprocessed_root / args.variant / str(args.tract) / args.patch
    if args.variant_source_mode == "targets":
        variant_clean, variant_center, variant_ignore = _variant_classes_from_existing_targets(
            source_table=coadd_table,
            coadd_center=coadd_center,
            coadd_ignore=coadd_ignore,
            variant_patch_root=variant_patch_root,
            band=args.band,
            group=args.group,
        )
    else:
        variant_clean, variant_center, variant_ignore = _classify_variant_sources(
            clean=coadd_clean,
            center=coadd_center,
            ignore=coadd_ignore,
            strict_center=strict_center,
            image=variant_image,
            image_origin=variant_origin,
            quality_mask=quality_mask,
            args=args,
        )

    outputs = []
    for dataset, image, origin, classes, image_path in (
        (
            "coadd",
            coadd_image,
            image_origin,
            {"clean": coadd_clean, "center_only": coadd_center, "ignore": coadd_ignore},
            coadd_image_path,
        ),
        (
            f"{args.variant}_{args.group}",
            variant_image,
            variant_origin,
            {"clean": variant_clean, "center_only": variant_center, "ignore": variant_ignore},
            variant_image_path,
        ),
    ):
        stem = f"{args.band}_{args.tract}_{args.patch.replace(',', '_')}_{dataset}"
        reg = args.out_dir / f"{stem}_full_gt_sources.reg"
        overlay = args.out_dir / f"{stem}_x{args.crop_x0}_y{args.crop_y0}_gt_sources_overlay.png"
        _write_reg(reg, classes, origin)
        _write_crop_overlay(
            overlay,
            image=image,
            image_origin=origin,
            classes=classes,
            crop_x0=args.crop_x0,
            crop_y0=args.crop_y0,
            crop_size=args.crop_size,
            title=f"{dataset} {args.band} {args.tract}/{args.patch}",
        )
        row = _summary_row(dataset, classes, reg, overlay)
        row["image"] = str(image_path)
        outputs.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    with (args.out_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(outputs[0].keys()))
        writer.writeheader()
        writer.writerows(outputs)
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
