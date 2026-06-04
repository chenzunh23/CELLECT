#!/usr/bin/env python3
"""Prepare two 512x512 Zangetsu demo cutouts with cropped catalogs/targets."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack

import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from astro_data_preprocessing import (  # noqa: E402
    TileSpec,
    _band_det_path,
    _crop_full_mask_for_tile,
    _read_det_background_mask,
    _run_lsst_detection_background,
    crop_catalog_for_tile,
    make_pu_dense_targets,
    write_table_pair,
    write_targets,
)


BANDS = ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"]
TRACT = "9813"
PATCH = "6,1"
TILE_SIZE = 512
CONFIDENCE_LEVELS = 5
CORE_RADIUS = 2
ELLIPSE_SIGMA = 1.0
CENTER_ONLY_WEIGHT = 0.25
X_COL = "base_SdssShape_x"
Y_COL = "base_SdssShape_y"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zangetsu-root", default="/home/czh23/Zangetsu_4.30arcmin")
    parser.add_argument("--coadd-root", default="/data1/czh23/Subaru/9813")
    parser.add_argument("--source-preprocessed-root", default="/nvme0/zc/scarlet/preprocessed/9813/6,1")
    parser.add_argument("--out-root", default="zangetsu_demo/preprocessed")
    parser.add_argument("--datasets", nargs="+", choices=("coadd", "noisy", "denoised"), default=["coadd", "noisy", "denoised"])
    parser.add_argument(
        "--lsst-background-policy",
        choices=("run-if-missing", "existing", "none"),
        default="run-if-missing",
        help="Use official det footprints when available; otherwise run LSST default detection for the cutout.",
    )
    parser.add_argument("--lsst-background-cache-root", type=Path, default=None)
    parser.add_argument("--lsst-detect-python", default="")
    parser.add_argument("--overwrite-lsst-background", action="store_true")
    parser.add_argument("--write-lsst-background-products", action="store_true")
    parser.add_argument(
        "--use-lsst-detection-calexp-cutouts",
        action="store_true",
        help="For noisy/denoised tiles, use LSST detection outputExposure as the cutout FITS.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def image_origin_from_header(header: fits.Header) -> tuple[int, int]:
    return -int(round(float(header.get("LTV1", 0.0)))), -int(round(float(header.get("LTV2", 0.0))))


def crop_header(header: fits.Header, *, x_local: int, y_local: int) -> fits.Header:
    out = header.copy()
    if "CRPIX1" in out:
        out["CRPIX1"] = float(out["CRPIX1"]) - float(x_local)
    if "CRPIX2" in out:
        out["CRPIX2"] = float(out["CRPIX2"]) - float(y_local)
    if "LTV1" in out:
        out["LTV1"] = float(out["LTV1"]) - float(x_local)
    if "LTV2" in out:
        out["LTV2"] = float(out["LTV2"]) - float(y_local)
    out["EXTNAME"] = "IMAGE"
    return out


def write_image_cutout(src: Path, dst: Path, *, x_local: int, y_local: int) -> None:
    with fits.open(src, memmap=False) as hdul:
        hdu = next(h for h in hdul if h.data is not None and getattr(h.data, "ndim", 0) >= 2)
        data = np.asarray(hdu.data, dtype=np.float32)
        crop = data[y_local : y_local + TILE_SIZE, x_local : x_local + TILE_SIZE]
        if crop.shape != (TILE_SIZE, TILE_SIZE):
            raise ValueError(f"bad crop shape {crop.shape} from {src}")
        header = crop_header(hdu.header, x_local=x_local, y_local=y_local)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(crop, header=header, name="IMAGE")]).writeto(dst, overwrite=True)


def write_global_image_cutout(src: Path, dst: Path, *, spec: TileSpec) -> None:
    with fits.open(src, memmap=False) as hdul:
        hdu = next(h for h in hdul if h.data is not None and getattr(h.data, "ndim", 0) >= 2)
        origin_x, origin_y = image_origin_from_header(hdu.header)
        x_local = int(spec.x0 - origin_x)
        y_local = int(spec.y0 - origin_y)
    write_image_cutout(src, dst, x_local=x_local, y_local=y_local)


def read_table(path: Path) -> Table:
    return Table.read(path)


def crop_many(table: Table, specs: Iterable[TileSpec], *, margin: float = 64.0) -> Table:
    if len(table) == 0:
        return table
    parts = [crop_catalog_for_tile(table, spec, x_col=X_COL, y_col=Y_COL, margin=margin) for spec in specs]
    if not parts:
        return Table()
    if len(parts) == 1:
        return parts[0]
    return vstack(parts, metadata_conflicts="silent")


def crop_tile_or_empty(table: Table, spec: TileSpec, *, margin: float = 64.0) -> Table:
    if len(table) == 0:
        return table
    return crop_catalog_for_tile(table, spec, x_col=X_COL, y_col=Y_COL, margin=margin)


def source_table(root: Path, kind: str, band: str) -> Table:
    path = root / kind / band / f"meas-{band}-{TRACT}-{PATCH}.fits"
    if not path.exists():
        return Table()
    return read_table(path)


def read_image_shape_and_origin(path: Path) -> tuple[tuple[int, int], tuple[int, int]]:
    with fits.open(path, memmap=False) as hdul:
        hdu = next(h for h in hdul if h.data is not None and getattr(h.data, "ndim", 0) >= 2)
        return tuple(int(v) for v in hdu.data.shape[-2:]), image_origin_from_header(hdu.header)


def official_lsst_background_for_tile(coadd_root: Path, band: str, spec: TileSpec) -> np.ndarray | None:
    det_path = _band_det_path(coadd_root, band, int(TRACT), PATCH)
    if det_path is None:
        return None
    image_path = coadd_root / band / PATCH / f"calexp-{band}-{TRACT}-{PATCH}.fits"
    shape_yx, parent_origin = read_image_shape_and_origin(image_path)
    full_background = _read_det_background_mask(det_path, shape_yx, origin_xy=parent_origin)
    return _crop_full_mask_for_tile(full_background, spec, parent_origin)


def generated_lsst_background_for_cutout(
    *,
    args: argparse.Namespace,
    out_patch: Path,
    cutout_path: Path,
    band: str,
    spec: TileSpec,
) -> np.ndarray | None:
    if str(args.lsst_background_policy) in {"none", "existing"}:
        return None
    output_calexp_path = (
        cutout_path.with_name(f"lsst-detect-{cutout_path.name}")
        if bool(getattr(args, "use_lsst_detection_calexp_cutouts", False))
        else None
    )
    det_path = _run_lsst_detection_background(
        args=args,
        output_root=out_patch,
        coadd_path=cutout_path,
        band=band,
        tract=int(TRACT),
        patch=f"{PATCH}_{spec.name}",
        output_calexp_path=output_calexp_path,
    )
    if output_calexp_path is not None and output_calexp_path.exists():
        shutil.copy2(output_calexp_path, cutout_path)
    return _read_det_background_mask(det_path, (TILE_SIZE, TILE_SIZE), origin_xy=(spec.x0, spec.y0))


def zangetsu_specs(zangetsu_root: Path) -> list[tuple[TileSpec, int, int]]:
    with fits.open(zangetsu_root / "HSC-I" / "noisy.fits", memmap=False) as hdul:
        image_hdu = next(h for h in hdul if h.data is not None and getattr(h.data, "ndim", 0) >= 2)
        height, width = image_hdu.data.shape[-2:]
        origin_x, origin_y = image_origin_from_header(image_hdu.header)

    crop_defs = [
        ("zangetsu_upper_left", 0, height - TILE_SIZE),
        ("zangetsu_lower_right", width - TILE_SIZE, 0),
    ]
    specs: list[tuple[TileSpec, int, int]] = []
    for label, x_local, y_local in crop_defs:
        x0, y0 = origin_x + x_local, origin_y + y_local
        spec = TileSpec(name=f"{label}_x{x0}_y{y0}", x0=x0, y0=y0, size=TILE_SIZE, kind="zangetsu")
        specs.append((spec, x_local, y_local))
    return specs


def write_dataset(kind: str, image_kind: str, args: argparse.Namespace) -> None:
    zangetsu_root = Path(args.zangetsu_root).expanduser().resolve()
    coadd_root = Path(args.coadd_root).expanduser().resolve()
    source_root = Path(args.source_preprocessed_root).expanduser().resolve()
    out_patch = Path(args.out_root).resolve() / kind / TRACT / PATCH
    if out_patch.exists() and args.overwrite:
        shutil.rmtree(out_patch)
    out_patch.mkdir(parents=True, exist_ok=True)

    specs_with_local = zangetsu_specs(zangetsu_root)
    specs = [item[0] for item in specs_with_local]

    for spec, x_local, y_local in specs_with_local:
        for band in BANDS:
            if image_kind == "calexp":
                src = coadd_root / band / PATCH / f"calexp-{band}-{TRACT}-{PATCH}.fits"
                dst = out_patch / "cutouts" / spec.name / band / f"calexp-{band}-{TRACT}-{PATCH}.fits"
                write_global_image_cutout(src, dst, spec=spec)
            else:
                src = zangetsu_root / band / f"{image_kind}.fits"
                dst = out_patch / "cutouts" / spec.name / band / f"{image_kind}-{band}-{TRACT}-{PATCH}.fits"
                write_image_cutout(src, dst, x_local=x_local, y_local=y_local)

    # Patch-level band catalogs cropped to the two Zangetsu tiles.
    band_clean: dict[str, Table] = {}
    band_center: dict[str, Table] = {}
    band_ignore: dict[str, Table] = {}
    band_strict: dict[str, Table] = {}
    for band in BANDS:
        clean = crop_many(source_table(source_root, "band_reference_catalogs", band), specs)
        center = crop_many(source_table(source_root, "band_reference_center_only", band), specs)
        ignore = crop_many(source_table(source_root, "band_reference_ignore", band), specs)
        strict = crop_many(source_table(source_root, "band_reference_strict_ignore", band), specs)
        band_clean[band], band_center[band], band_ignore[band], band_strict[band] = clean, center, ignore, strict
        write_table_pair(clean, out_patch / "band_reference_catalogs" / band / f"meas-{band}-{TRACT}-{PATCH}.fits")
        write_table_pair(center, out_patch / "band_reference_center_only" / band / f"meas-{band}-{TRACT}-{PATCH}.fits")
        write_table_pair(ignore, out_patch / "band_reference_ignore" / band / f"meas-{band}-{TRACT}-{PATCH}.fits")
        write_table_pair(strict, out_patch / "band_reference_strict_ignore" / band / f"meas-{band}-{TRACT}-{PATCH}.fits")

    for spec in specs:
        # The fused eval path still needs one tile-level catalog; HSC-I is used
        # only for record discovery/merged diagnostics while per-band targets
        # drive the detection metrics.
        tile_ref = crop_tile_or_empty(band_clean["HSC-I"], spec, margin=64.0)
        tile_ign = crop_tile_or_empty(band_ignore["HSC-I"], spec, margin=64.0)
        tile_strict = crop_tile_or_empty(band_strict["HSC-I"], spec, margin=64.0)
        write_table_pair(tile_ref, out_patch / "reference_catalogs" / f"{spec.name}_meas.fits")
        write_table_pair(tile_ign, out_patch / "ignore_catalogs" / f"{spec.name}_meas.fits")
        write_table_pair(tile_strict, out_patch / "strict_ignore_catalogs" / f"{spec.name}_meas.fits")
        for band in BANDS:
            clean = crop_tile_or_empty(band_clean[band], spec, margin=64.0)
            center = crop_tile_or_empty(band_center[band], spec, margin=64.0)
            ignore = crop_tile_or_empty(band_ignore[band], spec, margin=64.0)
            strict = crop_tile_or_empty(band_strict[band], spec, margin=64.0)
            if kind == "coadd":
                lsst_background = official_lsst_background_for_tile(coadd_root, band, spec)
            else:
                cutout_path = out_patch / "cutouts" / spec.name / band / f"{image_kind}-{band}-{TRACT}-{PATCH}.fits"
                lsst_background = generated_lsst_background_for_cutout(
                    args=args,
                    out_patch=out_patch,
                    cutout_path=cutout_path,
                    band=band,
                    spec=spec,
                )
            targets = make_pu_dense_targets(
                clean,
                center,
                ignore,
                spec,
                x_col=X_COL,
                y_col=Y_COL,
                ellipse_sigma=ELLIPSE_SIGMA,
                confidence_levels=CONFIDENCE_LEVELS,
                core_radius=CORE_RADIUS,
                center_only_weight=CENTER_ONLY_WEIGHT,
                lsst_background_mask=lsst_background,
                strict_ignore_sources=strict,
            )
            write_targets(targets, out_patch / "band_targets" / band / f"{spec.name}.npz", None)
    print(f"{kind}: wrote {len(specs)} tiles to {out_patch}")
    for spec in specs:
        print(f"  {spec.name}")


def main() -> int:
    args = parse_args()
    image_kinds = {"coadd": "calexp", "noisy": "noisy", "denoised": "denoised"}
    for dataset in args.datasets:
        write_dataset(dataset, image_kinds[dataset], args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
