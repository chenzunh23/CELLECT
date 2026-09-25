#!/usr/bin/env python
"""Export HSC/LSST calexp PSF samples to FITS.

Run inside an LSST stack environment, for example:

    source ~/lsst_stack/loadLSST.sh
    setup lsst_distrib
    python scripts/export_hsc_psf_fits.py --calexp ... --output ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calexp", required=True, help="Input HSC/LSST calexp FITS.")
    parser.add_argument("--output", required=True, help="Output PSF FITS path.")
    parser.add_argument("--x", type=float, help="PSF sample x coordinate.")
    parser.add_argument("--y", type=float, help="PSF sample y coordinate.")
    parser.add_argument(
        "--coords",
        choices=("parent", "local"),
        default="parent",
        help="Coordinate frame for --x/--y. local adds the calexp bbox minimum.",
    )
    parser.add_argument(
        "--grid",
        nargs=2,
        type=int,
        metavar=("ROWS", "COLS"),
        help="Sample a regular grid over the calexp and write a mosaic plus cube.",
    )
    parser.add_argument(
        "--grid-margin",
        type=float,
        default=0.08,
        help="Fractional margin excluded from each calexp edge for --grid.",
    )
    parser.add_argument(
        "--stamp-size",
        type=int,
        help="Center crop/pad each PSF stamp to this odd pixel size.",
    )
    parser.add_argument(
        "--psf-image",
        choices=("image", "kernel"),
        default="image",
        help="Use psf.computeImage or psf.computeKernelImage.",
    )
    parser.add_argument(
        "--normalize",
        choices=("sum", "max", "none"),
        default="sum",
        help="Normalize each exported stamp.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output FITS.")
    return parser.parse_args()


def _load_lsst():
    try:
        import lsst.afw.image as afw_image
        import lsst.geom as geom
    except Exception as exc:  # pragma: no cover - requires LSST runtime
        raise RuntimeError(
            "This script requires LSST stack Python. Run with: "
            "source ~/lsst_stack/loadLSST.sh && setup lsst_distrib && python ..."
        ) from exc
    return afw_image, geom


def _crop_or_pad_center(arr: np.ndarray, size: int | None) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if size is None:
        return arr
    size = int(size)
    if size <= 0 or size % 2 == 0:
        raise ValueError(f"--stamp-size must be a positive odd integer, got {size}")
    out = np.zeros((size, size), dtype=np.float32)
    height, width = arr.shape
    src_y0 = max(0, height // 2 - size // 2)
    src_x0 = max(0, width // 2 - size // 2)
    src_y1 = min(height, src_y0 + size)
    src_x1 = min(width, src_x0 + size)
    src = arr[src_y0:src_y1, src_x0:src_x1]
    dst_y0 = size // 2 - src.shape[0] // 2
    dst_x0 = size // 2 - src.shape[1] // 2
    out[dst_y0 : dst_y0 + src.shape[0], dst_x0 : dst_x0 + src.shape[1]] = src
    return out


def _normalize(arr: np.ndarray, mode: str) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "none":
        return arr
    if mode == "sum":
        denom = float(arr.sum(dtype=np.float64))
    elif mode == "max":
        denom = float(np.max(arr))
    else:
        raise ValueError(mode)
    if not np.isfinite(denom) or abs(denom) <= 0.0:
        raise ValueError("PSF stamp has invalid normalization denominator")
    return arr / np.float32(denom)


def _point_from_args(args: argparse.Namespace, bbox, geom):
    if (args.x is None) != (args.y is None):
        raise ValueError("--x and --y must be provided together")
    if args.x is None:
        return geom.Point2D(bbox.getCenter())
    x = float(args.x)
    y = float(args.y)
    if args.coords == "local":
        x += float(bbox.getMinX())
        y += float(bbox.getMinY())
    return geom.Point2D(x, y)


def _grid_points(args: argparse.Namespace, bbox, geom) -> list:
    rows, cols = (int(v) for v in args.grid)
    if rows <= 0 or cols <= 0:
        raise ValueError(f"--grid ROWS COLS must be positive, got {args.grid}")
    margin = float(args.grid_margin)
    if margin < 0.0 or margin >= 0.5:
        raise ValueError("--grid-margin must be in [0, 0.5)")
    x0, x1 = float(bbox.getMinX()), float(bbox.getMaxX())
    y0, y1 = float(bbox.getMinY()), float(bbox.getMaxY())
    dx = (x1 - x0) * margin
    dy = (y1 - y0) * margin
    xs = np.linspace(x0 + dx, x1 - dx, cols, dtype=np.float64)
    ys = np.linspace(y0 + dy, y1 - dy, rows, dtype=np.float64)
    return [geom.Point2D(float(x), float(y)) for y in ys for x in xs]


def _sample_psf(psf, point, args: argparse.Namespace) -> np.ndarray:
    if args.psf_image == "image":
        stamp = psf.computeImage(point)
    elif args.psf_image == "kernel":
        stamp = psf.computeKernelImage(point)
    else:
        raise ValueError(args.psf_image)
    arr = np.asarray(stamp.array, dtype=np.float32)
    arr = _crop_or_pad_center(arr, args.stamp_size)
    return _normalize(arr, args.normalize)


def _make_mosaic(cube: np.ndarray, rows: int, cols: int) -> np.ndarray:
    n, height, width = cube.shape
    if n != rows * cols:
        raise ValueError(f"cube length {n} does not match grid {rows}x{cols}")
    mosaic = np.zeros((rows * height, cols * width), dtype=np.float32)
    for index, stamp in enumerate(cube):
        row = index // cols
        col = index % cols
        mosaic[row * height : (row + 1) * height, col * width : (col + 1) * width] = stamp
    return mosaic


def _base_header(args: argparse.Namespace, exp, psf) -> fits.Header:
    bbox = exp.getBBox()
    header = fits.Header()
    header["CALEXP"] = str(Path(args.calexp))
    header["PSFCLS"] = type(psf).__name__
    header["BBOXMINX"] = int(bbox.getMinX())
    header["BBOXMINY"] = int(bbox.getMinY())
    header["BBOXMAXX"] = int(bbox.getMaxX())
    header["BBOXMAXY"] = int(bbox.getMaxY())
    header["PSFIMG"] = args.psf_image
    header["NORM"] = args.normalize
    if args.stamp_size is not None:
        header["STAMPSZ"] = int(args.stamp_size)
    return header


def main() -> int:
    args = _parse_args()
    afw_image, geom = _load_lsst()
    exp = afw_image.ExposureF(str(Path(args.calexp)))
    if not exp.hasPsf():
        raise RuntimeError(f"Exposure has no PSF: {args.calexp}")
    psf = exp.getPsf()
    bbox = exp.getBBox(afw_image.PARENT)
    header = _base_header(args, exp, psf)

    if args.grid is None:
        point = _point_from_args(args, bbox, geom)
        image = _sample_psf(psf, point, args)
        header["X"] = float(point.getX())
        header["Y"] = float(point.getY())
        header["SUM"] = float(image.sum(dtype=np.float64))
        header["MAX"] = float(np.max(image))
        hdul = fits.HDUList([fits.PrimaryHDU(image.astype(np.float32), header=header)])
    else:
        rows, cols = (int(v) for v in args.grid)
        points = _grid_points(args, bbox, geom)
        stamps = [_sample_psf(psf, point, args) for point in points]
        shapes = {tuple(stamp.shape) for stamp in stamps}
        if len(shapes) != 1:
            raise ValueError(
                "Grid PSF stamps have different shapes; pass --stamp-size to force a common size"
            )
        cube = np.stack(stamps, axis=0).astype(np.float32)
        mosaic = _make_mosaic(cube, rows, cols)
        coords = np.asarray([[point.getX(), point.getY()] for point in points], dtype=np.float64)
        header["GRIDROWS"] = rows
        header["GRIDCOLS"] = cols
        header["GRIDMRGN"] = float(args.grid_margin)
        hdul = fits.HDUList(
            [
                fits.PrimaryHDU(mosaic.astype(np.float32), header=header),
                fits.ImageHDU(cube, name="PSF_CUBE"),
                fits.ImageHDU(coords, name="XY_PARENT"),
            ]
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    hdul.writeto(output, overwrite=bool(args.overwrite))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
