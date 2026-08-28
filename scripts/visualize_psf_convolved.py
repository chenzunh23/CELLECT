#!/usr/bin/env python
"""Visualize a PSF-smoothed or matched-filtered HSC/LSST or ZTF image crop.

HSC calexp PSFs are read through the LSST afw Exposure API.  ZTF images use the
per-exposure ``*_sciimgdaopsfcent.fits`` PSF stamp by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from scipy.signal import fftconvolve


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input image FITS.")
    parser.add_argument("--mode", choices=["auto", "hsc-lsst", "ztf-stamp", "ztf-gaussian"], default="auto")
    parser.add_argument(
        "--operation",
        choices=["smooth", "matched"],
        default="matched",
        help=(
            "smooth writes image convolved with the normalized PSF. matched writes the "
            "PSF-transpose matched-filter statistic, approximately in S/N units."
        ),
    )
    parser.add_argument("--psf", help="ZTF PSF stamp FITS. Defaults to *_sciimgdaopsfcent.fits when possible.")
    parser.add_argument("--hdu", type=int, default=0, help="Astropy FITS image HDU for non-LSST input.")
    parser.add_argument("--x0", type=int, default=0, help="Crop x origin in image pixels.")
    parser.add_argument("--y0", type=int, default=0, help="Crop y origin in image pixels.")
    parser.add_argument("--size", type=int, default=512, help="Crop size in pixels.")
    parser.add_argument("--psf-x", type=float, help="PSF x position for spatially varying HSC PSF.")
    parser.add_argument("--psf-y", type=float, help="PSF y position for spatially varying HSC PSF.")
    parser.add_argument("--gaussian-sigma", type=float, help="Override Gaussian sigma in pixels for ztf-gaussian.")
    parser.add_argument("--output-dir", default="output/psf_convolved", help="Output directory.")
    parser.add_argument("--stem", help="Output filename stem.")
    parser.add_argument("--write-fits", action="store_true", help="Also write convolved crop and kernel FITS files.")
    return parser.parse_args()


def _zscale_limits(image: np.ndarray) -> tuple[float, float]:
    values = image[np.isfinite(image)]
    if values.size == 0:
        return 0.0, 1.0
    if np.nanmax(values) == np.nanmin(values):
        v = float(values[0])
        return v - 1.0, v + 1.0
    try:
        return tuple(float(v) for v in ZScaleInterval().get_limits(values))
    except Exception:
        lo, hi = np.nanpercentile(values, [1, 99])
        return float(lo), float(hi)


def _crop_bounds(shape: tuple[int, int], x0: int, y0: int, size: int, pad: int = 0) -> tuple[slice, slice]:
    height, width = shape
    x1 = max(0, int(x0) - pad)
    y1 = max(0, int(y0) - pad)
    x2 = min(width, int(x0) + int(size) + pad)
    y2 = min(height, int(y0) + int(size) + pad)
    return slice(y1, y2), slice(x1, x2)


def _normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=np.float32)
    kernel = np.nan_to_num(kernel, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(kernel.sum())
    if not np.isfinite(total) or total == 0.0:
        raise ValueError("PSF kernel has zero or invalid sum")
    return kernel / total


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(3, int(np.ceil(4.0 * float(sigma))))
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-0.5 * (x * x + y * y) / float(sigma * sigma))
    return _normalize_kernel(kernel)


def _robust_sigma(image: np.ndarray) -> float:
    values = image[np.isfinite(image)]
    if values.size == 0:
        return 1.0
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.nanstd(values))
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = 1.0
    return sigma


def _read_hsc_lsst(path: Path, x: float | None, y: float | None) -> tuple[np.ndarray, np.ndarray, str, tuple[int, int]]:
    try:
        import lsst.afw.image as afw_image
        import lsst.geom as geom
    except Exception as exc:  # pragma: no cover - depends on external LSST stack
        raise RuntimeError(
            "hsc-lsst mode requires LSST stack Python. Run with: "
            "source ~/lsst_stack/loadLSST.bash && setup lsst_distrib && python ..."
        ) from exc

    exp = afw_image.ExposureF.readFits(str(path))
    image = np.asarray(exp.image.array, dtype=np.float32)
    bbox = exp.getBBox()
    point = geom.Point2D(float(x), float(y)) if x is not None and y is not None else geom.Point2D(bbox.getCenter())
    kernel = _normalize_kernel(exp.getPsf().computeKernelImage(point).array)
    origin = (int(bbox.getMinX()), int(bbox.getMinY()))
    return image, kernel, type(exp.getPsf()).__name__, origin


def _infer_ztf_psf_stamp(path: Path) -> Path:
    name = path.name
    candidates = []
    if name.endswith("_sciimg.fits"):
        candidates.append(path.with_name(name.replace("_sciimg.fits", "_sciimgdaopsfcent.fits")))
    candidates.append(path.with_name(path.stem + "daopsfcent.fits"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Cannot infer ZTF PSF stamp; pass --psf explicitly.")


def _read_ztf_stamp(path: Path, psf_path: str | None, hdu: int) -> tuple[np.ndarray, np.ndarray, str, tuple[int, int]]:
    image = np.asarray(fits.getdata(path, hdu), dtype=np.float32)
    stamp_path = Path(psf_path) if psf_path else _infer_ztf_psf_stamp(path)
    kernel = _normalize_kernel(fits.getdata(stamp_path, 0))
    return image, kernel, f"ZTF stamp {stamp_path.name}", (0, 0)


def _read_ztf_gaussian(path: Path, hdu: int, sigma: float | None) -> tuple[np.ndarray, np.ndarray, str, tuple[int, int]]:
    image = np.asarray(fits.getdata(path, hdu), dtype=np.float32)
    header = fits.getheader(path, hdu)
    if sigma is None:
        seeing = header.get("SEEING") or header.get("FWHM")
        if seeing is None:
            raise ValueError("ztf-gaussian mode needs --gaussian-sigma or SEEING/FWHM in the FITS header")
        sigma = float(seeing) / 2.354820045
    return image, _gaussian_kernel(float(sigma)), f"Gaussian sigma={sigma:.3f}px", (0, 0)


def _convolve_crop(
    image: np.ndarray,
    kernel: np.ndarray | None,
    args: argparse.Namespace,
    origin: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, str]:
    local_x0 = int(args.x0) - int(origin[0])
    local_y0 = int(args.y0) - int(origin[1])
    if kernel is None:
        raise ValueError("PSF kernel is required")
    pad = int(max(kernel.shape) // 2)
    ys, xs = _crop_bounds(image.shape, local_x0, local_y0, args.size, pad=pad)
    padded = np.nan_to_num(image[ys, xs].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if padded.size == 0:
        raise ValueError(
            f"empty crop: requested physical x0={args.x0}, y0={args.y0}, size={args.size}, "
            f"image origin={origin}, image shape={image.shape}"
        )
    if args.operation == "smooth":
        conv_padded = fftconvolve(padded, kernel, mode="same").astype(np.float32)
        label = "PSF-smoothed"
    elif args.operation == "matched":
        # The HSC paper's optimal point-source detection image is a matched
        # filter: image convolved with the transpose of the PSF.  For uniform
        # sky noise, dividing by sigma*sqrt(sum(P^2)) gives an approximate S/N
        # map.  The point-source profile becomes the PSF autocorrelation; this
        # is expected and should not be interpreted as the original morphology.
        matched_kernel = kernel[::-1, ::-1]
        conv_num = fftconvolve(padded, matched_kernel, mode="same").astype(np.float32)
        sigma = _robust_sigma(padded)
        denom = sigma * float(np.sqrt(np.sum(matched_kernel * matched_kernel)))
        conv_padded = conv_num / denom if denom > 0.0 else conv_num
        label = "matched-filter S/N"
    else:
        raise ValueError(args.operation)

    y_offset = local_y0 - ys.start
    x_offset = local_x0 - xs.start
    crop = padded[y_offset : y_offset + args.size, x_offset : x_offset + args.size]
    conv = conv_padded[y_offset : y_offset + args.size, x_offset : x_offset + args.size]
    return crop, conv, label


def _save_panel(raw: np.ndarray, conv: np.ndarray, kernel: np.ndarray | None, product_label: str, title: str, out_png: Path) -> None:
    diff = conv - raw
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    panels = [
        ("raw", raw, "gray", _zscale_limits(raw)),
        ("PSF kernel", kernel if kernel is not None else np.zeros((3, 3), dtype=np.float32), "magma", None),
        (product_label, conv, "gray", _zscale_limits(conv)),
        (f"{product_label} - raw", diff, "coolwarm", None),
    ]
    dmax = float(np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 99)) if np.isfinite(diff).any() else 1.0
    for ax, (name, image, cmap, limits) in zip(axes, panels):
        if name == f"{product_label} - raw":
            limits = (-dmax, dmax)
        if limits is None:
            im = ax.imshow(image, origin="lower", cmap=cmap)
        else:
            im = ax.imshow(image, origin="lower", cmap=cmap, vmin=limits[0], vmax=limits[1])
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        if name == "PSF kernel":
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input)
    mode = args.mode
    if mode == "auto":
        mode = "ztf-stamp" if "ztf_" in input_path.name and input_path.name.endswith("_sciimg.fits") else "hsc-lsst"

    if mode == "hsc-lsst":
        psf_x = args.psf_x if args.psf_x is not None else args.x0 + args.size / 2.0
        psf_y = args.psf_y if args.psf_y is not None else args.y0 + args.size / 2.0
        image, kernel, psf_name, origin = _read_hsc_lsst(input_path, psf_x, psf_y)
        args.gaussian_sigma = None
    elif mode == "ztf-stamp":
        image, kernel, psf_name, origin = _read_ztf_stamp(input_path, args.psf, args.hdu)
        args.gaussian_sigma = None
    elif mode == "ztf-gaussian":
        image, kernel, psf_name, origin = _read_ztf_gaussian(input_path, args.hdu, args.gaussian_sigma)
        if args.gaussian_sigma is None:
            header = fits.getheader(input_path, args.hdu)
            args.gaussian_sigma = float(header.get("SEEING") or header.get("FWHM")) / 2.354820045
    else:
        raise ValueError(mode)

    raw, conv, product_label = _convolve_crop(image, kernel, args, origin)
    stem = args.stem or f"{input_path.stem}_x{args.x0}_y{args.y0}_{mode}_{args.operation}"
    out_dir = Path(args.output_dir)
    out_png = out_dir / f"{stem}_psf_convolved.png"
    _save_panel(raw, conv, kernel, product_label, f"{input_path.name} | {psf_name}", out_png)
    print(f"wrote {out_png}")

    if args.write_fits:
        out_dir.mkdir(parents=True, exist_ok=True)
        fits.writeto(out_dir / f"{stem}_raw_crop.fits", raw.astype(np.float32), overwrite=True)
        fits.writeto(out_dir / f"{stem}_{args.operation}.fits", conv.astype(np.float32), overwrite=True)
        if kernel is not None:
            fits.writeto(out_dir / f"{stem}_kernel.fits", kernel.astype(np.float32), overwrite=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
