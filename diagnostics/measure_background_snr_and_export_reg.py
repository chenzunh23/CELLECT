#!/usr/bin/env python3
"""Estimate background, measure SNR, and export SNR-annotated REG files.

This script is intended to provide one consistent diagnostic path:

1. Background estimate:
   - Pixel background location is estimated by SigmaEx on *only the 1D
     background pixels* selected by the provided background mask.  If no
     background mask is provided, SigmaEx is run on all finite pixels with
     ``mode=le_median`` by default.
   - Do not pass a zero-filled image to SigmaEx. SigmaEx treats zeros as real
     values and will fit the artificial zero peak.
   - The global noise denominator is estimated from fixed-aperture sums
     sampled inside the 512x512 background mask.

2. SNR measurements:
   - global_snr: no local background subtraction.
     (aperture_sum - sigmaex_pixel_median * aperture_pixels) / aperture_sum_std
   - local_snr: same local background style as CELLECT preprocessing:
     aperture background is the annulus mean and std after excluding
     clean/center-only/strict-center-only and quality-mask pixels.  SNR uses
     local annulus statistics:
     (aperture_sum - annulus_mean * aperture_pixels) /
     (annulus_std * sqrt(aperture_pixels)).

3. Output:
   - CSV with both SNR definitions.
   - REG with radius-7 circles, text only ``SNR=x``.  By default this uses
     local_snr; pass --reg-snr-field global_snr to write global SNR instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import numpy as np
from astropy.io import fits
from scipy import ndimage
from sigmaex import SigmaEx


REG_RE = re.compile(
    r"(circle|ellipse|point)\((?P<body>[^)]*)\)",
    re.IGNORECASE,
)
TEXT_EXTRACT_RE = re.compile(r"text=\{(?P<text>[^}]*)\}")
TEXT_ATTR_RE = re.compile(r"\s*text=\{[^}]*\}")


def _disk(radius: int) -> np.ndarray:
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy <= radius * radius).astype(np.float32)


def _read_image(path: Path, hdu: int | None) -> np.ndarray:
    if hdu is not None:
        return np.asarray(fits.getdata(path, ext=int(hdu)), dtype=np.float32)
    with fits.open(path, memmap=True) as hdul:
        for h in hdul:
            if h.data is not None and getattr(h.data, "ndim", None) == 2:
                return np.asarray(h.data, dtype=np.float32)
    raise RuntimeError(f"no 2D image found in {path}")


def _read_background_mask(path: Path) -> np.ndarray:
    payload = np.load(path)
    for key in ("background_mask", "background", "mask"):
        if key in payload:
            return np.asarray(payload[key], dtype=bool)
    if len(payload.files) == 1:
        return np.asarray(payload[payload.files[0]], dtype=bool)
    raise RuntimeError(f"cannot identify background mask key in {path}; keys={payload.files}")


def _crop_full_mask(mask: np.ndarray, x0: int, y0: int, size: int) -> np.ndarray:
    if mask.shape == (size, size):
        return mask
    return np.asarray(mask[y0 : y0 + size, x0 : x0 + size], dtype=bool)


def _read_source_exclude_mask(path: Path | None, *, size: int) -> np.ndarray | None:
    if path is None:
        return None
    data = np.load(path)
    clean = np.asarray(data["clean_mask"], dtype=bool) if "clean_mask" in data else False
    center = np.asarray(data["center_only_mask"], dtype=bool) if "center_only_mask" in data else False
    strict_center = (
        np.asarray(data["strict_center_only_mask"], dtype=bool) if "strict_center_only_mask" in data else False
    )
    mask = np.asarray(clean | center | strict_center, dtype=bool)
    if mask.shape != (size, size):
        raise ValueError(f"source mask shape {mask.shape} != {(size, size)} for {path}")
    return mask


def _mask_plane_bit_mapping(header: fits.Header) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for key, value in header.items():
        text_key = str(key).upper()
        if text_key.startswith("MP_"):
            try:
                mapping[text_key[3:]] = int(value)
            except Exception:
                pass
    return mapping


def _read_quality_mask(path: Path | None, mask_planes: Iterable[str], *, size: int) -> np.ndarray | None:
    if path is None:
        return None
    planes = [str(p).strip().upper() for p in mask_planes if str(p).strip()]
    if not planes:
        return None
    with fits.open(path, memmap=False) as hdul:
        if "MASK" in hdul:
            hdu = hdul["MASK"]
        elif len(hdul) > 2 and hdul[2].data is not None:
            hdu = hdul[2]
        else:
            return None
        data = np.asarray(hdu.data)
        bit_mapping = _mask_plane_bit_mapping(hdu.header)
        selected = np.zeros(data.shape, dtype=bool)
        values = data.astype(np.int64, copy=False)
        for plane in planes:
            bit = bit_mapping.get(plane)
            if bit is not None:
                selected |= (values & (1 << int(bit))) != 0
        if selected.shape != (size, size):
            raise ValueError(f"quality mask shape {selected.shape} != {(size, size)} for {path}")
        return selected


def _base_label(text: str | None, fallback: int) -> str:
    if text:
        return text.split()[0]
    return f"src_{fallback:04d}"


def _parse_reg(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = REG_RE.search(line)
        if not match:
            continue
        body = [float(x.strip()) for x in match.group("body").split(",")[:2]]
        if len(body) < 2:
            continue
        attrs = ""
        text = None
        if "#" in line:
            attrs = line.split("#", 1)[1].strip()
            text_match = TEXT_EXTRACT_RE.search(attrs)
            text = text_match.group("text") if text_match else None
            attrs = TEXT_ATTR_RE.sub("", attrs).strip()
        rows.append(
            {
                "label": _base_label(text, len(rows) + 1),
                "x": body[0],
                "y": body[1],
                "reg_attrs": attrs,
            }
        )
    return rows


def _parse_csv(path: Path, x_col: str, y_col: str, label_col: str | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, 1):
            rows.append(
                {
                    "label": str(row[label_col]) if label_col and label_col in row else f"src_{idx:04d}",
                    "x": float(row[x_col]),
                    "y": float(row[y_col]),
                }
            )
    return rows


def _read_sources(args: argparse.Namespace) -> list[dict[str, object]]:
    if str(args.input_format).lower() == "reg":
        return _parse_reg(args.input)
    if str(args.input_format).lower() == "csv":
        return _parse_csv(args.input, args.x_col, args.y_col, args.label_col)
    raise ValueError("--input-format must be reg or csv")


def _aperture_sum(image: np.ndarray, x: float, y: float, kernel: np.ndarray) -> tuple[float, int]:
    radius = kernel.shape[0] // 2
    cx = int(round(x))
    cy = int(round(y))
    y0 = max(0, cy - radius)
    y1 = min(image.shape[0], cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(image.shape[1], cx + radius + 1)
    ky0 = y0 - (cy - radius)
    ky1 = ky0 + (y1 - y0)
    kx0 = x0 - (cx - radius)
    kx1 = kx0 + (x1 - x0)
    values = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
    weights = np.asarray(kernel[ky0:ky1, kx0:kx1], dtype=np.float64)
    good = np.isfinite(values) & (weights > 0)
    return float(np.sum(values[good] * weights[good])), int(np.count_nonzero(good))


def _sample_background_aperture_sums(
    image: np.ndarray,
    background_mask: np.ndarray | None,
    *,
    radius: int,
    n_sample: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    kernel = _disk(radius)
    finite = np.isfinite(image)
    base = finite if background_mask is None else (background_mask & finite)
    valid = ndimage.binary_erosion(base, structure=kernel.astype(bool), border_value=0)
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        raise RuntimeError("no valid background aperture centers")
    rng = np.random.default_rng(seed)
    n = min(int(n_sample), int(xs.size))
    idx = rng.choice(xs.size, size=n, replace=False)
    sums_image = ndimage.convolve(np.where(finite, image, 0.0).astype(np.float32), kernel, mode="constant", cval=0.0)
    sums = np.asarray(sums_image[ys[idx], xs[idx]], dtype=np.float64)
    return sums, {
        "background_pixels": int(np.count_nonzero(background_mask)) if background_mask is not None else None,
        "background_mode_pixels": int(np.count_nonzero(base)),
        "valid_background_aperture_centers": int(xs.size),
        "sampled_background_apertures": int(n),
        "aperture_pixels": int(kernel.sum()),
    }


def _sigmaex_background(values: np.ndarray, *, sigma: float, mode: str, nbins: int, sample: int) -> dict[str, float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite.size == 0:
        raise RuntimeError("no finite background values for SigmaEx")
    sigma_arg: float | int = int(sigma) if float(sigma).is_integer() else float(sigma)
    fit = SigmaEx(finite, sigma=sigma_arg, mode=mode, nbins=nbins, sample=sample)
    return {
        "n": int(finite.size),
        "raw_mean": float(np.mean(finite)),
        "raw_median": float(np.median(finite)),
        "raw_std": float(np.std(finite, ddof=1)),
        "sigmaex_clip_mean": float(fit.sigma_clipped_mean),
        "sigmaex_clip_median": float(fit.sigma_clipped_median),
        "sigmaex_clip_std": float(fit.sigma_clipped_std),
        "sigmaex_fit_mu": float(fit.gaussian_fit_mu),
        "sigmaex_fit_sigma": float(abs(fit.gaussian_fit_sigma)),
        "sigmaex_redchi2": float(fit.redchi2),
    }


def _local_annulus_stats(
    image: np.ndarray,
    x: float,
    y: float,
    *,
    r_in: float,
    r_out: float,
    source_exclude: np.ndarray | None,
    quality_exclude: np.ndarray | None,
    min_pixels: int,
) -> tuple[float, float, float, int]:
    h, w = image.shape
    rmax = float(r_out)
    x0 = max(0, int(math.floor(float(x) - rmax - 1.0)))
    x1 = min(w, int(math.ceil(float(x) + rmax + 2.0)))
    y0 = max(0, int(math.floor(float(y) - rmax - 1.0)))
    y1 = min(h, int(math.ceil(float(y) + rmax + 2.0)))
    if x0 >= x1 or y0 >= y1:
        return math.nan, math.nan, math.nan, 0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.sqrt((xx.astype(np.float32) - float(x)) ** 2 + (yy.astype(np.float32) - float(y)) ** 2)
    patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float32)
    mask = (rr >= float(r_in)) & (rr < float(r_out)) & np.isfinite(patch)
    if source_exclude is not None:
        mask &= ~source_exclude[y0:y1, x0:x1]
    if quality_exclude is not None:
        mask &= ~quality_exclude[y0:y1, x0:x1]
    vals = patch[mask].astype(np.float64)
    if vals.size < int(min_pixels):
        return math.nan, math.nan, math.nan, int(vals.size)
    return float(np.mean(vals)), float(np.median(vals)), float(np.std(vals, ddof=1)), int(vals.size)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "label",
        "x",
        "y",
        "reg_attrs",
        "aperture_sum",
        "aperture_pixels",
        "global_background_per_pixel",
        "global_aperture_sigma",
        "global_snr",
        "local_annulus_mean",
        "local_annulus_median",
        "local_annulus_std",
        "local_annulus_pixels",
        "local_snr",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_reg(path: Path, rows: list[dict[str, object]], snr_field: str, radius: float) -> None:
    lines = [
        "# Region file format: DS9 version 4.1",
        'global color=green dashlist=8 3 width=2 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',
        "image",
    ]
    for row in rows:
        snr = float(row[snr_field])
        text = f"SNR={snr:.2f}" if math.isfinite(snr) else "SNR=nan"
        attrs = str(row.get("reg_attrs", "")).strip()
        if attrs:
            attrs = f"{attrs} text={{{text}}}"
        else:
            attrs = f"text={{{text}}}"
        lines.append(f"circle({float(row['x']):.3f},{float(row['y']):.3f},{float(radius):.3f}) # {attrs}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="512x512 image/cutout FITS used for photometry.")
    parser.add_argument("--image-hdu", type=int, default=None, help="Image HDU. If omitted, the first 2D HDU is used.")
    parser.add_argument(
        "--background-mask",
        type=Path,
        default=None,
        help="Optional 512x512 or full-patch background_mask.npz. If omitted, all finite pixels are used with le_median SigmaEx mode by default.",
    )
    parser.add_argument("--background-x0", type=int, default=0, help="Crop x0 if background mask is full patch.")
    parser.add_argument("--background-y0", type=int, default=0, help="Crop y0 if background mask is full patch.")
    parser.add_argument("--target-npz", type=Path, default=None, help="Target npz with clean/center-only masks.")
    parser.add_argument("--quality-fits", type=Path, default=None, help="FITS containing LSST MASK plane.")
    parser.add_argument("--quality-mask-planes", nargs="*", default=["SAT", "BAD", "BRIGHT_OBJECT", "NO_DATA", "EDGE", "UNMASKEDNAN"])
    parser.add_argument("--input", type=Path, required=True, help="Input REG or CSV source table.")
    parser.add_argument("--input-format", choices=("reg", "csv"), default="reg")
    parser.add_argument("--x-col", default="x")
    parser.add_argument("--y-col", default="y")
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--ap-radius", type=int, default=6)
    parser.add_argument("--annulus-r-in", type=float, default=10.0)
    parser.add_argument("--annulus-r-out", type=float, default=15.0)
    parser.add_argument("--min-annulus-pixels", type=int, default=50)
    parser.add_argument("--num-background-apertures", type=int, default=20000)
    parser.add_argument("--sigmaex-sigma", type=float, default=3.0)
    parser.add_argument("--sigmaex-mode", default="all")
    parser.add_argument(
        "--no-background-mask-sigmaex-mode",
        default="le_median",
        help="SigmaEx mode used when --background-mask is omitted.",
    )
    parser.add_argument("--sigmaex-nbins", type=int, default=300)
    parser.add_argument("--sigmaex-sample", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--reg-snr-field", choices=("global_snr", "local_snr"), default="local_snr")
    parser.add_argument("--reg-radius", type=float, default=7.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = _read_image(args.image, args.image_hdu)
    if image.shape != (512, 512):
        raise ValueError(f"--image must be 512x512 for this diagnostic, got {image.shape}")
    if args.background_mask is not None:
        background = _crop_full_mask(_read_background_mask(args.background_mask), args.background_x0, args.background_y0, 512)
        background_mode = str(args.sigmaex_mode)
        background_mode_name = "provided_background_mask"
    else:
        background = None
        background_mode = str(args.no_background_mask_sigmaex_mode)
        background_mode_name = "no_background_mask_all_finite_pixels"
    source_exclude = _read_source_exclude_mask(args.target_npz, size=512)
    quality_exclude = _read_quality_mask(args.quality_fits, args.quality_mask_planes, size=512)

    if background is None:
        bg_pixels = image[np.isfinite(image)]
    else:
        bg_pixels = image[background & np.isfinite(image)]
    pixel_bg = _sigmaex_background(
        bg_pixels,
        sigma=float(args.sigmaex_sigma),
        mode=background_mode,
        nbins=int(args.sigmaex_nbins),
        sample=int(args.sigmaex_sample),
    )
    bg_sums, bg_ap_meta = _sample_background_aperture_sums(
        image,
        background,
        radius=int(args.ap_radius),
        n_sample=int(args.num_background_apertures),
        seed=int(args.seed),
    )
    aperture_bg = _sigmaex_background(
        bg_sums,
        sigma=float(args.sigmaex_sigma),
        mode=background_mode,
        nbins=int(args.sigmaex_nbins),
        sample=int(args.sigmaex_sample),
    )
    global_bg = float(pixel_bg["sigmaex_clip_median"])
    global_sigma = float(aperture_bg["sigmaex_fit_sigma"])
    kernel = _disk(int(args.ap_radius))
    sources = _read_sources(args)
    rows: list[dict[str, object]] = []
    for source in sources:
        x = float(source["x"])
        y = float(source["y"])
        ap_sum, ap_n = _aperture_sum(image, x, y, kernel)
        global_flux = ap_sum - global_bg * float(ap_n)
        global_snr = global_flux / global_sigma if global_sigma > 0 else math.nan
        local_mean, local_median, local_std, local_n = _local_annulus_stats(
            image,
            x,
            y,
            r_in=float(args.annulus_r_in),
            r_out=float(args.annulus_r_out),
            source_exclude=source_exclude,
            quality_exclude=quality_exclude,
            min_pixels=int(args.min_annulus_pixels),
        )
        if math.isfinite(local_mean) and math.isfinite(local_std) and local_std > 0:
            local_flux = ap_sum - local_mean * float(ap_n)
            local_snr = local_flux / (local_std * math.sqrt(float(ap_n)))
        else:
            local_snr = math.nan
        rows.append(
            {
                "label": str(source["label"]),
                "x": x,
                "y": y,
                "reg_attrs": str(source.get("reg_attrs", "")),
                "aperture_sum": ap_sum,
                "aperture_pixels": ap_n,
                "global_background_per_pixel": global_bg,
                "global_aperture_sigma": global_sigma,
                "global_snr": global_snr,
                "local_annulus_mean": local_mean,
                "local_annulus_median": local_median,
                "local_annulus_std": local_std,
                "local_annulus_pixels": local_n,
                "local_snr": local_snr,
            }
        )

    out_prefix = args.out_prefix.expanduser().resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(out_prefix.with_suffix(".csv"), rows)
    _write_reg(out_prefix.with_suffix(".reg"), rows, args.reg_snr_field, float(args.reg_radius))
    summary = {
        "image": str(args.image),
        "background_mask": str(args.background_mask) if args.background_mask is not None else None,
        "background_mode": background_mode_name,
        "sigmaex_mode_used": background_mode,
        "target_npz": str(args.target_npz) if args.target_npz else None,
        "quality_fits": str(args.quality_fits) if args.quality_fits else None,
        "quality_mask_planes": [str(p) for p in args.quality_mask_planes],
        "sigmaex_zero_fill_warning": "SigmaEx must receive only selected 1D background pixels; zero-filled non-background pixels are fitted as real zeros.",
        "background_pixels": int(np.count_nonzero(background)) if background is not None else None,
        "finite_pixels_used_without_background_mask": int(np.count_nonzero(np.isfinite(image))) if background is None else None,
        "source_exclude_pixels": int(np.count_nonzero(source_exclude)) if source_exclude is not None else 0,
        "quality_exclude_pixels": int(np.count_nonzero(quality_exclude)) if quality_exclude is not None else 0,
        "pixel_background_sigmaex": pixel_bg,
        "aperture_background_sigmaex": aperture_bg,
        "aperture_background_sampling": bg_ap_meta,
        "num_sources": len(rows),
        "outputs": {
            "csv": str(out_prefix.with_suffix(".csv")),
            "reg": str(out_prefix.with_suffix(".reg")),
            "summary": str(out_prefix.with_suffix(".summary.json")),
        },
    }
    out_prefix.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
