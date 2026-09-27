"""Multi-scale SExtractor sky mask for JWST reference images.

True means usable sky. Every detected component is treated alike: no star
catalog, hand-picked seed, halo/dragon's-breath classifier, or LSST dependency.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
from astropy.io import fits
from scipy import ndimage

METHOD = 'sextractor_multiscale_sky_v1'


@dataclass(frozen=True)
class AggressiveSkyConfig:
    # Slightly tighter than the 2026-09-16 bright-star experiment: the broad
    # Gaussian is 28 instead of 40 px and detections use higher thresholds.
    factors: tuple[int, ...] = (1, 2, 4)
    smoothing_sigma_pixels: tuple[float, ...] = (2., 10., 28.)
    thresholds_sigma: tuple[float, ...] = (1.8, 2.2, 2.7)
    minarea_native_pixels: tuple[int, ...] = (16, 32, 128)
    grow_native_pixels: tuple[int, ...] = (6, 16, 32) # (12, 32, 64) too aggressive
    min_valid_fraction: float = 0.5
    # SExtractor sees each already smoothed image and must not fit its broad
    # source wings away through a local background mesh.
    deblend_nthresh: int = 1
    clean: bool = False

    def __post_init__(self):
        n = len(self.factors)
        if (n == 0 or any(len(v) != n for v in (
                self.smoothing_sigma_pixels, self.thresholds_sigma,
                self.minarea_native_pixels, self.grow_native_pixels))
                or any(not isinstance(f, int) or f < 1 for f in self.factors)
                or any(not np.isfinite(s) or s <= 0 for s in self.smoothing_sigma_pixels)
                or any(not np.isfinite(t) or t <= 0 for t in self.thresholds_sigma)
                or any(a < 1 for a in self.minarea_native_pixels)
                or any(g < 0 for g in self.grow_native_pixels)
                or not 0 < self.min_valid_fraction <= 1
                or self.deblend_nthresh < 1):
            raise ValueError('Invalid aggressive sky configuration')


def _block_mean(image, valid, factor):
    if factor == 1:
        return image, valid.astype(np.float32)
    h, w = image.shape
    ph, pw = (-h) % factor, (-w) % factor
    filled = np.pad(np.where(valid, image, 0.), ((0, ph), (0, pw)))
    weights = np.pad(valid.astype(np.float32), ((0, ph), (0, pw)))
    shape = ((h+ph)//factor, factor, (w+pw)//factor, factor)
    values = filled.reshape(shape).sum(axis=(1, 3))
    weights = weights.reshape(shape).sum(axis=(1, 3))
    np.divide(values, weights, out=values, where=weights > 0)
    values[weights == 0] = 0
    return values.astype(np.float32, copy=False), weights / factor**2


def _smooth_valid(image, coverage, sigma):
    good = coverage > 0
    signal = ndimage.gaussian_filter(image * coverage, sigma, mode='nearest',
                                     output=np.float32, truncate=3)
    norm = ndimage.gaussian_filter(coverage, sigma, mode='nearest',
                                   output=np.float32, truncate=3)
    np.divide(signal, norm, out=signal, where=norm > 0.1)
    signal[norm <= 0.1] = 0
    return signal, good


def _robust_background(values):
    from astropy.stats import sigma_clipped_stats
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 0., 0.
    sample = finite[::max(1, finite.size // 300000)]
    _, median, std = sigma_clipped_stats(sample, sigma=3., maxiters=5)
    return float(median), float(std)


def _detect_pass(image, valid, factor, sigma, nsigma, minarea, grow, work, sex, log, config):
    small, coverage = _block_mean(image, valid, factor)
    smoothed, _ = _smooth_valid(small, coverage, sigma / factor)
    good = coverage >= config.min_valid_fraction
    median, std = _robust_background(smoothed[good])
    if not np.isfinite(std) or std <= 0:
        return np.zeros(image.shape, bool), dict(factor=factor, std=std, detected_pixels=0, reason='no_noise')
    smoothed -= median
    smoothed[~good] = 0
    threshold = nsigma * std
    temp = Path(work)
    input_path, weight_path, seg_path = temp/'science.fits', temp/'weight.fits', temp/'segmentation.fits'
    fits.writeto(input_path, smoothed, overwrite=True)
    fits.writeto(weight_path, good.astype(np.float32), overwrite=True)
    # Same manual-background, no-CLEAN segmentation strategy as the 09-16 mask.
    # Every component contributes; no target-star seeds or morphology labels.
    opts = dict(CATALOG_NAME=str(temp/'catalog.cat'), CATALOG_TYPE='ASCII_HEAD',
        PARAMETERS_NAME=str(temp/'params'), DETECT_TYPE='CCD', THRESH_TYPE='ABSOLUTE',
        DETECT_THRESH=threshold, ANALYSIS_THRESH=threshold,
        DETECT_MINAREA=max(1, int(np.ceil(minarea / factor**2))),
        FILTER='N', DEBLEND_NTHRESH=config.deblend_nthresh, DEBLEND_MINCONT=1,
        CLEAN='Y' if config.clean else 'N', BACK_TYPE='MANUAL', BACK_VALUE=0,
        WEIGHT_TYPE='MAP_WEIGHT', WEIGHT_IMAGE=str(weight_path), WEIGHT_THRESH=.5,
        WEIGHT_GAIN='N', CHECKIMAGE_TYPE='SEGMENTATION', CHECKIMAGE_NAME=str(seg_path),
        MEMORY_OBJSTACK=65536, MEMORY_PIXSTACK=10000000,
        MEMORY_BUFSIZE=max(1024, smoothed.shape[1]), VERBOSE_TYPE='NORMAL')
    (temp/'params').write_text('NUMBER\nX_IMAGE\nY_IMAGE\n')
    config = temp/'config.sex'
    config.write_text('\n'.join(f'{k} {v}' for k, v in opts.items()) + '\n')
    with Path(log).open('a') as stream:
        stream.write(f'factor={factor} sigma={sigma} nsigma={nsigma} threshold={threshold} minarea={opts["DETECT_MINAREA"]} grow={grow}\n')
        stream.flush()
        try:
            subprocess.run([sex, str(input_path), '-c', str(config)], check=True,
                           stdout=stream, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            stream.flush()
            detail = '\n'.join(Path(log).read_text(errors='replace').splitlines()[-12:])
            raise RuntimeError(f'SExtractor failed with exit {exc.returncode}; log tail:\n{detail}') from exc
    # A successful exit does not guarantee the checkimage is readable. Retry
    # the rare missing-output case once before reporting a failed task.
    if not seg_path.is_file():
        time.sleep(0.25)
    if not seg_path.is_file():
        with Path(log).open('a') as stream:
            stream.write('Segmentation checkimage missing after successful SExtractor exit; retry once.\n')
            stream.flush()
            subprocess.run([sex, str(input_path), '-c', str(config)], check=True,
                           stdout=stream, stderr=subprocess.STDOUT)
    if not seg_path.is_file():
        raise FileNotFoundError(f'SExtractor did not produce segmentation checkimage: {seg_path}')
    labels = fits.getdata(seg_path, memmap=True)
    mask = np.asarray(labels > 0, bool)
    count = int(mask.sum())
    del labels, smoothed, small
    if grow:
        radius = int(np.ceil(grow / factor))
        mask = ndimage.distance_transform_edt(~mask) <= radius
    if factor > 1:
        mask = np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1)
        mask = mask[:image.shape[0], :image.shape[1]]
    return mask, dict(factor=factor, smoothing_sigma_pixels=sigma,
                      threshold_sigma=nsigma, threshold_image_units=threshold,
                      minarea_native_pixels=minarea, grow_native_pixels=grow,
                      detected_pixels=count)


def aggressive_sextractor_background(image, header, output, *,
                                      config=AggressiveSkyConfig(), source='', executable=None,
                                      scratch_dir=None):
    """Write/read an NPZ background mask; 1=True=finite sky, 0=source/invalid.

    ``header`` is accepted for the same calling convention as the single-pass
    SExtractor helper. The mask is on exactly the input pixel grid.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError('Expected a 2D science image')
    valid = np.isfinite(image)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_path = Path(source) if source else None
    source_info = (dict(path=str(source_path), size=source_path.stat().st_size,
                   mtime_ns=source_path.stat().st_mtime_ns)
                   if source_path and source_path.exists() else dict(source=str(source)))
    signature = dict(method=METHOD, config=json.loads(json.dumps(asdict(config))), source=source_info,
                     shape=list(image.shape),
                     image_sha256=hashlib.sha256(np.ascontiguousarray(image).view(np.uint8)).hexdigest())
    meta = output.with_suffix('.json')
    if output.exists() and meta.exists():
        old = json.loads(meta.read_text())
        if old.get('signature') == signature:
            with np.load(output) as data:
                cached = np.asarray(data['background_mask'], bool)
            if cached.shape == image.shape:
                return cached & valid
    sky = valid.copy()
    passes = []
    log = output.with_suffix('.log')
    if valid.any():
        sex = executable or shutil.which('sex') or shutil.which('source-extractor')
        if sex is None:
            raise FileNotFoundError('SExtractor executable not found')
        log.write_text(f'{METHOD}\nsource={source}\n')
        scratch = output.parent if scratch_dir is None else Path(scratch_dir)
        scratch.mkdir(parents=True, exist_ok=True)
        if ',' in str(scratch):
            raise ValueError(f'SExtractor scratch path cannot contain commas: {scratch}')
        with tempfile.TemporaryDirectory(prefix='aggressive-sex-', dir=scratch) as tmp:
            for i, (factor, sigma, nsigma, minarea, grow) in enumerate(zip(
                    config.factors, config.smoothing_sigma_pixels,
                    config.thresholds_sigma, config.minarea_native_pixels,
                    config.grow_native_pixels)):
                work = Path(tmp)/f'pass{i}'
                work.mkdir()
                found, stats = _detect_pass(image, valid, factor, sigma, nsigma,
                                             minarea, grow, work, sex, log, config)
                sky &= ~found
                stats['excluded_valid_pixels'] = int(np.count_nonzero(found & valid))
                passes.append(stats)
                del found
    else:
        log.write_text(f'{METHOD}\nno finite pixels\n')
    # A unique temporary name prevents concurrent writers from removing one
    # another's .tmp.npz before publication.
    with tempfile.NamedTemporaryFile(prefix=output.stem + '.', suffix='.tmp.npz',
                                     dir=output.parent, delete=False) as stream:
        temp = Path(stream.name)
        np.savez_compressed(stream, background_mask=sky)
    try:
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    meta.write_text(json.dumps(dict(signature=signature, finite_pixels=int(valid.sum()),
                    background_pixels=int(sky.sum()), passes=passes), indent=2)+'\n')
    return sky


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Input FITS (SCI extension if present)')
    parser.add_argument('output', type=Path, help='NPZ output; background_mask=True means usable sky')
    parser.add_argument('--crop', nargs=4, type=int, metavar=('X0','Y0','WIDTH','HEIGHT'),
                        help='0-based crop of the input; output uses that exact grid')
    args = parser.parse_args(argv)
    with fits.open(args.input, memmap=False) as hd:
        h = hd['SCI'] if 'SCI' in hd else next(x for x in hd if x.header.get('NAXIS') == 2)
        if args.crop:
            x, y, w, height = args.crop
            if min(x, y) < 0 or min(w, height) <= 0 or x+w > h.shape[1] or y+height > h.shape[0]:
                parser.error('Crop outside FITS image')
            array = np.asarray(h.section[y:y+height, x:x+w], np.float32)
        else:
            array = np.asarray(h.data, np.float32)
        header = h.header.copy()
    aggressive_sextractor_background(array, header, args.output,
                                    source=f'{args.input.resolve()}#size={args.input.stat().st_size}#mtime={args.input.stat().st_mtime_ns}#crop={args.crop}')


if __name__ == '__main__':
    main()
