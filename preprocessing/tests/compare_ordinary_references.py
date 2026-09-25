"""Read-only real-catalog comparison against the approved diagnostic products.

Run as python -m preprocessing.tests.compare_ordinary_references. No FITS/REG
products are rewritten; JSON summaries are printed for review.
"""

import csv
import importlib.util
import json
import inspect
import sys
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from preprocessing.labels import SourceClass as C, SourceLabels
from preprocessing.ordinary_common import OrdinaryInput, classify_ordinary_sources
from preprocessing.ordinary_a2744 import A2744OrdinaryConfig
from preprocessing.meas_processing import classify_catalog_basics
from preprocessing.utils.geometry import EllipseGeometry

ROOT = Path('/home/czh23/analysis/2026-09')
A_REF = ROOT / '2026-09-08/a2744_iterative_b_filter_aggressive_star_snr'
C_REF = ROOT / '2026-09-05/jwst_pointing19_f444w_no_morph_snr_phot_containment'
CLASSES = {'drop': C.DROPPED, 'ignore': C.ORDINARY_IGNORE, 'clean': C.CLEAN,
           'weak_shape': C.WEAK_SHAPE, 'strict_center_only': C.STRICT_CENTER_ONLY}


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def numeric(rows, key):
    return np.array([float(row[key]) for row in rows])


def compare_a2744():
    ref = module(A_REF / 'a2744_sextractor_hsc_style_filter.py', 'a2744_reference')
    reports = []
    for path in sorted(A_REF.glob('*_source_stages.csv')):
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        geom = EllipseGeometry(*(numeric(rows, k) for k in ('x_cutout', 'y_cutout', 'a_pix', 'b_pix')),
                               np.deg2rad(numeric(rows, 'theta_deg')), numeric(rows, 'area_pix'))
        data = OrdinaryInput(geom, numeric(rows, 'mag_auto'), snr=numeric(rows, 'snr'),
                             flags=numeric(rows, 'flags').astype(int))
        band = path.name.split('_')[-3]
        with fits.open(ref.raw_fits_path(band), memmap=True) as hdus:
            h = next(h for h in hdus if h.header.get('NAXIS') == 2)
            scale = float(np.mean(proj_plane_pixel_scales(WCS(h.header).celestial)) * 3600)
        old_drop = np.array([r['A_drop'] == 'True' for r in rows])
        labels = SourceLabels.empty(len(rows))
        labels.assign(old_drop, C.DROPPED, 'reference_A')
        out = classify_ordinary_sources(data, ~old_drop, labels, dataset='a2744',
            config=A2744OrdinaryConfig(ref.PSF_FWHM_ARCSEC[band]/scale, image_shape=(4096, 4096)))
        report = {'dataset': path.stem, 'sources': len(rows)}
        for stage, key in [('close_pair_flags', 'post_ab'), ('containment', 'post_containment'), ('snr', 'post_snr')]:
            expected = np.array([CLASSES[r[key]] for r in rows])
            mismatch = out.stages[stage] != expected
            report[f'{stage}_differences'] = int(mismatch.sum())
            report[f'{stage}_different_ids'] = [int(rows[i]['number']) for i in np.flatnonzero(mismatch)][:20]
        # Single-variable control: the legacy inner loop only checks j, so an
        # ignored i can keep demoting subsequent neighbors in the same pass.
        cat = dict(FLAGS=data.flags, FLUX_AUTO=10**((27-data.mag)/2.5),
                   FLUXERR_AUTO=np.full(len(rows), np.nan), SNR_WIN=data.snr,
                   KRON_RADIUS=np.ones(len(rows)), A_IMAGE=geom.major,
                   B_IMAGE=geom.minor, THETA_IMAGE=np.rad2deg(geom.theta))
        sel = dict(idx=np.arange(len(rows)), x=geom.x, y=geom.y)
        source = inspect.getsource(ref.classify).replace(
            'if final[j] in {"ignore", "drop"}:',
            'if final[i] in {"ignore", "drop"} or final[j] in {"ignore", "drop"}:')
        scope = dict(vars(ref))
        exec(compile(source, '<reference-live-pairs-control>', 'exec'), scope)
        control = scope['classify'](cat, sel, band, scale)
        expected_control = np.array([CLASSES[v] for v in control['post_snr_label']])
        report['live_pair_control_differences'] = int(np.count_nonzero(out.labels.source_class != expected_control))
        assert report['live_pair_control_differences'] == 0, report
        shared_a = classify_catalog_basics(geom, data.mag, dataset='a2744', pixel_scale_arcsec=scale)
        report['shared_A_changes'] = int(np.count_nonzero((shared_a.a_large | shared_a.a_faint_large) != old_drop))
        reports.append(report)
        print(json.dumps(report), flush=True)
    return reports


def compare_cosmos():
    ref = module(C_REF / 'jwst_no_morph_snr_phot_containment_three_blocks.py', 'cosmos_reference')
    pipe = ref.load_pipeline()
    ref.set_context(pipe)
    cache = pipe.load_catalog()
    image, wcs, scale = pipe.image_hdu(pipe.raw_fits_path('f444w'))
    ny, nx = image.shape
    reports = []
    for name, (x0, y0) in {'left_top': (0, ny-4096), 'center': (nx//2-2048, ny//2-2048),
                          'top_col3': (8192, ny-4096)}.items():
        sel = pipe.select_region(cache, wcs, x0, y0)
        idx = sel['idx']
        geom = EllipseGeometry(sel['x'], sel['y'], cache['kron2_a'][idx]/scale,
                               cache['kron2_b'][idx]/scale, np.deg2rad(cache['theta_world'][idx]),
                               cache['kron2_area'][idx])
        data = OrdinaryInput(geom, cache['mag_auto_f444w'][idx], snr=cache['snr_f444w'][idx],
                             flags=cache['warn_flag'][idx], star_mask=cache['flag_star'][idx],
                             model_mag=cache['mag_model_f444w'][idx])
        nan_drop = pipe.nan_component_center_mask(image[y0:y0+4096, x0:x0+4096], sel['x'], sel['y'])
        from preprocessing.utils.segmentation import cosmos_fill_ratios
        data.segmentation_fill_ratio = cosmos_fill_ratios(geom, cache['id'][idx], wcs,
            '/data/shared/jwst_foundation/catalog/COSMOS_1727_1837_5893/COSMOSWeb_mastercatalog_v1.1.fits',
            selected=~nan_drop & (data.mag < 25.5), origin=(x0, y0))
        drop = (geom.area > pipe.AREA_IGNORE_MAX_PIX) | nan_drop
        base = dict(a_ignore=drop, stage1_ignore=~drop & np.isin(data.flags, pipe.WARN_IGNORE_VALUES),
                    mag_auto=data.mag, mag_model=data.model_mag, dmag=np.abs(data.mag-data.model_mag),
                    snr=data.snr, area=geom.area)
        old = ref.ordered_classify(pipe, cache, sel, base, scale)
        labels = SourceLabels.empty(len(idx))
        labels.assign(drop, C.DROPPED, 'reference_A')
        new = classify_ordinary_sources(data, ~drop, labels, dataset='cosmos')
        expected = np.array([CLASSES[v] for v in old['final']])
        shared_a = classify_catalog_basics(geom, data.mag, dataset='cosmos', pixel_scale_arcsec=scale)
        report = dict(dataset=f'cosmos_{name}', sources=len(idx),
                      ordinary_differences=int(np.count_nonzero(new.labels.source_class != expected)),
                      shared_A_changes=int(np.count_nonzero((shared_a.a_large | shared_a.a_faint_large) != (geom.area > pipe.AREA_IGNORE_MAX_PIX))))
        assert report['ordinary_differences'] == 0, report
        reports.append(report)
        print(json.dumps(report), flush=True)
    return reports


def compare_noise():
    from utils.source_snr import measure_local_aperture_snr
    ref = module(A_REF / 'a2744_sextractor_hsc_style_filter.py', 'a2744_noise_reference')
    image = np.random.default_rng(9).normal(0, .01, (192, 192)).astype(np.float32)
    image[80:100, 80:100] += 1
    excluded = np.zeros(image.shape, bool)
    expected = ref.local_aperture_snr_fields(image, excluded, excluded)
    actual = measure_local_aperture_snr(ref.mjysr_to_hsc_flux(image), [90], [90], excluded_mask=excluded)
    for name in ('background', 'aperture_sigma', 'sky_aperture_count'):
        np.testing.assert_array_equal(actual[name], expected[name])
    np.testing.assert_array_equal(actual['sky_mask'], expected['lsst_background'])
    print(json.dumps(dict(noise_reference='background, aperture sigma/count and sky mask exact')), flush=True)


def compare_bright():
    """Same post-ordinary inputs to isolate the A2744 bright implementation."""
    from preprocessing.bright_label_jwst import JWSTBrightConfig, label_bright_sources
    ref = module(A_REF / 'a2744_sextractor_hsc_style_filter.py', 'a2744_bright_reference')
    original_gaia = ref.load_gaia()
    for path in sorted(A_REF.glob('*_source_stages.csv')):
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        geom = EllipseGeometry(*(numeric(rows, k) for k in ('x_cutout', 'y_cutout', 'a_pix', 'b_pix')),
                               np.deg2rad(numeric(rows, 'theta_deg')), numeric(rows, 'area_pix'))
        band = path.name.split('_')[-3]
        info = ref.image_hdu(ref.raw_fits_path(band))
        px, py = info.wcs.world_to_pixel_values(numeric(rows, 'ra')[:1], numeric(rows, 'dec')[:1])
        x0, y0 = int(round(px[0]-geom.x[0])), int(round(py[0]-geom.y[0]))
        gs = ref.select_gaia(original_gaia, info, x0, y0, obstime=ref.observation_time(info.path))
        gaia = [dict(x=float(x), y=float(y), source_id=i+1, phot_g_mean_mag=float(m))
                for i, (x, y, m) in enumerate(zip(gs['x'], gs['y'], gs['mag']))]
        state = dict(final=np.array([r['post_aperture_snr'] for r in rows], object),
                     mag=numeric(rows, 'mag_auto'), area=geom.area, a=geom.major, b=geom.minor,
                     theta=np.rad2deg(geom.theta), flags=numeric(rows, 'flags').astype(int), containment_pairs=0)
        for key in ('high_iou_ignore', 'containment_weak', 'containment_strict', 'containment_ignore'):
            state[key] = np.zeros(len(rows), bool)
        sel = dict(x=geom.x, y=geom.y)
        ref.apply_source_containment_dedup(sel, state, prefix='control')
        initial = state['final'].copy()
        mask_path = path.with_name(path.name.replace('_source_stages.csv', '_stage_masks.npz'))
        with np.load(mask_path) as masks:
            bright = masks['bright'].astype(bool)
        inserted, occupied, _ = ref.apply_bright_clusters(bright, sel, state, gs, pixscale=info.pixscale)
        more, big, _ = ref.bright_refit(bright, sel, state['final'], info.pixscale, gs, occupied)
        expected = sorted((round(float(p['x']), 5), round(float(p['y']), 5)) for p in inserted + more)
        initial_labels = SourceLabels.empty(len(rows))
        initial_labels.source_class[:] = [CLASSES[v] for v in initial]
        initial_labels.reason[:] = 'reference_ordinary_done'
        data = OrdinaryInput(geom, state['mag'], flags=state['flags'])
        out = label_bright_sources(data, initial_labels, image_shape=bright.shape,
                                  bright_region=bright, gaia_rows=gaia,
                                  config=JWSTBrightConfig('a2744', info.pixscale))
        actual = sorted((round(float(x), 5), round(float(y), 5)) for x, y in zip(out.strict_center_x, out.strict_center_y))
        differences = int(np.count_nonzero(out.labels.source_class != [CLASSES[v] for v in state['final']]))
        report = dict(bright_reference=path.stem, sources=len(rows), label_differences=differences,
                      expected_insertions=len(expected), actual_insertions=len(actual), centers_equal=expected == actual)
        print(json.dumps(report), flush=True)
        assert differences == 0 and expected == actual, report


def compare_cosmos_bright():
    """Reuse the prior same-policy Gaia control with its propagated positions."""
    from astropy.table import Table
    from preprocessing.bright_label_jwst import JWSTBrightConfig, label_bright_sources
    from preprocessing.utils.gaia import astrometric_evidence
    root = ROOT / '2026-09-08/shared_a_gaia_lupton_comparison'
    with (root / 'cosmos19_f444w/sources.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    with (root / 'cosmos19_f444w_same_gaia_policy/gaia_decisions.csv').open() as handle:
        decisions = list(csv.DictReader(handle))
    gaia_table = Table.read('/home/czh23/CELLECT/output/gaia_dr3_cosmos_full.fits')
    evidence = dict(zip(map(int, gaia_table['source_id']), astrometric_evidence(gaia_table)))
    gaia = [dict(x=float(r['x']), y=float(r['y']), source_id=int(r['gaia_source_id']),
                 phot_g_mean_mag=float(r['G']), astrometric_evidence=bool(evidence[int(r['gaia_source_id'])]))
            for r in decisions]
    path = '/data/shared/jwst_foundation/raw/COSMOS_1727_1837_5893/Pointing_0019/COSMOS_pointing_0019_F444W_detector_p001.fits'
    with fits.open(path, memmap=True) as hdus:
        h = next(h for h in hdus if h.header.get('NAXIS') == 2)
        shape = h.shape
        scale = float(np.mean(proj_plane_pixel_scales(WCS(h.header).celestial)) * 3600)
    n = len(rows)
    geom = EllipseGeometry(numeric(rows, 'x'), numeric(rows, 'y'), np.ones(n), np.ones(n), np.zeros(n), np.ones(n))
    initial = SourceLabels.empty(n)
    initial.source_class[:] = [CLASSES[r['reference_label']] for r in rows]
    initial.reason[:] = 'reference_ordinary_done'
    data = OrdinaryInput(geom, numeric(rows, 'mag'))
    out = label_bright_sources(data, initial, image_shape=shape, gaia_rows=gaia,
                              config=JWSTBrightConfig('cosmos', scale, .145))
    inserted = {int(r['gaia_source_id']) for r in out.cluster_rows if r['inserted']}
    accepted = {int(r['gaia_source_id']) for r in out.cluster_rows if r['accepted_kron']}
    expected = {int(r['gaia_source_id']) for r in decisions if r['insert'] == 'True'}
    expected_accepted = {int(r['gaia_source_id']) for r in decisions if r['accepted_kron'] == 'True'}
    assert inserted == expected and accepted == expected_accepted
    np.testing.assert_array_equal(out.labels.source_class, [CLASSES[r['reference_label']] for r in rows])
    print(json.dumps(dict(cosmos_bright_sources=n, inserted=len(inserted), accepted=len(accepted),
                          insertion_id_differences=0, ordinary_label_differences=0)), flush=True)


if __name__ == '__main__':
    compare_noise()
    compare_a2744()
    compare_cosmos()
