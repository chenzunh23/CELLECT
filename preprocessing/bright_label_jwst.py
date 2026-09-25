"""JWST bright postprocessing, after ordinary filtering (not a parallel branch).

COSMOS uses Gaia-centered catalog neighborhoods, A2744 uses bright components.
Inputs use the same zero-based image grid as OrdinaryInput; outputs retain the
HSC BrightLabelResult contract, including separate synthetic-center arrays.
"""

from dataclasses import dataclass
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .bright_label import (BrightLabelResult, project_gaia_rows, synthetic_gaia_source,
                          synthetic_component_center_source, apply_bright_rows_to_labels,
                          component_touches_image_boundary)
from .labels import SourceClass as C
from .ordinary_common import DatasetOrdinaryResult, retained, downgrade
from .utils.gaia import astrometric_evidence
from .utils.geometry import component_at, UnionFind
from .utils.containment import nearby_pairs, pixel_iou


@dataclass(frozen=True)
class JWSTBrightConfig:
    mode: str
    pixel_scale_arcsec: float
    psf_fwhm_arcsec: float | None = None
    gaia_mag_max: float = 22.0
    gaia_reference_epoch: float | None = None
    cosmos_match_arcsec: float = 1.0
    source_mag_max: float = 22.0
    component_search_radius: int = 5
    cluster_iou_threshold: float = 1.0 / 3.0
    cluster_max_area_hsc_pixels: float = 10000.0
    cluster_max_distance_hsc_pixels: float = 50.0
    source_match_arcsec: float = 6 * .168
    centroid_match_arcsec: float = 10 * .168
    fast_component_source_min: int = 256
    bright_area_min: int = 1000
    reject_boundary_components: bool = False


def _gaia_rows(table, rows, header, shape, config):
    if rows is not None and table is not None:
        raise ValueError('provide gaia_table or gaia_rows, not both')
    if rows is None:
        if table is not None:
            table = table.copy()
            if 'ref_epoch' not in table.colnames and config.gaia_reference_epoch is not None:
                table['ref_epoch'] = np.full(len(table), config.gaia_reference_epoch)
            evidence = astrometric_evidence(table)
            ids = table['source_id'] if 'source_id' in table.colnames else np.arange(len(table))
            evidence_by_id = {int(sid): bool(value) for sid, value in zip(ids, evidence)}
        else:
            evidence_by_id = {}
        rows = project_gaia_rows(table, image_shape=shape, image_header=header, pixel_origin=0)
        for row in rows:
            row['astrometric_evidence'] = evidence_by_id.get(int(row['source_id']), False)
    unique = {}
    for row in rows:
        x, y, mag = float(row['x']), float(row['y']), float(row['phot_g_mean_mag'])
        magnitude_ok = mag <= config.gaia_mag_max if config.mode == 'a2744' else mag < config.gaia_mag_max
        if np.isfinite(x+y+mag) and 0 <= x < shape[1] and 0 <= y < shape[0] and magnitude_ok:
            unique.setdefault(int(row['source_id']), dict(row))
    return list(unique.values())


def _clusters(data, indices, shape, config):
    geom = data.geometry
    mask = np.zeros(len(data.mag), bool)
    mask[indices] = True
    uf = UnionFind(list(indices))
    area_max = config.cluster_max_area_hsc_pixels * (.168 / config.pixel_scale_arcsec)**2
    distance = config.cluster_max_distance_hsc_pixels * .168 / config.pixel_scale_arcsec
    for i, j in nearby_pairs(geom, mask):
        if max(geom.area[i], geom.area[j]) >= area_max:
            continue
        if np.hypot(geom.x[i]-geom.x[j], geom.y[i]-geom.y[j]) > distance:
            continue
        if min(geom.area[i], geom.area[j]) / max(geom.area[i], geom.area[j]) < config.cluster_iou_threshold:
            continue
        if pixel_iou(geom, i, j, shape) >= config.cluster_iou_threshold:
            uf.union(i, j)
    groups = {}
    for i in indices:
        groups.setdefault(uf.find(i), []).append(int(i))
    return sorted(groups.values(), key=min)


def label_bright_sources(data, labels, *, image_shape, config: JWSTBrightConfig,
                         bright_region=None, component_labels=None, gaia_table=None,
                         gaia_rows=None, image_header=None, source_ids=None) -> BrightLabelResult:
    """Consume completed ordinary labels; ignored/dropped rows are never revived.

    COSMOS matches all catalog centers, including ignored rows, as in the
    reference policy. Accepting a match preserves its ordinary label. Supplied
    Gaia rows are already projected and may include astrometric_evidence.
    """
    if config.mode not in ('cosmos', 'a2744'):
        raise ValueError('JWST bright mode must be cosmos or a2744')
    if not np.isfinite(config.pixel_scale_arcsec) or config.pixel_scale_arcsec <= 0:
        raise ValueError('pixel scale must be positive and finite')
    if len(image_shape) != 2 or min(image_shape) <= 0:
        raise ValueError('image_shape must be a positive 2-D shape')
    n, geom = len(data.mag), data.geometry
    if labels.source_class.shape != (n,) or np.any(labels.reason == 'unassigned'):
        raise ValueError('JWST bright processing requires completed ordinary labels')
    ids = np.arange(n, dtype=np.int64) if source_ids is None else np.asarray(source_ids, dtype=np.int64)
    if ids.shape != (n,):
        raise ValueError('source_ids length mismatch')
    gaia = _gaia_rows(gaia_table, gaia_rows, image_header, image_shape, config)
    ordinary = DatasetOrdinaryResult(labels, np.ones(n, bool))
    added, cluster_rows, meta = [], [], {}
    seen_gaia = set()
    comp = np.zeros(n, dtype=np.int32)
    areas = {}
    ignore_components, bright_components = set(), set()

    def add_gaia(row, component=0, cluster=0, size=1, reason='jwst_gaia_center',
                 match_arcsec=0.0, match_mode='component'):
        sid = int(row['source_id'])
        if sid in seen_gaia:
            return
        seen_gaia.add(sid)
        added.append(synthetic_gaia_source(row, comp=component, component_area=areas.get(component, 0),
            cluster_id=cluster, cluster_size=size, source_id_offset=1,
            pixel_scale_arcsec=config.pixel_scale_arcsec, match_mode=match_mode,
            match_pixels=match_arcsec/config.pixel_scale_arcsec, reason=reason))

    if config.mode == 'cosmos':
        if config.psf_fwhm_arcsec is None or not np.isfinite(config.psf_fwhm_arcsec) or config.psf_fwhm_arcsec <= 0:
            raise ValueError('COSMOS requires a positive band PSF FWHM in arcsec')
        valid = np.isfinite(geom.x) & np.isfinite(geom.y)
        valid &= (geom.x >= 0) & (geom.x < image_shape[1]) & (geom.y >= 0) & (geom.y < image_shape[0])
        idx = np.flatnonzero(valid)
        xy = np.column_stack((geom.x[idx], geom.y[idx])) * config.pixel_scale_arcsec
        tree = cKDTree(xy) if len(idx) else None
        for row in gaia:
            point = np.array([row['x'], row['y']]) * config.pixel_scale_arcsec
            matches = [] if tree is None else tree.query_ball_point(point, config.cosmos_match_arcsec)
            evidence = bool(row.get('astrometric_evidence', False))
            accept = evidence and len(matches) == 1 and np.linalg.norm(xy[matches[0]]-point) < config.psf_fwhm_arcsec
            insert = not accept and (evidence or not matches)
            if insert:
                add_gaia(row, reason='cosmos_gaia_neighborhood_center')
            cluster_rows.append(dict(gaia_source_id=int(row['source_id']),
                source_indices=[int(idx[j]) for j in matches], accepted_kron=accept,
                inserted=insert, astrometric_evidence=evidence))
    else:
        if data.flags is None:
            raise ValueError('A2744 requires SExtractor FLAGS')
        if component_labels is None:
            if bright_region is None or np.shape(bright_region) != tuple(image_shape):
                raise ValueError('A2744 requires a bright mask on the image grid')
            component_labels, _ = ndimage.label(np.asarray(bright_region, bool))
        grid = np.asarray(component_labels)
        if grid.shape != tuple(image_shape) or not np.issubdtype(grid.dtype, np.integer) or np.any(grid < 0):
            raise ValueError('component labels must be nonnegative integers on the image grid')
        if bright_region is not None and not np.array_equal(grid > 0, np.asarray(bright_region, bool)):
            raise ValueError('bright mask and component labels disagree')
        values, counts = np.unique(grid[grid > 0], return_counts=True)
        areas = {int(k): int(v) for k, v in zip(values, counts)}
        for key, area in areas.items():
            boundary = component_touches_image_boundary(grid, key)
            keep = area >= config.bright_area_min and not (config.reject_boundary_components and boundary)
            (bright_components if keep else ignore_components).add(key)
            meta[key] = dict(area=area, touches_boundary=boundary)
        for i in np.flatnonzero(np.isfinite(geom.x) & np.isfinite(geom.y)):
            comp[i] = component_at(grid, geom.x[i], geom.y[i], config.component_search_radius)
        by_gaia = {}
        for row in gaia:
            key = component_at(grid, row['x'], row['y'], 0)
            if key > 0:
                by_gaia.setdefault(key, []).append(row)
        occupied = set()
        for key in sorted(set(comp.tolist())):
            if key > 0 and config.reject_boundary_components and meta[key]['touches_boundary']:
                continue
            indices = np.flatnonzero(retained(ordinary) & (comp == key) & np.isfinite(data.mag)
                                     & (data.mag <= config.source_mag_max))
            if not len(indices):
                continue
            if key == 0 or len(indices) == 1:
                for i in indices:
                    desired = {0: C.CLEAN, 2: C.WEAK_SHAPE}.get(int(data.flags[i]), C.ORDINARY_IGNORE)
                    mask = np.zeros(n, bool); mask[i] = True
                    downgrade(ordinary, mask, desired, 'a2744_isolated_bright_flag')
                    cluster_rows.append(dict(component_id=int(key), source_indices=[int(i)], isolated=True))
                    if key > 0 and retained(ordinary)[i]:
                        occupied.add(key)
                continue
            groups = [indices.tolist()] if len(indices) >= config.fast_component_source_min else _clusters(data, indices, image_shape, config)
            local_gaia = by_gaia.get(key, [])
            for group in groups:
                cluster_id = len(cluster_rows) + 1
                matches = []
                cx, cy = np.mean(geom.x[group]), np.mean(geom.y[group])
                for row in local_gaia:
                    if int(row['source_id']) in seen_gaia:
                        continue
                    source_dist = np.min(np.hypot(geom.x[group]-row['x'], geom.y[group]-row['y'])) * config.pixel_scale_arcsec
                    center_dist = np.hypot(cx-row['x'], cy-row['y']) * config.pixel_scale_arcsec
                    if source_dist <= config.source_match_arcsec or center_dist <= config.centroid_match_arcsec:
                        matches.append((source_dist if source_dist <= config.source_match_arcsec else center_dist,
                                        row['phot_g_mean_mag'], row,
                                        'source_center' if source_dist <= config.source_match_arcsec else 'cluster_centroid'))
                best = min(matches, key=lambda v: v[:2]) if matches else None
                matched = best[2] if best is not None else None
                if matched is not None:
                    add_gaia(matched, key, cluster_id, len(group), 'a2744_cluster_gaia_center',
                             match_arcsec=best[0], match_mode=best[3])
                    occupied.add(key)
                mask = np.zeros(n, bool); mask[group] = True
                downgrade(ordinary, mask, C.ORDINARY_IGNORE, 'a2744_cluster_gaia_replaced' if matched else 'a2744_cluster_no_gaia')
                cluster_rows.append(dict(component_id=int(key), source_indices=group, isolated=False,
                                          gaia_source_id=None if matched is None else int(matched['source_id'])))
            for row in local_gaia:
                add_gaia(row, key, reason='a2744_component_unmatched_gaia')
                occupied.add(key)
        for i in np.flatnonzero(retained(ordinary)):
            occupied.add(component_at(grid, geom.x[i], geom.y[i], 0))
        for key in sorted(bright_components - occupied):
            if by_gaia.get(key):
                row = min(by_gaia[key], key=lambda v: v['phot_g_mean_mag'])
                add_gaia(row, key, reason='a2744_empty_component_gaia')
            else:
                yy, xx = np.nonzero(grid == key)
                added.append(synthetic_component_center_source(comp=key, component_area=areas[key],
                    x=float(xx.mean()), y=float(yy.mean()), reason='a2744_empty_component_geometry'))

    names = {C.CLEAN: 'clean', C.WEAK_SHAPE: 'weak_shape', C.STRICT_CENTER_ONLY: 'strict_center_only',
             C.ORDINARY_IGNORE: 'ignore', C.STRICT_IGNORE: 'strict_ignore', C.RESTRICTED_BRIGHT_REGION: 'restricted_bright_region'}
    rows = []
    for i in range(n):
        if labels.source_class[i] == C.DROPPED:
            continue
        rows.append(dict(source_id=int(ids[i]), table_index=i, row_index=i,
            x=float(geom.x[i]), y=float(geom.y[i]), output_x=float(geom.x[i]), output_y=float(geom.y[i]),
            major=float(geom.major[i]), minor=float(geom.minor[i]), theta_deg=float(np.rad2deg(geom.theta[i])),
            area=float(geom.area[i]), mag=float(data.mag[i]), component_id=int(comp[i]), component_area=areas.get(int(comp[i]), 0),
            final_label=names[C(labels.source_class[i])], reason=str(labels.reason[i])))
    rows.extend(added)
    x, y, sid, reason, cid, restricted, _, ignore_sources = apply_bright_rows_to_labels(rows, labels, n_table_rows=n)
    return BrightLabelResult(labels, x, y, sid, reason, cid, restricted,
        np.asarray(sorted(bright_components), np.int32), np.asarray(sorted(ignore_components), np.int32),
        ignore_sources, rows, meta, cluster_rows)
