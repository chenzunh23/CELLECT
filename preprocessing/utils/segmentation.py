"""Native segmentation coverage for WCS-transformed Kron ellipses."""
import numpy as np
from dataclasses import dataclass
from scipy import ndimage
from preprocessing.labels import SourceClass


def read_cosmos_segment_cutout(wcs, shape, catalog_path, *, origin=(0, 0), chunk_rows=128):
    """Nearest-neighbor WCS sampling; segment values become unique catalog IDs.

    Unknown IDs, conflicting overlap and missing coverage are invalid for supervision.
    """
    from pathlib import Path
    from astropy.io import fits
    from astropy.wcs import WCS
    catalog_path = Path(catalog_path)
    h, w = shape
    result = np.zeros(shape, np.int32)
    stars = np.zeros(shape, bool)
    coverage = np.zeros(shape, bool)
    star_coverage = np.zeros(shape, bool)
    ambiguous = np.zeros(shape, bool)
    with fits.open(catalog_path, memmap=True) as hs:
        table = hs[1].data
        tiles = np.asarray(table['tile']).astype(str)
        ids = np.asarray(table['id'])
        segids = np.asarray(table['segment-id'])
    corners_x = np.array([0,w-1,w-1,0])+origin[0]
    corners_y = np.array([0,0,h-1,h-1])+origin[1]
    ra, dec = wcs.all_pix2world(corners_x,corners_y,0)
    for tile in np.unique(tiles):
        path = catalog_path.parent/'segmentation_maps'/f'detection_chi2pos_SWLW_{tile}_segmap_v1.3.fits'
        with fits.open(path, memmap=True, do_not_scale_image_data=True) as hs:
            sh = next(hh for hh in hs if hh.header.get('NAXIS') == 2)
            sw = WCS(sh.header).celestial
            sx,sy = sw.all_world2pix(ra,dec,0)
            if sx.max()<0 or sy.max()<0 or sx.min()>=sh.shape[1] or sy.min()>=sh.shape[0]:
                continue
            use = tiles == tile
            lookup = np.zeros(int(segids[use].max())+1,np.int32)
            lookup[segids[use].astype(int)] = ids[use]
            star_path = catalog_path.parent/'star_masks'/f'cosmos_web_starmask_jwst_{tile}.fits'
            with fits.open(star_path,memmap=True,do_not_scale_image_data=True) as mh:
                sm = next(hh for hh in mh if hh.header.get('NAXIS') == 2)
                mw = WCS(sm.header).celestial
                for y0 in range(0,h,chunk_rows):
                    y1=min(h,y0+chunk_rows)
                    yy,xx=np.mgrid[y0:y1,:w]
                    rr,dd=wcs.all_pix2world(xx+origin[0],yy+origin[1],0)
                    px,py=sw.all_world2pix(rr,dd,0)
                    xi,yi=np.rint(px).astype(int),np.rint(py).astype(int)
                    good=(xi>=0)&(yi>=0)&(xi<sh.shape[1])&(yi<sh.shape[0])
                    values=np.zeros(xx.shape,np.int64)
                    values[good]=sh.data[yi[good],xi[good]].astype(np.int64)*int(sh.header.get('BSCALE',1))+int(sh.header.get('BZERO',0))
                    mapped=np.zeros(xx.shape,np.int32)
                    known=good&(values>=0)&(values<len(lookup))
                    mapped[known]=lookup[values[known]]
                    existing=result[y0:y1]
                    ambiguous[y0:y1] |= good & (((values>0)&(mapped==0)) | ((existing>0)&(mapped>0)&(existing!=mapped)))
                    existing[(existing==0)&(mapped>0)] = mapped[(existing==0)&(mapped>0)]
                    coverage[y0:y1] |= good
                    px,py=mw.all_world2pix(rr,dd,0)
                    xi,yi=np.rint(px).astype(int),np.rint(py).astype(int)
                    good=(xi>=0)&(yi>=0)&(xi<sm.shape[1])&(yi<sm.shape[0])
                    stars[y0:y1][good] |= (sm.data[yi[good],xi[good]].astype(np.int64)*int(sm.header.get('BSCALE',1))+int(sm.header.get('BZERO',0)))>0
                    star_coverage[y0:y1] |= good
    return result, stars, coverage & star_coverage & ~ambiguous


def measure(raw, header, cx, cy, a, b, theta, segment_id):
    hx,hy=np.hypot(a*np.cos(theta),b*np.sin(theta)),np.hypot(a*np.sin(theta),b*np.cos(theta))
    x0,x1=int(np.floor(cx-hx)),int(np.ceil(cx+hx))+1
    y0,y1=int(np.floor(cy-hy)),int(np.ceil(cy+hy))+1
    yy,xx=np.ogrid[y0:y1,x0:x1]
    u=(xx-cx)*np.cos(theta)+(yy-cy)*np.sin(theta)
    v=-(xx-cx)*np.sin(theta)+(yy-cy)*np.cos(theta)
    ellipse=(u/a)**2+(v/b)**2<=1
    denominator=int(ellipse.sum())
    xa,ya,xb,yb=max(x0,0),max(y0,0),min(x1,raw.shape[1]),min(y1,raw.shape[0])
    result=dict(ellipse_pixels=denominator,seg_covered_pixels=0,own_pixels=0,any_pixels=0,
                own_fill=np.nan,any_fill=np.nan,own_fill_analytic=np.nan,coverage_fraction=0.,center_segment=-1)
    if denominator==0 or xa>=xb or ya>=yb: return result
    mask=ellipse[ya-y0:yb-y0,xa-x0:xb-x0]
    values=raw[ya:yb,xa:xb]
    valid=mask.copy()
    if 'BLANK' in header: valid &= values != header['BLANK']
    # FITS signed storage of uint32 segment IDs: decode only the small aperture.
    decoded=values.astype(np.int64)*int(header.get('BSCALE',1))+int(header.get('BZERO',0))
    covered=int(valid.sum());own=int(np.count_nonzero(valid&(decoded==segment_id)))
    any_count=int(np.count_nonzero(valid&(decoded>0)))
    ix,iy=int(np.floor(cx+.5)),int(np.floor(cy+.5))
    if 0<=ix<raw.shape[1] and 0<=iy<raw.shape[0]:
        result['center_segment']=int(raw[iy,ix])*int(header.get('BSCALE',1))+int(header.get('BZERO',0))
    result.update(seg_covered_pixels=covered,own_pixels=own,any_pixels=any_count,
        coverage_fraction=covered/denominator)
    # Missing map coverage is not a low-fill source: omit from distributions.
    if covered==denominator and segment_id>0:
        result.update(own_fill=own/denominator,any_fill=any_count/denominator,
                      own_fill_analytic=own/(np.pi*a*b))
    return result


@dataclass
class SegmentationTargets:
    # Compatibility preview: smaller instances overwrite enclosing ones.
    instance_ids: np.ndarray
    positive_weight: np.ndarray
    sources: list[dict]
    overlap_masks: list[dict]


SEGMENTATION_POLICY_VERSION = 'raw_nested_fill_gaussian_overlap_v3'


def _main_component(mask):
    components, n = ndimage.label(mask, np.ones((3, 3), bool))
    counts = np.bincount(components.ravel()); counts[0] = 0
    return components == int(counts.argmax()), int(n)


def _fill_mask(mask):
    return ndimage.binary_fill_holes(mask, structure=np.ones((3, 3), bool))


def isolated_segmentation_targets(segmentation, source_ids, labels, *, star_mask=None,
                                  valid_mask=None, clearance=0, weight=0.25,
                                  allow_nested=True, main_fraction_min=0.95,
                                  closing_radius=0, gaussian_sigma=1.0,
                                  gaussian_min_raw_area=100):
    """Raw geometry defines area/nesting; regularized masks define overlap.

    All known non-dropped IDs (including ignore/center-only) undergo the same
    dominant-component, fill, and optional Gaussian treatment before collision
    checks. Unknown/disconnected IDs remain conservative raw blockers. At zero
    clearance touching masks are permitted; shared pixels between nonrelatives
    reject both. Raw ancestors/descendants may overlap, but every child must
    train for its parent to train. Processing is per padded bounding box.
    """
    from collections import defaultdict
    original = np.asarray(segmentation)
    if original.ndim != 2 or not np.issubdtype(original.dtype, np.integer) or np.any(original < 0):
        raise ValueError('segmentation must be a nonnegative integer image')
    if (clearance < 0 or int(clearance) != clearance or not 0 < weight <= 1
            or not 0 < main_fraction_min <= 1 or closing_radius < 0
            or int(closing_radius) != closing_radius
            or not np.isfinite(gaussian_sigma) or gaussian_sigma < 0
            or gaussian_min_raw_area < 0 or int(gaussian_min_raw_area) != gaussian_min_raw_area):
        raise ValueError('invalid segmentation policy parameter')
    clearance, closing_radius = int(clearance), int(closing_radius)
    ids, classes = np.asarray(source_ids), np.asarray(labels.source_class)
    if ids.shape != classes.shape or ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError('source IDs/classes must be aligned and unique')
    valid = np.ones(original.shape, bool) if valid_mask is None else np.asarray(valid_mask, bool)
    star = np.zeros(original.shape, bool) if star_mask is None else np.asarray(star_mask, bool)
    if valid.shape != original.shape or star.shape != original.shape:
        raise ValueError('mask shape mismatch')
    star = ndimage.binary_fill_holes(star)
    allowed = set(ids[np.isin(classes, [SourceClass.CLEAN, SourceClass.WEAK_SHAPE])].tolist())
    known = set(map(int, ids))
    seg = original.copy()
    seg[np.isin(seg, ids[classes == SourceClass.DROPPED])] = 0
    present, inverse = np.unique(seg, return_inverse=True)
    compact = inverse.reshape(seg.shape).astype(np.int32)+1
    compact[seg == 0] = 0
    boxes = ndimage.find_objects(compact)
    boxes = {int(present[i-1]): box for i, box in enumerate(boxes, 1) if box is not None}
    del compact, inverse
    # Freeze raw pixel areas and strict containment BEFORE any cleanup/filling.
    areas = {sid: int(np.count_nonzero(seg[box] == sid)) for sid, box in boxes.items()}
    contains, ancestors = {}, {}
    for sid, box in boxes.items():
        own = seg[box] == sid
        filled = _fill_mask(np.pad(own, 1))[1:-1, 1:-1]
        children, counts = np.unique(seg[box][filled & ~own], return_counts=True)
        contains[sid] = ({int(c) for c, n in zip(children, counts)
                          if c > 0 and int(n) == areas[int(c)]} if allow_nested else set())
        for child in contains[sid]:
            ancestors.setdefault(child, set()).add(sid)
    h, w = seg.shape
    info, processed, unsafe = {}, {}, set()
    padding = max(1, int(np.ceil(4*gaussian_sigma))+1, closing_radius+1)
    yy, xx = np.mgrid[-clearance:clearance+1, -clearance:clearance+1]
    disk = xx*xx+yy*yy <= clearance*clearance
    for sid, (ys, xs) in boxes.items():
        raw = seg[ys, xs] == sid
        main, n = _main_component(raw)
        fraction = float(main.sum()/areas[sid])
        connected = n == 1 or (sid in known and fraction >= main_fraction_min)
        can_process = sid in known and connected
        mask = np.pad(main if can_process else raw, padding)
        cleaned_area = int(mask.sum())
        smoothed = can_process and areas[sid] > gaussian_min_raw_area and gaussian_sigma > 0
        if can_process:
            mask = _fill_mask(mask)
            if smoothed:
                mask = ndimage.gaussian_filter(mask.astype(np.float32), gaussian_sigma,
                            mode='constant', cval=0, truncate=4) >= .5
            if closing_radius:
                cy, cx = np.mgrid[-closing_radius:closing_radius+1, -closing_radius:closing_radius+1]
                mask = ndimage.binary_closing(mask, structure=cx*cx+cy*cy <= closing_radius**2)
            if mask.any():
                mask = _fill_mask(_main_component(mask)[0])
        info[sid] = dict(connected=connected, raw_area=areas[sid],
                         main_fraction=fraction, raw_components=n,
                         cleaned_area=cleaned_area, gaussian_applied=bool(smoothed))
        if not mask.any():
            unsafe.add(sid)
            # A failed/empty target must not erase the source as a blocker.
            mask = np.pad(raw, padding)
        py, px = np.nonzero(mask)
        ya, yb, xa, xb = int(py.min()), int(py.max())+1, int(px.min()), int(px.max())+1
        mask = mask[ya:yb, xa:xb]
        y0, x0 = ys.start-padding+ya, xs.start-padding+xa
        processed[sid] = dict(source_id=sid, x0=x0, y0=y0, mask=mask, weight=weight)
        # Retain protection against truncated source masks at the parent-image edge.
        if ys.start == 0 or xs.start == 0 or ys.stop == h or xs.stop == w:
            unsafe.add(sid)
        gy0, gx0 = y0-clearance, x0-clearance
        gy1, gx1 = y0+mask.shape[0]+clearance, x0+mask.shape[1]+clearance
        if min(gy0, gx0) < 0 or gy1 > h or gx1 > w:
            unsafe.add(sid)
        else:
            grown = (ndimage.binary_dilation(np.pad(mask, clearance), structure=disk)
                     if clearance else mask)
            if np.any(grown & (star[gy0:gy1, gx0:gx1] | ~valid[gy0:gy1, gx0:gx1])):
                unsafe.add(sid)

    def intersection(a, b):
        ya, xa = max(a['y0'], b['y0']), max(a['x0'], b['x0'])
        yb = min(a['y0']+a['mask'].shape[0], b['y0']+b['mask'].shape[0])
        xb = min(a['x0']+a['mask'].shape[1], b['x0']+b['mask'].shape[1])
        if ya >= yb or xa >= xb:
            return 0
        return int(np.count_nonzero(
            a['mask'][ya-a['y0']:yb-a['y0'], xa-a['x0']:xb-a['x0']] &
            b['mask'][ya-b['y0']:yb-b['y0'], xa-b['x0']:xb-b['x0']]))

    # Sparse bbox grid avoids O(N^2) all-pairs checks and a full-image mask per ID.
    grid, conflicts = defaultdict(list), {sid: set() for sid in boxes}
    cell = 128
    for sid, item in processed.items():
        y0, x0 = item['y0']-clearance, item['x0']-clearance
        y1 = item['y0']+item['mask'].shape[0]+clearance
        x1 = item['x0']+item['mask'].shape[1]+clearance
        cells = [(y, x) for y in range(y0//cell, (y1-1)//cell+1)
                          for x in range(x0//cell, (x1-1)//cell+1)]
        candidates = {other for key in cells for other in grid[key]}
        compare = item
        if clearance:
            compare = dict(item, y0=y0, x0=x0,
                mask=ndimage.binary_dilation(np.pad(item['mask'], clearance), structure=disk))
        for other in candidates - contains[sid] - ancestors.get(sid, set()):
            if intersection(compare, processed[other]):
                conflicts[sid].add(other); conflicts[other].add(sid)
        for key in cells:
            grid[key].append(sid)
    # Process descendants before parents even for thin rings with less raw area.
    accepted, masks, records = set(), [], []
    for sid in sorted(boxes, key=lambda k: (-len(ancestors.get(k, set())), k)):
        if (sid not in allowed or not info[sid]['connected'] or sid in unsafe
                or conflicts[sid] or not contains[sid].issubset(accepted)):
            continue
        item = processed[sid]
        if any(intersection(item, processed[c]) != int(processed[c]['mask'].sum())
               for c in contains[sid]):
            continue  # Smoothing cannot manufacture or violate raw containment.
        parents = ancestors.get(sid, set())
        parent = max(parents, key=lambda p: (len(ancestors.get(p, set())), -p)) if parents else None
        root = min(parents, key=lambda p: (len(ancestors.get(p, set())), p)) if parents else sid
        item = dict(item, parent_source_id=parent)
        masks.append(item); accepted.add(sid)
        py, px = np.nonzero(item['mask'])
        k = int(np.argmin((px-px.mean())**2+(py-py.mean())**2))
        records.append(dict(source_id=sid, area=areas[sid], mask_area=int(item['mask'].sum()),
            x=int(px[k]+item['x0']), y=int(py[k]+item['y0']),
            mask_type='nested_isolated' if contains[sid] or parents else 'isolated',
            isolation_group_id=root, parent_source_id=parent,
            enclosed_source_ids=sorted(contains[sid]), **info[sid]))
    result = np.zeros(seg.shape, np.int32)
    # Raw ancestry controls overwrite order; large raw area need not mean parent.
    for item in sorted(masks, key=lambda m: (len(ancestors.get(m['source_id'], set())), m['source_id'])):
        y, x = item['y0'], item['x0']; mh, mw = item['mask'].shape
        result[y:y+mh, x:x+mw][item['mask']] = item['source_id']
    overlap_masks = []
    for item in masks:
        y, x = item['y0'], item['x0']; mh, mw = item['mask'].shape
        if np.any(item['mask'] & (result[y:y+mh, x:x+mw] != item['source_id'])):
            overlap_masks.append(item)
    return SegmentationTargets(result, np.where(result>0, weight, 0).astype(np.float32),
                               sorted(records, key=lambda r: r['source_id']),
                               sorted(overlap_masks, key=lambda r: r['source_id']))


def transform_ellipses(wcs, segwcs, x, y, a, b, theta):
    px=np.stack([x,x+.5,x-.5,x,x],axis=1)
    py=np.stack([y,y,y,y+.5,y-.5],axis=1)
    ra,dec=wcs.all_pix2world(px,py,0)
    sx,sy=segwcs.all_world2pix(ra,dec,0)
    jac=np.stack([sx[:,1]-sx[:,2],sx[:,3]-sx[:,4],sy[:,1]-sy[:,2],sy[:,3]-sy[:,4]],axis=1).reshape(-1,2,2)
    rot=np.stack([a*np.cos(theta),-b*np.sin(theta),a*np.sin(theta),b*np.cos(theta)],axis=1).reshape(-1,2,2)
    axes=jac@rot;cov=axes@axes.transpose(0,2,1)
    vals,vecs=np.linalg.eigh(cov)
    return sx[:,0],sy[:,0],np.sqrt(vals[:,1]),np.sqrt(vals[:,0]),np.arctan2(vecs[:,1,1],vecs[:,0,1])


def cosmos_fill_ratios(geometry, source_ids, wcs, catalog_path, selected=None, origin=(0, 0)):
    """Measure own-ID coverage on native segmentation pixels; missing coverage is NaN."""
    from pathlib import Path
    from astropy.io import fits
    from astropy.wcs import WCS

    catalog_path = Path(catalog_path)
    ids = np.asarray(source_ids)
    result = np.full(len(ids), np.nan)
    use = geometry.valid()
    if selected is not None:
        use &= np.asarray(selected, dtype=bool)
    with fits.open(catalog_path, memmap=True) as hs:
        tab = hs[1].data
        order = np.argsort(tab['id'])
        sorted_ids = np.asarray(tab['id'])[order]
        pos = np.searchsorted(sorted_ids, ids)
        if np.any(pos >= len(order)) or not np.array_equal(sorted_ids[pos], ids):
            raise ValueError("Source IDs missing from COSMOS segmentation catalog")
        ci = order[pos]
        tiles = np.asarray(tab['tile'][ci]).astype(str)
        segments = np.asarray(tab['segment-id'][ci])
    for tile in np.unique(tiles[use]):
        inds = np.flatnonzero(use & (tiles == tile))
        path = catalog_path.parent/'segmentation_maps'/f'detection_chi2pos_SWLW_{tile}_segmap_v1.3.fits'
        with fits.open(path, memmap=True, do_not_scale_image_data=True) as hs:
            h = next(h for h in hs if h.header.get('NAXIS') == 2)
            mapped = transform_ellipses(wcs, WCS(h.header).celestial,
                geometry.x[inds]+origin[0], geometry.y[inds]+origin[1],
                geometry.major[inds], geometry.minor[inds], geometry.theta[inds])
            for j, i in enumerate(inds):
                values = [v[j] for v in mapped]
                if np.all(np.isfinite(values)) and min(values[2:4]) > 0:
                    result[i] = measure(h.data, h.header, *values, int(segments[i]))['own_fill']
    return result
