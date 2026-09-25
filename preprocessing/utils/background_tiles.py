"""Read SExtractor sky masks on arbitrary windows; never stitch source labels."""
import json
from pathlib import Path
import numpy as np


def check_source(signature, source):
    source = Path(source)
    if Path(signature['path']).resolve() != source.resolve():
        raise ValueError(f'Background belongs to a different science image: {source}')
    stat = source.stat()
    if stat.st_size != signature['bytes'] or stat.st_mtime_ns != signature['mtime_ns']:
        raise ValueError(f'Stale background for {source}')


def read_mask(path):
    with np.load(path) as z:
        return z['background_mask'].astype(bool)


def assemble(rows, x, y, width, height):
    """Conservative AND in overlapping Abell parents; uncovered pixels stay false."""
    out = np.ones((height, width), bool)
    count = np.zeros((height, width), np.uint8)
    used = []
    for row in rows:
        rx, ry = row['origin_xy']; h, w = row['shape_yx']
        xa, ya = max(x, rx), max(y, ry)
        xb, yb = min(x+width, rx+w), min(y+height, ry+h)
        if xb <= xa or yb <= ya:
            continue
        mask = read_mask(row['mask'])
        if mask.shape != (h, w):
            raise ValueError(f'Background mask shape mismatch: {row["mask"]}')
        sl = np.s_[ya-y:yb-y, xa-x:xb-x]
        out[sl] &= mask[ya-ry:yb-ry, xa-rx:xb-rx]
        count[sl] += 1
        used.append(str(row['mask']))
    out[count == 0] = False
    return out, dict(paths=used, covered_pixels=int(np.count_nonzero(count)),
                     overlap_pixels=int(np.count_nonzero(count > 1)), overlap_rule='AND')


def cosmos_region(manifest, source, x, y, width, height):
    doc = json.loads(Path(manifest).read_text())
    check_source(doc['source'], source)
    return assemble(doc['tiles'], x, y, width, height)


def abell_region(root, band, source, x, y, width, height):
    rows = []
    for receipt in sorted((Path(root)/'abell'/band).glob('*/complete.json')):
        doc = json.loads(receipt.read_text())
        if doc['status'] != 'done':
            continue
        tile = doc['job']['tile']
        tx, ty, size = tile['x0'], tile['y0'], tile['size']
        if tx >= x+width or ty >= y+height or tx+size <= x or ty+size <= y:
            continue
        check_source(doc['signature']['source'], source)
        rows.append(dict(origin_xy=[tx, ty], shape_yx=[size, size],
                         mask=str(receipt.parent/'background_mask.npz')))
    return assemble(rows, x, y, width, height)
