"""Sparse instance prompts and partial-label mask losses for SAM training."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def instance_prompts(batch, outputs, device):
    bands = int(outputs['confidence'].shape[1])
    h, w = outputs['confidence'].shape[-2:]
    rows, centers, shapes, indices, ids = [], [], [], [], []
    for b, by_band in enumerate(batch.get('band_mask_instances', [])):
        for band, instances in enumerate(by_band):
            if not instances or 'band_shape_source_ids' not in batch:
                continue
            lookup = {int(s): j for j, s in enumerate(batch['band_shape_source_ids'][b][band].tolist())}
            for row in instances:
                j = lookup.get(row['source_id'])
                if j is None:
                    continue  # A cropped fringe without its source center is not a prompt.
                center = batch['band_shape_source_centers'][b][band][j]
                shape = batch['band_shape_source_values'][b][band][j]
                if not (torch.isfinite(center).all() and torch.isfinite(shape).all()
                        and 0 <= center[0] < w and 0 <= center[1] < h and row['mask'].any()):
                    continue
                rows.append(row); centers.append(center); shapes.append(shape)
                indices.append(b*bands+band); ids.append(row['source_id'])
    if not rows:
        return None
    centers = torch.stack(centers).to(device=device, dtype=torch.float32)
    shapes = torch.stack(shapes).to(device=device, dtype=torch.float32)
    return dict(batch_indices=torch.tensor(indices, device=device), centers=centers,
        prompt_shapes=shapes, target_shapes=shapes, weights=centers.new_ones(len(rows)),
        mask_target_weights=centers.new_ones(len(rows)), source_ids=torch.tensor(ids, device=device),
        instance_masks=rows)


def combine_prompts(mandatory, gt, pred, *, pred_ratio, max_gt=128, max_pred=128):
    """Keep every labelled source; other points share the remaining per-image cap.

    GT/pred quota is selected by the epoch schedule, not by weighting the same
    labelled sources twice. A nonpositive cap retains the historical unlimited
    option for ordinary points. Labelled prompts survive the all-predicted phase.
    """
    sets = [(2, mandatory), (0, gt), (1, pred)]
    sets = [(kind, p) for kind, p in sets if p is not None]
    if not sets:
        return None
    anchor = sets[0][1]['centers']
    selected = []
    flats = sorted(set(int(i) for _, p in sets for i in p['batch_indices'].tolist()))
    for flat in flats:
        labelled = [] if mandatory is None else torch.where(mandatory['batch_indices'] == flat)[0].tolist()
        selected.extend((2, mandatory, j) for j in labelled)
        protected_ids = set() if mandatory is None else set(mandatory['source_ids'][labelled].tolist())
        protected_centers = None if not labelled else mandatory['centers'][labelled]
        pools = {}
        for kind, p in [(0, gt), (1, pred)]:
            if p is None:
                pools[kind] = []; continue
            ix = torch.where(p['batch_indices'] == flat)[0]
            if kind == 0 and 'source_ids' in p:
                ix = ix[torch.tensor([int(s) not in protected_ids for s in p['source_ids'][ix].tolist()], device=ix.device, dtype=torch.bool)]
            if kind == 1 and protected_centers is not None and len(ix):
                # Keep distinct nearby children; remove only coincident predicted prompts.
                ix = ix[torch.cdist(p['centers'][ix], protected_centers).min(dim=1).values > 1.0]
            pools[kind] = ix[torch.randperm(len(ix), device=ix.device)].tolist()
        cap = (max_gt if pred_ratio <= 0 else max_pred if pred_ratio >= 1
               else round((1-pred_ratio)*max_gt + pred_ratio*max_pred))
        if (max_gt <= 0 and pred_ratio < 1) or (max_pred <= 0 and pred_ratio > 0):
            cap = len(labelled) + len(pools[0]) + len(pools[1])
        remaining = max(0, cap-len(labelled))
        want_pred = round(remaining*pred_ratio)
        npred, ngt = min(want_pred, len(pools[1])), min(remaining-want_pred, len(pools[0]))
        spare = remaining-npred-ngt
        extra = min(spare, len(pools[1])-npred); npred += extra; spare -= extra
        ngt += min(spare, len(pools[0])-ngt)
        selected.extend((0, gt, j) for j in pools[0][:ngt])
        selected.extend((1, pred, j) for j in pools[1][:npred])
    if not selected:
        return None
    keys = ('batch_indices', 'centers', 'prompt_shapes', 'target_shapes', 'weights', 'mask_target_weights')
    result = {k: torch.stack([p[k][j] for _, p, j in selected]) for k in keys}
    result['source_kind'] = torch.tensor([kind for kind, _, _ in selected], device=anchor.device)
    result['instance_masks'] = [p['instance_masks'][j] if kind == 2 else None for kind, p, j in selected]
    return result


def partial_targets(prompts, batch, *, image_hw, mask_hw, device):
    """Instance coverage, trusted sky and observed coverage at mask resolution.

    Labelled prompts supervise mask/non-mask throughout observed coverage.
    Unlabelled prompts use only trusted sky as negatives; a low-resolution sky
    cell must be entirely trusted. Area averaging preserves small positives and
    partial observed coverage. Independent parent/child masks remain overlapping.
    """
    n = len(prompts['centers']); h, w = image_hw
    positive = torch.zeros((n, h, w), device=device)
    quality = torch.ones(n, device=device)
    labelled = torch.zeros(n, device=device, dtype=torch.bool)
    for i, row in enumerate(prompts.get('instance_masks', [None]*n)):
        if row is None:
            continue
        x, y = int(row['x0']), int(row['y0']); mh, mw = row['mask'].shape
        xa, ya, xb, yb = max(0, x), max(0, y), min(w, x+mw), min(h, y+mh)
        if xa >= xb or ya >= yb:
            continue
        positive[i, ya:yb, xa:xb] = row['mask'][ya-y:yb-y, xa-x:xb-x].to(device)
        quality[i] = float(row['weight'])
        labelled[i] = True
    flat = prompts['batch_indices']
    valid = torch.ones_like(positive)
    negative = torch.zeros_like(positive)
    if batch is not None:
        if 'band_valid_mask' in batch:
            valid = batch['band_valid_mask'].to(device).reshape(-1, h, w)[flat].float()
        if 'band_trusted_background' in batch:
            negative = batch['band_trusted_background'].to(device).reshape(-1, h, w)[flat].float()
    positive *= valid
    negative *= valid * (positive == 0)
    labelled &= positive.sum((-1, -2)) > 0
    if tuple(mask_hw) != tuple(image_hw):
        def down(x):
            return F.interpolate(x[:, None], size=mask_hw, mode='area')[:, 0]
        positive, negative, valid = down(positive), down(negative), down(valid)
        negative = (negative >= 1-1e-6).float()
    return positive, negative, valid, quality, labelled


def partial_bce_dice(logits, positive, negative, labelled, *, valid=None):
    """Full-instance BCE/Dice when labelled; trusted-sky BCE otherwise.

    ``positive`` and ``valid`` are area-averaged fractions of the original input
    cell, not a hard nearest-neighbour mask. For a labelled instance, every
    observed pixel outside that instance is negative, including dense-label
    ignore pixels. BCE is an ordinary pixel mean, without separately balancing
    positive/negative classes. Fractional no-data coverage remains excluded.

    For unlabelled prompts, only ``negative`` (trusted background) contributes
    BCE and Dice is disabled. The caller retains per-prompt weighting/reduction.
    """
    logits = logits.float()
    coverage = torch.ones_like(positive) if valid is None else valid
    coverage = coverage.to(dtype=logits.dtype).clamp(0, 1)[:, None]
    pos = torch.minimum(positive.to(dtype=logits.dtype).clamp_min(0)[:, None], coverage)
    neg = torch.minimum(negative.to(dtype=logits.dtype).clamp_min(0)[:, None], coverage)
    pos_count = pos.sum((-1, -2))
    valid_count = coverage.sum((-1, -2))
    neg_count = neg.sum((-1, -2))
    has_label = labelled[:, None] & (pos_count > 0)

    # Equivalent to BCEWithLogits(target=pos/coverage), weighted by coverage.
    # This form avoids treating the unobserved fraction of a cell as background.
    positive_loss = F.softplus(-logits)
    negative_loss = F.softplus(logits)
    full_bce = (positive_loss*pos + negative_loss*(coverage-pos)).sum((-1, -2))
    full_bce = full_bce / valid_count.clamp_min(1e-6)
    sky_bce = (negative_loss*neg).sum((-1, -2)) / neg_count.clamp_min(1e-6)
    bce = torch.where(has_label, full_bce, sky_bce)

    prob = logits.sigmoid()
    dice = 1-(2*(prob*pos).sum((-1, -2))+1)/(
        (prob*coverage).sum((-1, -2))+pos_count+1)
    dice = torch.where(has_label, dice, torch.zeros_like(dice))
    eligible = has_label | (neg_count > 0)
    return bce, dice, eligible[:, 0]


def area_ratio_penalty(ratio, lower, upper):
    if lower <= 0 or upper <= lower:
        raise ValueError('mask area bounds require 0 < lower < upper')
    log_r = ratio.clamp_min(1e-8).log()
    return F.relu(math.log(lower)-log_r) + F.relu(log_r-math.log(upper))
