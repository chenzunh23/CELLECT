"""Bounded SAM mask decoding and green browser overlays.

70% transparent means alpha=0.30. Instances remain independent (overlap is
supported); cached masks are bit-packed, not float logits or GPU tensors.
"""
from contextlib import nullcontext
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sam_backbone.losses import _boxes_from_centers_shapes, _sam_mask_loss_for_prompts

MASK_ALPHA = 0.30


def decode_masks(model, outputs, rows, *, device, amp='none', width=512, height=512,
                 chunk_size=32, threshold=0., box_scale=1., valid_mask=None, selection='pred_iou'):
    if selection not in ('pred_iou', 'geometry'):
        raise ValueError(f'Unknown mask selection: {selection}')
    if not hasattr(model, 'forward_sam_masks'):
        raise RuntimeError('Checkpoint model does not expose the SAM mask decoder')
    cfg = getattr(model, '_browser_mask_config', {})
    top = cfg.get('_top', cfg)
    args = top.get('args', cfg)
    loss_cfg = top.get('loss_weights', {})
    multimask = bool(loss_cfg.get('mask_multimask', True))
    ellipse_sigma = float(args.get('ellipse_sigma', 1.))
    if box_scale <= 0:
        raise ValueError('mask box scale must be positive')
    weights = SimpleNamespace(mask_selection='loss', mask_supervision_weight=0.,
        mask_multimask=multimask,mask_prompt_chunk_size=chunk_size,
        mask_dice=0., mask_bce=0., mask_pred_iou=0., mask_stability=0.,
        **{k:loss_cfg.get(k, v) for k,v in dict(mask_centroid=.4, mask_outside=1.,
            mask_min_area=1., mask_max_area=.1, mask_area_ratio_lower=.05,
            mask_area_ratio_upper=2., mask_max_area_ratio=.8,
            mask_outside_kron_scale=1.5).items()})
    # Geometry remains an explicit fallback. Predicted IoU ranks candidates
    # directly; scores are not probabilities and must not be clamped to [0, 1].
    valid = torch.ones((height,width),device=device) if valid_mask is None else torch.as_tensor(valid_mask,device=device).float()
    packed, bounds, row_indices, selected_tokens, candidate_areas, candidate_scores = [], [], [], [], [], []
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start:start+chunk_size]
        points = torch.tensor([[r['x'], r['y']] for r in chunk], device=device, dtype=torch.float32)
        shapes = torch.tensor([[r['major'],r['minor'],r.get('theta',0.)] for r in chunk],device=device,dtype=torch.float32)
        boxes = None if loss_cfg.get('mask_prompt_center_only',False) else _boxes_from_centers_shapes(
            points, shapes, image_size=(height,width), ellipse_sigma=ellipse_sigma*box_scale)
        indices = torch.zeros(len(chunk), device=device, dtype=torch.long)
        context = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' and amp == 'bf16' else nullcontext()
        with torch.inference_mode(), context:
            low, quality = model.forward_sam_masks(outputs['image_embeddings'], indices, points, boxes,
                                             multimask_output=multimask, chunk_size=chunk_size, output_size=None)
            if not torch.isfinite(quality).all():
                raise ValueError('Non-finite predicted IoU scores')
            if selection == 'pred_iou':
                best = quality.argmax(dim=1)
            else:
                class CachedCandidates:
                    def forward_sam_masks(self, *args, **kwargs):
                        return low, quality
                prompts = dict(batch_indices=indices,centers=points,prompt_shapes=shapes,
                               target_shapes=shapes,weights=points.new_ones(len(chunk)))
                rank = _sam_mask_loss_for_prompts(CachedCandidates(),outputs,prompts,
                    weights=weights,image_hw=(height,width),ellipse_sigma=ellipse_sigma,
                    batch={'band_valid_mask':valid[None,None]},return_selection=True)
                best = rank['selected_indices']
            candidate_scores.extend(quality.float().cpu().tolist())
            # Interpolate to padded input size, then crop, as forward_sam_masks
            # does. Ranking stays at training mask resolution.
            eh,ew=outputs['image_embeddings'].shape[-2:]
            padded=(int(eh)*getattr(model,'patch_size',16),int(ew)*getattr(model,'patch_size',16))
            full=F.interpolate(low.float(),size=padded,mode='bilinear',align_corners=False)[...,:height,:width]
            all_masks=(full>threshold)&valid.bool()[None,None]
            masks=all_masks[torch.arange(len(chunk),device=device),best]
            selected_tokens.extend((best+(1 if multimask else 0)).cpu().tolist())
            candidate_areas.extend(all_masks.sum((-1,-2)).cpu().tolist())
        batch = masks[:, :height, :width].cpu().numpy()
        for i, mask in enumerate(batch):
            ys, xs = np.nonzero(mask)
            if not len(xs):
                continue
            x0,y0,x1,y1 = xs.min(), ys.min(), xs.max(), ys.max()
            packed.append(np.packbits(mask[y0:y1+1,x0:x1+1].reshape(-1)))
            bounds.append((x0,y0,x1,y1))
            row_indices.append(start+i)
    offsets = np.r_[0, np.cumsum([len(p) for p in packed])].astype(np.int64)
    return dict(packed=np.concatenate(packed) if packed else np.empty(0,np.uint8), offsets=offsets,
                bounds=np.asarray(bounds, dtype=np.int32).reshape(-1,4),
                row_indices=np.asarray(row_indices, dtype=np.int32), height=height, width=width,
                selected_tokens=np.asarray(selected_tokens,dtype=np.int32),
                candidate_areas=np.asarray(candidate_areas,dtype=np.int64),
                candidate_pred_iou=np.asarray(candidate_scores,dtype=np.float32),
                selection=selection,box_scale=box_scale)


def overlay_masks(display_rgb, masks, selected_indices=None):
    """Input PNG array is top-down; mask coordinates are bottom-up astronomy pixels."""
    out = np.array(display_rgb, dtype=np.uint8, copy=True)
    if out.ndim == 2:
        out = np.repeat(out[...,None],3,axis=2)
    h, w = masks['height'], masks['width']
    if out.shape[:2] != (h,w):
        raise ValueError('Mask/display shapes differ')
    keep = np.ones(len(masks['bounds']), bool) if selected_indices is None else np.isin(masks['row_indices'], selected_indices)
    if not keep.any():
        return out
    # Cache each bounding ROI only; bright-field FP counts need not allocate
    # thousands of complete 512-square instance arrays. Overlap blends once.
    union = np.zeros((h,w), bool)
    for i in np.flatnonzero(keep):
        x0,y0,x1,y1 = masks['bounds'][i]
        lo,hi = masks['offsets'][i:i+2]
        roi = np.unpackbits(masks['packed'][lo:hi], count=int((y1-y0+1)*(x1-x0+1)))
        union[y0:y1+1,x0:x1+1] |= roi.reshape(y1-y0+1,x1-x0+1).astype(bool)
    union = np.flipud(union)
    green = np.array([0,255,0],np.float32)
    out[union] = np.rint((1-MASK_ALPHA)*out[union] + MASK_ALPHA*green).astype(np.uint8)
    image = Image.fromarray(out)
    draw = ImageDraw.Draw(image)
    for x0,y0,x1,y1 in masks['bounds'][keep]:
        draw.rectangle((int(x0),int(h-1-y1),int(x1),int(h-1-y0)),outline=(0,255,0),width=1)
    return np.asarray(image)
