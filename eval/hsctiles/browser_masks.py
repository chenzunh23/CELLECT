"""Bounded SAM mask decoding and green browser overlays.

70% transparent means alpha=0.30. Instances remain independent (overlap is
supported); cached masks are bit-packed, not float logits or GPU tensors.
"""
from contextlib import nullcontext
import numpy as np
import torch
from PIL import Image, ImageDraw
from eval.eval_utils import prompt_boxes

MASK_ALPHA = 0.30


def decode_masks(model, outputs, rows, *, device, amp='none', width=512, height=512,
                 chunk_size=32, threshold=0., box_scale=2.):
    if not hasattr(model, 'forward_sam_masks'):
        raise RuntimeError('Checkpoint model does not expose the SAM mask decoder')
    packed, bounds, row_indices = [], [], []
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start:start+chunk_size]
        points = torch.tensor([[r['x'], r['y']] for r in chunk], device=device, dtype=torch.float32)
        boxes = prompt_boxes(chunk, image_size=max(width,height), scale=box_scale).to(device)
        indices = torch.zeros(len(chunk), device=device, dtype=torch.long)
        context = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' and amp == 'bf16' else nullcontext()
        with torch.inference_mode(), context:
            low, _ = model.forward_sam_masks(outputs['image_embeddings'], indices, points, boxes,
                                             multimask_output=False, chunk_size=chunk_size, output_size=(height,width))
            masks = low[:,0] > threshold
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
                row_indices=np.asarray(row_indices, dtype=np.int32), height=height, width=width)


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
