"""Dense confidence diagnostics, independent of center thresholding and NMS."""
import numpy as np
import torch

from eval.eval_utils import CONF_COLORS
from utils.train_ops_utils import confidence_detection_score


def confidence_maps(outputs, *, width, height, valid_mask, score_mode):
    logits = outputs['confidence'].detach().float().cpu()
    if logits.ndim != 4 or logits.shape[0] != 1:
        raise ValueError(f'Expected confidence [1,classes,H,W], got {tuple(logits.shape)}')
    logits = logits[..., :height, :width].contiguous()
    if tuple(logits.shape[-2:]) != (height, width):
        raise ValueError('Confidence map does not cover the input image')
    valid = np.asarray(valid_mask, bool) & torch.isfinite(logits[0]).all(0).numpy()
    if valid.shape != (height, width):
        raise ValueError('Confidence validity shape mismatch')
    labels = logits[0].argmax(0).numpy().astype(np.uint8)
    labels[~valid] = 0
    scores = confidence_detection_score({'confidence': logits}, score_mode)[0].numpy().copy()
    ordinal = confidence_detection_score({'confidence': logits}, 'ordinal_expectation')[0].numpy().copy()
    scores[~valid] = np.nan
    ordinal[~valid] = np.nan
    return dict(logits=logits[0].numpy(), labels=labels, score=scores,
                ordinal_expectation=ordinal, valid=valid)


def confidence_overlay_uint8(background, labels, *, alpha=.60):
    """Background PNG is top-down; stored dense maps use lower-origin pixels."""
    base = np.asarray(background, dtype=np.uint8)
    if base.ndim == 2:
        base = np.repeat(base[..., None], 3, axis=-1)
    levels = np.flipud(np.asarray(labels))
    if base.shape[:2] != levels.shape:
        raise ValueError('Confidence overlay and background shape mismatch')
    rgb = base.astype(np.float32)
    for level, color in CONF_COLORS.items():
        selected = levels == level
        rgb[selected] = (1-alpha)*rgb[selected] + alpha*255*np.asarray(color)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def confidence_display_options(options=None):
    """Whitelist the snapshot of browser background settings used on export."""
    options = options or {}
    result = {}
    for key in ('input_image', 'invert_background', 'scale_custom'):
        if key in options:
            result[key] = options[key] is True or str(options[key]).lower() in ('1', 'true', 'yes')
    for key in ('display_scaling', 'smooth_mode'):
        if key in options:
            result[key] = str(options[key])
    for key in ('smooth_sigma', 'scale_low_pct', 'scale_high_pct', 'scale_low_value', 'scale_high_value'):
        if options.get(key) not in (None, '', 'null', 'None'):
            result[key] = float(options[key])
    if 'smooth_radius' in options:
        result['smooth_radius'] = int(options['smooth_radius'])
    return result
