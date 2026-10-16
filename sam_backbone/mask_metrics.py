"""Detached sufficient statistics for GT/predicted mask diagnostics.

Reduce numerators/denominators across batches and ranks BEFORE taking means.
These diagnostics never change the mixed training objective. Reliability q is
retained in the numerator, as in the training BCE/Dice and IoU calibration losses.
"""
import torch

COMPONENT_WEIGHTS = {
    'dice': 'mask_dice', 'bce': 'mask_bce', 'centroid': 'mask_centroid',
    'outside': 'mask_outside', 'area': 'mask_min_area', 'max_area': 'mask_max_area',
    'pred_iou': 'mask_pred_iou', 'stability': 'mask_stability',
}
GROUP_STAT_KEYS = tuple(f'_mask_group_{group}_{key}'
    for group in ('gt', 'pred')
    for key in (*COMPONENT_WEIGHTS, 'dice_weight', 'bce_weight', 'geom_weight',
                'count', 'supervised_count', 'bce_count'))


@torch.no_grad()
def group_statistics(selected, kinds, prompt_weights, labelled, bce_eligible,
                     quality, supervision_scale):
    values = []
    for group in ('gt', 'pred'):
        member = (kinds != 1) if group == 'gt' else (kinds == 1)
        weight = prompt_weights.detach().double() * member
        dw, bw = weight * labelled, weight * bce_eligible
        for component in COMPONENT_WEIGHTS:
            w = dw if component in ('dice', 'pred_iou') else bw if component == 'bce' else weight
            if component in ('dice', 'bce'):
                w = w * quality * supervision_scale
            elif component == 'pred_iou':
                w = w * quality
            values.append((selected[component].detach().double() * w).sum())
        values.extend([dw.sum(), bw.sum(), weight.sum(),
                       ((weight > 0)).sum(), ((weight > 0) & labelled).sum(),
                       ((weight > 0) & bce_eligible).sum()])
    return torch.stack(values)


def finish_group_metrics(metrics, weights):
    """Replace internal SUM statistics by independently normalized group means."""
    metrics = dict(metrics)
    raw = {k: metrics.pop(k, 0.) for k in GROUP_STAT_KEYS}
    for group in ('gt', 'pred'):
        prefix = f'_mask_group_{group}_'
        total = 0.
        for component, attr in COMPONENT_WEIGHTS.items():
            denom_key = 'dice_weight' if component in ('dice', 'pred_iou') else 'bce_weight' if component == 'bce' else 'geom_weight'
            denominator = raw[prefix + denom_key]
            mean = raw[prefix + component] / denominator if denominator > 0 else 0.
            metrics[f'mask_{group}_mean_{component}'] = mean
            total += float(getattr(weights, attr, 0.)) * mean
        metrics[f'mask_{group}_mean_total'] = total  # Before outer mask weight.
        for key in ('count', 'supervised_count', 'bce_count', 'dice_weight', 'bce_weight', 'geom_weight'):
            metrics[f'mask_{group}_{key}'] = raw[prefix + key]
    count = metrics['mask_gt_count'] + metrics['mask_pred_count']
    metrics['mask_actual_pred_fraction'] = metrics['mask_pred_count']/count if count else 0.
    return metrics
