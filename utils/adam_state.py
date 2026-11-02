"""Read-only, module-level Adam/AdamW summaries; no extra forwards or backwards."""
from __future__ import annotations

import json
import math
from pathlib import Path
import warnings

import torch


SCOPES = ('encoder', 'proposal_decoder', 'mask_decoder')


def parameter_scope(name: str) -> str | None:
    name = name.replace('_orig_mod.', '')
    if name.startswith('encoder.'):
        return 'encoder'
    if name.startswith('decoder.'):
        return 'proposal_decoder'
    if name.startswith(('mask_decoder.', 'prompt_encoder.')):
        return 'mask_decoder'
    return None


@torch.no_grad()
def adam_state_metrics(model, optimizer) -> dict[str, float]:
    """Summarize post-step moments and the just-completed adaptive update.

    adaptive_step excludes decoupled weight decay. It is reconstructed from the
    post-step moments, the LR used for that step, and per-parameter bias
    correction. Parameters with grad=None did not step and contribute no update.
    No per-parameter weights, gradients, optimizer state, or RNG are modified.
    """
    if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        raise TypeError('Adam state logging requires torch.optim.Adam or AdamW')
    names = {id(p): parameter_scope(n) for n, p in model.named_parameters()}
    # Keep reductions on the parameter device; one transfer per device/scope.
    buckets = {}
    for group in optimizer.param_groups:
        beta1, beta2 = group['betas']
        lr = group['lr']
        lr = float(lr.detach().cpu()) if torch.is_tensor(lr) else float(lr)
        eps = float(group['eps'])
        for p in group['params']:
            scope = names.get(id(p))
            if scope is None:
                continue
            key = (scope, p.device)
            if key not in buckets:
                buckets[key] = dict(sums=torch.zeros(8, device=p.device, dtype=torch.float64),
                                    step_min=torch.tensor(float('inf'), device=p.device, dtype=torch.float64),
                                    step_max=torch.zeros((), device=p.device, dtype=torch.float64),
                                    v_max=torch.zeros((), device=p.device, dtype=torch.float64),
                                    elements=0, active=0, initialized=0, tensors=0,
                                    lr_min=lr, lr_max=lr)
            b = buckets[key]
            b['elements'] += p.numel()
            b['tensors'] += 1
            b['lr_min'], b['lr_max'] = min(b['lr_min'], lr), max(b['lr_max'], lr)
            # [weight_sq, grad_sq, m_sq, v_sum, corrected_m_sq,
            #  corrected_v_sum, adaptive_step_sq, active_weight_sq]
            terms = [p.detach().norm(dtype=torch.float32).square()]
            zero = terms[0].new_zeros(())
            active = p.grad is not None
            if active:
                b['active'] += p.numel()
                terms.append(p.grad.detach().norm(dtype=torch.float32).square())
            else:
                terms.append(zero)
            state = optimizer.state.get(p, {})  # never create a state entry
            if 'exp_avg' not in state:
                terms.extend([zero] * 6)
            else:
                b['initialized'] += p.numel()
                step = torch.as_tensor(state['step'], device=p.device, dtype=torch.float64)
                b['step_min'] = torch.minimum(b['step_min'], step)
                b['step_max'] = torch.maximum(b['step_max'], step)
                m, v = state['exp_avg'].detach(), state['exp_avg_sq'].detach()
                m_sq = m.norm(dtype=torch.float32).square()
                v_sum = v.sum(dtype=torch.float32)
                b['v_max'] = torch.maximum(b['v_max'], v.max().double())
                bc1, bc2 = 1. - beta1 ** step, 1. - beta2 ** step
                terms.extend([m_sq, v_sum, m_sq / bc1.square(), v_sum / bc2])
                if active:
                    used_v = state['max_exp_avg_sq'].detach() if group.get('amsgrad', False) else v
                    denom = used_v.float().sqrt() / bc2.sqrt().float()
                    denom.add_(eps)
                    update_sq = (m.float() / denom).norm().square() * (lr / bc1).square()
                    terms.extend([update_sq, terms[0]])
                else:
                    terms.extend([zero, zero])
            b['sums'].add_(torch.stack([v.double() for v in terms]))
    combined = {}
    for (scope, _), b in buckets.items():
        values = torch.cat([b['sums'], torch.stack([b['step_min'], b['step_max'], b['v_max']])]).cpu().tolist()
        if scope not in combined:
            combined[scope] = dict(sums=[0.] * 8, step_min=float('inf'), step_max=0., v_max=0.,
                                   elements=0, active=0, initialized=0, tensors=0,
                                   lr_min=float('inf'), lr_max=0.)
        c = combined[scope]
        c['sums'] = [a + v for a, v in zip(c['sums'], values[:8])]
        c['step_min'] = min(c['step_min'], values[8])
        c['step_max'] = max(c['step_max'], values[9])
        c['v_max'] = max(c['v_max'], values[10])
        for name in ('elements', 'active', 'initialized', 'tensors'):
            c[name] += b[name]
        c['lr_min'], c['lr_max'] = min(c['lr_min'], b['lr_min']), max(c['lr_max'], b['lr_max'])
    result = {}
    for scope in SCOPES:
        if scope not in combined:
            continue
        c = combined[scope]
        n, active, initialized = c['elements'], c['active'], c['initialized']
        w2, g2, m2, vsum, mh2, vhsum, u2, aw2 = c['sums']
        root = lambda x: math.sqrt(max(x, 0.))
        metrics = dict(parameter_norm=root(w2), parameter_rms=root(w2 / n),
                       gradient_norm=root(g2), gradient_rms=root(g2 / max(active, 1)),
                       exp_avg_norm=root(m2), exp_avg_rms=root(m2 / max(initialized, 1)),
                       exp_avg_sq_mean=vsum / max(initialized, 1), exp_avg_sq_max=c['v_max'],
                       sqrt_exp_avg_sq_rms=root(vsum / max(initialized, 1)),
                       bias_corrected_exp_avg_norm=root(mh2),
                       bias_corrected_sqrt_exp_avg_sq_rms=root(vhsum / max(initialized, 1)),
                       adaptive_step_norm=root(u2), adaptive_step_rms=root(u2 / max(active, 1)),
                       adaptive_step_to_weight_ratio=root(u2) / max(root(aw2), 1e-30),
                       step_min=c['step_min'] if initialized else 0., step_max=c['step_max'],
                       initialized_fraction=initialized / n, active_fraction=active / n,
                       parameter_count=float(n), lr_min=c['lr_min'], lr_max=c['lr_max'])
        result.update({f'{scope}/{name}': float(value) for name, value in metrics.items()})
    return result


class AdamStateLogger:
    """Rank-zero logger, independent of regular loss logging and gradient probes."""

    def __init__(self, output_dir, *, interval=100, log_fn=None):
        if interval < 1:
            raise ValueError('Adam state log interval must be positive')
        self.interval, self.log_fn = int(interval), log_fn
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def maybe_log(self, model, optimizer, *, optimizer_step, iteration, epoch, mask_active):
        if optimizer_step % self.interval:
            return None
        try:
            metrics = adam_state_metrics(model, optimizer)
            record = dict(optimizer_step=int(optimizer_step), iteration=int(iteration),
                          epoch_index=int(epoch), mask_active=bool(mask_active), metrics=metrics)
            with (self.output_dir / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record, allow_nan=False) + '\n')
            if self.log_fn is not None:
                self.log_fn(metrics, iteration, epoch)
            return record
        except Exception as exc:
            error = dict(optimizer_step=int(optimizer_step), error=f'{type(exc).__name__}: {exc}')
            with (self.output_dir / 'errors.jsonl').open('a') as f:
                f.write(json.dumps(error) + '\n')
            warnings.warn(f'Adam state logging failed; training continues: {error["error"]}')
            return None
