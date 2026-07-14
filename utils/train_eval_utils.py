from __future__ import annotations

import argparse
import hashlib
import os
import re
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from astro_train_data import CutoutRecord, collate_cutouts
from astro_train_ops import LossWeights, unwrap_model


class _DDPModelWrapper(nn.Module):
    """Expose loss-only parameters to DDP from the wrapped forward."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._ddp_wrapped_model = model

    def forward(
        self,
        image: torch.Tensor,
        *,
        processing_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if processing_ids is None:
            outputs = self._ddp_wrapped_model(image)
        else:
            outputs = self._ddp_wrapped_model(image, processing_ids=processing_ids)
        dummy = image.new_zeros(())
        base_model = unwrap_model(self._ddp_wrapped_model)
        for name in ("EX", "EN", "prompt_encoder", "mask_decoder"):
            module = getattr(base_model, name, None)
            if module is None:
                continue
            for param in module.parameters():
                dummy = dummy + param.sum() * 0.0
        if dummy.requires_grad:
            outputs = dict(outputs)
            for key in ("confidence", "shape", "seg_logits"):
                if key in outputs:
                    outputs[key] = outputs[key] + dummy
        return outputs


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def distributed_requested(args: argparse.Namespace) -> bool:
    return bool(args.ddp) or env_int("WORLD_SIZE", 1) > 1


def setup_distributed(args: argparse.Namespace) -> Tuple[bool, int, int, int, torch.device, str]:
    distributed = distributed_requested(args)
    local_rank = env_int("LOCAL_RANK", args.local_rank)
    backend = args.dist_backend
    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() and torch.device(args.device).type == "cuda" else "gloo"

    if distributed:
        dist.init_process_group(
            backend=backend,
            init_method=args.dist_url,
            timeout=timedelta(minutes=max(1.0, float(args.ddp_timeout_minutes))),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    requested = torch.device(args.device)
    if distributed and requested.type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = requested
    return distributed, rank, world_size, local_rank, device, backend


def is_main(rank: int) -> bool:
    return int(rank) == 0


def sync_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def wrap_ddp(model: nn.Module, args: argparse.Namespace, device: torch.device, local_rank: int) -> nn.Module:
    if device.type == "cuda":
        device_ids = [local_rank]
        output_device = local_rank
    else:
        device_ids = None
        output_device = None
    static_mode = str(getattr(args, "ddp_static_graph", "auto")).lower()
    if static_mode == "auto":
        dynamic_graph = (
            int(getattr(args, "mask_loss_warmup_epochs", 0)) > 0
            or int(getattr(args, "freeze_proposal_after_epochs", -1)) >= 0
            or (
                int(getattr(args, "sam_lr_phase2_epoch", -1)) >= 0
                and getattr(args, "sam_head_lr_after", None) is not None
                and float(getattr(args, "sam_head_lr_after")) <= 0.0
            )
        )
        static_graph = not dynamic_graph
    else:
        static_graph = static_mode == "on"
    find_unused_parameters = bool(getattr(args, "ddp_find_unused_parameters", False)) or not bool(static_graph)
    setattr(args, "_ddp_static_graph_resolved", bool(static_graph))
    setattr(args, "_ddp_find_unused_parameters_resolved", bool(find_unused_parameters))
    return DistributedDataParallel(
        _DDPModelWrapper(model),
        device_ids=device_ids,
        output_device=output_device,
        find_unused_parameters=find_unused_parameters,
        static_graph=static_graph,
        gradient_as_bucket_view=True, # The last two options are under test for SAM
    )


def parse_patch_specs(specs: Iterable[str], patch_file: str | None = None) -> set[str]:
    patches: set[str] = set()
    for spec in specs:
        text = str(spec).strip()
        if text:
            patches.add(text.strip("/"))
    if patch_file:
        for line in Path(patch_file).expanduser().read_text(encoding="utf-8").splitlines():
            text = line.split("#", 1)[0].strip()
            if text:
                patches.add(text.strip("/"))
    return patches


def _selector_parts(spec: str) -> tuple[str, str]:
    text = str(spec).strip().strip("/")
    if "@" not in text:
        return text, ""
    patch_part, group_part = text.rsplit("@", 1)
    return patch_part.strip().strip("/"), group_part.strip()


def _record_group_name(rec: CutoutRecord) -> str:
    tile_name = str(getattr(rec, "tile_name", "") or "")
    match = re.match(r"^(group_[^_]+)_", tile_name)
    return str(match.group(1)) if match else ""


def _is_random_group_selector(group_selector: str) -> bool:
    text = str(group_selector).strip().lower()
    return text in {"random", "rand", "random-group", "random_group"}


def _stable_selector_seed(*parts: object) -> int:
    payload = "||".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def record_patch_aliases(rec: CutoutRecord, root: Path) -> set[str]:
    aliases: set[str] = set()
    source = str(getattr(rec, "dataset_source", "coadd") or "coadd")
    aliases.add(source)
    if rec.patch:
        aliases.add(rec.patch)
        aliases.add(f"{source}:{rec.patch}")
        if rec.tract:
            aliases.add(f"{rec.tract}/{rec.patch}")
            aliases.add(f"{source}:{rec.tract}/{rec.patch}")
        elif root.name:
            aliases.add(f"{root.name}/{rec.patch}")
            aliases.add(f"{source}:{root.name}/{rec.patch}")
    if rec.relative_root:
        relative = str(rec.relative_root).strip("/")
        aliases.add(relative)
        aliases.add(f"{source}:{relative}")
    parts = Path(rec.relative_root).parts
    if len(parts) >= 2:
        aliases.add(f"{parts[-2]}/{parts[-1]}")
        aliases.add(f"{source}:{parts[-2]}/{parts[-1]}")
        aliases.add(str(parts[-1]))
        aliases.add(f"{source}:{parts[-1]}")
    return aliases


def filter_records_by_patches(
    records: Sequence[CutoutRecord],
    patches: set[str],
    root: Path,
    *,
    seed: int = 0,
) -> list[CutoutRecord]:
    if not patches:
        return list(records)
    wanted = sorted({patch.strip("/") for patch in patches if patch.strip("/")})
    selected: list[CutoutRecord] = []
    seen_names: set[str] = set()
    for spec in wanted:
        patch_selector, group_selector = _selector_parts(spec)
        matched = [
            rec
            for rec in records
            if patch_selector and (record_patch_aliases(rec, root) & {patch_selector})
        ]
        if not matched:
            continue
        group_selector = str(group_selector).strip()
        if group_selector:
            if _is_random_group_selector(group_selector):
                groups = sorted({group for group in (_record_group_name(rec) for rec in matched) if group})
                if not groups:
                    matched = []
                else:
                    selector_seed = _stable_selector_seed(seed, spec, patch_selector, len(groups))
                    chosen_group = groups[selector_seed % len(groups)]
                    matched = [rec for rec in matched if _record_group_name(rec) == chosen_group]
            else:
                normalized_group = str(group_selector)
                if normalized_group.isdigit():
                    normalized_group = f"group_{int(normalized_group):02d}"
                matched = [rec for rec in matched if _record_group_name(rec) == normalized_group]
        for rec in matched:
            if rec.name in seen_names:
                continue
            selected.append(rec)
            seen_names.add(rec.name)
    return selected


def record_patch_label(rec: CutoutRecord, root: Path) -> str:
    source = str(getattr(rec, "dataset_source", "coadd") or "coadd")
    if rec.tract and rec.patch:
        label = f"{rec.tract}/{rec.patch}"
        return label if source == "coadd" else f"{source}:{label}"
    if rec.patch and root.name:
        label = f"{root.name}/{rec.patch}"
        return label if source == "coadd" else f"{source}:{label}"
    label = rec.patch or rec.relative_root or "-"
    return label if source == "coadd" else f"{source}:{label}"


def loader_kwargs(args: argparse.Namespace, *, shuffle: bool, sampler: DistributedSampler | None = None) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": args.num_workers,
        "collate_fn": collate_cutouts,
        "pin_memory": bool(args.pin_memory),
    }
    if int(args.num_workers) > 0:
        kwargs["persistent_workers"] = bool(args.persistent_workers)
        kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
    return kwargs


def format_float(value: float, digits: int) -> str:
    if not np.isfinite(value):
        return ""
    return f"{float(value):.{int(digits)}f}"


def band_name(band_idx: int, band_names: Sequence[str]) -> str:
    if 0 <= int(band_idx) < len(band_names):
        return str(band_names[int(band_idx)])
    return f"band{int(band_idx)}"


def wcs_for_path(path: str, fits_hdu: int, cache: dict[tuple[str, int], object]) -> object | None:
    key = (str(path), int(fits_hdu))
    if key in cache:
        value = cache[key]
        return value if value is not False else None
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
    except Exception:
        cache[key] = False
        return None

    try:
        with fits.open(path, memmap=True) as hdul:
            header = hdul[int(fits_hdu)].header
            wcs = WCS(header).celestial
            if not wcs.has_celestial:
                cache[key] = False
                return None
            cache[key] = wcs
            return wcs
    except Exception:
        cache[key] = False
        return None


def radec_from_wcs(wcs: object | None, x: float, y: float) -> tuple[float, float]:
    if wcs is None:
        return float("nan"), float("nan")
    try:
        ra, dec = wcs.all_pix2world([float(x)], [float(y)], 0)  # type: ignore[attr-defined]
        return float(ra[0]), float(dec[0])
    except Exception:
        return float("nan"), float("nan")


def as_numpy_centers(obj: object) -> np.ndarray:
    try:
        if hasattr(obj, "detach"):
            arr = obj.detach().cpu().numpy()  # type: ignore[attr-defined]
        else:
            arr = np.asarray(obj)
    except Exception:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return arr.reshape(-1, 2)


def as_numpy_ids(obj: object) -> np.ndarray:
    try:
        if hasattr(obj, "detach"):
            arr = obj.detach().cpu().numpy()  # type: ignore[attr-defined]
        else:
            arr = np.asarray(obj)
    except Exception:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(arr).reshape(-1)


def as_numpy_mask(obj: object | None) -> np.ndarray | None:
    if obj is None:
        return None
    try:
        if hasattr(obj, "detach"):
            arr = obj.detach().cpu().numpy()  # type: ignore[attr-defined]
        else:
            arr = np.asarray(obj)
    except Exception:
        return None
    arr = np.asarray(arr, dtype=bool)
    return arr if arr.ndim == 2 else None


def point_in_mask_np(mask_np: np.ndarray | None, x: float, y: float) -> bool:
    if mask_np is None:
        return False
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    if xi < 0 or yi < 0 or yi >= mask_np.shape[0] or xi >= mask_np.shape[1]:
        return False
    return bool(mask_np[yi, xi])


def greedy_point_mapping(pred_xy: np.ndarray, gt_xy: np.ndarray, radius: float) -> tuple[dict[int, int], dict[int, float]]:
    pred_xy = as_numpy_centers(pred_xy)
    gt_xy = as_numpy_centers(gt_xy)
    if pred_xy.size == 0 or gt_xy.size == 0:
        return {}, {}
    dist = np.sqrt(((pred_xy[:, None, :] - gt_xy[None, :, :]) ** 2).sum(axis=2))
    pairs: list[tuple[float, int, int]] = []
    for pred_idx in range(dist.shape[0]):
        gt_idx = int(np.argmin(dist[pred_idx]))
        if float(dist[pred_idx, gt_idx]) <= float(radius):
            pairs.append((float(dist[pred_idx, gt_idx]), pred_idx, gt_idx))
    pairs.sort(key=lambda item: item[0])
    mapping: dict[int, int] = {}
    distances: dict[int, float] = {}
    used_gt: set[int] = set()
    for distance, pred_idx, gt_idx in pairs:
        if pred_idx in mapping or gt_idx in used_gt:
            continue
        mapping[pred_idx] = gt_idx
        distances[pred_idx] = distance
        used_gt.add(gt_idx)
    return mapping, distances


def flat_per_band_outputs(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.reshape(value.shape[0] * value.shape[1], *value.shape[2:]) for key, value in outputs.items()}


def sam_optimizer_param_groups(
    model: nn.Module,
    *,
    head_lr: float,
    encoder_lr: float,
    weight_decay: float,
) -> list[dict[str, object]]:
    """Build SAM fine-tuning parameter groups with separate proposal/SAM decoder/encoder LR."""

    base_model = unwrap_model(model)
    head_params: list[nn.Parameter] = []
    sam_decoder_params: list[nn.Parameter] = []
    encoder_params: list[nn.Parameter] = []
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith((
            "encoder.style_router.",
            "encoder.style_prompt_",
            "encoder.image_encoder.style_adapters.",
        )):
            head_params.append(param)
        elif name.startswith("encoder."):
            encoder_params.append(param)
        elif name.startswith(("prompt_encoder.", "mask_decoder.")):
            sam_decoder_params.append(param)
        else:
            head_params.append(param)
    groups: list[dict[str, object]] = []
    if head_params:
        groups.append(
            {
                "params": head_params,
                "lr": float(head_lr),
                "base_lr": float(head_lr),
                "weight_decay": float(weight_decay),
                "name": "head",
            }
        )
    if sam_decoder_params:
        groups.append(
            {
                "params": sam_decoder_params,
                "lr": float(head_lr),
                "base_lr": float(head_lr),
                "weight_decay": float(weight_decay),
                "name": "sam_decoder",
            }
        )
    if encoder_params:
        groups.append(
            {
                "params": encoder_params,
                "lr": float(encoder_lr),
                "base_lr": float(encoder_lr),
                "weight_decay": float(weight_decay),
                "name": "encoder",
            }
        )
    if not groups:
        raise ValueError("No trainable parameters found for SAM optimizer")
    return groups


class WarmupStepIterationLR:
    """Iteration-wise linear warmup followed by multiplicative step decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        warmup_steps: int,
        drop_steps: Sequence[int],
        drop_gamma: float,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(1, int(warmup_steps))
        self.drop_steps = tuple(sorted(max(1, int(step)) for step in drop_steps))
        self.drop_gamma = float(drop_gamma)
        self.step_index = 0
        for group in self.optimizer.param_groups:
            group.setdefault("base_lr", float(group["lr"]))
        self._apply_lrs()

    def _multiplier(self, step_index: int) -> float:
        update_number = max(1, int(step_index) + 1)
        if update_number <= self.warmup_steps:
            mult = float(update_number) / float(self.warmup_steps)
        else:
            mult = 1.0
        for drop_step in self.drop_steps:
            if update_number >= drop_step:
                mult *= self.drop_gamma
        return float(mult)

    def _apply_lrs(self) -> None:
        mult = self._multiplier(self.step_index)
        for group in self.optimizer.param_groups:
            group["lr"] = float(group["base_lr"]) * mult

    def set_base_lrs(self, overrides: dict[str, float]) -> None:
        """Override named group base LRs and immediately apply the current multiplier."""

        for group in self.optimizer.param_groups:
            name = str(group.get("name", ""))
            if name in overrides:
                group["base_lr"] = float(overrides[name])
        self._apply_lrs()

    def step(self) -> None:
        self.step_index = min(self.step_index + 1, self.total_steps)
        self._apply_lrs()

    def state_dict(self) -> dict[str, object]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "drop_steps": self.drop_steps,
            "drop_gamma": self.drop_gamma,
            "step_index": self.step_index,
            "group_base_lrs": {
                str(group.get("name", idx)): float(group.get("base_lr", group["lr"]))
                for idx, group in enumerate(self.optimizer.param_groups)
            },
        }

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]


def checkpoint_payload(
    model: nn.Module,
    *,
    model_variant: str,
    epoch: int,
    args: argparse.Namespace,
    weights: LossWeights,
    center_radius_px: float,
    val_metrics: dict[str, float],
    det_metrics: dict[str, object],
) -> dict[str, object]:
    base_model = unwrap_model(model)
    return {
        "model": normalized_state_dict(base_model),
        "EX": base_model.EX.state_dict() if hasattr(base_model, "EX") else None,
        "EN": base_model.EN.state_dict() if hasattr(base_model, "EN") else None,
        "model_variant": model_variant,
        "epoch": epoch,
        "args": vars(args),
        "loss_weights": asdict(weights),
        "center_radius_px": center_radius_px,
        "val": val_metrics,
        "detection": det_metrics,
    }


def normalized_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        str(key).replace("._orig_mod.", "."): value
        for key, value in unwrap_model(model).state_dict().items()
    }


def state_dict_cpu_copy(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in normalized_state_dict(model).items()}
