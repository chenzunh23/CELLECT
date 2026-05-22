"""
Train and evaluate the 2D CELLECT-style astronomy model on LSST/HSC FITS cutouts.

The dense losses intentionally keep CELLECT's original constants.  Helper code
for dataset construction, losses, detection, and epoch execution lives in
astro_train_data.py and astro_train_ops.py so this file stays focused on CLI,
model construction, and distributed training orchestration.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from astro_cellect2d import AstroUNet2D, FusedEncoderMultiBandAstroCELLECT2D, MultiBandAstroCELLECT2D
from astro_match_eval import parse_ex_band_pairs as parse_matcher_ex_band_pairs
from astro_train_data import (
    AstroCutoutDataset,
    CutoutRecord,
    _expand_path,
    _record_name_aliases,
    collate_cutouts,
    discover_cutout_records,
    load_meas_catalog,
    make_targets,
    split_records,
)
from astro_train_ops import (
    HardTripletLoss,
    LossWeights,
    build_cellect_style_segmentation,
    center_localization_loss,
    detect_centers,
    detect_centers_with_en,
    detect_centers_with_ex_link,
    dense_losses,
    dense_losses_any,
    embedding_triplet_loss,
    en_deduplicate_centers,
    evaluate_detection,
    match_points,
    matcher_classification_loss,
    parse_ex_band_pairs,
    run_epoch,
    unwrap_model,
)


class _DDPModelWrapper(nn.Module):
    """Expose matcher parameters to DDP even though EX/EN are used in the loss."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._ddp_wrapped_model = model

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self._ddp_wrapped_model(image)
        dummy = image.new_zeros(())
        for name in ("EX", "EN"):
            matcher = getattr(self._ddp_wrapped_model, name, None)
            if matcher is None:
                continue
            for param in matcher.parameters():
                dummy = dummy + param.sum() * 0.0
        if dummy.requires_grad:
            outputs = dict(outputs)
            outputs["seg_logits"] = outputs["seg_logits"] + dummy
        return outputs


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _distributed_requested(args: argparse.Namespace) -> bool:
    return bool(args.ddp) or _env_int("WORLD_SIZE", 1) > 1


def _setup_distributed(args: argparse.Namespace) -> Tuple[bool, int, int, int, torch.device, str]:
    distributed = _distributed_requested(args)
    local_rank = _env_int("LOCAL_RANK", args.local_rank)
    backend = args.dist_backend
    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() and torch.device(args.device).type == "cuda" else "gloo"

    if distributed:
        dist.init_process_group(backend=backend, init_method=args.dist_url)
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


def _is_main(rank: int) -> bool:
    return int(rank) == 0


def _sync_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _wrap_ddp(model: nn.Module, args: argparse.Namespace, device: torch.device, local_rank: int) -> nn.Module:
    if device.type == "cuda":
        device_ids = [local_rank]
        output_device = local_rank
    else:
        device_ids = None
        output_device = None
    return DistributedDataParallel(
        _DDPModelWrapper(model),
        device_ids=device_ids,
        output_device=output_device,
        find_unused_parameters=False,
    )


def _parse_patch_specs(specs: Iterable[str], patch_file: str | None = None) -> set[str]:
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


def _record_patch_aliases(rec: CutoutRecord, root: Path) -> set[str]:
    aliases: set[str] = set()
    if rec.patch:
        aliases.add(rec.patch)
        if rec.tract:
            aliases.add(f"{rec.tract}/{rec.patch}")
        elif root.name:
            aliases.add(f"{root.name}/{rec.patch}")
    if rec.relative_root:
        aliases.add(str(rec.relative_root).strip("/"))
    parts = Path(rec.relative_root).parts
    if len(parts) >= 2:
        aliases.add(f"{parts[-2]}/{parts[-1]}")
        aliases.add(str(parts[-1]))
    return aliases


def _filter_records_by_patches(records: Sequence[CutoutRecord], patches: set[str], root: Path) -> list[CutoutRecord]:
    if not patches:
        return list(records)
    wanted = {patch.strip("/") for patch in patches if patch.strip("/")}
    return [rec for rec in records if _record_patch_aliases(rec, root) & wanted]


def _record_patch_label(rec: CutoutRecord, root: Path) -> str:
    if rec.tract and rec.patch:
        return f"{rec.tract}/{rec.patch}"
    if rec.patch and root.name:
        return f"{root.name}/{rec.patch}"
    return rec.patch or rec.relative_root or "-"


def _checkpoint_payload(
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
        "model": base_model.state_dict(),
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate AstroCELLECT2D on LSST/HSC FITS cutouts.")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument(
        "--root",
        default="~/segment-anything/lsst_pipeline/output/cutout_magnitude_experiment_grid",
        help="Root containing cutouts/ and reference_catalogs/, or <tract>/<patch>/cutouts and reference_catalogs.",
    )
    parser.add_argument("--reference-dir", default=None)
    parser.add_argument("--cutout-dir", default=None)
    parser.add_argument(
        "--band-reference-root",
        default=None,
        help="Optional root containing per-band meas catalogs, e.g. catalog/HSC-I/meas-HSC-I-*.fits. "
        "When set, each band uses its own catalog ids/centers for EX/EN training.",
    )
    parser.add_argument(
        "--targets-dir",
        default=None,
        help="Optional directory containing precomputed <tile>.npz dense targets from astro_data_preprocessing.py.",
    )
    parser.add_argument("--bands", nargs="+", default=("HSC-G", "HSC-R", "HSC-I"))
    parser.add_argument("--fits-hdu", type=int, default=1)
    parser.add_argument("--out-dir", default="./output/astro_cellect2d")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument(
        "--model-variant",
        choices=("auto", "fused", "per_band", "fused_encoder"),
        default="auto",
        help="fused treats bands as channels and outputs one map. per_band runs one shared single-band backbone per band. "
        "fused_encoder runs one multi-band encoder and lightweight per-band heads for EX/EN. "
        "auto uses fused_encoder for multi-band data or when EN is enabled.",
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--eval-patches",
        nargs="*",
        default=(),
        help="Restrict --mode eval to specific patches. Accepts patch names like 8,8 or tract/patch like 9813/8,8.",
    )
    parser.add_argument(
        "--eval-patches-file",
        default=None,
        help="Optional text file with one eval patch per line. Lines may use 8,8 or 9813/8,8; # comments are ignored.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--fixed-val-names",
        nargs="*",
        default=("sam_x18204_y20924",),
        help="Tile names that must be placed in the validation set. Default keeps the SAM comparison cutout in validation.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ellipse-sigma", type=float, default=2.0)
    parser.add_argument("--core-radius", type=int, default=2)
    parser.add_argument(
        "--shape-source",
        choices=("kron", "sdss", "circular_kron"),
        default="kron",
        help="Pseudo-mask ellipse source: Kron radius plus moment axis ratio, pure Sdss moments, or circular Kron.",
    )
    parser.add_argument(
        "--source-filter",
        choices=("nchild0", "all", "parent", "leaf_child"),
        default="nchild0",
        help="Catalog rows used as center/embedding/eval GT. Default is deblend_nChild==0 leaf sources.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument(
        "--nms-radius",
        type=int,
        default=1,
        help="Local-max suppression radius. CELLECT uses kernel_size=3, equivalent to radius=1.",
    )
    parser.add_argument(
        "--confidence-score",
        choices=("cellect", "raw", "ordinal_prob"),
        default="cellect",
        help="Score used for center detection. cellect applies DK1 smoothing plus kernel_size=3 local-max logic from CELLECT.",
    )
    parser.add_argument("--center-tolerance-arcsec", type=float, default=0.5)
    parser.add_argument("--pixel-scale-arcsec", type=float, default=0.168)
    parser.add_argument(
        "--match-radius",
        type=float,
        default=None,
        help="Center matching radius in pixels. Defaults to center_tolerance_arcsec / pixel_scale_arcsec.",
    )
    # Shut down the time consuming center loss by default since confidence map supervision is usually sufficient for good center detection, and the center loss can be a significant bottleneck when training with many small sources.
    parser.add_argument("--center-loss-weight", type=float, default=0.0)
    parser.add_argument("--segmentation-class-weights", type=float, nargs=3, default=(1.0, 32.0, 1.0))
    parser.add_argument("--confidence-pos-weight", type=float, default=32.0)
    parser.add_argument("--triplet-outer-weight", type=float, default=10.0)
    parser.add_argument("--enable-triplet", action="store_true")
    parser.add_argument(
        "--triplet-max-sources-per-group",
        type=int,
        default=256,
        help="Maximum unique source IDs per image/group used by hard triplet mining. "
        "Set <=0 to use all sources. Dense losses and EX/EN losses are unchanged."
        "IMPORTANT: Sampled based on tile/group even when --triplet-negative-scope is batch.",
    )
    parser.add_argument(
        "--triplet-negative-scope",
        choices=("tile", "batch"),
        default="batch",
        help="Hard-triplet negative mining scope. batch uses all selected sources in the local batch as negatives; "
        "tile keeps the previous behavior and only mines negatives inside each cutout.",
    )
    parser.add_argument(
        "--disable-ex-loss",
        action="store_true",
        help="Disable EX cross-band classification loss. By default EX is trained for per_band multi-band runs.",
    )
    parser.add_argument(
        "--ex-core-band",
        default="HSC-I",
        help="Core band used as EX anchor when --ex-band-pairs is not set. Short names like I are accepted.",
    )
    parser.add_argument(
        "--ex-band-pairs",
        nargs="*",
        default=None,
        help="Directed EX training pairs such as HSC-I:HSC-G HSC-I:HSC-R. "
        "Use 'all' to restore all directed cross-band pairs.",
    )
    parser.add_argument(
        "--enable-en-loss",
        action="store_true",
        help="Train EN same-band duplicate/none-of-above classification loss. Useful for single-band duplicate suppression too.",
    )
    parser.add_argument(
        "--use-en-postprocess",
        action="store_true",
        help="Apply CELLECT-style EN same-band deduplication after detect_centers during evaluation/inference.",
    )
    parser.add_argument("--en-postprocess-threshold", type=float, default=0.6)
    parser.add_argument(
        "--use-ex-link-postprocess",
        action="store_true",
        help="Apply CELLECT-style EX cross-band linking after per-band center detection. "
        "When enabled, detection metrics are object-level instead of band-level.",
    )
    parser.add_argument("--ex-link-threshold", type=float, default=0.5)
    parser.add_argument("--matcher-candidate-count", type=int, default=5)
    parser.add_argument(
        "--matcher-max-anchors-per-band",
        type=int,
        default=128,
        help="Maximum source anchors per batch item and band for EX/EN classification loss. "
        "Set <=0 to use all anchors.",
    )
    parser.add_argument("--matcher-outer-weight", type=float, default=10.0)
    parser.add_argument(
        "--image-cache-dir",
        default=None,
        help="Optional directory for zscale-preprocessed CHW tensors. Multi-patch caches keep <tract>/<patch>/cutouts.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Enable DistributedDataParallel. Also auto-enables when launched with torchrun WORLD_SIZE>1.",
    )
    parser.add_argument(
        "--dist-backend",
        choices=("auto", "nccl", "gloo"),
        default="auto",
        help="Distributed backend. auto uses nccl for CUDA and gloo for CPU.",
    )
    parser.add_argument(
        "--patch-val",
        action="store_true",
        help="Split training and validation by random patch instead of random cutout. This is a coarser split that may better reflect generalization to new sky areas, but results in higher variance.",
    )
    parser.add_argument("--dist-url", default="env://", help="Distributed init method, normally env:// for torchrun.")
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank fallback when LOCAL_RANK is not set.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    distributed, rank, world_size, local_rank, device, backend = _setup_distributed(args)
    is_main = _is_main(rank)

    try:
        seed = int(args.seed) + int(rank)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        root = _expand_path(args.root)
        reference_dir = _expand_path(args.reference_dir) if args.reference_dir else None
        cutout_dir = _expand_path(args.cutout_dir) if args.cutout_dir else None
        band_reference_root = _expand_path(args.band_reference_root) if args.band_reference_root else None
        targets_dir = _expand_path(args.targets_dir) if args.targets_dir else (root / "targets")
        if not targets_dir.exists():
            targets_dir = None
        image_cache_dir = _expand_path(args.image_cache_dir) if args.image_cache_dir else None
        out_dir = _expand_path(args.out_dir)
        if is_main:
            out_dir.mkdir(parents=True, exist_ok=True)
        _sync_distributed()

        if args.mode == "eval" and distributed and not is_main:
            _sync_distributed()
            return

        eval_patch_specs = _parse_patch_specs(args.eval_patches, args.eval_patches_file)
        records = discover_cutout_records(
            root,
            reference_dir=reference_dir,
            cutout_dir=cutout_dir,
            band_reference_root=band_reference_root,
            bands=args.bands,
            max_records=None if args.mode == "eval" and eval_patch_specs else args.max_records,
        )
        if args.mode == "eval" and eval_patch_specs:
            before_count = len(records)
            records = _filter_records_by_patches(records, eval_patch_specs, root)
            if args.max_records is not None:
                records = records[: int(args.max_records)]
            if not records:
                raise RuntimeError(f"No eval records matched requested patches: {sorted(eval_patch_specs)}")
            if is_main:
                selected_patches = sorted({_record_patch_label(rec, root) for rec in records})
                print(
                    f"Eval patch filter selected {len(records)} / {before_count} records "
                    f"from patches: {selected_patches}"
                )
        train_records, val_records = split_records(
            records,
            args.val_fraction,
            args.seed,
            fixed_val_names=args.fixed_val_names,
            patch_val=args.patch_val,
        )
        available_record_names = set()
        for rec in records:
            available_record_names.update(_record_name_aliases(rec))
        missing_fixed_val = sorted(set(args.fixed_val_names) - available_record_names)
        if missing_fixed_val and is_main:
            print(f"WARNING: fixed validation tile(s) not found and cannot be forced into val: {missing_fixed_val}")
        if args.mode == "eval":
            val_records = records

        center_radius_px = (
            float(args.match_radius)
            if args.match_radius is not None
            else float(args.center_tolerance_arcsec) / max(float(args.pixel_scale_arcsec), 1e-12)
        )
        weights = LossWeights(
            segmentation_3cls=tuple(float(v) for v in args.segmentation_class_weights),
            confidence_pos_weight=float(args.confidence_pos_weight),
            center_position=float(args.center_loss_weight),
            triplet_outer_weight=float(args.triplet_outer_weight),
            matcher_outer_weight=float(args.matcher_outer_weight),
        )
        common_ds = dict(
            fits_hdu=args.fits_hdu,
            confidence_levels=5,
            ellipse_sigma=args.ellipse_sigma,
            core_radius=args.core_radius,
            shape_source=args.shape_source,
            source_filter=args.source_filter,
            targets_dir=targets_dir,
            image_cache_dir=image_cache_dir,
        )
        train_ds = AstroCutoutDataset(train_records, augment=True, **common_ds)
        val_ds = AstroCutoutDataset(val_records, augment=False, **common_ds)
        train_sampler = (
            DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
            if distributed and args.mode == "train"
            else None
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_cutouts,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_cutouts,
        )

        if args.model_variant == "auto":
            model_variant = "fused_encoder" if len(args.bands) > 1 or args.enable_en_loss else "fused"
        else:
            model_variant = args.model_variant
        matcher_variant = model_variant in ("per_band", "fused_encoder")
        ex_enabled = matcher_variant and len(args.bands) > 1 and not args.disable_ex_loss
        en_enabled = matcher_variant and bool(args.enable_en_loss)
        en_postprocess_enabled = matcher_variant and (bool(args.use_en_postprocess) or en_enabled)
        ex_link_postprocess_enabled = matcher_variant and len(args.bands) > 1 and bool(args.use_ex_link_postprocess)
        ex_band_pairs = (
            parse_matcher_ex_band_pairs(args.bands, core_band=args.ex_core_band, pair_specs=args.ex_band_pairs)
            if ex_enabled
            else tuple()
        )

        if model_variant == "per_band":
            model = MultiBandAstroCELLECT2D(
                num_bands=len(args.bands),
                seg_classes=3,
                confidence_levels=5,
                embedding_dim=args.embedding_dim,
                base_channels=args.base_channels,
                shape_channels=3,
                candidate_count=args.matcher_candidate_count,
                shape_feature_dim=6,
            ).to(device)
        elif model_variant == "fused_encoder":
            model = FusedEncoderMultiBandAstroCELLECT2D(
                num_bands=len(args.bands),
                seg_classes=3,
                confidence_levels=5,
                embedding_dim=args.embedding_dim,
                base_channels=args.base_channels,
                shape_channels=3,
                candidate_count=args.matcher_candidate_count,
                shape_feature_dim=6,
            ).to(device)
        else:
            model = AstroUNet2D(
                in_channels=len(args.bands),
                seg_classes=3,
                confidence_levels=5,
                embedding_dim=args.embedding_dim,
                base_channels=args.base_channels,
                shape_channels=3,
            ).to(device)

        if args.checkpoint:
            ckpt = torch.load(_expand_path(args.checkpoint), map_location=device)
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            model.load_state_dict(state)
            if hasattr(model, "EX") and hasattr(model, "EN"):
                if isinstance(ckpt, dict) and ckpt.get("EX") is not None:
                    model.EX.load_state_dict(ckpt["EX"])
                if isinstance(ckpt, dict) and ckpt.get("EN") is not None:
                    model.EN.load_state_dict(ckpt["EN"])

        if is_main and distributed:
            print(
                f"DDP enabled: backend={backend}, world_size={world_size}, "
                f"rank={rank}, local_rank={local_rank}, device={device}"
            )

        if args.mode == "eval":
            dense = run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                weights=weights,
                triplet_loss_fn=HardTripletLoss(weights.triplet_margin),
                triplet_enabled=args.enable_triplet,
                ex_enabled=ex_enabled,
                en_enabled=en_enabled,
                matcher_candidate_count=args.matcher_candidate_count,
                matcher_max_anchors_per_band=args.matcher_max_anchors_per_band,
                triplet_max_sources_per_group=args.triplet_max_sources_per_group,
                triplet_negative_scope=args.triplet_negative_scope,
                ex_band_pairs=ex_band_pairs,
                center_radius_px=center_radius_px,
                show_progress=is_main,
            )
            det = evaluate_detection(
                model,
                val_loader,
                device=device,
                threshold=args.confidence_threshold,
                nms_radius=args.nms_radius,
                confidence_score=args.confidence_score,
                match_radius=center_radius_px,
                use_en_postprocess=en_postprocess_enabled,
                en_candidate_count=args.matcher_candidate_count,
                en_threshold=args.en_postprocess_threshold,
                use_ex_link_postprocess=ex_link_postprocess_enabled,
                ex_link_threshold=args.ex_link_threshold,
                ex_band_pairs=ex_band_pairs,
                band_names=args.bands,
                show_progress=is_main,
            )
            if is_main:
                print(json.dumps({"dense": dense, "detection": det}, indent=2))
            _sync_distributed()
            return

        if distributed:
            model = _wrap_ddp(model, args, device, local_rank)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
        triplet_loss_fn = HardTripletLoss(weights.triplet_margin)

        metadata = {
            "args": vars(args),
            "loss_weights": asdict(weights),
            "num_records": len(records),
            "num_train": len(train_records),
            "num_val": len(val_records),
            "fixed_val_names": list(args.fixed_val_names),
            "val_record_names": [rec.name for rec in val_records],
            "center_radius_px": center_radius_px,
            "targets_dir": str(targets_dir) if targets_dir is not None else None,
            "image_cache_dir": str(image_cache_dir) if image_cache_dir is not None else None,
            "band_reference_root": str(band_reference_root) if band_reference_root is not None else None,
            "model_variant": model_variant,
            "distributed": distributed,
            "world_size": world_size,
            "dist_backend": backend,
            "ex_enabled": ex_enabled,
            "ex_band_pairs": [[str(args.bands[src]), str(args.bands[dst])] for src, dst in ex_band_pairs],
            "matcher_max_anchors_per_band": int(args.matcher_max_anchors_per_band),
            "triplet_negative_scope": str(args.triplet_negative_scope),
            "en_enabled": en_enabled,
            "en_postprocess_enabled": en_postprocess_enabled,
            "ex_link_postprocess_enabled": ex_link_postprocess_enabled,
            "ex_link_threshold": float(args.ex_link_threshold),
        }
        if is_main:
            (out_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        _sync_distributed()

        best_val = float("inf")
        for epoch in range(args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_metrics = run_epoch(
                model,
                train_loader,
                optimizer=optimizer,
                device=device,
                weights=weights,
                triplet_loss_fn=triplet_loss_fn,
                triplet_enabled=args.enable_triplet,
                ex_enabled=ex_enabled,
                en_enabled=en_enabled,
                matcher_candidate_count=args.matcher_candidate_count,
                matcher_max_anchors_per_band=args.matcher_max_anchors_per_band,
                triplet_max_sources_per_group=args.triplet_max_sources_per_group,
                triplet_negative_scope=args.triplet_negative_scope,
                ex_band_pairs=ex_band_pairs,
                center_radius_px=center_radius_px,
                distributed=distributed,
                show_progress=is_main,
            )

            if is_main:
                eval_model = unwrap_model(model)
                val_metrics = run_epoch(
                    eval_model,
                    val_loader,
                    optimizer=None,
                    device=device,
                    weights=weights,
                    triplet_loss_fn=triplet_loss_fn,
                    triplet_enabled=args.enable_triplet,
                    ex_enabled=ex_enabled,
                    en_enabled=en_enabled,
                    matcher_candidate_count=args.matcher_candidate_count,
                    matcher_max_anchors_per_band=args.matcher_max_anchors_per_band,
                    triplet_max_sources_per_group=args.triplet_max_sources_per_group,
                    triplet_negative_scope=args.triplet_negative_scope,
                    ex_band_pairs=ex_band_pairs,
                    center_radius_px=center_radius_px,
                    show_progress=True,
                )
                det_metrics = evaluate_detection(
                    eval_model,
                    val_loader,
                    device=device,
                    threshold=args.confidence_threshold,
                    nms_radius=args.nms_radius,
                    confidence_score=args.confidence_score,
                    match_radius=center_radius_px,
                    use_en_postprocess=en_postprocess_enabled,
                    en_candidate_count=args.matcher_candidate_count,
                    en_threshold=args.en_postprocess_threshold,
                    use_ex_link_postprocess=ex_link_postprocess_enabled,
                    ex_link_threshold=args.ex_link_threshold,
                    ex_band_pairs=ex_band_pairs,
                    band_names=args.bands,
                    show_progress=True,
                )
                val_total = float(val_metrics["total"])
            else:
                val_metrics = {"total": 0.0}
                det_metrics = {}
                val_total = 0.0

            val_total_tensor = torch.tensor([val_total], device=device, dtype=torch.float64)
            if distributed:
                dist.broadcast(val_total_tensor, src=0)
            scheduler.step(float(val_total_tensor.item()))

            if is_main:
                log_line = {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                    "detection": det_metrics,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                print(json.dumps(log_line, indent=2))

                ckpt = _checkpoint_payload(
                    model,
                    model_variant=model_variant,
                    epoch=epoch,
                    args=args,
                    weights=weights,
                    center_radius_px=center_radius_px,
                    val_metrics=val_metrics,
                    det_metrics=det_metrics,
                )
                torch.save(ckpt, out_dir / "last.pt")
                if val_metrics["total"] < best_val:
                    best_val = val_metrics["total"]
                    torch.save(ckpt, out_dir / "best.pt")
            _sync_distributed()
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    main()
