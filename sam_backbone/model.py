from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .build import SamPerBandImageEncoder, build_per_band_sam_encoder
from .decoder import SamCellectDecoder

_SAM_ASTRO_ROOT = Path("/home/czh23/SAM-astro")
if str(_SAM_ASTRO_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAM_ASTRO_ROOT))


def _load_sam_modeling_class(module_name: str, class_name: str) -> type[nn.Module]:
    """Load SAM modeling submodules without importing torchvision-dependent predictor code."""

    package_name = "segment_anything"
    modeling_package = f"{package_name}.modeling"
    package_root = _SAM_ASTRO_ROOT / package_name
    modeling_root = package_root / "modeling"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(package_root)]  # type: ignore[attr-defined]
        sys.modules[package_name] = pkg
    if modeling_package not in sys.modules:
        modeling_pkg = types.ModuleType(modeling_package)
        modeling_pkg.__path__ = [str(modeling_root)]  # type: ignore[attr-defined]
        sys.modules[modeling_package] = modeling_pkg
    full_name = f"{modeling_package}.{module_name}"
    if full_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(full_name, modeling_root / f"{module_name}.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load SAM modeling module {full_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
    return getattr(sys.modules[full_name], class_name)


PromptEncoder = _load_sam_modeling_class("prompt_encoder", "PromptEncoder")
MaskDecoder = _load_sam_modeling_class("mask_decoder", "MaskDecoder")
TwoWayTransformer = _load_sam_modeling_class("transformer", "TwoWayTransformer")


class SamCellect2D(nn.Module):
    """SAM image encoder with dense confidence/shape proposal decoder."""

    def __init__(
        self,
        encoder: SamPerBandImageEncoder,
        decoder: SamCellectDecoder,
        *,
        image_size: int = 512,
        patch_size: int = 16,
        candidate_count: int = 5,
        shape_feature_dim: int = 6,
        enable_matchers: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        image_embedding_size = self.image_size // self.patch_size
        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

    def forward(
        self,
        x: Tensor,
        *,
        zscale_cache: Optional[Tensor] = None,
        input_is_preprocessed: bool = False,
    ) -> Dict[str, Tensor]:
        encoded = self.encoder(
            x,
            zscale_cache=zscale_cache,
            input_is_preprocessed=input_is_preprocessed,
            return_input=True,
        )
        features = encoded["features"]
        images = encoded["preprocessed_images"]
        outputs = self.decoder(features, images=images, output_size=tuple(x.shape[-2:]))
        outputs["image_embeddings"] = features
        return outputs

    def forward_sam_masks(
        self,
        image_embeddings: Tensor,
        prompt_batch_indices: Tensor,
        point_coords: Tensor,
        boxes: Optional[Tensor],
        *,
        multimask_output: bool = True,
        chunk_size: int = 128,
    ) -> Tuple[Tensor, Tensor]:
        """Run SAM mask decoder for many prompts grouped by source image.

        ``image_embeddings`` may be [B, band, 256, h, w] or [B*band, 256, h, w].
        Prompt coordinates and boxes are in the 512x512 image frame. ``boxes``
        may be ``None`` for center-only prompting.
        """

        if prompt_batch_indices.numel() == 0:
            h = self.image_size // 4
            empty_masks = image_embeddings.new_zeros((0, 3 if multimask_output else 1, h, h))
            empty_iou = image_embeddings.new_zeros((0, 3 if multimask_output else 1))
            return empty_masks, empty_iou
        if image_embeddings.ndim == 5:
            flat_embeddings = image_embeddings.reshape(
                image_embeddings.shape[0] * image_embeddings.shape[1],
                *image_embeddings.shape[2:],
            )
        elif image_embeddings.ndim == 4:
            flat_embeddings = image_embeddings
        else:
            raise ValueError(f"image_embeddings must be 4D or 5D, got {tuple(image_embeddings.shape)}")

        prompt_batch_indices = prompt_batch_indices.to(device=flat_embeddings.device, dtype=torch.long)
        point_coords = point_coords.to(device=flat_embeddings.device, dtype=flat_embeddings.dtype)
        if boxes is not None:
            boxes = boxes.to(device=flat_embeddings.device, dtype=flat_embeddings.dtype)
        order = torch.argsort(prompt_batch_indices)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        sorted_batch = prompt_batch_indices[order]
        sorted_points = point_coords[order]
        sorted_boxes = boxes[order] if boxes is not None else None

        mask_chunks: list[Tensor] = []
        iou_chunks: list[Tensor] = []
        dense_pe = self.prompt_encoder.get_dense_pe().to(device=flat_embeddings.device, dtype=flat_embeddings.dtype)
        for flat_index in torch.unique_consecutive(sorted_batch).tolist():
            positions = torch.where(sorted_batch == int(flat_index))[0]
            image_embedding = flat_embeddings[int(flat_index) : int(flat_index) + 1]
            for start in range(0, int(positions.numel()), int(chunk_size)):
                pos = positions[start : start + int(chunk_size)]
                coords = sorted_points[pos].unsqueeze(1)
                labels = torch.ones((coords.shape[0], 1), device=coords.device, dtype=torch.int64)
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=(coords, labels),
                    boxes=None if sorted_boxes is None else sorted_boxes[pos],
                    masks=None,
                )
                low_res_masks, iou_predictions = self.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=dense_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                mask_chunks.append(low_res_masks)
                iou_chunks.append(iou_predictions)
        sorted_masks = torch.cat(mask_chunks, dim=0)
        sorted_ious = torch.cat(iou_chunks, dim=0)
        return sorted_masks[inverse], sorted_ious[inverse]


def build_sam_cellect2d(
    model_type: str = "vit_b",
    *,
    checkpoint: Optional[str | Path] = None,
    num_bands: int = 5,
    image_size: int = 512,
    patch_size: int = 16,
    seg_classes: int = 2,
    confidence_levels: int = 5,
    embedding_dim: int = 64,
    shape_channels: int = 3,
    decoder_channels: Sequence[int] = (256, 128, 64, 32),
    use_cen: bool = True,
    cen_input_image: bool = True,
    cen_width: int = 8,
    candidate_count: int = 5,
    shape_feature_dim: int = 6,
    enable_matchers: bool = False,
    strict: bool = False,
    astro_preprocess_in_model: bool = True,
    astro_preprocess_clip_sigma: float = 3.0,
    astro_preprocess_sigma_iters: int = -1,
    astro_preprocess_z_clip: Optional[Tuple[float, float]] = None,
) -> SamCellect2D:
    encoder = build_per_band_sam_encoder(
        model_type,
        checkpoint=checkpoint,
        num_bands=num_bands,
        image_size=image_size,
        patch_size=patch_size,
        strict=strict,
        astro_preprocess_in_model=astro_preprocess_in_model,
        astro_preprocess_clip_sigma=astro_preprocess_clip_sigma,
        astro_preprocess_sigma_iters=astro_preprocess_sigma_iters,
        astro_preprocess_z_clip=astro_preprocess_z_clip,
    )
    decoder = SamCellectDecoder(
        in_channels=256,
        seg_classes=seg_classes,
        confidence_levels=confidence_levels,
        embedding_dim=embedding_dim,
        shape_channels=shape_channels,
        decoder_channels=decoder_channels,
        use_cen=use_cen,
        cen_input_image=cen_input_image,
        cen_width=cen_width,
    )
    model = SamCellect2D(
        encoder,
        decoder,
        image_size=image_size,
        patch_size=patch_size,
        candidate_count=candidate_count,
        shape_feature_dim=shape_feature_dim,
        enable_matchers=enable_matchers,
    )
    if checkpoint is not None:
        load_sam_prompt_mask_checkpoint(model, checkpoint)
    return model


def load_sam_prompt_mask_checkpoint(model: SamCellect2D, checkpoint: str | Path) -> None:
    state = torch.load(Path(checkpoint).expanduser(), map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    prompt_state = {
        key.removeprefix("prompt_encoder."): value
        for key, value in state.items()
        if str(key).startswith("prompt_encoder.")
    }
    mask_state = {
        key.removeprefix("mask_decoder."): value
        for key, value in state.items()
        if str(key).startswith("mask_decoder.")
    }
    if prompt_state:
        model.prompt_encoder.load_state_dict(prompt_state, strict=True)
    if mask_state:
        model.mask_decoder.load_state_dict(mask_state, strict=True)
