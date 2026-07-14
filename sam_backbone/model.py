from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .build import SamPerBandImageEncoder, build_per_band_sam_encoder
from .decoder import SamCellectDecoder
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .transformer import TwoWayTransformer


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
        self.supports_processing_ids = True
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
        processing_ids: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        encoded = self.encoder(
            x,
            zscale_cache=zscale_cache,
            input_is_preprocessed=input_is_preprocessed,
            return_input=True,
        )
        features = encoded["features"]
        images = encoded["preprocessed_images"]
        outputs = self.decoder(
            features,
            images=images,
            output_size=tuple(x.shape[-2:]),
            processing_ids=processing_ids,
        )
        outputs["image_embeddings"] = features
        if "style_logit" in encoded:
            outputs["style_logit"] = encoded["style_logit"]
            outputs["style_alpha"] = encoded["style_alpha"]
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
        mask_chunks: list[Tensor] = []
        iou_chunks: list[Tensor] = []
        dense_pe = self.prompt_encoder.get_dense_pe().to(device=flat_embeddings.device, dtype=flat_embeddings.dtype)
        total = int(prompt_batch_indices.numel())
        for start in range(0, total, int(chunk_size)):
            stop = min(start + int(chunk_size), total)
            pos = slice(start, stop)
            batch_chunk = prompt_batch_indices[pos]
            coords = point_coords[pos].unsqueeze(1)
            labels = torch.ones((coords.shape[0], 1), device=coords.device, dtype=torch.int64)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(coords, labels),
                boxes=None if boxes is None else boxes[pos],
                masks=None,
            )
            image_embedding = flat_embeddings[batch_chunk]
            image_pe = dense_pe.expand(image_embedding.shape[0], -1, -1, -1)
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            mask_chunks.append(low_res_masks)
            iou_chunks.append(iou_predictions)
        return torch.cat(mask_chunks, dim=0), torch.cat(iou_chunks, dim=0)


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
    decoder_denoised_film: bool = False,
    encoder_style_prompt: bool = False,
    style_prompt_dim: int = 32,
    style_prompt_layers: Sequence[int] = (2, 5, 8),
    style_adapter_dim: int = 32,
    style_router_temperature: float = 1.0,
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
        style_prompt_enabled=encoder_style_prompt,
        style_prompt_dim=style_prompt_dim,
        style_prompt_layers=style_prompt_layers,
        style_adapter_dim=style_adapter_dim,
        style_router_temperature=style_router_temperature,
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
        use_denoised_film=decoder_denoised_film,
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
