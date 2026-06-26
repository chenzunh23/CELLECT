# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from typing import Any, Dict, List, Optional, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        astro_preprocess_in_model: bool = False,
        astro_preprocess_clip_sigma: float = 3.0,
        astro_preprocess_sigma_iters: int = -1,
        astro_preprocess_z_clip: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.astro_preprocess_in_model = bool(astro_preprocess_in_model)
        self.astro_preprocess_clip_sigma = float(astro_preprocess_clip_sigma)
        self.astro_preprocess_sigma_iters = int(astro_preprocess_sigma_iters)
        self.astro_preprocess_z_clip = astro_preprocess_z_clip

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        if self.astro_preprocess_in_model:
            x = self.astro_preprocess(x)
        else:
            # Normalize colors
            x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def astro_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Per-image astro normalization immediately before the image encoder."""
        # print(f'[DEBUG] Running astro_preprocess_in_model with clip_sigma={self.astro_preprocess_clip_sigma}, sigma_iters={self.astro_preprocess_sigma_iters}, z_clip={self.astro_preprocess_z_clip}')
        x = x.to(dtype=torch.float32)
        squeeze_batch = x.ndim == 3
        if squeeze_batch:
            work = x.unsqueeze(0)
        elif x.ndim == 4:
            work = x
        else:
            raise ValueError(f"astro_preprocess expects CHW or BCHW input, got shape {tuple(x.shape)}")

        out = torch.empty_like(work)
        for b in range(work.shape[0]):
            for c in range(work.shape[1]):
                vals = work[b, c]
                finite = torch.isfinite(vals)
                if not bool(finite.any()):
                    out[b, c] = torch.zeros_like(vals)
                    continue

                finite_vals = vals[finite]
                raw_median = torch.median(finite_vals)
                raw_sigma = torch.std(finite_vals, unbiased=False)
                if not bool(torch.isfinite(raw_sigma)) or float(raw_sigma) <= 0.0:
                    raw_sigma = torch.ones((), dtype=vals.dtype, device=vals.device)
                # print(f'[DEBUG] astro preprocess clip sigma {self.astro_preprocess_clip_sigma} => clip_hi {raw_median + self.astro_preprocess_clip_sigma * raw_sigma:.6g}')
                clip_hi = raw_median + self.astro_preprocess_clip_sigma * raw_sigma

                clipped_vals = torch.minimum(finite_vals, clip_hi)
                mean, std = self.astro_sigma_clipped_mean_std(clipped_vals, vals)
                if not bool(torch.isfinite(std)) or float(std) <= 0.0:
                    std = torch.ones((), dtype=vals.dtype, device=vals.device)

                safe = torch.where(finite, vals, mean)
                clipped = torch.minimum(safe, clip_hi)
                z = (clipped - mean) / std
                if self.astro_preprocess_z_clip is not None:
                    z = torch.clamp(z, float(self.astro_preprocess_z_clip[0]), float(self.astro_preprocess_z_clip[1]))
                out[b, c] = z

        if squeeze_batch:
            out = out[0]
        return out

    def astro_sigma_clipped_mean_std(
        self,
        clipped_vals: torch.Tensor,
        like: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Astropy sigma_clipped_stats on a bright-clipped 1D tensor."""
        try:
            from astropy.stats import sigma_clipped_stats
        except Exception as exc:
            raise RuntimeError("astro_preprocess_in_model requires astropy for converged sigma-clipped stats.") from exc

        arr = clipped_vals.detach().cpu().numpy().astype(np.float64, copy=False)
        maxiters = None if self.astro_preprocess_sigma_iters < 0 else int(self.astro_preprocess_sigma_iters)
        mean, _median, std = sigma_clipped_stats(
            arr,
            sigma=self.astro_preprocess_clip_sigma,
            maxiters=maxiters,
        )
        if not np.isfinite(mean):
            mean = float(np.nanmean(arr))
        if not np.isfinite(std) or std <= 0:
            std = float(np.nanstd(arr))
        mean_t = torch.tensor(float(mean), dtype=like.dtype, device=like.device)
        std_t = torch.tensor(float(std), dtype=like.dtype, device=like.device)
        return mean_t, std_t
