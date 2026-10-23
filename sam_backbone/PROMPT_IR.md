# Source decoder PromptIR

`prompt_ir.py` implements image-conditioned, randomly initialized spatial prompt
banks with softmax routing and lightweight channel-attention/FFN interaction.
This is a PromptIR-style adaptation, not an exact copy of the restoration model.
Only the existing detection/shape/mask objectives train it; no metadata inputs,
auxiliary prediction heads, or new losses are required.

Each selected source stage has its own prompt bank. A shared condition encoder
pools mean/std statistics from SAM neck features before source InstanceNorm.
A second, shallow CNN reads the scaled input image, applies convolutions at
full resolution, and then builds a feature pyramid. At each enabled stage,
independently projected decoder/image features are combined at matching pixel
locations, followed by depthwise convolution, masked mean/std pooling, and a
small MLP. Its routing logits are added to the existing decoder/neck logits
before softmax selects the prompt mixture.

The new image features only affect prompt selection. They are never added or
concatenated to the main decoder feature stream; there is no spatial skip.
There are no PSF/metadata inputs or auxiliary supervision heads. All new layers
are trained by the existing task losses, in the `prompt_ir` LR group. A small
nonzero residual scale and nonzero output projection allow gradients to reach
the bank/router from the first update. All prompt modules, including the
image and neck condition encoders, have random initialization.

Default image-pyramid widths are 16/24/32/48/64 at input strides 1/2/4/8/16.
It computes only through the coarsest enabled stage. Bottom/right padding
matches the source decoder canvas, with no resizing of the input image.
Known nonfinite pixels and padded cells are excluded from image-condition
statistics. `SamCellect2D` preserves input validity before the SAM encoder
sanitizes NaNs; direct decoder callers can supply `image_valid_mask`. A NaN
already replaced with zero upstream is not recoverable as invalid from the
image alone. The legacy neck-condition pooling is unchanged.

## Positions

Available positions: `neck stem up1 up2 up3 up4 refine`.
For a 512-pixel input their resolutions are respectively
32, 32, 64, 128, 256, 512, 512 pixels. Defaults are `stem up1 up2`.
`up3` and `up4` are supported; their larger feature maps increase computation
and activation memory. Comparing the default against all four upsampling
stages is an experiment, not an assumption about which works better.

`neck` adapts a source-side copy of the encoder output. It does not change the
encoder or the image embeddings passed directly to SAM mask decoding.
Mask losses can still affect source prompts through existing proposal paths
where those paths are differentiable. Nothing is inserted inside SAM's mask
decoder or prompt encoder. The separate image-condition encoder is part of
this module, not an auxiliary prediction head.

## Run

Keep your usual dataset/checkpoint/device settings and prepend:

```bash
PROMPT_IR=1 PROMPT_IR_LR=0.00025 \
  bash scripts/train_mixed_zarr.sh
```

To include up3/up4:

```bash
PROMPT_IR=1 PROMPT_IR_LR=0.00025 \
PROMPT_IR_STAGES="stem up1 up2 up3 up4" \
  bash scripts/train_mixed_zarr.sh
```

The image branch is enabled by default when creating a new PromptIR module.
To explicitly enable it, including when warm-starting an older neck-only
PromptIR checkpoint:

```bash
PROMPT_IR=1 PROMPT_IR_IMAGE_CONDITIONING=1 \
PROMPT_IR_STAGES="stem up1 up2 up3 up4" \
PROMPT_IR_LR=0.00025 \
  bash scripts/train_mixed_zarr.sh --checkpoint /path/to/old_cellect.pt
```

Optional environment variables and Python equivalents:

| Environment | Python flag | Default |
|---|---|---|
| `PROMPT_IR` | `--sam-prompt-ir` / `--no-sam-prompt-ir` | Infer from checkpoint; off for old checkpoints/new models |
| `PROMPT_IR_STAGES` | `--sam-prompt-ir-stages` | `stem up1 up2` |
| `PROMPT_IR_LR` | `--sam-prompt-ir-lr` | 5 × max(head LR, encoder LR) |
| `PROMPT_IR_COMPONENTS` | `--sam-prompt-ir-components` | 5 |
| `PROMPT_IR_DIM` | `--sam-prompt-ir-dim` | 32 |
| `PROMPT_IR_SIZE` | `--sam-prompt-ir-size` | 16 |
| `PROMPT_IR_SCALE` | `--sam-prompt-ir-scale` | 0.1, learned after initialization |
| `PROMPT_IR_IMAGE_CONDITIONING` | `--sam-prompt-ir-image-conditioning` / `--no-sam-prompt-ir-image-conditioning` | On for new modules; infer from PromptIR checkpoints |
| `PROMPT_IR_IMAGE_CHANNELS` | `--sam-prompt-ir-image-channels` | 16 |

Example: head LR `5e-5`, encoder LR `1e-5` gives prompt LR `2.5e-4`.
The entire module has a separate optimizer group named `prompt_ir`, with the
same warmup/step-decay schedule and weight decay as the other groups. Existing
per-group LR logging includes it. When phase 2 changes the head LR, the prompt
LR follows at its original ratio; freezing the source decoder also freezes
its prompts. A higher LR does not guarantee useful routing: compare held-out
detection/mask results with a no-prompt baseline.

## Checkpoints and evaluation

Old CELLECT checkpoints can warm-start the new modules with `--checkpoint ...
--sam-prompt-ir`; the omitted prompt parameters are randomly initialized.
Existing runs remain unchanged when the module is off.

Older neck-only PromptIR checkpoints are reconstructed without the image branch
unless `--sam-prompt-ir-image-conditioning` explicitly enables the upgrade in
training. Their original router shapes and base architecture descriptor remain
unchanged; upgrading only adds randomly initialized image-pyramid/conditioning
parameters and an image-architecture descriptor. New image-conditioned
checkpoints reload the branch automatically in training and evaluation.

New checkpoints store both prompt weights and a tensor architecture descriptor.
Training, the central eval loader used by the browser/visualizer, and SAM demo
exporters infer positions/dimensions from the state dictionary. A separate
`run_config.json` is not needed to reconstruct prompt geometry. Changing prompt
geometry when loading trained prompts can produce incompatible parameter sizes;
use the saved geometry unless intentionally starting a new adaptation.

CPU tests in `tests/test_prompt_ir.py` cover all positions, first-update gradients
in both branches, independent image influence with fixed decoder/neck features,
the absence of a spatial skip, masking/padding, optimizer separation, scheduler
ratios, old-checkpoint compatibility/upgrading, exact prompt checkpoint reload,
source-only neck behavior, dynamic rectangular/RGB inputs, bfloat16, and eager
`torch.compile` graphs. They do not measure GPU memory/throughput or scientific
accuracy; those require an actual training comparison.

## DDP and throughput

`DDP_FIND_UNUSED=0` (Python: `--no-ddp-find-unused-parameters`) explicitly
disables DDP's unused-parameter search without enabling `static_graph`.
`DDP_FIND_UNUSED=1` enables it. Unset/`auto` preserves the existing policy:
search when static graph is off. Only explicitly disable the search for a
configuration whose trainable parameters remain connected to the backward
graph. The existing SAM wrapper adds zero-gradient anchors for loss-only
prompt/mask modules, including mask warmup; tests also cover proposal gradient
clearing. Other model/configuration changes can alter parameter coverage.
This flag takes effect on the next launch and does not modify a running job.

Small parameter count does not imply a cheap high-resolution interaction block.
CPU profiler operator counts for a single 512x512 image and the default source
decoder (forward only; supported FLOP-counted operators) are:

| Source decoder | GFLOPs |
|---|---:|
| PromptIR off | 45.584 |
| PromptIR at stem/up1/up2, image branch on | 50.394 |
| PromptIR at stem/up1/up2/up3/up4, image branch on | 92.735 |

These counts exclude the SAM encoder, SAM mask decoder, backward, and uncounted
normalization/elementwise operations. They are not a GPU latency benchmark.
Stage changes should be tested as a new architecture/ablation: removing stages
from an existing checkpoint is not an exact resume. Partial Zarr batches can
also trigger compile specialization, but `drop_last` can remove large-source
special-view samples, so do not enable it solely to silence a compile warning.
