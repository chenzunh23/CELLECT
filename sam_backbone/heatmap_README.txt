Native-resolution heatmap proposal experiment (2026-11-08)

Entry: scripts/train_mixed_zarr.sh, PROPOSAL_MODE=heatmap.
Default remains the existing five-level ordinal proposal when the option is
absent. A saved heatmap checkpoint is detected from its parameter names by
both astro_train_eval.py and the evaluation/browser loader.

Targets and losses
- Rebuild GT AFTER augmentation from saved floating source centers and strict
  centers. Existing Zarr images, source filters, source reliability/area weights,
  PU regions, splits and old ordinal arrays are retained. No Zarr rebuild needed.
- Fixed native-pixel Gaussian sigma=1.5; amplitude=1 at each selected anchor.
  Per-pixel max merge, circular cutoff at 4*sigma, zero outside all supports.
- Anchor is nearest integer pixel. Within 0.05px of a half-integer boundary,
  choose the brighter adjacent valid pixel (mean training RGB intensity); exact
  brightness ties use nearest distance. The tolerance is configurable.
- Regional confidence weights follow preprocessing.labels.LabelWeights. A
  trusted source Gaussian overrides regional ignore inside its support. Known
  invalid input pixels are not assigned new labels; previously allowed repaired
  positive centers keep their confidence support. Shape requires observed data.
- Offset: 5x5 window around selected anchor, target = floating_GT - query_pixel.
  Shape: 3x3 window, target = source ellipse attributes. Both use the nearest
  FLOATING trusted GT center as owner. Equal squared distances (within 1e-4 px^2)
  skip regression. Strict sources participate in ownership and offset, but do
  not inherit a neighbor's shape. Local losses average within each source first.
- Soft BCE is default. Focal is the CenterNet modified focal (alpha=2,beta=4):
  only target==1 is positive; Gaussian shoulders soften negative penalties.
  Its absolute loss scale is different from BCE, especially on empty images.
- Outputs: confidence is now ONE logit channel for compatibility; center_head,
  offset_head and shape head have independent final branches. Shape receives
  decoder features only, never center logits. Offset final weights/bias start
  at zero and its linear output is not restricted to +/-0.5px.
- Existing --confidence-loss-weight and --shape-loss-weight remain effective;
  --offset-loss-weight defaults to 1. Existing weak-shape factor remains .2.
  W&B loss/confidence is heatmap loss, loss/offset is local offset regression.
  Legacy center-localization and ordinal tiny-shape suppression are disabled.

PSF-EE sigma option
HEATMAP_SIGMA_MODE=psf-ee. For each EE fraction f in {0.10,0.35,0.60,0.70},
compute sigma_f = r_f / sqrt(-2*log(1-f)); take the median and clip to
[2/3,8/3] NATIVE pixels. This defines an equivalent Gaussian of the existing
EE calibration, not a fitted Gaussian image PSF or a new predicted PSF head.
JWST uses existing EE assets and the input FITS pixel scale. HSC uses the
store's PSF-EE metadata, or HEATMAP_PSF_EE_ROOT/<band>/<patch>.json. Missing
calibration fails preflight; no nominal or silently mixed fallback is used.
A compatible existing HSC cache is:
/data/czh23/analysis/2026-10/2026-10-21/training_zarr_gaia_star_wings/psf_ee
This follows the earlier full-reference CoaddPsf convention; it is not a new
measurement of the PSF of each noisy realization.

Inference
Sigmoid >0.5, 3x3 max-pooling NMS. Exactly equal-valued, 8-connected peak
plateaus produce one detection: average the valid offset-corrected votes.
Disconnected peaks and unequal scores are not merged. The original peak whose
corrected vote is nearest the centroid supplies shape; row-major order breaks
exact ties. Invalid/out-of-bounds votes are excluded; empty groups are dropped.
There is no ordinal smoothing, class-4 region merging, 3px distance merging,
or mask-IoU veto. Shape is NOT resampled at the shifted center; evaluation
exports and SAM predicted prompts retain the representative original anchor.
With the requested 5x5 offset / 3x3 shape setting, the outer regression ring
trains offset only. A peak displaced by two pixels may recover its position
without a directly supervised shape at that original cell. --shape-center-size
5 is available as a separate ablation; the requested default remains 3.
Centers quantized to the same anchor cannot both be represented by one peak;
local equal-distance regression is skipped, never averaged across sources.
Browser confidence colors are display bins at .1/.25/.5/.75; scores and saved
logits remain continuous and detection uses .5. These are not ordinal classes.

Controlled experiments (keep ROOT, splits, mask settings, optimizer and steps
identical across all three). Existing shell script supports these environment
variables; equivalent --proposal-mode/--heatmap-* CLI flags also work:

1. PROPOSAL_MODE=heatmap HEATMAP_LOSS=soft_bce HEATMAP_SIGMA=1.5
2. PROPOSAL_MODE=heatmap HEATMAP_LOSS=focal HEATMAP_SIGMA=1.5
3. Add IMAGE_FUSION=1 to the winning configuration. This optional
   narrow image CNN adds /4,/2,/1 features to source up2/up3/up4 only. It is
   disabled for experiments 1 and 2. SAM encoder and mask embeddings are unchanged.

IMAGE_FUSION is independent of the proposal mode and works with existing Zarr
stores, including five-level confidence labels. For the old confidence control,
use PROPOSAL_MODE=ordinal IMAGE_FUSION=1; its targets, CE loss and detection
thresholds stay unchanged. No Zarr regeneration is required. The Python switch
is --image-fusion / --no-image-fusion. HEATMAP_IMAGE_FUSION and the earlier
--heatmap-image-fusion CLI spelling remain aliases; IMAGE_FUSION takes precedence
over the old environment variable. If neither is set, checkpoint loading infers
fusion from the saved weights (new runs default to off). Internal parameter names
retain heatmap_image_pyramid for compatibility with existing fusion checkpoints.

Example HSC fixed-sigma experiment (not launched by this implementation):
CUDA_VISIBLE_DEVICES=0,1 NPROC=2 \
ROOT=/data/czh23/analysis/2026-11/2026-11-07/training_zarr_hsc_half_relative_snr57_manhattan/zarr \
TRAIN_FILE=/home/czh23/CELLECT/patches/train_zarr_psfee_hsc.txt \
VAL_FILE=/home/czh23/CELLECT/patches/val_zarr_psfee_hsc.txt \
OUT=/data/czh23/ckpts/sam_vit_b_heatmap_bce_1108 \
PROPOSAL_MODE=heatmap HEATMAP_LOSS=soft_bce HEATMAP_SIGMA=1.5 \
SAM_COMPILE=0 WANDB_RUN_NAME=sam_vit_b_heatmap_bce_1108 \
bash /home/czh23/CELLECT/scripts/train_mixed_zarr.sh

Use HEATMAP_SIGMA_MODE=psf-ee and HEATMAP_PSF_EE_ROOT=<cache> for the scale
ablation. Use a NEW output directory for each experiment. New heatmap runs can
start from the same official SAM --sam-checkpoint as controls. An ordinal
proposal checkpoint is not a drop-in resume checkpoint for the new head shapes.
The current train/val selector files above are reused verbatim, not resampled.

Validation: target/rasterization/ownership/ignore/empty-image/gradient/legacy
decoder/checkpoint/browser/mask-anchor tests, plus a real 512x512 HSC-G noisy
group0 1,4 block forward/backward. No full training or Zarr generation launched.
