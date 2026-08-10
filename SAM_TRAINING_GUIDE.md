# SAM Astro/CELLECT 训练说明

本文档说明 `astro_train_eval.py --model-variant sam_per_band` 的训练方式。当前 SAM 分支的目标是先训练高召回的 region proposal：输出 `confidence map` 和 `shape map`，再可选接入 SAM prompt mask decoder 输出实例 mask。

## 代码位置

- 训练入口：`astro_train_eval.py`
- epoch/loss 调用：`astro_train_ops.py`
- SAM 模型：`sam_backbone/model.py`
- SAM encoder 构建与 checkpoint 加载：`sam_backbone/build.py`
- CELLECT 风格 proposal decoder：`sam_backbone/decoder.py`
- SAM prompt/mask loss：`sam_backbone/losses.py`
- 预处理与 per-band 逻辑：`sam_backbone/preprocess.py`

## 模型结构

`sam_per_band` 使用原生 SAM image encoder，并替换/扩展输出头：

1. 输入为多波段图像，默认 `512 x 512`。
2. 每个 band 按单通道天文图像处理，输入 SAM encoder 前会复制成 RGB 三通道。
3. per-band 训练时实际 encoder batch 约为 `batch_size * band_count`。
4. SAM encoder 输出 `[B, band, 256, 32, 32]` image embedding。
5. CELLECT 风格 decoder 输出：
   - `confidence`: `[B, band, 5, 512, 512]`
   - `shape`: `[B, band, 3, 512, 512]`
   - `image_embeddings`: 给 SAM mask decoder 使用
6. 可选 SAM prompt mask decoder 使用 `center + bbox` prompt，不使用 negative prompt。

`sam_per_band` 会自动关闭旧 CELLECT 分支中不适合当前目标的训练项：

- segmentation loss
- triplet loss
- EX/EN loss
- EN/EX 后处理
- PU self-training

## 数据要求

训练数据目录应包含 cutout 图像、reference catalog 和可选的预计算 target。典型结构由 `astro_data_preprocessing.py` 生成：

```text
<root>/
  <tract>/<patch>/
    cutouts/
    reference_catalogs/
    band_reference_catalogs/
    targets/
    band_targets/
    tile_metadata/
    band_tile_metadata/
```

关键 target 字段：

- `confidence`: ordinal center heatmap。
- `shape`: 每个像素的 `(major, minor, theta)`。
- `shape_weight`: shape 监督权重。
- `clean_mask`: 干净源 Kron aperture union。
- `center_only_mask`: 亮源或低可信但应检出的源。
- `ignore_mask`: 普通 ignore 区域。
- `strict_center_only_mask` / `strict_ignore_mask`: 当前训练中统一按亮源/center-only 低权重源处理，不作为 hard ignore。
- `confidence_weight`: confidence loss 的统一权重入口。

SAM 训练不需要 PU self-training。即使命令行传入 `--enable-pu-self-training`，`sam_per_band` 也会强制关闭。

## 预处理

训练入口读取数据集时使用预处理后的图像或在线 zscale/sigma clip 结果。SAM 分支当前构建模型时设置：

```text
astro_preprocess_in_model=False
```

也就是说，训练数据应已经由数据管线完成 zscale/sigma clip，或由 `AstroCutoutDataset` 的缓存路径读取。若使用预计算 zscale cache，可通过：

```bash
--image-cache-dir /path/to/image_cache
```

## Loss 设计

### Proposal loss

默认 SAM proposal 训练主要使用：

1. `confidence loss`
   - 使用 ordinal confidence target。
   - 权重来自 `confidence_weight`。
   - clean aperture 为 full weight。
   - center-only / strict bright source 只在已有正 confidence 标签处给低权重。
   - LSST 背景和 ordinary ignore 不作为负例硬约束。

2. `shape loss`
   - 预测 `(major, minor, theta)`。
   - clean source 以 `shape_weight` 监督。
   - center-only / strict bright source 保留低权重 shape 监督。
   - 默认低权重由 `--center-only-shape-factor 0.2` 控制。

3. `center loss`
   - 默认关闭：`--center-loss-weight 0.0`。
   - 需要额外约束中心偏移时再打开。

### SAM mask decoder loss

通过 `--mask-loss-weight > 0` 打开。代码在 `sam_backbone/losses.py`。

prompt 只使用：

- positive center point
- 由 shape 生成的 bbox

prompt curriculum：

- epoch `< --mask-prompt-gt-epochs`：全部使用 GT center/shape。
- epoch `--mask-prompt-gt-epochs` 到 `--mask-prompt-pred-epoch`：线性增加 predicted prompt 比例。
- epoch `>= --mask-prompt-pred-epoch`：全部使用 predicted center/shape。

默认是：

```text
--mask-prompt-gt-epochs 5
--mask-prompt-pred-epoch 30
```

mask loss 细分：

- Dice loss
- BCE loss
- centroid loss：mask 几何中心接近 prompt/GT center。
- outside prior：惩罚 mask 超出 shape aperture 的区域。
- min-area prior：惩罚过小 mask，默认面积阈值 `15 px`。

SAM multimask 输出默认开启。训练时对多个 mask 计算 loss，并使用 best-of-K 监督。若要关闭：

```bash
--disable-mask-multimask
```

## LR schedule

`sam_per_band` 使用 iteration 级 LR schedule。

默认参数：

```text
head lr:    --lr 1e-4
encoder lr: --sam-encoder-lr 2e-5
warmup:     --sam-warmup-ratio 0.01
drop:       --sam-lr-drop-fractions 0.70 0.90
gamma:      --sam-lr-drop-gamma 0.1
epochs:     --epochs 100
```

总步数为：

```text
total_steps = epochs * len(train_loader)
```

训练前 1% iteration 线性 warmup，之后在 70% 和 90% total steps 处阶梯衰减。

## wandb 日志

默认启用 wandb：

```bash
--wandb-project Astro_CELLECT2D_SAM
--wandb-mode online
--wandb-log-interval 50
```

关闭：

```bash
--wandb-mode disabled
```

iteration 级日志每隔固定 optimizer iteration 上传：

- `train/iteration/loss/total`
- `train/iteration/loss/confidence`
- `train/iteration/loss/shape`
- `train/iteration/loss/mask`
- `train/iteration/loss/mask_dice`
- `train/iteration/loss/mask_bce`
- `train/iteration/loss/mask_centroid`
- `train/iteration/loss/mask_outside`
- `train/iteration/loss/mask_area`
- `train/iteration/loss/mask_prompts`
- `train/iteration/lr/head`
- `train/iteration/lr/encoder`
- `train/iteration/prompt/predicted_ratio`
- `train/iteration/prompt/gt_ratio`

epoch 级日志上传：

- `train/epoch/*`
- `val/epoch/*`
- `lr/*`
- prompt curriculum ratio

DDP 下 iteration loss 会先跨 rank all-reduce 合并，再由 rank0 上传。

## Checkpoint

有两类 checkpoint 参数：

1. `--sam-checkpoint`
   - 官方 SAM checkpoint。
   - 用于初始化 SAM image encoder、prompt encoder、mask decoder。
   - 支持 `vit_b`、`vit_l`、`vit_h`。

2. `--checkpoint`
   - 本项目训练保存的 checkpoint。
   - 用于恢复完整模型训练或评估。

常见官方 checkpoint：

```text
/home/czh23/sam_ckpts/sam_vit_b_01ec64.pth
/home/czh23/sam_ckpts/sam_vit_l_0b3195.pth
/home/czh23/sam_ckpts/sam_vit_h_4b8939.pth
```

训练输出目录中会保存：

- `last.pt`
- `best.pt`
- 可选 `epoch_XXXX.pt`
- `run_config.json`

## 使用示例

### 1. 单卡 proposal-only 训练

先只训练 `confidence + shape`，不训练 SAM mask decoder：

```bash
cd /home/czh23/CELLECT

/home/czh23/miniconda3/envs/cellect/bin/python astro_train_eval.py \
  --mode train \
  --model-variant sam_per_band \
  --sam-model-type vit_b \
  --sam-checkpoint /home/czh23/sam_ckpts/sam_vit_b_01ec64.pth \
  --root /path/to/dataset_root \
  --targets-dir /path/to/dataset_root/targets \
  --bands HSC-G HSC-R HSC-I HSC-Z HSC-Y \
  --out-dir /home/czh23/CELLECT/output/ckpts/SAM_vit_b_proposal \
  --epochs 100 \
  --batch-size 1 \
  --num-workers 4 \
  --pin-memory \
  --persistent-workers \
  --prefetch-factor 2 \
  --lr 1e-4 \
  --sam-encoder-lr 2e-5 \
  --sam-warmup-ratio 0.01 \
  --sam-lr-drop-fractions 0.70 0.90 \
  --sam-lr-drop-gamma 0.1 \
  --confidence-pos-weight 32 \
  --shape-loss-weight 1.0 \
  --center-only-shape-factor 0.2 \
  --mask-loss-weight 0.0 \
  --amp bf16 \
  --wandb-run-name SAM_vit_b_proposal \
  --wandb-log-interval 50
```

### 2. 两卡 DDP 训练 ViT-L proposal

```bash
cd /home/czh23/CELLECT

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 astro_train_eval.py \
  --mode train \
  --ddp \
  --model-variant sam_per_band \
  --sam-model-type vit_l \
  --sam-checkpoint /home/czh23/sam_ckpts/sam_vit_l_0b3195.pth \
  --root /path/to/dataset_root \
  --targets-dir /path/to/dataset_root/targets \
  --bands HSC-G HSC-R HSC-I HSC-Z HSC-Y \
  --out-dir /home/czh23/CELLECT/output/ckpts/SAM_vit_l_proposal \
  --epochs 100 \
  --batch-size 2 \
  --num-workers 4 \
  --pin-memory \
  --persistent-workers \
  --prefetch-factor 2 \
  --lr 1e-4 \
  --sam-encoder-lr 2e-5 \
  --sam-warmup-ratio 0.01 \
  --sam-lr-drop-fractions 0.70 0.90 \
  --sam-lr-drop-gamma 0.1 \
  --mask-loss-weight 0.0 \
  --amp bf16 \
  --wandb-run-name SAM_vit_l_proposal_ddp \
  --wandb-log-interval 20
```

这里 `--batch-size` 是每卡 batch size。global batch size 约为 `batch_size * world_size`。

### 3. 打开 SAM mask decoder loss

建议先用 proposal-only 训练若干 epoch，使 `confidence/shape` 稳定后，再打开 mask loss。

```bash
cd /home/czh23/CELLECT

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 astro_train_eval.py \
  --mode train \
  --ddp \
  --model-variant sam_per_band \
  --sam-model-type vit_b \
  --sam-checkpoint /home/czh23/sam_ckpts/sam_vit_b_01ec64.pth \
  --checkpoint /home/czh23/CELLECT/output/ckpts/SAM_vit_b_proposal/best.pt \
  --root /path/to/dataset_root \
  --targets-dir /path/to/dataset_root/targets \
  --bands HSC-G HSC-R HSC-I HSC-Z HSC-Y \
  --out-dir /home/czh23/CELLECT/output/ckpts/SAM_vit_b_mask \
  --epochs 100 \
  --batch-size 1 \
  --num-workers 4 \
  --pin-memory \
  --persistent-workers \
  --lr 1e-4 \
  --sam-encoder-lr 1e-5 \
  --mask-loss-weight 1.0 \
  --mask-dice-weight 0.5 \
  --mask-bce-weight 0.2 \
  --mask-centroid-weight 0.2 \
  --mask-outside-weight 0.5 \
  --mask-min-area-weight 0.1 \
  --mask-min-area-px 15 \
  --mask-unmatched-prompt-weight 0.2 \
  --mask-prompt-gt-epochs 5 \
  --mask-prompt-pred-epoch 30 \
  --mask-prompt-chunk-size 128 \
  --amp bf16 \
  --wandb-run-name SAM_vit_b_mask \
  --wandb-log-interval 10
```

### 4. 离线调试小数据

```bash
cd /home/czh23/CELLECT

/home/czh23/miniconda3/envs/cellect/bin/python astro_train_eval.py \
  --mode train \
  --model-variant sam_per_band \
  --sam-model-type vit_b \
  --sam-checkpoint /home/czh23/sam_ckpts/sam_vit_b_01ec64.pth \
  --root /path/to/dataset_root \
  --bands HSC-G HSC-R HSC-I HSC-Z HSC-Y \
  --max-records 8 \
  --epochs 2 \
  --batch-size 1 \
  --num-workers 0 \
  --out-dir /home/czh23/CELLECT/output/debug/SAM_small \
  --mask-loss-weight 0.0 \
  --wandb-mode disabled
```

### 5. 指定 train/val patch

```bash
cd /home/czh23/CELLECT

/home/czh23/miniconda3/envs/cellect/bin/python astro_train_eval.py \
  --mode train \
  --model-variant sam_per_band \
  --sam-model-type vit_b \
  --sam-checkpoint /home/czh23/sam_ckpts/sam_vit_b_01ec64.pth \
  --root /path/to/dataset_root \
  --bands HSC-G HSC-R HSC-I HSC-Z HSC-Y \
  --train-patches 9813/0,0 9813/0,1 9813/1,0 \
  --val-patches 9813/6,1 \
  --out-dir /home/czh23/CELLECT/output/ckpts/SAM_patch_split \
  --epochs 100 \
  --batch-size 1 \
  --mask-loss-weight 0.0
```

## 评估示例

```bash
cd /home/czh23/CELLECT

/home/czh23/miniconda3/envs/cellect/bin/python astro_train_eval.py \
  --mode eval \
  --model-variant sam_per_band \
  --checkpoint /home/czh23/CELLECT/output/ckpts/SAM_vit_b_proposal/best.pt \
  --root /path/to/dataset_root \
  --bands HSC-G HSC-R HSC-I HSC-Z HSC-Y \
  --out-dir /home/czh23/CELLECT/output/eval/SAM_vit_b_proposal \
  --confidence-threshold 2.0 \
  --nms-radius 1 \
  --confidence-score cellect \
  --center-refinement integer \
  --wandb-mode disabled
```

输出包括：

- detection metrics
- `eval_sources.csv`
- `run_config.json`

## 建议训练流程

1. 先跑小数据 smoke test，确认数据路径、checkpoint、target 维度无误。
2. 用 `vit_b` 做 proposal-only 训练，观察 recall、confidence loss、shape loss。
3. 若目标是尽快提高召回，先不要打开 mask loss。
4. proposal 稳定后打开 `--mask-loss-weight`，并降低 encoder LR。
5. ViT-L/ViT-H 显存压力大时优先：
   - 使用 `--amp bf16`
   - 减小 per-GPU batch size
   - 增加 GPU 数用 DDP
   - 降低 `--mask-prompt-chunk-size`
   - 先关闭 mask loss

## 常见问题

### 训练中没有 PU 是否正常？

正常。`sam_per_band` 会强制关闭 PU self-training。当前 SAM 目标是 proposal recall 和 prompt mask decoder，不需要旧 CELLECT 的 PU pseudo-label 扩增。

### `mask-loss-weight` 应该默认打开吗？

不建议一开始打开。训练初期 `confidence/shape` 质量较差，predicted prompt 会不稳定。推荐先 proposal-only 训练，或依赖 curriculum 前几轮的 GT prompt。

### prompt curriculum 如何看？

wandb 中查看：

```text
train/iteration/prompt/predicted_ratio
train/iteration/prompt/gt_ratio
```

默认第 0-4 epoch 全 GT，第 5-29 epoch 线性切换，第 30 epoch 后全 predicted。

### 小 mask 如何处理？

SAM mask loss 中，小于 `--mask-min-area-px` 的 mask 不参与 Dice/BCE/centroid/outside 监督，但仍保留 min-area prior 惩罚。

### strict ignore 现在怎么处理？

`strict_ignore_mask` 不再作为 hard ignore。它会被合入 center-only/bright source 低权重处理，目标是“应检尽检”，避免亮源或 artifact 区域被训练成不可检测。

### ViT-L 很慢怎么办？

先确认瓶颈：

- 若 GPU 利用率波动大，检查 DataLoader：`--num-workers`、`--pin-memory`、`--persistent-workers`。
- 若 forward/backward 主导，优先使用 `bf16`、DDP、关闭 mask loss。
- mask decoder 打开时，`--mask-prompt-chunk-size` 会直接影响显存和速度。

## 推荐默认配置

proposal-only 起步：

```text
--epochs 100
--lr 1e-4
--sam-encoder-lr 2e-5
--sam-warmup-ratio 0.01
--sam-lr-drop-fractions 0.70 0.90
--sam-lr-drop-gamma 0.1
--confidence-pos-weight 32
--shape-loss-weight 1.0
--center-loss-weight 0.0
--center-only-shape-factor 0.2
--mask-loss-weight 0.0
--amp bf16
```

mask 微调：

```text
--mask-loss-weight 1.0
--sam-encoder-lr 1e-5
--mask-prompt-gt-epochs 5
--mask-prompt-pred-epoch 30
--mask-prompt-chunk-size 128
```

## 小于 512 的动态输入尺寸

默认行为保持不变：`sam_per_band` 会把输入补到固定的 `512x512` SAM
画布。若数据本身小于等于 512，并希望只补到最近的 16 像素倍数，可启用：

```text
--sam-dynamic-image-size
```

动态模式会按实际 token 网格插值 ViT absolute position embedding，动态生成
PromptEncoder dense PE，并将 SAM low-resolution mask logits 先插值到 patch-aligned
尺寸，再裁回原始图像尺寸计算 mask loss。一个 batch 内的样本仍需具有相同的
空间尺寸；混合尺寸数据应使用 batch size 1 或按尺寸分桶。
