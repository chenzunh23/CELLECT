# CELLECT Data Workflow 0729

本文档梳理 2026-07-29 版本的数据预处理流程。重点是当前 SAM detector 训练所需的数据路径；历史 EX/linking 逻辑只保留在评测或旧实验中，不作为这里的训练目标。

## 1. 输入数据

### Official Coadd

官方 HSC/Subaru coadd 数据通常位于：

```text
<coadd_root>/<tract>/<band>/<patch>/calexp-<band>-<tract>-<patch>.fits
<coadd_root>/<tract>/<band>/<patch>/meas-<band>-<tract>-<patch>.fits
<coadd_root>/<tract>/<band>/<patch>/det-<band>-<tract>-<patch>.fits
```

当前服务器示例：

```text
/data/shared/Subaru/9813/HSC-I/4,5/calexp-HSC-I-9813-4,5.fits
/data/shared/Subaru/9813/HSC-I/4,5/meas-HSC-I-9813-4,5.fits
/data/shared/Subaru/9813/HSC-I/4,5/det-HSC-I-9813-4,5.fits
```

`calexp` 提供 IMAGE/MASK/VARIANCE，`meas` 提供源星表，`det` 提供 LSST detection footprint，可用于更可靠的背景定义。

### Kron Refit CSV

Kron aperture 形状优先来自 batch-heavyfp-kron-refit 的输出：

```text
<refit_root>/<tract>/<band>/<patch>/batch_heavyfp_kron_refit/batch_heavyfp_kron_refit.csv
```

默认半径列：

```text
proxy_nan0_flux_aperture_radius
```

如果 refit 失败或没有匹配，当前正式筛选一般不允许该源进入 clean；缺失策略由 `--require-kron-refit-match` 控制。

### Noisy / Denoised Variants

noisy/denoised FITS 与 official coadd 共用星表和 refit 结果：

```text
<denoised_fits_root>/patch_<x>_<y>/group_XX/<band>/denoised.fits
<denoised_fits_root>/patch_<x>_<y>/group_XX/<band>/noisy.fits
```

这些图像用于训练时增加图像域变化，但源定义仍来自 official coadd/refit 后的 catalog。

## 2. Coadd 星表筛选

代码入口：

```text
data_filtering/pu_source_filter.py
data_filtering/pu_config.py
astro_data_preprocessing.py::_classify_pu_catalog
direct_zarr_preprocessing/direct_preprocess_zarr.py::_classify_all_bands
```

### 2.1 初始源

默认源集合为 leaf source：

```text
source_filter = nchild0
```

基础要求：

```text
finite centroid
finite positive flux/shape/refit radius
deblend_nChild == 0
可匹配 refit CSV
```

### 2.2 A Filter

A filter 先处理明显不可用的大源或暗弱大椭圆：

```text
area > 10000                  -> drop，不进入 clean/center/ignore mask
area > 900 and mag > 28       -> A failed，可进入 ordinary ignore
```

这个步骤的目标是避免明显错误的大椭圆覆盖整片训练区域。

### 2.3 B Filter

B filter 从 A 通过的源中删去不适合 clean 监督的源：

```text
mag outside [18, 30]
或 band-limit 模式下 mag outside [m_limit - 5, m_limit)
abs(AP2_mag - Kron_mag) >= 1
axis_ratio > 5
base_SdssShape_flag
base_SdssCentroid_flag
close center pair within 0.5 arcsec: remove dimmer source
large source contains >=80% smaller source: remove larger source
```

B 删去的源一般成为 ordinary ignore。它们不作为背景，也不作为 clean shape 监督。

### 2.4 AP2 / Kron / Fill 规则

当前还包含几个重要的补充规则：

```text
area > 500 and aperture_pixel_count / aperture_area < 0.3
  -> center_only，不训练 shape

bright mag<22 refined AP2:
  center outside bright region and invalid/AP2-Kron absdiff >= 1 -> ordinary ignore
  center inside small bright region (area < 1000) and absdiff >= 2 -> ordinary ignore
  center inside large bright region (area >= 1000) -> skip AP2; bright flow handles it

AP2-Kron outlier 可用 refit aperture 重新测光救回：
  new absdiff < 1.0       -> clean
  1.0 <= new absdiff <=1.5 -> center_only
  invalid / >1.5          -> ignore
```

最终 coadd 源类别：

```text
clean                  center + confidence + shape
center_only            center + confidence，shape 弱或不训练
strict_center_only     明亮/饱和中心，只训练中心，不训 shape
ordinary ignore        不训练 foreground，也不能当背景
dropped                完全不进入 mask
```

## 3. 背景定义

背景不再默认等于“所有源椭圆之外”。正式流程优先使用 LSST pipeline detection footprint：

```text
background = outside(det footprint)
```

对应代码：

```text
astro_data_preprocessing.py::_read_det_background_mask
direct_zarr_preprocessing/direct_preprocess_zarr.py::_read_backgrounds
```

对于 noisy/denoised variant：

```text
--image-variant-background-source variant-lsst
--variant-lsst-background-policy run-if-missing
```

如果 variant 没有背景，脚本会调用 `lsst_detect_background.py` 在该 variant FITS 上跑默认 LSST detection，生成该 variant 自己的背景。需要 LSST stack Python：

```text
--lsst-detect-python /home/czh23/miniconda3/envs/lsst-scipipe-10.1.0/bin/python
```

这样 noisy/denoised 不会退回到过大的“椭圆补集背景”。

## 4. 亮源与亮区

亮源处理分两层：catalog 源标签和图像亮区 mask。

### 4.0 Current Formal Ordering

当前正式流程按下面顺序搭建，`astro_data_preprocessing.py` 和
`direct_zarr_preprocessing/direct_preprocess_zarr.py` 共用同一套核心逻辑：

```text
meas catalog
  -> batch-heavyfp-kron-refit CSV
  -> A filter
  -> B filter
       non-bright: mag / AP2-Kron / axis ratio / close pair / containment / B flags
       bright mag<22: axis ratio / close pair / refined AP2 only
         outside bright region: |AP2-Kron| < 1
         small bright region: |AP2-Kron| < 2
         large bright region: AP2 skipped
  -> non-bright SAT/EDGE/BAD and SNR filtering
       coadd: broad/narrow AP2-SNR standard
       noisy/denoised: variance/weight or local noncoadd SNR standard
  -> bright-source external labels v2
       zscore-no-upper: no image-threshold bright components; use source clusters + Gaia
       other scaling modes: use their own bright mask from image scaling
  -> dense target priority
       clean > center_only/strict_center_only > bright > LSST background > ignore
```

`zscore-no-upper` 的目的不是生成 `z>=3` 的亮区，而是让模型看到无上限
zscore 图像本身。因此该模式下 `build_bright_mask()` 返回空 mask；如果需要亮区，
只使用 external bright label 流程中 Gaia-matched source-cluster aperture union
写出的 `*_bright_mask.fits`。

### 4.1 Strict Center-Only

已经是 clean 的超亮源可转为 strict center-only：

```text
HSC-G 18.0
HSC-R 18.2
HSC-I 18.6
HSC-Z 17.7
HSC-Y 17.4
NB0387 14.8
NB0816 16.8
NB0921/NB0924 16.9
NB1010 14.8
```

它们提供中心监督，但不提供可靠 shape。

### 4.2 Bright Region Mask

共享实现：

```text
data_filtering/sam_input_scaling.py::build_bright_mask
```

可选模式：

```text
zscore:
  当前 clipped zscore reference；
  z >= threshold 作为 bright。

log-lupton:
  log image 和 Lupton image 分别 self-standardize；
  二者 z > 3 的交集作为 bright。

anscombe:
  Anscombe image self-standardize；
  z > 3 作为 bright。

raw / none:
  不生成 bright 区。

zscore-no-upper / zscore-unbounded:
  不生成图像 bright 区；
  使用 external bright labels v2 的 source-cluster/Gaia 简化流程。
```

CLI：

```text
--pu-enable-bright-background-mask
--pu-bright-mask-mode zscore|log-lupton|anscombe|zscore-no-upper|raw|none
--pu-bright-z-threshold 3.0
--external-bright-label-root output/data_filter_0729/external_bright_labels_v2_zscore_no_upper_no_bad
```

在正式训练用的 dense target 中，priority 是：

```text
clean > center_only / strict_center_only > bright > explicit ignore > LSST background
```

这表示 bright 区可以压制亮平台上的假源，但不能覆盖可信 clean/center-only 标签。显式 ordinary ignore 会优先于 LSST background；如果存在 LSST background，所有既不是 clean/center/bright/background 的像素也会并入 ignore。

### 4.3 External Bright Diagnostic

诊断脚本：

```text
data_filtering/build_external_bright_labels_v2.py
```

当前策略：

```text
existing clean/center_only 保持不变
面积 >=10000 的亮源先 ignore
亮区由 log-lupton 或其他 scaling 生成 component
同一 component 内按中心距离<=50px 且 Kron IoU>=1/3 聚类
Gaia DR3 用于识别亮星中心
SAT/BAD/EDGE 且有 Gaia 亮星的 cluster -> ignore
无 Gaia 亮星但 HSC galaxy 且 shape 合理 -> brightest center_only_external
其他 cluster member -> restricted_bright_region
Gaia bright star -> strict_center_only_external
Gaia-unmatched star/unknown -> ignore
```

注意：`build_external_bright_labels_v2.py` 的 priority partition 是诊断可视化产品，仍按 `clean > center_only > bright > background > ignore` 画图；正式训练 target 以 `astro_data_preprocessing.py::make_pu_dense_targets` 的优先级为准。

这个流程目前主要用于诊断和未来正式接入。原因是 Gaia 对大星系不完整，不能单独作为亮源真值。

## 5. Non-Coadd SNR 降级

noisy/denoised 图像共用 coadd catalog，但每张图像深度不同，因此 clean 源需要按图像可见性降级。

代码：

```text
data_filtering/noncoadd_snr.py
```

直接图像 SNR 模式：

```text
AP aperture radius = 6 px
background annulus = 10-15 px
annulus 排除其他源区域
可排除 BRIGHT_OBJECT/SAT/BAD/NO_DATA/EDGE/UNMASKEDNAN
```

默认降级：

```text
SNR < 2 or invalid       -> ignore
2 <= SNR < 3            -> center_only
SNR >= 3                -> clean
insufficient annulus    -> center_only
```

更严格实验可使用 3/5 阈值。

## 6. Variance / Weight-Ratio SNR

对于有 VARIANCE plane 的 noisy 图像，可以用 variance 比值估计有效曝光时间：

```text
scale = median(noisy_AP2_sum / coadd_AP2_sum)
T = var_coadd * scale^2 / var_noisy
SNR_noisy = SNR_coadd_AP2 * sqrt(T)
```

对于没有 variance plane 的产品，可以用 warp weight 比值：

```text
T_eff = sum(noisy selected warp weights) / sum(coadd warp weights)
SNR_noncoadd = SNR_coadd_AP2 * sqrt(T_eff)
```

注意：variance/weight SNR 只能把已经 clean 的 coadd 源降级，不能把原始 ignore/rejected 源救回 clean。

## 7. 窄带 Bad-Score 过滤

窄带不是每个 patch/band 都可用，而且存在条纹、缺失、大面积插值等问题。当前按 calexp MASK plane 评分。

共享代码：

```text
data_filtering/calexp_quality.py
data_filtering/analyze_calexp_mask_quality.py
data_filtering/overlay_calexp_tile_bad_score.py
```

默认 bad score 权重：

```text
NO_DATA      1.0
UNMASKEDNAN  1.0
EDGE         0.7
BAD          0.5
INTRP        0.3
```

过滤标准：

```text
whole patch bad_score >= 13% -> drop patch for that band
512x512 tile bad_score >= 13% -> drop tile for that band
tile_size=512, stride=368
不使用边缘 partial tile
```

这个过滤只用于 image-level SAM 训练数据。多波段评测仍使用常规 patch/tile 发现逻辑。

## 8. Legacy Preprocessed 输出

入口：

```text
data_preprocessing.sh
astro_data_preprocessing.py
```

典型输出：

```text
<output_root>/<tract>/<patch>/cutouts/
<output_root>/<tract>/<patch>/band_targets/<band>/*.npz
<output_root>/<tract>/<patch>/band_reference_catalogs/<band>/
<output_root>/<tract>/<patch>/band_reference_center_only/<band>/
<output_root>/<tract>/<patch>/band_reference_strict_center_only/<band>/
<output_root>/<tract>/<patch>/band_reference_ignore/<band>/
<zscale_root>/<tract>/<patch>/cutouts/*.pt
```

这些产物适用于旧的 `--data-format legacy` 训练/评测和很多诊断 notebook。

## 9. Direct Zarr 输出

入口：

```text
direct_zarr_preprocessing/run_direct_zarr.sh
direct_zarr_preprocessing/direct_preprocess_zarr.py
```

### 9.1 多波段 Patch Store

默认 direct zarr 仍可写多波段 patch store：

```text
<output_root>/<tract>/coadd/<patch>.zarr
<output_root>/<tract>/denoised/<patch>.zarr
<output_root>/<tract>/noisy/<patch>.zarr
```

这类 store 每条 sample 是一个多波段 tile/group，适合旧多波段评测。

### 9.2 SAM Image-Level Store

SAM detector 训练推荐使用 image-level zarr：

```text
<output_root>/<tract>/image_level/<dataset>/<band>/<patch>.zarr
<output_root>/<tract>/image_level/<dataset>/<band>/<patch>__group_XX.zarr
```

每条 sample 只含一个 band：

```text
image shape = [1, 512, 512]
band target shape = [1, ...]
```

这样训练时 batch size 表示“图片张数”，不是 group 数，也可以自然混合：

```text
coadd HSC-G/HSC-R/.../NB...
denoised group_XX HSC-G/HSC-R/...
noisy group_XX HSC-G/HSC-R/...
```

生成命令示例：

```bash
WRITE_IMAGE_LEVEL_ZARR=1 IMAGE_LEVEL_ONLY=1 QUALITY_FILTER=1 \
MISSING_BAND_POLICY=skip QUALITY_BAD_SCORE_THRESHOLD=0.13 \
LSST_DETECT_PYTHON=/home/czh23/miniconda3/envs/lsst-scipipe-10.1.0/bin/python \
DATA_ROOT=/data/czh23 \
COADD_ROOT=/data/shared/Subaru \
CATALOG_ROOT=/data/shared/Subaru \
BAND_CATALOG_ROOT=/data/shared/Subaru \
REFIT_ROOT=/data/czh23/refit \
PATCHES=all \
BANDS="HSC-G HSC-R HSC-I HSC-Z HSC-Y NB0387 NB0816 NB0921 NB1010" \
bash direct_zarr_preprocessing/run_direct_zarr.sh
```

## 10. SAM 训练

训练入口：

```text
astro_train_eval.py --data-format zarr --zarr-random-image-batches
```

该模式仅用于 SAM detector：

```text
--model-variant sam_per_band
EX/triplet 自动关闭
batch-size = 单图像数量
```

示例：

```bash
python astro_train_eval.py \
  --mode train \
  --data-format zarr \
  --zarr-random-image-batches \
  --root /data/czh23/direct_zarr/9813 \
  --bands HSC-G HSC-R HSC-I HSC-Z HSC-Y NB0387 NB0816 NB0921 NB1010 \
  --model-variant sam_per_band \
  --sam-model-type vit_b \
  --batch-size 16 \
  --num-workers 8 \
  --pin-memory \
  --out-dir output/sam_detector_0729
```

SAM encoder 内部会把每个输入 band 展平成独立图像送入 ViT。不开 style-router 时，训练可以输入单波段，评测可以输入多波段。

## 11. 评测

评测默认保持旧多波段逻辑，以便保留 per-band 指标和必要的 linking 兼容：

```bash
python astro_train_eval.py \
  --mode eval \
  --data-format zarr \
  --root /data/czh23/direct_zarr/9813 \
  --bands HSC-G HSC-R HSC-I HSC-Z HSC-Y \
  --model-variant sam_per_band \
  --checkpoint output/sam_detector_0729/best.pt \
  --eval-patches 4,5 6,1 \
  --out-dir output/sam_detector_0729_eval
```

不要在 eval 中使用 `--zarr-random-image-batches`。该开关是 train-only；eval 会使用普通 multiband zarr discovery。

## 12. 推荐 Debug 顺序

1. 跑 refit，确认每个 band/patch 有 `batch_heavyfp_kron_refit.csv`。
2. 用 `data_filtering/analyze_calexp_mask_quality.py` 统计 patch bad score。
3. 用 `data_filtering/overlay_calexp_tile_bad_score.py` 查看高分 tile 是否符合视觉坏区。
4. 用 `pu_source_filter.py` 或 preprocessed REG 查看 clean/center/ignore 是否合理。
5. 对 noisy/denoised 检查 variant LSST background 是否生成，而不是 fallback 到椭圆补集。
6. 用 1 patch、1 band、`--max-tiles 1` 跑 direct zarr smoke test。
7. 用 `astro_train_eval.py --zarr-random-image-batches --epochs 0` 验证训练入口。

## 13. 当前仍需谨慎的地方

- `build_external_bright_labels_v2.py` 仍是诊断流程，不建议直接默认接入所有生产数据。
- Gaia 对亮星有效，对大星系不完整；不能把 Gaia 没匹配当作“不是源”。
- `raw/none` bright mask 会关闭亮区压制，适合对照实验，不适合作为默认防假源策略。
- 窄带 tile bad-score 过滤只处理图像质量，不处理星表本身源少或 refit 失败的问题。
- variance/weight-ratio SNR 只能对 coadd clean 源做可见性降级，不能修复原始坏 label。
