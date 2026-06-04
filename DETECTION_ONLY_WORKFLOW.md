# Detection-only 训练流程

本文档记录当前推荐的 detection-only 流程：只训练中心检测 confidence map，关闭 segmentation、shape 和 EX linking，保留可选 EN 去重。

默认目录：

```bash
export TRACT=9813
export RAW_ROOT=/data1/czh23/Subaru
export PREP_ROOT=/data1/czh23/preprocessed
export FAST_ROOT=/nvme0/zc/scarlet/preprocessed
export ZSCALE_ROOT=/nvme0/zc/scarlet/cellect_zscale_cache
export OUT_DIR=output/detection_only_b5
export BANDS="HSC-G HSC-R HSC-I HSC-Z HSC-Y"
export DET_THRESHOLD=2.0
```

## 1. Kron/flux aperture refit

如果要使用 `batch-heavyfp-kron-refit` 生成的 proxy Kron/flux aperture 半径，先对每个 band/patch 生成 refit CSV。下面是单个 patch、单个 band 的命令：

```bash
conda run -n cellect python batch-heavyfp-kron-refit/batch_heavyfp_kron_refit.py \
  --meas-catalog ${RAW_ROOT}/${TRACT}/HSC-I/0,0/meas-HSC-I-${TRACT}-0,0.fits \
  --reference-image ${RAW_ROOT}/${TRACT}/HSC-I/0,0/calexp-HSC-I-${TRACT}-0,0.fits \
  --mag-min 15 \
  --mag-max 35 \
  --output-dir batch-heavyfp-kron-refit/refit/${TRACT}/HSC-I/0,0 \
  --allow-missing-heavy-footprints \
  --leaf-only \
  --include-non-primary
```

预处理阶段需要的 CSV 路径建议整理成如下模板：

```text
batch-heavyfp-kron-refit/refit/{tract}/{band}/{patch}/batch_heavyfp_kron_refit/kron_refit_rows.csv
```

如果实际文件名不同，只要在预处理命令里相应修改 `--pu-kron-refit-csv` 即可。

## 2. 预处理

目标：

- 使用 PU label mode。
- 使用每个 band 的极限星等 `[m-5, m)` 作为监督源范围。
- 使用 HSC 官方 saturation magnitude 筛亮源，亮源 `mag < saturation_mag` 的 `proxy_flux_aperture` 区域写入 `strict_ignore_mask`。
- `strict_ignore_mask` 同时并入 `ignore_mask`，训练和 eval metrics 都不统计。
- zscale、targets、catalog metadata 写到 SSD。

HSC 默认极限星等：

```text
G=27.4, R=27.1, I=26.9, Z=26.3, Y=25.3
```

HSC saturation 星等：

```text
G=18.0, R=18.2, I=18.6, Z=17.7, Y=17.4
```

完整预处理命令：

```bash
conda run -n cellect python astro_data_preprocessing.py \
  --coadd-root ${RAW_ROOT} \
  --catalog-root ${RAW_ROOT} \
  --band-catalog-root ${RAW_ROOT} \
  --tract ${TRACT} \
  --patches all \
  --bands ${BANDS} \
  --output-root ${PREP_ROOT} \
  --fast-root ${FAST_ROOT} \
  --zscale-root ${ZSCALE_ROOT} \
  --label-mode pu \
  --target-shape-source kron \
  --ellipse-sigma 1.0 \
  --source-filter nchild0 \
  --pu-use-band-limit-b-filter \
  --pu-band-limit-mags HSC-G=27.4 HSC-R=27.1 HSC-I=26.9 HSC-Z=26.3 HSC-Y=25.3 \
  --pu-band-limit-b-min-offset -5.0 \
  --pu-band-limit-b-max-offset 0.0 \
  --pu-keep-all-ab-clean \
  --pu-enable-strict-bright-ignore \
  --pu-strict-ignore-saturation-mags HSC-G=18.0 HSC-R=18.2 HSC-I=18.6 HSC-Z=17.7 HSC-Y=17.4 \
  --pu-strict-ignore-radius-column proxy_nan0_flux_aperture_radius \
  --pu-kron-refit-csv 'batch-heavyfp-kron-refit/refit/{tract}/{band}/{patch}/batch_heavyfp_kron_refit/kron_refit_rows.csv' \
  --pu-kron-refit-radius-column proxy_nan0_determine_radius_returned_radius \
  --bad-band-catalog-policy error \
  --num-workers 8 \
  --overwrite-zscale
```

如果 refit CSV 已完整生成，可加：

```bash
--pu-require-kron-refit-match
```

如果还没有 refit CSV，不要传 `--pu-kron-refit-csv`。这时 strict bright ignore 会回退到 catalog 自带 Kron 半径，而不是 proxy flux aperture 半径。

## 3. Detection-only DDP 训练

`--batch-size` 是 per-device batch size。两张 GPU、`--batch-size 8` 表示全局 batch size 为 16。

训练时使用 patch 级 val。这里固定用 `4,5` 和 `6,1` 做 validation：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 astro_train_eval.py \
  --mode train \
  --root ${FAST_ROOT}/${TRACT} \
  --image-cache-dir ${ZSCALE_ROOT}/${TRACT} \
  --bands ${BANDS} \
  --model-variant fused_encoder \
  --detection-only \
  --enable-en-loss \
  --use-en-postprocess \
  --epochs 100 \
  --batch-size 8 \
  --num-workers 16 \
  --pin-memory \
  --persistent-workers \
  --prefetch-factor 2 \
  --confidence-threshold ${DET_THRESHOLD} \
  --confidence-score cellect \
  --center-refinement softargmax \
  --center-refinement-radius 1 \
  --source-filter nchild0 \
  --val-patches 4,5 6,1 \
  --detect-every 5 \
  --out-dir ${OUT_DIR} \
  2>&1 | tee ${OUT_DIR}.log
```

`--detection-only` 会自动设置：

- `seg_loss_weight = 0`
- `shape_loss_weight = 0`
- `disable_ex_loss = True`
- `use_ex_link_postprocess = False`
- `train_detect_ex_link = False`

因此不要再传 `--use-ex-link-postprocess`。

## 4. 评测指定 patch

评测 `4,5` 和 `6,1`：

```bash
CUDA_VISIBLE_DEVICES=0 python astro_train_eval.py \
  --mode eval \
  --root ${FAST_ROOT}/${TRACT} \
  --image-cache-dir ${ZSCALE_ROOT}/${TRACT} \
  --bands ${BANDS} \
  --checkpoint ${OUT_DIR}/best.pt \
  --model-variant fused_encoder \
  --detection-only \
  --enable-en-loss \
  --use-en-postprocess \
  --confidence-threshold ${DET_THRESHOLD} \
  --confidence-score cellect \
  --center-refinement softargmax \
  --center-refinement-radius 1 \
  --source-filter nchild0 \
  --eval-patches 4,5 6,1 \
  --eval-sources-csv ${OUT_DIR}/eval_sources_4,5_6,1.csv \
  --out-dir ${OUT_DIR}/eval_4,5_6,1 \
  2>&1 | tee ${OUT_DIR}/eval_4,5_6,1.log
```

eval metrics 默认启用 `--ignore-mask-during-detection`，因此落在 `ignore_mask` 或 `strict_ignore_mask` 内的预测中心不会计入 TP/FP。

`eval_sources.csv` 会保留这些预测，并输出：

```text
x_local, y_local, x_parent, y_parent, ra_deg, dec_deg,
ignored_by_mask, ordinary_ignore, strict_ignore, eval_excluded_by_mask
```

其中：

- `ordinary_ignore=1`：预测落在普通 ignore 区域，不计入 eval 指标。
- `strict_ignore=1`：预测落在亮源 strict ignore 区域，不计入 eval 指标。
- `eval_excluded_by_mask=1`：该预测已被 mask 从 metrics 中排除。

## 5. 常用检查

确认 strict ignore 已写入 targets：

```bash
python - <<'PY'
import numpy as np
from pathlib import Path
p = next((Path("/nvme0/zc/scarlet/preprocessed/9813/4,5/targets")).glob("*.npz"))
with np.load(p) as d:
    strict = d["strict_ignore_mask"] if "strict_ignore_mask" in d else np.zeros_like(d["ignore_mask"])
    print("file:", p)
    print("keys:", sorted(k for k in d.keys() if "ignore" in k or "mask" in k))
    print("ignore pixels:", int(d["ignore_mask"].sum()))
    print("strict ignore pixels:", int(strict.sum()))
    print("background pixels:", int(d["background_mask"].sum()))
PY
```

确认 eval CSV 中被 ignore 的预测数量：

```bash
python - <<'PY'
import pandas as pd
p = "output/detection_only_b5/eval_sources_4,5_6,1.csv"
df = pd.read_csv(p)
print("all predictions:", len(df))
print("ordinary ignore:", int(df["ordinary_ignore"].sum()))
print("strict ignore:", int(df["strict_ignore"].sum()))
print("used by metrics:", int((df["eval_excluded_by_mask"] == 0).sum()))
PY
```
