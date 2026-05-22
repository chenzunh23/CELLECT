# AstroCELLECT 数据下载、预处理和训练/验证用法

本文档说明从 HSC 数据下载、预处理到 `astro_train_eval.py` 训练/评测的推荐流程。当前代码支持多 patch 数据，标准目录层级为：

```text
<root>/<tract>/<patch>/
```

例如 tract `9813`、patch `8,8` 对应：

```text
/data1/czh23/preprocessed/9813/8,8/
/nvme0/zc/scarlet/preprocessed/9813/8,8/
/nvme0/zc/scarlet/cellect_zscale_cache/9813/8,8/
```

## 1. 下载 HSC coadd/catalog FITS

脚本：`download_hsc_coadds.py`

下载脚本会把远端 FITS 保存为：

```text
<data-root>/<tract>/HSC-<band>/<patch>/
  calexp-HSC-<band>-<tract>-<patch>.fits
  meas-HSC-<band>-<tract>-<patch>.fits
  det-...
```

常用命令：

```bash
python download_hsc_coadds.py \
  --data-root /data1/czh23/Subaru \
  --tract 9813 \
  --bands G R I Z \
  --patches all \
  --file-types calexp meas \
  --workers 4 \
  --use-netrc
```

如果服务器需要代理，推荐用 `wget` 后端并显式传入代理，不依赖 `export http_proxy=...`：

```bash
python download_hsc_coadds.py \
  --data-root /data1/czh23/Subaru \
  --tract 9813 \
  --bands G R I Z \
  --patches all \
  --file-types calexp meas \
  --scan-backend wget \
  --download-backend wget \
  --proxy http://HOST:PORT \
  --workers 4 \
  --use-netrc
```

如果已有大量文件，只想补坏文件或不完整文件，使用：

```bash
python download_hsc_coadds.py \
  --data-root /data1/czh23/Subaru \
  --tract 9813 \
  --bands G R I Z \
  --patches all \
  --file-types calexp meas \
  --overwrite-smaller \
  --workers 4 \
  --use-netrc
```

`--overwrite-smaller` 的含义是：当本地文件大小小于远端 `Content-Length` 时重新下载，否则跳过。这个选项适合修复之前下载中断导致的半文件。

## 2. 预处理多 patch 数据

脚本：`astro_data_preprocessing.py`

预处理会生成训练用 tile、catalog、target，以及可选 zscale cache。输出结构如下：

```text
<output-root>/<tract>/<patch>/
  cutouts/<tile>/<band>/*.fits
  reference_catalogs/<tile>_meas.fits
  reference_catalogs_csv/<tile>_meas.csv
  band_reference_catalogs/<band>/<tile>_meas.fits
  targets/<tile>.npz
  sources/sources_filtered.fits
  sources/sources_rejected.fits
  manifest.json
  tiles.csv
  cutout_paths.json
```

推荐策略是：机械盘保存完整 FITS cutouts，SSD 保存训练直接读取的 metadata 和 zscale tensor。

```bash
python astro_data_preprocessing.py \
  --coadd-root /data1/czh23/Subaru \
  --catalog-root /data1/czh23/Subaru \
  --patches all \
  --bands HSC-G HSC-R HSC-I \
  --output-root /data1/czh23/preprocessed \
  --fast-root /nvme0/zc/scarlet/preprocessed \
  --zscale-root /nvme0/zc/scarlet/cellect_zscale_cache \
  --num-workers 8 \
  --bad-band-catalog-policy fallback-primary
```

相关说明：

- `--output-root`：完整预处理输出，通常放机械盘。
- `--fast-root`：只镜像训练 metadata，不复制 cutout FITS，适合放 SSD。
- `--zscale-root`：预先生成 zscale 后的 `.pt` tensor，适合放 SSD。
- `--num-workers`：patch 级并行。不同 worker 处理不同 patch。
- `--patches all`：处理 9x9 patch 网格。
- `--patch-file patches.txt`：从文件读取 patch 列表，每行一个 patch。
- `--skip-cutouts`：不生成 FITS cutouts，只写 catalog/target；如果已经有 zscale 或只需要 metadata，可用。
- `--overwrite-zscale`：强制重新生成已有 zscale tensor。
- `--dry-run`：只打印计划，不写文件。

如果已经有三波段预处理结果，后来补齐了 Z/Y cutout 和星表，不想重新裁剪和重算 target，可以复用已有 `<output-root>/<tract>/<patch>`，只刷新 SSD metadata 和生成新的五波段 zscale：

```bash
python astro_data_preprocessing.py \
  --output-root /data1/czh23/preprocessed \
  --patches all \
  --bands HSC-G HSC-R HSC-I HSC-Z HSC-Y \
  --reuse-existing-preprocessed \
  --fast-root /nvme0/zc/scarlet/preprocessed \
  --zscale-root /nvme0/zc/scarlet/cellect_zscale_cache \
  --num-workers 8
```

这个模式不会重新裁剪 FITS cutout，也不会重建 dense target；它会读取已有 cutout 目录，刷新 `cutout_paths.json`、`manifest.json`，按当前 `--bands` 生成 zscale，并把 catalogs/targets/manifest 同步到 `--fast-root`。

zscale cache 的路径为：

```text
<zscale-root>/<tract>/<patch>/cutouts/<tile>__<bands>__hdu<N>.pt
```

例如：

```text
/nvme0/zc/scarlet/cellect_zscale_cache/9813/8,8/cutouts/grid_r00_c00_x0_y0__HSC-G_HSC-R_HSC-I__hdu1.pt
```

预处理结束后会在 root 下写：

```text
preprocess_manifest.json
preprocess_failed_patches.json
preprocess_failed_patches.csv
```

如果有坏 patch，先看 `preprocess_failed_patches.csv`，修复对应下载文件后只重跑这些 patch。

## 3. 训练数据加载

脚本：`astro_train_eval.py`

训练脚本可以直接读取多 patch 结构。推荐 SSD 训练方式：

```bash
torchrun --standalone --nproc_per_node=2 astro_train_eval.py \
  --mode train \
  --root /nvme0/zc/scarlet/preprocessed/9813 \
  --image-cache-dir /nvme0/zc/scarlet/cellect_zscale_cache/9813 \
  --bands HSC-G HSC-R HSC-I \
  --model-variant fused_encoder \
  --batch-size 8 \
  --num-workers 32 \
  --shape-source sdss \
  --source-filter nchild0 \
  --out-dir output/ddp_run
```

这里的路径关系是：

```text
--root            -> 包含 <patch>/reference_catalogs、targets、manifest 的目录
--image-cache-dir -> 包含 <patch>/cutouts/*.pt 的 zscale cache 目录
```

如果 `--root` 指向更高一级，也可以：

```bash
--root /nvme0/zc/scarlet/preprocessed
--image-cache-dir /nvme0/zc/scarlet/cellect_zscale_cache
```

此时脚本会递归发现：

```text
<root>/<tract>/<patch>/reference_catalogs
<image-cache-dir>/<tract>/<patch>/cutouts
```

如果 `--root` 指向 `/nvme0/zc/scarlet/preprocessed/9813`，则 zscale root 也通常应指向 `/nvme0/zc/scarlet/cellect_zscale_cache/9813`。

## 4. DDP 和 batch size

用 `torchrun` 启动时会自动启用 DDP：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 astro_train_eval.py ...
```

`--batch-size` 是 per-device/per-rank batch size。全局 batch size 为：

```text
global_batch_size = batch_size * world_size
```

## 5. train/val 切分

当前有两种切分方式。

### 5.1 默认按 cutout 随机切分

默认行为是把所有 tile/cutout 混合后随机切分：

```bash
python astro_train_eval.py \
  --root /nvme0/zc/scarlet/preprocessed/9813 \
  --image-cache-dir /nvme0/zc/scarlet/cellect_zscale_cache/9813 \
  --val-fraction 0.15
```

可用 `--seed` 固定随机划分：

```bash
--seed 7
```

也可以强制某些 tile 进入 val：

```bash
--fixed-val-names grid_r02_c04_x18204_y20924 grid_r03_c04_x19000_y21000
```

注意：`--fixed-val-names` 是 tile/cutout 名，不是 patch 名。

### 5.2 按 patch 整体随机切分

使用 `--patch-val` 后，会先按 `(tract, patch)` 分组，然后随机选择一部分 patch 作为 val。一个 patch 内的所有 tile 会整体进入 train 或 val：

```bash
torchrun --standalone --nproc_per_node=2 astro_train_eval.py \
  --mode train \
  --root /nvme0/zc/scarlet/preprocessed/9813 \
  --image-cache-dir /nvme0/zc/scarlet/cellect_zscale_cache/9813 \
  --bands HSC-G HSC-R HSC-I \
  --patch-val \
  --val-fraction 0.15 \
  --seed 7 \
  --out-dir output/patch_val_run
```

这种方式更适合评估模型对新天区/新 patch 的泛化，但 val 指标方差会比 cutout 随机切分更大。

训练开始后，`output/<run>/run_config.json` 会记录：

```text
num_records
num_train
num_val
val_record_names
args
```

可用它检查本次实际 train/val 切分。

## 6. 评测

评测模式会把发现到的 records 全部作为 val/eval 数据：

```bash
python astro_train_eval.py \
  --mode eval \
  --root /nvme0/zc/scarlet/preprocessed/9813 \
  --image-cache-dir /nvme0/zc/scarlet/cellect_zscale_cache/9813 \
  --bands HSC-G HSC-R HSC-I \
  --checkpoint output/ddp_run/best.pt \
  --model-variant fused_encoder
```

多卡 eval 可以用 `torchrun` 启动，但当前实现只有 rank 0 执行 eval；通常单卡 eval 就够了。

如果只想在指定 patch 上评测，可以加 `--eval-patches`：

```bash
python astro_train_eval.py \
  --mode eval \
  --root /nvme0/zc/scarlet/preprocessed/9813 \
  --image-cache-dir /nvme0/zc/scarlet/cellect_zscale_cache/9813 \
  --bands HSC-G HSC-R HSC-I HSC-Z HSC-Y \
  --checkpoint output/ddp_run/best.pt \
  --model-variant fused_encoder \
  --use-ex-link-postprocess \
  --eval-patches 8,8 8,9
```

`--eval-patches` 支持 `8,8` 和 `9813/8,8` 两种写法。patch 较多时可使用文本文件：

```bash
--eval-patches-file val_patches.txt
```

文件每行一个 patch，支持 `#` 注释。

## 7. 常见问题

### 找不到 zscale cache

报错类似：

```text
No FITS cutouts are present ... expected precomputed zscale cache ...
```

通常是 `--root` 和 `--image-cache-dir` 的层级不匹配。检查两者是否都从同一层开始：

```text
--root /nvme0/zc/scarlet/preprocessed/9813
--image-cache-dir /nvme0/zc/scarlet/cellect_zscale_cache/9813
```

或：

```text
--root /nvme0/zc/scarlet/preprocessed
--image-cache-dir /nvme0/zc/scarlet/cellect_zscale_cache
```

### 预处理有坏 patch

查看：

```bash
cat /data1/czh23/preprocessed/preprocess_failed_patches.csv
```

修复下载后，只重跑坏 patch：

```bash
python astro_data_preprocessing.py \
  --coadd-root /data1/czh23/Subaru \
  --catalog-root /data1/czh23/Subaru \
  --patches 1,5 6,3 \
  --output-root /data1/czh23/preprocessed \
  --fast-root /nvme0/zc/scarlet/preprocessed \
  --zscale-root /nvme0/zc/scarlet/cellect_zscale_cache \
  --overwrite \
  --overwrite-zscale
```

### DDP 报 unused gradient

如果看到：

```text
Expected to have finished reduction in the prior iteration
```

先确认代码包含 `CenterEnhancementNet2D.conv_l2` 的 forward 修复；该问题来自 CEN 中定义了参数但未参与 forward。当前版本已经修复。
