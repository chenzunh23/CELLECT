# Abell 共同 WCS、4096 父块与 512 训练样本

## 固定网格

参考 WCS 来自原始完整 F444W FITS；原 `f250m_center` 蓝框的零基起点 `(8064,19102)` 为网格 `(0,0)`。父块尺寸 4096，步长 3968，邻块重叠 128。网格向正负方向延伸，不移动边缘块；超出覆盖部分为 NaN。块名如 `x+00_y+00`。

全部 20 个有效 FILTER/PUPIL 波段分别投影到这一网格。SCI 使用有限支撑归一化的双线性插值；保留 MJy/sr，不乘像素面积。此操作会改变原生采样和噪声相关性。数值背景和 NaN 不被当作零通量混合。参考完整 coadd 与训练半数 coadd 独立重投影；不会用完整图填补训练图。

训练半数父块只要有一个有限像素就写出，并保存对应完整参考块。输出位于数据盘 `parents/<band>/*_half.fits` 和 `*_full.fits`。不会把整张 mosaic 一次装入内存，也不会额外写一套全尺寸重投影 mosaic。

## 512 样本与大源

每个父块内规则小块为 512、步长 512（父块间仍重叠 128）。填充前无效面积超过 10% 就跳过。即使父块没有可用 512 样本，其 4096 FITS 仍保留。无效像素记录于 `band_valid_mask`，图像占位为 0，confidence/shape/segmentation loss 权重为 0，不会被当成背景。

大源是在最终星表筛选之后增加的样本，不是新的源分类规则：

- Abell：final clean/weak_shape，原始 Kron 半长轴 `a>100` 像素。
- COSMOS：final clean，原标准 `a>100 OR a>150 OR sqrt(a*b)>100`。
- 中心 `floor(center+0.5)`；512 居中裁切，不为贴边源移动中心。大小使用 dense target 的 100 像素截断之前的值。
- Abell 每个源/波段归最近父块中心负责，防止重叠父块重复增加同一个大源。中心图可跨父块边缘，直接从原始半数图映射读取；不把父块边缘误认为 FITS 边缘。父块外尚未分类的区域使用 ignore 标签，不猜测背景。
- 居中样本与负责父块使用同一套 4096 scaling 统计量；不会各自重新估计 512 的训练 scaling。

## 参考筛选与训练背景分开

星表筛选使用完整 coadd 参考父块，调用既有 `classify_catalog_basics`、`ordinary_a2744` 和 `bright_label_jwst`。Gaia 阈值、足迹生长和孔径 SNR 规则沿用现有实现。参考图的孔径 SNR 背景现由 `utils/jwst_background_sextractor.py` 在完整 coadd 参考父块上生成多尺度 SExtractor sky mask；输出 `background.npz` 的 `background_mask=True` 表示可用背景。缓存签名包括输入内容哈希和参数，旧 LSST mask 会自动失效。

**正式训练背景是半数训练父块的单尺度 SExtractor segmentation==0 且 SCI 有效区域；参考图 SNR 则使用多尺度激进 SExtractor mask。两者均不是 SExtractor 数值背景图。** 之后再由完整 Kron 椭圆、亮区和 ignore 标签排除相应区域，防止已知源落入背景监督。

SExtractor 参数可调，当前默认是 DETECT_THRESH=1.5、MINAREA=5、BACK_SIZE=64、BACK_FILTERSIZE=3、DEBLEND_NTHRESH=32、MINCONT=0.005，3×3 Gaussian 检测卷积，额外 grow=0。这些是显式的新默认值，不宣称是旧 LSST 的等效阈值。提供 `--sex-detect-thresh`、`--sex-minarea`、`--sex-back-size`、`--sex-grow`。配置、输入校验值、mask 和日志写入 `labels/<band>/<parent>/training_background.*`。SExtractor临时 FITS 随运行完成删除。

`utils/jwst_background_sextractor.py` 也可从 CLI 对 COSMOS 完整图或局部视图生成同格式mask；单尺度工具 `utils/sextractor_background.py::sextractor_background` 仍可供 COSMOS 使用；调用后将返回的 sky mask 传入 `fill_dense_regions(background_mask=...)`。COSMOS CLI 全自动源分类批次不在本次 Abell 调度中；已有最终 COSMOS PatchLabels 的 writer 已支持大源补充和 10% 无效门限。传入 writer 的 `valid_mask` 必须是填充前的 `~LoadedImage.bad`。

## 启动、继续与检查

```bash
bash /home/czh23/CELLECT/preprocessing/run_abell_preprocessing.sh
```

先对全部波段切父块，再生成 Zarr；默认切块 2 worker、Zarr 4 worker。每个任务结束进程退出，释放内存。可用 `CUTOUT_WORKERS=4 ZARR_WORKERS=4` 调整。

只规划可用 `build_image_level_zarr.py --datasets abell --abell-plan-only --output-root ...`；只切块 `--abell-stage cutouts`；只生成 Zarr `--abell-stage zarr`。测试可加 `--jwst-bands F444W --abell-parent x+00_y+00`（parent 限制只作用于 Zarr，切块仍覆盖所选波段）。

成功父块有 `labels/<band>/<parent>/complete.json`，包括最终输出清单及参数/输入签名；参数变化不会静默复用旧结果。未完成的父块可重跑。`manifest.json` 记录所有保留/空父块；`zarr_summary.json` 记录样本数量及失败。

网格不同应使用新的 output-root，或显式 `--overwrite`。不要同时手工启动同一父块；整套 bash 使用文件锁防止重复 scheduler。

## 划分训练/验证

`make_zarr_patch_splits.py` 现在递归读取 Zarr attrs，支持 HSC、COSMOS proposal 和 Abell 父块名，不再只识别 HSC 的数字 patch 文件名。Abell 选择器示例 `half_coadd:Abell2744/x+00_y+00`，同时选入该父块的规则和大源样本。COSMOS 使用 `coadd:COSMOS_1727/<patch>` 等 tract 限定，两个 proposal 的同一天区不能跨训练/验证。

选定 Abell 验证父块后，默认禁止其相邻重叠父块进入训练；检查中额外考虑居中 512 图可能伸出父块的范围。不要打开 `--allow-val-train-physical-overlap`，除非有意允许泄漏。这里提供机制，未替用户决定验证集天区。

```bash
python preprocessing/make_zarr_patch_splits.py \
  --root /data/czh23/JWST/Abell2744_preprocessing/zarr \
  --dataset-sources half_coadd --bands F115W F150W F200W F356W F444W F070W F250M F480M \
  --train-counts half_coadd=20 --val-selectors half_coadd:Abell2744/x+00_y+00 \
  --allow-short --dry-run
```

## 旧依赖移除

`TileSpec`、tile grid、HSC FITS/catalog 路径与 detection span 读取等实际使用函数已迁入 `preprocessing/utils/inputs.py`。生产模块和相关诊断不再 import `astro_data_preprocessing.py`。有单元测试确认导入新 builder 后该旧模块不在 `sys.modules`。

## NaN 中心源与运行状态（2026-09-23）

原始星表进入初始筛选时，使用填补 NaN 之前的图像/有效像素掩膜检查中心。中心无数据的源统一标为 `ORDINARY_IGNORE`（原因 `center_no_data`），保留 ignore 椭圆，不进入后续面积、近邻、SNR 候选。普通分支也支持传入同一有效性信息，防止独立调用绕过该规则。有限的零值不是 NO DATA。

第一步不判断 Gaia；Gaia 仍由最后的独立分支插入。Zarr 写入保留已插入 Gaia 的中心记录，但原图无效像素的训练权重始终为零。

本次已完成 20 波段父块切割（570 个非空父块），仅运行少量 Zarr 试验。用户要求暂不运行正式处理，未启动全量标签/Zarr 生成。NaN 策略修改后通过 78 项单元测试，既有试验 Zarr 尚未按最终策略重建，不应作为正式训练产品。

## 第一版训练 confidence 更新

PSF matched 已接入正式公共 Zarr 写入层，JWST 默认采用 FWHM `[1.6,8]` 像素截断；Abell 启动脚本也显式选用此配置。详见 `CONFIDENCE_zh.md`。历史 pilot 未重建，本次未启动正式处理。
