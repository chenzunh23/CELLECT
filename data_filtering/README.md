# CELLECT Data Filtering

这个目录存放“数据筛选标准”和诊断入口。原则上这里定义什么是
`clean`、`center_only`、`ignore`、`background`、亮源区和坏块；最终训练产物
如 dense target、zscale cache、zarr store 仍由外层 preprocessing 脚本生成。

## Core Documents

- `DATA_FILTERING_STANDARD.md`: 当前筛选标准，覆盖 coadd PU/AP2/SNR、non-coadd SNR、亮源筛选和窄带 bad-score。
- `DATA_WORKFLOW_0729.md`: 2026-07-29 版本的完整预处理流程，从 FITS/refit 到 SAM image-level zarr 和训练/评测入口。

## Shared Code

- `data_standards.py`: 宽带/窄带、饱和星等、PU、AP2-SNR、variance/weight-ratio SNR 的共享常量。
- `pu_config.py`: 把 `astro_data_preprocessing.py` / direct zarr 的 CLI 参数规整成运行时 PU 配置。
- `pu_source_filter.py`: 星表级 PU 分类，包括 Kron refit 半径接入、A/B filter、AP2-Kron、shape/flag/overlap 筛选。
- `noncoadd_snr.py`: noisy/denoised 图像上的 AP2 aperture + annulus SNR 测量和可见性降级。
- `sam_input_scaling.py`: SAM/CELLECT 共用的单波段 scaling 与亮区 mask 定义，包括 `log-lupton`、`anscombe`、`raw/none`。
- `calexp_quality.py`: calexp MASK plane 质量评分，供窄带 patch/tile bad-score 过滤和诊断脚本复用。

## Diagnostics

- `analyze_calexp_mask_quality.py`: 按 patch 统计 MASK plane 比例和 bad score，支持 patch 级并行。
- `overlay_calexp_tile_bad_score.py`: 在 zscale 图像上叠加 tile bad-score 区域。
- `visualize_sam_input_scalings.py`: 对比 zscore、no-first-clip zscore、asinh/Lupton/log/Anscombe scaling。
- `build_external_bright_labels.py`: 第一版亮源外部星表/亮区诊断。
- `build_external_bright_labels_v2.py`: 当前亮源重分类诊断，生成 priority partition 和 REG/CSV。
- `variance_snr_diagnostics.py`: 用 variance plane 估计 noisy 可见性。

旧路径 `scripts/analyze_calexp_mask_quality.py` 和
`scripts/overlay_calexp_tile_bad_score.py` 仍保留为兼容 wrapper，实际实现已移动到这里。
