# Confidence 生成选项（2026-09-24）

`build_image_level_zarr.py` / `write_classified_patch` 支持：

- `--confidence-mode auto`（默认）：JWST 使用 assets 定义的 `psf-ee`，HSC 使用原 `manhattan`。
- `--confidence-mode psf-ee`：原始归一化模拟PSF的 EE10/35/60/70 分别作为4/3/2/1级外边界。
- `--confidence-mode psf-matched`：保留2026-09-23 FWHM定义。
- `--confidence-mode manhattan`：保留原HSC核心。

## 当前 JWST 定义

```bash
--confidence-mode psf-ee
```

默认配置：`assets/jwst_confidence_ee10_35_60_70_v1.json`。含29个滤镜的PSF路径、扩展、校验和、实测角秒半径及能量阈值。使用OVERSAMP原始单位归一化，不除以有限stamp总和。

运行时读取冻结实测半径，除以训练图实际输出WCS像素尺度，绘制以源为中心的圆环；不重新读取大型PSF或逐源积分。可以用 `--confidence-config-path /absolute/path/config.json` 指定兼容配置。改变阈值必须同步重新测量半径。

4级保留r10内全部像素中心。若为空，使用floor(center+0.5)选一个最近像素补4；并列时选择较大坐标。图像边缘选择最近的现存数组像素。其它等级可以为空。源之间取最大等级；外圈之外为0；没有FWHM截断。最近点回退不绕过无效像素掩膜：writer随后会把无效像素标签和训练权重清零。

4级现在可以含多个像素，后续训练中心监督/解码器需要配套调整；本次只接入预处理，没有修改训练模型。

## 旧版可复现配置

```bash
--confidence-mode psf-matched --confidence-fwhm-min 1.6 --confidence-fwhm-max 8
```

旧表已保存到 `assets/jwst_confidence_fwhm_v1.json`，包含完整标称FWHM表及规则。`F=clip(FWHM_arcsec/output_scale,1.6,8)`，3/2/1级外半径为0.5/0.75/1.25F，4级是单个最近像素。标称值非当前coadd实测，亦非统一照抄官方某一列。

`--confidence-fwhm-pixels` 仍可为旧模式提供明确的像素FWHM；新EE模式拒绝该覆盖参数，避免静默忽略。显式FWHM上下限仅对旧模式有效。

## 接入范围与记录

- `StoreTask.confidence_mode`、`confidence_config_path` 和现有FWHM参数可供Python调用。
- `_tile_targets` 对clean、weak、strict center、后插入Gaia使用同一模式，保留源中心与ID。
- Abell启动脚本已改为EE模式，普通父块和居中大源块均使用其输出WCS；完成记录签名包含资产SHA256，防止旧auto结果误跳过。
- COSMOS公共 `write_classified_patch(..., input_source=source)` 同样适用。现有COSMOS自动星表适配整批CLI仍是独立待接入部分，本次未改变该编排限制。
- Zarr attrs 的 `confidence_config` 保存模式、定义版本、资产路径和SHA256、PSF来源、归一化、阈值、角秒/像素半径及回退规则。数组名称和0–4整数编码不变。
- 已有Zarr不会自动更新，未启动正式生成或训练。

## 验证

23项定向回归通过：临时Zarr写入/读回及配置记录、无效像素权重、旧FWHM兼容、HSC曼哈顿、EE多像素核心与空核回退、等距规则、边缘、源间重叠、WCS尺度及配置校验。
