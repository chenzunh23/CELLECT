# 2026-09-23 FWHM confidence 历史配置

`jwst_confidence_fwhm_v1.json` 保存昨日定义。显式 `--confidence-mode psf-matched` 可使用它。

每波段固定标称 FWHM 除以输出 WCS 像素尺度，默认截断到 [1.6,8] 像素。3/2/1级外半径分别为 FWHM 的 0.5/0.75/1.25 倍，最近像素以 floor(center+0.5) 设为4，重叠取最大等级。命令行仍支持旧版FWHM上下限与实测像素宽度覆盖参数。

此标称表来自历史 psf.py，不是本次模拟FITS实测，也不是完整照抄官方某一列。保存它用于旧实验复现；新的JWST auto默认使用EE定义，HSC保留曼哈顿定义。
