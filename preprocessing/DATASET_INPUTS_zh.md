# 统一图像路径与加载入口

默认路径集中在 `preprocessing/dataset_paths.json`，包括：

- HSC：原有 Subaru、refit、denoised、weight、Gaia 路径。原命令行参数仍可覆盖配置。
- COSMOS：`/data/czh23/JWST/COSMOS_1727_5893/manifest.json`，包含 129 张正式产品及其质量掩膜。
- Abell 训练图：`/data/shared/JWST/abell2744/stage3_fmodel/mosaic_v2_half/field_ra3p573_dec-30p376`。
- Abell 星表筛选参考图：`/data/shared/jwst_foundation/raw/Abell2744`，使用完整 coadd。
- Abell 星表资源：`/data/shared/jwst_foundation/catalog/A2744`。

## 生成输入清单，不生成 Zarr

在 `/home/czh23/CELLECT` 中运行：

```bash
/home/czh23/miniconda3/envs/jwst_env/bin/python -m preprocessing.build_image_level_zarr \
  --datasets hsc cosmos abell --dataset-sources coadd \
  --output-root /home/czh23/analysis/2026-09/2026-09-23/cellect_dataset_paths \
  --list-inputs
```

生成 `input_inventory.json`，记录训练图、参考图、各自的 shape/HDU/unit、proposal、质量掩膜和样本标识。
`--datasets` 是天区/数据集选择，`--dataset-sources` 仍是 HSC 的 coadd/denoised/noisy 选择，含义不同。
JWST 可加 `--jwst-bands F115W F444W`；COSMOS 可加 `--cosmos-proposals 1727 5893 --cosmos-pointings 11 19`。
Abell 自动识别有效 FILTER/PUPIL，F150W2+F162M 记为 F162M，不误归入 F150W。
每波段单独保留自身 WCS 和尺寸，不假设多个波段或者 half/full 共用像素网格。

可用 `--paths-config local_paths.json` 提供部分路径覆盖，未指定的项沿用默认配置。
相对路径按配置 JSON 所在目录解析；HSC 显式 CLI 参数优先于配置。

```json
{"abell": {"image_root": "/path/to/half", "reference_root": "/path/to/full"}}
```

## Python 加载

```python
from preprocessing.dataset_inputs import load_paths, discover_abell, discover_cosmos, load_image

paths = load_paths()
abell = discover_abell(paths["abell"], bands=["F444W"])[0]
training = load_image(abell, role="training", origin=(18000, 18000), shape=(512, 512))
reference = load_image(abell, role="reference", origin=(18000, 18000), shape=(512, 512))

cosmos = discover_cosmos(paths["cosmos"], proposals=[5893], pointings=[19], bands=["F115W"])[0]
cut = load_image(cosmos, origin=(10000, 10000), shape=(512, 512))
```

上述两个 Abell cutout 使用各自网格的同名像素坐标，仅演示读法，不代表同一天区！
`reference_to_training(source, x, y)` 将完整参考图零基像素坐标经天空 WCS 映射到训练图。
椭圆形状需用现有 `utils.segmentation.transform_ellipses` 转换。
已冻结的筛选结果可继续通过 `utils.frozen_sources.prepare_frozen_variant` 映射到 half coadd。

返回 `LoadedImage`：

- `image`：原始线性 float32 图像，保留 NaN；不做单位转换、缩放或背景扣除。
- `bad`：非有限 SCI、非正/非有限 WHT、TRAIN_BAD、DQ DO_NOT_USE、有效的 MASK 标志及 COSMOS NPZ 掩膜的并集。
- `header`：该 cutout 的 WCS/观测元数据；`full_header`：原始完整图 header。
- `origin`：本次截取的零基 `(x,y)`；`path`：实际读取的训练图或参考图。

读取参考图时不套用训练图的 NPZ，因为两者可能具有不同像素网格。
Abell 当前产品只有单个 SCI 主 HDU，读取器支持 HDU 0；COSMOS 支持 SCI 扩展；HSC 继续支持原格式。
缺失/形状不匹配的显式质量掩膜会报错，不会静默丢弃 P30 的 bad 区域。
大图优先传 `origin/shape`，切片直接从 FITS 读取；COSMOS 压缩 NPZ 必须解压完整掩膜后再切片。

## 与现有 Zarr 写入器连接

`build_image_level_zarr.load_registered_image` 是同一个 `load_image` 接口。
已有完成分类、且映射到训练网格的 `PatchLabels` 可这样写入：

```python
from preprocessing.build_image_level_zarr import write_classified_patch

result = write_classified_patch(
    task, linear_prepared_training_image, final_patch_labels,
    origin=training.origin,
    input_source=abell,
)
```

`input_source` 自动把输出归入：

```text
<output>/cosmos/proposal_1727/image_level/coadd/F115W/0019.zarr
<output>/cosmos/proposal_5893/image_level/coadd/F115W/0019.zarr
<output>/abell/image_level/half_coadd/F444W/field_ra3p573_dec-30p376__half.zarr
```

样本名包含 dataset/proposal，Zarr attrs 同时保存 `image_fits` 和 `reference_fits`。
HSC 既有输出路径不变。JWST 原始 MJy/sr 图像仍需按现有 `prepare_image` 流程转换一次，再交给写入器；不能把完整 coadd 的图像像素作为半数训练图写入。
生成 dense targets 前，应将 `LoadedImage.bad` 合入 `quality_ignore_mask`；填补 NaN 不取消该掩膜。

本次完成的是路径发现、数据加载和已有标签的写入绑定，**没有新增或更改星表筛选策略，也没有运行完整标签/Zarr 批量生产**。
CLI 的自动分类执行仍是 HSC；JWST 不会误走 HSC meas/refit 筛选。命令行选 JWST 而不指定 `--list-inputs` 会明确提示使用对应标签 adapter 与 `write_classified_patch`。
后续批量接入应先在 Abell full coadd 上筛选/冻结星表，再映射到 half coadd；不能用半数图重做这一步筛选。
