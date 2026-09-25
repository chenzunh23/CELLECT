# COSMOS segmentation：raw几何、填洞平滑与重叠筛选

实现：`utils/segmentation.py::isolated_segmentation_targets`，策略版本 `raw_nested_fill_gaussian_overlap_v3`（2026-09-25）。`build_image_level_zarr.write_classified_patch` 调用 `utils/image_level.attach_cosmos_segmentation`，默认启用下列规则。先在父图完成处理，再切512训练块。只改变辅助segmentation，不修改源类别、Kron、dense或confidence。

## 筛选顺序与默认参数

1. 已知DROPPED源从参与集合排除，其余类别及未知ID都作为阻挡源；只有CLEAN/WEAK_SHAPE可作为训练标签。
2. 在任何连通域清理、填洞或平滑之前，固定**raw像素面积与严格包含关系**。子源全部raw像素必须在父源raw外边界围住的内部。采用8连通外部路径，斜向开口视为与外部连通。不能通过删掉子源外部碎片或平滑封口产生新的包含豁免。
3. 已知非dropped源主8连通域占raw面积≥`main_fraction_min=0.95`时保留主分量，并填洞。占比不足时不训练且保留raw作为阻挡；未知ID保守保留raw，不自行修形。
4. **raw_area > 100**的可处理源在填洞后做`float32 Gaussian(sigma=1.0原生像素, truncate=4, constant=0)`，以`>=0.5`二值化，再次保留主连通域并填洞。raw_area≤100只进行主连通域与填洞。面积判断绝不使用填洞面积或清理后面积。所有已知非dropped类别按同样规则处理，包括ignore/center-only。
5. 默认`clearance=0`：使用处理后的**独立二值mask**判断共同像素，非父子mask有共同像素则双方不能训练；仅相接、对角相接、相距不足8像素均允许。raw祖先/后代允许相互重叠，兄弟源也可以相接，但不能重叠。使用稀疏bbox网格查询，避免全量两两比较或反复卷积整图。
6. 保留处理后mask与亮星/无效像素相交保护、raw源贴父图边界的截断保护。生成端在筛选前合并用户valid_mask与science有限值，防止先接受再裁掉无效区。父图边界不同于后续512训练块边界。
7. 按raw祖先深度从内到外筛选：父源要求所有非dropped后代可训练，且处理后子mask仍包含在父mask内。父源不合格不向下拒绝子源。父子面积排序也不能依赖处理后面积；薄环父源raw面积可能小于子源。
8. `sources`中的`area`与`raw_area`均为raw像素数；`mask_area`单独记录最终像素数，`cleaned_area`记录主分量像素数。`gaussian_applied`记录是否启用高斯。默认`closing_radius=0`，不叠加旧闭运算。

底层函数保留显式非零clearance/closing参数供诊断，Zarr生成默认均为0。`allow_nested=False`关闭raw包含豁免。Zarr的`segmentation_policy`记录版本、raw面积阈值、sigma、二值阈值、包含/重叠依据，方便区分已有v2数据。已有Zarr不会被就地改写，需要重新生成才使用新定义。

## 节省空间的混合存储

普通孤立源及不重叠子源使用既有 `band_segmentation_ids` (N,B,H,W,int32) 和 `band_segmentation_weight` (float32)，正像素权重0.25，其他像素未知/权重0。填充父源在ID预览中被较小子源覆盖。

**只对 ID 图不能完整表达的重叠父源保存补充 mask**，普通源不逐实例存图：

- `segmentation_overlap_meta`：(M,8) int64，列为 sample、band、source_id、parent_source_id（无则-1）、x0、y0、height、width；坐标为训练小块内0基坐标。
- `segmentation_overlap_data`：所有父源局部包围盒二值 mask 的逐行 `packbits(bitorder='little')` uint8 串；每个实例独立补齐字节。
- `segmentation_overlap_offsets`：M+1个字节偏移。
- `segmentation_overlap_weights`：每个实例正像素权重。

编码 `bbox_packbits_little_v1`。512×512父mask若覆盖整块只需32KiB二值数据（不计Zarr压缩），不是一个256KiB的bool/uint8整图；更小bbox按其尺寸存。父图上先完成筛选和平滑，再同步裁切ID图和补充mask；不在512块边界重新判断孤立性。

## 训练读取

```python
from preprocessing.utils.segmentation_storage import iter_training_segmentation_masks
for target in iter_training_segmentation_masks(zarr_group, sample=0, band=0):
    source_id = target['source_id']
    mask = target['mask']                  # 独立bool图，允许父子重叠
    positive_weight = target['positive_weight']
```

普通源从ID图还原，有补充记录的父源使用完整补充mask覆盖还原结果，不能把二者当作重复实例。支持只有旧ID图的历史Zarr。预处理已接入保存与读取辅助函数，**未修改SAM训练端调用或loss**；训练端需显式采用此读取方法。现有positive-only语义保持不变，mask外不能自动视为可信负背景。

## 验证与范围

回归覆盖100/101 raw面积边界、小源填洞不高斯、相接兄弟可训练、清理外部碎片不能改变raw包含、ignore/未知阻挡、薄环父子排序、NaN保护，以及Zarr默认调用、切块、补充mask序列化和读回。

2026-09-25实测465128、465127、465129以及外围clean源464088均保留；465113/465114仍为ignore、465115仍为center-only，不因几何改善而改变类别。诊断位于`/home/czh23/analysis/2026-09/2026-09-25/segmentation_zarr_regularization`。
