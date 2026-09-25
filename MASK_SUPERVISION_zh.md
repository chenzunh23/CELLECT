# 实例 mask 部分监督（2026-09-25 第一版）

适用于现有训练 Zarr，无需重新生成数据。通过 `--mask-loss-weight > 0` 开启 SAM mask 分支；外层权重默认仍为 0，避免无意启动 mask decoder 训练。

## 本版设定

- outside：在 **1.5 倍 Kron 半轴**的椭圆之外计算概率质量比例。仅放宽 outside 区域，不放大 prompt bbox，也不改变面积比的 Kron 参照。`ellipse_sigma` 沿用 1。
- area：`r = 可见预测软面积 / 可见Kron椭圆面积`，允许 `0.05 <= r <= 2`。图像内部用解析 `pi*a*b`；Kron 椭圆跨图边界或无效区时使用可见部分面积。
- 面积惩罚为 `relu(log(0.05)-log(r+eps)) + relu(log(r+eps)-log(2))`；不再使用原先的0.1截断，r<0.05仍有梯度。
- centroid：`sqrt((cx-x0)^2+(cy-y0)^2) / sqrt(a*b)`。距离换算回原图像素坐标，a、b取Kron半轴（最小1像素），不取1.5倍outside半轴。centroid继续用于有、无mask标签的prompt。
- max-area保留原来的整图面积保险：ratio默认0.5、权重0.1。本版没有全局放宽它。

## BCE、Dice及权重

- 正像素：当前实例mask；支持独立重叠的父/子实例。
- 负像素：Zarr中 `band_pu_class_mask==4` 的可信背景，且元数据明确为SExtractor方法，同时通过 `band_valid_mask`。
- 未知、ignore、invalid及其他实例的正mask不作为负例。没有SExtractor来源声明的旧Zarr，不猜测其背景来源，不提供这部分负监督。
- 有mask：正/负分别平均的BCE，再平均存在的正负两项；有标签源才计算Dice。
- 无mask：仅可信背景的负BCE，Dice严格为0，不使用椭圆替代真实实例mask。中心和几何先验继续提供约束。
- Dice仅在本实例正区域与可信负区域中计算，未知区域不进入分母。这是部分标签监督，不声称mask外全部是真背景。
- decoder低分辨率时按面积覆盖率缩小正mask，保留1像素小源；负像素要求对应区域全部为可信背景。正区域的监督域也按覆盖率加权，未观察/未知的子像素不被当成负例。
- `--mask-dice-weight 1`、`--mask-bce-weight 1`，共同再乘 `--mask-supervision-weight 0.2`；有mask实例再乘存储的可靠度（本批通常0.25），没有除以可靠度和，因此0.25不会抵消。有标签监督的有效系数通常为 `外层mask权重 × 0.2 × 0.25`。
- outside、area、centroid继续由各自权重控制，不乘上述BCE/Dice倍率。原pred-IoU/stability默认及实现未在本次改动中重定义。

## 取点和边界

- 每张图/每个波段先保留所有能按source ID关联到有效中心与shape的mask实例。
- 其余GT/pred点按epoch调度补足总预算128；有mask的源超过128时全部保留，不再增加普通点。
- `--mask-max-gt-per-sample` / `--mask-max-pred-per-sample` 默认均为128；两者不同则在混合阶段插值得到预算，非正数保留不限制普通点的选项。
- 带标签的GT实例不随epoch30后的predicted-only调度消失。普通prompt的GT/pred比例通过取点数控制，不再对已经混合的两组重复施加比例系数。
- 同ID的GT点不会重复抽取；与必选GT中心距离<=1像素的预测点不重复加入。独立父子实例按ID保留，即使中心相近也不互相覆盖。
- 所有mask截到512小块和有效覆盖内，不因碰到分块边缘而删除。bbox及mask同步处理水平翻转。
- 只有一段mask边缘落在块内、而源中心不在当前块的实例，因为没有有效中心prompt，不凭空造中心。跨4096的大源特写已有独立512样本，可提供居中监督。
- CPU batch保存bbox mask，普通实例不用提前展开成全512平面；SAM loss再按实际被选prompt生成监督。

## 显式参数示例

以下是追加到已有训练命令的mask相关参数，不是完整训练命令。本次没有启动正式训练。

```bash
--mask-loss-weight 1.0 \
--ellipse-sigma 1.0 \
--mask-outside-kron-scale 1.5 \
--mask-area-ratio-lower 0.05 \
--mask-area-ratio-upper 2.0 \
--mask-centroid-weight 0.2 \
--mask-outside-weight 0.5 \
--mask-min-area-weight 0.1 \
--mask-max-area-weight 0.1 \
--mask-max-area-ratio 0.5 \
--mask-bce-weight 1.0 \
--mask-dice-weight 1.0 \
--mask-supervision-weight 0.2 \
--mask-max-gt-per-sample 128 \
--mask-max-pred-per-sample 128 \
--mask-selection loss
```

不要不加检查地沿用旧README示例的外层mask权重5及area权重1；area/centroid定义已改变，同名loss数值与旧训练不能直接比较。若明确传入旧BCE/Dice权重0，则对应监督仍会关闭。

`--mask-min-area-px`为兼容旧参数保留，本版没有用它删除小mask。mask最大面积等其他限制也不是硬删除实例。

## 代码与验证

- `utils/instance_mask_data.py`：生产Zarr reader读取、重叠父mask恢复、有效区域和SExtractor背景、水平翻转。
- `astro_train_zarr_data.py`、`astro_train_data.py`：Dataset与collate传递稀疏实例。
- `sam_backbone/mask_supervision.py`：必选实例、总预算、部分监督域与面积区间loss。
- `sam_backbone/losses.py`：SAM调用、outside/centroid/area及完整loss汇总。
- `astro_train_ops.py`、`astro_train_eval.py`：参数与日志，新增 `mask_supervised_prompts`。

日志中的dice/bce已经包含supervision multiplier与实例可靠度，但没有乘各项weight及最外层mask weight。gt_total与pred_total记录对总loss的贡献，二者相加为total；supervised_prompts记录真实参与mask监督的点数。

验证记录位于 `/home/czh23/analysis/2026-09/2026-09-25/mask_loss_audit/implementation/`：包括边界裁剪、重叠、128点例外、epoch30、未知区域零BCE/Dice梯度、小mask缩小后不丢失、面积下限梯度、归一化centroid，以及实际COSMOS1727/5893、HSC half/noisy、Abell读取/翻转/反向传播检查。实际Zarr检查使用轻量mock decoder，并非正式训练或收敛验证。
