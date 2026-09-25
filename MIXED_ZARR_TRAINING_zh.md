# HSC / COSMOS / Abell 单图训练与评测

入口：`scripts/train_mixed_zarr.sh`。这次修改不启动正式训练。

## 修复内容

- 每个 Zarr 通过 `bands` 元数据传递真实 filter；单图输出 channel=0 不再被当作全局第一个 filter。检测计数、候选调试统计、CSV 都使用实际名称。
- `--detection-linking` 默认关闭；image-level Zarr 强制关闭。旧的“没有 EX 就自动邻波段 linking”不再隐式执行。同一源在不同图像中的检测独立计数。
- `per_band` 给出每波段 TP/FP/FN、precision/recall/F1 和 samples/evaluated；`overall` 是汇总计数后计算的 micro 指标，不是各波段 F1 的平均值。无验证样本的波段标记 evaluated=false。
- train/val+detect、独立 eval 都支持 `--zarr-random-image-batches`；`--bands all` 自动发现完成的单图 stores。NO DATA 中的预测中心被过滤，普通 ignore 保留原来的评测处理。
- Zarr 导出 CSV 从父块 WCS 和 tile_x0/y0 计算坐标；角距离使用各图 confidence 配置中的像素尺度。缺失 WCS 的旧 store 仍输出 NaN 天球坐标。
- 数字 half/noisy group、Abell `half` group 都能被清单选中。清单选择改成一次索引，避免数千 selector 反复全量扫描。

## 区域选择：沿用 train/val 清单

仍使用 `--train-patches-file` 和 `--val-patches-file`，每行一个区域或带 group 的 selector。未列入清单的区域不进入对应集合；不再默认使用上一轮 206488/24786 的大范围随机划分，不增加 test set。

启动脚本支持 `TRAIN_FILE` / `VAL_FILE` 环境变量，也支持在末尾直接传入原有两个 CLI 参数。必须明确提供清单；`SPLITS=/path` 可选，用于读取该目录的 train.txt/val.txt。不指定清单时只报缺少清单，不启动训练。

旧文件 `patches/train_zarr_v3.txt`、`patches/val_zarr_v3.txt` 未修改。新 root 的 HSC 输入名称是 half_coadd，因此另外提供同一区域的适配文件：

- `patches/train_zarr_psfee_hsc.txt`：当前为 72 条 half_coadd 区域及原 40 条 noisy group，共选入 58367 小块。
- `patches/val_zarr_psfee_hsc.txt`：原 5 条 selector，同样映射；实际选入 2802 小块。

这两份文件**只包含 HSC**，保留作独立子集；当前指定的混合区域使用下面的新清单。自选清单的 selector 例如 `coadd:COSMOS_1727/P0019_x08192_y08192`（具体 patch 名以实际 store 元数据为准）、`half_coadd:Abell2744/x+00_y+00@half`。大源特写有自己的 selector，应与其物理父区域分到同一侧；现有 `prepare_mixed_zarr_splits.py` 的物理组规划逻辑可供参考，但当前脚本不会自动运行它。

沿用旧 HSC 区域的启动示例：

```bash
TRAIN_FILE=/home/czh23/CELLECT/patches/train_zarr_psfee_hsc.txt \
VAL_FILE=/home/czh23/CELLECT/patches/val_zarr_psfee_hsc.txt \
CUDA_VISIBLE_DEVICES=0 \
bash /home/czh23/CELLECT/scripts/train_mixed_zarr.sh
```

同样可使用原有 CLI 形式：

```bash
bash /home/czh23/CELLECT/scripts/train_mixed_zarr.sh \
  --train-patches-file /path/to/selected_train.txt \
  --val-patches-file /path/to/selected_val.txt
```

多卡仍用 `NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3`；batch/worker 是每卡数值。

## 当前生效的混合区域（HSC half 扩展为72块）

使用 `patches/train_zarr_psfee.txt` 与 `patches/val_zarr_psfee.txt`。已保留用户手动删减的 COSMOS/Abell 清单；仅更新 HSC half 的训练选择，val 逐字节未修改。

HSC half 选取 0..8 × 0..8 中除 4,5、5,5、4,4、5,4、3,4、3,5、6,1、6,2、6,3 外的72块。当前修改前文件实际列有60块，本次小块数由29163增至34167。HSC noisy保持原选择。

| 数据集 | Train | Train占比 | Val | Val占比 |
|---|---:|---:|---:|---:|
| HSC half_coadd | 34,167 | 25.18% | 1,042 | 6.90% |
| HSC noisy | 24,200 | 17.84% | 1,760 | 11.66% |
| COSMOS 1727 | 51,731 | 38.13% | 6,381 | 42.26% |
| COSMOS 5893 | 4,500 | 3.32% | 1,481 | 9.81% |
| Abell | 21,089 | 15.54% | 4,434 | 29.37% |
| 合计 | 135,687 | 100% | 15,098 | 100% |

HSC 合计占训练集43.02%。比例按已生成的512×512单波段样本统计，包含大源特写；未修改HSC EDGE质量过滤。

按要求保留 val 中 noisy:8,4@group_03，同时新增 half_coadd:8,4 到训练。这是同一片天空的不同输入产品；虽然样本记录没有重复，物理天区仍共享。Abell相邻父块仍保留原128像素重叠。

```bash
bash /home/czh23/CELLECT/scripts/train_mixed_zarr.sh \
  --train-patches-file /home/czh23/CELLECT/patches/train_zarr_psfee.txt \
  --val-patches-file /home/czh23/CELLECT/patches/val_zarr_psfee.txt
```

逐区域占比见 `analysis/2026-09/2026-09-25/mixed_zarr_training/hsc72_region_fractions.csv`，逐波段及前后比较见同目录 `hsc72_regions_summary.json`。只读核验脚本为 `summarize_current_lists.py`。旧 `build_requested_regions.py` 是历史固定方案，不应用它重新生成并覆盖手动删减后的清单。

## 数据数量审计

root: `/data/czh23/analysis/2026-09/2026-09-25/training_zarr_psfee_v1/zarr`

完成 stores 的总样本数是 **243459**，不是默认应全部用于训练的数量。其中 HSC half 39349、HSC noisy 48180、COSMOS1727 122940、COSMOS5893 7467、Abell 25523；均包含已生成的大源特写。全部 confidence 配置为 PSF-EE。

0827 日志中的 132320 是 discovery 总数，实际 train=62283、val=2178。该次使用已不在原路径下的 direct_zarr_v4_lupton，配置选择 coadd/noisy；不能把运行名里的 half 当作当前 half_coadd 输入的证明。旧配置保存的 2178 个验证样本全部是 coadd，没有实际选入文件中列出的 noisy group。当前数字 group 适配已修复，所以沿用 selector 后验证数量也不会完全相同。

当前 HSC half 的 724 个输入中，229 个返回 no_valid_tiles，495 个写出 store。HSC-Z half 仅 6 个样本，存在明显过度过滤风险。实际检查 HSC-G/0,3、HSC-Z/4,5，SCI 全有限、NO_DATA 为零，但 EDGE 比例分别约 23.84%、72.95%，当前 `finite AND NOT(SAT|BAD|EDGE|NO_DATA|UNMASKEDNAN)` 再以无效面积>10%删块，使整张图无训练样本。应核对 half-coadd 的 EDGE 语义，不能把这些丢弃都归因于真实 NaN 覆盖。此轮只调查，未改动质量掩膜或重生成 Zarr。

审计文件位于 `/home/czh23/analysis/2026-09/2026-09-25/mixed_zarr_training/`：inventory_comparison.json、half_coverage_audit.json、legacy_regions_on_new_zarr.json。之前随机方案的 train.txt/val.txt 留作记录，启动脚本不再默认读取。

脚本使用 cellect 环境、SAM ViT-B 官方权重，默认 100 epochs。PSF-EE 标签直接从 Zarr 读取；confidence=ce_hard，保留 shape loss；mask 外层权重 1，BCE/Dice 额外系数 0.2 和各实例可靠度相乘，前 5 epoch 关闭 mask loss；prompt 预算 128，有标签实例必选且不受此上限限制，decoder 每次 32 prompts。outside=1.5 Kron，area=[0.05,2]，centroid 按 sqrt(ab) 归一化。DDP static graph 关闭以支持有/无 mask 的动态分支。

明确使用 `--match-radius 3`，即同为 3 个输出像素（HSC≈0.504″，JWST≈0.09″），避免套用统一 HSC 像素尺度。该命令没有改变原有 detection score/threshold。`--center-tolerance-arcsec` 在显式 match-radius 下不生效。

每 epoch 执行 val+detect。默认输出：`/data/czh23/analysis/2026-09/2026-09-25/cellect_mixed_psfee_v1`；`OUT=...` 可覆盖。检测指标保存 `detection_metrics_epoch_XXXX.json` 和 `detection_metrics_latest.json`；启用 W&B 时另记录 `val/detection/{filter|overall}/...`，默认 W&B disabled。没有 linking 结果文件。best.pt 是最佳验证损失模型。

只打印命令：在上述明确指定清单的命令前添加 `DRY_RUN=1`。
独立评测（默认 best.pt）：

```bash
MODE=eval VAL_FILE=/path/to/selected_val.txt CUDA_VISIBLE_DEVICES=0 \
  bash /home/czh23/CELLECT/scripts/train_mixed_zarr.sh
```

可用 `CHECKPOINT=/path/model.pt` 指定权重。eval 使用相同验证清单，输出 eval/eval_metrics.json 和源 CSV。脚本后续参数会传给 Python；例如 `--detect-every 5` 降低检测频率。`--checkpoint` 当前仅加载模型权重，不是完整 optimizer/scheduler 断点恢复。

验证：22 个局部回归测试；真实数据的全量清单匹配、4 类 store 读取/拼批、band 分桶和 CSV/WCS 检查通过。真实数据 smoke 使用已知中心模拟预测来测试评测接口，不代表模型精度；没有进行正式 GPU 训练或多卡压力测试。
