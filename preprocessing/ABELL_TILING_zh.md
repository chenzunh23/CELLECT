# Abell 共同 WCS、4096 父块与 512 训练样本

## BLUE4：至少四蓝带中的三带通过（2026-11-09）

当前代码策略 `uncover_native_abell_v9_three_of_four_blue_band_psf`：候选检测波段为
F090W、F115W、F150W、F200W，至少三个波段独立原始 FULL 拟合 SNR≥4，且通过
紧致性、拟合质量、中心散布、模型核与近邻排除。F090W缺覆盖时可由另外三带确认。
对单个候选，只有其自身通过检测的波段才可直接补 strict 中心；未通过的第四带
和检测组之外的所有波段（也包括F070W、F140M等短波）均需本波段固定PSF SNR≥4。
不再按滤波器波长小于2.12微米就一律免PSF筛选。2像素去重换中心、原有紧致风险源
PSF≤4否决、Gaia与孔径SNR等原规则保留。

检测实现为 `utils/abell_blue_detection.py`，没有依赖analysis脚本。输入为已对齐
原始完整coadd及各波段模型；原图与模型/残余基底去除图共同产生SExtractor种子，
四带确认只在原始FULL上同时拟合像素积分Gaussian和局部二次背景，不用减模型后的
低通量决定源的SNR。风险区也保留原先F277W/F356W模型的贡献，它们不参加候选确认。
沿用先前通过审核的渐进阈值10/5/3/2/1.2/1、MINAREA=3、
DEBLEND_NTHRESH=64、MINCONT=1e-5等条件。解析Jacobian加速非线性拟合，不改变模型。

所有51个已规划空间父块的检测命令（独立于Zarr生成）：

```bash
cd /home/czh23/CELLECT
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/czh23/miniconda3/envs/cellect/bin/python -u \
  -m preprocessing.utils.abell_blue_detection \
  --workers 4 --fit-workers 4 --output-root /data/czh23/analysis/2026-11/2026-11-09/abell_blue4_all
```

可用 `--parents x+00_y-01` 只检测一块。输出逐块`verified_centers.fits`、全部种子及
原图拟合诊断表`all_candidates.fits`、REG、SExtractor文本日志、完成签名。
不保留大尺寸中间FITS。每块完成后进程退出；失败块记录于summary.json，成功块可续跑。
输入/配置/代码签名改变时重做；签名相同则保留合并候选文件mtime，避免误触发Zarr重建。

所有块成功后生成统一天空星表 `verified_blue_centers.fits`，跨块2像素去重，保留
origin_parent/origin_debug_id。表内DETBANDS/MINBANDS和逐源`<band>_detected`记录跨带条件。
统一生成入口现在自动完成/核验全天区检测，再生成 Abell Zarr：

```bash
bash /home/czh23/analysis/2026-11/2026-11-09/space_zarr_uncover_blue/generate_new_zarr.sh
```

该专用入口的默认检测输出是同目录 `blue_detection/`，默认4父块×4局部拟合worker，
由 `BLUE_PARENT_WORKERS`、`BLUE_FIT_WORKERS` 调整。SExtractor像素栈3200万、对象栈6万、
图像缓冲1024行，分别由 `BLUE_SEX_MEMORY_PIXSTACK`、`BLUE_SEX_MEMORY_OBJSTACK`、
`BLUE_SEX_MEMORY_BUFSIZE` 调整；这些是条目/行数，不是进程内存上限，0不表示无限。
即使SExtractor返回0，日志中出现栈溢出仍将该块判为失败，禁止合并/继续Zarr生成。
所有请求父块成功后才发布统一星表。检测日志同步写入Zarr生成总日志，逐块完成后回收进程。

去混叠溢出使用独立回退：只有该参考图/阈值改用1024核心区+64像素halo检测；中心只归属
核心区，转换回父块坐标后沿用2像素去重。若仍溢出，递归缩小核心至128，再失败则停止。
不改变检测阈值及原始FULL拟合确认条件，也不把局部footprint作为科学形状标签。
步骤星表保存输入摘要、结果哈希及完成回执；正常阈值无需重复执行。
已知旧版8c99f17...6727d8对任何overflow均拒绝发布父块，所以其成功结果在配置/资产及
星表哈希均一致时可直接复用；未知旧代码不能绕过签名。恢复步骤保存`tiled/recovery.json`。

此入口默认复用1105 COSMOS，并在1109 `training_zarr_space_uncover_blue_manhattan` 原路径
重建策略/星表签名变化的Abell；逐父块完整写好临时store后再替换旧store。中断后重跑时，
已经按新签名完成的父块会跳过。不需要先删除整个Zarr，COSMOS不重新生成。
独立检测入口为同目录 `run_blue_detection.sh`，可以先并行检测、稍后再运行生成入口。
`BLUE_DETECT=0 UNCOVER_BLUE_CATALOG=/path/catalog.fits` 可显式使用外部星表。
通用预处理默认内置星表仍保留三带旧资产用于兼容；只有上述专用入口自动接入全天区BLUE4。
不要将仅单块试验的合并表误用于全区域正式补源。

生成完成后，同目录 `prepare_split.py` 自动更新 `split/train.txt` 和 `split/val.txt`：
x+00_y+00、x-01_y-01整父块验证，x+01_y+00保留x=368/736/1104与y=2208/2576六个窗口；
训练排除与验证实际视场相交的所有波段/相邻父块/独立大源样本。COSMOS划分沿用1109配置。

## 已验证紧致源后处理（2026-11-08）

`use_uncover` 正式流程在原有分类之后、Zarr 写出之前，默认执行
`uncover_native_abell_v8_verified_blue_postprocess`。规则父块和独立大源中心窗口
调用同一个入口；混合数据批次也走这个入口。

1. 原孔径 SNR 5/7、各波段测光、flag、Gaia/亮区处理及既有 Kron 邻源分割占比
   30% 规则保留。bCG 模型仍只定义风险区，FULL 测光与 HALF 输入均不减模型。
2. 加载已验证的 BLUE 星表，将 RA/Dec 投影到当前波段/窗口。BLUE 要求
   F115W/F150W/F200W 三波段各自 SNR≥4，并保留检测实验的紧致性、位置一致性、
   模型核心排除等筛选。蓝端补 strict 中心；红端仅补本波段固定 PSF SNR≥4 的中心。
   不加入只在红端发现的源。这里的 PSF≥4 是补中心条件，与下项原有源的≤4否决
   条件不同，沿用批准的实验边界值。
3. 与已保留源按 2 个输出像素（当前0.03″/pixel）去重，重复时使用 BLUE 中心，
   保留原 ID 和椭圆半轴/方向，并记录旧中心。多个 BLUE 互距≤2时优先保留
   三蓝端最低 SNR 更高者；可调整 `UNCOVER_COMPACT_DEDUP_PIXELS`。
4. 换中心后，对所有波段风险区内原有的紧致 CLEAN/WEAK/STRICT 源进行本波段固定
   PSF 复核。紧致性为原检测二阶矩圆化半径
   `sqrt(a_image*b_image) × 原生检测像素尺度 ≤0.10″`，不是 Kron 半径。
   原有源 PSF SNR≤4（含4）、非有限或拟合失败即降为 ignore；不提升原已 ignore/dropped
   的源。亮源星等保护不绕过这项复核；Gaia 理论 PSF 行不进入该原有 UNCOVER 源分支。
5. PSF 拟合在原始 FULL 上固定位置/形状，联合邻源振幅与局部二次背景；使用经验
   per-pixel RMS 和空白位置噪声校准。红端 BLUE 测量与原有源复核复用同一噪声尺度。
   被否决的原有源不会在本轮以重复 BLUE 中心重新插回。随后重画 dense 并交给原
   Zarr writer；strict 只有 confidence 中心，不产生 Kron shape 正监督。

本次没有新增分割归属/大源内部峰抑制，因此距原中心超过2像素的独立内部峰仍可能保留。

默认候选是仓库内独立资产 `assets/abell_verified_blue_centers_v1.fits`，2019个天空中心，
由两块已验证结果（x+00_y+00、x+01_y+00）合并去重而来；对应 JSON 记录输入哈希、
选择条件和旧 C 编号映射。运行不 import analysis 目录脚本。该资产仅补充这两片
已验证天区及与其相交的窗口，不声称已在 Abell 其余区域重新检测；原有源 PSF 复核
则在所有处理区域生效。后续可用新验证的同格式天空星表替换它。

默认配置在现有 `run_abell_preprocessing.sh`：

```bash
UNCOVER_BLUE_CATALOG=.../preprocessing/assets/abell_verified_blue_centers_v1.fits
UNCOVER_COMPACT_RED_SNR=4
UNCOVER_COMPACT_DEDUP_PIXELS=2
UNCOVER_NATIVE_PSF_REVIEW=1
UNCOVER_NATIVE_PSF_SNR=4
UNCOVER_NATIVE_PSF_MOMENT_MAX=0.10
```

对应 CLI 为 `--uncover-blue-catalog`、`--uncover-compact-red-snr`、
`--uncover-compact-dedup-pixels`、`--[no-]uncover-native-psf-review`、
`--uncover-native-psf-snr`、`--uncover-native-psf-moment-max`。
显式 `UNCOVER_BLUE_CATALOG=''` 可关闭新中心补充；原有源 PSF 复核仍开启。
显式 `UNCOVER_COMPACT_ROOT` 可优先读取既有逐 parent/band 的已测 BLUE/PSF 表，
缺少某表时回退到天空星表，而不漏掉该波段的补中心步骤。

每个 `labels/<band>/<parent>/` 除原 sources.fits、reference_uncover.json 外，
新增 `reference_blue_center_selection.fits`（投影位置、PSF、去重状态、保留源ID）、
`reference_extra_strict_centers.fits`；红端现场测量另有
`reference_blue_psf_measurements.fits`。大源独立窗口使用该源编号前缀的同类审计文件。
日志明确打印投影、BLUE PSF、原有源 PSF 各阶段与最终新增/移动/否决数。
策略版本、候选资产、阈值进入完成签名，旧版本标签不会被当作新版本缓存。

本次仅接入和验证代码，没有启动正式 Zarr 生成。下次使用现有启动脚本即可：

```bash
ABELL_STAGE=zarr USE_UNCOVER=1 ZARR_WORKERS=4 \
bash /home/czh23/CELLECT/preprocessing/run_abell_preprocessing.sh
```

## UNCOVER 实验入口（2026-10-29）

Python 增加 `--use-uncover`（别名 `--use_uncover`）；不指定时仍走原 foundation 星表。
现有 `run_abell_preprocessing.sh` 默认 `USE_UNCOVER=1`，输出到独立的
`/data/czh23/JWST/Abell2744_preprocessing_uncover`；`USE_UNCOVER=0` 恢复旧入口和旧默认输出目录。
这只限制训练 Zarr，图像注册与评测加载没有修改，仍可访问全部 Abell coadd。

配置直接写在该 bash 和 `dataset_paths.json` 的 Abell 部分：

- `UNCOVER_CATALOG`：原始 DR3 SUPER 星表。
- `UNCOVER_DETECTION`、`UNCOVER_SEGMENTATION`：`/data/czh23/JWST/UNCOVER/` 下的两份官方图。
- `UNCOVER_MODEL_ROOT`：默认 `/data/czh23/JWST/UNCOVER/models`；每个波段必须有对应的 `*_fXXXX_bcgs_models.fits[.gz]`，不跨波段借用，缺失时报错。
- `JWST_MAX_INVALID_FRACTION=0.10`：每个普通/大源 512 样本中，`NaN/NO DATA ∪ 无 UNCOVER 覆盖` 超过 10% 就删除。重叠无效部分只计一次；4096 parent 不按整块 10% 一刀删除。
- `SNR_IGNORE_THRESHOLD=5`、`SNR_WEAK_THRESHOLD=7`：普通源 SNR 下限/弱监督上限，传给 `--snr-ignore-threshold` / `--snr-weak-threshold`。
- `UNCOVER_NO_KRON_SNR=3`、`UNCOVER_RISK_NO_KRON_SNR=5`：区外/模型风险区内无 Kron 的附加中心阈值；实际取它与普通源下限的较大值，默认两处均为 5。
- `UNCOVER_RISK_BRIGHT_MAG=22`：使用**当前波段**的 `f_<band>/e_<band>`，AB ZP=28.9；有有效 Kron 的亮源仅跳过 risk/untrusted 剔除，不绕过 bad flag、真正低 SNR 等条件。

覆盖以官方 detection 的有限像素定义，通过 WCS 最近邻投影，不把 seg=0 当作覆盖。
保留的边界样本中，覆盖外像素的 confidence/shape/mask 权重为 0，源中心不进入标签；原图有限像素仍可作为输入上下文，NaN 填补不会恢复覆盖外监督。

**2026-10-30 起恢复 risk-only：模型只定义风险区，不从 FULL 测光参考或 HALF 训练图相减。**
模型按自身 WCS 像素面积从通量换成 MJy/sr，再投影到公共网格；FILTER/PUPIL 必须对应当前波段。
风险区为模型超过 `UNCOVER_RISK_SIGMA × 原始 FULL 空背景 MAD sigma` 后扩张半径 10 像素，并限制在 UNCOVER 覆盖内；sigma 的背景像素排除原有 HALF 亮区和 Gaia 足迹。
全部源仅在原始 FULL 测量一次孔径 SNR、空孔径噪声和 trusted，沿用原始多尺度 SExtractor sky、亮区/Gaia 排除，不再生成模型残差背景或混用残差噪声。
有效 Kron 且本波段 AB<22 的源仍可跳过风险区 untrusted 剔除，但不能绕过可配置的数值 SNR 下限。
全部原始星表条目重测，包括 low-SNR flag 条目。2026-11-03 起默认 SNR<5 或非有限进入 ignore；5≤SNR<7 在几何规则允许时为 weak shape；SNR≥7 通过 SNR 分类，但不提升已因嵌套等规则成为 strict 的源。无 Kron 源达到有效中心阈值后仍仅保留中心，不生成形状。
亮区回退中心、Gaia 判定/PSF 理论 Kron、独立大源补充的原函数继续复用。无 Kron 的 seg 只用于保护源像素不成为背景，不补造 Kron。

```bash
USE_UNCOVER=1 \
JWST_PYTHON=/home/czh23/miniconda3/envs/cellect/bin/python \
UNCOVER_MODEL_ROOT=/data/czh23/JWST/UNCOVER/models \
OUTPUT_ROOT=/data/czh23/analysis/2026-10/2026-10-29/abell_uncover_experiment \
bash /home/czh23/CELLECT/preprocessing/run_abell_preprocessing.sh
```

混合数据入口 `--training-batch` 同样接受这些 CLI 参数，仅对 Abell 生效。
通用 SNR CLI 参数同时用于 HSC、COSMOS 和 Abell；`--hsc-snr-ignore-threshold` / `--hsc-snr-weak-threshold`、`--cosmos-snr-ignore-threshold` / `--cosmos-snr-weak-threshold`、`--abell-snr-ignore-threshold` / `--abell-snr-weak-threshold` 可以逐数据集覆盖通用值。下限必须非负，且严格小于上限，两者必须有限。
各数据集原有测量和类别含义保留：HSC 宽带 `(下限,上限]` 为 weak shape，大面积源的 SNR≤8 弱形状规则不变；COSMOS 的 SNR≤下限为 ignore、`(下限,上限)` 为 strict center only，邻源嵌套资格同步使用下限；Abell 星表及 UNCOVER 使用 `[下限,上限)` 为 weak shape，原 foundation 的孔径复测弱源仍按原逻辑降为 strict center only。
HSC 窄带仍默认 5/8，可通过 `--narrow-snr-ignore-threshold` / `--narrow-snr-weak-threshold` 单独调节。恢复原宽带/JWST 规则可传 `--snr-ignore-threshold 3 --snr-weak-threshold 5`。阈值写入生产完成签名和 Zarr 属性；已有 Zarr 不会自动修改，需重新生成。
输入文件时间戳、模型、覆盖、星表、阈值和策略版本进入完成签名，改变开关或参数不会复用旧标签。
每个 parent 的 `complete.json` / `reference_uncover.json` 记录模型、波段测光字段、风险区比例、`model_subtraction=none` 及原始 FULL 测量方式。
当前策略版本及后处理见上方2026-11-08说明；沿用同一输出根目录即可重建旧 Zarr，已完成的同版本、同参数任务可断点续跑。
先构建临时 parent，成功后替换原 store；若重建结果零样本则删除旧 store，避免保留旧策略标签。旧的 `*_model_snr_background.*` 文件仅为历史缓存，不再读取。
只规划可用 `--abell-plan-only --use-uncover`；完整生成需要所有选中波段的模型下载完成。

## 固定网格

参考 WCS 来自原始完整 F444W FITS；原 `f250m_center` 蓝框的零基起点 `(8064,19102)` 为网格 `(0,0)`。父块尺寸 4096，步长 3968，邻块重叠 128。网格向正负方向延伸，不移动边缘块；超出覆盖部分为 NaN。块名如 `x+00_y+00`。

全部 20 个有效 FILTER/PUPIL 波段分别投影到这一网格。SCI 使用有限支撑归一化的双线性插值；保留 MJy/sr，不乘像素面积。此操作会改变原生采样和噪声相关性。数值背景和 NaN 不被当作零通量混合。参考完整 coadd 与训练半数 coadd 独立重投影；不会用完整图填补训练图。

训练半数父块只要有一个有限像素就写出，并保存对应完整参考块。输出位于数据盘 `parents/<band>/*_half.fits` 和 `*_full.fits`。不会把整张 mosaic 一次装入内存，也不会额外写一套全尺寸重投影 mosaic。

## 512 样本与大源

每个父块内规则小块为 512、步长 368（父块间仍重叠 128）。填充前无效面积超过 10% 就跳过。即使父块没有可用 512 样本，其 4096 FITS 仍保留。无效像素记录于 `band_valid_mask`，图像占位为 0，confidence/shape/segmentation loss 权重为 0，不会被当成背景。

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

日志包含带时间戳和 PID 的 `START/END/FAILED`、波段/parent 标识、耗时、缓存跳过和阶段名。
调度器每 30 秒没有任务完成时输出 `WAIT`（completed/unfinished），子任务仍正常计算时也会输出。
切块阶段的完成单位是一个波段内全部 HALF/FULL 父块；全部波段完成后才进入 Zarr，不能用尚无 Zarr 产物判断切块失败。
Zarr 阶段单独标记模型投影、背景、RAW/残差 SNR、scaling、独立大源和最终打包。
重复的 FITS `datfix/obsfix` 元数据修正警告每种只输出首次，其他警告保留。
这些日志改动不改变数据与筛选参数；已经载入旧代码的进程需到下一次启动才拥有全部新日志。
Bash 的主体作为完整函数先解析，再启动任务，防止运行中编辑文件造成后续命令错位。
启动参数使用 Bash 数组组织，日志会输出完整 `COMMAND`；Python 的 `START` 同时记录实际 `workers`、`use_uncover` 和 `background_root`，以实际 Python 参数为准。
`BACKGROUND_ROOT` 默认使用 1023 同款 `/data/czh23/analysis/2026-10/2026-10-19/background_sextractor15_half` 训练背景缓存。此选项不改变 FULL 测光参考的多尺度 SExtractor sky。
`ABELL_STAGE=all`（默认）顺序执行切块和 Zarr；`ABELL_STAGE=zarr` 可复用已经完成的 `manifest.json`，只续跑 Zarr；`ABELL_STAGE=cutouts` 仅切块。

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
