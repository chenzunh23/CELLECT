固定梯度探针配置

示例：abell_scenes.json 包含用户指定的8处Abell评测区域。
评测x4y2指512像素网格：[2048,2560)×[1024,1536)，不是训练stride=368的r02/c04。

启动（保留原有ROOT、OUT、TRAIN_FILE、VAL_FILE等配置）：
  GRADIENT_DEBUG_PROBES=/home/czh23/CELLECT/configs/gradient_debug/abell_scenes.json \
  GRADIENT_DEBUG_INTERVAL=500 GRADIENT_DEBUG_BATCH_SIZE=1 \
    bash scripts/train_mixed_zarr.sh

设置GRADIENT_DEBUG_PROBES或传--gradient-debug-probes FILE即启用诊断。
未提供配置时保持原来的随机固定探针/全量遍历接口。
不要将本示例的较大探针集继续设为每10步执行，额外耗时和CPU梯度内存会明显增加。

无模型预检：
  python -m utils.gradient_debug_probes --root "$ROOT" --train-file "$TRAIN_FILE" \
    --config configs/gradient_debug/abell_scenes.json --output /tmp/probes_resolved.json

顶层：version=1；random_samples_per_dataset为未单独设置的数据集随机探针数；
datasets按abell/cosmos/hsc设置；report_cohorts默认true。
datasets.<数据集>.random_samples是显式选中样本之外的随机补充数，0表示不补充。
与旧CLI的SAMPLES=0（全量遍历）不同，配置里的0绝不表示全量。
所有选择都来自本次实际训练records；找不到目标、精确位置不存在等情况直接报错，
不会偷偷替换为随机图或加入验证图。需要比较未加载的数据集时须先提供相应训练池。
配置模式关闭探针数据集的随机增广，并缓存固定输入；原训练数据的增广保持原设置。

probes每项支持：
  id：唯一英文标识；description：任意说明；cohort：场景分组名。
  name：精确的已保存样本名（见历史metadata或解析清单）。指定name后可省略空间条件。
  patch、band、dataset_source：筛选父块、波段、数据版本。
  group：可选，HSC noisy等多组数据时指定store后缀，例如0、1、half。
  grid：[x,y]，grid_size默认512；或直接给x0、y0。二者不能混用。
  size：请求方形范围的边长，默认grid_size或512。
  selection：overlap（默认）或exact。
  max_tiles：overlap时最多选择几个重叠最大的已保存训练样本，默认1。
  min_overlap：交集面积/请求区域面积的最低比例，默认0.25。
  include_large_source：默认false，避免大源特写覆盖普通网格选择；精确name不受此限制。
空间坐标使用训练record的原生x0/y0；Abell示例为父4096图像坐标。HSC/COSMOS
若record带有全局原点偏移，应根据解析清单使用相应x0/y0，或直接指定完整name。

当前训练Zarr只保存部分stride=368网格，8处请求范围不一定有完整的512覆盖。
overlap保留实际存储的原始512图像和标签，不拼接、不移动标签、不填补监督缺口。
因此它是“评测区域附近的固定训练探针”，不等同于精确重新裁剪的评测图片。
exact要求存在原点和尺寸都完全相符的已保存样本，否则报错。

示例当前解析：Abell14张（红端光晕8、小源邻域2、恒星2、平坦区过检2），
COSMOS16张、HSC16张；每处max_tiles=2是上限，小源邻域两处各只有1张达到重叠门限。
重叠排序、固定随机选择与去重均确定；修改ROOT或配置后应重新运行预检。

场景指标示例：
  gradient_debug/proposal_decoder/proposal_loss/abell_halo_red__cosmos/cosine
  gradient_debug/encoder/total/abell_small_neighbors__hsc/angle_deg
  gradient_debug/mask/mask_loss/norm/abell_star
  gradient_debug/encoder/mask_vs_proposal/abell_flat_fp/cosine
原有abell/cosmos/hsc/full指标保留。
cohort梯度直接复用同一探针的反传结果，不额外forward；因此场景报告要求microbatch=1。
一个样本可属于多个cohort，但在所属数据集均值中只计一次；full仍按训练records占比
混合abell/cosmos/hsc，cohort的mixture_weight=0，不重复加入full。
数据集内是每张已选样本等权，不是每个场景等权。
场景越多，精确CPU梯度向量及点积开销越大。cohort和其所属数据集/ full共享样本，
比较时不能视为独立数据集；与COSMOS/HSC的跨域比较不包含同一Abell样本。

输出：OUT/gradient_debug/probes_resolved.json保存配置SHA256、每个请求的像素范围、
实际sample名字/store URI、实际原点/尺寸、重叠率和cohort。metrics.jsonl保存同样的
配置标识、probe_selections、场景样本数和各场景梯度指标，便于识别配置是否变更。
