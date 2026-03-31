# 模型训练架构总文档（TrainValidator 版）

## 1. 文档目标与适用范围

本文档是当前训练框架的实现级说明，目标是回答以下问题：
- 每个模块做什么。
- 关键代码如何实现。
- 模块在整体框架中的作用。
- 整个训练-推理-回测链路如何协同工作。

适用范围：
- 训练入口：`Debug/xgboost_shap_train.py`
- 统一流水线入口：`Debug/run_pipeline.py`
- 核心框架：`ml_layer/`
- 回测门控兼容层：`Backtest/model_gate.py`

运行环境固定约定：
- 统一使用 conda 环境：`chan`。
- 默认已执行 `conda activate chan` 后再运行训练/回测命令。
- 该约定为长期默认项，流程内不再重复提示环境切换。
- 全流程默认后台运行（`nohup ... &`），保证终端关闭后任务不被中断。
- 新的 full 实验启动前，默认先清理同项目下旧 run 相关进程（`--kill-existing-runs-before-start`）。

## 2. 核心定位（为什么需要这个框架）

TrainValidator 架构的核心目标是在严格防泄露前提下，完成一套可回放、可解释、可并行扩展的交易信号学习系统：
- 训练：构建 Primary + Meta 两层模型。
- 评估：通过 PurgedKFold 生成更接近真实交易场景的 OOS 指标。
- 产出：落盘模型、特征元数据、可视化报告、运行清单。
- 闭环：模型产物可以直接进入回测框架执行。

简化理解：
- Primary 负责“发现机会”。
- Meta 负责“过滤机会”。
- 回测门控负责“把离线模型安全接入执行引擎”。

## 3. 版本状态与强制约束

### 3.1 版本状态

- 历史 TIMEOUT 三分类语义已淘汰。
- 当前唯一有效语义：二分类 PT/SL。
  - `label_raw=+1` 表示 PT 先触发。
  - `label_raw=-1` 表示 SL 先触发。
  - 训练标签映射：`label=1(PT)`，`label=0(SL)`。

### 3.2 强制约束

- 新增代码和脚本必须默认面向二分类语义。
- 所有训练、推理、回测入口不得引入旧 TIMEOUT 训练逻辑。
- 参数中保留的 timeout 相关项仅用于兼容 CLI，不参与当前标签判定。
- 每次迭代必须在 `backup/` 保留统一备份快照，且与当期架构说明文档绑定。

## 4. 性能优先原则（实现约束）

框架设计默认追求吞吐和可扩展：
- 样本采集层：按 symbol 多进程采集。
- 特征层：按 symbol 并发构建特征。
- 标签层：基于 numpy 数组扫描 OHLC，减少字典与 DataFrame 热点。
- 缓存层：按 symbol + 配置哈希落盘 Parquet（label/feature 双缓存）。
- 重要性评估：MDA 支持采样 + 并发。
- 报告层：SHAP 支持样本上限，避免解释阶段拖慢主流程。

工程原则：
- 正确性优先，正确性达标后优先向量化与并行化。
- 任何新增功能必须评估对主链路耗时的影响。

## 5. 全局架构总览

端到端流程如下：

1. `run_pipeline.py` 根据 mode 组装 train/predict/backtest 命令。
2. 训练阶段调用 `xgboost_shap_train.py`。
3. 训练脚本采集事件与K线，进入 `LabelEngine.transform` 打标签。
4. 标签阶段优先读取 `LabelCache`；未命中时再计算并回写缓存。
5. 特征阶段优先读取 `CachedFeatureEngine` 的 per-symbol 特征缓存；未命中时并发计算并回写缓存。
6. 合并特征后执行归一化，进入 `TrainValidator.fit` 训练与验证。
7. 生成模型与报告产物并落盘（模型、特征映射、可视化、manifest）。
8. 回测阶段由 `Backtest/model_gate.py` 加载模型并执行信号门控。

数据对象主线：
- 原始对象：`events`、`bars`
- 标签对象：`labeled samples`
- 训练对象：`X, y, w, t0_pos, t1_pos, returns`
- 产物对象：`model_*.json / meta_*.json / meta_model.pkl / report json/csv/html`

## 6. 两层模型设计（Primary + Meta）

### 6.1 设计思想

- Primary：Recall 优先，尽量不漏掉潜在机会。
- Meta：Precision 优先，减少低质量执行。

该设计把“方向识别”和“执行筛选”解耦，降低单模型同时优化多目标的难度。

### 6.2 Primary Model

- 类型：XGBoost 二分类。
- 训练实现：`ml_layer/models/primary_model.py`
  - `fit`：构建 DMatrix，支持 `train_mode=cpu/gpu/auto`。
  - `predict_proba`：统一概率形状，二分类时输出 `[p_sl, p_pt]`。
- 作用：输出方向概率，不直接决定是否执行。

### 6.3 Meta Model

- 类型：Logistic Regression（`class_weight='balanced'`）。
- 实现：`ml_layer/models/meta_model.py`
  - `fit`：样本不足或单类时自动禁用（`enabled=False`）。
  - `predict_proba`：禁用时返回全 1，等效“不过滤”。
- 作用：输出执行概率，用于阈值过滤。

### 6.4 Meta Label 定义

在每折训练中，先拿 Primary 在训练集预测得到 `c_train`，再构造：
- `meta_label=1`：`c_train == y_train`（方向正确）
- `meta_label=0`：`c_train != y_train`（方向错误）

这使 Meta 学习“当前样本是否值得执行”。

## 7. 标签引擎（LabelEngine）

实现文件：`ml_layer/label_engine.py`

### 7.1 输入与输出

- 输入：单 symbol 的 `events` 与对应 `bars`。
- 输出：带完整标签与交易路径信息的样本列表，包括：
  - `label_raw` / `label` / `label_text`
  - `entry_price` / `sl_price` / `pt_price`
  - `hit_event` / `holding_bars` / `realized_return`
  - `sample_weight`（TW-IBS 权重）

### 7.2 关键实现机制

1. 结构止损计算 `_compute_structural_sl`
- 买入信号：从 `bi_start`、`zs_low` 中选择有效且低于 entry 的候选，取更保守位置。
- 卖出信号：从 `bi_start`、`zs_high` 中选择有效且高于 entry 的候选。
- 若无有效结构，使用 entry 的固定比例兜底。

2. 触发判定
- 风险定义：`risk = |entry - sl|`
- 止盈定义：`pt = entry ± risk * pt_multiplier`
- 从 `t0_pos+1` 顺序扫描 OHLC：先命中 PT 则 +1，先命中 SL 则 -1。
- 同 bar 同时命中 PT/SL 时按保守原则记作 SL（`sl_tie`）。

3. 数据尾部兜底（取消 timeout 后）
- 到数据末尾仍未触发 PT/SL：
  - 若收益绝对值过小（`min_ret_threshold`），归为 SL（`ret_floor_sl`）。
  - 否则按收益符号归类为 PT 或 SL（`ret_sign_pt/sl`）。

4. 样本权重 `_compute_tw_ibs_weights`
- 使用时间重叠度近似拥挤度，权重与重叠度反比。
- 最后按均值归一化，保持整体权重尺度稳定。

### 7.3 在框架中的作用

- 是训练标签唯一来源。
- 决定模型学习目标是否与交易执行逻辑一致。
- 其输出字段被后续特征层、验证层、报告层反复复用。

## 8. 特征引擎（FeatureEngine）

实现文件：`ml_layer/feature_engine/engine.py`、`ml_layer/feature_engine/cached_engine.py`

### 8.1 输入与输出

- 输入：`samples`（已标注）与 `bars_by_symbol`。
- 输出：按时间索引排序的 `feature_df`（训练矩阵来源）。

### 8.2 关键实现机制

1. 分 symbol 特征构建
- `_build_symbol_bar_frame` 先构建每个 symbol 的指标表。
- `_transform_symbol_samples` 按 symbol 顺序生成样本特征。
- 支持 `FeatureConfig.symbol_workers` 并发。

2. 缓存增强（`CachedFeatureEngine`）
- 缓存键：`symbol + feature_config_hash`。
- 缓存介质：`data/cache/features/*.parquet`。
- 逻辑：先按 symbol 查缓存，未命中 symbol 再并发计算，完成后立即写回缓存。
- 兼容：关闭缓存时自动回退到原始 `FeatureEngine.transform`。

3. 多组特征融合
- 结构上下文特征。
- 多级别共振特征。
- 价量微观结构特征。
- 市场状态特征（波动/趋势/近期胜率等）。
- 历史遗留特征融合层，保证兼容过往特征语义。

4. 实时一致性
- 离线训练用 `fit_transform`。
- 在线推理用 `transform_realtime`，保证归一化口径一致。

### 8.3 在框架中的作用

- 把事件语义映射到模型可学习向量空间。
- 统一离线与实时特征口径，降低训练-推理漂移风险。

## 9. 防泄露验证（PurgedKFold）

实现文件：`ml_layer/validation/purged_kfold.py`

### 9.1 Purge 机制

对每个测试折，计算测试窗口 `[test_t0_min, test_t1_max]`，剔除训练集中与其时间区间重叠的样本：
- 条件：`(t1_pos >= test_t0_min) & (t0_pos <= test_t1_max)`

### 9.2 Embargo 机制

测试窗口结束后，额外剔除 `embargo_bars` 内的训练样本，减少邻接信息污染。

### 9.3 在框架中的作用

- 是 OOS 指标可信度的核心保障。
- 防止时序任务中“看起来高分、实则泄露”的离线幻觉。

## 10. 训练协调器（TrainValidator）

实现文件：`ml_layer/train_validator.py`

### 10.1 `fit` 总流程

1. 构建 PurgedKFold。
2. 对每折调用 `_fit_one_fold`：
   - 训练 Primary。
   - 构建并训练 Meta。
   - 在测试折生成 `c_test`、`exec_prob`、`keep_mask`。
3. 汇总 OOS 预测缓存：`oos_cls/oos_exec/oos_keep`。
4. 用全量数据重训最终 Primary 与 Meta。
5. 计算 MDI/MDA 并做特征筛选。
6. 生成 report、fold_df 与 OOS 指标。

### 10.2 `_fit_one_fold` 关键细节

- `p_train = primary.predict_proba(X_train)`
- `meta_y = (argmax(p_train) == y_train)`
- `meta_X = [X_train, p_train]` 横向拼接。
- OOS 阶段：`keep_mask = p_exec >= meta_threshold`。

### 10.3 指标语义

- `precision`：被保留信号中方向正确的比例。
- `recall`：全部样本中被正确保留的比例。
- `macro_f1`：方向分类质量。
- `sharpe`：被保留样本收益序列的年化 Sharpe。

### 10.4 特征筛选

规则：
- `mdi_keep = MDI 前 50%`
- `mda_keep = MDA > 0`
- `selected = mdi_keep ∩ mda_keep`

该规则兼顾树模型内部重要性与置换鲁棒性。

### 10.5 在框架中的作用

- 是训练阶段唯一主协调器。
- 聚合了训练、验证、特征解释、准入判定（pass_criteria）。

### 10.6 双模型诊断产物（新增）

训练阶段会额外输出双模型诊断：

- `dual_model_diagnostics.json`
- `primary_confidence_bucket_returns.csv`
- `meta_threshold_sensitivity.csv`

诊断项覆盖：

- Primary：类别分布与方向偏置、`bsp_type` 分组正确率、置信度分桶收益、月度/季度稳定性。
- Meta：`0.45~0.70` 阈值敏感性（收益/回撤/交易数）、Brier/LogLoss 校准、误杀漏放占比、特征依赖集中度。

该报告用于优先发现双模型失衡问题，再进入参数/特征迭代。

## 11. 缓存层（新增）

实现文件：`ml_layer/cache_layer.py`

### 11.1 设计目标

- 用空间换时间，避免重复训练中对确定性中间结果反复重算。
- 提供统一缓存接口，降低训练脚本侵入。
- 支持按配置自动失效（哈希变更即命中新文件）。

### 11.2 组件

- `LabelCache`
  - 目录：`data/cache/labels/<namespace>/`（`default` 命名空间兼容历史目录）
  - 键：`{symbol}_{label_config_hash}.parquet`
  - 内容：`LabelEngine.transform` 输出样本记录。
- `FeatureCache`
  - 目录：`data/cache/features/<namespace>/`（`default` 命名空间兼容历史目录）
  - 键：`{symbol}_{feature_config_hash}.parquet`
  - 内容：按 symbol 存储的特征 DataFrame。

命名空间用途：

- 同一实验续跑：固定同一个 namespace。
- 模型代际切换：切换到新 namespace，避免误复用旧缓存。

### 11.3 关键函数

- `_config_hash(config)`: 基于配置内容生成短哈希。
- `get_label_cache(namespace=...)` / `get_feature_cache(namespace=...)`: 命名空间隔离缓存实例。
- `cache_stats(namespace=...)`: 返回命名空间级 hit/miss、大小、文件数统计。
- `clear_cache_namespace(namespace)`: 清空指定命名空间缓存。
- `clear_all_caches()`: 清空 labels/features 缓存目录。

### 11.4 训练入口接入点

训练脚本 `Debug/xgboost_shap_train.py` 已接入：

- 标签缓存开关：`--disable-label-cache`
- 特征缓存开关：`--disable-feature-cache`
- 一键清空缓存：`--clear-cache`
- 缓存模式：`--cache-mode resume|fresh`
- 缓存命名空间：`--cache-namespace <name>`

并在训练结束输出缓存命中统计。

模式语义：

- `resume`: 复用缓存（命中后跳过重复计算），适用于长任务中断后续跑。
- `fresh`: 先清空命名空间并禁用缓存，强制全量重算。

## 12. 统一流水线参数透传（新增）

实现文件：`Debug/run_pipeline.py`

新增参数并透传到训练阶段：

- `--disable-label-cache`
- `--disable-feature-cache`
- `--clear-cache`
- `--cache-mode`
- `--cache-namespace`
- `--resume-from-run-id`
- `--kill-existing-runs-before-start`

用途：
- 正常训练默认启用缓存。
- A/B 评估可通过 disable 参数关闭缓存对比耗时。
- 维护时可通过 clear 参数先清缓存再运行。
- 新 full 实验默认先清理旧进程，避免多轮并行导致资源争抢与日志混淆。

## 13. 缓存运维工具（新增）

实现文件：`Script/cache_manager.py`

支持命令：

- `stats`: 查看标签/特征缓存统计
- `clear`: 清空全部缓存
- `clear-labels`: 仅清空标签缓存
- `clear-features`: 仅清空特征缓存
- `list-labels`: 列出标签缓存文件
- `list-features`: 列出特征缓存文件

## 10.7 回测综合优化目标（新增）

阈值调优统一采用 `v1_backtest_composite`：

- 硬约束：年化收益率上升、最大回撤不恶化、Sharpe 不下降、交易数在合理区间。
- 软目标：在满足硬约束后，按加权综合分数排序。

对应脚本：`Debug/tune_backtest_threshold.py`。

## 10.8 迭代看板（新增）

`run_pipeline.py` 每次运行输出 `iteration_dashboard.json`，固定展示：

- 收益、回撤、Sharpe、胜率、交易数
- 阈值参数（`signal_threshold`、`signal_margin`、`cooldown_bars`）
- 各阶段耗时与时间线

该看板用于跨 run 的同口径对比，避免主观判断偏差。

## 11. 特征重要性（MDI + MDA）

实现文件：`ml_layer/validation/feature_importance.py`

### 11.1 MDI

- 来源：XGBoost booster 的 gain。
- 优点：计算快，适合全量粗筛。

### 11.2 MDA

- 思路：单特征打乱后比较 macro-F1 下降量。
- 大样本下先采样（`mda_max_samples`），再并发计算（`mda_n_jobs`）。
- 优点：更接近“去掉该特征后的真实影响”。

### 11.3 在框架中的作用

- 为后续降维、提速、稳定性优化提供证据。

## 12. 推理引擎（InferenceEngine）

实现文件：`ml_layer/inference_engine.py`

### 12.1 执行逻辑

1. 进入 `infer` 后，先做实时特征归一化。
2. Primary 输出二分类概率 `p_primary`。
3. `argmax` 映射方向：`0 -> -1`，`1 -> +1`。
4. 构建 Meta 输入：`[X, p_primary]`。
5. `p_meta >= meta_threshold` 则放行，否则方向置 0。

### 12.2 输出

`SignalOutput` 包含：
- `direction`：最终执行方向（-1/0/+1）
- `primary_proba`：`short/long`
- `meta_proba`：执行概率
- `features_snapshot`：归一化后快照（便于追溯）

### 12.3 在框架中的作用

- 作为训练后在线推理模板，定义标准推理协议。

## 13. 回测接入层（DualModelGate）

实现文件：`Backtest/model_gate.py`

### 13.1 为什么需要兼容层

回测历史实现以三分类信号接口为基础，而当前训练是二分类。
`DualModelGate` 负责把新模型安全映射到旧执行接口，避免大规模回测框架重写。

### 13.2 关键兼容机制

1. Primary 输出兼容
- 若模型输出为 1D（二分类单概率），转为 `[p_sl, 0, p_pt]`。
- 若输出为 2D 两列（`[p_sl, p_pt]`），也转为 `[p_sl, 0, p_pt]`。

2. 方向映射
- 按三分类索引映射：`0 -> -1`, `2 -> +1`, `1 -> 0`（占位 timeout）。

3. Meta 输入维度自适应
- 根据 `meta_model.model.n_features_in_` 推断 Meta 期望的概率块列数。
- 自动从 `[p_sl, p_timeout, p_pt]` 退化到 `[p_sl, p_pt]` 或单列。
- 解决训练版本切换导致的维度不一致问题。

4. 空信号兜底
- 若阈值过滤后无任何信号，触发分位数自适应保底，避免回测全空。

5. 防未来函数执行约束
- 事件特征对齐时，`t0_pos` 采用 `floor(exec_time)`（仅映射到当前或历史K线），禁止“最近邻”落到未来K线。
- `next_bar_open` 执行模式下，成交K线时间必须严格满足 `fill_ts > signal_ts`，否则立即抛错。

### 13.3 在框架中的作用

- 保证“训练语义升级”与“执行接口稳定”并存。
- 是当前从模型到交易事件的最后一层风险缓冲。

## 14. 训练入口脚本（xgboost_shap_train）

实现文件：`Debug/xgboost_shap_train.py`

### 14.1 脚本职责

- 组织采集、标注、特征、训练、可视化与落盘。
- 提供并行参数和训练模式参数。
- 产出供回测直接消费的兼容文件名。

### 14.2 实现阶段

1. 数据采集（多进程）
- `collect_symbol_events_worker` 并行采集每个 symbol 的事件与 bars。

2. 标签阶段
- 每个 symbol 进入 `LabelEngine.transform`。

3. 特征阶段
- 全部样本进入 `FeatureEngine.transform`。

4. 训练验证阶段
- `TrainValidator.fit` 完成 CV + 全量模型重训。

5. 可视化阶段
- Primary：SHAP 报告。
- Meta：ROC/PR/阈值敏感性/系数解释报告。

6. 产物落盘
- 主模型：`primary_model.json`
- 兼容模型：`model_buy.json`、`model_sell.json`
- 特征映射：`primary_feature_meta.json`、`meta_buy.json`、`meta_sell.json`
- Meta 对象：`meta_model.pkl`
- 报告：`train_validator_report.json`、`feature_importance_*.csv`、`selected_features.json`

### 14.3 在框架中的作用

- 是单次训练的完整执行器。
- 产物结构直接决定后续回测与分析工具可用性。

## 15. 全流程编排（run_pipeline）

实现文件：`Debug/run_pipeline.py`

### 15.1 模式

- `full`：train -> predict -> backtest
- `predict-backtest`：复用已有 train 产物，快速迭代

### 15.2 编排机制

- `_run_full_pipeline` 负责创建 run 目录与命令串。
- `_run_cmd` 负责执行、日志写入、超时保护、卡死检测。
- 运行后生成：
  - `run_manifest.json`
  - `timing_summary.json`
  - `timing_history.csv`

### 15.3 目录结构（标准）

- `Debug/runs/<run_id>/train`
- `Debug/runs/<run_id>/predict`
- `Debug/runs/<run_id>/backtest`
- `Debug/runs/<run_id>/logs`

### 15.4 在框架中的作用

- 统一实验管理与产物归档。
- 是回归验证、性能对比、线上前验收的标准入口。

## 16. 配置系统与参数语义

实现文件：`ml_layer/config.py`

### 16.1 LabelConfig

- `pt_multiplier`：止盈倍数。
- `min_ret_threshold`：尾部收益符号兜底阈值。
- `timeout_*`：兼容保留，不参与当前标签主逻辑。

### 16.2 FeatureConfig

- `symbol_workers`：特征并发核心开关。
- `normalize_method` / `vol_regime_window`：特征归一化与状态统计参数。

### 16.3 ModelConfig

- Primary 默认 `objective=binary:logistic`，`eval_metric=logloss`。
- `meta_threshold` 控制执行过滤强度。

### 16.4 TrainValidatorConfig

- `n_splits`、`embargo_bars`：防泄露强度。
- `mda_max_samples`、`mda_n_jobs`：尾部耗时与稳定性平衡参数。

## 17. 输出产物与消费关系

### 17.1 训练阶段关键产物

- `model_buy.json` / `model_sell.json`：回测 Primary 读取入口（兼容命名）。
- `meta_buy.json` / `meta_sell.json`：特征名到索引映射。
- `meta_model.pkl`：Meta 执行过滤模型。
- `train_validator_report.json`：OOS 与 fold 级评估总报告。
- `feature_importance_mdi.csv` / `feature_importance_mda.csv`：解释与筛选依据。

### 17.2 回测阶段消费

`DualModelGate` 同时读取：
- Primary 模型文件。
- 特征元数据映射。
- 可选 Meta pkl。

并输出 `ScoredSignalEvent` 供执行层回测。

## 18. 运行时性能优化（当前已实现）

- 采集层：`ProcessPoolExecutor` 并发 symbol 采集。
- 特征层：`ThreadPoolExecutor` 并发 symbol 特征计算。
- 标签层：numpy 数组高频访问，避免逐行 DataFrame 过滤。
- MDA：采样 + 多线程并发置换。
- SHAP：解释样本上限，降低可视化尾部时延。
- 编排层：命令超时、卡死检测、日志实时落盘。

## 19. 风险控制与常见故障点

### 19.1 元模型维度不匹配

现象：Meta 推理报错 `X has n features, expected m`。

处理：
- `DualModelGate._meta_prob_batch` 根据 `n_features_in_` 自适应概率块列数。

### 19.2 全空信号

现象：回测无交易。

处理：
- 门控层提供 primary_conf 分位数兜底，保证最小信号密度。

### 19.3 样本不足

现象：PurgedKFold 无有效折。

处理：
- 扩大时间区间或增加 symbol。
- 降低折数与 embargo 强度。

## 20. 备份与文档同步规则（强制）

每次训练迭代必须遵守：

1. 在训练目录保存当期架构快照：`MODEL_TRAINING_ARCHITECTURE_<timestamp>.md`。
2. 使用新 `run_id`，不覆盖历史 run。
3. 至少保留 `train / logs / run_manifest`。
4. 若改动训练架构代码，必须同步更新本文件。

## 21. 关键参数清单（当前推荐）

- `--labeling-strategy trainvalidator_hierarchical`
- `--pt-multiplier 2.0`
- `--timeout-bars 20`（兼容保留，不参与二分类）
- `--weak-timeout-bars 10`（兼容保留，不参与二分类）
- `--weak-bsp-types 3`
- `--cv-splits 5`
- `--embargo-bars 10`
- `--meta-threshold 0.55`
- `--primary-model xgboost`
- `--meta-model logistic`
- `--optuna-trials 0`
- `--mda-max-samples 6000`
- `--mda-n-jobs auto`
- `--shap-sample-limit 5000`
- `--num-workers`（采集并行）
- `--feature-symbol-workers`（特征并行）
- `--symbol-workers`（回测并行）

## 22. 整体框架如何工作（端到端总结）

从工程角度，当前框架是一个“训练-验证-执行”闭环系统：

1. 通过事件与K线构建结构化样本。
2. 用 PT/SL 触发规则定义监督目标。
3. 用 PurgedKFold 提供可信 OOS 评估。
4. 用 Primary 学方向、Meta 学可执行性。
5. 通过兼容层把二分类模型稳定接入回测执行。
6. 通过 manifest、日志、可视化报告完成可追溯归档。

最终收益：
- 语义统一（全链路二分类）。
- 评估可信（防泄露）。
- 执行可用（回测兼容层）。
- 运维可控（标准化 run 产物与时间统计）。
