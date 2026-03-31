# 全流程产物管理与运行完整指南

## 0. 统一启动脚本设计（合并说明）

本项目采用**单一启动脚本**设计：所有流程均通过 `Debug/run_pipeline.py` 启动和管理。

### 0.0 运行环境约定（固定）

- 统一使用已存在的 conda 环境：`chan`。
- 默认前置动作：`conda activate chan`。
- 后续运行命令按该环境直接执行，不再重复提示环境切换。
- 文档阅读约定：除非命令里明确写了其他环境，本文档中的所有命令都默认在 conda 环境 `chan` 下执行。

### 0.0.1 币种读取与内存策略（最新）

- 运行脚本已改为每次从 `data/` 目录动态扫描全量 parquet 币种，不再固定历史 10 币种名单。
- 训练/测试币种按动态全量币种执行 4:1 划分（80% 训练，20% 测试）。
- 回测默认使用动态扫描到的全部币种。
- 并发执行默认带内存预留策略（预留约 30% 可用内存给系统和其他任务），避免任务吃满内存。

- 完整流程：`--mode full`（train -> predict -> backtest）
- 快速迭代：`--mode predict-backtest`（复用模型）
- 分阶段运行：`--stage train|predict|backtest`（独立运行单阶段）

这意味着后续功能迭代都集中在一个脚本中，不需要再创建新的启动脚本文件。

说明：
- 策略批量执行能力已并入 `Debug/run_pipeline.py`。
- 原独立脚本 `Debug/run_labeling_strategy_full_batch.py` 已合并移除，避免策略入口分散。
- 当前仅保留一个策略脚本：`Debug/strategy_xgb.py`（用于策略回测使用示例）。

### 0.1 架构总文档（强制同步）

- 模型训练与回测联动架构总文档：`Debug/MODEL_TRAINING_ARCHITECTURE.md`
- 约束：凡是修改训练架构代码（策略、特征、标签、参数语义、回测门控）时，必须同步更新该文档。
- 目的：保证“当前架构说明”与实际代码始终一致，避免文档过期。

### 0.3 当前标签架构（已重构）

- 原有买卖点双模型训练架构已下线。
- 当前唯一训练策略为：`trainvalidator_hierarchical`。
- 训练架构采用 Primary + Meta 两层模型。
- 评估采用 PurgedKFold + Embargo 防泄露机制。
- 标签仍为 Triple Barrier（PT/SL/TIMEOUT）事件标签。
- 训练核心代码已下沉到 `ml_layer/`（`label_engine.py`、`feature_engine/engine.py`、`train_validator.py`）。

### 0.2 迭代备份约定（模型研究）

- 每次训练迭代都必须保留上一版完整 run 目录，不覆盖历史 run。
- 每次训练输出都必须包含“当次架构说明快照”。
- 当前代码已在训练流程自动生成：`MODEL_TRAINING_ARCHITECTURE_<timestamp>.md`。
- 当前代码已在 `run_pipeline.py` 默认自动备份到 `backup/`（可用 `--no-backup-run` 关闭）。
- 归档时建议保存至少以下内容：
  - `train/`（模型、meta、metrics、filter、label records、架构快照）
  - `logs/`
  - `run_manifest*.json`
- 建议每次迭代只更新“当前推荐 run_id 指针”，历史 run 只读保留。

## 概述

本项目使用统一的全流程管理脚本 `run_pipeline.py`，支持三种运行方式：

1. **全流程模式 (full)**：从数据到模型到回测的完整训练流程
2. **预测+回测模式 (predict-backtest)**：快速迭代，复用已有训练模型
3. **分阶段模式 (stage)**：仅执行 train/predict/backtest 某一个阶段

---

## 1. 什么是"全流程"？

### 标准定义

**"全流程"一般情况下是指：使用 `data/` 文件夹下当前可用的全量币种历史数据进行完整的训练 → 预测 → 回测流程。**

### 数据来源
- **使用目录**：`data/` 下的全量币种历史数据（币种数量随数据更新动态变化）
  - 数据存储格式：Parquet 文件（15分钟 K线合并自 5 分钟原始数据）
  - 数据时间范围：**动态生成**，基于各币种实际覆盖的时间区间
    - 全量覆盖范围：2020-01-01 ~ 2026-02-28（约 6.16 年）
    - 各币种起始时间不同（BTC/ETH 最早，SOL/AVAX 最晚）
- **币种列表**：运行时从 `data/*.parquet` 动态扫描生成，不再写死固定名单。

### 训练与测试分割

#### 币种分割（Symbol Split）：4:1 比例
- **训练币种（80%）**：从 `data/` 动态扫描后的全量币种中按排序取前 80%
- **测试币种（20%）**：从 `data/` 动态扫描后的全量币种中按排序取后 20%
- **目的**：验证模型在未见币种上的泛化能力

#### 时间分割（Time Split）：4:1 比例

时间分割基于 `data/` 文件夹下各币种数据的**实际时间范围**动态计算：

- **总时间范围**：2020-01-01 ~ 2026-02-28（约 6.16 年）
  - **各币种覆盖**：
    - BTC, ETH：2020-01-01 ~ 2026-02-28 (完整 6.16 年)
    - LTC, XRP：2020-01-06/09 ~ 2026-02-28 (约 6.15 年)
    - BNB：2020-02-10 ~ 2026-02-28 (约 6.06 年)
    - ADA：2020-01-31 ~ 2026-02-28 (约 6.08 年)
    - DOGE, DOT, SOL, AVAX：2020-07 ~ 2020-09 ~ 2026-02-28 (约 5.4~5.6 年)

- **动态分割点**：考虑各币种共同时间范围（2020-01-01 起），按 80:20 划分
  - **训练期**：2020-01-01 ~ 2024-12-05 19:00 (80%)
  - **测试期**：2024-12-05 19:00 ~ 2026-02-28 (20%)

- **目的**：验证模型的时间序列泛化能力和跨周期鲁棒性

### 回测配置

#### 回测币种：data 目录动态全量币种
- 使用 `data/` 动态扫描到的全部币种进行回测
- **每个币种单独统计**收益、最大回撤、交易次数等指标
- **聚合统计**：整体投资组合的表现

### 产物组织

每次全流程运行会生成一个唯一的 `run_id` 目录：

```text
Debug/
  runs/
    run_20260322_153000/
      train/
        model_buy.json           # 买点 XGBoost 模型
        model_sell.json          # 卖点 XGBoost 模型
        meta_buy.json            # 买点元数据（特征名、重要性等）
        meta_sell.json           # 卖点元数据
        metrics_buy.json         # 买点训练指标（AUC、Accuracy 等）
        metrics_sell.json        # 卖点训练指标
        shap_report_buy.html     # 买点 SHAP 解释报告
        shap_report_sell.html    # 卖点 SHAP 解释报告
        filter_log_buy.json      # 买点特征筛选日志
        filter_log_sell.json     # 卖点特征筛选日志
        feature_buy.libsvm       # 买点特征训练数据（libsvm 格式）
        feature_sell.libsvm      # 卖点特征训练数据
      predict/
        shap_predict_report.html # 对选定币种的预测与 SHAP 解释
      backtest/
        backtest_metrics.json    # 回测结果：全量币种逐个、聚合统计
        model_signal_events.csv  # 信号事件（时间、币种、方向、概率）
        model_signal_bars.csv    # 信号 K线（时间、币种、OHLCV）
        xgb_backtest_report.html # 回测可视化报告（总体）
        xgb_backtest_report_detail.html  # 详细回测报告（可选）
      logs/
        pipeline.log   # 流程总览日志（阶段开始/结束、关键耗时）
        train.log      # 训练过程日志
        predict.log    # 预测过程日志
        backtest.log   # 回测过程日志
      run_manifest.json          # 本次运行配置、命令、产物清单
    latest_run.json              # 最近一次运行索引（含日志路径）
    current_run.json             # 当前正在运行索引（运行结束会自动移除）
```

日志统一约定：

1. 全流程日志统一写入 `Debug/runs/<run_id>/logs/`，不再额外写到 `result/`。
2. `pipeline.log` 用于快速判断阶段状态；各阶段详细输出分别在 `train.log` / `predict.log` / `backtest.log`。
3. `Debug/runs/current_run.json` 指向当前运行日志；`Debug/runs/latest_run.json` 指向最近一次运行日志。

---

## 2. 快速开始

### 2.0 后台运行约定（强制）

全流程训练+回测耗时较长，统一使用后台运行，避免终端断开导致任务中断。

1. 必须使用 `nohup` 启动，并重定向 stdout/stderr 到日志文件。
2. 启动后记录 `run_id` 与进程 PID，便于追踪与停止。
3. 关闭终端或 SSH 会话后，任务仍会继续执行。
4. 每次启动新的 full 实验前，必须先停止并清理旧实验进程；`run_pipeline.py` 在 full 模式下默认会执行该动作（`--kill-existing-runs-before-start`）。

示例（Linux）：

```bash
cd /root/code/chan.py
RUN_ID="full_dynamic_bg_$(date +%Y%m%d_%H%M%S)"
nohup conda run -n chan python Debug/run_pipeline.py \
  --mode full \
  --run-id "$RUN_ID" \
  --kill-existing-runs-before-start \
  --cache-mode resume \
  --begin-time 2020-01-01 \
  --end-time 2026-02-28 \
  --train-mode gpu \
  --num-workers 4 \
  --feature-symbol-workers 4 \
  --symbol-workers 8 \
  --backtest-parallel-mode process \
  --backtest-data-cache-size 8 \
  > "Debug/runs/${RUN_ID}_launcher.log" 2>&1 &
echo "run_id=$RUN_ID pid=$!"
```

状态检查：

```bash
ps -ef | grep -E "run_pipeline.py|xgboost_shap_train.py|run_vectorbt_backtest.py" | grep -v grep
tail -f Debug/runs/<run_id>/logs/train.log
tail -f Debug/runs/<run_id>/logs/backtest.log
```

补充：

- 默认不建议关闭旧进程清理；仅在排障时使用 `--no-kill-existing-runs-before-start`。
- `Debug/run_full_pipeline_bg.sh` 已默认包含 `--kill-existing-runs-before-start --cache-mode resume`。

### 2.0.4 detached 复用回测（推荐）

当你只想复用已有训练产物做一轮全量 backtest，并确保终端断开后继续运行时，使用以下命令：

```bash
cd /root/code/chan.py
SOURCE_RUN_ID="exp_v1_resume_20260329_0ea492a7"
RUN_ID="rebacktest_detached_$(date +%Y%m%d_%H%M%S)"
nohup conda run -n chan python Debug/run_pipeline.py \
  --mode predict-backtest \
  --stage backtest \
  --source-run-id "$SOURCE_RUN_ID" \
  --run-id "$RUN_ID" \
  --artifact-kind run \
  --source-artifact-kind run \
  --begin-time 2020-01-01 \
  --end-time 2026-02-28 \
  --symbol-workers 8 \
  --backtest-parallel-mode process \
  --backtest-data-cache-size 8 \
  --event-cache \
  --event-cache-namespace "$SOURCE_RUN_ID" \
  --no-empty-signal-fallback \
  --save-trades-csv \
  --save-equity-csv \
  --save-html-detail-report \
  > "Debug/runs/${RUN_ID}_launcher.log" 2>&1 &
echo "run_id=$RUN_ID pid=$! launcher_log=Debug/runs/${RUN_ID}_launcher.log"
```

状态检查：

```bash
ps -fp "$!"
tail -f "Debug/runs/${RUN_ID}_launcher.log"
tail -f "Debug/runs/${RUN_ID}/logs/backtest.log"
```

### 2.0.3 训练缓存续跑与全新重跑（新增）

为避免长任务中断后重复计算，训练阶段新增统一缓存控制参数：

- `--cache-mode resume|fresh`
  - `resume`：续跑模式，优先复用已完成的标签/特征缓存（推荐默认）。
  - `fresh`：全新模式，先清空对应命名空间并禁用缓存，强制全量重算。
- `--cache-namespace <name>`：缓存命名空间，用于区分实验代际。
- `--resume-from-run-id <run_id>`：在 `cache-mode=resume` 且未手动指定 `cache-namespace` 时，自动使用该 run_id 作为命名空间。

推荐用法：

1. 首次实验（建立可续跑缓存）

```bash
conda run -n chan python Debug/run_pipeline.py \
  --mode full \
  --run-id exp_v1_full_20260328 \
  --cache-mode resume \
  --cache-namespace exp_v1 \
  --begin-time 2020-01-01 \
  --end-time 2026-02-28
```

2. 中断后续跑（跳过已完成部分）

```bash
conda run -n chan python Debug/run_pipeline.py \
  --mode full \
  --run-id exp_v1_resume_20260329 \
  --cache-mode resume \
  --cache-namespace exp_v1 \
  --begin-time 2020-01-01 \
  --end-time 2026-02-28
```

3. 模型代际升级后全新重跑（不使用旧缓存）

```bash
conda run -n chan python Debug/run_pipeline.py \
  --mode full \
  --run-id exp_v2_fresh_20260401 \
  --cache-mode fresh \
  --cache-namespace exp_v2 \
  --begin-time 2020-01-01 \
  --end-time 2026-02-28
```

### 2.0 回测框架迭代约定（重要）

当目标是**完善回测框架本身**（例如：指标口径、交易明细语义、报表字段）时，建议优先采用以下流程：

1. 优先复用历史上已经跑通的完整全流程产物（`Debug/runs/<source_run_id>/train`）。
2. 使用 `--stage backtest` 或 `--mode predict-backtest` 直接验证回测行为，避免重复训练开销。
3. 验证时优先选择更长时间区间（例如 2020-01-01 到 2026-02-28），以覆盖更多市场状态与成交场景。

这样可以把迭代焦点放在回测框架逻辑上，并且更快复现、对比与回归测试。

### 2.0.1 并行执行约定（默认）

针对测试和正式训练流程，项目默认采用“能多进程就多进程”的策略：

1. 训练采样阶段默认按 CPU 核心数自动设置 `--num-workers`。
2. 回测阶段默认按 CPU 核心数自动设置 `--symbol-workers`，并默认 `--backtest-parallel-mode process`。
3. 当标的数量为 1 时，symbol 级并行不会带来加速，这属于预期行为。
4. 回测新增有界缓存参数 `--backtest-data-cache-size`（默认 8，0 关闭），避免无限增长占用内存。
5. 回测新增预加载参数 `--backtest-preload-bars`，在 `thread` 模式下可减少重复 I/O；`process` 模式通常建议关闭以控制内存。

补充：
- `ml_layer` 特征层内部也支持按 symbol 并发构建特征（`FeatureConfig.symbol_workers`）。
- 当训练币种较多、每个币种样本量较大时，可显著缩短 L3 特征构建时间。

建议仅在机器资源受限或排障时手动降低并行度。

### 2.0.2 新标签与交易过滤约定

1. 训练默认策略为 `trainvalidator_hierarchical`（Triple Barrier + Primary/Meta）。
2. 标签阶段默认使用结构止损 + RR 止盈 + 时间障碍 + tw-IBS 样本权重。
3. 回测支持两个交易过滤参数：
  - `--signal-margin`：在基础阈值上增加边际，实际触发阈值为 `signal-threshold + signal-margin`。
  - `--cooldown-bars`：执行交易后进入冷却窗口，冷却期内忽略新信号。

这两个过滤器用于降低噪声交易密度，提升成本后的可交易性。

### 2.1 完整全流程（推荐）

#### ⭐ 标准全流程命令（使用全量数据）

**这是最常用的"全流程"配置，用全量历史数据训练：**

一条命令执行：训练 → 预测 → 回测

```powershell
# 标准全流程：使用全量 6 年数据，训练测试 4:1 分割，回测为 data 目录动态全量币种
C:/Users/xueyu/anaconda3/envs/chan/python.exe Debug/run_pipeline.py \
  --mode full \
  --run-id my_baseline_v1 \
  --begin-time 2020-01-01 \
  --end-time 2026-02-28 \
  --train-mode auto \
  --num-workers 4 \
  --symbol-workers 8 \
  --backtest-parallel-mode process \
  --backtest-data-cache-size 8 \
  --labeling-strategy trainvalidator_hierarchical \
  --pt-multiplier 2.0 \
  --timeout-bars 20 \
  --weak-timeout-bars 10 \
  --weak-bsp-types 3 \
  --symbol-workers 2 \
  --signal-threshold 0.55 \
  --save-html-detail-report
```

**这个命令会：**
1. 使用全量 10 个币种的 6 年数据（2020-2026）
2. 按**时间 4:1 分割**：
   - 训练：2020-01-01 ~ 2024-12-05（约 5 年）
   - 测试：2024-12-05 ~ 2026-02-28（约 1.3 年）
3. 按**币种 4:1 分割**：8 个币种用于训练，2 个币种（SOLUSDT/XRPUSDT）用于验证
4. 对 `BTCUSDT` 生成预测报告和 SHAP 解释
5. 在全量 10 个币种上进行回测，**分别统计每个币种的收益**和聚合统计
6. 输出完整的耗时统计（各阶段分别计时）

训练产物中新增可视化报告：
- `train/primary_shap_report.html`：Primary 模型 SHAP 报告
- `train/meta_visual_report.html`：Meta 模型 ROC/PR/系数报告

**预期耗时**：2-4 小时（全量历史数据训练，GPU 加速推荐）

---

### 2.2 快速迭代：预测+回测模式

当已有训练模型时，快速调整阈值或尝试不同的回测配置：

```powershell
C:/Users/xueyu/anaconda3/envs/chan/python.exe Debug/run_pipeline.py \
  --mode predict-backtest \
  --source-run-id my_baseline_v1 \
  --run-id my_baseline_v1_threshold0.6 \
  --signal-threshold 0.6 \
  --symbol-workers 4 \
  --save-html-detail-report
```

**这个命令会：**
1. 复用 `my_baseline_v1` 的训练模型
2. 使用新阈值 (0.6) 预测
3. 在全量 10 个币种上回测
4. 生成新的结果目录

**预期耗时**：15-30 分钟（避免了长时间的训练）

### 2.2.1 阈值自动寻优（推荐）

脚本：`Debug/tune_backtest_threshold.py`

用途：复用已有训练产物，按阈值网格自动跑回测并选出最优阈值。

注意：

- 该脚本会对每个阈值逐个（或并发）触发一轮 `predict-backtest`。
- 日志中出现 `threshold=0.50` 跑完后继续 `threshold=0.55`，是网格迭代的正常行为，不是重复执行同一轮。

当前采用统一综合目标函数 `v1_backtest_composite`：

- 年化收益率上升（硬约束）
- 最大回撤不恶化（硬约束）
- Sharpe 不下降（硬约束）
- 交易数保持在合理区间（硬约束）
- 在满足硬约束后，按加权综合分数排序选择最优阈值

```powershell
C:/Users/xueyu/anaconda3/envs/chan/python.exe Debug/tune_backtest_threshold.py \
  --source-run-id labeling_suite_full_baseline_original_20260322_212132 \
  --begin-time 2020-01-01 \
  --end-time 2026-02-28 \
  --symbols BTCUSDT ETHUSDT SOLUSDT \
  --threshold-start 0.45 \
  --threshold-stop 0.75 \
  --threshold-step 0.02 \
  --max-concurrent-runs 2 \
  --resume-existing \
    --event-cache \
    --event-cache-namespace labeling_suite_full_baseline_original_20260322_212132 \
    --no-empty-signal-fallback \
    --save-trades-csv \
    --save-equity-csv \
  --signal-margin 0.02 \
  --cooldown-bars 4
```

并发与性能建议：

- `--max-concurrent-runs`：并发启动多个阈值子任务（多实例）。
- `--symbol-workers`：单个子任务内部的 symbol 并行度。
- 推荐先从 `max-concurrent-runs=2` 开始，避免与 `symbol-workers` 叠加导致资源争用。
- `--resume-existing`：中断后续跑时跳过已有 `backtest_metrics.json` 的子任务。

输出：`Debug/runs/threshold_tune_<timestamp>_summary.json`

结果中会新增：

- `objective_config`（本次权重与约束）
- `baseline_aggregate`（基线回测指标）
- `passed_count`（满足硬约束的阈值个数）
- 每个阈值下的 `objective` 详情（组件分数、约束通过情况）

### 2.2.2 双模型问题诊断与优化迭代（新增）

每次迭代建议先复用当前基线 run 的训练诊断结果，先定位问题再调参。

步骤：

1. 统一备份当前代码到 `backup/`（`run_pipeline.py` 默认开启 `--backup-run`）。
2. 运行双模型问题分析脚本，生成本轮聚焦问题清单。

```powershell
C:/Users/xueyu/anaconda3/envs/chan/python.exe Debug/analyze_dual_model_issues.py \
  --run-id full_allparts_20260327_1
```

输出：`Debug/runs/<run_id>/train/dual_model_issue_focus.json`

脚本会优先检查：

- Primary：方向偏置、`bsp_type` 拖后腿、置信度分桶收益是否反转、月度稳定性。
- Meta：阈值敏感性、Brier/LogLoss 校准质量、误杀/漏放占比、特征依赖集中度。

随后可直接在 `run_pipeline.py` 使用新参数开展迭代：

```powershell
C:/Users/xueyu/anaconda3/envs/chan/python.exe Debug/run_pipeline.py \
  --mode full \
  --run-id iter_dynamic_pt_calibrated \
  --begin-time 2020-01-01 \
  --end-time 2026-02-28 \
  --dynamic-pt-enabled \
  --pt-low-vol-multiplier 1.8 \
  --pt-mid-vol-multiplier 2.0 \
  --pt-high-vol-multiplier 2.3 \
  --meta-calibration isotonic \
  --pass-positive-month-ratio 0.70
```

若在回测阶段需要分组阈值（按方向/BSP）：

```powershell
C:/Users/xueyu/anaconda3/envs/chan/python.exe Debug/run_pipeline.py \
  --mode predict-backtest \
  --source-run-id iter_dynamic_pt_calibrated \
  --run-id iter_group_thresholds \
  --meta-threshold-by-direction '{"buy":0.55,"sell":0.60}' \
  --meta-threshold-by-bsp '{"1":0.53,"3":0.62,"sell_3":0.65}' \
  --backtest-fee 0.0004 \
  --backtest-slippage 0.0001
```

### 2.2.3 校准 + 阈值分层 + 回测对比一键联动（新增）

脚本：`Debug/run_meta_calibration_threshold_suite.py`

用途：

1. 对 `none/platt/isotonic` 分别训练（或复用指定 run）。
2. 对每个校准方案执行 BSP/方向阈值候选组合搜索（每组再做 signal-threshold 网格）。
3. 自动汇总每个校准方案的最优结果，并输出对比 `JSON + Markdown` 报告。

示例：

```bash
cd /root/code/chan.py
/root/anaconda3/envs/chan/bin/python Debug/run_meta_calibration_threshold_suite.py \
  --begin-time 2020-01-01 \
  --end-time 2026-02-28 \
  --baseline-run-id exp_v1_resume_20260329_0ea492a7 \
  --include-none-calibration \
  --calibrations platt isotonic \
  --threshold-start 0.50 \
  --threshold-stop 0.60 \
  --threshold-step 0.02 \
  --max-concurrent-runs 2 \
  --symbol-workers 8 \
  --backtest-parallel-mode process \
  --event-cache \
  --event-cache-namespace exp_v1_resume_20260329_0ea492a7 \
  --no-empty-signal-fallback \
  --save-trades-csv \
  --save-equity-csv \
  --bsp-threshold-candidate "" \
  --bsp-threshold-candidate "3=0.62,sell_3=0.65" \
  --direction-threshold-candidate "" \
  --direction-threshold-candidate "buy=0.55,sell=0.60"
```

输出：

- `Debug/runs/meta_calibration_suite_<timestamp>/suite_comparison.json`
- `Debug/runs/meta_calibration_suite_<timestamp>/suite_comparison.md`

---

### 2.3 分阶段运行（单脚本管理）

当你只想运行某一个阶段时，直接使用 `--stage` 即可：

```powershell
# 仅训练
C:/Users/xueyu/anaconda3/envs/chan/python.exe Debug/run_pipeline.py \
  --mode full \
  --stage train \
  --run-id stage_train_20260322 \
  --begin-time 2020-01-01 \
  --end-time 2026-02-28

# 仅预测（复用已有训练产物）
C:/Users/xueyu/anaconda3/envs/chan/python.exe Debug/run_pipeline.py \
  --stage predict \
  --source-run-id baseline_2026_03_22 \
  --run-id stage_predict_20260322 \
  --begin-time 2020-01-01 \
  --end-time 2026-02-28

# 仅回测（复用已有训练产物）
C:/Users/xueyu/anaconda3/envs/chan/python.exe Debug/run_pipeline.py \
  --stage backtest \
  --source-run-id baseline_2026_03_22 \
  --run-id stage_backtest_20260322 \
  --begin-time 2020-01-01 \
  --end-time 2026-02-28 \
  --save-html-detail-report
```

说明：
- `--stage train` 会自动只执行训练阶段
- `--stage predict/backtest` 会自动走模型复用路径（需要 `--source-run-id` 或 `--train-dir`）

### 2.4 批量策略运行（单脚本）

通过 `Debug/run_pipeline.py` 直接批量运行多个标签策略：

```powershell
C:/Users/xueyu/anaconda3/envs/chan/python.exe Debug/run_pipeline.py \
  --mode full \
  --batch-labeling-strategies trainvalidator_hierarchical \
  --batch-start-from-strategy trainvalidator_hierarchical \
  --batch-run-prefix labeling_batch \
  --begin-time 2020-01-01 \
  --end-time 2026-02-28
```

输出汇总：`Debug/runs/<batch_run_prefix>_summary_<timestamp>.json`

---

## 3. 参数详解

### 全局参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `full` | 运行模式：`full` 或 `predict-backtest` |
| `--stage` | 无 | 仅运行指定阶段：`train` / `predict` / `backtest` |
| `--run-id` | 自动生成 | 本次结果的目录名（e.g., `run_20260322_120000`） |
| `--debug-root` | `Debug` | 产物根目录 |
| `--begin-time` | `2020-01-01` | 时间范围起点（建议使用全量数据起点） |
| `--end-time` | `2026-02-28` | 时间范围终点（建议使用全量数据终点） |
| `--signal-threshold` | `0.55` | 买卖点判定概率阈值 |

### 训练参数（仅在 `mode=full` 时使用）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train-symbols` | 8 个币种 | 用于训练的币种列表 |
| `--test-symbols` | `SOLUSDT XRPUSDT` | 用于验证的币种列表（不参与训练）|
| `--train-mode` | `auto` | 训练设备：`cpu` / `gpu` / `auto` |
| `--train-mode gpu` | - | 强制GPU训练路径：优先 `tree_method=gpu_hist`，若当前XGBoost版本不支持则回退 `hist + device=cuda` |
| `--num-workers` | 自动（按CPU核数） | 数据采样的并行进程数 |
| `--feature-symbol-workers` | 自动（约为num-workers一半） | L3特征层按symbol并发构建线程数 |
| `--labeling-strategy` | `trainvalidator_hierarchical` | TrainValidator 分层策略 |
| `--labeling-strategy` | `trainvalidator_hierarchical` | TrainValidator 分层训练策略 |
| `--pt-multiplier` | `2.0` | 止盈障碍倍数（RR） |
| `--dynamic-pt-enabled` | False | 启用按波动状态动态调整 PT 倍数 |
| `--pt-low-vol-multiplier` | `1.8` | 低波动状态 PT 倍数 |
| `--pt-mid-vol-multiplier` | `2.0` | 中波动状态 PT 倍数 |
| `--pt-high-vol-multiplier` | `2.2` | 高波动状态 PT 倍数 |
| `--pt-vol-window` | `96` | 波动状态滚动窗口（bar） |
| `--pt-vol-quantile-low` | `0.33` | 低波动分位点 |
| `--pt-vol-quantile-high` | `0.67` | 高波动分位点 |
| `--timeout-bars` | `20` | 默认时间障碍bar数 |
| `--weak-timeout-bars` | `10` | 弱信号时间障碍bar数 |
| `--weak-bsp-types` | `3` | 弱信号主类型（逗号分隔） |
| `--cv-splits` | `5` | PurgedKFold 折数 |
| `--embargo-bars` | `10` | 时间隔离带 bar 数 |
| `--meta-threshold` | `0.55` | Meta执行阈值 |
| `--meta-calibration` | `none` | Meta 概率校准方法：`none/platt/isotonic` |
| `--meta-calibration-ratio` | `0.2` | 校准集占比（时间后段） |
| `--pass-positive-month-ratio` | `0.70` | 准入约束：月度正收益占比下限 |
| `--primary-model` | `xgboost` | Primary模型类型 |
| `--meta-model` | `logistic` | Meta模型类型 |
| `--optuna-trials` | `0` | 超参搜索次数（0关闭） |
| `--mda-max-samples` | `6000` | MDA置换重要性计算最大采样数 |
| `--mda-n-jobs` | 自动（约为num-workers一半） | MDA置换重要性并发线程数 |
| `--shap-sample-limit` | `5000` | Primary SHAP报告最大采样数 |

### 预测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--predict-symbol` | `BTCUSDT` | 生成 SHAP 详细报告的币种 |

### 回测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--backtest-symbols` | 全 10 个币种 | 回测的币种列表 |
| `--symbol-workers` | 自动（按CPU核数） | 回测时币种级并行 worker 数 |
| `--event-cache` | True | 是否启用回测事件缓存（可用 `--no-event-cache` 关闭） |
| `--event-cache-dir` | `data/cache/backtest_events` | 回测事件缓存目录 |
| `--event-cache-namespace` | `default` | 事件缓存命名空间 |
| `--signal-margin` | `0.0` | 信号边际，抬高触发阈值 |
| `--meta-threshold-by-direction` | 空 | 方向阈值 JSON 映射（buy/sell） |
| `--meta-threshold-by-bsp` | 空 | BSP 阈值 JSON 映射（含 buy_1/sell_3 组合键） |
| `--cooldown-bars` | `0` | 执行后冷却 bar 数 |
| `--empty-signal-fallback` | False | 某标的无任何放行信号时，是否启用补单策略 |
| `--empty-signal-target-rate` | `0.05` | 补单目标比例（仅 fallback 开启时生效） |
| `--backtest-fee` | `0.0004` | 回测手续费 |
| `--backtest-slippage` | `0.0001` | 回测滑点 |
| `--save-html-detail-report` | False | 是否生成详细 HTML 报告 |
| `--save-trades-csv` | True | 是否导出 `executed_trades.csv` |
| `--save-equity-csv` | True | 是否导出 `portfolio_equity_curve.csv` |

### 复用参数（仅在 `mode=predict-backtest` 时使用）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source-run-id` | 无 | 源 `run_id`，自动指向 `Debug/runs/<id>/train` |
| `--train-dir` | 无 | 直接指定训练产物目录 |

### 其他参数

| 参数 | 说明 |
|------|------|
| `--skip-predict` | 跳过预测阶段 |
| `--skip-backtest` | 跳过回测阶段 |

---

## 4. 常见使用场景

### 场景 1：基线模型训练与回测

```powershell
# 训练新模型，使用全量数据（2020年全量数据）
C:/Users/xueyu/anaconda3/envs/chan/python.exe Debug/run_pipeline.py \
  --mode full \
  --run-id baseline_2026_03_22 \
  --begin-time 2020-01-01 \
  --end-time 2026-02-28 \
  --symbol-workers 4 \
  --save-html-detail-report
```

**输出**：完整的产物在 `Debug/runs/baseline_2026_03_22/`

---

### 场景 2：调整阈值快速迭代

基于已有的模型，尝试不同的概率阈值：

```powershell
# 使用 threshold=0.6 快速迭代
C:/Users/xueyu/anaconda3/envs/chan/python.exe Debug/run_pipeline.py \
  --mode predict-backtest \
  --source-run-id baseline_2026_03_22 \
  --run-id baseline_2026_03_22_threshold0p6 \
  --signal-threshold 0.6 \
  --symbol-workers 4

# 使用 threshold=0.50 再试一次
C:/Users/xueyu/anaconda3/envs/chan/python.exe Debug/run_pipeline.py \
  --mode predict-backtest \
  --source-run-id baseline_2026_03_22 \
  --run-id baseline_2026_03_22_threshold0p5 \
  --signal-threshold 0.5 \
  --symbol-workers 4
```

**优点**：避免重复 45 分钟的训练，只需 15 分钟快速验证阈值效果

---

### 场景 3：不同时间范围的回测

如果想测试不同的时间窗口，需要完整重训（因为时间影响特征）：

```powershell
# 2025 年全年回测（使用全量训练数据）
C:/Users/xueyu/anaconda3/envs/chan/python.exe Debug/run_pipeline.py \
  --mode full \
  --run-id full_year_2025 \
  --begin-time 2020-01-01 \
  --end-time 2026-02-28 \
  --symbol-workers 4
```

---

### 场景 4：仅回测现有模型（不生成预测报告）

```powershell
C:/Users/xueyu/anaconda3/envs/chan/python.exe Debug/run_pipeline.py \
  --mode predict-backtest \
  --source-run-id baseline_2026_03_22 \
  --run-id baseline_2026_03_22_nopredict \
  --skip-predict \
  --symbol-workers 4
```

---

## 5. 查看结果

### 5.1 查看 Manifest（推荐首先查看）

```powershell
# 查看本次运行的所有配置和命令
cat Debug/runs/baseline_2026_03_22/run_manifest.json
```

**包含信息**：
- 训练币种、测试币种
- 训练/预测/回测耗时
- 所有产物路径
- 执行的命令

### 5.2 查看回测结果

```powershell
# 打开回测报告（浏览器）
start Debug/runs/baseline_2026_03_22/backtest/xgb_backtest_report.html

# 查看逐币种的详细指标（JSON 格式）
cat Debug/runs/baseline_2026_03_22/backtest/backtest_metrics.json
```

### 5.3 查看预测报告

```powershell
# 打开 SHAP 解释报告（浏览器）
start Debug/runs/baseline_2026_03_22/predict/shap_predict_report.html
```

### 5.4 查看训练日志

```powershell
# 查看完整的训练过程日志
cat Debug/runs/baseline_2026_03_22/logs/train.log

# 查看回测日志（如有错误）
cat Debug/runs/baseline_2026_03_22/logs/backtest.log
```

---

## 6. 版本管理与备份

完成一次重要的实验后，建议备份到 `backup/` 目录：

```powershell
C:/Users/xueyu/anaconda3/envs/chan/python.exe backup/archive_version.py \
  --version-name framework_baseline_20260322 \
  --train-dir Debug/runs/baseline_2026_03_22/train \
  --backtest-dir Debug/runs/baseline_2026_03_22/backtest \
  --notes "2026-03-22 基线模型：全量数据训练，10币种回测"
```

**备份目录结构**：

```text
backup/
  framework_baseline_20260322/
    code/                    # 代码快照
    artifacts/
      train/                 # 模型、元数据、指标
      backtest/              # 回测结果
    docs/
      version_notes.md       # 版本说明
    manifest.json            # 版本清单
```

---

## 7. 耗时参考

### 全流程 (full mode)

| 阶段 | 符号数 | 样本数 | 典型耗时 |
|------|--------|--------|---------|
| 训练 | 8 | ~20000 | 30-50 min |
| 预测 | 1 | ~10000 | 3-5 min |
| 回测 | 10 | ~20000 | 10-30 min |
| **总计** | - | - | **45-90 min** |

### 预测+回测 (predict-backtest mode)

| 阶段 | 符号数 | 典型耗时 |
|------|--------|---------|
| 预测 | 1 | 3-5 min |
| 回测 | 10 | 10-30 min |
| **总计** | - | **15-35 min** |

---

## 8. 诊断与故障排查

### 问题：训练阶段报错

**检查日志**：
```powershell
cat Debug/runs/<run_id>/logs/train.log | tail -50
```

**常见原因**：
- 数据文件缺失或损坏
- 特征计算错误
- GPU/CPU 内存不足

---

### 问题：回测阶段卡住或超时

**检查进程**：
```powershell
Get-Process python | Where-Object { $_.CommandLine -like '*vectorbt*' }
```

**解决**：
- 增加 `--symbol-workers` 的数值
- 减少 `--backtest-symbols` 的币种数
- 使用 CPU 而非 GPU

---

### 问题：预测报告为空

**可能原因**：
- 所选币种 (`--predict-symbol`) 在训练币种中不存在
- 模型训练失败

**检查**：
```powershell
ls Debug/runs/<run_id>/predict/
cat Debug/runs/<run_id>/logs/predict.log
```

---

## 9. 最佳实践

1. **命名规范**：使用描述性的 `run_id`，例如 `baseline_20260322` 或 `exp_threshold_search_v1`
2. **保存 Manifest**：每次重要运行后保存 JSON manifest 的文本版本
3. **增量迭代**：优先使用 `predict-backtest` 模式进行快速迭代
4. **定期备份**：使用 `archive_version.py` 定期备份重要版本
5. **版本对比**：比较不同 `run_id` 的 `backtest_metrics.json` 来评估改进
6. **日志记录**：记录每次实验的配置和结果到 Git 或文本文件

---

## 附录：完整示例工作流

### Step 1：训练基线模型

```powershell
# 完整全流程训练
$runId = "baseline_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
C:/Users/xueyu/anaconda3/envs/chan/python.exe Debug/run_pipeline.py \
  --mode full \
  --run-id $runId \
  --symbol-workers 4 \
  --save-html-detail-report
```

### Step 2：查看结果

```powershell
# 查看 Manifest
cat "Debug/runs/$runId/run_manifest.json" | ConvertFrom-Json | Format-Table

# 查看回测指标
cat "Debug/runs/$runId/backtest/backtest_metrics.json" | ConvertFrom-Json | Format-List
```

### Step 3：快速调整阈值

```powershell
# 尝试阈值 0.6
C:/Users/xueyu/anaconda3/envs/chan/python.exe Debug/run_pipeline.py \
  --mode predict-backtest \
  --source-run-id $runId \
  --run-id "${runId}_th0p6" \
  --signal-threshold 0.6 \
  --symbol-workers 4

# 尝试阈值 0.5
C:/Users/xueyu/anaconda3/envs/chan/python.exe Debug/run_pipeline.py \
  --mode predict-backtest \
  --source-run-id $runId \
  --run-id "${runId}_th0p5" \
  --signal-threshold 0.5 \
  --symbol-workers 4
```

### Step 4：备份最佳版本

```powershell
# 假设 th0p6 表现最好
C:/Users/xueyu/anaconda3/envs/chan/python.exe backup/archive_version.py \
  --version-name "framework_baseline_best_$(Get-Date -Format 'yyyyMMdd')" \
  --train-dir "Debug/runs/$runId/train" \
  --backtest-dir "Debug/runs/${runId}_th0p6/backtest" \
  --notes "最优阈值 0.6，全量10币种回测"
```

---

## 附录：回测结果统计说明

### 全流程回测的统计方式

完整全流程中的**回测阶段在全量 10 个币种上进行**，结果统计包括：

#### 1. 逐币种统计（Per-Symbol Statistics）
- **每个币种单独计算**：收益率、最大回撤、夏普比率、交易次数等
- **覆盖的币种**：ADAUSDT, AVAXUSDT, BNBUSDT, BTCUSDT, DOGEUSDT, DOTUSDT, ETHUSDT, LTCUSDT, SOLUSDT, XRPUSDT
- **输出位置**：`Debug/runs/<run_id>/backtest/backtest_metrics.json`（JSON 内的 per-symbol 字段）

#### 2. 聚合统计（Aggregate Statistics）
- **整体投资组合表现**：
  - 总收益率（weighted by symbol）
  - 组合最大回撤
  - 组合盈利因子（Profit Factor）
  - 总交易次数
  - 平均每笔交易收益
- **输出位置**：`Debug/runs/<run_id>/backtest/backtest_metrics.json`（JSON 内的 aggregate 字段）

#### 3. 信号事件（Signal Events）
- **所有生成的交易信号**：时间、币种、买卖方向、置信度概率
- **输出格式**：CSV（`model_signal_events.csv`）
- **用途**：详细分析每个交易决策的原因和效果

#### 4. 可视化报告（HTML Report）
- **总体报告**：`xgb_backtest_report.html`（默认生成）
- **详细报告**：`xgb_backtest_report_detail.html`（使用 `--save-html-detail-report` 生成）
- **包含内容**：收益曲线、回撤曲线、按币种分解的表现、交易列表、排名

### 关键指标解读

| 指标 | 含义 | 理想值 |
|------|------|--------|
| Total Return (%) | 全周期总收益率 | > 10% |
| Max Drawdown (%) | 最大回撤（从高点到低点） | < -20% |
| Profit Factor | 盈利笔数总利润 / 亏损笔数总损失 | > 1.5 |
| Sharpe Ratio | 风险调整后的收益 | > 1.0 |
| Win Rate (%) | 盈利交易 / 总交易 | > 50% |
| Avg Trade Return (%) | 平均单笔交易收益 | > 0.1% |

---

## 重要提示

⚠️ **关于全流程的"标准配置"**：
- 除特殊实验外，**全流程应该始终使用 `--begin-time 2020-01-01 --end-time 2026-02-28` 完整的历史数据**
- 这确保了模型训练基于**足够长的时间跨度**和**完整的币种覆盖**
- 不同的时间窗口会导致特征分布变化，需要重新训练而非复用模型

---

**最后更新**：2026-03-22  
**维护者**：ML Pipeline Team
