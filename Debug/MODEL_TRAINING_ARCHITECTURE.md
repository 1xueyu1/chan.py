# 模型训练架构总文档（TrainValidator 版）

## 1. 核心定位

TrainValidator 的职责是在严格防止信息泄露前提下完成：
- 模型训练
- 样本外评估（OOS）
- 最终模型产出
- 特征重要性分析

当前训练架构采用两层模型：Primary + Meta。

## 2. 两层模型架构（Primary + Meta）

### 2.1 设计思想

- 第一层 Primary：找信号（Recall优先）
- 第二层 Meta：筛信号（Precision优先）

本质是把“发现机会”与“筛选机会”解耦。

### 2.2 Primary Model

- 类型：XGBoost 多分类
- 输入：全特征（结构、动量、波动、成交量、BSP类型、方向特征）
- 输出：{-1,0,+1} 概率分布
  - 训练映射：0=SL(-1), 1=TIMEOUT(0), 2=PT(+1)
- 优化目标：高召回，尽量覆盖潜在机会

### 2.3 Meta Model

- 类型：Logistic Regression（当前版本）
- 输入：
  - 原始特征
  - Primary 概率输出
  - 环境代理特征（波动/趋势代理）
- 输出：执行概率 p in [0,1]
- 优化目标：提高执行精度，过滤低质量信号

### 2.4 Meta Label 定义

仅对 Primary 非零信号样本构建：
- meta_label=1：Primary 方向预测正确
- meta_label=0：Primary 方向预测错误

## 3. 标签引擎（与训练引擎配套）

当前采用 Triple Barrier：
- PT 先触发 -> +1
- SL 先触发 -> -1
- 超时 -> 0

障碍定义：
- SL：结构止损（笔起点/中枢边界）
- PT：entry ± |entry-sl| * pt_multiplier
- Timeout：默认20 bar，弱信号默认10 bar

## 4. 防泄露验证：Purged K-Fold

### 4.1 Purge

训练集剔除与测试窗口重叠样本，防止未来信息泄露。

### 4.2 Embargo

测试窗口后增加 embargo 窗口（默认10 bars），隔离邻近样本。

### 4.3 价值

- OOS评估更可信
- 防止离线高分实盘崩塌

## 5. 训练流程（TrainValidator.fit）

1. PurgedKFold 循环
- 训练 Primary
- 生成 Meta Label
- 训练 Meta
- 在 OOS 折上预测

2. OOS指标汇总
- Sharpe
- Precision / Recall
- Macro-F1
- Signal Count

3. 全量训练
- 训练最终 Primary
- 训练最终 Meta

4. 特征重要性
- MDI（树增益）
- MDA（置换后性能下降）

5. 输出评估报告
- fold级结果
- 均值/方差
- pass_criteria

## 6. 超参数优化

- 接口参数已预留：optuna_trials
- 当前实现：trials=0（关闭）
- 目标函数约定：后续将采用 Sharpe_mean - 1.5 * Sharpe_std

## 7. 特征筛选规则

最终建议特征：
- 保留 = (MDI前50%) ∩ (MDA > 0)

## 8. 关键文件

- 训练入口：Debug/xgboost_shap_train.py
- L3/L4 框架：ml_layer/
- 标签引擎：ml_layer/label_engine.py
- 特征引擎：ml_layer/feature_engine/engine.py
- 训练验证协调器：ml_layer/train_validator.py
- PurgedKFold：ml_layer/validation/purged_kfold.py
- 统一入口：Debug/run_pipeline.py
- 架构文档：Debug/MODEL_TRAINING_ARCHITECTURE.md

## 8.1 可视化训练产物

- Primary 模型报告：`primary_shap_report.html`
  - 基于 XGBoost + SHAP，展示全局重要性、依赖关系与样本解释。
- Meta 模型报告：`meta_visual_report.html`
  - 展示 ROC、PR、系数重要性、混淆矩阵与阈值统计。

两份报告路径会写入 `train_validator_report.json` 的 `visual_reports` 字段，便于流水线归档与回放。

## 9. 主要参数

- --labeling-strategy trainvalidator_hierarchical
- --pt-multiplier 2.0
- --timeout-bars 20
- --weak-timeout-bars 10
- --weak-bsp-types 3
- --cv-splits 5
- --embargo-bars 10
- --meta-threshold 0.55
- --primary-model xgboost
- --meta-model logistic
- --optuna-trials 0
- --mda-max-samples 6000
- --mda-n-jobs auto
- --shap-sample-limit 5000

## 10. 备份与文档同步规则（强制）

每次训练迭代：
1. 训练目录保存当期架构快照 MODEL_TRAINING_ARCHITECTURE_<timestamp>.md
2. 不覆盖历史 run，使用新 run_id
3. 至少保留 train/logs/manifest
4. 若改训练架构代码，必须同步更新本文件

## 11. 运行时性能优化（当前实现）

- 特征层按 symbol 并发构建：`FeatureConfig.symbol_workers` 控制并发数。
- 事件bar定位改为索引查表：避免每条事件重复布尔过滤 DataFrame。
- 标签生成使用 numpy 数组读取 OHLC：减少字典访问开销。
- MDA 特征重要性支持采样 + 并发：默认最多 6000 样本，且支持并发置换计算，显著降低训练尾耗时。
- Primary SHAP 报告支持采样上限：大样本下限制解释样本规模，降低可视化阶段耗时。
