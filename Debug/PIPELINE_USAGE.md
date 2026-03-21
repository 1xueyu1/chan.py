# 全流程产物管理与运行文档

本文档说明如何把训练、预测、回测的全部产物，按一次运行（run）统一归档到 `Debug/runs/<run_id>/`。

## 1. 目录结构

每次流程会生成一个新的 `run_id` 目录：

```text
Debug/
  runs/
    run_20260322_153000/
      train/
        model_buy.json
        model_sell.json
        meta_buy.json
        meta_sell.json
        metrics_buy.json
        metrics_sell.json
        shap_report_buy.html
        shap_report_sell.html
        filter_log_buy.json
        filter_log_sell.json
        feature_buy.libsvm
        feature_sell.libsvm
      predict/
        shap_predict_report.html
      backtest/
        backtest_metrics.json
        model_signal_events.csv
        model_signal_bars.csv
        xgb_backtest_report.html
        xgb_backtest_report_detail.html (可选)
      logs/
        train.log
        predict.log
        backtest.log
      run_manifest.json
```

`run_manifest.json` 会记录本次运行的命令、目录和关键产物路径。

## 2. 一键全流程（推荐）

脚本：`Debug/run_full_pipeline.py`

### 命令

```powershell
C:/Users/xueyu/anaconda3/envs/chan/python.exe Debug/run_full_pipeline.py \
  --run-id run_demo_01 \
  --symbols BTCUSDT ETHUSDT SOLUSDT \
  --predict-symbol BTCUSDT \
  --backtest-symbols BTCUSDT ETHUSDT \
  --begin-time 2024-01-01 \
  --end-time 2026-01-01 \
  --train-mode auto \
  --num-workers 4 \
  --symbol-workers 2 \
  --signal-threshold 0.55 \
  --save-html-detail-report
```

### 参数说明

- `--run-id`: 本次流程目录名；不传会自动生成时间戳。
- `--symbols`: 训练阶段用的币种列表。
- `--predict-symbol`: 预测报告使用的单币种。
- `--backtest-symbols`: 回测阶段币种列表。
- `--begin-time/--end-time`: 三个阶段共用时间区间。
- `--train-mode`: `cpu/gpu/auto`。
- `--num-workers`: 训练采样并行进程数。
- `--symbol-workers`: 回测按 symbol 并行 worker。
- `--signal-threshold`: 统一阈值。
- `--save-html-detail-report`: 回测额外保存 detail 报告。
- `--skip-predict`: 跳过预测阶段。
- `--skip-backtest`: 跳过回测阶段。

## 3. 分阶段运行（手动）

## 3.1 训练

脚本：`Debug/xgboost_shap_train.py`

```powershell
C:/Users/xueyu/anaconda3/envs/chan/python.exe Debug/xgboost_shap_train.py \
  --symbols BTCUSDT ETHUSDT SOLUSDT \
  --begin-time 2024-01-01 \
  --end-time 2026-01-01 \
  --train-mode auto \
  --num-workers 4 \
  --signal-threshold 0.55 \
  --output-dir Debug/runs/run_demo_01/train
```

关键参数：

- `--symbols`: 训练币种。
- `--begin-time/--end-time`: 训练样本时间范围。
- `--train-mode`: 训练设备。
- `--num-workers`: 并行采样。
- `--signal-threshold`: 标签阈值。
- `--output-dir`: 训练产物目录（建议固定到 `Debug/runs/<run_id>/train`）。

## 3.2 预测

脚本：`Debug/xgboost_shap_predict.py`

```powershell
C:/Users/xueyu/anaconda3/envs/chan/python.exe Debug/xgboost_shap_predict.py \
  --code BTCUSDT \
  --begin-time 2024-01-01 \
  --end-time 2026-01-01 \
  --model-buy-path Debug/runs/run_demo_01/train/model_buy.json \
  --model-sell-path Debug/runs/run_demo_01/train/model_sell.json \
  --meta-buy-path Debug/runs/run_demo_01/train/meta_buy.json \
  --meta-sell-path Debug/runs/run_demo_01/train/meta_sell.json \
  --signal-threshold 0.55 \
  --output-dir Debug/runs/run_demo_01/predict \
  --report-path Debug/runs/run_demo_01/predict/shap_predict_report.html
```

关键参数：

- `--code`: 预测标的。
- `--model-*-path` / `--meta-*-path`: 指向本 run 的训练产物。
- `--output-dir`: 预测输出目录。
- `--report-path`: 预测 HTML 报告路径。

## 3.3 回测

脚本：`Backtest/examples/run_vectorbt_backtest.py`

```powershell
C:/Users/xueyu/anaconda3/envs/chan/python.exe Backtest/examples/run_vectorbt_backtest.py \
  --symbols BTCUSDT ETHUSDT \
  --begin-time 2024-01-01 \
  --end-time 2026-01-01 \
  --model-buy-path Debug/runs/run_demo_01/train/model_buy.json \
  --model-sell-path Debug/runs/run_demo_01/train/model_sell.json \
  --meta-buy-path Debug/runs/run_demo_01/train/meta_buy.json \
  --meta-sell-path Debug/runs/run_demo_01/train/meta_sell.json \
  --signal-threshold 0.55 \
  --symbol-workers 2 \
  --output-dir Debug/runs/run_demo_01/backtest \
  --save-html-detail-report
```

关键参数：

- `--symbols`: 回测标的。
- `--symbol-workers`: symbol 级并行度。
- `--output-dir`: 回测产物目录。
- `--save-html-detail-report`: 额外输出 detail 报告。

## 4. 推荐操作规范

1. 每次实验都指定唯一 `run_id`。
2. 训练、预测、回测都只读写当前 `run_id` 目录。
3. 统一从 `run_manifest.json` 追踪本次命令和产物。
4. 对比实验时只比较不同 `run_id` 目录，避免文件覆盖。
