# backup 目录规范

本目录用于保存每一版框架对应的代码快照与训练/预测/回测产物。

## 统一备份（当前默认）

`Debug/run_pipeline.py` 已默认启用统一备份：

- 每次运行前自动将当前代码与架构文档快照到 `backup/`。
- 运行完成后自动将本次 `Debug/runs/<run_id>/` 产物归档到同一备份目录。
- 可通过 `--no-backup-run` 关闭。

默认命名：

- `backup/pipeline_backup_<run_id>_<timestamp>/`

## 命名规范

- 目录命名：framework_<框架名>_<可选日期或标签>
- 不再使用 backup_v* 形式

## 目录结构

```text
backup/
  pipeline_backup_<run_id>_<timestamp>/
    code/
      Debug/
      ml_layer/
      Backtest/
    artifacts/
      current_run/
      source_train/         # predict-backtest 模式会附带
    docs/
      backup_notes.md
    manifest.json

  framework_xxx/
    code/
    artifacts/train/
    artifacts/predict/
    artifacts/backtest/
    docs/version_notes.md
    manifest.json
```

## 归档命令

```powershell
C:/Users/xueyu/anaconda3/envs/chan/python.exe backup/archive_version.py \
  --version-name framework_full_v5_off \
  --train-dir Debug/full_v5_off \
  --backtest-dir result/full_symbols_1m_after_speedup \
  --notes "OFF 特征模式全流程基线"
```
