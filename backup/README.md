# backup 目录规范

本目录用于保存每一版框架对应的代码快照与训练/预测/回测产物。

## 命名规范

- 目录命名：framework_<框架名>_<可选日期或标签>
- 不再使用 backup_v* 形式

## 目录结构

```text
backup/
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
