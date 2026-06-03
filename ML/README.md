# 机器学习路线

当前 ML 目录的主线是 `btc_futures_v1`：使用 11 个高流动性 U 本位合约训练二类买卖点结构模型。

## 推荐运行方式

```powershell
Scripts\btc_futures_run_all.ps1
```

分步运行：

```powershell
Scripts\btc_futures_build_dataset.ps1
Scripts\btc_futures_train.ps1
Scripts\btc_futures_backtest.ps1
```

## 路线文档

- `ML\routes\btc_futures_v1\README.md`
- `ML\MODEL_REGISTRY.md`

## 旧路线说明

旧模型产物和旧回测结果已经从当前 `result` 目录清理。当前只保留 `label_bsp2_valid` 结构标签路线，以及“不定仓 / 结构风险定仓”两个对比回测结果。后续如果要继续扩展，建议在 `btc_futures_v1` 基础上新增新路线，不要再混用旧结果。
