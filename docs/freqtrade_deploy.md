# Freqtrade 模拟实盘部署说明

## 代码边界

Freqtrade 策略源码位于：

```text
user_data/strategies/
```

本地运行配置、数据库、日志不得提交到 git：

```text
user_data/*.json
user_data/*.sqlite*
user_data/logs/
```

## 当前原则

- Freqtrade 策略应复用当前正式模型路线的可见信号。
- 模拟实盘不得读取回测标签或未来路径。
- 自研回测和 Freqtrade 策略需要保持入场、出场、仓位口径一致。
- Telegram、交易所密钥、代理配置等只保留在本机私有配置中。

## 上线前检查

- [ ] 策略文件已提交。
- [ ] 配置文件未提交。
- [ ] 模型文件路径存在。
- [ ] 数据频率和交易对正确。
- [ ] dry-run 数据库不在 git 中。
- [ ] 日志不在 git 中。
- [ ] 自研回测和 Freqtrade smoke 信号一致。
