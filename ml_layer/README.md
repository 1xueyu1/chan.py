# ml_layer

L3 + L4 分层机器学习框架。

## 目录

- `config.py`: 统一配置 dataclass
- `label_engine.py`: Triple Barrier + tw-IBS 标签与权重
- `feature_engine/`: L3 特征构建层
	- `chan_structure_context_features.py`: 缠论结构上下文特征
	- `multi_timeframe_resonance_features.py`: 多周期共振特征
	- `price_volume_microstructure_features.py`: 量价微结构特征
	- `market_regime_state_features.py`: 市场状态特征
	- `legacy_feature_center_fusion_features.py`: 旧特征中心逐项融合
	- `expanding_window_normalizer.py`: Expanding 窗口归一化
	- `realtime_state_cache.py`: 实时特征状态缓存
- `models/`: Primary / Meta 模型包装
- `validation/`: PurgedKFold + 特征重要性
- `train_validator.py`: L4 训练验证主协调器
- `inference_engine.py`: 在线推断引擎（框架版）

## 接入点

当前训练入口是 `Debug/xgboost_shap_train.py`，它负责：

1. 从缠论引擎采集事件与bar。
2. 调用 LabelEngine 生成标签。
3. 调用 FeatureEngine 生成并归一化特征。
4. 调用 TrainValidator 完成 PurgedKFold + Primary/Meta 训练。
5. 输出兼容回测所需的模型与元数据文件。
