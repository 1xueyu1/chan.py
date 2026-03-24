from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class WalkForwardResult:
    fold: int
    train_size: int
    test_size: int
    sharpe: float


# 该模块保留接口位，当前 Walk-Forward 在 TrainValidator.fit 内部完成。
# 后续若需要扩展为独立可视化评估器，可在此实现。