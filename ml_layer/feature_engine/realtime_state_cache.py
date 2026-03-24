from __future__ import annotations

# flake8: noqa: E501

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RealtimeFeatureStateCache:
    latest_symbol_rows: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def update_symbol_row(self, symbol: str, row: Dict[str, float]) -> None:
        self.latest_symbol_rows[symbol] = dict(row)

    def get_symbol_row(self, symbol: str) -> Dict[str, float]:
        return dict(self.latest_symbol_rows.get(symbol, {}))


# 向后兼容旧命名。
RealtimeFeatureCache = RealtimeFeatureStateCache
