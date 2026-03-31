from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np


@dataclass
class BacktestObjectiveConfig:
    version: str = "v1_backtest_composite"
    ret_weight: float = 0.45
    mdd_weight: float = 0.25
    sharpe_weight: float = 0.20
    trade_weight: float = 0.10
    mdd_worsen_tolerance_pct: float = 1.0
    sharpe_drop_tolerance: float = 0.05
    trade_min_ratio: float = 0.60
    trade_max_ratio: float = 1.40
    trade_abs_min: float = 30.0
    trade_abs_max: float = 5000.0


@dataclass
class BacktestObjectiveScore:
    score: float
    passed_constraints: bool
    hard_constraints: Dict[str, bool]
    components: Dict[str, float]
    baseline: Dict[str, float]
    candidate: Dict[str, float]
    config: Dict[str, float | str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "score": float(self.score),
            "passed_constraints": bool(self.passed_constraints),
            "hard_constraints": dict(self.hard_constraints),
            "components": dict(self.components),
            "baseline": dict(self.baseline),
            "candidate": dict(self.candidate),
            "config": dict(self.config),
        }


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        x = float(value)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return float(default)


def _as_abs_drawdown(metrics: Dict[str, object]) -> float:
    return abs(_safe_float(metrics.get("max_drawdown_pct"), 0.0))


def _trade_bounds(
    base_trades: float,
    cfg: BacktestObjectiveConfig,
) -> tuple[float, float, float]:
    target = max(_safe_float(base_trades, 0.0), cfg.trade_abs_min)
    min_trades = max(cfg.trade_abs_min, target * cfg.trade_min_ratio)
    max_trades = min(
        cfg.trade_abs_max,
        max(target, cfg.trade_abs_min) * cfg.trade_max_ratio,
    )
    if min_trades > max_trades:
        max_trades = min_trades
    return target, min_trades, max_trades


def score_backtest_metrics(
    baseline_metrics: Dict[str, object],
    candidate_metrics: Dict[str, object],
    config: BacktestObjectiveConfig | None = None,
) -> BacktestObjectiveScore:
    cfg = config or BacktestObjectiveConfig()

    base_ret = _safe_float(baseline_metrics.get("annualized_return_pct"), 0.0)
    base_dd = _as_abs_drawdown(baseline_metrics)
    base_sharpe = _safe_float(baseline_metrics.get("sharpe"), 0.0)
    base_trades = _safe_float(baseline_metrics.get("total_trades"), 0.0)

    cand_ret = _safe_float(candidate_metrics.get("annualized_return_pct"), 0.0)
    cand_dd = _as_abs_drawdown(candidate_metrics)
    cand_sharpe = _safe_float(candidate_metrics.get("sharpe"), 0.0)
    cand_trades = _safe_float(candidate_metrics.get("total_trades"), 0.0)

    trade_target, trade_min, trade_max = _trade_bounds(base_trades, cfg)

    hard_constraints = {
        "annualized_return_up": bool(cand_ret >= base_ret),
        "max_drawdown_not_worse": bool(
            cand_dd <= (base_dd + cfg.mdd_worsen_tolerance_pct)
        ),
        "sharpe_not_down": bool(
            cand_sharpe >= (base_sharpe - cfg.sharpe_drop_tolerance)
        ),
        "trade_count_in_range": bool(trade_min <= cand_trades <= trade_max),
    }
    passed_constraints = bool(all(hard_constraints.values()))

    ret_delta = cand_ret - base_ret
    dd_delta = cand_dd - base_dd
    sharpe_delta = cand_sharpe - base_sharpe

    if cand_trades < trade_min:
        trade_deviation = (trade_min - cand_trades) / max(trade_target, 1.0)
    elif cand_trades > trade_max:
        trade_deviation = (cand_trades - trade_max) / max(trade_target, 1.0)
    else:
        trade_deviation = 0.0

    components = {
        "ret_component": float(np.tanh(ret_delta / 8.0)),
        "mdd_component": float(-np.tanh(dd_delta / 5.0)),
        "sharpe_component": float(np.tanh(sharpe_delta / 0.3)),
        "trade_component": float(-np.tanh(trade_deviation)),
        "ret_delta": float(ret_delta),
        "drawdown_delta_abs": float(dd_delta),
        "sharpe_delta": float(sharpe_delta),
        "trade_deviation": float(trade_deviation),
        "trade_target": float(trade_target),
        "trade_min": float(trade_min),
        "trade_max": float(trade_max),
    }

    raw_score = (
        cfg.ret_weight * components["ret_component"]
        + cfg.mdd_weight * components["mdd_component"]
        + cfg.sharpe_weight * components["sharpe_component"]
        + cfg.trade_weight * components["trade_component"]
    )

    # 未通过硬约束时施加固定惩罚，使排序优先满足可执行约束。
    penalty = 0.0 if passed_constraints else 1.0
    score = float(raw_score * 100.0 - penalty)

    return BacktestObjectiveScore(
        score=score,
        passed_constraints=passed_constraints,
        hard_constraints=hard_constraints,
        components=components,
        baseline={
            "annualized_return_pct": float(base_ret),
            "max_drawdown_abs_pct": float(base_dd),
            "sharpe": float(base_sharpe),
            "total_trades": float(base_trades),
        },
        candidate={
            "annualized_return_pct": float(cand_ret),
            "max_drawdown_abs_pct": float(cand_dd),
            "sharpe": float(cand_sharpe),
            "total_trades": float(cand_trades),
        },
        config=asdict(cfg),
    )
