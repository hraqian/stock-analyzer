"""Backend service wrapper for walk-forward testing.

Provides a synchronous ``run_walk_forward()`` function that returns
a plain dict suitable for JSON serialisation.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

_TRADE_MODE_OBJECTIVES = {
    "swing": "swing_trade",
    "long_term": "long_term",
}


def _safe(v: Any) -> Any:
    """Recursively sanitise for JSON."""
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(v, dict):
        return {str(k): _safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe(item) for item in v]
    return v


def run_walk_forward(
    ticker: str,
    trade_mode: str = "swing",
    train_years: int = 5,
    test_years: int = 1,
    max_windows: int = 10,
) -> dict:
    """Run walk-forward testing and return JSON-safe results."""
    from config import Config                            # type: ignore[import-untyped]
    from data.yahoo import YahooFinanceProvider          # type: ignore[import-untyped]
    from engine.walk_forward import WalkForwardEngine    # type: ignore[import-untyped]
    from engine.score_strategy import ScoreBasedStrategy # type: ignore[import-untyped]
    from engine.suitability import TradingMode           # type: ignore[import-untyped]

    cfg = Config.defaults()
    objective = _TRADE_MODE_OBJECTIVES.get(trade_mode)
    if objective and objective in cfg.available_objectives():
        cfg.apply_objective(objective)

    provider = YahooFinanceProvider()
    strat_section = cfg.section("strategy")

    def strategy_factory() -> ScoreBasedStrategy:
        return ScoreBasedStrategy(
            params=dict(strat_section),
            trading_mode=TradingMode.LONG_SHORT,
        )

    engine = WalkForwardEngine(
        data_provider=provider,
        strategy_factory=strategy_factory,
        cfg=cfg,
    )

    result = engine.run(
        ticker=ticker.upper(),
        train_years=train_years,
        test_years=test_years,
        max_windows=max_windows,
    )

    windows = [
        {
            "window_index": w.window_index,
            "train_start": w.train_start,
            "train_end": w.train_end,
            "test_start": w.test_start,
            "test_end": w.test_end,
            "total_return_pct": _safe(w.total_return_pct),
            "annualized_return_pct": _safe(w.annualized_return_pct),
            "max_drawdown_pct": _safe(w.max_drawdown_pct),
            "sharpe_ratio": _safe(w.sharpe_ratio),
            "win_rate_pct": _safe(w.win_rate_pct),
            "profit_factor": _safe(w.profit_factor),
            "total_trades": w.total_trades,
            "error": w.error,
        }
        for w in result.windows
    ]

    return {
        "ticker": result.ticker,
        "train_years": result.train_years,
        "test_years": result.test_years,
        "total_windows": result.total_windows,
        "windows": windows,
        "avg_return_pct": _safe(result.avg_return_pct),
        "avg_annualized_return_pct": _safe(result.avg_annualized_return_pct),
        "avg_max_drawdown_pct": _safe(result.avg_max_drawdown_pct),
        "avg_sharpe_ratio": _safe(result.avg_sharpe_ratio),
        "avg_win_rate_pct": _safe(result.avg_win_rate_pct),
        "avg_profit_factor": _safe(result.avg_profit_factor),
        "worst_return_pct": _safe(result.worst_return_pct),
        "worst_drawdown_pct": _safe(result.worst_drawdown_pct),
        "worst_window_index": result.worst_window_index,
        "return_std_dev": _safe(result.return_std_dev),
        "stability_score": _safe(result.stability_score),
        "verdict": result.verdict,
    }
