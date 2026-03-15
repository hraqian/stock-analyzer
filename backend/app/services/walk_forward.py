"""Backend service wrapper for walk-forward testing.

Provides a synchronous ``run_walk_forward()`` function that returns
a plain dict suitable for JSON serialisation.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

from app.services.shared import TRADE_MODE_OBJECTIVES, safe



def run_walk_forward(
    ticker: str,
    trade_mode: str = "swing",
    train_years: int = 5,
    test_years: int = 1,
    max_windows: int = 10,
    tax_marginal_rate: float = 0.0,
    tax_treatment: str = "",
) -> dict:
    """Run walk-forward testing and return JSON-safe results."""
    from config import Config                            # type: ignore[import-untyped]
    from data.yahoo import YahooFinanceProvider          # type: ignore[import-untyped]
    from engine.walk_forward import WalkForwardEngine    # type: ignore[import-untyped]
    from engine.score_strategy import ScoreBasedStrategy # type: ignore[import-untyped]
    from engine.suitability import TradingMode           # type: ignore[import-untyped]

    cfg = Config.defaults()
    objective = TRADE_MODE_OBJECTIVES.get(trade_mode)
    if objective and objective in cfg.available_objectives():
        cfg.apply_objective(objective)

    # Inject tax params into backtest config
    bt_section = cfg.section("backtest")
    bt_section["tax_marginal_rate"] = tax_marginal_rate
    bt_section["tax_treatment"] = tax_treatment

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
            "total_return_pct": safe(w.total_return_pct),
            "annualized_return_pct": safe(w.annualized_return_pct),
            "max_drawdown_pct": safe(w.max_drawdown_pct),
            "sharpe_ratio": safe(w.sharpe_ratio),
            "win_rate_pct": safe(w.win_rate_pct),
            "profit_factor": safe(w.profit_factor),
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
        "avg_return_pct": safe(result.avg_return_pct),
        "avg_annualized_return_pct": safe(result.avg_annualized_return_pct),
        "avg_max_drawdown_pct": safe(result.avg_max_drawdown_pct),
        "avg_sharpe_ratio": safe(result.avg_sharpe_ratio),
        "avg_win_rate_pct": safe(result.avg_win_rate_pct),
        "avg_profit_factor": safe(result.avg_profit_factor),
        "worst_return_pct": safe(result.worst_return_pct),
        "worst_drawdown_pct": safe(result.worst_drawdown_pct),
        "worst_window_index": result.worst_window_index,
        "return_std_dev": safe(result.return_std_dev),
        "stability_score": safe(result.stability_score),
        "verdict": result.verdict,
    }
