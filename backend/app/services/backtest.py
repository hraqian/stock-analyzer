"""Backend service wrapper for the backtest engine.

Provides a synchronous ``run_backtest()`` function that instantiates the
engine components (Config, DataProvider, Strategy, BacktestEngine) and
returns a plain dict suitable for JSON serialisation.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

# Trade mode → engine objective mapping (same as analysis)
_TRADE_MODE_OBJECTIVES = {
    "swing": "swing_trade",
    "long_term": "long_term",
}


def _safe(v: Any) -> Any:
    """Recursively sanitise a value for JSON (NaN/Inf → None, numpy/pandas → native)."""
    import numpy as np
    import pandas as pd

    if isinstance(v, (pd.Series, pd.Index)):
        return [_safe(x) for x in v.tolist()]
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        val = float(v)
        return None if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(v, np.ndarray):
        return [_safe(x) for x in v.tolist()]
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(v, dict):
        return {str(k): _safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe(item) for item in v]
    return v


def run_backtest(
    ticker: str,
    trade_mode: str = "swing",
    period: str = "2y",
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
    initial_cash: float = 100_000.0,
    commission_pct: float = 0.0,
    slippage_pct: float = 0.001,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
) -> dict:
    """Run a single-ticker backtest and return JSON-safe results.

    Parameters match the API request schema.  Engine modules are imported
    lazily because they are volume-mounted into the Docker container.
    """
    from config import Config                        # type: ignore[import-untyped]
    from data.yahoo import YahooFinanceProvider      # type: ignore[import-untyped]
    from engine.backtest import BacktestEngine       # type: ignore[import-untyped]
    from engine.score_strategy import ScoreBasedStrategy  # type: ignore[import-untyped]
    from engine.suitability import TradingMode       # type: ignore[import-untyped]

    # 1. Build config with the right objective preset
    cfg = Config.defaults()
    objective = _TRADE_MODE_OBJECTIVES.get(trade_mode)
    if objective and objective in cfg.available_objectives():
        cfg.apply_objective(objective)

    # 2. Override cost-model parameters from request
    bt_section = cfg.section("backtest")
    bt_section["initial_cash"] = initial_cash
    bt_section["commission_pct"] = commission_pct
    bt_section["slippage_pct"] = slippage_pct

    strat_section = cfg.section("strategy")
    if stop_loss_pct is not None:
        strat_section["stop_loss_pct"] = stop_loss_pct
    if take_profit_pct is not None:
        strat_section["take_profit_pct"] = take_profit_pct

    # 3. Build components
    provider = YahooFinanceProvider()
    strategy = ScoreBasedStrategy(
        params=strat_section,
        trading_mode=TradingMode.LONG_SHORT,
    )
    engine = BacktestEngine(
        data_provider=provider,
        strategy=strategy,
        cfg=cfg,
        trading_mode=TradingMode.LONG_SHORT,
    )

    # 4. Run
    result = engine.run(
        ticker=ticker.upper(),
        period=period if not start else None,
        interval=interval,
        start=start,
        end=end,
    )

    # 5. Convert to JSON-safe dict
    trades = [
        {
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "entry_price": _safe(t.entry_price),
            "exit_price": _safe(t.exit_price),
            "quantity": _safe(t.quantity),
            "side": t.side,
            "pnl": _safe(t.pnl),
            "pnl_pct": _safe(t.pnl_pct),
            "exit_reason": t.exit_reason,
            "entry_reason": t.entry_reason,
            "bars_held": t.bars_held,
        }
        for t in result.trades
    ]

    equity_curve = [
        {"date": pt["date"], "equity": _safe(pt["equity"])}
        for pt in result.equity_curve
    ]

    regime = None
    if result.regime:
        r = result.regime
        regime = {
            "regime": r.regime.value,
            "confidence": _safe(r.confidence),
            "label": r.label,
            "description": r.description,
        }

    return {
        "ticker": result.ticker,
        "period": result.period,
        "strategy_name": result.strategy_name,
        "initial_cash": _safe(result.initial_cash),
        "final_equity": _safe(result.final_equity),
        "total_return_pct": _safe(result.total_return_pct),
        "annualized_return_pct": _safe(result.annualized_return_pct),
        "max_drawdown_pct": _safe(result.max_drawdown_pct),
        "sharpe_ratio": _safe(result.sharpe_ratio),
        "win_rate_pct": _safe(result.win_rate_pct),
        "total_trades": result.total_trades,
        "profit_factor": _safe(result.profit_factor),
        "avg_trade_pnl_pct": _safe(result.avg_trade_pnl_pct),
        "best_trade_pnl_pct": _safe(result.best_trade_pnl_pct),
        "worst_trade_pnl_pct": _safe(result.worst_trade_pnl_pct),
        "avg_bars_held": _safe(result.avg_bars_held),
        "trades": trades,
        "equity_curve": equity_curve,
        "regime": regime,
        "warmup_bars": result.warmup_bars,
    }
