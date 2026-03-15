"""Backend service wrapper for the backtest engine.

Provides a synchronous ``run_backtest()`` function that runs walk-forward
validation (multiple rolling train/test windows) and returns a unified
result: detailed metrics from the most recent window (equity curve,
trades, regime) plus aggregated robustness stats across all windows.
"""

from __future__ import annotations

import logging
import math
import statistics
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


def _backtest_result_to_dict(result: Any) -> dict:
    """Convert a BacktestResult to a JSON-safe dict (detailed view)."""
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


def _generate_windows(
    train_years: int,
    test_years: int,
    max_windows: int = 5,
) -> list[dict[str, str]]:
    """Generate date windows working backwards from today.

    Returns a list of dicts with keys:
        train_start, train_end, test_start, test_end  (YYYY-MM-DD strings).
    """
    from datetime import datetime, timedelta

    today = datetime.now().date()
    windows: list[dict[str, str]] = []

    for i in range(max_windows):
        test_end = today - timedelta(days=i * test_years * 365)
        test_start = test_end - timedelta(days=test_years * 365)
        train_end = test_start - timedelta(days=1)
        train_start = test_start - timedelta(days=train_years * 365)

        # Don't go before 2000 (insufficient data)
        if train_start.year < 2000:
            break

        windows.append({
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
        })

    windows.reverse()  # chronological order
    return windows


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
    train_years: int = 3,
    test_years: int = 1,
    max_windows: int = 5,
) -> dict:
    """Run a unified backtest with walk-forward robustness analysis.

    1. Generates rolling train/test windows
    2. Runs backtest on each window
    3. Returns detailed results from the most recent window (equity curve,
       trades, regime) plus aggregated robustness metrics across all windows

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

    provider = YahooFinanceProvider()
    ticker_upper = ticker.upper()

    # 3. Generate walk-forward windows
    windows = _generate_windows(train_years, test_years, max_windows)

    # Per-window results for robustness tracking
    window_results: list[dict] = []
    # We'll keep the detailed backtest result from the LAST (most recent) window
    latest_bt_result = None

    for i, w in enumerate(windows):
        logger.info(
            "Walk-forward window %d/%d: test %s → %s",
            i + 1, len(windows), w["test_start"], w["test_end"],
        )

        wr: dict = {
            "window_index": i,
            "train_start": w["train_start"],
            "train_end": w["train_end"],
            "test_start": w["test_start"],
            "test_end": w["test_end"],
            "total_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            "error": None,
        }

        try:
            # Fresh strategy for each window
            strategy = ScoreBasedStrategy(
                params=dict(strat_section),
                trading_mode=TradingMode.LONG_SHORT,
            )
            engine = BacktestEngine(
                data_provider=provider,
                strategy=strategy,
                cfg=cfg,
                trading_mode=TradingMode.LONG_SHORT,
            )

            bt = engine.run(
                ticker=ticker_upper,
                period=None,
                interval="1d",
                start=w["train_start"],  # include train period for warmup
                end=w["test_end"],
            )

            wr["total_return_pct"] = _safe(bt.total_return_pct)
            wr["annualized_return_pct"] = _safe(bt.annualized_return_pct)
            wr["max_drawdown_pct"] = _safe(bt.max_drawdown_pct)
            wr["sharpe_ratio"] = _safe(bt.sharpe_ratio)
            wr["win_rate_pct"] = _safe(bt.win_rate_pct)
            wr["profit_factor"] = _safe(bt.profit_factor)
            wr["total_trades"] = bt.total_trades

            # Keep detailed result from the last (most recent) window
            if i == len(windows) - 1:
                latest_bt_result = bt

        except Exception as exc:
            logger.warning("Window %d failed: %s", i, exc)
            wr["error"] = str(exc)
            # If last window fails, try to use previous successful one
            if i == len(windows) - 1 and latest_bt_result is None:
                # Find most recent successful window to use as detailed result
                pass

        window_results.append(wr)

    # 4. If no windows were generated, fall back to a simple single backtest
    if not windows:
        logger.info("No walk-forward windows generated, running single backtest")
        strategy = ScoreBasedStrategy(
            params=dict(strat_section),
            trading_mode=TradingMode.LONG_SHORT,
        )
        engine = BacktestEngine(
            data_provider=provider,
            strategy=strategy,
            cfg=cfg,
            trading_mode=TradingMode.LONG_SHORT,
        )
        bt = engine.run(
            ticker=ticker_upper,
            period=period if not start else None,
            interval=interval,
            start=start,
            end=end,
        )
        result_dict = _backtest_result_to_dict(bt)
        result_dict.update({
            "train_years": train_years,
            "test_years": test_years,
            "total_windows": 0,
            "windows": [],
            "wf_avg_return_pct": 0.0,
            "wf_avg_annualized_return_pct": 0.0,
            "wf_avg_max_drawdown_pct": 0.0,
            "wf_avg_sharpe_ratio": 0.0,
            "wf_avg_win_rate_pct": 0.0,
            "wf_avg_profit_factor": 0.0,
            "wf_worst_return_pct": 0.0,
            "wf_worst_drawdown_pct": 0.0,
            "wf_worst_window_index": 0,
            "wf_return_std_dev": 0.0,
            "stability_score": 0.0,
            "verdict": "Not enough historical data for walk-forward analysis.",
        })
        return result_dict

    # 5. Build detailed result from the latest window
    if latest_bt_result is not None:
        result_dict = _backtest_result_to_dict(latest_bt_result)
    else:
        # All windows failed — return empty detailed result
        result_dict = {
            "ticker": ticker_upper,
            "period": period,
            "strategy_name": "ScoreBasedStrategy",
            "initial_cash": _safe(initial_cash),
            "final_equity": _safe(initial_cash),
            "total_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate_pct": 0.0,
            "total_trades": 0,
            "profit_factor": 0.0,
            "avg_trade_pnl_pct": 0.0,
            "best_trade_pnl_pct": 0.0,
            "worst_trade_pnl_pct": 0.0,
            "avg_bars_held": 0.0,
            "trades": [],
            "equity_curve": [],
            "regime": None,
            "warmup_bars": 0,
        }

    # 6. Compute walk-forward aggregate metrics
    ok = [w for w in window_results if w["error"] is None]

    wf_avg_return = 0.0
    wf_avg_ann_return = 0.0
    wf_avg_drawdown = 0.0
    wf_avg_sharpe = 0.0
    wf_avg_win_rate = 0.0
    wf_avg_pf = 0.0
    wf_worst_return = 0.0
    wf_worst_drawdown = 0.0
    wf_worst_idx = 0
    wf_std_dev = 0.0
    stability = 0.0
    verdict = ""

    if ok:
        wf_avg_return = statistics.mean(w["total_return_pct"] for w in ok)
        wf_avg_ann_return = statistics.mean(w["annualized_return_pct"] for w in ok)
        wf_avg_drawdown = statistics.mean(w["max_drawdown_pct"] for w in ok)
        wf_avg_sharpe = statistics.mean(w["sharpe_ratio"] for w in ok)
        wf_avg_win_rate = statistics.mean(w["win_rate_pct"] for w in ok)
        wf_avg_pf = statistics.mean(w["profit_factor"] for w in ok)

        # Worst case
        worst = min(ok, key=lambda w: w["total_return_pct"])
        wf_worst_return = worst["total_return_pct"]
        wf_worst_drawdown = worst["max_drawdown_pct"]
        wf_worst_idx = worst["window_index"]

        # Stability score
        if len(ok) >= 2:
            returns = [w["total_return_pct"] for w in ok]
            wf_std_dev = statistics.stdev(returns)
            raw = 100.0 - wf_std_dev * 5.0
            stability = max(0.0, min(100.0, raw))
        else:
            stability = 50.0  # unknown with single window

        # Verdict
        n_profitable = sum(1 for w in ok if w["total_return_pct"] > 0)
        pct_profitable = n_profitable / len(ok) * 100

        if stability >= 70 and pct_profitable >= 60:
            verdict = (
                f"Strategy is robust across {len(ok)} time periods. "
                f"{pct_profitable:.0f}% of windows were profitable "
                f"with a stability score of {stability:.0f}/100."
            )
        elif stability >= 40:
            verdict = (
                f"Strategy shows moderate consistency across {len(ok)} time periods. "
                f"{pct_profitable:.0f}% of windows were profitable. "
                f"Returns vary with a std dev of {wf_std_dev:.1f}%."
            )
        else:
            verdict = (
                f"Warning: significant performance variation across {len(ok)} time periods. "
                f"Only {pct_profitable:.0f}% were profitable. "
                f"The strategy may be overfit or highly regime-dependent."
            )
    else:
        verdict = "All walk-forward windows failed. Check data availability."

    result_dict.update({
        "train_years": train_years,
        "test_years": test_years,
        "total_windows": len(windows),
        "windows": window_results,
        "wf_avg_return_pct": _safe(wf_avg_return),
        "wf_avg_annualized_return_pct": _safe(wf_avg_ann_return),
        "wf_avg_max_drawdown_pct": _safe(wf_avg_drawdown),
        "wf_avg_sharpe_ratio": _safe(wf_avg_sharpe),
        "wf_avg_win_rate_pct": _safe(wf_avg_win_rate),
        "wf_avg_profit_factor": _safe(wf_avg_pf),
        "wf_worst_return_pct": _safe(wf_worst_return),
        "wf_worst_drawdown_pct": _safe(wf_worst_drawdown),
        "wf_worst_window_index": wf_worst_idx,
        "wf_return_std_dev": _safe(wf_std_dev),
        "stability_score": _safe(stability),
        "verdict": verdict,
    })

    return result_dict
