"""Backend service wrapper for the auto-tuner.

Provides a synchronous ``run_auto_tune()`` function that returns
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

VALID_OBJECTIVES = {
    "beat_buy_hold",
    "max_return",
    "max_risk_adjusted",
    "min_drawdown",
    "balanced",
}


def _safe(v: Any) -> Any:
    """Recursively sanitise for JSON (NaN/Inf → None, numpy/pandas → native)."""
    import numpy as np

    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
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


# Map sector names to representative tickers (imported from sectors service)
def _resolve_sector_tickers(sector: str) -> list[str]:
    """Resolve a sector name to its representative tickers."""
    from app.services.sectors import SECTOR_HOLDINGS  # late import
    holdings = SECTOR_HOLDINGS.get(sector)
    if not holdings:
        raise ValueError(f"Unknown sector: {sector}")
    return [ticker for ticker, _name in holdings]


def run_auto_tune(
    ticker: str | None = None,
    tickers: list[str] | None = None,
    sector: str | None = None,
    trade_mode: str = "swing",
    objective: str = "balanced",
    n_trials: int = 30,
    train_years: int = 3,
    test_years: int = 1,
    max_windows: int = 3,
) -> dict:
    """Run the auto-tuner and return JSON-safe results.

    Supports three modes:
      - Single ticker: provide ``ticker``
      - Sector group: provide ``sector`` (resolved to representative tickers)
      - Custom group: provide ``tickers`` list

    Parameters match the API request schema.  Engine modules are imported
    lazily because they are volume-mounted into the Docker container.
    """
    from config import Config                        # type: ignore[import-untyped]
    from data.yahoo import YahooFinanceProvider      # type: ignore[import-untyped]
    from engine.auto_tuner import AutoTuner          # type: ignore[import-untyped]

    # Determine mode and resolve ticker list
    if sector:
        mode = "sector"
        resolved_tickers = _resolve_sector_tickers(sector)
    elif tickers and len(tickers) > 0:
        mode = "custom"
        resolved_tickers = [t.upper() for t in tickers]
    elif ticker:
        mode = "single"
        resolved_tickers = [ticker.upper()]
    else:
        raise ValueError("Must provide ticker, tickers, or sector")

    # 1. Build config with the right objective preset
    cfg = Config.defaults()
    obj_key = _TRADE_MODE_OBJECTIVES.get(trade_mode)
    if obj_key and obj_key in cfg.available_objectives():
        cfg.apply_objective(obj_key)

    # 2. Build components
    provider = YahooFinanceProvider()
    tuner = AutoTuner(data_provider=provider, cfg=cfg)

    # 3. Run
    result = tuner.run(
        tickers=resolved_tickers,
        objective=objective,
        n_trials=n_trials,
        train_years=train_years,
        test_years=test_years,
        max_windows=max_windows,
        mode=mode,
        sector=sector,
    )

    # 4. Convert to JSON-safe dict
    sensitivity = [
        {
            "param_name": s.param_name,
            "importance": _safe(s.importance),
            "best_value": _safe(s.best_value),
            "value_range": _safe(s.value_range),
        }
        for s in result.sensitivity
    ]

    trials = [
        {
            "trial_number": t.trial_number,
            "params": _safe(t.params),
            "objective_value": _safe(t.objective_value),
            "avg_return_pct": _safe(t.avg_return_pct),
            "avg_annualized_return_pct": _safe(t.avg_annualized_return_pct),
            "avg_max_drawdown_pct": _safe(t.avg_max_drawdown_pct),
            "avg_sharpe_ratio": _safe(t.avg_sharpe_ratio),
            "avg_win_rate_pct": _safe(t.avg_win_rate_pct),
            "avg_profit_factor": _safe(t.avg_profit_factor),
            "stability_score": _safe(t.stability_score),
            "total_windows": t.total_windows,
        }
        for t in result.trials
    ]

    return {
        "ticker": result.ticker,
        "tickers": result.tickers,
        "mode": result.mode,
        "sector": result.sector,
        "objective": result.objective,
        "objective_label": result.objective_label,
        "n_trials": result.n_trials,
        "elapsed_seconds": _safe(result.elapsed_seconds),
        # Best trial
        "best_params": _safe(result.best_params),
        "best_objective_value": _safe(result.best_objective_value),
        "best_avg_return_pct": _safe(result.best_avg_return_pct),
        "best_avg_annualized_return_pct": _safe(result.best_avg_annualized_return_pct),
        "best_avg_max_drawdown_pct": _safe(result.best_avg_max_drawdown_pct),
        "best_avg_sharpe_ratio": _safe(result.best_avg_sharpe_ratio),
        "best_avg_win_rate_pct": _safe(result.best_avg_win_rate_pct),
        "best_avg_profit_factor": _safe(result.best_avg_profit_factor),
        "best_stability_score": _safe(result.best_stability_score),
        # Baseline
        "baseline_avg_return_pct": _safe(result.baseline_avg_return_pct),
        "baseline_avg_annualized_return_pct": _safe(result.baseline_avg_annualized_return_pct),
        "baseline_avg_max_drawdown_pct": _safe(result.baseline_avg_max_drawdown_pct),
        "baseline_avg_sharpe_ratio": _safe(result.baseline_avg_sharpe_ratio),
        "baseline_avg_win_rate_pct": _safe(result.baseline_avg_win_rate_pct),
        "baseline_objective_value": _safe(result.baseline_objective_value),
        # Buy-and-hold
        "buy_hold_return_pct": _safe(result.buy_hold_return_pct),
        # Sensitivity & trials (for power user mode)
        "sensitivity": sensitivity,
        "trials": trials,
        # Verdict
        "verdict": result.verdict,
        "improvement_pct": _safe(result.improvement_pct),
    }
