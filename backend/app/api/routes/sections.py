"""Routes for the 6 app sections.

Each section gets its own router.  Phase 2 implements Single Stock Analysis;
other sections are filled in as development progresses.
"""

from __future__ import annotations

import asyncio
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.routes.auth import get_current_user
from app.models.schemas import AnalysisResponse
from app.models.user import User

logger = logging.getLogger(__name__)

# Thread pool for blocking engine calls (indicator computation, yfinance fetch)
_analysis_pool = ThreadPoolExecutor(max_workers=2)

# Trade mode → engine objective mapping
_TRADE_MODE_OBJECTIVES = {
    "swing": "swing_trade",
    "long_term": "long_term",
}

# ---------------------------------------------------------------------------
# Section 1: Market Scanner
# ---------------------------------------------------------------------------

scanner_router = APIRouter(prefix="/scanner", tags=["scanner"])


@scanner_router.get("/")
async def scanner_status(user: User = Depends(get_current_user)):
    return {
        "section": "Market Scanner",
        "status": "coming_in_phase_2",
        "description": "Scan configurable universes for trade candidates.",
    }


# ---------------------------------------------------------------------------
# Section 2: Single Stock Analysis
# ---------------------------------------------------------------------------

analysis_router = APIRouter(prefix="/analysis", tags=["analysis"])


def _sanitize_float(v: Any) -> Any:
    """Replace NaN/Inf with None so JSON serialization works."""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, dict):
        return {k: _sanitize_float(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_sanitize_float(item) for item in v]
    return v


def _run_analysis(ticker: str, period: str, interval: str, trade_mode: str) -> dict:
    """Run the engine Analyzer synchronously (called from thread pool).

    This function imports the engine modules at call time because they
    are mounted into the container via Docker volumes, not installed as
    packages.
    """
    from config import Config  # type: ignore[import-untyped]
    from data.yahoo import YahooFinanceProvider  # type: ignore[import-untyped]
    from analysis.analyzer import Analyzer  # type: ignore[import-untyped]

    # Build config with objective preset
    cfg = Config.defaults()
    objective = _TRADE_MODE_OBJECTIVES.get(trade_mode)
    if objective and objective in cfg.available_objectives():
        cfg.apply_objective(objective)

    provider = YahooFinanceProvider()
    analyzer = Analyzer(cfg, provider)
    result = analyzer.run(ticker, period=period, interval=interval)

    # Convert DataFrame → list of OHLCV bars
    price_data = []
    for dt, row in result.df.iterrows():
        price_data.append({
            "date": dt.isoformat() if hasattr(dt, "isoformat") else str(dt),
            "open": _sanitize_float(float(row["open"])),
            "high": _sanitize_float(float(row["high"])),
            "low": _sanitize_float(float(row["low"])),
            "close": _sanitize_float(float(row["close"])),
            "volume": _sanitize_float(float(row.get("volume", 0))),
        })

    # Convert indicator results
    indicators = []
    for ir in result.indicator_results:
        indicators.append({
            "name": ir.name,
            "config_key": ir.config_key,
            "score": _sanitize_float(ir.score),
            "values": _sanitize_float(ir.values),
            "display": _sanitize_float(ir.display),
            "error": ir.error,
        })

    # Convert pattern results
    patterns = []
    for pr in result.pattern_results:
        patterns.append({
            "name": pr.name,
            "config_key": pr.config_key,
            "score": _sanitize_float(pr.score),
            "values": _sanitize_float(pr.values),
            "display": _sanitize_float(pr.display),
            "error": pr.error,
        })

    # Convert S/R levels
    support_levels = [
        {"price": lvl.price, "level_type": lvl.level_type, "source": lvl.source,
         "touches": lvl.touches, "label": lvl.label}
        for lvl in result.support_levels
    ]
    resistance_levels = [
        {"price": lvl.price, "level_type": lvl.level_type, "source": lvl.source,
         "touches": lvl.touches, "label": lvl.label}
        for lvl in result.resistance_levels
    ]

    # Convert composite scores
    composite_raw = _sanitize_float(result.composite)
    composite = {
        "overall": composite_raw.get("overall", 5.0),
        "breakdown": composite_raw.get("breakdown", []),
        "trend_score": composite_raw.get("trend_score"),
        "contrarian_score": composite_raw.get("contrarian_score"),
        "neutral_score": composite_raw.get("neutral_score"),
        "dominant_group": composite_raw.get("dominant_group"),
    }

    # Convert regime assessment
    regime = None
    if result.regime:
        r = result.regime
        regime = {
            "regime": r.regime.value,
            "confidence": _sanitize_float(r.confidence),
            "label": r.label,
            "description": r.description,
            "metrics": _sanitize_float({
                "adx": r.metrics.adx,
                "rolling_adx_mean": r.metrics.rolling_adx_mean,
                "total_return": r.metrics.total_return,
                "pct_above_ma": r.metrics.pct_above_ma,
                "atr_pct": r.metrics.atr_pct,
                "bb_width": r.metrics.bb_width,
                "bb_width_percentile": r.metrics.bb_width_percentile,
                "price_ma_distance": r.metrics.price_ma_distance,
                "direction_changes": r.metrics.direction_changes,
                "trend_direction": r.metrics.trend_direction,
            }),
            "reasons": r.reasons,
            "regime_scores": _sanitize_float(r.regime_scores),
            "sub_type": r.sub_type.value if r.sub_type else None,
            "sub_type_label": r.sub_type_label,
            "sub_type_description": r.sub_type_description,
        }

    return {
        "ticker": result.ticker,
        "period": result.period,
        "info": _sanitize_float(result.info),
        "price_data": price_data,
        "indicators": indicators,
        "patterns": patterns,
        "support_levels": support_levels,
        "resistance_levels": resistance_levels,
        "composite": composite,
        "pattern_composite": _sanitize_float(result.pattern_composite),
        "regime": regime,
    }


@analysis_router.get("/{ticker}", response_model=AnalysisResponse)
async def analyze_stock(
    ticker: str,
    period: str = Query("6mo", description="Data period: 1mo, 3mo, 6mo, 1y, 2y, 5y"),
    interval: str = Query("1d", description="Bar interval: 1d, 1wk, 1mo"),
    user: User = Depends(get_current_user),
):
    """Run full technical analysis on a single ticker.

    Uses the engine's Analyzer to compute indicators, patterns, support/resistance,
    composite scores, and regime classification.  The trade_mode from the user's
    profile selects the appropriate configuration objective (swing_trade, long_term).
    """
    valid_periods = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"}
    valid_intervals = {"1d", "5d", "1wk", "1mo"}
    if period not in valid_periods:
        raise HTTPException(400, f"Invalid period '{period}'. Must be one of: {sorted(valid_periods)}")
    if interval not in valid_intervals:
        raise HTTPException(400, f"Invalid interval '{interval}'. Must be one of: {sorted(valid_intervals)}")

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            _analysis_pool,
            _run_analysis,
            ticker.upper(),
            period,
            interval,
            user.trade_mode,
        )
    except Exception as exc:
        logger.exception("Analysis failed for %s", ticker)
        raise HTTPException(500, f"Analysis failed: {exc}") from exc

    return result


# ---------------------------------------------------------------------------
# Section 3: Sector & Segments
# ---------------------------------------------------------------------------

sectors_router = APIRouter(prefix="/sectors", tags=["sectors"])


@sectors_router.get("/overview")
async def sector_overview(user: User = Depends(get_current_user)):
    return {
        "section": "Sector & Segments",
        "status": "coming_in_phase_2",
        "description": "Sector heatmap, segment drill-down, rotation tracker, and relative strength.",
    }


# ---------------------------------------------------------------------------
# Section 4: Strategy Lab
# ---------------------------------------------------------------------------

strategy_router = APIRouter(prefix="/strategy", tags=["strategy"])


@strategy_router.get("/")
async def strategy_lab_status(user: User = Depends(get_current_user)):
    return {
        "section": "Strategy Lab",
        "status": "coming_in_phase_3",
        "description": "Backtester, walk-forward testing, auto-tuner, and strategy library.",
    }


@strategy_router.get("/library")
async def list_strategies(user: User = Depends(get_current_user)):
    return {
        "strategies": [],
        "status": "coming_in_phase_3",
    }


# ---------------------------------------------------------------------------
# Section 5: Portfolio Simulation
# ---------------------------------------------------------------------------

portfolio_router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@portfolio_router.get("/")
async def portfolio_status(user: User = Depends(get_current_user)):
    return {
        "section": "Portfolio Simulation",
        "status": "coming_in_phase_4",
        "description": "Full trading pipeline: signal intake, position selection, capital allocation, execution.",
    }


@portfolio_router.get("/summary")
async def portfolio_summary(user: User = Depends(get_current_user)):
    return {
        "cash_balance": user.starting_capital,
        "total_equity": user.starting_capital,
        "open_positions": 0,
        "total_pnl": 0.0,
        "total_pnl_pct": 0.0,
        "status": "coming_in_phase_4",
    }


# ---------------------------------------------------------------------------
# Section 6: Settings  (mostly handled by auth/me endpoints)
# ---------------------------------------------------------------------------

settings_router = APIRouter(prefix="/settings", tags=["settings"])


@settings_router.get("/")
async def get_settings(user: User = Depends(get_current_user)):
    return {
        "section": "Settings",
        "trade_mode": user.trade_mode,
        "user_mode": user.user_mode,
        "starting_capital": user.starting_capital,
        "risk_tolerance": user.risk_tolerance,
        "cost_model": {
            "commission_per_trade": user.commission_per_trade,
            "spread_pct": user.spread_pct,
            "slippage_pct": user.slippage_pct,
            "tax_rate_short_term": user.tax_rate_short_term,
            "tax_rate_long_term": user.tax_rate_long_term,
        },
        "data_providers": {
            "yahoo_finance": {"enabled": True, "status": "active"},
            "polygon": {"enabled": False, "status": "no_api_key"},
            "alpha_vantage": {"enabled": False, "status": "no_api_key"},
        },
    }
