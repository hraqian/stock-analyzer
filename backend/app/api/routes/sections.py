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
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import get_current_user
from app.core.database import get_db
from app.models.schemas import (
    AnalysisResponse,
    AutoTuneRequest,
    AutoTuneResponse,
    BacktestRequest,
    BacktestResponse,
    ScanRequest,
    ScanResponse,
    SectorDetailResponse,
    SectorHoldingsResponse,
    SectorHoldingsUpdateRequest,
    SectorOverviewResponse,
    StrategyCreateRequest,
    StrategyExportResponse,
    StrategyImportRequest,
    StrategyListResponse,
    StrategyResponse,
    StrategyUpdateRequest,
    UniverseListResponse,
    WalkForwardRequest,
    WalkForwardResponse,
    VALID_PRESETS,
    VALID_TUNER_OBJECTIVES,
    VALID_UNIVERSES,
)
from app.models.user import User

logger = logging.getLogger(__name__)

# Thread pool for blocking engine calls (indicator computation, yfinance fetch)
_analysis_pool = ThreadPoolExecutor(max_workers=2)
# Separate pool for scanner (can run long batch fetches)
_scanner_pool = ThreadPoolExecutor(max_workers=1)
# Separate pool for sector analysis
_sector_pool = ThreadPoolExecutor(max_workers=1)

from app.services.shared import TRADE_MODE_OBJECTIVES

# ---------------------------------------------------------------------------
# Section 1: Market Scanner
# ---------------------------------------------------------------------------

scanner_router = APIRouter(prefix="/scanner", tags=["scanner"])


def _run_scanner(
    tickers: list[str],
    preset: str,
    trade_mode: str,
    period: str,
    min_volume: int,
    min_price: float,
    max_atr_ratio: float | None,
    top_n: int,
    universe_name: str,
) -> dict:
    """Run the scanner engine synchronously (called from thread pool)."""
    from app.services.scanner import run_scan  # late import

    summary = run_scan(
        tickers=tickers,
        preset=preset,
        trade_mode=trade_mode,
        period=period,
        min_volume=min_volume,
        min_price=min_price,
        max_atr_ratio=max_atr_ratio,
        top_n=top_n,
        universe_name=universe_name,
    )
    return {
        "preset": summary.preset,
        "universe": summary.universe,
        "total_tickers": summary.total_tickers,
        "tickers_with_data": summary.tickers_with_data,
        "tickers_passing_filters": summary.tickers_passing_filters,
        "elapsed_seconds": summary.elapsed_seconds,
        "results": [
            {
                "rank": r.rank,
                "ticker": r.ticker,
                "signal": r.signal,
                "score": r.score,
                "confidence": r.confidence,
                "pattern": r.pattern,
                "regime": r.regime,
                "sector": r.sector,
                "breakdown": r.breakdown,
                "volume": r.volume,
                "price": r.price,
                "atr_ratio": r.atr_ratio,
            }
            for r in summary.results
        ],
    }


@scanner_router.post("/scan", response_model=ScanResponse)
async def run_market_scan(
    req: ScanRequest,
    user: User = Depends(get_current_user),
):
    """Run the market scanner on a universe of tickers.

    POST body fields:
    - universe: one of the 12 predefined universes or "custom"
    - custom_tickers: required if universe == "custom"
    - preset: "breakout", "pullback", "reversal", "dividend"
    - period: data lookback (e.g. "6mo")
    - min_volume, min_price, max_atr_ratio: filters
    - top_n: max results to return
    """
    # Validate preset
    if req.preset not in VALID_PRESETS:
        raise HTTPException(400, f"Invalid preset '{req.preset}'. Must be one of: {sorted(VALID_PRESETS)}")

    # Load tickers from universe
    if req.universe == "custom":
        if not req.custom_tickers:
            raise HTTPException(400, "custom_tickers required when universe is 'custom'")
        tickers = [t.upper().strip() for t in req.custom_tickers if t.strip()]
        if not tickers:
            raise HTTPException(400, "custom_tickers list is empty")
    else:
        if req.universe not in VALID_UNIVERSES:
            raise HTTPException(400, f"Invalid universe '{req.universe}'. Must be one of: {sorted(VALID_UNIVERSES)}")
        try:
            from data.universes import load as load_universe  # type: ignore[import-untyped]
            tickers = load_universe(req.universe)
        except FileNotFoundError:
            raise HTTPException(404, f"Universe file not found: {req.universe}")
        except Exception as exc:
            raise HTTPException(500, f"Failed to load universe: {exc}") from exc

    if not tickers:
        raise HTTPException(400, "No tickers in the selected universe")

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            _scanner_pool,
            _run_scanner,
            tickers,
            req.preset,
            user.trade_mode,
            req.period,
            req.min_volume,
            req.min_price,
            req.max_atr_ratio,
            req.top_n,
            req.universe,
        )
    except Exception as exc:
        logger.exception("Scanner failed for universe=%s preset=%s", req.universe, req.preset)
        raise HTTPException(500, f"Scanner failed: {exc}") from exc

    return result


@scanner_router.get("/universes", response_model=UniverseListResponse)
async def list_universes(user: User = Depends(get_current_user)):
    """List available universe names."""
    try:
        from data.universes import available as available_universes  # type: ignore[import-untyped]
        names = available_universes()
    except Exception:
        names = sorted(VALID_UNIVERSES - {"custom"})
    return {"universes": names}


# ---------------------------------------------------------------------------
# Section 2: Single Stock Analysis
# ---------------------------------------------------------------------------

analysis_router = APIRouter(prefix="/analysis", tags=["analysis"])


def _sanitize_float(v: Any) -> Any:
    """Replace NaN/Inf with None and convert pandas/numpy types to plain Python."""
    import numpy as np
    import pandas as pd

    if isinstance(v, (pd.Series, pd.Index)):
        return [_sanitize_float(x) for x in v.tolist()]
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        val = float(v)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(v, np.ndarray):
        return [_sanitize_float(x) for x in v.tolist()]
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, dict):
        return {str(k): _sanitize_float(val) for k, val in v.items()}
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
    objective = TRADE_MODE_OBJECTIVES.get(trade_mode)
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
        "overall_raw": composite_raw.get("overall_raw"),
        "breakdown": composite_raw.get("breakdown", {}),
        "n_scored": composite_raw.get("n_scored", 0),
        "weights_used": composite_raw.get("weights_used", {}),
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
    ai: bool = Query(False, description="Include LLM qualitative analysis"),
    user: User = Depends(get_current_user),
):
    """Run full technical analysis on a single ticker.

    Uses the engine's Analyzer to compute indicators, patterns, support/resistance,
    composite scores, and regime classification.  The trade_mode from the user's
    profile selects the appropriate configuration objective (swing_trade, long_term).

    Pass ``ai=true`` to also generate an LLM qualitative analysis (requires an
    API key for the user's configured LLM provider).
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

    # Optional LLM analysis (opt-in to control API costs)
    if ai:
        try:
            from app.services.llm import generate_analysis
            ai_text = await generate_analysis(
                result,
                user.llm_provider,
                api_key=user.llm_api_key,
                model=user.llm_model,
            )
            result["ai_analysis"] = ai_text
        except Exception as exc:
            logger.warning("LLM analysis failed for %s: %s", ticker, exc)
            err_str = str(exc)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "rate" in err_str.lower():
                result["ai_analysis"] = (
                    "AI analysis rate-limited — please wait a minute and try again. "
                    "Free-tier APIs have per-minute request limits."
                )
            elif "API key" in err_str or "api_key" in err_str.lower():
                result["ai_analysis"] = (
                    "AI analysis unavailable — API key not configured. "
                    "Go to Settings to add your API key, or set the "
                    "environment variable on the server."
                )
            else:
                result["ai_analysis"] = f"AI analysis unavailable: {exc}"

    return result


# ---------------------------------------------------------------------------
# Section 3: Sector & Segments
# ---------------------------------------------------------------------------

sectors_router = APIRouter(prefix="/sectors", tags=["sectors"])


def _run_sector_overview() -> dict:
    """Run sector overview synchronously (called from thread pool)."""
    from app.services.sectors import get_sector_overview

    result = get_sector_overview()
    return {
        "sectors": [
            {
                "etf": s.etf,
                "sector": s.sector,
                "return_1w": s.return_1w,
                "return_1m": s.return_1m,
                "return_3m": s.return_3m,
                "rs_1w": s.rs_1w,
                "rs_1m": s.rs_1m,
                "rs_3m": s.rs_3m,
                "current_price": s.current_price,
                "avg_volume": s.avg_volume,
                "regime": s.regime,
                "regime_confidence": s.regime_confidence,
                "momentum_score": s.momentum_score,
            }
            for s in result.sectors
        ],
        "benchmark_return_1w": result.benchmark_return_1w,
        "benchmark_return_1m": result.benchmark_return_1m,
        "benchmark_return_3m": result.benchmark_return_3m,
        "elapsed_seconds": result.elapsed_seconds,
    }


def _run_sector_detail(sector_name: str) -> dict:
    """Run sector detail synchronously (called from thread pool)."""
    from app.services.sectors import get_sector_detail

    result = get_sector_detail(sector_name)
    return {
        "etf": result.etf,
        "sector": result.sector,
        "return_1w": result.return_1w,
        "return_1m": result.return_1m,
        "return_3m": result.return_3m,
        "rs_1w": result.rs_1w,
        "rs_1m": result.rs_1m,
        "rs_3m": result.rs_3m,
        "regime": result.regime,
        "regime_confidence": result.regime_confidence,
        "momentum_score": result.momentum_score,
        "top_movers": [
            {
                "ticker": m.ticker,
                "name": m.name,
                "return_1m": m.return_1m,
                "current_price": m.current_price,
            }
            for m in result.top_movers
        ],
        "worst_movers": [
            {
                "ticker": m.ticker,
                "name": m.name,
                "return_1m": m.return_1m,
                "current_price": m.current_price,
            }
            for m in result.worst_movers
        ],
        "elapsed_seconds": result.elapsed_seconds,
    }


@sectors_router.get("/overview", response_model=SectorOverviewResponse)
async def sector_overview(user: User = Depends(get_current_user)):
    """Sector heatmap: momentum scores, relative strength, regime for all 11 sectors."""
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(_sector_pool, _run_sector_overview)
    except Exception as exc:
        logger.exception("Sector overview failed")
        raise HTTPException(500, f"Sector overview failed: {exc}") from exc
    return result


from app.services.sectors import SECTOR_ETF_MAP  # noqa: E402

VALID_SECTORS = set(SECTOR_ETF_MAP.values())


@sectors_router.get("/detail/{sector_name}", response_model=SectorDetailResponse)
async def sector_detail(sector_name: str, user: User = Depends(get_current_user)):
    """Drill-down for a single sector: returns, RS, regime, top/worst movers."""
    # URL-decode sector name (spaces come as %20)
    import urllib.parse
    sector_decoded = urllib.parse.unquote(sector_name)

    if sector_decoded not in VALID_SECTORS:
        raise HTTPException(
            400,
            f"Invalid sector '{sector_decoded}'. Must be one of: {sorted(VALID_SECTORS)}",
        )

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            _sector_pool, _run_sector_detail, sector_decoded,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("Sector detail failed for %s", sector_decoded)
        raise HTTPException(500, f"Sector detail failed: {exc}") from exc
    return result


@sectors_router.get("/list")
async def list_sectors(user: User = Depends(get_current_user)):
    """List available sector names."""
    return {"sectors": sorted(VALID_SECTORS)}


def _run_refresh_holdings(sector_name: str) -> dict:
    """Refresh sector holdings from yfinance (called from thread pool)."""
    from app.services.sectors import refresh_holdings_from_yfinance, get_sector_detail

    holdings = refresh_holdings_from_yfinance(sector_name)
    # Return the full detail with refreshed holdings
    result = get_sector_detail(sector_name)
    return {
        "etf": result.etf,
        "sector": result.sector,
        "return_1w": result.return_1w,
        "return_1m": result.return_1m,
        "return_3m": result.return_3m,
        "rs_1w": result.rs_1w,
        "rs_1m": result.rs_1m,
        "rs_3m": result.rs_3m,
        "regime": result.regime,
        "regime_confidence": result.regime_confidence,
        "momentum_score": result.momentum_score,
        "top_movers": [
            {"ticker": m.ticker, "name": m.name, "return_1m": m.return_1m, "current_price": m.current_price}
            for m in result.top_movers
        ],
        "worst_movers": [
            {"ticker": m.ticker, "name": m.name, "return_1m": m.return_1m, "current_price": m.current_price}
            for m in result.worst_movers
        ],
        "elapsed_seconds": result.elapsed_seconds,
        "holdings_refreshed": True,
        "holdings_count": len(holdings),
    }


@sectors_router.post("/refresh-holdings/{sector_name}", response_model=SectorDetailResponse)
async def refresh_sector_holdings(sector_name: str, user: User = Depends(get_current_user)):
    """Refresh sector holdings from yfinance and return updated detail.

    Dynamically fetches top ETF holdings from yfinance, falling back to
    the static list if the API is unavailable or slow.
    """
    import urllib.parse
    sector_decoded = urllib.parse.unquote(sector_name)

    if sector_decoded not in VALID_SECTORS:
        raise HTTPException(
            400,
            f"Invalid sector '{sector_decoded}'. Must be one of: {sorted(VALID_SECTORS)}",
        )

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            _sector_pool, _run_refresh_holdings, sector_decoded,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("Refresh holdings failed for %s", sector_decoded)
        raise HTTPException(500, f"Refresh holdings failed: {exc}") from exc
    return result


@sectors_router.get("/holdings/{sector_name}", response_model=SectorHoldingsResponse)
async def get_holdings(sector_name: str, user: User = Depends(get_current_user)):
    """Get the current holdings list for a sector.

    Returns configured holdings if an override exists, otherwise built-in defaults.
    """
    import urllib.parse
    sector_decoded = urllib.parse.unquote(sector_name)

    if sector_decoded not in VALID_SECTORS:
        raise HTTPException(
            400,
            f"Invalid sector '{sector_decoded}'. Must be one of: {sorted(VALID_SECTORS)}",
        )

    from app.services.holdings_config import get_sector_holdings
    from app.services.sectors import SECTOR_HOLDINGS

    override = get_sector_holdings(sector_decoded)
    if override:
        return {
            "sector": sector_decoded,
            "holdings": [{"ticker": h[0], "name": h[1]} for h in override],
            "source": "configured",
        }
    # Fall back to built-in defaults
    defaults = SECTOR_HOLDINGS.get(sector_decoded, [])
    return {
        "sector": sector_decoded,
        "holdings": [{"ticker": h[0], "name": h[1]} for h in defaults],
        "source": "default",
    }


@sectors_router.put("/holdings/{sector_name}", response_model=SectorHoldingsResponse)
async def update_holdings(
    sector_name: str,
    req: SectorHoldingsUpdateRequest,
    user: User = Depends(get_current_user),
):
    """Update holdings for a sector (power user only).

    Saves a custom holdings list that overrides the built-in defaults.
    """
    import urllib.parse
    sector_decoded = urllib.parse.unquote(sector_name)

    if sector_decoded not in VALID_SECTORS:
        raise HTTPException(
            400,
            f"Invalid sector '{sector_decoded}'. Must be one of: {sorted(VALID_SECTORS)}",
        )

    if user.user_mode != "power_user":
        raise HTTPException(403, "Only power users can edit sector holdings")

    if not req.holdings:
        raise HTTPException(400, "Holdings list cannot be empty")

    from app.services.holdings_config import set_sector_holdings

    holdings_list = [[h.ticker.upper().strip(), h.name.strip()] for h in req.holdings]
    set_sector_holdings(sector_decoded, holdings_list)

    return {
        "sector": sector_decoded,
        "holdings": [{"ticker": h[0], "name": h[1]} for h in holdings_list],
        "source": "configured",
    }


@sectors_router.delete("/holdings/{sector_name}", response_model=SectorHoldingsResponse)
async def reset_holdings(sector_name: str, user: User = Depends(get_current_user)):
    """Reset holdings for a sector back to built-in defaults (power user only)."""
    import urllib.parse
    sector_decoded = urllib.parse.unquote(sector_name)

    if sector_decoded not in VALID_SECTORS:
        raise HTTPException(
            400,
            f"Invalid sector '{sector_decoded}'. Must be one of: {sorted(VALID_SECTORS)}",
        )

    if user.user_mode != "power_user":
        raise HTTPException(403, "Only power users can reset sector holdings")

    from app.services.holdings_config import reset_sector_holdings
    from app.services.sectors import SECTOR_HOLDINGS

    reset_sector_holdings(sector_decoded)

    defaults = SECTOR_HOLDINGS.get(sector_decoded, [])
    return {
        "sector": sector_decoded,
        "holdings": [{"ticker": h[0], "name": h[1]} for h in defaults],
        "source": "default",
    }


# ---------------------------------------------------------------------------
# Section 4: Strategy Lab
# ---------------------------------------------------------------------------

strategy_router = APIRouter(prefix="/strategy", tags=["strategy"])

# Separate pool for backtests (can run 30s+ for long periods)
_backtest_pool = ThreadPoolExecutor(max_workers=1)


def _resolve_tax_params(user: "User", trade_mode: str) -> tuple[float, str]:
    """Compute marginal rate and resolve tax treatment from user settings.

    Returns (tax_marginal_rate, tax_treatment).  If the user hasn't
    configured tax (no province or zero income), returns (0.0, "")
    which disables tax in the engine.

    For ``tax_treatment == "auto"``, we resolve to a concrete treatment
    using a trade-mode heuristic (since we don't have backtest stats yet):
      - day_trade  → business_income
      - swing      → capital_gains
      - position   → capital_gains
    """
    province = getattr(user, "tax_province", None)
    income = getattr(user, "tax_annual_income", 0.0)
    treatment = getattr(user, "tax_treatment", "auto")

    if not province or income <= 0:
        return 0.0, ""

    from app.services.tax_calculator import get_combined_marginal_rate, VALID_PROVINCES

    if province not in VALID_PROVINCES:
        return 0.0, ""

    marginal_rate = get_combined_marginal_rate(income, province)

    # Resolve "auto" using trade-mode heuristic
    if treatment == "auto":
        if trade_mode == "day_trade":
            resolved = "business_income"
        else:
            resolved = "capital_gains"
    elif treatment in ("capital_gains", "business_income"):
        resolved = treatment
    else:
        resolved = "capital_gains"

    return marginal_rate, resolved


def _run_backtest(
    ticker: str,
    trade_mode: str,
    period: str,
    interval: str,
    start: str | None,
    end: str | None,
    initial_cash: float,
    commission_pct: float,
    slippage_pct: float,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
    train_years: int,
    test_years: int,
    max_windows: int,
    tax_marginal_rate: float = 0.0,
    tax_treatment: str = "",
) -> dict:
    """Run unified backtest with walk-forward (called from thread pool)."""
    from app.services.backtest import run_backtest  # late import

    return run_backtest(
        ticker=ticker,
        trade_mode=trade_mode,
        period=period,
        interval=interval,
        start=start,
        end=end,
        initial_cash=initial_cash,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        train_years=train_years,
        test_years=test_years,
        max_windows=max_windows,
        tax_marginal_rate=tax_marginal_rate,
        tax_treatment=tax_treatment,
    )


@strategy_router.post("/backtest", response_model=BacktestResponse)
async def run_strategy_backtest(
    req: BacktestRequest,
    user: User = Depends(get_current_user),
):
    """Run a single-ticker backtest with the score-based strategy.

    Uses the user's trade mode to select the appropriate objective preset.
    Cost model parameters (commission, slippage, stop loss, take profit)
    can be overridden per request.
    """
    valid_periods = {"6mo", "1y", "2y", "5y", "10y", "max"}
    valid_intervals = {"1d"}

    # Allow period to be skipped when start/end are provided
    if not req.start and req.period not in valid_periods:
        raise HTTPException(
            400,
            f"Invalid period '{req.period}'. Must be one of: {sorted(valid_periods)}",
        )
    if req.interval not in valid_intervals:
        raise HTTPException(
            400,
            f"Invalid interval '{req.interval}'. Must be one of: {sorted(valid_intervals)}",
        )

    tax_marginal_rate, tax_treatment = _resolve_tax_params(user, user.trade_mode)

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            _backtest_pool,
            _run_backtest,
            req.ticker.upper(),
            user.trade_mode,
            req.period,
            req.interval,
            req.start,
            req.end,
            req.initial_cash,
            req.commission_pct,
            req.slippage_pct,
            req.stop_loss_pct,
            req.take_profit_pct,
            req.train_years,
            req.test_years,
            req.max_windows,
            tax_marginal_rate,
            tax_treatment,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("Backtest failed for %s", req.ticker)
        raise HTTPException(500, f"Backtest failed: {exc}") from exc

    return result


@strategy_router.get("/")
async def strategy_lab_status(user: User = Depends(get_current_user)):
    return {
        "section": "Strategy Lab",
        "status": "active",
        "description": "Backtester, walk-forward testing, auto-tuner, and strategy library.",
    }


@strategy_router.get("/library", response_model=StrategyListResponse)
async def list_strategies_endpoint(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all strategies for the current user (including built-in presets)."""
    from app.services.strategy_library import list_strategies  # late import

    strategies = await list_strategies(db, user.id)
    return {"strategies": strategies}


@strategy_router.get("/library/{strategy_id}", response_model=StrategyResponse)
async def get_strategy_endpoint(
    strategy_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a single strategy by ID."""
    from app.services.strategy_library import get_strategy

    result = await get_strategy(db, user.id, strategy_id)
    if not result:
        raise HTTPException(404, "Strategy not found")
    return result


@strategy_router.post("/library", response_model=StrategyResponse)
async def create_strategy_endpoint(
    req: StrategyCreateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Save a new strategy."""
    from app.services.strategy_library import create_strategy

    if not req.name or not req.name.strip():
        raise HTTPException(400, "Strategy name is required")
    result = await create_strategy(db, user.id, req.model_dump())
    return result


@strategy_router.patch("/library/{strategy_id}", response_model=StrategyResponse)
async def update_strategy_endpoint(
    strategy_id: int,
    req: StrategyUpdateRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update an existing strategy.  Presets can only have is_active toggled."""
    from app.services.strategy_library import update_strategy

    data = {k: v for k, v in req.model_dump().items() if v is not None}
    result = await update_strategy(db, user.id, strategy_id, data)
    if not result:
        raise HTTPException(404, "Strategy not found or cannot be modified")
    return result


@strategy_router.delete("/library/{strategy_id}")
async def delete_strategy_endpoint(
    strategy_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a user strategy.  Presets cannot be deleted."""
    from app.services.strategy_library import delete_strategy

    success = await delete_strategy(db, user.id, strategy_id)
    if not success:
        raise HTTPException(404, "Strategy not found or is a built-in preset")
    return {"status": "deleted", "id": strategy_id}


@strategy_router.get("/library/{strategy_id}/export", response_model=StrategyExportResponse)
async def export_strategy_endpoint(
    strategy_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Export a strategy as a portable JSON document."""
    from app.services.strategy_library import export_strategy

    result = await export_strategy(db, user.id, strategy_id)
    if not result:
        raise HTTPException(404, "Strategy not found")
    return result


@strategy_router.post("/library/import", response_model=StrategyResponse)
async def import_strategy_endpoint(
    req: StrategyImportRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Import a strategy from a JSON document."""
    from app.services.strategy_library import import_strategy

    if not req.name or not req.name.strip():
        raise HTTPException(400, "Strategy name is required")
    result = await import_strategy(db, user.id, req.model_dump())
    return result


# Walk-forward pool (multiple windows = very long running)
_walk_forward_pool = ThreadPoolExecutor(max_workers=1)


def _run_walk_forward(
    ticker: str,
    trade_mode: str,
    train_years: int,
    test_years: int,
    max_windows: int,
    tax_marginal_rate: float = 0.0,
    tax_treatment: str = "",
) -> dict:
    """Run walk-forward testing synchronously (called from thread pool)."""
    from app.services.walk_forward import run_walk_forward  # late import

    return run_walk_forward(
        ticker=ticker,
        trade_mode=trade_mode,
        train_years=train_years,
        test_years=test_years,
        max_windows=max_windows,
        tax_marginal_rate=tax_marginal_rate,
        tax_treatment=tax_treatment,
    )


@strategy_router.post("/walk-forward", response_model=WalkForwardResponse)
async def run_walk_forward_test(
    req: WalkForwardRequest,
    user: User = Depends(get_current_user),
):
    """Run walk-forward (rolling out-of-sample) testing on a ticker.

    Splits history into rolling train/test windows and backtests each
    window independently.  Returns per-window metrics and an aggregate
    stability score.
    """
    if req.train_years < 1 or req.train_years > 20:
        raise HTTPException(400, "train_years must be between 1 and 20")
    if req.test_years < 1 or req.test_years > 5:
        raise HTTPException(400, "test_years must be between 1 and 5")
    if req.max_windows < 1 or req.max_windows > 20:
        raise HTTPException(400, "max_windows must be between 1 and 20")

    tax_marginal_rate, tax_treatment = _resolve_tax_params(user, user.trade_mode)

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            _walk_forward_pool,
            _run_walk_forward,
            req.ticker.upper(),
            user.trade_mode,
            req.train_years,
            req.test_years,
            req.max_windows,
            tax_marginal_rate,
            tax_treatment,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("Walk-forward failed for %s", req.ticker)
        raise HTTPException(500, f"Walk-forward test failed: {exc}") from exc

    return result


# Auto-tuner pool (very long running — runs N walk-forward trials)
_auto_tune_pool = ThreadPoolExecutor(max_workers=1)


def _run_auto_tune(
    ticker: str | None,
    tickers: list[str] | None,
    sector: str | None,
    trade_mode: str,
    objective: str,
    n_trials: int,
    train_years: int,
    test_years: int,
    max_windows: int,
    tax_marginal_rate: float = 0.0,
    tax_treatment: str = "",
) -> dict:
    """Run auto-tuner synchronously (called from thread pool)."""
    from app.services.auto_tuner import run_auto_tune  # late import

    return run_auto_tune(
        ticker=ticker,
        tickers=tickers,
        sector=sector,
        trade_mode=trade_mode,
        objective=objective,
        n_trials=n_trials,
        train_years=train_years,
        test_years=test_years,
        max_windows=max_windows,
        tax_marginal_rate=tax_marginal_rate,
        tax_treatment=tax_treatment,
    )


@strategy_router.post("/auto-tune", response_model=AutoTuneResponse)
async def run_auto_tuner(
    req: AutoTuneRequest,
    user: User = Depends(get_current_user),
):
    """Run Bayesian parameter optimisation on a ticker.

    Each trial runs walk-forward testing to validate parameter sets
    out-of-sample.  Returns the best parameter set, a comparison
    against the baseline (default params), and sensitivity analysis
    for power user mode.
    """
    # Validate that exactly one of ticker, tickers, or sector is provided
    sources = sum([
        req.ticker is not None and req.ticker.strip() != "",
        req.tickers is not None and len(req.tickers) > 0,
        req.sector is not None and req.sector.strip() != "",
    ])
    if sources == 0:
        raise HTTPException(400, "Must provide one of: ticker, tickers, or sector")
    if sources > 1:
        raise HTTPException(400, "Provide only one of: ticker, tickers, or sector")

    # Validate sector name
    if req.sector:
        if req.sector not in VALID_SECTORS:
            raise HTTPException(
                400,
                f"Invalid sector '{req.sector}'. Must be one of: {sorted(VALID_SECTORS)}",
            )

    if req.objective not in VALID_TUNER_OBJECTIVES:
        raise HTTPException(
            400,
            f"Invalid objective '{req.objective}'. "
            f"Must be one of: {sorted(VALID_TUNER_OBJECTIVES)}",
        )
    if req.n_trials < 5 or req.n_trials > 100:
        raise HTTPException(400, "n_trials must be between 5 and 100")
    if req.train_years < 1 or req.train_years > 10:
        raise HTTPException(400, "train_years must be between 1 and 10")
    if req.test_years < 1 or req.test_years > 5:
        raise HTTPException(400, "test_years must be between 1 and 5")
    if req.max_windows < 2 or req.max_windows > 10:
        raise HTTPException(400, "max_windows must be between 2 and 10")

    # Prepare params for the thread pool call
    ticker_val = req.ticker.upper() if req.ticker else None
    tickers_val = [t.upper() for t in req.tickers] if req.tickers else None
    sector_val = req.sector if req.sector else None
    label = sector_val or (", ".join(tickers_val[:3]) if tickers_val else ticker_val)

    tax_marginal_rate, tax_treatment = _resolve_tax_params(user, user.trade_mode)

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            _auto_tune_pool,
            _run_auto_tune,
            ticker_val,
            tickers_val,
            sector_val,
            user.trade_mode,
            req.objective,
            req.n_trials,
            req.train_years,
            req.test_years,
            req.max_windows,
            tax_marginal_rate,
            tax_treatment,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("Auto-tune failed for %s", label)
        raise HTTPException(500, f"Auto-tune failed: {exc}") from exc

    return result


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
        },
        "tax_settings": {
            "tax_province": user.tax_province,
            "tax_annual_income": user.tax_annual_income,
            "tax_treatment": user.tax_treatment,
        },
        "data_providers": {
            "yahoo_finance": {"enabled": True, "status": "active"},
            "polygon": {"enabled": False, "status": "no_api_key"},
            "alpha_vantage": {"enabled": False, "status": "no_api_key"},
        },
    }
