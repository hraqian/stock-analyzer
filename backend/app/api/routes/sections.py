"""Stub routes for the 6 app sections.

Each section gets its own router.  Phase 1 returns placeholder responses;
later phases fill in the real logic.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.routes.auth import get_current_user
from app.models.user import User

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


@analysis_router.get("/{ticker}")
async def analyze_stock(ticker: str, user: User = Depends(get_current_user)):
    return {
        "section": "Single Stock Analysis",
        "ticker": ticker.upper(),
        "status": "coming_in_phase_2",
        "description": "Deep-dive analysis with indicators, patterns, regime detection, and AI commentary.",
    }


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
