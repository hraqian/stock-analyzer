"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    username: str
    trade_mode: str
    user_mode: str
    starting_capital: float
    risk_tolerance: str
    commission_per_trade: float
    spread_pct: float
    slippage_pct: float
    tax_rate_short_term: float
    tax_rate_long_term: float

    model_config = {"from_attributes": True}


class UserUpdate(BaseModel):
    trade_mode: str | None = None
    user_mode: str | None = None
    starting_capital: float | None = None
    risk_tolerance: str | None = None
    commission_per_trade: float | None = None
    spread_pct: float | None = None
    slippage_pct: float | None = None
    tax_rate_short_term: float | None = None
    tax_rate_long_term: float | None = None


class RegisterRequest(BaseModel):
    username: str
    password: str


# ---------------------------------------------------------------------------
# Trade mode
# ---------------------------------------------------------------------------

VALID_TRADE_MODES = {"swing", "day_trade", "long_term"}
VALID_USER_MODES = {"normal", "power_user"}
VALID_RISK_TOLERANCES = {"conservative", "moderate", "aggressive"}


# ---------------------------------------------------------------------------
# Data provider
# ---------------------------------------------------------------------------

class HistoricalDataRequest(BaseModel):
    ticker: str
    start: str          # ISO date string, e.g. "2020-01-01"
    end: str | None = None
    interval: str = "1d"  # 1d, 1wk, 1mo


class LivePriceResponse(BaseModel):
    ticker: str
    price: float
    currency: str = "USD"
    timestamp: str


# ---------------------------------------------------------------------------
# Sector / Segments
# ---------------------------------------------------------------------------

class SectorInfo(BaseModel):
    ticker: str
    sector: str
    industry: str
    industry_group: str = ""


class SectorOverview(BaseModel):
    sector: str
    ticker_count: int
    avg_momentum: float
    top_tickers: list[str]


# ---------------------------------------------------------------------------
# Single Stock Analysis (Phase 2)
# ---------------------------------------------------------------------------

class IndicatorResultSchema(BaseModel):
    """One indicator's output (mirrors engine IndicatorResult)."""
    name: str
    config_key: str
    score: float
    values: dict = {}
    display: dict = {}
    error: str | None = None


class PatternResultSchema(BaseModel):
    """One pattern detector's output (mirrors engine PatternResult)."""
    name: str
    config_key: str
    score: float
    values: dict = {}
    display: dict = {}
    error: str | None = None


class SRLevelSchema(BaseModel):
    """A support or resistance level."""
    price: float
    level_type: str
    source: str
    touches: int = 1
    label: str = ""


class RegimeMetricsSchema(BaseModel):
    """Raw metrics from regime classification."""
    adx: float = 0.0
    rolling_adx_mean: float = 0.0
    total_return: float = 0.0
    pct_above_ma: float = 50.0
    atr_pct: float = 0.0
    bb_width: float = 0.0
    bb_width_percentile: float = 50.0
    price_ma_distance: float = 0.0
    direction_changes: float = 0.0
    trend_direction: str = "neutral"


class RegimeSchema(BaseModel):
    """Regime assessment result."""
    regime: str           # e.g. "strong_trend"
    confidence: float
    label: str
    description: str
    metrics: RegimeMetricsSchema
    reasons: list[str] = []
    regime_scores: dict[str, float] = {}
    sub_type: str | None = None
    sub_type_label: str = ""
    sub_type_description: str = ""


class OHLCVBar(BaseModel):
    """Single OHLCV bar for charting."""
    date: str          # ISO date string
    open: float
    high: float
    low: float
    close: float
    volume: float


class CompositeScoreSchema(BaseModel):
    """Composite score breakdown."""
    overall: float
    overall_raw: float | None = None
    breakdown: dict[str, float] = {}
    n_scored: int = 0
    weights_used: dict[str, float] = {}
    trend_score: float | None = None
    contrarian_score: float | None = None
    neutral_score: float | None = None
    dominant_group: str | None = None


class AnalysisResponse(BaseModel):
    """Full analysis response for a single ticker."""
    ticker: str
    period: str
    info: dict = {}
    price_data: list[OHLCVBar]
    indicators: list[IndicatorResultSchema]
    patterns: list[PatternResultSchema]
    support_levels: list[SRLevelSchema]
    resistance_levels: list[SRLevelSchema]
    composite: CompositeScoreSchema
    pattern_composite: dict = {}
    regime: RegimeSchema | None = None


# ---------------------------------------------------------------------------
# Scanner (stubs — expanded in Phase 2)
# ---------------------------------------------------------------------------

class ScanRequest(BaseModel):
    universe: str = "sp500"   # sp500, nasdaq100, tsx, russell1000, custom
    custom_tickers: list[str] | None = None
    min_volume: int = 1_000_000
    min_price: float = 5.0
    max_atr_ratio: float | None = None


# ---------------------------------------------------------------------------
# Strategy (stubs — expanded in Phase 3)
# ---------------------------------------------------------------------------

class StrategyMeta(BaseModel):
    id: str
    name: str
    trade_mode: str
    description: str
    created_at: str
    backtest_summary: dict | None = None


# ---------------------------------------------------------------------------
# Portfolio Simulation (stubs — expanded in Phase 4)
# ---------------------------------------------------------------------------

class PortfolioSummary(BaseModel):
    cash_balance: float
    total_equity: float
    open_positions: int
    total_pnl: float
    total_pnl_pct: float
