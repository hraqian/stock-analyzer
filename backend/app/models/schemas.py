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


class SectorMomentumSchema(BaseModel):
    """One sector's momentum data for the heatmap."""
    etf: str
    sector: str
    return_1w: float = 0.0
    return_1m: float = 0.0
    return_3m: float = 0.0
    rs_1w: float = 0.0
    rs_1m: float = 0.0
    rs_3m: float = 0.0
    current_price: float = 0.0
    avg_volume: float = 0.0
    regime: str = ""
    regime_confidence: float = 0.0
    momentum_score: float = 0.0


class SectorOverviewResponse(BaseModel):
    """Full sector heatmap response."""
    sectors: list[SectorMomentumSchema]
    benchmark_return_1w: float = 0.0
    benchmark_return_1m: float = 0.0
    benchmark_return_3m: float = 0.0
    elapsed_seconds: float = 0.0


class SectorTopMoverSchema(BaseModel):
    """A top or worst mover within a sector."""
    ticker: str
    name: str
    return_1m: float = 0.0
    current_price: float = 0.0


class SectorDetailResponse(BaseModel):
    """Sector drill-down response."""
    etf: str
    sector: str
    return_1w: float = 0.0
    return_1m: float = 0.0
    return_3m: float = 0.0
    rs_1w: float = 0.0
    rs_1m: float = 0.0
    rs_3m: float = 0.0
    regime: str = ""
    regime_confidence: float = 0.0
    momentum_score: float = 0.0
    top_movers: list[SectorTopMoverSchema] = []
    worst_movers: list[SectorTopMoverSchema] = []
    elapsed_seconds: float = 0.0


# Legacy alias kept for backward compat
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
# Scanner (Phase 2)
# ---------------------------------------------------------------------------

VALID_PRESETS = {"breakout", "pullback", "reversal", "dividend"}

VALID_UNIVERSES = {
    "dow30", "nasdaq100", "sp500", "russell1000", "russell2000",
    "tsx60", "tsx_composite", "sector_etfs", "crypto20", "bond_etfs",
    "us_dividend", "ca_dividend_etfs", "custom",
}


class ScanRequest(BaseModel):
    universe: str = "sp500"
    custom_tickers: list[str] | None = None
    preset: str = "breakout"           # breakout, pullback, reversal, dividend
    period: str = "6mo"                # data lookback
    min_volume: int = 1_000_000
    min_price: float = 5.0
    max_atr_ratio: float | None = None
    top_n: int = 50


class ScannerResultSchema(BaseModel):
    """One row in the scanner output table."""
    rank: int
    ticker: str
    signal: str
    score: float
    confidence: float
    pattern: str
    regime: str
    sector: str
    breakdown: dict[str, float] = {}
    volume: float = 0.0
    price: float = 0.0
    atr_ratio: float = 0.0


class ScanResponse(BaseModel):
    """Full scanner response."""
    preset: str
    universe: str
    total_tickers: int
    tickers_with_data: int
    tickers_passing_filters: int
    elapsed_seconds: float
    results: list[ScannerResultSchema]


class UniverseListResponse(BaseModel):
    """Available universes."""
    universes: list[str]


# ---------------------------------------------------------------------------
# Strategy — Backtest (Phase 3)
# ---------------------------------------------------------------------------

class BacktestRequest(BaseModel):
    """POST body for /api/strategy/backtest."""
    ticker: str
    period: str = "2y"                      # e.g. "1y", "2y", "5y"
    interval: str = "1d"                    # "1d" only for now
    start: str | None = None                # YYYY-MM-DD, overrides period
    end: str | None = None                  # YYYY-MM-DD
    initial_cash: float = 100_000.0
    commission_pct: float = 0.0
    slippage_pct: float = 0.001
    stop_loss_pct: float | None = None      # None → use config default
    take_profit_pct: float | None = None    # None → use config default


class BacktestTradeSchema(BaseModel):
    """A single round-trip trade from the backtest."""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    quantity: float
    side: str                               # "long" or "short"
    pnl: float
    pnl_pct: float
    exit_reason: str = ""
    entry_reason: str = ""
    bars_held: int = 0


class EquityPointSchema(BaseModel):
    """One point on the equity curve."""
    date: str
    equity: float


class BacktestRegimeSchema(BaseModel):
    """Simplified regime info for backtest results."""
    regime: str
    confidence: float
    label: str
    description: str


class BacktestResponse(BaseModel):
    """Full backtest result returned by POST /api/strategy/backtest."""
    ticker: str
    period: str
    strategy_name: str
    initial_cash: float
    final_equity: float
    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate_pct: float
    total_trades: int
    profit_factor: float
    avg_trade_pnl_pct: float
    best_trade_pnl_pct: float
    worst_trade_pnl_pct: float
    avg_bars_held: float
    trades: list[BacktestTradeSchema]
    equity_curve: list[EquityPointSchema]
    regime: BacktestRegimeSchema | None = None
    warmup_bars: int = 0


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
