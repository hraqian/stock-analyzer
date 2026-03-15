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
    # Canadian tax settings
    tax_province: str | None = None
    tax_annual_income: float = 0.0
    tax_treatment: str = "auto"

    model_config = {"from_attributes": True}


class UserUpdate(BaseModel):
    trade_mode: str | None = None
    user_mode: str | None = None
    starting_capital: float | None = None
    risk_tolerance: str | None = None
    commission_per_trade: float | None = None
    spread_pct: float | None = None
    slippage_pct: float | None = None
    # Canadian tax settings
    tax_province: str | None = None
    tax_annual_income: float | None = None
    tax_treatment: str | None = None


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


class SectorHoldingItem(BaseModel):
    """A single holding entry: ticker + company name."""
    ticker: str
    name: str


class SectorHoldingsResponse(BaseModel):
    """Holdings configuration for a sector."""
    sector: str
    holdings: list[SectorHoldingItem] = []
    source: str = "default"  # "default" | "configured" | "refreshed"


class SectorHoldingsUpdateRequest(BaseModel):
    """Request to update holdings for a sector (power user only)."""
    holdings: list[SectorHoldingItem]


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
    """POST body for /api/strategy/backtest.

    Walk-forward validation is always performed: the backtest runs across
    multiple rolling train/test windows for robustness analysis.
    """
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
    # Walk-forward parameters (always run)
    train_years: int = 3
    test_years: int = 1
    max_windows: int = 5


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
    # Tax fields (populated when tax calculation is enabled)
    tax_amount: float = 0.0                 # tax deducted from this trade
    pnl_after_tax: float = 0.0             # pnl minus tax


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
    """Full backtest result returned by POST /api/strategy/backtest.

    Includes detailed results from the most recent walk-forward window
    (equity curve, trades, regime) plus aggregated robustness metrics
    from all walk-forward windows.
    """
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
    # Walk-forward robustness fields
    train_years: int = 3
    test_years: int = 1
    total_windows: int = 0
    windows: list[WalkForwardWindowSchema] = []
    wf_avg_return_pct: float = 0.0
    wf_avg_annualized_return_pct: float = 0.0
    wf_avg_max_drawdown_pct: float = 0.0
    wf_avg_sharpe_ratio: float = 0.0
    wf_avg_win_rate_pct: float = 0.0
    wf_avg_profit_factor: float = 0.0
    wf_worst_return_pct: float = 0.0
    wf_worst_drawdown_pct: float = 0.0
    wf_worst_window_index: int = 0
    wf_return_std_dev: float = 0.0
    stability_score: float = 0.0
    verdict: str = ""
    # Tax-aware fields (populated when user has tax settings configured)
    tax_enabled: bool = False
    tax_treatment_used: str = ""            # "capital_gains" or "business_income"
    tax_province: str = ""
    tax_marginal_rate: float = 0.0          # combined federal + provincial
    total_tax_paid: float = 0.0
    after_tax_return_pct: float = 0.0
    after_tax_final_equity: float = 0.0


# ---------------------------------------------------------------------------
# Strategy — Walk-Forward (Phase 3B)
# ---------------------------------------------------------------------------

class WalkForwardRequest(BaseModel):
    """POST body for /api/strategy/walk-forward."""
    ticker: str
    train_years: int = 5
    test_years: int = 1
    max_windows: int = 10


class WalkForwardWindowSchema(BaseModel):
    """Metrics for one walk-forward window."""
    window_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    error: str | None = None


class WalkForwardResponse(BaseModel):
    """Full walk-forward analysis response."""
    ticker: str
    train_years: int
    test_years: int
    total_windows: int
    windows: list[WalkForwardWindowSchema]
    avg_return_pct: float = 0.0
    avg_annualized_return_pct: float = 0.0
    avg_max_drawdown_pct: float = 0.0
    avg_sharpe_ratio: float = 0.0
    avg_win_rate_pct: float = 0.0
    avg_profit_factor: float = 0.0
    worst_return_pct: float = 0.0
    worst_drawdown_pct: float = 0.0
    worst_window_index: int = 0
    return_std_dev: float = 0.0
    stability_score: float = 0.0
    verdict: str = ""


# ---------------------------------------------------------------------------
# Strategy — Auto-Tuner (Phase 3C)
# ---------------------------------------------------------------------------

VALID_TUNER_OBJECTIVES = {
    "beat_buy_hold",
    "max_return",
    "max_risk_adjusted",
    "min_drawdown",
    "balanced",
}


class AutoTuneRequest(BaseModel):
    """POST body for /api/strategy/auto-tune.

    Supports three modes (provide exactly one of ticker, tickers, or sector):
      - Single ticker: set ``ticker`` (default mode)
      - Sector group: set ``sector`` to one of the 11 GICS sector names
      - Custom group: set ``tickers`` to a list of symbols
    """
    ticker: str | None = None
    tickers: list[str] | None = None
    sector: str | None = None
    objective: str = "balanced"
    n_trials: int = 30
    train_years: int = 3
    test_years: int = 1
    max_windows: int = 3


class SensitivityEntrySchema(BaseModel):
    """How a single parameter affects the objective."""
    param_name: str
    importance: float = 0.0
    best_value: float | str | bool | None = None
    value_range: list = []


class AutoTuneTrialSchema(BaseModel):
    """Summary of a single optimisation trial."""
    trial_number: int
    params: dict = {}
    objective_value: float = 0.0
    avg_return_pct: float = 0.0
    avg_annualized_return_pct: float = 0.0
    avg_max_drawdown_pct: float = 0.0
    avg_sharpe_ratio: float = 0.0
    avg_win_rate_pct: float = 0.0
    avg_profit_factor: float = 0.0
    stability_score: float = 0.0
    total_windows: int = 0


class AutoTuneResponse(BaseModel):
    """Full auto-tuner result."""
    ticker: str
    tickers: list[str] = []
    mode: str = "single"
    sector: str | None = None
    objective: str
    objective_label: str
    n_trials: int
    elapsed_seconds: float = 0.0
    # Best trial
    best_params: dict = {}
    best_objective_value: float = 0.0
    best_avg_return_pct: float = 0.0
    best_avg_annualized_return_pct: float = 0.0
    best_avg_max_drawdown_pct: float = 0.0
    best_avg_sharpe_ratio: float = 0.0
    best_avg_win_rate_pct: float = 0.0
    best_avg_profit_factor: float = 0.0
    best_stability_score: float = 0.0
    # Baseline
    baseline_avg_return_pct: float = 0.0
    baseline_avg_annualized_return_pct: float = 0.0
    baseline_avg_max_drawdown_pct: float = 0.0
    baseline_avg_sharpe_ratio: float = 0.0
    baseline_avg_win_rate_pct: float = 0.0
    baseline_objective_value: float = 0.0
    # Buy-and-hold
    buy_hold_return_pct: float | None = None
    # Sensitivity & trials (power user mode)
    sensitivity: list[SensitivityEntrySchema] = []
    trials: list[AutoTuneTrialSchema] = []
    # Verdict
    verdict: str = ""
    improvement_pct: float = 0.0


# ---------------------------------------------------------------------------
# Strategy Library (Phase 3D)
# ---------------------------------------------------------------------------

class StrategyCreateRequest(BaseModel):
    """POST body for creating/saving a strategy."""
    name: str
    description: str | None = None
    trade_mode: str = "swing"
    ticker: str | None = None
    params: dict = {}
    # Optional metrics (from a backtest or auto-tuner run)
    total_return_pct: float | None = None
    annualized_return_pct: float | None = None
    sharpe_ratio: float | None = None
    max_drawdown_pct: float | None = None
    win_rate_pct: float | None = None
    profit_factor: float | None = None
    stability_score: float | None = None


class StrategyUpdateRequest(BaseModel):
    """PATCH body for updating a strategy."""
    name: str | None = None
    description: str | None = None
    trade_mode: str | None = None
    ticker: str | None = None
    params: dict | None = None
    is_active: bool | None = None
    total_return_pct: float | None = None
    annualized_return_pct: float | None = None
    sharpe_ratio: float | None = None
    max_drawdown_pct: float | None = None
    win_rate_pct: float | None = None
    profit_factor: float | None = None
    stability_score: float | None = None


class StrategyResponse(BaseModel):
    """Single strategy in API responses."""
    id: int
    name: str
    description: str | None = None
    version: int = 1
    is_preset: bool = False
    trade_mode: str = "swing"
    ticker: str | None = None
    params: dict = {}
    total_return_pct: float | None = None
    annualized_return_pct: float | None = None
    sharpe_ratio: float | None = None
    max_drawdown_pct: float | None = None
    win_rate_pct: float | None = None
    profit_factor: float | None = None
    stability_score: float | None = None
    is_active: bool = False
    created_at: str = ""
    updated_at: str = ""


class StrategyListResponse(BaseModel):
    """Response for listing strategies."""
    strategies: list[StrategyResponse] = []


class StrategyExportResponse(BaseModel):
    """Export format for a strategy (JSON-serializable snapshot)."""
    name: str
    description: str | None = None
    version: int = 1
    trade_mode: str = "swing"
    ticker: str | None = None
    params: dict = {}
    metrics: dict = {}


class StrategyImportRequest(BaseModel):
    """Import a strategy from JSON."""
    name: str
    description: str | None = None
    trade_mode: str = "swing"
    ticker: str | None = None
    params: dict = {}
    metrics: dict = {}


# ---------------------------------------------------------------------------
# Portfolio Simulation (stubs — expanded in Phase 4)
# ---------------------------------------------------------------------------

class PortfolioSummary(BaseModel):
    cash_balance: float
    total_equity: float
    open_positions: int
    total_pnl: float
    total_pnl_pct: float
