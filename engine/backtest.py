"""
engine/backtest.py — Bar-by-bar backtesting engine.

Walks through historical OHLCV data, recomputes indicators at each
rebalance interval, generates signals via the plugged-in Strategy, and
simulates trade execution with commission/slippage modelling.

No lookahead bias: each bar only uses data available up to that point.

Usage:
    from engine.backtest import BacktestEngine
    from engine.score_strategy import ScoreBasedStrategy

    engine = BacktestEngine(data_provider=provider, strategy=strategy, cfg=cfg)
    result = engine.run("AAPL", period="2y")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from config import Config
    from data.provider import DataProvider
    from engine.strategy import Strategy

from indicators.registry import IndicatorRegistry
from analysis.scorer import CompositeScorer
from patterns.registry import PatternRegistry
from analysis.pattern_scorer import PatternCompositeScorer
from engine.strategy import Signal, StrategyContext, TradeOrder
from engine.suitability import TradingMode
from engine.regime import RegimeClassifier, RegimeAssessment, RegimeType, RegimeSubType
from analysis.support_resistance import calculate_levels as calc_sr_levels

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_pattern_name(raw: str) -> str:
    """Convert snake_case pattern names to Title Case labels.

    Examples:
        ``"bullish_engulfing"`` → ``"Bullish Engulfing"``
        ``"three_white_soldiers"`` → ``"Three White Soldiers"``
        ``"inside_bar_bullish"`` → ``"Inside Bar Bullish"``
    """
    return raw.replace("_", " ").title()


# Detector-aware strength → human-readable confidence label.
# Each detector uses a different strength scale, so the thresholds differ.
_DEFAULT_STRENGTH_THRESHOLDS: dict[str, list[tuple[float, str]]] = {
    "Candlesticks":    [(0.4, "Weak"), (0.7, "Moderate"), (1.0, "Strong")],
    "Inside/Outside":  [(0.4, "Weak"), (0.6, "Moderate"), (0.8, "Strong")],
    "Gaps":            [(0.5, "Weak"), (1.0, "Moderate"), (1.5, "Strong")],
    "Spikes":          [(0.5, "Weak"), (1.0, "Moderate"), (1.5, "Strong")],
}


def _strength_label(
    strength: float,
    detector: str,
    custom_thresholds: dict[str, list[tuple[float, str]]] | None = None,
) -> str:
    """Return a human-readable confidence label for a pattern strength value.

    Uses detector-aware thresholds because the raw strength scales differ
    between detectors (e.g. candlesticks 0.3–1.3 vs gaps 0–2.0).
    Falls back to the Candlesticks thresholds for unknown detectors.

    *custom_thresholds* overrides the built-in defaults when provided (from
    config).
    """
    table = custom_thresholds if custom_thresholds else _DEFAULT_STRENGTH_THRESHOLDS
    fallback = table.get("Candlesticks", _DEFAULT_STRENGTH_THRESHOLDS["Candlesticks"])
    thresholds = table.get(detector, fallback)
    for upper, label in thresholds:
        if strength <= upper:
            return label
    return "Very Strong"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BacktestTrade:
    """A single completed (round-trip) trade recorded during backtesting."""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    quantity: float
    side: str              # "long" or "short"
    pnl: float             # absolute P&L
    pnl_pct: float         # percentage P&L
    exit_reason: str = ""  # "signal", "stop_loss", "take_profit"
    entry_reason: str = "" # strategy notes at entry (e.g. "ind=6.80 pat=5.20 eff=6.32")
    bars_held: int = 0     # number of bars position was held


@dataclass
class SignificantPattern:
    """A single significant pattern detected during the backtest period.

    Used to build a timeline showing all noteworthy pattern events so
    users can see potential entry/exit opportunities — including ones the
    strategy may have missed.
    """
    date: str              # YYYY-MM-DD (or full timestamp for intraday)
    detector: str          # detector name, e.g. "Candlesticks", "Gaps"
    pattern: str           # specific pattern, e.g. "hammer", "breakaway"
    signal: str            # "bullish", "bearish", or "neutral"
    strength: float        # raw strength / magnitude (detector-specific)
    confidence: str = ""   # human-readable label: Weak / Moderate / Strong / Very Strong
    detail: str = ""       # extra context, e.g. "gap_pct=1.2%" or "z=3.1 Confirmed"


@dataclass
class BacktestResult:
    """Aggregated results from a backtest run."""
    ticker: str
    period: str
    strategy_name: str
    initial_cash: float = 100_000.0
    final_equity: float = 0.0
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    # equity_curve: list of {"date": str, "equity": float}

    # Significant patterns detected during the backtest period
    significant_patterns: list[SignificantPattern] = field(default_factory=list)

    # Market regime classification (from the full data at end of backtest)
    regime: RegimeAssessment | None = None

    # Number of warmup bars at the start of equity_curve (no trades occur)
    warmup_bars: int = 0

    # Performance metrics (populated by _compute_metrics)
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate_pct: float = 0.0
    total_trades: int = 0
    profit_factor: float = 0.0
    avg_trade_pnl_pct: float = 0.0
    best_trade_pnl_pct: float = 0.0
    worst_trade_pnl_pct: float = 0.0
    avg_bars_held: float = 0.0


# ---------------------------------------------------------------------------
# Position tracker (internal)
# ---------------------------------------------------------------------------

@dataclass
class _Position:
    """Tracks an open position during simulation."""
    side: str               # "long" or "short"
    entry_date: str
    entry_price: float
    quantity: float
    bars_held: int = 0
    entry_reason: str = ""  # carried through to BacktestTrade on close
    entry_atr: float = 0.0  # ATR at time of entry (for adaptive stop loss)

    def unrealized_pnl(self, current_price: float) -> float:
        if self.side == "long":
            return (current_price - self.entry_price) * self.quantity
        return (self.entry_price - current_price) * self.quantity

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.entry_price == 0:
            return 0.0
        if self.side == "long":
            return (current_price - self.entry_price) / self.entry_price
        return (self.entry_price - current_price) / self.entry_price


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Runs a Strategy against historical OHLCV data bar-by-bar.

    Design:
      - Fetches the full historical DataFrame once.
      - Skips the first ``warmup_bars`` to allow indicators to stabilize.
      - Every ``rebalance_interval`` bars after warmup, recomputes indicators
        on the trailing data (up to the current bar — no lookahead).
      - Passes a StrategyContext to strategy.on_bar() to get a TradeOrder.
      - Executes the order with slippage + commission modelling.
      - Checks stop-loss / take-profit on every bar (not just rebalance bars).
      - Tracks equity curve and completed trades.
    """

    def __init__(
        self,
        data_provider: "DataProvider",
        strategy: "Strategy",
        cfg: "Config",
        trading_mode: TradingMode = TradingMode.LONG_SHORT,
    ) -> None:
        self._provider = data_provider
        self._strategy = strategy
        self._cfg = cfg
        self._trading_mode = trading_mode

        # Config sections
        bt_cfg = cfg.section("backtest")
        strat_cfg = cfg.section("strategy")

        self._initial_cash: float = float(bt_cfg.get("initial_cash", 100_000))
        self._commission_flat: float = float(bt_cfg.get("commission_per_trade", 0.0))
        self._commission_pct: float = float(bt_cfg.get("commission_pct", 0.0))
        self._commission_mode: str = str(bt_cfg.get("commission_mode", "additive"))
        self._slippage_pct: float = float(bt_cfg.get("slippage_pct", 0.001))
        self._warmup_bars: int = int(bt_cfg.get("warmup_bars", 200))

        self._rebalance_interval: int = int(strat_cfg.get("rebalance_interval", 5))
        self._max_hold_bars: int = int(strat_cfg.get("max_hold_bars", 0))
        self._stop_loss_pct: float = float(strat_cfg.get("stop_loss_pct", 0.05))
        self._take_profit_pct: float = float(strat_cfg.get("take_profit_pct", 0.15))
        self._flatten_eod: bool = bool(strat_cfg.get("flatten_eod", False))

        # ATR-adaptive stop loss
        self._atr_stop_multiplier: float = float(strat_cfg.get("atr_stop_multiplier", 2.5))
        self._atr_stop_period: int = int(strat_cfg.get("atr_stop_period", 14))
        self._atr_stop_enabled: bool = bool(strat_cfg.get("atr_stop_enabled", True))

        # Chandelier Exit stop
        self._chandelier_enabled: bool = bool(strat_cfg.get("chandelier_enabled", False))
        self._chandelier_atr_mult: float = float(strat_cfg.get("chandelier_atr_mult", 3.0))
        self._chandelier_lookback: int = int(strat_cfg.get("chandelier_lookback", 22))

        # Support/resistance-based stop
        self._support_stop_enabled: bool = bool(strat_cfg.get("support_stop_enabled", False))
        self._support_stop_buffer_pct: float = float(strat_cfg.get("support_stop_buffer_pct", 0.01))
        self._support_levels: list[float] = []   # populated at start of run()
        self._resistance_levels: list[float] = []  # populated at start of run()

        # Trend confirmation filter
        self._trend_confirm_enabled: bool = bool(strat_cfg.get("trend_confirm_enabled", True))
        self._trend_confirm_period: int = int(strat_cfg.get("trend_confirm_period", 20))

        # Proportional warmup
        self._max_warmup_ratio: float = float(bt_cfg.get("max_warmup_ratio", 0.5))

        # New configurable backtest parameters
        self._min_warmup_bars: int = int(bt_cfg.get("min_warmup_bars", 20))
        self._min_post_warmup_bars: int = int(bt_cfg.get("min_post_warmup_bars", 10))
        self._trading_days_per_year: int = int(bt_cfg.get("trading_days_per_year", 252))
        self._trading_day_minutes: int = int(bt_cfg.get("trading_day_minutes", 390))
        self._default_score: float = float(bt_cfg.get("default_score", 5.0))
        self._close_on_end_of_data: bool = bool(bt_cfg.get("close_on_end_of_data", True))

        # Strategy policy toggles (read from strategy section)
        self._allow_immediate_reversal: bool = bool(strat_cfg.get("allow_immediate_reversal", True))
        self._trailing_stop_require_profit: bool = bool(strat_cfg.get("trailing_stop_require_profit", True))
        self._disable_tp_in_strong_trend: bool = bool(strat_cfg.get("disable_take_profit_in_strong_trend", True))
        self._trend_confirm_ma_type: str = str(strat_cfg.get("trend_confirm_ma_type", "ema"))

        # Trailing stop high-water mark (reset when position opens/closes)
        self._trailing_high: float = 0.0

        # Strategy profiles (regime-aware auto-selection)
        profiles_cfg = cfg.section("strategy_profiles")
        self._profiles_enabled: bool = bool(profiles_cfg.get("enabled", False))
        self._profile_regime_mapping: dict[str, str] = dict(
            profiles_cfg.get("regime_mapping", {})
        )
        self._profiles: dict[str, dict] = dict(profiles_cfg.get("profiles", {}))
        self._active_profile_name: str | None = None

    # ------------------------------------------------------------------
    # Strategy profiles
    # ------------------------------------------------------------------

    def _select_profile(self, regime: RegimeType | None) -> str | None:
        """Select a strategy profile name based on the detected regime.

        Returns the profile name if one is mapped, otherwise None.
        """
        if not self._profiles_enabled or regime is None:
            return None
        profile_name = self._profile_regime_mapping.get(regime.value)
        if profile_name and profile_name in self._profiles:
            return profile_name
        return None

    def _apply_profile(self, profile_name: str) -> None:
        """Apply a strategy profile's param overrides to the engine.

        Overrides engine-level stop/take-profit/sizing settings and notifies
        the strategy to reinitialize with the new params.
        """
        profile = self._profiles.get(profile_name, {})
        if not profile:
            return

        self._active_profile_name = profile_name
        logger.info("Applying strategy profile '%s': %s", profile_name,
                     profile.get("description", ""))

        # Override engine-level params
        if "stop_loss_pct" in profile:
            self._stop_loss_pct = float(profile["stop_loss_pct"])
        if "take_profit_pct" in profile:
            self._take_profit_pct = float(profile["take_profit_pct"])
        if "rebalance_interval" in profile:
            self._rebalance_interval = int(profile["rebalance_interval"])
        if "atr_stop_multiplier" in profile:
            self._atr_stop_multiplier = float(profile["atr_stop_multiplier"])

        # Forward params to the strategy (score thresholds, sizing, etc.)
        # Build a partial params dict from the profile
        strategy_overrides: dict = {}
        for key in ("score_thresholds", "percent_equity", "position_sizing",
                     "trend_confirm_period", "indicator_weight", "pattern_weight",
                     "combination_mode", "boost_strength", "boost_dead_zone"):
            if key in profile:
                strategy_overrides[key] = profile[key]

        if strategy_overrides and hasattr(self._strategy, "apply_overrides"):
            self._strategy.apply_overrides(strategy_overrides)

    # ------------------------------------------------------------------
    # Interval helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bars_per_day(interval: str, trading_day_minutes: int = 390) -> int:
        """Estimate the number of intraday bars in a trading day.

        Returns 1 for daily intervals. For weekly/monthly, returns a
        fractional-equivalent via the convention that the caller
        multiplies by ``trading_days_per_year`` to get bars-per-year,
        so we return a value that makes that product correct:
          weekly  → 52 / trading_days_per_year  (≈ 0.206)
          monthly → 12 / trading_days_per_year  (≈ 0.048)
        Since the return type is int and we need precision for
        annualization, we instead provide a companion method
        ``_bars_per_year``.
        """
        interval = interval.lower().strip()
        # Map interval string to minutes per bar
        _interval_minutes: dict[str, int] = {
            "1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30,
            "60m": 60, "90m": 90, "1h": 60,
        }
        minutes = _interval_minutes.get(interval)
        if minutes is None:
            return 1  # daily (weekly/monthly handled by _bars_per_year)
        return max(1, math.ceil(trading_day_minutes / minutes))

    def _bars_per_year(self, interval: str) -> float:
        """Return estimated bars per year for the given interval.

        Correctly handles weekly and monthly intervals instead of
        treating them as daily.
        """
        interval_lower = interval.lower().strip()
        if interval_lower in ("1wk", "1w"):
            return 52.0
        if interval_lower in ("1mo", "1m_monthly", "3mo"):
            return 12.0 if interval_lower != "3mo" else 4.0
        bpd = self._bars_per_day(interval, self._trading_day_minutes)
        return float(bpd * self._trading_days_per_year)

    @staticmethod
    def _is_intraday(interval: str) -> bool:
        """Return True if the interval is intraday (sub-daily)."""
        return interval.lower().strip() in {
            "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        ticker: str,
        period: str | None = "2y",
        interval: str = "1d",
        start: str | None = None,
        end: str | None = None,
    ) -> BacktestResult:
        """Run the backtest and return results with performance metrics.

        Args:
            ticker:   Stock symbol.
            period:   Historical period (e.g. "2y", "5y"). Ignored when *start* is set.
            interval: Bar interval (e.g. "1d").
            start:    Start date in YYYY-MM-DD format. Overrides *period*.
            end:      End date in YYYY-MM-DD format (default: today).

        Returns:
            :class:`BacktestResult` with trades, equity curve, and metrics.
        """
        # 1. Fetch all data
        df = self._provider.fetch(ticker, period=period, interval=interval, start=start, end=end)

        # Build a display-friendly period label
        if start:
            period_label = f"{start} → {end or 'today'}"
        else:
            period_label = period or "2y"

        # 1b. Proportional warmup — cap warmup to a fraction of available data
        effective_warmup = self._warmup_bars
        max_warmup = int(len(df) * self._max_warmup_ratio)
        if effective_warmup > max_warmup:
            effective_warmup = max(self._min_warmup_bars, max_warmup)

        min_required = effective_warmup + self._min_post_warmup_bars
        if len(df) < min_required:
            raise ValueError(
                f"Not enough data for backtest. Got {len(df)} bars, "
                f"need at least {min_required} "
                f"(warmup={effective_warmup})."
            )

        # 1c. Pre-compute trend confirmation EMA (full series, no lookahead — EMA[i] uses data up to i)
        trend_ma_series: pd.Series | None = None
        if self._trend_confirm_enabled:
            if self._trend_confirm_ma_type.lower() == "sma":
                trend_ma_series = df["close"].rolling(window=self._trend_confirm_period, min_periods=1).mean()
            else:
                trend_ma_series = df["close"].ewm(span=self._trend_confirm_period, adjust=False).mean()

        # 1d. Pre-compute ATR series for adaptive stop loss
        atr_series: pd.Series | None = None
        if self._atr_stop_enabled:
            high = df["high"]
            low = df["low"]
            prev_close = df["close"].shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr_series = tr.rolling(window=self._atr_stop_period, min_periods=1).mean()

        # 1e. Support/resistance levels — recomputed per rebalance on trailing data.
        # (Config is read here; actual computation happens inside the bar loop.)
        sr_cfg: dict = {}
        if self._support_stop_enabled:
            sr_cfg = self._cfg.section("support_resistance")

        # 1e2. Pre-compute rolling high/low series for Chandelier Exit
        chandelier_high_series: pd.Series | None = None
        chandelier_low_series: pd.Series | None = None
        if self._chandelier_enabled:
            chandelier_high_series = df["high"].rolling(
                window=self._chandelier_lookback, min_periods=1
            ).max()
            chandelier_low_series = df["low"].rolling(
                window=self._chandelier_lookback, min_periods=1
            ).min()

        # 1f. Pre-compute rolling average volume for breakout volume-surge gate
        bt_adapt = self._cfg.section("strategy").get("regime_adaptation", {})
        bt_breakout_cfg = bt_adapt.get("breakout_transition", {})
        avg_vol_window = int(bt_breakout_cfg.get("avg_volume_window", 20))
        avg_volume_series: pd.Series = df["volume"].rolling(
            window=avg_vol_window, min_periods=1
        ).mean()

        # 2. State initialization
        cash: float = self._initial_cash
        position: _Position | None = None
        trades: list[BacktestTrade] = []
        equity_curve: list[dict[str, Any]] = []

        # Last computed scores (reused between rebalance bars)
        last_scores: dict[str, float] = {}
        last_overall: float = self._default_score
        last_pattern_overall: float = self._default_score

        # Regime classification (re-evaluated each rebalance)
        regime_classifier = RegimeClassifier(self._cfg)
        current_regime: RegimeType | None = None
        current_regime_trend: str = "neutral"
        current_regime_total_return: float = 0.0
        current_regime_sub_type: RegimeSubType | None = None

        # Regime sub-type is now recomputed at each rebalance on trailing data
        # (no lookahead).  The initial classification below uses only the first
        # portion of data (up to warmup) for profile selection — NOT the full
        # dataset.  If not enough data for a reliable classification at warmup
        # time, sub_type starts as None and gets set on the first rebalance.
        initial_regime_type: RegimeType | None = None
        initial_assessment: RegimeAssessment | None = None
        try:
            # Use data up to effective_warmup for initial classification
            warmup_df = df.iloc[: effective_warmup] if effective_warmup > 30 else df.iloc[: min(len(df), 60)]
            initial_assessment = regime_classifier.classify(warmup_df)
            current_regime_sub_type = initial_assessment.sub_type
            initial_regime_type = initial_assessment.regime
        except Exception:
            logger.warning("Initial regime classification failed; sub_type left as None", exc_info=True)

        # 1f. Apply strategy profile based on initial regime classification
        if self._profiles_enabled and initial_regime_type is not None:
            profile_name = self._select_profile(initial_regime_type)
            if profile_name:
                self._apply_profile(profile_name)

        # Notify strategy of start
        self._strategy.on_start({"ticker": ticker, "period": period})

        def _record_trade(t: BacktestTrade) -> None:
            """Append trade to list and notify strategy of the close."""
            trades.append(t)
            self._strategy.on_trade_close(t.pnl_pct, t.exit_reason)

        # 3. Bar-by-bar simulation
        bars_since_rebalance = self._rebalance_interval  # force recompute on first eligible bar
        intraday = self._is_intraday(interval)

        for i in range(len(df)):
            row = df.iloc[i]
            date_str = str(df.index[i])[:10]
            bar_timestamp = str(df.index[i])  # full timestamp for display
            close = float(row["close"])
            bar_dict = {
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": close,
                "volume": float(row["volume"]),
            }

            # Track position bars
            if position is not None:
                position.bars_held += 1

            # -- Skip warmup period --
            if i < effective_warmup:
                if position is None:
                    equity = cash
                elif position.side == "long":
                    equity = cash + position.quantity * close
                else:
                    equity = cash - position.quantity * close
                equity_curve.append({"date": date_str, "equity": equity})
                continue

            # -- Check stop-loss / take-profit on every bar --
            if position is not None:
                # Gather Chandelier and ATR data for this bar
                bar_atr = float(atr_series.iloc[i]) if atr_series is not None else 0.0
                bar_chandelier_high = (
                    float(chandelier_high_series.iloc[i])
                    if chandelier_high_series is not None else 0.0
                )
                bar_chandelier_low = (
                    float(chandelier_low_series.iloc[i])
                    if chandelier_low_series is not None else 0.0
                )
                exit_reason = self._check_exit_triggers(
                    position, close, current_regime, current_regime_sub_type,
                    bar_atr=bar_atr,
                    chandelier_high=bar_chandelier_high,
                    chandelier_low=bar_chandelier_low,
                )
                if exit_reason:
                    trade = self._close_position(
                        position, close, date_str, exit_reason
                    )
                    cash += self._trade_proceeds(trade, position)
                    _record_trade(trade)
                    position = None

            # -- Determine if this is the last bar of the trading day (for EOD flattening) --
            eod_bar = False
            if self._flatten_eod and intraday:
                eod_bar = (
                    i == len(df) - 1
                    or str(df.index[i + 1])[:10] != date_str
                )

            # -- Rebalance check: recompute indicators + get signal --
            bars_since_rebalance += 1
            if bars_since_rebalance >= self._rebalance_interval:
                bars_since_rebalance = 0

                # Trailing data up to current bar (inclusive) — no lookahead
                trailing_df = df.iloc[: i + 1].copy()

                last_scores, last_overall, last_pattern_overall = self._compute_scores(trailing_df)

                # Re-evaluate market regime on trailing data
                # Both regime type and sub-type update each rebalance — no
                # forward-looking bias.
                try:
                    regime_assessment = regime_classifier.classify(trailing_df)
                    current_regime = regime_assessment.regime
                    current_regime_trend = regime_assessment.metrics.trend_direction
                    current_regime_total_return = regime_assessment.metrics.total_return
                    current_regime_sub_type = regime_assessment.sub_type
                except Exception:
                    logger.warning(
                        "Regime re-evaluation failed at bar %d; using stale regime",
                        i, exc_info=True,
                    )

                # Recompute S/R levels on trailing data (no lookahead)
                if self._support_stop_enabled:
                    try:
                        sr_price = float(trailing_df["close"].iloc[-1])
                        sr_levels = calc_sr_levels(trailing_df, sr_cfg, sr_price)
                        self._support_levels = [
                            lvl.price for lvl in sr_levels.get("support", [])
                        ]
                        self._resistance_levels = [
                            lvl.price for lvl in sr_levels.get("resistance", [])
                        ]
                    except Exception:
                        logger.debug(
                            "S/R recomputation failed at bar %d; keeping previous levels",
                            i, exc_info=True,
                        )

            # Build context and ask strategy for order
            portfolio_value = cash
            if position is not None:
                if position.side == "long":
                    portfolio_value += position.quantity * close
                else:
                    portfolio_value -= position.quantity * close

            # Trend MA for confirmation filter
            current_trend_ma = 0.0
            if trend_ma_series is not None:
                current_trend_ma = float(trend_ma_series.iloc[i])

            # Current ATR for adaptive stop loss on new entries
            current_atr = 0.0
            if atr_series is not None:
                current_atr = float(atr_series.iloc[i])

            ctx = StrategyContext(
                bar=bar_dict,
                indicators={},       # raw values not needed for score strategy
                scores=last_scores,
                overall_score=last_overall,
                pattern_score=last_pattern_overall,
                position=position.quantity * (1 if position.side == "long" else -1)
                if position
                else 0.0,
                cash=cash,
                portfolio_value=portfolio_value,
                trend_ma=current_trend_ma,
                regime=current_regime,
                regime_sub_type=current_regime_sub_type,
                regime_trend=current_regime_trend,
                regime_total_return=current_regime_total_return,
                metadata={"avg_volume": float(avg_volume_series.iloc[i])},
            )

            order = self._strategy.on_bar(ctx)

            # -- Execute order (respecting trading mode) --
            # On EOD bars with flatten_eod, skip opening new positions
            can_short = self._trading_mode == TradingMode.LONG_SHORT
            can_trade = self._trading_mode != TradingMode.HOLD_ONLY
            signal_reason = f"signal: {order.notes}" if order.notes else "signal"

            if eod_bar:
                # EOD: only allow closing existing positions, not opening new ones
                if can_trade and position is not None:
                    if order.signal == Signal.BUY and position.side == "short":
                        # Close short (but don't open long)
                        trade = self._close_position(position, close, date_str, signal_reason)
                        cash += self._trade_proceeds(trade, position)
                        _record_trade(trade)
                        position = None
                    elif order.signal == Signal.SELL and position.side == "long":
                        # Close long (but don't open short)
                        trade = self._close_position(position, close, date_str, signal_reason)
                        cash += self._trade_proceeds(trade, position)
                        _record_trade(trade)
                        position = None
            elif not can_trade:
                pass  # hold_only: do nothing
            elif order.signal == Signal.BUY and position is None:
                qty = self._cap_quantity_to_cash(order.quantity, close, cash)
                if qty > 0:
                    position, cost = self._open_position(
                        "long", close, qty, date_str, order.notes, current_atr
                    )
                    cash -= cost
            elif order.signal == Signal.SELL and position is None and can_short:
                if order.quantity > 0:
                    position, cost = self._open_position(
                        "short", close, order.quantity, date_str, order.notes, current_atr
                    )
                    cash -= cost  # cost = commission only for short opens
            elif order.signal == Signal.BUY and position is not None and position.side == "short":
                # Close short
                trade = self._close_position(position, close, date_str, signal_reason)
                cash += self._trade_proceeds(trade, position)
                _record_trade(trade)
                # Open long only if immediate reversal is allowed
                if self._allow_immediate_reversal:
                    qty = self._cap_quantity_to_cash(order.quantity, close, cash)
                    if qty > 0:
                        position, cost = self._open_position(
                            "long", close, qty, date_str, order.notes, current_atr
                        )
                        cash -= cost
                    else:
                        position = None
                else:
                    position = None
            elif order.signal == Signal.SELL and position is not None and position.side == "long":
                # Close long
                trade = self._close_position(position, close, date_str, signal_reason)
                cash += self._trade_proceeds(trade, position)
                _record_trade(trade)
                position = None
                # Open short only if trading mode allows and immediate reversal is allowed
                if can_short and self._allow_immediate_reversal:
                    if order.quantity > 0:
                        position, cost = self._open_position(
                            "short", close, order.quantity, date_str, order.notes, current_atr
                        )
                        cash -= cost

            # -- EOD flattening: force-close any remaining position at end of day --
            if eod_bar and position is not None:
                trade = self._close_position(
                    position, close, date_str, "eod_flatten"
                )
                cash += self._trade_proceeds(trade, position)
                _record_trade(trade)
                position = None

            # Record equity
            equity = cash
            if position is not None:
                if position.side == "long":
                    equity += position.quantity * close
                else:
                    equity -= position.quantity * close
            equity_curve.append({"date": date_str, "equity": equity})

        # 4. Close any remaining position at last bar (if configured)
        if self._close_on_end_of_data and position is not None:
            last_close = float(df["close"].iloc[-1])
            last_date = str(df.index[-1])[:10]
            trade = self._close_position(position, last_close, last_date, "end_of_data")
            cash += self._trade_proceeds(trade, position)
            _record_trade(trade)
            position = None
            # Update last equity curve entry to reflect the post-close cash
            # (the entry recorded in the loop included the position's
            # unrealized value, which may differ from realised proceeds
            # due to commission/slippage applied during close).
            if equity_curve:
                equity_curve[-1]["equity"] = cash

        final_equity = cash
        # If position still open (close_on_end_of_data=False), include unrealized value
        if position is not None:
            last_close = float(df["close"].iloc[-1])
            if position.side == "long":
                final_equity += position.quantity * last_close
            else:
                final_equity -= position.quantity * last_close

        # 5. Extract significant patterns from the full post-warmup data
        post_warmup_df = df.iloc[effective_warmup:].copy()
        significant_patterns = self._extract_significant_patterns(post_warmup_df)

        # 5b. Final regime classification on the full dataset
        final_regime: RegimeAssessment | None = None
        try:
            final_regime = regime_classifier.classify(df)
        except Exception:
            logger.warning("Final regime classification failed", exc_info=True)

        # 6. Build result
        result = BacktestResult(
            ticker=ticker.upper(),
            period=period_label,
            strategy_name=self._strategy.name,
            initial_cash=self._initial_cash,
            final_equity=final_equity,
            trades=trades,
            equity_curve=equity_curve,
            significant_patterns=significant_patterns,
            regime=final_regime,
            warmup_bars=effective_warmup,
        )
        self._compute_metrics(result, interval)
        self._strategy.on_end({"total_trades": len(trades)})
        return result

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _calc_commission(self, notional: float) -> float:
        """Compute commission for a single leg (entry OR exit).

        Combines the flat per-trade fee and percentage-of-notional fee
        according to ``commission_mode``:

        - **additive**: ``flat + pct * notional`` — both always apply.
        - **max**: ``max(flat, pct * notional)`` — whichever is greater.

        Set either component to 0 to effectively disable it.
        """
        flat = self._commission_flat
        pct = self._commission_pct * abs(notional)
        if self._commission_mode == "max":
            return max(flat, pct)
        # Default: additive
        return flat + pct

    def _cap_quantity_to_cash(
        self, desired_qty: float, price: float, available_cash: float
    ) -> float:
        """Cap *desired_qty* so the long position cost doesn't exceed cash.

        Returns 0 if the desired quantity is zero/negative or cash is
        insufficient for even a single share.  For shorts, the cash
        constraint doesn't apply the same way (short proceeds add cash),
        so callers should only use this for long entries.

        The commission is notional-dependent (when ``commission_pct > 0``),
        so we solve: ``qty * price + commission(qty * price) <= cash``.
        For additive mode: ``qty * price * (1 + pct) + flat <= cash``.
        For max mode we use the conservative (additive) bound.
        """
        if desired_qty <= 0 or price <= 0:
            return 0.0
        # Solve for max affordable quantity conservatively
        effective_rate = 1.0 + self._commission_pct  # covers both modes conservatively
        max_cost = available_cash - self._commission_flat
        if max_cost <= 0:
            return 0.0
        max_qty = int(max_cost / (price * effective_rate))
        if max_qty <= 0:
            return 0.0
        return min(desired_qty, max_qty)

    def _open_position(
        self,
        side: str,
        price: float,
        quantity: float,
        date: str,
        entry_reason: str = "",
        entry_atr: float = 0.0,
    ) -> tuple[_Position, float]:
        """Open a new position. Returns (position, cash_cost).

        For long:  cash_cost = quantity * fill_price + commission
        For short: cash_cost = -(proceeds - commission) so cash increases by
                  proceeds minus commission when we do ``cash -= cash_cost``.
        """
        fill_price = self._apply_slippage(price, side, opening=True)
        # Reset trailing stop high-water mark for the new position
        self._trailing_high = fill_price
        pos = _Position(
            side=side,
            entry_date=date,
            entry_price=fill_price,
            quantity=quantity,
            entry_reason=entry_reason,
            entry_atr=entry_atr,
        )
        notional = fill_price * quantity
        commission = self._calc_commission(notional)
        if side == "long":
            cost = notional + commission
        else:
            cost = -(notional - commission)
        return pos, cost

    def _close_position(
        self,
        position: _Position,
        price: float,
        date: str,
        reason: str,
    ) -> BacktestTrade:
        """Close *position* and return a BacktestTrade record."""
        fill_price = self._apply_slippage(price, position.side, opening=False)

        if position.side == "long":
            pnl = (fill_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - fill_price) * position.quantity

        # Commissions on both entry and exit (notionals may differ)
        entry_notional = position.entry_price * position.quantity
        exit_notional = fill_price * position.quantity
        pnl -= (self._calc_commission(entry_notional)
                + self._calc_commission(exit_notional))

        entry_cost = position.entry_price * position.quantity
        pnl_pct = pnl / entry_cost if entry_cost != 0 else 0.0

        return BacktestTrade(
            entry_date=position.entry_date,
            exit_date=date,
            entry_price=position.entry_price,
            exit_price=fill_price,
            quantity=position.quantity,
            side=position.side,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            entry_reason=position.entry_reason,
            bars_held=position.bars_held,
        )

    def _trade_proceeds(self, trade: BacktestTrade, position: _Position) -> float:
        """Cash returned when closing a position.

        Long close:  quantity * exit_price - exit_commission
        Short close: -(quantity * exit_price + exit_commission)
        """
        exit_notional = trade.exit_price * trade.quantity
        exit_commission = self._calc_commission(exit_notional)
        if trade.side == "long":
            return exit_notional - exit_commission
        else:
            return -(exit_notional + exit_commission)

    def _apply_slippage(self, price: float, side: str, opening: bool) -> float:
        """Adjust fill price for slippage.

        Opening long / closing short → price moves up (unfavourable).
        Opening short / closing long → price moves down (unfavourable).
        """
        if (side == "long" and opening) or (side == "short" and not opening):
            return price * (1 + self._slippage_pct)
        return price * (1 - self._slippage_pct)

    def _check_exit_triggers(
        self,
        position: _Position,
        current_price: float,
        regime: RegimeType | None = None,
        regime_sub_type: RegimeSubType | None = None,
        *,
        bar_atr: float = 0.0,
        chandelier_high: float = 0.0,
        chandelier_low: float = 0.0,
    ) -> str:
        """Check stop-loss, trailing stop, Chandelier, support-based, and take-profit.

        Returns exit reason or empty string.

        Stop hierarchy (first triggered wins):
          0. Max hold bars (absolute time limit)
          1. Trailing stop (strong_trend regime)
          2. Chandelier Exit (ATR from recent high/low)
          3. Support/resistance-based stop
          4. Fixed / ATR stop-loss
          5. Take-profit

        The strategy picks the **tightest protective stop** among options 2-4
        per regime. In strong_trend, the trailing stop (option 1) takes priority.
        """
        pnl_pct = position.unrealized_pnl_pct(current_price)

        # ── Max hold bars (absolute time constraint) ────────────────────
        if self._max_hold_bars > 0 and position.bars_held >= self._max_hold_bars:
            return "max_hold"

        # ── Trailing stop management ────────────────────────────────────
        # Update the high-water mark for trailing stop
        if position.side == "long":
            if current_price > self._trailing_high:
                self._trailing_high = current_price
        else:
            if self._trailing_high == 0.0 or current_price < self._trailing_high:
                self._trailing_high = current_price

        # Check trailing stop in strong_trend regime
        if regime == RegimeType.STRONG_TREND and position.entry_atr > 0:
            regime_cfg = self._cfg.section("regime")
            adapt = dict(regime_cfg.get("strategy_adaptation", {}).get("strong_trend", {}))
            # Merge sub-type overrides on top of base params
            if regime_sub_type is not None:
                sub_overrides = adapt.get("sub_types", {}).get(regime_sub_type.value, {})
                if sub_overrides:
                    adapt.update(sub_overrides)
            if adapt.get("use_trailing_stop", True):
                trail_mult = float(adapt.get("trailing_stop_atr_mult", 4.0))
                trail_dist = trail_mult * position.entry_atr
                if position.side == "long":
                    trail_stop_price = self._trailing_high - trail_dist
                    in_profit = not self._trailing_stop_require_profit or self._trailing_high > position.entry_price
                    if current_price <= trail_stop_price and in_profit:
                        return "trailing_stop"
                else:
                    trail_stop_price = self._trailing_high + trail_dist
                    in_profit = not self._trailing_stop_require_profit or self._trailing_high < position.entry_price
                    if current_price >= trail_stop_price and in_profit:
                        return "trailing_stop"

        # ── Chandelier Exit ─────────────────────────────────────────────
        # Stop at highest_high - N*ATR (long) or lowest_low + N*ATR (short)
        if self._chandelier_enabled and bar_atr > 0:
            if position.side == "long" and chandelier_high > 0:
                chandelier_stop = chandelier_high - self._chandelier_atr_mult * bar_atr
                if current_price <= chandelier_stop:
                    return "chandelier_stop"
            elif position.side == "short" and chandelier_low > 0:
                chandelier_stop = chandelier_low + self._chandelier_atr_mult * bar_atr
                if current_price >= chandelier_stop:
                    return "chandelier_stop"

        # ── Support/resistance-based stop ───────────────────────────────
        # For longs: stop just below the nearest support level
        # For shorts: stop just above the nearest resistance level
        if self._support_stop_enabled:
            if position.side == "long" and self._support_levels:
                # Find the nearest support below entry price — this is the
                # level we expect to hold.  If current price drops through
                # that level (minus a small buffer), exit.
                supports_below = [s for s in self._support_levels if s < position.entry_price]
                if supports_below:
                    nearest_support = max(supports_below)
                    support_stop = nearest_support * (1 - self._support_stop_buffer_pct)
                    if current_price <= support_stop:
                        return "support_stop"
            elif position.side == "short" and self._resistance_levels:
                # Find the nearest resistance above entry price — this is the
                # level we expect to hold.  If current price rises through
                # that level (plus a small buffer), exit.
                resistances_above = [r for r in self._resistance_levels if r > position.entry_price]
                if resistances_above:
                    nearest_resistance = min(resistances_above)
                    resistance_stop = nearest_resistance * (1 + self._support_stop_buffer_pct)
                    if current_price >= resistance_stop:
                        return "resistance_stop"

        # ── Fixed / ATR stop-loss ───────────────────────────────────────
        # ATR stop is a floor (uses max), not a cap — allows wider breathing room
        effective_stop = self._stop_loss_pct  # fixed fallback
        if self._atr_stop_enabled and position.entry_atr > 0 and position.entry_price > 0:
            atr_stop = self._atr_stop_multiplier * position.entry_atr / position.entry_price
            effective_stop = max(self._stop_loss_pct, atr_stop)

        # Regime adjustment: widen stops in volatile/choppy markets
        if regime == RegimeType.VOLATILE_CHOPPY:
            regime_cfg = self._cfg.section("regime")
            adapt = regime_cfg.get("strategy_adaptation", {}).get("volatile_choppy", {})
            if adapt.get("widen_stops", True):
                stop_mult = float(adapt.get("stop_loss_mult", 1.5))
                effective_stop *= stop_mult

        if pnl_pct <= -effective_stop:
            return "stop_loss"

        # ── Take-profit ─────────────────────────────────────────────────
        # Optionally disabled in strong_trend — trailing stop handles exits instead
        if self._disable_tp_in_strong_trend and regime == RegimeType.STRONG_TREND:
            return ""  # let trailing stop manage the exit

        if pnl_pct >= self._take_profit_pct:
            return "take_profit"
        return ""

    # ------------------------------------------------------------------
    # Indicator recomputation
    # ------------------------------------------------------------------

    def _compute_scores(
        self, trailing_df: pd.DataFrame
    ) -> tuple[dict[str, float], float, float]:
        """Run all indicators and patterns on *trailing_df*.

        Returns:
            (per_indicator_scores, indicator_composite, pattern_composite)
        """
        # Indicator scores
        registry = IndicatorRegistry(self._cfg)
        results = registry.run_all(trailing_df)

        scorer = CompositeScorer(self._cfg)
        composite = scorer.score(results)

        scores = {r.config_key: r.score for r in results if not r.error}

        # Pattern scores
        pat_registry = PatternRegistry(self._cfg)
        pat_results = pat_registry.run_all(trailing_df)

        pat_scorer = PatternCompositeScorer(self._cfg)
        pat_composite = pat_scorer.score(pat_results)

        return scores, composite["overall"], pat_composite["overall"]

    # ------------------------------------------------------------------
    # Significant patterns extraction
    # ------------------------------------------------------------------

    def _extract_significant_patterns(
        self, df: pd.DataFrame
    ) -> list[SignificantPattern]:
        """Run all pattern detectors on the full data and extract individual events.

        Each detector stores its raw pattern list in ``values``.  This method
        normalises them into a flat :class:`SignificantPattern` list sorted by
        date, filtering out neutral / low-strength entries.

        The minimum strength threshold is configurable via
        ``config.yaml → backtest → significant_pattern_min_strength`` (default 0.5).
        """
        bt_cfg = self._cfg.section("backtest")
        min_strength = float(bt_cfg.get("significant_pattern_min_strength", 0.5))

        # Build custom strength thresholds from config if provided
        raw_st = bt_cfg.get("strength_thresholds", {})
        custom_st: dict[str, list[tuple[float, str]]] | None = None
        if raw_st:
            custom_st = {}
            for detector_name, levels in raw_st.items():
                custom_st[detector_name] = [
                    (float(entry[0]), str(entry[1])) for entry in levels
                ]

        pat_registry = PatternRegistry(self._cfg)
        pat_results = pat_registry.run_all(df)

        events: list[SignificantPattern] = []

        for pr in pat_results:
            if pr.error:
                continue

            # ── Candlesticks & Inside/Outside ───────────────────────────
            # values["patterns"]: list of {bar_index, date, pattern, signal, strength}
            if "patterns" in pr.values:
                for p in pr.values["patterns"]:
                    if p.get("signal", "neutral") == "neutral":
                        continue
                    strength = float(p.get("strength", 0.0))
                    if strength < min_strength:
                        continue
                    events.append(SignificantPattern(
                        date=p["date"],
                        detector=pr.name,
                        pattern=_format_pattern_name(p["pattern"]),
                        signal=p["signal"],
                        strength=strength,
                        confidence=_strength_label(strength, pr.name, custom_st),
                    ))

            # ── Gaps ────────────────────────────────────────────────────
            # values["gaps"]: list of {bar_index, date, direction, gap_pct, gap_type, volume_surge}
            if "gaps" in pr.values:
                gap_strength_cap = float(bt_cfg.get("gap_strength_cap", 2.0))
                for g in pr.values["gaps"]:
                    direction = g["direction"]
                    gap_type = g["gap_type"]
                    gap_pct = g["gap_pct"]

                    # Map gap direction + type to signal
                    if gap_type == "exhaustion":
                        # Exhaustion gaps signal reversal
                        signal = "bearish" if direction == "up" else "bullish"
                    else:
                        signal = "bullish" if direction == "up" else "bearish"

                    # Use gap_pct as a proxy for strength (typical gaps are 0.5%-3%)
                    strength = min(gap_pct * 100, gap_strength_cap)
                    if strength < min_strength:
                        continue

                    vol_tag = " [VOL]" if g["volume_surge"] else ""
                    events.append(SignificantPattern(
                        date=g["date"],
                        detector=pr.name,
                        pattern=f"{gap_type.capitalize()} Gap {direction.upper()}",
                        signal=signal,
                        strength=strength,
                        confidence=_strength_label(strength, pr.name, custom_st),
                        detail=f"{gap_pct:.1%}{vol_tag}",
                    ))

            # ── Spikes ──────────────────────────────────────────────────
            # values["spikes"]: list of {bar_index, date, direction, z_score, spike_level, confirmed}
            if "spikes" in pr.values:
                spike_z_divisor = float(bt_cfg.get("spike_z_divisor", 2.5))
                spike_strength_cap = float(bt_cfg.get("spike_strength_cap", 2.0))
                for s in pr.values["spikes"]:
                    direction = s["direction"]
                    confirmed = s["confirmed"]
                    z_score = abs(s["z_score"])

                    # Traps invert the signal
                    if confirmed is False:
                        signal = "bearish" if direction == "up" else "bullish"
                        status = "Trap"
                    elif confirmed is True:
                        signal = "bullish" if direction == "up" else "bearish"
                        status = "Confirmed"
                    else:
                        signal = "bullish" if direction == "up" else "bearish"
                        status = "Pending"

                    strength = min(z_score / spike_z_divisor, spike_strength_cap)  # normalise z-score
                    if strength < min_strength:
                        continue

                    events.append(SignificantPattern(
                        date=s["date"],
                        detector=pr.name,
                        pattern=f"Spike {direction.upper()}",
                        signal=signal,
                        strength=strength,
                        confidence=_strength_label(strength, pr.name, custom_st),
                        detail=f"z={s['z_score']:.1f} {status}",
                    ))

            # ── Volume-Range ────────────────────────────────────────────
            # This detector only returns aggregate regime, not per-bar events.
            # Skip — no individual patterns to extract.

        # Sort by date
        events.sort(key=lambda e: e.date)
        return events

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self, result: BacktestResult, interval: str = "1d") -> None:
        """Populate performance metrics on *result* in-place.

        Metrics that depend on time (annualized return, Sharpe ratio)
        use only the **post-warmup** portion of the equity curve so that
        the flat warmup period does not dilute the results.  Weekly and
        monthly intervals are correctly annualized.
        """
        trades = result.trades
        result.total_trades = len(trades)

        # Bars per trading year for this interval
        bars_per_year = self._bars_per_year(interval)

        # Total return (always based on full initial → final)
        if result.initial_cash > 0:
            result.total_return_pct = (
                (result.final_equity - result.initial_cash) / result.initial_cash * 100
            )
        else:
            result.total_return_pct = 0.0

        # Post-warmup equity curve (exclude flat warmup bars)
        curve = result.equity_curve
        warmup = result.warmup_bars
        trading_curve = curve[warmup:] if warmup < len(curve) else curve

        # Annualized return — use post-warmup equity window so the time
        # period matches the return period (no warmup inflation).
        if len(trading_curve) >= 2 and result.initial_cash > 0:
            n_bars = len(trading_curve)
            years = n_bars / bars_per_year
            start_equity = trading_curve[0]["equity"]
            end_equity = trading_curve[-1]["equity"]
            if years > 0 and start_equity > 0:
                total_return = end_equity / start_equity
                if total_return > 0:
                    result.annualized_return_pct = (
                        (total_return ** (1 / years) - 1) * 100
                    )
                else:
                    # Equity went negative — compute via total return pct
                    pct = (end_equity - start_equity) / start_equity * 100
                    result.annualized_return_pct = (
                        ((1 + pct / 100) ** (1 / years) - 1) * 100
                        if pct > -100
                        else -100.0
                    )

        # Max drawdown (use post-warmup curve for consistency)
        if trading_curve:
            peak = trading_curve[0]["equity"]
            max_dd = 0.0
            for pt in trading_curve:
                eq = pt["equity"]
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak if peak > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd
            result.max_drawdown_pct = max_dd * 100

        # Win rate
        if trades:
            wins = [t for t in trades if t.pnl > 0]
            result.win_rate_pct = len(wins) / len(trades) * 100

            # Profit factor
            gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
            result.profit_factor = (
                gross_profit / gross_loss if gross_loss > 0 else float("inf")
            )

            # Average trade PnL %
            result.avg_trade_pnl_pct = (
                sum(t.pnl_pct for t in trades) / len(trades) * 100
            )
            result.best_trade_pnl_pct = max(t.pnl_pct for t in trades) * 100
            result.worst_trade_pnl_pct = min(t.pnl_pct for t in trades) * 100

            # Average bars held
            result.avg_bars_held = sum(t.bars_held for t in trades) / len(trades)

        # Sharpe ratio — use post-warmup equity curve and sample std dev
        if len(trading_curve) >= 2:
            equities = [pt["equity"] for pt in trading_curve]
            bar_returns = []
            for j in range(1, len(equities)):
                if equities[j - 1] > 0:
                    bar_returns.append(equities[j] / equities[j - 1] - 1)
                else:
                    # Non-positive equity: treat as -100% return rather than
                    # silently dropping the bar (which would inflate Sharpe).
                    bar_returns.append(-1.0)
            if len(bar_returns) >= 2:
                mean_r = sum(bar_returns) / len(bar_returns)
                var_r = sum((r - mean_r) ** 2 for r in bar_returns) / (len(bar_returns) - 1)
                std_r = math.sqrt(var_r)
                if std_r > 0:
                    result.sharpe_ratio = (mean_r / std_r) * math.sqrt(bars_per_year)
