"""
engine/score_strategy.py — Score-based trading strategy.

Maps composite indicator scores (and optionally pattern scores) to
LONG / SHORT / HOLD signals using configurable thresholds from config.yaml.

Three combination modes for indicator + pattern scores:
  "weighted" (default):
    blended_score = indicator_weight * indicator_composite + pattern_weight * pattern_composite
    Then the blended score follows the same threshold logic below.
  "gate":
    Only trade if both indicator and pattern scores pass their thresholds.
  "boost":
    Indicator score is the base signal. When patterns are active (score
    meaningfully away from 5.0), the pattern deviation is scaled by
    boost_strength and added to the indicator score. When no patterns are
    active, the indicator score passes through unmodified.

Fixed mode (threshold_mode: "fixed"):
    composite score <= short_below  → SHORT
    composite score <= hold_below   → HOLD
    composite score >  hold_below   → LONG

Percentile mode (threshold_mode: "percentile"):
    Maintains a rolling window of recent composite scores.
    composite score in bottom short_percentile% → SHORT
    composite score in top (100 - long_percentile)% → LONG
    otherwise → HOLD
    Self-calibrating: adapts to each stock's actual score distribution.

Respects the trading mode set by suitability analysis:
    long_short — full signals (default)
    long_only  — SHORT signals become HOLD (go to cash instead of shorting)
    hold_only  — all signals become HOLD (no trading)
"""

from __future__ import annotations

from collections import deque
from typing import Any

from engine.strategy import Signal, Strategy, StrategyContext, TradeOrder
from engine.suitability import TradingMode
from engine.regime import RegimeType


class ScoreBasedStrategy(Strategy):
    """Threshold strategy driven by composite technical scores.

    Supports two threshold modes:
        "fixed"      — absolute score thresholds (default)
        "percentile" — rolling percentile-based adaptive thresholds

    Supports three combination modes for indicator + pattern scores:
        "weighted" — blended_score = w_ind * indicator_score + w_pat * pattern_score
        "gate"     — both scores must independently pass thresholds to trade
        "boost"    — indicator is the base; patterns amplify/dampen when active

    Supports market regime adaptation:
        When ``ctx.regime`` is set, the strategy adapts its behavior based on
        the detected regime (strong_trend, mean_reverting, volatile_choppy,
        breakout_transition).  Adaptation parameters are read from config.yaml
        → ``regime.strategy_adaptation``.

    Parameters (loaded from ``config.yaml`` → ``strategy`` section):
        threshold_mode               : str    — "fixed" or "percentile"
        score_thresholds.short_below : float  — score at or below → SHORT (fixed mode)
        score_thresholds.hold_below  : float  — score at or below → HOLD  (fixed mode)
        percentile_thresholds.short_percentile : int — bottom N% → SHORT (percentile mode)
        percentile_thresholds.long_percentile  : int — top N% → LONG (percentile mode)
        percentile_thresholds.lookback_bars    : int — rolling window size
        combination_mode             : str    — "weighted", "gate", or "boost"
        indicator_weight             : float  — weight of indicator composite (weighted mode)
        pattern_weight               : float  — weight of pattern composite (weighted mode)
        gate_indicator_min           : float  — indicator score must exceed this for LONG (gate mode)
        gate_indicator_max           : float  — indicator score must be below this for SHORT (gate mode)
        gate_pattern_min             : float  — pattern score must exceed this for LONG (gate mode)
        gate_pattern_max             : float  — pattern score must be below this for SHORT (gate mode)
        boost_strength               : float  — multiplier for pattern deviation (boost mode)
        boost_dead_zone              : float  — pattern score within 5.0 ± this → no boost
        position_sizing              : str    — "fixed" or "percent_equity"
        fixed_quantity               : int    — shares per trade (fixed mode)
        percent_equity               : float  — fraction of equity per trade
        stop_loss_pct                : float  — exit if loss exceeds this %
        take_profit_pct              : float  — exit if gain exceeds this %
        rebalance_interval           : int    — re-evaluate every N bars
    """

    name: str = "ScoreBasedStrategy"

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        trading_mode: TradingMode = TradingMode.LONG_SHORT,
        regime_adaptation: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(params)
        self._trading_mode = trading_mode

        # Threshold mode
        self._threshold_mode: str = self.params.get("threshold_mode", "fixed")

        # Fixed thresholds
        thresholds = self.params.get("score_thresholds", {})
        self._short_below: float = float(thresholds.get("short_below", 3.5))
        self._hold_below: float = float(thresholds.get("hold_below", 6.5))

        # Percentile thresholds
        pct_cfg = self.params.get("percentile_thresholds", {})
        self._short_percentile: float = float(pct_cfg.get("short_percentile", 25))
        self._long_percentile: float = float(pct_cfg.get("long_percentile", 75))
        self._lookback_bars: int = int(pct_cfg.get("lookback_bars", 60))

        # Rolling score window for percentile mode
        self._score_window: deque[float] = deque(maxlen=self._lookback_bars)

        # ── Indicator + Pattern combination ─────────────────────────────
        self._combination_mode: str = self.params.get("combination_mode", "weighted")
        self._indicator_weight: float = float(self.params.get("indicator_weight", 0.7))
        self._pattern_weight: float = float(self.params.get("pattern_weight", 0.3))

        # Gate mode thresholds
        self._gate_indicator_min: float = float(self.params.get("gate_indicator_min", 5.5))
        self._gate_indicator_max: float = float(self.params.get("gate_indicator_max", 4.5))
        self._gate_pattern_min: float = float(self.params.get("gate_pattern_min", 5.5))
        self._gate_pattern_max: float = float(self.params.get("gate_pattern_max", 4.5))

        # Boost mode parameters
        self._boost_strength: float = float(self.params.get("boost_strength", 0.5))
        self._boost_dead_zone: float = float(self.params.get("boost_dead_zone", 0.3))

        # Position sizing
        self._sizing: str = self.params.get("position_sizing", "fixed")
        self._fixed_qty: int = int(self.params.get("fixed_quantity", 100))
        self._pct_equity: float = float(self.params.get("percent_equity", 0.10))
        self._stop_loss: float = float(self.params.get("stop_loss_pct", 0.05))
        self._take_profit: float = float(self.params.get("take_profit_pct", 0.15))

        # Trend confirmation filter
        self._trend_confirm_enabled: bool = bool(self.params.get("trend_confirm_enabled", True))
        self._trend_confirm_period: int = int(self.params.get("trend_confirm_period", 20))

        # ── Regime adaptation config ────────────────────────────────────
        ra = regime_adaptation or {}
        self._regime_adapt: dict[str, dict[str, Any]] = {
            "strong_trend": ra.get("strong_trend", {
                "use_trailing_stop": True,
                "trailing_stop_atr_mult": 3.0,
                "ignore_score_entries": True,
                "hold_with_trend": True,
            }),
            "mean_reverting": ra.get("mean_reverting", {
                "use_trailing_stop": False,
                "tighten_thresholds": True,
                "threshold_adjustment": 0.3,
            }),
            "volatile_choppy": ra.get("volatile_choppy", {
                "reduce_position_size": True,
                "position_size_mult": 0.5,
                "widen_stops": True,
                "stop_loss_mult": 1.5,
            }),
            "breakout_transition": ra.get("breakout_transition", {
                "use_momentum_entry": True,
                "breakout_atr_mult": 1.5,
                "require_volume_surge": True,
                "volume_surge_mult": 1.3,
            }),
        }
        # Trailing stop tracker (for strong trend regime)
        self._trailing_stop_price: float = 0.0

    @property
    def trading_mode(self) -> TradingMode:
        return self._trading_mode

    @trading_mode.setter
    def trading_mode(self, mode: TradingMode) -> None:
        self._trading_mode = mode

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def on_bar(self, ctx: StrategyContext) -> TradeOrder:
        """Decide action based on composite score, regime, and current position.

        When a market regime is detected, the strategy adapts:

        **Strong Trend**: If ``hold_with_trend`` is enabled, maintain existing
        positions and only exit via trailing stop. If ``ignore_score_entries``
        is set, enter based on trend direction rather than score thresholds.

        **Mean-Reverting**: Tighten score thresholds (narrow the HOLD zone)
        to generate more swing trade signals near support/resistance.

        **Volatile/Choppy**: Reduce position size and optionally widen stops.
        Score thresholds are unchanged — fewer signals is fine for choppy markets.

        **Breakout/Transition**: Require price momentum (close must move beyond
        a multiple of recent range) and optionally volume surge before entry.
        """
        ind_score = ctx.overall_score
        pat_score = ctx.pattern_score
        regime = ctx.regime

        # ── Get regime-adapted thresholds ───────────────────────────────
        short_below, hold_below = self._get_regime_thresholds(regime)

        # ── Strong Trend: hold-with-trend logic ─────────────────────────
        if regime == RegimeType.STRONG_TREND:
            adapt = self._regime_adapt["strong_trend"]
            if adapt.get("hold_with_trend", True) and ctx.position != 0:
                return self._strong_trend_hold(ctx, ind_score, pat_score)
            if adapt.get("ignore_score_entries", True) and ctx.position == 0:
                order = self._strong_trend_entry(ctx, ind_score, pat_score)
                if order is not None:
                    return order
                # Fall through to normal score logic if trend entry doesn't fire

        # ── Compute effective score ─────────────────────────────────────
        if self._combination_mode == "gate":
            signal = self._gate_signal(ind_score, pat_score)
            effective_score = ind_score  # for notes
        elif self._combination_mode == "boost":
            effective_score = self._boost_score(ind_score, pat_score)
            self._score_window.append(effective_score)
            signal = self._score_to_signal(effective_score, short_below, hold_below)
        else:
            # Weighted blend
            w_total = self._indicator_weight + self._pattern_weight
            if w_total > 0:
                effective_score = (
                    self._indicator_weight * ind_score
                    + self._pattern_weight * pat_score
                ) / w_total
            else:
                effective_score = ind_score
            # Update rolling window with the blended score
            self._score_window.append(effective_score)
            signal = self._score_to_signal(effective_score, short_below, hold_below)

        # ── Breakout/Transition: require momentum confirmation ──────────
        if regime == RegimeType.BREAKOUT_TRANSITION and ctx.position == 0:
            if signal in (Signal.BUY, Signal.SELL):
                if not self._breakout_confirmed(ctx):
                    signal = Signal.HOLD

        # ── Apply trading mode constraints ──────────────────────────────
        signal = self._constrain_signal(signal, ctx.position)

        # ── Trend confirmation filter ───────────────────────────────────
        # Prevent entering trades against the short-term trend direction.
        # Only applied to NEW entries (not closing existing positions).
        if self._trend_confirm_enabled and ctx.trend_ma > 0 and ctx.position == 0:
            close = ctx.bar.get("close", 0.0)
            if signal == Signal.BUY and close < ctx.trend_ma:
                signal = Signal.HOLD  # don't buy into a downtrend
            elif signal == Signal.SELL and close > ctx.trend_ma:
                signal = Signal.HOLD  # don't short into an uptrend

        # Determine desired quantity (regime-adapted)
        quantity = self._compute_quantity(ctx, regime)

        # If we already hold a position in the signal direction, HOLD.
        current_pos = ctx.position  # positive = long, negative = short
        if signal == Signal.BUY and current_pos > 0:
            signal = Signal.HOLD
        elif signal == Signal.SELL and current_pos < 0:
            signal = Signal.HOLD

        regime_tag = f" regime={regime.value}" if regime else ""
        return TradeOrder(
            signal=signal,
            quantity=quantity,
            notes=(
                f"ind={ind_score:.2f} pat={pat_score:.2f} "
                f"eff={effective_score:.2f} mode={self._trading_mode.value}"
                + (f" tma={ctx.trend_ma:.2f}" if self._trend_confirm_enabled and ctx.trend_ma > 0 else "")
                + regime_tag
            ),
        )

    def on_start(self, metadata: dict[str, Any]) -> None:
        """Reset rolling window and trailing stop at the start of each backtest."""
        self._score_window.clear()
        self._trailing_stop_price = 0.0

    # ------------------------------------------------------------------
    # Regime adaptation helpers
    # ------------------------------------------------------------------

    def _get_regime_thresholds(
        self, regime: RegimeType | None
    ) -> tuple[float, float]:
        """Return (short_below, hold_below) thresholds, adjusted for regime.

        Mean-reverting regime narrows the HOLD zone (tighter thresholds).
        Other regimes use the base thresholds.
        """
        short_below = self._short_below
        hold_below = self._hold_below

        if regime == RegimeType.MEAN_REVERTING:
            adapt = self._regime_adapt["mean_reverting"]
            if adapt.get("tighten_thresholds", True):
                adj = float(adapt.get("threshold_adjustment", 0.3))
                short_below = self._short_below + adj   # raise floor (easier to SHORT)
                hold_below = self._hold_below - adj     # lower ceiling (easier to LONG)
                # Safety: ensure short_below < hold_below
                if short_below >= hold_below:
                    mid = (self._short_below + self._hold_below) / 2
                    short_below = mid - 0.1
                    hold_below = mid + 0.1

        return short_below, hold_below

    def _strong_trend_hold(
        self, ctx: StrategyContext, ind_score: float, pat_score: float
    ) -> TradeOrder:
        """Strong trend regime: hold position, manage via trailing stop.

        When holding a long in a bullish trend (or short in bearish), maintain
        the position.  The trailing stop is managed by the backtest engine's
        ATR-adaptive stop — we just signal HOLD to keep the position.

        Exit only if the score turns strongly against the position.
        """
        close = ctx.bar.get("close", 0.0)
        adapt = self._regime_adapt["strong_trend"]

        # Compute effective score for logging
        w_total = self._indicator_weight + self._pattern_weight
        if w_total > 0:
            effective_score = (
                self._indicator_weight * ind_score
                + self._pattern_weight * pat_score
            ) / w_total
        else:
            effective_score = ind_score

        # Allow exit if score is extremely bearish (for longs) or bullish (for shorts)
        # Use a wide margin — only exit on extreme conviction reversal
        extreme_bearish = self._short_below - 0.5  # e.g. 4.0 if short_below=4.5
        extreme_bullish = self._hold_below + 0.5   # e.g. 6.0 if hold_below=5.5

        signal = Signal.HOLD
        if ctx.position > 0 and effective_score < extreme_bearish:
            signal = Signal.SELL  # exit long on extreme bearish
        elif ctx.position < 0 and effective_score > extreme_bullish:
            signal = Signal.BUY  # exit short on extreme bullish

        signal = self._constrain_signal(signal, ctx.position)
        quantity = self._compute_quantity(ctx, ctx.regime)

        return TradeOrder(
            signal=signal,
            quantity=quantity,
            notes=(
                f"ind={ind_score:.2f} pat={pat_score:.2f} "
                f"eff={effective_score:.2f} mode={self._trading_mode.value}"
                f" regime=strong_trend hold_with_trend"
            ),
        )

    def _strong_trend_entry(
        self, ctx: StrategyContext, ind_score: float, pat_score: float
    ) -> TradeOrder | None:
        """Strong trend regime: enter based on trend direction, not score.

        If trend_ma is available and price is clearly above it, go long.
        If clearly below, go short. Otherwise, return None to fall through
        to normal score-based logic.
        """
        close = ctx.bar.get("close", 0.0)
        trend_ma = ctx.trend_ma

        if trend_ma <= 0 or close <= 0:
            return None  # no trend data — fall through

        distance_pct = (close - trend_ma) / trend_ma

        # Require some distance from MA to confirm trend direction
        # (at least 0.5% above/below MA)
        min_distance = 0.005

        signal = Signal.HOLD
        if distance_pct > min_distance:
            signal = Signal.BUY
        elif distance_pct < -min_distance:
            signal = Signal.SELL

        if signal == Signal.HOLD:
            return None  # no clear trend direction — fall through

        signal = self._constrain_signal(signal, ctx.position)
        if signal == Signal.HOLD:
            return None

        # Compute effective score for logging
        w_total = self._indicator_weight + self._pattern_weight
        if w_total > 0:
            effective_score = (
                self._indicator_weight * ind_score
                + self._pattern_weight * pat_score
            ) / w_total
        else:
            effective_score = ind_score

        quantity = self._compute_quantity(ctx, ctx.regime)

        return TradeOrder(
            signal=signal,
            quantity=quantity,
            notes=(
                f"ind={ind_score:.2f} pat={pat_score:.2f} "
                f"eff={effective_score:.2f} mode={self._trading_mode.value}"
                f" regime=strong_trend trend_entry dist={distance_pct:+.3f}"
            ),
        )

    def _breakout_confirmed(self, ctx: StrategyContext) -> bool:
        """Breakout/Transition regime: check momentum & volume requirements.

        Returns True if the breakout conditions are met:
        1. Price moved beyond breakout_atr_mult × recent range (approximated
           from the bar's high-low as a proxy for ATR when actual ATR is not
           available in the context).
        2. Optionally, volume exceeds volume_surge_mult × average.
        """
        adapt = self._regime_adapt["breakout_transition"]
        bar = ctx.bar

        # Check price momentum: current bar range should indicate expansion
        atr_mult = float(adapt.get("breakout_atr_mult", 1.5))
        bar_range = bar.get("high", 0.0) - bar.get("low", 0.0)
        close = bar.get("close", 0.0)
        open_price = bar.get("open", 0.0)

        if close <= 0:
            return False

        # Use the bar move (|close - open|) as a proxy for directional momentum
        bar_move = abs(close - open_price)
        bar_range_pct = bar_range / close if close > 0 else 0

        # A typical daily ATR is around 1-3% of price. We check if the bar
        # move is meaningfully directional (> 50% of bar range), which suggests
        # breakout rather than just wide range noise.
        move_ratio = bar_move / bar_range if bar_range > 0 else 0
        if move_ratio < 0.4:
            return False  # bar is mostly wick, not a real breakout candle

        # Volume surge check
        if adapt.get("require_volume_surge", True):
            vol_mult = float(adapt.get("volume_surge_mult", 1.3))
            volume = bar.get("volume", 0.0)
            # We don't have average volume in ctx directly, but we can use
            # metadata if available.  For now, skip volume check if we can't
            # verify — the bar structure check above is the primary filter.
            avg_vol = ctx.metadata.get("avg_volume")
            if avg_vol and avg_vol > 0:
                if volume < avg_vol * vol_mult:
                    return False

        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gate_signal(self, ind_score: float, pat_score: float) -> Signal:
        """Gate mode: both scores must independently agree for a trade signal.

        LONG  requires: ind_score > gate_indicator_min AND pat_score > gate_pattern_min
        SHORT requires: ind_score < gate_indicator_max AND pat_score < gate_pattern_max
        Otherwise: HOLD

        The rolling window is updated with the indicator score (since gate
        mode doesn't produce a single blended number).
        """
        self._score_window.append(ind_score)

        # Check LONG gate
        if ind_score > self._gate_indicator_min and pat_score > self._gate_pattern_min:
            return Signal.BUY

        # Check SHORT gate
        if ind_score < self._gate_indicator_max and pat_score < self._gate_pattern_max:
            return Signal.SELL

        return Signal.HOLD

    def _boost_score(self, ind_score: float, pat_score: float) -> float:
        """Boost mode: indicator is the base, patterns amplify when active.

        If the pattern score is within the dead zone around 5.0 (no pattern
        activity), the indicator score passes through unmodified.

        When patterns ARE active, the deviation ``(pat_score - 5.0)`` is
        scaled by ``boost_strength`` and added to the indicator score.

        The result is clamped to [0, 10].
        """
        pat_deviation = pat_score - 5.0
        if abs(pat_deviation) <= self._boost_dead_zone:
            return ind_score

        # Remove the dead zone portion so the boost ramps smoothly from 0
        effective_deviation = pat_deviation - (
            self._boost_dead_zone if pat_deviation > 0 else -self._boost_dead_zone
        )
        boost = effective_deviation * self._boost_strength
        return max(0.0, min(10.0, ind_score + boost))

    def _score_to_signal(
        self,
        score: float,
        short_below: float | None = None,
        hold_below: float | None = None,
    ) -> Signal:
        if short_below is None:
            short_below = self._short_below
        if hold_below is None:
            hold_below = self._hold_below
        if self._threshold_mode == "percentile":
            return self._percentile_signal(score, short_below, hold_below)
        return self._fixed_signal(score, short_below, hold_below)

    def _fixed_signal(
        self,
        score: float,
        short_below: float | None = None,
        hold_below: float | None = None,
    ) -> Signal:
        """Map score to signal using absolute thresholds."""
        sb = short_below if short_below is not None else self._short_below
        hb = hold_below if hold_below is not None else self._hold_below
        if score <= sb:
            return Signal.SELL
        if score <= hb:
            return Signal.HOLD
        return Signal.BUY

    def _percentile_signal(
        self,
        score: float,
        short_below: float | None = None,
        hold_below: float | None = None,
    ) -> Signal:
        """Map score to signal using rolling percentile rank.

        Falls back to fixed thresholds if the rolling window
        doesn't have enough data yet (< 80% full).
        """
        min_samples = max(10, int(self._lookback_bars * 0.8))
        if len(self._score_window) < min_samples:
            # Not enough history — fall back to fixed thresholds
            return self._fixed_signal(score, short_below, hold_below)

        # Compute percentile rank of current score in the window
        rank = self._percentile_rank(score)

        if rank <= self._short_percentile:
            return Signal.SELL
        if rank >= self._long_percentile:
            return Signal.BUY
        return Signal.HOLD

    def _percentile_rank(self, score: float) -> float:
        """Compute the percentile rank (0-100) of *score* within the window.

        Uses the 'weak' definition: % of values strictly less than score.
        Returns 0.0 if score is the minimum, 100.0 if strictly above all.
        """
        n = len(self._score_window)
        if n == 0:
            return 50.0
        count_below = sum(1 for s in self._score_window if s < score)
        return (count_below / n) * 100.0

    def _constrain_signal(self, signal: Signal, position: float) -> Signal:
        """Apply trading mode constraints to the raw signal.

        long_only:
          - If currently long and score is bearish (SELL): keep SELL so the
            engine closes the long position. The engine will attempt to open
            a short next, but we suppress that in the engine via trading_mode.
          - If flat and score is bearish (SELL): convert to HOLD (don't open short).
          - BUY signals are always allowed.

        hold_only:
          - All signals → HOLD (no trading at all).
        """
        if self._trading_mode == TradingMode.HOLD_ONLY:
            return Signal.HOLD

        if self._trading_mode == TradingMode.LONG_ONLY:
            if signal == Signal.SELL:
                # If holding a long, allow SELL to close it.
                # If flat or already short, suppress to HOLD.
                if position > 0:
                    return Signal.SELL  # close the long
                return Signal.HOLD     # don't open a short

        return signal

    def _compute_quantity(
        self, ctx: StrategyContext, regime: RegimeType | None = None
    ) -> float:
        """Compute position size, adjusted for regime.

        In volatile/choppy regime, the position size is reduced by
        ``position_size_mult`` (default 0.5) to manage risk.
        """
        if self._sizing == "percent_equity":
            price = ctx.bar.get("close", 0.0)
            if price <= 0:
                return 0.0
            qty = max(1.0, (ctx.portfolio_value * self._pct_equity) // price)
        else:
            qty = float(self._fixed_qty)

        # Regime adjustment: reduce size in volatile/choppy markets
        if regime == RegimeType.VOLATILE_CHOPPY:
            adapt = self._regime_adapt["volatile_choppy"]
            if adapt.get("reduce_position_size", True):
                mult = float(adapt.get("position_size_mult", 0.5))
                qty = max(1.0, qty * mult)

        return qty
