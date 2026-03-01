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

import copy
import math
from collections import deque
from typing import Any

from engine.strategy import Signal, Strategy, StrategyContext, TradeOrder
from engine.suitability import TradingMode
from engine.regime import RegimeType, RegimeSubType


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
        self._hold_below: float = float(thresholds.get("hold_below", 6.0))

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
                "trailing_stop_atr_mult": 8.0,
                "ignore_score_entries": True,
                "hold_with_trend": True,
                "min_distance": 0.01,       # 1% from MA for trend entry
                "min_score": 3.5,           # don't enter when indicators are strongly bearish
                "respect_trend_direction": True,  # only enter in the direction of the long-term trend
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

        # Re-entry grace period: after exiting a position, skip trend
        # confirmation for N bars to allow faster re-entry in trending markets.
        self._reentry_grace_bars: int = int(self.params.get("reentry_grace_bars", 10))
        self._bars_since_exit: int = 999  # start high so no grace on fresh start

        # ── Consecutive loss cooldown ───────────────────────────────────
        # After N consecutive losing trades, escalate entry requirements
        # to avoid repeated re-entries during extended pullbacks.
        self._consecutive_losses: int = 0
        self._cooldown_max_losses: int = int(
            self.params.get("cooldown_max_losses", 2)
        )
        self._cooldown_distance_mult: float = float(
            self.params.get("cooldown_distance_mult", 2.0)
        )
        self._cooldown_min_score: float = float(
            self.params.get("cooldown_min_score", 4.5)
        )

        # ── Global directional bias ────────────────────────────────────
        # When the regime's total return is strongly positive/negative,
        # suppress counter-trend entries even in non-strong_trend regimes.
        self._global_bias_enabled: bool = bool(
            self.params.get("global_trend_bias", True)
        )
        self._global_bias_threshold: float = float(
            self.params.get("global_bias_threshold", 0.10)
        )

        # ── New configurable strategy parameters ───────────────────────
        self._trend_bias_return_threshold: float = float(
            self.params.get("trend_bias_return_threshold", 0.15)
        )
        self._extreme_exit_score_offset: float = float(
            self.params.get("extreme_exit_score_offset", 1.5)
        )
        self._breakout_min_move_ratio: float = float(
            self.params.get("breakout_min_move_ratio", 0.4)
        )
        self._allow_pyramiding: bool = bool(
            self.params.get("allow_pyramiding", False)
        )
        self._allow_immediate_reversal: bool = bool(
            self.params.get("allow_immediate_reversal", True)
        )
        self._disable_tp_in_strong_trend: bool = bool(
            self.params.get("disable_take_profit_in_strong_trend", True)
        )
        self._trailing_stop_require_profit: bool = bool(
            self.params.get("trailing_stop_require_profit", True)
        )
        self._percentile_min_fill_ratio: float = float(
            self.params.get("percentile_min_fill_ratio", 0.8)
        )
        self._trend_confirm_ma_type: str = str(
            self.params.get("trend_confirm_ma_type", "ema")
        )
        self._trend_confirm_tolerance_pct: float = float(
            self.params.get("trend_confirm_tolerance_pct", 0.0)
        )
        self._cooldown_reset_on_breakeven: bool = bool(
            self.params.get("cooldown_reset_on_breakeven", True)
        )

    @property
    def trading_mode(self) -> TradingMode:
        return self._trading_mode

    @trading_mode.setter
    def trading_mode(self, mode: TradingMode) -> None:
        self._trading_mode = mode

    # ------------------------------------------------------------------
    # Sub-type adaptation
    # ------------------------------------------------------------------

    def _get_regime_adapt(
        self,
        regime: RegimeType | None,
        sub_type: RegimeSubType | None = None,
    ) -> dict[str, Any]:
        """Return regime adaptation params, with sub-type overrides merged.

        Looks up the base regime params from ``self._regime_adapt``, then
        if a sub-type is provided, merges any matching sub-type overrides
        on top (sub-type values win).

        Uses ``copy.deepcopy`` so callers can never accidentally mutate
        the canonical config dict.  The ``sub_types`` key is stripped from
        the returned dict since it's internal bookkeeping only.
        """
        if regime is None:
            return {}
        base = copy.deepcopy(self._regime_adapt.get(regime.value, {}))

        if sub_type is not None:
            sub_types = base.pop("sub_types", {})
            overrides = sub_types.get(sub_type.value, {})
            if overrides:
                base.update(overrides)
        else:
            base.pop("sub_types", None)

        return base

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

        # ── NaN guard: treat missing scores as neutral ──────────────────
        # NaN comparisons silently pass all threshold checks, producing
        # false BUY (fixed mode) or false SELL (percentile mode).
        if math.isnan(ind_score):
            ind_score = 5.0
        if math.isnan(pat_score):
            pat_score = 5.0

        # Track bars since last exit for re-entry grace period
        self._bars_since_exit += 1

        # ── Get regime-adapted thresholds ───────────────────────────────
        short_below, hold_below = self._get_regime_thresholds(regime)

        # ── Strong Trend: hold-with-trend logic ─────────────────────────
        if regime == RegimeType.STRONG_TREND:
            adapt = self._get_regime_adapt(regime, ctx.regime_sub_type)
            if adapt.get("hold_with_trend", True) and ctx.position != 0:
                return self._strong_trend_hold(ctx, ind_score, pat_score)
            if adapt.get("ignore_score_entries", True) and ctx.position == 0:
                order = self._strong_trend_entry(ctx, ind_score, pat_score)
                if order is not None:
                    return order
                # Fall through to normal score logic if trend entry doesn't fire

        # ── Compute effective score ─────────────────────────────────────
        if self._combination_mode == "gate":
            signal = self._gate_signal(ind_score, pat_score, regime)
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

        # ── Global directional bias: suppress counter-trend entries ─────
        # When the regime's total return is strongly positive/negative,
        # suppress counter-trend entries even in non-strong_trend regimes.
        # This prevents shorting in a +115% trend just because the current
        # regime window happens to classify as mean_reverting.
        if self._global_bias_enabled and ctx.position == 0:
            total_return = ctx.regime_total_return
            if total_return >= self._global_bias_threshold and signal == Signal.SELL:
                signal = Signal.HOLD  # don't short in a strong uptrend
            elif total_return <= -self._global_bias_threshold and signal == Signal.BUY:
                signal = Signal.HOLD  # don't go long in a strong downtrend

        # ── Apply trading mode constraints ──────────────────────────────
        signal = self._constrain_signal(signal, ctx.position)

        # ── Trend confirmation filter ───────────────────────────────────
        # Prevent entering trades against the short-term trend direction.
        # Only applied to NEW entries (not closing existing positions).
        # SKIP during re-entry grace period to allow faster re-entry after
        # forced exits (stop-loss, take-profit, trailing stop).
        in_grace = self._bars_since_exit <= self._reentry_grace_bars
        if (
            self._trend_confirm_enabled
            and ctx.trend_ma > 0
            and ctx.position == 0
            and not in_grace
        ):
            close = ctx.bar.get("close", 0.0)
            tolerance = ctx.trend_ma * self._trend_confirm_tolerance_pct
            if signal == Signal.BUY and close < ctx.trend_ma - tolerance:
                signal = Signal.HOLD  # don't buy into a downtrend
            elif signal == Signal.SELL and close > ctx.trend_ma + tolerance:
                signal = Signal.HOLD  # don't short into an uptrend

        # Determine desired quantity (regime-adapted)
        quantity = self._compute_quantity(ctx, regime)

        # If we already hold a position in the signal direction, HOLD
        # (unless pyramiding is explicitly enabled).
        current_pos = ctx.position  # positive = long, negative = short
        was_in_position = current_pos != 0
        if not self._allow_pyramiding:
            if signal == Signal.BUY and current_pos > 0:
                signal = Signal.HOLD
            elif signal == Signal.SELL and current_pos < 0:
                signal = Signal.HOLD

        # Track exits for re-entry grace period
        if was_in_position and signal != Signal.HOLD:
            # A SELL when long or BUY when short = closing a position
            if (current_pos > 0 and signal == Signal.SELL) or (current_pos < 0 and signal == Signal.BUY):
                self._bars_since_exit = 0

        regime_tag = f" regime={regime.value}" if regime else ""
        grace_tag = " grace" if in_grace and ctx.position == 0 else ""
        return TradeOrder(
            signal=signal,
            quantity=quantity,
            notes=(
                f"ind={ind_score:.2f} pat={pat_score:.2f} "
                f"eff={effective_score:.2f} mode={self._trading_mode.value}"
                + (f" tma={ctx.trend_ma:.2f}" if self._trend_confirm_enabled and ctx.trend_ma > 0 else "")
                + regime_tag + grace_tag
            ),
        )

    def on_start(self, metadata: dict[str, Any]) -> None:
        """Reset rolling window, trailing stop, cooldown, and re-entry grace at the start of each backtest."""
        self._score_window.clear()
        self._trailing_stop_price = 0.0
        self._bars_since_exit = 999
        self._consecutive_losses = 0

    def seed_score_window(self, scores: list[float]) -> None:
        """Pre-populate the rolling score window with historical effective scores.

        Used by recommendation and scanner code paths that call ``on_bar()``
        only once.  Without seeding, percentile mode silently falls back to
        fixed thresholds because the window never reaches ``min_samples``.

        Only the last ``lookback_bars`` entries are kept (deque maxlen).
        """
        for s in scores:
            self._score_window.append(s)

    def on_trade_close(self, pnl_pct: float, exit_reason: str) -> None:
        """Track consecutive losses and reset re-entry grace on engine exits.

        Engine-forced exits (stop_loss, trailing_stop, take_profit,
        eod_flatten) bypass on_bar, so the grace period counter is never
        reset there.  We reset it here so the strategy can re-enter
        quickly after a stop-loss instead of waiting for trend confirmation.
        """
        # Reset re-entry grace period on any exit
        self._bars_since_exit = 0

        if pnl_pct < 0:
            self._consecutive_losses += 1
        elif pnl_pct == 0 and not self._cooldown_reset_on_breakeven:
            pass  # breakeven doesn't reset counter when disabled
        else:
            self._consecutive_losses = 0

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
            adapt = self._get_regime_adapt(regime)
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

        When holding a position aligned with the trend direction, maintain
        it.  The trailing stop is managed by the backtest engine's
        ATR-adaptive stop — we just signal HOLD to keep the position.

        If holding a position AGAINST the trend direction (e.g. short in
        a bullish trend), exit immediately — this shouldn't happen with the
        updated entry logic, but acts as a safety net.

        Exit only if the score turns strongly against the position.
        """
        close = ctx.bar.get("close", 0.0)
        adapt = self._get_regime_adapt(RegimeType.STRONG_TREND, ctx.regime_sub_type)
        trend_direction = ctx.regime_trend
        total_return = ctx.regime_total_return

        # Determine effective bias from total return (same logic as entry)
        if total_return >= self._trend_bias_return_threshold:
            effective_bias = "bullish"
        elif total_return <= -self._trend_bias_return_threshold:
            effective_bias = "bearish"
        else:
            effective_bias = trend_direction

        # Compute effective score respecting combination_mode
        effective_score = self._effective_score(ind_score, pat_score)

        # Always update the rolling window so percentile mode stays calibrated
        # even while the strong-trend path bypasses normal score-to-signal logic.
        self._score_window.append(effective_score)

        # Safety: exit if holding against the trend direction
        if adapt.get("respect_trend_direction", True):
            if effective_bias == "bullish" and ctx.position < 0:
                # Short in a bullish trend — close immediately
                signal = self._constrain_signal(Signal.BUY, ctx.position)
                quantity = self._compute_quantity(ctx, ctx.regime)
                return TradeOrder(
                    signal=signal,
                    quantity=quantity,
                    notes=(
                        f"ind={ind_score:.2f} pat={pat_score:.2f} "
                        f"eff={effective_score:.2f} mode={self._trading_mode.value}"
                        f" regime=strong_trend close_counter_trend bias={effective_bias}"
                    ),
                )
            elif effective_bias == "bearish" and ctx.position > 0:
                # Long in a bearish trend — close immediately
                signal = self._constrain_signal(Signal.SELL, ctx.position)
                quantity = self._compute_quantity(ctx, ctx.regime)
                return TradeOrder(
                    signal=signal,
                    quantity=quantity,
                    notes=(
                        f"ind={ind_score:.2f} pat={pat_score:.2f} "
                        f"eff={effective_score:.2f} mode={self._trading_mode.value}"
                        f" regime=strong_trend close_counter_trend bias={effective_bias}"
                    ),
                )

        # Allow exit if score is extremely bearish (for longs) or bullish (for shorts)
        # Use a wide margin — only exit on extreme conviction reversal
        extreme_bearish = self._short_below - self._extreme_exit_score_offset
        extreme_bullish = self._hold_below + self._extreme_exit_score_offset

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
                f" regime=strong_trend hold_with_trend bias={effective_bias}"
            ),
        )

    def _strong_trend_entry(
        self, ctx: StrategyContext, ind_score: float, pat_score: float
    ) -> TradeOrder | None:
        """Strong trend regime: enter in the direction of the long-term trend.

        Key principles:
        - Uses **total return** over the analysis period as the PRIMARY
          directional signal.  A stock up 30%+ is bullish regardless of
          short-term ``trend_direction``.  Falls back to ``trend_direction``
          only if total return is ambiguous (between -15% and +15%).
        - When ``respect_trend_direction`` is enabled:
          - Positive total return: only allow LONG entries, never SHORT.
          - Negative total return: only allow SHORT entries, never LONG.
          - Ambiguous return: use ``trend_direction`` or price/MA relationship.
        - Require price to be meaningfully away from the MA (configurable
          ``min_distance``, default 1%) to confirm trend resumption.
        - Require a minimum effective score (``min_score``, default 3.5) so
          we don't enter when indicators are strongly against us.
        - If conditions aren't met, return None to fall through to normal
          score-based logic (which will likely HOLD in the wide 3.5-6.5 zone).
        """
        close = ctx.bar.get("close", 0.0)
        trend_ma = ctx.trend_ma
        trend_direction = ctx.regime_trend      # "bullish", "bearish", or "neutral"
        total_return = ctx.regime_total_return   # e.g. 1.15 for +115%

        if trend_ma <= 0 or close <= 0:
            return None  # no trend data — fall through

        adapt = self._get_regime_adapt(RegimeType.STRONG_TREND, ctx.regime_sub_type)
        min_distance = float(adapt.get("min_distance", 0.01))
        min_score = float(adapt.get("min_score", 3.5))
        respect_direction = adapt.get("respect_trend_direction", True)

        # ── Consecutive loss cooldown ───────────────────────────────────
        # After repeated losses, tighten entry requirements to avoid
        # repeated re-entries during extended pullbacks.
        if self._consecutive_losses >= self._cooldown_max_losses:
            min_distance *= self._cooldown_distance_mult
            min_score = max(min_score, self._cooldown_min_score)

        distance_pct = (close - trend_ma) / trend_ma

        # Compute effective score respecting combination_mode
        effective_score = self._effective_score(ind_score, pat_score)

        # Always update the rolling window so percentile mode stays calibrated
        # even while the strong-trend path bypasses normal score-to-signal logic.
        self._score_window.append(effective_score)

        # Minimum score gate: don't enter when indicators are strongly against us
        if effective_score < min_score:
            return None

        # ── Pattern veto ────────────────────────────────────────────────
        # Even if the blended effective_score passes min_score, a strong
        # indicator score can mask an actively bearish pattern score
        # (e.g. exhaustion gap, bearish engulfing).  If the raw pattern
        # score is below the veto threshold, defer entry until patterns
        # are at least neutral.
        pattern_veto = float(adapt.get("pattern_veto_threshold", 0.0))
        if pattern_veto > 0 and pat_score < pattern_veto:
            return None  # patterns actively warn against entry — defer

        # ── Determine effective trend bias ──────────────────────────────
        # Use total return as the primary directional signal.
        # total_return is a fraction (e.g. 1.15 = +115%, -0.20 = -20%)
        if total_return >= self._trend_bias_return_threshold:
            effective_bias = "bullish"
        elif total_return <= -self._trend_bias_return_threshold:
            effective_bias = "bearish"
        else:
            # Ambiguous total return — use trend_direction or price/MA
            effective_bias = trend_direction  # may be "neutral"

        # ── Determine allowed entry direction ───────────────────────────
        signal = Signal.HOLD

        if respect_direction:
            if effective_bias == "bullish":
                # Only long in bullish trend
                if distance_pct > min_distance:
                    signal = Signal.BUY
            elif effective_bias == "bearish":
                # Only short in bearish trend
                if distance_pct < -min_distance:
                    signal = Signal.SELL
            else:
                # Neutral: enter in the direction price is relative to MA
                if distance_pct > min_distance:
                    signal = Signal.BUY
                elif distance_pct < -min_distance:
                    signal = Signal.SELL
        else:
            # respect_trend_direction disabled: pure price/MA entry
            if distance_pct > min_distance:
                signal = Signal.BUY
            elif distance_pct < -min_distance:
                signal = Signal.SELL

        if signal == Signal.HOLD:
            return None  # conditions not met — fall through

        signal = self._constrain_signal(signal, ctx.position)
        if signal == Signal.HOLD:
            return None

        quantity = self._compute_quantity(ctx, ctx.regime)

        return TradeOrder(
            signal=signal,
            quantity=quantity,
            notes=(
                f"ind={ind_score:.2f} pat={pat_score:.2f} "
                f"eff={effective_score:.2f} mode={self._trading_mode.value}"
                f" regime=strong_trend trend_entry"
                f" bias={effective_bias} dist={distance_pct:+.3f}"
            ),
        )

    def _breakout_confirmed(self, ctx: StrategyContext) -> bool:
        """Breakout/Transition regime: check momentum & volume requirements.

        Returns True if the breakout conditions are met:
        1. The bar's range (high - low) as a percentage of price must exceed
           a minimum threshold (``min_bar_range_pct``).  This filters out
           narrow-range noise bars that happen to have a good move ratio.
        2. The bar move is meaningfully directional (move ratio check).
        3. Optionally, volume exceeds volume_surge_mult × average.
        """
        adapt = self._get_regime_adapt(RegimeType.BREAKOUT_TRANSITION, ctx.regime_sub_type)
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

        # ── Bar range expansion filter ──────────────────────────────────
        # Reject bars whose range is too small to represent a genuine
        # volatility expansion / squeeze release.  A typical daily ATR is
        # 1-3% of price; requiring at least min_bar_range_pct (default 1.5%)
        # filters out narrow-range noise bars that coincidentally have a
        # favourable move ratio or volume spike.
        min_bar_range_pct = float(adapt.get("min_bar_range_pct", 0.015))
        if bar_range_pct < min_bar_range_pct:
            return False  # bar too narrow — not a real breakout expansion

        # A typical daily ATR is around 1-3% of price. We check if the bar
        # move is meaningfully directional (> 50% of bar range), which suggests
        # breakout rather than just wide range noise.
        move_ratio = bar_move / bar_range if bar_range > 0 else 0
        if move_ratio < self._breakout_min_move_ratio:
            return False  # bar is mostly wick, not a real breakout candle

        # Volume surge check
        if adapt.get("require_volume_surge", True):
            vol_mult = float(adapt.get("volume_surge_mult", 1.3))
            volume = bar.get("volume", 0.0)
            avg_vol = ctx.metadata.get("avg_volume")
            if avg_vol and avg_vol > 0:
                if volume < avg_vol * vol_mult:
                    return False

        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gate_signal(
        self,
        ind_score: float,
        pat_score: float,
        regime: RegimeType | None = None,
    ) -> Signal:
        """Gate mode: both scores must independently agree for a trade signal.

        LONG  requires: ind_score > gate_indicator_min AND pat_score > gate_pattern_min
        SHORT requires: ind_score < gate_indicator_max AND pat_score < gate_pattern_max
        Otherwise: HOLD

        In mean-reverting regime, the gate thresholds are tightened by the
        same adjustment amount as the score thresholds (making it easier
        to generate signals near support/resistance).

        The rolling window is updated with the indicator score (since gate
        mode doesn't produce a single blended number).
        """
        self._score_window.append(ind_score)

        # Start with base gate thresholds
        g_ind_min = self._gate_indicator_min
        g_ind_max = self._gate_indicator_max
        g_pat_min = self._gate_pattern_min
        g_pat_max = self._gate_pattern_max

        # Regime adaptation: tighten gate thresholds in mean-reverting regime
        if regime == RegimeType.MEAN_REVERTING:
            adapt = self._get_regime_adapt(regime)
            if adapt.get("tighten_thresholds", True):
                adj = float(adapt.get("threshold_adjustment", 0.3))
                # Lower the LONG gates (easier to BUY) and raise the SHORT gates (easier to SELL)
                g_ind_min -= adj
                g_pat_min -= adj
                g_ind_max += adj
                g_pat_max += adj

        # Check LONG gate
        if ind_score > g_ind_min and pat_score > g_pat_min:
            return Signal.BUY

        # Check SHORT gate
        if ind_score < g_ind_max and pat_score < g_pat_max:
            return Signal.SELL

        return Signal.HOLD

    def _effective_score(self, ind_score: float, pat_score: float) -> float:
        """Compute effective score respecting the configured combination_mode.

        Returns a single numeric score regardless of mode:
        - **weighted**: blended indicator/pattern score
        - **boost**: indicator boosted by pattern deviation
        - **gate**: indicator score (gate logic is handled elsewhere via
          ``_gate_signal``; this returns the indicator score for logging
          and percentile-window tracking)

        Does NOT append to ``self._score_window`` — callers should do
        that explicitly after calling this.
        """
        if self._combination_mode == "boost":
            return self._boost_score(ind_score, pat_score)
        elif self._combination_mode == "gate":
            return ind_score
        else:
            # Weighted blend (default)
            w_total = self._indicator_weight + self._pattern_weight
            if w_total > 0:
                return (
                    self._indicator_weight * ind_score
                    + self._pattern_weight * pat_score
                ) / w_total
            return ind_score

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
        min_samples = max(10, int(self._lookback_bars * self._percentile_min_fill_ratio))
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

        Uses the 'mean' definition: ``(count_below + count_equal / 2) / n``.
        This avoids the pathological case where identical scores all get
        rank 0 (the 'weak'/'strict-less-than' formula), which would
        produce false SHORT signals when the window is flat.

        Returns 50.0 for an empty window (neutral).
        """
        n = len(self._score_window)
        if n == 0:
            return 50.0
        count_below = sum(1 for s in self._score_window if s < score)
        count_equal = sum(1 for s in self._score_window if s == score)
        return ((count_below + count_equal / 2) / n) * 100.0

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
        """Compute position size, adjusted for regime and sub-type.

        In volatile/choppy regime, the position size is reduced by
        ``position_size_mult`` (default 0.5) to manage risk.
        Sub-type overrides can also reduce position size (e.g.
        volatile_directionless within strong_trend).
        """
        if self._sizing == "percent_equity":
            price = ctx.bar.get("close", 0.0)
            if price <= 0:
                return 0.0
            qty = max(1.0, (ctx.portfolio_value * self._pct_equity) // price)
        else:
            qty = float(self._fixed_qty)

        # Regime adjustment: reduce size in volatile/choppy markets
        adapt = self._get_regime_adapt(regime, ctx.regime_sub_type)
        if adapt.get("reduce_position_size", False):
            mult = float(adapt.get("position_size_mult", 0.5))
            qty = max(1.0, qty * mult)

        return qty
