"""
engine/score_strategy.py — Score-based trading strategy.

Maps composite indicator scores (and optionally pattern scores) to
LONG / SHORT / HOLD signals using configurable thresholds from config.yaml.

Two combination modes for indicator + pattern scores:
  "weighted" (default):
    blended_score = indicator_weight * indicator_composite + pattern_weight * pattern_composite
    Then the blended score follows the same threshold logic below.
  "gate":
    Only trade if both indicator and pattern scores pass their thresholds.

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


class ScoreBasedStrategy(Strategy):
    """Threshold strategy driven by composite technical scores.

    Supports two threshold modes:
        "fixed"      — absolute score thresholds (default)
        "percentile" — rolling percentile-based adaptive thresholds

    Supports two combination modes for indicator + pattern scores:
        "weighted" — blended_score = w_ind * indicator_score + w_pat * pattern_score
        "gate"     — both scores must independently pass thresholds to trade

    Parameters (loaded from ``config.yaml`` → ``strategy`` section):
        threshold_mode               : str    — "fixed" or "percentile"
        score_thresholds.short_below : float  — score at or below → SHORT (fixed mode)
        score_thresholds.hold_below  : float  — score at or below → HOLD  (fixed mode)
        percentile_thresholds.short_percentile : int — bottom N% → SHORT (percentile mode)
        percentile_thresholds.long_percentile  : int — top N% → LONG (percentile mode)
        percentile_thresholds.lookback_bars    : int — rolling window size
        combination_mode             : str    — "weighted" or "gate"
        indicator_weight             : float  — weight of indicator composite (weighted mode)
        pattern_weight               : float  — weight of pattern composite (weighted mode)
        gate_indicator_min           : float  — indicator score must exceed this for LONG (gate mode)
        gate_indicator_max           : float  — indicator score must be below this for SHORT (gate mode)
        gate_pattern_min             : float  — pattern score must exceed this for LONG (gate mode)
        gate_pattern_max             : float  — pattern score must be below this for SHORT (gate mode)
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

        # Position sizing
        self._sizing: str = self.params.get("position_sizing", "fixed")
        self._fixed_qty: int = int(self.params.get("fixed_quantity", 100))
        self._pct_equity: float = float(self.params.get("percent_equity", 0.10))
        self._stop_loss: float = float(self.params.get("stop_loss_pct", 0.05))
        self._take_profit: float = float(self.params.get("take_profit_pct", 0.15))

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
        """Decide action based on composite score and current position.

        In "weighted" mode, blends indicator and pattern scores into a single
        effective score, then applies the normal threshold logic.

        In "gate" mode, both indicator and pattern scores must independently
        pass their gate thresholds for a trade signal to fire; otherwise HOLD.
        """
        ind_score = ctx.overall_score
        pat_score = ctx.pattern_score

        if self._combination_mode == "gate":
            signal = self._gate_signal(ind_score, pat_score)
            effective_score = ind_score  # for notes
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
            signal = self._score_to_signal(effective_score)

        # ── Apply trading mode constraints ──────────────────────────────
        signal = self._constrain_signal(signal, ctx.position)

        # Determine desired quantity
        quantity = self._compute_quantity(ctx)

        # If we already hold a position in the signal direction, HOLD.
        current_pos = ctx.position  # positive = long, negative = short
        if signal == Signal.BUY and current_pos > 0:
            signal = Signal.HOLD
        elif signal == Signal.SELL and current_pos < 0:
            signal = Signal.HOLD

        return TradeOrder(
            signal=signal,
            quantity=quantity,
            notes=(
                f"ind={ind_score:.2f} pat={pat_score:.2f} "
                f"eff={effective_score:.2f} mode={self._trading_mode.value}"
            ),
        )

    def on_start(self, metadata: dict[str, Any]) -> None:
        """Reset rolling window at the start of each backtest."""
        self._score_window.clear()

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

    def _score_to_signal(self, score: float) -> Signal:
        if self._threshold_mode == "percentile":
            return self._percentile_signal(score)
        return self._fixed_signal(score)

    def _fixed_signal(self, score: float) -> Signal:
        """Map score to signal using absolute thresholds."""
        if score <= self._short_below:
            return Signal.SELL
        if score <= self._hold_below:
            return Signal.HOLD
        return Signal.BUY

    def _percentile_signal(self, score: float) -> Signal:
        """Map score to signal using rolling percentile rank.

        Falls back to fixed thresholds if the rolling window
        doesn't have enough data yet (< 80% full).
        """
        min_samples = max(10, int(self._lookback_bars * 0.8))
        if len(self._score_window) < min_samples:
            # Not enough history — fall back to fixed thresholds
            return self._fixed_signal(score)

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

    def _compute_quantity(self, ctx: StrategyContext) -> float:
        if self._sizing == "percent_equity":
            price = ctx.bar.get("close", 0.0)
            if price <= 0:
                return 0.0
            return max(1.0, (ctx.portfolio_value * self._pct_equity) // price)
        return float(self._fixed_qty)
