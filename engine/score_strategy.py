"""
engine/score_strategy.py — Score-based trading strategy.

Maps composite indicator scores to LONG / SHORT / HOLD signals
using configurable thresholds from config.yaml.

    composite score <= short_below  → SHORT
    composite score <= hold_below   → HOLD
    composite score >  hold_below   → LONG
"""

from __future__ import annotations

from typing import Any

from engine.strategy import Signal, Strategy, StrategyContext, TradeOrder


class ScoreBasedStrategy(Strategy):
    """Simple threshold strategy driven by composite technical scores.

    Parameters (loaded from ``config.yaml`` → ``strategy`` section):
        score_thresholds.short_below : float  — score at or below → SHORT
        score_thresholds.hold_below  : float  — score at or below → HOLD
        position_sizing              : str    — "fixed" or "percent_equity"
        fixed_quantity               : int    — shares per trade (fixed mode)
        percent_equity               : float  — fraction of equity per trade
        stop_loss_pct                : float  — exit if loss exceeds this %
        take_profit_pct              : float  — exit if gain exceeds this %
        rebalance_interval           : int    — re-evaluate every N bars
    """

    name: str = "ScoreBasedStrategy"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        thresholds = self.params.get("score_thresholds", {})
        self._short_below: float = float(thresholds.get("short_below", 3.5))
        self._hold_below: float = float(thresholds.get("hold_below", 6.5))
        self._sizing: str = self.params.get("position_sizing", "fixed")
        self._fixed_qty: int = int(self.params.get("fixed_quantity", 100))
        self._pct_equity: float = float(self.params.get("percent_equity", 0.10))
        self._stop_loss: float = float(self.params.get("stop_loss_pct", 0.05))
        self._take_profit: float = float(self.params.get("take_profit_pct", 0.15))

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def on_bar(self, ctx: StrategyContext) -> TradeOrder:
        """Decide action based on composite score and current position."""
        score = ctx.overall_score
        signal = self._score_to_signal(score)

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
            notes=f"score={score:.2f}",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_to_signal(self, score: float) -> Signal:
        if score <= self._short_below:
            return Signal.SELL
        if score <= self._hold_below:
            return Signal.HOLD
        return Signal.BUY

    def _compute_quantity(self, ctx: StrategyContext) -> float:
        if self._sizing == "percent_equity":
            price = ctx.bar.get("close", 0.0)
            if price <= 0:
                return 0.0
            return max(1.0, (ctx.portfolio_value * self._pct_equity) // price)
        return float(self._fixed_qty)
