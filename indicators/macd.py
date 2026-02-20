"""
indicators/macd.py — MACD (Moving Average Convergence Divergence) indicator.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import ta.trend

from .base import BaseIndicator


class MACDIndicator(BaseIndicator):
    name = "MACD"
    config_key = "macd"

    def compute(self, df: pd.DataFrame) -> dict[str, Any]:
        fast = int(self.config.get("fast_period", 12))
        slow = int(self.config.get("slow_period", 26))
        signal = int(self.config.get("signal_period", 9))

        macd_obj = ta.trend.MACD(
            close=df["close"],
            window_fast=fast,
            window_slow=slow,
            window_sign=signal,
        )
        macd_line = macd_obj.macd()
        signal_line = macd_obj.macd_signal()
        histogram = macd_obj.macd_diff()

        current_macd = float(macd_line.iloc[-1])
        current_signal = float(signal_line.iloc[-1])
        current_hist = float(histogram.iloc[-1])
        price = float(df["close"].iloc[-1])

        # Detect recent crossover
        lookback = int(self.config.get("scoring", {}).get("crossover_lookback", 5))
        bullish_cross = False
        bearish_cross = False
        hist_vals = histogram.dropna().values
        if len(hist_vals) >= lookback + 1:
            recent = hist_vals[-(lookback + 1):]
            for i in range(1, len(recent)):
                if recent[i - 1] < 0 and recent[i] > 0:
                    bullish_cross = True
                if recent[i - 1] > 0 and recent[i] < 0:
                    bearish_cross = True

        return {
            "macd": current_macd,
            "signal": current_signal,
            "histogram": current_hist,
            "price": price,
            "bullish_cross": bullish_cross,
            "bearish_cross": bearish_cross,
            "fast": fast,
            "slow": slow,
            "signal_period": signal,
        }

    def score(self, values: dict[str, Any]) -> float:
        hist = values["histogram"]
        price = values["price"]
        scoring = self.config.get("scoring", {})

        strong_bull_pct = float(scoring.get("strong_bullish_pct", 0.005))
        mod_bull_pct = float(scoring.get("moderate_bullish_pct", 0.001))
        strong_bear_pct = float(scoring.get("strong_bearish_pct", -0.005))
        mod_bear_pct = float(scoring.get("moderate_bearish_pct", -0.001))
        bull_bonus = float(scoring.get("bullish_cross_bonus", 1.5))
        bear_penalty = float(scoring.get("bearish_cross_penalty", 1.5))

        # Normalise histogram by price
        hist_pct = hist / price if price != 0 else 0.0

        if hist_pct >= strong_bull_pct:
            base = self._linear_score(hist_pct, strong_bull_pct, strong_bull_pct * 2, 8.5, 10.0)
        elif hist_pct >= mod_bull_pct:
            base = self._linear_score(hist_pct, mod_bull_pct, strong_bull_pct, 6.5, 8.5)
        elif hist_pct >= 0:
            base = self._linear_score(hist_pct, 0, mod_bull_pct, 5.0, 6.5)
        elif hist_pct >= mod_bear_pct:
            base = self._linear_score(hist_pct, mod_bear_pct, 0, 3.5, 5.0)
        elif hist_pct >= strong_bear_pct:
            base = self._linear_score(hist_pct, strong_bear_pct, mod_bear_pct, 1.5, 3.5)
        else:
            base = self._linear_score(hist_pct, strong_bear_pct * 2, strong_bear_pct, 0.0, 1.5)

        if values["bullish_cross"]:
            base = min(10.0, base + bull_bonus)
        elif values["bearish_cross"]:
            base = max(0.0, base - bear_penalty)

        return self._clamp(base)

    def summary(self, values: dict[str, Any], score: float) -> dict[str, Any]:
        macd = values["macd"]
        signal = values["signal"]
        hist = values["histogram"]
        direction = "above" if macd > signal else "below"
        cross_note = ""
        if values["bullish_cross"]:
            cross_note = " | Bullish cross"
        elif values["bearish_cross"]:
            cross_note = " | Bearish cross"
        return {
            "value_str": f"{macd:.3f} / {signal:.3f}",
            "detail_str": f"Hist: {hist:+.3f} | MACD {direction} signal{cross_note}",
        }
