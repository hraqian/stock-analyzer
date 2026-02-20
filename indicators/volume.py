"""
indicators/volume.py — On-Balance Volume (OBV) trend indicator.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import ta.volume

from .base import BaseIndicator


class VolumeIndicator(BaseIndicator):
    name = "Volume (OBV)"
    config_key = "volume"

    def compute(self, df: pd.DataFrame) -> dict[str, Any]:
        obv_period = int(self.config.get("obv_trend_period", 20))
        price_period = int(self.config.get("price_trend_period", 20))

        obv_series = ta.volume.OnBalanceVolumeIndicator(
            close=df["close"], volume=df["volume"]
        ).on_balance_volume()

        # Trend = EMA slope: compare current EMA to EMA N bars ago
        obv_ema = obv_series.ewm(span=obv_period, adjust=False).mean()
        price_ema = df["close"].ewm(span=price_period, adjust=False).mean()

        obv_current = float(obv_ema.iloc[-1])
        obv_prev = float(obv_ema.iloc[-obv_period]) if len(obv_ema) > obv_period else float(obv_ema.iloc[0])

        price_current = float(price_ema.iloc[-1])
        price_prev = float(price_ema.iloc[-price_period]) if len(price_ema) > price_period else float(price_ema.iloc[0])

        obv_rising = obv_current > obv_prev
        price_rising = price_current > price_prev

        if obv_rising == price_rising:
            signal = "confirming"
        else:
            signal = "diverging"

        obv_change_pct = (
            ((obv_current - obv_prev) / abs(obv_prev) * 100)
            if obv_prev != 0 else 0.0
        )

        return {
            "obv": float(obv_series.iloc[-1]),
            "obv_ema": obv_current,
            "obv_rising": obv_rising,
            "price_rising": price_rising,
            "signal": signal,
            "obv_change_pct": obv_change_pct,
        }

    def score(self, values: dict[str, Any]) -> float:
        scoring = self.config.get("scoring", {})

        # Continuous scoring parameters (keyed by OBV change magnitude)
        bull_max = float(scoring.get("confirmation_bullish_max", 9.5))
        bull_min = float(scoring.get("confirmation_bullish_min", 6.5))
        bear_max = float(scoring.get("confirmation_bearish_max", 3.5))
        bear_min = float(scoring.get("confirmation_bearish_min", 0.5))
        divergence = float(scoring.get("divergence_score", 5.0))

        strong_pct = float(scoring.get("obv_strong_change_pct", 10.0))
        weak_pct = float(scoring.get("obv_weak_change_pct", 1.0))

        if values["signal"] != "confirming":
            return self._clamp(divergence)

        # Magnitude of OBV change (absolute %) drives the score within range
        magnitude = abs(values["obv_change_pct"])

        if values["price_rising"]:
            # Bullish confirmation: scale from bull_min → bull_max
            score = self._linear_score(magnitude, weak_pct, strong_pct, bull_min, bull_max)
        else:
            # Bearish confirmation: scale from bear_max → bear_min
            # (stronger bearish OBV = lower score)
            score = self._linear_score(magnitude, weak_pct, strong_pct, bear_max, bear_min)

        return self._clamp(score)

    def summary(self, values: dict[str, Any], score: float) -> dict[str, Any]:
        direction = "Rising" if values["obv_rising"] else "Falling"
        signal = values["signal"].capitalize()
        price_dir = "rising" if values["price_rising"] else "falling"
        return {
            "value_str": f"OBV EMA: {direction}",
            "detail_str": (
                f"{signal} with price ({price_dir}) | "
                f"OBV change: {values['obv_change_pct']:+.1f}%"
            ),
        }
