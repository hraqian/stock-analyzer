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
        confirmation = float(scoring.get("confirmation_score", 8.0))
        neutral = float(scoring.get("neutral_score", 5.0))
        divergence = float(scoring.get("divergence_score", 2.0))

        if values["signal"] == "confirming":
            base = confirmation
            # Tilt score based on direction
            if values["price_rising"]:
                return self._clamp(base)
            else:
                # Confirming a downtrend
                return self._clamp(10.0 - base)
        else:
            return self._clamp(neutral)

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
