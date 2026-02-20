"""
indicators/adx.py — Average Directional Index indicator.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import ta.trend

from .base import BaseIndicator


class ADXIndicator(BaseIndicator):
    name = "ADX"
    config_key = "adx"

    def compute(self, df: pd.DataFrame) -> dict[str, Any]:
        period = int(self.config.get("period", 14))

        adx_obj = ta.trend.ADXIndicator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=period,
        )
        adx_val = float(adx_obj.adx().iloc[-1])
        plus_di = float(adx_obj.adx_pos().iloc[-1])
        minus_di = float(adx_obj.adx_neg().iloc[-1])

        thresholds = self.config.get("thresholds", {})
        weak = float(thresholds.get("weak", 30))
        moderate = float(thresholds.get("moderate", 50))

        if adx_val < weak:
            trend_strength = "Weak"
        elif adx_val < moderate:
            trend_strength = "Moderate"
        else:
            trend_strength = "Strong"

        trend_direction = "Bullish" if plus_di > minus_di else "Bearish"

        return {
            "adx": adx_val,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "trend_strength": trend_strength,
            "trend_direction": trend_direction,
            "period": period,
        }

    def score(self, values: dict[str, Any]) -> float:
        adx = values["adx"]
        plus_di = values["plus_di"]
        minus_di = values["minus_di"]

        thresholds = self.config.get("thresholds", {})
        scoring = self.config.get("scoring", {})

        weak = float(thresholds.get("weak", 30))
        moderate = float(thresholds.get("moderate", 50))

        weak_mult = float(scoring.get("weak_multiplier", 0.4))
        mod_mult = float(scoring.get("moderate_multiplier", 0.75))
        strong_mult = float(scoring.get("strong_multiplier", 1.0))
        max_spread = float(scoring.get("max_directional_spread", 40))

        # Directional score: 0 = fully bearish, 10 = fully bullish
        di_spread = plus_di - minus_di
        directional_score = self._linear_score(
            di_spread, -max_spread, max_spread, 0.0, 10.0
        )

        # Pull toward neutral (5.0) based on trend strength
        if adx < weak:
            multiplier = weak_mult
        elif adx < moderate:
            multiplier = self._linear_score(adx, weak, moderate, weak_mult, mod_mult)
        else:
            multiplier = self._linear_score(adx, moderate, moderate * 2, mod_mult, strong_mult)
            multiplier = min(multiplier, strong_mult)

        score = 5.0 + (directional_score - 5.0) * multiplier
        return self._clamp(score)

    def summary(self, values: dict[str, Any], score: float) -> dict[str, Any]:
        return {
            "value_str": f"{values['adx']:.1f} ({values['trend_strength']})",
            "detail_str": (
                f"+DI: {values['plus_di']:.1f} / -DI: {values['minus_di']:.1f} "
                f"| {values['trend_direction']}"
            ),
        }
