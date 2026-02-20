"""
indicators/rsi.py — Relative Strength Index indicator.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import ta.momentum

from .base import BaseIndicator


class RSIIndicator(BaseIndicator):
    name = "RSI"
    config_key = "rsi"

    def compute(self, df: pd.DataFrame) -> dict[str, Any]:
        period = int(self.config.get("period", 14))
        rsi_series = ta.momentum.RSIIndicator(
            close=df["close"], window=period
        ).rsi()
        current = float(rsi_series.iloc[-1])
        prev = float(rsi_series.iloc[-2]) if len(rsi_series) > 1 else current
        return {
            "rsi": current,
            "rsi_prev": prev,
            "period": period,
            "series": rsi_series,
        }

    def score(self, values: dict[str, Any]) -> float:
        rsi = values["rsi"]
        thresholds = self.config.get("thresholds", {})
        scores_cfg = self.config.get("scores", {})

        oversold = float(thresholds.get("oversold", 30))
        overbought = float(thresholds.get("overbought", 70))
        oversold_score = float(scores_cfg.get("oversold_score", 9.0))
        overbought_score = float(scores_cfg.get("overbought_score", 1.0))
        neutral_score = float(scores_cfg.get("neutral_score", 5.0))

        if rsi <= oversold:
            return oversold_score
        if rsi >= overbought:
            return overbought_score
        if rsi <= 50:
            # oversold → neutral
            return self._linear_score(rsi, oversold, 50, oversold_score, neutral_score)
        else:
            # neutral → overbought
            return self._linear_score(rsi, 50, overbought, neutral_score, overbought_score)

    def summary(self, values: dict[str, Any], score: float) -> dict[str, Any]:
        rsi = values["rsi"]
        thresholds = self.config.get("thresholds", {})
        oversold = float(thresholds.get("oversold", 30))
        overbought = float(thresholds.get("overbought", 70))

        if rsi <= oversold:
            zone = "Oversold"
        elif rsi >= overbought:
            zone = "Overbought"
        elif rsi < 45:
            zone = "Bearish"
        elif rsi > 55:
            zone = "Bullish"
        else:
            zone = "Neutral"

        return {
            "value_str": f"{rsi:.1f}",
            "detail_str": f"Period: {values['period']} | Zone: {zone}",
        }
