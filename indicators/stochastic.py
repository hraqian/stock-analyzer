"""
indicators/stochastic.py — Stochastic Oscillator indicator.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import ta.momentum

from .base import BaseIndicator


class StochasticIndicator(BaseIndicator):
    name = "Stochastic"
    config_key = "stochastic"

    def compute(self, df: pd.DataFrame) -> dict[str, Any]:
        k_period = int(self.config.get("k_period", 14))
        d_period = int(self.config.get("d_period", 3))
        smooth_k = int(self.config.get("smooth_k", 3))

        stoch = ta.momentum.StochasticOscillator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=k_period,
            smooth_window=smooth_k,
        )
        k_series = stoch.stoch()
        d_series = stoch.stoch_signal()

        k = float(k_series.iloc[-1])
        d = float(d_series.iloc[-1])
        k_prev = float(k_series.iloc[-2]) if len(k_series) > 1 else k
        d_prev = float(d_series.iloc[-2]) if len(d_series) > 1 else d

        bullish_cross = (k_prev < d_prev) and (k > d)
        bearish_cross = (k_prev > d_prev) and (k < d)

        return {
            "k": k,
            "d": d,
            "bullish_cross": bullish_cross,
            "bearish_cross": bearish_cross,
            "k_period": k_period,
            "d_period": d_period,
        }

    def score(self, values: dict[str, Any]) -> float:
        k = values["k"]
        thresholds = self.config.get("thresholds", {})
        scores_cfg = self.config.get("scores", {})

        oversold = float(thresholds.get("oversold", 20))
        overbought = float(thresholds.get("overbought", 80))
        oversold_score = float(scores_cfg.get("oversold_score", 9.0))
        overbought_score = float(scores_cfg.get("overbought_score", 1.0))
        neutral_score = float(scores_cfg.get("neutral_score", 5.0))
        bull_bonus = float(scores_cfg.get("bullish_cross_bonus", 1.0))
        bear_penalty = float(scores_cfg.get("bearish_cross_penalty", 1.0))

        if k <= oversold:
            base = oversold_score
        elif k >= overbought:
            base = overbought_score
        elif k <= 50:
            base = self._linear_score(k, oversold, 50, oversold_score, neutral_score)
        else:
            base = self._linear_score(k, 50, overbought, neutral_score, overbought_score)

        if values["bullish_cross"]:
            base = min(10.0, base + bull_bonus)
        elif values["bearish_cross"]:
            base = max(0.0, base - bear_penalty)

        return self._clamp(base)

    def summary(self, values: dict[str, Any], score: float) -> dict[str, Any]:
        k = values["k"]
        d = values["d"]
        thresholds = self.config.get("thresholds", {})
        oversold = float(thresholds.get("oversold", 20))
        overbought = float(thresholds.get("overbought", 80))

        if k <= oversold:
            zone = "Oversold"
        elif k >= overbought:
            zone = "Overbought"
        else:
            zone = "Neutral"

        cross_note = ""
        if values["bullish_cross"]:
            cross_note = " | %K crossed above %D"
        elif values["bearish_cross"]:
            cross_note = " | %K crossed below %D"

        return {
            "value_str": f"%K: {k:.1f} / %D: {d:.1f}",
            "detail_str": f"Zone: {zone}{cross_note}",
        }
