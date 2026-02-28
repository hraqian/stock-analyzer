"""
indicators/bollinger.py — Bollinger Bands indicator.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import ta.volatility

from .base import BaseIndicator


class BollingerBandsIndicator(BaseIndicator):
    name = "Bollinger Bands"
    config_key = "bollinger_bands"

    def compute(self, df: pd.DataFrame) -> dict[str, Any]:
        period = int(self.config.get("period", 20))
        std_dev = float(self.config.get("std_dev", 2.0))

        bb = ta.volatility.BollingerBands(
            close=df["close"],
            window=period,
            window_dev=std_dev,
        )
        upper = float(bb.bollinger_hband().iloc[-1])
        middle = float(bb.bollinger_mavg().iloc[-1])
        lower = float(bb.bollinger_lband().iloc[-1])
        price = float(df["close"].iloc[-1])

        band_width = upper - lower
        pct_b = (price - lower) / band_width if band_width != 0 else 0.5
        squeeze = (band_width / middle) < float(
            self.config.get("scoring", {}).get("squeeze_threshold", 0.02)
        ) if middle != 0 else False

        return {
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "price": price,
            "pct_b": pct_b,
            "band_width": band_width,
            "squeeze": squeeze,
            "period": period,
            "std_dev": std_dev,
        }

    def score(self, values: dict[str, Any]) -> float:
        pct_b = values["pct_b"]
        scoring = self.config.get("scoring", {})
        lower_zone = float(scoring.get("lower_zone", 0.20))
        upper_zone = float(scoring.get("upper_zone", 0.80))

        if pct_b <= 0.0:
            return 9.5
        if pct_b <= lower_zone:
            return self._linear_score(pct_b, 0.0, lower_zone, 9.5, 7.5)
        if pct_b <= 0.5:
            return self._linear_score(pct_b, lower_zone, 0.5, 7.5, 5.0)
        if pct_b <= upper_zone:
            return self._linear_score(pct_b, 0.5, upper_zone, 5.0, 2.5)
        if pct_b <= 1.0:
            return self._linear_score(pct_b, upper_zone, 1.0, 2.5, 0.5)
        return 0.5

    def summary(self, values: dict[str, Any], score: float) -> dict[str, Any]:
        pct_b = values["pct_b"]
        squeeze_note = " | SQUEEZE" if values["squeeze"] else ""
        if pct_b <= 0.2:
            zone = "Near lower band"
        elif pct_b >= 0.8:
            zone = "Near upper band"
        else:
            zone = "Mid-band"
        return {
            "value_str": (
                f"U:{values['upper']:.2f} M:{values['middle']:.2f} L:{values['lower']:.2f}"
            ),
            "detail_str": f"%B: {pct_b:.2f} | {zone}{squeeze_note}",
        }
