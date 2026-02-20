"""
indicators/fibonacci.py — Fibonacci Retracement indicator.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseIndicator


class FibonacciIndicator(BaseIndicator):
    name = "Fibonacci"
    config_key = "fibonacci"

    def compute(self, df: pd.DataFrame) -> dict[str, Any]:
        lookback = int(self.config.get("swing_lookback", 60))
        levels_cfg: list[float] = self.config.get("levels", [0.236, 0.382, 0.5, 0.618, 0.786])

        window = df.tail(lookback)
        swing_high = float(window["high"].max())
        swing_low = float(window["low"].min())
        price = float(df["close"].iloc[-1])
        swing_range = swing_high - swing_low

        # Calculate retracement levels (from high down to low)
        fib_levels: dict[float, float] = {}
        for lvl in levels_cfg:
            fib_levels[lvl] = swing_high - lvl * swing_range

        # Find which level price is nearest to
        proximity_pct = float(self.config.get("scoring", {}).get("proximity_pct", 0.015))
        nearest_level: float | None = None
        nearest_distance = float("inf")

        for lvl, lvl_price in fib_levels.items():
            dist = abs(price - lvl_price) / price if price != 0 else float("inf")
            if dist < nearest_distance:
                nearest_distance = dist
                nearest_level = lvl

        at_level = nearest_distance <= proximity_pct

        # Trend context: is price in upper or lower half of the range?
        range_position = (price - swing_low) / swing_range if swing_range != 0 else 0.5

        return {
            "price": price,
            "swing_high": swing_high,
            "swing_low": swing_low,
            "fib_levels": fib_levels,
            "nearest_level": nearest_level,
            "nearest_distance_pct": nearest_distance * 100,
            "at_level": at_level,
            "range_position": range_position,
        }

    def score(self, values: dict[str, Any]) -> float:
        scoring = self.config.get("scoring", {})
        level_scores: dict = scoring.get("level_scores", {})
        no_level_score = float(scoring.get("no_level_score", 5.0))

        if not values["at_level"] or values["nearest_level"] is None:
            # Not near any level — score based on range position
            # Upper half of range is slightly bullish, lower half slightly bearish
            range_low = float(scoring.get("range_low_score", 2.0))
            range_high = float(scoring.get("range_high_score", 8.0))
            return self._linear_score(values["range_position"], 0.0, 1.0, range_low, range_high)

        lvl = values["nearest_level"]
        # level_scores keys may be floats or strings depending on YAML parsing
        raw_score = None
        for k, v in level_scores.items():
            if abs(float(k) - lvl) < 0.001:
                raw_score = float(v)
                break

        if raw_score is None:
            return no_level_score

        return self._clamp(raw_score)

    def summary(self, values: dict[str, Any], score: float) -> dict[str, Any]:
        fib_levels = values["fib_levels"]
        price = values["price"]

        # Build compact level string
        level_parts = []
        for lvl, lvl_price in sorted(fib_levels.items()):
            marker = " <--" if (values["at_level"] and abs(lvl - (values["nearest_level"] or -1)) < 0.001) else ""
            level_parts.append(f"{lvl:.3f}: ${lvl_price:.2f}{marker}")

        near_str = (
            f"Near {values['nearest_level']:.3f} level"
            if values["at_level"]
            else f"Range pos: {values['range_position']:.0%}"
        )

        return {
            "value_str": f"H: ${values['swing_high']:.2f} / L: ${values['swing_low']:.2f}",
            "detail_str": near_str + f" | {' | '.join(level_parts[:3])}",
        }
