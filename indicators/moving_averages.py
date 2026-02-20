"""
indicators/moving_averages.py — Simple/Exponential Moving Averages indicator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import BaseIndicator


class MovingAveragesIndicator(BaseIndicator):
    name = "Moving Averages"
    config_key = "moving_averages"

    def compute(self, df: pd.DataFrame) -> dict[str, Any]:
        periods: list[int] = self.config.get("periods", [20, 50, 200])
        ma_type: str = self.config.get("type", "sma").lower()
        price = float(df["close"].iloc[-1])

        ma_values: dict[int, float] = {}
        for p in periods:
            if len(df) < p:
                ma_values[p] = float("nan")
                continue
            if ma_type == "ema":
                ma_values[p] = float(df["close"].ewm(span=p, adjust=False).mean().iloc[-1])
            else:
                ma_values[p] = float(df["close"].rolling(window=p).mean().iloc[-1])

        # Golden / death cross detection (between the two longest MAs)
        cross_lookback = int(self.config.get("scoring", {}).get("cross_lookback", 10))
        golden_cross = False
        death_cross = False

        sorted_periods = sorted(p for p in periods if not np.isnan(ma_values.get(p, float("nan"))))
        if len(sorted_periods) >= 2:
            short_p = sorted_periods[-2]
            long_p = sorted_periods[-1]
            if len(df) >= long_p + cross_lookback:
                if ma_type == "ema":
                    short_series = df["close"].ewm(span=short_p, adjust=False).mean()
                    long_series = df["close"].ewm(span=long_p, adjust=False).mean()
                else:
                    short_series = df["close"].rolling(window=short_p).mean()
                    long_series = df["close"].rolling(window=long_p).mean()

                diff = (short_series - long_series).dropna()
                if len(diff) >= cross_lookback + 1:
                    recent_diff = diff.values[-(cross_lookback + 1):]
                    for i in range(1, len(recent_diff)):
                        if recent_diff[i - 1] < 0 and recent_diff[i] > 0:
                            golden_cross = True
                        if recent_diff[i - 1] > 0 and recent_diff[i] < 0:
                            death_cross = True

        return {
            "price": price,
            "ma_values": ma_values,
            "periods": periods,
            "ma_type": ma_type.upper(),
            "golden_cross": golden_cross,
            "death_cross": death_cross,
        }

    def score(self, values: dict[str, Any]) -> float:
        price = values["price"]
        ma_values = values["ma_values"]
        scoring = self.config.get("scoring", {})

        above_pts = float(scoring.get("price_above_ma_points", 1.5))
        aligned_pts = float(scoring.get("ma_aligned_bullish_points", 1.0))
        golden_bonus = float(scoring.get("golden_cross_bonus", 2.0))
        death_penalty = float(scoring.get("death_cross_penalty", 2.0))
        max_raw = float(scoring.get("max_raw_score", 9.5))

        raw = 0.0
        valid_mas = {p: v for p, v in ma_values.items() if not np.isnan(v)}

        # Points for price above each MA
        for ma_val in valid_mas.values():
            if price > ma_val:
                raw += above_pts

        # Points for bullish MA alignment (shorter MA above longer MA)
        sorted_mas = sorted(valid_mas.items())
        for i in range(len(sorted_mas) - 1):
            p_short, v_short = sorted_mas[i]
            p_long, v_long = sorted_mas[i + 1]
            if v_short > v_long:
                raw += aligned_pts

        # Golden / death cross
        if values["golden_cross"]:
            raw += golden_bonus
        elif values["death_cross"]:
            raw -= death_penalty

        # Normalize to 0-10
        # Max possible raw (all above + all aligned + golden cross)
        n = len(valid_mas)
        theoretical_max = n * above_pts + max(0, n - 1) * aligned_pts + golden_bonus
        if theoretical_max == 0:
            return 5.0

        normalized = (raw / theoretical_max) * max_raw
        return self._clamp(normalized)

    def summary(self, values: dict[str, Any], score: float) -> dict[str, Any]:
        price = values["price"]
        ma_values = values["ma_values"]
        ma_type = values["ma_type"]

        parts = []
        for p, v in sorted(ma_values.items()):
            if np.isnan(v):
                parts.append(f"{ma_type}{p}: N/A")
            else:
                arrow = "↑" if price > v else "↓"
                parts.append(f"{ma_type}{p}: {v:.2f}{arrow}")

        cross_note = ""
        if values["golden_cross"]:
            cross_note = " | Golden Cross"
        elif values["death_cross"]:
            cross_note = " | Death Cross"

        return {
            "value_str": " | ".join(parts[:2]),
            "detail_str": (f"{'  '.join(parts[2:]) if len(parts) > 2 else ''}{cross_note}").strip(),
        }
