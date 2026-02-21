"""
patterns/candlesticks.py — Candlestick pattern detection.

Detects common single-bar and two-bar candlestick patterns:
  - Doji: open ≈ close relative to bar range (indecision)
  - Dragonfly Doji: doji with long lower shadow, tiny upper shadow (bullish)
  - Gravestone Doji: doji with long upper shadow, tiny lower shadow (bearish)
  - Hammer / Hanging Man: small body at top, long lower shadow
  - Shooting Star / Inverted Hammer: small body at bottom, long upper shadow
  - Bullish Engulfing: bullish bar completely engulfs prior bearish bar
  - Bearish Engulfing: bearish bar completely engulfs prior bullish bar
  - Bullish Harami: small bullish bar contained within prior large bearish bar
  - Bearish Harami: small bearish bar contained within prior large bullish bar

Context matters: a doji after an uptrend is bearish (reversal), after a
downtrend is bullish (reversal). Dragonfly and gravestone doji carry inherent
directional bias from their shadow structure, amplified when trend-confirmed.

Scoring:
  Recent bullish patterns → score > 5
  Recent bearish patterns → score < 5
  No patterns → 5.0 (neutral)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import BasePattern


class CandlestickPattern(BasePattern):
    name = "Candlesticks"
    config_key = "candlesticks"

    def detect(self, df: pd.DataFrame) -> dict[str, Any]:
        doji_threshold = float(self.config.get("doji_threshold", 0.05))
        shadow_ratio = float(self.config.get("shadow_ratio", 2.0))
        dragonfly_shadow_min = float(self.config.get("dragonfly_shadow_min", 0.6))
        gravestone_shadow_min = float(self.config.get("gravestone_shadow_min", 0.6))
        doji_tiny_shadow_max = float(self.config.get("doji_tiny_shadow_max", 0.1))
        lookback = int(self.config.get("lookback", 10))
        trend_period = int(self.config.get("trend_period", 10))

        if len(df) < trend_period + 2:
            return {"patterns": [], "recent_patterns": [], "net_signal": 0.0}

        # Trend detection: simple EMA slope
        ema = df["close"].ewm(span=trend_period, adjust=False).mean()
        ema_slope = ema.diff()

        patterns = []

        for i in range(1, len(df)):
            o = float(df["open"].iloc[i])
            h = float(df["high"].iloc[i])
            l = float(df["low"].iloc[i])
            c = float(df["close"].iloc[i])
            bar_range = h - l

            if bar_range == 0:
                continue

            body = abs(c - o)
            body_pct = body / bar_range
            upper_shadow = h - max(o, c)
            lower_shadow = min(o, c) - l
            is_bullish_bar = c > o
            is_bearish_bar = c < o

            slope = float(ema_slope.iloc[i]) if not np.isnan(ema_slope.iloc[i]) else 0.0
            in_uptrend = slope > 0
            in_downtrend = slope < 0

            detected = []

            # --- Doji (generic, dragonfly, gravestone) ---
            if body_pct <= doji_threshold:
                # Dragonfly doji: long lower shadow, tiny upper shadow
                # Inherently bullish (buyers pushed price back up)
                if lower_shadow >= bar_range * dragonfly_shadow_min and upper_shadow <= bar_range * doji_tiny_shadow_max:
                    strength = 0.9 if in_downtrend else 0.6
                    detected.append(("dragonfly_doji", "bullish", strength))
                # Gravestone doji: long upper shadow, tiny lower shadow
                # Inherently bearish (sellers pushed price back down)
                elif upper_shadow >= bar_range * gravestone_shadow_min and lower_shadow <= bar_range * doji_tiny_shadow_max:
                    strength = 0.9 if in_uptrend else 0.6
                    detected.append(("gravestone_doji", "bearish", strength))
                # Generic doji: shadows balanced or short
                elif in_uptrend:
                    detected.append(("doji", "bearish", 0.7))
                elif in_downtrend:
                    detected.append(("doji", "bullish", 0.7))
                else:
                    detected.append(("doji", "neutral", 0.3))

            # --- Hammer / Hanging Man ---
            # Small body at top, long lower shadow
            if body_pct < 0.35 and lower_shadow >= body * shadow_ratio and upper_shadow < body:
                if in_downtrend:
                    # Hammer = bullish reversal signal
                    detected.append(("hammer", "bullish", 1.0))
                elif in_uptrend:
                    # Hanging man = bearish reversal signal
                    detected.append(("hanging_man", "bearish", 0.8))

            # --- Shooting Star / Inverted Hammer ---
            # Small body at bottom, long upper shadow
            if body_pct < 0.35 and upper_shadow >= body * shadow_ratio and lower_shadow < body:
                if in_uptrend:
                    # Shooting star = bearish reversal signal
                    detected.append(("shooting_star", "bearish", 1.0))
                elif in_downtrend:
                    # Inverted hammer = bullish reversal signal
                    detected.append(("inverted_hammer", "bullish", 0.8))

            # --- Engulfing Patterns (need previous bar) ---
            if i >= 1:
                prev_o = float(df["open"].iloc[i - 1])
                prev_c = float(df["close"].iloc[i - 1])
                prev_body = abs(prev_c - prev_o)
                prev_is_bullish = prev_c > prev_o
                prev_is_bearish = prev_c < prev_o

                # Bullish engulfing: prev bearish, curr bullish, curr body engulfs prev body
                if (prev_is_bearish and is_bullish_bar
                        and o <= prev_c and c >= prev_o
                        and body > prev_body):
                    strength = 1.0 if in_downtrend else 0.5
                    detected.append(("bullish_engulfing", "bullish", strength))

                # Bearish engulfing: prev bullish, curr bearish, curr body engulfs prev body
                if (prev_is_bullish and is_bearish_bar
                        and o >= prev_c and c <= prev_o
                        and body > prev_body):
                    strength = 1.0 if in_uptrend else 0.5
                    detected.append(("bearish_engulfing", "bearish", strength))

                # --- Harami Patterns (current body contained in previous body) ---
                harami_body_ratio = float(self.config.get("harami_body_ratio", 0.5))

                # Bullish harami: prev bearish with large body, curr bullish with small
                # body fully inside prev body
                if (prev_is_bearish and is_bullish_bar
                        and prev_body > 0
                        and body <= prev_body * harami_body_ratio
                        and min(o, c) >= min(prev_o, prev_c)
                        and max(o, c) <= max(prev_o, prev_c)):
                    strength = 0.8 if in_downtrend else 0.4
                    detected.append(("bullish_harami", "bullish", strength))

                # Bearish harami: prev bullish with large body, curr bearish with small
                # body fully inside prev body
                if (prev_is_bullish and is_bearish_bar
                        and prev_body > 0
                        and body <= prev_body * harami_body_ratio
                        and min(o, c) >= min(prev_o, prev_c)
                        and max(o, c) <= max(prev_o, prev_c)):
                    strength = 0.8 if in_uptrend else 0.4
                    detected.append(("bearish_harami", "bearish", strength))

            for pat_name, signal, strength in detected:
                patterns.append({
                    "bar_index": i,
                    "date": str(df.index[i])[:10],
                    "pattern": pat_name,
                    "signal": signal,
                    "strength": strength,
                })

        # Recent patterns only
        recent_patterns = [p for p in patterns if p["bar_index"] >= len(df) - lookback]

        # Compute net signal: weighted sum of recent pattern signals
        net_signal = 0.0
        for p in recent_patterns:
            bars_ago = len(df) - 1 - p["bar_index"]
            recency = max(0.1, 1.0 - (bars_ago / lookback))
            if p["signal"] == "bullish":
                net_signal += p["strength"] * recency
            elif p["signal"] == "bearish":
                net_signal -= p["strength"] * recency

        return {
            "patterns": patterns,
            "recent_patterns": recent_patterns,
            "net_signal": net_signal,
            "total_patterns": len(patterns),
            "recent_count": len(recent_patterns),
        }

    def score(self, values: dict[str, Any]) -> float:
        net = values["net_signal"]
        if not values["recent_patterns"]:
            return 5.0

        max_signal = float(self.config.get("max_signal_strength", 3.0))

        if net >= 0:
            return self._linear_score(net, 0, max_signal, 5.0, 9.5)
        else:
            return self._linear_score(net, -max_signal, 0, 0.5, 5.0)

    def summary(self, values: dict[str, Any], score: float) -> dict[str, Any]:
        recent = values["recent_patterns"]
        total = values["total_patterns"]

        if not recent:
            return {
                "value_str": "None detected",
                "detail_str": f"Total patterns: {total} | No recent patterns",
            }

        # Find the most recent significant pattern
        last = recent[-1]
        pattern_labels = {
            "doji": "Doji",
            "dragonfly_doji": "Dragonfly Doji",
            "gravestone_doji": "Gravestone Doji",
            "hammer": "Hammer",
            "hanging_man": "Hanging Man",
            "shooting_star": "Shooting Star",
            "inverted_hammer": "Inv Hammer",
            "bullish_engulfing": "Bull Engulf",
            "bearish_engulfing": "Bear Engulf",
            "bullish_harami": "Bull Harami",
            "bearish_harami": "Bear Harami",
        }
        pat_label = pattern_labels.get(last["pattern"], last["pattern"])
        signal_icon = "\u2191" if last["signal"] == "bullish" else ("\u2193" if last["signal"] == "bearish" else "\u25CB")

        # Count by type in recent
        bullish_count = sum(1 for p in recent if p["signal"] == "bullish")
        bearish_count = sum(1 for p in recent if p["signal"] == "bearish")

        return {
            "value_str": f"{pat_label} {signal_icon} ({len(recent)} recent)",
            "detail_str": (
                f"Last: {last['date']} {pat_label} ({last['signal']}) | "
                f"Recent: {bullish_count}\u2191 {bearish_count}\u2193 | Total: {total}"
            ),
        }
