"""
patterns/candlesticks.py — Candlestick pattern detection.

Detects common single-bar, two-bar, and three-bar candlestick patterns:

Single-bar:
  - Doji: open ≈ close relative to bar range (indecision)
  - Dragonfly Doji: doji with long lower shadow, tiny upper shadow (bullish)
  - Gravestone Doji: doji with long upper shadow, tiny lower shadow (bearish)
  - Hammer / Hanging Man: small body at top, long lower shadow
  - Shooting Star / Inverted Hammer: small body at bottom, long upper shadow
  - Bullish Marubozu: large bullish body with no/tiny shadows (strong conviction)
  - Bearish Marubozu: large bearish body with no/tiny shadows (strong conviction)

Two-bar:
  - Bullish Engulfing: bullish bar completely engulfs prior bearish bar
  - Bearish Engulfing: bearish bar completely engulfs prior bullish bar
  - Bullish Harami: small bullish bar contained within prior large bearish bar
  - Bearish Harami: small bearish bar contained within prior large bullish bar
  - Tweezer Top: two bars with matching highs at a peak (bearish reversal)
  - Tweezer Bottom: two bars with matching lows at a trough (bullish reversal)

Three-bar:
  - Morning Star: bearish → doji/small → bullish (bullish reversal)
  - Evening Star: bullish → doji/small → bearish (bearish reversal)
  - Three White Soldiers: three consecutive strong bullish bars (continuation)
  - Three Black Crows: three consecutive strong bearish bars (continuation)

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
        # Marubozu: body must fill >= this fraction of bar range, shadows <= max
        marubozu_body_min = float(self.config.get("marubozu_body_min", 0.90))
        marubozu_shadow_max = float(self.config.get("marubozu_shadow_max", 0.05))
        # Tweezer: highs/lows match within this fraction of bar range
        tweezer_tolerance = float(self.config.get("tweezer_tolerance", 0.002))
        # Morning/Evening Star: middle bar body must be <= this fraction of avg
        star_middle_body_max = float(self.config.get("star_middle_body_max", 0.30))
        # Three Soldiers/Crows: each bar body must be >= this fraction of range
        soldiers_body_min = float(self.config.get("soldiers_body_min", 0.60))
        soldiers_shadow_max = float(self.config.get("soldiers_shadow_max", 0.30))
        # Hammer/shooting star body_pct threshold
        hammer_body_max = float(self.config.get("hammer_body_max", 0.35))
        # Morning/Evening star bar body threshold (fraction of range)
        star_body_min = float(self.config.get("star_body_min", 0.5))

        # Pattern strength values: (with_trend, against_trend_or_default)
        _sv = self.config.get("strength_values", {})

        def _s(key: str, default_with: float, default_against: float) -> tuple[float, float]:
            v = _sv.get(key, {})
            return (float(v.get("with_trend", default_with)),
                    float(v.get("against_trend", default_against)))

        s_dragonfly = _s("dragonfly_doji", 0.9, 0.6)
        s_gravestone = _s("gravestone_doji", 0.9, 0.6)
        s_doji_dir = float(_sv.get("doji_directional", {}).get("value", 0.7) if isinstance(_sv.get("doji_directional"), dict) else _sv.get("doji_directional", 0.7))
        s_doji_neutral = float(_sv.get("doji_neutral", {}).get("value", 0.3) if isinstance(_sv.get("doji_neutral"), dict) else _sv.get("doji_neutral", 0.3))
        s_hammer = _s("hammer", 1.0, 0.8)
        s_shooting_star = _s("shooting_star", 1.0, 0.8)
        s_marubozu = _s("marubozu", 1.0, 0.7)
        s_engulfing = _s("engulfing", 1.0, 0.5)
        s_harami = _s("harami", 0.8, 0.4)
        s_tweezer = _s("tweezer", 0.9, 0.6)
        s_star = _s("star", 1.2, 0.6)
        s_soldiers = _s("soldiers_crows", 1.3, 0.9)

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
            is_doji = body_pct <= doji_threshold
            if is_doji:
                # Dragonfly doji: long lower shadow, tiny upper shadow
                # Inherently bullish (buyers pushed price back up)
                if lower_shadow >= bar_range * dragonfly_shadow_min and upper_shadow <= bar_range * doji_tiny_shadow_max:
                    strength = s_dragonfly[0] if in_downtrend else s_dragonfly[1]
                    detected.append(("dragonfly_doji", "bullish", strength))
                # Gravestone doji: long upper shadow, tiny lower shadow
                # Inherently bearish (sellers pushed price back down)
                elif upper_shadow >= bar_range * gravestone_shadow_min and lower_shadow <= bar_range * doji_tiny_shadow_max:
                    strength = s_gravestone[0] if in_uptrend else s_gravestone[1]
                    detected.append(("gravestone_doji", "bearish", strength))
                # Generic doji: shadows balanced or short
                elif in_uptrend:
                    detected.append(("doji", "bearish", s_doji_dir))
                elif in_downtrend:
                    detected.append(("doji", "bullish", s_doji_dir))
                else:
                    detected.append(("doji", "neutral", s_doji_neutral))

            # --- Hammer / Hanging Man ---
            # Small body at top, long lower shadow
            # Skip if already classified as doji (near-zero body bars should
            # not double-trigger as both doji and hammer).
            if not is_doji and body_pct < hammer_body_max and lower_shadow >= body * shadow_ratio and upper_shadow < body:
                if in_downtrend:
                    # Hammer = bullish reversal signal
                    detected.append(("hammer", "bullish", s_hammer[0]))
                elif in_uptrend:
                    # Hanging man = bearish reversal signal
                    detected.append(("hanging_man", "bearish", s_hammer[1]))

            # --- Shooting Star / Inverted Hammer ---
            # Small body at bottom, long upper shadow
            # Skip if already classified as doji.
            if not is_doji and body_pct < hammer_body_max and upper_shadow >= body * shadow_ratio and lower_shadow < body:
                if in_uptrend:
                    # Shooting star = bearish reversal signal
                    detected.append(("shooting_star", "bearish", s_shooting_star[0]))
                elif in_downtrend:
                    # Inverted hammer = bullish reversal signal
                    detected.append(("inverted_hammer", "bullish", s_shooting_star[1]))

            # --- Marubozu ---
            # Full-body candle with no/tiny shadows — shows strong conviction
            if body_pct >= marubozu_body_min and bar_range > 0:
                upper_shadow_pct = upper_shadow / bar_range
                lower_shadow_pct = lower_shadow / bar_range
                if upper_shadow_pct <= marubozu_shadow_max and lower_shadow_pct <= marubozu_shadow_max:
                    if is_bullish_bar:
                        strength = s_marubozu[0] if in_uptrend else s_marubozu[1]
                        detected.append(("bullish_marubozu", "bullish", strength))
                    elif is_bearish_bar:
                        strength = s_marubozu[0] if in_downtrend else s_marubozu[1]
                        detected.append(("bearish_marubozu", "bearish", strength))

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
                    strength = s_engulfing[0] if in_downtrend else s_engulfing[1]
                    detected.append(("bullish_engulfing", "bullish", strength))

                # Bearish engulfing: prev bullish, curr bearish, curr body engulfs prev body
                if (prev_is_bullish and is_bearish_bar
                        and o >= prev_c and c <= prev_o
                        and body > prev_body):
                    strength = s_engulfing[0] if in_uptrend else s_engulfing[1]
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
                    strength = s_harami[0] if in_downtrend else s_harami[1]
                    detected.append(("bullish_harami", "bullish", strength))

                # Bearish harami: prev bullish with large body, curr bearish with small
                # body fully inside prev body
                if (prev_is_bullish and is_bearish_bar
                        and prev_body > 0
                        and body <= prev_body * harami_body_ratio
                        and min(o, c) >= min(prev_o, prev_c)
                        and max(o, c) <= max(prev_o, prev_c)):
                    strength = s_harami[0] if in_uptrend else s_harami[1]
                    detected.append(("bearish_harami", "bearish", strength))

                # --- Tweezer Patterns ---
                # Two consecutive bars with matching extremes → failed breakout / reversal
                prev_h = float(df["high"].iloc[i - 1])
                prev_l = float(df["low"].iloc[i - 1])
                prev_price = max(prev_h, float(df["close"].iloc[i - 1]))
                tol = prev_price * tweezer_tolerance if prev_price > 0 else 0.01

                # Tweezer top: two bars with nearly matching highs in an uptrend
                if abs(h - prev_h) <= tol and in_uptrend:
                    strength = s_tweezer[0] if (prev_is_bullish and is_bearish_bar) else s_tweezer[1]
                    detected.append(("tweezer_top", "bearish", strength))

                # Tweezer bottom: two bars with nearly matching lows in a downtrend
                if abs(l - prev_l) <= tol and in_downtrend:
                    strength = s_tweezer[0] if (prev_is_bearish and is_bullish_bar) else s_tweezer[1]
                    detected.append(("tweezer_bottom", "bullish", strength))

            # --- Three-bar patterns (need 2 previous bars) ---
            if i >= 2:
                bar0_o = float(df["open"].iloc[i - 2])
                bar0_h = float(df["high"].iloc[i - 2])
                bar0_l = float(df["low"].iloc[i - 2])
                bar0_c = float(df["close"].iloc[i - 2])
                bar0_range = bar0_h - bar0_l
                bar0_body = abs(bar0_c - bar0_o)

                bar1_o = float(df["open"].iloc[i - 1])
                bar1_h = float(df["high"].iloc[i - 1])
                bar1_l = float(df["low"].iloc[i - 1])
                bar1_c = float(df["close"].iloc[i - 1])
                bar1_range = bar1_h - bar1_l
                bar1_body = abs(bar1_c - bar1_o)

                bar2_body = body       # current bar (i)
                bar2_range = bar_range

                # --- Morning Star ---
                # Bar 0: strong bearish, Bar 1: small body (indecision), Bar 2: strong bullish
                # Bullish reversal, best when preceded by downtrend
                if (bar0_range > 0 and bar1_range > 0 and bar2_range > 0
                        and bar0_c < bar0_o                             # bar 0 bearish
                        and bar0_body / bar0_range >= star_body_min     # bar 0 has solid body
                        and bar1_body / bar1_range <= star_middle_body_max  # bar 1 small body
                        and c > o                                       # bar 2 bullish
                        and bar2_body / bar2_range >= star_body_min     # bar 2 has solid body
                        and c > (bar0_o + bar0_c) / 2):                 # bar 2 closes above bar 0 midpoint
                    strength = s_star[0] if in_downtrend else s_star[1]
                    detected.append(("morning_star", "bullish", strength))

                # --- Evening Star ---
                # Bar 0: strong bullish, Bar 1: small body (indecision), Bar 2: strong bearish
                # Bearish reversal, best when preceded by uptrend
                if (bar0_range > 0 and bar1_range > 0 and bar2_range > 0
                        and bar0_c > bar0_o                             # bar 0 bullish
                        and bar0_body / bar0_range >= star_body_min     # bar 0 has solid body
                        and bar1_body / bar1_range <= star_middle_body_max  # bar 1 small body
                        and c < o                                       # bar 2 bearish
                        and bar2_body / bar2_range >= star_body_min     # bar 2 has solid body
                        and c < (bar0_o + bar0_c) / 2):                 # bar 2 closes below bar 0 midpoint
                    strength = s_star[0] if in_uptrend else s_star[1]
                    detected.append(("evening_star", "bearish", strength))

                # --- Three White Soldiers ---
                # Three consecutive strong bullish bars, each opening within prior body
                # and closing higher than prior close. Small upper shadows.
                if (bar0_range > 0 and bar1_range > 0 and bar2_range > 0
                        and bar0_c > bar0_o                                  # bar 0 bullish
                        and bar1_c > bar1_o                                  # bar 1 bullish
                        and c > o                                            # bar 2 bullish
                        and bar0_body / bar0_range >= soldiers_body_min
                        and bar1_body / bar1_range >= soldiers_body_min
                        and bar2_body / bar2_range >= soldiers_body_min
                        and bar1_c > bar0_c                                  # each closes higher
                        and c > bar1_c
                        and bar1_o >= bar0_o and bar1_o <= bar0_c            # opens within prior body
                        and o >= bar1_o and o <= bar1_c):
                    # Check small upper shadows
                    bar0_upper = bar0_h - max(bar0_o, bar0_c)
                    bar1_upper = bar1_h - max(bar1_o, bar1_c)
                    bar2_upper = h - max(o, c)
                    if (bar0_range > 0 and bar0_upper / bar0_range <= soldiers_shadow_max
                            and bar1_range > 0 and bar1_upper / bar1_range <= soldiers_shadow_max
                            and bar2_range > 0 and bar2_upper / bar2_range <= soldiers_shadow_max):
                        strength = s_soldiers[0] if in_downtrend else s_soldiers[1]
                        detected.append(("three_white_soldiers", "bullish", strength))

                # --- Three Black Crows ---
                # Three consecutive strong bearish bars, each opening within prior body
                # and closing lower than prior close. Small lower shadows.
                if (bar0_range > 0 and bar1_range > 0 and bar2_range > 0
                        and bar0_c < bar0_o                                  # bar 0 bearish
                        and bar1_c < bar1_o                                  # bar 1 bearish
                        and c < o                                            # bar 2 bearish
                        and bar0_body / bar0_range >= soldiers_body_min
                        and bar1_body / bar1_range >= soldiers_body_min
                        and bar2_body / bar2_range >= soldiers_body_min
                        and bar1_c < bar0_c                                  # each closes lower
                        and c < bar1_c
                        and bar1_o <= bar0_o and bar1_o >= bar0_c            # opens within prior body
                        and o <= bar1_o and o >= bar1_c):
                    # Check small lower shadows
                    bar0_lower = min(bar0_o, bar0_c) - bar0_l
                    bar1_lower = min(bar1_o, bar1_c) - bar1_l
                    bar2_lower = min(o, c) - l
                    if (bar0_range > 0 and bar0_lower / bar0_range <= soldiers_shadow_max
                            and bar1_range > 0 and bar1_lower / bar1_range <= soldiers_shadow_max
                            and bar2_range > 0 and bar2_lower / bar2_range <= soldiers_shadow_max):
                        strength = s_soldiers[0] if in_uptrend else s_soldiers[1]
                        detected.append(("three_black_crows", "bearish", strength))

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
            "bullish_marubozu": "Bull Marubozu",
            "bearish_marubozu": "Bear Marubozu",
            "tweezer_top": "Tweezer Top",
            "tweezer_bottom": "Tweezer Bottom",
            "morning_star": "Morning Star",
            "evening_star": "Evening Star",
            "three_white_soldiers": "3 White Soldiers",
            "three_black_crows": "3 Black Crows",
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
