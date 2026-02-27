"""
patterns/inside_outside.py — Inside Bar and Outside Bar detection.

Inside Bar:
  Current bar's range is fully contained within the prior bar's range
  (high <= prev high AND low >= prev low). Indicates consolidation and
  a potential breakout setup. When detected, the inside bar is recorded
  as "pending" and we wait up to ``breakout_bars`` for a breakout:
    - Subsequent bar breaks above the mother bar's high → bullish
      (signal attributed to the breakout bar, not the inside bar)
    - Subsequent bar breaks below the mother bar's low → bearish
    - No breakout within the window → neutral (score = 5.0)

  This avoids look-ahead bias: the breakout direction is only known
  when the breakout actually occurs.

Outside Bar (a.k.a. engulfing range):
  Current bar's range completely engulfs the prior bar's range
  (high >= prev high AND low <= prev low). Indicates volatility expansion
  and directional conviction:
    - Bullish close on outside bar → bullish
    - Bearish close on outside bar → bearish

Scoring:
  Recent bullish breakouts / outside bars → score > 5
  Recent bearish breakouts / outside bars → score < 5
  No patterns → 5.0 (neutral)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import BasePattern


class InsideOutsidePattern(BasePattern):
    name = "Inside/Outside Bars"
    config_key = "inside_outside"

    def detect(self, df: pd.DataFrame) -> dict[str, Any]:
        lookback = int(self.config.get("lookback", 20))
        trend_period = int(self.config.get("trend_period", 10))
        # How many bars after an inside bar to check for breakout
        breakout_bars = int(self.config.get("breakout_bars", 3))
        # Minimum range expansion for outside bar (curr_range / prev_range)
        outside_range_min = float(self.config.get("outside_range_min", 1.2))

        # Strength values: (with_trend, against_trend)
        _sv = self.config.get("strength_values", {})
        s_inside_breakout_with = float(_sv.get("inside_breakout_with_trend", 1.0))
        s_inside_breakout_against = float(_sv.get("inside_breakout_against_trend", 0.6))
        s_inside_pending = float(_sv.get("inside_pending", 0.3))
        s_outside_reversal = float(_sv.get("outside_reversal", 1.0))
        s_outside_continuation = float(_sv.get("outside_continuation", 0.7))

        if len(df) < trend_period + 2:
            return {
                "patterns": [], "recent_patterns": [],
                "net_signal": 0.0, "total_patterns": 0, "recent_count": 0,
            }

        # Trend detection: EMA slope
        ema = df["close"].ewm(span=trend_period, adjust=False).mean()
        ema_slope = ema.diff()

        patterns: list[dict[str, Any]] = []

        # Track pending inside bar for breakout detection (no look-ahead).
        # When an inside bar is found, we record the mother bar's levels
        # and wait for a subsequent bar to break out.
        pending_inside: dict[str, Any] | None = None
        pending_bars_remaining: int = 0

        for i in range(1, len(df)):
            h = float(df["high"].iloc[i])
            l = float(df["low"].iloc[i])
            c = float(df["close"].iloc[i])
            o = float(df["open"].iloc[i])
            bar_range = h - l

            prev_h = float(df["high"].iloc[i - 1])
            prev_l = float(df["low"].iloc[i - 1])
            prev_range = prev_h - prev_l

            if bar_range == 0 or prev_range == 0:
                # Still count down pending inside bar even if this bar is skipped
                if pending_inside is not None:
                    pending_bars_remaining -= 1
                    if pending_bars_remaining <= 0:
                        # Expired without breakout — record as pending/neutral
                        patterns.append(pending_inside)
                        pending_inside = None
                continue

            slope = float(ema_slope.iloc[i]) if not np.isnan(ema_slope.iloc[i]) else 0.0
            in_uptrend = slope > 0
            in_downtrend = slope < 0

            # --- Check for breakout of pending inside bar ---
            if pending_inside is not None:
                mother_h = pending_inside["_mother_h"]
                mother_l = pending_inside["_mother_l"]
                ib_uptrend = pending_inside["_in_uptrend"]
                ib_downtrend = pending_inside["_in_downtrend"]

                if h > mother_h:
                    # Bullish breakout — emit signal on THIS bar (breakout bar)
                    strength = s_inside_breakout_with if ib_uptrend else s_inside_breakout_against
                    patterns.append({
                        "bar_index": i,
                        "date": str(df.index[i])[:10],
                        "pattern": "inside_bar_bullish",
                        "signal": "bullish",
                        "strength": strength,
                    })
                    pending_inside = None
                elif l < mother_l:
                    # Bearish breakout — emit signal on THIS bar (breakout bar)
                    strength = s_inside_breakout_with if ib_downtrend else s_inside_breakout_against
                    patterns.append({
                        "bar_index": i,
                        "date": str(df.index[i])[:10],
                        "pattern": "inside_bar_bearish",
                        "signal": "bearish",
                        "strength": strength,
                    })
                    pending_inside = None
                else:
                    pending_bars_remaining -= 1
                    if pending_bars_remaining <= 0:
                        # Expired without breakout — record as neutral
                        patterns.append(pending_inside)
                        pending_inside = None

            # --- Inside Bar ---
            # Current range fully contained within prior range.
            # Record as pending and wait for breakout (no look-ahead).
            if h <= prev_h and l >= prev_l:
                # If there's already a pending inside bar that hasn't
                # broken out, finalize it as neutral before tracking the new one.
                if pending_inside is not None:
                    patterns.append(pending_inside)

                pending_inside = {
                    "bar_index": i,
                    "date": str(df.index[i])[:10],
                    "pattern": "inside_bar_pending",
                    "signal": "neutral",
                    "strength": s_inside_pending,
                    # Internal fields (stripped before returning)
                    "_mother_h": prev_h,
                    "_mother_l": prev_l,
                    "_in_uptrend": in_uptrend,
                    "_in_downtrend": in_downtrend,
                }
                pending_bars_remaining = breakout_bars

            # --- Outside Bar ---
            # Current range engulfs prior range with meaningful expansion
            elif h >= prev_h and l <= prev_l and (bar_range / prev_range) >= outside_range_min:
                is_bullish_close = c > o
                is_bearish_close = c < o

                if is_bullish_close:
                    strength = s_outside_reversal if in_downtrend else s_outside_continuation
                    patterns.append({
                        "bar_index": i,
                        "date": str(df.index[i])[:10],
                        "pattern": "outside_bar_bullish",
                        "signal": "bullish",
                        "strength": strength,
                    })
                elif is_bearish_close:
                    strength = s_outside_reversal if in_uptrend else s_outside_continuation
                    patterns.append({
                        "bar_index": i,
                        "date": str(df.index[i])[:10],
                        "pattern": "outside_bar_bearish",
                        "signal": "bearish",
                        "strength": strength,
                    })

        # Finalize any still-pending inside bar at end of data
        if pending_inside is not None:
            patterns.append(pending_inside)
            pending_inside = None

        # Strip internal tracking fields from inside bar patterns
        for p in patterns:
            p.pop("_mother_h", None)
            p.pop("_mother_l", None)
            p.pop("_in_uptrend", None)
            p.pop("_in_downtrend", None)

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
                "detail_str": f"Total: {total} | No recent inside/outside bars",
            }

        last = recent[-1]
        pattern_labels = {
            "inside_bar_bullish": "Inside \u2191",
            "inside_bar_bearish": "Inside \u2193",
            "inside_bar_pending": "Inside \u25CB",
            "outside_bar_bullish": "Outside \u2191",
            "outside_bar_bearish": "Outside \u2193",
        }
        pat_label = pattern_labels.get(last["pattern"], last["pattern"])

        # Count by type in recent
        inside_count = sum(1 for p in recent if "inside" in p["pattern"])
        outside_count = sum(1 for p in recent if "outside" in p["pattern"])
        bullish_count = sum(1 for p in recent if p["signal"] == "bullish")
        bearish_count = sum(1 for p in recent if p["signal"] == "bearish")

        return {
            "value_str": f"{pat_label} ({len(recent)} recent)",
            "detail_str": (
                f"Last: {last['date']} {pat_label} | "
                f"Inside: {inside_count} Outside: {outside_count} | "
                f"{bullish_count}\u2191 {bearish_count}\u2193 | Total: {total}"
            ),
        }
