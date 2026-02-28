"""
patterns/gaps.py — Gap detection and classification.

Detects price gaps between consecutive bars and classifies them:
  - Common gap:     small gap, no special context (neutral)
  - Breakaway gap:  gap out of a prior consolidation with high volume (directional)
  - Runaway gap:    gap in direction of an existing trend (continuation)
  - Exhaustion gap: gap at the end of an extended trend with high volume (reversal warning)

Classification follows standard TA definitions (Murphy, Bulkowski):
  - Breakaway: requires prior consolidation (low ATR% relative to recent history)
    plus a volume surge.  The gap breaks out of the range — directional signal.
  - Exhaustion: requires an extended prior trend (consecutive same-direction bars)
    plus a volume surge and a gap *with* that trend.  Reversal warning.

Scoring:
  Recent bullish gaps → score > 5 (bullish)
  Recent bearish gaps → score < 5 (bearish)
  Exhaustion gaps get inverse scoring (gap up = bearish warning, gap down = bullish).
  No gaps → 5.0 (neutral)

Gaps are less meaningful on intraday data (overnight gaps don't appear
between intraday bars), but session open gaps are still detected.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import BasePattern


class GapPattern(BasePattern):
    name = "Gaps"
    config_key = "gaps"

    def detect(self, df: pd.DataFrame) -> dict[str, Any]:
        lookback = int(self.config.get("lookback", 20))
        min_gap_pct = float(self.config.get("min_gap_pct", 0.005))
        volume_surge_mult = float(self.config.get("volume_surge_mult", 1.5))
        trend_period = int(self.config.get("trend_period", 20))

        # New classifier params
        consolidation_lookback = int(self.config.get("consolidation_lookback", 20))
        consolidation_atr_percentile = float(self.config.get("consolidation_atr_percentile", 50))
        trend_extension_bars = int(self.config.get("trend_extension_bars", 10))

        if len(df) < trend_period + 2:
            return {"gaps": [], "recent_gaps": [], "net_gap_score": 0.0}

        # Compute average volume for surge detection
        avg_volume = df["volume"].rolling(window=trend_period).mean()

        # Simple trend detection: EMA slope
        ema = df["close"].ewm(span=trend_period, adjust=False).mean()
        ema_slope = ema.diff()

        # ATR% series for consolidation detection
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=min(trend_period, 14)).mean()
        atr_pct = atr / df["close"]

        # Rolling percentile rank of ATR% (how current ATR% compares to
        # its own recent history).  Low rank = consolidation.
        atr_pct_rank = atr_pct.rolling(window=consolidation_lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
            if len(x) >= 2 else 50.0,
            raw=False,
        )

        # Consecutive same-direction EMA slope bars (for extended trend detection)
        slope_sign = np.sign(ema_slope.values)
        consec_trend = np.zeros(len(df), dtype=int)
        for i in range(1, len(df)):
            if slope_sign[i] == slope_sign[i - 1] and slope_sign[i] != 0:
                consec_trend[i] = consec_trend[i - 1] + 1
            elif slope_sign[i] != 0:
                consec_trend[i] = 1
            else:
                consec_trend[i] = 0

        gaps = []
        for i in range(1, len(df)):
            prev_high = float(df["high"].iloc[i - 1])
            prev_low = float(df["low"].iloc[i - 1])
            prev_close = float(df["close"].iloc[i - 1])
            curr_open = float(df["open"].iloc[i])
            curr_volume = float(df["volume"].iloc[i])

            # Gap up: current open > previous high
            # Gap down: current open < previous low
            if curr_open > prev_high:
                gap_pct = (curr_open - prev_high) / prev_close
                direction = "up"
            elif curr_open < prev_low:
                gap_pct = (prev_low - curr_open) / prev_close
                direction = "down"
            else:
                continue

            if gap_pct < min_gap_pct:
                continue

            # Context for classification
            avg_vol_val = float(avg_volume.iloc[i]) if not np.isnan(avg_volume.iloc[i]) else curr_volume
            volume_surge = curr_volume > (avg_vol_val * volume_surge_mult)
            slope_val = float(ema_slope.iloc[i]) if not np.isnan(ema_slope.iloc[i]) else 0.0

            # Trend alignment
            trending_up = slope_val > 0
            trending_down = slope_val < 0
            gap_with_trend = (direction == "up" and trending_up) or (direction == "down" and trending_down)

            # Consolidation: ATR% rank at the bar *before* the gap,
            # AND trend must be weak (not in a strong directional move).
            # True consolidation = low volatility + no clear trend direction.
            prior_atr_rank = float(atr_pct_rank.iloc[i - 1]) if (
                i >= 1 and not np.isnan(atr_pct_rank.iloc[i - 1])
            ) else 50.0
            low_volatility = prior_atr_rank <= consolidation_atr_percentile

            # Weak trend: few consecutive same-direction bars before the gap
            prior_consec = int(consec_trend[i - 1]) if i >= 1 else 0
            weak_trend = prior_consec < trend_extension_bars

            in_consolidation = low_volatility and weak_trend

            # Extended trend: consecutive same-direction slope bars before gap
            extended_trend = prior_consec >= trend_extension_bars

            # Classification (textbook-aligned)
            if volume_surge and in_consolidation:
                # Gap out of consolidation with volume = breakaway
                gap_type = "breakaway"
            elif volume_surge and extended_trend and gap_with_trend and gap_pct > min_gap_pct * 2:
                # Late-stage gap in direction of extended trend with volume = exhaustion
                gap_type = "exhaustion"
            elif gap_with_trend:
                gap_type = "runaway"
            else:
                gap_type = "common"

            gaps.append({
                "bar_index": i,
                "date": str(df.index[i])[:10],
                "direction": direction,
                "gap_pct": gap_pct,
                "gap_type": gap_type,
                "volume_surge": volume_surge,
            })

        # Only score recent gaps
        recent_gaps = [g for g in gaps if g["bar_index"] >= len(df) - lookback]

        # Compute net gap score: sum of weighted gap signals
        net_score = 0.0
        type_weights = self.config.get("type_weights", {
            "common": 0.3, "runaway": 0.7, "breakaway": 1.0, "exhaustion": 0.5,
        })
        gap_pct_scale = float(self.config.get("gap_pct_scale", 100))

        for g in recent_gaps:
            w = float(type_weights.get(g["gap_type"], 0.5))
            # Recency weighting: more recent gaps matter more
            bars_ago = len(df) - 1 - g["bar_index"]
            recency = max(0.1, 1.0 - (bars_ago / lookback))

            if g["direction"] == "up":
                if g["gap_type"] == "exhaustion":
                    # Exhaustion gap up = potential reversal DOWN (bearish)
                    net_score -= w * recency * g["gap_pct"] * gap_pct_scale
                else:
                    net_score += w * recency * g["gap_pct"] * gap_pct_scale
            else:
                if g["gap_type"] == "exhaustion":
                    # Exhaustion gap down = potential reversal UP (bullish)
                    net_score += w * recency * g["gap_pct"] * gap_pct_scale
                else:
                    net_score -= w * recency * g["gap_pct"] * gap_pct_scale

        return {
            "gaps": gaps,
            "recent_gaps": recent_gaps,
            "net_gap_score": net_score,
            "total_gaps": len(gaps),
            "recent_gap_count": len(recent_gaps),
        }

    def score(self, values: dict[str, Any]) -> float:
        net = values["net_gap_score"]
        if not values["recent_gaps"]:
            return 5.0

        # Map net_gap_score to 0-10 range
        # Positive net = bullish, negative = bearish
        max_signal = float(self.config.get("max_signal_strength", 3.0))
        if net >= 0:
            return self._linear_score(net, 0, max_signal, 5.0, 9.5)
        else:
            return self._linear_score(net, -max_signal, 0, 0.5, 5.0)

    def summary(self, values: dict[str, Any], score: float) -> dict[str, Any]:
        recent = values["recent_gaps"]
        total = values["total_gaps"]

        if not recent:
            return {
                "value_str": "None detected",
                "detail_str": f"Total gaps: {total} | No recent gaps",
            }

        # Summarize the most recent gap
        last = recent[-1]
        direction_arrow = "\u2191" if last["direction"] == "up" else "\u2193"
        type_label = last["gap_type"].capitalize()

        return {
            "value_str": f"{len(recent)} recent ({direction_arrow} {type_label})",
            "detail_str": (
                f"Last: {last['date']} {last['direction'].upper()} "
                f"{last['gap_pct']:.1%} ({type_label})"
                + (" [VOL]" if last["volume_surge"] else "")
                + f" | Total: {total}"
            ),
        }
