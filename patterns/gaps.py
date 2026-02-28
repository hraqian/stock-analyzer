"""
patterns/gaps.py — Gap detection and classification.

Detects price gaps between consecutive bars and classifies them:
  - Common gap:     small gap, no special context (neutral)
  - Breakaway gap:  gap out of a prior consolidation with high volume (directional)
  - Runaway gap:    gap in direction of an existing trend (continuation)
  - Exhaustion gap: gap at the end of an extended trend with high volume (reversal warning)

Classification follows standard TA definitions (Murphy, Bulkowski):
  - Breakaway: requires prior consolidation detected via Bollinger Band width
    percentile (K of M bars below threshold, percentile ranked over a separate
    longer window to avoid self-referential noise) PLUS near-zero net return
    over the consolidation window (weak-trend gate) PLUS a volume surge PLUS
    the gap clears the consolidation *close* range (open > rolling close-high
    for gap-up, open < rolling close-low for gap-down).
  - Exhaustion: requires an extended, *mature* prior trend — total return over
    the trend window exceeds threshold, return over a longer maturity window is
    also in the same direction (both halves), AND price is extended from its
    MA — PLUS a volume surge and a gap *with* that trend.  Reversal warning.

Scoring:
  Recent bullish gaps → score > 5 (bullish)
  Recent bearish gaps → score < 5 (bearish)
  Exhaustion gaps get inverse scoring (gap up = bearish warning, gap down = bullish).
  No gaps → 5.0 (neutral)

Intraday guardrail: if the DataFrame index has sub-daily frequency, a higher
``intraday_min_gap_pct`` is used automatically to avoid treating normal
intraday bar jumps as gaps.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import BasePattern


def _is_intraday(df: pd.DataFrame) -> bool:
    """Return True if the DataFrame appears to have intraday (sub-daily) bars."""
    if len(df) < 2:
        return False
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        return False
    # Check median time delta of the first 20 bars
    deltas = idx[:min(len(idx), 21)].to_series().diff().dropna()
    if deltas.empty:
        return False
    median_delta = deltas.median()
    # Anything less than ~20 hours is intraday
    return median_delta < pd.Timedelta(hours=20)


class GapPattern(BasePattern):
    name = "Gaps"
    config_key = "gaps"

    def detect(self, df: pd.DataFrame) -> dict[str, Any]:
        lookback = int(self.config.get("lookback", 20))
        min_gap_pct = float(self.config.get("min_gap_pct", 0.005))
        volume_surge_mult = float(self.config.get("volume_surge_mult", 1.5))
        trend_period = int(self.config.get("trend_period", 20))

        # Consolidation params (BB width based)
        consolidation_lookback = int(self.config.get("consolidation_lookback", 20))
        # Separate, longer window for BB width percentile ranking to avoid
        # self-referential noise when consolidation_lookback is short.
        bb_percentile_lookback = int(
            self.config.get("bb_percentile_lookback", consolidation_lookback * 2)
        )
        consolidation_bb_percentile = float(
            self.config.get("consolidation_bb_percentile", 50)
        )
        consolidation_min_bars = int(self.config.get("consolidation_min_bars", 5))
        # Weak-trend gate: absolute return over consolidation window must be
        # below this for true consolidation (prevents labeling low-vol trends).
        consolidation_max_return = float(
            self.config.get("consolidation_max_return", 0.03)
        )

        # Exhaustion params (total return + MA distance + trend maturity)
        exhaustion_min_return = float(
            self.config.get("exhaustion_min_return", 0.10)
        )
        exhaustion_min_distance_pct = float(
            self.config.get("exhaustion_min_distance_pct", 0.05)
        )
        exhaustion_min_trend_bars = int(
            self.config.get("exhaustion_min_trend_bars", 40)
        )

        # Intraday guardrail (#6)
        intraday_min_gap_pct = float(
            self.config.get("intraday_min_gap_pct", 0.01)
        )
        if _is_intraday(df):
            min_gap_pct = max(min_gap_pct, intraday_min_gap_pct)

        if len(df) < trend_period + 2:
            return {"gaps": [], "recent_gaps": [], "net_gap_score": 0.0}

        close = df["close"]

        # ----------------------------------------------------------
        # Pre-computed series
        # ----------------------------------------------------------
        # Average volume for surge detection
        avg_volume = df["volume"].rolling(window=trend_period).mean()

        # EMA + slope for trend direction
        ema = close.ewm(span=trend_period, adjust=False).mean()
        ema_slope = ema.diff()

        # Bollinger Band width for consolidation (#3)
        bb_ma = close.rolling(window=consolidation_lookback).mean()
        bb_std = close.rolling(window=consolidation_lookback).std(ddof=0)
        bb_width = (2.0 * bb_std) / bb_ma.where(bb_ma != 0, np.nan)

        # Rolling percentile rank of BB width over a longer window to avoid
        # self-referential noise (bb_percentile_lookback >= consolidation_lookback).
        bb_width_rank = bb_width.rolling(window=bb_percentile_lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
            if len(x) >= 2 else 50.0,
            raw=False,
        )

        # Boolean: this bar has low BB width (below consolidation percentile)
        bb_is_narrow = bb_width_rank <= consolidation_bb_percentile

        # Rolling count of narrow-BB bars over consolidation window (#4)
        # Convert boolean to int for rolling sum
        narrow_count = bb_is_narrow.astype(int).rolling(
            window=consolidation_lookback, min_periods=1
        ).sum()

        # Net return over consolidation window — true consolidation has near-
        # zero net return (price hasn't gone anywhere).
        consolidation_return = close.pct_change(periods=consolidation_lookback).abs()

        # Rolling high/low of *close* over consolidation window for the
        # breakaway range-clearing check.  Using close (not high/low) avoids
        # counting wick-clearing gaps as breakaways in instruments with long
        # wicks.
        rolling_high = close.rolling(window=consolidation_lookback).max()
        rolling_low = close.rolling(window=consolidation_lookback).min()

        # Total return over trend_period for exhaustion detection (#2)
        total_return = close.pct_change(periods=trend_period)

        # Trend maturity: return over the longer maturity window AND return
        # over the first half of that window.  Both must be in the gap
        # direction.  A fast rip from a flat base fails because the first-half
        # return is ~0 while all the gains are in the second half.
        maturity_return = close.pct_change(periods=exhaustion_min_trend_bars)
        half_maturity = max(1, exhaustion_min_trend_bars // 2)
        # Return from (maturity_bars ago) to (half_maturity bars ago):
        #   close[i - half] / close[i - maturity] - 1
        first_half_return = close.shift(half_maturity).pct_change(
            periods=exhaustion_min_trend_bars - half_maturity
        )

        # Distance from MA as fraction of price (#5)
        ma_distance_pct = (close - ema).abs() / ema.where(ema != 0, np.nan)

        gaps = []
        for i in range(1, len(df)):
            prev_high = float(df["high"].iloc[i - 1])
            prev_low = float(df["low"].iloc[i - 1])
            prev_close = float(close.iloc[i - 1])
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
            avg_vol_val = (
                float(avg_volume.iloc[i])
                if not np.isnan(avg_volume.iloc[i])
                else curr_volume
            )
            volume_surge = curr_volume > (avg_vol_val * volume_surge_mult)
            slope_val = (
                float(ema_slope.iloc[i])
                if not np.isnan(ema_slope.iloc[i])
                else 0.0
            )

            # Trend alignment
            trending_up = slope_val > 0
            trending_down = slope_val < 0
            gap_with_trend = (
                (direction == "up" and trending_up)
                or (direction == "down" and trending_down)
            )

            # -------------------------------------------------------
            # Consolidation detection (#3 + #4 + weak-trend gate)
            # -------------------------------------------------------
            # At least consolidation_min_bars of the last consolidation_lookback
            # bars must have had narrow BB width.
            prior_narrow = (
                float(narrow_count.iloc[i - 1])
                if i >= 1 and not np.isnan(narrow_count.iloc[i - 1])
                else 0
            )
            sustained_consolidation = prior_narrow >= consolidation_min_bars

            # Weak-trend gate: absolute return over the consolidation window
            # must be small.  A low-vol trending stock has narrow BB width but
            # non-zero net return — not true consolidation.
            prior_consol_ret = (
                float(consolidation_return.iloc[i - 1])
                if i >= 1 and not np.isnan(consolidation_return.iloc[i - 1])
                else 0.0
            )
            weak_trend = prior_consol_ret <= consolidation_max_return

            # Directional breakaway check (#1): gap must clear the
            # consolidation close range (rolling close high/low at bar before gap).
            prior_rolling_high = (
                float(rolling_high.iloc[i - 1])
                if i >= 1 and not np.isnan(rolling_high.iloc[i - 1])
                else prev_close
            )
            prior_rolling_low = (
                float(rolling_low.iloc[i - 1])
                if i >= 1 and not np.isnan(rolling_low.iloc[i - 1])
                else prev_close
            )
            clears_range = (
                (direction == "up" and curr_open > prior_rolling_high)
                or (direction == "down" and curr_open < prior_rolling_low)
            )

            in_consolidation = sustained_consolidation and weak_trend and clears_range

            # -------------------------------------------------------
            # Exhaustion detection (#2 + #5 + maturity)
            # -------------------------------------------------------
            prior_return = (
                float(total_return.iloc[i - 1])
                if i >= 1 and not np.isnan(total_return.iloc[i - 1])
                else 0.0
            )
            # Total return must exceed threshold in direction of gap
            extended_by_return = (
                (direction == "up" and prior_return >= exhaustion_min_return)
                or (direction == "down" and prior_return <= -exhaustion_min_return)
            )

            # Price must also be extended from its MA (#5)
            prior_ma_dist = (
                float(ma_distance_pct.iloc[i - 1])
                if i >= 1 and not np.isnan(ma_distance_pct.iloc[i - 1])
                else 0.0
            )
            extended_from_ma = prior_ma_dist >= exhaustion_min_distance_pct

            # Trend maturity: the return over the full maturity window must be
            # in the gap direction AND the first half of the maturity window
            # must also show movement in the same direction.  A 20-bar rip
            # from a flat base fails: full 40-bar return is positive, but
            # bars 40-ago to 20-ago were flat → first_half ≈ 0.
            prior_maturity = (
                float(maturity_return.iloc[i - 1])
                if i >= 1 and not np.isnan(maturity_return.iloc[i - 1])
                else 0.0
            )
            prior_first_half = (
                float(first_half_return.iloc[i - 1])
                if i >= 1 and not np.isnan(first_half_return.iloc[i - 1])
                else 0.0
            )
            # Both halves must agree with gap direction
            mature_trend = (
                (direction == "up"
                 and prior_maturity > 0
                 and prior_first_half > 0)
                or (direction == "down"
                    and prior_maturity < 0
                    and prior_first_half < 0)
            )

            extended_trend = extended_by_return and extended_from_ma and mature_trend

            # -------------------------------------------------------
            # Classification (textbook-aligned)
            # -------------------------------------------------------
            if volume_surge and in_consolidation:
                # Gap out of consolidation clearing the range with volume
                gap_type = "breakaway"
            elif (
                volume_surge
                and extended_trend
                and gap_with_trend
                and gap_pct > min_gap_pct * 2
            ):
                # Late-stage gap in direction of extended trend with volume
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
