"""
patterns/volume_range.py — Volume-range correlation analysis.

Measures the relationship between bar range (high - low) and volume:
  - Expansion days: wide range + high volume → confirms directional moves
  - Contraction days: narrow range + low volume → consolidation / indecision
  - Divergence: wide range + low volume (suspect) or narrow range + high volume (absorption)

Uses rolling ratios of:
  - Range ratio:  current range / average range
  - Volume ratio: current volume / average volume

Scoring logic:
  Bullish expansion (wide range up + high volume) → score > 5
  Bearish expansion (wide range down + high volume) → score < 5
  Contraction / neutral → 5.0
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import BasePattern


class VolumeRangePattern(BasePattern):
    name = "Volume-Range"
    config_key = "volume_range"

    def detect(self, df: pd.DataFrame) -> dict[str, Any]:
        period = int(self.config.get("period", 20))
        expansion_threshold = float(self.config.get("expansion_threshold", 1.5))
        contraction_threshold = float(self.config.get("contraction_threshold", 0.6))
        lookback = int(self.config.get("lookback", 10))

        if len(df) < period + 2:
            return {
                "range_ratio": 1.0, "volume_ratio": 1.0,
                "regime": "neutral", "recent_expansions": 0,
                "recent_contractions": 0, "directional_bias": 0.0,
            }

        # Compute bar range and its rolling average
        bar_range = df["high"] - df["low"]
        avg_range = bar_range.rolling(window=period).mean()
        range_ratio = bar_range / avg_range

        # Volume ratio
        avg_volume = df["volume"].rolling(window=period).mean()
        volume_ratio = df["volume"] / avg_volume

        # Price direction for each bar
        bar_direction = np.sign(df["close"].values - df["open"].values)

        # Classify recent bars
        recent_start = max(0, len(df) - lookback)
        recent_expansions_bull = 0
        recent_expansions_bear = 0
        recent_contractions = 0
        directional_sum = 0.0

        for i in range(recent_start, len(df)):
            rr = float(range_ratio.iloc[i]) if not np.isnan(range_ratio.iloc[i]) else 1.0
            vr = float(volume_ratio.iloc[i]) if not np.isnan(volume_ratio.iloc[i]) else 1.0
            direction = float(bar_direction[i])

            is_expansion = rr >= expansion_threshold and vr >= expansion_threshold
            is_contraction = rr <= contraction_threshold and vr <= contraction_threshold

            if is_expansion:
                if direction >= 0:
                    recent_expansions_bull += 1
                else:
                    recent_expansions_bear += 1
                # Weight by magnitude
                directional_sum += direction * rr * vr
            elif is_contraction:
                recent_contractions += 1

        total_expansions = recent_expansions_bull + recent_expansions_bear

        # Current bar classification
        curr_rr = float(range_ratio.iloc[-1]) if not np.isnan(range_ratio.iloc[-1]) else 1.0
        curr_vr = float(volume_ratio.iloc[-1]) if not np.isnan(volume_ratio.iloc[-1]) else 1.0
        curr_dir = float(bar_direction[-1])

        if curr_rr >= expansion_threshold and curr_vr >= expansion_threshold:
            regime = "expansion_bull" if curr_dir >= 0 else "expansion_bear"
        elif curr_rr <= contraction_threshold and curr_vr <= contraction_threshold:
            regime = "contraction"
        elif curr_rr >= expansion_threshold and curr_vr <= contraction_threshold:
            regime = "divergence_range"  # wide range but low volume (suspect)
        elif curr_rr <= contraction_threshold and curr_vr >= expansion_threshold:
            regime = "divergence_volume"  # narrow range but high volume (absorption)
        else:
            regime = "neutral"

        return {
            "range_ratio": curr_rr,
            "volume_ratio": curr_vr,
            "regime": regime,
            "recent_expansions_bull": recent_expansions_bull,
            "recent_expansions_bear": recent_expansions_bear,
            "recent_contractions": recent_contractions,
            "directional_bias": directional_sum,
            "lookback": lookback,
        }

    def score(self, values: dict[str, Any]) -> float:
        regime = values["regime"]
        bias = values["directional_bias"]
        bull_exp = values["recent_expansions_bull"]
        bear_exp = values["recent_expansions_bear"]

        scoring = self.config.get("scoring", {})
        expansion_bull_score = float(scoring.get("expansion_bull", 8.0))
        expansion_bear_score = float(scoring.get("expansion_bear", 2.0))
        contraction_score = float(scoring.get("contraction", 5.0))
        divergence_score = float(scoring.get("divergence", 5.0))

        # Start with regime-based score for the current bar
        if regime == "expansion_bull":
            base = expansion_bull_score
        elif regime == "expansion_bear":
            base = expansion_bear_score
        elif regime == "contraction":
            base = contraction_score
        elif regime in ("divergence_range", "divergence_volume"):
            base = divergence_score
        else:
            base = 5.0

        # Adjust based on recent expansion bias
        if bull_exp + bear_exp > 0:
            exp_bias = (bull_exp - bear_exp) / (bull_exp + bear_exp)
            # exp_bias in [-1, 1]; shift score toward that direction
            base += exp_bias * 1.5

        return self._clamp(base)

    def summary(self, values: dict[str, Any], score: float) -> dict[str, Any]:
        regime = values["regime"]
        rr = values["range_ratio"]
        vr = values["volume_ratio"]
        bull_exp = values["recent_expansions_bull"]
        bear_exp = values["recent_expansions_bear"]
        contractions = values["recent_contractions"]
        lookback = values["lookback"]

        regime_labels = {
            "expansion_bull": "Bullish Expansion",
            "expansion_bear": "Bearish Expansion",
            "contraction": "Contraction",
            "divergence_range": "Range Divergence",
            "divergence_volume": "Vol Absorption",
            "neutral": "Neutral",
        }
        regime_label = regime_labels.get(regime, regime)

        return {
            "value_str": f"{regime_label}",
            "detail_str": (
                f"Range: {rr:.2f}x avg | Vol: {vr:.2f}x avg | "
                f"Last {lookback}: {bull_exp}\u2191 {bear_exp}\u2193 {contractions}\u25A0"
            ),
        }
