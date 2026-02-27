"""
patterns/spikes.py — Spike detection and confirmation analysis.

Detects sudden price spikes (breakouts or traps):
  - Spike up:   price moves > N standard deviations above recent mean
  - Spike down: price moves > N standard deviations below recent mean

Confirmation logic:
  A spike is "confirmed" if the price holds above (for up) or below (for down)
  the spike level on subsequent bars. Unconfirmed spikes are likely traps.

This is inherently lagging: we can only confirm/reject a spike after several
bars have passed.

Scoring:
  Confirmed bullish spike → score > 5
  Confirmed bearish spike → score < 5
  Trap (false breakout) → inverse signal
  No spikes → 5.0 (neutral)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import BasePattern


class SpikePattern(BasePattern):
    name = "Spikes"
    config_key = "spikes"

    def detect(self, df: pd.DataFrame) -> dict[str, Any]:
        period = int(self.config.get("period", 20))
        spike_std = float(self.config.get("spike_std", 2.5))
        confirm_bars = int(self.config.get("confirm_bars", 3))
        confirm_pct = float(self.config.get("confirm_pct", 0.5))
        lookback = int(self.config.get("lookback", 20))
        z_magnitude_cap = float(self.config.get("z_magnitude_cap", 2.0))
        unconfirmed_weight = float(self.config.get("unconfirmed_weight", 0.3))

        if len(df) < period + confirm_bars + 2:
            return {
                "spikes": [], "recent_spikes": [],
                "net_signal": 0.0, "total_spikes": 0, "recent_count": 0,
            }

        # Rolling mean and std of close prices
        rolling_mean = df["close"].rolling(window=period).mean()
        rolling_std = df["close"].rolling(window=period).std()

        spikes = []

        for i in range(period, len(df)):
            close = float(df["close"].iloc[i])
            mean = float(rolling_mean.iloc[i])
            std = float(rolling_std.iloc[i])

            if np.isnan(mean) or np.isnan(std) or std == 0:
                continue

            z_score = (close - mean) / std

            if abs(z_score) < spike_std:
                continue

            direction = "up" if z_score > 0 else "down"
            spike_level = close

            # Check confirmation: did price hold above/below the spike level?
            confirmed = None  # None = can't confirm yet (not enough future bars)
            if i + confirm_bars < len(df):
                subsequent = df["close"].iloc[i + 1: i + 1 + confirm_bars]
                if direction == "up":
                    # Price should stay above spike_level for confirm_pct of bars
                    held = sum(1 for s in subsequent if float(s) >= spike_level)
                    confirmed = (held / confirm_bars) >= confirm_pct
                else:
                    held = sum(1 for s in subsequent if float(s) <= spike_level)
                    confirmed = (held / confirm_bars) >= confirm_pct

            spikes.append({
                "bar_index": i,
                "date": str(df.index[i])[:10],
                "direction": direction,
                "z_score": z_score,
                "spike_level": spike_level,
                "confirmed": confirmed,
            })

        recent_spikes = [s for s in spikes if s["bar_index"] >= len(df) - lookback]

        # Net signal
        net_signal = 0.0
        for s in recent_spikes:
            bars_ago = len(df) - 1 - s["bar_index"]
            recency = max(0.1, 1.0 - (bars_ago / lookback))
            z_magnitude = min(abs(s["z_score"]) / spike_std, z_magnitude_cap)

            if s["confirmed"] is True:
                # Confirmed spike = directional signal
                if s["direction"] == "up":
                    net_signal += z_magnitude * recency
                else:
                    net_signal -= z_magnitude * recency
            elif s["confirmed"] is False:
                # Trap = inverse signal (failed breakout)
                trap_weight = float(self.config.get("trap_weight", 0.7))
                if s["direction"] == "up":
                    net_signal -= z_magnitude * recency * trap_weight
                else:
                    net_signal += z_magnitude * recency * trap_weight
            # confirmed is None: too recent, count as mild directional
            else:
                if s["direction"] == "up":
                    net_signal += z_magnitude * recency * unconfirmed_weight
                else:
                    net_signal -= z_magnitude * recency * unconfirmed_weight

        return {
            "spikes": spikes,
            "recent_spikes": recent_spikes,
            "net_signal": net_signal,
            "total_spikes": len(spikes),
            "recent_count": len(recent_spikes),
        }

    def score(self, values: dict[str, Any]) -> float:
        net = values["net_signal"]
        if not values["recent_spikes"]:
            return 5.0

        max_signal = float(self.config.get("max_signal_strength", 3.0))

        if net >= 0:
            return self._linear_score(net, 0, max_signal, 5.0, 9.5)
        else:
            return self._linear_score(net, -max_signal, 0, 0.5, 5.0)

    def summary(self, values: dict[str, Any], score: float) -> dict[str, Any]:
        recent = values["recent_spikes"]
        total = values["total_spikes"]

        if not recent:
            return {
                "value_str": "None detected",
                "detail_str": f"Total spikes: {total} | No recent spikes",
            }

        last = recent[-1]
        direction_arrow = "\u2191" if last["direction"] == "up" else "\u2193"

        if last["confirmed"] is True:
            status = "Confirmed"
        elif last["confirmed"] is False:
            status = "TRAP"
        else:
            status = "Pending"

        # Counts
        confirmed_up = sum(1 for s in recent if s["direction"] == "up" and s["confirmed"] is True)
        confirmed_down = sum(1 for s in recent if s["direction"] == "down" and s["confirmed"] is True)
        traps = sum(1 for s in recent if s["confirmed"] is False)

        return {
            "value_str": f"{len(recent)} recent ({direction_arrow} {status})",
            "detail_str": (
                f"Last: {last['date']} {last['direction'].upper()} "
                f"z={last['z_score']:.1f} ({status}) | "
                f"Confirmed: {confirmed_up}\u2191 {confirmed_down}\u2193 Traps: {traps}"
            ),
        }
