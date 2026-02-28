"""
analysis/support_resistance.py — Calculate support and resistance levels.

Two methods are used and combined:
  1. Classic pivot points (based on prior bar HLC)
  2. Fractal-based local minima/maxima over a lookback window

Results are clustered to avoid showing near-identical levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SRLevel:
    """A single support or resistance price level."""

    price: float
    level_type: str          # "support" or "resistance"
    source: str              # "pivot", "fractal", or "fibonacci"
    touches: int = 1         # number of times price touched this level
    label: str = ""          # optional label e.g. "S1", "R2"


def calculate_levels(df: pd.DataFrame, config: dict, current_price: float) -> dict[str, list[SRLevel]]:
    """Return a dict with keys ``"support"`` and ``"resistance"``, each a
    sorted list of :class:`SRLevel` objects."""

    method: str = config.get("method", "both")
    num_levels: int = int(config.get("num_levels", 4))
    cluster_pct: float = float(config.get("cluster_pct", 0.015))

    raw_levels: list[SRLevel] = []

    if method in ("pivot", "both"):
        raw_levels += _pivot_levels(df)

    if method in ("fractal", "both"):
        lookback = int(config.get("fractal_lookback", 60))
        order = int(config.get("fractal_order", 5))
        raw_levels += _fractal_levels(df, lookback, order)

    # Cluster close levels
    clustered = _cluster(raw_levels, cluster_pct)

    # Split into support / resistance relative to current price
    support = sorted(
        [lvl for lvl in clustered if lvl.price < current_price],
        key=lambda x: x.price,
        reverse=True,   # nearest support first
    )[:num_levels]

    resistance = sorted(
        [lvl for lvl in clustered if lvl.price > current_price],
        key=lambda x: x.price,
    )[:num_levels]

    # Ensure level_type is consistent with the list each level landed in.
    # Clustering may have flipped the type via majority vote, but the
    # authoritative classification is price vs current_price.
    for lvl in support:
        lvl.level_type = "support"
    for lvl in resistance:
        lvl.level_type = "resistance"

    return {"support": support, "resistance": resistance}


# ---------------------------------------------------------------------------
# Pivot points
# ---------------------------------------------------------------------------

def _pivot_levels(df: pd.DataFrame) -> list[SRLevel]:
    """Standard pivot point calculation using the most recent completed bar."""
    if len(df) < 2:
        return []

    bar = df.iloc[-2]   # last completed bar
    high = float(bar["high"])
    low = float(bar["low"])
    close = float(bar["close"])

    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)

    levels = [
        SRLevel(price=s3, level_type="support",    source="pivot", label="S3"),
        SRLevel(price=s2, level_type="support",    source="pivot", label="S2"),
        SRLevel(price=s1, level_type="support",    source="pivot", label="S1"),
        SRLevel(price=pivot, level_type="support", source="pivot", label="P"),
        SRLevel(price=r1, level_type="resistance", source="pivot", label="R1"),
        SRLevel(price=r2, level_type="resistance", source="pivot", label="R2"),
        SRLevel(price=r3, level_type="resistance", source="pivot", label="R3"),
    ]
    return [lvl for lvl in levels if lvl.price > 0]


# ---------------------------------------------------------------------------
# Fractal-based local extrema
# ---------------------------------------------------------------------------

def _fractal_levels(df: pd.DataFrame, lookback: int, order: int) -> list[SRLevel]:
    """Detect local highs/lows over the lookback window."""
    window = df.tail(lookback)
    if len(window) < order * 2 + 1:
        return []

    highs = window["high"].values
    lows = window["low"].values

    levels: list[SRLevel] = []

    # Local maxima
    for i in range(order, len(highs) - order):
        if all(highs[i] >= highs[i - j] for j in range(1, order + 1)) and \
           all(highs[i] >= highs[i + j] for j in range(1, order + 1)):
            levels.append(SRLevel(
                price=float(highs[i]),
                level_type="resistance",
                source="fractal",
            ))

    # Local minima
    for i in range(order, len(lows) - order):
        if all(lows[i] <= lows[i - j] for j in range(1, order + 1)) and \
           all(lows[i] <= lows[i + j] for j in range(1, order + 1)):
            levels.append(SRLevel(
                price=float(lows[i]),
                level_type="support",
                source="fractal",
            ))

    return levels


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _cluster(levels: list[SRLevel], cluster_pct: float) -> list[SRLevel]:
    """Merge levels within cluster_pct of each other into a single level."""
    if not levels:
        return []

    sorted_levels = sorted(levels, key=lambda x: x.price)
    clusters: list[list[SRLevel]] = []
    current_cluster: list[SRLevel] = [sorted_levels[0]]

    for lvl in sorted_levels[1:]:
        ref_price = current_cluster[0].price
        if ref_price > 0 and abs(lvl.price - ref_price) / ref_price <= cluster_pct:
            current_cluster.append(lvl)
        else:
            clusters.append(current_cluster)
            current_cluster = [lvl]
    clusters.append(current_cluster)

    result: list[SRLevel] = []
    for cluster in clusters:
        avg_price = float(np.mean([c.price for c in cluster]))
        touches = len(cluster)
        # Prefer pivot label if available
        label = next((c.label for c in cluster if c.label), "")
        # Determine type by majority
        n_support = sum(1 for c in cluster if c.level_type == "support")
        level_type = "support" if n_support >= len(cluster) / 2 else "resistance"
        source = "+".join(sorted({c.source for c in cluster}))
        result.append(SRLevel(
            price=avg_price,
            level_type=level_type,
            source=source,
            touches=touches,
            label=label,
        ))

    return result
