"""Score timeseries helpers for percentile-based signals."""

from __future__ import annotations

import pandas as pd

from analysis.scorer import CompositeScorer
from analysis.pattern_scorer import PatternCompositeScorer
from indicators.registry import IndicatorRegistry
from patterns.registry import PatternRegistry

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config
    from data.provider import DataProvider


def compute_score_timeseries(
    cfg: "Config",
    provider: "DataProvider",
    *,
    ticker: str,
    period: str | None,
    interval: str,
    start: str | None,
    end: str | None,
    step: int,
) -> pd.DataFrame:
    """Compute indicator/pattern composite scores at regular intervals.

    Returns a DataFrame indexed by date with columns:
      indicator_score, pattern_score
    """
    df = provider.fetch(
        ticker,
        period=period if not start else None,
        interval=interval,
        start=start,
        end=end,
    )

    warmup = int(cfg.section("backtest").get("warmup_bars", 200))
    warmup_min = int(cfg.section("backtest").get("warmup_min_bars", 20))
    warmup = min(warmup, len(df) - 10) if len(df) > 10 else warmup
    if warmup < warmup_min:
        warmup = warmup_min

    dates: list[pd.Timestamp] = []
    ind_scores: list[float] = []
    pat_scores: list[float] = []

    for i in range(warmup, len(df), max(1, step)):
        trailing = df.iloc[: i + 1]

        ind_reg = IndicatorRegistry(cfg)
        ind_results = ind_reg.run_all(trailing)
        ind_scorer = CompositeScorer(cfg)
        ind_composite = ind_scorer.score(ind_results)

        pat_reg = PatternRegistry(cfg)
        pat_results = pat_reg.run_all(trailing)
        pat_scorer = PatternCompositeScorer(cfg)
        pat_composite = pat_scorer.score(pat_results)

        dates.append(df.index[i])
        ind_scores.append(float(ind_composite["overall"]))
        pat_scores.append(float(pat_composite["overall"]))

    score_df = pd.DataFrame({
        "indicator_score": ind_scores,
        "pattern_score": pat_scores,
    }, index=dates)
    score_df.index.name = "date"
    return score_df
