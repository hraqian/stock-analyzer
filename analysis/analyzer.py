"""
analysis/analyzer.py — Orchestrates all indicators, S/R levels, and scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from indicators.registry import IndicatorRegistry
from indicators.base import IndicatorResult
from analysis.support_resistance import calculate_levels, SRLevel
from analysis.scorer import CompositeScorer

if TYPE_CHECKING:
    from config import Config
    from data.provider import DataProvider


@dataclass
class AnalysisResult:
    """Full analysis output for one ticker."""

    ticker: str
    period: str
    info: dict[str, Any]
    indicator_results: list[IndicatorResult]
    support_levels: list[SRLevel]
    resistance_levels: list[SRLevel]
    composite: dict[str, Any]   # from CompositeScorer.score()
    df: pd.DataFrame = field(repr=False)


class Analyzer:
    """Runs the full technical analysis pipeline."""

    def __init__(
        self,
        cfg: "Config",
        provider: "DataProvider",
        only_indicators: list[str] | None = None,
    ) -> None:
        self._cfg = cfg
        self._provider = provider
        self._only = only_indicators

    def run(
        self,
        ticker: str,
        period: str | None = "6mo",
        interval: str = "1d",
        start: str | None = None,
        end: str | None = None,
    ) -> AnalysisResult:
        """Fetch data, run all indicators, compute S/R, return AnalysisResult."""

        # 1. Fetch data and metadata
        df = self._provider.fetch(ticker, period=period, interval=interval, start=start, end=end)
        info = self._provider.get_info(ticker)

        current_price = float(df["close"].iloc[-1])
        if info.get("current_price") is None:
            info["current_price"] = current_price

        # Build a display-friendly period label
        if start:
            period_label = f"{start} → {end or 'today'}"
        else:
            period_label = period or "6mo"

        # 2. Run indicators
        registry = IndicatorRegistry(self._cfg, only=self._only)
        indicator_results = registry.run_all(df)

        # 3. Support / Resistance
        sr_cfg = self._cfg.section("support_resistance")
        sr = calculate_levels(df, sr_cfg, current_price)

        # 4. Composite score
        scorer = CompositeScorer(self._cfg)
        composite = scorer.score(indicator_results)

        return AnalysisResult(
            ticker=ticker.upper(),
            period=period_label,
            info=info,
            indicator_results=indicator_results,
            support_levels=sr["support"],
            resistance_levels=sr["resistance"],
            composite=composite,
            df=df,
        )
