"""Score timeseries helpers for percentile-based signals."""

from __future__ import annotations

import logging

import pandas as pd

from analysis.scorer import CompositeScorer
from analysis.pattern_scorer import PatternCompositeScorer
from indicators.registry import IndicatorRegistry
from patterns.registry import PatternRegistry

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config
    from data.provider import DataProvider

logger = logging.getLogger(__name__)


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

    warmup = int(cfg.section("backtest").get("warmup_bars", 50))
    warmup_min = int(cfg.section("backtest").get("min_warmup_bars", 20))
    max_warmup_ratio = float(cfg.section("backtest").get("max_warmup_ratio", 0.5))

    # Proportional cap — same logic as BacktestEngine so warmup boundaries
    # are consistent between the score timeseries and actual backtests.
    max_warmup = int(len(df) * max_warmup_ratio)
    if warmup > max_warmup:
        warmup = max(warmup_min, max_warmup)
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


def compute_dca_score_df(
    cfg: "Config",
    provider: "DataProvider",
    *,
    ticker: str,
    period: str | None,
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
    step: int = 5,
) -> pd.DataFrame:
    """Build the score DataFrame required by DCA score-integrated mode.

    Runs the indicator and pattern pipeline at every *step* bars from warmup
    onward and extracts the five columns the DCA engine's
    ``_resolve_multiplier`` method expects:

    ========== ====================================================
    Column     Description
    ========== ====================================================
    composite  Overall composite indicator score (0-10).
    rsi_raw    Raw RSI value (0-100).
    bb_pctile  Bollinger Band %B scaled to 0-100.
    gap_type   Most recent gap classification (str, may be empty).
    gap_direction  Direction of the most recent gap (str, may be empty).
    ========== ====================================================

    Parameters
    ----------
    cfg : Config
        Application configuration.
    provider : DataProvider
        Data provider for fetching OHLCV data.
    ticker : str
        Stock ticker symbol.
    period : str | None
        yfinance period string (e.g. ``"5y"``).
    interval : str
        Bar interval (default ``"1d"``).
    start, end : str | None
        Optional date-range overrides.
    step : int
        Evaluate every *step*-th bar.  Lower = more precise but slower.
        Default 5 (roughly weekly for daily bars).

    Returns
    -------
    pd.DataFrame
        Indexed by date with the five columns above.
    """
    df = provider.fetch(
        ticker,
        period=period if not start else None,
        interval=interval,
        start=start,
        end=end,
    )

    warmup = int(cfg.section("backtest").get("warmup_bars", 50))
    warmup_min = int(cfg.section("backtest").get("min_warmup_bars", 20))
    max_warmup_ratio = float(cfg.section("backtest").get("max_warmup_ratio", 0.5))

    max_warmup = int(len(df) * max_warmup_ratio)
    if warmup > max_warmup:
        warmup = max(warmup_min, max_warmup)
    if warmup < warmup_min:
        warmup = warmup_min

    dates: list[pd.Timestamp] = []
    composites: list[float] = []
    rsi_values: list[float] = []
    bb_values: list[float] = []
    gap_types: list[str] = []
    gap_directions: list[str] = []

    for i in range(warmup, len(df), max(1, step)):
        trailing = df.iloc[: i + 1]

        # ── Indicators ───────────────────────────────────────────
        ind_reg = IndicatorRegistry(cfg)
        ind_results = ind_reg.run_all(trailing)
        ind_scorer = CompositeScorer(cfg)
        ind_composite = ind_scorer.score(ind_results)

        composite_val = float(ind_composite["overall"])

        # Extract RSI
        rsi_val = 50.0
        for r in ind_results:
            if r.config_key == "rsi":
                rsi_val = float(r.values.get("rsi", 50.0))
                break

        # Extract Bollinger Band %B → scale to 0-100
        bb_val = 50.0
        for r in ind_results:
            if r.config_key == "bollinger_bands":
                bb_val = float(r.values.get("pct_b", 0.5)) * 100.0
                break

        # ── Patterns (only gaps needed) ──────────────────────────
        gap_type = ""
        gap_dir = ""
        try:
            pat_reg = PatternRegistry(cfg)
            pat_results = pat_reg.run_all(trailing)
            for pr in pat_results:
                if pr.config_key == "gaps":
                    recent = pr.values.get("recent_gaps", [])
                    if recent:
                        last_gap = recent[-1]
                        gap_type = str(last_gap.get("gap_type", ""))
                        gap_dir = str(last_gap.get("direction", ""))
                    break
        except Exception:
            logger.debug("Gap detection failed at bar %d for %s", i, ticker)

        dates.append(df.index[i])
        composites.append(composite_val)
        rsi_values.append(rsi_val)
        bb_values.append(bb_val)
        gap_types.append(gap_type)
        gap_directions.append(gap_dir)

    score_df = pd.DataFrame({
        "composite": composites,
        "rsi_raw": rsi_values,
        "bb_pctile": bb_values,
        "gap_type": gap_types,
        "gap_direction": gap_directions,
    }, index=dates)
    score_df.index.name = "date"

    logger.info(
        "Computed DCA score_df for %s: %d rows (step=%d, warmup=%d)",
        ticker, len(score_df), step, warmup,
    )
    return score_df
