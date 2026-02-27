"""Cached data loading: analysis pipeline, backtest engine, score timeseries."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis.analyzer import Analyzer, AnalysisResult
from analysis.multi_timeframe import MultiTimeframeAnalyzer, MultiTimeframeResult
from analysis.score_timeseries import compute_score_timeseries as _compute_scores
from config import Config
from data.yahoo import YahooFinanceProvider
from engine.backtest import BacktestEngine, BacktestResult
from engine.score_strategy import ScoreBasedStrategy
from engine.suitability import SuitabilityAnalyzer, TradingMode

from .display_utils import is_intraday


@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def load_analysis(
    ticker: str,
    period: str | None,
    interval: str,
    start: str | None,
    end: str | None,
    config_hash: str,
    config_data: dict,
) -> tuple[AnalysisResult, dict]:
    """Run the full analysis pipeline and return (result, cfg_data)."""
    cfg = Config.from_dict(config_data)

    provider = YahooFinanceProvider()
    analyzer = Analyzer(cfg, provider)
    result = analyzer.run(
        ticker,
        period=period if not start else None,
        interval=interval,
        start=start,
        end=end,
    )
    return result, config_data


@st.cache_data(ttl=300, show_spinner="Running backtest...")
def load_backtest(
    ticker: str,
    period: str | None,
    interval: str,
    start: str | None,
    end: str | None,
    trading_mode_str: str,
    config_hash: str,
    config_data: dict,
) -> tuple[BacktestResult, str, "dict | None", dict]:
    """Run the backtest engine and return results.

    Returns (bt_result, trading_mode_value, assessment_dict_or_None, cfg_data).
    We return serialisable types so Streamlit can cache them.
    """
    cfg = Config.from_dict(config_data)

    provider = YahooFinanceProvider()

    # Determine trading mode
    forced = trading_mode_str != "auto"
    assessment_dict = None

    if forced:
        mode_map = {
            "long_short": TradingMode.LONG_SHORT,
            "long_only": TradingMode.LONG_ONLY,
            "hold_only": TradingMode.HOLD_ONLY,
        }
        trading_mode = mode_map[trading_mode_str]
    elif is_intraday(interval):
        trading_mode = TradingMode.LONG_SHORT
    else:
        df = provider.fetch(
            ticker,
            period=period if not start else None,
            interval=interval,
            start=start,
            end=end,
        )
        suit_analyzer = SuitabilityAnalyzer(cfg)
        assessment = suit_analyzer.assess(df)
        trading_mode = assessment.mode
        # Serialise assessment for cache-friendliness
        assessment_dict = {
            "mode": assessment.mode.value,
            "avg_daily_volume": assessment.avg_daily_volume,
            "adx_value": assessment.adx_value,
            "atr_pct": assessment.atr_pct,
            "reasons": assessment.reasons,
        }

    regime_adapt = cfg.section("regime").get("strategy_adaptation", {})
    strategy = ScoreBasedStrategy(
        params=cfg.section("strategy"),
        trading_mode=trading_mode,
        regime_adaptation=regime_adapt,
    )
    engine = BacktestEngine(
        data_provider=provider,
        strategy=strategy,
        cfg=cfg,
        trading_mode=trading_mode,
    )
    bt_result = engine.run(
        ticker,
        period=period if not start else None,
        interval=interval,
        start=start,
        end=end,
    )
    return bt_result, trading_mode.value, assessment_dict, config_data


# ---------------------------------------------------------------------------
# Score timeseries computation
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner="Computing score timeseries...")
def compute_score_timeseries(
    ticker: str,
    period: str | None,
    interval: str,
    start: str | None,
    end: str | None,
    config_hash: str,
    config_data: dict,
    step: int = 5,
) -> pd.DataFrame:
    """Compute indicator and pattern composite scores at regular intervals.

    Returns a DataFrame indexed by date with columns:
      indicator_score, pattern_score
    """
    cfg = Config.from_dict(config_data)
    provider = YahooFinanceProvider()
    return _compute_scores(
        cfg,
        provider,
        ticker=ticker,
        period=period,
        interval=interval,
        start=start,
        end=end,
        step=step,
    )


# ---------------------------------------------------------------------------
# Multi-timeframe analysis
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner="Running multi-timeframe analysis...")
def load_multi_timeframe(
    ticker: str,
    config_hash: str,
    config_data: dict,
) -> MultiTimeframeResult:
    """Run multi-timeframe analysis and return aggregated result.

    Reads ``multi_timeframe`` config section for timeframes, weights, and
    periods.  Calls ``MultiTimeframeAnalyzer.run()`` which internally runs
    ``Analyzer.run()`` for each timeframe.
    """
    cfg = Config.from_dict(config_data)
    provider = YahooFinanceProvider()
    mta = MultiTimeframeAnalyzer(cfg, provider)
    return mta.run(ticker)
