"""
analysis/multi_timeframe.py — Multi-timeframe confirmation analysis.

Runs the Analyzer on multiple timeframes (e.g. daily, weekly, monthly),
then aggregates the composite scores via weighted average to produce
a single multi-timeframe view.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config import Config
    from data.provider import DataProvider

from analysis.analyzer import Analyzer, AnalysisResult
from analysis.score_timeseries import compute_score_timeseries

logger = logging.getLogger(__name__)


@dataclass
class TimeframeResult:
    """Analysis result for a single timeframe."""

    timeframe: str          # e.g. "1d", "1wk", "1mo"
    period: str             # e.g. "2y", "5y", "max"
    weight: float           # configured weight for this timeframe
    indicator_score: float  # composite["overall"]
    pattern_score: float    # pattern_composite["overall"]
    signal: str             # "BUY" / "HOLD" / "SELL" (derived from thresholds)
    regime_label: str | None
    trend_score: float | None
    contrarian_score: float | None
    dominant_group: str | None
    error: str | None = None
    analysis_result: AnalysisResult | None = field(default=None, repr=False)


@dataclass
class MultiTimeframeResult:
    """Aggregated multi-timeframe analysis output."""

    ticker: str
    timeframe_results: list[TimeframeResult]
    aggregated_indicator_score: float   # weighted average of indicator scores
    aggregated_pattern_score: float     # weighted average of pattern scores
    aggregated_signal: str              # derived from aggregated scores
    agreement: str                      # "aligned", "mixed", "conflicting"

    @property
    def n_timeframes(self) -> int:
        return len(self.timeframe_results)

    @property
    def successful_timeframes(self) -> list[TimeframeResult]:
        return [tr for tr in self.timeframe_results if tr.error is None]


def _derive_signal(
    effective_score: float,
    short_below: float,
    hold_below: float,
) -> str:
    """Derive BUY/HOLD/SELL from an effective score and thresholds."""
    if effective_score <= short_below:
        return "SELL"
    elif effective_score > hold_below:
        return "BUY"
    return "HOLD"


def _compute_agreement(signals: list[str]) -> str:
    """Determine alignment across timeframes.

    Returns:
        "aligned"     — all timeframes agree on the same signal
        "mixed"       — at least 2 distinct signals, none in direct opposition
        "conflicting" — both BUY and SELL present
    """
    unique = set(signals)
    if len(unique) <= 1:
        return "aligned"
    if "BUY" in unique and "SELL" in unique:
        return "conflicting"
    return "mixed"


class MultiTimeframeAnalyzer:
    """Run analysis across multiple timeframes and aggregate scores."""

    def __init__(self, cfg: "Config", provider: "DataProvider") -> None:
        self._cfg = cfg
        self._provider = provider

    def run(self, ticker: str) -> MultiTimeframeResult:
        """Execute multi-timeframe analysis for *ticker*.

        Reads ``config.multi_timeframe`` for timeframes, weights, and periods.
        For each timeframe, runs ``Analyzer.run()`` and extracts composite scores.
        Aggregates via weighted average.
        """
        mt_cfg = self._cfg.section("multi_timeframe")
        timeframes: list[str] = mt_cfg.get("timeframes", ["1d", "1wk", "1mo"])
        weights: dict[str, float] = mt_cfg.get("weights", {
            "1d": 0.5, "1wk": 0.3, "1mo": 0.2,
        })
        periods: dict[str, str] = mt_cfg.get("periods", {
            "1d": "2y", "1wk": "5y", "1mo": "max",
        })

        # Score thresholds for signal derivation
        strat_cfg = self._cfg.section("strategy")
        thresholds = strat_cfg.get("score_thresholds", {})
        short_below = float(thresholds.get("short_below", 3.5))
        hold_below = float(thresholds.get("hold_below", 6.0))
        threshold_mode = str(strat_cfg.get("threshold_mode", "fixed"))
        pct_cfg = strat_cfg.get("percentile_thresholds", {})
        short_pct = float(pct_cfg.get("short_percentile", 25))
        long_pct = float(pct_cfg.get("long_percentile", 75))
        lookback_bars = int(pct_cfg.get("lookback_bars", 60))
        min_fill_ratio = float(strat_cfg.get("percentile_min_fill_ratio", 0.8))

        # Combination mode for effective score
        combination_mode = str(strat_cfg.get("combination_mode", "weighted"))
        ind_weight = float(strat_cfg.get("indicator_weight", 0.7))
        pat_weight = float(strat_cfg.get("pattern_weight", 0.3))
        boost_strength = float(strat_cfg.get("boost_strength", 0.5))
        boost_dead_zone = float(strat_cfg.get("boost_dead_zone", 0.3))
        gate_indicator_min = float(strat_cfg.get("gate_indicator_min", 5.5))
        gate_indicator_max = float(strat_cfg.get("gate_indicator_max", 4.5))
        gate_pattern_min = float(strat_cfg.get("gate_pattern_min", 5.5))
        gate_pattern_max = float(strat_cfg.get("gate_pattern_max", 4.5))

        results: list[TimeframeResult] = []

        # Rolling windows for percentile mode (per timeframe)
        score_windows: dict[str, list[float]] = {tf: [] for tf in timeframes}

        for tf in timeframes:
            w = weights.get(tf, 0.0)
            p = periods.get(tf, "2y")

            try:
                analyzer = Analyzer(self._cfg, self._provider)
                ar = analyzer.run(ticker, period=p, interval=tf)

                ind_score = ar.composite.get("overall", 5.0)
                pat_score = ar.pattern_composite.get("overall", 5.0)
                trend_s = ar.composite.get("trend_score")
                contr_s = ar.composite.get("contrarian_score")
                dom = ar.composite.get("dominant_group")

                # Compute effective score for signal derivation
                eff = _compute_effective_score(
                    ind_score, pat_score, combination_mode,
                    ind_weight, pat_weight, boost_strength, boost_dead_zone,
                )
                if threshold_mode == "percentile":
                    window = _build_percentile_window(
                        self._cfg,
                        self._provider,
                        ticker=ticker,
                        period=p,
                        interval=tf,
                        lookback_bars=lookback_bars,
                        step=max(1, int(strat_cfg.get("percentile_step", 5))),
                    )
                    score_windows[tf] = window
                signal = _derive_signal_with_mode(
                    eff,
                    ind_score,
                    pat_score,
                    combination_mode,
                    short_below,
                    hold_below,
                    threshold_mode,
                    score_windows[tf],
                    short_pct,
                    long_pct,
                    lookback_bars,
                    min_fill_ratio,
                    gate_indicator_min,
                    gate_indicator_max,
                    gate_pattern_min,
                    gate_pattern_max,
                )

                regime_label: str | None = None
                if ar.regime is not None:
                    regime_label = ar.regime.label

                results.append(TimeframeResult(
                    timeframe=tf,
                    period=p,
                    weight=w,
                    indicator_score=ind_score,
                    pattern_score=pat_score,
                    signal=signal,
                    regime_label=regime_label,
                    trend_score=trend_s,
                    contrarian_score=contr_s,
                    dominant_group=dom,
                    analysis_result=ar,
                ))
            except Exception as exc:
                logger.warning(
                    "Multi-timeframe: %s/%s failed: %s",
                    ticker, tf, exc, exc_info=True,
                )
                results.append(TimeframeResult(
                    timeframe=tf,
                    period=p,
                    weight=w,
                    indicator_score=5.0,
                    pattern_score=5.0,
                    signal="HOLD",
                    regime_label=None,
                    trend_score=None,
                    contrarian_score=None,
                    dominant_group=None,
                    error=str(exc),
                ))

        # -- Aggregate via weighted average (only successful timeframes) --
        ok = [r for r in results if r.error is None]
        if ok:
            total_w = sum(r.weight for r in ok)
            if total_w > 0:
                agg_ind = sum(r.indicator_score * r.weight for r in ok) / total_w
                agg_pat = sum(r.pattern_score * r.weight for r in ok) / total_w
            else:
                agg_ind = sum(r.indicator_score for r in ok) / len(ok)
                agg_pat = sum(r.pattern_score for r in ok) / len(ok)
        else:
            agg_ind = 5.0
            agg_pat = 5.0

        agg_eff = _compute_effective_score(
            agg_ind, agg_pat, combination_mode,
            ind_weight, pat_weight, boost_strength, boost_dead_zone,
        )
        # For the aggregated signal, always use fixed thresholds.
        # Building a percentile window from a single TF's distribution is
        # wrong — the aggregated score (weighted average of TFs) has a
        # different distribution than any individual TF.  The per-TF
        # signals already use percentile mode correctly; the aggregated
        # signal serves as a consensus override and fixed thresholds are
        # appropriate here.
        agg_signal = _derive_signal_with_mode(
            agg_eff,
            agg_ind,
            agg_pat,
            combination_mode,
            short_below,
            hold_below,
            "fixed",  # always fixed for aggregated signal
            [],       # no percentile window needed
            short_pct,
            long_pct,
            lookback_bars,
            min_fill_ratio,
            gate_indicator_min,
            gate_indicator_max,
            gate_pattern_min,
            gate_pattern_max,
        )

        signals = [r.signal for r in ok] if ok else ["HOLD"]
        agreement = _compute_agreement(signals)

        return MultiTimeframeResult(
            ticker=ticker.upper(),
            timeframe_results=results,
            aggregated_indicator_score=agg_ind,
            aggregated_pattern_score=agg_pat,
            aggregated_signal=agg_signal,
            agreement=agreement,
        )


def _compute_effective_score(
    ind_score: float,
    pat_score: float,
    combination_mode: str,
    ind_weight: float,
    pat_weight: float,
    boost_strength: float,
    boost_dead_zone: float,
) -> float:
    """Compute effective blended score (mirrors strategy logic)."""
    if combination_mode == "boost":
        pat_dev = pat_score - 5.0
        if abs(pat_dev) <= boost_dead_zone:
            return ind_score
        eff_dev = pat_dev - (boost_dead_zone if pat_dev > 0 else -boost_dead_zone)
        return max(0.0, min(10.0, ind_score + eff_dev * boost_strength))
    elif combination_mode == "gate":
        return ind_score
    else:
        # weighted (default)
        w_total = ind_weight + pat_weight
        if w_total > 0:
            return (ind_weight * ind_score + pat_weight * pat_score) / w_total
        return ind_score


def _gate_allows_trade(
    ind_score: float,
    pat_score: float,
    *,
    gate_indicator_min: float,
    gate_indicator_max: float,
    gate_pattern_min: float,
    gate_pattern_max: float,
) -> tuple[bool, bool]:
    """Return (allow_long, allow_short) for gate mode."""
    allow_long = ind_score > gate_indicator_min and pat_score > gate_pattern_min
    allow_short = ind_score < gate_indicator_max and pat_score < gate_pattern_max
    return allow_long, allow_short


def _derive_signal_with_mode(
    effective_score: float,
    ind_score: float,
    pat_score: float,
    combination_mode: str,
    short_below: float,
    hold_below: float,
    threshold_mode: str,
    score_window: list[float],
    short_pct: float,
    long_pct: float,
    lookback_bars: int,
    min_fill_ratio: float,
    gate_indicator_min: float,
    gate_indicator_max: float,
    gate_pattern_min: float,
    gate_pattern_max: float,
) -> str:
    """Derive BUY/HOLD/SELL honoring gate and percentile modes."""
    if combination_mode == "gate":
        allow_long, allow_short = _gate_allows_trade(
            ind_score,
            pat_score,
            gate_indicator_min=gate_indicator_min,
            gate_indicator_max=gate_indicator_max,
            gate_pattern_min=gate_pattern_min,
            gate_pattern_max=gate_pattern_max,
        )
        if allow_long:
            return "BUY"
        if allow_short:
            return "SELL"
        return "HOLD"

    # Percentile mode uses rolling ranks of effective scores
    if threshold_mode == "percentile":
        # Don't append the current score — the window is pre-built from
        # historical bars.  Appending would double-count the latest value.
        # Use a local slice to avoid mutating the caller's list.
        window = score_window[-lookback_bars:] if len(score_window) > lookback_bars else score_window
        min_samples = max(10, int(lookback_bars * min_fill_ratio))
        if len(window) < min_samples:
            return _derive_signal(effective_score, short_below, hold_below)
        # Use mean percentile rank (count_below + count_equal/2) / n
        # to avoid identical scores all ranking at 0 (false SELL).
        n = len(window)
        count_below = sum(1 for s in window if s < effective_score)
        count_equal = sum(1 for s in window if s == effective_score)
        rank = ((count_below + count_equal / 2) / n) * 100
        if rank <= short_pct:
            return "SELL"
        if rank >= long_pct:
            return "BUY"
        return "HOLD"

    return _derive_signal(effective_score, short_below, hold_below)


def _build_percentile_window(
    cfg: "Config",
    provider: "DataProvider",
    *,
    ticker: str,
    period: str | None,
    interval: str,
    lookback_bars: int,
    step: int,
) -> list[float]:
    """Build a rolling effective-score window for percentile signals."""
    df_scores = compute_score_timeseries(
        cfg,
        provider,
        ticker=ticker,
        period=period,
        interval=interval,
        start=None,
        end=None,
        step=step,
    )
    if df_scores.empty:
        return []
    ind_scores = df_scores["indicator_score"].tolist()
    pat_scores = df_scores["pattern_score"].tolist()
    strat_cfg = cfg.section("strategy")
    combination_mode = str(strat_cfg.get("combination_mode", "weighted"))
    ind_weight = float(strat_cfg.get("indicator_weight", 0.7))
    pat_weight = float(strat_cfg.get("pattern_weight", 0.3))
    boost_strength = float(strat_cfg.get("boost_strength", 0.5))
    boost_dead_zone = float(strat_cfg.get("boost_dead_zone", 0.3))

    eff_scores: list[float] = []
    for ind_score, pat_score in zip(ind_scores, pat_scores):
        eff = _compute_effective_score(
            float(ind_score),
            float(pat_score),
            combination_mode,
            ind_weight,
            pat_weight,
            boost_strength,
            boost_dead_zone,
        )
        eff_scores.append(eff)

    if len(eff_scores) > lookback_bars:
        return eff_scores[-lookback_bars:]
    return eff_scores
