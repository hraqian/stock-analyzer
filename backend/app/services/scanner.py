"""backend/app/services/scanner.py — Market Scanner engine.

Scans a universe of tickers for trade candidates using one of four preset
strategies (Breakout, Pullback, Reversal, Top Dividend).  Technical presets
use ``fetch_batch()`` for efficient bulk data download, then run indicators,
patterns, regime classification, and composite scoring on each ticker.

The Dividend preset wraps the existing ``DividendScanner`` from
``analysis/dividend.py``.

Designed for use inside a ThreadPoolExecutor — all calls are synchronous.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scanner result
# ---------------------------------------------------------------------------


@dataclass
class ScannerResult:
    """One row in the scanner output table."""

    rank: int
    ticker: str
    signal: str           # e.g. "Bullish Breakout", "Pullback Buy", "Reversal"
    score: float          # composite score (0-10)
    confidence: float     # 0.0-1.0
    pattern: str          # dominant pattern detected (or "")
    regime: str           # regime label (e.g. "Strong Trend")
    sector: str           # GICS sector (from info or "N/A")

    # Extra detail for frontend tooltips / power-user mode
    breakdown: dict[str, float] = field(default_factory=dict)
    volume: float = 0.0
    price: float = 0.0
    atr_ratio: float = 0.0
    ai_rating: float | None = None


@dataclass
class ScanSummary:
    """Summary stats for the scan."""

    preset: str
    universe: str
    total_tickers: int
    tickers_with_data: int
    tickers_passing_filters: int
    elapsed_seconds: float
    results: list[ScannerResult]


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

# Which indicators to compute for each technical preset.
# Using a subset is much faster than running all 8 indicators.
_PRESET_INDICATORS: dict[str, list[str]] = {
    "breakout": ["rsi", "macd", "bollinger_bands", "volume", "adx"],
    "pullback": ["rsi", "macd", "moving_averages", "stochastic"],
    "reversal": ["rsi", "stochastic", "bollinger_bands", "macd"],
}


def _signal_label(preset: str, score: float) -> str:
    """Generate a human-readable signal label."""
    if preset == "breakout":
        if score >= 7.0:
            return "Strong Breakout"
        elif score >= 5.5:
            return "Breakout Setup"
        return "Potential Breakout"
    elif preset == "pullback":
        if score >= 7.0:
            return "Strong Pullback Buy"
        elif score >= 5.5:
            return "Pullback Buy"
        return "Potential Pullback"
    elif preset == "reversal":
        if score >= 7.0:
            return "Strong Reversal"
        elif score >= 5.5:
            return "Reversal Signal"
        return "Potential Reversal"
    return "Signal"


# ---------------------------------------------------------------------------
# Main scanner function
# ---------------------------------------------------------------------------


def run_scan(
    tickers: list[str],
    preset: str,
    trade_mode: str = "swing",
    period: str = "6mo",
    min_volume: int = 1_000_000,
    min_price: float = 5.0,
    max_atr_ratio: float | None = None,
    top_n: int = 50,
    universe_name: str = "custom",
) -> ScanSummary:
    """Run a scan on the given tickers.

    This function is synchronous and should be called from a thread pool.

    Args:
        tickers:       List of ticker symbols to scan.
        preset:        Scan preset: "breakout", "pullback", "reversal", "dividend".
        trade_mode:    User's trade mode ("swing" or "long_term").
        period:        Data period for fetching (e.g. "6mo").
        min_volume:    Minimum average volume filter.
        min_price:     Minimum price filter.
        max_atr_ratio: Maximum ATR/price ratio filter (None = no filter).
        top_n:         Max number of results to return.
        universe_name: Name of the universe (for display).

    Returns:
        ScanSummary with ranked results.
    """
    t0 = time.time()

    if preset == "dividend":
        return _run_dividend_scan(
            tickers, trade_mode, min_volume, min_price, top_n, universe_name, t0,
        )

    return _run_technical_scan(
        tickers, preset, trade_mode, period, min_volume, min_price,
        max_atr_ratio, top_n, universe_name, t0,
    )


def _run_technical_scan(
    tickers: list[str],
    preset: str,
    trade_mode: str,
    period: str,
    min_volume: int,
    min_price: float,
    max_atr_ratio: float | None,
    top_n: int,
    universe_name: str,
    t0: float,
) -> ScanSummary:
    """Technical scan: fetch batch → indicators → patterns → score → rank."""
    # Late imports — engine modules are volume-mounted, not installed
    from config import Config                          # type: ignore[import-untyped]
    from data.yahoo import fetch_batch                 # type: ignore[import-untyped]
    from indicators.registry import IndicatorRegistry  # type: ignore[import-untyped]
    from patterns.registry import PatternRegistry      # type: ignore[import-untyped]
    from analysis.scorer import CompositeScorer        # type: ignore[import-untyped]
    from engine.regime import RegimeClassifier         # type: ignore[import-untyped]

    # Build config with trade-mode objective
    cfg = Config.defaults()
    objective_map = {"swing": "swing_trade", "long_term": "long_term"}
    objective = objective_map.get(trade_mode)
    if objective and objective in cfg.available_objectives():
        cfg.apply_objective(objective)

    # ── 1. Batch fetch ───────────────────────────────────────────────
    logger.info("Scanner: fetching %d tickers for preset=%s", len(tickers), preset)
    data = fetch_batch(tickers, period=period)
    logger.info("Scanner: got data for %d / %d tickers", len(data), len(tickers))

    # ── 2. Build reusable engine instances ───────────────────────────
    indicator_keys = _PRESET_INDICATORS.get(preset, ["rsi", "macd", "bollinger_bands"])
    ind_registry = IndicatorRegistry(cfg, only=indicator_keys)
    pat_registry = PatternRegistry(cfg)
    scorer = CompositeScorer(cfg)
    regime_clf = RegimeClassifier(cfg)

    # ── 3. Process each ticker ───────────────────────────────────────
    candidates: list[ScannerResult] = []

    for ticker, df in data.items():
        try:
            result = _analyze_one(
                ticker, df, ind_registry, pat_registry, scorer, regime_clf,
                preset, min_volume, min_price, max_atr_ratio,
            )
            if result is not None:
                candidates.append(result)
        except Exception:
            logger.debug("Scanner: error processing %s", ticker, exc_info=True)
            continue

    # ── 4. Rank by score descending ──────────────────────────────────
    candidates.sort(key=lambda r: r.score, reverse=True)
    top = candidates[:top_n]
    for i, r in enumerate(top, 1):
        r.rank = i

    # ── 5. ML scoring (if model exists) ──────────────────────────────
    try:
        from engine.ml_model import model_exists, predict_signal  # type: ignore[import-untyped]
        from analysis.pattern_scorer import PatternCompositeScorer  # type: ignore[import-untyped]

        if model_exists():
            pat_scorer = PatternCompositeScorer(cfg)
            # Full indicator registry for ML (scanner may use a subset)
            ml_ind_registry = IndicatorRegistry(cfg)
            ml_pat_registry = PatternRegistry(cfg)
            ml_scorer = CompositeScorer(cfg)
            ml_regime_clf = RegimeClassifier(cfg)

            for r in top:
                try:
                    ticker_df = data.get(r.ticker)
                    if ticker_df is None or ticker_df.empty or len(ticker_df) < 20:
                        continue
                    ml_indicators = ml_ind_registry.run_all(ticker_df)
                    ml_patterns = ml_pat_registry.run_all(ticker_df)
                    ml_composite = ml_scorer.score(ml_indicators)
                    ml_pat_composite = pat_scorer.score(ml_patterns)
                    ml_regime = None
                    try:
                        ml_regime = ml_regime_clf.classify(ticker_df)
                    except Exception:
                        pass
                    prediction = predict_signal(
                        ml_indicators, ml_patterns,
                        ml_composite, ml_pat_composite,
                        ml_regime, ticker_df,
                    )
                    if prediction is not None:
                        r.ai_rating = prediction.ai_rating
                except Exception:
                    logger.debug("ML scoring failed for %s", r.ticker, exc_info=True)
    except ImportError:
        logger.debug("ML model modules not available, skipping AI rating")

    elapsed = time.time() - t0
    logger.info(
        "Scanner: %d candidates from %d tickers (%.1fs)",
        len(top), len(tickers), elapsed,
    )

    return ScanSummary(
        preset=preset,
        universe=universe_name,
        total_tickers=len(tickers),
        tickers_with_data=len(data),
        tickers_passing_filters=len(candidates),
        elapsed_seconds=round(elapsed, 2),
        results=top,
    )


def _analyze_one(
    ticker: str,
    df: pd.DataFrame,
    ind_registry: "IndicatorRegistry",
    pat_registry: "PatternRegistry",
    scorer: "CompositeScorer",
    regime_clf: "RegimeClassifier",
    preset: str,
    min_volume: int,
    min_price: float,
    max_atr_ratio: float | None,
) -> ScannerResult | None:
    """Analyze a single ticker and return a ScannerResult, or None if filtered out."""
    if df.empty or len(df) < 20:
        return None

    # ── Price filter ─────────────────────────────────────────────────
    last_close = float(df["close"].iloc[-1])
    if last_close < min_price:
        return None

    # ── Volume filter (average over last 20 bars) ────────────────────
    avg_volume = float(df["volume"].tail(20).mean()) if "volume" in df.columns else 0
    if avg_volume < min_volume:
        return None

    # ── ATR ratio filter ─────────────────────────────────────────────
    atr_ratio = _compute_atr_ratio(df)
    if max_atr_ratio is not None and atr_ratio > max_atr_ratio:
        return None

    # ── Run indicators ───────────────────────────────────────────────
    try:
        indicator_results = ind_registry.run_all(df)
    except Exception:
        logger.debug("Indicators failed for %s", ticker, exc_info=True)
        return None

    # ── Composite score ──────────────────────────────────────────────
    composite = scorer.score(indicator_results)
    score = float(composite.get("overall", 5.0))

    # ── Preset-specific score adjustments ────────────────────────────
    score = _apply_preset_adjustments(score, preset, indicator_results, df)

    # ── Patterns ─────────────────────────────────────────────────────
    dominant_pattern = ""
    try:
        pattern_results = pat_registry.run_all(df)
        # Find the pattern with the most extreme score (away from 5.0)
        best_dev = 0.0
        for pr in pattern_results:
            if pr.error:
                continue
            dev = abs(pr.score - 5.0)
            if dev > best_dev:
                best_dev = dev
                dominant_pattern = pr.name
    except Exception:
        pass

    # ── Regime ───────────────────────────────────────────────────────
    regime_label = ""
    regime_confidence = 0.0
    try:
        regime = regime_clf.classify(df)
        regime_label = regime.label
        regime_confidence = regime.confidence
    except Exception:
        pass

    # ── Build result ─────────────────────────────────────────────────
    signal = _signal_label(preset, score)
    breakdown = {k: round(v, 2) for k, v in composite.get("breakdown", {}).items()}

    return ScannerResult(
        rank=0,  # assigned after sorting
        ticker=ticker,
        signal=signal,
        score=round(score, 2),
        confidence=round(regime_confidence, 2),
        pattern=dominant_pattern,
        regime=regime_label,
        sector="",  # populated by frontend or info lookup if needed
        breakdown=breakdown,
        volume=round(avg_volume, 0),
        price=round(last_close, 2),
        atr_ratio=round(atr_ratio, 4),
    )


def _compute_atr_ratio(df: pd.DataFrame, period: int = 14) -> float:
    """ATR as a fraction of current price."""
    if len(df) < period + 1:
        return 0.0
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = float(tr.rolling(window=period).mean().iloc[-1])
    last_close = float(close.iloc[-1])
    if last_close <= 0:
        return 0.0
    return atr / last_close


def _apply_preset_adjustments(
    score: float,
    preset: str,
    indicator_results: list,
    df: pd.DataFrame,
) -> float:
    """Apply preset-specific score bonuses/penalties.

    These adjustments reward conditions that match the preset's intent:
    - Breakout: bonus for high volume, BB squeeze, rising ADX
    - Pullback: bonus for oversold RSI near rising MAs
    - Reversal: bonus for extreme RSI + bullish divergence signals
    """
    adj = 0.0
    ind_map = {ir.config_key: ir for ir in indicator_results if not ir.error}

    if preset == "breakout":
        # Reward BB squeeze (bollinger score > 6 means near upper band or squeeze)
        bb = ind_map.get("bollinger_bands")
        if bb and bb.score >= 6.5:
            adj += 0.5
        # Reward strong volume
        vol = ind_map.get("volume")
        if vol and vol.score >= 7.0:
            adj += 0.5
        # Reward rising ADX
        adx = ind_map.get("adx")
        if adx and adx.score >= 6.5:
            adj += 0.3

    elif preset == "pullback":
        # Reward oversold RSI (contrarian buy)
        rsi = ind_map.get("rsi")
        if rsi and rsi.score >= 7.0:
            adj += 0.5
        # Reward price near moving averages (bounce zone)
        ma = ind_map.get("moving_averages")
        if ma:
            price_near_ma = ma.values.get("price_near_ma", False)
            if price_near_ma:
                adj += 0.3

    elif preset == "reversal":
        # Reward extreme RSI (oversold reversal)
        rsi = ind_map.get("rsi")
        if rsi and rsi.score >= 8.0:
            adj += 0.7
        # Reward stochastic oversold
        stoch = ind_map.get("stochastic")
        if stoch and stoch.score >= 7.0:
            adj += 0.3

    return min(10.0, max(0.0, score + adj))


# ---------------------------------------------------------------------------
# Dividend preset — wraps existing DividendScanner
# ---------------------------------------------------------------------------


def _run_dividend_scan(
    tickers: list[str],
    trade_mode: str,
    min_volume: int,
    min_price: float,
    top_n: int,
    universe_name: str,
    t0: float,
) -> ScanSummary:
    """Run the dividend scan using the existing DividendScanner engine."""
    from config import Config                            # type: ignore[import-untyped]
    from analysis.dividend import DividendScanner        # type: ignore[import-untyped]

    cfg = Config.defaults()
    div_cfg = cfg.section("dividend")

    scanner = DividendScanner(tickers, div_cfg)
    scanner.run()
    top = scanner.top(n=top_n)

    results: list[ScannerResult] = []
    for i, dr in enumerate(top, 1):
        results.append(ScannerResult(
            rank=i,
            ticker=dr.ticker,
            signal=_dividend_signal_label(dr.dividend_score),
            score=round(dr.dividend_score, 2),
            confidence=round(dr.payout_consistency, 2),
            pattern=f"Streak: {dr.increase_streak}yr",
            regime="",
            sector=dr.sector or "",
            breakdown={
                "yield": dr.yield_score,
                "growth": dr.growth_score,
                "consistency": dr.consistency_score,
                "streak": dr.streak_score,
            },
            volume=0.0,
            price=round(dr.current_price, 2),
            atr_ratio=0.0,
        ))

    summary = scanner.summary()
    elapsed = time.time() - t0

    return ScanSummary(
        preset="dividend",
        universe=universe_name,
        total_tickers=summary["total_tickers"],
        tickers_with_data=summary["scanned"],
        tickers_passing_filters=summary["passed_filters"],
        elapsed_seconds=round(elapsed, 2),
        results=results,
    )


def _dividend_signal_label(score: float) -> str:
    """Human-readable dividend signal label."""
    if score >= 8.0:
        return "Top Dividend"
    elif score >= 6.0:
        return "Strong Dividend"
    elif score >= 4.0:
        return "Moderate Dividend"
    return "Dividend"
