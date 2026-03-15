"""
engine/ml_features.py — Feature extraction for XGBoost signal scoring.

Transforms raw indicator results, pattern results, regime assessment,
and price context into a flat numeric feature vector suitable for
XGBoost classification.

Feature groups:
  1. Indicator values (RSI, MACD histogram, BB %B, MA slopes, etc.)
  2. Indicator scores (0-10 composite scores from each indicator)
  3. Pattern signals (encoded as numeric: bullish=1, bearish=-1, neutral=0)
  4. Regime classification (one-hot encoded)
  5. Volume context (current vs average ratio)
  6. Volatility context (ATR/price ratio)
  7. Multi-timeframe alignment score
  8. Composite scores (trend, contrarian, overall)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


# ── Feature names (stable ordering for model consistency) ──────────────

FEATURE_NAMES: list[str] = [
    # Indicator raw values
    "rsi",
    "rsi_prev",
    "macd_histogram",
    "macd_histogram_pct",       # histogram / price
    "macd_bullish_cross",
    "macd_bearish_cross",
    "bb_pct_b",
    "bb_bandwidth_pct",         # band_width / middle
    "bb_squeeze",
    "stochastic_k",
    "stochastic_d",
    "stochastic_bullish_cross",
    "stochastic_bearish_cross",
    "adx",
    "adx_plus_di",
    "adx_minus_di",
    "adx_di_spread",            # plus_di - minus_di
    "obv_change_pct",
    "obv_confirming",           # 1 if confirming, 0 if diverging
    "ma_price_above_count",     # how many MAs price is above (normalised)
    "ma_alignment_score",       # bullish alignment fraction
    "ma_golden_cross",
    "ma_death_cross",
    # Indicator scores (0-10)
    "score_rsi",
    "score_macd",
    "score_bollinger",
    "score_ma",
    "score_stochastic",
    "score_adx",
    "score_volume",
    "score_fibonacci",
    # Pattern scores (0-10)
    "score_gaps",
    "score_volume_range",
    "score_candlesticks",
    "score_spikes",
    "score_inside_outside",
    # Composite scores
    "composite_overall",
    "composite_trend",
    "composite_contrarian",
    "pattern_composite_overall",
    # Regime (one-hot)
    "regime_strong_trend",
    "regime_mean_reverting",
    "regime_volatile_choppy",
    "regime_breakout_transition",
    "regime_confidence",
    # Regime sub-type (one-hot)
    "regime_sub_explosive",
    "regime_sub_volatile_directionless",
    "regime_sub_steady",
    "regime_sub_stagnant",
    # Volume context
    "volume_ratio",             # current volume / 20-day avg
    # Volatility context
    "atr_price_ratio",          # ATR(14) / close price
    # Price context
    "price_change_5d_pct",
    "price_change_20d_pct",
]

NUM_FEATURES = len(FEATURE_NAMES)


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Convert to float, handling NaN/None gracefully."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return default


def _bool_to_float(val: Any) -> float:
    """Convert bool-like to 1.0/0.0."""
    return 1.0 if val else 0.0


# ── Main extraction function ──────────────────────────────────────────


def extract_features(
    indicator_results: list[Any],
    pattern_results: list[Any],
    composite: dict[str, Any],
    pattern_composite: dict[str, Any],
    regime: Any | None,
    df: pd.DataFrame,
) -> np.ndarray:
    """Extract a flat feature vector from analysis results.

    Args:
        indicator_results: List of IndicatorResult from the indicator registry.
        pattern_results: List of PatternResult from the pattern registry.
        composite: Dict from CompositeScorer.score() (indicator composite).
        pattern_composite: Dict from PatternCompositeScorer.score().
        regime: RegimeAssessment object or None.
        df: OHLCV DataFrame used for price/volume context.

    Returns:
        np.ndarray of shape (NUM_FEATURES,) with dtype float32.
    """
    features = np.zeros(NUM_FEATURES, dtype=np.float32)

    # Build lookup dicts by config_key
    ind_by_key: dict[str, Any] = {}
    for r in indicator_results:
        key = getattr(r, "config_key", None) or getattr(r, "name", "")
        ind_by_key[key] = r

    pat_by_key: dict[str, Any] = {}
    for r in pattern_results:
        key = getattr(r, "config_key", None) or getattr(r, "name", "")
        pat_by_key[key] = r

    # ── 1. Indicator raw values ───────────────────────────────────────

    # RSI
    rsi_r = ind_by_key.get("rsi")
    if rsi_r and not rsi_r.error:
        v = rsi_r.values
        features[0] = _safe_float(v.get("rsi"), 50.0)       # rsi
        features[1] = _safe_float(v.get("rsi_prev"), 50.0)   # rsi_prev

    # MACD
    macd_r = ind_by_key.get("macd")
    if macd_r and not macd_r.error:
        v = macd_r.values
        features[2] = _safe_float(v.get("histogram"))        # macd_histogram
        price = _safe_float(v.get("price"), 1.0)
        features[3] = features[2] / price if price != 0 else 0.0  # macd_histogram_pct
        features[4] = _bool_to_float(v.get("bullish_cross"))
        features[5] = _bool_to_float(v.get("bearish_cross"))

    # Bollinger Bands
    bb_r = ind_by_key.get("bollinger_bands")
    if bb_r and not bb_r.error:
        v = bb_r.values
        features[6] = _safe_float(v.get("pct_b"), 0.5)       # bb_pct_b
        middle = _safe_float(v.get("middle"), 1.0)
        bw = _safe_float(v.get("band_width"))
        features[7] = bw / middle if middle != 0 else 0.0    # bb_bandwidth_pct
        features[8] = _bool_to_float(v.get("squeeze"))

    # Stochastic
    stoch_r = ind_by_key.get("stochastic")
    if stoch_r and not stoch_r.error:
        v = stoch_r.values
        features[9] = _safe_float(v.get("k"), 50.0)
        features[10] = _safe_float(v.get("d"), 50.0)
        features[11] = _bool_to_float(v.get("bullish_cross"))
        features[12] = _bool_to_float(v.get("bearish_cross"))

    # ADX
    adx_r = ind_by_key.get("adx")
    if adx_r and not adx_r.error:
        v = adx_r.values
        features[13] = _safe_float(v.get("adx"), 25.0)
        features[14] = _safe_float(v.get("plus_di"), 20.0)
        features[15] = _safe_float(v.get("minus_di"), 20.0)
        features[16] = features[14] - features[15]           # di_spread

    # Volume / OBV
    vol_r = ind_by_key.get("volume")
    if vol_r and not vol_r.error:
        v = vol_r.values
        features[17] = _safe_float(v.get("obv_change_pct"))
        features[18] = 1.0 if v.get("signal") == "confirming" else 0.0

    # Moving Averages
    ma_r = ind_by_key.get("moving_averages")
    if ma_r and not ma_r.error:
        v = ma_r.values
        price = _safe_float(v.get("price"), 0.0)
        ma_vals = v.get("ma_values", {})
        valid_mas = {p: val for p, val in ma_vals.items()
                     if not (isinstance(val, float) and math.isnan(val))}
        n = len(valid_mas)

        # Count how many MAs price is above
        above = sum(1 for val in valid_mas.values() if price > val) if n > 0 else 0
        features[19] = above / n if n > 0 else 0.5           # normalised

        # MA alignment score (fraction of adjacent pairs in bullish order)
        sorted_mas = sorted(valid_mas.items())
        aligned = 0
        pairs = max(len(sorted_mas) - 1, 1)
        for i in range(len(sorted_mas) - 1):
            if sorted_mas[i][1] > sorted_mas[i + 1][1]:
                aligned += 1
        features[20] = aligned / pairs if pairs > 0 else 0.5

        features[21] = _bool_to_float(v.get("golden_cross"))
        features[22] = _bool_to_float(v.get("death_cross"))

    # ── 2. Indicator scores ───────────────────────────────────────────

    score_map = {
        "rsi": 23,
        "macd": 24,
        "bollinger_bands": 25,
        "moving_averages": 26,
        "stochastic": 27,
        "adx": 28,
        "volume": 29,
        "fibonacci": 30,
    }
    for key, idx in score_map.items():
        r = ind_by_key.get(key)
        if r and not r.error:
            features[idx] = _safe_float(r.score, 5.0)
        else:
            features[idx] = 5.0  # neutral default

    # ── 3. Pattern scores ─────────────────────────────────────────────

    pat_score_map = {
        "gaps": 31,
        "volume_range": 32,
        "candlesticks": 33,
        "spikes": 34,
        "inside_outside": 35,
    }
    for key, idx in pat_score_map.items():
        r = pat_by_key.get(key)
        if r and not r.error:
            features[idx] = _safe_float(r.score, 5.0)
        else:
            features[idx] = 5.0

    # ── 4. Composite scores ───────────────────────────────────────────

    features[36] = _safe_float(composite.get("overall"), 5.0)
    features[37] = _safe_float(composite.get("trend_score"), 5.0)
    features[38] = _safe_float(composite.get("contrarian_score"), 5.0)
    features[39] = _safe_float(pattern_composite.get("overall"), 5.0)

    # ── 5. Regime (one-hot) ───────────────────────────────────────────

    if regime is not None:
        regime_type = getattr(regime, "regime", None)
        if regime_type is not None:
            rt_name = regime_type.name if hasattr(regime_type, "name") else str(regime_type)
            regime_map = {
                "STRONG_TREND": 40,
                "MEAN_REVERTING": 41,
                "VOLATILE_CHOPPY": 42,
                "BREAKOUT_TRANSITION": 43,
            }
            idx = regime_map.get(rt_name)
            if idx is not None:
                features[idx] = 1.0

        features[44] = _safe_float(getattr(regime, "confidence", 0.0))

        sub_type = getattr(regime, "sub_type", None)
        if sub_type is not None:
            st_name = sub_type.name if hasattr(sub_type, "name") else str(sub_type)
            sub_map = {
                "EXPLOSIVE_MOVER": 45,
                "VOLATILE_DIRECTIONLESS": 46,
                "STEADY_COMPOUNDER": 47,
                "STAGNANT": 48,
            }
            idx = sub_map.get(st_name)
            if idx is not None:
                features[idx] = 1.0

    # ── 6. Volume context ─────────────────────────────────────────────

    if len(df) >= 20:
        current_vol = float(df["volume"].iloc[-1])
        avg_vol = float(df["volume"].iloc[-20:].mean())
        features[49] = current_vol / avg_vol if avg_vol > 0 else 1.0
    else:
        features[49] = 1.0

    # ── 7. Volatility context (ATR/price) ─────────────────────────────

    if len(df) >= 14:
        try:
            import ta as ta_lib
            atr_series = ta_lib.volatility.AverageTrueRange(
                high=df["high"], low=df["low"], close=df["close"], window=14
            ).average_true_range()
            atr = float(atr_series.iloc[-1])
            close_price = float(df["close"].iloc[-1])
            features[50] = atr / close_price if close_price > 0 else 0.0
        except Exception:
            features[50] = 0.0
    else:
        features[50] = 0.0

    # ── 8. Price context ──────────────────────────────────────────────

    close_series = df["close"]
    if len(close_series) >= 6:
        cur = float(close_series.iloc[-1])
        prev5 = float(close_series.iloc[-6])
        features[51] = (cur - prev5) / prev5 if prev5 != 0 else 0.0
    if len(close_series) >= 21:
        cur = float(close_series.iloc[-1])
        prev20 = float(close_series.iloc[-21])
        features[52] = (cur - prev20) / prev20 if prev20 != 0 else 0.0

    return features


def extract_features_dict(
    indicator_results: list[Any],
    pattern_results: list[Any],
    composite: dict[str, Any],
    pattern_composite: dict[str, Any],
    regime: Any | None,
    df: pd.DataFrame,
) -> dict[str, float]:
    """Same as extract_features() but returns a name→value dict.

    Useful for debugging and displaying feature importances.
    """
    arr = extract_features(
        indicator_results, pattern_results,
        composite, pattern_composite,
        regime, df,
    )
    return {name: float(arr[i]) for i, name in enumerate(FEATURE_NAMES)}
