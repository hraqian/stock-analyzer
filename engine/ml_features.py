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
  9. Temporal/momentum features (rate-of-change for key indicators)
  10. Cross-asset context (SPY returns, VIX, market breadth)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


# ── Feature names (stable ordering for model consistency) ──────────────

FEATURE_NAMES: list[str] = [
    # ── Group 1: Indicator raw values (23 features, indices 0-22) ──────
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

    # ── Group 2: Indicator scores 0-10 (8 features, indices 23-30) ─────
    "score_rsi",
    "score_macd",
    "score_bollinger",
    "score_ma",
    "score_stochastic",
    "score_adx",
    "score_volume",
    "score_fibonacci",

    # ── Group 3: Pattern scores 0-10 (5 features, indices 31-35) ───────
    "score_gaps",
    "score_volume_range",
    "score_candlesticks",
    "score_spikes",
    "score_inside_outside",

    # ── Group 4: Composite scores (4 features, indices 36-39) ──────────
    "composite_overall",
    "composite_trend",
    "composite_contrarian",
    "pattern_composite_overall",

    # ── Group 5: Regime one-hot (9 features, indices 40-48) ────────────
    "regime_strong_trend",
    "regime_mean_reverting",
    "regime_volatile_choppy",
    "regime_breakout_transition",
    "regime_confidence",
    "regime_sub_explosive",
    "regime_sub_volatile_directionless",
    "regime_sub_steady",
    "regime_sub_stagnant",

    # ── Group 6: Volume context (1 feature, index 49) ─────────────────
    "volume_ratio",             # current volume / 20-day avg

    # ── Group 7: Volatility context (1 feature, index 50) ─────────────
    "atr_price_ratio",          # ATR(14) / close price

    # ── Group 8: Price context (2 features, indices 51-52) ─────────────
    "price_change_5d_pct",
    "price_change_20d_pct",

    # ── Group 9: Temporal/momentum features (18 features, indices 53-70) ──
    "rsi_change_5",             # RSI change over 5 bars
    "rsi_oversold",             # 1 if RSI < 30
    "rsi_overbought",           # 1 if RSI > 70
    "macd_hist_slope_5",        # MACD histogram slope (change / 5 bars)
    "stoch_k_change_5",         # Stochastic K change over 5 bars
    "adx_change_5",             # ADX change over 5 bars
    "bb_pct_b_change_5",        # BB %B change over 5 bars
    "price_dist_ma20_pct",      # (price - MA20) / MA20
    "price_dist_ma50_pct",      # (price - MA50) / MA50
    "price_dist_ma200_pct",     # (price - MA200) / MA200
    "vol_trend_ratio",          # volume SMA(5) / SMA(20) — rising vol?
    "close_above_open_streak",  # consecutive bullish bars (normalised)
    "high_low_range_pct",       # (high - low) / close — intrabar volatility
    "price_change_1d_pct",      # 1-day return
    "price_change_3d_pct",      # 3-day return
    "rsi_ma_divergence",        # RSI direction vs price direction (divergence signal)
    "composite_change_proxy",   # |composite - 5| / 5 — strength of signal
    "score_dispersion",         # stdev of indicator scores — consensus vs mixed

    # ── Group 10: Cross-asset context (6 features, indices 71-76) ──────
    "spy_return_5d_pct",        # SPY 5-day return
    "spy_return_20d_pct",       # SPY 20-day return
    "spy_dist_ma50_pct",        # SPY distance from 50-day MA
    "vix_level",                # VIX close (or 0 if unavailable)
    "vix_change_5d_pct",        # VIX 5-day change
    "stock_spy_corr_20",        # 20-day rolling correlation with SPY
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


def _pct_change(series: pd.Series, periods: int) -> float:
    """Compute percentage change over N periods at end of series."""
    if len(series) < periods + 1:
        return 0.0
    cur = float(series.iloc[-1])
    prev = float(series.iloc[-periods - 1])
    if prev == 0 or math.isnan(prev) or math.isnan(cur):
        return 0.0
    return (cur - prev) / abs(prev)


def _sma(series: pd.Series, window: int) -> float:
    """Simple moving average of last `window` values."""
    if len(series) < window:
        return float("nan")
    return float(series.iloc[-window:].mean())


# ── Main extraction function ──────────────────────────────────────────


def extract_features(
    indicator_results: list[Any],
    pattern_results: list[Any],
    composite: dict[str, Any],
    pattern_composite: dict[str, Any],
    regime: Any | None,
    df: pd.DataFrame,
    *,
    spy_df: pd.DataFrame | None = None,
    vix_df: pd.DataFrame | None = None,
) -> np.ndarray:
    """Extract a flat feature vector from analysis results.

    Args:
        indicator_results: List of IndicatorResult from the indicator registry.
        pattern_results: List of PatternResult from the pattern registry.
        composite: Dict from CompositeScorer.score() (indicator composite).
        pattern_composite: Dict from PatternCompositeScorer.score().
        regime: RegimeAssessment object or None.
        df: OHLCV DataFrame used for price/volume context.
        spy_df: Optional SPY OHLCV DataFrame (same date range) for cross-asset.
        vix_df: Optional VIX OHLCV DataFrame for cross-asset.

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
    rsi_val = 50.0
    if rsi_r and not rsi_r.error:
        v = rsi_r.values
        rsi_val = _safe_float(v.get("rsi"), 50.0)
        features[0] = rsi_val                                  # rsi
        features[1] = _safe_float(v.get("rsi_prev"), 50.0)    # rsi_prev

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
    ma_vals: dict[Any, float] = {}
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
    ind_scores: list[float] = []
    for key, idx in score_map.items():
        r = ind_by_key.get(key)
        if r and not r.error:
            s = _safe_float(r.score, 5.0)
            features[idx] = s
            ind_scores.append(s)
        else:
            features[idx] = 5.0  # neutral default
            ind_scores.append(5.0)

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

    composite_overall = _safe_float(composite.get("overall"), 5.0)
    features[36] = composite_overall
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

    # ── 9. Temporal / momentum features ───────────────────────────────

    # 9a. RSI change over 5 bars
    # We only have current & prev RSI from the indicator — compute
    # 5-bar RSI change from the close series using pandas_ta / ta lib
    # as a lightweight proxy.
    features[53] = 0.0  # rsi_change_5 — filled below
    features[54] = 1.0 if rsi_val < 30 else 0.0   # rsi_oversold
    features[55] = 1.0 if rsi_val > 70 else 0.0   # rsi_overbought

    if len(close_series) >= 20:
        try:
            import ta as ta_lib
            rsi_series = ta_lib.momentum.RSIIndicator(
                close=close_series, window=14,
            ).rsi()
            if len(rsi_series) >= 6:
                rsi_now = float(rsi_series.iloc[-1])
                rsi_5ago = float(rsi_series.iloc[-6])
                if not (math.isnan(rsi_now) or math.isnan(rsi_5ago)):
                    features[53] = rsi_now - rsi_5ago   # rsi_change_5
        except Exception:
            pass

    # 9b. MACD histogram slope over 5 bars
    if len(close_series) >= 34:  # need 26+8 bars for MACD
        try:
            import ta as ta_lib
            macd_ind = ta_lib.trend.MACD(
                close=close_series, window_slow=26, window_fast=12, window_sign=9,
            )
            hist = macd_ind.macd_diff()
            if len(hist) >= 6:
                h_now = float(hist.iloc[-1])
                h_5ago = float(hist.iloc[-6])
                if not (math.isnan(h_now) or math.isnan(h_5ago)):
                    features[56] = (h_now - h_5ago) / 5.0  # macd_hist_slope_5
        except Exception:
            pass

    # 9c. Stochastic K change over 5 bars
    if len(df) >= 19:  # need 14+5 bars
        try:
            import ta as ta_lib
            stoch = ta_lib.momentum.StochasticOscillator(
                high=df["high"], low=df["low"], close=close_series,
                window=14, smooth_window=3,
            )
            k_series = stoch.stoch()
            if len(k_series) >= 6:
                k_now = float(k_series.iloc[-1])
                k_5ago = float(k_series.iloc[-6])
                if not (math.isnan(k_now) or math.isnan(k_5ago)):
                    features[57] = k_now - k_5ago  # stoch_k_change_5
        except Exception:
            pass

    # 9d. ADX change over 5 bars
    if len(df) >= 19:
        try:
            import ta as ta_lib
            adx_ind = ta_lib.trend.ADXIndicator(
                high=df["high"], low=df["low"], close=close_series, window=14,
            )
            adx_series = adx_ind.adx()
            if len(adx_series) >= 6:
                a_now = float(adx_series.iloc[-1])
                a_5ago = float(adx_series.iloc[-6])
                if not (math.isnan(a_now) or math.isnan(a_5ago)):
                    features[58] = a_now - a_5ago  # adx_change_5
        except Exception:
            pass

    # 9e. BB %B change over 5 bars
    if len(close_series) >= 25:
        try:
            import ta as ta_lib
            bb = ta_lib.volatility.BollingerBands(
                close=close_series, window=20, window_dev=2,
            )
            pct_b = bb.bollinger_pband()
            if len(pct_b) >= 6:
                b_now = float(pct_b.iloc[-1])
                b_5ago = float(pct_b.iloc[-6])
                if not (math.isnan(b_now) or math.isnan(b_5ago)):
                    features[59] = b_now - b_5ago  # bb_pct_b_change_5
        except Exception:
            pass

    # 9f. Price distance from key MAs (%, signed)
    cur_price = float(close_series.iloc[-1]) if len(close_series) > 0 else 0.0

    if len(close_series) >= 20:
        ma20 = _sma(close_series, 20)
        if not math.isnan(ma20) and ma20 > 0:
            features[60] = (cur_price - ma20) / ma20  # price_dist_ma20_pct
    if len(close_series) >= 50:
        ma50 = _sma(close_series, 50)
        if not math.isnan(ma50) and ma50 > 0:
            features[61] = (cur_price - ma50) / ma50  # price_dist_ma50_pct
    if len(close_series) >= 200:
        ma200 = _sma(close_series, 200)
        if not math.isnan(ma200) and ma200 > 0:
            features[62] = (cur_price - ma200) / ma200  # price_dist_ma200_pct

    # 9g. Volume trend ratio: SMA(vol,5) / SMA(vol,20)
    if len(df) >= 20:
        vol_sma5 = _sma(df["volume"], 5)
        vol_sma20 = _sma(df["volume"], 20)
        if not math.isnan(vol_sma5) and not math.isnan(vol_sma20) and vol_sma20 > 0:
            features[63] = vol_sma5 / vol_sma20  # vol_trend_ratio

    # 9h. Consecutive bullish bars (normalised to 0-1, max 10)
    if len(df) >= 2:
        streak = 0
        for i in range(len(df) - 1, max(len(df) - 11, -1), -1):
            if float(df["close"].iloc[i]) >= float(df["open"].iloc[i]):
                streak += 1
            else:
                break
        features[64] = streak / 10.0  # close_above_open_streak

    # 9i. Intra-bar range as % of close
    if len(df) >= 1:
        h = float(df["high"].iloc[-1])
        lo = float(df["low"].iloc[-1])
        c = float(df["close"].iloc[-1])
        features[65] = (h - lo) / c if c > 0 else 0.0  # high_low_range_pct

    # 9j. 1-day and 3-day returns
    if len(close_series) >= 2:
        features[66] = _pct_change(close_series, 1)  # price_change_1d_pct
    if len(close_series) >= 4:
        features[67] = _pct_change(close_series, 3)  # price_change_3d_pct

    # 9k. RSI-price divergence: RSI falling while price rising (or vice versa)
    # Simple proxy: sign(price_change_5d) != sign(rsi_change_5)
    price_5d = features[51]  # already computed
    rsi_5d = features[53]
    if price_5d != 0 and rsi_5d != 0:
        diverging = (price_5d > 0 and rsi_5d < 0) or (price_5d < 0 and rsi_5d > 0)
        features[68] = 1.0 if diverging else 0.0  # rsi_ma_divergence

    # 9l. Composite signal strength: |composite - 5| / 5
    features[69] = abs(composite_overall - 5.0) / 5.0  # composite_change_proxy

    # 9m. Score dispersion: stdev of indicator scores — high = mixed signals
    if ind_scores:
        features[70] = float(np.std(ind_scores))  # score_dispersion

    # ── 10. Cross-asset context ───────────────────────────────────────

    # SPY features
    if spy_df is not None and len(spy_df) >= 2:
        spy_close = spy_df["close"]
        if len(spy_close) >= 6:
            features[71] = _pct_change(spy_close, 5)    # spy_return_5d_pct
        if len(spy_close) >= 21:
            features[72] = _pct_change(spy_close, 20)   # spy_return_20d_pct
        if len(spy_close) >= 50:
            spy_ma50 = _sma(spy_close, 50)
            spy_cur = float(spy_close.iloc[-1])
            if not math.isnan(spy_ma50) and spy_ma50 > 0:
                features[73] = (spy_cur - spy_ma50) / spy_ma50  # spy_dist_ma50_pct

        # Stock-SPY correlation (20-bar rolling)
        if len(df) >= 20 and len(spy_df) >= 20:
            try:
                # Align by index (date), take last 20 common bars
                stock_ret = df["close"].pct_change().iloc[-20:]
                spy_ret = spy_df["close"].pct_change().iloc[-20:]
                # Merge on index for alignment
                merged = pd.DataFrame({
                    "stock": stock_ret, "spy": spy_ret,
                }).dropna()
                if len(merged) >= 10:
                    corr = float(merged["stock"].corr(merged["spy"]))
                    if not math.isnan(corr):
                        features[76] = corr  # stock_spy_corr_20
            except Exception:
                pass

    # VIX features
    if vix_df is not None and len(vix_df) >= 2:
        vix_close = vix_df["close"]
        features[74] = _safe_float(float(vix_close.iloc[-1]))  # vix_level
        if len(vix_close) >= 6:
            features[75] = _pct_change(vix_close, 5)  # vix_change_5d_pct

    return features


def extract_features_dict(
    indicator_results: list[Any],
    pattern_results: list[Any],
    composite: dict[str, Any],
    pattern_composite: dict[str, Any],
    regime: Any | None,
    df: pd.DataFrame,
    *,
    spy_df: pd.DataFrame | None = None,
    vix_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Same as extract_features() but returns a name→value dict.

    Useful for debugging and displaying feature importances.
    """
    arr = extract_features(
        indicator_results, pattern_results,
        composite, pattern_composite,
        regime, df,
        spy_df=spy_df,
        vix_df=vix_df,
    )
    return {name: float(arr[i]) for i, name in enumerate(FEATURE_NAMES)}
