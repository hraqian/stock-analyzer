"""
engine/regime.py — Market Regime classification system.

Characterizes the current market environment into one of four regimes
and provides a confidence score for each classification.  The regime
label drives strategy adaptation in the backtest engine — e.g. a
strong-trend regime uses trailing stops instead of score-based entries.

Regimes:
    STRONG_TREND       — High ADX, price consistently above MA, low pullbacks.
                         Optimal: buy-and-hold or wide trailing stop.
    MEAN_REVERTING     — Low ADX, price oscillates around MA, range-bound.
                         Optimal: swing-trade support/resistance levels.
    VOLATILE_CHOPPY    — High ATR%, frequent direction changes, no clear trend.
                         Optimal: reduce size, widen stops, or stay out.
    BREAKOUT_TRANSITION — Narrowing volatility (BB squeeze) then expansion.
                         Optimal: momentum entry on breakout confirmation.

All classification thresholds are configurable in config.yaml → ``regime``.

Usage:
    from engine.regime import RegimeClassifier, RegimeType

    classifier = RegimeClassifier(cfg)
    assessment = classifier.classify(df)
    print(assessment.regime, assessment.confidence, assessment.reasons)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from config import Config


class RegimeType(Enum):
    """Market regime categories."""
    STRONG_TREND = "strong_trend"
    MEAN_REVERTING = "mean_reverting"
    VOLATILE_CHOPPY = "volatile_choppy"
    BREAKOUT_TRANSITION = "breakout_transition"


# Human-readable display labels
REGIME_LABELS: dict[RegimeType, str] = {
    RegimeType.STRONG_TREND: "Strong Trend",
    RegimeType.MEAN_REVERTING: "Mean-Reverting / Range-Bound",
    RegimeType.VOLATILE_CHOPPY: "Volatile / Choppy",
    RegimeType.BREAKOUT_TRANSITION: "Breakout / Transition",
}

REGIME_DESCRIPTIONS: dict[RegimeType, str] = {
    RegimeType.STRONG_TREND: (
        "Price trending consistently in one direction. "
        "Best traded with trend-following or wide trailing stop."
    ),
    RegimeType.MEAN_REVERTING: (
        "Price oscillating around a mean — range-bound. "
        "Best traded with swing entries at support/resistance."
    ),
    RegimeType.VOLATILE_CHOPPY: (
        "High volatility with frequent direction changes. "
        "Reduce position size, widen stops, or avoid active trading."
    ),
    RegimeType.BREAKOUT_TRANSITION: (
        "Volatility contracting then expanding — potential regime change. "
        "Trade momentum breakouts with confirmation."
    ),
}


@dataclass
class RegimeMetrics:
    """Raw metrics computed during regime classification."""
    adx: float = 0.0                   # ADX value (0-100) — current/latest
    rolling_adx_mean: float = 0.0      # Mean ADX over the full period
    total_return: float = 0.0          # Total price return over the period (e.g. 0.80 = +80%)
    pct_above_ma: float = 50.0         # % of bars where close > trend MA
    atr_pct: float = 0.0               # ATR as % of price
    bb_width: float = 0.0              # Bollinger Band width (normalised)
    bb_width_percentile: float = 50.0  # BB width percentile vs recent history
    price_ma_distance: float = 0.0     # price distance from MA as % of price
    direction_changes: float = 0.0     # fraction of bars that reverse direction
    trend_direction: str = "neutral"   # "bullish", "bearish", or "neutral"


@dataclass
class RegimeAssessment:
    """Result of regime classification."""
    regime: RegimeType
    confidence: float                          # 0.0 – 1.0
    label: str = ""                            # human-readable label
    description: str = ""                      # brief guidance text
    metrics: RegimeMetrics = field(default_factory=RegimeMetrics)
    reasons: list[str] = field(default_factory=list)
    # Per-regime scores (0-1) — useful for seeing how close to other regimes
    regime_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label:
            self.label = REGIME_LABELS.get(self.regime, self.regime.value)
        if not self.description:
            self.description = REGIME_DESCRIPTIONS.get(self.regime, "")


class RegimeClassifier:
    """Classifies market data into one of four regimes.

    Uses a scoring approach: each regime gets a score based on how well
    the current metrics match its profile.  The regime with the highest
    score wins.  Confidence = winner_score / sum(all_scores).

    All thresholds are loaded from ``config.yaml → regime``.
    """

    def __init__(self, cfg: "Config") -> None:
        self._cfg = cfg
        r = cfg.section("regime")

        # ── MA period for trend analysis ────────────────────────────────
        self._trend_ma_period: int = int(r.get("trend_ma_period", 50))

        # ── ADX thresholds ──────────────────────────────────────────────
        self._adx_strong_trend: float = float(r.get("adx_strong_trend", 30.0))
        self._adx_weak: float = float(r.get("adx_weak", 20.0))

        # ── Trend consistency (% bars above MA) ─────────────────────────
        self._trend_consistency_high: float = float(r.get("trend_consistency_high", 70.0))
        self._trend_consistency_low: float = float(r.get("trend_consistency_low", 40.0))

        # ── ATR% thresholds ─────────────────────────────────────────────
        self._atr_pct_high: float = float(r.get("atr_pct_high", 0.03))
        self._atr_pct_low: float = float(r.get("atr_pct_low", 0.01))
        self._atr_period: int = int(r.get("atr_period", 14))

        # ── Bollinger squeeze detection ─────────────────────────────────
        self._bb_period: int = int(r.get("bb_period", 20))
        self._bb_std_dev: float = float(r.get("bb_std_dev", 2.0))
        self._bb_squeeze_percentile: float = float(r.get("bb_squeeze_percentile", 20.0))
        self._bb_expansion_percentile: float = float(r.get("bb_expansion_percentile", 80.0))

        # ── Direction change threshold ──────────────────────────────────
        self._direction_change_high: float = float(r.get("direction_change_high", 0.55))
        self._direction_change_period: int = int(r.get("direction_change_period", 20))

        # ── Price-MA distance ───────────────────────────────────────────
        self._price_ma_distance_extended: float = float(r.get("price_ma_distance_extended", 0.10))

        # ── Total return thresholds (primary trend signal) ──────────────
        self._total_return_strong: float = float(r.get("total_return_strong", 0.30))
        self._total_return_moderate: float = float(r.get("total_return_moderate", 0.15))

        # ── Classification guard ────────────────────────────────────────
        self._min_bars_for_classification: int = int(r.get("min_bars_for_classification", 20))

        # ── Trend direction thresholds ──────────────────────────────────
        self._trend_direction_bullish: float = float(r.get("trend_direction_bullish_threshold", 60))
        self._trend_direction_bearish: float = float(r.get("trend_direction_bearish_threshold", 40))

        # ── Reason building ─────────────────────────────────────────────
        self._adx_dip_threshold: float = float(r.get("adx_dip_threshold", 3))
        self._runner_up_proximity_ratio: float = float(r.get("runner_up_proximity_ratio", 0.7))

        # ── Scoring weights (nested dict) ───────────────────────────────
        self._scoring: dict[str, dict[str, float]] = r.get("scoring", {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, df: pd.DataFrame) -> RegimeAssessment:
        """Classify the regime of the OHLCV DataFrame.

        Args:
            df: DataFrame with columns: open, high, low, close, volume.
                Should contain enough history for indicator calculations.

        Returns:
            :class:`RegimeAssessment` with the detected regime, confidence,
            metrics, and reasoning.
        """
        metrics = self._compute_metrics(df)
        scores = self._score_regimes(metrics)
        reasons: list[str] = []

        # Pick the winner
        best_regime = max(scores, key=scores.get)  # type: ignore[arg-type]
        total = sum(scores.values())
        confidence = scores[best_regime] / total if total > 0 else 0.0

        # Build human-readable reasons
        reasons = self._build_reasons(metrics, scores)

        return RegimeAssessment(
            regime=best_regime,
            confidence=confidence,
            metrics=metrics,
            reasons=reasons,
            regime_scores={rt.value: round(s, 3) for rt, s in scores.items()},
        )

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def _compute_metrics(self, df: pd.DataFrame) -> RegimeMetrics:
        """Compute all raw metrics needed for regime classification."""
        metrics = RegimeMetrics()

        if df.empty or len(df) < self._min_bars_for_classification:
            return metrics

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # ── Total return over the period ────────────────────────────────
        first_close = float(close.iloc[0])
        last_close = float(close.iloc[-1])
        if first_close > 0:
            metrics.total_return = (last_close - first_close) / first_close

        # ── ADX (current) ───────────────────────────────────────────────
        metrics.adx = self._compute_adx(df)

        # ── Rolling ADX mean (over the full period) ─────────────────────
        metrics.rolling_adx_mean = self._compute_rolling_adx_mean(df)

        # ── Trend MA and % above ────────────────────────────────────────
        period = min(self._trend_ma_period, len(df) - 1)
        if period > 0:
            sma = close.rolling(window=period).mean()
            valid = sma.dropna()
            if not valid.empty:
                close_valid = close.loc[valid.index]
                above = (close_valid > valid).sum()
                metrics.pct_above_ma = float(above) / len(valid) * 100.0

                # Trend direction
                if metrics.pct_above_ma > self._trend_direction_bullish:
                    metrics.trend_direction = "bullish"
                elif metrics.pct_above_ma < self._trend_direction_bearish:
                    metrics.trend_direction = "bearish"
                else:
                    metrics.trend_direction = "neutral"

                # Price distance from MA (% of price)
                last_close = float(close.iloc[-1])
                last_ma = float(sma.iloc[-1])
                if last_close > 0:
                    metrics.price_ma_distance = abs(last_close - last_ma) / last_close

        # ── ATR% ────────────────────────────────────────────────────────
        metrics.atr_pct = self._compute_atr_pct(df)

        # ── Bollinger Band width ────────────────────────────────────────
        bb_period = min(self._bb_period, len(df) - 1)
        if bb_period >= 5:
            sma_bb = close.rolling(window=bb_period).mean()
            std_bb = close.rolling(window=bb_period).std()
            upper = sma_bb + self._bb_std_dev * std_bb
            lower = sma_bb - self._bb_std_dev * std_bb
            bb_width = ((upper - lower) / sma_bb).dropna()
            if not bb_width.empty:
                metrics.bb_width = float(bb_width.iloc[-1])
                # Percentile of current width vs full history
                current = float(bb_width.iloc[-1])
                count_below = (bb_width < current).sum()
                metrics.bb_width_percentile = float(count_below) / len(bb_width) * 100.0

        # ── Direction changes ───────────────────────────────────────────
        lookback = min(self._direction_change_period, len(df) - 2)
        if lookback > 0:
            recent = close.iloc[-lookback - 1:]
            diffs = recent.diff().dropna()
            if len(diffs) > 1:
                signs = diffs.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
                sign_changes = (signs.diff().dropna() != 0).sum()
                metrics.direction_changes = float(sign_changes) / len(diffs)

        return metrics

    def _compute_adx(self, df: pd.DataFrame) -> float:
        """Compute latest ADX value using Wilder's smoothing."""
        period = self._atr_period

        if len(df) < period * 3:
            return 0.0

        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        plus_dm = []
        minus_dm = []
        tr_list = []

        for i in range(1, len(df)):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]

            plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
            minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)

            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i - 1])
            tr3 = abs(low[i] - close[i - 1])
            tr_list.append(max(tr1, tr2, tr3))

        if len(tr_list) < period:
            return 0.0

        def wilder_smooth(values: list[float], n: int) -> list[float]:
            if len(values) < n:
                return []
            first = sum(values[:n]) / n
            smoothed = [first]
            for v in values[n:]:
                smoothed.append(smoothed[-1] * (1 - 1 / n) + v * (1 / n))
            return smoothed

        sm_plus_dm = wilder_smooth(plus_dm, period)
        sm_minus_dm = wilder_smooth(minus_dm, period)
        sm_tr = wilder_smooth(tr_list, period)

        if not sm_tr or not sm_plus_dm or not sm_minus_dm:
            return 0.0

        dx_values = []
        length = min(len(sm_plus_dm), len(sm_minus_dm), len(sm_tr))
        for i in range(length):
            if sm_tr[i] == 0:
                dx_values.append(0.0)
                continue
            plus_di = 100 * sm_plus_dm[i] / sm_tr[i]
            minus_di = 100 * sm_minus_dm[i] / sm_tr[i]
            di_sum = plus_di + minus_di
            if di_sum == 0:
                dx_values.append(0.0)
            else:
                dx_values.append(100 * abs(plus_di - minus_di) / di_sum)

        if len(dx_values) < period:
            return sum(dx_values) / len(dx_values) if dx_values else 0.0

        adx_smoothed = wilder_smooth(dx_values, period)
        return adx_smoothed[-1] if adx_smoothed else 0.0

    def _compute_atr_pct(self, df: pd.DataFrame) -> float:
        """ATR as a percentage of the current price."""
        period = self._atr_period

        if len(df) < period + 1:
            return 0.0

        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean().iloc[-1]
        last_close = float(close.iloc[-1])

        if last_close <= 0:
            return 0.0
        return float(atr) / last_close

    def _compute_rolling_adx_mean(self, df: pd.DataFrame) -> float:
        """Compute the mean ADX value over the full period.

        Instead of relying solely on the last ADX value (which can be low
        during short-term consolidations even within strong trends), this
        computes the full ADX series and returns the mean.
        """
        period = self._atr_period

        if len(df) < period * 3:
            return self._compute_adx(df)

        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        plus_dm = []
        minus_dm = []
        tr_list = []

        for i in range(1, len(df)):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]

            plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
            minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)

            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i - 1])
            tr3 = abs(low[i] - close[i - 1])
            tr_list.append(max(tr1, tr2, tr3))

        if len(tr_list) < period:
            return 0.0

        def wilder_smooth(values: list[float], n: int) -> list[float]:
            if len(values) < n:
                return []
            first = sum(values[:n]) / n
            smoothed = [first]
            for v in values[n:]:
                smoothed.append(smoothed[-1] * (1 - 1 / n) + v * (1 / n))
            return smoothed

        sm_plus_dm = wilder_smooth(plus_dm, period)
        sm_minus_dm = wilder_smooth(minus_dm, period)
        sm_tr = wilder_smooth(tr_list, period)

        if not sm_tr or not sm_plus_dm or not sm_minus_dm:
            return 0.0

        dx_values = []
        length = min(len(sm_plus_dm), len(sm_minus_dm), len(sm_tr))
        for i in range(length):
            if sm_tr[i] == 0:
                dx_values.append(0.0)
                continue
            plus_di = 100 * sm_plus_dm[i] / sm_tr[i]
            minus_di = 100 * sm_minus_dm[i] / sm_tr[i]
            di_sum = plus_di + minus_di
            if di_sum == 0:
                dx_values.append(0.0)
            else:
                dx_values.append(100 * abs(plus_di - minus_di) / di_sum)

        if len(dx_values) < period:
            return sum(dx_values) / len(dx_values) if dx_values else 0.0

        adx_smoothed = wilder_smooth(dx_values, period)
        if not adx_smoothed:
            return 0.0

        return sum(adx_smoothed) / len(adx_smoothed)

    # ------------------------------------------------------------------
    # Regime scoring
    # ------------------------------------------------------------------

    def _score_regimes(self, m: RegimeMetrics) -> dict[RegimeType, float]:
        """Score each regime based on how well metrics match its profile.

        Each regime gets a base score of 0.  Matching criteria add points.
        The regime with the highest total wins.

        Total return over the analysis period is the primary trend signal —
        a stock up 30%+ is in a trend regardless of current ADX.  Rolling
        ADX mean (average over the period) supplements current ADX, since
        the current value can dip during short-term consolidations.

        All scoring weights are read from config → regime → scoring.
        """
        scores: dict[RegimeType, float] = {
            RegimeType.STRONG_TREND: 0.0,
            RegimeType.MEAN_REVERTING: 0.0,
            RegimeType.VOLATILE_CHOPPY: 0.0,
            RegimeType.BREAKOUT_TRANSITION: 0.0,
        }

        # Shorthand accessors for each regime's scoring dict
        st = self._scoring.get("strong_trend", {})
        mr = self._scoring.get("mean_reverting", {})
        vc = self._scoring.get("volatile_choppy", {})
        bo = self._scoring.get("breakout", {})

        # Use the better of current ADX and rolling ADX mean — a stock
        # in a strong long-term trend that's temporarily consolidating
        # should still register its trend character.
        effective_adx = max(m.adx, m.rolling_adx_mean)
        abs_return = abs(m.total_return)

        # ════════════════════════════════════════════════════════════════
        # STRONG TREND scoring
        # ════════════════════════════════════════════════════════════════

        # --- Primary signal: total return over the period ---
        # A stock up 30%+ (or down 30%+) is definitively trending.
        if abs_return >= self._total_return_strong:
            return_strong_base = float(st.get("return_strong_base", 3.0))
            return_strong_cap = float(st.get("return_strong_cap", 0.70))
            return_strong_scale = float(st.get("return_strong_scale", 3.0))
            excess = min(abs_return - self._total_return_strong, return_strong_cap)
            scores[RegimeType.STRONG_TREND] += return_strong_base + (excess / return_strong_cap * return_strong_scale if return_strong_cap > 0 else 0)
        elif abs_return >= self._total_return_moderate:
            return_moderate_base = float(st.get("return_moderate_base", 1.0))
            return_moderate_scale = float(st.get("return_moderate_scale", 2.0))
            frac = (abs_return - self._total_return_moderate) / (self._total_return_strong - self._total_return_moderate)
            scores[RegimeType.STRONG_TREND] += return_moderate_base + frac * return_moderate_scale

        # --- ADX (use effective = max of current and rolling mean) ---
        adx_strong_base = float(st.get("adx_strong_base", 2.0))
        adx_strong_divisor = float(st.get("adx_strong_divisor", 20.0))
        adx_moderate_score = float(st.get("adx_moderate_score", 0.5))
        if effective_adx >= self._adx_strong_trend:
            scores[RegimeType.STRONG_TREND] += adx_strong_base + (effective_adx - self._adx_strong_trend) / adx_strong_divisor
        elif effective_adx >= self._adx_weak:
            scores[RegimeType.STRONG_TREND] += adx_moderate_score

        # Trend consistency: price consistently on one side of MA
        pct_away = abs(m.pct_above_ma - 50.0)  # 0 = perfectly balanced, 50 = always one side
        consistency_high_score = float(st.get("consistency_high_score", 2.0))
        consistency_moderate_score = float(st.get("consistency_moderate_score", 0.5))
        if pct_away > (self._trend_consistency_high - 50.0):
            scores[RegimeType.STRONG_TREND] += consistency_high_score
        elif pct_away > (self._trend_consistency_low - 50.0):
            scores[RegimeType.STRONG_TREND] += consistency_moderate_score

        # Extended price distance from MA (trending far away)
        extended_distance_score = float(st.get("extended_distance_score", 1.0))
        if m.price_ma_distance > self._price_ma_distance_extended:
            scores[RegimeType.STRONG_TREND] += extended_distance_score

        # Low direction changes favour trend
        dc_low = float(st.get("direction_change_low", 0.4))
        dc_low_score = float(st.get("direction_change_low_score", 1.0))
        dc_mid = float(st.get("direction_change_mid", 0.5))
        dc_mid_score = float(st.get("direction_change_mid_score", 0.3))
        if m.direction_changes < dc_low:
            scores[RegimeType.STRONG_TREND] += dc_low_score
        elif m.direction_changes < dc_mid:
            scores[RegimeType.STRONG_TREND] += dc_mid_score

        # ════════════════════════════════════════════════════════════════
        # MEAN REVERTING scoring
        # ════════════════════════════════════════════════════════════════

        # --- Total return suppresses mean-reverting ---
        return_strong_penalty = float(mr.get("return_strong_penalty", 2.0))
        return_moderate_penalty = float(mr.get("return_moderate_penalty", 1.0))
        return_small_bonus = float(mr.get("return_small_bonus", 1.5))
        if abs_return >= self._total_return_strong:
            scores[RegimeType.MEAN_REVERTING] -= return_strong_penalty
        elif abs_return >= self._total_return_moderate:
            scores[RegimeType.MEAN_REVERTING] -= return_moderate_penalty
        else:
            scores[RegimeType.MEAN_REVERTING] += return_small_bonus

        # Low ADX → range-bound
        mr_adx_low_score = float(mr.get("adx_low_score", 2.0))
        mr_adx_moderate_score = float(mr.get("adx_moderate_score", 1.0))
        if effective_adx < self._adx_weak:
            scores[RegimeType.MEAN_REVERTING] += mr_adx_low_score
        elif effective_adx < self._adx_strong_trend:
            scores[RegimeType.MEAN_REVERTING] += mr_adx_moderate_score

        # Price oscillates around MA (pct_above_ma near 50%)
        pct_away_tight = float(mr.get("pct_away_tight", 15))
        pct_away_tight_score = float(mr.get("pct_away_tight_score", 2.0))
        pct_away_moderate = float(mr.get("pct_away_moderate", 25))
        pct_away_moderate_score = float(mr.get("pct_away_moderate_score", 1.0))
        if pct_away < pct_away_tight:
            scores[RegimeType.MEAN_REVERTING] += pct_away_tight_score
        elif pct_away < pct_away_moderate:
            scores[RegimeType.MEAN_REVERTING] += pct_away_moderate_score

        # Low/moderate volatility
        mr_atr_below_high_score = float(mr.get("atr_below_high_score", 0.5))
        mr_atr_below_low_score = float(mr.get("atr_below_low_score", 0.5))
        if m.atr_pct < self._atr_pct_high:
            scores[RegimeType.MEAN_REVERTING] += mr_atr_below_high_score
        if m.atr_pct < self._atr_pct_low:
            scores[RegimeType.MEAN_REVERTING] += mr_atr_below_low_score

        # Price stays close to MA
        mr_price_ma_threshold = float(mr.get("price_ma_close_threshold", 0.03))
        mr_price_ma_score = float(mr.get("price_ma_close_score", 1.0))
        if m.price_ma_distance < mr_price_ma_threshold:
            scores[RegimeType.MEAN_REVERTING] += mr_price_ma_score

        # ════════════════════════════════════════════════════════════════
        # VOLATILE CHOPPY scoring
        # ════════════════════════════════════════════════════════════════
        # High ATR% → volatile
        vc_atr_high_base = float(vc.get("atr_high_base", 2.0))
        vc_atr_high_scale = float(vc.get("atr_high_scale", 30.0))
        vc_atr_moderate_score = float(vc.get("atr_moderate_score", 0.5))
        if m.atr_pct >= self._atr_pct_high:
            scores[RegimeType.VOLATILE_CHOPPY] += vc_atr_high_base + (m.atr_pct - self._atr_pct_high) * vc_atr_high_scale
        elif m.atr_pct >= self._atr_pct_low:
            scores[RegimeType.VOLATILE_CHOPPY] += vc_atr_moderate_score

        # Frequent direction changes → choppy
        vc_dc_high_score = float(vc.get("direction_change_high_score", 2.0))
        vc_dc_moderate = float(vc.get("direction_change_moderate", 0.45))
        vc_dc_moderate_score = float(vc.get("direction_change_moderate_score", 1.0))
        if m.direction_changes >= self._direction_change_high:
            scores[RegimeType.VOLATILE_CHOPPY] += vc_dc_high_score
        elif m.direction_changes >= vc_dc_moderate:
            scores[RegimeType.VOLATILE_CHOPPY] += vc_dc_moderate_score

        # Low ADX can also mean choppy (no trend) — but only if returns
        # are also small (otherwise it's a trend with volatile swings)
        vc_low_adx_small_return = float(vc.get("low_adx_small_return_score", 0.5))
        if effective_adx < self._adx_weak and abs_return < self._total_return_moderate:
            scores[RegimeType.VOLATILE_CHOPPY] += vc_low_adx_small_return

        # Wide Bollinger Bands → high volatility
        vc_wide_bb_score = float(vc.get("wide_bb_score", 1.0))
        if m.bb_width_percentile > self._bb_expansion_percentile:
            scores[RegimeType.VOLATILE_CHOPPY] += vc_wide_bb_score

        # ════════════════════════════════════════════════════════════════
        # BREAKOUT TRANSITION scoring
        # ════════════════════════════════════════════════════════════════
        # BB squeeze (low width percentile) → potential breakout building
        bo_bb_squeeze_score = float(bo.get("bb_squeeze_score", 2.5))
        if m.bb_width_percentile <= self._bb_squeeze_percentile:
            scores[RegimeType.BREAKOUT_TRANSITION] += bo_bb_squeeze_score

        # Moderate ADX (not too high, not too low) — transition zone
        bo_adx_moderate_score = float(bo.get("adx_moderate_score", 1.0))
        if self._adx_weak <= m.adx <= self._adx_strong_trend:
            scores[RegimeType.BREAKOUT_TRANSITION] += bo_adx_moderate_score

        # Low current volatility but non-negligible direction changes
        # suggest consolidation before a move
        bo_dc_threshold = float(bo.get("direction_change_threshold", 0.40))
        bo_low_atr_high_changes = float(bo.get("low_atr_high_changes_score", 1.0))
        if m.atr_pct < self._atr_pct_low and m.direction_changes > bo_dc_threshold:
            scores[RegimeType.BREAKOUT_TRANSITION] += bo_low_atr_high_changes

        # Price close to MA during consolidation
        bo_price_ma_threshold = float(bo.get("price_ma_close_threshold", 0.03))
        bo_bb_consol_pctl = float(bo.get("bb_consolidation_percentile", 40))
        bo_consolidation_score = float(bo.get("consolidation_score", 0.5))
        if m.price_ma_distance < bo_price_ma_threshold and m.bb_width_percentile < bo_bb_consol_pctl:
            scores[RegimeType.BREAKOUT_TRANSITION] += bo_consolidation_score

        # Floor all scores at 0 (penalties can push negative)
        for rt in scores:
            scores[rt] = max(scores[rt], 0.0)

        return scores

    # ------------------------------------------------------------------
    # Reason building
    # ------------------------------------------------------------------

    def _build_reasons(
        self,
        m: RegimeMetrics,
        scores: dict[RegimeType, float],
    ) -> list[str]:
        """Build human-readable reasons explaining the classification."""
        reasons: list[str] = []

        # Sorted regimes by score
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        winner = ranked[0][0]
        runner_up = ranked[1] if len(ranked) > 1 else None

        # Total return — the primary trend signal
        ret_pct = m.total_return * 100
        direction = "up" if m.total_return >= 0 else "down"
        abs_ret = abs(m.total_return)
        if abs_ret >= self._total_return_strong:
            reasons.append(f"Total return: {ret_pct:+.1f}% ({direction} — strong trend signal)")
        elif abs_ret >= self._total_return_moderate:
            reasons.append(f"Total return: {ret_pct:+.1f}% ({direction} — moderate trend signal)")
        else:
            reasons.append(f"Total return: {ret_pct:+.1f}% (small move — range-bound signal)")

        # ADX context — show both current and rolling mean
        effective_adx = max(m.adx, m.rolling_adx_mean)
        adx_note = ""
        if m.rolling_adx_mean > m.adx + self._adx_dip_threshold:
            adx_note = f", rolling mean {m.rolling_adx_mean:.1f} (higher — current dip likely temporary)"
        if effective_adx >= self._adx_strong_trend:
            reasons.append(f"ADX = {m.adx:.1f} (strong trend){adx_note}")
        elif effective_adx >= self._adx_weak:
            reasons.append(f"ADX = {m.adx:.1f} (moderate trend){adx_note}")
        else:
            reasons.append(f"ADX = {m.adx:.1f} (weak/no trend){adx_note}")

        # Trend consistency
        if m.pct_above_ma > self._trend_consistency_high:
            reasons.append(
                f"Price above {self._trend_ma_period}-MA for {m.pct_above_ma:.0f}% "
                f"of bars (strong {m.trend_direction} bias)"
            )
        elif m.pct_above_ma < (100 - self._trend_consistency_high):
            reasons.append(
                f"Price below {self._trend_ma_period}-MA for {100 - m.pct_above_ma:.0f}% "
                f"of bars (strong {m.trend_direction} bias)"
            )
        else:
            reasons.append(
                f"Price above {self._trend_ma_period}-MA for {m.pct_above_ma:.0f}% of bars "
                f"(balanced — range-bound signal)"
            )

        # Volatility
        if m.atr_pct >= self._atr_pct_high:
            reasons.append(f"ATR% = {m.atr_pct:.3f} (high volatility)")
        elif m.atr_pct >= self._atr_pct_low:
            reasons.append(f"ATR% = {m.atr_pct:.3f} (moderate volatility)")
        else:
            reasons.append(f"ATR% = {m.atr_pct:.3f} (low volatility)")

        # BB squeeze / expansion
        if m.bb_width_percentile <= self._bb_squeeze_percentile:
            reasons.append(
                f"Bollinger Band width at {m.bb_width_percentile:.0f}th percentile "
                f"(squeeze — breakout setup)"
            )
        elif m.bb_width_percentile >= self._bb_expansion_percentile:
            reasons.append(
                f"Bollinger Band width at {m.bb_width_percentile:.0f}th percentile "
                f"(wide — high volatility)"
            )

        # Direction changes
        if m.direction_changes >= self._direction_change_high:
            reasons.append(
                f"Direction changes: {m.direction_changes:.0%} of bars "
                f"(choppy / whipsaw)"
            )

        # Runner-up context
        if runner_up:
            runner_regime, runner_score = runner_up
            winner_score = scores[winner]
            if winner_score > 0 and runner_score / winner_score > self._runner_up_proximity_ratio:
                reasons.append(
                    f"Close to {REGIME_LABELS[runner_regime]} "
                    f"(score {runner_score:.1f} vs {winner_score:.1f})"
                )

        return reasons
