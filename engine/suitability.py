"""
engine/suitability.py — Trading mode suitability detection.

Analyzes a stock's characteristics (volume, trend strength, volatility,
and trend direction) to determine which trading strategy mode is appropriate:

    long_short — liquid, trending, volatile enough for both directions
    long_only  — can go long when bullish, but goes to cash when bearish
    hold_only  — unsuitable for active trading (display analysis only)

Auto-detection uses these criteria (all thresholds configurable in config.yaml):
    hold_only  if avg daily volume < min_volume OR ATR% < min_atr_pct
    long_only  if ADX < min_adx_for_short OR ATR% < min_atr_for_short
               OR avg volume < min_volume_for_short
               OR price is above long-term MA more than max_pct_above_ma
    long_short otherwise

Usage:
    from engine.suitability import SuitabilityAnalyzer, TradingMode

    analyzer = SuitabilityAnalyzer(cfg)
    assessment = analyzer.assess(df)  # df = OHLCV DataFrame
    print(assessment.mode, assessment.reasons)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from config import Config


class TradingMode(Enum):
    """Trading modes ordered from most restrictive to least."""
    HOLD_ONLY = "hold_only"
    LONG_ONLY = "long_only"
    LONG_SHORT = "long_short"


@dataclass
class SuitabilityAssessment:
    """Result of a suitability analysis."""
    mode: TradingMode
    reasons: list[str] = field(default_factory=list)
    # Raw metrics used for the decision (displayed to the user)
    avg_daily_volume: float = 0.0
    adx_value: float = 0.0
    atr_pct: float = 0.0        # ATR as percentage of price
    pct_above_ma: float = 0.0   # % of bars where price > long-term MA
    forced: bool = False         # True if mode was overridden by user


class SuitabilityAnalyzer:
    """Analyzes a stock's characteristics to determine the appropriate trading mode.

    All thresholds are loaded from the ``suitability`` section of config.yaml.
    """

    def __init__(self, cfg: "Config") -> None:
        self._cfg = cfg
        suit = cfg.section("suitability")

        # hold_only thresholds
        self._min_volume: float = float(suit.get("min_volume", 100_000))
        self._min_atr_pct: float = float(suit.get("min_atr_pct", 0.005))

        # long_only thresholds (below these → can't short effectively)
        self._min_adx_for_short: float = float(suit.get("min_adx_for_short", 25.0))
        self._min_atr_for_short: float = float(suit.get("min_atr_for_short", 0.01))
        self._min_volume_for_short: float = float(suit.get("min_volume_for_short", 500_000))

        # Trend direction filter — if price is above its long-term MA
        # for more than this % of the time, shorting is fighting the trend.
        self._trend_ma_period: int = int(suit.get("trend_ma_period", 200))
        self._max_pct_above_ma: float = float(suit.get("max_pct_above_ma", 65.0))

        # ATR calculation period
        self._atr_period: int = int(suit.get("atr_period", 14))

        # Algorithmic constants
        self._adx_min_data_mult: int = int(suit.get("adx_min_data_mult", 3))
        self._insufficient_data_pct: float = float(suit.get("insufficient_data_pct", 50.0))

    def assess(self, df: pd.DataFrame) -> SuitabilityAssessment:
        """Analyze the OHLCV DataFrame and return a suitability assessment.

        Args:
            df: DataFrame with columns: open, high, low, close, volume.
                Should contain enough bars for indicator calculation.

        Returns:
            :class:`SuitabilityAssessment` with the detected mode and reasons.
        """
        # ── Compute raw metrics ─────────────────────────────────────────
        avg_volume = self._compute_avg_volume(df)
        atr_pct = self._compute_atr_pct(df)
        adx_value = self._compute_adx(df)
        pct_above_ma = self._compute_pct_above_ma(df)

        reasons: list[str] = []
        mode = TradingMode.LONG_SHORT  # default: most permissive

        # ── Check hold_only conditions first (most restrictive) ─────────
        hold_only = False

        if avg_volume < self._min_volume:
            reasons.append(
                f"Avg daily volume ({avg_volume:,.0f}) below minimum "
                f"({self._min_volume:,.0f}) — too illiquid for active trading"
            )
            hold_only = True

        if atr_pct < self._min_atr_pct:
            reasons.append(
                f"ATR% ({atr_pct:.3f}) below minimum ({self._min_atr_pct:.3f}) "
                f"— price movement too low for active trading"
            )
            hold_only = True

        if hold_only:
            return SuitabilityAssessment(
                mode=TradingMode.HOLD_ONLY,
                reasons=reasons,
                avg_daily_volume=avg_volume,
                adx_value=adx_value,
                atr_pct=atr_pct,
                pct_above_ma=pct_above_ma,
            )

        # ── Check long_only conditions ──────────────────────────────────
        long_only = False

        if adx_value < self._min_adx_for_short:
            reasons.append(
                f"ADX ({adx_value:.1f}) below threshold ({self._min_adx_for_short:.1f}) "
                f"— trend too weak for effective shorting"
            )
            long_only = True

        if atr_pct < self._min_atr_for_short:
            reasons.append(
                f"ATR% ({atr_pct:.3f}) below short threshold ({self._min_atr_for_short:.3f}) "
                f"— volatility too low for short-term shorts"
            )
            long_only = True

        if avg_volume < self._min_volume_for_short:
            reasons.append(
                f"Avg daily volume ({avg_volume:,.0f}) below short threshold "
                f"({self._min_volume_for_short:,.0f}) — insufficient liquidity for shorting"
            )
            long_only = True

        # Trend direction filter: if the stock spends most of its time above
        # the long-term MA, it has a structural upward bias and shorts will
        # lose more often than they win over the long run.
        if pct_above_ma > self._max_pct_above_ma:
            reasons.append(
                f"Price above {self._trend_ma_period}-day MA for "
                f"{pct_above_ma:.0f}% of period (>{self._max_pct_above_ma:.0f}%) "
                f"— long-term uptrend makes shorting unprofitable"
            )
            long_only = True

        if long_only:
            return SuitabilityAssessment(
                mode=TradingMode.LONG_ONLY,
                reasons=reasons,
                avg_daily_volume=avg_volume,
                adx_value=adx_value,
                atr_pct=atr_pct,
                pct_above_ma=pct_above_ma,
            )

        # ── All clear → long_short ──────────────────────────────────────
        reasons.append("Sufficient volume, trend strength, and volatility for long/short trading")
        return SuitabilityAssessment(
            mode=TradingMode.LONG_SHORT,
            reasons=reasons,
            avg_daily_volume=avg_volume,
            adx_value=adx_value,
            atr_pct=atr_pct,
            pct_above_ma=pct_above_ma,
        )

    # ------------------------------------------------------------------
    # Internal metric computations
    # ------------------------------------------------------------------

    def _compute_avg_volume(self, df: pd.DataFrame) -> float:
        """Average daily volume over the entire dataset."""
        if "volume" not in df.columns or df.empty:
            return 0.0
        return float(df["volume"].mean())

    def _compute_atr_pct(self, df: pd.DataFrame) -> float:
        """Average True Range as a percentage of the current price.

        ATR% = ATR(period) / last_close
        """
        if len(df) < self._atr_period + 1:
            return 0.0

        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range components
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=self._atr_period).mean().iloc[-1]
        last_close = float(close.iloc[-1])

        if last_close <= 0:
            return 0.0
        return float(atr) / last_close

    def _compute_adx(self, df: pd.DataFrame) -> float:
        """Compute the latest ADX value.

        Delegates to :func:`ta_utils.compute_adx` — the single canonical
        Wilder's-smoothing ADX implementation used by both the regime
        classifier and suitability analyzer.
        """
        from engine.ta_utils import compute_adx

        if len(df) < self._atr_period * self._adx_min_data_mult:
            return 0.0
        return compute_adx(df, self._atr_period)

    def _compute_pct_above_ma(self, df: pd.DataFrame) -> float:
        """Percentage of bars where close > long-term SMA.

        Computes a rolling SMA of ``self._trend_ma_period`` on the close price,
        drops the initial NaN rows (first *period* bars won't have a valid MA),
        and returns the fraction of remaining bars where close > SMA, as a
        percentage (0-100).

        Returns 50.0 if there is insufficient data (fewer bars than the MA
        period), which is treated as "no strong directional bias".
        """
        if len(df) < self._trend_ma_period:
            return self._insufficient_data_pct

        close = df["close"]
        sma = close.rolling(window=self._trend_ma_period).mean()

        # Drop NaN rows — first (period-1) bars have no valid MA
        valid = sma.dropna()
        if valid.empty:
            return self._insufficient_data_pct

        close_valid = close.loc[valid.index]
        above = (close_valid > valid).sum()
        return float(above) / len(valid) * 100.0
