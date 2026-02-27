"""Technical-analysis utility functions shared across engine modules."""

from __future__ import annotations

import pandas as pd


def _wilder_smooth(values: list[float], n: int) -> list[float]:
    """Wilder's exponential smoothing."""
    if len(values) < n:
        return []
    first = sum(values[:n]) / n
    smoothed = [first]
    for v in values[n:]:
        smoothed.append(smoothed[-1] * (1 - 1 / n) + v * (1 / n))
    return smoothed


def _directional_components(
    df: pd.DataFrame,
) -> tuple[list[float], list[float], list[float]]:
    """Return (+DM, -DM, TR) lists from OHLC data."""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    plus_dm: list[float] = []
    minus_dm: list[float] = []
    tr_list: list[float] = []

    for i in range(1, len(df)):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(
            down_move if down_move > up_move and down_move > 0 else 0.0
        )

        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr_list.append(max(tr1, tr2, tr3))

    return plus_dm, minus_dm, tr_list


def compute_adx(df: pd.DataFrame, period: int) -> float:
    """Compute the latest ADX value using Wilder's smoothing."""
    if len(df) < period * 3:
        return 0.0

    plus_dm, minus_dm, tr_list = _directional_components(df)

    if len(tr_list) < period:
        return 0.0

    sm_plus_dm = _wilder_smooth(plus_dm, period)
    sm_minus_dm = _wilder_smooth(minus_dm, period)
    sm_tr = _wilder_smooth(tr_list, period)

    if not sm_tr or not sm_plus_dm or not sm_minus_dm:
        return 0.0

    dx_values: list[float] = []
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

    adx_smoothed = _wilder_smooth(dx_values, period)
    return adx_smoothed[-1] if adx_smoothed else 0.0


def compute_adx_mean(df: pd.DataFrame, period: int) -> float:
    """Compute the mean ADX value over the full period.

    Instead of relying solely on the last ADX value (which can be low
    during short-term consolidations even within strong trends), this
    computes the full ADX series and returns the mean.
    """
    if len(df) < period * 3:
        return compute_adx(df, period)

    plus_dm, minus_dm, tr_list = _directional_components(df)

    if len(tr_list) < period:
        return 0.0

    sm_plus_dm = _wilder_smooth(plus_dm, period)
    sm_minus_dm = _wilder_smooth(minus_dm, period)
    sm_tr = _wilder_smooth(tr_list, period)

    if not sm_tr or not sm_plus_dm or not sm_minus_dm:
        return 0.0

    dx_values: list[float] = []
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

    adx_smoothed = _wilder_smooth(dx_values, period)
    if not adx_smoothed:
        return 0.0

    return sum(adx_smoothed) / len(adx_smoothed)
