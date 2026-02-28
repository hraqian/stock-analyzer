"""analysis/dividend.py — Dividend metrics computation.

Fetches dividend history and info from yfinance and computes per-ticker
metrics used by the dividend scanner:

- **Current yield** — trailing 12-month dividend / current price.
- **Dividend growth rate** — CAGR of annual dividends over *n* years.
- **Payout consistency** — fraction of years with a non-zero dividend
  (over the evaluation window).
- **Streak length** — consecutive years of year-over-year dividend increases.

All thresholds and weights live in config (Step 3) — this module is pure
computation with no config dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DividendMetrics:
    """Computed dividend metrics for a single ticker."""

    ticker: str
    current_price: float
    currency: str

    # Raw dividend info
    annual_dividend: float          # trailing 12-month dividend per share
    current_yield: float            # annual_dividend / current_price (0–1 scale, e.g. 0.035 = 3.5%)

    # Growth
    dividend_cagr: float | None     # compound annual growth rate (None if insufficient data)
    cagr_years: int                 # number of years used for CAGR calculation

    # Consistency
    payout_consistency: float       # fraction of years with a dividend (0.0–1.0)
    consistency_years: int          # evaluation window in years

    # Streak
    increase_streak: int            # consecutive years of YoY increases (0 = no streak)

    # Metadata
    name: str = ""
    sector: str = ""
    ex_dividend_date: str = ""
    payout_ratio: float | None = None   # earnings payout ratio (from yfinance .info)
    five_year_avg_yield: float | None = None

    error: str = ""                 # non-empty if fetch/computation failed


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _annualise_dividends(dividends: pd.Series) -> dict[int, float]:
    """Group a dividend Series (datetime-indexed) by calendar year.

    Returns ``{year: total_dividends}`` for each year that had at least
    one payment.  Partial current year is excluded.
    """
    if dividends.empty:
        return {}

    # Ensure timezone-naive for grouping
    idx = dividends.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_localize(None)
    divs = dividends.copy()
    divs.index = idx

    current_year = datetime.now().year
    by_year: dict[int, float] = {}

    for dt, amount in divs.items():
        yr = dt.year  # type: ignore[union-attr]
        if yr >= current_year:
            continue  # skip partial current year
        by_year[yr] = by_year.get(yr, 0.0) + float(amount)

    return by_year


def _compute_cagr(annual_divs: dict[int, float], years: int) -> tuple[float | None, int]:
    """Compute CAGR of annual dividends over the most recent *years*.

    Returns ``(cagr, actual_years_used)``.  CAGR is ``None`` when there
    aren't at least 2 full years with dividends.
    """
    if len(annual_divs) < 2:
        return None, 0

    sorted_years = sorted(annual_divs.keys())

    # Take the most recent `years` entries
    window = sorted_years[-years:] if len(sorted_years) >= years else sorted_years
    if len(window) < 2:
        return None, 0

    start_val = annual_divs[window[0]]
    end_val = annual_divs[window[-1]]
    n = window[-1] - window[0]  # number of years between start and end

    if n <= 0 or start_val <= 0 or end_val <= 0:
        return None, 0

    cagr = (end_val / start_val) ** (1.0 / n) - 1.0
    return cagr, n


def _compute_consistency(annual_divs: dict[int, float], years: int) -> tuple[float, int]:
    """Fraction of the last *years* calendar years that had a dividend.

    Returns ``(consistency_ratio, window_years)``.
    """
    if not annual_divs:
        return 0.0, 0

    current_year = datetime.now().year
    eval_years = list(range(current_year - years, current_year))

    paid_count = sum(1 for yr in eval_years if annual_divs.get(yr, 0) > 0)
    return paid_count / len(eval_years), len(eval_years)


def _compute_streak(annual_divs: dict[int, float]) -> int:
    """Consecutive years of YoY dividend increases ending at the most recent
    full year.  Returns 0 if the most recent year had a decrease or no data.
    """
    if len(annual_divs) < 2:
        return 0

    sorted_years = sorted(annual_divs.keys())
    streak = 0

    # Walk backwards from most recent full year
    for i in range(len(sorted_years) - 1, 0, -1):
        yr = sorted_years[i]
        prev_yr = sorted_years[i - 1]
        if prev_yr != yr - 1:
            break  # gap in data
        if annual_divs[yr] > annual_divs[prev_yr]:
            streak += 1
        else:
            break

    return streak


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def fetch_dividend_metrics(
    ticker: str,
    *,
    cagr_years: int = 5,
    consistency_years: int = 10,
) -> DividendMetrics:
    """Fetch and compute dividend metrics for a single ticker.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g. ``"AAPL"``, ``"RY.TO"``).
    cagr_years : int
        Number of years for CAGR calculation (default 5).
    consistency_years : int
        Number of years for payout consistency evaluation (default 10).

    Returns a :class:`DividendMetrics` instance.  On failure the ``error``
    field is non-empty and numeric fields default to zero / None.
    """
    ticker = ticker.upper().strip()

    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        dividends = tk.dividends
        assert isinstance(dividends, pd.Series)
    except Exception as exc:
        return DividendMetrics(
            ticker=ticker,
            current_price=0.0,
            currency="",
            annual_dividend=0.0,
            current_yield=0.0,
            dividend_cagr=None,
            cagr_years=0,
            payout_consistency=0.0,
            consistency_years=0,
            increase_streak=0,
            error=f"fetch failed: {exc}",
        )

    # ── Extract info fields ──────────────────────────────────────────────
    current_price = float(
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
        or 0
    )
    currency = str(info.get("currency", "USD"))
    name = str(info.get("longName") or info.get("shortName") or ticker)
    sector = str(info.get("sector", "N/A"))
    payout_ratio = info.get("payoutRatio")  # may be None
    if payout_ratio is not None:
        payout_ratio = float(payout_ratio)
    five_yr_avg = info.get("fiveYearAvgDividendYield")
    if five_yr_avg is not None:
        five_yr_avg = float(five_yr_avg) / 100.0  # yfinance returns as %, convert to ratio

    ex_div_date = ""
    raw_ex_div = info.get("exDividendDate")
    if raw_ex_div:
        try:
            # yfinance sometimes returns epoch seconds
            if isinstance(raw_ex_div, (int, float)):
                ex_div_date = datetime.fromtimestamp(raw_ex_div).strftime("%Y-%m-%d")
            else:
                ex_div_date = str(raw_ex_div)
        except Exception:
            ex_div_date = str(raw_ex_div)

    # ── Trailing 12-month dividend ───────────────────────────────────────
    trailing_rate = float(info.get("trailingAnnualDividendRate") or 0)
    # If yfinance doesn't provide a trailing rate, compute manually from
    # the last 12 months of dividend payments.
    if trailing_rate <= 0 and not dividends.empty:
        one_year_ago = datetime.now() - timedelta(days=365)
        idx = dividends.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_localize(None)
        recent = dividends[idx >= one_year_ago]
        trailing_rate = float(recent.sum()) if not recent.empty else 0.0

    current_yield = trailing_rate / current_price if current_price > 0 else 0.0

    # ── Annual dividend history ──────────────────────────────────────────
    annual_divs = _annualise_dividends(dividends)

    cagr, actual_cagr_years = _compute_cagr(annual_divs, cagr_years)
    consistency, actual_consistency_years = _compute_consistency(annual_divs, consistency_years)
    streak = _compute_streak(annual_divs)

    return DividendMetrics(
        ticker=ticker,
        current_price=current_price,
        currency=currency,
        annual_dividend=trailing_rate,
        current_yield=current_yield,
        dividend_cagr=cagr,
        cagr_years=actual_cagr_years,
        payout_consistency=consistency,
        consistency_years=actual_consistency_years,
        increase_streak=streak,
        name=name,
        sector=sector,
        ex_dividend_date=ex_div_date,
        payout_ratio=payout_ratio,
        five_year_avg_yield=five_yr_avg,
    )
