"""analysis/dividend.py — Dividend metrics computation and scanning.

Fetches dividend history and info from yfinance and computes per-ticker
metrics used by the dividend scanner:

- **Current yield** — trailing 12-month dividend / current price.
- **Dividend growth rate** — CAGR of annual dividends over *n* years.
- **Payout consistency** — fraction of years with a non-zero dividend
  (over the evaluation window).
- **Streak length** — consecutive years of year-over-year dividend increases.

Scoring curves and thresholds are read from the ``dividend`` config section.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

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


# ─────────────────────────────────────────────────────────────────────────────
# Scoring — convert raw metrics to 0-10 scores using config curves
# ─────────────────────────────────────────────────────────────────────────────


def _lerp(value: float, lo: float, hi: float, lo_score: float, hi_score: float) -> float:
    """Linear interpolation clamped to [lo_score, hi_score]."""
    if hi == lo:
        return hi_score
    t = (value - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    return lo_score + t * (hi_score - lo_score)


def score_yield(current_yield: float, cfg: dict[str, Any]) -> float:
    """Score current dividend yield on 0–10 scale.

    Uses piecewise linear interpolation:
    - yield <= low_target  → 0
    - yield == mid_target  → 5
    - yield >= high_target → 10
    - yield >  distress    → penalised back toward 5
    """
    ys = cfg.get("yield_scoring", {})
    low = float(ys.get("low_target", 0.005))
    mid = float(ys.get("mid_target", 0.03))
    high = float(ys.get("high_target", 0.06))
    distress = float(ys.get("distress_threshold", 0.10))

    if current_yield <= low:
        return 0.0
    elif current_yield <= mid:
        return _lerp(current_yield, low, mid, 0.0, 5.0)
    elif current_yield <= high:
        return _lerp(current_yield, mid, high, 5.0, 10.0)
    elif current_yield <= distress:
        return 10.0  # still excellent, just high
    else:
        # Distress penalty: linearly drop back toward 5
        penalty_ratio = min((current_yield - distress) / distress, 1.0)
        return 10.0 - penalty_ratio * 5.0


def score_growth(cagr: float | None, cfg: dict[str, Any]) -> float:
    """Score dividend growth (CAGR) on 0–10 scale."""
    gs = cfg.get("growth_scoring", {})
    neg_score = float(gs.get("negative_score", 2.0))
    zero_score = float(gs.get("zero_score", 4.0))
    target = float(gs.get("target_cagr", 0.07))
    max_cagr = float(gs.get("max_cagr", 0.15))

    if cagr is None:
        return zero_score  # insufficient data — neutral-low score

    if cagr < 0:
        # Negative growth: interpolate between 0 and neg_score
        # Very negative (-20%+) gets 0, slightly negative gets neg_score
        return max(0.0, _lerp(cagr, -0.20, 0.0, 0.0, neg_score))
    elif cagr <= target:
        return _lerp(cagr, 0.0, target, zero_score, 8.0)
    else:
        return _lerp(cagr, target, max_cagr, 8.0, 10.0)


def score_consistency(consistency: float, cfg: dict[str, Any]) -> float:
    """Score payout consistency (0-1 fraction) on 0–10 scale."""
    cs = cfg.get("consistency_scoring", {})
    full_thresh = float(cs.get("full_score_threshold", 1.0))
    half_thresh = float(cs.get("half_score_threshold", 0.50))

    if consistency >= full_thresh:
        return 10.0
    elif consistency >= half_thresh:
        return _lerp(consistency, half_thresh, full_thresh, 5.0, 10.0)
    else:
        return _lerp(consistency, 0.0, half_thresh, 0.0, 5.0)


def score_streak(streak: int, cfg: dict[str, Any]) -> float:
    """Score consecutive years of dividend increases on 0–10 scale."""
    ss = cfg.get("streak_scoring", {})
    aristocrat = int(ss.get("aristocrat_years", 25))
    good = int(ss.get("good_years", 10))
    decent = int(ss.get("decent_years", 5))

    if streak >= aristocrat:
        return 10.0
    elif streak >= good:
        return _lerp(float(streak), float(good), float(aristocrat), 7.0, 10.0)
    elif streak >= decent:
        return _lerp(float(streak), float(decent), float(good), 5.0, 7.0)
    elif streak > 0:
        return _lerp(float(streak), 0.0, float(decent), 1.0, 5.0)
    else:
        return 0.0


def compute_dividend_score(
    metrics: DividendMetrics,
    div_cfg: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Compute a weighted composite dividend score for a ticker.

    Returns ``(composite_score, component_scores)`` where component_scores
    is a dict with keys ``yield``, ``growth``, ``consistency``, ``streak``.
    """
    weights = div_cfg.get("weights", {})
    w_yield = float(weights.get("yield", 0.30))
    w_growth = float(weights.get("growth", 0.30))
    w_consistency = float(weights.get("consistency", 0.20))
    w_streak = float(weights.get("streak", 0.20))
    w_total = w_yield + w_growth + w_consistency + w_streak

    s_yield = score_yield(metrics.current_yield, div_cfg)
    s_growth = score_growth(metrics.dividend_cagr, div_cfg)
    s_consistency = score_consistency(metrics.payout_consistency, div_cfg)
    s_streak = score_streak(metrics.increase_streak, div_cfg)

    if w_total > 0:
        composite = (
            w_yield * s_yield
            + w_growth * s_growth
            + w_consistency * s_consistency
            + w_streak * s_streak
        ) / w_total
    else:
        composite = (s_yield + s_growth + s_consistency + s_streak) / 4.0

    components = {
        "yield": round(s_yield, 2),
        "growth": round(s_growth, 2),
        "consistency": round(s_consistency, 2),
        "streak": round(s_streak, 2),
    }
    return round(composite, 2), components


# ─────────────────────────────────────────────────────────────────────────────
# Dividend scan result
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DividendScanResult:
    """Result of scanning a single ticker for dividend quality."""

    ticker: str
    name: str
    sector: str
    currency: str
    current_price: float

    # Raw metrics
    current_yield: float        # 0–1 ratio (e.g. 0.035 = 3.5%)
    annual_dividend: float
    dividend_cagr: float | None
    cagr_years: int
    payout_consistency: float
    consistency_years: int
    increase_streak: int

    # Scores (0–10)
    yield_score: float
    growth_score: float
    consistency_score: float
    streak_score: float
    dividend_score: float       # weighted composite

    # Optional technical blend
    technical_score: float | None = None    # from existing scanner (None = not blended)
    blended_score: float | None = None      # dividend + technical composite

    # Metadata
    payout_ratio: float | None = None
    five_year_avg_yield: float | None = None
    ex_dividend_date: str = ""
    error: str = ""

    @property
    def sort_key(self) -> float:
        """Primary sort key — use blended_score if available, else dividend_score."""
        return self.blended_score if self.blended_score is not None else self.dividend_score


# ─────────────────────────────────────────────────────────────────────────────
# Dividend scanner engine
# ─────────────────────────────────────────────────────────────────────────────


class DividendScanner:
    """Parallel dividend scanner.

    Scans a list of tickers, fetches dividend metrics, scores them using
    config-driven curves, filters by thresholds, and ranks results.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to scan.
    div_cfg : dict
        The ``dividend`` config section (from ``cfg.section("dividend")``).
    on_progress : callable | None
        ``(completed, total, ticker, result | None) -> None``
    """

    def __init__(
        self,
        tickers: list[str],
        div_cfg: dict[str, Any],
        on_progress: Callable[[int, int, str, DividendScanResult | None], None] | None = None,
    ) -> None:
        self._tickers = [t.upper().strip() for t in tickers]
        self._cfg = div_cfg
        self._on_progress = on_progress
        self._results: list[DividendScanResult] = []

    @property
    def results(self) -> list[DividendScanResult]:
        return list(self._results)

    def run(self) -> list[DividendScanResult]:
        """Run the dividend scan.  Returns all results (including filtered-out)."""
        self._results = []
        total = len(self._tickers)
        completed = 0
        max_workers = int(self._cfg.get("max_workers", 8))
        cagr_years = int(self._cfg.get("cagr_years", 5))
        consistency_years = int(self._cfg.get("consistency_years", 10))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    self._scan_one, ticker, cagr_years, consistency_years
                ): ticker
                for ticker in self._tickers
            }

            for future in as_completed(futures):
                ticker = futures[future]
                completed += 1
                try:
                    result = future.result()
                except Exception as exc:
                    result = DividendScanResult(
                        ticker=ticker, name=ticker, sector="", currency="",
                        current_price=0.0, current_yield=0.0,
                        annual_dividend=0.0, dividend_cagr=None, cagr_years=0,
                        payout_consistency=0.0, consistency_years=0,
                        increase_streak=0,
                        yield_score=0.0, growth_score=0.0,
                        consistency_score=0.0, streak_score=0.0,
                        dividend_score=0.0,
                        error=str(exc),
                    )
                self._results.append(result)
                if self._on_progress:
                    self._on_progress(completed, total, ticker, result)

        return list(self._results)

    def _scan_one(
        self, ticker: str, cagr_years: int, consistency_years: int,
    ) -> DividendScanResult:
        """Fetch metrics and score a single ticker."""
        metrics = fetch_dividend_metrics(
            ticker, cagr_years=cagr_years, consistency_years=consistency_years,
        )

        if metrics.error:
            return DividendScanResult(
                ticker=ticker, name=metrics.name or ticker, sector="", currency="",
                current_price=0.0, current_yield=0.0,
                annual_dividend=0.0, dividend_cagr=None, cagr_years=0,
                payout_consistency=0.0, consistency_years=0,
                increase_streak=0,
                yield_score=0.0, growth_score=0.0,
                consistency_score=0.0, streak_score=0.0,
                dividend_score=0.0,
                error=metrics.error,
            )

        composite, components = compute_dividend_score(metrics, self._cfg)

        return DividendScanResult(
            ticker=ticker,
            name=metrics.name,
            sector=metrics.sector,
            currency=metrics.currency,
            current_price=metrics.current_price,
            current_yield=metrics.current_yield,
            annual_dividend=metrics.annual_dividend,
            dividend_cagr=metrics.dividend_cagr,
            cagr_years=metrics.cagr_years,
            payout_consistency=metrics.payout_consistency,
            consistency_years=metrics.consistency_years,
            increase_streak=metrics.increase_streak,
            yield_score=components["yield"],
            growth_score=components["growth"],
            consistency_score=components["consistency"],
            streak_score=components["streak"],
            dividend_score=composite,
            payout_ratio=metrics.payout_ratio,
            five_year_avg_yield=metrics.five_year_avg_yield,
            ex_dividend_date=metrics.ex_dividend_date,
        )

    # ── Filtering & ranking ──────────────────────────────────────────────

    def filtered(self) -> list[DividendScanResult]:
        """Return results that pass the minimum-threshold filters."""
        min_yield = float(self._cfg.get("min_yield", 0.01))
        max_yield = float(self._cfg.get("max_yield", 0.15))
        min_cagr = float(self._cfg.get("min_cagr", -0.10))
        min_consistency = float(self._cfg.get("min_consistency", 0.50))
        min_streak = int(self._cfg.get("min_streak", 0))

        out: list[DividendScanResult] = []
        for r in self._results:
            if r.error:
                continue
            if r.current_yield < min_yield or r.current_yield > max_yield:
                continue
            if r.dividend_cagr is not None and r.dividend_cagr < min_cagr:
                continue
            if r.payout_consistency < min_consistency:
                continue
            if r.increase_streak < min_streak:
                continue
            out.append(r)
        return out

    def top(self, n: int = 20) -> list[DividendScanResult]:
        """Return top *n* dividend stocks by composite score (filtered)."""
        passing = self.filtered()
        passing.sort(key=lambda r: r.sort_key, reverse=True)
        return passing[:n]

    def errors(self) -> list[DividendScanResult]:
        """Return all tickers that failed during scan."""
        return [r for r in self._results if r.error]

    def summary(self) -> dict[str, Any]:
        """Quick summary dict."""
        ok = [r for r in self._results if not r.error]
        filtered = self.filtered()
        return {
            "total_tickers": len(self._tickers),
            "scanned": len(ok),
            "errors": len(self._results) - len(ok),
            "passed_filters": len(filtered),
            "avg_yield": (
                sum(r.current_yield for r in filtered) / len(filtered)
                if filtered else 0.0
            ),
            "avg_score": (
                sum(r.dividend_score for r in filtered) / len(filtered)
                if filtered else 0.0
            ),
        }
