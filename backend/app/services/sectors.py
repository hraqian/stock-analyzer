"""backend/app/services/sectors.py — Sector & Segments engine.

Provides sector-level analysis: heatmap data (momentum scores + color for
all 11 GICS sectors), relative strength vs SPY, regime per sector, rotation
tracking across 1W/1M/3M windows, and top movers per sector.

Uses the 11 SPDR Select Sector ETFs as proxies for GICS sectors:
  XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLB, XLRE, XLU, XLC

All functions are synchronous — designed for use inside a ThreadPoolExecutor.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sector ETF → GICS sector mapping
# ---------------------------------------------------------------------------

SECTOR_ETF_MAP: dict[str, str] = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLE":  "Energy",
    "XLV":  "Health Care",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLI":  "Industrials",
    "XLB":  "Materials",
    "XLRE": "Real Estate",
    "XLU":  "Utilities",
    "XLC":  "Communication Services",
}

# Reverse: sector name → ETF ticker
SECTOR_NAME_TO_ETF = {v: k for k, v in SECTOR_ETF_MAP.items()}

BENCHMARK = "SPY"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SectorMomentum:
    """Momentum data for a single sector across multiple time windows."""

    etf: str
    sector: str
    return_1w: float = 0.0
    return_1m: float = 0.0
    return_3m: float = 0.0
    # Relative strength vs benchmark
    rs_1w: float = 0.0
    rs_1m: float = 0.0
    rs_3m: float = 0.0
    # Current state
    current_price: float = 0.0
    avg_volume: float = 0.0
    regime: str = ""
    regime_confidence: float = 0.0
    # Momentum score (composite of returns + RS)
    momentum_score: float = 0.0


@dataclass
class SectorOverviewResult:
    """Full sector overview response."""

    sectors: list[SectorMomentum]
    benchmark_return_1w: float = 0.0
    benchmark_return_1m: float = 0.0
    benchmark_return_3m: float = 0.0
    elapsed_seconds: float = 0.0


@dataclass
class SectorTopMover:
    """A top mover within a sector."""

    ticker: str
    name: str
    return_1m: float = 0.0
    current_price: float = 0.0


@dataclass
class SectorDetailResult:
    """Detail view for a single sector."""

    etf: str
    sector: str
    return_1w: float = 0.0
    return_1m: float = 0.0
    return_3m: float = 0.0
    rs_1w: float = 0.0
    rs_1m: float = 0.0
    rs_3m: float = 0.0
    regime: str = ""
    regime_confidence: float = 0.0
    momentum_score: float = 0.0
    top_movers: list[SectorTopMover] = field(default_factory=list)
    worst_movers: list[SectorTopMover] = field(default_factory=list)
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Helper: compute returns for multiple windows
# ---------------------------------------------------------------------------


def _compute_returns(df: pd.DataFrame) -> dict[str, float]:
    """Compute 1W, 1M, 3M returns from a daily OHLCV DataFrame.

    Returns dict with keys: return_1w, return_1m, return_3m.
    Missing windows (not enough data) default to 0.0.
    """
    if df.empty:
        return {"return_1w": 0.0, "return_1m": 0.0, "return_3m": 0.0}

    close = df["close"]
    last = float(close.iloc[-1])
    n = len(close)

    def _ret(bars_back: int) -> float:
        if n <= bars_back:
            return 0.0
        prev = float(close.iloc[-bars_back - 1])
        if prev <= 0:
            return 0.0
        return (last - prev) / prev

    return {
        "return_1w": _ret(5),
        "return_1m": _ret(21),
        "return_3m": _ret(63),
    }


def _momentum_score(
    return_1w: float,
    return_1m: float,
    return_3m: float,
    rs_1m: float,
) -> float:
    """Compute a composite momentum score (0-10).

    Weights: 1M return 35%, 3M return 25%, 1W return 15%, RS 1M 25%.
    The raw weighted average is scaled from roughly [-0.3, +0.3] → [0, 10].
    """
    raw = (
        0.15 * return_1w
        + 0.35 * return_1m
        + 0.25 * return_3m
        + 0.25 * rs_1m
    )
    # Scale: 0% → 5, +/-30% → 0 or 10
    score = 5.0 + (raw / 0.30) * 5.0
    return max(0.0, min(10.0, round(score, 2)))


# ---------------------------------------------------------------------------
# Main functions
# ---------------------------------------------------------------------------


def get_sector_overview() -> SectorOverviewResult:
    """Fetch sector ETFs + SPY benchmark and compute heatmap data.

    Returns momentum, relative strength, and regime for all 11 sectors.
    """
    from data.yahoo import fetch_batch  # type: ignore[import-untyped]

    t0 = time.time()

    tickers = list(SECTOR_ETF_MAP.keys()) + [BENCHMARK]
    data = fetch_batch(tickers, period="6mo", interval="1d")

    # Benchmark returns
    spy_returns = {"return_1w": 0.0, "return_1m": 0.0, "return_3m": 0.0}
    if BENCHMARK in data:
        spy_returns = _compute_returns(data[BENCHMARK])

    # Optional: regime classifier
    regime_clf = None
    try:
        from config import Config  # type: ignore[import-untyped]
        from engine.regime import RegimeClassifier  # type: ignore[import-untyped]
        cfg = Config.defaults()
        regime_clf = RegimeClassifier(cfg)
    except Exception:
        logger.debug("Could not load RegimeClassifier", exc_info=True)

    sectors: list[SectorMomentum] = []

    for etf, sector_name in SECTOR_ETF_MAP.items():
        if etf not in data:
            sectors.append(SectorMomentum(etf=etf, sector=sector_name))
            continue

        df = data[etf]
        rets = _compute_returns(df)

        # Relative strength = sector return - benchmark return
        rs_1w = rets["return_1w"] - spy_returns["return_1w"]
        rs_1m = rets["return_1m"] - spy_returns["return_1m"]
        rs_3m = rets["return_3m"] - spy_returns["return_3m"]

        # Regime
        regime_label = ""
        regime_conf = 0.0
        if regime_clf is not None:
            try:
                regime = regime_clf.classify(df)
                regime_label = regime.label
                regime_conf = regime.confidence
            except Exception:
                pass

        # Current state
        current_price = float(df["close"].iloc[-1]) if not df.empty else 0.0
        avg_vol = (
            float(df["volume"].tail(20).mean())
            if "volume" in df.columns and not df.empty
            else 0.0
        )

        score = _momentum_score(
            rets["return_1w"], rets["return_1m"], rets["return_3m"], rs_1m,
        )

        sectors.append(SectorMomentum(
            etf=etf,
            sector=sector_name,
            return_1w=round(rets["return_1w"], 4),
            return_1m=round(rets["return_1m"], 4),
            return_3m=round(rets["return_3m"], 4),
            rs_1w=round(rs_1w, 4),
            rs_1m=round(rs_1m, 4),
            rs_3m=round(rs_3m, 4),
            current_price=round(current_price, 2),
            avg_volume=round(avg_vol, 0),
            regime=regime_label,
            regime_confidence=round(regime_conf, 2),
            momentum_score=score,
        ))

    # Sort by momentum score descending
    sectors.sort(key=lambda s: s.momentum_score, reverse=True)

    elapsed = time.time() - t0
    return SectorOverviewResult(
        sectors=sectors,
        benchmark_return_1w=round(spy_returns["return_1w"], 4),
        benchmark_return_1m=round(spy_returns["return_1m"], 4),
        benchmark_return_3m=round(spy_returns["return_3m"], 4),
        elapsed_seconds=round(elapsed, 2),
    )


# ---------------------------------------------------------------------------
# Top tickers per sector (representative holdings)
# ---------------------------------------------------------------------------

# Hand-curated representative holdings per sector (top 5-6 by market cap).
# This avoids slow yf.info() lookups at runtime.
SECTOR_HOLDINGS: dict[str, list[tuple[str, str]]] = {
    "Technology": [
        ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("NVDA", "Nvidia"),
        ("AVGO", "Broadcom"), ("CRM", "Salesforce"), ("ADBE", "Adobe"),
    ],
    "Financials": [
        ("JPM", "JPMorgan Chase"), ("V", "Visa"), ("MA", "Mastercard"),
        ("BAC", "Bank of America"), ("WFC", "Wells Fargo"), ("GS", "Goldman Sachs"),
    ],
    "Energy": [
        ("XOM", "Exxon Mobil"), ("CVX", "Chevron"), ("COP", "ConocoPhillips"),
        ("SLB", "Schlumberger"), ("EOG", "EOG Resources"), ("MPC", "Marathon Petroleum"),
    ],
    "Health Care": [
        ("UNH", "UnitedHealth"), ("JNJ", "Johnson & Johnson"), ("LLY", "Eli Lilly"),
        ("PFE", "Pfizer"), ("ABBV", "AbbVie"), ("MRK", "Merck"),
    ],
    "Consumer Discretionary": [
        ("AMZN", "Amazon"), ("TSLA", "Tesla"), ("HD", "Home Depot"),
        ("MCD", "McDonald's"), ("NKE", "Nike"), ("SBUX", "Starbucks"),
    ],
    "Consumer Staples": [
        ("PG", "Procter & Gamble"), ("KO", "Coca-Cola"), ("PEP", "PepsiCo"),
        ("COST", "Costco"), ("WMT", "Walmart"), ("PM", "Philip Morris"),
    ],
    "Industrials": [
        ("GE", "GE Aerospace"), ("CAT", "Caterpillar"), ("UNP", "Union Pacific"),
        ("HON", "Honeywell"), ("DE", "John Deere"), ("RTX", "RTX Corp"),
    ],
    "Materials": [
        ("LIN", "Linde"), ("SHW", "Sherwin-Williams"), ("APD", "Air Products"),
        ("FCX", "Freeport-McMoRan"), ("NUE", "Nucor"), ("ECL", "Ecolab"),
    ],
    "Real Estate": [
        ("PLD", "Prologis"), ("AMT", "American Tower"), ("EQIX", "Equinix"),
        ("SPG", "Simon Property"), ("O", "Realty Income"), ("PSA", "Public Storage"),
    ],
    "Utilities": [
        ("NEE", "NextEra Energy"), ("DUK", "Duke Energy"), ("SO", "Southern Co"),
        ("AEP", "American Electric"), ("D", "Dominion Energy"), ("SRE", "Sempra"),
    ],
    "Communication Services": [
        ("META", "Meta Platforms"), ("GOOG", "Alphabet"), ("NFLX", "Netflix"),
        ("DIS", "Disney"), ("TMUS", "T-Mobile"), ("CMCSA", "Comcast"),
    ],
}


def refresh_holdings_from_yfinance(sector_name: str) -> list[tuple[str, str]]:
    """Fetch top holdings for a sector's ETF from yfinance and save to config.

    Uses yf.Ticker(etf).info to get the fund's top holdings.
    Falls back to the static SECTOR_HOLDINGS list if the fetch fails.
    Successful results are persisted to the holdings config file.

    Args:
        sector_name: GICS sector name (e.g. "Technology").

    Returns:
        List of (ticker, company_name) tuples.
    """
    import yfinance as yf  # type: ignore[import-untyped]
    from app.services.holdings_config import set_sector_holdings

    etf = SECTOR_NAME_TO_ETF.get(sector_name)
    if not etf:
        raise ValueError(f"Unknown sector: {sector_name}")

    try:
        logger.info("Fetching holdings for %s (%s) from yfinance...", sector_name, etf)
        ticker_obj = yf.Ticker(etf)

        # yfinance exposes fund holdings via .funds_data.top_holdings (pandas DF)
        try:
            holdings_df = ticker_obj.funds_data.top_holdings
            if holdings_df is not None and not holdings_df.empty:
                result: list[tuple[str, str]] = []
                for sym in holdings_df.index[:8]:  # top 8
                    sym_str = str(sym).strip()
                    # Try to get the company name
                    name = str(holdings_df.loc[sym].get("Name", sym_str))
                    if name == sym_str or not name or name == "nan":
                        # Fallback: use a simple lookup from our static data
                        static = dict(SECTOR_HOLDINGS.get(sector_name, []))
                        name = static.get(sym_str, sym_str)
                    result.append((sym_str, name))
                if result:
                    # Persist to config file
                    set_sector_holdings(sector_name, [[t, n] for t, n in result])
                    logger.info("Refreshed %d holdings for %s", len(result), sector_name)
                    return result
        except Exception as exc:
            logger.warning("funds_data approach failed for %s: %s", etf, exc)

        # Fallback: try .info["holdings"] (older yfinance versions)
        try:
            info = ticker_obj.info or {}
            raw_holdings = info.get("holdings", [])
            if raw_holdings:
                result = []
                static = dict(SECTOR_HOLDINGS.get(sector_name, []))
                for h in raw_holdings[:8]:
                    sym_str = str(h.get("symbol", "")).strip()
                    name = h.get("holdingName", static.get(sym_str, sym_str))
                    if sym_str:
                        result.append((sym_str, name))
                if result:
                    # Persist to config file
                    set_sector_holdings(sector_name, [[t, n] for t, n in result])
                    logger.info("Refreshed %d holdings for %s (via info)", len(result), sector_name)
                    return result
        except Exception as exc:
            logger.warning("info approach failed for %s: %s", etf, exc)

        logger.warning("Could not fetch dynamic holdings for %s, using static fallback", sector_name)
    except Exception as exc:
        logger.warning("yfinance holdings fetch failed for %s: %s", sector_name, exc)

    return list(SECTOR_HOLDINGS.get(sector_name, []))


def get_effective_holdings(sector_name: str) -> list[tuple[str, str]]:
    """Get holdings for a sector: config overrides > built-in defaults."""
    from app.services.holdings_config import get_sector_holdings

    override = get_sector_holdings(sector_name)
    if override:
        return [(h[0], h[1]) for h in override]
    return list(SECTOR_HOLDINGS.get(sector_name, []))


def get_sector_detail(sector_name: str) -> SectorDetailResult:
    """Get detailed view for a single sector: returns, RS, regime, top/worst movers.

    Args:
        sector_name: GICS sector name (e.g. "Technology").

    Returns:
        SectorDetailResult with drill-down data.
    """
    from data.yahoo import fetch_batch  # type: ignore[import-untyped]

    t0 = time.time()

    etf = SECTOR_NAME_TO_ETF.get(sector_name)
    if not etf:
        raise ValueError(f"Unknown sector: {sector_name}")

    # Fetch the ETF + SPY + representative holdings
    holdings = get_effective_holdings(sector_name)
    holding_tickers = [h[0] for h in holdings]
    all_tickers = [etf, BENCHMARK] + holding_tickers

    data = fetch_batch(all_tickers, period="6mo", interval="1d")

    # Benchmark returns
    spy_returns = {"return_1w": 0.0, "return_1m": 0.0, "return_3m": 0.0}
    if BENCHMARK in data:
        spy_returns = _compute_returns(data[BENCHMARK])

    # ETF returns
    etf_returns = {"return_1w": 0.0, "return_1m": 0.0, "return_3m": 0.0}
    if etf in data:
        etf_returns = _compute_returns(data[etf])

    rs_1w = etf_returns["return_1w"] - spy_returns["return_1w"]
    rs_1m = etf_returns["return_1m"] - spy_returns["return_1m"]
    rs_3m = etf_returns["return_3m"] - spy_returns["return_3m"]

    # Regime
    regime_label = ""
    regime_conf = 0.0
    try:
        from config import Config  # type: ignore[import-untyped]
        from engine.regime import RegimeClassifier  # type: ignore[import-untyped]
        cfg = Config.defaults()
        regime_clf = RegimeClassifier(cfg)
        if etf in data:
            regime = regime_clf.classify(data[etf])
            regime_label = regime.label
            regime_conf = regime.confidence
    except Exception:
        pass

    score = _momentum_score(
        etf_returns["return_1w"], etf_returns["return_1m"],
        etf_returns["return_3m"], rs_1m,
    )

    # Compute 1M return for each holding
    holding_returns: list[tuple[str, str, float, float]] = []
    for ticker_sym, name in holdings:
        if ticker_sym in data:
            hdf = data[ticker_sym]
            hrets = _compute_returns(hdf)
            price = float(hdf["close"].iloc[-1]) if not hdf.empty else 0.0
            holding_returns.append((ticker_sym, name, hrets["return_1m"], price))
        else:
            holding_returns.append((ticker_sym, name, 0.0, 0.0))

    # Sort by 1M return
    holding_returns.sort(key=lambda x: x[2], reverse=True)

    top_movers = [
        SectorTopMover(
            ticker=t, name=n,
            return_1m=round(r, 4),
            current_price=round(p, 2),
        )
        for t, n, r, p in holding_returns[:5]
    ]
    worst_movers = [
        SectorTopMover(
            ticker=t, name=n,
            return_1m=round(r, 4),
            current_price=round(p, 2),
        )
        for t, n, r, p in reversed(holding_returns[-3:])
    ]

    elapsed = time.time() - t0

    return SectorDetailResult(
        etf=etf,
        sector=sector_name,
        return_1w=round(etf_returns["return_1w"], 4),
        return_1m=round(etf_returns["return_1m"], 4),
        return_3m=round(etf_returns["return_3m"], 4),
        rs_1w=round(rs_1w, 4),
        rs_1m=round(rs_1m, 4),
        rs_3m=round(rs_3m, 4),
        regime=regime_label,
        regime_confidence=round(regime_conf, 2),
        momentum_score=score,
        top_movers=top_movers,
        worst_movers=worst_movers,
        elapsed_seconds=round(elapsed, 2),
    )
