"""
data/yahoo.py — Yahoo Finance data provider (via yfinance).

Uses ``auto_adjust=False`` so that OHLC prices reflect the actual traded
values (split-adjusted only, *not* dividend-adjusted).  This prevents
dividend adjustments from distorting candlestick pattern geometry, gap
detection, and support/resistance levels.

The ``Adj Close`` column is dropped after fetch — the rest of the codebase
works with split-adjusted prices only.
"""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd
import yfinance as yf

from .provider import DataProvider

logger = logging.getLogger(__name__)

# Periods yfinance accepts
VALID_PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"}
VALID_INTERVALS = {
    # Intraday intervals (yfinance limits: 1m=7d, 5m/15m/30m=60d, 60m/1h=730d)
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
    # Daily and above
    "1d", "5d", "1wk", "1mo", "3mo",
}

# Maximum tickers per yf.download() call.  Yahoo occasionally rejects
# very large batches; 200 is a safe ceiling that still gives a huge
# speed-up over serial fetching.
_BATCH_CHUNK_SIZE = 200


def fetch_batch(
    tickers: Sequence[str],
    period: str = "6mo",
    interval: str = "1d",
    chunk_size: int = _BATCH_CHUNK_SIZE,
) -> dict[str, pd.DataFrame]:
    """Bulk-download OHLCV data for many tickers at once.

    Uses ``yf.download()`` which fetches all tickers in a single HTTP
    session and is 10–50× faster than calling ``Ticker.history()`` in a
    loop.  Large lists are split into chunks of *chunk_size* to stay
    within Yahoo's per-request limits.

    Args:
        tickers:    Sequence of ticker symbols.
        period:     How far back to fetch (e.g. ``"6mo"``).
        interval:   Bar interval (e.g. ``"1d"``).
        chunk_size: Max tickers per ``yf.download()`` call.

    Returns:
        ``{ticker: DataFrame}`` — only tickers that returned valid
        OHLCV data are included.  Tickers that failed silently
        (no data, delisted, etc.) are omitted.
    """
    if not tickers:
        return {}

    period = (period or "6mo").lower().strip()
    interval = interval.lower().strip()
    if period not in VALID_PERIODS:
        raise ValueError(f"Invalid period '{period}'. Valid: {sorted(VALID_PERIODS)}")
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Invalid interval '{interval}'. Valid: {sorted(VALID_INTERVALS)}")

    unique_tickers = list(dict.fromkeys(t.upper().strip() for t in tickers))
    result: dict[str, pd.DataFrame] = {}

    for i in range(0, len(unique_tickers), chunk_size):
        chunk = unique_tickers[i : i + chunk_size]
        logger.info(
            "fetch_batch: downloading %d tickers (chunk %d/%d)",
            len(chunk),
            i // chunk_size + 1,
            (len(unique_tickers) + chunk_size - 1) // chunk_size,
        )

        try:
            raw = yf.download(
                tickers=chunk,
                period=period,
                interval=interval,
                auto_adjust=False,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception:
            logger.exception("yf.download failed for chunk starting at index %d", i)
            continue

        if raw is None or raw.empty:
            logger.warning("yf.download returned empty for chunk starting at index %d", i)
            continue

        # ── Parse the multi-ticker result ────────────────────────────
        if len(chunk) == 1:
            # Single ticker: yf.download returns a flat DataFrame
            df = _clean_single(raw, chunk[0])
            if df is not None:
                result[chunk[0]] = df
        else:
            # Multiple tickers: columns are MultiIndex (ticker, field)
            for ticker in chunk:
                try:
                    if ticker in raw.columns.get_level_values(0):
                        ticker_df = raw[ticker].copy()
                    else:
                        continue
                except (KeyError, TypeError):
                    continue
                df = _clean_single(ticker_df, ticker)
                if df is not None:
                    result[ticker] = df

    logger.info(
        "fetch_batch: got data for %d / %d tickers", len(result), len(unique_tickers)
    )
    return result


def _clean_single(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Normalise a single-ticker DataFrame from yf.download().

    Returns None if the data is invalid (empty, missing columns, etc.).
    """
    if raw is None or raw.empty:
        return None

    df = raw.copy()
    df.columns = [c.strip().lower() if isinstance(c, str) else str(c).strip().lower() for c in df.columns]

    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    if len(keep) < 5:
        return None

    df = df[keep].copy()
    df.dropna(subset=["open", "high", "low", "close"], how="all", inplace=True)
    if df.empty:
        return None

    df.sort_index(inplace=True)
    df.index = pd.to_datetime(df.index)
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df


class YahooFinanceProvider(DataProvider):
    """Fetches data from Yahoo Finance using the yfinance library."""

    def fetch(
        self,
        ticker: str,
        period: str | None = "6mo",
        interval: str = "1d",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        ticker = ticker.upper().strip()
        interval = interval.lower().strip()

        if interval not in VALID_INTERVALS:
            raise ValueError(
                f"Invalid interval '{interval}'. "
                f"Valid options: {sorted(VALID_INTERVALS)}"
            )

        tk = yf.Ticker(ticker)

        if start:
            # Date-range mode: ignore period
            raw = tk.history(start=start, end=end, interval=interval, auto_adjust=False)
        else:
            # Period mode
            period = (period or "6mo").lower().strip()
            if period not in VALID_PERIODS:
                raise ValueError(
                    f"Invalid period '{period}'. "
                    f"Valid options: {sorted(VALID_PERIODS)}"
                )
            raw = tk.history(period=period, interval=interval, auto_adjust=False)
        assert isinstance(raw, pd.DataFrame), "yfinance did not return a DataFrame"

        df: pd.DataFrame = self._normalise_columns(raw)

        # Drop extra columns (dividends, splits, adj close, etc.) — we use
        # raw split-adjusted OHLCV only (auto_adjust=False).
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep].copy()

        self._validate_ohlcv(df, ticker)

        df.sort_index(inplace=True)
        df.index = pd.to_datetime(df.index)
        # Remove timezone info for consistent downstream handling
        if hasattr(df.index, "tz") and df.index.tz is not None:  # type: ignore[union-attr]
            df.index = df.index.tz_localize(None)  # type: ignore[assignment]

        return df

    def get_info(self, ticker: str) -> dict:
        ticker = ticker.upper().strip()
        tk = yf.Ticker(ticker)
        try:
            info = tk.info or {}
        except Exception:
            info = {}

        return {
            "name": info.get("longName") or info.get("shortName") or ticker,
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "N/A"),
            "current_price": (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("previousClose")
            ),
        }
