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

import pandas as pd
import yfinance as yf

from .provider import DataProvider

# Periods yfinance accepts
VALID_PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"}
VALID_INTERVALS = {
    # Intraday intervals (yfinance limits: 1m=7d, 5m/15m/30m=60d, 60m/1h=730d)
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
    # Daily and above
    "1d", "5d", "1wk", "1mo", "3mo",
}


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
