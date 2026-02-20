"""
data/yahoo.py — Yahoo Finance data provider (via yfinance).
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from .provider import DataProvider

# Periods yfinance accepts
VALID_PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"}
VALID_INTERVALS = {"1d", "1wk", "1mo"}


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
            raw = tk.history(start=start, end=end, interval=interval, auto_adjust=True)
        else:
            # Period mode
            period = (period or "6mo").lower().strip()
            if period not in VALID_PERIODS:
                raise ValueError(
                    f"Invalid period '{period}'. "
                    f"Valid options: {sorted(VALID_PERIODS)}"
                )
            raw = tk.history(period=period, interval=interval, auto_adjust=True)
        assert isinstance(raw, pd.DataFrame), "yfinance did not return a DataFrame"

        df: pd.DataFrame = self._normalise_columns(raw)

        # Drop any extra columns yfinance may return (dividends, splits, etc.)
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
