"""
data/provider.py — Abstract base class for all data providers.

To add a new data source (e.g. Alpha Vantage, Polygon.io), create a new
file in data/ and subclass DataProvider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataProvider(ABC):
    """Abstract interface for fetching OHLCV market data.

    All concrete providers must return DataFrames with these columns
    (case-insensitive matching is handled by each implementation):

        open, high, low, close, volume

    Index must be a DatetimeIndex.
    """

    # ------------------------------------------------------------------
    # Core interface — must be implemented by subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def fetch(
        self,
        ticker: str,
        period: str = "6mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data.

        Args:
            ticker:   Stock symbol, e.g. ``"AAPL"``.
            period:   How far back to fetch.  Accepted values:
                      ``1mo``, ``3mo``, ``6mo``, ``1y``, ``2y``, ``5y``.
            interval: Bar interval.  Accepted values:
                      ``1d``, ``1wk``, ``1mo``.

        Returns:
            DataFrame with columns [open, high, low, close, volume] and a
            DatetimeIndex, sorted ascending.

        Raises:
            ValueError: If ticker is invalid or no data is returned.
        """
        ...

    @abstractmethod
    def get_info(self, ticker: str) -> dict:
        """Return metadata for a ticker.

        Returns a dict that may include (but is not limited to):
            name, sector, industry, market_cap, currency, exchange
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers available to all providers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase all column names and strip whitespace."""
        df.columns = [c.strip().lower() for c in df.columns]
        return df

    @staticmethod
    def _validate_ohlcv(df: pd.DataFrame, ticker: str) -> None:
        """Raise ValueError if required columns are missing or df is empty."""
        if df is None or df.empty:
            raise ValueError(
                f"No data returned for '{ticker}'. "
                "Check the ticker symbol and your internet connection."
            )
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Data for '{ticker}' is missing columns: {missing}"
            )
