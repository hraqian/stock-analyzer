"""Abstract base class for data providers.

All data providers (Yahoo, Polygon, Alpha Vantage, etc.) implement this
interface so the rest of the app is provider-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from app.models.schemas import SectorInfo


class DataProvider(ABC):
    """Abstract data provider interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name (e.g. 'Yahoo Finance')."""

    @property
    @abstractmethod
    def supported_asset_types(self) -> list[str]:
        """Asset types this provider can fetch (e.g. ['stock', 'etf'])."""

    @abstractmethod
    async def get_historical(
        self,
        ticker: str,
        start: str,
        end: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data.

        Returns a DataFrame with columns:
            Date (index), Open, High, Low, Close, Volume
        """

    @abstractmethod
    async def get_live_price(self, ticker: str) -> float:
        """Fetch the latest price for a ticker."""

    @abstractmethod
    async def get_sector_info(self, ticker: str) -> SectorInfo | None:
        """Fetch GICS sector/industry classification for a ticker."""

    async def get_bulk_sector_info(
        self, tickers: list[str]
    ) -> dict[str, SectorInfo]:
        """Fetch sector info for many tickers.  Default: call get_sector_info
        in a loop.  Subclasses may override for batch efficiency."""
        result: dict[str, SectorInfo] = {}
        for t in tickers:
            info = await self.get_sector_info(t)
            if info:
                result[t] = info
        return result
