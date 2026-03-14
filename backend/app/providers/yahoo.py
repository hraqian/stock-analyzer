"""Yahoo Finance data provider — free, no API key required."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pandas as pd
import yfinance as yf

from app.models.schemas import SectorInfo
from app.providers.base import DataProvider

# Shared thread pool for blocking yfinance calls
_executor = ThreadPoolExecutor(max_workers=4)


class YahooFinanceProvider(DataProvider):
    """Data provider backed by yfinance (free, rate-limited)."""

    @property
    def name(self) -> str:
        return "Yahoo Finance"

    @property
    def supported_asset_types(self) -> list[str]:
        return ["stock", "etf"]

    async def get_historical(
        self,
        ticker: str,
        start: str,
        end: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV via yfinance (runs in thread pool)."""
        loop = asyncio.get_running_loop()
        df = await loop.run_in_executor(
            _executor,
            partial(self._fetch_history, ticker, start, end, interval),
        )
        return df

    async def get_live_price(self, ticker: str) -> float:
        """Fetch the latest price via yfinance fast_info."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor, partial(self._fetch_price, ticker)
        )

    async def get_sector_info(self, ticker: str) -> SectorInfo | None:
        """Fetch GICS sector/industry from Yahoo Finance ticker info."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor, partial(self._fetch_sector, ticker)
        )

    # ------------------------------------------------------------------
    # Sync helpers (run inside thread pool)
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_history(
        ticker: str, start: str, end: str | None, interval: str
    ) -> pd.DataFrame:
        t = yf.Ticker(ticker)
        kwargs: dict = {"start": start, "interval": interval, "auto_adjust": True}
        if end:
            kwargs["end"] = end
        df = t.history(**kwargs)
        if df.empty:
            return pd.DataFrame()
        # Normalise column names
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        # Keep only OHLCV columns
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[keep]

    @staticmethod
    def _fetch_price(ticker: str) -> float:
        t = yf.Ticker(ticker)
        fi = t.fast_info
        price = getattr(fi, "last_price", None)
        if price is None:
            price = getattr(fi, "previous_close", 0.0)
        return float(price)

    @staticmethod
    def _fetch_sector(ticker: str) -> SectorInfo | None:
        t = yf.Ticker(ticker)
        info = t.info
        sector = info.get("sector", "")
        industry = info.get("industry", "")
        if not sector:
            return None
        return SectorInfo(
            ticker=ticker.upper(),
            sector=sector,
            industry=industry,
            industry_group=info.get("industryGroup", ""),
        )
