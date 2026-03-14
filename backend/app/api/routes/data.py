"""Data provider routes — historical data, live prices, sector info."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import (
    HistoricalDataRequest,
    LivePriceResponse,
    SectorInfo,
)
from app.providers.yahoo import YahooFinanceProvider

router = APIRouter(prefix="/data", tags=["data"])

# For now, always use Yahoo.  When we add more providers, this becomes
# a dependency that resolves based on user settings / asset type.
_provider = YahooFinanceProvider()


@router.post("/historical")
async def get_historical(body: HistoricalDataRequest):
    """Fetch historical OHLCV data for a ticker."""
    df = await _provider.get_historical(
        ticker=body.ticker,
        start=body.start,
        end=body.end,
        interval=body.interval,
    )
    if df.empty:
        raise HTTPException(404, f"No data found for {body.ticker}")
    # Return as list of dicts with ISO date strings
    records = df.reset_index().to_dict(orient="records")
    # Convert Timestamps to ISO strings
    for rec in records:
        for k, v in rec.items():
            if hasattr(v, "isoformat"):
                rec[k] = v.isoformat()
    return {"ticker": body.ticker, "interval": body.interval, "data": records}


@router.get("/price/{ticker}", response_model=LivePriceResponse)
async def get_live_price(ticker: str):
    """Fetch the latest price for a single ticker."""
    try:
        price = await _provider.get_live_price(ticker)
    except Exception as exc:
        raise HTTPException(400, f"Could not fetch price for {ticker}: {exc}")
    from datetime import datetime, timezone

    return LivePriceResponse(
        ticker=ticker.upper(),
        price=price,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/sector/{ticker}", response_model=SectorInfo)
async def get_sector_info(ticker: str):
    """Fetch GICS sector/industry classification for a ticker."""
    info = await _provider.get_sector_info(ticker)
    if info is None:
        raise HTTPException(404, f"No sector info found for {ticker}")
    return info


@router.post("/sectors")
async def get_bulk_sector_info(
    tickers: list[str],
):
    """Fetch sector info for multiple tickers at once."""
    result = await _provider.get_bulk_sector_info(tickers)
    return {"sectors": {k: v.model_dump() for k, v in result.items()}}


@router.get("/universes")
async def list_universes():
    """List available predefined universes."""
    return {
        "universes": [
            {"id": "sp500", "name": "S&P 500", "description": "500 largest US companies", "count": 500},
            {"id": "nasdaq100", "name": "Nasdaq 100", "description": "100 largest Nasdaq companies", "count": 100},
            {"id": "tsx60", "name": "TSX 60", "description": "60 largest Canadian companies", "count": 60},
            {"id": "russell1000", "name": "Russell 1000", "description": "1000 largest US companies", "count": 1000},
            {"id": "russell2000", "name": "Russell 2000", "description": "2000 small-cap US companies", "count": 2000},
            {"id": "sector_etfs", "name": "Sector ETFs", "description": "SPDR sector ETFs (XLK, XLF, XLE, etc.)", "count": 11},
            {"id": "custom", "name": "Custom", "description": "User-defined ticker list", "count": 0},
        ]
    }
