"""Stock Analyzer — FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import init_db
from app.api.routes.auth import router as auth_router
from app.api.routes.data import router as data_router
from app.api.routes.sections import (
    scanner_router,
    analysis_router,
    sectors_router,
    strategy_router,
    portfolio_router,
    settings_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle events."""
    # Startup: create database tables
    await init_db()
    yield
    # Shutdown: nothing to clean up yet


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Stock Analyzer API — trading analysis, backtesting, and portfolio simulation.",
    lifespan=lifespan,
)

# CORS — allow the Next.js frontend to talk to the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all routers under /api prefix
app.include_router(auth_router, prefix="/api")
app.include_router(data_router, prefix="/api")
app.include_router(scanner_router, prefix="/api")
app.include_router(analysis_router, prefix="/api")
app.include_router(sectors_router, prefix="/api")
app.include_router(strategy_router, prefix="/api")
app.include_router(portfolio_router, prefix="/api")
app.include_router(settings_router, prefix="/api")


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}
