"""SQLAlchemy ORM model for saved strategies."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Strategy(Base):
    """A saved strategy configuration with optional backtest results."""

    __tablename__ = "strategies"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)

    # Identity
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str | None] = mapped_column(Text, default=None)
    version: Mapped[int] = mapped_column(Integer, default=1)
    is_preset: Mapped[bool] = mapped_column(default=False)

    # Trade mode used when this strategy was created/tuned
    trade_mode: Mapped[str] = mapped_column(String(20), default="swing")

    # The full parameter set as JSON (config overrides)
    params_json: Mapped[str] = mapped_column(Text, default="{}")

    # Ticker this was tuned/tested on (optional — presets are ticker-agnostic)
    ticker: Mapped[str | None] = mapped_column(String(20), default=None)

    # Key performance metrics (from the most recent backtest/walk-forward)
    total_return_pct: Mapped[float | None] = mapped_column(Float, default=None)
    annualized_return_pct: Mapped[float | None] = mapped_column(Float, default=None)
    sharpe_ratio: Mapped[float | None] = mapped_column(Float, default=None)
    max_drawdown_pct: Mapped[float | None] = mapped_column(Float, default=None)
    win_rate_pct: Mapped[float | None] = mapped_column(Float, default=None)
    profit_factor: Mapped[float | None] = mapped_column(Float, default=None)
    stability_score: Mapped[float | None] = mapped_column(Float, default=None)

    # Is this strategy currently "activated" for Portfolio Simulation?
    is_active: Mapped[bool] = mapped_column(default=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
