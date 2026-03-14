"""SQLAlchemy ORM models for user accounts and profiles."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class User(Base):
    """User account — authentication credentials."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # --- Profile / preferences (stored inline for simplicity) ---

    # Trade mode: "swing", "long_term"  (day_trade deferred)
    trade_mode: Mapped[str] = mapped_column(String(20), default="swing")
    # User mode: "normal" or "power_user"
    user_mode: Mapped[str] = mapped_column(String(20), default="normal")

    # Account
    starting_capital: Mapped[float] = mapped_column(Float, default=100_000.0)
    risk_tolerance: Mapped[str] = mapped_column(
        String(20), default="moderate"
    )  # conservative, moderate, aggressive

    # Cost model defaults
    commission_per_trade: Mapped[float] = mapped_column(Float, default=0.0)
    spread_pct: Mapped[float] = mapped_column(Float, default=0.02)
    slippage_pct: Mapped[float] = mapped_column(Float, default=0.01)
    tax_rate_short_term: Mapped[float] = mapped_column(Float, default=0.37)
    tax_rate_long_term: Mapped[float] = mapped_column(Float, default=0.20)

    # Serialised JSON blobs for complex preferences
    # (custom watchlists, filter defaults, display prefs, etc.)
    preferences_json: Mapped[str | None] = mapped_column(Text, default=None)
