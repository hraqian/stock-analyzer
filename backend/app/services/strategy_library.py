"""Strategy library service — CRUD and built-in presets."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.strategy import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------
# These are injected as is_preset=True strategies the first time a user
# accesses the library.  They can't be deleted or modified by the user.

BUILT_IN_PRESETS: list[dict] = [
    {
        "name": "Momentum Swing",
        "description": (
            "Aggressive swing-trade setup that favours strong trending stocks. "
            "Higher score thresholds mean fewer trades but higher conviction. "
            "Best in bullish or trending markets."
        ),
        "trade_mode": "swing",
        "params": {
            "score_thresholds.strong_buy": 7.5,
            "score_thresholds.buy": 6.0,
            "score_thresholds.sell": 4.0,
            "score_thresholds.strong_sell": 3.0,
            "indicator_weight": 0.7,
            "pattern_weight": 0.3,
            "combination_mode": "weighted",
            "risk_management.stop_loss_pct": 0.05,
            "risk_management.take_profit_pct": 0.15,
            "risk_management.percent_equity": 0.15,
        },
    },
    {
        "name": "Mean Reversion",
        "description": (
            "Looks for oversold stocks likely to bounce back. Uses contrarian "
            "indicators (RSI, Stochastic) more heavily. Works best in range-bound "
            "or choppy markets. Day-trade version coming soon."
        ),
        "trade_mode": "swing",
        "params": {
            "score_thresholds.strong_buy": 3.0,
            "score_thresholds.buy": 4.0,
            "score_thresholds.sell": 6.5,
            "score_thresholds.strong_sell": 7.5,
            "indicator_weight": 0.6,
            "pattern_weight": 0.4,
            "combination_mode": "weighted",
            "risk_management.stop_loss_pct": 0.04,
            "risk_management.take_profit_pct": 0.08,
            "risk_management.percent_equity": 0.10,
        },
    },
    {
        "name": "Long-Term Trend Following",
        "description": (
            "Patient buy-and-hold style with wider stops. Only enters on strong "
            "trending signals and holds through minor pullbacks. Fewer trades, "
            "larger moves. Best for position trading (weeks to months)."
        ),
        "trade_mode": "long_term",
        "params": {
            "score_thresholds.strong_buy": 7.0,
            "score_thresholds.buy": 6.0,
            "score_thresholds.sell": 4.5,
            "score_thresholds.strong_sell": 3.5,
            "indicator_weight": 0.8,
            "pattern_weight": 0.2,
            "combination_mode": "weighted",
            "risk_management.stop_loss_pct": 0.10,
            "risk_management.take_profit_pct": 0.30,
            "risk_management.percent_equity": 0.20,
        },
    },
    {
        "name": "Sector Rotation",
        "description": (
            "Rotates between sector ETFs based on momentum. Uses wider thresholds "
            "and longer holding periods. Designed for the Sector ETFs universe. "
            "Lower conviction but diversified exposure."
        ),
        "trade_mode": "swing",
        "params": {
            "score_thresholds.strong_buy": 6.5,
            "score_thresholds.buy": 5.5,
            "score_thresholds.sell": 4.5,
            "score_thresholds.strong_sell": 3.5,
            "indicator_weight": 0.75,
            "pattern_weight": 0.25,
            "combination_mode": "weighted",
            "risk_management.stop_loss_pct": 0.07,
            "risk_management.take_profit_pct": 0.20,
            "risk_management.percent_equity": 0.12,
        },
    },
]


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------


async def ensure_presets(db: AsyncSession, user_id: int) -> None:
    """Insert built-in presets for a user if they don't exist yet."""
    result = await db.execute(
        select(Strategy).where(
            Strategy.user_id == user_id,
            Strategy.is_preset == True,  # noqa: E712
        )
    )
    existing = result.scalars().all()
    existing_names = {s.name for s in existing}

    for preset in BUILT_IN_PRESETS:
        if preset["name"] not in existing_names:
            strategy = Strategy(
                user_id=user_id,
                name=preset["name"],
                description=preset["description"],
                trade_mode=preset["trade_mode"],
                params_json=json.dumps(preset["params"]),
                is_preset=True,
                version=1,
            )
            db.add(strategy)

    await db.commit()


async def list_strategies(db: AsyncSession, user_id: int) -> list[dict]:
    """List all strategies for a user (including presets)."""
    await ensure_presets(db, user_id)
    result = await db.execute(
        select(Strategy)
        .where(Strategy.user_id == user_id)
        .order_by(Strategy.is_preset.desc(), Strategy.updated_at.desc())
    )
    strategies = result.scalars().all()
    return [_strategy_to_dict(s) for s in strategies]


async def get_strategy(db: AsyncSession, user_id: int, strategy_id: int) -> dict | None:
    """Get a single strategy by ID."""
    result = await db.execute(
        select(Strategy).where(
            Strategy.id == strategy_id,
            Strategy.user_id == user_id,
        )
    )
    strategy = result.scalar_one_or_none()
    if not strategy:
        return None
    return _strategy_to_dict(strategy)


async def create_strategy(db: AsyncSession, user_id: int, data: dict) -> dict:
    """Create a new user strategy."""
    strategy = Strategy(
        user_id=user_id,
        name=data["name"],
        description=data.get("description"),
        trade_mode=data.get("trade_mode", "swing"),
        ticker=data.get("ticker"),
        params_json=json.dumps(data.get("params", {})),
        total_return_pct=data.get("total_return_pct"),
        annualized_return_pct=data.get("annualized_return_pct"),
        sharpe_ratio=data.get("sharpe_ratio"),
        max_drawdown_pct=data.get("max_drawdown_pct"),
        win_rate_pct=data.get("win_rate_pct"),
        profit_factor=data.get("profit_factor"),
        stability_score=data.get("stability_score"),
        version=1,
    )
    db.add(strategy)
    await db.commit()
    await db.refresh(strategy)
    return _strategy_to_dict(strategy)


async def update_strategy(
    db: AsyncSession, user_id: int, strategy_id: int, data: dict
) -> dict | None:
    """Update a strategy.  Cannot update presets."""
    result = await db.execute(
        select(Strategy).where(
            Strategy.id == strategy_id,
            Strategy.user_id == user_id,
        )
    )
    strategy = result.scalar_one_or_none()
    if not strategy:
        return None
    if strategy.is_preset:
        # Allow toggling is_active on presets, but nothing else
        if "is_active" in data and data["is_active"] is not None:
            strategy.is_active = data["is_active"]
            strategy.updated_at = datetime.now(timezone.utc)
            await db.commit()
            await db.refresh(strategy)
            return _strategy_to_dict(strategy)
        return None  # Cannot modify preset params

    # Bump version if params changed
    if "params" in data and data["params"] is not None:
        strategy.params_json = json.dumps(data["params"])
        strategy.version += 1

    for field in [
        "name", "description", "trade_mode", "ticker", "is_active",
        "total_return_pct", "annualized_return_pct", "sharpe_ratio",
        "max_drawdown_pct", "win_rate_pct", "profit_factor", "stability_score",
    ]:
        if field in data and data[field] is not None:
            setattr(strategy, field, data[field])

    strategy.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(strategy)
    return _strategy_to_dict(strategy)


async def delete_strategy(
    db: AsyncSession, user_id: int, strategy_id: int
) -> bool:
    """Delete a strategy.  Cannot delete presets."""
    result = await db.execute(
        select(Strategy).where(
            Strategy.id == strategy_id,
            Strategy.user_id == user_id,
        )
    )
    strategy = result.scalar_one_or_none()
    if not strategy or strategy.is_preset:
        return False
    await db.delete(strategy)
    await db.commit()
    return True


async def export_strategy(
    db: AsyncSession, user_id: int, strategy_id: int
) -> dict | None:
    """Export a strategy as a portable dict (JSON format)."""
    result = await db.execute(
        select(Strategy).where(
            Strategy.id == strategy_id,
            Strategy.user_id == user_id,
        )
    )
    strategy = result.scalar_one_or_none()
    if not strategy:
        return None
    return {
        "name": strategy.name,
        "description": strategy.description,
        "version": strategy.version,
        "trade_mode": strategy.trade_mode,
        "ticker": strategy.ticker,
        "params": json.loads(strategy.params_json),
        "metrics": {
            "total_return_pct": strategy.total_return_pct,
            "annualized_return_pct": strategy.annualized_return_pct,
            "sharpe_ratio": strategy.sharpe_ratio,
            "max_drawdown_pct": strategy.max_drawdown_pct,
            "win_rate_pct": strategy.win_rate_pct,
            "profit_factor": strategy.profit_factor,
            "stability_score": strategy.stability_score,
        },
    }


async def import_strategy(db: AsyncSession, user_id: int, data: dict) -> dict:
    """Import a strategy from a portable dict."""
    metrics = data.get("metrics", {})
    return await create_strategy(db, user_id, {
        "name": data["name"],
        "description": data.get("description"),
        "trade_mode": data.get("trade_mode", "swing"),
        "ticker": data.get("ticker"),
        "params": data.get("params", {}),
        "total_return_pct": metrics.get("total_return_pct"),
        "annualized_return_pct": metrics.get("annualized_return_pct"),
        "sharpe_ratio": metrics.get("sharpe_ratio"),
        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "win_rate_pct": metrics.get("win_rate_pct"),
        "profit_factor": metrics.get("profit_factor"),
        "stability_score": metrics.get("stability_score"),
    })


def _strategy_to_dict(strategy: Strategy) -> dict:
    """Convert a Strategy ORM object to a JSON-safe dict."""
    return {
        "id": strategy.id,
        "name": strategy.name,
        "description": strategy.description,
        "version": strategy.version,
        "is_preset": strategy.is_preset,
        "trade_mode": strategy.trade_mode,
        "ticker": strategy.ticker,
        "params": json.loads(strategy.params_json),
        "total_return_pct": strategy.total_return_pct,
        "annualized_return_pct": strategy.annualized_return_pct,
        "sharpe_ratio": strategy.sharpe_ratio,
        "max_drawdown_pct": strategy.max_drawdown_pct,
        "win_rate_pct": strategy.win_rate_pct,
        "profit_factor": strategy.profit_factor,
        "stability_score": strategy.stability_score,
        "is_active": strategy.is_active,
        "created_at": strategy.created_at.isoformat() if strategy.created_at else "",
        "updated_at": strategy.updated_at.isoformat() if strategy.updated_at else "",
    }
