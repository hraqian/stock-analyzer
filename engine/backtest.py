"""
engine/backtest.py — Bar-by-bar backtesting engine.

Walks through historical OHLCV data, recomputes indicators at each
rebalance interval, generates signals via the plugged-in Strategy, and
simulates trade execution with commission/slippage modelling.

No lookahead bias: each bar only uses data available up to that point.

Usage:
    from engine.backtest import BacktestEngine
    from engine.score_strategy import ScoreBasedStrategy

    engine = BacktestEngine(data_provider=provider, strategy=strategy, cfg=cfg)
    result = engine.run("AAPL", period="2y")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from config import Config
    from data.provider import DataProvider
    from engine.strategy import Strategy

from indicators.registry import IndicatorRegistry
from analysis.scorer import CompositeScorer
from patterns.registry import PatternRegistry
from analysis.pattern_scorer import PatternCompositeScorer
from engine.strategy import Signal, StrategyContext, TradeOrder
from engine.suitability import TradingMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_pattern_name(raw: str) -> str:
    """Convert snake_case pattern names to Title Case labels.

    Examples:
        ``"bullish_engulfing"`` → ``"Bullish Engulfing"``
        ``"three_white_soldiers"`` → ``"Three White Soldiers"``
        ``"inside_bar_bullish"`` → ``"Inside Bar Bullish"``
    """
    return raw.replace("_", " ").title()


# Detector-aware strength → human-readable confidence label.
# Each detector uses a different strength scale, so the thresholds differ.
_STRENGTH_THRESHOLDS: dict[str, list[tuple[float, str]]] = {
    "Candlesticks":    [(0.4, "Weak"), (0.7, "Moderate"), (1.0, "Strong")],
    "Inside/Outside":  [(0.4, "Weak"), (0.6, "Moderate"), (0.8, "Strong")],
    "Gaps":            [(0.5, "Weak"), (1.0, "Moderate"), (1.5, "Strong")],
    "Spikes":          [(0.5, "Weak"), (1.0, "Moderate"), (1.5, "Strong")],
}


def _strength_label(strength: float, detector: str) -> str:
    """Return a human-readable confidence label for a pattern strength value.

    Uses detector-aware thresholds because the raw strength scales differ
    between detectors (e.g. candlesticks 0.3–1.3 vs gaps 0–2.0).
    Falls back to the Candlesticks thresholds for unknown detectors.
    """
    thresholds = _STRENGTH_THRESHOLDS.get(detector, _STRENGTH_THRESHOLDS["Candlesticks"])
    for upper, label in thresholds:
        if strength <= upper:
            return label
    return "Very Strong"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BacktestTrade:
    """A single completed (round-trip) trade recorded during backtesting."""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    quantity: float
    side: str              # "long" or "short"
    pnl: float             # absolute P&L
    pnl_pct: float         # percentage P&L
    exit_reason: str = ""  # "signal", "stop_loss", "take_profit"
    entry_reason: str = "" # strategy notes at entry (e.g. "ind=6.80 pat=5.20 eff=6.32")


@dataclass
class SignificantPattern:
    """A single significant pattern detected during the backtest period.

    Used to build a timeline showing all noteworthy pattern events so
    users can see potential entry/exit opportunities — including ones the
    strategy may have missed.
    """
    date: str              # YYYY-MM-DD (or full timestamp for intraday)
    detector: str          # detector name, e.g. "Candlesticks", "Gaps"
    pattern: str           # specific pattern, e.g. "hammer", "breakaway"
    signal: str            # "bullish", "bearish", or "neutral"
    strength: float        # raw strength / magnitude (detector-specific)
    confidence: str = ""   # human-readable label: Weak / Moderate / Strong / Very Strong
    detail: str = ""       # extra context, e.g. "gap_pct=1.2%" or "z=3.1 Confirmed"


@dataclass
class BacktestResult:
    """Aggregated results from a backtest run."""
    ticker: str
    period: str
    strategy_name: str
    initial_cash: float = 100_000.0
    final_equity: float = 0.0
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    # equity_curve: list of {"date": str, "equity": float}

    # Significant patterns detected during the backtest period
    significant_patterns: list[SignificantPattern] = field(default_factory=list)

    # Performance metrics (populated by _compute_metrics)
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate_pct: float = 0.0
    total_trades: int = 0
    profit_factor: float = 0.0
    avg_trade_pnl_pct: float = 0.0
    best_trade_pnl_pct: float = 0.0
    worst_trade_pnl_pct: float = 0.0
    avg_bars_held: float = 0.0


# ---------------------------------------------------------------------------
# Position tracker (internal)
# ---------------------------------------------------------------------------

@dataclass
class _Position:
    """Tracks an open position during simulation."""
    side: str               # "long" or "short"
    entry_date: str
    entry_price: float
    quantity: float
    bars_held: int = 0
    entry_reason: str = ""  # carried through to BacktestTrade on close

    def unrealized_pnl(self, current_price: float) -> float:
        if self.side == "long":
            return (current_price - self.entry_price) * self.quantity
        return (self.entry_price - current_price) * self.quantity

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.entry_price == 0:
            return 0.0
        if self.side == "long":
            return (current_price - self.entry_price) / self.entry_price
        return (self.entry_price - current_price) / self.entry_price


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Runs a Strategy against historical OHLCV data bar-by-bar.

    Design:
      - Fetches the full historical DataFrame once.
      - Skips the first ``warmup_bars`` to allow indicators to stabilize.
      - Every ``rebalance_interval`` bars after warmup, recomputes indicators
        on the trailing data (up to the current bar — no lookahead).
      - Passes a StrategyContext to strategy.on_bar() to get a TradeOrder.
      - Executes the order with slippage + commission modelling.
      - Checks stop-loss / take-profit on every bar (not just rebalance bars).
      - Tracks equity curve and completed trades.
    """

    def __init__(
        self,
        data_provider: "DataProvider",
        strategy: "Strategy",
        cfg: "Config",
        trading_mode: TradingMode = TradingMode.LONG_SHORT,
    ) -> None:
        self._provider = data_provider
        self._strategy = strategy
        self._cfg = cfg
        self._trading_mode = trading_mode

        # Config sections
        bt_cfg = cfg.section("backtest")
        strat_cfg = cfg.section("strategy")

        self._initial_cash: float = float(bt_cfg.get("initial_cash", 100_000))
        self._commission: float = float(bt_cfg.get("commission_per_trade", 0.0))
        self._slippage_pct: float = float(bt_cfg.get("slippage_pct", 0.001))
        self._warmup_bars: int = int(bt_cfg.get("warmup_bars", 200))

        self._rebalance_interval: int = int(strat_cfg.get("rebalance_interval", 5))
        self._stop_loss_pct: float = float(strat_cfg.get("stop_loss_pct", 0.05))
        self._take_profit_pct: float = float(strat_cfg.get("take_profit_pct", 0.15))
        self._flatten_eod: bool = bool(strat_cfg.get("flatten_eod", False))

    # ------------------------------------------------------------------
    # Interval helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bars_per_day(interval: str) -> int:
        """Estimate the number of intraday bars in a 6.5-hour trading day.

        Returns 1 for daily-or-above intervals (used for annualization).
        """
        interval = interval.lower().strip()
        # Map interval string to minutes per bar
        _interval_minutes: dict[str, int] = {
            "1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30,
            "60m": 60, "90m": 90, "1h": 60,
        }
        minutes = _interval_minutes.get(interval)
        if minutes is None:
            return 1  # daily, weekly, monthly
        # US market: 6.5 hours = 390 minutes
        return max(1, 390 // minutes)

    @staticmethod
    def _is_intraday(interval: str) -> bool:
        """Return True if the interval is intraday (sub-daily)."""
        return interval.lower().strip() in {
            "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        ticker: str,
        period: str | None = "2y",
        interval: str = "1d",
        start: str | None = None,
        end: str | None = None,
    ) -> BacktestResult:
        """Run the backtest and return results with performance metrics.

        Args:
            ticker:   Stock symbol.
            period:   Historical period (e.g. "2y", "5y"). Ignored when *start* is set.
            interval: Bar interval (e.g. "1d").
            start:    Start date in YYYY-MM-DD format. Overrides *period*.
            end:      End date in YYYY-MM-DD format (default: today).

        Returns:
            :class:`BacktestResult` with trades, equity curve, and metrics.
        """
        # 1. Fetch all data
        df = self._provider.fetch(ticker, period=period, interval=interval, start=start, end=end)

        # Build a display-friendly period label
        if start:
            period_label = f"{start} → {end or 'today'}"
        else:
            period_label = period or "2y"

        if len(df) < self._warmup_bars + 10:
            raise ValueError(
                f"Not enough data for backtest. Got {len(df)} bars, "
                f"need at least {self._warmup_bars + 10} "
                f"(warmup={self._warmup_bars})."
            )

        # 2. State initialization
        cash: float = self._initial_cash
        position: _Position | None = None
        trades: list[BacktestTrade] = []
        equity_curve: list[dict[str, Any]] = []

        # Last computed scores (reused between rebalance bars)
        last_scores: dict[str, float] = {}
        last_overall: float = 5.0
        last_pattern_overall: float = 5.0

        # Notify strategy of start
        self._strategy.on_start({"ticker": ticker, "period": period})

        # 3. Bar-by-bar simulation
        bars_since_rebalance = self._rebalance_interval  # force recompute on first eligible bar
        intraday = self._is_intraday(interval)

        for i in range(len(df)):
            row = df.iloc[i]
            date_str = str(df.index[i])[:10]
            bar_timestamp = str(df.index[i])  # full timestamp for display
            close = float(row["close"])
            bar_dict = {
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": close,
                "volume": float(row["volume"]),
            }

            # Track position bars
            if position is not None:
                position.bars_held += 1

            # -- Skip warmup period --
            if i < self._warmup_bars:
                equity = cash + (position.unrealized_pnl(close) if position else 0.0)
                equity_curve.append({"date": date_str, "equity": equity})
                continue

            # -- Check stop-loss / take-profit on every bar --
            if position is not None:
                exit_reason = self._check_exit_triggers(position, close)
                if exit_reason:
                    trade = self._close_position(
                        position, close, date_str, exit_reason
                    )
                    cash += self._trade_proceeds(trade, position)
                    trades.append(trade)
                    position = None

            # -- Determine if this is the last bar of the trading day (for EOD flattening) --
            eod_bar = False
            if self._flatten_eod and intraday:
                eod_bar = (
                    i == len(df) - 1
                    or str(df.index[i + 1])[:10] != date_str
                )

            # -- Rebalance check: recompute indicators + get signal --
            bars_since_rebalance += 1
            if bars_since_rebalance >= self._rebalance_interval:
                bars_since_rebalance = 0

                # Trailing data up to current bar (inclusive) — no lookahead
                trailing_df = df.iloc[: i + 1].copy()

                last_scores, last_overall, last_pattern_overall = self._compute_scores(trailing_df)

            # Build context and ask strategy for order
            portfolio_value = cash
            if position is not None:
                portfolio_value += position.unrealized_pnl(close) + (
                    position.entry_price * position.quantity
                    if position.side == "long"
                    else 0.0
                )

            ctx = StrategyContext(
                bar=bar_dict,
                indicators={},       # raw values not needed for score strategy
                scores=last_scores,
                overall_score=last_overall,
                pattern_score=last_pattern_overall,
                position=position.quantity * (1 if position.side == "long" else -1)
                if position
                else 0.0,
                cash=cash,
                portfolio_value=portfolio_value,
            )

            order = self._strategy.on_bar(ctx)

            # -- Execute order (respecting trading mode) --
            # On EOD bars with flatten_eod, skip opening new positions
            can_short = self._trading_mode == TradingMode.LONG_SHORT
            can_trade = self._trading_mode != TradingMode.HOLD_ONLY
            signal_reason = f"signal: {order.notes}" if order.notes else "signal"

            if eod_bar:
                # EOD: only allow closing existing positions, not opening new ones
                if can_trade and position is not None:
                    if order.signal == Signal.BUY and position.side == "short":
                        # Close short (but don't open long)
                        trade = self._close_position(position, close, date_str, signal_reason)
                        cash += self._trade_proceeds(trade, position)
                        trades.append(trade)
                        position = None
                    elif order.signal == Signal.SELL and position.side == "long":
                        # Close long (but don't open short)
                        trade = self._close_position(position, close, date_str, signal_reason)
                        cash += self._trade_proceeds(trade, position)
                        trades.append(trade)
                        position = None
            elif not can_trade:
                pass  # hold_only: do nothing
            elif order.signal == Signal.BUY and position is None:
                position, cost = self._open_position(
                    "long", close, order.quantity, date_str, order.notes
                )
                cash -= cost
            elif order.signal == Signal.SELL and position is None and can_short:
                position, cost = self._open_position(
                    "short", close, order.quantity, date_str, order.notes
                )
                cash -= cost  # cost = commission only for short opens
            elif order.signal == Signal.BUY and position is not None and position.side == "short":
                # Close short, open long
                trade = self._close_position(position, close, date_str, signal_reason)
                cash += self._trade_proceeds(trade, position)
                trades.append(trade)
                position, cost = self._open_position(
                    "long", close, order.quantity, date_str, order.notes
                )
                cash -= cost
            elif order.signal == Signal.SELL and position is not None and position.side == "long":
                # Close long
                trade = self._close_position(position, close, date_str, signal_reason)
                cash += self._trade_proceeds(trade, position)
                trades.append(trade)
                position = None
                # Open short only if trading mode allows
                if can_short:
                    position, cost = self._open_position(
                        "short", close, order.quantity, date_str, order.notes
                    )
                    cash -= cost

            # -- EOD flattening: force-close any remaining position at end of day --
            if eod_bar and position is not None:
                trade = self._close_position(
                    position, close, date_str, "eod_flatten"
                )
                cash += self._trade_proceeds(trade, position)
                trades.append(trade)
                position = None

            # Record equity
            equity = cash
            if position is not None:
                if position.side == "long":
                    equity += position.quantity * close
                else:
                    # For short: cash already includes proceeds from short sale;
                    # we need to subtract cost to cover
                    equity += position.unrealized_pnl(close)
            equity_curve.append({"date": date_str, "equity": equity})

        # 4. Close any remaining position at last bar
        if position is not None:
            last_close = float(df["close"].iloc[-1])
            last_date = str(df.index[-1])[:10]
            trade = self._close_position(position, last_close, last_date, "end_of_data")
            cash += self._trade_proceeds(trade, position)
            trades.append(trade)
            position = None

        final_equity = cash

        # 5. Extract significant patterns from the full post-warmup data
        post_warmup_df = df.iloc[self._warmup_bars:].copy()
        significant_patterns = self._extract_significant_patterns(post_warmup_df)

        # 6. Build result
        result = BacktestResult(
            ticker=ticker.upper(),
            period=period_label,
            strategy_name=self._strategy.name,
            initial_cash=self._initial_cash,
            final_equity=final_equity,
            trades=trades,
            equity_curve=equity_curve,
            significant_patterns=significant_patterns,
        )
        self._compute_metrics(result, interval)
        self._strategy.on_end({"total_trades": len(trades)})
        return result

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _open_position(
        self,
        side: str,
        price: float,
        quantity: float,
        date: str,
        entry_reason: str = "",
    ) -> tuple[_Position, float]:
        """Open a new position. Returns (position, cash_cost).

        For long:  cash_cost = quantity * fill_price + commission
        For short: cash_cost = commission only (proceeds credited on close)
        """
        fill_price = self._apply_slippage(price, side, opening=True)
        pos = _Position(
            side=side,
            entry_date=date,
            entry_price=fill_price,
            quantity=quantity,
            entry_reason=entry_reason,
        )
        if side == "long":
            cost = fill_price * quantity + self._commission
        else:
            # Short: we receive proceeds on close; just pay commission now
            cost = self._commission
        return pos, cost

    def _close_position(
        self,
        position: _Position,
        price: float,
        date: str,
        reason: str,
    ) -> BacktestTrade:
        """Close *position* and return a BacktestTrade record."""
        fill_price = self._apply_slippage(price, position.side, opening=False)

        if position.side == "long":
            pnl = (fill_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - fill_price) * position.quantity

        pnl -= self._commission  # commission on close

        entry_cost = position.entry_price * position.quantity
        pnl_pct = pnl / entry_cost if entry_cost != 0 else 0.0

        return BacktestTrade(
            entry_date=position.entry_date,
            exit_date=date,
            entry_price=position.entry_price,
            exit_price=fill_price,
            quantity=position.quantity,
            side=position.side,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            entry_reason=position.entry_reason,
        )

    def _trade_proceeds(self, trade: BacktestTrade, position: _Position) -> float:
        """Cash returned when closing a position.

        Long close:  quantity * exit_price - commission (already in pnl)
        Short close: entry_price*qty - exit_price*qty + entry_price*qty
                   = pnl + entry_cost
        """
        if trade.side == "long":
            return trade.exit_price * trade.quantity - self._commission
        else:
            # Short: on close we return proceeds minus cost-to-cover
            return trade.pnl + self._commission  # net pnl (commission already deducted in trade)

    def _apply_slippage(self, price: float, side: str, opening: bool) -> float:
        """Adjust fill price for slippage.

        Opening long / closing short → price moves up (unfavourable).
        Opening short / closing long → price moves down (unfavourable).
        """
        if (side == "long" and opening) or (side == "short" and not opening):
            return price * (1 + self._slippage_pct)
        return price * (1 - self._slippage_pct)

    def _check_exit_triggers(self, position: _Position, current_price: float) -> str:
        """Check stop-loss and take-profit. Returns exit reason or empty string."""
        pnl_pct = position.unrealized_pnl_pct(current_price)

        if pnl_pct <= -self._stop_loss_pct:
            return "stop_loss"
        if pnl_pct >= self._take_profit_pct:
            return "take_profit"
        return ""

    # ------------------------------------------------------------------
    # Indicator recomputation
    # ------------------------------------------------------------------

    def _compute_scores(
        self, trailing_df: pd.DataFrame
    ) -> tuple[dict[str, float], float, float]:
        """Run all indicators and patterns on *trailing_df*.

        Returns:
            (per_indicator_scores, indicator_composite, pattern_composite)
        """
        # Indicator scores
        registry = IndicatorRegistry(self._cfg)
        results = registry.run_all(trailing_df)

        scorer = CompositeScorer(self._cfg)
        composite = scorer.score(results)

        scores = {r.config_key: r.score for r in results if not r.error}

        # Pattern scores
        pat_registry = PatternRegistry(self._cfg)
        pat_results = pat_registry.run_all(trailing_df)

        pat_scorer = PatternCompositeScorer(self._cfg)
        pat_composite = pat_scorer.score(pat_results)

        return scores, composite["overall"], pat_composite["overall"]

    # ------------------------------------------------------------------
    # Significant patterns extraction
    # ------------------------------------------------------------------

    def _extract_significant_patterns(
        self, df: pd.DataFrame
    ) -> list[SignificantPattern]:
        """Run all pattern detectors on the full data and extract individual events.

        Each detector stores its raw pattern list in ``values``.  This method
        normalises them into a flat :class:`SignificantPattern` list sorted by
        date, filtering out neutral / low-strength entries.

        The minimum strength threshold is configurable via
        ``config.yaml → backtest → significant_pattern_min_strength`` (default 0.5).
        """
        bt_cfg = self._cfg.section("backtest")
        min_strength = float(bt_cfg.get("significant_pattern_min_strength", 0.5))

        pat_registry = PatternRegistry(self._cfg)
        pat_results = pat_registry.run_all(df)

        events: list[SignificantPattern] = []

        for pr in pat_results:
            if pr.error:
                continue

            # ── Candlesticks & Inside/Outside ───────────────────────────
            # values["patterns"]: list of {bar_index, date, pattern, signal, strength}
            if "patterns" in pr.values:
                for p in pr.values["patterns"]:
                    if p.get("signal", "neutral") == "neutral":
                        continue
                    strength = float(p.get("strength", 0.0))
                    if strength < min_strength:
                        continue
                    events.append(SignificantPattern(
                        date=p["date"],
                        detector=pr.name,
                        pattern=_format_pattern_name(p["pattern"]),
                        signal=p["signal"],
                        strength=strength,
                        confidence=_strength_label(strength, pr.name),
                    ))

            # ── Gaps ────────────────────────────────────────────────────
            # values["gaps"]: list of {bar_index, date, direction, gap_pct, gap_type, volume_surge}
            if "gaps" in pr.values:
                for g in pr.values["gaps"]:
                    direction = g["direction"]
                    gap_type = g["gap_type"]
                    gap_pct = g["gap_pct"]

                    # Map gap direction + type to signal
                    if gap_type == "exhaustion":
                        # Exhaustion gaps signal reversal
                        signal = "bearish" if direction == "up" else "bullish"
                    else:
                        signal = "bullish" if direction == "up" else "bearish"

                    # Use gap_pct as a proxy for strength (typical gaps are 0.5%-3%)
                    strength = min(gap_pct * 100, 2.0)  # cap at 2.0
                    if strength < min_strength:
                        continue

                    vol_tag = " [VOL]" if g["volume_surge"] else ""
                    events.append(SignificantPattern(
                        date=g["date"],
                        detector=pr.name,
                        pattern=f"{gap_type.capitalize()} Gap {direction.upper()}",
                        signal=signal,
                        strength=strength,
                        confidence=_strength_label(strength, pr.name),
                        detail=f"{gap_pct:.1%}{vol_tag}",
                    ))

            # ── Spikes ──────────────────────────────────────────────────
            # values["spikes"]: list of {bar_index, date, direction, z_score, spike_level, confirmed}
            if "spikes" in pr.values:
                for s in pr.values["spikes"]:
                    direction = s["direction"]
                    confirmed = s["confirmed"]
                    z_score = abs(s["z_score"])

                    # Traps invert the signal
                    if confirmed is False:
                        signal = "bearish" if direction == "up" else "bullish"
                        status = "Trap"
                    elif confirmed is True:
                        signal = "bullish" if direction == "up" else "bearish"
                        status = "Confirmed"
                    else:
                        signal = "bullish" if direction == "up" else "bearish"
                        status = "Pending"

                    strength = min(z_score / 2.5, 2.0)  # normalise z-score
                    if strength < min_strength:
                        continue

                    events.append(SignificantPattern(
                        date=s["date"],
                        detector=pr.name,
                        pattern=f"Spike {direction.upper()}",
                        signal=signal,
                        strength=strength,
                        confidence=_strength_label(strength, pr.name),
                        detail=f"z={s['z_score']:.1f} {status}",
                    ))

            # ── Volume-Range ────────────────────────────────────────────
            # This detector only returns aggregate regime, not per-bar events.
            # Skip — no individual patterns to extract.

        # Sort by date
        events.sort(key=lambda e: e.date)
        return events

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self, result: BacktestResult, interval: str = "1d") -> None:
        """Populate performance metrics on *result* in-place.

        For intraday intervals, annualization uses bars_per_day * 252 instead
        of just 252 to correctly scale to yearly figures.
        """
        trades = result.trades
        result.total_trades = len(trades)

        # Bars per trading year for this interval
        bpd = self._bars_per_day(interval)
        bars_per_year = bpd * 252  # trading days per year

        # Total return
        if result.initial_cash > 0:
            result.total_return_pct = (
                (result.final_equity - result.initial_cash) / result.initial_cash * 100
            )
        else:
            result.total_return_pct = 0.0

        # Annualized return
        curve = result.equity_curve
        if len(curve) >= 2:
            n_bars = len(curve)
            years = n_bars / bars_per_year
            if years > 0 and result.final_equity > 0 and result.initial_cash > 0:
                result.annualized_return_pct = (
                    ((result.final_equity / result.initial_cash) ** (1 / years) - 1) * 100
                )

        # Max drawdown
        if curve:
            peak = curve[0]["equity"]
            max_dd = 0.0
            for pt in curve:
                eq = pt["equity"]
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak if peak > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd
            result.max_drawdown_pct = max_dd * 100

        # Win rate
        if trades:
            wins = [t for t in trades if t.pnl > 0]
            result.win_rate_pct = len(wins) / len(trades) * 100

            # Profit factor
            gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
            result.profit_factor = (
                gross_profit / gross_loss if gross_loss > 0 else float("inf")
            )

            # Average trade PnL %
            result.avg_trade_pnl_pct = (
                sum(t.pnl_pct for t in trades) / len(trades) * 100
            )
            result.best_trade_pnl_pct = max(t.pnl_pct for t in trades) * 100
            result.worst_trade_pnl_pct = min(t.pnl_pct for t in trades) * 100

        # Sharpe ratio (bar-level returns from equity curve)
        if len(curve) >= 2:
            equities = [pt["equity"] for pt in curve]
            bar_returns = []
            for j in range(1, len(equities)):
                if equities[j - 1] > 0:
                    bar_returns.append(equities[j] / equities[j - 1] - 1)
            if bar_returns:
                mean_r = sum(bar_returns) / len(bar_returns)
                var_r = sum((r - mean_r) ** 2 for r in bar_returns) / len(bar_returns)
                std_r = math.sqrt(var_r)
                if std_r > 0:
                    result.sharpe_ratio = (mean_r / std_r) * math.sqrt(bars_per_year)
