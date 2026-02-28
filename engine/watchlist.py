"""
engine/watchlist.py — Live signal monitor for a user-defined watchlist.

Runs the same ScoreBasedStrategy logic used by the backtest engine against
the latest market data to produce actionable BUY / SELL / HOLD signals.

Portfolio state (open positions, closed trades, strategy internals) is
persisted to a JSON file between runs so the strategy correctly tracks
consecutive-loss cooldowns, re-entry grace periods, and percentile windows.

Usage:
    from engine.watchlist import WatchlistMonitor
    monitor = WatchlistMonitor(provider, cfg)
    signals = monitor.scan()         # returns list[WatchlistSignal]
    monitor.save_state()             # persist to disk
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from config import Config
    from data.provider import DataProvider

from indicators.registry import IndicatorRegistry
from analysis.scorer import CompositeScorer
from patterns.registry import PatternRegistry
from analysis.pattern_scorer import PatternCompositeScorer
from engine.strategy import Signal, StrategyContext
from engine.score_strategy import ScoreBasedStrategy
from engine.suitability import TradingMode
from engine.regime import RegimeClassifier, RegimeType, RegimeSubType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WatchlistPosition:
    """An open position tracked by the watchlist monitor."""
    ticker: str
    side: str                   # "long" or "short"
    entry_date: str             # YYYY-MM-DD
    entry_price: float
    quantity: float
    entry_reason: str = ""

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Return unrealised P&L as a percentage."""
        if self.entry_price == 0:
            return 0.0
        if self.side == "long":
            return (current_price - self.entry_price) / self.entry_price * 100
        return (self.entry_price - current_price) / self.entry_price * 100


@dataclass
class WatchlistClosedTrade:
    """A closed round-trip trade."""
    ticker: str
    side: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl_pct: float
    exit_reason: str = ""


@dataclass
class WatchlistSignal:
    """Signal output for a single ticker scan."""
    ticker: str
    signal: Signal
    action: str                 # human-readable action needed
    indicator_score: float
    pattern_score: float
    effective_score: float
    regime: str
    regime_sub_type: str
    position: WatchlistPosition | None
    current_price: float
    signal_notes: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

@dataclass
class WatchlistState:
    """Serialisable state for the watchlist monitor."""
    positions: dict[str, WatchlistPosition] = field(default_factory=dict)
    closed_trades: list[WatchlistClosedTrade] = field(default_factory=list)
    # Per-ticker strategy state for warm-restart
    strategy_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_updated: str = ""

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        positions = {
            t: asdict(p) for t, p in self.positions.items()
        }
        closed = [asdict(ct) for ct in self.closed_trades]
        return {
            "positions": positions,
            "closed_trades": closed,
            "strategy_state": self.strategy_state,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WatchlistState":
        positions = {}
        for t, p in d.get("positions", {}).items():
            positions[t] = WatchlistPosition(**p)
        closed = [WatchlistClosedTrade(**ct) for ct in d.get("closed_trades", [])]
        return cls(
            positions=positions,
            closed_trades=closed,
            strategy_state=d.get("strategy_state", {}),
            last_updated=d.get("last_updated", ""),
        )

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "WatchlistState":
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            with open(path) as fh:
                return cls.from_dict(json.load(fh))
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("Failed to load watchlist state from %s: %s", path, exc)
            return cls()


# ---------------------------------------------------------------------------
# Monitor engine
# ---------------------------------------------------------------------------

class WatchlistMonitor:
    """Scans a watchlist of tickers and produces live trading signals.

    Re-uses exactly the same indicator → score → strategy pipeline as
    :class:`BacktestEngine`, so the signals are consistent with what the
    backtest would have produced.
    """

    def __init__(
        self,
        provider: "DataProvider",
        cfg: "Config",
        *,
        state_path: str | Path | None = None,
    ) -> None:
        self._provider = provider
        self._cfg = cfg

        wl_cfg = cfg.section("watchlist")
        self._tickers: list[str] = [
            t.strip().upper() for t in wl_cfg.get("tickers", []) if t.strip()
        ]
        self._data_period: str = wl_cfg.get("data_period", "1y")
        self._interval: str = wl_cfg.get("interval", "1d")

        mode_str = wl_cfg.get("trading_mode", "long_only")
        self._trading_mode = (
            TradingMode.LONG_SHORT if mode_str == "long_short" else TradingMode.LONG_ONLY
        )

        # Resolve state file path
        if state_path is None:
            state_file = wl_cfg.get("state_file", "watchlist_state.json")
            state_path = Path(cfg.path).parent / state_file if cfg.path else Path(state_file)
        self._state_path = Path(state_path)

        # Load persisted state
        self.state = WatchlistState.load(self._state_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def tickers(self) -> list[str]:
        return list(self._tickers)

    @tickers.setter
    def tickers(self, value: list[str]) -> None:
        self._tickers = [t.strip().upper() for t in value if t.strip()]

    def scan(self, tickers: list[str] | None = None) -> list[WatchlistSignal]:
        """Scan all watchlist tickers and return current signals.

        Args:
            tickers: Override the configured ticker list for this scan.

        Returns:
            List of :class:`WatchlistSignal`, one per ticker.
        """
        scan_tickers = tickers or self._tickers
        if not scan_tickers:
            return []

        results: list[WatchlistSignal] = []
        for ticker in scan_tickers:
            results.append(self._scan_ticker(ticker))

        self.state.last_updated = datetime.now().isoformat(timespec="seconds")
        return results

    def save_state(self) -> None:
        """Persist current state to disk."""
        self.state.save(self._state_path)

    def acknowledge_signal(self, ticker: str, signal: WatchlistSignal) -> None:
        """Update portfolio state after the user acts on a signal.

        Call this when the user confirms they executed the recommended trade.
        """
        ticker = ticker.upper()

        if signal.signal == Signal.BUY and signal.position is None:
            # Open new long position
            strat_cfg = self._cfg.section("strategy")
            sizing = strat_cfg.get("position_sizing", "fixed")
            if sizing == "fixed":
                qty = float(strat_cfg.get("fixed_quantity", 100))
            else:
                pct = float(strat_cfg.get("percent_equity", 1.0))
                qty = int(pct * 100_000 / signal.current_price) if signal.current_price > 0 else 0
            pos = WatchlistPosition(
                ticker=ticker,
                side="long",
                entry_date=datetime.now().strftime("%Y-%m-%d"),
                entry_price=signal.current_price,
                quantity=qty,
                entry_reason=signal.signal_notes,
            )
            self.state.positions[ticker] = pos

        elif signal.signal == Signal.SELL and ticker in self.state.positions:
            # Close existing position
            pos = self.state.positions.pop(ticker)
            pnl_pct = pos.unrealized_pnl_pct(signal.current_price)
            closed = WatchlistClosedTrade(
                ticker=ticker,
                side=pos.side,
                entry_date=pos.entry_date,
                exit_date=datetime.now().strftime("%Y-%m-%d"),
                entry_price=pos.entry_price,
                exit_price=signal.current_price,
                quantity=pos.quantity,
                pnl_pct=pnl_pct,
                exit_reason=signal.signal_notes,
            )
            self.state.closed_trades.append(closed)

            # Notify strategy state of trade close
            ss = self.state.strategy_state.get(ticker, {})
            losses = ss.get("consecutive_losses", 0)
            if pnl_pct < 0:
                ss["consecutive_losses"] = losses + 1
            else:
                ss["consecutive_losses"] = 0
            ss["bars_since_exit"] = 0
            self.state.strategy_state[ticker] = ss

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _scan_ticker(self, ticker: str) -> WatchlistSignal:
        """Run the full signal pipeline on a single ticker."""
        try:
            # 1. Fetch data
            df = self._provider.fetch(
                ticker, period=self._data_period, interval=self._interval,
            )
            if df.empty or len(df) < 30:
                return WatchlistSignal(
                    ticker=ticker, signal=Signal.HOLD, action="Insufficient data",
                    indicator_score=5.0, pattern_score=5.0, effective_score=5.0,
                    regime="unknown", regime_sub_type="unknown",
                    position=self.state.positions.get(ticker),
                    current_price=0.0,
                    error=f"Only {len(df)} bars available",
                )

            current_price = float(df["close"].iloc[-1])

            # 2. Compute scores on full available data
            ind_scores, ind_composite, pat_composite = self._compute_scores(df)

            # 3. Classify regime
            regime_type, regime_sub_type, regime_trend, regime_total_return = (
                self._classify_regime(df)
            )

            # 4. Compute trend MA
            strat_cfg = self._cfg.section("strategy")
            trend_period = int(strat_cfg.get("trend_confirm_period", 20))
            ma_type = strat_cfg.get("trend_confirm_ma_type", "ema")
            if ma_type.lower() == "sma":
                trend_ma = float(
                    df["close"].rolling(window=trend_period, min_periods=1).mean().iloc[-1]
                )
            else:
                trend_ma = float(
                    df["close"].ewm(span=trend_period, adjust=False).mean().iloc[-1]
                )

            # 5. Compute average volume for metadata
            bt_adapt = strat_cfg.get("regime_adaptation", {})
            bt_breakout = bt_adapt.get("breakout_transition", {})
            avg_vol_window = int(bt_breakout.get("avg_volume_window", 20))
            avg_volume = float(
                df["volume"].rolling(window=avg_vol_window, min_periods=1).mean().iloc[-1]
            )

            # 6. Build strategy and seed with historical scores
            strategy = self._build_strategy(ticker, df, regime_type)

            # 7. Determine current position state
            pos = self.state.positions.get(ticker)
            position_qty = 0.0
            if pos is not None:
                position_qty = pos.quantity * (1 if pos.side == "long" else -1)

            # 8. Build context for latest bar
            bar_dict = {
                "open": float(df["open"].iloc[-1]),
                "high": float(df["high"].iloc[-1]),
                "low": float(df["low"].iloc[-1]),
                "close": current_price,
                "volume": float(df["volume"].iloc[-1]),
            }

            ctx = StrategyContext(
                bar=bar_dict,
                indicators={},
                scores=ind_scores,
                overall_score=ind_composite,
                pattern_score=pat_composite,
                position=position_qty,
                cash=100_000.0,       # nominal — not used for signal logic
                portfolio_value=100_000.0,
                trend_ma=trend_ma,
                regime=regime_type,
                regime_sub_type=regime_sub_type,
                regime_trend=regime_trend,
                regime_total_return=regime_total_return,
                metadata={"avg_volume": avg_volume},
            )

            # 9. Get signal
            order = strategy.on_bar(ctx)

            # 10. Compute effective score (same blend the strategy uses)
            effective_score = self._compute_effective_score(
                ind_composite, pat_composite, strat_cfg,
            )

            # 11. Determine action
            action = self._determine_action(order.signal, pos)

            return WatchlistSignal(
                ticker=ticker,
                signal=order.signal,
                action=action,
                indicator_score=ind_composite,
                pattern_score=pat_composite,
                effective_score=effective_score,
                regime=regime_type.value if regime_type else "unknown",
                regime_sub_type=regime_sub_type.value if regime_sub_type else "none",
                position=pos,
                current_price=current_price,
                signal_notes=order.notes,
            )

        except Exception as exc:
            logger.error("Failed to scan %s: %s", ticker, exc, exc_info=True)
            return WatchlistSignal(
                ticker=ticker, signal=Signal.HOLD, action="Error",
                indicator_score=5.0, pattern_score=5.0, effective_score=5.0,
                regime="unknown", regime_sub_type="unknown",
                position=self.state.positions.get(ticker),
                current_price=0.0,
                error=str(exc),
            )

    def _compute_scores(
        self, df: pd.DataFrame,
    ) -> tuple[dict[str, float], float, float]:
        """Compute indicator and pattern scores — same as BacktestEngine."""
        registry = IndicatorRegistry(self._cfg)
        results = registry.run_all(df)

        scorer = CompositeScorer(self._cfg)
        composite = scorer.score(results)
        scores = {r.config_key: r.score for r in results if not r.error}

        pat_registry = PatternRegistry(self._cfg)
        pat_results = pat_registry.run_all(df)

        pat_scorer = PatternCompositeScorer(self._cfg)
        pat_composite = pat_scorer.score(pat_results)

        return scores, composite["overall"], pat_composite["overall"]

    def _classify_regime(
        self, df: pd.DataFrame,
    ) -> tuple[RegimeType | None, RegimeSubType | None, str, float]:
        """Classify market regime on the given data."""
        try:
            classifier = RegimeClassifier(self._cfg)
            assessment = classifier.classify(df)
            return (
                assessment.regime,
                assessment.sub_type,
                assessment.metrics.trend_direction,
                assessment.metrics.total_return,
            )
        except Exception:
            logger.warning("Regime classification failed; using defaults", exc_info=True)
            return None, None, "neutral", 0.0

    def _build_strategy(
        self,
        ticker: str,
        df: pd.DataFrame,
        regime_type: RegimeType | None,
    ) -> ScoreBasedStrategy:
        """Instantiate and warm up a strategy for the given ticker."""
        strat_cfg = self._cfg.section("strategy")
        regime_adapt = self._cfg.section("regime").get("strategy_adaptation", {})

        strategy = ScoreBasedStrategy(
            params=strat_cfg,
            trading_mode=self._trading_mode,
            regime_adaptation=regime_adapt,
        )
        strategy.on_start({"ticker": ticker})

        # Restore persisted strategy state
        ss = self.state.strategy_state.get(ticker, {})
        if ss.get("score_window"):
            strategy.seed_score_window(ss["score_window"])
        if "bars_since_exit" in ss:
            strategy._bars_since_exit = ss["bars_since_exit"]
        if "consecutive_losses" in ss:
            strategy._consecutive_losses = ss["consecutive_losses"]

        # If no persisted score window, warm up by running the strategy
        # over recent historical bars to build the percentile window
        if not ss.get("score_window") and len(df) > 60:
            self._warmup_strategy(strategy, df)

        # Save updated strategy state
        self.state.strategy_state[ticker] = {
            "score_window": list(strategy._score_window),
            "bars_since_exit": strategy._bars_since_exit,
            "consecutive_losses": strategy._consecutive_losses,
        }

        return strategy

    def _warmup_strategy(
        self, strategy: ScoreBasedStrategy, df: pd.DataFrame
    ) -> None:
        """Run strategy over historical bars to build percentile window.

        We sample bars at rebalance_interval spacing to mirror how the
        backtest engine would have fed the strategy.
        """
        strat_cfg = self._cfg.section("strategy")
        rebalance_interval = int(strat_cfg.get("rebalance_interval", 5))

        # Use the last N bars for warmup (enough for percentile window)
        lookback = int(strat_cfg.get("percentile_thresholds", {}).get("lookback_bars", 60))
        warmup_bars = lookback * rebalance_interval
        # Don't use more than 80% of data for warmup — keep last portion for actual signal
        max_warmup = int(len(df) * 0.8)
        warmup_bars = min(warmup_bars, max_warmup)

        start_idx = max(30, len(df) - warmup_bars)  # need at least 30 bars for indicators

        trend_period = int(strat_cfg.get("trend_confirm_period", 20))
        ma_type = strat_cfg.get("trend_confirm_ma_type", "ema")
        if ma_type.lower() == "sma":
            trend_ma_series = df["close"].rolling(window=trend_period, min_periods=1).mean()
        else:
            trend_ma_series = df["close"].ewm(span=trend_period, adjust=False).mean()

        avg_vol_series = df["volume"].rolling(window=20, min_periods=1).mean()

        for i in range(start_idx, len(df) - 1, rebalance_interval):
            trailing = df.iloc[: i + 1]
            try:
                scores, ind_comp, pat_comp = self._compute_scores(trailing)
            except Exception:
                continue

            bar = df.iloc[i]
            ctx = StrategyContext(
                bar={
                    "open": float(bar["open"]),
                    "high": float(bar["high"]),
                    "low": float(bar["low"]),
                    "close": float(bar["close"]),
                    "volume": float(bar["volume"]),
                },
                indicators={},
                scores=scores,
                overall_score=ind_comp,
                pattern_score=pat_comp,
                position=0.0,
                cash=100_000.0,
                portfolio_value=100_000.0,
                trend_ma=float(trend_ma_series.iloc[i]),
                regime=None,
                regime_sub_type=None,
                regime_trend="neutral",
                regime_total_return=0.0,
                metadata={"avg_volume": float(avg_vol_series.iloc[i])},
            )
            strategy.on_bar(ctx)  # populates score_window internally

    def _compute_effective_score(
        self,
        ind_composite: float,
        pat_composite: float,
        strat_cfg: dict,
    ) -> float:
        """Blend indicator and pattern scores the same way the strategy does."""
        mode = strat_cfg.get("combination_mode", "weighted")

        if mode == "weighted":
            ind_w = float(strat_cfg.get("indicator_weight", 0.7))
            pat_w = float(strat_cfg.get("pattern_weight", 0.3))
            total_w = ind_w + pat_w
            if total_w > 0:
                return (ind_w * ind_composite + pat_w * pat_composite) / total_w
            return ind_composite
        elif mode == "boost":
            strength = float(strat_cfg.get("boost_strength", 0.5))
            dead_zone = float(strat_cfg.get("boost_dead_zone", 0.3))
            deviation = pat_composite - 5.0
            if abs(deviation) <= dead_zone:
                return ind_composite
            return max(0.0, min(10.0, ind_composite + deviation * strength))
        else:
            # gate mode — just return indicator composite
            return ind_composite

    def _determine_action(
        self, signal: Signal, position: WatchlistPosition | None,
    ) -> str:
        """Translate a signal + current position into a human-readable action."""
        if signal == Signal.BUY:
            if position is None:
                return "OPEN LONG"
            elif position.side == "short":
                return "CLOSE SHORT → OPEN LONG"
            else:
                return "Hold (already long)"
        elif signal == Signal.SELL:
            if position is not None and position.side == "long":
                return "CLOSE LONG"
            elif position is None:
                return "OPEN SHORT"
            else:
                return "Hold (already short)"
        else:
            if position is not None:
                return f"HOLD ({position.side})"
            return "No action"
