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

import numpy as np
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
class DCAContext:
    """DCA (Dollar-Cost Averaging) context for a single ticker.

    Combines price-dip detection with regime awareness, technical score
    integration, and volatility normalisation to produce a more nuanced
    DCA buy recommendation than raw dip percentage alone.
    """
    # --- Price dip ---
    dip_pct: float              # percentage drop from rolling high
    rolling_high: float         # the rolling high price used for dip detection
    raw_tier: str               # tier from price dip alone (before adjustments)

    # --- Volatility ---
    volatility: float           # annualised volatility (%)
    dip_sigma: float            # dip expressed in standard deviations

    # --- Score integration ---
    rsi: float                  # raw RSI value (0-100)
    bb_pctile: float            # Bollinger Band %B (0-100 scale)
    composite_score: float      # composite indicator score (0-10)

    # --- Regime ---
    regime: str                 # regime type used for adjustment
    regime_trend: str           # trend direction (bullish/bearish/neutral)

    # --- Final assessment ---
    tier: str                   # final tier after all adjustments
    multiplier: float           # final DCA allocation multiplier
    is_dca_buy: bool            # True if this is a good DCA buy opportunity
    confidence: str             # "high", "medium", "low"
    explanation: list[str]      # human-readable reasoning lines


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
    dca: DCAContext | None = None


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
    # Portfolio cash balance — tracks available cash after opening/closing positions.
    # Initialised from config (backtest.initial_cash) on first use; updated by
    # acknowledge_signal() when positions are opened or closed.
    cash_balance: float | None = None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        positions = {
            t: asdict(p) for t, p in self.positions.items()
        }
        closed = [asdict(ct) for ct in self.closed_trades]
        d: dict[str, Any] = {
            "positions": positions,
            "closed_trades": closed,
            "strategy_state": self.strategy_state,
            "last_updated": self.last_updated,
        }
        if self.cash_balance is not None:
            d["cash_balance"] = self.cash_balance
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WatchlistState":
        positions = {}
        for t, p in d.get("positions", {}).items():
            positions[t] = WatchlistPosition(**p)
        closed = [WatchlistClosedTrade(**ct) for ct in d.get("closed_trades", [])]
        cash = d.get("cash_balance")  # None if not present (legacy state files)
        return cls(
            positions=positions,
            closed_trades=closed,
            strategy_state=d.get("strategy_state", {}),
            last_updated=d.get("last_updated", ""),
            cash_balance=float(cash) if cash is not None else None,
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

    def ensure_cash(self, initial_cash: float) -> None:
        """Initialise cash_balance from config if not yet set.

        Called once when the monitor starts, using the configured initial
        cash value (e.g. ``backtest.initial_cash``).  If the state file
        already has a cash balance (from a previous session), it is
        preserved — we never overwrite a tracked balance with the default.
        """
        if self.cash_balance is None:
            self.cash_balance = initial_cash

    def portfolio_value(self, prices: dict[str, float]) -> float:
        """Compute total portfolio value: cash + market value of open positions.

        Args:
            prices: dict mapping ticker → current price.

        Returns:
            Total portfolio value (cash + sum of position market values).
        """
        cash = self.cash_balance if self.cash_balance is not None else 0.0
        market_value = 0.0
        for ticker, pos in self.positions.items():
            price = prices.get(ticker, pos.entry_price)
            market_value += pos.quantity * price
        return cash + market_value


# ---------------------------------------------------------------------------
# Standalone DCA context helpers (importable by scanner & watchlist)
# ---------------------------------------------------------------------------


def map_dca_regime(
    regime_type: RegimeType | None,
    regime_trend: str,
    total_return: float,
    wl_ctx: dict,
) -> str:
    """Map technical regime classification to investor-friendly DCA labels.

    Returns one of: ``"bull"``, ``"bear"``, ``"crisis"``, ``"sideways"``,
    ``"recovery"``.

    Parameters
    ----------
    regime_type : RegimeType | None
        Technical regime classification.
    regime_trend : str
        ``"bullish"``, ``"bearish"``, or ``"neutral"``.
    total_return : float
        Cumulative return over the classification window (e.g. 0.20 = +20%).
    wl_ctx : dict
        ``dca.watchlist_context`` config section (needs
        ``crisis_return_threshold``).
    """
    crisis_threshold = float(wl_ctx.get("crisis_return_threshold", -0.20))

    if regime_type is None:
        return "sideways"

    # Crisis: volatile/choppy + bearish + deeply negative return
    if (
        regime_type == RegimeType.VOLATILE_CHOPPY
        and regime_trend == "bearish"
        and total_return <= crisis_threshold
    ):
        return "crisis"

    # Bear: bearish trend with negative return
    if regime_trend == "bearish" and total_return < 0:
        return "bear"

    # Bull: bullish trend with positive return
    if regime_trend == "bullish" and total_return > 0:
        return "bull"

    # Recovery: breakout/transition or mean-reverting turning bullish
    if regime_type == RegimeType.BREAKOUT_TRANSITION and regime_trend == "bullish":
        return "recovery"
    if regime_type == RegimeType.MEAN_REVERTING and regime_trend == "bullish":
        return "recovery"

    # Sideways: everything else
    return "sideways"


def compute_dca_context(
    df: pd.DataFrame,
    *,
    ind_composite: float,
    ind_results: list,
    regime_type: RegimeType | None,
    regime_trend: str,
    regime_total_return: float,
    dca_cfg: dict,
) -> DCAContext:
    """Compute enhanced DCA context for a ticker.

    This is the standalone version of the DCA context computation.  It
    combines four signals to produce a nuanced DCA recommendation:

    1. **Price dip** — rolling high lookback → dip % → base tier.
    2. **Volatility normalisation** — express the dip in standard
       deviations.
    3. **Score integration** — composite score, RSI, and BB %B can
       upgrade the tier.
    4. **Regime adjustment** — investor-friendly regime labels with
       multiplier caps/boosts.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame (must have ``"close"`` column).
    ind_composite : float
        Composite indicator score (0-10).
    ind_results : list
        List of ``IndicatorResult`` objects from the analysis pipeline.
    regime_type : RegimeType | None
        Technical regime classification.
    regime_trend : str
        ``"bullish"``, ``"bearish"``, or ``"neutral"``.
    regime_total_return : float
        Cumulative return over the regime classification window.
    dca_cfg : dict
        The full ``dca`` config section (``cfg.section("dca")`` as a dict).
    """
    notes: list[str] = []

    # ── 1. Price dip detection ───────────────────────────────────────
    dt = dca_cfg.get("dip_thresholds", {})
    lookback_days = int(dt.get("lookback_days", 30))
    mild_drop_pct = float(dt.get("mild_drop_pct", 5.0))
    strong_drop_pct = float(dt.get("strong_drop_pct", 10.0))
    extreme_drop_pct = float(dt.get("extreme_drop_pct", 20.0))

    ml = dca_cfg.get("multipliers", {})
    mult_normal = float(ml.get("normal", 1.0))
    mult_mild = float(ml.get("mild_dip", 1.5))
    mult_strong = float(ml.get("strong_dip", 2.0))
    mult_extreme = float(ml.get("extreme_dip", 3.0))

    safety = dca_cfg.get("safety", {})
    max_multiplier = float(safety.get("max_multiplier", 3.0))

    close = df["close"].values
    current_price = float(close[-1])
    rolling_high_val = float(
        pd.Series(close).rolling(window=lookback_days, min_periods=1).max().iloc[-1]
    )

    if rolling_high_val > 0:
        dip_pct = max(0.0, (rolling_high_val - current_price) / rolling_high_val * 100.0)
    else:
        dip_pct = 0.0

    # Base tier from price dip
    tier_order = [
        ("extreme_dip", extreme_drop_pct, mult_extreme),
        ("strong_dip",  strong_drop_pct,  mult_strong),
        ("mild_dip",    mild_drop_pct,    mult_mild),
    ]
    tier, multiplier = "normal", mult_normal
    for t_name, t_threshold, t_mult in tier_order:
        if dip_pct >= t_threshold:
            tier, multiplier = t_name, t_mult
            break

    raw_tier = tier  # save pre-adjustment tier

    if dip_pct < mild_drop_pct:
        notes.append(f"Price is {dip_pct:.1f}% below {lookback_days}-day high — no significant dip.")
    else:
        notes.append(
            f"Price is {dip_pct:.1f}% below {lookback_days}-day high "
            f"(${rolling_high_val:,.2f}) — {tier.replace('_', ' ')}."
        )

    # ── 2. Volatility normalisation ──────────────────────────────────
    wl_ctx = dca_cfg.get("watchlist_context", {})
    vol_window = int(wl_ctx.get("volatility_window", 60))

    returns = pd.Series(close).pct_change().dropna()
    if len(returns) >= 20:
        daily_vol = float(returns.iloc[-vol_window:].std()) if len(returns) >= vol_window else float(returns.std())
        annual_vol = daily_vol * (252 ** 0.5) * 100  # annualised %
        daily_vol_pct = daily_vol * 100
        dip_sigma = (dip_pct / daily_vol_pct) if daily_vol_pct > 0 else 0.0
    else:
        annual_vol = 0.0
        dip_sigma = 0.0

    # Interpret volatility context
    vol_severe_sigma = float(wl_ctx.get("vol_severe_sigma", 2.5))
    vol_notable_sigma = float(wl_ctx.get("vol_notable_sigma", 1.5))

    if annual_vol > 0:
        if dip_sigma >= vol_severe_sigma:
            notes.append(
                f"Dip is {dip_sigma:.1f} daily std devs — statistically severe "
                f"for this stock (annualised vol {annual_vol:.0f}%)."
            )
        elif dip_sigma >= vol_notable_sigma:
            notes.append(
                f"Dip is {dip_sigma:.1f} daily std devs — notable "
                f"(annualised vol {annual_vol:.0f}%)."
            )
        elif dip_pct >= mild_drop_pct:
            notes.append(
                f"Dip is only {dip_sigma:.1f} daily std devs — routine "
                f"for this stock's volatility ({annual_vol:.0f}% annualised)."
            )

    # ── 3. Score integration ─────────────────────────────────────────
    # Extract raw RSI and BB %B from indicator results
    rsi_val = 50.0
    bb_pctile_val = 50.0
    for r in ind_results:
        if r.error:
            continue
        if r.config_key == "rsi":
            rsi_val = float(r.values.get("rsi", 50.0))
        elif r.config_key == "bollinger_bands":
            # pct_b is 0.0-1.0, convert to 0-100
            bb_pctile_val = float(r.values.get("pct_b", 0.5)) * 100

    st_cfg = dca_cfg.get("score_thresholds", {})
    buy_zone_below = float(st_cfg.get("buy_zone_below", 3.5))
    oversold_rsi = float(st_cfg.get("oversold_rsi", 30.0))
    bb_pctile_low = float(st_cfg.get("bb_percentile_low", 10.0))

    # Score-based tier upgrades (never downgrades, same as DCA backtester)
    score_upgraded = False
    if ind_composite < buy_zone_below:
        if tier == "normal":
            tier, multiplier = "mild_dip", mult_mild
            score_upgraded = True
        elif tier == "mild_dip":
            tier, multiplier = "strong_dip", mult_strong
            score_upgraded = True
        notes.append(
            f"Composite score {ind_composite:.1f} is below buy zone "
            f"({buy_zone_below}) — indicators suggest undervaluation."
        )
    else:
        notes.append(f"Composite score {ind_composite:.1f} — neutral to positive.")

    if rsi_val < oversold_rsi:
        if tier in ("normal", "mild_dip"):
            if tier == "normal":
                tier, multiplier = "mild_dip", mult_mild
            else:
                tier, multiplier = "strong_dip", mult_strong
            score_upgraded = True
        notes.append(f"RSI {rsi_val:.0f} is oversold (below {oversold_rsi:.0f}).")
    elif rsi_val > 70:
        notes.append(f"RSI {rsi_val:.0f} is overbought — caution on increasing allocation.")

    if bb_pctile_val < bb_pctile_low:
        if tier == "mild_dip":
            tier, multiplier = "strong_dip", mult_strong
            score_upgraded = True
        elif tier == "strong_dip":
            tier, multiplier = "extreme_dip", mult_extreme
            score_upgraded = True
        notes.append(
            f"BB %B at {bb_pctile_val:.0f}% — price near lower Bollinger Band."
        )

    if score_upgraded:
        notes.append(f"Tier upgraded from {raw_tier.replace('_',' ')} to {tier.replace('_',' ')} by technical signals.")

    # ── 4. Regime adjustment ─────────────────────────────────────────
    dca_regime = map_dca_regime(regime_type, regime_trend, regime_total_return, wl_ctx)
    regime_adj = wl_ctx.get("regime_adjustment", {})
    bear_max_mult = float(regime_adj.get("bear_max_multiplier", 1.5))
    bull_pullback_bonus = float(regime_adj.get("bull_pullback_bonus", 0.5))

    if dca_regime == "crisis":
        if multiplier > bear_max_mult:
            multiplier = bear_max_mult
        notes.append(
            f"Market regime: crisis — "
            f"multiplier capped at {bear_max_mult:.1f}x (high risk of continued decline)."
        )
    elif dca_regime == "bear":
        if multiplier > bear_max_mult:
            multiplier = bear_max_mult
        notes.append(
            f"Market regime: bear — "
            f"multiplier capped at {bear_max_mult:.1f}x (elevated downside risk)."
        )
    elif dca_regime == "bull" and dip_pct >= mild_drop_pct:
        multiplier = min(multiplier + bull_pullback_bonus, max_multiplier)
        notes.append(
            f"Market regime: bull pullback — good DCA entry point, "
            f"multiplier boosted by {bull_pullback_bonus:.1f}x."
        )
    elif dca_regime == "bull":
        notes.append("Market regime: bull — steady accumulation recommended.")
    elif dca_regime == "recovery":
        notes.append("Market regime: recovery — conditions improving, standard DCA allocation.")
    else:
        # sideways
        notes.append("Market regime: sideways — range-bound, standard DCA allocation.")

    # Clamp final multiplier
    multiplier = min(multiplier, max_multiplier)

    # ── 5. Confidence assessment ─────────────────────────────────────
    is_dca_buy = dip_pct >= mild_drop_pct or ind_composite < buy_zone_below
    if is_dca_buy and dca_regime in ("bull", "recovery") and rsi_val < 50:
        confidence = "high"
    elif is_dca_buy and dca_regime in ("bear", "crisis"):
        confidence = "low"
    elif is_dca_buy:
        confidence = "medium"
    else:
        confidence = "low"

    return DCAContext(
        dip_pct=round(dip_pct, 1),
        rolling_high=round(rolling_high_val, 2),
        raw_tier=raw_tier,
        volatility=round(annual_vol, 1),
        dip_sigma=round(dip_sigma, 1),
        rsi=round(rsi_val, 1),
        bb_pctile=round(bb_pctile_val, 1),
        composite_score=round(ind_composite, 1),
        regime=dca_regime,
        regime_trend=regime_trend,
        tier=tier,
        multiplier=round(multiplier, 2),
        is_dca_buy=is_dca_buy,
        confidence=confidence,
        explanation=notes,
    )


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

        # Parse tickers — supports both plain strings and dicts with overrides
        from config import parse_watchlist_tickers
        raw_tickers = wl_cfg.get("tickers", [])
        parsed = parse_watchlist_tickers(raw_tickers)

        self._tickers: list[str] = [str(e["ticker"]) for e in parsed]

        # Per-ticker overrides: {ticker: {regime_override, sub_type_override}}
        self._ticker_overrides: dict[str, dict[str, str | None]] = {
            str(e["ticker"]): {
                "regime_override": e.get("regime_override"),
                "sub_type_override": e.get("sub_type_override"),
            }
            for e in parsed
        }

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

        # Initialise cash balance from config if this is a fresh state file
        initial_cash = float(wl_cfg.get(
            "initial_cash",
            cfg.section("backtest").get("initial_cash", 100_000.0),
        ))
        self.state.ensure_cash(initial_cash)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def tickers(self) -> list[str]:
        return list(self._tickers)

    @tickers.setter
    def tickers(self, value: list[str]) -> None:
        self._tickers = [t.strip().upper() for t in value if t.strip()]

    @property
    def ticker_overrides(self) -> dict[str, dict[str, str | None]]:
        """Per-ticker regime/sub-type overrides (read-only copy)."""
        return dict(self._ticker_overrides)

    def get_ticker_override(self, ticker: str) -> dict[str, str | None]:
        """Return overrides for a specific ticker (empty dict if none)."""
        return self._ticker_overrides.get(ticker.upper(), {
            "regime_override": None,
            "sub_type_override": None,
        })

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

    def acknowledge_signal(
        self,
        ticker: str,
        signal: WatchlistSignal,
        *,
        override_price: float | None = None,
        override_quantity: float | None = None,
    ) -> None:
        """Update portfolio state after the user acts on a signal.

        Call this when the user confirms they executed the recommended trade.
        The user may optionally override the execution price and/or quantity
        (e.g. if the actual fill differed from the recommendation).

        Args:
            ticker: Stock ticker symbol.
            signal: The signal that was acted on.
            override_price: Actual execution price (defaults to signal's current_price).
            override_quantity: Actual share count (defaults to computed quantity).
        """
        ticker = ticker.upper()
        exec_price = override_price if override_price is not None else signal.current_price

        if signal.signal == Signal.BUY and signal.position is None:
            # Compute quantity from portfolio equity if not overridden
            if override_quantity is not None:
                qty = float(override_quantity)
            else:
                strat_cfg = self._cfg.section("strategy")
                sizing = strat_cfg.get("position_sizing", "fixed")
                if sizing == "fixed":
                    qty = float(strat_cfg.get("fixed_quantity", 100))
                else:
                    pct = float(strat_cfg.get("percent_equity", 1.0))
                    # Use real portfolio equity: cash + market value of positions
                    equity = self._compute_equity()
                    qty = int(pct * equity / exec_price) if exec_price > 0 else 0

            cost = qty * exec_price
            pos = WatchlistPosition(
                ticker=ticker,
                side="long",
                entry_date=datetime.now().strftime("%Y-%m-%d"),
                entry_price=exec_price,
                quantity=qty,
                entry_reason=signal.signal_notes,
            )
            self.state.positions[ticker] = pos

            # Deduct cost from cash balance
            if self.state.cash_balance is not None:
                self.state.cash_balance -= cost

        elif signal.signal == Signal.SELL and ticker in self.state.positions:
            # Close existing position
            pos = self.state.positions.pop(ticker)
            pnl_pct = pos.unrealized_pnl_pct(exec_price)
            closed = WatchlistClosedTrade(
                ticker=ticker,
                side=pos.side,
                entry_date=pos.entry_date,
                exit_date=datetime.now().strftime("%Y-%m-%d"),
                entry_price=pos.entry_price,
                exit_price=exec_price,
                quantity=override_quantity if override_quantity is not None else pos.quantity,
                pnl_pct=pnl_pct,
                exit_reason=signal.signal_notes,
            )
            self.state.closed_trades.append(closed)

            # Return proceeds to cash balance
            proceeds = closed.quantity * exec_price
            if self.state.cash_balance is not None:
                self.state.cash_balance += proceeds

            # Notify strategy state of trade close
            ss = self.state.strategy_state.get(ticker, {})
            losses = ss.get("consecutive_losses", 0)
            if pnl_pct < 0:
                ss["consecutive_losses"] = losses + 1
            else:
                ss["consecutive_losses"] = 0
            ss["bars_since_exit"] = 0
            self.state.strategy_state[ticker] = ss

    def _compute_equity(self) -> float:
        """Compute total portfolio equity (cash + estimated position value).

        Uses entry prices for position valuation since we may not have
        live prices at the time of the acknowledge call.  The scan()
        method provides accurate live-price equity via
        ``state.portfolio_value()``.
        """
        cash = self.state.cash_balance if self.state.cash_balance is not None else 0.0
        market_value = sum(
            pos.quantity * pos.entry_price
            for pos in self.state.positions.values()
        )
        return cash + market_value

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
            ind_scores, ind_composite, pat_composite, ind_results = (
                self._compute_scores(df)
            )

            # 3. Classify regime
            regime_type, regime_sub_type, regime_trend, regime_total_return = (
                self._classify_regime(df)
            )

            # 3b. Apply per-ticker regime/sub-type overrides if configured
            overrides = self._ticker_overrides.get(ticker, {})
            regime_override_str = overrides.get("regime_override")
            sub_type_override_str = overrides.get("sub_type_override")

            if regime_override_str:
                try:
                    regime_type = RegimeType(regime_override_str)
                    logger.info(
                        "%s: regime overridden to %s", ticker, regime_type.value,
                    )
                except ValueError:
                    logger.warning(
                        "%s: invalid regime_override '%s' — using auto-detected",
                        ticker, regime_override_str,
                    )
            if sub_type_override_str:
                try:
                    regime_sub_type = RegimeSubType(sub_type_override_str)
                    logger.info(
                        "%s: sub-type overridden to %s", ticker, regime_sub_type.value,
                    )
                except ValueError:
                    logger.warning(
                        "%s: invalid sub_type_override '%s' — using auto-detected",
                        ticker, sub_type_override_str,
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
                cash=self.state.cash_balance or 0.0,
                portfolio_value=self._compute_equity(),
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

            # 12. Compute DCA context (dip tier, multiplier, buy guidance)
            try:
                dca_ctx = self._compute_dca_context(
                    df,
                    ind_composite=ind_composite,
                    ind_results=ind_results,
                    regime_type=regime_type,
                    regime_trend=regime_trend,
                    regime_total_return=regime_total_return,
                )
            except Exception:
                logger.warning("DCA context failed for %s", ticker, exc_info=True)
                dca_ctx = None

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
                dca=dca_ctx,
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
    ) -> tuple[dict[str, float], float, float, list]:
        """Compute indicator and pattern scores — same as BacktestEngine.

        Returns (scores_dict, indicator_composite, pattern_composite,
        raw_indicator_results).
        """
        registry = IndicatorRegistry(self._cfg)
        results = registry.run_all(df)

        scorer = CompositeScorer(self._cfg)
        composite = scorer.score(results)
        scores = {r.config_key: r.score for r in results if not r.error}

        pat_registry = PatternRegistry(self._cfg)
        pat_results = pat_registry.run_all(df)

        pat_scorer = PatternCompositeScorer(self._cfg)
        pat_composite = pat_scorer.score(pat_results)

        return scores, composite["overall"], pat_composite["overall"], results

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
                scores, ind_comp, pat_comp, _ = self._compute_scores(trailing)
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

    def _compute_dca_context(
        self,
        df: pd.DataFrame,
        *,
        ind_composite: float,
        ind_results: list,
        regime_type: RegimeType | None,
        regime_trend: str,
        regime_total_return: float,
    ) -> DCAContext:
        """Delegate to the standalone ``compute_dca_context()`` function."""
        dca_cfg = dict(self._cfg.section("dca"))
        return compute_dca_context(
            df,
            ind_composite=ind_composite,
            ind_results=ind_results,
            regime_type=regime_type,
            regime_trend=regime_trend,
            regime_total_return=regime_total_return,
            dca_cfg=dca_cfg,
        )

    @staticmethod
    def _map_dca_regime(
        regime_type: RegimeType | None,
        regime_trend: str,
        total_return: float,
        wl_ctx: dict,
    ) -> str:
        """Delegate to the standalone ``map_dca_regime()`` function."""
        return map_dca_regime(regime_type, regime_trend, total_return, wl_ctx)

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
