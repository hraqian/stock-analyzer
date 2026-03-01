"""
engine/dca.py — Dollar Cost Averaging backtester.

Simulates periodic fixed-amount purchases of a single ticker over a
historical period.  Three modes are supported:

* **pure** — fixed dollar amount at each interval, no timing intelligence.
* **dip_weighted** — base amount scaled by a multiplier when the price has
  dropped a configurable percentage from its recent high.
* **score_integrated** — full integration with the composite scoring engine;
  multiplier is derived from the technical score, RSI, and Bollinger Band
  percentile, with safety gates (breakaway-gap skip, volume floor, max cap).

All thresholds, multipliers, and safety parameters are read from the ``dca``
section of *config.yaml* and can be overridden at runtime via dashboard
sliders.

Usage::

    from engine.dca import DCABacktester
    result = DCABacktester(cfg=cfg).run("AAPL", period="5y")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frequency helpers
# ---------------------------------------------------------------------------

_FREQ_TRADING_DAYS = {
    "daily": 1,
    "weekly": 5,
    "biweekly": 10,
    "monthly": 21,
}


def _buy_dates(index: pd.DatetimeIndex, frequency: str) -> list[int]:
    """Return positional indices in *index* where a DCA buy should occur.

    We space buys by approximately the target number of trading days.  The
    first buy is always on the first bar.
    """
    step = _FREQ_TRADING_DAYS.get(frequency, 21)
    positions: list[int] = []
    next_pos = 0
    for i in range(len(index)):
        if i >= next_pos:
            positions.append(i)
            next_pos = i + step
    return positions


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DCAPurchase:
    """Record of a single DCA buy."""
    date: str
    price: float
    amount: float           # dollars invested this period (before commission)
    commission: float       # commission paid on this purchase
    shares: float           # shares acquired ((amount - commission) / price)
    multiplier: float       # applied multiplier (1.0 for pure DCA)
    dip_pct: float          # drop % from recent high (0.0 if none)
    tier: str               # "normal", "mild_dip", "strong_dip", "extreme_dip"
    cumulative_shares: float
    cumulative_invested: float
    portfolio_value: float  # cumulative_shares * price


@dataclass
class DCAResult:
    """Aggregated results from a DCA backtest run."""
    ticker: str
    period: str
    mode: str               # "pure", "dip_weighted", "score_integrated"
    frequency: str

    # Capital
    total_invested: float = 0.0
    total_commissions: float = 0.0
    final_value: float = 0.0
    total_shares: float = 0.0

    # Dividends (DRIP)
    total_dividends: float = 0.0
    drip_shares: float = 0.0

    # Purchase log
    purchases: list[DCAPurchase] = field(default_factory=list)

    # Equity curve: list of {"date": str, "value": float, "invested": float}
    equity_curve: list[dict[str, Any]] = field(default_factory=list)

    # Dividend events: list of {"date": str, "amount": float, "shares_acquired": float}
    dividend_events: list[dict[str, Any]] = field(default_factory=list)

    # Performance metrics (populated after run)
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    avg_cost_basis: float = 0.0
    current_price: float = 0.0
    max_drawdown_pct: float = 0.0
    num_purchases: int = 0
    num_dip_purchases: int = 0      # purchases where multiplier > 1.0
    avg_multiplier: float = 1.0
    best_purchase_return_pct: float = 0.0
    worst_purchase_return_pct: float = 0.0

    # Budget mode fields
    budget_mode: bool = False
    total_budget: float = 0.0
    budget_remaining: float = 0.0
    computed_base_amount: float = 0.0  # base amount computed from budget
    reserve_method: str = ""


# ---------------------------------------------------------------------------
# DCA Backtester
# ---------------------------------------------------------------------------

class DCABacktester:
    """Dollar Cost Averaging backtester with dip-weighting and score
    integration.

    Parameters
    ----------
    cfg : Config
        Application config (``dca`` section is read for all parameters).
    overrides : dict | None
        Runtime overrides for DCA parameters (e.g. from dashboard sliders).
        These are merged on top of the config values.
    """

    def __init__(self, cfg: "Config", overrides: dict | None = None) -> None:
        self._cfg = cfg
        dca = cfg.section("dca")

        # Merge overrides
        ov = overrides or {}

        # Base parameters
        self.base_amount: float = float(ov.get("base_amount", dca.get("base_amount", 500)))
        self.frequency: str = str(ov.get("frequency", dca.get("frequency", "monthly")))
        self.drip: bool = bool(ov.get("drip", dca.get("drip", True)))
        self.mode: str = str(ov.get("mode", dca.get("mode", "dip_weighted")))

        # Dip thresholds
        dt = {**dca.get("dip_thresholds", {}), **ov.get("dip_thresholds", {})}
        self.mild_drop_pct: float = float(dt.get("mild_drop_pct", 5.0))
        self.strong_drop_pct: float = float(dt.get("strong_drop_pct", 10.0))
        self.extreme_drop_pct: float = float(dt.get("extreme_drop_pct", 20.0))
        self.lookback_days: int = int(dt.get("lookback_days", 30))

        # Multipliers
        ml = {**dca.get("multipliers", {}), **ov.get("multipliers", {})}
        self.mult_normal: float = float(ml.get("normal", 1.0))
        self.mult_mild: float = float(ml.get("mild_dip", 1.5))
        self.mult_strong: float = float(ml.get("strong_dip", 2.0))
        self.mult_extreme: float = float(ml.get("extreme_dip", 3.0))

        # Score thresholds (for score_integrated mode)
        st_cfg = {**dca.get("score_thresholds", {}), **ov.get("score_thresholds", {})}
        self.buy_zone_below: float = float(st_cfg.get("buy_zone_below", 3.5))
        self.oversold_rsi: float = float(st_cfg.get("oversold_rsi", 30.0))
        self.bb_pctile_low: float = float(st_cfg.get("bb_percentile_low", 10.0))

        # Safety gates
        sf = {**dca.get("safety", {}), **ov.get("safety", {})}
        self.max_multiplier: float = float(sf.get("max_multiplier", 3.0))
        self.max_period_alloc: float = float(sf.get("max_period_allocation", 1500))
        self.skip_breakaway: bool = bool(sf.get("skip_breakaway_gaps", True))
        self.min_vol_ratio: float = float(sf.get("min_volume_ratio", 0.5))

        # Crisis suppression gate
        cs_cfg = sf.get("crisis_suppression", {})
        cs_ov = ov.get("safety", {}).get("crisis_suppression", {})
        cs = {**cs_cfg, **cs_ov}
        self.crisis_enabled: bool = bool(cs.get("enabled", True))
        self.crisis_min_signals: int = int(cs.get("min_signals", 2))
        self.crisis_composite_below: float = float(cs.get("composite_below", 2.0))
        self.crisis_panic_rsi: float = float(cs.get("panic_rsi_below", 20.0))
        self.crisis_vol_spike: float = float(cs.get("volume_spike_above", 3.0))

        # Commission model — reuse the backtest commission config
        bt_cfg = cfg.section("backtest")
        self._commission_flat: float = float(bt_cfg.get("commission_per_trade", 0.0))
        self._commission_pct: float = float(bt_cfg.get("commission_pct", 0.0))
        self._commission_mode: str = str(bt_cfg.get("commission_mode", "additive"))

        # Budget mode
        bg = {**dca.get("budget", {}), **ov.get("budget", {})}
        self.budget_enabled: bool = bool(bg.get("enabled", False))
        self.total_budget: float = float(bg.get("total_budget", 50000))
        self.reserve_method: str = str(bg.get("reserve_method", "conservative"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        ticker: str,
        period: str = "5y",
        interval: str = "1d",
        start: str | None = None,
        end: str | None = None,
        *,
        score_df: pd.DataFrame | None = None,
    ) -> DCAResult:
        """Run a DCA backtest on *ticker*.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        period : str
            yfinance period string (default ``"5y"``).
        interval : str
            Bar interval (default ``"1d"``).
        start, end : str | None
            Optional date-range overrides.
        score_df : DataFrame | None
            Pre-computed score DataFrame with columns like ``composite``,
            ``rsi_raw``, ``bb_pctile``, ``gap_type``.  Required for
            ``score_integrated`` mode.  For other modes, ignored.

        Returns
        -------
        DCAResult
        """
        import yfinance as yf

        # ── Fetch OHLCV data ─────────────────────────────────────────────
        tk = yf.Ticker(ticker)
        if start:
            ohlcv = tk.history(start=start, end=end, interval=interval,
                               auto_adjust=False)
        else:
            ohlcv = tk.history(period=period, interval=interval,
                               auto_adjust=False)
        assert isinstance(ohlcv, pd.DataFrame), f"No data for {ticker}"

        # Normalise columns
        ohlcv.columns = [c.lower().replace(" ", "_") for c in ohlcv.columns]
        if hasattr(ohlcv.index, "tz") and ohlcv.index.tz is not None:
            ohlcv.index = ohlcv.index.tz_localize(None)
        ohlcv.sort_index(inplace=True)

        if len(ohlcv) < 2:
            raise ValueError(f"Insufficient data for {ticker}: {len(ohlcv)} bars")

        close = ohlcv["close"].values
        volume = ohlcv["volume"].values if "volume" in ohlcv.columns else None
        dates = ohlcv.index

        # ── Fetch dividends for DRIP ─────────────────────────────────────
        div_series: pd.Series | None = None
        if self.drip:
            try:
                raw_divs = tk.dividends
                if raw_divs is not None and len(raw_divs) > 0:
                    if hasattr(raw_divs.index, "tz") and raw_divs.index.tz is not None:
                        raw_divs.index = raw_divs.index.tz_localize(None)
                    # Reindex to OHLCV dates — forward-fill NaN as 0
                    div_series = raw_divs.reindex(dates, fill_value=0.0)
            except Exception:
                logger.debug("Could not fetch dividends for %s", ticker)

        # ── Compute rolling high for dip detection ───────────────────────
        rolling_high = pd.Series(close, index=dates).rolling(
            window=self.lookback_days, min_periods=1,
        ).max().values

        # ── Average volume for safety gate ───────────────────────────────
        avg_vol: np.ndarray | None = None
        if volume is not None:
            avg_vol = pd.Series(volume, index=dates).rolling(
                window=self.lookback_days, min_periods=1,
            ).mean().values

        # ── Determine buy dates ──────────────────────────────────────────
        buy_positions = _buy_dates(dates, self.frequency)

        # ── Budget mode: compute base amount from total budget ───────────
        effective_base = self.base_amount
        budget_remaining: float | None = None
        if self.budget_enabled:
            num_periods = len(buy_positions)
            effective_base = self._compute_budget_base_amount(
                num_periods, close, dates, rolling_high, buy_positions
            )
            budget_remaining = self.total_budget
            logger.info(
                "Budget mode (%s): $%.0f over %d periods → base $%.2f/period",
                self.reserve_method, self.total_budget, num_periods, effective_base,
            )

        # ── Simulate ─────────────────────────────────────────────────────
        result = DCAResult(
            ticker=ticker,
            period=period,
            mode=self.mode,
            frequency=self.frequency,
        )

        cum_shares: float = 0.0
        cum_invested: float = 0.0
        cum_commissions: float = 0.0
        cum_dividends: float = 0.0
        drip_shares: float = 0.0
        purchases: list[DCAPurchase] = []
        equity_curve: list[dict[str, Any]] = []
        dividend_events: list[dict[str, Any]] = []

        buy_set = set(buy_positions)

        for i in range(len(dates)):
            price = float(close[i])
            date_str = str(dates[i].date())

            # ── DRIP: reinvest dividends ─────────────────────────────────
            if div_series is not None:
                div_amount = float(div_series.iloc[i])
                if div_amount > 0 and cum_shares > 0:
                    div_cash = div_amount * cum_shares
                    drip_new_shares = div_cash / price
                    cum_shares += drip_new_shares
                    drip_shares += drip_new_shares
                    cum_dividends += div_cash
                    dividend_events.append({
                        "date": date_str,
                        "per_share": div_amount,
                        "total_amount": round(div_cash, 2),
                        "shares_acquired": round(drip_new_shares, 4),
                    })

            # ── DCA buy ──────────────────────────────────────────────────
            if i in buy_set:
                # Skip if budget is exhausted
                if budget_remaining is not None and budget_remaining <= 0:
                    pass  # no purchase — budget depleted
                else:
                    dip_pct = self._compute_dip_pct(price, float(rolling_high[i]))
                    multiplier, tier = self._resolve_multiplier(
                        dip_pct=dip_pct,
                        bar_idx=i,
                        score_df=score_df,
                        dates=dates,
                        volume=volume,
                        avg_vol=avg_vol,
                    )
                    amount = self._apply_safety(effective_base * multiplier)

                    # Budget cap: clamp to remaining budget
                    if budget_remaining is not None:
                        amount = min(amount, budget_remaining)

                    commission = self._calc_commission(amount)
                    net_amount = amount - commission  # dollars actually buying shares
                    shares = net_amount / price if net_amount > 0 else 0.0

                    cum_shares += shares
                    cum_invested += amount  # total outlay includes commission
                    cum_commissions += commission

                    if budget_remaining is not None:
                        budget_remaining -= amount

                    purchases.append(DCAPurchase(
                        date=date_str,
                        price=round(price, 4),
                        amount=round(amount, 2),
                        commission=round(commission, 2),
                        shares=round(shares, 4),
                        multiplier=round(multiplier, 2),
                        dip_pct=round(dip_pct, 2),
                        tier=tier,
                        cumulative_shares=round(cum_shares, 4),
                        cumulative_invested=round(cum_invested, 2),
                        portfolio_value=round(cum_shares * price, 2),
                    ))

            # ── Daily equity curve entry ─────────────────────────────────
            equity_curve.append({
                "date": date_str,
                "value": round(cum_shares * price, 2),
                "invested": round(cum_invested, 2),
                "price": round(price, 4),
            })

        # ── Populate result ──────────────────────────────────────────────
        result.purchases = purchases
        result.equity_curve = equity_curve
        result.dividend_events = dividend_events
        result.total_invested = round(cum_invested, 2)
        result.total_commissions = round(cum_commissions, 2)
        result.final_value = round(cum_shares * float(close[-1]), 2)
        result.total_shares = round(cum_shares, 4)
        result.total_dividends = round(cum_dividends, 2)
        result.drip_shares = round(drip_shares, 4)
        result.current_price = round(float(close[-1]), 4)
        result.num_purchases = len(purchases)

        # Budget mode metadata
        if self.budget_enabled:
            result.budget_mode = True
            result.total_budget = self.total_budget
            result.budget_remaining = round(budget_remaining or 0.0, 2)
            result.computed_base_amount = round(effective_base, 2)
            result.reserve_method = self.reserve_method

        self._compute_metrics(result, close, dates)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_dip_pct(price: float, rolling_high: float) -> float:
        """Percentage drop from rolling high (0.0 = at high, 15.0 = 15% below)."""
        if rolling_high <= 0:
            return 0.0
        return max(0.0, (rolling_high - price) / rolling_high * 100.0)

    def _resolve_multiplier(
        self,
        dip_pct: float,
        bar_idx: int,
        score_df: pd.DataFrame | None,
        dates: pd.DatetimeIndex,
        volume: np.ndarray | None,
        avg_vol: np.ndarray | None,
    ) -> tuple[float, str]:
        """Return (multiplier, tier_name) based on the current mode.

        Pure mode always returns (normal, "normal").
        Dip-weighted mode uses price dip percentage.
        Score-integrated mode also factors in technical score, RSI, and BB.
        """
        if self.mode == "pure":
            return self.mult_normal, "normal"

        # ── Dip-weighted tier assignment ─────────────────────────────────
        if dip_pct >= self.extreme_drop_pct:
            mult, tier = self.mult_extreme, "extreme_dip"
        elif dip_pct >= self.strong_drop_pct:
            mult, tier = self.mult_strong, "strong_dip"
        elif dip_pct >= self.mild_drop_pct:
            mult, tier = self.mult_mild, "mild_dip"
        else:
            mult, tier = self.mult_normal, "normal"

        # ── Crisis suppression gate ──────────────────────────────────────
        # When multiple bearish signals align simultaneously, suppress any
        # dip overweight back to 1.0x to avoid doubling down into a
        # structural collapse (fraud, terrible earnings, etc.).
        if (
            self.crisis_enabled
            and mult > self.mult_normal
            and score_df is not None
        ):
            date = dates[bar_idx]
            if date in score_df.index:
                row = score_df.loc[date]
                crisis_signals = 0

                # Signal 1: composite score extremely low
                composite_val = float(row.get("composite", 5.0))
                if composite_val < self.crisis_composite_below:
                    crisis_signals += 1

                # Signal 2: RSI in panic territory
                rsi_val = float(row.get("rsi_raw", 50.0))
                if rsi_val < self.crisis_panic_rsi:
                    crisis_signals += 1

                # Signal 3: volume spike (confirms the move has conviction)
                if volume is not None and avg_vol is not None:
                    cur_vol = float(volume[bar_idx])
                    avg = float(avg_vol[bar_idx])
                    if avg > 0 and (cur_vol / avg) > self.crisis_vol_spike:
                        crisis_signals += 1

                if crisis_signals >= self.crisis_min_signals:
                    mult, tier = self.mult_normal, "normal"

        # ── Score-integrated adjustments ─────────────────────────────────
        if self.mode == "score_integrated" and score_df is not None:
            date = dates[bar_idx]
            if date in score_df.index:
                row = score_df.loc[date]
                composite = float(row.get("composite", 5.0))
                rsi = float(row.get("rsi_raw", 50.0))
                bb_pctile = float(row.get("bb_pctile", 50.0))
                gap_type = str(row.get("gap_type", ""))

                # Breakaway-down safety gate: suppress overweight
                if self.skip_breakaway and gap_type == "breakaway":
                    direction = str(row.get("gap_direction", ""))
                    if direction == "down":
                        return self.mult_normal, "normal"

                # Score-based boost: bump up one tier if in buy zone
                if composite < self.buy_zone_below and tier == "normal":
                    mult, tier = self.mult_mild, "mild_dip"
                elif composite < self.buy_zone_below and tier == "mild_dip":
                    mult, tier = self.mult_strong, "strong_dip"

                # RSI oversold boost: bump up one more tier
                if rsi < self.oversold_rsi and tier in ("mild_dip", "normal"):
                    if tier == "normal":
                        mult, tier = self.mult_mild, "mild_dip"
                    else:
                        mult, tier = self.mult_strong, "strong_dip"

                # BB percentile boost
                if bb_pctile < self.bb_pctile_low and tier in ("mild_dip", "strong_dip"):
                    if tier == "mild_dip":
                        mult, tier = self.mult_strong, "strong_dip"
                    elif tier == "strong_dip":
                        mult, tier = self.mult_extreme, "extreme_dip"

        # ── Volume safety gate ───────────────────────────────────────────
        if volume is not None and avg_vol is not None and mult > self.mult_normal:
            cur_vol = float(volume[bar_idx])
            avg = float(avg_vol[bar_idx])
            if avg > 0 and (cur_vol / avg) < self.min_vol_ratio:
                # Low volume — don't trust the dip, revert to normal
                mult, tier = self.mult_normal, "normal"

        # Clamp to max_multiplier
        mult = min(mult, self.max_multiplier)
        return mult, tier

    def _apply_safety(self, raw_amount: float) -> float:
        """Enforce max period allocation cap."""
        return min(raw_amount, self.max_period_alloc)

    def _compute_budget_base_amount(
        self,
        num_periods: int,
        close: np.ndarray,
        dates: pd.DatetimeIndex,
        rolling_high: np.ndarray,
        buy_positions: list[int],
    ) -> float:
        """Compute the per-period base amount from the total budget.

        **Conservative**: assumes every period could trigger max multiplier,
        so ``base = budget / (num_periods × max_multiplier)``.

        **Adaptive**: scans the historical price data to estimate the
        frequency of dip tiers, computes an expected average multiplier,
        and sets ``base = budget / (num_periods × avg_expected_multiplier)``.
        """
        if num_periods <= 0:
            return self.base_amount  # fallback

        if self.reserve_method == "adaptive":
            return self._adaptive_base_amount(
                num_periods, close, rolling_high, buy_positions
            )

        # Conservative: worst-case — every period at max multiplier
        max_mult = self.max_multiplier
        if max_mult <= 0:
            max_mult = 1.0
        return self.total_budget / (num_periods * max_mult)

    def _adaptive_base_amount(
        self,
        num_periods: int,
        close: np.ndarray,
        rolling_high: np.ndarray,
        buy_positions: list[int],
    ) -> float:
        """Use historical dip frequency to compute expected avg multiplier.

        Walk through the buy dates, classify each into a dip tier, tally
        the frequencies, and compute the weighted average multiplier.
        Then ``base = budget / (num_periods × avg_multiplier)``.
        """
        tier_counts = {"normal": 0, "mild": 0, "strong": 0, "extreme": 0}

        for pos in buy_positions:
            price = float(close[pos])
            rh = float(rolling_high[pos])
            dip_pct = self._compute_dip_pct(price, rh)

            if dip_pct >= self.extreme_drop_pct:
                tier_counts["extreme"] += 1
            elif dip_pct >= self.strong_drop_pct:
                tier_counts["strong"] += 1
            elif dip_pct >= self.mild_drop_pct:
                tier_counts["mild"] += 1
            else:
                tier_counts["normal"] += 1

        total = sum(tier_counts.values())
        if total <= 0:
            return self.total_budget / num_periods

        # Weighted average multiplier based on historical tier frequencies
        avg_mult = (
            tier_counts["normal"] * self.mult_normal
            + tier_counts["mild"] * self.mult_mild
            + tier_counts["strong"] * self.mult_strong
            + tier_counts["extreme"] * self.mult_extreme
        ) / total

        if avg_mult <= 0:
            avg_mult = 1.0

        return self.total_budget / (num_periods * avg_mult)

    def _calc_commission(self, notional: float) -> float:
        """Compute commission for a single DCA purchase.

        Uses the same commission model as the active backtest engine:

        - **additive**: ``flat + pct × notional``
        - **max**: ``max(flat, pct × notional)``
        """
        flat = self._commission_flat
        pct = self._commission_pct * abs(notional)
        if self._commission_mode == "max":
            return max(flat, pct)
        return flat + pct

    def _compute_metrics(
        self,
        result: DCAResult,
        close: np.ndarray,
        dates: pd.DatetimeIndex,
    ) -> None:
        """Populate computed performance metrics on *result*."""
        if result.total_invested <= 0 or result.total_shares <= 0:
            return

        final_price = float(close[-1])

        # Average cost basis
        result.avg_cost_basis = round(result.total_invested / result.total_shares, 4)

        # Total return
        result.total_return_pct = round(
            (result.final_value - result.total_invested) / result.total_invested * 100, 2
        )

        # Annualized return — use actual calendar days
        if len(dates) >= 2:
            days = (dates[-1] - dates[0]).days
            if days > 0:
                years = days / 365.25
                growth = result.final_value / result.total_invested
                if growth > 0:
                    result.annualized_return_pct = round(
                        (growth ** (1.0 / years) - 1.0) * 100, 2
                    )

        # Max drawdown (on portfolio value curve)
        if result.equity_curve:
            values = [e["value"] for e in result.equity_curve]
            peak = values[0]
            max_dd = 0.0
            for v in values:
                if v > peak:
                    peak = v
                if peak > 0:
                    dd = (peak - v) / peak * 100
                    if dd > max_dd:
                        max_dd = dd
            result.max_drawdown_pct = round(max_dd, 2)

        # Dip purchase stats
        dip_purchases = [p for p in result.purchases if p.multiplier > self.mult_normal]
        result.num_dip_purchases = len(dip_purchases)
        if result.purchases:
            result.avg_multiplier = round(
                sum(p.multiplier for p in result.purchases) / len(result.purchases), 2
            )

        # Best / worst individual purchase return
        if result.purchases:
            returns = [
                (final_price - p.price) / p.price * 100
                for p in result.purchases if p.price > 0
            ]
            if returns:
                result.best_purchase_return_pct = round(max(returns), 2)
                result.worst_purchase_return_pct = round(min(returns), 2)
