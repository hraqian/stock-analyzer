#!/usr/bin/env python3
"""Backtest sweep: compare sub-type-aware strategy params vs base-only params.

Runs all 18 tickers twice:
  1) "Before" — sub_types overrides removed (all strong_trend stocks use same base params)
  2) "After"  — sub_types overrides active (sub-type-specific trailing stop, position sizing)

Reports per-ticker and aggregate metrics, with special focus on strong_trend sub-types.
"""
from __future__ import annotations

import copy
import sys
import time

import pandas as pd

sys.path.insert(0, ".")

from config import Config, DEFAULT_CONFIG, _deep_merge
from data.provider import DataProvider
from data.yahoo import YahooFinanceProvider
from engine.backtest import BacktestEngine, BacktestResult
from engine.regime import RegimeSubType
from engine.score_strategy import ScoreBasedStrategy
from engine.suitability import TradingMode


# ─── Caching Provider (borrowed from optimize.py) ────────────────────────────
class CachingProvider(DataProvider):
    def __init__(self, real: DataProvider) -> None:
        self._real = real
        self._cache: dict[tuple, pd.DataFrame] = {}

    def fetch(self, ticker, period="6mo", interval="1d", start=None, end=None):
        key = (ticker, period, interval, start, end)
        if key in self._cache:
            return self._cache[key].copy()
        df = self._real.fetch(ticker, period=period, interval=interval, start=start, end=end)
        self._cache[key] = df.copy()
        return df

    def get_info(self, ticker):
        return self._real.get_info(ticker)


# ─── Ticker Universe ─────────────────────────────────────────────────────────
TICKERS = [
    # Strong trend — explosive movers (high vol + high momentum)
    "TSLA", "NVDA", "AAPL", "JPM", "XOM",
    # Strong trend — steady compounders (low vol + high momentum)
    "KO", "COST",
    # Non-strong-trend controls
    "AMD", "MARA",
]
PERIOD = "2y"


def run_sweep(provider: DataProvider, cfg: Config, label: str) -> list[dict]:
    """Run backtests for all tickers, return list of result dicts."""
    results = []
    regime_adapt = cfg.section("regime").get("strategy_adaptation", {})

    for ticker in TICKERS:
        try:
            strategy = ScoreBasedStrategy(
                params=cfg.section("strategy"),
                trading_mode=TradingMode.LONG_SHORT,
                regime_adaptation=regime_adapt,
            )
            engine = BacktestEngine(
                data_provider=provider,
                strategy=strategy,
                cfg=cfg,
                trading_mode=TradingMode.LONG_SHORT,
            )
            result = engine.run(ticker, period=PERIOD)

            regime_str = "unknown"
            sub_type_str = ""
            buyhold = 0.0
            atr_pct = 0.0

            if result.regime:
                regime_str = result.regime.regime.value
                buyhold = result.regime.metrics.total_return * 100
                atr_pct = result.regime.metrics.atr_pct
                if result.regime.sub_type:
                    sub_type_str = result.regime.sub_type_label

            strategy_ret = result.total_return_pct
            tracking_err = abs(strategy_ret - buyhold)

            results.append({
                "ticker": ticker,
                "regime": regime_str,
                "sub_type": sub_type_str,
                "atr_pct": round(atr_pct, 4),
                "buyhold": round(buyhold, 2),
                "strategy_return": round(strategy_ret, 2),
                "tracking_error": round(tracking_err, 2),
                "trades": result.total_trades,
                "max_drawdown": round(result.max_drawdown_pct, 2),
                "sharpe": round(result.sharpe_ratio, 2),
                "win_rate": round(result.win_rate_pct, 2),
                "label": label,
            })
            print(f"  {ticker:5s} | {regime_str:20s} | {sub_type_str:25s} | "
                  f"B&H={buyhold:+7.1f}% | Strat={strategy_ret:+7.1f}% | "
                  f"TrkErr={tracking_err:5.1f}% | Trades={result.total_trades}")
        except Exception as exc:
            print(f"  {ticker:5s} | ERROR: {exc}")
            results.append({
                "ticker": ticker, "regime": "error", "sub_type": "",
                "atr_pct": 0, "buyhold": 0, "strategy_return": 0,
                "tracking_error": 999, "trades": 0, "max_drawdown": 0,
                "sharpe": 0, "win_rate": 0, "label": label,
            })

    return results


def make_before_config() -> Config:
    """Create a config with sub_types overrides stripped (post-merge).

    We must strip sub_types AFTER deep-merging with DEFAULT_CONFIG, because
    DEFAULT_CONFIG also contains sub_types and would re-inject them.
    """
    cfg = Config.load()
    # Reach into the merged data and remove sub_types from all regimes
    adapt = cfg.section("regime").get("strategy_adaptation", {})
    for regime_key in list(adapt.keys()):
        if isinstance(adapt[regime_key], dict) and "sub_types" in adapt[regime_key]:
            del adapt[regime_key]["sub_types"]
    return cfg


def print_comparison(before: list[dict], after: list[dict]) -> None:
    """Print a side-by-side comparison table."""
    before_map = {r["ticker"]: r for r in before}
    after_map = {r["ticker"]: r for r in after}

    print("\n" + "=" * 120)
    print("COMPARISON: Before (no sub-type overrides) vs After (sub-type-aware)")
    print("=" * 120)
    print(f"{'Ticker':6s} {'Regime':20s} {'SubType':25s} {'B&H':>8s} "
          f"{'Before':>8s} {'After':>8s} {'Δ Return':>9s} "
          f"{'TrkErr B':>9s} {'TrkErr A':>9s} {'Δ TrkErr':>9s}")
    print("-" * 120)

    total_te_before = 0
    total_te_after = 0
    strong_trend_te_before = 0
    strong_trend_te_after = 0
    strong_trend_count = 0
    n = 0

    for ticker in TICKERS:
        b = before_map.get(ticker)
        a = after_map.get(ticker)
        if not b or not a or b["regime"] == "error" or a["regime"] == "error":
            continue

        delta_ret = a["strategy_return"] - b["strategy_return"]
        delta_te = a["tracking_error"] - b["tracking_error"]
        n += 1
        total_te_before += b["tracking_error"]
        total_te_after += a["tracking_error"]

        is_st = a["regime"] == "strong_trend"
        if is_st:
            strong_trend_te_before += b["tracking_error"]
            strong_trend_te_after += a["tracking_error"]
            strong_trend_count += 1

        marker = " *" if is_st else ""
        te_indicator = "↓" if delta_te < -0.5 else ("↑" if delta_te > 0.5 else "≈")

        print(f"{ticker:6s} {a['regime']:20s} {a['sub_type']:25s} {a['buyhold']:+7.1f}% "
              f"{b['strategy_return']:+7.1f}% {a['strategy_return']:+7.1f}% {delta_ret:+8.1f}% "
              f"{b['tracking_error']:8.1f}% {a['tracking_error']:8.1f}% {delta_te:+8.1f}% {te_indicator}{marker}")

    print("-" * 120)
    if n > 0:
        print(f"\n  ALL TICKERS ({n}):")
        print(f"    Avg Tracking Error  Before: {total_te_before / n:.2f}%   After: {total_te_after / n:.2f}%   "
              f"Δ: {(total_te_after - total_te_before) / n:+.2f}%")
    if strong_trend_count > 0:
        print(f"\n  STRONG TREND ONLY ({strong_trend_count}):")
        print(f"    Avg Tracking Error  Before: {strong_trend_te_before / strong_trend_count:.2f}%   "
              f"After: {strong_trend_te_after / strong_trend_count:.2f}%   "
              f"Δ: {(strong_trend_te_after - strong_trend_te_before) / strong_trend_count:+.2f}%")

    # Per-sub-type breakdown
    print(f"\n  PER SUB-TYPE BREAKDOWN (strong_trend only):")
    subtypes = {}
    for ticker in TICKERS:
        a = after_map.get(ticker)
        b = before_map.get(ticker)
        if not a or not b or a["regime"] != "strong_trend":
            continue
        st = a["sub_type"] or "(none)"
        if st not in subtypes:
            subtypes[st] = {"tickers": [], "te_before": 0, "te_after": 0,
                            "ret_before": 0, "ret_after": 0}
        subtypes[st]["tickers"].append(ticker)
        subtypes[st]["te_before"] += b["tracking_error"]
        subtypes[st]["te_after"] += a["tracking_error"]
        subtypes[st]["ret_before"] += b["strategy_return"]
        subtypes[st]["ret_after"] += a["strategy_return"]

    for st_name, st_data in sorted(subtypes.items()):
        cnt = len(st_data["tickers"])
        avg_te_b = st_data["te_before"] / cnt
        avg_te_a = st_data["te_after"] / cnt
        avg_ret_b = st_data["ret_before"] / cnt
        avg_ret_a = st_data["ret_after"] / cnt
        print(f"    {st_name:25s} ({', '.join(st_data['tickers'])})")
        print(f"      Avg Return:   Before={avg_ret_b:+.1f}%  After={avg_ret_a:+.1f}%  Δ={avg_ret_a - avg_ret_b:+.1f}%")
        print(f"      Avg TrkErr:   Before={avg_te_b:.1f}%  After={avg_te_a:.1f}%  Δ={avg_te_a - avg_te_b:+.1f}%")

    # Per-sub-type Sharpe comparison
    print(f"\n  SHARPE RATIO COMPARISON:")
    for ticker in TICKERS:
        b = before_map.get(ticker)
        a = after_map.get(ticker)
        if not b or not a or a["regime"] != "strong_trend":
            continue
        delta = a["sharpe"] - b["sharpe"]
        indicator = "↑" if delta > 0.05 else ("↓" if delta < -0.05 else "≈")
        print(f"    {ticker:6s} {a['sub_type']:25s}  Before={b['sharpe']:.2f}  After={a['sharpe']:.2f}  Δ={delta:+.2f} {indicator}")


def main():
    print("=" * 80)
    print("SUB-TYPE BACKTEST SWEEP — Before/After Comparison")
    print("=" * 80)

    # Create "before" config — same as current but with sub_types removed post-merge
    cfg_before = make_before_config()

    # "After" config is the real one (loaded fresh to avoid sharing mutation)
    cfg_after = Config.load()

    # Shared caching provider
    provider = CachingProvider(YahooFinanceProvider())

    # Prefetch all data
    print("\nPrefetching data for all tickers...")
    t0 = time.time()
    for ticker in TICKERS:
        try:
            provider.fetch(ticker, period=PERIOD)
            print(f"  ✓ {ticker}")
        except Exception as exc:
            print(f"  ✗ {ticker}: {exc}")
    print(f"Prefetch done in {time.time() - t0:.1f}s\n")

    # Run "Before" sweep
    print("─" * 80)
    print("BEFORE (no sub-type overrides — all strong_trend stocks use base params)")
    print("─" * 80)
    t1 = time.time()
    before_results = run_sweep(provider, cfg_before, "before")
    print(f"  [{time.time() - t1:.1f}s]\n")

    # Run "After" sweep
    print("─" * 80)
    print("AFTER (sub-type-aware — explosive mover / steady compounder / etc.)")
    print("─" * 80)
    t2 = time.time()
    after_results = run_sweep(provider, cfg_after, "after")
    print(f"  [{time.time() - t2:.1f}s]\n")

    # Comparison
    print_comparison(before_results, after_results)

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
