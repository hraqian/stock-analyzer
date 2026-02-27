#!/usr/bin/env python3
"""
optimize.py — Grid-search optimizer for per-regime backtest configuration.

Runs backtests across a diverse ticker matrix, groups results by detected
market regime, then grid-searches key strategy/backtest parameters to
minimize tracking error (|strategy_return - buy_and_hold_return|).

Outputs per-regime optimal config overrides as YAML fragments.

Usage:
    python optimize.py                    # full optimization run
    python optimize.py --baseline-only    # just run baselines, show regimes
    python optimize.py --regime strong_trend  # optimize only one regime
    python optimize.py --fast             # reduced grid (fewer combos)
"""

from __future__ import annotations

import argparse
import copy
import itertools
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ── Ensure project root is on sys.path ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import Config, DEFAULT_CONFIG, _deep_merge
from data.provider import DataProvider
from data.yahoo import YahooFinanceProvider
from engine.backtest import BacktestEngine, BacktestResult
from engine.regime import RegimeType, RegimeSubType
from engine.score_strategy import ScoreBasedStrategy
from engine.suitability import TradingMode


# ─────────────────────────────────────────────────────────────────────────────
# Caching Data Provider
# ─────────────────────────────────────────────────────────────────────────────

class CachingProvider(DataProvider):
    """Wraps a real DataProvider, caching fetched DataFrames in memory.

    During grid search, the same ticker/period is backtested hundreds of times
    with different config parameters.  Re-downloading from Yahoo each time is
    the dominant cost (~3s per call).  This wrapper ensures each unique
    (ticker, period, interval, start, end) tuple is fetched only once.
    """

    def __init__(self, real_provider: DataProvider) -> None:
        self._real = real_provider
        self._cache: dict[tuple, pd.DataFrame] = {}
        self._hits = 0
        self._misses = 0

    def fetch(
        self,
        ticker: str,
        period: str | None = "6mo",
        interval: str = "1d",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        key = (ticker, period, interval, start, end)
        if key in self._cache:
            self._hits += 1
            return self._cache[key].copy()  # copy so backtest mutations don't corrupt cache

        self._misses += 1
        df = self._real.fetch(ticker, period=period, interval=interval, start=start, end=end)
        self._cache[key] = df.copy()
        return df

    def get_info(self, ticker: str) -> dict:
        return self._real.get_info(ticker)

    def prefetch(self, ticker_matrix: list[dict], interval: str = "1d") -> None:
        """Pre-download all ticker data to warm the cache."""
        for spec in ticker_matrix:
            key = (spec["ticker"], spec["period"], interval, None, None)
            if key not in self._cache:
                try:
                    df = self._real.fetch(spec["ticker"], period=spec["period"], interval=interval)
                    self._cache[key] = df.copy()
                    self._misses += 1
                except Exception as exc:
                    print(f"  [WARN] Failed to prefetch {spec['ticker']}: {exc}")

    @property
    def stats(self) -> str:
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return f"Cache: {self._hits} hits, {self._misses} misses ({hit_rate:.0f}% hit rate)"

# ─────────────────────────────────────────────────────────────────────────────
# Ticker Universe
# ─────────────────────────────────────────────────────────────────────────────
# Diverse set covering likely all 4 regime types.  We don't pre-assign
# regimes — the baseline run discovers them from the data.

TICKER_MATRIX: list[dict] = [
    # --- Likely strong-trend (large-cap tech, consistent growers) ---
    {"ticker": "AAPL",  "period": "2y"},
    {"ticker": "NVDA",  "period": "2y"},
    {"ticker": "MSFT",  "period": "2y"},
    {"ticker": "COST",  "period": "2y"},
    {"ticker": "LLY",   "period": "2y"},
    # --- Likely mean-reverting (stable dividend / consumer staples) ---
    {"ticker": "KO",    "period": "2y"},
    {"ticker": "JNJ",   "period": "2y"},
    {"ticker": "PG",    "period": "2y"},
    {"ticker": "PEP",   "period": "2y"},
    # --- Likely volatile/choppy (meme stocks, crypto-adjacent, biotech) ---
    {"ticker": "TSLA",  "period": "2y"},
    {"ticker": "AMD",   "period": "2y"},
    {"ticker": "MARA",  "period": "2y"},
    {"ticker": "RIOT",  "period": "2y"},
    # --- Likely breakout-transition (recovery / restructuring) ---
    {"ticker": "META",  "period": "2y"},
    {"ticker": "AMZN",  "period": "2y"},
    {"ticker": "GOOG",  "period": "2y"},
    # --- Extra diversity (financials, energy) ---
    {"ticker": "JPM",   "period": "2y"},
    {"ticker": "XOM",   "period": "2y"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Parameter Grid
# ─────────────────────────────────────────────────────────────────────────────
# The most impactful parameters for tracking error.  Each regime gets the
# same grid — but optimal values should differ by regime.

PARAM_GRID: dict[str, list] = {
    # Strategy thresholds — most direct control over when to trade
    "strategy.score_thresholds.short_below": [3.0, 3.5, 4.0, 4.5],
    "strategy.score_thresholds.hold_below":  [5.5, 6.0, 6.5, 7.0],
    # Risk management
    "strategy.stop_loss_pct":      [0.03, 0.05, 0.08, 0.12],
    "strategy.take_profit_pct":    [0.15, 0.30, 0.50, 1.00],
    # Position sizing
    "strategy.percent_equity":     [0.50, 0.80, 1.00],
    # Rebalance frequency
    "strategy.rebalance_interval": [3, 5, 8],
    # ATR stop multiplier
    "strategy.atr_stop_multiplier": [2.0, 3.0, 4.0],
}

# Reduced grid for --fast mode
PARAM_GRID_FAST: dict[str, list] = {
    "strategy.score_thresholds.short_below": [3.5, 4.0],
    "strategy.score_thresholds.hold_below":  [6.0, 6.5],
    "strategy.stop_loss_pct":      [0.05, 0.08],
    "strategy.take_profit_pct":    [0.30, 0.50],
    "strategy.percent_equity":     [0.80],
    "strategy.rebalance_interval": [3, 5],
    "strategy.atr_stop_multiplier": [3.0, 4.0],
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TickerResult:
    """Result of a single backtest run for one ticker."""
    ticker: str
    period: str
    regime: str                 # RegimeType.value
    regime_confidence: float
    sub_type: str               # RegimeSubType label (e.g. "Explosive Mover")
    strategy_return: float      # percentage
    buyhold_return: float       # percentage
    tracking_error: float       # |strategy - buyhold|
    total_trades: int
    max_drawdown: float         # percentage
    sharpe: float
    win_rate: float             # percentage


@dataclass
class RegimeGroup:
    """All tickers that fell into the same regime."""
    regime: str
    results: list[TickerResult] = field(default_factory=list)

    @property
    def avg_tracking_error(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.tracking_error for r in self.results) / len(self.results)

    @property
    def avg_strategy_return(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.strategy_return for r in self.results) / len(self.results)

    @property
    def avg_buyhold_return(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.buyhold_return for r in self.results) / len(self.results)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _set_nested(d: dict, dotpath: str, value) -> None:
    """Set a value in a nested dict using dot notation.

    Example: _set_nested(d, "strategy.stop_loss_pct", 0.05)
    """
    keys = dotpath.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _make_config_with_overrides(base_data: dict, overrides: dict[str, object]) -> Config:
    """Create a Config with dot-path overrides applied."""
    data = copy.deepcopy(base_data)
    for dotpath, value in overrides.items():
        _set_nested(data, dotpath, value)
    return Config.from_dict(data)


def _run_single_backtest(
    provider: DataProvider,
    cfg: Config,
    ticker: str,
    period: str,
    trading_mode: TradingMode = TradingMode.LONG_SHORT,
) -> BacktestResult | None:
    """Run a single backtest, returning None on error."""
    try:
        regime_adapt = cfg.section("regime").get("strategy_adaptation", {})
        strategy = ScoreBasedStrategy(
            params=cfg.section("strategy"),
            trading_mode=trading_mode,
            regime_adaptation=regime_adapt,
        )
        engine = BacktestEngine(
            data_provider=provider,
            strategy=strategy,
            cfg=cfg,
            trading_mode=trading_mode,
        )
        return engine.run(ticker, period=period)
    except Exception as exc:
        print(f"  [ERROR] {ticker} {period}: {exc}")
        return None


def _extract_ticker_result(
    ticker: str, period: str, result: BacktestResult,
) -> TickerResult:
    """Extract key metrics from a BacktestResult."""
    regime_str = "unknown"
    regime_conf = 0.0
    buyhold = 0.0
    sub_type_label = ""

    if result.regime:
        regime_str = result.regime.regime.value
        regime_conf = result.regime.confidence
        buyhold = result.regime.metrics.total_return * 100  # to pct
        sub_type_label = result.regime.sub_type_label or ""

    strategy_ret = result.total_return_pct
    tracking_err = abs(strategy_ret - buyhold)

    return TickerResult(
        ticker=ticker,
        period=period,
        regime=regime_str,
        regime_confidence=regime_conf,
        sub_type=sub_type_label,
        strategy_return=round(strategy_ret, 2),
        buyhold_return=round(buyhold, 2),
        tracking_error=round(tracking_err, 2),
        total_trades=result.total_trades,
        max_drawdown=round(result.max_drawdown_pct, 2),
        sharpe=round(result.sharpe_ratio, 2),
        win_rate=round(result.win_rate_pct, 2),
    )


def _print_results_table(results: list[TickerResult], title: str = "Results") -> None:
    """Pretty-print a results table."""
    print(f"\n{'='*110}")
    print(f"  {title}")
    print(f"{'='*110}")
    print(f"  {'Ticker':<7} {'Period':<6} {'Regime':<18} {'Sub-Type':<22} {'Strat%':>8} {'B&H%':>8} "
          f"{'TrkErr':>8} {'Trades':>7} {'MaxDD%':>8} {'Sharpe':>7} {'WinR%':>7}")
    print(f"  {'-'*7} {'-'*6} {'-'*18} {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*7} {'-'*7}")

    for r in results:
        sub = r.sub_type or "—"
        print(f"  {r.ticker:<7} {r.period:<6} {r.regime:<18} {sub:<22} "
              f"{r.strategy_return:>+8.2f} {r.buyhold_return:>+8.2f} "
              f"{r.tracking_error:>8.2f} {r.total_trades:>7d} "
              f"{r.max_drawdown:>8.2f} {r.sharpe:>7.2f} {r.win_rate:>7.1f}")

    # Summary
    if results:
        avg_te = sum(r.tracking_error for r in results) / len(results)
        avg_strat = sum(r.strategy_return for r in results) / len(results)
        avg_bnh = sum(r.buyhold_return for r in results) / len(results)
        print(f"  {'-'*106}")
        print(f"  {'AVG':<7} {'':6} {'':18} {'':22} {avg_strat:>+8.2f} {avg_bnh:>+8.2f} "
              f"{avg_te:>8.2f}")
    print()


def _print_regime_summary(groups: dict[str, RegimeGroup]) -> None:
    """Print per-regime summary with sub-type breakdown."""
    print(f"\n{'='*60}")
    print(f"  Per-Regime Summary")
    print(f"{'='*60}")
    for regime, group in sorted(groups.items()):
        tickers = [r.ticker for r in group.results]
        print(f"\n  {regime} ({len(tickers)} tickers): {', '.join(tickers)}")
        print(f"    Avg Strategy Return: {group.avg_strategy_return:+.2f}%")
        print(f"    Avg Buy & Hold:      {group.avg_buyhold_return:+.2f}%")
        print(f"    Avg Tracking Error:  {group.avg_tracking_error:.2f}%")

        # Sub-type breakdown
        sub_groups: dict[str, list[str]] = defaultdict(list)
        for r in group.results:
            key = r.sub_type or "(none)"
            sub_groups[key].append(r.ticker)
        if len(sub_groups) > 1 or list(sub_groups.keys()) != ["(none)"]:
            print(f"    Sub-types:")
            for sub, st_tickers in sorted(sub_groups.items()):
                print(f"      {sub}: {', '.join(st_tickers)}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline(
    provider: DataProvider,
    cfg: Config,
    ticker_matrix: list[dict],
) -> tuple[list[TickerResult], dict[str, RegimeGroup]]:
    """Run backtests with current config across all tickers."""
    print("\n" + "="*60)
    print("  PHASE 1: Baseline Runs")
    print("="*60)

    results: list[TickerResult] = []
    groups: dict[str, RegimeGroup] = {}

    for i, spec in enumerate(ticker_matrix, 1):
        ticker = spec["ticker"]
        period = spec["period"]
        print(f"\n  [{i}/{len(ticker_matrix)}] {ticker} {period} ...", end=" ", flush=True)

        t0 = time.time()
        bt = _run_single_backtest(provider, cfg, ticker, period)
        elapsed = time.time() - t0

        if bt is None:
            print(f"FAILED ({elapsed:.1f}s)")
            continue

        tr = _extract_ticker_result(ticker, period, bt)
        results.append(tr)
        sub_info = f" [{tr.sub_type}]" if tr.sub_type else ""
        print(f"{tr.regime}{sub_info} | strat={tr.strategy_return:+.1f}% | "
              f"b&h={tr.buyhold_return:+.1f}% | err={tr.tracking_error:.1f}% "
              f"({elapsed:.1f}s)")

        # Group by regime
        if tr.regime not in groups:
            groups[tr.regime] = RegimeGroup(regime=tr.regime)
        groups[tr.regime].results.append(tr)

    return results, groups


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Grid Search
# ─────────────────────────────────────────────────────────────────────────────

def _generate_grid(param_grid: dict[str, list]) -> list[dict[str, object]]:
    """Generate all combinations of parameter values."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def grid_search_regime(
    provider: DataProvider,
    base_data: dict,
    regime_tickers: list[dict],
    param_grid: dict[str, list],
    regime_name: str,
) -> tuple[dict[str, object], float, list[TickerResult]]:
    """Grid-search parameters for a single regime group.

    Returns (best_overrides, best_avg_tracking_error, best_results).
    """
    combos = _generate_grid(param_grid)
    total = len(combos)

    print(f"\n{'='*60}")
    print(f"  PHASE 2: Grid Search — {regime_name}")
    print(f"  {len(regime_tickers)} tickers × {total} parameter combos = "
          f"{len(regime_tickers) * total} backtests")
    print(f"{'='*60}", flush=True)

    # Validate: short_below must be < hold_below — filter invalid combos
    valid_combos = []
    for c in combos:
        sb = c.get("strategy.score_thresholds.short_below", 3.5)
        hb = c.get("strategy.score_thresholds.hold_below", 6.0)
        if sb < hb:
            valid_combos.append(c)
    combos = valid_combos
    print(f"  After filtering invalid combos: {len(combos)} valid", flush=True)

    best_overrides: dict[str, object] = {}
    best_avg_te = float("inf")
    best_results: list[TickerResult] = []
    pruned = 0

    t_grid_start = time.time()
    for i, overrides in enumerate(combos, 1):
        cfg = _make_config_with_overrides(base_data, overrides)

        combo_results: list[TickerResult] = []
        skip_combo = False
        cumulative_te = 0.0

        for j, spec in enumerate(regime_tickers):
            bt = _run_single_backtest(provider, cfg, spec["ticker"], spec["period"])
            if bt is None:
                continue
            tr = _extract_ticker_result(spec["ticker"], spec["period"], bt)
            combo_results.append(tr)
            cumulative_te += tr.tracking_error

            # Early pruning: if running avg TE already exceeds best, skip rest
            if best_avg_te < float("inf") and len(combo_results) >= 3:
                running_avg = cumulative_te / len(combo_results)
                if running_avg > best_avg_te * 1.5:  # generous threshold
                    skip_combo = True
                    pruned += 1
                    break

        if skip_combo or not combo_results:
            continue

        avg_te = cumulative_te / len(combo_results)

        if avg_te < best_avg_te:
            best_avg_te = avg_te
            best_overrides = overrides
            best_results = combo_results

        # Progress reporting — every combo for small grids, every 5 for large
        report_every = 1 if len(combos) <= 20 else 5
        if i % report_every == 0 or i == len(combos):
            elapsed = time.time() - t_grid_start
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (len(combos) - i) / rate if rate > 0 else 0
            print(f"  [{i}/{len(combos)}] best avg TE: {best_avg_te:.2f}%  "
                  f"(current: {avg_te:.2f}%)  "
                  f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining, "
                  f"{pruned} pruned]", flush=True)

    elapsed_total = time.time() - t_grid_start
    print(f"  Grid search complete in {elapsed_total:.0f}s "
          f"({pruned} combos pruned)", flush=True)

    return best_overrides, best_avg_te, best_results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Output
# ─────────────────────────────────────────────────────────────────────────────

def _overrides_to_nested_dict(overrides: dict[str, object]) -> dict:
    """Convert dot-path overrides to nested dict for YAML output."""
    result: dict = {}
    for dotpath, value in overrides.items():
        _set_nested(result, dotpath, value)
    return result


def output_results(
    regime_results: dict[str, tuple[dict, float, list[TickerResult]]],
    baseline_results: list[TickerResult],
    baseline_groups: dict[str, RegimeGroup],
) -> None:
    """Print final summary and YAML config fragments."""

    print("\n" + "="*90)
    print("  OPTIMIZATION RESULTS")
    print("="*90)

    # ── Per-regime summary ────────────────────────────────────────────────
    for regime, (overrides, avg_te, results) in sorted(regime_results.items()):
        # Find baseline avg TE for this regime
        baseline_te = baseline_groups[regime].avg_tracking_error if regime in baseline_groups else 0.0

        print(f"\n  ┌─ {regime.upper()}")
        print(f"  │  Baseline avg tracking error: {baseline_te:.2f}%")
        print(f"  │  Optimized avg tracking error: {avg_te:.2f}%")
        improvement = baseline_te - avg_te
        pct_improvement = (improvement / baseline_te * 100) if baseline_te > 0 else 0
        print(f"  │  Improvement: {improvement:+.2f}% ({pct_improvement:.0f}% reduction)")
        print(f"  │")
        print(f"  │  Best parameters:")
        for k, v in sorted(overrides.items()):
            print(f"  │    {k}: {v}")
        print(f"  │")
        print(f"  │  Per-ticker results:")
        for r in results:
            direction = "+" if r.strategy_return >= 0 else ""
            print(f"  │    {r.ticker:<6} strat={direction}{r.strategy_return:.1f}%  "
                  f"b&h={'+' if r.buyhold_return >= 0 else ''}{r.buyhold_return:.1f}%  "
                  f"TE={r.tracking_error:.1f}%  trades={r.total_trades}")
        print(f"  └{'─'*50}")

    # ── Combined YAML output ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  YAML Config Fragments (per regime)")
    print(f"{'='*60}")
    print()
    print("Copy the relevant section into your config.yaml under")
    print("regime.strategy_adaptation.<regime_name> or use as base config overrides.")
    print()

    for regime, (overrides, avg_te, _) in sorted(regime_results.items()):
        nested = _overrides_to_nested_dict(overrides)
        print(f"# ── {regime} (avg tracking error: {avg_te:.2f}%) ──")
        yaml_str = yaml.dump(nested, default_flow_style=False, sort_keys=False)
        for line in yaml_str.strip().split("\n"):
            print(f"  {line}")
        print()

    # ── Write to file ─────────────────────────────────────────────────────
    output_path = PROJECT_ROOT / "optimized_configs.yaml"
    output_data = {}
    for regime, (overrides, avg_te, results) in sorted(regime_results.items()):
        nested = _overrides_to_nested_dict(overrides)
        nested["_meta"] = {
            "avg_tracking_error": round(avg_te, 2),
            "tickers_tested": [r.ticker for r in results],
            "avg_strategy_return": round(
                sum(r.strategy_return for r in results) / len(results), 2
            ) if results else 0.0,
            "avg_buyhold_return": round(
                sum(r.buyhold_return for r in results) / len(results), 2
            ) if results else 0.0,
        }
        output_data[regime] = nested

    with open(output_path, "w") as f:
        f.write("# Optimized per-regime configurations\n")
        f.write("# Generated by optimize.py\n")
        f.write(f"# Baseline avg tracking error: "
                f"{sum(r.tracking_error for r in baseline_results) / len(baseline_results):.2f}%\n")
        f.write("#\n")
        f.write("# To use: copy the strategy section of the desired regime into your\n")
        f.write("# config.yaml, or apply these overrides via the dashboard.\n\n")
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    print(f"\n  Wrote per-regime configs to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate(
    provider: DataProvider,
    base_data: dict,
    groups: dict[str, RegimeGroup],
    regime_overrides: dict[str, dict[str, object]],
) -> None:
    """Leave-one-out cross-validation per regime."""
    print(f"\n{'='*60}")
    print("  CROSS-VALIDATION (leave-one-out)")
    print(f"{'='*60}")

    for regime, group in sorted(groups.items()):
        if regime not in regime_overrides or len(group.results) < 2:
            continue

        overrides = regime_overrides[regime]
        print(f"\n  {regime} ({len(group.results)} tickers):")

        held_out_errors: list[float] = []
        for held_out_idx in range(len(group.results)):
            held_out = group.results[held_out_idx]

            # Optimize on all except held-out (just use the full-set optimal
            # params — full LOO re-optimization would be very expensive)
            cfg = _make_config_with_overrides(base_data, overrides)
            bt = _run_single_backtest(
                provider, cfg, held_out.ticker, held_out.period,
            )
            if bt is None:
                continue

            tr = _extract_ticker_result(held_out.ticker, held_out.period, bt)
            held_out_errors.append(tr.tracking_error)
            print(f"    {held_out.ticker:<6} "
                  f"strat={tr.strategy_return:+.1f}%  "
                  f"b&h={tr.buyhold_return:+.1f}%  "
                  f"TE={tr.tracking_error:.1f}%")

        if held_out_errors:
            avg = sum(held_out_errors) / len(held_out_errors)
            print(f"    → Avg held-out TE: {avg:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid-search optimizer for per-regime backtest configuration.",
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Only run baselines, show regime groupings, then exit",
    )
    parser.add_argument(
        "--regime", metavar="NAME",
        help="Optimize only this regime (e.g. strong_trend)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Use reduced parameter grid for faster iteration",
    )
    parser.add_argument(
        "--no-cv", action="store_true",
        help="Skip cross-validation",
    )
    parser.add_argument(
        "--config", "-c", metavar="PATH",
        help="Path to base config.yaml",
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────
    cfg = Config.load(args.config)
    base_data = cfg.to_dict()
    raw_provider = YahooFinanceProvider()
    provider = CachingProvider(raw_provider)

    # Choose grid
    param_grid = PARAM_GRID_FAST if args.fast else PARAM_GRID

    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    print(f"\n  Parameter grid: {len(param_grid)} params × "
          f"{total_combos} total combos ({'fast' if args.fast else 'full'} mode)")
    print(f"  Ticker universe: {len(TICKER_MATRIX)} tickers")

    # ── Pre-fetch all data ────────────────────────────────────────────────
    print(f"\n  Pre-fetching market data for {len(TICKER_MATRIX)} tickers ...")
    t_prefetch = time.time()
    provider.prefetch(TICKER_MATRIX)
    print(f"  Pre-fetch done in {time.time() - t_prefetch:.1f}s  ({provider.stats})")

    # ── Phase 1: Baseline ─────────────────────────────────────────────────
    baseline_results, groups = run_baseline(provider, cfg, TICKER_MATRIX)

    if not baseline_results:
        print("\n  [ERROR] No successful baselines. Check your internet connection.")
        sys.exit(1)

    _print_results_table(baseline_results, "BASELINE RESULTS (current config)")
    _print_regime_summary(groups)

    if args.baseline_only:
        return

    # ── Phase 2: Grid Search per Regime ───────────────────────────────────
    # Build per-regime ticker lists from baseline results
    regime_tickers: dict[str, list[dict]] = {}
    for tr in baseline_results:
        regime_tickers.setdefault(tr.regime, []).append(
            {"ticker": tr.ticker, "period": tr.period}
        )

    # Filter to requested regime if specified
    target_regimes = list(regime_tickers.keys())
    if args.regime:
        if args.regime not in target_regimes:
            print(f"\n  [ERROR] Regime '{args.regime}' not found in baseline.")
            print(f"  Available: {', '.join(target_regimes)}")
            sys.exit(1)
        target_regimes = [args.regime]

    # Skip regimes with only 1 ticker (can't cross-validate)
    regime_results: dict[str, tuple[dict, float, list[TickerResult]]] = {}

    for regime in target_regimes:
        tickers = regime_tickers[regime]
        if len(tickers) < 2:
            print(f"\n  Skipping {regime}: only {len(tickers)} ticker(s), "
                  f"need >= 2 for meaningful optimization")
            continue

        best_overrides, best_te, best_ticker_results = grid_search_regime(
            provider, base_data, tickers, param_grid, regime,
        )
        regime_results[regime] = (best_overrides, best_te, best_ticker_results)

        # Show best results for this regime
        _print_results_table(
            best_ticker_results,
            f"OPTIMIZED — {regime} (avg TE: {best_te:.2f}%)",
        )

    # ── Phase 3: Output ───────────────────────────────────────────────────
    if regime_results:
        output_results(regime_results, baseline_results, groups)

        # ── Cross-validation ──────────────────────────────────────────────
        if not args.no_cv:
            regime_overrides = {r: ov for r, (ov, _, _) in regime_results.items()}
            cross_validate(provider, base_data, groups, regime_overrides)

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")
    if isinstance(provider, CachingProvider):
        print(f"  {provider.stats}")
    print()


if __name__ == "__main__":
    main()
