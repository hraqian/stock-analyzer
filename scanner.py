"""scanner.py — Stock universe scanner engine.

Scans a list of tickers in parallel, runs the full analysis + recommendation
pipeline on each, and ranks them to surface the top BUY and SELL candidates.

Designed to be used from both the CLI (``scan.py``) and the Streamlit dashboard.

Usage (programmatic)::

    from scanner import Scanner
    scanner = Scanner(universe="dow30", period="2y", max_workers=8)
    results = scanner.run()          # list[ScanResult]
    buys  = scanner.top_buys(10)     # top 10 BUY signals by score
    sells = scanner.top_sells(10)    # top 10 SELL signals by score
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ── Ensure project root is on sys.path ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.analyzer import Analyzer
from config import Config
from data.provider import DataProvider
from data.universes import available as available_universes
from data.universes import load as load_universe
from data.yahoo import YahooFinanceProvider
from engine.regime import RegimeSubType, RegimeType
from engine.score_strategy import ScoreBasedStrategy
from engine.suitability import TradingMode
from engine.strategy import StrategyContext


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ScanResult:
    """Result of scanning a single ticker."""

    ticker: str
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: str  # "high", "medium", "low"
    effective_score: float
    indicator_score: float
    pattern_score: float
    price: float
    regime: str  # e.g. "strong_trend"
    regime_label: str
    sub_type: str  # e.g. "steady_compounder"
    sub_type_label: str
    strategy_notes: str
    trend_score: float = 5.0
    contrarian_score: float = 5.0
    dominant_group: str = "none"
    error: str = ""  # non-empty if the ticker failed

    @property
    def sort_key_buy(self) -> float:
        """Higher is better for BUY ranking."""
        return self.effective_score

    @property
    def sort_key_sell(self) -> float:
        """Lower is better for SELL ranking (inverted for sorting)."""
        return -self.effective_score


REGIME_LABELS = {
    "strong_trend": "Strong Trend",
    "mean_reverting": "Mean Reverting",
    "volatile_choppy": "Volatile / Choppy",
    "breakout_transition": "Breakout / Transition",
}


# ─────────────────────────────────────────────────────────────────────────────
# Scanner engine
# ─────────────────────────────────────────────────────────────────────────────


class Scanner:
    """Parallel stock-universe scanner.

    Parameters
    ----------
    universe : str | list[str]
        A built-in universe name (``dow30``, ``nasdaq100``, ``sp500``),
        a path to a custom ``.txt`` file, or an explicit list of ticker
        strings.
    period : str
        Analysis period (default ``"2y"``).
    max_workers : int
        Thread pool size for parallel fetching (default ``8``).
    cfg : Config | None
        Optional pre-built Config.  Uses ``Config.load()`` if *None*.
    on_progress : callable | None
        ``(completed: int, total: int, ticker: str, result: ScanResult | None) -> None``
        Called after each ticker finishes.  Useful for progress bars.
    """

    def __init__(
        self,
        universe: str | list[str] = "dow30",
        period: str = "2y",
        max_workers: int = 8,
        cfg: Config | None = None,
        on_progress: Callable[[int, int, str, ScanResult | None], None] | None = None,
    ) -> None:
        self._period = period
        self._max_workers = max_workers
        self._cfg = cfg or Config.load()
        self._on_progress = on_progress

        # Resolve tickers
        if isinstance(universe, list):
            self._tickers = [t.upper() for t in universe]
            self._universe_name = "custom"
        else:
            self._tickers = load_universe(universe)
            self._universe_name = universe

        self._results: list[ScanResult] = []

    @property
    def universe_name(self) -> str:
        return self._universe_name

    @property
    def tickers(self) -> list[str]:
        return list(self._tickers)

    @property
    def results(self) -> list[ScanResult]:
        return list(self._results)

    # ── Core scan ────────────────────────────────────────────────────────

    def run(self) -> list[ScanResult]:
        """Run the full scan.  Returns list of ScanResult (one per ticker)."""
        self._results = []
        total = len(self._tickers)
        completed = 0

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(self._scan_one, ticker): ticker
                for ticker in self._tickers
            }

            for future in as_completed(futures):
                ticker = futures[future]
                completed += 1
                try:
                    result = future.result()
                except Exception as exc:
                    result = ScanResult(
                        ticker=ticker,
                        signal="HOLD",
                        confidence="low",
                        effective_score=5.0,
                        indicator_score=5.0,
                        pattern_score=5.0,
                        price=0.0,
                        regime="",
                        regime_label="",
                        sub_type="",
                        sub_type_label="",
                        strategy_notes="",
                        error=str(exc),
                    )
                self._results.append(result)
                if self._on_progress:
                    self._on_progress(completed, total, ticker, result)

        return list(self._results)

    def _scan_one(self, ticker: str) -> ScanResult:
        """Analyse a single ticker and return its recommendation."""
        # Each thread gets its own provider (yfinance is thread-safe for
        # separate tickers, but sharing one instance can cause issues).
        provider = YahooFinanceProvider()
        analyzer = Analyzer(self._cfg, provider)

        result = analyzer.run(ticker, self._period, "1d")

        # ── Build recommendation (mirrors dashboard.compute_recommendation) ──
        strat_cfg = self._cfg.section("strategy")
        regime_adapt = self._cfg.section("regime").get("strategy_adaptation", {})

        strategy = ScoreBasedStrategy(
            params=strat_cfg,
            trading_mode=TradingMode.LONG_SHORT,
            regime_adaptation=regime_adapt,
        )
        strategy.on_start({"ticker": ticker, "recommendation": True})

        df = result.df
        if df is None or df.empty:
            raise ValueError(f"No data returned for {ticker}")

        last_row = df.iloc[-1]
        bar_dict = {
            "open": float(last_row["open"]),
            "high": float(last_row["high"]),
            "low": float(last_row["low"]),
            "close": float(last_row["close"]),
            "volume": float(last_row["volume"]),
        }

        overall_score = result.composite.get("overall", 5.0)
        pattern_score = result.pattern_composite.get("overall", 5.0)

        per_scores: dict[str, float] = {}
        if result.indicator_results:
            for ir in result.indicator_results:
                if not ir.error:
                    per_scores[ir.config_key] = ir.score

        # Trend MA
        trend_period = int(strat_cfg.get("trend_confirm_period", 20))
        ma_type = str(strat_cfg.get("trend_confirm_ma_type", "ema"))
        if ma_type.lower() == "sma":
            trend_ma_series = df["close"].rolling(window=trend_period, min_periods=1).mean()
        else:
            trend_ma_series = df["close"].ewm(span=trend_period, adjust=False).mean()
        current_trend_ma = float(trend_ma_series.iloc[-1])

        # Regime
        regime_type: RegimeType | None = None
        regime_sub: RegimeSubType | None = None
        regime_trend: str = "neutral"
        regime_total_return: float = 0.0

        if result.regime is not None:
            regime_type = result.regime.regime
            regime_sub = result.regime.sub_type
            regime_trend = result.regime.metrics.trend_direction
            regime_total_return = result.regime.metrics.total_return

        ctx = StrategyContext(
            bar=bar_dict,
            indicators={},
            scores=per_scores,
            overall_score=overall_score,
            pattern_score=pattern_score,
            position=0.0,
            cash=100_000.0,
            portfolio_value=100_000.0,
            trend_ma=current_trend_ma,
            regime=regime_type,
            regime_sub_type=regime_sub,
            regime_trend=regime_trend,
            regime_total_return=regime_total_return,
        )

        order = strategy.on_bar(ctx)
        signal = order.signal.value  # "BUY", "SELL", "HOLD"

        # Effective score (same blending as dashboard)
        combination_mode = str(strat_cfg.get("combination_mode", "weighted"))
        ind_weight = float(strat_cfg.get("indicator_weight", 0.7))
        pat_weight = float(strat_cfg.get("pattern_weight", 0.3))

        if combination_mode == "boost":
            boost_strength = float(strat_cfg.get("boost_strength", 0.5))
            boost_dead_zone = float(strat_cfg.get("boost_dead_zone", 0.3))
            pat_dev = pattern_score - 5.0
            if abs(pat_dev) <= boost_dead_zone:
                effective_score = overall_score
            else:
                eff_dev = pat_dev - (boost_dead_zone if pat_dev > 0 else -boost_dead_zone)
                effective_score = max(0.0, min(10.0, overall_score + eff_dev * boost_strength))
        elif combination_mode == "gate":
            effective_score = overall_score
        else:
            w_total = ind_weight + pat_weight
            if w_total > 0:
                effective_score = (ind_weight * overall_score + pat_weight * pattern_score) / w_total
            else:
                effective_score = overall_score

        # Confidence
        thresholds = strat_cfg.get("score_thresholds", {})
        short_below = float(thresholds.get("short_below", 3.5))
        hold_below = float(thresholds.get("hold_below", 6.5))

        if signal == "BUY":
            distance = effective_score - hold_below
            confidence = "high" if distance >= 1.5 else ("medium" if distance >= 0.5 else "low")
        elif signal == "SELL":
            distance = short_below - effective_score
            confidence = "high" if distance >= 1.5 else ("medium" if distance >= 0.5 else "low")
        else:
            dist_to_buy = hold_below - effective_score
            dist_to_sell = effective_score - short_below
            min_dist = min(dist_to_buy, dist_to_sell)
            confidence = "high" if min_dist >= 1.0 else ("medium" if min_dist >= 0.3 else "low")

        # Labels
        regime_val = regime_type.value if regime_type else ""
        regime_label = REGIME_LABELS.get(regime_val, regime_val)
        sub_val = regime_sub.value if regime_sub else ""
        sub_label = result.regime.sub_type_label if result.regime else ""

        return ScanResult(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            effective_score=effective_score,
            indicator_score=overall_score,
            pattern_score=pattern_score,
            price=bar_dict["close"],
            regime=regime_val,
            regime_label=regime_label,
            sub_type=sub_val,
            sub_type_label=sub_label,
            strategy_notes=order.notes or "",
            trend_score=result.composite.get("trend_score", 5.0),
            contrarian_score=result.composite.get("contrarian_score", 5.0),
            dominant_group=result.composite.get("dominant_group", "none"),
        )

    # ── Ranking helpers ──────────────────────────────────────────────────

    def top_buys(self, n: int = 10) -> list[ScanResult]:
        """Return top *n* BUY signals, ranked by effective score descending."""
        buys = [r for r in self._results if r.signal == "BUY" and not r.error]
        buys.sort(key=lambda r: r.sort_key_buy, reverse=True)
        return buys[:n]

    def top_sells(self, n: int = 10) -> list[ScanResult]:
        """Return top *n* SELL signals, ranked by effective score ascending."""
        sells = [r for r in self._results if r.signal == "SELL" and not r.error]
        sells.sort(key=lambda r: r.effective_score)
        return sells[:n]

    def holds(self) -> list[ScanResult]:
        """Return all HOLD signals, sorted by score descending."""
        holds = [r for r in self._results if r.signal == "HOLD" and not r.error]
        holds.sort(key=lambda r: r.effective_score, reverse=True)
        return holds

    def errors(self) -> list[ScanResult]:
        """Return all tickers that failed during scan."""
        return [r for r in self._results if r.error]

    def summary(self) -> dict:
        """Return a quick summary dict."""
        ok = [r for r in self._results if not r.error]
        return {
            "universe": self._universe_name,
            "period": self._period,
            "total_tickers": len(self._tickers),
            "scanned": len(ok),
            "errors": len(self._results) - len(ok),
            "buy_count": sum(1 for r in ok if r.signal == "BUY"),
            "sell_count": sum(1 for r in ok if r.signal == "SELL"),
            "hold_count": sum(1 for r in ok if r.signal == "HOLD"),
        }
