"""
engine/walk_forward.py — Walk-forward (rolling out-of-sample) testing.

Splits historical data into rolling train/test windows and runs the
backtest engine on each test window.  Aggregates per-window metrics
into a summary with a stability score and verdict.

Usage:
    from engine.walk_forward import WalkForwardEngine

    wf = WalkForwardEngine(data_provider, strategy_factory, cfg)
    result = wf.run("AAPL", train_years=5, test_years=1)
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from config import Config
    from data.provider import DataProvider
    from engine.strategy import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    """Metrics from a single train/test window."""
    window_index: int
    train_start: str          # YYYY-MM-DD
    train_end: str
    test_start: str
    test_end: str
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    error: str | None = None  # non-None if this window failed


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward analysis results."""
    ticker: str
    train_years: int
    test_years: int
    total_windows: int = 0
    windows: list[WindowResult] = field(default_factory=list)

    # Aggregate metrics (average across successful windows)
    avg_return_pct: float = 0.0
    avg_annualized_return_pct: float = 0.0
    avg_max_drawdown_pct: float = 0.0
    avg_sharpe_ratio: float = 0.0
    avg_win_rate_pct: float = 0.0
    avg_profit_factor: float = 0.0

    # Worst-case window
    worst_return_pct: float = 0.0
    worst_drawdown_pct: float = 0.0
    worst_window_index: int = 0

    # Stability
    return_std_dev: float = 0.0       # standard deviation of per-window returns
    stability_score: float = 0.0      # 0-100, higher = more stable
    verdict: str = ""                 # human-readable summary


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class WalkForwardEngine:
    """Runs rolling walk-forward validation on a ticker.

    For each window:
      1. Define train period [train_start, train_end] and test period
         [test_start, test_end].
      2. Run the backtest on the test period only (train period is
         implicit — the engine uses warmup bars from the start of data).
      3. Collect metrics from BacktestResult.

    Windows slide forward by ``test_years`` each step.  The engine
    generates as many windows as fit within the available data history.
    """

    def __init__(
        self,
        data_provider: "DataProvider",
        strategy_factory: Callable[[], "Strategy"],
        cfg: "Config",
    ) -> None:
        """
        Args:
            data_provider:    DataProvider instance (e.g. YahooFinanceProvider).
            strategy_factory: A callable that returns a *fresh* Strategy instance
                              for each window (avoids state leaking between windows).
            cfg:              Config instance.
        """
        self._provider = data_provider
        self._strategy_factory = strategy_factory
        self._cfg = cfg

    def _generate_windows(
        self,
        train_years: int,
        test_years: int,
        max_windows: int = 10,
    ) -> list[dict[str, str]]:
        """Generate date windows working backwards from today.

        Returns a list of dicts with keys:
            train_start, train_end, test_start, test_end
        (all YYYY-MM-DD strings).
        """
        today = datetime.now().date()
        windows: list[dict[str, str]] = []
        total_span = train_years + test_years

        for i in range(max_windows):
            test_end = today - timedelta(days=i * test_years * 365)
            test_start = test_end - timedelta(days=test_years * 365)
            train_end = test_start - timedelta(days=1)
            train_start = test_start - timedelta(days=train_years * 365)

            # Don't go before 2000 (insufficient data)
            if train_start.year < 2000:
                break

            windows.append({
                "train_start": train_start.isoformat(),
                "train_end": train_end.isoformat(),
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
            })

        windows.reverse()  # chronological order
        return windows

    def run(
        self,
        ticker: str,
        train_years: int = 5,
        test_years: int = 1,
        max_windows: int = 10,
    ) -> WalkForwardResult:
        """Run walk-forward testing and return aggregated results.

        Args:
            ticker:       Stock symbol.
            train_years:  Training window length in years.
            test_years:   Out-of-sample test window length in years.
            max_windows:  Maximum number of windows to generate.

        Returns:
            :class:`WalkForwardResult` with per-window and aggregate metrics.
        """
        from engine.backtest import BacktestEngine  # type: ignore[import-untyped]
        from engine.suitability import TradingMode  # type: ignore[import-untyped]

        windows = self._generate_windows(train_years, test_years, max_windows)
        result = WalkForwardResult(
            ticker=ticker.upper(),
            train_years=train_years,
            test_years=test_years,
            total_windows=len(windows),
        )

        if not windows:
            result.verdict = "Not enough historical data to create walk-forward windows."
            return result

        for i, w in enumerate(windows):
            logger.info(
                "Walk-forward window %d/%d: test %s → %s",
                i + 1, len(windows), w["test_start"], w["test_end"],
            )

            wr = WindowResult(
                window_index=i,
                train_start=w["train_start"],
                train_end=w["train_end"],
                test_start=w["test_start"],
                test_end=w["test_end"],
            )

            try:
                # Fresh strategy for each window
                strategy = self._strategy_factory()
                engine = BacktestEngine(
                    data_provider=self._provider,
                    strategy=strategy,
                    cfg=self._cfg,
                    trading_mode=TradingMode.LONG_SHORT,
                )

                # Run on the test window only.  We pass start/end so the
                # engine fetches that specific date range.  The engine's
                # warmup bars will use the initial portion of the data.
                bt = engine.run(
                    ticker=ticker.upper(),
                    period=None,
                    interval="1d",
                    start=w["train_start"],  # include train period for warmup
                    end=w["test_end"],
                )

                wr.total_return_pct = bt.total_return_pct
                wr.annualized_return_pct = bt.annualized_return_pct
                wr.max_drawdown_pct = bt.max_drawdown_pct
                wr.sharpe_ratio = bt.sharpe_ratio
                wr.win_rate_pct = bt.win_rate_pct
                wr.profit_factor = bt.profit_factor
                wr.total_trades = bt.total_trades

            except Exception as exc:
                logger.warning("Window %d failed: %s", i, exc)
                wr.error = str(exc)

            result.windows.append(wr)

        # Aggregate across successful windows
        ok = [w for w in result.windows if w.error is None]
        if ok:
            result.avg_return_pct = statistics.mean(w.total_return_pct for w in ok)
            result.avg_annualized_return_pct = statistics.mean(
                w.annualized_return_pct for w in ok
            )
            result.avg_max_drawdown_pct = statistics.mean(w.max_drawdown_pct for w in ok)
            result.avg_sharpe_ratio = statistics.mean(w.sharpe_ratio for w in ok)
            result.avg_win_rate_pct = statistics.mean(w.win_rate_pct for w in ok)
            result.avg_profit_factor = statistics.mean(w.profit_factor for w in ok)

            # Worst-case
            worst = min(ok, key=lambda w: w.total_return_pct)
            result.worst_return_pct = worst.total_return_pct
            result.worst_drawdown_pct = worst.max_drawdown_pct
            result.worst_window_index = worst.window_index

            # Stability score
            if len(ok) >= 2:
                returns = [w.total_return_pct for w in ok]
                result.return_std_dev = statistics.stdev(returns)
                # Stability: lower variance = higher score
                # Score = max(0, 100 - stdev * 5)  (capped at 0-100)
                raw = 100.0 - result.return_std_dev * 5.0
                result.stability_score = max(0.0, min(100.0, raw))
            else:
                result.stability_score = 50.0  # unknown with single window

            # Verdict
            n_profitable = sum(1 for w in ok if w.total_return_pct > 0)
            pct_profitable = n_profitable / len(ok) * 100

            if result.stability_score >= 70 and pct_profitable >= 60:
                result.verdict = (
                    f"Strategy is stable across {len(ok)} walk-forward windows. "
                    f"{pct_profitable:.0f}% of windows were profitable."
                )
            elif result.stability_score >= 40:
                result.verdict = (
                    f"Strategy shows moderate consistency across {len(ok)} windows. "
                    f"{pct_profitable:.0f}% of windows were profitable. "
                    f"Returns vary with a std dev of {result.return_std_dev:.1f}%."
                )
            else:
                result.verdict = (
                    f"Warning: significant performance variation across {len(ok)} windows. "
                    f"Only {pct_profitable:.0f}% were profitable. "
                    f"The strategy may be overfit or highly regime-dependent."
                )
        else:
            result.verdict = "All walk-forward windows failed. Check data availability."

        return result
