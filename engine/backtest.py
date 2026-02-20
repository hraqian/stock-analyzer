"""
engine/backtest.py — Backtesting engine stub.

This is ready for future implementation.

Planned usage:
    from engine.backtest import BacktestEngine
    from engine.strategy import MyStrategy

    engine = BacktestEngine(
        data_provider=YahooFinanceProvider(),
        strategy=MyStrategy(params={"stop_loss": 0.05}),
        cfg=cfg,
    )
    result = engine.run("AAPL", period="2y")
    result.summary()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config import Config
    from data.provider import DataProvider
    from engine.strategy import Strategy


@dataclass
class BacktestTrade:
    """A single completed trade recorded during backtesting."""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    quantity: float
    side: str           # "long" or "short"
    pnl: float
    pnl_pct: float
    notes: str = ""


@dataclass
class BacktestResult:
    """Aggregated results from a backtest run."""
    ticker: str
    period: str
    strategy_name: str
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    # Performance metrics (populated after run)
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate_pct: float = 0.0
    total_trades: int = 0
    profit_factor: float = 0.0

    def summary(self) -> None:
        """Print a summary of backtest results to stdout."""
        # TODO: implement rich-based display
        print(f"\nBacktest Results: {self.ticker} | {self.strategy_name} | {self.period}")
        print(f"  Total trades   : {self.total_trades}")
        print(f"  Total return   : {self.total_return_pct:+.2f}%")
        print(f"  Max drawdown   : {self.max_drawdown_pct:.2f}%")
        print(f"  Win rate       : {self.win_rate_pct:.1f}%")
        print(f"  Sharpe ratio   : {self.sharpe_ratio:.2f}")


class BacktestEngine:
    """Runs a Strategy against historical OHLCV data.

    This is a stub — the core simulation loop is not yet implemented.

    Planned features:
      - Bar-by-bar simulation with indicator recomputation
      - Configurable position sizing (fixed, % of equity, Kelly)
      - Commission and slippage modelling
      - Long and short support
      - Walk-forward validation
      - Multiple ticker portfolio backtesting
    """

    def __init__(
        self,
        data_provider: "DataProvider",
        strategy: "Strategy",
        cfg: "Config",
    ) -> None:
        self._provider = data_provider
        self._strategy = strategy
        self._cfg = cfg

    def run(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
        initial_cash: float = 100_000.0,
    ) -> BacktestResult:
        """Run the backtest.

        Args:
            ticker:       Stock symbol.
            period:       Historical period to test over.
            interval:     Bar interval.
            initial_cash: Starting portfolio cash.

        Returns:
            :class:`BacktestResult` with trades and performance metrics.

        Raises:
            NotImplementedError: Until the engine is fully implemented.
        """
        raise NotImplementedError(
            "BacktestEngine is not yet implemented. "
            "The interface (Strategy, Signal, StrategyContext) is ready — "
            "implement BacktestEngine.run() to activate backtesting."
        )
