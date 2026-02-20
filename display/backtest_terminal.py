"""
display/backtest_terminal.py — Rich-based terminal output for backtest results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

if TYPE_CHECKING:
    from config import Config
    from engine.backtest import BacktestResult
    from engine.suitability import SuitabilityAssessment

console = Console()


def _pnl_color(value: float) -> str:
    if value > 0:
        return "green"
    if value < 0:
        return "red"
    return "white"


def _mode_color(mode_value: str) -> str:
    """Color for trading mode labels."""
    if mode_value == "long_short":
        return "green"
    if mode_value == "long_only":
        return "yellow"
    return "red"  # hold_only


def render_suitability(
    assessment: "SuitabilityAssessment",
    ticker: str,
) -> None:
    """Render a standalone suitability assessment panel (used when hold_only)."""
    mode_label = assessment.mode.value.replace("_", " ").upper()
    color = _mode_color(assessment.mode.value)

    lines = [
        f"[bold white]{ticker}[/bold white]  "
        f"Trading Mode: [{color}][bold]{mode_label}[/bold][/{color}]"
        + ("  [dim](forced)[/dim]" if assessment.forced else "  [dim](auto-detected)[/dim]"),
        "",
    ]

    # Metrics
    lines.append("[bold]Metrics:[/bold]")
    lines.append(f"  Avg Daily Volume:  {assessment.avg_daily_volume:>12,.0f}")
    lines.append(f"  ADX:               {assessment.adx_value:>12.1f}")
    lines.append(f"  ATR%:              {assessment.atr_pct:>12.3f}  ({assessment.atr_pct * 100:.1f}%)")
    lines.append("")

    # Reasons
    lines.append("[bold]Reasons:[/bold]")
    for reason in assessment.reasons:
        lines.append(f"  [dim]- {reason}[/dim]")

    console.print(Panel(
        "\n".join(lines),
        title="[bold]Suitability Assessment[/bold]",
        box=box.ROUNDED,
        expand=True,
        padding=(0, 2),
    ))


def render_backtest(
    result: "BacktestResult",
    cfg: "Config",
    assessment: "SuitabilityAssessment | None" = None,
) -> None:
    """Render backtest results to the terminal."""
    disp_cfg = cfg.section("display")
    price_dp = int(disp_cfg.get("price_decimal_places", 2))

    # ── Header ──────────────────────────────────────────────────────────────
    ret_color = _pnl_color(result.total_return_pct)
    header_lines = [
        f"[bold white]Backtest Results[/bold white]  "
        f"[bold cyan]{result.ticker}[/bold cyan]  "
        f"[dim]Period: {result.period}  |  Strategy: {result.strategy_name}[/dim]",
    ]

    console.print()
    console.print(Panel(
        "\n".join(header_lines),
        box=box.DOUBLE_EDGE,
        style="bold",
        expand=True,
        padding=(0, 2),
    ))

    # ── Suitability Assessment Panel ────────────────────────────────────────
    if assessment is not None:
        render_suitability(assessment, result.ticker)

    # ── Performance Metrics Table ───────────────────────────────────────────
    metrics_table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold cyan",
        expand=True,
        padding=(0, 1),
        title="[bold]Performance Summary[/bold]",
        title_style="bold white",
    )
    metrics_table.add_column("Metric", style="bold white", min_width=24, no_wrap=True)
    metrics_table.add_column("Value", min_width=20, no_wrap=True)

    metrics = [
        ("Initial Cash", f"${result.initial_cash:,.2f}", "white"),
        ("Final Equity", f"${result.final_equity:,.2f}", _pnl_color(result.final_equity - result.initial_cash)),
        ("Total Return", f"{result.total_return_pct:+.2f}%", ret_color),
        ("Annualized Return", f"{result.annualized_return_pct:+.2f}%", _pnl_color(result.annualized_return_pct)),
        ("Max Drawdown", f"{result.max_drawdown_pct:.2f}%", "red" if result.max_drawdown_pct > 10 else "yellow"),
        ("Sharpe Ratio", f"{result.sharpe_ratio:.2f}", _pnl_color(result.sharpe_ratio)),
        ("", "", "white"),  # spacer
        ("Total Trades", f"{result.total_trades}", "white"),
        ("Win Rate", f"{result.win_rate_pct:.1f}%", "green" if result.win_rate_pct >= 50 else "red"),
        ("Profit Factor", f"{result.profit_factor:.2f}" if result.profit_factor != float("inf") else "inf (no losses)", _pnl_color(result.profit_factor - 1)),
        ("Avg Trade P&L", f"{result.avg_trade_pnl_pct:+.2f}%", _pnl_color(result.avg_trade_pnl_pct)),
        ("Best Trade", f"{result.best_trade_pnl_pct:+.2f}%", "green"),
        ("Worst Trade", f"{result.worst_trade_pnl_pct:+.2f}%", "red"),
    ]

    for label, value, color in metrics:
        if not label:
            metrics_table.add_section()
        else:
            metrics_table.add_row(label, f"[{color}]{value}[/{color}]")

    console.print(metrics_table)

    # ── Strategy Config Summary ─────────────────────────────────────────────
    strat_cfg = cfg.section("strategy")
    bt_cfg = cfg.section("backtest")
    thresholds = strat_cfg.get("score_thresholds", {})

    config_table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold cyan",
        expand=True,
        padding=(0, 1),
        title="[bold]Strategy Configuration[/bold]",
        title_style="bold white",
    )
    config_table.add_column("Parameter", style="bold white", min_width=24, no_wrap=True)
    config_table.add_column("Value", style="white", min_width=20, no_wrap=True)

    config_rows = []

    threshold_mode = strat_cfg.get("threshold_mode", "fixed")
    config_rows.append(("Threshold Mode", threshold_mode))

    if threshold_mode == "percentile":
        pct_cfg = strat_cfg.get("percentile_thresholds", {})
        config_rows.append(("SHORT when percentile <=", f"{pct_cfg.get('short_percentile', 25)}%"))
        config_rows.append(("LONG when percentile >=", f"{pct_cfg.get('long_percentile', 75)}%"))
        config_rows.append(("Lookback Window", f"{pct_cfg.get('lookback_bars', 60)} bars"))
        config_rows.append(("(Fallback) SHORT when score <=", f"{thresholds.get('short_below', 3.5)}"))
        config_rows.append(("(Fallback) LONG when score >", f"{thresholds.get('hold_below', 6.5)}"))
    else:
        config_rows.append(("Signal: SHORT when score <=", f"{thresholds.get('short_below', 3.5)}"))
        config_rows.append(("Signal: HOLD when score <=", f"{thresholds.get('hold_below', 6.5)}"))
        config_rows.append(("Signal: LONG when score >", f"{thresholds.get('hold_below', 6.5)}"))

    config_rows += [
        ("Position Sizing", strat_cfg.get("position_sizing", "fixed")),
        ("Fixed Quantity", f"{strat_cfg.get('fixed_quantity', 100)} shares"),
        ("Stop Loss", f"{strat_cfg.get('stop_loss_pct', 0.05) * 100:.1f}%"),
        ("Take Profit", f"{strat_cfg.get('take_profit_pct', 0.15) * 100:.1f}%"),
        ("Rebalance Interval", f"every {strat_cfg.get('rebalance_interval', 5)} bars"),
        ("Slippage", f"{bt_cfg.get('slippage_pct', 0.001) * 100:.2f}%"),
        ("Commission", f"${bt_cfg.get('commission_per_trade', 0.0):.2f} / trade"),
        ("Warmup Bars", f"{bt_cfg.get('warmup_bars', 200)}"),
    ]

    for label, value in config_rows:
        config_table.add_row(label, value)

    console.print(config_table)

    # ── Trade Log ───────────────────────────────────────────────────────────
    if result.trades:
        trade_table = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold cyan",
            expand=True,
            padding=(0, 1),
            title="[bold]Trade Log[/bold]",
            title_style="bold white",
        )
        trade_table.add_column("#", style="dim", min_width=4, no_wrap=True)
        trade_table.add_column("Side", min_width=6, no_wrap=True)
        trade_table.add_column("Entry Date", min_width=12, no_wrap=True)
        trade_table.add_column("Entry Price", min_width=12, no_wrap=True)
        trade_table.add_column("Exit Date", min_width=12, no_wrap=True)
        trade_table.add_column("Exit Price", min_width=12, no_wrap=True)
        trade_table.add_column("Qty", min_width=6, no_wrap=True)
        trade_table.add_column("P&L", min_width=12, no_wrap=True)
        trade_table.add_column("P&L %", min_width=10, no_wrap=True)
        trade_table.add_column("Exit Reason", style="dim", min_width=12, no_wrap=True)

        # Show up to 50 trades; summarize if more
        display_trades = result.trades
        truncated = False
        if len(display_trades) > 50:
            display_trades = result.trades[:25] + result.trades[-25:]
            truncated = True

        for idx, t in enumerate(display_trades):
            # Determine actual index
            if truncated and idx >= 25:
                real_idx = len(result.trades) - 50 + idx + 1
            else:
                real_idx = idx + 1

            color = _pnl_color(t.pnl)
            side_color = "cyan" if t.side == "long" else "magenta"

            trade_table.add_row(
                str(real_idx),
                f"[{side_color}]{t.side.upper()}[/{side_color}]",
                t.entry_date,
                f"${t.entry_price:.{price_dp}f}",
                t.exit_date,
                f"${t.exit_price:.{price_dp}f}",
                f"{t.quantity:.0f}",
                f"[{color}]${t.pnl:+,.2f}[/{color}]",
                f"[{color}]{t.pnl_pct * 100:+.2f}%[/{color}]",
                t.exit_reason,
            )

            # Insert separator when truncating
            if truncated and idx == 24:
                trade_table.add_section()
                trade_table.add_row(
                    "", f"[dim]... {len(result.trades) - 50} more trades ...[/dim]",
                    "", "", "", "", "", "", "", "",
                )
                trade_table.add_section()

        console.print(trade_table)
    else:
        console.print("\n  [dim]No trades were executed during the backtest period.[/dim]")

    # ── Equity Curve Summary ────────────────────────────────────────────────
    if result.equity_curve:
        curve = result.equity_curve
        n = len(curve)
        # Show a simple text-based sparkline summary: start, 25%, 50%, 75%, end
        checkpoints = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        eq_parts = []
        for ci in checkpoints:
            pt = curve[ci]
            eq_parts.append(f"{pt['date']}: ${pt['equity']:,.0f}")

        console.print()
        console.print(
            Panel(
                "  →  ".join(eq_parts),
                title="[bold]Equity Curve (checkpoints)[/bold]",
                box=box.ROUNDED,
                style="dim",
                expand=True,
                padding=(0, 2),
            )
        )

    # ── Disclaimer ──────────────────────────────────────────────────────────
    console.print()
    console.print(
        "  [dim italic]Backtest results are hypothetical and do not guarantee future "
        "performance. Past results are not indicative of future returns.[/dim italic]"
    )
    console.print()
