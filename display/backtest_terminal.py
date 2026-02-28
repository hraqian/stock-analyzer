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
    from engine.regime import RegimeAssessment

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


def _signal_color(signal: str) -> str:
    """Color for bullish/bearish signal labels."""
    if signal == "bullish":
        return "green"
    if signal == "bearish":
        return "red"
    return "yellow"


def _render_significant_patterns(result: "BacktestResult") -> None:
    """Render a timeline table of all significant patterns detected."""
    patterns = result.significant_patterns
    if not patterns:
        return

    pat_table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold cyan",
        expand=True,
        padding=(0, 1),
        title="[bold]Significant Patterns Timeline[/bold]",
        title_style="bold white",
    )
    pat_table.add_column("Date", min_width=10, no_wrap=True)
    pat_table.add_column("Detector", min_width=14, no_wrap=True)
    pat_table.add_column("Pattern", min_width=20, no_wrap=True)
    pat_table.add_column("Signal", min_width=10, no_wrap=True)
    pat_table.add_column("Str", min_width=5, no_wrap=True, justify="right")
    pat_table.add_column("Confidence", min_width=11, no_wrap=True)
    pat_table.add_column("Detail", style="white", min_width=14)

    # Truncate if too many patterns
    display_patterns = patterns
    truncated = False
    max_display = 80
    if len(patterns) > max_display:
        half = max_display // 2
        display_patterns = patterns[:half] + patterns[-half:]
        truncated = True

    for idx, p in enumerate(display_patterns):
        color = _signal_color(p.signal)
        signal_arrow = "\u2191" if p.signal == "bullish" else ("\u2193" if p.signal == "bearish" else "\u25CB")

        # Color-code confidence label
        conf = p.confidence
        if conf == "Very Strong":
            conf_styled = "[bold bright_green]Very Strong[/bold bright_green]"
        elif conf == "Strong":
            conf_styled = "[green]Strong[/green]"
        elif conf == "Moderate":
            conf_styled = "[yellow]Moderate[/yellow]"
        else:
            conf_styled = f"[dim]{conf}[/dim]"

        pat_table.add_row(
            p.date,
            f"[dim]{p.detector}[/dim]",
            p.pattern,
            f"[{color}]{signal_arrow} {p.signal.upper()}[/{color}]",
            f"{p.strength:.2f}",
            conf_styled,
            p.detail,
        )

        # Insert separator when truncating
        if truncated and idx == max_display // 2 - 1:
            pat_table.add_section()
            pat_table.add_row(
                "", f"[dim]... {len(patterns) - max_display} more patterns ...[/dim]",
                "", "", "", "", "",
            )
            pat_table.add_section()

    # Summary line
    bullish_count = sum(1 for p in patterns if p.signal == "bullish")
    bearish_count = sum(1 for p in patterns if p.signal == "bearish")
    console.print(pat_table)
    console.print(
        f"  [dim]Total: {len(patterns)} significant patterns  "
        f"([green]{bullish_count} bullish[/green]  [red]{bearish_count} bearish[/red])[/dim]"
    )


def _render_regime_panel(regime: "RegimeAssessment") -> None:
    """Render a market regime classification panel."""
    regime_colors = {
        "strong_trend": "green",
        "mean_reverting": "cyan",
        "volatile_choppy": "red",
        "breakout_transition": "yellow",
    }
    r_color = regime_colors.get(regime.regime.value, "white")
    confidence_pct = regime.confidence * 100

    lines = [
        f"[bold {r_color}]{regime.label}[/bold {r_color}]"
        + (f"  [bold white]({regime.sub_type_label})[/bold white]" if regime.sub_type_label else "")
        + f"  [dim]Confidence: {confidence_pct:.0f}%[/dim]",
        f"[dim italic]{regime.description}[/dim italic]",
    ]
    if regime.sub_type_description:
        lines.append(f"[dim italic]Sub-type: {regime.sub_type_description}[/dim italic]")
    lines.append("")

    # Metrics summary
    m = regime.metrics
    lines.append(
        f"[dim]Return: {m.total_return:+.1%}  |  "
        f"ADX: {m.adx:.1f} (avg {m.rolling_adx_mean:.1f})  |  "
        f"Trend MA {m.pct_above_ma:.0f}% above  |  "
        f"ATR%: {m.atr_pct:.3f}  |  "
        f"Dir changes: {m.direction_changes:.0%}[/dim]"
    )

    # Reasons
    if regime.reasons:
        lines.append("")
        for reason in regime.reasons[:4]:
            lines.append(f"  [dim]- {reason}[/dim]")

    # Regime scores comparison
    if regime.regime_scores:
        lines.append("")
        score_parts = []
        for label, score in sorted(regime.regime_scores.items(), key=lambda x: -x[1]):
            name = label.replace("_", " ").title()
            marker = " *" if label == regime.regime.value else ""
            score_parts.append(f"{name}: {score:.1f}{marker}")
        lines.append(f"[dim]Scores: {' | '.join(score_parts)}[/dim]")

    console.print(Panel(
        "\n".join(lines),
        title="[bold]Market Regime[/bold]",
        box=box.ROUNDED,
        expand=True,
        padding=(0, 2),
    ))


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
    lines.append(f"  Trend (% above 200 MA): {assessment.pct_above_ma:>7.1f}%")
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
        f"[dim]Period: {result.period}  |  Strategy: {result.strategy_name}"
        + (f"  |  Objective: [bold]{cfg.active_objective}[/bold]" if cfg.active_objective else "")
        + "[/dim]",
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

    # ── Market Regime Panel ─────────────────────────────────────────────────
    if result.regime is not None:
        _render_regime_panel(result.regime)

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
    ]

    # Show EOD flatten count if any trades used that exit reason
    eod_count = sum(1 for t in result.trades if t.exit_reason == "eod_flatten")
    if eod_count > 0:
        metrics.append(("  EOD Flattened", f"{eod_count}", "cyan"))

    metrics += [
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
        config_rows.append(("(Fallback) LONG when score >", f"{thresholds.get('hold_below', 6.0)}"))
    else:
        config_rows.append(("Signal: SHORT when score <=", f"{thresholds.get('short_below', 3.5)}"))
        config_rows.append(("Signal: HOLD when score <=", f"{thresholds.get('hold_below', 6.0)}"))
        config_rows.append(("Signal: LONG when score >", f"{thresholds.get('hold_below', 6.0)}"))

    # ── Pattern-Indicator Combination ───────────────────────────────────
    combo_mode = strat_cfg.get("combination_mode", "weighted")
    config_rows.append(("Score Combination", combo_mode))
    if combo_mode == "weighted":
        ind_w = float(strat_cfg.get("indicator_weight", 0.7))
        pat_w = float(strat_cfg.get("pattern_weight", 0.3))
        config_rows.append(("  Indicator Weight", f"{ind_w:.0%}"))
        config_rows.append(("  Pattern Weight", f"{pat_w:.0%}"))
    elif combo_mode == "boost":
        boost_str = float(strat_cfg.get("boost_strength", 0.5))
        dead_zone = float(strat_cfg.get("boost_dead_zone", 0.3))
        config_rows.append(("  Boost Strength", f"{boost_str}"))
        config_rows.append(("  Boost Dead Zone", f"±{dead_zone} around 5.0"))
    else:
        # gate mode
        config_rows.append(("  Gate Ind. LONG >", f"{strat_cfg.get('gate_indicator_min', 5.5)}"))
        config_rows.append(("  Gate Ind. SHORT <", f"{strat_cfg.get('gate_indicator_max', 4.5)}"))
        config_rows.append(("  Gate Pat. LONG >", f"{strat_cfg.get('gate_pattern_min', 5.5)}"))
        config_rows.append(("  Gate Pat. SHORT <", f"{strat_cfg.get('gate_pattern_max', 4.5)}"))

    config_rows += [
        ("Position Sizing", strat_cfg.get("position_sizing", "fixed")),
        ("Fixed Quantity", f"{strat_cfg.get('fixed_quantity', 100)} shares"),
        ("Stop Loss", f"{strat_cfg.get('stop_loss_pct', 0.05) * 100:.1f}%"),
        ("Take Profit", f"{strat_cfg.get('take_profit_pct', 0.15) * 100:.1f}%"),
        ("Rebalance Interval", f"every {strat_cfg.get('rebalance_interval', 5)} bars"),
        ("EOD Flatten", "[bold green]ON[/bold green]" if strat_cfg.get("flatten_eod", False) else "[dim]OFF[/dim]"),
        ("Slippage", f"{bt_cfg.get('slippage_pct', 0.001) * 100:.2f}%"),
        ("Commission (flat)", f"${bt_cfg.get('commission_per_trade', 0.0):.2f} / leg"),
        ("Commission (%)", f"{bt_cfg.get('commission_pct', 0.0) * 100:.3f}% / leg"),
        ("Commission Mode", bt_cfg.get("commission_mode", "additive")),
        ("Warmup Bars", f"{result.warmup_bars}"
            + (f" [dim](configured {bt_cfg.get('warmup_bars', 200)}, clamped)[/dim]"
               if result.warmup_bars != int(bt_cfg.get("warmup_bars", 200)) else "")),
    ]

    # Regime adaptation status
    if result.regime is not None:
        regime_label = result.regime.label
        sub_label = f" ({result.regime.sub_type_label})" if result.regime.sub_type_label else ""
        config_rows.append(("Regime Adaptation", f"[bold]{regime_label}{sub_label}[/bold]"))

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
        trade_table.add_column("Entry Reason", style="dim", min_width=20, no_wrap=True)
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
                t.entry_reason,
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
                    "", "", "", "", "", "", "", "", "",
                )
                trade_table.add_section()

        console.print(trade_table)
    else:
        console.print("\n  [dim]No trades were executed during the backtest period.[/dim]")

    # ── Significant Patterns Timeline ───────────────────────────────────────
    if result.significant_patterns:
        _render_significant_patterns(result)

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

    # ── Score Legend ──────────────────────────────────────────────────────────
    color_thresholds = disp_cfg.get("color_thresholds", {})
    bearish_max = float(color_thresholds.get("bearish_max", 3.5))
    neutral_max = float(color_thresholds.get("neutral_max", 6.5))
    legend = (
        f"  Score legend:  "
        f"[red]0 – {bearish_max:.1f} Bearish[/red]  "
        f"[yellow]{bearish_max:.1f} – {neutral_max:.1f} Neutral[/yellow]  "
        f"[green]{neutral_max:.1f} – 10 Bullish[/green]"
    )
    console.print(legend)

    # ── Disclaimer ──────────────────────────────────────────────────────────
    console.print()
    console.print(
        "  [dim italic]Backtest results are hypothetical and do not guarantee future "
        "performance. Past results are not indicative of future returns.[/dim italic]"
    )
    console.print()
