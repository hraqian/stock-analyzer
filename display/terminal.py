"""
display/terminal.py — Rich-based terminal output for analysis results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

if TYPE_CHECKING:
    from analysis.analyzer import AnalysisResult
    from config import Config

console = Console()


def _score_color(score: float, cfg: dict) -> str:
    bearish_max = float(cfg.get("bearish_max", 3.5))
    neutral_max = float(cfg.get("neutral_max", 6.5))
    if score <= bearish_max:
        return "red"
    if score <= neutral_max:
        return "yellow"
    return "green"


def _score_bar(score: float, width: int = 10) -> str:
    """Visual bar like ████░░░░░░ representing 0-10."""
    filled = round(score / 10 * width)
    return "█" * filled + "░" * (width - filled)


def _format_market_cap(mc: float | None) -> str:
    if mc is None:
        return "N/A"
    if mc >= 1e12:
        return f"${mc / 1e12:.2f}T"
    if mc >= 1e9:
        return f"${mc / 1e9:.2f}B"
    if mc >= 1e6:
        return f"${mc / 1e6:.2f}M"
    return f"${mc:,.0f}"


def render(result: "AnalysisResult", cfg: "Config") -> None:
    """Render the full analysis report to the terminal."""
    disp_cfg = cfg.section("display")
    color_thresholds = disp_cfg.get("color_thresholds", {})
    price_dp = int(disp_cfg.get("price_decimal_places", 2))
    score_dp = int(disp_cfg.get("score_decimal_places", 1))

    info = result.info
    price = info.get("current_price") or float(result.df["close"].iloc[-1])
    name = info.get("name", result.ticker)
    overall = result.composite["overall"]
    overall_color = _score_color(overall, color_thresholds)

    # ── Header ──────────────────────────────────────────────────────────────
    meta_parts = []
    if info.get("sector") and info["sector"] != "N/A":
        meta_parts.append(f"Sector: {info['sector']}")
    if info.get("exchange") and info["exchange"] != "N/A":
        meta_parts.append(f"Exchange: {info['exchange']}")
    if info.get("currency"):
        meta_parts.append(f"Currency: {info['currency']}")
    mc = _format_market_cap(info.get("market_cap"))
    if mc != "N/A":
        meta_parts.append(f"Mkt Cap: {mc}")

    header_lines = []
    header_lines.append(
        f"[bold white]{name}[/bold white]  "
        f"[bold cyan]({result.ticker})[/bold cyan]  "
        f"[bold green]${price:.{price_dp}f}[/bold green]  "
        f"[dim]Period: {result.period}[/dim]"
    )
    if cfg.active_objective:
        header_lines[0] += f"  [dim]Objective: [bold]{cfg.active_objective}[/bold][/dim]"
    if meta_parts:
        header_lines.append(f"[dim]{'  |  '.join(meta_parts)}[/dim]")

    console.print()
    console.print(Panel(
        "\n".join(header_lines),
        box=box.DOUBLE_EDGE,
        style="bold",
        expand=True,
        padding=(0, 2),
    ))

    # ── Indicators Table ─────────────────────────────────────────────────────
    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold cyan",
        expand=True,
        padding=(0, 1),
        title="[bold]Technical Indicators[/bold]",
        title_style="bold white",
    )
    table.add_column("Indicator",    style="bold white", min_width=18, no_wrap=True)
    table.add_column("Value",        style="white",      min_width=26, no_wrap=True)
    table.add_column("Detail",       style="dim white",  ratio=3,      no_wrap=True)
    table.add_column("Score / Bar",  min_width=20,       no_wrap=True)

    for r in result.indicator_results:
        color = _score_color(r.score, color_thresholds)
        score_str = f"{r.score:.{score_dp}f}"
        bar = _score_bar(r.score)

        if r.error:
            table.add_row(
                r.name,
                "[red]ERROR[/red]",
                f"[red]{r.error[:50]}[/red]",
                f"[dim]{score_str}[/dim]",
            )
        else:
            table.add_row(
                r.name,
                r.display.get("value_str", ""),
                r.display.get("detail_str", ""),
                f"[{color}]{score_str}  {bar}[/{color}]",
            )

    # Overall score row
    table.add_section()
    overall_bar = _score_bar(overall)
    table.add_row(
        "[bold]OVERALL SCORE[/bold]",
        "",
        f"[dim]({result.composite['n_scored']} indicators weighted)[/dim]",
        f"[bold {overall_color}]{overall:.{score_dp}f}  {overall_bar}[/bold {overall_color}]",
    )

    console.print(table)

    # ── Support & Resistance ─────────────────────────────────────────────────
    sr_table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold cyan",
        expand=True,
        padding=(0, 1),
        title="[bold]Support & Resistance[/bold]",
        title_style="bold white",
    )
    sr_table.add_column("Support Levels",    style="green",  min_width=22)
    sr_table.add_column("Source",            style="dim",    min_width=14)
    sr_table.add_column("  ",                min_width=3)
    sr_table.add_column("Resistance Levels", style="red",    min_width=22)
    sr_table.add_column("Source",            style="dim",    min_width=14)

    max_rows = max(len(result.support_levels), len(result.resistance_levels))
    for i in range(max_rows):
        # Support
        if i < len(result.support_levels):
            sl = result.support_levels[i]
            s_price = f"${sl.price:.{price_dp}f}"
            if sl.label:
                s_price += f"  [{sl.label}]"
            s_source = sl.source
        else:
            s_price = ""
            s_source = ""

        # Resistance
        if i < len(result.resistance_levels):
            rl = result.resistance_levels[i]
            r_price = f"${rl.price:.{price_dp}f}"
            if rl.label:
                r_price += f"  [{rl.label}]"
            r_source = rl.source
        else:
            r_price = ""
            r_source = ""

        sr_table.add_row(s_price, s_source, "│", r_price, r_source)

    console.print(sr_table)

    # ── Score Legend ─────────────────────────────────────────────────────────
    bearish_max = float(color_thresholds.get("bearish_max", 3.5))
    neutral_max = float(color_thresholds.get("neutral_max", 6.5))
    legend = (
        f"  Score legend:  "
        f"[red]0 – {bearish_max:.1f} Bearish[/red]  "
        f"[yellow]{bearish_max:.1f} – {neutral_max:.1f} Neutral[/yellow]  "
        f"[green]{neutral_max:.1f} – 10 Bullish[/green]"
    )
    console.print(legend)
    console.print()

    # ── Disclaimer ───────────────────────────────────────────────────────────
    if disp_cfg.get("show_disclaimer", True):
        console.print(
            "  [dim italic]For informational purposes only. "
            "Not financial advice. Do your own research.[/dim italic]"
        )
        console.print()
