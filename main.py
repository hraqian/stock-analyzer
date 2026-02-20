"""
main.py — CLI entry point for the Stock Technical Analysis tool.

Usage:
    python main.py AAPL                          # 6-month analysis (default)
    python main.py TSLA --period 1y              # 1-year analysis
    python main.py MSFT --period 1mo             # 1-month analysis
    python main.py AAPL --indicators rsi,macd    # run only specific indicators
    python main.py AAPL --config my_config.yaml  # use custom config file
    python main.py --generate-config             # write a fresh config.yaml
    python main.py --validate-config             # check config.yaml for errors
    python main.py --list-indicators             # list all available indicators
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stock_analyzer",
        description="Technical analysis of any stock — scored 0-10 per indicator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py AAPL
  python main.py TSLA --period 1y
  python main.py MSFT --period 3mo --indicators rsi,macd,adx
  python main.py --generate-config
  python main.py --validate-config
  python main.py --list-indicators
        """,
    )

    # Positional — ticker (optional so --generate-config works alone)
    parser.add_argument(
        "ticker",
        nargs="?",
        metavar="TICKER",
        help="Stock ticker symbol, e.g. AAPL, TSLA, MSFT",
    )

    # Analysis options
    parser.add_argument(
        "--period", "-p",
        default="6mo",
        metavar="PERIOD",
        help=(
            "Data period to fetch. "
            "Options: 1mo 3mo 6mo 1y 2y 5y ytd max  (default: 6mo)"
        ),
    )
    parser.add_argument(
        "--interval", "-i",
        default="1d",
        metavar="INTERVAL",
        help="Bar interval: 1d 1wk 1mo  (default: 1d)",
    )
    parser.add_argument(
        "--indicators",
        metavar="LIST",
        help=(
            "Comma-separated list of indicators to run. "
            "e.g. rsi,macd,adx   (default: all)"
        ),
    )

    # Config options
    parser.add_argument(
        "--config", "-c",
        metavar="PATH",
        help="Path to a custom config.yaml file",
    )
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Generate a fresh default config.yaml in the current directory and exit",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate config.yaml and report any issues, then exit",
    )

    # Utility
    parser.add_argument(
        "--list-indicators",
        action="store_true",
        help="List all available indicator keys and exit",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ── Add project root to path so imports work when run from any directory ──
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # ── Lazy imports (after sys.path is set) ──────────────────────────────────
    from config import Config
    from data.yahoo import YahooFinanceProvider
    from analysis.analyzer import Analyzer
    from display.terminal import render, console

    # ── --generate-config ─────────────────────────────────────────────────────
    if args.generate_config:
        Config.generate_default("config.yaml")
        return

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = Config.load(args.config)
    if cfg.path:
        console.print(f"[dim]Config: {cfg.path}[/dim]")
    else:
        console.print("[dim]Config: using built-in defaults (no config.yaml found)[/dim]")

    # ── --validate-config ─────────────────────────────────────────────────────
    if args.validate_config:
        errors = cfg.validate()
        if errors:
            console.print("[red]Config validation errors:[/red]")
            for e in errors:
                console.print(f"  [red]• {e}[/red]")
            sys.exit(1)
        else:
            console.print("[green]Config is valid.[/green]")
        return

    # ── --list-indicators ─────────────────────────────────────────────────────
    if args.list_indicators:
        from indicators.registry import IndicatorRegistry
        registry = IndicatorRegistry(cfg)
        console.print("\n[bold cyan]Available indicators:[/bold cyan]")
        for key in registry.indicator_names:
            console.print(f"  [white]{key}[/white]")
        console.print()
        return

    # ── Require ticker for analysis ───────────────────────────────────────────
    if not args.ticker:
        parser.print_help()
        sys.exit(1)

    ticker = args.ticker.upper().strip()
    only_indicators: list[str] | None = None
    if args.indicators:
        only_indicators = [s.strip().lower() for s in args.indicators.split(",") if s.strip()]

    # ── Run analysis ──────────────────────────────────────────────────────────
    console.print(f"\n[dim]Fetching data for [bold]{ticker}[/bold] ({args.period})...[/dim]")

    provider = YahooFinanceProvider()
    analyzer = Analyzer(cfg, provider, only_indicators=only_indicators)

    try:
        result = analyzer.run(ticker, period=args.period, interval=args.interval)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Unexpected error:[/red] {exc}")
        raise

    # ── Display ───────────────────────────────────────────────────────────────
    render(result, cfg)


if __name__ == "__main__":
    main()
