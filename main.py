"""
main.py — CLI entry point for the Stock Technical Analysis tool.

Usage:
    python main.py AAPL                          # 6-month analysis (default)
    python main.py TSLA --period 1y              # 1-year analysis
    python main.py MSFT --period 1mo             # 1-month analysis
    python main.py AAPL --start 2016-01-01       # custom date range (~10 years)
    python main.py AAPL --start 2020-01-01 --end 2023-12-31
    python main.py AAPL --indicators rsi,macd    # run only specific indicators
    python main.py AAPL --config my_config.yaml  # use custom config file
    python main.py AAPL --backtest --period 2y   # run backtest with score strategy
    python main.py AAPL --backtest --start 2016-01-01  # backtest over ~10 years
    python main.py AAPL --backtest --mode long_only    # force long-only mode
    python main.py AAPL --objective long_term          # use long-term indicator presets
    python main.py AAPL --objective short_term -b      # short-term backtest
    python main.py AAPL -o day_trading -b -i 5m --start 2025-12-23  # day trading
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
  python main.py AAPL --start 2016-01-01              # ~10 years of data
  python main.py AAPL --start 2020-01-01 --end 2023-12-31
  python main.py AAPL --backtest --period 2y
  python main.py AAPL --backtest --start 2016-01-01   # backtest over ~10 years
  python main.py AAPL --backtest --mode long_only      # force long-only
  python main.py AAPL --backtest --mode auto           # auto-detect mode
  python main.py AAPL --objective long_term            # long-term indicator presets
  python main.py AAPL --objective short_term -b -p 6mo # short-term backtest
  python main.py AAPL -o day_trading -b -i 5m --start 2025-12-23  # day trading
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
        help=(
            "Bar interval. "
            "Daily+: 1d 5d 1wk 1mo 3mo  |  "
            "Intraday: 1m 2m 5m 15m 30m 60m 90m 1h  (default: 1d)"
        ),
    )
    parser.add_argument(
        "--indicators",
        metavar="LIST",
        help=(
            "Comma-separated list of indicators to run. "
            "e.g. rsi,macd,adx   (default: all)"
        ),
    )

    # Date range (alternative to --period)
    parser.add_argument(
        "--start", "-s",
        metavar="DATE",
        help=(
            "Start date in YYYY-MM-DD format (e.g. 2016-02-20). "
            "Overrides --period when specified."
        ),
    )
    parser.add_argument(
        "--end", "-e",
        metavar="DATE",
        help=(
            "End date in YYYY-MM-DD format (default: today). "
            "Only used when --start is specified."
        ),
    )

    # Backtest mode
    parser.add_argument(
        "--backtest", "-b",
        action="store_true",
        help="Run a backtest using the score-based strategy instead of (or in addition to) analysis",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["auto", "long_short", "long_only", "hold_only"],
        default=None,
        metavar="MODE",
        help=(
            "Trading mode for backtest: auto (detect from data), "
            "long_short, long_only, hold_only. "
            "Overrides config.yaml suitability.mode_override. "
            "Only used with --backtest."
        ),
    )

    # Objective preset
    parser.add_argument(
        "--objective", "-o",
        metavar="NAME",
        help=(
            "Trading objective preset to apply "
            "(e.g. long_term, short_term, day_trading). "
            "Overrides indicator periods, strategy thresholds, weights, etc. "
            "Define custom presets in the 'objectives' section of config.yaml."
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

    # ── Apply objective preset ────────────────────────────────────────────────
    if args.objective:
        try:
            cfg.apply_objective(args.objective)
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            available = cfg.available_objectives()
            if available:
                console.print(f"[dim]Available objectives: {', '.join(available)}[/dim]")
            sys.exit(1)
        obj_desc = cfg.section("objectives").get(args.objective, {}).get("description", "")
        console.print(
            f"[dim]Objective: [bold]{args.objective}[/bold]"
            + (f" — {obj_desc}" if obj_desc else "")
            + "[/dim]"
        )

    # ── Validate interval vs objective ────────────────────────────────────────
    # day_trading requires an intraday interval — daily bars make no sense
    # with 1.5% stops and EOD flattening.
    intraday_intervals = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
    is_intraday = args.interval.lower().strip() in intraday_intervals

    if args.objective == "day_trading" and not is_intraday:
        console.print(
            "[red]Error:[/red] The [bold]day_trading[/bold] objective requires an intraday interval.\n"
            "  Add [bold]-i 5m[/bold] (or 1m, 15m, 30m, 1h) to your command.\n\n"
            "  Example: [dim]python main.py TSLA -o day_trading -b -i 5m --start 2025-12-23[/dim]\n\n"
            "  [dim]yfinance intraday limits:[/dim]\n"
            "  [dim]  1m = last 7 days  |  5m/15m/30m = last 60 days  |  1h = last 730 days[/dim]"
        )
        sys.exit(1)

    # Warn about yfinance intraday data limits when using --period
    if is_intraday and not args.start:
        _max_period: dict[str, list[str]] = {
            "1m": ["1mo"],
            "2m": ["1mo"],
            "5m": ["1mo", "3mo"],
            "15m": ["1mo", "3mo"],
            "30m": ["1mo", "3mo"],
            "60m": ["1mo", "3mo", "6mo", "1y", "2y"],
            "90m": ["1mo", "3mo", "6mo", "1y", "2y"],
            "1h": ["1mo", "3mo", "6mo", "1y", "2y"],
        }
        allowed = _max_period.get(args.interval.lower().strip(), [])
        if allowed and args.period.lower().strip() not in allowed:
            _limits = "1m=7d, 5m/15m/30m=60d, 1h=730d"
            console.print(
                f"[red]Error:[/red] Period [bold]{args.period}[/bold] is too long for "
                f"[bold]{args.interval}[/bold] interval data.\n"
                f"  yfinance limits: {_limits}\n"
                f"  Valid periods for {args.interval}: {', '.join(allowed)}\n\n"
                f"  [dim]Tip: Use --start DATE instead of --period for precise date ranges.[/dim]"
            )
            sys.exit(1)

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

    # ── Resolve date range vs period ──────────────────────────────────────────
    start_date: str | None = args.start
    end_date: str | None = args.end

    if start_date:
        # Validate date format
        from datetime import datetime
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            console.print(f"[red]Error:[/red] --start must be YYYY-MM-DD, got '{start_date}'")
            sys.exit(1)
        if end_date:
            try:
                datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                console.print(f"[red]Error:[/red] --end must be YYYY-MM-DD, got '{end_date}'")
                sys.exit(1)
        period_label = f"{start_date} → {end_date or 'today'}"
    else:
        if end_date:
            console.print("[red]Error:[/red] --end requires --start to be specified")
            sys.exit(1)
        period_label = args.period

    # ── Run analysis ──────────────────────────────────────────────────────────
    console.print(f"\n[dim]Fetching data for [bold]{ticker}[/bold] ({period_label})...[/dim]")

    provider = YahooFinanceProvider()

    # ── Backtest mode ─────────────────────────────────────────────────────────
    if args.backtest:
        from engine.score_strategy import ScoreBasedStrategy
        from engine.backtest import BacktestEngine
        from engine.suitability import SuitabilityAnalyzer, TradingMode
        from display.backtest_terminal import render_backtest

        # ── Determine trading mode ────────────────────────────────────────
        # Priority: --mode CLI flag > config.yaml mode_override > auto-detect
        # Note: suitability detection is designed for daily data; intraday
        # intervals bypass auto-detection and default to long_short.
        mode_str = args.mode  # CLI flag (None if not provided)
        if mode_str is None:
            mode_str = cfg.section("suitability").get("mode_override", "auto")

        # If a specific mode is forced (not "auto"), use it directly
        forced = mode_str != "auto"
        assessment = None

        if forced:
            mode_map = {
                "long_short": TradingMode.LONG_SHORT,
                "long_only": TradingMode.LONG_ONLY,
                "hold_only": TradingMode.HOLD_ONLY,
            }
            trading_mode = mode_map[mode_str]
        elif is_intraday:
            # Suitability thresholds are calibrated for daily bars;
            # intraday data has naturally lower ATR% and per-bar volume.
            # Default to long_short for intraday trading.
            trading_mode = TradingMode.LONG_SHORT
            console.print(
                f"[dim]Suitability check skipped (intraday interval: {args.interval}). "
                f"Using [bold]LONG SHORT[/bold] mode.[/dim]"
            )
        else:
            # Auto-detect: need to fetch data first for suitability analysis
            console.print(f"[dim]Analyzing suitability for [bold]{ticker}[/bold]...[/dim]")
            try:
                suit_df = provider.fetch(
                    ticker,
                    period=args.period if not start_date else None,
                    interval=args.interval,
                    start=start_date,
                    end=end_date,
                )
            except ValueError as exc:
                console.print(f"[red]Error:[/red] {exc}")
                sys.exit(1)

            suit_analyzer = SuitabilityAnalyzer(cfg)
            assessment = suit_analyzer.assess(suit_df)
            trading_mode = assessment.mode

        # ── If hold_only, show assessment and exit (no backtest) ──────────
        if trading_mode == TradingMode.HOLD_ONLY and not forced:
            from display.backtest_terminal import render_suitability
            assert assessment is not None
            console.print()
            render_suitability(assessment, ticker)
            console.print(
                "\n  [yellow]Stock detected as unsuitable for active trading.[/yellow]"
                "\n  [dim]Run without --backtest for analysis-only, "
                "or use --mode long_short / --mode long_only to force a mode.[/dim]\n"
            )
            return

        # ── Show mode selection ───────────────────────────────────────────
        mode_label = trading_mode.value.replace("_", " ").upper()
        if forced:
            console.print(f"[dim]Trading mode: [bold]{mode_label}[/bold] (forced via {'CLI' if args.mode else 'config'})[/dim]")
        else:
            console.print(f"[dim]Trading mode: [bold]{mode_label}[/bold] (auto-detected)[/dim]")

        # ── Create strategy and engine ────────────────────────────────────
        strategy = ScoreBasedStrategy(
            params=cfg.section("strategy"),
            trading_mode=trading_mode,
        )

        engine = BacktestEngine(
            data_provider=provider,
            strategy=strategy,
            cfg=cfg,
            trading_mode=trading_mode,
        )

        console.print(
            f"[dim]Running backtest with {strategy.name} "
            f"(rebalance every {cfg.section('strategy').get('rebalance_interval', 5)} bars)...[/dim]"
        )

        try:
            bt_result = engine.run(
                ticker,
                period=args.period if not start_date else None,
                interval=args.interval,
                start=start_date,
                end=end_date,
            )
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            sys.exit(1)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Unexpected error:[/red] {exc}")
            raise

        render_backtest(bt_result, cfg, assessment=assessment)
        return

    # ── Standard analysis mode ────────────────────────────────────────────────
    analyzer = Analyzer(cfg, provider, only_indicators=only_indicators)

    try:
        result = analyzer.run(
            ticker,
            period=args.period if not start_date else None,
            interval=args.interval,
            start=start_date,
            end=end_date,
        )
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
