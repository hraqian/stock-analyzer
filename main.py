"""
main.py — CLI entry point for the Stock Technical Analysis tool.

Usage:
    python main.py AAPL                          # 6-month analysis (default)
    python main.py TSLA --period 1y              # 1-year analysis
    python main.py MSFT --period 1mo             # 1-month analysis
    python main.py AAPL --start <10y-ago>        # custom date range (~10 years)
    python main.py AAPL --start <5y-ago> --end <2y-ago>  # custom window
    python main.py AAPL --indicators rsi,macd    # run only specific indicators
    python main.py AAPL --config my_config.yaml  # use custom config file
    python main.py AAPL --backtest --period 2y   # run backtest with score strategy
    python main.py AAPL --backtest --start <10y-ago>     # backtest over ~10 years
    python main.py AAPL --backtest --mode long_only      # force long-only mode
    python main.py AAPL --objective long_term            # use long-term indicator presets
    python main.py AAPL --objective short_term -b        # short-term backtest
    python main.py AAPL -o day_trading -b -i 5m --start <recent>  # day trading
    python main.py AAPL --dca --period 5y               # DCA backtest (5 years)
    python main.py AAPL --dca --dca-mode pure            # pure DCA (no dip weighting)
    python main.py AAPL --dca --dca-amount 1000 --dca-frequency weekly  # custom DCA
    python main.py --generate-config             # write a fresh config.yaml
    python main.py --validate-config             # check config.yaml for errors
    python main.py --list-indicators             # list all available indicators
    python main.py --list-patterns               # list all available pattern detectors

(Dates above are placeholders — actual examples are computed dynamically at runtime.)
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path


def _example_dates() -> dict[str, str]:
    """Compute dynamic example dates so help text never goes stale."""
    today = datetime.date.today()
    return {
        "ten_years_ago": (today - datetime.timedelta(days=365 * 10)).strftime("%Y-%m-%d"),
        "five_years_ago": (today - datetime.timedelta(days=365 * 5)).strftime("%Y-%m-%d"),
        "two_years_ago": (today - datetime.timedelta(days=365 * 2)).strftime("%Y-%m-%d"),
        "recent_intraday": (today - datetime.timedelta(days=5)).strftime("%Y-%m-%d"),
        "today": today.strftime("%Y-%m-%d"),
    }


def build_parser() -> argparse.ArgumentParser:
    d = _example_dates()
    parser = argparse.ArgumentParser(
        prog="stock_analyzer",
        description="Technical analysis of any stock — scored 0-10 per indicator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python main.py AAPL
  python main.py TSLA --period 1y
  python main.py MSFT --period 3mo --indicators rsi,macd,adx
  python main.py AAPL --start {d['ten_years_ago']}              # ~10 years of data
  python main.py AAPL --start {d['five_years_ago']} --end {d['two_years_ago']}
  python main.py AAPL --backtest --period 2y
  python main.py AAPL --backtest --start {d['ten_years_ago']}   # backtest over ~10 years
  python main.py AAPL --backtest --mode long_only      # force long-only
  python main.py AAPL --backtest --mode auto           # auto-detect mode
  python main.py AAPL --objective long_term            # long-term indicator presets
  python main.py AAPL --objective short_term -b -p 6mo # short-term backtest
  python main.py AAPL -o day_trading -b -i 5m --start {d['recent_intraday']}  # day trading
  python main.py AAPL --dca --period 5y            # DCA backtest (5 years, default dip-weighted)
  python main.py AAPL --dca --dca-mode pure         # pure DCA, no dip weighting
  python main.py AAPL --dca --dca-amount 1000       # DCA with $1000/period
  python main.py --generate-config
  python main.py --validate-config
  python main.py --list-indicators
  python main.py --list-patterns

Streamlit dashboard:
  streamlit run dashboard.py
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
            f"Start date in YYYY-MM-DD format (e.g. {d['ten_years_ago']}). "
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

    # DCA backtest mode
    parser.add_argument(
        "--dca",
        action="store_true",
        help=(
            "Run a Dollar Cost Averaging backtest. "
            "Mode is controlled by config.yaml dca.mode "
            "(pure, dip_weighted, score_integrated). "
            "Combines with --period (default 5y) and --start/--end."
        ),
    )
    parser.add_argument(
        "--dca-mode",
        choices=["pure", "dip_weighted", "score_integrated"],
        default=None,
        metavar="MODE",
        help="DCA mode override (default: from config.yaml dca.mode)",
    )
    parser.add_argument(
        "--dca-amount",
        type=float,
        default=None,
        metavar="AMOUNT",
        help="DCA base amount in dollars (default: from config.yaml dca.base_amount)",
    )
    parser.add_argument(
        "--dca-frequency",
        choices=["daily", "weekly", "biweekly", "monthly"],
        default=None,
        metavar="FREQ",
        help="DCA purchase frequency (default: from config.yaml dca.frequency)",
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
    parser.add_argument(
        "--list-patterns",
        action="store_true",
        help="List all available pattern detector keys and exit",
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
        _recent = (datetime.date.today() - datetime.timedelta(days=5)).strftime("%Y-%m-%d")
        console.print(
            "[red]Error:[/red] The [bold]day_trading[/bold] objective requires an intraday interval.\n"
            "  Add [bold]-i 5m[/bold] (or 1m, 15m, 30m, 1h) to your command.\n\n"
            f"  Example: [dim]python main.py TSLA -o day_trading -b -i 5m --start {_recent}[/dim]\n\n"
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

    # ── --list-patterns ───────────────────────────────────────────────────────
    if args.list_patterns:
        from patterns.registry import PatternRegistry
        registry = PatternRegistry(cfg)
        console.print("\n[bold cyan]Available pattern detectors:[/bold cyan]")
        for key in registry.pattern_names:
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
        try:
            datetime.datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            console.print(f"[red]Error:[/red] --start must be YYYY-MM-DD, got '{start_date}'")
            sys.exit(1)
        if end_date:
            try:
                datetime.datetime.strptime(end_date, "%Y-%m-%d")
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

    # ── DCA backtest mode ─────────────────────────────────────────────────────
    if args.dca:
        from engine.dca import DCABacktester
        from rich.table import Table

        # Build overrides from CLI flags
        dca_overrides: dict = {}
        if args.dca_mode:
            dca_overrides["mode"] = args.dca_mode
        if args.dca_amount:
            dca_overrides["base_amount"] = args.dca_amount
        if args.dca_frequency:
            dca_overrides["frequency"] = args.dca_frequency

        # Default to 5y for DCA if no period/start specified
        dca_period = args.period if args.period != "6mo" else "5y"

        dca_bt = DCABacktester(cfg=cfg, overrides=dca_overrides)
        mode_label = dca_bt.mode.replace("_", " ").title()
        console.print(
            f"[dim]Running DCA backtest: [bold]{mode_label}[/bold] — "
            f"${dca_bt.base_amount:,.0f} {dca_bt.frequency} "
            f"({'DRIP on' if dca_bt.drip else 'DRIP off'})...[/dim]"
        )

        try:
            dca_result = dca_bt.run(
                ticker,
                period=dca_period if not start_date else "5y",
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

        # ── Render DCA results ────────────────────────────────────────────
        console.print(f"\n[bold cyan]DCA Backtest Results — {ticker}[/bold cyan]")
        console.print(f"[dim]{mode_label} | {dca_result.frequency.title()} | "
                       f"{dca_result.period} | {dca_result.num_purchases} purchases[/dim]\n")

        # Summary table
        table = Table(show_header=True, header_style="bold")
        table.add_column("Metric", style="white")
        table.add_column("Value", justify="right", style="cyan")

        table.add_row("Total Invested", f"${dca_result.total_invested:,.2f}")
        table.add_row("Final Value", f"${dca_result.final_value:,.2f}")
        net = dca_result.final_value - dca_result.total_invested
        color = "green" if net >= 0 else "red"
        table.add_row("Net Profit", f"[{color}]${net:,.2f}[/{color}]")
        color = "green" if dca_result.total_return_pct >= 0 else "red"
        table.add_row("Total Return", f"[{color}]{dca_result.total_return_pct:+.2f}%[/{color}]")
        color = "green" if dca_result.annualized_return_pct >= 0 else "red"
        table.add_row("Annualized Return", f"[{color}]{dca_result.annualized_return_pct:+.2f}%[/{color}]")
        table.add_row("Max Drawdown", f"[red]-{dca_result.max_drawdown_pct:.2f}%[/red]")
        table.add_row("Avg Cost Basis", f"${dca_result.avg_cost_basis:,.2f}")
        table.add_row("Current Price", f"${dca_result.current_price:,.2f}")
        table.add_row("Total Shares", f"{dca_result.total_shares:.4f}")
        table.add_row("Purchases", f"{dca_result.num_purchases}")
        table.add_row("Dip Purchases", f"{dca_result.num_dip_purchases}")
        table.add_row("Avg Multiplier", f"{dca_result.avg_multiplier:.2f}x")
        if dca_result.total_commissions > 0:
            table.add_row("Total Commissions", f"${dca_result.total_commissions:,.2f}")
            comm_pct = (
                dca_result.total_commissions / dca_result.total_invested * 100
                if dca_result.total_invested > 0 else 0.0
            )
            table.add_row("Commission Drag", f"{comm_pct:.2f}%")
        if dca_result.total_dividends > 0:
            table.add_row("Dividends Reinvested", f"${dca_result.total_dividends:,.2f}")
            table.add_row("DRIP Shares", f"{dca_result.drip_shares:.4f}")
        table.add_row("Best Purchase Return", f"{dca_result.best_purchase_return_pct:+.2f}%")
        table.add_row("Worst Purchase Return", f"{dca_result.worst_purchase_return_pct:+.2f}%")

        console.print(table)

        # Dip purchase details
        dip_buys = [p for p in dca_result.purchases if p.multiplier > 1.0]
        if dip_buys:
            console.print(f"\n[bold]Dip Purchases ({len(dip_buys)}):[/bold]")
            dip_table = Table(show_header=True, header_style="bold dim")
            dip_table.add_column("Date")
            dip_table.add_column("Price", justify="right")
            dip_table.add_column("Dip %", justify="right")
            dip_table.add_column("Tier")
            dip_table.add_column("Multiplier", justify="right")
            dip_table.add_column("Amount", justify="right")
            for p in dip_buys:
                dip_table.add_row(
                    p.date,
                    f"${p.price:,.2f}",
                    f"{p.dip_pct:.1f}%",
                    p.tier.replace("_", " ").title(),
                    f"{p.multiplier:.1f}x",
                    f"${p.amount:,.2f}",
                )
            console.print(dip_table)

        console.print()
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
