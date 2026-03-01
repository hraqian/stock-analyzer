#!/usr/bin/env python3
"""
scan.py — CLI stock-universe scanner.

Scans a stock universe through the full analysis + strategy pipeline and
ranks the top BUY and SELL candidates.  Also supports DCA mode to rank
stocks by Dollar-Cost Averaging attractiveness.

Usage:
    python scan.py                          # scan Dow 30 (fast default)
    python scan.py --universe nasdaq100     # scan NASDAQ 100
    python scan.py --universe sp500         # scan S&P 500 (~10-15 min)
    python scan.py --universe dow30 --top 5 # show top 5 instead of 10
    python scan.py --period 1y              # use 1-year analysis period
    python scan.py --workers 12             # increase parallelism
    python scan.py --mode dca               # DCA attractiveness scan
    python scan.py --mode dca --all         # show all DCA results
    python scan.py --list                   # list available universes
    python scan.py --refresh                # refresh universe files from Wikipedia
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.universes import available as available_universes
from data.universes import load as load_universe
from scanner import Scanner, ScanResult, DCAScanner, DCAScanResult


# ─────────────────────────────────────────────────────────────────────────────
# Terminal formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
WHITE = "\033[37m"

SIGNAL_COLORS = {"BUY": GREEN, "SELL": RED, "HOLD": YELLOW}
CONFIDENCE_SYMBOLS = {"high": "***", "medium": "** ", "low": "*  "}


def _colored_signal(signal: str) -> str:
    color = SIGNAL_COLORS.get(signal, WHITE)
    return f"{color}{BOLD}{signal:4s}{RESET}"


def _confidence_bar(confidence: str) -> str:
    return CONFIDENCE_SYMBOLS.get(confidence, "   ")


def _print_table(title: str, results: list[ScanResult], color: str) -> None:
    """Print a ranked table of scan results."""
    if not results:
        print(f"\n{DIM}  No {title} signals found.{RESET}")
        return

    print(f"\n{color}{BOLD}  {'─' * 98}")
    print(f"  {title}")
    print(f"  {'─' * 98}{RESET}")
    print(
        f"  {DIM}{'#':>3}  {'Ticker':<7} {'Signal':6} {'Conf':4} "
        f"{'Score':>6} {'Ind':>5} {'Pat':>5} "
        f"{'Price':>10} {'Regime':<20} {'Sub-Type':<22}{RESET}"
    )
    print(f"  {DIM}{'─' * 98}{RESET}")

    for i, r in enumerate(results, 1):
        sig = _colored_signal(r.signal)
        conf = _confidence_bar(r.confidence)
        regime_display = r.regime_label or "—"
        sub_display = r.sub_type_label or "—"
        print(
            f"  {i:>3}  {BOLD}{r.ticker:<7}{RESET} {sig} {conf}  "
            f"{r.effective_score:>5.2f} {r.indicator_score:>5.2f} {r.pattern_score:>5.2f} "
            f"${r.price:>9.2f} {regime_display:<20} {sub_display:<22}"
        )


def _print_summary(scanner: Scanner, elapsed: float) -> None:
    """Print the scan summary."""
    s = scanner.summary()
    errors = scanner.errors()

    print(f"\n{BOLD}  Scan Summary{RESET}")
    print(f"  {'─' * 50}")
    print(f"  Universe:    {s['universe']} ({s['total_tickers']} tickers)")
    print(f"  Period:      {s['period']}")
    print(f"  Scanned:     {s['scanned']} ok, {s['errors']} errors")
    print(
        f"  Signals:     {GREEN}{s['buy_count']} BUY{RESET}  "
        f"{RED}{s['sell_count']} SELL{RESET}  "
        f"{YELLOW}{s['hold_count']} HOLD{RESET}"
    )
    print(f"  Elapsed:     {elapsed:.1f}s ({elapsed / max(s['total_tickers'], 1):.1f}s/ticker)")

    if errors:
        print(f"\n  {RED}Failed tickers:{RESET}")
        for r in errors:
            print(f"    {r.ticker}: {r.error[:80]}")


# ─────────────────────────────────────────────────────────────────────────────
# DCA scan output helpers
# ─────────────────────────────────────────────────────────────────────────────


def _print_dca_table(title: str, results: list[DCAScanResult], color: str) -> None:
    """Print a ranked table of DCA scan results."""
    if not results:
        print(f"\n{DIM}  No {title} found.{RESET}")
        return

    print(f"\n{color}{BOLD}  {'─' * 115}")
    print(f"  {title}")
    print(f"  {'─' * 115}{RESET}")
    print(
        f"  {DIM}{'#':>3}  {'Ticker':<7} {'DCA':>4} {'Conf':4} "
        f"{'Price':>10} {'Dip%':>5} {'Sigma':>5} "
        f"{'Tier':<12} {'Mult':>5} {'Regime':<12} "
        f"{'RSI':>5} {'BB%B':>5} {'Vol%':>5} {'Ind':>5}{RESET}"
    )
    print(f"  {DIM}{'─' * 115}{RESET}")

    for i, r in enumerate(results, 1):
        # DCA score coloring
        if r.dca_score >= 60:
            sc = GREEN
        elif r.dca_score >= 35:
            sc = YELLOW
        else:
            sc = DIM

        conf = _confidence_bar(r.confidence)

        # Tier coloring
        tier_colors = {
            "extreme_dip": RED,
            "strong_dip": YELLOW,
            "mild_dip": CYAN,
            "normal": DIM,
        }
        tc = tier_colors.get(r.tier, DIM)

        # Regime coloring
        regime_colors = {
            "bull": GREEN,
            "recovery": GREEN,
            "sideways": YELLOW,
            "bear": RED,
            "crisis": RED,
        }
        rc = regime_colors.get(r.regime, DIM)

        print(
            f"  {i:>3}  {BOLD}{r.ticker:<7}{RESET} "
            f"{sc}{r.dca_score:>4.0f}{RESET} {conf}  "
            f"${r.price:>9.2f} "
            f"{r.dip_pct:>5.1f} {r.dip_sigma:>4.1f}\u03c3 "
            f"{tc}{r.tier_label:<12}{RESET} "
            f"{r.multiplier:>4.1f}x "
            f"{rc}{r.regime_label:<12}{RESET} "
            f"{r.rsi:>5.0f} {r.bb_pctile:>4.0f}% "
            f"{r.volatility:>4.0f}% {r.composite_score:>5.1f}"
        )


def _print_dca_summary(scanner: DCAScanner, elapsed: float) -> None:
    """Print the DCA scan summary."""
    s = scanner.summary()
    errors = scanner.errors()

    print(f"\n{BOLD}  DCA Scan Summary{RESET}")
    print(f"  {'─' * 50}")
    print(f"  Universe:    {s['universe']} ({s['total_tickers']} tickers)")
    print(f"  Period:      {s['period']}")
    print(f"  Scanned:     {s['scanned']} ok, {s['errors']} errors")
    print(
        f"  DCA Buys:    {GREEN}{s['dca_buy_count']} opportunities{RESET}  "
        f"({s['non_buy_count']} non-buy)"
    )
    print(
        f"  Confidence:  {GREEN}{s['high_conf_count']} high{RESET}  "
        f"{YELLOW}{s['medium_conf_count']} medium{RESET}  "
        f"{DIM}{s['low_conf_count']} low{RESET}"
    )
    print(f"  Avg Score:   {s['avg_dca_score']:.0f}/100")
    print(f"  Elapsed:     {elapsed:.1f}s ({elapsed / max(s['total_tickers'], 1):.1f}s/ticker)")

    if errors:
        print(f"\n  {RED}Failed tickers:{RESET}")
        for r in errors:
            print(f"    {r.ticker}: {r.error[:80]}")


# ─────────────────────────────────────────────────────────────────────────────
# Universe refresh (scrapes Wikipedia for current constituents)
# ─────────────────────────────────────────────────────────────────────────────


def _refresh_universes() -> None:
    """Refresh universe files by scraping Wikipedia for current constituents."""
    try:
        import pandas as pd
    except ImportError:
        print("pandas is required for --refresh.  Install it: pip install pandas lxml")
        sys.exit(1)

    from io import StringIO

    from data.universes import universe_path

    sources = {
        "sp500": {
            "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            "table_idx": 0,
            "col": "Symbol",
            "header": "# S&P 500 — constituents",
        },
        "nasdaq100": {
            "url": "https://en.wikipedia.org/wiki/Nasdaq-100",
            "table_idx": 4,
            "col": "Ticker",
            "header": "# Nasdaq-100 — constituents",
        },
        "dow30": {
            "url": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
            "table_idx": 2,
            "col": "Symbol",
            "header": "# Dow Jones Industrial Average — 30 components",
        },
    }

    import urllib.request

    _HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

    for name, src in sources.items():
        print(f"  Refreshing {name}... ", end="", flush=True)
        try:
            req = urllib.request.Request(src["url"], headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8")
            tables = pd.read_html(StringIO(html))
            df = tables[src["table_idx"]]
            col = src["col"]
            if col not in df.columns:
                # Try to find a column containing "Symbol" or "Ticker"
                for c in df.columns:
                    if "symbol" in str(c).lower() or "ticker" in str(c).lower():
                        col = c
                        break
            tickers = sorted(
                df[col]
                .dropna()
                .astype(str)
                .str.strip()
                .str.upper()
                .unique()
                .tolist()
            )
            path = universe_path(name)
            import datetime

            date_str = datetime.date.today().isoformat()
            lines = [f"{src['header']}", f"# Last updated: {date_str}", f"# Source: {src['url']}"]
            lines.extend(tickers)
            path.write_text("\n".join(lines) + "\n")
            print(f"{GREEN}{len(tickers)} tickers written to {path.name}{RESET}")
        except Exception as exc:
            print(f"{RED}FAILED: {exc}{RESET}")

    print(f"\n  {BOLD}Done.{RESET} Universe files updated.")


# ─────────────────────────────────────────────────────────────────────────────
# Progress callback for CLI
# ─────────────────────────────────────────────────────────────────────────────


def _cli_progress(completed: int, total: int, ticker: str, result: ScanResult | None) -> None:
    """Print a simple progress line, overwriting the previous one."""
    pct = completed / total * 100
    status = ""
    if result and not result.error:
        sig_color = SIGNAL_COLORS.get(result.signal, WHITE)
        status = f" \u2192 {sig_color}{result.signal}{RESET} ({result.effective_score:.2f})"
    elif result and result.error:
        status = f" \u2192 {RED}ERROR{RESET}"

    print(f"\r  [{completed:>{len(str(total))}}/{total}] {pct:5.1f}%  {ticker:<7}{status}    ", end="", flush=True)


def _dca_cli_progress(completed: int, total: int, ticker: str, result: DCAScanResult | None) -> None:
    """Print a simple progress line for DCA scan, overwriting the previous one."""
    pct = completed / total * 100
    status = ""
    if result and not result.error:
        if result.dca_score >= 60:
            sc = GREEN
        elif result.dca_score >= 35:
            sc = YELLOW
        else:
            sc = DIM
        status = f" \u2192 {sc}DCA {result.dca_score:.0f}{RESET}"
    elif result and result.error:
        status = f" \u2192 {RED}ERROR{RESET}"

    print(f"\r  [{completed:>{len(str(total))}}/{total}] {pct:5.1f}%  {ticker:<7}{status}    ", end="", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan a stock universe for top BUY/SELL signals or DCA opportunities."
    )
    parser.add_argument(
        "--universe", "-u",
        default="dow30",
        help="Universe to scan: dow30, nasdaq100, sp500, or path to .txt file (default: dow30)",
    )
    parser.add_argument(
        "--period", "-p",
        default="2y",
        help="Analysis period (default: 2y)",
    )
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=10,
        help="Number of top results to show (default: 10)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available universes and exit",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh universe files from Wikipedia and exit",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all results, not just top N",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["technical", "dca"],
        default="technical",
        help="Scan mode: 'technical' for BUY/SELL signals (default), 'dca' for DCA attractiveness ranking",
    )

    args = parser.parse_args()

    # ── List mode ────────────────────────────────────────────────────────
    if args.list:
        print(f"\n{BOLD}  Available universes:{RESET}")
        for name in available_universes():
            tickers = load_universe(name)
            print(f"    {name:<15} {len(tickers):>4} tickers")
        print()
        return

    # ── Refresh mode ─────────────────────────────────────────────────────
    if args.refresh:
        print(f"\n{BOLD}  Refreshing universe files...{RESET}\n")
        _refresh_universes()
        return

    # ── DCA scan mode ────────────────────────────────────────────────────
    if args.mode == "dca":
        tickers = load_universe(args.universe)

        print(f"\n{BOLD}  DCA Scanner{RESET}")
        print(f"  {'─' * 50}")
        print(f"  Universe:  {args.universe} ({len(tickers)} tickers)")
        print(f"  Period:    {args.period}")
        print(f"  Workers:   {args.workers}")
        print(f"  Showing:   top {args.top} DCA opportunities")
        print()

        scanner = DCAScanner(
            universe=args.universe,
            period=args.period,
            max_workers=args.workers,
            on_progress=_dca_cli_progress,
        )

        t0 = time.time()
        scanner.run()
        elapsed = time.time() - t0

        # Clear the progress line
        print("\r" + " " * 80 + "\r", end="")

        # Display results
        top_n = args.top
        if args.all:
            top_n = len(tickers)

        dca_buys = scanner.top_dca(top_n)
        _print_dca_table(f"Top {len(dca_buys)} DCA Opportunities", dca_buys, GREEN)

        _print_dca_summary(scanner, elapsed)
        print()
        return

    # ── Technical scan mode (default) ────────────────────────────────────
    tickers = load_universe(args.universe)

    print(f"\n{BOLD}  Stock Scanner{RESET}")
    print(f"  {'─' * 50}")
    print(f"  Universe:  {args.universe} ({len(tickers)} tickers)")
    print(f"  Period:    {args.period}")
    print(f"  Workers:   {args.workers}")
    print(f"  Showing:   top {args.top} BUY + top {args.top} SELL")
    print()

    scanner = Scanner(
        universe=args.universe,
        period=args.period,
        max_workers=args.workers,
        on_progress=_cli_progress,
    )

    t0 = time.time()
    scanner.run()
    elapsed = time.time() - t0

    # Clear the progress line
    print("\r" + " " * 80 + "\r", end="")

    # Display results
    top_n = args.top
    if args.all:
        top_n = len(tickers)

    buys = scanner.top_buys(top_n)
    sells = scanner.top_sells(top_n)

    _print_table(f"Top {len(buys)} BUY Signals", buys, GREEN)
    _print_table(f"Top {len(sells)} SELL Signals", sells, RED)

    _print_summary(scanner, elapsed)
    print()


if __name__ == "__main__":
    main()
