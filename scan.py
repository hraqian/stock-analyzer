#!/usr/bin/env python3
"""
scan.py — CLI stock-universe scanner.

Scans a stock universe through the full analysis + strategy pipeline and
ranks the top BUY and SELL candidates.

Usage:
    python scan.py                          # scan Dow 30 (fast default)
    python scan.py --universe nasdaq100     # scan NASDAQ 100
    python scan.py --universe sp500         # scan S&P 500 (~10-15 min)
    python scan.py --universe dow30 --top 5 # show top 5 instead of 10
    python scan.py --period 1y              # use 1-year analysis period
    python scan.py --workers 12             # increase parallelism
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
from scanner import Scanner, ScanResult


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
        status = f" → {sig_color}{result.signal}{RESET} ({result.effective_score:.2f})"
    elif result and result.error:
        status = f" → {RED}ERROR{RESET}"

    print(f"\r  [{completed:>{len(str(total))}}/{total}] {pct:5.1f}%  {ticker:<7}{status}    ", end="", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan a stock universe for top BUY/SELL signals."
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
        help="Number of top BUY/SELL results to show (default: 10)",
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

    # ── Scan mode ────────────────────────────────────────────────────────
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
