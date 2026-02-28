"""data/universes/__init__.py — Ticker-universe loader.

Provides helpers to list available universes and load ticker symbols from
the static ``.txt`` files stored alongside this module.

Each file has one ticker per line.  Lines starting with ``#`` are comments.
Blank lines are ignored.
"""

from __future__ import annotations

from pathlib import Path

UNIVERSES_DIR = Path(__file__).parent

# Canonical names → filenames (without .txt extension)
BUILTIN_UNIVERSES: dict[str, str] = {
    "dow30": "dow30",
    "nasdaq100": "nasdaq100",
    "sp500": "sp500",
    "tsx60": "tsx60",
}


def available() -> list[str]:
    """Return sorted list of available universe names.

    Includes both built-in files and any user-added ``.txt`` files.
    """
    names: set[str] = set(BUILTIN_UNIVERSES.keys())
    for p in UNIVERSES_DIR.glob("*.txt"):
        names.add(p.stem)
    return sorted(names)


def load(name: str) -> list[str]:
    """Load ticker symbols from a universe file.

    *name* can be a built-in key (``dow30``, ``nasdaq100``, ``sp500``) or
    any ``.txt`` file stem inside the universes directory.  You can also
    pass an absolute path to a custom file.

    Returns a de-duplicated list of uppercase ticker symbols in file order.
    """
    # Resolve to a Path
    path = Path(name)
    if not path.is_absolute():
        # Try built-in alias first, then raw stem
        stem = BUILTIN_UNIVERSES.get(name.lower(), name)
        path = UNIVERSES_DIR / f"{stem}.txt"

    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")

    seen: set[str] = set()
    tickers: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        ticker = line.upper()
        if ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)
    return tickers


def universe_path(name: str) -> Path:
    """Return the Path to a universe file (for refresh/update scripts)."""
    stem = BUILTIN_UNIVERSES.get(name.lower(), name)
    return UNIVERSES_DIR / f"{stem}.txt"
