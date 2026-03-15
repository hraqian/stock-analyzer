"""Sector holdings configuration — persistent JSON file store.

Holdings have three layers (highest priority wins):
  1. Manual edits by power users (saved to the JSON config file)
  2. Dynamic refresh from yfinance (also saved to the JSON config file)
  3. Built-in defaults (hardcoded in sectors.py, used as fallback)

The JSON file stores per-sector overrides. When a sector is not present
in the file, the built-in default list is used.

File location: backend/app/data/sector_holdings.json
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Location of the config file (next to the app package)
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_HOLDINGS_FILE = _DATA_DIR / "sector_holdings.json"

# Thread lock for safe concurrent writes
_lock = threading.Lock()


def _ensure_data_dir() -> None:
    """Create the data directory if it doesn't exist."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_holdings() -> dict[str, list[list[str]]]:
    """Load sector holdings overrides from the JSON config file.

    Returns:
        Dict mapping sector name -> list of [ticker, company_name] pairs.
        Returns empty dict if the file doesn't exist yet.
    """
    if not _HOLDINGS_FILE.exists():
        return {}
    try:
        with open(_HOLDINGS_FILE, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning("Holdings file has invalid format, ignoring")
            return {}
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read holdings config: %s", exc)
        return {}


def save_holdings(holdings: dict[str, list[list[str]]]) -> None:
    """Save sector holdings overrides to the JSON config file.

    Args:
        holdings: Dict mapping sector name -> list of [ticker, company_name].
    """
    _ensure_data_dir()
    with _lock:
        try:
            with open(_HOLDINGS_FILE, "w") as f:
                json.dump(holdings, f, indent=2)
            logger.info("Saved holdings config for %d sectors", len(holdings))
        except OSError as exc:
            logger.error("Failed to write holdings config: %s", exc)
            raise


def get_sector_holdings(sector_name: str) -> list[list[str]] | None:
    """Get holdings override for a single sector.

    Returns:
        List of [ticker, name] pairs, or None if no override exists.
    """
    data = load_holdings()
    return data.get(sector_name)


def set_sector_holdings(sector_name: str, holdings: list[list[str]]) -> None:
    """Set holdings for a single sector (merges with existing config).

    Args:
        sector_name: GICS sector name.
        holdings: List of [ticker, company_name] pairs.
    """
    data = load_holdings()
    data[sector_name] = holdings
    save_holdings(data)


def reset_sector_holdings(sector_name: str) -> None:
    """Remove the override for a sector (reverts to built-in defaults).

    Args:
        sector_name: GICS sector name.
    """
    data = load_holdings()
    if sector_name in data:
        del data[sector_name]
        save_holdings(data)
