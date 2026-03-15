"""Shared constants and utilities for backend services.

Centralises values that were previously duplicated across multiple modules.
"""

from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# Trade mode → engine objective mapping
# ---------------------------------------------------------------------------

TRADE_MODE_OBJECTIVES: dict[str, str] = {
    "swing": "swing_trade",
    "long_term": "long_term",
}


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------

def safe(v: Any) -> Any:
    """Recursively sanitise a value for JSON (NaN/Inf → None, numpy/pandas → native).

    Handles numpy scalars/arrays, pandas Series/Index/Timestamp, and plain
    Python floats with NaN/Inf.  Safe to call even if numpy/pandas are not
    installed (falls back to plain-float handling only).
    """
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]

    try:
        import pandas as pd
    except ImportError:
        pd = None  # type: ignore[assignment]

    # --- pandas types ---
    if pd is not None:
        if isinstance(v, (pd.Series, pd.Index)):
            return [safe(x) for x in v.tolist()]
        if isinstance(v, pd.Timestamp):
            return v.isoformat()

    # --- numpy types ---
    if np is not None:
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            val = float(v)
            return None if (math.isnan(val) or math.isinf(val)) else val
        if isinstance(v, np.ndarray):
            return [safe(x) for x in v.tolist()]
        if isinstance(v, np.bool_):
            return bool(v)

    # --- plain Python ---
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(v, dict):
        return {str(k): safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [safe(item) for item in v]
    return v
