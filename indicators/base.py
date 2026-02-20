"""
indicators/base.py — Abstract base class for all technical indicators.

To add a new indicator:
  1. Create indicators/my_indicator.py
  2. Subclass BaseIndicator and implement compute(), score(), summary()
  3. Set class attributes: name, config_key
  4. The IndicatorRegistry will auto-discover it — no other changes needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class IndicatorResult:
    """Holds the output of one indicator's compute + score cycle."""

    name: str                        # Human-readable name, e.g. "RSI (14)"
    config_key: str                  # Key in config.yaml, e.g. "rsi"
    score: float                     # 0.0 – 10.0
    values: dict[str, Any] = field(default_factory=dict)   # raw computed values
    display: dict[str, Any] = field(default_factory=dict)  # formatted for display
    error: str | None = None         # set if the indicator failed to compute


class BaseIndicator(ABC):
    """All technical indicators implement this interface.

    Subclasses must define class-level attributes:
        name       (str) — display name, e.g. ``"RSI"``
        config_key (str) — key in config.yaml, e.g. ``"rsi"``
    """

    name: str = ""
    config_key: str = ""

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: The indicator's section from config.yaml, e.g.
                    ``cfg.section("rsi")``.
        """
        self.config = config

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute raw indicator values from OHLCV DataFrame.

        Args:
            df: DataFrame with columns [open, high, low, close, volume].

        Returns:
            Dict of computed values (indicator-specific keys).
        """
        ...

    @abstractmethod
    def score(self, values: dict[str, Any]) -> float:
        """Translate computed values into a score from 0 to 10.

        Args:
            values: The dict returned by :meth:`compute`.

        Returns:
            Float in [0.0, 10.0].
        """
        ...

    @abstractmethod
    def summary(self, values: dict[str, Any], score: float) -> dict[str, Any]:
        """Build a display-friendly summary dict for this indicator.

        The returned dict should include at minimum:
            ``value_str``   — the main value string shown in the table
            ``detail_str``  — secondary detail / context string

        Args:
            values: The dict returned by :meth:`compute`.
            score:  The float returned by :meth:`score`.

        Returns:
            Dict with display fields.
        """
        ...

    # ------------------------------------------------------------------
    # Convenience — run the full pipeline and return IndicatorResult
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> IndicatorResult:
        """Compute → score → summarise.  Catches exceptions gracefully."""
        try:
            values = self.compute(df)
            s = float(max(0.0, min(10.0, self.score(values))))
            display = self.summary(values, s)
            return IndicatorResult(
                name=self.name,
                config_key=self.config_key,
                score=s,
                values=values,
                display=display,
            )
        except Exception as exc:  # noqa: BLE001
            return IndicatorResult(
                name=self.name,
                config_key=self.config_key,
                score=5.0,
                error=str(exc),
                display={"value_str": "N/A", "detail_str": f"Error: {exc}"},
            )

    # ------------------------------------------------------------------
    # Shared scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp(value: float, lo: float = 0.0, hi: float = 10.0) -> float:
        return max(lo, min(hi, value))

    @staticmethod
    def _linear_score(
        value: float,
        low_val: float,
        high_val: float,
        low_score: float,
        high_score: float,
    ) -> float:
        """Linearly interpolate a score between two value endpoints."""
        if high_val == low_val:
            return (low_score + high_score) / 2
        t = (value - low_val) / (high_val - low_val)
        t = max(0.0, min(1.0, t))
        return low_score + t * (high_score - low_score)
