"""
config.py — Load, validate, and manage the YAML configuration file.

Provides a single Config object used throughout the application.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Default configuration (mirrors config.yaml — used when file is absent or
# a key is missing so we always have a fully-populated config dict).
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict[str, Any] = {
    "rsi": {
        "period": 14,
        "thresholds": {"oversold": 30, "overbought": 70},
        "scores": {"oversold_score": 9.0, "overbought_score": 1.0, "neutral_score": 5.0},
    },
    "macd": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
        "scoring": {
            "strong_bullish_pct": 0.005,
            "moderate_bullish_pct": 0.001,
            "strong_bearish_pct": -0.005,
            "moderate_bearish_pct": -0.001,
            "crossover_lookback": 5,
            "bullish_cross_bonus": 1.5,
            "bearish_cross_penalty": 1.5,
        },
    },
    "bollinger_bands": {
        "period": 20,
        "std_dev": 2.0,
        "scoring": {
            "lower_zone": 0.20,
            "upper_zone": 0.80,
            "squeeze_threshold": 0.02,
        },
    },
    "moving_averages": {
        "periods": [20, 50, 200],
        "type": "sma",
        "scoring": {
            "price_above_ma_points": 1.5,
            "ma_aligned_bullish_points": 1.0,
            "golden_cross_bonus": 2.0,
            "death_cross_penalty": 2.0,
            "cross_lookback": 10,
            "max_raw_score": 9.5,
        },
    },
    "stochastic": {
        "k_period": 14,
        "d_period": 3,
        "smooth_k": 3,
        "thresholds": {"oversold": 20, "overbought": 80},
        "scores": {
            "oversold_score": 9.0,
            "overbought_score": 1.0,
            "neutral_score": 5.0,
            "bullish_cross_bonus": 1.0,
            "bearish_cross_penalty": 1.0,
        },
    },
    "adx": {
        "period": 14,
        "thresholds": {"weak": 30, "moderate": 50},
        "scoring": {
            "weak_multiplier": 0.4,
            "moderate_multiplier": 0.75,
            "strong_multiplier": 1.0,
            "max_directional_spread": 40,
        },
    },
    "volume": {
        "obv_trend_period": 20,
        "price_trend_period": 20,
        "scoring": {
            "confirmation_score": 8.0,
            "neutral_score": 5.0,
            "divergence_score": 2.0,
        },
    },
    "fibonacci": {
        "swing_lookback": 60,
        "levels": [0.236, 0.382, 0.5, 0.618, 0.786],
        "scoring": {
            "proximity_pct": 0.015,
            "level_scores": {0.236: 7.0, 0.382: 6.5, 0.5: 5.5, 0.618: 4.5, 0.786: 3.0},
            "no_level_score": 5.0,
        },
    },
    "support_resistance": {
        "method": "both",
        "pivot_levels": ["S1", "S2", "S3", "P", "R1", "R2", "R3"],
        "fractal_lookback": 60,
        "fractal_order": 5,
        "num_levels": 4,
        "cluster_pct": 0.015,
        "min_touches": 1,
    },
    "overall": {
        "weights": {
            "rsi": 0.15,
            "macd": 0.15,
            "bollinger_bands": 0.10,
            "moving_averages": 0.20,
            "stochastic": 0.10,
            "adx": 0.10,
            "volume": 0.10,
            "fibonacci": 0.10,
        }
    },
    "display": {
        "show_disclaimer": True,
        "score_decimal_places": 1,
        "price_decimal_places": 2,
        "color_thresholds": {"bearish_max": 3.5, "neutral_max": 6.5},
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class Config:
    """Loads and validates the YAML config, falling back to defaults for
    any missing keys.

    Usage::

        cfg = Config.load("config.yaml")
        rsi_period = cfg["rsi"]["period"]
    """

    def __init__(self, data: dict[str, Any], path: str | None = None) -> None:
        self._data = data
        self.path = path

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        """Load config from *path*.  Falls back to defaults for missing keys.

        If *path* is None, looks for ``config.yaml`` next to this file, then
        in the current working directory.
        """
        search_paths = []
        if path:
            search_paths.append(Path(path))
        else:
            search_paths += [
                Path(__file__).parent / "config.yaml",
                Path(os.getcwd()) / "config.yaml",
            ]

        loaded: dict = {}
        resolved_path: str | None = None
        for p in search_paths:
            if p.exists():
                with open(p, "r") as fh:
                    loaded = yaml.safe_load(fh) or {}
                resolved_path = str(p)
                break

        merged = _deep_merge(DEFAULT_CONFIG, loaded)
        cfg = cls(merged, resolved_path)
        errors = cfg.validate()
        if errors:
            print("[config] Validation warnings:")
            for e in errors:
                print(f"  • {e}")
        return cfg

    @classmethod
    def defaults(cls) -> "Config":
        """Return a Config populated entirely with defaults."""
        return cls(copy.deepcopy(DEFAULT_CONFIG), None)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of validation error/warning strings (empty = OK)."""
        errors: list[str] = []

        # Weights must be positive
        weights: dict = self._data.get("overall", {}).get("weights", {})
        for k, v in weights.items():
            if not isinstance(v, (int, float)) or v < 0:
                errors.append(f"overall.weights.{k} must be a non-negative number, got {v!r}")

        # RSI thresholds
        rsi = self._data.get("rsi", {}).get("thresholds", {})
        if rsi.get("oversold", 30) >= rsi.get("overbought", 70):
            errors.append("rsi.thresholds.oversold must be less than overbought")

        # Stochastic thresholds
        stoch = self._data.get("stochastic", {}).get("thresholds", {})
        if stoch.get("oversold", 20) >= stoch.get("overbought", 80):
            errors.append("stochastic.thresholds.oversold must be less than overbought")

        # ADX thresholds
        adx = self._data.get("adx", {}).get("thresholds", {})
        if adx.get("weak", 30) >= adx.get("moderate", 50):
            errors.append("adx.thresholds.weak must be less than moderate")

        # MA periods must be sorted ascending
        ma_periods = self._data.get("moving_averages", {}).get("periods", [20, 50, 200])
        if ma_periods != sorted(ma_periods):
            errors.append("moving_averages.periods should be in ascending order")

        return errors

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def section(self, key: str) -> dict:
        """Return a config section, guaranteed to be a dict."""
        return self._data.get(key) or {}

    def normalized_weights(self) -> dict[str, float]:
        """Return overall weights normalized so they sum to 1.0."""
        weights = self._data["overall"]["weights"]
        total = sum(weights.values())
        if total == 0:
            equal = 1.0 / len(weights)
            return {k: equal for k in weights}
        return {k: v / total for k, v in weights.items()}

    # ------------------------------------------------------------------
    # Config generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_default(output_path: str = "config.yaml") -> None:
        """Write a fresh default config.yaml to *output_path*."""
        template_path = Path(__file__).parent / "config.yaml"
        if template_path.exists():
            import shutil
            shutil.copy(template_path, output_path)
            print(f"Default config written to: {output_path}")
        else:
            # Fallback: dump defaults as YAML
            with open(output_path, "w") as fh:
                yaml.dump(DEFAULT_CONFIG, fh, default_flow_style=False, sort_keys=False)
            print(f"Default config written to: {output_path}")
