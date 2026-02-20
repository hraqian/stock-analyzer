"""
indicators/registry.py — Auto-discovers and manages all BaseIndicator subclasses.

Any module placed in the indicators/ package that defines a BaseIndicator
subclass with a non-empty config_key will be automatically picked up.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .base import BaseIndicator, IndicatorResult

if TYPE_CHECKING:
    from config import Config

# Modules to skip during auto-discovery
_SKIP_MODULES = {"base", "registry", "__init__"}


class IndicatorRegistry:
    """Discovers all BaseIndicator subclasses in the indicators package,
    instantiates them with their config section, and runs them.

    Usage::

        registry = IndicatorRegistry(cfg, only=["rsi", "macd"])
        results = registry.run_all(df)
    """

    def __init__(
        self,
        cfg: "Config",
        only: list[str] | None = None,
    ) -> None:
        """
        Args:
            cfg:  The application Config object.
            only: Optional allowlist of config_key strings.  If provided,
                  only those indicators are loaded.
        """
        self._cfg = cfg
        self._only = {k.lower() for k in only} if only else None
        self._indicators: list[BaseIndicator] = []
        self._discover()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover(self) -> None:
        """Import every module in indicators/ and collect BaseIndicator subclasses."""
        pkg_path = Path(__file__).parent
        pkg_name = __name__.rsplit(".", 1)[0]  # "indicators"

        for finder, module_name, _ in pkgutil.iter_modules([str(pkg_path)]):
            if module_name in _SKIP_MODULES:
                continue
            full_name = f"{pkg_name}.{module_name}"
            try:
                mod = importlib.import_module(full_name)
            except Exception as exc:  # noqa: BLE001
                print(f"[registry] Could not import {full_name}: {exc}")
                continue

            for attr_name in dir(mod):
                obj = getattr(mod, attr_name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, BaseIndicator)
                    and obj is not BaseIndicator
                    and obj.config_key  # skip abstract stubs without a key
                ):
                    if self._only is None or obj.config_key in self._only:
                        instance = obj(self._cfg.section(obj.config_key))
                        self._indicators.append(instance)

        # Stable ordering: alphabetical by config_key
        self._indicators.sort(key=lambda i: i.config_key)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_all(self, df: pd.DataFrame) -> list[IndicatorResult]:
        """Run every registered indicator against *df* and return results."""
        return [ind.run(df) for ind in self._indicators]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def indicator_names(self) -> list[str]:
        return [i.config_key for i in self._indicators]

    def __len__(self) -> int:
        return len(self._indicators)
