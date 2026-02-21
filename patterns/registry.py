"""
patterns/registry.py — Auto-discovers and manages all BasePattern subclasses.

Any module placed in the patterns/ package that defines a BasePattern
subclass with a non-empty config_key will be automatically picked up.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .base import BasePattern, PatternResult

if TYPE_CHECKING:
    from config import Config

# Modules to skip during auto-discovery
_SKIP_MODULES = {"base", "registry", "__init__"}


class PatternRegistry:
    """Discovers all BasePattern subclasses in the patterns package,
    instantiates them with their config section, and runs them.

    Usage::

        registry = PatternRegistry(cfg)
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
                  only those patterns are loaded.
        """
        self._cfg = cfg
        self._only = {k.lower() for k in only} if only else None
        self._patterns: list[BasePattern] = []
        self._discover()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover(self) -> None:
        """Import every module in patterns/ and collect BasePattern subclasses."""
        pkg_path = Path(__file__).parent
        pkg_name = __name__.rsplit(".", 1)[0]  # "patterns"

        for finder, module_name, _ in pkgutil.iter_modules([str(pkg_path)]):
            if module_name in _SKIP_MODULES:
                continue
            full_name = f"{pkg_name}.{module_name}"
            try:
                mod = importlib.import_module(full_name)
            except Exception as exc:  # noqa: BLE001
                print(f"[pattern_registry] Could not import {full_name}: {exc}")
                continue

            for attr_name in dir(mod):
                obj = getattr(mod, attr_name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, BasePattern)
                    and obj is not BasePattern
                    and obj.config_key  # skip abstract stubs without a key
                ):
                    if self._only is None or obj.config_key in self._only:
                        instance = obj(self._cfg.section(obj.config_key))
                        self._patterns.append(instance)

        # Stable ordering: alphabetical by config_key
        self._patterns.sort(key=lambda p: p.config_key)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_all(self, df: pd.DataFrame) -> list[PatternResult]:
        """Run every registered pattern against *df* and return results."""
        return [pat.run(df) for pat in self._patterns]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def pattern_names(self) -> list[str]:
        return [p.config_key for p in self._patterns]

    def __len__(self) -> int:
        return len(self._patterns)
