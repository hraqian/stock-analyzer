"""
analysis/scorer.py — Weighted composite scoring engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config
    from indicators.base import IndicatorResult


class CompositeScorer:
    """Computes a weighted average score across all indicator results."""

    def __init__(self, cfg: "Config") -> None:
        self._weights = cfg.normalized_weights()

    def score(self, results: list["IndicatorResult"]) -> dict:
        """Return composite score details.

        Returns:
            dict with keys:
                ``overall``     — weighted average score (float)
                ``breakdown``   — dict of config_key → weighted contribution
                ``n_scored``    — number of indicators that contributed
        """
        total_weight = 0.0
        weighted_sum = 0.0
        breakdown: dict[str, float] = {}

        for result in results:
            if result.error:
                continue
            weight = self._weights.get(result.config_key, 0.0)
            if weight <= 0:
                continue
            weighted_sum += result.score * weight
            total_weight += weight
            breakdown[result.config_key] = result.score

        if total_weight == 0:
            overall = 5.0
        else:
            overall = weighted_sum / total_weight

        return {
            "overall": round(overall, 4),
            "breakdown": breakdown,
            "n_scored": len(breakdown),
            "weights_used": {k: self._weights.get(k, 0.0) for k in breakdown},
        }
