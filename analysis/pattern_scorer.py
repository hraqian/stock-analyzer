"""
analysis/pattern_scorer.py — Weighted composite scoring for pattern signals.

Mirrors analysis/scorer.py but operates on PatternResult objects.
Uses weights from config.yaml → overall_patterns.weights.
Supports post-composite score spreading to combat clustering at 5.0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config
    from patterns.base import PatternResult


def _spread(score: float, factor: float) -> float:
    """Rescale *score* around midpoint (5.0) by *factor*, clamped to [0, 10]."""
    return max(0.0, min(10.0, 5.0 + (score - 5.0) * factor))


class PatternCompositeScorer:
    """Computes a weighted average score across all pattern results."""

    def __init__(self, cfg: "Config") -> None:
        self._weights = cfg.normalized_pattern_weights()

        # Score spreading config from overall_patterns section
        pat_cfg = cfg.section("overall_patterns")
        spread_cfg = pat_cfg.get("score_spreading", {})
        self._spread_enabled: bool = bool(spread_cfg.get("enabled", True))
        self._spread_factor: float = float(spread_cfg.get("factor", 2.0))

    def score(self, results: list["PatternResult"]) -> dict:
        """Return composite pattern score details.

        Returns:
            dict with keys:
                ``overall``     — final composite score (float), after spreading
                ``overall_raw`` — pre-spread composite score (float)
                ``breakdown``   — dict of config_key → individual score
                ``n_scored``    — number of patterns that contributed
                ``weights_used``— dict of config_key → weight
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
            overall_raw = 5.0
        else:
            overall_raw = weighted_sum / total_weight

        # Apply score spreading
        if self._spread_enabled and self._spread_factor != 1.0:
            overall = _spread(overall_raw, self._spread_factor)
        else:
            overall = overall_raw

        return {
            "overall": round(overall, 4),
            "overall_raw": round(overall_raw, 4),
            "breakdown": breakdown,
            "n_scored": len(breakdown),
            "weights_used": {k: self._weights.get(k, 0.0) for k in breakdown},
        }
