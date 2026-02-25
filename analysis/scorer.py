"""
analysis/scorer.py — Weighted composite scoring engine with subgroup
awareness and score spreading to combat averaging-induced compression.

The problem: averaging 8 indicators (some trend-following, some contrarian)
crushes variance — composites cluster around 5.0 regardless of market
conditions.  Two mechanisms combat this:

1. **Subgroup scoring** — Indicators are classified as *trend*, *contrarian*,
   or *neutral*.  Each subgroup is scored independently, then the dominant
   signal (the subgroup deviating most from 5.0) is amplified.

2. **Score spreading** — A post-composite rescaling step that multiplies
   the distance from the midpoint (5.0) by a configurable factor, expanding
   the effective range from ~4.0-6.0 back toward 0-10.

Both mechanisms are fully configurable and can be disabled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config
    from indicators.base import IndicatorResult


# Default subgroup assignments
DEFAULT_INDICATOR_GROUPS: dict[str, list[str]] = {
    "trend": ["moving_averages", "macd", "adx", "volume"],
    "contrarian": ["rsi", "stochastic", "bollinger_bands"],
    "neutral": ["fibonacci"],
}


def _weighted_avg(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted average of scores for a subset of indicators."""
    total_w = 0.0
    total_s = 0.0
    for key, s in scores.items():
        w = weights.get(key, 0.0)
        if w > 0:
            total_s += s * w
            total_w += w
    return total_s / total_w if total_w > 0 else 5.0


def _spread(score: float, factor: float) -> float:
    """Rescale *score* around midpoint (5.0) by *factor*, clamped to [0, 10]."""
    return max(0.0, min(10.0, 5.0 + (score - 5.0) * factor))


class CompositeScorer:
    """Computes a composite score with optional subgroup scoring and spreading.

    Config keys (under ``overall``):

    - ``weights`` — per-indicator weights (existing)
    - ``subgroup_mode`` — ``"directional"`` (default) or ``"average"`` (legacy)
    - ``indicator_groups.trend`` — list of trend-following indicator keys
    - ``indicator_groups.contrarian`` — list of contrarian indicator keys
    - ``indicator_groups.neutral`` — list of neutral indicator keys
    - ``subgroup_blend.dominant_weight`` — weight for the dominant subgroup (0.6)
    - ``subgroup_blend.other_weight`` — weight for the other subgroup (0.25)
    - ``subgroup_blend.neutral_weight`` — weight for neutral subgroup (0.15)
    - ``score_spreading.enabled`` — enable post-composite rescaling (true)
    - ``score_spreading.factor`` — spread multiplier (2.0)
    """

    def __init__(self, cfg: "Config") -> None:
        self._weights = cfg.normalized_weights()
        overall = cfg.section("overall")

        # Subgroup configuration
        self._subgroup_mode: str = overall.get("subgroup_mode", "directional")
        groups = overall.get("indicator_groups", {})
        self._trend_keys: list[str] = groups.get(
            "trend", DEFAULT_INDICATOR_GROUPS["trend"]
        )
        self._contrarian_keys: list[str] = groups.get(
            "contrarian", DEFAULT_INDICATOR_GROUPS["contrarian"]
        )
        self._neutral_keys: list[str] = groups.get(
            "neutral", DEFAULT_INDICATOR_GROUPS["neutral"]
        )

        blend = overall.get("subgroup_blend", {})
        self._dominant_weight: float = float(blend.get("dominant_weight", 0.6))
        self._other_weight: float = float(blend.get("other_weight", 0.25))
        self._neutral_weight: float = float(blend.get("neutral_weight", 0.15))

        # Score spreading
        spread_cfg = overall.get("score_spreading", {})
        self._spread_enabled: bool = bool(spread_cfg.get("enabled", True))
        self._spread_factor: float = float(spread_cfg.get("factor", 2.0))

    def score(self, results: list["IndicatorResult"]) -> dict:
        """Return composite score details.

        Returns:
            dict with keys:
                ``overall``         — final composite score (float)
                ``overall_raw``     — pre-spread composite (float)
                ``breakdown``       — dict of config_key → individual score
                ``n_scored``        — number of indicators that contributed
                ``weights_used``    — dict of config_key → weight
                ``trend_score``     — subgroup score for trend indicators
                ``contrarian_score``— subgroup score for contrarian indicators
                ``neutral_score``   — subgroup score for neutral indicators
                ``dominant_group``  — which subgroup dominated ("trend" / "contrarian")
        """
        # Collect valid scored indicators
        breakdown: dict[str, float] = {}
        for result in results:
            if result.error:
                continue
            weight = self._weights.get(result.config_key, 0.0)
            if weight <= 0:
                continue
            breakdown[result.config_key] = result.score

        if not breakdown:
            return {
                "overall": 5.0,
                "overall_raw": 5.0,
                "breakdown": breakdown,
                "n_scored": 0,
                "weights_used": {},
                "trend_score": 5.0,
                "contrarian_score": 5.0,
                "neutral_score": 5.0,
                "dominant_group": "none",
            }

        # ── Compute composite ─────────────────────────────────────────────
        if self._subgroup_mode == "directional":
            overall_raw = self._directional_composite(breakdown)
        else:
            # Legacy flat weighted average
            overall_raw = _weighted_avg(breakdown, self._weights)

        # ── Subgroup scores (always computed for display) ──────────────────
        trend_scores = {k: v for k, v in breakdown.items() if k in self._trend_keys}
        contrarian_scores = {k: v for k, v in breakdown.items() if k in self._contrarian_keys}
        neutral_scores = {k: v for k, v in breakdown.items() if k in self._neutral_keys}

        trend_avg = _weighted_avg(trend_scores, self._weights) if trend_scores else 5.0
        contrarian_avg = _weighted_avg(contrarian_scores, self._weights) if contrarian_scores else 5.0
        neutral_avg = _weighted_avg(neutral_scores, self._weights) if neutral_scores else 5.0

        trend_dev = abs(trend_avg - 5.0)
        contrarian_dev = abs(contrarian_avg - 5.0)
        dominant = "trend" if trend_dev >= contrarian_dev else "contrarian"

        # ── Score spreading ───────────────────────────────────────────────
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
            "trend_score": round(trend_avg, 4),
            "contrarian_score": round(contrarian_avg, 4),
            "neutral_score": round(neutral_avg, 4),
            "dominant_group": dominant,
        }

    def _directional_composite(self, breakdown: dict[str, float]) -> float:
        """Compute composite using directional subgroup blending.

        1. Score each subgroup (trend, contrarian, neutral) independently.
        2. Determine which of trend/contrarian deviates more from 5.0.
        3. Weight the dominant subgroup higher.
        """
        trend_scores = {k: v for k, v in breakdown.items() if k in self._trend_keys}
        contrarian_scores = {k: v for k, v in breakdown.items() if k in self._contrarian_keys}
        neutral_scores = {k: v for k, v in breakdown.items() if k in self._neutral_keys}

        trend_avg = _weighted_avg(trend_scores, self._weights) if trend_scores else 5.0
        contrarian_avg = _weighted_avg(contrarian_scores, self._weights) if contrarian_scores else 5.0
        neutral_avg = _weighted_avg(neutral_scores, self._weights) if neutral_scores else 5.0

        # Which subgroup has a stronger opinion?
        trend_dev = abs(trend_avg - 5.0)
        contrarian_dev = abs(contrarian_avg - 5.0)

        if trend_dev >= contrarian_dev:
            dominant_avg = trend_avg
            other_avg = contrarian_avg
        else:
            dominant_avg = contrarian_avg
            other_avg = trend_avg

        # Blend with dominant subgroup weighted higher
        # Normalize blend weights to handle cases where a group is empty
        w_dom = self._dominant_weight
        w_oth = self._other_weight
        w_neu = self._neutral_weight if neutral_scores else 0.0
        w_total = w_dom + w_oth + w_neu
        if w_total > 0:
            composite = (
                w_dom * dominant_avg + w_oth * other_avg + w_neu * neutral_avg
            ) / w_total
        else:
            composite = 5.0

        return composite
