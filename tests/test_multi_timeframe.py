"""Tests for analysis.multi_timeframe — MultiTimeframeAnalyzer and helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analysis.multi_timeframe import (
    MultiTimeframeAnalyzer,
    MultiTimeframeResult,
    TimeframeResult,
    _compute_agreement,
    _compute_effective_score,
    _derive_signal,
    _derive_signal_with_mode,
)
from config import Config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> Config:
    return Config.defaults()


@pytest.fixture
def mock_provider():
    return MagicMock()


def _make_analysis_result(
    ticker: str = "AAPL",
    overall: float = 6.0,
    pattern_overall: float = 5.5,
    trend_score: float = 6.5,
    contrarian_score: float = 5.0,
    dominant_group: str = "trend",
    regime_label: str = "Strong Trend",
):
    """Build a mock AnalysisResult with the composite values we need."""
    ar = MagicMock()
    ar.ticker = ticker
    ar.composite = {
        "overall": overall,
        "trend_score": trend_score,
        "contrarian_score": contrarian_score,
        "dominant_group": dominant_group,
    }
    ar.pattern_composite = {"overall": pattern_overall}
    ar.regime = MagicMock()
    ar.regime.label = regime_label
    return ar


# ===========================================================================
# Tests: _derive_signal
# ===========================================================================


class TestDeriveSignal:
    def test_buy_signal(self):
        assert _derive_signal(7.0, short_below=3.5, hold_below=6.0) == "BUY"

    def test_sell_signal(self):
        assert _derive_signal(3.0, short_below=3.5, hold_below=6.0) == "SELL"

    def test_hold_signal_middle(self):
        assert _derive_signal(5.0, short_below=3.5, hold_below=6.0) == "HOLD"

    def test_exactly_at_short_threshold(self):
        assert _derive_signal(3.5, short_below=3.5, hold_below=6.0) == "SELL"

    def test_exactly_at_hold_threshold(self):
        # hold_below=6.0 means BUY when > 6.0, so exactly 6.0 is HOLD
        assert _derive_signal(6.0, short_below=3.5, hold_below=6.0) == "HOLD"

    def test_just_above_hold(self):
        assert _derive_signal(6.01, short_below=3.5, hold_below=6.0) == "BUY"

    def test_just_below_short(self):
        assert _derive_signal(3.49, short_below=3.5, hold_below=6.0) == "SELL"

    def test_extreme_buy(self):
        assert _derive_signal(10.0, short_below=3.5, hold_below=6.0) == "BUY"

    def test_extreme_sell(self):
        assert _derive_signal(0.0, short_below=3.5, hold_below=6.0) == "SELL"


# ==========================================================================
# Tests: _derive_signal_with_mode
# ==========================================================================


class TestDeriveSignalWithMode:
    def test_gate_mode_ignores_thresholds(self):
        # Gate allows long even if effective score is below hold_below
        sig = _derive_signal_with_mode(
            effective_score=5.0,
            ind_score=6.0,
            pat_score=6.0,
            combination_mode="gate",
            short_below=3.5,
            hold_below=6.0,
            threshold_mode="fixed",
            score_window=[],
            short_pct=25,
            long_pct=75,
            lookback_bars=60,
            min_fill_ratio=0.8,
            gate_indicator_min=5.5,
            gate_indicator_max=4.5,
            gate_pattern_min=5.5,
            gate_pattern_max=4.5,
        )
        assert sig == "BUY"

    def test_percentile_fallback_uses_fixed(self):
        # Not enough samples -> fall back to fixed thresholds
        sig = _derive_signal_with_mode(
            effective_score=3.0,
            ind_score=3.0,
            pat_score=3.0,
            combination_mode="weighted",
            short_below=3.5,
            hold_below=6.0,
            threshold_mode="percentile",
            score_window=[],
            short_pct=25,
            long_pct=75,
            lookback_bars=60,
            min_fill_ratio=0.8,
            gate_indicator_min=5.5,
            gate_indicator_max=4.5,
            gate_pattern_min=5.5,
            gate_pattern_max=4.5,
        )
        assert sig == "SELL"

    def test_percentile_rank_strict_less(self):
        # All scores equal -> strict-less rank = 0
        window = [2.0] * 60
        sig = _derive_signal_with_mode(
            effective_score=2.0,
            ind_score=2.0,
            pat_score=2.0,
            combination_mode="weighted",
            short_below=3.5,
            hold_below=6.0,
            threshold_mode="percentile",
            score_window=window,
            short_pct=10,
            long_pct=90,
            lookback_bars=60,
            min_fill_ratio=0.8,
            gate_indicator_min=5.5,
            gate_indicator_max=4.5,
            gate_pattern_min=5.5,
            gate_pattern_max=4.5,
        )
        assert sig == "SELL"


# ===========================================================================
# Tests: _compute_agreement
# ===========================================================================


class TestComputeAgreement:
    def test_all_buy(self):
        assert _compute_agreement(["BUY", "BUY", "BUY"]) == "aligned"

    def test_all_hold(self):
        assert _compute_agreement(["HOLD", "HOLD"]) == "aligned"

    def test_all_sell(self):
        assert _compute_agreement(["SELL"]) == "aligned"

    def test_buy_and_sell_is_conflicting(self):
        assert _compute_agreement(["BUY", "SELL", "HOLD"]) == "conflicting"

    def test_buy_and_sell_only(self):
        assert _compute_agreement(["BUY", "SELL"]) == "conflicting"

    def test_buy_and_hold_is_mixed(self):
        assert _compute_agreement(["BUY", "HOLD"]) == "mixed"

    def test_sell_and_hold_is_mixed(self):
        assert _compute_agreement(["SELL", "HOLD"]) == "mixed"

    def test_empty_list(self):
        assert _compute_agreement([]) == "aligned"


# ===========================================================================
# Tests: _compute_effective_score
# ===========================================================================


class TestComputeEffectiveScore:
    def test_weighted_mode(self):
        result = _compute_effective_score(
            ind_score=8.0, pat_score=6.0,
            combination_mode="weighted",
            ind_weight=0.7, pat_weight=0.3,
            boost_strength=0.5, boost_dead_zone=0.3,
        )
        expected = (0.7 * 8.0 + 0.3 * 6.0) / 1.0
        assert abs(result - expected) < 0.01

    def test_gate_mode_uses_indicator_only(self):
        result = _compute_effective_score(
            ind_score=3.0, pat_score=9.0,
            combination_mode="gate",
            ind_weight=0.7, pat_weight=0.3,
            boost_strength=0.5, boost_dead_zone=0.3,
        )
        assert result == 3.0

    def test_boost_mode_inside_dead_zone(self):
        result = _compute_effective_score(
            ind_score=6.0, pat_score=5.2,  # dev = 0.2, within dead zone 0.3
            combination_mode="boost",
            ind_weight=0.7, pat_weight=0.3,
            boost_strength=0.5, boost_dead_zone=0.3,
        )
        assert result == 6.0  # no boost

    def test_boost_mode_positive_deviation(self):
        result = _compute_effective_score(
            ind_score=6.0, pat_score=7.0,  # dev = 2.0, eff_dev = 1.7
            combination_mode="boost",
            ind_weight=0.7, pat_weight=0.3,
            boost_strength=0.5, boost_dead_zone=0.3,
        )
        expected = 6.0 + (2.0 - 0.3) * 0.5  # 6.85
        assert abs(result - expected) < 0.01

    def test_boost_mode_negative_deviation(self):
        result = _compute_effective_score(
            ind_score=6.0, pat_score=3.0,  # dev = -2.0, eff_dev = -1.7
            combination_mode="boost",
            ind_weight=0.7, pat_weight=0.3,
            boost_strength=0.5, boost_dead_zone=0.3,
        )
        expected = 6.0 + (-2.0 + 0.3) * 0.5  # 5.15
        assert abs(result - expected) < 0.01

    def test_boost_mode_clamps_to_10(self):
        result = _compute_effective_score(
            ind_score=9.0, pat_score=10.0,
            combination_mode="boost",
            ind_weight=0.7, pat_weight=0.3,
            boost_strength=1.0, boost_dead_zone=0.0,
        )
        assert result == 10.0

    def test_boost_mode_clamps_to_0(self):
        result = _compute_effective_score(
            ind_score=1.0, pat_score=0.0,
            combination_mode="boost",
            ind_weight=0.7, pat_weight=0.3,
            boost_strength=1.0, boost_dead_zone=0.0,
        )
        assert result == 0.0

    def test_weighted_zero_weights(self):
        result = _compute_effective_score(
            ind_score=7.0, pat_score=3.0,
            combination_mode="weighted",
            ind_weight=0.0, pat_weight=0.0,
            boost_strength=0.5, boost_dead_zone=0.3,
        )
        assert result == 7.0  # fallback to indicator


# ===========================================================================
# Tests: TimeframeResult dataclass
# ===========================================================================


class TestTimeframeResult:
    def test_create(self):
        tr = TimeframeResult(
            timeframe="1d", period="2y", weight=0.5,
            indicator_score=7.0, pattern_score=6.0,
            signal="BUY", regime_label="Strong Trend",
            trend_score=7.5, contrarian_score=5.0,
            dominant_group="trend",
        )
        assert tr.timeframe == "1d"
        assert tr.error is None

    def test_with_error(self):
        tr = TimeframeResult(
            timeframe="1mo", period="max", weight=0.2,
            indicator_score=5.0, pattern_score=5.0,
            signal="HOLD", regime_label=None,
            trend_score=None, contrarian_score=None,
            dominant_group=None, error="Network timeout",
        )
        assert tr.error == "Network timeout"


# ===========================================================================
# Tests: MultiTimeframeResult dataclass
# ===========================================================================


class TestMultiTimeframeResult:
    def test_properties(self):
        tr1 = TimeframeResult(
            timeframe="1d", period="2y", weight=0.5,
            indicator_score=7.0, pattern_score=6.0,
            signal="BUY", regime_label="Strong Trend",
            trend_score=7.5, contrarian_score=5.0,
            dominant_group="trend",
        )
        tr2 = TimeframeResult(
            timeframe="1wk", period="5y", weight=0.3,
            indicator_score=5.0, pattern_score=5.0,
            signal="HOLD", regime_label=None,
            trend_score=None, contrarian_score=None,
            dominant_group=None, error="Failed",
        )
        mtr = MultiTimeframeResult(
            ticker="AAPL",
            timeframe_results=[tr1, tr2],
            aggregated_indicator_score=6.5,
            aggregated_pattern_score=5.8,
            aggregated_signal="BUY",
            agreement="mixed",
        )
        assert mtr.n_timeframes == 2
        assert len(mtr.successful_timeframes) == 1
        assert mtr.successful_timeframes[0].timeframe == "1d"


# ===========================================================================
# Tests: MultiTimeframeAnalyzer
# ===========================================================================


class TestMultiTimeframeAnalyzer:
    """Test the analyzer using mocked Analyzer.run() calls."""

    def test_basic_aggregation(self, default_config, mock_provider):
        """All 3 timeframes succeed — weighted average is correct."""
        daily = _make_analysis_result(overall=8.0, pattern_overall=7.0)
        weekly = _make_analysis_result(overall=6.0, pattern_overall=5.0)
        monthly = _make_analysis_result(overall=4.0, pattern_overall=3.0)

        with patch("analysis.multi_timeframe.Analyzer") as MockAnalyzer:
            instance = MockAnalyzer.return_value
            instance.run.side_effect = [daily, weekly, monthly]

            mta = MultiTimeframeAnalyzer(default_config, mock_provider)
            result = mta.run("AAPL")

        assert result.ticker == "AAPL"
        assert result.n_timeframes == 3
        assert len(result.successful_timeframes) == 3

        # Expected weighted average: (8*0.5 + 6*0.3 + 4*0.2) / (0.5+0.3+0.2) = 6.6
        assert abs(result.aggregated_indicator_score - 6.6) < 0.01
        # Pattern: (7*0.5 + 5*0.3 + 3*0.2) / 1.0 = 5.6
        assert abs(result.aggregated_pattern_score - 5.6) < 0.01

    def test_all_timeframes_agree_buy(self, default_config, mock_provider):
        """All timeframes produce BUY — agreement should be 'aligned'."""
        buy_result = _make_analysis_result(overall=8.0, pattern_overall=8.0)

        with patch("analysis.multi_timeframe.Analyzer") as MockAnalyzer:
            instance = MockAnalyzer.return_value
            instance.run.return_value = buy_result

            mta = MultiTimeframeAnalyzer(default_config, mock_provider)
            result = mta.run("TSLA")

        assert result.agreement == "aligned"
        for tr in result.timeframe_results:
            assert tr.signal == "BUY"

    def test_conflicting_signals(self, default_config, mock_provider):
        """Daily=BUY, Weekly=SELL → 'conflicting'."""
        daily = _make_analysis_result(overall=8.0, pattern_overall=8.0)
        weekly = _make_analysis_result(overall=2.0, pattern_overall=2.0)
        monthly = _make_analysis_result(overall=5.0, pattern_overall=5.0)

        with patch("analysis.multi_timeframe.Analyzer") as MockAnalyzer:
            instance = MockAnalyzer.return_value
            instance.run.side_effect = [daily, weekly, monthly]

            mta = MultiTimeframeAnalyzer(default_config, mock_provider)
            result = mta.run("MSFT")

        assert result.agreement == "conflicting"

    def test_mixed_signals(self, default_config, mock_provider):
        """Daily=BUY, Weekly=HOLD, Monthly=HOLD → 'mixed'."""
        daily = _make_analysis_result(overall=8.0, pattern_overall=8.0)
        weekly = _make_analysis_result(overall=5.0, pattern_overall=5.0)
        monthly = _make_analysis_result(overall=5.0, pattern_overall=5.0)

        with patch("analysis.multi_timeframe.Analyzer") as MockAnalyzer:
            instance = MockAnalyzer.return_value
            instance.run.side_effect = [daily, weekly, monthly]

            mta = MultiTimeframeAnalyzer(default_config, mock_provider)
            result = mta.run("GOOG")

        assert result.agreement == "mixed"

    def test_one_timeframe_fails(self, default_config, mock_provider):
        """One timeframe throws — should still produce result with error."""
        daily = _make_analysis_result(overall=7.0, pattern_overall=6.0)
        monthly = _make_analysis_result(overall=5.0, pattern_overall=4.0)

        with patch("analysis.multi_timeframe.Analyzer") as MockAnalyzer:
            instance = MockAnalyzer.return_value
            instance.run.side_effect = [
                daily,
                ValueError("No data for weekly"),
                monthly,
            ]

            mta = MultiTimeframeAnalyzer(default_config, mock_provider)
            result = mta.run("NVDA")

        assert result.n_timeframes == 3
        assert len(result.successful_timeframes) == 2

        errors = [tr for tr in result.timeframe_results if tr.error is not None]
        assert len(errors) == 1
        assert "No data for weekly" in errors[0].error

    def test_all_timeframes_fail(self, default_config, mock_provider):
        """All timeframes fail — defaults to 5.0/HOLD."""
        with patch("analysis.multi_timeframe.Analyzer") as MockAnalyzer:
            instance = MockAnalyzer.return_value
            instance.run.side_effect = Exception("boom")

            mta = MultiTimeframeAnalyzer(default_config, mock_provider)
            result = mta.run("FAIL")

        assert result.aggregated_indicator_score == 5.0
        assert result.aggregated_pattern_score == 5.0
        assert result.aggregated_signal == "HOLD"
        assert len(result.successful_timeframes) == 0

    def test_custom_weights(self, mock_provider):
        """Custom weights are respected in aggregation."""
        cfg_data = {
            "multi_timeframe": {
                "enabled": True,
                "timeframes": ["1d", "1wk"],
                "weights": {"1d": 0.8, "1wk": 0.2},
                "periods": {"1d": "1y", "1wk": "2y"},
            },
        }
        cfg = Config.from_dict(cfg_data)

        daily = _make_analysis_result(overall=10.0, pattern_overall=10.0)
        weekly = _make_analysis_result(overall=0.0, pattern_overall=0.0)

        with patch("analysis.multi_timeframe.Analyzer") as MockAnalyzer:
            instance = MockAnalyzer.return_value
            instance.run.side_effect = [daily, weekly]

            mta = MultiTimeframeAnalyzer(cfg, mock_provider)
            result = mta.run("TEST")

        # Expected: (10*0.8 + 0*0.2) / 1.0 = 8.0
        assert abs(result.aggregated_indicator_score - 8.0) < 0.01
        assert result.n_timeframes == 2

    def test_aggregation_excludes_failed(self, default_config, mock_provider):
        """Failed timeframes are excluded from weighted average."""
        # Only daily succeeds (weight 0.5)
        daily = _make_analysis_result(overall=8.0, pattern_overall=7.0)

        with patch("analysis.multi_timeframe.Analyzer") as MockAnalyzer:
            instance = MockAnalyzer.return_value
            instance.run.side_effect = [
                daily,
                ValueError("weekly fail"),
                ValueError("monthly fail"),
            ]

            mta = MultiTimeframeAnalyzer(default_config, mock_provider)
            result = mta.run("ONLY_DAILY")

        # Only daily succeeded — aggregated = daily scores
        assert abs(result.aggregated_indicator_score - 8.0) < 0.01
        assert abs(result.aggregated_pattern_score - 7.0) < 0.01

    def test_regime_label_propagated(self, default_config, mock_provider):
        """Regime labels from AnalysisResult are carried into TimeframeResult."""
        ar = _make_analysis_result(regime_label="Mean Reverting")

        with patch("analysis.multi_timeframe.Analyzer") as MockAnalyzer:
            instance = MockAnalyzer.return_value
            instance.run.return_value = ar

            mta = MultiTimeframeAnalyzer(default_config, mock_provider)
            result = mta.run("SPY")

        for tr in result.successful_timeframes:
            assert tr.regime_label == "Mean Reverting"

    def test_no_regime(self, default_config, mock_provider):
        """AnalysisResult with regime=None produces regime_label=None."""
        ar = _make_analysis_result()
        ar.regime = None

        with patch("analysis.multi_timeframe.Analyzer") as MockAnalyzer:
            instance = MockAnalyzer.return_value
            instance.run.return_value = ar

            mta = MultiTimeframeAnalyzer(default_config, mock_provider)
            result = mta.run("QQQ")

        for tr in result.successful_timeframes:
            assert tr.regime_label is None

    def test_timeframe_periods_used_correctly(self, default_config, mock_provider):
        """Each timeframe is called with its configured period."""
        ar = _make_analysis_result()

        with patch("analysis.multi_timeframe.Analyzer") as MockAnalyzer:
            instance = MockAnalyzer.return_value
            instance.run.return_value = ar

            mta = MultiTimeframeAnalyzer(default_config, mock_provider)
            mta.run("AAPL")

        calls = instance.run.call_args_list
        assert len(calls) == 3

        # Default periods: 1d=2y, 1wk=5y, 1mo=max
        assert calls[0].kwargs.get("period") == "2y" or calls[0][1].get("period") == "2y"
        assert calls[0].kwargs.get("interval") == "1d" or calls[0][1].get("interval") == "1d"

        assert calls[1].kwargs.get("period") == "5y" or calls[1][1].get("period") == "5y"
        assert calls[1].kwargs.get("interval") == "1wk" or calls[1][1].get("interval") == "1wk"

        assert calls[2].kwargs.get("period") == "max" or calls[2][1].get("period") == "max"
        assert calls[2].kwargs.get("interval") == "1mo" or calls[2][1].get("interval") == "1mo"


# ===========================================================================
# Tests: Config integration
# ===========================================================================


class TestConfigIntegration:
    def test_default_config_has_multi_timeframe(self):
        cfg = Config.defaults()
        mt = cfg.section("multi_timeframe")
        assert mt["enabled"] is False
        assert mt["timeframes"] == ["1d", "1wk", "1mo"]
        assert mt["weights"]["1d"] == 0.5
        assert mt["weights"]["1wk"] == 0.3
        assert mt["weights"]["1mo"] == 0.2
        assert mt["periods"]["1d"] == "2y"
        assert mt["periods"]["1wk"] == "5y"
        assert mt["periods"]["1mo"] == "max"

    def test_from_dict_override(self):
        cfg = Config.from_dict({
            "multi_timeframe": {
                "enabled": True,
                "weights": {"1d": 0.6, "1wk": 0.3, "1mo": 0.1},
            },
        })
        mt = cfg.section("multi_timeframe")
        assert mt["enabled"] is True
        assert mt["weights"]["1d"] == 0.6
        assert mt["weights"]["1mo"] == 0.1
        # Timeframes should still be default
        assert mt["timeframes"] == ["1d", "1wk", "1mo"]
