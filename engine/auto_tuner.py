"""
engine/auto_tuner.py — Bayesian parameter optimisation via Optuna.

Uses walk-forward testing as the validation method (never just in-sample)
to find optimal strategy parameters for a given objective.

Objectives:
    beat_buy_hold     – Maximise excess return over buy-and-hold
    max_return        – Maximise absolute annualised return
    max_risk_adjusted – Maximise Sharpe ratio
    min_drawdown      – Minimise max drawdown while keeping positive returns
    balanced          – Weighted combination of return, drawdown, and win rate

Usage:
    from engine.auto_tuner import AutoTuner

    tuner = AutoTuner(data_provider, cfg)
    result = tuner.run("AAPL", objective="balanced", n_trials=50)
"""

from __future__ import annotations

import copy
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import optuna

# Silence Optuna's verbose logs by default
optuna.logging.set_verbosity(optuna.logging.WARNING)

if TYPE_CHECKING:
    from config import Config
    from data.provider import DataProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    """Result of a single Optuna trial."""
    trial_number: int
    params: dict[str, Any]
    objective_value: float
    # Walk-forward aggregate metrics
    avg_return_pct: float = 0.0
    avg_annualized_return_pct: float = 0.0
    avg_max_drawdown_pct: float = 0.0
    avg_sharpe_ratio: float = 0.0
    avg_win_rate_pct: float = 0.0
    avg_profit_factor: float = 0.0
    stability_score: float = 0.0
    total_windows: int = 0
    buy_hold_return_pct: float | None = None


@dataclass
class SensitivityEntry:
    """How a single parameter affects the objective."""
    param_name: str
    importance: float       # 0-1 relative importance
    best_value: Any         # value in the best trial
    value_range: list[Any]  # [min_tried, max_tried]


@dataclass
class AutoTuneResult:
    """Complete auto-tuner output."""
    ticker: str
    objective: str
    objective_label: str
    n_trials: int
    elapsed_seconds: float = 0.0

    # Best trial
    best_params: dict[str, Any] = field(default_factory=dict)
    best_objective_value: float = 0.0

    # Best trial walk-forward metrics
    best_avg_return_pct: float = 0.0
    best_avg_annualized_return_pct: float = 0.0
    best_avg_max_drawdown_pct: float = 0.0
    best_avg_sharpe_ratio: float = 0.0
    best_avg_win_rate_pct: float = 0.0
    best_avg_profit_factor: float = 0.0
    best_stability_score: float = 0.0

    # Baseline (default params) walk-forward metrics
    baseline_avg_return_pct: float = 0.0
    baseline_avg_annualized_return_pct: float = 0.0
    baseline_avg_max_drawdown_pct: float = 0.0
    baseline_avg_sharpe_ratio: float = 0.0
    baseline_avg_win_rate_pct: float = 0.0
    baseline_objective_value: float = 0.0

    # Buy-and-hold comparison (average across walk-forward windows)
    buy_hold_return_pct: float | None = None

    # Sensitivity analysis (for power user mode)
    sensitivity: list[SensitivityEntry] = field(default_factory=list)

    # All trial results (for power user mode)
    trials: list[TrialResult] = field(default_factory=list)

    # Verdict
    verdict: str = ""
    improvement_pct: float = 0.0  # % improvement over baseline


# ---------------------------------------------------------------------------
# Objective labels
# ---------------------------------------------------------------------------

OBJECTIVE_LABELS = {
    "beat_buy_hold": "Beat Buy-and-Hold",
    "max_return": "Maximise Return",
    "max_risk_adjusted": "Maximise Risk-Adjusted Return",
    "min_drawdown": "Minimise Drawdown",
    "balanced": "Balanced",
}


# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------

def _define_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """Define the Optuna search space for high-impact parameters.

    Returns a flat dict of parameter overrides keyed by their config path
    (e.g. "strategy.score_thresholds.short_below").

    We focus on the ~25 highest-impact params to keep the search tractable
    while covering all major strategy levers.
    """
    params: dict[str, Any] = {}

    # --- Strategy thresholds (highest impact) ---
    short_below = trial.suggest_float("strategy.score_thresholds.short_below", 1.5, 5.0, step=0.25)
    hold_below = trial.suggest_float("strategy.score_thresholds.hold_below", short_below + 0.5, 8.5, step=0.25)
    params["strategy.score_thresholds.short_below"] = short_below
    params["strategy.score_thresholds.hold_below"] = hold_below

    # --- Risk management ---
    params["strategy.stop_loss_pct"] = trial.suggest_float("strategy.stop_loss_pct", 0.02, 0.12, step=0.01)
    params["strategy.take_profit_pct"] = trial.suggest_float("strategy.take_profit_pct", 0.08, 0.60, step=0.02)
    params["strategy.atr_stop_multiplier"] = trial.suggest_float("strategy.atr_stop_multiplier", 2.0, 8.0, step=0.5)

    # --- Indicator composite weights ---
    params["overall.weights.rsi"] = trial.suggest_float("overall.weights.rsi", 0.05, 0.30, step=0.05)
    params["overall.weights.macd"] = trial.suggest_float("overall.weights.macd", 0.05, 0.30, step=0.05)
    params["overall.weights.bollinger_bands"] = trial.suggest_float("overall.weights.bollinger_bands", 0.0, 0.25, step=0.05)
    params["overall.weights.moving_averages"] = trial.suggest_float("overall.weights.moving_averages", 0.05, 0.35, step=0.05)
    params["overall.weights.stochastic"] = trial.suggest_float("overall.weights.stochastic", 0.0, 0.25, step=0.05)
    params["overall.weights.adx"] = trial.suggest_float("overall.weights.adx", 0.0, 0.25, step=0.05)
    params["overall.weights.volume"] = trial.suggest_float("overall.weights.volume", 0.0, 0.25, step=0.05)

    # --- Combination mode & weights ---
    params["strategy.combination_mode"] = trial.suggest_categorical(
        "strategy.combination_mode", ["weighted", "boost"]
    )
    params["strategy.indicator_weight"] = trial.suggest_float("strategy.indicator_weight", 0.4, 0.95, step=0.05)
    params["strategy.pattern_weight"] = trial.suggest_float("strategy.pattern_weight", 0.05, 0.6, step=0.05)

    # --- Score spreading ---
    params["overall.score_spreading.factor"] = trial.suggest_float("overall.score_spreading.factor", 1.0, 3.5, step=0.25)

    # --- Key indicator periods ---
    params["rsi.period"] = trial.suggest_int("rsi.period", 7, 25)
    params["macd.fast_period"] = trial.suggest_int("macd.fast_period", 6, 18)
    params["macd.slow_period"] = trial.suggest_int("macd.slow_period", 18, 40)

    # --- Strategy behaviour ---
    params["strategy.trend_confirm_period"] = trial.suggest_int("strategy.trend_confirm_period", 8, 50)
    params["strategy.rebalance_interval"] = trial.suggest_int("strategy.rebalance_interval", 1, 15)
    params["strategy.max_hold_bars"] = trial.suggest_int("strategy.max_hold_bars", 0, 40, step=5)
    params["strategy.global_trend_bias"] = trial.suggest_categorical(
        "strategy.global_trend_bias", [True, False]
    )

    return params


def _apply_params_to_config(cfg: "Config", params: dict[str, Any]) -> "Config":
    """Apply a flat param dict to a deep-copied Config.

    Keys use dot notation (e.g. "strategy.stop_loss_pct").  Nested keys
    like "strategy.score_thresholds.short_below" are expanded.
    """
    new_cfg = type(cfg)(copy.deepcopy(cfg._data), cfg.path)
    new_cfg._active_objective = cfg._active_objective

    for key, value in params.items():
        parts = key.split(".")
        d = new_cfg._data
        for p in parts[:-1]:
            if p not in d:
                d[p] = {}
            d = d[p]
        d[parts[-1]] = value

    return new_cfg


# ---------------------------------------------------------------------------
# Objective scoring functions
# ---------------------------------------------------------------------------

def _compute_buy_hold_return(provider: "DataProvider", ticker: str,
                              train_years: int, test_years: int) -> float | None:
    """Estimate buy-and-hold return averaged over walk-forward windows.

    Uses the same window layout as WalkForwardEngine.  Returns average
    annualised return, or None if data is unavailable.
    """
    from engine.walk_forward import WalkForwardEngine  # type: ignore[import-untyped]
    import pandas as pd

    try:
        # Fetch a long history
        total_years = train_years + test_years + 10  # extra buffer
        df = provider.get_ohlcv(ticker, period=f"{total_years}y", interval="1d")
        if df is None or len(df) < 252:
            return None

        # Generate windows
        engine = WalkForwardEngine.__new__(WalkForwardEngine)
        engine._provider = provider
        engine._strategy_factory = lambda: None  # unused
        engine._cfg = None  # unused
        windows = engine._generate_windows(train_years, test_years, max_windows=10)

        if not windows:
            return None

        returns = []
        for w in windows:
            start = pd.Timestamp(w["test_start"])
            end = pd.Timestamp(w["test_end"])
            mask = (df.index >= start) & (df.index <= end)
            window_df = df[mask]
            if len(window_df) < 20:
                continue
            ret = (window_df["Close"].iloc[-1] / window_df["Close"].iloc[0] - 1) * 100
            returns.append(ret)

        return statistics.mean(returns) if returns else None
    except Exception:
        return None


def _score_trial(
    objective: str,
    avg_return: float,
    avg_annualized_return: float,
    avg_max_dd: float,
    avg_sharpe: float,
    avg_win_rate: float,
    avg_profit_factor: float,
    stability: float,
    buy_hold_return: float | None,
) -> float:
    """Compute a scalar score for an Optuna trial based on the objective.

    Higher is always better (Optuna maximises).
    """
    if objective == "beat_buy_hold":
        bh = buy_hold_return if buy_hold_return is not None else 0.0
        excess = avg_return - bh
        # Reward excess return, penalise instability
        return excess + 0.1 * stability

    elif objective == "max_return":
        # Pure return, small stability bonus
        return avg_annualized_return + 0.05 * stability

    elif objective == "max_risk_adjusted":
        # Sharpe is the primary metric
        return avg_sharpe * 100 + 0.1 * stability

    elif objective == "min_drawdown":
        # Minimise drawdown = maximise negative drawdown
        # But also require positive return
        return_bonus = max(0, avg_annualized_return) * 0.5
        return -abs(avg_max_dd) + return_bonus + 0.1 * stability

    elif objective == "balanced":
        # Weighted combination
        return (
            avg_annualized_return * 0.35
            + avg_sharpe * 30                   # ~30x to scale with returns
            - abs(avg_max_dd) * 0.20
            + avg_win_rate * 0.10
            + stability * 0.05
        )

    else:
        # Fallback: balanced
        return avg_annualized_return


# ---------------------------------------------------------------------------
# Auto-Tuner engine
# ---------------------------------------------------------------------------

class AutoTuner:
    """Bayesian optimisation of strategy parameters using walk-forward validation.

    Each trial:
      1. Samples parameters from the search space.
      2. Builds a Config with those parameters applied.
      3. Runs walk-forward testing (3 windows for speed, using 3Y train / 1Y test).
      4. Computes the objective score from walk-forward aggregate metrics.
    """

    def __init__(
        self,
        data_provider: "DataProvider",
        cfg: "Config",
    ) -> None:
        self._provider = data_provider
        self._base_cfg = cfg

    def run(
        self,
        ticker: str,
        objective: str = "balanced",
        n_trials: int = 30,
        train_years: int = 3,
        test_years: int = 1,
        max_windows: int = 3,
    ) -> AutoTuneResult:
        """Run the auto-tuner and return results.

        Args:
            ticker:       Stock symbol.
            objective:    One of the 5 objective keys.
            n_trials:     Number of Optuna trials (more = better but slower).
            train_years:  Walk-forward training window length.
            test_years:   Walk-forward test window length.
            max_windows:  Walk-forward windows per trial (fewer = faster).

        Returns:
            :class:`AutoTuneResult` with best params, metrics, and sensitivity.
        """
        from engine.walk_forward import WalkForwardEngine  # type: ignore[import-untyped]
        from engine.score_strategy import ScoreBasedStrategy  # type: ignore[import-untyped]
        from engine.suitability import TradingMode  # type: ignore[import-untyped]

        ticker = ticker.upper()
        start_time = time.time()

        result = AutoTuneResult(
            ticker=ticker,
            objective=objective,
            objective_label=OBJECTIVE_LABELS.get(objective, objective),
            n_trials=n_trials,
        )

        # Pre-compute buy-and-hold return for the "beat_buy_hold" objective
        buy_hold_ret = _compute_buy_hold_return(
            self._provider, ticker, train_years, test_years
        )
        result.buy_hold_return_pct = buy_hold_ret

        # --- Run baseline (default params) ---
        logger.info("Running baseline walk-forward for %s ...", ticker)
        baseline_wf = self._run_walk_forward(
            ticker, self._base_cfg, train_years, test_years, max_windows,
        )
        result.baseline_avg_return_pct = baseline_wf["avg_return_pct"]
        result.baseline_avg_annualized_return_pct = baseline_wf["avg_annualized_return_pct"]
        result.baseline_avg_max_drawdown_pct = baseline_wf["avg_max_drawdown_pct"]
        result.baseline_avg_sharpe_ratio = baseline_wf["avg_sharpe_ratio"]
        result.baseline_avg_win_rate_pct = baseline_wf["avg_win_rate_pct"]
        result.baseline_objective_value = _score_trial(
            objective,
            baseline_wf["avg_return_pct"],
            baseline_wf["avg_annualized_return_pct"],
            baseline_wf["avg_max_drawdown_pct"],
            baseline_wf["avg_sharpe_ratio"],
            baseline_wf["avg_win_rate_pct"],
            baseline_wf["avg_profit_factor"],
            baseline_wf["stability_score"],
            buy_hold_ret,
        )

        # --- Optuna study ---
        all_trials: list[TrialResult] = []

        def optuna_objective(trial: optuna.Trial) -> float:
            params = _define_search_space(trial)
            trial_cfg = _apply_params_to_config(self._base_cfg, params)

            wf = self._run_walk_forward(
                ticker, trial_cfg, train_years, test_years, max_windows,
            )

            score = _score_trial(
                objective,
                wf["avg_return_pct"],
                wf["avg_annualized_return_pct"],
                wf["avg_max_drawdown_pct"],
                wf["avg_sharpe_ratio"],
                wf["avg_win_rate_pct"],
                wf["avg_profit_factor"],
                wf["stability_score"],
                buy_hold_ret,
            )

            tr = TrialResult(
                trial_number=trial.number,
                params=params,
                objective_value=score,
                avg_return_pct=wf["avg_return_pct"],
                avg_annualized_return_pct=wf["avg_annualized_return_pct"],
                avg_max_drawdown_pct=wf["avg_max_drawdown_pct"],
                avg_sharpe_ratio=wf["avg_sharpe_ratio"],
                avg_win_rate_pct=wf["avg_win_rate_pct"],
                avg_profit_factor=wf["avg_profit_factor"],
                stability_score=wf["stability_score"],
                total_windows=wf["total_windows"],
                buy_hold_return_pct=buy_hold_ret,
            )
            all_trials.append(tr)

            return score

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=False)

        # --- Extract best trial ---
        best = study.best_trial
        best_tr = next((t for t in all_trials if t.trial_number == best.number), None)

        if best_tr:
            result.best_params = best_tr.params
            result.best_objective_value = best_tr.objective_value
            result.best_avg_return_pct = best_tr.avg_return_pct
            result.best_avg_annualized_return_pct = best_tr.avg_annualized_return_pct
            result.best_avg_max_drawdown_pct = best_tr.avg_max_drawdown_pct
            result.best_avg_sharpe_ratio = best_tr.avg_sharpe_ratio
            result.best_avg_win_rate_pct = best_tr.avg_win_rate_pct
            result.best_avg_profit_factor = best_tr.avg_profit_factor
            result.best_stability_score = best_tr.stability_score

        # --- Improvement ---
        if result.baseline_objective_value != 0:
            result.improvement_pct = (
                (result.best_objective_value - result.baseline_objective_value)
                / abs(result.baseline_objective_value)
                * 100
            )
        elif result.best_objective_value > 0:
            result.improvement_pct = 100.0

        # --- Sensitivity analysis ---
        try:
            importances = optuna.importance.get_param_importances(study)
            for param_name, importance_val in importances.items():
                tried_values = [t.params.get(param_name) for t in all_trials if param_name in t.params]
                numeric_vals = [v for v in tried_values if isinstance(v, (int, float))]
                val_range = (
                    [min(numeric_vals), max(numeric_vals)]
                    if numeric_vals
                    else [tried_values[0], tried_values[-1]] if tried_values else []
                )
                best_val = result.best_params.get(param_name)
                result.sensitivity.append(SensitivityEntry(
                    param_name=param_name,
                    importance=importance_val,
                    best_value=best_val,
                    value_range=val_range,
                ))
        except Exception as exc:
            logger.warning("Could not compute parameter importances: %s", exc)

        # --- All trials (sorted by objective desc) ---
        result.trials = sorted(all_trials, key=lambda t: t.objective_value, reverse=True)
        result.n_trials = len(all_trials)

        # --- Verdict ---
        result.elapsed_seconds = time.time() - start_time
        result.verdict = self._make_verdict(result)

        return result

    def _run_walk_forward(
        self,
        ticker: str,
        cfg: "Config",
        train_years: int,
        test_years: int,
        max_windows: int,
    ) -> dict[str, Any]:
        """Run walk-forward testing with the given config and return metric dict."""
        from engine.walk_forward import WalkForwardEngine  # type: ignore[import-untyped]
        from engine.score_strategy import ScoreBasedStrategy  # type: ignore[import-untyped]
        from engine.suitability import TradingMode  # type: ignore[import-untyped]

        strat_section = cfg.section("strategy")

        def factory() -> ScoreBasedStrategy:
            return ScoreBasedStrategy(
                params=dict(strat_section),
                trading_mode=TradingMode.LONG_SHORT,
            )

        wf_engine = WalkForwardEngine(
            data_provider=self._provider,
            strategy_factory=factory,
            cfg=cfg,
        )

        wf_result = wf_engine.run(
            ticker=ticker,
            train_years=train_years,
            test_years=test_years,
            max_windows=max_windows,
        )

        return {
            "avg_return_pct": wf_result.avg_return_pct,
            "avg_annualized_return_pct": wf_result.avg_annualized_return_pct,
            "avg_max_drawdown_pct": wf_result.avg_max_drawdown_pct,
            "avg_sharpe_ratio": wf_result.avg_sharpe_ratio,
            "avg_win_rate_pct": wf_result.avg_win_rate_pct,
            "avg_profit_factor": wf_result.avg_profit_factor,
            "stability_score": wf_result.stability_score,
            "total_windows": wf_result.total_windows,
        }

    @staticmethod
    def _make_verdict(result: AutoTuneResult) -> str:
        """Generate a human-readable verdict."""
        obj_label = result.objective_label
        imp = result.improvement_pct

        if imp > 0:
            verdict = (
                f"Optimised for {obj_label}. "
                f"The tuned parameters improved the objective by {imp:.0f}% over defaults. "
            )
        else:
            verdict = (
                f"Optimised for {obj_label}. "
                f"Tuned parameters did not improve over defaults — "
                f"the default configuration may already be well-suited. "
            )

        # Add concrete numbers
        if result.objective == "beat_buy_hold" and result.buy_hold_return_pct is not None:
            verdict += (
                f"Strategy return: {result.best_avg_return_pct:.1f}% "
                f"vs buy-and-hold: {result.buy_hold_return_pct:.1f}%."
            )
        elif result.objective == "max_return":
            verdict += (
                f"Annualised return: {result.best_avg_annualized_return_pct:.1f}% "
                f"(baseline: {result.baseline_avg_annualized_return_pct:.1f}%)."
            )
        elif result.objective == "max_risk_adjusted":
            verdict += (
                f"Sharpe ratio: {result.best_avg_sharpe_ratio:.2f} "
                f"(baseline: {result.baseline_avg_sharpe_ratio:.2f})."
            )
        elif result.objective == "min_drawdown":
            verdict += (
                f"Max drawdown: {result.best_avg_max_drawdown_pct:.1f}% "
                f"(baseline: {result.baseline_avg_max_drawdown_pct:.1f}%)."
            )
        else:  # balanced
            verdict += (
                f"Return: {result.best_avg_annualized_return_pct:.1f}%, "
                f"Sharpe: {result.best_avg_sharpe_ratio:.2f}, "
                f"Max DD: {result.best_avg_max_drawdown_pct:.1f}%."
            )

        return verdict
