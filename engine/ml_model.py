"""
engine/ml_model.py — XGBoost signal scoring model: training, evaluation, prediction.

This module implements the full ML pipeline for scoring trading signals:
  1. Training data generation — runs analysis at historical points, labels
     with forward returns.
  2. Walk-forward training — trains on expanding windows, validates on
     held-out recent data.
  3. Prediction — scores new signals with probability + feature importances.

The model is saved/loaded from disk so it persists across restarts.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from engine.ml_features import (
    FEATURE_NAMES,
    NUM_FEATURES,
    extract_features,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_FILE = MODEL_DIR / "xgb_signal_model.pkl"
META_FILE = MODEL_DIR / "xgb_signal_meta.json"

# Forward return horizons by trade mode
FORWARD_BARS: dict[str, int] = {
    "swing": 10,
    "long_term": 60,
}

# Minimum win threshold — the forward return must exceed this to be
# labelled "win". Using a small positive threshold avoids counting
# tiny gains as wins when they wouldn't cover costs.
WIN_THRESHOLD_PCT = 0.5  # 0.5%

# Walk-forward configuration
WF_TRAIN_RATIO = 0.75     # 75% train, 25% test in each window
WF_NUM_WINDOWS = 4        # number of walk-forward windows
WF_MIN_SAMPLES = 200      # minimum training samples per window

# Sampling: how often to take a sample from each ticker's history
SAMPLE_INTERVAL_BARS = 20  # every 20 bars ≈ monthly for daily data


# ── Data classes ──────────────────────────────────────────────────────


@dataclass
class TrainingSample:
    """A single labeled training sample."""
    ticker: str
    date: str
    features: np.ndarray       # shape (NUM_FEATURES,)
    label: int                 # 1 = win, 0 = loss
    forward_return_pct: float  # actual forward return


@dataclass
class ModelMetrics:
    """Evaluation metrics for the model."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc_roc: float = 0.0
    n_train: int = 0
    n_test: int = 0
    feature_importances: dict[str, float] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Results from one walk-forward window."""
    window_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    metrics: ModelMetrics


@dataclass
class TrainingResult:
    """Full training result with walk-forward validation."""
    total_samples: int
    total_tickers: int
    trade_mode: str
    forward_bars: int
    walk_forward_results: list[WalkForwardResult] = field(default_factory=list)
    final_metrics: ModelMetrics = field(default_factory=ModelMetrics)
    elapsed_seconds: float = 0.0
    trained_at: str = ""
    model_path: str = ""


@dataclass
class PredictionResult:
    """Prediction for a single signal."""
    probability: float           # 0-100 scale (win probability)
    ai_rating: float             # same as probability, 0-100
    label: str                   # "Bullish", "Bearish", "Neutral"
    confidence: str              # "High", "Medium", "Low"
    top_features: list[dict]     # top contributing features [{name, value, importance}]


# ── Training data generation ──────────────────────────────────────────


def generate_training_data(
    tickers: list[str],
    trade_mode: str = "swing",
    period: str = "5y",
    progress_callback: Any | None = None,
) -> list[TrainingSample]:
    """Generate labeled training samples from historical data.

    For each ticker, we:
      1. Fetch 5y of daily data.
      2. At regular intervals (every SAMPLE_INTERVAL_BARS bars), slice
         the data up to that point, run the full indicator/pattern/regime
         pipeline, and extract features.
      3. Look at the forward return over the next N bars to label as
         win (1) or loss (0).

    Args:
        tickers: List of ticker symbols.
        trade_mode: "swing" or "long_term" — determines forward horizon.
        period: How much history to fetch per ticker.
        progress_callback: Optional callable(ticker, i, total) for progress.

    Returns:
        List of TrainingSample objects.
    """
    # Late imports for engine modules (volume-mounted in Docker)
    from config import Config                           # type: ignore[import-untyped]
    from data.yahoo import fetch_batch                  # type: ignore[import-untyped]
    from indicators.registry import IndicatorRegistry   # type: ignore[import-untyped]
    from patterns.registry import PatternRegistry       # type: ignore[import-untyped]
    from analysis.scorer import CompositeScorer         # type: ignore[import-untyped]
    from analysis.pattern_scorer import PatternCompositeScorer  # type: ignore[import-untyped]
    from engine.regime import RegimeClassifier          # type: ignore[import-untyped]

    forward_bars = FORWARD_BARS.get(trade_mode, 10)
    cfg = Config.defaults()

    # Apply trade-mode objective
    obj_map = {"swing": "swing_trade", "long_term": "long_term"}
    obj = obj_map.get(trade_mode)
    if obj and obj in cfg.available_objectives():
        cfg.apply_objective(obj)

    # Build reusable engine instances
    ind_registry = IndicatorRegistry(cfg)
    pat_registry = PatternRegistry(cfg)
    scorer = CompositeScorer(cfg)
    pat_scorer = PatternCompositeScorer(cfg)
    regime_clf = RegimeClassifier(cfg)

    # Minimum bars needed to compute indicators + have forward return
    min_bars = 200 + forward_bars

    # ── Batch fetch ──────────────────────────────────────────────────
    logger.info(
        "ML training: fetching %d tickers (period=%s, forward=%d bars)",
        len(tickers), period, forward_bars,
    )
    all_data = fetch_batch(tickers, period=period)
    logger.info("ML training: got data for %d / %d tickers", len(all_data), len(tickers))

    samples: list[TrainingSample] = []
    total = len(all_data)

    for i, (ticker, df) in enumerate(all_data.items()):
        if progress_callback:
            progress_callback(ticker, i + 1, total)

        if len(df) < min_bars:
            logger.debug("ML training: skipping %s — only %d bars", ticker, len(df))
            continue

        try:
            ticker_samples = _generate_samples_for_ticker(
                ticker, df, forward_bars,
                ind_registry, pat_registry, scorer, pat_scorer, regime_clf,
            )
            samples.extend(ticker_samples)
        except Exception:
            logger.debug("ML training: error on %s", ticker, exc_info=True)
            continue

    logger.info(
        "ML training: generated %d samples from %d tickers",
        len(samples), total,
    )
    return samples


def _generate_samples_for_ticker(
    ticker: str,
    df: pd.DataFrame,
    forward_bars: int,
    ind_registry: Any,
    pat_registry: Any,
    scorer: Any,
    pat_scorer: Any,
    regime_clf: Any,
) -> list[TrainingSample]:
    """Generate samples from a single ticker's historical data.

    We walk through the data, stopping every SAMPLE_INTERVAL_BARS bars.
    At each stop, we slice df[:bar_idx], run the full analysis, extract
    features, and compute the forward return from bar_idx to
    bar_idx + forward_bars.
    """
    samples: list[TrainingSample] = []
    n = len(df)

    # Start after 200 bars (need enough history for indicators)
    # Stop forward_bars before the end (need forward return)
    start_bar = 200
    end_bar = n - forward_bars

    for bar_idx in range(start_bar, end_bar, SAMPLE_INTERVAL_BARS):
        try:
            # Slice data up to this point (no lookahead)
            df_slice = df.iloc[:bar_idx + 1].copy()

            # Run indicators on the slice
            indicator_results = ind_registry.run_all(df_slice)
            pattern_results = pat_registry.run_all(df_slice)
            composite = scorer.score(indicator_results)
            pattern_composite = pat_scorer.score(pattern_results)

            # Regime
            regime = None
            try:
                regime = regime_clf.classify(df_slice)
            except Exception:
                pass

            # Extract features
            features = extract_features(
                indicator_results, pattern_results,
                composite, pattern_composite,
                regime, df_slice,
            )

            # Compute forward return
            current_price = float(df["close"].iloc[bar_idx])
            future_price = float(df["close"].iloc[bar_idx + forward_bars])
            forward_return_pct = (
                (future_price - current_price) / current_price * 100
            )

            # Label: 1 = win (return > threshold), 0 = loss
            label = 1 if forward_return_pct > WIN_THRESHOLD_PCT else 0

            # Date for this sample
            date_val = df.index[bar_idx]
            date_str = (
                date_val.strftime("%Y-%m-%d")
                if hasattr(date_val, "strftime")
                else str(date_val)
            )

            samples.append(TrainingSample(
                ticker=ticker,
                date=date_str,
                features=features,
                label=label,
                forward_return_pct=round(forward_return_pct, 4),
            ))

        except Exception:
            continue

    return samples


# ── Walk-forward training ─────────────────────────────────────────────


def train_model(
    samples: list[TrainingSample],
    trade_mode: str = "swing",
    progress_callback: Any | None = None,
) -> TrainingResult:
    """Train an XGBoost model with walk-forward validation.

    Walk-forward approach:
      1. Sort samples by date.
      2. Split into WF_NUM_WINDOWS expanding windows.
      3. In each window, train on the expanding historical portion,
         validate on the held-out recent portion.
      4. The final model is trained on ALL data (for production use).

    Args:
        samples: List of TrainingSample from generate_training_data().
        trade_mode: "swing" or "long_term".
        progress_callback: Optional callable(window_idx, total_windows).

    Returns:
        TrainingResult with walk-forward metrics and model path.
    """
    import xgboost as xgb
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score,
    )

    t0 = time.time()
    forward_bars = FORWARD_BARS.get(trade_mode, 10)

    # Sort by date
    sorted_samples = sorted(samples, key=lambda s: s.date)
    n = len(sorted_samples)

    if n < WF_MIN_SAMPLES:
        raise ValueError(
            f"Not enough training samples ({n}). Need at least "
            f"{WF_MIN_SAMPLES}. Try scanning more tickers or using a "
            f"longer history period."
        )

    # Build feature matrix and labels
    X = np.array([s.features for s in sorted_samples], dtype=np.float32)
    y = np.array([s.label for s in sorted_samples], dtype=np.int32)
    dates = [s.date for s in sorted_samples]
    tickers_used = list(set(s.ticker for s in sorted_samples))

    # ── Walk-forward windows ──────────────────────────────────────────
    wf_results: list[WalkForwardResult] = []
    window_size = n // WF_NUM_WINDOWS

    for w in range(WF_NUM_WINDOWS):
        if progress_callback:
            progress_callback(w + 1, WF_NUM_WINDOWS)

        # Expanding train window: [0, split_point)
        # Test window: [split_point, split_point + window_size)
        test_start = w * window_size + int(n * WF_TRAIN_RATIO) // WF_NUM_WINDOWS
        test_end = min(test_start + window_size, n)

        if test_start >= n or test_end <= test_start:
            continue

        # Train on everything before the test window
        train_end = test_start

        if train_end < WF_MIN_SAMPLES:
            # Not enough training data for this window
            continue

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]

        if len(X_test) == 0:
            continue

        # Train XGBoost
        model = _train_xgb(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = ModelMetrics(
            accuracy=round(float(accuracy_score(y_test, y_pred)), 4),
            precision=round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            recall=round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            f1=round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            auc_roc=round(float(roc_auc_score(y_test, y_prob)), 4)
                    if len(set(y_test)) > 1 else 0.0,
            n_train=len(X_train),
            n_test=len(X_test),
        )

        wf_results.append(WalkForwardResult(
            window_idx=w,
            train_start=dates[0],
            train_end=dates[train_end - 1],
            test_start=dates[test_start],
            test_end=dates[test_end - 1],
            metrics=metrics,
        ))

    # ── Final model: train on ALL data ────────────────────────────────
    final_model = _train_xgb(X, y)

    # Feature importances from final model
    importances = final_model.feature_importances_
    feat_imp = {
        FEATURE_NAMES[i]: round(float(importances[i]), 6)
        for i in range(NUM_FEATURES)
    }
    # Sort by importance descending
    feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

    # Final metrics (last walk-forward window, or overall if only one)
    if wf_results:
        final_metrics = wf_results[-1].metrics
    else:
        # Fallback: evaluate on last 20% of data
        split = int(n * 0.8)
        y_pred = final_model.predict(X[split:])
        y_prob = final_model.predict_proba(X[split:])[:, 1]
        y_actual = y[split:]
        final_metrics = ModelMetrics(
            accuracy=round(float(accuracy_score(y_actual, y_pred)), 4),
            precision=round(float(precision_score(y_actual, y_pred, zero_division=0)), 4),
            recall=round(float(recall_score(y_actual, y_pred, zero_division=0)), 4),
            f1=round(float(f1_score(y_actual, y_pred, zero_division=0)), 4),
            auc_roc=round(float(roc_auc_score(y_actual, y_prob)), 4)
                    if len(set(y_actual)) > 1 else 0.0,
            n_train=split,
            n_test=n - split,
        )

    final_metrics.feature_importances = feat_imp

    # ── Save model ────────────────────────────────────────────────────
    _save_model(final_model, trade_mode, final_metrics, len(tickers_used), n)

    elapsed = time.time() - t0

    return TrainingResult(
        total_samples=n,
        total_tickers=len(tickers_used),
        trade_mode=trade_mode,
        forward_bars=forward_bars,
        walk_forward_results=wf_results,
        final_metrics=final_metrics,
        elapsed_seconds=round(elapsed, 2),
        trained_at=datetime.now().isoformat(),
        model_path=str(MODEL_FILE),
    )


def _train_xgb(X: np.ndarray, y: np.ndarray) -> Any:
    """Train an XGBoost classifier with tuned hyperparameters."""
    import xgboost as xgb

    # Class balance weight
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X, y)
    return model


# ── Model persistence ─────────────────────────────────────────────────


def _save_model(
    model: Any,
    trade_mode: str,
    metrics: ModelMetrics,
    n_tickers: int,
    n_samples: int,
) -> None:
    """Save the trained model and metadata to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    # Save metadata
    meta = {
        "trade_mode": trade_mode,
        "trained_at": datetime.now().isoformat(),
        "n_tickers": n_tickers,
        "n_samples": n_samples,
        "forward_bars": FORWARD_BARS.get(trade_mode, 10),
        "feature_names": FEATURE_NAMES,
        "metrics": {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "auc_roc": metrics.auc_roc,
            "n_train": metrics.n_train,
            "n_test": metrics.n_test,
        },
        "feature_importances": metrics.feature_importances,
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("ML model saved to %s", MODEL_FILE)


def load_model() -> Any | None:
    """Load the trained model from disk. Returns None if not found."""
    if not MODEL_FILE.exists():
        return None
    try:
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)  # noqa: S301
    except Exception:
        logger.warning("Failed to load ML model", exc_info=True)
        return None


def load_model_meta() -> dict[str, Any] | None:
    """Load model metadata. Returns None if not found."""
    if not META_FILE.exists():
        return None
    try:
        with open(META_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def model_exists() -> bool:
    """Check if a trained model exists on disk."""
    return MODEL_FILE.exists() and META_FILE.exists()


# ── Prediction ────────────────────────────────────────────────────────


def predict_signal(
    indicator_results: list[Any],
    pattern_results: list[Any],
    composite: dict[str, Any],
    pattern_composite: dict[str, Any],
    regime: Any | None,
    df: pd.DataFrame,
) -> PredictionResult | None:
    """Score a single signal using the trained XGBoost model.

    Returns None if no model is available.

    Args:
        indicator_results: From IndicatorRegistry.run_all()
        pattern_results: From PatternRegistry.run_all()
        composite: From CompositeScorer.score()
        pattern_composite: From PatternCompositeScorer.score()
        regime: RegimeAssessment or None
        df: OHLCV DataFrame

    Returns:
        PredictionResult with probability, AI rating, and top features.
    """
    model = load_model()
    if model is None:
        return None

    # Extract features
    features = extract_features(
        indicator_results, pattern_results,
        composite, pattern_composite,
        regime, df,
    )

    # Predict
    X = features.reshape(1, -1)
    prob = float(model.predict_proba(X)[0, 1])  # probability of win
    ai_rating = round(prob * 100, 1)             # 0-100 scale

    # Label
    if ai_rating >= 65:
        label = "Bullish"
    elif ai_rating <= 35:
        label = "Bearish"
    else:
        label = "Neutral"

    # Confidence
    distance_from_50 = abs(ai_rating - 50)
    if distance_from_50 >= 25:
        confidence = "High"
    elif distance_from_50 >= 10:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Top contributing features
    importances = model.feature_importances_
    top_features = _get_top_features(features, importances, top_n=5)

    return PredictionResult(
        probability=round(prob, 4),
        ai_rating=ai_rating,
        label=label,
        confidence=confidence,
        top_features=top_features,
    )


def _get_top_features(
    features: np.ndarray,
    importances: np.ndarray,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Get the top N most important features for this prediction."""
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    result = []
    for idx in indices:
        result.append({
            "name": FEATURE_NAMES[idx],
            "value": round(float(features[idx]), 4),
            "importance": round(float(importances[idx]), 4),
        })
    return result
