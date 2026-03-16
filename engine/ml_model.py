"""
engine/ml_model.py — XGBoost signal scoring model: training, evaluation, prediction.

This module implements the full ML pipeline for scoring trading signals:
  1. Training data generation — runs analysis at historical points, labels
     with forward returns using a 3-class target (win / neutral / loss).
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

# ── 3-class labeling thresholds ───────────────────────────────────────
# Class 2: strong win  — forward return > WIN_ATR_MULT * ATR%
# Class 1: neutral     — between -LOSS_ATR_MULT * ATR% and +WIN_ATR_MULT * ATR%
# Class 0: strong loss — forward return < -LOSS_ATR_MULT * ATR%
DYNAMIC_WIN_THRESHOLD = True    # use ATR-based thresholds
WIN_ATR_MULTIPLIER = 1.0        # win requires return > 1.0 * ATR%
LOSS_ATR_MULTIPLIER = 0.5       # loss requires return < -0.5 * ATR%
WIN_THRESHOLD_PCT = 1.0         # fallback if ATR unavailable
LOSS_THRESHOLD_PCT = -0.5       # fallback if ATR unavailable

# Walk-forward configuration
WF_TRAIN_RATIO = 0.75     # 75% train, 25% test in each window
WF_NUM_WINDOWS = 4        # number of walk-forward windows
WF_MIN_SAMPLES = 200      # minimum training samples per window

# Sampling configuration
SAMPLE_INTERVAL_BARS = 20  # every 20 bars ≈ monthly for daily data

# Signal-based sampling: only take samples where the rule-based engine
# produces a non-trivial signal.
SIGNAL_SAMPLING = True
SIGNAL_SCORE_MIN_DEVIATION = 0.5  # |composite - 5.0| must exceed this

# Number of output classes
NUM_CLASSES = 3  # 0=loss, 1=neutral, 2=win


# ── Data classes ──────────────────────────────────────────────────────


@dataclass
class TrainingSample:
    """A single labeled training sample."""
    ticker: str
    date: str
    features: np.ndarray       # shape (NUM_FEATURES,)
    label: int                 # 0=loss, 1=neutral, 2=win
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
    class_distribution: dict[str, int] = field(default_factory=dict)
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


# ── Cross-asset data fetching ─────────────────────────────────────────


def _fetch_cross_asset_data(period: str = "5y") -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Fetch SPY and VIX data for cross-asset features.

    Returns (spy_df, vix_df) — either can be None if fetch fails.
    """
    try:
        from data.yahoo import YahooFinanceProvider  # type: ignore[import-untyped]
        provider = YahooFinanceProvider()

        spy_df = None
        vix_df = None

        try:
            spy_df = provider.fetch("SPY", period=period)
        except Exception:
            logger.debug("Failed to fetch SPY for cross-asset features")

        try:
            vix_df = provider.fetch("^VIX", period=period)
        except Exception:
            logger.debug("Failed to fetch VIX for cross-asset features")

        return spy_df, vix_df
    except Exception:
        return None, None


def _align_cross_asset(
    stock_df: pd.DataFrame,
    cross_df: pd.DataFrame | None,
    bar_idx: int,
) -> pd.DataFrame | None:
    """Slice cross-asset data to align with the stock slice ending at bar_idx.

    Returns a DataFrame sliced to the same date range as stock_df[:bar_idx+1],
    or None if alignment fails.
    """
    if cross_df is None or len(cross_df) == 0:
        return None

    try:
        target_date = stock_df.index[bar_idx]
        # Find the closest date in cross_df that is <= target_date
        mask = cross_df.index <= target_date
        if not mask.any():
            return None
        aligned = cross_df.loc[mask]
        # Return last 200+ bars to have enough history
        return aligned
    except Exception:
        return None


# ── Training data generation ──────────────────────────────────────────


def generate_training_data(
    tickers: list[str],
    trade_mode: str = "swing",
    period: str = "5y",
    progress_callback: Any | None = None,
    sample_interval: int | None = None,
) -> list[TrainingSample]:
    """Generate labeled training samples from historical data.

    For each ticker, we:
      1. Fetch 5y of daily data.
      2. At regular intervals (every sample_interval bars), slice
         the data up to that point, run the full indicator/pattern/regime
         pipeline, and extract features.
      3. Look at the forward return over the next N bars to assign a
         3-class label: 0=loss, 1=neutral, 2=win.

    Args:
        tickers: List of ticker symbols.
        trade_mode: "swing" or "long_term" — determines forward horizon.
        period: How much history to fetch per ticker.
        progress_callback: Optional callable(ticker, i, total) for progress.
        sample_interval: Bars between samples (default: SAMPLE_INTERVAL_BARS).

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

    # ── Fetch cross-asset data ────────────────────────────────────────
    logger.info("ML training: fetching cross-asset data (SPY, VIX)...")
    spy_df, vix_df = _fetch_cross_asset_data(period=period)
    if spy_df is not None:
        logger.info("ML training: SPY data loaded (%d bars)", len(spy_df))
    if vix_df is not None:
        logger.info("ML training: VIX data loaded (%d bars)", len(vix_df))

    # ── Batch fetch ──────────────────────────────────────────────────
    logger.info(
        "ML training: fetching %d tickers (period=%s, forward=%d bars)",
        len(tickers), period, forward_bars,
    )
    actual_interval = sample_interval if sample_interval is not None else SAMPLE_INTERVAL_BARS

    all_data = fetch_batch(tickers, period=period)
    logger.info(
        "ML training: got data for %d / %d tickers, sample_interval=%d",
        len(all_data), len(tickers), actual_interval,
    )

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
                sample_interval=actual_interval,
                spy_df=spy_df, vix_df=vix_df,
            )
            samples.extend(ticker_samples)
        except Exception:
            logger.debug("ML training: error on %s", ticker, exc_info=True)
            continue

    # Log class distribution
    labels = [s.label for s in samples]
    dist = {0: labels.count(0), 1: labels.count(1), 2: labels.count(2)}
    logger.info(
        "ML training: generated %d samples from %d tickers — "
        "class distribution: loss=%d, neutral=%d, win=%d",
        len(samples), total, dist[0], dist[1], dist[2],
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
    sample_interval: int = 20,
    spy_df: pd.DataFrame | None = None,
    vix_df: pd.DataFrame | None = None,
) -> list[TrainingSample]:
    """Generate samples from a single ticker's historical data.

    We walk through the data, stopping every sample_interval bars.
    At each stop, we slice df[:bar_idx], run the full analysis, extract
    features (including cross-asset context), and compute the forward
    return to assign a 3-class label.
    """
    samples: list[TrainingSample] = []
    n = len(df)

    # Start after 200 bars (need enough history for indicators)
    # Stop forward_bars before the end (need forward return)
    start_bar = 200
    end_bar = n - forward_bars

    for bar_idx in range(start_bar, end_bar, sample_interval):
        try:
            # Slice data up to this point (no lookahead)
            df_slice = df.iloc[:bar_idx + 1].copy()

            # Run indicators on the slice
            indicator_results = ind_registry.run_all(df_slice)
            pattern_results = pat_registry.run_all(df_slice)
            composite = scorer.score(indicator_results)
            pattern_composite = pat_scorer.score(pattern_results)

            # Signal-based sampling filter: skip neutral signals
            if SIGNAL_SAMPLING:
                overall_score = composite.get("overall", 5.0)
                if abs(overall_score - 5.0) < SIGNAL_SCORE_MIN_DEVIATION:
                    continue  # too close to neutral — skip this bar

            # Regime
            regime = None
            try:
                regime = regime_clf.classify(df_slice)
            except Exception:
                pass

            # Align cross-asset data to this bar's date
            spy_slice = _align_cross_asset(df, spy_df, bar_idx)
            vix_slice = _align_cross_asset(df, vix_df, bar_idx)

            # Extract features (now includes temporal + cross-asset)
            features = extract_features(
                indicator_results, pattern_results,
                composite, pattern_composite,
                regime, df_slice,
                spy_df=spy_slice, vix_df=vix_slice,
            )

            # Compute forward return
            current_price = float(df["close"].iloc[bar_idx])
            future_price = float(df["close"].iloc[bar_idx + forward_bars])
            forward_return_pct = (
                (future_price - current_price) / current_price * 100
            )

            # 3-class labeling
            label = _compute_label(df, bar_idx, forward_return_pct)

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


def _compute_label(
    df: pd.DataFrame,
    bar_idx: int,
    forward_return_pct: float,
) -> int:
    """Compute 3-class label based on forward return vs ATR threshold.

    Returns:
        2 = strong win (return > +WIN_ATR_MULT * ATR%)
        1 = neutral (between thresholds)
        0 = strong loss (return < -LOSS_ATR_MULT * ATR%)
    """
    win_thresh = WIN_THRESHOLD_PCT
    loss_thresh = LOSS_THRESHOLD_PCT

    if DYNAMIC_WIN_THRESHOLD:
        atr_threshold = _compute_atr_threshold(df, bar_idx)
        if atr_threshold is not None:
            win_thresh = atr_threshold * WIN_ATR_MULTIPLIER
            loss_thresh = -atr_threshold * LOSS_ATR_MULTIPLIER

    if forward_return_pct > win_thresh:
        return 2  # win
    elif forward_return_pct < loss_thresh:
        return 0  # loss
    else:
        return 1  # neutral


def _compute_atr_threshold(df: pd.DataFrame, bar_idx: int) -> float | None:
    """Compute ATR as percentage of price at a specific bar.

    Returns the ATR percentage, or None if it cannot be computed.
    Uses a 14-period ATR ending at bar_idx.
    """
    if bar_idx < 14:
        return None

    try:
        window = df.iloc[max(0, bar_idx - 14):bar_idx + 1]
        high = window["high"].values
        low = window["low"].values
        close = window["close"].values

        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        atr = float(np.mean(tr))
        price = float(close[-1])

        if price <= 0:
            return None

        # Return ATR as % of price
        return (atr / price) * 100

    except Exception:
        return None


# ── Walk-forward training ─────────────────────────────────────────────


def train_model(
    samples: list[TrainingSample],
    trade_mode: str = "swing",
    progress_callback: Any | None = None,
) -> TrainingResult:
    """Train an XGBoost model with walk-forward validation.

    Uses 3-class classification: 0=loss, 1=neutral, 2=win.

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
    from sklearn.preprocessing import StandardScaler

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
    X_raw = np.array([s.features for s in sorted_samples], dtype=np.float32)
    y = np.array([s.label for s in sorted_samples], dtype=np.int32)
    dates = [s.date for s in sorted_samples]
    tickers_used = list(set(s.ticker for s in sorted_samples))

    # Class distribution
    class_dist = {
        "loss": int((y == 0).sum()),
        "neutral": int((y == 1).sum()),
        "win": int((y == 2).sum()),
    }

    # ── Feature normalization ─────────────────────────────────────────
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)

    logger.info(
        "ML train: %d samples, %d features, class dist: loss=%d neutral=%d win=%d",
        n, X.shape[1], class_dist["loss"], class_dist["neutral"], class_dist["win"],
    )

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

        train_end = test_start

        if train_end < WF_MIN_SAMPLES:
            continue

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]

        if len(X_test) == 0:
            continue

        # Train XGBoost (no tuning for walk-forward windows — speed)
        model = _train_xgb(X_train, y_train, tune_hyperparams=False)

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        metrics = _compute_metrics(y_test, y_pred, y_prob, len(X_train), len(X_test))

        wf_results.append(WalkForwardResult(
            window_idx=w,
            train_start=dates[0],
            train_end=dates[train_end - 1],
            test_start=dates[test_start],
            test_end=dates[test_end - 1],
            metrics=metrics,
        ))

    # ── Final model: train on ALL data with hyperparameter tuning ─────
    final_model = _train_xgb(X, y, tune_hyperparams=True)

    # Feature importances from final model
    importances = final_model.feature_importances_
    feat_imp = {
        FEATURE_NAMES[i]: round(float(importances[i]), 6)
        for i in range(NUM_FEATURES)
    }
    feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

    # Final metrics (last walk-forward window, or overall if only one)
    if wf_results:
        final_metrics = wf_results[-1].metrics
    else:
        # Fallback: evaluate on last 20% of data
        split = int(n * 0.8)
        y_pred = final_model.predict(X[split:])
        y_prob = final_model.predict_proba(X[split:])
        y_actual = y[split:]
        final_metrics = _compute_metrics(y_actual, y_pred, y_prob, split, n - split)

    final_metrics.feature_importances = feat_imp
    final_metrics.class_distribution = class_dist

    # ── Save model + scaler ───────────────────────────────────────────
    _save_model(final_model, trade_mode, final_metrics, len(tickers_used), n, scaler)

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


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_train: int,
    n_test: int,
) -> ModelMetrics:
    """Compute evaluation metrics for 3-class classification."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score,
    )

    acc = round(float(accuracy_score(y_true, y_pred)), 4)
    prec = round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4)
    rec = round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4)
    f1 = round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4)

    # AUC-ROC for multi-class (one-vs-rest)
    auc = 0.0
    try:
        if len(set(y_true)) > 1 and y_prob.shape[1] == NUM_CLASSES:
            auc = round(float(roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="weighted",
            )), 4)
    except Exception:
        pass

    return ModelMetrics(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        auc_roc=auc,
        n_train=n_train,
        n_test=n_test,
    )


def _train_xgb(
    X: np.ndarray,
    y: np.ndarray,
    tune_hyperparams: bool = False,
) -> Any:
    """Train an XGBoost classifier with optional hyperparameter tuning.

    Uses multi-class softmax for 3-class classification.
    """
    import xgboost as xgb

    if tune_hyperparams and len(X) >= 500:
        return _train_xgb_tuned(X, y)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X, y)
    return model


def _train_xgb_tuned(
    X: np.ndarray,
    y: np.ndarray,
) -> Any:
    """Train XGBoost with randomized hyperparameter search (3-class)."""
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

    base = xgb.XGBClassifier(
        subsample=0.8,
        colsample_bytree=0.7,
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    param_distributions = {
        "n_estimators": [200, 300, 400],
        "max_depth": [4, 5, 6],
        "learning_rate": [0.02, 0.03, 0.05],
        "min_child_weight": [3, 5, 7],
        "gamma": [0.0, 0.1, 0.2],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        base, param_distributions,
        n_iter=12,
        cv=cv,
        scoring="accuracy",   # multi-class: use accuracy instead of roc_auc
        n_jobs=1,
        refit=True,
        random_state=42,
        verbose=0,
    )
    search.fit(X, y)

    logger.info(
        "XGBoost tuning: best accuracy=%.4f, params=%s",
        search.best_score_, search.best_params_,
    )

    return search.best_estimator_


# ── Model persistence ─────────────────────────────────────────────────


SCALER_FILE = MODEL_DIR / "xgb_scaler.pkl"


def _save_model(
    model: Any,
    trade_mode: str,
    metrics: ModelMetrics,
    n_tickers: int,
    n_samples: int,
    scaler: Any = None,
) -> None:
    """Save the trained model, scaler, and metadata to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    # Save scaler (for feature normalization at prediction time)
    if scaler is not None:
        with open(SCALER_FILE, "wb") as f:
            pickle.dump(scaler, f)

    # Save metadata
    meta = {
        "trade_mode": trade_mode,
        "trained_at": datetime.now().isoformat(),
        "n_tickers": n_tickers,
        "n_samples": n_samples,
        "forward_bars": FORWARD_BARS.get(trade_mode, 10),
        "num_classes": NUM_CLASSES,
        "feature_names": FEATURE_NAMES,
        "metrics": {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "auc_roc": metrics.auc_roc,
            "n_train": metrics.n_train,
            "n_test": metrics.n_test,
            "class_distribution": metrics.class_distribution,
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


def load_scaler() -> Any | None:
    """Load the feature scaler from disk. Returns None if not found."""
    if not SCALER_FILE.exists():
        return None
    try:
        with open(SCALER_FILE, "rb") as f:
            return pickle.load(f)  # noqa: S301
    except Exception:
        logger.warning("Failed to load ML scaler", exc_info=True)
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


def model_fully_exists() -> bool:
    """Check if model, metadata, AND scaler all exist."""
    return MODEL_FILE.exists() and META_FILE.exists() and SCALER_FILE.exists()


# ── Cached cross-asset data for predictions ──────────────────────────

# Simple module-level cache so we don't re-fetch SPY/VIX on every
# predict_signal() call.  Expires after 1 hour.
_cross_asset_cache: dict[str, Any] = {
    "spy_df": None,
    "vix_df": None,
    "fetched_at": 0.0,
}
_CACHE_TTL = 3600  # 1 hour


def _get_cross_asset_cached() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Return cached SPY/VIX data, refreshing if stale."""
    now = time.time()
    if now - _cross_asset_cache["fetched_at"] < _CACHE_TTL:
        return _cross_asset_cache["spy_df"], _cross_asset_cache["vix_df"]

    spy_df, vix_df = _fetch_cross_asset_data(period="1y")
    _cross_asset_cache["spy_df"] = spy_df
    _cross_asset_cache["vix_df"] = vix_df
    _cross_asset_cache["fetched_at"] = now
    return spy_df, vix_df


# ── Prediction ────────────────────────────────────────────────────────


def predict_signal(
    indicator_results: list[Any],
    pattern_results: list[Any],
    composite: dict[str, Any],
    pattern_composite: dict[str, Any],
    regime: Any | None,
    df: pd.DataFrame,
    *,
    spy_df: pd.DataFrame | None = None,
    vix_df: pd.DataFrame | None = None,
) -> PredictionResult | None:
    """Score a single signal using the trained XGBoost model.

    Returns None if no model is available.

    If spy_df/vix_df are not provided, they are auto-fetched and cached
    (1 hour TTL) so callers don't need to worry about cross-asset data.

    For 3-class models:
      - ai_rating = P(win) * 100 — the probability of the "win" class.
    """
    model = load_model()
    if model is None:
        return None

    # Auto-fetch cross-asset data if not provided
    if spy_df is None or vix_df is None:
        try:
            cached_spy, cached_vix = _get_cross_asset_cached()
            if spy_df is None:
                spy_df = cached_spy
            if vix_df is None:
                vix_df = cached_vix
        except Exception:
            pass  # proceed without cross-asset features

    # Extract features (with cross-asset context)
    features = extract_features(
        indicator_results, pattern_results,
        composite, pattern_composite,
        regime, df,
        spy_df=spy_df, vix_df=vix_df,
    )

    # Apply the same scaling used during training
    X = features.reshape(1, -1)
    scaler = load_scaler()
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            pass  # shape mismatch from old scaler — use unscaled

    # Predict
    probs = model.predict_proba(X)[0]

    # Handle both 2-class (legacy) and 3-class models
    if len(probs) == NUM_CLASSES:
        p_loss = float(probs[0])
        p_neutral = float(probs[1])
        p_win = float(probs[2])
    else:
        # Legacy 2-class model
        p_loss = float(probs[0])
        p_neutral = 0.0
        p_win = float(probs[1]) if len(probs) > 1 else 0.0

    ai_rating = round(p_win * 100, 1)  # 0-100 scale

    # Label based on highest probability class
    if p_win > p_loss and p_win > p_neutral:
        label = "Bullish"
    elif p_loss > p_win and p_loss > p_neutral:
        label = "Bearish"
    else:
        label = "Neutral"

    # Confidence based on margin between top two classes
    sorted_probs = sorted([p_loss, p_neutral, p_win], reverse=True)
    margin = sorted_probs[0] - sorted_probs[1]
    if margin >= 0.25:
        confidence = "High"
    elif margin >= 0.10:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Top contributing features
    importances = model.feature_importances_
    top_features = _get_top_features(features, importances, top_n=5)

    return PredictionResult(
        probability=round(p_win, 4),
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
    # Handle potential size mismatch (old model vs new features)
    n = min(len(features), len(importances))
    indices = np.argsort(importances[:n])[::-1][:top_n]
    result = []
    for idx in indices:
        name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feature_{idx}"
        result.append({
            "name": name,
            "value": round(float(features[idx]), 4),
            "importance": round(float(importances[idx]), 4),
        })
    return result
