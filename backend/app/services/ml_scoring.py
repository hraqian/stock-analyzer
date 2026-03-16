"""Backend service for ML signal scoring — wraps engine/ml_model.py.

Provides async-friendly wrappers for training, prediction, and model
status. Training is CPU-intensive and runs in a thread pool to avoid
blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def get_model_status() -> dict[str, Any]:
    """Return the current ML model status and metadata.

    Returns a dict with:
      - trained: bool
      - trade_mode, trained_at, n_tickers, n_samples, forward_bars
      - metrics: {accuracy, precision, recall, f1, auc_roc}
      - feature_importances: {name: importance} (top 10)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_model_status_sync)


def _get_model_status_sync() -> dict[str, Any]:
    from engine.ml_model import model_exists, load_model_meta  # type: ignore[import-untyped]

    if not model_exists():
        return {"trained": False}

    meta = load_model_meta()
    if meta is None:
        return {"trained": False}

    # Top 10 feature importances
    all_imp = meta.get("feature_importances", {})
    top10 = dict(list(all_imp.items())[:10])

    return {
        "trained": True,
        "trade_mode": meta.get("trade_mode", "unknown"),
        "trained_at": meta.get("trained_at", ""),
        "n_tickers": meta.get("n_tickers", 0),
        "n_samples": meta.get("n_samples", 0),
        "forward_bars": meta.get("forward_bars", 0),
        "metrics": meta.get("metrics", {}),
        "feature_importances": top10,
    }


async def train_ml_model(
    universe: str,
    trade_mode: str = "swing",
    period: str = "5y",
) -> dict[str, Any]:
    """Train the ML model on a universe of tickers.

    This is a long-running operation (minutes for large universes).
    It runs in a thread pool to avoid blocking the event loop.

    Args:
        universe: Universe name (e.g. "sp500", "nasdaq100").
        trade_mode: "swing" or "long_term".
        period: Historical data period (default "5y").

    Returns:
        Dict with training results and metrics.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _train_ml_model_sync, universe, trade_mode, period,
    )


def _train_ml_model_sync(
    universe: str,
    trade_mode: str,
    period: str,
) -> dict[str, Any]:
    from engine.ml_model import (  # type: ignore[import-untyped]
        generate_training_data,
        train_model,
    )

    # Load universe tickers
    tickers = _load_universe_tickers(universe)
    if not tickers:
        raise ValueError(f"Universe '{universe}' not found or empty.")

    logger.info(
        "ML train: universe=%s (%d tickers), mode=%s, period=%s",
        universe, len(tickers), trade_mode, period,
    )

    # Step 1: Generate training data
    logger.info("ML train: generating training data...")
    samples = generate_training_data(
        tickers=tickers,
        trade_mode=trade_mode,
        period=period,
    )

    if not samples:
        raise ValueError(
            "No training samples generated. The universe may have "
            "insufficient data or all tickers failed to download."
        )

    logger.info("ML train: %d samples generated, starting walk-forward training...", len(samples))

    # Step 2: Train with walk-forward validation
    result = train_model(samples, trade_mode=trade_mode)

    logger.info(
        "ML train: complete! %d samples, %d tickers, accuracy=%.3f, elapsed=%.1fs",
        result.total_samples, result.total_tickers,
        result.final_metrics.accuracy, result.elapsed_seconds,
    )

    # Convert to serialisable dict
    wf_dicts = []
    for wf in result.walk_forward_results:
        wf_dicts.append({
            "window_idx": wf.window_idx,
            "train_start": wf.train_start,
            "train_end": wf.train_end,
            "test_start": wf.test_start,
            "test_end": wf.test_end,
            "metrics": {
                "accuracy": wf.metrics.accuracy,
                "precision": wf.metrics.precision,
                "recall": wf.metrics.recall,
                "f1": wf.metrics.f1,
                "auc_roc": wf.metrics.auc_roc,
                "n_train": wf.metrics.n_train,
                "n_test": wf.metrics.n_test,
            },
        })

    return {
        "total_samples": result.total_samples,
        "total_tickers": result.total_tickers,
        "trade_mode": result.trade_mode,
        "forward_bars": result.forward_bars,
        "walk_forward_results": wf_dicts,
        "final_metrics": {
            "accuracy": result.final_metrics.accuracy,
            "precision": result.final_metrics.precision,
            "recall": result.final_metrics.recall,
            "f1": result.final_metrics.f1,
            "auc_roc": result.final_metrics.auc_roc,
            "n_train": result.final_metrics.n_train,
            "n_test": result.final_metrics.n_test,
            "feature_importances": dict(
                list(result.final_metrics.feature_importances.items())[:15]
            ),
        },
        "elapsed_seconds": result.elapsed_seconds,
        "trained_at": result.trained_at,
    }


async def predict_signal_score(
    indicator_results: list[Any],
    pattern_results: list[Any],
    composite: dict[str, Any],
    pattern_composite: dict[str, Any],
    regime: Any | None,
    df: Any,
) -> dict[str, Any] | None:
    """Score a signal using the trained ML model.

    Returns None if no model is available.
    Returns a dict with probability, ai_rating, label, confidence,
    and top_features.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        _predict_sync,
        indicator_results, pattern_results,
        composite, pattern_composite, regime, df,
    )


def _predict_sync(
    indicator_results: list[Any],
    pattern_results: list[Any],
    composite: dict[str, Any],
    pattern_composite: dict[str, Any],
    regime: Any | None,
    df: Any,
) -> dict[str, Any] | None:
    from engine.ml_model import predict_signal  # type: ignore[import-untyped]

    result = predict_signal(
        indicator_results, pattern_results,
        composite, pattern_composite, regime, df,
    )
    if result is None:
        return None

    return {
        "probability": result.probability,
        "ai_rating": result.ai_rating,
        "label": result.label,
        "confidence": result.confidence,
        "top_features": result.top_features,
    }


def _load_universe_tickers(universe: str) -> list[str]:
    """Load ticker symbols from a universe file.

    Delegates to the canonical ``data.universes.load()`` function which
    is the same loader the scanner uses.  This ensures consistent path
    resolution regardless of whether we run locally or inside Docker.
    """
    from data.universes import load as load_universe  # type: ignore[import-untyped]

    try:
        return load_universe(universe)
    except FileNotFoundError:
        logger.warning("Universe '%s' not found", universe)
        return []
