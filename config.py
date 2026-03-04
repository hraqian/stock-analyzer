"""
config.py — Load, validate, and manage the YAML configuration file.

Provides a single Config object used throughout the application.
"""

from __future__ import annotations

import copy
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

# ---------------------------------------------------------------------------
# Default configuration (mirrors config.yaml — used when file is absent or
# a key is missing so we always have a fully-populated config dict).
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict[str, Any] = {
    "rsi": {
        "period": 14,
        "thresholds": {"oversold": 30, "overbought": 70},
        "scores": {"oversold_score": 9.0, "overbought_score": 1.0, "neutral_score": 5.0},
    },
    "macd": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
        "scoring": {
            "strong_bullish_pct": 0.005,
            "moderate_bullish_pct": 0.001,
            "strong_bearish_pct": -0.005,
            "moderate_bearish_pct": -0.001,
            "crossover_lookback": 5,
            "bullish_cross_bonus": 1.5,
            "bearish_cross_penalty": 1.5,
        },
    },
    "bollinger_bands": {
        "period": 20,
        "std_dev": 2.0,
        "scoring": {
            "lower_zone": 0.20,
            "upper_zone": 0.80,
            "squeeze_threshold": 0.02,
        },
    },
    "moving_averages": {
        "periods": [20, 50, 200],
        "type": "sma",
        "scoring": {
            "price_above_ma_points": 1.5,
            "ma_aligned_bullish_points": 1.0,
            "golden_cross_bonus": 2.0,
            "death_cross_penalty": 2.0,
            "cross_lookback": 10,
            "max_raw_score": 9.5,
        },
    },
    "stochastic": {
        "k_period": 14,
        "d_period": 3,
        "smooth_k": 3,
        "thresholds": {"oversold": 20, "overbought": 80},
        "scores": {
            "oversold_score": 9.0,
            "overbought_score": 1.0,
            "neutral_score": 5.0,
            "bullish_cross_bonus": 1.0,
            "bearish_cross_penalty": 1.0,
        },
    },
    "adx": {
        "period": 14,
        "thresholds": {"weak": 20, "moderate": 40},
        "scoring": {
            "weak_multiplier": 0.6,
            "moderate_multiplier": 0.85,
            "strong_multiplier": 1.0,
            "max_directional_spread": 25,
        },
    },
    "volume": {
        "obv_trend_period": 20,
        "price_trend_period": 20,
        "scoring": {
            "confirmation_bullish_max": 9.5,
            "confirmation_bullish_min": 6.5,
            "confirmation_bearish_max": 3.5,
            "confirmation_bearish_min": 0.5,
            "divergence_score": 5.0,
            "obv_strong_change_pct": 10.0,
            "obv_weak_change_pct": 1.0,
        },
    },
    "fibonacci": {
        "swing_lookback": 60,
        "levels": [0.236, 0.382, 0.5, 0.618, 0.786],
        "scoring": {
            "proximity_pct": 0.015,
            "level_scores": {0.236: 8.0, 0.382: 7.0, 0.5: 5.0, 0.618: 3.0, 0.786: 1.5},
            "no_level_score": 5.0,
            "range_low_score": 2.0,
            "range_high_score": 8.0,
        },
    },
    "support_resistance": {
        "method": "both",
        "pivot_levels": ["S1", "S2", "S3", "P", "R1", "R2", "R3"],
        "fractal_lookback": 60,
        "fractal_order": 5,
        "num_levels": 4,
        "cluster_pct": 0.015,
        "min_touches": 1,
    },
    "overall": {
        "weights": {
            "rsi": 0.15,
            "macd": 0.15,
            "bollinger_bands": 0.10,
            "moving_averages": 0.20,
            "stochastic": 0.10,
            "adx": 0.10,
            "volume": 0.10,
            "fibonacci": 0.10,
        },
        "subgroup_mode": "directional",  # "directional" or "average" (legacy)
        "indicator_groups": {
            "trend": ["moving_averages", "macd", "adx", "volume"],
            "contrarian": ["rsi", "stochastic", "bollinger_bands"],
            "neutral": ["fibonacci"],
        },
        "subgroup_blend": {
            "dominant_weight": 0.6,
            "other_weight": 0.25,
            "neutral_weight": 0.15,
        },
        "score_spreading": {
            "enabled": True,
            "factor": 2.0,
        },
    },
    "display": {
        "show_disclaimer": True,
        "score_decimal_places": 1,
        "price_decimal_places": 2,
        "color_thresholds": {"bearish_max": 3.5, "neutral_max": 6.0},
    },
    "dashboard": {
        "default_ticker": "SPY",
    },
    "strategy": {
        "threshold_mode": "fixed",
        "score_thresholds": {
            "short_below": 3.0,
            "hold_below": 6.5,
        },
        "percentile_thresholds": {
            "short_percentile": 25,
            "long_percentile": 75,
            "lookback_bars": 60,
            "percentile_step": 5,        # bar step for building percentile window (auto-clamped to ensure enough samples)
        },
        "position_sizing": "percent_equity",
        "fixed_quantity": 100,
        "percent_equity": 1.00,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.30,
        "rebalance_interval": 10,
        "max_hold_bars": 0,    # force-exit after N bars (0 = disabled); 20 bars ≈ 4 weeks on daily
        "flatten_eod": False,  # force-close all positions at end of each trading day
        # ATR-adaptive stop loss — uses max(fixed %, ATR-based %) so ATR widens the stop
        "atr_stop_enabled": True,       # use ATR-based stop instead of fixed %
        "atr_stop_multiplier": 5.0,     # stop = N × ATR at entry time
        "atr_stop_period": 14,          # ATR calculation period
        # Trend confirmation filter
        "trend_confirm_enabled": True,   # require price above/below trend MA for entry
        "trend_confirm_period": 30,      # EMA period for trend confirmation
        "trend_confirm_ma_type": "ema",  # "ema" or "sma" for trend confirmation MA
        "trend_confirm_tolerance_pct": 0.005,  # tolerance band around MA
        # Re-entry grace period: after exiting, skip trend confirmation for N bars
        "reentry_grace_bars": 10,        # bars after exit to allow unfiltered re-entry
        # Consecutive loss cooldown: after N consecutive losses, tighten entry requirements
        "cooldown_max_losses": 2,        # consecutive losses before cooldown activates
        "cooldown_distance_mult": 2.0,   # multiply min_distance by this during cooldown
        "cooldown_min_score": 4.5,       # minimum score required during cooldown
        "cooldown_reset_on_breakeven": True,  # whether 0% PnL resets consecutive loss counter
        # Global trend bias: suppress counter-trend entries when total return is strong
        "global_trend_bias": True,       # enable global directional bias
        "global_bias_threshold": 0.10,   # |total_return| above this → suppress counter-trend
        # Strong trend hold/entry: total return threshold for determining bullish/bearish bias
        "trend_bias_return_threshold": 0.15,  # |total_return| >= this → definitive bias
        # Extreme score exit: exit strong-trend positions when score is this far beyond thresholds
        "extreme_exit_score_offset": 1.5,  # e.g. 1.5 → exit long if score < short_below - 1.5
        # Breakout confirmation: minimum directional move ratio for a breakout candle
        "breakout_min_move_ratio": 0.5,   # |close-open|/range >= this for breakout candle
        # Position management
        "allow_pyramiding": False,        # whether to add to existing same-direction positions
        "allow_immediate_reversal": True, # close + reopen in opposite direction on signal flip
        # Take-profit in strong trend
        "disable_take_profit_in_strong_trend": True,  # let trailing stop handle exits instead
        # Trailing stop: require position to be in profit before trail activates
        "trailing_stop_require_profit": True,
        # Percentile mode tuning
        "percentile_min_fill_ratio": 0.8,  # min fraction of lookback window before percentile activates
        # Pattern-indicator combination for strategy decisions
        "combination_mode": "weighted",  # "weighted", "gate", or "boost"
        "indicator_weight": 0.7,         # weight of indicator composite in blended score
        "pattern_weight": 0.3,           # weight of pattern composite in blended score
        # Gate mode: only trade if both scores pass their respective thresholds
        "gate_indicator_min": 5.5,       # indicator score must exceed this for LONG
        "gate_indicator_max": 4.5,       # indicator score must be below this for SHORT
        "gate_pattern_min": 5.5,         # pattern score must exceed this for LONG
        "gate_pattern_max": 4.5,         # pattern score must be below this for SHORT
        # Boost mode: patterns amplify indicator score when active
        "boost_strength": 0.5,           # multiplier for pattern deviation from 5.0
        "boost_dead_zone": 0.3,          # pattern score within 5.0 ± this → no boost
        # Confidence distance thresholds (used in scanner + dashboard signal labeling)
        "confidence_thresholds": {
            "high": 1.5,         # score distance from threshold for HIGH confidence
            "medium": 0.5,       # score distance from threshold for MEDIUM confidence
            "hold_high": 1.0,    # distance inside hold zone for HIGH confidence
            "hold_medium": 0.3,  # distance inside hold zone for MEDIUM confidence
        },
    },
    "backtest": {
        "initial_cash": 100_000.0,
        "commission_per_trade": 10.0,
        "commission_pct": 0.005,            # percentage commission per leg (0.005 = 0.5%)
        "commission_mode": "max",           # "additive" (flat + pct) or "max" (whichever is greater)
        "slippage_pct": 0.001,
        "warmup_bars": 50,
        "max_warmup_ratio": 0.5,       # warmup can't exceed this fraction of total data
        "significant_pattern_min_strength": 0.5,
        "min_warmup_bars": 20,          # absolute floor for proportional warmup
        "min_post_warmup_bars": 10,     # minimum tradeable bars after warmup
        "trading_days_per_year": 252,   # for annualization
        "trading_day_minutes": 390,     # US market: 6.5 hours = 390 minutes
        "default_score": 5.0,           # neutral starting score before first rebalance
        "close_on_end_of_data": True,   # whether to force-close position at end of data
        # Strength thresholds per detector — empty dict = use built-in defaults
        "strength_thresholds": {},
        "gap_strength_cap": 2.0,        # cap on gap detector strength contribution
        "spike_z_divisor": 2.5,         # z-score divisor for spike strength normalization
        "spike_strength_cap": 2.0,      # cap on spike detector strength contribution
    },
    # ------------------------------------------------------------------
    # Pattern Signal Detectors
    # ------------------------------------------------------------------
    "gaps": {
        "lookback": 20,
        "min_gap_pct": 0.005,
        "volume_surge_mult": 1.5,
        "trend_period": 20,
        "consolidation_lookback": 20,        # BB width window & rolling high/low window
        "bb_percentile_lookback": 40,        # longer window for BB width percentile ranking
        "consolidation_bb_percentile": 50,   # BB width rank <= this = narrow (consolidation)
        "consolidation_min_bars": 5,         # K of last M bars must be narrow for breakaway
        "consolidation_max_return": 0.03,    # abs return over consolidation window must be <= this (weak-trend gate)
        "exhaustion_min_return": 0.10,       # total return over trend_period for exhaustion (10%)
        "exhaustion_min_distance_pct": 0.05, # price must be >= 5% from MA for exhaustion
        "exhaustion_min_trend_bars": 40,     # return over this many bars must also confirm direction
        "intraday_min_gap_pct": 0.01,        # min gap size auto-applied on intraday data (1%)
        "type_weights": {
            "common": 0.3,
            "runaway": 0.7,
            "breakaway": 1.0,
            "exhaustion": 0.5,
        },
        "max_signal_strength": 3.0,
        "gap_pct_scale": 100,           # multiplier to convert gap_pct into strength units
    },
    "volume_range": {
        "period": 20,
        "expansion_threshold": 1.5,
        "contraction_threshold": 0.6,
        "lookback": 10,
        "scoring": {
            "expansion_bull": 8.0,
            "expansion_bear": 2.0,
            "contraction": 5.0,
            "divergence": 5.0,
        },
        "expansion_bias_multiplier": 1.5,  # multiplier for expansion directional bias
    },
    "candlesticks": {
        "doji_threshold": 0.05,
        "shadow_ratio": 2.0,
        "harami_body_ratio": 0.5,
        "dragonfly_shadow_min": 0.6,      # lower shadow >= range * this for dragonfly doji
        "gravestone_shadow_min": 0.6,     # upper shadow >= range * this for gravestone doji
        "doji_tiny_shadow_max": 0.1,      # opposite shadow <= range * this for dragonfly/gravestone
        "marubozu_body_min": 0.90,        # body fills >= this fraction of range for marubozu
        "marubozu_shadow_max": 0.05,      # each shadow <= this fraction of range for marubozu
        "tweezer_tolerance": 0.002,       # highs/lows match within this fraction for tweezer
        "star_middle_body_max": 0.30,     # middle bar body <= this for morning/evening star
        "soldiers_body_min": 0.60,        # each bar body >= this for 3 soldiers/crows
        "soldiers_shadow_max": 0.30,      # shadow <= this for 3 soldiers/crows
        "lookback": 10,
        "trend_period": 10,
        "max_signal_strength": 3.0,
        "hammer_body_max": 0.35,          # max body ratio for hammer/shooting star detection
        "star_body_min": 0.5,             # min body ratio for morning/evening star center candle
        # Per-pattern strength values — empty dict = use built-in defaults.
        # Keys: dragonfly_doji, gravestone_doji, doji_directional, doji_neutral,
        #        hammer, shooting_star, marubozu, engulfing, harami, tweezer,
        #        star, soldiers_crows.  Each maps to {with_trend: float, against_trend: float}.
        "strength_values": {},
    },
    "spikes": {
        "period": 20,
        "spike_std": 2.5,
        "confirm_bars": 3,
        "confirm_pct": 0.5,
        "lookback": 20,
        "trap_weight": 0.7,
        "max_signal_strength": 3.0,
        "z_magnitude_cap": 2.0,         # cap on z-score magnitude for strength calculation
        "unconfirmed_weight": 0.3,      # weight multiplier for unconfirmed spikes
    },
    "inside_outside": {
        "lookback": 20,
        "trend_period": 10,
        "breakout_bars": 3,
        "outside_range_min": 1.2,
        "max_signal_strength": 3.0,
        # Per-pattern strength values — empty dict = use built-in defaults.
        # Keys: inside_breakout_with_trend, inside_breakout_against_trend,
        #        inside_pending, outside_reversal, outside_continuation.
        "strength_values": {},
    },
    # ------------------------------------------------------------------
    # Pattern composite scoring
    # ------------------------------------------------------------------
    "overall_patterns": {
        "weights": {
            "gaps": 0.20,
            "volume_range": 0.25,
            "candlesticks": 0.25,
            "spikes": 0.15,
            "inside_outside": 0.15,
        },
        "score_spreading": {
            "enabled": True,
            "factor": 2.0,
        },
    },
    # ------------------------------------------------------------------
    # Multi-timeframe analysis
    # ------------------------------------------------------------------
    "multi_timeframe": {
        "timeframes": ["1d", "1wk", "1mo"],
        "weights": {"1d": 0.5, "1wk": 0.3, "1mo": 0.2},
        "periods": {"1d": "2y", "1wk": "5y", "1mo": "max"},
        "agreement_multipliers": {
            "aligned": 1.0,
            "mixed": 0.7,
            "conflicting": 0.4,
        },
    },
    # ------------------------------------------------------------------
    # Market Regime Classification
    # ------------------------------------------------------------------
    "regime": {
        "trend_ma_period": 50,           # MA period for trend analysis
        # ADX thresholds
        "adx_strong_trend": 30.0,        # ADX above this → strong trend signal
        "adx_weak": 20.0,               # ADX below this → weak/no trend
        # Trend consistency (% of bars above MA)
        "trend_consistency_high": 70.0,  # above this → strong directional bias
        "trend_consistency_low": 40.0,   # below 100-this on bear side → strong bear bias
        # ATR% volatility thresholds
        "atr_pct_high": 0.03,           # ATR% above this → high volatility
        "atr_pct_low": 0.01,            # ATR% below this → low volatility
        "atr_period": 14,               # ATR calculation period
        # Bollinger Band squeeze detection
        "bb_period": 20,                # BB calculation period
        "bb_std_dev": 2.0,             # BB standard deviation multiplier
        "bb_squeeze_percentile": 20.0,  # BB width below this percentile → squeeze
        "bb_expansion_percentile": 80.0, # BB width above this → expansion
        # Direction changes
        "direction_change_high": 0.55,  # fraction of bars reversing → choppy
        "direction_change_period": 20,  # lookback for direction change calculation
        # Price-MA distance
        "price_ma_distance_extended": 0.10,  # price > 10% from MA → extended trend
        # Total return thresholds (primary trend signal)
        "total_return_strong": 0.30,         # |return| > 30% → definitively trending
        "total_return_moderate": 0.15,       # |return| > 15% → moderate trend signal
        # Classification guard
        "min_bars_for_classification": 20,   # minimum bars needed to classify
        # Trend direction thresholds (pct_above_ma → bullish/bearish)
        "trend_direction_bullish_threshold": 60,  # pct_above_ma > this → bullish
        "trend_direction_bearish_threshold": 40,  # pct_above_ma < this → bearish
        # Reason building
        "adx_dip_threshold": 3,              # rolling mean > current + this → "temporary dip"
        "runner_up_proximity_ratio": 0.7,    # runner-up score / winner score > this → mention
        # ── Sub-type classification (Volatility × Momentum 2×2 matrix) ──
        "sub_type": {
            "atr_pct_threshold": 0.035,      # ATR% >= this → high volatility
            "momentum_threshold": 0.20,      # |total_return| >= this → high momentum
        },
        # ── Regime scoring weights ──────────────────────────────────────
        # All magic numbers used in _score_regimes(), organized by regime.
        "scoring": {
            "strong_trend": {
                "return_strong_base": 3.0,
                "return_strong_cap": 0.70,
                "return_strong_scale": 3.0,
                "return_moderate_base": 1.0,
                "return_moderate_scale": 2.0,
                "adx_strong_base": 2.0,
                "adx_strong_divisor": 20.0,
                "adx_moderate_score": 0.5,
                "consistency_high_score": 2.0,
                "consistency_moderate_score": 0.5,
                "extended_distance_score": 1.0,
                "direction_change_low": 0.4,
                "direction_change_low_score": 1.0,
                "direction_change_mid": 0.5,
                "direction_change_mid_score": 0.3,
            },
            "mean_reverting": {
                "return_strong_penalty": 2.0,
                "return_moderate_penalty": 1.0,
                "return_small_bonus": 1.5,
                "adx_low_score": 2.0,
                "adx_moderate_score": 1.0,
                "pct_away_tight": 15,
                "pct_away_tight_score": 2.0,
                "pct_away_moderate": 25,
                "pct_away_moderate_score": 1.0,
                "atr_below_high_score": 0.5,
                "atr_below_low_score": 0.5,
                "price_ma_close_threshold": 0.03,
                "price_ma_close_score": 1.0,
            },
            "volatile_choppy": {
                "atr_high_base": 2.0,
                "atr_high_scale": 30.0,
                "atr_moderate_score": 0.5,
                "direction_change_high_score": 2.0,
                "direction_change_moderate": 0.45,
                "direction_change_moderate_score": 1.0,
                "low_adx_small_return_score": 0.5,
                "wide_bb_score": 1.0,
            },
            "breakout": {
                "bb_squeeze_score": 2.5,
                "adx_moderate_score": 1.0,
                "direction_change_threshold": 0.40,
                "low_atr_high_changes_score": 1.0,
                "price_ma_close_threshold": 0.03,
                "bb_consolidation_percentile": 40,
                "consolidation_score": 0.5,
            },
        },
        # Strategy adaptation per regime
        "strategy_adaptation": {
            "strong_trend": {
                "use_trailing_stop": True,       # trail stop instead of score-based exits
                "trailing_stop_atr_mult": 8.0,   # trailing stop = N × ATR (wide for volatile names)
                "ignore_score_entries": True,     # don't use score thresholds for entry
                "hold_with_trend": True,          # stay in position while trend persists
                "min_distance": 0.015,           # price must be 1.5%+ from MA for trend entry
                "min_score": 3.5,                # don't enter when indicators are strongly bearish
                "respect_trend_direction": True,  # only enter in direction of long-term trend
                "pattern_veto_threshold": 3.5,   # skip entry if pattern score below this (0=disabled)
                # Sub-type overrides — sub-type is classified from warmup data at
                # backtest start and re-evaluated at every rebalance, so it can
                # change mid-run as more data arrives.
                # Low-vol names (SC) get tighter trail; high-vol (EM) use default.
                "sub_types": {
                    "explosive_mover": {},
                    "steady_compounder": {
                        "trailing_stop_atr_mult": 6.0,  # tighter trail for low-vol names
                    },
                    "volatile_directionless": {},
                    "stagnant": {},
                },
            },
            "mean_reverting": {
                "use_trailing_stop": False,       # use score-based entries/exits
                "tighten_thresholds": True,       # narrow the HOLD zone for more trades
                "threshold_adjustment": 0.2,      # narrow by this amount on each side
            },
            "volatile_choppy": {
                "reduce_position_size": True,     # halve position size
                "position_size_mult": 0.5,        # multiplier for position sizing
                "widen_stops": True,              # widen stop loss
                "stop_loss_mult": 1.5,            # multiply stop loss % by this
            },
            "breakout_transition": {
                "use_momentum_entry": True,       # enter on breakout confirmation
                "breakout_atr_mult": 1.5,         # price must move N×ATR from squeeze level
                "min_bar_range_pct": 0.020,       # bar range must exceed 2% of price for expansion
                "require_volume_surge": True,      # require above-avg volume for entry
                "volume_surge_mult": 1.5,         # volume must be N× the average
                "avg_volume_window": 20,          # rolling window for average volume baseline
            },
        },
    },
    "suitability": {
        "mode_override": "auto",
        "min_volume": 100_000,
        "min_atr_pct": 0.005,
        "min_adx_for_short": 25.0,
        "min_atr_for_short": 0.01,
        "min_volume_for_short": 500_000,
        "atr_period": 14,
        "trend_ma_period": 200,
        "max_pct_above_ma": 65.0,
        "adx_min_data_mult": 3,              # require period*N bars for ADX calculation
        "insufficient_data_pct": 50.0,       # pct_above_ma fallback when data is insufficient
    },
    # ------------------------------------------------------------------
    # Trading objective presets — partial overrides applied on top of the
    # base config above.  Select via --objective <name> or leave unset
    # to use the base config as-is.
    # ------------------------------------------------------------------
    "objectives": {
        "long_term": {
            "description": "Position trading — weeks to months. Slower indicators, wider stops, higher warmup.",
            "rsi": {
                "period": 21,
                "thresholds": {"oversold": 25, "overbought": 75},
            },
            "macd": {
                "fast_period": 19,
                "slow_period": 39,
                "signal_period": 9,
            },
            "bollinger_bands": {
                "period": 30,
            },
            "moving_averages": {
                "periods": [50, 100, 200],
            },
            "stochastic": {
                "k_period": 21,
                "d_period": 5,
                "smooth_k": 5,
                "thresholds": {"oversold": 20, "overbought": 80},
            },
            "adx": {
                "period": 21,
            },
            "volume": {
                "obv_trend_period": 30,
                "price_trend_period": 30,
            },
            "fibonacci": {
                "swing_lookback": 120,
            },
            "support_resistance": {
                "fractal_lookback": 120,
                "fractal_order": 8,
            },
            "overall": {
                "weights": {
                    "rsi": 0.10,
                    "macd": 0.10,
                    "bollinger_bands": 0.10,
                    "moving_averages": 0.25,
                    "stochastic": 0.05,
                    "adx": 0.15,
                    "volume": 0.15,
                    "fibonacci": 0.10,
                },
            },
            "strategy": {
                "rebalance_interval": 5,
                "stop_loss_pct": 0.10,
                "take_profit_pct": 1.00,
                "indicator_weight": 0.8,
                "pattern_weight": 0.2,
                "atr_stop_multiplier": 4.0,    # wider ATR stop for position trading
                "trend_confirm_period": 50,     # slower trend confirmation for long-term
            },
            "backtest": {
                "warmup_bars": 200,
            },
        },
        "short_term": {
            "description": "Swing trading — days to weeks. Faster indicators, tighter stops, lower warmup.",
            "rsi": {
                "period": 9,
                "thresholds": {"oversold": 35, "overbought": 65},
            },
            "macd": {
                "fast_period": 8,
                "slow_period": 17,
                "signal_period": 9,
            },
            "bollinger_bands": {
                "period": 14,
            },
            "moving_averages": {
                "periods": [10, 20, 50],
            },
            "stochastic": {
                "k_period": 9,
                "d_period": 3,
                "smooth_k": 3,
                "thresholds": {"oversold": 25, "overbought": 75},
            },
            "adx": {
                "period": 10,
            },
            "volume": {
                "obv_trend_period": 10,
                "price_trend_period": 10,
            },
            "fibonacci": {
                "swing_lookback": 30,
            },
            "support_resistance": {
                "fractal_lookback": 30,
                "fractal_order": 3,
            },
            "gaps": {
                "lookback": 15,
                "trend_period": 10,
                "consolidation_lookback": 15,
                "bb_percentile_lookback": 30,
                "consolidation_min_bars": 4,
                "exhaustion_min_trend_bars": 30,
            },
            "volume_range": {
                "period": 10,
                "lookback": 8,
            },
            "candlesticks": {
                "lookback": 8,
                "trend_period": 8,
            },
            "spikes": {
                "period": 10,
                "lookback": 15,
                "confirm_bars": 2,
            },
            "inside_outside": {
                "lookback": 15,
                "trend_period": 8,
                "breakout_bars": 2,
            },
            "overall_patterns": {
                "weights": {
                    "gaps": 0.15,
                    "volume_range": 0.20,
                    "candlesticks": 0.25,
                    "spikes": 0.15,
                    "inside_outside": 0.25,   # breakout setups very relevant for swing trading
                },
            },
            "overall": {
                "weights": {
                    "rsi": 0.15,
                    "macd": 0.20,
                    "bollinger_bands": 0.15,
                    "moving_averages": 0.10,
                    "stochastic": 0.15,
                    "adx": 0.10,
                    "volume": 0.10,
                    "fibonacci": 0.05,
                },
            },
            "strategy": {
                "rebalance_interval": 3,
                "stop_loss_pct": 0.03,
                "take_profit_pct": 0.20,
                "indicator_weight": 0.6,
                "pattern_weight": 0.4,
                "atr_stop_multiplier": 2.5,    # tighter ATR stop for swing trading
                "trend_confirm_period": 10,     # faster trend confirmation
            },
            "backtest": {
                "warmup_bars": 60,
            },
        },
        "day_trading": {
            "description": "Day trading — intraday positions only. Very fast indicators, tight stops, EOD flattening.",
            "rsi": {
                "period": 5,
                "thresholds": {"oversold": 30, "overbought": 70},
            },
            "macd": {
                "fast_period": 5,
                "slow_period": 13,
                "signal_period": 6,
            },
            "bollinger_bands": {
                "period": 10,
                "std_dev": 2.0,
            },
            "moving_averages": {
                "periods": [5, 10, 20],
            },
            "stochastic": {
                "k_period": 5,
                "d_period": 3,
                "smooth_k": 2,
                "thresholds": {"oversold": 20, "overbought": 80},
            },
            "adx": {
                "period": 7,
            },
            "volume": {
                "obv_trend_period": 10,
                "price_trend_period": 10,
            },
            "fibonacci": {
                "swing_lookback": 20,
            },
            "support_resistance": {
                "fractal_lookback": 20,
                "fractal_order": 3,
            },
            "gaps": {
                "lookback": 10,
                "min_gap_pct": 0.003,
                "trend_period": 7,
                "consolidation_lookback": 15,
                "bb_percentile_lookback": 30,
                "consolidation_min_bars": 3,
                "consolidation_max_return": 0.015,
                "exhaustion_min_return": 0.06,
                "exhaustion_min_distance_pct": 0.03,
                "exhaustion_min_trend_bars": 25,
                "intraday_min_gap_pct": 0.008,
            },
            "volume_range": {
                "period": 10,
                "lookback": 5,
            },
            "candlesticks": {
                "lookback": 5,
                "trend_period": 5,
            },
            "spikes": {
                "period": 10,
                "spike_std": 2.0,
                "lookback": 10,
                "confirm_bars": 2,
            },
            "inside_outside": {
                "lookback": 10,
                "trend_period": 5,
                "breakout_bars": 2,
                "outside_range_min": 1.15,   # lower threshold intraday — smaller bars
            },
            "overall_patterns": {
                "weights": {
                    "gaps": 0.10,              # gaps less meaningful intraday
                    "volume_range": 0.30,
                    "candlesticks": 0.25,
                    "spikes": 0.15,
                    "inside_outside": 0.20,    # breakout setups important intraday
                },
            },
            "overall": {
                "weights": {
                    "rsi": 0.15,
                    "macd": 0.25,
                    "bollinger_bands": 0.10,
                    "moving_averages": 0.05,
                    "stochastic": 0.20,
                    "adx": 0.10,
                    "volume": 0.10,
                    "fibonacci": 0.05,
                },
            },
            "strategy": {
                "rebalance_interval": 1,
                "stop_loss_pct": 0.015,
                "take_profit_pct": 0.03,
                "flatten_eod": True,
                "indicator_weight": 0.5,
                "pattern_weight": 0.5,
                "boost_strength": 0.7,      # stronger boost — patterns more meaningful intraday
                "boost_dead_zone": 0.2,     # narrower dead zone — react to smaller pattern signals
                "atr_stop_multiplier": 1.5,  # tight ATR stop for day trading
                "atr_stop_period": 10,       # shorter ATR period for intraday
                "trend_confirm_period": 10,  # fast confirmation for intraday
            },
            "backtest": {
                "warmup_bars": 30,
            },
        },
    },
    # ── Dividend scanner ──────────────────────────────────────────────────
    "dividend": {
        # Data fetch parameters
        "cagr_years": 5,            # years for dividend CAGR calculation
        "consistency_years": 10,    # years for payout consistency evaluation

        # Minimum thresholds — tickers below these are excluded from results
        "min_yield": 0.01,          # 1% minimum current yield
        "max_yield": 0.15,          # 15% cap (above this signals distress)
        "min_cagr": -0.10,          # allow small declines; -10% floor
        "min_consistency": 0.50,    # at least 50% of years paid a dividend
        "min_streak": 0,            # no minimum streak by default

        # Scoring weights — how much each metric counts in the composite score
        "weights": {
            "yield": 0.30,
            "growth": 0.30,
            "consistency": 0.20,
            "streak": 0.20,
        },

        # Yield scoring curve (linear interpolation)
        # yield <= low_target  → 0 points
        # yield == mid_target  → 5 points
        # yield >= high_target → 10 points
        # yield >  distress    → score penalised back toward 5
        "yield_scoring": {
            "low_target": 0.005,    # 0.5%
            "mid_target": 0.03,     # 3.0%
            "high_target": 0.06,    # 6.0%
            "distress_threshold": 0.10,  # 10% — above this starts penalty
        },

        # Growth scoring (CAGR → 0-10 score)
        "growth_scoring": {
            "negative_score": 2.0,  # score when CAGR < 0
            "zero_score": 4.0,      # score when CAGR ~ 0
            "target_cagr": 0.07,    # CAGR that maps to score 8
            "max_cagr": 0.15,       # CAGR that maps to score 10
        },

        # Consistency scoring (fraction → 0-10)
        "consistency_scoring": {
            "full_score_threshold": 1.0,    # 100% consistency → 10
            "half_score_threshold": 0.50,   # 50% → 5
        },

        # Streak scoring
        "streak_scoring": {
            "aristocrat_years": 25,     # >= 25yr → score 10
            "good_years": 10,           # >= 10yr → score 7
            "decent_years": 5,          # >= 5yr → score 5
        },

        # Technical score blending (optional)
        "blend_technical": False,       # whether to factor in technical analysis score
        "technical_weight": 0.20,       # weight for technical score when blending
        # When blending, dividend weight = 1 - technical_weight

        # Scanner concurrency
        "max_workers": 8,
    },

    # ── Dollar Cost Averaging (DCA) ──────────────────────────────────────
    "dca": {
        # Base parameters
        "base_amount": 500,           # dollars per buy
        "frequency": "monthly",       # "weekly", "biweekly", "monthly"
        "drip": True,                 # reinvest dividends automatically

        # Mode: "pure", "dip_weighted", "score_integrated"
        "mode": "dip_weighted",

        # Dip detection thresholds (percentage drop from recent high)
        "dip_thresholds": {
            "mild_drop_pct": 5.0,     # price down 5% from recent high
            "strong_drop_pct": 10.0,  # price down 10%
            "extreme_drop_pct": 20.0, # price down 20%
            "lookback_days": 30,      # window for "recent high"
        },

        # Multiplier tiers — how much to scale base_amount at each dip level
        "multipliers": {
            "normal": 1.0,
            "mild_dip": 1.5,
            "strong_dip": 2.0,
            "extreme_dip": 3.0,
        },

        # Score-integrated mode thresholds
        "score_thresholds": {
            "buy_zone_below": 3.5,    # score below this = BUY zone
            "oversold_rsi": 30.0,     # RSI below this = oversold
            "bb_percentile_low": 10.0,  # BB percentile below this = near lower band
        },

        # Watchlist DCA context — enhanced dip analysis for the live
        # watchlist monitor.  These settings augment the raw dip-detection
        # tiers with volatility normalisation and regime awareness.
        "watchlist_context": {
            "volatility_window": 60,         # trailing days for daily vol calc
            "vol_severe_sigma": 2.5,         # dip ≥ this many σ → statistically severe
            "vol_notable_sigma": 1.5,        # dip ≥ this many σ → notable
            "crisis_return_threshold": -0.20, # total return ≤ this → crisis regime
            "regime_adjustment": {
                "bear_max_multiplier": 1.5,  # cap multiplier in bear/crisis
                "bull_pullback_bonus": 0.5,  # boost multiplier for bull pullback
            },
        },

        # DCA scanner — ranks entire universe by DCA attractiveness.
        # The composite score (0-100) is a weighted average of
        # normalised sub-scores.
        "scanner": {
            "score_weights": {
                "dip_sigma": 0.30,           # volatility-normalised dip severity
                "tier_multiplier": 0.25,     # allocation multiplier tier
                "technical": 0.20,           # inverse composite (lower = more attractive)
                "confidence": 0.10,          # DCA confidence level
                "regime": 0.15,              # regime favourability
            },
            "min_dca_score": 0,              # minimum score to show in results (0=all)
        },

        # Safety gates
        "safety": {
            "max_multiplier": 3.0,           # absolute cap on any single multiplier
            "max_period_allocation": 1500,   # max dollars in any single period
            "skip_breakaway_gaps": True,     # don't overweight on breakaway-down gaps
            "min_volume_ratio": 0.5,         # skip overweight if volume < 50% of avg
            "crisis_suppression": {
                "enabled": True,             # master toggle
                "min_signals": 2,            # how many signals must fire to suppress
                "composite_below": 2.0,      # composite score < this = crisis signal
                "panic_rsi_below": 20.0,     # RSI < this = crisis signal
                "volume_spike_above": 3.0,   # volume / avg_volume > this = crisis signal
            },
        },

        # Budget mode — alternative to fixed base_amount.
        # Instead of specifying $/period, specify a total budget over the
        # backtest period and let the engine compute the base amount.
        "budget": {
            "enabled": False,               # toggle budget mode
            "total_budget": 50000,          # total dollars to invest over the period
            "reserve_method": "conservative",  # "conservative" or "adaptive"
            # conservative: assumes every period could be max multiplier
            # adaptive: uses historical dip frequency to estimate reserve
        },
    },
    # ------------------------------------------------------------------
    # Watchlist — live signal monitor
    # ------------------------------------------------------------------
    "watchlist": {
        # Each entry: plain string "AAPL" or dict {"ticker": "AAPL",
        # "regime_override": "strong_trend", "sub_type_override": "steady_compounder"}
        "tickers": [],
        "data_period": "1y",                # history to fetch for indicator warmup
        "interval": "1d",                   # data interval
        "trading_mode": "long_only",        # "long_only" or "long_short"
        "state_file": "watchlist_state.json",  # portfolio state persistence file
        "initial_cash": 100_000.0,          # starting cash (first run only)
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class Config:
    """Loads and validates the YAML config, falling back to defaults for
    any missing keys.

    Usage::

        cfg = Config.load("config.yaml")
        rsi_period = cfg["rsi"]["period"]
    """

    def __init__(self, data: dict[str, Any], path: str | None = None) -> None:
        self._data = data
        self.path = path
        self._active_objective: str | None = None

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        """Load config from *path*.  Falls back to defaults for missing keys.

        If *path* is None, looks for ``config.yaml`` next to this file, then
        in the current working directory.

        Tickers are loaded from a separate file (``watchlist_tickers.yaml``)
        and merged into the ``watchlist`` section.  If the ticker file does
        not exist, the ``watchlist.tickers`` value from ``config.yaml`` (or
        defaults) is preserved for backward compatibility.
        """
        search_paths = []
        if path:
            search_paths.append(Path(path))
        else:
            search_paths += [
                Path(__file__).parent / "config.yaml",
                Path(os.getcwd()) / "config.yaml",
            ]

        loaded: dict = {}
        resolved_path: str | None = None
        for p in search_paths:
            if p.exists():
                with open(p, "r") as fh:
                    loaded = yaml.safe_load(fh) or {}
                resolved_path = str(p)
                break

        merged = _deep_merge(DEFAULT_CONFIG, loaded)

        # Overlay tickers from the dedicated ticker file (if it exists).
        project_root = Path(resolved_path).parent if resolved_path else Path(__file__).parent
        ext_tickers = load_watchlist_tickers(project_root)
        if ext_tickers:
            # Convert back to the raw list-of-dicts format config expects.
            merged.setdefault("watchlist", {})["tickers"] = [
                {k: v for k, v in t.items() if v is not None}
                for t in ext_tickers
            ]

        cfg = cls(merged, resolved_path)
        errors = cfg.validate()
        if errors:
            print("[config] Validation warnings:")
            for e in errors:
                print(f"  • {e}")
        return cfg

    @classmethod
    def defaults(cls) -> "Config":
        """Return a Config populated entirely with defaults."""
        return cls(copy.deepcopy(DEFAULT_CONFIG), None)

    # ------------------------------------------------------------------
    # Objective presets
    # ------------------------------------------------------------------

    @property
    def active_objective(self) -> str | None:
        """The name of the currently applied objective, or ``None``."""
        return self._active_objective

    def available_objectives(self) -> list[str]:
        """Return the list of defined objective names."""
        objectives = self._data.get("objectives", {})
        return [k for k in objectives if isinstance(objectives[k], dict)]

    def apply_objective(self, name: str) -> None:
        """Apply a named objective preset on top of the current config.

        The preset is a partial dict deep-merged over the base config, so
        only the keys specified in the preset are overridden.

        Raises ``ValueError`` if the objective name is not found.
        """
        objectives = self._data.get("objectives", {})
        if name not in objectives:
            available = self.available_objectives()
            raise ValueError(
                f"Unknown objective '{name}'. "
                f"Available: {', '.join(available) if available else '(none defined)'}"
            )

        preset = objectives[name]
        if not isinstance(preset, dict):
            raise ValueError(f"Objective '{name}' must be a dict, got {type(preset).__name__}")

        # Extract overrides (everything except 'description')
        overrides = {k: v for k, v in preset.items() if k != "description"}

        # Deep-merge overrides into the current config data
        self._data = _deep_merge(self._data, overrides)
        self._active_objective = name

        # Re-validate after applying the preset
        errors = self.validate()
        if errors:
            print(f"[config] Validation warnings after applying objective '{name}':")
            for e in errors:
                print(f"  • {e}")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of validation error/warning strings (empty = OK)."""
        errors: list[str] = []

        # Weights must be positive
        weights: dict = self._data.get("overall", {}).get("weights", {})
        for k, v in weights.items():
            if not isinstance(v, (int, float)) or v < 0:
                errors.append(f"overall.weights.{k} must be a non-negative number, got {v!r}")

        # Pattern weights must be positive
        pat_weights: dict = self._data.get("overall_patterns", {}).get("weights", {})
        for k, v in pat_weights.items():
            if not isinstance(v, (int, float)) or v < 0:
                errors.append(f"overall_patterns.weights.{k} must be a non-negative number, got {v!r}")

        # Strategy combination weights must sum sensibly
        combo_mode = self._data.get("strategy", {}).get("combination_mode", "weighted")
        if combo_mode not in ("weighted", "gate", "boost"):
            errors.append(
                f"strategy.combination_mode must be 'weighted', 'gate', or 'boost', got {combo_mode!r}"
            )

        # RSI thresholds
        rsi = self._data.get("rsi", {}).get("thresholds", {})
        if rsi.get("oversold", 30) >= rsi.get("overbought", 70):
            errors.append("rsi.thresholds.oversold must be less than overbought")

        # Stochastic thresholds
        stoch = self._data.get("stochastic", {}).get("thresholds", {})
        if stoch.get("oversold", 20) >= stoch.get("overbought", 80):
            errors.append("stochastic.thresholds.oversold must be less than overbought")

        # ADX thresholds
        adx = self._data.get("adx", {}).get("thresholds", {})
        if adx.get("weak", 30) >= adx.get("moderate", 50):
            errors.append("adx.thresholds.weak must be less than moderate")

        # MA periods must be sorted ascending
        ma_periods = self._data.get("moving_averages", {}).get("periods", [20, 50, 200])
        if ma_periods != sorted(ma_periods):
            errors.append("moving_averages.periods should be in ascending order")

        # Strategy thresholds
        strat = self._data.get("strategy", {})
        st = strat.get("score_thresholds", {})
        short_below = st.get("short_below", 3.5)
        hold_below = st.get("hold_below", 6.0)
        if short_below >= hold_below:
            errors.append(
                "strategy.score_thresholds.short_below must be less than hold_below"
            )
        if not (0 <= short_below <= 10) or not (0 <= hold_below <= 10):
            errors.append(
                "strategy.score_thresholds values must be between 0 and 10"
            )

        sizing = strat.get("position_sizing", "fixed")
        if sizing not in ("fixed", "percent_equity"):
            errors.append(
                f"strategy.position_sizing must be 'fixed' or 'percent_equity', got {sizing!r}"
            )

        # Threshold mode
        threshold_mode = strat.get("threshold_mode", "fixed")
        if threshold_mode not in ("fixed", "percentile"):
            errors.append(
                f"strategy.threshold_mode must be 'fixed' or 'percentile', got {threshold_mode!r}"
            )

        # Percentile thresholds (validate even if mode is "fixed" — config should be valid)
        pct_cfg = strat.get("percentile_thresholds", {})
        short_pct = pct_cfg.get("short_percentile", 25)
        long_pct = pct_cfg.get("long_percentile", 75)
        lookback = pct_cfg.get("lookback_bars", 60)

        if not isinstance(short_pct, (int, float)) or not (0 <= short_pct <= 100):
            errors.append(
                f"strategy.percentile_thresholds.short_percentile must be 0-100, got {short_pct!r}"
            )
        if not isinstance(long_pct, (int, float)) or not (0 <= long_pct <= 100):
            errors.append(
                f"strategy.percentile_thresholds.long_percentile must be 0-100, got {long_pct!r}"
            )
        if isinstance(short_pct, (int, float)) and isinstance(long_pct, (int, float)):
            if short_pct >= long_pct:
                errors.append(
                    "strategy.percentile_thresholds.short_percentile must be less than long_percentile"
                )
        if not isinstance(lookback, int) or lookback < 10:
            errors.append(
                f"strategy.percentile_thresholds.lookback_bars must be an integer >= 10, got {lookback!r}"
            )

        for pct_key in ("stop_loss_pct", "take_profit_pct"):
            val = strat.get(pct_key)
            if val is not None and (not isinstance(val, (int, float)) or val <= 0):
                errors.append(f"strategy.{pct_key} must be a positive number, got {val!r}")

        rebal = strat.get("rebalance_interval", 5)
        if not isinstance(rebal, int) or rebal < 1:
            errors.append(
                f"strategy.rebalance_interval must be a positive integer, got {rebal!r}"
            )

        # ATR-adaptive stop loss
        atr_mult = strat.get("atr_stop_multiplier", 2.5)
        if not isinstance(atr_mult, (int, float)) or atr_mult <= 0:
            errors.append(
                f"strategy.atr_stop_multiplier must be a positive number, got {atr_mult!r}"
            )

        atr_period = strat.get("atr_stop_period", 14)
        if not isinstance(atr_period, int) or atr_period < 1:
            errors.append(
                f"strategy.atr_stop_period must be a positive integer, got {atr_period!r}"
            )

        # Trend confirmation filter
        trend_period = strat.get("trend_confirm_period", 20)
        if not isinstance(trend_period, int) or trend_period < 1:
            errors.append(
                f"strategy.trend_confirm_period must be a positive integer, got {trend_period!r}"
            )

        # Indicator / pattern weights must be non-negative
        ind_weight = strat.get("indicator_weight", 0.7)
        pat_weight = strat.get("pattern_weight", 0.3)
        if isinstance(ind_weight, (int, float)) and ind_weight < 0:
            errors.append(
                f"strategy.indicator_weight must be non-negative, got {ind_weight!r}"
            )
        if isinstance(pat_weight, (int, float)) and pat_weight < 0:
            errors.append(
                f"strategy.pattern_weight must be non-negative, got {pat_weight!r}"
            )

        # Gate thresholds: SHORT threshold must be below LONG threshold
        gate_ind_min = strat.get("gate_indicator_min", 5.5)
        gate_ind_max = strat.get("gate_indicator_max", 4.5)
        gate_pat_min = strat.get("gate_pattern_min", 5.5)
        gate_pat_max = strat.get("gate_pattern_max", 4.5)
        if (
            isinstance(gate_ind_min, (int, float))
            and isinstance(gate_ind_max, (int, float))
            and gate_ind_max >= gate_ind_min
        ):
            errors.append(
                "strategy.gate_indicator_max (SHORT) must be less than "
                f"gate_indicator_min (LONG), got max={gate_ind_max!r} min={gate_ind_min!r}"
            )
        if (
            isinstance(gate_pat_min, (int, float))
            and isinstance(gate_pat_max, (int, float))
            and gate_pat_max >= gate_pat_min
        ):
            errors.append(
                "strategy.gate_pattern_max (SHORT) must be less than "
                f"gate_pattern_min (LONG), got max={gate_pat_max!r} min={gate_pat_min!r}"
            )

        # Confidence thresholds: high > medium > 0
        conf = strat.get("confidence_thresholds", {})
        conf_high = conf.get("high", 1.5)
        conf_medium = conf.get("medium", 0.5)
        if (
            isinstance(conf_high, (int, float))
            and isinstance(conf_medium, (int, float))
        ):
            if conf_medium <= 0:
                errors.append(
                    f"strategy.confidence_thresholds.medium must be positive, got {conf_medium!r}"
                )
            if conf_high <= conf_medium:
                errors.append(
                    "strategy.confidence_thresholds.high must be greater than medium, "
                    f"got high={conf_high!r} medium={conf_medium!r}"
                )

        # Backtest parameters
        bt = self._data.get("backtest", {})
        cash = bt.get("initial_cash", 100_000)
        if not isinstance(cash, (int, float)) or cash <= 0:
            errors.append(f"backtest.initial_cash must be positive, got {cash!r}")

        warmup = bt.get("warmup_bars", 50)
        if not isinstance(warmup, int) or warmup < 1:
            errors.append(f"backtest.warmup_bars must be a positive integer, got {warmup!r}")

        max_warmup_ratio = bt.get("max_warmup_ratio", 0.5)
        if not isinstance(max_warmup_ratio, (int, float)) or not (0 < max_warmup_ratio < 1):
            errors.append(
                f"backtest.max_warmup_ratio must be between 0 and 1 (exclusive), got {max_warmup_ratio!r}"
            )

        slippage = bt.get("slippage_pct", 0.001)
        if not isinstance(slippage, (int, float)) or slippage < 0:
            errors.append(f"backtest.slippage_pct must be non-negative, got {slippage!r}")

        commission_pct = bt.get("commission_pct", 0.0)
        if not isinstance(commission_pct, (int, float)) or commission_pct < 0:
            errors.append(f"backtest.commission_pct must be non-negative, got {commission_pct!r}")

        commission_mode = bt.get("commission_mode", "additive")
        if commission_mode not in ("additive", "max"):
            errors.append(
                f"backtest.commission_mode must be 'additive' or 'max', got {commission_mode!r}"
            )

        # Suitability parameters
        suit = self._data.get("suitability", {})
        mode_override = suit.get("mode_override", "auto")
        valid_modes = ("auto", "long_short", "long_only", "hold_only")
        if mode_override not in valid_modes:
            errors.append(
                f"suitability.mode_override must be one of {valid_modes}, got {mode_override!r}"
            )

        for key in ("min_volume", "min_volume_for_short"):
            val = suit.get(key)
            if val is not None and (not isinstance(val, (int, float)) or val < 0):
                errors.append(f"suitability.{key} must be non-negative, got {val!r}")

        for key in ("min_atr_pct", "min_atr_for_short", "min_adx_for_short"):
            val = suit.get(key)
            if val is not None and (not isinstance(val, (int, float)) or val < 0):
                errors.append(f"suitability.{key} must be non-negative, got {val!r}")

        atr_period = suit.get("atr_period", 14)
        if not isinstance(atr_period, int) or atr_period < 1:
            errors.append(f"suitability.atr_period must be a positive integer, got {atr_period!r}")

        trend_ma_period = suit.get("trend_ma_period", 200)
        if not isinstance(trend_ma_period, int) or trend_ma_period < 1:
            errors.append(
                f"suitability.trend_ma_period must be a positive integer, got {trend_ma_period!r}"
            )

        max_pct = suit.get("max_pct_above_ma", 65.0)
        if not isinstance(max_pct, (int, float)) or not (0 <= max_pct <= 100):
            errors.append(
                f"suitability.max_pct_above_ma must be between 0 and 100, got {max_pct!r}"
            )

        # Regime parameters
        regime = self._data.get("regime", {})
        for key in ("adx_strong_trend", "adx_weak", "atr_pct_high", "atr_pct_low"):
            val = regime.get(key)
            if val is not None and (not isinstance(val, (int, float)) or val < 0):
                errors.append(f"regime.{key} must be non-negative, got {val!r}")

        adx_strong = regime.get("adx_strong_trend", 30.0)
        adx_weak = regime.get("adx_weak", 20.0)
        if isinstance(adx_strong, (int, float)) and isinstance(adx_weak, (int, float)):
            if adx_weak >= adx_strong:
                errors.append("regime.adx_weak must be less than regime.adx_strong_trend")

        for key in ("trend_consistency_high", "trend_consistency_low"):
            val = regime.get(key)
            if val is not None and (not isinstance(val, (int, float)) or not (0 <= val <= 100)):
                errors.append(f"regime.{key} must be between 0 and 100, got {val!r}")

        for key in ("bb_squeeze_percentile", "bb_expansion_percentile"):
            val = regime.get(key)
            if val is not None and (not isinstance(val, (int, float)) or not (0 <= val <= 100)):
                errors.append(f"regime.{key} must be between 0 and 100, got {val!r}")

        regime_adapt = regime.get("strategy_adaptation", {})
        if not isinstance(regime_adapt, dict):
            errors.append("regime.strategy_adaptation must be a dict")

        # Objectives — must be dicts with valid structure
        objectives = self._data.get("objectives", {})
        if not isinstance(objectives, dict):
            errors.append("objectives must be a dict")
        else:
            for obj_name, obj_data in objectives.items():
                if not isinstance(obj_data, dict):
                    errors.append(f"objectives.{obj_name} must be a dict")

        return errors

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def section(self, key: str) -> dict:
        """Return a config section, guaranteed to be a dict."""
        return self._data.get(key) or {}

    def normalized_weights(self) -> dict[str, float]:
        """Return overall indicator weights normalized so they sum to 1.0."""
        weights = self._data["overall"]["weights"]
        total = sum(weights.values())
        if total == 0:
            equal = 1.0 / len(weights)
            return {k: equal for k in weights}
        return {k: v / total for k, v in weights.items()}

    def normalized_pattern_weights(self) -> dict[str, float]:
        """Return overall pattern weights normalized so they sum to 1.0."""
        weights = self._data.get("overall_patterns", {}).get("weights", {})
        if not weights:
            return {}
        total = sum(weights.values())
        if total == 0:
            equal = 1.0 / len(weights)
            return {k: equal for k in weights}
        return {k: v / total for k, v in weights.items()}

    # ------------------------------------------------------------------
    # Config generation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy of the internal configuration dictionary."""
        return copy.deepcopy(self._data)

    def save(self, output_path: str) -> None:
        """Persist the current configuration to a YAML file.

        This writes the full merged config (defaults + user overrides +
        objective preset if any).  The resulting file can later be loaded
        with ``Config.load(path)``.
        """
        with open(output_path, "w") as fh:
            yaml.dump(
                self._data,
                fh,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create a Config from an arbitrary dict (e.g. dashboard session state).

        Deep-merges *data* over ``DEFAULT_CONFIG`` so that any missing keys
        fall back to defaults, then validates.
        """
        merged = _deep_merge(DEFAULT_CONFIG, data)
        cfg = cls(merged, path=None)
        errors = cfg.validate()
        if errors:
            print("[config] Validation warnings:")
            for e in errors:
                print(f"  • {e}")
        return cfg

    @staticmethod
    def generate_default(output_path: str = "config.yaml") -> None:
        """Write a fresh default config.yaml to *output_path*."""
        template_path = Path(__file__).parent / "config.yaml"
        if template_path.exists():
            import shutil
            shutil.copy(template_path, output_path)
            print(f"Default config written to: {output_path}")
        else:
            # Fallback: dump defaults as YAML
            with open(output_path, "w") as fh:
                yaml.dump(DEFAULT_CONFIG, fh, default_flow_style=False, sort_keys=False)
            print(f"Default config written to: {output_path}")


# ---------------------------------------------------------------------------
# Watchlist ticker file — tickers are stored separately from config.yaml
# ---------------------------------------------------------------------------

TICKER_FILE_NAME = "watchlist_tickers.yaml"

# Type alias: each ticker entry is either a plain string or a dict with
# at least a "ticker" key, and optional "regime_override" / "sub_type_override".
WatchlistTickerEntry = str | dict[str, str | None]


def _normalise_ticker_entry(entry: WatchlistTickerEntry) -> dict[str, str | None]:
    """Ensure a ticker entry is in dict form."""
    if isinstance(entry, str):
        return {
            "ticker": entry.upper().strip(),
            "regime_override": None,
            "sub_type_override": None,
        }
    return {
        "ticker": str(entry.get("ticker", "")).upper().strip(),
        "regime_override": entry.get("regime_override"),
        "sub_type_override": entry.get("sub_type_override"),
    }


def _ticker_file_path(project_root: str | Path | None = None) -> Path:
    """Return the path to the watchlist ticker file.

    Searches next to this file first, then in *project_root* if given.
    """
    candidates: list[Path] = [Path(__file__).parent / TICKER_FILE_NAME]
    if project_root:
        candidates.append(Path(project_root) / TICKER_FILE_NAME)
    for p in candidates:
        if p.exists():
            return p
    # Default write location: next to this file
    return candidates[0]


def load_watchlist_tickers(
    project_root: str | Path | None = None,
) -> list[dict[str, str | None]]:
    """Load tickers from the dedicated watchlist ticker file.

    Returns a list of normalised dicts (same format as
    ``parse_watchlist_tickers``).  Returns an empty list if the file
    does not exist.
    """
    path = _ticker_file_path(project_root)
    if not path.exists():
        return []
    try:
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
    except (yaml.YAMLError, OSError):
        return []
    return parse_watchlist_tickers(data.get("tickers", []))


def save_watchlist_tickers(
    project_root: str | Path | None,
    tickers: Sequence[WatchlistTickerEntry],
) -> None:
    """Write the watchlist ticker list to the dedicated ticker file.

    The file (``watchlist_tickers.yaml``) is separate from ``config.yaml``
    because tickers are user data, not application configuration.

    The *project_root* argument determines where the file is written.
    If ``None``, it defaults to the directory containing this module.
    """
    if project_root:
        path = Path(project_root) / TICKER_FILE_NAME
    else:
        path = Path(__file__).parent / TICKER_FILE_NAME

    entries: list[dict[str, str | None]] = []
    for raw in tickers:
        entry = _normalise_ticker_entry(raw)
        if entry["ticker"]:
            entries.append(
                {k: v for k, v in entry.items() if v is not None}
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.dump(
            {"tickers": entries},
            fh,
            default_flow_style=False,
            sort_keys=False,
        )


def parse_watchlist_tickers(
    raw_tickers: list,
) -> list[dict[str, str | None]]:
    """Parse watchlist tickers from config (supports both formats).

    Handles both legacy plain-string lists (``["AAPL", "MSFT"]``) and the
    new dict format (``[{ticker: "AAPL", regime_override: "strong_trend"}]``).

    Returns a list of normalised dicts with keys:
        ``ticker``, ``regime_override``, ``sub_type_override``
    (override values are ``None`` when not set).
    """
    result: list[dict[str, str | None]] = []
    for entry in raw_tickers:
        norm = _normalise_ticker_entry(entry)
        if norm["ticker"]:
            result.append(norm)
    return result
