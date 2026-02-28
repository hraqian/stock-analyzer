"""
dashboard.py — Streamlit-based graphical dashboard for the stock analyzer.

Phase 2: Interactive dashboard with editable parameters and config loadouts.
Run with:  streamlit run dashboard.py

Features:
  - Sidebar: ticker, period/date-range, interval, objective preset
  - **Interactive parameter editing** for all indicator, pattern, strategy,
    scoring, and backtest params — changes auto re-run analysis
  - **3 config loadout slots** — save/load full config snapshots
  - Candlestick price chart with indicator & pattern score overlays
  - Support/resistance levels on the price chart
  - Indicator & pattern breakdown tables
  - Backtest equity curve with trade entry/exit markers
  - Performance metrics summary
  - Suitability assessment display
  - Score distribution histogram
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so all internal imports work
_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analysis.analyzer import Analyzer, AnalysisResult
from analysis.multi_timeframe import _build_percentile_window
from analysis.scorer import CompositeScorer
from analysis.pattern_scorer import PatternCompositeScorer
from config import Config, DEFAULT_CONFIG
from data.yahoo import YahooFinanceProvider
from engine.backtest import BacktestEngine, BacktestResult
from engine.dca import DCABacktester, DCAResult
from engine.score_strategy import ScoreBasedStrategy
from engine.suitability import SuitabilityAnalyzer, TradingMode
from engine.regime import RegimeAssessment, RegimeType, RegimeSubType, REGIME_LABELS
from engine.strategy import StrategyContext
from indicators.registry import IndicatorRegistry
from patterns.registry import PatternRegistry
from scanner import Scanner, ScanResult
from analysis.dividend import DividendScanner, DividendScanResult
from data.universes import available as available_universes, load as load_universe

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Stock Technical Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
INTRADAY_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]
DAILY_INTERVALS = ["1d", "5d", "1wk", "1mo", "3mo"]
ALL_INTERVALS = INTRADAY_INTERVALS + DAILY_INTERVALS
TRADING_MODES = ["auto", "long_short", "long_only", "hold_only"]

LOADOUT_DIR = _PROJECT_ROOT
LOADOUT_SLOTS = 3

# Color palette
COLOR_BULLISH = "#26a69a"
COLOR_BEARISH = "#ef5350"
COLOR_NEUTRAL = "#ffca28"
COLOR_BG = "#0e1117"
COLOR_GRID = "#1e2130"
COLOR_IND_SCORE = "#42a5f5"
COLOR_PAT_SCORE = "#ab47bc"
COLOR_EQUITY = "#42a5f5"
COLOR_BENCHMARK = "#78909c"


# ---------------------------------------------------------------------------
# Config hashing (for cache invalidation)
# ---------------------------------------------------------------------------

def _config_hash(data: dict) -> str:
    """Return a stable hash of a config dict for use as a cache key."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Config ↔ session state helpers
# ---------------------------------------------------------------------------

def _get_config() -> Config:
    """Build a Config object from the current session state config_data."""
    data = st.session_state.get("config_data")
    if data is None:
        cfg = Config.load()
        st.session_state["config_data"] = cfg.to_dict()
        return cfg
    return Config.from_dict(data)


def _init_config_data() -> None:
    """Ensure session_state['config_data'] is initialised."""
    if "config_data" not in st.session_state:
        cfg = Config.load()
        st.session_state["config_data"] = cfg.to_dict()


def _apply_objective_to_session(objective: str | None) -> None:
    """Re-load base config, apply objective if any, store in session state.

    This is called when the objective dropdown changes.  We must start from
    the base YAML (not the already-mutated session state) so that switching
    objectives gives a clean slate.
    """
    cfg = Config.load()
    if objective:
        cfg.apply_objective(objective)
    st.session_state["config_data"] = cfg.to_dict()


# ---------------------------------------------------------------------------
# Loadout helpers
# ---------------------------------------------------------------------------

def _loadout_path(slot: int) -> Path:
    return LOADOUT_DIR / f"config_slot_{slot}.yaml"


def _save_loadout(slot: int) -> None:
    cfg = _get_config()
    cfg.save(str(_loadout_path(slot)))


def _load_loadout(slot: int) -> bool:
    path = _loadout_path(slot)
    if not path.exists():
        return False
    cfg = Config.load(str(path))
    st.session_state["config_data"] = cfg.to_dict()
    return True


def _loadout_exists(slot: int) -> bool:
    return _loadout_path(slot).exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def score_color(score: float) -> str:
    """Return hex color for a 0-10 score."""
    if score <= 3.5:
        return COLOR_BEARISH
    if score <= 6.0:
        return COLOR_NEUTRAL
    return COLOR_BULLISH


def score_bar_html(score: float, width: int = 100) -> str:
    """Return an HTML progress-bar-style visual for a score, color-coded."""
    pct = max(0, min(100, score / 10 * 100))
    color = score_color(score)
    return (
        f'<div style="display:flex;align-items:center;gap:6px;">'
        f'<span style="color:{color};font-weight:600;min-width:28px;">{score:.1f}</span>'
        f'<div style="background:#333;border-radius:3px;width:{width}px;height:12px;flex-shrink:0;">'
        f'<div style="background:{color};border-radius:3px;width:{pct:.0f}%;height:100%;"></div>'
        f'</div>'
        f'</div>'
    )


def is_intraday(interval: str) -> bool:
    return interval.lower().strip() in set(INTRADAY_INTERVALS)


def _build_score_table_html(
    rows: list[dict],
    name_col: str,
    columns: list[str],
) -> str:
    """Build a dark-themed HTML table with color-coded score bars.

    Args:
        rows: List of row dicts.  Each dict must have keys matching *columns*.
        name_col: Which column holds the row name (e.g. "Indicator" or "Pattern").
        columns: Ordered list of column names to render.

    Special handling:
      - "Score" column → rendered via ``score_bar_html()``.
      - "Weight" column → if float, formatted as percentage; if already str, used as-is.
      - Last row is rendered bold (OVERALL row).
    """
    # --- CSS ---
    style = (
        "<style>"
        ".score-table { width:100%; border-collapse:collapse; font-size:0.85rem; }"
        ".score-table th { text-align:left; padding:6px 8px; border-bottom:2px solid #444; "
        "  color:#aaa; font-weight:600; }"
        ".score-table td { padding:5px 8px; border-bottom:1px solid #2a2a2a; color:#ddd; "
        "  vertical-align:middle; }"
        ".score-table tr:last-child td { border-bottom:none; font-weight:700; }"
        ".score-table tr:hover td { background:#1a1d2e; }"
        "</style>"
    )

    # --- Header ---
    header_cells = "".join(f"<th>{c}</th>" for c in columns)
    header = f"<tr>{header_cells}</tr>"

    # --- Rows ---
    body_rows = []
    for row in rows:
        cells = []
        for col in columns:
            val = row.get(col, "")
            if col == "Score":
                # Render color-coded bar
                cell_html = score_bar_html(float(val))
            elif col == "Weight":
                if isinstance(val, (int, float)):
                    cell_html = f"{val * 100:.0f}%"
                else:
                    cell_html = str(val)
            else:
                cell_html = str(val)
            cells.append(f"<td>{cell_html}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return (
        f'{style}<table class="score-table">'
        f"<thead>{header}</thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        f"</table>"
    )


# ---------------------------------------------------------------------------
# Sidebar — parameter editors
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Parameter descriptions — short guidance shown beneath each widget
# ---------------------------------------------------------------------------

PARAM_DESCRIPTIONS: dict[str, str] = {
    # -- Indicator weights --
    "iw_rsi": "higher weight = RSI has more influence on composite score",
    "iw_macd": "higher weight = MACD has more influence on composite score",
    "iw_bollinger_bands": "higher weight = Bollinger Bands has more influence on composite score",
    "iw_moving_averages": "higher weight = MA crossovers have more influence on composite score",
    "iw_stochastic": "higher weight = Stochastic has more influence on composite score",
    "iw_adx": "higher weight = trend strength has more influence on composite score",
    "iw_volume": "higher weight = volume analysis has more influence on composite score",
    "iw_fibonacci": "higher weight = Fibonacci levels have more influence on composite score",
    # -- Indicator params --
    "rsi.period": "shorter = more reactive to price changes, longer = smoother",
    "rsi.thresholds.oversold": "higher = fewer oversold signals, more conservative",
    "rsi.thresholds.overbought": "lower = fewer overbought signals, more conservative",
    "macd.fast_period": "shorter = more sensitive to recent price moves",
    "macd.slow_period": "shorter = more reactive, longer = filters out noise",
    "macd.signal_period": "shorter = faster signal crossovers, more trades",
    "bollinger_bands.period": "shorter = tighter bands, more frequent breakout signals",
    "bollinger_bands.std_dev": "higher = wider bands, fewer signals but more reliable",
    "moving_averages.periods": "add shorter periods for faster signals, longer for trend confirmation",
    "stochastic.k_period": "shorter = more reactive, longer = smoother",
    "stochastic.d_period": "shorter = faster signal line, more crossovers",
    "stochastic.smooth_k": "higher = smoother %K, fewer false signals",
    "stochastic.thresholds.oversold": "higher = fewer oversold signals, more conservative",
    "stochastic.thresholds.overbought": "lower = fewer overbought signals, more conservative",
    "adx.period": "shorter = detects trend changes faster, noisier",
    "adx.thresholds.weak": "higher = requires stronger trend to score above neutral",
    "adx.thresholds.moderate": "higher = requires very strong trend for high scores",
    "volume.obv_trend_period": "shorter = faster OBV trend detection, noisier",
    "volume.price_trend_period": "shorter = faster price-volume divergence detection",
    "fibonacci.swing_lookback": "shorter = more local swings, longer = major support/resistance",
    # -- Pattern weights --
    "pw_gaps": "higher = gap patterns have more influence on pattern score",
    "pw_volume_range": "higher = volume-range correlation has more influence",
    "pw_candlesticks": "higher = candlestick patterns have more influence",
    "pw_spikes": "higher = price spikes have more influence",
    "pw_inside_outside": "higher = inside/outside bars have more influence",
    # -- Combination mode --
    "strategy.combination_mode": "weighted = blend scores, gate = both must agree, boost = patterns amplify indicator signal",
    "strategy.indicator_weight": "higher = indicator score dominates the blended signal",
    "strategy.pattern_weight": "higher = pattern score dominates the blended signal",
    "strategy.gate_indicator_min": "higher = requires stronger indicator signal to go long",
    "strategy.gate_indicator_max": "lower = requires weaker indicator signal to go short",
    "strategy.gate_pattern_min": "higher = requires stronger pattern signal to go long",
    "strategy.gate_pattern_max": "lower = requires weaker pattern signal to go short",
    "strategy.boost_strength": "higher = patterns amplify/dampen the indicator signal more",
    "strategy.boost_dead_zone": "higher = patterns need to deviate more from 5.0 to have any effect",
    # -- Scoring thresholds --
    "strategy.threshold_mode": "fixed = static thresholds, percentile = adaptive to recent score distribution",
    "strategy.score_thresholds.short_below": "lower = fewer short signals, more conservative",
    "strategy.score_thresholds.hold_below": "lower = more long signals, more aggressive",
    "strategy.percentile_thresholds.short_percentile": "lower = fewer short signals, more conservative",
    "strategy.percentile_thresholds.long_percentile": "lower = more long signals, more aggressive",
    "strategy.percentile_thresholds.lookback_bars": "shorter = adapts to recent conditions faster",
    # -- Strategy params --
    "strategy.stop_loss_pct": "tighter = limits losses but more likely to be stopped out",
    "strategy.take_profit_pct": "tighter = locks in gains sooner but caps upside; disabled in strong_trend regime",
    "strategy.position_sizing": "percent_equity = risk scales with portfolio, fixed = constant share count",
    "strategy.percent_equity": "higher = larger positions, more risk per trade",
    "strategy.fixed_quantity": "higher = more shares per trade",
    "strategy.rebalance_interval": "lower = checks signals more often, more responsive; higher = less whipsaw",
    "strategy.flatten_eod": "enable for day-trading to close all positions at end of day",
    "strategy.reentry_grace_bars": "bars after exit where trend filter is skipped for faster re-entry",
    # -- Consecutive loss cooldown --
    "strategy.cooldown_max_losses": "after N consecutive losses, tighten entry requirements",
    "strategy.cooldown_distance_mult": "multiply min_distance by this factor during cooldown",
    "strategy.cooldown_min_score": "raise minimum score to this during cooldown (overrides base min_score)",
    # -- Global directional bias --
    "strategy.global_trend_bias": "suppress counter-trend entries when total return is strongly directional",
    "strategy.global_bias_threshold": "total return above this triggers counter-trend suppression (e.g. 0.10 = 10%)",
    # -- Backtest params --
    "backtest.initial_cash": "starting portfolio value for the simulation",
    "backtest.commission_per_trade": "flat dollar fee per trade leg (entry & exit each)",
    "backtest.commission_pct": "percentage fee per trade leg (e.g. 0.1% of notional)",
    "backtest.commission_mode": "how flat + pct fees combine: additive (sum) or max (whichever is greater)",
    "backtest.slippage_pct": "higher = more realistic fill price slippage",
    "backtest.warmup_bars": "more bars = indicators are fully warmed up before trading begins",
    "backtest.significant_pattern_min_strength": "higher = only the strongest patterns appear in the timeline",
    "backtest.max_warmup_ratio": "higher = allows warmup to consume more of short datasets (0.1–0.9)",
    # -- ATR-adaptive stop --
    "strategy.atr_stop_enabled": "use ATR-based dynamic stop; widens the fixed stop when ATR is larger",
    "strategy.atr_stop_multiplier": "higher = wider stop, fewer stop-outs; stop = max(fixed %, N × ATR / price)",
    "strategy.atr_stop_period": "shorter = reacts to recent volatility faster, longer = smoother ATR",
    # -- Trend confirmation --
    "strategy.trend_confirm_enabled": "require price on correct side of trend EMA before entering",
    "strategy.trend_confirm_period": "shorter = faster trend detection, more trades; longer = stronger filter",
    # -- Regime classification thresholds --
    "regime.trend_ma_period": "MA period used to measure trend consistency and direction",
    "regime.adx_strong_trend": "ADX above this = strong trend signal; higher = stricter",
    "regime.adx_weak": "ADX below this = no meaningful trend; lower = more lenient",
    "regime.trend_consistency_high": "% bars above MA needed for strong directional bias",
    "regime.trend_consistency_low": "% bars above MA below this (or above 100-this on bear side) = ranging",
    "regime.atr_pct_high": "ATR% above this = high volatility regime",
    "regime.atr_pct_low": "ATR% below this = low volatility / calm market",
    "regime.atr_period": "ATR lookback period for volatility measurement",
    "regime.bb_period": "Bollinger Band period for squeeze detection",
    "regime.bb_std_dev": "BB standard deviation multiplier",
    "regime.bb_squeeze_percentile": "BB width below this percentile = squeeze (potential breakout)",
    "regime.bb_expansion_percentile": "BB width above this percentile = expanded volatility",
    "regime.direction_change_high": "fraction of bars reversing above this = choppy market",
    "regime.direction_change_period": "lookback window for counting direction changes",
    "regime.price_ma_distance_extended": "price > this % from MA = extended trend, potential reversion",
    "regime.total_return_strong": "absolute return above this over the period = definitively trending",
    "regime.total_return_moderate": "absolute return above this = moderate trend signal",
    # -- Regime strategy adaptation --
    "regime.strong_trend.use_trailing_stop": "trail stop with trend instead of score-based exits",
    "regime.strong_trend.trailing_stop_atr_mult": "trailing stop distance = N x ATR; higher = wider stop",
    "regime.strong_trend.ignore_score_entries": "skip score thresholds; enter based on trend direction instead",
    "regime.strong_trend.hold_with_trend": "hold position as long as trend persists, ignore sell signals",
    "regime.strong_trend.min_distance": "minimum price distance from trend MA to trigger trend entry (e.g. 0.01 = 1%)",
    "regime.strong_trend.min_score": "minimum effective score required for trend entry",
    "regime.strong_trend.respect_trend_direction": "use total return to determine bias (bullish/bearish) for entries",
    "regime.mean_reverting.tighten_thresholds": "narrow the HOLD zone to generate more swing trades",
    "regime.mean_reverting.threshold_adjustment": "how much to narrow thresholds on each side (0.1–0.5)",
    "regime.volatile_choppy.reduce_position_size": "halve position size to limit risk in choppy markets",
    "regime.volatile_choppy.position_size_mult": "position size multiplier (0.1–1.0); lower = smaller positions",
    "regime.volatile_choppy.widen_stops": "widen stop loss to avoid frequent stop-outs in volatile markets",
    "regime.volatile_choppy.stop_loss_mult": "stop loss multiplier (1.0–3.0); higher = wider stops",
    "regime.breakout_transition.use_momentum_entry": "enter on breakout confirmation instead of score-based entry",
    "regime.breakout_transition.breakout_atr_mult": "price must move N x ATR from squeeze level to confirm breakout",
    "regime.breakout_transition.require_volume_surge": "require above-average volume to confirm breakout",
    "regime.breakout_transition.volume_surge_mult": "volume must exceed average by this multiplier (1.0–3.0)",
    # -- New strategy params (config extraction) --
    "strategy.trend_confirm_ma_type": "ema = exponential (responsive), sma = simple (smoother) for trend filter",
    "strategy.trend_confirm_tolerance_pct": "tolerance band around MA; 0 = price must cross exactly, 0.01 = 1% band",
    "strategy.cooldown_reset_on_breakeven": "if on, a 0% PnL trade resets the consecutive loss counter",
    "strategy.trend_bias_return_threshold": "|total_return| >= this → definitive bullish/bearish bias for entries",
    "strategy.extreme_exit_score_offset": "exit strong-trend position when score is this far beyond thresholds",
    "strategy.breakout_min_move_ratio": "|close-open|/range >= this to qualify as a breakout candle",
    "strategy.allow_pyramiding": "if on, can add to existing same-direction positions",
    "strategy.allow_immediate_reversal": "if on, close + reopen in opposite direction on signal flip",
    "strategy.disable_take_profit_in_strong_trend": "let trailing stop handle exits instead of fixed take-profit",
    "strategy.trailing_stop_require_profit": "trailing stop only activates after position is in profit",
    "strategy.percentile_min_fill_ratio": "min fraction of lookback window filled before percentile mode activates",
    # -- New backtest params (config extraction) --
    "backtest.min_warmup_bars": "absolute floor for proportional warmup calculation",
    "backtest.min_post_warmup_bars": "minimum tradeable bars required after warmup period",
    "backtest.trading_days_per_year": "used for return annualization (252 for US equities)",
    "backtest.trading_day_minutes": "minutes in a trading day for intraday bar counting (390 for US)",
    "backtest.default_score": "neutral starting score used before first rebalance computes actual scores",
    "backtest.close_on_end_of_data": "if on, force-close any open position at end of data",
    # -- New regime params (config extraction) --
    "regime.min_bars_for_classification": "fewer bars than this → skip regime classification entirely",
    "regime.trend_direction_bullish_threshold": "pct_above_ma > this → classify trend direction as bullish",
    "regime.trend_direction_bearish_threshold": "pct_above_ma < this → classify trend direction as bearish",
    "regime.adx_dip_threshold": "rolling ADX mean > current + this → mention 'temporary dip' in reasons",
    "regime.runner_up_proximity_ratio": "runner-up score / winner > this → mention it in classification reasons",
    # -- Suitability params --
    "suitability.mode_override": "auto = detect from data, or force long_short / long_only / hold_only",
    "suitability.min_volume": "avg daily volume below this → stock is too illiquid (hold_only)",
    "suitability.min_atr_pct": "ATR/price below this → price movement too low for active trading (hold_only)",
    "suitability.min_adx_for_short": "ADX below this → trend too weak for effective shorting (long_only)",
    "suitability.min_atr_for_short": "ATR/price below this → volatility too low for shorts (long_only)",
    "suitability.min_volume_for_short": "avg volume below this → insufficient liquidity for shorting (long_only)",
    "suitability.trend_ma_period": "long-term MA period for structural trend direction check",
    "suitability.max_pct_above_ma": "price above MA for more than this % → structural uptrend, suppress shorts",
    "suitability.atr_period": "ATR calculation lookback period for suitability checks",
    "suitability.adx_min_data_mult": "require period × this many bars before ADX can be computed",
    "suitability.insufficient_data_pct": "pct_above_ma fallback when not enough data (50 = no directional bias)",
}


def _get_default(path: str) -> object:
    """Look up a value in DEFAULT_CONFIG by dot-separated path.

    Example: ``_get_default("rsi.thresholds.oversold")`` → 30
    Returns ``None`` if the path doesn't exist.
    """
    node: object = DEFAULT_CONFIG
    for part in path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def _default_hint(value: object, desc: str | None = None) -> None:
    """Render a small gray caption showing the default value and optional description."""
    parts: list[str] = []
    if value is not None:
        if isinstance(value, float):
            text = f"{value:.4g}"
        elif isinstance(value, list):
            text = ", ".join(str(v) for v in value)
        else:
            text = str(value)
        parts.append(f"default: {text}")
    if desc:
        parts.append(desc)
    if parts:
        st.caption(" · ".join(parts))

def _edit_indicator_weights(data: dict) -> None:
    """Editable indicator weight sliders (auto-normalised display)."""
    weights = data.setdefault("overall", {}).setdefault("weights", {})
    names = list(weights.keys())
    default_weights = DEFAULT_CONFIG.get("overall", {}).get("weights", {})

    for name in names:
        raw = float(weights.get(name, 0))
        new_val = st.slider(
            f"{name}",
            min_value=0.0,
            max_value=1.0,
            value=raw,
            step=0.05,
            key=f"iw_{name}",
            format="%.2f",
        )
        weights[name] = new_val
        _default_hint(default_weights.get(name), PARAM_DESCRIPTIONS.get(f"iw_{name}"))

    new_total = sum(weights.values())
    if new_total > 0:
        st.caption(f"Sum: {new_total:.2f} (normalised at runtime)")
    else:
        st.warning("All weights are zero — equal weighting will be used.")


def _edit_composite_scoring(data: dict) -> None:
    """Editable composite scoring params: subgroup mode and score spreading."""
    overall = data.setdefault("overall", {})
    defaults = DEFAULT_CONFIG.get("overall", {})

    # Subgroup mode
    modes = ["directional", "average"]
    current_mode = str(overall.get("subgroup_mode", "directional"))
    idx = modes.index(current_mode) if current_mode in modes else 0
    new_mode = st.selectbox(
        "Subgroup mode", modes, index=idx, key="cs_subgroup_mode",
    )
    overall["subgroup_mode"] = new_mode
    _default_hint("directional", "directional = dominant signal wins; average = legacy flat avg")

    if new_mode == "directional":
        blend = overall.setdefault("subgroup_blend", {})
        def_blend = defaults.get("subgroup_blend", {})

        dom_w = st.slider(
            "Dominant weight", 0.0, 1.0,
            float(blend.get("dominant_weight", 0.6)),
            step=0.05, key="cs_dom_w", format="%.2f",
        )
        blend["dominant_weight"] = dom_w
        _default_hint(def_blend.get("dominant_weight"), "weight for the stronger subgroup")

        oth_w = st.slider(
            "Other weight", 0.0, 1.0,
            float(blend.get("other_weight", 0.25)),
            step=0.05, key="cs_oth_w", format="%.2f",
        )
        blend["other_weight"] = oth_w
        _default_hint(def_blend.get("other_weight"), "weight for the weaker subgroup")

        neu_w = st.slider(
            "Neutral weight", 0.0, 1.0,
            float(blend.get("neutral_weight", 0.15)),
            step=0.05, key="cs_neu_w", format="%.2f",
        )
        blend["neutral_weight"] = neu_w
        _default_hint(def_blend.get("neutral_weight"), "weight for neutral indicators (fibonacci)")

    # Score spreading
    spread = overall.setdefault("score_spreading", {})
    def_spread = defaults.get("score_spreading", {})

    spread_on = st.checkbox(
        "Enable score spreading",
        value=bool(spread.get("enabled", True)),
        key="cs_spread_on",
    )
    spread["enabled"] = spread_on
    _default_hint(True, "rescale scores around 5.0 to expand range")

    if spread_on:
        factor = st.slider(
            "Spread factor", 1.0, 4.0,
            float(spread.get("factor", 2.0)),
            step=0.1, key="cs_spread_f", format="%.1f",
        )
        spread["factor"] = factor
        _default_hint(def_spread.get("factor"), "2.0 expands ~4-6 range to ~2-8")


def _edit_indicator_params(data: dict) -> None:
    """Editable indicator parameter inputs, one expander per indicator."""
    indicator_params = {
        "rsi": [
            ("period", "Period", int, 2, 50),
            ("thresholds.oversold", "Oversold threshold", int, 0, 100),
            ("thresholds.overbought", "Overbought threshold", int, 0, 100),
        ],
        "macd": [
            ("fast_period", "Fast period", int, 2, 50),
            ("slow_period", "Slow period", int, 5, 100),
            ("signal_period", "Signal period", int, 2, 50),
        ],
        "bollinger_bands": [
            ("period", "Period", int, 5, 100),
            ("std_dev", "Std deviation", float, 0.5, 4.0),
        ],
        "moving_averages": [
            ("periods", "Periods (comma-sep)", "int_list", None, None),
        ],
        "stochastic": [
            ("k_period", "K period", int, 2, 50),
            ("d_period", "D period", int, 2, 20),
            ("smooth_k", "Smooth K", int, 1, 10),
            ("thresholds.oversold", "Oversold", int, 0, 100),
            ("thresholds.overbought", "Overbought", int, 0, 100),
        ],
        "adx": [
            ("period", "Period", int, 2, 50),
            ("thresholds.weak", "Weak threshold", int, 0, 100),
            ("thresholds.moderate", "Moderate threshold", int, 0, 100),
        ],
        "volume": [
            ("obv_trend_period", "OBV trend period", int, 5, 100),
            ("price_trend_period", "Price trend period", int, 5, 100),
        ],
        "fibonacci": [
            ("swing_lookback", "Swing lookback", int, 10, 300),
        ],
    }

    for ind_key, params in indicator_params.items():
        section = data.setdefault(ind_key, {})
        with st.expander(ind_key.replace("_", " ").title()):
            for path, label, ptype, pmin, pmax in params:
                parts = path.split(".")
                # Navigate to the right nested dict
                target = section
                for p in parts[:-1]:
                    target = target.setdefault(p, {})
                field = parts[-1]

                # Look up default from DEFAULT_CONFIG
                default_val = _get_default(f"{ind_key}.{path}")
                desc = PARAM_DESCRIPTIONS.get(f"{ind_key}.{path}")

                if ptype == "int_list":
                    # Special: comma-separated integer list
                    current = target.get(field, [20, 50, 200])
                    current_str = ", ".join(str(x) for x in current)
                    new_str = st.text_input(label, value=current_str, key=f"ip_{ind_key}_{path}")
                    _default_hint(default_val, desc)
                    try:
                        parsed = [int(x.strip()) for x in new_str.split(",") if x.strip()]
                        target[field] = sorted(parsed)
                    except ValueError:
                        st.warning("Enter comma-separated integers")
                elif ptype is int:
                    current = int(target.get(field, pmin))
                    new_val = st.number_input(
                        label, min_value=pmin, max_value=pmax,
                        value=current, step=1, key=f"ip_{ind_key}_{path}",
                    )
                    _default_hint(default_val, desc)
                    target[field] = int(new_val)
                elif ptype is float:
                    current = float(target.get(field, pmin))
                    new_val = st.number_input(
                        label, min_value=float(pmin), max_value=float(pmax),
                        value=current, step=0.1, key=f"ip_{ind_key}_{path}",
                        format="%.2f",
                    )
                    _default_hint(default_val, desc)
                    target[field] = float(new_val)


def _edit_pattern_weights(data: dict) -> None:
    """Editable pattern weight sliders."""
    weights = data.setdefault("overall_patterns", {}).setdefault("weights", {})
    names = list(weights.keys())
    default_weights = DEFAULT_CONFIG.get("overall_patterns", {}).get("weights", {})

    for name in names:
        raw = float(weights.get(name, 0))
        new_val = st.slider(
            f"{name}",
            min_value=0.0,
            max_value=1.0,
            value=raw,
            step=0.05,
            key=f"pw_{name}",
            format="%.2f",
        )
        weights[name] = new_val
        _default_hint(default_weights.get(name), PARAM_DESCRIPTIONS.get(f"pw_{name}"))

    new_total = sum(weights.values())
    if new_total > 0:
        st.caption(f"Sum: {new_total:.2f} (normalised at runtime)")
    else:
        st.warning("All weights are zero — equal weighting will be used.")


def _edit_pattern_indicator_combination(data: dict) -> None:
    """Editable combination mode and related params."""
    strat = data.setdefault("strategy", {})
    _ds = DEFAULT_CONFIG.get("strategy", {})

    modes = ["weighted", "gate", "boost"]
    current_mode = strat.get("combination_mode", "weighted")
    idx = modes.index(current_mode) if current_mode in modes else 0
    mode = st.selectbox("Combination mode", modes, index=idx, key="combo_mode")
    strat["combination_mode"] = mode
    _default_hint(_ds.get("combination_mode"), PARAM_DESCRIPTIONS.get("strategy.combination_mode"))

    if mode == "weighted":
        ind_w = st.slider(
            "Indicator weight",
            0.0, 1.0,
            value=float(strat.get("indicator_weight", 0.7)),
            step=0.05, key="combo_ind_w", format="%.2f",
        )
        _default_hint(_ds.get("indicator_weight"), PARAM_DESCRIPTIONS.get("strategy.indicator_weight"))
        pat_w = st.slider(
            "Pattern weight",
            0.0, 1.0,
            value=float(strat.get("pattern_weight", 0.3)),
            step=0.05, key="combo_pat_w", format="%.2f",
        )
        _default_hint(_ds.get("pattern_weight"), PARAM_DESCRIPTIONS.get("strategy.pattern_weight"))
        strat["indicator_weight"] = ind_w
        strat["pattern_weight"] = pat_w
        total = ind_w + pat_w
        if total > 0:
            st.caption(f"Effective: indicator {ind_w/total:.0%} / pattern {pat_w/total:.0%}")
    elif mode == "gate":
        strat["gate_indicator_min"] = st.number_input(
            "Indicator LONG threshold",
            0.0, 10.0, float(strat.get("gate_indicator_min", 5.5)),
            step=0.1, key="gate_ind_min", format="%.1f",
        )
        _default_hint(_ds.get("gate_indicator_min"), PARAM_DESCRIPTIONS.get("strategy.gate_indicator_min"))
        strat["gate_indicator_max"] = st.number_input(
            "Indicator SHORT threshold",
            0.0, 10.0, float(strat.get("gate_indicator_max", 4.5)),
            step=0.1, key="gate_ind_max", format="%.1f",
        )
        _default_hint(_ds.get("gate_indicator_max"), PARAM_DESCRIPTIONS.get("strategy.gate_indicator_max"))
        strat["gate_pattern_min"] = st.number_input(
            "Pattern LONG threshold",
            0.0, 10.0, float(strat.get("gate_pattern_min", 5.5)),
            step=0.1, key="gate_pat_min", format="%.1f",
        )
        _default_hint(_ds.get("gate_pattern_min"), PARAM_DESCRIPTIONS.get("strategy.gate_pattern_min"))
        strat["gate_pattern_max"] = st.number_input(
            "Pattern SHORT threshold",
            0.0, 10.0, float(strat.get("gate_pattern_max", 4.5)),
            step=0.1, key="gate_pat_max", format="%.1f",
        )
        _default_hint(_ds.get("gate_pattern_max"), PARAM_DESCRIPTIONS.get("strategy.gate_pattern_max"))
    elif mode == "boost":
        strat["boost_strength"] = st.slider(
            "Boost strength",
            0.0, 2.0, float(strat.get("boost_strength", 0.5)),
            step=0.1, key="boost_str", format="%.1f",
        )
        _default_hint(_ds.get("boost_strength"), PARAM_DESCRIPTIONS.get("strategy.boost_strength"))
        strat["boost_dead_zone"] = st.slider(
            "Dead zone (+-)",
            0.0, 2.0, float(strat.get("boost_dead_zone", 0.3)),
            step=0.1, key="boost_dz", format="%.1f",
        )
        _default_hint(_ds.get("boost_dead_zone"), PARAM_DESCRIPTIONS.get("strategy.boost_dead_zone"))


def _edit_scoring_thresholds(data: dict) -> None:
    """Editable scoring threshold mode and values."""
    strat = data.setdefault("strategy", {})
    _ds = DEFAULT_CONFIG.get("strategy", {})

    mode = st.radio(
        "Threshold mode",
        ["fixed", "percentile"],
        index=0 if strat.get("threshold_mode", "fixed") == "fixed" else 1,
        horizontal=True,
        key="threshold_mode_radio",
    )
    strat["threshold_mode"] = mode
    _default_hint(_ds.get("threshold_mode"), PARAM_DESCRIPTIONS.get("strategy.threshold_mode"))

    if mode == "fixed":
        _ds_thr = _ds.get("score_thresholds", {})
        thresholds = strat.setdefault("score_thresholds", {})
        short_below = st.slider(
            "SHORT when score <=",
            0.0, 10.0, float(thresholds.get("short_below", 3.5)),
            step=0.1, key="fix_short", format="%.1f",
        )
        _default_hint(_ds_thr.get("short_below"), PARAM_DESCRIPTIONS.get("strategy.score_thresholds.short_below"))
        hold_below = st.slider(
            "LONG when score >",
            0.0, 10.0, float(thresholds.get("hold_below", 6.0)),
            step=0.1, key="fix_long", format="%.1f",
        )
        _default_hint(_ds_thr.get("hold_below"), PARAM_DESCRIPTIONS.get("strategy.score_thresholds.hold_below"))
        thresholds["short_below"] = short_below
        thresholds["hold_below"] = hold_below
        if short_below >= hold_below:
            st.warning("SHORT threshold should be below LONG threshold.")
    else:
        _ds_pct = _ds.get("percentile_thresholds", {})
        pct = strat.setdefault("percentile_thresholds", {})
        pct["short_percentile"] = st.number_input(
            "SHORT percentile <=",
            0, 100, int(pct.get("short_percentile", 25)),
            step=1, key="pct_short",
        )
        _default_hint(_ds_pct.get("short_percentile"), PARAM_DESCRIPTIONS.get("strategy.percentile_thresholds.short_percentile"))
        pct["long_percentile"] = st.number_input(
            "LONG percentile >=",
            0, 100, int(pct.get("long_percentile", 75)),
            step=1, key="pct_long",
        )
        _default_hint(_ds_pct.get("long_percentile"), PARAM_DESCRIPTIONS.get("strategy.percentile_thresholds.long_percentile"))
        pct["lookback_bars"] = st.number_input(
            "Lookback bars",
            10, 500, int(pct.get("lookback_bars", 60)),
            step=5, key="pct_lookback",
        )
        _default_hint(_ds_pct.get("lookback_bars"), PARAM_DESCRIPTIONS.get("strategy.percentile_thresholds.lookback_bars"))


def _edit_strategy_params(data: dict) -> None:
    """Editable strategy execution params."""
    strat = data.setdefault("strategy", {})
    _ds = DEFAULT_CONFIG.get("strategy", {})

    strat["stop_loss_pct"] = st.number_input(
        "Stop loss %",
        0.1, 50.0,
        value=float(strat.get("stop_loss_pct", 0.05)) * 100,
        step=0.5, key="sl_pct", format="%.1f",
    ) / 100.0
    _default_hint(f"{_ds.get('stop_loss_pct', 0.05) * 100:.1f}%", PARAM_DESCRIPTIONS.get("strategy.stop_loss_pct"))

    strat["take_profit_pct"] = st.number_input(
        "Take profit %",
        0.1, 100.0,
        value=float(strat.get("take_profit_pct", 0.20)) * 100,
        step=0.5, key="tp_pct", format="%.1f",
    ) / 100.0
    _default_hint(f"{_ds.get('take_profit_pct', 0.20) * 100:.1f}%", PARAM_DESCRIPTIONS.get("strategy.take_profit_pct"))

    sizing_modes = ["percent_equity", "fixed"]
    current_sizing = strat.get("position_sizing", "percent_equity")
    sizing_idx = sizing_modes.index(current_sizing) if current_sizing in sizing_modes else 0
    strat["position_sizing"] = st.selectbox(
        "Position sizing", sizing_modes, index=sizing_idx, key="pos_sizing",
    )
    _default_hint(_ds.get("position_sizing"), PARAM_DESCRIPTIONS.get("strategy.position_sizing"))

    if strat["position_sizing"] == "percent_equity":
        strat["percent_equity"] = st.slider(
            "Equity % (0-1)",
            0.1, 1.0, float(strat.get("percent_equity", 0.80)),
            step=0.05, key="pct_equity", format="%.2f",
        )
        _default_hint(_ds.get("percent_equity"), PARAM_DESCRIPTIONS.get("strategy.percent_equity"))
    else:
        strat["fixed_quantity"] = st.number_input(
            "Fixed quantity",
            1, 10000, int(strat.get("fixed_quantity", 100)),
            step=10, key="fix_qty",
        )
        _default_hint(_ds.get("fixed_quantity"), PARAM_DESCRIPTIONS.get("strategy.fixed_quantity"))

    strat["rebalance_interval"] = st.number_input(
        "Rebalance interval (bars)",
        1, 100, int(strat.get("rebalance_interval", 5)),
        step=1, key="rebal",
    )
    _default_hint(_ds.get("rebalance_interval"), PARAM_DESCRIPTIONS.get("strategy.rebalance_interval"))

    strat["flatten_eod"] = st.checkbox(
        "Flatten at EOD",
        value=bool(strat.get("flatten_eod", False)),
        key="flatten_eod_cb",
    )
    _default_hint(_ds.get("flatten_eod"), PARAM_DESCRIPTIONS.get("strategy.flatten_eod"))

    st.markdown("##### ATR-Adaptive Stop")
    strat["atr_stop_enabled"] = st.checkbox(
        "Enable ATR stop",
        value=bool(strat.get("atr_stop_enabled", True)),
        key="atr_stop_en",
    )
    _default_hint(_ds.get("atr_stop_enabled"), PARAM_DESCRIPTIONS.get("strategy.atr_stop_enabled"))

    if strat["atr_stop_enabled"]:
        strat["atr_stop_multiplier"] = st.number_input(
            "ATR stop multiplier",
            0.5, 5.0,
            value=float(strat.get("atr_stop_multiplier", 2.5)),
            step=0.1, key="atr_stop_mult", format="%.1f",
        )
        _default_hint(_ds.get("atr_stop_multiplier"), PARAM_DESCRIPTIONS.get("strategy.atr_stop_multiplier"))

        strat["atr_stop_period"] = st.number_input(
            "ATR period",
            5, 50,
            value=int(strat.get("atr_stop_period", 14)),
            step=1, key="atr_stop_per",
        )
        _default_hint(_ds.get("atr_stop_period"), PARAM_DESCRIPTIONS.get("strategy.atr_stop_period"))

    st.markdown("##### Trend Confirmation")
    strat["trend_confirm_enabled"] = st.checkbox(
        "Enable trend filter",
        value=bool(strat.get("trend_confirm_enabled", True)),
        key="trend_conf_en",
    )
    _default_hint(_ds.get("trend_confirm_enabled"), PARAM_DESCRIPTIONS.get("strategy.trend_confirm_enabled"))

    if strat["trend_confirm_enabled"]:
        strat["trend_confirm_period"] = st.number_input(
            "Trend EMA period",
            5, 100,
            value=int(strat.get("trend_confirm_period", 20)),
            step=1, key="trend_conf_per",
        )
        _default_hint(_ds.get("trend_confirm_period"), PARAM_DESCRIPTIONS.get("strategy.trend_confirm_period"))

        strat["reentry_grace_bars"] = st.number_input(
            "Re-entry grace bars",
            0, 50,
            value=int(strat.get("reentry_grace_bars", 10)),
            step=1, key="reentry_grace",
        )
        _default_hint(_ds.get("reentry_grace_bars"), PARAM_DESCRIPTIONS.get("strategy.reentry_grace_bars"))

    st.markdown("##### Consecutive Loss Cooldown")
    strat["cooldown_max_losses"] = st.number_input(
        "Max consecutive losses",
        1, 10,
        value=int(strat.get("cooldown_max_losses", 2)),
        step=1, key="cooldown_max",
    )
    _default_hint(_ds.get("cooldown_max_losses"), PARAM_DESCRIPTIONS.get("strategy.cooldown_max_losses"))

    strat["cooldown_distance_mult"] = st.number_input(
        "Distance multiplier",
        1.0, 5.0,
        value=float(strat.get("cooldown_distance_mult", 2.0)),
        step=0.5, key="cooldown_dist", format="%.1f",
    )
    _default_hint(_ds.get("cooldown_distance_mult"), PARAM_DESCRIPTIONS.get("strategy.cooldown_distance_mult"))

    strat["cooldown_min_score"] = st.number_input(
        "Cooldown min score",
        0.0, 10.0,
        value=float(strat.get("cooldown_min_score", 4.5)),
        step=0.1, key="cooldown_score", format="%.1f",
    )
    _default_hint(_ds.get("cooldown_min_score"), PARAM_DESCRIPTIONS.get("strategy.cooldown_min_score"))

    st.markdown("##### Global Directional Bias")
    strat["global_trend_bias"] = st.checkbox(
        "Enable global trend bias",
        value=bool(strat.get("global_trend_bias", True)),
        key="global_bias_en",
    )
    _default_hint(_ds.get("global_trend_bias"), PARAM_DESCRIPTIONS.get("strategy.global_trend_bias"))

    if strat["global_trend_bias"]:
        strat["global_bias_threshold"] = st.number_input(
            "Bias threshold (total return)",
            0.01, 1.0,
            value=float(strat.get("global_bias_threshold", 0.10)),
            step=0.01, key="global_bias_thr", format="%.2f",
        )
        _default_hint(_ds.get("global_bias_threshold"), PARAM_DESCRIPTIONS.get("strategy.global_bias_threshold"))

    st.markdown("##### Advanced Strategy Tuning")

    # Trend confirmation MA type
    ma_types = ["ema", "sma"]
    current_ma_type = strat.get("trend_confirm_ma_type", "ema")
    ma_idx = ma_types.index(current_ma_type) if current_ma_type in ma_types else 0
    strat["trend_confirm_ma_type"] = st.selectbox(
        "Trend confirm MA type", ma_types, index=ma_idx, key="trend_ma_type",
    )
    _default_hint(_ds.get("trend_confirm_ma_type"), PARAM_DESCRIPTIONS.get("strategy.trend_confirm_ma_type"))

    strat["trend_confirm_tolerance_pct"] = st.number_input(
        "Trend confirm tolerance %",
        0.0, 10.0,
        value=float(strat.get("trend_confirm_tolerance_pct", 0.0)) * 100,
        step=0.5, key="trend_tol_pct", format="%.1f",
    ) / 100.0
    _default_hint(
        f"{_ds.get('trend_confirm_tolerance_pct', 0.0) * 100:.1f}%",
        PARAM_DESCRIPTIONS.get("strategy.trend_confirm_tolerance_pct"),
    )

    strat["trend_bias_return_threshold"] = st.number_input(
        "Trend bias return threshold",
        0.01, 1.0,
        value=float(strat.get("trend_bias_return_threshold", 0.15)),
        step=0.01, key="trend_bias_ret", format="%.2f",
    )
    _default_hint(_ds.get("trend_bias_return_threshold"), PARAM_DESCRIPTIONS.get("strategy.trend_bias_return_threshold"))

    strat["extreme_exit_score_offset"] = st.number_input(
        "Extreme exit score offset",
        0.0, 5.0,
        value=float(strat.get("extreme_exit_score_offset", 1.5)),
        step=0.1, key="extreme_exit_off", format="%.1f",
    )
    _default_hint(_ds.get("extreme_exit_score_offset"), PARAM_DESCRIPTIONS.get("strategy.extreme_exit_score_offset"))

    strat["breakout_min_move_ratio"] = st.slider(
        "Breakout min move ratio",
        0.0, 1.0,
        value=float(strat.get("breakout_min_move_ratio", 0.4)),
        step=0.05, key="brkout_move", format="%.2f",
    )
    _default_hint(_ds.get("breakout_min_move_ratio"), PARAM_DESCRIPTIONS.get("strategy.breakout_min_move_ratio"))

    strat["percentile_min_fill_ratio"] = st.slider(
        "Percentile min fill ratio",
        0.1, 1.0,
        value=float(strat.get("percentile_min_fill_ratio", 0.8)),
        step=0.05, key="pct_fill", format="%.2f",
    )
    _default_hint(_ds.get("percentile_min_fill_ratio"), PARAM_DESCRIPTIONS.get("strategy.percentile_min_fill_ratio"))

    strat["cooldown_reset_on_breakeven"] = st.checkbox(
        "Cooldown reset on breakeven",
        value=bool(strat.get("cooldown_reset_on_breakeven", True)),
        key="cooldown_be",
    )
    _default_hint(_ds.get("cooldown_reset_on_breakeven"), PARAM_DESCRIPTIONS.get("strategy.cooldown_reset_on_breakeven"))

    strat["allow_pyramiding"] = st.checkbox(
        "Allow pyramiding",
        value=bool(strat.get("allow_pyramiding", False)),
        key="allow_pyr",
    )
    _default_hint(_ds.get("allow_pyramiding"), PARAM_DESCRIPTIONS.get("strategy.allow_pyramiding"))

    strat["allow_immediate_reversal"] = st.checkbox(
        "Allow immediate reversal",
        value=bool(strat.get("allow_immediate_reversal", True)),
        key="allow_rev",
    )
    _default_hint(_ds.get("allow_immediate_reversal"), PARAM_DESCRIPTIONS.get("strategy.allow_immediate_reversal"))

    strat["disable_take_profit_in_strong_trend"] = st.checkbox(
        "Disable TP in strong trend",
        value=bool(strat.get("disable_take_profit_in_strong_trend", True)),
        key="disable_tp_st",
    )
    _default_hint(_ds.get("disable_take_profit_in_strong_trend"), PARAM_DESCRIPTIONS.get("strategy.disable_take_profit_in_strong_trend"))

    strat["trailing_stop_require_profit"] = st.checkbox(
        "Trailing stop requires profit",
        value=bool(strat.get("trailing_stop_require_profit", True)),
        key="trail_req_profit",
    )
    _default_hint(_ds.get("trailing_stop_require_profit"), PARAM_DESCRIPTIONS.get("strategy.trailing_stop_require_profit"))


def _edit_backtest_params(data: dict) -> None:
    """Editable backtest engine params."""
    bt = data.setdefault("backtest", {})
    _db = DEFAULT_CONFIG.get("backtest", {})

    bt["initial_cash"] = st.number_input(
        "Initial capital ($)",
        1000.0, 10_000_000.0,
        value=float(bt.get("initial_cash", 100_000)),
        step=10_000.0, key="init_cash", format="%.0f",
    )
    _default_hint(f"${_db.get('initial_cash', 100_000):,.0f}", PARAM_DESCRIPTIONS.get("backtest.initial_cash"))

    bt["commission_per_trade"] = st.number_input(
        "Commission per trade ($)",
        0.0, 100.0,
        value=float(bt.get("commission_per_trade", 0.0)),
        step=1.0, key="commission", format="%.2f",
    )
    _default_hint(_db.get("commission_per_trade"), PARAM_DESCRIPTIONS.get("backtest.commission_per_trade"))

    bt["commission_pct"] = st.number_input(
        "Commission %",
        0.0, 5.0,
        value=float(bt.get("commission_pct", 0.0)) * 100,
        step=0.01, key="commission_pct", format="%.3f",
    ) / 100.0
    _default_hint(f"{_db.get('commission_pct', 0.0) * 100:.3f}%", PARAM_DESCRIPTIONS.get("backtest.commission_pct"))

    bt["commission_mode"] = st.selectbox(
        "Commission mode",
        options=["additive", "max"],
        index=["additive", "max"].index(bt.get("commission_mode", "additive")),
        key="commission_mode",
    )
    _default_hint(_db.get("commission_mode", "additive"), PARAM_DESCRIPTIONS.get("backtest.commission_mode"))

    bt["slippage_pct"] = st.number_input(
        "Slippage %",
        0.0, 5.0,
        value=float(bt.get("slippage_pct", 0.001)) * 100,
        step=0.01, key="slippage", format="%.3f",
    ) / 100.0
    _default_hint(f"{_db.get('slippage_pct', 0.001) * 100:.3f}%", PARAM_DESCRIPTIONS.get("backtest.slippage_pct"))

    bt["warmup_bars"] = st.number_input(
        "Warmup bars",
        10, 1000,
        value=int(bt.get("warmup_bars", 200)),
        step=10, key="warmup",
    )
    _default_hint(_db.get("warmup_bars"), PARAM_DESCRIPTIONS.get("backtest.warmup_bars"))

    bt["significant_pattern_min_strength"] = st.number_input(
        "Sig. pattern min strength",
        0.0, 3.0,
        value=float(bt.get("significant_pattern_min_strength", 0.5)),
        step=0.1, key="sig_min_str", format="%.1f",
    )
    _default_hint(_db.get("significant_pattern_min_strength"), PARAM_DESCRIPTIONS.get("backtest.significant_pattern_min_strength"))

    bt["max_warmup_ratio"] = st.slider(
        "Max warmup ratio",
        0.1, 0.9,
        value=float(bt.get("max_warmup_ratio", 0.5)),
        step=0.05, key="max_warmup_r", format="%.2f",
    )
    _default_hint(_db.get("max_warmup_ratio"), PARAM_DESCRIPTIONS.get("backtest.max_warmup_ratio"))

    st.markdown("##### Advanced Backtest Settings")

    bt["min_warmup_bars"] = st.number_input(
        "Min warmup bars",
        5, 200,
        value=int(bt.get("min_warmup_bars", 20)),
        step=5, key="min_warmup",
    )
    _default_hint(_db.get("min_warmup_bars"), PARAM_DESCRIPTIONS.get("backtest.min_warmup_bars"))

    bt["min_post_warmup_bars"] = st.number_input(
        "Min post-warmup bars",
        1, 100,
        value=int(bt.get("min_post_warmup_bars", 10)),
        step=1, key="min_post_warmup",
    )
    _default_hint(_db.get("min_post_warmup_bars"), PARAM_DESCRIPTIONS.get("backtest.min_post_warmup_bars"))

    bt["trading_days_per_year"] = st.number_input(
        "Trading days per year",
        200, 365,
        value=int(bt.get("trading_days_per_year", 252)),
        step=1, key="trade_days_yr",
    )
    _default_hint(_db.get("trading_days_per_year"), PARAM_DESCRIPTIONS.get("backtest.trading_days_per_year"))

    bt["trading_day_minutes"] = st.number_input(
        "Trading day minutes",
        60, 480,
        value=int(bt.get("trading_day_minutes", 390)),
        step=30, key="trade_day_min",
    )
    _default_hint(_db.get("trading_day_minutes"), PARAM_DESCRIPTIONS.get("backtest.trading_day_minutes"))

    bt["default_score"] = st.number_input(
        "Default score",
        0.0, 10.0,
        value=float(bt.get("default_score", 5.0)),
        step=0.5, key="default_score", format="%.1f",
    )
    _default_hint(_db.get("default_score"), PARAM_DESCRIPTIONS.get("backtest.default_score"))

    bt["close_on_end_of_data"] = st.checkbox(
        "Close on end of data",
        value=bool(bt.get("close_on_end_of_data", True)),
        key="close_eod_cb",
    )
    _default_hint(_db.get("close_on_end_of_data"), PARAM_DESCRIPTIONS.get("backtest.close_on_end_of_data"))


def _scoring_weight_inputs(
    target: dict,
    defaults: dict,
    prefix: str,
    fields: list[tuple[str, str, float, float, float]],
) -> None:
    """Render number_input widgets for a group of regime scoring weights.

    Args:
        target: The mutable scoring sub-dict (e.g. scoring["strong_trend"]).
        defaults: The matching DEFAULT_CONFIG sub-dict for default hints.
        prefix: Unique key prefix for Streamlit widget keys.
        fields: List of (key, label, min, max, step) tuples.
    """
    for key, label, fmin, fmax, fstep in fields:
        default_val = defaults.get(key, fmin)
        current = float(target.get(key, default_val))
        target[key] = st.number_input(
            label,
            min_value=fmin, max_value=fmax,
            value=current,
            step=fstep, key=f"{prefix}_{key}", format="%.2f",
        )
        _default_hint(default_val)


def _edit_regime_params(data: dict) -> None:
    """Edit regime classification thresholds and strategy adaptation params."""
    regime = data.setdefault("regime", {})
    _dr = DEFAULT_CONFIG.get("regime", {})

    st.markdown("**Classification Thresholds**")

    regime["trend_ma_period"] = st.number_input(
        "Trend MA period", min_value=5, max_value=200,
        value=int(regime.get("trend_ma_period", 50)),
        step=5, key="reg_trend_ma",
    )
    _default_hint(_dr.get("trend_ma_period"), PARAM_DESCRIPTIONS.get("regime.trend_ma_period"))

    regime["adx_strong_trend"] = st.slider(
        "ADX strong trend", 15.0, 50.0,
        value=float(regime.get("adx_strong_trend", 30.0)),
        step=1.0, key="reg_adx_strong",
    )
    _default_hint(_dr.get("adx_strong_trend"), PARAM_DESCRIPTIONS.get("regime.adx_strong_trend"))

    regime["adx_weak"] = st.slider(
        "ADX weak", 5.0, 35.0,
        value=float(regime.get("adx_weak", 20.0)),
        step=1.0, key="reg_adx_weak",
    )
    _default_hint(_dr.get("adx_weak"), PARAM_DESCRIPTIONS.get("regime.adx_weak"))

    regime["trend_consistency_high"] = st.slider(
        "Trend consistency high %", 50.0, 95.0,
        value=float(regime.get("trend_consistency_high", 70.0)),
        step=1.0, key="reg_tc_high",
    )
    _default_hint(_dr.get("trend_consistency_high"), PARAM_DESCRIPTIONS.get("regime.trend_consistency_high"))

    regime["trend_consistency_low"] = st.slider(
        "Trend consistency low %", 20.0, 60.0,
        value=float(regime.get("trend_consistency_low", 40.0)),
        step=1.0, key="reg_tc_low",
    )
    _default_hint(_dr.get("trend_consistency_low"), PARAM_DESCRIPTIONS.get("regime.trend_consistency_low"))

    regime["atr_pct_high"] = st.slider(
        "ATR% high", 0.01, 0.10,
        value=float(regime.get("atr_pct_high", 0.03)),
        step=0.005, key="reg_atr_high", format="%.3f",
    )
    _default_hint(_dr.get("atr_pct_high"), PARAM_DESCRIPTIONS.get("regime.atr_pct_high"))

    regime["atr_pct_low"] = st.slider(
        "ATR% low", 0.001, 0.05,
        value=float(regime.get("atr_pct_low", 0.01)),
        step=0.001, key="reg_atr_low", format="%.3f",
    )
    _default_hint(_dr.get("atr_pct_low"), PARAM_DESCRIPTIONS.get("regime.atr_pct_low"))

    regime["atr_period"] = st.number_input(
        "ATR period", min_value=5, max_value=50,
        value=int(regime.get("atr_period", 14)),
        step=1, key="reg_atr_period",
    )
    _default_hint(_dr.get("atr_period"), PARAM_DESCRIPTIONS.get("regime.atr_period"))

    regime["bb_period"] = st.number_input(
        "BB period", min_value=5, max_value=50,
        value=int(regime.get("bb_period", 20)),
        step=5, key="reg_bb_period",
    )
    _default_hint(_dr.get("bb_period"), PARAM_DESCRIPTIONS.get("regime.bb_period"))

    regime["bb_std_dev"] = st.slider(
        "BB std dev", 1.0, 4.0,
        value=float(regime.get("bb_std_dev", 2.0)),
        step=0.25, key="reg_bb_std", format="%.2f",
    )
    _default_hint(_dr.get("bb_std_dev"), PARAM_DESCRIPTIONS.get("regime.bb_std_dev"))

    regime["bb_squeeze_percentile"] = st.slider(
        "BB squeeze percentile", 5.0, 40.0,
        value=float(regime.get("bb_squeeze_percentile", 20.0)),
        step=1.0, key="reg_bb_squeeze",
    )
    _default_hint(_dr.get("bb_squeeze_percentile"), PARAM_DESCRIPTIONS.get("regime.bb_squeeze_percentile"))

    regime["bb_expansion_percentile"] = st.slider(
        "BB expansion percentile", 60.0, 95.0,
        value=float(regime.get("bb_expansion_percentile", 80.0)),
        step=1.0, key="reg_bb_expand",
    )
    _default_hint(_dr.get("bb_expansion_percentile"), PARAM_DESCRIPTIONS.get("regime.bb_expansion_percentile"))

    regime["direction_change_high"] = st.slider(
        "Direction change high", 0.30, 0.80,
        value=float(regime.get("direction_change_high", 0.55)),
        step=0.05, key="reg_dir_high", format="%.2f",
    )
    _default_hint(_dr.get("direction_change_high"), PARAM_DESCRIPTIONS.get("regime.direction_change_high"))

    regime["direction_change_period"] = st.number_input(
        "Direction change period", min_value=5, max_value=50,
        value=int(regime.get("direction_change_period", 20)),
        step=5, key="reg_dir_period",
    )
    _default_hint(_dr.get("direction_change_period"), PARAM_DESCRIPTIONS.get("regime.direction_change_period"))

    regime["price_ma_distance_extended"] = st.slider(
        "Price-MA distance extended", 0.02, 0.30,
        value=float(regime.get("price_ma_distance_extended", 0.10)),
        step=0.01, key="reg_pma_ext", format="%.2f",
    )
    _default_hint(_dr.get("price_ma_distance_extended"), PARAM_DESCRIPTIONS.get("regime.price_ma_distance_extended"))

    regime["total_return_strong"] = st.slider(
        "Total return strong", 0.10, 1.00,
        value=float(regime.get("total_return_strong", 0.30)),
        step=0.05, key="reg_ret_strong", format="%.2f",
    )
    _default_hint(_dr.get("total_return_strong"), PARAM_DESCRIPTIONS.get("regime.total_return_strong"))

    regime["total_return_moderate"] = st.slider(
        "Total return moderate", 0.05, 0.50,
        value=float(regime.get("total_return_moderate", 0.15)),
        step=0.05, key="reg_ret_moderate", format="%.2f",
    )
    _default_hint(_dr.get("total_return_moderate"), PARAM_DESCRIPTIONS.get("regime.total_return_moderate"))

    st.markdown("**Advanced Classification**")

    regime["min_bars_for_classification"] = st.number_input(
        "Min bars for classification",
        5, 200,
        value=int(regime.get("min_bars_for_classification", 20)),
        step=5, key="reg_min_bars",
    )
    _default_hint(_dr.get("min_bars_for_classification"), PARAM_DESCRIPTIONS.get("regime.min_bars_for_classification"))

    regime["trend_direction_bullish_threshold"] = st.slider(
        "Trend dir. bullish threshold", 50.0, 90.0,
        value=float(regime.get("trend_direction_bullish_threshold", 60)),
        step=1.0, key="reg_dir_bull",
    )
    _default_hint(_dr.get("trend_direction_bullish_threshold"), PARAM_DESCRIPTIONS.get("regime.trend_direction_bullish_threshold"))

    regime["trend_direction_bearish_threshold"] = st.slider(
        "Trend dir. bearish threshold", 10.0, 50.0,
        value=float(regime.get("trend_direction_bearish_threshold", 40)),
        step=1.0, key="reg_dir_bear",
    )
    _default_hint(_dr.get("trend_direction_bearish_threshold"), PARAM_DESCRIPTIONS.get("regime.trend_direction_bearish_threshold"))

    regime["adx_dip_threshold"] = st.number_input(
        "ADX dip threshold",
        0, 20,
        value=int(regime.get("adx_dip_threshold", 3)),
        step=1, key="reg_adx_dip",
    )
    _default_hint(_dr.get("adx_dip_threshold"), PARAM_DESCRIPTIONS.get("regime.adx_dip_threshold"))

    regime["runner_up_proximity_ratio"] = st.slider(
        "Runner-up proximity ratio", 0.0, 1.0,
        value=float(regime.get("runner_up_proximity_ratio", 0.7)),
        step=0.05, key="reg_runnerup", format="%.2f",
    )
    _default_hint(_dr.get("runner_up_proximity_ratio"), PARAM_DESCRIPTIONS.get("regime.runner_up_proximity_ratio"))

    # -- Sub-type classification thresholds --
    st.markdown("---")
    st.markdown("**Sub-Type Classification (Volatility x Momentum)**")
    sub_type_cfg = regime.setdefault("sub_type", {})
    _dst_cfg = _dr.get("sub_type", {})

    sub_type_cfg["atr_pct_threshold"] = st.number_input(
        "ATR% threshold (high vol)",
        0.005, 0.10,
        value=float(sub_type_cfg.get("atr_pct_threshold", 0.02)),
        step=0.005, key="reg_sub_atr", format="%.3f",
    )
    _default_hint(_dst_cfg.get("atr_pct_threshold"), "ATR% >= this classifies as high volatility")

    sub_type_cfg["momentum_threshold"] = st.number_input(
        "Momentum threshold (high mom)",
        0.05, 1.0,
        value=float(sub_type_cfg.get("momentum_threshold", 0.20)),
        step=0.05, key="reg_sub_mom", format="%.2f",
    )
    _default_hint(_dst_cfg.get("momentum_threshold"), "|total_return| >= this classifies as high momentum")

    # -- Regime scoring weights --
    st.markdown("---")
    st.markdown("**Regime Scoring Weights**")
    scoring = regime.setdefault("scoring", {})
    _dscoring = _dr.get("scoring", {})

    with st.expander("Strong Trend Scoring"):
        st_sc = scoring.setdefault("strong_trend", {})
        _dst_sc = _dscoring.get("strong_trend", {})
        _scoring_weight_inputs(st_sc, _dst_sc, "reg_st_sc", [
            ("return_strong_base", "Return strong base", 0.0, 10.0, 0.5),
            ("return_strong_cap", "Return strong cap", 0.0, 2.0, 0.1),
            ("return_strong_scale", "Return strong scale", 0.0, 10.0, 0.5),
            ("return_moderate_base", "Return moderate base", 0.0, 10.0, 0.5),
            ("return_moderate_scale", "Return moderate scale", 0.0, 10.0, 0.5),
            ("adx_strong_base", "ADX strong base", 0.0, 10.0, 0.5),
            ("adx_strong_divisor", "ADX strong divisor", 1.0, 50.0, 1.0),
            ("adx_moderate_score", "ADX moderate score", 0.0, 5.0, 0.1),
            ("consistency_high_score", "Consistency high score", 0.0, 5.0, 0.5),
            ("consistency_moderate_score", "Consistency moderate score", 0.0, 5.0, 0.1),
            ("extended_distance_score", "Extended distance score", 0.0, 5.0, 0.5),
            ("direction_change_low", "Dir. change low", 0.0, 1.0, 0.05),
            ("direction_change_low_score", "Dir. change low score", 0.0, 5.0, 0.5),
            ("direction_change_mid", "Dir. change mid", 0.0, 1.0, 0.05),
            ("direction_change_mid_score", "Dir. change mid score", 0.0, 5.0, 0.1),
        ])

    with st.expander("Mean Reverting Scoring"):
        mr_sc = scoring.setdefault("mean_reverting", {})
        _dmr_sc = _dscoring.get("mean_reverting", {})
        _scoring_weight_inputs(mr_sc, _dmr_sc, "reg_mr_sc", [
            ("return_strong_penalty", "Return strong penalty", 0.0, 10.0, 0.5),
            ("return_moderate_penalty", "Return moderate penalty", 0.0, 10.0, 0.5),
            ("return_small_bonus", "Return small bonus", 0.0, 5.0, 0.5),
            ("adx_low_score", "ADX low score", 0.0, 5.0, 0.5),
            ("adx_moderate_score", "ADX moderate score", 0.0, 5.0, 0.5),
            ("pct_away_tight", "Pct away tight", 0.0, 50.0, 1.0),
            ("pct_away_tight_score", "Pct away tight score", 0.0, 5.0, 0.5),
            ("pct_away_moderate", "Pct away moderate", 0.0, 50.0, 1.0),
            ("pct_away_moderate_score", "Pct away moderate score", 0.0, 5.0, 0.5),
            ("atr_below_high_score", "ATR below high score", 0.0, 5.0, 0.1),
            ("atr_below_low_score", "ATR below low score", 0.0, 5.0, 0.1),
            ("price_ma_close_threshold", "Price-MA close threshold", 0.0, 0.20, 0.01),
            ("price_ma_close_score", "Price-MA close score", 0.0, 5.0, 0.5),
        ])

    with st.expander("Volatile/Choppy Scoring"):
        vc_sc = scoring.setdefault("volatile_choppy", {})
        _dvc_sc = _dscoring.get("volatile_choppy", {})
        _scoring_weight_inputs(vc_sc, _dvc_sc, "reg_vc_sc", [
            ("atr_high_base", "ATR high base", 0.0, 10.0, 0.5),
            ("atr_high_scale", "ATR high scale", 1.0, 100.0, 5.0),
            ("atr_moderate_score", "ATR moderate score", 0.0, 5.0, 0.1),
            ("direction_change_high_score", "Dir. change high score", 0.0, 5.0, 0.5),
            ("direction_change_moderate", "Dir. change moderate", 0.0, 1.0, 0.05),
            ("direction_change_moderate_score", "Dir. change moderate score", 0.0, 5.0, 0.5),
            ("low_adx_small_return_score", "Low ADX small return score", 0.0, 5.0, 0.1),
            ("wide_bb_score", "Wide BB score", 0.0, 5.0, 0.5),
        ])

    with st.expander("Breakout Scoring"):
        bo_sc = scoring.setdefault("breakout", {})
        _dbo_sc = _dscoring.get("breakout", {})
        _scoring_weight_inputs(bo_sc, _dbo_sc, "reg_bo_sc", [
            ("bb_squeeze_score", "BB squeeze score", 0.0, 10.0, 0.5),
            ("adx_moderate_score", "ADX moderate score", 0.0, 5.0, 0.5),
            ("direction_change_threshold", "Dir. change threshold", 0.0, 1.0, 0.05),
            ("low_atr_high_changes_score", "Low ATR high changes score", 0.0, 5.0, 0.5),
            ("price_ma_close_threshold", "Price-MA close threshold", 0.0, 0.20, 0.01),
            ("bb_consolidation_percentile", "BB consol. percentile", 0.0, 100.0, 5.0),
            ("consolidation_score", "Consolidation score", 0.0, 5.0, 0.1),
        ])

    # -- Strategy adaptation per regime --
    st.markdown("---")
    st.markdown("**Strategy Adaptation**")

    adapt = regime.setdefault("strategy_adaptation", {})
    _da = _dr.get("strategy_adaptation", {})

    # Strong Trend
    st.markdown("_Strong Trend_")
    st_adapt = adapt.setdefault("strong_trend", {})
    _dst = _da.get("strong_trend", {})

    st_adapt["use_trailing_stop"] = st.checkbox(
        "Use trailing stop", value=st_adapt.get("use_trailing_stop", True),
        key="reg_st_trail",
    )
    _default_hint(_dst.get("use_trailing_stop"), PARAM_DESCRIPTIONS.get("regime.strong_trend.use_trailing_stop"))

    st_adapt["trailing_stop_atr_mult"] = st.slider(
        "Trailing stop ATR mult", 1.0, 6.0,
        value=float(st_adapt.get("trailing_stop_atr_mult", 3.0)),
        step=0.5, key="reg_st_trail_mult", format="%.1f",
    )
    _default_hint(_dst.get("trailing_stop_atr_mult"), PARAM_DESCRIPTIONS.get("regime.strong_trend.trailing_stop_atr_mult"))

    st_adapt["ignore_score_entries"] = st.checkbox(
        "Ignore score entries", value=st_adapt.get("ignore_score_entries", True),
        key="reg_st_ignore_score",
    )
    _default_hint(_dst.get("ignore_score_entries"), PARAM_DESCRIPTIONS.get("regime.strong_trend.ignore_score_entries"))

    st_adapt["hold_with_trend"] = st.checkbox(
        "Hold with trend", value=st_adapt.get("hold_with_trend", True),
        key="reg_st_hold",
    )
    _default_hint(_dst.get("hold_with_trend"), PARAM_DESCRIPTIONS.get("regime.strong_trend.hold_with_trend"))

    st_adapt["min_distance"] = st.number_input(
        "Min distance from MA",
        0.001, 0.10,
        value=float(st_adapt.get("min_distance", 0.01)),
        step=0.005, key="reg_st_min_dist", format="%.3f",
    )
    _default_hint(_dst.get("min_distance"), PARAM_DESCRIPTIONS.get("regime.strong_trend.min_distance"))

    st_adapt["min_score"] = st.number_input(
        "Min effective score",
        0.0, 10.0,
        value=float(st_adapt.get("min_score", 3.5)),
        step=0.1, key="reg_st_min_score", format="%.1f",
    )
    _default_hint(_dst.get("min_score"), PARAM_DESCRIPTIONS.get("regime.strong_trend.min_score"))

    st_adapt["respect_trend_direction"] = st.checkbox(
        "Respect trend direction",
        value=bool(st_adapt.get("respect_trend_direction", True)),
        key="reg_st_respect_dir",
    )
    _default_hint(_dst.get("respect_trend_direction"), PARAM_DESCRIPTIONS.get("regime.strong_trend.respect_trend_direction"))

    # Sub-type overrides for Strong Trend
    with st.expander("Sub-Type Overrides (Strong Trend)"):
        st_subs = st_adapt.setdefault("sub_types", {})

        st.caption("Sub-types classify stocks by Volatility × Momentum for display purposes. "
                   "Strategy overrides are currently empty — tested entry-only overrides "
                   "(min_distance, min_score) but they were either no-ops or caused regressions. "
                   "Base strong_trend params already handle all sub-types well. "
                   "You can add custom overrides below; they merge on top of base params.")

        for sub_name, sub_desc in [
            ("explosive_mover", "High vol + high momentum (TSLA, NVDA, MARA)"),
            ("steady_compounder", "Low vol + high momentum (KO, JPM, COST, AAPL, XOM)"),
            ("volatile_directionless", "High vol + low momentum (RIOT, AMD)"),
            ("stagnant", "Low vol + low momentum"),
        ]:
            st.markdown(f"_{sub_name.replace('_', ' ').title()}_ — {sub_desc}")
            st_subs.setdefault(sub_name, {})
            st.caption("No overrides (uses base strong_trend params).")

    # Mean Reverting
    st.markdown("_Mean Reverting_")
    mr_adapt = adapt.setdefault("mean_reverting", {})
    _dmr = _da.get("mean_reverting", {})

    mr_adapt["tighten_thresholds"] = st.checkbox(
        "Tighten thresholds", value=mr_adapt.get("tighten_thresholds", True),
        key="reg_mr_tighten",
    )
    _default_hint(_dmr.get("tighten_thresholds"), PARAM_DESCRIPTIONS.get("regime.mean_reverting.tighten_thresholds"))

    mr_adapt["threshold_adjustment"] = st.slider(
        "Threshold adjustment", 0.1, 1.0,
        value=float(mr_adapt.get("threshold_adjustment", 0.3)),
        step=0.05, key="reg_mr_adj", format="%.2f",
    )
    _default_hint(_dmr.get("threshold_adjustment"), PARAM_DESCRIPTIONS.get("regime.mean_reverting.threshold_adjustment"))

    # Volatile / Choppy
    st.markdown("_Volatile / Choppy_")
    vc_adapt = adapt.setdefault("volatile_choppy", {})
    _dvc = _da.get("volatile_choppy", {})

    vc_adapt["reduce_position_size"] = st.checkbox(
        "Reduce position size", value=vc_adapt.get("reduce_position_size", True),
        key="reg_vc_reduce",
    )
    _default_hint(_dvc.get("reduce_position_size"), PARAM_DESCRIPTIONS.get("regime.volatile_choppy.reduce_position_size"))

    vc_adapt["position_size_mult"] = st.slider(
        "Position size multiplier", 0.1, 1.0,
        value=float(vc_adapt.get("position_size_mult", 0.5)),
        step=0.05, key="reg_vc_size", format="%.2f",
    )
    _default_hint(_dvc.get("position_size_mult"), PARAM_DESCRIPTIONS.get("regime.volatile_choppy.position_size_mult"))

    vc_adapt["widen_stops"] = st.checkbox(
        "Widen stops", value=vc_adapt.get("widen_stops", True),
        key="reg_vc_widen",
    )
    _default_hint(_dvc.get("widen_stops"), PARAM_DESCRIPTIONS.get("regime.volatile_choppy.widen_stops"))

    vc_adapt["stop_loss_mult"] = st.slider(
        "Stop loss multiplier", 1.0, 3.0,
        value=float(vc_adapt.get("stop_loss_mult", 1.5)),
        step=0.1, key="reg_vc_stop", format="%.1f",
    )
    _default_hint(_dvc.get("stop_loss_mult"), PARAM_DESCRIPTIONS.get("regime.volatile_choppy.stop_loss_mult"))

    # Breakout / Transition
    st.markdown("_Breakout / Transition_")
    bo_adapt = adapt.setdefault("breakout_transition", {})
    _dbo = _da.get("breakout_transition", {})

    bo_adapt["use_momentum_entry"] = st.checkbox(
        "Use momentum entry", value=bo_adapt.get("use_momentum_entry", True),
        key="reg_bo_momentum",
    )
    _default_hint(_dbo.get("use_momentum_entry"), PARAM_DESCRIPTIONS.get("regime.breakout_transition.use_momentum_entry"))

    bo_adapt["breakout_atr_mult"] = st.slider(
        "Breakout ATR mult", 0.5, 3.0,
        value=float(bo_adapt.get("breakout_atr_mult", 1.5)),
        step=0.25, key="reg_bo_atr", format="%.2f",
    )
    _default_hint(_dbo.get("breakout_atr_mult"), PARAM_DESCRIPTIONS.get("regime.breakout_transition.breakout_atr_mult"))

    bo_adapt["require_volume_surge"] = st.checkbox(
        "Require volume surge", value=bo_adapt.get("require_volume_surge", True),
        key="reg_bo_vol",
    )
    _default_hint(_dbo.get("require_volume_surge"), PARAM_DESCRIPTIONS.get("regime.breakout_transition.require_volume_surge"))

    bo_adapt["volume_surge_mult"] = st.slider(
        "Volume surge multiplier", 1.0, 3.0,
        value=float(bo_adapt.get("volume_surge_mult", 1.3)),
        step=0.1, key="reg_bo_vol_mult", format="%.1f",
    )
    _default_hint(_dbo.get("volume_surge_mult"), PARAM_DESCRIPTIONS.get("regime.breakout_transition.volume_surge_mult"))


def _edit_suitability_params(data: dict) -> None:
    """Edit suitability (trading mode detection) thresholds."""
    suit = data.setdefault("suitability", {})
    _ds = DEFAULT_CONFIG.get("suitability", {})

    suit["mode_override"] = st.selectbox(
        "Mode override",
        ["auto", "long_short", "long_only", "hold_only"],
        index=["auto", "long_short", "long_only", "hold_only"].index(
            suit.get("mode_override", "auto"),
        ),
        key="suit_mode",
    )
    _default_hint(_ds.get("mode_override"), PARAM_DESCRIPTIONS.get("suitability.mode_override"))

    st.markdown("**Hold-Only Thresholds**")

    suit["min_volume"] = st.number_input(
        "Min avg daily volume",
        min_value=0, max_value=10_000_000,
        value=int(suit.get("min_volume", 100_000)),
        step=10_000, key="suit_min_vol",
    )
    _default_hint(_ds.get("min_volume"), PARAM_DESCRIPTIONS.get("suitability.min_volume"))

    suit["min_atr_pct"] = st.number_input(
        "Min ATR%",
        min_value=0.0, max_value=0.10,
        value=float(suit.get("min_atr_pct", 0.005)),
        step=0.001, key="suit_min_atr", format="%.3f",
    )
    _default_hint(_ds.get("min_atr_pct"), PARAM_DESCRIPTIONS.get("suitability.min_atr_pct"))

    st.markdown("**Long-Only Thresholds**")

    suit["min_adx_for_short"] = st.slider(
        "Min ADX for shorting", 5.0, 50.0,
        value=float(suit.get("min_adx_for_short", 25.0)),
        step=1.0, key="suit_min_adx",
    )
    _default_hint(_ds.get("min_adx_for_short"), PARAM_DESCRIPTIONS.get("suitability.min_adx_for_short"))

    suit["min_atr_for_short"] = st.number_input(
        "Min ATR% for shorting",
        min_value=0.0, max_value=0.10,
        value=float(suit.get("min_atr_for_short", 0.01)),
        step=0.001, key="suit_min_atr_short", format="%.3f",
    )
    _default_hint(_ds.get("min_atr_for_short"), PARAM_DESCRIPTIONS.get("suitability.min_atr_for_short"))

    suit["min_volume_for_short"] = st.number_input(
        "Min volume for shorting",
        min_value=0, max_value=10_000_000,
        value=int(suit.get("min_volume_for_short", 500_000)),
        step=50_000, key="suit_min_vol_short",
    )
    _default_hint(_ds.get("min_volume_for_short"), PARAM_DESCRIPTIONS.get("suitability.min_volume_for_short"))

    st.markdown("**Trend Direction Filter**")

    suit["trend_ma_period"] = st.number_input(
        "Trend MA period",
        min_value=20, max_value=500,
        value=int(suit.get("trend_ma_period", 200)),
        step=10, key="suit_trend_ma",
    )
    _default_hint(_ds.get("trend_ma_period"), PARAM_DESCRIPTIONS.get("suitability.trend_ma_period"))

    suit["max_pct_above_ma"] = st.slider(
        "Max % above MA", 50.0, 95.0,
        value=float(suit.get("max_pct_above_ma", 65.0)),
        step=1.0, key="suit_max_pct",
    )
    _default_hint(_ds.get("max_pct_above_ma"), PARAM_DESCRIPTIONS.get("suitability.max_pct_above_ma"))

    suit["atr_period"] = st.number_input(
        "ATR period",
        min_value=5, max_value=50,
        value=int(suit.get("atr_period", 14)),
        step=1, key="suit_atr_period",
    )
    _default_hint(_ds.get("atr_period"), PARAM_DESCRIPTIONS.get("suitability.atr_period"))

    st.markdown("**Advanced**")

    suit["adx_min_data_mult"] = st.number_input(
        "ADX min data multiplier",
        min_value=1, max_value=10,
        value=int(suit.get("adx_min_data_mult", 3)),
        step=1, key="suit_adx_mult",
    )
    _default_hint(_ds.get("adx_min_data_mult"), PARAM_DESCRIPTIONS.get("suitability.adx_min_data_mult"))

    suit["insufficient_data_pct"] = st.number_input(
        "Insufficient data fallback %",
        min_value=0.0, max_value=100.0,
        value=float(suit.get("insufficient_data_pct", 50.0)),
        step=5.0, key="suit_insuf_pct", format="%.1f",
    )
    _default_hint(_ds.get("insufficient_data_pct"), PARAM_DESCRIPTIONS.get("suitability.insufficient_data_pct"))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Constants — classification periods
# ---------------------------------------------------------------------------

CLASSIFICATION_PERIODS = ["1y", "2y", "5y", "max"]
DEFAULT_CLASSIFICATION_PERIOD = "2y"


# ---------------------------------------------------------------------------
# Sidebar (minimal — just ticker + classification period + loadouts)
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    """Render the simplified sidebar: ticker, classification period, loadouts."""
    _init_config_data()

    st.sidebar.title("Stock Analyzer")
    st.sidebar.markdown("---")

    ticker = st.sidebar.text_input("Ticker", value="AAPL").upper().strip()

    st.sidebar.markdown("#### Analysis Period")
    period_idx = CLASSIFICATION_PERIODS.index(DEFAULT_CLASSIFICATION_PERIOD)
    period = st.sidebar.selectbox(
        "Period",
        CLASSIFICATION_PERIODS,
        index=period_idx,
        help="Data window for regime classification and quick backtest. 2y is the tuned default.",
    )
    st.sidebar.caption("Used for regime classification and quick backtest.")

    # ------------------------------------------------------------------
    # Loadout save / load
    # ------------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Config Loadouts")

    for slot in range(1, LOADOUT_SLOTS + 1):
        exists = _loadout_exists(slot)
        c_load, c_save = st.sidebar.columns(2)
        with c_load:
            if st.button(
                f"Load #{slot}" if exists else f"#{slot} (empty)",
                key=f"load_{slot}",
                disabled=not exists,
                width="stretch",
            ):
                if _load_loadout(slot):
                    st.toast(f"Loaded config slot #{slot}")
                    st.rerun()
        with c_save:
            if st.button(f"Save #{slot}", key=f"save_{slot}", width="stretch"):
                _save_loadout(slot)
                st.toast(f"Saved to config slot #{slot}")
                st.rerun()

    return {
        "ticker": ticker,
        "period": period,
    }

@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def load_analysis(
    ticker: str,
    period: str | None,
    interval: str,
    start: str | None,
    end: str | None,
    config_hash: str,
    config_data: dict,
) -> tuple[AnalysisResult, dict]:
    """Run the full analysis pipeline and return (result, cfg_data)."""
    cfg = Config.from_dict(config_data)

    provider = YahooFinanceProvider()
    analyzer = Analyzer(cfg, provider)
    result = analyzer.run(
        ticker,
        period=period if not start else None,
        interval=interval,
        start=start,
        end=end,
    )
    return result, config_data


@st.cache_data(ttl=300, show_spinner="Running backtest...")
def load_backtest(
    ticker: str,
    period: str | None,
    interval: str,
    start: str | None,
    end: str | None,
    trading_mode_str: str,
    config_hash: str,
    config_data: dict,
) -> tuple[BacktestResult, str, "dict | None", dict]:
    """Run the backtest engine and return results.

    Returns (bt_result, trading_mode_value, assessment_dict_or_None, cfg_data).
    We return serialisable types so Streamlit can cache them.
    """
    cfg = Config.from_dict(config_data)

    provider = YahooFinanceProvider()

    # Determine trading mode
    forced = trading_mode_str != "auto"
    assessment_dict = None

    if forced:
        mode_map = {
            "long_short": TradingMode.LONG_SHORT,
            "long_only": TradingMode.LONG_ONLY,
            "hold_only": TradingMode.HOLD_ONLY,
        }
        trading_mode = mode_map[trading_mode_str]
    elif is_intraday(interval):
        trading_mode = TradingMode.LONG_SHORT
    else:
        df = provider.fetch(
            ticker,
            period=period if not start else None,
            interval=interval,
            start=start,
            end=end,
        )
        suit_analyzer = SuitabilityAnalyzer(cfg)
        assessment = suit_analyzer.assess(df)
        trading_mode = assessment.mode
        # Serialise assessment for cache-friendliness
        assessment_dict = {
            "mode": assessment.mode.value,
            "avg_daily_volume": assessment.avg_daily_volume,
            "adx_value": assessment.adx_value,
            "atr_pct": assessment.atr_pct,
            "reasons": assessment.reasons,
        }

    regime_adapt = cfg.section("regime").get("strategy_adaptation", {})
    strategy = ScoreBasedStrategy(
        params=cfg.section("strategy"),
        trading_mode=trading_mode,
        regime_adaptation=regime_adapt,
    )
    engine = BacktestEngine(
        data_provider=provider,
        strategy=strategy,
        cfg=cfg,
        trading_mode=trading_mode,
    )
    bt_result = engine.run(
        ticker,
        period=period if not start else None,
        interval=interval,
        start=start,
        end=end,
    )
    return bt_result, trading_mode.value, assessment_dict, config_data


# ---------------------------------------------------------------------------
# Score timeseries computation
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner="Computing score timeseries...")
def compute_score_timeseries(
    ticker: str,
    period: str | None,
    interval: str,
    start: str | None,
    end: str | None,
    config_hash: str,
    config_data: dict,
    step: int = 5,
) -> pd.DataFrame:
    """Compute indicator and pattern composite scores at regular intervals.

    Returns a DataFrame indexed by date with columns:
      indicator_score, pattern_score
    """
    cfg = Config.from_dict(config_data)

    provider = YahooFinanceProvider()
    df = provider.fetch(
        ticker,
        period=period if not start else None,
        interval=interval,
        start=start,
        end=end,
    )

    warmup = int(cfg.section("backtest").get("warmup_bars", 50))
    warmup_min = int(cfg.section("backtest").get("min_warmup_bars", 20))
    if len(df) <= warmup_min:
        # Not enough data to compute meaningful scores
        return pd.DataFrame(columns=["indicator_score", "pattern_score"])
    warmup = min(warmup, len(df) - 10)  # ensure we have some data after warmup
    warmup = max(warmup, warmup_min)

    step = max(1, step)  # ensure step is at least 1

    dates = []
    ind_scores = []
    pat_scores = []

    for i in range(warmup, len(df), step):
        trailing = df.iloc[: i + 1]

        # Indicator scores
        ind_reg = IndicatorRegistry(cfg)
        ind_results = ind_reg.run_all(trailing)
        ind_scorer = CompositeScorer(cfg)
        ind_composite = ind_scorer.score(ind_results)

        # Pattern scores
        pat_reg = PatternRegistry(cfg)
        pat_results = pat_reg.run_all(trailing)
        pat_scorer = PatternCompositeScorer(cfg)
        pat_composite = pat_scorer.score(pat_results)

        dates.append(df.index[i])
        ind_scores.append(float(ind_composite["overall"]))
        pat_scores.append(float(pat_composite["overall"]))

    score_df = pd.DataFrame({
        "indicator_score": ind_scores,
        "pattern_score": pat_scores,
    }, index=dates)
    score_df.index.name = "date"
    return score_df


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def create_price_chart(
    result: AnalysisResult,
    score_df: pd.DataFrame | None = None,
    cfg: Config | None = None,
) -> go.Figure:
    """Create a candlestick price chart with score overlay and S/R levels."""
    df = result.df.copy()
    # Strip timezone if present
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    has_scores = score_df is not None and not score_df.empty

    # Create subplots: price + volume, with optional score overlay
    fig = make_subplots(
        rows=3 if has_scores else 2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2] if has_scores else [0.7, 0.3],
        subplot_titles=(
            ["Price", "Volume", "Scores"] if has_scores else ["Price", "Volume"]
        ),
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color=COLOR_BULLISH,
            decreasing_line_color=COLOR_BEARISH,
            name="Price",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Support levels
    for sl in result.support_levels[:4]:
        fig.add_hline(
            y=sl.price,
            line_dash="dash",
            line_color=COLOR_BULLISH,
            line_width=1,
            opacity=0.5,
            annotation_text=f"S {sl.price:.2f}",
            annotation_position="bottom left",
            annotation_font_size=9,
            annotation_font_color=COLOR_BULLISH,
            row=1, col=1,
        )

    # Resistance levels
    for rl in result.resistance_levels[:4]:
        fig.add_hline(
            y=rl.price,
            line_dash="dash",
            line_color=COLOR_BEARISH,
            line_width=1,
            opacity=0.5,
            annotation_text=f"R {rl.price:.2f}",
            annotation_position="top left",
            annotation_font_size=9,
            annotation_font_color=COLOR_BEARISH,
            row=1, col=1,
        )

    # Volume bars
    colors = [
        COLOR_BULLISH if c >= o else COLOR_BEARISH
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            marker_color=colors,
            opacity=0.5,
            name="Volume",
            showlegend=False,
        ),
        row=2, col=1,
    )

    # Score overlay
    if has_scores:
        score_idx = score_df.index
        if hasattr(score_idx, "tz") and score_idx.tz is not None:
            score_idx = score_idx.tz_localize(None)

        fig.add_trace(
            go.Scatter(
                x=score_idx,
                y=score_df["indicator_score"],
                mode="lines",
                name="Indicator Score",
                line=dict(color=COLOR_IND_SCORE, width=2),
            ),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=score_idx,
                y=score_df["pattern_score"],
                mode="lines",
                name="Pattern Score",
                line=dict(color=COLOR_PAT_SCORE, width=2, dash="dot"),
            ),
            row=3, col=1,
        )

        # Threshold lines on score chart
        # In percentile mode, the actual buy/sell boundaries are dynamic
        # (based on rolling rank), so fixed threshold lines would be
        # misleading.  Only show them in fixed (or gate) mode.
        if cfg is not None:
            strat_cfg_chart = cfg.section("strategy")
            thresholds = strat_cfg_chart.get("score_thresholds", {})
            threshold_mode = str(strat_cfg_chart.get("threshold_mode", "fixed"))
        else:
            thresholds = {}
            threshold_mode = "fixed"

        if threshold_mode != "percentile":
            short_below = float(thresholds.get("short_below", 3.5))
            hold_below = float(thresholds.get("hold_below", 6.0))

            fig.add_hline(
                y=hold_below, line_dash="dot", line_color=COLOR_BULLISH,
                line_width=1, opacity=0.6, row=3, col=1,
                annotation_text=f"LONG > {hold_below}",
                annotation_font_size=9,
            )
            fig.add_hline(
                y=short_below, line_dash="dot", line_color=COLOR_BEARISH,
                line_width=1, opacity=0.6, row=3, col=1,
                annotation_text=f"SHORT < {short_below}",
                annotation_font_size=9,
            )
        fig.update_yaxes(range=[0, 10], row=3, col=1, title_text="Score (0-10)")

    # Layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        height=700 if has_scores else 550,
        margin=dict(l=60, r=30, t=40, b=30),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=11),
    )
    fig.update_xaxes(gridcolor=COLOR_GRID, showgrid=True)
    fig.update_yaxes(gridcolor=COLOR_GRID, showgrid=True)

    return fig


def create_equity_chart(
    bt_result: BacktestResult,
    df: pd.DataFrame,
) -> go.Figure:
    """Create equity curve chart with buy-and-hold comparison and trade markers."""
    curve = bt_result.equity_curve
    if not curve:
        return go.Figure()

    eq_dates = [pt["date"] for pt in curve]
    eq_values = [pt["equity"] for pt in curve]

    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=["Equity Curve"],
    )

    # Strategy equity
    fig.add_trace(go.Scatter(
        x=eq_dates,
        y=eq_values,
        mode="lines",
        name="Strategy",
        line=dict(color=COLOR_EQUITY, width=2),
    ))

    # Buy-and-hold benchmark — start from the same bar as trading begins
    # (after warmup) so the benchmark doesn't get a head start.
    warmup_bars = getattr(bt_result, "warmup_bars", 0) or 0
    if len(df) >= len(curve) and warmup_bars < len(df):
        initial_price = float(df["close"].iloc[warmup_bars])
        initial_cash = bt_result.initial_cash
        bh_values = [
            initial_cash * float(df["close"].iloc[i]) / initial_price
            for i in range(len(curve))
        ]
        bh_dates = eq_dates[:len(bh_values)]
        fig.add_trace(go.Scatter(
            x=bh_dates,
            y=bh_values,
            mode="lines",
            name="Buy & Hold",
            line=dict(color=COLOR_BENCHMARK, width=1.5, dash="dash"),
        ))

    # Trade markers
    for trade in bt_result.trades:
        # Entry marker
        entry_color = COLOR_BULLISH if trade.side == "long" else COLOR_BEARISH
        entry_symbol = "triangle-up" if trade.side == "long" else "triangle-down"

        # Find equity at entry/exit dates
        entry_eq = None
        exit_eq = None
        for pt in curve:
            if pt["date"] == trade.entry_date and entry_eq is None:
                entry_eq = pt["equity"]
            if pt["date"] == trade.exit_date:
                exit_eq = pt["equity"]

        if entry_eq is not None:
            fig.add_trace(go.Scatter(
                x=[trade.entry_date],
                y=[entry_eq],
                mode="markers",
                marker=dict(
                    symbol=entry_symbol,
                    size=8,
                    color=entry_color,
                    line=dict(width=1, color="white"),
                ),
                name=f"{trade.side.upper()} entry",
                showlegend=False,
                hovertext=f"{trade.side.upper()} entry @ ${trade.entry_price:.2f}",
                hoverinfo="text",
            ))

        if exit_eq is not None:
            exit_color = COLOR_BULLISH if trade.pnl > 0 else COLOR_BEARISH
            fig.add_trace(go.Scatter(
                x=[trade.exit_date],
                y=[exit_eq],
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=7,
                    color=exit_color,
                    line=dict(width=1, color="white"),
                ),
                name=f"Exit ({trade.exit_reason})",
                showlegend=False,
                hovertext=(
                    f"Exit @ ${trade.exit_price:.2f} | "
                    f"P&L: {trade.pnl_pct * 100:+.2f}% | {trade.exit_reason}"
                ),
                hoverinfo="text",
            ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        height=400,
        margin=dict(l=60, r=30, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=11),
        yaxis_title="Equity ($)",
    )
    fig.update_xaxes(gridcolor=COLOR_GRID, showgrid=True)
    fig.update_yaxes(gridcolor=COLOR_GRID, showgrid=True)

    return fig


def create_score_histogram(score_df: pd.DataFrame, cfg: Config | None = None) -> go.Figure:
    """Create a histogram of indicator composite scores with threshold markers."""
    if score_df is None or score_df.empty:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=score_df["indicator_score"],
        nbinsx=30,
        name="Indicator Score",
        marker_color=COLOR_IND_SCORE,
        opacity=0.7,
    ))
    fig.add_trace(go.Histogram(
        x=score_df["pattern_score"],
        nbinsx=30,
        name="Pattern Score",
        marker_color=COLOR_PAT_SCORE,
        opacity=0.5,
    ))

    # Threshold lines (only shown in fixed/gate mode — percentile thresholds are dynamic)
    if cfg is not None:
        thresholds = cfg.section("strategy").get("score_thresholds", {})
        threshold_mode = str(cfg.section("strategy").get("threshold_mode", "fixed"))
    else:
        thresholds = {}
        threshold_mode = "fixed"

    if threshold_mode != "percentile":
        short_below = float(thresholds.get("short_below", 3.5))
        hold_below = float(thresholds.get("hold_below", 6.0))

        fig.add_vline(x=short_below, line_dash="dash", line_color=COLOR_BEARISH,
                      annotation_text=f"SHORT < {short_below}", annotation_font_size=10)
        fig.add_vline(x=hold_below, line_dash="dash", line_color=COLOR_BULLISH,
                      annotation_text=f"LONG > {hold_below}", annotation_font_size=10)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        height=300,
        margin=dict(l=60, r=30, t=30, b=30),
        barmode="overlay",
        xaxis_title="Score (0-10)",
        yaxis_title="Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(size=11),
    )
    fig.update_xaxes(gridcolor=COLOR_GRID, range=[0, 10])
    fig.update_yaxes(gridcolor=COLOR_GRID)

    return fig


# ---------------------------------------------------------------------------
# Display sections
# ---------------------------------------------------------------------------

def render_header(result: AnalysisResult, cfg: Config) -> None:
    """Render the ticker header with key info."""
    info = result.info
    price = info.get("current_price") or float(result.df["close"].iloc[-1])
    name = info.get("name", result.ticker)
    overall = result.composite["overall"]
    pat_overall = result.pattern_composite["overall"]

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.markdown(f"### {name} ({result.ticker})")
        meta_parts = []
        if info.get("sector") and info["sector"] != "N/A":
            meta_parts.append(info["sector"])
        if info.get("exchange") and info["exchange"] != "N/A":
            meta_parts.append(info["exchange"])
        if meta_parts:
            st.caption(" | ".join(meta_parts))
    with col2:
        st.metric("Price", f"${price:.2f}")
    with col3:
        sc = score_color(overall)
        st.metric("Indicator Score", f"{overall:.1f} / 10")
    with col4:
        st.metric("Pattern Score", f"{pat_overall:.1f} / 10")


def render_indicator_table(result: AnalysisResult, cfg: Config) -> None:
    """Render indicator breakdown with color-coded score bars."""
    weights = cfg.normalized_weights()

    rows = []
    for r in result.indicator_results:
        rows.append({
            "Indicator": r.name,
            "Value": r.display.get("value_str", "N/A") if not r.error else "ERROR",
            "Detail": r.display.get("detail_str", "") if not r.error else r.error[:60],
            "Score": r.score,
            "Weight": weights.get(r.config_key, 0),
        })

    # Add overall row
    rows.append({
        "Indicator": "**OVERALL**",
        "Value": "",
        "Detail": f"{result.composite['n_scored']} indicators weighted",
        "Score": result.composite["overall"],
        "Weight": 1.0,
    })

    # Build HTML table
    html = _build_score_table_html(
        rows,
        name_col="Indicator",
        columns=["Indicator", "Value", "Detail", "Score", "Weight"],
    )
    st.markdown(html, unsafe_allow_html=True)

    # Subgroup breakdown (if directional mode is active)
    trend_s = result.composite.get("trend_score")
    contrarian_s = result.composite.get("contrarian_score")
    neutral_s = result.composite.get("neutral_score")
    dominant = result.composite.get("dominant_group")
    overall_raw = result.composite.get("overall_raw")
    if trend_s is not None and contrarian_s is not None:
        parts = [
            f"Trend: **{trend_s:.1f}**",
            f"Contrarian: **{contrarian_s:.1f}**",
            f"Neutral: **{neutral_s:.1f}**" if neutral_s is not None else None,
            f"Dominant: **{dominant}**" if dominant and dominant != "none" else None,
        ]
        if overall_raw is not None:
            parts.append(f"Pre-spread: {overall_raw:.1f}")
        detail = " · ".join(p for p in parts if p)
        st.caption(f"Subgroup scores: {detail}")


def render_pattern_table(result: AnalysisResult, cfg: Config) -> None:
    """Render pattern breakdown with color-coded score bars."""
    weights = cfg.normalized_pattern_weights()

    rows = []
    for r in result.pattern_results:
        rows.append({
            "Pattern": r.name,
            "Signal": r.display.get("value_str", "N/A") if not r.error else "ERROR",
            "Detail": r.display.get("detail_str", "") if not r.error else r.error[:60],
            "Score": r.score,
            "Weight": weights.get(r.config_key, 0),
        })

    # Add overall row
    rows.append({
        "Pattern": "**PATTERN OVERALL**",
        "Signal": "",
        "Detail": f"{result.pattern_composite['n_scored']} patterns weighted",
        "Score": result.pattern_composite["overall"],
        "Weight": 1.0,
    })

    # Build HTML table
    html = _build_score_table_html(
        rows,
        name_col="Pattern",
        columns=["Pattern", "Signal", "Detail", "Score", "Weight"],
    )
    st.markdown(html, unsafe_allow_html=True)


def render_suitability(assessment_dict: dict, ticker: str) -> None:
    """Render suitability assessment as an info/warning box."""
    mode_value = assessment_dict["mode"]
    mode_label = mode_value.replace("_", " ").upper()
    mode_map = {
        "long_short": "success",
        "long_only": "warning",
        "hold_only": "error",
    }
    msg_type = mode_map.get(mode_value, "info")

    cols = st.columns(4)
    cols[0].metric("Trading Mode", mode_label)
    cols[1].metric("Avg Volume", f"{assessment_dict['avg_daily_volume']:,.0f}")
    cols[2].metric("ADX", f"{assessment_dict['adx_value']:.1f}")
    cols[3].metric("ATR%", f"{assessment_dict['atr_pct'] * 100:.2f}%")

    for reason in assessment_dict["reasons"]:
        if msg_type == "error":
            st.error(reason)
        elif msg_type == "warning":
            st.warning(reason)
        else:
            st.success(reason)


def render_regime(regime: RegimeAssessment | None) -> None:
    """Render market regime classification as a colored info box with metrics."""
    if regime is None:
        return

    # Color and icon based on regime type
    regime_styles = {
        RegimeType.STRONG_TREND: ("#2ecc71", "trending_up"),        # green
        RegimeType.MEAN_REVERTING: ("#3498db", "swap_horiz"),       # blue
        RegimeType.VOLATILE_CHOPPY: ("#e74c3c", "warning"),         # red
        RegimeType.BREAKOUT_TRANSITION: ("#f39c12", "bolt"),        # orange
    }
    color, icon = regime_styles.get(regime.regime, ("#95a5a6", "help"))
    confidence_pct = regime.confidence * 100

    # Header with colored badge
    sub_type_html = ""
    if regime.sub_type_label:
        sub_type_html = (
            f'<span style="margin-left:8px; background:#555; color:#fff; '
            f'padding:2px 8px; border-radius:12px; font-size:0.85em;">'
            f'{regime.sub_type_label}</span>'
        )
    st.markdown(
        f'<div style="background:{color}22; border-left:4px solid {color}; '
        f'padding:12px 16px; border-radius:4px; margin-bottom:8px;">'
        f'<span style="font-size:1.2em; font-weight:bold; color:{color};">'
        f'{regime.label}</span>'
        f'{sub_type_html}'
        f'<span style="margin-left:12px; color:#888;">Confidence: {confidence_pct:.0f}%</span>'
        f'<br><span style="color:#aaa; font-size:0.9em;">{regime.description}</span>'
        + (f'<br><span style="color:#bbb; font-size:0.85em;">Sub-type: {regime.sub_type_description}</span>'
           if regime.sub_type_description else "")
        + f'</div>',
        unsafe_allow_html=True,
    )

    # Metrics row
    m = regime.metrics
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.metric("Total Return", f"{m.total_return:+.1%}")
    mc2.metric("ADX (current)", f"{m.adx:.1f}")
    mc3.metric("ADX (avg)", f"{m.rolling_adx_mean:.1f}")
    mc4.metric("Trend MA %", f"{m.pct_above_ma:.0f}%")
    mc5.metric("ATR%", f"{m.atr_pct:.3f}")
    mc6.metric("Dir Changes", f"{m.direction_changes:.0%}")

    # Reasons
    if regime.reasons:
        with st.expander("Classification Reasoning", expanded=False):
            for reason in regime.reasons:
                st.markdown(f"- {reason}")

            # Regime scores comparison
            if regime.regime_scores:
                st.markdown("**Regime Scores:**")
                score_parts = []
                for label, score in sorted(
                    regime.regime_scores.items(), key=lambda x: -x[1]
                ):
                    name = label.replace("_", " ").title()
                    marker = " **" if label == regime.regime.value else ""
                    end_marker = "**" if label == regime.regime.value else ""
                    score_parts.append(f"- {marker}{name}{end_marker}: {score:.2f}")
                st.markdown("\n".join(score_parts))

def render_backtest_metrics(bt_result: BacktestResult) -> None:
    """Render backtest performance metrics as metric cards."""
    c1, c2, c3, c4 = st.columns(4)

    ret_delta = f"{bt_result.total_return_pct:+.2f}%"
    c1.metric("Total Return", ret_delta)
    c2.metric("Annualized", f"{bt_result.annualized_return_pct:+.2f}%")
    c3.metric("Max Drawdown", f"{bt_result.max_drawdown_pct:.2f}%")
    c4.metric("Sharpe Ratio", f"{bt_result.sharpe_ratio:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total Trades", f"{bt_result.total_trades}")
    c6.metric("Win Rate", f"{bt_result.win_rate_pct:.1f}%")
    c7.metric("Profit Factor", f"{bt_result.profit_factor:.2f}")
    c8.metric("Avg Trade P&L", f"{bt_result.avg_trade_pnl_pct:+.2f}%")


def render_trade_log(bt_result: BacktestResult) -> None:
    """Render the trade log as an expandable dataframe."""
    if not bt_result.trades:
        st.info("No trades were executed during the backtest period.")
        return

    rows = []
    for i, t in enumerate(bt_result.trades, 1):
        rows.append({
            "#": i,
            "Side": t.side.upper(),
            "Entry Date": t.entry_date,
            "Entry Price": f"${t.entry_price:.2f}",
            "Entry Reason": t.entry_reason,
            "Exit Date": t.exit_date,
            "Exit Price": f"${t.exit_price:.2f}",
            "Qty": int(t.quantity),
            "P&L": f"${t.pnl:+,.2f}",
            "P&L %": f"{t.pnl_pct * 100:+.2f}%",
            "Exit Reason": t.exit_reason,
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True, height=400)


def render_significant_patterns(bt_result: BacktestResult) -> None:
    """Render significant patterns timeline as an expandable dataframe."""
    patterns = bt_result.significant_patterns
    if not patterns:
        st.info("No significant patterns were detected during the backtest period.")
        return

    rows = []
    for p in patterns:
        # Color-code signal
        if p.signal == "bullish":
            signal_display = "\u2191 BULLISH"
        elif p.signal == "bearish":
            signal_display = "\u2193 BEARISH"
        else:
            signal_display = "\u25CB NEUTRAL"

        rows.append({
            "Date": p.date,
            "Detector": p.detector,
            "Pattern": p.pattern,
            "Signal": signal_display,
            "Strength": f"{p.strength:.2f}",
            "Confidence": p.confidence,
            "Detail": p.detail,
        })

    df = pd.DataFrame(rows)

    # Summary stats
    bullish_count = sum(1 for p in patterns if p.signal == "bullish")
    bearish_count = sum(1 for p in patterns if p.signal == "bearish")

    st.markdown(
        f'<div style="font-size:0.85rem;color:#aaa;margin-bottom:8px;">'
        f'Total: {len(patterns)} significant patterns &nbsp;'
        f'(<span style="color:{COLOR_BULLISH};">{bullish_count} bullish</span>'
        f'&nbsp;&nbsp;<span style="color:{COLOR_BEARISH};">{bearish_count} bearish</span>)'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(df, width="stretch", hide_index=True, height=400)


def render_strategy_config(cfg: Config) -> None:
    """Render strategy configuration summary (read-only view of active config)."""
    strat_cfg = cfg.section("strategy")
    bt_cfg = cfg.section("backtest")
    thresholds = strat_cfg.get("score_thresholds", {})

    rows = []

    threshold_mode = strat_cfg.get("threshold_mode", "fixed")
    rows.append(("Threshold Mode", threshold_mode))

    if threshold_mode == "percentile":
        pct_cfg = strat_cfg.get("percentile_thresholds", {})
        rows.append(("SHORT percentile <=", f"{pct_cfg.get('short_percentile', 25)}%"))
        rows.append(("LONG percentile >=", f"{pct_cfg.get('long_percentile', 75)}%"))
        rows.append(("Lookback", f"{pct_cfg.get('lookback_bars', 60)} bars"))
    else:
        rows.append(("SHORT when score <=", f"{thresholds.get('short_below', 3.5)}"))
        rows.append(("LONG when score >", f"{thresholds.get('hold_below', 6.0)}"))

    combo_mode = strat_cfg.get("combination_mode", "weighted")
    rows.append(("Combination Mode", combo_mode))
    if combo_mode == "weighted":
        rows.append(("Indicator Weight", f"{float(strat_cfg.get('indicator_weight', 0.7)):.0%}"))
        rows.append(("Pattern Weight", f"{float(strat_cfg.get('pattern_weight', 0.3)):.0%}"))
    elif combo_mode == "boost":
        rows.append(("Boost Strength", f"{strat_cfg.get('boost_strength', 0.5)}"))
        rows.append(("Dead Zone", f"+/-{strat_cfg.get('boost_dead_zone', 0.3)}"))

    rows.append(("Stop Loss", f"{strat_cfg.get('stop_loss_pct', 0.05) * 100:.1f}%"))
    rows.append(("Take Profit", f"{strat_cfg.get('take_profit_pct', 0.15) * 100:.1f}%"))
    rows.append(("Rebalance", f"every {strat_cfg.get('rebalance_interval', 5)} bars"))
    rows.append(("EOD Flatten", "ON" if strat_cfg.get("flatten_eod", False) else "OFF"))

    # ATR-adaptive stop
    atr_enabled = strat_cfg.get("atr_stop_enabled", True)
    rows.append(("ATR Stop", "ON" if atr_enabled else "OFF"))
    if atr_enabled:
        rows.append(("ATR Multiplier", f"{strat_cfg.get('atr_stop_multiplier', 2.5):.1f}x"))
        rows.append(("ATR Period", f"{strat_cfg.get('atr_stop_period', 14)}"))

    # Trend confirmation
    trend_enabled = strat_cfg.get("trend_confirm_enabled", True)
    rows.append(("Trend Filter", "ON" if trend_enabled else "OFF"))
    if trend_enabled:
        rows.append(("Trend EMA Period", f"{strat_cfg.get('trend_confirm_period', 20)}"))
        rows.append(("Re-entry Grace", f"{strat_cfg.get('reentry_grace_bars', 10)} bars"))

    # Consecutive loss cooldown
    rows.append(("Cooldown Max Losses", f"{strat_cfg.get('cooldown_max_losses', 2)}"))
    rows.append(("Cooldown Distance Mult", f"{strat_cfg.get('cooldown_distance_mult', 2.0):.1f}x"))
    rows.append(("Cooldown Min Score", f"{strat_cfg.get('cooldown_min_score', 4.5):.1f}"))

    # Global directional bias
    bias_enabled = strat_cfg.get("global_trend_bias", True)
    rows.append(("Global Trend Bias", "ON" if bias_enabled else "OFF"))
    if bias_enabled:
        rows.append(("Bias Threshold", f"{strat_cfg.get('global_bias_threshold', 0.10) * 100:.0f}%"))

    # New strategy params
    rows.append(("Trend Confirm MA", strat_cfg.get("trend_confirm_ma_type", "ema").upper()))
    tol = strat_cfg.get("trend_confirm_tolerance_pct", 0.0)
    if tol > 0:
        rows.append(("Trend Tolerance", f"{tol * 100:.1f}%"))
    rows.append(("Trend Bias Return Thr.", f"{strat_cfg.get('trend_bias_return_threshold', 0.15) * 100:.0f}%"))
    rows.append(("Extreme Exit Offset", f"{strat_cfg.get('extreme_exit_score_offset', 1.5):.1f}"))
    rows.append(("Breakout Min Move", f"{strat_cfg.get('breakout_min_move_ratio', 0.4):.2f}"))
    rows.append(("Pyramiding", "ON" if strat_cfg.get("allow_pyramiding", False) else "OFF"))
    rows.append(("Immediate Reversal", "ON" if strat_cfg.get("allow_immediate_reversal", True) else "OFF"))
    rows.append(("Disable TP (Strong Trend)", "ON" if strat_cfg.get("disable_take_profit_in_strong_trend", True) else "OFF"))
    rows.append(("Trail Requires Profit", "ON" if strat_cfg.get("trailing_stop_require_profit", True) else "OFF"))
    rows.append(("Pctile Min Fill Ratio", f"{strat_cfg.get('percentile_min_fill_ratio', 0.8):.2f}"))
    rows.append(("Cooldown Reset on BE", "ON" if strat_cfg.get("cooldown_reset_on_breakeven", True) else "OFF"))

    rows.append(("Slippage", f"{bt_cfg.get('slippage_pct', 0.001) * 100:.2f}%"))
    rows.append(("Warmup Bars", f"{bt_cfg.get('warmup_bars', 200)}"))
    rows.append(("Max Warmup Ratio", f"{bt_cfg.get('max_warmup_ratio', 0.5):.0%}"))
    rows.append(("Min Warmup Bars", f"{bt_cfg.get('min_warmup_bars', 20)}"))
    rows.append(("Min Post-Warmup Bars", f"{bt_cfg.get('min_post_warmup_bars', 10)}"))
    rows.append(("Trading Days/Year", f"{bt_cfg.get('trading_days_per_year', 252)}"))
    rows.append(("Default Score", f"{bt_cfg.get('default_score', 5.0):.1f}"))
    rows.append(("Close on End", "ON" if bt_cfg.get("close_on_end_of_data", True) else "OFF"))

    df = pd.DataFrame(rows, columns=["Parameter", "Value"])
    st.dataframe(df, width="stretch", hide_index=True)



# ---------------------------------------------------------------------------
# Stock overview (Step 1)
# ---------------------------------------------------------------------------

def render_stock_overview(
    result: AnalysisResult, cfg: Config,
) -> None:
    """Render the stock overview: basic info, regime, sub-type, scores, price chart."""

    # ── Header: name, price, scores ───────────────────────────────────────
    info = result.info
    price = info.get("current_price") or float(result.df["close"].iloc[-1])
    name = info.get("name", result.ticker)
    overall_ind = result.composite["overall"]
    overall_pat = result.pattern_composite["overall"]

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.markdown(f"### {name} ({result.ticker})")
        meta_parts = []
        if info.get("sector") and info["sector"] != "N/A":
            meta_parts.append(info["sector"])
        if info.get("exchange") and info["exchange"] != "N/A":
            meta_parts.append(info["exchange"])
        if meta_parts:
            st.caption(" | ".join(meta_parts))
    with col2:
        st.metric("Price", f"${price:.2f}")
    with col3:
        st.metric("Indicator Score", f"{overall_ind:.1f} / 10")
    with col4:
        st.metric("Pattern Score", f"{overall_pat:.1f} / 10")

    # Subgroup score summary (compact)
    trend_s = result.composite.get("trend_score")
    contrarian_s = result.composite.get("contrarian_score")
    dominant = result.composite.get("dominant_group")
    if trend_s is not None and contrarian_s is not None:
        dom_label = dominant.title() if dominant and dominant != "none" else "—"
        st.caption(
            f"Subgroups — Trend: {trend_s:.1f} · "
            f"Contrarian: {contrarian_s:.1f} · "
            f"Dominant: {dom_label}"
        )

    # ── Regime & Sub-Type (prominent display) ─────────────────────────────
    if result.regime is not None:
        render_regime(result.regime)

    # ── Quick Signal (recommendation without backtest) ────────────────────
    st.markdown("---")
    st.subheader("Quick Signal")
    rec = compute_recommendation(result, bt_result=None, cfg=cfg)
    render_recommendation(rec, context="overview")

    st.markdown("---")

    # ── Price chart ───────────────────────────────────────────────────────
    st.subheader("Price Chart")
    price_fig = create_price_chart(result, score_df=None, cfg=cfg)
    st.plotly_chart(price_fig, width="stretch")

    # ── Indicator & Pattern breakdown (collapsed by default) ──────────────
    with st.expander("Indicator Breakdown"):
        render_indicator_table(result, cfg)
    with st.expander("Pattern Signals"):
        render_pattern_table(result, cfg)


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------

def compute_recommendation(
    result: AnalysisResult,
    bt_result: BacktestResult | None,
    cfg: Config,
) -> dict:
    """Compute a BUY/HOLD/SELL recommendation using the strategy's own decision logic.

    Creates a fresh ``ScoreBasedStrategy`` and calls ``on_bar()`` with a
    ``StrategyContext`` built from the analysis result's last bar.  Position is
    set to 0 (flat) so the answer is "if you had money to deploy right now,
    what would the strategy say?"

    When percentile threshold mode is active, the strategy's score window is
    pre-seeded with historical effective scores (via ``_build_percentile_window``)
    so that percentile ranking works correctly despite the single ``on_bar()``
    call.

    The strategy's full decision pipeline applies: score thresholds, trend
    confirmation, regime adaptation, global directional bias, breakout gates,
    and trading mode constraints.

    Returns a dict with:
        signal: "BUY" | "HOLD" | "SELL"
        confidence: "high" | "medium" | "low"
        reasons: list[str]
        score: float   (effective blended score for display)
    """
    strat_cfg = cfg.section("strategy")
    regime_adapt = cfg.section("regime").get("strategy_adaptation", {})

    # ── Determine trading mode via suitability analysis ────────────────────
    # Auto-detect whether the stock is suitable for long/short, long-only,
    # or hold-only rather than blindly assuming long_short.
    df_for_suit = result.df
    trading_mode = TradingMode.LONG_SHORT  # fallback
    if df_for_suit is not None and not df_for_suit.empty:
        try:
            suit_assessment = SuitabilityAnalyzer(cfg).assess(df_for_suit)
            trading_mode = suit_assessment.mode
        except Exception:
            pass  # graceful degradation — default to LONG_SHORT

    # ── Create a fresh strategy instance (no accumulated state) ──────────
    strategy = ScoreBasedStrategy(
        params=strat_cfg,
        trading_mode=trading_mode,
        regime_adaptation=regime_adapt,
    )
    strategy.on_start({"ticker": result.ticker, "recommendation": True})

    # ── Seed percentile window if percentile mode is active ──────────────
    # Without seeding, percentile mode always falls back to fixed thresholds
    # because a single on_bar() call can never fill the min_samples window.
    threshold_mode = str(strat_cfg.get("threshold_mode", "fixed"))
    if threshold_mode == "percentile":
        try:
            pct_cfg = strat_cfg.get("percentile_thresholds", {})
            lookback_bars = int(pct_cfg.get("lookback_bars", 60))
            pct_step = max(1, int(pct_cfg.get("percentile_step", 5)))
            combination_mode = str(strat_cfg.get("combination_mode", "weighted"))
            ind_weight = float(strat_cfg.get("indicator_weight", 0.7))
            pat_weight = float(strat_cfg.get("pattern_weight", 0.3))
            boost_strength = float(strat_cfg.get("boost_strength", 0.5))
            boost_dead_zone = float(strat_cfg.get("boost_dead_zone", 0.3))

            provider = YahooFinanceProvider()

            # result.period is a display label (e.g. "6mo" or "2024-01-01 → today").
            # For date-range periods, parse start/end and pass them explicitly
            # so the Yahoo fetch receives valid parameters.
            _pct_period: str | None = result.period
            _pct_start: str | None = None
            _pct_end: str | None = None
            if _pct_period and "\u2192" in _pct_period:
                # Format: "YYYY-MM-DD → YYYY-MM-DD" or "YYYY-MM-DD → today"
                parts = [p.strip() for p in _pct_period.split("\u2192")]
                _pct_start = parts[0]
                _pct_end = None if parts[1] == "today" else parts[1]
                _pct_period = None  # use start/end instead

            window = _build_percentile_window(
                cfg,
                provider,
                ticker=result.ticker,
                period=_pct_period,
                interval="1d",
                lookback_bars=lookback_bars,
                step=pct_step,
                combination_mode=combination_mode,
                ind_weight=ind_weight,
                pat_weight=pat_weight,
                boost_strength=boost_strength,
                boost_dead_zone=boost_dead_zone,
                start=_pct_start,
                end=_pct_end,
            )
            if window:
                strategy.seed_score_window(window)
        except Exception:
            pass  # graceful degradation — percentile falls back to fixed

    # ── Build StrategyContext from the analysis result's last bar ─────────
    df = result.df
    if df is None or df.empty:
        return {
            "signal": "HOLD",
            "confidence": "low",
            "reasons": ["No data available for recommendation."],
            "score": 5.0,
        }

    last_row = df.iloc[-1]
    bar_dict = {
        "open": float(last_row["open"]),
        "high": float(last_row["high"]),
        "low": float(last_row["low"]),
        "close": float(last_row["close"]),
        "volume": float(last_row["volume"]),
    }

    # Overall scores from analysis
    overall_score = result.composite.get("overall", 5.0)
    pattern_score = result.pattern_composite.get("overall", 5.0)

    # Per-indicator scores (needed for StrategyContext.scores)
    per_scores: dict[str, float] = {}
    if result.indicator_results:
        for ir in result.indicator_results:
            if not ir.error:
                per_scores[ir.config_key] = ir.score

    # Compute trend MA for the last bar (same logic as backtest engine)
    trend_confirm_period = int(strat_cfg.get("trend_confirm_period", 20))
    trend_confirm_ma_type = str(strat_cfg.get("trend_confirm_ma_type", "ema"))
    if trend_confirm_ma_type.lower() == "sma":
        trend_ma_series = df["close"].rolling(
            window=trend_confirm_period, min_periods=1
        ).mean()
    else:
        trend_ma_series = df["close"].ewm(
            span=trend_confirm_period, adjust=False
        ).mean()
    current_trend_ma = float(trend_ma_series.iloc[-1])

    # Regime information from the analysis result
    regime: RegimeType | None = None
    regime_sub_type: RegimeSubType | None = None
    regime_trend: str = "neutral"
    regime_total_return: float = 0.0

    regime_source = result.regime
    if bt_result is not None and bt_result.regime is not None:
        regime_source = bt_result.regime

    if regime_source is not None:
        regime = regime_source.regime
        regime_sub_type = regime_source.sub_type
        regime_trend = regime_source.metrics.trend_direction
        regime_total_return = regime_source.metrics.total_return

    # Average volume for breakout volume-surge gate
    ra_cfg = strat_cfg.get("regime_adaptation", {})
    bk_cfg = ra_cfg.get("breakout_transition", {})
    avg_vol_win = int(bk_cfg.get("avg_volume_window", 20))
    avg_volume = float(
        df["volume"].iloc[-avg_vol_win:].mean() if len(df) >= avg_vol_win
        else df["volume"].mean()
    )

    ctx = StrategyContext(
        bar=bar_dict,
        indicators={},
        scores=per_scores,
        overall_score=overall_score,
        pattern_score=pattern_score,
        position=0.0,           # flat — "fresh entry" question
        cash=100_000.0,
        portfolio_value=100_000.0,
        trend_ma=current_trend_ma,
        regime=regime,
        regime_sub_type=regime_sub_type,
        regime_trend=regime_trend,
        regime_total_return=regime_total_return,
        metadata={"avg_volume": avg_volume},
    )

    # ── Call the strategy ────────────────────────────────────────────────
    order = strategy.on_bar(ctx)
    signal = order.signal.value  # "BUY", "SELL", or "HOLD"

    # ── Derive confidence from score distance to thresholds ──────────────
    thresholds = strat_cfg.get("score_thresholds", {})
    short_below = float(thresholds.get("short_below", 3.5))
    hold_below = float(thresholds.get("hold_below", 6.0))

    # Regime adaptation: mean-reverting tightens thresholds
    if regime == RegimeType.MEAN_REVERTING:
        regime_adapt_cfg = cfg.section("regime").get("strategy_adaptation", {})
        mr_cfg = regime_adapt_cfg.get("mean_reverting", {})
        if mr_cfg.get("tighten_thresholds", True):
            adj = float(mr_cfg.get("threshold_adjustment", 0.3))
            short_below += adj
            hold_below -= adj
            if short_below >= hold_below:
                mid = (float(thresholds.get("short_below", 3.5)) + float(thresholds.get("hold_below", 6.0))) / 2
                short_below = mid - 0.1
                hold_below = mid + 0.1

    # Compute the effective blended score (same as strategy internals)
    combination_mode = str(strat_cfg.get("combination_mode", "weighted"))
    ind_weight = float(strat_cfg.get("indicator_weight", 0.7))
    pat_weight = float(strat_cfg.get("pattern_weight", 0.3))

    if combination_mode == "boost":
        boost_strength = float(strat_cfg.get("boost_strength", 0.5))
        boost_dead_zone = float(strat_cfg.get("boost_dead_zone", 0.3))
        pat_dev = pattern_score - 5.0
        if abs(pat_dev) <= boost_dead_zone:
            effective_score = overall_score
        else:
            eff_dev = pat_dev - (boost_dead_zone if pat_dev > 0 else -boost_dead_zone)
            effective_score = max(0.0, min(10.0, overall_score + eff_dev * boost_strength))
    elif combination_mode == "gate":
        effective_score = overall_score  # gate mode doesn't blend
    else:
        w_total = ind_weight + pat_weight
        if w_total > 0:
            effective_score = (ind_weight * overall_score + pat_weight * pattern_score) / w_total
        else:
            effective_score = overall_score

    # Confidence: how far the effective score is from the nearest threshold
    conf_cfg = strat_cfg.get("confidence_thresholds", {})
    conf_high = float(conf_cfg.get("high", 1.5))
    conf_med = float(conf_cfg.get("medium", 0.5))
    hold_conf_high = float(conf_cfg.get("hold_high", 1.0))
    hold_conf_med = float(conf_cfg.get("hold_medium", 0.3))

    if combination_mode == "gate":
        # Gate mode: confidence is based on both scores' distances from
        # their respective gate thresholds, using the weakest link.
        g_ind_min = float(strat_cfg.get("gate_indicator_min", 5.5))
        g_ind_max = float(strat_cfg.get("gate_indicator_max", 4.5))
        g_pat_min = float(strat_cfg.get("gate_pattern_min", 5.5))
        g_pat_max = float(strat_cfg.get("gate_pattern_max", 4.5))

        if signal == "BUY":
            # Both had to pass their min thresholds — weakest margin
            distance = min(overall_score - g_ind_min, pattern_score - g_pat_min)
            confidence = "high" if distance >= conf_high else ("medium" if distance >= conf_med else "low")
        elif signal == "SELL":
            # Both had to pass below their max thresholds — weakest margin
            distance = min(g_ind_max - overall_score, g_pat_max - pattern_score)
            confidence = "high" if distance >= conf_high else ("medium" if distance >= conf_med else "low")
        else:
            # HOLD — how close is the nearest score to flipping a gate?
            dist_to_buy = min(g_ind_min - overall_score, g_pat_min - pattern_score)
            dist_to_sell = min(overall_score - g_ind_max, pattern_score - g_pat_max)
            min_dist = min(dist_to_buy, dist_to_sell)
            confidence = "high" if min_dist >= hold_conf_high else ("medium" if min_dist >= hold_conf_med else "low")
    else:
        # Weighted / boost: use effective score distance from thresholds
        if signal == "BUY":
            distance = effective_score - hold_below
            confidence = "high" if distance >= conf_high else ("medium" if distance >= conf_med else "low")
        elif signal == "SELL":
            distance = short_below - effective_score
            confidence = "high" if distance >= conf_high else ("medium" if distance >= conf_med else "low")
        else:
            # HOLD — confidence reflects how firmly neutral
            dist_to_buy = hold_below - effective_score
            dist_to_sell = effective_score - short_below
            min_dist = min(dist_to_buy, dist_to_sell)
            confidence = "high" if min_dist >= hold_conf_high else ("medium" if min_dist >= hold_conf_med else "low")

    # ── Build reasons list ───────────────────────────────────────────────
    reasons: list[str] = []
    reasons.append(f"Effective score: {effective_score:.1f}/10 (indicator: {overall_score:.1f}, pattern: {pattern_score:.1f})")
    if combination_mode == "gate":
        g_ind_min = float(strat_cfg.get("gate_indicator_min", 5.5))
        g_ind_max = float(strat_cfg.get("gate_indicator_max", 4.5))
        reasons.append(f"Gate thresholds: BUY ind>{g_ind_min}, SELL ind<{g_ind_max}")
    else:
        reasons.append(f"Thresholds: BUY > {hold_below:.1f}, SELL <= {short_below:.1f}")

    if regime is not None:
        regime_label = REGIME_LABELS.get(regime, regime.value)
        reasons.append(f"Regime: {regime_label}")
        if regime_total_return != 0:
            reasons.append(f"Regime total return: {regime_total_return:+.1%}")

    close = bar_dict["close"]
    if current_trend_ma > 0:
        ma_dist = (close - current_trend_ma) / current_trend_ma
        side = "above" if ma_dist >= 0 else "below"
        reasons.append(f"Price {side} trend MA by {abs(ma_dist):.1%}")

    # Strategy notes often contain useful detail about why
    if order.notes:
        reasons.append(f"Strategy: {order.notes}")

    return {
        "signal": signal,
        "confidence": confidence,
        "reasons": reasons,
        "score": effective_score,
    }


def render_recommendation(rec: dict, context: str = "backtest") -> None:
    """Render the recommendation panel — single BUY/HOLD/SELL signal.

    *context* controls the explanatory caption:
    - ``"overview"``: shown in the stock overview before any backtest.
    - ``"backtest"``: shown alongside backtest results.
    """
    signal = rec["signal"]
    confidence = rec["confidence"]

    color_map = {"BUY": "#2ecc71", "SELL": "#e74c3c", "HOLD": "#f39c12"}
    conf_badge = {"high": ("#2ecc71", "High"), "medium": ("#f39c12", "Medium"), "low": ("#e74c3c", "Low")}

    color = color_map.get(signal, "#95a5a6")
    conf_color, conf_label = conf_badge.get(confidence, ("#95a5a6", "?"))

    st.markdown(
        f'<div style="background:{color}22; border-left:4px solid {color}; '
        f'padding:16px 20px; border-radius:4px; margin-bottom:12px;">'
        f'<div style="display:flex; align-items:center; gap:12px;">'
        f'<span style="font-size:2em; font-weight:bold; color:{color};">{signal}</span>'
        f'<span style="background:{conf_color}33; color:{conf_color}; padding:2px 10px; '
        f'border-radius:12px; font-size:0.9em;">{conf_label} confidence</span>'
        f'</div>'
        f'<div style="margin-top:8px; color:#aaa; font-size:0.9em;">'
        f'Effective score: {rec["score"]:.1f}/10'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Reasons + context-appropriate caption
    with st.expander("Recommendation details"):
        for reason in rec["reasons"]:
            st.markdown(f"- {reason}")
        if context == "overview":
            st.caption(
                "This is a **quick signal** based on the strategy's decision logic "
                "(score thresholds, trend confirmation, regime adaptation, directional bias) "
                "applied to the latest bar. It uses current indicator and pattern scores "
                "with the detected market regime. "
                "Run a backtest to see how the strategy actually performs on this stock "
                "and to get a **backtested recommendation** that accounts for historical "
                "strategy performance. Not financial advice."
            )
        else:
            st.caption(
                "This **backtested recommendation** uses the same strategy logic as the "
                "quick signal above, but sources its regime classification from the backtest "
                "period. Read it alongside the backtest performance metrics — a BUY signal "
                "is more meaningful when the strategy is profitable on this stock. "
                "Not financial advice."
            )


# ---------------------------------------------------------------------------
# Custom backtest parameter panel
# ---------------------------------------------------------------------------

def render_custom_backtest_params() -> dict:
    """Render custom backtest parameter controls and return selections.

    Returns dict with: period, interval, start, end, objective, trading_mode
    """
    data = st.session_state["config_data"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Data Range")
        date_mode = st.radio(
            "Range mode",
            ["Period", "Custom dates"],
            horizontal=True,
            key="custom_bt_range_mode",
        )

        period = None
        start_date = None
        end_date = None

        if date_mode == "Period":
            period = st.selectbox(
                "Period", VALID_PERIODS, index=VALID_PERIODS.index("2y"),
                key="custom_bt_period",
            )
        else:
            today = datetime.date.today()
            default_start = today - datetime.timedelta(days=365 * 2)
            start_date = st.date_input("Start date", value=default_start, key="custom_bt_start")
            end_date = st.date_input("End date", value=today, key="custom_bt_end")
            start_date = start_date.strftime("%Y-%m-%d")
            end_date = end_date.strftime("%Y-%m-%d")

        interval = st.selectbox(
            "Interval", ALL_INTERVALS, index=ALL_INTERVALS.index("1d"),
            key="custom_bt_interval",
        )

    with col2:
        st.markdown("##### Strategy")
        cfg_temp = Config.load()
        objectives = ["(none)"] + cfg_temp.available_objectives()
        objective = st.selectbox(
            "Objective preset", objectives, index=0, key="custom_bt_objective",
        )
        if objective == "(none)":
            objective = None

        # When objective changes, re-apply from scratch
        prev_obj = st.session_state.get("_prev_custom_objective")
        if objective != prev_obj:
            _apply_objective_to_session(objective)
            st.session_state["_prev_custom_objective"] = objective

        trading_mode = st.selectbox(
            "Trading mode", TRADING_MODES, index=0, key="custom_bt_mode",
        )

    # ── Parameter tuning expanders ────────────────────────────────────────
    st.markdown("##### Parameter Tuning")

    with st.expander("Indicator Weights"):
        _edit_indicator_weights(data)

    with st.expander("Composite Scoring"):
        _edit_composite_scoring(data)

    with st.expander("Indicator Parameters"):
        _edit_indicator_params(data)

    with st.expander("Pattern Weights"):
        _edit_pattern_weights(data)

    with st.expander("Pattern-Indicator Combination"):
        _edit_pattern_indicator_combination(data)

    with st.expander("Scoring Thresholds"):
        _edit_scoring_thresholds(data)

    with st.expander("Strategy"):
        _edit_strategy_params(data)

    with st.expander("Backtest"):
        _edit_backtest_params(data)

    with st.expander("Regime Classification"):
        _edit_regime_params(data)

    with st.expander("Suitability Detection"):
        _edit_suitability_params(data)

    # Write back to session state
    st.session_state["config_data"] = data

    return {
        "period": period,
        "start": start_date,
        "end": end_date,
        "interval": interval,
        "objective": objective,
        "trading_mode": trading_mode,
    }


# ---------------------------------------------------------------------------
# Backtest results display
# ---------------------------------------------------------------------------

def render_backtest_section(
    bt_result: BacktestResult,
    result: AnalysisResult,
    cfg: Config,
    trading_mode_val: str,
    assessment_dict: dict | None,
) -> None:
    """Render the full backtest results section with recommendation."""

    # ── Recommendation ────────────────────────────────────────────────────
    st.subheader("Backtested Recommendation")
    rec = compute_recommendation(result, bt_result, cfg)
    render_recommendation(rec, context="backtest")

    # ── Suitability ───────────────────────────────────────────────────────
    if assessment_dict is not None:
        with st.expander("Suitability Assessment"):
            render_suitability(assessment_dict, bt_result.ticker)

    # Market regime (from backtest)
    if bt_result.regime is not None:
        with st.expander("Market Regime (Backtest)"):
            render_regime(bt_result.regime)

    # Trading mode badge
    mode_label = trading_mode_val.replace("_", " ").upper()
    st.info(f"Trading mode: **{mode_label}**")

    # ── Performance ───────────────────────────────────────────────────────
    st.subheader("Performance Summary")
    render_backtest_metrics(bt_result)

    # ── Equity curve ──────────────────────────────────────────────────────
    st.subheader("Equity Curve")
    eq_fig = create_equity_chart(bt_result, result.df)
    st.plotly_chart(eq_fig, width="stretch")

    # ── Strategy config ───────────────────────────────────────────────────
    with st.expander("Strategy Configuration"):
        render_strategy_config(cfg)

    # ── Trade log ─────────────────────────────────────────────────────────
    with st.expander(f"Trade Log ({bt_result.total_trades} trades)"):
        render_trade_log(bt_result)

    # ── Significant patterns ──────────────────────────────────────────────
    sig_count = len(bt_result.significant_patterns)
    with st.expander(f"Significant Patterns Timeline ({sig_count} patterns)"):
        render_significant_patterns(bt_result)


# ---------------------------------------------------------------------------
# DCA Backtest tab
# ---------------------------------------------------------------------------

_DCA_MODES = ["pure", "dip_weighted", "score_integrated"]
_DCA_MODE_LABELS = {
    "pure": "Pure DCA",
    "dip_weighted": "Dip-Weighted DCA",
    "score_integrated": "Score-Integrated DCA",
}
_DCA_FREQUENCIES = ["weekly", "biweekly", "monthly"]


def _render_dca_params(cfg: Config) -> dict:
    """Render DCA parameter controls in the sidebar and return overrides dict."""
    dca = cfg.section("dca")
    ov: dict = {}

    mode = st.radio(
        "DCA Mode",
        _DCA_MODES,
        index=_DCA_MODES.index(dca.get("mode", "dip_weighted")),
        format_func=lambda m: _DCA_MODE_LABELS.get(m, m),
        key="dca_mode",
        horizontal=True,
        help=(
            "**Pure:** Fixed dollar amount every interval — no timing, serves as baseline. "
            "**Dip-Weighted:** Buys more when price drops from recent high (mild/strong/extreme tiers). "
            "**Score-Integrated:** Uses technical score, RSI, and Bollinger Bands to modulate buy amounts, "
            "with safety gates for breakaway gaps and low volume."
        ),
    )
    ov["mode"] = mode

    col1, col2 = st.columns(2)
    with col1:
        ov["base_amount"] = st.number_input(
            "Base Amount ($)",
            min_value=50, max_value=50_000, step=50,
            value=int(dca.get("base_amount", 500)),
            key="dca_base_amount",
            help=(
                "The fixed dollar amount invested each period. In dip-weighted / "
                "score-integrated modes, this is the baseline — actual amount = "
                "base × multiplier when dips are detected."
            ),
        )
    with col2:
        freq = st.selectbox(
            "Frequency",
            _DCA_FREQUENCIES,
            index=_DCA_FREQUENCIES.index(dca.get("frequency", "monthly")),
            key="dca_freq",
            help=(
                "How often to buy. Weekly ≈ every 5 trading days, biweekly ≈ "
                "every 10, monthly ≈ every 21 trading days."
            ),
        )
        ov["frequency"] = freq

    ov["drip"] = st.checkbox(
        "Reinvest Dividends (DRIP)",
        value=bool(dca.get("drip", True)),
        key="dca_drip",
        help=(
            "Dividend Reinvestment Plan. When enabled, dividends are automatically "
            "reinvested as new shares at the current price on ex-dividend date. "
            "When off, dividends are not tracked."
        ),
    )

    if mode != "pure":
        with st.expander("Dip Thresholds & Multipliers"):
            dt = dca.get("dip_thresholds", {})
            ml = dca.get("multipliers", {})

            st.markdown("**Dip Detection Thresholds** (% drop from recent high)")
            c1, c2, c3 = st.columns(3)
            with c1:
                mild = st.slider("Mild Dip %", 1.0, 15.0,
                                 float(dt.get("mild_drop_pct", 5.0)),
                                 step=0.5, key="dca_mild_pct",
                                 help="How far the price must drop from its recent high (within lookback window) to trigger the mild multiplier tier.")
            with c2:
                strong = st.slider("Strong Dip %", 5.0, 30.0,
                                   float(dt.get("strong_drop_pct", 10.0)),
                                   step=0.5, key="dca_strong_pct",
                                   help="How far the price must drop from its recent high (within lookback window) to trigger the strong multiplier tier.")
            with c3:
                extreme = st.slider("Extreme Dip %", 10.0, 50.0,
                                    float(dt.get("extreme_drop_pct", 20.0)),
                                    step=1.0, key="dca_extreme_pct",
                                    help="How far the price must drop from its recent high (within lookback window) to trigger the extreme multiplier tier.")

            lookback = st.slider("Lookback Days", 5, 90,
                                 int(dt.get("lookback_days", 30)),
                                 step=5, key="dca_lookback",
                                 help="Rolling window (in trading days) used to determine the 'recent high' for dip detection. 30 days ≈ 1.5 calendar months.")

            ov["dip_thresholds"] = {
                "mild_drop_pct": mild,
                "strong_drop_pct": strong,
                "extreme_drop_pct": extreme,
                "lookback_days": lookback,
            }

            st.markdown("**Multipliers** (how much to scale base amount)")
            c1, c2, c3 = st.columns(3)
            with c1:
                m_mild = st.slider("Mild Multiplier", 1.0, 5.0,
                                   float(ml.get("mild_dip", 1.5)),
                                   step=0.1, key="dca_m_mild",
                                   help="How much to scale the base amount on a mild dip. E.g., 1.5× on a $500 base = $750 that period.")
            with c2:
                m_strong = st.slider("Strong Multiplier", 1.0, 5.0,
                                     float(ml.get("strong_dip", 2.0)),
                                     step=0.1, key="dca_m_strong",
                                     help="How much to scale the base amount on a strong dip. E.g., 2.0× on a $500 base = $1,000 that period.")
            with c3:
                m_extreme = st.slider("Extreme Multiplier", 1.0, 5.0,
                                      float(ml.get("extreme_dip", 3.0)),
                                      step=0.1, key="dca_m_extreme",
                                      help="How much to scale the base amount on an extreme dip. E.g., 3.0× on a $500 base = $1,500 that period.")

            ov["multipliers"] = {
                "normal": 1.0,
                "mild_dip": m_mild,
                "strong_dip": m_strong,
                "extreme_dip": m_extreme,
            }

        with st.expander("Safety Gates"):
            sf = dca.get("safety", {})
            max_mult = st.slider("Max Multiplier Cap", 1.0, 5.0,
                                 float(sf.get("max_multiplier", 3.0)),
                                 step=0.5, key="dca_max_mult",
                                 help="Absolute ceiling on any multiplier, regardless of how many signals fire.")
            max_alloc = st.number_input(
                "Max Period Allocation ($)",
                min_value=100, max_value=100_000, step=100,
                value=int(sf.get("max_period_allocation", 1500)),
                key="dca_max_alloc",
                help="Hard dollar cap per period — even if multiplier × base exceeds this, allocation is capped here.",
            )
            skip_breakaway = st.checkbox(
                "Skip overweight on breakaway-down gaps",
                value=bool(sf.get("skip_breakaway_gaps", True)),
                key="dca_skip_brkwy",
                help="When enabled, suppresses overweighting if a breakaway-down gap is detected (likely a trend change, not a temporary dip).",
            )
            min_vol = st.slider("Min Volume Ratio", 0.0, 2.0,
                                float(sf.get("min_volume_ratio", 0.5)),
                                step=0.1, key="dca_min_vol",
                                help="If current volume / average volume is below this ratio, the dip is in low-liquidity conditions — overweighting is suppressed.")
            ov["safety"] = {
                "max_multiplier": max_mult,
                "max_period_allocation": max_alloc,
                "skip_breakaway_gaps": skip_breakaway,
                "min_volume_ratio": min_vol,
            }

    if mode == "score_integrated":
        with st.expander("Score Integration Thresholds"):
            sc = dca.get("score_thresholds", {})
            buy_zone = st.slider("Buy Zone Below (score)", 0.0, 10.0,
                                 float(sc.get("buy_zone_below", 3.5)),
                                 step=0.5, key="dca_buy_zone",
                                 help="Technical composite score threshold. Scores below this are considered 'buy zone' and boost the multiplier.")
            rsi_os = st.slider("Oversold RSI", 10.0, 50.0,
                               float(sc.get("oversold_rsi", 30.0)),
                               step=1.0, key="dca_rsi_os",
                               help="RSI below this value triggers an additional multiplier boost (classic oversold signal).")
            bb_low = st.slider("BB Percentile Low", 1.0, 30.0,
                               float(sc.get("bb_percentile_low", 10.0)),
                               step=1.0, key="dca_bb_low",
                               help="Bollinger Band percentile below this value triggers an additional boost (price near lower band).")
            ov["score_thresholds"] = {
                "buy_zone_below": buy_zone,
                "oversold_rsi": rsi_os,
                "bb_percentile_low": bb_low,
            }

    return ov


def _create_dca_equity_chart(
    dca_result: DCAResult,
    bt_result: BacktestResult | None = None,
) -> go.Figure:
    """Build a Plotly equity chart for DCA results, optionally overlaying
    the active strategy equity curve for comparison."""
    fig = go.Figure()

    # DCA portfolio value
    if dca_result.equity_curve:
        dates = [e["date"] for e in dca_result.equity_curve]
        values = [e["value"] for e in dca_result.equity_curve]
        invested = [e["invested"] for e in dca_result.equity_curve]

        fig.add_trace(go.Scatter(
            x=dates, y=values,
            name="DCA Portfolio Value",
            line=dict(color="#2196F3", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=invested,
            name="Total Invested",
            line=dict(color="#9E9E9E", width=1, dash="dash"),
        ))

    # Active strategy overlay
    if bt_result and bt_result.equity_curve:
        bt_dates = [e["date"] for e in bt_result.equity_curve]
        bt_values = [e["equity"] for e in bt_result.equity_curve]
        fig.add_trace(go.Scatter(
            x=bt_dates, y=bt_values,
            name="Active Strategy Equity",
            line=dict(color="#FF9800", width=2, dash="dot"),
        ))

    # Mark dip purchases
    dip_buys = [p for p in dca_result.purchases if p.multiplier > 1.0]
    if dip_buys:
        fig.add_trace(go.Scatter(
            x=[p.date for p in dip_buys],
            y=[p.portfolio_value for p in dip_buys],
            mode="markers",
            name="Dip Purchases",
            marker=dict(
                color="#4CAF50", size=10, symbol="triangle-up",
                line=dict(width=1, color="white"),
            ),
            text=[f"{p.tier} ({p.multiplier}x, -{p.dip_pct}%)" for p in dip_buys],
            hoverinfo="text+x+y",
        ))

    fig.update_layout(
        title="DCA Equity Curve",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        hovermode="x unified",
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _render_dca_metrics(result: DCAResult) -> None:
    """Render DCA performance metrics in a 4-column layout."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Invested", f"${result.total_invested:,.2f}")
        st.metric("Final Value", f"${result.final_value:,.2f}")
        st.metric("Net Profit", f"${result.final_value - result.total_invested:,.2f}")
    with c2:
        st.metric("Total Return", f"{result.total_return_pct:+.2f}%")
        st.metric("Annualized Return", f"{result.annualized_return_pct:+.2f}%")
        st.metric("Max Drawdown", f"-{result.max_drawdown_pct:.2f}%")
    with c3:
        st.metric("Avg Cost Basis", f"${result.avg_cost_basis:,.2f}")
        st.metric("Current Price", f"${result.current_price:,.2f}")
        basis_vs_price = (
            (result.current_price - result.avg_cost_basis) / result.avg_cost_basis * 100
            if result.avg_cost_basis > 0 else 0.0
        )
        st.metric("Price vs Basis", f"{basis_vs_price:+.2f}%")
    with c4:
        st.metric("Total Purchases", str(result.num_purchases))
        st.metric("Dip Purchases", str(result.num_dip_purchases))
        st.metric("Avg Multiplier", f"{result.avg_multiplier:.2f}x")

    if result.drip_shares > 0:
        c1, c2, c3, _ = st.columns(4)
        with c1:
            st.metric("Dividends Reinvested", f"${result.total_dividends:,.2f}")
        with c2:
            st.metric("DRIP Shares", f"{result.drip_shares:.4f}")
        with c3:
            st.metric("Total Shares", f"{result.total_shares:.4f}")


def _render_dca_purchase_log(result: DCAResult) -> None:
    """Render the DCA purchase log as a dataframe."""
    import pandas as pd
    if not result.purchases:
        st.info("No purchases to display.")
        return

    rows = []
    for p in result.purchases:
        rows.append({
            "Date": p.date,
            "Price": f"${p.price:,.2f}",
            "Dip %": f"{p.dip_pct:.1f}%",
            "Tier": p.tier.replace("_", " ").title(),
            "Multiplier": f"{p.multiplier:.1f}x",
            "Amount": f"${p.amount:,.2f}",
            "Shares": f"{p.shares:.4f}",
            "Cumulative Shares": f"{p.cumulative_shares:.4f}",
            "Cumulative Invested": f"${p.cumulative_invested:,.2f}",
            "Portfolio Value": f"${p.portfolio_value:,.2f}",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True, height=400)


def _render_dca_comparison(dca_result: DCAResult, bt_result: BacktestResult) -> None:
    """Render a side-by-side comparison table: DCA vs Active Strategy."""
    import pandas as pd

    dca_profit = dca_result.final_value - dca_result.total_invested
    active_profit = bt_result.final_equity - bt_result.initial_cash

    rows = [
        ("Starting Capital", f"${dca_result.total_invested:,.2f} (accumulated)",
         f"${bt_result.initial_cash:,.2f} (lump sum)"),
        ("Final Value", f"${dca_result.final_value:,.2f}",
         f"${bt_result.final_equity:,.2f}"),
        ("Net Profit", f"${dca_profit:,.2f}", f"${active_profit:,.2f}"),
        ("Total Return", f"{dca_result.total_return_pct:+.2f}%",
         f"{bt_result.total_return_pct:+.2f}%"),
        ("Annualized Return", f"{dca_result.annualized_return_pct:+.2f}%",
         f"{bt_result.annualized_return_pct:+.2f}%"),
        ("Max Drawdown", f"-{dca_result.max_drawdown_pct:.2f}%",
         f"-{bt_result.max_drawdown_pct:.2f}%"),
        ("Number of Trades", f"{dca_result.num_purchases} purchases",
         f"{bt_result.total_trades} round-trips"),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "DCA Strategy", "Active Strategy"])
    st.dataframe(df, width="stretch", hide_index=True)


def render_dca_section(
    dca_result: DCAResult,
    bt_result: BacktestResult | None = None,
) -> None:
    """Render the full DCA backtest results section."""

    # ── Performance summary ──────────────────────────────────────────────
    mode_label = _DCA_MODE_LABELS.get(dca_result.mode, dca_result.mode)
    st.subheader(f"DCA Results — {mode_label}")
    st.info(
        f"**{dca_result.frequency.title()}** purchases of "
        f"**${dca_result.purchases[0].amount if dca_result.purchases else 0:,.0f}** base "
        f"over **{dca_result.period}** "
        f"({dca_result.num_purchases} purchases, "
        f"{dca_result.num_dip_purchases} dip buys)"
    )

    _render_dca_metrics(dca_result)

    # ── Equity chart ─────────────────────────────────────────────────────
    st.subheader("Equity Curve")
    eq_fig = _create_dca_equity_chart(dca_result, bt_result)
    st.plotly_chart(eq_fig, width="stretch")

    # ── Comparison vs active strategy ────────────────────────────────────
    if bt_result is not None:
        with st.expander("DCA vs Active Strategy Comparison"):
            _render_dca_comparison(dca_result, bt_result)

    # ── Purchase log ─────────────────────────────────────────────────────
    with st.expander(f"Purchase Log ({dca_result.num_purchases} purchases)"):
        _render_dca_purchase_log(dca_result)

    # ── Dividend events ──────────────────────────────────────────────────
    if dca_result.dividend_events:
        with st.expander(f"Dividend Events ({len(dca_result.dividend_events)} payments)"):
            import pandas as pd
            div_rows = []
            for d in dca_result.dividend_events:
                div_rows.append({
                    "Date": d["date"],
                    "Per Share": f"${d['per_share']:.4f}",
                    "Total Amount": f"${d['total_amount']:,.2f}",
                    "DRIP Shares": f"{d['shares_acquired']:.4f}",
                })
            df = pd.DataFrame(div_rows)
            st.dataframe(df, width="stretch", hide_index=True, height=300)


# ---------------------------------------------------------------------------
# Scanner tab
# ---------------------------------------------------------------------------

SCANNER_UNIVERSES = ["dow30", "nasdaq100", "sp500", "tsx60", "ca_dividend_etfs"]
SCANNER_UNIVERSE_LABELS = {
    "dow30": "Dow 30",
    "nasdaq100": "NASDAQ 100",
    "sp500": "S&P 500",
    "tsx60": "S&P/TSX 60",
    "ca_dividend_etfs": "Canadian Dividend ETFs",
}

def _signal_color(signal: str) -> str:
    """Return hex color for a BUY/SELL/HOLD signal."""
    return {
        "BUY": COLOR_BULLISH,
        "SELL": COLOR_BEARISH,
        "HOLD": COLOR_NEUTRAL,
    }.get(signal, "#aaa")


def _confidence_dots(confidence: str) -> str:
    """Return dot-style confidence indicator."""
    if confidence == "high":
        return "●●●"
    elif confidence == "medium":
        return "●●○"
    return "●○○"


def _build_scanner_results_html(
    results: list[ScanResult],
    title: str,
    signal_type: str,
    show_mt: bool = False,
) -> str:
    """Build an HTML table for scanner results (BUY or SELL group)."""
    if not results:
        return f"<p style='color:#666;padding:8px 0;'>No {signal_type} signals found.</p>"

    title_color = _signal_color(signal_type)

    mt_headers = ""
    if show_mt:
        mt_headers = (
            "<th style='width:80px;'>MT Score</th>"
            "<th>MT Signal</th>"
            "<th>Daily</th>"
            "<th>Weekly</th>"
            "<th>Monthly</th>"
            "<th>Agreement</th>"
        )

    header = (
        "<tr>"
        "<th style='width:30px;'>#</th>"
        "<th>Ticker</th>"
        "<th>Signal</th>"
        "<th>Confidence</th>"
        "<th>Score</th>"
        "<th style='width:80px;'>Trend</th>"
        "<th style='width:80px;'>Contrarian</th>"
        "<th style='width:80px;'>Pattern</th>"
        "<th style='text-align:right;'>Price</th>"
        "<th>Regime</th>"
        "<th>Sub-Type</th>"
        f"{mt_headers}"
        "</tr>"
    )

    rows_html = []
    for i, r in enumerate(results, 1):
        sig_color = _signal_color(r.signal)
        conf_color = {
            "high": COLOR_BULLISH, "medium": COLOR_NEUTRAL, "low": "#666",
        }.get(r.confidence, "#666")

        # Dominant group indicator
        dom = r.dominant_group if hasattr(r, "dominant_group") else "none"
        trend_label = f"{r.trend_score:.1f}" if hasattr(r, "trend_score") else "—"
        contr_label = f"{r.contrarian_score:.1f}" if hasattr(r, "contrarian_score") else "—"
        trend_style = "font-weight:700;" if dom == "trend" else ""
        contr_style = "font-weight:700;" if dom == "contrarian" else ""

        mt_cells = ""
        if show_mt:
            agreement_color = {
                "aligned": COLOR_BULLISH,
                "mixed": COLOR_NEUTRAL,
                "conflicting": COLOR_BEARISH,
            }.get(r.mt_agreement, "#666")

            def _mt_signal_cell(sig: str) -> str:
                if not sig:
                    return "<td style='color:#555;'>—</td>"
                c = _signal_color(sig)
                return f"<td style='color:{c};font-weight:700;'>{sig}</td>"

            mt_score_html = (
                score_bar_html(r.mt_aggregated_score, width=60)
                if r.mt_aggregated_score > 0 else "—"
            )

            mt_cells = (
                f"<td>{mt_score_html}</td>"
                f"{_mt_signal_cell(r.mt_aggregated_signal)}"
                f"{_mt_signal_cell(r.mt_daily_signal)}"
                f"{_mt_signal_cell(r.mt_weekly_signal)}"
                f"{_mt_signal_cell(r.mt_monthly_signal)}"
                f"<td style='color:{agreement_color};font-weight:600;'>"
                f"{r.mt_agreement.capitalize() if r.mt_agreement else '—'}</td>"
            )

        row = (
            f"<tr>"
            f"<td style='color:#888;'>{i}</td>"
            f"<td style='font-weight:700;'>{r.ticker}</td>"
            f"<td style='color:{sig_color};font-weight:700;'>{r.signal}</td>"
            f"<td style='color:{conf_color};font-size:0.75rem;'>"
            f"{_confidence_dots(r.confidence)}</td>"
            f"<td>{score_bar_html(r.effective_score, width=80)}</td>"
            f"<td style='{trend_style}'>"
            f"{score_bar_html(r.trend_score, width=60) if hasattr(r, 'trend_score') else '—'}</td>"
            f"<td style='{contr_style}'>"
            f"{score_bar_html(r.contrarian_score, width=60) if hasattr(r, 'contrarian_score') else '—'}</td>"
            f"<td>{score_bar_html(r.pattern_score, width=60)}</td>"
            f"<td style='text-align:right;font-family:monospace;'>"
            f"${r.price:,.2f}</td>"
            f"<td style='color:#aaa;'>{r.regime_label or '—'}</td>"
            f"<td style='color:#aaa;'>{r.sub_type_label or '—'}</td>"
            f"{mt_cells}"
            f"</tr>"
        )
        rows_html.append(row)

    style = (
        "<style>"
        ".scanner-table { width:100%; border-collapse:collapse; font-size:0.82rem; }"
        ".scanner-table th { text-align:left; padding:6px 8px; border-bottom:2px solid #444; "
        "  color:#aaa; font-weight:600; }"
        ".scanner-table td { padding:5px 8px; border-bottom:1px solid #2a2a2a; color:#ddd; "
        "  vertical-align:middle; }"
        ".scanner-table tr:hover td { background:#1a1d2e; }"
        "</style>"
    )

    return (
        f"{style}"
        f"<div style='border-left:3px solid {title_color};padding-left:12px;"
        f"margin:12px 0 4px 0;'>"
        f"<span style='color:{title_color};font-weight:700;font-size:1rem;'>"
        f"{title}</span></div>"
        f"<div style='overflow-x:auto;'>"
        f'<table class="scanner-table">'
        f"<thead>{header}</thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        f"</table></div>"
    )


def render_scanner() -> None:
    """Render the stock universe scanner tab."""
    st.header("Stock Universe Scanner")

    # ── Scan mode selector ────────────────────────────────────────────────
    scan_mode = st.radio(
        "Scan Mode",
        ["Technical", "Dividend"],
        index=0,
        horizontal=True,
        key="scan_mode",
        help="**Technical**: full indicator + pattern analysis with BUY/SELL/HOLD signals. "
             "**Dividend**: rank stocks by dividend quality (yield, growth, consistency, streak).",
    )

    if scan_mode == "Dividend":
        _render_dividend_scanner()
    else:
        _render_technical_scanner()


def _render_dividend_scanner() -> None:
    """Render the dividend scanner UI."""
    st.markdown(
        "Scan a stock universe for **dividend quality** — ranked by yield, "
        "growth rate, payout consistency, and increase streak."
    )

    cfg = _get_config()
    div_cfg = dict(cfg.section("dividend"))

    # ── Controls row 1: universe, top N, workers ──────────────────────────
    col_univ, col_top, col_workers = st.columns([2, 1, 1])

    with col_univ:
        universe_options = SCANNER_UNIVERSES
        universe_labels = [
            f"{SCANNER_UNIVERSE_LABELS[u]} ({len(load_universe(u))} tickers)"
            for u in universe_options
        ]
        universe_idx = st.selectbox(
            "Universe",
            range(len(universe_options)),
            format_func=lambda i: universe_labels[i],
            index=0,
            key="div_scanner_universe",
        )
        universe = universe_options[universe_idx]

    with col_top:
        top_n = st.number_input(
            "Top N",
            min_value=1,
            max_value=100,
            value=20,
            step=1,
            key="div_scanner_top_n",
        )

    with col_workers:
        workers = st.number_input(
            "Workers",
            min_value=1,
            max_value=16,
            value=int(div_cfg.get("max_workers", 8)),
            step=1,
            key="div_scanner_workers",
        )

    # ── Sidebar-style controls: thresholds & weights ──────────────────────
    with st.expander("Dividend Thresholds & Weights", expanded=False):
        tc1, tc2, tc3 = st.columns(3)

        with tc1:
            st.markdown("**Minimum Thresholds**")
            min_yield = st.slider(
                "Minimum Yield (%)",
                min_value=0.0,
                max_value=10.0,
                value=float(div_cfg.get("min_yield", 0.01)) * 100,
                step=0.25,
                key="div_min_yield",
                help="Exclude stocks with yield below this percentage.",
            ) / 100.0

            max_yield = st.slider(
                "Maximum Yield (%)",
                min_value=5.0,
                max_value=30.0,
                value=float(div_cfg.get("max_yield", 0.15)) * 100,
                step=0.5,
                key="div_max_yield",
                help="Exclude stocks with yield above this (likely distressed).",
            ) / 100.0

            min_consistency = st.slider(
                "Minimum Consistency",
                min_value=0.0,
                max_value=1.0,
                value=float(div_cfg.get("min_consistency", 0.50)),
                step=0.05,
                key="div_min_consistency",
                help="Fraction of years that must have paid a dividend.",
            )

            min_streak = st.number_input(
                "Minimum Streak (years)",
                min_value=0,
                max_value=50,
                value=int(div_cfg.get("min_streak", 0)),
                step=1,
                key="div_min_streak",
            )

        with tc2:
            st.markdown("**Scoring Weights**")
            w_yield = st.slider(
                "Yield Weight",
                min_value=0.0,
                max_value=1.0,
                value=float(div_cfg.get("weights", {}).get("yield", 0.30)),
                step=0.05,
                key="div_w_yield",
            )
            w_growth = st.slider(
                "Growth Weight",
                min_value=0.0,
                max_value=1.0,
                value=float(div_cfg.get("weights", {}).get("growth", 0.30)),
                step=0.05,
                key="div_w_growth",
            )
            w_consistency = st.slider(
                "Consistency Weight",
                min_value=0.0,
                max_value=1.0,
                value=float(div_cfg.get("weights", {}).get("consistency", 0.20)),
                step=0.05,
                key="div_w_consistency",
            )
            w_streak = st.slider(
                "Streak Weight",
                min_value=0.0,
                max_value=1.0,
                value=float(div_cfg.get("weights", {}).get("streak", 0.20)),
                step=0.05,
                key="div_w_streak",
            )

        with tc3:
            st.markdown("**Data Parameters**")
            cagr_years = st.number_input(
                "CAGR Years",
                min_value=2,
                max_value=20,
                value=int(div_cfg.get("cagr_years", 5)),
                step=1,
                key="div_cagr_years",
                help="Years of history for dividend growth rate calculation.",
            )
            consistency_years = st.number_input(
                "Consistency Years",
                min_value=2,
                max_value=30,
                value=int(div_cfg.get("consistency_years", 10)),
                step=1,
                key="div_consistency_years",
                help="Years of history for payout consistency evaluation.",
            )

    # ── Build override config from UI controls ────────────────────────────
    ui_div_cfg = dict(div_cfg)  # start from config.yaml values
    ui_div_cfg["min_yield"] = min_yield
    ui_div_cfg["max_yield"] = max_yield
    ui_div_cfg["min_consistency"] = min_consistency
    ui_div_cfg["min_streak"] = min_streak
    ui_div_cfg["cagr_years"] = cagr_years
    ui_div_cfg["consistency_years"] = consistency_years
    ui_div_cfg["max_workers"] = int(workers)
    ui_div_cfg["weights"] = {
        "yield": w_yield,
        "growth": w_growth,
        "consistency": w_consistency,
        "streak": w_streak,
    }

    # ── Run button ────────────────────────────────────────────────────────
    run_scan = st.button("Run Dividend Scan", type="primary", key="run_div_scan_btn")

    if run_scan:
        tickers = load_universe(universe)
        total = len(tickers)

        progress_bar = st.progress(0, text=f"Scanning {SCANNER_UNIVERSE_LABELS[universe]} for dividends...")
        status_text = st.empty()

        def _on_progress(
            completed: int, total_: int, ticker: str, result: DividendScanResult | None,
        ) -> None:
            pct = completed / total_
            if result and not result.error:
                status_text.markdown(
                    f"**[{completed}/{total_}]** {ticker} "
                    f"— yield {result.current_yield*100:.1f}%, "
                    f"score {result.dividend_score:.1f}"
                )
            elif result and result.error:
                status_text.markdown(
                    f"**[{completed}/{total_}]** {ticker} — *error*"
                )
            progress_bar.progress(pct, text=f"Scanning... {completed}/{total_}")

        import time as _time

        scanner = DividendScanner(
            tickers=tickers,
            div_cfg=ui_div_cfg,
            on_progress=_on_progress,
        )

        t0 = _time.time()
        scanner.run()
        elapsed = _time.time() - t0

        progress_bar.empty()
        status_text.empty()

        st.session_state["div_scanner_results"] = scanner.results
        st.session_state["div_scanner_summary"] = scanner.summary()
        st.session_state["div_scanner_elapsed"] = elapsed
        st.session_state["div_scanner_cfg"] = ui_div_cfg

    # ── Display results ───────────────────────────────────────────────────
    if "div_scanner_results" in st.session_state:
        results_all: list[DividendScanResult] = st.session_state["div_scanner_results"]
        summary = st.session_state["div_scanner_summary"]
        elapsed = st.session_state["div_scanner_elapsed"]
        stored_cfg = st.session_state.get("div_scanner_cfg", ui_div_cfg)

        # Summary metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Scanned", f"{summary['scanned']}")
        c2.metric("Passed Filters", f"{summary['passed_filters']}")
        c3.metric("Avg Yield", f"{summary['avg_yield']*100:.1f}%")
        c4.metric("Avg Score", f"{summary['avg_score']:.1f}")
        c5.metric("Time", f"{elapsed:.1f}s")

        if summary["errors"] > 0:
            st.warning(f"{summary['errors']} ticker(s) failed during scan.")

        # Build ranked list using stored config for filtering
        temp_scanner = DividendScanner([], stored_cfg)
        temp_scanner._results = results_all
        top_results = temp_scanner.top(top_n)

        st.markdown(
            _build_dividend_results_html(top_results, f"Top {len(top_results)} Dividend Stocks"),
            unsafe_allow_html=True,
        )

        # Expandable: errors
        err_results = [r for r in results_all if r.error]
        if err_results:
            with st.expander(f"Failed Tickers ({len(err_results)})", expanded=False):
                for r in err_results:
                    st.text(f"{r.ticker}: {r.error[:120]}")


def _build_dividend_results_html(
    results: list[DividendScanResult],
    title: str,
) -> str:
    """Build an HTML table for dividend scanner results."""
    if not results:
        return "<p style='color:#666;padding:8px 0;'>No dividend stocks passed the filters.</p>"

    header = (
        "<tr>"
        "<th style='width:30px;'>#</th>"
        "<th>Ticker</th>"
        "<th>Name</th>"
        "<th>Sector</th>"
        "<th style='text-align:right;'>Price</th>"
        "<th style='text-align:right;'>Yield</th>"
        "<th style='text-align:right;'>Annual Dividend</th>"
        "<th style='text-align:right;'>CAGR</th>"
        "<th style='text-align:right;'>Consistency</th>"
        "<th style='text-align:right;'>Streak</th>"
        "<th>Yield Score</th>"
        "<th>Growth Score</th>"
        "<th>Consistency Score</th>"
        "<th>Streak Score</th>"
        "<th>Composite Score</th>"
        "</tr>"
    )

    rows_html = []
    for i, r in enumerate(results, 1):
        cagr_str = f"{r.dividend_cagr*100:+.1f}%" if r.dividend_cagr is not None else "N/A"
        cagr_color = (
            COLOR_BULLISH if r.dividend_cagr is not None and r.dividend_cagr > 0.03
            else COLOR_BEARISH if r.dividend_cagr is not None and r.dividend_cagr < 0
            else "#aaa"
        )

        # Yield color coding
        if r.current_yield >= 0.06:
            yield_color = COLOR_BULLISH
        elif r.current_yield >= 0.03:
            yield_color = COLOR_NEUTRAL
        else:
            yield_color = "#aaa"

        # Currency-aware price formatting
        price_prefix = "C$" if r.currency == "CAD" else "$"

        row = (
            f"<tr>"
            f"<td style='color:#888;'>{i}</td>"
            f"<td style='font-weight:700;'>{r.ticker}</td>"
            f"<td style='color:#ccc;max-width:180px;overflow:hidden;text-overflow:ellipsis;"
            f"white-space:nowrap;'>{r.name}</td>"
            f"<td style='color:#888;'>{r.sector}</td>"
            f"<td style='text-align:right;font-family:monospace;'>"
            f"{price_prefix}{r.current_price:,.2f}</td>"
            f"<td style='text-align:right;color:{yield_color};font-weight:600;'>"
            f"{r.current_yield*100:.2f}%</td>"
            f"<td style='text-align:right;font-family:monospace;'>"
            f"{price_prefix}{r.annual_dividend:.2f}</td>"
            f"<td style='text-align:right;color:{cagr_color};'>{cagr_str}</td>"
            f"<td style='text-align:right;'>{r.payout_consistency*100:.0f}%</td>"
            f"<td style='text-align:right;font-weight:600;'>{r.increase_streak}yr</td>"
            f"<td>{score_bar_html(r.yield_score, width=50)}</td>"
            f"<td>{score_bar_html(r.growth_score, width=50)}</td>"
            f"<td>{score_bar_html(r.consistency_score, width=50)}</td>"
            f"<td>{score_bar_html(r.streak_score, width=50)}</td>"
            f"<td>{score_bar_html(r.dividend_score, width=70)}</td>"
            f"</tr>"
        )
        rows_html.append(row)

    style = (
        "<style>"
        ".div-scanner-table { width:100%; border-collapse:collapse; font-size:0.80rem; }"
        ".div-scanner-table th { text-align:left; padding:6px 8px; border-bottom:2px solid #444; "
        "  color:#aaa; font-weight:600; white-space:nowrap; }"
        ".div-scanner-table td { padding:5px 8px; border-bottom:1px solid #2a2a2a; color:#ddd; "
        "  vertical-align:middle; }"
        ".div-scanner-table tr:hover td { background:#1a1d2e; }"
        "</style>"
    )

    title_color = COLOR_BULLISH

    return (
        f"{style}"
        f"<div style='border-left:3px solid {title_color};padding-left:12px;"
        f"margin:12px 0 4px 0;'>"
        f"<span style='color:{title_color};font-weight:700;font-size:1rem;'>"
        f"{title}</span></div>"
        f"<div style='overflow-x:auto;'>"
        f'<table class="div-scanner-table">'
        f"<thead>{header}</thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        f"</table></div>"
    )


def _render_technical_scanner() -> None:
    """Render the original technical scanner UI."""
    st.markdown(
        "Scan a predefined stock universe through the full analysis + strategy "
        "pipeline to surface the top **BUY** and **SELL** candidates."
    )

    # ── Controls ──────────────────────────────────────────────────────────
    col_univ, col_period, col_top, col_workers = st.columns([2, 1, 1, 1])

    with col_univ:
        universe_options = SCANNER_UNIVERSES
        universe_labels = [
            f"{SCANNER_UNIVERSE_LABELS[u]} ({len(load_universe(u))} tickers)"
            for u in universe_options
        ]
        universe_idx = st.selectbox(
            "Universe",
            range(len(universe_options)),
            format_func=lambda i: universe_labels[i],
            index=0,
            key="scanner_universe",
        )
        universe = universe_options[universe_idx]

    with col_period:
        period = st.selectbox(
            "Period",
            CLASSIFICATION_PERIODS,
            index=CLASSIFICATION_PERIODS.index(DEFAULT_CLASSIFICATION_PERIOD),
            key="scanner_period",
        )

    with col_top:
        top_n = st.number_input(
            "Top N",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            key="scanner_top_n",
        )

    with col_workers:
        workers = st.number_input(
            "Workers",
            min_value=1,
            max_value=16,
            value=8,
            step=1,
            key="scanner_workers",
            help="Number of parallel threads for fetching data.",
        )

    enable_mt = st.checkbox(
        "Enable multi-timeframe confirmation (daily + weekly + monthly)",
        value=False,
        key="scanner_multi_timeframe",
        help="Runs analysis on daily, weekly, and monthly timeframes for each ticker. "
             "Significantly slower but provides cross-timeframe signal confirmation.",
    )

    # ── Run button ────────────────────────────────────────────────────────
    run_scan = st.button("Run Scan", type="primary", key="run_scan_btn")

    # ── Execute scan ──────────────────────────────────────────────────────
    if run_scan:
        tickers = load_universe(universe)
        total = len(tickers)

        progress_bar = st.progress(0, text=f"Scanning {SCANNER_UNIVERSE_LABELS[universe]}...")
        status_text = st.empty()

        # Accumulate results for display
        scan_results: list[ScanResult] = []
        errors: list[ScanResult] = []

        def _on_progress(
            completed: int, total_: int, ticker: str, result: ScanResult | None
        ) -> None:
            pct = completed / total_
            if result and not result.error:
                sig = result.signal
                status_text.markdown(
                    f"**[{completed}/{total_}]** {ticker} "
                    f"→ **{sig}** ({result.effective_score:.2f})"
                )
            elif result and result.error:
                status_text.markdown(
                    f"**[{completed}/{total_}]** {ticker} → *error*"
                )
            progress_bar.progress(pct, text=f"Scanning... {completed}/{total_}")

        cfg = _get_config()
        scanner = Scanner(
            universe=universe,
            period=period,
            max_workers=int(workers),
            cfg=cfg,
            on_progress=_on_progress,
            multi_timeframe=enable_mt,
        )

        import time as _time

        t0 = _time.time()
        scanner.run()
        elapsed = _time.time() - t0

        # Clear progress
        progress_bar.empty()
        status_text.empty()

        # Store results in session state so they persist across reruns
        st.session_state["scanner_results"] = scanner.results
        st.session_state["scanner_summary"] = scanner.summary()
        st.session_state["scanner_elapsed"] = elapsed
        st.session_state["scanner_mt_enabled"] = enable_mt

    # ── Display results (from session state) ──────────────────────────────
    if "scanner_results" in st.session_state:
        results_all = st.session_state["scanner_results"]
        summary = st.session_state["scanner_summary"]
        elapsed = st.session_state["scanner_elapsed"]
        display_n = st.session_state.get("scanner_top_n", 10)
        show_mt = st.session_state.get("scanner_mt_enabled", False)

        # Summary metrics
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Universe", f"{summary['universe']}")
        c2.metric("Scanned", f"{summary['scanned']}")
        c3.metric("BUY", f"{summary['buy_count']}")
        c4.metric("SELL", f"{summary['sell_count']}")
        c5.metric("HOLD", f"{summary['hold_count']}")
        c6.metric("Time", f"{elapsed:.1f}s")

        if summary["errors"] > 0:
            st.warning(f"{summary['errors']} ticker(s) failed during scan.")

        # Build ranked lists
        ok_results = [r for r in results_all if not r.error]
        if show_mt:
            # When MT is enabled, group by MT aggregated signal and rank by
            # a combination of MT score and agreement.
            mt_cfg = _get_config().section("multi_timeframe")
            _agreement_mult = mt_cfg.get("agreement_multipliers", {
                "aligned": 1.0, "mixed": 0.7, "conflicting": 0.4,
            })

            def _mt_sort_key(r: ScanResult) -> float:
                base = r.mt_aggregated_score if r.mt_aggregated_score > 0 else r.effective_score
                mult = _agreement_mult.get(r.mt_agreement, 0.7)
                return base * mult

            sort_key = _mt_sort_key
            signal_key = lambda r: r.mt_aggregated_signal or r.signal
        else:
            sort_key = lambda r: r.effective_score
            signal_key = lambda r: r.signal

        buys = sorted(
            [r for r in ok_results if signal_key(r) == "BUY"],
            key=sort_key,
            reverse=True,
        )[:display_n]
        sells = sorted(
            [r for r in ok_results if signal_key(r) == "SELL"],
            key=sort_key,
        )[:display_n]

        # Render tables (stacked vertically – too many columns for side-by-side)
        st.markdown(
            _build_scanner_results_html(
                buys, f"Top {len(buys)} BUY Signals", "BUY", show_mt=show_mt,
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            _build_scanner_results_html(
                sells, f"Top {len(sells)} SELL Signals", "SELL", show_mt=show_mt,
            ),
            unsafe_allow_html=True,
        )

        # Expandable: all HOLD signals
        holds = sorted(
            [r for r in ok_results if signal_key(r) == "HOLD"],
            key=sort_key,
            reverse=True,
        )
        if holds:
            with st.expander(f"All HOLD Signals ({len(holds)})", expanded=False):
                st.markdown(
                    _build_scanner_results_html(
                        holds, "HOLD Signals", "HOLD", show_mt=show_mt,
                    ),
                    unsafe_allow_html=True,
                )

        # Expandable: errors
        err_results = [r for r in results_all if r.error]
        if err_results:
            with st.expander(f"Failed Tickers ({len(err_results)})", expanded=False):
                for r in err_results:
                    st.text(f"{r.ticker}: {r.error[:120]}")


# ---------------------------------------------------------------------------
# Help / User Guide tab
# ---------------------------------------------------------------------------

def render_help() -> None:
    """Render the Help tab with architecture diagrams and decision-point explanations."""

    st.header("How It Works")
    st.caption(
        "End-to-end architecture of the stock analyzer — "
        "how analysis, backtesting, and scanning work under the hood."
    )

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1: STOCK ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("1. Stock Analysis Flow")
    st.markdown(
        "When you enter a ticker in the **Analyze** tab (or run `python main.py AAPL` "
        "on the CLI), this is what happens:"
    )

    st.markdown("""
```mermaid
flowchart TD
    A["Enter Ticker + Period"] --> B["Fetch OHLCV Data<br/><i>YahooFinanceProvider</i>"]
    B --> C{"--indicators<br/>flag set?"}
    C -- "Yes" --> D["Run Selected Indicators Only"]
    C -- "No (default)" --> E["Run All Indicators"]
    E --> F["Run All Pattern Detectors"]
    E --> G["Calculate S/R Levels"]
    E --> H["Classify Market Regime"]
    D --> I["Composite Indicator Score"]
    F --> J["Composite Pattern Score"]
    I --> K["Display Results"]
    J --> K
    G --> K
    H --> K

    style A fill:#1e3a5f,stroke:#4a90d9,color:#fff
    style B fill:#1e3a5f,stroke:#4a90d9,color:#fff
    style K fill:#2d5a1e,stroke:#6abf4b,color:#fff
```
""")

    with st.expander("Indicators (8 total) — what they measure and how they score"):
        st.markdown("""
Each indicator produces a **0 – 10 score** where 0 = extremely bearish, 5 = neutral, 10 = extremely bullish.

| Indicator | What It Measures | Scoring Logic |
|-----------|-----------------|---------------|
| **RSI** | Momentum oscillator (0-100) | RSI ≤ 30 → 9.0 (oversold/bullish). RSI ≥ 70 → 1.0 (overbought/bearish). Linear between. |
| **MACD** | Trend momentum via histogram | Histogram normalized by price (hist%). Strong bull (≥0.5%) → 8.5–10. Strong bear (≤-0.5%) → 0–1.5. Crossover bonus ±1.5. |
| **Bollinger Bands** | Price position within bands | %B (0=lower, 1=upper). Near lower band → 9.5 (oversold). Near upper band → 0.5 (overbought). |
| **Moving Averages** | Price vs SMA/EMA alignment | Points for: price above each MA (+1.5), bullish MA stacking (+1.0), golden cross (+2.0). Normalized to 0–10. |
| **Stochastic** | Momentum (%K oscillator) | %K ≤ 20 → 9.0 (oversold). %K ≥ 80 → 1.0 (overbought). Cross bonus ±1.0. |
| **ADX** | Trend strength + direction | DI spread sets direction (0–10). ADX multiplier pulls toward 5.0 in weak trends — ADX < 30 dampens signal to ±2.0 from neutral. |
| **Volume (OBV)** | Volume-price confirmation | Confirming + rising → 6.5–9.5. Confirming + falling → 0.5–3.5. Diverging → 5.0 (neutral). |
| **Fibonacci** | Proximity to Fib retracement levels | Near a Fib level → level-specific score. Otherwise: linear by range position (low=2.0, high=8.0). |
""")

    with st.expander("Pattern Detectors (5 total) — what they detect"):
        st.markdown("""
Each pattern produces a **0 – 10 score** where 5.0 = no signal (neutral baseline).

| Pattern | What It Detects | Key Logic |
|---------|----------------|-----------|
| **Gaps** | Price gaps between bars | Classified as **common** (0.3 weight), **runaway** (0.7, with-trend), **breakaway** (1.0, out of consolidation), or **exhaustion** (0.5, reversal warning). Uses BB width percentile for consolidation, total return + MA distance for exhaustion. |
| **Volume-Range** | Bar range vs volume correlation | Expansion bull (wide + high vol) → 8.0. Expansion bear → 2.0. Adjusted ±1.5 by recent bull/bear expansion bias. |
| **Candlesticks** | 18 classic patterns | Single-bar (doji, hammer, marubozu, etc.), two-bar (engulfing, harami, tweezers), three-bar (morning/evening star, soldiers/crows). Recency-weighted net signal. |
| **Spikes** | Sudden price moves (>N σ) | Confirmed spikes signal continuation. Unconfirmed "trap" spikes signal reversal (inverse). |
| **Inside/Outside** | Consolidation & breakout bars | Inside bars wait for breakout direction (no lookahead). Outside bars signal based on close direction, stronger as reversals. |
""")

    with st.expander("Composite Scoring — how individual scores become one number"):
        st.markdown("""
**Indicator Composite** has two weighting modes (set via `scoring.mode`):

- **Directional** (default): Groups indicators into **Trend** (MA, ADX, MACD), **Contrarian** (RSI, Stochastic, Bollinger), and **Neutral** (Volume, Fibonacci). The dominant group (biggest deviation from 5.0) gets 60% weight, secondary 25%, tertiary 15%. This lets trend and contrarian signals take turns leading.
- **Average**: Simple weighted average of all indicator scores.

After weighting, a **score spread factor** amplifies deviations from 5.0: `final = 5.0 + (raw - 5.0) × spread_factor`.

**Pattern Composite** is a straight weighted average of all pattern scores.

**Effective Score** combines both composites via `combination_mode`:

| Mode | Formula | When to use |
|------|---------|-------------|
| **weighted** (default) | `0.7 × indicator + 0.3 × pattern` | General purpose |
| **boost** | Indicator ± pattern deviation × boost_strength | When patterns should amplify, not override |
| **gate** | Indicator score, but BUY/SELL requires both indicator AND pattern to agree | Conservative — reduces false signals |
""")

    with st.expander("Market Regime Classification"):
        st.markdown("""
The regime classifier scores 4 regime types and picks the highest:

```mermaid
flowchart LR
    M["Compute Metrics<br/>ADX, ATR%, Return,<br/>Dir Changes, MA%"] --> S1["Score: Strong Trend"]
    M --> S2["Score: Mean Reverting"]
    M --> S3["Score: Volatile Choppy"]
    M --> S4["Score: Breakout Transition"]
    S1 --> W{"Highest<br/>Score Wins"}
    S2 --> W
    S3 --> W
    S4 --> W
    W --> R["Regime + Confidence"]
    R --> ST["Sub-Type via 2×2 Matrix"]

    style W fill:#5a3a1e,stroke:#d9a04a,color:#fff
```

**Regime types and what drives them:**
| Regime | Key Drivers |
|--------|------------|
| **Strong Trend** | High ADX (>30), consistent direction, strong total return, price persistently above/below MA |
| **Mean Reverting** | Low ADX (<25), many direction changes, price oscillating around MA |
| **Volatile Choppy** | High ATR%, many direction changes, inconsistent trend |
| **Breakout Transition** | Recent ADX spike, low rolling ADX but rising, early directional move |

**Sub-types** (2×2 matrix of volatility × return):
| | High |Return| Low |Return| |
|---|---|---|---|---|---|
| **High ATR%** | Explosive Mover | | Volatile Directionless | |
| **Low ATR%** | Steady Compounder | | Stagnant | |
""")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2: BACKTEST
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("2. Backtest Flow")
    st.markdown(
        "When you click **Run Quick Backtest** (or run `python main.py AAPL --backtest`), "
        "the engine replays history bar-by-bar, simulating trades."
    )

    st.markdown("""
```mermaid
flowchart TD
    A["Start Backtest"] --> B["Suitability Check"]
    B --> C{"Trading Mode?"}
    C -- "HOLD_ONLY" --> D["Stop — Stock Unsuitable"]
    C -- "LONG_ONLY" --> E["Engine Loop<br/><i>SELL signals blocked<br/>except to close longs</i>"]
    C -- "LONG_SHORT" --> F["Engine Loop<br/><i>Full signal set</i>"]

    E --> G["Warmup Phase<br/><i>No trading, build indicator history</i>"]
    F --> G
    G --> H{"Every N bars<br/>(rebalance)"}
    H -- "Rebalance bar" --> I["Recompute indicators,<br/>patterns, regime, S/R<br/>on trailing data only<br/><i>(no lookahead)</i>"]
    I --> J["Strategy.on_bar()"]
    H -- "Non-rebalance bar" --> K["Check Exit Triggers Only"]
    J --> L{"Signal?"}
    L -- "BUY" --> M["Open/Adjust Position"]
    L -- "SELL" --> N["Open Short / Close Long"]
    L -- "HOLD" --> K
    K --> O["Next Bar"]
    M --> O
    N --> O
    O --> H

    style A fill:#1e3a5f,stroke:#4a90d9,color:#fff
    style D fill:#5a1e1e,stroke:#d94a4a,color:#fff
    style G fill:#3a3a1e,stroke:#d9d94a,color:#fff
```
""")

    with st.expander("Suitability Assessment — how trading mode is determined"):
        st.markdown("""
Before the backtest starts, the engine evaluates whether the stock is suitable for active trading:

| Condition | Result | Rationale |
|-----------|--------|-----------|
| Avg volume < 100K **or** ATR% < 0.5% | **HOLD_ONLY** | Too illiquid or too flat for any active strategy |
| ADX < 25 **or** ATR% < 1% **or** Avg volume < 500K **or** % above 200-MA > 65% | **LONG_ONLY** | Moderate liquidity/trend — shorting is risky |
| All checks pass | **LONG_SHORT** | Sufficient liquidity, volatility, and trend for both directions |

These thresholds are configurable in the `suitability` section of config.yaml. The CLI `--mode` flag or config `mode_override` can force any mode.
""")

    with st.expander("Strategy Decision Tree — how on_bar() decides BUY / SELL / HOLD"):
        st.markdown("""
The `ScoreBasedStrategy.on_bar()` method runs a **10-stage decision pipeline** on each rebalance bar:

```mermaid
flowchart TD
    A["Receive bar context:<br/>ind_score, pat_score,<br/>regime, price, position"] --> B["NaN guard:<br/>replace NaN scores with 5.0"]
    B --> C{"Regime =<br/>Strong Trend?"}
    C -- "Yes + Holding" --> D["Strong Trend Hold:<br/>Stay in position unless<br/>extreme counter-signal<br/>(score ±1.5 beyond threshold)"]
    C -- "Yes + Flat" --> E["Strong Trend Entry:<br/>Require price distance<br/>from MA + min score.<br/>Cooldown after consecutive losses."]
    C -- "No" --> F["Compute Effective Score<br/>(weighted / boost / gate)"]
    E -- "Conditions not met" --> F
    F --> G["Map Score → Raw Signal<br/>(fixed thresholds or percentile)"]
    G --> H{"Regime =<br/>Breakout?"}
    H -- "Yes" --> I["Breakout Gate:<br/>Require large bar move +<br/>volume surge to confirm"]
    H -- "No" --> J["Global Directional Bias:<br/>Block SELL in strong uptrend<br/>Block BUY in strong downtrend"]
    I -- "Not confirmed" --> HOLD["→ HOLD"]
    I -- "Confirmed" --> J
    J --> K["Trading Mode Constraint:<br/>LONG_ONLY blocks new shorts"]
    K --> L{"Trend Confirmation<br/>enabled + flat?"}
    L -- "Yes" --> M["Price vs Trend MA:<br/>BUY blocked if below MA<br/>SELL blocked if above MA"]
    L -- "No / Grace period" --> N["Anti-Pyramiding:<br/>Block same-direction signal<br/>if already in position"]
    M --> N
    N --> O["Final Signal"]

    style A fill:#1e3a5f,stroke:#4a90d9,color:#fff
    style HOLD fill:#3a3a1e,stroke:#d9d94a,color:#fff
    style O fill:#2d5a1e,stroke:#6abf4b,color:#fff
```

**Key config knobs:**
- `threshold_mode`: `"fixed"` (score ≤ 3.5 = SELL, > 6.0 = BUY) or `"percentile"` (rolling window rank)
- `trend_confirm_enabled`: whether price must be on the "right side" of the trend MA
- `reentry_grace_bars`: bars after exit where trend confirmation is skipped (default 10)
- `hold_with_trend` / `ignore_score_entries`: strong-trend special handling (both default True)
- `global_trend_bias`: block counter-trend entries when total return > ±10%
""")

    with st.expander("Exit Triggers — checked every bar (not just rebalance bars)"):
        st.markdown("""
Exit triggers are checked in this priority order on **every bar**:

| Priority | Trigger | Condition | Config Keys |
|----------|---------|-----------|-------------|
| 1 | **Trailing Stop (Strong Trend)** | In strong trend regime: trail by ATR × multiplier from highest profit | `trailing_stop_atr_mult` |
| 2 | **Chandelier Exit** | Price drops ATR × multiplier from N-bar high (longs) or rises from N-bar low (shorts) | `chandelier_atr_mult`, `chandelier_period` |
| 3 | **S/R Stop** | Long: price breaks below nearest support × (1 - buffer). Short: price breaks above nearest resistance × (1 + buffer) | `support_stop_enabled`, `support_stop_buffer_pct` |
| 4 | **Fixed / ATR Stop-Loss** | Price moves against entry by stop_loss_pct or ATR × multiplier | `stop_loss_pct`, `atr_stop_mult` |
| 5 | **Take-Profit** | Price moves in favor of entry by take_profit_pct (disabled in strong trend if configured) | `take_profit_pct`, `disable_take_profit_in_strong_trend` |

The first trigger that fires wins — later triggers are not checked.
""")

    with st.expander("Commission Model"):
        st.markdown("""
Commissions are applied per trade leg (entry and exit separately):

- **Flat fee**: `commission_per_trade` (e.g., $1.00)
- **Percentage fee**: `commission_pct` × trade value (e.g., 0.1%)
- **Combination**: `commission_mode` = `"additive"` (flat + pct) or `"max"` (whichever is greater)

Default: $0 flat + 0.1% percentage, additive mode.
""")

    with st.expander("Performance Metrics — what the results show"):
        st.markdown("""
All metrics are computed on the **post-warmup** portion only (the flat warmup period is excluded):

| Metric | Description |
|--------|------------|
| **Total Return %** | `(final_equity - initial) / initial × 100` |
| **Annualized Return %** | Geometric annualization using post-warmup bar count and bars-per-year for the interval |
| **Max Drawdown %** | Largest peak-to-trough decline in the equity curve |
| **Win Rate %** | Percentage of closed trades with positive PnL |
| **Profit Factor** | Gross profit / gross loss (∞ if no losses) |
| **Sharpe Ratio** | `(mean_bar_return / std_bar_return) × √(bars_per_year)` — annualized risk-adjusted return |
| **Avg Trade PnL %** | Mean PnL percentage across all closed trades |
| **Best / Worst Trade** | Largest single-trade gain and loss |
| **Avg Bars Held** | Mean holding period in bars |
""")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 3: SCANNER / TOP 10
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("3. Scanner / Top 10 Flow")
    st.markdown(
        "When you run a scan in the **Scanner** tab, the engine analyzes every stock "
        "in a universe in parallel and ranks them by signal strength."
    )

    st.markdown("""
```mermaid
flowchart TD
    A["Select Universe<br/>(DOW 30 / NASDAQ 100 / S&P 500)"] --> B["Load Ticker List<br/>from .txt file"]
    B --> C["ThreadPoolExecutor<br/>(parallel, N workers)"]
    C --> D["Per Ticker: _scan_one()"]

    D --> E["Fetch OHLCV + Run Full Analysis<br/><i>(indicators + patterns + regime)</i>"]
    E --> F["Suitability Assessment<br/>→ Trading Mode"]
    F --> G["Create Strategy Instance"]
    G --> H{"Percentile<br/>mode?"}
    H -- "Yes" --> I["Seed score window<br/>with historical scores"]
    H -- "No" --> J["Call strategy.on_bar()<br/>with last bar context"]
    I --> J
    J --> K["Extract signal:<br/>BUY / SELL / HOLD"]
    K --> L["Compute effective_score<br/>+ confidence"]

    L --> M["Collect All Results"]
    M --> N["Rank: Top BUYs<br/><i>by effective_score ↓</i>"]
    M --> O["Rank: Top SELLs<br/><i>by effective_score ↑</i>"]

    style A fill:#1e3a5f,stroke:#4a90d9,color:#fff
    style C fill:#3a3a1e,stroke:#d9d94a,color:#fff
    style N fill:#2d5a1e,stroke:#6abf4b,color:#fff
    style O fill:#5a1e1e,stroke:#d94a4a,color:#fff
```
""")

    with st.expander("Ranking Logic — how the top 10 are selected"):
        st.markdown("""
**Effective score** is the single number that drives all ranking:

1. Each ticker gets a full analysis (all 8 indicators + 5 patterns)
2. The strategy's `on_bar()` runs once on the latest bar to produce a BUY/SELL/HOLD signal
3. The effective score is computed using the same combination mode as the strategy:
   - **Weighted**: `(0.7 × indicator_score + 0.3 × pattern_score)`
   - **Boost**: indicator ± pattern deviation × boost_strength
   - **Gate**: indicator_score (patterns used only as entry gate)

**Top BUYs** = filter signal == "BUY", sort by effective_score **descending** (strongest buy conviction first)

**Top SELLs** = filter signal == "SELL", sort by effective_score **ascending** (strongest sell conviction first)

**Confidence** is derived from distance to the nearest threshold:
- **High**: ≥ 1.5 points past the BUY/SELL threshold
- **Medium**: ≥ 0.5 points past
- **Low**: barely past the threshold
""")

    with st.expander("Multi-Timeframe Confirmation (optional)"):
        st.markdown("""
When enabled, each ticker is analyzed on **three timeframes** (daily, weekly, monthly) with configurable weights (default 50% / 30% / 20%).

The scanner checks whether all timeframes **agree**:
- **Aligned**: all timeframes give the same signal → full weight (×1.0)
- **Mixed**: some agree, some neutral → reduced weight (×0.7)
- **Conflicting**: timeframes disagree → heavily reduced (×0.4)

The aggregated MT score replaces the single-timeframe effective_score for ranking when MT mode is active.
""")

    with st.expander("Available Universes"):
        st.markdown("""
Universes are plain text files in `data/universes/`:

| Universe | File | Description |
|----------|------|-------------|
| **DOW 30** | `dow30.txt` | 30 Dow Jones Industrial Average components |
| **NASDAQ 100** | `nasdaq100.txt` | 100 largest NASDAQ-listed non-financial companies |
| **S&P 500** | `sp500.txt` | 500 largest US-listed companies |
| **S&P/TSX 60** | `tsx60.txt` | 60 largest Canadian stocks (Toronto Stock Exchange, `.TO` suffix) |
| **Canadian Dividend ETFs** | `ca_dividend_etfs.txt` | 17 popular TSX-listed dividend-focused ETFs |

You can add custom universes by placing a `.txt` file (one ticker per line, `#` for comments) in the same directory.
""")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 4: DIVIDEND SCANNER
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("4. Dividend Scanner Flow")
    st.markdown(
        "When you select **Dividend** mode in the Scanner tab, the engine fetches "
        "dividend data for every stock in the universe and ranks them by dividend quality."
    )

    st.markdown("""
```mermaid
flowchart TD
    A["Select Universe +<br/>Dividend Mode"] --> B["Load Ticker List"]
    B --> C["ThreadPoolExecutor<br/>(parallel, N workers)"]
    C --> D["Per Ticker:<br/>fetch_dividend_metrics()"]

    D --> E["yfinance: .info + .dividends"]
    E --> F["Compute Metrics"]
    F --> F1["Current Yield<br/><i>trailing 12m / price</i>"]
    F --> F2["Dividend CAGR<br/><i>N-year compound growth</i>"]
    F --> F3["Payout Consistency<br/><i>% of years with dividend</i>"]
    F --> F4["Increase Streak<br/><i>consecutive YoY increases</i>"]

    F1 --> G["Score Each Metric<br/>(0-10 via config curves)"]
    F2 --> G
    F3 --> G
    F4 --> G

    G --> H["Weighted Composite Score"]
    H --> I["Apply Threshold Filters"]
    I --> J["Rank by Score ↓"]
    J --> K["Display Top N"]

    style A fill:#1e3a5f,stroke:#4a90d9,color:#fff
    style C fill:#3a3a1e,stroke:#d9d94a,color:#fff
    style K fill:#2d5a1e,stroke:#6abf4b,color:#fff
```
""")

    with st.expander("Dividend Metrics — what each measures"):
        st.markdown("""
| Metric | What It Measures | Data Source |
|--------|-----------------|-------------|
| **Current Yield** | Annual dividend / current price | `trailingAnnualDividendRate` from yfinance `.info`, or computed from last 12 months of payments |
| **Dividend CAGR** | Compound annual growth rate of total annual dividends | Calendar-year aggregation of `.dividends` history (configurable N years, default 5) |
| **Payout Consistency** | Fraction of years with at least one dividend payment | Calendar-year check over N years (default 10) |
| **Increase Streak** | Consecutive years of year-over-year dividend increases | Walk backwards from most recent full year |

**Additional metadata** (displayed but not scored): payout ratio, 5-year average yield, ex-dividend date, sector.
""")

    with st.expander("Scoring Curves — how raw metrics become 0-10 scores"):
        st.markdown("""
Each metric is converted to a **0-10 score** using piecewise linear interpolation. All curve parameters are configurable.

**Yield Scoring:**
| Yield | Score | Logic |
|-------|-------|-------|
| ≤ 0.5% | 0 | Too low to be meaningful |
| 3.0% | 5 | Moderate yield |
| 6.0% | 10 | Excellent yield |
| 6-10% | 10 | Still excellent |
| > 10% | 10→5 | Distress penalty — extremely high yields often signal a dividend cut |

**Growth Scoring (CAGR):**
| CAGR | Score | Logic |
|------|-------|-------|
| -20% or worse | 0 | Severe decline |
| 0% | 4 | Flat growth |
| 7% | 8 | Target growth |
| 15%+ | 10 | Exceptional growth |

**Consistency Scoring:**
- 100% → 10, 50% → 5, 0% → 0 (linear)

**Streak Scoring:**
| Streak | Score | Label |
|--------|-------|-------|
| 0 years | 0 | No streak |
| 5 years | 5 | Decent |
| 10 years | 7 | Good |
| 25+ years | 10 | Dividend Aristocrat |

**Composite** = weighted average of all four scores (default: yield 30%, growth 30%, consistency 20%, streak 20%).
""")

    with st.expander("Threshold Filters — which stocks are excluded"):
        st.markdown("""
Before ranking, stocks are filtered by minimum thresholds (all configurable via UI sliders):

| Filter | Default | Purpose |
|--------|---------|---------|
| **Minimum Yield** | 1% | Exclude stocks with negligible dividends |
| **Maximum Yield** | 15% | Exclude likely-distressed yields |
| **Minimum CAGR** | -10% | Exclude stocks with severe dividend cuts |
| **Minimum Consistency** | 50% | Exclude unreliable payers |
| **Minimum Streak** | 0 years | No streak filter by default |

Stocks that fail any filter are excluded from the ranked results but still appear in the error/excluded count.
""")

    # ── Section 5: DCA Backtest ───────────────────────────────────────────
    st.subheader("5. DCA Backtest Flow")
    st.markdown("""
```mermaid
graph TD
    A[Select Ticker + Period] --> B[Choose DCA Mode]
    B --> C{Mode?}
    C -->|Pure| D[Fixed amount every interval]
    C -->|Dip-Weighted| E[Compute dip % from rolling high]
    C -->|Score-Integrated| F[Factor in technical score + RSI + BB]
    E --> G[Apply multiplier tier]
    F --> H[Apply multiplier + safety gates]
    D --> I[Execute buy]
    G --> I
    H --> I
    I --> J{DRIP enabled?}
    J -->|Yes| K[Reinvest dividends as new shares]
    J -->|No| L[Skip dividends]
    K --> M[Track equity curve]
    L --> M
    M --> N[Compute metrics + compare vs active strategy]
```

**DCA Modes:**

| Mode | Description |
|------|-------------|
| **Pure** | Fixed dollar amount at each interval — no timing intelligence. Serves as baseline. |
| **Dip-Weighted** | Base amount scaled by multiplier when price drops from recent high. Buys more during dips. |
| **Score-Integrated** | Full integration with technical scoring engine. Uses composite score, RSI, and Bollinger Band percentile to modulate buy amounts. Includes safety gates for breakaway gaps and low volume. |

**Multiplier Tiers:**

| Tier | Default Trigger | Default Multiplier | Description |
|------|----------------|-------------------|-------------|
| Normal | No dip detected | 1.0x | Standard DCA amount |
| Mild Dip | Price down 5%+ | 1.5x | Modest overweight |
| Strong Dip | Price down 10%+ | 2.0x | Aggressive overweight |
| Extreme Dip | Price down 20%+ | 3.0x | Maximum conviction |

**Safety Gates** (prevent catching falling knives):

| Gate | Default | Description |
|------|---------|-------------|
| Max Multiplier | 3.0x | Absolute cap on any single multiplier |
| Max Period Allocation | $1,500 | Maximum dollars in any single period |
| Skip Breakaway Gaps | On | Don't overweight on breakaway-down gaps (trend change, not a dip) |
| Min Volume Ratio | 0.5 | Skip overweight if volume < 50% of average (illiquid) |

**DRIP** (Dividend Reinvestment Plan): When enabled, dividends are automatically reinvested as additional shares at the current price on the ex-dividend date.

**CLI Usage:**
```
python main.py AAPL --dca --period 5y               # dip-weighted DCA (default)
python main.py AAPL --dca --dca-mode pure            # pure DCA, no dip weighting
python main.py AAPL --dca --dca-amount 1000          # $1000 per period
python main.py AAPL --dca --dca-frequency weekly     # weekly purchases
```

All thresholds, multipliers, and safety parameters are configurable via `config.yaml` (under the `dca` section) and via dashboard sliders in the DCA Backtest tab.
""")


# ---------------------------------------------------------------------------
# Main (new stepped flow)
# ---------------------------------------------------------------------------

def main() -> None:
    params = render_sidebar()

    # ── Top-level navigation tabs ─────────────────────────────────────────
    tab_analyze, tab_scanner, tab_help = st.tabs(["Analyze", "Scanner", "Help"])

    # ══════════════════════════════════════════════════════════════════════
    # SCANNER TAB
    # ══════════════════════════════════════════════════════════════════════
    with tab_scanner:
        render_scanner()

    # ══════════════════════════════════════════════════════════════════════
    # HELP TAB
    # ══════════════════════════════════════════════════════════════════════
    with tab_help:
        render_help()

    # ══════════════════════════════════════════════════════════════════════
    # ANALYZE TAB
    # ══════════════════════════════════════════════════════════════════════
    with tab_analyze:
        if not params["ticker"]:
            st.info("Enter a ticker symbol in the sidebar to begin.")
            return

        ticker = params["ticker"]
        period = params["period"]

        # ── Build Config ──────────────────────────────────────────────────
        cfg = _get_config()
        cfg_data = st.session_state["config_data"]
        cfg_h = _config_hash(cfg_data)

        # ══════════════════════════════════════════════════════════════════
        # STEP 1: Stock Overview (always runs)
        # ══════════════════════════════════════════════════════════════════
        try:
            result, _ = load_analysis(
                ticker=ticker,
                period=period,
                interval="1d",
                start=None,
                end=None,
                config_hash=cfg_h,
                config_data=cfg_data,
            )
        except Exception as e:
            st.error(f"Failed to fetch data for {ticker}: {e}")
            return

        render_stock_overview(result, cfg)

        # ══════════════════════════════════════════════════════════════════
        # STEP 2: Choose Backtest Path
        # ══════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.header("Backtest")

        tab_auto, tab_custom, tab_dca = st.tabs([
            "Quick Backtest (Recommended)",
            "Custom Backtest",
            "DCA Backtest",
        ])

        # ── Tab 1: Quick Backtest ─────────────────────────────────────────
        with tab_auto:
            st.markdown(
                f"Run a backtest using the **{period}** classification period with "
                f"**auto-detected** trading mode and **regime-adapted** strategy parameters."
            )
            if result.regime:
                regime_label = result.regime.label
                sub_label = result.regime.sub_type_label
                desc = f"Detected: **{regime_label}**"
                if sub_label:
                    desc += f" / **{sub_label}**"
                st.markdown(desc)

            run_quick = st.button("Run Quick Backtest", type="primary", key="run_quick_bt")

            if run_quick or st.session_state.get("_quick_bt_ran"):
                st.session_state["_quick_bt_ran"] = True
                try:
                    bt_result, trading_mode_val, assessment_dict, _ = load_backtest(
                        ticker=ticker,
                        period=period,
                        interval="1d",
                        start=None,
                        end=None,
                        trading_mode_str="auto",
                        config_hash=cfg_h,
                        config_data=cfg_data,
                    )
                except Exception as e:
                    st.error(f"Backtest failed: {e}")
                    return

                render_backtest_section(
                    bt_result, result, cfg, trading_mode_val, assessment_dict,
                )

        # ── Tab 2: Custom Backtest ────────────────────────────────────────
        with tab_custom:
            st.markdown("Configure custom parameters for the backtest.")

            custom_params = render_custom_backtest_params()

            # Validate day_trading + interval
            if custom_params["objective"] == "day_trading" and not is_intraday(custom_params["interval"]):
                st.error(
                    "The **day_trading** objective requires an intraday interval "
                    "(1m, 5m, 15m, 30m, 1h). Please change the interval."
                )
            else:
                run_custom = st.button(
                    "Run Custom Backtest", type="primary", key="run_custom_bt",
                )

                if run_custom or st.session_state.get("_custom_bt_ran"):
                    st.session_state["_custom_bt_ran"] = True

                    # Re-read config (may have been mutated by param editors)
                    custom_cfg_data = st.session_state["config_data"]
                    custom_cfg_h = _config_hash(custom_cfg_data)
                    custom_cfg = Config.from_dict(custom_cfg_data)

                    try:
                        bt_result, trading_mode_val, assessment_dict, _ = load_backtest(
                            ticker=ticker,
                            period=custom_params["period"],
                            interval=custom_params["interval"],
                            start=custom_params["start"],
                            end=custom_params["end"],
                            trading_mode_str=custom_params["trading_mode"],
                            config_hash=custom_cfg_h,
                            config_data=custom_cfg_data,
                        )
                    except Exception as e:
                        st.error(f"Backtest failed: {e}")
                        return

                    # For custom backtest, run analysis on the custom period/interval
                    # for the recommendation engine
                    try:
                        custom_result, _ = load_analysis(
                            ticker=ticker,
                            period=custom_params["period"],
                            interval=custom_params["interval"],
                            start=custom_params["start"],
                            end=custom_params["end"],
                            config_hash=custom_cfg_h,
                            config_data=custom_cfg_data,
                        )
                    except Exception:
                        custom_result = result  # fall back to overview analysis

                    render_backtest_section(
                        bt_result, custom_result, custom_cfg,
                        trading_mode_val, assessment_dict,
                    )

        # ── Tab 3: DCA Backtest ───────────────────────────────────────────
        with tab_dca:
            st.markdown(
                "Simulate **Dollar Cost Averaging** — periodic fixed-amount "
                "purchases with optional dip-weighted or score-integrated "
                "multipliers. Compare against the active trading strategy."
            )

            dca_overrides = _render_dca_params(cfg)

            dca_period_opts = ["1y", "2y", "3y", "5y", "max"]
            dca_period = st.selectbox(
                "DCA Period",
                dca_period_opts,
                index=dca_period_opts.index("5y"),
                key="dca_period",
            )

            run_dca = st.button("Run DCA Backtest", type="primary", key="run_dca_bt")

            if run_dca or st.session_state.get("_dca_bt_ran"):
                st.session_state["_dca_bt_ran"] = True

                with st.spinner("Running DCA backtest..."):
                    try:
                        dca_bt = DCABacktester(cfg=cfg, overrides=dca_overrides)
                        dca_result = dca_bt.run(ticker, period=dca_period)
                    except Exception as e:
                        st.error(f"DCA backtest failed: {e}")
                        return

                # Try to get active backtest result for comparison
                active_bt: BacktestResult | None = None
                try:
                    active_bt, _, _, _ = load_backtest(
                        ticker=ticker,
                        period=dca_period,
                        interval="1d",
                        start=None,
                        end=None,
                        trading_mode_str="auto",
                        config_hash=cfg_h,
                        config_data=cfg_data,
                    )
                except Exception:
                    pass  # comparison is optional

                render_dca_section(dca_result, active_bt)

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "For informational purposes only. Not financial advice. "
        "Do your own research. Backtest results are hypothetical and "
        "do not guarantee future performance."
    )



if __name__ == "__main__":
    main()
