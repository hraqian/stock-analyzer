# Stock Technical Analysis — User Guide

Complete reference for the CLI tool, Streamlit dashboard, and every
configurable parameter in `config.yaml`.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [CLI Reference](#cli-reference)
3. [Streamlit Dashboard](#streamlit-dashboard)
4. [Configuration Overview](#configuration-overview)
5. [Indicator Parameters](#indicator-parameters)
6. [Pattern Detector Parameters](#pattern-detector-parameters)
7. [Composite Scoring](#composite-scoring)
8. [Display Settings](#display-settings)
9. [Strategy Parameters](#strategy-parameters)
10. [Backtest Parameters](#backtest-parameters)
11. [Market Regime Classification](#market-regime-classification)
    - [Sub-Type Classification](#sub-type-classification)
12. [Trading Mode Suitability](#trading-mode-suitability)
13. [Objective Presets](#objective-presets)
14. [Multi-Timeframe Analysis](#multi-timeframe-analysis)
15. [Tips & Troubleshooting](#tips--troubleshooting)

---

## Quick Start

```bash
# Basic analysis (6-month default)
python main.py AAPL

# 2-year analysis
python main.py TSLA --period 2y

# Run a backtest
python main.py AAPL --backtest --period 2y

# Use a preset objective
python main.py AAPL --objective long_term -b --period 5y

# Day trading (requires intraday interval)
python main.py AAPL -o day_trading -b -i 5m --start 2026-02-17

# Launch the interactive dashboard
streamlit run dashboard.py
```

---

## CLI Reference

```
stock_analyzer [-h] [TICKER] [options]
```

### Positional argument

| Argument | Description |
|----------|-------------|
| `TICKER` | Stock ticker symbol (e.g. `AAPL`, `TSLA`). Optional when using utility flags. |

### Analysis options

| Flag | Default | Description |
|------|---------|-------------|
| `--period`, `-p` | `6mo` | Data period. Options: `1mo 3mo 6mo 1y 2y 5y ytd max` |
| `--interval`, `-i` | `1d` | Bar interval. Daily: `1d 5d 1wk 1mo 3mo`. Intraday: `1m 2m 5m 15m 30m 60m 90m 1h` |
| `--indicators` | all | Comma-separated indicator list, e.g. `rsi,macd,adx` |
| `--start`, `-s` | — | Start date `YYYY-MM-DD`. Overrides `--period`. |
| `--end`, `-e` | today | End date `YYYY-MM-DD`. Only used with `--start`. |

### Backtest options

| Flag | Default | Description |
|------|---------|-------------|
| `--backtest`, `-b` | off | Run a backtest instead of (or alongside) analysis |
| `--mode`, `-m` | auto | Trading mode: `auto`, `long_short`, `long_only`, `hold_only` |
| `--objective`, `-o` | — | Apply a preset: `long_term`, `short_term`, `day_trading` (or custom) |

### Config & utility

| Flag | Description |
|------|-------------|
| `--config`, `-c` | Path to a custom `config.yaml` |
| `--generate-config` | Write a fresh default `config.yaml` and exit |
| `--validate-config` | Check `config.yaml` for errors and exit |
| `--list-indicators` | List all available indicator keys and exit |
| `--list-patterns` | List all available pattern detector keys and exit |

### yfinance data limits (intraday)

| Interval | Maximum history |
|----------|----------------|
| `1m` | ~7 days |
| `5m`, `15m`, `30m` | ~60 days |
| `1h` | ~730 days |
| `1d` and above | unlimited |

Supported period values: `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `ytd`, `max`.
Note: `10y` and day-count periods like `60d` are **not** supported by yfinance.

---

## Streamlit Dashboard

Launch with:

```bash
streamlit run dashboard.py
```

The dashboard provides:

- **Sidebar controls** for every config parameter, grouped into collapsible
  sections (Indicators, Patterns, Strategy, Backtest, Regime, Suitability).
- **Default hints** below each widget: small gray text showing the default
  value and a brief description.
- **Live analysis and backtesting** — change a parameter and re-run
  instantly.
- **Config export** — download your tuned configuration as YAML.

All parameter changes are applied via `Config.from_dict()` and are
equivalent to editing `config.yaml` directly.

---

## Configuration Overview

All parameters live in `config.yaml` (or the built-in `DEFAULT_CONFIG`
fallback). The config is a flat YAML dict with top-level sections:

| Section | Purpose |
|---------|---------|
| `rsi`, `macd`, `bollinger_bands`, ... | Per-indicator tuning |
| `gaps`, `candlesticks`, `spikes`, ... | Per-pattern-detector tuning |
| `overall` | Indicator composite weights |
| `overall_patterns` | Pattern composite weights |
| `display` | Terminal output formatting |
| `strategy` | Trading decision thresholds, position sizing, risk management |
| `backtest` | Engine parameters (cash, slippage, warmup) |
| `regime` | Market regime classification thresholds and scoring weights |
| `suitability` | Trading mode auto-detection thresholds |
| `objectives` | Named presets that override base values |

To regenerate a fresh config:

```bash
python main.py --generate-config
```

To validate your config:

```bash
python main.py --validate-config
```

---

## Indicator Parameters

All scores range from **0** (strongly bearish) to **10** (strongly bullish).

### RSI (Relative Strength Index)

| Key | Default | Description |
|-----|---------|-------------|
| `rsi.period` | 14 | RSI calculation lookback period |
| `rsi.thresholds.oversold` | 30 | RSI below this = oversold (potential reversal up) |
| `rsi.thresholds.overbought` | 70 | RSI above this = overbought (potential reversal down) |
| `rsi.scores.oversold_score` | 9.0 | Score when RSI <= oversold |
| `rsi.scores.overbought_score` | 1.0 | Score when RSI >= overbought |
| `rsi.scores.neutral_score` | 5.0 | Score at RSI = 50 |

### MACD (Moving Average Convergence Divergence)

| Key | Default | Description |
|-----|---------|-------------|
| `macd.fast_period` | 12 | Fast EMA period |
| `macd.slow_period` | 26 | Slow EMA period |
| `macd.signal_period` | 9 | Signal line EMA period |
| `macd.scoring.strong_bullish_pct` | 0.005 | Histogram > price * this = strong bullish |
| `macd.scoring.moderate_bullish_pct` | 0.001 | Histogram > price * this = moderate bullish |
| `macd.scoring.strong_bearish_pct` | -0.005 | Histogram < price * this = strong bearish |
| `macd.scoring.moderate_bearish_pct` | -0.001 | Histogram < price * this = moderate bearish |
| `macd.scoring.crossover_lookback` | 5 | Bars to look back for signal crossover |
| `macd.scoring.bullish_cross_bonus` | 1.5 | Score bonus for bullish crossover |
| `macd.scoring.bearish_cross_penalty` | 1.5 | Score penalty for bearish crossover |

### Bollinger Bands

| Key | Default | Description |
|-----|---------|-------------|
| `bollinger_bands.period` | 20 | SMA period for band calculation |
| `bollinger_bands.std_dev` | 2.0 | Standard deviation multiplier |
| `bollinger_bands.scoring.lower_zone` | 0.20 | %B below this = near support (score 8-10) |
| `bollinger_bands.scoring.upper_zone` | 0.80 | %B above this = near resistance (score 0-2) |
| `bollinger_bands.scoring.squeeze_threshold` | 0.02 | Band width / price below this = volatility squeeze |

### Moving Averages

| Key | Default | Description |
|-----|---------|-------------|
| `moving_averages.periods` | [20, 50, 200] | MA periods (must be ascending) |
| `moving_averages.type` | "sma" | "sma" or "ema" |
| `moving_averages.scoring.price_above_ma_points` | 1.5 | Points per MA that price is above |
| `moving_averages.scoring.ma_aligned_bullish_points` | 1.0 | Points per bullish MA alignment (20>50, 50>200) |
| `moving_averages.scoring.golden_cross_bonus` | 2.0 | Bonus when 50 MA crosses above 200 MA |
| `moving_averages.scoring.death_cross_penalty` | 2.0 | Penalty when 50 MA crosses below 200 MA |
| `moving_averages.scoring.cross_lookback` | 10 | Bars to look back for golden/death cross |
| `moving_averages.scoring.max_raw_score` | 9.5 | Cap before normalization to 0-10 |

### Stochastic Oscillator

| Key | Default | Description |
|-----|---------|-------------|
| `stochastic.k_period` | 14 | %K calculation period |
| `stochastic.d_period` | 3 | %D smoothing period |
| `stochastic.smooth_k` | 3 | %K smoothing factor |
| `stochastic.thresholds.oversold` | 20 | %K below this = oversold |
| `stochastic.thresholds.overbought` | 80 | %K above this = overbought |
| `stochastic.scores.oversold_score` | 9.0 | Score when oversold |
| `stochastic.scores.overbought_score` | 1.0 | Score when overbought |
| `stochastic.scores.neutral_score` | 5.0 | Score at midpoint |
| `stochastic.scores.bullish_cross_bonus` | 1.0 | Bonus when %K crosses above %D |
| `stochastic.scores.bearish_cross_penalty` | 1.0 | Penalty when %K crosses below %D |

### ADX (Average Directional Index)

| Key | Default | Description |
|-----|---------|-------------|
| `adx.period` | 14 | ADX calculation period |
| `adx.thresholds.weak` | 20 | ADX below this = weak/no trend |
| `adx.thresholds.moderate` | 40 | ADX 20-40 = moderate trend |
| `adx.scoring.weak_multiplier` | 0.6 | Directional score multiplier in weak trends |
| `adx.scoring.moderate_multiplier` | 0.85 | Multiplier in moderate trends |
| `adx.scoring.strong_multiplier` | 1.0 | Multiplier in strong trends |
| `adx.scoring.max_directional_spread` | 25 | +DI minus -DI spread for full directional score |

**Note:** ADX measures current trend *strength*, not long-term trajectory.
A stock in a powerful multi-year uptrend can have low ADX during consolidation.

### Volume (OBV-based)

| Key | Default | Description |
|-----|---------|-------------|
| `volume.obv_trend_period` | 20 | EMA period for OBV trend |
| `volume.price_trend_period` | 20 | EMA period for price trend comparison |
| `volume.scoring.confirmation_bullish_max` | 9.5 | Max score when OBV confirms bullish price |
| `volume.scoring.confirmation_bullish_min` | 6.5 | Min score when OBV weakly confirms bullish |
| `volume.scoring.confirmation_bearish_max` | 3.5 | Score when OBV weakly confirms bearish |
| `volume.scoring.confirmation_bearish_min` | 0.5 | Score when OBV strongly confirms bearish |
| `volume.scoring.divergence_score` | 5.0 | Score when OBV diverges from price |
| `volume.scoring.obv_strong_change_pct` | 10.0 | OBV change % for max confirmation score |
| `volume.scoring.obv_weak_change_pct` | 1.0 | OBV change % for min confirmation score |

### Fibonacci Retracement

| Key | Default | Description |
|-----|---------|-------------|
| `fibonacci.swing_lookback` | 60 | Bars to look back for swing high/low |
| `fibonacci.levels` | [0.236, 0.382, 0.5, 0.618, 0.786] | Fibonacci retracement levels |
| `fibonacci.scoring.proximity_pct` | 0.015 | Within 1.5% of a level = "at that level" |
| `fibonacci.scoring.level_scores` | {0.236: 8.0, ..., 0.786: 1.5} | Score for each Fibonacci level |
| `fibonacci.scoring.no_level_score` | 5.0 | Score when not near any level |
| `fibonacci.scoring.range_low_score` | 2.0 | Score at bottom of swing range |
| `fibonacci.scoring.range_high_score` | 8.0 | Score at top of swing range |

### Support & Resistance (display only, not scored)

| Key | Default | Description |
|-----|---------|-------------|
| `support_resistance.method` | "both" | Detection method: "pivot", "fractal", or "both" |
| `support_resistance.pivot_levels` | ["S1"..."R3"] | Classic pivot levels to compute |
| `support_resistance.fractal_lookback` | 60 | Bars to scan for fractal extrema |
| `support_resistance.fractal_order` | 5 | Bars each side for local extremum |
| `support_resistance.num_levels` | 4 | Number of support AND resistance levels to show |
| `support_resistance.cluster_pct` | 0.015 | Merge levels within 1.5% of each other |
| `support_resistance.min_touches` | 1 | Minimum price touches to confirm a level |

---

## Pattern Detector Parameters

Pattern detectors measure *what just happened* (a gap, a reversal candle, a
volume spike) — a separate layer from indicators which measure *where the
stock is*. Each pattern scores 0-10, with 5.0 = no pattern / neutral.

### Gaps

| Key | Default | Description |
|-----|---------|-------------|
| `gaps.lookback` | 20 | Recent bars to consider for scoring |
| `gaps.min_gap_pct` | 0.005 | Minimum gap size as fraction of price (0.5%) |
| `gaps.volume_surge_mult` | 1.5 | Volume > avg * this = surge (for gap classification) |
| `gaps.trend_period` | 20 | EMA period for trend detection |
| `gaps.type_weights.common` | 0.3 | Weight of common gaps in net signal |
| `gaps.type_weights.runaway` | 0.7 | Weight of runaway (continuation) gaps |
| `gaps.type_weights.breakaway` | 1.0 | Weight of breakaway gaps |
| `gaps.type_weights.exhaustion` | 0.5 | Weight of exhaustion gaps (inverse signal) |
| `gaps.max_signal_strength` | 3.0 | Net signal beyond this = full score (0.5 or 9.5) |
| `gaps.gap_pct_scale` | 100 | Multiplier to convert gap_pct into strength units |

### Volume-Range Analysis

| Key | Default | Description |
|-----|---------|-------------|
| `volume_range.period` | 20 | Rolling average period for range and volume |
| `volume_range.expansion_threshold` | 1.5 | Range/volume ratio >= this = expansion |
| `volume_range.contraction_threshold` | 0.6 | Range/volume ratio <= this = contraction |
| `volume_range.lookback` | 10 | Recent bars to count expansions/contractions |
| `volume_range.scoring.expansion_bull` | 8.0 | Bullish expansion base score |
| `volume_range.scoring.expansion_bear` | 2.0 | Bearish expansion base score |
| `volume_range.scoring.contraction` | 5.0 | Contraction = neutral |
| `volume_range.scoring.divergence` | 5.0 | Range/volume divergence = neutral |
| `volume_range.expansion_bias_multiplier` | 1.5 | Multiplier for expansion directional bias |

### Candlestick Patterns

| Key | Default | Description |
|-----|---------|-------------|
| `candlesticks.doji_threshold` | 0.05 | Body/range <= this = doji |
| `candlesticks.shadow_ratio` | 2.0 | Shadow must be >= body * this for hammer/star |
| `candlesticks.harami_body_ratio` | 0.5 | Current body <= prev body * this for harami |
| `candlesticks.dragonfly_shadow_min` | 0.6 | Lower shadow >= range * this for dragonfly doji |
| `candlesticks.gravestone_shadow_min` | 0.6 | Upper shadow >= range * this for gravestone doji |
| `candlesticks.doji_tiny_shadow_max` | 0.1 | Opposite shadow <= range * this for dragonfly/gravestone |
| `candlesticks.marubozu_body_min` | 0.90 | Body fills >= this fraction of range for marubozu |
| `candlesticks.marubozu_shadow_max` | 0.05 | Each shadow <= this fraction of range for marubozu |
| `candlesticks.tweezer_tolerance` | 0.002 | Highs/lows match within this fraction for tweezer |
| `candlesticks.star_middle_body_max` | 0.30 | Middle bar body <= this fraction for morning/evening star |
| `candlesticks.soldiers_body_min` | 0.60 | Each bar body >= this fraction for 3 soldiers/crows |
| `candlesticks.soldiers_shadow_max` | 0.30 | Shadow <= this fraction for 3 soldiers/crows |
| `candlesticks.lookback` | 10 | Recent bars to scan for patterns |
| `candlesticks.trend_period` | 10 | EMA period for trend context |
| `candlesticks.max_signal_strength` | 3.0 | Net signal beyond this = full score |
| `candlesticks.hammer_body_max` | 0.35 | Max body ratio for hammer/shooting star detection |
| `candlesticks.star_body_min` | 0.5 | Min body ratio for morning/evening star center candle |
| `candlesticks.strength_values` | {} | Per-pattern strength overrides (empty = built-in defaults). Keys: `dragonfly_doji`, `gravestone_doji`, `doji_directional`, `doji_neutral`, `hammer`, `shooting_star`, `marubozu`, `engulfing`, `harami`, `tweezer`, `star`, `soldiers_crows`. Each maps to `{with_trend: float, against_trend: float}`. |

### Spike Detection

| Key | Default | Description |
|-----|---------|-------------|
| `spikes.period` | 20 | Rolling mean/std calculation period |
| `spikes.spike_std` | 2.5 | Z-score threshold for spike detection |
| `spikes.confirm_bars` | 3 | Bars to check for post-spike confirmation |
| `spikes.confirm_pct` | 0.5 | Fraction of confirm bars that must hold level |
| `spikes.lookback` | 20 | Recent bars to consider for scoring |
| `spikes.trap_weight` | 0.7 | Weight of trap (failed breakout) inverse signal |
| `spikes.max_signal_strength` | 3.0 | Net signal beyond this = full score |
| `spikes.z_magnitude_cap` | 2.0 | Cap on z-score magnitude for strength calculation |
| `spikes.unconfirmed_weight` | 0.3 | Weight multiplier for unconfirmed spikes |

### Inside/Outside Bars

| Key | Default | Description |
|-----|---------|-------------|
| `inside_outside.lookback` | 20 | Recent bars to consider for scoring |
| `inside_outside.trend_period` | 10 | EMA period for trend context |
| `inside_outside.breakout_bars` | 3 | Bars after inside bar to check for breakout |
| `inside_outside.outside_range_min` | 1.2 | Current range / prev range >= this for outside bar |
| `inside_outside.max_signal_strength` | 3.0 | Net signal beyond this = full score |
| `inside_outside.strength_values` | {} | Per-pattern strength overrides (empty = built-in defaults). Keys: `inside_breakout_with_trend`, `inside_breakout_against_trend`, `inside_pending`, `outside_reversal`, `outside_continuation`. Each maps to a float. |

---

## Composite Scoring

### Indicator Composite Weights

Section: `overall.weights`

Each indicator's 0-10 score is multiplied by its weight, then all weighted
scores are summed and normalized. Weights don't need to sum to 1.0 — they
are auto-normalized.

| Key | Default | Description |
|-----|---------|-------------|
| `overall.weights.rsi` | 0.15 | RSI weight |
| `overall.weights.macd` | 0.15 | MACD weight |
| `overall.weights.bollinger_bands` | 0.10 | Bollinger Bands weight |
| `overall.weights.moving_averages` | 0.20 | Moving Averages weight (highest default) |
| `overall.weights.stochastic` | 0.10 | Stochastic weight |
| `overall.weights.adx` | 0.10 | ADX weight |
| `overall.weights.volume` | 0.10 | Volume weight |
| `overall.weights.fibonacci` | 0.10 | Fibonacci weight |

**Score compression note:** Averaging 8 indicators crushes variance. The
effective composite score stdev is ~0.50, so almost all scores fall in the
3.5-6.5 range. Set strategy thresholds within that actual range, or use
`threshold_mode: "percentile"` for self-calibrating thresholds.

### Directional Subgroup Scoring

Section: `overall`

Instead of a simple weighted average, the composite scorer can group
indicators into **trend**, **contrarian**, and **neutral** buckets. When the
groups agree, the dominant group gets extra weight; when they conflict, it
dampens extremes. Score spreading then re-scales the result away from 5.0.

| Key | Default | Description |
|-----|---------|-------------|
| `overall.subgroup_mode` | "directional" | `"directional"` = subgroup blend; `"average"` = plain weighted average (legacy) |
| `overall.indicator_groups.trend` | ["moving_averages", "macd", "adx", "volume"] | Indicators in the trend-following group |
| `overall.indicator_groups.contrarian` | ["rsi", "stochastic", "bollinger_bands"] | Indicators in the mean-reversion group |
| `overall.indicator_groups.neutral` | ["fibonacci"] | Indicators that are direction-agnostic |
| `overall.subgroup_blend.dominant_weight` | 0.6 | Weight given to the dominant subgroup (trend or contrarian) |
| `overall.subgroup_blend.other_weight` | 0.25 | Weight given to the non-dominant subgroup |
| `overall.subgroup_blend.neutral_weight` | 0.15 | Weight given to the neutral subgroup |
| `overall.score_spreading.enabled` | true | Stretch final composite away from 5.0 for wider signal range |
| `overall.score_spreading.factor` | 2.0 | Multiplier for deviation from 5.0 (clamped to 0-10) |

### Pattern Composite Weights

Section: `overall_patterns.weights`

| Key | Default | Description |
|-----|---------|-------------|
| `overall_patterns.weights.gaps` | 0.20 | Gap detector weight |
| `overall_patterns.weights.volume_range` | 0.25 | Volume-range weight |
| `overall_patterns.weights.candlesticks` | 0.25 | Candlestick weight |
| `overall_patterns.weights.spikes` | 0.15 | Spike detector weight |
| `overall_patterns.weights.inside_outside` | 0.15 | Inside/outside bar weight |
| `overall_patterns.score_spreading.enabled` | true | Stretch pattern composite away from 5.0 |
| `overall_patterns.score_spreading.factor` | 2.0 | Multiplier for deviation from 5.0 |

---

## Display Settings

Section: `display`

| Key | Default | Description |
|-----|---------|-------------|
| `display.show_disclaimer` | true | Show disclaimer text in analysis output |
| `display.score_decimal_places` | 1 | Decimal places for score display |
| `display.price_decimal_places` | 2 | Decimal places for price display |
| `display.color_thresholds.bearish_max` | 3.5 | Score <= this = red (bearish) |
| `display.color_thresholds.neutral_max` | 6.0 | Score <= this = yellow (neutral), above = green (bullish) |

---

## Strategy Parameters

Section: `strategy`

The strategy maps composite scores to trading decisions (LONG, SHORT, HOLD).

### Threshold Mode

| Key | Default | Description |
|-----|---------|-------------|
| `strategy.threshold_mode` | "fixed" | "fixed" = absolute score values; "percentile" = rolling percentile ranks |

#### Fixed thresholds

| Key | Default | Description |
|-----|---------|-------------|
| `strategy.score_thresholds.short_below` | 3.5 | Score <= this = SHORT signal |
| `strategy.score_thresholds.hold_below` | 6.0 | Score <= this = HOLD; above = LONG |

#### Percentile thresholds (when `threshold_mode: "percentile"`)

| Key | Default | Description |
|-----|---------|-------------|
| `strategy.percentile_thresholds.short_percentile` | 25 | Score in bottom 25% = SHORT |
| `strategy.percentile_thresholds.long_percentile` | 75 | Score in top 25% = LONG |
| `strategy.percentile_thresholds.lookback_bars` | 60 | Rolling window for percentile calculation |
| `strategy.percentile_thresholds.percentile_step` | 5 | Bar step when building percentile window (auto-retries with step=1 if too few samples) |
| `strategy.percentile_min_fill_ratio` | 0.8 | Min fraction of window before percentile activates |

### Position Sizing

| Key | Default | Description |
|-----|---------|-------------|
| `strategy.position_sizing` | "percent_equity" | "fixed" or "percent_equity" |
| `strategy.fixed_quantity` | 100 | Shares per trade (when sizing = "fixed") |
| `strategy.percent_equity` | 1.00 | Fraction of equity per trade (when sizing = "percent_equity") |

### Risk Management

| Key | Default | Description |
|-----|---------|-------------|
| `strategy.stop_loss_pct` | 0.05 | Exit if position loses this fraction (5%) |
| `strategy.take_profit_pct` | 0.30 | Exit if position gains this fraction (30%) |
| `strategy.atr_stop_enabled` | true | Use ATR-adaptive stop (widens beyond fixed %) |
| `strategy.atr_stop_multiplier` | 4.0 | Stop = N x ATR at entry |
| `strategy.atr_stop_period` | 14 | ATR calculation lookback |

The effective stop is `max(fixed_stop, ATR_stop)` — the ATR can only widen,
never tighten, the stop.

### Trend Confirmation

| Key | Default | Description |
|-----|---------|-------------|
| `strategy.trend_confirm_enabled` | true | Require price on correct side of MA for entry |
| `strategy.trend_confirm_period` | 20 | EMA period for trend filter |
| `strategy.trend_confirm_ma_type` | "ema" | "ema" or "sma" |
| `strategy.trend_confirm_tolerance_pct` | 0.0 | Tolerance band around MA (0 = exact) |

BUY requires `close > EMA(period)`. SHORT requires `close < EMA(period)`.

### Rebalancing & Re-entry

| Key | Default | Description |
|-----|---------|-------------|
| `strategy.rebalance_interval` | 5 | Re-score every N bars |
| `strategy.reentry_grace_bars` | 10 | After exit, skip trend confirmation for N bars |

### Consecutive Loss Cooldown

After repeated losing trades, entry requirements tighten to avoid bleeding
capital.

| Key | Default | Description |
|-----|---------|-------------|
| `strategy.cooldown_max_losses` | 2 | Consecutive losses before cooldown activates |
| `strategy.cooldown_distance_mult` | 2.0 | Multiply `min_distance` by this during cooldown |
| `strategy.cooldown_min_score` | 4.5 | Minimum score required during cooldown |
| `strategy.cooldown_reset_on_breakeven` | true | Whether 0% PnL resets the loss counter |

### Global Trend Bias

Suppresses counter-trend entries when the overall period return is strongly
directional.

| Key | Default | Description |
|-----|---------|-------------|
| `strategy.global_trend_bias` | true | Enable global directional bias |
| `strategy.global_bias_threshold` | 0.10 | \|total_return\| above this = suppress counter-trend entries |

### Strong Trend Behavior

| Key | Default | Description |
|-----|---------|-------------|
| `strategy.trend_bias_return_threshold` | 0.15 | \|total_return\| >= this = definitive bias direction |
| `strategy.extreme_exit_score_offset` | 1.5 | Exit strong-trend position when score is this far beyond thresholds |
| `strategy.breakout_min_move_ratio` | 0.4 | \|close-open\|/range >= this for valid breakout bar |
| `strategy.disable_take_profit_in_strong_trend` | true | Let trailing stop handle exits instead of take-profit |
| `strategy.trailing_stop_require_profit` | true | Trailing stop only activates when position is in profit |

### Position Management

| Key | Default | Description |
|-----|---------|-------------|
| `strategy.allow_pyramiding` | false | Add to existing same-direction positions |
| `strategy.allow_immediate_reversal` | true | Close + reopen in opposite direction on signal flip |
| `strategy.flatten_eod` | false | Force-close all positions at end of each trading day (intraday only) |

### Pattern-Indicator Combination

Three modes control how the indicator composite and pattern composite are
combined into a single trading signal.

| Key | Default | Description |
|-----|---------|-------------|
| `strategy.combination_mode` | "weighted" | "weighted", "gate", or "boost" |

#### Weighted mode (default)

```
blended = indicator_weight * indicator_score + pattern_weight * pattern_score
```

| Key | Default | Description |
|-----|---------|-------------|
| `strategy.indicator_weight` | 0.7 | Weight of indicator composite |
| `strategy.pattern_weight` | 0.3 | Weight of pattern composite |

#### Gate mode

Only trade if **both** composites pass their respective thresholds.

| Key | Default | Description |
|-----|---------|-------------|
| `strategy.gate_indicator_min` | 5.5 | Indicator must exceed this for LONG |
| `strategy.gate_indicator_max` | 4.5 | Indicator must be below this for SHORT |
| `strategy.gate_pattern_min` | 5.5 | Pattern must exceed this for LONG |
| `strategy.gate_pattern_max` | 4.5 | Pattern must be below this for SHORT |

#### Boost mode

Indicators are the base signal. Active patterns (score outside dead zone)
amplify or dampen the indicator score. Inactive patterns (score near 5.0)
have zero effect — avoiding the weighted-mode problem where neutral patterns
dilute the indicator signal.

| Key | Default | Description |
|-----|---------|-------------|
| `strategy.boost_strength` | 0.5 | Multiplier for pattern deviation from 5.0 |
| `strategy.boost_dead_zone` | 0.3 | Pattern within 5.0 +/- this = no boost |

### Signal Confidence Labeling

Used in scanner and dashboard recommendation to label signal confidence
(HIGH / MEDIUM / LOW) based on how far the score is from the threshold.

| Key | Default | Description |
|-----|---------|-------------|
| `strategy.confidence_thresholds.high` | 1.5 | Score distance from threshold for HIGH confidence |
| `strategy.confidence_thresholds.medium` | 0.5 | Score distance from threshold for MEDIUM confidence |
| `strategy.confidence_thresholds.hold_high` | 1.0 | Distance inside hold zone for HIGH hold confidence |
| `strategy.confidence_thresholds.hold_medium` | 0.3 | Distance inside hold zone for MEDIUM hold confidence |

---

## Backtest Parameters

Section: `backtest`

| Key | Default | Description |
|-----|---------|-------------|
| `backtest.initial_cash` | 100000.0 | Starting portfolio cash |
| `backtest.commission_per_trade` | 0.0 | Flat dollar commission per trade leg (entry & exit each) |
| `backtest.commission_pct` | 0.0 | Percentage commission per trade leg as a fraction (0.001 = 0.1%) |
| `backtest.commission_mode` | "additive" | How flat + pct fees combine: `"additive"` (sum both) or `"max"` (whichever is greater) |
| `backtest.slippage_pct` | 0.001 | Simulated slippage as fraction of price (0.1%) |
| `backtest.warmup_bars` | 50 | Minimum bars before first signal (must be >= longest indicator lookback) |
| `backtest.max_warmup_ratio` | 0.5 | Maximum fraction of data that warmup can consume |
| `backtest.significant_pattern_min_strength` | 0.5 | Minimum strength for pattern to appear in timeline |
| `backtest.min_warmup_bars` | 20 | Absolute floor for proportional warmup |
| `backtest.min_post_warmup_bars` | 10 | Minimum tradeable bars after warmup |
| `backtest.trading_days_per_year` | 252 | For annualization of returns |
| `backtest.trading_day_minutes` | 390 | US market: 6.5 hours (for intraday bar counts) |
| `backtest.default_score` | 5.0 | Neutral starting score before first rebalance |
| `backtest.close_on_end_of_data` | true | Force-close any open position at end of data |

### Pattern Strength Tuning

These control how raw pattern detector outputs are converted into strength
values for the backtest event timeline.

| Key | Default | Description |
|-----|---------|-------------|
| `backtest.strength_thresholds` | {} | Per-detector strength overrides (empty = use built-in defaults) |
| `backtest.gap_strength_cap` | 2.0 | Cap on gap detector strength contribution |
| `backtest.spike_z_divisor` | 2.5 | Z-score divisor for spike strength normalization |
| `backtest.spike_strength_cap` | 2.0 | Cap on spike detector strength contribution |

**Warmup note:** Effective warmup = `min(warmup_bars, int(len(data) * max_warmup_ratio))`,
clamped to at least `min_warmup_bars`. Short datasets (e.g. 6mo = ~126 bars)
get proportionally reduced warmup.

---

## Market Regime Classification

Section: `regime`

Classifies the current market environment into one of four regimes, which
then drives strategy adaptation during backtesting.

### The four regimes

| Regime | When | Strategy adaptation |
|--------|------|---------------------|
| `strong_trend` | High ADX, consistent direction, large total return | Trailing stop, hold with trend |
| `mean_reverting` | Low ADX, oscillates around MA, small return | Tighten thresholds, swing trade |
| `volatile_choppy` | High ATR%, frequent reversals | Reduce position size, widen stops |
| `breakout_transition` | BB squeeze then expansion | Momentum entry on breakout confirmation |

### Core thresholds

| Key | Default | Description |
|-----|---------|-------------|
| `regime.trend_ma_period` | 50 | MA period for trend analysis |
| `regime.adx_strong_trend` | 30.0 | ADX above this = strong trend signal |
| `regime.adx_weak` | 20.0 | ADX below this = weak/no trend |
| `regime.trend_consistency_high` | 70.0 | % bars above MA = strong directional bias |
| `regime.trend_consistency_low` | 40.0 | Below 100-this on bear side = strong bear bias |
| `regime.atr_pct_high` | 0.03 | ATR% above this = high volatility |
| `regime.atr_pct_low` | 0.01 | ATR% below this = low volatility |
| `regime.atr_period` | 14 | ATR calculation period |
| `regime.bb_period` | 20 | Bollinger Band period |
| `regime.bb_std_dev` | 2.0 | BB standard deviation multiplier |
| `regime.bb_squeeze_percentile` | 20.0 | BB width below this percentile = squeeze |
| `regime.bb_expansion_percentile` | 80.0 | BB width above this = expansion |
| `regime.direction_change_high` | 0.55 | Fraction of bars reversing direction = choppy |
| `regime.direction_change_period` | 20 | Lookback for direction change calculation |
| `regime.price_ma_distance_extended` | 0.10 | Price > 10% from MA = extended trend |
| `regime.total_return_strong` | 0.30 | \|return\| > 30% = definitively trending |
| `regime.total_return_moderate` | 0.15 | \|return\| > 15% = moderate trend signal |
| `regime.min_bars_for_classification` | 20 | Minimum bars before regime can be classified |

### Trend direction thresholds

| Key | Default | Description |
|-----|---------|-------------|
| `regime.trend_direction_bullish_threshold` | 60 | pct_above_ma > this = bullish |
| `regime.trend_direction_bearish_threshold` | 40 | pct_above_ma < this = bearish |

### Reason building

| Key | Default | Description |
|-----|---------|-------------|
| `regime.adx_dip_threshold` | 3 | Rolling ADX mean > current + this = "temporary dip" note |
| `regime.runner_up_proximity_ratio` | 0.7 | Runner-up score / winner > this = mention runner-up |

### Regime Scoring Weights

Section: `regime.scoring`

These control the numerical scores assigned to each regime during
classification. Higher scores pull the classification toward that regime.
Organized by regime type — see `config.yaml` for the full list of ~50 keys.

#### Strong Trend scoring

| Key | Default | Description |
|-----|---------|-------------|
| `scoring.strong_trend.return_strong_base` | 3.0 | Base score when \|return\| >= total_return_strong |
| `scoring.strong_trend.return_strong_cap` | 0.70 | Cap on excess return for scaling |
| `scoring.strong_trend.return_strong_scale` | 3.0 | Max bonus from excess return |
| `scoring.strong_trend.return_moderate_base` | 1.0 | Base for moderate return |
| `scoring.strong_trend.return_moderate_scale` | 2.0 | Max bonus for moderate return |
| `scoring.strong_trend.adx_strong_base` | 2.0 | Base when ADX >= adx_strong_trend |
| `scoring.strong_trend.adx_strong_divisor` | 20.0 | Divisor for ADX excess bonus |
| `scoring.strong_trend.adx_moderate_score` | 0.5 | When ADX >= adx_weak |
| `scoring.strong_trend.consistency_high_score` | 2.0 | pct_above_ma strongly one-sided |
| `scoring.strong_trend.consistency_moderate_score` | 0.5 | Moderately one-sided |
| `scoring.strong_trend.extended_distance_score` | 1.0 | Price far from MA |
| `scoring.strong_trend.direction_change_low` | 0.4 | Threshold for low direction changes |
| `scoring.strong_trend.direction_change_low_score` | 1.0 | Score when changes < threshold |
| `scoring.strong_trend.direction_change_mid` | 0.5 | Mid threshold |
| `scoring.strong_trend.direction_change_mid_score` | 0.3 | Score when changes < mid threshold |

#### Mean Reverting scoring

| Key | Default | Description |
|-----|---------|-------------|
| `scoring.mean_reverting.return_strong_penalty` | 2.0 | Penalty when \|return\| >= strong |
| `scoring.mean_reverting.return_moderate_penalty` | 1.0 | Penalty for moderate return |
| `scoring.mean_reverting.return_small_bonus` | 1.5 | Bonus when return is small |
| `scoring.mean_reverting.adx_low_score` | 2.0 | ADX < adx_weak |
| `scoring.mean_reverting.adx_moderate_score` | 1.0 | ADX < adx_strong_trend |
| `scoring.mean_reverting.pct_away_tight` | 15 | pct_above_ma within 50 +/- this = range-bound |
| `scoring.mean_reverting.pct_away_tight_score` | 2.0 | Score for tight range |
| `scoring.mean_reverting.pct_away_moderate` | 25 | Moderate range |
| `scoring.mean_reverting.pct_away_moderate_score` | 1.0 | Score for moderate range |
| `scoring.mean_reverting.atr_below_high_score` | 0.5 | ATR% below high threshold |
| `scoring.mean_reverting.atr_below_low_score` | 0.5 | ATR% below low threshold (additional) |
| `scoring.mean_reverting.price_ma_close_threshold` | 0.03 | Price within this % of MA |
| `scoring.mean_reverting.price_ma_close_score` | 1.0 | Score when close to MA |

#### Volatile Choppy scoring

| Key | Default | Description |
|-----|---------|-------------|
| `scoring.volatile_choppy.atr_high_base` | 2.0 | Base when ATR% >= high |
| `scoring.volatile_choppy.atr_high_scale` | 30.0 | Multiplier for excess ATR% |
| `scoring.volatile_choppy.atr_moderate_score` | 0.5 | ATR% >= low |
| `scoring.volatile_choppy.direction_change_high_score` | 2.0 | High direction changes |
| `scoring.volatile_choppy.direction_change_moderate` | 0.45 | Moderate threshold |
| `scoring.volatile_choppy.direction_change_moderate_score` | 1.0 | Score for moderate |
| `scoring.volatile_choppy.low_adx_small_return_score` | 0.5 | Low ADX + small return |
| `scoring.volatile_choppy.wide_bb_score` | 1.0 | BB width > expansion percentile |

#### Breakout Transition scoring

| Key | Default | Description |
|-----|---------|-------------|
| `scoring.breakout.bb_squeeze_score` | 2.5 | BB width <= squeeze percentile |
| `scoring.breakout.adx_moderate_score` | 1.0 | ADX in moderate zone |
| `scoring.breakout.direction_change_threshold` | 0.40 | Min changes for consolidation |
| `scoring.breakout.low_atr_high_changes_score` | 1.0 | Low ATR + high changes |
| `scoring.breakout.price_ma_close_threshold` | 0.03 | Price near MA during consolidation |
| `scoring.breakout.bb_consolidation_percentile` | 40 | BB width below this + near MA |
| `scoring.breakout.consolidation_score` | 0.5 | Score for consolidation |

### Strategy Adaptation per Regime

Section: `regime.strategy_adaptation`

Each regime has adaptation rules that override base strategy behavior during
backtesting.

#### Strong Trend adaptation

| Key | Default | Description |
|-----|---------|-------------|
| `strategy_adaptation.strong_trend.use_trailing_stop` | true | Trail stop instead of score-based exits |
| `strategy_adaptation.strong_trend.trailing_stop_atr_mult` | 8.0 | Trailing stop = N x ATR |
| `strategy_adaptation.strong_trend.ignore_score_entries` | true | Don't use score thresholds for entry |
| `strategy_adaptation.strong_trend.hold_with_trend` | true | Stay in position while trend persists |
| `strategy_adaptation.strong_trend.min_distance` | 0.01 | Price must be 1%+ from MA for entry |
| `strategy_adaptation.strong_trend.min_score` | 3.5 | Don't enter when indicators are strongly bearish |
| `strategy_adaptation.strong_trend.respect_trend_direction` | true | Only enter in direction of long-term trend |

#### Mean Reverting adaptation

| Key | Default | Description |
|-----|---------|-------------|
| `strategy_adaptation.mean_reverting.use_trailing_stop` | false | Use score-based entries/exits |
| `strategy_adaptation.mean_reverting.tighten_thresholds` | true | Narrow HOLD zone for more trades |
| `strategy_adaptation.mean_reverting.threshold_adjustment` | 0.3 | Narrow thresholds by this on each side |

#### Volatile Choppy adaptation

| Key | Default | Description |
|-----|---------|-------------|
| `strategy_adaptation.volatile_choppy.reduce_position_size` | true | Halve position size |
| `strategy_adaptation.volatile_choppy.position_size_mult` | 0.5 | Position size multiplier |
| `strategy_adaptation.volatile_choppy.widen_stops` | true | Widen stop loss |
| `strategy_adaptation.volatile_choppy.stop_loss_mult` | 1.5 | Stop loss multiplier |

#### Breakout Transition adaptation

| Key | Default | Description |
|-----|---------|-------------|
| `strategy_adaptation.breakout_transition.use_momentum_entry` | true | Enter on breakout confirmation |
| `strategy_adaptation.breakout_transition.breakout_atr_mult` | 1.5 | Price must move N x ATR from squeeze level |
| `strategy_adaptation.breakout_transition.require_volume_surge` | true | Require above-avg volume |
| `strategy_adaptation.breakout_transition.volume_surge_mult` | 1.3 | Volume must be N x average |
| `strategy_adaptation.breakout_transition.avg_volume_window` | 20 | Rolling window for average volume baseline |

### Sub-Type Classification

Section: `regime.sub_type`

Within each regime, stocks are further classified along two axes into a
**Volatility × Momentum 2×2 matrix**. This provides more granular labels
for display in both the CLI and dashboard.

|                    | High Momentum (≥ 20%) | Low Momentum (< 20%) |
|--------------------|----------------------|----------------------|
| **High Vol (≥ 3.5% ATR)** | **Explosive Mover** (TSLA, NVDA, MARA) | **Volatile Directionless** (AMD, RIOT) |
| **Low Vol (< 3.5% ATR)** | **Steady Compounder** (KO, JPM, AAPL, XOM) | **Stagnant** (PEP, PG, MSFT) |

The sub-type label and description appear in the regime section of both
CLI analysis output and backtest results. They help you quickly understand
a stock's character beyond just the primary regime.

#### Sub-type thresholds

| Key | Default | Description |
|-----|---------|-------------|
| `regime.sub_type.atr_pct_threshold` | 0.035 | ATR% ≥ this → high volatility |
| `regime.sub_type.momentum_threshold` | 0.20 | \|total_return\| ≥ this → high momentum |

#### Sub-type strategy overrides

Section: `regime.strategy_adaptation.strong_trend.sub_types`

Override slots exist for each sub-type (`explosive_mover`,
`steady_compounder`, `volatile_directionless`, `stagnant`). Values merge
on top of the base regime params via shallow merge.

**Most overrides are empty** — the base `strong_trend` params
(trailing stop at 8× ATR, etc.) handle most sub-types well. The one
exception is `steady_compounder`, which uses `trailing_stop_atr_mult: 6.0`
(a tighter trail suited to low-volatility trending names). Other override
slots remain available for future tuning.

> **Design constraint**: Sub-type classification uses trailing-window
> metrics that can change mid-trade. Overrides should be limited to
> **entry-time parameters** (like `min_distance`, `min_score`), never
> ongoing position management params (`trailing_stop_atr_mult`,
> `reduce_position_size`) — those cause harmful mid-trade disruption.

---

## Trading Mode Suitability

Section: `suitability`

Auto-detects whether a stock is suitable for shorting, active trading, or
should be held only. Three modes:

| Mode | Behavior |
|------|----------|
| `long_short` | Both long and short positions allowed |
| `long_only` | Only long positions; goes to cash when bearish |
| `hold_only` | Unsuitable for active trading; analysis display only |

### Suitability thresholds

| Key | Default | Description |
|-----|---------|-------------|
| `suitability.mode_override` | "auto" | "auto", "long_short", "long_only", "hold_only" |
| `suitability.min_volume` | 100000 | Avg daily volume below this = hold_only |
| `suitability.min_atr_pct` | 0.005 | ATR/price below this = hold_only (0.5%) |
| `suitability.min_adx_for_short` | 25.0 | ADX below this = no shorting (long_only) |
| `suitability.min_atr_for_short` | 0.01 | ATR/price below this = no shorting (1%) |
| `suitability.min_volume_for_short` | 500000 | Avg volume below this = no shorting |
| `suitability.trend_ma_period` | 200 | Long-term MA period for structural trend check |
| `suitability.max_pct_above_ma` | 65.0 | Price above MA > this % of time = long_only |
| `suitability.atr_period` | 14 | ATR calculation period |

### Advanced (rarely changed)

| Key | Default | Description |
|-----|---------|-------------|
| `suitability.adx_min_data_mult` | 3 | Require period x this many bars for ADX calculation |
| `suitability.insufficient_data_pct` | 50.0 | pct_above_ma fallback when data < trend_ma_period (50 = no bias) |

---

## Objective Presets

Section: `objectives`

Named presets that override specific base config values. Select via CLI:

```bash
python main.py AAPL --objective long_term
python main.py AAPL -o short_term -b
python main.py AAPL -o day_trading -b -i 5m --start 2026-02-17
```

Presets are partial configs — only keys listed in the preset are
overridden. All other settings keep their base values.

### Built-in presets

| Preset | Holding period | Key differences |
|--------|---------------|-----------------|
| `long_term` | Weeks to months | Slower periods (RSI 21, MACD 19/39), wider stops (10% / 100% take-profit), heavier MA/ADX/volume weights, 200-bar warmup |
| `short_term` | Days to weeks | Faster periods (RSI 9, MACD 8/17), tighter stops (3% / 20%), heavier MACD/Stochastic weights, 60-bar warmup |
| `day_trading` | Intraday only | Very fast periods (RSI 5, MACD 5/13), tight stops (1.5% / 3%), EOD flattening, rebalance every bar, 30-bar warmup |

### Creating custom presets

Add a new key under `objectives` in `config.yaml`:

```yaml
objectives:
  my_preset:
    description: "My custom preset"
    rsi:
      period: 10
    strategy:
      stop_loss_pct: 0.04
      rebalance_interval: 3
    backtest:
      warmup_bars: 80
```

Then use it:

```bash
python main.py AAPL -o my_preset -b
```

---

## Multi-Timeframe Analysis

Section: `multi_timeframe`

Analyzes the same ticker across multiple timeframes and combines the
signals. Each timeframe gets its own independent analysis; the final score
is a weighted blend adjusted by inter-timeframe agreement.

| Key | Default | Description |
|-----|---------|-------------|
| `multi_timeframe.timeframes` | ["1d", "1wk", "1mo"] | List of timeframes to analyze |
| `multi_timeframe.weights.1d` | 0.5 | Weight for daily timeframe |
| `multi_timeframe.weights.1wk` | 0.3 | Weight for weekly timeframe |
| `multi_timeframe.weights.1mo` | 0.2 | Weight for monthly timeframe |
| `multi_timeframe.periods.1d` | "2y" | Data period to fetch for daily |
| `multi_timeframe.periods.1wk` | "5y" | Data period to fetch for weekly |
| `multi_timeframe.periods.1mo` | "max" | Data period to fetch for monthly |
| `multi_timeframe.agreement_multipliers.aligned` | 1.0 | Multiplier when all timeframes agree on direction |
| `multi_timeframe.agreement_multipliers.mixed` | 0.7 | Multiplier when timeframes are mixed |
| `multi_timeframe.agreement_multipliers.conflicting` | 0.4 | Multiplier when timeframes conflict |

---

## Tips & Troubleshooting

### Score compression

The composite score (weighted average of 8 indicators) has much lower
variance than individual indicators. Typical effective range is ~3.7-5.9.
**Set strategy thresholds within that actual range**, or use
`threshold_mode: "percentile"` for self-calibrating thresholds.

### Warmup eating your data

With default `warmup_bars: 50` on a 6-month dataset (~126 bars), warmup
consumes ~50% of your data (capped by `max_warmup_ratio: 0.5`). Solutions:
- Use a longer period (`--period 2y` or more)
- Lower `warmup_bars` (but must be >= longest indicator lookback)
- Use `--objective short_term` which sets `warmup_bars: 60`

### yfinance quirks

- `10y` period is **not** supported — use `5y` or `max`, or `--start` dates
- Day-count periods like `60d` are **not** supported
- Our data provider rejects `5d` — minimum is `1mo`
- Intraday data has limited history (see table above)

### LSP type errors

The `ta`, `yfinance`, `plotly`, and `streamlit` libraries have poor type
stubs. Errors about DataFrame/Series/ndarray type mismatches from your
editor's language server are benign and can be ignored.

### Backtest performance issues

If your backtest underperforms the buy-and-hold:
1. Check that warmup isn't eating most of your data
2. Score compression may prevent entries — try `threshold_mode: "percentile"`
3. Tune thresholds in `config.yaml` — all 169+ parameters are now
   configurable without code changes
4. Try different objectives: `--objective long_term` for trending stocks

### Config access patterns (for developers)

```python
cfg = Config.load()                # from config.yaml
cfg = Config.from_dict(data)       # from arbitrary dict
cfg.get("key")                     # top-level only (no dot-notation)
cfg.section("strategy").get("key") # nested access
cfg.normalized_weights()           # auto-normalized indicator weights
```
