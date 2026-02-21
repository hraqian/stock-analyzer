# Stock Technical Analysis Tool — User Guide

---

## Table of Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Quick Start](#3-quick-start)
4. [Command-Line Reference](#4-command-line-reference)
5. [Technical Indicators](#5-technical-indicators)
   - 5.1 [RSI (Relative Strength Index)](#51-rsi-relative-strength-index)
   - 5.2 [MACD (Moving Average Convergence Divergence)](#52-macd-moving-average-convergence-divergence)
   - 5.3 [Bollinger Bands](#53-bollinger-bands)
   - 5.4 [Moving Averages](#54-moving-averages)
   - 5.5 [Stochastic Oscillator](#55-stochastic-oscillator)
   - 5.6 [ADX (Average Directional Index)](#56-adx-average-directional-index)
   - 5.7 [Volume (OBV)](#57-volume-obv)
   - 5.8 [Fibonacci Retracement](#58-fibonacci-retracement)
6. [Pattern Signal Detection](#6-pattern-signal-detection)
   - 6.1 [Gaps](#61-gaps)
   - 6.2 [Candlesticks](#62-candlesticks)
   - 6.3 [Volume-Range Correlation](#63-volume-range-correlation)
   - 6.4 [Spikes](#64-spikes)
   - 6.5 [Pattern Composite Scoring](#65-pattern-composite-scoring)
   - 6.6 [Adding New Patterns](#66-adding-new-patterns)
7. [Composite Scoring System](#7-composite-scoring-system)
8. [Support and Resistance Levels](#8-support-and-resistance-levels)
9. [Backtesting Engine](#9-backtesting-engine)
   - 9.1 [How Backtesting Works](#91-how-backtesting-works)
   - 9.2 [Strategy: Signal Generation](#92-strategy-signal-generation)
   - 9.3 [Position Management](#93-position-management)
   - 9.4 [Risk Management](#94-risk-management)
   - 9.5 [Performance Metrics](#95-performance-metrics)
10. [Trading Mode Suitability Detection](#10-trading-mode-suitability-detection)
11. [Trading Objective Presets](#11-trading-objective-presets)
    - 11.1 [Long-Term (Position Trading)](#111-long-term-position-trading)
    - 11.2 [Short-Term (Swing Trading)](#112-short-term-swing-trading)
    - 11.3 [Day Trading (Intraday)](#113-day-trading-intraday)
    - 11.4 [Custom Presets](#114-custom-presets)
12. [Configuration](#12-configuration)
    - 12.1 [Config File Loading](#121-config-file-loading)
    - 12.2 [Full Configuration Reference](#122-full-configuration-reference)
    - 12.3 [Generating and Validating Config](#123-generating-and-validating-config)
13. [Data Provider and Limitations](#13-data-provider-and-limitations)
14. [Display and Output](#14-display-and-output)
15. [Extending the Tool: Adding New Indicators](#15-extending-the-tool-adding-new-indicators)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. Overview

The Stock Technical Analysis Tool is a Python command-line application that performs comprehensive technical analysis on any stock. It fetches historical OHLCV (Open, High, Low, Close, Volume) data, computes eight technical indicators — each scored on a 0–10 scale — detects four types of pattern signals (also scored 0–10), calculates support/resistance levels, and produces weighted composite scores for both indicators and patterns. It also includes a full backtesting engine with configurable strategies, automatic trading mode detection, and trading objective presets for different investment horizons.

**Key Features:**

- **8 technical indicators** with configurable parameters, each scored 0 (strongly bearish) to 10 (strongly bullish)
- **4 pattern signal detectors** — gaps, candlesticks, volume-range correlation, and spikes — scored independently from indicators
- **Dual composite scoring** — separate indicator and pattern composite scores, combined via configurable weighted blend or gate mode
- **Support/resistance detection** using pivot points and/or fractal analysis
- **Backtesting engine** with bar-by-bar simulation, stop-loss, take-profit, slippage, and commission modeling
- **Trading mode auto-detection** that determines whether a stock is suitable for long/short trading, long-only, or hold-only
- **Trading objective presets** for long-term, short-term, and day trading — each overriding indicator periods, strategy thresholds, and weights
- **Plugin architecture** — add new indicators or patterns by dropping a file into the `indicators/` or `patterns/` directory
- **Fully configurable** via `config.yaml` — all scoring thresholds, weights, strategy parameters, indicator settings, and pattern parameters can be tuned without touching code

---

## 2. Installation

### Prerequisites

- Python 3.10 or later
- pip (Python package manager)

### Steps

1. Clone or download the project to your local machine.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

The required packages are:

| Package | Version | Purpose |
|---------|---------|---------|
| yfinance | >= 0.2.31 | Market data from Yahoo Finance |
| pandas | >= 2.0.0 | Data manipulation |
| numpy | >= 1.24.0 | Numerical computation |
| ta | >= 0.11.0 | Technical analysis library |
| rich | >= 13.0.0 | Terminal formatting and display |
| pyyaml | >= 6.0.1 | YAML configuration parsing |
| scipy | >= 1.11.0 | Scientific computation |

3. (Optional) Generate a default configuration file:

```bash
python main.py --generate-config
```

This creates `config.yaml` in the current directory with all default settings.

---

## 3. Quick Start

### Basic Analysis

Run a 6-month analysis (the default) for Apple:

```bash
python main.py AAPL
```

Run a 1-year analysis for Tesla:

```bash
python main.py TSLA --period 1y
```

Run analysis for a custom date range:

```bash
python main.py MSFT --start 2020-01-01 --end 2023-12-31
```

### Selective Indicators

Run only RSI, MACD, and ADX:

```bash
python main.py AAPL --indicators rsi,macd,adx
```

### Backtesting

Run a backtest over 2 years:

```bash
python main.py AAPL --backtest --period 2y
```

Run a backtest with a specific trading mode:

```bash
python main.py AAPL --backtest --mode long_only
```

### Objective Presets

Use long-term indicator settings:

```bash
python main.py AAPL --objective long_term
```

Day trading backtest with 5-minute bars:

```bash
python main.py AAPL -o day_trading -b -i 5m --start 2026-02-10
```

---

## 4. Command-Line Reference

```
usage: stock_analyzer [-h] [--period PERIOD] [--interval INTERVAL]
                      [--indicators LIST] [--start DATE] [--end DATE]
                      [--backtest] [--mode MODE] [--objective NAME]
                      [--config PATH] [--generate-config] [--validate-config]
                      [--list-indicators]
                      [TICKER]
```

### Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `TICKER` | — | *(required for analysis)* | Stock ticker symbol (e.g., `AAPL`, `TSLA`, `MSFT`). Not required for `--generate-config`, `--validate-config`, or `--list-indicators`. |
| `--period` | `-p` | `6mo` | Data period to fetch. Options: `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `ytd`, `max`. |
| `--interval` | `-i` | `1d` | Bar interval. **Daily+:** `1d`, `5d`, `1wk`, `1mo`, `3mo`. **Intraday:** `1m`, `2m`, `5m`, `15m`, `30m`, `60m`, `90m`, `1h`. |
| `--indicators` | — | all | Comma-separated list of indicators to run (e.g., `rsi,macd,adx`). |
| `--start` | `-s` | — | Start date in `YYYY-MM-DD` format. Overrides `--period` when specified. |
| `--end` | `-e` | today | End date in `YYYY-MM-DD` format. Only used when `--start` is specified. |
| `--backtest` | `-b` | off | Run a backtest using the score-based strategy. |
| `--mode` | `-m` | — | Force trading mode for backtest: `auto`, `long_short`, `long_only`, `hold_only`. Overrides config. |
| `--objective` | `-o` | — | Apply a trading objective preset (e.g., `long_term`, `short_term`, `day_trading`). |
| `--config` | `-c` | — | Path to a custom `config.yaml` file. |
| `--generate-config` | — | — | Generate a fresh default `config.yaml` in the current directory and exit. |
| `--validate-config` | — | — | Validate `config.yaml` and report any issues, then exit. |
| `--list-indicators` | — | — | List all available indicator keys and exit. |

### Validation Rules

- The `day_trading` objective requires an intraday interval (`-i 5m`, `-i 15m`, etc.). The tool will exit with an error if you use `day_trading` with a daily interval.
- Intraday intervals with `--period` are validated against yfinance data limits (see [Section 13](#13-data-provider-and-limitations)).
- `--end` requires `--start` to be specified.

---

## 5. Technical Indicators

All indicators produce a score from **0.0** (strongly bearish) to **10.0** (strongly bullish), with **5.0** being neutral. Every scoring parameter is configurable in `config.yaml`.

### 5.1 RSI (Relative Strength Index)

**What it measures:** A momentum oscillator (0–100) that measures the speed and magnitude of recent price changes to evaluate overbought or oversold conditions.

**Interpretation:** Low RSI (oversold) suggests a potential bullish reversal opportunity, so it receives a high score. High RSI (overbought) suggests a potential bearish reversal, so it receives a low score.

**Default Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `period` | 14 | RSI lookback window (bars) |
| `thresholds.oversold` | 30 | RSI below this is oversold |
| `thresholds.overbought` | 70 | RSI above this is overbought |
| `scores.oversold_score` | 9.0 | Score when RSI ≤ oversold |
| `scores.overbought_score` | 1.0 | Score when RSI ≥ overbought |
| `scores.neutral_score` | 5.0 | Score when RSI = 50 |

**Scoring logic:**

- RSI ≤ 30 → score 9.0
- RSI 30–50 → linearly interpolated from 9.0 down to 5.0
- RSI 50–70 → linearly interpolated from 5.0 down to 1.0
- RSI ≥ 70 → score 1.0

### 5.2 MACD (Moving Average Convergence Divergence)

**What it measures:** A trend-following momentum indicator that shows the relationship between two exponential moving averages of price. The histogram (MACD line minus signal line) indicates momentum strength and direction.

**Interpretation:** A positive histogram indicates bullish momentum; a negative histogram indicates bearish momentum. Recent crossovers (histogram changing sign) add conviction.

**Default Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fast_period` | 12 | Fast EMA period |
| `slow_period` | 26 | Slow EMA period |
| `signal_period` | 9 | Signal line EMA period |
| `scoring.strong_bullish_pct` | 0.005 | Histogram > 0.5% of price → strong bullish |
| `scoring.moderate_bullish_pct` | 0.001 | Histogram > 0.1% of price → moderate bullish |
| `scoring.strong_bearish_pct` | -0.005 | Histogram < -0.5% of price → strong bearish |
| `scoring.moderate_bearish_pct` | -0.001 | Histogram < -0.1% of price → moderate bearish |
| `scoring.crossover_lookback` | 5 | Bars to check for recent crossover |
| `scoring.bullish_cross_bonus` | 1.5 | Score bonus for bullish crossover |
| `scoring.bearish_cross_penalty` | 1.5 | Score penalty for bearish crossover |

**Scoring logic:**

1. The histogram is normalized as a percentage of price.
2. The normalized value is mapped to a base score across six linear zones (0–10 scale, with thresholds at the configured percentages).
3. If a bullish crossover occurred within the lookback window (histogram went from negative to positive), +1.5 is added to the score.
4. If a bearish crossover occurred, -1.5 is subtracted.
5. The final score is clamped to [0, 10].

### 5.3 Bollinger Bands

**What it measures:** Volatility indicator using standard deviation bands around a simple moving average. The %B metric measures where the current price sits relative to the bands (0 = at lower band, 1 = at upper band).

**Interpretation:** Price near the lower band suggests oversold conditions (high score); price near the upper band suggests overbought conditions (low score). A volatility squeeze (narrow bands) can signal an impending breakout.

**Default Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `period` | 20 | SMA period for the middle band |
| `std_dev` | 2.0 | Standard deviations for band width |
| `scoring.lower_zone` | 0.20 | %B below this → near support (score 7.5–9.5) |
| `scoring.upper_zone` | 0.80 | %B above this → near resistance (score 0.5–2.5) |
| `scoring.squeeze_threshold` | 0.02 | Band width / price below this = squeeze |

**Scoring logic:**

| %B Range | Score Range |
|----------|-------------|
| ≤ 0.0 (below lower band) | 9.5 |
| 0.0 – 0.20 | 9.5 → 7.5 |
| 0.20 – 0.50 | 7.5 → 5.0 |
| 0.50 – 0.80 | 5.0 → 2.5 |
| 0.80 – 1.0 | 2.5 → 0.5 |
| > 1.0 (above upper band) | 0.5 |

### 5.4 Moving Averages

**What it measures:** Trend direction and strength based on the price's position relative to multiple moving averages (SMA or EMA), the alignment of the averages with each other, and the occurrence of golden/death crosses.

**Interpretation:** Price above all MAs with bullish alignment (shorter MA > longer MA) is strongly bullish. A golden cross (50-day MA crossing above 200-day MA) adds conviction; a death cross subtracts.

**Default Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `periods` | [20, 50, 200] | MA periods to compute |
| `type` | "sma" | "sma" or "ema" |
| `scoring.price_above_ma_points` | 1.5 | Points per MA that price is above |
| `scoring.ma_aligned_bullish_points` | 1.0 | Points per consecutive bullish alignment |
| `scoring.golden_cross_bonus` | 2.0 | Bonus for golden cross within lookback |
| `scoring.death_cross_penalty` | 2.0 | Penalty for death cross within lookback |
| `scoring.cross_lookback` | 10 | Bars to search for golden/death cross |
| `scoring.max_raw_score` | 9.5 | Maximum score before normalization |

**Scoring logic:**

1. Start at 0 raw points.
2. Add 1.5 points for each MA that the current price is above.
3. Add 1.0 point for each consecutive pair of MAs in bullish alignment (shorter > longer).
4. Add 2.0 for a golden cross; subtract 2.0 for a death cross.
5. Compute the theoretical maximum possible raw score.
6. Normalize: `score = (raw / theoretical_max) × 9.5`, clamped to [0, 10].

> **Note:** When data is too short for a given MA (e.g., 6 months of data is insufficient for SMA-200), that MA is shown as "N/A" and excluded from scoring.

### 5.5 Stochastic Oscillator

**What it measures:** A momentum indicator that compares a stock's closing price to its high-low range over a period. %K is the fast line (0–100); %D is the smoothed signal line.

**Interpretation:** Like RSI — low values (oversold) score high; high values (overbought) score low. Crossovers of %K above/below %D add/subtract conviction.

**Default Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k_period` | 14 | %K lookback period |
| `d_period` | 3 | %D smoothing period |
| `smooth_k` | 3 | Additional %K smoothing |
| `thresholds.oversold` | 20 | %K below this is oversold |
| `thresholds.overbought` | 80 | %K above this is overbought |
| `scores.oversold_score` | 9.0 | Score at oversold |
| `scores.overbought_score` | 1.0 | Score at overbought |
| `scores.neutral_score` | 5.0 | Score at midpoint |
| `scores.bullish_cross_bonus` | 1.0 | Bonus when %K crosses above %D |
| `scores.bearish_cross_penalty` | 1.0 | Penalty when %K crosses below %D |

### 5.6 ADX (Average Directional Index)

**What it measures:** Trend strength (ADX value) and trend direction (+DI vs. -DI). ADX itself only measures *how strong* a trend is, not its direction. The directional indicators (+DI and -DI) determine whether bulls or bears are dominant.

**Interpretation:** The score combines direction (+DI vs. -DI spread) with a strength multiplier (ADX value). In a strong uptrend (+DI >> -DI, high ADX), the score is high. In a strong downtrend, it is low. In a weak/trendless market, the score is pulled toward 5.0 (neutral).

**Default Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `period` | 14 | ADX/DI calculation period |
| `thresholds.weak` | 20 | ADX below this → weak trend |
| `thresholds.moderate` | 40 | ADX 20–40 → moderate trend |
| `scoring.weak_multiplier` | 0.6 | Multiplier for weak trends |
| `scoring.moderate_multiplier` | 0.85 | Multiplier for moderate trends |
| `scoring.strong_multiplier` | 1.0 | Multiplier for strong trends |
| `scoring.max_directional_spread` | 25 | +DI − -DI spread for full score |

**Scoring logic:**

1. Compute a directional score from the DI spread: linear interpolation from 0 to 10 based on `(+DI − -DI)` ranging from `-25` to `+25`.
2. Determine a multiplier based on ADX value (0.6 for weak, up to 1.0 for strong).
3. Final score: `5.0 + (directional_score − 5.0) × multiplier`. This pulls the score toward 5.0 in trendless markets and allows full directional expression in strong trends.

### 5.7 Volume (OBV)

**What it measures:** On-Balance Volume (OBV) trends relative to price trends. OBV adds volume on up days and subtracts on down days, creating a cumulative volume flow indicator.

**Interpretation:** When OBV confirms the price trend (both rising or both falling), the volume supports the move, producing a strong directional score. When OBV diverges from price, the score is neutral (5.0), suggesting the move may lack conviction.

**Default Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `obv_trend_period` | 20 | EMA period for OBV trend |
| `price_trend_period` | 20 | EMA period for price trend |
| `scoring.confirmation_bullish_max` | 9.5 | Strong bullish confirmation score |
| `scoring.confirmation_bullish_min` | 6.5 | Weak bullish confirmation score |
| `scoring.confirmation_bearish_max` | 3.5 | Weak bearish confirmation score |
| `scoring.confirmation_bearish_min` | 0.5 | Strong bearish confirmation score |
| `scoring.divergence_score` | 5.0 | Score when OBV diverges from price |
| `scoring.obv_strong_change_pct` | 10.0 | OBV change ≥ 10% → strong reading |
| `scoring.obv_weak_change_pct` | 1.0 | OBV change ≤ 1% → weak reading |

**Scoring logic:**

- If OBV and price trends **diverge**: score = 5.0.
- If both are **rising** (bullish confirmation): score scaled from 6.5 to 9.5 based on OBV change magnitude (1%–10%).
- If both are **falling** (bearish confirmation): score scaled from 3.5 to 0.5 based on OBV change magnitude.

### 5.8 Fibonacci Retracement

**What it measures:** Where the current price sits relative to Fibonacci retracement levels calculated from the swing high/low over the lookback period.

**Interpretation:** Shallow retracements (near 0.236) suggest the prior trend is still strong (bullish score). Deep retracements (near 0.786) suggest the trend may be reversing (bearish score). If price is not near any specific Fibonacci level, scoring falls back to the overall position within the swing range.

**Default Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `swing_lookback` | 60 | Bars to find swing high/low |
| `levels` | [0.236, 0.382, 0.5, 0.618, 0.786] | Fibonacci retracement levels |
| `scoring.proximity_pct` | 0.015 | Within 1.5% of a level = "at that level" |
| `scoring.level_scores` | {0.236: 8.0, 0.382: 7.0, 0.5: 5.0, 0.618: 3.0, 0.786: 1.5} | Score per level |
| `scoring.no_level_score` | 5.0 | Default when not near any level |
| `scoring.range_low_score` | 2.0 | Score at bottom of swing range |
| `scoring.range_high_score` | 8.0 | Score at top of swing range |

**Scoring logic:**

1. If price is within 1.5% of a Fibonacci level, return that level's score (e.g., near 0.236 → 8.0, near 0.618 → 3.0).
2. If not near any level, compute the price's position within the swing range (0 = swing low, 1 = swing high) and linearly interpolate from 2.0 (bottom) to 8.0 (top).

---

## 6. Pattern Signal Detection

Patterns form a **second scoring axis** alongside the indicator composite. While indicators measure *where the stock is* (overbought, trending, near support), patterns measure *what just happened* (a gap, a reversal candle, a volume spike). The two systems are scored independently and combined in the strategy layer.

All pattern detectors produce a score from **0.0** (strongly bearish) to **10.0** (strongly bullish), with **5.0** meaning no pattern detected or neutral. Every parameter is configurable in `config.yaml`.

### 6.1 Gaps

**What it detects:** Price gaps between consecutive bars — where the open of one bar is meaningfully above or below the close of the previous bar.

**Gap classification:**

| Type | Criteria | Significance |
|------|----------|-------------|
| **Breakaway** | Gap with volume surge AND trend-aligned direction | Strong trend initiation signal |
| **Runaway** | Gap in the direction of the existing trend (no volume surge needed) | Trend continuation |
| **Exhaustion** | Gap against the current trend | Possible trend reversal |
| **Common** | Gap with low volume, no clear trend context | Low significance |

**Scoring logic:** Recent gaps (within the lookback window) are weighted by recency (more recent = higher weight) and type weight. Bullish gaps push the score above 5.0; bearish gaps push below. The `max_signal_strength` config caps the maximum deviation from neutral.

**Default Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback` | 20 | Bars to search for gaps |
| `min_gap_pct` | 0.005 | Minimum gap size (0.5% of price) to qualify |
| `volume_surge_mult` | 1.5 | Volume must be this × average for breakaway |
| `trend_period` | 20 | Period for determining trend direction |
| `type_weights` | common: 0.3, runaway: 0.7, breakaway: 1.0, exhaustion: 0.5 | Weight per gap type |
| `max_signal_strength` | 3.0 | Max score deviation from 5.0 |

### 6.2 Candlesticks

**What it detects:** Classic single-bar and two-bar candlestick patterns, interpreted in context of the prevailing trend direction.

**Detected patterns:**

| Pattern | Criteria | Bullish/Bearish |
|---------|----------|----------------|
| **Doji** | Body size < 5% of bar range | Neutral (reversal hint) |
| **Hammer** | Small body at top, long lower shadow (≥ 2× body), in downtrend | Bullish reversal |
| **Hanging Man** | Same shape as hammer, but in uptrend | Bearish reversal |
| **Inverted Hammer** | Small body at bottom, long upper shadow, in downtrend | Bullish reversal |
| **Shooting Star** | Same shape as inverted hammer, but in uptrend | Bearish reversal |
| **Bullish Engulfing** | Current bar's body fully engulfs previous bar's body, current is up | Bullish reversal |
| **Bearish Engulfing** | Current bar's body fully engulfs previous bar's body, current is down | Bearish reversal |

**Context awareness:** Trend direction (computed over `trend_period` bars) determines whether a pattern is a reversal signal. A hammer in a downtrend is bullish; the same shape in an uptrend is a hanging man (bearish).

**Default Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `doji_threshold` | 0.05 | Body/range ratio below this = doji |
| `shadow_ratio` | 2.0 | Shadow must be this × body size for hammer/star |
| `lookback` | 10 | Bars to search for recent patterns |
| `trend_period` | 10 | Bars for trend direction detection |
| `max_signal_strength` | 3.0 | Max score deviation from 5.0 |

### 6.3 Volume-Range Correlation

**What it detects:** The relationship between price range (high - low) and volume, identifying expansion, contraction, and divergence regimes.

**Regimes:**

| Regime | Condition | Interpretation |
|--------|-----------|----------------|
| **Expansion** | Both range and volume above average | Strong directional move with conviction |
| **Contraction** | Both range and volume below average | Consolidation / indecision |
| **Divergence** | Range and volume disagree (one high, one low) | Move may lack conviction |

**Scoring:** Expansion during a bullish move scores high; expansion during a bearish move scores low. Contraction and divergence score near 5.0 (neutral).

**Default Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `period` | 20 | Lookback for average range and volume |
| `expansion_threshold` | 1.5 | Ratio above this = expansion |
| `contraction_threshold` | 0.6 | Ratio below this = contraction |
| `lookback` | 10 | Bars to analyze for current regime |
| `scoring.expansion_bull` | 8.0 | Score for bullish expansion |
| `scoring.expansion_bear` | 2.0 | Score for bearish expansion |
| `scoring.contraction` | 5.0 | Score for contraction |
| `scoring.divergence` | 5.0 | Score for divergence |

### 6.4 Spikes

**What it detects:** Abnormal price moves (z-score > threshold), then classifies them as confirmed breakouts or failed traps.

**Detection logic:**

1. Compute a z-score of the current bar's return vs. rolling mean/std.
2. If z-score exceeds `spike_std` (default 2.5), a spike is detected.
3. **Confirmation:** If price holds above/below the spike level for `confirm_bars` (default 3) bars, the move is confirmed.
4. **Trap:** If price reverses back through the spike level, it's classified as a failed breakout. Traps generate an *inverse* signal (bullish spike that fails = bearish signal).

**Scoring:** Confirmed bullish spikes push the score above 5.0; confirmed bearish spikes push below. Traps push the score in the opposite direction. The `trap_weight` config controls how strongly traps influence the score.

**Default Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `period` | 20 | Lookback for mean/std of returns |
| `spike_std` | 2.5 | Z-score threshold for spike detection |
| `confirm_bars` | 3 | Bars to wait for confirmation |
| `confirm_pct` | 0.5 | Price must hold this % of spike move |
| `lookback` | 20 | Bars to search for recent spikes |
| `trap_weight` | 0.7 | How strongly traps influence score (0-1) |
| `max_signal_strength` | 3.0 | Max score deviation from 5.0 |

### 6.5 Pattern Composite Scoring

Pattern scores are combined into a weighted composite, independent of the indicator composite.

**Default Pattern Weights:**

| Pattern | Weight |
|---------|--------|
| Gaps | 0.25 |
| Volume-Range | 0.30 |
| Candlesticks | 0.25 |
| Spikes | 0.20 |

Weights are normalized to sum to 1.0, just like indicator weights.

**How patterns combine with indicators in the strategy:**

The strategy supports two combination modes (configured in the `strategy` section):

#### Weighted Mode (default)

```
effective_score = (indicator_weight × indicator_composite + pattern_weight × pattern_composite) / (indicator_weight + pattern_weight)
```

The effective score is then used with the normal threshold logic (fixed or percentile) to generate BUY/SELL/HOLD signals.

Default weights: indicator 70%, pattern 30%. Objective presets adjust these (e.g., day_trading uses 50/50).

#### Gate Mode

Both scores must independently pass their thresholds:

- **BUY** requires: indicator score > `gate_indicator_min` AND pattern score > `gate_pattern_min`
- **SELL** requires: indicator score < `gate_indicator_max` AND pattern score < `gate_pattern_max`
- Otherwise: **HOLD**

Gate mode is more conservative — it only trades when both systems agree.

**Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `combination_mode` | "weighted" | "weighted" or "gate" |
| `indicator_weight` | 0.7 | Indicator weight in blended score |
| `pattern_weight` | 0.3 | Pattern weight in blended score |
| `gate_indicator_min` | 5.5 | Indicator must exceed this for BUY (gate mode) |
| `gate_indicator_max` | 4.5 | Indicator must be below this for SELL (gate mode) |
| `gate_pattern_min` | 5.5 | Pattern must exceed this for BUY (gate mode) |
| `gate_pattern_max` | 4.5 | Pattern must be below this for SELL (gate mode) |

### 6.6 Adding New Patterns

The pattern plugin architecture mirrors the indicator architecture. To add a new pattern:

1. Create `patterns/my_pattern.py`
2. Subclass `BasePattern` and implement `detect()`, `score()`, `summary()`
3. Set class attributes: `name`, `config_key`
4. Add configuration in `config.yaml` and `config.py`
5. Add a weight in `overall_patterns.weights`

The `PatternRegistry` auto-discovers the new class — no other code changes needed. See [Section 15](#15-extending-the-tool-adding-new-indicators) for the equivalent indicator walkthrough.

---

## 7. Composite Scoring System

The composite score is a **weighted average** of all indicator scores.

### Default Weights

| Indicator | Weight |
|-----------|--------|
| RSI | 0.15 |
| MACD | 0.15 |
| Bollinger Bands | 0.10 |
| Moving Averages | 0.20 |
| Stochastic | 0.10 |
| ADX | 0.10 |
| Volume (OBV) | 0.10 |
| Fibonacci | 0.10 |

Weights are automatically **normalized** to sum to 1.0. If you change a single weight, all others adjust proportionally.

### Score Distribution

Because the composite score is an average of eight indicators, its variance is naturally lower than individual indicators. Typical composite scores range from roughly **3.5 to 6.5**. This is expected behavior — the strategy thresholds (see [Section 9.2](#92-strategy-signal-generation)) are calibrated for this narrower range.

### Interpreting the Score

| Score Range | Interpretation | Color |
|-------------|---------------|-------|
| 0.0 – 3.5 | Bearish | Red |
| 3.5 – 6.5 | Neutral | Yellow |
| 6.5 – 10.0 | Bullish | Green |

> **Important:** The composite score shows the current technical posture of the stock. It is not a recommendation. Recommendations only emerge through the backtesting strategy system.

---

## 8. Support and Resistance Levels

The tool identifies key price levels where the stock has historically found support (price floor) or resistance (price ceiling).

### Detection Methods

**Pivot Points** — Classic calculation using the last completed bar's High, Low, and Close:
- Pivot (P) = (H + L + C) / 3
- Support levels: S1, S2, S3 (calculated below pivot)
- Resistance levels: R1, R2, R3 (calculated above pivot)

**Fractal Analysis** — Detects local extrema (swing highs and lows) over the lookback window:
- A local high is confirmed when it is the highest point within N bars on each side.
- A local low is confirmed when it is the lowest point within N bars on each side.

**Both** (default) — Combines pivot and fractal methods, then clusters nearby levels.

### Clustering

Levels within 1.5% of each other are merged into a single cluster. The cluster's price is the average of its constituent levels, and the number of constituent levels is reported as "touches" (more touches = stronger level).

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | "both" | "pivot", "fractal", or "both" |
| `fractal_lookback` | 60 | Bars for fractal detection |
| `fractal_order` | 5 | Bars on each side for local extremum |
| `num_levels` | 4 | Support and resistance levels to display |
| `cluster_pct` | 0.015 | Merge levels within 1.5% |
| `min_touches` | 1 | Minimum touches to confirm a level |

---

## 9. Backtesting Engine

The backtesting engine simulates trading a score-based strategy over historical data to evaluate how well the indicator system would have performed.

### 9.1 How Backtesting Works

1. **Data Fetch:** The full historical OHLCV dataset is loaded for the specified period/date range and interval.
2. **Warmup Period:** The first N bars (default: 200) are used to initialize indicator calculations. No trades are executed during warmup.
3. **Bar-by-Bar Simulation:** Starting after warmup, the engine processes each bar sequentially:
   - Check stop-loss and take-profit on every bar.
   - Every N bars (the rebalance interval), re-compute all indicators and the composite score.
   - Generate a trade signal (BUY, SELL, or HOLD) based on the score.
   - Execute the trade order with simulated slippage and commission.
4. **Final Close:** Any open position at the end of the data is closed automatically.
5. **Metrics:** Performance statistics are calculated from the equity curve and trade history.

> **No Look-Ahead Bias:** At each bar, the engine only uses data up to and including that bar. It never looks at future prices.

### 9.2 Strategy: Signal Generation

The score-based strategy supports two threshold modes:

#### Fixed Mode (default)

The composite score is compared directly against fixed thresholds:

| Composite Score | Signal |
|----------------|--------|
| ≤ 4.5 | **SELL** (enter short or close long) |
| 4.5 – 5.5 | **HOLD** (maintain current position) |
| > 5.5 | **BUY** (enter long or close short) |

#### Percentile Mode

Instead of fixed thresholds, the strategy ranks the current score against a rolling window of recent scores:

| Percentile Rank | Signal |
|----------------|--------|
| ≤ 25th percentile | **SELL** |
| ≥ 75th percentile | **BUY** |
| In between | **HOLD** |

Percentile mode is **self-calibrating** — it adapts to each stock's actual score distribution. If the rolling window has insufficient data (< 80% of the required samples), it falls back to fixed thresholds.

**Configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold_mode` | "fixed" | "fixed" or "percentile" |
| `score_thresholds.short_below` | 4.5 | Fixed: SELL when score ≤ this |
| `score_thresholds.hold_below` | 5.5 | Fixed: HOLD when score ≤ this; BUY above |
| `percentile_thresholds.short_percentile` | 25 | Percentile: SELL at or below this rank |
| `percentile_thresholds.long_percentile` | 75 | Percentile: BUY at or above this rank |
| `percentile_thresholds.lookback_bars` | 60 | Rolling window size |

### 9.3 Position Management

**Position Sizing:**

| Mode | Default | Description |
|------|---------|-------------|
| `"percent_equity"` | 80% of equity | Allocates a fraction of portfolio value per trade |
| `"fixed"` | 100 shares | Fixed number of shares per trade |

**Slippage and Commission:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `slippage_pct` | 0.001 (0.1%) | Price slippage applied against the trader (buys fill higher, sells fill lower) |
| `commission_per_trade` | 0.0 | Flat fee per trade |

**Rebalance Interval:**

The strategy only re-evaluates (re-runs all indicators and generates a new signal) every N bars. Between rebalance points, the only actions are stop-loss and take-profit checks.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rebalance_interval` | 5 | Re-score every N bars |

### 9.4 Risk Management

**Stop-Loss and Take-Profit** are checked on every bar (not just rebalance bars):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stop_loss_pct` | 0.05 (5%) | Close position if unrealized loss exceeds this |
| `take_profit_pct` | 0.20 (20%) | Close position if unrealized gain exceeds this |

**EOD Flattening** (for intraday/day trading):

When `flatten_eod` is enabled, all open positions are force-closed at the end of each trading day. This ensures no overnight risk. On end-of-day bars, only position-closing orders are accepted — no new positions are opened.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `flatten_eod` | false | Force-close all positions at end of each day |

### 9.5 Performance Metrics

The backtest report includes these metrics:

| Metric | Description |
|--------|-------------|
| Total Return % | Overall gain/loss from initial capital |
| Annualized Return % | Return normalized to a yearly rate |
| Max Drawdown % | Largest peak-to-trough decline in equity |
| Sharpe Ratio | Risk-adjusted return (annualized mean return / standard deviation) |
| Win Rate % | Percentage of trades that were profitable |
| Profit Factor | Gross profit / gross loss (higher is better; > 1.0 means profitable overall) |
| Avg Trade P&L % | Average percentage gain/loss per trade |
| Best Trade P&L % | Highest single-trade return |
| Worst Trade P&L % | Lowest single-trade return |

**Annualization for intraday data:** The Sharpe ratio and annualized return use `bars_per_day × 252 trading days` to convert bar-level returns to annual figures.

| Interval | Bars Per Day |
|----------|-------------|
| 1m | 390 |
| 2m | 195 |
| 5m | 78 |
| 15m | 26 |
| 30m | 13 |
| 60m / 1h | 6 |
| 90m | 4 |
| 1d (daily+) | 1 |

---

## 10. Trading Mode Suitability Detection

Before running a backtest, the tool can automatically assess whether a stock is suitable for different trading strategies. This prevents backtesting a short-selling strategy on a stock that fundamentally does not support it.

### Three Modes

| Mode | Description | What's Allowed |
|------|-------------|---------------|
| **LONG SHORT** | Stock has sufficient liquidity, trend strength, and volatility | Long and short positions |
| **LONG ONLY** | Stock can be traded long but shorting is not viable | Long positions only; sells only to close |
| **HOLD ONLY** | Stock is unsuitable for active trading | No trades; analysis display only |

### Priority Order

The trading mode is determined by (highest priority first):

1. **`--mode` CLI flag** — if specified, overrides everything else
2. **`config.yaml suitability.mode_override`** — if set to something other than "auto"
3. **Auto-detection** — the tool analyzes the stock's characteristics

> **Intraday intervals** skip auto-detection entirely and default to **LONG SHORT**, because the suitability thresholds are calibrated for daily bars.

### Auto-Detection Logic

The detector computes four metrics and applies checks in order:

| Metric | How It's Computed |
|--------|-------------------|
| Average Daily Volume | Mean of all volume bars |
| ATR % | Average True Range (14-period) as a percentage of the last closing price |
| ADX | Average Directional Index (14-period) |
| % Above 200-Day MA | Percentage of bars where close > SMA(200) |

**Step 1 — Check for HOLD ONLY (most restrictive):**

| Condition | Threshold | Reason |
|-----------|-----------|--------|
| Avg volume < min_volume | 100,000 | Too illiquid for active trading |
| ATR % < min_atr_pct | 0.5% | Price movement too low for active trading |

If either triggers → **HOLD ONLY**.

**Step 2 — Check for LONG ONLY:**

| Condition | Threshold | Reason |
|-----------|-----------|--------|
| ADX < min_adx_for_short | 25.0 | Trend too weak for effective shorting |
| ATR % < min_atr_for_short | 1.0% | Volatility too low for short-term shorts |
| Avg volume < min_volume_for_short | 500,000 | Insufficient liquidity for shorting |
| % above MA > max_pct_above_ma | 65% | Long-term uptrend makes shorting unprofitable |

If any triggers → **LONG ONLY**.

**Step 3 — Otherwise → LONG SHORT.**

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode_override` | "auto" | Force a specific mode (bypasses auto-detection) |
| `min_volume` | 100,000 | Hold-only threshold for avg daily volume |
| `min_atr_pct` | 0.005 | Hold-only threshold for ATR/price |
| `min_adx_for_short` | 25.0 | Long-only threshold for ADX |
| `min_atr_for_short` | 0.01 | Long-only threshold for ATR/price |
| `min_volume_for_short` | 500,000 | Long-only threshold for avg volume |
| `trend_ma_period` | 200 | MA period for trend direction check |
| `max_pct_above_ma` | 65.0 | If price above MA for > this %, force long-only |
| `atr_period` | 14 | ATR calculation period |

---

## 11. Trading Objective Presets

Objective presets are **partial configuration overrides** that adapt the tool to different trading horizons. When you apply a preset, only the settings specified in the preset are changed — everything else keeps its base value.

### 11.1 Long-Term (Position Trading)

```bash
python main.py AAPL --objective long_term
python main.py AAPL --objective long_term --backtest --period 5y
```

**Designed for:** Holding periods of weeks to months.

**Key changes from base:**

| Category | Base → Long-Term |
|----------|-----------------|
| Indicator periods | Standard → ~1.5× longer (RSI 14→21, MACD 12/26→19/39, Bollinger 20→30) |
| Moving Average periods | [20, 50, 200] → [50, 100, 200] |
| Weight emphasis | Balanced → Heavier on trend indicators (MA 25%, ADX 15%, Volume 15%) |
| Stop-loss / Take-profit | 5% / 20% → 8% / 30% |
| Rebalance interval | 5 bars → 10 bars |
| Warmup bars | 200 → 250 |

### 11.2 Short-Term (Swing Trading)

```bash
python main.py AAPL --objective short_term
python main.py AAPL --objective short_term --backtest --period 6mo
```

**Designed for:** Holding periods of days to weeks.

**Key changes from base:**

| Category | Base → Short-Term |
|----------|------------------|
| Indicator periods | Standard → ~0.7× shorter (RSI 14→9, MACD 12/26→8/17, Bollinger 20→14) |
| Moving Average periods | [20, 50, 200] → [10, 20, 50] |
| Weight emphasis | Balanced → Heavier on momentum (MACD 20%, Stochastic 15%, Bollinger 15%) |
| Stop-loss / Take-profit | 5% / 20% → 3% / 10% |
| Rebalance interval | 5 bars → 3 bars |
| Warmup bars | 200 → 60 |

### 11.3 Day Trading (Intraday)

```bash
python main.py AAPL -o day_trading -b -i 5m --start 2026-02-10
python main.py AAPL -o day_trading -b -i 15m --period 1mo
```

**Designed for:** Intraday positions only — all positions closed by end of day.

**Requirements:** Must use an intraday interval (`-i 5m`, `-i 15m`, etc.). The tool will reject daily intervals with the `day_trading` objective.

**Key changes from base:**

| Category | Base → Day Trading |
|----------|-------------------|
| Indicator periods | Standard → Very fast (RSI 14→5, MACD 12/26→5/13, Bollinger 20→10) |
| Moving Average periods | [20, 50, 200] → [5, 10, 20] |
| Weight emphasis | Balanced → Momentum-heavy (MACD 25%, Stochastic 20%) |
| Stop-loss / Take-profit | 5% / 20% → **1.5% / 3%** |
| Rebalance interval | 5 bars → **1 bar** (every single bar) |
| EOD Flattening | Off → **On** (force-close all positions at end of each day) |
| Warmup bars | 200 → 30 |

### 11.4 Custom Presets

You can define your own presets by adding entries under the `objectives` section in `config.yaml`. A preset is simply a partial config — only the keys you specify are overridden.

**Example:** Adding a "scalping" preset:

```yaml
objectives:
  scalping:
    description: "Ultra-fast scalping — sub-minute trades."

    rsi:
      period: 3

    macd:
      fast_period: 3
      slow_period: 8
      signal_period: 4

    moving_averages:
      periods: [3, 5, 10]

    strategy:
      rebalance_interval: 1
      stop_loss_pct: 0.005    # 0.5% stop
      take_profit_pct: 0.01   # 1% target
      flatten_eod: true

    backtest:
      warmup_bars: 15
```

Then use it:

```bash
python main.py AAPL -o scalping -b -i 1m --start 2026-02-18
```

---

## 12. Configuration

### 12.1 Config File Loading

The tool looks for configuration in this order:

1. **`--config PATH`** — if you specify a custom path, that file is loaded.
2. **`config.yaml` in the project directory** — next to `config.py`.
3. **`config.yaml` in the current working directory** — where you run the command.
4. **Built-in defaults** — if no config file is found, all built-in defaults are used.

Loaded configuration is **deep-merged** with built-in defaults. This means you only need to include the settings you want to change in your `config.yaml` — any missing keys automatically fall back to defaults.

### 12.2 Full Configuration Reference

Below is the complete configuration structure with all sections and their defaults:

```yaml
# ── Indicator Configuration ──────────────────────────────────────

rsi:
  period: 14
  thresholds:
    oversold: 30
    overbought: 70
  scores:
    oversold_score: 9.0
    overbought_score: 1.0
    neutral_score: 5.0

macd:
  fast_period: 12
  slow_period: 26
  signal_period: 9
  scoring:
    strong_bullish_pct: 0.005
    moderate_bullish_pct: 0.001
    strong_bearish_pct: -0.005
    moderate_bearish_pct: -0.001
    crossover_lookback: 5
    bullish_cross_bonus: 1.5
    bearish_cross_penalty: 1.5

bollinger_bands:
  period: 20
  std_dev: 2.0
  scoring:
    lower_zone: 0.20
    upper_zone: 0.80
    squeeze_threshold: 0.02

moving_averages:
  periods: [20, 50, 200]
  type: "sma"                # "sma" or "ema"
  scoring:
    price_above_ma_points: 1.5
    ma_aligned_bullish_points: 1.0
    golden_cross_bonus: 2.0
    death_cross_penalty: 2.0
    cross_lookback: 10
    max_raw_score: 9.5

stochastic:
  k_period: 14
  d_period: 3
  smooth_k: 3
  thresholds:
    oversold: 20
    overbought: 80
  scores:
    oversold_score: 9.0
    overbought_score: 1.0
    neutral_score: 5.0
    bullish_cross_bonus: 1.0
    bearish_cross_penalty: 1.0

adx:
  period: 14
  thresholds:
    weak: 20
    moderate: 40
  scoring:
    weak_multiplier: 0.6
    moderate_multiplier: 0.85
    strong_multiplier: 1.0
    max_directional_spread: 25

volume:
  obv_trend_period: 20
  price_trend_period: 20
  scoring:
    confirmation_bullish_max: 9.5
    confirmation_bullish_min: 6.5
    confirmation_bearish_max: 3.5
    confirmation_bearish_min: 0.5
    divergence_score: 5.0
    obv_strong_change_pct: 10.0
    obv_weak_change_pct: 1.0

fibonacci:
  swing_lookback: 60
  levels: [0.236, 0.382, 0.5, 0.618, 0.786]
  scoring:
    proximity_pct: 0.015
    level_scores:
      0.236: 8.0
      0.382: 7.0
      0.5: 5.0
      0.618: 3.0
      0.786: 1.5
    no_level_score: 5.0
    range_low_score: 2.0
    range_high_score: 8.0

# ── Support/Resistance ───────────────────────────────────────────

support_resistance:
  method: "both"             # "pivot", "fractal", or "both"
  pivot_levels: ["S1", "S2", "S3", "P", "R1", "R2", "R3"]
  fractal_lookback: 60
  fractal_order: 5
  num_levels: 4
  cluster_pct: 0.015
  min_touches: 1

# ── Composite Scoring ────────────────────────────────────────────

overall:
  weights:
    rsi: 0.15
    macd: 0.15
    bollinger_bands: 0.10
    moving_averages: 0.20
    stochastic: 0.10
    adx: 0.10
    volume: 0.10
    fibonacci: 0.10

# ── Pattern Signal Detectors ─────────────────────────────────────

gaps:
  lookback: 20
  min_gap_pct: 0.005
  volume_surge_mult: 1.5
  trend_period: 20
  type_weights:
    common: 0.3
    runaway: 0.7
    breakaway: 1.0
    exhaustion: 0.5
  max_signal_strength: 3.0

volume_range:
  period: 20
  expansion_threshold: 1.5
  contraction_threshold: 0.6
  lookback: 10
  scoring:
    expansion_bull: 8.0
    expansion_bear: 2.0
    contraction: 5.0
    divergence: 5.0

candlesticks:
  doji_threshold: 0.05
  shadow_ratio: 2.0
  lookback: 10
  trend_period: 10
  max_signal_strength: 3.0

spikes:
  period: 20
  spike_std: 2.5
  confirm_bars: 3
  confirm_pct: 0.5
  lookback: 20
  trap_weight: 0.7
  max_signal_strength: 3.0

overall_patterns:
  weights:
    gaps: 0.25
    volume_range: 0.30
    candlesticks: 0.25
    spikes: 0.20

# ── Display ──────────────────────────────────────────────────────

display:
  show_disclaimer: true
  score_decimal_places: 1
  price_decimal_places: 2
  color_thresholds:
    bearish_max: 3.5
    neutral_max: 6.5

# ── Strategy ─────────────────────────────────────────────────────

strategy:
  threshold_mode: "fixed"    # "fixed" or "percentile"
  score_thresholds:
    short_below: 4.5
    hold_below: 5.5
  percentile_thresholds:
    short_percentile: 25
    long_percentile: 75
    lookback_bars: 60
  position_sizing: "percent_equity"
  fixed_quantity: 100
  percent_equity: 0.80
  stop_loss_pct: 0.05
  take_profit_pct: 0.20
  rebalance_interval: 5
  flatten_eod: false
  # Pattern-Indicator Combination
  combination_mode: "weighted"   # "weighted" or "gate"
  indicator_weight: 0.7          # indicator weight in blended score
  pattern_weight: 0.3            # pattern weight in blended score
  # Gate mode thresholds (only used when combination_mode = "gate")
  gate_indicator_min: 5.5
  gate_indicator_max: 4.5
  gate_pattern_min: 5.5
  gate_pattern_max: 4.5

# ── Backtesting ──────────────────────────────────────────────────

backtest:
  initial_cash: 100000.0
  commission_per_trade: 0.0
  slippage_pct: 0.001
  warmup_bars: 200

# ── Suitability Detection ───────────────────────────────────────

suitability:
  mode_override: "auto"
  min_volume: 100000
  min_atr_pct: 0.005
  min_adx_for_short: 25.0
  min_atr_for_short: 0.01
  min_volume_for_short: 500000
  trend_ma_period: 200
  max_pct_above_ma: 65.0
  atr_period: 14

# ── Objective Presets ────────────────────────────────────────────

objectives:
  long_term:
    description: "Position trading — weeks to months."
    # ... (partial overrides for indicators, patterns, strategy, backtest)
  short_term:
    description: "Swing trading — days to weeks."
    # ... (partial overrides for indicators, patterns, strategy, backtest)
  day_trading:
    description: "Day trading — intraday positions only."
    # ... (partial overrides for indicators, patterns, overall_patterns, strategy, backtest)
```

### 12.3 Generating and Validating Config

**Generate a fresh config file:**

```bash
python main.py --generate-config
```

This writes a complete `config.yaml` with all defaults and comments to the current directory.

**Validate your config:**

```bash
python main.py --validate-config
```

This checks for common errors including:
- Invalid threshold ordering (e.g., oversold > overbought)
- Negative weights
- Invalid position sizing mode
- Invalid threshold mode
- Out-of-range percentile values
- Non-positive backtest parameters
- Invalid suitability mode override

---

## 13. Data Provider and Limitations

The tool uses **Yahoo Finance** (via the `yfinance` library) as its data source.

### Supported Periods

| Period | Description |
|--------|-------------|
| `1mo` | 1 month |
| `3mo` | 3 months |
| `6mo` | 6 months (default) |
| `1y` | 1 year |
| `2y` | 2 years |
| `5y` | 5 years |
| `ytd` | Year to date |
| `max` | All available history |

> **Note:** yfinance does not support `10y`. For periods longer than 5 years, use `--start` with a specific date.

### Custom Date Ranges

Use `--start` and optionally `--end` for precise date ranges:

```bash
python main.py AAPL --start 2015-01-01                  # 2015 to today
python main.py AAPL --start 2015-01-01 --end 2020-12-31 # 2015 to 2020
```

When `--start` is specified, `--period` is ignored.

### Intraday Data Limits

Yahoo Finance imposes strict limits on how far back intraday data is available:

| Interval | Maximum History |
|----------|----------------|
| 1m | Last 7 days |
| 2m | Last 60 days |
| 5m | Last 60 days |
| 15m | Last 60 days |
| 30m | Last 60 days |
| 60m / 1h | Last 730 days |
| 90m | Last 730 days |
| Daily+ (1d, 1wk, etc.) | Unlimited |

The tool validates your period/date selection against these limits and will show an error with valid options if you exceed them.

### Data Normalization

The tool automatically:
- Uses adjusted prices (splits and dividends accounted for)
- Normalizes column names to lowercase
- Removes timezone information from timestamps
- Validates that required columns (open, high, low, close, volume) are present

---

## 14. Display and Output

### Analysis Output

The standard analysis display includes:

1. **Header Panel** — Company name, ticker, current price, period, active objective (if set), sector, exchange, currency, and market cap.

2. **Technical Indicators Table** — One row per indicator showing:
   - Indicator name
   - Primary value (e.g., RSI: 45.2)
   - Descriptive detail (e.g., "14-period | Neutral")
   - Score with visual bar (`████░░░░░░ 6.2`)

3. **Overall Score** — Weighted composite of all indicator scores.

4. **Pattern Signals Table** — One row per pattern detector showing:
   - Pattern name
   - Primary value (e.g., "3 gaps detected")
   - Descriptive detail (e.g., "1 breakaway ↑, 2 common")
   - Score with visual bar
   - PATTERN SCORE composite at the bottom

5. **Support and Resistance Table** — Up to 4 support levels and 4 resistance levels with prices, labels, and detection sources.

6. **Score Legend** — Color-coded interpretation guide.

7. **Disclaimer** — "For informational purposes only. Not financial advice."

### Backtest Output

The backtest display includes:

1. **Header Panel** — Ticker, period, strategy name, active objective.

2. **Suitability Assessment** (if auto-detected) — Trading mode, metrics, and reasons.

3. **Performance Summary Table** — All metrics from [Section 9.5](#95-performance-metrics), color-coded green (positive) or red (negative).

4. **Strategy Configuration Table** — All active strategy parameters, including combination mode (weighted or gate) and indicator/pattern weight split.

5. **Trade Log** — Detailed table of every trade:
   - Trade number, side (LONG/SHORT), entry/exit dates and prices
   - Quantity, P&L dollar amount and percentage, exit reason
   - Up to 50 trades shown; if more, first 25 + last 25 with a count of omitted trades

6. **Equity Curve** — Text-based checkpoints showing portfolio value at start, 25%, 50%, 75%, and end of the backtest period.

7. **Backtest Disclaimer** — "Backtest results are hypothetical and do not guarantee future performance."

---

## 15. Extending the Tool: Adding New Indicators

The plugin architecture makes it straightforward to add new indicators without modifying any existing code.

### Step 1: Create the Indicator File

Create a new Python file in the `indicators/` directory, e.g., `indicators/vwap.py`:

```python
from __future__ import annotations
from typing import Any

import pandas as pd
from .base import BaseIndicator


class VWAPIndicator(BaseIndicator):
    # Display name shown in the terminal output
    name = "VWAP"

    # Key used in config.yaml and weights — must be unique
    config_key = "vwap"

    def compute(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute raw indicator values from OHLCV DataFrame."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cumulative_tp_vol = (typical_price * df["volume"]).cumsum()
        cumulative_vol = df["volume"].cumsum()
        vwap = cumulative_tp_vol / cumulative_vol
        current_vwap = float(vwap.iloc[-1])
        current_price = float(df["close"].iloc[-1])
        return {
            "vwap": current_vwap,
            "price": current_price,
            "deviation_pct": (current_price - current_vwap) / current_vwap * 100,
        }

    def score(self, values: dict[str, Any]) -> float:
        """Translate computed values into a 0-10 score."""
        dev = values["deviation_pct"]
        threshold = self.config.get("deviation_threshold", 2.0)
        # Price above VWAP = bullish; below = bearish
        raw = self._linear_score(dev, -threshold, threshold, 1.0, 9.0)
        return self._clamp(raw)

    def summary(self, values: dict[str, Any], score: float) -> dict[str, Any]:
        """Build display strings for the terminal table."""
        return {
            "value_str": f"${values['vwap']:.2f}",
            "detail_str": f"Price deviation: {values['deviation_pct']:+.2f}%",
        }
```

### Step 2: Add Configuration

In `config.yaml`, add a section for your indicator:

```yaml
vwap:
  deviation_threshold: 2.0
```

Also add it to the `DEFAULT_CONFIG` dictionary in `config.py`:

```python
"vwap": {
    "deviation_threshold": 2.0,
},
```

### Step 3: Add a Weight

In the `overall.weights` section of `config.yaml` (and `DEFAULT_CONFIG`):

```yaml
overall:
  weights:
    rsi: 0.15
    macd: 0.15
    bollinger_bands: 0.10
    moving_averages: 0.15
    stochastic: 0.10
    adx: 0.10
    volume: 0.10
    fibonacci: 0.05
    vwap: 0.10       # new indicator weight
```

### That's It

No other code changes needed. The indicator registry automatically discovers any class in the `indicators/` directory that:
- Subclasses `BaseIndicator`
- Has a non-empty `config_key` class attribute

### Available Helper Methods

All indicators inherit these from `BaseIndicator`:

| Method | Description |
|--------|-------------|
| `self._clamp(value, lo=0.0, hi=10.0)` | Clamp a value to a range |
| `self._linear_score(value, low_val, high_val, low_score, high_score)` | Linearly interpolate a score between two value endpoints |
| `self.config` | The indicator's config section from `config.yaml` |
| `self.run(df)` | Convenience method that calls compute → score → summary with error handling |

---

## 16. Troubleshooting

### "No data found for ticker"

- Verify the ticker symbol is correct and traded on a Yahoo Finance-supported exchange.
- Check your internet connection.

### "Period X is too long for interval Y"

- Intraday data has strict history limits (see [Section 13](#13-data-provider-and-limitations)).
- Use `--start` with a recent date instead of `--period`, or use a wider interval.

### SMA-200 shows "N/A"

- With short data periods (e.g., `--period 3mo`), there aren't enough bars to compute a 200-day moving average.
- Use a longer period or the `short_term` / `day_trading` objective (which uses shorter MA periods).

### Composite scores are always near 5.0

- This is expected when averaging 8 indicators. The composite naturally has lower variance than individual indicators.
- The strategy thresholds (4.5/5.5 for fixed mode) are calibrated for this narrow range.
- Consider using `threshold_mode: "percentile"` for self-calibrating thresholds that adapt to each stock.

### "day_trading objective requires an intraday interval"

- The `day_trading` preset is designed for intraday data. Add an interval flag:
  ```bash
  python main.py AAPL -o day_trading -b -i 5m --start 2026-02-10
  ```

### Backtest shows "No trades were executed"

- The warmup period may consume all your data. Ensure your data has at least `warmup_bars + 10` bars after warmup.
- For short-term / day trading presets, warmup is smaller (60 / 30 bars), so this is less likely.
- Check that your score thresholds actually produce BUY/SELL signals — if all scores fall in the HOLD zone, no trades are generated.

### Config validation warnings

- Run `python main.py --validate-config` to see all issues.
- Common issues: oversold ≥ overbought, short_below ≥ hold_below, negative weights.

---

*This tool is for informational and educational purposes only. It does not constitute financial advice. Backtest results are hypothetical and do not guarantee future performance. Always do your own research before making investment decisions.*
