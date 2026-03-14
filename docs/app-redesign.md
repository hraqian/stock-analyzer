# Stock Analyzer — App Redesign Specification

**Date:** 2026-03-14
**Status:** Draft

---

## Overview

Major redesign of the Stock Analyzer from a single-page dashboard into a
multi-section application covering the full trading workflow: discovery,
analysis, backtesting, portfolio simulation, and strategy management.

**Platform:** Web-first (replace Streamlit with a proper web framework).
Desktop may follow later.

**Target users:** Power users (2 people), with a simplified "normal mode" that
hides advanced parameters and relies on AI auto-tuning.

**Data sources:** Pluggable architecture starting with Yahoo Finance (free),
with support for adding paid providers (Polygon, Alpha Vantage, IEX Cloud)
for real-time data, crypto, bonds, and Canadian markets.

**Asset coverage:** US and Canadian stocks, ETFs, crypto, bonds — as broad as
the configured data providers allow.

---

## Global Controls

These live in the app header, always visible:

| Control | Description |
|---------|-------------|
| **Trade Mode** | `Swing Trade` / `Day Trade` / `Long-Term Investment` — global toggle that adjusts indicator parameters, holding periods, position sizing defaults, scanner thresholds, and backtest horizons across the entire app. Day Trade is visible but marked "Coming Soon — requires paid data provider" until intraday data support is added. |
| **User Mode** | `Normal` / `Power User` — Normal mode hides advanced parameters and favors AI auto-tuning. Power User mode exposes all configurable parameters. |
| **Account Summary** | Cash balance, total equity, open position count (visible when portfolio simulation is active). |

---

## Section 1: Market Scanner

**Purpose:** "What should I look at today?"

Scans a configurable universe of tickers and surfaces the best trade
candidates, ranked by signal quality.

### Features

| Feature | Description |
|---------|-------------|
| **Universe selector** | Predefined universes (S&P 500, Nasdaq 100, TSX Composite, Russell 1000/2000, Sector ETFs, Crypto, Bonds) plus user-defined custom watchlists. Multiple universes can be combined. |
| **Scan presets** | One-click scans: Top 10 Breakout Setups, Top 10 Pullback Setups, Top 10 Reversal Setups. |
| **Liquidity filter** | Minimum average volume (default: 1M), minimum ATR, minimum price (default: $5). Configurable. |
| **Volatility filter** | ATR/price ratio threshold — when exceeded, signal confidence is reduced (not eliminated). |
| **Multi-timeframe confirmation** | Daily trend must agree with 4H signal. 1H used for entry timing. Signals that lack multi-timeframe alignment are flagged/downgraded. |
| **Pattern reliability weighting** | Each detected pattern carries a reliability weight based on historical performance (e.g. bull flag = high, double bottom = medium, head & shoulders = medium, ascending triangle = high). Weights are configurable. |
| **AI signal scoring** | ML model rates each signal based on historical outcome data (e.g. "this setup has 72% historical win rate"). LLM adds a qualitative one-line summary. |
| **Signal decay indicator** | Shows how signal strength is expected to degrade over time (1, 3, 5 bars out), helping users prioritize urgency. |

### Output

A ranked table of trade candidates:

```
Rank  Ticker  Signal     Score  AI Rating  Confidence  Pattern          Sector
1     NVDA    Breakout   8.7    9/10       High        Bull flag        Semiconductors
2     AMZN    Pullback   7.9    8/10       High        Ascending tri.   Web/Cloud
3     XLE     Reversal   6.4    7/10       Medium      Double bottom    Energy
```

Clicking a ticker opens it in **Single Stock Analysis**.

---

## Section 2: Single Stock Analysis

**Purpose:** "Tell me everything about this one ticker."

Deep-dive view for a single stock, combining chart analysis, indicators,
pattern detection, regime classification, and AI commentary.

### Features

| Feature | Description |
|---------|-------------|
| **Interactive chart** | Candlestick chart with configurable indicators (RSI, MACD, Bollinger Bands, moving averages, volume, etc.) and pattern overlays drawn directly on the chart. |
| **Auto support & resistance** | Automatically detected via price clustering. Displayed as horizontal zones on the chart. |
| **Market regime detection** | Classifies current regime (trending / sideways / high-volatility) and adjusts indicator weights dynamically. Displayed as a badge/banner. |
| **Multi-timeframe view** | Side-by-side or stacked charts for daily, 4H, 1H timeframes. Shows whether timeframes agree or conflict. |
| **Indicator panel** | All active indicators with current values, signal direction, and contribution to overall score. |
| **Pattern panel** | Detected patterns with reliability rating, historical win rate for this pattern on this ticker (if sufficient data). |
| **AI analysis panel** | LLM-generated qualitative analysis: "NVDA is in a strong uptrend with a bull flag forming near resistance at $950. Semiconductor sector momentum is strong. Risk: elevated volatility after earnings." |
| **Signal summary** | Current signal (BUY/SELL/HOLD), composite score, confidence level, signal age/decay. |
| **Trade mode context** | Analysis adjusts based on global trade mode — swing shows multi-day setups, day trade shows intraday levels, long-term shows weekly trends. |

---

## Section 3: Sector & Segments

**Purpose:** "Where is the money flowing?"

Provides a top-down view of market structure by sector and industry segment,
helping users identify which areas of the market are strong or weak.

### Sector Classification

Uses GICS (Global Industry Classification Standard) as the base taxonomy,
sourced from Yahoo Finance or equivalent data provider:

```
Sector (Level 1)          Industry Group (Level 2)     Industry (Level 3)
Technology                 Semiconductors               NVDA, AMD, AVGO, TSM
                           Software                     MSFT, ORCL, CRM
                           IT Services                  ACN, IBM
Consumer Discretionary     Internet Retail               AMZN, BABA
                           Specialty Retail              HD, TJX
Energy                     Oil & Gas E&P                 XOM, CVX
                           Energy Equipment              SLB, HAL
...
```

Users can define **custom segments** on top of GICS (e.g. "AI Plays" = NVDA,
AMD, MSFT, GOOG, META).

### Features

| Feature | Description |
|---------|-------------|
| **Sector heatmap** | All sectors displayed with color-coded momentum scores (green = strong, red = weak). Click to drill down. |
| **Segment drill-down** | Expand a sector to see sub-industries with momentum, top movers, and worst performers in each. |
| **Sector rotation tracker** | Tracks capital flows between sectors over 1W, 1M, 3M. Shows which sectors are gaining/losing relative strength. |
| **Relative strength vs benchmark** | Each sector compared to SPY (or user-chosen benchmark). Shows outperformance/underperformance trend. |
| **Top tickers per segment** | "Semiconductors: strong momentum — NVDA (+12%), AMD (+8%), AVGO (+6%)" |
| **Regime per sector** | Each sector has its own regime classification. Sectors can be in different regimes simultaneously (Tech trending while Energy is sideways). |
| **Cross-sector signals** | Identifies divergences: "Semis breaking out while Retail breaking down" — useful for rotation strategies. |
| **Custom segments** | User-defined groupings layered on top of GICS classification. |

### Integration with other sections

- **Scanner:** Filter scans to only strong/weak sectors.
- **Single Stock:** Shows sector context for any ticker being analyzed.
- **Strategy Lab:** Backtest strategy constrained to specific sectors.
- **Portfolio Sim:** Enforce sector diversification limits (e.g. max 30% in
  any one sector).

---

## Section 4: Strategy Lab

**Purpose:** "Build, test, and tune trading strategies."

The research workbench where strategies are developed, validated, and saved.
Contains four sub-sections.

### 4a. Backtester

Single-stock or universe-level backtesting with realistic market friction.

| Feature | Description |
|---------|-------------|
| **Single-stock backtest** | Run a strategy on one ticker over a date range. Full metrics: total trades, win rate, profit factor, max drawdown, annual return, Sharpe ratio, Sortino ratio. |
| **Universe-level backtest** | Run a strategy across an entire universe (e.g. S&P 500). Aggregate results show how the strategy performs across many stocks, not just cherry-picked ones. |
| **Realistic friction model** | Commission (per trade), spread (bid-ask), slippage (market impact), tax (short-term capital gains, long-term capital gains, income — configurable rates). Applied to every simulated trade. |
| **Look-ahead bias protection** | Engine strictly enforces point-in-time data. No future data leaks into signal generation or position sizing. |
| **Equity curve & trade log** | Visual equity curve, drawdown chart, and detailed trade-by-trade log with entry/exit reasons. |

### 4b. Walk-Forward Testing

Rolling out-of-sample validation — the only honest way to know if a strategy
generalizes.

**Method:**

```
Window 1:  Train 2015–2020  →  Test 2021
Window 2:  Train 2016–2021  →  Test 2022
Window 3:  Train 2017–2022  →  Test 2023
Window 4:  Train 2018–2023  →  Test 2024
Window 5:  Train 2019–2024  →  Test 2025
```

**Default windows by trade mode:**

| Trade Mode | Train Period | Test Period | Rationale |
|-----------|-------------|-------------|-----------|
| Long-Term | 5 years | 1 year | Long-term patterns need deep history. |
| Swing Trade | 5 years | 1 year | Same default; swing setups are multi-day so yearly windows work. |
| Day Trade | Auto-shortened (e.g. 1–2 years train, 3–6 months test) | — | Intraday patterns shift faster; shorter windows capture regime changes. |

All windows are user-configurable.

**Output per window:**

- Win rate, profit factor, max drawdown, annual return
- Parameter stability: did the optimal parameters change drastically between
  windows? If yes, strategy may be overfit.

**Aggregate output:**

- Average performance across all test windows
- Worst-case window (stress test)
- Parameter stability score (low variance = good)
- Verdict: "Strategy is stable across walk-forward windows" or "Warning:
  significant degradation in recent windows"

### 4c. Auto-Tuner

AI-driven parameter optimization based on user objectives.

**Objectives (user picks one):**

| Objective | Optimizes for |
|-----------|--------------|
| Beat buy-and-hold | Excess return over simple buy-and-hold of the same ticker/universe |
| Maximize return | Highest absolute annual return (accepts more risk) |
| Maximize risk-adjusted return | Best Sharpe or Sortino ratio |
| Minimize drawdown | Lowest max drawdown while maintaining positive returns |
| Balanced | Weighted combination of return, drawdown, and win rate |

**Modes:**

| Mode | Behavior |
|------|----------|
| **Normal mode** | User picks an objective. Auto-tuner runs walk-forward optimization and returns the best parameter set. User sees: "Optimized for swing trade, beat buy-and-hold. Result: +24% annual vs +18% buy-and-hold." |
| **Power User mode** | Same as normal, but also shows all parameters, sensitivity analysis, and lets user manually override any parameter. |

**Constraints:**

- Auto-tuner always validates via walk-forward (never just in-sample).
- Reports both in-sample and out-of-sample performance.
- Warns if out-of-sample performance degrades significantly vs in-sample
  (sign of overfitting).

### 4d. Strategy Library

Save, manage, and compare strategy configurations.

| Feature | Description |
|---------|-------------|
| **Save strategy** | Snapshot of all parameters, indicators, patterns, trade mode, and backtest results. Named and versioned. |
| **Compare strategies** | Side-by-side comparison table: Strategy A vs B vs C with all key metrics. |
| **Built-in presets** | Momentum Swing, Mean Reversion Day Trade, Long-Term Trend Following, Sector Rotation, etc. |
| **Import/Export** | Strategies saved as YAML/JSON, shareable between users. |
| **Version history** | Track parameter changes over time. "v3 changed RSI threshold from 30→25, improved win rate by 3%." |
| **Activate** | Sends a strategy to Portfolio Simulation as a live signal source (see below). |

---

## Section 5: Portfolio Simulation

**Purpose:** "Run the full trading system end-to-end."

This is where everything comes together. Strategies from the Strategy Lab
generate signals, the portfolio manager ranks and selects the best ones,
capital is allocated, and trades are tracked with full P&L accounting.

### Strategy → Portfolio Pipeline

```
Strategy Lab                          Portfolio Simulation
+-------------------+                +------------------------+
| Strategy Library  |   "Activate"   | Active Strategies      |
|  - Swing Mom. v3  | ------------> |  - Swing Momentum v3   |
|  - Mean Rev. v2   | ------------> |  - Mean Reversion v2   |
|  - Sector Rot.    |               |                        |
+-------------------+                | Signal Engine scans    |
                                     | universe per strategy, |
                                     | generates BUY/SELL     |
                                     | recommendations when   |
                                     | conditions are met.    |
                                     +------------------------+
```

### Sub-sections

#### 5a. Active Strategies

Manage which strategies are running and on which universe.

```
Strategy             Universe        Status     Last Scan    Signals
Swing Momentum v3    S&P 500         Active     2 hours ago  3 pending
Mean Reversion v2    Nasdaq 100      Active     2 hours ago  1 pending
Sector Rotation      Sector ETFs     Paused     Yesterday    0 pending
```

Each strategy can be activated, paused, or removed independently.

#### 5b. Recommendations

Pending buy/sell signals awaiting user action. Each recommendation includes
full provenance: which strategy generated it, why, and with what confidence.

```
BUY NVDA — 45 shares @ $930.00 ($41,850)
  Strategy:     Swing Momentum v3
  Signal:       Bull flag breakout + RSI divergence
  Timeframes:   Daily: bullish | 4H: pullback entry | 1H: reversal confirmed
  AI Rating:    8.2/10 (72% historical win rate for this setup)
  Regime:       Strong trend (Semiconductors in uptrend)
  Sizing:       4.2% of equity (risk-based: account * risk / stop distance)
  Stop Loss:    $895 (-3.8%)
  Target:       $1,020 (+9.7%)
  Risk/Reward:  2.6:1
  Est. Cost:    $41,892 (incl. $12 commission + $30 slippage est.)

  [Confirm]  [Edit Quantity]  [Skip]
```

User actions:
- **Confirm** — execute the trade as recommended.
- **Edit Quantity** — adjust the share count before confirming.
- **Skip** — dismiss the recommendation.

#### 5c. Open Positions

Current portfolio state.

```
Ticker  Strategy        Side  Entry     Qty   Cost Basis   Current   P&L     P&L%   Days
NVDA    Swing Mom. v3   Long  03-10     45    $41,850      $42,975   +$1,125  +2.7%  4
MSFT    Mean Rev. v2    Long  03-07     30    $12,450      $12,180   -$270    -2.2%  7
AAPL    Swing Mom. v3   Long  03-03     55    $12,375      $13,090   +$715    +5.8%  11
```

Shows: entry strategy, stop loss levels, target levels, days held,
unrealized P&L. Alerts when stop loss or max hold period is approaching.

#### 5d. Performance Dashboard

| Widget | Description |
|--------|-------------|
| **Equity curve** | Total portfolio value over time. |
| **Drawdown chart** | Current and max drawdown visualization. |
| **P&L summary** | Total realized + unrealized P&L, broken down by strategy. |
| **Sector exposure** | Pie/bar chart showing portfolio concentration by sector. Warns if exceeding limits. |
| **Key metrics** | Win rate, profit factor, Sharpe ratio, max drawdown — for the live portfolio. |
| **Strategy attribution** | Which strategy is contributing most/least to returns. |

#### 5e. Trade History

Closed trades with full detail.

```
Ticker  Strategy        Side  Entry     Exit      Qty   P&L       P&L%   Friction  Net P&L
GOOG    Swing Mom. v3   Long  02-15     03-01     20    +$1,240   +6.2%  -$45      +$1,195
META    Mean Rev. v2    Long  02-20     03-05     25    -$380     -2.1%  -$38      -$418
```

Includes: commission, spread, slippage, and estimated tax impact per trade.

#### 5f. Historical Replay

Run the full pipeline (scan → rank → allocate → execute) over historical data
to validate the integrated system before going live.

- Fully automated (no user confirmation — simulates all trades).
- Shows what the portfolio would have looked like over a historical period.
- Uses walk-forward windows so each trade decision uses only data available
  at that point in time.
- Compares against benchmarks (SPY, buy-and-hold of universe, etc.).

### Portfolio-Level Risk Rules

| Rule | Description |
|------|-------------|
| **Max positions** | Maximum number of simultaneous open positions. |
| **Max per sector** | Maximum % of portfolio in any single sector (default: 30%). |
| **Max per strategy** | Maximum % of portfolio allocated to one strategy. |
| **Max portfolio drawdown** | If portfolio drawdown exceeds threshold, pause new entries. |
| **Position sizing** | `account_value * risk_per_trade / stop_loss_distance` — risk-based sizing with configurable risk_per_trade %. |
| **Cash reserve** | Minimum cash % to keep uninvested (default: 0%). |

---

## Section 6: Settings

**Purpose:** "Set it and forget it."

Global configuration that applies across all sections.

| Category | Settings |
|----------|----------|
| **Account Profile** | Starting capital, risk tolerance (conservative / moderate / aggressive), preferred trade mode. |
| **Cost Model** | Default commission per trade, estimated spread %, slippage %, tax rates (short-term CG, long-term CG, income tax). |
| **Data Sources** | Yahoo Finance (default, free). Add API keys for: Polygon.io, Alpha Vantage, IEX Cloud, or others. Each provider enables additional asset classes and real-time data. |
| **Universe Management** | Define custom watchlists/universes. Import ticker lists from CSV. |
| **Default Filters** | Liquidity (min volume, min ATR), volatility (max ATR/price ratio), min price. Applied globally unless overridden per scan. |
| **AI Configuration** | LLM provider + API key (for qualitative analysis). ML model training preferences (retrain frequency, minimum data requirements). |
| **Display Preferences** | Theme (dark/light), default chart type, default timeframe, number format. |

---

## Technical Considerations

### Tech Stack

```
Frontend:  Next.js (React + TypeScript)
           - TailwindCSS for styling
           - Recharts or Lightweight Charts (TradingView) for financial charts
           - TanStack Query for API state management

Backend:   FastAPI (Python)
           - Serves the existing engine code as REST API
           - WebSocket support for scan progress / live updates
           - SQLite or PostgreSQL for strategy/portfolio persistence

Engine:    Python (existing code, refactored)
           - Indicators, patterns, regime detection
           - Backtester, walk-forward, auto-tuner
           - Scanner, signal scoring
           - Portfolio simulation engine

AI:        ML model (scikit-learn or XGBoost) for signal scoring
           LLM (pluggable: OpenAI / Anthropic) for qualitative analysis
```

### Walk-Forward Default Windows

| Trade Mode | Train | Test | Notes |
|-----------|-------|------|-------|
| Long-Term Investment | 5 years | 1 year | Standard. |
| Swing Trade | 5 years | 1 year | Standard. |
| Day Trade | 1–2 years | 3–6 months | Auto-shortened because intraday patterns shift faster. Exact windows TBD based on data availability. |

### Friction Model

Applied to every simulated and live-sim trade:

```
Gross P&L
  - Commission (flat per trade, e.g. $1–$10)
  - Spread (bid-ask, estimated as % of price, varies by liquidity)
  - Slippage (market impact, estimated as % of price)
  = Pre-tax P&L
  - Tax (rate depends on holding period: <1 year = short-term CG rate,
    >=1 year = long-term CG rate; day trades may be income tax rate)
  = Net P&L
```

### Look-Ahead Bias Prevention

- All indicator calculations use only data available at the time of the bar.
- Position sizing uses only the equity known at decision time.
- Walk-forward testing enforces strict train/test separation.
- Scanner signals are generated using only historical + current bar data.

### Data Provider Architecture

```
DataProvider (abstract)
  |-- YahooFinanceProvider (current, free)
  |-- PolygonProvider (future)
  |-- AlphaVantageProvider (future)
  |-- IEXCloudProvider (future)

Each provider implements:
  - get_historical(ticker, start, end, interval) -> DataFrame
  - get_live_price(ticker) -> float
  - get_sector_info(ticker) -> SectorInfo
  - supported_asset_types() -> list[str]
```

### Pattern Reliability Defaults

| Pattern | Default Reliability | Notes |
|---------|-------------------|-------|
| Bull flag | High | Strong continuation pattern in uptrends |
| Ascending triangle | High | Reliable breakout pattern |
| Double bottom | Medium | Requires volume confirmation |
| Head & shoulders | Medium | Classic but often misidentified |
| Descending triangle | Medium-High | Reliable breakdown pattern |
| Cup & handle | Medium | Longer formation, decent reliability |
| Falling wedge | Medium | Bullish reversal, moderate reliability |
| Rising wedge | Medium | Bearish reversal, moderate reliability |

Reliability weights are configurable and can be refined by the ML model
based on actual backtest outcomes.

---

## Implementation Priority

This is a large redesign. Suggested phased approach:

### Phase 1: Foundation
- [ ] New web framework setup (replace Streamlit)
- [ ] Global trade mode toggle
- [ ] Pluggable data provider architecture
- [ ] GICS sector classification integration

### Phase 2: Core Analysis
- [ ] Single Stock Analysis (chart, indicators, patterns, S&R, regime)
- [ ] Market Scanner (universe selector, filters, signal scoring)
- [ ] Sector & Segments (heatmap, drill-down, rotation tracker)

### Phase 3: Strategy Engine
- [ ] Backtester with friction model
- [ ] Walk-forward testing
- [ ] Auto-tuner (normal + power user mode)
- [ ] Strategy library (save, compare, export)

### Phase 4: Portfolio Simulation
- [ ] Activate strategy pipeline
- [ ] Recommendation queue (confirm/edit/skip)
- [ ] Position tracking, P&L, equity curve
- [ ] Historical replay
- [ ] Risk management rules

### Phase 5: AI Layer
- [ ] ML signal scoring model
- [ ] LLM qualitative analysis
- [ ] AI-assisted auto-tuning
- [ ] Signal decay prediction

---

## Resolved Decisions

1. **Web framework** — **Next.js (React/TypeScript) frontend + FastAPI
   (Python) backend.** React gives us the interactive UI needed for complex
   charts, multi-panel layouts, and real-time recommendation cards. FastAPI
   serves the Python engine (indicators, backtester, ML models) as a REST
   API. This keeps the existing Python engine code reusable.

2. **Day trade mode** — **Deferred.** Day trade mode will appear in the UI
   trade mode toggle but marked as "Coming Soon — requires paid data
   provider". Intraday data (1-min, 5-min bars) needs a paid provider
   (Polygon, etc.) for reliable coverage. The UI, data models, and walk-
   forward window logic will be designed to support it, but actual intraday
   scanning and backtesting is not implemented in the initial release.

3. **LLM cost management** — **ML for bulk ranking, LLM for top-N only.**
   The scanner uses the ML model to score all candidates (cheap, fast).
   LLM qualitative analysis runs only on tickers the user clicks into
   (Single Stock Analysis) or the top N recommendations in Portfolio
   Simulation. This keeps API costs proportional to user engagement, not
   universe size.

4. **ML model approach** — **XGBoost classification with probability output.**

   Features (computed at signal time):
   - Indicator values: RSI, MACD histogram, Bollinger %B, MA slopes
   - Pattern type + reliability weight
   - Market regime classification
   - Sector relative strength
   - Volume ratio (current vs 20-day average)
   - ATR/price ratio (volatility context)
   - Multi-timeframe alignment score
   - Days since last signal on same ticker

   Target variable: Forward return after N bars (win/loss classification).
   N depends on trade mode: ~5-10 bars for swing, ~60-120 bars for
   long-term.

   Training data: Every historical signal generated by the backtester
   across the universe becomes a labeled sample. A 5-year backtest across
   500 stocks produces tens of thousands of samples.

   Output: Probability score (0-100) used as the "AI Rating". XGBoost
   feature importances are shown to users to explain *why* a signal
   scored high.

   Retrain frequency: Quarterly, walk-forward style — train on all data
   up to 3 months ago, validate on the most recent 3 months. App warns
   the user if model performance degrades.

5. **LLM provider** — **Anthropic (Claude) as default, pluggable interface.**

   ```
   LLMProvider (abstract)
     |-- AnthropicProvider (default)
     |-- OpenAIProvider
     |-- LocalProvider (Ollama / llama.cpp for zero API cost)
   ```

   The prompt template is provider-agnostic — only the API call differs.
   Claude is the default because it's strong at structured technical
   analysis. Local models (via Ollama) are supported for users who want
   zero API cost, with the trade-off of lower analysis quality.

6. **Real-time vs on-demand** — **Three-tier rollout.**

   | Tier | Description | When |
   |------|-------------|------|
   | On-demand | User clicks "Scan", results appear. | Phase 1 (launch) |
   | Auto-refresh | Configurable schedule (e.g. every 30 min during market hours). Backend runs scans on a timer, pushes new recommendations via WebSocket. | Phase 2-3 |
   | Real-time streaming | Live price feed, signals update tick-by-tick. Requires paid data provider with WebSocket/SSE support. | Future (only needed for day trade mode) |

   For swing trade and long-term investment, on-demand + auto-refresh is
   sufficient. Real-time streaming is only necessary for day trading,
   which is already deferred.

7. **Authentication** — **Simple multi-user with per-user profiles.**

   Since there are only 2 users, keep it minimal:
   - Each user has a profile stored server-side (SQLite) containing:
     portfolio state, activated strategies, settings, watchlists.
   - Login via username + password (bcrypt-hashed, stored locally).
   - FastAPI backend scopes all data by user ID.
   - No OAuth, email verification, or role-based permissions.
   - If more users are needed later, bolt on NextAuth.js with OAuth
     providers at that point.

## Open Questions

None at this time. All major architectural decisions have been resolved.
