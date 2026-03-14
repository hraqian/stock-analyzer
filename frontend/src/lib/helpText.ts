/**
 * Centralized help text for all technical terms used in the app.
 *
 * Keep explanations short (1-2 sentences), jargon-free, and aimed at
 * someone who has never traded before.  This file is the single source
 * of truth — update text here and it updates everywhere.
 */

// ---------------------------------------------------------------------------
// Analysis page — section headings
// ---------------------------------------------------------------------------

export const HELP_COMPOSITE_SCORES =
  "A single number (0-10) that combines all the individual indicator scores into one overall rating. Higher means more bullish (likely to go up), lower means more bearish (likely to go down).";

export const HELP_OVERALL_SCORE =
  "The weighted average of all indicator scores. Above 6 is generally bullish, below 4 is bearish, and around 5 is neutral.";

export const HELP_TREND_SCORE =
  "How strong the current price trend is, based on trend-following indicators like Moving Averages and ADX. High = strong uptrend.";

export const HELP_CONTRARIAN_SCORE =
  "Based on indicators that look for reversals — like RSI and Stochastic. A high score suggests the stock may be oversold (due to bounce up).";

export const HELP_NEUTRAL_SCORE =
  "Score from indicators that don't strongly favor trend or reversal, like Volume and Bollinger Bands.";

export const HELP_PATTERN_SCORE =
  "A combined score from chart pattern detectors (gaps, candlesticks, volume spikes, etc.). Above 5 = bullish patterns detected.";

export const HELP_DOMINANT_GROUP =
  "Whether trend-following or contrarian (reversal) indicators are giving stronger signals right now. Helps you understand what's driving the overall score.";

// ---------------------------------------------------------------------------
// Regime
// ---------------------------------------------------------------------------

export const HELP_REGIME =
  "The current market 'personality' of this stock. Is it trending steadily, bouncing in a range, choppy, or breaking out? Each regime suggests a different trading approach.";

export const HELP_REGIME_CONFIDENCE =
  "How certain the algorithm is about its regime classification. Higher confidence means the stock clearly fits one category.";

export const HELP_REGIME_SUBTYPE =
  "A finer classification based on volatility (how much price swings) and momentum (how fast price moves in one direction).";

export const HELP_ADX =
  "Average Directional Index — measures trend strength from 0 to 100. Below 20 = weak/no trend, 20-40 = moderate trend, above 40 = strong trend. It doesn't tell you the direction, just how strong the trend is.";

export const HELP_ATR_PCT =
  "Average True Range as a percentage of price — measures how much the stock typically moves per day. Higher = more volatile. Useful for setting stop-losses.";

export const HELP_TOTAL_RETURN =
  "The total price change over the selected time period. For example, +50% means the stock went up 50%.";

export const HELP_PCT_ABOVE_MA =
  "What percentage of the time the price was above its moving average. Above 50% suggests an uptrend; below 50% a downtrend.";

export const HELP_TREND_DIRECTION =
  "Whether the stock is currently in an uptrend (bullish), downtrend (bearish), or moving sideways (neutral).";

// ---------------------------------------------------------------------------
// Technical indicators
// ---------------------------------------------------------------------------

export const HELP_INDICATORS =
  "Technical indicators are math formulas applied to price and volume data. Each one measures something different (momentum, trend, volatility) and produces a score from 0 (very bearish) to 10 (very bullish).";

export const HELP_RSI =
  "Relative Strength Index — measures momentum on a 0-100 scale. Below 30 = oversold (may bounce up), above 70 = overbought (may pull back). The score converts this to our 0-10 scale.";

export const HELP_MACD =
  "Moving Average Convergence Divergence — compares a fast and slow moving average. When the fast crosses above the slow, it's a bullish signal. Shows momentum direction and strength.";

export const HELP_BOLLINGER =
  "Bollinger Bands — a channel around the price based on volatility. Price near the lower band may be oversold; near the upper band may be overbought. A tight squeeze often precedes a big move.";

export const HELP_MOVING_AVERAGES =
  "Moving Averages smooth out price over a period (e.g. 20, 50, 200 days). Price above the moving average = uptrend. When shorter MAs cross above longer ones ('Golden Cross'), it's bullish.";

export const HELP_STOCHASTIC =
  "Stochastic Oscillator — compares the closing price to the price range over a period. Like RSI, readings below 20 suggest oversold and above 80 overbought.";

export const HELP_VOLUME =
  "Volume analysis — looks at trading volume patterns. Rising volume on up days confirms buying pressure. Falling volume on a rally can signal weakness.";

export const HELP_FIBONACCI =
  "Fibonacci Retracement — key price levels (23.6%, 38.2%, 50%, 61.8%) where stocks often find support or resistance during pullbacks. Based on the Fibonacci number sequence.";

// ---------------------------------------------------------------------------
// Patterns
// ---------------------------------------------------------------------------

export const HELP_PATTERNS =
  "Chart patterns are recognizable shapes or events in price/volume data that often signal what might happen next. Each pattern is scored 0-10 (bearish to bullish).";

export const HELP_GAPS =
  "Gaps happen when the price opens significantly higher or lower than the previous close. Gap ups on high volume are bullish; gap downs are bearish. Some gaps get 'filled' (price returns to the gap).";

export const HELP_VOLUME_RANGE =
  "Detects unusual volume combined with price range patterns — like volume spikes on breakouts or volume drying up during consolidation.";

export const HELP_CANDLESTICKS =
  "Classic candlestick patterns (like Doji, Hammer, Engulfing) that suggest potential reversals or continuations based on the shape of recent price bars.";

export const HELP_SPIKES =
  "Sudden large price moves (up or down) that may signal capitulation, panic, or breakout events.";

export const HELP_INSIDE_OUTSIDE =
  "Inside bars (range within prior bar) signal consolidation; outside bars (range exceeds prior bar) signal expansion. Breakouts from inside bars can be powerful.";

// ---------------------------------------------------------------------------
// Support & Resistance
// ---------------------------------------------------------------------------

export const HELP_SUPPORT_RESISTANCE =
  "Support levels are prices where the stock tends to stop falling and bounce up. Resistance levels are where it tends to stop rising and pull back. Think of them as a floor and ceiling.";

export const HELP_SUPPORT =
  "Price levels below the current price where buying pressure has historically prevented further decline. The stock may bounce up from these levels.";

export const HELP_RESISTANCE =
  "Price levels above the current price where selling pressure has historically prevented further rise. The stock may pull back from these levels.";

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

export const HELP_PERIOD =
  "How far back to look at price data. '6mo' means 6 months of history. Longer periods show the bigger picture; shorter periods focus on recent action.";

export const HELP_INTERVAL =
  "The time each bar/candle represents. '1d' means each candle is one trading day. '1wk' means weekly candles (less noise, bigger picture).";

// ---------------------------------------------------------------------------
// Header / global
// ---------------------------------------------------------------------------

export const HELP_TRADE_MODE =
  "Changes how indicators are calculated. Swing Trade uses fast, responsive settings (days to weeks). Long-Term uses slower settings for position trading (weeks to months).";

export const HELP_USER_MODE =
  "Normal mode shows a clean, simplified view. Power User mode will show additional detail and advanced controls (coming soon).";

// ---------------------------------------------------------------------------
// Indicator lookup by config_key (for dynamic table rows)
// ---------------------------------------------------------------------------

export const INDICATOR_HELP: Record<string, string> = {
  rsi: HELP_RSI,
  macd: HELP_MACD,
  bollinger_bands: HELP_BOLLINGER,
  moving_averages: HELP_MOVING_AVERAGES,
  stochastic: HELP_STOCHASTIC,
  adx: HELP_ADX,
  volume: HELP_VOLUME,
  fibonacci: HELP_FIBONACCI,
};

export const PATTERN_HELP: Record<string, string> = {
  gaps: HELP_GAPS,
  volume_range: HELP_VOLUME_RANGE,
  candlesticks: HELP_CANDLESTICKS,
  spikes: HELP_SPIKES,
  inside_outside: HELP_INSIDE_OUTSIDE,
};

// ---------------------------------------------------------------------------
// Scanner page
// ---------------------------------------------------------------------------

export const HELP_SCANNER =
  "The Market Scanner scans a group of stocks (called a 'universe') and ranks them by how strongly they match a trading strategy. Think of it as a filter that finds the best opportunities for you.";

export const HELP_UNIVERSE =
  "A pre-defined list of stocks to scan. For example, S&P 500 contains the 500 largest US companies. Pick a universe that matches what you want to trade.";

export const HELP_PRESET =
  "The type of trade setup you're looking for. Each preset uses different indicators and scoring to find stocks matching that strategy.";

export const HELP_PRESET_BREAKOUT =
  "Looks for stocks about to break out of a consolidation range — typically accompanied by increasing volume and narrowing Bollinger Bands.";

export const HELP_PRESET_PULLBACK =
  "Finds stocks in an uptrend that have temporarily pulled back to a support level — a potential buying opportunity at a better price.";

export const HELP_PRESET_REVERSAL =
  "Identifies stocks that may be reversing direction — e.g. oversold stocks showing signs of turning around, like extreme RSI readings.";

export const HELP_PRESET_DIVIDEND =
  "Ranks stocks by dividend quality: yield (how much they pay), growth (are dividends increasing?), consistency, and streak length.";

export const HELP_MIN_VOLUME =
  "Minimum average daily trading volume. Stocks with very low volume are harder to buy/sell and have wider bid-ask spreads. 1M shares/day is a good default.";

export const HELP_MIN_PRICE =
  "Filters out very cheap stocks (penny stocks). These are often more volatile and risky. $5 is a common minimum for swing trading.";

export const HELP_MAX_ATR_RATIO =
  "Maximum ATR/price ratio. ATR measures how much a stock moves per day as a percentage of its price. Higher = more volatile. Leave blank for no filter.";

export const HELP_SCAN_SCORE =
  "The composite score from the scan, 0-10. Higher means the stock more closely matches the selected preset's ideal setup.";

export const HELP_SCAN_SIGNAL =
  "A plain-language label describing the type and strength of the detected trading signal.";

export const HELP_SCAN_CONFIDENCE =
  "How confidently the algorithm classified the stock's market regime. Higher confidence means the stock clearly fits one market condition.";

export const HELP_SCAN_PATTERN =
  "The most notable chart pattern detected for this stock, if any. Patterns can confirm or add context to the trading signal.";

// ---------------------------------------------------------------------------
// Sectors & Segments page
// ---------------------------------------------------------------------------

export const HELP_SECTORS =
  "Shows which parts of the market (sectors) are strong or weak. Green sectors have upward momentum; red sectors are declining. Use this to find where the money is flowing.";

export const HELP_SECTOR_HEATMAP =
  "A visual grid of all 11 market sectors, colored by their momentum score. Dark green = strong uptrend, dark red = strong downtrend, gray = neutral.";

export const HELP_MOMENTUM_SCORE =
  "A combined score (0-10) measuring how strong a sector's price trend is. It blends 1-week, 1-month, and 3-month returns with relative strength vs the S&P 500.";

export const HELP_RELATIVE_STRENGTH =
  "How a sector performs compared to the overall market (S&P 500). Positive = outperforming the market, negative = underperforming. Helps identify where money is rotating.";

export const HELP_SECTOR_ROTATION =
  "Capital flows between sectors over time. When one sector strengthens while another weakens, it suggests investors are 'rotating' money from one area to another.";

export const HELP_SECTOR_REGIME =
  "Each sector can be in a different market condition — one might be trending strongly while another is stuck in a range. This shows the current 'personality' of each sector.";

export const HELP_RETURN_1W =
  "How much the sector ETF's price changed over the past week (5 trading days). Positive = up, negative = down.";

export const HELP_RETURN_1M =
  "How much the sector ETF's price changed over the past month (~21 trading days).";

export const HELP_RETURN_3M =
  "How much the sector ETF's price changed over the past 3 months (~63 trading days). Shows the bigger trend.";

export const HELP_TOP_MOVERS =
  "The best-performing stocks within this sector over the past month. These are the stocks driving the sector higher.";

export const HELP_WORST_MOVERS =
  "The worst-performing stocks within this sector over the past month. These are dragging the sector down.";

export const HELP_SECTOR_ETF =
  "A Sector ETF (Exchange-Traded Fund) tracks all major stocks in one sector. For example, XLK tracks all Technology stocks in the S&P 500. We use these as a quick proxy for sector health.";

export const HELP_BENCHMARK =
  "The S&P 500 (SPY) is used as the benchmark. Relative strength compares each sector against this benchmark to see which sectors are outperforming or underperforming the broader market.";
