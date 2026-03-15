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

export const HELP_HOLDINGS_EDITOR =
  "The list of stocks that represent this sector. Power Users can add, remove, or reorder holdings. Changes are saved globally and affect all users. Use 'Refresh' to fetch the latest holdings from yfinance, or 'Reset' to restore the built-in defaults.";

export const HELP_HOLDINGS_SOURCE =
  "Where the current holdings list came from. 'Default' = built-in list, 'Configured' = customized by a Power User or refreshed from yfinance.";

// ---------------------------------------------------------------------------
// Strategy Lab — Backtest
// ---------------------------------------------------------------------------

export const HELP_BACKTEST =
  "A backtest simulates how a trading strategy would have performed on historical data. It's like a 'time machine' that tests your strategy on past prices to see if it would have made money.";

export const HELP_INITIAL_CASH =
  "The amount of money you start with in the simulation. This is virtual money — no real trades are placed.";

export const HELP_COMMISSION =
  "The fee your broker charges per trade, expressed as a percentage of the trade value. Most modern brokers charge $0 (0%), but some charge 0.01-0.1%.";

export const HELP_SLIPPAGE =
  "The difference between the expected price and the actual execution price. In real markets, you rarely get the exact price you see — slippage accounts for this gap.";

export const HELP_STOP_LOSS =
  "An automatic exit that limits how much you can lose on a single trade. For example, 5% means if the price drops 5% from your entry, the position is automatically closed.";

export const HELP_TAKE_PROFIT =
  "An automatic exit that locks in gains when the price reaches a target. For example, 15% means if the price rises 15% from your entry, the position is automatically closed.";

export const HELP_BT_TOTAL_RETURN =
  "The total percentage gain or loss from start to finish. If you started with $100k and ended with $120k, total return is +20%.";

export const HELP_ANNUALIZED_RETURN =
  "The average yearly return, adjusted for the time period. A 50% return over 2 years is about 22% annualized. This lets you compare strategies across different time periods.";

export const HELP_MAX_DRAWDOWN =
  "The largest peak-to-trough decline during the backtest. If your portfolio went from $120k to $90k at its worst, that's a 25% drawdown. Lower is better — it measures your worst-case pain.";

export const HELP_SHARPE_RATIO =
  "A measure of risk-adjusted return. It tells you how much return you get per unit of risk. Above 1.0 is good, above 2.0 is great, below 0 means the strategy lost money.";

export const HELP_WIN_RATE =
  "The percentage of trades that were profitable. A 60% win rate means 6 out of every 10 trades made money. Note: a low win rate can still be profitable if winners are much larger than losers.";

export const HELP_PROFIT_FACTOR =
  "Total profits divided by total losses. Above 1.0 means the strategy is profitable overall. For example, 1.5 means you made $1.50 for every $1.00 lost.";

export const HELP_TOTAL_TRADES =
  "How many round-trip trades (buy then sell, or short then cover) the strategy made during the backtest period.";

export const HELP_AVG_TRADE_PNL =
  "The average profit or loss per trade, as a percentage. Positive means trades are profitable on average.";

export const HELP_BEST_TRADE =
  "The single most profitable trade during the backtest, shown as a percentage gain.";

export const HELP_WORST_TRADE =
  "The single biggest losing trade during the backtest, shown as a percentage loss.";

export const HELP_AVG_BARS_HELD =
  "On average, how many trading days each position was held before being closed. Shorter = more active trading.";

export const HELP_EQUITY_CURVE =
  "A chart showing how your portfolio value changed over time. An upward-sloping line means the strategy was making money; dips show drawdown periods.";

export const HELP_TRADE_LOG =
  "A detailed list of every trade the strategy made — when it entered, when it exited, whether it won or lost, and why it exited (signal change, stop loss, or take profit).";

// ---------------------------------------------------------------------------
// Strategy Lab — Walk-Forward Testing
// ---------------------------------------------------------------------------

export const HELP_ROBUSTNESS =
  "Robustness analysis tests your strategy across multiple non-overlapping time periods. If it works consistently across different market conditions, you can be more confident it's not just overfitting to one lucky period.";

export const HELP_TRAIN_YEARS =
  "How many years of data to use as the 'training' period in each window. The strategy uses this data to warm up its indicators before trading begins.";

export const HELP_TEST_YEARS =
  "How many years each out-of-sample test window covers. The strategy is evaluated only on this period — it's data the strategy hasn't 'seen' during warmup.";

export const HELP_STABILITY_SCORE =
  "A score from 0-100 measuring how consistent the strategy is across windows. High (70+) = reliable performance. Low (<40) = results vary wildly, strategy may be overfit.";

export const HELP_WORST_WINDOW =
  "The single worst-performing test window. This is your 'stress test' — it shows the strategy's performance during its most challenging market period.";

export const HELP_RETURN_STD_DEV =
  "Standard deviation of returns across windows. Lower means the strategy performs consistently; higher means it's feast-or-famine.";

// ---------------------------------------------------------------------------
// Strategy Lab — Auto-Tuner
// ---------------------------------------------------------------------------

export const HELP_AUTO_TUNER =
  "The auto-tuner uses AI-driven optimisation to find the best strategy parameters for your chosen objective. It tests many different parameter combinations and validates each one using walk-forward testing — so results are based on out-of-sample data, not just curve-fitting.";

export const HELP_TUNER_OBJECTIVE =
  "What you want the optimiser to prioritise. 'Beat Buy-and-Hold' tries to outperform simply holding the stock. 'Balanced' aims for a good mix of return, risk, and consistency.";

export const HELP_N_TRIALS =
  "How many different parameter combinations to test. More trials = better results, but takes longer. 30 is a good balance of speed and quality.";

export const HELP_IMPROVEMENT =
  "How much better the tuned parameters performed compared to the default settings, as a percentage of the objective score.";

export const HELP_BASELINE =
  "Performance using the default (un-tuned) parameters. The auto-tuner's job is to beat this baseline.";

export const HELP_BEST_PARAMS =
  "The parameter values that produced the best objective score across all trials. These can be saved as a strategy in the Strategy Library.";

export const HELP_SENSITIVITY =
  "Shows which parameters matter most for your chosen objective. Parameters with high importance have the biggest impact on performance — these are the ones worth tuning.";

export const HELP_BEAT_BUY_HOLD =
  "Optimises for excess return over buy-and-hold. The goal is to outperform simply buying the stock and holding it for the same period.";

export const HELP_MAX_RETURN =
  "Optimises for the highest possible annual return. Accepts more risk in pursuit of maximum gains.";

export const HELP_MAX_RISK_ADJUSTED =
  "Optimises for the best Sharpe ratio — high returns relative to the amount of risk taken. Good for consistent performance.";

export const HELP_MIN_DRAWDOWN =
  "Optimises to minimise the largest peak-to-trough drop while still maintaining positive returns. Best for risk-averse investors.";

export const HELP_BALANCED =
  "A weighted combination of return, risk, drawdown, and win rate. Good default for most users — tries to balance everything.";

export const HELP_TUNER_MODE =
  "Choose how to tune: Single Ticker optimises for one stock. Sector tunes across all major stocks in a GICS sector (e.g. Technology). Custom Group lets you pick your own list of stocks to optimise together.";

export const HELP_TUNER_SECTOR =
  "Pick a market sector. The auto-tuner will optimise parameters across the top stocks in that sector, finding settings that work well for the group — not just one stock.";

export const HELP_TUNER_CUSTOM_GROUP =
  "Enter a comma-separated list of tickers (e.g. TSLA, RIVN, NIO). The auto-tuner will find parameters that perform well across all of them — useful for stocks in the same industry or theme.";

export const HELP_TUNER_GROUP_BENEFIT =
  "Tuning across a group of similar stocks produces more robust parameters. Instead of overfitting to one stock's history, the optimiser finds settings that generalise across the whole group.";

// ---------------------------------------------------------------------------
// Strategy Lab — Strategy Library
// ---------------------------------------------------------------------------

export const HELP_STRATEGY_LIBRARY =
  "Your saved strategy configurations. Each strategy is a set of parameters that control how the trading algorithm makes decisions. Save winning parameter sets here, compare them, and activate them for portfolio simulation.";

export const HELP_STRATEGY_PRESET =
  "A built-in strategy template created by experts. You can't edit presets, but you can use them as a starting point, activate them for portfolio simulation, or export them.";

export const HELP_STRATEGY_VERSION =
  "Each time you update a strategy's parameters, the version number increases. This helps you track changes over time.";

export const HELP_STRATEGY_ACTIVE =
  "An 'active' strategy will be used in Portfolio Simulation to generate trade signals. You can have multiple strategies active at once.";

export const HELP_STRATEGY_PARAMS =
  "The specific parameter values that control this strategy — things like score thresholds, indicator weights, stop-loss levels, etc. These are what make each strategy unique.";

export const HELP_STRATEGY_COMPARE =
  "Compare two or more strategies side-by-side on key metrics like return, Sharpe ratio, drawdown, and win rate. Helps you pick the best one.";

export const HELP_STRATEGY_EXPORT =
  "Download a strategy as a JSON file that you can share with others or back up. The file contains all parameters and performance metrics.";

export const HELP_STRATEGY_IMPORT =
  "Load a strategy from a JSON file. This is useful for importing strategies shared by others or restoring from a backup.";

// ---------------------------------------------------------------------------
// Settings — Canadian Tax
// ---------------------------------------------------------------------------

export const HELP_TAX_PROVINCE =
  "Your province or territory of residence. Tax rates vary significantly between provinces — for example, Alberta has the lowest provincial rates, while Nova Scotia has some of the highest.";

export const HELP_TAX_ANNUAL_INCOME =
  "Your annual employment income before trading profits. This determines your marginal tax bracket — the rate applied to the next dollar you earn. Trading profits are taxed at this marginal rate.";

export const HELP_TAX_TREATMENT =
  "How the CRA classifies your trading profits. Capital Gains: only 50% of profit is taxable (most investors). Business Income: 100% is taxable (frequent traders). Auto-detect guesses based on your trade mode.";

export const HELP_TAX_MARGINAL_RATE =
  "Your combined federal + provincial marginal tax rate. This is the rate applied to each additional dollar of income. It's NOT your average tax rate — it's the rate on your trading profits specifically.";

export const HELP_TOTAL_TAX_PAID =
  "The total tax deducted from profitable trades during the backtest. Tax is only applied to winning trades — losing trades are not taxed (though capital losses can offset future gains in real life).";

export const HELP_AFTER_TAX_RETURN =
  "Your total return after deducting Canadian income tax from each profitable trade. This is the 'real' return you would actually keep.";

// ---------------------------------------------------------------------------
// AI / LLM
// ---------------------------------------------------------------------------

export const HELP_LLM_PROVIDER =
  "Which AI model generates the qualitative analysis. Gemini has a free tier, Anthropic (Claude) and OpenAI (GPT) require paid API keys. You can add your key below or set it as a server environment variable.";

export const HELP_AI_ANALYSIS =
  "An AI-generated plain-English summary of all the technical data above. It highlights the key takeaways so you don't have to interpret every indicator yourself. This costs one API call each time you request it.";

export const HELP_LLM_API_KEY =
  "Your personal API key for the selected provider. Get one from the provider's website (e.g. aistudio.google.com for Gemini). Your key is stored securely on the server and never shown in full after saving.";

export const HELP_LLM_MODEL =
  "Which specific model to use from your chosen provider. 'Provider default' uses the recommended model. More capable models (like GPT-4o or Claude Sonnet) produce better analysis but cost more per call.";
