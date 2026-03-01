"""scanner.py — Stock universe scanner engine.

Scans a list of tickers in parallel, runs the full analysis + recommendation
pipeline on each, and ranks them to surface the top BUY and SELL candidates.

Also provides a **DCA scanner** mode that ranks tickers by Dollar-Cost
Averaging attractiveness — answering "where should I allocate my next DCA
dollar?"

Designed to be used from both the CLI (``scan.py``) and the Streamlit dashboard.

Usage (programmatic)::

    from scanner import Scanner, DCAScanner
    scanner = Scanner(universe="dow30", period="2y", max_workers=8)
    results = scanner.run()          # list[ScanResult]
    buys  = scanner.top_buys(10)     # top 10 BUY signals by score

    dca = DCAScanner(universe="dow30", period="2y", max_workers=8)
    dca_results = dca.run()          # list[DCAScanResult]
    top_dca = dca.top_dca(10)        # top 10 DCA opportunities
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ── Ensure project root is on sys.path ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.analyzer import Analyzer
from analysis.multi_timeframe import MultiTimeframeAnalyzer, _build_percentile_window
from config import Config
from data.provider import DataProvider
from data.universes import available as available_universes
from data.universes import load as load_universe
from data.yahoo import YahooFinanceProvider
from engine.regime import RegimeSubType, RegimeType
from engine.score_strategy import ScoreBasedStrategy
from engine.suitability import SuitabilityAnalyzer, TradingMode
from engine.strategy import StrategyContext
from engine.watchlist import DCAContext, compute_dca_context


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ScanResult:
    """Result of scanning a single ticker."""

    ticker: str
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: str  # "high", "medium", "low"
    effective_score: float
    indicator_score: float
    pattern_score: float
    price: float
    regime: str  # e.g. "strong_trend"
    regime_label: str
    sub_type: str  # e.g. "steady_compounder"
    sub_type_label: str
    strategy_notes: str
    trend_score: float = 5.0
    contrarian_score: float = 5.0
    dominant_group: str = "none"
    error: str = ""  # non-empty if the ticker failed
    # Multi-timeframe fields (populated when multi_timeframe=True)
    mt_agreement: str = ""          # "aligned", "mixed", "conflicting", or ""
    mt_aggregated_signal: str = ""  # aggregated signal across timeframes
    mt_aggregated_score: float = 0.0
    mt_daily_signal: str = ""
    mt_weekly_signal: str = ""
    mt_monthly_signal: str = ""

    @property
    def sort_key_buy(self) -> float:
        """Higher is better for BUY ranking."""
        return self.effective_score

    @property
    def sort_key_sell(self) -> float:
        """Lower is better for SELL ranking (inverted for sorting)."""
        return -self.effective_score


REGIME_LABELS = {
    "strong_trend": "Strong Trend",
    "mean_reverting": "Mean-Reverting / Range-Bound",
    "volatile_choppy": "Volatile / Choppy",
    "breakout_transition": "Breakout / Transition",
}

DCA_TIER_LABELS = {
    "normal": "Normal",
    "mild_dip": "Mild Dip",
    "strong_dip": "Strong Dip",
    "extreme_dip": "Extreme Dip",
}

DCA_REGIME_LABELS = {
    "bull": "Bull",
    "bear": "Bear",
    "crisis": "Crisis",
    "sideways": "Sideways",
    "recovery": "Recovery",
}


@dataclass
class DCAScanResult:
    """Result of DCA-scanning a single ticker.

    Produced by :class:`DCAScanner` — ranks tickers by DCA attractiveness
    rather than active trading signals.
    """

    ticker: str
    price: float
    # ── DCA context fields ──
    dca_score: float          # 0-100 composite DCA attractiveness score
    dip_pct: float            # percentage drop from rolling high
    dip_sigma: float          # dip expressed in standard deviations
    tier: str                 # final tier after all adjustments
    tier_label: str           # human-readable tier label
    multiplier: float         # DCA allocation multiplier
    is_dca_buy: bool          # True if this is a good DCA buy opportunity
    confidence: str           # "high", "medium", "low"
    regime: str               # investor-friendly: bull/bear/crisis/sideways/recovery
    regime_label: str         # human-readable regime label
    rsi: float                # raw RSI value (0-100)
    bb_pctile: float          # Bollinger Band %B (0-100 scale)
    volatility: float         # annualised volatility (%)
    composite_score: float    # composite indicator score (0-10)
    explanation: list[str] = field(default_factory=list)
    error: str = ""           # non-empty if the ticker failed

    @property
    def sort_key(self) -> float:
        """Higher is better for DCA ranking."""
        return self.dca_score


# ─────────────────────────────────────────────────────────────────────────────
# Scanner engine
# ─────────────────────────────────────────────────────────────────────────────


class Scanner:
    """Parallel stock-universe scanner.

    Parameters
    ----------
    universe : str | list[str]
        A built-in universe name (``dow30``, ``nasdaq100``, ``sp500``),
        a path to a custom ``.txt`` file, or an explicit list of ticker
        strings.
    period : str
        Analysis period (default ``"2y"``).
    max_workers : int
        Thread pool size for parallel fetching (default ``8``).
    cfg : Config | None
        Optional pre-built Config.  Uses ``Config.load()`` if *None*.
    on_progress : callable | None
        ``(completed: int, total: int, ticker: str, result: ScanResult | None) -> None``
        Called after each ticker finishes.  Useful for progress bars.
    """

    def __init__(
        self,
        universe: str | list[str] = "dow30",
        period: str = "2y",
        max_workers: int = 8,
        cfg: Config | None = None,
        on_progress: Callable[[int, int, str, ScanResult | None], None] | None = None,
        multi_timeframe: bool = False,
    ) -> None:
        self._period = period
        self._max_workers = max_workers
        self._cfg = cfg or Config.load()
        self._on_progress = on_progress
        self._multi_timeframe = multi_timeframe

        # Resolve tickers
        if isinstance(universe, list):
            self._tickers = [t.upper() for t in universe]
            self._universe_name = "custom"
        else:
            self._tickers = load_universe(universe)
            self._universe_name = universe

        self._results: list[ScanResult] = []

    @property
    def universe_name(self) -> str:
        return self._universe_name

    @property
    def tickers(self) -> list[str]:
        return list(self._tickers)

    @property
    def results(self) -> list[ScanResult]:
        return list(self._results)

    # ── Core scan ────────────────────────────────────────────────────────

    def run(self) -> list[ScanResult]:
        """Run the full scan.  Returns list of ScanResult (one per ticker)."""
        self._results = []
        total = len(self._tickers)
        completed = 0

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(self._scan_one, ticker): ticker
                for ticker in self._tickers
            }

            for future in as_completed(futures):
                ticker = futures[future]
                completed += 1
                try:
                    result = future.result()
                except Exception as exc:
                    result = ScanResult(
                        ticker=ticker,
                        signal="HOLD",
                        confidence="low",
                        effective_score=5.0,
                        indicator_score=5.0,
                        pattern_score=5.0,
                        price=0.0,
                        regime="",
                        regime_label="",
                        sub_type="",
                        sub_type_label="",
                        strategy_notes="",
                        error=str(exc),
                    )
                self._results.append(result)
                if self._on_progress:
                    self._on_progress(completed, total, ticker, result)

        return list(self._results)

    def _scan_one(self, ticker: str) -> ScanResult:
        """Analyse a single ticker and return its recommendation."""
        # Each thread gets its own provider (yfinance is thread-safe for
        # separate tickers, but sharing one instance can cause issues).
        provider = YahooFinanceProvider()
        analyzer = Analyzer(self._cfg, provider)

        result = analyzer.run(ticker, self._period, "1d")

        # ── Build recommendation (mirrors dashboard.compute_recommendation) ──
        strat_cfg = self._cfg.section("strategy")
        regime_adapt = self._cfg.section("regime").get("strategy_adaptation", {})

        # ── Determine trading mode via suitability analysis ──────────────
        trading_mode = TradingMode.LONG_SHORT  # fallback
        try:
            suit_assessment = SuitabilityAnalyzer(self._cfg).assess(result.df)
            trading_mode = suit_assessment.mode
        except Exception:
            pass  # graceful degradation — default to LONG_SHORT

        strategy = ScoreBasedStrategy(
            params=strat_cfg,
            trading_mode=trading_mode,
            regime_adaptation=regime_adapt,
        )
        strategy.on_start({"ticker": ticker, "recommendation": True})

        # ── Seed percentile window if percentile mode is active ──────────
        # Without seeding, percentile mode always falls back to fixed
        # thresholds because a single on_bar() call can never fill the
        # min_samples window.
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

                window = _build_percentile_window(
                    self._cfg,
                    provider,
                    ticker=ticker,
                    period=self._period,
                    interval="1d",
                    lookback_bars=lookback_bars,
                    step=pct_step,
                    combination_mode=combination_mode,
                    ind_weight=ind_weight,
                    pat_weight=pat_weight,
                    boost_strength=boost_strength,
                    boost_dead_zone=boost_dead_zone,
                )
                if window:
                    strategy.seed_score_window(window)
            except Exception:
                pass  # graceful degradation — percentile falls back to fixed

        df = result.df
        if df is None or df.empty:
            raise ValueError(f"No data returned for {ticker}")

        last_row = df.iloc[-1]
        bar_dict = {
            "open": float(last_row["open"]),
            "high": float(last_row["high"]),
            "low": float(last_row["low"]),
            "close": float(last_row["close"]),
            "volume": float(last_row["volume"]),
        }

        overall_score = result.composite.get("overall", 5.0)
        pattern_score = result.pattern_composite.get("overall", 5.0)

        per_scores: dict[str, float] = {}
        if result.indicator_results:
            for ir in result.indicator_results:
                if not ir.error:
                    per_scores[ir.config_key] = ir.score

        # Trend MA
        trend_period = int(strat_cfg.get("trend_confirm_period", 20))
        ma_type = str(strat_cfg.get("trend_confirm_ma_type", "ema"))
        if ma_type.lower() == "sma":
            trend_ma_series = df["close"].rolling(window=trend_period, min_periods=1).mean()
        else:
            trend_ma_series = df["close"].ewm(span=trend_period, adjust=False).mean()
        current_trend_ma = float(trend_ma_series.iloc[-1])

        # Regime
        regime_type: RegimeType | None = None
        regime_sub: RegimeSubType | None = None
        regime_trend: str = "neutral"
        regime_total_return: float = 0.0

        if result.regime is not None:
            regime_type = result.regime.regime
            regime_sub = result.regime.sub_type
            regime_trend = result.regime.metrics.trend_direction
            regime_total_return = result.regime.metrics.total_return

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
            position=0.0,
            cash=100_000.0,
            portfolio_value=100_000.0,
            trend_ma=current_trend_ma,
            regime=regime_type,
            regime_sub_type=regime_sub,
            regime_trend=regime_trend,
            regime_total_return=regime_total_return,
            metadata={"avg_volume": avg_volume},
        )

        order = strategy.on_bar(ctx)
        signal = order.signal.value  # "BUY", "SELL", "HOLD"

        # Effective score (same blending as dashboard)
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
            effective_score = overall_score
        else:
            w_total = ind_weight + pat_weight
            if w_total > 0:
                effective_score = (ind_weight * overall_score + pat_weight * pattern_score) / w_total
            else:
                effective_score = overall_score

        # Confidence
        thresholds = strat_cfg.get("score_thresholds", {})
        short_below = float(thresholds.get("short_below", 3.5))
        hold_below = float(thresholds.get("hold_below", 6.0))

        conf_cfg = strat_cfg.get("confidence_thresholds", {})
        conf_high = float(conf_cfg.get("high", 1.5))
        conf_med = float(conf_cfg.get("medium", 0.5))
        hold_conf_high = float(conf_cfg.get("hold_high", 1.0))
        hold_conf_med = float(conf_cfg.get("hold_medium", 0.3))

        if signal == "BUY":
            distance = effective_score - hold_below
            confidence = "high" if distance >= conf_high else ("medium" if distance >= conf_med else "low")
        elif signal == "SELL":
            distance = short_below - effective_score
            confidence = "high" if distance >= conf_high else ("medium" if distance >= conf_med else "low")
        else:
            dist_to_buy = hold_below - effective_score
            dist_to_sell = effective_score - short_below
            min_dist = min(dist_to_buy, dist_to_sell)
            confidence = "high" if min_dist >= hold_conf_high else ("medium" if min_dist >= hold_conf_med else "low")

        # Labels
        regime_val = regime_type.value if regime_type else ""
        regime_label = REGIME_LABELS.get(regime_val, regime_val)
        sub_val = regime_sub.value if regime_sub else ""
        sub_label = result.regime.sub_type_label if result.regime else ""

        # ── Multi-timeframe confirmation ────────────────────────────────────
        mt_agreement = ""
        mt_aggregated_signal = ""
        mt_aggregated_score = 0.0
        mt_daily_signal = ""
        mt_weekly_signal = ""
        mt_monthly_signal = ""

        if self._multi_timeframe:
            try:
                mt_provider = YahooFinanceProvider()
                mt_analyzer = MultiTimeframeAnalyzer(self._cfg, mt_provider)
                mt_result = mt_analyzer.run(ticker)
                mt_agreement = mt_result.agreement
                mt_aggregated_signal = mt_result.aggregated_signal
                mt_aggregated_score = (
                    mt_result.aggregated_indicator_score * ind_weight
                    + mt_result.aggregated_pattern_score * pat_weight
                ) / max(ind_weight + pat_weight, 0.001)

                tf_signals = {
                    tr.timeframe: tr.signal
                    for tr in mt_result.timeframe_results
                    if tr.error is None
                }
                mt_daily_signal = tf_signals.get("1d", "")
                mt_weekly_signal = tf_signals.get("1wk", "")
                mt_monthly_signal = tf_signals.get("1mo", "")
            except Exception:
                pass  # graceful degradation — leave fields empty

        return ScanResult(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            effective_score=effective_score,
            indicator_score=overall_score,
            pattern_score=pattern_score,
            price=bar_dict["close"],
            regime=regime_val,
            regime_label=regime_label,
            sub_type=sub_val,
            sub_type_label=sub_label,
            strategy_notes=order.notes or "",
            trend_score=result.composite.get("trend_score", 5.0),
            contrarian_score=result.composite.get("contrarian_score", 5.0),
            dominant_group=result.composite.get("dominant_group", "none"),
            mt_agreement=mt_agreement,
            mt_aggregated_signal=mt_aggregated_signal,
            mt_aggregated_score=mt_aggregated_score,
            mt_daily_signal=mt_daily_signal,
            mt_weekly_signal=mt_weekly_signal,
            mt_monthly_signal=mt_monthly_signal,
        )

    # ── Ranking helpers ──────────────────────────────────────────────────

    def top_buys(self, n: int = 10) -> list[ScanResult]:
        """Return top *n* BUY signals, ranked by effective score descending."""
        buys = [r for r in self._results if r.signal == "BUY" and not r.error]
        buys.sort(key=lambda r: r.sort_key_buy, reverse=True)
        return buys[:n]

    def top_sells(self, n: int = 10) -> list[ScanResult]:
        """Return top *n* SELL signals, ranked by effective score ascending."""
        sells = [r for r in self._results if r.signal == "SELL" and not r.error]
        sells.sort(key=lambda r: r.effective_score)
        return sells[:n]

    def holds(self) -> list[ScanResult]:
        """Return all HOLD signals, sorted by score descending."""
        holds = [r for r in self._results if r.signal == "HOLD" and not r.error]
        holds.sort(key=lambda r: r.effective_score, reverse=True)
        return holds

    def errors(self) -> list[ScanResult]:
        """Return all tickers that failed during scan."""
        return [r for r in self._results if r.error]

    def summary(self) -> dict:
        """Return a quick summary dict."""
        ok = [r for r in self._results if not r.error]
        return {
            "universe": self._universe_name,
            "period": self._period,
            "total_tickers": len(self._tickers),
            "scanned": len(ok),
            "errors": len(self._results) - len(ok),
            "buy_count": sum(1 for r in ok if r.signal == "BUY"),
            "sell_count": sum(1 for r in ok if r.signal == "SELL"),
            "hold_count": sum(1 for r in ok if r.signal == "HOLD"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# DCA Scanner engine
# ─────────────────────────────────────────────────────────────────────────────


def _compute_dca_score(ctx: DCAContext, weights: dict) -> float:
    """Compute a 0-100 DCA attractiveness score from a DCAContext.

    The score is a weighted combination of normalised sub-scores:

    * **dip_sigma** — dip severity normalised by volatility (0-100).
    * **tier_multiplier** — allocation multiplier mapped to 0-100.
    * **technical** — inverse of composite indicator score (lower
      composite = more attractive for DCA, since that means oversold).
    * **confidence** — high=100, medium=60, low=20.
    * **regime** — regime favourability for DCA: bull pullback and
      recovery are best; bear/crisis are penalised.

    Weights are configurable via ``dca.scanner.score_weights``.
    """
    w_dip = float(weights.get("dip_sigma", 0.30))
    w_tier = float(weights.get("tier_multiplier", 0.25))
    w_tech = float(weights.get("technical", 0.20))
    w_conf = float(weights.get("confidence", 0.10))
    w_regime = float(weights.get("regime", 0.15))

    # Normalise dip_sigma to 0-100 (cap at 5 sigma = 100)
    dip_sigma_score = min(ctx.dip_sigma / 5.0, 1.0) * 100

    # Tier multiplier: 1.0=0, 1.5=33, 2.0=67, 3.0=100
    tier_score = min((ctx.multiplier - 1.0) / 2.0, 1.0) * 100

    # Technical: lower composite = more DCA attractive.
    # Score 0 → 100, score 10 → 0.
    tech_score = max(0.0, (10.0 - ctx.composite_score) / 10.0) * 100

    # Confidence: high=100, medium=60, low=20
    conf_map = {"high": 100.0, "medium": 60.0, "low": 20.0}
    conf_score = conf_map.get(ctx.confidence, 20.0)

    # Regime: bull pullback is ideal, recovery good, sideways neutral,
    # bear/crisis penalised.
    regime_map = {
        "bull": 70.0,       # bull without dip is ok but not great
        "recovery": 80.0,
        "sideways": 50.0,
        "bear": 25.0,
        "crisis": 10.0,
    }
    regime_score = regime_map.get(ctx.regime, 50.0)
    # Bull + dip = best (already captured by high dip_sigma_score & tier)

    total_weight = w_dip + w_tier + w_tech + w_conf + w_regime
    if total_weight <= 0:
        return 0.0

    raw = (
        w_dip * dip_sigma_score
        + w_tier * tier_score
        + w_tech * tech_score
        + w_conf * conf_score
        + w_regime * regime_score
    ) / total_weight

    return round(max(0.0, min(100.0, raw)), 1)


class DCAScanner:
    """Parallel DCA attractiveness scanner.

    Scans a universe of tickers and ranks them by DCA (Dollar-Cost Averaging)
    attractiveness — answering "where should I allocate my next DCA dollar?"

    Unlike the :class:`Scanner` which ranks by active trading signal strength,
    this scanner looks for: deep dips, high volatility-normalised dip severity,
    oversold technicals, and favourable regime context.

    Parameters
    ----------
    universe : str | list[str]
        Built-in universe name or explicit list of tickers.
    period : str
        Analysis period (default ``"2y"``).
    max_workers : int
        Thread pool size for parallel fetching (default ``8``).
    cfg : Config | None
        Optional pre-built Config.
    on_progress : callable | None
        ``(completed: int, total: int, ticker: str, result: DCAScanResult | None) -> None``
    """

    def __init__(
        self,
        universe: str | list[str] = "dow30",
        period: str = "2y",
        max_workers: int = 8,
        cfg: Config | None = None,
        on_progress: Callable[[int, int, str, DCAScanResult | None], None] | None = None,
    ) -> None:
        self._period = period
        self._max_workers = max_workers
        self._cfg = cfg or Config.load()
        self._on_progress = on_progress

        # Resolve tickers
        if isinstance(universe, list):
            self._tickers = [t.upper() for t in universe]
            self._universe_name = "custom"
        else:
            self._tickers = load_universe(universe)
            self._universe_name = universe

        self._results: list[DCAScanResult] = []

    @property
    def universe_name(self) -> str:
        return self._universe_name

    @property
    def tickers(self) -> list[str]:
        return list(self._tickers)

    @property
    def results(self) -> list[DCAScanResult]:
        return list(self._results)

    # ── Core scan ────────────────────────────────────────────────────────

    def run(self) -> list[DCAScanResult]:
        """Run the full DCA scan.  Returns list of DCAScanResult (one per ticker)."""
        self._results = []
        total = len(self._tickers)
        completed = 0

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(self._scan_one, ticker): ticker
                for ticker in self._tickers
            }

            for future in as_completed(futures):
                ticker = futures[future]
                completed += 1
                try:
                    result = future.result()
                except Exception as exc:
                    result = DCAScanResult(
                        ticker=ticker,
                        price=0.0,
                        dca_score=0.0,
                        dip_pct=0.0,
                        dip_sigma=0.0,
                        tier="normal",
                        tier_label="Normal",
                        multiplier=1.0,
                        is_dca_buy=False,
                        confidence="low",
                        regime="sideways",
                        regime_label="Sideways",
                        rsi=50.0,
                        bb_pctile=50.0,
                        volatility=0.0,
                        composite_score=5.0,
                        error=str(exc),
                    )
                self._results.append(result)
                if self._on_progress:
                    self._on_progress(completed, total, ticker, result)

        return list(self._results)

    def _scan_one(self, ticker: str) -> DCAScanResult:
        """Analyse a single ticker for DCA attractiveness."""
        provider = YahooFinanceProvider()
        analyzer = Analyzer(self._cfg, provider)

        result = analyzer.run(ticker, self._period, "1d")

        df = result.df
        if df is None or df.empty:
            raise ValueError(f"No data returned for {ticker}")

        last_row = df.iloc[-1]
        price = float(last_row["close"])

        # ── Indicator composite & results ────────────────────────────────
        ind_composite = result.composite.get("overall", 5.0)
        ind_results = result.indicator_results or []

        # ── Regime ───────────────────────────────────────────────────────
        regime_type: RegimeType | None = None
        regime_trend: str = "neutral"
        regime_total_return: float = 0.0

        if result.regime is not None:
            regime_type = result.regime.regime
            regime_trend = result.regime.metrics.trend_direction
            regime_total_return = result.regime.metrics.total_return

        # ── Compute DCA context ──────────────────────────────────────────
        dca_cfg = dict(self._cfg.section("dca"))
        dca_ctx = compute_dca_context(
            df,
            ind_composite=ind_composite,
            ind_results=ind_results,
            regime_type=regime_type,
            regime_trend=regime_trend,
            regime_total_return=regime_total_return,
            dca_cfg=dca_cfg,
        )

        # ── Compute DCA attractiveness score ─────────────────────────────
        scanner_cfg = dca_cfg.get("scanner", {})
        score_weights = scanner_cfg.get("score_weights", {})
        dca_score = _compute_dca_score(dca_ctx, score_weights)

        return DCAScanResult(
            ticker=ticker,
            price=price,
            dca_score=dca_score,
            dip_pct=dca_ctx.dip_pct,
            dip_sigma=dca_ctx.dip_sigma,
            tier=dca_ctx.tier,
            tier_label=DCA_TIER_LABELS.get(dca_ctx.tier, dca_ctx.tier),
            multiplier=dca_ctx.multiplier,
            is_dca_buy=dca_ctx.is_dca_buy,
            confidence=dca_ctx.confidence,
            regime=dca_ctx.regime,
            regime_label=DCA_REGIME_LABELS.get(dca_ctx.regime, dca_ctx.regime),
            rsi=dca_ctx.rsi,
            bb_pctile=dca_ctx.bb_pctile,
            volatility=dca_ctx.volatility,
            composite_score=dca_ctx.composite_score,
            explanation=dca_ctx.explanation,
        )

    # ── Ranking helpers ──────────────────────────────────────────────────

    def top_dca(self, n: int = 10) -> list[DCAScanResult]:
        """Return top *n* DCA opportunities, ranked by DCA score descending.

        Only includes tickers where ``is_dca_buy`` is True.
        """
        candidates = [r for r in self._results if r.is_dca_buy and not r.error]
        candidates.sort(key=lambda r: r.sort_key, reverse=True)
        return candidates[:n]

    def all_ranked(self, n: int | None = None) -> list[DCAScanResult]:
        """Return all results ranked by DCA score descending (including non-buys)."""
        ranked = [r for r in self._results if not r.error]
        ranked.sort(key=lambda r: r.sort_key, reverse=True)
        if n is not None:
            return ranked[:n]
        return ranked

    def errors(self) -> list[DCAScanResult]:
        """Return all tickers that failed during scan."""
        return [r for r in self._results if r.error]

    def summary(self) -> dict:
        """Return a quick summary dict."""
        ok = [r for r in self._results if not r.error]
        buys = [r for r in ok if r.is_dca_buy]
        return {
            "universe": self._universe_name,
            "period": self._period,
            "total_tickers": len(self._tickers),
            "scanned": len(ok),
            "errors": len(self._results) - len(ok),
            "dca_buy_count": len(buys),
            "non_buy_count": len(ok) - len(buys),
            "avg_dca_score": round(sum(r.dca_score for r in ok) / len(ok), 1) if ok else 0.0,
            "high_conf_count": sum(1 for r in buys if r.confidence == "high"),
            "medium_conf_count": sum(1 for r in buys if r.confidence == "medium"),
            "low_conf_count": sum(1 for r in buys if r.confidence == "low"),
        }
