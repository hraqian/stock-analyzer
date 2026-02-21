"""
dashboard.py — Streamlit-based graphical dashboard for the stock analyzer.

Phase 1: Read-only dashboard for visual analysis and backtesting.
Run with:  streamlit run dashboard.py

Features:
  - Sidebar: ticker, period/date-range, interval, objective preset
  - Candlestick price chart with indicator & pattern score overlays
  - Support/resistance levels on the price chart
  - Indicator & pattern breakdown tables
  - Backtest equity curve with trade entry/exit markers
  - Performance metrics summary
  - Suitability assessment display
  - Score distribution histogram
"""

from __future__ import annotations

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
from analysis.scorer import CompositeScorer
from analysis.pattern_scorer import PatternCompositeScorer
from config import Config
from data.yahoo import YahooFinanceProvider
from engine.backtest import BacktestEngine, BacktestResult
from engine.score_strategy import ScoreBasedStrategy
from engine.suitability import SuitabilityAnalyzer, TradingMode
from indicators.registry import IndicatorRegistry
from patterns.registry import PatternRegistry

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
# Helpers
# ---------------------------------------------------------------------------

def score_color(score: float) -> str:
    """Return hex color for a 0-10 score."""
    if score <= 3.5:
        return COLOR_BEARISH
    if score <= 6.5:
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
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    """Render sidebar controls and return user selections."""
    st.sidebar.title("Stock Analyzer")
    st.sidebar.markdown("---")

    ticker = st.sidebar.text_input("Ticker", value="AAPL").upper().strip()

    st.sidebar.markdown("#### Data Range")
    date_mode = st.sidebar.radio(
        "Range mode",
        ["Period", "Custom dates"],
        horizontal=True,
    )

    period = None
    start_date = None
    end_date = None

    if date_mode == "Period":
        period = st.sidebar.selectbox("Period", VALID_PERIODS, index=2)
    else:
        today = datetime.date.today()
        default_start = today - datetime.timedelta(days=365 * 2)
        start_date = st.sidebar.date_input("Start date", value=default_start)
        end_date = st.sidebar.date_input("End date", value=today)
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")

    interval = st.sidebar.selectbox("Interval", ALL_INTERVALS, index=ALL_INTERVALS.index("1d"))

    st.sidebar.markdown("---")

    # Objective preset
    cfg_temp = Config.load()
    objectives = ["(none)"] + cfg_temp.available_objectives()
    objective = st.sidebar.selectbox("Objective preset", objectives, index=0)
    if objective == "(none)":
        objective = None

    st.sidebar.markdown("---")

    # Backtest toggle
    run_backtest = st.sidebar.checkbox("Run backtest", value=False)
    trading_mode = "auto"
    if run_backtest:
        trading_mode = st.sidebar.selectbox("Trading mode", TRADING_MODES, index=0)

    return {
        "ticker": ticker,
        "period": period,
        "start": start_date,
        "end": end_date,
        "interval": interval,
        "objective": objective,
        "run_backtest": run_backtest,
        "trading_mode": trading_mode,
    }


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def load_analysis(
    ticker: str,
    period: str | None,
    interval: str,
    start: str | None,
    end: str | None,
    objective: str | None,
    config_path: str | None = None,
) -> tuple[AnalysisResult, Config]:
    """Run the full analysis pipeline and return (result, cfg)."""
    cfg = Config.load(config_path)
    if objective:
        cfg.apply_objective(objective)

    provider = YahooFinanceProvider()
    analyzer = Analyzer(cfg, provider)
    result = analyzer.run(
        ticker,
        period=period if not start else None,
        interval=interval,
        start=start,
        end=end,
    )
    return result, cfg


@st.cache_data(ttl=300, show_spinner="Running backtest...")
def load_backtest(
    ticker: str,
    period: str | None,
    interval: str,
    start: str | None,
    end: str | None,
    objective: str | None,
    trading_mode_str: str,
    config_path: str | None = None,
) -> tuple[BacktestResult, TradingMode, "SuitabilityAssessment | None", Config]:
    """Run the backtest engine and return results."""
    cfg = Config.load(config_path)
    if objective:
        cfg.apply_objective(objective)

    provider = YahooFinanceProvider()

    # Determine trading mode
    forced = trading_mode_str != "auto"
    assessment = None

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

    strategy = ScoreBasedStrategy(
        params=cfg.section("strategy"),
        trading_mode=trading_mode,
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
    return bt_result, trading_mode, assessment, cfg


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
    objective: str | None,
    step: int = 5,
) -> pd.DataFrame:
    """Compute indicator and pattern composite scores at regular intervals.

    Returns a DataFrame indexed by date with columns:
      indicator_score, pattern_score

    Walks through the data and computes scores every *step* bars using
    trailing data up to each point (same approach as the backtest engine).
    """
    cfg = Config.load()
    if objective:
        cfg.apply_objective(objective)

    provider = YahooFinanceProvider()
    df = provider.fetch(
        ticker,
        period=period if not start else None,
        interval=interval,
        start=start,
        end=end,
    )

    warmup = int(cfg.section("backtest").get("warmup_bars", 200))
    warmup = min(warmup, len(df) - 10)  # ensure we have some data after warmup
    if warmup < 20:
        warmup = 20

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
        ind_scores.append(ind_composite["overall"])
        pat_scores.append(pat_composite["overall"])

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
        strat_cfg = Config.load()
        thresholds = strat_cfg.section("strategy").get("score_thresholds", {})
        short_below = float(thresholds.get("short_below", 4.5))
        hold_below = float(thresholds.get("hold_below", 5.5))

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

    # Buy-and-hold benchmark
    if len(df) >= len(curve):
        initial_price = float(df["close"].iloc[0])
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


def create_score_histogram(score_df: pd.DataFrame) -> go.Figure:
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

    # Threshold lines
    cfg = Config.load()
    thresholds = cfg.section("strategy").get("score_thresholds", {})
    short_below = float(thresholds.get("short_below", 4.5))
    hold_below = float(thresholds.get("hold_below", 5.5))

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


def render_suitability(assessment, ticker: str) -> None:
    """Render suitability assessment as an info/warning box."""
    mode_label = assessment.mode.value.replace("_", " ").upper()
    mode_map = {
        "long_short": "success",
        "long_only": "warning",
        "hold_only": "error",
    }
    msg_type = mode_map.get(assessment.mode.value, "info")

    cols = st.columns(4)
    cols[0].metric("Trading Mode", mode_label)
    cols[1].metric("Avg Volume", f"{assessment.avg_daily_volume:,.0f}")
    cols[2].metric("ADX", f"{assessment.adx_value:.1f}")
    cols[3].metric("ATR%", f"{assessment.atr_pct * 100:.2f}%")

    for reason in assessment.reasons:
        if msg_type == "error":
            st.error(reason)
        elif msg_type == "warning":
            st.warning(reason)
        else:
            st.success(reason)


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
    st.dataframe(df, use_container_width=True, hide_index=True, height=400)


def render_strategy_config(cfg: Config) -> None:
    """Render strategy configuration summary."""
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
        rows.append(("SHORT when score <=", f"{thresholds.get('short_below', 4.5)}"))
        rows.append(("LONG when score >", f"{thresholds.get('hold_below', 5.5)}"))

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
    rows.append(("Slippage", f"{bt_cfg.get('slippage_pct', 0.001) * 100:.2f}%"))
    rows.append(("Warmup Bars", f"{bt_cfg.get('warmup_bars', 200)}"))

    df = pd.DataFrame(rows, columns=["Parameter", "Value"])
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    params = render_sidebar()

    if not params["ticker"]:
        st.info("Enter a ticker symbol in the sidebar to begin.")
        return

    # Validate day_trading + interval
    if params["objective"] == "day_trading" and not is_intraday(params["interval"]):
        st.error(
            "The **day_trading** objective requires an intraday interval "
            "(1m, 5m, 15m, 30m, 1h). Please change the interval in the sidebar."
        )
        return

    # ── Run analysis ──────────────────────────────────────────────────────
    try:
        result, cfg = load_analysis(
            ticker=params["ticker"],
            period=params["period"],
            interval=params["interval"],
            start=params["start"],
            end=params["end"],
            objective=params["objective"],
        )
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return

    # ── Header ────────────────────────────────────────────────────────────
    render_header(result, cfg)
    st.markdown("---")

    # ── Compute score timeseries (expensive — adjust step for performance) ─
    n_bars = len(result.df)
    step = max(1, n_bars // 150)  # aim for ~150 data points

    try:
        score_df = compute_score_timeseries(
            ticker=params["ticker"],
            period=params["period"],
            interval=params["interval"],
            start=params["start"],
            end=params["end"],
            objective=params["objective"],
            step=step,
        )
    except Exception:
        score_df = None

    # ── Price Chart ───────────────────────────────────────────────────────
    st.subheader("Price Chart")
    price_fig = create_price_chart(result, score_df)
    st.plotly_chart(price_fig, use_container_width=True)

    # ── Indicator & Pattern Tables ────────────────────────────────────────
    col_ind, col_pat = st.columns(2)
    with col_ind:
        st.subheader("Indicator Breakdown")
        render_indicator_table(result, cfg)
    with col_pat:
        st.subheader("Pattern Signals")
        render_pattern_table(result, cfg)

    # Score legend
    st.markdown(
        f'<div style="text-align:center;font-size:0.85rem;padding:4px 0 8px 0;color:#aaa;">'
        f'Score legend: '
        f'<span style="color:{COLOR_BEARISH};font-weight:600;">0 – 3.5 Bearish</span>'
        f'&nbsp;&nbsp;'
        f'<span style="color:{COLOR_NEUTRAL};font-weight:600;">3.5 – 6.5 Neutral</span>'
        f'&nbsp;&nbsp;'
        f'<span style="color:{COLOR_BULLISH};font-weight:600;">6.5 – 10 Bullish</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Score Distribution ────────────────────────────────────────────────
    if score_df is not None and not score_df.empty:
        st.subheader("Score Distribution")
        hist_fig = create_score_histogram(score_df)
        st.plotly_chart(hist_fig, use_container_width=True)

        # Score stats
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        ind_scores = score_df["indicator_score"]
        col_s1.metric("Ind. Mean", f"{ind_scores.mean():.2f}")
        col_s2.metric("Ind. Std", f"{ind_scores.std():.2f}")
        col_s3.metric("Ind. Min", f"{ind_scores.min():.2f}")
        col_s4.metric("Ind. Max", f"{ind_scores.max():.2f}")

    # ── Backtest ──────────────────────────────────────────────────────────
    if params["run_backtest"]:
        st.markdown("---")
        st.header("Backtest Results")

        try:
            bt_result, trading_mode, assessment, bt_cfg = load_backtest(
                ticker=params["ticker"],
                period=params["period"],
                interval=params["interval"],
                start=params["start"],
                end=params["end"],
                objective=params["objective"],
                trading_mode_str=params["trading_mode"],
            )
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            return

        # Suitability
        if assessment is not None:
            st.subheader("Suitability Assessment")
            render_suitability(assessment, params["ticker"])

        # Trading mode badge
        mode_label = trading_mode.value.replace("_", " ").upper()
        st.info(f"Trading mode: **{mode_label}**")

        # Metrics
        st.subheader("Performance Summary")
        render_backtest_metrics(bt_result)

        # Equity curve
        st.subheader("Equity Curve")
        eq_fig = create_equity_chart(bt_result, result.df)
        st.plotly_chart(eq_fig, use_container_width=True)

        # Strategy config
        with st.expander("Strategy Configuration"):
            render_strategy_config(bt_cfg)

        # Trade log
        with st.expander(f"Trade Log ({bt_result.total_trades} trades)"):
            render_trade_log(bt_result)

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "For informational purposes only. Not financial advice. "
        "Do your own research. Backtest results are hypothetical and "
        "do not guarantee future performance."
    )


if __name__ == "__main__":
    main()
