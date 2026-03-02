"""Stock Analyzer — macOS Desktop Prototype.

A native PySide6 (Qt) desktop app that wraps the existing watchlist engine.
Provides: portfolio summary, open positions table, scan with signals &
trade recommendations (Confirm / Skip), and closed trades history.

Launch:  python -m desktop.app          (from the project root)
    or:  python desktop/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from functools import partial

from PySide6.QtCore import Qt, QThread, Signal as QSignal, QSize
from PySide6.QtGui import QAction, QColor, QFont, QPalette, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QStackedWidget,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QFrame,
    QGroupBox,
    QMessageBox,
    QStatusBar,
    QProgressBar,
    QSpinBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QSizePolicy,
    QScrollArea,
)

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so engine imports work
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import Config
from data.yahoo import YahooFinanceProvider
from engine.watchlist import (
    WatchlistMonitor,
    WatchlistSignal,
    WatchlistState,
    WatchlistPosition,
    WatchlistClosedTrade,
    TradeRecommendation,
)
from engine.strategy import Signal


# ---------------------------------------------------------------------------
# Colours & styling helpers
# ---------------------------------------------------------------------------

_GREEN = "#2ecc71"
_RED = "#e74c3c"
_ORANGE = "#f39c12"
_BLUE = "#3498db"
_DARK_BG = "#1e1e2e"
_CARD_BG = "#2a2a3c"
_TEXT = "#cdd6f4"
_TEXT_DIM = "#6c7086"

STYLESHEET = f"""
QMainWindow {{
    background-color: {_DARK_BG};
}}
QWidget {{
    color: {_TEXT};
    font-family: -apple-system, "SF Pro Text", "Helvetica Neue";
    font-size: 13px;
}}
QListWidget {{
    background-color: {_CARD_BG};
    border: none;
    outline: none;
    padding: 8px 4px;
    font-size: 14px;
}}
QListWidget::item {{
    padding: 10px 16px;
    border-radius: 6px;
    margin: 2px 4px;
}}
QListWidget::item:selected {{
    background-color: {_BLUE};
    color: white;
}}
QListWidget::item:hover:!selected {{
    background-color: #3a3a4c;
}}
QGroupBox {{
    background-color: {_CARD_BG};
    border: 1px solid #3a3a4c;
    border-radius: 8px;
    margin-top: 14px;
    padding: 16px 12px 12px 12px;
    font-weight: bold;
    font-size: 13px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}}
QTableWidget {{
    background-color: {_CARD_BG};
    border: 1px solid #3a3a4c;
    border-radius: 6px;
    gridline-color: #3a3a4c;
    selection-background-color: {_BLUE};
}}
QTableWidget::item {{
    padding: 4px 8px;
}}
QHeaderView::section {{
    background-color: #3a3a4c;
    color: {_TEXT};
    padding: 6px 8px;
    border: none;
    font-weight: bold;
    font-size: 12px;
}}
QPushButton {{
    background-color: #3a3a4c;
    border: 1px solid #4a4a5c;
    border-radius: 6px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: #4a4a5c;
}}
QPushButton:pressed {{
    background-color: #5a5a6c;
}}
QPushButton[primary="true"] {{
    background-color: {_BLUE};
    border: none;
    color: white;
    font-weight: bold;
}}
QPushButton[primary="true"]:hover {{
    background-color: #2980b9;
}}
QPushButton[danger="true"] {{
    background-color: {_RED};
    border: none;
    color: white;
}}
QPushButton[danger="true"]:hover {{
    background-color: #c0392b;
}}
QPushButton[success="true"] {{
    background-color: {_GREEN};
    border: none;
    color: white;
}}
QPushButton[success="true"]:hover {{
    background-color: #27ae60;
}}
QPushButton:disabled {{
    background-color: #2a2a3c;
    color: #5a5a6c;
    border: 1px solid #3a3a4c;
}}
QStatusBar {{
    background-color: {_CARD_BG};
    color: {_TEXT_DIM};
    font-size: 12px;
}}
QProgressBar {{
    background-color: #3a3a4c;
    border: none;
    border-radius: 4px;
    text-align: center;
    height: 8px;
}}
QProgressBar::chunk {{
    background-color: {_BLUE};
    border-radius: 4px;
}}
QScrollArea {{
    border: none;
    background-color: transparent;
}}
"""


# ---------------------------------------------------------------------------
# Worker thread for running scans off the main thread
# ---------------------------------------------------------------------------

class ScanWorker(QThread):
    """Runs WatchlistMonitor.scan() in a background thread."""

    finished = QSignal(list, list)  # (signals, recommendations)
    error = QSignal(str)
    progress = QSignal(str)

    def __init__(self, monitor: WatchlistMonitor, parent=None):
        super().__init__(parent)
        self._monitor = monitor

    def run(self):
        try:
            self.progress.emit("Scanning tickers...")
            signals = self._monitor.scan()
            self.progress.emit("Generating recommendations...")
            recs = self._monitor.generate_recommendations(signals)
            self._monitor.save_state()
            self.finished.emit(signals, recs)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Metric card widget
# ---------------------------------------------------------------------------

class MetricCard(QFrame):
    """A small card showing a label + value (and optional delta)."""

    def __init__(
        self,
        label: str,
        value: str,
        delta: str = "",
        delta_color: str = _TEXT_DIM,
        parent=None,
    ):
        super().__init__(parent)
        self.setStyleSheet(
            f"background-color: {_CARD_BG}; border: 1px solid #3a3a4c; "
            f"border-radius: 8px; padding: 12px;"
        )
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(12, 10, 12, 10)

        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 11px; font-weight: 500; border: none;")
        layout.addWidget(lbl)

        val = QLabel(value)
        val.setStyleSheet("font-size: 22px; font-weight: bold; border: none;")
        layout.addWidget(val)

        if delta:
            d = QLabel(delta)
            d.setStyleSheet(f"color: {delta_color}; font-size: 11px; border: none;")
            layout.addWidget(d)

        self._value_label = val
        self._delta_label = None
        if delta:
            self._delta_label = layout.itemAt(2).widget()

    def update_value(self, value: str, delta: str = "", delta_color: str = _TEXT_DIM):
        self._value_label.setText(value)
        if self._delta_label:
            self._delta_label.setText(delta)
            self._delta_label.setStyleSheet(
                f"color: {delta_color}; font-size: 11px; border: none;"
            )


# ---------------------------------------------------------------------------
# Recommendation card widget
# ---------------------------------------------------------------------------

class RecommendationCard(QFrame):
    """A card for a single trade recommendation with Confirm / Edit / Skip."""

    confirmed = QSignal(object)   # emits the TradeRecommendation
    skipped = QSignal(object)

    def __init__(self, rec: TradeRecommendation, parent=None):
        super().__init__(parent)
        self.rec = rec
        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet(
            f"background-color: {_CARD_BG}; border: 1px solid #3a3a4c; "
            f"border-radius: 8px; padding: 8px;"
        )
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(14, 12, 14, 12)

        rec = self.rec
        sig = rec.signal
        is_buy = rec.side == "buy"

        # Header
        icon = "\U0001f7e2" if is_buy else "\U0001f534"
        action = "BUY" if is_buy else "SELL"
        header_text = (
            f"{icon}  {action} {rec.ticker}  —  "
            f"{rec.recommended_quantity:.0f} shares "
            f"@ ${rec.recommended_price:,.2f}"
        )
        if is_buy:
            header_text += f"  (${rec.estimated_cost:,.0f})"
        elif rec.estimated_pnl_pct != 0:
            sign = "+" if rec.estimated_pnl_pct >= 0 else ""
            header_text += f"  ({sign}{rec.estimated_pnl_pct:.1f}%)"

        header = QLabel(header_text)
        header.setStyleSheet("font-size: 15px; font-weight: bold; border: none;")
        layout.addWidget(header)

        # Metrics row
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(12)
        if is_buy:
            metrics_layout.addWidget(self._mini_metric("Cost", f"${rec.estimated_cost:,.0f}"))
            metrics_layout.addWidget(self._mini_metric("Cash Before", f"${rec.cash_before:,.0f}"))
            after_color = _RED if rec.cash_after < 0 else _GREEN
            metrics_layout.addWidget(self._mini_metric(
                "Cash After", f"${rec.cash_after:,.0f}", after_color,
            ))
        else:
            metrics_layout.addWidget(self._mini_metric("Proceeds", f"${rec.estimated_cost:,.0f}"))
            metrics_layout.addWidget(self._mini_metric(
                "Entry", f"${rec.entry_price:,.2f}",
            ))
            pnl_color = _GREEN if rec.estimated_pnl_pct >= 0 else _RED
            metrics_layout.addWidget(self._mini_metric(
                "P&L", f"{rec.estimated_pnl_pct:+.1f}%", pnl_color,
            ))

        if sig:
            metrics_layout.addWidget(self._mini_metric("Regime", sig.regime.replace("_", " ").title()))
            metrics_layout.addWidget(self._mini_metric("Eff. Score", f"{sig.effective_score:.1f}"))

        layout.addLayout(metrics_layout)

        # Sizing info
        sizing_parts = [f"Sizing: {rec.sizing_mode.replace('_', ' ')}"]
        if rec.regime_adjustment:
            sizing_parts.append(f"regime adj {rec.regime_adjustment}")
        if rec.dca_multiplier > 1.0:
            sizing_parts.append(f"DCA {rec.dca_multiplier:.1f}x ({rec.dca_tier.replace('_', ' ')})")
        sizing_label = QLabel(" · ".join(sizing_parts))
        sizing_label.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 11px; border: none;")
        layout.addWidget(sizing_label)

        # Warning for insufficient cash
        if rec.insufficient_cash:
            warn = QLabel(
                f"Warning: This trade costs ${rec.estimated_cost:,.0f} but you only "
                f"have ${rec.cash_before:,.0f} in cash."
            )
            warn.setStyleSheet(f"color: {_ORANGE}; font-size: 12px; font-weight: 500; border: none;")
            warn.setWordWrap(True)
            layout.addWidget(warn)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        if rec.insufficient_cash and is_buy:
            # Disabled confirm + "Confirm Anyway"
            confirm_btn = QPushButton("Confirm")
            confirm_btn.setEnabled(False)
            btn_layout.addWidget(confirm_btn)

            force_btn = QPushButton("Confirm Anyway")
            force_btn.setProperty("danger", True)
            force_btn.style().unpolish(force_btn)
            force_btn.style().polish(force_btn)
            force_btn.clicked.connect(lambda: self.confirmed.emit(self.rec))
            btn_layout.addWidget(force_btn)
        else:
            confirm_btn = QPushButton(f"Confirm {action}")
            confirm_btn.setProperty("primary", True)
            confirm_btn.style().unpolish(confirm_btn)
            confirm_btn.style().polish(confirm_btn)
            confirm_btn.clicked.connect(lambda: self.confirmed.emit(self.rec))
            btn_layout.addWidget(confirm_btn)

        skip_btn = QPushButton("Skip")
        skip_btn.clicked.connect(lambda: self.skipped.emit(self.rec))
        btn_layout.addWidget(skip_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    @staticmethod
    def _mini_metric(label: str, value: str, color: str = _TEXT) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("border: none;")
        layout = QVBoxLayout(frame)
        layout.setSpacing(1)
        layout.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 10px; border: none;")
        val = QLabel(value)
        val.setStyleSheet(f"color: {color}; font-size: 13px; font-weight: bold; border: none;")
        layout.addWidget(lbl)
        layout.addWidget(val)
        return frame


# ---------------------------------------------------------------------------
# Portfolio page
# ---------------------------------------------------------------------------

class PortfolioPage(QWidget):
    """Main portfolio view: summary metrics, open positions, recommendations,
    and closed trades."""

    def __init__(self, monitor: WatchlistMonitor, parent=None):
        super().__init__(parent)
        self._monitor = monitor
        self._signals: list[WatchlistSignal] = []
        self._recs: list[TradeRecommendation] = []
        self._worker: ScanWorker | None = None
        self._build_ui()
        self._refresh_portfolio()

    def _build_ui(self):
        # Root layout with scroll area
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root.addWidget(scroll)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setSpacing(16)
        self._layout.setContentsMargins(20, 16, 20, 16)
        scroll.setWidget(container)

        # ── Header row ────────────────────────────────────────────────
        header_row = QHBoxLayout()
        title = QLabel("Watchlist Portfolio")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        header_row.addWidget(title)
        header_row.addStretch()

        self._scan_btn = QPushButton("Scan Watchlist")
        self._scan_btn.setProperty("primary", True)
        self._scan_btn.setFixedHeight(36)
        self._scan_btn.clicked.connect(self._start_scan)
        header_row.addWidget(self._scan_btn)

        self._layout.addLayout(header_row)

        # ── Summary metrics ───────────────────────────────────────────
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(12)
        self._cash_card = MetricCard("Cash Balance", "$0")
        self._equity_card = MetricCard("Total Equity", "$0")
        self._positions_card = MetricCard("Open Positions", "0")
        self._last_scan_card = MetricCard("Last Scan", "Never")
        metrics_row.addWidget(self._cash_card)
        metrics_row.addWidget(self._equity_card)
        metrics_row.addWidget(self._positions_card)
        metrics_row.addWidget(self._last_scan_card)
        self._layout.addLayout(metrics_row)

        # ── Scan progress ─────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setFixedHeight(4)
        self._progress.hide()
        self._layout.addWidget(self._progress)

        # ── Recommendations area ──────────────────────────────────────
        self._recs_group = QGroupBox("Trade Recommendations")
        self._recs_layout = QVBoxLayout(self._recs_group)
        self._recs_layout.setSpacing(10)
        self._recs_placeholder = QLabel("Run a scan to see recommendations.")
        self._recs_placeholder.setStyleSheet(f"color: {_TEXT_DIM}; border: none;")
        self._recs_layout.addWidget(self._recs_placeholder)
        self._recs_group.hide()
        self._layout.addWidget(self._recs_group)

        # ── Signals table ─────────────────────────────────────────────
        self._signals_group = QGroupBox("Scan Signals")
        sig_layout = QVBoxLayout(self._signals_group)
        self._signals_table = QTableWidget()
        self._signals_table.setColumnCount(8)
        self._signals_table.setHorizontalHeaderLabels([
            "Ticker", "Signal", "Action", "Price",
            "Ind. Score", "Pat. Score", "Eff. Score", "Regime",
        ])
        self._signals_table.horizontalHeader().setStretchLastSection(True)
        self._signals_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents,
        )
        self._signals_table.verticalHeader().setVisible(False)
        self._signals_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._signals_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._signals_table.setAlternatingRowColors(True)
        sig_layout.addWidget(self._signals_table)
        self._signals_group.hide()
        self._layout.addWidget(self._signals_group)

        # ── Open positions table ──────────────────────────────────────
        self._positions_group = QGroupBox("Open Positions")
        pos_layout = QVBoxLayout(self._positions_group)
        self._positions_table = QTableWidget()
        self._positions_table.setColumnCount(7)
        self._positions_table.setHorizontalHeaderLabels([
            "Ticker", "Side", "Entry Date", "Entry Price",
            "Quantity", "Market Value", "P&L %",
        ])
        self._positions_table.horizontalHeader().setStretchLastSection(True)
        self._positions_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents,
        )
        self._positions_table.verticalHeader().setVisible(False)
        self._positions_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._positions_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._positions_table.setAlternatingRowColors(True)
        pos_layout.addWidget(self._positions_table)
        self._layout.addWidget(self._positions_group)

        # ── Closed trades table ───────────────────────────────────────
        self._closed_group = QGroupBox("Closed Trades")
        closed_layout = QVBoxLayout(self._closed_group)
        self._closed_table = QTableWidget()
        self._closed_table.setColumnCount(8)
        self._closed_table.setHorizontalHeaderLabels([
            "Ticker", "Side", "Entry", "Exit",
            "Entry Price", "Exit Price", "Quantity", "P&L %",
        ])
        self._closed_table.horizontalHeader().setStretchLastSection(True)
        self._closed_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents,
        )
        self._closed_table.verticalHeader().setVisible(False)
        self._closed_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._closed_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._closed_table.setAlternatingRowColors(True)
        closed_layout.addWidget(self._closed_table)
        self._layout.addWidget(self._closed_group)

        # Spacer
        self._layout.addStretch()

    # ------------------------------------------------------------------
    # Data refresh
    # ------------------------------------------------------------------

    def _refresh_portfolio(self):
        """Refresh all widgets from current monitor state."""
        state = self._monitor.state
        cash = state.cash_balance if state.cash_balance is not None else 0.0

        # Build live prices from last scan if available
        live_prices = {
            s.ticker: s.current_price
            for s in self._signals
            if s.current_price > 0 and not s.error
        }

        # Compute equity
        market_value = 0.0
        for ticker, pos in state.positions.items():
            price = live_prices.get(ticker, pos.entry_price)
            market_value += pos.quantity * price
        equity = cash + market_value

        # Update metric cards
        self._cash_card.update_value(f"${cash:,.0f}")
        self._equity_card.update_value(
            f"${equity:,.0f}",
            delta=f"Positions: ${market_value:,.0f}" if market_value else "",
        )
        self._positions_card.update_value(str(len(state.positions)))
        last = state.last_updated or "Never"
        self._last_scan_card.update_value(last)

        # Positions table
        self._populate_positions_table(state, live_prices)
        # Closed trades
        self._populate_closed_table(state)

    def _populate_positions_table(
        self, state: WatchlistState, prices: dict[str, float],
    ):
        positions = list(state.positions.values())
        self._positions_table.setRowCount(len(positions))
        for i, pos in enumerate(positions):
            price = prices.get(pos.ticker, pos.entry_price)
            pnl = pos.unrealized_pnl_pct(price)
            mkt_val = pos.quantity * price

            self._positions_table.setItem(i, 0, QTableWidgetItem(pos.ticker))
            self._positions_table.setItem(i, 1, QTableWidgetItem(pos.side.upper()))
            self._positions_table.setItem(i, 2, QTableWidgetItem(pos.entry_date))
            self._positions_table.setItem(i, 3, self._money_item(pos.entry_price))
            self._positions_table.setItem(i, 4, self._num_item(pos.quantity))
            self._positions_table.setItem(i, 5, self._money_item(mkt_val))
            self._positions_table.setItem(i, 6, self._pnl_item(pnl))

    def _populate_closed_table(self, state: WatchlistState):
        trades = list(reversed(state.closed_trades))  # newest first
        self._closed_table.setRowCount(len(trades))
        for i, ct in enumerate(trades):
            self._closed_table.setItem(i, 0, QTableWidgetItem(ct.ticker))
            self._closed_table.setItem(i, 1, QTableWidgetItem(ct.side.upper()))
            self._closed_table.setItem(i, 2, QTableWidgetItem(ct.entry_date))
            self._closed_table.setItem(i, 3, QTableWidgetItem(ct.exit_date))
            self._closed_table.setItem(i, 4, self._money_item(ct.entry_price))
            self._closed_table.setItem(i, 5, self._money_item(ct.exit_price))
            self._closed_table.setItem(i, 6, self._num_item(ct.quantity))
            self._closed_table.setItem(i, 7, self._pnl_item(ct.pnl_pct))

    def _populate_signals_table(self, signals: list[WatchlistSignal]):
        self._signals_table.setRowCount(len(signals))
        for i, sig in enumerate(signals):
            self._signals_table.setItem(i, 0, QTableWidgetItem(sig.ticker))

            sig_item = QTableWidgetItem(sig.signal.name)
            if sig.signal == Signal.BUY:
                sig_item.setForeground(QColor(_GREEN))
            elif sig.signal == Signal.SELL:
                sig_item.setForeground(QColor(_RED))
            self._signals_table.setItem(i, 1, sig_item)

            action_item = QTableWidgetItem(sig.action)
            if "OPEN LONG" in sig.action or "CLOSE SHORT" in sig.action:
                action_item.setForeground(QColor(_GREEN))
            elif "CLOSE LONG" in sig.action or "OPEN SHORT" in sig.action:
                action_item.setForeground(QColor(_RED))
            self._signals_table.setItem(i, 2, action_item)

            self._signals_table.setItem(i, 3, self._money_item(sig.current_price))
            self._signals_table.setItem(i, 4, self._score_item(sig.indicator_score))
            self._signals_table.setItem(i, 5, self._score_item(sig.pattern_score))
            self._signals_table.setItem(i, 6, self._score_item(sig.effective_score))
            self._signals_table.setItem(
                i, 7,
                QTableWidgetItem(sig.regime.replace("_", " ").title()),
            )

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------

    def _start_scan(self):
        if self._worker is not None and self._worker.isRunning():
            return
        self._scan_btn.setEnabled(False)
        self._scan_btn.setText("Scanning...")
        self._progress.show()

        self._worker = ScanWorker(self._monitor, self)
        self._worker.finished.connect(self._on_scan_finished)
        self._worker.error.connect(self._on_scan_error)
        self._worker.start()

    def _on_scan_finished(self, signals: list, recs: list):
        self._progress.hide()
        self._scan_btn.setEnabled(True)
        self._scan_btn.setText("Scan Watchlist")
        self._signals = signals
        self._recs = recs

        # Show signals
        self._signals_group.show()
        self._populate_signals_table(signals)

        # Show recommendations
        self._show_recommendations(recs)

        # Refresh portfolio
        self._refresh_portfolio()

        self.window().statusBar().showMessage(
            f"Scan complete: {len(signals)} tickers, "
            f"{len(recs)} recommendations",
            5000,
        )

    def _on_scan_error(self, error_msg: str):
        self._progress.hide()
        self._scan_btn.setEnabled(True)
        self._scan_btn.setText("Scan Watchlist")
        QMessageBox.warning(self, "Scan Error", f"Scan failed:\n{error_msg}")

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _show_recommendations(self, recs: list[TradeRecommendation]):
        # Clear old cards
        while self._recs_layout.count():
            child = self._recs_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if not recs:
            placeholder = QLabel("No actionable recommendations from this scan.")
            placeholder.setStyleSheet(f"color: {_TEXT_DIM}; border: none;")
            self._recs_layout.addWidget(placeholder)
            self._recs_group.show()
            return

        for rec in recs:
            card = RecommendationCard(rec)
            card.confirmed.connect(self._on_confirm_rec)
            card.skipped.connect(self._on_skip_rec)
            self._recs_layout.addWidget(card)

        self._recs_group.show()

    def _on_confirm_rec(self, rec: TradeRecommendation):
        """Execute a confirmed trade recommendation."""
        try:
            state = self._monitor.state
            sig = rec.signal
            if sig is None:
                return

            ticker = rec.ticker.upper()
            exec_price = rec.recommended_price
            exec_qty = rec.recommended_quantity

            if rec.side == "buy" and ticker not in state.positions:
                cost = exec_qty * exec_price
                pos = WatchlistPosition(
                    ticker=ticker,
                    side="long",
                    entry_date=datetime.now().strftime("%Y-%m-%d"),
                    entry_price=exec_price,
                    quantity=exec_qty,
                    entry_reason=sig.signal_notes,
                )
                state.positions[ticker] = pos
                if state.cash_balance is not None:
                    state.cash_balance -= cost

            elif rec.side == "sell" and ticker in state.positions:
                pos = state.positions.pop(ticker)
                pnl_pct = pos.unrealized_pnl_pct(exec_price)
                closed = WatchlistClosedTrade(
                    ticker=ticker,
                    side=pos.side,
                    entry_date=pos.entry_date,
                    exit_date=datetime.now().strftime("%Y-%m-%d"),
                    entry_price=pos.entry_price,
                    exit_price=exec_price,
                    quantity=exec_qty,
                    pnl_pct=pnl_pct,
                    exit_reason=sig.signal_notes,
                )
                state.closed_trades.append(closed)
                if state.cash_balance is not None:
                    state.cash_balance += exec_qty * exec_price

                # Update strategy state
                ss = state.strategy_state.get(ticker, {})
                losses = ss.get("consecutive_losses", 0)
                if pnl_pct < 0:
                    ss["consecutive_losses"] = losses + 1
                else:
                    ss["consecutive_losses"] = 0
                ss["bars_since_exit"] = 0
                state.strategy_state[ticker] = ss

            self._monitor.save_state()
            self._refresh_portfolio()
            self._remove_rec_card(rec)

            action = "Bought" if rec.side == "buy" else "Sold"
            self.window().statusBar().showMessage(
                f"{action} {rec.ticker}: {rec.recommended_quantity:.0f} shares "
                f"@ ${rec.recommended_price:,.2f}",
                5000,
            )

        except Exception as exc:
            QMessageBox.warning(
                self, "Execution Error", f"Failed to execute trade:\n{exc}",
            )

    def _on_skip_rec(self, rec: TradeRecommendation):
        self._remove_rec_card(rec)
        self.window().statusBar().showMessage(
            f"Skipped {rec.ticker} recommendation.", 3000,
        )

    def _remove_rec_card(self, rec: TradeRecommendation):
        for i in range(self._recs_layout.count()):
            child = self._recs_layout.itemAt(i)
            if child and child.widget() and isinstance(child.widget(), RecommendationCard):
                if child.widget().rec is rec:
                    child.widget().deleteLater()
                    break

    # ------------------------------------------------------------------
    # Table item helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _money_item(value: float) -> QTableWidgetItem:
        item = QTableWidgetItem(f"${value:,.2f}")
        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return item

    @staticmethod
    def _num_item(value: float) -> QTableWidgetItem:
        item = QTableWidgetItem(f"{value:,.0f}")
        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return item

    @staticmethod
    def _pnl_item(pnl: float) -> QTableWidgetItem:
        item = QTableWidgetItem(f"{pnl:+.1f}%")
        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        if pnl > 0:
            item.setForeground(QColor(_GREEN))
        elif pnl < 0:
            item.setForeground(QColor(_RED))
        return item

    @staticmethod
    def _score_item(score: float) -> QTableWidgetItem:
        item = QTableWidgetItem(f"{score:.1f}")
        item.setTextAlignment(Qt.AlignCenter)
        if score >= 7:
            item.setForeground(QColor(_GREEN))
        elif score <= 3:
            item.setForeground(QColor(_RED))
        return item


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Top-level window with sidebar navigation and stacked pages."""

    def __init__(self, monitor: WatchlistMonitor):
        super().__init__()
        self._monitor = monitor

        self.setWindowTitle("Stock Analyzer")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)

        # ── Central widget ────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Sidebar ───────────────────────────────────────────────────
        self._sidebar = QListWidget()
        self._sidebar.setFixedWidth(180)
        self._sidebar.setIconSize(QSize(20, 20))

        pages = ["Portfolio", "Scanner", "Settings"]
        for name in pages:
            item = QListWidgetItem(name)
            item.setSizeHint(QSize(170, 40))
            self._sidebar.addItem(item)

        self._sidebar.setCurrentRow(0)
        self._sidebar.currentRowChanged.connect(self._switch_page)
        main_layout.addWidget(self._sidebar)

        # Subtle separator
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"color: #3a3a4c;")
        sep.setFixedWidth(1)
        main_layout.addWidget(sep)

        # ── Page stack ────────────────────────────────────────────────
        self._stack = QStackedWidget()

        self._portfolio_page = PortfolioPage(monitor)
        self._stack.addWidget(self._portfolio_page)

        # Placeholder pages
        scanner_placeholder = QLabel("Scanner — coming soon")
        scanner_placeholder.setAlignment(Qt.AlignCenter)
        scanner_placeholder.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 16px;")
        self._stack.addWidget(scanner_placeholder)

        settings_placeholder = QLabel("Settings — coming soon")
        settings_placeholder.setAlignment(Qt.AlignCenter)
        settings_placeholder.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 16px;")
        self._stack.addWidget(settings_placeholder)

        main_layout.addWidget(self._stack, 1)

        # ── Status bar ────────────────────────────────────────────────
        self.statusBar().showMessage("Ready")

    def _switch_page(self, index: int):
        self._stack.setCurrentIndex(index)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Load config
    config_path = str(_PROJECT_ROOT / "config.yaml")
    cfg = Config.load(config_path)

    # Create data provider and initialise monitor
    provider = YahooFinanceProvider()
    monitor = WatchlistMonitor(provider, cfg)

    # Create Qt app
    app = QApplication(sys.argv)
    app.setApplicationName("Stock Analyzer")
    app.setApplicationDisplayName("Stock Analyzer")
    app.setStyleSheet(STYLESHEET)

    window = MainWindow(monitor)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
