"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { createChart, ColorType, LineData, Time } from "lightweight-charts";
import HelpTip from "@/components/HelpTip";
import {
  runBacktest,
  runWalkForward,
  runAutoTune,
  listStrategies,
  createStrategy,
  updateStrategy,
  deleteStrategy,
  exportStrategy,
  importStrategy,
  type BacktestResult,
  type BacktestRequest,
  type BacktestTrade,
  type EquityPoint,
  type WalkForwardResult,
  type WalkForwardRequest,
  type WalkForwardWindow,
  type AutoTuneResult,
  type AutoTuneRequest,
  type SensitivityEntry,
  type AutoTuneTrial,
  type StrategyItem,
  type StrategyCreateRequest,
  type StrategyExport,
} from "@/lib/api";
import {
  HELP_BACKTEST,
  HELP_INITIAL_CASH,
  HELP_COMMISSION,
  HELP_SLIPPAGE,
  HELP_STOP_LOSS,
  HELP_TAKE_PROFIT,
  HELP_BT_TOTAL_RETURN,
  HELP_ANNUALIZED_RETURN,
  HELP_MAX_DRAWDOWN,
  HELP_SHARPE_RATIO,
  HELP_WIN_RATE,
  HELP_PROFIT_FACTOR,
  HELP_TOTAL_TRADES,
  HELP_AVG_TRADE_PNL,
  HELP_BEST_TRADE,
  HELP_WORST_TRADE,
  HELP_AVG_BARS_HELD,
  HELP_EQUITY_CURVE,
  HELP_TRADE_LOG,
  HELP_WALK_FORWARD,
  HELP_TRAIN_YEARS,
  HELP_TEST_YEARS,
  HELP_STABILITY_SCORE,
  HELP_WORST_WINDOW,
  HELP_RETURN_STD_DEV,
  HELP_AUTO_TUNER,
  HELP_TUNER_OBJECTIVE,
  HELP_N_TRIALS,
  HELP_IMPROVEMENT,
  HELP_BASELINE,
  HELP_BEST_PARAMS,
  HELP_SENSITIVITY,
  HELP_BEAT_BUY_HOLD,
  HELP_MAX_RETURN,
  HELP_MAX_RISK_ADJUSTED,
  HELP_MIN_DRAWDOWN,
  HELP_BALANCED,
  HELP_STRATEGY_LIBRARY,
  HELP_STRATEGY_PRESET,
  HELP_STRATEGY_VERSION,
  HELP_STRATEGY_ACTIVE,
  HELP_STRATEGY_PARAMS,
  HELP_STRATEGY_COMPARE,
  HELP_STRATEGY_EXPORT,
  HELP_STRATEGY_IMPORT,
} from "@/lib/helpText";

// ---------------------------------------------------------------------------
// Equity curve chart (lightweight-charts line series)
// ---------------------------------------------------------------------------

function EquityCurveChart({
  data,
  warmupBars,
  height = 350,
}: {
  data: EquityPoint[];
  warmupBars: number;
  height?: number;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current || data.length === 0) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#111827" },
        textColor: "#9CA3AF",
      },
      grid: {
        vertLines: { color: "#1F2937" },
        horzLines: { color: "#1F2937" },
      },
      width: containerRef.current.clientWidth,
      height,
      handleScroll: { mouseWheel: false },
      handleScale: { mouseWheel: false },
      rightPriceScale: {
        borderColor: "#374151",
      },
      timeScale: {
        borderColor: "#374151",
      },
    });

    // Post-warmup equity data
    const lineData: LineData[] = data.slice(warmupBars).map((pt) => ({
      time: pt.date as Time,
      value: pt.equity,
    }));

    const series = chart.addLineSeries({
      color: "#10B981",
      lineWidth: 2,
      crosshairMarkerVisible: true,
      priceFormat: {
        type: "custom",
        formatter: (price: number) => `$${price.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
      },
    });
    series.setData(lineData);
    chart.timeScale().fitContent();

    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [data, warmupBars, height]);

  return <div ref={containerRef} />;
}

// ---------------------------------------------------------------------------
// Metric card component
// ---------------------------------------------------------------------------

function MetricCard({
  label,
  value,
  helpText,
  color,
}: {
  label: string;
  value: string;
  helpText?: string;
  color?: "green" | "red" | "yellow" | "default";
}) {
  const colorClass =
    color === "green"
      ? "text-emerald-400"
      : color === "red"
        ? "text-red-400"
        : color === "yellow"
          ? "text-yellow-400"
          : "text-white";

  return (
    <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-3">
      <div className="text-xs text-gray-400 flex items-center gap-1 mb-1">
        {label}
        {helpText && <HelpTip text={helpText} />}
      </div>
      <div className={`text-lg font-semibold ${colorClass}`}>{value}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Trade log table
// ---------------------------------------------------------------------------

function TradeLogTable({ trades }: { trades: BacktestTrade[] }) {
  const [sortCol, setSortCol] = useState<string>("entry_date");
  const [sortAsc, setSortAsc] = useState(true);
  const [page, setPage] = useState(0);
  const perPage = 20;

  const sorted = [...trades].sort((a, b) => {
    const aVal = (a as unknown as Record<string, unknown>)[sortCol];
    const bVal = (b as unknown as Record<string, unknown>)[sortCol];
    if (typeof aVal === "number" && typeof bVal === "number") {
      return sortAsc ? aVal - bVal : bVal - aVal;
    }
    return sortAsc
      ? String(aVal).localeCompare(String(bVal))
      : String(bVal).localeCompare(String(aVal));
  });

  const paged = sorted.slice(page * perPage, (page + 1) * perPage);
  const totalPages = Math.ceil(trades.length / perPage);

  const handleSort = (col: string) => {
    if (sortCol === col) {
      setSortAsc(!sortAsc);
    } else {
      setSortCol(col);
      setSortAsc(true);
    }
  };

  const cols: { key: string; label: string; align?: string }[] = [
    { key: "entry_date", label: "Entry" },
    { key: "exit_date", label: "Exit" },
    { key: "side", label: "Side" },
    { key: "entry_price", label: "Entry $", align: "right" },
    { key: "exit_price", label: "Exit $", align: "right" },
    { key: "quantity", label: "Qty", align: "right" },
    { key: "pnl_pct", label: "P&L %", align: "right" },
    { key: "pnl", label: "P&L $", align: "right" },
    { key: "bars_held", label: "Bars", align: "right" },
    { key: "exit_reason", label: "Exit Reason" },
  ];

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-gray-700 text-gray-400">
              {cols.map((c) => (
                <th
                  key={c.key}
                  onClick={() => handleSort(c.key)}
                  className={`py-2 px-2 cursor-pointer hover:text-gray-200 select-none ${
                    c.align === "right" ? "text-right" : "text-left"
                  }`}
                >
                  {c.label}
                  {sortCol === c.key && (sortAsc ? " \u25B2" : " \u25BC")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paged.map((t, i) => (
              <tr
                key={i}
                className="border-b border-gray-800 hover:bg-gray-800/40"
              >
                <td className="py-1.5 px-2 text-gray-300">{t.entry_date}</td>
                <td className="py-1.5 px-2 text-gray-300">{t.exit_date}</td>
                <td className="py-1.5 px-2">
                  <span
                    className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                      t.side === "long"
                        ? "bg-emerald-900/50 text-emerald-400"
                        : "bg-red-900/50 text-red-400"
                    }`}
                  >
                    {t.side.toUpperCase()}
                  </span>
                </td>
                <td className="py-1.5 px-2 text-right text-gray-300">
                  ${t.entry_price.toFixed(2)}
                </td>
                <td className="py-1.5 px-2 text-right text-gray-300">
                  ${t.exit_price.toFixed(2)}
                </td>
                <td className="py-1.5 px-2 text-right text-gray-300">
                  {t.quantity}
                </td>
                <td
                  className={`py-1.5 px-2 text-right font-medium ${
                    t.pnl_pct >= 0 ? "text-emerald-400" : "text-red-400"
                  }`}
                >
                  {t.pnl_pct >= 0 ? "+" : ""}
                  {(t.pnl_pct * 100).toFixed(2)}%
                </td>
                <td
                  className={`py-1.5 px-2 text-right font-medium ${
                    t.pnl >= 0 ? "text-emerald-400" : "text-red-400"
                  }`}
                >
                  {t.pnl >= 0 ? "+" : ""}${t.pnl.toFixed(2)}
                </td>
                <td className="py-1.5 px-2 text-right text-gray-300">
                  {t.bars_held}
                </td>
                <td className="py-1.5 px-2 text-gray-400 capitalize">
                  {t.exit_reason.replace(/_/g, " ")}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-3 text-xs text-gray-400">
          <span>
            Showing {page * perPage + 1}-
            {Math.min((page + 1) * perPage, trades.length)} of {trades.length}{" "}
            trades
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => setPage(Math.max(0, page - 1))}
              disabled={page === 0}
              className="px-2 py-1 bg-gray-800 rounded disabled:opacity-30 hover:bg-gray-700"
            >
              Prev
            </button>
            <button
              onClick={() => setPage(Math.min(totalPages - 1, page + 1))}
              disabled={page >= totalPages - 1}
              className="px-2 py-1 bg-gray-800 rounded disabled:opacity-30 hover:bg-gray-700"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page component
// ---------------------------------------------------------------------------

export default function StrategyPage() {
  // Form state
  const [ticker, setTicker] = useState("AAPL");
  const [period, setPeriod] = useState("2y");
  const [initialCash, setInitialCash] = useState(100_000);
  const [commissionPct, setCommissionPct] = useState(0);
  const [slippagePct, setSlippagePct] = useState(0.1);
  const [stopLossPct, setStopLossPct] = useState(5);
  const [takeProfitPct, setTakeProfitPct] = useState(15);

  // Result state
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Active tab: "metrics" | "trades"
  const [activeTab, setActiveTab] = useState<"metrics" | "trades">("metrics");

  const handleRun = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const req: BacktestRequest = {
        ticker: ticker.trim().toUpperCase(),
        period,
        initial_cash: initialCash,
        commission_pct: commissionPct / 100,   // UI shows %, API wants decimal
        slippage_pct: slippagePct / 100,
        stop_loss_pct: stopLossPct / 100,
        take_profit_pct: takeProfitPct / 100,
      };
      const res = await runBacktest(req);
      setResult(res);
      setActiveTab("metrics");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Backtest failed");
    } finally {
      setLoading(false);
    }
  }, [ticker, period, initialCash, commissionPct, slippagePct, stopLossPct, takeProfitPct]);

  // Determine metric colors
  const metricColor = (val: number | null | undefined): "green" | "red" | "default" =>
    val == null ? "default" : val > 0 ? "green" : val < 0 ? "red" : "default";

  const fmtPct = (v: number | null | undefined) =>
    v == null ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
  const fmtNum = (v: number | null | undefined, decimals = 2) =>
    v == null ? "—" : v.toFixed(decimals);
  const fmtMoney = (v: number | null | undefined) =>
    v == null ? "—" : `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-semibold text-white flex items-center gap-2">
          Strategy Lab
          <HelpTip text={HELP_BACKTEST} />
        </h2>
        <p className="text-sm text-gray-500 mt-1">
          Backtest the score-based strategy on any ticker with customizable
          parameters.
        </p>
      </div>

      {/* Backtest form */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h3 className="text-sm font-medium text-gray-300 mb-4">
          Backtest Configuration
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
          {/* Ticker */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">Ticker</label>
            <input
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm
                         focus:outline-none focus:border-blue-500"
              placeholder="AAPL"
            />
          </div>

          {/* Period */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">
              Period
            </label>
            <select
              value={period}
              onChange={(e) => setPeriod(e.target.value)}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm
                         focus:outline-none focus:border-blue-500"
            >
              <option value="6mo">6 Months</option>
              <option value="1y">1 Year</option>
              <option value="2y">2 Years</option>
              <option value="5y">5 Years</option>
            </select>
          </div>

          {/* Initial Cash */}
          <div>
            <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
              Starting Capital <HelpTip text={HELP_INITIAL_CASH} />
            </label>
            <input
              type="number"
              value={initialCash}
              onChange={(e) => setInitialCash(Number(e.target.value))}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm
                         focus:outline-none focus:border-blue-500"
              min={1000}
              step={10000}
            />
          </div>

          {/* Commission */}
          <div>
            <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
              Commission % <HelpTip text={HELP_COMMISSION} />
            </label>
            <input
              type="number"
              value={commissionPct}
              onChange={(e) => setCommissionPct(Number(e.target.value))}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm
                         focus:outline-none focus:border-blue-500"
              min={0}
              max={5}
              step={0.01}
            />
          </div>

          {/* Slippage */}
          <div>
            <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
              Slippage % <HelpTip text={HELP_SLIPPAGE} />
            </label>
            <input
              type="number"
              value={slippagePct}
              onChange={(e) => setSlippagePct(Number(e.target.value))}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm
                         focus:outline-none focus:border-blue-500"
              min={0}
              max={5}
              step={0.01}
            />
          </div>

          {/* Stop Loss */}
          <div>
            <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
              Stop Loss % <HelpTip text={HELP_STOP_LOSS} />
            </label>
            <input
              type="number"
              value={stopLossPct}
              onChange={(e) => setStopLossPct(Number(e.target.value))}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm
                         focus:outline-none focus:border-blue-500"
              min={0.5}
              max={50}
              step={0.5}
            />
          </div>

          {/* Take Profit */}
          <div>
            <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
              Take Profit % <HelpTip text={HELP_TAKE_PROFIT} />
            </label>
            <input
              type="number"
              value={takeProfitPct}
              onChange={(e) => setTakeProfitPct(Number(e.target.value))}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm
                         focus:outline-none focus:border-blue-500"
              min={1}
              max={100}
              step={1}
            />
          </div>

          {/* Run button */}
          <div className="flex items-end">
            <button
              onClick={handleRun}
              disabled={loading || !ticker.trim()}
              className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700
                         disabled:text-gray-500 text-white text-sm font-medium rounded-lg
                         transition-colors"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg
                    className="animate-spin h-4 w-4"
                    viewBox="0 0 24 24"
                    fill="none"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                    />
                  </svg>
                  Running...
                </span>
              ) : (
                "Run Backtest"
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-900/30 border border-red-800 rounded-lg p-4 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <>
          {/* Summary banner */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div>
                <span className="text-lg font-bold text-white">
                  {result.ticker}
                </span>
                <span className="text-gray-500 text-sm ml-2">
                  {result.period} &middot; {result.strategy_name}
                </span>
                {result.regime && (
                  <span className="ml-2 px-2 py-0.5 bg-gray-800 text-gray-400 text-xs rounded">
                    {result.regime.label}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-6 text-sm">
                <div>
                  <span className="text-gray-500">Start: </span>
                  <span className="text-white">{fmtMoney(result.initial_cash)}</span>
                </div>
                <div>
                  <span className="text-gray-500">End: </span>
                  <span
                    className={
                      result.final_equity >= result.initial_cash
                        ? "text-emerald-400"
                        : "text-red-400"
                    }
                  >
                    {fmtMoney(result.final_equity)}
                  </span>
                </div>
                <div
                  className={`text-lg font-bold ${
                    result.total_return_pct >= 0
                      ? "text-emerald-400"
                      : "text-red-400"
                  }`}
                >
                  {fmtPct(result.total_return_pct)}
                </div>
              </div>
            </div>
          </div>

          {/* Equity curve */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <h3 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-1">
              Equity Curve <HelpTip text={HELP_EQUITY_CURVE} />
            </h3>
            <EquityCurveChart
              data={result.equity_curve}
              warmupBars={result.warmup_bars}
            />
          </div>

          {/* Tabs: Metrics / Trade Log */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl">
            <div className="flex border-b border-gray-800">
              <button
                onClick={() => setActiveTab("metrics")}
                className={`px-5 py-3 text-sm font-medium transition-colors ${
                  activeTab === "metrics"
                    ? "text-blue-400 border-b-2 border-blue-400"
                    : "text-gray-500 hover:text-gray-300"
                }`}
              >
                Performance Metrics
              </button>
              <button
                onClick={() => setActiveTab("trades")}
                className={`px-5 py-3 text-sm font-medium transition-colors flex items-center gap-1 ${
                  activeTab === "trades"
                    ? "text-blue-400 border-b-2 border-blue-400"
                    : "text-gray-500 hover:text-gray-300"
                }`}
              >
                Trade Log <HelpTip text={HELP_TRADE_LOG} />
                <span className="ml-1 px-1.5 py-0.5 bg-gray-800 text-gray-400 text-xs rounded">
                  {result.total_trades}
                </span>
              </button>
            </div>

            <div className="p-5">
              {activeTab === "metrics" && (
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
                  <MetricCard
                    label="Total Return"
                    value={fmtPct(result.total_return_pct)}
                    helpText={HELP_BT_TOTAL_RETURN}
                    color={metricColor(result.total_return_pct)}
                  />
                  <MetricCard
                    label="Annualized Return"
                    value={fmtPct(result.annualized_return_pct)}
                    helpText={HELP_ANNUALIZED_RETURN}
                    color={metricColor(result.annualized_return_pct)}
                  />
                  <MetricCard
                    label="Max Drawdown"
                    value={fmtPct(-Math.abs(result.max_drawdown_pct))}
                    helpText={HELP_MAX_DRAWDOWN}
                    color={result.max_drawdown_pct > 20 ? "red" : result.max_drawdown_pct > 10 ? "yellow" : "green"}
                  />
                  <MetricCard
                    label="Sharpe Ratio"
                    value={fmtNum(result.sharpe_ratio)}
                    helpText={HELP_SHARPE_RATIO}
                    color={
                      result.sharpe_ratio >= 1
                        ? "green"
                        : result.sharpe_ratio >= 0
                          ? "yellow"
                          : "red"
                    }
                  />
                  <MetricCard
                    label="Win Rate"
                    value={`${fmtNum(result.win_rate_pct, 1)}%`}
                    helpText={HELP_WIN_RATE}
                    color={
                      result.win_rate_pct >= 50
                        ? "green"
                        : result.win_rate_pct >= 40
                          ? "yellow"
                          : "red"
                    }
                  />
                  <MetricCard
                    label="Profit Factor"
                    value={fmtNum(result.profit_factor)}
                    helpText={HELP_PROFIT_FACTOR}
                    color={
                      result.profit_factor >= 1.5
                        ? "green"
                        : result.profit_factor >= 1.0
                          ? "yellow"
                          : "red"
                    }
                  />
                  <MetricCard
                    label="Total Trades"
                    value={String(result.total_trades)}
                    helpText={HELP_TOTAL_TRADES}
                  />
                  <MetricCard
                    label="Avg Trade P&L"
                    value={`${(result.avg_trade_pnl_pct * 100).toFixed(2)}%`}
                    helpText={HELP_AVG_TRADE_PNL}
                    color={metricColor(result.avg_trade_pnl_pct)}
                  />
                  <MetricCard
                    label="Best Trade"
                    value={`+${(result.best_trade_pnl_pct * 100).toFixed(2)}%`}
                    helpText={HELP_BEST_TRADE}
                    color="green"
                  />
                  <MetricCard
                    label="Worst Trade"
                    value={`${(result.worst_trade_pnl_pct * 100).toFixed(2)}%`}
                    helpText={HELP_WORST_TRADE}
                    color="red"
                  />
                  <MetricCard
                    label="Avg Bars Held"
                    value={fmtNum(result.avg_bars_held, 1)}
                    helpText={HELP_AVG_BARS_HELD}
                  />
                </div>
              )}

              {activeTab === "trades" && (
                <>
                  {result.trades.length === 0 ? (
                    <p className="text-gray-500 text-sm">
                      No trades were executed during this backtest period.
                    </p>
                  ) : (
                    <TradeLogTable trades={result.trades} />
                  )}
                </>
              )}
            </div>
          </div>
        </>
      )}

      {/* ================================================================= */}
      {/* Walk-Forward Testing Section                                      */}
      {/* ================================================================= */}
      <WalkForwardSection />

      {/* ================================================================= */}
      {/* Auto-Tuner Section                                                */}
      {/* ================================================================= */}
      <AutoTunerSection />

      {/* ================================================================= */}
      {/* Strategy Library Section                                          */}
      {/* ================================================================= */}
      <StrategyLibrarySection />
    </div>
  );
}


// ---------------------------------------------------------------------------
// Walk-Forward Testing sub-component
// ---------------------------------------------------------------------------

function WalkForwardSection() {
  const [ticker, setTicker] = useState("AAPL");
  const [trainYears, setTrainYears] = useState(5);
  const [testYears, setTestYears] = useState(1);
  const [maxWindows, setMaxWindows] = useState(5);

  const [result, setResult] = useState<WalkForwardResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleRun = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await runWalkForward({
        ticker: ticker.trim().toUpperCase(),
        train_years: trainYears,
        test_years: testYears,
        max_windows: maxWindows,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Walk-forward test failed");
    } finally {
      setLoading(false);
    }
  }, [ticker, trainYears, testYears, maxWindows]);

  const fmtPct = (v: number | null | undefined) =>
    v == null ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
  const fmtNum = (v: number | null | undefined, d = 2) =>
    v == null ? "—" : v.toFixed(d);

  // Stability color
  const stabilityColor = (score: number) =>
    score >= 70 ? "text-emerald-400" : score >= 40 ? "text-yellow-400" : "text-red-400";

  return (
    <div className="space-y-4">
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h3 className="text-sm font-medium text-gray-300 mb-4 flex items-center gap-1">
          Walk-Forward Testing <HelpTip text={HELP_WALK_FORWARD} />
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Ticker</label>
            <input
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm
                         focus:outline-none focus:border-blue-500"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
              Train Years <HelpTip text={HELP_TRAIN_YEARS} />
            </label>
            <select
              value={trainYears}
              onChange={(e) => setTrainYears(Number(e.target.value))}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm
                         focus:outline-none focus:border-blue-500"
            >
              {[2, 3, 4, 5, 7, 10].map((n) => (
                <option key={n} value={n}>{n} years</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
              Test Years <HelpTip text={HELP_TEST_YEARS} />
            </label>
            <select
              value={testYears}
              onChange={(e) => setTestYears(Number(e.target.value))}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm
                         focus:outline-none focus:border-blue-500"
            >
              {[1, 2, 3].map((n) => (
                <option key={n} value={n}>{n} year{n > 1 ? "s" : ""}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Max Windows</label>
            <select
              value={maxWindows}
              onChange={(e) => setMaxWindows(Number(e.target.value))}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm
                         focus:outline-none focus:border-blue-500"
            >
              {[3, 5, 7, 10].map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </div>
          <div className="flex items-end">
            <button
              onClick={handleRun}
              disabled={loading || !ticker.trim()}
              className="w-full px-4 py-2 bg-purple-600 hover:bg-purple-500 disabled:bg-gray-700
                         disabled:text-gray-500 text-white text-sm font-medium rounded-lg
                         transition-colors"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Running...
                </span>
              ) : (
                "Run Walk-Forward"
              )}
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="bg-red-900/30 border border-red-800 rounded-lg p-4 text-red-300 text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-5">
          {/* Verdict banner */}
          <div
            className={`p-4 rounded-lg border ${
              result.stability_score >= 70
                ? "bg-emerald-900/20 border-emerald-800"
                : result.stability_score >= 40
                  ? "bg-yellow-900/20 border-yellow-800"
                  : "bg-red-900/20 border-red-800"
            }`}
          >
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div>
                <span className="text-white font-semibold">{result.ticker}</span>
                <span className="text-gray-400 text-sm ml-2">
                  {result.total_windows} windows &middot; {result.train_years}Y train / {result.test_years}Y test
                </span>
              </div>
              <div className="flex items-center gap-4">
                <div className="text-center">
                  <div className="text-xs text-gray-400 flex items-center gap-1">
                    Stability <HelpTip text={HELP_STABILITY_SCORE} />
                  </div>
                  <div className={`text-xl font-bold ${stabilityColor(result.stability_score)}`}>
                    {result.stability_score.toFixed(0)}
                  </div>
                </div>
              </div>
            </div>
            <p className="text-sm text-gray-300 mt-2">{result.verdict}</p>
          </div>

          {/* Aggregate metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <MetricCard
              label="Avg Return"
              value={fmtPct(result.avg_return_pct)}
              color={result.avg_return_pct >= 0 ? "green" : "red"}
            />
            <MetricCard
              label="Avg Sharpe"
              value={fmtNum(result.avg_sharpe_ratio)}
              helpText={HELP_SHARPE_RATIO}
              color={result.avg_sharpe_ratio >= 1 ? "green" : result.avg_sharpe_ratio >= 0 ? "yellow" : "red"}
            />
            <MetricCard
              label="Avg Win Rate"
              value={`${fmtNum(result.avg_win_rate_pct, 1)}%`}
              helpText={HELP_WIN_RATE}
              color={result.avg_win_rate_pct >= 50 ? "green" : "yellow"}
            />
            <MetricCard
              label="Avg Drawdown"
              value={fmtPct(-Math.abs(result.avg_max_drawdown_pct))}
              helpText={HELP_MAX_DRAWDOWN}
              color={result.avg_max_drawdown_pct > 20 ? "red" : "yellow"}
            />
            <MetricCard
              label="Worst Return"
              value={fmtPct(result.worst_return_pct)}
              helpText={HELP_WORST_WINDOW}
              color={result.worst_return_pct >= 0 ? "green" : "red"}
            />
            <MetricCard
              label="Return Std Dev"
              value={`${result.return_std_dev.toFixed(1)}%`}
              helpText={HELP_RETURN_STD_DEV}
              color={result.return_std_dev < 10 ? "green" : result.return_std_dev < 25 ? "yellow" : "red"}
            />
          </div>

          {/* Per-window table */}
          <div>
            <h4 className="text-xs font-medium text-gray-400 mb-2">Per-Window Results</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-gray-700 text-gray-400">
                    <th className="py-2 px-2 text-left">#</th>
                    <th className="py-2 px-2 text-left">Test Period</th>
                    <th className="py-2 px-2 text-right">Return</th>
                    <th className="py-2 px-2 text-right">Sharpe</th>
                    <th className="py-2 px-2 text-right">Win Rate</th>
                    <th className="py-2 px-2 text-right">Profit Factor</th>
                    <th className="py-2 px-2 text-right">Max DD</th>
                    <th className="py-2 px-2 text-right">Trades</th>
                    <th className="py-2 px-2 text-left">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {result.windows.map((w) => (
                    <tr
                      key={w.window_index}
                      className={`border-b border-gray-800 ${
                        w.window_index === result.worst_window_index
                          ? "bg-red-900/10"
                          : "hover:bg-gray-800/40"
                      }`}
                    >
                      <td className="py-1.5 px-2 text-gray-400">{w.window_index + 1}</td>
                      <td className="py-1.5 px-2 text-gray-300">
                        {w.test_start.slice(0, 7)} → {w.test_end.slice(0, 7)}
                      </td>
                      <td className={`py-1.5 px-2 text-right font-medium ${
                        w.error ? "text-gray-600" : w.total_return_pct >= 0 ? "text-emerald-400" : "text-red-400"
                      }`}>
                        {w.error ? "—" : fmtPct(w.total_return_pct)}
                      </td>
                      <td className="py-1.5 px-2 text-right text-gray-300">
                        {w.error ? "—" : fmtNum(w.sharpe_ratio)}
                      </td>
                      <td className="py-1.5 px-2 text-right text-gray-300">
                        {w.error ? "—" : `${fmtNum(w.win_rate_pct, 1)}%`}
                      </td>
                      <td className="py-1.5 px-2 text-right text-gray-300">
                        {w.error ? "—" : fmtNum(w.profit_factor)}
                      </td>
                      <td className="py-1.5 px-2 text-right text-gray-300">
                        {w.error ? "—" : fmtPct(-Math.abs(w.max_drawdown_pct))}
                      </td>
                      <td className="py-1.5 px-2 text-right text-gray-300">
                        {w.error ? "—" : w.total_trades}
                      </td>
                      <td className="py-1.5 px-2">
                        {w.error ? (
                          <span className="text-red-400 text-xs" title={w.error}>Failed</span>
                        ) : (
                          <span className="text-emerald-400 text-xs">OK</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}


// ---------------------------------------------------------------------------
// Auto-Tuner sub-component
// ---------------------------------------------------------------------------

const OBJECTIVE_OPTIONS: { value: string; label: string; help: string }[] = [
  { value: "balanced", label: "Balanced", help: HELP_BALANCED },
  { value: "beat_buy_hold", label: "Beat Buy-and-Hold", help: HELP_BEAT_BUY_HOLD },
  { value: "max_return", label: "Maximize Return", help: HELP_MAX_RETURN },
  { value: "max_risk_adjusted", label: "Max Risk-Adjusted", help: HELP_MAX_RISK_ADJUSTED },
  { value: "min_drawdown", label: "Minimize Drawdown", help: HELP_MIN_DRAWDOWN },
];

function AutoTunerSection() {
  const [ticker, setTicker] = useState("AAPL");
  const [objective, setObjective] = useState("balanced");
  const [nTrials, setNTrials] = useState(30);
  const [trainYears, setTrainYears] = useState(3);
  const [testYears, setTestYears] = useState(1);
  const [maxWindows, setMaxWindows] = useState(3);

  const [result, setResult] = useState<AutoTuneResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Track elapsed time while running
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const handleRun = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setElapsed(0);
    // Start elapsed timer
    timerRef.current = setInterval(() => setElapsed((t) => t + 1), 1000);
    try {
      const res = await runAutoTune({
        ticker: ticker.trim().toUpperCase(),
        objective,
        n_trials: nTrials,
        train_years: trainYears,
        test_years: testYears,
        max_windows: maxWindows,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Auto-tune failed");
    } finally {
      setLoading(false);
      if (timerRef.current) clearInterval(timerRef.current);
    }
  }, [ticker, objective, nTrials, trainYears, testYears, maxWindows]);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const fmtPct = (v: number | null | undefined) =>
    v == null ? "\u2014" : `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
  const fmtNum = (v: number | null | undefined, d = 2) =>
    v == null ? "\u2014" : v.toFixed(d);
  const fmtTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return m > 0 ? `${m}m ${s}s` : `${s}s`;
  };

  // Format param value for display
  const fmtParamValue = (v: unknown): string => {
    if (v === null || v === undefined) return "\u2014";
    if (typeof v === "boolean") return v ? "Yes" : "No";
    if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4);
    return String(v);
  };

  // Humanize param name: "score_thresholds.strong_buy" -> "Strong Buy"
  const humanizeParam = (key: string): string => {
    return key
      .split(".")
      .pop()!
      .replace(/_/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase());
  };

  // Improvement color
  const improvementColor = (pct: number) =>
    pct > 10 ? "text-emerald-400" : pct > 0 ? "text-yellow-400" : "text-red-400";

  return (
    <div className="space-y-4">
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h3 className="text-sm font-medium text-gray-300 mb-4 flex items-center gap-1">
          Auto-Tuner <HelpTip text={HELP_AUTO_TUNER} />
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-7 gap-4">
          {/* Ticker */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">Ticker</label>
            <input
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm
                         focus:outline-none focus:border-blue-500"
            />
          </div>

          {/* Objective */}
          <div className="col-span-2 sm:col-span-1">
            <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
              Objective <HelpTip text={HELP_TUNER_OBJECTIVE} />
            </label>
            <select
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm
                         focus:outline-none focus:border-blue-500"
            >
              {OBJECTIVE_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>

          {/* N Trials */}
          <div>
            <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
              Trials <HelpTip text={HELP_N_TRIALS} />
            </label>
            <select
              value={nTrials}
              onChange={(e) => setNTrials(Number(e.target.value))}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm
                         focus:outline-none focus:border-blue-500"
            >
              {[10, 20, 30, 50, 75, 100].map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </div>

          {/* Train Years */}
          <div>
            <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
              Train Years <HelpTip text={HELP_TRAIN_YEARS} />
            </label>
            <select
              value={trainYears}
              onChange={(e) => setTrainYears(Number(e.target.value))}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm
                         focus:outline-none focus:border-blue-500"
            >
              {[1, 2, 3, 4, 5, 7, 10].map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </div>

          {/* Test Years */}
          <div>
            <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
              Test Years <HelpTip text={HELP_TEST_YEARS} />
            </label>
            <select
              value={testYears}
              onChange={(e) => setTestYears(Number(e.target.value))}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm
                         focus:outline-none focus:border-blue-500"
            >
              {[1, 2, 3, 5].map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </div>

          {/* Run button */}
          <div className="flex items-end">
            <button
              onClick={handleRun}
              disabled={loading || !ticker.trim()}
              className="w-full px-4 py-2 bg-amber-600 hover:bg-amber-500 disabled:bg-gray-700
                         disabled:text-gray-500 text-white text-sm font-medium rounded-lg
                         transition-colors"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  {fmtTime(elapsed)}
                </span>
              ) : (
                "Run Auto-Tune"
              )}
            </button>
          </div>
        </div>

        {/* Objective description row */}
        <div className="mt-3 text-xs text-gray-500">
          {OBJECTIVE_OPTIONS.find((o) => o.value === objective)?.help}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-900/30 border border-red-800 rounded-lg p-4 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-5">
          {/* Verdict banner */}
          <div
            className={`p-4 rounded-lg border ${
              result.improvement_pct > 10
                ? "bg-emerald-900/20 border-emerald-800"
                : result.improvement_pct > 0
                  ? "bg-yellow-900/20 border-yellow-800"
                  : "bg-red-900/20 border-red-800"
            }`}
          >
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div>
                <span className="text-white font-semibold">{result.ticker}</span>
                <span className="text-gray-400 text-sm ml-2">
                  {result.objective_label} &middot; {result.n_trials} trials &middot; {fmtTime(Math.round(result.elapsed_seconds))}
                </span>
              </div>
              <div className="flex items-center gap-4">
                <div className="text-center">
                  <div className="text-xs text-gray-400 flex items-center gap-1">
                    Improvement <HelpTip text={HELP_IMPROVEMENT} />
                  </div>
                  <div className={`text-xl font-bold ${improvementColor(result.improvement_pct)}`}>
                    {result.improvement_pct >= 0 ? "+" : ""}{result.improvement_pct.toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>
            <p className="text-sm text-gray-300 mt-2">{result.verdict}</p>
          </div>

          {/* Side-by-side comparison: Baseline vs Optimized */}
          <div>
            <h4 className="text-xs font-medium text-gray-400 mb-3 flex items-center gap-1">
              Baseline vs Optimized <HelpTip text={HELP_BASELINE} />
            </h4>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-700 text-gray-400 text-xs">
                    <th className="py-2 px-3 text-left">Metric</th>
                    <th className="py-2 px-3 text-right">Baseline (Default)</th>
                    <th className="py-2 px-3 text-right">Optimized</th>
                    <th className="py-2 px-3 text-right">Change</th>
                  </tr>
                </thead>
                <tbody>
                  {([
                    {
                      label: "Avg Annual Return",
                      baseline: result.baseline_avg_annualized_return_pct,
                      best: result.best_avg_annualized_return_pct,
                      fmt: "pct" as const,
                      higherIsBetter: true,
                    },
                    {
                      label: "Avg Return",
                      baseline: result.baseline_avg_return_pct,
                      best: result.best_avg_return_pct,
                      fmt: "pct" as const,
                      higherIsBetter: true,
                    },
                    {
                      label: "Sharpe Ratio",
                      baseline: result.baseline_avg_sharpe_ratio,
                      best: result.best_avg_sharpe_ratio,
                      fmt: "num" as const,
                      higherIsBetter: true,
                    },
                    {
                      label: "Max Drawdown",
                      baseline: result.baseline_avg_max_drawdown_pct,
                      best: result.best_avg_max_drawdown_pct,
                      fmt: "pct_abs" as const,
                      higherIsBetter: false,
                    },
                    {
                      label: "Win Rate",
                      baseline: result.baseline_avg_win_rate_pct,
                      best: result.best_avg_win_rate_pct,
                      fmt: "pct_plain" as const,
                      higherIsBetter: true,
                    },
                    {
                      label: "Stability Score",
                      baseline: null as number | null,
                      best: result.best_stability_score,
                      fmt: "num0" as const,
                      higherIsBetter: true,
                    },
                  ]).map((row) => {
                    const diff = row.baseline != null ? row.best - row.baseline : null;
                    const improved = diff != null
                      ? row.higherIsBetter ? diff > 0 : diff < 0
                      : null;
                    return (
                      <tr key={row.label} className="border-b border-gray-800 hover:bg-gray-800/40">
                        <td className="py-2 px-3 text-gray-300">{row.label}</td>
                        <td className="py-2 px-3 text-right text-gray-400">
                          {row.baseline == null
                            ? "\u2014"
                            : row.fmt === "pct"
                              ? fmtPct(row.baseline)
                              : row.fmt === "pct_abs"
                                ? `-${Math.abs(row.baseline).toFixed(2)}%`
                                : row.fmt === "pct_plain"
                                  ? `${row.baseline.toFixed(1)}%`
                                  : fmtNum(row.baseline)}
                        </td>
                        <td className="py-2 px-3 text-right text-white font-medium">
                          {row.fmt === "pct"
                            ? fmtPct(row.best)
                            : row.fmt === "pct_abs"
                              ? `-${Math.abs(row.best).toFixed(2)}%`
                              : row.fmt === "pct_plain"
                                ? `${row.best.toFixed(1)}%`
                                : row.fmt === "num0"
                                  ? row.best.toFixed(0)
                                  : fmtNum(row.best)}
                        </td>
                        <td className={`py-2 px-3 text-right font-medium ${
                          diff == null ? "text-gray-500" : improved ? "text-emerald-400" : "text-red-400"
                        }`}>
                          {diff == null
                            ? "\u2014"
                            : row.fmt === "pct" || row.fmt === "pct_abs" || row.fmt === "pct_plain"
                              ? `${diff >= 0 ? "+" : ""}${diff.toFixed(2)}%`
                              : `${diff >= 0 ? "+" : ""}${diff.toFixed(2)}`}
                        </td>
                      </tr>
                    );
                  })}
                  {/* Buy-and-hold comparison row */}
                  {result.buy_hold_return_pct != null && (
                    <tr className="border-b border-gray-800 bg-gray-800/20">
                      <td className="py-2 px-3 text-gray-400 italic">Buy-and-Hold Return</td>
                      <td className="py-2 px-3 text-right text-gray-500" colSpan={3}>
                        {fmtPct(result.buy_hold_return_pct)}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Objective score comparison */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <MetricCard
              label="Baseline Objective"
              value={fmtNum(result.baseline_objective_value)}
              helpText={HELP_BASELINE}
              color="default"
            />
            <MetricCard
              label="Optimized Objective"
              value={fmtNum(result.best_objective_value)}
              color={result.best_objective_value > result.baseline_objective_value ? "green" : "red"}
            />
            <MetricCard
              label="Profit Factor"
              value={fmtNum(result.best_avg_profit_factor)}
              helpText={HELP_PROFIT_FACTOR}
              color={result.best_avg_profit_factor >= 1.5 ? "green" : result.best_avg_profit_factor >= 1.0 ? "yellow" : "red"}
            />
          </div>

          {/* Best parameters */}
          <div>
            <h4 className="text-xs font-medium text-gray-400 mb-2 flex items-center gap-1">
              Best Parameters <HelpTip text={HELP_BEST_PARAMS} />
            </h4>
            <div className="overflow-x-auto">
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
                {Object.entries(result.best_params).map(([key, value]) => (
                  <div
                    key={key}
                    className="bg-gray-800/50 border border-gray-700 rounded-lg px-3 py-2"
                  >
                    <div className="text-xs text-gray-400 truncate" title={key}>
                      {humanizeParam(key)}
                    </div>
                    <div className="text-sm text-white font-mono mt-0.5">
                      {fmtParamValue(value)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Sensitivity analysis */}
          {result.sensitivity.length > 0 && (
            <div>
              <h4 className="text-xs font-medium text-gray-400 mb-2 flex items-center gap-1">
                Parameter Sensitivity <HelpTip text={HELP_SENSITIVITY} />
              </h4>
              <div className="space-y-2">
                {result.sensitivity
                  .sort((a, b) => b.importance - a.importance)
                  .map((s) => (
                    <div key={s.param_name} className="flex items-center gap-3">
                      <div className="w-40 text-xs text-gray-400 truncate" title={s.param_name}>
                        {humanizeParam(s.param_name)}
                      </div>
                      <div className="flex-1 h-3 bg-gray-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-amber-500 rounded-full transition-all"
                          style={{ width: `${Math.max(s.importance * 100, 1)}%` }}
                        />
                      </div>
                      <div className="w-12 text-xs text-gray-400 text-right">
                        {(s.importance * 100).toFixed(0)}%
                      </div>
                      <div className="w-20 text-xs text-gray-500 text-right font-mono">
                        {fmtParamValue(s.best_value)}
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* Top 10 trials table */}
          {result.trials.length > 0 && (
            <div>
              <h4 className="text-xs font-medium text-gray-400 mb-2">
                Top Trials ({result.trials.length} total)
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-gray-700 text-gray-400">
                      <th className="py-2 px-2 text-left">#</th>
                      <th className="py-2 px-2 text-right">Objective</th>
                      <th className="py-2 px-2 text-right">Avg Return</th>
                      <th className="py-2 px-2 text-right">Sharpe</th>
                      <th className="py-2 px-2 text-right">Win Rate</th>
                      <th className="py-2 px-2 text-right">Max DD</th>
                      <th className="py-2 px-2 text-right">Stability</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.trials
                      .sort((a, b) => b.objective_value - a.objective_value)
                      .slice(0, 10)
                      .map((t, i) => (
                        <tr
                          key={t.trial_number}
                          className={`border-b border-gray-800 ${
                            i === 0 ? "bg-amber-900/10" : "hover:bg-gray-800/40"
                          }`}
                        >
                          <td className="py-1.5 px-2 text-gray-400">{i + 1}</td>
                          <td className="py-1.5 px-2 text-right text-white font-medium">
                            {fmtNum(t.objective_value)}
                          </td>
                          <td className={`py-1.5 px-2 text-right font-medium ${
                            t.avg_return_pct >= 0 ? "text-emerald-400" : "text-red-400"
                          }`}>
                            {fmtPct(t.avg_return_pct)}
                          </td>
                          <td className="py-1.5 px-2 text-right text-gray-300">
                            {fmtNum(t.avg_sharpe_ratio)}
                          </td>
                          <td className="py-1.5 px-2 text-right text-gray-300">
                            {fmtNum(t.avg_win_rate_pct, 1)}%
                          </td>
                          <td className="py-1.5 px-2 text-right text-gray-300">
                            -{Math.abs(t.avg_max_drawdown_pct).toFixed(2)}%
                          </td>
                          <td className="py-1.5 px-2 text-right text-gray-300">
                            {t.stability_score.toFixed(0)}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}


// ---------------------------------------------------------------------------
// Strategy Library sub-component
// ---------------------------------------------------------------------------

type LibraryView = "list" | "create" | "compare";

function StrategyLibrarySection() {
  const [strategies, setStrategies] = useState<StrategyItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<LibraryView>("list");

  // Create form state
  const [newName, setNewName] = useState("");
  const [newDescription, setNewDescription] = useState("");
  const [newTradeMode, setNewTradeMode] = useState("swing");
  const [newTicker, setNewTicker] = useState("");
  const [newParamsJson, setNewParamsJson] = useState("{}");
  const [saving, setSaving] = useState(false);

  // Compare state
  const [compareIds, setCompareIds] = useState<Set<number>>(new Set());

  // Import state
  const importRef = useRef<HTMLInputElement>(null);
  const [importing, setImporting] = useState(false);

  // Confirmation dialog state
  const [deleteId, setDeleteId] = useState<number | null>(null);

  // Fetch strategies on mount
  const fetchStrategies = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await listStrategies();
      setStrategies(res.strategies);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load strategies");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStrategies();
  }, [fetchStrategies]);

  // Toggle active state
  const handleToggleActive = async (s: StrategyItem) => {
    try {
      const updated = await updateStrategy(s.id, { is_active: !s.is_active });
      setStrategies((prev) =>
        prev.map((x) => (x.id === updated.id ? updated : x))
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update strategy");
    }
  };

  // Delete strategy
  const handleDelete = async (id: number) => {
    try {
      await deleteStrategy(id);
      setStrategies((prev) => prev.filter((x) => x.id !== id));
      setCompareIds((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
      setDeleteId(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete strategy");
    }
  };

  // Export strategy
  const handleExport = async (id: number) => {
    try {
      const data = await exportStrategy(id);
      const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `strategy-${data.name.replace(/\s+/g, "-").toLowerCase()}-v${data.version}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to export strategy");
    }
  };

  // Import strategy
  const handleImportFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImporting(true);
    setError(null);
    try {
      const text = await file.text();
      const data: StrategyExport = JSON.parse(text);
      await importStrategy(data);
      await fetchStrategies();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to import strategy"
      );
    } finally {
      setImporting(false);
      if (importRef.current) importRef.current.value = "";
    }
  };

  // Create strategy
  const handleCreate = async () => {
    setSaving(true);
    setError(null);
    try {
      let params = {};
      if (newParamsJson.trim()) {
        params = JSON.parse(newParamsJson);
      }
      const req: StrategyCreateRequest = {
        name: newName.trim(),
        description: newDescription.trim() || undefined,
        trade_mode: newTradeMode,
        ticker: newTicker.trim() || undefined,
        params,
      };
      await createStrategy(req);
      await fetchStrategies();
      setNewName("");
      setNewDescription("");
      setNewTradeMode("swing");
      setNewTicker("");
      setNewParamsJson("{}");
      setView("list");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save strategy");
    } finally {
      setSaving(false);
    }
  };

  // Toggle compare selection
  const toggleCompare = (id: number) => {
    setCompareIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const comparedStrategies = strategies.filter((s) => compareIds.has(s.id));

  // Formatting helpers
  const fmtPct = (v: number | null | undefined) =>
    v == null ? "\u2014" : `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
  const fmtNum = (v: number | null | undefined, d = 2) =>
    v == null ? "\u2014" : v.toFixed(d);

  const metricColor = (
    v: number | null | undefined,
    higherIsBetter = true
  ): string => {
    if (v == null) return "text-gray-500";
    if (higherIsBetter) return v > 0 ? "text-emerald-400" : v < 0 ? "text-red-400" : "text-gray-300";
    return v < 0 ? "text-emerald-400" : v > 0 ? "text-red-400" : "text-gray-300";
  };

  return (
    <div className="space-y-4">
      {/* Header + actions */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <div className="flex items-center justify-between flex-wrap gap-3 mb-4">
          <h3 className="text-sm font-medium text-gray-300 flex items-center gap-1">
            Strategy Library <HelpTip text={HELP_STRATEGY_LIBRARY} />
          </h3>
          <div className="flex items-center gap-2">
            {view === "list" && compareIds.size >= 2 && (
              <button
                onClick={() => setView("compare")}
                className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium rounded-lg transition-colors"
              >
                Compare ({compareIds.size})
              </button>
            )}
            {view === "compare" && (
              <button
                onClick={() => setView("list")}
                className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-white text-xs font-medium rounded-lg transition-colors"
              >
                Back to List
              </button>
            )}
            <button
              onClick={() => setView(view === "create" ? "list" : "create")}
              className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
                view === "create"
                  ? "bg-gray-700 hover:bg-gray-600 text-white"
                  : "bg-emerald-600 hover:bg-emerald-500 text-white"
              }`}
            >
              {view === "create" ? "Cancel" : "+ New Strategy"}
            </button>
            <label
              className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors cursor-pointer
                          ${importing ? "bg-gray-700 text-gray-500" : "bg-gray-700 hover:bg-gray-600 text-white"}`}
            >
              {importing ? "Importing..." : "Import"}
              <HelpTip text={HELP_STRATEGY_IMPORT} />
              <input
                ref={importRef}
                type="file"
                accept=".json"
                className="hidden"
                onChange={handleImportFile}
                disabled={importing}
              />
            </label>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-900/30 border border-red-800 rounded-lg p-3 text-red-300 text-sm mb-4">
            {error}
            <button
              onClick={() => setError(null)}
              className="ml-2 text-red-400 hover:text-red-300 text-xs"
            >
              dismiss
            </button>
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div className="text-center py-8">
            <svg
              className="animate-spin h-6 w-6 text-gray-400 mx-auto"
              viewBox="0 0 24 24"
              fill="none"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
              />
            </svg>
            <p className="text-gray-500 text-sm mt-2">Loading strategies...</p>
          </div>
        )}

        {/* ---- CREATE VIEW ---- */}
        {view === "create" && (
          <div className="space-y-4">
            <h4 className="text-xs font-medium text-gray-400">
              Save a New Strategy
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  Name *
                </label>
                <input
                  type="text"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm
                             focus:outline-none focus:border-blue-500"
                  placeholder="My Momentum Strategy"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  Ticker (optional)
                </label>
                <input
                  type="text"
                  value={newTicker}
                  onChange={(e) => setNewTicker(e.target.value.toUpperCase())}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm
                             focus:outline-none focus:border-blue-500"
                  placeholder="AAPL"
                />
              </div>
              <div className="sm:col-span-2">
                <label className="block text-xs text-gray-400 mb-1">
                  Description
                </label>
                <input
                  type="text"
                  value={newDescription}
                  onChange={(e) => setNewDescription(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm
                             focus:outline-none focus:border-blue-500"
                  placeholder="Optimized for high-momentum large-cap stocks"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  Trade Mode
                </label>
                <select
                  value={newTradeMode}
                  onChange={(e) => setNewTradeMode(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm
                             focus:outline-none focus:border-blue-500"
                >
                  <option value="swing">Swing Trade</option>
                  <option value="long_term">Long-Term</option>
                </select>
              </div>
              <div className="sm:col-span-2">
                <label className="block text-xs text-gray-400 mb-1 flex items-center gap-1">
                  Parameters (JSON) <HelpTip text={HELP_STRATEGY_PARAMS} />
                </label>
                <textarea
                  value={newParamsJson}
                  onChange={(e) => setNewParamsJson(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm
                             font-mono focus:outline-none focus:border-blue-500 min-h-[80px]"
                  rows={3}
                  placeholder='{"score_thresholds": {"strong_buy": 7.5}, ...}'
                />
              </div>
            </div>
            <div className="flex justify-end">
              <button
                onClick={handleCreate}
                disabled={saving || !newName.trim()}
                className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-700
                           disabled:text-gray-500 text-white text-sm font-medium rounded-lg transition-colors"
              >
                {saving ? "Saving..." : "Save Strategy"}
              </button>
            </div>
          </div>
        )}

        {/* ---- COMPARE VIEW ---- */}
        {view === "compare" && comparedStrategies.length >= 2 && (
          <div className="space-y-4">
            <h4 className="text-xs font-medium text-gray-400 flex items-center gap-1">
              Strategy Comparison <HelpTip text={HELP_STRATEGY_COMPARE} />
            </h4>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-700 text-gray-400 text-xs">
                    <th className="py-2 px-3 text-left">Metric</th>
                    {comparedStrategies.map((s) => (
                      <th key={s.id} className="py-2 px-3 text-right">
                        <div className="flex items-center justify-end gap-1">
                          {s.name}
                          {s.is_preset && (
                            <span className="px-1 py-0.5 bg-blue-900/50 text-blue-400 text-[10px] rounded">
                              PRESET
                            </span>
                          )}
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {([
                    { label: "Total Return", key: "total_return_pct" as const, fmt: "pct", higher: true },
                    { label: "Annualized Return", key: "annualized_return_pct" as const, fmt: "pct", higher: true },
                    { label: "Sharpe Ratio", key: "sharpe_ratio" as const, fmt: "num", higher: true },
                    { label: "Max Drawdown", key: "max_drawdown_pct" as const, fmt: "pct_neg", higher: false },
                    { label: "Win Rate", key: "win_rate_pct" as const, fmt: "pct_plain", higher: true },
                    { label: "Profit Factor", key: "profit_factor" as const, fmt: "num", higher: true },
                    { label: "Stability Score", key: "stability_score" as const, fmt: "num0", higher: true },
                    { label: "Trade Mode", key: "trade_mode" as const, fmt: "text", higher: true },
                    { label: "Version", key: "version" as const, fmt: "num0", higher: true },
                  ] as const).map((row) => {
                    // Find best value for highlighting
                    const numValues = comparedStrategies
                      .map((s) => s[row.key])
                      .filter((v): v is number => typeof v === "number");
                    const bestVal = numValues.length > 0
                      ? (row.higher ? Math.max(...numValues) : Math.min(...numValues))
                      : null;

                    return (
                      <tr
                        key={row.label}
                        className="border-b border-gray-800 hover:bg-gray-800/40"
                      >
                        <td className="py-2 px-3 text-gray-400 text-xs">
                          {row.label}
                        </td>
                        {comparedStrategies.map((s) => {
                          const val = s[row.key];
                          const isBest = typeof val === "number" && val === bestVal;
                          let display: string;
                          if (row.fmt === "text") {
                            display = String(val ?? "\u2014");
                          } else if (row.fmt === "pct") {
                            display = fmtPct(val as number | null);
                          } else if (row.fmt === "pct_neg") {
                            display =
                              val == null
                                ? "\u2014"
                                : `-${Math.abs(val as number).toFixed(2)}%`;
                          } else if (row.fmt === "pct_plain") {
                            display =
                              val == null ? "\u2014" : `${(val as number).toFixed(1)}%`;
                          } else if (row.fmt === "num0") {
                            display =
                              val == null ? "\u2014" : (val as number).toFixed(0);
                          } else {
                            display = fmtNum(val as number | null);
                          }

                          return (
                            <td
                              key={s.id}
                              className={`py-2 px-3 text-right text-sm ${
                                isBest ? "text-emerald-400 font-semibold" : "text-gray-300"
                              }`}
                            >
                              {display}
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ---- LIST VIEW ---- */}
        {view === "list" && !loading && (
          <>
            {strategies.length === 0 ? (
              <p className="text-gray-500 text-sm py-4 text-center">
                No strategies saved yet. Create one or run a backtest to get started.
              </p>
            ) : (
              <>
                {compareIds.size > 0 && compareIds.size < 2 && (
                  <p className="text-xs text-gray-500 mb-3">
                    Select at least 2 strategies to compare.
                  </p>
                )}
                <div className="space-y-3">
                  {strategies.map((s) => (
                    <div
                      key={s.id}
                      className={`border rounded-lg p-4 transition-colors ${
                        compareIds.has(s.id)
                          ? "border-indigo-500 bg-indigo-900/10"
                          : "border-gray-700 bg-gray-800/30 hover:bg-gray-800/50"
                      }`}
                    >
                      {/* Card header row */}
                      <div className="flex items-start justify-between gap-3 mb-3">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="text-sm font-medium text-white truncate">
                              {s.name}
                            </span>
                            {s.is_preset && (
                              <span className="px-1.5 py-0.5 bg-blue-900/50 text-blue-400 text-[10px] rounded flex items-center gap-0.5">
                                PRESET <HelpTip text={HELP_STRATEGY_PRESET} />
                              </span>
                            )}
                            <span className="text-[10px] text-gray-500 flex items-center gap-0.5">
                              v{s.version} <HelpTip text={HELP_STRATEGY_VERSION} />
                            </span>
                            <span className="px-1.5 py-0.5 bg-gray-800 text-gray-400 text-[10px] rounded">
                              {s.trade_mode === "long_term" ? "Long-Term" : "Swing"}
                            </span>
                            {s.ticker && (
                              <span className="px-1.5 py-0.5 bg-gray-800 text-gray-300 text-[10px] rounded font-mono">
                                {s.ticker}
                              </span>
                            )}
                          </div>
                          {s.description && (
                            <p className="text-xs text-gray-500 mt-1 truncate">
                              {s.description}
                            </p>
                          )}
                        </div>

                        {/* Actions */}
                        <div className="flex items-center gap-2 flex-shrink-0">
                          {/* Compare checkbox */}
                          <label className="flex items-center gap-1 text-xs text-gray-400 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={compareIds.has(s.id)}
                              onChange={() => toggleCompare(s.id)}
                              className="rounded border-gray-600 bg-gray-800 text-indigo-500 focus:ring-indigo-500 focus:ring-offset-0 h-3.5 w-3.5"
                            />
                            Compare
                          </label>

                          {/* Active toggle */}
                          <button
                            onClick={() => handleToggleActive(s)}
                            className={`px-2 py-1 text-[10px] font-medium rounded transition-colors flex items-center gap-0.5 ${
                              s.is_active
                                ? "bg-emerald-900/50 text-emerald-400 hover:bg-emerald-900/70"
                                : "bg-gray-800 text-gray-500 hover:bg-gray-700 hover:text-gray-300"
                            }`}
                            title={s.is_active ? "Deactivate strategy" : "Activate for Portfolio Simulation"}
                          >
                            {s.is_active ? "Active" : "Inactive"}
                            <HelpTip text={HELP_STRATEGY_ACTIVE} />
                          </button>

                          {/* Export */}
                          <button
                            onClick={() => handleExport(s.id)}
                            className="px-2 py-1 bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-gray-200 text-[10px] rounded transition-colors flex items-center gap-0.5"
                            title="Export as JSON"
                          >
                            Export <HelpTip text={HELP_STRATEGY_EXPORT} />
                          </button>

                          {/* Delete (user strategies only) */}
                          {!s.is_preset && (
                            <button
                              onClick={() => setDeleteId(s.id)}
                              className="px-2 py-1 bg-gray-800 hover:bg-red-900/50 text-gray-500 hover:text-red-400 text-[10px] rounded transition-colors"
                              title="Delete strategy"
                            >
                              Delete
                            </button>
                          )}
                        </div>
                      </div>

                      {/* Metrics row */}
                      {s.total_return_pct != null && (
                        <div className="grid grid-cols-3 sm:grid-cols-4 lg:grid-cols-7 gap-2">
                          <div>
                            <div className="text-[10px] text-gray-500">Return</div>
                            <div
                              className={`text-xs font-medium ${metricColor(s.total_return_pct)}`}
                            >
                              {fmtPct(s.total_return_pct)}
                            </div>
                          </div>
                          <div>
                            <div className="text-[10px] text-gray-500">Annual</div>
                            <div
                              className={`text-xs font-medium ${metricColor(s.annualized_return_pct)}`}
                            >
                              {fmtPct(s.annualized_return_pct)}
                            </div>
                          </div>
                          <div>
                            <div className="text-[10px] text-gray-500">Sharpe</div>
                            <div
                              className={`text-xs font-medium ${metricColor(s.sharpe_ratio)}`}
                            >
                              {fmtNum(s.sharpe_ratio)}
                            </div>
                          </div>
                          <div>
                            <div className="text-[10px] text-gray-500">Max DD</div>
                            <div
                              className={`text-xs font-medium ${metricColor(
                                s.max_drawdown_pct != null ? -s.max_drawdown_pct : null
                              )}`}
                            >
                              {s.max_drawdown_pct != null
                                ? `-${Math.abs(s.max_drawdown_pct).toFixed(2)}%`
                                : "\u2014"}
                            </div>
                          </div>
                          <div>
                            <div className="text-[10px] text-gray-500">Win Rate</div>
                            <div className="text-xs font-medium text-gray-300">
                              {s.win_rate_pct != null
                                ? `${s.win_rate_pct.toFixed(1)}%`
                                : "\u2014"}
                            </div>
                          </div>
                          <div>
                            <div className="text-[10px] text-gray-500">P.Factor</div>
                            <div
                              className={`text-xs font-medium ${metricColor(
                                s.profit_factor != null ? s.profit_factor - 1 : null
                              )}`}
                            >
                              {fmtNum(s.profit_factor)}
                            </div>
                          </div>
                          <div>
                            <div className="text-[10px] text-gray-500">Stability</div>
                            <div
                              className={`text-xs font-medium ${
                                s.stability_score == null
                                  ? "text-gray-500"
                                  : s.stability_score >= 70
                                    ? "text-emerald-400"
                                    : s.stability_score >= 40
                                      ? "text-yellow-400"
                                      : "text-red-400"
                              }`}
                            >
                              {s.stability_score != null
                                ? s.stability_score.toFixed(0)
                                : "\u2014"}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </>
            )}
          </>
        )}
      </div>

      {/* Delete confirmation dialog */}
      {deleteId != null && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-900 border border-gray-700 rounded-xl p-6 max-w-sm w-full mx-4">
            <h4 className="text-white font-medium mb-2">Delete Strategy?</h4>
            <p className="text-gray-400 text-sm mb-4">
              This action cannot be undone. The strategy and all its data will be
              permanently removed.
            </p>
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setDeleteId(null)}
                className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 text-sm rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => handleDelete(deleteId)}
                className="px-4 py-2 bg-red-600 hover:bg-red-500 text-white text-sm font-medium rounded-lg transition-colors"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
