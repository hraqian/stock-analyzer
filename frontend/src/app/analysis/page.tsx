"use client";

import { useState, useCallback } from "react";
import { analyzeStock } from "@/lib/api";
import type { AnalysisResult } from "@/lib/api";
import CandlestickChart from "@/components/CandlestickChart";

const PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y"];
const INTERVALS = ["1d", "1wk", "1mo"];

/** Color for a 0-10 score. */
function scoreColor(score: number): string {
  if (score >= 7) return "text-green-400";
  if (score >= 5.5) return "text-green-300";
  if (score >= 4.5) return "text-gray-400";
  if (score >= 3) return "text-red-300";
  return "text-red-400";
}

/** Background bar width for a 0-10 score. */
function scoreBarStyle(score: number): React.CSSProperties {
  const pct = Math.min(100, Math.max(0, (score / 10) * 100));
  const color =
    score >= 7
      ? "rgba(34, 197, 94, 0.2)"
      : score >= 4.5
        ? "rgba(156, 163, 175, 0.15)"
        : "rgba(239, 68, 68, 0.2)";
  return {
    background: `linear-gradient(to right, ${color} ${pct}%, transparent ${pct}%)`,
  };
}

/** Format large numbers. */
function fmtNum(n: unknown): string {
  if (n == null) return "N/A";
  const v = Number(n);
  if (isNaN(v)) return "N/A";
  if (v >= 1e12) return `$${(v / 1e12).toFixed(2)}T`;
  if (v >= 1e9) return `$${(v / 1e9).toFixed(2)}B`;
  if (v >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
  return v.toLocaleString();
}

export default function AnalysisPage() {
  const [ticker, setTicker] = useState("");
  const [period, setPeriod] = useState("6mo");
  const [interval, setInterval] = useState("1d");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const runAnalysis = useCallback(async () => {
    const t = ticker.trim().toUpperCase();
    if (!t) return;
    setLoading(true);
    setError(null);
    try {
      const data = await analyzeStock(t, period, interval);
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Analysis failed");
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, [ticker, period, interval]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") runAnalysis();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-semibold text-white">
          Single Stock Analysis
        </h2>
        <p className="text-sm text-gray-500 mt-1">
          Deep-dive into one ticker with price charts, technical indicators,
          pattern recognition, regime detection, and composite scoring.
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <label className="block text-xs text-gray-500 mb-1">Ticker</label>
          <input
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            onKeyDown={handleKeyDown}
            placeholder="AAPL"
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm
                       focus:outline-none focus:border-blue-500 w-28"
          />
        </div>

        <div>
          <label className="block text-xs text-gray-500 mb-1">Period</label>
          <select
            value={period}
            onChange={(e) => setPeriod(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm
                       focus:outline-none focus:border-blue-500"
          >
            {PERIODS.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs text-gray-500 mb-1">Interval</label>
          <select
            value={interval}
            onChange={(e) => setInterval(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm
                       focus:outline-none focus:border-blue-500"
          >
            {INTERVALS.map((i) => (
              <option key={i} value={i}>
                {i}
              </option>
            ))}
          </select>
        </div>

        <button
          onClick={runAnalysis}
          disabled={loading || !ticker.trim()}
          className="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500
                     text-white text-sm font-medium px-5 py-2 rounded-lg transition-colors"
        >
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-900/30 border border-red-800 rounded-lg p-4 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-12 text-center">
          <div className="animate-pulse text-gray-400">
            <p className="text-lg">Analyzing {ticker.toUpperCase()}...</p>
            <p className="text-sm text-gray-600 mt-2">
              Fetching data, computing indicators, patterns, and regime...
            </p>
          </div>
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <div className="space-y-6">
          {/* Ticker Info Bar */}
          <TickerInfoBar result={result} />

          {/* Composite Scores */}
          <CompositeScoreCard result={result} />

          {/* Price Chart */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-3">
              Price Chart &mdash; {result.ticker} ({result.period})
            </h3>
            <CandlestickChart
              data={result.price_data}
              supportLevels={result.support_levels}
              resistanceLevels={result.resistance_levels}
            />
          </div>

          {/* Regime Assessment */}
          {result.regime && <RegimeCard regime={result.regime} />}

          {/* Indicators Table */}
          <ResultsTable
            title="Technical Indicators"
            items={result.indicators}
          />

          {/* Patterns Table */}
          {result.patterns.length > 0 && (
            <ResultsTable
              title="Pattern Detection"
              items={result.patterns}
            />
          )}

          {/* Support & Resistance */}
          <SRLevelsCard result={result} />
        </div>
      )}
    </div>
  );
}

// -------------------------------------------------------------------
// Sub-components
// -------------------------------------------------------------------

function TickerInfoBar({ result }: { result: AnalysisResult }) {
  const info = result.info;
  const lastBar = result.price_data[result.price_data.length - 1];
  const firstBar = result.price_data[0];
  const change = lastBar && firstBar ? lastBar.close - firstBar.close : 0;
  const changePct =
    lastBar && firstBar && firstBar.close !== 0
      ? (change / firstBar.close) * 100
      : 0;

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
        <div>
          <span className="text-2xl font-bold text-white">{result.ticker}</span>
          {info.shortName && (
            <span className="text-gray-500 text-sm ml-2">
              {String(info.shortName)}
            </span>
          )}
        </div>

        {lastBar && (
          <div className="flex items-baseline gap-2">
            <span className="text-xl font-semibold text-white">
              ${lastBar.close.toFixed(2)}
            </span>
            <span
              className={`text-sm font-medium ${change >= 0 ? "text-green-400" : "text-red-400"}`}
            >
              {change >= 0 ? "+" : ""}
              {change.toFixed(2)} ({changePct >= 0 ? "+" : ""}
              {changePct.toFixed(2)}%)
            </span>
          </div>
        )}

        <div className="flex gap-4 text-xs text-gray-500 ml-auto">
          {info.sector && <span>{String(info.sector)}</span>}
          {info.industry && <span>{String(info.industry)}</span>}
          {info.marketCap && <span>MCap: {fmtNum(info.marketCap)}</span>}
        </div>
      </div>
    </div>
  );
}

function CompositeScoreCard({ result }: { result: AnalysisResult }) {
  const c = result.composite;
  const pc = result.pattern_composite;
  const pcOverall = typeof pc.overall === "number" ? pc.overall : null;

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-400 mb-3">
        Composite Scores
      </h3>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <ScoreBox label="Overall" score={c.overall} />
        {c.trend_score != null && (
          <ScoreBox label="Trend" score={c.trend_score} />
        )}
        {c.contrarian_score != null && (
          <ScoreBox label="Contrarian" score={c.contrarian_score} />
        )}
        {c.neutral_score != null && (
          <ScoreBox label="Neutral" score={c.neutral_score} />
        )}
        {pcOverall != null && (
          <ScoreBox label="Pattern" score={pcOverall} />
        )}
      </div>
      {c.dominant_group && (
        <p className="text-xs text-gray-500 mt-2">
          Dominant group: <span className="text-gray-300">{c.dominant_group}</span>
        </p>
      )}
    </div>
  );
}

function ScoreBox({ label, score }: { label: string; score: number }) {
  return (
    <div className="bg-gray-800 rounded-lg p-3 text-center">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className={`text-2xl font-bold ${scoreColor(score)}`}>
        {score.toFixed(1)}
      </div>
      <div className="text-xs text-gray-600">/10</div>
    </div>
  );
}

function RegimeCard({ regime }: { regime: NonNullable<AnalysisResult["regime"]> }) {
  const confidencePct = (regime.confidence * 100).toFixed(0);

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-400 mb-3">
        Market Regime
      </h3>
      <div className="flex flex-wrap items-start gap-4">
        <div>
          <span className="text-lg font-semibold text-white">
            {regime.label}
          </span>
          <span className="text-gray-500 text-sm ml-2">
            ({confidencePct}% confidence)
          </span>
          <p className="text-sm text-gray-400 mt-1">{regime.description}</p>
        </div>

        {regime.sub_type && (
          <div className="bg-gray-800 rounded-lg p-3">
            <div className="text-xs text-gray-500 mb-1">Sub-Type</div>
            <div className="text-sm text-white font-medium">
              {regime.sub_type_label}
            </div>
            <p className="text-xs text-gray-500 mt-1">
              {regime.sub_type_description}
            </p>
          </div>
        )}
      </div>

      {/* Key metrics */}
      <div className="grid grid-cols-3 sm:grid-cols-5 gap-2 mt-4">
        <MetricPill label="ADX" value={regime.metrics.adx.toFixed(1)} />
        <MetricPill
          label="ATR%"
          value={`${(regime.metrics.atr_pct * 100).toFixed(2)}%`}
        />
        <MetricPill
          label="Return"
          value={`${(regime.metrics.total_return * 100).toFixed(1)}%`}
        />
        <MetricPill
          label="% Above MA"
          value={`${regime.metrics.pct_above_ma.toFixed(0)}%`}
        />
        <MetricPill
          label="Direction"
          value={regime.metrics.trend_direction}
        />
      </div>

      {/* Reasons */}
      {regime.reasons.length > 0 && (
        <div className="mt-3">
          <div className="text-xs text-gray-500 mb-1">Classification Reasons:</div>
          <ul className="text-xs text-gray-400 space-y-0.5">
            {regime.reasons.map((r, i) => (
              <li key={i}>- {r}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function MetricPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-800 rounded px-2 py-1.5 text-center">
      <div className="text-[10px] text-gray-500 leading-tight">{label}</div>
      <div className="text-xs text-gray-300 font-medium">{value}</div>
    </div>
  );
}

function ResultsTable({
  title,
  items,
}: {
  title: string;
  items: Array<{
    name: string;
    score: number;
    display: Record<string, unknown>;
    error: string | null;
  }>;
}) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-400 mb-3">{title}</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-500 text-xs border-b border-gray-800">
              <th className="text-left py-2 pr-4">Name</th>
              <th className="text-right py-2 px-3 w-20">Score</th>
              <th className="text-left py-2 px-3">Value</th>
              <th className="text-left py-2 pl-3">Detail</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <tr
                key={item.name}
                className="border-b border-gray-800/50 hover:bg-gray-800/30"
                style={scoreBarStyle(item.score)}
              >
                <td className="py-2 pr-4 text-gray-300 font-medium">
                  {item.name}
                </td>
                <td
                  className={`py-2 px-3 text-right font-bold ${scoreColor(item.score)}`}
                >
                  {item.score.toFixed(1)}
                </td>
                <td className="py-2 px-3 text-gray-400">
                  {item.error
                    ? "Error"
                    : String(item.display.value_str ?? "")}
                </td>
                <td className="py-2 pl-3 text-gray-500 text-xs">
                  {item.error
                    ? item.error
                    : String(item.display.detail_str ?? "")}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SRLevelsCard({ result }: { result: AnalysisResult }) {
  if (
    result.support_levels.length === 0 &&
    result.resistance_levels.length === 0
  ) {
    return null;
  }

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-400 mb-3">
        Support &amp; Resistance Levels
      </h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Support */}
        <div>
          <div className="text-xs text-green-400 font-medium mb-2">
            Support
          </div>
          {result.support_levels.length === 0 ? (
            <p className="text-xs text-gray-600">None detected</p>
          ) : (
            <div className="space-y-1">
              {result.support_levels.map((lvl, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between bg-gray-800 rounded px-3 py-1.5"
                >
                  <span className="text-sm text-green-300 font-medium">
                    ${lvl.price.toFixed(2)}
                  </span>
                  <span className="text-xs text-gray-500">
                    {lvl.source} &middot; {lvl.touches} touch
                    {lvl.touches !== 1 ? "es" : ""}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Resistance */}
        <div>
          <div className="text-xs text-red-400 font-medium mb-2">
            Resistance
          </div>
          {result.resistance_levels.length === 0 ? (
            <p className="text-xs text-gray-600">None detected</p>
          ) : (
            <div className="space-y-1">
              {result.resistance_levels.map((lvl, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between bg-gray-800 rounded px-3 py-1.5"
                >
                  <span className="text-sm text-red-300 font-medium">
                    ${lvl.price.toFixed(2)}
                  </span>
                  <span className="text-xs text-gray-500">
                    {lvl.source} &middot; {lvl.touches} touch
                    {lvl.touches !== 1 ? "es" : ""}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
