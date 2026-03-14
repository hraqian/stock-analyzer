"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import HelpTip from "@/components/HelpTip";
import {
  runScan,
  ScanResponse,
  ScannerResultRow,
  ScanRequest,
} from "@/lib/api";
import {
  HELP_SCANNER,
  HELP_UNIVERSE,
  HELP_PRESET,
  HELP_PRESET_BREAKOUT,
  HELP_PRESET_PULLBACK,
  HELP_PRESET_REVERSAL,
  HELP_PRESET_DIVIDEND,
  HELP_MIN_VOLUME,
  HELP_MIN_PRICE,
  HELP_MAX_ATR_RATIO,
  HELP_SCAN_SCORE,
  HELP_SCAN_SIGNAL,
  HELP_SCAN_CONFIDENCE,
  HELP_SCAN_PATTERN,
  HELP_REGIME,
} from "@/lib/helpText";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const UNIVERSES = [
  { value: "sp500", label: "S&P 500", count: "~500" },
  { value: "nasdaq100", label: "Nasdaq 100", count: "~100" },
  { value: "dow30", label: "Dow 30", count: "30" },
  { value: "russell1000", label: "Russell 1000", count: "~1000" },
  { value: "russell2000", label: "Russell 2000", count: "~1930" },
  { value: "tsx60", label: "TSX 60", count: "60" },
  { value: "tsx_composite", label: "TSX Composite", count: "~229" },
  { value: "sector_etfs", label: "Sector ETFs", count: "11" },
  { value: "crypto20", label: "Crypto 20", count: "20" },
  { value: "bond_etfs", label: "Bond ETFs", count: "15" },
  { value: "us_dividend", label: "US Dividend", count: "~95" },
  { value: "ca_dividend_etfs", label: "CA Dividend ETFs", count: "~20" },
];

const PRESETS = [
  { value: "breakout", label: "Breakout", help: HELP_PRESET_BREAKOUT, icon: "🚀" },
  { value: "pullback", label: "Pullback", help: HELP_PRESET_PULLBACK, icon: "📉" },
  { value: "reversal", label: "Reversal", help: HELP_PRESET_REVERSAL, icon: "🔄" },
  { value: "dividend", label: "Top Dividend", help: HELP_PRESET_DIVIDEND, icon: "💰" },
];

type SortField = "rank" | "ticker" | "score" | "price" | "volume" | "confidence";
type SortDir = "asc" | "desc";

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function ScannerPage() {
  const router = useRouter();

  // ── Form state ────────────────────────────────────────────────────
  const [universe, setUniverse] = useState("sp500");
  const [customTickers, setCustomTickers] = useState("");
  const [preset, setPreset] = useState("breakout");
  const [minVolume, setMinVolume] = useState(1_000_000);
  const [minPrice, setMinPrice] = useState(5);
  const [maxAtrRatio, setMaxAtrRatio] = useState("");
  const [topN, setTopN] = useState(50);

  // ── Scan state ────────────────────────────────────────────────────
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ScanResponse | null>(null);

  // ── Table sort state ──────────────────────────────────────────────
  const [sortField, setSortField] = useState<SortField>("rank");
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  // ── Run scan ──────────────────────────────────────────────────────
  const handleScan = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const req: ScanRequest = {
        universe: universe === "custom" ? "custom" : universe,
        preset,
        period: "6mo",
        min_volume: minVolume,
        min_price: minPrice,
        max_atr_ratio: maxAtrRatio ? parseFloat(maxAtrRatio) : null,
        top_n: topN,
      };
      if (universe === "custom" && customTickers.trim()) {
        req.custom_tickers = customTickers
          .split(/[\s,;]+/)
          .map((t) => t.trim().toUpperCase())
          .filter(Boolean);
      }
      const data = await runScan(req);
      setResult(data);
      setSortField("rank");
      setSortDir("asc");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Scan failed");
    } finally {
      setLoading(false);
    }
  }, [universe, customTickers, preset, minVolume, minPrice, maxAtrRatio, topN]);

  // ── Sort logic ────────────────────────────────────────────────────
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortDir(field === "rank" ? "asc" : "desc");
    }
  };

  const sortedResults: ScannerResultRow[] = result
    ? [...result.results].sort((a, b) => {
        const av = a[sortField];
        const bv = b[sortField];
        if (typeof av === "string" && typeof bv === "string") {
          return sortDir === "asc"
            ? av.localeCompare(bv)
            : bv.localeCompare(av);
        }
        const diff = (av as number) - (bv as number);
        return sortDir === "asc" ? diff : -diff;
      })
    : [];

  // ── Navigate to single stock analysis ─────────────────────────────
  const goToAnalysis = (ticker: string) => {
    router.push(`/analysis?ticker=${encodeURIComponent(ticker)}`);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-semibold text-white">
          Market Scanner
          <HelpTip text={HELP_SCANNER} />
        </h2>
        <p className="text-sm text-gray-500 mt-1">
          Scan a universe of stocks for trade candidates matching your selected strategy.
        </p>
      </div>

      {/* Controls panel */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-4">
        {/* Row 1: Universe + Preset */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Universe selector */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">
              Universe
              <HelpTip text={HELP_UNIVERSE} />
            </label>
            <select
              value={universe}
              onChange={(e) => setUniverse(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2
                         text-white text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              {UNIVERSES.map((u) => (
                <option key={u.value} value={u.value}>
                  {u.label} ({u.count})
                </option>
              ))}
              <option value="custom">Custom Tickers</option>
            </select>
            {universe === "custom" && (
              <textarea
                value={customTickers}
                onChange={(e) => setCustomTickers(e.target.value)}
                placeholder="AAPL, MSFT, GOOGL, ..."
                rows={2}
                className="mt-2 w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2
                           text-white text-sm placeholder-gray-600 focus:outline-none
                           focus:ring-1 focus:ring-blue-500"
              />
            )}
          </div>

          {/* Preset selector */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">
              Scan Preset
              <HelpTip text={HELP_PRESET} />
            </label>
            <div className="grid grid-cols-2 gap-2">
              {PRESETS.map((p) => (
                <button
                  key={p.value}
                  onClick={() => setPreset(p.value)}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm
                    transition-colors ${
                      preset === p.value
                        ? "border-blue-500 bg-blue-500/10 text-blue-400"
                        : "border-gray-700 bg-gray-800 text-gray-400 hover:border-gray-600 hover:text-gray-300"
                    }`}
                >
                  <span>{p.icon}</span>
                  <span>{p.label}</span>
                  <HelpTip text={p.help} size={12} />
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Row 2: Filters */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">
              Min Volume
              <HelpTip text={HELP_MIN_VOLUME} size={12} />
            </label>
            <input
              type="number"
              value={minVolume}
              onChange={(e) => setMinVolume(Number(e.target.value))}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5
                         text-white text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">
              Min Price ($)
              <HelpTip text={HELP_MIN_PRICE} size={12} />
            </label>
            <input
              type="number"
              value={minPrice}
              onChange={(e) => setMinPrice(Number(e.target.value))}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5
                         text-white text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">
              Max ATR Ratio
              <HelpTip text={HELP_MAX_ATR_RATIO} size={12} />
            </label>
            <input
              type="text"
              value={maxAtrRatio}
              onChange={(e) => setMaxAtrRatio(e.target.value)}
              placeholder="e.g. 0.05"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5
                         text-white text-sm placeholder-gray-600 focus:outline-none
                         focus:ring-1 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1">
              Max Results
            </label>
            <input
              type="number"
              value={topN}
              onChange={(e) => setTopN(Number(e.target.value))}
              min={1}
              max={200}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5
                         text-white text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </div>
        </div>

        {/* Scan button */}
        <div className="flex items-center gap-4">
          <button
            onClick={handleScan}
            disabled={loading}
            className="px-6 py-2.5 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700
                       disabled:text-gray-500 text-white text-sm font-medium rounded-lg
                       transition-colors"
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                    fill="none"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                  />
                </svg>
                Scanning...
              </span>
            ) : (
              "Run Scan"
            )}
          </button>

          {loading && (
            <span className="text-sm text-gray-500">
              This may take 30-60 seconds for large universes...
            </span>
          )}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-900/20 border border-red-800 rounded-xl px-4 py-3 text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-3">
          {/* Summary bar */}
          <div className="flex flex-wrap items-center gap-4 text-sm text-gray-400">
            <span>
              <span className="text-white font-medium">{result.results.length}</span>{" "}
              results from{" "}
              <span className="text-white">{result.tickers_with_data}</span> /
              <span className="text-white"> {result.total_tickers}</span> tickers
            </span>
            <span className="text-gray-600">|</span>
            <span>
              Preset: <span className="text-blue-400 capitalize">{result.preset}</span>
            </span>
            <span className="text-gray-600">|</span>
            <span>{result.elapsed_seconds}s</span>
          </div>

          {/* Results table */}
          {result.results.length === 0 ? (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-8 text-center text-gray-500">
              No stocks matched the current filters. Try a different universe or relax the filters.
            </div>
          ) : (
            <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800 text-gray-500 text-xs uppercase tracking-wider">
                    <SortHeader
                      field="rank"
                      label="#"
                      current={sortField}
                      dir={sortDir}
                      onSort={handleSort}
                    />
                    <SortHeader
                      field="ticker"
                      label="Ticker"
                      current={sortField}
                      dir={sortDir}
                      onSort={handleSort}
                    />
                    <th className="px-3 py-3 text-left">
                      Signal
                      <HelpTip text={HELP_SCAN_SIGNAL} size={11} />
                    </th>
                    <SortHeader
                      field="score"
                      label="Score"
                      current={sortField}
                      dir={sortDir}
                      onSort={handleSort}
                      helpText={HELP_SCAN_SCORE}
                    />
                    <SortHeader
                      field="confidence"
                      label="Confidence"
                      current={sortField}
                      dir={sortDir}
                      onSort={handleSort}
                      helpText={HELP_SCAN_CONFIDENCE}
                    />
                    <th className="px-3 py-3 text-left">
                      Pattern
                      <HelpTip text={HELP_SCAN_PATTERN} size={11} />
                    </th>
                    <th className="px-3 py-3 text-left">
                      Regime
                      <HelpTip text={HELP_REGIME} size={11} />
                    </th>
                    <SortHeader
                      field="price"
                      label="Price"
                      current={sortField}
                      dir={sortDir}
                      onSort={handleSort}
                    />
                    <SortHeader
                      field="volume"
                      label="Avg Vol"
                      current={sortField}
                      dir={sortDir}
                      onSort={handleSort}
                    />
                  </tr>
                </thead>
                <tbody>
                  {sortedResults.map((row) => (
                    <tr
                      key={row.ticker}
                      onClick={() => goToAnalysis(row.ticker)}
                      className="border-b border-gray-800/50 hover:bg-gray-800/40
                                 cursor-pointer transition-colors"
                    >
                      <td className="px-3 py-2.5 text-gray-500">{row.rank}</td>
                      <td className="px-3 py-2.5 text-white font-medium">
                        {row.ticker}
                      </td>
                      <td className="px-3 py-2.5">
                        <SignalBadge signal={row.signal} />
                      </td>
                      <td className="px-3 py-2.5">
                        <ScoreBadge score={row.score} />
                      </td>
                      <td className="px-3 py-2.5 text-gray-400">
                        {(row.confidence * 100).toFixed(0)}%
                      </td>
                      <td className="px-3 py-2.5 text-gray-400 truncate max-w-[140px]">
                        {row.pattern || "—"}
                      </td>
                      <td className="px-3 py-2.5 text-gray-400 truncate max-w-[160px]">
                        {row.regime || "—"}
                      </td>
                      <td className="px-3 py-2.5 text-gray-300">
                        ${row.price.toFixed(2)}
                      </td>
                      <td className="px-3 py-2.5 text-gray-400">
                        {formatVolume(row.volume)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function SortHeader({
  field,
  label,
  current,
  dir,
  onSort,
  helpText,
}: {
  field: SortField;
  label: string;
  current: SortField;
  dir: SortDir;
  onSort: (f: SortField) => void;
  helpText?: string;
}) {
  const active = current === field;
  return (
    <th
      className="px-3 py-3 text-left cursor-pointer select-none hover:text-gray-300 transition-colors"
      onClick={() => onSort(field)}
    >
      <span className="inline-flex items-center gap-1">
        {label}
        {helpText && <HelpTip text={helpText} size={11} />}
        {active && (
          <span className="text-blue-400">{dir === "asc" ? " ▲" : " ▼"}</span>
        )}
      </span>
    </th>
  );
}

function ScoreBadge({ score }: { score: number }) {
  let color = "text-gray-400";
  if (score >= 7.0) color = "text-green-400";
  else if (score >= 5.5) color = "text-blue-400";
  else if (score < 4.0) color = "text-red-400";
  return <span className={`font-medium ${color}`}>{score.toFixed(1)}</span>;
}

function SignalBadge({ signal }: { signal: string }) {
  const lower = signal.toLowerCase();
  let bg = "bg-gray-700/50 text-gray-400";
  if (lower.includes("strong")) bg = "bg-green-900/40 text-green-400";
  else if (lower.includes("top") || lower.includes("buy") || lower.includes("breakout"))
    bg = "bg-blue-900/40 text-blue-400";
  else if (lower.includes("potential")) bg = "bg-yellow-900/30 text-yellow-400";
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${bg}`}>
      {signal}
    </span>
  );
}

function formatVolume(vol: number): string {
  if (vol >= 1_000_000) return `${(vol / 1_000_000).toFixed(1)}M`;
  if (vol >= 1_000) return `${(vol / 1_000).toFixed(0)}K`;
  return vol.toFixed(0);
}
