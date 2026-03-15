"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import {
  getSectorOverview,
  getSectorDetail,
  refreshSectorHoldings,
  getSectorHoldings,
  updateSectorHoldings,
  resetSectorHoldings,
  type SectorMomentum,
  type SectorOverviewResponse,
  type SectorDetailResponse,
  type SectorHoldingItem,
  type SectorHoldingsResponse,
} from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";
import HelpTip from "@/components/HelpTip";
import {
  HELP_SECTORS,
  HELP_SECTOR_HEATMAP,
  HELP_MOMENTUM_SCORE,
  HELP_RELATIVE_STRENGTH,
  HELP_SECTOR_ROTATION,
  HELP_SECTOR_REGIME,
  HELP_RETURN_1W,
  HELP_RETURN_1M,
  HELP_RETURN_3M,
  HELP_TOP_MOVERS,
  HELP_WORST_MOVERS,
  HELP_SECTOR_ETF,
  HELP_BENCHMARK,
  HELP_HOLDINGS_EDITOR,
  HELP_HOLDINGS_SOURCE,
} from "@/lib/helpText";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

type TimeWindow = "1w" | "1m" | "3m";

/** Format a decimal return (0.05 = 5%) as a percentage string. */
function fmtPct(v: number): string {
  const pct = v * 100;
  const sign = pct >= 0 ? "+" : "";
  return `${sign}${pct.toFixed(2)}%`;
}

/** Pick the return for the selected time window. */
function pickReturn(s: SectorMomentum, w: TimeWindow): number {
  if (w === "1w") return s.return_1w;
  if (w === "1m") return s.return_1m;
  return s.return_3m;
}

/** Pick the relative strength for the selected time window. */
function pickRS(s: SectorMomentum, w: TimeWindow): number {
  if (w === "1w") return s.rs_1w;
  if (w === "1m") return s.rs_1m;
  return s.rs_3m;
}

/** Pick benchmark return for the selected time window. */
function pickBenchmark(data: SectorOverviewResponse, w: TimeWindow): number {
  if (w === "1w") return data.benchmark_return_1w;
  if (w === "1m") return data.benchmark_return_1m;
  return data.benchmark_return_3m;
}

/** Return a tailwind bg-color class based on momentum score (0-10). */
function heatColor(score: number): string {
  if (score >= 8) return "bg-green-600";
  if (score >= 7) return "bg-green-700";
  if (score >= 6) return "bg-green-800/80";
  if (score >= 5.5) return "bg-green-900/60";
  if (score >= 4.5) return "bg-gray-700";
  if (score >= 4) return "bg-red-900/60";
  if (score >= 3) return "bg-red-800/80";
  if (score >= 2) return "bg-red-700";
  return "bg-red-600";
}

/** Return text color for return values. */
function retColor(v: number): string {
  if (v > 0.001) return "text-green-400";
  if (v < -0.001) return "text-red-400";
  return "text-gray-400";
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function BenchmarkBar({
  data,
  window,
}: {
  data: SectorOverviewResponse;
  window: TimeWindow;
}) {
  const ret = pickBenchmark(data, window);
  return (
    <div className="flex items-center gap-3 bg-gray-900 border border-gray-800 rounded-lg px-4 py-2 text-sm">
      <span className="text-gray-400">
        S&amp;P 500 (SPY)
        <HelpTip text={HELP_BENCHMARK} />
      </span>
      <span className={`font-mono font-medium ${retColor(ret)}`}>
        {fmtPct(ret)}
      </span>
      <span className="text-gray-600 text-xs">
        {window === "1w" ? "1 Week" : window === "1m" ? "1 Month" : "3 Months"}
      </span>
    </div>
  );
}

function SectorTile({
  sector,
  window,
  isSelected,
  onClick,
}: {
  sector: SectorMomentum;
  window: TimeWindow;
  isSelected: boolean;
  onClick: () => void;
}) {
  const ret = pickReturn(sector, window);
  const rs = pickRS(sector, window);

  return (
    <button
      onClick={onClick}
      className={`${heatColor(sector.momentum_score)} rounded-xl p-4 text-left transition-all
        hover:ring-2 hover:ring-blue-500/50 cursor-pointer
        ${isSelected ? "ring-2 ring-blue-500" : ""}
      `}
    >
      <div className="flex items-start justify-between mb-1">
        <div className="text-white font-medium text-sm truncate pr-1">
          {sector.sector}
        </div>
        <div className="text-xs text-gray-300 font-mono shrink-0">
          {sector.etf}
        </div>
      </div>

      {/* Return */}
      <div className={`text-lg font-mono font-bold ${retColor(ret)}`}>
        {fmtPct(ret)}
      </div>

      {/* Momentum score + RS */}
      <div className="flex items-center justify-between mt-2 text-xs">
        <span className="text-gray-300">
          Score: {sector.momentum_score.toFixed(1)}
        </span>
        <span className={`font-mono ${retColor(rs)}`}>
          RS {fmtPct(rs)}
        </span>
      </div>

      {/* Regime badge */}
      {sector.regime && (
        <div className="mt-1.5">
          <span className="inline-block text-[10px] bg-black/30 text-gray-300 rounded px-1.5 py-0.5 truncate max-w-full">
            {sector.regime}
          </span>
        </div>
      )}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Holdings Editor (shown inside SectorDetailPanel)
// ---------------------------------------------------------------------------

function HoldingsEditor({
  sectorName,
  isPowerUser,
  onHoldingsChanged,
}: {
  sectorName: string;
  isPowerUser: boolean;
  onHoldingsChanged: () => void;
}) {
  const [holdings, setHoldings] = useState<SectorHoldingItem[]>([]);
  const [source, setSource] = useState<string>("default");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState<SectorHoldingItem[]>([]);
  const [newTicker, setNewTicker] = useState("");
  const [newName, setNewName] = useState("");

  // Fetch holdings when sector changes
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setSuccessMsg(null);
    setEditing(false);

    getSectorHoldings(sectorName)
      .then((res) => {
        if (!cancelled) {
          setHoldings(res.holdings);
          setSource(res.source);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load holdings");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
  }, [sectorName]);

  // Exit editing mode if user switches away from power user
  useEffect(() => {
    if (!isPowerUser && editing) {
      setEditing(false);
      setDraft([]);
      setNewTicker("");
      setNewName("");
    }
  }, [isPowerUser, editing]);

  const startEditing = () => {
    setDraft(holdings.map((h) => ({ ...h })));
    setEditing(true);
    setSuccessMsg(null);
  };

  const cancelEditing = () => {
    setEditing(false);
    setDraft([]);
    setNewTicker("");
    setNewName("");
  };

  const addHolding = () => {
    const ticker = newTicker.trim().toUpperCase();
    const name = newName.trim();
    if (!ticker) return;
    if (draft.some((h) => h.ticker === ticker)) {
      setError(`${ticker} is already in the list`);
      return;
    }
    setDraft([...draft, { ticker, name: name || ticker }]);
    setNewTicker("");
    setNewName("");
    setError(null);
  };

  const removeHolding = (ticker: string) => {
    setDraft(draft.filter((h) => h.ticker !== ticker));
  };

  const moveHolding = (index: number, direction: -1 | 1) => {
    const newIndex = index + direction;
    if (newIndex < 0 || newIndex >= draft.length) return;
    const newDraft = [...draft];
    [newDraft[index], newDraft[newIndex]] = [newDraft[newIndex], newDraft[index]];
    setDraft(newDraft);
  };

  const saveHoldings = async () => {
    if (draft.length === 0) {
      setError("Holdings list cannot be empty");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const res = await updateSectorHoldings(sectorName, draft);
      setHoldings(res.holdings);
      setSource(res.source);
      setEditing(false);
      setDraft([]);
      setSuccessMsg("Holdings saved");
      onHoldingsChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save holdings");
    } finally {
      setSaving(false);
    }
  };

  const handleReset = async () => {
    setResetting(true);
    setError(null);
    try {
      const res = await resetSectorHoldings(sectorName);
      setHoldings(res.holdings);
      setSource(res.source);
      setEditing(false);
      setDraft([]);
      setSuccessMsg("Holdings reset to defaults");
      onHoldingsChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to reset holdings");
    } finally {
      setResetting(false);
    }
  };

  if (loading) {
    return (
      <div className="animate-pulse space-y-2">
        <div className="h-4 bg-gray-800 rounded w-1/3" />
        <div className="h-3 bg-gray-800 rounded w-2/3" />
      </div>
    );
  }

  const displayList = editing ? draft : holdings;

  return (
    <div className="space-y-3">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h4 className="text-sm font-medium text-gray-400">
            Holdings ({displayList.length})
            <HelpTip text={HELP_HOLDINGS_EDITOR} />
          </h4>
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-gray-800 text-gray-500">
            {source} <HelpTip text={HELP_HOLDINGS_SOURCE} />
          </span>
        </div>

        {isPowerUser && !editing && (
          <div className="flex items-center gap-2">
            {source === "configured" && (
              <button
                onClick={handleReset}
                disabled={resetting}
                className="px-2 py-1 text-xs text-gray-400 hover:text-white
                           bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg
                           transition-colors disabled:opacity-50"
              >
                {resetting ? "Resetting..." : "Reset to Defaults"}
              </button>
            )}
            <button
              onClick={startEditing}
              className="px-2 py-1 text-xs text-blue-400 hover:text-blue-300
                         bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg
                         transition-colors"
            >
              Edit Holdings
            </button>
          </div>
        )}

        {isPowerUser && editing && (
          <div className="flex items-center gap-2">
            <button
              onClick={cancelEditing}
              className="px-2 py-1 text-xs text-gray-400 hover:text-white
                         bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg
                         transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={saveHoldings}
              disabled={saving}
              className="px-2 py-1 text-xs text-emerald-400 hover:text-emerald-300
                         bg-gray-800 hover:bg-gray-700 border border-emerald-800 rounded-lg
                         transition-colors disabled:opacity-50"
            >
              {saving ? "Saving..." : "Save"}
            </button>
          </div>
        )}
      </div>

      {/* Normal user note */}
      {!isPowerUser && (
        <p className="text-xs text-gray-600">
          Holdings can be customized by Power Users — toggle Power User mode
          in the header.
        </p>
      )}

      {/* Error / success */}
      {error && (
        <p className="text-xs text-red-400">{error}</p>
      )}
      {successMsg && (
        <p className="text-xs text-emerald-400">{successMsg}</p>
      )}

      {/* Holdings list */}
      <div className="space-y-1 max-h-64 overflow-y-auto">
        {displayList.map((h, i) => (
          <div
            key={`${h.ticker}-${i}`}
            className="flex items-center justify-between bg-gray-800/40 rounded px-3 py-1.5 text-sm"
          >
            <div className="flex items-center gap-2 min-w-0">
              <span className="text-white font-medium font-mono text-xs shrink-0">
                {h.ticker}
              </span>
              <span className="text-gray-500 text-xs truncate">{h.name}</span>
            </div>
            {editing && (
              <div className="flex items-center gap-1 shrink-0 ml-2">
                <button
                  onClick={() => moveHolding(i, -1)}
                  disabled={i === 0}
                  className="text-gray-500 hover:text-white disabled:opacity-30 p-0.5"
                  title="Move up"
                >
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                  </svg>
                </button>
                <button
                  onClick={() => moveHolding(i, 1)}
                  disabled={i === displayList.length - 1}
                  className="text-gray-500 hover:text-white disabled:opacity-30 p-0.5"
                  title="Move down"
                >
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
                <button
                  onClick={() => removeHolding(h.ticker)}
                  className="text-red-500 hover:text-red-400 p-0.5 ml-1"
                  title="Remove"
                >
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Add new holding (editing mode) */}
      {editing && (
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={newTicker}
            onChange={(e) => setNewTicker(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") addHolding();
            }}
            placeholder="TICKER"
            className="w-24 px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded
                       text-white placeholder-gray-600 focus:border-blue-500 focus:outline-none
                       font-mono uppercase"
          />
          <input
            type="text"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") addHolding();
            }}
            placeholder="Company Name (optional)"
            className="flex-1 px-2 py-1 text-xs bg-gray-800 border border-gray-700 rounded
                       text-white placeholder-gray-600 focus:border-blue-500 focus:outline-none"
          />
          <button
            onClick={addHolding}
            disabled={!newTicker.trim()}
            className="px-2 py-1 text-xs text-blue-400 hover:text-blue-300
                       bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded
                       transition-colors disabled:opacity-30"
          >
            Add
          </button>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sector Detail Panel
// ---------------------------------------------------------------------------

function SectorDetailPanel({
  detail,
  loading,
  window,
  onDetailUpdate,
  isPowerUser,
}: {
  detail: SectorDetailResponse | null;
  loading: boolean;
  window: TimeWindow;
  onDetailUpdate?: (d: SectorDetailResponse) => void;
  isPowerUser: boolean;
}) {
  const [refreshing, setRefreshing] = useState(false);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const [refreshed, setRefreshed] = useState(false);
  const [holdingsKey, setHoldingsKey] = useState(0);

  const handleRefreshHoldings = async () => {
    if (!detail || refreshing) return;
    setRefreshing(true);
    setRefreshError(null);
    try {
      const updated = await refreshSectorHoldings(detail.sector);
      onDetailUpdate?.(updated);
      setRefreshed(true);
      // Bump key to force HoldingsEditor to re-fetch
      setHoldingsKey((k) => k + 1);
    } catch (err) {
      setRefreshError(err instanceof Error ? err.message : "Refresh failed");
    } finally {
      setRefreshing(false);
    }
  };

  // Reset refreshed flag when sector changes
  useEffect(() => {
    setRefreshed(false);
    setRefreshError(null);
    setHoldingsKey(0);
  }, [detail?.sector]);

  if (loading) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 animate-pulse">
        <div className="h-5 bg-gray-800 rounded w-1/3 mb-4" />
        <div className="h-4 bg-gray-800 rounded w-2/3 mb-2" />
        <div className="h-4 bg-gray-800 rounded w-1/2" />
      </div>
    );
  }

  if (!detail) return null;

  const ret =
    window === "1w" ? detail.return_1w : window === "1m" ? detail.return_1m : detail.return_3m;
  const rs =
    window === "1w" ? detail.rs_1w : window === "1m" ? detail.rs_1m : detail.rs_3m;

  const handleHoldingsChanged = () => {
    // Re-fetch the sector detail to pick up new movers
    if (detail) {
      getSectorDetail(detail.sector).then((d) => onDetailUpdate?.(d)).catch(() => {});
    }
  };

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-white">{detail.sector}</h3>
          <span className="text-sm text-gray-500">
            ETF: {detail.etf}
            <HelpTip text={HELP_SECTOR_ETF} />
          </span>
        </div>
        <div className="text-right">
          <div className={`text-xl font-mono font-bold ${retColor(ret)}`}>
            {fmtPct(ret)}
          </div>
          <div className="text-xs text-gray-500">
            {window === "1w" ? "1 Week" : window === "1m" ? "1 Month" : "3 Months"}
          </div>
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
        <div>
          <div className="text-gray-500">
            Momentum
            <HelpTip text={HELP_MOMENTUM_SCORE} />
          </div>
          <div className="text-white font-mono font-medium">
            {detail.momentum_score.toFixed(1)} / 10
          </div>
        </div>
        <div>
          <div className="text-gray-500">
            Rel. Strength
            <HelpTip text={HELP_RELATIVE_STRENGTH} />
          </div>
          <div className={`font-mono font-medium ${retColor(rs)}`}>
            {fmtPct(rs)}
          </div>
        </div>
        <div>
          <div className="text-gray-500">
            Regime
            <HelpTip text={HELP_SECTOR_REGIME} />
          </div>
          <div className="text-white">{detail.regime || "N/A"}</div>
        </div>
        <div>
          <div className="text-gray-500">Confidence</div>
          <div className="text-white font-mono">
            {(detail.regime_confidence * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      {/* Return windows */}
      <div className="grid grid-cols-3 gap-3 text-sm">
        <div className="bg-gray-800/50 rounded-lg p-3">
          <div className="text-gray-500 text-xs">
            1 Week <HelpTip text={HELP_RETURN_1W} />
          </div>
          <div className={`font-mono font-medium ${retColor(detail.return_1w)}`}>
            {fmtPct(detail.return_1w)}
          </div>
        </div>
        <div className="bg-gray-800/50 rounded-lg p-3">
          <div className="text-gray-500 text-xs">
            1 Month <HelpTip text={HELP_RETURN_1M} />
          </div>
          <div className={`font-mono font-medium ${retColor(detail.return_1m)}`}>
            {fmtPct(detail.return_1m)}
          </div>
        </div>
        <div className="bg-gray-800/50 rounded-lg p-3">
          <div className="text-gray-500 text-xs">
            3 Months <HelpTip text={HELP_RETURN_3M} />
          </div>
          <div className={`font-mono font-medium ${retColor(detail.return_3m)}`}>
            {fmtPct(detail.return_3m)}
          </div>
        </div>
      </div>

      {/* Holdings section */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-medium text-gray-400 flex items-center gap-1">
            Sector Holdings
            <HelpTip text={HELP_TOP_MOVERS} />
          </h4>
          <div className="flex items-center gap-2">
            {refreshed && (
              <span className="text-xs text-emerald-400">Updated</span>
            )}
            {refreshError && (
              <span className="text-xs text-red-400">{refreshError}</span>
            )}
            <button
              onClick={handleRefreshHoldings}
              disabled={refreshing}
              className="flex items-center gap-1.5 px-2.5 py-1 text-xs text-gray-400 hover:text-white
                         bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg
                         transition-colors disabled:opacity-50"
              title="Fetch latest holdings from yfinance (may be slow)"
            >
              <svg
                className={`w-3.5 h-3.5 ${refreshing ? "animate-spin" : ""}`}
                fill="none" stroke="currentColor" viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              {refreshing ? "Refreshing..." : "Refresh from yfinance"}
            </button>
          </div>
        </div>
        {refreshing && (
          <p className="text-xs text-amber-400/70 mb-2">
            Fetching live holdings from yfinance \u2014 this may take a few seconds...
          </p>
        )}

        {/* Holdings editor / viewer */}
        <HoldingsEditor
          key={`${detail.sector}-${holdingsKey}`}
          sectorName={detail.sector}
          isPowerUser={isPowerUser}
          onHoldingsChanged={handleHoldingsChanged}
        />
      </div>

      {/* Top & Worst movers */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Top movers */}
        <div>
          <h4 className="text-sm font-medium text-gray-400 mb-2">
            Top Movers (1M)
            <HelpTip text={HELP_TOP_MOVERS} />
          </h4>
          {detail.top_movers.length === 0 ? (
            <p className="text-xs text-gray-600">No data</p>
          ) : (
            <div className="space-y-1">
              {detail.top_movers.map((m) => (
                <div
                  key={m.ticker}
                  className="flex items-center justify-between bg-gray-800/40 rounded px-3 py-1.5 text-sm"
                >
                  <div>
                    <span className="text-white font-medium">{m.ticker}</span>
                    <span className="text-gray-500 ml-2 text-xs">{m.name}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className={`font-mono text-xs ${retColor(m.return_1m)}`}>
                      {fmtPct(m.return_1m)}
                    </span>
                    <span className="text-gray-400 font-mono text-xs">
                      ${m.current_price.toFixed(2)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Worst movers */}
        <div>
          <h4 className="text-sm font-medium text-gray-400 mb-2">
            Worst Movers (1M)
            <HelpTip text={HELP_WORST_MOVERS} />
          </h4>
          {detail.worst_movers.length === 0 ? (
            <p className="text-xs text-gray-600">No data</p>
          ) : (
            <div className="space-y-1">
              {detail.worst_movers.map((m) => (
                <div
                  key={m.ticker}
                  className="flex items-center justify-between bg-gray-800/40 rounded px-3 py-1.5 text-sm"
                >
                  <div>
                    <span className="text-white font-medium">{m.ticker}</span>
                    <span className="text-gray-500 ml-2 text-xs">{m.name}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className={`font-mono text-xs ${retColor(m.return_1m)}`}>
                      {fmtPct(m.return_1m)}
                    </span>
                    <span className="text-gray-400 font-mono text-xs">
                      ${m.current_price.toFixed(2)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function SectorsPage() {
  const { user } = useAuth();
  const [overview, setOverview] = useState<SectorOverviewResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Default time window based on trade mode: swing \u2192 1w, long_term \u2192 3m
  const [window, setWindow] = useState<TimeWindow>("1m");
  const windowSetByUser = useRef(false);
  const [selectedSector, setSelectedSector] = useState<string | null>(null);
  const [detail, setDetail] = useState<SectorDetailResponse | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const detailRef = useRef<HTMLDivElement>(null);

  const isPowerUser = user?.user_mode === "power_user";

  // Sync default window with trade mode on initial load
  useEffect(() => {
    if (user && !windowSetByUser.current) {
      const tw: TimeWindow =
        user.trade_mode === "long_term" ? "3m" : user.trade_mode === "swing" ? "1w" : "1m";
      setWindow(tw);
    }
  }, [user]);

  // Fetch overview on mount
  const fetchOverview = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getSectorOverview();
      setOverview(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load sectors");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchOverview();
  }, [fetchOverview]);

  // Fetch detail when a sector is selected
  useEffect(() => {
    if (!selectedSector) {
      setDetail(null);
      return;
    }

    let cancelled = false;
    setDetailLoading(true);

    getSectorDetail(selectedSector)
      .then((data) => {
        if (!cancelled) setDetail(data);
      })
      .catch(() => {
        if (!cancelled) setDetail(null);
      })
      .finally(() => {
        if (!cancelled) setDetailLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedSector]);

  // Handle tile click (toggle)
  const handleTileClick = (sectorName: string) => {
    setSelectedSector((prev) => (prev === sectorName ? null : sectorName));
  };

  // Scroll to detail panel when a sector is selected
  useEffect(() => {
    if (selectedSector && detailRef.current) {
      // Small delay so the panel renders before scrolling
      setTimeout(() => {
        detailRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 100);
    }
  }, [selectedSector]);

  return (
    <div className="space-y-5">
      {/* Header */}
      <div>
        <h2 className="text-xl font-semibold text-white">
          Sectors &amp; Segments
          <HelpTip text={HELP_SECTORS} />
        </h2>
        <p className="text-sm text-gray-500 mt-1">
          GICS-based sector heatmap, rotation tracker, and relative strength
          analysis.
        </p>
      </div>

      {/* Controls row */}
      <div className="flex flex-wrap items-center gap-4">
        {/* Time window selector */}
        <div className="flex items-center gap-1 bg-gray-900 border border-gray-800 rounded-lg p-1">
          {(["1w", "1m", "3m"] as TimeWindow[]).map((w) => (
            <button
              key={w}
              onClick={() => { windowSetByUser.current = true; setWindow(w); }}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors
                ${
                  window === w
                    ? "bg-blue-600 text-white"
                    : "text-gray-400 hover:text-white hover:bg-gray-800"
                }
              `}
            >
              {w === "1w" ? "1W" : w === "1m" ? "1M" : "3M"}
            </button>
          ))}
          <HelpTip text={HELP_SECTOR_ROTATION} />
        </div>

        {/* Benchmark */}
        {overview && <BenchmarkBar data={overview} window={window} />}

        {/* Refresh button */}
        <button
          onClick={fetchOverview}
          disabled={loading}
          className="ml-auto px-3 py-1.5 rounded-lg bg-gray-800 border border-gray-700
                     text-sm text-gray-300 hover:bg-gray-700 hover:text-white
                     disabled:opacity-50 transition-colors"
        >
          {loading ? "Loading..." : "Refresh"}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-900/20 border border-red-800 rounded-lg p-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* Loading skeleton */}
      {loading && !overview && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
          {Array.from({ length: 11 }).map((_, i) => (
            <div
              key={i}
              className="bg-gray-800 rounded-xl p-4 animate-pulse h-28"
            />
          ))}
        </div>
      )}

      {/* Heatmap */}
      {overview && (
        <>
          <div>
            <h3 className="text-sm font-medium text-gray-400 mb-2">
              Sector Heatmap
              <HelpTip text={HELP_SECTOR_HEATMAP} />
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
              {overview.sectors.map((s) => (
                <SectorTile
                  key={s.etf}
                  sector={s}
                  window={window}
                  isSelected={selectedSector === s.sector}
                  onClick={() => handleTileClick(s.sector)}
                />
              ))}
            </div>
          </div>

          {/* Info line */}
          <p className="text-xs text-gray-600">
            Data from {overview.sectors.length} SPDR sector ETFs
            <HelpTip text={HELP_SECTOR_ETF} />
            {" "}\u00b7 Loaded in {overview.elapsed_seconds.toFixed(1)}s
            {" "}\u00b7 Click a sector for details
          </p>
        </>
      )}

      {/* Rotation table */}
      {overview && (
        <div>
          <h3 className="text-sm font-medium text-gray-400 mb-2">
            Sector Rotation Tracker
            <HelpTip text={HELP_SECTOR_ROTATION} />
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 text-xs border-b border-gray-800">
                  <th className="text-left py-2 pr-3">Sector</th>
                  <th className="text-right py-2 px-2">
                    1W <HelpTip text={HELP_RETURN_1W} />
                  </th>
                  <th className="text-right py-2 px-2">
                    1M <HelpTip text={HELP_RETURN_1M} />
                  </th>
                  <th className="text-right py-2 px-2">
                    3M <HelpTip text={HELP_RETURN_3M} />
                  </th>
                  <th className="text-right py-2 px-2">
                    RS (1M) <HelpTip text={HELP_RELATIVE_STRENGTH} />
                  </th>
                  <th className="text-right py-2 px-2">
                    Score <HelpTip text={HELP_MOMENTUM_SCORE} />
                  </th>
                  <th className="text-left py-2 pl-3">
                    Regime <HelpTip text={HELP_SECTOR_REGIME} />
                  </th>
                </tr>
              </thead>
              <tbody>
                {overview.sectors.map((s) => (
                  <tr
                    key={s.etf}
                    onClick={() => handleTileClick(s.sector)}
                    className={`border-b border-gray-800/50 cursor-pointer hover:bg-gray-800/30 transition-colors
                      ${selectedSector === s.sector ? "bg-gray-800/50" : ""}
                    `}
                  >
                    <td className="py-2 pr-3">
                      <span className="text-white font-medium">{s.sector}</span>
                      <span className="text-gray-600 ml-2 text-xs">{s.etf}</span>
                    </td>
                    <td className={`text-right py-2 px-2 font-mono ${retColor(s.return_1w)}`}>
                      {fmtPct(s.return_1w)}
                    </td>
                    <td className={`text-right py-2 px-2 font-mono ${retColor(s.return_1m)}`}>
                      {fmtPct(s.return_1m)}
                    </td>
                    <td className={`text-right py-2 px-2 font-mono ${retColor(s.return_3m)}`}>
                      {fmtPct(s.return_3m)}
                    </td>
                    <td className={`text-right py-2 px-2 font-mono ${retColor(s.rs_1m)}`}>
                      {fmtPct(s.rs_1m)}
                    </td>
                    <td className="text-right py-2 px-2 font-mono text-white">
                      {s.momentum_score.toFixed(1)}
                    </td>
                    <td className="py-2 pl-3 text-gray-400 text-xs">
                      {s.regime || "\u2014"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Detail panel */}
      {(selectedSector || detailLoading) && (
        <div ref={detailRef}>
          <SectorDetailPanel
            detail={detail}
            loading={detailLoading}
            window={window}
            onDetailUpdate={setDetail}
            isPowerUser={isPowerUser}
          />
        </div>
      )}
    </div>
  );
}
