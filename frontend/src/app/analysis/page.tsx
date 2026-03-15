"use client";

import { Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { useCallback } from "react";
import SingleStockTab from "./SingleStockTab";
import SectorsTab from "./SectorsTab";

type AnalysisTab = "stock" | "sectors";

function AnalysisContent() {
  const searchParams = useSearchParams();
  const router = useRouter();

  // Determine active tab from URL params (default: "stock")
  const tabParam = searchParams.get("tab");
  const activeTab: AnalysisTab =
    tabParam === "sectors" ? "sectors" : "stock";

  // Ticker param for single-stock auto-run (e.g. from scanner)
  const initialTicker = searchParams.get("ticker") || undefined;

  // Switch tab by updating URL params (preserves other params)
  const switchTab = useCallback(
    (tab: AnalysisTab) => {
      const params = new URLSearchParams(searchParams.toString());
      if (tab === "stock") {
        params.delete("tab");
      } else {
        params.set("tab", tab);
      }
      // Clear ticker when switching to sectors (it's not relevant there)
      if (tab === "sectors") {
        params.delete("ticker");
      }
      router.push(`/analysis?${params.toString()}`);
    },
    [searchParams, router]
  );

  // Called from SectorsTab when user clicks "Analyze" on a ticker/ETF
  // Switches to single stock tab with that ticker
  const handleAnalyzeTicker = useCallback(
    (ticker: string) => {
      router.push(`/analysis?ticker=${encodeURIComponent(ticker)}`);
    },
    [router]
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-semibold text-white">Analysis</h2>
        <p className="text-sm text-gray-500 mt-1">
          Deep-dive into individual stocks or explore sector-level trends.
        </p>
      </div>

      {/* Tab bar */}
      <div className="flex items-center gap-1 bg-gray-900 border border-gray-800 rounded-lg p-1 w-fit">
        <button
          onClick={() => switchTab("stock")}
          className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors
            ${
              activeTab === "stock"
                ? "bg-blue-600 text-white"
                : "text-gray-400 hover:text-white hover:bg-gray-800"
            }
          `}
        >
          Single Stock
        </button>
        <button
          onClick={() => switchTab("sectors")}
          className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors
            ${
              activeTab === "sectors"
                ? "bg-blue-600 text-white"
                : "text-gray-400 hover:text-white hover:bg-gray-800"
            }
          `}
        >
          Sectors
        </button>
      </div>

      {/* Active tab content */}
      {activeTab === "stock" ? (
        <SingleStockTab initialTicker={initialTicker} />
      ) : (
        <SectorsTab onAnalyzeTicker={handleAnalyzeTicker} />
      )}
    </div>
  );
}

export default function AnalysisPage() {
  return (
    <Suspense fallback={<div className="text-gray-500 text-sm p-4">Loading...</div>}>
      <AnalysisContent />
    </Suspense>
  );
}
