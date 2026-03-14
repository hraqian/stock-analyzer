"use client";

export default function ScannerPage() {
  return (
    <div>
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-white">Market Scanner</h2>
        <p className="text-sm text-gray-500 mt-1">
          Scan configurable universes for trade candidates using technical filters,
          pattern detection, and AI-powered ranking.
        </p>
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-xl p-8 text-center">
        <div className="text-gray-600 text-sm">
          <svg
            className="w-12 h-12 mx-auto mb-3 text-gray-700"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
          <p className="text-gray-400 font-medium">Coming in Phase 2</p>
          <p className="mt-1">
            Universe selection, filter configuration, scan results with
            sortable columns, and AI-ranked candidates.
          </p>
        </div>
      </div>
    </div>
  );
}
