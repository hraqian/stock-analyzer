"use client";

export default function AnalysisPage() {
  return (
    <div>
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-white">Single Stock Analysis</h2>
        <p className="text-sm text-gray-500 mt-1">
          Deep-dive into one ticker with price charts, technical indicators,
          pattern recognition, regime detection, and AI commentary.
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
              d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z"
            />
          </svg>
          <p className="text-gray-400 font-medium">Coming in Phase 2</p>
          <p className="mt-1">
            Ticker search, interactive price chart, indicator overlays,
            pattern annotations, and AI-generated analysis summary.
          </p>
        </div>
      </div>
    </div>
  );
}
