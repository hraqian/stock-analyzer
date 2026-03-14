"use client";

export default function StrategyPage() {
  return (
    <div>
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-white">Strategy Lab</h2>
        <p className="text-sm text-gray-500 mt-1">
          Backtest strategies with walk-forward validation, auto-tune parameters,
          and browse the strategy library.
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
              d="M9.75 3v4.5m4.5-4.5v4.5M9.75 7.5h4.5M7.5 12l-3 9h15l-3-9M7.5 12h9"
            />
          </svg>
          <p className="text-gray-400 font-medium">Coming in Phase 3</p>
          <p className="mt-1">
            Strategy backtester with walk-forward testing, parameter auto-tuner,
            and a library of built-in strategies.
          </p>
        </div>
      </div>
    </div>
  );
}
