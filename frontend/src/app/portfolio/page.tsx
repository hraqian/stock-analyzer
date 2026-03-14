"use client";

export default function PortfolioPage() {
  return (
    <div>
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-white">Portfolio Simulation</h2>
        <p className="text-sm text-gray-500 mt-1">
          Activated strategies generate trade signals. Confirm or skip trades,
          manage capital allocation, and track simulated P&amp;L.
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
              d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"
            />
          </svg>
          <p className="text-gray-400 font-medium">Coming in Phase 4</p>
          <p className="mt-1">
            Signal queue, trade confirmation workflow, position sizing,
            capital allocation, and equity curve tracking.
          </p>
        </div>
      </div>
    </div>
  );
}
