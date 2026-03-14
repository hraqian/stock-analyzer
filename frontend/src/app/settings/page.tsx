"use client";

import { useAuth } from "@/contexts/AuthContext";

export default function SettingsPage() {
  const { user } = useAuth();

  return (
    <div>
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-white">Settings</h2>
        <p className="text-sm text-gray-500 mt-1">
          Account profile, cost model, data sources, and AI configuration.
        </p>
      </div>

      {/* Profile overview — basic placeholder showing current settings */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Account</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-500">Username</span>
              <span className="text-gray-300">{user?.username ?? "—"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Trade Mode</span>
              <span className="text-gray-300">{user?.trade_mode ?? "—"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">User Mode</span>
              <span className="text-gray-300">{user?.user_mode ?? "—"}</span>
            </div>
          </div>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Cost Model</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-500">Starting Capital</span>
              <span className="text-gray-300">
                ${user?.starting_capital?.toLocaleString() ?? "—"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Commission / Trade</span>
              <span className="text-gray-300">
                ${user?.commission_per_trade ?? "—"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Spread</span>
              <span className="text-gray-300">
                {user?.spread_pct != null ? `${user.spread_pct}%` : "—"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Slippage</span>
              <span className="text-gray-300">
                {user?.slippage_pct != null ? `${user.slippage_pct}%` : "—"}
              </span>
            </div>
          </div>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Tax Rates</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-500">Short-Term</span>
              <span className="text-gray-300">
                {user?.tax_rate_short_term != null
                  ? `${user.tax_rate_short_term}%`
                  : "—"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Long-Term</span>
              <span className="text-gray-300">
                {user?.tax_rate_long_term != null
                  ? `${user.tax_rate_long_term}%`
                  : "—"}
              </span>
            </div>
          </div>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">
            Data &amp; AI
          </h3>
          <div className="space-y-2 text-sm text-gray-600">
            <p>Data provider and AI configuration coming later.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
