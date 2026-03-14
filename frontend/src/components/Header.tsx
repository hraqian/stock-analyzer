"use client";

import { useAuth } from "@/contexts/AuthContext";

const TRADE_MODES = [
  { value: "swing", label: "Swing Trade" },
  { value: "day_trade", label: "Day Trade", disabled: true, badge: "Coming Soon" },
  { value: "long_term", label: "Long-Term" },
];

const USER_MODES = [
  { value: "normal", label: "Normal" },
  { value: "power_user", label: "Power User" },
];

export function Header() {
  const { user, updateUser, logout } = useAuth();

  if (!user) return null;

  const handleTradeMode = async (mode: string) => {
    if (mode === "day_trade") return;
    await updateUser({ trade_mode: mode });
  };

  const handleUserMode = async (mode: string) => {
    await updateUser({ user_mode: mode });
  };

  return (
    <header className="h-14 bg-gray-900 border-b border-gray-800 flex items-center justify-between px-5">
      {/* Left: Trade mode toggle */}
      <div className="flex items-center gap-1">
        <span className="text-xs text-gray-500 mr-2 font-medium">Mode:</span>
        {TRADE_MODES.map((mode) => (
          <button
            key={mode.value}
            onClick={() => handleTradeMode(mode.value)}
            disabled={mode.disabled}
            className={`
              relative px-3 py-1.5 rounded-md text-xs font-medium transition-colors
              ${mode.disabled
                ? "text-gray-600 cursor-not-allowed"
                : user.trade_mode === mode.value
                  ? "bg-blue-600 text-white"
                  : "text-gray-400 hover:bg-gray-800 hover:text-gray-200"
              }
            `}
            title={mode.disabled ? "Coming Soon — requires paid data provider" : ""}
          >
            {mode.label}
            {mode.badge && (
              <span className="absolute -top-1.5 -right-1 text-[8px] bg-amber-600/80 text-amber-100 px-1 rounded">
                {mode.badge}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Right: User mode + account */}
      <div className="flex items-center gap-4">
        {/* User mode toggle */}
        <div className="flex items-center gap-1">
          {USER_MODES.map((mode) => (
            <button
              key={mode.value}
              onClick={() => handleUserMode(mode.value)}
              className={`
                px-2.5 py-1 rounded-md text-xs font-medium transition-colors
                ${user.user_mode === mode.value
                  ? "bg-gray-700 text-gray-200"
                  : "text-gray-500 hover:bg-gray-800 hover:text-gray-300"
                }
              `}
            >
              {mode.label}
            </button>
          ))}
        </div>

        {/* Divider */}
        <div className="w-px h-6 bg-gray-800" />

        {/* User info + logout */}
        <div className="flex items-center gap-3">
          <span className="text-sm text-gray-400">{user.username}</span>
          <button
            onClick={logout}
            className="text-xs text-gray-600 hover:text-gray-400 transition-colors"
          >
            Logout
          </button>
        </div>
      </div>
    </header>
  );
}
