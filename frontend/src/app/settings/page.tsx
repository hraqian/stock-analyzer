"use client";

import { useState, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import type { User } from "@/lib/api";
import HelpTip from "@/components/HelpTip";
import {
  HELP_TAX_PROVINCE,
  HELP_TAX_ANNUAL_INCOME,
  HELP_TAX_TREATMENT,
  HELP_TAX_MARGINAL_RATE,
  HELP_LLM_PROVIDER,
  HELP_LLM_API_KEY,
  HELP_LLM_MODEL,
} from "@/lib/helpText";

/** Province options for the dropdown. */
const PROVINCES: { code: string; name: string }[] = [
  { code: "AB", name: "Alberta" },
  { code: "BC", name: "British Columbia" },
  { code: "MB", name: "Manitoba" },
  { code: "NB", name: "New Brunswick" },
  { code: "NL", name: "Newfoundland & Labrador" },
  { code: "NS", name: "Nova Scotia" },
  { code: "NT", name: "Northwest Territories" },
  { code: "NU", name: "Nunavut" },
  { code: "ON", name: "Ontario" },
  { code: "PE", name: "Prince Edward Island" },
  { code: "QC", name: "Quebec" },
  { code: "SK", name: "Saskatchewan" },
  { code: "YT", name: "Yukon" },
];

const TREATMENT_OPTIONS = [
  { value: "auto", label: "Auto-detect" },
  { value: "capital_gains", label: "Capital Gains" },
  { value: "business_income", label: "Business Income" },
];

export default function SettingsPage() {
  const { user, updateUser } = useAuth();

  // Tax form state
  const [taxProvince, setTaxProvince] = useState<string>("");
  const [taxIncome, setTaxIncome] = useState<string>("");
  const [taxTreatment, setTaxTreatment] = useState<string>("auto");
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);

  // LLM provider state
  const [llmProvider, setLlmProvider] = useState<string>("gemini");
  const [llmApiKey, setLlmApiKey] = useState<string>("");
  const [llmModel, setLlmModel] = useState<string>("");
  const [savingLlm, setSavingLlm] = useState(false);
  const [llmMsg, setLlmMsg] = useState<string | null>(null);

  // Sync form state when user loads
  useEffect(() => {
    if (user) {
      setTaxProvince(user.tax_province ?? "");
      setTaxIncome(user.tax_annual_income ? String(user.tax_annual_income) : "");
      setTaxTreatment(user.tax_treatment ?? "auto");
      setLlmProvider(user.llm_provider ?? "gemini");
      // Don't overwrite the key input with the masked value if user is typing
      if (!llmApiKey) {
        setLlmApiKey(user.llm_api_key ?? "");
      }
      setLlmModel(user.llm_model ?? "");
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  async function handleSaveTax() {
    setSaving(true);
    setSaveMsg(null);
    try {
      const incomeNum = taxIncome ? parseFloat(taxIncome) : 0;
      if (isNaN(incomeNum) || incomeNum < 0) {
        setSaveMsg("Income must be a positive number.");
        setSaving(false);
        return;
      }
      await updateUser({
        tax_province: taxProvince || null,
        tax_annual_income: incomeNum,
        tax_treatment: taxTreatment,
      } as Partial<User>);
      setSaveMsg("Saved!");
      setTimeout(() => setSaveMsg(null), 3000);
    } catch (e: unknown) {
      setSaveMsg(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  const taxEnabled = !!taxProvince && parseFloat(taxIncome || "0") > 0;

  return (
    <div>
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-white">Settings</h2>
        <p className="text-sm text-gray-500 mt-1">
          Account profile, cost model, tax settings, and data sources.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Account card (read-only) */}
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

        {/* Cost Model card (read-only) */}
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

        {/* Canadian Tax Settings card (editable) */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 md:col-span-2">
          <div className="flex items-center gap-2 mb-4">
            <h3 className="text-sm font-semibold text-gray-300">
              Canadian Tax Settings
            </h3>
            <span
              className={`text-xs px-2 py-0.5 rounded-full ${
                taxEnabled
                  ? "bg-green-900/50 text-green-400"
                  : "bg-gray-800 text-gray-500"
              }`}
            >
              {taxEnabled ? "Enabled" : "Disabled"}
            </span>
          </div>

          <p className="text-xs text-gray-500 mb-4">
            When enabled, backtest results deduct Canadian income tax from each
            profitable trade. The auto-tuner optimises for after-tax returns.
            Set your province and income to enable.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {/* Province */}
            <div>
              <div className="flex items-center gap-1 mb-1">
                <label
                  htmlFor="tax-province"
                  className="text-xs text-gray-400"
                >
                  Province / Territory
                </label>
                <HelpTip text={HELP_TAX_PROVINCE} size={12} />
              </div>
              <select
                id="tax-province"
                value={taxProvince}
                onChange={(e) => setTaxProvince(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
              >
                <option value="">— Not set —</option>
                {PROVINCES.map((p) => (
                  <option key={p.code} value={p.code}>
                    {p.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Annual Income */}
            <div>
              <div className="flex items-center gap-1 mb-1">
                <label
                  htmlFor="tax-income"
                  className="text-xs text-gray-400"
                >
                  Annual Income (CAD)
                </label>
                <HelpTip text={HELP_TAX_ANNUAL_INCOME} size={12} />
              </div>
              <div className="relative">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500 text-sm">
                  $
                </span>
                <input
                  id="tax-income"
                  type="text"
                  inputMode="numeric"
                  value={taxIncome}
                  onChange={(e) => {
                    // Allow only digits and one decimal point
                    const v = e.target.value.replace(/[^0-9.]/g, "");
                    setTaxIncome(v);
                  }}
                  placeholder="85000"
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg pl-7 pr-3 py-2 text-sm text-gray-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
              </div>
            </div>

            {/* Tax Treatment */}
            <div>
              <div className="flex items-center gap-1 mb-1">
                <label
                  htmlFor="tax-treatment"
                  className="text-xs text-gray-400"
                >
                  Tax Treatment
                </label>
                <HelpTip text={HELP_TAX_TREATMENT} size={12} />
              </div>
              <select
                id="tax-treatment"
                value={taxTreatment}
                onChange={(e) => setTaxTreatment(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
              >
                {TREATMENT_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Marginal rate display */}
          {taxEnabled && (
            <div className="mt-3 flex items-center gap-1 text-xs text-gray-400">
              <span>
                Your marginal tax rate will be calculated from {" "}
                {PROVINCES.find((p) => p.code === taxProvince)?.name ?? taxProvince}
                {" "} federal + provincial brackets at ${parseFloat(taxIncome || "0").toLocaleString()} income.
              </span>
              <HelpTip text={HELP_TAX_MARGINAL_RATE} size={12} />
            </div>
          )}

          {/* Save button + message */}
          <div className="mt-4 flex items-center gap-3">
            <button
              onClick={handleSaveTax}
              disabled={saving}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 text-white text-sm font-medium rounded-lg transition-colors"
            >
              {saving ? "Saving..." : "Save Tax Settings"}
            </button>
            {saveMsg && (
              <span
                className={`text-sm ${
                  saveMsg === "Saved!" ? "text-green-400" : "text-red-400"
                }`}
              >
                {saveMsg}
              </span>
            )}
          </div>
        </div>

        {/* AI / LLM Settings card */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">
            AI Analysis
          </h3>
          <p className="text-xs text-gray-500 mb-4">
            Choose which LLM provider powers the qualitative AI analysis on
            the Analysis page. You can add your own API key here, or fall
            back to the server&apos;s environment variable if one is set.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
            {/* Provider */}
            <div>
              <div className="flex items-center gap-1 mb-1">
                <span className="text-xs text-gray-400">LLM Provider</span>
                <HelpTip text={HELP_LLM_PROVIDER} size={12} />
              </div>
              <select
                value={llmProvider}
                onChange={(e) => {
                  setLlmProvider(e.target.value);
                  setLlmModel("");  // reset model when provider changes
                }}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
              >
                <option value="gemini">Google Gemini (Free tier)</option>
                <option value="anthropic">Anthropic (Claude)</option>
                <option value="openai">OpenAI (GPT)</option>
              </select>
            </div>

            {/* API Key */}
            <div>
              <div className="flex items-center gap-1 mb-1">
                <span className="text-xs text-gray-400">API Key</span>
                <HelpTip text={HELP_LLM_API_KEY} size={12} />
              </div>
              <input
                type="password"
                value={llmApiKey}
                onChange={(e) => setLlmApiKey(e.target.value)}
                placeholder={user?.llm_api_key ? user.llm_api_key : "Paste your key"}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:ring-1 focus:ring-blue-500 font-mono"
              />
              <p className="text-xs text-gray-600 mt-1">
                {user?.llm_api_key ? "Key configured" : "No key set"} — leave blank to keep current
              </p>
            </div>

            {/* Model */}
            <div>
              <div className="flex items-center gap-1 mb-1">
                <span className="text-xs text-gray-400">Model</span>
                <HelpTip text={HELP_LLM_MODEL} size={12} />
              </div>
              <select
                value={llmModel}
                onChange={(e) => setLlmModel(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
              >
                <option value="">Provider default</option>
                {llmProvider === "gemini" && (
                  <>
                    <option value="gemini-2.5-flash">gemini-2.5-flash</option>
                    <option value="gemini-3.1-flash-lite-preview">gemini-3.1-flash-lite-preview</option>
                  </>
                )}
                {llmProvider === "anthropic" && (
                  <>
                    <option value="claude-sonnet-4-20250514">claude-sonnet-4-20250514</option>
                    <option value="claude-3-5-haiku-20241022">claude-3-5-haiku-20241022</option>
                  </>
                )}
                {llmProvider === "openai" && (
                  <>
                    <option value="gpt-4o-mini">gpt-4o-mini</option>
                    <option value="gpt-4o">gpt-4o</option>
                    <option value="gpt-4-turbo">gpt-4-turbo</option>
                  </>
                )}
              </select>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={async () => {
                setSavingLlm(true);
                setLlmMsg(null);
                try {
                  const updates: Partial<User> = {
                    llm_provider: llmProvider,
                  };
                  // Only send api_key if user typed something new
                  // (don't send the masked placeholder back)
                  if (llmApiKey && !llmApiKey.includes("...")) {
                    (updates as Record<string, unknown>).llm_api_key = llmApiKey;
                  }
                  // Send model (empty string -> null on backend)
                  (updates as Record<string, unknown>).llm_model = llmModel || null;
                  await updateUser(updates);
                  setLlmMsg("Saved!");
                  setLlmApiKey("");  // clear the input after save
                  setTimeout(() => setLlmMsg(null), 3000);
                } catch (e: unknown) {
                  setLlmMsg(e instanceof Error ? e.message : "Save failed");
                } finally {
                  setSavingLlm(false);
                }
              }}
              disabled={savingLlm}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 text-white text-sm font-medium rounded-lg transition-colors"
            >
              {savingLlm ? "Saving..." : "Save AI Settings"}
            </button>
            {llmMsg && (
              <span
                className={`text-sm ${
                  llmMsg === "Saved!" ? "text-green-400" : "text-red-400"
                }`}
              >
                {llmMsg}
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
