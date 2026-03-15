/**
 * Shared formatting utilities for the frontend.
 *
 * Centralises number/percent/money formatting that was previously
 * duplicated across strategy, analysis, and scanner components.
 */

/** Format a number as a percentage with sign: +12.34% or -5.67% */
export function fmtPct(v: number | null | undefined): string {
  if (v == null) return "\u2014";
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
}

/** Format a raw number to fixed decimals: 1.23 */
export function fmtNum(v: number | null | undefined, decimals = 2): string {
  if (v == null) return "\u2014";
  return v.toFixed(decimals);
}

/** Format a dollar amount: $100,000 */
export function fmtMoney(v: number | null | undefined): string {
  if (v == null) return "\u2014";
  return `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

/** Format a decimal as a percentage (0.05 → +5.00%). Used for sector returns. */
export function fmtDecimalPct(v: number): string {
  const pct = v * 100;
  const sign = pct >= 0 ? "+" : "";
  return `${sign}${pct.toFixed(2)}%`;
}

/** Format large numbers with suffixes: $1.23T, $4.56B, $7.8M */
export function fmtMarketCap(n: unknown): string {
  if (n == null) return "N/A";
  const v = Number(n);
  if (isNaN(v)) return "N/A";
  if (v >= 1e12) return `$${(v / 1e12).toFixed(2)}T`;
  if (v >= 1e9) return `$${(v / 1e9).toFixed(2)}B`;
  if (v >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
  return `$${v.toLocaleString()}`;
}
