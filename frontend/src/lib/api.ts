/** API client for the FastAPI backend. */

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/** Stored JWT token (client-side only). */
let token: string | null = null;

export function setToken(t: string | null) {
  token = t;
  if (typeof window !== "undefined") {
    if (t) {
      localStorage.setItem("token", t);
    } else {
      localStorage.removeItem("token");
    }
  }
}

export function getToken(): string | null {
  if (token) return token;
  if (typeof window !== "undefined") {
    token = localStorage.getItem("token");
  }
  return token;
}

/** Generic fetch wrapper that injects the JWT token. */
async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string> || {}),
  };

  const t = getToken();
  if (t) {
    headers["Authorization"] = `Bearer ${t}`;
  }

  const res = await fetch(`${API_URL}${path}`, {
    ...options,
    headers,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `API error: ${res.status}`);
  }

  return res.json();
}

// -------------------------------------------------------------------
// Auth
// -------------------------------------------------------------------

export interface LoginResponse {
  access_token: string;
  token_type: string;
}

export interface User {
  id: number;
  username: string;
  trade_mode: string;
  user_mode: string;
  starting_capital: number;
  risk_tolerance: string;
  commission_per_trade: number;
  spread_pct: number;
  slippage_pct: number;
  tax_rate_short_term: number;
  tax_rate_long_term: number;
}

export async function login(
  username: string,
  password: string
): Promise<LoginResponse> {
  const data = await apiFetch<LoginResponse>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
  setToken(data.access_token);
  return data;
}

export async function register(
  username: string,
  password: string
): Promise<User> {
  return apiFetch<User>("/api/auth/register", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export async function getMe(): Promise<User> {
  return apiFetch<User>("/api/auth/me");
}

export async function updateMe(
  updates: Partial<User>
): Promise<User> {
  return apiFetch<User>("/api/auth/me", {
    method: "PATCH",
    body: JSON.stringify(updates),
  });
}

export function logout() {
  setToken(null);
}

// -------------------------------------------------------------------
// Health
// -------------------------------------------------------------------

export async function healthCheck(): Promise<{ status: string; version: string }> {
  return apiFetch("/api/health");
}

// -------------------------------------------------------------------
// Data
// -------------------------------------------------------------------

export interface Universe {
  id: string;
  name: string;
  description: string;
  count: number;
}

export async function getUniverses(): Promise<{ universes: Universe[] }> {
  return apiFetch("/api/data/universes");
}

export async function getLivePrice(
  ticker: string
): Promise<{ ticker: string; price: number; timestamp: string }> {
  return apiFetch(`/api/data/price/${ticker}`);
}

export async function getSectorInfo(
  ticker: string
): Promise<{ ticker: string; sector: string; industry: string }> {
  return apiFetch(`/api/data/sector/${ticker}`);
}

// -------------------------------------------------------------------
// Analysis
// -------------------------------------------------------------------

export interface OHLCVBar {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface IndicatorResult {
  name: string;
  config_key: string;
  score: number;
  values: Record<string, unknown>;
  display: Record<string, unknown>;
  error: string | null;
}

export interface PatternResult {
  name: string;
  config_key: string;
  score: number;
  values: Record<string, unknown>;
  display: Record<string, unknown>;
  error: string | null;
}

export interface SRLevel {
  price: number;
  level_type: string;
  source: string;
  touches: number;
  label: string;
}

export interface RegimeMetrics {
  adx: number;
  rolling_adx_mean: number;
  total_return: number;
  pct_above_ma: number;
  atr_pct: number;
  bb_width: number;
  bb_width_percentile: number;
  price_ma_distance: number;
  direction_changes: number;
  trend_direction: string;
}

export interface RegimeAssessment {
  regime: string;
  confidence: number;
  label: string;
  description: string;
  metrics: RegimeMetrics;
  reasons: string[];
  regime_scores: Record<string, number>;
  sub_type: string | null;
  sub_type_label: string;
  sub_type_description: string;
}

export interface CompositeScore {
  overall: number;
  overall_raw: number | null;
  breakdown: Record<string, number>;
  n_scored: number;
  weights_used: Record<string, number>;
  trend_score: number | null;
  contrarian_score: number | null;
  neutral_score: number | null;
  dominant_group: string | null;
}

export interface AnalysisResult {
  ticker: string;
  period: string;
  info: Record<string, unknown>;
  price_data: OHLCVBar[];
  indicators: IndicatorResult[];
  patterns: PatternResult[];
  support_levels: SRLevel[];
  resistance_levels: SRLevel[];
  composite: CompositeScore;
  pattern_composite: Record<string, unknown>;
  regime: RegimeAssessment | null;
}

export async function analyzeStock(
  ticker: string,
  period: string = "6mo",
  interval: string = "1d"
): Promise<AnalysisResult> {
  return apiFetch(
    `/api/analysis/${encodeURIComponent(ticker)}?period=${period}&interval=${interval}`
  );
}

// -------------------------------------------------------------------
// Scanner
// -------------------------------------------------------------------

export interface ScannerResultRow {
  rank: number;
  ticker: string;
  signal: string;
  score: number;
  confidence: number;
  pattern: string;
  regime: string;
  sector: string;
  breakdown: Record<string, number>;
  volume: number;
  price: number;
  atr_ratio: number;
}

export interface ScanResponse {
  preset: string;
  universe: string;
  total_tickers: number;
  tickers_with_data: number;
  tickers_passing_filters: number;
  elapsed_seconds: number;
  results: ScannerResultRow[];
}

export interface ScanRequest {
  universe: string;
  custom_tickers?: string[];
  preset: string;
  period?: string;
  min_volume?: number;
  min_price?: number;
  max_atr_ratio?: number | null;
  top_n?: number;
}

export async function runScan(req: ScanRequest): Promise<ScanResponse> {
  return apiFetch<ScanResponse>("/api/scanner/scan", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function getUniverseList(): Promise<{ universes: string[] }> {
  return apiFetch("/api/scanner/universes");
}
