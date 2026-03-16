/** API client for the FastAPI backend. */

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/** Stored JWT token (client-side only). */
let token: string | null = null;

function setToken(t: string | null) {
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
    // Handle expired/invalid JWT — redirect to login
    if (res.status === 401) {
      setToken(null);
      if (typeof window !== "undefined") {
        window.location.href = "/login";
      }
      throw new Error("Session expired. Please log in again.");
    }
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
  tax_province: string | null;
  tax_annual_income: number;
  tax_treatment: string;  // "auto" | "capital_gains" | "business_income"
  llm_provider: string;   // "anthropic" | "openai" | "gemini"
  llm_api_key: string | null;  // masked value from server (e.g. "sk-ant...****")
  llm_model: string | null;    // user-chosen model override
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
// Data (types only — unused fetch functions removed)
// -------------------------------------------------------------------

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

export interface MlScoreResult {
  ai_rating: number;
  probability: number;
  label: string;
  confidence: string;
  top_features: Array<{
    name: string;
    value: number;
    importance: number;
  }>;
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
  ai_analysis: string | null;
  ml_score: MlScoreResult | null;
}

export async function analyzeStock(
  ticker: string,
  period: string = "6mo",
  interval: string = "1d",
  ai: boolean = false
): Promise<AnalysisResult> {
  const params = new URLSearchParams({ period, interval });
  if (ai) params.set("ai", "true");
  return apiFetch(
    `/api/analysis/${encodeURIComponent(ticker)}?${params.toString()}`
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
  ai_rating: number | null;
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



// -------------------------------------------------------------------
// Sectors
// -------------------------------------------------------------------

export interface SectorMomentum {
  etf: string;
  sector: string;
  return_1w: number;
  return_1m: number;
  return_3m: number;
  rs_1w: number;
  rs_1m: number;
  rs_3m: number;
  current_price: number;
  avg_volume: number;
  regime: string;
  regime_confidence: number;
  momentum_score: number;
}

export interface SectorOverviewResponse {
  sectors: SectorMomentum[];
  benchmark_return_1w: number;
  benchmark_return_1m: number;
  benchmark_return_3m: number;
  elapsed_seconds: number;
}

export interface SectorTopMover {
  ticker: string;
  name: string;
  return_1m: number;
  current_price: number;
}

export interface SectorDetailResponse {
  etf: string;
  sector: string;
  return_1w: number;
  return_1m: number;
  return_3m: number;
  rs_1w: number;
  rs_1m: number;
  rs_3m: number;
  regime: string;
  regime_confidence: number;
  momentum_score: number;
  top_movers: SectorTopMover[];
  worst_movers: SectorTopMover[];
  elapsed_seconds: number;
}

export async function getSectorOverview(): Promise<SectorOverviewResponse> {
  return apiFetch("/api/sectors/overview");
}

export async function getSectorDetail(
  sectorName: string
): Promise<SectorDetailResponse> {
  return apiFetch(`/api/sectors/detail/${encodeURIComponent(sectorName)}`);
}



export async function refreshSectorHoldings(
  sectorName: string
): Promise<SectorDetailResponse> {
  return apiFetch(`/api/sectors/refresh-holdings/${encodeURIComponent(sectorName)}`, {
    method: "POST",
  });
}

// Holdings management

export interface SectorHoldingItem {
  ticker: string;
  name: string;
}

export interface SectorHoldingsResponse {
  sector: string;
  holdings: SectorHoldingItem[];
  source: "default" | "configured" | "refreshed";
}

export async function getSectorHoldings(
  sectorName: string
): Promise<SectorHoldingsResponse> {
  return apiFetch(`/api/sectors/holdings/${encodeURIComponent(sectorName)}`);
}

export async function updateSectorHoldings(
  sectorName: string,
  holdings: SectorHoldingItem[]
): Promise<SectorHoldingsResponse> {
  return apiFetch(`/api/sectors/holdings/${encodeURIComponent(sectorName)}`, {
    method: "PUT",
    body: JSON.stringify({ holdings }),
  });
}

export async function resetSectorHoldings(
  sectorName: string
): Promise<SectorHoldingsResponse> {
  return apiFetch(`/api/sectors/holdings/${encodeURIComponent(sectorName)}`, {
    method: "DELETE",
  });
}

// -------------------------------------------------------------------
// Strategy Lab — Backtest
// -------------------------------------------------------------------

export interface BacktestTrade {
  entry_date: string;
  exit_date: string;
  entry_price: number;
  exit_price: number;
  quantity: number;
  side: string;
  pnl: number;
  pnl_pct: number;
  exit_reason: string;
  entry_reason: string;
  bars_held: number;
  tax_amount: number;
  pnl_after_tax: number;
}

export interface EquityPoint {
  date: string;
  equity: number;
}

export interface BacktestRegime {
  regime: string;
  confidence: number;
  label: string;
  description: string;
}

export interface BacktestResult {
  ticker: string;
  period: string;
  strategy_name: string;
  initial_cash: number;
  final_equity: number;
  total_return_pct: number;
  annualized_return_pct: number;
  max_drawdown_pct: number;
  sharpe_ratio: number;
  win_rate_pct: number;
  total_trades: number;
  profit_factor: number;
  avg_trade_pnl_pct: number;
  best_trade_pnl_pct: number;
  worst_trade_pnl_pct: number;
  avg_bars_held: number;
  trades: BacktestTrade[];
  equity_curve: EquityPoint[];
  regime: BacktestRegime | null;
  warmup_bars: number;
  // After-tax fields (present when tax is enabled)
  tax_enabled: boolean;
  tax_treatment_used: string | null;
  tax_marginal_rate: number;
  total_tax_paid: number;
  after_tax_return_pct: number;
  after_tax_final_equity: number;
  // Walk-forward robustness fields
  train_years: number;
  test_years: number;
  total_windows: number;
  windows: WalkForwardWindow[];
  wf_avg_return_pct: number;
  wf_avg_annualized_return_pct: number;
  wf_avg_max_drawdown_pct: number;
  wf_avg_sharpe_ratio: number;
  wf_avg_win_rate_pct: number;
  wf_avg_profit_factor: number;
  wf_worst_return_pct: number;
  wf_worst_drawdown_pct: number;
  wf_worst_window_index: number;
  wf_return_std_dev: number;
  stability_score: number;
  verdict: string;
}

export interface BacktestRequest {
  ticker: string;
  interval?: string;
  start?: string;
  end?: string;
  initial_cash?: number;
  commission_pct?: number;
  slippage_pct?: number;
  stop_loss_pct?: number | null;
  take_profit_pct?: number | null;
  // Walk-forward parameters
  train_years?: number;
  test_years?: number;
  max_windows?: number;
}

export async function runBacktest(req: BacktestRequest): Promise<BacktestResult> {
  return apiFetch<BacktestResult>("/api/strategy/backtest", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

// Walk-forward types (used by BacktestResult)

export interface WalkForwardWindow {
  window_index: number;
  train_start: string;
  train_end: string;
  test_start: string;
  test_end: string;
  total_return_pct: number;
  annualized_return_pct: number;
  max_drawdown_pct: number;
  sharpe_ratio: number;
  win_rate_pct: number;
  profit_factor: number;
  total_trades: number;
  error: string | null;
}

// -------------------------------------------------------------------
// Strategy Lab — Auto-Tuner
// -------------------------------------------------------------------

export interface SensitivityEntry {
  param_name: string;
  importance: number;
  best_value: number | string | boolean | null;
  value_range: (number | string | boolean)[];
}

export interface AutoTuneTrial {
  trial_number: number;
  params: Record<string, unknown>;
  objective_value: number;
  avg_return_pct: number;
  avg_annualized_return_pct: number;
  avg_max_drawdown_pct: number;
  avg_sharpe_ratio: number;
  avg_win_rate_pct: number;
  avg_profit_factor: number;
  stability_score: number;
  total_windows: number;
}

export interface AutoTuneResult {
  ticker: string;
  tickers: string[];
  mode: string;  // "single" | "sector" | "custom"
  sector: string | null;
  objective: string;
  objective_label: string;
  n_trials: number;
  elapsed_seconds: number;
  // Best trial
  best_params: Record<string, unknown>;
  best_objective_value: number;
  best_avg_return_pct: number;
  best_avg_annualized_return_pct: number;
  best_avg_max_drawdown_pct: number;
  best_avg_sharpe_ratio: number;
  best_avg_win_rate_pct: number;
  best_avg_profit_factor: number;
  best_stability_score: number;
  // Baseline
  baseline_avg_return_pct: number;
  baseline_avg_annualized_return_pct: number;
  baseline_avg_max_drawdown_pct: number;
  baseline_avg_sharpe_ratio: number;
  baseline_avg_win_rate_pct: number;
  baseline_objective_value: number;
  // Buy-and-hold
  buy_hold_return_pct: number | null;
  // Sensitivity & trials
  sensitivity: SensitivityEntry[];
  trials: AutoTuneTrial[];
  // Verdict
  verdict: string;
  improvement_pct: number;
}

export interface AutoTuneRequest {
  ticker?: string;
  tickers?: string[];
  sector?: string;
  objective?: string;
  n_trials?: number;
  train_years?: number;
  test_years?: number;
  max_windows?: number;
}

export async function runAutoTune(req: AutoTuneRequest): Promise<AutoTuneResult> {
  return apiFetch<AutoTuneResult>("/api/strategy/auto-tune", {
    method: "POST",
    body: JSON.stringify(req),
  });
}


// ---------------------------------------------------------------------------
// Strategy Library
// ---------------------------------------------------------------------------

export interface StrategyItem {
  id: number;
  name: string;
  description: string | null;
  version: number;
  is_preset: boolean;
  trade_mode: string;
  ticker: string | null;
  params: Record<string, unknown>;
  total_return_pct: number | null;
  annualized_return_pct: number | null;
  sharpe_ratio: number | null;
  max_drawdown_pct: number | null;
  win_rate_pct: number | null;
  profit_factor: number | null;
  stability_score: number | null;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface StrategyCreateRequest {
  name: string;
  description?: string;
  trade_mode?: string;
  ticker?: string;
  params?: Record<string, unknown>;
  total_return_pct?: number;
  annualized_return_pct?: number;
  sharpe_ratio?: number;
  max_drawdown_pct?: number;
  win_rate_pct?: number;
  profit_factor?: number;
  stability_score?: number;
}

export interface StrategyUpdateRequest {
  name?: string;
  description?: string;
  trade_mode?: string;
  ticker?: string;
  params?: Record<string, unknown>;
  is_active?: boolean;
  total_return_pct?: number;
  annualized_return_pct?: number;
  sharpe_ratio?: number;
  max_drawdown_pct?: number;
  win_rate_pct?: number;
  profit_factor?: number;
  stability_score?: number;
}

export interface StrategyExport {
  name: string;
  description: string | null;
  version: number;
  trade_mode: string;
  ticker: string | null;
  params: Record<string, unknown>;
  metrics: Record<string, unknown>;
}

export async function listStrategies(): Promise<{ strategies: StrategyItem[] }> {
  return apiFetch<{ strategies: StrategyItem[] }>("/api/strategy/library");
}

export async function createStrategy(req: StrategyCreateRequest): Promise<StrategyItem> {
  return apiFetch<StrategyItem>("/api/strategy/library", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function updateStrategy(id: number, req: StrategyUpdateRequest): Promise<StrategyItem> {
  return apiFetch<StrategyItem>(`/api/strategy/library/${id}`, {
    method: "PATCH",
    body: JSON.stringify(req),
  });
}

export async function deleteStrategy(id: number): Promise<void> {
  await apiFetch(`/api/strategy/library/${id}`, { method: "DELETE" });
}

export async function exportStrategy(id: number): Promise<StrategyExport> {
  return apiFetch<StrategyExport>(`/api/strategy/library/${id}/export`);
}

export async function importStrategy(data: StrategyExport): Promise<StrategyItem> {
  return apiFetch<StrategyItem>("/api/strategy/library/import", {
    method: "POST",
    body: JSON.stringify(data),
  });
}


// ---------------------------------------------------------------------------
// ML Signal Scoring
// ---------------------------------------------------------------------------

export interface MlModelStatus {
  trained: boolean;
  trade_mode?: string;
  trained_at?: string;
  n_tickers?: number;
  n_samples?: number;
  forward_bars?: number;
  metrics?: Record<string, number>;
  feature_importances?: Record<string, number>;
}

export interface MlWalkForwardWindow {
  window_idx: number;
  train_start: string;
  train_end: string;
  test_start: string;
  test_end: string;
  metrics: Record<string, number>;
}

export interface MlTrainResult {
  total_samples: number;
  total_tickers: number;
  trade_mode: string;
  forward_bars: number;
  walk_forward_results: MlWalkForwardWindow[];
  final_metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    auc_roc: number;
    n_train: number;
    n_test: number;
    feature_importances: Record<string, number>;
  };
  elapsed_seconds: number;
  trained_at: string;
}

export async function getMlModelStatus(): Promise<MlModelStatus> {
  return apiFetch<MlModelStatus>("/api/ml/status");
}

export async function trainMlModel(
  universe: string = "sp500",
  tradeMode: string = "swing",
  period: string = "5y",
): Promise<MlTrainResult> {
  const params = new URLSearchParams({
    universe,
    trade_mode: tradeMode,
    period,
  });
  // Training can take 10+ minutes for large universes — use a long timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 15 * 60 * 1000); // 15 min
  try {
    return await apiFetch<MlTrainResult>(`/api/ml/train?${params}`, {
      method: "POST",
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timeoutId);
  }
}

export interface MlPrediction {
  ticker: string;
  ai_rating: number;
  probability: number;
  label: string;
  confidence: string;
  top_features: Array<{ name: string; value: number; importance: number }>;
}

export async function getMlPrediction(
  ticker: string,
  period: string = "6mo",
): Promise<MlPrediction> {
  return apiFetch<MlPrediction>(`/api/ml/predict/${encodeURIComponent(ticker)}?period=${period}`);
}
