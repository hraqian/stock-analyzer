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
