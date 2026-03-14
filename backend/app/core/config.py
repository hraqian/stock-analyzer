"""Application settings loaded from environment variables."""

from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration — values come from env vars or .env file."""

    # --- App ---
    app_name: str = "Stock Analyzer"
    debug: bool = False

    # --- Auth / JWT ---
    secret_key: str = "CHANGE-ME-in-production-use-a-real-secret"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440  # 24 hours

    # --- Database ---
    # SQLite by default; path is relative to backend/ directory.
    database_url: str = "sqlite+aiosqlite:///./stock_analyzer.db"

    # --- CORS (frontend origin) ---
    cors_origins: list[str] = ["http://localhost:3000"]

    # --- Data providers ---
    # Yahoo Finance is always available (free).  Add API keys here to
    # enable paid providers.
    polygon_api_key: str = ""
    alpha_vantage_api_key: str = ""
    iex_cloud_api_key: str = ""

    # --- AI ---
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # --- Paths ---
    # Root of the original engine code (one level up from backend/)
    engine_root: Path = Path(__file__).resolve().parent.parent.parent

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
