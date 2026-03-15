"""LLM provider abstraction for qualitative stock analysis.

Supports Anthropic (Claude), OpenAI (GPT), and Google Gemini as backends.
API keys come from environment variables or user-level settings;
the user's profile selects which provider to use.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Interface for LLM providers."""

    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Send a prompt and return the text response."""
        ...


# ---------------------------------------------------------------------------
# Anthropic (Claude)
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        if not api_key:
            raise ValueError("Anthropic API key is required")
        self._api_key = api_key
        self._model = model

    async def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        message = await client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract text from content blocks
        return "".join(
            block.text for block in message.content if hasattr(block, "text")
        )


# ---------------------------------------------------------------------------
# OpenAI (GPT)
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self._api_key = api_key
        self._model = model

    async def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self._api_key)
        response = await client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------

class GeminiProvider(LLMProvider):
    """Google Gemini provider (free tier: 15 req/min)."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        if not api_key:
            raise ValueError("Google Gemini API key is required")
        self._api_key = api_key
        self._model = model

    async def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        from google import genai

        client = genai.Client(
            api_key=self._api_key,
            http_options=genai.types.HttpOptions(api_version="v1beta"),
        )
        response = await client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                max_output_tokens=max_tokens,
            ),
        )
        return response.text or ""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_llm_provider(
    provider_name: str,
    api_key: str | None = None,
    model: str | None = None,
) -> LLMProvider:
    """Create an LLM provider instance.

    User-level ``api_key`` and ``model`` take precedence when provided;
    otherwise env-var keys and provider defaults are used.

    Args:
        provider_name: "anthropic", "openai", or "gemini"
        api_key: Optional user-level API key override.
        model: Optional model name override.

    Returns:
        An LLMProvider instance.

    Raises:
        ValueError: If the provider is unknown or no API key is available.
    """
    from app.core.config import settings

    if provider_name == "anthropic":
        key = api_key or settings.anthropic_api_key
        if not key:
            raise ValueError(
                "Anthropic API key not configured. "
                "Add your key in Settings or set ANTHROPIC_API_KEY env var."
            )
        kwargs: dict = {"api_key": key}
        if model:
            kwargs["model"] = model
        return AnthropicProvider(**kwargs)

    if provider_name == "openai":
        key = api_key or settings.openai_api_key
        if not key:
            raise ValueError(
                "OpenAI API key not configured. "
                "Add your key in Settings or set OPENAI_API_KEY env var."
            )
        kwargs = {"api_key": key}
        if model:
            kwargs["model"] = model
        return OpenAIProvider(**kwargs)

    if provider_name == "gemini":
        key = api_key or settings.gemini_api_key
        if not key:
            raise ValueError(
                "Google Gemini API key not configured. "
                "Add your key in Settings or set GEMINI_API_KEY env var."
            )
        kwargs = {"api_key": key}
        if model:
            kwargs["model"] = model
        return GeminiProvider(**kwargs)

    raise ValueError(
        f"Unknown LLM provider '{provider_name}'. "
        f"Supported: 'anthropic', 'openai', 'gemini'"
    )


# ---------------------------------------------------------------------------
# Prompt builder + analysis runner
# ---------------------------------------------------------------------------

def _build_analysis_prompt(analysis_data: dict[str, Any]) -> str:
    """Build the LLM prompt from structured analysis data.

    The prompt gives the LLM all the technical data and asks for a
    concise, plain-English qualitative analysis.
    """
    ticker = analysis_data.get("ticker", "???")
    info = analysis_data.get("info", {})
    composite = analysis_data.get("composite", {})
    regime = analysis_data.get("regime")
    indicators = analysis_data.get("indicators", [])
    patterns = analysis_data.get("patterns", [])
    support_levels = analysis_data.get("support_levels", [])
    resistance_levels = analysis_data.get("resistance_levels", [])
    price_data = analysis_data.get("price_data", [])

    # Current price context
    last_bar = price_data[-1] if price_data else {}
    first_bar = price_data[0] if price_data else {}
    current_price = last_bar.get("close", 0)
    period_start_price = first_bar.get("close", 0)
    period_return = (
        ((current_price - period_start_price) / period_start_price * 100)
        if period_start_price
        else 0
    )

    # Format indicators
    ind_lines = []
    for ind in indicators:
        score = ind.get("score", 5.0)
        name = ind.get("name", "")
        display = ind.get("display", {})
        value_str = display.get("value_str", "")
        detail_str = display.get("detail_str", "")
        ind_lines.append(f"  - {name}: score={score:.1f}/10, value={value_str}, {detail_str}")

    # Format patterns
    pat_lines = []
    for pat in patterns:
        score = pat.get("score", 5.0)
        name = pat.get("name", "")
        display = pat.get("display", {})
        value_str = display.get("value_str", "")
        pat_lines.append(f"  - {name}: score={score:.1f}/10, {value_str}")

    # Format S/R levels
    support_str = ", ".join(
        f"${lvl.get('price', 0):.2f} ({lvl.get('touches', 0)} touches)"
        for lvl in support_levels[:5]
    ) or "None detected"
    resistance_str = ", ".join(
        f"${lvl.get('price', 0):.2f} ({lvl.get('touches', 0)} touches)"
        for lvl in resistance_levels[:5]
    ) or "None detected"

    # Regime section
    regime_str = "Not available"
    if regime:
        regime_str = (
            f"{regime.get('label', '?')} "
            f"(confidence: {regime.get('confidence', 0) * 100:.0f}%)\n"
            f"  Sub-type: {regime.get('sub_type_label', 'N/A')}\n"
            f"  Description: {regime.get('description', '')}"
        )
        if regime.get("reasons"):
            regime_str += "\n  Reasons: " + "; ".join(regime["reasons"])

    prompt = f"""You are a professional stock analyst. Analyze the following technical data for {ticker} and provide a concise qualitative summary.

COMPANY: {ticker}
  Name: {info.get('shortName', 'N/A')}
  Sector: {info.get('sector', 'N/A')}
  Industry: {info.get('industry', 'N/A')}
  Market Cap: {info.get('marketCap', 'N/A')}

PRICE: ${current_price:.2f}
  Period return: {period_return:+.1f}%

COMPOSITE SCORES (0-10 scale):
  Overall: {composite.get('overall', 5.0):.1f}
  Trend: {composite.get('trend_score', 'N/A')}
  Contrarian: {composite.get('contrarian_score', 'N/A')}
  Dominant group: {composite.get('dominant_group', 'N/A')}

MARKET REGIME:
  {regime_str}

TECHNICAL INDICATORS:
{chr(10).join(ind_lines) if ind_lines else '  None'}

PATTERN DETECTION:
{chr(10).join(pat_lines) if pat_lines else '  No patterns detected'}

SUPPORT LEVELS: {support_str}
RESISTANCE LEVELS: {resistance_str}

---

Provide a 3-5 sentence qualitative analysis covering:
1. The current trend and momentum picture
2. Key support/resistance levels and what they mean for near-term price action
3. Pattern formations and what they suggest
4. The overall risk/reward outlook

Be specific with price levels. Use plain English that a retail trader would understand.
Do NOT use bullet points — write flowing prose.
Do NOT include disclaimers about not being financial advice.
Keep it under 150 words."""

    return prompt


async def generate_analysis(
    analysis_data: dict[str, Any],
    provider_name: str,
    api_key: str | None = None,
    model: str | None = None,
) -> str:
    """Generate a qualitative LLM analysis for a ticker.

    Args:
        analysis_data: The full analysis result dict (from _run_analysis).
        provider_name: "anthropic", "openai", or "gemini".
        api_key: Optional user-level API key (overrides env var).
        model: Optional model name override.

    Returns:
        The LLM-generated analysis text.
    """
    provider = get_llm_provider(provider_name, api_key=api_key, model=model)
    prompt = _build_analysis_prompt(analysis_data)
    logger.info(
        "Generating LLM analysis for %s via %s",
        analysis_data.get("ticker", "?"),
        provider_name,
    )
    return await provider.generate(prompt, max_tokens=512)
