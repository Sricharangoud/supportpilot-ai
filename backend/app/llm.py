"""Provider-agnostic LLM interface with Groq, OpenAI, and Gemini adapters."""
from abc import ABC, abstractmethod
import time
import httpx

from app.config import Settings


class LLMProvider(ABC):
    """Abstract interface isolating LLM vendor details from agent logic."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Send chat messages and return assistant text."""


class OpenAICompatibleProvider(LLMProvider):
    """Calls any OpenAI-compatible /chat/completions endpoint (Groq, OpenAI, Gemini)."""

    def __init__(self, api_key: str, model: str, base_url: str, timeout: float = 30.0):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY is not configured.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(4):
                resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                if resp.status_code == 429 and attempt < 3:
                    retry_after = float(resp.headers.get("retry-after", 2 * (attempt + 1)))
                    time.sleep(min(retry_after, 15.0))
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()


def get_llm_provider(settings: Settings) -> LLMProvider:
    """Factory returning the configured LLMProvider implementation."""
    provider = settings.llm_provider.lower().strip()
    if provider == "groq":
        return OpenAICompatibleProvider(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url="https://api.groq.com/openai/v1",
        )
    if provider == "openai":
        return OpenAICompatibleProvider(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url="https://api.openai.com/v1",
        )
    if provider == "gemini":
        return OpenAICompatibleProvider(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        )
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")

