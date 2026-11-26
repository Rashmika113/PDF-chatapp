"""Pluggable LLM providers behind one tiny interface: `stream(system, messages)`.

- ollama:    free, runs locally (default)
- groq:      free tier, hosted open models, very fast
- openai:    paid
- anthropic: paid (Claude)

Ollama, Groq and OpenAI all speak the OpenAI chat-completions protocol, so they
share one class. Claude uses the official Anthropic SDK.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

Message = dict[str, str]  # {"role": "user" | "assistant", "content": "..."}


class LLM(Protocol):
    name: str

    def stream(self, system: str, messages: list[Message]) -> Iterator[str]: ...


@dataclass(frozen=True)
class ProviderInfo:
    key: str
    label: str
    default_model: str
    needs_api_key: bool
    api_key_env: str | None = None


PROVIDERS: dict[str, ProviderInfo] = {
    "ollama": ProviderInfo("ollama", "Ollama (local, free)", os.getenv("OLLAMA_MODEL", "llama3.2"), False),
    "groq": ProviderInfo(
        "groq", "Groq (free tier)", os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), True, "GROQ_API_KEY"
    ),
    "openai": ProviderInfo(
        "openai", "OpenAI", os.getenv("OPENAI_MODEL", "gpt-5-mini"), True, "OPENAI_API_KEY"
    ),
    "anthropic": ProviderInfo(
        "anthropic",
        "Anthropic Claude",
        os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5"),
        True,
        "ANTHROPIC_API_KEY",
    ),
}


class OpenAICompatibleLLM:
    def __init__(self, model: str, api_key: str, base_url: str | None = None, name: str = "openai"):
        from openai import OpenAI

        self.name = name
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def stream(self, system: str, messages: list[Message]) -> Iterator[str]:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, *messages],
            stream=True,
        )
        for event in response:
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content


class AnthropicLLM:
    def __init__(self, model: str, api_key: str | None = None):
        import anthropic

        self.name = "anthropic"
        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key or None)

    def stream(self, system: str, messages: list[Message]) -> Iterator[str]:
        with self._client.beta.messages.stream(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=messages,
            # Q&A over retrieved text doesn't need deep reasoning; low effort keeps it fast.
            output_config={"effort": "low"},
            # If a request is declined by a safety classifier, retry on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            yield from stream.text_stream
            if stream.get_final_message().stop_reason == "refusal":
                yield "\n\n_The model declined to answer this question._"


def get_llm(provider: str, model: str | None = None, api_key: str | None = None) -> LLM:
    """Build an LLM client. `api_key` falls back to the provider's env variable."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}. Choose one of: {', '.join(PROVIDERS)}")
    info = PROVIDERS[provider]
    model = model or info.default_model
    key = api_key or (os.getenv(info.api_key_env) if info.api_key_env else None)
    if info.needs_api_key and not key:
        raise ValueError(f"{info.label} needs an API key. Set {info.api_key_env} or paste it in the sidebar.")

    if provider == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OpenAICompatibleLLM(model, api_key="ollama", base_url=f"{base.rstrip('/')}/v1", name="ollama")
    if provider == "groq":
        return OpenAICompatibleLLM(model, api_key=key, base_url="https://api.groq.com/openai/v1", name="groq")
    if provider == "openai":
        return OpenAICompatibleLLM(model, api_key=key, name="openai")
    return AnthropicLLM(model, api_key=key)
