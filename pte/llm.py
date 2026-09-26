"""Minimal LLM clients: any OpenAI-compatible endpoint, Anthropic, and a scripted fake.

Deliberately dependency-light (``httpx`` only) and tool-call free, so that
small local models (Ollama, llama.cpp, vLLM) work as well as frontier APIs:
the model just answers with a Python code block.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import httpx


@dataclass
class Message:
    role: str      # "user" or "assistant"
    content: str


@dataclass
class LLMReply:
    text: str
    input_tokens: int
    output_tokens: int


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    name: str
    price_in_per_mtok: float
    price_out_per_mtok: float

    def complete(self, system: str, messages: Sequence[Message]) -> LLMReply: ...


@dataclass
class ModelSpec:
    provider: str                  # "openai" (any OpenAI-compatible server), "anthropic", "scripted"
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    price_in_per_mtok: float = 0.0   # USD per million input tokens (set from your provider's price list)
    price_out_per_mtok: float = 0.0  # USD per million output tokens
    timeout_s: float = 300.0
    max_retries: int = 4


_RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class _HTTPModel:
    def __init__(self, spec: ModelSpec, client: httpx.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.spec = spec
        self.name = spec.model
        self.price_in_per_mtok = spec.price_in_per_mtok
        self.price_out_per_mtok = spec.price_out_per_mtok
        self._client = client or httpx.Client(timeout=spec.timeout_s)
        self._sleep = sleep

    def _api_key(self) -> str | None:
        if not self.spec.api_key_env:
            return None
        key = os.environ.get(self.spec.api_key_env)
        if not key:
            raise LLMError(f"environment variable {self.spec.api_key_env} is not set")
        return key

    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        delay = 2.0
        for attempt in range(self.spec.max_retries + 1):
            try:
                r = self._client.post(url, headers=headers, json=payload)
            except httpx.TransportError as exc:
                if attempt == self.spec.max_retries:
                    raise LLMError(f"network error calling {url}: {exc}") from exc
            else:
                if r.status_code == 200:
                    return r.json()
                if r.status_code not in _RETRYABLE or attempt == self.spec.max_retries:
                    raise LLMError(f"{url} returned HTTP {r.status_code}: {r.text[:500]}")
            self._sleep(delay)
            delay = min(delay * 2, 60.0)
        raise LLMError("unreachable")


class OpenAICompatible(_HTTPModel):
    """OpenAI, OpenRouter, Together, Groq, Ollama (``http://localhost:11434/v1``), vLLM, ..."""

    def complete(self, system: str, messages: Sequence[Message]) -> LLMReply:
        headers = {"Content-Type": "application/json"}
        key = self._api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        payload = {
            "model": self.spec.model,
            "messages": [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.spec.temperature,
            "max_tokens": self.spec.max_tokens,
        }
        data = self._post(self.spec.base_url.rstrip("/") + "/chat/completions", headers, payload)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response: {str(data)[:500]}") from exc
        usage = data.get("usage") or {}
        return LLMReply(text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))


class Anthropic(_HTTPModel):
    """Anthropic Messages API."""

    API_URL = "https://api.anthropic.com/v1/messages"

    def complete(self, system: str, messages: Sequence[Message]) -> LLMReply:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key() or "",
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.spec.model,
            "system": system,
            "max_tokens": self.spec.max_tokens,
            "temperature": self.spec.temperature,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        data = self._post(self.spec.base_url or self.API_URL, headers, payload)
        try:
            text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        except (KeyError, TypeError) as exc:
            raise LLMError(f"unexpected response: {str(data)[:500]}") from exc
        usage = data.get("usage") or {}
        return LLMReply(text, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)))


class Scripted:
    """Deterministic fake LLM for tests and the offline demo.

    ``responder(system, messages) -> str``; token counts are estimated as chars/4.
    """

    def __init__(self, responder: Callable[[str, Sequence[Message]], str], name: str = "scripted",
                 price_in_per_mtok: float = 0.0, price_out_per_mtok: float = 0.0):
        self.responder = responder
        self.name = name
        self.price_in_per_mtok = price_in_per_mtok
        self.price_out_per_mtok = price_out_per_mtok
        self.calls = 0

    def complete(self, system: str, messages: Sequence[Message]) -> LLMReply:
        self.calls += 1
        text = self.responder(system, messages)
        n_in = (len(system) + sum(len(m.content) for m in messages)) // 4
        return LLMReply(text, n_in, len(text) // 4)


def build_llm(spec: ModelSpec) -> LLM:
    if spec.provider == "openai":
        if not spec.base_url:
            raise ValueError("provider 'openai' needs base_url (e.g. https://api.openai.com/v1 or http://localhost:11434/v1)")
        return OpenAICompatible(spec)
    if spec.provider == "anthropic":
        return Anthropic(spec)
    raise ValueError(f"unknown provider {spec.provider!r} (expected 'openai' or 'anthropic')")
