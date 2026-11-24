"""LLM providers.

Provider order when ``RAG_LLM_PROVIDER=auto``:

1. MiniMax platform API (needs ``MINIMAX_API_KEY``)
2. MiniMax runtime session token (reuses the signed-in MiniMax Code session)
3. Any OpenAI-compatible endpoint (``OPENAI_API_KEY``)
4. Local Ollama server
5. Extractive fallback (no LLM - quotes original sentences only)

All providers implement the Anthropic-style ``/messages`` contract or the
OpenAI ``/chat/completions`` contract, and expose the same ``complete`` and
``stream`` methods.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .config import settings

MINIMAX_AUTH_FILES = [
    Path.home() / ".minimax" / "auth" / "prod" / "en" / "mcode-public" / "auth.json",
    Path.home() / ".minimax" / "auth" / "prod" / "mcode-public" / "auth.json",
]


class LLMError(RuntimeError):
    """Raised when a provider call fails."""


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str


def _runtime_token() -> tuple[str | None, str | None]:
    """Return ``(token, base_url)`` from the signed-in MiniMax Code session."""

    for path in MINIMAX_AUTH_FILES:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for record in data.get("records", {}).values():
                token = record.get("accessToken")
                if token:
                    return token, "https://agent.minimax.io/mavis/api/v1/llm/v1"
        except Exception:  # noqa: BLE001
            continue
    return None, None


def _post_json(
    url: str,
    payload: dict,
    headers: dict[str, str],
    timeout: float = 120.0,
) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise LLMError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"Request to {url} failed: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Non-JSON response from {url}: {raw[:300]}") from exc


# --------------------------------------------------------------------------- #
# Anthropic-style provider (MiniMax platform + runtime session)
# --------------------------------------------------------------------------- #
class AnthropicStyleProvider:
    name = "minimax"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        bearer_token: str | None = None,
        label: str = "minimax",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.bearer_token = bearer_token
        self.label = label

    def _headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self.bearer_token:
            headers["authorization"] = f"Bearer {self.bearer_token}"
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _payload(self, system: str | None, messages: list[dict], max_tokens: int, stream: bool) -> dict:
        payload: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            payload["system"] = system
        if stream:
            payload["stream"] = True
        return payload

    def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1500,
        temperature: float = 0.1,
    ) -> LLMResponse:
        payload = self._payload(system, messages, max_tokens, stream=False)
        payload["temperature"] = temperature
        data = _post_json(f"{self.base_url}/messages", payload, self._headers())
        parts: list[str] = []
        for block in data.get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        text = "".join(parts).strip()
        if not text:
            raise LLMError(f"Empty response from {self.label}: {str(data)[:200]}")
        return LLMResponse(text=text, model=data.get("model", self.model), provider=self.label)

    def stream(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1500,
        temperature: float = 0.1,
    ) -> Iterator[str]:
        payload = self._payload(system, messages, max_tokens, stream=True)
        payload["temperature"] = temperature
        request = urllib.request.Request(
            f"{self.base_url}/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            response = urllib.request.urlopen(request, timeout=180)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise LLMError(f"HTTP {exc.code}: {detail}") from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Stream request failed: {exc}") from exc

        with response:
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta") or {}
                    piece = delta.get("text")
                    if piece:
                        yield piece


# --------------------------------------------------------------------------- #
# OpenAI-compatible provider
# --------------------------------------------------------------------------- #
class OpenAICompatibleProvider:
    name = "openai"

    def __init__(self, base_url: str, model: str, api_key: str | None = None, label: str = "openai") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.label = label

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers

    def _messages(self, system: str | None, messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        if system:
            out.append({"role": "system", "content": system})
        out.extend(messages)
        return out

    def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1500,
        temperature: float = 0.1,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": self._messages(system, messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        data = _post_json(f"{self.base_url}/chat/completions", payload, self._headers())
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"Empty response from {self.label}: {str(data)[:200]}")
        text = (choices[0].get("message") or {}).get("content", "").strip()
        if not text:
            raise LLMError(f"Empty message content from {self.label}")
        return LLMResponse(text=text, model=data.get("model", self.model), provider=self.label)

    def stream(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1500,
        temperature: float = 0.1,
    ) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": self._messages(system, messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            response = urllib.request.urlopen(request, timeout=180)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise LLMError(f"HTTP {exc.code}: {detail}") from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Stream request failed: {exc}") from exc

        with response:
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for choice in event.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        yield piece


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #
class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.label = "ollama"

    def _payload(self, system: str | None, messages: list[dict], max_tokens: int, temperature: float, stream: bool) -> dict:
        payload = {
            "model": self.model,
            "messages": (([{"role": "system", "content": system}] if system else []) + messages),
            "stream": stream,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        return payload

    def complete(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1500,
        temperature: float = 0.1,
    ) -> LLMResponse:
        payload = self._payload(system, messages, max_tokens, temperature, stream=False)
        data = _post_json(f"{self.base_url}/api/chat", payload, {"content-type": "application/json"})
        text = ((data.get("message") or {}).get("content") or "").strip()
        if not text:
            raise LLMError(f"Empty response from ollama: {str(data)[:200]}")
        return LLMResponse(text=text, model=self.model, provider="ollama")

    def stream(
        self,
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 1500,
        temperature: float = 0.1,
    ) -> Iterator[str]:
        payload = self._payload(system, messages, max_tokens, temperature, stream=True)
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        response = urllib.request.urlopen(request, timeout=300)
        with response:
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                piece = (event.get("message") or {}).get("content")
                if piece:
                    yield piece


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_provider():
    """Return ``(provider, note)`` for the configured provider chain."""

    choice = (settings.llm_provider or "auto").lower()

    if choice == "none":
        # Explicitly disable the LLM: answers are quoted verbatim from the
        # documents. Useful for public deployments where nobody should be able
        # to spend an API key, and for offline testing.
        return None, "extractive"

    if choice in {"minimax", "auto"}:
        if settings.minimax_api_key:
            provider = AnthropicStyleProvider(
                base_url=settings.minimax_base_url,
                model=settings.minimax_model,
                api_key=settings.minimax_api_key,
                label="minimax",
            )
            return provider, f"minimax:{settings.minimax_model}"

    if choice in {"minimax", "auto"}:
        token, base_url = _runtime_token()
        if token:
            provider = AnthropicStyleProvider(
                base_url=base_url or "https://agent.minimax.io/mavis/api/v1/llm/v1",
                model=settings.minimax_model,
                bearer_token=token,
                label="minimax-session",
            )
            return provider, f"minimax-session:{settings.minimax_model}"

    if choice in {"openai", "auto"} and settings.openai_api_key:
        provider = OpenAICompatibleProvider(
            base_url=settings.openai_base_url,
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            label="openai",
        )
        return provider, f"openai:{settings.openai_model}"

    if choice in {"ollama", "auto"}:
        try:
            request = urllib.request.Request(f"{settings.ollama_base_url}/api/tags", method="GET")
            with urllib.request.urlopen(request, timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8", "replace"))
            names = [m.get("name", "") for m in payload.get("models", [])]
            model = settings.ollama_model
            if model not in names:
                match = next((n for n in names if n.startswith(settings.ollama_model.split(":")[0])), None)
                model = match or (names[0] if names else settings.ollama_model)
            return OllamaProvider(settings.ollama_base_url, model), f"ollama:{model}"
        except Exception:  # noqa: BLE001
            pass

    return None, "extractive"


_CITATION_RE = re.compile(r"\[(\d+)\]")


def citations_in(text: str) -> list[int]:
    return sorted({int(m) for m in _CITATION_RE.findall(text)})
