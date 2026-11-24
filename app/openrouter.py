"""OpenRouter integration.

Two things live here:

1. :class:`OpenRouterProvider` - a normal OpenAI-compatible chat provider with
   the extra attribution headers OpenRouter expects.
2. Free-model discovery - OpenRouter changes its free lineup regularly, so the
   app asks the API which models are currently free and picks the strongest
   one for the job. The result is cached and refreshed on a timer, so the app
   always follows "the best free model available right now" without hammering
   the API.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from .config import settings
from .llm import OpenAICompatibleProvider

# OpenRouter's own router: it always resolves to whichever free model they
# currently consider best. When it is available it is the correct answer to
# "use the best free model present at this time".
AUTO_ROUTER = "openrouter/free"

# Preference order for the free lineup, best first. Model ids rotate over
# time, so these are substring matches rather than exact ids. Anything not
# listed here still qualifies as a free model, it just ranks below these.
_PREFERENCE_ORDER = (
    AUTO_ROUTER,
    "nemotron-3-ultra",
    "ernie-5.5",
    "gemma-4-31b",
    "ling-3.2-flash",
    "particle-1",
    "nemotron-3.5-super",
    "nemotron-3.5-lightning",
    "gemma-4-26b",
    "flint",
    "lfm-3",
    "dots-3-note",
    "nemotron-3-nano",
)

_lock = threading.Lock()
_cache: dict[str, Any] = {"models": [], "fetched_at": 0.0, "error": None}


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter cannot be reached or returns an error."""


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "documind",
    }
    if settings.openrouter_api_key:
        headers["Authorization"] = f"Bearer {settings.openrouter_api_key}"
    if settings.openrouter_referer:
        headers["HTTP-Referer"] = settings.openrouter_referer
    if settings.openrouter_title:
        headers["X-Title"] = settings.openrouter_title
    return headers


def _is_free(pricing: dict) -> bool:
    try:
        return float(pricing.get("prompt", "1")) == 0 and float(pricing.get("completion", "1")) == 0
    except (TypeError, ValueError):
        return False


def _is_chat_model(model: dict) -> bool:
    """Keep text-in/text-out chat models; skip audio and embedding models."""

    arch = model.get("architecture") or {}
    inputs = arch.get("input_modalities") or []
    outputs = arch.get("output_modalities") or []
    if "text" not in inputs:
        return False
    if outputs != ["text"]:
        return False
    model_id = (model.get("id") or "").lower()
    for blocked in ("embedding", "whisper", "tts", "moderation", "rerank", "lyria"):
        if blocked in model_id:
            return False
    return True


def _rank(model: dict) -> tuple[int, int, int]:
    """Sort key: preference tier, then context length, then recency."""

    model_id = (model.get("id") or "").lower()
    tier = len(_PREFERENCE_ORDER)
    for index, marker in enumerate(_PREFERENCE_ORDER):
        if marker in model_id:
            tier = index
            break
    context = model.get("context_length") or 0
    created = model.get("created") or 0
    return (tier, -context, -created)


def fetch_free_models(force: bool = False) -> list[dict]:
    """Return the currently-free chat models, cached for a short while."""

    ttl = max(settings.openrouter_refresh_seconds, 60)
    now = time.time()
    with _lock:
        fresh = (now - _cache["fetched_at"]) < ttl
        if fresh and not force and _cache["models"]:
            return list(_cache["models"])
        if fresh and _cache["error"]:
            return list(_cache["models"])

    url = f"{settings.openrouter_base_url.rstrip('/')}/models"
    request = urllib.request.Request(url, headers=_headers(), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        with _lock:
            _cache["error"] = f"HTTP {exc.code}: {detail}"
        return list(_cache["models"])
    except Exception as exc:  # noqa: BLE001
        with _lock:
            _cache["error"] = str(exc)
        return list(_cache["models"])

    models = [
        model
        for model in payload.get("data", [])
        if _is_free(model.get("pricing") or {}) and _is_chat_model(model)
    ]
    models.sort(key=_rank)

    with _lock:
        _cache["models"] = models
        _cache["fetched_at"] = now
        _cache["error"] = None
    return models


def select_best_free_model(force: bool = False) -> tuple[str | None, str]:
    """Pick the best free model available right now.

    Returns ``(model_id, reason)``. ``model_id`` is ``None`` when no free
    model could be found - the caller should then fall back to another
    provider or to extractive mode.
    """

    configured = (settings.openrouter_model or "").strip()
    if configured:
        return configured, "pinned via OPENROUTER_MODEL"

    models = fetch_free_models(force=force)
    if not models:
        with _lock:
            error = _cache["error"]
        return None, f"no free model found ({error or 'empty list'})"

    best = models[0]
    model_id = best.get("id", "")
    if model_id == AUTO_ROUTER:
        reason = "OpenRouter free-model router (always their best free model)"
    else:
        context = best.get("context_length") or 0
        reason = f"highest-ranked free model ({context:,} token context)"
    return model_id, reason


def free_model_summary() -> dict:
    """Snapshot of the free lineup, for display in the UI."""

    models = fetch_free_models()
    with _lock:
        error = _cache["error"]
        fetched_at = _cache["fetched_at"]
    return {
        "count": len(models),
        "models": [
            {
                "id": model.get("id"),
                "name": model.get("name"),
                "context_length": model.get("context_length"),
            }
            for model in models
        ],
        "error": error,
        "fetched_at": fetched_at,
    }


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenAI-compatible chat provider for OpenRouter."""

    def __init__(self, api_key: str, base_url: str, model: str, label: str = "openrouter") -> None:
        super().__init__(base_url=base_url, model=model, api_key=api_key, label=label)

    def _headers(self) -> dict[str, str]:
        return _headers()
