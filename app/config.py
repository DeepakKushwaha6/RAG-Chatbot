"""Configuration loading for the RAG chatbot."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    base_dir: Path = BASE_DIR
    data_dir: Path = field(
        default_factory=lambda: Path(_env("RAG_DATA_DIR") or (BASE_DIR / "data"))
    )

    # LLM providers
    llm_provider: str = field(default_factory=lambda: _env("RAG_LLM_PROVIDER", "auto"))
    minimax_api_key: str = field(default_factory=lambda: _env("MINIMAX_API_KEY"))
    minimax_base_url: str = field(
        default_factory=lambda: _env("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic")
    )
    minimax_model: str = field(
        default_factory=lambda: _env("MINIMAX_MODEL", "MiniMax-M3.1-Flash")
    )
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_base_url: str = field(
        default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))

    # OpenRouter - the app auto-selects the best free model it offers
    openrouter_api_key: str = field(default_factory=lambda: _env("OPENROUTER_API_KEY"))
    openrouter_base_url: str = field(
        default_factory=lambda: _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )
    openrouter_model: str = field(default_factory=lambda: _env("OPENROUTER_MODEL"))
    openrouter_refresh_seconds: int = field(
        default_factory=lambda: _env_int("OPENROUTER_REFRESH_SECONDS", 1800)
    )
    openrouter_referer: str = field(default_factory=lambda: _env("OPENROUTER_REFERER"))
    openrouter_title: str = field(
        default_factory=lambda: _env("OPENROUTER_TITLE", "DocuMind RAG Chatbot")
    )
    ollama_base_url: str = field(
        default_factory=lambda: _env("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    )
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "llama3.1"))

    # Retrieval tuning
    chunk_chars: int = field(default_factory=lambda: _env_int("RAG_CHUNK_CHARS", 1200))
    chunk_overlap: int = field(default_factory=lambda: _env_int("RAG_CHUNK_OVERLAP", 200))
    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 6))
    max_context_chars: int = field(
        default_factory=lambda: _env_int("RAG_MAX_CONTEXT_CHARS", 16000)
    )
    min_score: float = field(default_factory=lambda: _env_float("RAG_MIN_SCORE", 0.05))
    candidate_multiplier: int = field(
        default_factory=lambda: _env_int("RAG_CANDIDATE_MULTIPLIER", 5)
    )

    # Server
    host: str = field(default_factory=lambda: _env("RAG_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("RAG_PORT", 0) or _env_int("PORT", 8000))

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
