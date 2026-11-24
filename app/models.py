"""Core data structures shared across the RAG pipeline."""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Section:
    """A logical piece of a document (a PDF page, a sheet, a heading block)."""

    text: str
    page: int | None = None
    section: str | None = None
    order: int = 0

    def label(self) -> str:
        if self.page is not None:
            return f"Page {self.page}"
        if self.section:
            return self.section
        return "Content"


@dataclass
class Document:
    doc_id: str
    filename: str
    filetype: str
    size_bytes: int
    added_at: float
    pages: int = 0
    chunks: int = 0
    characters: int = 0
    path: str | None = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    filename: str
    text: str
    page: int | None
    section: str | None
    order: int
    char_start: int = 0
    char_end: int = 0
    vector: list[float] | None = None

    @property
    def location(self) -> str:
        if self.page is not None:
            return f"Page {self.page}"
        if self.section:
            return self.section
        return "Document"

    def citation(self) -> str:
        return f"{self.filename} — {self.location}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("vector", None)
        return data


@dataclass
class Source:
    """A retrieved chunk surfaced to the UI alongside the answer."""

    index: int
    chunk_id: str
    doc_id: str
    filename: str
    page: int | None
    section: str | None
    snippet: str
    score: float
    citation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Source]
    mode: str
    elapsed_ms: int
    grounded: bool = True
    used_docs: list[str] = field(default_factory=list)
    follow_ups: list[str] = field(default_factory=list)
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "mode": self.mode,
            "elapsed_ms": self.elapsed_ms,
            "grounded": self.grounded,
            "used_docs": self.used_docs,
            "follow_ups": self.follow_ups,
            "warning": self.warning,
        }


def new_doc_id(filename: str) -> str:
    stamp = time.strftime("%Y%m%d%H%M%S")
    digest = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:6]
    return f"{stamp}-{digest}-{uuid.uuid4().hex[:4]}"


def new_chunk_id(doc_id: str, order: int) -> str:
    return f"{doc_id}:{order}"
