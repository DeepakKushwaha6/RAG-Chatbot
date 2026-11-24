"""Chunking.

Chunks are built per section (page / sheet / slide) so that citations stay
precise. Large sections are split on paragraph and sentence boundaries with a
character-level overlap, which keeps answerable spans intact.
"""

from __future__ import annotations

import re

from .config import settings
from .models import Chunk, Section, new_chunk_id

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_END.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_section(
    section: Section,
    doc_id: str,
    filename: str,
    chunk_chars: int | None = None,
    overlap: int | None = None,
    start_order: int = 0,
) -> list[Chunk]:
    """Split one section into overlapping chunks.

    ``start_order`` makes chunk ids unique across the whole document - without
    it every section would restart at 0 and chunk ids would collide.
    """

    chunk_chars = chunk_chars or settings.chunk_chars
    overlap = overlap if overlap is not None else settings.chunk_overlap
    overlap = min(overlap, max(0, chunk_chars // 2))

    units: list[str] = []
    for paragraph in _split_paragraphs(section.text):
        if len(paragraph) <= chunk_chars:
            units.append(paragraph)
            continue
        buffer = ""
        for sentence in _split_sentences(paragraph):
            candidate = f"{buffer} {sentence}".strip() if buffer else sentence
            if len(candidate) <= chunk_chars:
                buffer = candidate
                continue
            if buffer:
                units.append(buffer)
            if len(sentence) > chunk_chars:
                # Hard-split very long sentences.
                for start in range(0, len(sentence), chunk_chars - overlap):
                    units.append(sentence[start : start + chunk_chars])
                buffer = ""
            else:
                buffer = sentence
        if buffer:
            units.append(buffer)

    if not units:
        return []

    chunks: list[Chunk] = []
    buffer = ""
    for unit in units:
        candidate = f"{buffer}\n\n{unit}" if buffer else unit
        if len(candidate) <= chunk_chars or not buffer:
            buffer = candidate
            continue
        chunks.append(buffer)
        tail = buffer[-overlap:] if overlap else ""
        buffer = f"{tail}\n\n{unit}".strip() if tail else unit
    if buffer.strip():
        chunks.append(buffer)

    result: list[Chunk] = []
    offset = 0
    for position, text in enumerate(chunks):
        clean = _normalize(text)
        if len(clean) < 20:
            offset += len(text)
            continue
        result.append(
            Chunk(
                chunk_id=new_chunk_id(doc_id, start_order + position),
                doc_id=doc_id,
                filename=filename,
                text=clean,
                page=section.page,
                section=section.section,
                order=start_order + position,
                char_start=offset,
                char_end=offset + len(text),
            )
        )
        offset += len(text)
    return result


def chunk_sections(
    sections: list[Section],
    doc_id: str,
    filename: str,
    chunk_chars: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        produced = chunk_section(
            section, doc_id, filename, chunk_chars, overlap, start_order=len(chunks)
        )
        chunks.extend(produced)
    return chunks
