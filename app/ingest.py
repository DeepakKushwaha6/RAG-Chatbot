"""Shared document ingestion.

Both the FastAPI server and the Streamlit app use this so they behave
identically: the same parsing, the same chunking, the same index updates.
"""

from __future__ import annotations

import time
from pathlib import Path

from .chunking import chunk_sections
from .loaders import UnsupportedDocument, is_supported, load_document
from .models import Document, new_doc_id
from .store import Store


class IngestError(Exception):
    """Raised when a file cannot be indexed."""


def ingest_bytes(store: Store, filename: str, data: bytes) -> Document:
    """Index the given file content and add it to ``store``.

    Raises :class:`IngestError` with a message suitable for the user.
    """

    safe_name = Path(filename).name or "document"
    if not is_supported(safe_name):
        from .loaders import SUPPORTED_EXTENSIONS

        raise IngestError(
            f"Unsupported file type '{Path(safe_name).suffix or safe_name}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if not data:
        raise IngestError("The file is empty.")

    doc_id = new_doc_id(safe_name)
    suffix = Path(safe_name).suffix.lower()
    stored_path = store.upload_dir / f"{doc_id}{suffix}"
    stored_path.write_bytes(data)

    try:
        sections = load_document(stored_path)
    except UnsupportedDocument as exc:
        stored_path.unlink(missing_ok=True)
        raise IngestError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        stored_path.unlink(missing_ok=True)
        raise IngestError(f"Could not read the file: {exc}") from exc

    if not sections:
        stored_path.unlink(missing_ok=True)
        raise IngestError("The document contains no readable text.")

    chunks = chunk_sections(sections, doc_id, safe_name)
    if not chunks:
        stored_path.unlink(missing_ok=True)
        raise IngestError("No indexable content was found.")

    document = Document(
        doc_id=doc_id,
        filename=safe_name,
        filetype=suffix.lstrip("."),
        size_bytes=len(data),
        added_at=time.time(),
        pages=len(sections),
        chunks=len(chunks),
        characters=sum(len(section.text) for section in sections),
        path=str(stored_path),
        meta={"sections": len(sections)},
    )
    store.add_document(document, chunks)
    return document


def remove_document(store: Store, doc_id: str) -> bool:
    """Drop a document from the index and delete its stored copy."""

    if not store.remove_document(doc_id):
        return False
    for path in store.upload_dir.glob(f"{doc_id}.*"):
        path.unlink(missing_ok=True)
    return True


def clear_all(store: Store) -> None:
    """Remove every indexed document and its stored copy."""

    store.clear()
    for path in store.upload_dir.glob("*"):
        path.unlink(missing_ok=True)
