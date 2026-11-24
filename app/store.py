"""Document store + hybrid (dense + lexical) retrieval index.

The index is persisted to disk so uploaded documents survive a server restart:

    data/index/chunks.json     chunk metadata
    data/index/embeddings.npy  float32 matrix aligned with chunks.json
    data/index/documents.json  document metadata
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Iterable

import numpy as np

from . import embeddings
from .config import settings
from .models import Chunk, Document

_TOKEN = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a", "about", "above", "after", "all", "also", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "get", "had", "has", "have", "he",
    "her", "here", "him", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "just", "me", "more", "most", "my", "no", "not", "of", "on", "or",
    "our", "out", "over", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "to", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would", "you",
    "your", "document", "documents", "please", "tell", "give", "answer",
    "question", "based", "according",
}

# Conservative suffix stripping so "deadlines" matches "deadline" and
# "reimbursement" matches "reimburse". Only suffixes that reliably collapse
# to the same stem are removed.
_SUFFIXES = (
    "ization", "izations", "iveness", "fulness", "ousness",
    "ements", "ement", "ments", "ment", "ences", "ence",
    "ingly", "edly", "ings", "ing",
    "ers", "er", "est", "ed",
)


def stem(word: str) -> str:
    """Very light stemmer - enough to bridge plural/verb forms."""

    if len(word) <= 3 or word.isdigit():
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ied") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    return [
        stem(t)
        for t in tokens
        if t not in _STOPWORDS and len(t) > 1 and stem(t) not in _STOPWORDS
    ]


class Store:
    """Thread-safe in-memory index with on-disk persistence."""

    def __init__(self, index_dir: Path | None = None) -> None:
        self.index_dir = index_dir or settings.index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.documents: dict[str, Document] = {}
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray = np.zeros((0, embeddings.EMBED_DIM), dtype=np.float32)
        self._bm25 = None
        self._bm25_tokens: list[list[str]] = []
        self._load()

    # ------------------------------------------------------------------ paths
    @property
    def _chunks_path(self) -> Path:
        return self.index_dir / "chunks.json"

    @property
    def _vectors_path(self) -> Path:
        return self.index_dir / "embeddings.npy"

    @property
    def _documents_path(self) -> Path:
        return self.index_dir / "documents.json"

    # -------------------------------------------------------------- persistence
    def _load(self) -> None:
        if self._documents_path.exists():
            try:
                raw = json.loads(self._documents_path.read_text(encoding="utf-8"))
                for item in raw:
                    doc = Document(**item)
                    self.documents[doc.doc_id] = doc
            except Exception:  # noqa: BLE001
                self.documents = {}
        if self._chunks_path.exists():
            try:
                raw = json.loads(self._chunks_path.read_text(encoding="utf-8"))
                self.chunks = [Chunk(**item) for item in raw]
            except Exception:  # noqa: BLE001
                self.chunks = []
        if self._vectors_path.exists() and self.chunks:
            try:
                matrix = np.load(self._vectors_path)
                if matrix.shape[0] == len(self.chunks):
                    self.vectors = matrix.astype(np.float32)
            except Exception:  # noqa: BLE001
                self.vectors = np.zeros((0, embeddings.EMBED_DIM), dtype=np.float32)
        self._rebuild_bm25()

    def _persist(self) -> None:
        self._chunks_path.write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False),
            encoding="utf-8",
        )
        np.save(self._vectors_path, self.vectors)
        self._documents_path.write_text(
            json.dumps([d.to_dict() for d in self.documents.values()], ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ mutate
    def add_document(self, doc: Document, chunks: list[Chunk]) -> None:
        with self._lock:
            self.documents[doc.doc_id] = doc
            self.chunks = [c for c in self.chunks if c.doc_id != doc.doc_id]
            if chunks:
                matrix = embeddings.embed([c.text for c in chunks])
                start = len(self.chunks)
                self.chunks.extend(chunks)
                if self.vectors.size == 0:
                    self.vectors = matrix
                else:
                    self.vectors = np.vstack([self.vectors, matrix])
                del start
            doc.chunks = len(chunks)
            self._rebuild_bm25()
            self._persist()

    def remove_document(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self.documents:
                return False
            keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
            if keep:
                self.chunks = [self.chunks[i] for i in keep]
                self.vectors = self.vectors[keep] if self.vectors.size else self.vectors
            else:
                self.chunks = []
                self.vectors = np.zeros((0, embeddings.EMBED_DIM), dtype=np.float32)
            self.documents.pop(doc_id, None)
            self._rebuild_bm25()
            self._persist()
            return True

    def clear(self) -> None:
        with self._lock:
            self.documents = {}
            self.chunks = []
            self.vectors = np.zeros((0, embeddings.EMBED_DIM), dtype=np.float32)
            self._bm25 = None
            self._bm25_tokens = []
            self._persist()

    # ------------------------------------------------------------------ bm25
    def _rebuild_bm25(self) -> None:
        self._bm25_tokens = [tokenize(c.text) for c in self.chunks]
        self._bm25 = None
        if not self._bm25_tokens:
            return
        try:
            from rank_bm25 import BM25Okapi

            corpus = [tokens or ["_empty_"] for tokens in self._bm25_tokens]
            self._bm25 = BM25Okapi(corpus)
        except Exception:  # noqa: BLE001
            self._bm25 = None

    def _bm25_scores(self, query_tokens: list[str]) -> np.ndarray:
        if self._bm25 is None or not self.chunks:
            return np.zeros(len(self.chunks), dtype=np.float32)
        scores = np.asarray(self._bm25.get_scores(query_tokens), dtype=np.float32)
        return scores

    # ----------------------------------------------------------------- search
    def search(
        self,
        query: str,
        top_k: int | None = None,
        doc_ids: Iterable[str] | None = None,
        candidate_multiplier: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Hybrid search returning ``(chunk, score)`` pairs, best first.

        Dense cosine similarity and BM25 are fused with Reciprocal Rank Fusion
        so that both semantic matches and exact keyword matches rank highly.
        """

        top_k = top_k or settings.top_k
        candidate_multiplier = candidate_multiplier or settings.candidate_multiplier
        allowed = set(doc_ids) if doc_ids else None

        with self._lock:
            if not self.chunks:
                return []
            indices = [
                i
                for i, c in enumerate(self.chunks)
                if allowed is None or c.doc_id in allowed
            ]
            if not indices:
                return []

            query_tokens = tokenize(query)
            dense_scores = np.zeros(len(indices), dtype=np.float32)
            if self.vectors.size and query_tokens or self.vectors.size:
                qvec = embeddings.embed([query])[0]
                sub = self.vectors[indices]
                dense_scores = sub @ qvec

            bm25_all = self._bm25_scores(query_tokens)
            bm25_scores = bm25_all[indices] if bm25_all.size else np.zeros(len(indices))

            fused = self._fuse(dense_scores, bm25_scores, indices)

            ranked = sorted(
                zip(indices, fused), key=lambda pair: pair[1], reverse=True
            )[: max(top_k * candidate_multiplier, top_k)]

            results: list[tuple[Chunk, float]] = []
            for idx, score in ranked:
                chunk = self.chunks[idx]
                results.append((chunk, float(score)))
            return results[:top_k]

    @staticmethod
    def _fuse(
        dense: np.ndarray, lexical: np.ndarray, indices: list[int]
    ) -> np.ndarray:
        """Reciprocal Rank Fusion over the two ranked lists."""

        rrf_k = 60.0
        fused = np.zeros(len(indices), dtype=np.float32)

        dense_order = np.argsort(-dense)
        for rank, pos in enumerate(dense_order):
            if dense[pos] <= 0.01:
                continue
            fused[pos] += 1.0 / (rrf_k + rank + 1)

        lexical_order = np.argsort(-lexical)
        for rank, pos in enumerate(lexical_order):
            if lexical[pos] <= 0.0:
                continue
            fused[pos] += 1.0 / (rrf_k + rank + 1)

        # Keep a small absolute-similarity signal so that a weak lexical match
        # on an unrelated chunk cannot outrank a genuinely similar passage.
        if dense.size:
            positive = dense[dense > 0]
            scale = float(positive.max()) if positive.size else 1.0
            fused = fused + 0.15 * (dense / (scale or 1.0))
        return fused

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        with self._lock:
            return {
                "documents": len(self.documents),
                "chunks": len(self.chunks),
                "embeddings_ready": bool(self.vectors.size),
            }

    def list_documents(self) -> list[Document]:
        with self._lock:
            return sorted(self.documents.values(), key=lambda d: d.added_at, reverse=True)
