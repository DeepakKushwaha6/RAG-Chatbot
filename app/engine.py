"""The RAG engine: retrieve -> ground -> answer -> verify."""

from __future__ import annotations

import re
import time
from typing import Iterator

from . import llm, prompts
from .config import settings
from .models import Answer, Chunk, Source
from .store import Store, tokenize

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")
_QUESTION_SPLIT = re.compile(r"(?:\n+|(?<=[.?])\s+(?=(?:and|also|then|what|who|when|where|why|how)\b))", re.I)
_FOLLOWUP_RE = re.compile(
    r"^(and|also|then|what about|why|how|who|when|where|which|but|so)\b", re.I
)
# "compare A with B", "difference between A and B", "A versus B"
_COMPARE_RE = re.compile(
    r"compare\s+(?P<left>[^,]+?)\s+(?:with|to|against)\s+(?P<right>.+?)"
    r"(?:\s*[?.]|$)",
    re.I,
)
_COMPARE_RE2 = re.compile(
    r"difference\s+between\s+(?P<left>[^,]+?)\s+and\s+(?P<right>.+?)"
    r"(?:\s*[?.]|$)",
    re.I,
)


class RAGEngine:
    def __init__(self, store: Store) -> None:
        self.store = store
        self._provider = None
        self._provider_note = ""
        self._provider_checked = False
        self._provider_built_at = 0.0

    # ------------------------------------------------------------- provider
    def provider(self):
        """Return the active LLM provider, rebuilding it when it goes stale.

        The provider is not cached forever: OpenRouter rotates its free model
        lineup, so the selection is re-evaluated periodically and the app
        follows whichever free model is best at that moment.
        """

        now = time.time()
        if not self._provider_checked:
            self._provider, self._provider_note = llm.build_provider()
            self._provider_checked = True
            self._provider_built_at = now
            return self._provider

        label = getattr(self._provider, "label", "") or ""
        if label == "openrouter":
            ttl = max(settings.openrouter_refresh_seconds, 60)
        elif self._provider is None:
            ttl = 60
        else:
            ttl = 3600

        if now - self._provider_built_at > ttl:
            self._provider, self._provider_note = llm.build_provider()
            self._provider_built_at = now
        return self._provider

    def provider_status(self) -> dict:
        provider = self.provider()
        return {
            "available": provider is not None,
            "provider": self._provider_note,
            "mode": "llm" if provider else "extractive",
        }

    # --------------------------------------------------------------- query
    def _expand_query(self, question: str, history: list[dict]) -> list[str]:
        """Build a small set of search queries from the question + history."""

        queries = [question.strip()]
        tokens = tokenize(question)
        if tokens:
            queries.append(" ".join(tokens))

        # Multi-part question: search each part separately as well.
        parts = [
            p.strip()
            for p in _QUESTION_SPLIT.split(question)
            if p and len(p.strip()) > 8
        ]
        if len(parts) > 1:
            queries.extend(parts[:4])

        # Comparative questions ("compare A with B") need both sides retrieved
        # independently, otherwise one document can crowd the other out.
        for pattern in (_COMPARE_RE, _COMPARE_RE2):
            for match in pattern.finditer(question):
                left, right = match.group("left"), match.group("right")
                tail = " ".join(tokens)
                queries.append(f"{left} {tail}".strip())
                queries.append(f"{right} {tail}".strip())

        # Follow-up question: prepend the previous turn's subject for context.
        if history and _FOLLOWUP_RE.match(question.strip()):
            last_question = next(
                (m.get("content", "") for m in reversed(history) if m.get("role") == "user"),
                "",
            )
            if last_question:
                queries.append(f"{last_question} {question}".strip())

        seen: set[str] = set()
        unique: list[str] = []
        for query in queries:
            key = query.lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(query)
        return unique[:8]

    # ------------------------------------------------------------ retrieve
    def retrieve(
        self,
        question: str,
        doc_ids: list[str] | None = None,
        history: list[dict] | None = None,
    ) -> list[tuple[Chunk, float]]:
        queries = self._expand_query(question, history or [])
        merged: dict[str, tuple[Chunk, float]] = {}
        for query in queries:
            for chunk, score in self.store.search(
                query,
                top_k=max(settings.top_k * 2, settings.top_k + 4),
                doc_ids=doc_ids,
            ):
                existing = merged.get(chunk.chunk_id)
                if existing is None or score > existing[1]:
                    merged[chunk.chunk_id] = (chunk, score)
        ranked = sorted(merged.values(), key=lambda pair: pair[1], reverse=True)

        # Per-document diversification: keep at most 3 chunks per document in
        # the top slice so a single long document cannot crowd out others.
        selected: list[tuple[Chunk, float]] = []
        per_doc: dict[str, int] = {}
        for chunk, score in ranked:
            count = per_doc.get(chunk.doc_id, 0)
            if count >= 3 and len(selected) < settings.top_k:
                continue
            per_doc[chunk.doc_id] = count + 1
            selected.append((chunk, score))
            if len(selected) >= settings.top_k:
                break

        if not selected and ranked:
            selected = ranked[: settings.top_k]
        return selected

    # -------------------------------------------------------------- context
    @staticmethod
    def _build_context(chunks: list[tuple[Chunk, float]]) -> tuple[list[str], list[Source]]:
        blocks: list[str] = []
        sources: list[Source] = []
        used_chars = 0
        for index, (chunk, score) in enumerate(chunks, start=1):
            header = f"{chunk.filename} — {chunk.location}"
            body = chunk.text
            block = f"({header})\n{body}"
            if used_chars + len(block) > settings.max_context_chars and blocks:
                break
            blocks.append(block)
            used_chars += len(block)
            sources.append(
                Source(
                    index=index,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    filename=chunk.filename,
                    page=chunk.page,
                    section=chunk.section,
                    snippet=body[:1200],
                    score=round(float(score), 5),
                    citation=chunk.citation(),
                )
            )
        return blocks, sources

    # --------------------------------------------------------------- answer
    def answer(
        self,
        question: str,
        doc_ids: list[str] | None = None,
        history: list[dict] | None = None,
    ) -> Answer:
        started = time.time()
        retrieved = self.retrieve(question, doc_ids=doc_ids, history=history)
        blocks, sources = self._build_context(retrieved)

        if not blocks:
            return Answer(
                question=question,
                answer=(
                    "I could not find this in the uploaded document(s). "
                    "Upload a document or check the source filter - no matching text was found."
                ),
                sources=[],
                mode="no-context",
                elapsed_ms=int((time.time() - started) * 1000),
                grounded=False,
            )

        provider = self.provider()
        if provider is None:
            text = self._extractive_answer(question, sources)
            return Answer(
                question=question,
                answer=text,
                sources=sources,
                mode="extractive",
                elapsed_ms=int((time.time() - started) * 1000),
                grounded=True,
                used_docs=sorted({s.filename for s in sources}),
            )

        system = prompts.SYSTEM_PROMPT.format(max_source=len(blocks))
        user_prompt = prompts.build_user_prompt(question, blocks)
        messages = [{"role": "user", "content": user_prompt}]

        try:
            response = provider.complete(
                messages=messages, system=system, max_tokens=1800, temperature=0.1
            )
            text = response.text
            mode = "llm"
        except Exception as exc:  # noqa: BLE001
            text = self._extractive_answer(question, sources)
            return Answer(
                question=question,
                answer=text,
                sources=sources,
                mode="extractive-fallback",
                elapsed_ms=int((time.time() - started) * 1000),
                grounded=True,
                used_docs=sorted({s.filename for s in sources}),
                warning=f"The language model was unavailable ({exc}); showing the original matching passages instead.",
            )

        text = self._postprocess(text, len(sources))
        grounded = self._verify(text, sources)
        return Answer(
            question=question,
            answer=text,
            sources=sources,
            mode=mode,
            elapsed_ms=int((time.time() - started) * 1000),
            grounded=grounded,
            used_docs=sorted({s.filename for s in sources}),
            follow_ups=self._suggest_follow_ups(question, sources),
        )

    def stream_answer(
        self,
        question: str,
        doc_ids: list[str] | None = None,
        history: list[dict] | None = None,
    ) -> Iterator[dict]:
        """Yield SSE-friendly events: sources, then token deltas, then done."""

        started = time.time()
        retrieved = self.retrieve(question, doc_ids=doc_ids, history=history)
        blocks, sources = self._build_context(retrieved)

        if not blocks:
            yield {
                "event": "done",
                "data": {
                    "answer": (
                        "I could not find this in the uploaded document(s). "
                        "Upload a document or check the source filter."
                    ),
                    "sources": [],
                    "mode": "no-context",
                    "grounded": False,
                    "elapsed_ms": int((time.time() - started) * 1000),
                },
            }
            return

        yield {"event": "sources", "data": {"sources": [s.to_dict() for s in sources]}}

        provider = self.provider()
        if provider is None:
            text = self._extractive_answer(question, sources)
            yield {"event": "delta", "data": {"text": text}}
            yield {
                "event": "done",
                "data": {
                    "answer": text,
                    "sources": [s.to_dict() for s in sources],
                    "mode": "extractive",
                    "grounded": True,
                    "elapsed_ms": int((time.time() - started) * 1000),
                },
            }
            return

        system = prompts.SYSTEM_PROMPT.format(max_source=len(blocks))
        user_prompt = prompts.build_user_prompt(question, blocks)
        messages = [{"role": "user", "content": user_prompt}]

        collected: list[str] = []
        try:
            for piece in provider.stream(
                messages=messages, system=system, max_tokens=1800, temperature=0.1
            ):
                collected.append(piece)
                yield {"event": "delta", "data": {"text": piece}}
        except Exception as exc:  # noqa: BLE001
            text = self._extractive_answer(question, sources)
            yield {
                "event": "done",
                "data": {
                    "answer": text,
                    "sources": [s.to_dict() for s in sources],
                    "mode": "extractive-fallback",
                    "grounded": True,
                    "warning": f"Streaming failed ({exc}); showing original passages instead.",
                    "elapsed_ms": int((time.time() - started) * 1000),
                },
            }
            return

        raw = "".join(collected)
        final = self._postprocess(raw, len(sources))
        yield {
            "event": "done",
            "data": {
                "answer": final,
                "sources": [s.to_dict() for s in sources],
                "mode": "llm",
                "grounded": self._verify(final, sources),
                "elapsed_ms": int((time.time() - started) * 1000),
            },
        }

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _postprocess(text: str, max_source: int) -> str:
        text = text.strip()
        # Drop any citation numbers the model invented beyond the real range.
        def fix(match: re.Match) -> str:
            number = int(match.group(1))
            return match.group(0) if 1 <= number <= max_source else ""

        text = _CITATION_RE.sub(fix, text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _verify(text: str, sources: list[Source]) -> bool:
        """A grounded answer cites at least one valid source and is not a refusal."""

        lowered = text.lower()
        if "could not find this in the uploaded" in lowered:
            return False
        cited = {int(n) for n in _CITATION_RE.findall(text)}
        valid = {s.index for s in sources}
        return bool(cited & valid)

    def _extractive_answer(self, question: str, sources: list[Source]) -> str:
        """No-LLM fallback: quote the best matching original sentences."""

        if not sources:
            return "I could not find this in the uploaded document(s)."

        query_tokens = set(tokenize(question))
        lines: list[str] = []
        for source in sources[:4]:
            best_sentences = _best_sentences(source.snippet, query_tokens, limit=3)
            if not best_sentences:
                continue
            lines.append(f"**{source.citation}**")
            for sentence in best_sentences:
                lines.append(f"- {sentence} [{source.index}]")
        if not lines:
            top = sources[0]
            lines.append(f"**{top.citation}**")
            lines.append(f"- {top.snippet[:600]} [{top.index}]")
        lines.append("")
        lines.append("_Extractive mode: showing the original passages that matched, because no language model is connected._")
        return "\n".join(lines)

    @staticmethod
    def _suggest_follow_ups(question: str, sources: list[Source]) -> list[str]:
        if not sources:
            return []
        filename = sources[0].filename
        return [
            f"Summarise {filename} in 5 bullet points",
            "What are the key dates and numbers mentioned?",
        ]


def _best_sentences(text: str, query_tokens: set[str], limit: int = 3) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 25]
    if not sentences:
        return []
    scored: list[tuple[int, str]] = []
    for sentence in sentences:
        overlap = len(query_tokens & set(tokenize(sentence)))
        scored.append((overlap, sentence))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    chosen = [s for score, s in scored[:limit] if score > 0]
    if not chosen:
        chosen = [scored[0][1]]
    return chosen
