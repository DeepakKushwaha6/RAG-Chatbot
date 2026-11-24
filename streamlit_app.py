"""DocuMind - RAG chatbot over uploaded documents (Streamlit edition).

Deploy to Streamlit Community Cloud by pointing it at this repository; the
file name and location are what the platform looks for.

Secrets (Settings -> Secrets in the Streamlit UI):

    OPENROUTER_API_KEY = "sk-or-v1-..."

The app auto-selects the best free model OpenRouter currently offers and
re-checks periodically, so it follows the free lineup as it changes. With no
key configured it falls back to extractive mode and quotes the original
passages from the uploaded documents.
"""

from __future__ import annotations

import os
import secrets
import shutil
import time
from pathlib import Path

import streamlit as st

# --------------------------------------------------------------------------
# Secrets must be in the environment before app.config is imported, because
# the settings module reads environment variables at import time.
# --------------------------------------------------------------------------
_SECRET_NAMES = (
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_MODEL",
    "OPENROUTER_REFERER",
    "OPENROUTER_TITLE",
    "OPENROUTER_REFRESH_SECONDS",
    "MINIMAX_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "RAG_LLM_PROVIDER",
    "RAG_DATA_DIR",
    "RAG_TOP_K",
    "RAG_CHUNK_CHARS",
)


def _load_secrets() -> None:
    try:
        secrets = st.secrets
    except Exception:  # noqa: BLE001 - no secrets configured, or not on Cloud
        return
    for name in _SECRET_NAMES:
        if os.getenv(name):
            continue
        try:
            value = secrets.get(name)
        except Exception:  # noqa: BLE001
            value = None
        if value:
            os.environ[name] = str(value)


_load_secrets()

from app.config import settings  # noqa: E402
from app.engine import RAGEngine  # noqa: E402
from app.ingest import IngestError, clear_all, ingest_bytes, remove_document  # noqa: E402
from app.loaders import SUPPORTED_EXTENSIONS  # noqa: E402
from app.store import Store  # noqa: E402

st.set_page_config(
    page_title="DocuMind - RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

FREE_MODELS_FILE = Path(settings.data_dir) / "free_models.json"
SESSIONS_DIR = Path(settings.data_dir) / "sessions"

# How long an idle session's documents are kept on disk. Public deployments
# share one container, so old session folders are swept away.
SESSION_TTL_SECONDS = 6 * 60 * 60


def _file_key(file) -> str:
    """A stable identity for an uploaded file.

    Streamlit keeps the uploader's files in the widget state, so they are still
    present on every rerun. Without this key the same file would be indexed
    again on each rerun (every chat message triggers one).
    """

    file_id = getattr(file, "file_id", None)
    if file_id:
        return f"id:{file_id}"
    return f"hash:{file.name}:{file.size}:{hash(file.getvalue())}"


@st.cache_resource(show_spinner="Loading the document index…")
def sweep_old_sessions() -> int:
    """Delete session folders that have not been touched for a while."""

    if not SESSIONS_DIR.exists():
        return 0
    removed = 0
    cutoff = time.time() - SESSION_TTL_SECONDS
    for folder in SESSIONS_DIR.iterdir():
        if not folder.is_dir():
            continue
        try:
            if folder.stat().st_mtime < cutoff:
                shutil.rmtree(folder, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed


def get_store() -> Store:
    """Return this browser session's own index.

    Every visitor gets an isolated document set, so nobody can see the files
    another visitor uploaded to the same public deployment.
    """

    store = st.session_state.get("_store")
    if store is None:
        session_id = st.session_state.get("_session_id")
        if not session_id:
            session_id = secrets.token_hex(8)
            st.session_state["_session_id"] = session_id
            sweep_old_sessions()
        base = SESSIONS_DIR / session_id
        base.mkdir(parents=True, exist_ok=True)
        store = Store(index_dir=base / "index", upload_dir=base / "uploads")
        st.session_state["_store"] = store
    else:
        # Keep the session folder "recent" so the sweeper leaves it alone.
        try:
            store.index_dir.parent.touch()
        except OSError:
            pass
    return store


def get_engine(store: Store) -> RAGEngine:
    """Return this session's answer engine."""

    engine = st.session_state.get("_engine")
    if engine is None:
        engine = RAGEngine(store)
        st.session_state["_engine"] = engine
    return engine


def free_model_panel() -> None:
    """Show the live free-model lineup and let the user refresh it."""

    from app import openrouter

    with st.expander("Free model selection", expanded=False):
        st.caption(
            "The app asks OpenRouter which models are free right now and uses "
            "the best one, re-checking every "
            f"{max(settings.openrouter_refresh_seconds, 60) // 60} minutes."
        )
        if st.button("Refresh free models now", use_container_width=True):
            openrouter.fetch_free_models(force=True)
            st.rerun()

        summary = openrouter.free_model_summary()
        if summary["error"]:
            st.warning(f"Could not reach OpenRouter: {summary['error']}")
        if not summary["count"]:
            st.info("No free model found. Check the API key, or the app is in extractive mode.")
        for index, model in enumerate(summary["models"], start=1):
            marker = " ← selected" if index == 1 else ""
            st.markdown(f"`{model['id']}`{marker}")


def sidebar() -> None:
    store = get_store()
    engine = get_engine(store)

    with st.sidebar:
        st.markdown("### 📄 DocuMind")
        st.caption("RAG chatbot over your own documents")

        st.divider()

        uploaded = st.file_uploader(
            "Upload documents",
            type=[ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)],
            accept_multiple_files=True,
            help="PDF, DOCX, PPTX, XLSX, CSV, TXT, Markdown, HTML, JSON and source files.",
        )
        ingested = st.session_state.setdefault("_ingested", {})
        if uploaded:
            for file in uploaded:
                key = _file_key(file)
                if key in ingested:
                    # Already indexed on an earlier rerun - do not add it twice.
                    continue
                try:
                    with st.spinner(f"Indexing {file.name}…"):
                        document = ingest_bytes(store, file.name, file.read())
                    ingested[key] = document.doc_id
                    st.success(f"{document.filename} · {document.chunks} chunks")
                except IngestError as exc:
                    st.error(f"{file.name}: {exc}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"{file.name}: {exc}")

        documents = store.list_documents()
        stats = store.stats()
        st.caption(f"**{stats['documents']}** documents · **{stats['chunks']}** chunks")

        if documents:
            st.markdown("**Indexed documents**")
            for document in documents:
                column, remove = st.columns([5, 1])
                column.markdown(
                    f"`{document.filename}`<br>"
                    f"<span style='color:#888;font-size:0.8em'>{document.pages} sections "
                    f"· {document.chunks} chunks</span>",
                    unsafe_allow_html=True,
                )
                if remove.button("🗑", key=f"del_{document.doc_id}", help="Remove"):
                    remove_document(store, document.doc_id)
                    for key, doc_id in list(ingested.items()):
                        if doc_id == document.doc_id:
                            ingested.pop(key, None)
                    st.rerun()

        st.divider()
        status = engine.provider_status()
        if status["available"]:
            st.success(f"Engine: {status['provider']}")
        else:
            st.warning("Engine: extractive mode (no LLM key)")

        free_model_panel()

        if documents and st.button("Clear all documents", use_container_width=True):
            clear_all(store)
            st.rerun()


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)})"):
        for source in sources:
            st.markdown(f"**[{source['index']}] {source['filename']}** — {source['citation']}")
            st.caption(f"retrieval score {source['score']:.4f}")
            st.code(source["snippet"], language=None)


def render_answer(message: dict) -> None:
    st.markdown(message["content"])
    if message.get("warning"):
        st.warning(message["warning"])
    render_sources(message.get("sources", []))
    meta = []
    if message.get("mode"):
        meta.append(f"mode: {message['mode']}")
    meta.append("grounded" if message.get("grounded") else "not found in documents")
    if message.get("elapsed_ms") is not None:
        meta.append(f"{message['elapsed_ms']} ms")
    st.caption(" · ".join(meta))


def main() -> None:
    sidebar()
    store = get_store()
    engine = get_engine(store)

    st.title("Ask anything about your documents")
    st.caption(
        "Upload a document on the left, then ask a question. Answers are generated "
        "only from the uploaded content, and every claim links back to its source."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "📄"):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_answer(message)

    prompt = st.chat_input(
        "Ask a question about your documents…"
        if store.stats()["documents"]
        else "Upload a document first, then ask your question"
    )
    if not prompt:
        if not store.stats()["documents"]:
            st.info("No documents indexed yet — upload one to get started.")
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages[:-1][-8:]
    ]

    with st.chat_message("assistant", avatar="📄"):
        with st.spinner("Searching your documents…"):
            result = engine.answer(prompt, history=history)

        st.markdown(result.answer)
        if result.warning:
            st.warning(result.warning)
        render_sources([source.to_dict() for source in result.sources])
        meta = [f"mode: {result.mode}"]
        meta.append("grounded" if result.grounded else "not found in documents")
        meta.append(f"{result.elapsed_ms} ms")
        st.caption(" · ".join(meta))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "sources": [source.to_dict() for source in result.sources],
            "mode": result.mode,
            "grounded": result.grounded,
            "elapsed_ms": result.elapsed_ms,
            "warning": result.warning,
        }
    )


if __name__ == "__main__":
    main()
else:
    main()
