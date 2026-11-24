"""FastAPI application: upload documents, index them, chat over them."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import settings
from .engine import RAGEngine
from .ingest import IngestError, clear_all, ingest_bytes, remove_document
from .loaders import SUPPORTED_EXTENSIONS
from .store import Store

app = FastAPI(title="DocuMind RAG Chatbot", version="1.0.0")

store = Store()
engine = RAGEngine(store)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    doc_ids: list[str] | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    mode: str
    grounded: bool
    elapsed_ms: int
    used_docs: list[str] = []
    follow_ups: list[str] = []
    warning: str | None = None


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/api/health")
@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "store": store.stats(),
        "llm": engine.provider_status(),
        "supported": sorted(SUPPORTED_EXTENSIONS),
    }


@app.get("/v1/models")
async def openai_models() -> dict:
    """OpenAI-compatible model list, so tooling can discover this service."""

    status = engine.provider_status()
    model_id = status["provider"].split(":", 1)[-1] if status["provider"] else "extractive"
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": created,
                "owned_by": status["provider"] or "local",
            }
        ],
    }


@app.get("/api/documents")
async def list_documents() -> dict:
    return {
        "documents": [d.to_dict() for d in store.list_documents()],
        "stats": store.stats(),
        "llm": engine.provider_status(),
    }


@app.post("/api/documents")
async def upload_documents(files: list[UploadFile] = File(...)) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    results: list[dict] = []
    for upload in files:
        filename = Path(upload.filename or "document").name
        data = await upload.read()
        try:
            document = await run_in_threadpool(ingest_bytes, store, filename, data)
        except IngestError as exc:
            results.append({"filename": filename, "ok": False, "error": str(exc)})
            continue
        except Exception as exc:  # noqa: BLE001
            results.append({"filename": filename, "ok": False, "error": f"Parse error: {exc}"})
            continue
        results.append({"filename": filename, "ok": True, "document": document.to_dict()})

    return {"results": results, "stats": store.stats()}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str) -> dict:
    if not remove_document(store, doc_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"ok": True, "stats": store.stats()}


@app.post("/api/reset")
async def reset() -> dict:
    await run_in_threadpool(clear_all, store)
    return {"ok": True, "stats": store.stats()}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")
    result = await run_in_threadpool(
        engine.answer, question, payload.doc_ids, payload.history[-8:]
    )
    return ChatResponse(**result.to_dict())


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest) -> StreamingResponse:
    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    async def event_source():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def produce():
            try:
                for event in engine.stream_answer(
                    question, payload.doc_ids, payload.history[-8:]
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"event": "done", "data": {"answer": f"Unexpected error: {exc}", "sources": []}},
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        task = loop.run_in_executor(None, produce)
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
        finally:
            await task

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def run_in_threadpool(func, *args):
    return await asyncio.to_thread(func, *args)
