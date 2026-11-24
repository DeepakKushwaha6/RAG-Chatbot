# DocuMind — RAG Chatbot over Your Own Documents

Upload a document (PDF, Word, PowerPoint, Excel, CSV, TXT, Markdown, HTML, JSON…),
ask any question about it, and get an answer that is **generated only from the
uploaded content**, with every claim linked back to the exact source passage.

The retrieval layer is fully local and offline: documents are parsed, chunked and
embedded with a local ONNX model of `all-MiniLM-L6-v2`, then searched with a
hybrid of dense cosine similarity and BM25. Only the final wording comes from an
LLM, and it is strictly grounded in the retrieved excerpts.

---

## Quick start

```bash
pip install -r requirements.txt
python run.py
```

Then open **http://127.0.0.1:8000** in a browser.

On Windows you can also double-click **`start.bat`**.

The first run downloads the ~80 MB embedding model once; after that everything
except the LLM call works offline.

---

## How it works

```
upload file
    │
    ▼
parse  ──►  PDF / DOCX / PPTX / XLSX / CSV / TXT / MD / HTML / JSON
             each parsed into "sections" that keep their location
             (page number, sheet name, slide index, heading)
    │
    ▼
chunk  ──►  ~1200-char overlapping chunks, still carrying that location
    │
    ▼
embed  ──►  local all-MiniLM-L6-v2 (384-dim, ONNX, no API calls)
    │
    ▼
index  ──►  dense matrix + BM25, persisted to data/index/
    │
    ▼
retrieve ─►  query expansion → hybrid search → Reciprocal Rank Fusion
             → per-document diversification → top-k excerpts
    │
    ▼
answer ──►  LLM sees ONLY the excerpts, must cite [1] [2] …
             citation numbers outside the real range are stripped
    │
    ▼
show   ──►  answer + clickable citations + source panel with the
             original passage text
```

### Why the answers are correct

- **Grounded generation.** The model receives only retrieved excerpts. The
  system prompt forbids outside knowledge and requires a citation after every
  claim.
- **Honest refusals.** If the answer is not in the excerpts the model must say
  *"I could not find this in the uploaded document(s)"* — the UI marks this
  answer as **not found in documents**.
- **Source panel.** Every answer is accompanied by the exact passages used, with
  file name, page/sheet/slide and the original wording, so you can verify it.
- **No fabricated citations.** Citation numbers beyond the real excerpt range
  are removed before the answer is displayed.
- **Hybrid retrieval.** Dense embeddings catch paraphrases; BM25 catches exact
  terms, figures and identifiers (with light stemming so *deadlines* matches
  *deadline*). Reciprocal Rank Fusion merges both rankings.
- **Multi-document support.** Several files can be indexed at once and searched
  together; the source panel shows which file each excerpt came from.
- **Comparative questions.** "Compare A with B" style questions are split and
  each side retrieved independently so neither document crowds the other out.

---

## Supported file types

| Type | Extensions | Location shown in citations |
|------|-----------|------------------------------|
| PDF | `.pdf` | Page number |
| Word | `.docx` | Heading section |
| PowerPoint | `.pptx` | Slide number |
| Excel | `.xlsx` | Sheet name |
| CSV / TSV | `.csv`, `.tsv` | Row numbers |
| Text | `.txt`, `.md`, `.rst`, `.log`, … | Paragraph |
| Web | `.html`, `.htm`, `.xml` | Page title |
| Data | `.json` | Key path |
| Code | `.py`, `.js`, `.java`, `.sql`, … | Paragraph |

PDF text is extracted with `pdfplumber` (including tables) and falls back to
`pypdf`. Scanned/image-only PDFs have no text layer and are reported as
unreadable rather than silently returning nothing.

---

## LLM configuration

The answer wording needs an LLM. The app finds one automatically in this order:

1. **MiniMax platform API** — set `MINIMAX_API_KEY` in `.env`.
2. **MiniMax session token** — if you are signed in to MiniMax Code on this
   machine, the running session token is reused automatically (labelled
   `minimax-session` in the sidebar).
3. **Any OpenAI-compatible endpoint** — set `OPENAI_API_KEY` (works with OpenAI,
   Groq, Together, LM Studio, vLLM, …).
4. **Local Ollama** — if a server is running on `OLLAMA_BASE_URL`.
5. **Extractive fallback** — no LLM at all. The app returns the original
   matching sentences quoted verbatim from your documents, each with its source.
   Answers stay correct, they are just not rephrased.

Copy `.env.example` to `.env` to override anything:

```bash
cp .env.example .env
```

| Variable | Purpose |
|----------|---------|
| `RAG_LLM_PROVIDER` | `auto` (default), `minimax`, `openai`, `ollama` |
| `MINIMAX_API_KEY` | MiniMax platform key |
| `MINIMAX_MODEL` | Default `MiniMax-M3.1-Flash` |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` | OpenAI-compatible provider |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | Local Ollama |
| `RAG_TOP_K` | How many excerpts to retrieve (default 6) |
| `RAG_CHUNK_CHARS` / `RAG_CHUNK_OVERLAP` | Chunk size and overlap |
| `RAG_MAX_CONTEXT_CHARS` | Context budget sent to the LLM |
| `RAG_HOST` / `RAG_PORT` | Server bind address (default 127.0.0.1:8000) |

---

## Command line

The same engine works without the browser:

```bash
python -m app.cli add report.pdf notes.docx      # index documents
python -m app.cli list                            # show indexed documents
python -m app.cli ask "What is the expense claim deadline?"
python -m app.cli ask --doc report.pdf "Summarise this document"
python -m app.cli remove <doc_id>
python -m app.cli clear
```

---

## HTTP API

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/api/health` | Status, index stats, active LLM provider |
| `GET` | `/api/documents` | List indexed documents |
| `POST` | `/api/documents` | Upload (multipart, field name `files`) |
| `DELETE` | `/api/documents/{doc_id}` | Remove one document |
| `POST` | `/api/reset` | Remove everything |
| `POST` | `/api/chat` | One-shot answer + sources |
| `POST` | `/api/chat/stream` | SSE stream: `sources` → `delta` → `done` |

Example:

```bash
curl -F "files=@report.pdf" http://127.0.0.1:8000/api/documents

curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the expense claim deadline?","history":[],"doc_ids":null}'
```

---

## Project layout

```
app/
  config.py       settings loaded from .env
  models.py       Document / Chunk / Source / Answer dataclasses
  loaders.py      per-format text extraction with location metadata
  chunking.py     overlap-aware chunking
  embeddings.py   local ONNX MiniLM embeddings
  store.py        persistent index + hybrid dense/BM25 search
  engine.py       retrieve → ground → answer → verify
  llm.py          MiniMax / OpenAI / Ollama providers
  prompts.py      grounded-answer prompt
  main.py         FastAPI app and routes
  cli.py          terminal interface
web/              ChatGPT-style frontend (vanilla JS, no build step)
tests/            end-to-end pipeline and HTTP tests
data/             uploaded files + index (created at runtime)
```

---

## Tests

```bash
python tests/test_pipeline.py    # builds sample docs, asks 12 questions, checks answers
python run.py                    # then, in another shell:
python tests/test_server.py      # upload → chat → stream → delete over HTTP
```

`test_pipeline.py` generates its own sample PDF/DOCX/PPTX/CSV/TXT, indexes them,
and verifies that policy figures come back exactly, that comparative questions
reach both documents, and that an out-of-scope question is refused.

---

## Notes and limits

- Scanned PDFs (image-only) need OCR, which is not included.
- `.doc` (legacy Word) is not supported; save as `.docx`.
- The index is stored per machine in `data/index/`; uploaded files stay in
  `data/uploads/`.
- The server binds to `127.0.0.1` by default. Set `RAG_HOST=0.0.0.0` to expose
  it on your network — it has no authentication, so only do that on a trusted
  network.
