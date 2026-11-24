"""Command-line interface for DocuMind.

Ask questions about documents straight from the terminal.

    python -m app.cli add report.pdf notes.docx
    python -m app.cli ask "What is the expense claim deadline?"
    python -m app.cli ask --doc report.pdf "Summarise this document"
    python -m app.cli list
    python -m app.cli remove <doc_id>
    python -m app.cli clear
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from .chunking import chunk_sections
from .config import settings
from .engine import RAGEngine
from .loaders import UnsupportedDocument, is_supported, load_document
from .models import Document, new_doc_id
from .store import Store

store = Store()
engine = RAGEngine(store)


def cmd_add(args: argparse.Namespace) -> int:
    added = 0
    for raw in args.files:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            print(f"✕ {path} does not exist")
            continue
        if not is_supported(path.name):
            print(f"✕ {path.name} - unsupported file type")
            continue

        doc_id = new_doc_id(path.name)
        stored = settings.upload_dir / f"{doc_id}{path.suffix.lower()}"
        shutil.copyfile(path, stored)

        try:
            sections = load_document(stored)
        except UnsupportedDocument as exc:
            print(f"✕ {path.name} - {exc}")
            stored.unlink(missing_ok=True)
            continue

        chunks = chunk_sections(sections, doc_id, path.name)
        if not chunks:
            print(f"✕ {path.name} - no indexable content")
            stored.unlink(missing_ok=True)
            continue

        document = Document(
            doc_id=doc_id,
            filename=path.name,
            filetype=path.suffix.lower().lstrip("."),
            size_bytes=stored.stat().st_size,
            added_at=time.time(),
            pages=len(sections),
            chunks=len(chunks),
            characters=sum(len(s.text) for s in sections),
            path=str(stored),
        )
        store.add_document(document, chunks)
        print(f"✓ {path.name} - {len(sections)} sections, {len(chunks)} chunks indexed")
        added += 1
    return 0 if added else 1


def cmd_ask(args: argparse.Namespace) -> int:
    doc_ids = None
    if args.doc:
        names = {d.filename: d.doc_id for d in store.list_documents()}
        doc_ids = [names[n] for n in args.doc if n in names]
        if not doc_ids:
            print(f"No indexed document named {args.doc}")
            return 1

    history: list[dict] = []
    for question in args.questions:
        result = engine.answer(question, doc_ids=doc_ids, history=history)
        print(f"\nQ: {question}")
        print(f"A: {result.answer}")
        print(f"   [{result.mode} | grounded={result.grounded} | {result.elapsed_ms} ms]")
        for source in result.sources:
            print(f"   [{source.index}] {source.citation}")
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result.answer})
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    documents = store.list_documents()
    if not documents:
        print("No documents indexed. Use: python -m app.cli add <file>")
        return 0
    for doc in documents:
        print(f"{doc.doc_id}  {doc.filename}  ({doc.chunks} chunks, {doc.pages} sections)")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    if not store.remove_document(args.doc_id):
        print("Document not found")
        return 1
    for path in settings.upload_dir.glob(f"{args.doc_id}.*"):
        path.unlink(missing_ok=True)
    print(f"Removed {args.doc_id}")
    return 0


def cmd_clear(_: argparse.Namespace) -> int:
    store.clear()
    for path in settings.upload_dir.glob("*"):
        path.unlink(missing_ok=True)
    print("Cleared all documents")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="app.cli", description="DocuMind RAG chatbot CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="Index one or more documents")
    add.add_argument("files", nargs="+")
    add.set_defaults(func=cmd_add)

    ask = sub.add_parser("ask", help="Ask a question")
    ask.add_argument("questions", nargs="+")
    ask.add_argument("--doc", action="append", help="Restrict to a filename (repeatable)")
    ask.set_defaults(func=cmd_ask)

    listing = sub.add_parser("list", help="List indexed documents")
    listing.set_defaults(func=cmd_list)

    remove = sub.add_parser("remove", help="Remove one document by id")
    remove.add_argument("doc_id")
    remove.set_defaults(func=cmd_remove)

    clear = sub.add_parser("clear", help="Remove all documents")
    clear.set_defaults(func=cmd_clear)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
