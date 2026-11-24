"""Retrieval diagnostic: show what the hybrid index returns for a query."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.engine import RAGEngine  # noqa: E402
from app.store import Store  # noqa: E402

store = Store(ROOT / "data" / "index")
engine = RAGEngine(store)

QUERIES = [
    "Compare the leave policy in the handbook with the travel policy: what notice periods and deadlines apply?",
    "what is the expense claim submission deadline",
    "How many days of annual leave do I get after 8 years of service?",
    "notice period probation employee employer",
]

for query in QUERIES:
    print("=" * 78)
    print("Q:", query)
    print("-- expanded queries --")
    for expanded in engine._expand_query(query, []):
        print("   *", expanded)
    print("-- retrieved --")
    for chunk, score in engine.retrieve(query):
        print(f"  {score:.4f}  {chunk.filename} — {chunk.location}")
        print(f"          {chunk.text[:150]}")
