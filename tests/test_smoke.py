"""Quick smoke test of the running server."""

from __future__ import annotations

import json
import urllib.request

BASE = "http://127.0.0.1:8000"

payload = json.dumps(
    {
        "message": "What is the maximum hotel reimbursement per night in a Tier 1 city, "
        "and how quickly must a security incident be reported?",
        "history": [],
        "doc_ids": None,
    }
).encode()

request = urllib.request.Request(
    f"{BASE}/api/chat",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=180) as response:
    result = json.loads(response.read().decode("utf-8"))

print("ANSWER:", result["answer"])
print("MODE:", result["mode"], "| grounded:", result["grounded"], "|", result["elapsed_ms"], "ms")
for source in result["sources"][:3]:
    print(f"  [{source['index']}] {source['citation']}")
