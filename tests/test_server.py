"""HTTP-level check of the running server: upload, list, chat, stream, delete."""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8000"
SAMPLES = ROOT / "samples"


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def upload(paths: list[Path]) -> dict:
    boundary = "----documindboundary"
    body = b""
    for path in paths:
        body += f"--{boundary}\r\n".encode()
        body += (
            f'Content-Disposition: form-data; name="files"; filename="{path.name}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        body += path.read_bytes()
        body += b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    request = urllib.request.Request(
        f"{BASE}/api/documents",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def chat(message: str) -> dict:
    payload = json.dumps({"message": message, "history": [], "doc_ids": None}).encode()
    request = urllib.request.Request(
        f"{BASE}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def chat_stream(message: str) -> list[dict]:
    payload = json.dumps({"message": message, "history": [], "doc_ids": None}).encode()
    request = urllib.request.Request(
        f"{BASE}/api/chat/stream",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    events: list[dict] = []
    with urllib.request.urlopen(request, timeout=180) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("event:"):
                events.append({"event": line[6:].strip()})
            elif line.startswith("data:"):
                events[-1]["data"] = json.loads(line[5:].strip())
    return events


def main() -> int:
    failures = 0
    print("health:", json.dumps(get("/api/health")["llm"]))

    files = [p for p in sorted(SAMPLES.iterdir()) if p.suffix.lower() in {".txt", ".csv", ".docx", ".pdf", ".pptx"}]
    print("uploading:", [f.name for f in files])
    result = upload(files)
    for item in result["results"]:
        print(f"  {'OK ' if item['ok'] else 'ERR'} {item['filename']} {item.get('error','')}")
        if not item["ok"]:
            failures += 1

    docs = get("/api/documents")
    print("documents:", len(docs["documents"]), "stats:", docs["stats"])

    answer = chat("What is the maximum hotel reimbursement per night in a Tier 1 city?")
    print("\nchat answer:", answer["answer"])
    print("sources:", [s["citation"] for s in answer["sources"]])
    print("grounded:", answer["grounded"], "mode:", answer["mode"])
    if "1200" not in answer["answer"]:
        print("!! FAIL: expected 1200 USD")
        failures += 1

    events = chat_stream("When does the Apollo product launch?")
    kinds = [e["event"] for e in events]
    print("\nstream events:", kinds)
    text = "".join(e.get("data", {}).get("text", "") for e in events if e["event"] == "delta")
    print("stream text:", text)
    done = [e for e in events if e["event"] == "done"]
    if not done:
        print("!! FAIL: no done event")
        failures += 1
    elif "15 October 2025" not in done[0]["data"]["answer"]:
        print("!! FAIL: streamed answer missing the launch date")
        failures += 1
    if "sources" not in kinds:
        print("!! FAIL: no sources event")
        failures += 1

    # HTML shell
    with urllib.request.urlopen(f"{BASE}/", timeout=30) as response:
        html = response.read().decode("utf-8")
    print("\nindex.html served:", response.status, len(html), "bytes")
    for asset in ("/static/app.js", "/static/styles.css"):
        with urllib.request.urlopen(f"{BASE}{asset}", timeout=30) as response:
            print(f"{asset}:", response.status, len(response.read()), "bytes")
            if response.status != 200:
                failures += 1

    # delete a document
    target = docs["documents"][0]
    request = urllib.request.Request(f"{BASE}/api/documents/{target['doc_id']}", method="DELETE")
    with urllib.request.urlopen(request, timeout=60) as response:
        print("\ndelete:", json.loads(response.read().decode("utf-8")))

    print("\nFAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
