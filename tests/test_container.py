"""Container-equivalent end-to-end check.

Simulates what the Docker image does at runtime:

* writes to a fresh RAG_DATA_DIR (the container uses /data)
* binds 0.0.0.0 on the $PORT the platform provides
* runs with RAG_LLM_PROVIDER=none, i.e. no API key at all - the mode a public
  deployment without a key will run in
* uploads a document, asks a question and checks the answer quotes the
  original passage

Run it while nothing else is bound to the port:

    python tests/test_container.py
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def get(url: str, timeout: float = 20.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_chat(base: str, message: str) -> dict:
    payload = json.dumps({"message": message, "history": [], "doc_ids": None}).encode()
    request = urllib.request.Request(
        f"{base}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def upload(base: str, path: Path) -> dict:
    boundary = "----documindtest"
    body = f"--{boundary}\r\n".encode()
    body += (
        f'Content-Disposition: form-data; name="files"; filename="{path.name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    body += path.read_bytes()
    body += f"\r\n--{boundary}--\r\n".encode()

    request = urllib.request.Request(
        f"{base}/api/documents",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    port = free_port()
    data_dir = ROOT / "data" / "_container_test"
    if data_dir.exists():
        import shutil

        shutil.rmtree(data_dir)

    env = dict(os.environ)
    env.update(
        {
            "RAG_DATA_DIR": str(data_dir),
            "RAG_HOST": "0.0.0.0",
            "PORT": str(port),
            "RAG_LLM_PROVIDER": "none",
        }
    )
    env.pop("MINIMAX_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)

    base = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    failures = 0
    try:
        for _ in range(80):
            time.sleep(0.5)
            try:
                health = get(f"{base}/health")
                break
            except Exception:  # noqa: BLE001
                health = None
        else:
            print("!! FAIL: server never became healthy")
            return 1

        print("health:", json.dumps(health["llm"]))
        print("store :", health["store"])
        if health["llm"]["available"]:
            print("!! FAIL: expected extractive mode with no key")
            failures += 1

        models = get(f"{base}/v1/models")
        print("models:", models["data"][0]["id"])
        if models["data"][0]["object"] != "model":
            failures += 1

        sample = ROOT / "samples" / "acme_policy.txt"
        if not sample.exists():
            print("!! FAIL: sample document missing")
            return 1
        result = upload(base, sample)
        if not result["results"][0]["ok"]:
            print("!! FAIL: upload failed:", result["results"][0].get("error"))
            failures += 1
        else:
            print("upload:", result["results"][0]["document"]["filename"], "indexed")

        answer = post_chat(base, "What is the maximum hotel reimbursement per night in a Tier 1 city?")
        print("\nanswer:", answer["answer"])
        print("mode  :", answer["mode"], "| grounded:", answer["grounded"])
        if "1200" not in answer["answer"]:
            print("!! FAIL: expected the original '1200 USD' figure")
            failures += 1
        if answer["mode"] != "extractive":
            print("!! FAIL: expected extractive mode")
            failures += 1
        if not answer["sources"]:
            print("!! FAIL: no sources returned")
            failures += 1
        else:
            print("source:", answer["sources"][0]["citation"])

        # The data directory must be the one we pointed the container at.
        if not (data_dir / "index" / "chunks.json").exists():
            print("!! FAIL: index was not written to RAG_DATA_DIR")
            failures += 1
        else:
            print("index written to:", data_dir / "index")
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()

    print("\nFAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
