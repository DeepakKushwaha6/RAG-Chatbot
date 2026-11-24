"""Start DocuMind as a detached background process.

The server is launched with ``subprocess`` in a way that survives the parent
process (and any terminal/agent session that started it). Logs are written to
``data/server.log``.

    python serve.py            # start (no-op if already running)
    python serve.py --stop     # stop it
    python serve.py --status   # is it running?
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_FILE = DATA_DIR / "server.log"
PID_FILE = DATA_DIR / "server.pid"
HEALTH_URL = "http://127.0.0.1:8000/health"


def is_running(timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=timeout) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001
        return False


def read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def start() -> int:
    if is_running():
        print("DocuMind is already running at http://127.0.0.1:8000")
        return 0

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_FILE, "ab", buffering=0)  # noqa: SIM115 - closed by the child
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008  # DETACHED_PROCESS

    process = subprocess.Popen(
        [sys.executable, str(BASE_DIR / "run.py")],
        cwd=str(BASE_DIR),
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
        close_fds=True,
    )
    PID_FILE.write_text(str(process.pid), encoding="utf-8")

    for _ in range(60):
        time.sleep(0.5)
        if is_running():
            print(f"DocuMind started (pid {process.pid}) -> http://127.0.0.1:8000")
            print(f"Logs: {LOG_FILE}")
            return 0

    print("DocuMind did not become healthy in 30s. Check the log:")
    print(LOG_FILE.read_text(encoding="utf-8", errors="replace")[-2000:])
    return 1


def stop() -> int:
    pid = read_pid()
    if pid is None:
        print("No pid file found; nothing to stop.")
        return 0
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                capture_output=True,
            )
        else:
            os.kill(pid, 15)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not stop pid {pid}: {exc}")
        return 1
    PID_FILE.unlink(missing_ok=True)
    print(f"Stopped DocuMind (pid {pid}).")
    return 0


def status() -> int:
    running = is_running()
    print("running" if running else "stopped")
    if running:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=3) as response:
                print(response.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return 0 if running else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DocuMind as a detached service.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stop", action="store_true", help="stop the running server")
    group.add_argument("--status", action="store_true", help="show whether it is running")
    args = parser.parse_args()

    if args.stop:
        return stop()
    if args.status:
        return status()
    return start()


if __name__ == "__main__":
    sys.exit(main())
