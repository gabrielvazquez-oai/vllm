#!/usr/bin/env python3
"""Pass Codex through unchanged and capture bounded diagnostics in its private home.

The workflow redacts the private log before publishing it. Never record stdin,
successful RPC results, agent messages, tool output, auth.json, or session files.
"""

import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time

MAX_LINE = 256 * 1024
MAX_LOG = 2 * 1024 * 1024


def main():
    real_cli = os.environ["CODEX_DIAGNOSTICS_REAL_CLI"]
    environment = dict(os.environ)
    environment.pop("CODEX_DIAGNOSTICS_REAL_CLI", None)
    environment.pop("CODEX_CLI_PATH", None)
    if sys.argv[1:2] != ["app-server"]:
        os.execve(real_cli, [real_cli, *sys.argv[1:]], environment)

    log_path = pathlib.Path(environment["CODEX_HOME"]) / "app-server-diagnostics.jsonl"
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    lock = threading.Lock()
    written = 0
    capped = False

    with os.fdopen(log_fd, "w", encoding="utf-8") as log:
        def record(source, **details):
            nonlocal written, capped
            entry = json.dumps({"time": time.time(), "source": source, **details}) + "\n"
            size = len(entry.encode("utf-8"))
            with lock:
                if written + size > MAX_LOG:
                    if not capped:
                        log.write('{"source":"diagnostics","message":"Log size limit reached; further records omitted."}\n')
                        log.flush()
                        capped = True
                    return
                if capped:
                    return
                log.write(entry)
                log.flush()
                written += size

        def observe(event):
            if not isinstance(event, dict):
                return
            params = event.get("params")
            params = params if isinstance(params, dict) else {}
            method = event.get("method", "")
            if "error" in event:
                record("rpc", id=event.get("id"), error=event["error"])
            elif method == "error":
                record("rpc", method=method, error=params.get("error"),
                       thread_id=params.get("threadId"), turn_id=params.get("turnId"),
                       will_retry=params.get("willRetry"))
            elif method in ("turn/started", "turn/completed"):
                turn = params.get("turn")
                if isinstance(turn, dict):
                    record("rpc", method=method, thread_id=params.get("threadId"),
                           turn_id=turn.get("id"), status=turn.get("status"),
                           error=turn.get("error"))
            elif method == "codex/event/error":
                message = params.get("msg")
                if isinstance(message, dict):
                    record("rpc", method=method, error=message.get("message"),
                           error_info=message.get("codex_error_info"))

        record("process", event="start", command="app-server")
        try:
            child = subprocess.Popen([real_cli, *sys.argv[1:]], stdin=sys.stdin,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     env=environment)
        except OSError as error:
            record("process", event="start_failed", error=str(error))
            return 1

        def forward_signal(signum, _frame):
            if child.poll() is None:
                child.send_signal(signum)

        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, forward_signal)

        def forward(stream, destination, source):
            oversized = False
            try:
                while chunk := stream.readline(MAX_LINE):
                    destination.write(chunk)
                    destination.flush()
                    complete = chunk.endswith(b"\n") or len(chunk) < MAX_LINE
                    if not complete:
                        if not oversized:
                            record("diagnostics", message=f"Oversized {source} line omitted.")
                        oversized = True
                        continue
                    if oversized:
                        oversized = False
                        continue
                    text = chunk.decode("utf-8", errors="replace").rstrip("\r\n")
                    if source == "stderr":
                        if text:
                            record("stderr", message=text)
                    else:
                        try:
                            observe(json.loads(text))
                        except (ValueError, TypeError, RecursionError):
                            pass
            except (BrokenPipeError, OSError):
                if child.poll() is None:
                    child.terminate()
            finally:
                stream.close()

        readers = [
            threading.Thread(target=forward, args=(child.stdout, sys.stdout.buffer, "stdout")),
            threading.Thread(target=forward, args=(child.stderr, sys.stderr.buffer, "stderr")),
        ]
        for reader in readers:
            reader.start()
        status = child.wait()
        for reader in readers:
            reader.join()
        record("process", event="exit", returncode=status)
        return status if status >= 0 else 128 - status


if __name__ == "__main__":
    raise SystemExit(main())
