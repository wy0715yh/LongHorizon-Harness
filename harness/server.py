"""Phase 9 - the service layer: a thin HTTP wrapper around the Engine.

The Engine is the product. P9 makes it *deployable*: a tiny stdlib
``http.server`` that accepts a task (a prompt + optional config) and returns
the answer plus its trace as JSON. We deliberately use the library that ships
with Python - no FastAPI/Flask - so the "zero third-party dependency" promise
holds all the way to serving in production.

Design: the service is a THIN WRAPPER. It does not change the loop; it only:
  * builds an Engine from the request config (via an injectable factory),
  * calls Engine.run(),
  * serializes Engine.export_trace() to JSON.

``ThreadingHTTPServer`` gives one thread per request, so concurrent tasks run
in parallel - and our StateStore/Tracer are already thread-safe (P8), so that
is safe.

Endpoints
---------
  GET  /health        -> {"status": "ok"}
  POST /run           -> body JSON {"prompt", "model", "max_steps", "db",
                                     "task_id", ...}; returns
                                     {"answer", "task_id", "trace"}
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

from .engine import Engine
from .models import DummyModel
from .tools import ToolRegistry, calculator
from .types import Message, Role, ToolCall


# A factory the caller supplies: given a request config dict, return a READY
# Engine. This keeps the server decoupled from HOW engines are built (real API
# keys, which tools, etc.) - dependency inversion again.
EngineFactory = Callable[[dict], Engine]


def demo_registry() -> ToolRegistry:
    """A small registry the offline server / CLI can use out of the box."""
    reg = ToolRegistry()
    reg.register("calculator", "Evaluate a basic arithmetic expression",
                 {"type": "object",
                  "properties": {"expr": {"type": "string"}},
                  "required": ["expr"]},
                 calculator)
    reg.register("echo", "Echo the given text back",
                 {"type": "object",
                  "properties": {"text": {"type": "string"}},
                  "required": ["text"]},
                 lambda a: str(a.get("text", "")))
    return reg


def demo_engine_factory(config: dict) -> Engine:
    """Default offline engine: DummyModel + demo tools, no persistence.

    The CLI and the service both fall back to this when no custom factory is
    injected - so ``python -m harness serve`` works with zero setup."""
    reg = demo_registry()
    return Engine(DummyModel(), reg,
                  max_steps=int(config.get("max_steps", 10)),
                  parallel=bool(config.get("parallel", True)),
                  max_parallel=int(config.get("max_parallel", 8)),
                  stream=False)


class HarnessHandler(BaseHTTPRequestHandler):
    """One request handler. The class-level ``factory`` is set before serving
    (defaults to the offline ``demo_engine_factory``)."""

    factory: EngineFactory = staticmethod(demo_engine_factory)

    # ---- helpers -----------------------------------------------------------
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- routes ------------------------------------------------------------
    def do_GET(self):
        if self.path.rstrip("/") in ("", "/health"):
            self._send(200, {"status": "ok", "service": "LongHorizon-Harness"})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/run":
            self._send(404, {"error": "only POST /run is supported"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length else b"{}"
            req = json.loads(raw or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send(400, {"error": "invalid JSON body"})
            return

        prompt = req.get("prompt")
        if not prompt:
            self._send(400, {"error": "missing 'prompt'"})
            return

        # Optional durable store -> enables resume via task_id on later calls.
        store = None
        db_path = req.get("db")
        if db_path:
            from .state import StateStore
            store = StateStore(db_path)
        task_id = req.get("task_id")  # resume an existing task if given

        try:
            engine = self.factory(req)
            answer = engine.run(prompt, task_id=task_id, store=store)
            trace = engine.export_trace(task_id=engine.last_task_id, fmt="json")
        except Exception as exc:  # never let one bad task crash the server
            self._send(500, {"error": f"engine failed: {exc}"})
            return

        self._send(200, {
            "answer": answer,
            "task_id": engine.last_task_id,
            "trace": trace,
        })

    # Quieter default access logs.
    def log_message(self, fmt, *args):
        print(f"[http] {fmt % args}")


def serve(host: str = "127.0.0.1", port: int = 8080,
          factory: Optional[EngineFactory] = None,
          daemon: bool = False) -> ThreadingHTTPServer:
    """Start the HTTP service.

    If ``daemon`` is True, run in a background thread and RETURN the server
    (used by the benchmark / tests so they can hit it without blocking).
    Otherwise this blocks until Ctrl-C."""
    if factory is not None:
        HarnessHandler.factory = staticmethod(factory)
    httpd = ThreadingHTTPServer((host, port), HarnessHandler)
    if daemon:
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        print(f"[http] serving on http://{host}:{port} (background thread)")
        return httpd
    print(f"LongHorizon-Harness serving on http://{host}:{port}  (POST /run)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down.")
    finally:
        httpd.server_close()
    return httpd
