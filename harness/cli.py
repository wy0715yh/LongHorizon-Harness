"""Phase 9 - the command-line interface.

Two sub-commands, both thin wrappers around the Engine (the same Engine the
HTTP layer builds, so the CLI and the service are two faces of one product):

  run    run ONE prompt and print the answer (+ optional trace)
  serve  start the HTTP service (see harness.server)

The CLI is the fastest way to *see* the harness work end-to-end, and it is the
on-ramp to the interview demo: ``python -m harness run "what is 6*7"``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from .engine import Engine
from .models import DummyModel, OpenAICompatibleModel
from .server import demo_registry


def build_engine(args) -> Engine:
    """Construct an Engine from parsed CLI args (or a config dict)."""
    if getattr(args, "model", "dummy") == "openai":
        key = getattr(args, "openai_key", None) or os.environ.get("OPENAI_API_KEY")
        if not key:
            sys.exit("ERROR: --model openai requires --openai-key or "
                     "OPENAI_API_KEY env var")
        model = OpenAICompatibleModel(
            api_key=key,
            base_url=getattr(args, "openai_base", "https://api.openai.com/v1"),
            model=getattr(args, "openai_model", "gpt-4o-mini"),
        )
    else:
        model = DummyModel()
    reg = demo_registry()
    return Engine(model, reg,
                  max_steps=int(getattr(args, "max_steps", 10)),
                  parallel=bool(getattr(args, "parallel", True)),
                  max_parallel=int(getattr(args, "max_parallel", 8)),
                  stream=False)


def cmd_run(args) -> int:
    engine = build_engine(args)
    store = None
    db = getattr(args, "store", None)
    if db:
        from .state import StateStore
        store = StateStore(db)
    answer = engine.run(args.prompt, store=store)
    print("\n=== ANSWER ===")
    print(answer)
    if getattr(args, "trace", False):
        print("\n=== TRACE ===")
        rendered = engine.export_trace(task_id=engine.last_task_id, fmt="text")
        print(rendered.get("rendered", ""))
    return 0


def cmd_serve(args) -> int:
    from .server import serve
    serve(host=getattr(args, "host", "127.0.0.1"),
          port=int(getattr(args, "port", 8080)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="harness",
        description="LongHorizon-Harness - a lightweight reliable Agent runtime.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- run ---------------------------------------------------------------
    run = sub.add_parser("run", help="run a single prompt and print the answer")
    run.add_argument("prompt", help="the user prompt / task")
    run.add_argument("--max-steps", type=int, default=10)
    run.add_argument("--model", choices=["dummy", "openai"], default="dummy")
    run.add_argument("--openai-key")
    run.add_argument("--openai-base", default="https://api.openai.com/v1")
    run.add_argument("--openai-model", default="gpt-4o-mini")
    run.add_argument("--parallel", dest="parallel", action="store_true",
                     default=True)
    run.add_argument("--no-parallel", dest="parallel", action="store_false")
    run.add_argument("--max-parallel", type=int, default=8)
    run.add_argument("--store", default=None,
                     help="optional SQLite path for durable state + resume")
    run.add_argument("--trace", action="store_true",
                     help="print the execution trace after the answer")
    run.set_defaults(func=cmd_run)

    # ---- serve -------------------------------------------------------------
    sv = sub.add_parser("serve", help="start the HTTP service")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8080)
    sv.set_defaults(func=cmd_serve)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
