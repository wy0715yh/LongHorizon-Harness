"""Phase 8 - advanced orchestration.

P8 adds four capabilities on top of the plan-act-observe loop. Each is a SMALL
enhancement of the loop - none of them is a rewrite:

  1. PARALLEL TOOLS  - when the model emits several tool calls in one turn, run
     the independent ones concurrently in a thread pool instead of one-by-one.
     The admission gates (guardrail / circuit / backpressure) still run first
     and SEQUENTIALLY; only the *execution* is parallelised. Results are
     committed back to the conversation on the MAIN thread, so the durable
     event log stays single-writer and safe.

  2. HUMAN-IN-THE-LOOP - a tool can be flagged ``human=True``. When the model
     calls it, the engine PAUSES and asks a callback (default: input()) for the
     answer, then feeds that back as the tool result. The autonomous loop stops
     and waits for a person - exactly what "human in the loop" means.

  3. SUB-TASK DELEGATION - a ``delegate`` meta-tool runs a NESTED engine on a
     sub-prompt and returns its final answer. This is hierarchical planning: a
     boss agent breaks work into a sub-task, delegates, and folds the result
     back in. The nested engine is a LEAF (no further delegation, no critic) so
     we never recurse forever.

  4. STREAMING - the model may yield its next message token-by-token. The engine
     assembles the deltas and prints them live, then treats the assembled
     message exactly like a normal one (tool_calls + usage come from the last
     delta). This is the same contract a real SSE/streaming API uses.

Zero third-party dependencies. The thread pool uses stdlib concurrent.futures;
StateStore and Tracer are thread-safe (see those modules) so parallel workers
can append events and spans concurrently.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from .types import ToolCall, ToolResult
from .tools import ToolExecutor


# A human-in-the-loop callback: given a prompt + the tool args, return the
# human's answer (a string). Interactive default usage (in a real TTY):
#     lambda prompt, args: input(prompt + " ")
HumanInput = Callable[[str, dict], str]


def execute_calls(executor: ToolExecutor, calls: list[ToolCall], *,
                  store=None, task_id=None, tracer=None,
                  parallel: bool = True, max_parallel: int = 8) -> dict[str, ToolResult]:
    """Run tool calls and return {tool_call_id: ToolResult}.

    If ``parallel`` and there is more than one call, they run in a bounded
    thread pool. Each call's retries (P3) and tool_attempt events (Phase 5)
    happen inside its own worker thread; StateStore + Tracer are thread-safe
    so those writes are safe. We also record the top-level ``tool`` span here,
    from the worker thread, with the REAL execution duration. The ENGINE then
    commits the final tool_result / message events on its OWN thread (right
    after this returns) - so the conversation log stays single-writer and the
    order of results is deterministic (keyed by tool_call id).
    """
    results: dict[str, ToolResult] = {}

    def _run(tc: ToolCall) -> ToolResult:
        s = time.perf_counter()
        r = executor.execute(tc, store=store, task_id=task_id)
        e = time.perf_counter()
        attrs = {"duration_ms": round((e - s) * 1000, 2), "tool": tc.name,
                 "ok": r.ok, "attempts": r.attempts}
        if tracer is not None:
            tracer.span(f"tool.{tc.name}", "tool", s, e, attrs,
                        status="ok" if r.ok else "error")
        if store is not None:
            store.append(task_id, "span", {"kind": "tool", "name": tc.name,
                         **attrs, "status": "ok" if r.ok else "error"})
        return r

    if parallel and len(calls) > 1:
        workers = min(max_parallel, len(calls))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run, tc): tc for tc in calls}
            for fut in as_completed(futures):
                tc = futures[fut]
                results[tc.id] = fut.result()
    else:
        for tc in calls:
            results[tc.id] = _run(tc)
    return results


class Delegate:
    """A meta-tool that runs a SUB-task in a fresh nested Engine.

    The boss engine calls the tool named ``tool_name`` (default "delegate")
    with a ``sub_prompt`` argument. We spin up a leaf engine - the SAME registry
    of tools, but a *worker* model we pass in (so the worker actually solves,
    while the boss only plans) - and return its final answer as the tool result.

    The nested engine is intentionally a LEAF: no further delegation, no critic,
    no streaming, no human pause, and it runs IN-MEMORY (no durable store). That
    guarantees we never recurse infinitely and keeps the sub-task cheap and
    bounded by ``max_steps``. The parent still records one 'delegate' span so
    the hand-off is observable.
    """

    def __init__(self, worker_model, *, tool_name: str = "delegate",
                 max_steps: int = 6):
        self.worker_model = worker_model
        self.tool_name = tool_name
        self.max_steps = max_steps

    def run(self, parent, sub_prompt: str) -> str:
        # Local import avoids a circular import at module load time (engine.py
        # imports this module, and we import Engine from it).
        from .engine import Engine
        sub = Engine(self.worker_model, parent.registry,
                     max_steps=self.max_steps, system=parent.system,
                     parallel=parent._parallel, max_parallel=parent._max_parallel)
        # In-memory: the sub-task is a leaf and must not touch the parent's
        # durable event log or nest task bookkeeping.
        return sub.run(sub_prompt)
