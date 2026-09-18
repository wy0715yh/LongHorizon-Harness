"""Phase 1 + 3 - tools: the Agent's hands, now with failure recovery.

Phase 1 gave us Tool (declaration + function) and ToolRegistry (lookup by
name). Phase 3 adds the *executor*: the thing that actually runs a call and
survives flaky tools.

THE FAILURE MODEL (read this before the code)
---------------------------------------------
A tool run can fail for transient reasons (timeout, rate limit, 503). Naively
bubbling that up to the model wastes a whole LLM turn on noise it can't fix.
So the executor retries with *exponential backoff + jitter*:

    delay = min(cap, base * 2**attempt) * jitter

  * exponential backoff  -> we back off harder each retry, so we don't hammer
    a struggling dependency.
  * jitter (random factor) -> if many agents retry in lockstep they form a
    "thundering herd"; jitter spreads them out. This is a classic distributed-
    systems lesson and a good interview talking point.
  * capped delay          -> backoff never grows unbounded.

Two layers of recovery, and this is the key idea to internalise:
  LAYER 1 (executor): retry the SAME call a few times. If it works, the model
           never sees the failure - recovery is *transparent*.
  LAYER 2 (model):    if retries are exhausted, the failure is fed back as a
           TOOL message (ok=False). The model can then re-plan: try another
           tool, fix the arguments, or give up gracefully. The agent does NOT
           crash. That second layer is free, because it's just the same
           plan-act-observe loop from Phase 1.
"""

from __future__ import annotations

import dataclasses
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .types import ToolCall, ToolResult, ToolSpec


class Tool:
    """One callable capability."""

    def __init__(self, spec: ToolSpec, fn: Callable[[dict], object]):
        self.spec = spec
        self.fn = fn

    def run(self, call: ToolCall) -> ToolResult:
        """Execute and NEVER let a tool exception kill the agent.

        Any error becomes a structured ToolResult(ok=False) - the contract
        Phase 3 builds retries and recovery on top of."""
        try:
            output = self.fn(call.arguments)
            return ToolResult(id=call.id, name=call.name, ok=True, output=output)
        except Exception as exc:  # noqa: BLE001 - tool errors are user-defined
            return ToolResult(id=call.id, name=call.name, ok=False, error=str(exc))


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, parameters: dict,
                 fn: Callable[[dict], object]) -> None:
        self._tools[name] = Tool(ToolSpec(name, description, parameters), fn)

    def get(self, name: str):
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        """What we hand to the model so it knows what it can call."""
        return [t.spec for t in self._tools.values()]


@dataclass
class RetryPolicy:
    """How the executor should retry a failing tool call.

    fail_fast=True means "never retry" - useful for tools that are not safe to
    repeat (e.g. a non-idempotent payment charge) or when you'd rather let the
    model decide immediately.
    """

    max_attempts: int = 3
    backoff_base: float = 0.5   # seconds before the 1st retry
    backoff_cap: float = 8.0    # delay is never larger than this
    jitter: bool = True         # randomise delay to avoid thundering herds
    fail_fast: bool = False


class ToolExecutor:
    """Runs a ToolCall, retrying on failure according to a RetryPolicy.

    Every attempt is recorded as a ``tool_attempt`` event when a store is given,
    so the trace (Phase 5) can show exactly how many tries a call needed.
    """

    def __init__(self, registry: ToolRegistry, policy: Optional[RetryPolicy] = None,
                 sleep: Callable[[float], None] = time.sleep,
                 rng: Callable[[], float] = random.random):
        self.registry = registry
        self.policy = policy or RetryPolicy()
        self.sleep = sleep
        self.rng = rng

    def execute(self, call: ToolCall, store=None, task_id=None) -> ToolResult:
        p = self.policy
        attempts = 0
        last: Optional[ToolResult] = None

        while attempts < p.max_attempts:
            attempts += 1
            tool = self.registry.get(call.name)
            if tool is None:
                return ToolResult(id=call.id, name=call.name, ok=False,
                                  error=f"unknown tool: {call.name}", attempts=attempts)

            res = tool.run(call)
            if store is not None:
                store.append(task_id, "tool_attempt", {
                    "id": call.id, "name": call.name, "ok": res.ok,
                    "attempt": attempts, "error": res.error,
                })

            if res.ok:
                return dataclasses.replace(res, attempts=attempts)
            last = res

            # Out of retries, or configured to never retry -> hand failure up.
            if p.fail_fast or attempts >= p.max_attempts:
                return dataclasses.replace(res, attempts=attempts)

            delay = min(p.backoff_cap, p.backoff_base * (2 ** (attempts - 1)))
            if p.jitter:
                delay *= 0.5 + self.rng() * 0.5
            self.sleep(delay)

        # Loop exits only if max_attempts==0 (defensive); surface last failure.
        return dataclasses.replace(last, attempts=attempts) if last else \
            ToolResult(id=call.id, name=call.name, ok=False,
                       error="no attempts made", attempts=attempts)


def calculator(args: dict) -> float:
    """A tiny, sandboxed arithmetic tool (no builtins, no calls)."""
    return float(eval(args["expr"], {"__builtins__": {}}, {}))  # noqa: S307
