"""Phase 7 - circuit breaking, rate limiting, backpressure (offline demo).

    python examples/circuit_demo.py

Scenario A  : a tool keeps failing. The per-tool circuit breaker trips OPEN
              after N consecutive failures, then FAST-FAILS further calls and
              feeds the model a structured "CIRCUIT OPEN" result - so the model
              switches to a working tool instead of hammering the dead one. We
              also prove it feeds Phase 6's durable error memory.
Scenario A2 : unit-test the breaker STATE MACHINE (CLOSED->OPEN->HALF_OPEN->
              CLOSED) with a virtual clock, isolated from the engine.
Scenario B  : the token-bucket rate limiter. A virtual clock shows acquire()
              blocking until tokens refill - the blocking IS backpressure. Then
              we wire the same limiter into an Engine and prove the model-call
              span records the wait.
Scenario C  : the backpressure queue. The model emits 5 parallel tool calls in
              one turn; with maxsize=2 only 2 are admitted, the other 3 are
              SHED with a soft failure, and the model retries them next turn
              until everything converges.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.types import Message, Role, ToolCall
from harness.tools import ToolRegistry, calculator, ToolExecutor, RetryPolicy
from harness.models import Model
from harness.engine import Engine
from harness.state import StateStore
from harness.resilience import (CircuitBreaker, CircuitBreakerSet, RateLimiter,
                                BackpressureQueue, CircuitState)


# A virtual clock + sleep so the resilience timers run deterministically and
# instantly (no real waiting in the demo).
class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt


# ------------------------------------------------------------- scenario models
class CircuitModel(Model):
    """Stubborn: keeps calling the broken 'boom' tool until it sees a
    'CIRCUIT OPEN' result, then switches to the working 'calculator'."""

    def chat(self, messages, tools=None, temperature=0.7):
        last = messages[-1]
        if last.role == Role.TOOL and "42" in last.content:
            return Message(Role.ASSISTANT, "根据 calculator 结果，答案是 42。")
        if any("CIRCUIT OPEN" in m.content for m in messages):
            return Message(Role.ASSISTANT, "",
                           tool_calls=[ToolCall(id="c_calc", name="calculator",
                                                arguments={"expr": "6*7"})])
        return Message(Role.ASSISTANT, "",
                       tool_calls=[ToolCall(id="c_boom", name="boom",
                                            arguments={"x": 1})])


def _boom(args):
    raise RuntimeError("boom: simulated permanent failure")


class ThrottleModel(Model):
    """Calls 'calculator' once, then answers - a few model turns so the rate
    limiter has something to throttle."""

    def chat(self, messages, tools=None, temperature=0.7):
        last = messages[-1]
        if last.role == Role.TOOL:
            return Message(Role.ASSISTANT, f"tool said: {last.content}")
        return Message(Role.ASSISTANT, "",
                       tool_calls=[ToolCall(id="c1", name="calculator",
                                            arguments={"expr": "2+3"})])


class BackpressureModel(Model):
    """Emits 5 parallel 'echo' calls, then retries ONLY the ones that came back
    SHED (BACKPRESSURE) - and not ones that already succeeded - until all 5
    finish. The 'done' set prevents re-emitting completed work, which would
    otherwise waste both admission slots and keep the final 'c5' perpetually
    shed (maxsize=2 means only 2 of the 3 pending calls fit each turn)."""

    def chat(self, messages, tools=None, temperature=0.7):
        done = {m.tool_call_id for m in messages
                if m.role == Role.TOOL and not m.content.startswith("BACKPRESSURE")}
        pending = [m.tool_call_id for m in messages
                   if m.role == Role.TOOL and m.content.startswith("BACKPRESSURE")]
        # Only retry calls that are still pending AND not yet done - this is
        # exactly what a real agent would do after reading the soft failures.
        pending = [p for p in pending if p not in done]
        if pending:
            return Message(Role.ASSISTANT, "", tool_calls=[
                ToolCall(id=i, name="echo", arguments={"text": i}) for i in pending])
        if len(done) >= 5:
            return Message(Role.ASSISTANT, f"all 5 done: {sorted(done)}")
        return Message(Role.ASSISTANT, "", tool_calls=[
            ToolCall(id=f"c{i}", name="echo", arguments={"text": f"c{i}"})
            for i in range(1, 6)])


def _echo(args):
    return args.get("text", "")


# ------------------------------------------------------------------- scenario A
def _scenario_a() -> None:
    print("=== Scenario A: circuit breaker trips, fast-fails, model re-plans ===")
    reg = ToolRegistry()
    reg.register("calculator", "safe arithmetic",
                 {"type": "object", "properties": {"expr": {"type": "string"}},
                  "required": ["expr"]}, calculator)
    reg.register("boom", "always fails",
                 {"type": "object", "properties": {"x": {"type": "number"}}}, _boom)

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    store = StateStore(tmp.name)

    # Retry 2x each turn (P3) so the breaker counts a failure only AFTER the
    # retries are exhausted. Breaker trips after 2 consecutive failures.
    executor = ToolExecutor(reg, policy=RetryPolicy(max_attempts=2))
    circuit = CircuitBreakerSet(failure_threshold=2, cooldown_s=1.0)
    engine = Engine(CircuitModel(), reg, max_steps=12, executor=executor,
                    circuit=circuit)
    answer = engine.run("compute 6*7, but the calculator is flaky", store=store)
    print("\nFINAL ANSWER:", answer)

    log = store.event_log(engine.last_task_id)
    circuit_spans = [e for e in log if e["type"] == "span"
                     and e["payload"].get("kind") == "circuit"]
    boom_fails = [e for e in log if e["type"] == "tool_result"
                  and e["payload"].get("name") == "boom" and not e["payload"]["ok"]]

    br = circuit.get("boom")
    print(f"\n[check] breaker state: {br.state.value}  (opens={br.total_opens}, "
          f"rejections={br.total_rejections}, total_failures={br.total_failures})")
    print(f"[check] boom executed & failed {len(boom_fails)} turn(s) before trip")
    print(f"[check] circuit-open fast-fails recorded: {len(circuit_spans)}")
    print(f"[check] Phase 6 error memory for 'boom': {store.error_count('boom')}")

    assert br.state == CircuitState.OPEN
    assert br.total_opens >= 1
    assert br.total_rejections >= 1          # the fast-fail turn
    assert len(circuit_spans) >= 1
    assert store.error_count("boom") >= 1    # breaker fed durable error memory
    assert "42" in answer
    print("[check] breaker tripped, fast-failed, model switched tools: PASS")
    store.close()
    os.unlink(tmp.name)


# ---------------------------------------------------------------- scenario A2
def _scenario_a2() -> None:
    print("\n=== Scenario A2: breaker state machine (virtual clock) ===")
    clock = FakeClock()
    cb = CircuitBreaker("x", failure_threshold=2, cooldown_s=1.0, clock=clock.now)
    assert cb.state == CircuitState.CLOSED and cb.allow()
    print(f"[check] start: {cb.state.value}")

    cb.on_failure()
    cb.on_failure()                          # 2 consecutive -> trip OPEN
    assert cb.state == CircuitState.OPEN
    assert cb.allow() is False
    print(f"[check] after 2 failures: {cb.state.value} (allow={cb.allow()})")

    clock.t += 1.0                           # cooldown elapsed -> HALF_OPEN
    assert cb.allow() is True
    assert cb.state == CircuitState.HALF_OPEN
    print(f"[check] after cooldown: {cb.state.value} (probe allowed)")

    cb.on_success()                          # probe succeeds -> CLOSED
    assert cb.state == CircuitState.CLOSED
    print(f"[check] after probe success: {cb.state.value} (recovered)")

    # Re-open then prove HALF_OPEN failure re-opens (and restarts cooldown).
    cb.on_failure(); cb.on_failure()
    assert cb.state == CircuitState.OPEN
    clock.t += 1.0
    assert cb.allow() is True                # HALF_OPEN
    cb.on_failure()                          # probe fails -> OPEN again
    assert cb.state == CircuitState.OPEN and cb.total_opens == 3
    print(f"[check] half-open probe failure re-opens: {cb.state.value} "
          f"(total_opens={cb.total_opens})")
    print("[check] state machine CLOSED->OPEN->HALF_OPEN->CLOSED: PASS")


# ------------------------------------------------------------------- scenario B
def _scenario_b() -> None:
    print("\n=== Scenario B: token-bucket rate limiter (virtual clock) ===")
    clock = FakeClock()
    lim = RateLimiter(capacity=2, refill_per_s=2.0,
                      clock=clock.now, sleep=clock.sleep)
    for i in range(5):
        ok = lim.acquire()
        assert ok
    print(f"[check] acquired {lim.total_acquired} tokens; waited "
          f"{lim.total_waited_s:.3f}s of virtual time (capacity was 2)")
    assert lim.total_acquired == 5
    assert lim.total_waited_s > 0            # 3rd+ had to wait for refill
    print("[check] token bucket throttled & blocked until refill: PASS")

    # Now wire the SAME limiter into an Engine (virtual clock -> instant) and
    # prove the model-call span records the wait.
    print("\n=== Scenario B (engine): rate limiter wired into the loop ===")
    reg = ToolRegistry()
    reg.register("calculator", "safe arithmetic",
                 {"type": "object", "properties": {"expr": {"type": "string"}},
                  "required": ["expr"]}, calculator)
    engine = Engine(ThrottleModel(), reg, max_steps=6,
                    rate_limiter=RateLimiter(capacity=1, refill_per_s=1.0,
                                             clock=clock.now, sleep=clock.sleep))
    engine.run("add 2+3")
    rl_spans = [s for s in engine.tracer.spans if s.kind == "ratelimit"]
    waited = [s.attributes.get("waited_ms", 0) for s in rl_spans]
    print(f"[check] ratelimit spans: {len(rl_spans)}, waited_ms per call: {waited}")
    assert len(rl_spans) >= 2
    assert any(w > 0 for w in waited)        # at least one model call was throttled
    print("[check] engine throttles model calls & records the wait: PASS")


# ------------------------------------------------------------------- scenario C
def _scenario_c() -> None:
    print("\n=== Scenario C: backpressure sheds overflow parallel calls ===")
    reg = ToolRegistry()
    reg.register("echo", "echo text back",
                 {"type": "object", "properties": {"text": {"type": "string"}},
                  "required": ["text"]}, _echo)

    bp = BackpressureQueue(maxsize=2)
    engine = Engine(BackpressureModel(), reg, max_steps=12, backpressure=bp)
    answer = engine.run("do 5 things in parallel")
    print("\nFINAL ANSWER:", answer)

    # The engine records a 'backpressure' span per shed call (in-memory tracer);
    # no store needed. The queue's own counters also accumulate across turns.
    shed = [s for s in engine.tracer.spans if s.kind == "backpressure"]
    print(f"[check] backpressure shed spans: {len(shed)}")
    print(f"[check] queue stats: {bp.stats()}")
    assert bp.total_shed >= 3                # 5 calls vs maxsize 2 -> overflow
    assert "all 5 done" in answer
    assert len(shed) >= 1
    print("[check] overflow shed with soft failure, model retried, converged: PASS")


def main() -> None:
    _scenario_a()
    _scenario_a2()
    _scenario_b()
    _scenario_c()
    print("\nALL P7 CHECKS PASSED")


if __name__ == "__main__":
    main()
