"""Phase 6 - reflection & self-correction demo (offline, no API key).

    python examples/reflection_demo.py

Scenario A: the agent gets stuck calling a broken tool repeatedly. The Critic
catches the loop, forces a re-plan (injecting its note into the conversation),
and the agent switches to the working calculator and finishes.

Scenario B: unit-tests the error-memory check directly - a tool that has failed
twice in the durable store is flagged before the agent even calls it again.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.types import Message, Role, ToolCall
from harness.tools import ToolRegistry, calculator, ToolExecutor, RetryPolicy
from harness.models import Model
from harness.engine import Engine
from harness.state import StateStore
from harness.critic import RuleCritic


class ReflectionModel(Model):
    """A deliberately stubborn agent: it keeps calling the broken 'boom' tool
    until a Critic tells it to stop, then it switches to 'calculator'."""

    def chat(self, messages, tools=None, temperature=0.7):
        last = messages[-1]
        # 1) Highest priority: we already have the computed value -> answer.
        if last.role == Role.TOOL and "42" in last.content:
            return Message(Role.ASSISTANT, "根据工具计算结果，答案是 42。任务完成。")
        # 2) A Critic has spoken -> re-plan using the working calculator.
        if any("[Critic]" in m.content for m in messages):
            return Message(Role.ASSISTANT, "",
                           tool_calls=[ToolCall(id="c_calc", name="calculator",
                                                arguments={"expr": "6*7"})])
        # 3) Otherwise: stubbornly call the broken tool again.
        return Message(Role.ASSISTANT, "",
                       tool_calls=[ToolCall(id="c_boom", name="boom",
                                            arguments={"x": 1})])


def _boom(args):
    raise RuntimeError("boom: simulated permanent failure")


def _scenario_a() -> None:
    print("=== Scenario A: loop detection forces self-correction ===")
    reg = ToolRegistry()
    reg.register("calculator", "safe arithmetic",
                 {"type": "object", "properties": {"expr": {"type": "string"}},
                  "required": ["expr"]}, calculator)
    reg.register("boom", "always fails",
                 {"type": "object", "properties": {"x": {"type": "number"}}}, _boom)

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    store = StateStore(tmp.name)
    # fail_fast + 1 attempt so the broken tool fails immediately (demo stays fast)
    executor = ToolExecutor(reg, policy=RetryPolicy(max_attempts=1, fail_fast=True))
    engine = Engine(ReflectionModel(), reg, max_steps=10,
                    executor=executor, critic=RuleCritic(loop_window=2,
                                                          error_threshold=2))
    answer = engine.run("compute 6*7 using whatever tool works", store=store)
    print("\nFINAL ANSWER:", answer)

    log = store.event_log(engine.last_task_id)
    reflections = [e for e in log if e["type"] == "reflection"]
    boom_fails = [e for e in log if e["type"] == "tool_result"
                  and e["payload"].get("name") == "boom" and not e["payload"]["ok"]]

    assert reflections, "expected at least one reflection event"
    assert any(r["payload"]["kind"] == "loop" for r in reflections)
    assert "42" in answer
    print(f"\n[check] reflection events: {len(reflections)} "
          f"(kinds={[r['payload']['kind'] for r in reflections]})")
    print(f"[check] boom actually executed & failed: {len(boom_fails)} time(s) "
          f"before the critic intervened")
    print("[check] agent re-planned and finished with the correct value: PASS")
    store.close()
    os.unlink(tmp.name)


def _scenario_b() -> None:
    print("\n=== Scenario B: error memory warns against a repeatedly-failing tool ===")
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    store = StateStore(tmp.name)
    # Durable, cross-task memory: 'flaky' has failed twice in the past.
    store.record_error("flaky", "timeout")
    store.record_error("flaky", "timeout")
    critic = RuleCritic(error_threshold=2)
    last = Message(Role.ASSISTANT, "",
                   tool_calls=[ToolCall(id="c1", name="flaky", arguments={"x": 1})])
    c = critic.review([Message(Role.USER, "retry it")], last, store)
    print(f"[check] critic.kind={c.kind} ok={c.ok}")
    print(f"[check] reason: {c.reason}")
    assert (not c.ok) and c.kind == "error_memory"
    print("[check] error-memory critique fired: PASS")
    store.close()
    os.unlink(tmp.name)


def main() -> None:
    _scenario_a()
    _scenario_b()
    print("\nALL P6 CHECKS PASSED")


if __name__ == "__main__":
    main()
