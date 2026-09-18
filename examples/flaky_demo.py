"""Phase 3 - tool failure recovery, demonstrated in two layers.

Run it:
    python examples/flaky_demo.py

LAYER 1 (executor retries a flaky tool):
    `flaky` fails its first 2 runs, succeeds on the 3rd. The executor retries
    with backoff, the model never sees the failure - recovery is transparent.

LAYER 2 (model re-plans after exhaustion):
    `boom` ALWAYS fails. The executor gives up after max_attempts and feeds the
    failure (with attempt count) back to the model as a TOOL message. The model
    re-plans to a graceful answer instead of crashing.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.engine import Engine
from harness.models import Model
from harness.state import StateStore
from harness.tools import ToolExecutor, ToolRegistry
from harness.types import Message, Role, ToolCall


def make_flaky(fail_times: int):
    """A tool that fails the first `fail_times` calls, then works."""
    state = {"n": 0}

    def fn(args):
        state["n"] += 1
        if state["n"] <= fail_times:
            raise RuntimeError(f"transient error (attempt {state['n']})")
        return 42

    return fn


def boom(args):
    raise RuntimeError("permanent failure - dependency is down")


class DemoModel(Model):
    """Context-driven: calls the first tool, then answers. On a FAIL message it
    re-plans to a graceful conclusion instead of crashing (LAYER 2)."""

    def chat(self, messages, tools=None, temperature=0.7):
        last = messages[-1]
        if last.role == Role.TOOL:
            if last.content.startswith("FAIL"):
                return Message(Role.ASSISTANT,
                               "工具持续失败，我改为给出兜底结论：无法获取该值，请稍后重试。")
            return Message(Role.ASSISTANT, f"已拿到工具结果 {last.content}，任务完成。")
        first = tools[0].name if tools else "calculator"
        return Message(Role.ASSISTANT, "", tool_calls=[ToolCall(id="c1", name=first,
                                                                arguments={})])


def _run_part(title, registry, sleep):
    print(f"\n================ {title} ================")
    executor = ToolExecutor(registry, sleep=sleep)
    engine = Engine(DemoModel(), registry, executor=executor)

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()
    store = StateStore(db_path)

    answer = engine.run("do the task", store=store)
    print("\nFINAL ANSWER:", answer)

    print("\n-- event log (note tool_attempt retries) --")
    for ev in store.event_log(engine.last_task_id):
        print(f"  [{ev['seq']:>2}] {ev['type']:<11} {ev['payload']}")
    store.close()
    os.unlink(db_path)


def main() -> None:
    # LAYER 1
    reg_a = ToolRegistry()
    reg_a.register("flaky", "a flaky tool", {"type": "object", "properties": {}},
                   make_flaky(fail_times=2))
    _run_part("LAYER 1: flaky tool recovers via retries", reg_a,
              sleep=lambda d: print(f"    (backoff {d:.2f}s before retry)"))

    # LAYER 2
    reg_b = ToolRegistry()
    reg_b.register("boom", "a tool that always fails", {"type": "object",
                   "properties": {}}, boom)
    _run_part("LAYER 2: permanent failure is escalated to the model", reg_b,
              sleep=lambda d: None)


if __name__ == "__main__":
    main()
