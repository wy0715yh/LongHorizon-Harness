"""Phase 5 demo - observability: structured traces and metrics.

Part A: run a task that does a RETRYING tool call (flaky) + a normal one
        (calculator). Print the trace summary + waterfall, and export JSON.
Part B: prove the trace is DURABLE - crash mid-task, reopen the same file,
        resume, then export the trace straight from the store. The spans from
        BEFORE the crash are still there, because spans are events too.

Run:  python examples/trace_demo.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.engine import Engine, SimulatedCrash
from harness.models import Model, Role, ToolCall
from harness.state import StateStore
from harness.tools import ToolRegistry
from harness.types import Message


def flaky(args: dict) -> float:
    """Fails twice, then succeeds - so the trace shows a 3-attempt call."""
    st = getattr(flaky, "_n", 0)
    flaky._n = st + 1
    if flaky._n <= 2:
        raise RuntimeError("transient dependency error")
    return args["x"] + args["y"]


def calculator(args: dict) -> float:
    return float(eval(args["expr"], {"__builtins__": {}}, {}))  # noqa: S307


class ScriptedModel(Model):
    """Calls flaky, then calculator, then gives a final answer."""

    @staticmethod
    def _msg(content: str, tool_calls=None) -> Message:
        # A real Model fills `usage` from the API; here it's a deterministic
        # estimate so the trace shows tokens + cost.
        m = Message(Role.ASSISTANT, content, tool_calls=tool_calls or [])
        m.usage = {"prompt_tokens": 12, "completion_tokens": 6}
        return m

    def chat(self, messages, tools=None, temperature=0.7) -> Message:
        last_tool = None
        for m in reversed(messages):
            if m.role == Role.TOOL:
                last_tool = m.content
                break
        if last_tool is None:
            return self._msg("", [ToolCall(id="c1", name="flaky",
                                            arguments={"x": 6, "y": 7})])
        if "13" in last_tool:  # flaky returned 13 -> now use the calculator
            return self._msg("", [ToolCall(id="c2", name="calculator",
                                            arguments={"expr": "6*7"})])
        return self._msg("Done: computed the values via tools.")


def main() -> None:
    reg = ToolRegistry()
    reg.register("flaky", "a sometimes-failing tool",
                 {"type": "object",
                  "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                  "required": ["x", "y"]}, flaky)
    reg.register("calculator", "evaluate arithmetic",
                 {"type": "object",
                  "properties": {"expr": {"type": "string"}},
                  "required": ["expr"]}, calculator)

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    store = StateStore(tmp.name)
    engine = Engine(ScriptedModel(), reg, max_steps=10, model_name="gpt-4o-mini")

    print("=== PART A: run with a retrying tool + a normal tool ===")
    answer = engine.run("compute some values", store=store)
    print(f"FINAL ANSWER: {answer}")

    trace = engine.export_trace(task_id=engine.last_task_id, fmt="text")
    print("\n--- trace waterfall ---")
    print(trace["rendered"])
    print("\n--- trace summary ---")
    for k, v in trace["summary"].items():
        print(f"  {k}: {v}")

    # Export the machine-readable trace to a file (what a backend would ingest).
    import json
    out_path = os.path.join(tempfile.gettempdir(), "harness_trace.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)
    print(f"\n[trace JSON written to {out_path}]")

    # ---- PART B: the trace must survive a crash ----------------------------
    print("\n=== PART B: crash mid-task, then resume, trace from store ===")
    # Prime flaky so the crash-healed retry succeeds (3rd attempt), giving the
    # resumed trace a real tool span to show.
    flaky._n = 2
    store2 = StateStore(tmp.name)  # fresh connection, same file
    eng2 = Engine(ScriptedModel(), reg, max_steps=10, model_name="gpt-4o-mini")
    try:
        eng2.run("compute some values (will crash)", store=store2, crash_after=1)
    except SimulatedCrash as e:
        print(f"  [crashed] {e}  (state is on disk)")

    # Reopen the file in a "new process" and resume.
    store3 = StateStore(tmp.name)
    eng3 = Engine(ScriptedModel(), reg, max_steps=10, model_name="gpt-4o-mini")
    eng3.run(task_id=eng2.last_task_id, store=store3)
    print(f"  [resumed] finished task {eng2.last_task_id}")

    # Export the trace from the DURABLE store (process could be long gone).
    durable = eng3.export_trace(task_id=eng2.last_task_id, fmt="text")
    print("\n--- durable trace (rebuilt from 'span' events on disk) ---")
    print(durable["rendered"])
    print(f"  spans persisted across crash: "
          f"{durable['summary']['model_calls']} model calls recorded")

    # ---- Optional: real OTel export (needs the SDK; skipped if absent) ----
    print("\n=== OTel export (optional) ===")
    try:
        from harness.tracing import OTelExporter
        OTelExporter().export(eng3.tracer.spans)
        print("  OTel spans emitted (SDK present).")
    except RuntimeError as e:
        print(f"  skipped: {e}")

    store.close()
    store2.close()
    store3.close()
    os.unlink(tmp.name)


if __name__ == "__main__":
    main()
