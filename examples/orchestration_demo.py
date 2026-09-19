"""Phase 8 - advanced orchestration (offline demo).

    python examples/orchestration_demo.py

Scenario A : PARALLEL TOOLS. The model asks for 3 independent slow calls in one
             turn. With parallel=True they run in a thread pool and finish in
             ~one call's time; with parallel=False they run one-by-one (~3x).
Scenario B : HUMAN-IN-THE-LOOP. The model calls a `confirm` tool flagged
             human=True; the engine PAUSES, asks the human_input callback, and
             folds the answer back in so the agent proceeds.
Scenario C : SUB-TASK DELEGATION. The "boss" model delegates a computation to a
             nested engine (leaf) and incorporates its final answer - hierarchical
             planning, one TOOL message hides the whole sub-task.
Scenario D : STREAMING. The model yields its final reply token-by-token; the
             engine prints it live and still assembles a correct message.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.types import Message, Role, ToolCall
from harness.tools import ToolRegistry, calculator
from harness.models import Model, DummyModel
from harness.engine import Engine
from harness.orchestration import Delegate


# ------------------------------------------------------------- scenario models
class ParallelModel(Model):
    """Emits 3 independent 'slow_echo' calls in one turn, then answers."""
    def chat(self, messages, tools=None, temperature=0.7):
        last = messages[-1]
        if last.role == Role.TOOL:
            parts = [m.content for m in messages if m.role == Role.TOOL]
            return Message(Role.ASSISTANT, f"all done: {parts}")
        return Message(Role.ASSISTANT, "", tool_calls=[
            ToolCall(id=f"c{i}", name="slow_echo", arguments={"text": f"t{i}"})
            for i in range(1, 4)])


def _slow(args):
    time.sleep(0.1)  # a deliberately slow tool; 3 of these should parallelise
    return args.get("text", "")


class HumanModel(Model):
    """Asks the human to confirm before proceeding."""
    def chat(self, messages, tools=None, temperature=0.7):
        last = messages[-1]
        if last.role == Role.TOOL and "approved" in last.content:
            return Message(Role.ASSISTANT,
                           f"proceeding; human said: {last.content}")
        return Message(Role.ASSISTANT, "", tool_calls=[
            ToolCall(id="h1", name="confirm",
                     arguments={"question": "Delete the production DB?"})])


class BossModel(Model):
    """Delegates a computation to a sub-task, then folds the answer in."""
    def chat(self, messages, tools=None, temperature=0.7):
        last = messages[-1]
        if last.role == Role.TOOL and "42" in last.content:
            return Message(Role.ASSISTANT,
                           f"boss final: sub-task returned '{last.content[:30]}...'")
        return Message(Role.ASSISTANT, "", tool_calls=[
            ToolCall(id="d1", name="delegate",
                     arguments={"sub_prompt": "compute 6*7"})])


class StreamModel(Model):
    """Returns a plain final text answer (so streaming is visible)."""
    def chat(self, messages, tools=None, temperature=0.7):
        return Message(Role.ASSISTANT,
                       "这是一段会逐字流式输出的、较长的最终回答，用于演示 SSE 风格输出。")


# ------------------------------------------------------------------- scenario A
def _scenario_a() -> None:
    print("\n=== Scenario A: parallel tool execution ===")
    reg = ToolRegistry()
    reg.register("slow_echo", "slow echo",
                 {"type": "object", "properties": {"text": {"type": "string"}},
                  "required": ["text"]}, _slow)

    seq_engine = Engine(ParallelModel(), reg, max_steps=6, parallel=False)
    t0 = time.perf_counter()
    seq_ans = seq_engine.run("run 3 slow calls")
    seq_dt = time.perf_counter() - t0

    par_engine = Engine(ParallelModel(), reg, max_steps=6, parallel=True,
                        max_parallel=3)
    t0 = time.perf_counter()
    par_ans = par_engine.run("run 3 slow calls")
    par_dt = time.perf_counter() - t0

    print(f"[check] sequential wall={seq_dt:.3f}s  parallel wall={par_dt:.3f}s")
    print(f"[check] seq tool spans={len([s for s in seq_engine.tracer.spans if s.kind=='tool'])} "
          f"par tool spans={len([s for s in par_engine.tracer.spans if s.kind=='tool'])}")
    assert "all done" in seq_ans and "all done" in par_ans
    # 3 calls x 0.1s = 0.3s sequential; ~0.1s parallel. Strong speedup.
    assert par_dt < seq_dt * 0.75, f"parallel ({par_dt}) not faster than seq ({seq_dt})"
    print("[check] 3 slow calls ran concurrently, ~3x faster: PASS")


# ------------------------------------------------------------------- scenario B
def _scenario_b() -> None:
    print("\n=== Scenario B: human-in-the-loop ===")
    reg = ToolRegistry()
    # human=True: the engine pauses and asks a person instead of running fn.
    reg.register("confirm", "ask a human to confirm",
                 {"type": "object", "properties": {"question": {"type": "string"}}},
                 lambda a: "never called", human=True)

    scripted = ["yes, approved"]
    def human_input(question, args):
        print(f"    (human prompted: {question!r}) -> {scripted[0]!r}")
        return scripted.pop(0)

    engine = Engine(HumanModel(), reg, max_steps=6, human_input=human_input)
    ans = engine.run("please confirm first")
    print("\nFINAL ANSWER:", ans)

    assert engine.orchestration_report()["human_in_the_loop"] is True
    assert "approved" in ans
    print("[check] engine paused for human, folded answer back: PASS")


# ------------------------------------------------------------------- scenario C
def _scenario_c() -> None:
    print("\n=== Scenario C: sub-task delegation ===")
    reg = ToolRegistry()
    reg.register("calculator", "safe arithmetic",
                 {"type": "object", "properties": {"expr": {"type": "string"}},
                  "required": ["expr"]}, calculator)
    # Register the delegate tool so the model "knows" it exists; fn is never
    # called because the engine intercepts the call and runs a nested engine.
    reg.register("delegate", "hand a sub-task to a nested agent",
                 {"type": "object",
                  "properties": {"sub_prompt": {"type": "string"}},
                  "required": ["sub_prompt"]}, lambda a: "never called")

    # The worker model actually solves the sub-task (DummyModel calls the
    # calculator and answers). The boss only plans + delegates.
    delegate = Delegate(worker_model=DummyModel(), tool_name="delegate",
                        max_steps=4)
    engine = Engine(BossModel(), reg, max_steps=6, delegate=delegate)
    ans = engine.run("compute something via a sub-agent")
    print("\nFINAL ANSWER:", ans)

    del_spans = [s for s in engine.tracer.spans if s.kind == "delegate"]
    print(f"[check] delegate spans recorded: {len(del_spans)}")
    assert "42" in ans
    assert len(del_spans) >= 1
    print("[check] boss delegated to a nested engine, folded result back: PASS")


# ------------------------------------------------------------------- scenario D
def _scenario_d() -> None:
    print("\n=== Scenario D: streaming output ===")
    reg = ToolRegistry()
    reg.register("echo", "echo",
                 {"type": "object", "properties": {"text": {"type": "string"}},
                  "required": ["text"]}, lambda a: a.get("text", ""))
    engine = Engine(StreamModel(), reg, max_steps=6, stream=True)
    print("(live stream below)")
    ans = engine.run("give me a final answer")
    print("FINAL ANSWER:", ans)

    assert engine.orchestration_report()["stream"] is True
    assert "流式输出" in ans
    print("[check] model reply streamed + assembled correctly: PASS")


def main() -> None:
    _scenario_a()
    _scenario_b()
    _scenario_c()
    _scenario_d()
    print("\nALL P8 CHECKS PASSED")


if __name__ == "__main__":
    main()
