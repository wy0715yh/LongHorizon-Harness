"""Phase 4 demo - behavior norms and guardrails, end to end.

We use a ScriptedModel (not the DummyModel) so we can SEE the agent hit each
fence and react:

  beat 1  model tries a FORBIDDEN tool (shell rm -rf)      -> allowlist blocks it
  beat 2  model retries with a tool whose ARG has wrong type -> schema blocks it
  beat 3  model finally behaves, but embeds a SECRET in the arg -> stored REDACTED
  beat 4  model calls an allowed tool (calculator)         -> succeeds, done

After the run we read the durable event log and PROVE three things:
  * the forbidden shell command was never executed,
  * the schema violation was caught before execution,
  * the secret string never reached disk (it was scrubbed).

Run:  python examples/guardrails_demo.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.engine import Engine
from harness.guardrails import Guardrails
from harness.models import Model, Role
from harness.state import StateStore, message_to_dict
from harness.tools import ToolRegistry
from harness.types import Message, ToolCall, ToolResult


# --- a tool whose ONLY job is to echo back its argument (so we can see if the
#     secret survived into storage) -------------------------------------------
def save_note(args: dict) -> str:
    return f"recorded: {args['text']}"


def shell(args: dict) -> str:
    # A genuinely dangerous tool. Under guardrails it must never run.
    return f"ran: {args['cmd']}"


def calculator(args: dict) -> float:
    return float(eval(args["expr"], {"__builtins__": {}}, {}))  # noqa: S307


class ScriptedModel(Model):
    """A model that reacts to guardrail feedback, beat by beat.

    It is not smart - it just plays a fixed script that exercises every fence
    so the demo is deterministic and readable. A real model would re-plan on
    its own, using exactly the structured failure we feed it."""

    def chat(self, messages, tools=None, temperature=0.7) -> Message:
        last_tool = None
        for m in reversed(messages):
            if m.role == Role.TOOL:
                last_tool = m.content
                break

        if last_tool is None:
            # beat 1: reach for the forbidden shell tool
            return Message(Role.ASSISTANT, "",
                           tool_calls=[ToolCall(id="c1", name="shell",
                                                arguments={"cmd": "rm -rf /"})])

        if last_tool.startswith("GUARDRAIL BLOCKED"):
            if "shell" in last_tool:
                # beat 2: blocked -> try a permitted tool but with a BAD arg
                return Message(Role.ASSISTANT, "",
                               tool_calls=[ToolCall(id="c2", name="save_note",
                                                    arguments={"text": 42})])  # int, not str
            # beat 3: schema blocked -> now a VALID call that carries a SECRET
            return Message(
                Role.ASSISTANT, "",
                tool_calls=[ToolCall(
                    id="c3", name="save_note",
                    arguments={"text": "deploy using sk-ABC123DEF456GHI789JKL token"})])

        if "recorded:" in last_tool:
            # beat 4: a clean, allowed tool call
            return Message(Role.ASSISTANT, "",
                           tool_calls=[ToolCall(id="c4", name="calculator",
                                                arguments={"expr": "6*7"})])

        return Message(Role.ASSISTANT,
                       "Done. I respected the guardrails and finished the task.")


def main() -> None:
    reg = ToolRegistry()
    reg.register("shell", "run a shell command",
                 {"type": "object",
                  "properties": {"cmd": {"type": "string"}},
                  "required": ["cmd"]}, shell)
    reg.register("save_note", "persist a text note",
                 {"type": "object",
                  "properties": {"text": {"type": "string"}},
                  "required": ["text"]}, save_note)
    reg.register("calculator", "evaluate arithmetic",
                 {"type": "object",
                  "properties": {"expr": {"type": "string"}},
                  "required": ["expr"]}, calculator)

    guardrails = Guardrails(allowed_tools={"calculator", "save_note"})

    # Show the system prompt the model actually receives (policy injection).
    print("=" * 70)
    print("SYSTEM PROMPT THE MODEL RECEIVES:")
    print("=" * 70)
    print(guardrails.build_system_prompt(
        "You are a reliable agent. Call tools when you need values."))
    print("=" * 70)

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    store = StateStore(tmp.name)
    engine = Engine(ScriptedModel(), reg, max_steps=10, guardrails=guardrails)

    print("\n--- running the guarded agent ---")
    answer = engine.run("do the risky thing, then record a note", store=store)
    task_id = engine.last_task_id
    print(f"\nFINAL ANSWER: {answer}")

    # ---- now PROVE the fences worked by inspecting what hit disk ----------
    log = store.event_log(task_id)
    raw = json.dumps(log, ensure_ascii=False)
    secret = "sk-ABC123DEF456GHI789JKL"

    checks = []
    # 1) forbidden shell never executed: no successful tool_result for shell
    shell_executed = any(
        ev["type"] == "tool_result" and ev["payload"].get("name") == "shell"
        for ev in log
    )
    checks.append(("forbidden 'shell' never executed", not shell_executed))

    # 2) schema violation was caught
    validation_blocked = any(
        ev["type"] == "guardrail" and "invalid arguments" in ev["payload"]["reason"]
        for ev in log
    )
    checks.append(("bad-arg call blocked by schema validation", validation_blocked))

    # 3) allowlist block recorded
    allow_blocked = any(
        ev["type"] == "guardrail" and "allowed list" in ev["payload"]["reason"]
        for ev in log
    )
    checks.append(("forbidden tool blocked by allowlist", allow_blocked))

    # 4) the secret string is NOT anywhere on disk
    checks.append(("secret never persisted (redacted)", secret not in raw))

    # 5) the redacted marker IS in the stored save_note output
    redacted_present = "***REDACTED***" in raw
    checks.append(("redaction marker present in trace", redacted_present))

    print("\n--- event log (what hit disk) ---")
    for ev in log:
        if ev["type"] in ("guardrail", "tool_result"):
            short = json.dumps(ev["payload"], ensure_ascii=False)
            if len(short) > 90:
                short = short[:90] + "..."
            print(f"  [{ev['seq']}] {ev['type']:10} {short}")

    print("\n--- guardrail checks ---")
    all_ok = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok
    print("\nRESULT:", "ALL PASS" if all_ok else "SOME FAILED")

    store.close()
    os.unlink(tmp.name)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
