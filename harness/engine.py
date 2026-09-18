"""Phase 2 - the engine, now *durable*.

Same plan-act-observe loop as Phase 1, but every step is optionally appended
to a StateStore. Two things become possible that were impossible before:

  1. RESUME: call run(task_id=..., store=...) again and the engine rebuilds the
     conversation from the event log, then keeps going. The process can die
     between steps and the task still finishes.

  2. RECOVERY: if a crash left an assistant message that *asked* for a tool but
     we never recorded the tool's result, resume() detects the dangling call
     and re-runs just that tool - self-healing instead of deadlocking.

How the loop changed vs Phase 1:
  * step 1 builds the message list. With a store it comes from the log (resume)
    or from [system, user] (new task); without a store it is just [user].
  * after every model call AND every tool result, we append an event (if store).
  * crash_after=N raises SimulatedCrash right after step N's model call, so a
    demo can show "died -> reopened the file -> resumed -> completed".
"""

from __future__ import annotations

import time
from typing import Optional

from .models import Model
from .state import StateStore, message_to_dict
from .tools import ToolRegistry, ToolExecutor, RetryPolicy
from .types import Message, Role, ToolResult
from .guardrails import Guardrails, validate_args
from .tracing import Tracer, estimate_cost, spans_from_events


class SimulatedCrash(Exception):
    """Raised by ``crash_after`` to mimic a process dying mid-task.

    This is NOT a real crash (os._exit). It exists so a demo can prove the
    state on disk survived: you catch it, open a *fresh* StateStore on the same
    file, and resume - which is exactly what a real restart would do.
    """

    def __init__(self, task_id: str, step: int):
        self.task_id = task_id
        self.step = step
        super().__init__(f"simulated crash at step {step} (task {task_id})")


_DEFAULT_SYSTEM = (
    "You are a reliable agent. When you need a value, call a tool; "
    "otherwise answer the user directly and concisely."
)


class Engine:
    def __init__(self, model: Model, registry: ToolRegistry,
                 max_steps: int = 10, system: Optional[str] = None,
                 executor: Optional[ToolExecutor] = None,
                 guardrails: Optional[Guardrails] = None,
                 model_name: str = "dummy",
                 tracer: Optional[Tracer] = None):
        self.model = model
        self.registry = registry
        self.max_steps = max_steps
        # An explicit system prompt wins; otherwise the guardrails (if any)
        # render the base prompt + the behavior policy block.
        if system is not None:
            self.system = system
        elif guardrails is not None:
            self.system = guardrails.build_system_prompt(_DEFAULT_SYSTEM)
        else:
            self.system = _DEFAULT_SYSTEM
        # One executor shared by the loop. Defaults to a 3-attempt policy.
        self.executor = executor or ToolExecutor(registry)
        self.guardrails = guardrails
        self.model_name = model_name
        # Phase 5: every measured unit of work lands here (and, when a store is
        # present, also as a durable 'span' event).
        self.tracer = tracer or Tracer()
        self.last_task_id: Optional[str] = None
        self._last_store: Optional[StateStore] = None

    # --------------------------------------------------------------------- public
    def run(self, prompt: Optional[str] = None, *,
            task_id: Optional[str] = None,
            store: Optional[StateStore] = None,
            crash_after: Optional[int] = None) -> str:
        """Run (or resume) a task.

        Modes:
          * no store           -> in-memory only, like Phase 1 (prompt required)
          * store, new task    -> prompt required; a task_id is created
          * store, task_id set -> resume that task from its event log
        """
        messages, task_id = self._build_messages(prompt, task_id, store)
        self.last_task_id = task_id
        self._last_store = store

        # Phase 4: if guardrails are on, arm the durable store's redactor so
        # EVERY event written from here on (messages, tool results, guardrail
        # blocks) is scrubbed for secrets before it touches disk.
        if store is not None and self.guardrails is not None:
            store.redactor = self.guardrails.redact_obj

        # Heal any tool call the previous run promised but never finished.
        if store is not None:
            messages = self._heal_dangling(messages, store, task_id)

        assistant_count = sum(1 for m in messages if m.role == Role.ASSISTANT)
        resumed = bool(store and task_id and assistant_count > 0)

        for step in range(assistant_count + 1, assistant_count + 1 + self.max_steps):
            tag = f"step {step}" + (" (resumed)" if resumed else "")
            print(f"\n--- {tag}: calling model ---")
            # ---- PHASE 5: measure the model call ----
            m_start = time.perf_counter()
            resp = self.model.chat(messages, tools=self.registry.specs())
            m_end = time.perf_counter()
            usage = getattr(resp, "usage", None) or {}
            p_tok = int(usage.get("prompt_tokens", 0))
            c_tok = int(usage.get("completion_tokens", 0))
            cost = estimate_cost(self.model_name, p_tok, c_tok) \
                if (p_tok or c_tok) else None
            m_attrs = {"duration_ms": round((m_end - m_start) * 1000, 2),
                       "prompt_tokens": p_tok, "completion_tokens": c_tok,
                       "model": self.model_name}
            if cost is not None:
                m_attrs["cost_usd"] = round(cost, 6)
            self.tracer.span("model.chat", "model", m_start, m_end, m_attrs)
            if store:
                store.append(task_id, "span", {
                    "kind": "model", "name": "model.chat",
                    **m_attrs, "status": "ok",
                })
            messages.append(resp)
            if store:
                store.append(task_id, "message", message_to_dict(resp))

            if crash_after == step:
                if store:
                    store.set_status(task_id, "crashed")
                raise SimulatedCrash(task_id, step)

            if not resp.tool_calls:
                if store:
                    store.set_status(task_id, "done")
                print("model produced a final answer.")
                return resp.content

            for tc in resp.tool_calls:
                print(f"model calls tool: {tc.name}({tc.arguments})")

                # ---- PHASE 4 GUARDRAILS: one more check before we run ----
                # A block is NON-FATAL: we record it and feed the model a
                # structured failure (like a flaky tool), so it can re-plan.
                blocked, reason = self._check_guardrail(tc)
                if blocked:
                    print(f"  [guardrail] BLOCKED: {reason}")
                    g_start = time.perf_counter()
                    if store:
                        store.append(task_id, "guardrail", {
                            "reason": reason, "tool": tc.name,
                            "arguments": tc.arguments,
                        })
                    self.tracer.span("guardrail", "guardrail", g_start,
                                     time.perf_counter(),
                                     {"tool": tc.name, "reason": reason},
                                     status="blocked")
                    messages.append(Message(
                        Role.TOOL,
                        content=f"GUARDRAIL BLOCKED: {reason} (tool "
                                f"'{tc.name}' was NOT executed)",
                        tool_call_id=tc.id,
                    ))
                    if store:
                        store.append(task_id, "message",
                                     message_to_dict(messages[-1]))
                    continue

                # LAYER 1: the executor retries transient failures for us.
                t_start = time.perf_counter()
                result = self.executor.execute(tc, store=store, task_id=task_id)
                t_end = time.perf_counter()
                # ---- PHASE 5: measure the tool call (incl. any retries) ----
                t_attrs = {"duration_ms": round((t_end - t_start) * 1000, 2),
                           "tool": tc.name, "ok": result.ok,
                           "attempts": result.attempts}
                self.tracer.span(f"tool.{tc.name}", "tool", t_start, t_end,
                                 t_attrs, status="ok" if result.ok else "error")
                if store:
                    store.append(task_id, "span", {
                        "kind": "tool", "name": tc.name,
                        **t_attrs, "status": "ok" if result.ok else "error",
                    })
                # On exhaustion the failure text (with attempt count) is fed
                # back as a TOOL message - that's LAYER 2, handled by the loop.
                content = (str(result.output) if result.ok
                           else f"FAIL: {result.error} (after {result.attempts} attempts)")
                messages.append(Message(Role.TOOL, content=content,
                                        tool_call_id=tc.id))
                if store:
                    store.append(task_id, "tool_result", {
                        "id": tc.id, "name": tc.name, "ok": result.ok,
                        "output": str(result.output) if result.ok else None,
                        "error": result.error, "attempts": result.attempts,
                    })
                    store.append(task_id, "message", message_to_dict(messages[-1]))

        if store:
            store.set_status(task_id, "max_steps")
        return "(reached max_steps without a final answer)"

    # ----------------------------------------------------------------- observability
    def export_trace(self, task_id: Optional[str] = None,
                     fmt: str = "text") -> dict:
        """Return the trace for a task: spans + summary, optionally rendered.

        If ``task_id`` is given and this run used a store, the trace is rebuilt
        from the DURABLE 'span' events - so you can export a trace for a task
        that crashed and was resumed, long after the process is gone."""
        if task_id and self._last_store is not None:
            spans = spans_from_events(self._last_store.event_log(task_id))
        else:
            spans = self.tracer.spans

        # Rebuild a Tracer so we can reuse its summary()/waterfall() helpers.
        t = Tracer()
        for s in spans:
            t.record(s)
        out = {"task_id": task_id, "summary": t.summary(),
               "spans": [s.to_dict() for s in spans]}
        if fmt == "text":
            sm = t.summary()
            head = (f"TRACE task={task_id}  model_calls={sm['model_calls']} "
                    f"tool_calls={sm['tool_calls']} retries={sm['retries']}  "
                    f"cost=${sm['total_cost_usd']:.6f}")
            out["rendered"] = head + "\n" + t.waterfall()
        return out

    # -------------------------------------------------------------------- internals
    def _check_guardrail(self, tc) -> tuple[bool, Optional[str]]:
        """Return (blocked?, reason) for a tool call.

        Two checks:
          * allowlist  - is the requested tool permitted at all?
          * validation - do its arguments match the declared schema?
        Both are guardrails: a failure here means "don't run this", not
        "the tool crashed". The difference matters for the model: a guardrail
        block tells it to choose a different action; a crash tells it the
        action it chose failed."""
        g = self.guardrails
        if g is None:
            return False, None
        if not g.is_tool_allowed(tc.name):
            return True, f"tool '{tc.name}' is not on the allowed list"
        if g.validate_args:
            tool = self.registry.get(tc.name)
            if tool is not None:
                ok, err = validate_args(tool.spec, tc.arguments)
                if not ok:
                    return True, f"invalid arguments: {err}"
        return False, None

    def _build_messages(self, prompt, task_id, store):
        if store is None:
            if prompt is None:
                raise ValueError("prompt is required when no store is given")
            return [Message(Role.USER, prompt)], None

        if task_id and store.exists(task_id):
            # RESUME: rebuild the whole conversation from the durable log.
            print(f"[resume] rebuilding task {task_id} from event log")
            return store.load_messages(task_id), task_id

        if prompt is None:
            raise ValueError("prompt is required to start a new task")
        task_id = store.new_task()
        messages = [Message(Role.SYSTEM, self.system)]
        store.append(task_id, "message", message_to_dict(messages[-1]))
        messages.append(Message(Role.USER, prompt))
        store.append(task_id, "message", message_to_dict(messages[-1]))
        return messages, task_id

    def _run_once(self, tc) -> ToolResult:
        """Run a tool exactly once (no retries). Used by crash-healing, where
        we are re-running a specific call and don't want retry storms."""
        tool = self.registry.get(tc.name)
        if tool is None:
            return ToolResult(id=tc.id, name=tc.name, ok=False,
                              error=f"unknown tool: {tc.name}")
        return tool.run(tc)

    def _heal_dangling(self, messages, store, task_id) -> list[Message]:
        """If a previous run died after asking for a tool but before recording
        its result, re-run exactly that tool so the conversation is legal again.

        Why this matters: an ASSISTANT message with tool_calls but no matching
        TOOL message is invalid input to any real LLM API. Left alone it would
        crash on resume. Re-running the missing call turns a corrupt mid-step
        into a clean, complete one."""
        answered = {m.tool_call_id for m in messages
                    if m.role == Role.TOOL and m.tool_call_id}
        pending = []
        for m in reversed(messages):
            if m.role == Role.ASSISTANT and m.tool_calls:
                pending = [tc for tc in m.tool_calls if tc.id not in answered]
                break  # only the most recent assistant turn can be unfinished

        if not pending:
            return messages

        print(f"[recovery] {len(pending)} dangling tool call(s) from a crash - re-running")
        for tc in pending:
            store.append(task_id, "recovery", {
                "reason": "dangling tool_call", "call": message_to_dict(
                    Message(Role.ASSISTANT, "", tool_calls=[tc]))["tool_calls"][0],
            })
            h_start = time.perf_counter()
            result = self._run_once(tc)
            h_end = time.perf_counter()
            # Phase 5: a healed call is still a measured unit of work - record it
            # so the trace shows the crash's aftermath, not a blank.
            h_attrs = {"duration_ms": round((h_end - h_start) * 1000, 2),
                       "tool": tc.name, "ok": result.ok,
                       "attempts": 1, "healed": True}
            self.tracer.span(f"tool.{tc.name}", "tool", h_start, h_end, h_attrs,
                             status="ok" if result.ok else "error")
            if store:
                store.append(task_id, "span", {
                    "kind": "tool", "name": tc.name, **h_attrs,
                    "status": "ok" if result.ok else "error",
                })
            messages.append(Message(
                Role.TOOL,
                content=str(result.output) if result.ok else (result.error or "error"),
                tool_call_id=tc.id,
            ))
            store.append(task_id, "tool_result", {
                "id": tc.id, "name": tc.name, "ok": result.ok,
                "output": str(result.output) if result.ok else None,
                "error": result.error,
            })
            store.append(task_id, "message", message_to_dict(messages[-1]))
        return messages
