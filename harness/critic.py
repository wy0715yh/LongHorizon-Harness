"""Phase 6 - reflection & self-correction.

A Critic looks at what the agent just did (the latest assistant message plus
the running conversation) and decides whether the agent should KEEP GOING or
STOP AND REFLECT.

Reflection is NOT a new control flow. It is the same move we have used since
Phase 3: append one more message to the conversation and call the model
again. The only novelty is *who* decides to append it - not a tool failure,
but a Critic that judges the agent's OWN behavior. That is the difference
between "the tool crashed" (Phase 3) and "you are repeating yourself, stop"
(Phase 6).

Two flavors ship here:
  * RuleCritic - deterministic rules (loop detection, error memory). Offline,
                 no second LLM call. Perfect for demos and tests.
  * LLMCritic  - a second Model acts as "LLM-as-a-judge" over the agent's last
                 step. Dropped in with zero engine changes (Dependency Inversion
                 again: the engine only depends on the Critic interface).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .types import Message, Role
from .state import StateStore
from .models import Model


@dataclass
class Critique:
    """The verdict of a review.

    ``ok=False`` means "reflect first": the engine will inject ``reason`` as a
    message and re-call the model instead of running the tool calls.
    """

    ok: bool
    reason: str
    kind: str = "generic"  # loop | error_memory | stall | llm


class Critic(ABC):
    """Anything that can judge the agent's last step."""

    @abstractmethod
    def review(self, messages: list[Message], last_resp: Message,
               store: Optional[StateStore] = None,
               task_id: Optional[str] = None) -> Critique:
        ...


def _recent_assistant_sigs(messages: list[Message]) -> list[str]:
    """Signatures of the most recent assistant tool-call turns, oldest -> newest.

    A signature collapses one assistant turn's tool calls into a stable string
    (tool name + sorted JSON args), so we can tell "called the same thing twice"
    from "called two different things"."""
    out: list[str] = []
    for m in messages:
        if m.role == Role.ASSISTANT and m.tool_calls:
            sig = "|".join(
                f"{tc.name}:{json.dumps(tc.arguments, sort_keys=True, ensure_ascii=False)}"
                for tc in m.tool_calls
            )
            out.append(sig)
    return out


class RuleCritic(Critic):
    """Deterministic, offline reflection. Two independent checks:

    1) LOOP DETECTION - the agent issues the identical tool call N times in a
       row with no progress. Classic agent failure mode ("I'll just retry the
       same broken call forever").
    2) ERROR MEMORY - a tool has failed often *in the past* (durably, across
       tasks). Re-calling it blindly is usually wrong; pick another tool.
    """

    def __init__(self, loop_window: int = 2, error_threshold: int = 2,
                 enable_stall: bool = False):
        self.loop_window = loop_window
        self.error_threshold = error_threshold
        self.enable_stall = enable_stall

    def review(self, messages, last_resp, store=None, task_id=None) -> Critique:
        # 1) LOOP DETECTION -------------------------------------------------
        sigs = _recent_assistant_sigs(messages)
        if (len(sigs) >= self.loop_window
                and len(set(sigs[-self.loop_window:])) == 1 and sigs[-1]):
            name = sigs[-1].split(":", 1)[0]
            return Critique(
                ok=False, kind="loop",
                reason=(f"You have called the same tool '{name}' "
                        f"{self.loop_window} times with identical arguments and made "
                        f"no progress. Stop repeating; either choose a different tool "
                        f"or give the final answer."),
            )

        # 2) ERROR MEMORY ---------------------------------------------------
        if store is not None and last_resp.tool_calls:
            for tc in last_resp.tool_calls:
                n = store.error_count(tc.name)
                if n >= self.error_threshold:
                    return Critique(
                        ok=False, kind="error_memory",
                        reason=(f"Tool '{tc.name}' has failed {n} times in memory. "
                                f"Do not call it again; switch tools or conclude."),
                    )

        # 3) STALL DETECTION (optional) ------------------------------------
        # A final answer with no successful tool use when the task clearly
        # needed one. Off by default: trivial tasks would false-positive.
        if self.enable_stall and not last_resp.tool_calls:
            used = any(m.role == Role.TOOL and "FAIL" not in m.content
                       for m in messages)
            if not used:
                return Critique(
                    ok=False, kind="stall",
                    reason="You produced a final answer without using any tool result. "
                           "If the task needs a value, call a tool first.",
                )

        return Critique(ok=True, kind="ok", reason="")


class LLMCritic(Critic):
    """LLM-as-a-judge: a SECOND model reviews the agent's last step.

    The critic model sees the recent conversation and must reply with exactly
    one line: 'OK' (behavior is fine) or 'CRITIQUE: <what is wrong>'. This is
    the self-reflection pattern used by many production agents. It needs a real
    (or scripted) model, but the engine integration is identical to RuleCritic.
    """

    _SYSTEM = (
        "You are a strict reviewer of an agent's last action. Reply with exactly "
        "one line: 'OK' if the last step was reasonable, or 'CRITIQUE: <reason>' "
        "if the agent is looping, misusing a tool, or about to give a wrong answer."
    )

    def __init__(self, model: Model):
        self.model = model

    def review(self, messages, last_resp, store=None, task_id=None) -> Critique:
        window = [Message(Role.SYSTEM, self._SYSTEM)] + messages[-6:]
        verdict = self.model.chat(window).content.strip()
        if verdict.upper().startswith("CRITIQUE:"):
            return Critique(ok=False, kind="llm",
                            reason=verdict[len("CRITIQUE:"):].strip())
        return Critique(ok=True, kind="ok", reason="")
