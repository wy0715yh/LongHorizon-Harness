"""Phase 1 - the shared vocabulary of an Agent.

Before any logic, we need a common *language*. Every other module speaks in
these types, so changes stay local. Think of this file as the "wire format"
between the model, the tools and the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Role(str, Enum):
    """Who produced a message. Mirrors the OpenAI chat schema 1:1 on purpose,
    so swapping in a real model later requires zero translation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """One turn in the conversation.

    For a tool result, set ``role=TOOL`` and ``tool_call_id`` so the model
    knows which call this answers. ``tool_calls`` only appears on ASSISTANT
    messages when the model wants to invoke a tool.
    """

    role: Role
    content: str
    tool_calls: list["ToolCall"] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    # Token usage, filled by a real Model from the API response (Phase 5).
    # DummyModel fills a deterministic estimate so the trace demo shows numbers.
    usage: Optional[dict] = None


@dataclass
class ToolCall:
    """A concrete invocation the model is asking for."""

    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    """The outcome of running a ToolCall.

    ``ok`` is the single most important field: it lets the engine decide
    whether to give the model a success to build on, or a failure to recover
    from. ``attempts`` (added in Phase 3) records how many tries it took - it
    feeds both the model's re-planning and the trace."""

    id: str
    name: str
    ok: bool
    output: Any = None
    error: Optional[str] = None
    attempts: int = 1


@dataclass
class ToolSpec:
    """A tool's *declaration* handed to the model (OpenAI function schema).

    The model only sees ``name`` + ``description`` + ``parameters``; it never
    sees the Python function. This separation is what makes tools swappable
    and safe to expose.
    """

    name: str
    description: str
    parameters: dict
