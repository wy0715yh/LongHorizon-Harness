"""Phase 4 - behavior norms and guardrails.

An agent that works in a demo is NOT the same as an agent you'd trust in
production. Production agents need fences. This module provides four:

  1. SYSTEM POLICY   - behavioral rules baked into the system prompt, so the
     model is told up front what it may and may not do.
  2. TOOL ALLOWLIST  - the model may ONLY call tools we explicitly permit. A
     request for anything else is rejected and fed back as a structured
     failure, so the model re-plans instead of doing something dangerous.
  3. I/O VALIDATION  - tool arguments are checked against the tool's declared
     schema BEFORE execution. Bad input never reaches a real (possibly
     destructive) tool. Outputs are validated too.
  4. SECRET REDACTION - anything written to the durable trace is scrubbed for
     API keys / passwords first, so a leaked secret never lands on disk.

THE KEY DESIGN IDEA (same as Phase 3)
-------------------------------------
A guardrail violation is NON-FATAL. It returns a structured failure, exactly
like a flaky tool. The loop already knows how to handle failures - so a
guardrail is just "one more check before we run a tool", not a new control
flow. We are reusing the plan-act-observe loop, not forking it.

Zero third-party dependencies. The schema check is a deliberately small
subset of JSON Schema, sufficient for the tool specs we declare.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .types import ToolSpec


# Patterns we scrub before anything touches durable storage. Tunable per use.
_DEFAULT_SECRET_PATTERNS: list[str] = [
    r"sk-[A-Za-z0-9]{8,}",            # OpenAI-style keys
    r"AKIA[0-9A-Z]{16}",             # AWS access key ids
    r"Bearer\s+[A-Za-z0-9._\-]+",    # bearer tokens
    r"(?i)password[\"']?\s*[:=]\s*\S+",  # password = xxx
    r"(?i)api[_-]?key[\"']?\s*[:=]\s*\S+",  # api_key = xxx
]

_DEFAULT_RULES: list[str] = [
    "Only call tools that are explicitly provided to you.",
    "If a guardrail blocks a tool call, acknowledge it and pick an allowed "
    "alternative instead of retrying the blocked call.",
    "Never put secret material (API keys, passwords, tokens) into tool "
    "arguments or outputs.",
    "Prefer the simplest tool that accomplishes the goal.",
]


@dataclass
class Guardrails:
    """The configuration of an agent's behavioral fence.

    Pass an instance to ``Engine(guardrails=...)``. ``None`` on the engine
    means "no guardrails" - everything that worked in P1-P3 keeps working.
    """

    # The model may ONLY call tools whose name is in this set.
    # None (default) -> no restriction. Empty set -> block every tool.
    allowed_tools: Optional[set[str]] = None

    # Extra behavioral rules injected into the system prompt.
    rules: list[str] = field(default_factory=lambda: list(_DEFAULT_RULES))

    # Validate tool arguments against their declared JSON schema before run.
    validate_args: bool = True

    # Patterns of secrets to redact before writing to the trace.
    secret_patterns: list[str] = field(
        default_factory=lambda: list(_DEFAULT_SECRET_PATTERNS)
    )

    def __post_init__(self) -> None:
        self._compiled = [re.compile(p) for p in self.secret_patterns]

    # --------------------------------------------------------------- policy text
    def build_system_prompt(self, base: str) -> str:
        """Render the behavioral rules into the system prompt.

        The base system text (P1/P2) stays, with a clearly delimited policy
        block appended. Delimiting it makes it easy for a human to audit what
        constraints the agent is running under."""
        if not self.rules:
            return base
        rules_block = "\n".join(f"{i+1}. {r}" for i, r in enumerate(self.rules))
        return (
            f"{base}\n\n"
            "# Behavior policy (you MUST follow these)\n"
            f"{rules_block}\n"
        )

    # --------------------------------------------------------------- allowlist
    def is_tool_allowed(self, name: str) -> bool:
        """A tool passes only if no allowlist is set, or it is on the list."""
        if self.allowed_tools is None:
            return True
        return name in self.allowed_tools

    # -------------------------------------------------------------- redaction
    def redact(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        out = text
        for pat in self._compiled:
            out = pat.sub("***REDACTED***", out)
        return out

    def redact_obj(self, o: Any) -> Any:
        """Recursively scrub any string inside a payload before it is stored.

        The StateStore calls this on EVERY event (message, tool_result,
        guardrail, ...) so secrets in arguments, outputs or model text are
        scrubbed uniformly - no code path can forget to redact."""
        if isinstance(o, str):
            return self.redact(o)
        if isinstance(o, dict):
            return {k: self.redact_obj(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [self.redact_obj(v) for v in o]
        return o


# ------------------------------------------------------------------- validation
_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def validate_args(spec: ToolSpec, args: Any) -> tuple[bool, str]:
    """Minimal JSON-Schema validation (stdlib only) for tool arguments.

    Supports: required properties, per-property type checks, and
    additionalProperties:false. This is enough for the tool specs we declare
    and for the teaching goal; a production system would use ``jsonschema``.
    """
    schema = spec.parameters or {}
    if schema.get("type") and schema["type"] != "object":
        # We only validate object-shaped arguments; anything else is out of scope.
        return True, ""

    props = schema.get("properties", {}) or {}
    required = schema.get("required", []) or []
    additional = schema.get("additionalProperties", True)

    if not isinstance(args, dict):
        return False, "arguments must be a JSON object"

    for name in required:
        if name not in args:
            return False, f"missing required argument: '{name}'"

    for name, val in args.items():
        pdef = props.get(name)
        if pdef is None:
            if additional is False:
                return False, f"unexpected argument: '{name}'"
            continue
        expected = pdef.get("type")
        if expected in _TYPE_CHECKS and not _TYPE_CHECKS[expected](val):
            got = "bool" if isinstance(val, bool) else type(val).__name__
            return False, f"argument '{name}' must be {expected}, got {got}"

    return True, ""
