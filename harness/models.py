"""Phase 1 - the Model abstraction.

Why an abstraction here? The engine must not care *how* it gets the next
assistant message - only *that* it gets one. That lets us:
  * develop offline with a DummyModel (no API key, no network),
  * later drop in an OpenAICompatibleModel, an OllamaModel, a MockForTests...
without touching the engine. This is the Dependency Inversion Principle in
one file.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from .types import Message, Role, ToolCall, ToolSpec


class Model(ABC):
    """Anything that can take a conversation + tool specs and return the next
    assistant message."""

    @abstractmethod
    def chat(self, messages: list[Message], tools: Optional[list[ToolSpec]] = None,
             temperature: float = 0.7) -> Message:
        ...

    def chat_stream(self, messages: list[Message],
                    tools: Optional[list[ToolSpec]] = None,
                    temperature: float = 0.7):
        """Yield the next assistant message as one or more deltas.

        Default: a single delta holding the whole message, so a model that
        does not stream behaves correctly when the engine asks for a stream.
        A real API-backed model overrides this to yield true SSE/streaming
        chunks. The engine assembles deltas into the final Message and treats
        the LAST delta's ``tool_calls``/``usage`` as authoritative."""
        yield self.chat(messages, tools=tools, temperature=temperature)


class DummyModel(Model):
    """A deterministic, offline stand-in for a real LLM.

    It always does the same thing so we can *see the loop* clearly:
      step 1 -> ask to call the first available tool
      step 2 -> once a tool result is present, return a final answer
    """

    def chat(self, messages: list[Message], tools: Optional[list[ToolSpec]] = None,
             temperature: float = 0.7) -> Message:
        last = messages[-1]

        # The model has just seen a tool result -> it can answer now.
        if last.role == Role.TOOL:
            resp = Message(Role.ASSISTANT, f"计算结果是 {last.content}。任务完成。")
            prompt_text = "\n".join(m.content for m in messages)
            resp.usage = {
                "prompt_tokens": max(1, len(prompt_text) // 4),
                "completion_tokens": max(1, len(resp.content) // 4),
            }
            return resp

        # Otherwise, the model decides to call a tool. We call the first
        # registered one with a fixed argument just to demonstrate the flow.
        first_tool = tools[0].name if tools else "calculator"
        resp = Message(
            Role.ASSISTANT,
            "",  # no free-text content when issuing a tool call
            tool_calls=[ToolCall(id="c1", name=first_tool, arguments={"expr": "6*7"})],
        )
        # Phase 5: a deterministic, plausible token estimate so the trace
        # demo shows numbers. A real Model fills this from the API response.
        prompt_text = "\n".join(m.content for m in messages)
        resp.usage = {
            "prompt_tokens": max(1, len(prompt_text) // 4),
            "completion_tokens": max(1, len(resp.content or "") // 4),
        }
        return resp

    def chat_stream(self, messages, tools=None, temperature=0.7):
        """Offline streaming: yield the final answer in a few character chunks
        so the engine's live-printing is visible without a real API. A tool-call
        turn has empty content, so it yields once (carrying the tool_calls)."""
        resp = self.chat(messages, tools=tools, temperature=temperature)
        text = resp.content or ""
        if not text:
            yield resp
            return
        # Split into ~4 visible chunks (works for CJK too, since we slice chars).
        size = max(1, len(text) // 4)
        last = None
        for i in range(0, len(text), size):
            piece = text[i:i + size]
            delta = Message(Role.ASSISTANT, piece)
            last = delta
            yield delta
        # The LAST delta carries tool_calls + usage - the engine reads these.
        if last is not None:
            last.tool_calls = resp.tool_calls
            last.usage = resp.usage


# --------------------------------------------------------------------------- #
# OpenAI-compatible client (stdlib urllib, ZERO third-party dependencies).
#
# This is the piece the roadmap always promised but was never written. It makes
# the "swap the model, the engine doesn't change" story REAL: the Engine depends
# only on the ``Model`` interface, so pointing it at a real LLM (OpenAI, vLLM,
# Ollama, a local server, or any national-cloud model that speaks the same
# schema) is a one-line construction change - no engine edits. We use urllib so
# the "zero third-party dependency" claim holds all the way to production.
# --------------------------------------------------------------------------- #
def _to_oai(m: Message) -> dict:
    """Our Message -> OpenAI chat schema. ``Role`` mirrors the schema 1:1, so
    this is mostly field reshaping. A TOOL message must carry its ``tool_call_id``;
    an ASSISTANT message that only issues tool calls sends ``content=None``."""
    content = m.content
    if m.role == Role.ASSISTANT and m.tool_calls and not content:
        content = None
    d: dict = {"role": m.role.value, "content": content}
    if m.role == Role.TOOL:
        d["tool_call_id"] = m.tool_call_id
    if m.tool_calls:
        d["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name,
                          "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
            for tc in m.tool_calls
        ]
    return d


def _from_oai(msg: dict) -> Message:
    """OpenAI chat message -> our Message. ``arguments`` arrives as a JSON
    string; we parse it (falling back to {} on malformed input) so the engine
    gets a real dict to pass to the tool."""
    tcs = []
    for tc in msg.get("tool_calls", []) or []:
        fn = tc.get("function", {})
        raw = fn.get("arguments", "{}")
        try:
            arguments = json.loads(raw) if isinstance(raw, str) else raw
        except (ValueError, TypeError):
            arguments = {}
        tcs.append(ToolCall(id=tc["id"], name=fn.get("name", ""),
                            arguments=arguments or {}))
    return Message(role=Role(msg.get("role", "assistant")),
                   content=msg.get("content") or "", tool_calls=tcs)


class OpenAICompatibleModel(Model):
    """Talk to any OpenAI-compatible ``/chat/completions`` endpoint over urllib.

    Stdlib only. The Engine already depends solely on the ``Model`` interface,
    so wiring a real backend is just ``Engine(OpenAICompatibleModel(...), ...)`` -
    no other change. ``chat`` parses usage into ``Message.usage`` (Phase 5 cost).
    """

    def __init__(self, api_key: str, base_url: str, model: str,
                 temperature: float = 0.7, timeout: float = 60.0):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages: list[Message], tools=None,
             temperature: float = 0.7) -> Message:
        import json
        import urllib.error
        import urllib.request
        payload = {
            "model": self.model,
            "messages": [_to_oai(m) for m in messages],
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = [{
                "type": "function",
                "function": {"name": t.name, "description": t.description,
                             "parameters": t.parameters},
            } for t in tools]
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=data,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:  # surface a readable error, not a traceback
            detail = e.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"LLM HTTP {e.code}: {detail}") from e
        msg = body["choices"][0]["message"]
        out = _from_oai(msg)
        out.usage = body.get("usage") or None
        return out

    def chat_stream(self, messages, tools=None, temperature=0.7):
        """urllib has no first-class SSE reader; yield one full delta so the
        engine's streaming path still works. A real deployment can swap in a
        streaming client - the Engine only needs the LAST delta's tool_calls."""
        yield self.chat(messages, tools=tools, temperature=temperature)
