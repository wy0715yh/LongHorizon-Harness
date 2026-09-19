"""Phase 1 - the Model abstraction.

Why an abstraction here? The engine must not care *how* it gets the next
assistant message - only *that* it gets one. That lets us:
  * develop offline with a DummyModel (no API key, no network),
  * later drop in an OpenAICompatibleModel, an OllamaModel, a MockForTests...
without touching the engine. This is the Dependency Inversion Principle in
one file.
"""

from __future__ import annotations

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
