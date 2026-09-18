"""Phase 5 - observability: structured traces, metrics, and OTel export.

The events table (Phase 2) already records WHAT the agent did. Phase 5 adds
HOW WELL it did: latency per model/tool call, token usage, and cost. That
turns the log from "a story" into "a dashboard" - and a dashboard is what you
need to answer "why was this agent slow / expensive / flaky?".

Two consumers of this data:
  * the developer, who wants a readable waterfall + a summary (export_trace),
  * a real observability backend, which wants OpenTelemetry spans (OTelExporter).

Design choices worth internalising
-----------------------------------
  * A SPAN is one measured unit of work (a model call, a tool call, a step).
    Spans stack, so the trace is a TREE, not a flat list.
  * We store spans as durable 'span' events - exactly like every other event.
    So a trace survives a crash and is queryable after the fact. Observability
    MUST outlive the process; a trace that vanishes when the agent dies is
    useless for debugging crashes (which are when you need it most).
  * The OTel path is OPTIONAL and lazily imported. The core stays zero-dep;
    if you want spans in Jaeger/Tempo/OTLP you `pip install` the SDK and flip
    a flag. This is the standard "pluggable exporter" pattern: the harness
    emits its own neutral span format, and exporters translate it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


# (prompt $/1K tokens, completion $/1K tokens) - a tiny, editable price table.
# A real deployment would fetch live prices or read them from config.
_COST_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-3.5-turbo": (0.0005, 0.0015),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """USD cost for a call, or None if the model price is unknown."""
    rate = _COST_PER_1K.get(model)
    if rate is None:
        return None
    return prompt_tokens / 1000 * rate[0] + completion_tokens / 1000 * rate[1]


@dataclass
class Span:
    """One measured unit of work in the trace."""

    name: str
    kind: str            # "model" | "tool" | "guardrail" | "step"
    start: float         # perf_counter seconds
    end: float
    duration_ms: float
    attributes: dict = field(default_factory=dict)
    status: str = "ok"   # "ok" | "error" | "blocked"

    def to_dict(self) -> dict:
        return {
            "name": self.name, "kind": self.kind,
            "start": self.start, "end": self.end,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes, "status": self.status,
        }


class Tracer:
    """Collects spans for one Engine instance."""

    def __init__(self) -> None:
        self.spans: list[Span] = []

    def record(self, span: Span) -> None:
        self.spans.append(span)

    def span(self, name: str, kind: str, start: float, end: float,
             attributes: Optional[dict] = None, status: str = "ok") -> Span:
        s = Span(name, kind, start, end, (end - start) * 1000,
                 attributes or {}, status)
        self.spans.append(s)
        return s

    # ----------------------------------------------------------------- summary
    def summary(self) -> dict:
        model = [s for s in self.spans if s.kind == "model"]
        tool = [s for s in self.spans if s.kind == "tool"]
        retries = sum(max(0, int(s.attributes.get("attempts", 1)) - 1)
                      for s in tool)
        return {
            "model_calls": len(model),
            "tool_calls": len(tool),
            "retries": retries,
            "total_model_ms": round(sum(s.duration_ms for s in model), 2),
            "total_tool_ms": round(sum(s.duration_ms for s in tool), 2),
            "prompt_tokens": sum(int(s.attributes.get("prompt_tokens", 0)) for s in model),
            "completion_tokens": sum(int(s.attributes.get("completion_tokens", 0)) for s in model),
            "total_cost_usd": round(
                sum(float(s.attributes.get("cost_usd", 0) or 0) for s in model), 6),
        }

    # ---------------------------------------------------------------- waterfall
    def waterfall(self, indent: str = "  ") -> str:
        """A readable top-down view: model calls at the root, everything else
        nested beneath the call they belong to."""
        lines = []
        for s in self.spans:
            if s.kind == "model":
                prefix = ""
            else:
                prefix = indent
            attr = s.attributes
            bits = [f"{s.duration_ms:8.2f}ms"]
            if s.kind == "model":
                bits.append(f"prompt={attr.get('prompt_tokens', 0)} "
                            f"comp={attr.get('completion_tokens', 0)}")
                if "cost_usd" in attr:
                    bits.append(f"${attr['cost_usd']:.6f}")
            elif s.kind == "tool":
                bits.append(f"ok={attr.get('ok')} attempts={attr.get('attempts')}")
            if s.status != "ok":
                bits.append(f"[{s.status}]")
            lines.append(f"{prefix}{s.name:18} " + "  ".join(bits))
        return "\n".join(lines)


def spans_from_events(events: list[dict]) -> list[Span]:
    """Rebuild spans from durable 'span' events (used when tracing a resumed
    task straight from the store, so the trace survives a crash)."""
    out = []
    for ev in events:
        if ev["type"] != "span":
            continue
        p = ev["payload"]
        out.append(Span(
            name=p.get("name", ev["type"]), kind=p.get("kind", "step"),
            start=p.get("start", 0.0), end=p.get("end", 0.0),
            duration_ms=float(p.get("duration_ms", 0.0)),
            attributes={k: v for k, v in p.items()
                        if k not in ("name", "kind", "start", "end", "duration_ms")},
            status=p.get("status", "ok"),
        ))
    return out


class OTelExporter:
    """Optional, lazily-imported OpenTelemetry exporter.

    The core harness is zero-dependency. If you want real spans in a backend,
    ``pip install opentelemetry-sdk opentelemetry-exporter-otlp`` and this just
    works. If the SDK is missing, construction raises a clear, actionable
    error instead of failing silently."""

    def __init__(self, service_name: str = "agent-harness") -> None:
        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.sdk.resources import SERVICE_NAME, Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor, ConsoleSpanExporter)
            from opentelemetry.trace import StatusCode
        except ImportError as exc:  # pragma: no cover - exercised only w/o SDK
            raise RuntimeError(
                "OTel export needs the SDK. Install with:\n"
                "  pip install opentelemetry-sdk opentelemetry-exporter-otlp"
            ) from exc

        provider = TracerProvider(resource=Resource({SERVICE_NAME: service_name}))
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        otel_trace.set_tracer_provider(provider)
        self._otel_trace = otel_trace
        self._tracer = otel_trace.get_tracer(service_name)

    @staticmethod
    def _attr(v: Any) -> Any:
        if isinstance(v, (str, bool, int, float)):
            return v
        return str(v)

    def export(self, spans: list[Span]) -> None:
        """Translate our neutral spans into OTel spans and emit them."""
        with self._tracer.start_as_current_span("agent.task") as root:
            for s in spans:
                with self._tracer.start_as_current_span(s.name) as child:
                    for k, v in s.attributes.items():
                        child.set_attribute(str(k), self._attr(v))
                    if s.status == "error":
                        child.set_status(StatusCode.ERROR)
                    elif s.status == "blocked":
                        child.set_status(StatusCode.UNSET)
