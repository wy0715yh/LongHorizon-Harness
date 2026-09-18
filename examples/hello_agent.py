"""Phase 1 - run the minimal agent end to end (offline, no API key).

    python examples/hello_agent.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.engine import Engine
from harness.models import DummyModel
from harness.tools import ToolRegistry, calculator


def main() -> None:
    registry = ToolRegistry()
    registry.register(
        "calculator",
        "Evaluate a basic arithmetic expression, e.g. '6*7'.",
        {"type": "object", "properties": {"expr": {"type": "string"}},
         "required": ["expr"]},
        calculator,
    )

    engine = Engine(DummyModel(), registry)
    answer = engine.run("what is 6*7?")
    print("\nFINAL ANSWER:", answer)


if __name__ == "__main__":
    main()
