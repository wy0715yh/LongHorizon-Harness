"""Phase 2 - crash, reopen the file, resume, finish.

Run it:
    python examples/resume_demo.py

It proves the durable state survives a "process death":
  1. run a task, but crash_after=1 simulates the process dying after step 1.
  2. a FRESH StateStore (a new "process") opens the same .db file and resumes.
  3. the task completes; we then dump the event log so you can see the agent's
     whole memory on disk.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.engine import Engine, SimulatedCrash
from harness.models import DummyModel
from harness.state import StateStore
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

    # A real on-disk file. NamedTemporaryFile just keeps it out of the repo.
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()

    store = StateStore(db_path)
    task_id = None
    try:
        print("# run 1: start the task, then crash after step 1")
        engine.run("what is 6*7?", store=store, crash_after=1)
    except SimulatedCrash as exc:
        task_id = exc.task_id
        print(f"\n!!! {exc} - but every event so far is safe on disk")
    store.close()  # a real restart would release the first connection

    # ---- a NEW process: brand-new StateStore on the SAME file ----
    store2 = StateStore(db_path)
    print(f"\n# run 2: a fresh process reopens the file")
    print(f"  status on disk: {store2.status(task_id)}")
    answer = engine.run(task_id=task_id, store=store2)
    print("\nFINAL ANSWER:", answer)

    print("\n# the agent's durable memory (the event log):")
    for ev in store2.event_log(task_id):
        print(f"  [{ev['seq']:>2}] {ev['type']:<11} {ev['payload']}")
    store2.close()

    os.unlink(db_path)  # clean up the demo file


if __name__ == "__main__":
    main()
