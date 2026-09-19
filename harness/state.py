"""Phase 2 - durable state via SQLite *event sourcing*.

THE BIG IDEA
------------
If the process dies at any point, we want to reopen the same task and continue
*exactly* where we left off. The classic (wrong) way is to keep a snapshot of
the conversation and overwrite it every step. The problem: a crash between
"do the step" and "save the snapshot" loses everything.

The robust way is *event sourcing*: every meaningful thing the agent does is
appended to an append-only log (the ``events`` table). The conversation is
never stored as a blob we overwrite - it is *reconstructed* by replaying the
log. So a crash simply means "we stopped appending"; reopening replays up to
the last durable event. No snapshot can go stale, because there is no snapshot.

Tables
------
  tasks  : one row per run (status, timestamps, free-form meta)
  events : append-only log; payload is JSON. This IS the agent's memory.
  errors : durable failure counts per tool (Phase 6 error memory). Cross-task.

Why SQLite?
  * ships with Python stdlib (zero third-party deps),
  * one file = one task store, trivial to back up / move / inspect,
  * WAL mode gives durable writes + safe concurrent readers,
  * on an edge box you don't want a Postgres just to remember a todo.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from typing import Any, Optional

from .types import Message, Role, ToolCall


def _now() -> float:
    return time.time()


def _jsonable(o: Any) -> Any:
    """Best-effort JSON coercion; fall back to str for weird tool outputs."""
    try:
        json.dumps(o)
        return o
    except (TypeError, ValueError):
        return str(o)


def message_to_dict(m: Message) -> dict:
    """Serialize a Message to a plain dict (the on-disk 'wire format')."""
    return {
        "role": m.role.value,
        "content": m.content,
        "tool_calls": [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in m.tool_calls
        ],
        "tool_call_id": m.tool_call_id,
    }


def dict_to_message(d: dict) -> Message:
    """Inverse of message_to_dict - used when replaying the event log."""
    return Message(
        role=Role(d["role"]),
        content=d.get("content", ""),
        tool_calls=[
            ToolCall(id=tc["id"], name=tc["name"], arguments=tc.get("arguments", {}))
            for tc in d.get("tool_calls", [])
        ],
        tool_call_id=d.get("tool_call_id"),
    )


class StateStore:
    """The single source of truth for a task's durable history."""

    def __init__(self, path: str = "harness.db", redactor=None):
        self.path = path
        self.redactor = redactor  # callable(obj)->obj, set by Phase 4 guardrails
        # Phase 8: check_same_thread=False + a lock let the SAME store be used
        # from the engine's main thread AND the worker threads that run tool
        # calls in parallel. SQLite is not safe for concurrent writes from
        # different threads, so every write path below takes this lock.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.RLock()
        # WAL: writes are durable, and a reader (e.g. the trace exporter) will
        # not block the writer. On a single-file edge store this matters.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_schema()

    # ------------------------------------------------------------------ schema
    def _ensure_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id  TEXT PRIMARY KEY,
                status   TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                meta     TEXT
            );
            CREATE TABLE IF NOT EXISTS events (
                seq     INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                type    TEXT NOT NULL,
                payload TEXT NOT NULL,
                ts      REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_task ON events(task_id, seq);
            CREATE TABLE IF NOT EXISTS errors (
                tool       TEXT NOT NULL,
                error_type TEXT NOT NULL,
                n          INTEGER NOT NULL,
                last_at    REAL NOT NULL,
                PRIMARY KEY(tool, error_type)
            );
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------- task lifecycle
    def new_task(self, meta: Optional[dict] = None) -> str:
        task_id = uuid.uuid4().hex[:12]
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO tasks(task_id,status,created_at,updated_at,meta) "
                "VALUES(?,?,?,?,?)",
                (task_id, "running", now, now, json.dumps(meta or {})),
            )
            self._conn.commit()
        return task_id

    def exists(self, task_id: str) -> bool:
        with self._lock:
            return (
                self._conn.execute(
                    "SELECT 1 FROM tasks WHERE task_id=?", (task_id,)
                ).fetchone()
                is not None
            )

    def status(self, task_id: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        return row[0] if row else None

    def set_status(self, task_id: str, status: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE tasks SET status=?, updated_at=? WHERE task_id=?",
                (status, _now(), task_id),
            )
            self._conn.commit()

    # ----------------------------------------------------------------- event log
    def append(self, task_id: str, etype: str, payload: dict) -> None:
        """The only write path that matters: append one event.

        If a ``redactor`` was set (Phase 4), the payload is scrubbed for
        secrets before it is serialized - so nothing sensitive is ever
        persisted. Guarded by a lock so parallel tool workers (Phase 8) can
        each append their own event safely."""
        if self.redactor is not None:
            payload = self.redactor(payload)
        with self._lock:
            self._conn.execute(
                "INSERT INTO events(task_id,type,payload,ts) VALUES(?,?,?,?)",
                (task_id, etype, json.dumps(payload, ensure_ascii=False), _now()),
            )
            self._conn.commit()

    def event_log(self, task_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq,type,payload,ts FROM events WHERE task_id=? ORDER BY seq",
                (task_id,),
            ).fetchall()
        return [
            {"seq": r[0], "type": r[1], "payload": json.loads(r[2]), "ts": r[3]}
            for r in rows
        ]

    # ----------------------------------------------- Phase 6: error memory
    def record_error(self, tool: str, error_type: str) -> None:
        """Durably count a failed tool call. Upsert on (tool, error_type).

        This is the agent's *memory of past failures*: it survives across steps,
        tasks and even process crashes, so the Critic can warn against blindly
        re-calling a tool that keeps blowing up."""
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO errors(tool,error_type,n,last_at) VALUES(?,?,1,?) "
                "ON CONFLICT(tool,error_type) DO UPDATE SET n=n+1, last_at=?",
                (tool, error_type, now, now),
            )
            self._conn.commit()

    def error_count(self, tool: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(n),0) FROM errors WHERE tool=?", (tool,)
            ).fetchone()
        return int(row[0])

    def error_stats(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT tool, error_type, n FROM errors ORDER BY n DESC"
            ).fetchall()
        return [{"tool": r[0], "error_type": r[1], "n": r[2]} for r in rows]

    # ------------------------------------------------- replay = the recovery core
    def load_messages(self, task_id: str) -> list[Message]:
        """Rebuild the conversation by replaying ``message`` events.

        This is the heart of crash recovery: the agent's memory *is* the event
        log, not a snapshot we might have lost. Whatever survived on disk is
        exactly what we resume from - no more, no less."""
        msgs: list[Message] = []
        for ev in self.event_log(task_id):
            if ev["type"] == "message":
                msgs.append(dict_to_message(ev["payload"]))
        return msgs

    def close(self) -> None:
        self._conn.close()
