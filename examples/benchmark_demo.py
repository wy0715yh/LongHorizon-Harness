"""Phase 9 - benchmark / evaluation.

Quantifies the THREE numbers an interviewer will ask about for a runtime that
claims to be "lightweight and reliable on the edge":

  1. COLD START   - how long from process start to first answer (import + build
                    + first model call). Measured two ways: a true out-of-process
                    ``import harness`` (the real cold start), and in-process
                    build+run latency.
  2. MEMORY PEAK   - peak RSS-ish memory during a task, via tracemalloc. The
                    whole point of "lightweight" is a small, bounded footprint.
  3. RECOVERY TIME - wall-clock to RESUME a task after a simulated crash
                    (reopen the SQLite file, replay, heal dangling, finish).
                    This is the concrete payoff of the event-sourcing design.

Plus a 4th check: an offline HTTP round-trip against the service layer, so the
"deployable" claim is demonstrated, not just asserted.

Everything runs OFFLINE (DummyModel). Run:  python examples/benchmark_demo.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import tracemalloc
import urllib.request

sys.path.insert(0, "D:/LongHorizon-Harness")

from harness.engine import Engine, SimulatedCrash
from harness.models import DummyModel
from harness.state import StateStore
from harness.tools import ToolRegistry, calculator

HERE = "D:/LongHorizon-Harness"
DB = f"{HERE}/bench_tmp.db"


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register("calculator", "eval arithmetic",
                 {"type": "object", "properties": {"expr": {"type": "string"}},
                  "required": ["expr"]}, calculator)
    return reg


def scenario_a_cold_start() -> dict:
    """True cold start: a fresh python process importing the package."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import time; t=time.perf_counter(); import harness; "
         "print('%.4f' % (time.perf_counter()-t))"],
        cwd=HERE, capture_output=True, text=True)
    import_s = float(out.stdout.strip().splitlines()[-1]) if out.stdout.strip() else 0.0

    # In-process: build the engine + run one task end-to-end.
    build_t = time.perf_counter()
    eng = Engine(DummyModel(), _registry(), max_steps=8)
    build_ms = (time.perf_counter() - build_t) * 1000
    run_t = time.perf_counter()
    ans = eng.run("what is 6*7")
    e2e_ms = (time.perf_counter() - run_t) * 1000
    tr = eng.export_trace(fmt="json")
    return {"import_s": round(import_s, 4), "build_ms": round(build_ms, 2),
            "e2e_ms": round(e2e_ms, 2), "answer": ans,
            "model_calls": tr["summary"]["model_calls"],
            "tool_calls": tr["summary"]["tool_calls"],
            "cost_usd": tr["summary"]["total_cost_usd"]}


def scenario_b_memory() -> dict:
    """Peak memory during a run, via tracemalloc."""
    tracemalloc.start()
    eng = Engine(DummyModel(), _registry(), max_steps=8)
    eng.run("what is 6*7")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"peak_mb": round(peak / (1024 * 1024), 3)}


def scenario_c_recovery() -> dict:
    """Crash at step 1, then resume from the SAME db file and finish."""
    store = StateStore(DB)
    eng = Engine(DummyModel(), _registry(), max_steps=8)
    crash_t = time.perf_counter()
    try:
        eng.run("what is 6*7", store=store, crash_after=1)
        raise AssertionError("expected SimulatedCrash")
    except SimulatedCrash as e:
        task_id = e.task_id
    crash_ms = (time.perf_counter() - crash_t) * 1000

    # Fresh engine, SAME file. This is exactly what a process restart does.
    resume_t = time.perf_counter()
    eng2 = Engine(DummyModel(), _registry(), max_steps=8)
    answer = eng2.run(None, task_id=task_id, store=store)
    recovery_ms = (time.perf_counter() - resume_t) * 1000
    final_status = store.status(task_id)
    store.close()
    return {"crash_ms": round(crash_ms, 2), "recovery_ms": round(recovery_ms, 2),
            "answer": answer, "status": final_status}


def scenario_d_http_roundtrip() -> dict:
    """Offline HTTP service: POST /run (with a db so the service returns a real
    task_id and supports resume), expect JSON {answer, task_id, trace}."""
    from harness.server import serve
    port = 8099
    httpd = serve(host="127.0.0.1", port=port, daemon=True)
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/run",
            data=json.dumps({"prompt": "what is 6*7",
                             "db": f"{HERE}/bench_http.db"}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        health = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=5).read().decode()
    finally:
        httpd.shutdown()
        httpd.server_close()
    return {"answer": body.get("answer"), "has_task_id": bool(body.get("task_id")),
            "has_trace": bool(body.get("trace")), "health": health}


def main() -> None:
    # Best-effort cleanup of temp DBs left by a previous (interrupted) run.
    import glob
    import os
    for f in glob.glob(f"{HERE}/bench_*.db*"):
        try:
            os.remove(f)
        except OSError:
            pass

    print("=" * 64)
    print(" LongHorizon-Harness benchmark (offline, DummyModel)")
    print("=" * 64)

    a = scenario_a_cold_start()
    print("\n[A] COLD START")
    print(f"    import harness (fresh process) : {a['import_s']:.4f} s")
    print(f"    Engine build                  : {a['build_ms']:.2f} ms")
    print(f"    one task end-to-end           : {a['e2e_ms']:.2f} ms")
    print(f"    -> model_calls={a['model_calls']} tool_calls={a['tool_calls']} "
          f"cost=${a['cost_usd']:.6f}")
    print(f"    -> answer: {a['answer']}")

    b = scenario_b_memory()
    print("\n[B] MEMORY PEAK (tracemalloc during one task)")
    print(f"    peak : {b['peak_mb']:.3f} MB")

    c = scenario_c_recovery()
    print("\n[C] CRASH + RECOVERY")
    print(f"    crash at step 1 : {c['crash_ms']:.2f} ms")
    print(f"    resume + finish : {c['recovery_ms']:.2f} ms  "
          f"(event-sourcing replay + heal)")
    print(f"    -> recovered answer: {c['answer']}")

    d = scenario_d_http_roundtrip()
    print("\n[D] HTTP SERVICE (offline round-trip)")
    print(f"    POST /run -> answer={d['answer']!r} "
          f"task_id={d['has_task_id']} trace={d['has_trace']}")
    print(f"    GET /health -> {d['health']}")

    # ---- checks ----
    print("\n" + "-" * 64)
    checks = [
        ("cold start produces a final answer", "42" in a["answer"]),
        ("build is sub-second", a["build_ms"] < 1000),
        ("task is cheap (bounded memory)", b["peak_mb"] < 50),
        ("recovery finishes with correct answer", "42" in c["answer"]),
        ("recovery is fast (<1s offline)", c["recovery_ms"] < 1000),
        ("HTTP returns answer+task_id+trace",
         d["has_task_id"] and d["has_trace"] and "42" in (d["answer"] or "")),
        ("health endpoint ok", "ok" in d["health"]),
    ]
    ok = True
    for name, passed in checks:
        print(f"    [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print("-" * 64)
    print("ALL BENCHMARK CHECKS PASSED" if ok else "BENCHMARK FAILED")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
