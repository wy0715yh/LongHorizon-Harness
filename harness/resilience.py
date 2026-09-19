"""Phase 7 - circuit breaking, rate limiting, and backpressure.

The agent calls TWO kinds of unreliable things, and both can take the whole
loop down if we're naive:

  * TOOLS  - a flaky dependency (a 500, a timeout, an outage). If we keep
             hammering it we waste turns and money, and we might pile onto a
             service that is already on fire.
  * MODELS - a rate-limited API (429s). If we call too fast we get throttled
             or hit with a surprise bill.

Phase 7 adds three classic resilience patterns. Each is a SMALL addition to
the plan-act-observe loop - none of them is a new control flow:

  1. CIRCUIT BREAKER (per tool) - after N consecutive failures, STOP calling
     the tool and fail FAST with a structured result, so the model re-plans
     instead of retrying into the void. After a cooldown we probe once
     (HALF_OPEN); success closes the breaker, another failure re-opens it.
  2. RATE LIMITER (model-level) - a token bucket caps calls/sec. When empty we
     WAIT (this waiting IS backpressure: pressure propagates upstream instead
     of us getting 429'd).
  3. BACKPRESSURE QUEUE        - bounds how many tool calls we admit per turn;
     overflow is SHED with a soft failure the model can retry next turn,
     instead of us buffering unbounded work and risking OOM / lockup.

HOW THIS BUILDS ON EARLIER PHASES (this is the interview thread):
  * It sits ON TOP of Phase 3's executor: the breaker only counts a failure
    AFTER the executor's retries are exhausted. Retry first, then trip.
  * It FEEDS Phase 6's error memory: when a breaker trips OPEN we can durably
    record the trip, so the Critic (P6) can warn the agent not to keep
    calling a dead tool. Circuit breaker = fast local reaction; error memory =
    slow cross-task learning.

Zero third-party dependencies. Clocks and sleeps are injectable so the demos
run deterministically and instantly.
"""

from __future__ import annotations

import collections
import time
from enum import Enum
from typing import Any, Callable, Optional


# --------------------------------------------------------------------- circuit
class CircuitState(str, Enum):
    """The three states of a circuit breaker (the standard state machine)."""

    CLOSED = "closed"     # healthy; calls flow through
    OPEN = "open"         # tripped; calls are fast-failed
    HALF_OPEN = "half_open"  # probing after cooldown; limited admits


class CircuitOpen(Exception):
    """Raised by ``CircuitBreaker.call()`` when OPEN (hard fast-fail).

    The engine normally does NOT let this propagate: it catches the open state
    and turns it into a soft ToolResult the model can react to. The exception
    exists for callers who want a hard failure instead of a graceful one.
    """

    def __init__(self, name: str):
        self.name = name
        super().__init__(f"circuit breaker '{name}' is OPEN")


class CircuitBreaker:
    """A per-tool circuit breaker.

    States & transitions
    ---------------------
      CLOSED --(N consecutive failures)--> OPEN
      OPEN   --(cooldown elapses)--> HALF_OPEN
      HALF_OPEN --(success x k)--> CLOSED
      HALF_OPEN --(any failure)--> OPEN   (and restarts the cooldown)

    A breaker is "consecutive-failure" based: a single blip that the executor
    (P3) retries through does NOT trip it. Only *sustained* failure does - that
    is precisely why it sits after the retry layer, not before it.
    """

    def __init__(self, name: str, *, failure_threshold: int = 3,
                 cooldown_s: float = 5.0, half_open_successes: int = 1,
                 clock: Callable[[], float] = time.monotonic):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self.half_open_successes = half_open_successes
        self.clock = clock

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._half_open_ok = 0
        self._opened_at = 0.0

        # Metrics - observable via stats() / export_trace; "you can't improve
        # what you can't measure" applies to resilience too.
        self.total_failures = 0
        self.total_rejections = 0   # calls refused while OPEN (fast-fail)
        self.total_opens = 0
        self.total_closes = 0

    # ------------------------------------------------------------------ runtime
    @property
    def state(self) -> CircuitState:
        return self._state

    def allow(self) -> bool:
        """May a call proceed right now?

        Transitions OPEN -> HALF_OPEN once the cooldown has elapsed, and in
        HALF_OPEN admits up to ``half_open_successes`` probes before deciding.
        """
        if self._state == CircuitState.OPEN:
            if self.clock() - self._opened_at >= self.cooldown_s:
                self._state = CircuitState.HALF_OPEN
                self._half_open_ok = 0
                return True
            return False
        if self._state == CircuitState.HALF_OPEN:
            return self._half_open_ok < self.half_open_successes
        return True  # CLOSED

    def on_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_ok += 1
            if self._half_open_ok >= self.half_open_successes:
                self._state = CircuitState.CLOSED
                self._consecutive_failures = 0
                self.total_closes += 1
        else:
            # Any success in CLOSED resets the consecutive-failure counter.
            self._consecutive_failures = 0

    def on_failure(self) -> None:
        self.total_failures += 1
        self._consecutive_failures += 1
        if self._state == CircuitState.HALF_OPEN:
            self._trip()
        elif self._consecutive_failures >= self.failure_threshold:
            self._trip()

    def on_rejected(self) -> None:
        """Count a fast-fail (a call refused while OPEN)."""
        self.total_rejections += 1

    def _trip(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = self.clock()
        self.total_opens += 1

    def call(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        """Convenience wrapper: enforce allow(), run, record success/failure."""
        if not self.allow():
            self.on_rejected()
            raise CircuitOpen(self.name)
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self.on_failure()
            raise
        self.on_success()
        return result

    def stats(self) -> dict:
        return {
            "name": self.name,
            "state": self._state.value,
            "consecutive_failures": self._consecutive_failures,
            "total_failures": self.total_failures,
            "total_rejections": self.total_rejections,
            "total_opens": self.total_opens,
            "total_closes": self.total_closes,
        }


class CircuitBreakerSet:
    """One breaker per tool name, created lazily. Pass to ``Engine(circuit=...)``.

    Configuration is shared across the set; breakers are per-name so a broken
    ``search`` tool doesn't trip the breaker for a healthy ``calculator``.
    """

    def __init__(self, *, failure_threshold: int = 3, cooldown_s: float = 5.0,
                 half_open_successes: int = 1,
                 clock: Callable[[], float] = time.monotonic):
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self.half_open_successes = half_open_successes
        self.clock = clock
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, name: str) -> CircuitBreaker:
        b = self._breakers.get(name)
        if b is None:
            b = CircuitBreaker(
                name, failure_threshold=self.failure_threshold,
                cooldown_s=self.cooldown_s,
                half_open_successes=self.half_open_successes,
                clock=self.clock,
            )
            self._breakers[name] = b
        return b

    def stats(self) -> list[dict]:
        return [b.stats() for b in self._breakers.values()]


# ----------------------------------------------------------------- rate limit
class RateLimiter:
    """Token-bucket limiter for the model call site (controls QPS).

    ``capacity`` tokens max; ``refill_per_s`` tokens appear each second.
    ``acquire()`` takes ``tokens`` (default 1) and, if blocking, sleeps until
    they're free. That sleep IS backpressure: instead of firing calls into a
    429 wall, we slow down and let the upstream (the model) wait.

    ``sleep`` and ``clock`` are injectable so demos run instantly and
    deterministically (a virtual clock advances only when sleep is called).
    """

    def __init__(self, capacity: int = 5, refill_per_s: float = 1.0,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.capacity = float(capacity)
        self.refill_per_s = refill_per_s
        self.clock = clock
        self.sleep = sleep
        self._tokens = float(capacity)
        self._last = clock()

        self.total_acquired = 0
        self.total_waited_s = 0.0

    def _refill(self) -> None:
        now = self.clock()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_s)
            self._last = now

    def available(self) -> float:
        """Tokens currently available (also triggers a refill)."""
        self._refill()
        return self._tokens

    def acquire(self, tokens: int = 1, block: bool = True,
                timeout: Optional[float] = None) -> bool:
        """Take ``tokens``. Returns True if granted.

        If ``block`` and not enough tokens, sleep in small slices until enough
        arrive or ``timeout`` elapses (None = wait forever). Non-blocking mode
        returns immediately with False when starved - useful for "shed don't
        block" callers.
        """
        if tokens > self.capacity:
            # A single request can never be satisfied; caller must shrink it.
            raise ValueError("tokens requested exceeds bucket capacity")
        deadline = None if timeout is None else self.clock() + timeout
        while True:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                self.total_acquired += 1
                return True
            if not block:
                return False
            if deadline is not None and self.clock() >= deadline:
                return False
            # Wait a fraction of a refill tick, but never overshoot badly.
            wait = (min(0.02, 1.0 / self.refill_per_s)
                    if self.refill_per_s else 0.02)
            self.sleep(wait)
            self.total_waited_s += wait

    def stats(self) -> dict:
        return {
            "capacity": self.capacity,
            "refill_per_s": self.refill_per_s,
            "available": round(self.available(), 4),
            "total_acquired": self.total_acquired,
            "total_waited_s": round(self.total_waited_s, 4),
        }


# --------------------------------------------------------------- backpressure
class BackpressureQueue:
    """Bounded admission buffer for tool calls.

    The engine drains tool calls through this queue. If more calls arrive than
    ``maxsize``, the overflow is SHED (dropped) and the caller gets a soft
    failure to retry next turn - instead of us buffering unbounded work.

    This is backpressure in the classic sense: when the consumer (the tool
    executor) can't keep up, we signal the producer (the model) to slow down by
    shedding, rather than growing memory without bound. In a synchronous loop
    the queue is usually at depth 1, so it stays transparent - it only bites
    under a burst (e.g. the model emits many parallel tool calls at once).
    """

    def __init__(self, maxsize: int = 8):
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = maxsize
        self._q: "collections.deque" = collections.deque()
        self.total_admitted = 0
        self.total_shed = 0

    def admit(self, item: Any) -> bool:
        """Try to admit ``item``. Returns True if admitted, False if shed.

        The queue bounds how many tool calls the engine may admit in ONE turn
        (it is cleared at the start of each model turn). When it is already
        full we REFUSE the new call and return False - we do NOT drop an older
        pending call to make room. The engine turns this False into a soft
        failure (a TOOL message) so the model retries the shed call next turn.
        Dropping the oldest would be wrong here: the engine has already
        committed to running those admitted calls this turn, so "make room"
        would silently lose work the loop expects to execute.
        """
        if len(self._q) >= self.maxsize:
            self.total_shed += 1
            return False
        self._q.append(item)
        self.total_admitted += 1
        return True

    def depth(self) -> int:
        return len(self._q)

    def clear(self) -> None:
        self._q.clear()

    def stats(self) -> dict:
        return {
            "maxsize": self.maxsize,
            "depth": len(self._q),
            "total_admitted": self.total_admitted,
            "total_shed": self.total_shed,
        }
