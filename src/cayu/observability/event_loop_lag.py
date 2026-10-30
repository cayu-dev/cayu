"""Measure how late one process's event loop runs scheduled work.

A Python process runs all of its asyncio work on one CPU core. When sessions,
tools and stores in one process need more CPU than that core has, every
callback waits behind the others and timers fire late. The delay of a timer
that should have fired is the event-loop lag: near zero in a healthy process,
and growing while runtime work queues.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from time import monotonic

logger = logging.getLogger(__name__)

# One timer callback this often is the whole cost of watching a process.
DEFAULT_SAMPLE_INTERVAL_S = 0.5
# A sample measures one pass of the event loop, and a session step waits for
# the loop many times, so a loop that stays a quarter of a second late already
# adds seconds to every step.
DEFAULT_WARNING_THRESHOLD_S = 0.25
# One slow callback or garbage collection is not saturation: lag must stay at
# or above the threshold for this many samples in a row before it warns.
DEFAULT_SUSTAINED_SAMPLES = 4
# Several monitors in one process watch the same core; one warning per this
# many seconds is enough.
DEFAULT_WARNING_INTERVAL_S = 60.0

_WARNING_LOCK = threading.Lock()
_last_warning_at: float | None = None


@dataclass(frozen=True)
class EventLoopLagSnapshot:
    """Content-free lag measurements of one monitor since it was created."""

    samples: int
    current_lag_s: float
    maximum_lag_s: float
    total_lag_s: float
    lagging_samples: int


class EventLoopLagMonitor:
    """Sample event-loop lag with one timer callback every ``interval_s``.

    Each sample is how late the timer ran. A sample at or above
    ``warning_threshold_s`` counts as lagging. When ``sustained_samples``
    lagging samples come in a row, the monitor logs a warning, at most once
    per ``warning_interval_s`` for the whole process. ``None`` as the
    threshold disables the warning. ``active_sessions``, when given, returns
    how many sessions the caller is running: a lagging sample counts towards a
    warning only while it is above zero, and the warning names the count.
    ``on_sample`` receives every sample in seconds, for an application's own
    metrics. A callback that raises is logged once and dropped.

    ``start()`` and ``stop()`` must be called on the loop's thread.
    ``snapshot()`` can be read from any thread.
    """

    def __init__(
        self,
        *,
        interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
        warning_threshold_s: float | None = DEFAULT_WARNING_THRESHOLD_S,
        sustained_samples: int = DEFAULT_SUSTAINED_SAMPLES,
        warning_interval_s: float = DEFAULT_WARNING_INTERVAL_S,
        active_sessions: Callable[[], int] | None = None,
        on_sample: Callable[[float], None] | None = None,
    ) -> None:
        _require_positive_seconds(interval_s, "interval_s")
        if warning_threshold_s is not None:
            _require_positive_seconds(warning_threshold_s, "warning_threshold_s")
        if type(sustained_samples) is not int or sustained_samples < 1:
            raise ValueError("sustained_samples must be a positive integer.")
        _require_positive_seconds(warning_interval_s, "warning_interval_s")
        if active_sessions is not None and not callable(active_sessions):
            raise TypeError("active_sessions must be callable.")
        if on_sample is not None and not callable(on_sample):
            raise TypeError("on_sample must be callable.")
        self._interval_s = float(interval_s)
        self._warning_threshold_s = warning_threshold_s
        self._sustained_samples = sustained_samples
        self._warning_interval_s = float(warning_interval_s)
        self._active_sessions = active_sessions
        self._on_sample = on_sample
        self._loop: asyncio.AbstractEventLoop | None = None
        self._handle: asyncio.TimerHandle | None = None
        self._lock = threading.Lock()
        self._samples = 0
        self._current_lag_s = 0.0
        self._maximum_lag_s = 0.0
        self._total_lag_s = 0.0
        self._lagging_samples = 0
        # Only the loop's thread reads and writes the run of lagging samples.
        self._lagging_in_a_row = 0

    @property
    def running(self) -> bool:
        return self._handle is not None

    def start(self) -> None:
        """Start sampling the running event loop."""

        if self._handle is not None:
            raise RuntimeError("The event-loop lag monitor is already running.")
        self._loop = asyncio.get_running_loop()
        # Late samples from an earlier monitoring period are not sustained lag.
        self._lagging_in_a_row = 0
        self._schedule(self._loop.time())

    def stop(self) -> None:
        """Stop sampling; the recorded measurements stay readable."""

        handle, self._handle = self._handle, None
        if handle is not None:
            handle.cancel()

    def snapshot(self) -> EventLoopLagSnapshot:
        with self._lock:
            return EventLoopLagSnapshot(
                samples=self._samples,
                current_lag_s=self._current_lag_s,
                maximum_lag_s=self._maximum_lag_s,
                total_lag_s=self._total_lag_s,
                lagging_samples=self._lagging_samples,
            )

    def _schedule(self, now: float) -> None:
        assert self._loop is not None
        due = now + self._interval_s
        self._handle = self._loop.call_at(due, self._sample, due)

    def _sample(self, due: float) -> None:
        assert self._loop is not None
        now = self._loop.time()
        self._record(max(now - due, 0.0))
        # The callback may have stopped the monitor.
        if self._handle is not None:
            self._schedule(now)

    def _record(self, lag_s: float) -> None:
        lagging = self._warning_threshold_s is not None and lag_s >= self._warning_threshold_s
        with self._lock:
            self._samples += 1
            self._current_lag_s = lag_s
            self._maximum_lag_s = max(self._maximum_lag_s, lag_s)
            self._total_lag_s += lag_s
            self._lagging_samples += int(lagging)
        # Sessions are counted only for a lagging sample; a healthy loop pays nothing.
        sessions = self._count_active_sessions() if lagging else None
        if lagging and sessions != 0:
            self._lagging_in_a_row += 1
        else:
            self._lagging_in_a_row = 0
        if self._lagging_in_a_row >= self._sustained_samples and _claim_warning(
            self._warning_interval_s
        ):
            self._warn(lag_s, sessions)
        if self._on_sample is not None:
            try:
                self._on_sample(lag_s)
            except Exception:
                self._on_sample = None
                logger.warning("An event-loop lag callback failed and was removed.", exc_info=True)

    def _count_active_sessions(self) -> int | None:
        if self._active_sessions is None:
            return None
        try:
            sessions = self._active_sessions()
            if type(sessions) is not int or sessions < 0:
                raise TypeError("active_sessions must return a non-negative integer.")
        except Exception:
            self._active_sessions = None
            logger.warning("An event-loop lag session count failed and was removed.", exc_info=True)
            return None
        return sessions

    def _warn(self, lag_s: float, sessions: int | None) -> None:
        assert self._warning_threshold_s is not None
        samples = self._lagging_in_a_row
        running = (
            ""
            if sessions is None
            else f" while {_count(sessions, 'session')} {'was' if sessions == 1 else 'were'} running"
        )
        logger.warning(
            "The event loop in this process stayed %.2f s or more late for %s in a row "
            "(%.2f s now)%s. Runtime work is queuing behind other work on this process's "
            "one CPU core; run sessions in more processes: a task worker with cayu worker "
            "NAME --processes N or cayu.runtime.supervise_process_workers, an eval with "
            "cayu eval run --processes N or a lower max_concurrency.",
            self._warning_threshold_s,
            _count(samples, "sample"),
            lag_s,
            running,
        )


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _claim_warning(interval_s: float) -> bool:
    global _last_warning_at
    now = monotonic()
    with _WARNING_LOCK:
        if _last_warning_at is not None and now - _last_warning_at < interval_s:
            return False
        _last_warning_at = now
        return True


def _require_positive_seconds(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite, positive number of seconds.")
