"""Record event-loop lag on the sessions it slowed.

``EventLoopLagMonitor`` warns in the process log when the event loop stays late. Someone
looking at one slow session through `cayu session show`, `events` or `cayu task show` never
sees that log. ``SessionEventLoopLagRecorder`` also records the lag on each session that was
running in the process while the loop was late, as a ``runtime.event_loop.lagged`` event.

The recorder counts every late sample against the sessions running at that moment and
writes their counts each time the monitor warns: at most once a minute per process, one
event per session that saw lag since its last event. A session that ends between two
warnings loses the lag it saw after its last event. The events are written in the
background; a write that fails is logged and dropped, since the lag is diagnostic only.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass

from cayu._task_wait import cancel_and_join
from cayu.events import Event, EventType
from cayu.observability.event_loop_lag import (
    DEFAULT_SAMPLE_INTERVAL_S,
    DEFAULT_WARNING_THRESHOLD_S,
)

logger = logging.getLogger(__name__)

__all__ = ["SessionEventLoopLagRecorder", "app_session_event_loop_lag_recorder"]


@dataclass
class _SessionLag:
    since: float
    lagging_samples: int = 0
    max_lag_s: float = 0.0


class SessionEventLoopLagRecorder:
    """Tie one process's event-loop lag to the sessions it was running.

    Pass ``sample`` as the monitor's ``on_sample`` and ``warned`` as its ``on_warning``.
    ``active_session_ids`` returns the sessions with a run in progress in this process, and
    ``emit`` writes one event to its session. Call ``aclose`` when the monitor stops.
    """

    def __init__(
        self,
        *,
        active_session_ids: Callable[[], Iterable[str]],
        emit: Callable[[Event], Awaitable[object]],
        threshold_s: float = DEFAULT_WARNING_THRESHOLD_S,
        sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(active_session_ids) or not callable(emit):
            raise TypeError("active_session_ids and emit must be callable.")
        self._active_session_ids = active_session_ids
        self._emit = emit
        self._threshold_s = float(threshold_s)
        self._interval_s = float(sample_interval_s)
        self._monotonic = monotonic
        self._sessions: dict[str, _SessionLag] = {}
        self._last_counted: float | None = None
        self._write: asyncio.Task[None] | None = None
        self._failed = False

    def sample(self, lag_s: float) -> None:
        """Count one late sample against every session running now."""

        if lag_s < self._threshold_s:
            return
        now = self._monotonic()
        if self._last_counted is not None and now - self._last_counted < self._interval_s * 0.9:
            # Several workers of one app share this recorder; their monitors sample the same
            # loop, so a second sample of the same interval counts once.
            return
        running = self._running()
        if running is None:
            return
        self._last_counted = now
        counted: dict[str, _SessionLag] = {}
        for session_id in running:
            # A sample measures the interval before it, so a session's first late sample
            # starts its window one interval back.
            entry = self._sessions.get(session_id) or _SessionLag(since=now - self._interval_s)
            entry.lagging_samples += 1
            entry.max_lag_s = max(entry.max_lag_s, lag_s)
            counted[session_id] = entry
        # A session that is no longer running keeps nothing.
        self._sessions = counted

    def warned(self, lag_s: float) -> None:
        """Write each running session's lag since its last event, in the background."""

        if self._write is not None and not self._write.done():
            # The last writes are still queued behind the late loop; keep counting.
            return
        running = self._running()
        if running is None:
            return
        running_ids = set(running)
        now = self._monotonic()
        events = [
            self._event(session_id, entry, lag_s=lag_s, now=now, running=len(running_ids))
            for session_id, entry in self._sessions.items()
            if session_id in running_ids
        ]
        self._sessions = {}
        if events:
            self._write = asyncio.get_running_loop().create_task(
                self._write_events(events), name="cayu-event-loop-lag-sessions"
            )

    async def aclose(self, *, timeout_s: float = 5.0) -> None:
        """Wait a bounded time for writes in flight, then stop them."""

        write, self._write = self._write, None
        if write is None or write.done():
            return
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(asyncio.shield(write), timeout_s)
            return
        if (failure := await cancel_and_join(write)) is not None:
            raise failure

    def _event(
        self, session_id: str, entry: _SessionLag, *, lag_s: float, now: float, running: int
    ) -> Event:
        interval_ms = round(self._interval_s * 1000)
        return Event(
            type=EventType.RUNTIME_EVENT_LOOP_LAGGED,
            session_id=session_id,
            payload={
                "lag_ms": _ms(lag_s),
                "max_lag_ms": _ms(entry.max_lag_s),
                "lagging_samples": entry.lagging_samples,
                "lagging_ms": entry.lagging_samples * interval_ms,
                "window_ms": _ms(now - entry.since),
                "sample_interval_ms": interval_ms,
                "threshold_ms": _ms(self._threshold_s),
                "sessions_running": running,
                "pid": os.getpid(),
            },
        )

    async def _write_events(self, events: list[Event]) -> None:
        for event in events:
            running = self._running()
            if running is None or event.session_id not in running:
                # It ended while earlier writes waited; its terminal event stays last.
                continue
            try:
                await self._emit(event)
            except Exception:
                if not self._failed:
                    self._failed = True
                    logger.warning(
                        "Recording event-loop lag on a session failed; later failures are "
                        "logged at debug level.",
                        exc_info=True,
                    )
                else:
                    logger.debug("Recording event-loop lag on a session failed.", exc_info=True)

    def _running(self) -> tuple[str, ...] | None:
        try:
            return tuple(self._active_session_ids())
        except Exception:
            logger.debug("Reading the running sessions for event-loop lag failed.", exc_info=True)
            return None


# Kept on the app itself, so the recorder (which refers back to the app through its
# event writer) lives and dies with it.
_APP_RECORDER_ATTRIBUTE = "_cayu_session_event_loop_lag_recorder"


def app_session_event_loop_lag_recorder(app: object) -> SessionEventLoopLagRecorder | None:
    """The recorder over a ``CayuApp``'s running sessions and event writer; ``None`` for a
    runtime that isn't one.

    Every worker of one app shares one recorder, so a session's late samples are counted
    and reported once however many worker loops the app runs.
    """

    session_ids = getattr(getattr(app, "_session_control", None), "active_session_ids", None)
    emit = getattr(getattr(app, "_event_writer", None), "emit", None)
    if not callable(session_ids) or not callable(emit):
        return None
    recorder = getattr(app, _APP_RECORDER_ATTRIBUTE, None)
    if isinstance(recorder, SessionEventLoopLagRecorder):
        return recorder
    recorder = SessionEventLoopLagRecorder(active_session_ids=session_ids, emit=emit)
    with contextlib.suppress(AttributeError):
        setattr(app, _APP_RECORDER_ATTRIBUTE, recorder)
    return recorder


def _ms(seconds: float) -> int:
    return max(round(seconds * 1000), 0)
