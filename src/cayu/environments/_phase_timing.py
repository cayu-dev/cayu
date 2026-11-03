"""Wall-clock timing of the steps inside one environment operation.

An environment operation (a factory allocation, a binding, a finalization) runs several
steps: staging admission, archive preparation, transfer, revision checks, copy-back, a Git
baseline, setup commands, releasing a sandbox. Bindings already report those steps as
lifecycle progress, but the runtime publishes progress only when the environment sets a
lifecycle policy, and progress carries no step durations once it is coalesced.

The runtime installs one ``EnvironmentPhaseTimer`` per operation. Lifecycle progress
reports feed it (``observe_environment_phase``) whether or not a policy is set, and code can
time a step of its own with ``environment_phase_step``. The runtime records what the timer
measured on the operation's terminal event as ``phase_timings``: one entry per step, in the
order the steps started, each with its start time and duration. Without an installed timer
both functions do nothing, so a binding used outside the runtime pays nothing.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from cayu.environments.lifecycle import (
    EnvironmentLifecyclePhase,
    EnvironmentLifecycleProgressStatus,
)

__all__ = [
    "MAX_PHASE_TIMINGS",
    "EnvironmentPhaseTimer",
    "environment_phase_step",
    "observe_environment_phase",
]

#: Steps one operation records; later steps are left out.
MAX_PHASE_TIMINGS = 64
_PHASE_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
_UNCOUNTED_PHASES = frozenset({EnvironmentLifecyclePhase.STAGING_ADMISSION.value})
_MAX_COUNTER = 2**53 - 1

PhaseOutcome = Literal["completed", "failed", "unfinished"]


@dataclass
class _Span:
    phase: str
    started_at: datetime
    started_monotonic: float
    depth: int
    duration_ms: int | None = None
    outcome: PhaseOutcome | None = None
    items: int | None = None
    bytes: int | None = None


class EnvironmentPhaseTimer:
    """Collect the start time and duration of each step of one environment operation."""

    def __init__(
        self,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._now = now
        self._monotonic = monotonic
        self._spans: list[_Span] = []
        self._open: list[_Span] = []

    def observe(
        self,
        phase: EnvironmentLifecyclePhase | str,
        status: EnvironmentLifecycleProgressStatus,
        *,
        items: int | None = None,
        bytes_: int | None = None,
    ) -> None:
        """Turn one lifecycle progress report into the start or end of a step."""

        name = _phase_name(phase)
        if name is None:
            return
        if name in _UNCOUNTED_PHASES:
            # Their counters describe the shared staging queue, not this step's work.
            items = bytes_ = None
        if status is EnvironmentLifecycleProgressStatus.STARTED:
            self._start(name)
        elif status is EnvironmentLifecycleProgressStatus.ADVANCED:
            span = self._latest_open(name)
            if span is not None:
                _count(span, items, bytes_)
        elif status is EnvironmentLifecycleProgressStatus.COMPLETED:
            self._finish(name, "completed", items, bytes_)
        else:
            # Failed, deadline exceeded or retained: the step ended without completing.
            self._finish(name, "failed", items, bytes_)

    @contextmanager
    def step(self, phase: str) -> Iterator[None]:
        """Time the block as step ``phase``; an exception records it as failed."""

        name = _phase_name(phase)
        if name is None:
            raise ValueError("A phase name is lowercase letters, digits and underscores.")
        span = self._start(name)
        try:
            yield
        except BaseException:
            self._close(span, "failed")
            raise
        self._close(span, "completed")

    @contextmanager
    def installed(self) -> Iterator[EnvironmentPhaseTimer]:
        """Make this the timer that progress reports and steps in this context feed."""

        token = _CURRENT_TIMER.set(self)
        try:
            yield self
        finally:
            _CURRENT_TIMER.reset(token)

    def payload(self, *, include_unfinished: bool = True) -> list[dict[str, Any]]:
        """The recorded steps in start order, as JSON.

        A step still running is left out, or with ``include_unfinished`` reported as
        ``unfinished`` with the time it has run so far.
        """

        now = self._monotonic()
        rows: list[dict[str, Any]] = []
        for span in self._spans:
            outcome = span.outcome
            duration_ms = span.duration_ms
            if outcome is None:
                if not include_unfinished:
                    continue
                outcome = "unfinished"
                duration_ms = _elapsed_ms(now - span.started_monotonic)
            row: dict[str, Any] = {
                "phase": span.phase,
                "started_at": span.started_at.isoformat(timespec="milliseconds"),
                "duration_ms": duration_ms,
                "outcome": outcome,
                "depth": span.depth,
            }
            if span.items is not None:
                row["items"] = span.items
            if span.bytes is not None:
                row["bytes"] = span.bytes
            rows.append(row)
        return rows

    def _start(self, name: str) -> _Span | None:
        if len(self._spans) >= MAX_PHASE_TIMINGS:
            return None
        started_at = self._now()
        if started_at.tzinfo is None or started_at.utcoffset() is None:
            started_at = started_at.replace(tzinfo=UTC)
        span = _Span(
            phase=name,
            started_at=started_at.astimezone(UTC),
            started_monotonic=self._monotonic(),
            depth=len(self._open),
        )
        self._spans.append(span)
        self._open.append(span)
        return span

    def _finish(
        self, name: str, outcome: PhaseOutcome, items: int | None, bytes_: int | None
    ) -> None:
        span = self._latest_open(name)
        if span is None:
            return
        _count(span, items, bytes_)
        self._close(span, outcome)

    def _close(self, span: _Span | None, outcome: PhaseOutcome) -> None:
        if span is None or span.outcome is not None:
            return
        span.duration_ms = _elapsed_ms(self._monotonic() - span.started_monotonic)
        span.outcome = outcome
        self._open.remove(span)

    def _latest_open(self, name: str) -> _Span | None:
        for span in reversed(self._open):
            if span.phase == name:
                return span
        return None


_CURRENT_TIMER: ContextVar[EnvironmentPhaseTimer | None] = ContextVar(
    "cayu_environment_phase_timer", default=None
)


def observe_environment_phase(
    phase: EnvironmentLifecyclePhase | str,
    status: EnvironmentLifecycleProgressStatus,
    *,
    items_completed: int | None = None,
    bytes_completed: int | None = None,
    **_counters: int | None,
) -> None:
    """Feed one lifecycle progress report to the current timer, if one is installed."""

    timer = _CURRENT_TIMER.get()
    if timer is not None:
        timer.observe(phase, status, items=items_completed, bytes_=bytes_completed)


@contextmanager
def environment_phase_step(phase: str, *, when: bool = True) -> Iterator[None]:
    """Time the block as step ``phase`` of the current operation, if a timer is installed.

    ``when=False`` skips the timing, for a step that has nothing to do.
    """

    timer = _CURRENT_TIMER.get()
    if timer is None or not when:
        yield
        return
    with timer.step(phase):
        yield


def _phase_name(phase: EnvironmentLifecyclePhase | str) -> str | None:
    value = phase.value if isinstance(phase, EnvironmentLifecyclePhase) else phase
    if type(value) is not str or _PHASE_NAME.fullmatch(value) is None:
        return None
    return value


def _count(span: _Span, items: int | None, bytes_: int | None) -> None:
    if type(items) is int and 0 <= items <= _MAX_COUNTER:
        span.items = items
    if type(bytes_) is int and 0 <= bytes_ <= _MAX_COUNTER:
        span.bytes = bytes_


def _elapsed_ms(seconds: float) -> int:
    return min(_MAX_COUNTER, max(0, round(seconds * 1000)))
