"""Exclusive, context-scoped phase accumulation; no events/checkpoints are written."""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict, deque
from contextlib import contextmanager, suppress
from contextvars import Context, ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import wraps
from hashlib import sha256
from inspect import isasyncgenfunction
from threading import Lock
from time import monotonic
from typing import Any

from cayu.observability.timing import (
    ModelStepPreparationTiming,
    RuntimePhaseTiming,
    RuntimeTimingConfig,
    RuntimeTimingRecord,
    RuntimeTimingStatus,
    ToolCallTiming,
    ToolRoundTiming,
)

_LOG = logging.getLogger(__name__)
_ROUND_PHASES = (
    "authorization",
    "admission",
    "started_persistence",
    "effect_state",
    "execution",
    "result_processing",
    "staging",
    "sibling_wait",
    "publication_queue_wait",
    "publication",
    "round_commit",
    "unattributed",
)
_WAIT_PHASES = frozenset({"sibling_wait", "publication_queue_wait"})
_PREPARATION_PHASES = ("handoff", "context_policy", "recall", "counting", "preparation")
_CURRENT: ContextVar[Any] = ContextVar("cayu_runtime_timing", default=None)
_CALL: ContextVar[str | None] = ContextVar("cayu_runtime_timing_call", default=None)
_PHASE: ContextVar[Any] = ContextVar("cayu_runtime_timing_phase", default=None)


@dataclass
class _Counters:
    duration_seconds: float = 0
    store_transaction_count: int = 0
    store_lock_wait_seconds: float = 0
    store_execution_seconds: float = 0
    store_commit_seconds: float = 0
    store_bytes_written: int = 0
    first_started: float | None = None
    last_ended: float | None = None
    _lock: Lock = field(default_factory=Lock)

    def add(self, **values):
        # SQLite reads and validation can run off-thread in copied contexts.
        with self._lock:
            for key, value in values.items():
                setattr(self, key, getattr(self, key) + value)

    def observe(self, started, ended, duration):
        """Add exclusive time and widen the phase's first-entry/last-exit window."""
        with self._lock:
            self.duration_seconds += duration
            if self.first_started is None or started < self.first_started:
                self.first_started = started
            if self.last_ended is None or ended > self.last_ended:
                self.last_ended = ended

    def snapshot(self):
        with self._lock:
            return {
                key: value
                for key, value in vars(self).items()
                if key not in {"_lock", "first_started", "last_ended"}
            }

    def window(self):
        with self._lock:
            return self.first_started, self.last_ended


@dataclass
class _Phase:
    counters: _Counters
    parent: Any
    builder: Any
    children: list[tuple[float, float]] = field(default_factory=list)


def current_store_counters():
    phase = _PHASE.get()
    return None if phase is None else phase.counters


@contextmanager
def timing_scope(builder):
    token = _CURRENT.set(builder)
    try:
        yield
    finally:
        _CURRENT.reset(token)


@contextmanager
def call_scope(call_id):
    token = _CALL.set(call_id)
    try:
        yield
    finally:
        _CALL.reset(token)


@contextmanager
def phase_scope(name, *, call_id=None):
    builder = _CURRENT.get()
    if builder is None or builder.closed:
        yield
        return
    selected_call = _CALL.get() if call_id is None else call_id
    if selected_call is not None and selected_call not in builder.calls:
        selected_call = _CALL.get()
        if selected_call not in builder.calls:
            selected_call = None
    counters = builder.counters.setdefault((selected_call, name), _Counters())
    parent = _PHASE.get()
    if parent is not None and parent.builder is not builder:
        # A nested session, such as a foreground subagent started in a task,
        # copies the caller's context. Its phases belong to its own record and
        # must not be subtracted from the caller's execution phase.
        parent = None
    phase = _Phase(counters, parent, builder)
    token = _PHASE.set(phase)
    started = monotonic()
    try:
        yield
    finally:
        ended = monotonic()
        elapsed = max(0, ended - started)
        _PHASE.reset(token)
        # Subtract the union of nested intervals. Parallel siblings overlap,
        # so summing their durations would hide the parent's remaining work.
        covered = 0
        frontier = started
        for first, last in sorted(phase.children):
            first, last = max(started, first), min(ended, last)
            if last > max(first, frontier):
                covered += last - max(first, frontier)
                frontier = last
        counters.observe(started, ended, max(0, elapsed - covered))
        if parent is not None and parent is not phase:
            parent.children.append((started, ended))


def _call_id(arguments, positional: tuple[Any, ...] = ()):
    call = arguments.get("tool_call")
    if call is not None:
        return call.id
    explicit = arguments.get("tool_call_id")
    if explicit is not None:
        return explicit
    for value in positional[1:3]:
        # A string that is not a call in this round is ignored by phase_scope.
        if type(value) is str:
            return value
        selected = getattr(value, "tool_call_id", None) or getattr(
            getattr(value, "call", None), "id", None
        )
        if type(selected) is str:
            return selected
    return None


def timed_phase(name, *, shared=False):
    """Measure only generator execution, excluding downstream consumer backpressure.

    ``shared`` charges whole-round work to the round, not to a call it names.
    """

    def decorate(function):
        if isasyncgenfunction(function):

            @wraps(function)
            async def stream(*args, **kwargs):
                iterator = function(*args, **kwargs)
                selected = None if shared else _call_id(kwargs, args) or _CALL.get()
                try:
                    while True:
                        with call_scope(selected), phase_scope(name):
                            try:
                                value = await anext(iterator)
                            except StopAsyncIteration:
                                return
                        yield value
                finally:
                    with call_scope(selected), phase_scope(name):
                        await iterator.aclose()

            return stream

        @wraps(function)
        async def call(*args, **kwargs):
            selected = None if shared else _call_id(kwargs, args) or _CALL.get()
            with call_scope(selected), phase_scope(name):
                return await function(*args, **kwargs)

        return call

    return decorate


class _Builder:
    def __init__(
        self, recorder, session_id, identity, *, run_epoch=None, preparation=False, recovered=False
    ):
        self.recorder = recorder
        self.session_id = session_id
        self.identity = identity
        self.preparation = preparation
        self.recovered = recovered
        self.run_epoch = run_epoch
        self.started = monotonic()
        self.started_at = datetime.now(UTC)
        self.calls = OrderedDict()
        self.calls_truncated = 0
        self.counters = {}
        self.staged = {}
        self.queued = {}
        self.dispatch_sealed = None
        self.timestamps = {}
        self.committed = False
        self.prepared = False
        self.closed = False
        self.after_round = None
        if preparation:
            # Consume the baseline: a later step in the same epoch, such as a
            # structured-output repair, must not measure from this old commit.
            previous = recorder.last_commits.pop(session_id, None)
            if previous is not None and previous[3] == run_epoch:
                self.started, self.started_at, self.after_round, _ = previous
                now = monotonic()
                handoff = _Counters()
                handoff.observe(self.started, now, max(0, now - self.started))
                self.counters[(None, "handoff")] = handoff

    def register_calls(self, calls):
        omitted = 0
        for call in calls:
            call_id = getattr(call, "id", None) or call.tool_call_id
            name = getattr(call, "name", None) or call.tool_name
            if call_id in self.calls:
                continue
            if len(self.calls) >= self.recorder.config.max_calls_per_round:
                omitted += 1
                continue
            self.calls[call_id] = name
        self.calls_truncated = max(self.calls_truncated, omitted)

    def wall_time(self, value):
        return None if value is None else self.started_at + timedelta(seconds=value - self.started)

    def phases(self, names, *, call_id=None, aggregate=False):
        result = []
        for name in names:
            values = _Counters().snapshot()
            first = last = None
            for (selected, phase), counters in self.counters.items():
                if phase == name and (aggregate or selected == call_id):
                    for field_name, value in counters.snapshot().items():
                        values[field_name] += value
                    started, ended = counters.window()
                    if started is not None and (first is None or started < first):
                        first = started
                    if ended is not None and (last is None or ended > last):
                        last = ended
            result.append(
                RuntimePhaseTiming(
                    name=name,
                    first_started_at=self.wall_time(first),
                    last_completed_at=self.wall_time(last),
                    **values,
                )
            )
        return tuple(result)

    def observe_event(self, event):
        call_id = event.payload.get("tool_call_id")
        if call_id not in self.calls:
            call_id = _CALL.get()
        if call_id not in self.calls:
            return
        timestamps = self.timestamps.setdefault(call_id, {})
        for name in (
            "tool_effect_completed_at",
            "tool_terminal_staged_at",
            "tool_terminal_publication_started_at",
        ):
            value = event.payload.get(name)
            if isinstance(value, str):
                with suppress(ValueError):
                    timestamps[name] = datetime.fromisoformat(value)

    def mark_staged(self, call_id, event):
        if call_id in self.calls:
            self.staged[call_id] = monotonic()
            self.observe_event(event)

    def _wait(self, call_id, name, started, ended):
        if ended > started:
            self.counters.setdefault((call_id, name), _Counters()).observe(
                started, ended, ended - started
            )

    def seal_dispatch(self):
        """End sibling waits: no call can still be running once dispatch is sealed.

        A call waits for siblings only until the last other call stages its
        terminal. Time after that, including round sealing, the pre-publication
        read and earlier calls' ordered publication, is the publication queue.
        """
        if self.dispatch_sealed is not None:
            return
        sealed = monotonic()
        self.dispatch_sealed = sealed
        for call_id, staged in self.staged.items():
            last_sibling = max(
                (other for other_id, other in self.staged.items() if other_id != call_id),
                default=staged,
            )
            waited_until = min(max(staged, last_sibling), sealed)
            self._wait(call_id, "sibling_wait", staged, waited_until)
            self.queued[call_id] = waited_until
        self.staged.clear()

    def mark_publication_started(self, call_id):
        # Publication only starts after dispatch; paths without an explicit
        # seal (continuations and stops) are sealed by their first publication.
        self.seal_dispatch()
        queued = self.queued.pop(call_id, None)
        if queued is not None:
            self._wait(call_id, "publication_queue_wait", queued, monotonic())

    def mark_committed(self):
        self.committed = True
        self.recorder.last_commits[self.session_id] = (
            monotonic(),
            datetime.now(UTC),
            self.identity.tool_round_id,
            self.run_epoch,
        )
        self.recorder.last_commits.move_to_end(self.session_id)
        while len(self.recorder.last_commits) > self.recorder.config.recent_capacity:
            self.recorder.last_commits.popitem(last=False)

    def finish(self, *, complete=False):
        if self.closed:
            return
        self.closed = True
        elapsed = max(0, monotonic() - self.started)
        # Derived on the monotonic timeline the phases use, so every phase lies
        # within the record and its bounds match its duration even when the
        # wall clock drifts or is adjusted during the round.
        now = self.wall_time(self.started + elapsed)
        safe = self.recorder.reference
        session_id = safe(self.session_id, "session_id")
        if self.preparation:
            record = ModelStepPreparationTiming(
                session_id=session_id,
                model_step_id=safe(self.identity.model_step_id, "model_step_id", self.session_id),
                after_tool_round_id=None
                if self.after_round is None
                else safe(self.after_round, "tool_round_id", self.session_id),
                started_at=self.started_at,
                completed_at=now,
                duration_seconds=elapsed,
                incomplete=not complete,
                phases=self.phases(_PREPARATION_PHASES, aggregate=True),
            )
        else:
            round_id = safe(self.identity.tool_round_id, "tool_round_id", self.session_id)
            calls = tuple(
                ToolCallTiming(
                    session_id=session_id,
                    tool_round_id=round_id,
                    tool_call_id=safe(call_id, "tool_call_id", self.session_id),
                    tool_name=self.recorder.label(name),
                    phases=self.phases(_ROUND_PHASES, call_id=call_id),
                    **self.timestamps.get(call_id, {}),
                )
                for call_id, name in self.calls.items()
            )
            record = ToolRoundTiming(
                session_id=session_id,
                tool_round_id=round_id,
                model_step_id=safe(self.identity.model_step_id, "model_step_id", self.session_id),
                model_attempt_id=safe(
                    self.identity.model_attempt_id, "model_attempt_id", self.session_id
                ),
                started_at=self.started_at,
                completed_at=now,
                duration_seconds=elapsed,
                incomplete=not self.committed,
                recovered=self.recovered,
                calls_truncated=self.calls_truncated,
                phases=self.phases(_ROUND_PHASES, aggregate=True),
                calls=calls,
            )
        self.recorder.publish(self.session_id, tuple(self.calls), record)


@dataclass(frozen=True)
class _TimingDelivery:
    private_session_id: str
    private_call_ids: tuple[str, ...]
    record: RuntimeTimingRecord


class RuntimeTimingRecorder:
    def __init__(self, *, config, sinks, redactor, codec):
        self.config = RuntimeTimingConfig() if config is None else config
        if type(self.config) is not RuntimeTimingConfig:
            raise TypeError("runtime_timing must be RuntimeTimingConfig.")
        self.sinks = tuple(sinks)
        if any(not callable(getattr(sink, "emit_timing", None)) for sink in self.sinks):
            raise TypeError("Timing sinks must provide async emit_timing(record).")
        self.redactor = redactor
        self.codec = codec
        self.recent = deque(maxlen=self.config.recent_capacity)
        self.last_commits = OrderedDict()
        # The queue and worker belong to one event loop. A CayuApp may be used
        # from successive asyncio.run() calls, so both are bound lazily.
        self.queue = None
        self.loop = None
        self.worker = None
        self.dropped = 0
        self.failed = 0

    def label(self, value):
        redacted = self.redactor.redact_text(value)
        return (
            redacted if 0 < len(redacted) <= 512 else "sha256:" + sha256(value.encode()).hexdigest()
        )

    def reference(self, value, field_name, session_id=None):
        if self.redactor.redact_text(value) == value and len(value) <= 512:
            return value
        if self.codec is not None:
            return self.codec.encode(value, field_name=field_name, session_id=session_id)
        return "sha256:" + sha256(value.encode()).hexdigest()

    def begin(
        self, session_id, identity, *, calls=(), run_epoch=None, preparation=False, recovered=False
    ):
        if not self.config.enabled:
            return None
        builder = _Builder(
            self,
            session_id,
            identity,
            run_epoch=run_epoch,
            preparation=preparation,
            recovered=recovered,
        )
        builder.register_calls(calls)
        return builder

    def _bind_loop(self):
        loop = asyncio.get_running_loop()
        if self.loop is loop:
            return self.queue
        # Records left by a previous loop move to a queue owned by this loop;
        # the previous loop's worker and waiters are never awaited here.
        pending = []
        while self.queue is not None and not self.queue.empty():
            pending.append(self.queue.get_nowait())
        self.queue = asyncio.Queue(maxsize=self.config.sink_queue_capacity)
        for delivery in pending:
            self.queue.put_nowait(delivery)
        self.loop = loop
        self.worker = None
        return self.queue

    def _start_worker(self):
        if self.worker is None or self.worker.done():
            # Non-durable observation must not inherit any invocation authority.
            self.worker = asyncio.create_task(self._deliver(self.queue), context=Context())

    def publish(self, session_id, call_ids, record):
        self.recent.append((session_id, record))
        if not self.sinks:
            return
        try:
            queue = self._bind_loop()
        except RuntimeError:
            # No running loop (for example generator finalization at shutdown).
            self.dropped += 1
            return
        try:
            queue.put_nowait(_TimingDelivery(session_id, call_ids, record))
        except asyncio.QueueFull:
            self.dropped += 1
            return
        self._start_worker()

    async def _deliver(self, queue):
        while not queue.empty():
            delivery = queue.get_nowait()
            try:
                for sink in self.sinks:
                    try:
                        async with asyncio.timeout(self.config.sink_timeout_seconds):
                            from cayu.observability.otel import OpenTelemetryEventSink

                            if type(sink) is OpenTelemetryEventSink:
                                from cayu.observability.otel import _emit_opentelemetry_timing

                                await _emit_opentelemetry_timing(sink, delivery)
                            else:
                                await sink.emit_timing(delivery.record)
                    except asyncio.CancelledError:
                        # A sink can raise cancellation without the worker being
                        # cancelled. Keep draining the bounded observation queue.
                        worker = asyncio.current_task()
                        if worker is not None and worker.cancelling():
                            # Shutdown or loop teardown; this record is not delivered.
                            self.dropped += 1
                            raise
                        self.failed += 1
                    except Exception as exc:
                        self.failed += 1
                        _LOG.warning("Runtime timing delivery failed: %s", type(exc).__name__)
            finally:
                queue.task_done()

    async def flush(self):
        if self.queue is None:
            return
        queue = self._bind_loop()
        if not queue.empty():
            self._start_worker()
        await queue.join()

    async def aclose(self):
        """Stop delivery at application shutdown after one bounded flush attempt."""
        if self.queue is None:
            return
        queue = self._bind_loop()
        with suppress(TimeoutError):
            async with asyncio.timeout(self.config.sink_timeout_seconds):
                await self.flush()
        worker, self.worker = self.worker, None
        if worker is not None and not worker.done():
            worker.cancel()
            # A sink that ignores cancellation cannot hold shutdown open.
            await asyncio.wait({worker}, timeout=self.config.sink_timeout_seconds)
        while not queue.empty():
            queue.get_nowait()
            queue.task_done()
            self.dropped += 1
        from cayu.observability.otel import OpenTelemetryEventSink

        for sink in self.sinks:
            if type(sink) is OpenTelemetryEventSink:
                # Records still waiting for a session or tool span are exported
                # under the best known parent rather than left pending.
                try:
                    sink._flush_all_pending_timing()
                except Exception as exc:
                    self.failed += 1
                    _LOG.warning("Runtime timing delivery failed: %s", type(exc).__name__)

    def inspect(self, session_id, limit, record_type):
        if type(limit) is not int or not 1 <= limit <= 1024:
            raise ValueError("Timing limit must be an integer between 1 and 1024.")
        return tuple(
            record
            for private_id, record in reversed(self.recent)
            if (private_id == session_id or record.session_id == session_id)
            and type(record) is record_type
        )[:limit]

    def status(self):
        return RuntimeTimingStatus(
            recent_records=len(self.recent),
            queued_records=0 if self.queue is None else self.queue.qsize(),
            dropped_records=self.dropped,
            failed_deliveries=self.failed,
        )


def current_builder():
    return _CURRENT.get()


def observe_timing_event(event):
    builder = _CURRENT.get()
    if builder is not None and not builder.closed and builder.session_id == event.session_id:
        builder.observe_event(event)
        if builder.preparation and event.type.value == "model.started":
            builder.prepared = True


def timed_tool_round(function):
    @wraps(function)
    async def stream(self, *args, **kwargs):
        recorder = self._executor._event_writer.timing
        builder = recorder.begin(
            self._session.id,
            kwargs["tool_round_identity"],
            calls=kwargs["tool_calls"],
            run_epoch=self._session.run_epoch,
        )
        iterator = function(self, *args, **kwargs)
        try:
            while True:
                with timing_scope(builder), phase_scope("unattributed"):
                    try:
                        value = await anext(iterator)
                    except StopAsyncIteration:
                        return
                yield value
        finally:
            try:
                with timing_scope(builder), phase_scope("unattributed"):
                    await iterator.aclose()
            finally:
                finish_builder(builder)

    return stream


def timed_model_step(function):
    @wraps(function)
    async def stream(self, *args, **kwargs):
        recorder = self._executor._event_writer.timing
        builder = recorder.begin(
            self._session.id,
            kwargs["model_step_identity"],
            preparation=True,
            run_epoch=self._session.run_epoch,
        )
        iterator = function(self, *args, **kwargs)
        try:
            while True:
                with timing_scope(builder), phase_scope("preparation"):
                    try:
                        value = await anext(iterator)
                    except StopAsyncIteration:
                        return
                if builder is not None and builder.prepared:
                    finish_builder(builder, complete=True)
                yield value
        finally:
            try:
                with timing_scope(builder), phase_scope("preparation"):
                    await iterator.aclose()
            finally:
                finish_builder(builder)

    return stream


def finish_builder(builder, *, complete=False):
    if builder is not None:
        try:
            builder.finish(complete=complete)
        except Exception as exc:
            # Observations cannot replace a workload outcome or its cancellation.
            builder.recorder.failed += 1
            _LOG.warning("Runtime timing record failed: %s", type(exc).__name__)


def _live_builder(owner):
    existing = _CURRENT.get()
    if (
        existing is not None
        and not existing.closed
        and not existing.preparation
        and existing.session_id == owner._session.id
        and existing.identity.tool_round_id == owner._identity.tool_round_id
    ):
        return existing
    return None


def _owned_builder(owner, *, recovered, calls=None):
    """Borrow the live round's builder, or the one this round owner already holds."""
    existing = _live_builder(owner)
    if existing is not None:
        return existing, True
    builder = owner._timing
    if builder is None or builder.closed:
        builder = owner._event_writer.timing.begin(
            owner._session.id,
            owner._identity,
            run_epoch=owner._session.run_epoch,
            recovered=recovered,
        )
        owner._timing = builder
    if builder is not None and calls is not None:
        builder.register_calls(calls)
    return builder, False


def _owned_calls(owner, kwargs):
    calls = kwargs.get("tool_calls")
    if calls is not None:
        return calls
    pending = kwargs.get("pending_round")
    if pending is not None:
        return pending.tool_calls
    for phase in (owner._execution, owner._continuation):
        if phase is not None:
            return getattr(phase, "tool_calls", None) or getattr(phase, "_tool_calls", ())
    return ()


def seal_owned_dispatch(owner):
    builder = _live_builder(owner) or owner._timing
    if builder is not None and not builder.closed:
        builder.seal_dispatch()


def finish_owned_timing(owner, *, committed=False):
    """Finish a round owner's record when its caller owns the durable close."""
    builder, owner._timing = owner._timing, None
    if builder is not None and not builder.closed:
        if committed:
            builder.mark_committed()
        finish_builder(builder)


def timed_owned_stage(name, *, recovered=False):
    """Measure one owner step; a later owned publication finishes its record."""

    def decorate(function):
        @wraps(function)
        async def call(self, *args, **kwargs):
            builder, borrowed = _owned_builder(
                self, recovered=recovered, calls=_owned_calls(self, kwargs)
            )
            if builder is None:
                return await function(self, *args, **kwargs)
            selected = _call_id(kwargs, (self, *args)) or _CALL.get()
            try:
                with timing_scope(builder), call_scope(selected), phase_scope(name):
                    return await function(self, *args, **kwargs)
            except BaseException:
                if not borrowed:
                    finish_owned_timing(self)
                raise

        return call

    return decorate


def timed_owned_round(*, recovered, finish=True):
    """Borrow live dispatch timing or observe an owned publication.

    ``recovered`` is explicit: publication after process loss differs from live
    interruption, limit closure or paused-round continuation by its caller, not
    by method naming. It is a bool or a function of the call's keyword
    arguments. ``finish=False`` leaves the durable close to the caller.
    """

    def decorate(function):
        @wraps(function)
        async def stream(self, *args, **kwargs):
            builder, borrowed = _owned_builder(
                self,
                recovered=recovered(kwargs) if callable(recovered) else recovered,
                calls=_owned_calls(self, kwargs),
            )
            iterator = function(self, *args, **kwargs)
            completed = False
            try:
                while True:
                    with timing_scope(builder), phase_scope("unattributed"):
                        try:
                            value = await anext(iterator)
                        except StopAsyncIteration:
                            completed = True
                            return
                    yield value
            finally:
                try:
                    with timing_scope(builder), phase_scope("unattributed"):
                        await iterator.aclose()
                finally:
                    if not borrowed and (finish or not completed):
                        finish_owned_timing(self)

        return stream

    return decorate


def timed_stream(builder, iterator):
    """Run another owner's stream inside a round owner's timing scope."""

    async def stream():
        try:
            while True:
                with timing_scope(builder):
                    try:
                        value = await anext(iterator)
                    except StopAsyncIteration:
                        return
                yield value
        finally:
            with timing_scope(builder):
                await iterator.aclose()

    return stream()
