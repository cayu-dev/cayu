"""Provider-neutral model-stream deadline ownership.

The four clocks deliberately observe different facts. Raw transport activity
must never authenticate decoded protocol activity or semantic model progress.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Iterable
from contextlib import suppress
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from threading import Lock
from typing import Any, TypeVar, cast
from weakref import finalize

DEFAULT_TRANSPORT_IDLE_TIMEOUT_SECONDS = 120.0
DEFAULT_PROTOCOL_IDLE_TIMEOUT_SECONDS = 120.0
DEFAULT_SEMANTIC_PROGRESS_TIMEOUT_SECONDS = 120.0
DEFAULT_ABSOLUTE_STREAM_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_CONCURRENT_PROVIDER_STREAMS = 100


def _positive_finite_seconds(value: object, field_name: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{field_name} must be a number.")
    try:
        normalized = float(cast("int | float", value))
    except OverflowError:
        raise ValueError(f"{field_name} must be finite and greater than zero.") from None
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{field_name} must be finite and greater than zero.")
    return normalized


def _positive_stream_capacity(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an int.")
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1.")
    return value


@dataclass(frozen=True, slots=True)
class ProviderStreamDeadlines:
    transport_idle_timeout_s: float = DEFAULT_TRANSPORT_IDLE_TIMEOUT_SECONDS
    protocol_idle_timeout_s: float = DEFAULT_PROTOCOL_IDLE_TIMEOUT_SECONDS
    semantic_progress_timeout_s: float = DEFAULT_SEMANTIC_PROGRESS_TIMEOUT_SECONDS
    absolute_stream_timeout_s: float = DEFAULT_ABSOLUTE_STREAM_TIMEOUT_SECONDS
    max_concurrent_streams: int = field(
        default=DEFAULT_MAX_CONCURRENT_PROVIDER_STREAMS,
        compare=False,
    )

    def __post_init__(self) -> None:
        for field_name in (
            "transport_idle_timeout_s",
            "protocol_idle_timeout_s",
            "semantic_progress_timeout_s",
            "absolute_stream_timeout_s",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive_finite_seconds(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "max_concurrent_streams",
            _positive_stream_capacity(self.max_concurrent_streams, "max_concurrent_streams"),
        )


def _resolve_provider_stream_deadlines(
    *,
    transport_idle_timeout_s: float = DEFAULT_TRANSPORT_IDLE_TIMEOUT_SECONDS,
    protocol_idle_timeout_s: float = DEFAULT_PROTOCOL_IDLE_TIMEOUT_SECONDS,
    semantic_progress_timeout_s: float = DEFAULT_SEMANTIC_PROGRESS_TIMEOUT_SECONDS,
    absolute_stream_timeout_s: float = DEFAULT_ABSOLUTE_STREAM_TIMEOUT_SECONDS,
    max_concurrent_streams: int = DEFAULT_MAX_CONCURRENT_PROVIDER_STREAMS,
    stream_idle_timeout_s: float | None = None,
) -> ProviderStreamDeadlines:
    """Resolve the temporary pre-four-clock provider constructor alias."""

    resolved = ProviderStreamDeadlines(
        transport_idle_timeout_s=transport_idle_timeout_s,
        protocol_idle_timeout_s=protocol_idle_timeout_s,
        semantic_progress_timeout_s=semantic_progress_timeout_s,
        absolute_stream_timeout_s=absolute_stream_timeout_s,
        max_concurrent_streams=max_concurrent_streams,
    )
    if stream_idle_timeout_s is None:
        return resolved
    legacy_idle_timeout_s = _positive_finite_seconds(
        stream_idle_timeout_s,
        "stream_idle_timeout_s",
    )
    defaults = ProviderStreamDeadlines()
    if (
        resolved.transport_idle_timeout_s != defaults.transport_idle_timeout_s
        or resolved.protocol_idle_timeout_s != defaults.protocol_idle_timeout_s
        or resolved.semantic_progress_timeout_s != defaults.semantic_progress_timeout_s
    ):
        raise ValueError(
            "stream_idle_timeout_s cannot be combined with non-default transport, "
            "protocol, or semantic idle timeouts."
        )
    return ProviderStreamDeadlines(
        transport_idle_timeout_s=legacy_idle_timeout_s,
        protocol_idle_timeout_s=legacy_idle_timeout_s,
        semantic_progress_timeout_s=legacy_idle_timeout_s,
        absolute_stream_timeout_s=resolved.absolute_stream_timeout_s,
        max_concurrent_streams=resolved.max_concurrent_streams,
    )


class ProviderDeadlineKind(StrEnum):
    TRANSPORT_IDLE = "transport_idle"
    PROTOCOL_IDLE = "protocol_idle"
    SEMANTIC_IDLE = "semantic_idle"
    ABSOLUTE = "absolute"


class ProviderProgressKind(StrEnum):
    RESPONSE_IDENTITY = "response_identity"
    REASONING = "reasoning"
    CONTENT = "content"
    TOOL_CALL = "tool_call"
    HOSTED_TOOL = "hosted_tool"
    CITATION = "citation"
    USAGE = "usage"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class ProviderStreamDeadlineEvidence:
    deadline_kind: ProviderDeadlineKind
    configured_timeout_s: float
    elapsed_s: float
    last_progress_kind: ProviderProgressKind | None
    last_progress_elapsed_s: float | None
    last_progress_at: datetime | None

    def __post_init__(self) -> None:
        if type(self.deadline_kind) is not ProviderDeadlineKind:
            raise TypeError("deadline_kind must be ProviderDeadlineKind.")
        _positive_finite_seconds(self.configured_timeout_s, "configured_timeout_s")
        if type(self.elapsed_s) not in {int, float} or not math.isfinite(self.elapsed_s):
            raise ValueError("elapsed_s must be a finite non-negative number.")
        if self.elapsed_s < 0:
            raise ValueError("elapsed_s must be a finite non-negative number.")
        if self.last_progress_kind is None:
            if self.last_progress_elapsed_s is not None or self.last_progress_at is not None:
                raise ValueError("Progress evidence must be present or absent as one unit.")
            return
        if type(self.last_progress_kind) is not ProviderProgressKind:
            raise TypeError("last_progress_kind must be ProviderProgressKind or None.")
        last_progress_elapsed_s = self.last_progress_elapsed_s
        if (
            last_progress_elapsed_s is None
            or type(last_progress_elapsed_s) not in {int, float}
            or not math.isfinite(last_progress_elapsed_s)
            or last_progress_elapsed_s < 0
            or last_progress_elapsed_s > self.elapsed_s
        ):
            raise ValueError("last_progress_elapsed_s must be within stream elapsed time.")
        if self.last_progress_at is None:
            raise ValueError("last_progress_at is required with progress evidence.")
        if self.last_progress_at.tzinfo is None or self.last_progress_at.utcoffset() is None:
            raise ValueError("last_progress_at must include a timezone.")

    def payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "provider_deadline_kind": self.deadline_kind.value,
            "provider_deadline_timeout_s": self.configured_timeout_s,
            "provider_stream_elapsed_s": self.elapsed_s,
        }
        if self.last_progress_kind is not None:
            payload["provider_last_progress_kind"] = self.last_progress_kind.value
        if self.last_progress_elapsed_s is not None:
            payload["provider_last_progress_elapsed_s"] = self.last_progress_elapsed_s
        if self.last_progress_at is not None:
            payload["provider_last_progress_at"] = self.last_progress_at.isoformat()
        return payload


class ProviderStreamDeadlineExceeded(TimeoutError):
    """Private typed expiry propagated until a provider mints its public error."""

    def __init__(
        self,
        evidence: ProviderStreamDeadlineEvidence,
        *,
        stream_cleanup_failed: bool = False,
    ) -> None:
        if type(stream_cleanup_failed) is not bool:
            raise TypeError("stream_cleanup_failed must be a bool.")
        self.evidence = evidence
        self.stream_cleanup_failed = stream_cleanup_failed
        label = evidence.deadline_kind.value.replace("_", " ")
        super().__init__(
            f"Model provider stream exceeded its {label} deadline after "
            f"{evidence.elapsed_s:g} seconds."
        )


_T = TypeVar("_T")
_DEADLINE_PRECEDENCE = (
    ProviderDeadlineKind.ABSOLUTE,
    ProviderDeadlineKind.SEMANTIC_IDLE,
    ProviderDeadlineKind.PROTOCOL_IDLE,
    ProviderDeadlineKind.TRANSPORT_IDLE,
)
_CURRENT_PROVIDER_DEADLINE_CONTROLLER: ContextVar[ProviderStreamDeadlineController | None] = (
    ContextVar("cayu_provider_deadline_controller", default=None)
)
_CURRENT_PROVIDER_DEADLINE_ADMISSION: ContextVar[ProviderStreamDeadlineAdmission | None] = (
    ContextVar("cayu_provider_deadline_admission", default=None)
)
_PROVIDER_DEADLINE_AWAIT_OWNERS_LOCK = Lock()
_PROVIDER_DEADLINE_AWAIT_OWNERS: set[_ProviderDeadlineAwaitOwnership] = set()


class _ProviderDeadlineAwaitOwnership:
    """Bound one dispatch until every retained provider read actually settles."""

    __slots__ = (
        "_awaits",
        "_release_requested",
        "_released",
    )

    def __init__(self) -> None:
        self._awaits: set[asyncio.Future[Any]] = set()
        self._release_requested = False
        self._released = False

    def retain(self, awaited: asyncio.Future[Any]) -> None:
        if awaited.done():
            _consume_deadline_await_outcome(awaited)
            return
        with _PROVIDER_DEADLINE_AWAIT_OWNERS_LOCK:
            if self._released:
                raise RuntimeError("Provider deadline-read ownership is already released.")
            if awaited in self._awaits:
                return
            self._awaits.add(awaited)
        awaited.add_done_callback(self._settled)

    def release(self) -> None:
        with _PROVIDER_DEADLINE_AWAIT_OWNERS_LOCK:
            if self._released:
                return
            self._release_requested = True
            self._release_if_settled_locked()

    def _settled(self, awaited: asyncio.Future[Any]) -> None:
        with _PROVIDER_DEADLINE_AWAIT_OWNERS_LOCK:
            self._awaits.discard(awaited)
            self._release_if_settled_locked()
        _consume_deadline_await_outcome(awaited)

    def _release_if_settled_locked(self) -> None:
        if self._released or not self._release_requested or self._awaits:
            return
        self._released = True
        _PROVIDER_DEADLINE_AWAIT_OWNERS.discard(self)


def _reserve_provider_deadline_await_ownership(
    max_concurrent_streams: int,
) -> _ProviderDeadlineAwaitOwnership:
    """Reserve fixed-capacity ownership before any provider stream code runs."""

    with _PROVIDER_DEADLINE_AWAIT_OWNERS_LOCK:
        if len(_PROVIDER_DEADLINE_AWAIT_OWNERS) >= max_concurrent_streams:
            raise RuntimeError("Provider stream deadline-read capacity is exhausted.")
        ownership = _ProviderDeadlineAwaitOwnership()
        _PROVIDER_DEADLINE_AWAIT_OWNERS.add(ownership)
    return ownership


class ProviderStreamDeadlineAdmission:
    """Own one bounded provider-read slot before durable dispatch."""

    __slots__ = (
        "__weakref__",
        "_claimed",
        "_deadlines",
        "_max_concurrent_streams",
        "_ownership",
        "_ownership_finalizer",
    )

    def __init__(self, deadlines: ProviderStreamDeadlines) -> None:
        if type(deadlines) is not ProviderStreamDeadlines:
            raise TypeError("deadlines must be ProviderStreamDeadlines.")
        ownership = _reserve_provider_deadline_await_ownership(deadlines.max_concurrent_streams)
        self._claimed = False
        self._deadlines = deadlines
        self._max_concurrent_streams = deadlines.max_concurrent_streams
        self._ownership = ownership
        self._ownership_finalizer = finalize(self, ownership.release)

    @property
    def deadlines(self) -> ProviderStreamDeadlines:
        return self._deadlines

    @property
    def max_concurrent_streams(self) -> int:
        return self._max_concurrent_streams

    def claim(self, deadlines: ProviderStreamDeadlines) -> _ProviderDeadlineAwaitOwnership:
        if type(deadlines) is not ProviderStreamDeadlines:
            raise TypeError("deadlines must be ProviderStreamDeadlines.")
        if deadlines != self._deadlines:
            raise ValueError("Provider stream deadline policy changed after dispatch admission.")
        if deadlines.max_concurrent_streams != self._max_concurrent_streams:
            raise ValueError("Provider stream capacity changed after dispatch admission.")
        with _PROVIDER_DEADLINE_AWAIT_OWNERS_LOCK:
            if self._claimed:
                raise RuntimeError("Provider stream deadline admission was already claimed.")
            if not self._ownership_finalizer.alive:
                raise RuntimeError("Provider stream deadline admission is already closed.")
            self._claimed = True
            self._ownership_finalizer.detach()
            return self._ownership

    def close(self) -> None:
        """Release an unused admission; a claimed controller owns its release."""

        self._ownership_finalizer()


def _consume_deadline_await_outcome(awaited: asyncio.Future[Any]) -> None:
    if not awaited.done() or awaited.cancelled():
        return
    with suppress(BaseException):
        awaited.exception()


def _wake_deadline_waiter(wakeup: asyncio.Future[None]) -> None:
    if not wakeup.done():
        wakeup.set_result(None)


class ProviderStreamDeadlineController:
    """One dispatch-scoped monotonic owner for all provider stream clocks."""

    def __init__(
        self,
        deadlines: ProviderStreamDeadlines,
        *,
        admission: ProviderStreamDeadlineAdmission | None = None,
        started_wall_at: datetime | None = None,
    ) -> None:
        if type(deadlines) is not ProviderStreamDeadlines:
            raise TypeError("deadlines must be ProviderStreamDeadlines.")
        loop = asyncio.get_running_loop()
        admitted = current_provider_deadline_admission() if admission is None else admission
        ownership = (
            _reserve_provider_deadline_await_ownership(deadlines.max_concurrent_streams)
            if admitted is None
            else admitted.claim(deadlines)
        )
        self._await_ownership = ownership
        self.deadlines = deadlines
        self._await_ownership_finalizer = finalize(self, ownership.release)
        self._loop = loop
        self._started_at = loop.time()
        wall = datetime.now(UTC) if started_wall_at is None else started_wall_at
        if wall.tzinfo is None or wall.utcoffset() is None:
            raise ValueError("started_wall_at must include a timezone.")
        self._started_wall_at = wall.astimezone(UTC)
        self._last_transport_at = self._started_at
        self._last_protocol_at = self._started_at
        self._last_semantic_at = self._started_at
        self._last_progress_observed_at: float | None = None
        self._last_progress_kind: ProviderProgressKind | None = None
        self._terminal_observed = False

    def close(self) -> None:
        """Release admission only after every retained provider read settles."""

        self._await_ownership_finalizer()

    def retain_dispatched_operation(self, operation: asyncio.Future[Any]) -> None:
        """Retain one already-dispatched provider operation until settlement."""

        if not isinstance(operation, asyncio.Future):
            raise TypeError("operation must be an asyncio Future.")
        if operation.get_loop() is not self._loop:
            raise ValueError("operation must belong to the deadline controller event loop.")
        self._await_ownership.retain(operation)

    def observe_transport(self) -> None:
        self._last_transport_at = self._loop.time()

    def observe_protocol(self) -> None:
        self._last_protocol_at = self._loop.time()

    def observe_semantic(self, kind: ProviderProgressKind) -> None:
        if type(kind) is not ProviderProgressKind:
            raise TypeError("kind must be ProviderProgressKind.")
        observed_at = self._loop.time()
        self._last_semantic_at = observed_at
        self._last_progress_observed_at = observed_at
        self._last_progress_kind = kind
        if kind is ProviderProgressKind.TERMINAL:
            self._terminal_observed = True

    def idle_pause_started(self) -> float:
        return self._loop.time()

    def exclude_idle_pause(
        self,
        started_at: float,
        *,
        kinds: Iterable[ProviderDeadlineKind],
    ) -> None:
        elapsed = max(0.0, self._loop.time() - started_at)
        selected = frozenset(kinds)
        if ProviderDeadlineKind.TRANSPORT_IDLE in selected:
            self._last_transport_at += elapsed
        if ProviderDeadlineKind.PROTOCOL_IDLE in selected:
            self._last_protocol_at += elapsed
        if ProviderDeadlineKind.SEMANTIC_IDLE in selected:
            self._last_semantic_at += elapsed

    def _deadline_at(self, kind: ProviderDeadlineKind) -> float:
        if kind is ProviderDeadlineKind.ABSOLUTE:
            return self._started_at + self.deadlines.absolute_stream_timeout_s
        if kind is ProviderDeadlineKind.SEMANTIC_IDLE:
            return self._last_semantic_at + self.deadlines.semantic_progress_timeout_s
        if kind is ProviderDeadlineKind.PROTOCOL_IDLE:
            return self._last_protocol_at + self.deadlines.protocol_idle_timeout_s
        return self._last_transport_at + self.deadlines.transport_idle_timeout_s

    def _configured_timeout(self, kind: ProviderDeadlineKind) -> float:
        if kind is ProviderDeadlineKind.ABSOLUTE:
            return self.deadlines.absolute_stream_timeout_s
        if kind is ProviderDeadlineKind.SEMANTIC_IDLE:
            return self.deadlines.semantic_progress_timeout_s
        if kind is ProviderDeadlineKind.PROTOCOL_IDLE:
            return self.deadlines.protocol_idle_timeout_s
        return self.deadlines.transport_idle_timeout_s

    def evidence(
        self,
        kinds: Iterable[ProviderDeadlineKind],
        *,
        now: float | None = None,
    ) -> ProviderStreamDeadlineEvidence:
        observed_at = self._loop.time() if now is None else now
        available = frozenset(kinds)
        if not available:
            raise ValueError("At least one deadline kind is required.")
        selected: ProviderDeadlineKind | None = None
        for candidate in _DEADLINE_PRECEDENCE:
            if candidate in available and observed_at >= self._deadline_at(candidate):
                selected = candidate
                break
        if selected is None:
            selected = min(available, key=self._deadline_at)
        elapsed = max(0.0, observed_at - self._started_at)
        last_progress_elapsed = (
            None
            if self._last_progress_observed_at is None
            else max(0.0, self._last_progress_observed_at - self._started_at)
        )
        return ProviderStreamDeadlineEvidence(
            deadline_kind=selected,
            configured_timeout_s=self._configured_timeout(selected),
            elapsed_s=elapsed,
            last_progress_kind=self._last_progress_kind,
            last_progress_elapsed_s=last_progress_elapsed,
            last_progress_at=(
                None
                if last_progress_elapsed is None
                else self._started_wall_at + timedelta(seconds=last_progress_elapsed)
            ),
        )

    async def wait_for(
        self,
        awaitable: Awaitable[_T],
        *,
        kinds: Iterable[ProviderDeadlineKind],
        accept_cancelled_result: Callable[[_T], bool] | None = None,
    ) -> _T:
        selected = tuple(dict.fromkeys(kinds))
        if not selected:
            raise ValueError("At least one deadline kind is required.")
        now = self._loop.time()
        remaining = min(self._deadline_at(kind) - now for kind in selected)
        if remaining <= 0:
            close = getattr(awaitable, "close", None)
            if callable(close):
                close()
            raise ProviderStreamDeadlineExceeded(self.evidence(selected, now=now))
        operation = asyncio.ensure_future(awaitable)
        try:
            while True:
                now = self._loop.time()
                remaining = min(self._deadline_at(kind) - now for kind in selected)
                if remaining <= 0:
                    terminal_was_observed = self._terminal_observed
                    operation.cancel()
                    # Give cooperative reads one scheduling boundary to settle,
                    # but never await opaque cleanup indefinitely. A real caller
                    # cancellation delivered here remains authoritative.
                    await asyncio.sleep(0)
                    accepted, accepted_result = _accepted_cancelled_result(
                        operation,
                        accept_cancelled_result,
                        terminal_was_observed=terminal_was_observed,
                    )
                    if accepted:
                        return cast("_T", accepted_result)
                    cleanup_failed = not operation.cancelled()
                    if not operation.done():
                        cleanup_failed = True
                    elif not operation.cancelled():
                        operation_failure = operation.exception()
                        if operation_failure is not None and _contains_process_control(
                            operation_failure
                        ):
                            raise operation_failure
                        cleanup_failed = True
                    self._await_ownership.retain(operation)
                    raise ProviderStreamDeadlineExceeded(
                        self.evidence(selected),
                        stream_cleanup_failed=cleanup_failed,
                    )

                wakeup: asyncio.Future[None] = self._loop.create_future()
                wakeup_handle = self._loop.call_later(
                    remaining,
                    _wake_deadline_waiter,
                    wakeup,
                )
                try:
                    await asyncio.wait(
                        (operation, wakeup),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    wakeup_handle.cancel()
                    wakeup.cancel()

                now = self._loop.time()
                expired = any(now >= self._deadline_at(kind) for kind in selected)
                if operation.done():
                    if operation.cancelled():
                        return operation.result()
                    operation_failure = operation.exception()
                    if operation_failure is not None:
                        raise operation_failure
                    if expired:
                        accepted, accepted_result = _accepted_cancelled_result(
                            operation,
                            accept_cancelled_result,
                            terminal_was_observed=self._terminal_observed,
                        )
                        if accepted:
                            return cast("_T", accepted_result)
                        raise ProviderStreamDeadlineExceeded(self.evidence(selected, now=now))
                    return operation.result()
                if expired:
                    # Re-enter through the common expiry path so cancellation
                    # settlement and evidence precedence have one owner.
                    continue
                # An observed transport/protocol transition may have extended an
                # idle clock while the read itself remained pending. Re-arm the
                # wakeup against the controller's current deadlines.
        except ProviderStreamDeadlineExceeded:
            # Expiry already cancelled and retained the provider read. Do not
            # inject a second cancellation while it is settling cooperatively.
            self._await_ownership.retain(operation)
            raise
        except BaseException as failure:
            terminal_was_observed = self._terminal_observed
            if not operation.done():
                operation.cancel()
            if isinstance(failure, asyncio.CancelledError) and accept_cancelled_result is not None:
                # A bundled parser can turn cancellation delivered after its
                # authoritative terminal frame into the normalized completion
                # that runtime must publish first. Give only that cooperative
                # transition the same single scheduling boundary used by
                # deadline cleanup; never wait for opaque provider code.
                try:
                    await asyncio.sleep(0)
                except BaseException:
                    self._await_ownership.retain(operation)
                    raise
                accepted, accepted_result = _accepted_cancelled_result(
                    operation,
                    accept_cancelled_result,
                    terminal_was_observed=terminal_was_observed,
                )
                if accepted:
                    return cast("_T", accepted_result)
            self._await_ownership.retain(operation)
            raise


def _accepted_cancelled_result(
    operation: asyncio.Future[_T],
    accept_result: Callable[[_T], bool] | None,
    *,
    terminal_was_observed: bool,
) -> tuple[bool, _T | None]:
    """Accept only a trusted result for terminal evidence seen before cancellation."""

    if (
        accept_result is None
        or not terminal_was_observed
        or not operation.done()
        or operation.cancelled()
    ):
        return False, None
    operation_failure = operation.exception()
    if operation_failure is not None:
        if _contains_process_control(operation_failure):
            raise operation_failure
        return False, None
    result = operation.result()
    return (True, result) if accept_result(result) else (False, None)


def _contains_process_control(exc: BaseException) -> bool:
    pending = [exc]
    while pending:
        current = pending.pop()
        if isinstance(current, (GeneratorExit, KeyboardInterrupt, SystemExit)):
            return True
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
    return False


def current_provider_deadline_admission() -> ProviderStreamDeadlineAdmission | None:
    return _CURRENT_PROVIDER_DEADLINE_ADMISSION.get()


def bind_provider_deadline_admission(
    admission: ProviderStreamDeadlineAdmission,
) -> Token[ProviderStreamDeadlineAdmission | None]:
    if type(admission) is not ProviderStreamDeadlineAdmission:
        raise TypeError("admission must be ProviderStreamDeadlineAdmission.")
    return _CURRENT_PROVIDER_DEADLINE_ADMISSION.set(admission)


def reset_provider_deadline_admission(
    token: Token[ProviderStreamDeadlineAdmission | None],
) -> None:
    _CURRENT_PROVIDER_DEADLINE_ADMISSION.reset(token)


def current_provider_deadline_controller() -> ProviderStreamDeadlineController | None:
    return _CURRENT_PROVIDER_DEADLINE_CONTROLLER.get()


def bind_provider_deadline_controller(
    controller: ProviderStreamDeadlineController,
) -> Token[ProviderStreamDeadlineController | None]:
    if type(controller) is not ProviderStreamDeadlineController:
        raise TypeError("controller must be ProviderStreamDeadlineController.")
    return _CURRENT_PROVIDER_DEADLINE_CONTROLLER.set(controller)


def reset_provider_deadline_controller(
    token: Token[ProviderStreamDeadlineController | None],
) -> None:
    _CURRENT_PROVIDER_DEADLINE_CONTROLLER.reset(token)


def observe_provider_semantic_progress(kind: ProviderProgressKind) -> None:
    controller = current_provider_deadline_controller()
    if controller is not None:
        controller.observe_semantic(kind)


__all__ = [
    "DEFAULT_ABSOLUTE_STREAM_TIMEOUT_SECONDS",
    "DEFAULT_MAX_CONCURRENT_PROVIDER_STREAMS",
    "DEFAULT_PROTOCOL_IDLE_TIMEOUT_SECONDS",
    "DEFAULT_SEMANTIC_PROGRESS_TIMEOUT_SECONDS",
    "DEFAULT_TRANSPORT_IDLE_TIMEOUT_SECONDS",
    "ProviderDeadlineKind",
    "ProviderProgressKind",
    "ProviderStreamDeadlineAdmission",
    "ProviderStreamDeadlineController",
    "ProviderStreamDeadlineEvidence",
    "ProviderStreamDeadlineExceeded",
    "ProviderStreamDeadlines",
    "bind_provider_deadline_admission",
    "bind_provider_deadline_controller",
    "current_provider_deadline_admission",
    "current_provider_deadline_controller",
    "observe_provider_semantic_progress",
    "reset_provider_deadline_admission",
    "reset_provider_deadline_controller",
]
