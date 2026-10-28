"""Runtime execution finish-by boundaries (independent of cost/start-by limits)."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, TypeVar, get_args

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, field_validator

from cayu._clock import normalize_utc_datetime
from cayu._exception_groups import iter_exception_tree

EXECUTION_DEADLINE_METADATA_KEY = "cayu:execution_deadline"
EXECUTION_DEADLINE_ORIGIN_METADATA_KEY = "cayu:execution_deadline_origin"
EXECUTION_DEADLINE_RENEWAL_METADATA_KEY = "cayu:execution_deadline_renewal"
TASK_EVENT_DEADLINE_METADATA_KEY = "cayu:task_event_deadline"
TASK_EVENT_BINDING_METADATA_KEY = "cayu:task_event_binding"
# Every runtime-owned session metadata entry that describes the stored deadline.
EXECUTION_DEADLINE_METADATA_KEYS = (
    EXECUTION_DEADLINE_METADATA_KEY,
    EXECUTION_DEADLINE_ORIGIN_METADATA_KEY,
    EXECUTION_DEADLINE_RENEWAL_METADATA_KEY,
    TASK_EVENT_DEADLINE_METADATA_KEY,
    TASK_EVENT_BINDING_METADATA_KEY,
)

DeadlineOrigin = Literal["explicit", "run_default", "enclosing"]
"""How a session got its execution deadline.

``explicit``: the request that started (or resumed) the session set
``execution_deadline``. ``run_default``: the session took the default bound by
``default_run_deadline``, which ``run_task_worker(execution_timeout_s=...)``
binds around each handler call. ``enclosing``: the session started inside an
``execution_deadline_scope``, such as a workflow step or an eval case. A
deadline recorded before origins existed has origin ``None``.
"""

_DEADLINE_ORIGINS: frozenset[str] = frozenset(get_args(DeadlineOrigin))


class ExecutionDeadlineExceeded(TimeoutError):
    """Execution reached its deadline; this does not attest external settlement."""

    def __init__(self, deadline: ExecutionDeadline, stage: str) -> None:
        self.diagnostics = {**deadline.inspection(), "denied_stage": stage}
        super().__init__(
            f"Execution deadline expired before {stage}; the deadline was {deadline.describe()}."
        )


class ExecutionDeadline(BaseModel):
    """Portable UTC expiry with a live, process-local monotonic upper bound.

    None is explicit absence of a deadline. Only UTC/source/scope are serialized;
    loading a durable value recomputes remaining time, never its original duration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    expires_at: datetime | None = None
    source: str = "runtime"
    scope: str = "execution"
    _monotonic_expiry: float | None = PrivateAttr(default=None)
    _expired_observed: bool = PrivateAttr(default=False)
    # Process-local like the monotonic expiry; sessions store it next to the
    # deadline (EXECUTION_DEADLINE_ORIGIN_METADATA_KEY), not inside it.
    _origin: DeadlineOrigin | None = PrivateAttr(default=None)
    # Continuation planning proves the fenced turn has no pending human wait.
    _event_rebinding_allowed: bool = PrivateAttr(default=False)

    @field_validator("expires_at")
    @classmethod
    def validate_expiry(cls, value: datetime | None) -> datetime | None:
        return None if value is None else normalize_utc_datetime(value, "expires_at")

    @field_validator("source", "scope")
    @classmethod
    def validate_label(cls, value: str) -> str:
        # Diagnostic identifiers, never arbitrary prompts or application content.
        if (
            not value
            or len(value) > 128
            or any(not (c.isascii() and (c.isalnum() or c in "._:-")) for c in value)
        ):
            raise ValueError("Deadline source/scope must be bounded diagnostic identifiers.")
        return value

    def model_post_init(self, context: Any) -> None:
        if self.expires_at is not None:
            self._monotonic_expiry = time.monotonic() + max(
                0.0, (self.expires_at - datetime.now(UTC)).total_seconds()
            )

    @classmethod
    def after(
        cls, seconds: float | None, *, source: str = "runtime", scope: str = "execution"
    ) -> ExecutionDeadline:
        if seconds is not None and (
            isinstance(seconds, bool) or not math.isfinite(seconds) or seconds < 0
        ):
            raise ValueError("Deadline duration must be finite and nonnegative.")
        return cls(
            expires_at=None if seconds is None else datetime.now(UTC) + timedelta(seconds=seconds),
            source=source,
            scope=scope,
        )

    def remaining_seconds(self) -> float | None:
        if self.expires_at is None:
            return None
        if self._expired_observed:
            return 0.0
        wall_remaining = (self.expires_at - datetime.now(UTC)).total_seconds()
        assert self._monotonic_expiry is not None
        remaining = max(0.0, min(wall_remaining, self._monotonic_expiry - time.monotonic()))
        if remaining == 0:
            self._expired_observed = True
        return remaining

    @property
    def expired(self) -> bool:
        return self.remaining_seconds() == 0.0

    def require_admission(self, stage: str) -> None:
        if self.expired:
            raise ExecutionDeadlineExceeded(self, stage)

    @property
    def origin(self) -> DeadlineOrigin | None:
        """How the session got this deadline (see ``DeadlineOrigin``), if recorded."""
        return self._origin

    def with_origin(self, origin: DeadlineOrigin | None) -> ExecutionDeadline:
        """A copy with the same expiry (wall clock and monotonic) and this origin."""
        if origin is not None and origin not in _DEADLINE_ORIGINS:
            raise ValueError(f"Unknown execution deadline origin: {origin!r}.")
        copied = self.model_copy()
        copied._origin = origin
        return copied

    def describe(self) -> str:
        """Name the expiry, how the session got the deadline and who set it, for messages."""
        if self.expires_at is None:
            return "unbounded"
        labels = f"source {self.source}" + (
            "" if self.scope == "execution" else f", scope {self.scope}"
        )
        expiry = self.expires_at.isoformat(timespec="seconds").replace("+00:00", "Z")
        how = _ORIGIN_DESCRIPTIONS.get(self._origin)
        return f"{expiry}, {how} ({labels})" if how else f"{expiry} ({labels})"

    def inspection(self) -> dict[str, Any]:
        """Safe point-in-time evidence; call again for live remaining time."""
        return {**self.model_dump(mode="json"), "remaining_seconds": self.remaining_seconds()}

    def model_context(self) -> str:
        """Explicit opt-in text for a freshly built model request, not a live clock."""
        remaining = self.remaining_seconds()
        if remaining is None:
            return "Runtime execution has no configured deadline."
        assert self.expires_at is not None
        return (
            f"Runtime execution deadline: {self.expires_at.isoformat()}; "
            f"remaining seconds at context construction: {remaining:.3f}."
        )


_ORIGIN_DESCRIPTIONS: dict[str | None, str] = {
    "explicit": "set explicitly with execution_deadline",
    "run_default": "the default for runs started inside a task handler",
    "enclosing": "inherited from an enclosing execution scope",
}


def effective_deadline(*boundaries: ExecutionDeadline) -> ExecutionDeadline:
    """Compose nested scopes by clamping extensions to the earliest expiry."""
    bounded = [boundary for boundary in boundaries if boundary.expires_at is not None]
    if not bounded:
        return ExecutionDeadline()
    # On a tie, keep the first boundary that records how the session got it.
    selected = min(bounded, key=lambda value: (value.expires_at, value._origin is None))
    local_expiries = [
        boundary._monotonic_expiry for boundary in bounded if boundary._monotonic_expiry is not None
    ]
    local_expiry = min(local_expiries)
    expired = any(boundary.expired for boundary in bounded)
    if selected._monotonic_expiry == local_expiry and selected._expired_observed == expired:
        return selected
    # A wall-clock rollback can make a freshly constructed, earlier-UTC child
    # have more local remaining time than its parent. Preserve both constraints.
    selected = selected.model_copy()
    selected._monotonic_expiry = local_expiry
    selected._expired_observed = expired
    return selected


_CURRENT: ContextVar[ExecutionDeadline | None] = ContextVar("cayu_execution_deadline", default=None)
_TIMERS: ContextVar[tuple[tuple[asyncio.Timeout, ExecutionDeadline], ...]] = ContextVar(
    "cayu_deadline_timers", default=()
)


def _deadline_expired_in_task(exc: BaseException, task: asyncio.Future[Any]) -> bool:
    """Check live timer ownership, never reconstruct it from diagnostic metadata."""
    return exc.__dict__.get("_cayu_deadline_expiry_task") is task


def expired_execution_deadline() -> ExecutionDeadline | None:
    """Timer-owned expiry evidence during cancellation settlement, never clock inference."""
    return next((deadline for timer, deadline in reversed(_TIMERS.get()) if timer.expired()), None)


_OWNER: ContextVar[asyncio.Task[Any] | None] = ContextVar("cayu_deadline_owner", default=None)


def current_execution_deadline() -> ExecutionDeadline:
    return _CURRENT.get() or ExecutionDeadline()


_TASK_EVENT: ContextVar[ExecutionDeadline | None] = ContextVar(
    "cayu_task_event_deadline", default=None
)


_TASK_EVENT_BASE: ContextVar[ExecutionDeadline | None] = ContextVar(
    "cayu_task_event_base", default=None
)


_TASK_EVENT_ID: ContextVar[str | None] = ContextVar("cayu_task_event_id", default=None)
# Raw non-event scopes can be longer than the effective event limit. Keep them
# independently so a later event cannot erase their session-level bound.
_TASK_EVENT_INNER: ContextVar[ExecutionDeadline | None] = ContextVar(
    "cayu_task_event_inner_deadline", default=None
)


def independent_event_deadline(
    requested: ExecutionDeadline, *, explicit: bool
) -> ExecutionDeadline:
    """Retain session/outer-scope bounds independently of the current task event."""
    event = current_task_event_deadline()
    current = current_execution_deadline()
    independent_current = (
        current
        if event.expires_at is None
        or (current.expires_at is not None and current.expires_at < event.expires_at)
        else ExecutionDeadline()
    )
    default = _DEFAULT_RUN.get()
    independent_default = (
        default.with_origin("run_default")
        if not explicit and default is not None and default.expires_at != event.expires_at
        else ExecutionDeadline()
    )
    return effective_deadline(
        _TASK_EVENT_BASE.get() or ExecutionDeadline(),
        independent_default,
        independent_current,
        _TASK_EVENT_INNER.get() or ExecutionDeadline(),
        requested if explicit or requested.expires_at != event.expires_at else ExecutionDeadline(),
    )


def current_task_event_id() -> str | None:
    return _TASK_EVENT_ID.get()


def current_task_event_deadline() -> ExecutionDeadline:
    """The immutable event bound to retain when a task admits a continuation."""
    return _TASK_EVENT.get() or ExecutionDeadline()


_DEFAULT_RUN: ContextVar[ExecutionDeadline | None] = ContextVar(
    "cayu_default_run_deadline", default=None
)


def new_run_execution_deadline() -> ExecutionDeadline:
    """Select a new top-level run's default at admission, not deserialization."""
    current = _CURRENT.get()
    default = _DEFAULT_RUN.get()
    if current is not None:
        if _TASK_EVENT.get() is not None and default is not None:
            return effective_deadline(
                default.with_origin("run_default"), current.with_origin("enclosing")
            )
        return current.with_origin("enclosing")
    return ExecutionDeadline() if default is None else default.with_origin("run_default")


@contextmanager
def default_run_deadline(deadline: ExecutionDeadline) -> Iterator[None]:
    """Bound new runs that set no ``execution_deadline`` of their own.

    Unlike ``execution_deadline_scope``, this is only a default: an explicit
    ``RunRequest.execution_deadline`` replaces it, and resuming or continuing an
    existing session keeps that session's stored deadline. It is selected when
    the run starts; constructing or deserializing a request does not apply it. Nested defaults keep
    the earlier expiry. ``run_task_worker(execution_timeout_s=...)`` uses this
    around each handler call.

    A session that took this default records origin ``run_default``. When it is
    top-level and pauses for an approval or a question, settling the pause
    gives it back the time it had left when it paused: the wait for a person
    doesn't count against a default it only inherited. Pass an explicit
    ``execution_deadline`` (for example ``current_handler_time_limit().run_deadline()``)
    to make the wait count.
    """
    if type(deadline) is not ExecutionDeadline:
        raise TypeError("default_run_deadline requires an ExecutionDeadline.")
    token = _DEFAULT_RUN.set(
        effective_deadline(_DEFAULT_RUN.get() or ExecutionDeadline(), deadline)
    )
    try:
        yield
    finally:
        _DEFAULT_RUN.reset(token)


_HANDLER_LIMIT: ContextVar[HandlerTimeLimit | None] = ContextVar(
    "cayu_handler_time_limit", default=None
)
# Enough to identify a retry loop's latest attempts without growing a task
# failure payload with every session a long-running handler started.
_HANDLER_LIMIT_SESSION_IDS = 20


_DEFAULT_HANDLER_GRACE_SECONDS = 30.0


def require_handler_seconds(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite, nonnegative number of seconds.")
    return float(value)


def handler_grace_seconds(timeout_s: float, grace_s: float | None) -> float:
    """Validate a handler time limit and return the grace period inside it.

    ``timeout_s`` is the whole limit. ``grace_s=None`` means 30 seconds, or
    half of ``timeout_s`` when that is shorter. An explicit ``grace_s`` must be
    smaller than ``timeout_s`` (zero is always allowed); a larger one is
    refused with ``ValueError`` instead of being shortened.
    """
    timeout_s = require_handler_seconds(timeout_s, "timeout_s")
    if grace_s is None:
        return min(_DEFAULT_HANDLER_GRACE_SECONDS, timeout_s / 2)
    grace_s = require_handler_seconds(grace_s, "grace_s")
    if grace_s and grace_s >= timeout_s:
        raise ValueError(
            "grace_s must be smaller than timeout_s: the grace period is the end of the "
            "time limit, not extra time after it."
        )
    return grace_s


class HandlerTimeLimit:
    """Bound one call of a task handler, or any other block of handler code.

    ``async with HandlerTimeLimit(300, grace_s=30) as limit:`` starts the clock,
    and the block ends at ``timeout_s`` (300 seconds) at the latest. At
    ``timeout_s - grace_s`` (270 seconds) the limit has passed: ``limit.expired``
    is true and a new top-level ``app.run`` started inside the block is refused
    with ``ExecutionDeadlineExceeded``. Runs already in flight and other code
    keep going for the last ``grace_s`` seconds, so a result that is being
    delivered can finish. At ``timeout_s`` the block is cancelled and leaves with
    ``ExecutionDeadlineExceeded`` (scope ``task_handler``); a run that was still
    in flight ends ``session.interrupted``. Cancellation from outside the block
    passes through unchanged.

    ``grace_s`` defaults to 30 seconds, or half of ``timeout_s`` when that is
    shorter. An explicit ``grace_s`` must be smaller than ``timeout_s`` (or
    zero); otherwise the constructor raises ``ValueError``.

    The limit does not change the deadline stored on sessions. Runs started in
    the block without their own deadline take ``run_deadline()`` as their
    default (see ``default_run_deadline``), and a top-level one that pauses for
    a person gets back, when the pause is settled, the time it had left when it
    paused. A continuation consumed inside another bounded handler is still
    subject to that handler's limit.
    ``session_ids`` lists the sessions started inside the block (the most
    recent ones) and ``sessions_started`` counts all of them.
    ``run_task_worker(execution_timeout_s=..., execution_grace_s=...)`` wraps
    each handler call in one and settles the task when it is exceeded; a
    hand-written worker can wrap its own handler and fail its own task.
    """

    def __init__(
        self,
        timeout_s: float,
        *,
        grace_s: float | None = None,
        source: str = "task_handler",
    ) -> None:
        effective_grace_s = handler_grace_seconds(timeout_s, grace_s)
        ExecutionDeadline(source=source)  # Validates the diagnostic label.
        self.timeout_s = float(timeout_s)
        self.grace_s = effective_grace_s
        self.source = source
        self._deadline: ExecutionDeadline | None = None
        self._session_ids: list[str] = []
        self._sessions_started = 0
        self._timer: asyncio.Timeout | None = None
        self._hard_limit_handle: asyncio.TimerHandle | None = None
        self._token: Any = None
        self._cancelled = asyncio.Event()
        self._expired_at_exit: bool | None = None
        self._initial_cancellations = 0
        self._parent: HandlerTimeLimit | None = None

    @property
    def soft_timeout_s(self) -> float:
        """Seconds after the start when new runs are refused: ``timeout_s - grace_s``."""
        return self.timeout_s - self.grace_s

    @property
    def deadline(self) -> ExecutionDeadline:
        """The soft limit (``timeout_s - grace_s``), fixed when the block was entered."""
        if self._deadline is None:
            raise RuntimeError("HandlerTimeLimit has not been entered.")
        return self._deadline

    def run_deadline(self) -> ExecutionDeadline:
        """The default deadline for runs started in the block: the soft limit, same clock.

        It carries the soft limit's exact UTC and monotonic expiry, so a run that
        ends at its default deadline always finds ``expired`` true afterwards.
        """
        # model_copy keeps the private monotonic expiry.
        return self.deadline.model_copy(update={"scope": "execution"})

    @property
    def started(self) -> bool:
        return self._deadline is not None

    @property
    def expired(self) -> bool:
        """Whether the block passed its soft limit, fixed when the block exits."""
        if self._expired_at_exit is not None:
            return self._expired_at_exit
        return self._deadline is not None and self._deadline.expired

    @property
    def cancelled(self) -> bool:
        """True once ``timeout_s`` passed and the block was cancelled."""
        return self._cancelled.is_set()

    def remaining_seconds(self) -> float | None:
        """Seconds until the soft limit passes, or ``None`` before the block starts."""
        return None if self._deadline is None else self._deadline.remaining_seconds()

    async def wait_cancelled(self) -> None:
        """Return once ``timeout_s`` passed and the block was cancelled."""
        await self._cancelled.wait()

    @property
    def session_ids(self) -> tuple[str, ...]:
        return tuple(self._session_ids)

    @property
    def sessions_started(self) -> int:
        return self._sessions_started

    def exceeded(self, stage: str = "handler") -> ExecutionDeadlineExceeded:
        """Build the error this limit raises, for code that settles it itself."""
        failure = ExecutionDeadlineExceeded(self.deadline, stage)
        failure.diagnostics["grace_seconds"] = self.grace_s
        return failure

    def inspection(self) -> dict[str, Any]:
        """Safe evidence for a task failure payload."""
        return {
            "timeout_seconds": self.timeout_s,
            "grace_seconds": self.grace_s,
            "runs_refused_after_seconds": self.soft_timeout_s,
            "execution_deadline": self.deadline.inspection(),
            "handler_cancelled": self.cancelled,
            "session_ids": list(self._session_ids),
            "sessions_started": self._sessions_started,
        }

    def _record_session(self, session_id: str) -> None:
        if self._parent is not None:
            self._parent._record_session(session_id)
        self._sessions_started += 1
        self._session_ids.append(session_id)
        del self._session_ids[:-_HANDLER_LIMIT_SESSION_IDS]

    async def __aenter__(self) -> HandlerTimeLimit:
        if self._deadline is not None:
            raise RuntimeError("A HandlerTimeLimit can be entered only once.")
        self._deadline = ExecutionDeadline.after(
            self.soft_timeout_s, source=self.source, scope="task_handler"
        )
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        assert task is not None
        self._initial_cancellations = task.cancelling()
        self._timer = asyncio.timeout(self.timeout_s)
        await self._timer.__aenter__()
        # Mark the hard limit in the same loop turn as the timer's cancellation,
        # before the block resumes with it.
        when = self._timer.when()
        assert when is not None
        self._hard_limit_handle = loop.call_at(when, self._cancelled.set)
        self._parent = _HANDLER_LIMIT.get()
        self._token = _HANDLER_LIMIT.set(self)
        return self

    async def __aexit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool | None:
        self._expired_at_exit = self.deadline.expired
        _HANDLER_LIMIT.reset(self._token)
        assert self._timer is not None and self._hard_limit_handle is not None
        self._hard_limit_handle.cancel()
        timer = self._timer
        try:
            suppressed = await timer.__aexit__(exc_type, exc, tb)
        except TimeoutError as timeout:
            # Only asyncio.timeout's translation of its own cancellation lands
            # here; a TimeoutError raised by the block itself passes through.
            if not timer.expired() or exc is None or timeout is exc:
                raise
            raise self.exceeded() from timeout.__cause__
        task = asyncio.current_task()
        if (
            timer.expired()
            and isinstance(exc, BaseExceptionGroup)
            and task is not None
            and task.cancelling() <= self._initial_cancellations
        ):
            leaves = [
                part
                for part in iter_exception_tree(exc)
                if not isinstance(part, BaseExceptionGroup)
            ]
            if any(isinstance(part, asyncio.CancelledError) for part in leaves) and all(
                isinstance(part, (Exception, asyncio.CancelledError)) for part in leaves
            ):
                # The timer consumed its cancellation request, but asyncio.timeout
                # leaves grouped cancellations intact. Preserve cleanup failures as
                # the cause while exposing the owned timeout to task settlement.
                raise self.exceeded() from exc
        return suppressed


def current_handler_time_limit() -> HandlerTimeLimit | None:
    """The innermost active ``HandlerTimeLimit``, if any."""
    return _HANDLER_LIMIT.get()


def admit_handler_run() -> HandlerTimeLimit | None:
    """Refuse a new top-level run after its handler's soft limit."""
    limit = current_handler_time_limit()
    enclosing = limit
    while enclosing is not None:
        if enclosing.expired:
            raise enclosing.exceeded("run")
        enclosing = enclosing._parent
    return limit


def deadline_from_metadata(metadata: Mapping[str, Any]) -> ExecutionDeadline:
    raw = metadata.get(EXECUTION_DEADLINE_METADATA_KEY)
    if EXECUTION_DEADLINE_METADATA_KEY not in metadata:
        return task_event_deadline_from_metadata(metadata)
    deadline = ExecutionDeadline.model_validate(raw)
    origin = metadata.get(EXECUTION_DEADLINE_ORIGIN_METADATA_KEY)
    # An origin this runtime does not know reads as unrecorded, never as an error.
    deadline = deadline.with_origin(origin) if origin in _DEADLINE_ORIGINS else deadline
    return effective_deadline(deadline, task_event_deadline_from_metadata(metadata))


def task_event_deadline_from_metadata(metadata: Mapping[str, Any]) -> ExecutionDeadline:
    """A task event's immutable outer bound, retained across human-wait renewal."""
    raw = metadata.get(TASK_EVENT_DEADLINE_METADATA_KEY)
    return (
        ExecutionDeadline()
        if raw is None
        else ExecutionDeadline.model_validate(raw).with_origin("enclosing")
    )


def waiting_for_a_person_counts(deadline: ExecutionDeadline, *, top_level: bool) -> bool:
    """Whether a session's pause for an approval or a question uses up this deadline.

    It does unless the session is top-level and took the deadline as the default
    for runs started inside a task handler (origin ``run_default``). Settling
    such a pause gives the session back the time it had left when it paused.
    """
    return not (top_level and deadline.origin == "run_default")


class ExecutionDeadlineReport(BaseModel):
    """A paused session's stored execution deadline, as the inbox and pending actions show it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    expires_at: datetime
    source: str
    origin: DeadlineOrigin | None = Field(
        default=None,
        description=(
            "How the session got the deadline: explicit (set on the request), run_default "
            "(the default for runs started inside a task handler) or enclosing (an enclosing "
            "execution scope). Null when the session predates recorded origins."
        ),
    )
    waiting_counts: bool = Field(
        description=(
            "Whether the time this item waits for a person counts against the deadline. "
            "False when settling it gives the session back the time it had left when it paused."
        )
    )

    @classmethod
    def from_metadata(
        cls, metadata: Mapping[str, Any], *, top_level: bool
    ) -> ExecutionDeadlineReport | None:
        """Report a session's stored deadline; ``None`` when it has none or it can't be read."""
        try:
            deadline = deadline_from_metadata(metadata)
        except ValidationError:
            return None
        if deadline.expires_at is None:
            return None
        return cls(
            expires_at=deadline.expires_at,
            source=deadline.source,
            origin=deadline.origin,
            waiting_counts=waiting_for_a_person_counts(deadline, top_level=top_level),
        )


class ExecutionDeadlineRenewal(BaseModel):
    """The time a session had left when it paused for a person, fixed once per pause.

    Stored with a renewed ``run_default`` deadline, so settling the same pause
    again (a retry, or a concurrent settle) gives back the same time instead of
    counting the earlier renewal as time left.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    pause_event_id: str = Field(min_length=1, max_length=256)
    seconds_left: float = Field(ge=0, allow_inf_nan=False)


def renewal_from_metadata(metadata: Mapping[str, Any]) -> ExecutionDeadlineRenewal | None:
    """The stored renewal record, or ``None`` when there is none or it can't be read."""
    raw = metadata.get(EXECUTION_DEADLINE_RENEWAL_METADATA_KEY)
    if raw is None:
        return None
    try:
        return ExecutionDeadlineRenewal.model_validate(raw)
    except ValidationError:
        return None


def deadline_metadata(deadline: ExecutionDeadline) -> dict[str, Any]:
    """The session metadata entries that store ``deadline``; empty when unbounded."""
    if deadline.expires_at is None:
        return {}
    entries: dict[str, Any] = {EXECUTION_DEADLINE_METADATA_KEY: deadline.model_dump(mode="json")}
    if deadline.origin is not None:
        entries[EXECUTION_DEADLINE_ORIGIN_METADATA_KEY] = deadline.origin
    event = _TASK_EVENT.get()
    if event is not None and event.expires_at is not None:
        entries[TASK_EVENT_DEADLINE_METADATA_KEY] = event.model_dump(mode="json")
        base = deadline  # Creation replaces this with the separately retained session bound.
        entries[TASK_EVENT_BINDING_METADATA_KEY] = {
            "event_id": current_task_event_id(),
            "base_deadline": base.model_dump(mode="json"),
            "base_origin": base.origin,
        }
    return entries


@asynccontextmanager
async def execution_deadline_scope(
    deadline: ExecutionDeadline, *, _event_bound: bool = False
) -> AsyncIterator[asyncio.Timeout]:
    """Bind/inherit a boundary and enforce it through ordinary task cancellation.

    Exit runs after enclosed cancellation settlement. Cleanup must not call
    require_admission: it has its own existing bounded settlement budgets.
    """
    parent = current_execution_deadline()
    selected = effective_deadline(parent, deadline)
    token = _CURRENT.set(selected)
    independent_token = _TASK_EVENT_INNER.set(
        _TASK_EVENT_INNER.get()
        if _event_bound or _TASK_EVENT.get() is None
        else effective_deadline(_TASK_EVENT_INNER.get() or ExecutionDeadline(), deadline)
    )
    task = asyncio.current_task()
    already_owned = _OWNER.get() is task and selected.expires_at == parent.expires_at
    owner_token = _OWNER.set(task)
    timer: asyncio.Timeout | None = None
    timeout_cancellation: asyncio.CancelledError | None = None
    try:
        async with asyncio.timeout(
            None if already_owned else selected.remaining_seconds()
        ) as timer:
            timer_token = _TIMERS.set((*_TIMERS.get(), (timer, selected)))
            try:
                yield timer
            except asyncio.CancelledError as cancellation:
                if timer.expired():
                    timeout_cancellation = cancellation
                raise
            finally:
                _TIMERS.reset(timer_token)
            # Cooperative extension code may consume cancellation, or return
            # after CPU work without another await. Neither grants completion
            # after the authoritative boundary has been observed expired.
            selected.require_admission("execution_completion")
    except BaseException as exc:
        failure = exc
        if timer is not None and timer.expired():
            # Only asyncio.timeout's translation of our escaping cancellation
            # becomes the public deadline exception. An application/provider
            # timeout or secondary cleanup failure keeps its original identity.
            if (
                type(exc) is TimeoutError
                and timeout_cancellation is not None
                and exc.__cause__ is timeout_cancellation
            ):
                failure = ExecutionDeadlineExceeded(selected, "execution")
            failure.__dict__["execution_deadline"] = selected.inspection()
            if task is not None and task.cancelling() == 0:
                # asyncio.timeout has consumed its own cancellation request.
                # Keep this process-local authority on the escaping exception;
                # public/serialized deadline observations are not stop authority.
                failure.__dict__["_cayu_deadline_expiry_task"] = task
        if failure is not exc:
            raise failure from timeout_cancellation
        raise
    finally:
        _OWNER.reset(owner_token)
        _TASK_EVENT_INNER.reset(independent_token)
        _CURRENT.reset(token)


_T = TypeVar("_T")


@contextmanager
def _retain_event_inner_deadline(deadline: ExecutionDeadline) -> Iterator[None]:
    """Retain an independent stream bound without changing timer ownership."""
    token = _TASK_EVENT_INNER.set(
        _TASK_EVENT_INNER.get()
        if _TASK_EVENT.get() is None
        else effective_deadline(_TASK_EVENT_INNER.get() or ExecutionDeadline(), deadline)
    )
    try:
        yield
    finally:
        _TASK_EVENT_INNER.reset(token)


async def deadline_stream(
    stream: AsyncIterator[_T],
    deadline: ExecutionDeadline,
    *,
    independent_deadline: ExecutionDeadline | None = None,
) -> AsyncGenerator[_T, None]:
    """Bind only while driving the producer, never across a yield to its caller."""
    iterator = aiter(stream)
    event = current_task_event_deadline()
    retained = independent_deadline
    if retained is None:
        retained = (
            ExecutionDeadline()
            if event.expires_at is not None and deadline.expires_at == event.expires_at
            else deadline
        )
    try:
        if deadline.expires_at is None and current_execution_deadline().expires_at is None:
            while True:
                with (
                    _retain_event_inner_deadline(retained),
                    bind_execution_deadline(deadline, _event_bound=True),
                ):
                    try:
                        item = await anext(iterator)
                    except StopAsyncIteration:
                        return
                yield item
        while True:
            with _retain_event_inner_deadline(retained):
                async with execution_deadline_scope(deadline, _event_bound=True):
                    try:
                        if current_execution_deadline().expired:
                            # Let the owned timer deliver before advancing a
                            # producer that may yield without suspending.
                            await asyncio.sleep(0)
                            current_execution_deadline().require_admission("execution")
                        item = await anext(iterator)
                        if current_execution_deadline().expired:
                            await asyncio.sleep(0)
                            current_execution_deadline().require_admission("execution_completion")
                    except StopAsyncIteration:
                        return
                    except BaseException as failure:
                        # Producer cleanup classifies an owned deadline while its
                        # timer is still live, including expiry between phases.
                        close = getattr(iterator, "aclose", None)
                        if close is not None:
                            await _close_after_deadline_failure(close, failure)
                        raise
            yield item
    finally:
        close = getattr(iterator, "aclose", None)
        if close is not None:
            with (
                _retain_event_inner_deadline(retained),
                bind_execution_deadline(deadline, _event_bound=True),
            ):
                await close()


async def _close_after_deadline_failure(
    close: Callable[[], Awaitable[None]], failure: BaseException
) -> None:
    # The iterator may hold ContextVar tokens or task-bound context managers
    # across yields, so close it in the same task that advanced it. Runtime
    # interruption settlement isolates its fresh cleanup work separately.
    try:
        await close()
    except BaseException as cleanup_failure:
        if isinstance(cleanup_failure, Exception) or (
            isinstance(failure, asyncio.CancelledError)
            and isinstance(cleanup_failure, asyncio.CancelledError)
        ):
            # Keep the primary cancellation bare so an owning asyncio.timeout
            # can translate it. Ordinary cleanup failures remain as evidence,
            # not a synthetic process-control group that skips except Exception.
            raise failure from cleanup_failure
        raise


def resumed_execution_deadline(stored: ExecutionDeadline) -> ExecutionDeadline:
    """A session lifetime deadline is immutable, including across continuation.

    A new, shorter external scope cannot silently become transient authority on
    an existing session: create a child with the shorter boundary instead.
    A task event must already be stored on the session before it can bound recovery.
    """
    enclosing = current_execution_deadline()
    if enclosing.expires_at is not None and (
        stored.expires_at is None or enclosing.expires_at < stored.expires_at
    ):
        raise ValueError(
            "Cannot tighten an existing execution on resume; create a child execution."
        )
    return effective_deadline(enclosing, stored)


@contextmanager
def bind_execution_deadline(
    deadline: ExecutionDeadline, *, _event_bound: bool = False
) -> Iterator[None]:
    """Restore portable context in a process whose parent owns cancellation."""
    token = _CURRENT.set(effective_deadline(current_execution_deadline(), deadline))
    independent_token = _TASK_EVENT_INNER.set(
        _TASK_EVENT_INNER.get()
        if _event_bound or _TASK_EVENT.get() is None
        else effective_deadline(_TASK_EVENT_INNER.get() or ExecutionDeadline(), deadline)
    )
    try:
        yield
    finally:
        _TASK_EVENT_INNER.reset(independent_token)
        _CURRENT.reset(token)


@asynccontextmanager
async def task_event_deadline_scope(
    deadline: ExecutionDeadline, *, event_id: str | None = None
) -> AsyncIterator[None]:
    """Carry a task's outer event deadline into owned sessions and their continuations."""
    parent = _TASK_EVENT.get()
    selected = deadline if parent is None else effective_deadline(parent, deadline)
    base_token = _TASK_EVENT_BASE.set(
        _TASK_EVENT_BASE.get() if parent is not None else current_execution_deadline()
    )
    token = _TASK_EVENT.set(selected)
    identity_token = _TASK_EVENT_ID.set(
        _TASK_EVENT_ID.get() if parent is not None else event_id or str(deadline.expires_at)
    )
    try:
        enclosing = current_execution_deadline()
        limit = current_handler_time_limit()
        if (
            limit is None
            or limit.source != "task_event"
            or limit.grace_s != 0
            or limit._timer is None
            or (
                enclosing.expires_at is not None
                and (selected.expires_at is None or enclosing.expires_at < selected.expires_at)
            )
        ):
            async with execution_deadline_scope(selected, _event_bound=True):
                yield
        else:
            # The handler's event timer is the only cancellation owner. Keep
            # that same live timer visible during session/effect settlement;
            # a passive nested timeout cannot provide expiry evidence.
            with bind_execution_deadline(selected, _event_bound=True):
                owner_token = _OWNER.set(asyncio.current_task())
                timer_token = _TIMERS.set((*_TIMERS.get(), (limit._timer, selected)))
                try:
                    yield
                finally:
                    _TIMERS.reset(timer_token)
                    _OWNER.reset(owner_token)
    finally:
        _TASK_EVENT_BASE.reset(base_token)
        _TASK_EVENT_ID.reset(identity_token)
        _TASK_EVENT.reset(token)
