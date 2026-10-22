"""Runtime execution finish-by boundaries (independent of cost/start-by limits)."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncGenerator, AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator

from cayu._clock import normalize_utc_datetime
from cayu._exception_groups import iter_exception_tree

EXECUTION_DEADLINE_METADATA_KEY = "cayu:execution_deadline"


class ExecutionDeadlineExceeded(TimeoutError):
    """Execution reached its deadline; this does not attest external settlement."""

    def __init__(self, deadline: ExecutionDeadline, stage: str) -> None:
        self.diagnostics = {**deadline.inspection(), "denied_stage": stage}
        super().__init__(f"Execution deadline expired before {stage}.")


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


def effective_deadline(*boundaries: ExecutionDeadline) -> ExecutionDeadline:
    """Compose nested scopes by clamping extensions to the earliest expiry."""
    bounded = [boundary for boundary in boundaries if boundary.expires_at is not None]
    if not bounded:
        return ExecutionDeadline()
    selected = min(bounded, key=lambda value: value.expires_at)
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


_DEFAULT_RUN: ContextVar[ExecutionDeadline | None] = ContextVar(
    "cayu_default_run_deadline", default=None
)


def new_run_execution_deadline() -> ExecutionDeadline:
    """Select a new top-level run's default at admission, not deserialization."""
    current = _CURRENT.get()
    return current if current is not None else _DEFAULT_RUN.get() or ExecutionDeadline()


@contextmanager
def default_run_deadline(deadline: ExecutionDeadline) -> Iterator[None]:
    """Bound new runs that set no ``execution_deadline`` of their own.

    Unlike ``execution_deadline_scope``, this is only a default: an explicit
    ``RunRequest.execution_deadline`` replaces it, and resuming or continuing an
    existing session keeps that session's stored deadline. It is selected when
    the run starts; constructing or deserializing a request does not apply it. Nested defaults keep
    the earlier expiry. ``run_task_worker(execution_timeout_s=...)`` uses this
    around each handler call.
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


class HandlerTimeLimit:
    """Bound one call of a task handler, or any other block of handler code.

    ``async with HandlerTimeLimit(300, grace_s=30) as limit:`` starts the clock.
    At ``timeout_s`` the limit has passed: ``limit.expired`` is true and a new
    top-level ``app.run`` started inside the block is refused with
    ``ExecutionDeadlineExceeded``. Runs already in flight and other code keep
    going for ``grace_s`` more seconds, so a result that is being delivered can
    finish. Then the block is cancelled and leaves with
    ``ExecutionDeadlineExceeded`` (scope ``task_handler``); a run that was still
    in flight ends ``session.interrupted``. Cancellation from outside the block
    passes through unchanged.

    The limit does not change the deadline stored on sessions, so a session
    that pauses for a person retains its deadline. A continuation consumed
    inside another bounded handler is still subject to that handler's limit.
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
        grace_s: float = 30.0,
        source: str = "task_handler",
    ) -> None:
        for name, value in (("timeout_s", timeout_s), ("grace_s", grace_s)):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite, nonnegative number of seconds.")
        ExecutionDeadline(source=source)  # Validates the diagnostic label.
        self.timeout_s = float(timeout_s)
        self.grace_s = float(grace_s)
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
    def deadline(self) -> ExecutionDeadline:
        """The soft limit, fixed when the block was entered."""
        if self._deadline is None:
            raise RuntimeError("HandlerTimeLimit has not been entered.")
        return self._deadline

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
        """True once the grace period ended and the block was cancelled."""
        return self._cancelled.is_set()

    def remaining_seconds(self) -> float | None:
        """Seconds until the limit passes, or ``None`` before the block starts."""
        return None if self._deadline is None else self._deadline.remaining_seconds()

    async def wait_cancelled(self) -> None:
        """Return once the grace period ended and the block was cancelled."""
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
            self.timeout_s, source=self.source, scope="task_handler"
        )
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        assert task is not None
        self._initial_cancellations = task.cancelling()
        self._timer = asyncio.timeout(self.timeout_s + self.grace_s)
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
        return ExecutionDeadline()
    return ExecutionDeadline.model_validate(raw)


@asynccontextmanager
async def execution_deadline_scope(deadline: ExecutionDeadline) -> AsyncIterator[asyncio.Timeout]:
    """Bind/inherit a boundary and enforce it through ordinary task cancellation.

    Exit runs after enclosed cancellation settlement. Cleanup must not call
    require_admission: it has its own existing bounded settlement budgets.
    """
    parent = current_execution_deadline()
    selected = effective_deadline(parent, deadline)
    token = _CURRENT.set(selected)
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
        _CURRENT.reset(token)


_T = TypeVar("_T")


async def deadline_stream(
    stream: AsyncIterator[_T], deadline: ExecutionDeadline
) -> AsyncGenerator[_T, None]:
    """Bind only while driving the producer, never across a yield to its caller."""
    iterator = aiter(stream)
    try:
        if deadline.expires_at is None and current_execution_deadline().expires_at is None:
            while True:
                with bind_execution_deadline(deadline):
                    try:
                        item = await anext(iterator)
                    except StopAsyncIteration:
                        return
                yield item
        while True:
            async with execution_deadline_scope(deadline):
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
                except BaseException:
                    # Producer cleanup classifies an owned deadline while its
                    # timer is still live, including expiry between phases.
                    close = getattr(iterator, "aclose", None)
                    if close is not None:
                        await close()
                    raise
            yield item
    finally:
        close = getattr(iterator, "aclose", None)
        if close is not None:
            with bind_execution_deadline(deadline):
                await close()


def resumed_execution_deadline(stored: ExecutionDeadline) -> ExecutionDeadline:
    """A session lifetime deadline is immutable, including across continuation.

    A new, shorter external scope cannot silently become transient authority on
    an existing session: create a child with the shorter boundary instead.
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
def bind_execution_deadline(deadline: ExecutionDeadline) -> Iterator[None]:
    """Restore portable context in a process whose parent owns cancellation."""
    token = _CURRENT.set(effective_deadline(current_execution_deadline(), deadline))
    try:
        yield
    finally:
        _CURRENT.reset(token)
