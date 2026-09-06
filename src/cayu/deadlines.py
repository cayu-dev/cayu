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

EXECUTION_DEADLINE_METADATA_KEY = "cayu:execution_deadline"


class ExecutionDeadlineExceeded(TimeoutError):
    """New application work was denied; this does not attest external settlement."""

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
_OWNER: ContextVar[asyncio.Task[Any] | None] = ContextVar("cayu_deadline_owner", default=None)


def current_execution_deadline() -> ExecutionDeadline:
    return _CURRENT.get() or ExecutionDeadline()


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
    try:
        async with asyncio.timeout(
            None if already_owned else selected.remaining_seconds()
        ) as timer:
            yield timer
            # Cooperative extension code may consume cancellation, or return
            # after CPU work without another await. Neither grants completion
            # after the authoritative boundary has been observed expired.
            selected.require_admission("execution_completion")
    except BaseException as exc:
        if timer is not None and timer.expired():
            # Keep the original exception/cause, including secondary cleanup
            # failures. Expiry evidence does not claim effects have stopped.
            exc.__dict__["execution_deadline"] = selected.inspection()
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
            async for item in iterator:
                yield item
            return
        while True:
            async with execution_deadline_scope(deadline):
                try:
                    item = await anext(iterator)
                except StopAsyncIteration:
                    return
            yield item
    finally:
        close = getattr(iterator, "aclose", None)
        if close is not None:
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
