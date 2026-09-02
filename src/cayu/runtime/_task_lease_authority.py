"""Process-local authority shared by one managed task worker and its handler."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from datetime import datetime


class TaskLeaseAuthority:
    """The latest positively acknowledged lease owned by one worker handler."""

    def __init__(
        self,
        lease_expires_at: datetime,
        handoff_id: str | None = None,
        *,
        task_id: str | None = None,
        worker_id: str | None = None,
        deadline_monotonic: float | None = None,
    ) -> None:
        self.task_id = task_id
        self.worker_id = worker_id
        self.initial_lease_expires_at = lease_expires_at
        self._lease_expires_at = lease_expires_at
        self.deadline_monotonic = deadline_monotonic
        self.handoff_id = handoff_id
        self.lock = asyncio.Lock()

    @property
    def lease_expires_at(self) -> datetime:
        return self._lease_expires_at

    @lease_expires_at.setter
    def lease_expires_at(self, value: datetime) -> None:
        self._lease_expires_at = value

    def matches(
        self,
        *,
        task_id: str,
        worker_id: str,
        handoff_id: str | None = None,
        presented_lease_expires_at: datetime | None = None,
    ) -> bool:
        """Authenticate an operation as belonging to this exact handler."""

        if self.task_id != task_id or self.worker_id != worker_id:
            return False
        if self.handoff_id != handoff_id:
            return False
        if presented_lease_expires_at is None:
            return True
        return presented_lease_expires_at in {
            self.initial_lease_expires_at,
            self.lease_expires_at,
        }


_ACTIVE_TASK_LEASE_AUTHORITY: ContextVar[TaskLeaseAuthority | None] = ContextVar(
    "cayu_active_task_lease_authority",
    default=None,
)
_ACTIVE_TASK_LEASE_MUTATION_AUTHORITY: ContextVar[TaskLeaseAuthority | None] = ContextVar(
    "cayu_active_task_lease_mutation_authority",
    default=None,
)


@contextmanager
def bind_task_lease_authority(authority: TaskLeaseAuthority) -> Iterator[None]:
    """Bind runtime-owned lease authority while creating one handler task."""

    token = _ACTIVE_TASK_LEASE_AUTHORITY.set(authority)
    try:
        yield
    finally:
        _ACTIVE_TASK_LEASE_AUTHORITY.reset(token)


def active_task_lease_authority(
    *,
    task_id: str,
    worker_id: str,
    handoff_id: str | None = None,
    presented_lease_expires_at: datetime | None = None,
) -> TaskLeaseAuthority | None:
    """Return matching runtime-owned authority without trusting equal caller data."""

    authority = _ACTIVE_TASK_LEASE_AUTHORITY.get()
    if authority is None or not authority.matches(
        task_id=task_id,
        worker_id=worker_id,
        handoff_id=handoff_id,
        presented_lease_expires_at=presented_lease_expires_at,
    ):
        return None
    return authority


@asynccontextmanager
async def managed_task_lease_mutation(
    *,
    task_id: str,
    worker_id: str | None,
    handoff_id: str | None,
    presented_lease_expires_at: datetime | None,
) -> AsyncIterator[datetime | None]:
    """Serialize a handler mutation with its latest acknowledged lease.

    Caller-provided values never create this authority. Only a task spawned by
    ``run_task_worker`` inherits the private context, and every accepted value
    must be one of that owner's positively acknowledged generations.
    """

    if worker_id is None:
        yield presented_lease_expires_at
        return
    authority = active_task_lease_authority(
        task_id=task_id,
        worker_id=worker_id,
        handoff_id=handoff_id,
        presented_lease_expires_at=presented_lease_expires_at,
    )
    if authority is None:
        yield presented_lease_expires_at
        return
    if _ACTIVE_TASK_LEASE_MUTATION_AUTHORITY.get() is authority:
        yield authority.lease_expires_at
        return
    async with authority.lock:
        token = _ACTIVE_TASK_LEASE_MUTATION_AUTHORITY.set(authority)
        try:
            yield authority.lease_expires_at
        finally:
            _ACTIVE_TASK_LEASE_MUTATION_AUTHORITY.reset(token)
