"""Process-local authority shared by one managed task worker and its handler."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime


class TaskLeaseAuthority:
    """One worker handler's claim: its fixed generation and latest acknowledged lease.

    ``claim_generation`` identifies the claim for the callback's whole lifetime
    and is what every store mutation presents. ``lease_expires_at`` and
    ``deadline_monotonic`` are only the local deadline bookkeeping of renewals.
    """

    def __init__(
        self,
        lease_expires_at: datetime,
        handoff_id: str | None = None,
        *,
        claim_generation: int,
        task_id: str | None = None,
        worker_id: str | None = None,
        deadline_monotonic: float | None = None,
    ) -> None:
        self.task_id = task_id
        self.worker_id = worker_id
        self.claim_generation = claim_generation
        self.initial_lease_expires_at = lease_expires_at
        self._lease_expires_at = lease_expires_at
        self.deadline_monotonic = deadline_monotonic
        self.keep_renewing_on_cancellation = False
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
        presented_claim_generation: int | None = None,
    ) -> bool:
        """Authenticate an operation as belonging to this exact handler."""

        if self.task_id != task_id or self.worker_id != worker_id:
            return False
        if presented_claim_generation is not None:
            return presented_claim_generation == self.claim_generation
        if self.handoff_id != handoff_id:
            return False
        if presented_lease_expires_at is None:
            return True
        return presented_lease_expires_at in {
            self.initial_lease_expires_at,
            self.lease_expires_at,
        }


@dataclass(frozen=True, slots=True)
class ManagedTaskLease:
    """The authority one managed handler mutation presents to its task store.

    ``claim_generation`` is set when the mutation runs inside the handler that
    holds the claim; stores then fence by it and ignore the lease expiry.
    """

    lease_expires_at: datetime | None
    claim_generation: int | None = None


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


def current_task_id() -> str | None:
    """The managed task whose handler this code runs in, or None outside one.

    Only attribution reads it (a brokered sandbox's provenance); it grants nothing.
    """

    authority = _ACTIVE_TASK_LEASE_AUTHORITY.get()
    return None if authority is None else authority.task_id


def active_task_lease_authority(
    *,
    task_id: str,
    worker_id: str,
    handoff_id: str | None = None,
    presented_lease_expires_at: datetime | None = None,
    presented_claim_generation: int | None = None,
) -> TaskLeaseAuthority | None:
    """Return matching runtime-owned authority without trusting equal caller data."""

    authority = _ACTIVE_TASK_LEASE_AUTHORITY.get()
    if authority is None or not authority.matches(
        task_id=task_id,
        worker_id=worker_id,
        handoff_id=handoff_id,
        presented_lease_expires_at=presented_lease_expires_at,
        presented_claim_generation=presented_claim_generation,
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
    presented_claim_generation: int | None = None,
) -> AsyncIterator[ManagedTaskLease]:
    """Resolve a handler mutation to the claim generation its handler holds.

    Caller-provided values never create this authority. Only a task spawned by
    ``run_task_worker`` inherits the private context, and an accepted lease
    must be one of that owner's positively acknowledged ones. The mutation is
    still serialized with renewals, so a store that only understands the
    exact lease expiry receives the latest acknowledged one.
    """

    presented = ManagedTaskLease(presented_lease_expires_at, presented_claim_generation)
    if worker_id is None:
        yield presented
        return
    authority = active_task_lease_authority(
        task_id=task_id,
        worker_id=worker_id,
        handoff_id=handoff_id,
        presented_lease_expires_at=presented_lease_expires_at,
        presented_claim_generation=presented_claim_generation,
    )
    if authority is None:
        yield presented
        return
    if _ACTIVE_TASK_LEASE_MUTATION_AUTHORITY.get() is authority:
        yield ManagedTaskLease(authority.lease_expires_at, authority.claim_generation)
        return
    async with authority.lock:
        managed = ManagedTaskLease(authority.lease_expires_at, authority.claim_generation)
        token = _ACTIVE_TASK_LEASE_MUTATION_AUTHORITY.set(authority)
        try:
            yield managed
        finally:
            _ACTIVE_TASK_LEASE_MUTATION_AUTHORITY.reset(token)
