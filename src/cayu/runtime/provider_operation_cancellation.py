"""Process-local ownership for provider-operation cancellation tasks."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from cayu.providers import (
    ProviderOperationAdapter,
    ProviderOperationSnapshot,
    ProviderOperationState,
)

DEFAULT_MAX_PROVIDER_OPERATION_CANCELLATION_OWNERS = 1024

ProviderOperationCancellation = Callable[[], Awaitable[ProviderOperationSnapshot]]
_ProviderOperationCancellationKey = tuple[int, str, str]


class ProviderOperationCancellationLifecycleSnapshot(BaseModel):
    """Content-free process-local provider-cancellation ownership state."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    admissions_sealed: bool
    max_active_owners: StrictInt = Field(ge=1)
    active_owners: StrictInt = Field(ge=0)
    active_tasks: StrictInt = Field(ge=0)
    completed_owners: StrictInt = Field(ge=0)
    failed_owners: StrictInt = Field(ge=0)
    cancelled_owners: StrictInt = Field(ge=0)
    shutdown_cancellation_requests: StrictInt = Field(ge=0)
    admission_rejections: StrictInt = Field(ge=0)
    capacity_rejections: StrictInt = Field(ge=0)


class ProviderOperationCancellationAdmissionsSealed(RuntimeError):
    """A new cancellation owner cannot start after the shutdown boundary."""


class ProviderOperationCancellationCapacityExceeded(RuntimeError):
    """A new cancellation owner would exceed the process-local owner bound."""

    def __init__(self, *, max_active_owners: int) -> None:
        self.max_active_owners = max_active_owners
        super().__init__(
            "Provider-operation cancellation capacity is exhausted; at most "
            f"{max_active_owners} cancellation owners may be active."
        )


@dataclass
class _OwnedProviderOperationCancellation:
    task: asyncio.Task[ProviderOperationSnapshot] | None = None
    children: set[asyncio.Task[None]] = field(default_factory=set)
    outcome_recorded: bool = False
    shutdown_cancellation_requested: bool = False


class ProviderOperationCancellationLifecycle:
    """Bound, deduplicate, cancel, and drain provider-operation cancellation."""

    def __init__(
        self,
        *,
        max_active_owners: int = DEFAULT_MAX_PROVIDER_OPERATION_CANCELLATION_OWNERS,
    ) -> None:
        if type(max_active_owners) is not int or max_active_owners <= 0:
            raise ValueError("max_active_owners must be a positive integer.")
        self._max_active_owners = max_active_owners
        self._owners: dict[
            _ProviderOperationCancellationKey,
            _OwnedProviderOperationCancellation,
        ] = {}
        self._admissions_sealed = False
        self._completed_owners = 0
        self._failed_owners = 0
        self._cancelled_owners = 0
        self._shutdown_cancellation_requests = 0
        self._admission_rejections = 0
        self._capacity_rejections = 0
        self._next_owner_ordinal = 0

    def snapshot(self) -> ProviderOperationCancellationLifecycleSnapshot:
        active = tuple(
            owner
            for owner in self._owners.values()
            if (owner.task is not None and not owner.task.done())
            or any(not child.done() for child in owner.children)
        )
        return ProviderOperationCancellationLifecycleSnapshot(
            admissions_sealed=self._admissions_sealed,
            max_active_owners=self._max_active_owners,
            active_owners=len(active),
            active_tasks=sum(
                int(owner.task is not None and not owner.task.done())
                + sum(not child.done() for child in owner.children)
                for owner in active
            ),
            completed_owners=self._completed_owners,
            failed_owners=self._failed_owners,
            cancelled_owners=self._cancelled_owners,
            shutdown_cancellation_requests=self._shutdown_cancellation_requests,
            admission_rejections=self._admission_rejections,
            capacity_rejections=self._capacity_rejections,
        )

    def admit(
        self,
        *,
        adapter: ProviderOperationAdapter,
        state: ProviderOperationState,
        cancellation: ProviderOperationCancellation,
        ownership_lost: asyncio.Event | None,
    ) -> asyncio.Task[ProviderOperationSnapshot]:
        """Return the one active task that cancels this exact provider operation."""

        key = (id(adapter), state.operation_id, state.stream_protocol)
        existing = self._owners.get(key)
        if existing is not None and existing.task is not None:
            return existing.task
        if self._admissions_sealed:
            self._admission_rejections += 1
            raise ProviderOperationCancellationAdmissionsSealed(
                "Provider-operation cancellation admissions are sealed for shutdown."
            )
        if len(self._owners) >= self._max_active_owners:
            self._capacity_rejections += 1
            raise ProviderOperationCancellationCapacityExceeded(
                max_active_owners=self._max_active_owners
            )

        self._next_owner_ordinal += 1
        ordinal = self._next_owner_ordinal
        owner = _OwnedProviderOperationCancellation()

        async def run_cancellation() -> ProviderOperationSnapshot:
            return await cancellation()

        task = asyncio.create_task(
            run_cancellation(),
            name=f"cayu-provider-operation-cancellation:{ordinal}",
        )
        owner.task = task
        self._owners[key] = owner

        def release_if_settled() -> None:
            if (
                task.done()
                and all(child.done() for child in owner.children)
                and self._owners.get(key) is owner
            ):
                self._owners.pop(key, None)

        if ownership_lost is not None:

            async def cancel_after_ownership_loss() -> None:
                await ownership_lost.wait()
                if not task.done():
                    task.cancel("Provider-operation cancellation ownership was lost.")

            ownership_task = asyncio.create_task(
                cancel_after_ownership_loss(),
                name=f"cayu-provider-operation-cancel-ownership:{ordinal}",
            )
            owner.children.add(ownership_task)

            def ownership_settled(completed: asyncio.Task[None]) -> None:
                with suppress(asyncio.CancelledError):
                    completed.exception()
                release_if_settled()

            ownership_task.add_done_callback(ownership_settled)

        def settled(completed: asyncio.Task[ProviderOperationSnapshot]) -> None:
            for child in owner.children:
                if not child.done():
                    child.cancel()
            if not owner.outcome_recorded:
                owner.outcome_recorded = True
                try:
                    failure = completed.exception()
                except asyncio.CancelledError:
                    self._cancelled_owners += 1
                else:
                    if failure is None:
                        self._completed_owners += 1
                    else:
                        self._failed_owners += 1
            release_if_settled()

        task.add_done_callback(settled)
        return task

    def seal(self) -> None:
        """Reject new owners and request cancellation from every active owner once."""

        self._admissions_sealed = True
        for owner in tuple(self._owners.values()):
            task = owner.task
            if (
                task is not None
                and not task.done()
                and not task.cancelling()
                and not owner.shutdown_cancellation_requested
            ):
                owner.shutdown_cancellation_requested = True
                if task.cancel("CayuApp provider-operation cancellation shutdown"):
                    self._shutdown_cancellation_requests += 1
            for child in owner.children:
                if not child.done() and not child.cancelling():
                    child.cancel()

    async def drain(self, *, timeout_s: float) -> bool:
        """Seal cancellation admission and wait boundedly for every exact owner."""

        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, int | float)
            or not isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be a finite positive number.")
        self.seal()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + float(timeout_s)
        while True:
            pending = tuple(
                task
                for owner in self._owners.values()
                for task in (
                    *((owner.task,) if owner.task is not None else ()),
                    *owner.children,
                )
                if not task.done()
            )
            if not pending:
                return True
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            done, _pending = await asyncio.wait(
                pending,
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                return False
            await asyncio.sleep(0)


__all__ = [
    "DEFAULT_MAX_PROVIDER_OPERATION_CANCELLATION_OWNERS",
    "ProviderOperationCancellationAdmissionsSealed",
    "ProviderOperationCancellationCapacityExceeded",
    "ProviderOperationCancellationLifecycle",
    "ProviderOperationCancellationLifecycleSnapshot",
]
