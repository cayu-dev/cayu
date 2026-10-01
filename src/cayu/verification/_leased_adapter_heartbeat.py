"""Shared renewal ownership for verification claims and publication leases.

Lease adapters retain durable authority, acknowledgement checks and diagnostic
policy. This driver owns the clock, exact renewal task and loss notification.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import NoReturn, Protocol, TypeVar

from cayu._task_wait import await_shielded_task_outcome, restore_task_cancellation_requests
from cayu.workspaces.observation_recovery import (
    retain_workspace_observation_pending_cancellation_requests,
)

_RenewalT = TypeVar("_RenewalT")


class LeaseHeartbeatPolicy(Protocol[_RenewalT]):
    @property
    def interval_seconds(self) -> float: ...

    @property
    def renewal_task_name(self) -> str: ...

    @property
    def cancel_renewal_on_shutdown(self) -> bool: ...

    @property
    def deadline_can_extend(self) -> bool: ...

    @property
    def deadline(self) -> float: ...

    async def renew(self) -> _RenewalT: ...

    def acknowledge(self, result: _RenewalT | None, started: float) -> None: ...

    def expiry_failure(self, *, renewing: bool) -> BaseException: ...

    def safe_cancellation(self, cancellation: asyncio.CancelledError) -> asyncio.CancelledError: ...

    def raise_late_failure(self, expiry: BaseException, failure: BaseException) -> NoReturn: ...

    def raise_cancellation(self, cancellation: asyncio.CancelledError) -> NoReturn: ...


async def run_leased_adapter_heartbeat(
    lease: LeaseHeartbeatPolicy[_RenewalT],
    *,
    stop: asyncio.Event,
    ownership_lost: asyncio.Future[BaseException],
    clock: Callable[[], float],
) -> None:
    """Maintain a lease while retaining every in-flight renewal through settlement.

    Verification cancels its renewal on owned shutdown. Publication lets the
    mutation finish. A publication deadline can also advance concurrently with
    renewal, so an elapsed wait must recheck that exact shared deadline.
    """

    def record_loss(failure: BaseException) -> None:
        if not ownership_lost.done():
            ownership_lost.set_result(failure)

    while True:
        remaining = lease.deadline - clock()
        if remaining <= 0:
            failure = lease.expiry_failure(renewing=False)
            record_loss(failure)
            raise failure from None
        try:
            await asyncio.wait_for(stop.wait(), timeout=min(lease.interval_seconds, remaining / 2))
            return
        except TimeoutError:
            pass
        started = clock()
        renewal = asyncio.create_task(lease.renew(), name=lease.renewal_task_name)
        try:
            while True:
                outcome = await await_shielded_task_outcome(
                    renewal,
                    timeout_s=max(0.0, lease.deadline - clock()),
                    timeout_after_cancellation_s=0 if lease.cancel_renewal_on_shutdown else None,
                )
                if (
                    not outcome.timed_out
                    or not lease.deadline_can_extend
                    or clock() >= lease.deadline
                ):
                    break
            if outcome.cancellation is not None:
                if lease.cancel_renewal_on_shutdown:
                    renewal.cancel()
                    await renewal
                else:
                    settlement = await await_shielded_task_outcome(
                        renewal, cancellation=outcome.cancellation
                    )
                    if settlement.error is not None:
                        lease.raise_late_failure(outcome.cancellation, settlement.error)
                raise outcome.cancellation
            if outcome.timed_out:
                failure = lease.expiry_failure(renewing=True)
                record_loss(failure)
                settlement = await await_shielded_task_outcome(renewal)
                if settlement.cancellation is not None:
                    cancellation = lease.safe_cancellation(settlement.cancellation)
                    cancellation.__cause__ = failure
                    cancellation.__suppress_context__ = True
                    retain_workspace_observation_pending_cancellation_requests(
                        cancellation, max(settlement.cancellation_requests_consumed, 1)
                    )
                    restore_task_cancellation_requests(
                        settlement.cancellation_requests_consumed, cancellation=cancellation
                    )
                    raise cancellation
                if settlement.error is not None:
                    lease.raise_late_failure(failure, settlement.error)
                raise failure from None
            if outcome.error is not None:
                record_loss(outcome.error)
                raise outcome.error
            lease.acknowledge(outcome.result, started)
        except asyncio.CancelledError as cancellation:
            lease.raise_cancellation(cancellation)
