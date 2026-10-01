from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from time import monotonic
from typing import NoReturn

import pytest

import cayu.verification._leased_adapter_heartbeat as heartbeat_module
from cayu.verification._leased_adapter_heartbeat import run_leased_adapter_heartbeat


class _ControlledLease:
    interval_seconds = 0.001
    renewal_task_name = "test-exact-lease-renewal"

    def __init__(self, *, cancel_on_shutdown: bool = False, extendable: bool = False) -> None:
        self.cancel_renewal_on_shutdown = cancel_on_shutdown
        self.deadline_can_extend = extendable
        self.deadline = monotonic() + 60
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.renewal: asyncio.Task | None = None
        self.failure: BaseException | None = None
        self.acknowledgements = 0

    async def renew(self) -> object:
        assert self.renewal is None, "A successor must not replace a pending renewal."
        self.renewal = asyncio.current_task()
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            await self.release.wait()
            raise
        if self.failure is not None:
            raise self.failure
        return "renewed"

    def acknowledge(self, result: object | None, started: float) -> None:
        assert result == "renewed"
        self.acknowledgements += 1

    def expiry_failure(self, *, renewing: bool) -> BaseException:
        return RuntimeError("renewal expired" if renewing else "lease expired before renewal")

    def safe_cancellation(self, cancellation: asyncio.CancelledError) -> asyncio.CancelledError:
        return asyncio.CancelledError("safe shutdown")

    def raise_late_failure(self, expiry: BaseException, failure: BaseException) -> NoReturn:
        raise expiry from failure

    def raise_cancellation(self, cancellation: asyncio.CancelledError) -> NoReturn:
        raise cancellation


@asynccontextmanager
async def _running_heartbeat(
    lease: _ControlledLease, *, clock: Callable[[], float] = monotonic
) -> AsyncIterator[tuple[asyncio.Task[None], asyncio.Event, asyncio.Future[BaseException]]]:
    lost: asyncio.Future[BaseException] = asyncio.get_running_loop().create_future()
    stop = asyncio.Event()
    operation = asyncio.create_task(
        run_leased_adapter_heartbeat(lease, stop=stop, ownership_lost=lost, clock=clock)
    )
    try:
        yield operation, stop, lost
    finally:
        stop.set()
        lease.release.set()
        if not operation.done():
            operation.cancel()
        with suppress(BaseException):
            await operation


@pytest.mark.parametrize("cancel_on_shutdown", [False, True])
def test_shutdown_retains_the_exact_renewal_until_its_policy_settles(
    cancel_on_shutdown: bool,
) -> None:
    async def scenario() -> None:
        lease = _ControlledLease(cancel_on_shutdown=cancel_on_shutdown)
        async with _running_heartbeat(lease, clock=monotonic) as (operation, _stop, lost):
            await asyncio.wait_for(lease.entered.wait(), timeout=1)
            renewal = lease.renewal
            assert renewal is not None
            operation.cancel("owned shutdown")
            if cancel_on_shutdown:
                await asyncio.wait_for(lease.cancelled.wait(), timeout=1)
            else:
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                assert not lease.cancelled.is_set()
            assert not operation.done()
            assert not renewal.done()
            lease.release.set()
            with pytest.raises(asyncio.CancelledError):
                await operation
            assert lease.renewal is renewal
            assert renewal.done()
            assert lease.cancelled.is_set() is cancel_on_shutdown
            assert not lost.done()
            assert lease.acknowledgements == 0

    asyncio.run(scenario())


def test_expiry_notifies_loss_before_waiting_for_late_renewal_failure() -> None:
    async def scenario() -> None:
        lease = _ControlledLease()
        lease.deadline = 0.02
        lease.failure = ValueError("late renewal cleanup failed")
        async with _running_heartbeat(lease, clock=lambda: 0.0) as (operation, _stop, lost):
            await asyncio.wait_for(lease.entered.wait(), timeout=1)
            failure = await asyncio.wait_for(asyncio.shield(lost), timeout=1)
            assert str(failure) == "renewal expired"
            assert not operation.done()
            assert lease.renewal is not None and not lease.renewal.done()
            assert not lease.cancelled.is_set()
            lease.release.set()
            with pytest.raises(RuntimeError) as captured:
                await operation
            assert captured.value is failure
            assert captured.value.__cause__ is lease.failure
            assert lease.acknowledgements == 0

    asyncio.run(scenario())


def test_extended_publication_deadline_reuses_the_same_pending_renewal(monkeypatch) -> None:
    async def scenario() -> None:
        lease = _ControlledLease(extendable=True)
        lease.deadline = 0.02
        reprobed = asyncio.Event()
        original = heartbeat_module.await_shielded_task_outcome
        waits = 0

        async def observed_wait(task, **kwargs):
            nonlocal waits
            waits += 1
            if waits == 2:
                assert kwargs["timeout_s"] == 60
                reprobed.set()
            return await original(task, **kwargs)

        monkeypatch.setattr(heartbeat_module, "await_shielded_task_outcome", observed_wait)
        async with _running_heartbeat(lease, clock=lambda: 0.0) as (operation, stop, lost):
            await asyncio.wait_for(lease.entered.wait(), timeout=1)
            renewal = lease.renewal
            lease.deadline = 60
            await asyncio.wait_for(reprobed.wait(), timeout=1)
            assert not lost.done()
            assert lease.renewal is renewal
            stop.set()
            lease.release.set()
            await operation
            assert lease.acknowledgements == 1
            assert renewal is not None and renewal.done()

    asyncio.run(scenario())


@pytest.mark.parametrize("prior_loss", [False, True])
def test_expiry_precedes_an_idle_stop_and_keeps_the_first_loss(prior_loss: bool) -> None:
    async def scenario() -> None:
        lease = _ControlledLease()
        lease.deadline = -1
        lost = asyncio.get_running_loop().create_future()
        first_failure = RuntimeError("earlier ownership loss")
        if prior_loss:
            lost.set_result(first_failure)
        stop = asyncio.Event()
        stop.set()
        with pytest.raises(RuntimeError, match="lease expired before renewal") as captured:
            await run_leased_adapter_heartbeat(
                lease, stop=stop, ownership_lost=lost, clock=lambda: 0.0
            )
        assert lost.result() is (first_failure if prior_loss else captured.value)
        assert lease.renewal is None

    asyncio.run(scenario())


def test_repeated_cancellation_during_expired_renewal_settlement_restores_requests() -> None:
    async def scenario() -> None:
        lease = _ControlledLease()
        lease.deadline = 0.02
        async with _running_heartbeat(lease, clock=lambda: 0.0) as (operation, _stop, lost):
            await asyncio.wait_for(lease.entered.wait(), timeout=1)
            failure = await asyncio.wait_for(asyncio.shield(lost), timeout=1)
            operation.cancel("first caller request")
            await asyncio.sleep(0)
            operation.cancel("second caller request")
            await asyncio.sleep(0)
            assert not operation.done()
            assert not lease.cancelled.is_set()
            lease.release.set()
            with pytest.raises(asyncio.CancelledError) as captured:
                await operation
            assert operation.cancelling() == 2
            assert captured.value.args == ("safe shutdown",)
            assert captured.value.__cause__ is failure
            assert lease.renewal is not None and lease.renewal.done()

    asyncio.run(scenario())
