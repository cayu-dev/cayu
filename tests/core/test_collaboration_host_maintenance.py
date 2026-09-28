"""Deadline/stop behavior around native producer-owner readback."""

import asyncio
from dataclasses import replace

import pytest

from cayu.collaboration._contracts import OperationRef
from cayu.collaboration._host_ownership import HostOwnership, HostOwnershipLimits
from cayu.collaboration._host_producer_maintenance import (
    HostMaintenanceResult,
    HostProducerMaintenance,
    service_producer_maintenance,
)
from cayu.collaboration._host_producer_tasks import (
    acknowledge_producer_maintenance,
    start_producer_maintenance,
)
from cayu.collaboration._producer_recovery import ProducerOutputRecovery
from cayu.collaboration.access import CollaborationAccessContext
from cayu.vaults.redaction import SecretRedactor


def intent():
    return HostProducerMaintenance(
        recovery=ProducerOutputRecovery(
            registration=OperationRef(
                application_scope="host",
                namespace_incarnation="namespace",
                generation=1,
                caller_key="producer",
            ),
            registration_commitment="sha256:" + "a" * 64,
        ),
        action="retain_completion",
    )


def test_stopped_or_expired_turn_never_calls_an_owner():
    class App:
        _secret_redactor = SecretRedactor()

        async def lookup_producer_registration(self, *args, **kwargs):
            pytest.fail("Stopped host turn called an owner")

    async def scenario():
        for expired in (False, True):
            stop = asyncio.Event()
            if not expired:
                stop.set()
            result = await service_producer_maintenance(
                App(),
                intent(),
                context=CollaborationAccessContext(principal="host"),
                disclosure_context=None,
                observation_deadline=asyncio.get_running_loop().time() + (-1 if expired else 60),
                stop=stop,
            )
            assert isinstance(result, HostMaintenanceResult)
            assert not result.dispatched
            assert result.receipt is None

    asyncio.run(scenario())


def test_cancellation_during_readback_remains_cancellation(monkeypatch):
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()

        class App:
            _secret_redactor = SecretRedactor()

        async def lookup(*args, **kwargs):
            assert kwargs["wait_for_settlement"] is True
            entered.set()
            await release.wait()
            pytest.fail("Cancelled readback resumed")

        monkeypatch.setattr(
            "cayu.collaboration._host_producer_maintenance.lookup_producer_registration", lookup
        )

        observer = asyncio.create_task(
            service_producer_maintenance(
                App(),
                intent(),
                context=CollaborationAccessContext(principal="host"),
                disclosure_context=None,
                observation_deadline=asyncio.get_running_loop().time() + 60,
                stop=asyncio.Event(),
            )
        )
        await entered.wait()
        observer.cancel()
        assert observer.cancelling() == 1
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert observer.cancelled()

    asyncio.run(scenario())


def test_retained_native_turn_survives_cancelled_observers_without_restarting(monkeypatch):
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        failure = RuntimeError("native owner unavailable")
        calls = []

        class App:
            _secret_redactor = SecretRedactor()

        async def lookup(*args, **kwargs):
            assert kwargs["wait_for_settlement"] is True
            calls.append("read")
            entered.set()
            await release.wait()
            raise failure

        monkeypatch.setattr(
            "cayu.collaboration._host_producer_maintenance.lookup_producer_registration", lookup
        )

        app = App()
        owned = HostOwnership(HostOwnershipLimits(1, 1, 2, 256 * 1024))
        deadline = asyncio.get_running_loop().time() + 60
        context = CollaborationAccessContext(principal="host")

        def start():
            return start_producer_maintenance(
                app,
                owned,
                intent(),
                context=context,
                disclosure_context=None,
                observation_deadline=deadline,
            )

        identity = start()
        await entered.wait()
        for _ in range(2):
            observer = asyncio.create_task(owned.observe(10))
            await asyncio.sleep(0)
            observer.cancel()
            assert observer.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await observer
            assert observer.cancelled()
            assert start() == identity
        assert calls == ["read"]
        release.set()
        outcome = (await owned.observe(1)).completed[0]
        assert outcome.error is failure
        assert outcome.reconciled and not outcome.value.dispatched
        assert not acknowledge_producer_maintenance(owned, outcome)
        assert owned.pending == 1
        assert owned.inspect().uncertain == (identity,)
        assert acknowledge_producer_maintenance(owned, replace(outcome, error=None))
        assert owned.pending == 0

    asyncio.run(scenario())


def test_undispatched_turn_releases_local_slot_only_after_owned_observation():
    async def scenario():
        class App:
            _secret_redactor = SecretRedactor()

        app = App()
        owned = HostOwnership(HostOwnershipLimits(1, 1, 2, 256 * 1024))
        start_producer_maintenance(
            app,
            owned,
            intent(),
            context=CollaborationAccessContext(principal="host"),
            disclosure_context=None,
            observation_deadline=asyncio.get_running_loop().time() - 1,
        )
        outcome = (await owned.observe(1)).completed[0]
        assert not outcome.value.dispatched
        assert owned.pending == 1
        assert acknowledge_producer_maintenance(owned, outcome)
        assert owned.pending == 0

    asyncio.run(scenario())
