"""Host lifetime/composition tests; native owner qualification is separate."""

import asyncio
from contextlib import suppress
from dataclasses import replace

import pytest

from cayu.collaboration._contracts import OwnerRef
from cayu.collaboration._host import CollaborationHost, _HostRegistration, _ProducerSource
from cayu.collaboration._host_ownership import HostOwnershipLimits
from cayu.collaboration._producer_recovery import ProducerPendingPage
from cayu.collaboration.access import CollaborationAccessContext
from cayu.collaboration.participants import ParticipantRef
from cayu.vaults.redaction import SecretRedactor


def registration(*participants):
    owner = OwnerRef(application_scope="host", owner_id="owner", incarnation="one")
    return _HostRegistration(
        limits=HostOwnershipLimits(1, 1, 2, 256 * 1024),
        producer_sources=tuple(
            _ProducerSource(
                ParticipantRef(owner=owner, participant_id=name, incarnation="one"),
                CollaborationAccessContext(principal="operator"),
            )
            for name in participants or ("participant",)
        ),
        producer_rules=(),
        observation_timeout_s=0.01,
        shutdown_timeout_s=0.01,
    )


class App:
    _secret_redactor = SecretRedactor()

    def __init__(self):
        self.calls = []

    async def pending_producer_outputs(self, participant, **kwargs):
        self.calls.append(participant.participant_id)
        return ProducerPendingPage(items=(), next_cursor=None)

    async def aclose(self):
        pytest.fail("Host must not close its application")


@pytest.fixture(autouse=True)
def native_discovery_double(monkeypatch):
    async def discover(app, participant, **kwargs):
        assert kwargs.pop("wait_for_settlement") is True
        return await app.pending_producer_outputs(participant, **kwargs)

    monkeypatch.setattr("cayu.collaboration._host_discovery.pending_producer_outputs", discover)


def test_inert_construction_and_explicit_service_never_claim_global_coverage():
    app = App()
    host = CollaborationHost(app, registration())
    assert app.calls == []
    assert not host.inspect().servicing_pending

    async def scenario():
        async with host:
            observed = await host.service_once()
            assert not observed.coverage_complete
            assert observed.serviced == 0
        assert host.inspect().closing

    asyncio.run(scenario())
    assert app.calls == ["participant"]


def test_discovery_configuration_cannot_guarantee_permanent_capacity_refusal():
    app = App()
    with pytest.raises(ValueError, match="complete query"):
        CollaborationHost(app, replace(registration(), discovery_bytes=65536))
    assert app.calls == []


def test_failed_source_does_not_starve_another_participant():
    async def scenario():
        class FailingSource(App):
            async def pending_producer_outputs(self, participant, **kwargs):
                if participant.participant_id == "bad":
                    self.calls.append("bad")
                    raise RuntimeError("source unavailable")
                return await super().pending_producer_outputs(participant, **kwargs)

        app = FailingSource()
        host = CollaborationHost(app, registration("bad", "good"))
        await host.service_once()
        # A short observation may end between sources. Continue the same owned
        # pass before starting another one; never restart on observation timeout.
        while host.inspect().servicing_pending:
            await host.service_once()
        assert app.calls == ["bad", "good"]
        assert host.inspect().source_failures == 1
        await host.aclose()

    asyncio.run(scenario())


def test_empty_producer_scan_preserves_other_families_blocked_count(monkeypatch):
    async def scenario():
        host = CollaborationHost(App(), replace(registration(), observation_timeout_s=1))

        async def deferred_continuations(deadline, stop):
            host._observed_blocked += 2

        monkeypatch.setattr(host, "_service_continuations", deferred_continuations)
        async with host:
            for _ in range(3):
                observed = await host.service_once()
                assert observed.observed_blocked == 2
                assert observed.active == observed.uncertain == 0

    asyncio.run(scenario())


def test_every_family_gets_first_access_to_one_maintenance_slot(monkeypatch):
    from cayu.collaboration._host_ownership import HostCapacityExceeded, HostOperationIdentity
    from cayu.collaboration._host_producer_maintenance import HostMaintenanceResult

    async def scenario():
        host = CollaborationHost(App(), registration())
        families = (
            "_service_request_maintenance",
            "_service_clarification_maintenance",
            "_service_clarifications",
            "_service_continuations",
            "_service_waits",
            "_service_plans",
            "_service_registrations",
            "_service_planned_producers",
            "_service_producers",
            "_service_producer_maintenance",
        )
        started = []

        def service_family(name):
            async def service(deadline, stop):
                async def action(_stop):
                    started.append(name)
                    return HostMaintenanceResult(False)

                with suppress(HostCapacityExceeded):
                    host._owned.start(
                        HostOperationIdentity(name, "a" * 64),
                        role="maintenance",
                        reserved_bytes=65536,
                        action=action,
                    )

            return service

        for name in families:
            monkeypatch.setattr(host, name, service_family(name))
        # Every family is perpetually eligible. A settled no-op still occupies
        # the sole slot until observed; reversing two orders is insufficient.
        for _ in families:
            await host.service_once()
            while host.inspect().servicing_pending:
                await host.service_once()
        await host.aclose()
        assert started == list(families)
        assert host.inspect().uncertain == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("busy_passes", [3, 12, 21])
def test_busy_passes_do_not_erase_the_next_maintenance_family_turn(monkeypatch, busy_passes):
    from cayu.collaboration._host_ownership import HostCapacityExceeded, HostOperationIdentity
    from cayu.collaboration._host_producer_maintenance import HostMaintenanceResult

    async def scenario():
        host = CollaborationHost(App(), replace(registration(), observation_timeout_s=1))
        release = asyncio.Event()
        entered = asyncio.Event()
        started = []

        def family(name):
            async def service(deadline, stop):
                async def action(_stop):
                    started.append(name)
                    if name == "wait":
                        entered.set()
                        await release.wait()
                    return HostMaintenanceResult(False)

                with suppress(HostCapacityExceeded):
                    host._owned.start(
                        HostOperationIdentity(name, "a" * 64),
                        role="maintenance",
                        reserved_bytes=65536,
                        action=action,
                    )

            return service

        monkeypatch.setattr(host, "_service_waits", family("wait"))
        monkeypatch.setattr(host, "_service_producers", family("producer"))
        try:
            await host.service_once()
            await asyncio.wait_for(entered.wait(), 10)
            assert started == ["wait"]
            # Poll the occupied slot across the old order's wraparound. These
            # passes did not give the producer any opportunity to run.
            for _ in range(busy_passes):
                await host.service_once()
            release.set()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            await host.service_once()
            await asyncio.sleep(0)
            assert started[:2] == ["wait", "producer"]
        finally:
            release.set()
            await host.aclose()

    asyncio.run(scenario())


def test_real_cancellation_and_bounded_close_retain_discovery_until_it_returns():
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        blocked_calls = 0

        class BlockedSource(App):
            async def pending_producer_outputs(self, participant, **kwargs):
                nonlocal blocked_calls
                if participant.participant_id == "blocked":
                    blocked_calls += 1
                    entered.set()
                    await release.wait()
                return await super().pending_producer_outputs(participant, **kwargs)

        app = BlockedSource()
        host = CollaborationHost(app, registration("blocked", "good"))
        observer = asyncio.create_task(host.service_once())
        await entered.wait()
        observer.cancel()
        assert observer.cancelling() == 1
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert observer.cancelled()
        assert host.inspect().discovery_pending >= 1
        async with asyncio.timeout(1):
            while "good" not in app.calls:
                await host.service_once()
        for _ in range(3):
            await host.service_once()
        assert blocked_calls == 1
        assert "blocked" not in app.calls
        closed = await host.aclose()
        assert not closed.servicing_pending
        assert closed.discovery_pending == 1
        release.set()
        async with asyncio.timeout(1):
            while (await host.aclose()).discovery_pending:
                await asyncio.sleep(0)
        assert app.calls.count("blocked") == 1

    asyncio.run(scenario())


def test_context_body_cancellation_remains_cancellation_when_shutdown_fails():
    async def scenario():
        host = CollaborationHost(App(), registration())
        entered = asyncio.Event()
        cleanup = RuntimeError("cleanup failure")

        async def close():
            raise cleanup

        host.aclose = close

        async def run():
            async with host:
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(run())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert task.cancelled()
        assert task.cancelling() == 1
        assert caught.value.__cause__ is cleanup
        assert cleanup.__context__ is None

    asyncio.run(scenario())


def test_close_deadline_retains_completed_but_unobserved_handoff(monkeypatch):
    from cayu.collaboration._host_ownership import HostOperationIdentity
    from cayu.collaboration._host_producer_maintenance import HostMaintenanceResult

    async def scenario():
        host = CollaborationHost(App(), registration())
        ready = asyncio.Event()

        async def action(stop):
            ready.set()
            return HostMaintenanceResult(False)

        host._owned.start(
            HostOperationIdentity("completed", "a" * 64),
            role="maintenance",
            reserved_bytes=65536,
            action=action,
        )
        await ready.wait()
        original_close = host._lifecycle.aclose

        async def delay_join(**kwargs):
            result = await original_close(**kwargs)
            await asyncio.sleep(2 * host._registration.shutdown_timeout_s)
            return result

        monkeypatch.setattr(host._lifecycle, "aclose", delay_join)
        observed = await host.aclose()
        assert observed.uncertain == 1 and observed.failed == 0
        monkeypatch.setattr(host._lifecycle, "aclose", original_close)
        assert not (await host.aclose()).pending

    asyncio.run(scenario())


def test_context_body_and_shutdown_errors_keep_ordered_originals():
    async def scenario():
        host = CollaborationHost(App(), registration())
        primary = ValueError("body failure")
        cleanup = RuntimeError("cleanup failure")

        async def close():
            raise cleanup

        host.aclose = close
        with pytest.raises(ExceptionGroup) as caught:
            async with host:
                raise primary
        assert caught.value.exceptions == (primary, cleanup)
        assert cleanup.__context__ is None

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_read", [False, True])
@pytest.mark.parametrize("effect_failure", [False, True])
def test_public_close_propagates_retained_read_failures_once(cancel_read, effect_failure):
    from cayu.collaboration._host_ownership import HostOperationIdentity, HostReconciledResult
    from cayu.collaboration._host_producer_maintenance import HostMaintenanceResult

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        primary = OSError("effect acknowledgement failure")
        read_failure = ExceptionGroup("read cleanup", [ValueError("first"), OSError("second")])
        read_task = None

        class BlockedSource(App):
            async def pending_producer_outputs(self, participant, **kwargs):
                nonlocal read_task
                self.calls.append(participant.participant_id)
                read_task = asyncio.current_task()
                entered.set()
                await release.wait()
                raise read_failure

        app = BlockedSource()
        host = CollaborationHost(app, registration())
        async with asyncio.timeout(5):
            await host.service_once()
            await entered.wait()
            while host.inspect().servicing_pending:
                await host.service_once()
            assert host.inspect().discovery_pending == 1
            if effect_failure:

                async def settled(stop):
                    return HostReconciledResult(HostMaintenanceResult(False), primary)

                host._owned.start(
                    HostOperationIdentity("settled", "a" * 64),
                    role="maintenance",
                    reserved_bytes=65536,
                    action=settled,
                )
            if cancel_read:
                read_task.cancel()
                read_task.cancel()
            else:
                release.set()
            # Public close owns delivery of the read's original failure. The
            # ordinary cancellation handler must work, including task state.
            closer = asyncio.create_task(host.aclose())
            if cancel_read:
                with pytest.raises(asyncio.CancelledError) as caught:
                    await closer
                assert closer.cancelled()
                assert read_task.cancelling() == 2
                if effect_failure:
                    assert caught.value.__cause__ is primary
            else:
                with pytest.raises(ExceptionGroup) as caught:
                    await closer
                if effect_failure:
                    assert caught.value.exceptions == (primary, read_failure)
                else:
                    assert caught.value is read_failure
            assert not host.inspect().pending
            assert host.inspect().source_failures == 1
            assert host.inspect().failed == 0
            assert not (await host.aclose()).pending
            assert app.calls == ["participant"]

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_body", [False, True])
def test_context_exit_preserves_body_and_real_discovery_cleanup(cancel_body):
    async def scenario():
        entered, release, body_ready = asyncio.Event(), asyncio.Event(), asyncio.Event()
        primary = ValueError("body failure")
        cleanup = OSError("source read failed")

        class BlockedSource(App):
            async def pending_producer_outputs(self, participant, **kwargs):
                entered.set()
                await release.wait()
                raise cleanup

        host = CollaborationHost(BlockedSource(), registration())

        async def run():
            async with host:
                await host.service_once()
                await entered.wait()
                while host.inspect().servicing_pending:
                    await host.service_once()
                body_ready.set()
                try:
                    if cancel_body:
                        await asyncio.Event().wait()
                    raise primary
                finally:
                    release.set()

        async with asyncio.timeout(5):
            task = asyncio.create_task(run())
            await body_ready.wait()
            if cancel_body:
                task.cancel()
                task.cancel()
                with pytest.raises(asyncio.CancelledError) as caught:
                    await task
                assert task.cancelled() and task.cancelling() == 2
                assert caught.value.__cause__ is cleanup
            else:
                with pytest.raises(ExceptionGroup) as caught:
                    await task
                assert caught.value.exceptions == (primary, cleanup)
            assert not (await host.aclose()).pending

    asyncio.run(scenario())


def test_close_preserves_all_discovery_pools_and_reports_each_failure_once():
    async def scenario():
        host = CollaborationHost(App(), replace(registration(), shutdown_timeout_s=0.01))
        release = asyncio.Event()
        failures = (
            OSError("ordinary read"),
            RuntimeError("maintenance read"),
            LookupError("clarification maintenance read"),
            ValueError("producer maintenance read"),
        )

        async def blocked(error):
            await release.wait()
            raise error

        pools = (
            host._reads,
            host._maintenance_reads,
            host._clarification_maintenance_reads,
            host._producer_maintenance_reads,
        )
        for pool, error in zip(pools, failures, strict=True):
            await pool.observe(
                "blocked",
                expectation=b"exact",
                reserved_bytes=131072,
                read=lambda error=error: blocked(error),
            )
        assert host.inspect().discovery_pending == 4
        assert (await host.aclose()).discovery_pending == 4
        release.set()
        for error in failures:
            with pytest.raises(type(error)) as caught:
                await host.aclose()
            assert caught.value is error
        assert not (await host.aclose()).pending
        assert host.inspect().source_failures == 4

    asyncio.run(scenario())


def test_cancelled_public_close_retains_read_until_later_failure_observation(monkeypatch):
    async def scenario():
        entered, release, closing = asyncio.Event(), asyncio.Event(), asyncio.Event()
        failure = OSError("late source failure")

        class BlockedSource(App):
            async def pending_producer_outputs(self, participant, **kwargs):
                self.calls.append(participant.participant_id)
                entered.set()
                await release.wait()
                raise failure

        app = BlockedSource()
        host = CollaborationHost(app, replace(registration(), shutdown_timeout_s=1))
        close_reads = host._reads.close

        async def observe_close(timeout):
            closing.set()
            return await close_reads(timeout)

        monkeypatch.setattr(host._reads, "close", observe_close)
        async with asyncio.timeout(5):
            while not entered.is_set():
                await host.service_once()
            while host.inspect().servicing_pending:
                await host.service_once()
            closer = asyncio.create_task(host.aclose())
            await closing.wait()
            closer.cancel()
            closer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closer
            assert closer.cancelled() and closer.cancelling() == 2
            assert host.inspect().discovery_pending == 1
            assert host.inspect().source_failures == 0
            release.set()
            with pytest.raises(OSError) as caught:
                await host.aclose()
            assert caught.value is failure
            assert not (await host.aclose()).pending
            assert app.calls == ["participant"]

    asyncio.run(scenario())
