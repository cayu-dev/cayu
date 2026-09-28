"""Local supervision only; native host integration is qualified separately."""

import asyncio

import pytest

from cayu.collaboration._host_ownership import (
    HostOperationIdentity,
    HostOwnership,
    HostOwnershipLimits,
)


def owner():
    return HostOwnership(HostOwnershipLimits(1, 1, 2, 4096))


def identity(key="work", commitment="a"):
    return HostOperationIdentity(key, commitment * 64)


def test_construction_is_inert_and_limits_are_strict():
    assert owner().pending == 0
    for values in ((True, 1, 2, 4096), (1, 0, 2, 4096), (1, 1, 1, 4096)):
        with pytest.raises(ValueError):
            HostOwnershipLimits(*values)
    for timeout in (True, 0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            asyncio.run(owner().observe(timeout))


@pytest.mark.parametrize("before_entry", [False, True])
def test_owned_cancellation_reports_once_and_reconciles_without_redispatch(before_entry):
    async def scenario():
        from cayu.collaboration._host import CollaborationHost

        owned = owner()
        entered, settled = asyncio.Event(), asyncio.Event()
        calls = []
        later = OSError("new read failure")
        reads = 0

        async def effect(stop):
            calls.append(True)
            entered.set()
            await asyncio.Event().wait()

        async def reconcile():
            nonlocal reads
            reads += 1
            if reads == 1:
                raise later
            return "exact handoff" if settled.is_set() else None

        owned.start(
            identity(), role="execution", reserved_bytes=1024, action=effect, reconcile=reconcile
        )
        task = owned._operations[identity().key].task
        if not before_entry:
            await entered.wait()
        task.cancel()
        task.cancel()
        await owned.observe(1)
        signals = owned.take_control_failures()
        assert len(signals) == 1 and isinstance(signals[0], asyncio.CancelledError)
        assert task.cancelling() == 2
        assert task.cancelled()
        assert not owned.take_control_failures()

        async def report():
            CollaborationHost._raise_reconciled_failures(signals)

        observer = asyncio.create_task(report())
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert observer.cancelled() and observer.cancelling() == 0
        assert not owned.has_slot("execution")
        sibling = identity("independent-maintenance")

        async def maintenance(stop):
            return "independent settlement"

        owned.start(sibling, role="maintenance", reserved_bytes=1024, action=maintenance)
        await owned.observe(1)
        await owned.observe(1)
        assert any(
            outcome.identity == sibling and outcome.value == "independent settlement"
            for outcome in owned.inspect().completed
        )
        owned.release_settled(sibling)
        failures = owned.take_reconciliation_failures()
        assert failures == [later]
        # A historical signal is not a new cancellation of this observer.
        with pytest.raises(OSError) as caught:
            CollaborationHost._raise_reconciled_failures(failures)
        assert caught.value is later
        assert later.__cause__ is signals[0]
        await owned.observe(1)
        assert not owned.inspect().completed[0].reconciled
        settled.set()
        await owned.observe(1)
        assert owned.inspect().completed[0].reconciled
        owned.release_settled(identity())
        assert owned.has_slot("execution")
        assert calls == ([] if before_entry else [True])

    asyncio.run(scenario())


def test_failed_turn_retains_one_read_and_reports_each_recovery_failure():
    async def scenario():
        owned = owner()
        original = ExceptionGroup("effect", [OSError("dispatch"), RuntimeError("cleanup")])
        later = ExceptionGroup("read", [ConnectionError("read"), OSError("read cleanup")])
        reads, dispatches = [], []
        entered, release = asyncio.Event(), asyncio.Event()

        async def effect(stop):
            dispatches.append(True)
            raise original

        async def reconcile():
            reads.append(True)
            if len(reads) == 1:
                entered.set()
                await release.wait()
                raise later
            return None if len(reads) == 2 else "exact handoff"

        owned.start(
            identity(), role="maintenance", reserved_bytes=1024, action=effect, reconcile=reconcile
        )
        assert (await owned.observe(1)).completed[0].error is original
        observer = asyncio.create_task(owned.observe(10))
        await entered.wait()
        observer.cancel()
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert observer.cancelled() and observer.cancelling() == 2
        for _ in range(3):
            await owned.observe(0.001)
        assert reads == dispatches == [True]
        assert owned.pending == 1 and not owned.has_slot("maintenance")
        release.set()
        await owned.observe(1)
        failures = owned.take_reconciliation_failures()
        assert len(failures) == 1 and failures[0].exceptions == (original, later)
        assert not owned.take_reconciliation_failures()
        await owned.observe(1)
        assert owned.inspect().completed[0].error is original
        assert owned.pending == 1  # An unavailable read is not exclusion.
        await owned.close(1)
        outcome = owned.inspect().completed[0]
        assert outcome.reconciled and outcome.value == "exact handoff"
        assert outcome.error is original
        assert len(reads) == 3 and dispatches == [True]
        owned.release_settled(identity())
        assert owned.pending == 0

    asyncio.run(scenario())


def test_cancelled_reconciliation_keeps_normal_signal_and_original_failure():
    async def scenario():
        from cayu.collaboration._host import CollaborationHost

        owned = owner()
        original = OSError("original effect")
        entered = asyncio.Event()

        async def effect(stop):
            raise original

        async def reconcile():
            entered.set()
            await asyncio.Event().wait()

        owned.start(
            identity(), role="maintenance", reserved_bytes=1024, action=effect, reconcile=reconcile
        )
        await owned.observe(1)
        await owned.observe(0.001)
        await entered.wait()
        read = owned._operations[identity().key].reconciliation
        read.cancel()
        assert read.cancelling() == 1
        await owned.observe(1)
        failures = owned.take_reconciliation_failures()
        assert len(failures) == 1 and isinstance(failures[0], asyncio.CancelledError)
        assert failures[0].__cause__ is original

        async def collect():
            CollaborationHost._raise_reconciled_failures(failures)

        observer = asyncio.create_task(collect())
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert observer.cancelled()
        assert owned.pending == 1 and not owned.has_slot("maintenance")
        assert owned.inspect().completed[0].error is original

    asyncio.run(scenario())


@pytest.mark.parametrize("close", [False, True])
def test_prepared_work_waits_for_explicit_window_or_shutdown(close):
    async def scenario():
        owned = owner()
        entered = asyncio.Event()
        expired = asyncio.get_running_loop().time() - 1

        async def prepared(stop):
            entered.set()
            return await owned.wait_for_dispatch_window(identity(), initial_deadline=expired)

        owned.start(identity(), role="execution", reserved_bytes=1, action=prepared)
        await entered.wait()
        assert not (await owned.observe(0.01)).completed
        assert not owned.has_slot("execution")
        if close:
            owned.request_close()
            # Later scheduling cannot undo close or authorize deferred work.
            owned.renew_dispatch_windows(asyncio.get_running_loop().time() + 10)
        else:
            owned.renew_dispatch_windows(asyncio.get_running_loop().time() + 10)
        result = await owned.observe(1)
        assert len(result.completed) == 1
        assert result.completed[0].error is None
        assert result.completed[0].value is (not close)
        owned.release_settled(identity())
        assert not owned.pending

    asyncio.run(scenario())


def test_bounded_pass_retains_work_and_maintenance_capacity():
    async def scenario():
        owned = owner()
        entered = asyncio.Event()
        release = asyncio.Event()
        launches = []

        async def work(stop):
            launches.append("work")
            owned.mark_active(identity())
            entered.set()
            await release.wait()
            return "completed"

        assert owned.start(identity(), role="execution", reserved_bytes=2048, action=work)
        assert not owned.has_slot("execution")
        assert owned.has_slot("maintenance")
        await entered.wait()
        for _ in range(3):
            observed = await owned.observe(0.001)
            assert observed.active == (identity(),)
            assert not observed.completed
            assert not owned.start(identity(), role="execution", reserved_bytes=2048, action=work)
        with pytest.raises(RuntimeError, match="capacity"):
            owned.start(identity("other"), role="execution", reserved_bytes=1, action=work)

        async def maintenance(stop):
            return "cleanup observed"

        cleanup = identity("cleanup")
        owned.start(cleanup, role="maintenance", reserved_bytes=2048, action=maintenance)
        observed = await owned.observe(1)
        assert observed.completed[0].value == "cleanup observed"
        assert owned.pending == 2  # returning is not settlement evidence
        assert not owned.has_slot("maintenance")
        owned.release_settled(cleanup)
        assert owned.has_slot("maintenance")
        release.set()
        observed = await owned.observe(1)
        assert observed.completed[0].value == "completed"
        assert not owned.has_slot("execution")
        assert launches == ["work"]
        owned.release_settled(identity())
        assert owned.pending == 0
        assert owned.has_slot("execution")

    asyncio.run(scenario())


def test_real_repeated_observer_cancellation_preserves_owned_work():
    async def scenario():
        owned = owner()
        entered = asyncio.Event()
        release = asyncio.Event()
        stop_seen = []

        async def work(stop):
            entered.set()
            await release.wait()
            stop_seen.append(stop.is_set())

        owned.start(identity(), role="execution", reserved_bytes=1, action=work)
        await entered.wait()
        for _ in range(2):
            observer = asyncio.create_task(owned.observe(10))
            await asyncio.sleep(0)
            observer.cancel()
            assert observer.cancelling() == 1
            try:
                await observer
            except asyncio.CancelledError:
                pass
            else:
                pytest.fail("Caller cancellation was swallowed")
            assert observer.cancelled()
            assert owned.pending == 1
            assert owned.inspect().uncertain == (identity(),)
        release.set()
        await owned.observe(1)
        assert stop_seen == [False]
        owned.release_settled(identity())

    asyncio.run(scenario())


def test_shutdown_is_bounded_and_does_not_discard_opaque_work():
    async def scenario():
        owned = owner()
        entered = asyncio.Event()
        release = asyncio.Event()
        stop_seen = asyncio.Event()

        async def work(stop):
            entered.set()
            await stop.wait()
            stop_seen.set()
            await release.wait()

        owned.start(identity(), role="execution", reserved_bytes=1, action=work)
        await entered.wait()
        result = await owned.close(0.001)
        assert stop_seen.is_set()
        assert result.uncertain == (identity(),)
        assert owned.pending == 1
        with pytest.raises(RuntimeError, match="closing"):
            owned.start(identity("new"), role="execution", reserved_bytes=1, action=work)
        assert (await owned.close(0.001)).uncertain == (identity(),)
        release.set()
        assert len((await owned.close(1)).completed) == 1
        assert owned.pending == 1
        owned.release_settled(identity())
        assert not (await owned.close(1)).uncertain

    asyncio.run(scenario())


def test_error_graph_retained_exactly_once_without_implicit_settlement():
    async def scenario():
        owned = owner()
        primary = ValueError("primary")
        cleanup = OSError("cleanup")
        error = ExceptionGroup("ordered", [primary, ExceptionGroup("nested", [cleanup])])

        async def work(stop):
            raise error

        owned.start(identity(), role="execution", reserved_bytes=1, action=work)
        first = await owned.observe(1)
        assert first.completed[0].error is error
        assert first.completed[0].error.exceptions[0] is primary
        assert not (await owned.observe(0.001)).completed
        assert owned.pending == 1
        owned.release_settled(identity())

    asyncio.run(scenario())


def test_conflicting_intent_and_overlapping_observation_are_rejected():
    async def scenario():
        owned = owner()
        release = asyncio.Event()

        async def work(stop):
            await release.wait()

        owned.start(identity(), role="execution", reserved_bytes=1, action=work)
        with pytest.raises(ValueError, match="conflicts"):
            owned.start(identity(commitment="b"), role="execution", reserved_bytes=1, action=work)
        first = asyncio.create_task(owned.observe(10))
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="already active"):
            await owned.observe(1)
        with pytest.raises(ValueError, match="observed exact"):
            owned.release_settled(identity())
        release.set()
        await first
        owned.release_settled(identity())

    asyncio.run(scenario())
