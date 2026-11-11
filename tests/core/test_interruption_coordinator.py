import asyncio
import contextlib
from contextvars import Context

import pytest

from cayu.runtime import _interruption_coordinator as interruption_coordinator
from cayu.runtime._foreground_child_delivery import (
    ForegroundChildDeliveryOwner,
    ForegroundChildDeliverySealed,
)
from cayu.sessions.event_delivery import PersistedEventSideEffectClaimLost


def test_interruption_cascade_suppression_restores_same_context() -> None:
    context = Context()

    def exercise() -> tuple[bool, bool, bool]:
        before = interruption_coordinator.interruption_cascade_suppressed()
        with interruption_coordinator.suppress_interruption_cascade():
            during = interruption_coordinator.interruption_cascade_suppressed()
        after = interruption_coordinator.interruption_cascade_suppressed()
        return before, during, after

    assert context.run(exercise) == (False, True, False)


def test_cross_context_suppression_close_does_not_overwrite_closer() -> None:
    creator = Context()
    closer = Context()
    scope = interruption_coordinator.suppress_interruption_cascade()

    creator.run(scope.__enter__)
    closer.run(interruption_coordinator._SUPPRESS_BACKGROUND_INTERRUPTION_CASCADE.set, True)
    closer.run(scope.__exit__, None, None, None)

    assert closer.run(interruption_coordinator.interruption_cascade_suppressed) is True


def _bare_coordinator() -> interruption_coordinator.BackgroundInterruptionCoordinator:
    async def unused(*_args, **_kwargs):  # pragma: no cover - never reached
        raise AssertionError("unused")

    return interruption_coordinator.BackgroundInterruptionCoordinator(
        session_control=None,  # ty: ignore[invalid-argument-type]
        session_store=None,  # ty: ignore[invalid-argument-type]
        event_writer=None,  # ty: ignore[invalid-argument-type]
        clock=None,  # ty: ignore[invalid-argument-type]
        interrupt_session=unused,  # ty: ignore[invalid-argument-type]
    )


def test_cancelled_drain_keeps_the_callers_cancellation_request() -> None:

    async def scenario() -> None:
        coordinator = _bare_coordinator()

        async def never_finishes() -> None:
            await asyncio.Event().wait()

        cascade = asyncio.create_task(never_finishes())
        coordinator._tasks.add(cascade)
        observed: list[int] = []
        started = asyncio.Event()

        async def drain_and_observe() -> None:
            started.set()
            try:
                await coordinator.drain(timeout_s=30)
            except asyncio.CancelledError:
                current = asyncio.current_task()
                assert current is not None
                observed.append(current.cancelling())
                raise

        drainer = asyncio.create_task(drain_and_observe())
        await started.wait()
        await asyncio.sleep(0)
        drainer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await drainer
        # The external Task.cancel() is still the drainer's own cancellation.
        assert observed == [1]
        assert drainer.cancelled()
        # Cleanup still cancelled the in-memory cascade work.
        assert cascade.cancelled() or cascade.cancelling()

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_worker", [False, True])
def test_abandoned_interruption_node_invalidates_its_cascade(monkeypatch, cancel_worker) -> None:
    async def scenario() -> None:
        coordinator = _bare_coordinator()
        state = interruption_coordinator._BackgroundInterruptionCascadeState(
            parent_session_id="parent",
            attempt_id="attempt",
            generation=1,
            claim_id="claim",
            claim_deadline_monotonic=0.0,
            reason=None,
            metadata={},
            requested_by=None,
        )
        coordinator._enqueue_background_interruption_node(state, session_id="parent", session=None)
        if cancel_worker:
            started = asyncio.Event()

            async def blocked_node(_node) -> None:
                started.set()
                await asyncio.Event().wait()

            monkeypatch.setattr(coordinator, "_process_background_interruption_node", blocked_node)
            worker = asyncio.create_task(
                coordinator._background_interruption_worker(asyncio.Event())
            )
            await started.wait()
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker
        else:
            coordinator._discard_background_interruption_queue()

        assert state.outstanding == 0
        assert state.done.is_set()
        assert state.claim_lost.is_set()
        await coordinator._queue.join()

    asyncio.run(scenario())


@pytest.mark.parametrize("action", ["complete", "timeout", "cancel"])
def test_direct_cascade_drain_waits_without_polling_through_claim_release(
    monkeypatch, action
) -> None:
    async def scenario() -> None:
        coordinator = _bare_coordinator()
        started = asyncio.Event()
        finish_traversal = asyncio.Event()
        release_started = asyncio.Event()
        finish_release = asyncio.Event()
        finish_caller = asyncio.Event()
        original_sleep = asyncio.sleep
        zero_yields = 0

        async def claim(*_args, **_kwargs):
            return {"attempt_id": "attempt", "generation": 1, "claim_id": "claim"}

        async def traverse(state):
            started.set()
            if action == "complete":
                await finish_traversal.wait()
            else:
                await state.claim_lost.wait()

        async def release(*_args):
            release_started.set()
            await finish_release.wait()

        async def recover():
            await coordinator.run_cascade(
                parent_session_id="parent",
                interrupt_payload={"interruption_type": "operator_requested"},
            )
            # Drain must not wait for unrelated work in the caller's task.
            await finish_caller.wait()

        async def observe_sleep(delay, *args, **kwargs):
            nonlocal zero_yields
            if asyncio.current_task() is drainer and delay == 0:
                zero_yields += 1
            await original_sleep(delay, *args, **kwargs)

        monkeypatch.setattr(coordinator, "claim_pending_interruption_cascade", claim)
        monkeypatch.setattr(coordinator, "_run_claimed_background_interruption_cascade", traverse)
        monkeypatch.setattr(coordinator, "release_pending_interruption_cascade_claim", release)
        recovery = asyncio.create_task(recover())
        await started.wait()
        drainer = asyncio.create_task(coordinator.drain(timeout_s=0.05))
        monkeypatch.setattr(asyncio, "sleep", observe_sleep)
        try:
            await original_sleep(0)
            assert zero_yields == 0
            assert not drainer.done()
            if action == "complete":
                finish_traversal.set()
            else:
                if action == "cancel":
                    drainer.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await drainer
                else:
                    assert await drainer is False
                assert coordinator.pending
                assert await coordinator.drain(timeout_s=0.01) is False
                drainer = asyncio.create_task(coordinator.drain(timeout_s=1))
            await asyncio.wait_for(release_started.wait(), 1)
            assert coordinator.pending
            assert not drainer.done()
            finish_release.set()
            assert await asyncio.wait_for(drainer, 1) is True
            assert not coordinator.pending
            assert not recovery.done()
        finally:
            finish_traversal.set()
            finish_release.set()
            finish_caller.set()
            await asyncio.gather(recovery, drainer, return_exceptions=True)

    asyncio.run(scenario())


def test_sealed_coordinator_refuses_new_cascades_but_drain_does_not_seal() -> None:

    async def scenario() -> None:
        coordinator = _bare_coordinator()
        await coordinator.drain(timeout_s=1)
        assert not coordinator.sealed
        coordinator.seal()
        assert coordinator.sealed
        assert (
            coordinator.schedule(
                parent_session_id="parent",
                interrupt_payload={},
                create_if_missing=False,
            )
            is None
        )
        coordinator.defer(
            parent_session_id="parent",
            interrupt_payload={},
            retry_after_seconds=0.0,
            drain_required=False,
            retry_request=None,
        )
        assert not coordinator.pending

    asyncio.run(scenario())


def test_foreground_delivery_drain_waits_for_work_started_during_the_drain() -> None:

    async def scenario() -> None:
        owner = ForegroundChildDeliveryOwner(store=None)  # ty: ignore[invalid-argument-type]
        first_release = asyncio.Event()
        second_release = asyncio.Event()

        async def worker(release: asyncio.Event):
            await release.wait()

        first = asyncio.create_task(worker(first_release))
        owner._workers["first"] = first
        drain = asyncio.create_task(owner.drain(timeout_s=5))
        await asyncio.sleep(0)
        # Settling the first delivery starts another before the drain looks again.
        second = asyncio.create_task(worker(second_release))
        owner._workers["second"] = second
        del owner._workers["first"]
        first_release.set()
        await asyncio.sleep(0.01)
        assert not drain.done()
        del owner._workers["second"]
        second_release.set()
        assert await asyncio.wait_for(drain, 1) is True

    asyncio.run(scenario())


def test_sealed_foreground_delivery_refuses_new_work_as_a_retryable_claim_loss() -> None:

    async def scenario() -> None:
        owner = ForegroundChildDeliveryOwner(store=None)  # ty: ignore[invalid-argument-type]
        owner.seal()

        async def operation(_before_mutation):  # pragma: no cover - never called
            raise AssertionError("refused before running")

        with pytest.raises(ForegroundChildDeliverySealed) as raised:
            await owner.run(None, operation)  # ty: ignore[invalid-argument-type]
        assert isinstance(raised.value, PersistedEventSideEffectClaimLost)

    asyncio.run(scenario())
