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
        session_store=None,  # ty: ignore[invalid-argument-type]
        event_writer=None,  # ty: ignore[invalid-argument-type]
        clock=None,  # ty: ignore[invalid-argument-type]
        interrupt_session=unused,  # ty: ignore[invalid-argument-type]
        load_pending_session_interrupt_payload=unused,
        latest_session_interrupted_event=unused,
        load_pending_interruption_cascade=unused,
        claim_pending_interruption_cascade=unused,
        mark_pending_interruption_cascade_failed=unused,
        complete_pending_interruption_cascade=unused,
        renew_pending_interruption_cascade_claim=unused,
        release_pending_interruption_cascade_claim=unused,
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
