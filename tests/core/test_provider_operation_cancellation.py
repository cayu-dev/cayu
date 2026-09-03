from __future__ import annotations

import asyncio

import pytest

from cayu.providers import (
    ProviderOperationAdapter,
    ProviderOperationConnection,
    ProviderOperationSnapshot,
    ProviderOperationStartRequest,
    ProviderOperationState,
    ProviderOperationStatus,
)
from cayu.runtime.provider_operation_cancellation import (
    ProviderOperationCancellationAdmissionsSealed,
    ProviderOperationCancellationCapacityExceeded,
    ProviderOperationCancellationLifecycle,
)


class _CancellationAdapter(ProviderOperationAdapter):
    def __init__(self, *, expected_calls: int, resist_cancellation: bool = False) -> None:
        self.expected_calls = expected_calls
        self.resist_cancellation = resist_cancellation
        self.calls: list[ProviderOperationState] = []
        self.cancelled_calls = 0
        self.all_entered = asyncio.Event()
        self.release = asyncio.Event()

    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        del request
        raise AssertionError("start is not used")

    async def retrieve(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        del state
        raise AssertionError("retrieve is not used")

    async def reconnect(self, state: ProviderOperationState) -> ProviderOperationConnection:
        del state
        raise AssertionError("reconnect is not used")

    async def cancel(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        self.calls.append(state)
        if len(self.calls) == self.expected_calls:
            self.all_entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled_calls += 1
            if self.resist_cancellation:
                await self.release.wait()
        return ProviderOperationSnapshot(
            state=state,
            status=ProviderOperationStatus.CANCELLED,
        )


def _state(index: int) -> ProviderOperationState:
    return ProviderOperationState(
        operation_id=f"operation-{index}",
        stream_protocol="test-v1",
    )


def test_shutdown_drains_many_provider_operation_cancellation_owners() -> None:
    async def scenario() -> None:
        owner_count = 64
        lifecycle = ProviderOperationCancellationLifecycle(max_active_owners=owner_count)
        adapter = _CancellationAdapter(expected_calls=owner_count)
        tasks = [
            lifecycle.admit(
                adapter=adapter,
                state=state,
                cancellation=lambda state=state: adapter.cancel(state),
                ownership_lost=None,
            )
            for state in (_state(index) for index in range(owner_count))
        ]
        await asyncio.wait_for(adapter.all_entered.wait(), timeout=1)

        before = lifecycle.snapshot()
        assert before.admissions_sealed is False
        assert before.active_owners == owner_count
        assert before.active_tasks == owner_count

        assert await lifecycle.drain(timeout_s=1) is True
        snapshots = await asyncio.gather(*tasks)
        assert {snapshot.state.operation_id for snapshot in snapshots} == {
            f"operation-{index}" for index in range(owner_count)
        }
        after = lifecycle.snapshot()
        assert after.admissions_sealed is True
        assert after.active_owners == 0
        assert after.active_tasks == 0
        assert after.completed_owners == owner_count
        assert after.shutdown_cancellation_requests == owner_count
        assert adapter.cancelled_calls == owner_count

        assert await lifecycle.drain(timeout_s=1) is True
        assert lifecycle.snapshot() == after
        assert not [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and task.get_name().startswith("cayu-provider-operation-cancel")
        ]

    asyncio.run(scenario())


def test_shutdown_reports_unresolved_owner_and_repeated_drain_is_idempotent() -> None:
    async def scenario() -> None:
        lifecycle = ProviderOperationCancellationLifecycle(max_active_owners=1)
        adapter = _CancellationAdapter(expected_calls=1, resist_cancellation=True)
        state = _state(1)
        first = lifecycle.admit(
            adapter=adapter,
            state=state,
            cancellation=lambda: adapter.cancel(state),
            ownership_lost=None,
        )
        duplicate = lifecycle.admit(
            adapter=adapter,
            state=state,
            cancellation=lambda: adapter.cancel(state),
            ownership_lost=None,
        )
        assert duplicate is first
        await asyncio.wait_for(adapter.all_entered.wait(), timeout=1)

        with pytest.raises(ProviderOperationCancellationCapacityExceeded):
            lifecycle.admit(
                adapter=adapter,
                state=_state(2),
                cancellation=lambda: adapter.cancel(_state(2)),
                ownership_lost=None,
            )

        assert await lifecycle.drain(timeout_s=0.001) is False
        unresolved = lifecycle.snapshot()
        assert unresolved.admissions_sealed is True
        assert unresolved.active_owners == 1
        assert unresolved.shutdown_cancellation_requests == 1
        assert len(adapter.calls) == 1

        with pytest.raises(ProviderOperationCancellationAdmissionsSealed):
            lifecycle.admit(
                adapter=adapter,
                state=_state(2),
                cancellation=lambda: adapter.cancel(_state(2)),
                ownership_lost=None,
            )

        adapter.release.set()
        assert await lifecycle.drain(timeout_s=1) is True
        assert (await first).status is ProviderOperationStatus.CANCELLED
        settled = lifecycle.snapshot()
        assert settled.active_owners == 0
        assert settled.shutdown_cancellation_requests == 1
        assert settled.admission_rejections == 1
        assert settled.capacity_rejections == 1
        assert len(adapter.calls) == 1

    asyncio.run(scenario())


def test_owner_remains_tracked_until_ownership_watcher_settles() -> None:
    async def scenario() -> None:
        lifecycle = ProviderOperationCancellationLifecycle(max_active_owners=1)
        adapter = _CancellationAdapter(expected_calls=1)
        ownership_lost = asyncio.Event()
        state = _state(1)
        task = lifecycle.admit(
            adapter=adapter,
            state=state,
            cancellation=lambda: adapter.cancel(state),
            ownership_lost=ownership_lost,
        )
        await asyncio.wait_for(adapter.all_entered.wait(), timeout=1)
        assert lifecycle.snapshot().active_tasks == 2

        ownership_lost.set()
        assert (await task).status is ProviderOperationStatus.CANCELLED
        assert await lifecycle.drain(timeout_s=1) is True
        settled = lifecycle.snapshot()
        assert settled.active_owners == 0
        assert settled.active_tasks == 0
        assert settled.completed_owners == 1
        assert not [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and task.get_name().startswith("cayu-provider-operation-cancel")
        ]

    asyncio.run(scenario())
