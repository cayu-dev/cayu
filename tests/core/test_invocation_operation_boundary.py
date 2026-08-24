from __future__ import annotations

import asyncio
import threading

import pytest

import cayu.tools._operation_boundary as operation_boundary_module
from cayu.tools._operation_boundary import (
    BoundedInvocationOperationRegistry,
    InvocationOperationCapacityError,
    await_invocation_operation,
)


def test_abandonable_operation_registry_tracks_executor_work_until_physical_completion() -> None:
    async def run() -> None:
        registry = BoundedInvocationOperationRegistry(max_operations=1)
        started = threading.Event()
        finished = threading.Event()
        release = threading.Event()
        calls: list[str] = []

        def blocking_read(name: str) -> str:
            calls.append(name)
            started.set()
            try:
                if not release.wait(timeout=5):
                    raise RuntimeError("test executor barrier timed out")
                return name
            finally:
                finished.set()

        async def opaque_read(name: str) -> str:
            return await asyncio.to_thread(blocking_read, name)

        first = asyncio.create_task(
            await_invocation_operation(
                lambda: opaque_read("first"),
                request_child_cancellation=False,
                abandon_on_caller_cancellation=True,
                operation_registry=registry,
            )
        )
        while not started.is_set():
            await asyncio.sleep(0)
        first.cancel("caller stopped waiting")
        first_outcome = await asyncio.wait_for(first, timeout=1)
        assert first_outcome.operation_started is True
        assert isinstance(first_outcome.cancellation, asyncio.CancelledError)
        assert finished.is_set() is False
        assert len(registry) == 1

        second_outcome = await await_invocation_operation(
            lambda: opaque_read("second"),
            request_child_cancellation=False,
            abandon_on_caller_cancellation=True,
            operation_registry=registry,
        )
        assert second_outcome.operation_started is False
        assert isinstance(second_outcome.error, InvocationOperationCapacityError)
        assert calls == ["first"]
        assert len(registry) == 1

        release.set()
        while registry:
            await asyncio.sleep(0)

        third_outcome = await await_invocation_operation(
            lambda: opaque_read("third"),
            request_child_cancellation=False,
            abandon_on_caller_cancellation=True,
            operation_registry=registry,
        )
        assert third_outcome.operation_started is True
        assert third_outcome.result == "third"
        assert third_outcome.error is None
        assert calls == ["first", "third"]
        assert len(registry) == 0

    asyncio.run(run())


def test_abandonable_operation_registry_shutdown_cancels_cooperative_children() -> None:
    async def run() -> tuple[bool, bool, int, InvocationOperationCapacityError | None]:
        registry = BoundedInvocationOperationRegistry(max_operations=1)
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def cooperative_read() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        caller = asyncio.create_task(
            await_invocation_operation(
                cooperative_read,
                request_child_cancellation=False,
                abandon_on_caller_cancellation=True,
                operation_registry=registry,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        caller.cancel("caller stopped waiting")
        outcome = await caller
        assert isinstance(outcome.cancellation, asyncio.CancelledError)
        assert len(registry) == 1

        drained = await registry.aclose(timeout_s=0)
        await asyncio.wait_for(stopped.wait(), timeout=1)
        rejected = await await_invocation_operation(
            lambda: asyncio.sleep(0),
            request_child_cancellation=False,
            abandon_on_caller_cancellation=True,
            operation_registry=registry,
        )
        return (
            drained,
            registry.closed,
            len(registry),
            (
                rejected.error
                if isinstance(rejected.error, InvocationOperationCapacityError)
                else None
            ),
        )

    drained, closed, remaining, rejection = asyncio.run(run())

    assert drained is False
    assert closed is True
    assert remaining == 0
    assert isinstance(rejection, InvocationOperationCapacityError)


def test_pending_cancellation_proves_operation_was_not_started() -> None:
    async def run():
        calls = 0

        async def operation() -> None:
            nonlocal calls
            calls += 1

        async def invoke():
            current = asyncio.current_task()
            assert current is not None
            current.cancel("cancel before operation dispatch")
            outcome = await await_invocation_operation(operation)
            return outcome, current.cancelling()

        task = asyncio.create_task(invoke())
        outcome, cancelling = await task
        return outcome, cancelling, task.cancelled(), calls

    outcome, cancelling, cancelled, calls = asyncio.run(run())

    assert outcome.operation_started is False
    assert isinstance(outcome.cancellation, asyncio.CancelledError)
    assert outcome.cancellation.args == ("cancel before operation dispatch",)
    assert cancelling == 1
    assert cancelled is False
    assert calls == 0


def test_operation_factory_failure_is_conservatively_started() -> None:
    async def run():
        def operation():
            raise RuntimeError("factory failed")

        return await await_invocation_operation(operation)

    outcome = asyncio.run(run())

    assert outcome.operation_started is True
    assert type(outcome.error) is RuntimeError
    assert str(outcome.error) == "factory failed"


def test_abandonable_operation_requires_noncancelling_child_ownership() -> None:
    async def run() -> None:
        registry = BoundedInvocationOperationRegistry(max_operations=1)

        with pytest.raises(ValueError, match="retain their child"):
            await await_invocation_operation(
                lambda: asyncio.sleep(0),
                abandon_on_caller_cancellation=True,
                operation_registry=registry,
            )

        assert len(registry) == 0

    asyncio.run(run())


@pytest.mark.parametrize(
    "signal_factory",
    [
        lambda: GeneratorExit("supervisor abandoned invocation"),
        lambda: BaseExceptionGroup(
            "supervisory failures",
            [SystemExit(17), RuntimeError("secondary cleanup failure")],
        ),
    ],
    ids=("exact", "grouped"),
)
def test_supervisory_exit_transfers_started_operation_until_settlement(
    monkeypatch,
    signal_factory,
) -> None:
    async def run() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        retained_probes = []
        original_shield = asyncio.shield
        delivered = False
        signal = signal_factory()

        async def operation() -> None:
            started.set()
            await release.wait()

        async def supervisory_shield(awaitable):  # type: ignore[no-untyped-def]
            nonlocal delivered
            await started.wait()
            if not delivered:
                delivered = True
                raise signal
            return await original_shield(awaitable)

        monkeypatch.setattr(operation_boundary_module.asyncio, "shield", supervisory_shield)
        with pytest.raises(BaseException) as raised:
            await await_invocation_operation(
                operation,
                request_child_cancellation=False,
                on_unsettled_supervisory_exit=retained_probes.append,
            )
        assert raised.value is signal
        assert len(retained_probes) == 1

        monkeypatch.setattr(operation_boundary_module.asyncio, "shield", original_shield)
        settlement = asyncio.create_task(retained_probes[0]())
        await asyncio.sleep(0)
        assert settlement.done() is False

        release.set()
        assert await settlement is True

    asyncio.run(run())


@pytest.mark.parametrize(
    "signal_factory",
    [
        lambda: GeneratorExit("supervisor abandoned completed invocation"),
        lambda: BaseExceptionGroup(
            "post-completion supervisory failures",
            [SystemExit(17), RuntimeError("secondary cleanup failure")],
        ),
    ],
    ids=("exact", "grouped"),
)
def test_supervisory_exit_at_post_completion_checkpoint_retains_operation(
    monkeypatch,
    signal_factory,
) -> None:
    async def run() -> None:
        completed = asyncio.Event()
        retained_probes = []
        original_sleep = asyncio.sleep
        delivered = False
        signal = signal_factory()

        async def operation() -> str:
            completed.set()
            return "deferred-cleanup-outcome"

        async def supervisory_sleep(delay):  # type: ignore[no-untyped-def]
            nonlocal delivered
            result = await original_sleep(delay)
            if completed.is_set() and not delivered:
                delivered = True
                raise signal
            return result

        monkeypatch.setattr(operation_boundary_module.asyncio, "sleep", supervisory_sleep)
        with pytest.raises(BaseException) as raised:
            await await_invocation_operation(
                operation,
                request_child_cancellation=False,
                on_unsettled_supervisory_exit=retained_probes.append,
            )
        assert raised.value is signal
        assert delivered is True
        assert len(retained_probes) == 1

        outcome = await retained_probes[0].outcome()
        assert outcome is not None
        assert outcome.operation_started is True
        assert outcome.result == "deferred-cleanup-outcome"

    asyncio.run(run())
