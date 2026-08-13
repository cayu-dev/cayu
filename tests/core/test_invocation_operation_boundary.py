from __future__ import annotations

import asyncio
import threading

import pytest

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
        assert isinstance(first_outcome.cancellation, asyncio.CancelledError)
        assert finished.is_set() is False
        assert len(registry) == 1

        second_outcome = await await_invocation_operation(
            lambda: opaque_read("second"),
            request_child_cancellation=False,
            abandon_on_caller_cancellation=True,
            operation_registry=registry,
        )
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
        assert third_outcome.result == "third"
        assert third_outcome.error is None
        assert calls == ["first", "third"]
        assert len(registry) == 0

    asyncio.run(run())


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
