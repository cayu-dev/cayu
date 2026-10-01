from __future__ import annotations

import asyncio
import gc
import weakref

import pytest

from cayu.sessions.execution import (
    current_execution_owner_kind,
    execution_owned_by,
    execution_owner_kind,
)


@pytest.mark.parametrize("streaming", [False, True], ids=["coroutine", "stream"])
@pytest.mark.parametrize("keyword", [False, True], ids=["positional", "keyword"])
def test_rejected_execution_does_not_keep_the_original_request(streaming, keyword):
    class Request:
        pass

    @execution_owned_by("recovery")
    async def call(request):
        del request
        raise ValueError("request rejected")

    @execution_owned_by("recovery")
    async def stream(request):
        del request
        raise ValueError("request rejected")
        yield  # pragma: no cover

    async def scenario():
        request = Request()
        reference = weakref.ref(request)
        operation = stream if streaming else call
        pending = operation(request=request) if keyword else operation(request)
        del request
        with pytest.raises(ValueError, match="request rejected") as failure:
            if streaming:
                await anext(pending)
            else:
                await pending
        # Keep the exception and traceback alive: neither may retain the request
        # once the entry point has discarded it.
        assert failure.value.__traceback__ is not None
        gc.collect()
        assert reference() is None

    asyncio.run(scenario())


def test_execution_stream_forwards_send_and_throw_under_its_owner_context():
    observed = []
    injected = ValueError("consumer rejected event")

    @execution_owned_by("recovery")
    async def stream():
        try:
            observed.append(current_execution_owner_kind())
            sent = yield "first"
            observed.append((sent, current_execution_owner_kind()))
            try:
                yield "second"
            except ValueError as error:
                assert error is injected
                observed.append(current_execution_owner_kind())
                yield "handled"
        finally:
            observed.append(("closed", current_execution_owner_kind()))

    async def scenario():
        with execution_owner_kind("server_stream"):
            iterator = stream()
            assert await anext(iterator) == "first"
            assert current_execution_owner_kind() == "server_stream"
            assert await iterator.asend("sent value") == "second"
            assert current_execution_owner_kind() == "server_stream"
            assert await iterator.athrow(injected) == "handled"
            assert current_execution_owner_kind() == "server_stream"
            await iterator.aclose()
            assert current_execution_owner_kind() == "server_stream"

    asyncio.run(scenario())
    assert observed == [
        "recovery",
        ("sent value", "recovery"),
        "recovery",
        ("closed", "recovery"),
    ]


@pytest.mark.parametrize("streaming", [False, True], ids=["coroutine", "stream"])
def test_execution_cancellation_restores_the_caller_context_after_cleanup(streaming):
    async def scenario():
        entered = asyncio.Event()
        never = asyncio.Event()
        observed = []

        @execution_owned_by("recovery")
        async def call():
            try:
                observed.append(current_execution_owner_kind())
                entered.set()
                await never.wait()
            finally:
                observed.append(("closed", current_execution_owner_kind()))

        @execution_owned_by("recovery")
        async def stream():
            try:
                observed.append(current_execution_owner_kind())
                entered.set()
                await never.wait()
                yield "unreachable"
            finally:
                observed.append(("closed", current_execution_owner_kind()))

        async def consume():
            with execution_owner_kind("server_stream"):
                try:
                    if streaming:
                        async for _ in stream():
                            pass
                    else:
                        await call()
                finally:
                    observed.append(("caller", current_execution_owner_kind()))

        task = asyncio.create_task(consume())
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert observed == [
            "recovery",
            ("closed", "recovery"),
            ("caller", "server_stream"),
        ]

    asyncio.run(scenario())
