from __future__ import annotations

import asyncio

import pytest

from cayu import Message, ModelStreamEvent, ScriptedModelProvider
from cayu.providers._credential_boundary import (
    aclosing_provider_stream,
    provider_cancellation_failures,
)
from cayu.providers.base import ModelRequest


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("stop", ["cancel", "deadline", "repeat_cancel"])
@pytest.mark.parametrize(
    "cleanup",
    [
        "cooperative",
        "read_failure",
        "nested_failure",
        "close_failure",
        "delayed_read",
        "delayed_close",
        "suppressed",
        "stop_group",
    ],
)
def test_interrupted_read_precedes_close(streaming, stop, cleanup):
    async def run():
        initial_tasks = asyncio.all_tasks()
        entered = asyncio.Event()
        release = asyncio.Event()
        closed = asyncio.Event()
        resumed = []
        read_tasks = set()
        close_tasks = set()
        read_running = False
        timer = None
        consumer = None

        async def events():
            try:
                if streaming:
                    yield ModelStreamEvent.text_delta("partial")
                entered.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if stop == "repeat_cancel":
                    asyncio.get_running_loop().call_soon(consumer.cancel)
                if cleanup == "delayed_read":
                    await release.wait()
                elif cleanup == "nested_failure":

                    class BrokenClose:
                        async def aclose(self):
                            raise OSError("private nested close")

                    async with aclosing_provider_stream(BrokenClose(), cancellation_baseline=0):
                        raise asyncio.CancelledError() from None
                elif cleanup == "read_failure":
                    raise RuntimeError("private read cleanup") from None
                elif cleanup == "stop_group":
                    raise BaseExceptionGroup(
                        "private grouped cleanup", [asyncio.CancelledError(), RuntimeError()]
                    ) from None
                elif cleanup == "suppressed":
                    yield ModelStreamEvent.text_delta("must not resume")
                    return
                raise

        source = events()

        class Stream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                nonlocal read_running
                read_tasks.add(asyncio.current_task())
                read_running = True
                try:
                    return await anext(source)
                finally:
                    read_running = False

            async def aclose(self):
                close_tasks.add(asyncio.current_task())
                assert not read_running, "close overlapped a pending read"
                if cleanup == "delayed_close":
                    await release.wait()
                await source.aclose()
                closed.set()
                if cleanup == "close_failure":
                    raise ValueError("private close failure")

        class Provider(ScriptedModelProvider):
            def stream(self, request):
                return Stream()

        provider = Provider([])

        async def consume():
            nonlocal timer
            async with asyncio.timeout(None) as timer:
                async for event in provider.runtime_stream(
                    ModelRequest(model="test", messages=[Message.text("user", "check")])
                ):
                    resumed.append(event.delta)

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(entered.wait(), 2)
        if stop == "deadline":
            timer.reschedule(asyncio.get_running_loop().time())
        else:
            consumer.cancel()
        try:
            with pytest.raises(
                TimeoutError if stop == "deadline" else asyncio.CancelledError
            ) as raised:
                await asyncio.wait_for(consumer, 2)
            cancellation = raised.value.__cause__ if stop == "deadline" else raised.value
            diagnostics = provider_cancellation_failures(cancellation)
            if cleanup == "cooperative" and stop != "repeat_cancel":
                assert diagnostics == ()
            elif cleanup != "cooperative":
                assert diagnostics
                assert any(item["phase"] == "provider_stream_cleanup" for item in diagnostics), (
                    diagnostics
                )
                diagnostics = tuple(
                    item for item in diagnostics if item["phase"] == "provider_stream_cleanup"
                )
                assert diagnostics
                assert all(item["remote_settlement_state"] == "unknown" for item in diagnostics)
                if cleanup in {"delayed_read", "delayed_close"}:
                    assert any(item["stream_close_state"] == "pending" for item in diagnostics)
                    assert not closed.is_set()
                elif stop != "repeat_cancel":
                    if cleanup == "nested_failure":
                        assert any(
                            item["cleanup_exception_type"] == "OSError" for item in diagnostics
                        )
                    expected = {"read_failure": "RuntimeError", "close_failure": "ValueError"}
                    if cleanup in expected:
                        assert diagnostics[0]["cleanup_exception_type"] == expected[cleanup]
            assert resumed == (["partial"] if streaming else [])
        finally:
            release.set()
            await asyncio.wait_for(closed.wait(), 2)
            # Drain retained close callbacks before inspecting live task ownership.
            for _ in range(5):
                await asyncio.sleep(0)
            assert all(task.done() for task in read_tasks | close_tasks)
            assert source.ag_frame is None
            assert asyncio.all_tasks() <= initial_tasks

    asyncio.run(run())


@pytest.mark.parametrize("delayed", [False, True])
def test_cancellation_cleanup_keeps_dispatch_capacity_until_close(delayed):
    from cayu.providers import deadlines as deadline_module
    from cayu.providers.deadlines import ProviderStreamDeadlines

    async def run():
        owners_before = set(deadline_module._PROVIDER_DEADLINE_AWAIT_OWNERS)
        started = 0
        entered = asyncio.Event()
        release = asyncio.Event()
        closed = 0
        count = 80  # Exceeds the separate legacy deadline-close registry's limit.

        class Stream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                nonlocal started
                started += 1
                if started == count:
                    entered.set()
                await asyncio.Event().wait()

            async def aclose(self):
                nonlocal closed
                if delayed:
                    await release.wait()
                closed += 1

        class Provider(ScriptedModelProvider):
            @property
            def stream_deadlines(self):
                return ProviderStreamDeadlines(max_concurrent_streams=len(owners_before) + count)

            def stream(self, request):
                return Stream()

        provider = Provider([])
        request = ModelRequest(model="test", messages=[Message.text("user", "check")])
        tasks = [asyncio.create_task(anext(provider.runtime_stream(request))) for _ in range(count)]
        await asyncio.wait_for(entered.wait(), 3)
        for task in tasks:
            task.cancel()
        try:
            # Await individually: gather(return_exceptions=True) detaches Task cancellation details.
            for task in tasks:
                with pytest.raises(asyncio.CancelledError) as raised:
                    await task
                diagnostics = provider_cancellation_failures(raised.value)
                assert bool(diagnostics) is delayed
                if delayed:
                    assert diagnostics[0]["cleanup_reason"] == "cleanup_pending"
            assert closed == (0 if delayed else count)
            if delayed:
                with pytest.raises(RuntimeError, match="deadline-read capacity is exhausted"):
                    await anext(provider.runtime_stream(request))
                assert started == count
        finally:
            release.set()
            async with asyncio.timeout(3):
                while deadline_module._PROVIDER_DEADLINE_AWAIT_OWNERS - owners_before:
                    await asyncio.sleep(0)
            assert closed == count
            assert all(task.done() for task in tasks)

    asyncio.run(run())


@pytest.mark.parametrize("source_deadline", [False, True])
def test_completed_read_is_not_misclassified_as_failed_cancellation(source_deadline):
    from cayu.providers.base import ModelStreamDeadlineError
    from cayu.providers.deadlines import (
        ProviderDeadlineKind,
        ProviderStreamDeadlineExceeded,
        current_provider_deadline_controller,
    )

    async def run():
        settled = False

        class Provider(ScriptedModelProvider):
            async def stream(self, request):
                nonlocal settled
                controller = current_provider_deadline_controller()
                try:
                    if source_deadline:
                        raise ProviderStreamDeadlineExceeded(
                            controller.evidence((ProviderDeadlineKind.ABSOLUTE,))
                        )
                    # Move the clock boundary while the read itself completes,
                    # exercising expiry inspection after a ready value, not a
                    # read that swallowed cancellation.
                    controller._started_at -= controller.deadlines.absolute_stream_timeout_s + 1
                    yield ModelStreamEvent.text_delta("late value")
                finally:
                    settled = True

        request = ModelRequest(model="test", messages=[Message.text("user", "check")])
        with pytest.raises(ModelStreamDeadlineError) as raised:
            await anext(Provider([]).runtime_stream(request))
        assert not raised.value.stream_cleanup_failed
        assert settled

    asyncio.run(run())
