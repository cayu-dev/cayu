"""A live interruption must keep an unconfirmed remote operation recoverable."""

import asyncio

import pytest
from tests.core.test_provider_operation_offline_recovery import (
    _LiveCancellableBudgetedOperationAdapter,
    _OfflineOperationAdapter,
    _OfflineOperationProvider,
)

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest, SessionStatus
from cayu.providers import ModelStreamEvent
from cayu.providers.operations import ProviderOperationCancellationSupport, ProviderOperationStatus
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.requests import InterruptSessionRequest
from cayu.storage.sqlite import SQLiteSessionStore


class _ContinuableLiveAdapter(_LiveCancellableBudgetedOperationAdapter):
    async def start(self, request):
        if self.start_calls == 0:
            return await super().start(request)
        # A subsequent message may legitimately start its own operation after
        # the original result is recovered. Do not block that new interaction.
        self.start_events = (ModelStreamEvent.completed({"finish_reason": "stop"}),)
        return await _OfflineOperationAdapter.start(self, request)


class _UnsupportedLiveAdapter(_ContinuableLiveAdapter):
    @property
    def cancellation_support(self):
        return ProviderOperationCancellationSupport.UNSUPPORTED


class _FailedCancellationLiveAdapter(_ContinuableLiveAdapter):
    async def cancel(self, state):
        self.cancel_calls.append(state)
        raise TimeoutError("cancellation request acknowledgement lost")


class _UncancelledLiveProvider(_OfflineOperationProvider):
    def __init__(self, cancellation):
        self.adapter = (
            _UnsupportedLiveAdapter()
            if cancellation == "unsupported"
            else _FailedCancellationLiveAdapter()
        )


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("cancellation", ["unsupported", "timeout"])
async def test_live_interrupt_preserves_interaction_for_uncancelled_provider_operation(
    tmp_path, backend, cancellation
):
    store = (
        InMemorySessionStore()
        if backend == "memory"
        else SQLiteSessionStore(tmp_path / "live-provider-pause.sqlite")
    )
    provider = _UncancelledLiveProvider(cancellation)
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    session_id = "live-provider-pause"

    async def run():
        return [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "finish remotely")],
                )
            )
        ]

    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(provider.adapter.stream_started.wait(), 10)
        _ = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(session_id=session_id, reason="stop observing")
            )
        ]
        await asyncio.wait_for(task, 10)
        assert (await store.load(session_id)).status is SessionStatus.INTERRUPTED
        assert await store.load_active_model_completion_stage(session_id) is not None
        events = await store.load_events(session_id)
        lifecycle = [
            event
            for event in events
            if event.type in {EventType.INTERACTION_PAUSED, EventType.INTERACTION_INTERRUPTED}
        ]
        # The remote operation was never confirmed cancelled and subsequently
        # finishes. Its exact output must remain publishable by ordinary resume.
        provider.adapter.status = ProviderOperationStatus.COMPLETED

        async def resume():
            return [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "continue")],
                    )
                )
            ]

        resumed = await asyncio.wait_for(resume(), 10)
        assert resumed[-1].type is EventType.SESSION_COMPLETED
        assert provider.adapter.start_calls == 2
        assert len(provider.adapter.retrieve_calls) == 1
        assert [event.type for event in lifecycle] == [EventType.INTERACTION_PAUSED], (
            "An operation that cannot be cancelled still owns its interaction; "
            f"published lifecycle: {[event.type for event in lifecycle]}"
        )
        assert lifecycle[0].payload["pending_action_kind"] == "provider_operation_recovery"
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await app.aclose()
        if backend == "sqlite":
            await store.close()
