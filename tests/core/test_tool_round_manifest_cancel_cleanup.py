"""Manifest storage cleanup must not replace an authentic caller cancellation."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_mcp import FakeProvider, _fake_toolset

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.memory import InMemorySessionStore


@pytest.mark.parametrize("boundary", ["load", "publication"])
@pytest.mark.parametrize("cleanup_fails", [False, True], ids=["clean", "cleanup-error"])
def test_manifest_cleanup_failure_keeps_caller_cancellation(boundary, cleanup_fails):
    async def scenario():
        class CleanupFailingManifestStore(InMemorySessionStore):
            invocation_lifecycle_command_version = 1

            def __init__(self):
                super().__init__()
                self.operation_started = asyncio.Event()

            async def wait_for_cancellation(self):
                self.operation_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    # A failed rollback/connection cleanup replaces the error
                    # raised by the storage driver's interrupted operation.
                    if cleanup_fails:
                        raise OSError("manifest connection cleanup failed") from None
                    raise

            async def load_mcp_manifest_baselines(self, history_keys):
                if boundary == "load":
                    await self.wait_for_cancellation()
                return await super().load_mcp_manifest_baselines(history_keys)

            async def compare_and_publish_mcp_manifest_checks(self, *args, **kwargs):
                await self.wait_for_cancellation()
                return await super().compare_and_publish_mcp_manifest_checks(*args, **kwargs)

        store = CleanupFailingManifestStore()
        provider = FakeProvider([[ModelStreamEvent.completed({})]])
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=_fake_toolset().tools,
        )
        events = []

        async def consume():
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"manifest-cancel-cleanup-{boundary}",
                    messages=[Message.text("user", "hello")],
                )
            ):
                events.append(event)

        task = asyncio.create_task(consume())
        await asyncio.wait_for(store.operation_started.wait(), timeout=5)
        task.cancel("caller cancelled")
        propagated_cancellation = False
        try:
            await asyncio.wait_for(task, timeout=5)
        except asyncio.CancelledError:
            propagated_cancellation = True

        assert provider.requests == []
        assert propagated_cancellation, [
            (event.type, event.payload.get("error"))
            for event in events
            if event.type is EventType.SESSION_FAILED
        ]
        assert task.cancelled()
        assert not any(event.type is EventType.SESSION_FAILED for event in events)

    asyncio.run(scenario())
