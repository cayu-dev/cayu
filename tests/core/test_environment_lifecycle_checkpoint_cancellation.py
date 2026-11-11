"""Checkpoint reconciliation must preserve cancellation owned by its caller."""

from __future__ import annotations

import asyncio

import pytest
from tests.core._workload_secret_support import FakeProvider, collect_events
from tests.core.test_environment_lifecycle import _lifecycle

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.environments import (
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryResult,
    EnvironmentSpec,
)
from cayu.runtime._environment_lifecycle import ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY
from cayu.sessions import InMemorySessionStore
from cayu.sessions.records import SessionIdentity


class _DelayedAcknowledgementStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self):
        super().__init__()
        self.write_committed = asyncio.Event()
        self.read_started = asyncio.Event()
        self.allow_read = asyncio.Event()
        self.pause_write = True
        self.reconciling = False

    async def transform_checkpoint(self, session_id, transform):
        reconnect_written = False

        def inspect_write(session, checkpoint):
            nonlocal reconnect_written
            updated = transform(session, checkpoint)
            reconnect_written = (
                updated is not None and ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY in updated
            )
            return updated

        await super().transform_checkpoint(session_id, inspect_write)
        if reconnect_written and self.pause_write:
            self.pause_write = False
            self.write_committed.set()
            try:
                # Model a committed write whose acknowledgement is interrupted.
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.reconciling = True
                raise

    async def load_checkpoint(self, session_id):
        if self.reconciling:
            self.read_started.set()
            await self.allow_read.wait()
            self.reconciling = False
        return await super().load_checkpoint(session_id)


@pytest.mark.parametrize("entry", ["lifecycle", "app"])
@pytest.mark.parametrize("external_cancellation", [False, True], ids=["timeout-only", "shutdown"])
def test_checkpoint_reconciliation_does_not_turn_shutdown_into_timeout(
    entry, external_cancellation
):
    async def scenario():
        store = _DelayedAcknowledgementStore()
        session_id = "checkpoint-shutdown"
        request = RunRequest(
            session_id=session_id,
            agent_name="assistant",
            messages=[Message.text("user", "run")],
        )
        app = None
        releases = []
        if entry == "lifecycle":
            await store.create(
                request,
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            lifecycle = _lifecycle(store)

            async def operation():
                await lifecycle._checkpoint_factory_reconnect_metadata(
                    session_id=session_id,
                    environment_name="sandbox",
                    reconnect_metadata={"sandbox_id": "existing-allocation"},
                    allocation_fingerprint=None,
                )
        else:

            class Factory(EnvironmentFactory):
                async def create(self, request):
                    async def release(action):
                        releases.append(action)

                    return EnvironmentFactoryResult(
                        Environment(EnvironmentSpec(name=request.environment_name)),
                        reconnect_metadata={"sandbox_id": "existing-allocation"},
                        release=release,
                    )

            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(FakeProvider([]), default=True)
            app.register_environment_factory(
                EnvironmentSpec(name="sandbox"), Factory(), default=True
            )
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))

            async def operation():
                await collect_events(app, request)

        timeout = None

        async def checkpoint_under_deadline():
            nonlocal timeout
            async with asyncio.timeout(None) as timeout:
                await operation()

        task = asyncio.create_task(checkpoint_under_deadline())
        try:
            await asyncio.wait_for(store.write_committed.wait(), timeout=3)
            assert timeout is not None
            timeout.reschedule(asyncio.get_running_loop().time())
            await asyncio.wait_for(store.read_started.wait(), timeout=3)
            assert timeout.expired()
            if external_cancellation:
                # Unlike the deadline's own cancellation, shutdown must escape
                # the timeout context as CancelledError, not as a retryable timeout.
                task.cancel("operator shutdown")
                await asyncio.sleep(0)
            store.allow_read.set()
            expected = asyncio.CancelledError if external_cancellation else TimeoutError
            with pytest.raises(expected):
                await task
            if app is not None:
                assert len(releases) == 1
        finally:
            store.allow_read.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            if app is not None:
                await app.aclose()

    asyncio.run(scenario())
