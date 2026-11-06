"""Cancellation during queued event fan-out must leave a resumable successor."""

import asyncio

import pytest
from tests.core.test_queued_session_messages import BlockingTwoTurnProvider

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest, SessionStatus
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.observability.events import EventSink
from cayu.sessions import EnqueueSessionMessageRequest
from cayu.sessions.memory import InMemorySessionStore


class GatedQueuedDeliverySink(EventSink):
    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def emit(self, event):
        if event.type is EventType.SESSION_MESSAGE_DELIVERED:
            self.entered.set()
            await self.release.wait()


@pytest.mark.parametrize("with_environment", [False, True])
@pytest.mark.parametrize("cancel_at_delivery", [False, True])
def test_queued_delivery_cancellation_settles_successor(cancel_at_delivery, with_environment):
    async def scenario():
        store = InMemorySessionStore()
        provider = BlockingTwoTurnProvider()
        sink = GatedQueuedDeliverySink()
        app = CayuApp(session_store=store, event_sinks=[sink], enable_logging=False)
        if with_environment:
            app.register_environment(Environment(EnvironmentSpec(name="local")), default=True)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "queued-delivery-cancellation"

        async def execute():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "initial")],
                    )
                )
            ]

        task = asyncio.create_task(execute())
        try:
            await asyncio.wait_for(provider.first_started.wait(), 10)
            await app.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id=session_id,
                    idempotency_key="successor-input",
                    content="Please handle this follow-up",
                    delivery_mode="on_idle",
                )
            )
            provider.release_first.set()
            await asyncio.wait_for(sink.entered.wait(), 10)
            durable = await store.load_events(session_id)
            starts = [e for e in durable if e.type is EventType.INTERACTION_STARTED]
            assert len(starts) == 2
            successor_id = starts[-1].interaction_id
            if cancel_at_delivery:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                sink.release.set()
                await task

            sink.release.set()
            await app.drain_environment_cleanups()
            session = await store.load(session_id)
            expected = SessionStatus.INTERRUPTED if cancel_at_delivery else SessionStatus.COMPLETED
            assert session.status is expected
            durable_before_resume = await store.load_events(session_id)

            # The owner and all its cleanup have returned: public resume must
            # not remain fenced by the predecessor's stale invocation context.
            resumed = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "continue")],
                    )
                )
            ]
            assert resumed[-1].type is EventType.SESSION_COMPLETED
            expected_event = (
                EventType.INTERACTION_INTERRUPTED
                if cancel_at_delivery
                else EventType.INTERACTION_COMPLETED
            )
            assert any(
                e.type is expected_event and e.interaction_id == successor_id
                for e in durable_before_resume
            )
        finally:
            sink.release.set()
            provider.release_first.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await app.drain_environment_cleanups()

    asyncio.run(scenario())
