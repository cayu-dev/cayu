"""Grouped cancellation during admitted resume setup must settle the new turn."""

import asyncio

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest, SessionStatus
from cayu.evals import ScriptedModelProvider
from cayu.observability.events import EventSink
from cayu.providers import ModelStreamEvent
from cayu.sessions import InMemorySessionStore
from cayu.storage.sqlite import SQLiteSessionStore


class AdmissionSink(EventSink):
    def __init__(self, grouped):
        self.grouped = grouped
        self.armed = False
        self.entered = asyncio.Event()

    async def emit(self, event):
        if self.armed and event.type is EventType.INTERACTION_STARTED:
            self.armed = False
            self.entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                if self.grouped:
                    raise BaseExceptionGroup(
                        "Event delivery cancelled and connection cleanup failed",
                        [cancellation, OSError("sink connection close failed")],
                    ) from None
                raise


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "grouped", [False, True], ids=["plain-cancellation", "grouped-cancellation"]
)
def test_resume_admission_cancellation_settles_interaction(tmp_path, backend, grouped):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "resume-group.sqlite")
        )
        sink = AdmissionSink(grouped)
        provider = ScriptedModelProvider(
            [ModelStreamEvent.text_delta("initial answer"), ModelStreamEvent.completed()]
        )
        app = CayuApp(session_store=store, event_sinks=[sink], enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="scripted-model"))
        session_id = "resume-group-cancellation"
        try:
            initial = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "first question")],
                    )
                )
            ]
            assert initial[-1].type is EventType.SESSION_COMPLETED
            assert len(provider.requests) == 1
            sink.armed = True

            async def resume():
                return [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id=session_id,
                            messages=[Message.text("user", "second question")],
                        )
                    )
                ]

            task = asyncio.create_task(resume())
            await asyncio.wait_for(sink.entered.wait(), timeout=10)
            task.cancel("caller disconnected during resume admission")
            with pytest.raises(BaseExceptionGroup if grouped else asyncio.CancelledError):
                await task
            assert len(provider.requests) == 1
            session = await store.load(session_id)
            assert session.status is SessionStatus.INTERRUPTED
            events = await store.load_events(session_id)
            started = [event for event in events if event.type is EventType.INTERACTION_STARTED]
            assert len(started) == 2
            assert any(
                event.type is EventType.INTERACTION_INTERRUPTED
                and event.interaction_id == started[-1].interaction_id
                for event in events
            )
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
