"""An admitted resume settles when any checkpoint read after admission fails."""

import asyncio

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest, SessionStatus
from cayu.evals import ScriptedModelProvider
from cayu.observability.events import EventSink
from cayu.providers import ModelStreamEvent
from cayu.sessions import InMemorySessionStore, ModelFailoverPolicy, ModelTarget
from cayu.storage.sqlite import SQLiteSessionStore


class EntranceReadFailure:
    invocation_lifecycle_command_version = 1
    fail_at = 0
    reads_since_admission: int | None = None

    async def load_checkpoint(self, session_id):
        if self.reads_since_admission is not None:
            self.reads_since_admission += 1
            if self.reads_since_admission == self.fail_at:
                self.reads_since_admission = None
                self.failed = True
                raise ConnectionError("checkpoint temporarily unavailable")
        return await super().load_checkpoint(session_id)


class AdmissionSink(EventSink):
    def __init__(self, store):
        self.store = store

    async def emit(self, event):
        if self.store.fail_at and event.type is EventType.INTERACTION_STARTED:
            self.store.reads_since_admission = 0


class MemoryStore(EntranceReadFailure, InMemorySessionStore):
    invocation_lifecycle_command_version = 1


class SQLiteStore(EntranceReadFailure, SQLiteSessionStore):
    invocation_lifecycle_command_version = 1


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("fail_at", range(1, 6))
def test_resume_entrance_read_failure_settles_admitted_resume(backend, fail_at, tmp_path):
    async def scenario():
        store = MemoryStore() if backend == "memory" else SQLiteStore(tmp_path / "sessions.db")
        store.failed = False
        provider = ScriptedModelProvider(
            [
                [ModelStreamEvent.text_delta("first"), ModelStreamEvent.completed()],
                [ModelStreamEvent.text_delta("second"), ModelStreamEvent.completed()],
                [ModelStreamEvent.text_delta("third"), ModelStreamEvent.completed()],
            ],
            name="primary",
        )
        app = CayuApp(session_store=store, event_sinks=[AdmissionSink(store)], enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_provider(ScriptedModelProvider([], name="backup"))
        app.register_agent(AgentSpec(name="worker", model="small"))
        failover = ModelFailoverPolicy(
            fallbacks=(ModelTarget(provider_name="backup", model="large"),)
        )
        try:
            async for _ in app.run(
                RunRequest(
                    session_id="entrance-read",
                    agent_name="worker",
                    messages=[Message.text("user", "hello")],
                    failover=failover,
                )
            ):
                pass
            store.fail_at = fail_at
            try:
                async for _ in app.resume(
                    ResumeRequest(
                        session_id="entrance-read",
                        messages=[Message.text("user", "again")],
                        failover=failover,
                    )
                ):
                    pass
            except ConnectionError:
                pass
            store.fail_at = 0
            assert await app.drain_recovery_cleanups(timeout_s=5)
            assert await app.drain_environment_cleanups(timeout_s=5)
            session = await store.load("entrance-read")
            assert session.status in {
                SessionStatus.COMPLETED,
                SessionStatus.FAILED,
                SessionStatus.INTERRUPTED,
            }, (fail_at, store.failed, session.status)
            retried = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id="entrance-read",
                        messages=[Message.text("user", "after the failure")],
                        failover=failover,
                    )
                )
            ]
            assert retried[-1].type is EventType.SESSION_COMPLETED
        finally:
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
