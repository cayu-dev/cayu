"""An admitted failover invocation must settle if its entrance read fails."""

import asyncio
import traceback

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, SessionStatus
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.sessions import InMemorySessionStore, ModelFailoverPolicy, ModelTarget
from cayu.storage.sqlite import SQLiteSessionStore


class EntranceReadFailure:
    invocation_lifecycle_command_version = 1

    async def replace_initial_transcript_messages(self, *args, **kwargs):
        result = await super().replace_initial_transcript_messages(*args, **kwargs)
        self.fail_next_checkpoint = self.inject_failure
        return result

    async def load_checkpoint(self, session_id):
        if getattr(self, "fail_next_checkpoint", False):
            self.fail_next_checkpoint = False
            self.failed_reads += 1
            raise ConnectionError("model-selection checkpoint temporarily unavailable")
        return await super().load_checkpoint(session_id)


class MemoryStore(EntranceReadFailure, InMemorySessionStore):
    invocation_lifecycle_command_version = 1


class SQLiteStore(EntranceReadFailure, SQLiteSessionStore):
    invocation_lifecycle_command_version = 1


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("inject_failure", [False, True])
def test_failover_entrance_read_failure_does_not_strand_admitted_session(
    backend, inject_failure, tmp_path
):
    async def scenario():
        store = MemoryStore() if backend == "memory" else SQLiteStore(tmp_path / "sessions.db")
        store.inject_failure = inject_failure
        store.failed_reads = 0
        provider = ScriptedModelProvider(
            [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()], name="primary"
        )
        backup = ScriptedModelProvider([], name="backup")
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_provider(backup)
        app.register_agent(AgentSpec(name="worker", model="small"))
        events = []
        try:
            try:
                async for event in app.run(
                    RunRequest(
                        session_id="entrance-read",
                        agent_name="worker",
                        messages=[Message.text("user", "hello")],
                        failover=ModelFailoverPolicy(
                            fallbacks=(ModelTarget(provider_name="backup", model="large"),)
                        ),
                    )
                ):
                    events.append(event)
            except ConnectionError as error:
                # Whether the store error is propagated or rendered as an event,
                # the invocation has already been admitted and must be settled.
                assert any(
                    frame.name == "_run_session_with_deadline"
                    for frame in traceback.extract_tb(error.__traceback__)
                )
            assert store.failed_reads == int(inject_failure)
            if inject_failure:
                assert provider.requests == backup.requests == []
            session = await store.load("entrance-read")
            assert session.status in {
                SessionStatus.COMPLETED,
                SessionStatus.FAILED,
                SessionStatus.INTERRUPTED,
            }, (session.status, [event.type for event in events])
            if not inject_failure:
                assert events[-1].type is EventType.SESSION_COMPLETED
        finally:
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
