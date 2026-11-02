"""Closing after task failure must not abandon session failure settlement."""

import asyncio

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, SessionStatus
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.sessions.memory import InMemorySessionStore
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskStatus


class FailInitialTranscriptOnce(InMemorySessionStore):
    invocation_lifecycle_command_version = 1
    terminal_interaction_publication_version = 1

    def __init__(self):
        super().__init__()
        self.failed = False

    async def replace_initial_transcript_messages(self, *args, **kwargs):
        if not self.failed:
            self.failed = True
            raise ConnectionError("initial transcript publication temporarily unavailable")
        return await super().replace_initial_transcript_messages(*args, **kwargs)


@pytest.mark.parametrize("close_on_task_failure", [False, True])
def test_setup_task_failure_still_settles_session_on_stream_close(close_on_task_failure):
    async def scenario():
        sessions = FailInitialTranscriptOnce()
        tasks = InMemoryTaskStore()
        await tasks.create_task(TaskCreate(task_id="setup-task", type="job"))
        provider = ScriptedModelProvider(
            [ModelStreamEvent.text_delta("unreached"), ModelStreamEvent.completed()]
        )
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="scripted-model"))
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id="setup-failure-close",
                task_id="setup-task",
                messages=[Message.text("user", "hello")],
            )
        )
        observed = []
        try:
            async for event in stream:
                observed.append(event.type)
                if close_on_task_failure and event.type is EventType.TASK_FAILED:
                    break
        finally:
            await stream.aclose()

        assert EventType.TASK_FAILED in observed
        assert (await tasks.load_task("setup-task")).status is TaskStatus.FAILED
        assert provider.requests == []
        session = await sessions.load("setup-failure-close")
        assert session.status in {SessionStatus.FAILED, SessionStatus.INTERRUPTED}
        if not close_on_task_failure:
            assert session.status is SessionStatus.FAILED
        events = await sessions.load_events(session.id)
        assert any(
            event.type in {EventType.SESSION_FAILED, EventType.SESSION_INTERRUPTED}
            for event in events
        )

    asyncio.run(scenario())
