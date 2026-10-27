"""A durable subagent is submitted under Cayu Cloud's managed settings too."""

import asyncio

from tests.core.test_durable_subagents import (
    _collect,
    _DurableSubagentProvider,
    _RecordingPreparedSubagentDispatcher,
    _register_durable_subagent_agents,
)
from tests.core.test_managed_settings_controller import RELEASE, FakeCloud, _controller

from cayu import CayuApp, EventType, Message, RunRequest
from cayu.releases import ReleaseIdentity
from cayu.sessions import InMemorySessionStore, SessionQuery
from cayu.tasks import InMemoryTaskStore, TaskQuery


def test_durable_subagent_submission_matches_its_seed_with_managed_settings() -> None:
    async def run() -> None:
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore()
        cloud = FakeCloud()
        cloud.values = {"config.run.retry_policy.max_attempts": 3}
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=_RecordingPreparedSubagentDispatcher(tasks),
            enable_logging=False,
            managed_settings=_controller(cloud),
            release_identity=ReleaseIdentity(id=RELEASE),
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)
        events = await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id="durable-parent",
                    messages=[Message.text("user", "parent task")],
                )
            )
        )
        assert events[-1].type == EventType.SESSION_COMPLETED
        children = (
            await sessions.list_sessions(SessionQuery(parent_session_id="durable-parent"))
        ).sessions
        assert len(children) == 1
        assert len(await tasks.list_tasks(TaskQuery())) == 1
        # Cloud changes the retry policy before a worker runs the queued child; the
        # child keeps the policy it was prepared with and still matches its profile.
        cloud.set(config__run__retry_policy__max_attempts=4)
        await app.managed_settings_controller.poll_once()
        handle = await app.dispatcher.process_next(app, worker_id="worker")
        assert not handle.metadata.get("requeued", False)
        await app.aclose()

    asyncio.run(run())
