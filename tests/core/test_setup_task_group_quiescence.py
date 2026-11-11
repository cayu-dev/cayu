"""An initial setup failure must retire a task group's execution obligation."""

import asyncio

import pytest
from tests.core.test_workspace_mutation_receipts import _portable_environment_spec

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, SessionStatus
from cayu.environments import Environment
from cayu.environments.factory import EnvironmentFactory, EnvironmentFactoryResult
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.sessions import InMemorySessionStore
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tasks import (
    InMemoryTaskStore,
    TaskCreate,
    TaskGraphCreate,
    TaskGraphNode,
    TaskGroupCreate,
    TaskGroupFinalizerStatus,
    TaskGroupPolicy,
    TaskGroupQuiescencePolicy,
    TaskGroupQuiescenceStatus,
    TaskStatus,
)


class _TranscriptFailure:
    fail_transcript = True

    async def replace_initial_transcript_messages(self, *args, **kwargs):
        if self.fail_transcript:
            self.fail_transcript = False
            raise ConnectionError("initial transcript publication unavailable")
        return await super().replace_initial_transcript_messages(*args, **kwargs)


class _SetupMemoryStore(_TranscriptFailure, InMemorySessionStore):
    invocation_lifecycle_command_version = 1
    terminal_interaction_publication_version = 1


class _SetupSQLiteStore(_TranscriptFailure, SQLiteSessionStore):
    invocation_lifecycle_command_version = 1
    terminal_interaction_publication_version = 1


class _SetupFactory(EnvironmentFactory):
    def __init__(self, *, fail):
        self.fail = fail
        self.releases = 0

    async def create(self, request):
        if self.fail:
            raise ConnectionError("environment allocation unavailable")

        async def release(request):
            self.releases += 1

        return EnvironmentFactoryResult(
            environment=Environment(_portable_environment_spec(request.environment_name)),
            release=release,
        )


@pytest.mark.parametrize("close_on_failure", [False, True])
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "failure", [None, "factory", "transcript"], ids=["success", "factory", "transcript"]
)
def test_setup_failure_releases_task_group_after_cleanup(
    failure, backend, tmp_path, close_on_failure
):
    async def scenario():
        if backend == "memory":
            sessions = InMemorySessionStore() if failure != "transcript" else _SetupMemoryStore()
        else:
            store_type = SQLiteSessionStore if failure != "transcript" else _SetupSQLiteStore
            sessions = store_type(tmp_path / "setup.sqlite")
        tasks = InMemoryTaskStore()
        factory = _SetupFactory(fail=failure == "factory")
        provider = ScriptedModelProvider(
            [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]
        )
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            _portable_environment_spec("dynamic"), factory, default=True
        )
        app.register_agent(AgentSpec(name="worker", model="scripted-model"))
        try:
            await app.create_task_group(
                TaskGroupCreate(
                    group_id="setup-group",
                    graph=TaskGraphCreate(
                        graph_id="setup-graph",
                        nodes=tuple(
                            TaskGraphNode(task=TaskCreate(task_id=identity, type="work"))
                            for identity in ("winner", "setup", "finalize")
                        ),
                    ),
                    member_task_ids=("winner", "setup"),
                    policy=TaskGroupPolicy(kind="first_success"),
                    quiescence=TaskGroupQuiescencePolicy(timeout_seconds=60),
                    finalizer_task_id="finalize",
                )
            )
            stream = app.run(
                RunRequest(
                    agent_name="worker",
                    session_id="setup-session",
                    task_id="setup",
                    messages=[Message.text("user", "work")],
                )
            )
            events = []
            try:
                async for event in stream:
                    events.append(event)
                    if close_on_failure and event.type is EventType.SESSION_FAILED:
                        break
            finally:
                await stream.aclose()
            failed = failure is not None
            assert events[-1].type is (
                EventType.SESSION_FAILED if failed else EventType.SESSION_COMPLETED
            )
            assert EventType.TASK_STARTED in [event.type for event in events]
            assert (await tasks.load_task("setup")).status is (
                TaskStatus.FAILED if failed else TaskStatus.COMPLETED
            )
            assert (await sessions.load("setup-session")).status is (
                SessionStatus.FAILED if failed else SessionStatus.COMPLETED
            )
            assert len(provider.requests) == (0 if failed else 1)
            assert await app.drain_recovery_cleanups(timeout_s=5)
            assert await app.drain_environment_cleanups(timeout_s=5)
            if failure == "transcript":
                assert factory.releases == 1
            session = await sessions.load("setup-session")
            active = active_invocation_execution_profile_from_checkpoint(
                await sessions.load_checkpoint(session.id)
            )
            assert active is not None and session.run_epoch == active.run_epoch + 1

            if failed:
                await tasks.complete_task("winner", {"ok": True})
            snapshot = await app.reconcile_task_group("setup-group")
            assert snapshot.quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT, (
                "The initial setup owner returned after cleanup, but the group still "
                f"retains its execution: {snapshot.quiescence}"
            )
            assert snapshot.quiescence.finalizer_status is TaskGroupFinalizerStatus.RELEASED
            assert (await tasks.load_task("finalize")).status is TaskStatus.PENDING
        finally:
            await app.aclose()
            if backend == "sqlite":
                await sessions.close()

    asyncio.run(scenario())
