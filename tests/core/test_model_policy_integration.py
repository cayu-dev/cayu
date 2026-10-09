"""Policy ownership across queued sessions and application resource lifetimes."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from tests.core.test_durable_subagents import (
    _DurableSubagentProvider,
    _register_durable_subagent_agents,
)
from tests.core.test_model_policy_runtime import Channel, make_app
from tests.core.test_model_policy_runtime import store_factory as store_factory

from cayu import CayuApp, InMemorySessionStore, Message, RunRequest
from cayu.model_policy import ModelPolicy, ModelPolicyController, ModelPolicyStore
from cayu.server import ServerConfig, create_server
from cayu.sessions.outcomes import run_to_completion
from cayu.storage.migrations import SchemaMode
from cayu.tasks.dispatch import TaskStoreDispatcher
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import TaskStatus

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.parametrize("exit_mode", ["normal", "failure", "cancel", "startup"])
@pytest.mark.parametrize("entrance", ["create", "mount"])
async def test_server_resources_enclose_policy_workers(store_factory, exit_mode, entrance):
    native, channel = store_factory(), Channel()
    live = False
    commands = []

    class ResourceStore(ModelPolicyStore):
        async def execute(self, command):
            assert live, "Policy storage accessed outside its resource lifespan"
            commands.append(command.action)
            result = await native.execute(command)
            if exit_mode == "startup" and command.action == "claim":
                raise OSError("claim acknowledgement lost")
            return result

    app, controller, _ = make_app(ResourceStore(), channel)
    entered = asyncio.Event()

    @asynccontextmanager
    async def resources(server):
        nonlocal live
        live = True
        try:
            yield {"resource": "ready"}
        finally:
            assert controller.status == "stopped"
            assert not app.model_policy._tasks
            assert commands[-1] == "release"
            await native.close()
            live = False

    if entrance == "create":
        server = create_server(
            app, config=ServerConfig.local_development(), fastapi_options={"lifespan": resources}
        )
    else:
        from fastapi import FastAPI

        from cayu.server import OpenAccess, mount_cayu

        server = FastAPI(lifespan=resources)
        mount_cayu(server, app, dashboard=False, access=OpenAccess())

    async def serve():
        async with server.router.lifespan_context(server) as state:
            assert state == {"resource": "ready"}
            assert controller.selection().target.model == "model-a"
            outcome = await run_to_completion(
                app, RunRequest(agent_name="assistant", messages=[Message.text("user", "hello")])
            )
            assert outcome.ok, outcome.error
            assert (await app.session_store.load(outcome.session_id)).model == "model-a"
            entered.set()
            if exit_mode == "failure":
                raise ValueError("body failed")
            if exit_mode == "cancel":
                await asyncio.Event().wait()

    task = asyncio.create_task(serve())
    if exit_mode == "cancel":
        # Startup and the first run can take seconds on a loaded PostgreSQL
        # runner. Wait for the body or its failure instead of a fixed bound.
        entered_wait = asyncio.create_task(entered.wait())
        done, _ = await asyncio.wait({task, entered_wait}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            entered_wait.cancel()
            await task
            pytest.fail("Server body exited before cancellation")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled() and task.cancelling() == 1
    elif exit_mode == "failure":
        with pytest.raises(ValueError, match="body failed"):
            await task
    elif exit_mode == "startup":
        with pytest.raises(OSError, match="claim acknowledgement lost"):
            await task
        assert not entered.is_set()
    else:
        await task
    assert not live


@pytest.mark.parametrize("origin", ["policy", "agent_spec"])
async def test_participant_execution_retains_creation_policy_after_restart(
    store_factory, tmp_path, request, origin
):
    import json

    from tests.core._execution_profile_fixtures import versioned_test_provider_identity
    from tests.core.test_model_policy_runtime import CatalogProvider
    from tests.core.test_participant_identity import CONTEXT, create, registration

    from cayu import AgentSpec, ModelStreamEvent
    from cayu.collaboration.memory import InMemoryCollaborationStore
    from cayu.sessions.context_views import (
        ParticipantSessionCreationRequest,
        ParticipantSessionExecutionRequest,
    )

    policy_store, channel = store_factory(), Channel()
    if origin == "agent_spec":
        channel.model = None
    backend = type(policy_store).__name__
    memory_sessions, memory_collaboration = InMemorySessionStore(), InMemoryCollaborationStore()
    reg = registration()

    class StableProvider(CatalogProvider):
        @property
        def execution_profile_identity(self):
            return versioned_test_provider_identity(self)

    def stores():
        if backend == "InMemoryModelPolicyStore":
            return memory_sessions, memory_collaboration
        if backend == "SQLiteModelPolicyStore":
            from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
            from cayu.storage.sqlite import SQLiteSessionStore

            return SQLiteSessionStore(
                tmp_path / "participants-sessions.db"
            ), SQLiteCollaborationStore(tmp_path / "participants.db")
        from cayu.storage.collaboration_postgres import PostgresCollaborationStore
        from cayu.storage.postgres import PostgresSessionStore

        dsn = request.getfixturevalue("postgres_dsn")
        return (
            PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE),
            PostgresCollaborationStore(dsn, schema_mode=SchemaMode.CREATE),
        )

    def application(sessions, collaboration):
        provider = StableProvider(
            response_factory=lambda request: [
                ModelStreamEvent.text_delta("ok"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        controller = ModelPolicyController(
            store=policy_store,
            agent_name="reviewer",
            provider_name=provider.name,
            scope=channel.scope,
            incarnation=channel.incarnation,
            channel=channel,
        )
        app = CayuApp(
            session_store=sessions,
            collaboration_store=collaboration,
            collaboration=reg,
            model_policy=ModelPolicy([controller]),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="reviewer", model="model-a"))
        return app

    sessions, collaboration = stores()
    first = application(sessions, collaboration)
    try:
        async with first.model_policy_lifespan():
            initialized = await first.initialize_collaboration()
            _, created = await create(first, initialized)
            participant = created.participants[0].reference
            creation = ParticipantSessionCreationRequest(
                creation_key=f"policy-session-{origin}",
                request=RunRequest(agent_name="reviewer", messages=[Message.text("user", "start")]),
            )
            session, receipt = await first.create_participant_session(
                creation, participant=participant, context=CONTEXT
            )
        if backend != "InMemoryModelPolicyStore":
            await sessions.close()
            await collaboration.close()
            sessions, collaboration = stores()
        channel.model, channel.revision = "model-b", 2
        second = application(sessions, collaboration)
        async with second.model_policy_lifespan():
            await second.initialize_collaboration()
            execution = ParticipantSessionExecutionRequest(
                request=creation.request.model_copy(update={"session_id": session.id}),
                session_instance_id=session.instance_id,
                execution_key="activate",
            )
            events = [
                event
                async for event in second.execute_participant_session(
                    execution, participant=participant, context=CONTEXT
                )
            ]
            assert (await sessions.load(session.id)).status == "completed"
            assert (await sessions.load(session.id)).model == "model-a"
            evidence = [e.payload["model_policy"] for e in events if "model_policy" in e.payload]
            if origin == "policy":
                assert evidence and all(
                    value == json.loads(receipt.binding.policy_evidence_json) for value in evidence
                )
            else:
                assert not evidence
            outcome = await run_to_completion(
                second,
                RunRequest(
                    agent_name="reviewer",
                    messages=[Message.text("user", "new")],
                    metadata={
                        "model_policy": (
                            json.loads(receipt.binding.policy_evidence_json)
                            if receipt.binding.policy_evidence_json is not None
                            else {"model": "model-a"}
                        )
                    },
                ),
            )
            assert outcome.ok, outcome.error
            assert (await sessions.load(outcome.session_id)).model == "model-b"
    finally:
        if backend != "InMemoryModelPolicyStore":
            await sessions.close()
        await collaboration.close()
        await policy_store.close()


class PolicySubagentProvider(_DurableSubagentProvider):
    async def get_models(self):
        return [{"id": "model-a"}, {"id": "model-b"}]


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("origin", ["policy", "agent_spec"])
async def test_queued_child_retains_policy_target_and_provenance(
    store_factory, tmp_path, request, restart, origin
):
    policy_store, channel = store_factory(), Channel()
    if origin == "agent_spec":
        channel.model = None
    backend = type(policy_store).__name__
    memory_sessions, memory_tasks = InMemorySessionStore(), InMemoryTaskStore()

    def stores():
        if backend == "InMemoryModelPolicyStore":
            return memory_sessions, memory_tasks
        if backend == "SQLiteModelPolicyStore":
            from cayu.storage.sqlite import SQLiteSessionStore, SQLiteTaskStore

            return SQLiteSessionStore(tmp_path / "sessions.db"), SQLiteTaskStore(
                tmp_path / "tasks.db"
            )
        from cayu.storage.postgres import PostgresSessionStore
        from cayu.storage.tasks_postgres import PostgresTaskStore

        dsn = request.getfixturevalue("postgres_dsn")
        return (
            PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE),
            PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE),
        )

    def build(sessions, tasks, policy):
        provider = PolicySubagentProvider()
        controller = ModelPolicyController(
            store=policy,
            agent_name="reviewer",
            provider_name=provider.name,
            scope=channel.scope,
            incarnation=channel.incarnation,
            channel=channel,
        )
        dispatcher = TaskStoreDispatcher(tasks)
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            model_policy=ModelPolicy([controller]),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        _register_durable_subagent_agents(app)
        return app, controller, dispatcher, provider

    sessions, tasks = stores()
    app, controller, dispatcher, provider = build(sessions, tasks, policy_store)
    await app.start_model_policy()
    try:
        selection = controller.selection()
        original = None if selection is None else selection.evidence
        child_model = "model-a" if origin == "policy" else "model"
        result = await run_to_completion(
            app, RunRequest(agent_name="parent", messages=[Message.text("user", "parent task")])
        )
        assert result.ok, result.error
        queued = await tasks.list_tasks(TaskQuery(status=TaskStatus.PENDING))
        assert len(queued) == 1
        child_id = queued[0].input["dispatch"]["prepared_subagent"]["authority"]["child_session_id"]
        assert (await sessions.load(child_id)).model == child_model
        channel.model, channel.revision = "model-b", 2
        await controller.poll_once()
        if restart:
            await app.stop_model_policy()
            if backend != "InMemoryModelPolicyStore":
                await sessions.close()
                await tasks.close()
            await policy_store.close()
            policy_store = store_factory()
            sessions, tasks = stores()
            app, controller, dispatcher, provider = build(sessions, tasks, policy_store)
            await app.start_model_policy()
        assert controller.selection().target.model == "model-b"
        handle = await dispatcher.process_next(app, worker_id="policy-child-worker")
        assert handle is not None and handle.status.value == "completed"
        assert (await sessions.load(child_id)).model == child_model
        assert provider.requests[-1].model == child_model
        from cayu.runtime._policy_wire import decode

        events = await sessions.load_events(child_id)
        evidence = [
            event.payload["model_policy"] for event in events if "model_policy" in event.payload
        ]
        if original is None:
            assert not evidence
        else:
            assert evidence and all(item == decode(original) for item in evidence)
        # Identical caller metadata is not the authenticated queued handoff.
        ordinary = await run_to_completion(
            app,
            RunRequest(
                agent_name="reviewer",
                messages=[Message.text("user", "durable child task")],
                metadata={"model_policy": {} if original is None else decode(original)},
            ),
        )
        assert ordinary.ok, ordinary.error
        assert (await sessions.load(ordinary.session_id)).model == "model-b"
        assert all(
            event.payload["model_policy"]["model"] == "model-b"
            for event in ordinary.events
            if "model_policy" in event.payload
        )
    finally:
        await app.stop_model_policy()
        if backend != "InMemoryModelPolicyStore":
            await sessions.close()
            await tasks.close()
        await policy_store.close()
