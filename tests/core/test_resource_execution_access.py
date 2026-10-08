from __future__ import annotations

import asyncio

import pytest

from cayu import AgentSpec, CayuApp, Message, ModelStreamEvent, RunRequest, ScriptedModelProvider
from cayu.resource_access import ResourceAccessPolicy, current_binding
from cayu.sessions.access import (
    SessionAccessDenied,
    SessionAccessRule,
    SessionAccessScope,
    SessionAccessSelector,
)
from cayu.sessions.base import ResumeRequest
from cayu.storage.sqlite import SQLiteSessionStore


def scope(org):
    rule = SessionAccessRule(selectors=(SessionAccessSelector(key="organization", values=(org,)),))
    return SessionAccessScope(read=(rule,), create=(rule,), execute=(rule,), inspect_state=(rule,))


class Policy(ResourceAccessPolicy):
    authority = "test-policy-v1"

    def __init__(self):
        self.current = scope("acme")
        self.calls = 0
        self.failure = None

    async def resolve(self, subject):
        assert subject == "alice"
        self.calls += 1
        if self.failure:
            raise self.failure
        return self.current


def app_for(store, policy):
    app = CayuApp(session_store=store, resource_access_policy=policy)
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("ok"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="shared", model="scripted-model"))
    return app


def test_execution_binding_survives_restart_and_revocation(tmp_path):
    async def run():
        path = tmp_path / "access.db"
        store = SQLiteSessionStore(path)
        policy = Policy()
        app = app_for(store, policy)
        access = await app.access("alice")
        events = [
            event
            async for event in access.run(
                RunRequest(
                    agent_name="shared",
                    session_id="own",
                    messages=[Message.text("user", "hi")],
                    labels={"organization": "acme"},
                )
            )
        ]
        assert events
        stored = await store.load("own")
        assert stored.invocation.resource_access.subject == "alice"
        assert current_binding() is None
        await store.close()
        store = SQLiteSessionStore(path)
        policy.current = SessionAccessScope()
        restarted = app_for(store, policy)
        with pytest.raises(SessionAccessDenied):
            _ = [
                event
                async for event in restarted.resume(
                    ResumeRequest(session_id="own", messages=[Message.text("user", "again")])
                )
            ]
        assert current_binding() is None
        missing_policy = app_for(store, None)
        with pytest.raises(SessionAccessDenied):
            _ = [
                event
                async for event in missing_policy.resume(
                    ResumeRequest(session_id="own", messages=[Message.text("user", "again")])
                )
            ]
        await store.close()

    asyncio.run(run())


def test_expansion_does_not_widen_existing_handle(tmp_path):
    async def run():
        store = SQLiteSessionStore(tmp_path / "access.db")
        policy = Policy()
        app = app_for(store, policy)
        access = await app.access("alice")
        all_rule = SessionAccessRule(allow_all=True)
        policy.current = SessionAccessScope(
            read=(all_rule,), create=(all_rule,), execute=(all_rule,)
        )
        with pytest.raises(SessionAccessDenied):
            _ = [
                event
                async for event in access.run(
                    RunRequest(
                        agent_name="shared",
                        session_id="foreign",
                        messages=[],
                        labels={"organization": "other"},
                    )
                )
            ]
        assert await store.load("foreign") is None
        assert current_binding() is None
        await store.close()

    asyncio.run(run())


def test_stream_context_is_not_exposed_to_consumer(tmp_path):
    async def run():
        from contextlib import aclosing

        store = SQLiteSessionStore(tmp_path / "stream.db")
        policy = Policy()
        app = app_for(store, policy)
        access = await app.access("alice")
        async with aclosing(
            access.run(
                RunRequest(
                    agent_name="shared",
                    messages=[Message.text("user", "hi")],
                    labels={"organization": "acme"},
                )
            )
        ) as events:
            async for _event in events:
                assert current_binding() is None
        assert current_binding() is None
        await store.close()

    asyncio.run(run())


def test_scoped_task_retains_execution_authority():
    from cayu.tasks.creation import TaskCreate

    async def run():
        from cayu.tasks.memory import InMemoryTaskStore

        policy = Policy()
        app = CayuApp(resource_access_policy=policy, task_store=InMemoryTaskStore())
        access = await app.access("alice")
        task = await access.tasks.create(
            TaskCreate(type="test", title="classified"), labels={"organization": "acme"}
        )
        assert task.invocation.resource_access.subject == "alice"
        assert task.invocation.access_labels == {"organization": "acme"}
        assert current_binding() is None

    asyncio.run(run())


def test_revoked_task_never_calls_handler_and_releases_worker_authority():
    from cayu.tasks.creation import TaskCreate
    from cayu.tasks.memory import InMemoryTaskStore
    from cayu.tasks.worker import run_task_worker

    async def run():
        policy = Policy()
        store = InMemoryTaskStore()
        app = CayuApp(resource_access_policy=policy, task_store=store)
        access = await app.access("alice")
        task = await access.tasks.create(
            TaskCreate(type="test", title="classified"), labels={"organization": "acme"}
        )
        policy.current = SessionAccessScope()
        calls = []

        async def handler(app, task, worker_id):
            calls.append(task.id)

        assert await run_task_worker(app, store, handler, worker_id="worker", max_tasks=1) == 1
        assert not calls
        stored = await store.load_task(task.id)
        assert stored.status == "failed"
        assert stored.worker_id is None and stored.lease_expires_at is None
        assert current_binding() is None

    asyncio.run(run())


def test_foreground_delegation_preserves_authority():
    from cayu.sessions.queries import SessionQuery
    from cayu.tools.subagents import SubagentSpec, SubagentTool

    async def run():
        policy = Policy()
        app = CayuApp(resource_access_policy=policy, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(
                            id="call",
                            name="subagent",
                            arguments={"agent": "reviewer", "task": "Review"},
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [
                        ModelStreamEvent.text_delta("reviewed"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ],
                    [
                        ModelStreamEvent.text_delta("done"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ],
                ]
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="parent", model="scripted-model"),
            tools=[SubagentTool(app, agents={"reviewer": SubagentSpec(agent_name="reviewer")})],
        )
        app.register_agent(AgentSpec(name="reviewer", model="scripted-model"))
        access = await app.access("alice")
        events = [
            event
            async for event in access.run(
                RunRequest(
                    agent_name="parent",
                    session_id="parent",
                    messages=[Message.text("user", "delegate")],
                    labels={"organization": "acme"},
                )
            )
        ]
        assert events[-1].type == "session.completed", [(e.type, e.payload) for e in events]
        parent = await app.session_store.load("parent")
        children = (
            await app.session_store.list_sessions(SessionQuery(parent_session_id="parent"))
        ).sessions
        assert len(children) == 1
        assert children[0].status == "completed"
        assert children[0].labels == parent.labels
        assert children[0].invocation.resource_access == parent.invocation.resource_access
        assert current_binding() is None

    asyncio.run(run())


def test_decision_revision_expiry_outage_and_family_separation(tmp_path):
    from datetime import UTC, datetime, timedelta

    from cayu.resource_access import ResourceAccessDecision, ResourceAccessGrant
    from cayu.sessions.base import RunRequest, SessionIdentity

    async def run():
        store = SQLiteSessionStore(tmp_path / "decisions.db")
        policy = Policy()
        policy.current = ResourceAccessDecision(ResourceAccessGrant(sessions=scope("acme")), 4)
        app = app_for(store, policy)
        access = await app.access("alice")
        await store.create(
            RunRequest(
                agent_name="shared", messages=[], session_id="own", labels={"organization": "acme"}
            ),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        assert (await access.sessions.load("own")).id == "own"
        assert not (await access._current("tasks")).read
        policy.current = ResourceAccessDecision(scope("acme"), 3)
        with pytest.raises(SessionAccessDenied):
            await access.sessions.load("own")
        policy.current = ResourceAccessDecision(
            scope("acme"), 5, datetime.now(UTC) - timedelta(seconds=1)
        )
        with pytest.raises(SessionAccessDenied):
            await access.sessions.load("own")
        policy.failure = RuntimeError("policy unavailable")
        with pytest.raises(RuntimeError, match="policy unavailable"):
            await access.sessions.load("own")
        await store.close()

    asyncio.run(run())


def test_stream_revocation_stops_producer_and_recovery_allows_only_settlement():
    from cayu._resource_access_binding import ResourceExecutionBinding
    from cayu.resource_access import encode_scope, guard_stream, recovery_access, require_dispatch
    from cayu.sessions.base import InMemorySessionStore, SessionIdentity

    async def run():
        policy = Policy()
        binding = ResourceExecutionBinding(
            authority=policy.authority, subject="alice", admitted_json=encode_scope(policy.current)
        )
        steps = []

        async def producer():
            try:
                steps.append("first")
                yield 1
                steps.append("second")
                yield 2
            finally:
                steps.append("settled")

        stream = guard_stream(
            producer(), binding=binding, policy=policy, labels={"organization": "acme"}
        )
        assert await anext(stream) == 1
        policy.current = SessionAccessScope()
        with pytest.raises(SessionAccessDenied):
            await anext(stream)
        assert steps == ["first", "settled"]
        assert current_binding() is None
        store = InMemorySessionStore()
        session = await store.create(
            RunRequest(
                agent_name="shared", messages=[], session_id="own", labels={"organization": "acme"}
            ),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        from cayu.sessions.invocation import SessionInvocation

        session = session.model_copy(
            update={
                "invocation": SessionInvocation.model_validate(
                    {
                        **session.invocation.model_dump(mode="json"),
                        "resource_access": binding.model_dump(mode="json"),
                    }
                )
            }
        )
        async with recovery_access(session, policy, store):
            # The retained binding permits cleanup bookkeeping, not new effects.
            assert current_binding() == binding
            with pytest.raises(SessionAccessDenied):
                await require_dispatch()
        assert current_binding() is None

    asyncio.run(run())


def test_durable_worker_run_inherits_task_classification_after_restart(tmp_path):
    from cayu.storage.sqlite import SQLiteTaskStore
    from cayu.tasks.creation import TaskCreate
    from cayu.tasks.worker import run_task_worker

    async def run():
        path = tmp_path / "worker.db"
        tasks = SQLiteTaskStore(path)
        sessions = SQLiteSessionStore(path)
        policy = Policy()
        app = CayuApp(session_store=sessions, task_store=tasks, resource_access_policy=policy)
        await (await app.access("alice")).tasks.create(
            TaskCreate(task_id="job", type="test", assigned_agent_name="shared"),
            labels={"organization": "acme"},
        )
        await tasks.close()
        await sessions.close()
        tasks = SQLiteTaskStore(path)
        sessions = SQLiteSessionStore(path)
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            resource_access_policy=policy,
            enable_logging=False,
        )
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.text_delta("done"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                ]
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="shared", model="scripted-model"))
        observed = []

        async def handler(app, task, worker_id):
            observed.extend(
                [
                    event
                    async for event in app.run(
                        RunRequest(
                            agent_name="shared",
                            session_id="run",
                            task_id=task.id,
                            task_worker_id=worker_id,
                            task_lease_expires_at=task.lease_expires_at,
                            messages=[Message.text("user", "hello")],
                        )
                    )
                ]
            )

        await run_task_worker(app, tasks, handler, worker_id="worker", max_tasks=1)
        assert observed and observed[-1].type == "session.completed", (
            await tasks.load_task("job")
        ).error
        session = await sessions.load("run")
        task = await tasks.load_task("job")
        assert session.labels == task.invocation.access_labels == {"organization": "acme"}
        assert session.invocation.resource_access == task.invocation.resource_access
        assert task.status == "completed"
        await sessions.close()
        await tasks.close()

    asyncio.run(run())


@pytest.mark.parametrize("revoke", [False, True])
def test_scoped_workspace_effect_dispatch_and_revocation(tmp_path, revoke):
    from cayu import Environment, EnvironmentSpec
    from cayu.tools.files import ReadFileTool, WriteFileTool
    from cayu.workspaces.local import LocalWorkspace

    async def run():
        policy = Policy()
        app = CayuApp(resource_access_policy=policy, enable_logging=False)
        app.register_environment(
            Environment(EnvironmentSpec(name="files"), workspace=LocalWorkspace(tmp_path)),
            default=True,
        )
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(
                            id="write",
                            name="write_file",
                            arguments={"path": "note.txt", "content": "hello", "mode": "create"},
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [
                        ModelStreamEvent.tool_call(
                            id="read", name="read_file", arguments={"path": "note.txt"}
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [
                        ModelStreamEvent.text_delta("done"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ],
                ]
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="shared", model="scripted-model"),
            tools=[WriteFileTool(), ReadFileTool()],
        )
        access = await app.access("alice")
        observed = []

        async def consume():
            async for event in access.run(
                RunRequest(
                    agent_name="shared",
                    session_id="work",
                    labels={"organization": "acme"},
                    messages=[Message.text("user", "write and read")],
                )
            ):
                observed.append(event)
                if revoke and event.type == "model.completed":
                    policy.current = SessionAccessScope()

        if revoke:
            with pytest.raises(SessionAccessDenied):
                await consume()
            assert not (tmp_path / "note.txt").exists()
        else:
            await consume()
            assert observed[-1].type == "session.completed"
            assert (tmp_path / "note.txt").read_text() == "hello"
            failures = [e.payload for e in observed if e.type == "tool.failed"]
            assert not failures
        assert current_binding() is None

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("action", ["approval", "input"])
def test_scoped_foreground_child_pause_without_checkpoint_grant(tmp_path, backend, action):
    from tests.core.test_foreground_subagent_recovery import _identity, _Provider
    from tests.core.test_tool_round_execution_identities import _RecordingTool

    from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
    from cayu.approvals.user_input import UserInputResponse
    from cayu.sessions.base import InMemorySessionStore
    from cayu.sessions.queries import SessionQuery
    from cayu.tools.policy import AlwaysRequireApprovalToolPolicy
    from cayu.tools.subagents import SubagentSpec, SubagentTool
    from cayu.tools.user_input import UserInputTool

    async def run():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "child-pause.db")
        )
        policy = Policy()
        rule = policy.current.read
        policy.current = SessionAccessScope(read=rule, create=rule, execute=rule)

        class RestrictedRecordingTool(_RecordingTool):
            async def run(self, ctx, arguments):
                # Runtime reconciliation may inspect checkpoints; tool code still cannot.
                with pytest.raises(SessionAccessDenied):
                    await store.load_checkpoint("parent")
                with pytest.raises(NotImplementedError, match="operator"):
                    await store.query_latest_interaction_events("parent")
                return await super().run(ctx, arguments)

        protected = RestrictedRecordingTool()
        provider = _Provider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="spawn", name="subagent", arguments={"agent": "child", "task": "work"}
                    ),
                    ModelStreamEvent.completed(),
                ],
                [
                    ModelStreamEvent.tool_call(
                        id="child-action",
                        name="ask_user" if action == "input" else "record",
                        arguments={"question": "Which value?"}
                        if action == "input"
                        else {"value": 7},
                    ),
                    ModelStreamEvent.completed(),
                ],
                [ModelStreamEvent.text_delta("child finished"), ModelStreamEvent.completed()],
                [ModelStreamEvent.text_delta("parent finished"), ModelStreamEvent.completed()],
            ]
        )
        app = CayuApp(session_store=store, resource_access_policy=policy, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="test"),
            tools=[
                SubagentTool(
                    app,
                    agents={"child": SubagentSpec(agent_name="child")},
                    execution_profile_identity=_identity("scoped-paused-child"),
                )
            ],
        )
        app.register_agent(
            AgentSpec(name="child", model="test"),
            tools=[UserInputTool()] if action == "input" else [protected],
            tool_policy=None
            if action == "input"
            else AlwaysRequireApprovalToolPolicy(tools=["record"]),
        )
        try:
            access = await app.access("alice")
            events = [
                event
                async for event in access.run(
                    RunRequest(
                        session_id="parent",
                        agent_name="parent",
                        messages=[Message.text("user", "delegate")],
                        labels={"organization": "acme"},
                    )
                )
            ]
            assert events[-1].type == "session.interrupted", [(e.type, e.payload) for e in events]
            assert any(event.type == "interaction.paused" for event in events)
            assert protected.values == []
            children = (
                await store.list_sessions(SessionQuery(parent_session_id="parent"))
            ).sessions
            assert len(children) == 1
            child = children[0]
            parent = await store.load("parent")
            assert child.invocation.resource_access == parent.invocation.resource_access
            checkpoint = await store.load_checkpoint("parent")
            assert checkpoint["foreground_child_wait"]["child_session_id"] == child.id
            child_events = await store.load_events(child.id)
            if action == "input":
                pending = next(e for e in child_events if e.type == "session.awaiting_user_input")
                resolution = app.resolve_user_input(
                    UserInputResponse(
                        session_id=child.id,
                        input_id=pending.payload["input_id"],
                        answer="7",
                    )
                )
            else:
                pending = next(e for e in child_events if e.type == "tool.call.approval_requested")
                resolution = app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=child.id,
                        approval_id=pending.payload["approval"]["approval_id"],
                        tool_round_id=pending.payload["tool_round_id"],
                        tool_call_id=pending.payload["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                    )
                )
            _ = [event async for event in resolution]
            assert (await store.load(child.id)).status == "completed"
            assert (await store.load("parent")).status == "completed"
            assert len(provider.requests) == 4
            assert protected.values == ([] if action == "input" else [7])
            assert current_binding() is None
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(run())
