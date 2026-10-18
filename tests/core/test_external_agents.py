from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from cayu import CayuApp, RunRequest
from cayu.budgets import BudgetLimit, ModelPrice, PriceBook
from cayu.deadlines import ExecutionDeadline, ExecutionDeadlineExceeded
from cayu.external_agents.base import ExternalAgentBudgetExceeded, ExternalAgentEvent
from cayu.external_agents.recording import run_external_agent
from cayu.external_agents.tools import CayuToolBridge, ExternalPermissionTool
from cayu.sessions import InterruptSessionRequest
from cayu.storage import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks import InMemoryTaskStore, TaskQuery, TaskStatus
from cayu.tools.policy import AllowAllToolPolicy, StaticToolPolicy
from cayu.vaults.redaction import SecretRedactor


class ScriptedAdapter:
    name = "scripted-external"
    provider_name = "scripted-external"
    model = "test"

    def __init__(self, observations):
        self.observations = observations
        self.consumed = 0
        self.closed = False

    async def stream(self, context):
        try:
            for observation in self.observations:
                self.consumed += 1
                yield observation
        finally:
            self.closed = True


def request(**kwargs):
    return RunRequest(
        agent_name="vendor-agent",
        messages=[],
        execution_deadline=ExecutionDeadline.after(20),
        **kwargs,
    )


def pricing():
    return PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="scripted-external",
                model="test",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("2"),
            ),
        )
    )


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_external_runs_are_sessions_and_tasks_with_deduplicated_spend(backend, tmp_path):
    async def scenario():
        app = CayuApp(
            task_store=InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.db"),
            session_store=None
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.db"),
            enable_logging=False,
        )
        usage = ExternalAgentEvent(
            "usage", {"input_tokens": 1_000_000, "output_tokens": 1_000_000}, "step-1"
        )
        adapter = ScriptedAdapter(
            [usage, usage, ExternalAgentEvent("result", {"output": {"answer": 42}}, "result")]
        )
        result = await run_external_agent(app, request(), adapter, prompt="test")
        assert result.output == {"answer": 42}
        assert (await app.session_store.load(result.session_id)).runtime_name == "external_agent"
        assert (await app.task_store.load_task(result.task_id)).status is TaskStatus.COMPLETED
        cost = await app.get_session_cost(result.session_id, pricing=pricing())
        assert cost.total_cost == Decimal(3)
        assert cost.model_steps == 1
        assert adapter.closed
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_external_session_fork_rejection_discards_private_authority(backend, tmp_path):
    from tests.core.session_store_conformance.support import (
        _assert_exception_omits_private_values,
        _open_store,
    )

    from cayu.sessions import ForkSessionRequest
    from cayu.sessions.records import SessionIdentity

    async def scenario():
        secret = "external-fork-private-source"
        app = CayuApp(
            session_store=await _open_store((backend, tmp_path, None)),
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        try:
            source = await app.session_store.create(
                request(session_id=f"legacy-{secret}-{backend}"),
                identity=SessionIdentity(
                    provider_name="test", model="test", runtime_name="external_agent"
                ),
            )
            public_source_id = app.project_session_id_for_exposure(source.id)
            await app.session_store.register_public_authority_alias(
                public_source_id, field_name="session_id", private_value=source.id
            )
            with pytest.raises(ValueError, match="External agent sessions cannot fork") as raised:
                async for _ in app.fork_session(
                    ForkSessionRequest(source_session_id=public_source_id, session_id="fork")
                ):
                    pass
            _assert_exception_omits_private_values(raised.value, source.id, secret)
            assert await app.session_store.load("fork") is None
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_external_budget_stops_consumption_after_reported_usage():
    async def scenario():
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        adapter = ScriptedAdapter(
            [
                ExternalAgentEvent(
                    "usage", {"input_tokens": 2_000_000, "output_tokens": 0}, "step-1"
                ),
                ExternalAgentEvent("result", {"output": "must not return"}, "result"),
            ]
        )
        with pytest.raises(ExternalAgentBudgetExceeded):
            await run_external_agent(
                app,
                request(budget_limits=(BudgetLimit.usd("1", pricing=pricing()),)),
                adapter,
                prompt="test",
            )
        assert adapter.consumed == 1 and adapter.closed
        (task,) = await app.task_store.list_tasks(TaskQuery())
        assert task.status is TaskStatus.FAILED
        await app.aclose()

    asyncio.run(scenario())


def test_external_notification_budget_allows_run_to_finish():
    async def scenario():
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        adapter = ScriptedAdapter(
            [
                ExternalAgentEvent(
                    "usage", {"input_tokens": 2_000_000, "output_tokens": 0}, "step"
                ),
                ExternalAgentEvent("result", {"output": "done"}, "result"),
            ]
        )
        result = await run_external_agent(
            app,
            request(budget_limits=(BudgetLimit.usd("1", pricing=pricing(), action="notify"),)),
            adapter,
            prompt="test",
        )
        assert result.output == "done" and adapter.consumed == 2
        assert (await app.task_store.load_task(result.task_id)).status is TaskStatus.COMPLETED
        await app.aclose()

    asyncio.run(scenario())


def test_external_failure_settlement_cannot_cancel_its_owner(monkeypatch):
    async def scenario():
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        fail = app.task_store.fail_task

        async def slow_fail(*args, **kwargs):
            await asyncio.sleep(0.3)
            return await fail(*args, **kwargs)

        monkeypatch.setattr(app.task_store, "fail_task", slow_fail)
        adapter = ScriptedAdapter(
            [ExternalAgentEvent("usage", {"input_tokens": 2_000_000}, "over-budget")]
        )
        with pytest.raises(ExternalAgentBudgetExceeded):
            await run_external_agent(
                app,
                request(budget_limits=(BudgetLimit.usd("1", pricing=pricing()),)),
                adapter,
                prompt="test",
            )
        assert asyncio.current_task().cancelling() == 0
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_implicit_external_task_is_unclaimable_during_setup(backend, tmp_path, monkeypatch):
    async def scenario():
        app = CayuApp(
            task_store=InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.db"),
            enable_logging=False,
        )
        create = app.session_store.create
        claims = []

        async def competing_worker(request, **kwargs):
            claims.append(await app.task_store.claim_task("competing-worker"))
            return await create(request, **kwargs)

        monkeypatch.setattr(app.session_store, "create", competing_worker)
        await run_external_agent(
            app,
            request(),
            ScriptedAdapter([ExternalAgentEvent("result", {"output": "done"}, "result")]),
            prompt="test",
        )
        assert claims == [None]
        await app.aclose()

    asyncio.run(scenario())


def test_external_self_recovery_is_refused_before_starting_cleanup():
    from cayu.external_agents import recover_external_agent_run

    async def scenario():
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        quiesced = []

        class Adapter(ScriptedAdapter):
            async def stream(self, context):
                async def quiesce(session):
                    quiesced.append(session.id)
                    raise AssertionError("Self recovery must be refused before quiescence.")

                with pytest.raises(ValueError, match="cannot recover itself"):
                    await recover_external_agent_run(
                        app, context.session_id, quiesce=quiesce, evidence_id="self"
                    )
                yield ExternalAgentEvent("result", {"output": "done"}, "result")

        await run_external_agent(app, request(), Adapter([]), prompt="test")
        assert not quiesced
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("operator", [False, True])
def test_external_stop_joins_adapter_cleanup_and_settles_task(operator):
    async def scenario():
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        session_ids = []

        class Stall(ScriptedAdapter):
            async def stream(self, context):
                session_ids.append(context.session_id)
                entered.set()
                try:
                    await asyncio.Event().wait()
                    yield
                finally:
                    cleaning.set()
                    await release.wait()
                    self.closed = True

        adapter = Stall([])
        boundary = ExecutionDeadline.after(20 if operator else 0.3)
        operation = asyncio.create_task(
            run_external_agent(
                app,
                RunRequest(agent_name="external", messages=[], execution_deadline=boundary),
                adapter,
                prompt="test",
            )
        )
        await asyncio.wait_for(entered.wait(), 5)
        if operator:
            async for _ in app.interrupt_session(
                InterruptSessionRequest(session_id=session_ids[0])
            ):
                pass
        await asyncio.wait_for(cleaning.wait(), 5)
        assert not operation.done()
        release.set()
        with pytest.raises(asyncio.CancelledError if operator else ExecutionDeadlineExceeded):
            await operation
        assert adapter.closed
        (task,) = await app.task_store.list_tasks(TaskQuery())
        assert task.status is (TaskStatus.CANCELLED if operator else TaskStatus.FAILED)
        await app.aclose()

    asyncio.run(scenario())


def test_external_text_redacts_secrets_split_between_sdk_chunks():
    async def scenario():
        app = CayuApp(
            task_store=InMemoryTaskStore(),
            enable_logging=False,
            secret_redactor=SecretRedactor("secret-token-123"),
        )
        adapter = ScriptedAdapter(
            [
                ExternalAgentEvent("text", {"text": "prefix secret-to"}, "a"),
                ExternalAgentEvent("text", {"text": "ken-123 suffix"}, "b"),
                ExternalAgentEvent("result", {"output": "done"}, "r"),
            ]
        )
        result = await run_external_agent(app, request(), adapter, prompt="test")
        events = await app.session_store.load_events(result.session_id)
        text = "".join(e.payload.get("text", "") for e in events)
        assert "secret-token-123" not in text
        assert "prefix " in text and " suffix" in text
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("allow", [False, True])
def test_external_permission_bridge_uses_native_tool_policy(allow):
    async def scenario():
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        bridge = CayuToolBridge(
            app,
            name="permissions",
            tools=[ExternalPermissionTool("Read")],
            policy=AllowAllToolPolicy() if allow else StaticToolPolicy(allow=()),
        )

        class Adapter(ScriptedAdapter):
            async def stream(self, context):
                allowed = await bridge.authorize(
                    "Read",
                    {"path": "test"},
                    "read-1",
                    parent_session_id=context.session_id,
                    execution_deadline=context.execution_deadline,
                )
                yield ExternalAgentEvent("result", {"output": allowed}, "result")

        result = await run_external_agent(app, request(), Adapter([]), prompt="test")
        assert result.output is allow
        await app.aclose()

    asyncio.run(scenario())


def test_external_bridge_keeps_private_arguments_out_of_events_and_transcript():
    from cayu.tools import Tool, ToolEffect, ToolResult, ToolSpec

    async def scenario():
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        calls = []
        private = "private document contents that are not a registered secret"

        class PrivateTool(Tool):
            spec = ToolSpec(name="private", effect=ToolEffect.NONE)

            @property
            def _publish_arguments(self):
                return False

            async def run(self, ctx, args):
                calls.append((ctx.session_id, args))
                return ToolResult(content="stored")

        bridge = CayuToolBridge(
            app, name="private", tools=[PrivateTool()], policy=AllowAllToolPolicy()
        )

        class Adapter(ScriptedAdapter):
            async def stream(self, context):
                result = await bridge.call(
                    "private",
                    {"text": private},
                    parent_session_id=context.session_id,
                    execution_deadline=context.execution_deadline,
                    call_id="private-call",
                )
                assert not result.is_error
                for kind in ("tool_call", "tool_result"):
                    for item_type in ("mcpToolCall", "commandExecution", "fileChange"):
                        yield ExternalAgentEvent(
                            kind,
                            {
                                "id": "private-call",
                                "type": item_type,
                                "server": "native",
                                "tool": "private",
                                "status": "completed",
                                "arguments": {"text": private},
                                "command": private,
                                "changes": [{"content": private}],
                                "output": private,
                                "content": private,
                            },
                            f"{kind}-{item_type}",
                        )
                yield ExternalAgentEvent("result", {"output": "done"}, "result")

        result = await run_external_agent(app, request(), Adapter([]), prompt="test")
        [(session_id, arguments)] = calls
        assert arguments == {"text": private}
        events = await app.session_store.load_events(session_id)
        transcript = await app.session_store.load_transcript(session_id)
        assert private not in str([event.model_dump() for event in events])
        assert private not in str([message.model_dump() for message in transcript])
        parent_events = await app.session_store.load_events(result.session_id)
        assert private not in str([event.model_dump() for event in parent_events])
        observations = [
            event.payload
            for event in parent_events
            if str(event.type).startswith("custom.external_agent.tool_")
        ]
        assert len(observations) == 6
        assert all(
            item["name"] == "private" and item["call_id"] == "private-call" for item in observations
        )
        assert not bridge.calls
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("allow", [False, True])
def test_external_bridge_preserves_native_mcp_manifest_policy(allow):
    from tests.core.test_mcp import _fake_toolset

    from cayu.runtime import McpManifestPolicy, McpManifestPolicyAction

    async def scenario():
        toolset = _fake_toolset()
        app = CayuApp(
            task_store=InMemoryTaskStore(),
            enable_logging=False,
            mcp_manifest_policy=McpManifestPolicy(
                on_first_seen=(
                    McpManifestPolicyAction.ALLOW if allow else McpManifestPolicyAction.BLOCK
                )
            ),
        )
        bridge = CayuToolBridge(app, name="mcp", tools=toolset.tools, policy=AllowAllToolPolicy())

        class Adapter(ScriptedAdapter):
            async def stream(self, context):
                result = await bridge.call(
                    toolset.tools[0].name,
                    {"text": "private data"},
                    parent_session_id=context.session_id,
                    execution_deadline=context.execution_deadline,
                    call_id="mcp-call",
                )
                assert result.is_error is not allow
                yield ExternalAgentEvent("result", {"output": "done"}, "result")

        await run_external_agent(app, request(), Adapter([]), prompt="test")
        assert toolset.session.calls == ([("echo", {"text": "private data"})] if allow else [])
        await app.aclose()

    asyncio.run(scenario())


def test_setup_failure_settles_implicitly_owned_task_and_protects_input(monkeypatch):
    from cayu.external_agents import ExternalAgentError

    async def scenario():
        app = CayuApp(
            task_store=InMemoryTaskStore(),
            enable_logging=False,
            secret_redactor=SecretRedactor("private-model-key"),
        )
        create = app.session_store.create
        seen = []

        async def commit_then_fail(request, **kwargs):
            session = await create(request, **kwargs)
            seen.append(session.id)
            raise RuntimeError("private-model-key transport lost")

        monkeypatch.setattr(app.session_store, "create", commit_then_fail)
        adapter = ScriptedAdapter([])
        with pytest.raises(ExternalAgentError) as raised:
            await run_external_agent(app, request(), adapter, prompt="private-model-key")
        assert "private-model-key" not in str(raised.value)
        (task,) = await app.task_store.list_tasks(TaskQuery())
        assert task.status is TaskStatus.FAILED
        state = await app.session_store.load_state(seen[0])
        assert "private-model-key" not in str(state.model_dump())
        assert adapter.consumed == 0
        with pytest.raises(ValueError, match="max_steps"):
            await run_external_agent(app, request(max_steps=5), adapter, prompt="x")
        assert len(await app.task_store.list_tasks(TaskQuery())) == 1
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("abandon", [False, True, "inflight"])
def test_native_approval_can_resolve_live_call_but_not_revoked_call(abandon):
    from cayu.approvals import PendingToolApprovalEventView
    from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
    from cayu.events import EventType
    from cayu.tools import Tool, ToolEffect, ToolResult, ToolSpec
    from cayu.tools.policy import AlwaysRequireApprovalToolPolicy

    async def scenario():
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        calls = []
        entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class Read(Tool):
            spec = ToolSpec(name="read", effect=ToolEffect.NONE)

            async def run(self, ctx, args):
                calls.append(ctx.session_id)
                if abandon == "inflight":
                    entered.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        cleaning.set()
                        await release.wait()
                return ToolResult(content="native-result")

        bridge = CayuToolBridge(
            app, name="native", tools=[Read()], policy=AlwaysRequireApprovalToolPolicy()
        )

        class Adapter(ScriptedAdapter):
            async def stream(self, context):
                result = await bridge.call(
                    "read",
                    {},
                    parent_session_id=context.session_id,
                    execution_deadline=context.execution_deadline,
                    call_id="read",
                )
                yield ExternalAgentEvent("result", {"output": result.content}, "final")

        running = asyncio.create_task(
            run_external_agent(app, request(), Adapter([]), prompt="review")
        )
        async with asyncio.timeout(10):
            while True:
                if bridge.active:
                    child_id = next(iter(bridge.active))
                    if await app.session_store.load(child_id) is None:
                        await asyncio.sleep(0.01)
                        continue
                    events = await app.session_store.load_events(child_id)
                    pending_event = next(
                        (e for e in events if e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED),
                        None,
                    )
                    if pending_event is not None:
                        break
                await asyncio.sleep(0.01)
        approval = PendingToolApprovalEventView.from_event(pending_event)
        if abandon is True:
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running

        async def resolve():
            return [
                e
                async for e in app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=child_id,
                        approval_id=approval.approval_id,
                        tool_round_id=approval.tool_round_id,
                        tool_call_id=approval.tool_call_id,
                        decision=ToolApprovalDecision.APPROVE,
                    )
                )
            ]

        if abandon == "inflight":
            resolver = asyncio.create_task(resolve())
            await asyncio.wait_for(entered.wait(), 5)
            running.cancel()
            await asyncio.wait_for(cleaning.wait(), 5)
            assert not running.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(running, 5)
            await asyncio.gather(resolver, return_exceptions=True)
            assert not bridge.invocations and not bridge.active
        elif abandon:
            await resolve()
            assert not calls
        else:
            await resolve()
            assert (await running).output == "native-result"
            assert calls == [child_id]
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_worker_restart_joins_external_harness_and_allocates_fresh_session(backend, tmp_path):
    from cayu.tasks import TaskCreate, run_task_worker

    async def scenario():
        def application():
            return CayuApp(
                task_store=InMemoryTaskStore()
                if backend == "memory"
                else SQLiteTaskStore(tmp_path / "tasks.db"),
                session_store=None
                if backend == "memory"
                else SQLiteSessionStore(tmp_path / "sessions.db"),
                enable_logging=False,
            )

        app = application()
        created = await app.create_task(TaskCreate(type="vendor", restartable=True))
        entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        sessions = []

        class Adapter(ScriptedAdapter):
            async def stream(self, context):
                sessions.append(context.session_id)
                entered.set()
                try:
                    await asyncio.Event().wait()
                    yield
                finally:
                    cleaning.set()
                    await release.wait()

        adapter = Adapter([])

        async def handle(current_app, task, worker):
            await run_external_agent(
                current_app,
                request(
                    task_id=task.id,
                    task_worker_id=worker,
                    task_lease_expires_at=task.lease_expires_at,
                ),
                adapter,
                prompt="review",
            )

        worker = asyncio.create_task(
            run_task_worker(
                app, app.task_store, handle, worker_id="first", max_tasks=1, poll_interval_s=0.01
            )
        )
        await asyncio.wait_for(entered.wait(), 10)
        worker.cancel()
        await asyncio.wait_for(cleaning.wait(), 10)
        assert not worker.done()
        assert await app.task_store.claim_task("competitor") is None
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(worker, 10)
        pending = await app.task_store.load_task(created.id)
        assert pending.status is TaskStatus.PENDING and pending.restart_count == 1
        assert pending.session_id is None
        if backend == "sqlite":
            await app.aclose()
            app = application()
        adapter = ScriptedAdapter([ExternalAgentEvent("result", {"output": "fresh"}, "r")])
        await run_task_worker(app, app.task_store, handle, worker_id="second", max_tasks=1)
        completed = await app.task_store.load_task(created.id)
        assert completed.status is TaskStatus.COMPLETED
        assert completed.session_id != sessions[0]
        assert (await app.session_store.load(sessions[0])).status.value == "interrupted"
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("cross_process", [False, True])
def test_operator_interruption_stops_only_one_external_worker_task(
    backend, cross_process, tmp_path
):
    from cayu.sessions.records import SessionStatus
    from cayu.tasks import TaskCreate, run_task_worker

    async def scenario():
        app = CayuApp(
            task_store=InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.db"),
            enable_logging=False,
        )
        first = await app.create_task(TaskCreate(type="vendor", restartable=True))
        second = await app.create_task(TaskCreate(type="vendor"))
        entered = asyncio.Event()
        sessions = []

        class Adapter(ScriptedAdapter):
            async def stream(self, context):
                sessions.append(context.session_id)
                if len(sessions) == 1:
                    entered.set()
                    await asyncio.Event().wait()
                yield ExternalAgentEvent("result", {"output": "done"}, "result")

        async def handle(current_app, task, worker_id):
            await run_external_agent(
                current_app,
                request(
                    task_id=task.id,
                    task_worker_id=worker_id,
                    task_lease_expires_at=task.lease_expires_at,
                ),
                Adapter([]),
                prompt="review",
            )

        worker = asyncio.create_task(
            run_task_worker(
                app, app.task_store, handle, worker_id="worker", max_tasks=2, poll_interval_s=0.01
            )
        )
        await asyncio.wait_for(entered.wait(), 5)
        if cross_process:
            await app.session_store.transition_status(
                sessions[0],
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.INTERRUPTING,
            )
        else:
            async for _ in app.interrupt_session(InterruptSessionRequest(session_id=sessions[0])):
                pass
        await asyncio.wait_for(worker, 10)
        cancelled = await app.task_store.load_task(first.id)
        assert cancelled.status is TaskStatus.CANCELLED and cancelled.restart_count == 0
        assert (await app.task_store.load_task(second.id)).status is TaskStatus.COMPLETED
        assert len(sessions) == 2
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_abandoned_external_recovery_requires_positive_quiescence(backend, tmp_path):
    from tests.core.test_mcp import FakeProvider

    from cayu import AgentSpec
    from cayu.events import EventType
    from cayu.external_agents import recover_external_agent_run
    from cayu.sessions import IncompleteSessionRecoveryRequest
    from cayu.sessions.records import SessionIdentity, SessionStatus
    from cayu.sessions.requests import run_request_with_task_invocation
    from cayu.tasks import TaskCreate

    async def scenario():
        app = CayuApp(
            task_store=InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.db"),
            session_store=None
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.db"),
            enable_logging=False,
        )
        provider = FakeProvider([])
        provider.name = "test"
        app.register_provider(provider)
        app.register_agent(AgentSpec(name="vendor-agent", provider_name="test", model="test"))
        task = await app.create_task(TaskCreate(type="external_agent"))
        bound = run_request_with_task_invocation(
            request(task_id=task.id), await app.task_store.load_invocation_snapshot(task.id)
        )
        session = await app.session_store.create(
            bound,
            identity=SessionIdentity(
                provider_name="test", model="test", runtime_name="external_agent"
            ),
        )
        session = await app.session_store.transition_status(
            session.id, from_statuses={SessionStatus.PENDING}, to_status=SessionStatus.RUNNING
        )
        await app._session_tasks.start_task(
            task_id=task.id, session=session, worker_id=None, lease_expires_at=None
        )

        await app.recover_abandoned_sessions(inactive_for_seconds=0)
        assert (await app.session_store.load(session.id)).status is SessionStatus.RUNNING
        with pytest.raises(ValueError, match="recover_external_agent_run"):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session.id, inactive_for_seconds=0)
            )
        assert not await app.session_store.load_events(session.id)

        async def unproven(_):
            raise RuntimeError("old guest still exists")

        with pytest.raises(RuntimeError, match="old guest"):
            await recover_external_agent_run(
                app, session.id, quiesce=unproven, evidence_id="receipt"
            )
        assert (await app.task_store.load_task(task.id)).status is TaskStatus.RUNNING
        assert (await app.session_store.load(session.id)).status is SessionStatus.RUNNING
        proofs = []

        async def stopped(current):
            assert current.instance_id == session.instance_id
            proofs.append(current.id)

        recovered = await recover_external_agent_run(
            app, session.id, quiesce=stopped, evidence_id="receipt"
        )
        assert recovered.status is SessionStatus.INTERRUPTED
        assert (await app.task_store.load_task(task.id)).status is TaskStatus.CANCELLED
        await recover_external_agent_run(app, session.id, quiesce=unproven, evidence_id="receipt")
        assert proofs == [session.id]
        events = await app.session_store.load_events(session.id)
        assert sum(e.type is EventType.MODEL_COMPLETED for e in events) == 1
        assert any(e.payload.get("evidence_id") == "receipt" for e in events)
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("already_interrupted", [False, True])
def test_external_recovery_accounts_unknown_usage_in_current_budget_window(
    backend, already_interrupted, tmp_path
):
    from datetime import UTC, datetime, timedelta

    from cayu.budgets import BudgetWindow, InMemoryBudgetStore, SessionBudgetStore
    from cayu.events import EventType
    from cayu.external_agents import recover_external_agent_run
    from cayu.sessions import InMemorySessionStore
    from cayu.sessions.records import SessionIdentity, SessionStatus
    from cayu.sessions.requests import run_request_with_task_invocation
    from cayu.tasks import TaskCreate

    async def scenario():
        clock = [datetime.now(UTC) - timedelta(minutes=10)]
        sessions = (
            InMemorySessionStore(ownership_clock=lambda: clock[0])
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.db", ownership_clock=lambda: clock[0])
        )
        app = CayuApp(
            task_store=InMemoryTaskStore(),
            session_store=sessions,
            budget_store=InMemoryBudgetStore()
            if backend == "memory"
            else SessionBudgetStore(sessions),
            enable_logging=False,
        )
        task = await app.create_task(TaskCreate(type="external_agent"))
        bound = run_request_with_task_invocation(
            request(task_id=task.id), await app.task_store.load_invocation_snapshot(task.id)
        )
        session = await app.session_store.create(
            bound,
            identity=SessionIdentity(
                provider_name=ScriptedAdapter.provider_name,
                model=ScriptedAdapter.model,
                runtime_name="external_agent",
            ),
        )
        await app.session_store.transition_status(
            session.id, from_statuses={SessionStatus.PENDING}, to_status=SessionStatus.RUNNING
        )
        await app._session_tasks.start_task(
            task_id=task.id, session=session, worker_id=None, lease_expires_at=None
        )
        if already_interrupted:
            await app.session_store.transition_status(
                session.id,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.INTERRUPTED,
            )
        clock[0] = datetime.now(UTC)
        before = clock[0]

        async def stopped(_):
            pass

        await recover_external_agent_run(
            app, session.id, quiesce=stopped, evidence_id="fresh-receipt"
        )
        usage = [
            event
            for event in await app.session_store.load_events(session.id)
            if event.type is EventType.MODEL_COMPLETED
        ]
        assert len(usage) == 1 and usage[0].timestamp >= before
        adapter = ScriptedAdapter(
            [ExternalAgentEvent("result", {"output": "must not run"}, "result")]
        )
        with pytest.raises(ExternalAgentBudgetExceeded):
            await run_external_agent(
                app,
                request(
                    budget_limits=(
                        BudgetLimit.usd(
                            "10",
                            scope="app",
                            pricing=pricing(),
                            window=BudgetWindow.rolling(seconds=300),
                        ),
                    )
                ),
                adapter,
                prompt="next",
            )
        assert adapter.consumed == 0
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("failure_point", ["status", "task"])
def test_external_recovery_retries_after_partial_settlement(
    backend, failure_point, tmp_path, monkeypatch
):
    from cayu.events import EventType
    from cayu.external_agents import recover_external_agent_run
    from cayu.sessions.records import SessionIdentity, SessionStatus
    from cayu.sessions.requests import run_request_with_task_invocation
    from cayu.tasks import TaskCreate

    async def scenario():
        app = CayuApp(
            task_store=InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.db"),
            session_store=None
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.db"),
            enable_logging=False,
        )
        task = await app.create_task(TaskCreate(type="external_agent"))
        bound = run_request_with_task_invocation(
            request(task_id=task.id), await app.task_store.load_invocation_snapshot(task.id)
        )
        session = await app.session_store.create(
            bound,
            identity=SessionIdentity(
                provider_name="test", model="test", runtime_name="external_agent"
            ),
        )
        session = await app.session_store.transition_status(
            session.id, from_statuses={SessionStatus.PENDING}, to_status=SessionStatus.RUNNING
        )
        await app._session_tasks.start_task(
            task_id=task.id, session=session, worker_id=None, lease_expires_at=None
        )

        async def stopped(_):
            pass

        async def unavailable(*args, **kwargs):
            raise OSError("settlement storage unavailable")

        with monkeypatch.context() as failing:
            if failure_point == "status":
                failing.setattr(app.session_store, "transition_status", unavailable)
            else:
                failing.setattr(app.task_store, "cancel_task", unavailable)
            with pytest.raises(OSError, match="settlement storage unavailable"):
                await recover_external_agent_run(
                    app, session.id, quiesce=stopped, evidence_id="first-supervisor-receipt"
                )
        await recover_external_agent_run(
            app, session.id, quiesce=stopped, evidence_id="next-supervisor-receipt"
        )
        assert (await app.session_store.load(session.id)).status is SessionStatus.INTERRUPTED
        assert (await app.task_store.load_task(task.id)).status is TaskStatus.CANCELLED
        events = await app.session_store.load_events(session.id)
        assert sum(event.type is EventType.MODEL_COMPLETED for event in events) == 1
        assert sum(event.type is EventType.SESSION_INTERRUPTED for event in events) == 1
        assert any(
            event.payload.get("evidence_id") == "first-supervisor-receipt" for event in events
        )
        await app.aclose()

    asyncio.run(scenario())


def test_external_permission_approval_refuses_edits_and_preserves_original_call():
    from cayu.approvals import PendingToolApprovalEventView
    from cayu.approvals.tools import (
        ToolApprovalArgumentsRefused,
        ToolApprovalDecision,
        ToolApprovalRequest,
    )
    from cayu.events import EventType
    from cayu.tools.policy import AlwaysRequireApprovalToolPolicy

    async def scenario():
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        bridge = CayuToolBridge(
            app,
            name="permissions",
            tools=[ExternalPermissionTool("Read")],
            policy=AlwaysRequireApprovalToolPolicy(),
        )
        authorized = []

        class Adapter(ScriptedAdapter):
            async def stream(self, context):
                allowed = await bridge.authorize(
                    "Read",
                    {"path": "original"},
                    "read-1",
                    parent_session_id=context.session_id,
                    execution_deadline=context.execution_deadline,
                )
                authorized.append(allowed)
                yield ExternalAgentEvent("result", {"output": allowed}, "result")

        running = asyncio.create_task(
            run_external_agent(app, request(), Adapter([]), prompt="test")
        )
        try:
            async with asyncio.timeout(10):
                while True:
                    if bridge.active:
                        child_id = next(iter(bridge.active))
                        if await app.session_store.load(child_id) is not None:
                            events = await app.session_store.load_events(child_id)
                            pending = next(
                                (
                                    event
                                    for event in events
                                    if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
                                ),
                                None,
                            )
                            if pending is not None:
                                break
                    await asyncio.sleep(0.01)
            approval = PendingToolApprovalEventView.from_event(pending)
            resolution = ToolApprovalRequest(
                session_id=child_id,
                approval_id=approval.approval_id,
                tool_round_id=approval.tool_round_id,
                tool_call_id=approval.tool_call_id,
                decision=ToolApprovalDecision.APPROVE,
            )
            before = await app.session_store.load_checkpoint(child_id)
            with pytest.raises(ToolApprovalArgumentsRefused, match="Tool policy denied"):
                async for _ in app.resolve_tool_approval(
                    resolution.model_copy(update={"arguments": {"path": "edited"}})
                ):
                    pass
            assert authorized == [] and not running.done()
            assert await app.session_store.load_checkpoint(child_id) == before
            async for _ in app.resolve_tool_approval(resolution):
                pass
            assert (await running).output is True
            assert authorized == [True]
        finally:
            if not running.done():
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
            await app.aclose()

    asyncio.run(scenario())


def test_external_tool_permission_never_forwards_a_known_capability():
    async def scenario():
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        authorized = []

        async def authorize(name, arguments, call_id):
            authorized.append((name, arguments, call_id))
            return True

        class Adapter(ScriptedAdapter):
            async def stream(self, context):
                context.protect_secret("per-run-capability-secret")
                assert not await context.allow_tool(
                    "Read", {"path": "/tmp/per-run-capability-secret"}, "request-1"
                )
                assert not await context.allow_tool(
                    "Read", {"path": "/tmp/file"}, "per-run-capability-secret"
                )
                assert await context.allow_tool("Read", {"path": "/tmp/file"}, "request-2")
                yield ExternalAgentEvent("result", {"output": "done"}, "result")

        await run_external_agent(app, request(), Adapter([]), prompt="test", authorize=authorize)
        assert authorized == [("Read", {"path": "/tmp/file"}, "request-2")]
        await app.aclose()

    asyncio.run(scenario())
