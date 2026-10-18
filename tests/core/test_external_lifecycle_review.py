"""Public regressions for SDK result, interruption and recovery boundaries."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from tests.core.test_external_agents import ScriptedAdapter, pricing, request

from cayu import CayuApp
from cayu._resource_access_errors import ResourceAccessDenied
from cayu.budgets import BudgetLimit
from cayu.deadlines import ExecutionDeadline, ExecutionDeadlineExceeded
from cayu.events import EventType
from cayu.external_agents import (
    ExternalAgentContext,
    ExternalAgentError,
    ExternalAgentEvent,
    ExternalAgentLaunch,
    recover_external_agent_run,
    run_external_agent,
)
from cayu.external_agents.base import ExternalAgentBudgetExceeded
from cayu.external_agents.claude import ClaudeAgentAdapter
from cayu.sessions import InterruptSessionRequest
from cayu.sessions.records import SessionStatus
from cayu.storage import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskQuery, TaskStatus, run_task_worker


class _Result:
    model_usage = {}
    usage = {"input_tokens": 0, "output_tokens": 0}
    total_cost_usd = None
    is_error = False
    structured_output = {"answer": 42}
    result = "done"


def _sdk(monkeypatch, receive, disconnect):
    class Client:
        def __init__(self, *, options):
            self.options = options

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            await disconnect()

        async def query(self, prompt):
            await anext(prompt)

        def receive_response(self):
            return receive(self.options)

    monkeypatch.setattr(
        "cayu.external_agents.claude.import_module",
        lambda _: SimpleNamespace(
            ClaudeSDKClient=Client,
            ClaudeAgentOptions=SimpleNamespace,
            HookMatcher=SimpleNamespace,
            AssistantMessage=type("Assistant", (), {}),
            UserMessage=type("User", (), {}),
            StreamEvent=type("Stream", (), {}),
            ResultMessage=_Result,
            PermissionResultAllow=lambda **kw: SimpleNamespace(allowed=True, **kw),
            PermissionResultDeny=lambda **kw: SimpleNamespace(allowed=False, **kw),
        ),
    )


async def _noop():
    pass


@asynccontextmanager
async def _launch(context):
    yield ExternalAgentLaunch("/sandbox/claude", "/checkout", _noop)


@pytest.mark.parametrize(
    "guard", ["none", "builtins", "schema", "missing", "secret", "budget", "access", "deadline"]
)
def test_claude_structured_result_preserves_run_guards(monkeypatch, guard):
    async def scenario():
        checks = []
        decisions = []
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        arguments = {"answer": "invalid" if guard == "schema" else 42}

        async def authorize(name, arguments, call_id):
            decisions.append(name)
            return name == "Read"

        async def check():
            checks.append(True)
            if guard == "budget":
                raise ExternalAgentBudgetExceeded("exhausted")
            if guard == "access":
                raise ResourceAccessDenied()

        async def receive(options):
            hook = options.hooks["PreToolUse"][0].hooks[0]
            expected = guard in {"none", "builtins"}
            for _ in range(2):
                result = await hook(
                    {"tool_name": "StructuredOutput", "tool_input": arguments}, "result", {}
                )
                assert result["hookSpecificOutput"]["permissionDecision"] == (
                    "allow" if expected else "deny"
                )
                permission = await options.can_use_tool(
                    "StructuredOutput", arguments, SimpleNamespace(tool_use_id="result")
                )
                assert permission.allowed is expected
            changed = await hook(
                {"tool_name": "StructuredOutput", "tool_input": {"answer": 99}}, "result", {}
            )
            assert changed["hookSpecificOutput"]["permissionDecision"] == "deny"
            yield _Result()

        _sdk(monkeypatch, receive, _noop)
        context = ExternalAgentContext(
            "session",
            "review",
            ExecutionDeadline.after(0 if guard == "deadline" else 30),
            output_schema=None if guard == "missing" else schema,
            authorize=authorize if guard in {"builtins", "missing"} else None,
            check_budget=check,
            contains_secret=lambda _: guard == "secret",
        )
        adapter = ClaudeAgentAdapter(model="claude", launch=_launch)
        error = {
            "budget": ExternalAgentBudgetExceeded,
            "access": ResourceAccessDenied,
            "deadline": ExecutionDeadlineExceeded,
        }.get(guard)
        if error:
            with pytest.raises(error):
                [event async for event in adapter.stream(context)]
        else:
            events = [event async for event in adapter.stream(context)]
            assert events[-1].kind == "result"
            assert checks == ([] if guard == "secret" else [True])
            assert decisions == (["StructuredOutput"] if guard == "missing" else [])

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("cross_process", [False, True])
def test_operator_interrupt_joins_slow_sdk_disconnect_and_keeps_worker(
    monkeypatch, tmp_path, backend, cross_process
):
    async def scenario():
        entered = asyncio.Event()
        closing = asyncio.Event()
        closed = []
        started = []

        async def receive(options):
            started.append(options)
            if len(started) == 1:
                entered.set()
                await asyncio.Event().wait()
            yield _Result()

        async def disconnect():
            closing.set()
            await asyncio.sleep(0.3)
            closed.append(True)

        _sdk(monkeypatch, receive, disconnect)
        app = CayuApp(
            task_store=InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.db"),
            enable_logging=False,
        )
        first = await app.create_task(TaskCreate(type="vendor", restartable=True))
        second = await app.create_task(TaskCreate(type="vendor"))

        async def handle(current_app, task, worker_id):
            await run_external_agent(
                current_app,
                request(
                    task_id=task.id,
                    task_worker_id=worker_id,
                    task_lease_expires_at=task.lease_expires_at,
                ),
                ClaudeAgentAdapter(model="claude", launch=_launch),
                prompt="review",
            )

        worker = asyncio.create_task(
            run_task_worker(
                app, app.task_store, handle, worker_id="worker", max_tasks=2, poll_interval_s=0.01
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            active = await app.task_store.load_task(first.id)
            if cross_process:
                await app.session_store.transition_status(
                    active.session_id,
                    from_statuses={SessionStatus.RUNNING},
                    to_status=SessionStatus.INTERRUPTING,
                )
            else:
                async for _ in app.interrupt_session(
                    InterruptSessionRequest(session_id=active.session_id)
                ):
                    pass
                # A duplicate request during cleanup must also remain job-scoped.
                await asyncio.wait_for(closing.wait(), 5)
                async for _ in app.interrupt_session(
                    InterruptSessionRequest(session_id=active.session_id)
                ):
                    pass
            await asyncio.wait_for(worker, 10)
            cancelled = await app.task_store.load_task(first.id)
            assert cancelled.status is TaskStatus.CANCELLED and cancelled.restart_count == 0
            assert (await app.task_store.load_task(second.id)).status is TaskStatus.COMPLETED
            assert len(started) == len(closed) == 2
        finally:
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_failed_external_recovery_restores_missing_accounting(monkeypatch, tmp_path, backend):
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

        class Failing(ScriptedAdapter):
            async def stream(self, context):
                raise RuntimeError("SDK failed after dispatch")
                yield

        emit = app.emit_event

        async def fail_usage(event):
            if event.type is EventType.MODEL_COMPLETED:
                raise OSError("usage storage unavailable")
            return await emit(event)

        with monkeypatch.context() as failing:
            failing.setattr(app, "emit_event", fail_usage)
            with pytest.raises(ExternalAgentError, match="cleanup failed"):
                await run_external_agent(app, request(), Failing([]), prompt="review")
        [task] = await app.task_store.list_tasks(TaskQuery())
        assert task.status is TaskStatus.RUNNING
        assert (await app.session_store.load(task.session_id)).status is SessionStatus.FAILED
        assert not any(
            event.type is EventType.MODEL_COMPLETED
            for event in await app.session_store.load_events(task.session_id)
        )
        proofs = []

        async def quiesce(session):
            proofs.append(session.id)

        for _ in range(2):
            session = await recover_external_agent_run(
                app, task.session_id, quiesce=quiesce, evidence_id="stopped"
            )
            assert session.status is SessionStatus.FAILED
        assert proofs == [task.session_id]
        assert (await app.task_store.load_task(task.id)).status is TaskStatus.CANCELLED
        events = await app.session_store.load_events(task.session_id)
        usage = [event for event in events if event.type is EventType.MODEL_COMPLETED]
        assert len(usage) == 1 and usage[0].payload["usage_outcome"] == "unknown_after_crash"
        assert not any(event.type is EventType.SESSION_INTERRUPTED for event in events)
        adapter = ScriptedAdapter([ExternalAgentEvent("result", {"output": "blocked"}, "result")])
        with pytest.raises(ExternalAgentBudgetExceeded):
            await run_external_agent(
                app,
                request(budget_limits=(BudgetLimit.usd("10", scope="app", pricing=pricing()),)),
                adapter,
                prompt="next",
            )
        assert adapter.consumed == 0
        await app.aclose()

    asyncio.run(scenario())


def test_local_operator_interrupt_wins_before_the_next_observer_tick():
    async def scenario():
        entered = asyncio.Event()
        finish = asyncio.Event()
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        first = await app.create_task(TaskCreate(type="vendor", restartable=True))
        second = await app.create_task(TaskCreate(type="vendor"))
        sessions = []

        class Finishing(ScriptedAdapter):
            async def stream(self, context):
                sessions.append(context.session_id)
                if len(sessions) == 1:
                    entered.set()
                    await finish.wait()
                yield ExternalAgentEvent("result", {"output": "done"}, "result")

        async def handle(current_app, task, worker_id):
            await run_external_agent(
                current_app,
                request(
                    task_id=task.id,
                    task_worker_id=worker_id,
                    task_lease_expires_at=task.lease_expires_at,
                ),
                Finishing([]),
                prompt="review",
            )

        worker = asyncio.create_task(
            run_task_worker(
                app, app.task_store, handle, worker_id="worker", max_tasks=2, poll_interval_s=0.01
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            async for _ in app.interrupt_session(InterruptSessionRequest(session_id=sessions[0])):
                pass
            finish.set()
            await asyncio.wait_for(worker, 10)
            assert (await app.task_store.load_task(first.id)).status is TaskStatus.CANCELLED
            assert (await app.session_store.load(sessions[0])).status is SessionStatus.INTERRUPTED
            assert (await app.task_store.load_task(second.id)).status is TaskStatus.COMPLETED
        finally:
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await app.aclose()

    asyncio.run(scenario())
