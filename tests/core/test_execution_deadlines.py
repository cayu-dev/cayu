from __future__ import annotations

import asyncio
import subprocess
import sys
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    ExecutionDeadline,
    ExecutionDeadlineExceeded,
    Message,
    ModelStreamEvent,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
    WorkflowBase,
    WorkflowSpec,
    current_execution_deadline,
    execution_deadline_scope,
)
from cayu.runtime import InMemorySessionStore, SessionIdentity, SessionStatus
from cayu.workflows import StepRunOptions, parallel, step


@pytest.fixture
def clock(monkeypatch):
    import cayu.deadlines as module

    state = SimpleNamespace(wall=datetime(2026, 9, 5, tzinfo=UTC), mono=100.0)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return state.wall

    monkeypatch.setattr(module, "datetime", Clock)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: state.mono))
    return state


def test_live_remaining_and_exact_expiry(clock):
    assert ExecutionDeadline().remaining_seconds() is None
    boundary = ExecutionDeadline.after(10, source="evaluator", scope="case")
    assert boundary.remaining_seconds() == 10
    clock.wall += timedelta(seconds=4)
    clock.mono += 4
    assert boundary.remaining_seconds() == 6
    assert "6.000" in boundary.model_context()
    clock.wall += timedelta(seconds=6)
    clock.mono += 6
    assert boundary.expired
    assert boundary.remaining_seconds() == 0
    with pytest.raises(ExecutionDeadlineExceeded) as exc:
        boundary.require_admission("model")
    assert exc.value.diagnostics["denied_stage"] == "model"
    clock.wall += timedelta(seconds=50)
    assert boundary.remaining_seconds() == 0


def test_clock_rollback_does_not_extend_local_execution(clock):
    boundary = ExecutionDeadline.after(10)
    clock.wall -= timedelta(days=1)
    clock.mono += 10
    assert boundary.expired


def test_portable_expiry_reloads_live_time_and_never_serializes_monotonic(clock):
    boundary = ExecutionDeadline.after(10)
    payload = boundary.model_dump_json()
    assert "monotonic" not in payload and "remaining" not in payload
    clock.wall += timedelta(seconds=7)
    clock.mono = -9000  # unrelated process-local epoch
    restored = ExecutionDeadline.model_validate_json(payload)
    assert restored.expires_at == boundary.expires_at
    assert restored.remaining_seconds() == 3
    clock.wall += timedelta(seconds=3)
    assert restored.expired


@pytest.mark.parametrize("seconds", [-1, float("nan"), float("inf"), True])
def test_invalid_durations(seconds):
    with pytest.raises(ValueError):
        ExecutionDeadline.after(seconds)


def test_aware_utc_required_and_labels_bounded():
    with pytest.raises(ValueError):
        ExecutionDeadline(expires_at=datetime(2026, 9, 5))
    with pytest.raises(ValueError):
        ExecutionDeadline(source="raw prompt with secrets")


def test_nested_and_parallel_scope_composition(clock):
    async def run():
        parent = ExecutionDeadline.after(100)
        shorter = ExecutionDeadline.after(10)

        async def child(boundary):
            async with execution_deadline_scope(boundary):
                await asyncio.sleep(0)
                return current_execution_deadline().expires_at

        async with execution_deadline_scope(parent):
            assert await asyncio.gather(child(shorter), child(ExecutionDeadline.after(200))) == [
                shorter.expires_at,
                parent.expires_at,
            ]
            assert current_execution_deadline().expires_at == parent.expires_at
        assert current_execution_deadline().expires_at is None

    asyncio.run(run())


def _app(store=None, provider=None):
    app = CayuApp(enable_logging=False, session_store=store)
    provider = provider or ScriptedModelProvider(
        [
            [ModelStreamEvent.text_delta("verified"), ModelStreamEvent.completed({})]
            for _ in range(5)
        ]
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker", model="scripted-model"))
    return app, provider


@pytest.mark.parametrize("bounded", [False, True])
def test_workflow_execute_close_awaits_authored_async_cleanup(bounded):
    async def scenario():
        cleanup_completed = False

        class CleanupWorkflow(WorkflowBase):
            spec = WorkflowSpec(name="close-cleanup")

            async def run(self, session_id):
                nonlocal cleanup_completed
                try:
                    yield self.context(session_id).event(EventType.WORKFLOW_STARTED)
                finally:
                    await asyncio.sleep(0)
                    cleanup_completed = True

        workflow = CleanupWorkflow(CayuApp(enable_logging=False))
        stream = workflow.execute(
            "close-cleanup",
            execution_deadline=ExecutionDeadline.after(30) if bounded else None,
        )
        await anext(stream)
        assert not cleanup_completed
        await stream.aclose()
        assert cleanup_completed

    asyncio.run(scenario())


class FinalizeWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="deadline-finalization")

    async def run(self, session_id):
        ctx = self.context(session_id)
        yield await ctx.start()
        remaining = await ctx.remaining_seconds()
        # Application policy, intentionally not encoded in Runtime.
        if remaining is not None and remaining < 20:
            yield await ctx.completed({"answer": "reuse verified result"})
            return
        await parallel(
            [
                step(ctx, agent="worker", step_id="a", prompt="work"),
                step(
                    ctx,
                    agent="worker",
                    step_id="b",
                    prompt="work",
                    run_options=StepRunOptions(execution_deadline=ExecutionDeadline.after(25)),
                ),
            ]
        )
        yield await ctx.completed({"answer": "verified"})


def test_ordinary_workflow_finalizes_without_unnecessary_dispatch(clock):
    async def run():
        app, provider = _app()
        boundary = ExecutionDeadline.after(10)
        events = [
            e async for e in FinalizeWorkflow(app).execute("finalize", execution_deadline=boundary)
        ]
        assert events[-1].type == EventType.WORKFLOW_COMPLETED
        assert events[-1].payload["answer"] == "reuse verified result"
        assert provider.requests == []
        anchor = await app.session_store.load("finalize")
        assert anchor.execution_deadline.expires_at == boundary.expires_at

    asyncio.run(run())


def test_workflow_parallel_children_persist_inheritance_and_tightening():
    async def run():
        app, _ = _app()
        boundary = ExecutionDeadline.after(100)
        events = [
            e async for e in FinalizeWorkflow(app).execute("parallel", execution_deadline=boundary)
        ]
        records = await app.session_store.load_events("parallel")
        completions = {
            event.payload["step_id"]: event.payload["child_session_id"]
            for event in records
            if event.type == EventType.WORKFLOW_STEP_COMPLETED
        }
        assert set(completions) == {"a", "b"}
        for step_id, child_id in completions.items():
            child = await app.session_store.load(child_id)
            assert child.execution_deadline.expires_at <= boundary.expires_at
            if step_id == "a":
                assert child.execution_deadline.expires_at == boundary.expires_at
            else:
                assert child.execution_deadline.remaining_seconds() <= 25
        assert events[-1].type == EventType.WORKFLOW_COMPLETED
        assert current_execution_deadline().expires_at is None

    asyncio.run(run())


@pytest.mark.parametrize("sqlite", [False, True])
def test_store_parent_clamps_extension_and_preserves_metadata(tmp_path, sqlite):
    async def run():
        store = SQLiteSessionStore(tmp_path / "deadline.db") if sqlite else InMemorySessionStore()
        identity = SessionIdentity(provider_name="test", model="test")
        boundary = ExecutionDeadline.after(100)
        parent = await store.create(
            RunRequest(agent_name="worker", messages=[], execution_deadline=boundary),
            identity=identity,
        )
        child = await store.create(
            RunRequest(
                agent_name="worker",
                messages=[],
                parent_session_id=parent.id,
                execution_deadline=ExecutionDeadline.after(200),
            ),
            identity=identity,
        )
        assert child.execution_deadline.expires_at == boundary.expires_at
        await store.update_metadata(child.id, {"label": "safe"})
        loaded = await store.load(child.id)
        assert loaded.execution_deadline.expires_at == boundary.expires_at
        if sqlite:
            await store.close()

    asyncio.run(run())


def test_expired_fresh_and_resumed_runs_dispatch_nothing(tmp_path, clock):
    async def run():
        store = SQLiteSessionStore(tmp_path / "resume.db")
        app, provider = _app(store)
        with pytest.raises(ExecutionDeadlineExceeded):
            _ = [
                e
                async for e in app.run(
                    RunRequest(
                        agent_name="worker",
                        messages=[],
                        execution_deadline=ExecutionDeadline.after(0),
                    )
                )
            ]
        assert provider.requests == []
        boundary = ExecutionDeadline.after(100)
        events = [
            e
            async for e in app.run(
                RunRequest(
                    agent_name="worker",
                    messages=[Message.text("user", "go")],
                    execution_deadline=boundary,
                )
            )
        ]
        session_id = events[0].session_id
        await store.close()
        clock.wall += timedelta(seconds=100)
        clock.mono = 0
        recovered_store = SQLiteSessionStore(tmp_path / "resume.db")
        recovered, recovered_provider = _app(recovered_store)
        with pytest.raises(ExecutionDeadlineExceeded):
            _ = [
                e
                async for e in recovered.resume(
                    ResumeRequest(
                        session_id=session_id, messages=[Message.text("user", "continue")]
                    )
                )
            ]
        assert recovered_provider.requests == []
        assert (
            await recovered_store.load(session_id)
        ).execution_deadline.expires_at == boundary.expires_at
        await recovered_store.close()

    asyncio.run(run())


def test_workflow_recovery_retains_expiry_and_denies_completion(tmp_path, clock):
    async def run():
        store = SQLiteSessionStore(tmp_path / "workflow.db")
        app, _ = _app(store)
        boundary = ExecutionDeadline.after(10)
        _ = [e async for e in FinalizeWorkflow(app).execute("wf", execution_deadline=boundary)]
        await store.close()
        clock.wall += timedelta(seconds=10)
        recovered_store = SQLiteSessionStore(tmp_path / "workflow.db")
        recovered, provider = _app(recovered_store)
        ctx = FinalizeWorkflow(recovered).context("wf")
        assert await ctx.remaining_seconds() == 0
        with pytest.raises(ExecutionDeadlineExceeded):
            await step(ctx, agent="worker", step_id="new", prompt="go")
        with pytest.raises(ExecutionDeadlineExceeded):
            await ctx.completed({"answer": "false success"})
        assert provider.requests == []
        await recovered_store.close()

    asyncio.run(run())


def test_process_roundtrip_uses_absolute_expiry():
    expired = ExecutionDeadline(expires_at=datetime(2000, 1, 1, tzinfo=UTC))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from cayu import ExecutionDeadline; import sys; d=ExecutionDeadline.model_validate_json(sys.argv[1]); print(d.remaining_seconds())",
            expired.model_dump_json(),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "0.0"


@pytest.mark.parametrize("kind", ["run", "resume", "fork"])
def test_deadline_metadata_cannot_be_forged_in_request(kind):
    from cayu import ForkSessionRequest

    metadata = {"cayu:execution_deadline": {}}
    with pytest.raises(ValueError, match="deadline"):
        if kind == "run":
            RunRequest(agent_name="worker", messages=[], metadata=metadata)
        elif kind == "resume":
            ResumeRequest(session_id="s", messages=[Message.text("user", "go")], metadata=metadata)
        else:
            ForkSessionRequest(source_session_id="s", metadata=metadata)


def test_inflight_model_cancellation_waits_for_settlement():
    settled = []

    class BlockingProvider(ScriptedModelProvider):
        async def stream(self, request):
            try:
                await asyncio.Event().wait()
                yield ModelStreamEvent.completed({})
            finally:
                await asyncio.sleep(0)
                settled.append(True)

    async def run():
        app, _ = _app(provider=BlockingProvider([]))
        events = []
        with pytest.raises(TimeoutError):
            async for event in app.run(
                RunRequest(
                    agent_name="worker",
                    messages=[Message.text("user", "go")],
                    execution_deadline=ExecutionDeadline.after(0.5),
                )
            ):
                events.append(event)
        assert settled == [True]
        session_id = events[0].session_id
        stored = await app.session_store.load(session_id)
        assert stored.status != SessionStatus.COMPLETED
        persisted = await app.session_store.load_events(session_id)
        assert any(e.type == EventType.SESSION_INTERRUPTED for e in persisted)
        assert not any(e.type == EventType.SESSION_COMPLETED for e in persisted)

    asyncio.run(run())


def test_observed_wall_expiry_stays_expired_after_clock_rollback(clock):
    boundary = ExecutionDeadline.after(10)
    clock.wall += timedelta(seconds=11)
    assert boundary.expired
    clock.wall -= timedelta(seconds=11)
    assert boundary.expired


def test_no_new_tools_or_models_when_wall_expiry_is_observed(clock):
    from cayu import Tool, ToolResult, ToolSpec

    calls = []

    class NeverTool(Tool):
        spec = ToolSpec(
            name="never",
            description="Never dispatched",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx, args):
            calls.append("tool")
            return ToolResult(content="unexpected")

    class ExpiringProvider(ScriptedModelProvider):
        async def stream(self, request):
            calls.append("model")
            clock.wall += timedelta(seconds=100)
            yield ModelStreamEvent.tool_call(id="call", name="never", arguments={})
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})

    async def run():
        app = CayuApp(enable_logging=False)
        app.register_provider(ExpiringProvider([]), default=True)
        app.register_agent(AgentSpec(name="worker", model="scripted-model"), tools=[NeverTool()])
        events = []
        with pytest.raises(TimeoutError):
            async for event in app.run(
                RunRequest(
                    agent_name="worker",
                    messages=[Message.text("user", "go")],
                    execution_deadline=ExecutionDeadline.after(100),
                )
            ):
                events.append(event)
        assert calls == ["model"]
        assert not any(e.type == EventType.SESSION_COMPLETED for e in events)
        session = await app.session_store.load(events[0].session_id)
        assert session.status != SessionStatus.COMPLETED

    asyncio.run(run())


def test_inflight_tool_keeps_context_and_cleanup_after_expiry():
    from cayu import Tool, ToolResult, ToolSpec

    seen = []

    class BlockingTool(Tool):
        spec = ToolSpec(
            name="blocking",
            description="Blocking",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx, args):
            seen.append(
                (
                    "entered",
                    ctx.execution_deadline.expires_at,
                    current_execution_deadline().expires_at,
                )
            )
            try:
                await asyncio.Event().wait()
                return ToolResult(content="unexpected")
            finally:
                await asyncio.sleep(0)
                seen.append(("settled", ctx.execution_deadline.remaining_seconds()))

    async def run():
        app = CayuApp(enable_logging=False)
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="call", name="blocking", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker", model="scripted-model"), tools=[BlockingTool()])
        boundary = ExecutionDeadline.after(1)
        events = []
        with pytest.raises(TimeoutError):
            async for event in app.run(
                RunRequest(
                    agent_name="worker",
                    messages=[Message.text("user", "go")],
                    execution_deadline=boundary,
                )
            ):
                events.append(event)
        assert seen == [("entered", boundary.expires_at, boundary.expires_at), ("settled", 0.0)]
        persisted = await app.session_store.load_events(events[0].session_id)
        assert any(e.type == EventType.SESSION_INTERRUPTED for e in persisted)
        assert not any(e.type == EventType.SESSION_COMPLETED for e in persisted)
        assert len(provider.requests) == 1

    asyncio.run(run())


def test_failed_step_retry_does_not_receive_a_fresh_child_duration(clock):
    from cayu.workflows import StepError

    async def run():
        provider = ScriptedModelProvider(
            [
                [ModelStreamEvent.error("failed"), ModelStreamEvent.completed({})],
                [ModelStreamEvent.text_delta("retry succeeded"), ModelStreamEvent.completed({})],
            ]
        )
        app, _ = _app(provider=provider)
        workflow = FinalizeWorkflow(app)
        parent = ExecutionDeadline.after(100)
        original_child = ExecutionDeadline.after(10)
        async with execution_deadline_scope(parent):
            with pytest.raises(StepError):
                await step(
                    workflow.context("retry"),
                    agent="worker",
                    step_id="same",
                    prompt="go",
                    run_options=StepRunOptions(execution_deadline=original_child),
                )
            clock.wall += timedelta(seconds=10)
            clock.mono += 10
            with pytest.raises(ExecutionDeadlineExceeded):
                await step(
                    workflow.context("retry"),
                    agent="worker",
                    step_id="same",
                    prompt="go",
                    run_options=StepRunOptions(execution_deadline=ExecutionDeadline.after(10)),
                )
        assert len(provider.requests) == 1

    asyncio.run(run())


def test_continuation_uses_original_expiry_and_rejects_transient_tightening(clock):
    async def run():
        app, provider = _app()
        boundary = ExecutionDeadline.after(100)
        events = [
            e
            async for e in app.run(
                RunRequest(
                    agent_name="worker",
                    messages=[Message.text("user", "go")],
                    execution_deadline=boundary,
                )
            )
        ]
        session_id = events[0].session_id
        clock.wall += timedelta(seconds=10)
        clock.mono += 10
        async with execution_deadline_scope(ExecutionDeadline.after(200)):
            _ = [
                e
                async for e in app.resume(
                    ResumeRequest(
                        session_id=session_id, messages=[Message.text("user", "continue")]
                    )
                )
            ]
        assert (
            await app.session_store.load(session_id)
        ).execution_deadline.expires_at == boundary.expires_at
        assert len(provider.requests) == 2
        async with execution_deadline_scope(ExecutionDeadline.after(5)):
            with pytest.raises(ValueError, match="Cannot tighten"):
                _ = [
                    e
                    async for e in app.resume(
                        ResumeRequest(
                            session_id=session_id, messages=[Message.text("user", "continue")]
                        )
                    )
                ]
        assert len(provider.requests) == 2

    asyncio.run(run())


def test_isolated_context_projection_carries_only_portable_deadline():
    import json

    from tests.core.test_process_isolated_tools import _tool

    from cayu import ToolContext
    from cayu.runtime._isolated_tool_process import _project_context

    boundary = ExecutionDeadline.after(60)
    projected = _project_context(
        tool=_tool(),
        context=ToolContext(session_id="isolated", execution_deadline=boundary),
    )
    payload = projected.model_dump(mode="json")
    assert projected.execution_deadline.expires_at == boundary.expires_at
    assert "monotonic" not in json.dumps(payload)
    assert "remaining_seconds" not in json.dumps(payload)
    code = (
        "import sys; from cayu.core.isolated_tools import ProcessIsolatedToolContext; "
        "from cayu.deadlines import bind_execution_deadline; from cayu import RunRequest; "
        "ctx=ProcessIsolatedToolContext.model_validate_json(sys.argv[1]); "
        "scope=bind_execution_deadline(ctx.execution_deadline); scope.__enter__(); "
        "r=RunRequest(agent_name='child',messages=[]); "
        "print(r.execution_deadline.expires_at.isoformat()); scope.__exit__(None,None,None)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, projected.model_dump_json()],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == boundary.expires_at.isoformat()


def test_model_retry_uses_live_original_deadline(clock):
    from cayu import RetryPolicy

    seen = []

    class RetryProvider(ScriptedModelProvider):
        async def stream(self, request):
            boundary = current_execution_deadline()
            seen.append((boundary.expires_at, boundary.remaining_seconds()))
            if len(seen) == 1:
                clock.wall += timedelta(seconds=5)
                clock.mono += 5
                raise TimeoutError("stream idle timeout")
            yield ModelStreamEvent.text_delta("verified")
            yield ModelStreamEvent.completed({})

    async def run():
        app, _ = _app(provider=RetryProvider([]))
        boundary = ExecutionDeadline.after(100)
        events = [
            e
            async for e in app.run(
                RunRequest(
                    agent_name="worker",
                    messages=[Message.text("user", "go")],
                    execution_deadline=boundary,
                    retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0),
                )
            )
        ]
        assert events[-1].type == EventType.SESSION_COMPLETED, events[-1].payload
        assert seen == [(boundary.expires_at, 100), (boundary.expires_at, 95)]

    asyncio.run(run())


def test_malformed_persisted_deadline_fails_closed():
    from cayu.deadlines import deadline_from_metadata

    with pytest.raises(ValueError):
        deadline_from_metadata({"cayu:execution_deadline": None})


def test_timeout_retains_secondary_cleanup_failure_and_expiry():
    class CleanupFailure(RuntimeError):
        pass

    async def run():
        boundary = ExecutionDeadline.after(0.01)
        with pytest.raises(CleanupFailure) as raised:
            async with execution_deadline_scope(boundary):
                try:
                    await asyncio.Event().wait()
                finally:
                    raise CleanupFailure("cleanup did not settle")
        assert raised.value.execution_deadline["remaining_seconds"] == 0
        assert isinstance(raised.value.__context__, (asyncio.CancelledError, TimeoutError))

    asyncio.run(run())


def test_postgres_deadline_parent_composition_and_reopen(postgres_dsn):
    from cayu import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    async def run():
        store = PostgresSessionStore(
            postgres_dsn, min_size=1, max_size=4, schema_mode=SchemaMode.CREATE
        )
        boundary = ExecutionDeadline.after(100)
        identity = SessionIdentity(provider_name="test", model="test")
        try:
            parent = await store.create(
                RunRequest(agent_name="worker", messages=[], execution_deadline=boundary),
                identity=identity,
            )
            child = await store.create(
                RunRequest(
                    agent_name="worker",
                    messages=[],
                    parent_session_id=parent.id,
                    execution_deadline=ExecutionDeadline.after(200),
                ),
                identity=identity,
            )
            assert child.execution_deadline.expires_at == boundary.expires_at
            await store.update_metadata(child.id, {"label": "safe"})
        finally:
            await store.close()
        restored = PostgresSessionStore(postgres_dsn, min_size=1, max_size=4)
        try:
            session = await restored.load(child.id)
            assert session.execution_deadline.expires_at == boundary.expires_at
            assert session.execution_deadline.remaining_seconds() <= 100
        finally:
            await restored.close()

    asyncio.run(run())


def test_fork_inherits_tighter_scope_and_replays_committed_result_after_expiry(clock):
    from cayu import ForkSessionRequest

    async def run():
        app, provider = _app()
        original = ExecutionDeadline.after(100)
        events = [
            e
            async for e in app.run(
                RunRequest(
                    agent_name="worker",
                    messages=[Message.text("user", "go")],
                    execution_deadline=original,
                )
            )
        ]
        source_id = events[0].session_id
        fork_request = ForkSessionRequest(source_session_id=source_id, session_id="deadline-fork")
        tighter = ExecutionDeadline.after(20)
        async with execution_deadline_scope(tighter):
            forked = [e async for e in app.fork_session(fork_request)]
        child = await app.session_store.load("deadline-fork")
        assert child.execution_deadline.expires_at == tighter.expires_at
        clock.wall += timedelta(seconds=100)
        clock.mono += 100
        with pytest.raises(ExecutionDeadlineExceeded):
            _ = [
                e
                async for e in app.fork_session(
                    ForkSessionRequest(source_session_id=source_id, session_id="expired-fork")
                )
            ]
        assert await app.session_store.load("expired-fork") is None
        replayed = [e async for e in app.fork_session(fork_request)]
        assert [e.id for e in replayed] == [e.id for e in forked]
        assert len(provider.requests) == 1

    asyncio.run(run())


def test_child_cannot_extend_local_parent_after_wall_clock_rollback(clock):
    from cayu.deadlines import effective_deadline

    parent = ExecutionDeadline.after(100)
    clock.wall -= timedelta(seconds=50)
    clock.mono += 50
    child = ExecutionDeadline.after(100)
    assert child.expires_at < parent.expires_at
    effective = effective_deadline(parent, child)
    assert effective.expires_at == child.expires_at
    assert effective.remaining_seconds() == parent.remaining_seconds() == 50
    clock.wall += timedelta(seconds=200)
    assert parent.expired
    clock.wall -= timedelta(seconds=200)
    assert effective_deadline(parent, ExecutionDeadline.after(10)).expired


def test_unbounded_requests_and_process_context_keep_legacy_serialized_shape():
    from cayu import ToolContext
    from cayu.core.isolated_tools import ProcessIsolatedToolContext

    assert "execution_deadline" not in RunRequest(agent_name="worker", messages=[]).model_dump(
        mode="json"
    )
    assert "execution_deadline" not in ToolContext(session_id="s").model_dump(mode="json")
    assert "execution_deadline" not in ProcessIsolatedToolContext().model_dump(mode="json")


def test_consumed_cancellation_cannot_report_successful_execution():
    async def run():
        with pytest.raises(ExecutionDeadlineExceeded) as raised:
            async with execution_deadline_scope(ExecutionDeadline.after(0.01)):
                with suppress(asyncio.CancelledError):
                    await asyncio.Event().wait()
        assert raised.value.diagnostics["denied_stage"] == "execution_completion"
        assert raised.value.diagnostics["remaining_seconds"] == 0

    asyncio.run(run())
