from __future__ import annotations

import asyncio
import subprocess
import sys
from contextlib import aclosing, suppress
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.context.structured_output import StructuredOutputSpec
from cayu.deadlines import (
    ExecutionDeadline,
    ExecutionDeadlineExceeded,
    current_execution_deadline,
    execution_deadline_scope,
)
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import ResumeRequest, RunRequest
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.workflows.base import WorkflowSpec
from cayu.workflows.workflow import StepRunOptions, WorkflowBase, parallel, step


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
            "from cayu.deadlines import ExecutionDeadline; import sys; d=ExecutionDeadline.model_validate_json(sys.argv[1]); print(d.remaining_seconds())",
            expired.model_dump_json(),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "0.0"


@pytest.mark.parametrize("kind", ["run", "resume", "fork"])
def test_deadline_metadata_cannot_be_forged_in_request(kind):
    from cayu.sessions.requests import ForkSessionRequest

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
        async for event in app.run(
            RunRequest(
                agent_name="worker",
                messages=[Message.text("user", "go")],
                execution_deadline=ExecutionDeadline.after(3),
            )
        ):
            events.append(event)
        assert settled == [True]
        terminal = next(event for event in events if event.type is EventType.SESSION_INTERRUPTED)
        assert terminal.payload["reason"] == "limit_reached"
        assert terminal.payload["limit"] == "execution_deadline"
        assert terminal.payload["execution_deadline"]["remaining_seconds"] == 0
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


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
@pytest.mark.parametrize("delivery", ["advance", "close"])
@pytest.mark.parametrize("structured", [False, True])
def test_expiry_after_resume_admission_finalizes_before_delivery_returns(
    clock, tmp_path, store_kind, delivery, structured
):
    async def scenario():
        store = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "resume-admission.sqlite3")
        )
        output = StructuredOutputSpec(
            strategy="native",
            json_schema={
                "type": "object",
                "properties": {"answer": {"type": "integer"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        )
        app, provider = _app(
            store=store,
            provider=ScriptedModelProvider(
                [[ModelStreamEvent.text_delta('{"answer": 42}'), ModelStreamEvent.completed({})]],
                supports_native_structured_output=True,
            ),
        )
        try:
            initial = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="worker",
                        session_id="resume-admission",
                        messages=[Message.text("user", "First")],
                        structured_output=output if structured else None,
                        execution_deadline=ExecutionDeadline.after(3600),
                    )
                )
            ]
            assert initial[-1].type is EventType.SESSION_COMPLETED
            previous = await store.load("resume-admission")
            stream = app.resume(
                ResumeRequest(
                    session_id="resume-admission",
                    messages=[Message.text("user", "Next")],
                    structured_output=output if structured else None,
                )
            )
            admitted = await anext(stream)
            assert admitted.type is EventType.INTERACTION_STARTED
            admitted_session = await store.load("resume-admission")
            assert admitted_session.run_epoch == previous.run_epoch + 1
            clock.wall += timedelta(seconds=3601)
            clock.mono += 3601
            if delivery == "advance":
                remaining = [event async for event in stream]
                assert remaining[-1].type is EventType.SESSION_INTERRUPTED
                assert remaining[-1].payload["limit"] == "execution_deadline"
                assert remaining[-1].payload["structured_output_pending"] is structured
            else:
                await stream.aclose()
            current = await store.load("resume-admission")
            assert current.status is SessionStatus.INTERRUPTED
            assert current.run_epoch >= admitted_session.run_epoch
            assert len(provider.requests) == 1
            terminals = [
                event
                for event in await store.load_events("resume-admission")
                if event.type is EventType.SESSION_INTERRUPTED
            ]
            assert len(terminals) == 1
            assert terminals[0].payload["interruption_type"] == (
                "limit_reached" if delivery == "advance" else "runtime_interrupted"
            )
            assert (
                terminals[0].payload["failure_evidence"]["run_epoch"] == admitted_session.run_epoch
            )
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_no_new_tools_or_models_when_wall_expiry_is_observed(clock):
    from cayu.tools.base import Tool, ToolResult, ToolSpec

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


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
@pytest.mark.parametrize("entrypoint", ["run", "workflow_step"])
@pytest.mark.parametrize("tool_count", [1, 2])
def test_inflight_tool_keeps_context_and_cleanup_after_expiry(
    clock, tmp_path, store_kind, entrypoint, tool_count
):
    from cayu.configuration import CayuConfig, ToolExecutionConfig
    from cayu.deadlines import _TIMERS
    from cayu.tools.base import Tool, ToolResult, ToolSpec
    from cayu.workflows.models import StepError

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
            # Expire only once every tool is admitted, including parallel children.
            if len(seen) == tool_count:
                clock.wall += timedelta(seconds=3600)
                clock.mono += 3600
                # Expire the real asyncio timers only after admission. A short
                # setup timer races SQLite I/O on a busy CI host.
                for timer, _deadline in _TIMERS.get():
                    if timer.when() is not None:
                        timer.reschedule(asyncio.get_running_loop().time())
            try:
                await asyncio.Event().wait()
                return ToolResult(content="unexpected")
            finally:
                await asyncio.sleep(0)
                seen.append(("settled", ctx.execution_deadline.remaining_seconds()))

    async def run():
        loop_errors = []
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
        store = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "deadline.sqlite3")
        )
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            config=CayuConfig(
                tool_execution=ToolExecutionConfig(max_parallel_tool_calls=tool_count)
            ),
        )
        provider = ScriptedModelProvider(
            [
                [
                    *[
                        ModelStreamEvent.tool_call(
                            id=f"call-{index}", name="blocking", arguments={}
                        )
                        for index in range(tool_count)
                    ],
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker", model="scripted-model"), tools=[BlockingTool()])
        boundary = ExecutionDeadline.after(3600)
        events = []
        if entrypoint == "run":
            async for event in app.run(
                RunRequest(
                    agent_name="worker",
                    messages=[Message.text("user", "go")],
                    execution_deadline=boundary,
                )
            ):
                events.append(event)
            session_id = events[0].session_id
        else:
            with pytest.raises(StepError) as step_failure:
                await step(
                    FinalizeWorkflow(app).context("deadline-parent"),
                    agent="worker",
                    step_id="timed-out-tool",
                    prompt="go",
                    run_options=StepRunOptions(execution_deadline=boundary),
                )
            assert step_failure.value.evidence.classification == "deadline"
            assert step_failure.value.evidence.deadline_phase == "in_flight"
            session_id = step_failure.value.session_id
            assert session_id is not None
        assert (
            seen
            == [("entered", boundary.expires_at, boundary.expires_at)] * tool_count
            + [("settled", 0.0)] * tool_count
        )
        persisted = await app.session_store.load_events(session_id)
        assert any(e.type == EventType.SESSION_INTERRUPTED for e in persisted)
        assert not any(e.type == EventType.SESSION_COMPLETED for e in persisted)
        assert len(provider.requests) == 1
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        # Cancellation settles the waiter, not the external effect. Keep the
        # original round available for exact reconciliation instead of inventing
        # failed tool outcomes and allowing the round to be retired.
        assert "pending_tool_round" in checkpoint
        assert "pending_session_interrupt" not in checkpoint
        terminals = [e for e in persisted if e.type == EventType.TOOL_CALL_FAILED]
        assert terminals == []
        uncertain = [e for e in persisted if e.type == EventType.TOOL_EFFECT_OUTCOME_UNKNOWN]
        assert {e.payload["tool_call_id"] for e in uncertain} == {
            f"call-{index}" for index in range(tool_count)
        }
        assert not any(e.type == EventType.TOOL_CALL_COMPLETED for e in persisted)
        terminal = next(e for e in persisted if e.type is EventType.SESSION_INTERRUPTED)
        assert terminal.payload["reason"] == "limit_reached"
        assert terminal.payload["limit"] == "execution_deadline"
        assert terminal.payload["execution_deadline"]["remaining_seconds"] == 0.0
        assert terminal.payload["failure_evidence"]["classification"] == "deadline"
        assert terminal.payload["failure_evidence"]["deadline_phase"] == "in_flight"
        await loop.shutdown_asyncgens()
        await asyncio.sleep(0)
        assert loop_errors == []

    async def bounded_run():
        async with asyncio.timeout(30):
            await run()

    asyncio.run(bounded_run())


def test_failed_step_retry_does_not_receive_a_fresh_child_duration(clock):
    from cayu.workflows.models import StepError

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
            with pytest.raises(StepError) as denied:
                await step(
                    workflow.context("retry"),
                    agent="worker",
                    step_id="same",
                    prompt="go",
                    run_options=StepRunOptions(execution_deadline=ExecutionDeadline.after(10)),
                )
        assert isinstance(denied.value.__cause__, ExecutionDeadlineExceeded)
        assert denied.value.evidence.deadline_phase == "admission"
        assert denied.value.evidence.deadline.expires_at == original_child.expires_at
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

    from cayu.runtime._isolated_tool_process import _project_context
    from cayu.tools.base import ToolContext

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
        "import sys\nfrom cayu.tools.isolated import ProcessIsolatedToolContext; "
        "from cayu.deadlines import bind_execution_deadline\nfrom cayu import RunRequest; "
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
    from cayu.providers.retry_policy import RetryPolicy

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
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

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
    from cayu.sessions.requests import ForkSessionRequest

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
    from cayu.tools.base import ToolContext
    from cayu.tools.isolated import ProcessIsolatedToolContext

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


@pytest.mark.parametrize("streamed", [False, True])
def test_owned_deadline_timeout_has_typed_diagnostics_after_cleanup(streamed):
    from cayu.deadlines import _deadline_expired_in_task, deadline_stream

    async def run():
        boundary = ExecutionDeadline.after(0.01, source="test", scope="case")
        joined = False

        async def producer():
            nonlocal joined
            try:
                await asyncio.Event().wait()
                yield "unreachable"
            finally:
                await asyncio.sleep(0)
                joined = True

        with pytest.raises(ExecutionDeadlineExceeded) as raised:
            if streamed:
                async for _ in deadline_stream(producer(), boundary):
                    pass
            else:
                async with execution_deadline_scope(boundary):
                    async for _ in producer():
                        pass
        assert joined
        assert raised.value.diagnostics["source"] == "test"
        assert raised.value.diagnostics["scope"] == "case"
        assert raised.value.diagnostics["remaining_seconds"] == 0
        assert raised.value.__dict__["execution_deadline"] == boundary.inspection()
        assert _deadline_expired_in_task(raised.value, asyncio.current_task())

    asyncio.run(run())


@pytest.mark.parametrize("during_cleanup", [False, True])
def test_deadline_scope_preserves_independent_timeout_errors(during_cleanup):
    async def run():
        failure = TimeoutError("application cleanup timed out")
        with pytest.raises(TimeoutError) as raised:
            async with execution_deadline_scope(ExecutionDeadline.after(0.01)):
                if during_cleanup:
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        raise failure from None
                raise failure
        assert raised.value is failure
        assert type(raised.value) is TimeoutError

    asyncio.run(run())


@pytest.mark.parametrize("worker_owned", [False, True])
@pytest.mark.parametrize("blocked", ["provider", "tool", "session_started"])
@pytest.mark.parametrize("sqlite", [False, True])
def test_task_deadline_releases_ownership_after_cancellation_drains(
    worker_owned,
    blocked,
    sqlite,
    sqlite_resources,
    monkeypatch,
):
    from cayu import Tool, ToolEffect, ToolResult, ToolSpec
    from cayu.storage import SQLiteTaskStore
    from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskStatus, run_task_worker

    async def scenario():
        async with sqlite_resources as resources:
            store = (
                resources.own(SQLiteTaskStore(resources.path())) if sqlite else InMemoryTaskStore()
            )
            entered, settled = [], []

            class StalledProvider(ScriptedModelProvider):
                async def stream(self, request):
                    entered.append("provider")
                    try:
                        await asyncio.Event().wait()
                        yield ModelStreamEvent.completed()
                    finally:
                        await asyncio.sleep(0)
                        settled.append("provider")

            class StalledTool(Tool):
                spec = ToolSpec(name="wait", effect=ToolEffect.NONE)

                async def run(self, ctx, arguments):
                    entered.append("tool")
                    try:
                        await asyncio.Event().wait()
                        return ToolResult(content="unreachable")
                    finally:
                        await asyncio.sleep(0)
                        settled.append("tool")

            provider = (
                StalledProvider([])
                if blocked == "provider"
                else ScriptedModelProvider(
                    [
                        ModelStreamEvent.tool_call(id="wait", name="wait", arguments={}),
                        ModelStreamEvent.completed(),
                    ]
                )
            )
            app = CayuApp(task_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="worker", model="scripted"), tools=[StalledTool()])
            if blocked == "session_started":
                emit = app._event_writer.emit

                async def stall_after_session_started(event):
                    recorded = await emit(event)
                    if event.type is EventType.SESSION_STARTED:
                        entered.append("session_started")
                        try:
                            await asyncio.Event().wait()
                        finally:
                            settled.append("session_started")
                    return recorded

                monkeypatch.setattr(app._event_writer, "emit", stall_after_session_started)
            task = await store.create_task(TaskCreate(type="deadline-test"))

            async def handler(app, task, worker_id=None):
                async for _ in app.run(
                    RunRequest(
                        agent_name="worker",
                        session_id="timed-out-task",
                        messages=[Message.text("user", "go")],
                        task_id=task.id,
                        task_worker_id=worker_id,
                        task_lease_expires_at=task.lease_expires_at,
                        execution_deadline=ExecutionDeadline.after(5),
                    )
                ):
                    pass

            if worker_owned:
                assert (
                    await asyncio.wait_for(
                        run_task_worker(
                            app,
                            store,
                            handler,
                            worker_id="deadline-worker",
                            max_tasks=1,
                        ),
                        15,
                    )
                    == 1
                )
            else:
                await asyncio.wait_for(handler(app, task), 15)
            final = await store.load_task(task.id)
            assert entered == [blocked]
            assert settled == [blocked]
            assert final.status is TaskStatus.FAILED
            assert final.worker_id is None and final.lease_expires_at is None
            assert final.error["error"] == "ExecutionDeadlineExceeded"
            assert final.error["failure_evidence"]["classification"] == "deadline"
            assert final.error["failure_evidence"]["settlement"] == "unknown"
            records = await app.session_store.load_events("timed-out-task")
            terminal = next(
                event for event in reversed(records) if event.type is EventType.SESSION_INTERRUPTED
            )
            assert final.error["deadline_failure"]["terminal_event_id"] == terminal.id

    asyncio.run(scenario())


def test_worker_default_does_not_change_serialized_requests():
    from cayu._validation import canonical_bounded_durable_json_bytes
    from cayu.collaboration.recipient_preparation import preparation_run_request
    from cayu.deadlines import default_run_deadline
    from cayu.runtime._durable_subagents import durable_subagent_request_sha256

    original = RunRequest(agent_name="a", messages=[])
    document = original.model_dump(mode="json")
    encoded = canonical_bounded_durable_json_bytes(
        document, "request", max_bytes=16384, max_nodes=8192
    ).decode()
    digest = durable_subagent_request_sha256(original)
    with default_run_deadline(ExecutionDeadline.after(60, source="task_worker")):
        assert RunRequest.model_validate(document).model_dump(mode="json") == document
        assert (
            RunRequest.model_validate_json(original.model_dump_json()).model_dump(mode="json")
            == document
        )
        assert RunRequest(agent_name="a", messages=[]).execution_deadline.expires_at is None
        restored = preparation_run_request(encoded)
        assert restored.model_dump(mode="json") == document
        assert durable_subagent_request_sha256(restored) == digest


def test_task_worker_execution_timeout_is_validated():
    from cayu.tasks import InMemoryTaskStore, run_task_worker

    async def handler(app, task, worker_id):
        raise AssertionError("unreachable")

    for bad in (-1, float("inf"), True):
        with pytest.raises(ValueError):
            asyncio.run(
                run_task_worker(
                    CayuApp(enable_logging=False),
                    InMemoryTaskStore(),
                    handler,
                    worker_id="w",
                    max_tasks=1,
                    execution_timeout_s=bad,
                )
            )


@pytest.mark.parametrize("copied", [False, True])
def test_task_worker_execution_timeout_defaults_runs_and_keeps_explicit_deadlines(copied):
    from cayu.sessions.requests import copy_run_request
    from cayu.tasks import (
        InMemoryTaskStore,
        TaskCreate,
        TaskRetryBudget,
        TaskStatus,
        run_task_worker,
    )
    from cayu.tasks.worker import complete_managed_task

    async def scenario():
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await store.create_task(
            TaskCreate(type="deadline-test", retry_budget=TaskRetryBudget.unlimited())
        )
        explicit = ExecutionDeadline.after(3600, source="application")
        app.register_provider(
            ScriptedModelProvider(
                [
                    [ModelStreamEvent.text_delta("ok"), ModelStreamEvent.completed({})]
                    for _ in range(3)
                ]
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="a", model="scripted"))
        seen = []

        async def handler(app, task, worker_id):
            for kwargs in (
                {},
                {"execution_deadline": explicit},
                {"execution_deadline": ExecutionDeadline()},
            ):
                request = RunRequest(
                    agent_name="a", messages=[Message.text("user", "hi")], **kwargs
                )
                if copied:
                    request = copy_run_request(request)
                events = [event async for event in app.run(request)]
                session = await app.session_store.load(events[-1].session_id)
                seen.append(session.execution_deadline)
            await complete_managed_task(store, task, worker_id, {})

        handled = await run_task_worker(
            app, store, handler, worker_id="w", max_tasks=1, execution_timeout_s=30
        )
        assert handled == 1
        assert (await store.load_task(task.id)).status is TaskStatus.COMPLETED
        default, kept, unbounded = seen
        assert default.source == "task_worker"
        assert 0 < default.remaining_seconds() <= 30
        assert kept.model_dump() == explicit.model_dump()
        assert unbounded.expires_at is None

    asyncio.run(scenario())


@pytest.mark.parametrize("attached", [True, False])
def test_task_worker_execution_timeout_ends_hung_run_and_releases_task(attached, clock):
    from cayu.deadlines import _TIMERS
    from cayu.sessions.outcomes import run_to_completion
    from cayu.tasks import (
        InMemoryTaskStore,
        TaskCreate,
        TaskHandlerOutcome,
        TaskRetryBudget,
        TaskStatus,
        run_task_worker,
    )

    async def scenario():
        store = InMemoryTaskStore()

        class StalledProvider(ScriptedModelProvider):
            async def stream(self, request):
                # Expire only after provider admission; setup time is host-dependent.
                clock.wall += timedelta(seconds=3600)
                clock.mono += 3600
                timers = [timer for timer, _deadline in _TIMERS.get() if timer.when() is not None]
                assert timers
                for timer in timers:
                    timer.reschedule(asyncio.get_running_loop().time())
                await asyncio.Event().wait()
                yield ModelStreamEvent.completed()

        app = CayuApp(task_store=store, enable_logging=False)
        app.register_provider(StalledProvider([]), default=True)
        app.register_agent(AgentSpec(name="worker", model="scripted"))
        task = await store.create_task(
            TaskCreate(type="deadline-test", retry_budget=TaskRetryBudget.unlimited())
        )
        outcomes = []

        async def handler(app, task, worker_id):
            # No deadline of its own, and a few attempts before giving up.
            task_fields = (
                {
                    "task_id": task.id,
                    "task_worker_id": worker_id,
                    "task_lease_expires_at": task.lease_expires_at,
                }
                if attached
                else {}
            )
            for attempt in range(3):
                outcome = await run_to_completion(
                    app,
                    RunRequest(
                        agent_name="worker",
                        session_id=f"hung-{attempt}",
                        messages=[Message.text("user", "go")],
                        **task_fields,
                    ),
                )
                outcomes.append(outcome)
                if attached:
                    # The documented handler shape hands off any interruption.
                    return (
                        TaskHandlerOutcome.SESSION_INTERRUPTED
                        if outcome.status is SessionStatus.INTERRUPTED
                        else None
                    )
                if outcome.ok:
                    return None
            raise RuntimeError(outcome.error)

        handled = await asyncio.wait_for(
            run_task_worker(
                app, store, handler, worker_id="w", max_tasks=1, execution_timeout_s=60
            ),
            15,
        )
        assert handled == 1
        final = await store.load_task(task.id)
        assert final.status is TaskStatus.FAILED
        assert final.worker_id is None and final.lease_expires_at is None
        first, *retries = outcomes
        assert first.status is SessionStatus.INTERRUPTED
        assert first.interruption.reason == "limit_reached"
        assert first.interruption.limit == "execution_deadline"
        records = await app.session_store.load_events("hung-0")
        assert any(
            record.type is EventType.SESSION_INTERRUPTED
            and record.payload["reason"] == "limit_reached"
            for record in records
        )
        assert (await app.session_store.load("hung-0")).execution_deadline.source == "task_worker"
        if attached:
            # The runtime fails the attached task once the run settles.
            assert retries == []
            assert final.error["error"] == "ExecutionDeadlineExceeded"
        else:
            # The limit has passed, so each retry is refused at once, and the
            # task fails with the handler's time limit.
            assert [outcome.status for outcome in retries] == [SessionStatus.FAILED] * 2
            assert all("ExecutionDeadlineExceeded" in outcome.error for outcome in retries)
            assert final.error["error"] == "ExecutionDeadlineExceeded"
            assert final.error["scope"] == "task_handler"
            assert final.error["session_ids"] == ["hung-0"]
            assert final.error["handler_failure"]["error"] == "RuntimeError"

    asyncio.run(scenario())


@pytest.mark.parametrize("bounded", [False, True])
def test_worker_default_does_not_tighten_subagents_or_workflow_anchors(bounded):
    from cayu.deadlines import default_run_deadline
    from cayu.sessions.queries import SessionQuery
    from cayu.tools.subagents import SubagentSpec, SubagentTool

    async def scenario():
        boundary = ExecutionDeadline.after(3600) if bounded else ExecutionDeadline()
        app = CayuApp(enable_logging=False)
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call", name="subagent", arguments={"agent": "child", "task": "answer"}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.text_delta("child done"), ModelStreamEvent.completed({})],
                [ModelStreamEvent.text_delta("parent done"), ModelStreamEvent.completed({})],
            ]
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="scripted"),
            tools=[SubagentTool(app, agents={"child": SubagentSpec(agent_name="child")})],
        )
        app.register_agent(AgentSpec(name="child", model="scripted"))
        with default_run_deadline(ExecutionDeadline.after(60, source="task_worker")):
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="parent",
                        session_id="parent",
                        messages=[Message.text("user", "delegate")],
                        execution_deadline=boundary,
                    )
                )
            ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        children = (
            await app.session_store.list_sessions(SessionQuery(parent_session_id="parent"))
        ).sessions
        assert len(children) == 1
        assert children[0].execution_deadline.expires_at == boundary.expires_at

        class AnchorWorkflow(WorkflowBase):
            spec = WorkflowSpec(name="anchor")

            async def run(self, session_id):
                ctx = self.context(session_id)
                yield await ctx.start()
                yield await ctx.completed({"done": True})

        with default_run_deadline(ExecutionDeadline.after(60, source="task_worker")):
            events = [
                event
                async for event in AnchorWorkflow(app).execute(
                    "anchor", execution_deadline=boundary
                )
            ]
        anchor = await app.session_store.load("anchor")
        assert anchor.execution_deadline.expires_at == boundary.expires_at
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("late_cancellation", [False, True])
def test_setup_interruption_cleanup_preserves_deadline_stream_cancellation(late_cancellation):
    from cayu.deadlines import _TIMERS, deadline_stream
    from cayu.runtime._session_engine import _settle_setup_interruption

    async def scenario():
        boundary = ExecutionDeadline.after(60)
        cleaned = False
        caller = asyncio.current_task()

        async def producer():
            nonlocal cleaned
            try:
                # Expiry observed between producer events adds a checkpoint.
                # Caller cancellation arrives there, while the producer is suspended.
                boundary._expired_observed = True
                if not late_cancellation:
                    caller.cancel("caller")
                yield "setup"
            finally:

                async def finalize():
                    nonlocal cleaned
                    # Interruption finalization normalizes its own task's state.
                    task = asyncio.current_task()
                    while task.cancelling():
                        task.uncancel()
                    if late_cancellation:
                        caller.cancel("caller")
                    # The run timer fires while that finalization is in progress.
                    timer, _ = _TIMERS.get()[-1]
                    timer.reschedule(asyncio.get_running_loop().time())
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
                    cleaned = True

                await _settle_setup_interruption(finalize)

        with pytest.raises(asyncio.CancelledError, match="caller"):
            async for _ in deadline_stream(producer(), boundary):
                pass
        assert cleaned

    asyncio.run(scenario())


def test_run_setup_cancellation_survives_deadline_during_finalization(monkeypatch):
    from cayu.deadlines import _TIMERS
    from cayu.sessions.outcomes import run_to_completion

    async def scenario():
        app, _ = _app()
        boundary = ExecutionDeadline.after(60)
        run = app._session_engine.run
        finalize = app._session_finalization.handle_session_interrupted
        finalized = False

        async def cancel_after_setup(**kwargs):
            injected = False
            async with aclosing(run(**kwargs)) as stream:
                async for event in stream:
                    if not injected:
                        injected = True
                        current_execution_deadline()._expired_observed = True
                        asyncio.current_task().cancel("setup caller")
                    yield event

        async def finalize_with_run_expiry(**kwargs):
            nonlocal finalized
            timer, _ = _TIMERS.get()[-1]
            timer.reschedule(asyncio.get_running_loop().time())
            async for event in finalize(**kwargs):
                yield event
            finalized = True

        monkeypatch.setattr(app._session_engine, "run", cancel_after_setup)
        monkeypatch.setattr(
            app._session_finalization, "handle_session_interrupted", finalize_with_run_expiry
        )
        try:
            task = asyncio.create_task(
                run_to_completion(
                    app,
                    RunRequest(
                        agent_name="worker",
                        session_id="setup-cancel",
                        messages=[],
                        execution_deadline=boundary,
                    ),
                )
            )
            with pytest.raises(asyncio.CancelledError, match="setup caller"):
                await task
            assert finalized
            session = await app.session_store.load("setup-cancel")
            assert session.status is SessionStatus.INTERRUPTED
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("prestarted", [False, True])
@pytest.mark.parametrize("exit_kind", ["deadline", "close"])
def test_deadline_stream_keeps_producer_context_and_task_through_cleanup(prestarted, exit_kind):
    from contextvars import ContextVar

    from cayu.deadlines import deadline_stream

    async def scenario():
        state = ContextVar("workflow_state", default="outside")
        boundary = ExecutionDeadline.after(60)
        owner = asyncio.current_task()
        cleaned = False

        async def producer():
            nonlocal cleaned
            token = state.set("inside")
            try:
                yield "first"
                assert asyncio.current_task() is owner
                assert state.get() == "inside"
                yield "second"
            finally:
                assert asyncio.current_task() is owner
                state.reset(token)
                cleaned = True

        source = producer()
        if prestarted:
            assert await anext(source) == "first"
        stream = deadline_stream(source, boundary)
        if not prestarted:
            assert await anext(stream) == "first"
        assert await anext(stream) == "second"
        if exit_kind == "deadline":
            boundary._expired_observed = True
            with pytest.raises(ExecutionDeadlineExceeded):
                await anext(stream)
        else:
            await stream.aclose()
        assert cleaned
        assert state.get() == "outside"

    asyncio.run(scenario())


@pytest.mark.parametrize("external_cancellation", [False, True])
def test_deadline_stream_cleanup_error_keeps_primary_failure_catchable(external_cancellation):
    from cayu.deadlines import deadline_stream

    async def scenario():
        boundary = ExecutionDeadline.after(60)
        cleanup_error = RuntimeError("cleanup store write failed")

        async def producer():
            try:
                yield "first"
                if external_cancellation:
                    boundary._expired_observed = True
                    asyncio.current_task().cancel("caller")
                yield "second"
            finally:
                raise cleanup_error

        stream = deadline_stream(producer(), boundary)
        assert await anext(stream) == "first"
        if external_cancellation:
            expected = asyncio.CancelledError
        else:
            boundary._expired_observed = True
            expected = ExecutionDeadlineExceeded
        with pytest.raises(expected) as caught:
            await anext(stream)
        if external_cancellation:
            assert str(caught.value) == "caller"
        else:
            assert isinstance(caught.value, Exception)
            assert asyncio.current_task().cancelling() == 0
        causes = []
        cause = caught.value
        while cause is not None and cause not in causes:
            causes.append(cause)
            cause = cause.__cause__
        assert cleanup_error in causes

    asyncio.run(scenario())


@pytest.mark.parametrize("fatal", [False, True])
def test_setup_interruption_preserves_concurrent_cancellation_and_cleanup_failure(fatal):
    from cayu._exception_groups import iter_exception_tree
    from cayu.runtime._session_engine import _settle_setup_interruption

    async def scenario():
        caller = asyncio.current_task()
        failure = SystemExit("cleanup control") if fatal else RuntimeError("cleanup failed")
        cleaned = False

        async def finalize():
            nonlocal cleaned
            caller.cancel("caller")
            await asyncio.sleep(0)
            cleaned = True
            raise failure

        expected = BaseExceptionGroup if fatal else asyncio.CancelledError
        with pytest.raises(expected) as caught:
            await _settle_setup_interruption(finalize)
        assert cleaned
        assert caller.cancelling() == 1
        if fatal:
            leaves = list(iter_exception_tree(caught.value))
            assert failure in leaves
            assert any(isinstance(part, asyncio.CancelledError) for part in leaves)
        else:
            assert str(caught.value) == "caller"
            assert caught.value.__cause__ is failure

    asyncio.run(scenario())
