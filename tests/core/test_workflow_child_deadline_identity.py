"""Controlled child publication and cancellation boundaries; no provider credentials."""

import asyncio
import contextlib

import pytest
from tests.core.test_workflow_failure_evidence import Verifiers

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    ExecutionDeadline,
    InMemorySessionStore,
    ModelStreamEvent,
    ScriptedModelProvider,
    SQLiteSessionStore,
    StepError,
    Tool,
    ToolResult,
    ToolSpec,
    execution_deadline_scope,
    parallel,
    step,
)
from cayu.failure_evidence import exception_evidence


@pytest.mark.anyio
@pytest.mark.parametrize("sqlite", [False, True])
@pytest.mark.parametrize(
    "stop,view",
    [
        (stop, view)
        for stop in ("deadline", "cancel", "repeated")
        for view in (
            "normal",
            "missing_terminal",
            "read_failure",
            "newer_run",
            "terminal_write_failure",
        )
    ]
    + [("repeated_terminal", "normal")],
)
async def test_child_identity_after_tool_publication(tmp_path, monkeypatch, sqlite, stop, view):
    store = SQLiteSessionStore(tmp_path / "child.db") if sqlite else InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    tool_calls = []

    class Probe(Tool):
        spec = ToolSpec(name="probe", input_schema={"type": "object", "properties": {}})

        async def run(self, ctx, args):
            tool_calls.append(ctx.session_id)
            return ToolResult(content="settled result")

    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="probe-call", name="probe", arguments={}),
                    ModelStreamEvent.completed({}),
                ],
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="worker", model="test"), tools=[Probe()])
    ctx = Verifiers(app).context("parent")
    await ctx.start()
    published = asyncio.Event()
    original_run = app.run
    child_id = None

    async def paused_run(request):
        nonlocal child_id
        child_id = request.session_id
        async with contextlib.aclosing(original_run(request)) as stream:
            async for event in stream:
                if event.type == EventType.TOOL_CALL_COMPLETED:
                    published.set()
                    await asyncio.Event().wait()
                yield event

    original_append = store.append_event
    terminal_published = asyncio.Event()
    release_terminal = asyncio.Event()

    async def append_without_terminal(session_id, event):
        if view == "terminal_write_failure" and event.type == EventType.SESSION_INTERRUPTED:
            raise OSError("synthetic terminal publication failure")
        result = await original_append(session_id, event)
        if stop == "repeated_terminal" and event.type == EventType.SESSION_INTERRUPTED:
            terminal_published.set()
            await release_terminal.wait()
        return result

    monkeypatch.setattr(store, "append_event", append_without_terminal)
    monkeypatch.setattr(app, "run", paused_run)
    import cayu.workflows.workflow as workflow_module

    original_failure_state = workflow_module._child_failure_state

    async def failure_view(context, session_id):
        if view == "read_failure":
            raise OSError("synthetic diagnostic read failure")
        state = await original_failure_state(context, session_id)
        if view == "missing_terminal":
            state.evidence = state.evidence.model_copy(update={"terminal_event_id": None})
        elif view == "newer_run":
            state.run_epoch += 1
            state.evidence = state.evidence.model_copy(
                update={
                    "run_epoch": state.run_epoch,
                    "terminal_event_id": "newer-run-terminal",
                }
            )
        return state

    monkeypatch.setattr(workflow_module, "_child_failure_state", failure_view)
    original_recover = app.recover_incomplete_session
    recovering = asyncio.Event()
    release_recovery = asyncio.Event()

    async def paused_recover(request):
        recovering.set()
        if stop == "repeated":
            await release_recovery.wait()
        return await original_recover(request)

    monkeypatch.setattr(app, "recover_incomplete_session", paused_recover)
    timer = None

    async def invoke():
        nonlocal timer
        async with execution_deadline_scope(ExecutionDeadline.after(60)) as timer:
            await step(ctx, agent="worker", step_id="check", prompt="go")

    task = asyncio.create_task(invoke())
    await asyncio.wait_for(published.wait(), 10)
    if stop == "deadline":
        timer.reschedule(asyncio.get_running_loop().time())
    else:
        task.cancel()
    if stop == "repeated_terminal":
        await asyncio.wait_for(terminal_published.wait(), 10)
        task.cancel("second signal at terminal acknowledgement")
        await asyncio.sleep(0)
        task.cancel("third signal at terminal acknowledgement")
        release_terminal.set()
    if stop == "repeated":
        await asyncio.wait_for(recovering.wait(), 10)
        task.cancel("second signal")
        await asyncio.sleep(0)
        task.cancel("third signal")
        release_recovery.set()
    with pytest.raises(TimeoutError if stop == "deadline" else asyncio.CancelledError) as caught:
        await asyncio.wait_for(task, 10)
    evidence = exception_evidence(caught.value)
    events = await store.load_events(child_id)
    started = next(e for e in events if e.type == EventType.SESSION_STARTED)
    terminals = [
        e for e in events if e.type in {EventType.SESSION_FAILED, EventType.SESSION_INTERRUPTED}
    ]
    assert evidence.session_id == child_id
    assert evidence.run_epoch == started.payload["run_epoch"]
    assert len(terminals) == (0 if view == "terminal_write_failure" else 1)
    assert evidence.terminal_event_id == (terminals[0].id if view == "normal" else None)
    assert evidence.secondary_failures is (view == "terminal_write_failure")
    assert evidence.classification == ("deadline" if stop == "deadline" else "interruption")
    assert evidence.settlement == "unknown"
    assert tool_calls == [child_id]
    monkeypatch.setattr(workflow_module, "_child_failure_state", original_failure_state)
    monkeypatch.setattr(app, "run", original_run)
    monkeypatch.setattr(app, "recover_incomplete_session", original_recover)
    replay_ctx = Verifiers(app).context("parent")
    await replay_ctx.start()
    with pytest.raises(StepError):
        await step(replay_ctx, agent="worker", step_id="check", prompt="go")
    assert tool_calls == [child_id]
    if sqlite:
        await store.close()
        reopened = SQLiteSessionStore(tmp_path / "child.db")
        assert [e.id for e in await reopened.load_events(child_id)] == [e.id for e in events]
        await reopened.close()


@pytest.mark.anyio
@pytest.mark.parametrize("sqlite", [False, True])
@pytest.mark.parametrize("phase", ["before_create", "after_create"])
async def test_deadline_at_create_acknowledgement(tmp_path, monkeypatch, sqlite, phase):
    reached = asyncio.Event()
    child_id = None
    created_epoch = None

    class PausedCreateStore(SQLiteSessionStore if sqlite else InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def create(self, request, **kwargs):
            nonlocal child_id, created_epoch
            if request.session_id == "parent":
                return await super().create(request, **kwargs)
            child_id = request.session_id
            if phase == "before_create":
                reached.set()
                await asyncio.Event().wait()
            result = await super().create(request, **kwargs)
            created_epoch = result.run_epoch
            reached.set()
            await asyncio.Event().wait()
            return result

    store = PausedCreateStore(tmp_path / "create.db") if sqlite else PausedCreateStore()
    provider = ScriptedModelProvider([])
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker", model="test"))
    ctx = Verifiers(app).context("parent")
    await ctx.start()
    timer = None

    async def invoke():
        nonlocal timer
        async with execution_deadline_scope(ExecutionDeadline.after(60)) as timer:
            await step(ctx, agent="worker", step_id="check", prompt="go")

    task = asyncio.create_task(invoke())
    await asyncio.wait_for(reached.wait(), 10)
    timer.reschedule(asyncio.get_running_loop().time())
    with pytest.raises(TimeoutError) as caught:
        await asyncio.wait_for(task, 10)
    evidence = exception_evidence(caught.value)
    assert evidence.classification == "deadline"
    assert evidence.settlement == "unknown"
    assert not provider.requests
    child = await store.load(child_id)
    if phase == "before_create":
        assert child is None
        assert evidence.session_id is evidence.run_epoch is evidence.terminal_event_id is None
    else:
        assert child is not None
        terminal = next(
            e for e in await store.load_events(child_id) if e.type == EventType.SESSION_INTERRUPTED
        )
        assert evidence.session_id == child_id
        assert evidence.run_epoch == created_epoch
        # Pre-start recovery currently emits a terminal without an executing
        # epoch. Do not manufacture a run-to-terminal association from status.
        assert "failure_evidence" not in terminal.payload
        assert evidence.terminal_event_id is None
    if sqlite:
        await store.close()


def test_failure_group_does_not_borrow_another_child_identity():
    from cayu import FailureEvidence
    from cayu.failure_evidence import retain_child_failure_identity

    first, second = asyncio.CancelledError(), asyncio.CancelledError()
    retain_child_failure_identity(
        first, FailureEvidence(session_id="first", run_epoch=1, terminal_event_id="terminal-first")
    )
    retain_child_failure_identity(
        second,
        FailureEvidence(session_id="second", run_epoch=2, terminal_event_id="terminal-second"),
    )
    wrapped = TimeoutError()
    wrapped.__cause__ = first
    assert exception_evidence(wrapped).session_id == "first"
    group = BaseExceptionGroup("multiple children", [first, second])
    evidence = exception_evidence(group)
    assert evidence.session_id is evidence.run_epoch is evidence.terminal_event_id is None
    assert evidence.secondary_failures
    truncated = BaseExceptionGroup("bounded evidence", [first, *[ValueError() for _ in range(16)]])
    evidence = exception_evidence(truncated)
    assert evidence.truncated
    assert evidence.session_id is None


@pytest.mark.anyio
@pytest.mark.parametrize("sqlite", [False, True])
async def test_successful_child_replay_does_not_repeat_tool(tmp_path, sqlite):
    store = SQLiteSessionStore(tmp_path / "success.db") if sqlite else InMemorySessionStore()
    tool_calls = []

    class Probe(Tool):
        spec = ToolSpec(name="probe", input_schema={"type": "object", "properties": {}})

        async def run(self, ctx, args):
            tool_calls.append(ctx.session_id)
            return ToolResult(content="settled")

    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(id="probe-call", name="probe", arguments={}),
                ModelStreamEvent.completed({}),
            ],
            [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})],
        ]
    )

    def make_app(store):
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker", model="test"), tools=[Probe()])
        return app

    app = make_app(store)
    ctx = Verifiers(app).context("parent")
    await ctx.start()
    result = await step(ctx, agent="worker", step_id="check", prompt="go")
    assert result.text == "done"
    if sqlite:
        await store.close()
        store = SQLiteSessionStore(tmp_path / "success.db")
        app = make_app(store)
    replay_ctx = Verifiers(app).context("parent")
    await replay_ctx.start()
    replay = await step(replay_ctx, agent="worker", step_id="check", prompt="go")
    assert replay == result
    assert tool_calls == [result.session_id]
    assert len(provider.requests) == 2
    if sqlite:
        await store.close()


@pytest.mark.anyio
@pytest.mark.parametrize("cleanup_group", [False, True])
async def test_parallel_collects_wrapped_child_deadline_identity(monkeypatch, cleanup_group):
    entered = asyncio.Event()
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [[ModelStreamEvent.text_delta("published"), ModelStreamEvent.completed({})]]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="worker", model="test"))
    ctx = Verifiers(app).context("parent")
    await ctx.start()
    original_run = app.run
    child_id = None

    async def paused_run(request):
        nonlocal child_id
        child_id = request.session_id
        async with contextlib.aclosing(original_run(request)) as stream:
            async for event in stream:
                if event.type == EventType.MODEL_TEXT_DELTA:
                    entered.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError as exc:
                        if cleanup_group:
                            raise BaseExceptionGroup(
                                "synthetic cleanup", [exc, ValueError()]
                            ) from exc
                        raise
                yield event

    monkeypatch.setattr(app, "run", paused_run)
    timer = None

    async def invoke():
        nonlocal timer
        async with execution_deadline_scope(ExecutionDeadline.after(60)) as timer:
            return await step(ctx, agent="worker", step_id="check", prompt="go")

    task = asyncio.create_task(parallel([invoke()]))
    await asyncio.wait_for(entered.wait(), 10)
    timer.reschedule(asyncio.get_running_loop().time())
    result = await asyncio.wait_for(task, 10)
    (failure,) = result.failures
    assert failure.session_id == failure.evidence.session_id == child_id
    assert failure.evidence.classification == "deadline"
    assert failure.evidence.run_epoch is not None
    assert failure.evidence.terminal_event_id is not None
    assert failure.evidence.secondary_failures is cleanup_group
    assert failure.evidence.settlement == "unknown"
    assert not result.ok


@pytest.mark.anyio
@pytest.mark.parametrize("sqlite", [False, True])
@pytest.mark.parametrize("stop", ["cancel", "deadline", "repeated"])
@pytest.mark.parametrize("outcome", ["normal", "recovery_error", "cleanup_group", "read_failure"])
async def test_cancel_during_create_reconciliation(tmp_path, monkeypatch, sqlite, stop, outcome):
    store = SQLiteSessionStore(tmp_path / "reconcile.db") if sqlite else InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    provider = ScriptedModelProvider([])
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker", model="test"))
    ctx = Verifiers(app).context("parent")
    await ctx.start()
    original_run = app.run
    original_recover = app.recover_incomplete_session
    entered, release = asyncio.Event(), asyncio.Event()
    child_id = None
    authenticated_epoch = None

    async def lost_ack(request):
        nonlocal child_id
        child_id = request.session_id
        async with contextlib.aclosing(original_run(request)) as stream:
            await anext(stream)
            raise OSError("lost stream acknowledgement")
            yield  # pragma: no cover - async generator boundary

    async def paused_recover(request):
        nonlocal authenticated_epoch
        child = await store.load(request.session_id)
        authenticated_epoch = child.run_epoch
        entered.set()
        await release.wait()
        if outcome == "recovery_error":
            raise OSError("reconciliation failed")
        if outcome == "cleanup_group":
            raise BaseExceptionGroup("cleanup", [asyncio.CancelledError(), ValueError()])
        return await original_recover(request)

    monkeypatch.setattr(app, "run", lost_ack)
    monkeypatch.setattr(app, "recover_incomplete_session", paused_recover)
    if outcome == "read_failure":
        import cayu.workflows.workflow as workflow_module

        async def failed_read(*args):
            raise OSError("diagnostic read failed")

        monkeypatch.setattr(workflow_module, "_child_failure_state", failed_read)
    timer = None

    async def invoke():
        nonlocal timer
        async with execution_deadline_scope(ExecutionDeadline.after(60)) as timer:
            return await step(ctx, agent="worker", step_id="check", prompt="go")

    task = asyncio.create_task(invoke())
    await asyncio.wait_for(entered.wait(), 10)
    if stop == "deadline":
        timer.reschedule(asyncio.get_running_loop().time())
    else:
        task.cancel("first cancellation")
    await asyncio.sleep(0)
    if stop == "repeated":
        task.cancel("second cancellation")
        await asyncio.sleep(0)
        task.cancel("third cancellation")
        await asyncio.sleep(0)
    release.set()
    with pytest.raises(TimeoutError if stop == "deadline" else asyncio.CancelledError) as caught:
        await asyncio.wait_for(task, 10)
    evidence = exception_evidence(caught.value)
    assert evidence.session_id == child_id
    assert evidence.run_epoch == authenticated_epoch
    assert evidence.classification == ("deadline" if stop == "deadline" else "interruption")
    assert evidence.secondary_failures is (outcome in {"recovery_error", "cleanup_group"})
    assert evidence.settlement == "unknown"
    if stop != "deadline":
        assert caught.value.args == ("first cancellation",)
    if evidence.terminal_event_id is not None:
        terminal = next(
            e for e in await store.load_events(child_id) if e.id == evidence.terminal_event_id
        )
        assert terminal.payload["failure_evidence"]["run_epoch"] == authenticated_epoch
    assert not provider.requests
    if sqlite:
        await store.close()
