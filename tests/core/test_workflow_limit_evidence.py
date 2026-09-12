"""Native hard-limit identity, accounting, admission, and durable replay."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    InMemorySessionStore,
    ModelStreamEvent,
    RunLimits,
    RuntimeHook,
    ScriptedModelProvider,
    SQLiteSessionStore,
    StepError,
    Tool,
    ToolResult,
    ToolSpec,
    WorkflowBase,
    WorkflowSpec,
    parallel,
    step,
)
from cayu.workflows import StepRunOptions


class Workflow(WorkflowBase):
    spec = WorkflowSpec(name="limit-evidence")

    async def run(self, session_id):
        yield await self.context(session_id).start()


@pytest.mark.parametrize("sqlite", [False, True])
@pytest.mark.parametrize(
    "kind",
    [
        "total_tokens",
        "proposed_tool",
        "cumulative",
        "input_tokens",
        "output_tokens",
        "tool_calls",
        "model_steps",
        "elapsed_seconds",
    ],
)
def test_native_limit_identity_and_replay(tmp_path, sqlite, kind):
    async def run():
        now = datetime.now(UTC)
        calls = []
        executed = []
        completed = []

        class CompletionHook(RuntimeHook):
            async def after_session_completed(self, context):
                completed.append(context.session.id)

        class Probe(Tool):
            spec = ToolSpec(name="probe", input_schema={"type": "object", "properties": {}})

            async def run(self, ctx, args):
                executed.append(ctx.session_id)
                return ToolResult(content="ok")

        class Provider(ScriptedModelProvider):
            async def stream(self, request):
                nonlocal now
                calls.append(request.model)
                limited = request.model == "limited"
                completion = calls.count("limited")
                if limited and kind in {"proposed_tool", "cumulative", "tool_calls", "model_steps"}:
                    yield ModelStreamEvent.tool_call(
                        id=f"call-{completion}", name="probe", arguments={}
                    )
                else:
                    yield ModelStreamEvent.text_delta("synthetic response")
                if limited and kind == "elapsed_seconds":
                    now += timedelta(seconds=2)
                tokens = (550 if kind in {"cumulative", "tool_calls"} else 1100) if limited else 10
                if kind in {"total_tokens", "proposed_tool"}:
                    yield ModelStreamEvent.completed(
                        {
                            "usage": {
                                "input_tokens": tokens,
                                "output_tokens": 10,
                                "total_tokens": tokens + 10,
                            }
                        }
                    )
                    return
                yield ModelStreamEvent.completed(
                    {
                        "usage_metrics": {
                            "provider_name": self.name,
                            "model": request.model,
                            "input_tokens": tokens,
                            "output_tokens": 10,
                            "total_tokens": tokens + 10,
                            "cache": {
                                "read_tokens": 500 if limited else 0,
                                "cached_input_tokens": 500 if limited else 0,
                                "uncached_input_tokens": tokens - (500 if limited else 0),
                            },
                        }
                    }
                )

        path = tmp_path / "sessions.db"
        # Session creation and elapsed-limit evaluation must share the test
        # clock; real setup latency must not consume the synthetic advance.
        store = (
            SQLiteSessionStore(path, ownership_clock=lambda: now)
            if sqlite
            else InMemorySessionStore(ownership_clock=lambda: now)
        )
        provider = Provider([])

        def make_app(store):
            app = CayuApp(
                enable_logging=False,
                clock=lambda: now,
                runtime_hooks=[CompletionHook()],
                **({"session_store": store} if store else {}),
            )
            app.register_provider(provider, default=True)
            for name in ("healthy", "limited"):
                app.register_agent(AgentSpec(name=name, model=name), tools=[Probe()])
            return app

        app = make_app(store)
        ctx = Workflow(app).context("root")
        await ctx.start()
        limit = (
            kind
            if kind
            in {"input_tokens", "output_tokens", "tool_calls", "model_steps", "elapsed_seconds"}
            else "total_tokens"
        )
        maximum = {
            "output_tokens": 10,
            "tool_calls": 1,
            "model_steps": 1,
            "elapsed_seconds": 1,
        }.get(limit, 1000)
        result = await parallel(
            [
                step(
                    ctx,
                    agent=name,
                    step_id=name,
                    prompt="check",
                    run_options=StepRunOptions(
                        limits=RunLimits(
                            **{f"max_{limit}": maximum},
                            scope="session" if kind == "elapsed_seconds" else "run",
                        )
                        if name == "limited" and kind != "model_steps"
                        else RunLimits(),
                        max_steps=1 if name == "limited" and kind == "model_steps" else 64,
                    ),
                )
                for name in ("healthy", "limited")
            ]
        )
        assert len(result.successes) == len(result.failures) == 1
        assert result.successes[0].text == "synthetic response"
        failure = result.failures[0]
        assert failure.workflow_attempt_id == ctx.attempt_id
        assert failure.step_id == "limited"
        evidence = failure.evidence
        assert evidence.classification == "interruption", result.failures[0].error
        assert evidence.deadline is None
        assert evidence.settlement == "unknown"
        assert not evidence.secondary_failures
        events = await app.session_store.load_events(failure.session_id)
        started = next(e for e in events if e.type == "session.started")
        assert evidence.run_epoch == started.payload["run_epoch"]
        terminal = next(e for e in events if e.id == evidence.terminal_event_id)
        assert terminal.type == "session.interrupted"
        assert terminal.payload["failure_evidence"]["run_epoch"] == evidence.run_epoch
        assert terminal.payload["reason"] == "limit_reached"
        assert terminal.payload["limit"] == limit
        assert terminal.payload["maximum"] == maximum
        usage = await app.get_session_usage(failure.session_id)
        assert usage.usage.total_tokens == (1120 if kind in {"cumulative", "tool_calls"} else 1110)
        assert usage.usage.input_tokens == 1100
        assert usage.usage.cache.cached_input_tokens == (
            0
            if kind in {"total_tokens", "proposed_tool"}
            else 1000
            if kind in {"cumulative", "tool_calls"}
            else 500
        )
        assert usage.model_steps == (2 if kind in {"cumulative", "tool_calls"} else 1)
        assert usage.tool_calls == len(executed)
        if kind == "elapsed_seconds":
            assert 1 <= terminal.payload["actual"] <= 2
        else:
            assert terminal.payload["actual"] == {
                "input_tokens": 1100,
                "output_tokens": 10,
                "tool_calls": 2,
                "model_steps": 1,
                "elapsed_seconds": 2,
            }.get(limit, usage.usage.total_tokens)
        assert len(executed) == (1 if kind in {"cumulative", "tool_calls", "model_steps"} else 0)
        assert len(calls) == (3 if kind in {"cumulative", "tool_calls"} else 2)
        assert completed == [result.successes[0].session_id]
        assert not any(e.type == "session.completed" for e in events)
        before = list(calls), list(executed)
        if sqlite:
            await store.close()
            store = SQLiteSessionStore(path, ownership_clock=lambda: now)
            app = make_app(store)
        replay = Workflow(app).context("root")
        await replay.start()
        healthy = await step(replay, agent="healthy", step_id="healthy", prompt="check")
        assert healthy.text == result.successes[0].text
        with pytest.raises(StepError) as raised:
            await step(replay, agent="limited", step_id="limited", prompt="check")
        assert raised.value.evidence == evidence
        assert raised.value.workflow_attempt_id == replay.attempt_id
        assert (calls, executed) == before
        if sqlite:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("race", ["newer_run", "missing", "stale", "lookup_failure"])
def test_limit_correlation_does_not_adopt_unresolved_or_newer_terminal(tmp_path, monkeypatch, race):
    from cayu import (
        ExecutionProfileAdoptionIntent,
        ExecutionProfileAuthorityDecision,
        ExecutionProfilePolicy,
        ExecutionProfilePolicyAction,
        ExecutionProfilePolicyResult,
        Message,
        ResolutionActor,
        ResolutionActorSource,
        ResumeRequest,
    )

    class AllowExplicitResume(ExecutionProfilePolicy):
        identity = "test:limit-race-resume:v1"

        async def decide(self, request):
            return ExecutionProfilePolicyResult(
                action=ExecutionProfilePolicyAction.ADOPT,
                reason="Explicit test-only resume of a workflow child.",
                authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
            )

    async def run():
        store = SQLiteSessionStore(tmp_path / "race.db")
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.completed(
                        {"usage": {"input_tokens": 1100, "output_tokens": 10, "total_tokens": 1110}}
                    )
                ]
            ]
        )
        app = CayuApp(
            enable_logging=False,
            session_store=store,
            execution_profile_policy=AllowExplicitResume(),
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="limited", model="test"))
        original_run = app.run
        original_load = store.load_events
        original_terminal = None
        newer_terminal = None

        async def run_with_race(request):
            nonlocal original_terminal, newer_terminal
            async for event in original_run(request):
                yield event
            events = await original_load(request.session_id)
            original_terminal = next(e for e in events if e.type == "session.interrupted")
            if race == "newer_run":
                # A real second invocation stops at admission against cumulative
                # usage. The workflow still observed only the first invocation.
                async for _ in app.resume(
                    ResumeRequest(
                        session_id=request.session_id,
                        messages=[Message.text("user", "resume")],
                        limits=RunLimits(max_total_tokens=1000, scope="session"),
                        profile_adoption=ExecutionProfileAdoptionIntent(
                            idempotency_key="resume-for-race",
                            reason="Exercise a newer invocation before workflow correlation.",
                            requested_by=ResolutionActor(
                                subject="test", source=ResolutionActorSource.REQUEST
                            ),
                        ),
                    )
                ):
                    pass
                newer_terminal = [
                    e
                    for e in await original_load(request.session_id)
                    if e.type == "session.interrupted"
                ][-1]
            else:

                async def faulty_lookup(session_id):
                    if session_id != request.session_id:
                        return await original_load(session_id)
                    if race == "lookup_failure":
                        raise OSError("store unavailable")
                    if race == "missing":
                        return [e for e in events if e.type != "session.interrupted"]
                    stale = original_terminal.model_copy(
                        update={
                            "payload": {
                                **original_terminal.payload,
                                "failure_evidence": {
                                    **original_terminal.payload["failure_evidence"],
                                    "run_epoch": 0,
                                },
                            }
                        }
                    )
                    return [stale if e.id == stale.id else e for e in events]

                monkeypatch.setattr(store, "load_events", faulty_lookup)

        monkeypatch.setattr(app, "run", run_with_race)
        ctx = Workflow(app).context("race-root")
        await ctx.start()
        result = await parallel(
            [
                step(
                    ctx,
                    agent="limited",
                    step_id="limited",
                    prompt="check",
                    run_options=StepRunOptions(limits=RunLimits(max_total_tokens=1000)),
                )
            ]
        )
        evidence = result.failures[0].evidence
        assert evidence.classification == "interruption", result.failures[0].error
        assert evidence.run_epoch == original_terminal.payload["failure_evidence"]["run_epoch"]
        assert evidence.terminal_event_id is None
        assert evidence.settlement == "unknown"
        if newer_terminal:
            assert newer_terminal.id != original_terminal.id
            assert newer_terminal.payload["failure_evidence"]["run_epoch"] > evidence.run_epoch
            resumed = [
                e
                for e in await original_load(result.failures[0].session_id)
                if e.type == "session.resumed"
            ][-1]
            assert (
                newer_terminal.payload["failure_evidence"]["run_epoch"]
                == resumed.payload["run_epoch"]
            )
        await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("boundary", ["session.limit_reached", "session.interrupted"])
def test_limit_publication_loses_to_store_epoch_fence(tmp_path, monkeypatch, boundary):
    from cayu import SessionStatus

    async def run():
        store = SQLiteSessionStore(tmp_path / "fence.db")
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.completed(
                        {"usage": {"input_tokens": 1100, "output_tokens": 10, "total_tokens": 1110}}
                    )
                ]
            ]
        )
        app = CayuApp(enable_logging=False, session_store=store)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="limited", model="test"))
        original_append = store.append_event
        fenced = None

        async def append_with_takeover(session_id, event):
            nonlocal fenced
            if event.type == boundary and fenced is None:

                async def takeover():
                    nonlocal fenced
                    fenced = await store.fence_run_and_transform_checkpoint(
                        session_id,
                        statuses={SessionStatus.RUNNING, SessionStatus.INTERRUPTED},
                        checkpoint_transform=lambda session, checkpoint: checkpoint or {},
                    )
                    await store.release_run_fence(session_id)

                # Ownership is task-local. The competing owner uses the real
                # SQLite transaction, then the old invocation attempts its write.
                await asyncio.create_task(takeover())
            await original_append(session_id, event)

        monkeypatch.setattr(store, "append_event", append_with_takeover)
        ctx = Workflow(app).context("fenced-root")
        await ctx.start()
        result = await parallel(
            [
                step(
                    ctx,
                    agent="limited",
                    step_id="limited",
                    prompt="check",
                    run_options=StepRunOptions(limits=RunLimits(max_total_tokens=1000)),
                )
            ]
        )
        failure = result.failures[0]
        assert fenced is not None
        assert "SessionRunFenced" in failure.evidence.exception_types
        assert failure.evidence.session_id == failure.session_id
        assert failure.evidence.terminal_event_id is None
        assert failure.evidence.settlement == "unknown"
        events = await store.load_events(failure.session_id)
        assert not any(
            e.type in {"session.interrupted", "session.failed", "session.completed"} for e in events
        )
        started = next(e for e in events if e.type == "session.started")
        assert failure.evidence.run_epoch == started.payload["run_epoch"]
        assert fenced.run_epoch > failure.evidence.run_epoch
        await store.close()

    asyncio.run(run())


def test_cleanup_failure_at_token_boundary_preserves_exception_evidence(tmp_path):
    async def run():
        calls = []

        class Provider(ScriptedModelProvider):
            async def stream(self, request):
                calls.append(request.model)
                try:
                    yield ModelStreamEvent.text_delta("synthetic response")
                    yield ModelStreamEvent.completed(
                        {
                            "usage": {
                                "input_tokens": 1100 if request.model == "limited" else 10,
                                "output_tokens": 10,
                                "total_tokens": 1110 if request.model == "limited" else 20,
                            }
                        }
                    )
                finally:
                    if request.model == "limited":
                        raise ExceptionGroup(
                            "cleanup failed", [RuntimeError("close"), OSError("cleanup")]
                        )

        path = tmp_path / "cleanup.db"
        store = SQLiteSessionStore(path)
        provider = Provider([])

        def make_app(store):
            app = CayuApp(enable_logging=False, session_store=store)
            app.register_provider(provider, default=True)
            for name in ("healthy", "limited"):
                app.register_agent(AgentSpec(name=name, model=name))
            return app

        app = make_app(store)
        ctx = Workflow(app).context("cleanup-root")
        await ctx.start()
        result = await parallel(
            [
                step(
                    ctx,
                    agent=name,
                    step_id=name,
                    session_id=name,
                    prompt="check",
                    run_options=StepRunOptions(limits=RunLimits(max_total_tokens=1000)),
                )
                for name in ("healthy", "limited")
            ]
        )
        assert result.successes[0].text == "synthetic response"
        evidence = result.failures[0].evidence
        assert evidence.classification == "failure"
        assert evidence.secondary_failures
        assert {"RuntimeError", "OSError"} <= set(evidence.exception_types)
        assert evidence.settlement == "unknown"
        terminal = next(
            e for e in await store.load_events("limited") if e.id == evidence.terminal_event_id
        )
        assert terminal.type == "session.failed"
        assert evidence.run_epoch == terminal.payload["failure_evidence"]["run_epoch"]
        await store.close()
        store = SQLiteSessionStore(path)
        ctx = Workflow(make_app(store)).context("cleanup-root")
        await ctx.start()
        with pytest.raises(StepError) as raised:
            await step(
                ctx, agent="limited", step_id="limited", session_id="limited", prompt="check"
            )
        assert raised.value.evidence == evidence
        assert sorted(calls) == ["healthy", "limited"]
        await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("session_id", ["s" * 2048, "é" * 1024], ids=["ascii", "utf8"])
def test_limit_evidence_preserves_maximum_native_session_id(tmp_path, session_id):
    async def run():
        store = SQLiteSessionStore(tmp_path / "long-id.db")
        app = CayuApp(enable_logging=False, session_store=store)
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.completed(
                        {
                            "usage": {
                                "input_tokens": 1100,
                                "output_tokens": 10,
                                "total_tokens": 1110,
                            }
                        }
                    )
                ]
            ]
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="limited", model="test"))
        ctx = Workflow(app).context("long-id-root")
        await ctx.start()
        result = await parallel(
            [
                step(
                    ctx,
                    agent="limited",
                    step_id="limited",
                    session_id=session_id,
                    prompt="check",
                    run_options=StepRunOptions(limits=RunLimits(max_total_tokens=1000)),
                )
            ]
        )
        evidence = result.failures[0].evidence
        assert evidence.classification == "interruption"
        assert evidence.session_id == session_id
        terminal = next(
            e for e in await store.load_events(session_id) if e.id == evidence.terminal_event_id
        )
        assert terminal.type == "session.interrupted"
        assert evidence.run_epoch == terminal.payload["failure_evidence"]["run_epoch"]
        assert type(evidence).model_validate_json(evidence.model_dump_json()) == evidence
        await store.close()
        store = SQLiteSessionStore(tmp_path / "long-id.db")
        app = CayuApp(enable_logging=False, session_store=store)
        app.register_provider(ScriptedModelProvider([]), default=True)
        app.register_agent(AgentSpec(name="limited", model="test"))
        ctx = Workflow(app).context("long-id-root")
        await ctx.start()
        with pytest.raises(StepError) as raised:
            await step(
                ctx, agent="limited", step_id="limited", session_id=session_id, prompt="check"
            )
        assert raised.value.evidence == evidence
        await store.close()

    asyncio.run(run())
