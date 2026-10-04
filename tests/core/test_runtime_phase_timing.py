from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pytest
from tests.core.test_runtime import VersionedFakeProvider
from tests.core.test_tool_round_publication_failure_matrix import store_factory as store_factory

from cayu import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    AgentSpec,
    CayuApp,
    EventType,
    LoggingEventSink,
    Message,
    ModelStepPreparationTiming,
    ModelStreamEvent,
    PostgresSessionStore,
    PostgresTaskStore,
    RunRequest,
    RuntimeTimingConfig,
    SecretRedactor,
    SQLiteSessionStore,
    SQLiteTaskStore,
    StructuredOutputSpec,
    SubagentSpec,
    SubagentTool,
    TaskCreate,
    Tool,
    ToolResult,
    ToolRoundTiming,
    ToolSpec,
)
from cayu.configuration import CayuConfig, ToolExecutionConfig
from cayu.observability.events import InMemoryEventSink
from cayu.runtime._phase_timing import (
    RuntimeTimingRecorder,
    current_builder,
    current_store_counters,
)
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions.base import IncompleteSessionRecoveryRequest
from cayu.storage import _sqlite_support
from cayu.storage.migrations import SchemaMode
from cayu.tools.base import ToolEffect
from cayu.tools.policy import ToolPolicy, ToolPolicyDecision, ToolPolicyResult

_CONTENT = "private-prompt-arguments-and-result"
_PHASES = {
    "authorization",
    "admission",
    "started_persistence",
    "effect_state",
    "execution",
    "result_processing",
    "staging",
    "sibling_wait",
    "publication_queue_wait",
    "publication",
    "round_commit",
    "unattributed",
}
_WAITS = {"sibling_wait", "publication_queue_wait"}


class _TimedTool(Tool):
    spec = ToolSpec(
        name="timed_tool",
        description="Run measured application work.",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
    )

    async def run(self, ctx, args):
        await asyncio.sleep(0.02)
        return ToolResult(content=args["text"])


def _app(
    store,
    *,
    config=None,
    sinks=(),
    timing_sinks=(),
    calls=1,
    tool=None,
    parallel=1,
    policy=None,
    enable_logging=False,
):
    app = CayuApp(
        session_store=store,
        enable_logging=enable_logging,
        event_sinks=sinks,
        runtime_timing=config,
        timing_sinks=timing_sinks,
        config=CayuConfig(tool_execution=ToolExecutionConfig(max_parallel_tool_calls=parallel)),
    )
    app.register_provider(
        VersionedFakeProvider(
            [
                [
                    *[
                        ModelStreamEvent.tool_call(
                            name="timed_tool", arguments={"text": _CONTENT}, id=f"call-{i}"
                        )
                        for i in range(calls)
                    ],
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
            ]
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool or _TimedTool()],
        tool_policy=policy,
    )
    return app


async def _run(app, sid="timing"):
    events = [
        e
        async for e in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=sid,
                messages=[Message.text("user", _CONTENT)],
            )
        )
    ]
    assert events[-1].type == EventType.SESSION_COMPLETED, [
        (e.type, e.payload) for e in events[-2:]
    ]
    await app.flush_runtime_timing()
    return events


@asynccontextmanager
async def _store(backend, request, sqlite_resources):
    async with sqlite_resources as resources:
        if backend == "sqlite":
            yield resources.own(SQLiteSessionStore(resources.path("timing.sqlite")))
        else:
            store = PostgresSessionStore(
                request.getfixturevalue("postgres_dsn"),
                min_size=1,
                max_size=2,
                schema_mode=SchemaMode.CREATE,
            )
            try:
                yield store
            finally:
                await store.close()


def _assert_phases_cover_round(record, tolerance=0.15):
    """Exclusive active phases add up to the round's observed wall time.

    Waits overlap other calls' active phases and are excluded. The remainder is
    consumer time between yielded events, which tests consume immediately.
    """
    active = sum(p.duration_seconds for p in record.phases if p.name not in _WAITS)
    assert active <= record.duration_seconds + 0.005, (active, record)
    assert active >= record.duration_seconds * (1 - tolerance) - 0.01, (active, record)
    for phase in record.phases:
        assert phase.duration_seconds >= 0
        assert phase.store_commit_seconds >= 0 and phase.store_lock_wait_seconds >= 0
        if phase.duration_seconds:
            assert phase.first_started_at <= phase.last_completed_at
            assert record.started_at <= phase.first_started_at
            assert phase.last_completed_at <= record.completed_at


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_runtime_round_and_call_costs_and_preparation_are_content_free(
    backend,
    request,
    sqlite_resources,
):
    async def scenario():
        async with _store(backend, request, sqlite_resources) as store:
            sink = InMemoryEventSink()
            app = _app(store, sinks=[sink], calls=2)
            await _run(app)
            records = await app.inspect_recent_tool_round_timing("timing")
            assert len(records) == 1
            record = records[0]
            assert not record.incomplete and len(record.calls) == 2
            assert set(p.name for p in record.phases) == _PHASES
            phases = {p.name: p for p in record.phases}
            for name in (
                "admission",
                "started_persistence",
                "staging",
                "publication",
                "round_commit",
            ):
                assert phases[name].store_transaction_count > 0, (name, phases)
            assert phases["staging"].store_bytes_written > 0
            assert phases["execution"].duration_seconds >= 0.035
            assert phases["execution"].store_transaction_count == 0
            assert phases["result_processing"].store_transaction_count == 0
            for name in ("staging", "publication", "round_commit"):
                assert phases[name].store_commit_seconds > 0, (name, phases[name])
            assert sum(p.store_lock_wait_seconds for p in record.phases) > 0
            _assert_phases_cover_round(record)
            # Serial dispatch: the first call waits for its sibling to stage;
            # the last call has no sibling left and waits only for publication.
            first, last = ({p.name: p for p in call.phases} for call in record.calls)
            assert first["sibling_wait"].duration_seconds >= 0.015
            assert last["sibling_wait"].duration_seconds == 0
            assert last["publication_queue_wait"].duration_seconds > 0
            for call in record.calls:
                assert call.session_id == record.session_id
                assert call.tool_round_id == record.tool_round_id
                assert set(p.name for p in call.phases) == _PHASES
                assert call.tool_effect_completed_at <= call.tool_terminal_staged_at
                assert call.tool_terminal_staged_at <= call.tool_terminal_publication_started_at
            preparation = await app.inspect_recent_model_step_preparation_timing("timing")
            following = next(
                p for p in preparation if p.after_tool_round_id == record.tool_round_id
            )
            assert not following.incomplete and following.started_at <= following.completed_at
            assert {p.name for p in following.phases} == {
                "handoff",
                "preparation",
                "context_policy",
                "recall",
                "counting",
            }
            assert all(_CONTENT not in p.model_dump_json() for p in sink.timings)
            assert record in sink.timings

    asyncio.run(scenario())


class _DelayedConnection:
    def __init__(self, connection, delay, writes):
        object.__setattr__(self, "raw", connection)
        object.__setattr__(self, "delay", delay)
        connection.set_trace_callback(
            lambda sql: (
                writes.append(sql.split()[0].upper())
                if sql.split()
                and sql.split()[0].upper() in {"INSERT", "UPDATE", "DELETE", "REPLACE", "COMMIT"}
                else None
            )
        )

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def __setattr__(self, name, value):
        setattr(self.raw, name, value)

    def _delay(self):
        if self.raw.in_transaction and current_store_counters() is not None:
            time.sleep(self.delay)

    def commit(self):
        self._delay()
        return self.raw.commit()

    def __enter__(self):
        self.raw.__enter__()
        return self

    def __exit__(self, *args):
        if args[0] is None:
            self._delay()
        return self.raw.__exit__(*args)


def test_fixed_commit_delay_is_storage_time_not_tool_execution(monkeypatch, sqlite_resources):
    connect = _sqlite_support.sqlite3.connect
    monkeypatch.setattr(
        _sqlite_support.sqlite3,
        "connect",
        lambda *a, **kw: _DelayedConnection(connect(*a, **kw), 0.01, []),
    )

    async def scenario():
        async with sqlite_resources as resources:
            store = resources.own(SQLiteSessionStore(resources.path("delayed.sqlite")))
            app = _app(store)
            await _run(app)
            (record,) = await app.inspect_recent_tool_round_timing("timing")
            phases = {p.name: p for p in record.phases}
            for name in ("staging", "publication", "round_commit"):
                assert phases[name].store_commit_seconds >= 0.009, (name, phases)
                assert phases[name].duration_seconds >= phases[name].store_commit_seconds
            assert phases["execution"].store_commit_seconds == 0
            assert phases["execution"].duration_seconds < 0.08

    asyncio.run(scenario())


def test_timing_adds_no_durable_writes_and_disabled_path_has_no_records(
    monkeypatch, sqlite_resources
):
    connect = _sqlite_support.sqlite3.connect
    writes = []
    monkeypatch.setattr(
        _sqlite_support.sqlite3,
        "connect",
        lambda *a, **kw: _DelayedConnection(connect(*a, **kw), 0, writes),
    )

    async def scenario():
        async with sqlite_resources as resources:
            observed = []
            for enabled in (False, True):
                store = resources.own(SQLiteSessionStore(resources.path(f"{enabled}.sqlite")))
                app = _app(store, config=RuntimeTimingConfig(enabled=enabled))
                writes.clear()
                events = await _run(app)
                observed.append((list(writes), [e.type for e in events]))
                assert bool(await app.inspect_recent_tool_round_timing("timing")) is enabled
            assert observed[0] == observed[1]

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["exception", "cancel", "timeout"])
def test_timing_sink_failure_cannot_change_work_or_inherit_runtime_context(
    failure, sqlite_resources
):
    class Sink:
        async def emit_timing(self, record):
            assert current_builder() is None and current_store_counters() is None
            if failure == "cancel":
                raise asyncio.CancelledError
            if failure == "timeout":
                await asyncio.sleep(10)
            raise RuntimeError(_CONTENT)

    async def scenario():
        async with sqlite_resources as resources:
            store = resources.own(SQLiteSessionStore(resources.path("sink.sqlite")))
            app = _app(
                store, timing_sinks=[Sink()], config=RuntimeTimingConfig(sink_timeout_seconds=0.01)
            )
            await asyncio.wait_for(_run(app), timeout=10)
            assert app.runtime_timing_status().failed_deliveries >= 1
            assert len(await app.inspect_recent_tool_round_timing("timing")) == 1

    asyncio.run(scenario())


def test_recent_and_call_buffers_are_bounded_and_identifiers_redacted(sqlite_resources):
    async def scenario():
        async with sqlite_resources as resources:
            store = resources.own(SQLiteSessionStore(resources.path("bounded.sqlite")))
            app = _app(
                store, config=RuntimeTimingConfig(recent_capacity=2, max_calls_per_round=1), calls=2
            )
            app._event_writer.timing.redactor = SecretRedactor([_CONTENT])
            await _run(app, sid=_CONTENT)
            (record,) = await app.inspect_recent_tool_round_timing(_CONTENT)
            assert len(record.calls) == 1 and record.calls_truncated == 1
            assert _CONTENT not in record.model_dump_json()
            assert await app.inspect_recent_tool_round_timing(record.session_id) == (record,)
            assert app.runtime_timing_status().recent_records == 2
            assert await app.inspect_recent_tool_round_timing("different") == ()
            with pytest.raises(ValueError):
                await app.inspect_recent_tool_round_timing(_CONTENT, limit=0)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_native_task_mutations_inside_tool_are_charged_to_execution(
    backend,
    request,
    sqlite_resources,
):
    async def scenario():
        async with _store(backend, request, sqlite_resources) as store:
            tasks = (
                sqlite_resources.own(SQLiteTaskStore(":memory:"))
                if backend == "sqlite"
                else PostgresTaskStore(
                    request.getfixturevalue("postgres_dsn"),
                    min_size=1,
                    max_size=2,
                    schema_mode=SchemaMode.CREATE,
                )
            )

            class StoreTool(_TimedTool):
                async def run(self, ctx, args):
                    await tasks.create_task(TaskCreate(task_id="timed-task", type="job"))
                    return ToolResult(content="created")

            try:
                app = _app(store, tool=StoreTool())
                await _run(app, sid="native-timing")
                (record,) = await app.inspect_recent_tool_round_timing("native-timing")
                execution = next(p for p in record.calls[0].phases if p.name == "execution")
                assert execution.store_transaction_count >= 1
                assert execution.store_commit_seconds > 0
                assert execution.store_bytes_written > 0
            finally:
                if backend == "postgres":
                    await tasks.close()

    asyncio.run(scenario())


def test_slow_sink_queue_drops_observations_without_waiting_for_sink(sqlite_resources):
    class Sink:
        def __init__(self):
            self.release = asyncio.Event()

        async def emit_timing(self, record):
            await self.release.wait()

    async def scenario():
        async with sqlite_resources as resources:
            sink = Sink()
            store = resources.own(SQLiteSessionStore(resources.path("queue.sqlite")))
            app = _app(
                store,
                timing_sinks=[sink],
                config=RuntimeTimingConfig(sink_queue_capacity=1, sink_timeout_seconds=5),
            )
            events = [
                e
                async for e in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="timing",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            assert events[-1].type == EventType.SESSION_COMPLETED
            assert app.runtime_timing_status().dropped_records >= 1
            assert len(await app.inspect_recent_tool_round_timing("timing")) == 1
            sink.release.set()
            await asyncio.wait_for(app.flush_runtime_timing(), timeout=2)
            assert app.runtime_timing_status().queued_records == 0

    asyncio.run(scenario())


def test_single_call_round_has_no_sibling_wait(sqlite_resources):
    async def scenario():
        async with sqlite_resources as resources:
            store = resources.own(SQLiteSessionStore(resources.path("single.sqlite")))
            app = _app(store)
            await _run(app)
            (record,) = await app.inspect_recent_tool_round_timing("timing")
            (call,) = record.calls
            phases = {p.name: p for p in call.phases}
            assert phases["sibling_wait"].duration_seconds == 0
            assert phases["publication_queue_wait"].duration_seconds > 0
            assert phases["publication_queue_wait"].store_transaction_count == 0
            _assert_phases_cover_round(record)

    asyncio.run(scenario())


def test_parallel_dispatch_overlaps_call_execution(sqlite_resources):
    class SlowTool(_TimedTool):
        async def run(self, ctx, args):
            await asyncio.sleep(0.1)
            return ToolResult(content="done")

    async def scenario():
        async with sqlite_resources as resources:
            store = resources.own(SQLiteSessionStore(resources.path("parallel.sqlite")))
            app = _app(store, calls=3, parallel=3, tool=SlowTool())
            await _run(app)
            (record,) = await app.inspect_recent_tool_round_timing("timing")
            assert not record.incomplete and len(record.calls) == 3
            executions = [next(p for p in c.phases if p.name == "execution") for c in record.calls]
            assert all(p.duration_seconds >= 0.09 for p in executions)
            # Execution windows overlap, so per-call durations are not additive.
            assert max(p.first_started_at for p in executions) < min(
                p.last_completed_at for p in executions
            )
            total = sum(p.duration_seconds for p in executions)
            window = max(p.last_completed_at for p in executions) - min(
                p.first_started_at for p in executions
            )
            assert total > 0.27 and window.total_seconds() < total - 0.1
            for call in record.calls:
                phases = {p.name: p for p in call.phases}
                assert phases["staging"].store_transaction_count > 0
                assert phases["publication"].store_transaction_count > 0

    asyncio.run(scenario())


def test_failed_and_blocked_calls_are_recorded_with_their_terminal_costs(sqlite_resources):
    class Policy(ToolPolicy):
        async def authorize(self, request):
            if request.tool_call_id == "call-2":
                return ToolPolicyResult(decision=ToolPolicyDecision.DENY, reason="Denied.")
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)

    class FailingTool(_TimedTool):
        spec = _TimedTool.spec.model_copy(update={"effect": ToolEffect.IDEMPOTENT})

        calls = 0

        async def run(self, ctx, args):
            await asyncio.sleep(0.02)
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError(_CONTENT)
            return ToolResult(content="done")

    async def scenario():
        async with sqlite_resources as resources:
            store = resources.own(SQLiteSessionStore(resources.path("outcomes.sqlite")))
            app = _app(store, calls=3, tool=FailingTool(), policy=Policy())
            events = await _run(app)
            terminals = [
                e.type
                for e in events
                if e.type
                in {
                    EventType.TOOL_CALL_COMPLETED,
                    EventType.TOOL_CALL_FAILED,
                    EventType.TOOL_CALL_BLOCKED,
                }
            ]
            assert terminals == [
                EventType.TOOL_CALL_FAILED,
                EventType.TOOL_CALL_COMPLETED,
                EventType.TOOL_CALL_BLOCKED,
            ]
            (record,) = await app.inspect_recent_tool_round_timing("timing")
            assert not record.incomplete
            assert _CONTENT not in record.model_dump_json()
            calls = {c.tool_call_id: {p.name: p for p in c.phases} for c in record.calls}
            assert calls["call-0"]["execution"].duration_seconds >= 0.015
            assert calls["call-2"]["execution"].duration_seconds == 0
            for phases in calls.values():
                assert phases["publication"].store_transaction_count > 0
            _assert_phases_cover_round(record)

    asyncio.run(scenario())


def test_cancelled_round_is_recorded_incomplete(sqlite_resources):
    started = asyncio.Event()

    class BlockingTool(_TimedTool):
        async def run(self, ctx, args):
            started.set()
            await asyncio.sleep(30)
            return ToolResult(content="never")

    async def scenario():
        async with sqlite_resources as resources:
            store = resources.own(SQLiteSessionStore(resources.path("cancel.sqlite")))
            app = _app(store, tool=BlockingTool())

            async def consume():
                async for _ in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="timing",
                        messages=[Message.text("user", "go")],
                    )
                ):
                    pass

            task = asyncio.create_task(consume())
            await asyncio.wait_for(started.wait(), timeout=10)
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            (record,) = await app.inspect_recent_tool_round_timing("timing")
            assert record.incomplete and not record.recovered
            (call,) = record.calls
            execution = next(p for p in call.phases if p.name == "execution")
            assert execution.duration_seconds >= 0.04
            assert app.runtime_timing_status().failed_deliveries == 0

    asyncio.run(scenario())


def test_structured_output_repair_does_not_reuse_an_earlier_commit(sqlite_resources):
    async def scenario():
        async with sqlite_resources as resources:
            store = resources.own(SQLiteSessionStore(resources.path("repair.sqlite")))
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(
                VersionedFakeProvider(
                    [
                        [
                            ModelStreamEvent.tool_call(
                                name="timed_tool", arguments={"text": "x"}, id="call-0"
                            ),
                            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                        ],
                        [
                            ModelStreamEvent.text_delta("not structured"),
                            ModelStreamEvent.completed({"finish_reason": "stop"}),
                        ],
                        [
                            ModelStreamEvent.tool_call(
                                id="call-valid",
                                name=STRUCTURED_OUTPUT_TOOL_NAME,
                                arguments={"output": {"answer": "fixed"}},
                            ),
                            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                        ],
                    ]
                ),
                default=True,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"), tools=[_TimedTool()]
            )
            events = [
                e
                async for e in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="repair",
                        messages=[Message.text("user", "go")],
                        structured_output=StructuredOutputSpec(
                            json_schema={
                                "type": "object",
                                "properties": {"answer": {"type": "string"}},
                                "required": ["answer"],
                                "additionalProperties": False,
                            },
                            max_retries=1,
                        ),
                    )
                )
            ]
            assert events[-1].type == EventType.SESSION_COMPLETED
            first, after_round, repair = reversed(
                await app.inspect_recent_model_step_preparation_timing("repair")
            )
            (record, _) = reversed(await app.inspect_recent_tool_round_timing("repair"))
            assert first.after_tool_round_id is None
            assert after_round.after_tool_round_id == record.tool_round_id
            # The repair step follows a model step, not the earlier round commit.
            assert repair.after_tool_round_id is None
            assert repair.started_at >= after_round.completed_at
            handoff = next(p for p in repair.phases if p.name == "handoff")
            assert handoff.duration_seconds == 0

    asyncio.run(scenario())


def test_subagent_child_time_stays_in_the_parent_call_execution(sqlite_resources):
    class ChildTool(_TimedTool):
        async def run(self, ctx, args):
            await asyncio.sleep(0.1)
            return ToolResult(content="child")

    async def scenario():
        async with sqlite_resources as resources:
            store = resources.own(SQLiteSessionStore(resources.path("subagent.sqlite")))
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(
                VersionedFakeProvider(
                    [
                        [
                            ModelStreamEvent.tool_call(
                                id="call-sub",
                                name="subagent",
                                arguments={"agent": "reviewer", "task": "review"},
                            ),
                            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                        ],
                        [
                            ModelStreamEvent.tool_call(
                                id="call-child", name="timed_tool", arguments={"text": "x"}
                            ),
                            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                        ],
                        [ModelStreamEvent.completed({"finish_reason": "stop"})],
                        [ModelStreamEvent.completed({"finish_reason": "stop"})],
                    ]
                ),
                default=True,
            )
            app.register_agent(
                AgentSpec(name="parent", model="fake-model"),
                tools=[
                    SubagentTool(
                        app,
                        agents={
                            "reviewer": SubagentSpec(agent_name="reviewer", description="Review.")
                        },
                    )
                ],
            )
            app.register_agent(AgentSpec(name="reviewer", model="fake-model"), tools=[ChildTool()])
            events = [
                e
                async for e in app.run(
                    RunRequest(
                        agent_name="parent",
                        session_id="parent",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            assert events[-1].type == EventType.SESSION_COMPLETED
            (parent,) = await app.inspect_recent_tool_round_timing("parent")
            child = next(
                record
                for _, record in app._event_writer.timing.recent
                if isinstance(record, ToolRoundTiming) and record.session_id != "parent"
            )
            assert not child.incomplete
            assert [c.tool_call_id for c in child.calls] == ["call-child"]
            (call,) = parent.calls
            execution = next(p for p in call.phases if p.name == "execution")
            # The child's own round phases are reported in its own record and
            # are not subtracted from the parent's tool execution.
            assert execution.duration_seconds >= child.duration_seconds
            _assert_phases_cover_round(parent)

    asyncio.run(scenario())


def test_approval_paused_round_and_continuation_are_recorded(sqlite_resources):
    from tests.core.test_tool_round_continuation_backends import _drain, _pause, _resolve, _runtime

    async def scenario():
        async with sqlite_resources as resources:
            store = resources.own(SQLiteSessionStore(resources.path("approval.sqlite")))
            app, _, tool = _runtime(store, "approve")
            request = await _pause(app, "approve", "approval-timing")
            (paused,) = await app.inspect_recent_tool_round_timing("approval-timing")
            assert paused.incomplete and not paused.recovered
            assert {c.tool_call_id for c in paused.calls} == {"pause", "allowed", "denied"}
            # The pause write is the paused attempt's round commit, charged to
            # the round rather than to the call awaiting approval.
            paused_phases = {p.name: p for p in paused.phases}
            pause_write, rest = paused_phases["round_commit"], paused_phases["unattributed"]
            assert pause_write.store_transaction_count > 0
            assert pause_write.store_commit_seconds > 0
            assert rest.store_transaction_count < pause_write.store_transaction_count
            assert rest.duration_seconds < pause_write.duration_seconds
            assert rest.duration_seconds < paused.duration_seconds * 0.25
            for call in paused.calls:
                assert (
                    next(p for p in call.phases if p.name == "round_commit").duration_seconds == 0
                )
            _assert_phases_cover_round(paused)
            events = await _drain(_resolve(app, request))
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert len(tool.calls) == 2
            records = await app.inspect_recent_tool_round_timing("approval-timing")
            continued = next(r for r in records if r is not paused)
            assert continued.tool_round_id == paused.tool_round_id
            assert not continued.incomplete and not continued.recovered
            phases = {p.name: p for p in continued.phases}
            for name in ("admission", "staging", "publication", "round_commit"):
                assert phases[name].store_transaction_count > 0, (name, phases[name])
            assert phases["execution"].duration_seconds > 0
            calls = {c.tool_call_id: c for c in continued.calls}
            for call_id in ("pause", "allowed"):
                call = calls[call_id]
                assert call.tool_terminal_staged_at is not None
                assert call.tool_terminal_publication_started_at is not None
                assert call.tool_terminal_staged_at <= call.tool_terminal_publication_started_at
            preparation = await app.inspect_recent_model_step_preparation_timing("approval-timing")
            assert any(p.after_tool_round_id == continued.tool_round_id for p in preparation)

    asyncio.run(scenario())


@pytest.mark.parametrize("store_factory", ["sqlite", "postgres"], indirect=True)
def test_recovery_published_round_is_recorded_as_recovered(store_factory):
    from tests.core.test_tool_round_publication_failure_matrix import _TwoCallProvider
    from tests.core.test_tool_round_recovery_backends import (
        NativeRecoveryTool,
        NeverRunTool,
        RecoveryStore,
        seed_round,
    )

    async def scenario():
        async with store_factory(RecoveryStore) as store:
            provider = _TwoCallProvider([])
            tools = [NeverRunTool("known"), NativeRecoveryTool("confirmed"), NeverRunTool("other")]
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=tools)
            session_id, pending = await seed_round(store, app, provider, tools, started=True)
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id, inactive_for_seconds=0)
            )
            (record,) = await app.inspect_recent_tool_round_timing(session_id)
            assert record.recovered and not record.incomplete
            assert record.tool_round_id == pending.tool_round_id
            assert [c.tool_call_id for c in record.calls] == [
                "call-known",
                "call-native",
                "call-other",
            ]
            phases = {p.name: p for p in record.phases}
            assert phases["round_commit"].store_transaction_count > 0
            assert phases["round_commit"].store_commit_seconds > 0
            assert phases["publication"].store_transaction_count > 0
            native = {p.name: p for p in record.calls[1].phases}
            assert native["effect_state"].duration_seconds > 0
            assert native["execution"].duration_seconds == 0

    asyncio.run(scenario())


def test_default_logging_sink_does_not_start_timing_delivery(sqlite_resources):
    async def scenario():
        async with sqlite_resources as resources:
            store = resources.own(SQLiteSessionStore(resources.path("logging.sqlite")))
            app = _app(store, enable_logging=True)
            assert app._event_writer.timing.sinks == ()
            await _run(app)
            assert app._event_writer.timing.worker is None
            assert app.runtime_timing_status().queued_records == 0
            await app.close_runtime_timing()
            opted_in = LoggingEventSink(log_runtime_timing=True)
            store_b = resources.own(SQLiteSessionStore(resources.path("logging-b.sqlite")))
            assert _app(store_b, sinks=[opted_in])._event_writer.timing.sinks == (opted_in,)

    asyncio.run(scenario())


def test_recorder_rebinds_across_event_loops_and_closes_its_worker():
    released = []

    class Sink:
        async def emit_timing(self, record):
            if not released:
                await asyncio.sleep(30)
            released.append(record)

    recorder = RuntimeTimingRecorder(
        config=RuntimeTimingConfig(sink_timeout_seconds=0.2),
        sinks=[Sink()],
        redactor=SecretRedactor(),
        codec=None,
    )
    now = datetime.now(UTC)
    record = ModelStepPreparationTiming(
        session_id="loop",
        model_step_id="step",
        after_tool_round_id=None,
        started_at=now,
        completed_at=now,
        duration_seconds=0,
        incomplete=False,
        phases=(),
    )

    async def first_loop():
        recorder.publish("loop", (), record)
        # A timed-out flush binds the queue's waiters to this loop.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(recorder.flush(), 0.05)

    async def second_loop():
        released.append(None)
        recorder.publish("loop", (), record)
        await asyncio.wait_for(recorder.flush(), 2)
        assert released[-1] is record
        released.clear()
        recorder.publish("loop", (), record)
        worker = recorder.worker
        await asyncio.wait_for(recorder.aclose(), 2)
        assert worker.done() and recorder.worker is None
        assert recorder.status().queued_records == 0

    asyncio.run(first_loop())
    asyncio.run(second_loop())
    assert recorder.dropped + recorder.failed >= 1


def test_interrupted_structured_output_close_is_not_recorded_as_recovered():
    from tests.core.test_structured_output_tool_round_recovery import (
        _answer_spec,
        _publish_structured_model_step,
        _RecordingProvider,
        _register_runtime,
        _structured_call,
    )

    from cayu.runtime import _tool_round_recovery as tool_round_recovery
    from cayu.runtime._durable_tool_round import InterruptedToolRoundRequest
    from cayu.sessions.base import InMemorySessionStore, _activate_session_interaction

    async def scenario():
        store = InMemorySessionStore()
        provider = _RecordingProvider()
        staged = await _publish_structured_model_step(
            store,
            session_id="structured-interrupted",
            provider=provider,
            spec=_answer_spec(),
            tool_calls=[_structured_call(call_id="call-final", output={"answer": "done"})],
        )
        app = _register_runtime(store, provider)
        session = await store.load(staged.session.id)
        pending_round = staged.pending_round
        _activate_session_interaction(session.id, f"interaction-{session.id}")
        # Interruption before the live runner closes the round through recovery.
        events = [
            event
            async for event in app._recovery_coordinator.close_interrupted_tool_round(
                InterruptedToolRoundRequest(
                    session=session,
                    registered_agent=app._get_registered_agent("assistant"),
                    registered_environment=None,
                    messages=await store.load_transcript(session.id),
                    tool_calls=tool_round_recovery.pending_round_tool_calls(pending_round),
                    tool_outcomes=[],
                    tool_round_identity=pending_rounds.pending_tool_round_identity(pending_round),
                    cancellation_artifacts=None,
                    cancellation_artifacts_by_id=None,
                )
            )
        ]
        assert events
        (record,) = await app.inspect_recent_tool_round_timing(session.id)
        assert record.tool_round_id == pending_round.tool_round_id
        # Like an ordinary interrupted close, this commits the round without
        # being publication after process loss.
        assert not record.recovered and not record.incomplete
        phases = {p.name: p for p in record.phases}
        assert phases["round_commit"].duration_seconds > 0

    asyncio.run(scenario())
