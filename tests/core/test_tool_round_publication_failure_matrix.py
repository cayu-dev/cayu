from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager
from types import SimpleNamespace
from typing import Literal

import pytest
from tests.core._execution_profile_fixtures import versioned_test_provider_identity

from cayu import CayuConfig, ToolExecutionConfig
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.run_limits import RunLimits
from cayu.events import Event, EventType
from cayu.messages import Message, ToolResultPart
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import _run_limits as run_limits
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions.base import (
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    InterruptSessionRequest,
    RunRequest,
    RuntimePublicationRequest,
    RuntimePublicationResult,
    SessionStore,
)
from cayu.sessions.records import SessionStatus
from cayu.storage.migrations import SchemaMode
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec

_WATCHDOG_SECONDS = 20  # Includes durable-backend startup; faults use explicit barriers.


class _TwoCallProvider(ModelProvider):
    name = "tool-round-publication-matrix"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return versioned_test_provider_identity(self)

    def __init__(self, responses: list[list[ModelStreamEvent]]) -> None:
        self._responses = responses
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        response_index = len(self.requests)
        self.requests.append(request)
        if response_index >= len(self._responses):
            raise AssertionError("Recovery unexpectedly redispatched the model provider.")
        for event in self._responses[response_index]:
            yield event


class _SideEffectTool(Tool):
    spec = ToolSpec(
        name="side_effect",
        description="Record one externally visible call.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:tool-round-publication:side-effect-tool",
            behavior_version="1",
            implementation_version="1",
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx
        await asyncio.sleep(0)
        self.calls.append(args["value"])
        return ToolResult(content=f"executed {args['value']}")


class _PublicationBarrierStore:
    invocation_lifecycle_command_version = 1

    def __init__(self, *args, boundary: Literal["before-commit", "after-commit"], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.boundary = boundary
        self.boundary_reached = asyncio.Event()
        self.release_publication = asyncio.Event()
        self.blocked_once = False

    async def publish_runtime_publication(
        self,
        session_id: str,
        *,
        request: RuntimePublicationRequest,
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> RuntimePublicationResult:
        should_block = request.kind == "tool-round" and not self.blocked_once
        if should_block:
            self.blocked_once = True
            if self.boundary == "before-commit":
                self.boundary_reached.set()
                await self.release_publication.wait()
        result = await super().publish_runtime_publication(
            session_id,
            request=request,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )
        if should_block and self.boundary == "after-commit":
            self.boundary_reached.set()
            await self.release_publication.wait()
        return result


class _SimulatedProcessLoss(BaseException):
    pass


class _ProcessLossStore:
    invocation_lifecycle_command_version = 1

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fail_before_tool_publication = True
        self.tool_publication_attempted = asyncio.Event()

    async def publish_runtime_publication(
        self,
        session_id: str,
        *,
        request: RuntimePublicationRequest,
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> RuntimePublicationResult:
        if request.kind == "tool-round" and self.fail_before_tool_publication:
            self.tool_publication_attempted.set()
            raise _SimulatedProcessLoss("process exited before tool-round publication")
        return await super().publish_runtime_publication(
            session_id,
            request=request,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )


class _ConcurrentProcessLossStore(_ProcessLossStore):
    invocation_lifecycle_command_version = 1

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block_recovery_claim = False
        self.claim_fenced = asyncio.Event()
        self.release_claim = asyncio.Event()
        self.blocked_claim_once = False

    async def fence_run_and_transform_checkpoint(self, *args, **kwargs):
        fenced = await super().fence_run_and_transform_checkpoint(*args, **kwargs)
        if self.block_recovery_claim and not self.blocked_claim_once:
            self.blocked_claim_once = True
            self.claim_fenced.set()
            await self.release_claim.wait()
        return fenced


class _LostAcknowledgementStore(_ProcessLossStore):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.publication_requests: list[RuntimePublicationRequest] = []
        self.publication_results: list[RuntimePublicationResult] = []

    async def publish_runtime_publication(self, session_id, *, request, **kwargs):
        result = await super().publish_runtime_publication(session_id, request=request, **kwargs)
        if request.kind == "tool-round":
            self.publication_requests.append(request.model_copy(deep=True))
            self.publication_results.append(result)
            if len(self.publication_results) == 1:
                assert result.replayed is False
                raise ConnectionError("tool-round publication acknowledgement lost")
        return result


@pytest.fixture(params=["memory", "sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def store_factory(request, tmp_path):
    """Keep the fault schedule above the real backend's transaction boundary."""
    backend = request.param
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None
    if backend == "postgres":
        from cayu.storage.postgres import PostgresSessionStore

        backend_type = PostgresSessionStore
    else:
        backend_type = InMemorySessionStore if backend == "memory" else SQLiteSessionStore

    @asynccontextmanager
    async def open_store(fault_type, **fault_options):
        class FaultStore(fault_type, backend_type):
            invocation_lifecycle_command_version = 1
            session_access_version = 1

        if backend == "memory":
            store = FaultStore(**fault_options)
        elif backend == "sqlite":
            store = FaultStore(tmp_path / "round-publication.sqlite", **fault_options)
        else:
            store = FaultStore(
                dsn, schema_mode=SchemaMode.CREATE, min_size=1, max_size=2, **fault_options
            )
        try:
            yield store
        finally:
            if backend != "memory":
                await store.close()

    return open_store


def _tool_call_response() -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(
            id="call-side-effect-a",
            name="side_effect",
            arguments={"value": "first"},
        ),
        ModelStreamEvent.tool_call(
            id="call-side-effect-b",
            name="side_effect",
            arguments={"value": "second"},
        ),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def _runtime(
    store: SessionStore,
    provider: _TwoCallProvider,
    tool: _SideEffectTool,
    *,
    max_parallel_tool_calls: int,
) -> CayuApp:
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        config=CayuConfig(
            tool_execution=ToolExecutionConfig(max_parallel_tool_calls=max_parallel_tool_calls)
        ),
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )
    return app


async def _collect_run(app: CayuApp, *, session_id: str) -> list[Event]:
    return [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "execute both calls once")],
            )
        )
    ]


async def _assert_published_round(
    store: SessionStore,
    *,
    session_id: str,
) -> None:
    transcript = await store.load_transcript(session_id)
    checkpoint = await store.load_checkpoint(session_id)
    events = await store.load_events(session_id)
    terminal_events = [
        event
        for event in events
        if event.type == EventType.TOOL_CALL_COMPLETED
        and event.payload.get("tool_call_id") in {"call-side-effect-a", "call-side-effect-b"}
    ]
    assert len(terminal_events) == 2
    assert len({event.payload["tool_round_id"] for event in terminal_events}) == 1
    round_id = terminal_events[0].payload["tool_round_id"]
    receipt = await store.load_runtime_publication_receipt(
        session_id,
        f"tool-round:{round_id}",
    )

    tool_messages = [message for message in transcript if message.role.value == "tool"]
    assert len(tool_messages) == 1
    assert [
        part.tool_call_id for part in tool_messages[0].content if isinstance(part, ToolResultPart)
    ] == ["call-side-effect-a", "call-side-effect-b"]
    assert pending_round_reader.pending_tool_round_from_checkpoint(checkpoint) is None
    assert receipt is not None
    assert len({event.id for event in events}) == len(events)


@pytest.mark.parametrize(
    "max_parallel_tool_calls",
    [
        pytest.param(1, id="serial"),
        pytest.param(2, id="parallel"),
    ],
)
@pytest.mark.parametrize(
    "boundary",
    [
        pytest.param("after-commit", id="commit-then-cancel"),
        pytest.param("before-commit", id="cancel-before-commit"),
    ],
)
def test_two_call_round_survives_cancellation_at_publication_boundary(
    max_parallel_tool_calls: int,
    boundary: Literal["before-commit", "after-commit"],
    store_factory,
) -> None:
    async def scenario() -> None:
        async with store_factory(_PublicationBarrierStore, boundary=boundary) as store:
            provider = _TwoCallProvider([_tool_call_response()])
            tool = _SideEffectTool()
            session_id = f"tool-round-{boundary}-{max_parallel_tool_calls}"
            app = _runtime(
                store,
                provider,
                tool,
                max_parallel_tool_calls=max_parallel_tool_calls,
            )

            running = asyncio.create_task(_collect_run(app, session_id=session_id))
            try:
                await asyncio.wait_for(store.boundary_reached.wait(), timeout=_WATCHDOG_SECONDS)
                running.cancel(f"{boundary} caller cancellation")
                store.release_publication.set()
                with pytest.raises(asyncio.CancelledError, match=boundary):
                    await asyncio.wait_for(running, timeout=_WATCHDOG_SECONDS)
            finally:
                store.release_publication.set()
                if not running.done():
                    running.cancel()
                await asyncio.wait_for(
                    asyncio.gather(running, return_exceptions=True), timeout=_WATCHDOG_SECONDS
                )

            assert sorted(tool.calls) == ["first", "second"]
            assert len(tool.calls) == 2
            assert len(provider.requests) == 1
            assert running.cancelling() == 0
            assert running.cancelled() is True
            await _assert_published_round(store, session_id=session_id)

    asyncio.run(scenario())


async def _create_process_loss_round(
    *,
    store: _ProcessLossStore,
    max_parallel_tool_calls: int,
    session_id: str,
) -> tuple[CayuApp, _TwoCallProvider, _SideEffectTool]:
    provider = _TwoCallProvider([_tool_call_response()])
    tool = _SideEffectTool()
    app = _runtime(
        store,
        provider,
        tool,
        max_parallel_tool_calls=max_parallel_tool_calls,
    )

    try:
        await _collect_run(app, session_id=session_id)
    except _SimulatedProcessLoss:
        pass
    else:  # pragma: no cover - the injected process boundary is mandatory
        raise AssertionError("The simulated process loss did not occur.")

    assert store.tool_publication_attempted.is_set()
    assert sorted(tool.calls) == ["first", "second"]
    assert len(tool.calls) == 2
    assert len(provider.requests) == 1
    checkpoint = await store.load_checkpoint(session_id)
    assert pending_round_reader.pending_tool_round_from_checkpoint(checkpoint) is not None
    assert not [
        message
        for message in await store.load_transcript(session_id)
        if message.role.value == "tool"
    ]

    store.fail_before_tool_publication = False
    await store.release_run_fence(session_id)
    await store.update_status(session_id, SessionStatus.INTERRUPTED)
    return app, provider, tool


@pytest.mark.parametrize(
    "max_parallel_tool_calls",
    [
        pytest.param(1, id="serial"),
        pytest.param(2, id="parallel"),
    ],
)
def test_two_call_round_recovers_after_process_loss_without_reexecution(
    max_parallel_tool_calls: int,
    store_factory,
) -> None:
    async def scenario() -> None:
        async with store_factory(_ProcessLossStore) as store:
            session_id = f"tool-round-process-loss-{max_parallel_tool_calls}"
            app, provider, tool = await _create_process_loss_round(
                store=store,
                max_parallel_tool_calls=max_parallel_tool_calls,
                session_id=session_id,
            )

            recovery = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )

            assert recovery.actions == (
                IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,
                IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND,
            )
            assert sorted(tool.calls) == ["first", "second"]
            assert len(tool.calls) == 2
            assert len(provider.requests) == 1
            await _assert_published_round(store, session_id=session_id)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "max_parallel_tool_calls",
    [
        pytest.param(1, id="serial"),
        pytest.param(2, id="parallel"),
    ],
)
def test_two_call_round_concurrent_recovery_has_one_publication_winner(
    max_parallel_tool_calls: int,
    store_factory,
) -> None:
    async def scenario() -> None:
        async with store_factory(_ConcurrentProcessLossStore) as store:
            session_id = f"tool-round-concurrent-recovery-{max_parallel_tool_calls}"
            app, provider, tool = await _create_process_loss_round(
                store=store,
                max_parallel_tool_calls=max_parallel_tool_calls,
                session_id=session_id,
            )
            competing_provider = _TwoCallProvider([])
            competing_tool = _SideEffectTool()
            competing_app = _runtime(
                store,
                competing_provider,
                competing_tool,
                max_parallel_tool_calls=max_parallel_tool_calls,
            )
            store.block_recovery_claim = True
            request = IncompleteSessionRecoveryRequest(session_id=session_id)

            first_recovery = asyncio.create_task(app.recover_incomplete_session(request))
            try:
                await asyncio.wait_for(store.claim_fenced.wait(), timeout=_WATCHDOG_SECONDS)
                competing = await asyncio.wait_for(
                    competing_app.recover_incomplete_session(request),
                    timeout=_WATCHDOG_SECONDS,
                )
                assert competing.actions == (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,)

                store.release_claim.set()
                winner = await asyncio.wait_for(first_recovery, timeout=_WATCHDOG_SECONDS)
            finally:
                store.release_claim.set()
                if not first_recovery.done():
                    first_recovery.cancel()
                await asyncio.wait_for(
                    asyncio.gather(first_recovery, return_exceptions=True),
                    timeout=_WATCHDOG_SECONDS,
                )

            assert winner.actions == (
                IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,
                IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND,
            )
            assert sorted(tool.calls) == ["first", "second"]
            assert len(tool.calls) == 2
            assert competing_tool.calls == []
            assert len(provider.requests) == 1
            assert competing_provider.requests == []
            await _assert_published_round(store, session_id=session_id)

    asyncio.run(scenario())


@pytest.mark.parametrize("entrance", ["run", "recovery"])
@pytest.mark.parametrize(
    "max_parallel_tool_calls", [pytest.param(1, id="serial"), pytest.param(2, id="parallel")]
)
def test_two_call_round_replays_exact_request_after_lost_acknowledgement(
    store_factory, entrance, max_parallel_tool_calls
) -> None:
    async def scenario() -> None:
        async with store_factory(_LostAcknowledgementStore) as store:
            session_id = f"tool-round-lost-ack-{entrance}-{max_parallel_tool_calls}"
            if entrance == "recovery":
                app, provider, tool = await _create_process_loss_round(
                    store=store,
                    max_parallel_tool_calls=max_parallel_tool_calls,
                    session_id=session_id,
                )
                recovery = await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id=session_id)
                )
                assert recovery.actions == (
                    IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,
                    IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND,
                )
                assert len(provider.requests) == 1
            else:
                store.fail_before_tool_publication = False
                provider = _TwoCallProvider(
                    [
                        _tool_call_response(),
                        [
                            ModelStreamEvent.text_delta("done"),
                            ModelStreamEvent.completed({"finish_reason": "stop"}),
                        ],
                    ]
                )
                tool = _SideEffectTool()
                app = _runtime(
                    store, provider, tool, max_parallel_tool_calls=max_parallel_tool_calls
                )
                events = await _collect_run(app, session_id=session_id)
                assert events[-1].type is EventType.SESSION_COMPLETED
                assert (await store.load(session_id)).status is SessionStatus.COMPLETED
                assert len(provider.requests) == 2

            assert sorted(tool.calls) == ["first", "second"]
            assert len(store.publication_requests) == 2
            assert store.publication_requests[0] == store.publication_requests[1]
            committed, replayed = store.publication_results
            assert committed.replayed is False and replayed.replayed is True
            assert committed.receipt == replayed.receipt
            await _assert_published_round(store, session_id=session_id)

    asyncio.run(scenario())


def _limited_runtime(store, *, completed_first, monkeypatch, session_id):
    # Freeze limit accounting through setup; only the first completed tool advances it.
    elapsed = [time.monotonic()]
    monkeypatch.setattr(
        run_limits,
        "time",
        SimpleNamespace(monotonic=lambda: elapsed[0]),
    )

    class LimitedTool(_SideEffectTool):
        async def run(self, ctx, args):
            result = await super().run(ctx, args)
            elapsed[0] = time.monotonic() + 5
            return result

    provider = _TwoCallProvider([_tool_call_response()])
    tool = LimitedTool()
    app = _runtime(store, provider, tool, max_parallel_tool_calls=1)
    request = RunRequest(
        agent_name="assistant",
        session_id=session_id,
        messages=[Message.text("user", "stop at the configured limit")],
        limits=(
            RunLimits(max_elapsed_seconds=1) if completed_first else RunLimits(max_tool_calls=1)
        ),
    )
    return app, provider, tool, request


async def _assert_limited_round(store, app, tool, *, completed_first, session_id):
    assert tool.calls == (["first"] if completed_first else [])
    transcript = await store.load_transcript(session_id)
    tool_messages = [message for message in transcript if message.role.value == "tool"]
    assert len(tool_messages) == 1
    results = [part for part in tool_messages[0].content if isinstance(part, ToolResultPart)]
    assert [part.tool_call_id for part in results] == ["call-side-effect-a", "call-side-effect-b"]
    events = await store.load_events(session_id)
    terminals = [
        event
        for event in events
        if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
    ]
    assert len(terminals) == 2
    assert [event.type for event in terminals] == [
        EventType.TOOL_CALL_COMPLETED if completed_first else EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_FAILED,
    ]
    for event in terminals[int(completed_first) :]:
        assert event.payload["reason"] == "limit_reached"
        assert event.payload["result"]["structured"]["skipped"] is True
    round_id = terminals[0].payload["tool_round_id"]
    receipt = await store.load_runtime_publication_receipt(session_id, f"tool-round:{round_id}")
    assert receipt is not None
    assert (
        pending_round_reader.pending_tool_round_from_checkpoint(
            await store.load_checkpoint(session_id)
        )
        is None
    )
    assert len({event.id for event in events}) == len(events)
    metrics = app.tool_terminal_publication_status()
    assert metrics.active_round_reservations == metrics.staged_count == 0


@pytest.mark.parametrize("completed_first", [False, True], ids=["before-dispatch", "partial"])
def test_limited_round_replays_exact_publication_and_repeated_close(
    store_factory, monkeypatch, completed_first
):
    async def scenario():
        async with store_factory(_LostAcknowledgementStore) as store:
            store.fail_before_tool_publication = False
            app, provider, tool, request = _limited_runtime(
                store,
                completed_first=completed_first,
                monkeypatch=monkeypatch,
                session_id=f"limited-lost-ack-{completed_first}",
            )
            close = app._session_engine._close_limited_tool_round
            close_arguments = {}

            async def capture_close(**kwargs):
                close_arguments.update(kwargs)
                async with aclosing(close(**kwargs)) as stream:
                    async for event in stream:
                        yield event

            monkeypatch.setattr(app._session_engine, "_close_limited_tool_round", capture_close)
            events = [event async for event in app.run(request)]
            assert events[-1].type is EventType.SESSION_INTERRUPTED
            assert (await store.load(request.session_id)).status is SessionStatus.INTERRUPTED
            assert len(provider.requests) == 1
            assert len(store.publication_requests) == 2
            assert store.publication_requests[0] == store.publication_requests[1]
            committed, replayed = store.publication_results
            assert committed.replayed is False and replayed.replayed is True
            assert committed.receipt == replayed.receipt
            await _assert_limited_round(
                store, app, tool, completed_first=completed_first, session_id=request.session_id
            )

            before_transcript = await store.load_transcript(request.session_id)
            before_events = await store.load_events(request.session_id)
            assert [event async for event in close(**close_arguments)] == []
            assert await store.load_transcript(request.session_id) == before_transcript
            assert await store.load_events(request.session_id) == before_events
            assert len(store.publication_requests) == 2
            assert close_arguments["messages"] == before_transcript
            await _assert_limited_round(
                store, app, tool, completed_first=completed_first, session_id=request.session_id
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("completed_first", [False, True], ids=["before-dispatch", "partial"])
@pytest.mark.parametrize("boundary", ["before-commit", "after-commit"])
def test_limited_round_preserves_repeated_cancellation_at_publication(
    store_factory, monkeypatch, completed_first, boundary
):
    async def scenario():
        async with store_factory(_PublicationBarrierStore, boundary=boundary) as store:
            app, provider, tool, request = _limited_runtime(
                store,
                completed_first=completed_first,
                monkeypatch=monkeypatch,
                session_id=f"limited-cancel-{boundary}-{completed_first}",
            )

            async def consume():
                return [event async for event in app.run(request)]

            running = asyncio.create_task(consume())
            try:
                await asyncio.wait_for(store.boundary_reached.wait(), timeout=_WATCHDOG_SECONDS)
                running.cancel("limited round cancellation")
                running.cancel("limited round cancellation")
                store.release_publication.set()
                with pytest.raises(asyncio.CancelledError, match="limited round cancellation"):
                    await asyncio.wait_for(running, timeout=_WATCHDOG_SECONDS)
            finally:
                store.release_publication.set()
                if not running.done():
                    running.cancel()
                await asyncio.wait_for(
                    asyncio.gather(running, return_exceptions=True), timeout=_WATCHDOG_SECONDS
                )
            assert running.cancelled()
            assert len(provider.requests) == 1
            await _assert_limited_round(
                store, app, tool, completed_first=completed_first, session_id=request.session_id
            )

    asyncio.run(scenario())


def _interruptible_runtime(store, *, completed_first, effect=ToolEffect.NONE):
    class InterruptibleTool(_SideEffectTool):
        spec = _SideEffectTool.spec.model_copy(update={"effect": effect})

        def __init__(self):
            super().__init__()
            self.blocked = asyncio.Event()

        async def run(self, ctx, args):
            if completed_first and args["value"] == "first":
                return await super().run(ctx, args)
            self.calls.append(args["value"])
            self.blocked.set()
            await asyncio.Event().wait()
            raise AssertionError("Interrupted tool unexpectedly resumed.")

    provider = _TwoCallProvider([_tool_call_response()])
    tool = InterruptibleTool()
    return _runtime(store, provider, tool, max_parallel_tool_calls=1), provider, tool


async def _interrupt(app, session_id):
    return [
        event
        async for event in app.interrupt_session(
            InterruptSessionRequest(session_id=session_id, reason="operator stop")
        )
    ]


async def _finish_tasks(*tasks):
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.wait_for(
        asyncio.gather(*tasks, return_exceptions=True), timeout=_WATCHDOG_SECONDS
    )


async def _assert_interrupted_round(store, app, tool, *, completed_first, session_id):
    assert tool.calls == (["first", "second"] if completed_first else ["first"])
    transcript = await store.load_transcript(session_id)
    tool_messages = [message for message in transcript if message.role.value == "tool"]
    assert len(tool_messages) == 1
    results = [part for part in tool_messages[0].content if isinstance(part, ToolResultPart)]
    assert [part.tool_call_id for part in results] == ["call-side-effect-a", "call-side-effect-b"]
    if completed_first:
        assert results[0].content == "executed first"
        assert results[0].is_error is False
    for result in results[int(completed_first) :]:
        assert result.is_error is True
        assert result.structured["interrupted"] is True
    events = await store.load_events(session_id)
    terminals = [
        event
        for event in events
        if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
    ]
    assert [event.payload["tool_call_id"] for event in terminals] == [
        "call-side-effect-a",
        "call-side-effect-b",
    ]
    assert [event.type for event in terminals] == [
        EventType.TOOL_CALL_COMPLETED if completed_first else EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_FAILED,
    ]
    round_id = terminals[0].payload["tool_round_id"]
    assert await store.load_runtime_publication_receipt(session_id, f"tool-round:{round_id}")
    assert (
        pending_round_reader.pending_tool_round_from_checkpoint(
            await store.load_checkpoint(session_id)
        )
        is None
    )
    assert len({event.id for event in events}) == len(events)
    metrics = app.tool_terminal_publication_status()
    assert metrics.active_round_reservations == metrics.staged_count == 0


@pytest.mark.parametrize("completed_first", [False, True], ids=["no-completed-call", "partial"])
def test_interrupted_round_replays_exact_publication_and_repeated_close(
    store_factory, monkeypatch, completed_first
):
    async def scenario():
        async with store_factory(_LostAcknowledgementStore) as store:
            store.fail_before_tool_publication = False
            session_id = f"interrupted-lost-ack-{completed_first}"
            app, provider, tool = _interruptible_runtime(store, completed_first=completed_first)
            executor = app._session_engine._tool_round_executor
            close = executor._close_interrupted_round
            close_requests = []

            async def capture_close(request):
                close_requests.append(request)
                async with aclosing(close(request)) as stream:
                    async for event in stream:
                        yield event

            monkeypatch.setattr(executor, "_close_interrupted_round", capture_close)
            running = asyncio.create_task(_collect_run(app, session_id=session_id))
            try:
                await asyncio.wait_for(tool.blocked.wait(), timeout=_WATCHDOG_SECONDS)
                interrupted = await asyncio.wait_for(
                    _interrupt(app, session_id), timeout=_WATCHDOG_SECONDS
                )
                events = await asyncio.wait_for(running, timeout=_WATCHDOG_SECONDS)
            finally:
                await _finish_tasks(running)
                assert await app.drain_background_interruptions()
            assert events[-1].type is EventType.SESSION_INTERRUPTED
            assert events[-1].id == interrupted[-1].id
            assert (await store.load(session_id)).status is SessionStatus.INTERRUPTED
            assert len(provider.requests) == 1
            assert len(store.publication_requests) == 2
            assert store.publication_requests[0] == store.publication_requests[1]
            committed, replayed = store.publication_results
            assert committed.replayed is False and replayed.replayed is True
            assert committed.receipt == replayed.receipt
            await _assert_interrupted_round(
                store, app, tool, completed_first=completed_first, session_id=session_id
            )

            before_transcript = await store.load_transcript(session_id)
            before_events = await store.load_events(session_id)
            assert len(close_requests) == 1
            assert [event async for event in close(close_requests[0])] == []
            assert await store.load_transcript(session_id) == before_transcript
            assert await store.load_events(session_id) == before_events
            assert close_requests[0].messages == before_transcript
            assert len(store.publication_requests) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("boundary", ["before-commit", "after-commit"])
def test_interrupted_round_preserves_repeated_cancellation_at_publication(
    store_factory, monkeypatch, boundary
):
    async def scenario():
        async with store_factory(_PublicationBarrierStore, boundary=boundary) as store:
            session_id = f"interrupted-cancel-{boundary}"
            app, provider, tool = _interruptible_runtime(store, completed_first=True)
            executor = app._session_engine._tool_round_executor
            close = executor._close_interrupted_round
            cancellations = []

            async def observe_close(request):
                try:
                    async with aclosing(close(request)) as stream:
                        async for event in stream:
                            yield event
                except asyncio.CancelledError as exc:
                    cancellations.append(exc.args)
                    raise

            monkeypatch.setattr(executor, "_close_interrupted_round", observe_close)
            running = asyncio.create_task(_collect_run(app, session_id=session_id))
            interrupting = None
            try:
                await asyncio.wait_for(tool.blocked.wait(), timeout=_WATCHDOG_SECONDS)
                interrupting = asyncio.create_task(_interrupt(app, session_id))
                await asyncio.wait_for(store.boundary_reached.wait(), timeout=_WATCHDOG_SECONDS)
                running.cancel("interrupted round publication cancellation")
                running.cancel("interrupted round publication cancellation")
                store.release_publication.set()
                events = await asyncio.wait_for(running, timeout=_WATCHDOG_SECONDS)
                interrupted = await asyncio.wait_for(interrupting, timeout=_WATCHDOG_SECONDS)
            finally:
                store.release_publication.set()
                await _finish_tasks(running, *([] if interrupting is None else [interrupting]))
                assert await app.drain_background_interruptions()
            # Closure retains cancellation; session control normalizes the operator stop.
            assert cancellations == [("interrupted round publication cancellation",)]
            assert events[-1].type is EventType.SESSION_INTERRUPTED
            assert events[-1].id == interrupted[-1].id
            assert len(provider.requests) == 1
            await _assert_interrupted_round(
                store, app, tool, completed_first=True, session_id=session_id
            )

    asyncio.run(scenario())


def test_interrupted_external_effect_keeps_round_pending_without_publication(store_factory):
    async def scenario():
        async with store_factory(_LostAcknowledgementStore) as store:
            store.fail_before_tool_publication = False
            session_id = "interrupted-external-effect"
            app, provider, tool = _interruptible_runtime(
                store, completed_first=True, effect=ToolEffect.EXTERNAL
            )
            running = asyncio.create_task(_collect_run(app, session_id=session_id))
            try:
                await asyncio.wait_for(tool.blocked.wait(), timeout=_WATCHDOG_SECONDS)
                await asyncio.wait_for(_interrupt(app, session_id), timeout=_WATCHDOG_SECONDS)
                await asyncio.wait_for(running, timeout=_WATCHDOG_SECONDS)
            finally:
                await _finish_tasks(running)
                assert await app.drain_background_interruptions()
            assert tool.calls == ["first", "second"]
            assert len(provider.requests) == 1
            assert store.publication_requests == []
            pending = pending_round_reader.pending_tool_round_from_checkpoint(
                await store.load_checkpoint(session_id)
            )
            assert pending is not None
            assert (
                await store.load_runtime_publication_receipt(
                    session_id, f"tool-round:{pending.tool_round_id}"
                )
                is None
            )
            events = await store.load_events(session_id)
            unknown = [
                event for event in events if event.type is EventType.TOOL_EFFECT_OUTCOME_UNKNOWN
            ]
            assert [event.payload["tool_call_id"] for event in unknown] == ["call-side-effect-b"]
            completed = [event for event in events if event.type is EventType.TOOL_CALL_COMPLETED]
            assert len(completed) == 1
            assert completed[0].payload["result"]["content"] == "executed first"
            assert not any(event.type is EventType.TOOL_CALL_FAILED for event in events)
            assert not any(
                message.role.value == "tool" for message in await store.load_transcript(session_id)
            )

    asyncio.run(scenario())
