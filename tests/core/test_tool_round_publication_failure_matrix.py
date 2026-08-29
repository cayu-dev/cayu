from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Literal

import pytest
from tests.core._execution_profile_fixtures import versioned_test_provider_identity

from cayu.core import AgentSpec, Event, EventType, ExecutionProfileBehaviorIdentity, Message
from cayu.core.messages import ToolResultPart
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    RunRequest,
    SessionStatus,
)
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime.sessions import (
    RuntimePublicationRequest,
    RuntimePublicationResult,
)


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


class _PublicationBarrierStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self, *, boundary: Literal["before-commit", "after-commit"]) -> None:
        super().__init__()
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


class _ProcessLossStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
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

    def __init__(self) -> None:
        super().__init__()
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
    store: InMemorySessionStore,
    provider: _TwoCallProvider,
    tool: _SideEffectTool,
    *,
    max_parallel_tool_calls: int,
) -> CayuApp:
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        max_parallel_tool_calls=max_parallel_tool_calls,
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
    store: InMemorySessionStore,
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
    assert tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint) is None
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
) -> None:
    async def scenario() -> None:
        store = _PublicationBarrierStore(boundary=boundary)
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
        await asyncio.wait_for(store.boundary_reached.wait(), timeout=5)
        running.cancel(f"{boundary} caller cancellation")
        store.release_publication.set()
        with pytest.raises(asyncio.CancelledError, match=boundary):
            await asyncio.wait_for(running, timeout=5)

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
    assert tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint) is not None
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
) -> None:
    async def scenario() -> None:
        store = _ProcessLossStore()
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
) -> None:
    async def scenario() -> None:
        store = _ConcurrentProcessLossStore()
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
        await asyncio.wait_for(store.claim_fenced.wait(), timeout=5)
        competing = await asyncio.wait_for(
            competing_app.recover_incomplete_session(request),
            timeout=5,
        )
        assert competing.actions == (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,)

        store.release_claim.set()
        winner = await asyncio.wait_for(first_recovery, timeout=5)

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
