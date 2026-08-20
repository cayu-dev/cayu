from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from tests.core._execution_profile_fixtures import create_admitted_session

from cayu import SQLiteSessionStore
from cayu.core import (
    AgentSpec,
    Event,
    EventType,
    ExecutionProfileBehaviorIdentity,
    Message,
    ThinkingConfig,
    ThinkingPart,
)
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.providers import (
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
    ProviderOperationAdapter,
    ProviderOperationConnection,
    ProviderOperationMode,
    ProviderOperationSnapshot,
    ProviderOperationStartIdempotencySupport,
    ProviderOperationStartRequest,
    ProviderOperationState,
    ProviderOperationStatus,
)
from cayu.providers._credential_boundary import ProviderStreamCleanupError
from cayu.runtime import (
    AllowAllToolPolicy,
    CayuApp,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    ModelCompletionManualRecoveryRequired,
    RunRequest,
    Session,
    SessionRunFenced,
    SessionStatus,
    SessionStore,
    session_usage_summary,
)
from cayu.runtime._model_step_executor import ModelCompletionRecoveryContext
from cayu.runtime.execution_units import ModelAttemptIdentity
from cayu.runtime.provider_operations import (
    ProviderOperationEvidenceError,
    ProviderOperationInspectionStatus,
    ProviderOperationUnavailableReason,
    commit_provider_operation_progress,
    inspect_provider_operation,
    load_recoverable_provider_operation,
    load_recoverable_provider_operation_start,
    provider_operation_progress_event_id,
    provider_operation_progress_payload,
)
from cayu.runtime.sessions import (
    ModelCompletionStage,
    ModelCompletionStageRequest,
    SessionOperationTransform,
)
from cayu.vaults import SecretRedactor

_PROFILE_UNSET = object()
_PROFILE_MISSING = object()


def _profile_evidence_payload(
    value: object,
    *,
    default: str,
) -> dict[str, object]:
    selected = default if value is _PROFILE_UNSET else value
    if selected is _PROFILE_MISSING:
        return {}
    return {"execution_profile_fingerprint": selected}


def _stage_execution_profile_fingerprint(stage: ModelCompletionStage) -> str:
    recovery_context = stage.intent.get("recovery_context")
    assert type(recovery_context) is dict
    fingerprint = recovery_context.get("execution_profile_fingerprint")
    assert type(fingerprint) is str
    return fingerprint


class _CursorReplayAdapter(ProviderOperationAdapter):
    def __init__(self) -> None:
        self.initial_state = ProviderOperationState(
            operation_id="response_cursor_123",
            stream_protocol="responses-v1",
            recovery_metadata={"cursor": 0, "opaque": {"token": "start"}},
        )
        self.start_calls = 0
        self.retrieve_calls = 0
        self.reconnect_calls: list[ProviderOperationState] = []

    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        del request
        self.start_calls += 1
        raise AssertionError("cursor recovery must not submit a replacement operation")

    async def retrieve(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        del state
        self.retrieve_calls += 1
        raise AssertionError("partial progress must reconnect instead of retrieve")

    async def reconnect(self, state: ProviderOperationState) -> ProviderOperationConnection:
        self.reconnect_calls.append(state)

        async def events() -> AsyncIterator[ModelStreamEvent]:
            # The provider deliberately replays its inclusive boundary. Cayu must
            # accept the exact replay without re-appending or re-transcripting it.
            yield ModelStreamEvent.text_delta(
                "l",
                recovery_metadata={"cursor": 2, "opaque": {"token": "after-l"}},
            )
            yield ModelStreamEvent.text_delta(
                "o",
                recovery_metadata={"cursor": 3, "opaque": {"token": "after-o"}},
            )
            yield ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
                recovery_metadata={"cursor": 4, "opaque": {"token": "terminal"}},
            )
            yield ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
                recovery_metadata={"cursor": 4, "opaque": {"token": "terminal"}},
            )

        return ProviderOperationConnection(
            state=state,
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )

    async def cancel(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        raise AssertionError(f"cursor recovery must not cancel {state.operation_id}")


class _CursorReplayProvider(ModelProvider):
    name = "cursor-replay"

    def __init__(self) -> None:
        self.adapter = _CursorReplayAdapter()

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name=f"tests:provider-operation-cursor:{type(self).__name__}",
            behavior_version="1",
            implementation_version="1",
        )

    @property
    def provider_operation_mode(self) -> ProviderOperationMode:
        return ProviderOperationMode.BACKGROUND

    @property
    def provider_operations(self) -> ProviderOperationAdapter:
        return self.adapter

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        raise AssertionError(f"background provider must not synchronously stream {request.model}")
        yield  # pragma: no cover


class _PendingCursorAdapter(_CursorReplayAdapter):
    async def reconnect(self, state: ProviderOperationState) -> ProviderOperationConnection:
        self.reconnect_calls.append(state)

        async def events() -> AsyncIterator[ModelStreamEvent]:
            cursor = state.recovery_metadata.cursor
            assert cursor is not None
            yield ModelStreamEvent.text_delta(
                "hel",
                recovery_metadata={"cursor": cursor, "opaque": {"token": "after-hel"}},
            )

        return ProviderOperationConnection(
            state=state,
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )


class _PendingCursorProvider(_CursorReplayProvider):
    def __init__(self) -> None:
        self.adapter = _PendingCursorAdapter()


class _PostTerminalSnapshotAdapter(_CursorReplayAdapter):
    async def retrieve(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        self.retrieve_calls += 1
        return ProviderOperationSnapshot(
            state=state,
            status=ProviderOperationStatus.COMPLETED,
            events=(
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {"input_tokens": 3, "output_tokens": 2},
                    },
                    recovery_metadata={"cursor": 1},
                ),
                ModelStreamEvent.text_delta(
                    "contradictory tail",
                    recovery_metadata={"cursor": 2},
                ),
            ),
        )


class _PostTerminalSnapshotProvider(_CursorReplayProvider):
    def __init__(self) -> None:
        self.adapter = _PostTerminalSnapshotAdapter()


class _CloseTrackedGapEvents:
    def __init__(self, event: ModelStreamEvent) -> None:
        self._event = event
        self._yielded = False
        self.closed = False

    def __aiter__(self) -> _CloseTrackedGapEvents:
        return self

    async def __anext__(self) -> ModelStreamEvent:
        if self._yielded:
            raise StopAsyncIteration
        self._yielded = True
        return self._event

    async def aclose(self) -> None:
        self.closed = True


class _ClosingGapAdapter(_CursorReplayAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.reconnect_events: _CloseTrackedGapEvents | None = None

    async def reconnect(self, state: ProviderOperationState) -> ProviderOperationConnection:
        self.reconnect_calls.append(state)
        cursor = state.recovery_metadata.cursor
        assert cursor is not None
        self.reconnect_events = _CloseTrackedGapEvents(
            ModelStreamEvent.text_delta(
                "gap",
                recovery_metadata={"cursor": cursor + 2},
            )
        )
        return ProviderOperationConnection(
            state=state,
            status=ProviderOperationStatus.IN_PROGRESS,
            events=self.reconnect_events,
        )


class _ClosingGapProvider(_CursorReplayProvider):
    def __init__(self) -> None:
        self.adapter = _ClosingGapAdapter()


class _TerminalThenBlockingEvents:
    def __init__(
        self,
        event: ModelStreamEvent,
        *,
        close_error: BaseException | None = None,
    ) -> None:
        self._event = event
        self._close_error = close_error
        self._yielded = False
        self.closed = False

    def __aiter__(self) -> _TerminalThenBlockingEvents:
        return self

    async def __anext__(self) -> ModelStreamEvent:
        if not self._yielded:
            self._yielded = True
            return self._event
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        self.closed = True
        if self._close_error is not None:
            raise self._close_error


class _TerminalThenBlockingAdapter(_CursorReplayAdapter):
    def __init__(self, *, close_error: BaseException | None = None) -> None:
        super().__init__()
        self.close_error = close_error
        self.reconnect_events: _TerminalThenBlockingEvents | None = None

    async def reconnect(self, state: ProviderOperationState) -> ProviderOperationConnection:
        self.reconnect_calls.append(state)
        cursor = state.recovery_metadata.cursor
        assert cursor is not None
        self.reconnect_events = _TerminalThenBlockingEvents(
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
                recovery_metadata={"cursor": cursor + 1},
            ),
            close_error=self.close_error,
        )
        return ProviderOperationConnection(
            state=state,
            status=ProviderOperationStatus.IN_PROGRESS,
            events=self.reconnect_events,
        )


class _TerminalThenBlockingProvider(_CursorReplayProvider):
    def __init__(self, *, close_error: BaseException | None = None) -> None:
        self.adapter = _TerminalThenBlockingAdapter(close_error=close_error)


class _CancellationBlockingCloseEvents(_TerminalThenBlockingEvents):
    def __init__(self, event: ModelStreamEvent) -> None:
        super().__init__(event)
        self.close_started = asyncio.Event()

    async def aclose(self) -> None:
        self.closed = True
        self.close_started.set()
        await asyncio.Event().wait()


class _CancellationBlockingCloseAdapter(_CursorReplayAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.connection_created = asyncio.Event()
        self.reconnect_events: _CancellationBlockingCloseEvents | None = None

    async def reconnect(self, state: ProviderOperationState) -> ProviderOperationConnection:
        self.reconnect_calls.append(state)
        cursor = state.recovery_metadata.cursor
        assert cursor is not None
        self.reconnect_events = _CancellationBlockingCloseEvents(
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
                recovery_metadata={"cursor": cursor + 1},
            )
        )
        self.connection_created.set()
        return ProviderOperationConnection(
            state=state,
            status=ProviderOperationStatus.IN_PROGRESS,
            events=self.reconnect_events,
        )


class _CancellationBlockingCloseProvider(_CursorReplayProvider):
    def __init__(self) -> None:
        self.adapter = _CancellationBlockingCloseAdapter()


class _InvalidCompletionAdapter(_CursorReplayAdapter):
    async def reconnect(self, state: ProviderOperationState) -> ProviderOperationConnection:
        self.reconnect_calls.append(state)

        async def events() -> AsyncIterator[ModelStreamEvent]:
            cursor = state.recovery_metadata.cursor
            assert cursor is not None
            valid = ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
                recovery_metadata={"cursor": cursor + 1},
            )
            yield valid.model_copy(
                update={
                    "payload": {
                        **valid.payload,
                        "invalid": object(),
                    }
                }
            )

        return ProviderOperationConnection(
            state=state,
            status=ProviderOperationStatus.COMPLETED,
            events=events(),
        )


class _InvalidCompletionProvider(_CursorReplayProvider):
    def __init__(self) -> None:
        self.adapter = _InvalidCompletionAdapter()


class _MutatingReconnectAdapter(_CursorReplayAdapter):
    async def reconnect(self, state: ProviderOperationState) -> ProviderOperationConnection:
        self.reconnect_calls.append(state)
        state.recovery_metadata.opaque["token"] = "adapter-mutated"

        async def events() -> AsyncIterator[ModelStreamEvent]:
            cursor = state.recovery_metadata.cursor
            assert cursor is not None
            yield ModelStreamEvent.completed(
                {"finish_reason": "stop"},
                recovery_metadata={"cursor": cursor + 1},
            )

        return ProviderOperationConnection(
            state=state,
            status=ProviderOperationStatus.COMPLETED,
            events=events(),
        )


class _MutatingReconnectProvider(_CursorReplayProvider):
    def __init__(self) -> None:
        self.adapter = _MutatingReconnectAdapter()


class _SecretRecoveryAdapter(_CursorReplayAdapter):
    def __init__(self, *, secret: str, secret_in_key: bool) -> None:
        super().__init__()
        self.secret = secret
        self.secret_in_key = secret_in_key

    async def reconnect(self, state: ProviderOperationState) -> ProviderOperationConnection:
        self.reconnect_calls.append(state)

        async def events() -> AsyncIterator[ModelStreamEvent]:
            cursor = state.recovery_metadata.cursor
            assert cursor is not None
            opaque = {self.secret: "safe"} if self.secret_in_key else {"continuation": self.secret}
            yield ModelStreamEvent.text_delta(
                "safe output",
                recovery_metadata={"cursor": cursor + 1, "opaque": opaque},
            )

        return ProviderOperationConnection(
            state=state,
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )


class _SecretRecoveryProvider(_CursorReplayProvider):
    def __init__(self, *, secret: str, secret_in_key: bool) -> None:
        self.adapter = _SecretRecoveryAdapter(secret=secret, secret_in_key=secret_in_key)


class _MalformedOutputRecoveryAdapter(_CursorReplayAdapter):
    def __init__(self, event_kind: str) -> None:
        super().__init__()
        self.event_kind = event_kind

    async def reconnect(self, state: ProviderOperationState) -> ProviderOperationConnection:
        self.reconnect_calls.append(state)

        async def events() -> AsyncIterator[ModelStreamEvent]:
            cursor = state.recovery_metadata.cursor
            assert cursor is not None
            if self.event_kind == "tool_call":
                valid = ModelStreamEvent.tool_call(
                    name="lookup",
                    arguments={},
                    recovery_metadata={"cursor": cursor + 1},
                )
                yield valid.model_copy(update={"payload": {"arguments": {}}})
                return
            valid = ModelStreamEvent.thinking(
                "private",
                recovery_metadata={"cursor": cursor + 1},
            )
            yield valid.model_copy(update={"payload": {"provider_state": True}})

        return ProviderOperationConnection(
            state=state,
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )


class _MalformedOutputRecoveryProvider(_CursorReplayProvider):
    def __init__(self, event_kind: str) -> None:
        self.adapter = _MalformedOutputRecoveryAdapter(event_kind)


class _ProgressPublishBarrierStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.pause_next_progress = False
        self.progress_publish_entered = asyncio.Event()
        self.progress_publish_release = asyncio.Event()

    async def publish_session_operation_guarded(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: SessionOperationTransform,
        commit_guard: Callable[[], None],
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        if self.pause_next_progress:
            self.pause_next_progress = False
            self.progress_publish_entered.set()
            await self.progress_publish_release.wait()
        return await super().publish_session_operation_guarded(
            session_id,
            idempotency_key=idempotency_key,
            operation_transform=operation_transform,
            commit_guard=commit_guard,
            events=events,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )


class _ToolCursorReplayAdapter(_CursorReplayAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.initial_state = ProviderOperationState(
            operation_id="response_tool_cursor_123",
            stream_protocol="responses-v1",
            recovery_metadata={"cursor": 0, "opaque": {"token": "start"}},
        )

    async def reconnect(self, state: ProviderOperationState) -> ProviderOperationConnection:
        self.reconnect_calls.append(state)

        async def events() -> AsyncIterator[ModelStreamEvent]:
            tool_call = ModelStreamEvent.tool_call(
                id="lookup-1",
                name="lookup",
                arguments={"query": "recovered"},
                recovery_metadata={"cursor": 1, "opaque": {"token": "after-tool"}},
            )
            yield tool_call
            yield ModelStreamEvent.completed(
                {"finish_reason": "tool_calls"},
                recovery_metadata={"cursor": 2, "opaque": {"token": "terminal"}},
            )
            yield ModelStreamEvent.completed(
                {"finish_reason": "tool_calls"},
                recovery_metadata={"cursor": 2, "opaque": {"token": "terminal"}},
            )

        return ProviderOperationConnection(
            state=state,
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )


class _ToolCursorReplayProvider(_CursorReplayProvider):
    def __init__(self) -> None:
        self.adapter = _ToolCursorReplayAdapter()


class _ThinkingCursorReplayAdapter(_CursorReplayAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.initial_state = ProviderOperationState(
            operation_id="response_thinking_cursor_123",
            stream_protocol="responses-v1",
            recovery_metadata={"cursor": 0, "opaque": {"token": "start"}},
        )

    async def reconnect(self, state: ProviderOperationState) -> ProviderOperationConnection:
        self.reconnect_calls.append(state)

        async def events() -> AsyncIterator[ModelStreamEvent]:
            # Inclusive replay of the last readable reasoning block must not
            # duplicate its public delta or transcript part.
            yield ModelStreamEvent.thinking(
                "consider",
                provider_state={"type": "thinking", "signature": "SIG-1"},
                recovery_metadata={"cursor": 1, "opaque": {"token": "after-readable"}},
            )
            yield ModelStreamEvent.thinking(
                provider_state={"type": "redacted_thinking", "data": "OPAQUE"},
                recovery_metadata={"cursor": 2, "opaque": {"token": "after-opaque"}},
            )
            yield ModelStreamEvent.text_delta(
                "answer",
                recovery_metadata={"cursor": 3, "opaque": {"token": "after-answer"}},
            )
            yield ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 2, "output_tokens": 1},
                },
                recovery_metadata={"cursor": 4, "opaque": {"token": "terminal"}},
            )

        return ProviderOperationConnection(
            state=state,
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )


class _ThinkingCursorReplayProvider(_CursorReplayProvider):
    def __init__(self) -> None:
        self.adapter = _ThinkingCursorReplayAdapter()


class _LookupTool(Tool):
    spec = ToolSpec(
        name="lookup",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        effect=ToolEffect.NONE,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:provider-operation-cursor-recovery:lookup-tool",
            behavior_version="1",
            implementation_version="1",
        ),
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        return ToolResult(content="lookup complete")


def test_partial_provider_stream_reconnects_after_cursor_without_duplicate_output() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _CursorReplayProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "provider-cursor-replay"

        await _stage_partial_operation(
            store,
            session_id=session_id,
            provider=provider,
            advances=2,
        )

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

        assert provider.adapter.start_calls == 0
        assert provider.adapter.retrieve_calls == 0
        assert len(provider.adapter.reconnect_calls) == 1
        reconnect_state = provider.adapter.reconnect_calls[0]
        assert reconnect_state.recovery_metadata.cursor == 2
        assert reconnect_state.recovery_metadata.opaque == {"token": "after-l"}

        transcript = await store.load_transcript(session_id)
        assert transcript[-1].content[0].text == "hello"
        events = await store.load_events(session_id)
        text_events = [event for event in events if event.type == EventType.MODEL_TEXT_DELTA]
        assert [event.payload["delta"] for event in text_events] == ["hel", "l", "o"]
        completed = [event for event in events if event.type == EventType.MODEL_COMPLETED]
        assert len(completed) == 1
        assert completed[0].payload["usage_metrics"]["total_tokens"] == 5
        assert (
            completed[0].payload["provider_operation_progress"]["stream_event"][
                "recovery_metadata"
            ]["cursor"]
            == 4
        )

    asyncio.run(scenario())


def test_pending_cursor_reconnect_returns_to_scheduled_inspection_state() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _PendingCursorProvider()
        session_id = "provider-cursor-still-pending"
        await _stage_partial_operation(
            store,
            session_id=session_id,
            provider=provider,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

        assert len(provider.adapter.reconnect_calls) == 1
        assert provider.adapter.reconnect_calls[0].recovery_metadata.cursor == 1
        inspection = await inspect_provider_operation(store, session_id)
        assert inspection.status is ProviderOperationInspectionStatus.RECONNECT_SCHEDULED
        recovery_events = [
            event
            for event in await store.load_events(session_id)
            if event.type
            in {
                EventType.PROVIDER_OPERATION_RECONNECT_SCHEDULED,
                EventType.PROVIDER_OPERATION_RECONNECT_STARTED,
            }
        ]
        assert [event.type for event in recovery_events] == [
            EventType.PROVIDER_OPERATION_RECONNECT_SCHEDULED,
            EventType.PROVIDER_OPERATION_RECONNECT_STARTED,
            EventType.PROVIDER_OPERATION_RECONNECT_SCHEDULED,
        ]
        assert await store.load_active_model_completion_stage(session_id) is not None

    asyncio.run(scenario())


def test_retrieved_post_terminal_output_preserves_non_turn_completion_and_usage() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _PostTerminalSnapshotProvider()
        session_id = "provider-snapshot-post-terminal-output"
        await _stage_partial_operation(
            store,
            session_id=session_id,
            provider=provider,
            partial_events=(),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        with pytest.raises(RuntimeError, match="output after completion"):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_before=datetime.now(UTC) + timedelta(seconds=1),
                )
            )

        assert provider.adapter.start_calls == 0
        assert provider.adapter.retrieve_calls == 1
        assert provider.adapter.reconnect_calls == []
        stored = await store.load_events(session_id)
        completed = [event for event in stored if event.type is EventType.MODEL_COMPLETED]
        assert len(completed) == 1
        assert completed[0].payload["usage_metrics"]["total_tokens"] == 5
        assert completed[0].payload["step_classification"]["type"] == "failed"
        assert completed[0].payload["transcript_cursor"] == 1
        transcript = await store.load_transcript(session_id)
        assert len(transcript) == 1
        assert transcript[0].role == "user"

    asyncio.run(scenario())


def test_partial_tool_call_replay_materializes_one_pending_call() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _ToolCursorReplayProvider()
        session_id = "provider-tool-cursor-replay"
        accepted_tool = ModelStreamEvent.tool_call(
            id="lookup-1",
            name="lookup",
            arguments={"query": "recovered"},
            recovery_metadata={"cursor": 1, "opaque": {"token": "after-tool"}},
        )
        await _stage_partial_operation(
            store,
            session_id=session_id,
            provider=provider,
            partial_events=(accepted_tool,),
            tools=(_LookupTool(),),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[_LookupTool()],
            tool_policy=AllowAllToolPolicy(),
        )

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

        assert provider.adapter.start_calls == 0
        assert provider.adapter.retrieve_calls == 0
        assert len(provider.adapter.reconnect_calls) == 1
        assert provider.adapter.reconnect_calls[0].recovery_metadata.cursor == 1
        events = await store.load_events(session_id)
        progress = [
            event for event in events if event.type == EventType.PROVIDER_OPERATION_PROGRESS
        ]
        assert len(progress) == 1
        assert progress[0].payload["provider_operation_progress"]["stream_event"]["payload"] == {
            "name": "lookup",
            "arguments": {"query": "recovered"},
            "id": "lookup-1",
        }
        completed = [event for event in events if event.type == EventType.MODEL_COMPLETED]
        assert len(completed) == 1
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        pending = checkpoint["pending_tool_round"]
        assert [call["tool_call_id"] for call in pending["tool_calls"]] == ["lookup-1"]

    asyncio.run(scenario())


def test_partial_thinking_replay_preserves_state_without_duplicate_public_delta() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _ThinkingCursorReplayProvider()
        session_id = "provider-thinking-cursor-replay"
        thinking = ThinkingConfig(include_in_transcript=False)
        accepted_thinking = ModelStreamEvent.thinking(
            "consider",
            provider_state={"type": "thinking", "signature": "SIG-1"},
            recovery_metadata={"cursor": 1, "opaque": {"token": "after-readable"}},
        )
        await _stage_partial_operation(
            store,
            session_id=session_id,
            provider=provider,
            partial_events=(accepted_thinking,),
            thinking=thinking,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model", thinking=thinking))

        result = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

        assert len(provider.adapter.reconnect_calls) == 1
        assert provider.adapter.reconnect_calls[0].recovery_metadata.cursor == 1
        public_thinking = [
            event for event in result.events if event.type == EventType.MODEL_THINKING_DELTA
        ]
        assert public_thinking == []
        public_opaque = [
            event for event in result.events if event.type == EventType.PROVIDER_OPERATION_PROGRESS
        ]
        assert len(public_opaque) == 1
        assert "provider_operation_progress" not in public_opaque[0].payload
        assert "provider_state" not in public_opaque[0].payload

        events = await store.load_events(session_id)
        readable = [event for event in events if event.type == EventType.MODEL_THINKING_DELTA]
        assert [event.payload["delta"] for event in readable] == ["consider"]
        opaque = [event for event in events if event.type == EventType.PROVIDER_OPERATION_PROGRESS]
        assert len(opaque) == 1
        assert opaque[0].payload["provider_operation_progress"]["stream_event"]["payload"] == {
            "provider_state": {"type": "redacted_thinking", "data": "OPAQUE"}
        }

        transcript = await store.load_transcript(session_id)
        thinking_parts = [part for part in transcript[-1].content if isinstance(part, ThinkingPart)]
        assert [(part.text, part.provider_state) for part in thinking_parts] == [
            ("consider", {"type": "thinking", "signature": "SIG-1"}),
            ("", {"type": "redacted_thinking", "data": "OPAQUE"}),
        ]

    asyncio.run(scenario())


def test_competing_partial_cursor_recovery_workers_publish_once() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _CursorReplayProvider()
        session_id = "competing-provider-cursor-replay"
        await _stage_partial_operation(
            store,
            session_id=session_id,
            provider=provider,
            advances=2,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        request = IncompleteSessionRecoveryRequest(
            session_id=session_id,
            inactive_before=datetime.now(UTC),
        )

        outcomes = await asyncio.gather(
            app.recover_incomplete_session(request),
            app.recover_incomplete_session(request),
        )

        assert (
            sum(
                IncompleteSessionRecoveryAction.SKIPPED_ACTIVE in outcome.actions
                for outcome in outcomes
            )
            == 1
        )
        assert len(provider.adapter.reconnect_calls) == 1
        events = await store.load_events(session_id)
        assert sum(event.type == EventType.MODEL_COMPLETED for event in events) == 1
        assert [
            event.payload["delta"] for event in events if event.type == EventType.MODEL_TEXT_DELTA
        ] == ["hel", "l", "o"]
        transcript = await store.load_transcript(session_id)
        assert len(transcript) == 2
        assert transcript[-1].content[0].text == "hello"
        usage = session_usage_summary(session_id, events)
        assert usage.model_steps == 1
        assert usage.usage.total_tokens == 5

    asyncio.run(scenario())


def test_cursor_validation_failure_closes_stream_and_requires_resolution() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _ClosingGapProvider()
        session_id = "provider-cursor-gap-close"
        await _stage_partial_operation(
            store,
            session_id=session_id,
            provider=provider,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

        assert provider.adapter.reconnect_events is not None
        assert provider.adapter.reconnect_events.closed is True
        inspection = await inspect_provider_operation(store, session_id)
        assert inspection.status is ProviderOperationInspectionStatus.PROVIDER_OPERATION_UNAVAILABLE
        assert inspection.recovery_reason is ProviderOperationUnavailableReason.MALFORMED
        assert inspection.duplicate_request_risk

    asyncio.run(scenario())


def test_reconnect_stops_and_closes_stream_after_terminal_event() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _TerminalThenBlockingProvider()
        session_id = "provider-cursor-terminal-close"
        await _stage_partial_operation(
            store,
            session_id=session_id,
            provider=provider,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await asyncio.wait_for(
            app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_before=datetime.now(UTC) + timedelta(seconds=1),
                )
            ),
            timeout=1.0,
        )

        assert provider.adapter.reconnect_events is not None
        assert provider.adapter.reconnect_events.closed is True
        stored = await store.load_events(session_id)
        assert sum(event.type is EventType.MODEL_COMPLETED for event in stored) == 1

    asyncio.run(scenario())


def test_reconnect_cleanup_failure_preserves_non_turn_completion_and_usage() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _TerminalThenBlockingProvider(close_error=RuntimeError("provider close failed"))
        session_id = "provider-cursor-terminal-close-failure"
        await _stage_partial_operation(
            store,
            session_id=session_id,
            provider=provider,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        with pytest.raises(ProviderStreamCleanupError):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_before=datetime.now(UTC) + timedelta(seconds=1),
                )
            )

        assert provider.adapter.start_calls == 0
        assert len(provider.adapter.reconnect_calls) == 1
        assert provider.adapter.reconnect_events is not None
        assert provider.adapter.reconnect_events.closed is True
        stored = await store.load_events(session_id)
        completed = [event for event in stored if event.type is EventType.MODEL_COMPLETED]
        assert len(completed) == 1
        assert completed[0].payload["usage_metrics"]["total_tokens"] == 5
        assert completed[0].payload["step_classification"]["type"] == "failed"
        assert completed[0].payload["transcript_cursor"] == 1
        assert EventType.PROVIDER_OPERATION_RECONCILED not in {event.type for event in stored}
        transcript = await store.load_transcript(session_id)
        assert len(transcript) == 1
        assert transcript[0].role == "user"
        usage = session_usage_summary(session_id, stored)
        assert usage.model_steps == 1
        assert usage.usage.total_tokens == 5

    asyncio.run(scenario())


def test_invalid_recovered_completion_preserves_non_turn_usage_evidence() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _InvalidCompletionProvider()
        session_id = "provider-cursor-invalid-completion"
        await _stage_partial_operation(
            store,
            session_id=session_id,
            provider=provider,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        with pytest.raises(ModelProviderError, match="invalid completion metadata"):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_before=datetime.now(UTC) + timedelta(seconds=1),
                )
            )

        assert provider.adapter.start_calls == 0
        assert len(provider.adapter.reconnect_calls) == 1
        stored = await store.load_events(session_id)
        completed = [event for event in stored if event.type is EventType.MODEL_COMPLETED]
        assert len(completed) == 1
        assert completed[0].payload["usage_metrics"]["total_tokens"] == 5
        assert completed[0].payload["step_classification"]["type"] == "failed"
        assert completed[0].payload["completion_outcome"] == "invalid_metadata"
        assert completed[0].payload["transcript_cursor"] == 1
        assert EventType.PROVIDER_OPERATION_RECONCILED not in {event.type for event in stored}
        transcript = await store.load_transcript(session_id)
        assert len(transcript) == 1
        usage = session_usage_summary(session_id, stored)
        assert usage.model_steps == 1
        assert usage.usage.total_tokens == 5

    asyncio.run(scenario())


def test_cancellation_during_reconnect_close_preserves_completion_then_propagates() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _CancellationBlockingCloseProvider()
        session_id = "provider-cursor-terminal-close-cancelled"
        await _stage_partial_operation(
            store,
            session_id=session_id,
            provider=provider,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        recovery = asyncio.create_task(
            app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_before=datetime.now(UTC) + timedelta(seconds=1),
                )
            )
        )
        await asyncio.wait_for(provider.adapter.connection_created.wait(), timeout=1.0)
        reconnect_events = provider.adapter.reconnect_events
        assert reconnect_events is not None
        await asyncio.wait_for(reconnect_events.close_started.wait(), timeout=1.0)
        assert recovery.cancelling() == 0
        recovery.cancel()
        with pytest.raises(asyncio.CancelledError):
            await recovery

        assert recovery.cancelling() == 1
        assert recovery.cancelled() is True
        assert provider.adapter.start_calls == 0
        assert len(provider.adapter.reconnect_calls) == 1
        assert reconnect_events.closed is True
        stored = await store.load_events(session_id)
        completed = [event for event in stored if event.type is EventType.MODEL_COMPLETED]
        assert len(completed) == 1
        assert completed[0].payload["usage_metrics"]["total_tokens"] == 5
        assert completed[0].payload["step_classification"]["type"] == "failed"
        assert completed[0].payload["transcript_cursor"] == 1
        transcript = await store.load_transcript(session_id)
        assert len(transcript) == 1

    asyncio.run(scenario())


def test_reconnect_adapter_cannot_mutate_runtime_owned_operation_state() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _MutatingReconnectProvider()
        session_id = "provider-cursor-mutated-state"
        await _stage_partial_operation(
            store,
            session_id=session_id,
            provider=provider,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

        assert provider.adapter.reconnect_calls[0].recovery_metadata.opaque == {
            "token": "adapter-mutated"
        }
        stored = await store.load_events(session_id)
        progress = next(event for event in stored if event.type == EventType.MODEL_TEXT_DELTA)
        assert progress.payload["provider_operation_progress"]["stream_event"]["recovery_metadata"][
            "opaque"
        ] == {"token": "after-hel"}
        assert EventType.MODEL_COMPLETED not in {event.type for event in stored}
        inspection = await inspect_provider_operation(store, session_id)
        assert inspection.recovery_reason is ProviderOperationUnavailableReason.WRONG_PROVIDER
        assert inspection.duplicate_request_risk

    asyncio.run(scenario())


@pytest.mark.parametrize("secret_in_key", [False, True])
def test_reconnect_rejects_secret_bearing_opaque_state_before_progress_commit(
    secret_in_key: bool,
) -> None:
    async def scenario() -> None:
        secret = "cursor-recovery-secret-ABCDEFGHIJKLMNOP"
        store = InMemorySessionStore()
        provider = _SecretRecoveryProvider(
            secret=secret,
            secret_in_key=secret_in_key,
        )
        session_id = f"provider-cursor-secret-{secret_in_key}"
        await _stage_partial_operation(
            store,
            session_id=session_id,
            provider=provider,
        )
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

        stored = await store.load_events(session_id)
        assert [
            event.payload["delta"] for event in stored if event.type == EventType.MODEL_TEXT_DELTA
        ] == ["hel"]
        assert secret not in repr(stored)
        inspection = await inspect_provider_operation(store, session_id)
        assert inspection.recovery_reason is ProviderOperationUnavailableReason.MALFORMED
        assert inspection.duplicate_request_risk

    asyncio.run(scenario())


@pytest.mark.parametrize("secret_in_key", [False, True])
def test_reconnect_revalidates_stored_opaque_state_against_current_secrets(
    secret_in_key: bool,
) -> None:
    async def scenario() -> None:
        secret = "rotated-cursor-secret-ABCDEFGHIJKLMNOP"
        opaque = {secret: "safe"} if secret_in_key else {"continuation": secret}
        store = InMemorySessionStore()
        provider = _CursorReplayProvider()
        session_id = f"provider-cursor-rotated-secret-{secret_in_key}"
        await _stage_partial_operation(
            store,
            session_id=session_id,
            provider=provider,
            partial_events=(
                ModelStreamEvent.text_delta(
                    "hel",
                    recovery_metadata={"cursor": 1, "opaque": opaque},
                ),
            ),
        )
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        with pytest.raises(ProviderOperationEvidenceError, match="current workload secret"):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_before=datetime.now(UTC) + timedelta(seconds=1),
                )
            )

        assert provider.adapter.start_calls == 0
        assert provider.adapter.reconnect_calls == []

    asyncio.run(scenario())


@pytest.mark.parametrize("event_kind", ["tool_call", "thinking"])
def test_reconnect_rejects_malformed_output_before_cursor_commit(event_kind: str) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _MalformedOutputRecoveryProvider(event_kind)
        session_id = f"provider-cursor-malformed-{event_kind}"
        await _stage_partial_operation(
            store,
            session_id=session_id,
            provider=provider,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

        stored = await store.load_events(session_id)
        assert [
            event.payload["delta"] for event in stored if event.type == EventType.MODEL_TEXT_DELTA
        ] == ["hel"]
        assert sum("provider_operation_progress" in event.payload for event in stored) == 1
        inspection = await inspect_provider_operation(store, session_id)
        assert inspection.recovery_reason is ProviderOperationUnavailableReason.MALFORMED
        assert inspection.duplicate_request_risk

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "identity_overrides",
    [
        {"interaction_id": "interaction-from-another-attempt"},
        {"attempt": 2, "max_attempts": 2},
    ],
)
def test_recovery_rejects_operation_identity_that_conflicts_with_model_started(
    identity_overrides: dict[str, object],
) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _CursorReplayProvider()
        session_id = "provider-operation-conflicting-owner-" + str(
            identity_overrides.get("attempt", "interaction")
        )
        await _stage_partial_operation(
            store,
            session_id=session_id,
            provider=provider,
            operation_identity_overrides=identity_overrides,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        with pytest.raises(ModelCompletionManualRecoveryRequired):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_before=datetime.now(UTC) + timedelta(seconds=1),
                )
            )

        assert provider.adapter.start_calls == 0
        assert provider.adapter.retrieve_calls == 0
        assert provider.adapter.reconnect_calls == []
        stored = await store.load_events(session_id)
        assert not any(
            event.type
            in {
                EventType.PROVIDER_OPERATION_RECONNECT_SCHEDULED,
                EventType.PROVIDER_OPERATION_RECONNECT_STARTED,
            }
            for event in stored
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "evidence_kind",
    ["model_started", "operation_started", "progress"],
)
def test_recovery_rejects_profile_evidence_that_conflicts_with_active_stage(
    store_kind: str,
    evidence_kind: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store: SessionStore = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / f"profile-conflict-{evidence_kind}.db")
        )
        provider = _CursorReplayProvider()
        conflicting_profile = "0" * 64
        stage, _identity, _state = await _stage_partial_operation(
            store,
            session_id=f"provider-profile-conflict-{store_kind}-{evidence_kind}",
            provider=provider,
            model_started_profile_fingerprint=(
                conflicting_profile if evidence_kind == "model_started" else _PROFILE_UNSET
            ),
            operation_profile_fingerprint=(
                conflicting_profile if evidence_kind == "operation_started" else _PROFILE_UNSET
            ),
            progress_profile_fingerprint=(
                conflicting_profile if evidence_kind == "progress" else _PROFILE_UNSET
            ),
        )

        with pytest.raises(ProviderOperationEvidenceError, match="execution profile"):
            await load_recoverable_provider_operation(store, stage)

        assert provider.adapter.start_calls == 0
        assert provider.adapter.retrieve_calls == 0
        assert provider.adapter.reconnect_calls == []

    asyncio.run(scenario())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "evidence_kind",
    ["model_started", "operation_started", "progress"],
)
def test_profiled_recovery_requires_profile_on_every_governed_event(
    store_kind: str,
    evidence_kind: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store: SessionStore = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / f"profile-missing-{evidence_kind}.db")
        )
        provider = _CursorReplayProvider()
        stage, _identity, _state = await _stage_partial_operation(
            store,
            session_id=f"provider-profile-missing-{store_kind}-{evidence_kind}",
            provider=provider,
            model_started_profile_fingerprint=(
                _PROFILE_MISSING if evidence_kind == "model_started" else _PROFILE_UNSET
            ),
            operation_profile_fingerprint=(
                _PROFILE_MISSING if evidence_kind == "operation_started" else _PROFILE_UNSET
            ),
            progress_profile_fingerprint=(
                _PROFILE_MISSING if evidence_kind == "progress" else _PROFILE_UNSET
            ),
        )

        with pytest.raises(ProviderOperationEvidenceError, match="no execution-profile evidence"):
            await load_recoverable_provider_operation(store, stage)

        assert provider.adapter.start_calls == 0
        assert provider.adapter.retrieve_calls == 0
        assert provider.adapter.reconnect_calls == []

    asyncio.run(scenario())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_genuine_legacy_stage_accepts_legacy_provider_operation_events(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store: SessionStore = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "profile-legacy.db")
        )
        provider = _CursorReplayProvider()
        stage, identity, _state = await _stage_partial_operation(
            store,
            session_id=f"provider-profile-legacy-{store_kind}",
            provider=provider,
            stage_profile_fingerprint=_PROFILE_MISSING,
            model_started_profile_fingerprint=_PROFILE_MISSING,
            operation_profile_fingerprint=_PROFILE_MISSING,
            progress_profile_fingerprint=_PROFILE_MISSING,
        )

        recovered = await load_recoverable_provider_operation(store, stage)

        assert recovered is not None
        assert recovered.model_attempt_identity == identity

    asyncio.run(scenario())


def test_start_only_recovery_rejects_profile_that_conflicts_with_active_stage() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _CursorReplayProvider()
        session_id = "provider-starting-profile-conflict"
        stage, identity, _state = await _stage_partial_operation(
            store,
            session_id=session_id,
            provider=provider,
            partial_events=(),
        )
        start_id = f"provider-operation:{identity.model_attempt_id}"
        start_stage = stage.model_copy(
            update={
                "intent": {
                    **stage.intent,
                    "provider_operation_start": {
                        "schema_version": 1,
                        "idempotency_support": (
                            ProviderOperationStartIdempotencySupport.EXACT.value
                        ),
                        "idempotency_key": start_id,
                    },
                }
            },
            deep=True,
        )
        await store.append_event(
            session_id,
            Event(
                type=EventType.PROVIDER_OPERATION_STARTING,
                session_id=session_id,
                interaction_id=f"interaction-{session_id}",
                agent_name="assistant",
                payload={
                    "provider": provider.name,
                    "model": "fake-model",
                    "step": 1,
                    "attempt": 1,
                    "max_attempts": 1,
                    **identity.payload(),
                    "source_run_epoch": stage.source_run_epoch,
                    "start_id": start_id,
                    "start_idempotency_support": (
                        ProviderOperationStartIdempotencySupport.EXACT.value
                    ),
                    "execution_profile_fingerprint": "0" * 64,
                },
            ),
        )

        with pytest.raises(ProviderOperationEvidenceError, match="execution profile"):
            await load_recoverable_provider_operation_start(store, start_stage)

        assert provider.adapter.start_calls == 0
        assert provider.adapter.retrieve_calls == 0
        assert provider.adapter.reconnect_calls == []

    asyncio.run(scenario())


def test_profiled_start_only_recovery_requires_profile_evidence() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _CursorReplayProvider()
        session_id = "provider-starting-profile-missing"
        stage, identity, _state = await _stage_partial_operation(
            store,
            session_id=session_id,
            provider=provider,
            partial_events=(),
        )
        start_id = f"provider-operation:{identity.model_attempt_id}"
        start_stage = stage.model_copy(
            update={
                "intent": {
                    **stage.intent,
                    "provider_operation_start": {
                        "schema_version": 1,
                        "idempotency_support": (
                            ProviderOperationStartIdempotencySupport.EXACT.value
                        ),
                        "idempotency_key": start_id,
                    },
                }
            },
            deep=True,
        )
        await store.append_event(
            session_id,
            Event(
                type=EventType.PROVIDER_OPERATION_STARTING,
                session_id=session_id,
                interaction_id=f"interaction-{session_id}",
                agent_name="assistant",
                payload={
                    "provider": provider.name,
                    "model": "fake-model",
                    "step": 1,
                    "attempt": 1,
                    "max_attempts": 1,
                    **identity.payload(),
                    "source_run_epoch": stage.source_run_epoch,
                    "start_id": start_id,
                    "start_idempotency_support": (
                        ProviderOperationStartIdempotencySupport.EXACT.value
                    ),
                },
            ),
        )

        with pytest.raises(ProviderOperationEvidenceError, match="no execution-profile evidence"):
            await load_recoverable_provider_operation_start(store, start_stage)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "untracked_event_type",
    [EventType.MODEL_TEXT_DELTA, EventType.PROVIDER_OPERATION_PROGRESS],
)
def test_recovery_rejects_ambiguous_output_after_valid_cursor_progress(
    untracked_event_type: EventType,
) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _CursorReplayProvider()
        session_id = "provider-cursor-followed-by-ambiguous-output"
        await _stage_partial_operation(
            store,
            session_id=session_id,
            provider=provider,
        )
        await store.append_events(
            session_id,
            [
                Event(
                    type=untracked_event_type,
                    session_id=session_id,
                    interaction_id=f"interaction-{session_id}",
                    agent_name="assistant",
                    payload={"delta": "untracked"},
                )
            ],
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        with pytest.raises(ModelCompletionManualRecoveryRequired):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_before=datetime.now(UTC) + timedelta(seconds=1),
                )
            )

        assert provider.adapter.start_calls == 0
        assert provider.adapter.reconnect_calls == []

    asyncio.run(scenario())


def test_discarded_attempt_cannot_claim_cursor_progress_authority() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _CursorReplayProvider()
        session_id = "provider-cursor-discarded-attempt"
        stage, identity, _state = await _stage_partial_operation(
            store,
            session_id=session_id,
            provider=provider,
        )
        await store.append_event(
            session_id,
            Event(
                type=EventType.MODEL_ATTEMPT_DISCARDED,
                session_id=session_id,
                interaction_id=f"interaction-{session_id}",
                agent_name="assistant",
                payload={
                    "provider": provider.name,
                    "model": "fake-model",
                    "step": 1,
                    "attempt": 1,
                    "max_attempts": 1,
                    **identity.payload(),
                    "provider_operation_progress": {},
                    "execution_profile_fingerprint": (_stage_execution_profile_fingerprint(stage)),
                },
            ),
        )

        with pytest.raises(ProviderOperationEvidenceError, match="unsafe"):
            await load_recoverable_provider_operation(store, stage)

    asyncio.run(scenario())


def test_profiled_recovery_requires_profile_on_matching_output_evidence() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _CursorReplayProvider()
        session_id = "provider-output-profile-missing"
        stage, identity, _state = await _stage_partial_operation(
            store,
            session_id=session_id,
            provider=provider,
        )
        await store.append_event(
            session_id,
            Event(
                type=EventType.MODEL_ATTEMPT_DISCARDED,
                session_id=session_id,
                interaction_id=f"interaction-{session_id}",
                agent_name="assistant",
                payload={
                    "provider": provider.name,
                    "model": "fake-model",
                    "step": 1,
                    "attempt": 1,
                    "max_attempts": 1,
                    **identity.payload(),
                    "provider_operation_progress": {},
                },
            ),
        )

        with pytest.raises(ProviderOperationEvidenceError, match="no execution-profile evidence"):
            await load_recoverable_provider_operation(store, stage)

    asyncio.run(scenario())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_provider_progress_commit_is_atomic_monotonic_and_replay_safe(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store: SessionStore = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "provider-progress.db")
        )
        provider = _CursorReplayProvider()
        session_id = f"provider-progress-{store_kind}"
        stage, identity, state = await _stage_partial_operation(
            store,
            session_id=session_id,
            provider=provider,
            release_fence=False,
        )
        first = ModelStreamEvent.text_delta(
            "hel",
            recovery_metadata={"cursor": 1, "opaque": {"token": "after-hel"}},
        )
        exact_replay = await commit_provider_operation_progress(
            store,
            stage=stage,
            model_attempt_identity=identity,
            current_state=provider.adapter.initial_state,
            stream_event=first,
            event=_progress_event(
                session_id=session_id,
                stage_id=stage.stage_id,
                identity=identity,
                state=provider.adapter.initial_state,
                stream_event=first,
                execution_profile_fingerprint=_stage_execution_profile_fingerprint(stage),
            ),
            expected_run_epoch=stage.source_run_epoch,
        )
        assert exact_replay.replayed is True

        conflicting = ModelStreamEvent.text_delta(
            "different",
            recovery_metadata={"cursor": 1, "opaque": {"token": "after-hel"}},
        )
        with pytest.raises(ProviderOperationEvidenceError, match="reused|regressed"):
            await commit_provider_operation_progress(
                store,
                stage=stage,
                model_attempt_identity=identity,
                current_state=state,
                stream_event=conflicting,
                event=_progress_event(
                    session_id=session_id,
                    stage_id=stage.stage_id,
                    identity=identity,
                    state=state,
                    stream_event=conflicting,
                    execution_profile_fingerprint=_stage_execution_profile_fingerprint(stage),
                ),
                expected_run_epoch=stage.source_run_epoch,
            )

        gap = ModelStreamEvent.text_delta(
            "gap",
            recovery_metadata={"cursor": 3, "opaque": {"token": "after-gap"}},
        )
        with pytest.raises(ProviderOperationEvidenceError, match="gap"):
            await commit_provider_operation_progress(
                store,
                stage=stage,
                model_attempt_identity=identity,
                current_state=state,
                stream_event=gap,
                event=_progress_event(
                    session_id=session_id,
                    stage_id=stage.stage_id,
                    identity=identity,
                    state=state,
                    stream_event=gap,
                    execution_profile_fingerprint=_stage_execution_profile_fingerprint(stage),
                ),
                expected_run_epoch=stage.source_run_epoch,
            )

        cross_attempt = ModelAttemptIdentity(
            model_step_id=identity.model_step_id,
            model_attempt_id="matt_" + "c" * 32,
        )
        next_event = ModelStreamEvent.text_delta(
            "lo",
            recovery_metadata={"cursor": 2, "opaque": {"token": "after-lo"}},
        )
        with pytest.raises(ProviderOperationEvidenceError, match="another operation"):
            await commit_provider_operation_progress(
                store,
                stage=stage,
                model_attempt_identity=cross_attempt,
                current_state=state,
                stream_event=next_event,
                event=_progress_event(
                    session_id=session_id,
                    stage_id=stage.stage_id,
                    identity=cross_attempt,
                    state=state,
                    stream_event=next_event,
                    execution_profile_fingerprint=_stage_execution_profile_fingerprint(stage),
                ),
                expected_run_epoch=stage.source_run_epoch,
            )

        competing = await asyncio.gather(
            *(
                commit_provider_operation_progress(
                    store,
                    stage=stage,
                    model_attempt_identity=identity,
                    current_state=state,
                    stream_event=next_event,
                    event=_progress_event(
                        session_id=session_id,
                        stage_id=stage.stage_id,
                        identity=identity,
                        state=state,
                        stream_event=next_event,
                        execution_profile_fingerprint=(_stage_execution_profile_fingerprint(stage)),
                    ),
                    expected_run_epoch=stage.source_run_epoch,
                )
                for _ in range(2)
            )
        )
        assert sorted(result.replayed for result in competing) == [False, True]
        advanced = next(result for result in competing if not result.replayed)
        assert advanced.state.recovery_metadata.cursor == 2

        old_boundary_replay = await commit_provider_operation_progress(
            store,
            stage=stage,
            model_attempt_identity=identity,
            current_state=provider.adapter.initial_state,
            stream_event=first,
            event=_progress_event(
                session_id=session_id,
                stage_id=stage.stage_id,
                identity=identity,
                state=provider.adapter.initial_state,
                stream_event=first,
                execution_profile_fingerprint=_stage_execution_profile_fingerprint(stage),
            ),
            expected_run_epoch=stage.source_run_epoch,
        )
        assert old_boundary_replay.replayed is True

        stale_advance = ModelStreamEvent.text_delta(
            "stale advance",
            recovery_metadata={"cursor": 3, "opaque": {"token": "stale-advance"}},
        )
        with pytest.raises(ProviderOperationEvidenceError, match="stale continuation"):
            await commit_provider_operation_progress(
                store,
                stage=stage,
                model_attempt_identity=identity,
                current_state=state,
                stream_event=stale_advance,
                event=_progress_event(
                    session_id=session_id,
                    stage_id=stage.stage_id,
                    identity=identity,
                    state=state,
                    stream_event=stale_advance,
                    execution_profile_fingerprint=_stage_execution_profile_fingerprint(stage),
                ),
                expected_run_epoch=stage.source_run_epoch,
            )

        text_events = [
            event
            for event in await store.load_events(session_id)
            if event.type == EventType.MODEL_TEXT_DELTA
        ]
        assert [event.payload["delta"] for event in text_events] == ["hel", "lo"]
        fenced = await store.fence_stalled_run(
            session_id,
            statuses={SessionStatus.RUNNING},
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )
        assert fenced is not None
        stale = ModelStreamEvent.text_delta(
            "stale",
            recovery_metadata={"cursor": 3, "opaque": {"token": "stale"}},
        )
        with pytest.raises(SessionRunFenced):
            await commit_provider_operation_progress(
                store,
                stage=stage,
                model_attempt_identity=identity,
                current_state=advanced.state,
                stream_event=stale,
                event=_progress_event(
                    session_id=session_id,
                    stage_id=stage.stage_id,
                    identity=identity,
                    state=advanced.state,
                    stream_event=stale,
                    execution_profile_fingerprint=_stage_execution_profile_fingerprint(stage),
                ),
                expected_run_epoch=stage.source_run_epoch,
            )
        assert [
            event.payload["delta"]
            for event in await store.load_events(session_id)
            if event.type == EventType.MODEL_TEXT_DELTA
        ] == ["hel", "lo"]
        await store.release_run_fence(session_id)

    asyncio.run(scenario())


def test_old_boundary_replay_remains_exact_while_competing_worker_advances() -> None:
    async def scenario() -> None:
        store = _ProgressPublishBarrierStore()
        provider = _CursorReplayProvider()
        session_id = "provider-progress-racing-old-replay"
        stage, identity, initial_state = await _stage_partial_operation(
            store,
            session_id=session_id,
            provider=provider,
            release_fence=False,
            partial_events=(),
        )

        def event(delta: str, cursor: int) -> ModelStreamEvent:
            return ModelStreamEvent.text_delta(
                delta,
                recovery_metadata={"cursor": cursor, "opaque": {"after": cursor}},
            )

        async def commit(
            stream_event: ModelStreamEvent,
            current_state: ProviderOperationState,
        ):
            return await commit_provider_operation_progress(
                store,
                stage=stage,
                model_attempt_identity=identity,
                current_state=current_state,
                stream_event=stream_event,
                event=_progress_event(
                    session_id=session_id,
                    stage_id=stage.stage_id,
                    identity=identity,
                    state=current_state,
                    stream_event=stream_event,
                    execution_profile_fingerprint=_stage_execution_profile_fingerprint(stage),
                ),
                expected_run_epoch=stage.source_run_epoch,
            )

        first_event = event("one", 1)
        store.pause_next_progress = True
        stale_replay = asyncio.create_task(commit(first_event, initial_state))
        await store.progress_publish_entered.wait()
        try:
            first = await commit(first_event, initial_state)
            second = await commit(event("two", 2), first.state)
        finally:
            store.progress_publish_release.set()
        replay = await stale_replay

        assert first.replayed is False
        assert second.replayed is False
        assert replay.replayed is True
        assert replay.state.recovery_metadata.cursor == 2
        assert [
            stored.payload["delta"]
            for stored in await store.load_events(session_id)
            if stored.type == EventType.MODEL_TEXT_DELTA
        ] == ["one", "two"]
        await store.release_run_fence(session_id)

    asyncio.run(scenario())


async def _stage_partial_operation(
    store: SessionStore,
    *,
    session_id: str,
    provider: _CursorReplayProvider,
    release_fence: bool = True,
    advances: int = 1,
    partial_events: tuple[ModelStreamEvent, ...] | None = None,
    thinking: ThinkingConfig | None = None,
    tools: tuple[Tool, ...] = (),
    operation_identity_overrides: dict[str, object] | None = None,
    stage_profile_fingerprint: object = _PROFILE_UNSET,
    model_started_profile_fingerprint: object = _PROFILE_UNSET,
    operation_profile_fingerprint: object = _PROFILE_UNSET,
    progress_profile_fingerprint: object = _PROFILE_UNSET,
) -> tuple[ModelCompletionStage, ModelAttemptIdentity, ProviderOperationState]:
    if advances not in {1, 2}:
        raise ValueError("advances must be 1 or 2")
    user_message = Message.text("user", "finish after reconnect")
    interaction_id = f"interaction-{session_id}"
    admitted = await create_admitted_session(
        store,
        request=RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[user_message],
        ),
        provider_name=provider.name,
        model="fake-model",
        tools=tools,
        provider=provider,
        interaction_id=interaction_id,
    )
    session = admitted.session
    execution_profile = admitted.active_invocation_profile.profile
    selected_stage_profile = (
        execution_profile.fingerprint
        if stage_profile_fingerprint is _PROFILE_UNSET
        else stage_profile_fingerprint
    )
    recovery_context = ModelCompletionRecoveryContext(
        execution_profile_fingerprint=execution_profile.fingerprint,
        thinking=thinking,
    ).model_dump(mode="json")
    if selected_stage_profile is _PROFILE_MISSING:
        recovery_context.pop("execution_profile_fingerprint")
    else:
        recovery_context["execution_profile_fingerprint"] = selected_stage_profile
    event_profile_default = execution_profile.fingerprint
    if type(selected_stage_profile) is str:
        event_profile_default = selected_stage_profile
    identity = ModelAttemptIdentity(
        model_step_id="mstep_" + "a" * 32,
        model_attempt_id="matt_" + "b" * 32,
    )
    stage_id = f"{identity.model_step_id}:dispatch:0"
    stage_result = await store.prepare_model_completion_stage(
        session_id,
        request=ModelCompletionStageRequest(
            stage_id=stage_id,
            logical_step_id=identity.model_step_id,
            dispatch_ordinal=0,
            intent={
                "schema_version": 1,
                "purpose": "assistant-turn",
                **identity.payload(),
                "logical_step_id": identity.model_step_id,
                "provider_name": provider.name,
                "requested_model": "fake-model",
                "source_transcript_cursor": 1,
                "request_fingerprint": "c" * 64,
                "recovery_context": recovery_context,
            },
        ),
        expected_statuses={session.status},
        expected_run_epoch=session.run_epoch,
        expected_transcript_cursor=1,
    )
    operation_event = Event(
        type=EventType.PROVIDER_OPERATION_STARTED,
        session_id=session_id,
        interaction_id=interaction_id,
        agent_name="assistant",
        payload={
            "provider": provider.name,
            "model": "fake-model",
            "step": 1,
            "attempt": 1,
            "max_attempts": 1,
            **identity.payload(),
            "source_run_epoch": session.run_epoch,
            "start_id": f"provider-operation:{identity.model_attempt_id}",
            "state_version": provider.adapter.initial_state.version,
            "operation_id": provider.adapter.initial_state.operation_id,
            "stream_protocol": provider.adapter.initial_state.stream_protocol,
            "status": ProviderOperationStatus.IN_PROGRESS.value,
            "recovery_metadata": provider.adapter.initial_state.recovery_metadata.model_dump(
                mode="json"
            ),
            **_profile_evidence_payload(
                operation_profile_fingerprint,
                default=event_profile_default,
            ),
        },
    )
    if operation_identity_overrides:
        operation_payload = dict(operation_event.payload)
        operation_payload.update(
            {
                key: value
                for key, value in operation_identity_overrides.items()
                if key != "interaction_id"
            }
        )
        operation_event = operation_event.model_copy(
            update={
                "interaction_id": operation_identity_overrides.get(
                    "interaction_id",
                    interaction_id,
                ),
                "payload": operation_payload,
            },
            deep=True,
        )
    await store.append_events(
        session_id,
        [
            Event(
                type=EventType.MODEL_STARTED,
                session_id=session_id,
                interaction_id=interaction_id,
                agent_name="assistant",
                payload={
                    "provider": provider.name,
                    "model": "fake-model",
                    "step": 1,
                    "attempt": 1,
                    "max_attempts": 1,
                    **identity.payload(),
                    **_profile_evidence_payload(
                        model_started_profile_fingerprint,
                        default=event_profile_default,
                    ),
                },
            ),
            operation_event,
        ],
    )
    if partial_events is None:
        partial_events = (
            ModelStreamEvent.text_delta(
                "hel",
                recovery_metadata={"cursor": 1, "opaque": {"token": "after-hel"}},
            ),
            *(
                (
                    ModelStreamEvent.text_delta(
                        "l",
                        recovery_metadata={"cursor": 2, "opaque": {"token": "after-l"}},
                    ),
                )
                if advances == 2
                else ()
            ),
        )
    current_state = provider.adapter.initial_state
    for accepted in partial_events:
        committed = await commit_provider_operation_progress(
            store,
            stage=stage_result.stage,
            model_attempt_identity=identity,
            current_state=current_state,
            stream_event=accepted,
            event=_progress_event(
                session_id=session_id,
                stage_id=stage_id,
                identity=identity,
                state=current_state,
                stream_event=accepted,
                identity_overrides=operation_identity_overrides,
                execution_profile_fingerprint=(
                    event_profile_default
                    if progress_profile_fingerprint is _PROFILE_UNSET
                    else progress_profile_fingerprint
                ),
            ),
            expected_run_epoch=session.run_epoch,
        )
        current_state = committed.state
    if release_fence:
        await store.release_run_fence(session_id)
    return stage_result.stage, identity, current_state


def _progress_event(
    *,
    session_id: str,
    stage_id: str,
    identity: ModelAttemptIdentity,
    state: ProviderOperationState,
    stream_event: ModelStreamEvent,
    identity_overrides: dict[str, object] | None = None,
    execution_profile_fingerprint: object,
) -> Event:
    metadata = stream_event.recovery_metadata
    assert metadata is not None and metadata.cursor is not None
    payload = {
        "step": 1,
        "attempt": 1,
        "max_attempts": 1,
        **identity.payload(),
        "provider_operation_progress": provider_operation_progress_payload(
            state,
            stream_event,
        ),
        **(
            {}
            if execution_profile_fingerprint is _PROFILE_MISSING
            else {"execution_profile_fingerprint": execution_profile_fingerprint}
        ),
    }
    if stream_event.type.value == "text_delta":
        event_type = EventType.MODEL_TEXT_DELTA
        payload["delta"] = stream_event.delta
    elif stream_event.type.value == "thinking" and stream_event.delta:
        event_type = EventType.MODEL_THINKING_DELTA
        payload["delta"] = stream_event.delta
    else:
        event_type = EventType.PROVIDER_OPERATION_PROGRESS
        payload.update(
            {
                "provider": "cursor-replay",
                "operation_id": state.operation_id,
                "stream_protocol": state.stream_protocol,
            }
        )
    if identity_overrides:
        payload.update(
            {key: value for key, value in identity_overrides.items() if key != "interaction_id"}
        )
    return Event(
        id=provider_operation_progress_event_id(stage_id, metadata.cursor),
        type=event_type,
        session_id=session_id,
        interaction_id=(
            identity_overrides.get("interaction_id", f"interaction-{session_id}")
            if identity_overrides
            else f"interaction-{session_id}"
        ),
        agent_name="assistant",
        payload=payload,
    )
