from __future__ import annotations

import asyncio
import gc
import weakref
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any

import pytest
from tests.core._execution_profile_fixtures import profiled_session_identity
from tests.provider_traceback_assertions import is_cayu_source_filename

from cayu import (
    REDACTED_SECRET,
    AgentSpec,
    AlwaysRequireApprovalToolPolicy,
    CayuApp,
    Event,
    EventQuery,
    EventType,
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAuthorityDecision,
    ExecutionProfileBehaviorIdentity,
    ExecutionProfilePolicy,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyRequest,
    ExecutionProfilePolicyResult,
    FileAttachment,
    FileAttachmentKind,
    FilePart,
    ForkExecutionProfileSelection,
    ForkSessionRequest,
    InMemorySessionStore,
    Message,
    MessageRole,
    ModelTarget,
    ProviderStatePart,
    ResolutionActor,
    ResolutionActorSource,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
    SecretRedactor,
    SessionIdentity,
    SessionStatus,
    SQLiteSessionStore,
    StructuredOutputSpec,
    TextPart,
    ThinkingPart,
    Tool,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolCallPart,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from cayu.providers import (
    AnthropicProvider,
    BedrockProvider,
    ChatCompletionsProvider,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    OpenAIProvider,
    OpenAISubscriptionProvider,
    VertexProvider,
)
from cayu.providers.base import _preflight_provider_portable_messages
from cayu.runtime import SessionStore
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime.approvals import PendingToolCallApproval


def _fork_profile_adoption(key: str) -> ExecutionProfileAdoptionIntent:
    return ExecutionProfileAdoptionIntent(
        idempotency_key=key,
        reason="Adopt the explicitly requested child model profile.",
        requested_by=ResolutionActor(
            subject="test",
            source=ResolutionActorSource.SYSTEM,
        ),
    )


class _AuthorizeForkProfilePolicy(ExecutionProfilePolicy):
    @property
    def identity(self) -> str:
        return "test:model-switch-fork-authority:v1"

    async def decide(
        self,
        request: ExecutionProfilePolicyRequest,
    ) -> ExecutionProfilePolicyResult:
        assert request.authority_review_required is True
        return ExecutionProfilePolicyResult(
            action=ExecutionProfilePolicyAction.ADOPT,
            reason="Authorize the model-switch fork.",
            authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
        )


class _NamedProvider(ModelProvider):
    def __init__(
        self,
        name: str,
        batches: list[list[ModelStreamEvent]],
        *,
        reject_portable_messages: bool = False,
    ) -> None:
        self.name = name
        self._batches = batches
        self.reject_portable_messages = reject_portable_messages
        self.requests: list[ModelRequest] = []
        self.preflight_model: str | None = None
        self.preflight_messages: list[Message] | None = None
        self.preflight_tools: list[dict[str, Any]] | None = None

    def preflight_portable_messages(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> None:
        _preflight_provider_portable_messages(
            model=model,
            messages=messages,
            tools=tools,
            supports_system_messages=True,
            supports_tool_history=True,
            supports_tool_definitions=True,
            supports_file_attachments=True,
        )
        self.preflight_model = model
        self.preflight_messages = [message.model_copy(deep=True) for message in messages]
        self.preflight_tools = deepcopy(tools)
        if self.reject_portable_messages:
            raise ValueError("Target provider cannot render the portable transcript.")

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        batch_index = len(self.requests) - 1
        if batch_index >= len(self._batches):
            raise AssertionError(f"No provider batch for request {batch_index}.")
        for event in self._batches[batch_index]:
            yield event


class _InheritedTextOnlyProvider(ModelProvider):
    name = "text-only"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if any(
            message.role not in {MessageRole.USER, MessageRole.ASSISTANT}
            or any(type(part) is not TextPart for part in message.content)
            for message in request.messages
        ):
            raise ValueError("Text-only provider cannot render this message role or content.")
        yield ModelStreamEvent.completed()


class _CapabilityProvider(ModelProvider):
    def __init__(
        self,
        *,
        name: str,
        supports_file_attachments: bool,
        supports_system_messages: bool = True,
    ) -> None:
        self.name = name
        self.supports_file_attachments = supports_file_attachments
        self.supports_system_messages = supports_system_messages
        self.requests: list[ModelRequest] = []

    def preflight_portable_messages(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> None:
        _preflight_provider_portable_messages(
            model=model,
            messages=messages,
            tools=tools,
            supports_system_messages=self.supports_system_messages,
            supports_tool_history=True,
            supports_tool_definitions=True,
            supports_file_attachments=self.supports_file_attachments,
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        yield ModelStreamEvent.completed()


class _RecordingOpenAIProvider(OpenAIProvider):
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        yield ModelStreamEvent.completed()


class _PendingRoundRaceStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.inject_pending_round = False

    async def transition_status_and_checkpoint(
        self,
        session_id,
        *,
        from_statuses,
        to_status,
        checkpoint_transform,
        interaction_started_event=None,
        interaction_source_messages=None,
        continued_interaction_id=None,
        defer_interaction_source=False,
        model_transition=None,
        execution_profile=None,
        execution_profile_decision=None,
    ):
        if self.inject_pending_round and model_transition is not None:
            self.inject_pending_round = False
            pending_round = tool_round_recovery.PendingToolRound(
                tool_round_id="tround_00000000000000000000000000000001",
                model_step_id="mstep_00000000000000000000000000000001",
                model_attempt_id="matt_00000000000000000000000000000001",
                agent_name="assistant",
                tool_calls=[
                    PendingToolCallApproval(
                        tool_call_id="raced-call",
                        tool_name="echo",
                        arguments={"value": "raced"},
                    )
                ],
            )
            checkpoint = await self.load_checkpoint(session_id)
            raced_checkpoint = dict(checkpoint or {})
            raced_checkpoint[tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY] = (
                pending_round.model_dump(mode="json")
            )
            await self.checkpoint(session_id, raced_checkpoint)
        return await super().transition_status_and_checkpoint(
            session_id,
            from_statuses=from_statuses,
            to_status=to_status,
            checkpoint_transform=checkpoint_transform,
            interaction_started_event=interaction_started_event,
            interaction_source_messages=interaction_source_messages,
            continued_interaction_id=continued_interaction_id,
            defer_interaction_source=defer_interaction_source,
            model_transition=model_transition,
            execution_profile=execution_profile,
            execution_profile_decision=execution_profile_decision,
        )


class _SnapshotTrackingStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.snapshot_refs: list[weakref.ReferenceType] = []

    async def load_transcript_snapshot(self, session_id):
        snapshot = await super().load_transcript_snapshot(session_id)
        self.snapshot_refs.append(weakref.ref(snapshot))
        return snapshot


class _CorruptingProjectionStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.corrupt_projection_cursor = False

    async def load(self, session_id: str):
        session = await super().load(session_id)
        if session is None or not self.corrupt_projection_cursor:
            return session
        metadata = session.model_copy(deep=True).metadata
        projection = metadata["cayu:model_target_projection"]
        assert type(projection) is dict
        projection["transcript_cursor"] = 10_000
        return session.model_copy(update={"metadata": metadata})


class _FailingToolRoundPublicationSQLiteStore(SQLiteSessionStore):
    def __init__(self, path) -> None:
        super().__init__(path)
        self.failed_tool_round_close_once = False

    async def publish_runtime_publication(
        self,
        session_id,
        *,
        request,
        expected_statuses=None,
        expected_run_epoch=None,
        expected_transcript_cursor=None,
    ):
        if not self.failed_tool_round_close_once and request.kind == "tool-round":
            self.failed_tool_round_close_once = True
            raise RuntimeError("ordinary tool round close unavailable")
        return await super().publish_runtime_publication(
            session_id,
            request=request,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )


class _EchoTool(Tool):
    def __init__(self, *, name: str = "echo") -> None:
        super().__init__(
            ToolSpec(
                name=name,
                description="Echo one value.",
                input_schema={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name=f"tests:session-model-switch:echo-tool:{name}",
                    behavior_version="1",
                    implementation_version="1",
                ),
            )
        )
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx
        self.calls.append(dict(args))
        return ToolResult(content=str(args["value"]))


def _profiled_source_identity(
    *,
    tool: Tool | None = None,
    require_approval: bool = False,
) -> SessionIdentity:
    direct_tool = tool or _EchoTool()
    return profiled_session_identity(
        provider_name="source",
        model="source-model",
        tools=[direct_tool],
        tool_policy=(AlwaysRequireApprovalToolPolicy() if require_approval else None),
    )


async def _collect(stream: AsyncIterator[Event]) -> list[Event]:
    return [event async for event in stream]


def _app(
    source: ModelProvider,
    target: ModelProvider,
    *,
    store: SessionStore | None = None,
    require_approval: bool = False,
    secret_redactor: SecretRedactor | None = None,
    tool: Tool | None = None,
    authorize_fork_profiles: bool = False,
) -> tuple[CayuApp, SessionStore]:
    session_store = store or InMemorySessionStore()
    app = CayuApp(
        session_store=session_store,
        secret_redactor=secret_redactor,
        execution_profile_policy=(
            _AuthorizeForkProfilePolicy() if authorize_fork_profiles else None
        ),
        enable_logging=False,
    )
    app.register_provider(source, default=True)
    app.register_provider(target)
    app.register_agent(
        AgentSpec(name="assistant", model="source-model"),
        tools=[tool or _EchoTool()],
        tool_policy=(AlwaysRequireApprovalToolPolicy() if require_approval else None),
    )
    return app, session_store


def test_cross_provider_resume_durably_projects_opaque_state() -> None:
    source = _NamedProvider(
        "source",
        [
            [
                ModelStreamEvent.thinking(
                    "private reasoning",
                    provider_state={"type": "thinking", "signature": "signed"},
                ),
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "provider_state": [
                            {
                                "provider": "source",
                                "state": {"type": "response_ref", "id": "response-1"},
                            }
                        ],
                    }
                ),
            ]
        ],
    )
    target = _NamedProvider(
        "target",
        [
            [
                ModelStreamEvent.thinking(
                    "target reasoning",
                    provider_state={"type": "thinking", "signature": "target-signed"},
                ),
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed(
                    {
                        "provider_state": [
                            {
                                "provider": "target",
                                "state": {"type": "response_ref", "id": "target-response"},
                            }
                        ]
                    }
                ),
            ],
            [ModelStreamEvent.text_delta("third answer"), ModelStreamEvent.completed()],
        ],
    )
    app, store = _app(source, target)

    asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="switch-opaque-state",
                    messages=[Message.text("user", "first")],
                )
            )
        )
    )
    events = asyncio.run(
        _collect(
            app.resume(
                ResumeRequest(
                    session_id="switch-opaque-state",
                    messages=[Message.text("user", "second")],
                    target=ModelTarget(provider_name="target", model="target-model"),
                )
            )
        )
    )

    assert events[0].type == EventType.SESSION_EXECUTION_PROFILE_DECIDED
    assert events[0].payload["decision"] == "adopted"
    assert events[1].type == EventType.SESSION_MODEL_SWITCHED
    assert events[1].payload == {
        "source_provider_name": "source",
        "source_model": "source-model",
        "target_provider_name": "target",
        "target_model": "target-model",
        "provider_changed": True,
        "model_changed": True,
        "provider_state_parts_dropped": 1,
        "thinking_parts_dropped": 1,
        "source_transcript_cursor": 2,
        "cache_state_dropped": True,
        "full_transcript_projection": True,
    }
    assert len(target.requests) == 1
    assert target.requests[0].model == "target-model"
    assert target.preflight_model == "target-model"
    assert target.preflight_messages is not None
    assert target.preflight_tools == target.requests[0].tools
    assert all(
        type(part) not in {ProviderStatePart, ThinkingPart}
        for message in target.requests[0].messages
        for part in message.content
    )

    session = asyncio.run(store.load("switch-opaque-state"))
    assert session is not None
    assert (session.provider_name, session.model) == ("target", "target-model")
    transcript = asyncio.run(store.load_transcript(session.id))
    assert any(
        type(part) in {ProviderStatePart, ThinkingPart}
        for message in transcript
        for part in message.content
    )
    assert session.metadata["cayu:model_target_projection"]["transcript_cursor"] == 2
    stored_switches = asyncio.run(
        store.query_events(
            EventQuery(session_id=session.id, event_type=EventType.SESSION_MODEL_SWITCHED)
        )
    )
    assert len(stored_switches) == 1
    assert stored_switches[0].event.payload == events[1].payload

    asyncio.run(
        _collect(
            app.resume(
                ResumeRequest(
                    session_id=session.id,
                    messages=[Message.text("user", "third")],
                )
            )
        )
    )
    assert len(target.requests) == 2
    assert all(
        type(part) not in {ProviderStatePart, ThinkingPart}
        for message in target.requests[1].messages[:2]
        for part in message.content
    )
    assert any(
        type(part) is ThinkingPart and part.text == "target reasoning"
        for message in target.requests[1].messages[2:]
        for part in message.content
    )
    assert any(
        type(part) is ProviderStatePart and part.provider == "target"
        for message in target.requests[1].messages[2:]
        for part in message.content
    )


def test_model_switch_releases_transcript_snapshots_before_streaming() -> None:
    async def run() -> None:
        store = _SnapshotTrackingStore()
        source = _NamedProvider(
            "source",
            [[ModelStreamEvent.text_delta("source answer"), ModelStreamEvent.completed()]],
        )
        target = _NamedProvider(
            "target",
            [[ModelStreamEvent.text_delta("target answer"), ModelStreamEvent.completed()]],
        )
        app, _ = _app(source, target, store=store)
        await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="switch-snapshot-lifetime",
                    messages=[Message.text("user", "first")],
                )
            )
        )

        stream = app.resume(
            ResumeRequest(
                session_id="switch-snapshot-lifetime",
                messages=[Message.text("user", "second")],
                target=ModelTarget(provider_name="target", model="target-model"),
            )
        )
        first_event = await stream.__anext__()
        assert first_event.type is EventType.SESSION_EXECUTION_PROFILE_DECIDED
        second_event = await stream.__anext__()
        assert second_event.type is EventType.SESSION_MODEL_SWITCHED
        gc.collect()
        assert store.snapshot_refs
        assert all(snapshot_ref() is None for snapshot_ref in store.snapshot_refs)
        assert (await _collect(stream))[-1].type is EventType.SESSION_COMPLETED

    asyncio.run(run())


def test_sqlite_restart_preserves_model_switch_projection_boundary(tmp_path) -> None:
    async def run() -> None:
        db_path = tmp_path / "model-switch-restart.sqlite"
        source = _NamedProvider(
            "source",
            [
                [
                    ModelStreamEvent.text_delta("source answer"),
                    ModelStreamEvent.completed(
                        {
                            "provider_state": [
                                {
                                    "provider": "source",
                                    "state": {"type": "response_ref", "id": "source-state"},
                                }
                            ]
                        }
                    ),
                ]
            ],
        )
        first_target = _NamedProvider(
            "target",
            [[ModelStreamEvent.text_delta("target answer"), ModelStreamEvent.completed()]],
        )
        store = SQLiteSessionStore(db_path)
        app, _ = _app(source, first_target, store=store)
        await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="switch-restart",
                    messages=[Message.text("user", "first")],
                )
            )
        )
        await _collect(
            app.resume(
                ResumeRequest(
                    session_id="switch-restart",
                    messages=[Message.text("user", "second")],
                    target=ModelTarget(provider_name="target", model="target-model"),
                )
            )
        )
        await store.close()

        reopened = SQLiteSessionStore(db_path)
        resumed_target = _NamedProvider(
            "target",
            [[ModelStreamEvent.text_delta("after restart"), ModelStreamEvent.completed()]],
        )
        restarted_app, _ = _app(
            _NamedProvider("source", []),
            resumed_target,
            store=reopened,
        )
        try:
            await _collect(
                restarted_app.resume(
                    ResumeRequest(
                        session_id="switch-restart",
                        messages=[Message.text("user", "third")],
                    )
                )
            )
            assert len(resumed_target.requests) == 1
            assert all(
                not (type(part) is ProviderStatePart and part.provider == "source")
                for message in resumed_target.requests[0].messages
                for part in message.content
            )
        finally:
            await reopened.close()

    asyncio.run(run())


def test_sqlite_transcript_retention_cannot_invalidate_admitted_model_switch(
    tmp_path,
) -> None:
    async def run() -> None:
        store = SQLiteSessionStore(tmp_path / "model-switch-retention.sqlite")
        source = _NamedProvider("source", [])
        target = _NamedProvider(
            "target",
            [[ModelStreamEvent.text_delta("target answer"), ModelStreamEvent.completed()]],
        )
        app, _ = _app(source, target, store=store)
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="switch-retention",
                messages=[],
            ),
            identity=_profiled_source_identity(),
        )
        await store.append_transcript_messages(
            "switch-retention",
            [
                Message.text("user", "first"),
                Message.text("assistant", "source answer"),
            ],
        )
        await store.update_status("switch-retention", SessionStatus.COMPLETED)

        stream = app.resume(
            ResumeRequest(
                session_id="switch-retention",
                messages=[Message.text("user", "switch")],
                target=ModelTarget(provider_name="target", model="target-model"),
            )
        )
        try:
            assert (await anext(stream)).type is EventType.SESSION_EXECUTION_PROFILE_DECIDED
            assert (await anext(stream)).type is EventType.SESSION_MODEL_SWITCHED
            assert (await anext(stream)).type is EventType.INTERACTION_STARTED

            assert await store.compact_transcript("switch-retention", keep_last=1) == 0

            remaining = await _collect(stream)
            assert remaining[-1].type is EventType.SESSION_COMPLETED
            assert len(target.requests) == 1
            assert [
                message.content[0].text
                for message in await store.load_transcript("switch-retention")
                if message.role.value in {"user", "assistant"}
            ] == ["first", "source answer", "switch", "target answer"]

        finally:
            await stream.aclose()
            await store.close()

    asyncio.run(run())


def test_sqlite_model_switch_uses_absolute_cursor_after_prior_retention(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteSessionStore(tmp_path / "model-switch-after-retention.sqlite")
        target = _NamedProvider(
            "target",
            [[ModelStreamEvent.text_delta("target answer"), ModelStreamEvent.completed()]],
        )
        app, _ = _app(_NamedProvider("source", []), target, store=store)
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="switch-after-retention",
                messages=[],
            ),
            identity=_profiled_source_identity(),
        )
        await store.append_transcript_messages(
            "switch-after-retention",
            [Message.text("user", f"m{index}") for index in range(5)],
        )
        await store.update_status("switch-after-retention", SessionStatus.COMPLETED)
        assert await store.compact_transcript("switch-after-retention", keep_last=2) == 3

        events = await _collect(
            app.resume(
                ResumeRequest(
                    session_id="switch-after-retention",
                    messages=[Message.text("user", "switch")],
                    target=ModelTarget(provider_name="target", model="target-model"),
                )
            )
        )

        switch_event = next(
            event for event in events if event.type is EventType.SESSION_MODEL_SWITCHED
        )
        assert switch_event.payload["source_transcript_cursor"] == 5
        session = await store.load("switch-after-retention")
        assert session is not None
        assert session.metadata["cayu:model_target_projection"]["transcript_cursor"] == 5
        assert len(target.requests) == 1
        assert [message.content[0].text for message in target.requests[0].messages] == [
            "m3",
            "m4",
            "switch",
        ]
        snapshot = await store.load_transcript_snapshot("switch-after-retention")
        assert snapshot.cursor == 7
        assert await store.load_transcript_cursor("switch-after-retention") == 7
        assert [record.index for record in snapshot.records] == [3, 4, 5, 6]
        await store.close()

    asyncio.run(run())


def test_sqlite_approval_continuation_uses_absolute_cursor_after_prior_retention(
    tmp_path,
) -> None:
    async def run() -> None:
        store = SQLiteSessionStore(tmp_path / "model-switch-retained-approval.sqlite")
        tool = _EchoTool()
        target = _NamedProvider(
            "target",
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call-retained",
                        name="echo",
                        arguments={"value": "retained"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ],
        )
        app, _ = _app(
            _NamedProvider("source", []),
            target,
            store=store,
            require_approval=True,
            tool=tool,
        )
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="retained-approval",
                messages=[],
            ),
            identity=_profiled_source_identity(tool=tool, require_approval=True),
        )
        await store.append_transcript_messages(
            "retained-approval",
            [Message.text("user", f"m{index}") for index in range(5)],
        )
        await store.update_status("retained-approval", SessionStatus.COMPLETED)
        assert await store.compact_transcript("retained-approval", keep_last=2) == 3

        first_events = await _collect(
            app.resume(
                ResumeRequest(
                    session_id="retained-approval",
                    messages=[Message.text("user", "switch and call")],
                    target=ModelTarget(provider_name="target", model="target-model"),
                )
            )
        )
        assert first_events[-1].type is EventType.SESSION_INTERRUPTED
        approval_event = next(
            event for event in first_events if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        approval = approval_event.payload["approval"]

        resolution_events = await _collect(
            app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="retained-approval",
                    approval_id=approval["approval_id"],
                    tool_round_id=approval_event.payload["tool_round_id"],
                    tool_call_id=approval_event.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        )

        assert resolution_events[-1].type is EventType.SESSION_COMPLETED
        assert tool.calls == [{"value": "retained"}]
        assert len(target.requests) == 2
        session = await store.load("retained-approval")
        assert session is not None
        assert session.status is SessionStatus.COMPLETED
        await store.close()

    asyncio.run(run())


def test_sqlite_pending_tool_recovery_uses_absolute_cursor_after_prior_retention(
    tmp_path,
) -> None:
    async def run() -> None:
        store = _FailingToolRoundPublicationSQLiteStore(
            tmp_path / "model-switch-retained-tool-recovery.sqlite"
        )
        tool = _EchoTool()
        target = _NamedProvider(
            "target",
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call-retained-recovery",
                        name="echo",
                        arguments={"value": "retained"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("recovered"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ],
        )
        app, _ = _app(
            _NamedProvider("source", []),
            target,
            store=store,
            tool=tool,
        )
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="retained-tool-recovery",
                messages=[],
            ),
            identity=_profiled_source_identity(tool=tool),
        )
        await store.append_transcript_messages(
            "retained-tool-recovery",
            [Message.text("user", f"m{index}") for index in range(5)],
        )
        await store.update_status("retained-tool-recovery", SessionStatus.COMPLETED)
        assert await store.compact_transcript("retained-tool-recovery", keep_last=2) == 3

        first_events = await _collect(
            app.resume(
                ResumeRequest(
                    session_id="retained-tool-recovery",
                    messages=[Message.text("user", "switch and call")],
                    target=ModelTarget(provider_name="target", model="target-model"),
                )
            )
        )
        assert first_events[-1].type is EventType.SESSION_FAILED
        checkpoint = await store.load_checkpoint("retained-tool-recovery")
        assert checkpoint is not None and "pending_tool_round" in checkpoint
        assert tool.calls == [{"value": "retained"}]

        with pytest.raises(
            RuntimeError,
            match="execution profile cannot be adopted while model or tool recovery is pending",
        ):
            await _collect(
                app.resume(
                    ResumeRequest(
                        session_id="retained-tool-recovery",
                        messages=[Message.text("user", "unsafe adoption")],
                        profile_adoption=ExecutionProfileAdoptionIntent(
                            idempotency_key="pending-recovery-adoption-v1",
                            reason="Attempt adoption during recovery.",
                            requested_by=ResolutionActor(
                                subject="maintainer",
                                source=ResolutionActorSource.REQUEST,
                            ),
                        ),
                    )
                )
            )
        assert "pending_tool_round" in (await store.load_checkpoint("retained-tool-recovery") or {})
        assert len(target.requests) == 1

        recovery_events = await _collect(
            app.resume(
                ResumeRequest(
                    session_id="retained-tool-recovery",
                    messages=[Message.text("user", "continue")],
                )
            )
        )

        assert recovery_events[-1].type is EventType.SESSION_COMPLETED
        assert tool.calls == [{"value": "retained"}]
        assert len(target.requests) == 2
        assert "pending_tool_round" not in (
            await store.load_checkpoint("retained-tool-recovery") or {}
        )
        await store.close()

    asyncio.run(run())


def test_switching_again_advances_projection_and_drops_all_earlier_native_state() -> None:
    source = _NamedProvider(
        "source",
        [
            [
                ModelStreamEvent.text_delta("source answer"),
                ModelStreamEvent.completed(
                    {
                        "provider_state": [
                            {
                                "provider": "source",
                                "state": {"type": "response_ref", "id": "source-state"},
                            }
                        ]
                    }
                ),
            ],
            [ModelStreamEvent.text_delta("source again"), ModelStreamEvent.completed()],
        ],
    )
    target = _NamedProvider(
        "target",
        [
            [
                ModelStreamEvent.text_delta("target answer"),
                ModelStreamEvent.completed(
                    {
                        "provider_state": [
                            {
                                "provider": "target",
                                "state": {"type": "response_ref", "id": "target-state"},
                            }
                        ]
                    }
                ),
            ]
        ],
    )
    app, store = _app(source, target)
    asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="switch-twice",
                    messages=[Message.text("user", "first")],
                )
            )
        )
    )
    asyncio.run(
        _collect(
            app.resume(
                ResumeRequest(
                    session_id="switch-twice",
                    messages=[Message.text("user", "second")],
                    target=ModelTarget(provider_name="target", model="target-model"),
                )
            )
        )
    )

    events = asyncio.run(
        _collect(
            app.resume(
                ResumeRequest(
                    session_id="switch-twice",
                    messages=[Message.text("user", "third")],
                    target=ModelTarget(provider_name="source", model="source-model"),
                )
            )
        )
    )

    assert events[0].type is EventType.SESSION_EXECUTION_PROFILE_DECIDED
    switch_event = events[1]
    assert switch_event.type is EventType.SESSION_MODEL_SWITCHED
    assert switch_event.payload["provider_state_parts_dropped"] == 2
    assert switch_event.payload["source_transcript_cursor"] == 4
    assert len(source.requests) == 2
    assert all(
        type(part) is not ProviderStatePart
        for message in source.requests[1].messages[:4]
        for part in message.content
    )
    session = asyncio.run(store.load("switch-twice"))
    assert session is not None
    assert session.metadata["cayu:model_target_projection"]["transcript_cursor"] == 4


def test_cross_provider_resume_keeps_complete_neutral_tool_history() -> None:
    source = _NamedProvider(
        "source",
        [
            [
                ModelStreamEvent.tool_call(
                    id="call-1",
                    name="echo",
                    arguments={"value": "portable"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("tool finished"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ],
    )
    target = _NamedProvider(
        "target",
        [[ModelStreamEvent.text_delta("continued"), ModelStreamEvent.completed()]],
    )
    app, _store = _app(source, target)

    asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="switch-tool-history",
                    messages=[Message.text("user", "use the tool")],
                )
            )
        )
    )
    events = asyncio.run(
        _collect(
            app.resume(
                ResumeRequest(
                    session_id="switch-tool-history",
                    messages=[Message.text("user", "continue")],
                    target=ModelTarget(provider_name="target", model="target-model"),
                )
            )
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    roles = [message.role.value for message in target.requests[0].messages]
    assert roles == ["user", "assistant", "tool", "assistant", "user"]


def test_cross_provider_resume_drops_reasoning_only_assistant_shell() -> None:
    async def run() -> None:
        source = _NamedProvider("source", [])
        target = _NamedProvider(
            "target",
            [[ModelStreamEvent.text_delta("continued"), ModelStreamEvent.completed()]],
        )
        app, store = _app(source, target)
        neutral_tool_call = Message.tool_call(
            tool_call_id="portable-call",
            tool_name="echo",
            arguments={"value": "portable"},
        )
        neutral_tool_result = Message.tool_result(
            tool_call_id="portable-call",
            tool_name="echo",
            content="portable",
        )
        durable_prefix = [
            Message.text("user", "opening"),
            Message(
                role=MessageRole.ASSISTANT,
                content=(ThinkingPart(text="source-only reasoning"),),
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content=(
                    TextPart(text="portable answer"),
                    ProviderStatePart(
                        provider="source",
                        state={"type": "response_ref", "id": "source-state"},
                    ),
                ),
            ),
            Message.text("user", "use the tool"),
            neutral_tool_call,
            neutral_tool_result,
            Message.text("assistant", "tool complete"),
        ]
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="switch-reasoning-only-shell",
                messages=[],
            ),
            identity=_profiled_source_identity(),
        )
        await store.append_transcript_messages(
            "switch-reasoning-only-shell",
            durable_prefix,
        )
        await store.update_status(
            "switch-reasoning-only-shell",
            SessionStatus.COMPLETED,
        )
        durable_before = await store.load_transcript("switch-reasoning-only-shell")

        events = await _collect(
            app.resume(
                ResumeRequest(
                    session_id="switch-reasoning-only-shell",
                    messages=[Message.text("user", "continue")],
                    target=ModelTarget(provider_name="target", model="target-model"),
                )
            )
        )

        expected_request = [
            Message.text("user", "opening"),
            Message.text("assistant", "portable answer"),
            Message.text("user", "use the tool"),
            neutral_tool_call,
            neutral_tool_result,
            Message.text("assistant", "tool complete"),
            Message.text("user", "continue"),
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert target.preflight_messages == expected_request
        assert len(target.requests) == 1
        assert target.requests[0].messages == expected_request
        durable_after = await store.load_transcript("switch-reasoning-only-shell")
        assert durable_after[: len(durable_before)] == durable_before

    asyncio.run(run())


def test_reasoning_only_shell_does_not_break_pending_tool_round_recovery(tmp_path) -> None:
    async def run() -> None:
        store = _FailingToolRoundPublicationSQLiteStore(
            tmp_path / "model-switch-reasoning-shell-recovery.sqlite"
        )
        tool = _EchoTool()
        target = _NamedProvider(
            "target",
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call-after-shell",
                        name="echo",
                        arguments={"value": "portable"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("recovered"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ],
        )
        app, _ = _app(
            _NamedProvider("source", []),
            target,
            store=store,
            tool=tool,
        )
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="switch-reasoning-shell-recovery",
                messages=[],
            ),
            identity=_profiled_source_identity(tool=tool),
        )
        durable_prefix = [
            Message.text("user", "opening"),
            Message(
                role=MessageRole.ASSISTANT,
                content=(ThinkingPart(text="source-only reasoning"),),
            ),
            Message.text("assistant", "portable answer"),
        ]
        await store.append_transcript_messages(
            "switch-reasoning-shell-recovery",
            durable_prefix,
        )
        await store.update_status(
            "switch-reasoning-shell-recovery",
            SessionStatus.COMPLETED,
        )

        first_events = await _collect(
            app.resume(
                ResumeRequest(
                    session_id="switch-reasoning-shell-recovery",
                    messages=[Message.text("user", "switch and call")],
                    target=ModelTarget(provider_name="target", model="target-model"),
                )
            )
        )
        assert first_events[-1].type is EventType.SESSION_FAILED
        assert tool.calls == [{"value": "portable"}]
        checkpoint = await store.load_checkpoint("switch-reasoning-shell-recovery")
        assert checkpoint is not None and "pending_tool_round" in checkpoint

        recovery_events = await _collect(
            app.resume(
                ResumeRequest(
                    session_id="switch-reasoning-shell-recovery",
                    messages=[Message.text("user", "continue")],
                )
            )
        )

        assert recovery_events[-1].type is EventType.SESSION_COMPLETED
        assert tool.calls == [{"value": "portable"}]
        assert len(target.requests) == 2
        assert all(
            type(part) not in {ProviderStatePart, ThinkingPart}
            for message in target.requests[1].messages[:2]
            for part in message.content
        )
        assert "pending_tool_round" not in (
            await store.load_checkpoint("switch-reasoning-shell-recovery") or {}
        )
        durable_after = await store.load_transcript("switch-reasoning-shell-recovery")
        assert durable_after[: len(durable_prefix)] == durable_prefix
        await store.close()

    asyncio.run(run())


def test_post_switch_tool_round_cannot_restore_source_provider_state() -> None:
    source = _NamedProvider(
        "source",
        [
            [
                ModelStreamEvent.text_delta("source answer"),
                ModelStreamEvent.completed(
                    {
                        "provider_state": [
                            {
                                "provider": "source",
                                "state": {"type": "response_ref", "id": "source-state"},
                            }
                        ]
                    }
                ),
            ]
        ],
    )
    target = _NamedProvider(
        "target",
        [
            [
                ModelStreamEvent.tool_call(
                    id="target-call",
                    name="echo",
                    arguments={"value": "portable"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.text_delta("target finished"), ModelStreamEvent.completed()],
        ],
    )
    app, _store = _app(source, target)
    asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="switch-then-tool",
                    messages=[Message.text("user", "first")],
                )
            )
        )
    )

    events = asyncio.run(
        _collect(
            app.resume(
                ResumeRequest(
                    session_id="switch-then-tool",
                    messages=[Message.text("user", "use the target tool")],
                    target=ModelTarget(provider_name="target", model="target-model"),
                )
            )
        )
    )

    assert events[-1].type is EventType.SESSION_COMPLETED
    assert len(target.requests) == 2
    assert all(
        not (type(part) is ProviderStatePart and part.provider == "source")
        for request in target.requests
        for message in request.messages
        for part in message.content
    )


def test_model_switch_rejects_pending_tool_approval_without_mutation() -> None:
    source = _NamedProvider(
        "source",
        [
            [
                ModelStreamEvent.tool_call(id="call-1", name="echo", arguments={"value": "x"}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ],
    )
    target = _NamedProvider("target", [])
    app, store = _app(source, target, require_approval=True)
    initial_events = asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="switch-pending-approval",
                    messages=[Message.text("user", "use the tool")],
                )
            )
        )
    )
    assert initial_events[-1].type == EventType.SESSION_INTERRUPTED

    with pytest.raises(RuntimeError, match="pending tool approval"):
        asyncio.run(
            _collect(
                app.resume(
                    ResumeRequest(
                        session_id="switch-pending-approval",
                        messages=[Message.text("user", "continue")],
                        target=ModelTarget(provider_name="target", model="target-model"),
                    )
                )
            )
        )

    session = asyncio.run(store.load("switch-pending-approval"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert (session.provider_name, session.model) == ("source", "source-model")
    assert target.requests == []


def test_model_switch_atomically_rejects_a_concurrent_pending_tool_round() -> None:
    async def run() -> None:
        store = _PendingRoundRaceStore()
        source = _NamedProvider(
            "source",
            [
                [
                    ModelStreamEvent.text_delta("source answer"),
                    ModelStreamEvent.completed({"finish_reason": "stop", "end_turn": False}),
                ]
            ],
        )
        target = _NamedProvider("target", [])
        app, _ = _app(source, target, store=store)
        initial_events = await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="switch-pending-round-race",
                    messages=[Message.text("user", "source history")],
                    max_steps=1,
                )
            )
        )
        assert initial_events[-1].type is EventType.SESSION_INTERRUPTED
        session_before = await store.load("switch-pending-round-race")
        transcript_before = await store.load_transcript("switch-pending-round-race")
        store.inject_pending_round = True

        with pytest.raises(RuntimeError, match="tool-round recovery is pending"):
            await _collect(
                app.resume(
                    ResumeRequest(
                        session_id="switch-pending-round-race",
                        messages=[Message.text("user", "switch")],
                        target=ModelTarget(
                            provider_name="target",
                            model="target-model",
                        ),
                    )
                )
            )

        session_after = await store.load("switch-pending-round-race")
        assert session_before is not None
        assert session_after is not None
        assert session_after.status is session_before.status
        assert session_after.run_epoch == session_before.run_epoch
        assert (session_after.provider_name, session_after.model) == (
            session_before.provider_name,
            session_before.model,
        )
        assert session_after.metadata == session_before.metadata
        assert await store.load_transcript("switch-pending-round-race") == transcript_before
        checkpoint = await store.load_checkpoint("switch-pending-round-race")
        assert tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint) is not None
        switch_events = await store.query_events(
            EventQuery(
                session_id="switch-pending-round-race",
                event_type=EventType.SESSION_MODEL_SWITCHED,
            )
        )
        assert switch_events == []
        assert target.requests == []

    asyncio.run(run())


def test_model_switch_rejects_unmatched_tool_history_without_mutation() -> None:
    source = _NamedProvider("source", [])
    target = _NamedProvider("target", [])
    app, store = _app(source, target)

    async def seed_unmatched_history() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="switch-unmatched-tool",
                messages=[Message.text("user", "first")],
            ),
            identity=_profiled_source_identity(),
        )
        await store.append_transcript_messages(
            "switch-unmatched-tool",
            [
                Message.text("user", "first"),
                Message(
                    role="assistant",
                    content=(
                        ToolCallPart(
                            tool_call_id="call-unmatched",
                            tool_name="echo",
                            arguments={"value": "x"},
                        ),
                    ),
                ),
            ],
        )
        await store.update_status("switch-unmatched-tool", SessionStatus.COMPLETED)

    asyncio.run(seed_unmatched_history())

    with pytest.raises(ValueError, match="matching tool results"):
        asyncio.run(
            _collect(
                app.resume(
                    ResumeRequest(
                        session_id="switch-unmatched-tool",
                        messages=[Message.text("user", "continue")],
                        target=ModelTarget(provider_name="target", model="target-model"),
                    )
                )
            )
        )

    session = asyncio.run(store.load("switch-unmatched-tool"))
    assert session is not None
    assert session.status is SessionStatus.COMPLETED
    assert (session.provider_name, session.model) == ("source", "source-model")
    assert target.preflight_messages is None
    assert target.requests == []


def test_model_switch_preflight_failure_leaves_session_unchanged() -> None:
    source = _NamedProvider(
        "source",
        [[ModelStreamEvent.text_delta("answer"), ModelStreamEvent.completed()]],
    )
    target = _NamedProvider("target", [], reject_portable_messages=True)
    app, store = _app(source, target)
    asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="switch-preflight-rejected",
                    messages=[Message.text("user", "first")],
                )
            )
        )
    )
    before = asyncio.run(store.load("switch-preflight-rejected"))
    before_transcript = asyncio.run(store.load_transcript("switch-preflight-rejected"))

    with pytest.raises(ValueError, match="cannot render"):
        asyncio.run(
            _collect(
                app.resume(
                    ResumeRequest(
                        session_id="switch-preflight-rejected",
                        messages=[Message.text("user", "second")],
                        target=ModelTarget(provider_name="target", model="target-model"),
                    )
                )
            )
        )

    assert asyncio.run(store.load("switch-preflight-rejected")) == before
    assert asyncio.run(store.load_transcript("switch-preflight-rejected")) == before_transcript
    assert target.requests == []


def test_inherited_portability_preflight_rejects_tool_history_before_adoption() -> None:
    source = _NamedProvider(
        "source",
        [
            [
                ModelStreamEvent.tool_call(
                    id="call-1",
                    name="echo",
                    arguments={"value": "first"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.text_delta("answer"), ModelStreamEvent.completed()],
        ],
    )
    target = _InheritedTextOnlyProvider()
    app, store = _app(source, target)
    asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="switch-default-preflight",
                    messages=[Message.text("user", "use the tool")],
                )
            )
        )
    )
    before = asyncio.run(store.load("switch-default-preflight"))
    before_transcript = asyncio.run(store.load_transcript("switch-default-preflight"))

    with pytest.raises(ValueError, match="does not declare portable tool-history support"):
        asyncio.run(
            _collect(
                app.resume(
                    ResumeRequest(
                        session_id="switch-default-preflight",
                        messages=[Message.text("user", "switch")],
                        target=ModelTarget(
                            provider_name="text-only",
                            model="text-model",
                        ),
                    )
                )
            )
        )

    assert target.requests == []
    assert asyncio.run(store.load("switch-default-preflight")) == before
    assert asyncio.run(store.load_transcript("switch-default-preflight")) == before_transcript


def test_inherited_portability_preflight_rejects_system_role_before_adoption() -> None:
    store = InMemorySessionStore()
    source = _NamedProvider(
        "source",
        [[ModelStreamEvent.text_delta("answer"), ModelStreamEvent.completed()]],
    )
    target = _InheritedTextOnlyProvider()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(source, default=True)
    app.register_provider(target)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="source-model",
            system_prompt="System instruction visible to every model request.",
        ),
        tools=[],
    )
    asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="switch-default-system-preflight",
                    messages=[Message.text("user", "first")],
                )
            )
        )
    )
    before = asyncio.run(store.load("switch-default-system-preflight"))
    before_transcript = asyncio.run(store.load_transcript("switch-default-system-preflight"))
    before_events = asyncio.run(
        store.query_events(EventQuery(session_id="switch-default-system-preflight"))
    )

    with pytest.raises(ValueError, match="does not declare system-message support"):
        asyncio.run(
            _collect(
                app.resume(
                    ResumeRequest(
                        session_id="switch-default-system-preflight",
                        messages=[Message.text("user", "switch")],
                        target=ModelTarget(
                            provider_name="text-only",
                            model="text-model",
                        ),
                    )
                )
            )
        )

    assert target.requests == []
    assert asyncio.run(store.load("switch-default-system-preflight")) == before
    assert (
        asyncio.run(store.load_transcript("switch-default-system-preflight")) == before_transcript
    )
    assert (
        asyncio.run(store.query_events(EventQuery(session_id="switch-default-system-preflight")))
        == before_events
    )


def test_model_switch_preflights_generated_structured_output_instruction() -> None:
    source = _NamedProvider(
        "source",
        [[ModelStreamEvent.text_delta("answer"), ModelStreamEvent.completed()]],
    )
    target = _CapabilityProvider(
        name="target",
        supports_file_attachments=True,
        supports_system_messages=False,
    )
    app, store = _app(source, target)
    asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="switch-structured-instruction-preflight",
                    messages=[Message.text("user", "first")],
                )
            )
        )
    )
    before = asyncio.run(store.load("switch-structured-instruction-preflight"))
    before_transcript = asyncio.run(
        store.load_transcript("switch-structured-instruction-preflight")
    )
    before_events = asyncio.run(
        store.query_events(EventQuery(session_id="switch-structured-instruction-preflight"))
    )

    with pytest.raises(ValueError, match="does not declare system-message support"):
        asyncio.run(
            _collect(
                app.resume(
                    ResumeRequest(
                        session_id="switch-structured-instruction-preflight",
                        messages=[Message.text("user", "switch")],
                        target=ModelTarget(provider_name="target", model="target-model"),
                        structured_output=StructuredOutputSpec(
                            json_schema={"type": "object"},
                        ),
                    )
                )
            )
        )

    assert target.requests == []
    assert asyncio.run(store.load("switch-structured-instruction-preflight")) == before
    assert (
        asyncio.run(store.load_transcript("switch-structured-instruction-preflight"))
        == before_transcript
    )
    assert (
        asyncio.run(
            store.query_events(EventQuery(session_id="switch-structured-instruction-preflight"))
        )
        == before_events
    )


def test_builtin_preflight_rejects_invalid_tool_history_before_adoption() -> None:
    invalid_tool_name = "invalid tool name"
    source = _NamedProvider(
        "source",
        [
            [
                ModelStreamEvent.tool_call(
                    id="call-1",
                    name=invalid_tool_name,
                    arguments={"value": "first"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.text_delta("answer"), ModelStreamEvent.completed()],
        ],
    )
    target = _RecordingOpenAIProvider()
    app, store = _app(source, target, tool=_EchoTool(name=invalid_tool_name))
    asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="switch-adapter-tool-name-preflight",
                    messages=[Message.text("user", "use the tool")],
                )
            )
        )
    )
    before = asyncio.run(store.load("switch-adapter-tool-name-preflight"))
    before_transcript = asyncio.run(store.load_transcript("switch-adapter-tool-name-preflight"))

    with pytest.raises(ValueError, match="OpenAI tool names"):
        asyncio.run(
            _collect(
                app.resume(
                    ResumeRequest(
                        session_id="switch-adapter-tool-name-preflight",
                        messages=[Message.text("user", "switch")],
                        target=ModelTarget(provider_name="openai", model="gpt-test"),
                    )
                )
            )
        )

    assert target.requests == []
    assert asyncio.run(store.load("switch-adapter-tool-name-preflight")) == before
    assert (
        asyncio.run(store.load_transcript("switch-adapter-tool-name-preflight"))
        == before_transcript
    )
    assert (
        asyncio.run(
            store.query_events(
                EventQuery(
                    session_id="switch-adapter-tool-name-preflight",
                    event_type=EventType.SESSION_MODEL_SWITCHED,
                )
            )
        )
        == []
    )


def test_builtin_preflight_rejects_incompatible_uninvoked_tool_before_adoption() -> None:
    invalid_tool_name = "invalid tool name"
    source = _NamedProvider(
        "source",
        [[ModelStreamEvent.text_delta("answer"), ModelStreamEvent.completed()]],
    )
    target = _RecordingOpenAIProvider()
    app, store = _app(source, target, tool=_EchoTool(name=invalid_tool_name))
    asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="switch-active-tool-name-preflight",
                    messages=[Message.text("user", "answer without using the tool")],
                )
            )
        )
    )
    before = asyncio.run(store.load("switch-active-tool-name-preflight"))
    before_transcript = asyncio.run(store.load_transcript("switch-active-tool-name-preflight"))
    before_events = asyncio.run(
        store.query_events(EventQuery(session_id="switch-active-tool-name-preflight"))
    )

    with pytest.raises(ValueError, match="OpenAI tool names"):
        asyncio.run(
            _collect(
                app.resume(
                    ResumeRequest(
                        session_id="switch-active-tool-name-preflight",
                        messages=[Message.text("user", "switch")],
                        target=ModelTarget(provider_name="openai", model="gpt-test"),
                    )
                )
            )
        )

    assert target.requests == []
    assert asyncio.run(store.load("switch-active-tool-name-preflight")) == before
    assert (
        asyncio.run(store.load_transcript("switch-active-tool-name-preflight")) == before_transcript
    )
    assert (
        asyncio.run(store.query_events(EventQuery(session_id="switch-active-tool-name-preflight")))
        == before_events
    )


@pytest.mark.parametrize(
    ("artifact", "supports_file_attachments", "error_pattern"),
    [
        (
            FileAttachment(
                artifact_id="artifact-1",
                kind=FileAttachmentKind.IMAGE,
                filename="image.png",
                content_type="image/png",
                size_bytes=1,
            ).model_dump(mode="json"),
            False,
            "does not declare portable file-attachment support",
        ),
        (
            {
                "type": "cayu.file_attachment.v1",
                "artifact_id": "artifact-1",
            },
            True,
            "claims an invalid file attachment",
        ),
    ],
)
def test_model_switch_rejects_nested_tool_result_attachments_before_adoption(
    artifact: dict,
    supports_file_attachments: bool,
    error_pattern: str,
) -> None:
    async def run() -> None:
        target = _CapabilityProvider(
            name="capability-target",
            supports_file_attachments=supports_file_attachments,
        )
        source = _NamedProvider(
            "source",
            [[ModelStreamEvent.text_delta("initial answer"), ModelStreamEvent.completed()]],
        )
        app, store = _app(source, target)
        session_id = "switch-nested-tool-result-attachment"
        await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "initial request")],
                )
            )
        )
        await store.append_transcript_messages(
            session_id,
            [
                Message.text("user", "inspect the tool output"),
                Message.tool_call(
                    tool_call_id="call-1",
                    tool_name="echo",
                    arguments={"value": "x"},
                ),
                Message.tool_result(
                    tool_call_id="call-1",
                    tool_name="echo",
                    content="x",
                    artifacts=[artifact],
                ),
            ],
        )
        before = await store.load(session_id)
        before_transcript = await store.load_transcript(session_id)
        before_events = await store.query_events(EventQuery(session_id=session_id))

        with pytest.raises(ValueError, match=error_pattern):
            await _collect(
                app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "switch")],
                        target=ModelTarget(
                            provider_name="capability-target",
                            model="target-model",
                        ),
                    )
                )
            )

        assert target.requests == []
        assert await store.load(session_id) == before
        assert await store.load_transcript(session_id) == before_transcript
        assert await store.query_events(EventQuery(session_id=session_id)) == before_events

    asyncio.run(run())


def test_portability_preflight_keeps_non_file_tool_result_artifacts_portable() -> None:
    messages = [
        Message.tool_call(
            tool_call_id="call-1",
            tool_name="echo",
            arguments={},
        ),
        Message.tool_result(
            tool_call_id="call-1",
            tool_name="echo",
            content="x",
            artifacts=[{"type": "example.custom-artifact.v1", "value": "x"}],
        ),
    ]

    _preflight_provider_portable_messages(
        model="portable-model",
        messages=messages,
        tools=[],
        supports_system_messages=True,
        supports_tool_history=True,
        supports_tool_definitions=True,
        supports_file_attachments=False,
    )


@pytest.mark.parametrize(
    "provider_type",
    [
        AnthropicProvider,
        BedrockProvider,
        ChatCompletionsProvider,
        OpenAIProvider,
        OpenAISubscriptionProvider,
        ScriptedModelProvider,
        VertexProvider,
    ],
)
def test_builtin_providers_explicitly_admit_neutral_tools_and_files(provider_type) -> None:
    attachment = FileAttachment(
        artifact_id="artifact-1",
        kind=FileAttachmentKind.IMAGE,
        filename="image.png",
        content_type="image/png",
        size_bytes=1,
    )
    messages = [
        Message.text("system", "System instruction."),
        Message(
            role=MessageRole.USER,
            content=(FilePart(attachment=attachment.model_dump(mode="json")),),
        ),
        Message.tool_call(
            tool_call_id="call-1",
            tool_name="echo",
            arguments={"value": "x"},
        ),
        Message.tool_result(
            tool_call_id="call-1",
            tool_name="echo",
            content="x",
        ),
    ]
    provider = provider_type.__new__(provider_type)
    if provider_type is ChatCompletionsProvider:
        provider.clean_schemas = True
        provider.strip_additional_properties = False

    provider.preflight_portable_messages(
        model="portable-model",
        messages=messages,
        tools=[
            {
                "name": "echo",
                "description": "Echo one value.",
                "input_schema": {"type": "object"},
            }
        ],
    )


@pytest.mark.parametrize(
    "provider_type",
    [
        AnthropicProvider,
        BedrockProvider,
        ChatCompletionsProvider,
        OpenAIProvider,
        OpenAISubscriptionProvider,
        VertexProvider,
    ],
)
def test_builtin_providers_reject_adapter_invalid_portable_tool_names(provider_type) -> None:
    messages = [
        Message.tool_call(
            tool_call_id="call-1",
            tool_name="invalid tool name",
            arguments={},
        ),
        Message.tool_result(
            tool_call_id="call-1",
            tool_name="invalid tool name",
            content="x",
        ),
    ]
    provider = provider_type.__new__(provider_type)
    if provider_type is ChatCompletionsProvider:
        provider.clean_schemas = True
        provider.strip_additional_properties = False

    with pytest.raises(ValueError, match="tool names"):
        provider.preflight_portable_messages(
            model="portable-model",
            messages=messages,
            tools=[],
        )


def test_model_switch_preflight_receives_workload_redacted_projection() -> None:
    async def run() -> None:
        secret = "model-switch-preflight-secret-canary"
        store = InMemorySessionStore()
        target = _NamedProvider(
            "target",
            [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]],
        )
        app, _ = _app(
            _NamedProvider("source", []),
            target,
            store=store,
            secret_redactor=SecretRedactor(secret),
        )
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="switch-redacted-preflight",
                messages=[],
            ),
            identity=_profiled_source_identity(),
        )
        await store.append_transcript_messages(
            "switch-redacted-preflight",
            [Message.text("user", f"durable text {secret}")],
        )
        await store.update_status("switch-redacted-preflight", SessionStatus.COMPLETED)

        events = await _collect(
            app.resume(
                ResumeRequest(
                    session_id="switch-redacted-preflight",
                    messages=[Message.text("user", "switch")],
                    target=ModelTarget(provider_name="target", model="target-model"),
                )
            )
        )

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert target.preflight_messages is not None
        assert target.preflight_messages[0].content[0].text == (f"durable text {REDACTED_SECRET}")
        assert target.requests[0].messages[0].content[0].text == (f"durable text {REDACTED_SECRET}")

    asyncio.run(run())


def test_model_switch_preflight_failure_does_not_retain_raw_workload_secret() -> None:
    async def run() -> None:
        secret = "model-switch-preflight-traceback-secret"
        store = InMemorySessionStore()
        target = _NamedProvider("target", [], reject_portable_messages=True)
        app, _ = _app(
            _NamedProvider("source", []),
            target,
            store=store,
            secret_redactor=SecretRedactor(secret),
        )
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="switch-preflight-traceback-redaction",
                messages=[],
            ),
            identity=_profiled_source_identity(),
        )
        await store.append_transcript_messages(
            "switch-preflight-traceback-redaction",
            [Message.text("user", f"durable text {secret}")],
        )
        await store.update_status(
            "switch-preflight-traceback-redaction",
            SessionStatus.COMPLETED,
        )

        with pytest.raises(ValueError, match="cannot render") as exc_info:
            await _collect(
                app.resume(
                    ResumeRequest(
                        session_id="switch-preflight-traceback-redaction",
                        messages=[Message.text("user", "switch")],
                        target=ModelTarget(
                            provider_name="target",
                            model="target-model",
                        ),
                    )
                )
            )

        traceback = exc_info.value.__traceback__
        while traceback is not None:
            if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
                assert secret not in repr(traceback.tb_frame.f_locals)
            traceback = traceback.tb_next

    asyncio.run(run())


def test_model_switch_rejects_provider_state_in_new_resume_messages() -> None:
    source = _NamedProvider(
        "source",
        [[ModelStreamEvent.text_delta("answer"), ModelStreamEvent.completed()]],
    )
    target = _NamedProvider("target", [])
    app, store = _app(source, target)
    asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="switch-new-opaque-input",
                    messages=[Message.text("user", "first")],
                )
            )
        )
    )
    before = asyncio.run(store.load("switch-new-opaque-input"))

    with pytest.raises(ValueError, match="cannot contain provider state"):
        asyncio.run(
            _collect(
                app.resume(
                    ResumeRequest(
                        session_id="switch-new-opaque-input",
                        messages=[
                            Message(
                                role="assistant",
                                content=(
                                    TextPart(text="caller-supplied assistant state"),
                                    ProviderStatePart(
                                        provider="source",
                                        state={"type": "response_ref", "id": "untrusted"},
                                    ),
                                ),
                            )
                        ],
                        target=ModelTarget(provider_name="target", model="target-model"),
                    )
                )
            )
        )

    assert asyncio.run(store.load("switch-new-opaque-input")) == before
    assert target.preflight_messages is None
    assert target.requests == []


def test_later_resume_cannot_append_opaque_state_beyond_switch_cursor() -> None:
    source = _NamedProvider(
        "source",
        [[ModelStreamEvent.text_delta("source answer"), ModelStreamEvent.completed()]],
    )
    target = _NamedProvider(
        "target",
        [[ModelStreamEvent.text_delta("target answer"), ModelStreamEvent.completed()]],
    )
    app, store = _app(source, target)
    asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="switch-later-opaque-input",
                    messages=[Message.text("user", "first")],
                )
            )
        )
    )
    asyncio.run(
        _collect(
            app.resume(
                ResumeRequest(
                    session_id="switch-later-opaque-input",
                    messages=[Message.text("user", "switch")],
                    target=ModelTarget(provider_name="target", model="target-model"),
                )
            )
        )
    )
    session_before = asyncio.run(store.load("switch-later-opaque-input"))
    transcript_before = asyncio.run(store.load_transcript("switch-later-opaque-input"))

    with pytest.raises(ValueError, match="cannot contain provider state"):
        asyncio.run(
            _collect(
                app.resume(
                    ResumeRequest(
                        session_id="switch-later-opaque-input",
                        messages=[
                            Message(
                                role="assistant",
                                content=(
                                    TextPart(text="forged continuation"),
                                    ProviderStatePart(
                                        provider="source",
                                        state={"type": "response_ref", "id": "forged"},
                                    ),
                                ),
                            )
                        ],
                    )
                )
            )
        )

    assert asyncio.run(store.load("switch-later-opaque-input")) == session_before
    assert asyncio.run(store.load_transcript("switch-later-opaque-input")) == transcript_before
    assert len(target.requests) == 1


def test_fork_inherits_the_source_model_projection_boundary() -> None:
    source = _NamedProvider(
        "source",
        [
            [
                ModelStreamEvent.text_delta("source answer"),
                ModelStreamEvent.completed(
                    {
                        "provider_state": [
                            {
                                "provider": "source",
                                "state": {"type": "response_ref", "id": "source-state"},
                            }
                        ]
                    }
                ),
            ]
        ],
    )
    target = _NamedProvider(
        "target",
        [
            [ModelStreamEvent.text_delta("target answer"), ModelStreamEvent.completed()],
            [ModelStreamEvent.text_delta("fork answer"), ModelStreamEvent.completed()],
        ],
    )
    app, store = _app(source, target)
    asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="switched-fork-source",
                    messages=[Message.text("user", "first")],
                )
            )
        )
    )
    asyncio.run(
        _collect(
            app.resume(
                ResumeRequest(
                    session_id="switched-fork-source",
                    messages=[Message.text("user", "second")],
                    target=ModelTarget(provider_name="target", model="target-model"),
                )
            )
        )
    )

    asyncio.run(
        _collect(
            app.fork_session(
                ForkSessionRequest(
                    source_session_id="switched-fork-source",
                    session_id="switched-fork-child",
                )
            )
        )
    )
    source_session = asyncio.run(store.load("switched-fork-source"))
    child_session = asyncio.run(store.load("switched-fork-child"))
    assert source_session is not None
    assert child_session is not None
    assert (
        child_session.metadata["cayu:model_target_projection"]
        == source_session.metadata["cayu:model_target_projection"]
    )

    asyncio.run(
        _collect(
            app.resume(
                ResumeRequest(
                    session_id="switched-fork-child",
                    messages=[Message.text("user", "continue the fork")],
                )
            )
        )
    )
    assert len(target.requests) == 2
    assert all(
        not (type(part) is ProviderStatePart and part.provider == "source")
        for message in target.requests[1].messages
        for part in message.content
    )


def test_same_provider_model_override_fork_preflights_and_projects_transcript() -> None:
    source = _NamedProvider(
        "source",
        [
            [
                ModelStreamEvent.thinking(
                    "model-specific reasoning",
                    provider_state={"type": "thinking", "signature": "signed"},
                ),
                ModelStreamEvent.text_delta("source answer"),
                ModelStreamEvent.completed(
                    {
                        "provider_state": [
                            {
                                "provider": "source",
                                "state": {"type": "response_ref", "id": "source-state"},
                            }
                        ]
                    }
                ),
            ],
            [ModelStreamEvent.text_delta("fork answer"), ModelStreamEvent.completed()],
        ],
    )
    app, store = _app(
        source,
        _NamedProvider("target", []),
        authorize_fork_profiles=True,
    )
    asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="model-override-fork-source",
                    messages=[Message.text("user", "first")],
                )
            )
        )
    )

    asyncio.run(
        _collect(
            app.fork_session(
                ForkSessionRequest(
                    source_session_id="model-override-fork-source",
                    session_id="model-override-fork-child",
                    model="second-model",
                    execution_profile_selection=ForkExecutionProfileSelection.CURRENT_CHILD,
                    profile_adoption=_fork_profile_adoption("model-override-fork"),
                )
            )
        )
    )
    child = asyncio.run(store.load("model-override-fork-child"))
    assert child is not None
    assert child.model == "second-model"
    assert source.preflight_model == "second-model"
    assert child.metadata["cayu:model_target_projection"]["transcript_cursor"] == 2
    assert source.preflight_messages is not None
    assert all(
        type(part) not in {ProviderStatePart, ThinkingPart}
        for message in source.preflight_messages
        for part in message.content
    )

    asyncio.run(
        _collect(
            app.resume(
                ResumeRequest(
                    session_id="model-override-fork-child",
                    messages=[Message.text("user", "continue")],
                )
            )
        )
    )
    assert source.requests[1].model == "second-model"
    assert all(
        type(part) not in {ProviderStatePart, ThinkingPart}
        for message in source.requests[1].messages[:2]
        for part in message.content
    )


def test_same_provider_model_override_fork_accepts_an_empty_retained_prefix() -> None:
    async def run() -> None:
        source = _NamedProvider("source", [])
        app, store = _app(
            source,
            _NamedProvider("target", []),
            authorize_fork_profiles=True,
        )
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="empty-model-fork-source",
                messages=[],
            ),
            identity=_profiled_source_identity(),
        )
        await store.update_status("empty-model-fork-source", SessionStatus.COMPLETED)

        events = await _collect(
            app.fork_session(
                ForkSessionRequest(
                    source_session_id="empty-model-fork-source",
                    session_id="empty-model-fork-child",
                    model="second-model",
                    execution_profile_selection=ForkExecutionProfileSelection.CURRENT_CHILD,
                    profile_adoption=_fork_profile_adoption("empty-model-fork"),
                )
            )
        )

        assert [event.type for event in events] == [EventType.SESSION_FORKED]
        child = await store.load("empty-model-fork-child")
        assert child is not None
        assert child.model == "second-model"
        assert source.preflight_model == "second-model"
        assert source.preflight_messages == []

    asyncio.run(run())


def test_sqlite_model_override_partial_fork_translates_absolute_retained_cursor(
    tmp_path,
) -> None:
    async def run() -> None:
        store = SQLiteSessionStore(tmp_path / "model-switch-retained-fork.sqlite")
        source = _NamedProvider(
            "source",
            [[ModelStreamEvent.text_delta("fork answer"), ModelStreamEvent.completed()]],
        )
        app, _ = _app(
            source,
            _NamedProvider("target", []),
            store=store,
            authorize_fork_profiles=True,
        )
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="model-override-retained-source",
                messages=[],
            ),
            identity=_profiled_source_identity(),
        )
        await store.append_transcript_messages(
            "model-override-retained-source",
            [Message.text("user", f"m{index}") for index in range(5)],
        )
        await store.update_status("model-override-retained-source", SessionStatus.COMPLETED)
        assert await store.compact_transcript("model-override-retained-source", keep_last=2) == 3

        await _collect(
            app.fork_session(
                ForkSessionRequest(
                    source_session_id="model-override-retained-source",
                    session_id="model-override-retained-child",
                    transcript_cursor=4,
                    copy_checkpoint=False,
                    model="second-model",
                    execution_profile_selection=ForkExecutionProfileSelection.CURRENT_CHILD,
                    profile_adoption=_fork_profile_adoption("retained-model-fork"),
                )
            )
        )

        child = await store.load("model-override-retained-child")
        assert child is not None
        assert child.metadata["cayu:model_target_projection"]["transcript_cursor"] == 1
        assert [
            message.content[0].text
            for message in await store.load_transcript("model-override-retained-child")
        ] == ["m3"]
        assert source.preflight_messages is not None
        assert [message.content[0].text for message in source.preflight_messages] == ["m3"]

        await _collect(
            app.resume(
                ResumeRequest(
                    session_id="model-override-retained-child",
                    messages=[Message.text("user", "continue")],
                )
            )
        )
        assert source.requests[0].model == "second-model"
        assert [message.content[0].text for message in source.requests[0].messages] == [
            "m3",
            "continue",
        ]
        await store.close()

    asyncio.run(run())


def test_model_override_fork_preflight_failure_does_not_create_child() -> None:
    source = _NamedProvider(
        "source",
        [[ModelStreamEvent.text_delta("source answer"), ModelStreamEvent.completed()]],
        reject_portable_messages=True,
    )
    app, store = _app(
        source,
        _NamedProvider("target", []),
        authorize_fork_profiles=True,
    )
    asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="rejected-model-fork-source",
                    messages=[Message.text("user", "first")],
                )
            )
        )
    )

    with pytest.raises(ValueError, match="cannot render"):
        asyncio.run(
            _collect(
                app.fork_session(
                    ForkSessionRequest(
                        source_session_id="rejected-model-fork-source",
                        session_id="rejected-model-fork-child",
                        model="second-model",
                        execution_profile_selection=ForkExecutionProfileSelection.CURRENT_CHILD,
                        profile_adoption=_fork_profile_adoption("rejected-model-fork"),
                    )
                )
            )
        )

    assert asyncio.run(store.load("rejected-model-fork-child")) is None


def test_model_override_fork_checks_workload_secrets_before_provider_preflight() -> None:
    secret = "fork-model-switch-secret-canary"
    source = _NamedProvider(
        "source",
        [[ModelStreamEvent.text_delta("source answer"), ModelStreamEvent.completed()]],
    )
    app, store = _app(
        source,
        _NamedProvider("target", []),
        secret_redactor=SecretRedactor(secret),
        authorize_fork_profiles=True,
    )
    asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="secret-model-fork-source",
                    messages=[Message.text("user", "first")],
                )
            )
        )
    )
    asyncio.run(
        store.append_transcript_messages(
            "secret-model-fork-source",
            [Message.text("user", f"durable corruption: {secret}")],
        )
    )

    with pytest.raises(ValueError, match="workload secret") as exc_info:
        asyncio.run(
            _collect(
                app.fork_session(
                    ForkSessionRequest(
                        source_session_id="secret-model-fork-source",
                        session_id="secret-model-fork-child",
                        model="second-model",
                        execution_profile_selection=ForkExecutionProfileSelection.CURRENT_CHILD,
                        profile_adoption=_fork_profile_adoption("secret-model-fork"),
                    )
                )
            )
        )

    traceback = exc_info.value.__traceback__
    while traceback is not None:
        if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
            assert secret not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next

    assert source.preflight_messages is None
    assert asyncio.run(store.load("secret-model-fork-child")) is None


def test_fork_rejects_caller_supplied_model_projection_authority() -> None:
    source = _NamedProvider(
        "source",
        [[ModelStreamEvent.text_delta("source answer"), ModelStreamEvent.completed()]],
    )
    app, store = _app(source, _NamedProvider("target", []))
    asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="forged-model-projection-source",
                    messages=[Message.text("user", "first")],
                )
            )
        )
    )

    with pytest.raises(ValueError, match="runtime-owned model-target authority"):
        asyncio.run(
            _collect(
                app.fork_session(
                    ForkSessionRequest(
                        source_session_id="forged-model-projection-source",
                        session_id="forged-model-projection-child",
                        metadata={
                            "cayu:model_target_projection": {
                                "record_type": "cayu.model-target-projection",
                                "schema_version": 1,
                                "provider_name": "target",
                                "model": "forged",
                                "transcript_cursor": 1,
                            }
                        },
                    )
                )
            )
        )

    assert asyncio.run(store.load("forged-model-projection-child")) is None


def test_run_rejects_caller_supplied_model_projection_authority() -> None:
    with pytest.raises(ValueError, match="runtime-owned model-target authority"):
        RunRequest(
            agent_name="assistant",
            session_id="forged-model-projection-run",
            messages=[Message.text("user", "first")],
            metadata={
                "cayu:model_target_projection": {
                    "record_type": "cayu.model-target-projection",
                    "schema_version": 1,
                    "provider_name": "target",
                    "model": "forged",
                    "transcript_cursor": 1,
                }
            },
        )


def test_resume_rejects_corrupt_projection_cursor_before_mutation() -> None:
    source = _NamedProvider(
        "source",
        [[ModelStreamEvent.text_delta("source answer"), ModelStreamEvent.completed()]],
    )
    target = _NamedProvider(
        "target",
        [[ModelStreamEvent.text_delta("target answer"), ModelStreamEvent.completed()]],
    )
    store = _CorruptingProjectionStore()
    app, _ = _app(source, target, store=store)
    asyncio.run(
        _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="corrupt-projection-cursor",
                    messages=[Message.text("user", "first")],
                )
            )
        )
    )
    asyncio.run(
        _collect(
            app.resume(
                ResumeRequest(
                    session_id="corrupt-projection-cursor",
                    target=ModelTarget(provider_name="target", model="target-model"),
                    messages=[Message.text("user", "switch")],
                )
            )
        )
    )
    session_before = asyncio.run(store.load("corrupt-projection-cursor"))
    transcript_before = asyncio.run(store.load_transcript("corrupt-projection-cursor"))
    assert session_before is not None

    store.corrupt_projection_cursor = True
    with pytest.raises(ValueError, match="cursor exceeds"):
        asyncio.run(
            _collect(
                app.resume(
                    ResumeRequest(
                        session_id="corrupt-projection-cursor",
                        messages=[Message.text("user", "must not be admitted")],
                    )
                )
            )
        )
    store.corrupt_projection_cursor = False

    assert asyncio.run(store.load("corrupt-projection-cursor")) == session_before
    assert asyncio.run(store.load_transcript("corrupt-projection-cursor")) == transcript_before
    assert len(target.requests) == 1
