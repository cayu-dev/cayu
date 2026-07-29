from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from cayu.core import AgentSpec, Event, EventType, Message
from cayu.core.messages import ToolCallPart, ToolResultPart
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    InteractionStatus,
    InteractionSummaryEvidence,
    ResumeRequest,
    RunLimits,
    RunRequest,
    Session,
    SessionIdentity,
    SessionStatus,
    StructuredOutputSpec,
)
from cayu.runtime import _model_completion_publication as model_completion_publication
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _structured_output_tool_round as structured_output_tool_round
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime.execution_units import ModelAttemptIdentity, ToolRoundIdentity
from cayu.runtime.sessions import (
    ModelCompletionStageRequest,
    RuntimePublicationReceipt,
    RuntimePublicationRequest,
    RuntimePublicationResult,
    runtime_publication_checkpoint_mutation,
)
from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME
from cayu.vaults import REDACTED_SECRET, SecretRedactor


class _RecordingProvider(ModelProvider):
    name = "structured-output-recovery"

    def __init__(self, responses: list[list[ModelStreamEvent]] | None = None) -> None:
        self._responses = [] if responses is None else responses
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        response_index = len(self.requests)
        self.requests.append(request)
        if response_index >= len(self._responses):
            raise AssertionError("Recovery unexpectedly dispatched the model provider.")
        for event in self._responses[response_index]:
            yield event


class _SideEffectTool(Tool):
    spec = ToolSpec(
        name="side_effect",
        description="Record a side effect.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx
        self.calls.append(args)
        return ToolResult(content="side effect executed")


class _ToolPublicationAcknowledgementLostStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.tool_publication_calls = 0
        self.lost_acknowledgement = False

    async def publish_runtime_publication(
        self,
        session_id: str,
        *,
        request: RuntimePublicationRequest,
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> RuntimePublicationResult:
        if request.kind == "tool-round":
            self.tool_publication_calls += 1
        result = await super().publish_runtime_publication(
            session_id,
            request=request,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )
        if request.kind == "tool-round" and not self.lost_acknowledgement and not result.replayed:
            self.lost_acknowledgement = True
            raise ConnectionError("structured-output publication acknowledgement lost")
        return result


@dataclass(frozen=True)
class _PublishedStructuredStep:
    session: Session
    user_message: Message
    assistant_message: Message
    pending_round: tool_round_recovery.PendingToolRound
    completion_event: Event
    model_intent: dict
    model_receipt: RuntimePublicationReceipt


def _answer_spec(*, max_retries: int = 2) -> StructuredOutputSpec:
    return StructuredOutputSpec(
        name="answer",
        json_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
        max_retries=max_retries,
    )


def _structured_call(*, call_id: str, output: dict) -> runtime_records.ToolCallRequest:
    return runtime_records.ToolCallRequest(
        id=call_id,
        name=STRUCTURED_OUTPUT_TOOL_NAME,
        arguments={"output": output},
    )


def _register_runtime(
    store: InMemorySessionStore,
    provider: _RecordingProvider,
    *,
    tools: list[Tool] | None = None,
    max_parallel_tool_calls: int = 4,
    secret_redactor: SecretRedactor | None = None,
) -> CayuApp:
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        max_parallel_tool_calls=max_parallel_tool_calls,
        secret_redactor=secret_redactor,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[] if tools is None else tools,
    )
    return app


async def _publish_structured_model_step(
    store: InMemorySessionStore,
    *,
    session_id: str,
    provider_name: str,
    spec: StructuredOutputSpec,
    tool_calls: list[runtime_records.ToolCallRequest],
    redactor: SecretRedactor | None = None,
) -> _PublishedStructuredStep:
    user_message = Message.text("user", "produce the final structured answer")
    interaction_id = f"interaction-{session_id}"
    started_event_id = f"{session_id}:interaction-started"
    started_at = datetime.now(UTC)
    started_event = Event(
        id=started_event_id,
        type=EventType.INTERACTION_STARTED,
        session_id=session_id,
        interaction_id=interaction_id,
        timestamp=started_at,
        agent_name="assistant",
        payload=InteractionSummaryEvidence(
            status=InteractionStatus.ACTIVE,
            start_event_id=started_event_id,
            started_at=started_at,
        ).model_dump(mode="json"),
    )
    running = await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[user_message],
            structured_output=spec,
        ),
        identity=SessionIdentity(
            provider_name=provider_name,
            model="fake-model",
        ),
        interaction_started_event=started_event,
        interaction_source_messages=[user_message],
    )
    await store.replace_initial_transcript_messages(
        session_id,
        [user_message],
        [user_message],
        interaction_id=interaction_id,
    )

    source_transcript_cursor = 1
    model_attempt_identity = ModelAttemptIdentity(
        model_step_id=f"mstep_{'7' * 32}",
        model_attempt_id=f"matt_{'8' * 32}",
    )
    logical_step_id = model_attempt_identity.model_step_id
    stage_id = f"{logical_step_id}:dispatch:0"
    intent = {
        "schema_version": 1,
        "purpose": "assistant-turn",
        **model_attempt_identity.payload(),
        "logical_step_id": logical_step_id,
        "provider_name": provider_name,
        "requested_model": "fake-model",
        "source_transcript_cursor": source_transcript_cursor,
        "request_fingerprint": "5" * 64,
    }
    await store.prepare_model_completion_stage(
        session_id,
        request=ModelCompletionStageRequest(
            stage_id=stage_id,
            logical_step_id=logical_step_id,
            dispatch_ordinal=0,
            purpose="assistant-turn",
            intent=intent,
        ),
        expected_statuses={SessionStatus.RUNNING},
        expected_run_epoch=running.run_epoch,
        expected_transcript_cursor=source_transcript_cursor,
    )

    tool_round_identity = ToolRoundIdentity(
        **model_attempt_identity.payload(),
        tool_round_id=f"tround_{'9' * 32}",
    )
    round_id = tool_round_identity.tool_round_id
    validation = structured_output_tool_round._validate_structured_output_tool_round(
        tool_calls=tool_calls,
        spec=spec,
    )
    durable_tool_calls = tool_calls
    durable_validation = validation
    if redactor is not None:
        durable_tool_calls = [
            runtime_records.ToolCallRequest(
                id=call.id,
                name=call.name,
                arguments=redactor.redact_json_values(call.arguments),
            )
            for call in tool_calls
        ]
        durable_validation = structured_output_tool_round._redact_structured_output_validation(
            validation,
            redactor,
        )
    target_checkpoint, pending_round = tool_round_recovery.checkpoint_with_pending_tool_round(
        None,
        agent_name="assistant",
        environment_name=None,
        task_id=None,
        tool_calls=durable_tool_calls,
        policy_outcomes=None,
        structured_output=spec,
        tool_round_identity=tool_round_identity,
        source_model_step_id=logical_step_id,
        source_transcript_cursor=source_transcript_cursor,
        model_step=1,
        structured_output_attempt=1,
        structured_output_validation=durable_validation,
    )
    assistant_message = transcript_helpers.assistant_message_with_tool_round(
        Message.tool_call(
            calls=[
                ToolCallPart(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    arguments=call.arguments,
                )
                for call in durable_tool_calls
            ]
        ),
        tool_round_identity,
    )
    completion_event = Event(
        id=f"{session_id}:model-completed",
        type=EventType.MODEL_COMPLETED,
        session_id=session_id,
        interaction_id=interaction_id,
        agent_name="assistant",
        payload={
            **tool_round_identity.payload(),
            "step": 1,
            "attempt": 1,
            "completion": {
                "finish_reason": "tool_calls",
                "raw_finish_reason": "tool_calls",
                "status": None,
            },
            "step_classification": {
                "type": "continue",
                "reason": "assistant requested tool calls",
            },
            "transcript_cursor": source_transcript_cursor + 1,
        },
    )
    pointer = model_completion_publication.ModelStepPublicationCheckpoint(
        logical_step_id=logical_step_id,
        stage_id=stage_id,
        source_transcript_cursor=source_transcript_cursor,
        transcript_end_cursor=source_transcript_cursor + 1,
        completion_event_id=completion_event.id,
        classification=completion_event.payload["step_classification"],
        assistant_message_published=True,
        tool_round_id=round_id,
    )
    target_checkpoint[model_completion_publication.LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY] = (
        pointer.model_dump(mode="json")
    )
    publication = RuntimePublicationRequest(
        publication_id=logical_step_id,
        kind="model-step",
        interaction_id=interaction_id,
        intent=intent,
        mutation=runtime_publication_checkpoint_mutation(None, target_checkpoint),
        transcript_messages=(assistant_message,),
        events=(completion_event,),
    )
    completed = await store.complete_model_completion_stage(
        session_id,
        stage_id=stage_id,
        publication=publication,
    )
    assert completed.stage.state == "completed"
    promoted = await store.promote_model_completion_stage(
        session_id,
        stage_id=stage_id,
        expected_run_epoch=running.run_epoch,
    )
    assert promoted.replayed is False
    assert await store.load_active_model_completion_stage(session_id) is None
    return _PublishedStructuredStep(
        session=running,
        user_message=user_message,
        assistant_message=assistant_message,
        pending_round=pending_round,
        completion_event=completion_event,
        model_intent=intent,
        model_receipt=promoted.receipt,
    )


def _round_events(events: list[Event], round_id: str) -> list[Event]:
    return [
        event
        for event in events
        if event.payload.get("tool_round_id") == round_id
        and event.type
        in {
            EventType.TOOL_CALL_COMPLETED,
            EventType.TOOL_CALL_FAILED,
            EventType.STRUCTURED_OUTPUT_VALIDATING,
            EventType.STRUCTURED_OUTPUT_VALIDATED,
            EventType.STRUCTURED_OUTPUT_FAILED,
            EventType.STRUCTURED_OUTPUT_RETRY,
        }
    ]


def _assert_unique_event_ids(events: list[Event]) -> None:
    event_ids = [event.id for event in events]
    assert len(event_ids) == len(set(event_ids))


def test_incomplete_recovery_atomically_finalizes_valid_structured_output_round() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _RecordingProvider()
        spec = _answer_spec()
        staged = await _publish_structured_model_step(
            store,
            session_id="structured-recovery-valid",
            provider_name=provider.name,
            spec=spec,
            tool_calls=[
                _structured_call(
                    call_id="call-final-valid",
                    output={"answer": "recovered"},
                )
            ],
        )
        await store.release_run_fence(staged.session.id)
        app = _register_runtime(store, provider)

        recovery = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=staged.session.id)
        )
        transcript = await store.load_transcript(staged.session.id)
        events = await store.load_events(staged.session.id)
        checkpoint = await store.load_checkpoint(staged.session.id)
        session = await store.load(staged.session.id)
        receipt = await store.load_runtime_publication_receipt(
            staged.session.id,
            f"tool-round:{staged.pending_round.tool_round_id}",
        )

        assert provider.requests == []
        assert session is not None
        assert session.status == SessionStatus.INTERRUPTED
        assert IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND in recovery.actions
        assert [message.role.value for message in transcript] == [
            "user",
            "assistant",
            "tool",
        ]
        grouped_result = transcript[-1]
        assert len(grouped_result.content) == 1
        result_part = grouped_result.content[0]
        assert isinstance(result_part, ToolResultPart)
        assert result_part.tool_call_id == "call-final-valid"
        assert result_part.content == "Structured output accepted."
        assert result_part.structured == {"output": {"answer": "recovered"}}
        assert result_part.is_error is False
        assert tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint) is None

        round_events = _round_events(events, staged.pending_round.tool_round_id)
        assert [event.type for event in round_events] == [
            EventType.TOOL_CALL_COMPLETED,
            EventType.STRUCTURED_OUTPUT_VALIDATING,
            EventType.STRUCTURED_OUTPUT_VALIDATED,
        ]
        terminal_event, *auxiliary_events = round_events
        assert receipt is not None
        assert receipt.transcript_start_cursor == 2
        assert receipt.transcript_end_cursor == 3
        assert receipt.appended_event_ids == tuple(event.id for event in auxiliary_events)
        assert tuple(reference.event_id for reference in receipt.referenced_events) == (
            terminal_event.id,
        )
        assert receipt.intent["round_id"] == staged.pending_round.tool_round_id
        assert receipt.intent["tool_call_ids"] == ["call-final-valid"]
        assert len(receipt.intent["pending_round_digest"]) == 64
        assert receipt.intent["auxiliary"] == {
            "schema_version": 1,
            "kind": "structured-output-validation",
            "step": 1,
            "attempt": 1,
            "valid": True,
            "retry_scheduled": False,
            "event_ids": [event.id for event in auxiliary_events],
        }
        assert staged.model_receipt.intent == staged.model_intent
        assert staged.model_receipt.appended_event_ids == (staged.completion_event.id,)
        assert {event.id for event in round_events}.issubset(
            {event.id for event in recovery.events}
        )
        _assert_unique_event_ids(events)

    asyncio.run(scenario())


def test_incomplete_recovery_uses_authoritative_validation_before_redaction() -> None:
    async def scenario() -> None:
        secret = "structured-output-authoritative-secret-12345"
        redactor = SecretRedactor(secret)
        spec = StructuredOutputSpec(
            name="answer",
            json_schema={
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "minLength": 30,
                    }
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
        )
        store = InMemorySessionStore()
        provider = _RecordingProvider()
        staged = await _publish_structured_model_step(
            store,
            session_id="structured-recovery-redacted-valid",
            provider_name=provider.name,
            spec=spec,
            tool_calls=[
                _structured_call(
                    call_id="call-final-redacted-valid",
                    output={"answer": secret},
                )
            ],
            redactor=redactor,
        )
        checkpoint_before_recovery = await store.load_checkpoint(staged.session.id)
        assert staged.pending_round.structured_output_validation is not None
        assert staged.pending_round.structured_output_validation.valid is True
        assert staged.pending_round.structured_output_validation.output == {
            "answer": REDACTED_SECRET
        }
        assert secret not in str(checkpoint_before_recovery)

        await store.release_run_fence(staged.session.id)
        app = _register_runtime(
            store,
            provider,
            secret_redactor=redactor,
        )
        recovery = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=staged.session.id)
        )
        transcript = await store.load_transcript(staged.session.id)
        events = await store.load_events(staged.session.id)

        assert provider.requests == []
        assert IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND in recovery.actions
        assert transcript[-1].content[0].structured == {"output": {"answer": REDACTED_SECRET}}
        validated = next(
            event for event in events if event.type is EventType.STRUCTURED_OUTPUT_VALIDATED
        )
        assert validated.payload["output"] == {"answer": REDACTED_SECRET}
        assert secret not in str([message.model_dump(mode="json") for message in transcript])
        assert secret not in str([event.model_dump(mode="json") for event in events])

    asyncio.run(scenario())


def test_resume_recovers_invalid_round_then_continues_once_at_next_attempt() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _RecordingProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call-final-attempt-2",
                        name=STRUCTURED_OUTPUT_TOOL_NAME,
                        arguments={"output": {"answer": "fixed"}},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        )
        spec = _answer_spec(max_retries=2)
        staged = await _publish_structured_model_step(
            store,
            session_id="structured-recovery-retry",
            provider_name=provider.name,
            spec=spec,
            tool_calls=[
                _structured_call(
                    call_id="call-final-attempt-1",
                    output={"wrong": "value"},
                )
            ],
        )
        await store.release_run_fence(staged.session.id)
        await store.update_status(staged.session.id, SessionStatus.INTERRUPTED)
        app = _register_runtime(store, provider)
        continuation = Message.text("user", "continue after repairing the crashed round")

        resumed_events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id=staged.session.id,
                    messages=[continuation],
                    structured_output=spec,
                )
            )
        ]
        transcript = await store.load_transcript(staged.session.id)
        durable_events = await store.load_events(staged.session.id)
        checkpoint = await store.load_checkpoint(staged.session.id)
        session = await store.load(staged.session.id)
        first_receipt = await store.load_runtime_publication_receipt(
            staged.session.id,
            f"tool-round:{staged.pending_round.tool_round_id}",
        )

        assert len(provider.requests) == 1
        request_messages = [
            message for message in provider.requests[0].messages if message.role.value != "system"
        ]
        assert [message.role.value for message in request_messages] == [
            "user",
            "assistant",
            "tool",
            "user",
        ]
        recovered_result = request_messages[2].content[0]
        assert isinstance(recovered_result, ToolResultPart)
        assert recovered_result.tool_call_id == "call-final-attempt-1"
        assert recovered_result.is_error is True
        assert request_messages[-1] == continuation

        first_round_events = _round_events(
            durable_events,
            staged.pending_round.tool_round_id,
        )
        assert [event.type for event in first_round_events] == [
            EventType.TOOL_CALL_FAILED,
            EventType.STRUCTURED_OUTPUT_VALIDATING,
            EventType.STRUCTURED_OUTPUT_FAILED,
            EventType.STRUCTURED_OUTPUT_RETRY,
        ]
        first_terminal, *first_auxiliary = first_round_events
        assert all(event.payload["attempt"] == 1 for event in first_auxiliary)
        assert first_receipt is not None
        assert first_receipt.appended_event_ids == tuple(event.id for event in first_auxiliary)
        assert tuple(reference.event_id for reference in first_receipt.referenced_events) == (
            first_terminal.id,
        )
        assert first_receipt.intent["auxiliary"] == {
            "schema_version": 1,
            "kind": "structured-output-validation",
            "step": 1,
            "attempt": 1,
            "valid": False,
            "retry_scheduled": True,
            "event_ids": [event.id for event in first_auxiliary],
        }

        retry_index = next(
            index
            for index, event in enumerate(resumed_events)
            if event.type == EventType.STRUCTURED_OUTPUT_RETRY
        )
        model_started_index = next(
            index
            for index, event in enumerate(resumed_events)
            if event.type == EventType.MODEL_STARTED
        )
        assert retry_index < model_started_index
        second_terminal = next(
            event
            for event in durable_events
            if event.payload.get("tool_call_id") == "call-final-attempt-2"
            and event.type == EventType.TOOL_CALL_COMPLETED
        )
        second_round_id = second_terminal.payload["tool_round_id"]
        second_round_events = _round_events(durable_events, second_round_id)
        assert [event.type for event in second_round_events] == [
            EventType.TOOL_CALL_COMPLETED,
            EventType.STRUCTURED_OUTPUT_VALIDATING,
            EventType.STRUCTURED_OUTPUT_VALIDATED,
        ]
        assert second_round_events[-1].payload["attempt"] == 2
        second_receipt = await store.load_runtime_publication_receipt(
            staged.session.id,
            f"tool-round:{second_round_id}",
        )
        assert second_receipt is not None
        assert second_receipt.intent["auxiliary"]["attempt"] == 2
        assert second_receipt.intent["auxiliary"]["valid"] is True
        assert second_receipt.intent["auxiliary"]["retry_scheduled"] is False

        assert [message.role.value for message in transcript] == [
            "user",
            "assistant",
            "tool",
            "user",
            "assistant",
            "tool",
        ]
        assert session is not None
        assert session.status == SessionStatus.COMPLETED
        assert tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint) is None
        assert sum(event.type == EventType.MODEL_COMPLETED for event in durable_events) == 2
        _assert_unique_event_ids(durable_events)

    asyncio.run(scenario())


def test_incomplete_recovery_fails_mixed_round_without_retry_or_side_effects() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _RecordingProvider()
        side_effect = _SideEffectTool()
        spec = _answer_spec(max_retries=2)
        staged = await _publish_structured_model_step(
            store,
            session_id="structured-recovery-interrupted",
            provider_name=provider.name,
            spec=spec,
            tool_calls=[
                runtime_records.ToolCallRequest(
                    id="call-side-effect",
                    name=side_effect.spec.name,
                    arguments={"value": "must not run"},
                ),
                _structured_call(
                    call_id="call-final-mixed",
                    output={"answer": "invalid mixed round"},
                ),
            ],
        )
        await store.release_run_fence(staged.session.id)
        app = _register_runtime(store, provider, tools=[side_effect])

        recovery = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=staged.session.id)
        )
        transcript = await store.load_transcript(staged.session.id)
        events = await store.load_events(staged.session.id)
        checkpoint = await store.load_checkpoint(staged.session.id)
        receipt = await store.load_runtime_publication_receipt(
            staged.session.id,
            f"tool-round:{staged.pending_round.tool_round_id}",
        )

        assert provider.requests == []
        assert side_effect.calls == []
        assert IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND in recovery.actions
        assert tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint) is None
        round_events = _round_events(events, staged.pending_round.tool_round_id)
        assert [event.type for event in round_events] == [
            EventType.TOOL_CALL_FAILED,
            EventType.TOOL_CALL_FAILED,
            EventType.STRUCTURED_OUTPUT_VALIDATING,
            EventType.STRUCTURED_OUTPUT_FAILED,
        ]
        assert EventType.STRUCTURED_OUTPUT_RETRY not in [event.type for event in round_events]
        assert EventType.TOOL_CALL_STARTED not in [event.type for event in events]
        terminal_events = round_events[:2]
        auxiliary_events = round_events[2:]
        assert all(
            event.payload["structured_output_validation"] is True for event in terminal_events
        )
        grouped_result = transcript[-1]
        assert grouped_result.role.value == "tool"
        assert len(grouped_result.content) == 2
        assert all(
            isinstance(part, ToolResultPart) and part.is_error for part in grouped_result.content
        )
        assert receipt is not None
        assert receipt.appended_event_ids == tuple(event.id for event in auxiliary_events)
        assert tuple(reference.event_id for reference in receipt.referenced_events) == tuple(
            event.id for event in terminal_events
        )
        assert receipt.intent["auxiliary"]["valid"] is False
        assert receipt.intent["auxiliary"]["retry_scheduled"] is False
        assert receipt.intent["auxiliary"]["event_ids"] == [event.id for event in auxiliary_events]
        _assert_unique_event_ids(events)

    asyncio.run(scenario())


def test_recovery_replays_lost_structured_publication_acknowledgement_exactly() -> None:
    async def scenario() -> None:
        store = _ToolPublicationAcknowledgementLostStore()
        provider = _RecordingProvider()
        spec = _answer_spec()
        staged = await _publish_structured_model_step(
            store,
            session_id="structured-recovery-ack-loss",
            provider_name=provider.name,
            spec=spec,
            tool_calls=[
                _structured_call(
                    call_id="call-final-ack-loss",
                    output={"answer": "once"},
                )
            ],
        )
        await store.release_run_fence(staged.session.id)
        app = _register_runtime(store, provider)

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=staged.session.id)
        )
        transcript = await store.load_transcript(staged.session.id)
        events = await store.load_events(staged.session.id)
        checkpoint = await store.load_checkpoint(staged.session.id)
        receipt = await store.load_runtime_publication_receipt(
            staged.session.id,
            f"tool-round:{staged.pending_round.tool_round_id}",
        )

        assert store.lost_acknowledgement is True
        assert store.tool_publication_calls == 2
        assert provider.requests == []
        assert [message.role.value for message in transcript] == [
            "user",
            "assistant",
            "tool",
        ]
        assert tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint) is None
        round_events = _round_events(events, staged.pending_round.tool_round_id)
        assert [event.type for event in round_events] == [
            EventType.TOOL_CALL_COMPLETED,
            EventType.STRUCTURED_OUTPUT_VALIDATING,
            EventType.STRUCTURED_OUTPUT_VALIDATED,
        ]
        assert receipt is not None
        assert receipt.appended_event_ids == tuple(event.id for event in round_events[1:])
        assert tuple(reference.event_id for reference in receipt.referenced_events) == (
            round_events[0].id,
        )
        assert receipt.intent["auxiliary"]["event_ids"] == [event.id for event in round_events[1:]]
        _assert_unique_event_ids(events)

    asyncio.run(scenario())


def test_live_structured_round_replays_lost_publication_acknowledgement() -> None:
    async def scenario() -> None:
        store = _ToolPublicationAcknowledgementLostStore()
        provider = _RecordingProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call-final-live-ack-loss",
                        name=STRUCTURED_OUTPUT_TOOL_NAME,
                        arguments={"output": {"answer": "once"}},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        )
        app = _register_runtime(store, provider)

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="structured-live-ack-loss",
                    messages=[Message.text("user", "produce a structured answer")],
                    structured_output=_answer_spec(),
                )
            )
        ]

        session = await store.load("structured-live-ack-loss")
        transcript = await store.load_transcript("structured-live-ack-loss")
        stored_events = await store.load_events("structured-live-ack-loss")
        checkpoint = await store.load_checkpoint("structured-live-ack-loss")

        assert store.lost_acknowledgement is True
        assert store.tool_publication_calls == 2
        assert len(provider.requests) == 1
        assert session is not None and session.status is SessionStatus.COMPLETED
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert [message.role.value for message in transcript] == [
            "user",
            "assistant",
            "tool",
        ]
        assert tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint) is None
        assert [event.type for event in stored_events].count(
            EventType.STRUCTURED_OUTPUT_VALIDATED
        ) == 1
        _assert_unique_event_ids(stored_events)

    asyncio.run(scenario())


def test_live_structured_round_validates_before_durable_redaction() -> None:
    async def scenario() -> None:
        secret = "structured-output-live-authoritative-secret-12345"
        redactor = SecretRedactor(secret)
        spec = StructuredOutputSpec(
            name="answer",
            json_schema={
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "minLength": 30,
                    }
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
        )
        store = InMemorySessionStore()
        provider = _RecordingProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call-final-live-redacted-valid",
                        name=STRUCTURED_OUTPUT_TOOL_NAME,
                        arguments={"output": {"answer": secret}},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        )
        app = _register_runtime(
            store,
            provider,
            secret_redactor=redactor,
        )
        session_id = "structured-live-redacted-valid"

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "produce a structured answer")],
                    structured_output=spec,
                )
            )
        ]
        session = await store.load(session_id)
        transcript = await store.load_transcript(session_id)
        durable_events = await store.load_events(session_id)

        assert session is not None and session.status is SessionStatus.COMPLETED
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len(provider.requests) == 1
        assert transcript[-1].content[0].structured == {"output": {"answer": REDACTED_SECRET}}
        assert any(
            event.type is EventType.STRUCTURED_OUTPUT_VALIDATED
            and event.payload["output"] == {"answer": REDACTED_SECRET}
            for event in durable_events
        )
        assert secret not in str([message.model_dump(mode="json") for message in transcript])
        assert secret not in str([event.model_dump(mode="json") for event in durable_events])

    asyncio.run(scenario())


def test_live_structured_retry_scopes_reused_call_id_to_current_round() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _RecordingProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call-final-reused",
                        name=STRUCTURED_OUTPUT_TOOL_NAME,
                        arguments={"output": {"wrong": "value"}},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.tool_call(
                        id="call-final-reused",
                        name=STRUCTURED_OUTPUT_TOOL_NAME,
                        arguments={"output": {"answer": "current round"}},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
            ]
        )
        app = _register_runtime(store, provider)
        session_id = "structured-live-reused-call-id"

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "retry the structured answer")],
                    structured_output=_answer_spec(max_retries=1),
                )
            )
        ]

        session = await store.load(session_id)
        transcript = await store.load_transcript(session_id)
        terminal_events = [
            event
            for event in await store.load_events(session_id)
            if event.payload.get("tool_call_id") == "call-final-reused"
            and event.type
            in {
                EventType.TOOL_CALL_COMPLETED,
                EventType.TOOL_CALL_FAILED,
            }
        ]

        assert session is not None and session.status is SessionStatus.COMPLETED
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len(provider.requests) == 2
        assert [event.type for event in terminal_events] == [
            EventType.TOOL_CALL_FAILED,
            EventType.TOOL_CALL_COMPLETED,
        ]
        assert len({event.payload["tool_round_id"] for event in terminal_events}) == 2
        assert [message.role.value for message in transcript] == [
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
        ]
        assert transcript[-1].content[0].structured == {"output": {"answer": "current round"}}

    asyncio.run(scenario())


def test_limit_closure_scopes_reused_call_id_to_current_round() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        tool = _SideEffectTool()
        provider = _RecordingProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call-limit-reused",
                        name="side_effect",
                        arguments={"value": "first round"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.tool_call(
                        id="call-limit-reused",
                        name="side_effect",
                        arguments={"value": "limited round"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
            ]
        )
        app = _register_runtime(store, provider, tools=[tool])
        session_id = "ordinary-limit-reused-call-id"

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "use the tool twice")],
                    limits=RunLimits(max_tool_calls=1),
                )
            )
        ]

        session = await store.load(session_id)
        transcript = await store.load_transcript(session_id)
        terminal_events = [
            event
            for event in await store.load_events(session_id)
            if event.payload.get("tool_call_id") == "call-limit-reused"
            and event.type
            in {
                EventType.TOOL_CALL_COMPLETED,
                EventType.TOOL_CALL_FAILED,
            }
        ]

        assert tool.calls == [{"value": "first round"}]
        assert session is not None and session.status is SessionStatus.INTERRUPTED
        assert events[-1].type is EventType.SESSION_INTERRUPTED
        assert EventType.SESSION_LIMIT_REACHED in [event.type for event in events]
        assert [event.type for event in terminal_events] == [
            EventType.TOOL_CALL_COMPLETED,
            EventType.TOOL_CALL_FAILED,
        ]
        assert len({event.payload["tool_round_id"] for event in terminal_events}) == 2
        assert [message.role.value for message in transcript] == [
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
        ]
        assert transcript[-1].content[0].is_error is True

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "max_parallel_tool_calls",
    [
        pytest.param(1, id="serial"),
        pytest.param(2, id="parallel"),
    ],
)
def test_live_ordinary_round_replays_lost_publication_acknowledgement(
    max_parallel_tool_calls: int,
) -> None:
    async def scenario() -> None:
        store = _ToolPublicationAcknowledgementLostStore()
        session_id = f"ordinary-live-ack-loss-{max_parallel_tool_calls}"
        provider = _RecordingProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call-side-effect-live-ack-loss-a",
                        name="side_effect",
                        arguments={"value": "first"},
                    ),
                    ModelStreamEvent.tool_call(
                        id="call-side-effect-live-ack-loss-b",
                        name="side_effect",
                        arguments={"value": "second"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        tool = _SideEffectTool()
        app = _register_runtime(
            store,
            provider,
            tools=[tool],
            max_parallel_tool_calls=max_parallel_tool_calls,
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run both side effects")],
                )
            )
        ]

        session = await store.load(session_id)
        transcript = await store.load_transcript(session_id)
        stored_events = await store.load_events(session_id)
        checkpoint = await store.load_checkpoint(session_id)
        terminal_events = [
            event
            for event in stored_events
            if event.type == EventType.TOOL_CALL_COMPLETED
            and event.payload.get("tool_call_id")
            in {
                "call-side-effect-live-ack-loss-a",
                "call-side-effect-live-ack-loss-b",
            }
        ]
        assert len(terminal_events) == 2
        round_ids = {event.payload["tool_round_id"] for event in terminal_events}
        assert len(round_ids) == 1
        (round_id,) = round_ids
        receipt = await store.load_runtime_publication_receipt(
            session_id,
            f"tool-round:{round_id}",
        )

        assert store.lost_acknowledgement is True
        assert store.tool_publication_calls == 2
        assert len(provider.requests) == 2
        assert sorted(call["value"] for call in tool.calls) == ["first", "second"]
        assert len(tool.calls) == 2
        assert session is not None and session.status is SessionStatus.COMPLETED
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert [message.role.value for message in transcript] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        tool_result_parts = transcript[2].content
        assert all(type(part) is ToolResultPart for part in tool_result_parts)
        assert [part.tool_call_id for part in tool_result_parts] == [
            "call-side-effect-live-ack-loss-a",
            "call-side-effect-live-ack-loss-b",
        ]
        assert tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint) is None
        assert receipt is not None
        expected_references: list[str] = []
        for tool_call_id in (
            "call-side-effect-live-ack-loss-a",
            "call-side-effect-live-ack-loss-b",
        ):
            expected_references.extend(
                event.id
                for event in stored_events
                if event.payload.get("tool_round_id") == round_id
                and event.payload.get("tool_call_id") == tool_call_id
                and event.type
                in {
                    EventType.TOOL_CALL_STARTED,
                    EventType.TOOL_CALL_COMPLETED,
                }
            )
        assert tuple(reference.event_id for reference in receipt.referenced_events) == tuple(
            expected_references
        )
        _assert_unique_event_ids(stored_events)

    asyncio.run(scenario())
