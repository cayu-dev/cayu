from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from cayu.core import AgentSpec, Event, EventType, Message
from cayu.core.messages import ToolResultPart
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    ModelCompletionManualRecoveryRequired,
    ResumeRequest,
    RunRequest,
    Session,
    SessionIdentity,
    SessionStatus,
)
from cayu.runtime import _model_completion_publication as model_completion_publication
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime.execution_units import ModelAttemptIdentity, ToolRoundIdentity
from cayu.runtime.sessions import (
    ModelCompletionStage,
    ModelCompletionStageRequest,
    RuntimePublicationRequest,
    runtime_publication_checkpoint_mutation,
)


class _RecordingProvider(ModelProvider):
    name = "model-completion-recovery"

    def __init__(self, responses: list[list[ModelStreamEvent]] | None = None) -> None:
        self._responses = [] if responses is None else responses
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        call_index = len(self.requests)
        self.requests.append(request)
        if call_index >= len(self._responses):
            raise AssertionError("Recovery must not redispatch the staged model completion.")
        for event in self._responses[call_index]:
            yield event


class _NeverExecutedTool(Tool):
    spec = ToolSpec(
        name="echo",
        description="Echo one value.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        self.calls += 1
        return ToolResult(content="must not execute during recovery")


class _PromotionAcknowledgementLostStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.promotion_calls = 0
        self.lost_acknowledgement = False

    async def _promote_model_completion_stage_atomic(self, **kwargs):
        self.promotion_calls += 1
        result = await super()._promote_model_completion_stage_atomic(**kwargs)
        if not self.lost_acknowledgement and result.replayed is False:
            self.lost_acknowledgement = True
            raise ConnectionError("recovered model promotion acknowledgement lost")
        return result


@dataclass(frozen=True)
class _StagedCompletion:
    session: Session
    stage: ModelCompletionStage
    user_message: Message
    assistant_message: Message
    completion_event: Event
    pointer: model_completion_publication.ModelStepPublicationCheckpoint
    publication: RuntimePublicationRequest


def _register_runtime(
    store: InMemorySessionStore,
    provider: _RecordingProvider,
    *,
    tool: _NeverExecutedTool | None = None,
) -> CayuApp:
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[] if tool is None else [tool],
    )
    return app


async def _stage_completed_model_boundary(
    store: InMemorySessionStore,
    *,
    session_id: str,
    provider_name: str,
    with_tool_call: bool = False,
) -> _StagedCompletion:
    user_message = Message.text("user", "complete this model step once")
    created = await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[user_message],
        ),
        identity=SessionIdentity(
            provider_name=provider_name,
            model="fake-model",
        ),
    )
    await store.append_transcript_messages(created.id, [user_message])
    running = await store.transition_status(
        created.id,
        from_statuses={SessionStatus.PENDING},
        to_status=SessionStatus.RUNNING,
    )

    source_cursor = 1
    model_attempt_identity = ModelAttemptIdentity(
        model_step_id=f"mstep_{'1' * 32}",
        model_attempt_id=f"matt_{'2' * 32}",
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
        "source_transcript_cursor": source_cursor,
        "request_fingerprint": "0" * 64,
    }
    await store.prepare_model_completion_stage(
        session_id,
        request=ModelCompletionStageRequest(
            stage_id=stage_id,
            logical_step_id=logical_step_id,
            dispatch_ordinal=0,
            purpose="assistant-turn",
            intent=intent,
            reservation_ids=("reservation:model-recovery",),
        ),
        expected_statuses={SessionStatus.RUNNING},
        expected_run_epoch=running.run_epoch,
        expected_transcript_cursor=source_cursor,
    )

    tool_round_id: str | None = None
    target_checkpoint: dict = {}
    if with_tool_call:
        tool_round_identity = ToolRoundIdentity(
            **model_attempt_identity.payload(),
            tool_round_id=f"tround_{'3' * 32}",
        )
        assistant_message = transcript_helpers.assistant_message_with_tool_round(
            Message.tool_call(
                tool_call_id="call-recovered",
                tool_name="echo",
                arguments={"value": "recover me"},
            ),
            tool_round_identity,
        )
        tool_call = runtime_records.ToolCallRequest(
            id="call-recovered",
            name="echo",
            arguments={"value": "recover me"},
        )
        tool_round_id = tool_round_identity.tool_round_id
        target_checkpoint, _pending_round = tool_round_recovery.checkpoint_with_pending_tool_round(
            None,
            agent_name="assistant",
            environment_name=None,
            task_id=None,
            tool_calls=[tool_call],
            policy_outcomes=None,
            structured_output=None,
            tool_round_identity=tool_round_identity,
            source_model_step_id=logical_step_id,
            source_transcript_cursor=source_cursor,
            model_step=1,
        )
        classification = {
            "type": "continue",
            "reason": "assistant requested tool calls",
        }
    else:
        assistant_message = Message.text("assistant", "recovered authoritative answer")
        classification = {
            "type": "final",
            "reason": "assistant produced user-visible content",
        }

    completion_event = Event(
        id=f"{session_id}:model-completed",
        type=EventType.MODEL_COMPLETED,
        session_id=session_id,
        agent_name="assistant",
        payload={
            **model_attempt_identity.payload(),
            **({} if tool_round_id is None else {"tool_round_id": tool_round_id}),
            "step": 1,
            "attempt": 1,
            "completion": {
                "finish_reason": "tool_calls" if with_tool_call else "stop",
                "raw_finish_reason": "tool_calls" if with_tool_call else "stop",
                "status": None,
            },
            "step_classification": classification,
            "transcript_cursor": source_cursor + 1,
        },
    )
    pointer = model_completion_publication.ModelStepPublicationCheckpoint(
        logical_step_id=logical_step_id,
        stage_id=stage_id,
        source_transcript_cursor=source_cursor,
        transcript_end_cursor=source_cursor + 1,
        completion_event_id=completion_event.id,
        classification=classification,
        assistant_message_published=True,
        tool_round_id=tool_round_id,
    )
    target_checkpoint[model_completion_publication.LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY] = (
        pointer.model_dump(mode="json")
    )
    publication = RuntimePublicationRequest(
        publication_id=logical_step_id,
        kind="model-step",
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
    assert await store.load_active_model_completion_stage(session_id) is not None
    assert await store.load_transcript(session_id) == [user_message]
    assert await store.load_events(session_id) == []
    return _StagedCompletion(
        session=running,
        stage=completed.stage,
        user_message=user_message,
        assistant_message=assistant_message,
        completion_event=completion_event,
        pointer=pointer,
        publication=publication,
    )


async def _stage_in_flight_model_boundary(
    store: InMemorySessionStore,
    *,
    session_id: str,
    provider_name: str,
) -> tuple[Session, Message, ModelCompletionStage]:
    user_message = Message.text("user", "do not dispatch twice")
    await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[user_message],
        ),
        identity=SessionIdentity(
            provider_name=provider_name,
            model="fake-model",
        ),
    )
    await store.append_transcript_messages(session_id, [user_message])
    running = await store.transition_status(
        session_id,
        from_statuses={SessionStatus.PENDING},
        to_status=SessionStatus.RUNNING,
    )
    logical_step_id = f"mstep_{'4' * 32}"
    prepared = await store.prepare_model_completion_stage(
        session_id,
        request=ModelCompletionStageRequest(
            stage_id=f"{logical_step_id}:dispatch:0",
            logical_step_id=logical_step_id,
            dispatch_ordinal=0,
            intent={
                "schema_version": 1,
                "purpose": "assistant-turn",
                "logical_step_id": logical_step_id,
                "provider_name": provider_name,
                "requested_model": "fake-model",
                "source_transcript_cursor": 1,
                "request_fingerprint": "1" * 64,
            },
            reservation_ids=("reservation:ambiguous-dispatch",),
        ),
        expected_statuses={SessionStatus.RUNNING},
        expected_run_epoch=running.run_epoch,
        expected_transcript_cursor=1,
    )
    return running, user_message, prepared.stage


def test_incomplete_recovery_promotes_completed_model_boundary_without_redispatch() -> None:
    async def run():
        store = InMemorySessionStore()
        provider = _RecordingProvider()
        staged = await _stage_completed_model_boundary(
            store,
            session_id="model-recovery-completed",
            provider_name=provider.name,
        )
        await store.release_run_fence(staged.session.id)
        app = _register_runtime(store, provider)

        recovery = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=staged.session.id)
        )
        return store, provider, staged, recovery

    store, provider, staged, recovery = asyncio.run(run())
    session_id = staged.session.id
    transcript = asyncio.run(store.load_transcript(session_id))
    durable_events = asyncio.run(store.load_events(session_id))
    checkpoint = asyncio.run(store.load_checkpoint(session_id))
    receipt = asyncio.run(
        store.load_runtime_publication_receipt(
            session_id,
            staged.stage.logical_step_id,
        )
    )

    assert provider.requests == []
    assert transcript == [staged.user_message, staged.assistant_message]
    assert [event for event in durable_events if event.type == EventType.MODEL_COMPLETED] == [
        staged.completion_event
    ]
    assert receipt is not None
    assert receipt.publication_id == staged.stage.logical_step_id
    assert receipt.transcript_start_cursor == 1
    assert receipt.transcript_end_cursor == 2
    assert receipt.appended_event_ids == (staged.completion_event.id,)
    assert (
        model_completion_publication.model_step_publication_from_checkpoint(checkpoint)
        == staged.pointer
    )
    assert asyncio.run(store.load_active_model_completion_stage(session_id)) is None
    assert staged.completion_event in recovery.events


def test_resume_rejects_in_flight_model_boundary_before_status_change() -> None:
    async def run():
        store = InMemorySessionStore()
        provider = _RecordingProvider()
        running, user_message, stage = await _stage_in_flight_model_boundary(
            store,
            session_id="model-recovery-in-flight",
            provider_name=provider.name,
        )
        await store.release_run_fence(running.id)
        interrupted = await store.update_status(running.id, SessionStatus.INTERRUPTED)
        app = _register_runtime(store, provider)

        with pytest.raises(ModelCompletionManualRecoveryRequired):
            async for _event in app.resume(
                ResumeRequest(
                    session_id=running.id,
                    messages=[Message.text("user", "unsafe retry")],
                )
            ):
                pass
        return store, provider, interrupted, user_message, stage

    store, provider, interrupted, user_message, stage = asyncio.run(run())
    persisted = asyncio.run(store.load(interrupted.id))
    active = asyncio.run(store.load_active_model_completion_stage(interrupted.id))

    assert persisted == interrupted
    assert active is not None
    assert active.stage == stage
    assert active.stage.reservation_ids == ("reservation:ambiguous-dispatch",)
    assert asyncio.run(store.load_transcript(interrupted.id)) == [user_message]
    assert provider.requests == []


def test_resume_promotes_tool_completion_and_recovers_round_before_next_provider_call() -> None:
    async def run():
        store = InMemorySessionStore()
        provider = _RecordingProvider(
            [
                [
                    ModelStreamEvent.text_delta("continued after durable recovery"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        )
        tool = _NeverExecutedTool()
        staged = await _stage_completed_model_boundary(
            store,
            session_id="model-recovery-tool-round",
            provider_name=provider.name,
            with_tool_call=True,
        )
        await store.release_run_fence(staged.session.id)
        await store.update_status(staged.session.id, SessionStatus.INTERRUPTED)
        app = _register_runtime(store, provider, tool=tool)

        events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id=staged.session.id,
                    messages=[Message.text("user", "continue after recovery")],
                )
            )
        ]
        return store, provider, tool, staged, events

    store, provider, tool, staged, events = asyncio.run(run())
    session_id = staged.session.id
    transcript = asyncio.run(store.load_transcript(session_id))
    durable_events = asyncio.run(store.load_events(session_id))
    model_receipt = asyncio.run(
        store.load_runtime_publication_receipt(
            session_id,
            staged.stage.logical_step_id,
        )
    )
    tool_receipt = asyncio.run(
        store.load_runtime_publication_receipt(
            session_id,
            f"tool-round:{staged.pointer.tool_round_id}",
        )
    )

    assert tool.calls == 0
    assert len(provider.requests) == 1
    assert [message.role.value for message in provider.requests[0].messages] == [
        "user",
        "assistant",
        "tool",
        "user",
    ]
    recovered_result = provider.requests[0].messages[2].content[0]
    assert recovered_result.type == "tool_result"
    assert recovered_result.tool_call_id == "call-recovered"
    assert recovered_result.structured is not None
    assert recovered_result.structured["recovery_reason"] == "pending_tool_round_not_started"
    assert [message.role.value for message in transcript] == [
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
    ]
    assert model_receipt is not None
    assert tool_receipt is not None
    assert (
        tool_round_recovery.pending_tool_round_from_checkpoint(
            asyncio.run(store.load_checkpoint(session_id))
        )
        is None
    )
    assert [event.id for event in events].index(staged.completion_event.id) < next(
        index for index, event in enumerate(events) if event.type == EventType.MODEL_STARTED
    )
    assert sum(event.type == EventType.MODEL_COMPLETED for event in durable_events) == 2


@pytest.mark.parametrize(
    ("case", "tail"),
    [
        (
            "user-message",
            Message.text("user", "this does not close the assistant tool call"),
        ),
        (
            "mismatched-result",
            Message.tool_result(
                tool_call_id="wrong-call",
                tool_name="echo",
                content="wrong identity",
            ),
        ),
        (
            "duplicate-result",
            Message.tool_result(
                results=[
                    ToolResultPart(
                        tool_call_id="call-recovered",
                        tool_name="echo",
                        content="first",
                    ),
                    ToolResultPart(
                        tool_call_id="call-recovered",
                        tool_name="echo",
                        content="duplicate",
                    ),
                ]
            ),
        ),
    ],
)
def test_model_boundary_rejects_invalid_tail_after_missing_tool_round_marker(
    case: str,
    tail: Message,
) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = _RecordingProvider()
        staged = await _stage_completed_model_boundary(
            store,
            session_id=f"model-recovery-invalid-tool-tail-{case}",
            provider_name=provider.name,
            with_tool_call=True,
        )
        promoted = await store.promote_model_completion_stage(
            staged.session.id,
            stage_id=staged.stage.stage_id,
            expected_run_epoch=staged.session.run_epoch,
        )
        checkpoint = await store.load_checkpoint(staged.session.id)
        assert checkpoint is not None
        checkpoint.pop("pending_tool_round")
        await store.checkpoint(staged.session.id, checkpoint)
        await store.append_transcript_messages(
            staged.session.id,
            [tail],
        )
        app = _register_runtime(store, provider)

        with pytest.raises(RuntimeError, match="does not exactly close"):
            await app._recovery_coordinator.reconcile_model_completion_boundary(promoted.session)

        assert provider.requests == []

    asyncio.run(run())


def test_incomplete_recovery_replays_lost_promotion_acknowledgement_exactly() -> None:
    async def run():
        store = _PromotionAcknowledgementLostStore()
        provider = _RecordingProvider()
        staged = await _stage_completed_model_boundary(
            store,
            session_id="model-recovery-promotion-ack-loss",
            provider_name=provider.name,
        )
        await store.release_run_fence(staged.session.id)
        app = _register_runtime(store, provider)

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=staged.session.id)
        )
        return store, provider, staged

    store, provider, staged = asyncio.run(run())
    session_id = staged.session.id
    transcript = asyncio.run(store.load_transcript(session_id))
    durable_events = asyncio.run(store.load_events(session_id))

    assert store.lost_acknowledgement is True
    assert store.promotion_calls == 2
    assert provider.requests == []
    assert transcript == [staged.user_message, staged.assistant_message]
    assert [event for event in durable_events if event.type == EventType.MODEL_COMPLETED] == [
        staged.completion_event
    ]
    assert asyncio.run(store.load_active_model_completion_stage(session_id)) is None
    assert (
        asyncio.run(
            store.load_runtime_publication_receipt(
                session_id,
                staged.stage.logical_step_id,
            )
        )
        is not None
    )
