from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from tests.core.test_context_view_admission import _close, _factory, _selection
from tests.core.test_participant_identity import CONTEXT, app, create, registration

from cayu import (
    AlwaysRequireApprovalToolPolicy,
    ToolApprovalDecision,
    ToolApprovalRequest,
    UserInputResponse,
)
from cayu.agents import AgentSpec
from cayu.collaboration.lifecycle import ParticipantLifecycleChange
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.context import CheckpointCompactionContextPolicy
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._model_completion_publication import model_step_publication_from_checkpoint
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.base import CompactSessionRequest, Message, ResumeRequest, RunRequest
from cayu.sessions.context_views import (
    ContextViewPublicationRequest,
    ParticipantSessionCreationRequest,
    ParticipantSessionExecutionRequest,
)
from cayu.tools.base import Tool, ToolResult, ToolSpec
from cayu.tools.user_input import UserInputTool


class CountingTool(Tool):
    spec = ToolSpec(
        name="count",
        description="Count calls.",
        input_schema={"type": "object", "properties": {}},
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:participant-count", behavior_version="1", implementation_version="1"
        ),
    )

    def __init__(self, entered=None, release=None):
        super().__init__()
        self.calls = 0
        self.entered = entered
        self.release = release

    async def run(self, ctx, args):
        self.calls += 1
        if self.entered is not None:
            self.entered.set()
            await self.release.wait()
        return ToolResult(content="counted historical result")


async def activate(value, participant, *, older_turn=False):
    messages = (
        [Message.text("user", "old question"), Message.text("assistant", "old answer")]
        if older_turn
        else []
    ) + [Message.text("user", "start")]
    creation = ParticipantSessionCreationRequest(
        RunRequest(agent_name="reviewer", messages=messages), str(uuid4())
    )
    session, _ = await value.create_participant_session(
        creation, participant=participant, context=CONTEXT
    )
    request = ParticipantSessionExecutionRequest(
        request=creation.request.model_copy(update={"session_id": session.id}),
        session_instance_id=session.instance_id,
        execution_key="execute",
    )
    return session, request


async def collect(stream):
    return [event async for event in stream]


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("entrance", ["approval", "input", "compaction"])
@pytest.mark.parametrize("lifecycle", ["disabled", "retired"])
def test_participant_continuation_requires_current_authority(
    backend, entrance, lifecycle, tmp_path, request
):
    from tests.core.test_explicit_session_compaction import RecordingCompactor

    factory = _factory(backend, tmp_path, request, [datetime.now(UTC)])

    async def run():
        store = factory()
        collaboration = InMemoryCollaborationStore()
        tool = CountingTool()
        compactor = RecordingCompactor()
        first = [ModelStreamEvent.text_delta("done")]
        if entrance != "compaction":
            first = [ModelStreamEvent.tool_call(id="count-1", name="count", arguments={})]
            if entrance == "input":
                first.insert(
                    0,
                    ModelStreamEvent.tool_call(
                        id="input-1", name="ask_user", arguments={"question": "Continue?"}
                    ),
                )
        provider = ScriptedModelProvider(
            [
                [
                    *first,
                    ModelStreamEvent.completed(
                        {"finish_reason": "stop" if entrance == "compaction" else "tool_calls"}
                    ),
                ],
                [
                    ModelStreamEvent.text_delta("finished"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        reg = registration()

        def configured_app():
            value = app(collaboration, reg, session_store=store)
            value.register_provider(provider, default=True)
            value.register_agent(
                AgentSpec(name="reviewer", model="model"),
                tools=[tool, UserInputTool()],
                tool_policy=AlwaysRequireApprovalToolPolicy(tools=["count"])
                if entrance == "approval"
                else None,
                context_policy=CheckpointCompactionContextPolicy(
                    compactor=compactor, max_user_turns=1, compact_after_messages=100
                ),
            )
            return value

        value = configured_app()
        try:
            initialized = await value.initialize_collaboration()
            _, created = await create(value, initialized)
            participant = created.participants[0].reference
            session, execution = await activate(
                value, participant, older_turn=entrance == "compaction"
            )
            await collect(
                value.execute_participant_session(
                    execution, participant=participant, context=CONTEXT
                )
            )
            if entrance == "compaction":
                await collect(
                    value.resume(
                        ResumeRequest(
                            session_id=session.id, messages=[Message.text("user", "next question")]
                        ),
                        context=CONTEXT,
                    )
                )
            events = await store.load_events(session.id)
            if entrance == "approval":
                event = next(
                    event
                    for event in events
                    if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
                )
                intent = ToolApprovalRequest(
                    session_id=session.id,
                    approval_id=event.payload["approval_id"],
                    tool_round_id=event.payload["tool_round_id"],
                    tool_call_id=event.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
                method = value.resolve_tool_approval
            elif entrance == "input":
                event = next(
                    event for event in events if event.type is EventType.SESSION_AWAITING_USER_INPUT
                )
                intent = UserInputResponse(
                    session_id=session.id, input_id=event.payload["input_id"], answer="yes"
                )
                method = value.resolve_user_input
            else:
                current = await store.load(session.id)
                intent = CompactSessionRequest(
                    session_id=session.id,
                    idempotency_key="compact",
                    expected_run_epoch=current.run_epoch,
                    expected_transcript_cursor=await store.load_transcript_cursor(session.id),
                )
                method = value.compact_session
            before = (
                await store.load(session.id),
                await store.load_checkpoint(session.id),
                await store.load_events(session.id),
            )
            with pytest.raises(PermissionError, match="administration"):
                await collect(method(intent))
            await value.change_participant_lifecycle(
                ParticipantLifecycleChange(
                    operation=initialized.operation("deactivate"),
                    participant=participant,
                    expected_lifecycle_revision=1,
                    state=lifecycle,
                ),
                context=CONTEXT,
            )
            if backend != "memory":
                await store.close()
                store = factory()
                value = configured_app()
                await value.initialize_collaboration()
                method = getattr(
                    value,
                    {
                        "approval": "resolve_tool_approval",
                        "input": "resolve_user_input",
                        "compaction": "compact_session",
                    }[entrance],
                )
            with pytest.raises(PermissionError, match="active participants"):
                await collect(method(intent, context=CONTEXT))
            assert before == (
                await store.load(session.id),
                await store.load_checkpoint(session.id),
                await store.load_events(session.id),
            )
            assert tool.calls == 0 and compactor.requests == []
            assert len(provider.requests) == (2 if entrance == "compaction" else 1)
            if lifecycle == "disabled":
                await value.change_participant_lifecycle(
                    ParticipantLifecycleChange(
                        operation=initialized.operation("reactivate"),
                        participant=participant,
                        expected_lifecycle_revision=2,
                        state="active",
                    ),
                    context=CONTEXT,
                )
                await collect(method(intent, context=CONTEXT))
                assert len(compactor.requests) == (entrance == "compaction")
                assert tool.calls == (entrance != "compaction")
        finally:
            await _close(store)
            await collaboration.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("round_kind", ["ordinary", "approval", "input"])
def test_publication_retains_completed_tool_round_while_parent_continues(
    backend, round_kind, tmp_path, request
):
    factory = _factory(backend, tmp_path, request, [datetime.now(UTC)])

    async def run():
        store = factory()
        collaboration = InMemoryCollaborationStore()
        tool_entered, tool_release, provider_entered, provider_release = (
            asyncio.Event() for _ in range(4)
        )
        tool = CountingTool(tool_entered, tool_release)

        class Provider(ScriptedModelProvider):
            async def stream(self, request):
                if self.requests:
                    provider_entered.set()
                    await provider_release.wait()
                async for event in super().stream(request):
                    yield event

        provider = Provider(
            [
                [
                    *(
                        [
                            ModelStreamEvent.tool_call(
                                id="input-1", name="ask_user", arguments={"question": "Continue?"}
                            )
                        ]
                        if round_kind == "input"
                        else []
                    ),
                    ModelStreamEvent.tool_call(id="count-1", name="count", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("next turn"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        value = app(collaboration, registration(), session_store=store)
        value.register_provider(provider, default=True)
        value.register_agent(
            AgentSpec(name="reviewer", model="model"),
            tools=[tool, UserInputTool()],
            tool_policy=AlwaysRequireApprovalToolPolicy(tools=["count"])
            if round_kind == "approval"
            else None,
        )
        running = None
        try:
            initialized = await value.initialize_collaboration()
            _, created = await create(value, initialized)
            participant = created.participants[0].reference
            session, execution = await activate(value, participant)
            running = asyncio.create_task(
                collect(
                    value.execute_participant_session(
                        execution, participant=participant, context=CONTEXT
                    )
                )
            )
            if round_kind != "ordinary":
                await running
                events = await store.load_events(session.id)
                if round_kind == "approval":
                    event = next(
                        event
                        for event in events
                        if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
                    )
                    stream = value.resolve_tool_approval(
                        ToolApprovalRequest(
                            session_id=session.id,
                            approval_id=event.payload["approval_id"],
                            tool_round_id=event.payload["tool_round_id"],
                            tool_call_id=event.payload["tool_call_id"],
                            decision=ToolApprovalDecision.APPROVE,
                        ),
                        context=CONTEXT,
                    )
                else:
                    event = next(
                        event
                        for event in events
                        if event.type is EventType.SESSION_AWAITING_USER_INPUT
                    )
                    stream = value.resolve_user_input(
                        UserInputResponse(
                            session_id=session.id, input_id=event.payload["input_id"], answer="yes"
                        ),
                        context=CONTEXT,
                    )
                running = asyncio.create_task(collect(stream))
            await asyncio.wait_for(tool_entered.wait(), 20)
            pointer = model_step_publication_from_checkpoint(
                await store.load_checkpoint(session.id)
            )
            completion = next(
                event
                for event in await store.load_events(session.id)
                if event.id == pointer.completion_event_id
            )
            intent = ContextViewPublicationRequest(
                source_session_id=session.id,
                source_session_instance_id=session.instance_id,
                view_id=str(uuid4()),
                interaction_id=completion.interaction_id,
                boundary_id=pointer.logical_step_id,
                publication_key=str(uuid4()),
                projection_schema="whole-turn.v1",
            )
            with pytest.raises(ValueError, match="partial tool round|closure evidence"):
                await value.publish_completed_context_view(
                    intent, participant=participant, context=CONTEXT
                )
            assert await store.lookup_context_view_publication(intent.publication_key) is None
            tool_release.set()
            await asyncio.wait_for(provider_entered.wait(), 20)
            manifest = await value.publish_completed_context_view(
                intent, participant=participant, context=CONTEXT
            )
            messages = json.loads(manifest.messages_json)
            assert [message["role"] for message in messages] == ["assistant", "tool"]
            assert "counted historical result" in manifest.messages_json
            assert manifest.transcript_cursor == pointer.source_transcript_cursor + 2
            selected = await value.select_context_view(
                _selection(manifest, "select"), participant=participant, context=CONTEXT
            )
            provider_release.set()
            await running
            if backend != "memory":
                await store.close()
                store = factory()
                value = app(
                    collaboration, value._participant_coordinator._registration, session_store=store
                )
                await value.initialize_collaboration()
            readback = await value.read_context_view(
                manifest.view_id,
                source_session_id=session.id,
                participant=participant,
                context=CONTEXT,
            )
            assert readback.view == selected.view == manifest
            assert (
                await value.publish_completed_context_view(
                    intent, participant=participant, context=CONTEXT
                )
                == manifest
            )
            assert tool.calls == 1 and len(provider.requests) == 2
        finally:
            tool_release.set()
            provider_release.set()
            if running is not None and not running.done():
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
            await _close(store)
            await collaboration.close()

    asyncio.run(run())
