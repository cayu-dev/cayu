"""A genuine recipient creation can enter native participant execution exactly."""

import asyncio
from uuid import uuid4

import pytest
from tests.core.test_collaboration_request_foundation import setup
from tests.core.test_participant_identity import CONTEXT, app
from tests.core.test_prepared_admission_public import native_stores as native_stores

from cayu.agents import AgentSpec
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions import RunRequest
from cayu.sessions.context_views import (
    ParticipantSessionExecutionRequest,
    RecipientSessionCreationRequest,
)


@pytest.mark.anyio
@pytest.mark.parametrize("explicit_id", [False, True])
@pytest.mark.parametrize("competing_input", [False, True])
async def test_recipient_native_execution_preserves_creation_provenance(
    native_stores, explicit_id, competing_input, monkeypatch
):
    collaboration, sessions, *_ = native_stores
    original, _initialized, _, recipient, *_ = await setup(collaboration)
    application = app(
        collaboration,
        original._participant_coordinator._registration,
        session_store=sessions,
    )
    provider = ScriptedModelProvider(
        [[ModelStreamEvent.text_delta("result"), ModelStreamEvent.completed()]], name="provider"
    )
    application.register_provider(provider, default=True)
    application.register_agent(AgentSpec(name="reviewer", model="model", system_prompt="system"))
    await application.initialize_collaboration()
    case_id = uuid4().hex
    creation = RecipientSessionCreationRequest(
        request=RunRequest(
            agent_name="reviewer",
            session_id="recipient-" + case_id if explicit_id else None,
            messages=[Message.text("user", "input")],
        ),
        creation_key="native-recipient:" + case_id,
        recipient=recipient.reference,
    )
    session, receipt = await application.create_recipient_session(creation, context=CONTEXT)
    assert receipt.participant_receipt.recipient_metadata_json is not None
    execution = ParticipantSessionExecutionRequest(
        request=creation.participant_request.request.model_copy(update={"session_id": session.id}),
        session_instance_id=session.instance_id,
        execution_key="native-output",
    )
    changed = ParticipantSessionExecutionRequest(
        request=execution.request.model_copy(update={"messages": [Message.text("user", "other")]}),
        session_instance_id=session.instance_id,
        execution_key=execution.execution_key,
    )
    with pytest.raises(ValueError, match="creation request"):
        _ = [
            event
            async for event in application.execute_participant_session(
                changed, participant=recipient.reference, context=CONTEXT
            )
        ]
    assert provider.requests == []
    assert (await sessions.load(session.id)).run_epoch == 0

    async def execute():
        return [
            event
            async for event in application.execute_participant_session(
                execution, participant=recipient.reference, context=CONTEXT
            )
        ]

    if competing_input:
        entered, release = asyncio.Event(), asyncio.Event()
        register = collaboration._register_permit

        async def paused(*args, **kwargs):
            entered.set()
            await release.wait()
            return await register(*args, **kwargs)

        monkeypatch.setattr(collaboration, "_register_permit", paused)
        task = asyncio.create_task(execute())
        try:
            await asyncio.wait_for(entered.wait(), 5)
            await sessions.append_transcript_messages(
                session.id, [Message.text("user", "competing input")], interaction_id=None
            )
            retained = await sessions.load_transcript(session.id)
            release.set()
            with pytest.raises(RuntimeError, match="Initial transcript changed"):
                await asyncio.wait_for(task, 5)
            assert provider.requests == []
            assert await sessions.load_transcript(session.id) == retained
            unchanged = await sessions.load(session.id)
            assert unchanged.run_epoch == 0 and unchanged.status == "pending"
            assert await sessions.load_checkpoint(session.id) is None
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
        return
    events = await execute()
    assert events
    if len(provider.requests) != 1:
        pytest.fail("\n".join(f"{event.type}: {event.payload}" for event in events))
    assert (await sessions.load(session.id)).status == "completed"
    transcript = await sessions.load_transcript(session.id)
    assert [message.role for message in transcript] == ["system", "user", "assistant"]
    assert await sessions.load_transcript_cursor(session.id) == len(transcript)
    assert (
        await sessions.load_participant_session_creation_receipt(session.id)
        == receipt.participant_receipt
    )
