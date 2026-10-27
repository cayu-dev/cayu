"""A participant session created with Cayu Cloud's model keeps it when it executes."""

from uuid import uuid4

import pytest
from tests.core.test_collaboration_request_foundation import setup
from tests.core.test_managed_settings_controller import RELEASE, FakeCloud, _controller
from tests.core.test_participant_identity import CONTEXT, app
from tests.core.test_prepared_admission_public import native_stores as native_stores

from cayu.agents import AgentSpec
from cayu.evals.testing import ScriptedModelProvider
from cayu.managed_settings import model_setting
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.releases import ReleaseIdentity
from cayu.sessions import RunRequest
from cayu.sessions._model_failover import ModelTarget
from cayu.sessions.context_views import (
    ParticipantSessionExecutionRequest,
    RecipientSessionCreationRequest,
)


@pytest.mark.anyio
@pytest.mark.parametrize("explicit_target", [False, True])
@pytest.mark.parametrize("cloud_changes", [False, True])
async def test_participant_execution_keeps_what_it_was_created_with(
    native_stores, explicit_target, cloud_changes
):
    collaboration, sessions, *_ = native_stores
    original, _initialized, _, recipient, *_ = await setup(collaboration)
    cloud = FakeCloud()
    cloud.values = {
        "agent.reviewer.model": "model-b",
        "config.run.max_steps": 40,
        "config.run.retry_policy.max_attempts": 3,
    }
    application = app(
        collaboration,
        original._participant_coordinator._registration,
        session_store=sessions,
        managed_settings=_controller(cloud),
        release_identity=ReleaseIdentity(id=RELEASE),
    )
    provider = ScriptedModelProvider(
        [[ModelStreamEvent.text_delta("result"), ModelStreamEvent.completed()]], name="provider"
    )
    application.register_provider(provider, default=True)
    application.register_agent(
        AgentSpec(name="reviewer", model="model-a", system_prompt="system"),
        settings=[model_setting()],
    )
    await application.initialize_collaboration()
    case_id = uuid4().hex
    creation = RecipientSessionCreationRequest(
        request=RunRequest(
            agent_name="reviewer",
            messages=[Message.text("user", "input")],
            target=ModelTarget(provider_name="provider", model="model-c")
            if explicit_target
            else None,
        ),
        creation_key="managed-recipient:" + case_id,
        recipient=recipient.reference,
    )
    session, _receipt = await application.create_recipient_session(creation, context=CONTEXT)
    expected = "model-c" if explicit_target else "model-b"
    assert session.model == expected
    if cloud_changes:
        # Cloud changes the model and run controls before the session executes.
        cloud.set(
            agent__reviewer__model="model-z",
            config__run__max_steps=60,
            config__run__retry_policy__max_attempts=4,
        )
        await application.managed_settings_controller.poll_once()
    execution = ParticipantSessionExecutionRequest(
        request=creation.participant_request.request.model_copy(update={"session_id": session.id}),
        session_instance_id=session.instance_id,
        execution_key="managed-output",
    )
    events = [
        event
        async for event in application.execute_participant_session(
            execution, participant=recipient.reference, context=CONTEXT
        )
    ]
    assert [request.model for request in provider.requests] == [expected]
    started = next(event for event in events if event.type.value == "session.started")
    recorded = started.payload["managed_settings"]["values"]
    # Only Cloud values the session actually runs with are recorded.
    assert recorded.get("agent.reviewer.model") == (
        None if explicit_target or cloud_changes else "model-b"
    )
    assert recorded.get("config.run.max_steps") == (None if cloud_changes else 40)
    await application.aclose()
