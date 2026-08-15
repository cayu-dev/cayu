from __future__ import annotations

import asyncio

from tests.core._execution_profile_fixtures import create_admitted_session

from cayu.core import EventType, Message
from cayu.runtime import InMemorySessionStore, RunRequest, SessionStatus
from cayu.runtime.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.runtime.execution_profiles import (
    active_invocation_execution_profile_from_checkpoint,
    execution_profile_from_session_metadata,
)
from cayu.runtime.sessions import (
    INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY,
    SESSION_CREATE_CLAIM_METADATA_KEY,
    run_request_with_runtime_generated_authority,
    run_request_with_runtime_session_create_claim,
)


def test_create_admitted_session_builds_complete_runtime_creation_authority() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        source = Message.text("user", "continue the durable work")
        system = Message.text("system", "Stay within the durable contract.")
        request = run_request_with_runtime_generated_authority(
            RunRequest(
                agent_name="assistant",
                session_id="sess-admitted-fixture",
                messages=[source],
                metadata={"owner": "caller"},
            ),
            "session_id",
        )
        request, _claim = run_request_with_runtime_session_create_claim(
            request,
            claim_id="admitted-fixture-claim",
        )

        admitted = await create_admitted_session(
            store,
            request=request,
            provider_name="fixture-provider",
            model="fixture-model",
            durable_system_prompt="Stay within the durable contract.",
        )

        session = await store.load(admitted.session.id)
        assert session is not None
        assert session.status is SessionStatus.RUNNING
        assert session.run_epoch == 1
        assert session.metadata["owner"] == "caller"
        assert (
            session.metadata[SESSION_CREATE_CLAIM_METADATA_KEY]
            == (admitted.request.metadata[SESSION_CREATE_CLAIM_METADATA_KEY])
        )
        assert session.metadata[SESSION_CREATE_CLAIM_METADATA_KEY]["interaction_id"] == (
            admitted.active_invocation_profile.interaction_id
        )
        assert execution_profile_from_session_metadata(session.metadata) == (
            admitted.active_invocation_profile.profile
        )
        assert admitted.identity.execution_profile == admitted.active_invocation_profile.profile

        events = await store.load_events(session.id)
        assert [event.type for event in events] == [EventType.INTERACTION_STARTED]
        assert events[0] == admitted.interaction_started_event
        assert events[0].interaction_id == admitted.active_invocation_profile.interaction_id

        assert await store.load_transcript(session.id) == [system, source]
        checkpoint = await store.load_checkpoint(session.id)
        assert checkpoint is not None
        assert checkpoint[CHECKPOINT_SCHEMA_VERSION_KEY] == CURRENT_CHECKPOINT_SCHEMA_VERSION
        assert INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY not in checkpoint
        assert active_invocation_execution_profile_from_checkpoint(checkpoint) == (
            admitted.active_invocation_profile
        )
        assert admitted.active_invocation_profile.session_id == session.id
        assert admitted.active_invocation_profile.run_epoch == session.run_epoch

    asyncio.run(run())
