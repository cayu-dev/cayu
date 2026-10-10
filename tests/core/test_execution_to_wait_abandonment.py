"""Real public stream closure preserves or explicitly settles wait ownership."""

import pytest
from tests.core.test_collaboration_request_foundation import public_setup
from tests.core.test_collaboration_waits import wait_for
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_participant_identity import stores as stores

from cayu.agents import AgentSpec
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._session_continuation import continuation_digest
from cayu.sessions.base import InMemorySessionStore
from cayu.sessions.context_views import (
    ParticipantSessionCreationRequest,
    ParticipantSessionExecutionRequest,
)
from cayu.sessions.requests import RunRequest
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresSessionStore
from cayu.storage.sqlite import SQLiteSessionStore

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("parked", (False, True))
async def test_public_execution_wait_stream_close(stores, tmp_path, request, parked):
    source = stores()
    if isinstance(source, InMemoryCollaborationStore):
        store = InMemorySessionStore()
    elif isinstance(source, SQLiteCollaborationStore):
        store = SQLiteSessionStore(tmp_path / "abandoned-wait.sqlite")
    else:
        store = PostgresSessionStore(
            request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
        )
    try:
        app, resolver, values = await public_setup(source, session_store=store)
        accepted = await app.accept_collaboration_request(values[4], context=resolver.context)
        wait = wait_for(accepted, values[1]).model_copy(
            update={"service_policy": "clarification", "failure_policy": "return_and_report"}
        )
        provider = ScriptedModelProvider(
            [[ModelStreamEvent.text_delta("Ready."), ModelStreamEvent.completed()]]
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="root", model="model"))
        creation = ParticipantSessionCreationRequest(
            creation_key="abandonment:" + values[1].binding.application_scope,
            request=RunRequest(agent_name="root", messages=[Message.text("user", "Wait.")]),
        )
        participant = values[2].reference
        session, _ = await app.create_participant_session(
            creation, participant=participant, context=CONTEXT
        )
        execution = ParticipantSessionExecutionRequest(
            request=creation.request.model_copy(update={"session_id": session.id}),
            session_instance_id=session.instance_id,
            execution_key="abandoned-execution",
        )

        def stream():
            return app.execute_participant_session_to_wait(
                execution,
                wait,
                participant=participant,
                context=CONTEXT,
                wait_context=resolver.context,
            )

        selected = EventType.SESSION_INTERRUPTED if parked else EventType.MODEL_TEXT_DELTA
        observed = stream()
        try:
            async for event in observed:
                if event.type == selected:
                    break
            else:
                pytest.fail("Execution did not reach the selected public close boundary")
        finally:
            await observed.aclose()
        assert len(provider.requests) == 1
        ticket = await store.load_continuation_ticket(
            session.id,
            session_instance_id=session.instance_id,
            registration_key="execution-wait:" + continuation_digest(wait.operation),
        )
        assert ticket is not None
        released = await store.load(session.id)
        assert released is not None
        assert released.run_epoch == ticket.ticket.writer_generation + 1
        assert released.status.value not in {"running", "interrupting"}
        if not parked:
            assert ticket.ticket.state == "ARMING"
            excluded = await app.exclude_participant_session_wait(
                execution,
                wait,
                participant=participant,
                context=CONTEXT,
                wait_context=resolver.context,
            )
            assert excluded.delivery == "excluded"
            before = await store.load(session.id)
            events = await store.load_events(session.id)
            assert (
                await app.exclude_participant_session_wait(
                    execution,
                    wait,
                    participant=participant,
                    context=CONTEXT,
                    wait_context=resolver.context,
                )
                == excluded
            )
            after = await store.load(session.id)
            assert after is not None and before is not None
            # Exact acknowledgement publication may refresh native activity
            # timestamps. It cannot change execution authority or other state.
            assert after.model_dump(
                exclude={"last_activity_at", "updated_at"}
            ) == before.model_dump(exclude={"last_activity_at", "updated_at"})
            assert await store.load_events(session.id) == events
            retired = await store.load_continuation_ticket(
                session.id,
                session_instance_id=session.instance_id,
                registration_key=ticket.ticket.registration_key,
            )
            assert retired is not None and retired.ticket.state == "RETIRED"
            assert retired.retirement_acknowledged
        else:
            assert ticket.ticket.state == "WAITING"
            before = await store.load(session.id)
            events = await store.load_events(session.id)
            assert [event async for event in stream()] == []
            assert await store.load(session.id) == before
            assert await store.load_events(session.id) == events
        assert len(provider.requests) == 1
    finally:
        if not isinstance(store, InMemorySessionStore):
            await store.close()
