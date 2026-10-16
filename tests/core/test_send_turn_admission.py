"""Terminal turns retain native ownership and participant access decisions."""

import asyncio
from uuid import uuid4

import pytest
from tests.core.test_abandoned_session_resume import _child, _Proposal, _proposal_app
from tests.core.test_participant_identity import CONTEXT, create, registration
from tests.core.test_participant_identity import app as participant_app
from tests.core.test_send_turn import Store, text_response
from tests.core.test_session_execution_presence import _stores
from tests.core.test_tool_round_publication_failure_matrix import store_factory as store_factory
from tests.runtime.runtime_provider_support import VersionedFakeProvider

from cayu import AgentSpec, Message, RunRequest, SessionStatus, ToolEffect, TurnRequest, send_turn
from cayu.collaboration.access import CollaborationAccessContext, CollaborationAccessDenied
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.evals import ScriptedModelProvider
from cayu.sessions import SessionExecutionInProgress
from cayu.sessions.context_views import (
    ParticipantSessionCreationRequest,
    ParticipantSessionExecutionRequest,
)


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("owner_alive", [False, True])
def test_turn_recovers_only_a_lost_execution_owner(
    backend, owner_alive, request, sqlite_resources, tmp_path
):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, reopen):
            receipt = tmp_path / "proposal-receipt"
            session_id = "turn-owner-" + uuid4().hex
            child = _child(
                store,
                backend,
                request,
                receipt,
                session_id,
                ToolEffect.NONE,
                "block" if owner_alive else "crash",
            )
            try:
                if owner_alive:
                    line = await asyncio.wait_for(asyncio.to_thread(child.stdout.readline), 30)
                    assert line.strip() == "committed"
                else:
                    stdout, stderr = await asyncio.wait_for(
                        asyncio.to_thread(child.communicate), 30
                    )
                    assert child.returncode == 137, (stdout, stderr)
                observer = reopen()
                original = await observer.load(session_id)
                assert original.status is SessionStatus.RUNNING
                provider = VersionedFakeProvider(text_response("Recovered"))
                app = _proposal_app(
                    observer, _Proposal(ToolEffect.NONE, receipt), provider, replay=False
                )
                turn = TurnRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "Any update?")],
                )
                if owner_alive:
                    with pytest.raises(SessionExecutionInProgress):
                        await send_turn(app, turn)
                    assert (await observer.load(session_id)).run_epoch == original.run_epoch
                    assert not provider.requests
                else:
                    outcome = await send_turn(app, turn)
                    assert outcome.ok, outcome.error
                    assert outcome.final_text == "Recovered"
                    assert (await observer.load(session_id)).run_epoch > original.run_epoch
                assert receipt.read_text() == "committed\n"
            finally:
                if child.poll() is None:
                    child.kill()
                    await asyncio.to_thread(child.wait)

    asyncio.run(scenario())


def test_participant_turn_authorizes_before_selection_and_passes_context_to_resume(store_factory):
    async def scenario():
        async with store_factory(Store) as store:
            collaboration = InMemoryCollaborationStore()
            app = participant_app(collaboration, registration(), session_store=store)
            provider = ScriptedModelProvider([text_response("Initial"), text_response("Continued")])
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="reviewer", model="model"))
            initialized = await app.initialize_collaboration()
            _, created = await create(app, initialized)
            participant = created.participants[0].reference
            creation = ParticipantSessionCreationRequest(
                creation_key="turn-create",
                request=RunRequest(agent_name="reviewer", messages=[Message.text("user", "Start")]),
            )
            session, _ = await app.create_participant_session(
                creation, participant=participant, context=CONTEXT
            )
            execution = ParticipantSessionExecutionRequest(
                request=creation.request.model_copy(update={"session_id": session.id}),
                session_instance_id=session.instance_id,
                execution_key="turn-execute",
            )
            async for _ in app.execute_participant_session(
                execution, participant=participant, context=CONTEXT
            ):
                pass
            before = await store.load_transcript(session.id)
            for options in ({}, {"agent_name": "wrong"}, {"labels": {}}):
                turn = TurnRequest(
                    **{
                        "agent_name": "reviewer",
                        "session_id": session.id,
                        "messages": [Message.text("user", "Continue")],
                        **options,
                    }
                )
                with pytest.raises(PermissionError):
                    await send_turn(app, turn)
                with pytest.raises(CollaborationAccessDenied):
                    await send_turn(
                        app, turn, context=CollaborationAccessContext(principal="outsider")
                    )
                assert await store.load_transcript(session.id) == before
            outcome = await send_turn(
                app,
                TurnRequest(
                    agent_name="reviewer",
                    session_id=session.id,
                    messages=[Message.text("user", "Continue")],
                ),
                context=CONTEXT,
            )
            assert outcome.ok, outcome.error
            assert outcome.final_text == "Continued"
            assert len(provider.requests) == 2
            await collaboration.close()

    asyncio.run(scenario())
