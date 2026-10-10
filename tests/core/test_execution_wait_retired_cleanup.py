"""An absent foreign registration cannot orphan a released native preparation."""

import pytest
from tests.core._execution_wait_cleanup_flow import prune_and_reopen
from tests.core.test_collaboration_request_foundation import public_setup
from tests.core.test_collaboration_waits import wait_for
from tests.core.test_participant_identity import CONTEXT
from tests.core.test_participant_identity import stores as stores

from cayu.agents import AgentSpec
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import Message
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


async def test_absent_registration_cleanup_after_namespace_pruning(
    stores, tmp_path, request, monkeypatch
):
    source = stores()
    if isinstance(source, InMemoryCollaborationStore):
        store = InMemorySessionStore()
    elif isinstance(source, SQLiteCollaborationStore):
        store = SQLiteSessionStore(tmp_path / "execution-wait.sqlite")
    else:
        store = PostgresSessionStore(
            request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
        )
    try:
        application, resolver, values = await public_setup(source, session_store=store)
        initialized, participant = values[1], values[2].reference
        request_receipt = await application.accept_collaboration_request(
            values[4], context=resolver.context
        )
        wait = wait_for(request_receipt, initialized)
        provider = ScriptedModelProvider([])
        application.register_provider(provider, default=True)
        application.register_agent(AgentSpec(name="root", model="model"))
        creation = ParticipantSessionCreationRequest(
            creation_key="pruned-missing-wait",
            request=RunRequest(agent_name="root", messages=[Message.text("user", "Wait.")]),
        )
        session, _ = await application.create_participant_session(
            creation, participant=participant, context=CONTEXT
        )
        execution = ParticipantSessionExecutionRequest(
            request=creation.request.model_copy(update={"session_id": session.id}),
            session_instance_id=session.instance_id,
            execution_key="missing-registration",
        )

        async def unavailable(*args, **kwargs):
            raise OSError("Foreign registration unavailable before commit")

        monkeypatch.setattr(application._wait_coordinator, "register", unavailable)
        async for _ in application.execute_participant_session_to_wait(
            execution,
            wait,
            participant=participant,
            context=CONTEXT,
            wait_context=resolver.context,
        ):
            pass
        native = await store.load_continuation_ticket(
            session.id,
            session_instance_id=session.instance_id,
            registration_key="execution-wait:" + continuation_digest(wait.operation),
        )
        assert native is not None and native.ticket.state == "ARMING"
        bound = wait.model_copy(update={"delivery_ticket": native.preparation.intent})
        assert (
            await source.load_wait(initialized, bound, redactor=application._secret_redactor)
            is None
        )
        application, source, store = await prune_and_reopen(
            application,
            source,
            store,
            stores,
            tmp_path,
            request,
            resolver,
            initialized,
            request_receipt,
            bound,
        )
        result = await application.exclude_participant_session_wait(
            execution,
            wait,
            participant=participant,
            context=CONTEXT,
            wait_context=resolver.context,
        )
        assert result.delivery == "excluded"
        assert (
            await application.exclude_participant_session_wait(
                execution,
                wait,
                participant=participant,
                context=CONTEXT,
                wait_context=resolver.context,
            )
            == result
        )
        assert (
            await source.load_wait(initialized, bound, redactor=application._secret_redactor)
            is None
        )
        assert not provider.requests
        await store.delete_session(session.id)
        assert await store.load(session.id) is None
    finally:
        if not isinstance(store, InMemorySessionStore):
            await store.close()
        await source.close()
