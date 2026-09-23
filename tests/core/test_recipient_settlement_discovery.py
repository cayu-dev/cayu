"""Terminal creation is not acknowledgement of its cross-store settlement."""

import asyncio
from uuid import uuid4

import pytest
from tests.core.test_participant_identity import CONTEXT, create, registration
from tests.core.test_session_creation_fence import (
    _application,
    _collaboration_factory,
    _store_factory,
)

from cayu.collaboration._contracts import CollaborationConflict
from cayu.collaboration.lifecycle import ParticipantLifecycleChange
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.messages import Message
from cayu.sessions._recipient_admission import settle_recipient_creation
from cayu.sessions.base import RunRequest
from cayu.sessions.context_views import RecipientSessionCreationRequest
from cayu.sessions.creation_fence import _SESSION_CREATION_AUTHORITY


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("terminal", ["created", "excluded"])
@pytest.mark.parametrize("settlement_committed", [False, True])
async def test_terminal_creation_remains_discoverable_until_settlement_ack(
    backend, terminal, settlement_committed, tmp_path, request, monkeypatch
):
    session_factory = _store_factory(backend, tmp_path, request)
    collaboration_factory = _collaboration_factory(backend, tmp_path, request)
    sessions, collaboration = session_factory(), collaboration_factory()
    config = registration()
    application, initialized, provider = await _application(sessions, collaboration, config)
    _, participants = await create(application, initialized)
    participant = participants.participants[0].reference
    creation = RecipientSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[Message.text("user", "input")]),
        recipient=participant,
        creation_key="terminal-discovery-" + uuid4().hex,
    )
    original_create = sessions.create_participant_owned_session
    settlement_method = "_exclude_permit" if terminal == "excluded" else "_settle_permit"
    original_settle = getattr(collaboration, settlement_method)
    entered = asyncio.Event()

    async def exclude_then_create(*args, **kwargs):
        await sessions._exclude_session_creation_target(
            kwargs["creation_target"], authority=_SESSION_CREATION_AUTHORITY
        )
        return await original_create(*args, **kwargs)

    async def interrupted_settlement(*args, **kwargs):
        if settlement_committed:
            await original_settle(*args, **kwargs)
        entered.set()
        await asyncio.Future()

    with monkeypatch.context() as patch:
        if terminal == "excluded":
            patch.setattr(sessions, "create_participant_owned_session", exclude_then_create)
        patch.setattr(collaboration, settlement_method, interrupted_settlement)
        task = asyncio.create_task(application.create_recipient_session(creation, context=CONTEXT))
        try:
            await asyncio.wait_for(entered.wait(), 15)
            task.cancel()
            assert task.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 15)
            assert task.cancelled()
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    assert provider.requests == []
    if backend != "memory":
        await sessions.close()
        await collaboration.close()
    # Recovery deliberately has neither the request nor a retained target handle.
    del creation, application, task
    reopened, recovered_collaboration = session_factory(), collaboration_factory()
    recovered, _, _ = await _application(reopened, recovered_collaboration, config)
    try:
        page = await reopened.list_pending_session_creations(participant.owner, limit=1)
        assert len(page.decisions) == 1 and page.next_cursor is None
        decision = page.decisions[0]
        assert decision.state == terminal
        assert not decision.settlement_acknowledged
        inspection = await recovered.inspect_participant(participant, context=CONTEXT)
        assert inspection.outstanding_obligations == int(not settlement_committed)
        await settle_recipient_creation(recovered, decision.target)
        await settle_recipient_creation(recovered, decision.target)
        assert not (await reopened.list_pending_session_creations(participant.owner)).decisions
        inspection = await recovered.inspect_participant(participant, context=CONTEXT)
        assert inspection.outstanding_obligations == 0
        found = await reopened.read_session_creation_decision(decision.target)
        assert found.receipt.state == terminal
        assert found.receipt.settlement_acknowledged
        assert found.receipt.session_instance_id == decision.session_instance_id
    finally:
        if backend != "memory":
            await reopened.close()
            await recovered_collaboration.close()


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_terminal_exclusion_fences_never_registered_responsibility(
    backend, tmp_path, request, monkeypatch
):
    session_factory = _store_factory(backend, tmp_path, request)
    collaboration_factory = _collaboration_factory(backend, tmp_path, request)
    sessions, collaboration = session_factory(), collaboration_factory()
    config = registration()
    application, initialized, provider = await _application(sessions, collaboration, config)
    _, participants = await create(application, initialized)
    participant = participants.participants[0].reference

    async def refuse_registration(*args, **kwargs):
        raise ConnectionError("registration unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(collaboration, "_register_permit", refuse_registration)
        with pytest.raises(CollaborationUnavailable):
            await application.create_recipient_session(
                RecipientSessionCreationRequest(
                    request=RunRequest(agent_name="reviewer", messages=[]),
                    recipient=participant,
                    creation_key="unadmitted-discovery-" + uuid4().hex,
                ),
                context=CONTEXT,
            )
    await application.change_participant_lifecycle(
        ParticipantLifecycleChange(
            operation=initialized.operation("disable-unadmitted"),
            participant=participant,
            expected_lifecycle_revision=1,
            state="disabled",
        ),
        context=CONTEXT,
    )
    if backend != "memory":
        await sessions.close()
        await collaboration.close()
    reopened, recovered_collaboration = session_factory(), collaboration_factory()
    recovered, recovered_initialization, _ = await _application(
        reopened, recovered_collaboration, config
    )
    try:
        page = await reopened.list_pending_session_creations(participant.owner)
        assert len(page.decisions) == 1
        target = page.decisions[0].target
        assert not page.decisions[0].responsibility_registered
        await reopened._exclude_session_creation_target(
            target, authority=_SESSION_CREATION_AUTHORITY
        )
        await settle_recipient_creation(recovered, target)
        await settle_recipient_creation(recovered, target)
        assert not (await reopened.list_pending_session_creations(participant.owner)).decisions
        inspection = await recovered.inspect_participant(participant, context=CONTEXT)
        assert inspection.outstanding_obligations == inspection.issued_permit_frontier == 0
        with pytest.raises(CollaborationConflict):
            await recovered_collaboration._register_permit(
                recovered_initialization, target.permit, redactor=recovered._secret_redactor
            )
        assert provider.requests == []
    finally:
        if backend != "memory":
            await reopened.close()
            await recovered_collaboration.close()
