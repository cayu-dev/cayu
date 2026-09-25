"""Native FORK composition; final planner-stage qualification is separate."""

from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from tests.core.test_context_selection_exclusion import reopened_store
from tests.core.test_participant_identity import CONTEXT, app, create
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_prepared_admission_public import prepared_scenario
from tests.core.test_request_planning_view_owner import source_scenario

from cayu.collaboration._contracts import CollaborationConflict, ExactMatch, ExactNotFound
from cayu.collaboration._planning_creation_evidence import _CreationReadback
from cayu.collaboration._planning_creation_types import RequestCreationStageCommand
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.request_access import RequestRegistration
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions import RunRequest
from cayu.sessions._planning_creation_owner import NativePlanningCreationOwner
from cayu.sessions._planning_view_owner import NativePlanningViewOwner
from cayu.sessions._recipient_preparation import fork_creation_request, resolve_fork_recipient
from cayu.sessions.context_views import ContextViewOwnershipRequest, RecipientSessionCreationRequest

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("lost_ack", [False, True])
async def test_frozen_cross_participant_fork_native_creation_and_release(
    native_stores, tmp_path, monkeypatch, lost_ack
):
    value, resolver, admission, provider, _, initialized = await prepared_scenario(
        native_stores,
        request_ttl_ms=300_000,
        provider_events=[
            [
                ModelStreamEvent.text_delta("historical answer"),
                ModelStreamEvent.completed(
                    {"finish_reason": "stop", "usage": {"input_tokens": 1, "output_tokens": 1}}
                ),
            ]
        ],
    )
    _, created = await create(value, initialized, key="fork-source-participant")
    source_participant = created.participants[0].reference
    value, _, _, source, provider = await source_scenario(
        native_stores, configured=(value, initialized, source_participant), provider=provider
    )
    base = RecipientSessionCreationRequest(
        request=RunRequest(
            agent_name="reviewer", messages=[Message.text("user", "new child input")]
        ),
        creation_key="fork-child-" + uuid4().hex,
        recipient=admission.prepared.recipient,
    )
    blueprint = await value.prepare_recipient_fork(
        base,
        source,
        source_participant=source_participant,
        context=CONTEXT,
        deadline_at_ms=admission.expected.intent.selection.expires_at_ms,
    )
    native = native_stores[1]
    assert await native.lookup_context_view_selection(source.selection_key) is None
    assert isinstance(
        await native.read_session_creation_decision(blueprint.base.creation), ExactNotFound
    )
    assert len(provider.requests) == 1
    receiver = NativePlanningViewOwner(value)
    assert (await receiver.select(blueprint.view, context=CONTEXT)).receipt.state == "selected"
    retained = await receiver.adopt(blueprint.view, context=CONTEXT)
    assert isinstance(retained, ExactMatch) and retained.receipt.participant == base.recipient
    resolved = await resolve_fork_recipient(value, blueprint, retained.receipt, context=CONTEXT)
    # The final operation has the same stable creation identity but distinct
    # material. Only this resolved target may be registered for FORK creation.
    assert resolved.creation.permit.operation == blueprint.base.creation.permit.operation
    assert resolved.creation.request_commitment != blueprint.base.creation.request_commitment
    assert isinstance(await native.read_session_creation_decision(resolved.creation), ExactNotFound)
    selected = await native.lookup_context_view_selection(source.selection_key)
    creation = fork_creation_request(blueprint, retained.receipt, selected)
    session, receipt = await value.create_recipient_session(
        creation, context=CONTEXT, preparation=resolved
    )
    assert session.status == "pending" and receipt.mode == "fork"
    assert (
        receipt.participant_receipt.execution_profile_json == blueprint.base.execution_profile_json
    )
    transcript = await native.load_transcript(session.id)
    assert "historical answer" in "\n".join(message.model_dump_json() for message in transcript)
    assert "new child input" in "\n".join(message.model_dump_json() for message in transcript)
    assert len(provider.requests) == 1
    evidence = await value.prepare_recipient_admission(creation, context=CONTEXT)
    assert evidence.target.kind == "fork"
    assert evidence.target.manifest_commitment == selected.view.manifest_commitment
    stage_command = RequestCreationStageCommand(
        operation=resolved.creation.permit.operation,
        expected=admission.expected,
        preparation=resolved,
    )
    stage_readback = await NativePlanningCreationOwner(value).read(stage_command)
    assert isinstance(stage_readback, _CreationReadback)
    assert stage_readback.receipt.prepared == evidence
    command = admission.model_copy(
        update={
            "operation": initialized.operation("fork-admission"),
            "decision": "fork",
            "prepared": evidence,
        }
    )
    before_rejection = await value.inspect_collaboration_namespace(context=CONTEXT)
    for field, replacement in (
        ("manifest_commitment", "sha256:" + "0" * 64),
        ("selected_view_commitment", "sha256:" + "0" * 64),
        ("view_id", "another-view"),
        ("source_session_id", "another-source"),
        ("source_session_instance_id", "another-incarnation"),
    ):
        changed = command.model_copy(
            update={
                "prepared": evidence.model_copy(
                    update={"target": evidence.target.model_copy(update={field: replacement})}
                )
            }
        )
        with pytest.raises(CollaborationConflict):
            await value.admit_collaboration_request(changed, context=resolver.recipient.context)
    assert await value.inspect_collaboration_namespace(context=CONTEXT) == before_rejection
    if lost_ack:
        store = native_stores[0]
        original = store._transaction
        key = (
            command.operation.namespace_incarnation,
            command.operation.generation,
            command.operation.caller_key,
        )
        committed = False

        @asynccontextmanager
        async def commit_then_raise(scope, *, write):
            nonlocal committed
            async with original(scope, write=write) as tx:
                yield tx
                if write and await tx.get("operations", key) is not None:
                    committed = True
            if write and committed:
                raise OSError("FORK admission acknowledgement lost")

        monkeypatch.setattr(store, "_transaction", commit_then_raise)
        with pytest.raises(CollaborationUnavailable):
            await value.admit_collaboration_request(command, context=resolver.recipient.context)
        monkeypatch.setattr(store, "_transaction", original)
        assert committed
    admitted = await value.admit_collaboration_request(command, context=resolver.recipient.context)
    assert admitted.state == "admitted" and admitted.command.prepared == evidence
    release = ContextViewOwnershipRequest(
        selection_key=source.selection_key,
        view_id=selected.view.view_id,
        pin_commitment=selected.pin_commitment,
        expected_state="adopted",
        expected_revision=retained.receipt.revision,
        operation="release",
        current_owner=base.recipient.owner,
        current_participant=base.recipient,
        operation_key="release-fork-" + source.selection_key,
    )
    assert (
        await receiver.release(blueprint.view, release, context=CONTEXT)
    ).receipt.state == "released"
    assert (
        await value.admit_collaboration_request(command, context=resolver.recipient.context)
        == admitted
    )
    reopened = native
    if native_stores[3][0] != "memory":
        await native.close()
        reopened = reopened_store(native_stores, tmp_path)
    try:
        application = app(
            native_stores[2](),
            value._participant_coordinator._registration,
            session_store=reopened,
            collaboration_requests=RequestRegistration(mandates=resolver, max_ttl_ms=300_000),
        )
        await application.initialize_collaboration()
        # Historical native child lookup must not require the now-released pin
        # or rebuild current provider/agent configuration to replay its receipt.
        assert await application.lookup_recipient_session(creation, context=CONTEXT) == (
            session,
            receipt,
        )
        assert await application.create_recipient_session(
            creation, context=CONTEXT, preparation=resolved
        ) == (session, receipt)
        assert len(provider.requests) == 1
        assert (
            await NativePlanningCreationOwner(application).create(stage_command, context=CONTEXT)
            == stage_readback
        )
        found = await application.collaboration_admission_reader().lookup(
            command, context=resolver.recipient.context
        )
        assert isinstance(found, ExactMatch) and found.receipt == admitted
    finally:
        if reopened is not native:
            await reopened.close()
