"""Native read-only target preparation remains separate from creation admission."""

import pytest
from tests.core.test_participant_identity import CONTEXT, app
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_prepared_admission_public import prepared_scenario

from cayu import Message, RunRequest
from cayu.agents import AgentSpec
from cayu.collaboration._contracts import CollaborationConflict, ExactNotFound
from cayu.collaboration._planning_creation_types import (
    RequestCreationStageCommand,
    RequestCreationStageReceipt,
)
from cayu.collaboration.prepared_admission import prepared_budget
from cayu.collaboration.recipient_preparation import FreshRecipientPreparation
from cayu.sessions._participant_creation_preflight import prepare_participant_creation_material
from cayu.sessions._planning_creation_owner import NativePlanningCreationOwner
from cayu.sessions._recipient_admission import (
    _prepare_recipient_creation_target,
    admit_recipient_creation,
)
from cayu.sessions.context_views import RecipientSessionCreationRequest, json_commitment

pytestmark = pytest.mark.anyio


async def test_public_preparation_request_size_boundary_is_inert(native_stores):
    from cayu._validation import canonical_durable_json_bytes
    from cayu.collaboration.recipient_preparation import MAX_PREPARATION_REQUEST_BYTES

    application, _, command, provider, _, _ = await prepared_scenario(native_stores)
    baseline = RunRequest(agent_name="reviewer", messages=[Message.text("user", "x")])
    overhead = len(canonical_durable_json_bytes(baseline.model_dump(mode="json"), "request")) - 1
    before = await application.inspect_collaboration_namespace(context=CONTEXT)
    for offset in (-1, 0, 1):
        size = MAX_PREPARATION_REQUEST_BYTES + offset
        creation = RecipientSessionCreationRequest(
            request=baseline.model_copy(
                update={"messages": [Message.text("user", "x" * (size - overhead))]}
            ),
            creation_key=f"bounded-preparation-{offset}",
            recipient=command.prepared.recipient,
        )
        assert (
            len(canonical_durable_json_bytes(creation.request.model_dump(mode="json"), "request"))
            == size
        )
        if offset > 0:
            with pytest.raises(ValueError):
                await application.prepare_recipient_creation(creation, context=CONTEXT)
        else:
            proposal = await application.prepare_recipient_creation(creation, context=CONTEXT)
            assert len(proposal.request_json.encode()) == size
            assert proposal.creation_request == creation
            assert isinstance(
                await native_stores[1].read_session_creation_decision(proposal.creation),
                ExactNotFound,
            )
        assert await application.lookup_recipient_session(creation, context=CONTEXT) is None
        assert await application.inspect_collaboration_namespace(context=CONTEXT) == before
    assert provider.requests == []


@pytest.mark.parametrize("changed", ["definition", "sponsor"])
async def test_public_creation_rejects_changed_preflight_before_handoff(native_stores, changed):
    original, _, command, provider, _, initialized = await prepared_scenario(native_stores)
    creation = RecipientSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[Message.text("user", "input")]),
        creation_key="changed-preflight:" + initialized.owner.application_scope,
        recipient=command.prepared.recipient,
    )
    proposal = await original.prepare_recipient_creation(creation, context=CONTEXT)
    binding = prepared_budget(proposal.budget_binding_json)

    class BudgetReceiver:
        async def resolve_budget_binding(self, *, request):
            return (
                binding.model_copy(update={"sponsor": "another-sponsor"})
                if changed == "sponsor"
                else binding
            )

    reopened = app(
        native_stores[2](),
        original._participant_coordinator._registration,
        session_store=native_stores[1],
        collaboration_requests=original._request_coordinator._registration,
        budget_binding_receiver=BudgetReceiver(),
        enable_common_root_budget_binding=True,
    )
    reopened.register_provider(provider, default=True)
    reopened.register_agent(
        AgentSpec(
            name="reviewer",
            model="model",
            system_prompt="changed" if changed == "definition" else "system",
        )
    )
    await reopened.initialize_collaboration()
    with pytest.raises((ValueError, PermissionError)):
        await reopened.create_recipient_session(creation, context=CONTEXT, preparation=proposal)
    assert isinstance(
        await native_stores[1].read_session_creation_decision(proposal.creation), ExactNotFound
    )
    assert (
        await native_stores[1].lookup_participant_session_creation(creation.participant_request)
        is None
    )
    assert (
        await native_stores[0]._lookup_registered_permit(
            initialized, proposal.creation.permit.operation, redactor=original._secret_redactor
        )
        is None
    )
    assert provider.requests == []


async def test_read_only_creation_target_matches_public_native_handoff(native_stores, monkeypatch):
    application, _, command, provider, _, initialized = await prepared_scenario(native_stores)
    participant = command.prepared.recipient
    creation = RecipientSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[Message.text("user", "input")]),
        creation_key="preflight-child:" + initialized.owner.application_scope,
        recipient=participant,
    )
    snapshot = (await application.inspect_participant(participant, context=CONTEXT)).participant
    store = application.session_store

    async def no_receiving_write(*args, **kwargs):
        pytest.fail("Read-only target preparation attempted a receiving mutation.")

    with monkeypatch.context() as patch:
        patch.setattr(store, "_prepare_session_creation_target", no_receiving_write)
        patch.setattr(store, "_register_session_creation_target", no_receiving_write)
        patch.setattr(native_stores[0], "_register_permit", no_receiving_write)
        proposal = await application.prepare_recipient_creation(creation, context=CONTEXT)
        material = await prepare_participant_creation_material(
            application, creation.participant_request
        )
        (
            target,
            registered,
            source_store,
            source_initialization,
        ) = await _prepare_recipient_creation_target(
            application,
            creation.participant_request,
            participant,
            CONTEXT,
            snapshot,
            json_commitment(material.initial_input_json, "initial_input"),
            json_commitment(material.profile_json, "execution_profile"),
        )
    assert registered is None
    assert proposal.creation == target
    assert proposal.creation_request == creation
    assert FreshRecipientPreparation.model_validate_json(proposal.model_dump_json()) == proposal
    assert proposal.historical_definition_commitment == json_commitment(
        material.historical_definition_json
    )
    assert source_store is native_stores[0] and source_initialization == initialized
    assert isinstance(await store.read_session_creation_decision(target), ExactNotFound)
    assert await store.lookup_participant_session_creation(creation.participant_request) is None
    assert (
        await native_stores[0]._lookup_registered_permit(
            initialized, target.permit.operation, redactor=application._secret_redactor
        )
        is None
    )

    # Every decision-bearing target field is compared before either store can
    # acquire responsibility. These constructor-valid values are not authority.
    with monkeypatch.context() as patch:
        patch.setattr(store, "_prepare_session_creation_target", no_receiving_write)
        patch.setattr(store, "_register_session_creation_target", no_receiving_write)
        patch.setattr(native_stores[0], "_register_permit", no_receiving_write)
        for field, value in (
            ("creation_key", "different-key"),
            ("requested_session_id", "different-session"),
            ("request_commitment", "sha256:" + "a" * 64),
            ("material_commitment", "sha256:" + "b" * 64),
            ("execution_identity_commitment", "sha256:" + "c" * 64),
        ):
            with pytest.raises(CollaborationConflict):
                await admit_recipient_creation(
                    application,
                    creation.participant_request,
                    participant,
                    CONTEXT,
                    snapshot,
                    target.material_commitment,
                    target.execution_identity_commitment,
                    expected_target=target.model_copy(update={field: value}),
                )

    with monkeypatch.context() as patch:
        patch.setattr(store, "_prepare_session_creation_target", no_receiving_write)
        patch.setattr(store, "_register_session_creation_target", no_receiving_write)
        patch.setattr(native_stores[0], "_register_permit", no_receiving_write)
        with pytest.raises(ValueError, match="Resolved recipient preparation"):
            await application.create_recipient_session(
                creation,
                context=CONTEXT,
                preparation=proposal.model_copy(
                    update={"historical_definition_commitment": "sha256:" + "d" * 64}
                ),
            )
    session, receipt = await application.create_recipient_session(
        creation, context=CONTEXT, preparation=proposal
    )
    assert await application.create_recipient_session(
        creation, context=CONTEXT, preparation=proposal
    ) == (session, receipt)
    prepared = await application.prepare_recipient_admission(creation, context=CONTEXT)
    assert prepared.target.creation == target
    assert prepared.execution_profile_json == material.profile_json
    assert prepared.target.session_id == session.id
    assert session.status == "pending" and provider.requests == []
    native = await store.read_session_creation_decision(target)
    stage_command = RequestCreationStageCommand(
        operation=target.permit.operation, expected=command.expected, preparation=proposal
    )
    evidence = RequestCreationStageReceipt(
        command=stage_command,
        decision=native.receipt,
        definition_commitment=prepared.target.definition_commitment,
    )
    assert evidence.prepared == prepared
    owned = await NativePlanningCreationOwner(application).read(stage_command)
    assert owned.receipt == evidence
    from cayu.collaboration._planning_creation_evidence import require_creation_source_settlement

    async with native_stores[0]._transaction(
        initialized.owner.application_scope, write=False
    ) as tx:
        await require_creation_source_settlement(
            tx, owned.receipt, redactor=application._secret_redactor
        )
    reconstructed = RequestCreationStageReceipt.model_validate_json(evidence.model_dump_json())
    assert reconstructed == evidence and reconstructed.prepared == prepared
    assert "prepared" not in evidence.model_dump()
    with pytest.raises(ValueError):
        RequestCreationStageReceipt(
            command=stage_command,
            decision=native.receipt.model_copy(update={"settlement_acknowledged": False}),
            definition_commitment=prepared.target.definition_commitment,
        )
