"""Read-only native preparation before a planner retains creation responsibility."""

from dataclasses import replace

from pydantic import Field

from cayu._validation import canonical_bounded_durable_json_bytes
from cayu.artifacts._resource_material_types import ResourceMaterialReference
from cayu.collaboration._contracts import ContractValue
from cayu.collaboration._preparation import prepare_contract, require_exact_contract
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.prepared_admission import (
    prepared_budget,
    prepared_budget_snapshot,
    require_prepared_budget_target,
)
from cayu.collaboration.recipient_preparation import (
    MAX_PREPARATION_REQUEST_BYTES,
    ForkRecipientCreationPreparation,
    ForkRecipientPreparation,
    FreshRecipientPreparation,
    MaterialRecipientCreationPreparation,
    ResourceRecipientCreationPreparation,
    fork_blueprint_commitment,
    fork_retained_selection,
    preparation_budget_request,
    preparation_run_request,
    require_secret_free_preparation,
)
from cayu.sessions._participant_creation_preflight import prepare_participant_creation_material
from cayu.sessions._recipient_admission import _prepare_recipient_creation_target
from cayu.sessions.context_views import RecipientSessionCreationRequest, json_commitment


class _SelectedResources(ContractValue):
    resources: tuple[ResourceMaterialReference, ...] = Field(min_length=1, max_length=32)


def checked_creation_preparation(app, creation, proposal, *, context):
    """An expected tuple is not a shortcut around ordinary creation authority."""
    schema = (
        ForkRecipientCreationPreparation
        if type(proposal) is ForkRecipientCreationPreparation
        else (
            ResourceRecipientCreationPreparation
            if type(proposal) is ResourceRecipientCreationPreparation
            else FreshRecipientPreparation
        )
    )
    checked = prepare_contract(schema, proposal, redactor=app._secret_redactor)
    base = checked.base if isinstance(checked, MaterialRecipientCreationPreparation) else checked
    require_secret_free_preparation(base, app._secret_redactor)
    if isinstance(checked, MaterialRecipientCreationPreparation):
        expected = resolved_material_creation_request(
            checked,
            creation.selected_view,
            creation.resource_transfers,
            creation.preparation_receipts,
        )
        if creation.participant_request.request_commitment != checked.creation.request_commitment:
            raise ValueError("FORK material differs from its retained creation target.")
    else:
        expected = checked.creation_request
    if expected != creation or checked.creation.permit.initiator.principal != context.principal:
        raise ValueError("Recipient preparation differs from the creation request or creator.")
    return checked


async def require_prepared_creation_material(app, proposal, material):
    """Compare current preflight before the native handoff can mutate either store."""
    require_exact_contract(
        app._request_coordinator.prepared_receiver_ref(),
        proposal.receiver,
        redactor=app._secret_redactor,
    )
    if (
        material.profile_json != proposal.execution_profile_json
        or json_commitment(material.historical_definition_json)
        != proposal.historical_definition_commitment
        or json_commitment(material.initial_input_json, "initial_input")
        != proposal.creation.material_commitment
    ):
        raise ValueError("Resolved recipient preparation differs from the retained proposal.")
    binding = await app._run_limit_controller.inspect_budget_binding(
        request=preparation_budget_request(
            target=proposal.base.creation
            if isinstance(proposal, MaterialRecipientCreationPreparation)
            else proposal.creation,
            profile=proposal.execution_profile_json,
        )
    )
    if binding != prepared_budget(proposal.budget_binding_json):
        raise PermissionError("Recipient preparation sponsor authority changed.")
    require_prepared_budget_target(
        binding,
        provider_name=material.initial_run.session_identity.provider_name,
        model=material.initial_run.session_identity.model,
        environment_name=material.initial_run.request.environment_name,
    )


def require_created_preparation(proposal, receipt):
    """Historical creation replay compares native material, not current defaults."""
    if (
        receipt.execution_profile_json != proposal.execution_profile_json
        or json_commitment(receipt.binding.historical_definition_json)
        != proposal.historical_definition_commitment
        or receipt.initial_input_commitment != proposal.creation.material_commitment
        or receipt.binding.request_commitment != proposal.creation.request_commitment
    ):
        raise ValueError("Created recipient differs from the retained preparation.")
    if isinstance(proposal, MaterialRecipientCreationPreparation):
        from cayu.collaboration.prepared_admission import created_admission_target

        target = created_admission_target(proposal.creation, receipt)
        if target.resources != proposal.resources:
            raise ValueError("Created recipient differs from its retained resource material.")
    if isinstance(proposal, ForkRecipientCreationPreparation) and (
        target.kind != "fork"
        or target.selected_view_commitment != proposal.selection.receipt_commitment
    ):
        raise ValueError("Created recipient differs from its retained FORK selection.")


async def prepare_fresh_recipient(app, creation, *, context):
    if type(creation) is not RecipientSessionCreationRequest:
        raise TypeError("Recipient preparation requires a typed creation request.")
    creation = replace(creation)
    if creation.mode != "fresh" or creation.resource_transfers or creation.preparation_receipts:
        raise CollaborationUnavailable(
            "Base preparation cannot carry acquired history or resources."
        )
    # Initial attachment references are frozen input data, not retained material.
    # The planner must resolve their resource recipes; ordinary native creation
    # still proves complete transfer coverage and physical-store metadata under
    # its resource-owner fence before committing a child.
    receiver = app._request_coordinator.prepared_receiver_ref()
    participant = (
        await app._participant_coordinator.inspect(
            creation.recipient, context=context, action="administration"
        )
    ).participant
    if participant.reference != creation.recipient or participant.lifecycle != "active":
        raise PermissionError("Recipient preparation requires its active participant.")
    material = await prepare_participant_creation_material(app, creation.participant_request)
    target, _, _, _ = await _prepare_recipient_creation_target(
        app,
        creation.participant_request,
        creation.recipient,
        context,
        participant,
        json_commitment(material.initial_input_json, "initial_input"),
        json_commitment(material.profile_json, "execution_profile"),
    )
    # No placeholder session/incarnation: the exact existing native creation
    # operation is sufficient for a trusted sponsor's preparation decision.
    binding = await app._run_limit_controller.inspect_budget_binding(
        request=preparation_budget_request(target=target, profile=material.profile_json)
    )
    require_prepared_budget_target(
        binding,
        provider_name=material.initial_run.session_identity.provider_name,
        model=material.initial_run.session_identity.model,
        environment_name=material.initial_run.request.environment_name,
    )
    result = FreshRecipientPreparation(
        receiver=receiver,
        creation=target,
        request_json=canonical_bounded_durable_json_bytes(
            creation.request.model_dump(mode="json", warnings=False),
            "recipient preparation request",
            max_bytes=MAX_PREPARATION_REQUEST_BYTES,
            max_nodes=8192,
            max_nesting=64,
        ).decode(),
        execution_profile_json=material.profile_json,
        historical_definition_commitment=json_commitment(material.historical_definition_json),
        budget_binding_json=prepared_budget_snapshot(binding),
    )
    checked = prepare_contract(FreshRecipientPreparation, result, redactor=app._secret_redactor)
    require_secret_free_preparation(checked, app._secret_redactor)
    return checked


async def prepare_fork_recipient(
    app, creation, source, *, source_participant, context, deadline_at_ms
):
    """Freeze the existing base preflight before native selection or adoption."""
    from cayu.sessions._planning_view_owner import NativePlanningViewOwner

    base = await prepare_fresh_recipient(app, creation, context=context)
    view = await NativePlanningViewOwner(app).prepare(
        source,
        participant=source_participant,
        recipient=creation.recipient,
        context=context,
        deadline_at_ms=deadline_at_ms,
    )
    return prepare_contract(
        ForkRecipientPreparation,
        ForkRecipientPreparation(base=base, view=view),
        redactor=app._secret_redactor,
    )


def fork_creation_request(blueprint, retained, selected):
    """Rebuild expected input data; this helper never authenticates a receipt."""
    if (
        selected is None
        or selected.selection_key != blueprint.view.request.selection_key
        or selected.view.view_id != retained.selection.view_id
        or selected.view.manifest_commitment != retained.selection.manifest_commitment
        or selected.pin_commitment != retained.selection.pin_commitment
        or selected.owner != retained.owner
        or selected.owner_participant != retained.participant
        or selected.state != retained.state
        or selected.ownership_revision != retained.revision
        or selected.expires_at_ms != retained.expires_at_ms
    ):
        raise CollaborationUnavailable("FORK material differs from its exact retained pin.")
    base = blueprint.base
    return RecipientSessionCreationRequest(
        request=preparation_run_request(base.request_json),
        creation_key=base.creation.creation_key[len("recipient:") :],
        recipient=base.creation.permit.intent.request.participant,
        mode="fork",
        selected_view=selected,
    )


def resolved_fork_creation_request(proposal, selected):
    from cayu.collaboration.prepared_admission import selected_view_commitment

    if (
        selected is None
        or selected.state != "adopted"
        or selected.ownership_revision != 2
        or selected.owner_participant != proposal.base.creation.permit.intent.request.participant
        or selected.selection_key != proposal.selection.selection_key
        or selected.view.view_id != proposal.selection.view_id
        or selected.view.source_session_id != proposal.selection.source_session_id
        or selected.view.source_session_instance_id != proposal.selection.source_session_instance_id
        or selected.view.manifest_commitment != proposal.selection.manifest_commitment
        or selected.pin_commitment != proposal.selection.pin_commitment
        or selected_view_commitment(selected) != proposal.selection.receipt_commitment
    ):
        raise CollaborationUnavailable("FORK selection differs from its resolved creation.")
    base = proposal.base
    return RecipientSessionCreationRequest(
        request=preparation_run_request(base.request_json),
        creation_key=base.creation.creation_key[len("recipient:") :],
        recipient=base.creation.permit.intent.request.participant,
        mode="fork",
        selected_view=selected,
    )


def resolved_material_creation_request(proposal, selected, transfers=(), preparations=()):
    """Reconstruct expected data; native creation still authenticates every receipt."""
    from cayu.artifacts._resource_material import material_reference

    if (
        len(transfers) != len(preparations)
        or tuple(
            material_reference(transfer, preparation)
            for transfer, preparation in zip(transfers, preparations, strict=True)
        )
        != proposal.resources
    ):
        raise CollaborationUnavailable("Recipient material differs from its resolved preparation.")
    if isinstance(proposal, ForkRecipientCreationPreparation):
        base = resolved_fork_creation_request(proposal, selected)
    else:
        if selected is not None:
            raise CollaborationUnavailable("Fresh resource preparation cannot inherit history.")
        base = proposal.base.creation_request
    return replace(base, resource_transfers=transfers, preparation_receipts=preparations)


async def resolve_fork_recipient(app, blueprint, retained, *, context):
    """Resolve final material-bound identity without registering its permit."""
    from cayu.collaboration._contracts import ExactMatch
    from cayu.sessions._context_selection_fence import ContextViewRetentionEvidence

    blueprint = prepare_contract(ForkRecipientPreparation, blueprint, redactor=app._secret_redactor)
    retained = prepare_contract(
        ContextViewRetentionEvidence, retained, redactor=app._secret_redactor
    )
    found = await app.session_store._read_context_view_retention(blueprint.view)
    if not isinstance(found, ExactMatch) or found.receipt != retained:
        raise CollaborationUnavailable("Exact adopted FORK material is unavailable.")
    selected = await app.session_store.lookup_context_view_selection(
        blueprint.view.request.selection_key
    )
    creation = fork_creation_request(blueprint, retained, selected)
    readback = await app.read_context_view(
        selected.view.view_id,
        source_session_id=selected.view.source_session_id,
        participant=creation.recipient,
        context=context,
    )
    if readback.view != selected.view:
        raise CollaborationUnavailable("FORK source material changed during preparation.")
    participant = (
        await app._participant_coordinator.inspect(
            creation.recipient, context=context, action="administration"
        )
    ).participant
    material = await prepare_participant_creation_material(app, creation.participant_request)
    target, _, _, _ = await _prepare_recipient_creation_target(
        app,
        creation.participant_request,
        creation.recipient,
        context,
        participant,
        json_commitment(material.initial_input_json, "initial_input"),
        json_commitment(material.profile_json, "execution_profile"),
    )
    resolved = prepare_contract(
        ForkRecipientCreationPreparation,
        ForkRecipientCreationPreparation(
            base=blueprint.base,
            blueprint_commitment=fork_blueprint_commitment(blueprint),
            creation=target,
            selection=fork_retained_selection(blueprint, selected),
        ),
        redactor=app._secret_redactor,
    )
    await require_prepared_creation_material(app, resolved, material)
    return resolved


async def resolve_resource_recipient(app, proposal, resources, *, context):
    """Resolve a material-bound target through registered native journal readback.

    Neither the base creation permit nor this helper registers an operation.
    The final target must be retained before calling the ordinary creation owner.
    """
    from cayu.artifacts.resources import LocalArtifactResourceOwner

    schema = (
        ForkRecipientCreationPreparation
        if type(proposal) is ForkRecipientCreationPreparation
        else FreshRecipientPreparation
    )
    proposal = prepare_contract(schema, proposal, redactor=app._secret_redactor)
    resources = prepare_contract(
        _SelectedResources, {"resources": resources}, redactor=app._secret_redactor
    ).resources
    base = proposal.base if isinstance(proposal, ForkRecipientCreationPreparation) else proposal
    recipient = base.creation.permit.intent.request.participant
    participant = (
        await app._participant_coordinator.inspect(
            recipient, context=context, action="administration"
        )
    ).participant
    if participant.lifecycle != "active":
        raise PermissionError("Resource preparation requires its active recipient.")
    if len({(item.owner, item.operation) for item in resources}) != len(resources) or any(
        item.owner != recipient.owner for item in resources
    ):
        raise CollaborationUnavailable("Recipient resource references conflict.")
    owner = app._request_coordinator._resource_owners.get(recipient.owner)
    if not isinstance(owner, LocalArtifactResourceOwner):
        raise CollaborationUnavailable("Recipient resource owner is not qualified.")
    materials = [await owner.read_material(reference) for reference in resources]
    if isinstance(proposal, ForkRecipientCreationPreparation):
        if proposal.resources:
            raise CollaborationUnavailable("Recipient material was already resolved.")
        selected = await app.session_store.lookup_context_view_selection(
            proposal.selection.selection_key
        )
        creation = resolved_fork_creation_request(proposal, selected)
    else:
        creation = base.creation_request
    creation = replace(
        creation,
        resource_transfers=tuple(item.transfer for item in materials),
        preparation_receipts=tuple(item.preparation for item in materials),
    )
    material = await prepare_participant_creation_material(app, creation.participant_request)
    target, _, _, _ = await _prepare_recipient_creation_target(
        app,
        creation.participant_request,
        recipient,
        context,
        participant,
        json_commitment(material.initial_input_json, "initial_input"),
        json_commitment(material.profile_json, "execution_profile"),
    )
    if isinstance(proposal, ForkRecipientCreationPreparation):
        resolved = prepare_contract(
            ForkRecipientCreationPreparation,
            proposal.model_copy(update={"resources": resources, "creation": target}),
            redactor=app._secret_redactor,
        )
    else:
        resolved = prepare_contract(
            ResourceRecipientCreationPreparation,
            ResourceRecipientCreationPreparation(
                base=base,
                blueprint_commitment=fork_blueprint_commitment(base),
                creation=target,
                resources=resources,
            ),
            redactor=app._secret_redactor,
        )
    await require_prepared_creation_material(app, resolved, material)
    return resolved
