"""Exact foreign-stage data; only native receiving readback authenticates it."""

from typing import Literal

from pydantic import model_validator

from cayu.collaboration._contracts import ContractValue, OperationRef
from cayu.collaboration.prepared_admission import (
    ForkRecipientAdmissionTarget,
    FreshRecipientAdmissionTarget,
    NativeCommitment,
    PreparedRecipientAdmission,
)
from cayu.collaboration.recipient_preparation import (
    ForkRecipientCreationPreparation,
    FreshRecipientPreparation,
    MaterialRecipientCreationPreparation,
    ResourceRecipientCreationPreparation,
    fork_blueprint_commitment,
)
from cayu.collaboration.requests import RequestCommand
from cayu.sessions.creation_fence import SessionCreationDecision


class RequestCreationStageCommand(ContractValue):
    mode: Literal["request_recipient_creation"] = "request_recipient_creation"
    operation: OperationRef
    expected: RequestCommand
    preparation: (
        FreshRecipientPreparation
        | ForkRecipientCreationPreparation
        | ResourceRecipientCreationPreparation
    )

    @model_validator(mode="after")
    def coherent(self):
        target = self.preparation.creation
        selected = self.expected.intent.selection
        if (
            self.operation != target.permit.operation
            or target.permit.intent.request.participant != selected.recipient.reference
            or target.permit.intent.request.expected_lifecycle_revision
            != selected.recipient.lifecycle_revision
            or target.permit.intent.request.expected_configuration_revision
            != selected.recipient.configuration_revision
            or target.permit.intent.request.admission_generation
            != selected.recipient.admission_generation
            or target.receiving_owner != selected.reference.owner
            or self.operation == self.expected.operation
            or (
                self.operation.application_scope,
                self.operation.namespace_incarnation,
                self.operation.generation,
            )
            != (
                self.expected.operation.application_scope,
                self.expected.operation.namespace_incarnation,
                self.expected.operation.generation,
            )
        ):
            raise ValueError("Creation stage differs from its exact receiving operation.")
        return self


def creation_stage_command(record, *, preparation=None):
    from cayu.collaboration.planning import RequestPlanningFork, RequestPlanningFresh

    if not isinstance(record.decision, (RequestPlanningFresh, RequestPlanningFork)):
        raise ValueError("Planning decision has no native creation proposal.")
    expected = record.receipt.command
    if isinstance(record.decision, RequestPlanningFork):
        if (
            not isinstance(preparation, ForkRecipientCreationPreparation)
            or preparation.base != record.decision.preparation.base
            or preparation.blueprint_commitment
            != fork_blueprint_commitment(record.decision.preparation)
        ):
            raise ValueError("FORK creation lacks its exact resolved blueprint.")
        proposal = preparation
    elif record.decision.resources:
        if (
            not isinstance(preparation, ResourceRecipientCreationPreparation)
            or preparation.base != record.decision.preparation
            or preparation.blueprint_commitment
            != fork_blueprint_commitment(record.decision.preparation)
        ):
            raise ValueError("Resource creation lacks its exact resolved blueprint.")
        proposal = preparation
    else:
        proposal = record.decision.preparation
    from cayu.artifacts._resource_material import material_commitment

    references = (
        proposal.resources if isinstance(proposal, MaterialRecipientCreationPreparation) else ()
    )
    if len(references) != len(record.decision.resources) or any(
        reference.owner != recipe.transfer.destination
        or reference.operation != recipe.transfer.operation
        or reference.template_commitment != material_commitment(recipe.transfer)
        for reference, recipe in zip(references, record.decision.resources, strict=True)
    ):
        raise ValueError("Resolved creation differs from its frozen resource recipes.")
    if proposal.creation.permit.initiator.principal != expected.initiator.principal:
        raise ValueError("Planning creator differs from its authenticated initiator.")
    return RequestCreationStageCommand(
        operation=proposal.creation.permit.operation,
        expected=expected.expected,
        preparation=proposal,
    )


class RequestCreationStageReceipt(ContractValue):
    """Positive created/excluded native evidence, never absence or timeout."""

    command: RequestCreationStageCommand
    decision: SessionCreationDecision
    definition_commitment: NativeCommitment | None

    @property
    def prepared(self) -> PreparedRecipientAdmission | None:
        """Derive admission data once from authoritative command/child evidence.

        Persisting a second complete profile/budget/creation tuple both duplicates
        authority and consumes the bounded terminal envelope unnecessarily.
        """
        if self.decision.state != "created":
            return None
        proposal = self.command.preparation
        target = proposal.creation
        registration = target.permit.intent.request
        assert registration.expected_configuration_revision is not None
        fields = dict(
            creation=target,
            session_id=self.decision.session_id,
            session_instance_id=self.decision.session_instance_id,
            creation_receipt_commitment=self.decision.creation_receipt_commitment,
            initial_input_commitment=target.material_commitment,
            definition_commitment=self.definition_commitment,
            resources=proposal.resources
            if isinstance(proposal, MaterialRecipientCreationPreparation)
            else (),
        )
        if isinstance(proposal, ForkRecipientCreationPreparation):
            selection = proposal.selection
            prepared_target = ForkRecipientAdmissionTarget.model_validate(
                {
                    **fields,
                    "selected_view_commitment": selection.receipt_commitment,
                    "manifest_commitment": selection.manifest_commitment,
                    "view_id": selection.view_id,
                    "source_session_id": selection.source_session_id,
                    "source_session_instance_id": selection.source_session_instance_id,
                }
            )
        else:
            prepared_target = FreshRecipientAdmissionTarget.model_validate(fields)
        return PreparedRecipientAdmission(
            receiver=proposal.receiver,
            recipient=registration.participant,
            lifecycle_revision=registration.expected_lifecycle_revision,
            configuration_revision=registration.expected_configuration_revision,
            admission_generation=registration.admission_generation,
            target=prepared_target,
            execution_profile_json=proposal.execution_profile_json,
            budget_binding_json=proposal.budget_binding_json,
        )

    @model_validator(mode="after")
    def coherent(self):
        target = self.command.preparation.creation
        decision = self.decision
        if (
            decision.target != target
            or decision.state == "pending"
            or not decision.settlement_acknowledged
            or (decision.state == "created") != (self.definition_commitment is not None)
        ):
            raise ValueError("Creation stage lacks exact settled native evidence.")
        # Validate the derived public/native target before storing its ingredients.
        _ = self.prepared
        return self


async def prepared_creation_from_stage(tx, record, *, redactor):
    """Derive final admission data only from the retained authenticated first stage."""
    from cayu.collaboration._planning_records import RequestPlanningStageRecord
    from cayu.collaboration._planning_stages import read_stage
    from cayu.collaboration._preparation import prepare_contract, require_exact_contract
    from cayu.collaboration.planning import preparation_stage_count

    ordinal = preparation_stage_count(record.decision) - 1
    stages = await tx.scan_request_plan_stages(
        record.receipt.command.operation, limit=record.receipt.command.limits.max_stages + 1
    )
    if len(stages) < ordinal:
        return None
    first = prepare_contract(RequestPlanningStageRecord, stages[ordinal - 1], redactor=redactor)
    if not isinstance(first.intent.command, RequestCreationStageCommand):
        raise ValueError("Creation stage has another native command.")
    require_exact_contract(
        creation_stage_command(record, preparation=first.intent.command.preparation),
        first.intent.command,
        redactor=redactor,
    )
    first = await read_stage(tx, first.intent, redactor=redactor)
    if first is None or first.receipt is None:
        return None
    if not isinstance(first.receipt, RequestCreationStageReceipt):
        raise ValueError("Planning creation has another receiving receipt.")
    return first.receipt.prepared
