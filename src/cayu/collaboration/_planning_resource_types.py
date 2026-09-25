"""Retained resource-stage commands and positive native discharge evidence.

The planning operation is local to the request namespace. Resource commands and
permits retain their own complete native identities, including foreign owners;
they are never rewritten into the planner's namespace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import Field, model_validator

from cayu.artifacts._resource_material_types import ResourceMaterialReference
from cayu.artifacts.resources import (
    ResourceAcquisitionReceipt,
    ResourcePreparationReceipt,
    ResourceTransferReceipt,
    resource_operation_digest,
)
from cayu.collaboration._contracts import ContractValue, Identifier, OperationRef
from cayu.collaboration._permits import ReceivingSettlementReceipt
from cayu.collaboration.clarifications import Commitment
from cayu.collaboration.prepared_admission import NativeCommitment
from cayu.collaboration.requests import RequestCommand
from cayu.collaboration.resource_preparation import RequestPlanningResource

if TYPE_CHECKING:
    from cayu.collaboration._planning_creation_types import RequestCreationStageCommand
    from cayu.collaboration.planning import RequestPlanningRequest


class _ResourceStageCommand(ContractValue):
    operation: OperationRef
    expected: RequestCommand
    resource: RequestPlanningResource

    @model_validator(mode="after")
    def coherent(self):
        parent = self.expected.operation
        selected = self.expected.intent.selection
        recipient = selected.recipient
        receiving = self.resource.transfer_permit.intent.request
        if (
            self.operation
            in (
                parent,
                self.resource.acquisition.operation,
                self.resource.acquisition_permit.operation,
                self.resource.acquisition_permit.intent.request.settlement_operation,
                self.resource.transfer.operation,
                self.resource.transfer_permit.operation,
                self.resource.transfer_permit.intent.request.settlement_operation,
            )
            or (
                self.operation.application_scope,
                self.operation.namespace_incarnation,
                self.operation.generation,
            )
            != (parent.application_scope, parent.namespace_incarnation, parent.generation)
            or receiving.participant != recipient.reference
            or receiving.expected_lifecycle_revision != recipient.lifecycle_revision
            or receiving.expected_configuration_revision != recipient.configuration_revision
            or receiving.admission_generation != recipient.admission_generation
            or self.resource.acquisition.intent.deadline_at_ms is None
            or not selected.accepted_at_ms
            < self.resource.acquisition.intent.deadline_at_ms
            <= selected.expires_at_ms
        ):
            raise ValueError("Resource stage differs from its exact request recipient or bounds.")
        return self


class RequestResourceAcquisitionStageCommand(_ResourceStageCommand):
    mode: Literal["request_resource_acquisition"] = "request_resource_acquisition"

    @property
    def native_command(self):
        return self.resource.acquisition

    @property
    def permit(self):
        return self.resource.acquisition_permit


class RequestResourceTransferStageCommand(_ResourceStageCommand):
    mode: Literal["request_resource_transfer"] = "request_resource_transfer"
    acquisition: ResourceAcquisitionReceipt

    @model_validator(mode="after")
    def exact_acquisition(self):
        if (
            self.acquisition.command != self.resource.acquisition
            or self.acquisition.stage != "owned"
        ):
            raise ValueError("Resource transfer stage differs from its exact acquisition.")
        return self

    @property
    def native_command(self):
        return self.resource.transfer.bind(self.acquisition)

    @property
    def permit(self):
        return self.resource.transfer_permit


ResourceStageCommand = RequestResourceAcquisitionStageCommand | RequestResourceTransferStageCommand


@dataclass(frozen=True, slots=True)
class _ResourceAcquisitionReadback:
    expected: RequestPlanningResource
    receipt: ResourceAcquisitionReceipt


@dataclass(frozen=True, slots=True)
class _ResourceTransferReadback:
    expected: RequestPlanningResource
    receipt: ResourceTransferReceipt
    preparation: ResourcePreparationReceipt

    @property
    def reference(self) -> ResourceMaterialReference:
        from cayu.artifacts._resource_material import material_reference

        return material_reference(self.receipt, self.preparation)


@dataclass(frozen=True, slots=True)
class _ResourceCreationReadback:
    """Native material resolution for one exact retained planning operation."""

    expected: RequestPlanningRequest
    command: RequestCreationStageCommand


def resource_stage_command(record, index, *, redactor, acquisition=None):
    """Exact local stage identity; native operations retain their own namespace."""
    from hashlib import sha256

    from cayu.collaboration._preparation import contract_bytes
    from cayu.collaboration.planning import (
        RequestPlanningFork,
        RequestPlanningFresh,
        preparation_stage_count,
    )

    proposal = record.decision
    expected = record.receipt.command
    if (
        not isinstance(proposal, (RequestPlanningFresh, RequestPlanningFork))
        or type(index) is not int
        or not 0 <= index < len(proposal.resources)
        or len(proposal.resources) > expected.limits.max_resources
        or preparation_stage_count(proposal) > expected.limits.max_stages
    ):
        raise ValueError("Resource stage has no exact bounded frozen recipe.")
    resource = proposal.resources[index]
    if (
        resource.acquisition.intent.deadline_at_ms is None
        or resource.acquisition.intent.deadline_at_ms > expected.deadline_at_ms
    ):
        raise ValueError("Resource preparation outlives its planning deadline.")
    digest = sha256(contract_bytes(expected, redactor=redactor)).hexdigest()
    operation = expected.operation.model_copy(
        update={
            "caller_key": f"plan-resource-{'transfer' if acquisition is not None else 'acquire'}-"
            f"{index}-{digest}"
        }
    )
    fields = dict(operation=operation, expected=expected.expected, resource=resource)
    if acquisition is None:
        return RequestResourceAcquisitionStageCommand(**fields)
    return RequestResourceTransferStageCommand(**fields, acquisition=acquisition)


class RequestResourceStageRelease(ContractValue):
    """Exact permanent native fence, not absence or an observer's cancellation."""

    command: ResourceStageCommand = Field(discriminator="mode")
    receiving: ReceivingSettlementReceipt

    @model_validator(mode="after")
    def coherent(self):
        native = self.command.native_command
        permit = self.command.permit
        if (
            self.receiving.expected != permit
            or self.receiving.receiving_owner != permit.intent.request.target.owner
            or self.receiving.receipt_id != "resource-settled-" + resource_operation_digest(native)
            or self.receiving.outcome != "quiescent"
            or not self.receiving.admission_excluded
        ):
            raise ValueError("Resource stage lacks exact native discharge evidence.")
        return self


@dataclass(frozen=True, slots=True)
class _ResourceReleaseReadback:
    """Mint only after the registered native journal and permit owner settle."""

    receipt: RequestResourceStageRelease


class RequestResourceStageAdoption(ContractValue):
    """Planning handoff settled into a child, NOT release of its retained pin.

    The complete creation receipt belongs to the creation stage. These exact
    references must be corroborated against that stage on durable readback.
    Parsing this value alone never authenticates adoption.
    """

    command: RequestResourceTransferStageCommand
    creation_operation: OperationRef
    creation_sha256: Commitment
    session_id: Identifier
    session_instance_id: Identifier
    creation_receipt_commitment: NativeCommitment
    material: ResourceMaterialReference

    @model_validator(mode="after")
    def coherent(self):
        from cayu.artifacts._resource_material import material_commitment

        if (
            self.material.operation != self.command.resource.transfer.operation
            or self.material.owner != self.command.resource.transfer.destination
            or self.material.template_commitment
            != material_commitment(self.command.resource.transfer)
        ):
            raise ValueError("Resource adoption differs from its exact transfer template.")
        return self


def resource_stage_adoption(command, creation, *, redactor):
    """Project already-authenticated creation evidence; not an authority entrance."""
    from hashlib import sha256

    from cayu.collaboration._planning_creation_types import RequestCreationStageReceipt
    from cayu.collaboration._preparation import contract_bytes

    if not isinstance(creation, RequestCreationStageReceipt):
        raise ValueError("Resource adoption requires exact native creation evidence.")
    prepared = creation.prepared
    if (
        prepared is None
        or creation.command.expected != command.expected
        or prepared.recipient != command.resource.transfer_permit.intent.request.participant
    ):
        raise ValueError("Resource adoption has no matching created recipient.")
    matches = tuple(
        reference
        for reference in prepared.target.resources
        if reference.operation == command.resource.transfer.operation
        and reference.owner == command.resource.transfer.destination
    )
    if len(matches) != 1:
        raise ValueError("Created recipient did not adopt the exact resource operation.")
    return RequestResourceStageAdoption(
        command=command,
        creation_operation=creation.command.operation,
        creation_sha256=sha256(contract_bytes(creation, redactor=redactor)).hexdigest(),
        session_id=prepared.target.session_id,
        session_instance_id=prepared.target.session_instance_id,
        creation_receipt_commitment=prepared.target.creation_receipt_commitment,
        material=matches[0],
    )


@dataclass(frozen=True, slots=True)
class _ResourceAdoptionReadback:
    """Mint only after native child and accepted material readback agree."""

    receipt: RequestResourceStageAdoption
