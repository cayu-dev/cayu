"""Exact retained FORK handoff and bounded terminal retention evidence."""

from dataclasses import dataclass
from typing import Literal

from pydantic import model_validator

from cayu.collaboration._contracts import ContractValue, Identifier, OperationRef
from cayu.collaboration.prepared_admission import NativeCommitment
from cayu.collaboration.recipient_preparation import (
    ForkRecipientCreationPreparation,
    ForkRecipientPreparation,
)
from cayu.collaboration.requests import RequestCommand


class RequestViewStageCommand(ContractValue):
    mode: Literal["request_context_view"] = "request_context_view"
    operation: OperationRef
    expected: RequestCommand
    preparation: ForkRecipientPreparation

    @model_validator(mode="after")
    def coherent(self):
        view = self.preparation.view
        recipient = self.expected.intent.selection.recipient
        receiving = (view.recipient_permit or view.permit).intent.request
        if (
            self.operation != view.permit.operation
            or receiving.participant != recipient.reference
            or receiving.expected_lifecycle_revision != recipient.lifecycle_revision
            or receiving.expected_configuration_revision != recipient.configuration_revision
            or receiving.admission_generation != recipient.admission_generation
            or view.deadline_at_ms > self.expected.intent.selection.expires_at_ms
        ):
            raise ValueError("View stage differs from the exact request recipient.")
        return self


class RequestViewStageReceipt(ContractValue):
    """Positive terminal pin evidence, not a selectable manifest or disclosure."""

    command: RequestViewStageCommand
    state: Literal["excluded", "released", "expired"]
    view_id: Identifier | None = None
    manifest_commitment: NativeCommitment | None = None
    pin_commitment: NativeCommitment | None = None

    @model_validator(mode="after")
    def coherent(self):
        fields = (self.view_id, self.manifest_commitment, self.pin_commitment)
        if (self.state == "excluded" and any(value is not None for value in fields)) or (
            self.state != "excluded" and any(value is None for value in fields)
        ):
            raise ValueError("Terminal view evidence contradicts its disposition.")
        return self


@dataclass(frozen=True, slots=True)
class _ForkPreparationReadback:
    """Minted only by the application-wired native resolution owner."""

    command: RequestViewStageCommand
    preparation: ForkRecipientCreationPreparation


@dataclass(frozen=True, slots=True)
class _ViewReadback:
    """Minted only after native pin settlement and source-permit reconciliation."""

    receipt: RequestViewStageReceipt


def view_stage_command(record):
    from cayu.collaboration.planning import RequestPlanningFork

    if not isinstance(record.decision, RequestPlanningFork):
        raise ValueError("Planning decision has no exact view preparation.")
    preparation = record.decision.preparation
    if (
        preparation.view.permit.initiator.principal != record.receipt.command.initiator.principal
        or preparation.view.deadline_at_ms > record.receipt.command.deadline_at_ms
    ):
        raise ValueError("View preparation differs from the planning initiator or deadline.")
    return RequestViewStageCommand(
        operation=preparation.view.permit.operation,
        expected=record.receipt.command.expected,
        preparation=preparation,
    )
