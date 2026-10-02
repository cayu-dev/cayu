"""Bounded prepared-recipient evidence; constructing a value grants no authority.

The registered receiving owner authenticates these values against native creation
and budget owners. Historical reconstruction never registers a new budget binding
or authorizes execution.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import Field, StrictStr, field_validator, model_validator

from cayu._validation import canonical_bounded_durable_json_bytes
from cayu.artifacts._resource_material_types import ResourceMaterialReference
from cayu.collaboration._contracts import (
    MAX_DEPTH,
    MAX_NODES,
    CollaborationContractError,
    ContractValue,
    Generation,
    Identifier,
    ObjectRef,
)
from cayu.collaboration.participants import ParticipantRef, VersionOne
from cayu.sessions._recipient_continuation import RecipientContinuationSelection
from cayu.sessions.creation_fence import SessionCreationTarget

if TYPE_CHECKING:
    from cayu.budgets.binding import BudgetBinding
    from cayu.execution_profiles import (
        ExecutionProfileIdentity,
    )
    from cayu.vaults.redaction import SecretRedactor

MAX_PREPARED_PROFILE_BYTES = 16 * 1024
MAX_PREPARED_BUDGET_BYTES = 8 * 1024
MAX_PREPARED_ADMISSION_BYTES = 48 * 1024


def require_prepared_budget_target(binding, *, provider_name, model, environment_name):
    """One target restriction check for native preflight and final receiving admission."""
    from cayu.collaboration.access import CollaborationAccessDenied

    for expected, actual in (
        (binding.provider_name, provider_name),
        (binding.model, model),
        (binding.environment_name, environment_name),
    ):
        if expected is not None and expected != actual:
            raise CollaborationAccessDenied("Prepared budget target conflicts.")


def prepared_budget_request(*, session_id: str, session_instance_id: str, profile: str):
    """One receiver input for proposal and verification; no caller admission authority."""
    return {
        "kind": "request_prepared_admission",
        "session_id": session_id,
        "session_instance_id": session_instance_id,
        "execution_profile_fingerprint": prepared_profile(profile).fingerprint,
    }


NativeCommitment = Annotated[StrictStr, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


def _snapshot_document(value: str, *, max_bytes: int) -> dict[str, Any]:
    """Decode only bounded JSON, with content-free errors outside exception handlers."""
    document = None
    try:
        if type(value) is str and len(value.encode("utf-8")) <= max_bytes:
            decoded = json.loads(value)
            if type(decoded) is dict:
                encoded = canonical_bounded_durable_json_bytes(
                    decoded,
                    "prepared admission snapshot",
                    max_bytes=max_bytes,
                    max_nodes=MAX_NODES,
                    max_nesting=MAX_DEPTH,
                ).decode("utf-8")
                if encoded == value:
                    document = decoded
    except (TypeError, ValueError, RecursionError):
        pass
    if document is None:
        raise CollaborationContractError("Invalid prepared admission snapshot.")
    return document


def prepared_profile(value: str) -> ExecutionProfileIdentity:
    """Reconstruct identity data using the existing profile validator, not authority."""
    from cayu.execution_profiles import (
        ExecutionProfileIdentity,
    )

    document = _snapshot_document(value, max_bytes=MAX_PREPARED_PROFILE_BYTES)
    profile = None
    try:
        profile = ExecutionProfileIdentity.model_validate(document)
        canonical = canonical_bounded_durable_json_bytes(
            profile.model_dump(mode="json"),
            "prepared profile",
            max_bytes=MAX_PREPARED_PROFILE_BYTES,
            max_nodes=MAX_NODES,
            max_nesting=MAX_DEPTH,
        ).decode("utf-8")
        if canonical != value:
            profile = None
    except (TypeError, ValueError, RecursionError):
        profile = None
    if profile is None:
        raise CollaborationContractError("Invalid prepared execution profile.")
    return profile


def prepared_budget(value: str) -> BudgetBinding:
    """Reconstruct the complete original sponsor binding without resolving new rights."""
    from cayu.budgets.binding import BudgetBinding

    document = _snapshot_document(value, max_bytes=MAX_PREPARED_BUDGET_BYTES)
    binding = None
    try:
        binding = BudgetBinding.model_validate(document)
        canonical = canonical_bounded_durable_json_bytes(
            binding.model_dump(mode="json"),
            "prepared budget",
            max_bytes=MAX_PREPARED_BUDGET_BYTES,
            max_nodes=MAX_NODES,
            max_nesting=MAX_DEPTH,
        ).decode("utf-8")
        if canonical != value:
            binding = None
    except (TypeError, ValueError, RecursionError):
        binding = None
    if binding is None:
        raise CollaborationContractError("Invalid prepared budget binding.")
    return binding


def prepared_budget_snapshot(binding: BudgetBinding) -> str:
    """Safely project receiver output, then validate it as bounded identity data.

    A trusted receiver may accidentally return a post-construction mutation.
    Serializer warnings contain raw input, independently of hide_input_in_errors.
    Suppressing them is safe only because the resulting primitive document must
    pass the authoritative budget validator and canonical round-trip below.
    Serialization failures likewise never escape with their original diagnostics.
    """
    from cayu.budgets.binding import BudgetBinding

    encoded = None
    try:
        if type(binding) is BudgetBinding:
            encoded = canonical_bounded_durable_json_bytes(
                binding.model_dump(mode="json", warnings=False),
                "prepared budget",
                max_bytes=MAX_PREPARED_BUDGET_BYTES,
                max_nodes=MAX_NODES,
                max_nesting=MAX_DEPTH,
            ).decode("utf-8")
    except (TypeError, ValueError, RecursionError):
        pass
    if encoded is None:
        raise CollaborationContractError("Invalid prepared budget binding.")
    prepared_budget(encoded)
    return encoded


class _CreatedRecipientAdmissionTarget(ContractValue):
    """Exact native creation identity and its resolved inert child."""

    creation: SessionCreationTarget
    session_id: Identifier
    session_instance_id: Identifier
    creation_receipt_commitment: NativeCommitment
    initial_input_commitment: NativeCommitment
    definition_commitment: NativeCommitment
    resources: tuple[ResourceMaterialReference, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def requested_identity(self):
        if (
            self.creation.requested_session_id is not None
            and self.creation.requested_session_id != self.session_id
        ):
            raise CollaborationContractError("Prepared session identity conflicts.")
        identities = tuple((item.owner, item.operation) for item in self.resources)
        if len(set(identities)) != len(identities) or any(
            item.owner != self.creation.receiving_owner for item in self.resources
        ):
            raise CollaborationContractError("Prepared resource identity conflicts.")
        return self


class FreshRecipientAdmissionTarget(_CreatedRecipientAdmissionTarget):
    kind: Literal["fresh"] = "fresh"


class ForkRecipientAdmissionTarget(_CreatedRecipientAdmissionTarget):
    """Exact native child and historical selection; never current disclosure rights."""

    kind: Literal["fork"] = "fork"
    selected_view_commitment: NativeCommitment
    manifest_commitment: NativeCommitment
    view_id: Identifier
    source_session_id: Identifier
    source_session_instance_id: Identifier


def selected_view_commitment(selected):
    from cayu.sessions.context_views import json_commitment

    return json_commitment(
        canonical_bounded_durable_json_bytes(
            selected.model_dump(mode="json", warnings=False),
            "selected view",
            max_bytes=512 * 1024,
            max_nodes=MAX_NODES,
            max_nesting=MAX_DEPTH,
        ).decode()
    )


def created_admission_target(creation, receipt):
    """Project authenticated native creation data; this function grants no authority."""
    from cayu.artifacts._resource_material import material_reference
    from cayu.sessions.context_views import ContextViewSelectionReceipt

    metadata = _snapshot_document(receipt.recipient_metadata_json, max_bytes=512 * 1024)
    definition = _snapshot_document(
        receipt.binding.historical_definition_json, max_bytes=256 * 1024
    )
    fields = dict(
        creation=creation,
        session_id=receipt.binding.session_id,
        session_instance_id=receipt.binding.session_instance_id,
        creation_receipt_commitment=receipt.receipt_commitment,
        initial_input_commitment=receipt.initial_input_commitment,
        definition_commitment=definition["agent_definition_commitment"],
    )
    from cayu.artifacts.resources import ResourcePreparationReceipt, ResourceTransferReceipt

    transfers = metadata.get("resource_transfers")
    preparations = metadata.get("preparation_receipts")
    if (
        type(transfers) is not list
        or type(preparations) is not list
        or len(transfers) != len(preparations)
        or len(transfers) > 32
    ):
        raise CollaborationContractError("Recipient resource evidence is malformed.")
    fields["resources"] = tuple(
        material_reference(
            ResourceTransferReceipt.model_validate(transfer),
            ResourcePreparationReceipt.model_validate(preparation),
        )
        for transfer, preparation in zip(transfers, preparations, strict=True)
    )
    if metadata.get("mode") == "fresh" and metadata.get("selected_view") is None:
        return FreshRecipientAdmissionTarget(**fields)
    if metadata.get("mode") != "fork":
        raise CollaborationContractError("Recipient admission mode is not qualified.")
    selected = ContextViewSelectionReceipt.model_validate(metadata.get("selected_view"))
    if (
        selected.state not in {"adopted", "transferred"}
        or selected.owner_participant != receipt.binding.participant
    ):
        raise CollaborationContractError("Historical selection contradicts its recipient.")
    return ForkRecipientAdmissionTarget(
        **fields,
        selected_view_commitment=selected_view_commitment(selected),
        manifest_commitment=selected.view.manifest_commitment,
        view_id=selected.view.view_id,
        source_session_id=selected.view.source_session_id,
        source_session_instance_id=selected.view.source_session_instance_id,
    )


class RecipientContinuationRequest(ContractValue):
    """Select one existing incarnation; neither a writer claim nor input append."""

    session_id: Identifier
    session_instance_id: Identifier
    participant: ParticipantRef


class ContinueRecipientAdmissionTarget(ContractValue):
    """Exact native completed boundary, independently checked by its receiver."""

    kind: Literal["continue"] = "continue"
    selection: RecipientContinuationSelection


class PreparedRecipientAdmission(ContractValue):
    """Immutable proposal authenticated only by a registered receiving owner.

    Target families require explicit receiver qualification. The envelope carries
    evidence, never a portable execution permit or writer lease.
    """

    schema_version: VersionOne = 1
    receiver: ObjectRef
    recipient: ParticipantRef
    lifecycle_revision: Generation
    configuration_revision: Generation
    admission_generation: Generation
    target: Annotated[
        FreshRecipientAdmissionTarget
        | ForkRecipientAdmissionTarget
        | ContinueRecipientAdmissionTarget,
        Field(discriminator="kind"),
    ]
    execution_profile_json: StrictStr
    budget_binding_json: StrictStr

    @field_validator("execution_profile_json")
    @classmethod
    def profile_snapshot(cls, value: str) -> str:
        prepared_profile(value)
        return value

    @field_validator("budget_binding_json")
    @classmethod
    def budget_snapshot(cls, value: str) -> str:
        prepared_budget(value)
        return value

    @model_validator(mode="after")
    def exact_recipient(self) -> PreparedRecipientAdmission:
        binding = prepared_budget(self.budget_binding_json)
        if (
            self.receiver.owner != self.recipient.owner
            or self.receiver.revision is None
            or self.recipient.owner.application_scope != binding.application_scope
        ):
            raise CollaborationContractError("Prepared recipient evidence conflicts.")
        if isinstance(self.target, _CreatedRecipientAdmissionTarget):
            registration = self.target.creation.permit.intent.request
            if (
                registration.participant != self.recipient
                or self.target.creation.receiving_owner.application_scope
                != self.recipient.owner.application_scope
            ):
                raise CollaborationContractError("Prepared recipient evidence conflicts.")
        elif (
            self.target.selection.participant != self.recipient
            or self.target.selection.execution_profile_json != self.execution_profile_json
        ):
            raise CollaborationContractError("Prepared continuation evidence conflicts.")
        return self


def require_secret_free_prepared(
    value: PreparedRecipientAdmission, redactor: SecretRedactor
) -> None:
    """Inspect decoded bounded snapshots so JSON escaping cannot hide secret values."""
    for encoded, bound in (
        (value.execution_profile_json, MAX_PREPARED_PROFILE_BYTES),
        (value.budget_binding_json, MAX_PREPARED_BUDGET_BYTES),
    ):
        decoded = _snapshot_document(encoded, max_bytes=bound)
        if redactor.redact_json(decoded) != decoded:
            raise CollaborationContractError("Prepared identity contains a secret.")
