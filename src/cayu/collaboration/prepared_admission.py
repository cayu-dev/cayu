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
from cayu.sessions.creation_fence import SessionCreationTarget

if TYPE_CHECKING:
    from cayu.budgets.binding import BudgetBinding
    from cayu.runtime.execution_profiles import ExecutionProfileIdentity
    from cayu.vaults.redaction import SecretRedactor

MAX_PREPARED_PROFILE_BYTES = 16 * 1024
MAX_PREPARED_BUDGET_BYTES = 8 * 1024
MAX_PREPARED_ADMISSION_BYTES = 48 * 1024


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
    from cayu.runtime.execution_profiles import ExecutionProfileIdentity

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


class FreshRecipientAdmissionTarget(ContractValue):
    """Exact native creation identity and its resolved inert child."""

    kind: Literal["fresh"] = "fresh"
    creation: SessionCreationTarget
    session_id: Identifier
    session_instance_id: Identifier
    creation_receipt_commitment: NativeCommitment
    initial_input_commitment: NativeCommitment
    definition_commitment: NativeCommitment

    @model_validator(mode="after")
    def requested_identity(self) -> FreshRecipientAdmissionTarget:
        if (
            self.creation.requested_session_id is not None
            and self.creation.requested_session_id != self.session_id
        ):
            raise CollaborationContractError("Prepared session identity conflicts.")
        return self


class PreparedRecipientAdmission(ContractValue):
    """Immutable proposal authenticated only by a registered receiving owner.

    Version one qualifies FRESH without resource/view preparation. Future target
    families require explicit native qualification, not a permissive fallback.
    """

    schema_version: VersionOne = 1
    receiver: ObjectRef
    recipient: ParticipantRef
    lifecycle_revision: Generation
    configuration_revision: Generation
    admission_generation: Generation
    target: FreshRecipientAdmissionTarget
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
        registration = self.target.creation.permit.intent.request
        binding = prepared_budget(self.budget_binding_json)
        if (
            registration.participant != self.recipient
            or self.receiver.owner != self.recipient.owner
            or self.receiver.revision is None
            or self.recipient.owner.application_scope != binding.application_scope
            or self.target.creation.receiving_owner.application_scope
            != self.recipient.owner.application_scope
        ):
            raise CollaborationContractError("Prepared recipient evidence conflicts.")
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
