"""Frozen FRESH preparation proposals; none of these values grants authority."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field, StrictStr, field_validator, model_validator

from cayu._validation import canonical_bounded_durable_json_bytes
from cayu.artifacts._resource_material_types import ResourceMaterialReference
from cayu.collaboration._contracts import (
    CollaborationContractError,
    ContractValue,
    Identifier,
    ObjectRef,
)
from cayu.collaboration.participants import VersionOne
from cayu.collaboration.prepared_admission import (
    MAX_PREPARED_BUDGET_BYTES,
    MAX_PREPARED_PROFILE_BYTES,
    NativeCommitment,
    _snapshot_document,
    prepared_budget,
    prepared_profile,
)
from cayu.sessions._context_selection_fence import ContextViewSelectionTarget
from cayu.sessions.context_views import RecipientSessionCreationRequest, json_commitment
from cayu.sessions.creation_fence import SessionCreationTarget

if TYPE_CHECKING:
    from cayu.sessions import RunRequest
    from cayu.vaults.redaction import SecretRedactor

MAX_PREPARATION_REQUEST_BYTES = 16 * 1024


def preparation_run_request(encoded: str) -> RunRequest:
    """Reconstruct bounded request data, without installing runtime authority."""
    from cayu.sessions import RunRequest

    document = _snapshot_document(encoded, max_bytes=MAX_PREPARATION_REQUEST_BYTES)
    result = None
    try:
        candidate = RunRequest.model_validate(document)
        canonical = canonical_bounded_durable_json_bytes(
            candidate.model_dump(mode="json", warnings=False),
            "recipient preparation request",
            max_bytes=MAX_PREPARATION_REQUEST_BYTES,
            max_nodes=8192,
            max_nesting=64,
        ).decode()
        if canonical == encoded:
            result = candidate
    except (TypeError, ValueError, RecursionError):
        pass
    if result is None:
        raise CollaborationContractError("Invalid recipient preparation request.")
    return result


class FreshRecipientPreparation(ContractValue):
    """Exact future creation, pinned native profile/definition and sponsor.

    SessionStore has not assigned an incarnation. The creation operation is the
    identity of this proposal; a requested public ID is never substituted for it.
    """

    schema_version: VersionOne = 1
    receiver: ObjectRef
    creation: SessionCreationTarget
    request_json: StrictStr
    execution_profile_json: StrictStr
    historical_definition_commitment: NativeCommitment
    budget_binding_json: StrictStr

    @field_validator("request_json")
    @classmethod
    def request_snapshot(cls, value: str) -> str:
        preparation_run_request(value)
        return value

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

    @property
    def creation_request(self) -> RecipientSessionCreationRequest:
        key = self.creation.creation_key
        if not key.startswith("recipient:"):
            raise CollaborationContractError("Preparation lacks a native recipient creation key.")
        return RecipientSessionCreationRequest(
            request=preparation_run_request(self.request_json),
            creation_key=key[len("recipient:") :],
            recipient=self.creation.permit.intent.request.participant,
        )

    @model_validator(mode="after")
    def coherent(self) -> FreshRecipientPreparation:
        request = self.creation_request.participant_request
        participant = self.creation.permit.intent.request.participant
        budget = prepared_budget(self.budget_binding_json)
        if (
            request.creation_key != self.creation.creation_key
            or request.request_commitment != self.creation.request_commitment
            or request.request.session_id != self.creation.requested_session_id
            or self.receiver.owner != participant.owner
            or self.receiver.revision is None
            or budget.application_scope != participant.owner.application_scope
            or json_commitment(self.execution_profile_json, "execution_profile")
            != self.creation.execution_identity_commitment
        ):
            raise CollaborationContractError("Recipient preparation identity conflicts.")
        return self


class ForkRecipientPreparation(ContractValue):
    """Frozen base preflight and exact view admission, before acquiring history.

    The base's unadmitted creation proposal freezes request/profile/definition
    and sponsor data. Its resource-free creation permit MUST NOT be registered
    for this FORK. The material-bound target is resolved and durably retained
    separately after native selection/adoption, before child creation.
    """

    schema_version: VersionOne = 1
    base: FreshRecipientPreparation
    view: ContextViewSelectionTarget

    @model_validator(mode="after")
    def coherent(self):
        recipient = self.base.creation.permit.intent.request
        receiving = (self.view.recipient_permit or self.view.permit).intent.request
        if (
            receiving.participant != recipient.participant
            or receiving.expected_lifecycle_revision != recipient.expected_lifecycle_revision
            or receiving.expected_configuration_revision
            != recipient.expected_configuration_revision
            or receiving.admission_generation != recipient.admission_generation
            or self.base.creation.permit.initiator.principal != self.view.permit.initiator.principal
            or self.view.permit.operation.application_scope
            != self.base.creation.permit.operation.application_scope
            or self.view.permit.operation.namespace_incarnation
            != self.base.creation.permit.operation.namespace_incarnation
            or self.view.permit.operation.generation
            != self.base.creation.permit.operation.generation
        ):
            raise CollaborationContractError("FORK blueprint authority conflicts.")
        return self


class ForkRetainedSelection(ContractValue):
    """Compact exact selection identity; large native material is not duplicated."""

    view_id: Identifier
    manifest_commitment: NativeCommitment
    pin_commitment: NativeCommitment
    receipt_commitment: NativeCommitment
    selection_key: Identifier
    source_session_id: Identifier
    source_session_instance_id: Identifier


def fork_retained_selection(blueprint, selected):
    from cayu.collaboration.prepared_admission import selected_view_commitment

    if (
        selected is None
        or selected.selection_key != blueprint.view.request.selection_key
        or selected.state != "adopted"
        or selected.ownership_revision != 2
        or selected.owner_participant != blueprint.view.recipient
        or selected.owner != blueprint.view.recipient.owner
        or selected.view.source_session_id != blueprint.view.request.source_session_id
        or selected.view.source_session_instance_id
        != blueprint.view.request.source_session_instance_id
    ):
        raise CollaborationContractError("FORK selection differs from its frozen blueprint.")
    return ForkRetainedSelection(
        view_id=selected.view.view_id,
        manifest_commitment=selected.view.manifest_commitment,
        pin_commitment=selected.pin_commitment,
        receipt_commitment=selected_view_commitment(selected),
        selection_key=selected.selection_key,
        source_session_id=selected.view.source_session_id,
        source_session_instance_id=selected.view.source_session_instance_id,
    )


def fork_blueprint_commitment(blueprint):
    return json_commitment(
        canonical_bounded_durable_json_bytes(
            blueprint.model_dump(mode="json", warnings=False),
            "FORK blueprint",
            max_bytes=64 * 1024,
            max_nodes=8192,
            max_nesting=64,
        ).decode()
    )


class MaterialRecipientCreationPreparation(ContractValue):
    """Resolved bounded native identities; large historical material stays native."""

    schema_version: VersionOne = 1
    base: FreshRecipientPreparation
    blueprint_commitment: NativeCommitment
    creation: SessionCreationTarget
    selection: ForkRetainedSelection | None = None
    resources: tuple[ResourceMaterialReference, ...] = Field(default=(), max_length=32)

    @property
    def receiver(self):
        return self.base.receiver

    @property
    def execution_profile_json(self):
        return self.base.execution_profile_json

    @property
    def historical_definition_commitment(self):
        return self.base.historical_definition_commitment

    @property
    def budget_binding_json(self):
        return self.base.budget_binding_json

    @model_validator(mode="after")
    def coherent(self):
        base = self.base.creation
        recipient = base.permit.intent.request
        actual = self.creation.permit.intent.request
        resource_ids = tuple((item.owner, item.operation) for item in self.resources)
        if (
            self.creation.creation_key != base.creation_key
            or self.creation.requested_session_id != base.requested_session_id
            or self.creation.receiving_owner != base.receiving_owner
            or self.creation.execution_identity_commitment != base.execution_identity_commitment
            or self.creation.permit.operation != base.permit.operation
            or self.creation.permit.initiator != base.permit.initiator
            or actual.participant != recipient.participant
            or actual.expected_lifecycle_revision != recipient.expected_lifecycle_revision
            or actual.expected_configuration_revision != recipient.expected_configuration_revision
            or actual.admission_generation != recipient.admission_generation
            or actual.source_operation != recipient.source_operation
            or actual.target != recipient.target
            or actual.settlement_operation != recipient.settlement_operation
            or self.creation.permit.intent.limits != base.permit.intent.limits
            or len(set(resource_ids)) != len(resource_ids)
            or any(item.owner != self.creation.receiving_owner for item in self.resources)
        ):
            raise CollaborationContractError(
                "Resolved material creation differs from its frozen blueprint."
            )
        return self


class ForkRecipientCreationPreparation(MaterialRecipientCreationPreparation):
    selection: ForkRetainedSelection


class ResourceRecipientCreationPreparation(MaterialRecipientCreationPreparation):
    """FRESH creation with authenticated material, but no inherited history."""

    selection: None = None
    resources: tuple[ResourceMaterialReference, ...] = Field(min_length=1, max_length=32)


def preparation_budget_request(*, target: SessionCreationTarget, profile: str):
    """Existing budget receiver input before native incarnation allocation."""
    return {
        "kind": "request_recipient_preparation",
        "schema_version": 1,
        "creation_target": target.model_dump(mode="json"),
        "execution_profile_fingerprint": prepared_profile(profile).fingerprint,
    }


def require_secret_free_preparation(value: FreshRecipientPreparation, redactor: SecretRedactor):
    for encoded, bound in (
        (value.request_json, MAX_PREPARATION_REQUEST_BYTES),
        (value.execution_profile_json, MAX_PREPARED_PROFILE_BYTES),
        (value.budget_binding_json, MAX_PREPARED_BUDGET_BYTES),
    ):
        document = _snapshot_document(encoded, max_bytes=bound)
        if redactor.redact_json(document) != document:
            raise CollaborationContractError("Recipient preparation contains a secret.")
