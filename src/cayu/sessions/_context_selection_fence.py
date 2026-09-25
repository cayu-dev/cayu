"""Native selection/exclusion decisions; none of these values grants access.

The source owner authenticates cleanup before using the private mutation entrance.
Selection and exclusion must compare the same complete request under the native
selection transaction. A selected decision does not release or transfer its pin.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, StrictBool, StrictInt, model_validator

from cayu._validation import canonical_bounded_durable_json_bytes
from cayu.collaboration._contracts import ContractValue, Generation, Identifier, ObjectRef, OwnerRef
from cayu.collaboration._permits import PermitCommand
from cayu.collaboration.participants import ParticipantRef
from cayu.sessions.context_views import (
    ContextViewOwnershipRequest,
    ContextViewSelectionReceipt,
    ContextViewSelectionRequest,
    json_commitment,
)

_CONTEXT_SELECTION_AUTHORITY = object()
CONTEXT_SELECTION_MAX_CONTROLS_PER_OWNER = 4096


class ContextViewSelectionExcluded(PermissionError):
    """The exact selection was durably refused before any pin was acquired."""


class ContextViewSelectionConflict(ValueError):
    """A selection key was reused with a different complete request."""


def selection_admission_commitment(request, deadline_at_ms, recipient_permit=None):
    from hashlib import sha256

    return sha256(
        canonical_bounded_durable_json_bytes(
            {
                "request": request.model_dump(mode="json"),
                "deadline_at_ms": deadline_at_ms,
                "recipient_permit": None
                if recipient_permit is None
                else recipient_permit.model_dump(mode="json"),
            },
            "context selection admission",
            max_bytes=256 * 1024,
            max_nodes=8192,
            max_nesting=64,
        )
    ).hexdigest()


def selection_receiving_target(request):
    return ObjectRef(
        owner=request.source_owner,
        kind="context_view_selection",
        object_id=request.selection_key,
        incarnation=request.source_session_instance_id,
        revision=1,
    )


class ContextViewSelectionTarget(ContractValue):
    """Exact participant responsibility; this document alone grants nothing."""

    schema_version: Literal[1] = 1
    request: ContextViewSelectionRequest
    permit: PermitCommand
    deadline_at_ms: StrictInt = Field(ge=1, le=2**53 - 1)
    recipient_permit: PermitCommand | None = None

    @model_validator(mode="after")
    def coherent(self):
        registered = self.permit.intent.request
        if (
            registered.participant.owner != self.request.source_owner
            or registered.target != selection_receiving_target(self.request)
            or registered.target_state != "existing"
            or registered.effect_scope != "context_view_selection"
            or registered.required_settlement != "exclusion"
            or registered.expected_configuration_revision is None
            or registered.admission_commitment
            != selection_admission_commitment(
                self.request, self.deadline_at_ms, self.recipient_permit
            )
        ):
            raise ValueError("Context-view selection permit conflicts with its exact target.")
        if self.recipient_permit is not None:
            recipient = self.recipient_permit.intent.request
            if (
                recipient.target != registered.target
                or recipient.target_state != "existing"
                or recipient.effect_scope != "context_view_retention"
                or recipient.required_settlement != "exclusion"
                or recipient.expected_configuration_revision is None
                or recipient.participant.owner != self.request.source_owner
                or recipient.admission_commitment
                != selection_admission_commitment(self.request, self.deadline_at_ms)
                or self.recipient_permit.operation == self.permit.operation
                or self.recipient_permit.initiator.principal != self.permit.initiator.principal
            ):
                raise ValueError("Context-view recipient retention authority conflicts.")
        if len(self.model_dump_json().encode()) > 48 * 1024:
            raise ValueError("Context-view selection target exceeds its durable bound.")
        return self

    @property
    def recipient(self):
        return (self.recipient_permit or self.permit).intent.request.participant

    @property
    def permits(self):
        return (
            (self.permit,)
            if self.recipient_permit is None
            else (self.permit, self.recipient_permit)
        )


class ContextViewSelectionDecision(ContractValue):
    """Bounded receiving evidence, independent of mutable pin ownership state."""

    schema_version: Literal[1] = 1
    request: ContextViewSelectionRequest
    state: Literal["reserved", "selected", "excluded"]
    view_id: Identifier | None = None
    manifest_commitment: Identifier | None = None
    pin_commitment: Identifier | None = None
    target: ContextViewSelectionTarget | None = None
    responsibility_registered: StrictBool = False

    @model_validator(mode="after")
    def coherent(self) -> ContextViewSelectionDecision:
        if (
            (self.target is None and self.responsibility_registered)
            or (self.target is not None and self.target.request != self.request)
            or (
                self.target is not None
                and self.state == "selected"
                and not self.responsibility_registered
            )
        ):
            raise ValueError("Context-view selection responsibility conflicts.")
        fields = (self.view_id, self.manifest_commitment, self.pin_commitment)
        if (
            any(value is None for value in fields)
            if self.state == "selected"
            else any(value is not None for value in fields)
        ):
            raise ValueError("Context-view selection decision identity conflicts.")
        return self


class ContextViewRetentionEvidence(ContractValue):
    """Bounded current native pin state, distinct from immutable acquisition."""

    selection: ContextViewSelectionDecision
    state: Literal["selected", "adopted", "transferred", "released", "expired"]
    owner: OwnerRef
    participant: ParticipantRef | None
    revision: Generation
    expires_at_ms: StrictInt = Field(ge=0, le=2**53 - 1)

    @model_validator(mode="after")
    def coherent(self):
        if (
            self.selection.state != "selected"
            or self.selection.target is None
            or (self.participant is not None and self.participant.owner != self.owner)
        ):
            raise ValueError("Context-view retention evidence conflicts with acquisition.")
        return self


def retention_evidence(target, decision, receipt):
    target = ContextViewSelectionTarget.model_validate(target)
    if decision is None or decision.state != "selected":
        return None
    if decision.target != target:
        raise ContextViewSelectionConflict("Retention readback conflicts with its exact target.")
    return ContextViewRetentionEvidence(
        selection=decision,
        state=receipt.state,
        owner=receipt.owner,
        participant=receipt.owner_participant,
        revision=receipt.ownership_revision,
        expires_at_ms=receipt.expires_at_ms,
    )


def snapshot_request(request: ContextViewSelectionRequest) -> ContextViewSelectionRequest:
    if type(request) is not ContextViewSelectionRequest:
        raise TypeError("Context-view selection requires an exact typed request.")
    return ContextViewSelectionRequest.model_validate(request)


def request_commitment(request: ContextViewSelectionRequest) -> str:
    request = snapshot_request(request)
    return json_commitment(
        canonical_bounded_durable_json_bytes(
            request.model_dump(mode="json"),
            "context view selection request",
            max_bytes=256 * 1024,
            max_nodes=8192,
            max_nesting=64,
        ).decode(),
        "context view selection request",
    )


def require_authority(authority: object) -> None:
    if authority is not _CONTEXT_SELECTION_AUTHORITY:
        raise PermissionError("Selection exclusion requires its trusted receiving owner.")


def require_selection_fence_store(store) -> None:
    """A forwarding wrapper must qualify the complete native decision boundary."""
    version = type(store).__dict__.get("context_view_selection_fence_version")
    if type(version) is not int or version != 1:
        raise NotImplementedError("Context-view selection exclusion is not qualified.")


def require_not_excluded(
    request, excluded: ContextViewSelectionDecision | None, *, target=None
) -> None:
    if excluded is None:
        if target is not None:
            raise PermissionError("Context-view selection lacks its retained responsibility.")
        return
    excluded = ContextViewSelectionDecision.model_validate(excluded)
    if excluded.request != request:
        raise ContextViewSelectionConflict("Context-view selection request conflicts.")
    if excluded.target != target:
        raise ContextViewSelectionConflict("Context-view selection admission conflicts.")
    if target is not None and not excluded.responsibility_registered:
        raise PermissionError("Context-view selection permit has not been registered.")
    if excluded.state == "reserved":
        return
    if excluded.state != "excluded":
        raise ValueError("Invalid native selection exclusion evidence.")
    raise ContextViewSelectionExcluded("Context-view selection has been excluded.")


def selected_decision(request, receipt, commitment, control=None) -> ContextViewSelectionDecision:
    if commitment != request_commitment(request):
        raise ContextViewSelectionConflict("Context-view selection request conflicts.")
    receipt = ContextViewSelectionReceipt.model_validate(receipt)
    view = receipt.view
    require_selected_participant(None if control is None else control.target, view)
    if (
        receipt.selection_key != request.selection_key
        or view.source_owner != request.source_owner
        or view.source_session_id != request.source_session_id
        or view.source_session_instance_id != request.source_session_instance_id
        or view.projection_schema != request.projection_schema
        or view.extension_set_commitment != request.extension_set_commitment
        or (request.selector == "exact" and view.view_id != request.exact_view_id)
        or (
            request.minimum_transcript_cursor is not None
            and view.transcript_cursor < request.minimum_transcript_cursor
        )
    ):
        raise ContextViewSelectionConflict("Context-view selection receipt conflicts.")
    return ContextViewSelectionDecision(
        request=request,
        state="selected",
        view_id=view.view_id,
        manifest_commitment=view.manifest_commitment,
        pin_commitment=receipt.pin_commitment,
        target=None if control is None else control.target,
        responsibility_registered=False if control is None else control.responsibility_registered,
    )


def control_transition(request, prior, *, exclude, target=None, register=False):
    """Pure exact transition; its caller owns the native transaction and quota."""
    if target is not None:
        target = ContextViewSelectionTarget.model_validate(target)
        if target.request != request:
            raise ContextViewSelectionConflict("Selection control differs from its target.")
    if prior is not None:
        if prior.request != request or prior.target != target:
            raise ContextViewSelectionConflict("Selection control authority conflicts.")
        if prior.state != "reserved" or (not exclude and not register):
            return prior
    if register and (target is None or prior is None):
        raise PermissionError("Selection registration requires its prepared native target.")
    return ContextViewSelectionDecision(
        request=request,
        state="excluded" if exclude else "reserved",
        target=target,
        responsibility_registered=register
        or (prior is not None and prior.responsibility_registered),
    )


def require_selection_source(target, binding, *, now_ms):
    if target is None:
        return
    if now_ms >= target.deadline_at_ms:
        raise PermissionError("Context-view selection admission has expired.")
    if (
        binding is None
        or binding.participant != target.permit.intent.request.participant
        or binding.session_id != target.request.source_session_id
        or binding.session_instance_id != target.request.source_session_instance_id
    ):
        raise PermissionError("Selection admission does not own the exact source session.")


def require_selected_participant(target, view):
    if target is not None and view.participant != target.permit.intent.request.participant:
        raise PermissionError("Selected view differs from its admitted source participant.")


def selection_adoption_request(target, selection):
    from hashlib import sha256

    if selection.target != target or selection.state != "selected":
        raise ContextViewSelectionConflict("Adoption requires exact positive selection evidence.")
    participant = target.permit.intent.request.participant
    return ContextViewOwnershipRequest(
        selection_key=target.request.selection_key,
        view_id=selection.view_id,
        pin_commitment=selection.pin_commitment,
        expected_state="selected",
        expected_revision=1,
        operation="adopt",
        current_owner=participant.owner,
        current_participant=participant,
        destination_owner=target.recipient.owner,
        destination_participant=target.recipient,
        operation_key="plan-view-adopt:" + sha256(target.model_dump_json().encode()).hexdigest(),
    )


def require_selection_adoption(request, decision, *, target=None):
    if decision is None or decision.target is None:
        if target is not None:
            raise PermissionError("Adoption lacks its native selection responsibility.")
        return
    if target is None:
        if request.operation != "release":
            raise PermissionError("This pin belongs to its exact admitted recipient handoff.")
        return
    if not decision.responsibility_registered or request != selection_adoption_request(
        target, decision
    ):
        raise ContextViewSelectionConflict("Adoption conflicts with its registered exact handoff.")


def require_adoption_deadline(target, now_ms):
    if target is not None and now_ms >= target.deadline_at_ms:
        raise PermissionError("Context-view adoption admission has expired.")
