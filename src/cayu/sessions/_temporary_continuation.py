"""Exact temporary-service values for the existing continuation receiving owner.

Construction is not admission. Native publication must authenticate the retained
ticket, participant binding, question, registered permit and invocation command.
These values are deliberately not exported as a public execution capability.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, StrictBool, StrictInt, model_validator

from cayu.collaboration._contracts import (
    ContractValue,
    Identifier,
    InitiatorBinding,
    ObjectRef,
    OperationRef,
)
from cayu.collaboration._permits import PermitCommand, ReceivingSettlementReceipt
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration.clarifications import (
    MAX_CLARIFICATION_DEPTH,
    MAX_CLARIFICATION_TURNS,
    ClarificationQuestion,
    Commitment,
)
from cayu.collaboration.participants import VersionOne
from cayu.collaboration.peer_content import PeerContentAppendRequest
from cayu.sessions._session_continuation import (
    CONTINUATION_SERVICE_PREFIX,
    ContinuationConflict,
    ContinuationServiceReference,
    ContinuationTicket,
    continuation_digest,
)
from cayu.vaults.redaction import SecretRedactor

ServiceReleasedSessionStatus = Literal[
    "pending", "running", "interrupting", "completed", "failed", "interrupted"
]


def temporary_service_invocation_id(operation: OperationRef) -> str:
    """Stable runtime identity, namespaced by the complete service operation.

    Queue delivery identities include the interaction identity and are shared
    across sessions. A caller-selected label is not a safe execution identity.
    """
    return str(
        uuid5(
            NAMESPACE_URL,
            "cayu:temporary-continuation:interaction:" + continuation_digest(operation),
        )
    )


class TemporaryServiceIntent(ContractValue):
    """Complete immutable service selection, separate from the original ticket."""

    schema_version: VersionOne = 1
    operation: OperationRef
    initiator: InitiatorBinding
    ticket: ContinuationTicket
    question: ClarificationQuestion
    service_generation: StrictInt = Field(ge=1, le=MAX_CLARIFICATION_TURNS)
    parent_service: OperationRef | None
    depth: StrictInt = Field(ge=1, le=MAX_CLARIFICATION_DEPTH)
    mode: Literal["same_session", "side_session"]
    target: ObjectRef
    participant_binding_sha256: Commitment
    execution_profile_sha256: Commitment
    resume_sha256: Commitment
    invocation_id: Identifier
    # Durable timestamp for exact event reconstruction, not an admission clock.
    # Range ends at the final millisecond representable by Python datetime.
    prepared_at_ms: StrictInt = Field(ge=1, le=253402300799999)
    budget_binding: ObjectRef
    budget_authority_sha256: Commitment
    # Public host selection commitment; runtime preparation remains private.
    selection_sha256: Commitment | None = None
    # Exact receiving dependency, retained across callback loss/restart. Only
    # runtime-owned selectors authenticate it; its presence is not an append.
    required_peer_append: PeerContentAppendRequest | None = None

    @property
    def prepared_at(self) -> datetime:
        return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=self.prepared_at_ms)

    @model_validator(mode="after")
    def exact_selection(self) -> TemporaryServiceIntent:
        from cayu.collaboration.waits import request_object_ref

        scope = self.operation.application_scope
        if self.required_peer_append is not None:
            key = self.required_peer_append.append_key
            if (
                key.target_session_id != self.target.object_id
                or key.target_session_instance_id != self.target.incarnation
                or self.required_peer_append.wake_policy != "none"
            ):
                raise ValueError("Service delivery belongs to another receiving target.")
        same_session = (self.target.object_id, self.target.incarnation) == (
            self.ticket.session_id,
            self.ticket.session_instance_id,
        )
        if (
            self.ticket.owner.application_scope != scope
            or self.invocation_id != temporary_service_invocation_id(self.operation)
            or (
                self.depth == 1
                and request_object_ref(self.question.request) not in self.ticket.targets
            )
            or self.initiator.issuer.application_scope != scope
            or self.question.operation.application_scope != scope
            or self.target.owner != self.ticket.owner
            or self.target.kind != "session"
            or self.target.revision is not None
            or (
                self.target.object_id == self.ticket.session_id
                and self.target.incarnation != self.ticket.session_instance_id
            )
            or (self.mode == "same_session") != same_session
            or self.budget_binding != self.question.budget_binding
            or self.budget_authority_sha256 != self.question.budget_authority_sha256
            or self.depth != self.question.depth
            or (self.parent_service is None) != (self.depth == 1)
            or (
                self.parent_service is not None
                and (
                    self.parent_service.application_scope != scope
                    or self.parent_service == self.operation
                )
            )
            or self.ticket.state not in {"WAITING", "SERVICING"}
            or (self.ticket.state == "SERVICING" and self.parent_service is None)
        ):
            raise ValueError("Temporary service selection conflicts with its retained authority.")
        return self


class TemporaryServiceDispatch(ContractValue):
    """Permit-independent command payload committed before permit registration."""

    intent: TemporaryServiceIntent
    admission_payload_sha256: Commitment
    # Admission and subsequent writer release each advance the epoch once.
    expected_run_epoch: StrictInt = Field(ge=1, le=2**53 - 3)


class TemporaryServicePreparation(ContractValue):
    """Exact expected permit before registration, not an execution grant.

    This tuple remains usable when a registration acknowledgement is lost. Native
    capacity/exclusion must not invent a receipt or final command commitment.
    """

    dispatch: TemporaryServiceDispatch
    permit: PermitCommand

    @model_validator(mode="after")
    def exact_permit(self) -> TemporaryServicePreparation:
        registration = self.permit.intent.request
        intent = self.dispatch.intent
        if (
            registration.source_operation != intent.operation
            or registration.target != intent.target
            or registration.target_state != "existing"
            or registration.expected_configuration_revision is None
            or registration.participant != intent.question.responder
            or self.permit.initiator != intent.initiator
            or registration.admission_commitment != continuation_digest(self.dispatch)
            or registration.required_settlement != "quiescence"
            or registration.effect_scope != "clarification_service"
        ):
            raise ValueError("Temporary service permit does not bind its exact admission.")
        return self


class TemporaryServiceAdmission(TemporaryServicePreparation):
    """Acknowledged permit and final native command, required for execution."""

    permit_receipt_sha256: Commitment
    admission_command_sha256: Commitment

    @property
    def preparation(self) -> TemporaryServicePreparation:
        return TemporaryServicePreparation(dispatch=self.dispatch, permit=self.permit)


class TemporaryServiceExclusion(ContractValue):
    """Positive receiving fence for an exact prepared permit, never mere absence.

    Construction alone confers no authority. Only the receiving store's atomic
    admission/exclusion owner may persist or issue this record.
    """

    preparation: TemporaryServicePreparation
    receipt: ReceivingSettlementReceipt

    @model_validator(mode="after")
    def exact_fence(self) -> TemporaryServiceExclusion:
        if (
            self.receipt.expected != self.preparation.permit
            or self.receipt.outcome != "quiescent"
            or not self.receipt.admission_excluded
        ):
            raise ValueError("Temporary service exclusion lacks its exact receiving fence.")
        return self


class TemporaryServiceExecution(ContractValue):
    """Exact native admission readback projection, not a caller execution grant.

    The receiving owner must reconstruct this from its invocation-lifecycle
    receipt. Keeping only its identity and commitment avoids duplicating the
    receipt's executable session/profile in clarification history.
    """

    receipt_id: Identifier
    receipt_sha256: Commitment
    admission_command_sha256: Commitment
    session_id: Identifier
    session_instance_id: Identifier
    invocation_id: Identifier
    run_epoch: StrictInt = Field(ge=1, le=2**53 - 1)


class TemporaryServiceRecord(ContractValue):
    """One bounded service responsibility, independently of final wait latching.

    A missing acknowledgement, timeout or cancellation is deliberately not a
    terminal state. Only exact receiving-owner readback may populate execution
    or settlement. Native publication authenticates that readback and owns the
    transition; reconstruction here checks its complete internal linkage.
    """

    schema_version: VersionOne = 1
    admission: TemporaryServiceAdmission | TemporaryServicePreparation
    state: Literal["prepared", "reserved", "admitted", "returned", "excluded"]
    execution: TemporaryServiceExecution | None = None
    settlement: ReceivingSettlementReceipt | None = None
    # Set by the registered owner only after BOTH foreign permit and lineage
    # settlement succeed. Native return alone must not release this evidence.
    settlement_acknowledged: StrictBool = False
    returned_writer_generation: StrictInt | None = Field(default=None, ge=1, le=2**53 - 1)
    # Historical receiving evidence, not today's session status or a claim that
    # quiescence implies successful completion.
    released_session_status: ServiceReleasedSessionStatus | None = None

    @property
    def intent(self) -> TemporaryServiceIntent:
        return self.admission.dispatch.intent

    @property
    def acknowledged_admission(self) -> TemporaryServiceAdmission:
        if not isinstance(self.admission, TemporaryServiceAdmission):
            raise ContinuationConflict("Temporary service registration remains unacknowledged.")
        return self.admission

    @model_validator(mode="after")
    def exact_responsibility(self) -> TemporaryServiceRecord:
        if self.settlement_acknowledged and self.state not in {"returned", "excluded"}:
            raise ValueError("Temporary service acknowledgement requires terminal evidence.")
        if self.state not in {"prepared", "excluded"} and not isinstance(
            self.admission, TemporaryServiceAdmission
        ):
            raise ValueError("Temporary execution requires acknowledged registration evidence.")
        if (self.state in {"admitted", "returned"}) != (self.execution is not None):
            raise ValueError("Temporary service state conflicts with native admission evidence.")
        if (self.state in {"returned", "excluded"}) != (self.settlement is not None):
            raise ValueError("Temporary service terminal state lacks exact settlement.")
        if (self.state == "returned") != (self.returned_writer_generation is not None):
            raise ValueError("Temporary service return lacks writer succession evidence.")
        if (self.state == "returned") != (self.released_session_status is not None):
            raise ValueError("Temporary service return lacks its exact released session status.")
        if self.execution is not None:
            execution = self.execution
            dispatch = self.admission.dispatch
            if (
                execution.admission_command_sha256
                != self.acknowledged_admission.admission_command_sha256
                or execution.session_id != self.intent.target.object_id
                or execution.session_instance_id != self.intent.target.incarnation
                or execution.invocation_id != self.intent.invocation_id
                or execution.run_epoch != dispatch.expected_run_epoch + 1
            ):
                raise ValueError("Temporary service execution conflicts with its exact dispatch.")
            if (
                self.returned_writer_generation is not None
                and self.returned_writer_generation != execution.run_epoch + 1
            ):
                raise ValueError("Temporary service return has a different writer successor.")
        if self.settlement is not None:
            if self.settlement.expected != self.admission.permit:
                raise ValueError("Temporary service settlement belongs to another permit.")
            if self.state == "excluded" and not self.settlement.proves_exclusion:
                raise ValueError("Quiescence alone cannot exclude temporary service admission.")
        return self


def advance_temporary_service_record(
    current: TemporaryServiceRecord,
    proposed: TemporaryServiceRecord,
) -> TemporaryServiceRecord:
    """Validate a native owner's proposed transition without authenticating it.

    Full readback may reconcile reserved directly to returned after a lost
    acknowledgement. That is safe only with both admission and quiescence
    evidence. There is intentionally no failure/timeout-to-exclusion shortcut.
    """
    redactor = SecretRedactor()
    current = prepare_contract(TemporaryServiceRecord, current, redactor=redactor)
    proposed = prepare_contract(TemporaryServiceRecord, proposed, redactor=redactor)
    if current == proposed:
        return current
    if (
        current.state in {"returned", "excluded"}
        and not current.settlement_acknowledged
        and proposed == current.model_copy(update={"settlement_acknowledged": True})
    ):
        return proposed
    preparation_upgrade = (
        current.state == "prepared"
        and isinstance(proposed.admission, TemporaryServiceAdmission)
        and current.admission == proposed.admission.preparation
    )
    if current.admission != proposed.admission and not preparation_upgrade:
        raise ContinuationConflict("Temporary service operation changed its admission tuple.")
    allowed = {
        "prepared": {"reserved", "excluded"},
        "reserved": {"admitted", "returned", "excluded"},
        "admitted": {"returned"},
        "returned": set(),
        "excluded": set(),
    }
    if proposed.state not in allowed[current.state] or (
        current.execution is not None and current.execution != proposed.execution
    ):
        raise ContinuationConflict("Temporary service has conflicting receiving evidence.")
    return proposed


def temporary_service_key(operation: OperationRef) -> str:
    return CONTINUATION_SERVICE_PREFIX + continuation_digest(operation)


def temporary_admission_payload_sha256(command: object) -> str:
    """Commit all command effects before its derived permit proof exists.

    Only the two permit-evidence fields are omitted. The native receiving check
    compares those fields separately and retains the complete final command hash.
    This avoids a permit/command self-reference without discarding effect inputs.
    """
    from cayu.sessions._invocation_lifecycle import (
        AdmitInvocationCommand,
        copy_invocation_lifecycle_command,
        invocation_admission_command_sha256,
    )

    checked = copy_invocation_lifecycle_command(command)
    if type(checked) is not AdmitInvocationCommand:
        raise TypeError("Temporary service requires a typed admission command.")
    payload = checked.model_copy(
        update={"participant_permit_operation": None, "participant_permit_commitment": None}
    )
    return invocation_admission_command_sha256(payload)


def require_temporary_service_command(
    admission: TemporaryServiceAdmission, command: object
) -> None:
    """Compare an authenticated handoff to the exact command at native admission."""
    from cayu.sessions._invocation_lifecycle import (
        AdmitInvocationCommand,
        copy_invocation_lifecycle_command,
        invocation_admission_command_sha256,
    )

    admission = prepare_contract(TemporaryServiceAdmission, admission, redactor=SecretRedactor())
    checked = copy_invocation_lifecycle_command(command)
    if type(checked) is not AdmitInvocationCommand:
        raise TypeError("Temporary service requires a typed admission command.")
    dispatch = admission.dispatch
    if (
        temporary_admission_payload_sha256(checked) != dispatch.admission_payload_sha256
        or invocation_admission_command_sha256(checked) != admission.admission_command_sha256
        or checked.participant_permit_operation != admission.permit.operation.caller_key
        or checked.participant_permit_commitment != admission.permit_receipt_sha256
        or checked.temporary_service_operation_key
        != temporary_service_key(dispatch.intent.operation)
        or checked.session_id != dispatch.intent.target.object_id
        or checked.expected_session_instance_id != dispatch.intent.target.incarnation
        or checked.expected_run_epoch != dispatch.expected_run_epoch
        or checked.target_active_profile.profile.fingerprint
        != dispatch.intent.execution_profile_sha256
        or checked.target_active_profile.interaction_id != dispatch.intent.invocation_id
    ):
        raise ContinuationConflict(
            "Temporary service command conflicts with its exact permit handoff."
        )


def reference_for_service(
    record: TemporaryServiceRecord, references: tuple[ContinuationServiceReference, ...]
) -> ContinuationServiceReference:
    """One projection for publication and every reconstructed index comparison."""
    intent = record.intent
    parent_generation = None
    if intent.parent_service is not None:
        parent_key = temporary_service_key(intent.parent_service)
        parent = next((item for item in references if item.key == parent_key), None)
        if parent is None:
            raise ContinuationConflict("Temporary service has no retained parent reference.")
        parent_generation = parent.generation
    return ContinuationServiceReference(
        key=temporary_service_key(intent.operation),
        record_sha256=continuation_digest(record),
        generation=intent.service_generation,
        parent_generation=parent_generation,
        state=record.state,
        mode=intent.mode,
        expected_run_epoch=record.admission.dispatch.expected_run_epoch,
        returned_writer_generation=record.returned_writer_generation,
    )


def require_temporary_service_capacity(
    record: TemporaryServiceRecord, *, envelope_overhead: int = 0
) -> None:
    """Prove the reserved envelope fits the largest receiving terminal receipt.

    These values are size witnesses only and never get published. Quotation
    marks use the maximum JSON expansion of a valid bounded receipt identifier.
    All other identifiers/authority are already frozen in the exact admission.
    """
    dispatch = record.admission.dispatch
    intent = dispatch.intent
    largest_receipt_id = '"' * 512
    # A preparation reserves the eventual full envelope before foreign permit
    # registration. These fixed-size hash witnesses never become authority.
    acknowledged = (
        record.admission
        if isinstance(record.admission, TemporaryServiceAdmission)
        else TemporaryServiceAdmission(
            dispatch=record.admission.dispatch,
            permit=record.admission.permit,
            permit_receipt_sha256="f" * 64,
            admission_command_sha256="f" * 64,
        )
    )
    terminal = TemporaryServiceRecord(
        admission=acknowledged,
        state="returned",
        execution=TemporaryServiceExecution(
            receipt_id=largest_receipt_id,
            receipt_sha256="f" * 64,
            admission_command_sha256=acknowledged.admission_command_sha256,
            session_id=intent.target.object_id,
            session_instance_id=intent.target.incarnation,
            invocation_id=intent.invocation_id,
            run_epoch=dispatch.expected_run_epoch + 1,
        ),
        settlement=ReceivingSettlementReceipt(
            expected=record.admission.permit,
            receiving_owner=intent.target.owner,
            receipt_id=largest_receipt_id,
            outcome="quiescent",
            admission_excluded=False,
        ),
        returned_writer_generation=dispatch.expected_run_epoch + 2,
        released_session_status="interrupting",
    )
    from cayu.collaboration._contracts import MAX_ENVELOPE_BYTES
    from cayu.collaboration._preparation import contract_bytes

    if type(envelope_overhead) is not int or envelope_overhead < 0:
        raise ValueError("Temporary service envelope overhead must be nonnegative.")
    if (
        len(contract_bytes(terminal, redactor=SecretRedactor())) + envelope_overhead
        > MAX_ENVELOPE_BYTES
    ):
        raise ContinuationConflict("Temporary service terminal envelope capacity is unavailable.")
