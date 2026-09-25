"""Typed retained planning evidence and native index projections.

These values describe receiving-owner records; parsing them does not authenticate
their origin. Store readback must also verify receipts, events and parent links.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Annotated, Literal, cast

from pydantic import Discriminator, Field, StrictInt, Tag, model_validator

from cayu._validation import canonical_bounded_durable_json_bytes
from cayu.collaboration._clarification_commands import (
    ClarificationOpenCommand,
    ClarificationOpenReceipt,
)
from cayu.collaboration._contracts import (
    MAX_DEPTH,
    MAX_ENVELOPE_BYTES,
    MAX_NODES,
    CollaborationContractError,
    ContractValue,
    Generation,
    Identifier,
    InitiatorBinding,
    OperationRef,
    snapshot_input,
)
from cayu.collaboration._planning_creation_types import (
    RequestCreationStageCommand,
    RequestCreationStageReceipt,
)
from cayu.collaboration._planning_fork_types import RequestViewStageCommand, RequestViewStageReceipt
from cayu.collaboration._planning_resource_types import (
    RequestResourceAcquisitionStageCommand,
    RequestResourceStageAdoption,
    RequestResourceStageRelease,
    RequestResourceTransferStageCommand,
)
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration.clarifications import Commitment
from cayu.collaboration.participants import ParticipantRef, VersionOne
from cayu.collaboration.planning import (
    MAX_REQUEST_PLANNING_GENERATIONS,
    MAX_REQUEST_PLANNING_PAGE,
    MAX_REQUEST_PLANNING_STAGES,
    ConfiguredRequestPlanningPolicy,
    RequestPlanningControl,
    RequestPlanningDecision,
    RequestPlanningDefer,
    RequestPlanningPrerequisite,
    RequestPlanningRequest,
    _selected_policy_proposal,
)
from cayu.collaboration.requests import (
    Millis,
    RequestAdmissionCommand,
    RequestAdmissionReceipt,
    RequestRef,
)
from cayu.vaults.redaction import SecretRedactor

MAX_REQUEST_PLAN_EVENTS = 8
PENDING_PLANNING_STATES = ("evaluating", "decided", "deferred", "clarifying", "preparing")


def _validated_bytes(value: ContractValue) -> bytes:
    """Encode fields already validated by this model's after-validator.

    ContractValue always revalidates nested instances. Calling contract_bytes
    here would recursively validate the same entire tree again at every parent.
    Public/store preparation still performs secret checks and bounded encoding.
    This helper is not an entrance for untrusted values or authentication.
    """
    return canonical_bounded_durable_json_bytes(
        snapshot_input(value),
        "planning commitment",
        max_bytes=MAX_ENVELOPE_BYTES,
        max_nodes=MAX_NODES,
        max_nesting=MAX_DEPTH,
    )


PlanningState = Literal[
    "evaluating",
    "decided",
    "deferred",
    "clarifying",
    "preparing",
    "admitted",
    "declined",
    "cancelled",
    "expired",
    "superseded",
]
PlanningEventType = Literal[
    "plan_retained",
    "plan_decided",
    "plan_waiting",
    "plan_preparing",
    "plan_admitted",
    "plan_declined",
    "plan_cancelled",
    "plan_expired",
    "plan_superseded",
    "plan_stage_retained",
    "plan_stage_settled",
    "plan_stage_excluded",
]


class RequestPlanningEvent(ContractValue):
    id: Identifier
    sequence: Generation
    operation: OperationRef
    plan: OperationRef
    request: RequestRef
    type: PlanningEventType
    commitment: Commitment
    participants: tuple[ParticipantRef, ...] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def coherent(self) -> RequestPlanningEvent:
        if (
            self.operation.application_scope != self.request.owner.application_scope
            or self.plan.application_scope != self.operation.application_scope
            or any(item.owner != self.request.owner for item in self.participants)
            or len(set(self.participants)) != len(self.participants)
        ):
            raise ValueError("Planning event belongs to another owner.")
        return self


class RequestPlanningReceipt(ContractValue):
    """Immutable retained evaluation intent, before any policy or foreign effect."""

    command: RequestPlanningRequest
    policy: ConfiguredRequestPlanningPolicy
    retained_at_ms: Millis
    event: RequestPlanningEvent

    @model_validator(mode="after")
    def coherent(self) -> RequestPlanningReceipt:
        command = self.command
        selected = command.expected.intent.selection
        if (
            self.policy.reference != command.policy
            or sha256(_validated_bytes(self.policy)).hexdigest() != command.policy_sha256
            or len(_validated_bytes(self.policy)) > self.policy.limits.max_record_bytes
            or any(
                getattr(command.limits, field) > getattr(self.policy.limits, field)
                for field in type(command.limits).model_fields
            )
            or not selected.accepted_at_ms <= self.retained_at_ms < command.deadline_at_ms
            or self.event.operation != command.operation
            or self.event.plan != command.operation
            or self.event.request != selected.reference
            or self.event.type != "plan_retained"
            or self.event.commitment != sha256(_validated_bytes(command)).hexdigest()
            or self.event.participants
            != tuple(dict.fromkeys((selected.sender.reference, selected.recipient.reference)))
        ):
            raise ValueError("Planning receipt contradicts its exact evaluation intent.")
        return self


class RequestPlanningSuccessor(ContractValue):
    request: RequestPlanningRequest
    prerequisite: RequestAdmissionReceipt | None


class RequestPlanningDisposition(ContractValue):
    """Control expectation bound to the immutable command already in this record."""

    expected_sha256: Commitment
    expected_revision: Generation
    kind: Literal["cancelled", "expired"]
    initiator: InitiatorBinding


def planning_disposition(control: RequestPlanningControl) -> RequestPlanningDisposition:
    return RequestPlanningDisposition(
        expected_sha256=sha256(_validated_bytes(control.expected)).hexdigest(),
        expected_revision=control.expected_revision,
        kind=control.kind,
        initiator=control.initiator,
    )


class RequestPlanningRecord(ContractValue):
    """Business state and pending responsibility are deliberately independent."""

    schema_version: VersionOne = 1
    receipt: RequestPlanningReceipt
    revision: Generation
    state: PlanningState
    decision_commitment: Commitment | None
    stage_count: StrictInt = Field(ge=0, le=MAX_REQUEST_PLANNING_STAGES)
    pending_stages: StrictInt = Field(ge=0, le=MAX_REQUEST_PLANNING_STAGES)
    pruned_stages: StrictInt = Field(default=0, ge=0, le=MAX_REQUEST_PLANNING_STAGES)
    next_due_at_ms: Millis
    event_sequences: tuple[Generation, ...] = Field(
        min_length=1, max_length=MAX_REQUEST_PLAN_EVENTS
    )
    reserved_bytes: StrictInt = Field(ge=0, le=2**53 - 1)
    reserved_events: StrictInt = Field(ge=0, le=MAX_REQUEST_PLAN_EVENTS)
    disposition: RequestPlanningDisposition | None = None
    successor: RequestPlanningSuccessor | None = None

    @property
    def decision(self) -> RequestPlanningDecision | None:
        """The retained policy owns the complete immutable selected proposal."""
        if self.decision_commitment is None:
            return None
        return _selected_policy_proposal(
            self.receipt.policy, self.receipt.command.expected_input_revision
        )

    @property
    def control(self) -> RequestPlanningControl | None:
        """Reconstruct the full public expectation without duplicating it durably."""
        if self.disposition is None:
            return None
        return RequestPlanningControl(
            expected=self.receipt.command,
            expected_revision=self.disposition.expected_revision,
            kind=self.disposition.kind,
            initiator=self.disposition.initiator,
        )

    @model_validator(mode="after")
    def coherent(self) -> RequestPlanningRecord:
        decision = self.decision
        if self.decision_commitment is not None and (
            decision is None
            or self.decision_commitment != sha256(_validated_bytes(decision)).hexdigest()
        ):
            raise ValueError("Planning decision differs from the exact retained policy selection.")
        if (
            self.disposition is not None
            and self.disposition.expected_sha256
            != sha256(_validated_bytes(self.receipt.command)).hexdigest()
        ):
            raise ValueError("Planning disposition contradicts its exact retained command.")
        if self.successor is not None:
            successor = self.successor.request
            predecessor = successor.predecessor
            if (
                self.state != "superseded"
                or predecessor is None
                or predecessor.operation != self.receipt.command.operation
                or predecessor.revision + 1 != self.revision
                or successor.expected != self.receipt.command.expected
                or successor.planning_generation != self.receipt.command.planning_generation + 1
            ):
                raise ValueError("Planning successor contradicts its predecessor.")
            prerequisite = (
                self.decision.prerequisite
                if isinstance(self.decision, RequestPlanningDefer)
                else None
            )
            if isinstance(prerequisite, RequestPlanningPrerequisite):
                if (
                    self.successor.prerequisite is None
                    or self.successor.prerequisite.command != prerequisite.expected
                ):
                    raise ValueError("Planning successor lacks exact prerequisite evidence.")
            elif self.successor.prerequisite is not None:
                raise ValueError("Planning successor carries unrelated prerequisite evidence.")
        control = self.control
        if control is not None and (
            control.expected != self.receipt.command
            or control.kind != self.state
            or control.expected_revision + 1 != self.revision
        ):
            raise ValueError("Planning control contradicts its retained revision.")
        if (
            self.pending_stages > self.stage_count
            or self.pruned_stages > self.stage_count
            or (
                self.pruned_stages
                and (
                    self.state in PENDING_PLANNING_STATES
                    or self.pending_stages
                    or self.reserved_bytes
                    or self.reserved_events
                )
            )
            or self.stage_count > self.receipt.command.limits.max_stages
            or (self.stage_count > 0 and self.decision is None)
            or not self.receipt.retained_at_ms
            <= self.next_due_at_ms
            <= self.receipt.command.deadline_at_ms
            or self.event_sequences != tuple(sorted(set(self.event_sequences)))
            or self.event_sequences[0] != self.receipt.event.sequence
            or self.revision != len(self.event_sequences)
            or (self.state == "evaluating" and self.decision is not None)
            or (
                self.state
                in {"decided", "deferred", "clarifying", "preparing", "admitted", "declined"}
                and self.decision is None
            )
            or (self.state in {"admitted", "declined", "superseded"} and self.pending_stages)
        ):
            raise ValueError("Planning state contradicts its retained frontier.")
        if self.state in {"deferred", "clarifying", "declined"} and (
            self.decision is None
            or self.decision.decision
            != {"deferred": "defer", "clarifying": "clarify", "declined": "decline"}[self.state]
        ):
            raise ValueError("Planning state contradicts its decision.")
        return self


class RequestPlanningStageIntent(ContractValue):
    """Exact local stage identity and expected receiving command."""

    operation: OperationRef
    plan: OperationRef
    plan_sha256: Commitment
    ordinal: StrictInt = Field(ge=1, le=MAX_REQUEST_PLANNING_STAGES)
    command: (
        RequestAdmissionCommand
        | ClarificationOpenCommand
        | RequestCreationStageCommand
        | RequestViewStageCommand
        | RequestResourceAcquisitionStageCommand
        | RequestResourceTransferStageCommand
    ) = Field(discriminator="mode")

    @model_validator(mode="after")
    def coherent(self) -> RequestPlanningStageIntent:
        parent = self.command.expected.operation
        refs = (self.operation, self.plan, self.command.operation, parent)
        if len(set(refs)) != len(refs) or any(
            (ref.application_scope, ref.namespace_incarnation, ref.generation)
            != (parent.application_scope, parent.namespace_incarnation, parent.generation)
            for ref in refs
        ):
            raise ValueError("Planning stage requires distinct exact namespace identities.")
        return self


def _stage_receipt_variant(value: object) -> str | None:
    """Choose a schema from positive tags; the selected schema still validates all data.

    Receipts already carry the command's mode. Do not reconstruct every sibling
    command (and its nested native profile/permit) just to discover that tag.
    Resource transfer has two terminal shapes, distinguished by their explicit
    receiving-settlement or child-material evidence. Neither shape is authority.
    """
    if isinstance(value, ContractValue):
        value = object.__getattribute__(value, "__dict__")
    if type(value) is not dict:
        return None
    fields = cast("dict[str, object]", value)
    command = fields.get("command")
    if isinstance(command, ContractValue):
        command = object.__getattribute__(command, "__dict__")
    if type(command) is not dict:
        return None
    mode = cast("dict[str, object]", command).get("mode")
    if type(mode) is not str:
        return None
    if mode in {"request_resource_acquisition", "request_resource_transfer"}:
        if "receiving" in fields and "material" not in fields:
            return "resource_release"
        if (
            mode == "request_resource_transfer"
            and "material" in fields
            and "receiving" not in fields
        ):
            return "resource_adoption"
        return None
    return mode


class RequestPlanningStageRecord(ContractValue):
    """An exclusion is native owner evidence, never absence inferred by a waiter."""

    schema_version: VersionOne = 1
    mode: Literal["request_plan_stage"] = "request_plan_stage"
    intent: RequestPlanningStageIntent
    state: Literal["pending", "settled", "excluded"]
    registered_at_ms: Millis
    settled_at_ms: Millis | None
    receipt: (
        Annotated[
            Annotated[RequestAdmissionReceipt, Tag("request_admission")]
            | Annotated[ClarificationOpenReceipt, Tag("clarification_open")]
            | Annotated[RequestCreationStageReceipt, Tag("request_recipient_creation")]
            | Annotated[RequestViewStageReceipt, Tag("request_context_view")]
            | Annotated[RequestResourceStageRelease, Tag("resource_release")]
            | Annotated[RequestResourceStageAdoption, Tag("resource_adoption")],
            Discriminator(_stage_receipt_variant),
        ]
        | None
    )
    registration_event: RequestPlanningEvent
    settlement_event: RequestPlanningEvent | None
    reserved_bytes: StrictInt = Field(ge=0, le=2**53 - 1)

    @model_validator(mode="after")
    def coherent(self) -> RequestPlanningStageRecord:
        selected = self.intent.command.expected.intent.selection
        if (
            isinstance(
                self.intent.command,
                (
                    RequestCreationStageCommand,
                    RequestResourceAcquisitionStageCommand,
                    RequestResourceTransferStageCommand,
                ),
            )
            and self.state == "excluded"
        ):
            raise ValueError("Foreign preparation requires a positive native terminal receipt.")
        if not selected.accepted_at_ms <= self.registered_at_ms < selected.expires_at_ms:
            raise ValueError("Planning stage was registered outside its request deadline.")
        for event in (self.registration_event, self.settlement_event):
            if event is not None and (
                event.operation != self.intent.operation
                or event.plan != self.intent.plan
                or event.request != selected.reference
                or event.participants
                != tuple(dict.fromkeys((selected.sender.reference, selected.recipient.reference)))
            ):
                raise ValueError("Planning stage event contradicts its operation.")
        if self.registration_event.type != "plan_stage_retained":
            raise ValueError("Planning stage lacks its registration event.")
        intent_commitment = sha256(_validated_bytes(self.intent)).hexdigest()
        if self.registration_event.commitment != intent_commitment:
            raise ValueError("Planning stage registration commitment conflicts.")
        if self.state == "pending":
            if any(
                value is not None
                for value in (self.settled_at_ms, self.receipt, self.settlement_event)
            ):
                raise ValueError("Pending stage cannot claim settlement.")
        elif (
            self.settled_at_ms is None
            or self.settled_at_ms < self.registered_at_ms
            or self.settlement_event is None
            or self.settlement_event.sequence <= self.registration_event.sequence
            or self.settlement_event.type != "plan_stage_" + self.state
            or self.reserved_bytes
        ):
            raise ValueError("Terminal stage lacks exact settlement evidence.")
        if (self.state == "settled") != (self.receipt is not None):
            raise ValueError("Stage settlement requires a receiving receipt.")
        if self.receipt is not None and self.receipt.command != self.intent.command:
            raise ValueError("Stage receiving receipt conflicts with its full command.")
        if self.settlement_event is not None and self.settlement_event.commitment != (
            intent_commitment
            if self.receipt is None
            else sha256(_validated_bytes(self.receipt)).hexdigest()
        ):
            raise ValueError("Planning stage settlement commitment conflicts.")
        return self


PLANNING_RECORD_FAMILIES = frozenset(
    {"request_plans", "request_plan_stages", "request_plan_events"}
)


class RequestPlanningCursor(ContractValue):
    next_due_at_ms: Millis
    operation: OperationRef


class RequestPlanningPage(ContractValue):
    """Bounded maintenance readback, not a lease or permission to execute."""

    records: tuple[RequestPlanningRecord, ...] = Field(max_length=MAX_REQUEST_PLANNING_PAGE)
    next_cursor: RequestPlanningCursor | None
    observed_at_ms: Millis


def planning_cursor_key(cursor: RequestPlanningCursor) -> tuple[int, str, int, str]:
    operation = cursor.operation
    return (
        cursor.next_due_at_ms,
        operation.namespace_incarnation,
        operation.generation,
        operation.caller_key,
    )


def prepare_planning_scan(value: object, *, scope: str, limit: int, kind: str):
    redactor = SecretRedactor()
    if kind in {"stages", "native_stage"}:
        maximum = MAX_REQUEST_PLANNING_STAGES + 1 if kind == "stages" else 1
        if type(limit) is not int or not 1 <= limit <= maximum:
            raise CollaborationContractError("Planning stage lookup requires a bounded limit.")
        operation = prepare_contract(OperationRef, value, redactor=redactor)
        if operation.application_scope != scope:
            raise CollaborationContractError("Planning stage lookup belongs to another scope.")
        return operation
    if kind == "request":
        if type(limit) is not int or not 1 <= limit <= MAX_REQUEST_PLANNING_GENERATIONS + 1:
            raise CollaborationContractError("Planning request scan requires a bounded limit.")
        request = prepare_contract(RequestRef, value, redactor=redactor)
        if request.owner.application_scope != scope:
            raise CollaborationContractError("Planning request scan belongs to another scope.")
        return request
    if kind == "pending":
        if type(limit) is not int or not 1 <= limit <= MAX_REQUEST_PLANNING_PAGE:
            raise CollaborationContractError("Planning recovery scan requires a bounded limit.")
        if value is None:
            return None
        cursor = prepare_contract(RequestPlanningCursor, value, redactor=redactor)
        if cursor.operation.application_scope != scope:
            raise CollaborationContractError("Planning cursor belongs to another scope.")
        return cursor
    raise CollaborationContractError("Unknown planning scan kind.")


def planning_record_projection(
    family: str, value: object, *, scope: str, key: tuple[str | int, ...]
) -> tuple[ContractValue, tuple[str | int, ...]]:
    """Validate canonical documents before comparing each native index column."""
    redactor = SecretRedactor()
    if family == "request_plans":
        record = prepare_contract(RequestPlanningRecord, value, redactor=redactor)
        command = record.receipt.command
        operation = command.operation
        request = command.expected.intent.selection.reference
        projection = (
            request.request_id,
            request.incarnation,
            command.expected.intent.selection.recipient.reference.participant_id,
            command.planning_generation,
            record.state,
            record.pending_stages,
            record.next_due_at_ms,
        )
    elif family == "request_plan_stages":
        record = prepare_contract(RequestPlanningStageRecord, value, redactor=redactor)
        operation = record.intent.operation
        plan = record.intent.plan
        native = record.intent.command.operation
        projection = (
            plan.namespace_incarnation,
            plan.generation,
            plan.caller_key,
            record.intent.ordinal,
            native.namespace_incarnation,
            native.generation,
            native.caller_key,
            record.state,
        )
    elif family == "request_plan_events":
        event = prepare_contract(RequestPlanningEvent, value, redactor=redactor)
        if key != (event.sequence,) or event.operation.application_scope != scope:
            raise CollaborationContractError("Planning event index contradicts its record.")
        return event, ()
    else:
        raise CollaborationContractError("Unknown planning record family.")
    if key != (operation.namespace_incarnation, operation.generation, operation.caller_key) or (
        operation.application_scope != scope
    ):
        raise CollaborationContractError("Planning record index contradicts its identity.")
    return record, projection
