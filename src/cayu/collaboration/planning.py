"""Bounded deterministic planning values, never effect or admission authority.

The retained policy configuration is executable only by the closed algorithm
version below. Constructing or evaluating these values does not open a question,
create a session, inspect a transcript, or invoke a registered receiver.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Annotated, Literal

from pydantic import Field, StrictInt, model_validator

from cayu._validation import canonical_bounded_durable_json_bytes
from cayu.collaboration._clarification_commands import ClarificationOpenCommand
from cayu.collaboration._contracts import (
    MAX_DEPTH,
    MAX_NODES,
    CollaborationContractError,
    ContractValue,
    Generation,
    Identifier,
    InitiatorBinding,
    ObjectRef,
    OperationRef,
    OwnerRef,
    snapshot_input,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration.clarifications import Commitment
from cayu.collaboration.exports import SessionExportRequest
from cayu.collaboration.participants import VersionOne
from cayu.collaboration.prepared_admission import (
    PreparedRecipientAdmission,
    require_secret_free_prepared,
)
from cayu.collaboration.recipient_preparation import (
    ForkRecipientPreparation,
    FreshRecipientPreparation,
    require_secret_free_preparation,
)
from cayu.collaboration.requests import (
    MAX_CONTROL_INITIATOR_BYTES,
    MAX_REQUEST_CLARIFICATIONS,
    Millis,
    RequestAdmissionCommand,
    RequestCommand,
)
from cayu.collaboration.resource_preparation import RequestPlanningResource
from cayu.vaults.redaction import SecretRedactor

MAX_REQUEST_PLANNING_GENERATIONS = 32
MAX_REQUEST_PLANNING_STAGES = 128
MAX_REQUEST_PLANNING_RESOURCES = 32
MAX_REQUEST_PLANNING_PAGE = 32
MAX_REQUEST_PLANNING_RULES = 32
MAX_REQUEST_PLANNING_RECORD_BYTES = 64 * 1024

PlanningInputRevision = Annotated[StrictInt, Field(ge=0, le=MAX_REQUEST_CLARIFICATIONS)]


class RequestPlanningLimits(ContractValue):
    """Required finite ceilings; store/application limits may only narrow them."""

    max_generations: StrictInt = Field(ge=1, le=MAX_REQUEST_PLANNING_GENERATIONS)
    max_stages: StrictInt = Field(ge=1, le=MAX_REQUEST_PLANNING_STAGES)
    max_resources: StrictInt = Field(ge=0, le=MAX_REQUEST_PLANNING_RESOURCES)
    max_record_bytes: StrictInt = Field(ge=1, le=MAX_REQUEST_PLANNING_RECORD_BYTES)
    max_recovery_items: StrictInt = Field(ge=1, le=MAX_REQUEST_PLANNING_PAGE)


class RequestPlanningTimer(ContractValue):
    """Absolute owner-time threshold, not a renewed relative timeout."""

    kind: Literal["timer"] = "timer"
    not_before_ms: Millis
    deadline_at_ms: Millis

    @model_validator(mode="after")
    def finite_interval(self) -> RequestPlanningTimer:
        if self.not_before_ms >= self.deadline_at_ms:
            raise ValueError("Planning timer has no eligible interval.")
        return self


class RequestPlanningPrerequisite(ContractValue):
    """Expected receiving-owner evidence, not a receipt supplied by a caller."""

    kind: Literal["admission"] = "admission"
    reader: ObjectRef
    expected: RequestAdmissionCommand
    deadline_at_ms: Millis

    @model_validator(mode="after")
    def pinned_reader(self) -> RequestPlanningPrerequisite:
        if (
            self.reader.revision is None
            or self.reader.owner.application_scope != self.expected.operation.application_scope
        ):
            raise ValueError("Planning prerequisite requires an exact registered reader.")
        return self


class RequestPlanningDefer(ContractValue):
    decision: Literal["defer"] = "defer"
    prerequisite: Annotated[
        RequestPlanningTimer | RequestPlanningPrerequisite, Field(discriminator="kind")
    ]


class RequestPlanningDecline(ContractValue):
    decision: Literal["decline"] = "decline"
    reason: Identifier


class RequestPlanningClarify(ContractValue):
    """An exact native question proposal; disclosure must still be authenticated."""

    decision: Literal["clarify"] = "clarify"
    opening: ClarificationOpenCommand
    source: SessionExportRequest

    @model_validator(mode="after")
    def exact_source(self) -> RequestPlanningClarify:
        question = self.opening.question
        if (
            self.source.ref != question.source.export
            or self.source.projector != question.source.projector
            or self.source.policy != question.source.policy
            or self.source.source_selection != question.source.selection
            or question.source.audience
            != ObjectRef(
                owner=question.responder.owner,
                kind="participant",
                object_id=question.responder.participant_id,
                incarnation=question.responder.incarnation,
            )
            or self.source.audience
            != OwnerRef(
                application_scope=question.responder.owner.application_scope,
                owner_id=question.responder.participant_id,
                incarnation=question.responder.incarnation,
            )
        ):
            raise ValueError("Planning question source conflicts with its selection.")
        return self


class RequestPlanningContinue(ContractValue):
    """Frozen selection data; the native receiver must authenticate it again."""

    decision: Literal["continue"] = "continue"
    prepared: PreparedRecipientAdmission

    @model_validator(mode="after")
    def exact_kind(self) -> RequestPlanningContinue:
        if self.prepared.target.kind != "continue":
            raise ValueError("CONTINUE planning requires an exact continuation target.")
        return self


class RequestPlanningFresh(ContractValue):
    """Frozen native preparation, consumed only through its retained stage owner."""

    decision: Literal["fresh"] = "fresh"
    preparation: FreshRecipientPreparation
    resources: tuple[RequestPlanningResource, ...] = Field(
        default=(), max_length=MAX_REQUEST_PLANNING_RESOURCES
    )

    @model_validator(mode="after")
    def resource_destinations(self):
        _require_resource_proposal(self.preparation, self.resources)
        return self


class RequestPlanningFork(ContractValue):
    """Frozen source selection and child preflight; native owners supply authority."""

    decision: Literal["fork"] = "fork"
    preparation: ForkRecipientPreparation
    resources: tuple[RequestPlanningResource, ...] = Field(
        default=(), max_length=MAX_REQUEST_PLANNING_RESOURCES
    )

    @model_validator(mode="after")
    def resource_destinations(self):
        _require_resource_proposal(
            self.preparation.base, self.resources, extra_permits=self.preparation.view.permits
        )
        return self


def _require_resource_proposal(preparation, resources, *, extra_permits=()):
    """Frozen recipes cannot switch the child owner or reuse native operations."""
    if not resources and preparation.creation_request.input_artifact_ids:
        raise ValueError("Initial attachments require explicit retained-resource preparation.")
    creation = preparation.creation
    receiving = creation.permit.intent.request
    identities = [creation.permit.operation, receiving.settlement_operation]
    for permit in extra_permits:
        identities.extend((permit.operation, permit.intent.request.settlement_operation))
    for resource in resources:
        target = resource.transfer_permit.intent.request
        if (
            resource.transfer.destination != creation.receiving_owner
            or target.participant != receiving.participant
            or target.expected_lifecycle_revision != receiving.expected_lifecycle_revision
            or target.expected_configuration_revision != receiving.expected_configuration_revision
            or target.admission_generation != receiving.admission_generation
        ):
            raise ValueError("Planned resource differs from its frozen recipient authority.")
        identities.extend(
            (
                resource.acquisition.operation,
                resource.acquisition_permit.operation,
                resource.acquisition_permit.intent.request.settlement_operation,
                resource.transfer.operation,
                resource.transfer_permit.operation,
                target.settlement_operation,
            )
        )
    if len(identities) != len(set(identities)):
        raise ValueError("Planned preparation operations must have distinct exact identities.")


def preparation_stage_count(proposal):
    """Closed bounded stage schedule: optional view, resource pairs, child, admission."""
    if not isinstance(proposal, (RequestPlanningFresh, RequestPlanningFork)):
        raise ValueError("Planning decision has no preparation stage schedule.")
    return (3 if isinstance(proposal, RequestPlanningFork) else 2) + 2 * len(proposal.resources)


RequestPlanningDecision = Annotated[
    RequestPlanningDefer
    | RequestPlanningDecline
    | RequestPlanningClarify
    | RequestPlanningContinue
    | RequestPlanningFresh
    | RequestPlanningFork,
    Field(discriminator="decision"),
]


class RequestPlanningRule(ContractValue):
    input_revision: PlanningInputRevision
    proposal: RequestPlanningDecision


class ConfiguredRequestPlanningPolicy(ContractValue):
    """Closed, deterministic strategy with reconstructable immutable configuration.

    A rule matches one exact effective-input revision. Otherwise the required
    default applies. No callback, model, ambient application state or implicit
    clock participates in evaluation. Owner-time eligibility is checked later
    by the durable planning owner, not inferred from this strategy's choice.
    """

    schema_version: VersionOne = 1
    algorithm: Literal["input_revision_rules_v1"] = "input_revision_rules_v1"
    reference: ObjectRef
    limits: RequestPlanningLimits
    rules: tuple[RequestPlanningRule, ...] = Field(max_length=MAX_REQUEST_PLANNING_RULES)
    default: RequestPlanningDecision

    @model_validator(mode="after")
    def canonical_rules(self) -> ConfiguredRequestPlanningPolicy:
        revisions = tuple(rule.input_revision for rule in self.rules)
        if self.reference.revision is None:
            raise ValueError("Planning policy requires a pinned version.")
        if revisions != tuple(sorted(set(revisions))):
            raise ValueError("Planning rules must have unique ordered input revisions.")
        return self


class RequestPlanningPredecessor(ContractValue):
    """Exact state being replaced; a reference alone is not successor authority."""

    operation: OperationRef
    revision: Generation


class RequestPlanningRequest(ContractValue):
    """Complete caller expectation; policy registration and authority remain external."""

    schema_version: VersionOne = 1
    mode: Literal["request_plan"] = "request_plan"
    operation: OperationRef
    expected: RequestCommand
    expected_revision: Generation
    expected_input_revision: PlanningInputRevision
    expected_input_sha256: Commitment
    planning_generation: StrictInt = Field(ge=1, le=MAX_REQUEST_PLANNING_GENERATIONS)
    admission_operation: OperationRef
    admission_generation: Generation
    initiator: InitiatorBinding
    policy: ObjectRef
    policy_sha256: Commitment
    limits: RequestPlanningLimits
    deadline_at_ms: Millis
    predecessor: RequestPlanningPredecessor | None

    @model_validator(mode="after")
    def coherent(self) -> RequestPlanningRequest:
        parent = self.expected.operation
        recipient = self.expected.intent.selection.recipient.reference
        namespace = (parent.application_scope, parent.namespace_incarnation, parent.generation)
        operations = (self.operation, self.admission_operation)
        if (
            any(
                (item.application_scope, item.namespace_incarnation, item.generation) != namespace
                for item in operations
            )
            or len(set((parent, *operations))) != 3
        ):
            raise ValueError("Planning requires distinct operations in the request namespace.")
        if (
            self.initiator.issuer != recipient.owner
            or self.initiator.participant
            != ObjectRef(
                owner=recipient.owner,
                kind="participant",
                object_id=recipient.participant_id,
                incarnation=recipient.incarnation,
            )
            or self.policy.owner.application_scope != parent.application_scope
            or self.policy.revision is None
            or self.planning_generation > self.limits.max_generations
            or not self.expected.intent.selection.accepted_at_ms
            < self.deadline_at_ms
            <= self.expected.intent.selection.expires_at_ms
        ):
            raise ValueError("Planning identity or bounds conflict with the accepted request.")
        if (self.predecessor is None) != (self.planning_generation == 1):
            raise ValueError("Successor planning requires an exact predecessor.")
        if self.predecessor is not None:
            previous = self.predecessor.operation
            if (
                previous.application_scope,
                previous.namespace_incarnation,
                previous.generation,
            ) != namespace or previous in (parent, *operations):
                raise ValueError("Planning predecessor conflicts with its request.")
        return self


class RequestPlanningControl(ContractValue):
    """Exact disposition of one retained revision; never a new execution permit."""

    expected: RequestPlanningRequest
    expected_revision: Generation
    kind: Literal["cancelled", "expired"]
    initiator: InitiatorBinding

    @model_validator(mode="after")
    def same_owner(self) -> RequestPlanningControl:
        canonical_bounded_durable_json_bytes(
            snapshot_input(self.initiator),
            "planning control initiator",
            max_bytes=MAX_CONTROL_INITIATOR_BYTES,
            max_nodes=MAX_NODES,
            max_nesting=MAX_DEPTH,
        )
        if self.initiator.issuer != self.expected.expected.intent.selection.reference.owner:
            raise ValueError("Planning control belongs to another owner.")
        return self


class _EvaluationInput(ContractValue):
    input_revision: PlanningInputRevision


def _checked_policy(policy: object, redactor: SecretRedactor) -> ConfiguredRequestPlanningPolicy:
    checked = prepare_contract(ConfiguredRequestPlanningPolicy, policy, redactor=redactor)
    for proposal in (checked.default, *(rule.proposal for rule in checked.rules)):
        if isinstance(proposal, (RequestPlanningFresh, RequestPlanningFork)) and (
            len(proposal.resources) > checked.limits.max_resources
            or preparation_stage_count(proposal) > checked.limits.max_stages
        ):
            raise CollaborationContractError("Preparation exceeds its resource or stage ceiling.")
        if isinstance(proposal, RequestPlanningFresh):
            require_secret_free_preparation(proposal.preparation, redactor)
        elif isinstance(proposal, RequestPlanningFork):
            require_secret_free_preparation(proposal.preparation.base, redactor)
        prepared = (
            proposal.prepared
            if isinstance(proposal, RequestPlanningContinue)
            else (
                proposal.prerequisite.expected.prepared
                if isinstance(proposal, RequestPlanningDefer)
                and isinstance(proposal.prerequisite, RequestPlanningPrerequisite)
                else None
            )
        )
        if prepared is not None:
            require_secret_free_prepared(prepared, redactor)
    if len(contract_bytes(checked, redactor=redactor)) > checked.limits.max_record_bytes:
        raise CollaborationContractError("Planning policy exceeds its retained byte ceiling.")
    return checked


def planning_policy_commitment(
    policy: ConfiguredRequestPlanningPolicy, *, redactor: SecretRedactor
) -> str:
    """Complete canonical configuration identity, not proof of registration."""
    checked = _checked_policy(policy, redactor)
    return sha256(contract_bytes(checked, redactor=redactor)).hexdigest()


def evaluate_configured_request_policy(
    policy: ConfiguredRequestPlanningPolicy,
    *,
    input_revision: int,
    redactor: SecretRedactor,
) -> RequestPlanningDecision:
    """Evaluate retained values only; the caller owns durable input authentication."""
    checked = _checked_policy(policy, redactor)
    revision = prepare_contract(
        _EvaluationInput, {"input_revision": input_revision}, redactor=redactor
    ).input_revision
    proposal = _selected_policy_proposal(checked, revision)
    # Return detached revalidated values, not an alias into a registration.
    return prepare_contract(type(proposal), proposal, redactor=redactor)


def _selected_policy_proposal(policy, input_revision):
    """Closed pure selection for validated immutable policy/receipt reconstruction."""
    return next(
        (rule.proposal for rule in policy.rules if rule.input_revision == input_revision),
        policy.default,
    )
