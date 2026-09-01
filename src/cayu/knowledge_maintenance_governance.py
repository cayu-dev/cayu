"""Application-owned authority for evaluated knowledge maintenance proposals.

The governance boundary never discovers candidates, plans maintenance, or
schedules work.  It presents one already-persisted and independently evaluated
proposal to an application policy, then either routes it to review or delegates
the exact approved/rejected outcome to the existing atomic maintenance store
transaction.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from threading import Lock
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    require_durable_clean_nonblank,
)
from cayu.knowledge_maintenance_persistence import (
    KnowledgeMaintenanceAcceptedPlan,
    KnowledgeMaintenanceProposalPublication,
    KnowledgeMaintenanceProposalPublicationReceipt,
)
from cayu.storage.memory import (
    KNOWLEDGE_MAINTENANCE_GOVERNANCE_METADATA_KEY,
    KnowledgeAccessScope,
    KnowledgeActorType,
    KnowledgeGovernanceConfig,
    KnowledgeGovernanceMode,
    KnowledgeMaintenanceConflict,
    KnowledgeMaintenanceDecision,
    KnowledgeMaintenanceDecisionKind,
    KnowledgeMaintenanceDecisionReceipt,
    KnowledgeMaintenanceOutcome,
    KnowledgeMaintenanceProposal,
    KnowledgeStore,
    copy_knowledge_access_scope,
    copy_knowledge_maintenance_decision,
    copy_knowledge_maintenance_decision_receipt,
    copy_knowledge_maintenance_proposal,
    knowledge_access_scope_sha256,
)

MAX_KNOWLEDGE_MAINTENANCE_GOVERNANCE_ANNOTATION_BYTES = 4_096
MAX_KNOWLEDGE_MAINTENANCE_GOVERNANCE_REQUEST_BYTES = 512_000
MAX_KNOWLEDGE_MAINTENANCE_GOVERNANCE_RECEIPT_BYTES = 640_000

REVIEWED_MAINTENANCE_ROUTING_POLICY_IDENTITY = "cayu.reviewed-maintenance-routing"
REVIEWED_MAINTENANCE_ROUTING_POLICY_VERSION = "1"
_IDENTITY_MAX_BYTES = 256
_SHA256_HEX = frozenset("0123456789abcdef")
_MAX_RETAINED_POLICY_TASKS = 256
_RETAINED_POLICY_TASKS: set[asyncio.Task[object]] = set()
_RETAINED_POLICY_TASKS_LOCK = Lock()


def _clean(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"`{field_name}` must be a string.")
    clean = require_durable_clean_nonblank(value, field_name)
    if len(clean.encode("utf-8")) > _IDENTITY_MAX_BYTES:
        raise ValueError(f"`{field_name}` must be at most {_IDENTITY_MAX_BYTES} UTF-8 bytes.")
    return clean


def _sha256(value: object, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"`{field_name}` must be lowercase SHA-256 hex.")
    return value


def _fingerprint(value: object, field_name: str) -> str:
    return sha256(canonical_durable_json_bytes(value, field_name)).hexdigest()


class _GovernanceModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        validate_default=True,
    )


class KnowledgeMaintenanceGovernanceDisposition(StrEnum):
    """Application-policy disposition for one evaluated proposal."""

    APPROVE = "approve"
    ROUTE_TO_REVIEW = "route_to_review"
    REJECT = "reject"


class KnowledgeMaintenanceGovernanceRequest(_GovernanceModel):
    """Copied bounded material presented to one maintenance policy."""

    schema_version: Literal[1] = 1
    operation_id: str
    mode: KnowledgeGovernanceMode
    proposal: KnowledgeMaintenanceProposal
    publication_operation_id: str
    publication_request_sha256: str
    accepted_plan_fingerprint: str
    routing_request_fingerprint: str
    routing_result_fingerprint: str
    routing_configuration_fingerprint: str
    planning_configuration_fingerprint: str
    plan_fingerprint: str
    evaluation_fingerprint: str
    planner_identity: str
    planner_version: str
    evaluator_identity: str
    evaluator_version: str
    access_scope: KnowledgeAccessScope
    forbidden_authority_identities: tuple[str, ...]

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("`schema_version` must be the integer 1.")
        return value

    @field_validator(
        "operation_id",
        "publication_operation_id",
        "planner_identity",
        "planner_version",
        "evaluator_identity",
        "evaluator_version",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator(
        "publication_request_sha256",
        "accepted_plan_fingerprint",
        "routing_request_fingerprint",
        "routing_result_fingerprint",
        "routing_configuration_fingerprint",
        "planning_configuration_fingerprint",
        "plan_fingerprint",
        "evaluation_fingerprint",
    )
    @classmethod
    def validate_sha256(cls, value: str, info) -> str:
        return _sha256(value, info.field_name)

    @field_validator("proposal", mode="before")
    @classmethod
    def copy_proposal(cls, value: object) -> object:
        if type(value) is KnowledgeMaintenanceProposal:
            return value.model_dump(mode="python", warnings=False)
        return value

    @field_validator("access_scope", mode="before")
    @classmethod
    def copy_access_scope(cls, value: object) -> object:
        if type(value) is KnowledgeAccessScope:
            return value.model_dump(mode="python", warnings=False)
        return value

    @field_validator("forbidden_authority_identities", mode="before")
    @classmethod
    def copy_forbidden_identities(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, list | tuple):
            raise TypeError("`forbidden_authority_identities` must be a list or tuple.")
        copied = sorted({_clean(item, "forbidden_authority_identities") for item in value})
        if not copied or len(copied) > 16:
            raise ValueError("Between 1 and 16 forbidden authority identities are required.")
        return tuple(copied)

    @model_validator(mode="after")
    def validate_bindings(self) -> KnowledgeMaintenanceGovernanceRequest:
        if self.proposal.access_scope != self.access_scope:
            raise ValueError("Maintenance governance must enforce the published proposal scope.")
        if self.proposal.metadata.get("accepted_plan_fingerprint") != (
            self.accepted_plan_fingerprint
        ):
            raise ValueError("The proposal does not bind the accepted plan fingerprint.")
        if (
            len(
                canonical_durable_json_bytes(
                    self.model_dump(mode="json"),
                    "knowledge maintenance governance request",
                )
            )
            > MAX_KNOWLEDGE_MAINTENANCE_GOVERNANCE_REQUEST_BYTES
        ):
            raise ValueError("Maintenance governance request exceeds its byte ceiling.")
        return self

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "contract": "cayu.knowledge-maintenance-governance-request.v1",
                "request": self.model_dump(mode="json"),
            },
            "knowledge maintenance governance request fingerprint",
        )

    @property
    def access_scope_sha256(self) -> str:
        return knowledge_access_scope_sha256(self.access_scope)


class KnowledgeMaintenanceGovernanceDecision(_GovernanceModel):
    """Exact application-policy decision over one request."""

    schema_version: Literal[1] = 1
    request_sha256: str
    disposition: KnowledgeMaintenanceGovernanceDisposition
    policy_identity: str
    policy_version: str
    code: str
    annotations: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("`schema_version` must be the integer 1.")
        return value

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        return _sha256(value, "request_sha256")

    @field_validator("policy_identity", "policy_version", "code")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator("annotations", mode="before")
    @classmethod
    def copy_annotations(cls, value: object) -> dict[str, Any]:
        copied = copy_durable_json_object(value, "annotations")
        if len(canonical_durable_json_bytes(copied, "annotations")) > (
            MAX_KNOWLEDGE_MAINTENANCE_GOVERNANCE_ANNOTATION_BYTES
        ):
            raise ValueError("Maintenance governance annotations exceed their byte ceiling.")
        return copied

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "contract": "cayu.knowledge-maintenance-governance-decision.v1",
                "decision": self.model_dump(mode="json"),
            },
            "knowledge maintenance governance decision fingerprint",
        )


class KnowledgeMaintenanceGovernanceAuthority(_GovernanceModel):
    """Validated request/decision pair accepted by the persistence boundary."""

    schema_version: Literal[1] = 1
    request: KnowledgeMaintenanceGovernanceRequest
    decision: KnowledgeMaintenanceGovernanceDecision

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("`schema_version` must be the integer 1.")
        return value

    @field_validator("request", "decision", mode="before")
    @classmethod
    def copy_models(cls, value: object) -> object:
        if isinstance(
            value,
            KnowledgeMaintenanceGovernanceRequest | KnowledgeMaintenanceGovernanceDecision,
        ):
            return value.model_dump(mode="python", warnings=False)
        return value

    @model_validator(mode="after")
    def validate_binding(self) -> KnowledgeMaintenanceGovernanceAuthority:
        if self.decision.request_sha256 != self.request.fingerprint:
            raise ValueError("The maintenance governance decision does not bind its request.")
        if self.decision.policy_identity in self.request.forbidden_authority_identities:
            raise ValueError(
                "A generator, planner, evaluator, model, or proposal policy cannot "
                "authorize maintenance."
            )
        if (
            self.request.mode is KnowledgeGovernanceMode.REVIEWED
            and self.decision.disposition
            is not KnowledgeMaintenanceGovernanceDisposition.ROUTE_TO_REVIEW
        ):
            raise ValueError("Reviewed maintenance governance can only route to review.")
        return self


class KnowledgeMaintenanceGovernanceReceipt(_GovernanceModel):
    """Store-authored evidence for one exact maintenance authority outcome."""

    schema_version: Literal[1] = 1
    operation_id: str
    proposal_id: str
    proposal_fingerprint: str
    authority: KnowledgeMaintenanceGovernanceAuthority
    maintenance_receipt: KnowledgeMaintenanceDecisionReceipt | None = None
    committed_at: datetime
    replayed: bool = False

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("`schema_version` must be the integer 1.")
        return value

    @field_validator("operation_id", "proposal_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator("proposal_fingerprint")
    @classmethod
    def validate_proposal_fingerprint(cls, value: str) -> str:
        return _sha256(value, "proposal_fingerprint")

    @field_validator("authority", "maintenance_receipt", mode="before")
    @classmethod
    def copy_nested_models(cls, value: object) -> object:
        if isinstance(
            value,
            KnowledgeMaintenanceGovernanceAuthority | KnowledgeMaintenanceDecisionReceipt,
        ):
            return value.model_dump(mode="python", warnings=False)
        return value

    @field_validator("committed_at")
    @classmethod
    def validate_committed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("`committed_at` must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("replayed", mode="before")
    @classmethod
    def validate_replayed(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError("`replayed` must be a boolean.")
        return value

    @model_validator(mode="after")
    def validate_outcome(self) -> KnowledgeMaintenanceGovernanceReceipt:
        request = self.authority.request
        disposition = self.authority.decision.disposition
        if (
            self.operation_id != request.operation_id
            or self.proposal_id != request.proposal.id
            or self.proposal_fingerprint != request.proposal.fingerprint
        ):
            raise ValueError("Maintenance governance receipt does not bind its authority.")
        expected_outcome = {
            KnowledgeMaintenanceGovernanceDisposition.APPROVE: (
                KnowledgeMaintenanceOutcome.APPLIED
            ),
            KnowledgeMaintenanceGovernanceDisposition.REJECT: (
                KnowledgeMaintenanceOutcome.REJECTED
            ),
        }.get(disposition)
        if expected_outcome is None:
            if self.maintenance_receipt is not None:
                raise ValueError("A routed proposal cannot carry a maintenance outcome.")
        elif (
            self.maintenance_receipt is None
            or self.maintenance_receipt.operation_id != self.operation_id
            or self.maintenance_receipt.proposal_id != self.proposal_id
            or self.maintenance_receipt.proposal_fingerprint != self.proposal_fingerprint
            or self.maintenance_receipt.outcome is not expected_outcome
            or self.maintenance_receipt.committed_at != self.committed_at
        ):
            raise ValueError("Maintenance governance receipt does not bind its atomic outcome.")
        if (
            len(
                canonical_durable_json_bytes(
                    self.model_dump(mode="json"),
                    "knowledge maintenance governance receipt",
                )
            )
            > MAX_KNOWLEDGE_MAINTENANCE_GOVERNANCE_RECEIPT_BYTES
        ):
            raise ValueError("Maintenance governance receipt exceeds its byte ceiling.")
        return self


class KnowledgeMaintenanceGovernancePolicy(Protocol):
    """Application authority for one copied, bounded maintenance request."""

    async def decide_maintenance(
        self,
        request: KnowledgeMaintenanceGovernanceRequest,
    ) -> KnowledgeMaintenanceGovernanceDecision: ...


class KnowledgeMaintenanceGovernancePolicyError(RuntimeError):
    """A maintenance policy failed closed before any store mutation."""

    def __init__(self, code: str) -> None:
        self.code = _clean(code, "code")
        super().__init__("Knowledge maintenance governance policy failed closed.")


def copy_knowledge_maintenance_governance_request(
    request: KnowledgeMaintenanceGovernanceRequest,
) -> KnowledgeMaintenanceGovernanceRequest:
    if type(request) is not KnowledgeMaintenanceGovernanceRequest:
        raise TypeError("Maintenance governance requests must not be subclasses.")
    return KnowledgeMaintenanceGovernanceRequest.model_validate(
        request.model_dump(mode="python", warnings=False)
    )


def copy_knowledge_maintenance_governance_decision(
    decision: KnowledgeMaintenanceGovernanceDecision,
) -> KnowledgeMaintenanceGovernanceDecision:
    if type(decision) is not KnowledgeMaintenanceGovernanceDecision:
        raise TypeError("Maintenance governance decisions must not be subclasses.")
    return KnowledgeMaintenanceGovernanceDecision.model_validate(
        decision.model_dump(mode="python", warnings=False)
    )


def copy_knowledge_maintenance_governance_authority(
    authority: KnowledgeMaintenanceGovernanceAuthority,
) -> KnowledgeMaintenanceGovernanceAuthority:
    if type(authority) is not KnowledgeMaintenanceGovernanceAuthority:
        raise TypeError("Maintenance governance authorities must not be subclasses.")
    return KnowledgeMaintenanceGovernanceAuthority.model_validate(
        authority.model_dump(mode="python", warnings=False)
    )


def copy_knowledge_maintenance_governance_receipt(
    receipt: KnowledgeMaintenanceGovernanceReceipt,
    *,
    replayed: bool | None = None,
) -> KnowledgeMaintenanceGovernanceReceipt:
    if type(receipt) is not KnowledgeMaintenanceGovernanceReceipt:
        raise TypeError("Maintenance governance receipts must not be subclasses.")
    values = receipt.model_dump(mode="python", exclude={"replayed"}, warnings=False)
    if replayed is True and receipt.maintenance_receipt is not None:
        values["maintenance_receipt"] = copy_knowledge_maintenance_decision_receipt(
            receipt.maintenance_receipt,
            replayed=True,
        )
    return KnowledgeMaintenanceGovernanceReceipt(
        **values,
        replayed=receipt.replayed if replayed is None else replayed,
    )


def prepare_knowledge_maintenance_governance_request(
    publication: KnowledgeMaintenanceProposalPublication,
    *,
    operation_id: str,
    mode: KnowledgeGovernanceMode,
) -> KnowledgeMaintenanceGovernanceRequest:
    """Build the exact bounded policy input from a persisted accepted proposal."""

    if type(publication) is not KnowledgeMaintenanceProposalPublication:
        raise TypeError("publication must be a KnowledgeMaintenanceProposalPublication.")
    return _prepare_knowledge_maintenance_governance_request_from_records(
        publication.proposal,
        publication.accepted_plan,
        publication.receipt,
        operation_id=operation_id,
        mode=mode,
    )


def _prepare_knowledge_maintenance_governance_request_from_records(
    proposal: KnowledgeMaintenanceProposal,
    accepted: KnowledgeMaintenanceAcceptedPlan,
    publication_receipt: KnowledgeMaintenanceProposalPublicationReceipt,
    *,
    operation_id: str,
    mode: KnowledgeGovernanceMode,
) -> KnowledgeMaintenanceGovernanceRequest:
    """Reconstruct the only valid governance request for persisted records."""

    forbidden = {
        proposal.proposed_by,
        proposal.policy_id,
        accepted.planner_id,
        accepted.evaluator_id,
    }
    return KnowledgeMaintenanceGovernanceRequest(
        operation_id=operation_id,
        mode=mode,
        proposal=copy_knowledge_maintenance_proposal(proposal),
        publication_operation_id=publication_receipt.operation_id,
        publication_request_sha256=publication_receipt.request_sha256,
        accepted_plan_fingerprint=accepted.fingerprint,
        routing_request_fingerprint=accepted.request_fingerprint,
        routing_result_fingerprint=accepted.routing_result_fingerprint,
        routing_configuration_fingerprint=accepted.routing_configuration_fingerprint,
        planning_configuration_fingerprint=accepted.configuration_fingerprint,
        plan_fingerprint=accepted.plan.fingerprint,
        evaluation_fingerprint=accepted.evaluation.fingerprint,
        planner_identity=accepted.planner_id,
        planner_version=accepted.planner_version,
        evaluator_identity=accepted.evaluator_id,
        evaluator_version=accepted.evaluator_version,
        access_scope=copy_knowledge_access_scope(proposal.access_scope),
        forbidden_authority_identities=tuple(forbidden),
    )


def require_knowledge_maintenance_governance_authority_records(
    authority: KnowledgeMaintenanceGovernanceAuthority,
    proposal: KnowledgeMaintenanceProposal,
    accepted_plan: KnowledgeMaintenanceAcceptedPlan,
    publication_receipt: KnowledgeMaintenanceProposalPublicationReceipt,
) -> KnowledgeMaintenanceGovernanceAuthority:
    """Reject an authority whose request is not an exact persisted projection."""

    copied = copy_knowledge_maintenance_governance_authority(authority)
    expected = _prepare_knowledge_maintenance_governance_request_from_records(
        proposal,
        accepted_plan,
        publication_receipt,
        operation_id=copied.request.operation_id,
        mode=copied.request.mode,
    )
    if copied.request != expected:
        raise KnowledgeMaintenanceConflict("governance_request_mismatch")
    return copied


def _observe_policy_task(task: asyncio.Task[object]) -> None:
    with _RETAINED_POLICY_TASKS_LOCK:
        _RETAINED_POLICY_TASKS.discard(task)
    if not task.cancelled():
        with suppress(BaseException):
            task.exception()


def _start_policy_task(
    decide: Callable[[KnowledgeMaintenanceGovernanceRequest], Awaitable[object]],
    request: KnowledgeMaintenanceGovernanceRequest,
) -> asyncio.Task[object] | None:
    async def invoke() -> object:
        return await decide(request)

    with _RETAINED_POLICY_TASKS_LOCK:
        if len(_RETAINED_POLICY_TASKS) >= _MAX_RETAINED_POLICY_TASKS:
            return None
        task = asyncio.create_task(invoke(), name="cayu-knowledge-maintenance-policy")
        _RETAINED_POLICY_TASKS.add(task)
    task.add_done_callback(_observe_policy_task)
    return task


async def _invoke_policy(
    decide: Callable[[KnowledgeMaintenanceGovernanceRequest], Awaitable[object]],
    request: KnowledgeMaintenanceGovernanceRequest,
    *,
    timeout_seconds: float,
) -> object:
    await asyncio.sleep(0)
    task = _start_policy_task(decide, request)
    if task is None:
        raise KnowledgeMaintenanceGovernancePolicyError("policy_capacity_exhausted")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    try:
        await asyncio.wait({task}, timeout=timeout_seconds)
    except asyncio.CancelledError:
        task.cancel("Knowledge maintenance policy was cancelled by its caller.")
        raise
    if not task.done() or loop.time() >= deadline:
        if not task.done():
            task.cancel("Knowledge maintenance policy exceeded its deadline.")
        raise KnowledgeMaintenanceGovernancePolicyError("policy_timed_out")
    try:
        await asyncio.sleep(0)
    except asyncio.CancelledError:
        task.cancel("Knowledge maintenance policy was cancelled by its caller.")
        raise
    if loop.time() >= deadline:
        raise KnowledgeMaintenanceGovernancePolicyError("policy_timed_out")
    if task.cancelled():
        raise KnowledgeMaintenanceGovernancePolicyError("policy_failed")
    try:
        return task.result()
    except asyncio.CancelledError:
        raise KnowledgeMaintenanceGovernancePolicyError("policy_failed") from None
    except Exception:
        raise KnowledgeMaintenanceGovernancePolicyError("policy_failed") from None


async def decide_knowledge_maintenance_governance(
    request: KnowledgeMaintenanceGovernanceRequest,
    *,
    config: KnowledgeGovernanceConfig,
    policy: KnowledgeMaintenanceGovernancePolicy | None = None,
) -> KnowledgeMaintenanceGovernanceAuthority:
    """Return exact maintenance authority or fail closed before persistence."""

    copied_request = copy_knowledge_maintenance_governance_request(request)
    if type(config) is not KnowledgeGovernanceConfig:
        raise TypeError("config must be a KnowledgeGovernanceConfig.")
    copied_config = KnowledgeGovernanceConfig.model_validate(
        config.model_dump(mode="python", warnings=False)
    )
    if copied_request.mode is not copied_config.mode:
        raise KnowledgeMaintenanceGovernancePolicyError("governance_mode_mismatch")
    if copied_config.mode is KnowledgeGovernanceMode.REVIEWED:
        if policy is not None:
            raise KnowledgeMaintenanceGovernancePolicyError("reviewed_mode_policy_configured")
        return KnowledgeMaintenanceGovernanceAuthority(
            request=copied_request,
            decision=KnowledgeMaintenanceGovernanceDecision(
                request_sha256=copied_request.fingerprint,
                disposition=KnowledgeMaintenanceGovernanceDisposition.ROUTE_TO_REVIEW,
                policy_identity=REVIEWED_MAINTENANCE_ROUTING_POLICY_IDENTITY,
                policy_version=REVIEWED_MAINTENANCE_ROUTING_POLICY_VERSION,
                code="review_required",
            ),
        )
    decide = getattr(policy, "decide_maintenance", None)
    if not callable(decide):
        raise KnowledgeMaintenanceGovernancePolicyError("policy_missing")
    raw = await _invoke_policy(
        decide,
        copy_knowledge_maintenance_governance_request(copied_request),
        timeout_seconds=copied_config.policy_timeout_seconds,
    )
    if type(raw) is not KnowledgeMaintenanceGovernanceDecision:
        raise KnowledgeMaintenanceGovernancePolicyError("policy_output_invalid")
    try:
        decision = copy_knowledge_maintenance_governance_decision(raw)
        if (
            decision.policy_identity != copied_config.policy_identity
            or decision.policy_version != copied_config.policy_version
        ):
            raise ValueError("Policy identity does not match the host configuration.")
        return KnowledgeMaintenanceGovernanceAuthority(
            request=copied_request,
            decision=decision,
        )
    except (TypeError, ValueError):
        raise KnowledgeMaintenanceGovernancePolicyError("policy_output_invalid") from None


def _require_authority_matches_config(
    authority: KnowledgeMaintenanceGovernanceAuthority,
    config: KnowledgeGovernanceConfig,
) -> None:
    """Fence idempotent replay to the original host-owned authority identity."""

    if authority.request.mode is not config.mode:
        raise KnowledgeMaintenanceConflict("governance_operation_reuse")
    decision = authority.decision
    if config.mode is KnowledgeGovernanceMode.REVIEWED:
        expected = (
            REVIEWED_MAINTENANCE_ROUTING_POLICY_IDENTITY,
            REVIEWED_MAINTENANCE_ROUTING_POLICY_VERSION,
        )
    else:
        expected = (config.policy_identity, config.policy_version)
    if (decision.policy_identity, decision.policy_version) != expected:
        raise KnowledgeMaintenanceConflict("governance_operation_reuse")


def _governance_metadata(
    authority: KnowledgeMaintenanceGovernanceAuthority,
) -> dict[str, Any]:
    """Return the bounded capsule persisted inside an atomic maintenance decision."""

    return {
        "schema_version": 1,
        "mode": authority.request.mode.value,
        "decision": authority.decision.model_dump(mode="json"),
    }


def maintenance_decision_from_governance(
    authority: KnowledgeMaintenanceGovernanceAuthority,
    *,
    decided_at: datetime,
) -> KnowledgeMaintenanceDecision:
    """Project approve/reject authority into the existing mechanical contract."""

    copied = copy_knowledge_maintenance_governance_authority(authority)
    disposition = copied.decision.disposition
    if disposition is KnowledgeMaintenanceGovernanceDisposition.ROUTE_TO_REVIEW:
        raise ValueError("A routed proposal does not have a maintenance decision.")
    return KnowledgeMaintenanceDecision(
        operation_id=copied.request.operation_id,
        proposal_id=copied.request.proposal.id,
        proposal_fingerprint=copied.request.proposal.fingerprint,
        kind=(
            KnowledgeMaintenanceDecisionKind.APPROVE
            if disposition is KnowledgeMaintenanceGovernanceDisposition.APPROVE
            else KnowledgeMaintenanceDecisionKind.REJECT
        ),
        reviewer_type=KnowledgeActorType.APP,
        reviewer=copied.decision.policy_identity,
        reason=copied.decision.code,
        decided_at=decided_at,
        metadata={KNOWLEDGE_MAINTENANCE_GOVERNANCE_METADATA_KEY: _governance_metadata(copied)},
    )


def governance_authority_from_maintenance_decision(
    publication: KnowledgeMaintenanceProposalPublication,
    decision: KnowledgeMaintenanceDecision,
) -> KnowledgeMaintenanceGovernanceAuthority | None:
    """Reconstruct tagged automatic authority from immutable durable records."""

    if type(publication) is not KnowledgeMaintenanceProposalPublication:
        raise TypeError("publication must be a KnowledgeMaintenanceProposalPublication.")
    return governance_authority_from_maintenance_records(
        publication.proposal,
        publication.accepted_plan,
        publication.receipt,
        decision,
    )


def governance_authority_from_maintenance_records(
    proposal: KnowledgeMaintenanceProposal,
    accepted_plan: KnowledgeMaintenanceAcceptedPlan,
    publication_receipt: KnowledgeMaintenanceProposalPublicationReceipt,
    decision: KnowledgeMaintenanceDecision,
) -> KnowledgeMaintenanceGovernanceAuthority | None:
    """Validate reserved attribution against the store's immutable records."""

    copied_decision = copy_knowledge_maintenance_decision(decision)
    raw = copied_decision.metadata.get(KNOWLEDGE_MAINTENANCE_GOVERNANCE_METADATA_KEY)
    if raw is None:
        return None
    if type(raw) is not dict or set(raw) != {"schema_version", "mode", "decision"}:
        raise KnowledgeMaintenanceConflict("malformed_governance_attribution")
    try:
        if raw["schema_version"] != 1:
            raise ValueError
        mode = KnowledgeGovernanceMode(raw["mode"])
        request = _prepare_knowledge_maintenance_governance_request_from_records(
            proposal,
            accepted_plan,
            publication_receipt,
            operation_id=copied_decision.operation_id,
            mode=mode,
        )
        governance_decision = KnowledgeMaintenanceGovernanceDecision.model_validate(raw["decision"])
        authority = KnowledgeMaintenanceGovernanceAuthority(
            request=request,
            decision=governance_decision,
        )
        expected = maintenance_decision_from_governance(
            authority,
            decided_at=copied_decision.decided_at,
        )
        if expected != copied_decision:
            raise ValueError
        return authority
    except (TypeError, ValueError):
        raise KnowledgeMaintenanceConflict("malformed_governance_attribution") from None


def governance_receipt_from_maintenance_records(
    publication: KnowledgeMaintenanceProposalPublication,
    decision: KnowledgeMaintenanceDecision,
    maintenance_receipt: KnowledgeMaintenanceDecisionReceipt,
) -> KnowledgeMaintenanceGovernanceReceipt | None:
    authority = governance_authority_from_maintenance_decision(publication, decision)
    if authority is None:
        return None
    return KnowledgeMaintenanceGovernanceReceipt(
        operation_id=authority.request.operation_id,
        proposal_id=authority.request.proposal.id,
        proposal_fingerprint=authority.request.proposal.fingerprint,
        authority=authority,
        maintenance_receipt=copy_knowledge_maintenance_decision_receipt(maintenance_receipt),
        committed_at=maintenance_receipt.committed_at,
    )


async def load_knowledge_maintenance_governance_receipt(
    store: KnowledgeStore,
    *,
    operation_id: str,
    access_scope: KnowledgeAccessScope | None = None,
) -> KnowledgeMaintenanceGovernanceReceipt | None:
    """Load route or terminal governance attribution without invoking a policy."""

    routed = await store.load_maintenance_governance_route(
        operation_id,
        access_scope=access_scope,
    )
    if routed is not None:
        return copy_knowledge_maintenance_governance_receipt(routed)
    decision = await store.load_maintenance_decision(
        operation_id,
        access_scope=access_scope,
    )
    if decision is None:
        return None
    if KNOWLEDGE_MAINTENANCE_GOVERNANCE_METADATA_KEY not in decision.metadata:
        return None
    maintenance_receipt = await store.load_maintenance_decision_receipt(
        operation_id,
        access_scope=access_scope,
    )
    publication = await store.load_maintenance_proposal_publication(
        decision.proposal_id,
        access_scope=access_scope,
    )
    if maintenance_receipt is None or publication is None:
        raise KnowledgeMaintenanceConflict("malformed_governance_attribution")
    governed = governance_receipt_from_maintenance_records(
        publication,
        decision,
        maintenance_receipt,
    )
    return None if governed is None else copy_knowledge_maintenance_governance_receipt(governed)


class KnowledgeMaintenanceGovernor:
    """Explicit high-level operation over persisted evaluated proposals."""

    def __init__(
        self,
        store: KnowledgeStore,
        *,
        config: KnowledgeGovernanceConfig,
        policy: KnowledgeMaintenanceGovernancePolicy | None = None,
    ) -> None:
        if type(config) is not KnowledgeGovernanceConfig:
            raise TypeError("config must be a KnowledgeGovernanceConfig.")
        self._store = store
        self._config = KnowledgeGovernanceConfig.model_validate(
            config.model_dump(mode="python", warnings=False)
        )
        self._policy = policy

    async def govern(
        self,
        *,
        operation_id: str,
        proposal_id: str,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeMaintenanceGovernanceReceipt:
        """Govern one exact persisted proposal and return durable attribution."""

        publication = await self._store.load_maintenance_proposal_publication(
            proposal_id,
            access_scope=access_scope,
        )
        if publication is None:
            raise KnowledgeMaintenanceGovernancePolicyError("proposal_not_found")
        request = prepare_knowledge_maintenance_governance_request(
            publication,
            operation_id=operation_id,
            mode=self._config.mode,
        )
        scope = request.access_scope
        effective_scope = self._store.bound_access_scope() if access_scope is None else access_scope
        if (
            effective_scope is None
            or copy_knowledge_access_scope(effective_scope) != request.access_scope
        ):
            raise KnowledgeMaintenanceGovernancePolicyError("access_scope_mismatch")

        routed = await self._store.load_maintenance_governance_route(
            operation_id,
            access_scope=scope,
        )
        if routed is not None:
            copied = copy_knowledge_maintenance_governance_receipt(routed, replayed=True)
            if copied.authority.request != request:
                raise KnowledgeMaintenanceConflict("governance_operation_reuse")
            _require_authority_matches_config(copied.authority, self._config)
            return copied

        existing_decision = await self._store.load_maintenance_decision(
            operation_id,
            access_scope=scope,
        )
        if existing_decision is not None:
            if existing_decision.proposal_id != publication.proposal.id:
                raise KnowledgeMaintenanceConflict("governance_operation_reuse")
            existing_receipt = await self._store.load_maintenance_decision_receipt(
                operation_id,
                access_scope=scope,
            )
            if existing_receipt is None:
                raise KnowledgeMaintenanceConflict("malformed_governance_attribution")
            governed = governance_receipt_from_maintenance_records(
                publication,
                existing_decision,
                existing_receipt,
            )
            if governed is None or governed.authority.request != request:
                raise KnowledgeMaintenanceConflict("governance_operation_reuse")
            _require_authority_matches_config(governed.authority, self._config)
            return copy_knowledge_maintenance_governance_receipt(governed, replayed=True)

        authority = await decide_knowledge_maintenance_governance(
            request,
            config=self._config,
            policy=self._policy,
        )
        if (
            authority.decision.disposition
            is KnowledgeMaintenanceGovernanceDisposition.ROUTE_TO_REVIEW
        ):
            result = await self._store.record_maintenance_governance_route(
                authority,
                access_scope=scope,
            )
            return copy_knowledge_maintenance_governance_receipt(result)

        # Publication time is deterministic across concurrent callers; actual
        # commit time remains store-authored on both returned receipts.
        maintenance_decision = maintenance_decision_from_governance(
            authority,
            decided_at=publication.receipt.committed_at,
        )
        maintenance_receipt = await self._store.apply_maintenance_decision(
            publication.proposal,
            maintenance_decision,
            access_scope=scope,
        )
        return KnowledgeMaintenanceGovernanceReceipt(
            operation_id=operation_id,
            proposal_id=publication.proposal.id,
            proposal_fingerprint=publication.proposal.fingerprint,
            authority=authority,
            maintenance_receipt=maintenance_receipt,
            committed_at=maintenance_receipt.committed_at,
            replayed=maintenance_receipt.replayed,
        )


__all__ = [
    "KNOWLEDGE_MAINTENANCE_GOVERNANCE_METADATA_KEY",
    "MAX_KNOWLEDGE_MAINTENANCE_GOVERNANCE_ANNOTATION_BYTES",
    "MAX_KNOWLEDGE_MAINTENANCE_GOVERNANCE_RECEIPT_BYTES",
    "MAX_KNOWLEDGE_MAINTENANCE_GOVERNANCE_REQUEST_BYTES",
    "KnowledgeMaintenanceGovernanceAuthority",
    "KnowledgeMaintenanceGovernanceDecision",
    "KnowledgeMaintenanceGovernanceDisposition",
    "KnowledgeMaintenanceGovernancePolicy",
    "KnowledgeMaintenanceGovernancePolicyError",
    "KnowledgeMaintenanceGovernanceReceipt",
    "KnowledgeMaintenanceGovernanceRequest",
    "KnowledgeMaintenanceGovernor",
    "copy_knowledge_maintenance_governance_authority",
    "copy_knowledge_maintenance_governance_decision",
    "copy_knowledge_maintenance_governance_receipt",
    "copy_knowledge_maintenance_governance_request",
    "decide_knowledge_maintenance_governance",
    "governance_authority_from_maintenance_decision",
    "governance_receipt_from_maintenance_records",
    "load_knowledge_maintenance_governance_receipt",
    "maintenance_decision_from_governance",
    "prepare_knowledge_maintenance_governance_request",
]
