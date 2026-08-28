"""Durable handoff from accepted maintenance planning to explicit review.

This module materializes one semantically accepted, read-only planning result as
an immutable pending replacement and proposal.  Publication is atomic across the
entry revision, chunks, exact source evidence, proposal record, accepted-plan
record, change outbox row, and replay receipt.  It never activates the
replacement, archives a source, or publishes a relation; those effects remain
owned by :class:`cayu.storage.knowledge_review.KnowledgeReviewWorkflow` after an
external decision.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from functools import cached_property
from hashlib import sha256
from typing import Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._validation import (
    canonical_durable_json_bytes,
    require_durable_clean_nonblank,
)
from cayu.knowledge_maintenance import (
    KnowledgeMaintenanceRoutedCandidate,
    KnowledgeMaintenanceRoutingRequest,
    KnowledgeMaintenanceRoutingResult,
)
from cayu.knowledge_maintenance_planning import (
    KNOWLEDGE_MAINTENANCE_DETERMINISTIC_EVALUATOR_VERSION,
    KnowledgeMaintenanceEvaluationVerdict,
    KnowledgeMaintenancePlanDraft,
    KnowledgeMaintenancePlanEndpointKind,
    KnowledgeMaintenancePlanEvaluation,
    KnowledgeMaintenancePlanningOutcome,
    KnowledgeMaintenancePlanningResult,
)
from cayu.storage.knowledge_indexer import (
    DEFAULT_KNOWLEDGE_CHUNK_OVERLAP_BYTES,
    DEFAULT_KNOWLEDGE_CHUNK_TARGET_BYTES,
    DEFAULT_KNOWLEDGE_INDEX_MAX_CHUNKS,
    MIN_KNOWLEDGE_TEXT_BYTES,
    KnowledgeIndexer,
    KnowledgeIndexRequest,
    KnowledgeIndexResult,
)
from cayu.storage.memory import (
    MAX_KNOWLEDGE_MAINTENANCE_BYTES,
    MAX_KNOWLEDGE_REVISION,
    KnowledgeAccessScope,
    KnowledgeActorType,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeEvidence,
    KnowledgeEvidenceDisposition,
    KnowledgeEvidenceRole,
    KnowledgeMaintenanceProposal,
    KnowledgeRelation,
    KnowledgeRelationKind,
    KnowledgeRevisionRef,
    KnowledgeStatus,
    copy_knowledge_access_scope,
    copy_knowledge_maintenance_proposal,
    copy_knowledge_revision_ref,
    prepare_knowledge_publication,
)

KNOWLEDGE_MAINTENANCE_PROPOSAL_PUBLICATION_SCHEMA_VERSION = 1
KNOWLEDGE_MAINTENANCE_PROPOSAL_PIPELINE_VERSION = (
    "cayu.knowledge-maintenance-proposal-publication.v1"
)

_IDENTITY_MAX_BYTES = 256
_SHA256_HEX = frozenset("0123456789abcdef")
_MAINTENANCE_METADATA_KEY = "cayu_knowledge_maintenance"


class _PersistenceModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        validate_default=True,
    )


def _clean(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"`{field_name}` must be a string.")
    value = require_durable_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > _IDENTITY_MAX_BYTES:
        raise ValueError(f"`{field_name}` must be at most {_IDENTITY_MAX_BYTES} UTF-8 bytes.")
    return value


def _fingerprint(value: object, field_name: str) -> str:
    return sha256(canonical_durable_json_bytes(value, field_name)).hexdigest()


def _validate_sha256(value: object, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"`{field_name}` must be lowercase SHA-256 hex.")
    return value


def _utc(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"`{field_name}` must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"`{field_name}` must be timezone-aware.")
    return value.astimezone(UTC)


class KnowledgeMaintenanceProposalPublicationOutcome(StrEnum):
    """Durable outcome of one accepted-plan handoff."""

    PENDING_PERSISTED = "pending_persisted"
    EXISTING_PENDING = "existing_pending"
    EXISTING_DECIDED = "existing_decided"


class KnowledgeMaintenanceAcceptedPlan(_PersistenceModel):
    """Attempt-independent accepted semantic plan retained for review.

    Provider usage and processing timestamps remain attempt telemetry on
    ``KnowledgeMaintenancePlanningResult``.  The durable review artifact keeps
    the exact plan, evaluation, routing/configuration bindings, and component
    identities that determine review authority.  Equivalent accepted attempts
    therefore converge on one proposal identity.
    """

    schema_version: Literal[1] = KNOWLEDGE_MAINTENANCE_PROPOSAL_PUBLICATION_SCHEMA_VERSION
    request_id: StrictStr
    request_fingerprint: StrictStr
    routing_result_fingerprint: StrictStr
    routing_configuration_fingerprint: StrictStr
    configuration_fingerprint: StrictStr
    planner_id: StrictStr
    planner_version: StrictStr
    evaluator_id: StrictStr
    evaluator_version: StrictStr
    deterministic_evaluator_version: Literal[
        "cayu.knowledge-maintenance-deterministic-evaluator.v1"
    ] = KNOWLEDGE_MAINTENANCE_DETERMINISTIC_EVALUATOR_VERSION
    plan: KnowledgeMaintenancePlanDraft
    evaluation: KnowledgeMaintenancePlanEvaluation

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("`schema_version` must be the integer 1.")
        return value

    @field_validator(
        "request_id",
        "planner_id",
        "planner_version",
        "evaluator_id",
        "evaluator_version",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator(
        "request_fingerprint",
        "routing_result_fingerprint",
        "routing_configuration_fingerprint",
        "configuration_fingerprint",
    )
    @classmethod
    def validate_fingerprint(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("plan", mode="before")
    @classmethod
    def copy_plan(cls, value: object) -> object:
        if isinstance(value, KnowledgeMaintenancePlanDraft):
            return value.model_dump(mode="python")
        return value

    @field_validator("evaluation", mode="before")
    @classmethod
    def copy_evaluation(cls, value: object) -> object:
        if isinstance(value, KnowledgeMaintenancePlanEvaluation):
            return value.model_dump(mode="python")
        return value

    @model_validator(mode="after")
    def validate_acceptance(self) -> KnowledgeMaintenanceAcceptedPlan:
        if self.plan.routing_request_fingerprint != self.request_fingerprint:
            raise ValueError("The accepted plan does not bind the routing request.")
        if self.plan.routing_result_fingerprint != self.routing_result_fingerprint:
            raise ValueError("The accepted plan does not bind the routing result.")
        if self.plan.configuration_fingerprint != self.configuration_fingerprint:
            raise ValueError("The accepted plan does not bind the planning configuration.")
        if self.evaluation.plan_fingerprint != self.plan.fingerprint:
            raise ValueError("The accepted evaluation does not bind the plan.")
        if self.evaluation.routing_result_fingerprint != self.routing_result_fingerprint:
            raise ValueError("The accepted evaluation does not bind the routing result.")
        if self.evaluation.configuration_fingerprint != self.configuration_fingerprint:
            raise ValueError("The accepted evaluation does not bind the planning configuration.")
        if (
            self.evaluation.evaluator_id != self.evaluator_id
            or self.evaluation.evaluator_version != self.evaluator_version
        ):
            raise ValueError("The accepted evaluation does not bind the evaluator identity.")
        if self.evaluation.deterministic_evaluator_version != (
            self.deterministic_evaluator_version
        ):
            raise ValueError("The accepted evaluation does not bind the deterministic evaluator.")
        if (
            not self.evaluation.evaluator_invoked
            or self.evaluation.verdict is not KnowledgeMaintenanceEvaluationVerdict.ACCEPTED
            or self.evaluation.findings
        ):
            raise ValueError("A durable accepted plan requires an invoked accepted evaluation.")
        if (
            len(
                canonical_durable_json_bytes(
                    self.model_dump(mode="json"),
                    "knowledge maintenance accepted plan",
                )
            )
            > MAX_KNOWLEDGE_MAINTENANCE_BYTES
        ):
            raise ValueError("Knowledge maintenance accepted plan exceeds its byte ceiling.")
        return self

    @cached_property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "contract": "cayu.knowledge-maintenance-accepted-plan.v1",
                "accepted_plan": self.model_dump(mode="json"),
            },
            "knowledge maintenance accepted plan fingerprint",
        )


class KnowledgeMaintenanceProposalPublicationReceipt(_PersistenceModel):
    """Immutable replay evidence for one atomic pending proposal publication."""

    schema_version: Literal[1] = KNOWLEDGE_MAINTENANCE_PROPOSAL_PUBLICATION_SCHEMA_VERSION
    operation_id: StrictStr
    proposal_id: StrictStr
    proposal_fingerprint: StrictStr
    accepted_plan_fingerprint: StrictStr
    request_sha256: StrictStr
    replacement: KnowledgeRevisionRef
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

    @field_validator(
        "proposal_fingerprint",
        "accepted_plan_fingerprint",
        "request_sha256",
    )
    @classmethod
    def validate_fingerprint(cls, value: str, info) -> str:
        return _validate_sha256(value, info.field_name)

    @field_validator("replacement", mode="before")
    @classmethod
    def copy_replacement(cls, value: object) -> object:
        if isinstance(value, KnowledgeRevisionRef):
            return copy_knowledge_revision_ref(value)
        return value

    @field_validator("committed_at")
    @classmethod
    def validate_committed_at(cls, value: datetime) -> datetime:
        return _utc(value, "committed_at")

    @field_validator("replayed", mode="before")
    @classmethod
    def validate_replayed(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError("`replayed` must be a boolean.")
        return value


class KnowledgeMaintenanceProposalPublication(_PersistenceModel):
    """One durable proposal, accepted semantic record, and pending replacement."""

    proposal: KnowledgeMaintenanceProposal
    accepted_plan: KnowledgeMaintenanceAcceptedPlan
    replacement: KnowledgeEntry
    receipt: KnowledgeMaintenanceProposalPublicationReceipt
    outcome: KnowledgeMaintenanceProposalPublicationOutcome

    @field_validator("proposal", mode="before")
    @classmethod
    def copy_proposal(cls, value: object) -> object:
        if isinstance(value, KnowledgeMaintenanceProposal):
            return value.model_dump(mode="python")
        return value

    @field_validator("accepted_plan", mode="before")
    @classmethod
    def copy_accepted_plan(cls, value: object) -> object:
        if isinstance(value, KnowledgeMaintenanceAcceptedPlan):
            return value.model_dump(mode="python")
        return value

    @field_validator("replacement", mode="before")
    @classmethod
    def copy_replacement(cls, value: object) -> object:
        if isinstance(value, KnowledgeEntry):
            return value.model_dump(mode="python")
        return value

    @field_validator("receipt", mode="before")
    @classmethod
    def copy_receipt(cls, value: object) -> object:
        if isinstance(value, KnowledgeMaintenanceProposalPublicationReceipt):
            return value.model_dump(mode="python")
        return value

    @model_validator(mode="after")
    def validate_publication(self) -> KnowledgeMaintenanceProposalPublication:
        replacement_ref = KnowledgeRevisionRef(
            entry_id=self.replacement.id,
            revision=self.replacement.revision,
        )
        if self.proposal.replacement != replacement_ref:
            raise ValueError("The proposal does not bind the pending replacement.")
        if self.replacement.status is not KnowledgeStatus.PENDING:
            raise ValueError("A proposal publication requires a pending replacement.")
        if self.receipt.proposal_id != self.proposal.id:
            raise ValueError("The publication receipt does not bind the proposal.")
        if self.receipt.proposal_fingerprint != self.proposal.fingerprint:
            raise ValueError("The publication receipt does not bind the proposal fingerprint.")
        if self.receipt.accepted_plan_fingerprint != self.accepted_plan.fingerprint:
            raise ValueError("The publication receipt does not bind the accepted plan.")
        if self.receipt.replacement != replacement_ref:
            raise ValueError("The publication receipt does not bind the pending replacement.")
        if self.proposal.sources != list(self.accepted_plan.plan.source_references):
            raise ValueError("The publication proposal does not bind accepted plan sources.")
        if self.proposal.policy_id != self.accepted_plan.plan.policy_id:
            raise ValueError("The publication proposal does not bind the accepted policy.")
        if self.proposal.metadata.get("accepted_plan_fingerprint") != (
            self.accepted_plan.fingerprint
        ):
            raise ValueError("The publication proposal does not bind the accepted plan.")
        replacement_draft = self.accepted_plan.plan.replacement
        if (
            self.replacement.text != replacement_draft.text
            or self.replacement.title != replacement_draft.title
            or self.replacement.kind != replacement_draft.kind
            or self.replacement.aspects != list(replacement_draft.aspects)
            or self.replacement.impact_targets != list(replacement_draft.impact_targets)
            or self.replacement.created_at != self.proposal.created_at
            or self.replacement.updated_at != self.proposal.created_at
        ):
            raise ValueError("The publication replacement does not bind the accepted draft.")
        if self.receipt.committed_at < self.proposal.created_at:
            raise ValueError("The publication receipt cannot predate its proposal.")
        if self.outcome is KnowledgeMaintenanceProposalPublicationOutcome.PENDING_PERSISTED:
            if self.receipt.replayed:
                raise ValueError("A newly persisted proposal cannot carry a replay receipt.")
        elif not self.receipt.replayed:
            raise ValueError("An existing proposal publication requires replay evidence.")
        return self


class KnowledgeMaintenanceProposalPublisherConfig(_PersistenceModel):
    """Application-owned publication identity and deterministic chunk bounds."""

    schema_version: Literal[1] = KNOWLEDGE_MAINTENANCE_PROPOSAL_PUBLICATION_SCHEMA_VERSION
    publisher_id: StrictStr
    publisher_version: StrictStr
    created_by: StrictStr = "knowledge_maintenance"
    proposed_by: StrictStr = "knowledge_maintenance"
    pipeline_version: StrictStr = KNOWLEDGE_MAINTENANCE_PROPOSAL_PIPELINE_VERSION
    chunk_target_bytes: StrictInt = DEFAULT_KNOWLEDGE_CHUNK_TARGET_BYTES
    chunk_overlap_bytes: StrictInt = DEFAULT_KNOWLEDGE_CHUNK_OVERLAP_BYTES
    max_chunks: StrictInt = DEFAULT_KNOWLEDGE_INDEX_MAX_CHUNKS

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("`schema_version` must be the integer 1.")
        return value

    @field_validator(
        "publisher_id",
        "publisher_version",
        "created_by",
        "proposed_by",
        "pipeline_version",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator("chunk_target_bytes", "chunk_overlap_bytes", "max_chunks")
    @classmethod
    def validate_chunk_bound(cls, value: int, info) -> int:
        if info.field_name == "chunk_overlap_bytes":
            if value < 0:
                raise ValueError("`chunk_overlap_bytes` must be non-negative.")
            return value
        if value <= 0:
            raise ValueError(f"`{info.field_name}` must be greater than 0.")
        if info.field_name == "chunk_target_bytes" and value < MIN_KNOWLEDGE_TEXT_BYTES:
            raise ValueError(f"`chunk_target_bytes` must be at least {MIN_KNOWLEDGE_TEXT_BYTES}.")
        return value

    @model_validator(mode="after")
    def validate_chunk_bounds(self) -> KnowledgeMaintenanceProposalPublisherConfig:
        if self.chunk_overlap_bytes >= self.chunk_target_bytes:
            raise ValueError("`chunk_overlap_bytes` must be less than `chunk_target_bytes`.")
        if self.chunk_overlap_bytes > self.chunk_target_bytes // 2:
            raise ValueError("`chunk_overlap_bytes` must be at most half `chunk_target_bytes`.")
        if self.max_chunks > DEFAULT_KNOWLEDGE_INDEX_MAX_CHUNKS:
            raise ValueError(f"`max_chunks` must be at most {DEFAULT_KNOWLEDGE_INDEX_MAX_CHUNKS}.")
        return self

    @cached_property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "contract": "cayu.knowledge-maintenance-proposal-publisher-config.v1",
                "config": self.model_dump(mode="json"),
            },
            "knowledge maintenance proposal publisher configuration",
        )


class KnowledgeMaintenanceProposalPublicationConflict(ValueError):
    """A proposal publication identity is occupied by different material."""

    def __init__(self, reason: str) -> None:
        self.reason = _clean(reason, "reason")
        super().__init__("Knowledge maintenance proposal publication conflicts with stored state.")


class _ProposalStore(Protocol):
    async def publish_maintenance_proposal(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk],
        *,
        evidence: list[KnowledgeEvidence],
        proposal: KnowledgeMaintenanceProposal,
        accepted_plan: KnowledgeMaintenanceAcceptedPlan,
        operation_id: str,
        access_scope: KnowledgeAccessScope,
    ) -> KnowledgeMaintenanceProposalPublicationReceipt: ...

    async def load_maintenance_proposal_publication(
        self,
        proposal_id: str,
        *,
        access_scope: KnowledgeAccessScope,
    ) -> KnowledgeMaintenanceProposalPublication | None: ...


class KnowledgeMaintenanceProposalPublisher:
    """Persist accepted planning output as one immutable pending review artifact."""

    def __init__(
        self,
        store: _ProposalStore,
        *,
        access_scope: KnowledgeAccessScope,
        config: KnowledgeMaintenanceProposalPublisherConfig,
    ) -> None:
        for method_name in (
            "publish_maintenance_proposal",
            "load_maintenance_proposal_publication",
        ):
            if not callable(getattr(store, method_name, None)):
                raise TypeError("store must implement maintenance proposal publication.")
        if type(access_scope) is not KnowledgeAccessScope:
            raise TypeError("access_scope must be a KnowledgeAccessScope.")
        if type(config) is not KnowledgeMaintenanceProposalPublisherConfig:
            raise TypeError("config must be a KnowledgeMaintenanceProposalPublisherConfig.")
        self._store = store
        self._access_scope = copy_knowledge_access_scope(access_scope)
        self._config = KnowledgeMaintenanceProposalPublisherConfig.model_validate(
            config.model_dump(mode="python")
        )

    @property
    def config(self) -> KnowledgeMaintenanceProposalPublisherConfig:
        return KnowledgeMaintenanceProposalPublisherConfig.model_validate(
            self._config.model_dump(mode="python")
        )

    async def publish(
        self,
        routing_request: KnowledgeMaintenanceRoutingRequest,
        routing_result: KnowledgeMaintenanceRoutingResult,
        planning_result: KnowledgeMaintenancePlanningResult,
    ) -> KnowledgeMaintenanceProposalPublication:
        """Materialize one accepted result without granting lifecycle authority."""

        request, result, planning = _copy_publication_inputs(
            routing_request,
            routing_result,
            planning_result,
        )
        accepted_plan = _accepted_plan(request, result, planning)
        _validate_accepted_plan_against_routing(request, result, accepted_plan)
        candidates = tuple(result.candidates)
        _require_identical_source_boundary(candidates)
        # Publication content must be attempt-independent.  A wall-clock value
        # here would make two equivalent accepted attempts reuse the same
        # deterministic operation id with different request bytes.  The store
        # records the actual transaction time on the receipt instead.
        proposed_at = max(candidate.entry.updated_at for candidate in candidates)
        publication_key = _publication_key(
            accepted_plan,
            access_scope=self._access_scope,
            config=self._config,
        )
        entry_id = f"maintenance-replacement-{publication_key}"
        proposal_id = f"maintenance-proposal-{publication_key}"
        operation_id = f"maintenance-proposal-publication-{publication_key}"
        indexed = _build_pending_replacement(
            accepted_plan,
            candidates=candidates,
            entry_id=entry_id,
            proposal_id=proposal_id,
            processed_at=proposed_at,
            config=self._config,
        )
        proposal = _build_proposal(
            accepted_plan,
            replacement=indexed.entry,
            proposal_id=proposal_id,
            publication_key=publication_key,
            access_scope=self._access_scope,
            proposed_at=proposed_at,
            config=self._config,
        )
        evidence = _build_source_evidence(
            accepted_plan,
            candidates=candidates,
            replacement=indexed.entry,
            proposed_at=proposed_at,
        )
        request_sha256 = _proposal_publication_request_sha256(
            operation_id,
            indexed.entry,
            indexed.chunks,
            evidence,
            proposal,
            accepted_plan,
        )
        receipt = copy_knowledge_maintenance_proposal_publication_receipt(
            await self._store.publish_maintenance_proposal(
                indexed.entry,
                indexed.chunks,
                evidence=evidence,
                proposal=proposal,
                accepted_plan=accepted_plan,
                operation_id=operation_id,
                access_scope=self._access_scope,
            )
        )
        validate_knowledge_maintenance_proposal_publication_replay(
            receipt,
            operation_id=operation_id,
            proposal=proposal,
            accepted_plan=accepted_plan,
            entry=indexed.entry,
            request_sha256=request_sha256,
        )
        if receipt.replayed:
            loaded = await self._store.load_maintenance_proposal_publication(
                proposal.id,
                access_scope=self._access_scope,
            )
            if loaded is None:
                raise RuntimeError("Proposal replay receipt exists without its durable artifact.")
            replay_receipt = copy_knowledge_maintenance_proposal_publication_receipt(
                receipt,
                replayed=True,
            )
            if (
                loaded.proposal != proposal
                or loaded.accepted_plan != accepted_plan
                or loaded.replacement != indexed.entry
                or loaded.receipt != replay_receipt
            ):
                raise KnowledgeMaintenanceProposalPublicationConflict("malformed_receipt")
            if loaded.outcome not in {
                KnowledgeMaintenanceProposalPublicationOutcome.EXISTING_PENDING,
                KnowledgeMaintenanceProposalPublicationOutcome.EXISTING_DECIDED,
            }:
                raise KnowledgeMaintenanceProposalPublicationConflict("malformed_receipt")
            return loaded
        return KnowledgeMaintenanceProposalPublication(
            proposal=proposal,
            accepted_plan=accepted_plan,
            replacement=indexed.entry,
            receipt=receipt,
            outcome=KnowledgeMaintenanceProposalPublicationOutcome.PENDING_PERSISTED,
        )

    async def load(self, proposal_id: str) -> KnowledgeMaintenanceProposalPublication | None:
        """Load one exact pending or already-decided durable proposal artifact."""

        return await self._store.load_maintenance_proposal_publication(
            _clean(proposal_id, "proposal_id"),
            access_scope=self._access_scope,
        )


def _copy_publication_inputs(
    routing_request: KnowledgeMaintenanceRoutingRequest,
    routing_result: KnowledgeMaintenanceRoutingResult,
    planning_result: KnowledgeMaintenancePlanningResult,
) -> tuple[
    KnowledgeMaintenanceRoutingRequest,
    KnowledgeMaintenanceRoutingResult,
    KnowledgeMaintenancePlanningResult,
]:
    if type(routing_request) is not KnowledgeMaintenanceRoutingRequest:
        raise TypeError("routing_request must be a KnowledgeMaintenanceRoutingRequest.")
    if type(routing_result) is not KnowledgeMaintenanceRoutingResult:
        raise TypeError("routing_result must be a KnowledgeMaintenanceRoutingResult.")
    if type(planning_result) is not KnowledgeMaintenancePlanningResult:
        raise TypeError("planning_result must be a KnowledgeMaintenancePlanningResult.")
    return (
        KnowledgeMaintenanceRoutingRequest.model_validate(
            routing_request.model_dump(mode="python")
        ),
        KnowledgeMaintenanceRoutingResult.model_validate(routing_result.model_dump(mode="python")),
        KnowledgeMaintenancePlanningResult.model_validate(
            planning_result.model_dump(mode="python")
        ),
    )


def _accepted_plan(
    request: KnowledgeMaintenanceRoutingRequest,
    result: KnowledgeMaintenanceRoutingResult,
    planning: KnowledgeMaintenancePlanningResult,
) -> KnowledgeMaintenanceAcceptedPlan:
    if planning.outcome is not KnowledgeMaintenancePlanningOutcome.ACCEPTED:
        raise ValueError("Only an accepted maintenance planning result can be published.")
    if planning.plan is None or planning.evaluation is None:
        raise ValueError("An accepted maintenance planning result is incomplete.")
    if planning.request_id != request.id or planning.request_fingerprint != request.fingerprint:
        raise ValueError("The planning result does not bind the routing request.")
    if planning.routing_result_fingerprint != result.fingerprint:
        raise ValueError("The planning result does not bind the routing result.")
    return KnowledgeMaintenanceAcceptedPlan(
        request_id=planning.request_id,
        request_fingerprint=planning.request_fingerprint,
        routing_result_fingerprint=planning.routing_result_fingerprint,
        routing_configuration_fingerprint=result.configuration_fingerprint,
        configuration_fingerprint=planning.configuration_fingerprint,
        planner_id=planning.planner_id,
        planner_version=planning.planner_version,
        evaluator_id=planning.evaluator_id,
        evaluator_version=planning.evaluator_version,
        plan=planning.plan,
        evaluation=planning.evaluation,
    )


def _validate_accepted_plan_against_routing(
    request: KnowledgeMaintenanceRoutingRequest,
    result: KnowledgeMaintenanceRoutingResult,
    accepted: KnowledgeMaintenanceAcceptedPlan,
) -> None:
    if result.request_id != request.id or result.request_fingerprint != request.fingerprint:
        raise ValueError("The routing result does not bind the routing request.")
    if (
        accepted.request_id != request.id
        or accepted.request_fingerprint != request.fingerprint
        or accepted.routing_result_fingerprint != result.fingerprint
        or accepted.routing_configuration_fingerprint != result.configuration_fingerprint
    ):
        raise ValueError("The accepted plan does not bind the exact routing result.")
    if result.truncated or not result.candidates:
        raise ValueError("A published maintenance plan requires complete non-empty routing.")
    request_signals = {signal.id: signal for signal in request.signals}
    if (
        result.signal_count != len(request_signals)
        or {
            *(signal.id for signal in result.routed_signals),
            *(omission.signal_id for omission in result.omissions),
        }
        != set(request_signals)
        or any(request_signals.get(signal.id) != signal for signal in result.routed_signals)
    ):
        raise ValueError("The routing result dispositions do not bind the request.")
    plan = accepted.plan
    routed_sources = {
        (candidate.reference.entry_id, candidate.reference.revision)
        for candidate in result.candidates
    }
    planned_sources = {
        (reference.entry_id, reference.revision) for reference in plan.source_references
    }
    if planned_sources != routed_sources or len(plan.source_references) != len(routed_sources):
        raise ValueError("The accepted plan does not exactly cover routed sources.")
    if plan.policy_id != request.policy_id:
        raise ValueError("The accepted plan does not bind the routing policy.")
    mapping_by_id = {mapping.id: mapping for mapping in plan.evidence_mappings}
    evidence_sources = {
        (reference.entry_id, reference.revision)
        for mapping in plan.evidence_mappings
        for reference in mapping.source_references
    }
    if evidence_sources != routed_sources:
        raise ValueError("The accepted plan does not exactly cover routed source evidence.")
    relation_sources: set[tuple[str, int]] = set()
    referenced_mapping_ids: set[str] = set()
    for relation in plan.relations:
        source = relation.source_reference
        source_key = source.entry_id, source.revision
        if source_key in relation_sources or source_key not in routed_sources:
            raise ValueError("The accepted plan relation coverage is invalid.")
        relation_sources.add(source_key)
        if (
            relation.kind in {KnowledgeRelationKind.SUPERSEDES, KnowledgeRelationKind.DERIVED_FROM}
            and relation.subject.kind is not KnowledgeMaintenancePlanEndpointKind.REPLACEMENT
        ):
            raise ValueError("The accepted plan relation orientation is invalid.")
        if relation.kind is KnowledgeRelationKind.SUPERSEDES and source.revision >= (
            MAX_KNOWLEDGE_REVISION
        ):
            raise ValueError("The accepted plan cannot supersede an exhausted revision.")
        for mapping_id in relation.evidence_mapping_ids:
            mapping = mapping_by_id.get(mapping_id)
            if mapping is None or source not in mapping.source_references:
                raise ValueError("The accepted relation evidence mapping is invalid.")
            referenced_mapping_ids.add(mapping_id)
    if relation_sources != routed_sources or referenced_mapping_ids != set(mapping_by_id):
        raise ValueError("The accepted plan relation evidence is incomplete.")


def _require_identical_source_boundary(
    candidates: tuple[KnowledgeMaintenanceRoutedCandidate, ...],
) -> None:
    boundaries = {
        (
            candidate.entry.namespace,
            tuple(sorted(candidate.entry.labels.items())),
            candidate.entry.visibility,
        )
        for candidate in candidates
    }
    if len(boundaries) != 1:
        raise ValueError(
            "A maintenance replacement cannot combine different namespace, label, or "
            "visibility boundaries."
        )


def _publication_key(
    accepted_plan: KnowledgeMaintenanceAcceptedPlan,
    *,
    access_scope: KnowledgeAccessScope,
    config: KnowledgeMaintenanceProposalPublisherConfig,
) -> str:
    return _fingerprint(
        {
            "contract": "cayu.knowledge-maintenance-proposal-identity.v1",
            "accepted_plan_fingerprint": accepted_plan.fingerprint,
            "access_scope": access_scope.model_dump(mode="json"),
            "publisher_configuration_fingerprint": config.fingerprint,
        },
        "knowledge maintenance proposal identity",
    )


def _build_pending_replacement(
    accepted_plan: KnowledgeMaintenanceAcceptedPlan,
    *,
    candidates: tuple[KnowledgeMaintenanceRoutedCandidate, ...],
    entry_id: str,
    proposal_id: str,
    processed_at: datetime,
    config: KnowledgeMaintenanceProposalPublisherConfig,
) -> KnowledgeIndexResult:
    first = candidates[0].entry
    expirations = [candidate.entry.expires_at for candidate in candidates]
    expires_at = (
        min(value for value in expirations if value is not None)
        if any(value is not None for value in expirations)
        else None
    )
    metadata = {
        _MAINTENANCE_METADATA_KEY: {
            "proposal_id": proposal_id,
            "accepted_plan_fingerprint": accepted_plan.fingerprint,
            "publisher_id": config.publisher_id,
            "publisher_version": config.publisher_version,
            "pipeline_version": config.pipeline_version,
        }
    }
    indexed = KnowledgeIndexer().build(
        KnowledgeIndexRequest(
            text=accepted_plan.plan.replacement.text,
            entry_id=entry_id,
            namespace=first.namespace,
            labels=dict(first.labels),
            kind=accepted_plan.plan.replacement.kind,
            visibility=first.visibility,
            status=KnowledgeStatus.PENDING,
            created_by_type=KnowledgeActorType.APP,
            created_by=config.created_by,
            source_type="knowledge_maintenance",
            source_id=proposal_id,
            aspects=list(accepted_plan.plan.replacement.aspects),
            impact_targets=list(accepted_plan.plan.replacement.impact_targets),
            expires_at=expires_at,
            title=accepted_plan.plan.replacement.title,
            metadata=metadata,
            chunk_metadata={
                "maintenance_accepted_plan_fingerprint": accepted_plan.fingerprint,
            },
            entry_text_max_bytes=MAX_KNOWLEDGE_MAINTENANCE_BYTES,
            chunk_target_bytes=config.chunk_target_bytes,
            chunk_overlap_bytes=config.chunk_overlap_bytes,
            max_chunks=config.max_chunks,
            skip_unchanged=False,
        )
    )
    if indexed.truncated:
        raise ValueError("The accepted replacement exceeds configured chunk capacity.")
    entry = indexed.entry.model_copy(
        update={"created_at": processed_at, "updated_at": processed_at}
    )
    return indexed.model_copy(update={"entry": entry})


def _build_proposal(
    accepted_plan: KnowledgeMaintenanceAcceptedPlan,
    *,
    replacement: KnowledgeEntry,
    proposal_id: str,
    publication_key: str,
    access_scope: KnowledgeAccessScope,
    proposed_at: datetime,
    config: KnowledgeMaintenanceProposalPublisherConfig,
) -> KnowledgeMaintenanceProposal:
    pending_ref = KnowledgeRevisionRef(entry_id=replacement.id, revision=replacement.revision)
    active_ref = KnowledgeRevisionRef(
        entry_id=replacement.id,
        revision=replacement.revision + 1,
    )
    relations: list[KnowledgeRelation] = []
    for draft in accepted_plan.plan.relations:
        source = draft.source_reference
        if draft.kind is KnowledgeRelationKind.CONTRADICTS:
            endpoints = sorted(
                (active_ref, source),
                key=lambda reference: (reference.entry_id, reference.revision),
            )
            subject, object_ = endpoints
        elif draft.subject.kind is KnowledgeMaintenancePlanEndpointKind.REPLACEMENT:
            subject, object_ = active_ref, source
        else:
            subject, object_ = source, active_ref
        relation_id = "maintenance-relation-" + _fingerprint(
            {
                "publication_key": publication_key,
                "draft_relation_id": draft.id,
                "kind": draft.kind.value,
                "subject": subject.model_dump(mode="json"),
                "object": object_.model_dump(mode="json"),
            },
            "knowledge maintenance durable relation identity",
        )
        relations.append(
            KnowledgeRelation(
                id=relation_id,
                subject=subject,
                object=object_,
                kind=draft.kind,
                created_by_type=KnowledgeActorType.APP,
                created_by=config.created_by,
                policy_id=accepted_plan.plan.policy_id,
                created_at=proposed_at,
                metadata={
                    "planning_relation_id": draft.id,
                    "accepted_plan_fingerprint": accepted_plan.fingerprint,
                },
            )
        )
    return KnowledgeMaintenanceProposal(
        id=proposal_id,
        replacement=pending_ref,
        sources=list(accepted_plan.plan.source_references),
        relations=relations,
        access_scope=access_scope,
        policy_id=accepted_plan.plan.policy_id,
        proposed_by_type=KnowledgeActorType.APP,
        proposed_by=config.proposed_by,
        created_at=proposed_at,
        rationale=accepted_plan.plan.rationale,
        evidence_summary=accepted_plan.plan.evidence_summary,
        metadata={
            "accepted_plan_fingerprint": accepted_plan.fingerprint,
            "publisher_configuration_fingerprint": config.fingerprint,
        },
    )


def _build_source_evidence(
    accepted_plan: KnowledgeMaintenanceAcceptedPlan,
    *,
    candidates: tuple[KnowledgeMaintenanceRoutedCandidate, ...],
    replacement: KnowledgeEntry,
    proposed_at: datetime,
) -> list[KnowledgeEvidence]:
    result: list[KnowledgeEvidence] = []
    for ordinal, candidate in enumerate(candidates):
        source = candidate.entry
        source_document = source.model_dump(mode="json")
        source_hash = _fingerprint(source_document, "maintenance source revision")
        evidence_id = "maintenance-evidence-" + _fingerprint(
            {
                "accepted_plan_fingerprint": accepted_plan.fingerprint,
                "replacement_entry_id": replacement.id,
                "source": candidate.reference.model_dump(mode="json"),
            },
            "knowledge maintenance source evidence identity",
        )
        result.append(
            KnowledgeEvidence(
                id=evidence_id,
                entry_id=replacement.id,
                entry_revision=replacement.revision,
                role=KnowledgeEvidenceRole.ORIGIN,
                source_type="knowledge_revision",
                source_id=source.id,
                source_revision=str(source.revision),
                source_hash=source_hash,
                locator={
                    "entry_id": source.id,
                    "revision": source.revision,
                },
                disposition=KnowledgeEvidenceDisposition.LIVE,
                created_at=proposed_at,
                metadata={
                    "source_ordinal": ordinal,
                    "accepted_plan_fingerprint": accepted_plan.fingerprint,
                },
            )
        )
    return result


def _proposal_publication_request_sha256(
    operation_id: str,
    entry: KnowledgeEntry,
    chunks: list[KnowledgeChunk],
    evidence: list[KnowledgeEvidence],
    proposal: KnowledgeMaintenanceProposal,
    accepted_plan: KnowledgeMaintenanceAcceptedPlan,
) -> str:
    return _fingerprint(
        {
            "contract": "cayu.knowledge-maintenance-proposal-publication.v1",
            "operation_id": operation_id,
            "entry": entry.model_dump(mode="json"),
            "chunks": [
                chunk.model_dump(mode="json")
                for chunk in sorted(chunks, key=lambda item: item.chunk_index)
            ],
            "evidence": [
                item.model_dump(mode="json") for item in sorted(evidence, key=lambda item: item.id)
            ],
            "proposal": proposal.model_dump(mode="json"),
            "accepted_plan": accepted_plan.model_dump(mode="json"),
        },
        "knowledge maintenance proposal publication request",
    )


def prepare_knowledge_maintenance_proposal_publication(
    entry: KnowledgeEntry,
    chunks: list[KnowledgeChunk],
    *,
    evidence: list[KnowledgeEvidence],
    proposal: KnowledgeMaintenanceProposal,
    accepted_plan: KnowledgeMaintenanceAcceptedPlan,
    operation_id: str,
) -> tuple[
    str,
    KnowledgeEntry,
    list[KnowledgeChunk],
    list[KnowledgeEvidence],
    KnowledgeMaintenanceProposal,
    KnowledgeMaintenanceAcceptedPlan,
    str,
]:
    """Copy, validate, and fingerprint one composite publication request."""

    (
        operation_id,
        copied_entry,
        copied_chunks,
        copied_evidence,
        _entry_request_sha256,
    ) = prepare_knowledge_publication(
        entry,
        chunks,
        evidence=evidence,
        operation_id=operation_id,
        expected_revision=None,
    )
    copied_proposal = copy_knowledge_maintenance_proposal(proposal)
    if type(accepted_plan) is not KnowledgeMaintenanceAcceptedPlan:
        raise TypeError("accepted_plan must be a KnowledgeMaintenanceAcceptedPlan.")
    copied_plan = KnowledgeMaintenanceAcceptedPlan.model_validate(
        accepted_plan.model_dump(mode="python")
    )
    replacement = KnowledgeRevisionRef(
        entry_id=copied_entry.id,
        revision=copied_entry.revision,
    )
    if copied_entry.status is not KnowledgeStatus.PENDING or copied_entry.revision != 1:
        raise ValueError("A maintenance proposal publication must create pending revision 1.")
    if copied_proposal.replacement != replacement:
        raise ValueError("The maintenance proposal does not bind the published replacement.")
    if copied_proposal.sources != list(copied_plan.plan.source_references):
        raise ValueError("The maintenance proposal does not bind accepted plan sources.")
    if copied_proposal.policy_id != copied_plan.plan.policy_id:
        raise ValueError("The maintenance proposal does not bind the accepted policy.")
    if copied_proposal.metadata.get("accepted_plan_fingerprint") != copied_plan.fingerprint:
        raise ValueError("The maintenance proposal does not bind the accepted plan fingerprint.")
    replacement_draft = copied_plan.plan.replacement
    if (
        copied_entry.text != replacement_draft.text
        or copied_entry.title != replacement_draft.title
        or copied_entry.kind != replacement_draft.kind
        or copied_entry.aspects != list(replacement_draft.aspects)
        or copied_entry.impact_targets != list(replacement_draft.impact_targets)
        or copied_entry.created_at != copied_proposal.created_at
        or copied_entry.updated_at != copied_proposal.created_at
        or copied_entry.source_type != "knowledge_maintenance"
        or copied_entry.source_id != copied_proposal.id
    ):
        raise ValueError("The pending replacement does not bind the accepted replacement draft.")
    maintenance_metadata = copied_entry.metadata.get(_MAINTENANCE_METADATA_KEY)
    if (
        type(maintenance_metadata) is not dict
        or maintenance_metadata.get("proposal_id") != copied_proposal.id
        or maintenance_metadata.get("accepted_plan_fingerprint") != copied_plan.fingerprint
    ):
        raise ValueError("The pending replacement metadata does not bind the accepted plan.")
    if any(
        item.created_at != copied_proposal.created_at
        or item.metadata.get("accepted_plan_fingerprint") != copied_plan.fingerprint
        for item in copied_evidence
    ):
        raise ValueError("Maintenance source evidence does not bind the accepted plan.")
    draft_by_id = {relation.id: relation for relation in copied_plan.plan.relations}
    if len(copied_proposal.relations) != len(draft_by_id):
        raise ValueError("The maintenance proposal does not bind every planned relation.")
    active_replacement = KnowledgeRevisionRef(
        entry_id=copied_entry.id,
        revision=copied_entry.revision + 1,
    )
    seen_drafts: set[str] = set()
    for relation in copied_proposal.relations:
        draft_id = relation.metadata.get("planning_relation_id")
        if type(draft_id) is not str or draft_id in seen_drafts:
            raise ValueError("Durable relations must uniquely bind planned relation identities.")
        draft = draft_by_id.get(draft_id)
        if draft is None or relation.kind is not draft.kind:
            raise ValueError("A durable relation does not bind its planned relation.")
        if relation.metadata.get("accepted_plan_fingerprint") != copied_plan.fingerprint:
            raise ValueError("A durable relation does not bind the accepted plan.")
        source = draft.source_reference
        if draft.kind is KnowledgeRelationKind.CONTRADICTS:
            expected_endpoints = {active_replacement, source}
            if {relation.subject, relation.object} != expected_endpoints:
                raise ValueError("A durable contradiction does not bind planned endpoints.")
        elif relation.subject != active_replacement or relation.object != source:
            raise ValueError("A durable directed relation does not bind planned endpoints.")
        seen_drafts.add(draft_id)
    if seen_drafts != set(draft_by_id):
        raise ValueError("The maintenance proposal omits a planned relation.")
    request_sha256 = _proposal_publication_request_sha256(
        operation_id,
        copied_entry,
        copied_chunks,
        copied_evidence,
        copied_proposal,
        copied_plan,
    )
    return (
        operation_id,
        copied_entry,
        copied_chunks,
        copied_evidence,
        copied_proposal,
        copied_plan,
        request_sha256,
    )


def validate_knowledge_maintenance_proposal_publication_replay(
    receipt: KnowledgeMaintenanceProposalPublicationReceipt,
    *,
    operation_id: str,
    proposal: KnowledgeMaintenanceProposal,
    accepted_plan: KnowledgeMaintenanceAcceptedPlan,
    entry: KnowledgeEntry,
    request_sha256: str,
) -> None:
    """Reject a publication receipt carrying different request material."""

    if (
        receipt.operation_id != operation_id
        or receipt.proposal_id != proposal.id
        or receipt.proposal_fingerprint != proposal.fingerprint
        or receipt.accepted_plan_fingerprint != accepted_plan.fingerprint
        or receipt.replacement != KnowledgeRevisionRef(entry_id=entry.id, revision=entry.revision)
        or receipt.request_sha256 != request_sha256
    ):
        raise KnowledgeMaintenanceProposalPublicationConflict("operation_replay_mismatch")


def copy_knowledge_maintenance_accepted_plan(
    accepted_plan: KnowledgeMaintenanceAcceptedPlan,
) -> KnowledgeMaintenanceAcceptedPlan:
    if type(accepted_plan) is not KnowledgeMaintenanceAcceptedPlan:
        raise TypeError("accepted_plan must be a KnowledgeMaintenanceAcceptedPlan.")
    return KnowledgeMaintenanceAcceptedPlan.model_validate(accepted_plan.model_dump(mode="python"))


def copy_knowledge_maintenance_proposal_publication_receipt(
    receipt: KnowledgeMaintenanceProposalPublicationReceipt,
    *,
    replayed: bool | None = None,
) -> KnowledgeMaintenanceProposalPublicationReceipt:
    if type(receipt) is not KnowledgeMaintenanceProposalPublicationReceipt:
        raise TypeError("receipt must be a KnowledgeMaintenanceProposalPublicationReceipt.")
    values = receipt.model_dump(mode="python")
    if replayed is not None:
        values["replayed"] = replayed
    return KnowledgeMaintenanceProposalPublicationReceipt.model_validate(values)


def copy_knowledge_maintenance_proposal_publication(
    publication: KnowledgeMaintenanceProposalPublication,
) -> KnowledgeMaintenanceProposalPublication:
    if type(publication) is not KnowledgeMaintenanceProposalPublication:
        raise TypeError("publication must be a KnowledgeMaintenanceProposalPublication.")
    return KnowledgeMaintenanceProposalPublication.model_validate(
        publication.model_dump(mode="python")
    )


__all__ = [
    "KNOWLEDGE_MAINTENANCE_PROPOSAL_PIPELINE_VERSION",
    "KNOWLEDGE_MAINTENANCE_PROPOSAL_PUBLICATION_SCHEMA_VERSION",
    "KnowledgeMaintenanceAcceptedPlan",
    "KnowledgeMaintenanceProposalPublication",
    "KnowledgeMaintenanceProposalPublicationConflict",
    "KnowledgeMaintenanceProposalPublicationOutcome",
    "KnowledgeMaintenanceProposalPublicationReceipt",
    "KnowledgeMaintenanceProposalPublisher",
    "KnowledgeMaintenanceProposalPublisherConfig",
]
