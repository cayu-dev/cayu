"""Backend-independent contracts for reviewed knowledge-maintenance decisions."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import canonical_durable_json_bytes, copy_durable_metadata
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.knowledge.records import (
    MAX_KNOWLEDGE_REVISION,
    KnowledgeActorType,
    KnowledgeRevisionRef,
    _bounded_knowledge_identity,
    copy_knowledge_revision_ref,
)
from cayu.knowledge.relations import (
    MAX_KNOWLEDGE_RELATION_BATCH,
    KnowledgeRelation,
    KnowledgeRelationKind,
    _knowledge_relation_identity,
    _knowledge_relation_semantic_key,
    copy_knowledge_relation,
)
from cayu.knowledge.scopes import KnowledgeAccessScope, copy_knowledge_access_scope

KNOWLEDGE_MAINTENANCE_GOVERNANCE_METADATA_KEY = "cayu_knowledge_maintenance_governance"

_SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}\Z")

MAX_KNOWLEDGE_MAINTENANCE_SOURCES = 50
MAX_KNOWLEDGE_MAINTENANCE_BYTES = 256_000
MAX_KNOWLEDGE_MAINTENANCE_TEXT_BYTES = 16_384
MAX_KNOWLEDGE_MAINTENANCE_METADATA_BYTES = 16_384


class KnowledgeMaintenanceDecisionKind(StrEnum):
    """Reviewer disposition for one exact maintenance proposal."""

    APPROVE = "approve"
    REJECT = "reject"


class KnowledgeMaintenanceOutcome(StrEnum):
    """Durable outcome of an applied maintenance decision."""

    APPLIED = "applied"
    REJECTED = "rejected"


class KnowledgeMaintenanceConflict(RuntimeError):
    """A maintenance identity conflicts with immutable durable state."""

    def __init__(self, reason: str) -> None:
        self.reason = require_clean_nonblank(reason, "reason")
        super().__init__("Knowledge maintenance conflicts with durable state.")


class KnowledgeMaintenanceStale(RuntimeError):
    """A reviewed maintenance proposal no longer matches current knowledge."""

    def __init__(self, reason: str) -> None:
        self.reason = require_clean_nonblank(reason, "reason")
        super().__init__("Knowledge maintenance proposal is stale.")


class KnowledgeMaintenanceProposal(BaseModel):
    """Immutable reviewed plan over exact current knowledge revisions.

    Relations bind the replacement's deterministic active successor revision.
    A ``supersedes`` relation archives its object; ``derived_from`` and
    ``contradicts`` preserve their source as active knowledge.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    id: str
    replacement: KnowledgeRevisionRef
    sources: list[KnowledgeRevisionRef]
    relations: list[KnowledgeRelation]
    access_scope: KnowledgeAccessScope
    policy_id: str
    proposed_by_type: KnowledgeActorType = KnowledgeActorType.APP
    proposed_by: str = "app"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    rationale: str
    evidence_summary: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("`schema_version` must be the integer 1.")
        return value

    @field_validator("id", "policy_id", "proposed_by")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _knowledge_maintenance_identity(value, info.field_name)

    @field_validator("replacement", mode="before")
    @classmethod
    def copy_replacement(cls, value):
        if isinstance(value, KnowledgeRevisionRef):
            return copy_knowledge_revision_ref(value)
        return value

    @field_validator("sources", mode="before")
    @classmethod
    def copy_sources(cls, value) -> list[KnowledgeRevisionRef]:
        if type(value) is not list:
            raise ValueError("`sources` must be a list.")
        if not value or len(value) > MAX_KNOWLEDGE_MAINTENANCE_SOURCES:
            raise ValueError(
                "`sources` must contain between 1 and "
                f"{MAX_KNOWLEDGE_MAINTENANCE_SOURCES} exact revisions."
            )
        copied = [
            copy_knowledge_revision_ref(item)
            if isinstance(item, KnowledgeRevisionRef)
            else KnowledgeRevisionRef.model_validate(item)
            for item in value
        ]
        copied.sort(key=lambda item: (item.entry_id, item.revision))
        if len({(item.entry_id, item.revision) for item in copied}) != len(copied):
            raise ValueError("`sources` cannot repeat an exact revision.")
        if len({item.entry_id for item in copied}) != len(copied):
            raise ValueError("`sources` cannot repeat a logical entry.")
        return copied

    @field_validator("relations", mode="before")
    @classmethod
    def copy_relations(cls, value) -> list[KnowledgeRelation]:
        if type(value) is not list:
            raise ValueError("`relations` must be a list.")
        if not value or len(value) > MAX_KNOWLEDGE_RELATION_BATCH:
            raise ValueError(
                f"`relations` must contain between 1 and {MAX_KNOWLEDGE_RELATION_BATCH} records."
            )
        copied = [
            copy_knowledge_relation(item)
            if type(item) is KnowledgeRelation
            else copy_knowledge_relation(KnowledgeRelation.model_validate(item))
            for item in value
        ]
        copied.sort(key=lambda item: item.id)
        if len({item.id for item in copied}) != len(copied):
            raise ValueError("`relations` cannot repeat an identity.")
        semantic_keys = [_knowledge_relation_semantic_key(item) for item in copied]
        if len(set(semantic_keys)) != len(semantic_keys):
            raise ValueError("`relations` cannot repeat one semantic relation.")
        return copied

    @field_validator("access_scope", mode="before")
    @classmethod
    def copy_access_scope(cls, value):
        if isinstance(value, KnowledgeAccessScope):
            return copy_knowledge_access_scope(value)
        return value

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("`created_at` must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("rationale", "evidence_summary")
    @classmethod
    def validate_bounded_text(cls, value: str, info) -> str:
        value = require_clean_nonblank(value, info.field_name)
        if len(value.encode("utf-8")) > MAX_KNOWLEDGE_MAINTENANCE_TEXT_BYTES:
            raise ValueError(
                f"`{info.field_name}` must be at most "
                f"{MAX_KNOWLEDGE_MAINTENANCE_TEXT_BYTES} UTF-8 bytes."
            )
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        copied = copy_durable_metadata(value, "metadata")
        if len(canonical_durable_json_bytes(copied, "knowledge maintenance metadata")) > (
            MAX_KNOWLEDGE_MAINTENANCE_METADATA_BYTES
        ):
            raise ValueError("`metadata` exceeds the knowledge maintenance metadata budget.")
        return copied

    @model_validator(mode="after")
    def validate_plan(self) -> KnowledgeMaintenanceProposal:
        if self.replacement.entry_id in {source.entry_id for source in self.sources}:
            raise ValueError("The replacement cannot also be a maintenance source.")
        if self.replacement.revision >= MAX_KNOWLEDGE_REVISION:
            raise ValueError("The replacement revision cannot advance safely.")
        active_replacement = KnowledgeRevisionRef(
            entry_id=self.replacement.entry_id,
            revision=self.replacement.revision + 1,
        )
        source_keys = {(source.entry_id, source.revision) for source in self.sources}
        referenced_sources: set[tuple[str, int]] = set()
        for relation in self.relations:
            if relation.created_at > self.created_at:
                raise ValueError("A proposed relation cannot postdate its proposal.")
            if relation.policy_id != self.policy_id:
                raise ValueError("Every proposed relation must use the proposal policy identity.")
            if relation.kind is KnowledgeRelationKind.CONTRADICTS:
                endpoints = {relation.subject, relation.object}
                if active_replacement not in endpoints:
                    raise ValueError(
                        "A proposed contradiction must include the active replacement revision."
                    )
                source = (
                    relation.object if relation.subject == active_replacement else relation.subject
                )
            else:
                if relation.subject != active_replacement:
                    raise ValueError(
                        "A proposed directed relation must start at the active replacement revision."
                    )
                source = relation.object
            source_key = (source.entry_id, source.revision)
            if source_key not in source_keys:
                raise ValueError("Every proposed relation must target a reviewed source revision.")
            if (
                relation.kind is KnowledgeRelationKind.SUPERSEDES
                and source.revision >= MAX_KNOWLEDGE_REVISION
            ):
                raise ValueError("A superseded source revision must be able to advance safely.")
            if source_key in referenced_sources:
                raise ValueError(
                    "Every reviewed source must have exactly one maintenance disposition."
                )
            referenced_sources.add(source_key)
        if referenced_sources != source_keys:
            raise ValueError("Every reviewed source must participate in a proposed relation.")
        if (
            len(
                canonical_durable_json_bytes(
                    self.model_dump(mode="json"),
                    "knowledge maintenance proposal",
                )
            )
            > MAX_KNOWLEDGE_MAINTENANCE_BYTES
        ):
            raise ValueError("Knowledge maintenance proposal exceeds its canonical byte budget.")
        return self

    @property
    def fingerprint(self) -> str:
        return sha256(
            canonical_durable_json_bytes(
                {
                    "contract": "cayu-knowledge-maintenance-proposal-v1",
                    "proposal": self.model_dump(mode="json"),
                },
                "knowledge maintenance proposal fingerprint",
            )
        ).hexdigest()


class KnowledgeMaintenanceDecision(BaseModel):
    """Immutable external review over one exact maintenance proposal."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    operation_id: str
    proposal_id: str
    proposal_fingerprint: str
    kind: KnowledgeMaintenanceDecisionKind
    reviewer_type: KnowledgeActorType
    reviewer: str
    reason: str
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("`schema_version` must be the integer 1.")
        return value

    @field_validator("operation_id", "proposal_id", "reviewer")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _knowledge_maintenance_identity(value, info.field_name)

    @field_validator("proposal_fingerprint")
    @classmethod
    def validate_proposal_fingerprint(cls, value: str) -> str:
        if type(value) is not str or _SHA256_HEX_RE.fullmatch(value) is None:
            raise ValueError("`proposal_fingerprint` must be lowercase SHA-256 hex.")
        return value

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        value = require_clean_nonblank(value, "reason")
        if len(value.encode("utf-8")) > MAX_KNOWLEDGE_MAINTENANCE_TEXT_BYTES:
            raise ValueError(
                f"`reason` must be at most {MAX_KNOWLEDGE_MAINTENANCE_TEXT_BYTES} UTF-8 bytes."
            )
        return value

    @field_validator("decided_at")
    @classmethod
    def validate_decided_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("`decided_at` must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        copied = copy_durable_metadata(value, "metadata")
        if len(canonical_durable_json_bytes(copied, "knowledge decision metadata")) > (
            MAX_KNOWLEDGE_MAINTENANCE_METADATA_BYTES
        ):
            raise ValueError("`metadata` exceeds the knowledge decision metadata budget.")
        return copied

    @model_validator(mode="after")
    def validate_reviewer(self) -> KnowledgeMaintenanceDecision:
        if self.reviewer_type is KnowledgeActorType.MODEL:
            raise ValueError("Model output cannot authorize a knowledge maintenance decision.")
        if (
            len(
                canonical_durable_json_bytes(
                    self.model_dump(mode="json"),
                    "knowledge maintenance decision",
                )
            )
            > MAX_KNOWLEDGE_MAINTENANCE_BYTES
        ):
            raise ValueError("Knowledge maintenance decision exceeds its canonical byte budget.")
        return self

    @property
    def fingerprint(self) -> str:
        return sha256(
            canonical_durable_json_bytes(
                {
                    "contract": "cayu-knowledge-maintenance-decision-v1",
                    "decision": self.model_dump(mode="json"),
                },
                "knowledge maintenance decision fingerprint",
            )
        ).hexdigest()


class KnowledgeMaintenanceDecisionReceipt(BaseModel):
    """Immutable evidence for one atomic reviewed maintenance outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    operation_id: str
    proposal_id: str
    proposal_fingerprint: str
    request_sha256: str
    outcome: KnowledgeMaintenanceOutcome
    replacement: KnowledgeRevisionRef | None = None
    archived_revisions: list[KnowledgeRevisionRef] = Field(default_factory=list)
    relation_ids: list[str] = Field(default_factory=list)
    committed_at: datetime
    replayed: bool = False

    @field_validator("operation_id", "proposal_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _knowledge_maintenance_identity(value, info.field_name)

    @field_validator("proposal_fingerprint", "request_sha256")
    @classmethod
    def validate_sha256(cls, value: str, info) -> str:
        if type(value) is not str or _SHA256_HEX_RE.fullmatch(value) is None:
            raise ValueError(f"`{info.field_name}` must be lowercase SHA-256 hex.")
        return value

    @field_validator("replacement", mode="before")
    @classmethod
    def copy_replacement(cls, value):
        if value is None:
            return None
        if isinstance(value, KnowledgeRevisionRef):
            return copy_knowledge_revision_ref(value)
        return value

    @field_validator("archived_revisions", mode="before")
    @classmethod
    def copy_archived_revisions(cls, value) -> list[KnowledgeRevisionRef]:
        if type(value) is not list:
            raise ValueError("`archived_revisions` must be a list.")
        if len(value) > MAX_KNOWLEDGE_MAINTENANCE_SOURCES:
            raise ValueError("`archived_revisions` cannot exceed the maintenance source bound.")
        copied = [
            copy_knowledge_revision_ref(item)
            if isinstance(item, KnowledgeRevisionRef)
            else KnowledgeRevisionRef.model_validate(item)
            for item in value
        ]
        copied.sort(key=lambda item: (item.entry_id, item.revision))
        if len({(item.entry_id, item.revision) for item in copied}) != len(copied):
            raise ValueError("`archived_revisions` cannot repeat a revision.")
        if len({item.entry_id for item in copied}) != len(copied):
            raise ValueError("`archived_revisions` cannot repeat a logical entry.")
        return copied

    @field_validator("relation_ids", mode="before")
    @classmethod
    def copy_relation_ids(cls, value) -> list[str]:
        if type(value) is not list:
            raise ValueError("`relation_ids` must be a list.")
        if len(value) > MAX_KNOWLEDGE_MAINTENANCE_SOURCES:
            raise ValueError("`relation_ids` cannot exceed the maintenance source bound.")
        copied = sorted(_knowledge_relation_identity(item, "relation_ids") for item in value)
        if len(set(copied)) != len(copied):
            raise ValueError("`relation_ids` cannot repeat an identity.")
        return copied

    @field_validator("committed_at")
    @classmethod
    def validate_committed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("`committed_at` must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("replayed", mode="before")
    @classmethod
    def validate_replayed(cls, value) -> bool:
        if type(value) is not bool:
            raise ValueError("`replayed` must be a boolean.")
        return value

    @model_validator(mode="after")
    def validate_outcome(self) -> KnowledgeMaintenanceDecisionReceipt:
        if self.outcome is KnowledgeMaintenanceOutcome.REJECTED:
            if self.replacement is not None or self.archived_revisions or self.relation_ids:
                raise ValueError("A rejected maintenance receipt cannot claim lifecycle changes.")
        elif self.replacement is None or not self.relation_ids:
            raise ValueError("An applied maintenance receipt requires replacement and relations.")
        elif self.replacement.entry_id in {
            reference.entry_id for reference in self.archived_revisions
        }:
            raise ValueError("The active replacement cannot also be an archived revision.")
        elif len(self.archived_revisions) > len(self.relation_ids):
            raise ValueError("Archived revisions cannot outnumber approved relations.")
        if (
            len(
                canonical_durable_json_bytes(
                    self.model_dump(mode="json"),
                    "knowledge maintenance receipt",
                )
            )
            > MAX_KNOWLEDGE_MAINTENANCE_BYTES
        ):
            raise ValueError("Knowledge maintenance receipt exceeds its canonical byte budget.")
        return self


def copy_knowledge_maintenance_proposal(
    proposal: KnowledgeMaintenanceProposal,
) -> KnowledgeMaintenanceProposal:
    if type(proposal) is not KnowledgeMaintenanceProposal:
        raise TypeError("KnowledgeMaintenanceProposal instances must not be subclasses.")
    return KnowledgeMaintenanceProposal.model_validate(
        proposal.model_dump(mode="python", warnings=False)
    )


def copy_knowledge_maintenance_decision(
    decision: KnowledgeMaintenanceDecision,
) -> KnowledgeMaintenanceDecision:
    if type(decision) is not KnowledgeMaintenanceDecision:
        raise TypeError("KnowledgeMaintenanceDecision instances must not be subclasses.")
    return KnowledgeMaintenanceDecision.model_validate(
        decision.model_dump(mode="python", warnings=False)
    )


def copy_knowledge_maintenance_decision_receipt(
    receipt: KnowledgeMaintenanceDecisionReceipt,
    *,
    replayed: bool | None = None,
) -> KnowledgeMaintenanceDecisionReceipt:
    if type(receipt) is not KnowledgeMaintenanceDecisionReceipt:
        raise TypeError("KnowledgeMaintenanceDecisionReceipt instances must not be subclasses.")
    return KnowledgeMaintenanceDecisionReceipt(
        **receipt.model_dump(
            mode="python",
            exclude={"replayed"},
            warnings=False,
        ),
        replayed=receipt.replayed if replayed is None else replayed,
    )


def prepare_knowledge_maintenance_decision(
    proposal: KnowledgeMaintenanceProposal,
    decision: KnowledgeMaintenanceDecision,
) -> tuple[KnowledgeMaintenanceProposal, KnowledgeMaintenanceDecision, str]:
    """Copy and bind one exact proposal and external review decision."""

    copied_proposal = copy_knowledge_maintenance_proposal(proposal)
    copied_decision = copy_knowledge_maintenance_decision(decision)
    if copied_decision.proposal_id != copied_proposal.id:
        raise ValueError("The maintenance decision references another proposal identity.")
    if copied_decision.proposal_fingerprint != copied_proposal.fingerprint:
        raise ValueError("The maintenance decision references another proposal fingerprint.")
    if copied_decision.decided_at < copied_proposal.created_at:
        raise ValueError("The maintenance decision cannot predate its proposal.")
    request_sha256 = sha256(
        canonical_durable_json_bytes(
            {
                "contract": "cayu-knowledge-maintenance-application-v1",
                "proposal": copied_proposal.model_dump(mode="json"),
                "decision": copied_decision.model_dump(mode="json"),
            },
            "knowledge maintenance application",
        )
    ).hexdigest()
    return copied_proposal, copied_decision, request_sha256


def _validate_knowledge_maintenance_replay(
    stored_proposal: KnowledgeMaintenanceProposal,
    stored_decision: KnowledgeMaintenanceDecision,
    receipt: KnowledgeMaintenanceDecisionReceipt,
    *,
    proposal: KnowledgeMaintenanceProposal,
    decision: KnowledgeMaintenanceDecision,
    request_sha256: str,
) -> None:
    _validate_knowledge_maintenance_record(stored_proposal, stored_decision, receipt)
    if (
        stored_proposal != proposal
        or stored_decision != decision
        or receipt.operation_id != decision.operation_id
        or receipt.proposal_id != proposal.id
        or receipt.proposal_fingerprint != proposal.fingerprint
        or receipt.request_sha256 != request_sha256
    ):
        raise KnowledgeMaintenanceConflict("operation_reuse")


def _validate_knowledge_maintenance_record(
    proposal: KnowledgeMaintenanceProposal,
    decision: KnowledgeMaintenanceDecision,
    receipt: KnowledgeMaintenanceDecisionReceipt,
) -> None:
    try:
        _, _, request_sha256 = prepare_knowledge_maintenance_decision(proposal, decision)
        expected_outcome = (
            KnowledgeMaintenanceOutcome.APPLIED
            if decision.kind is KnowledgeMaintenanceDecisionKind.APPROVE
            else KnowledgeMaintenanceOutcome.REJECTED
        )
        expected_replacement = (
            KnowledgeRevisionRef(
                entry_id=proposal.replacement.entry_id,
                revision=proposal.replacement.revision + 1,
            )
            if expected_outcome is KnowledgeMaintenanceOutcome.APPLIED
            else None
        )
        superseded = {
            (relation.object.entry_id, relation.object.revision)
            for relation in proposal.relations
            if relation.kind is KnowledgeRelationKind.SUPERSEDES
        }
        expected_archived = (
            sorted(
                (
                    KnowledgeRevisionRef(
                        entry_id=source.entry_id,
                        revision=source.revision + 1,
                    )
                    for source in proposal.sources
                    if (source.entry_id, source.revision) in superseded
                ),
                key=lambda item: (item.entry_id, item.revision),
            )
            if expected_outcome is KnowledgeMaintenanceOutcome.APPLIED
            else []
        )
        expected_relation_ids = (
            [relation.id for relation in proposal.relations]
            if expected_outcome is KnowledgeMaintenanceOutcome.APPLIED
            else []
        )
        if (
            receipt.operation_id != decision.operation_id
            or receipt.proposal_id != proposal.id
            or receipt.proposal_fingerprint != proposal.fingerprint
            or receipt.request_sha256 != request_sha256
            or receipt.outcome is not expected_outcome
            or receipt.replacement != expected_replacement
            or receipt.archived_revisions != expected_archived
            or receipt.relation_ids != expected_relation_ids
            or receipt.committed_at < proposal.created_at
            or receipt.committed_at < decision.decided_at
            or receipt.replayed
        ):
            raise ValueError("Maintenance record conflicts with its reviewed authority.")
    except KnowledgeMaintenanceConflict:
        raise
    except Exception:
        raise KnowledgeMaintenanceConflict("malformed_receipt") from None


def _knowledge_maintenance_identity(value: str, field_name: str) -> str:
    return _bounded_knowledge_identity(value, field_name, max_bytes=256)
