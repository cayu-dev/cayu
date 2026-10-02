"""Backend-independent knowledge relation, lineage, and publication contracts."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import canonical_durable_json_bytes, copy_durable_metadata
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.knowledge.records import (
    DEFAULT_KNOWLEDGE_LIMIT,
    DEFAULT_KNOWLEDGE_MAX_BYTES,
    KnowledgeActorType,
    KnowledgeRevisionRef,
    KnowledgeStatus,
    _bounded_knowledge_identity,
    _validate_positive_int,
    copy_knowledge_revision_ref,
)

_SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}\Z")


MAX_KNOWLEDGE_RELATION_BATCH = 100


MAX_KNOWLEDGE_RELATION_BYTES = 8_192


MAX_KNOWLEDGE_RELATION_CURSOR_BYTES = 2_048


MAX_KNOWLEDGE_RELATION_LIMIT = 1_000


class KnowledgeRelationKind(StrEnum):
    """Closed semantic vocabulary for exact cross-entry knowledge lineage."""

    SUPERSEDES = "supersedes"
    DERIVED_FROM = "derived_from"
    CONTRADICTS = "contradicts"


class KnowledgeRelationDirection(StrEnum):
    OUTGOING = "outgoing"
    INCOMING = "incoming"
    BOTH = "both"


class KnowledgeLineageRole(StrEnum):
    """Meaning of one relation from the inspected revision's perspective."""

    SUPERSEDES = "supersedes"
    SUPERSEDED_BY = "superseded_by"
    DERIVED_FROM = "derived_from"
    DERIVATION_SOURCE_FOR = "derivation_source_for"
    CONTRADICTS = "contradicts"


class KnowledgeLineageCurrentness(StrEnum):
    """Whether both exact relation endpoints are still their logical current revisions."""

    CURRENT = "current"
    STALE = "stale"


class KnowledgeRelationConflict(RuntimeError):
    """A relation publication conflicts with immutable durable state."""

    def __init__(self, reason: str) -> None:
        self.reason = require_clean_nonblank(reason, "reason")
        super().__init__("Knowledge relation publication conflicts with durable state.")


class KnowledgeRelation(BaseModel):
    """Immutable semantic lineage between two exact entry revisions.

    Direction is meaningful: a replacement ``supersedes`` its predecessor and a
    derived revision ``derived_from`` its source. ``contradicts`` is symmetric;
    stores canonicalize its endpoint order before publication.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str
    subject: KnowledgeRevisionRef
    object: KnowledgeRevisionRef
    kind: KnowledgeRelationKind
    created_by_type: KnowledgeActorType = KnowledgeActorType.APP
    created_by: str = "app"
    policy_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _knowledge_relation_identity(value, "id")

    @field_validator("subject", "object", mode="before")
    @classmethod
    def copy_reference(cls, value):
        if isinstance(value, KnowledgeRevisionRef):
            return copy_knowledge_revision_ref(value)
        return value

    @field_validator("created_by")
    @classmethod
    def validate_created_by(cls, value: str) -> str:
        return _knowledge_relation_identity(value, "created_by")

    @field_validator("policy_id")
    @classmethod
    def validate_policy_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _knowledge_relation_identity(value, "policy_id")

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("`created_at` must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        copied = copy_durable_metadata(value, "metadata")
        if len(canonical_durable_json_bytes(copied, "knowledge relation metadata")) > (
            MAX_KNOWLEDGE_RELATION_BYTES // 2
        ):
            raise ValueError("`metadata` exceeds the bounded knowledge relation metadata budget.")
        return copied

    @model_validator(mode="after")
    def validate_relation(self) -> KnowledgeRelation:
        if self.subject.entry_id == self.object.entry_id:
            raise ValueError("Knowledge relations must connect different logical entries.")
        if (
            len(
                canonical_durable_json_bytes(
                    self.model_dump(mode="json"),
                    "knowledge relation",
                )
            )
            > MAX_KNOWLEDGE_RELATION_BYTES
        ):
            raise ValueError(
                f"Knowledge relations must be at most {MAX_KNOWLEDGE_RELATION_BYTES} "
                "canonical UTF-8 bytes."
            )
        return self


class KnowledgeRelationPublicationReceipt(BaseModel):
    """Immutable replay evidence for one atomic relation batch publication."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    operation_id: str
    relation_ids: list[str]
    request_sha256: str
    committed_at: datetime
    replayed: bool = False

    @field_validator("operation_id")
    @classmethod
    def validate_operation_id(cls, value: str) -> str:
        return _knowledge_relation_identity(value, "operation_id")

    @field_validator("relation_ids", mode="before")
    @classmethod
    def validate_relation_ids(cls, value) -> list[str]:
        if type(value) is not list:
            raise ValueError("`relation_ids` must be a list.")
        copied = [_knowledge_relation_identity(item, "relation_ids") for item in value]
        if not copied or len(copied) > MAX_KNOWLEDGE_RELATION_BATCH:
            raise ValueError(
                "`relation_ids` must contain between 1 and "
                f"{MAX_KNOWLEDGE_RELATION_BATCH} identities."
            )
        if copied != sorted(set(copied)):
            raise ValueError("`relation_ids` must be unique and bytewise sorted.")
        return copied

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        if type(value) is not str or _SHA256_HEX_RE.fullmatch(value) is None:
            raise ValueError("`request_sha256` must be lowercase SHA-256 hex.")
        return value

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


class KnowledgeRelationQuery(BaseModel):
    """Bounded exact-revision relation lookup."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    reference: KnowledgeRevisionRef
    direction: KnowledgeRelationDirection = KnowledgeRelationDirection.BOTH
    kinds: list[KnowledgeRelationKind] = Field(default_factory=list)
    limit: int = DEFAULT_KNOWLEDGE_LIMIT
    max_bytes: int = DEFAULT_KNOWLEDGE_MAX_BYTES
    cursor: str | None = None

    @field_validator("reference", mode="before")
    @classmethod
    def copy_reference(cls, value):
        if isinstance(value, KnowledgeRevisionRef):
            return copy_knowledge_revision_ref(value)
        return value

    @field_validator("kinds", mode="before")
    @classmethod
    def copy_kinds(cls, value) -> list[KnowledgeRelationKind]:
        if type(value) is not list:
            raise ValueError("`kinds` must be a list.")
        copied = [
            item if isinstance(item, KnowledgeRelationKind) else KnowledgeRelationKind(item)
            for item in value
        ]
        return sorted(set(copied), key=lambda item: item.value)

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, value: int) -> int:
        _validate_positive_int(value, "limit")
        if value > MAX_KNOWLEDGE_RELATION_LIMIT:
            raise ValueError(f"`limit` must be at most {MAX_KNOWLEDGE_RELATION_LIMIT}.")
        return value

    @field_validator("max_bytes")
    @classmethod
    def validate_max_bytes(cls, value: int) -> int:
        _validate_positive_int(value, "max_bytes")
        if value < MAX_KNOWLEDGE_RELATION_BYTES:
            raise ValueError(
                f"`max_bytes` must be at least {MAX_KNOWLEDGE_RELATION_BYTES} so "
                "one valid relation can always advance the cursor."
            )
        return value

    @field_validator("cursor")
    @classmethod
    def validate_cursor(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_knowledge_relation_cursor(value, "cursor")


class KnowledgeRelationResult(BaseModel):
    """One honest bounded page of accessible revision-bound relations."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    query: KnowledgeRelationQuery
    relations: list[KnowledgeRelation] = Field(default_factory=list)
    truncated: bool = False
    next_cursor: str | None = None

    @field_validator("query", mode="before")
    @classmethod
    def copy_query(cls, value):
        if isinstance(value, KnowledgeRelationQuery):
            return copy_knowledge_relation_query(value)
        return value

    @field_validator("relations", mode="before")
    @classmethod
    def copy_relations(cls, value):
        if type(value) is not list:
            raise ValueError("`relations` must be a list.")
        return [
            copy_knowledge_relation(item) if isinstance(item, KnowledgeRelation) else item
            for item in value
        ]

    @field_validator("truncated", mode="before")
    @classmethod
    def validate_truncated(cls, value) -> bool:
        if type(value) is not bool:
            raise ValueError("`truncated` must be a boolean.")
        return value

    @field_validator("next_cursor")
    @classmethod
    def validate_next_cursor(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_knowledge_relation_cursor(value, "next_cursor")

    @model_validator(mode="after")
    def validate_page(self) -> KnowledgeRelationResult:
        if len(self.relations) > self.query.limit:
            raise ValueError("`relations` cannot contain more records than `query.limit`.")
        if len({item.id for item in self.relations}) != len(self.relations):
            raise ValueError("Knowledge relation identities must be unique within a page.")
        keys = [(item.created_at, item.id) for item in self.relations]
        if keys != sorted(set(keys)):
            raise ValueError("Knowledge relations must have unique increasing page order.")
        if any(
            (self.query.kinds and item.kind not in self.query.kinds)
            or not _knowledge_relation_matches_query(item, self.query)
            for item in self.relations
        ):
            raise ValueError("Knowledge relations must match the result query.")
        serialized_bytes = sum(
            len(
                canonical_durable_json_bytes(
                    item.model_dump(mode="json"),
                    "knowledge relation",
                )
            )
            for item in self.relations
        )
        if serialized_bytes > self.query.max_bytes:
            raise ValueError("Knowledge relation page exceeds `query.max_bytes`.")
        if self.truncated != (self.next_cursor is not None):
            raise ValueError("A truncated relation page requires exactly one next cursor.")
        if self.next_cursor is not None and not self.relations:
            raise ValueError("An empty relation page cannot have a next cursor.")
        return self


class KnowledgeLineageLink(BaseModel):
    """Privacy-safe relation projection around one exact inspected revision.

    The projection deliberately excludes entry text, relation metadata, actors, and
    policy payloads. It is therefore suitable for recall explanations without
    granting ordinary read access to an archived predecessor's content.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    relation_id: str
    kind: KnowledgeRelationKind
    role: KnowledgeLineageRole
    counterpart: KnowledgeRevisionRef
    counterpart_current: KnowledgeRevisionRef
    counterpart_status: KnowledgeStatus
    currentness: KnowledgeLineageCurrentness
    unresolved_contradiction: bool = False
    created_at: datetime

    @field_validator("relation_id")
    @classmethod
    def validate_relation_id(cls, value: str) -> str:
        return _knowledge_relation_identity(value, "relation_id")

    @field_validator("counterpart", "counterpart_current", mode="before")
    @classmethod
    def copy_reference(cls, value):
        if isinstance(value, KnowledgeRevisionRef):
            return copy_knowledge_revision_ref(value)
        return value

    @field_validator("unresolved_contradiction", mode="before")
    @classmethod
    def validate_unresolved_contradiction(cls, value) -> bool:
        if type(value) is not bool:
            raise ValueError("`unresolved_contradiction` must be a boolean.")
        return value

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("`created_at` must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_projection(self) -> KnowledgeLineageLink:
        expected_kinds = {
            KnowledgeLineageRole.SUPERSEDES: KnowledgeRelationKind.SUPERSEDES,
            KnowledgeLineageRole.SUPERSEDED_BY: KnowledgeRelationKind.SUPERSEDES,
            KnowledgeLineageRole.DERIVED_FROM: KnowledgeRelationKind.DERIVED_FROM,
            KnowledgeLineageRole.DERIVATION_SOURCE_FOR: KnowledgeRelationKind.DERIVED_FROM,
            KnowledgeLineageRole.CONTRADICTS: KnowledgeRelationKind.CONTRADICTS,
        }
        if expected_kinds[self.role] is not self.kind:
            raise ValueError("Knowledge lineage role conflicts with relation kind.")
        if self.counterpart.entry_id != self.counterpart_current.entry_id:
            raise ValueError("Lineage counterpart exact/current identities must match.")
        if self.counterpart.revision > self.counterpart_current.revision:
            raise ValueError("A lineage counterpart cannot postdate its current revision.")
        if (
            self.currentness is KnowledgeLineageCurrentness.CURRENT
            and self.counterpart != self.counterpart_current
        ):
            raise ValueError("A current lineage link requires a current counterpart.")
        if self.unresolved_contradiction and (
            self.kind is not KnowledgeRelationKind.CONTRADICTS
            or self.currentness is not KnowledgeLineageCurrentness.CURRENT
            or self.counterpart_status is not KnowledgeStatus.ACTIVE
        ):
            raise ValueError("Only a current active contradiction can be unresolved.")
        return self


class KnowledgeLineageQuery(BaseModel):
    """Bounded filters for safe exact-revision lineage inspection."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        validate_default=True,
    )

    reference: KnowledgeRevisionRef
    direction: KnowledgeRelationDirection = KnowledgeRelationDirection.BOTH
    kinds: list[KnowledgeRelationKind] = Field(default_factory=list)
    currentnesses: list[KnowledgeLineageCurrentness] = Field(
        default_factory=lambda: list(KnowledgeLineageCurrentness)
    )
    counterpart_statuses: list[KnowledgeStatus] = Field(
        default_factory=lambda: list(KnowledgeStatus)
    )
    unresolved_only: bool = False
    limit: int = DEFAULT_KNOWLEDGE_LIMIT
    max_bytes: int = DEFAULT_KNOWLEDGE_MAX_BYTES
    cursor: str | None = None

    @field_validator("reference", mode="before")
    @classmethod
    def copy_reference(cls, value):
        if isinstance(value, KnowledgeRevisionRef):
            return copy_knowledge_revision_ref(value)
        return value

    @field_validator("kinds", mode="before")
    @classmethod
    def copy_kinds(cls, value) -> list[KnowledgeRelationKind]:
        return _copy_enum_filter(value, KnowledgeRelationKind, "kinds")

    @field_validator("currentnesses", mode="before")
    @classmethod
    def copy_currentnesses(cls, value) -> list[KnowledgeLineageCurrentness]:
        return _copy_enum_filter(value, KnowledgeLineageCurrentness, "currentnesses")

    @field_validator("counterpart_statuses", mode="before")
    @classmethod
    def copy_counterpart_statuses(cls, value) -> list[KnowledgeStatus]:
        return _copy_enum_filter(value, KnowledgeStatus, "counterpart_statuses")

    @field_validator("currentnesses", "counterpart_statuses")
    @classmethod
    def validate_nonempty_filters(cls, value: list[Any], info) -> list[Any]:
        if not value:
            raise ValueError(f"`{info.field_name}` cannot be empty.")
        return value

    @field_validator("unresolved_only", mode="before")
    @classmethod
    def validate_unresolved_only(cls, value) -> bool:
        if type(value) is not bool:
            raise ValueError("`unresolved_only` must be a boolean.")
        return value

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, value: int) -> int:
        _validate_positive_int(value, "limit")
        if value > MAX_KNOWLEDGE_RELATION_LIMIT:
            raise ValueError(f"`limit` must be at most {MAX_KNOWLEDGE_RELATION_LIMIT}.")
        return value

    @field_validator("max_bytes")
    @classmethod
    def validate_max_bytes(cls, value: int) -> int:
        _validate_positive_int(value, "max_bytes")
        if value < MAX_KNOWLEDGE_RELATION_BYTES:
            raise ValueError(
                f"`max_bytes` must be at least {MAX_KNOWLEDGE_RELATION_BYTES} so "
                "one valid lineage link can always advance the cursor."
            )
        return value

    @field_validator("cursor")
    @classmethod
    def validate_cursor(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_knowledge_relation_cursor(value, "cursor")

    @model_validator(mode="after")
    def validate_unresolved_filter(self) -> KnowledgeLineageQuery:
        if (
            self.unresolved_only
            and self.kinds
            and (KnowledgeRelationKind.CONTRADICTS not in self.kinds)
        ):
            raise ValueError("`unresolved_only` requires the contradiction kind.")
        if self.unresolved_only and (
            KnowledgeLineageCurrentness.CURRENT not in self.currentnesses
            or KnowledgeStatus.ACTIVE not in self.counterpart_statuses
        ):
            raise ValueError("`unresolved_only` requires current, active counterparts.")
        return self


class KnowledgeLineageResult(BaseModel):
    """One bounded safe lineage page for an authorized exact revision."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    query: KnowledgeLineageQuery
    reference_current: KnowledgeRevisionRef
    reference_status: KnowledgeStatus
    links: list[KnowledgeLineageLink] = Field(default_factory=list)
    truncated: bool = False
    next_cursor: str | None = None

    @field_validator("query", mode="before")
    @classmethod
    def copy_query(cls, value):
        if isinstance(value, KnowledgeLineageQuery):
            return copy_knowledge_lineage_query(value)
        return value

    @field_validator("reference_current", mode="before")
    @classmethod
    def copy_reference(cls, value):
        if isinstance(value, KnowledgeRevisionRef):
            return copy_knowledge_revision_ref(value)
        return value

    @field_validator("links", mode="before")
    @classmethod
    def copy_links(cls, value):
        if type(value) is not list:
            raise ValueError("`links` must be a list.")
        return [
            copy_knowledge_lineage_link(item) if isinstance(item, KnowledgeLineageLink) else item
            for item in value
        ]

    @field_validator("truncated", mode="before")
    @classmethod
    def validate_truncated(cls, value) -> bool:
        if type(value) is not bool:
            raise ValueError("`truncated` must be a boolean.")
        return value

    @field_validator("next_cursor")
    @classmethod
    def validate_next_cursor(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_knowledge_relation_cursor(value, "next_cursor")

    @model_validator(mode="after")
    def validate_page(self) -> KnowledgeLineageResult:
        if self.reference_current.entry_id != self.query.reference.entry_id:
            raise ValueError("Lineage exact/current reference identities must match.")
        if self.reference_current.revision < self.query.reference.revision:
            raise ValueError("Lineage current reference cannot predate the inspected revision.")
        if len(self.links) > self.query.limit:
            raise ValueError("`links` cannot contain more records than `query.limit`.")
        keys = [(item.created_at, item.relation_id) for item in self.links]
        if keys != sorted(set(keys)):
            raise ValueError("Knowledge lineage links must have unique increasing page order.")
        if any(not _knowledge_lineage_link_matches_query(item, self.query) for item in self.links):
            raise ValueError("Knowledge lineage links must match the result query.")
        if any(
            item.currentness
            is not (
                KnowledgeLineageCurrentness.CURRENT
                if (
                    self.query.reference == self.reference_current
                    and item.counterpart == item.counterpart_current
                )
                else KnowledgeLineageCurrentness.STALE
            )
            for item in self.links
        ):
            raise ValueError("A lineage link misstates its exact-revision currentness.")
        if any(
            item.unresolved_contradiction
            != (
                item.kind is KnowledgeRelationKind.CONTRADICTS
                and item.currentness is KnowledgeLineageCurrentness.CURRENT
                and self.reference_status is KnowledgeStatus.ACTIVE
                and item.counterpart_status is KnowledgeStatus.ACTIVE
            )
            for item in self.links
        ):
            raise ValueError("A lineage contradiction misstates its unresolved lifecycle.")
        serialized_bytes = sum(_knowledge_lineage_link_bytes(item) for item in self.links)
        if serialized_bytes > self.query.max_bytes:
            raise ValueError("Knowledge lineage page exceeds `query.max_bytes`.")
        if self.truncated != (self.next_cursor is not None):
            raise ValueError("A truncated lineage page requires exactly one next cursor.")
        if self.next_cursor is not None and not self.links:
            raise ValueError("An empty lineage page cannot have a next cursor.")
        return self


def copy_knowledge_relation(relation: KnowledgeRelation) -> KnowledgeRelation:
    if type(relation) is not KnowledgeRelation:
        raise TypeError("KnowledgeRelation instances must not be subclasses.")
    subject = copy_knowledge_revision_ref(relation.subject)
    object_ = copy_knowledge_revision_ref(relation.object)
    if relation.kind is KnowledgeRelationKind.CONTRADICTS and (
        object_.entry_id,
        object_.revision,
    ) < (subject.entry_id, subject.revision):
        subject, object_ = object_, subject
    return KnowledgeRelation(
        id=relation.id,
        subject=subject,
        object=object_,
        kind=relation.kind,
        created_by_type=relation.created_by_type,
        created_by=relation.created_by,
        policy_id=relation.policy_id,
        created_at=relation.created_at,
        metadata=copy_durable_metadata(relation.metadata, "metadata"),
    )


def copy_knowledge_relation_query(query: KnowledgeRelationQuery) -> KnowledgeRelationQuery:
    if type(query) is not KnowledgeRelationQuery:
        raise TypeError("KnowledgeRelationQuery instances must not be subclasses.")
    return KnowledgeRelationQuery(
        reference=copy_knowledge_revision_ref(query.reference),
        direction=query.direction,
        kinds=list(query.kinds),
        limit=query.limit,
        max_bytes=query.max_bytes,
        cursor=query.cursor,
    )


def copy_knowledge_lineage_link(link: KnowledgeLineageLink) -> KnowledgeLineageLink:
    if type(link) is not KnowledgeLineageLink:
        raise TypeError("KnowledgeLineageLink instances must not be subclasses.")
    return KnowledgeLineageLink(
        relation_id=link.relation_id,
        kind=link.kind,
        role=link.role,
        counterpart=copy_knowledge_revision_ref(link.counterpart),
        counterpart_current=copy_knowledge_revision_ref(link.counterpart_current),
        counterpart_status=link.counterpart_status,
        currentness=link.currentness,
        unresolved_contradiction=link.unresolved_contradiction,
        created_at=link.created_at,
    )


def copy_knowledge_lineage_query(query: KnowledgeLineageQuery) -> KnowledgeLineageQuery:
    if type(query) is not KnowledgeLineageQuery:
        raise TypeError("KnowledgeLineageQuery instances must not be subclasses.")
    return KnowledgeLineageQuery(
        reference=copy_knowledge_revision_ref(query.reference),
        direction=query.direction,
        kinds=list(query.kinds),
        currentnesses=list(query.currentnesses),
        counterpart_statuses=list(query.counterpart_statuses),
        unresolved_only=query.unresolved_only,
        limit=query.limit,
        max_bytes=query.max_bytes,
        cursor=query.cursor,
    )


def copy_knowledge_relation_publication_receipt(
    receipt: KnowledgeRelationPublicationReceipt,
    *,
    replayed: bool | None = None,
) -> KnowledgeRelationPublicationReceipt:
    if type(receipt) is not KnowledgeRelationPublicationReceipt:
        raise TypeError("KnowledgeRelationPublicationReceipt instances must not be subclasses.")
    return KnowledgeRelationPublicationReceipt(
        operation_id=receipt.operation_id,
        relation_ids=list(receipt.relation_ids),
        request_sha256=receipt.request_sha256,
        committed_at=receipt.committed_at,
        replayed=receipt.replayed if replayed is None else replayed,
    )


def prepare_knowledge_relations(
    relations: list[KnowledgeRelation],
    *,
    operation_id: str,
) -> tuple[str, list[KnowledgeRelation], str]:
    """Copy, canonicalize, bound, and fingerprint one relation publication."""

    operation_id = _knowledge_relation_identity(operation_id, "operation_id")
    if type(relations) is not list:
        raise TypeError("`relations` must be a list.")
    if not relations or len(relations) > MAX_KNOWLEDGE_RELATION_BATCH:
        raise ValueError(
            f"`relations` must contain between 1 and {MAX_KNOWLEDGE_RELATION_BATCH} records."
        )
    copied = sorted(
        (copy_knowledge_relation(relation) for relation in relations),
        key=lambda relation: relation.id,
    )
    ids = [relation.id for relation in copied]
    if len(ids) != len(set(ids)):
        raise ValueError("`relations` cannot contain duplicate identities.")
    semantic_keys = [_knowledge_relation_semantic_key(relation) for relation in copied]
    if len(semantic_keys) != len(set(semantic_keys)):
        raise ValueError("`relations` cannot repeat one semantic relation.")
    request_sha256 = sha256(
        canonical_durable_json_bytes(
            {
                "contract": "cayu-knowledge-relation-publication-v1",
                "relations": [relation.model_dump(mode="json") for relation in copied],
            },
            "knowledge relation publication",
        )
    ).hexdigest()
    return operation_id, copied, request_sha256


def _validate_knowledge_relation_publication_replay(
    receipt: KnowledgeRelationPublicationReceipt,
    *,
    relations: list[KnowledgeRelation],
    request_sha256: str,
) -> None:
    receipt = copy_knowledge_relation_publication_receipt(receipt)
    if (
        receipt.relation_ids != [relation.id for relation in relations]
        or receipt.request_sha256 != request_sha256
    ):
        raise KnowledgeRelationConflict("operation_reuse")


def _knowledge_relation_semantic_key(
    relation: KnowledgeRelation,
) -> tuple[str, str, int, str, int]:
    relation = copy_knowledge_relation(relation)
    return (
        relation.kind.value,
        relation.subject.entry_id,
        relation.subject.revision,
        relation.object.entry_id,
        relation.object.revision,
    )


def _knowledge_relation_matches_query(
    relation: KnowledgeRelation,
    query: KnowledgeRelationQuery,
) -> bool:
    reference = query.reference
    subject_matches = relation.subject == reference
    object_matches = relation.object == reference
    if relation.kind is KnowledgeRelationKind.CONTRADICTS:
        return subject_matches or object_matches
    if query.direction is KnowledgeRelationDirection.OUTGOING:
        return subject_matches
    if query.direction is KnowledgeRelationDirection.INCOMING:
        return object_matches
    return subject_matches or object_matches


def _knowledge_lineage_link_matches_query(
    link: KnowledgeLineageLink,
    query: KnowledgeLineageQuery,
) -> bool:
    direction_matches = (
        query.direction is KnowledgeRelationDirection.BOTH
        or link.role is KnowledgeLineageRole.CONTRADICTS
        or (
            query.direction is KnowledgeRelationDirection.OUTGOING
            and link.role in {KnowledgeLineageRole.SUPERSEDES, KnowledgeLineageRole.DERIVED_FROM}
        )
        or (
            query.direction is KnowledgeRelationDirection.INCOMING
            and link.role
            in {
                KnowledgeLineageRole.SUPERSEDED_BY,
                KnowledgeLineageRole.DERIVATION_SOURCE_FOR,
            }
        )
    )
    return (
        direction_matches
        and (not query.kinds or link.kind in query.kinds)
        and link.currentness in query.currentnesses
        and link.counterpart_status in query.counterpart_statuses
        and (not query.unresolved_only or link.unresolved_contradiction)
    )


def _knowledge_lineage_link_bytes(link: KnowledgeLineageLink) -> int:
    return len(
        canonical_durable_json_bytes(
            link.model_dump(mode="json"),
            "knowledge lineage link",
        )
    )


def _knowledge_relation_identity(value: str, field_name: str) -> str:
    return _bounded_knowledge_identity(value, field_name, max_bytes=256)


def _bounded_knowledge_relation_cursor(value: str, field_name: str) -> str:
    value = require_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > MAX_KNOWLEDGE_RELATION_CURSOR_BYTES:
        raise ValueError(
            f"`{field_name}` must be at most {MAX_KNOWLEDGE_RELATION_CURSOR_BYTES} UTF-8 bytes."
        )
    return value


def _copy_enum_filter(value, enum_type: type[Any], field_name: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"`{field_name}` must be a list.")
    copied = [item if isinstance(item, enum_type) else enum_type(item) for item in value]
    return sorted(set(copied), key=lambda item: item.value)
