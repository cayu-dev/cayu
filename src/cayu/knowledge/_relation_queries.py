"""Shared relation and lineage projections, cursor binding and bounded pages."""

from __future__ import annotations

import base64
import binascii
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from cayu._validation import canonical_durable_json_bytes
from cayu.knowledge.changes import _validate_knowledge_change_sequence
from cayu.knowledge.records import KnowledgeRevisionRef, KnowledgeStatus
from cayu.knowledge.relations import (
    _SHA256_HEX_RE,
    KnowledgeLineageCurrentness,
    KnowledgeLineageLink,
    KnowledgeLineageQuery,
    KnowledgeLineageResult,
    KnowledgeLineageRole,
    KnowledgeRelation,
    KnowledgeRelationDirection,
    KnowledgeRelationKind,
    KnowledgeRelationQuery,
    KnowledgeRelationResult,
    _bounded_knowledge_relation_cursor,
    _knowledge_lineage_link_bytes,
    _knowledge_relation_identity,
    copy_knowledge_lineage_link,
    copy_knowledge_lineage_query,
    copy_knowledge_relation,
    copy_knowledge_relation_query,
)
from cayu.knowledge.scopes import KnowledgeAccessScope, _knowledge_access_scope_sha256


class _KnowledgeRelationCursor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    version: Literal[1]
    fingerprint: str
    created_at: datetime
    relation_id: str

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        if type(value) is not str or _SHA256_HEX_RE.fullmatch(value) is None:
            raise ValueError("`fingerprint` must be lowercase SHA-256 hex.")
        return value

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("`created_at` must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("relation_id")
    @classmethod
    def validate_relation_id(cls, value: str) -> str:
        return _knowledge_relation_identity(value, "relation_id")


def _knowledge_relation_matches_lineage_query(
    relation: KnowledgeRelation,
    query: KnowledgeLineageQuery,
) -> bool:
    if query.kinds and relation.kind not in query.kinds:
        return False
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


def _knowledge_lineage_link(
    *,
    relation_id: str,
    kind: KnowledgeRelationKind,
    subject: KnowledgeRevisionRef,
    object_: KnowledgeRevisionRef,
    created_at: datetime,
    reference: KnowledgeRevisionRef,
    subject_current: KnowledgeRevisionRef,
    subject_status: KnowledgeStatus,
    object_current: KnowledgeRevisionRef,
    object_status: KnowledgeStatus,
) -> KnowledgeLineageLink:
    subject_matches = subject == reference
    object_matches = object_ == reference
    if not subject_matches and not object_matches:
        raise ValueError("The inspected revision is not a relation endpoint.")
    if kind is KnowledgeRelationKind.CONTRADICTS:
        role = KnowledgeLineageRole.CONTRADICTS
    elif kind is KnowledgeRelationKind.SUPERSEDES:
        role = (
            KnowledgeLineageRole.SUPERSEDES
            if subject_matches
            else KnowledgeLineageRole.SUPERSEDED_BY
        )
    else:
        role = (
            KnowledgeLineageRole.DERIVED_FROM
            if subject_matches
            else KnowledgeLineageRole.DERIVATION_SOURCE_FOR
        )
    counterpart = object_ if subject_matches else subject
    counterpart_current = object_current if subject_matches else subject_current
    counterpart_status = object_status if subject_matches else subject_status
    currentness = (
        KnowledgeLineageCurrentness.CURRENT
        if (
            subject.revision == subject_current.revision
            and object_.revision == object_current.revision
        )
        else KnowledgeLineageCurrentness.STALE
    )
    unresolved = (
        kind is KnowledgeRelationKind.CONTRADICTS
        and currentness is KnowledgeLineageCurrentness.CURRENT
        and subject_status is KnowledgeStatus.ACTIVE
        and object_status is KnowledgeStatus.ACTIVE
    )
    return KnowledgeLineageLink(
        relation_id=relation_id,
        kind=kind,
        role=role,
        counterpart=counterpart,
        counterpart_current=counterpart_current,
        counterpart_status=counterpart_status,
        currentness=currentness,
        unresolved_contradiction=unresolved,
        created_at=created_at,
    )


def _knowledge_relation_query_fingerprint(
    query: KnowledgeRelationQuery,
    access_scope: KnowledgeAccessScope,
) -> str:
    query = copy_knowledge_relation_query(query)
    return sha256(
        canonical_durable_json_bytes(
            {
                "contract": "cayu-knowledge-relation-query-v1",
                "reference": query.reference.model_dump(mode="json"),
                "direction": query.direction.value,
                "kinds": [kind.value for kind in query.kinds],
                "access_scope_sha256": _knowledge_access_scope_sha256(access_scope),
            },
            "knowledge relation query",
        )
    ).hexdigest()


def _knowledge_lineage_query_fingerprint(
    query: KnowledgeLineageQuery,
    access_scope: KnowledgeAccessScope,
    *,
    through_change_sequence: int | None = None,
) -> str:
    query = copy_knowledge_lineage_query(query)
    payload: dict[str, Any] = {
        "contract": "cayu-knowledge-lineage-query-v1",
        "reference": query.reference.model_dump(mode="json"),
        "direction": query.direction.value,
        "kinds": [kind.value for kind in query.kinds],
        "currentnesses": [value.value for value in query.currentnesses],
        "counterpart_statuses": [status.value for status in query.counterpart_statuses],
        "unresolved_only": query.unresolved_only,
        "access_scope_sha256": _knowledge_access_scope_sha256(access_scope),
    }
    if through_change_sequence is not None:
        _validate_knowledge_change_sequence(
            through_change_sequence,
            "through_change_sequence",
        )
        payload["through_change_sequence"] = through_change_sequence
    return sha256(
        canonical_durable_json_bytes(
            payload,
            "knowledge lineage query",
        )
    ).hexdigest()


def _encode_knowledge_relation_cursor(
    *,
    fingerprint: str,
    relation: KnowledgeRelation,
) -> str:
    cursor = _KnowledgeRelationCursor(
        version=1,
        fingerprint=fingerprint,
        created_at=relation.created_at,
        relation_id=relation.id,
    )
    raw = canonical_durable_json_bytes(
        cursor.model_dump(mode="json"),
        "knowledge relation cursor",
    )
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return _bounded_knowledge_relation_cursor(encoded, "next_cursor")


def _encode_knowledge_lineage_cursor(
    *,
    fingerprint: str,
    link: KnowledgeLineageLink,
) -> str:
    cursor = _KnowledgeRelationCursor(
        version=1,
        fingerprint=fingerprint,
        created_at=link.created_at,
        relation_id=link.relation_id,
    )
    raw = canonical_durable_json_bytes(
        cursor.model_dump(mode="json"),
        "knowledge lineage cursor",
    )
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return _bounded_knowledge_relation_cursor(encoded, "next_cursor")


def _decode_knowledge_relation_cursor(
    cursor: str | None,
    *,
    fingerprint: str,
) -> _KnowledgeRelationCursor | None:
    if cursor is None:
        return None
    cursor = _bounded_knowledge_relation_cursor(cursor, "cursor")
    try:
        encoded = cursor.encode("ascii")
        padding = b"=" * (-len(encoded) % 4)
        raw = base64.b64decode(encoded + padding, altchars=b"-_", validate=True)
        if base64.urlsafe_b64encode(raw).rstrip(b"=") != encoded:
            raise ValueError("Non-canonical relation cursor encoding.")
        parsed = _KnowledgeRelationCursor.model_validate_json(raw)
    except (binascii.Error, TypeError, UnicodeError, ValueError) as exc:
        raise ValueError("Invalid knowledge relation cursor.") from exc
    if parsed.fingerprint != fingerprint:
        raise ValueError(
            "Knowledge relation cursor does not match this reference, direction, "
            "kind filter, and access scope."
        )
    return parsed


def _decode_knowledge_lineage_cursor(
    cursor: str | None,
    *,
    fingerprint: str,
) -> _KnowledgeRelationCursor | None:
    if cursor is None:
        return None
    cursor = _bounded_knowledge_relation_cursor(cursor, "cursor")
    try:
        encoded = cursor.encode("ascii")
        padding = b"=" * (-len(encoded) % 4)
        raw = base64.b64decode(encoded + padding, altchars=b"-_", validate=True)
        if base64.urlsafe_b64encode(raw).rstrip(b"=") != encoded:
            raise ValueError("Non-canonical lineage cursor encoding.")
        parsed = _KnowledgeRelationCursor.model_validate_json(raw)
    except (binascii.Error, TypeError, UnicodeError, ValueError) as exc:
        raise ValueError("Invalid knowledge lineage cursor.") from exc
    if parsed.fingerprint != fingerprint:
        raise ValueError(
            "Knowledge lineage cursor does not match this reference, direction, "
            "relation filters, lifecycle filters, and access scope."
        )
    return parsed


def _bounded_knowledge_relation_result(
    query: KnowledgeRelationQuery,
    candidates: list[KnowledgeRelation],
    *,
    fingerprint: str,
) -> KnowledgeRelationResult:
    selected: list[KnowledgeRelation] = []
    used_bytes = 0
    for relation in candidates:
        if len(selected) >= query.limit:
            break
        relation_bytes = len(
            canonical_durable_json_bytes(
                relation.model_dump(mode="json"),
                "knowledge relation",
            )
        )
        if used_bytes + relation_bytes > query.max_bytes:
            break
        selected.append(copy_knowledge_relation(relation))
        used_bytes += relation_bytes
    truncated = len(selected) < len(candidates)
    next_cursor = (
        _encode_knowledge_relation_cursor(
            fingerprint=fingerprint,
            relation=selected[-1],
        )
        if truncated and selected
        else None
    )
    if truncated and not selected:
        raise RuntimeError("A valid relation did not fit the minimum relation byte budget.")
    return KnowledgeRelationResult(
        query=query,
        relations=selected,
        truncated=truncated,
        next_cursor=next_cursor,
    )


def _bounded_knowledge_lineage_result(
    query: KnowledgeLineageQuery,
    *,
    reference_current: KnowledgeRevisionRef,
    reference_status: KnowledgeStatus,
    candidates: list[KnowledgeLineageLink],
    fingerprint: str,
) -> KnowledgeLineageResult:
    selected: list[KnowledgeLineageLink] = []
    used_bytes = 0
    for link in candidates:
        if len(selected) >= query.limit:
            break
        link_bytes = _knowledge_lineage_link_bytes(link)
        if used_bytes + link_bytes > query.max_bytes:
            break
        selected.append(copy_knowledge_lineage_link(link))
        used_bytes += link_bytes
    truncated = len(selected) < len(candidates)
    next_cursor = (
        _encode_knowledge_lineage_cursor(
            fingerprint=fingerprint,
            link=selected[-1],
        )
        if truncated and selected
        else None
    )
    if truncated and not selected:
        raise RuntimeError("A valid lineage link did not fit the minimum byte budget.")
    return KnowledgeLineageResult(
        query=query,
        reference_current=reference_current,
        reference_status=reference_status,
        links=selected,
        truncated=truncated,
        next_cursor=next_cursor,
    )
