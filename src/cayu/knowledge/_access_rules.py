"""Shared knowledge authorization snapshots, change audiences and access decisions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from cayu._validation import canonical_durable_json_bytes
from cayu.knowledge.activation_contracts import (
    KnowledgeActivationConflict,
    _KnowledgeActivationRetirement,
)
from cayu.knowledge.changes import KnowledgeChange, KnowledgeChangeKind
from cayu.knowledge.maintenance_contracts import MAX_KNOWLEDGE_MAINTENANCE_SOURCES
from cayu.knowledge.records import KnowledgeEntry, KnowledgeStatus
from cayu.knowledge.scopes import (
    KnowledgeAccessDenied,
    KnowledgeAccessScope,
    _knowledge_access_snapshot,
    _KnowledgeAccessSnapshot,
)

_KNOWLEDGE_RETIREMENT_STATUSES = frozenset({KnowledgeStatus.ARCHIVED, KnowledgeStatus.DELETED})


class _KnowledgeRelationAccessSnapshot(BaseModel):
    """Immutable exact-and-current authorization projection for one relation."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    subject_exact: _KnowledgeAccessSnapshot
    subject_current: _KnowledgeAccessSnapshot
    object_exact: _KnowledgeAccessSnapshot
    object_current: _KnowledgeAccessSnapshot

    @field_validator(
        "subject_exact",
        "subject_current",
        "object_exact",
        "object_current",
        mode="before",
    )
    @classmethod
    def copy_snapshot(cls, value) -> _KnowledgeAccessSnapshot:
        if type(value) is _KnowledgeAccessSnapshot:
            return value.model_copy(deep=True)
        if type(value) is dict:
            return _KnowledgeAccessSnapshot.model_validate(value)
        raise TypeError("Relation access authorities require access snapshots.")


class _KnowledgeMaintenanceAccessSnapshot(BaseModel):
    """Immutable authorization projection for the pre-transition reviewed entries."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    entries: list[_KnowledgeAccessSnapshot]

    @field_validator("entries", mode="before")
    @classmethod
    def copy_entries(cls, value) -> list[_KnowledgeAccessSnapshot]:
        if type(value) is not list or not value:
            raise ValueError("Maintenance access snapshots require a non-empty list.")
        if len(value) > MAX_KNOWLEDGE_MAINTENANCE_SOURCES + 1:
            raise ValueError("Maintenance access snapshots exceed the reviewed entry bound.")
        return [
            item.model_copy(deep=True)
            if type(item) is _KnowledgeAccessSnapshot
            else _KnowledgeAccessSnapshot.model_validate(item)
            for item in value
        ]


class _KnowledgeChangeAudience(BaseModel):
    """One immutable authorization audience for a knowledge change."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    kind: Literal[
        "before",
        "after",
        "subject_exact",
        "subject_current",
        "object_exact",
        "object_current",
    ]
    snapshot: _KnowledgeAccessSnapshot
    requires_include_expired: bool = False

    @field_validator("snapshot", mode="before")
    @classmethod
    def copy_snapshot(cls, value: _KnowledgeAccessSnapshot) -> _KnowledgeAccessSnapshot:
        if type(value) is not _KnowledgeAccessSnapshot:
            raise TypeError("Knowledge change audiences require an access snapshot.")
        return value.model_copy(deep=True)

    @field_validator("requires_include_expired", mode="before")
    @classmethod
    def validate_requires_include_expired(cls, value) -> bool:
        if type(value) is not bool:
            raise ValueError("`requires_include_expired` must be a boolean.")
        return value


def _knowledge_relation_access_snapshot(
    *,
    subject_exact: KnowledgeEntry,
    subject_current: KnowledgeEntry,
    object_exact: KnowledgeEntry,
    object_current: KnowledgeEntry,
) -> _KnowledgeRelationAccessSnapshot:
    for role, exact, current in (
        ("subject", subject_exact, subject_current),
        ("object", object_exact, object_current),
    ):
        if exact.id != current.id or exact.revision > current.revision:
            raise ValueError(f"Relation {role} exact/current authorities do not match.")
    return _KnowledgeRelationAccessSnapshot(
        subject_exact=_knowledge_access_snapshot(subject_exact),
        subject_current=_knowledge_access_snapshot(subject_current),
        object_exact=_knowledge_access_snapshot(object_exact),
        object_current=_knowledge_access_snapshot(object_current),
    )


def _knowledge_maintenance_access_snapshot(
    entries: list[KnowledgeEntry],
) -> _KnowledgeMaintenanceAccessSnapshot:
    if type(entries) is not list or not entries:
        raise ValueError("Knowledge maintenance access requires reviewed entries.")
    return _KnowledgeMaintenanceAccessSnapshot(
        entries=[_knowledge_access_snapshot(entry) for entry in entries]
    )


def _knowledge_relation_access_snapshot_json(
    snapshot: _KnowledgeRelationAccessSnapshot,
) -> str:
    if type(snapshot) is not _KnowledgeRelationAccessSnapshot:
        raise TypeError("snapshot must be a _KnowledgeRelationAccessSnapshot.")
    return canonical_durable_json_bytes(
        snapshot.model_dump(mode="json"),
        "knowledge relation access snapshot",
    ).decode("utf-8")


def _knowledge_maintenance_access_snapshot_json(
    snapshot: _KnowledgeMaintenanceAccessSnapshot,
) -> str:
    if type(snapshot) is not _KnowledgeMaintenanceAccessSnapshot:
        raise TypeError("snapshot must be a _KnowledgeMaintenanceAccessSnapshot.")
    return canonical_durable_json_bytes(
        snapshot.model_dump(mode="json"),
        "knowledge maintenance access snapshot",
    ).decode("utf-8")


def _parse_knowledge_relation_access_snapshot_json(
    value: str,
) -> _KnowledgeRelationAccessSnapshot:
    if type(value) is not str:
        raise TypeError("Knowledge relation access snapshot must be JSON text.")
    return _KnowledgeRelationAccessSnapshot.model_validate_json(value)


def _parse_knowledge_maintenance_access_snapshot_json(
    value: str,
) -> _KnowledgeMaintenanceAccessSnapshot:
    if type(value) is not str:
        raise TypeError("Knowledge maintenance access snapshot must be JSON text.")
    return _KnowledgeMaintenanceAccessSnapshot.model_validate_json(value)


def _knowledge_scope_allows_snapshot(
    scope: KnowledgeAccessScope,
    snapshot: _KnowledgeAccessSnapshot,
    *,
    now: datetime | None = None,
) -> bool:
    if not _knowledge_scope_allows_snapshot_dimensions(scope, snapshot):
        return False
    cutoff = datetime.now(UTC) if now is None else now
    return scope.include_expired or snapshot.expires_at is None or snapshot.expires_at > cutoff


def _knowledge_scope_allows_activation_receipt(
    scope: KnowledgeAccessScope,
    snapshot: _KnowledgeAccessSnapshot,
    current_entry: KnowledgeEntry | None,
    *,
    retirement: _KnowledgeActivationRetirement | None,
    entry_id: str,
    entry_revision: int,
    now: datetime | None = None,
) -> bool:
    """Authorize content-bearing activation history against its current entry.

    Expiration pruning deliberately retains a separate content-free final access
    authority. A missing current entry without that explicit marker is malformed;
    receipt-local expiry is never treated as evidence that pruning occurred.
    """

    cutoff = datetime.now(UTC) if now is None else now
    if not _knowledge_scope_allows_snapshot(scope, snapshot, now=cutoff):
        return False
    if current_entry is not None:
        if retirement is not None:
            raise KnowledgeActivationConflict("malformed_retirement")
        return _knowledge_scope_allows_entry(scope, current_entry, now=cutoff)
    if (
        retirement is None
        or retirement.entry_id != entry_id
        or retirement.entry_revision < entry_revision
    ):
        raise KnowledgeActivationConflict("malformed_receipt")
    return scope.include_expired and _knowledge_scope_allows_snapshot_dimensions(
        scope,
        retirement.access_snapshot,
    )


def _require_knowledge_activation_retirement_access(
    scope: KnowledgeAccessScope,
    retirement: _KnowledgeActivationRetirement,
    *,
    operation: str,
) -> None:
    if not scope.include_expired or not _knowledge_scope_allows_snapshot_dimensions(
        scope,
        retirement.access_snapshot,
    ):
        raise KnowledgeAccessDenied(operation)


def _knowledge_scope_allows_snapshot_dimensions(
    scope: KnowledgeAccessScope,
    snapshot: _KnowledgeAccessSnapshot,
) -> bool:
    from cayu.knowledge.access import matches

    if not matches(scope, snapshot.labels):
        return False
    if not scope.allow_all_namespaces and snapshot.namespace not in scope.allowed_namespaces:
        return False
    for key, value in scope.required_labels.items():
        if snapshot.labels.get(key) != value:
            return False
    if snapshot.visibility not in scope.allowed_visibilities:
        return False
    if (
        scope.allowed_source_types is not None
        and snapshot.source_type not in scope.allowed_source_types
    ):
        return False
    if scope.allowed_source_ids is not None and snapshot.source_id not in scope.allowed_source_ids:
        return False
    return snapshot.status in scope.allowed_statuses


def _knowledge_scope_allows_relation_access_snapshot(
    scope: KnowledgeAccessScope,
    snapshot: _KnowledgeRelationAccessSnapshot,
) -> bool:
    now = datetime.now(UTC)
    return all(
        _knowledge_scope_allows_snapshot(scope, authority, now=now)
        for authority in (
            snapshot.subject_exact,
            snapshot.subject_current,
            snapshot.object_exact,
            snapshot.object_current,
        )
    )


def _knowledge_scope_allows_maintenance_access_snapshot(
    scope: KnowledgeAccessScope,
    snapshot: _KnowledgeMaintenanceAccessSnapshot,
) -> bool:
    now = datetime.now(UTC)
    return all(
        _knowledge_scope_allows_snapshot(scope, authority, now=now)
        for authority in snapshot.entries
    )


def _knowledge_scope_allows_change_audience(
    scope: KnowledgeAccessScope,
    audience: _KnowledgeChangeAudience,
) -> bool:
    return (
        scope.include_expired or not audience.requires_include_expired
    ) and _knowledge_scope_allows_snapshot_dimensions(scope, audience.snapshot)


def _knowledge_scope_allows_change(
    scope: KnowledgeAccessScope,
    change: KnowledgeChange,
    audiences: tuple[_KnowledgeChangeAudience, ...],
) -> bool:
    results = [_knowledge_scope_allows_change_audience(scope, audience) for audience in audiences]
    if not results:
        return False
    if change.kind is KnowledgeChangeKind.RELATION_PUBLISHED:
        return (
            len(results) == 4
            and {audience.kind for audience in audiences}
            == {
                "subject_exact",
                "subject_current",
                "object_exact",
                "object_current",
            }
            and all(results)
        )
    return any(results)


def _knowledge_change_audiences(
    change: KnowledgeChange,
    *,
    before_entry: KnowledgeEntry | None,
    after_entry: KnowledgeEntry | None,
    before_requires_include_expired: bool | None = None,
) -> tuple[_KnowledgeChangeAudience, ...]:
    if before_entry is None and after_entry is None:
        raise ValueError("A knowledge change requires a before or after entry.")
    if before_entry is not None and before_entry.id != change.entry_id:
        raise ValueError("Knowledge change before-entry identity does not match the change.")
    if after_entry is not None and (
        after_entry.id != change.entry_id or after_entry.revision != change.entry_revision
    ):
        raise ValueError("Knowledge change after-entry identity does not match the change.")
    if after_entry is None and (
        before_entry is None or before_entry.revision != change.entry_revision
    ):
        raise ValueError("Knowledge removal change revision does not match its before entry.")

    audiences: list[_KnowledgeChangeAudience] = []
    for kind, entry in (("after", after_entry), ("before", before_entry)):
        if entry is None:
            continue
        snapshot = _knowledge_access_snapshot(entry)
        requires_include_expired = (
            snapshot.expires_at is not None and snapshot.expires_at <= change.committed_at
        )
        if kind == "before" and before_requires_include_expired is not None:
            # Preserve the expiration audience captured when this exact revision
            # was published. A slow consumer can then receive its removal signal,
            # while an entry already expired at publication never widens access.
            requires_include_expired = before_requires_include_expired
        if any(
            existing.snapshot == snapshot
            and existing.requires_include_expired == requires_include_expired
            for existing in audiences
        ):
            continue
        audiences.append(
            _KnowledgeChangeAudience(
                kind=kind,
                snapshot=snapshot,
                requires_include_expired=requires_include_expired,
            )
        )
    return tuple(audiences)


def _knowledge_relation_change_audiences(
    change: KnowledgeChange,
    *,
    access_snapshot: _KnowledgeRelationAccessSnapshot,
) -> tuple[_KnowledgeChangeAudience, ...]:
    if change.kind is not KnowledgeChangeKind.RELATION_PUBLISHED:
        raise ValueError("A relation audience requires a relation publication change.")
    if type(access_snapshot) is not _KnowledgeRelationAccessSnapshot:
        raise TypeError("A relation audience requires a relation access snapshot.")
    return tuple(
        _KnowledgeChangeAudience(
            kind=kind,
            snapshot=snapshot,
            requires_include_expired=(
                snapshot.expires_at is not None and snapshot.expires_at <= change.committed_at
            ),
        )
        for kind, snapshot in (
            ("subject_exact", access_snapshot.subject_exact),
            ("subject_current", access_snapshot.subject_current),
            ("object_exact", access_snapshot.object_exact),
            ("object_current", access_snapshot.object_current),
        )
    )


def _knowledge_scope_allows_entry(
    scope: KnowledgeAccessScope,
    entry: KnowledgeEntry,
    *,
    now: datetime | None = None,
) -> bool:
    return _knowledge_scope_allows_snapshot(
        scope,
        _knowledge_access_snapshot(entry),
        now=now,
    )


def _require_knowledge_entry_access(
    scope: KnowledgeAccessScope,
    entry: KnowledgeEntry,
    *,
    operation: str,
) -> None:
    if not _knowledge_scope_allows_entry(scope, entry):
        raise KnowledgeAccessDenied(operation)


def _require_knowledge_successor_access(
    scope: KnowledgeAccessScope,
    entry: KnowledgeEntry,
    *,
    operation: str,
) -> None:
    """Authorize a successor without coupling retirement to audit visibility.

    A principal that can mutate the current revision may retire it without also
    receiving read access to archived or deleted material. Every other scope
    dimension remains enforced, and promotion/reactivation still requires the
    destination status to be present in the supplied scope.
    """

    if (
        entry.status in _KNOWLEDGE_RETIREMENT_STATUSES
        and entry.status not in scope.allowed_statuses
    ):
        retirement_scope = scope.model_copy(
            update={
                "allowed_statuses": sorted(
                    {*scope.allowed_statuses, entry.status},
                    key=str,
                )
            }
        )
        _require_knowledge_entry_access(retirement_scope, entry, operation=operation)
        return
    _require_knowledge_entry_access(scope, entry, operation=operation)


def _knowledge_scope_allows_lineage_endpoint(
    scope: KnowledgeAccessScope,
    exact: KnowledgeEntry,
    current: KnowledgeEntry,
    *,
    now: datetime | None = None,
) -> bool:
    """Authorize a safe reference while keeping archived content inaccessible.

    The exact revision must satisfy the caller's complete scope. Its logical current
    revision must satisfy the same scope, except that reviewed archival may be
    observed as lifecycle metadata. Deleted, pending, expired, or otherwise
    inaccessible authorities are never widened.
    """

    if exact.id != current.id or exact.revision > current.revision:
        raise ValueError("Lineage exact/current endpoint authorities do not match.")
    cutoff = datetime.now(UTC) if now is None else now
    if not _knowledge_scope_allows_entry(scope, exact, now=cutoff):
        return False
    if _knowledge_scope_allows_entry(scope, current, now=cutoff):
        return True
    if current.status is not KnowledgeStatus.ARCHIVED:
        return False
    archived_scope = scope.model_copy(
        update={
            "allowed_statuses": sorted(
                {*scope.allowed_statuses, KnowledgeStatus.ARCHIVED},
                key=str,
            )
        }
    )
    return _knowledge_scope_allows_entry(archived_scope, current, now=cutoff)
