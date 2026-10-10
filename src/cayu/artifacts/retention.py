"""Age and size retention for artifact stores.

Artifacts carry no reference index, so retention decides what is still needed
from the stores that can name them:

- durable pins (public, resource-retention and workspace-checkpoint pins),
  read with :meth:`ArtifactStore.has_retention_pins` and enforced again by
  ``delete``;
- the owning session: a session-scoped artifact is kept while its session
  still exists or while a session-closure claim fences it, so session
  retention runs first;
- references other stores report (transcripts, session operations, eval and
  knowledge evidence), passed in ``references``;
- identifiers the caller names in ``protected_artifact_ids``.

An artifact store has no transaction to share with an audit record, so every
apply writes its audit through a :class:`RetentionAuditSink`, one entry after
each deletion. A store that cannot report pins keeps every artifact.
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from cayu._task_wait import await_shielded_task_outcome, restore_task_cancellation_requests
from cayu.artifacts.base import ArtifactMetadata, ArtifactStore
from cayu.storage._artifact_retention import artifact_reference_guard, reference_databases
from cayu.storage.retention import (
    MAX_RETENTION_ITEMS_PER_RUN,
    ArtifactRetentionPolicy,
    RetentionAuditEntry,
    RetentionAuditSink,
    RetentionDisposition,
    RetentionItem,
    RetentionPhase,
    RetentionProgress,
    RetentionProgressCallback,
    RetentionProtection,
    RetentionReport,
    bounded_retention_detail,
    copy_protected_ids,
    retention_audit_summary,
)

STORE_KIND = "artifacts"


@dataclass(frozen=True)
class _Candidate:
    metadata: ArtifactMetadata

    @property
    def id(self) -> str:
        return self.metadata.id


def _item(
    metadata: ArtifactMetadata,
    disposition: RetentionDisposition,
    *,
    protections: Collection[RetentionProtection] = (),
    detail: str | None = None,
) -> RetentionItem:
    return RetentionItem(
        item_id=metadata.id,
        status=metadata.scope.value,
        last_updated_at=metadata.created_at,
        disposition=disposition,
        protections=tuple(sorted(set(protections), key=lambda value: value.value)),
        detail=bounded_retention_detail(detail),
        counts=(
            {"artifacts_removed": 1} if disposition is not RetentionDisposition.PROTECTED else {}
        ),
        bytes=metadata.size_bytes if disposition is not RetentionDisposition.PROTECTED else 0,
    )


async def _protections(
    store: ArtifactStore,
    metadata: ArtifactMetadata,
    *,
    session_store: Any,
    references: Mapping[RetentionProtection, frozenset[str]],
    caller_protected: frozenset[str],
) -> tuple[set[RetentionProtection], str | None]:
    reasons: set[RetentionProtection] = set()
    detail: str | None = None
    if metadata.id in caller_protected:
        reasons.add(RetentionProtection.CALLER_PROTECTED)
    for protection, identifiers in references.items():
        if metadata.id in identifiers:
            reasons.add(protection)
    pinned = await store.has_retention_pins(metadata.id)
    if pinned is None:
        reasons.add(RetentionProtection.PINNED)
        detail = "the artifact store cannot report durable pins"
    elif pinned:
        reasons.add(RetentionProtection.PINNED)
    if metadata.session_id is not None:
        if session_store is None:
            reasons.add(RetentionProtection.SESSION_REFERENCE)
            detail = detail or "no session store was given to check the owning session"
        elif (
            store.supports_session_closure_claims
            and await store.load_session_closure_claim(metadata.session_id) is not None
        ):
            reasons.add(RetentionProtection.SESSION_REFERENCE)
            detail = detail or "a session closure claim fences the artifact"
    return reasons, detail


async def apply_artifact_retention_policy(
    store: ArtifactStore,
    policy: ArtifactRetentionPolicy,
    *,
    session_store: Any = None,
    eval_store: Any = None,
    snapshot_stores: Iterable[Any] = (),
    references: Mapping[RetentionProtection, Collection[str]] | None = None,
    protected_artifact_ids: Collection[str] = (),
    audit: RetentionAuditSink | None = None,
    progress: RetentionProgressCallback | None = None,
) -> RetentionReport:
    """Dry-run or apply ``policy`` to one artifact store; an operator-only action.

    Pass every session, eval and snapshot store that can publish references.
    Apply fences their databases through each deletion; unsupported sources
    protect candidates with ``erasure_guard``. Explicit ``references`` only
    add exclusions and cannot replace the live stores.
    """

    if not isinstance(store, ArtifactStore):
        raise TypeError("store must be an ArtifactStore.")
    if type(policy) is not ArtifactRetentionPolicy:
        raise TypeError("policy must be an ArtifactRetentionPolicy.")
    if not policy.dry_run and audit is None:
        raise ValueError("An artifact retention apply requires a durable audit sink.")
    caller_protected = copy_protected_ids(protected_artifact_ids, "protected_artifact_ids")
    known_references = {
        RetentionProtection(protection): copy_protected_ids(identifiers, "references")
        for protection, identifiers in (references or {}).items()
    }
    databases = await reference_databases((session_store, eval_store, *snapshot_stores))
    started_at = datetime.now(UTC)
    cutoff = started_at - policy.older_than
    listing = await store.list(limit=None)
    all_artifacts = sorted(listing.artifacts, key=lambda item: (item.created_at, item.id))
    remaining_total = sum(item.size_bytes for item in all_artifacts)
    candidates = [
        item
        for item in all_artifacts
        if item.scope.value in policy.scopes
        and item.created_at <= cutoff
        and item.size_bytes >= policy.min_size_bytes
    ]
    selected: list[ArtifactMetadata] = []
    protected: list[RetentionItem] = []
    deferred = 0
    selected_bytes = 0
    for index, metadata in enumerate(candidates):
        if (
            len(selected) >= policy.max_items
            or (policy.max_bytes is not None and selected_bytes >= policy.max_bytes)
            or (
                policy.target_total_bytes is not None
                and remaining_total <= policy.target_total_bytes
            )
        ):
            deferred = len(candidates) - index
            break
        reasons, detail = await _protections(
            store,
            metadata,
            session_store=session_store,
            references=known_references,
            caller_protected=caller_protected,
        )
        if not reasons:
            async with artifact_reference_guard(databases, metadata, apply=False) as current:
                reasons.update(current)
            if RetentionProtection.ERASURE_GUARD in reasons:
                detail = "configured reference stores cannot fence artifact deletion"
        if reasons:
            protected.append(
                _item(metadata, RetentionDisposition.PROTECTED, protections=reasons, detail=detail)
            )
            continue
        selected.append(metadata)
        selected_bytes += metadata.size_bytes
        remaining_total -= metadata.size_bytes
    policy_document = {**policy.policy_document(), "store_id": store.id}
    if progress is not None:
        await progress(
            RetentionProgress(
                store_kind=STORE_KIND,
                phase=RetentionPhase.PLANNED,
                dry_run=policy.dry_run,
                planned_items=len(selected),
                protected=tuple(protected[:MAX_RETENTION_ITEMS_PER_RUN]),
            )
        )
    if policy.dry_run:
        return _report(
            policy,
            policy_document,
            started_at,
            cutoff,
            None,
            [_item(metadata, RetentionDisposition.SELECTED) for metadata in selected],
            protected,
            deferred,
        )
    assert audit is not None
    audit_id = await audit.begin_retention_audit(
        store_kind=STORE_KIND,
        mode=policy.mode,
        policy=policy_document,
        started_at=started_at,
    )
    audit_sink = audit

    async def apply_candidate(
        metadata: ArtifactMetadata,
    ) -> tuple[tuple[RetentionItem, ...], tuple[RetentionItem, ...]]:
        batch_applied: tuple[RetentionItem, ...] = ()
        batch_protected: tuple[RetentionItem, ...] = ()
        async with artifact_reference_guard(databases, metadata, apply=True) as current:
            reasons, detail = await _protections(
                store,
                metadata,
                session_store=session_store,
                references=known_references,
                caller_protected=caller_protected,
            )
            reasons.update(current)
            if RetentionProtection.ERASURE_GUARD in reasons:
                detail = "configured reference stores cannot fence artifact deletion"
            if not reasons:
                try:
                    await store.delete(metadata.id)
                except ValueError as refused:
                    reasons, detail = {RetentionProtection.PINNED}, str(refused)
        if reasons:
            batch_protected = (
                _item(metadata, RetentionDisposition.PROTECTED, protections=reasons, detail=detail),
            )
        else:
            item = _item(metadata, RetentionDisposition.APPLIED)
            await audit_sink.record_retention_entry(
                audit_id,
                RetentionAuditEntry(
                    item_id=metadata.id,
                    action=policy.mode,
                    counts=item.counts,
                    bytes=item.bytes,
                    recorded_at=datetime.now(UTC),
                ),
            )
            batch_applied = (item,)
        return batch_applied, batch_protected

    applied: list[RetentionItem] = []
    for index, metadata in enumerate(selected):
        # Deletion may dispatch blocking I/O. Keep the database fences until
        # that work settles, then record its audit before honoring cancellation.
        outcome = await await_shielded_task_outcome(asyncio.create_task(apply_candidate(metadata)))
        if outcome.cancellation is not None:
            restore_task_cancellation_requests(
                outcome.cancellation_requests_consumed, cancellation=outcome.cancellation
            )
            raise outcome.cancellation from outcome.error
        if outcome.error is not None:
            raise outcome.error
        assert outcome.result is not None
        batch_applied, batch_protected = outcome.result
        applied.extend(batch_applied)
        protected.extend(batch_protected)
        if progress is not None:
            await progress(
                RetentionProgress(
                    store_kind=STORE_KIND,
                    phase=RetentionPhase.BATCH,
                    dry_run=False,
                    audit_id=audit_id,
                    batch_index=index,
                    planned_items=len(selected),
                    applied=batch_applied,
                    protected=batch_protected,
                )
            )
        await asyncio.sleep(0)
    report = _report(
        policy, policy_document, started_at, cutoff, audit_id, applied, protected, deferred
    )
    await audit.complete_retention_audit(
        audit_id, completed_at=report.completed_at, summary=retention_audit_summary(report)
    )
    return report


def _report(
    policy: ArtifactRetentionPolicy,
    policy_document: dict[str, Any],
    started_at: datetime,
    cutoff: datetime,
    audit_id: str | None,
    items: list[RetentionItem],
    protected: list[RetentionItem],
    deferred: int,
) -> RetentionReport:
    return RetentionReport(
        store_kind=STORE_KIND,
        mode=policy.mode,
        dry_run=policy.dry_run,
        policy=policy_document,
        started_at=started_at,
        completed_at=datetime.now(UTC),
        cutoff=cutoff,
        audit_id=audit_id,
        items=tuple(items),
        protected=tuple(protected[:MAX_RETENTION_ITEMS_PER_RUN]),
        protected_truncated=len(protected) > MAX_RETENTION_ITEMS_PER_RUN,
        deferred_count=deferred,
    )


__all__ = ["apply_artifact_retention_policy"]
