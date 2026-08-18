from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from cayu.storage import (
    KnowledgeAccessDenied,
    KnowledgeAccessScope,
    KnowledgeChunk,
    KnowledgeChunkConflict,
    KnowledgeEntry,
    KnowledgeListQuery,
    KnowledgeQuery,
    KnowledgeStatus,
    KnowledgeStore,
    KnowledgeVisibility,
)


def _entry(
    entry_id: str,
    *,
    namespace: str,
    project: str,
    source_id: str,
    status: KnowledgeStatus = KnowledgeStatus.ACTIVE,
    visibility: KnowledgeVisibility = KnowledgeVisibility.GLOBAL,
    expires_at: datetime | None = None,
) -> KnowledgeEntry:
    return KnowledgeEntry(
        id=entry_id,
        text=f"sharedscope {entry_id}",
        namespace=namespace,
        labels={"project": project},
        visibility=visibility,
        status=status,
        source_type="document",
        source_id=source_id,
        expires_at=expires_at,
    )


def _chunk(
    entry: KnowledgeEntry,
    *,
    chunk_id: str | None = None,
    chunk_index: int = 0,
) -> KnowledgeChunk:
    return KnowledgeChunk(
        id=f"{entry.id}:0" if chunk_id is None else chunk_id,
        entry_id=entry.id,
        chunk_index=chunk_index,
        text=entry.text,
    )


async def assert_knowledge_access_scope_conformance(store: KnowledgeStore) -> None:
    """Exercise the same fail-closed boundary against every first-party backend."""

    privileged = KnowledgeAccessScope.privileged()
    allowed = _entry(
        "allowed",
        namespace="tenant-a",
        project="alpha",
        source_id="source-a",
    )
    denied_namespace = _entry(
        "denied-namespace",
        namespace="tenant-b",
        project="alpha",
        source_id="source-a",
    )
    denied_label = _entry(
        "denied-label",
        namespace="tenant-a",
        project="beta",
        source_id="source-a",
    )
    denied_source = _entry(
        "denied-source",
        namespace="tenant-a",
        project="alpha",
        source_id="source-b",
    )
    denied_visibility = _entry(
        "denied-visibility",
        namespace="tenant-a",
        project="alpha",
        source_id="source-a",
        visibility=KnowledgeVisibility.SESSION,
    )
    denied_status = _entry(
        "denied-status",
        namespace="tenant-a",
        project="alpha",
        source_id="source-a",
        status=KnowledgeStatus.ARCHIVED,
    )
    denied_expired = _entry(
        "denied-expired",
        namespace="tenant-a",
        project="alpha",
        source_id="source-a",
        expires_at=datetime.now(UTC) - timedelta(minutes=1),
    )
    for entry in (
        allowed,
        denied_namespace,
        denied_label,
        denied_source,
        denied_visibility,
        denied_status,
        denied_expired,
    ):
        await store.put_entry_with_chunks(
            entry,
            [_chunk(entry)],
            access_scope=privileged,
        )

    scope = KnowledgeAccessScope.for_namespace(
        "tenant-a",
        required_labels={"project": "alpha"},
        allowed_visibilities=[KnowledgeVisibility.GLOBAL],
        allowed_source_types=["document"],
        allowed_source_ids=["source-a"],
        allowed_statuses=[KnowledgeStatus.ACTIVE],
    )

    search = await store.search(
        KnowledgeQuery(text="sharedscope", namespace="tenant-a"),
        access_scope=scope,
    )
    assert [hit.entry.id for hit in search.hits] == ["allowed"]
    listed = await store.list_entries(KnowledgeListQuery(limit=20), access_scope=scope)
    assert [item.entry.id for item in listed.entries] == ["allowed"]
    assert await store.get_entry(allowed.id, access_scope=scope) == allowed
    assert [chunk.id for chunk in await store.read_chunks(allowed.id, access_scope=scope)] == [
        "allowed:0"
    ]

    denied_ids = (
        denied_namespace.id,
        denied_label.id,
        denied_source.id,
        denied_visibility.id,
        denied_status.id,
        denied_expired.id,
    )
    for entry_id in denied_ids:
        assert await store.get_entry(entry_id, access_scope=scope) is None
        assert await store.read_chunks(entry_id, access_scope=scope) == []

    with pytest.raises(KnowledgeAccessDenied):
        await store.update_entry_status(
            denied_namespace.id,
            KnowledgeStatus.ARCHIVED,
            access_scope=scope,
        )
    with pytest.raises(KnowledgeAccessDenied):
        await store.transition_entry_status(
            denied_namespace.id,
            from_status=KnowledgeStatus.ACTIVE,
            to_status=KnowledgeStatus.ARCHIVED,
            access_scope=scope,
        )
    with pytest.raises(KnowledgeAccessDenied):
        await store.delete_entry(denied_namespace.id, access_scope=scope, hard=True)
    with pytest.raises(KnowledgeAccessDenied):
        await store.replace_chunks(
            denied_namespace.id,
            [_chunk(denied_namespace)],
            access_scope=scope,
        )

    takeover = allowed.model_copy(update={"id": denied_namespace.id})
    with pytest.raises(KnowledgeAccessDenied):
        await store.put_entry(takeover, access_scope=scope)
    with pytest.raises(KnowledgeAccessDenied):
        await store.put_entry_with_chunks(
            takeover,
            [_chunk(takeover)],
            access_scope=scope,
        )
    with pytest.raises(KnowledgeAccessDenied):
        await store.publish_entry_with_chunks(
            takeover,
            [_chunk(takeover)],
            operation_id="scoped-entry-takeover",
            access_scope=scope,
        )

    receipt_entry = allowed.model_copy(
        update={"id": "receipt-entry", "text": "sharedscope durable receipt"}
    )
    receipt = await store.publish_entry_with_chunks(
        receipt_entry,
        [_chunk(receipt_entry)],
        operation_id="scoped-receipt",
        access_scope=scope,
    )
    assert receipt.replayed is False
    await store.delete_entry(receipt_entry.id, access_scope=scope, hard=True)
    assert (
        await store.load_entry_publication_receipt(
            "scoped-receipt",
            access_scope=scope,
        )
        is not None
    )
    replay = await store.publish_entry_with_chunks(
        receipt_entry,
        [_chunk(receipt_entry)],
        operation_id="scoped-receipt",
        access_scope=scope,
    )
    assert replay.replayed is True
    other_scope = KnowledgeAccessScope.for_namespace(
        "tenant-b",
        allowed_source_types=["document"],
        allowed_source_ids=["source-a"],
    )
    assert (
        await store.load_entry_publication_receipt(
            "scoped-receipt",
            access_scope=other_scope,
        )
        is None
    )
    other_receipt_entry = _entry(
        "other-receipt-entry",
        namespace="tenant-b",
        project="alpha",
        source_id="source-a",
    )
    with pytest.raises(KnowledgeAccessDenied):
        await store.publish_entry_with_chunks(
            other_receipt_entry,
            [_chunk(other_receipt_entry)],
            operation_id="scoped-receipt",
            access_scope=other_scope,
        )

    foreign_chunk_owner = _entry(
        "foreign-chunk-owner",
        namespace="tenant-b",
        project="alpha",
        source_id="source-a",
    )
    foreign_chunk_id = "foreign-global-chunk"
    foreign_default_chunk_id = "foreign-default-probe:0"
    await store.put_entry_with_chunks(
        foreign_chunk_owner,
        [
            _chunk(foreign_chunk_owner, chunk_id=foreign_chunk_id),
            _chunk(
                foreign_chunk_owner,
                chunk_id=foreign_default_chunk_id,
                chunk_index=1,
            ),
        ],
        access_scope=privileged,
    )
    foreign_replace_target = _entry(
        "foreign-replace-target",
        namespace="tenant-a",
        project="alpha",
        source_id="source-a",
    )
    foreign_replace_original = _chunk(foreign_replace_target)
    await store.put_entry_with_chunks(
        foreign_replace_target,
        [foreign_replace_original],
        access_scope=scope,
    )
    foreign_default_probe = _entry(
        "foreign-default-probe",
        namespace="tenant-a",
        project="alpha",
        source_id="source-a",
    )
    with pytest.raises(KnowledgeAccessDenied) as denied_default:
        await store.put_entry(foreign_default_probe, access_scope=scope)
    assert foreign_chunk_owner.id not in str(denied_default.value)
    assert foreign_default_chunk_id not in str(denied_default.value)
    assert await store.get_entry(foreign_default_probe.id, access_scope=scope) is None

    foreign_put_probe = _entry(
        "foreign-put-probe",
        namespace="tenant-a",
        project="alpha",
        source_id="source-a",
    )
    with pytest.raises(KnowledgeAccessDenied):
        await store.put_entry_with_chunks(
            foreign_put_probe,
            [_chunk(foreign_put_probe, chunk_id=foreign_chunk_id)],
            access_scope=scope,
        )
    assert await store.get_entry(foreign_put_probe.id, access_scope=scope) is None

    foreign_publication_probe = _entry(
        "foreign-publication-probe",
        namespace="tenant-a",
        project="alpha",
        source_id="source-a",
    )
    with pytest.raises(KnowledgeAccessDenied):
        await store.publish_entry_with_chunks(
            foreign_publication_probe,
            [_chunk(foreign_publication_probe, chunk_id=foreign_chunk_id)],
            operation_id="foreign-chunk-publication",
            access_scope=scope,
        )
    assert await store.get_entry(foreign_publication_probe.id, access_scope=scope) is None
    assert (
        await store.load_entry_publication_receipt(
            "foreign-chunk-publication",
            access_scope=scope,
        )
        is None
    )

    with pytest.raises(KnowledgeAccessDenied):
        await store.replace_chunks(
            foreign_replace_target.id,
            [_chunk(foreign_replace_target, chunk_id=foreign_chunk_id)],
            access_scope=scope,
        )
    assert await store.read_chunks(foreign_replace_target.id, access_scope=scope) == [
        foreign_replace_original
    ]

    accessible_chunk_owner = _entry(
        "accessible-chunk-owner",
        namespace="tenant-a",
        project="alpha",
        source_id="source-a",
    )
    accessible_chunk_id = "accessible-global-chunk"
    accessible_default_chunk_id = "accessible-default-probe:0"
    await store.put_entry_with_chunks(
        accessible_chunk_owner,
        [
            _chunk(accessible_chunk_owner, chunk_id=accessible_chunk_id),
            _chunk(
                accessible_chunk_owner,
                chunk_id=accessible_default_chunk_id,
                chunk_index=1,
            ),
        ],
        access_scope=scope,
    )
    accessible_replace_target = _entry(
        "accessible-replace-target",
        namespace="tenant-a",
        project="alpha",
        source_id="source-a",
    )
    accessible_replace_original = _chunk(accessible_replace_target)
    await store.put_entry_with_chunks(
        accessible_replace_target,
        [accessible_replace_original],
        access_scope=scope,
    )
    accessible_default_probe = _entry(
        "accessible-default-probe",
        namespace="tenant-a",
        project="alpha",
        source_id="source-a",
    )
    with pytest.raises(KnowledgeChunkConflict):
        await store.put_entry(accessible_default_probe, access_scope=scope)
    assert await store.get_entry(accessible_default_probe.id, access_scope=scope) is None

    accessible_put_probe = _entry(
        "accessible-put-probe",
        namespace="tenant-a",
        project="alpha",
        source_id="source-a",
    )
    with pytest.raises(KnowledgeChunkConflict):
        await store.put_entry_with_chunks(
            accessible_put_probe,
            [_chunk(accessible_put_probe, chunk_id=accessible_chunk_id)],
            access_scope=scope,
        )
    assert await store.get_entry(accessible_put_probe.id, access_scope=scope) is None

    accessible_publication_probe = _entry(
        "accessible-publication-probe",
        namespace="tenant-a",
        project="alpha",
        source_id="source-a",
    )
    with pytest.raises(KnowledgeChunkConflict):
        await store.publish_entry_with_chunks(
            accessible_publication_probe,
            [_chunk(accessible_publication_probe, chunk_id=accessible_chunk_id)],
            operation_id="accessible-chunk-publication",
            access_scope=scope,
        )
    assert await store.get_entry(accessible_publication_probe.id, access_scope=scope) is None
    assert (
        await store.load_entry_publication_receipt(
            "accessible-chunk-publication",
            access_scope=scope,
        )
        is None
    )

    with pytest.raises(KnowledgeChunkConflict):
        await store.replace_chunks(
            accessible_replace_target.id,
            [_chunk(accessible_replace_target, chunk_id=accessible_chunk_id)],
            access_scope=scope,
        )
    assert await store.read_chunks(accessible_replace_target.id, access_scope=scope) == [
        accessible_replace_original
    ]

    future_expiring = _entry(
        "future-expiring",
        namespace="tenant-a",
        project="alpha",
        source_id="source-a",
        expires_at=datetime.now(UTC) + timedelta(days=1),
    )
    await store.put_entry_with_chunks(
        future_expiring,
        [_chunk(future_expiring)],
        access_scope=privileged,
    )
    assert (
        await store.prune_expired(
            access_scope=scope,
            now=future_expiring.expires_at + timedelta(days=1),
        )
        == 0
    )
    assert await store.get_entry(future_expiring.id, access_scope=scope) == future_expiring
