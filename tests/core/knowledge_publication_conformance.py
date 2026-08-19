from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

from cayu.storage import (
    KnowledgeChunk,
    KnowledgeChunkConflict,
    KnowledgeEntry,
    KnowledgePublicationConflict,
    KnowledgePublicationReceipt,
    KnowledgeRevisionConflict,
    KnowledgeStatus,
)


def publication_material(
    *,
    entry_id: str = "owned_publication_entry",
    text: str = "Credential rotation requires an acknowledged handoff.",
    timestamp_offset: int = 0,
) -> tuple[KnowledgeEntry, list[KnowledgeChunk]]:
    timestamp = datetime(2026, 8, 12, 9, 0, tzinfo=UTC) + timedelta(seconds=timestamp_offset)
    entry = KnowledgeEntry(
        id=entry_id,
        text=text,
        namespace="ops",
        labels={"project": "cayu"},
        kind="procedure",
        source_hash=f"source-{timestamp_offset}",
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"owner": f"writer-{timestamp_offset}"},
    )
    chunks = [
        KnowledgeChunk(
            id=f"{entry_id}:0",
            entry_id=entry_id,
            chunk_index=0,
            text=text,
            content_hash=f"chunk-{timestamp_offset}",
            metadata={"owner": f"writer-{timestamp_offset}"},
        )
    ]
    return entry, chunks


async def assert_owned_publication_conformance(store: Any) -> None:
    entry, chunks = publication_material()
    receipt = await store.publish_entry_revision(
        entry,
        chunks,
        operation_id="operation-primary",
    )
    assert type(receipt) is KnowledgePublicationReceipt
    assert receipt.replayed is False
    assert receipt.entry_revision == 1
    assert receipt.expected_revision is None
    assert await store.get_entry(entry.id) == entry
    assert await store.read_chunks(entry.id) == chunks

    invalid_entry, invalid_chunks = publication_material(entry_id="invalid-publication-entry")
    invalid_chunks[0].metadata["non_finite"] = float("inf")
    try:
        await store.publish_entry_revision(
            invalid_entry,
            invalid_chunks,
            operation_id="invalid-publication-operation",
        )
    except (TypeError, ValueError):
        pass
    else:  # pragma: no cover - assertion branch
        raise AssertionError("Invalid publication material reached durable state.")
    await assert_failed_publication_left_no_state(
        store,
        entry_id=invalid_entry.id,
        operation_id="invalid-publication-operation",
    )

    replay = await store.publish_entry_revision(
        entry,
        chunks,
        operation_id="operation-primary",
    )
    assert replay.replayed is True
    assert replay.committed_at == receipt.committed_at
    assert await store.load_entry_publication_receipt("operation-primary") == receipt

    conflicting_entry = entry.model_copy(update={"title": "conflicting request"})
    try:
        await store.publish_entry_revision(
            conflicting_entry,
            chunks,
            operation_id="operation-primary",
        )
    except KnowledgePublicationConflict as exc:
        assert exc.reason == "operation_mismatch"
        assert text_not_in_exception(exc, entry.text)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("An operation identity accepted conflicting material.")

    try:
        await store.publish_entry_revision(
            entry,
            chunks,
            operation_id="operation-second-writer",
        )
    except KnowledgeRevisionConflict as exc:
        assert exc.expected_revision is None
        assert exc.actual_revision == 1
    else:  # pragma: no cover - assertion branch
        raise AssertionError("A second operation overwrote an occupied entry identity.")
    assert await store.get_entry(entry.id) == entry
    assert await store.read_chunks(entry.id) == chunks

    updated = await store.transition_entry_status(
        entry.id,
        expected_revision=entry.revision,
        from_status=KnowledgeStatus.ACTIVE,
        to_status=KnowledgeStatus.ARCHIVED,
    )
    historical_replay = await store.publish_entry_revision(
        entry,
        chunks,
        operation_id="operation-primary",
    )
    assert historical_replay.replayed is True
    assert await store.get_entry(entry.id) == updated

    # Discarding the first result models an acknowledgement lost after commit.
    # The exact retry must return the original immutable receipt without
    # creating or replacing publication material.
    ack_lost_entry, ack_lost_chunks = publication_material(
        entry_id="acknowledgement_loss_publication",
        timestamp_offset=2,
    )
    committed = await store.publish_entry_revision(
        ack_lost_entry,
        ack_lost_chunks,
        operation_id="acknowledgement-loss-operation",
    )
    replayed = await store.publish_entry_revision(
        ack_lost_entry,
        ack_lost_chunks,
        operation_id="acknowledgement-loss-operation",
    )
    assert replayed.replayed is True
    assert replayed.committed_at == committed.committed_at
    assert await store.get_entry(ack_lost_entry.id) == ack_lost_entry
    assert await store.read_chunks(ack_lost_entry.id) == ack_lost_chunks


async def assert_concurrent_publication_conformance(store: Any) -> None:
    entry_a, chunks_a = publication_material(entry_id="concurrent_publication_entry")
    entry_b, chunks_b = publication_material(
        entry_id="concurrent_publication_entry",
        text="A different concurrent owner proposed this material.",
        timestamp_offset=1,
    )

    async def publish(operation_id: str, entry: KnowledgeEntry, chunks: list[KnowledgeChunk]):
        try:
            return (
                operation_id,
                await store.publish_entry_revision(
                    entry,
                    chunks,
                    operation_id=operation_id,
                ),
            )
        except KnowledgeRevisionConflict as exc:
            return operation_id, exc

    outcomes = await asyncio.gather(
        publish("concurrent-a", entry_a, chunks_a),
        publish("concurrent-b", entry_b, chunks_b),
    )
    receipts = [item for item in outcomes if isinstance(item[1], KnowledgePublicationReceipt)]
    conflicts = [item for item in outcomes if isinstance(item[1], KnowledgeRevisionConflict)]
    assert len(receipts) == 1
    assert len(conflicts) == 1
    expected_entry, expected_chunks = (
        (entry_a, chunks_a) if receipts[0][0] == "concurrent-a" else (entry_b, chunks_b)
    )
    assert await store.get_entry(expected_entry.id) == expected_entry
    assert await store.read_chunks(expected_entry.id) == expected_chunks

    chunk_entry_a, chunk_material_a = publication_material(
        entry_id="concurrent_chunk_publication_a",
        text="First owner of a concurrently proposed chunk identity.",
        timestamp_offset=3,
    )
    chunk_entry_b, chunk_material_b = publication_material(
        entry_id="concurrent_chunk_publication_b",
        text="Second owner of a concurrently proposed chunk identity.",
        timestamp_offset=4,
    )
    shared_chunk_id = "concurrent-global-chunk"
    chunk_material_a = [chunk_material_a[0].model_copy(update={"id": shared_chunk_id})]
    chunk_material_b = [chunk_material_b[0].model_copy(update={"id": shared_chunk_id})]

    async def publish_shared_chunk(
        operation_id: str,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk],
    ) -> tuple[str, KnowledgePublicationReceipt | KnowledgeChunkConflict]:
        try:
            return (
                operation_id,
                await store.publish_entry_revision(
                    entry,
                    chunks,
                    operation_id=operation_id,
                ),
            )
        except KnowledgeChunkConflict as exc:
            return operation_id, exc

    chunk_outcomes = await asyncio.gather(
        publish_shared_chunk("concurrent-chunk-a", chunk_entry_a, chunk_material_a),
        publish_shared_chunk("concurrent-chunk-b", chunk_entry_b, chunk_material_b),
    )
    chunk_receipts = [
        item for item in chunk_outcomes if isinstance(item[1], KnowledgePublicationReceipt)
    ]
    chunk_conflicts = [
        item for item in chunk_outcomes if isinstance(item[1], KnowledgeChunkConflict)
    ]
    assert len(chunk_receipts) == 1
    assert len(chunk_conflicts) == 1
    winning_entry, winning_chunks, losing_entry, losing_operation_id = (
        (chunk_entry_a, chunk_material_a, chunk_entry_b, "concurrent-chunk-b")
        if chunk_receipts[0][0] == "concurrent-chunk-a"
        else (chunk_entry_b, chunk_material_b, chunk_entry_a, "concurrent-chunk-a")
    )
    assert await store.get_entry(winning_entry.id) == winning_entry
    assert await store.read_chunks(winning_entry.id) == winning_chunks
    assert await store.get_entry(losing_entry.id) is None
    assert await store.load_entry_publication_receipt(losing_operation_id) is None


async def assert_stale_operation_cannot_replace_newer_publication(store: Any) -> None:
    old_entry, old_chunks = publication_material(entry_id="reused_publication_entry")
    await store.publish_entry_revision(
        old_entry,
        old_chunks,
        operation_id="stale-operation",
    )
    deleted = await store.delete_entry(
        old_entry.id,
        expected_revision=old_entry.revision,
        hard=True,
    )
    assert deleted is not None

    new_entry, new_chunks = publication_material(
        entry_id=old_entry.id,
        text="A newer owner committed this replacement.",
        timestamp_offset=1,
    )
    await store.publish_entry_revision(
        new_entry,
        new_chunks,
        operation_id="newer-operation",
    )

    replay = await store.publish_entry_revision(
        old_entry,
        old_chunks,
        operation_id="stale-operation",
    )
    assert replay.replayed is True
    assert await store.get_entry(old_entry.id) == new_entry
    assert await store.read_chunks(old_entry.id) == new_chunks


async def assert_failed_publication_left_no_state(
    store: Any,
    *,
    entry_id: str,
    operation_id: str,
) -> None:
    """Shared atomicity assertion for backend-specific fault injection."""

    assert await store.get_entry(entry_id) is None
    assert await store.read_chunks(entry_id) == []
    assert await store.load_entry_publication_receipt(operation_id) is None


def text_not_in_exception(exc: BaseException, text: str) -> bool:
    return text not in str(exc) and text not in repr(exc)
