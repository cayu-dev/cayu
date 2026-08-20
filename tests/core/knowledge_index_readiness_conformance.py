from __future__ import annotations

import asyncio

import pytest

from cayu.storage import (
    KnowledgeChunk,
    KnowledgeEmbeddingIdentity,
    KnowledgeEntry,
    KnowledgeIndexReadinessConflict,
    KnowledgeIndexReadinessUpdate,
    KnowledgeIndexState,
    KnowledgeStore,
    knowledge_chunk_embedding_identity,
)


def embedding_identity(
    entry: KnowledgeEntry,
    *,
    chunk: KnowledgeChunk,
) -> KnowledgeEmbeddingIdentity:
    assert chunk.entry_id == entry.id
    assert chunk.entry_revision == entry.revision
    return knowledge_chunk_embedding_identity(
        chunk,
        embedding_model="conformance-embedding-v1",
        dimensions=3,
    )


async def assert_index_readiness_conformance(store: KnowledgeStore) -> None:
    created = await store.create_entry(
        KnowledgeEntry(id="readiness-entry", text="Revision-bound projection readiness.")
    )
    chunks = await store.read_chunks(created.id)
    assert len(chunks) == 1
    chunk = chunks[0]
    identity = embedding_identity(
        created,
        chunk=chunk,
    )
    pending = KnowledgeIndexReadinessUpdate(
        identity=identity,
        state=KnowledgeIndexState.PENDING,
        attempt_id="attempt-1",
    )

    with pytest.raises(
        KnowledgeIndexReadinessConflict,
        match="conflicts with durable state",
    ) as invalid_initial:
        await store.publish_index_readiness(
            pending.model_copy(update={"state": KnowledgeIndexState.READY}),
            expected_sequence=None,
            operation_id="invalid-initial-ready",
        )
    assert invalid_initial.value.reason == "initial_state_must_be_pending"

    pending_record = await store.publish_index_readiness(
        pending,
        expected_sequence=None,
        operation_id="readiness-pending-1",
    )
    assert pending_record.sequence == 1
    assert pending_record.identity == identity
    assert pending_record.state is KnowledgeIndexState.PENDING

    replay = await store.publish_index_readiness(
        pending,
        expected_sequence=None,
        operation_id="readiness-pending-1",
    )
    assert replay == pending_record

    with pytest.raises(KnowledgeIndexReadinessConflict) as operation_reuse:
        await store.publish_index_readiness(
            pending.model_copy(update={"attempt_id": "other-attempt"}),
            expected_sequence=pending_record.sequence,
            operation_id="readiness-pending-1",
        )
    assert operation_reuse.value.reason == "operation_reuse"

    ready = await store.publish_index_readiness(
        KnowledgeIndexReadinessUpdate(
            identity=identity,
            state=KnowledgeIndexState.READY,
            attempt_id=pending_record.attempt_id,
        ),
        expected_sequence=pending_record.sequence,
        operation_id="readiness-ready-1",
    )
    assert ready.sequence == 2
    assert ready.state is KnowledgeIndexState.READY

    retry = await store.publish_index_readiness(
        KnowledgeIndexReadinessUpdate(
            identity=identity,
            state=KnowledgeIndexState.PENDING,
            attempt_id="attempt-2",
        ),
        expected_sequence=ready.sequence,
        operation_id="readiness-pending-2",
    )
    assert retry.sequence == 3

    with pytest.raises(KnowledgeIndexReadinessConflict) as stale_worker:
        await store.publish_index_readiness(
            KnowledgeIndexReadinessUpdate(
                identity=identity,
                state=KnowledgeIndexState.FAILED,
                attempt_id="attempt-1",
                failure_code="late_failure",
            ),
            expected_sequence=pending_record.sequence,
            operation_id="readiness-late-failure",
        )
    assert stale_worker.value.reason == "stale_sequence"

    failed = await store.publish_index_readiness(
        KnowledgeIndexReadinessUpdate(
            identity=identity,
            state=KnowledgeIndexState.FAILED,
            attempt_id=retry.attempt_id,
            failure_code="provider_unavailable",
        ),
        expected_sequence=retry.sequence,
        operation_id="readiness-failed-2",
    )
    assert failed.sequence == 4
    assert await store.load_index_readiness(identity) == failed

    first_page = await store.read_index_readiness(limit=2)
    assert [item.sequence for item in first_page.readiness] == [1, 2]
    assert first_page.high_water_sequence == 4
    assert first_page.next_after_sequence == 2
    assert first_page.truncated is True
    second_page = await store.read_index_readiness(
        after_sequence=first_page.next_after_sequence,
        limit=2,
    )
    assert [item.sequence for item in second_page.readiness] == [3, 4]
    assert second_page.next_after_sequence == 4
    assert second_page.truncated is False

    successor = created.model_copy(
        update={
            "revision": 2,
            "text": "A newer canonical revision.",
        }
    )
    await store.append_entry_revision(successor, expected_revision=created.revision)
    assert await store.load_index_readiness(identity) == failed
    with pytest.raises(KnowledgeIndexReadinessConflict) as stale_identity:
        await store.publish_index_readiness(
            KnowledgeIndexReadinessUpdate(
                identity=identity,
                state=KnowledgeIndexState.PENDING,
                attempt_id="attempt-3",
            ),
            expected_sequence=failed.sequence,
            operation_id="readiness-stale-revision",
        )
    assert stale_identity.value.reason == "stale_identity"

    race_entry = await store.create_entry(
        KnowledgeEntry(id="readiness-race", text="Concurrent readiness publication.")
    )
    race_chunk = (await store.read_chunks(race_entry.id))[0]
    race_identity = embedding_identity(
        race_entry,
        chunk=race_chunk,
    )

    async def publish(attempt: str):
        return await store.publish_index_readiness(
            KnowledgeIndexReadinessUpdate(
                identity=race_identity,
                state=KnowledgeIndexState.PENDING,
                attempt_id=attempt,
            ),
            expected_sequence=None,
            operation_id=f"readiness-race-{attempt}",
        )

    race_results = await asyncio.gather(
        publish("attempt-a"),
        publish("attempt-b"),
        return_exceptions=True,
    )
    winners = [result for result in race_results if not isinstance(result, BaseException)]
    conflicts = [
        result for result in race_results if isinstance(result, KnowledgeIndexReadinessConflict)
    ]
    assert len(winners) == 1
    assert len(conflicts) == 1
    assert conflicts[0].reason == "stale_sequence"
