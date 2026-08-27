from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import NoReturn

import pytest
from pydantic import ValidationError

import cayu
from cayu.embeddings import (
    TextEmbedding,
    TextEmbeddingProvider,
    TextEmbeddingRequest,
    TextEmbeddingResult,
)
from cayu.storage import (
    MAX_KNOWLEDGE_RELATION_BATCH,
    MAX_KNOWLEDGE_RELATION_BYTES,
    InMemoryEmbeddingKnowledgeStore,
    InMemoryKnowledgeStore,
    KnowledgeChange,
    KnowledgeChangeKind,
    KnowledgeEntry,
    KnowledgeLineageCurrentness,
    KnowledgeLineageLink,
    KnowledgeLineageQuery,
    KnowledgeLineageResult,
    KnowledgeLineageRole,
    KnowledgeRelation,
    KnowledgeRelationDirection,
    KnowledgeRelationKind,
    KnowledgeRelationPublicationReceipt,
    KnowledgeRelationQuery,
    KnowledgeRelationResult,
    KnowledgeRevisionRef,
    KnowledgeStatus,
    prepare_knowledge_relations,
)

_NOW = datetime(2026, 8, 25, 9, 0, tzinfo=UTC)


class _RecordingEmbeddingProvider(TextEmbeddingProvider):
    name = "relation-recording-test"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
        self.calls.append(list(request.texts))
        return TextEmbeddingResult(
            model=request.model,
            embeddings=[
                TextEmbedding(index=index, vector=[1.0, 0.0])
                for index, _ in enumerate(request.texts)
            ],
        )


class _RejectFullRelationScan(dict[str, KnowledgeRelation]):
    def values(self) -> NoReturn:
        raise AssertionError("Relation reads and deletes must use the endpoint index.")


def _relation(
    relation_id: str = "relation-id",
    *,
    subject: str = "subject-entry",
    object_: str = "object-entry",
    kind: KnowledgeRelationKind = KnowledgeRelationKind.DERIVED_FROM,
    metadata: dict | None = None,
) -> KnowledgeRelation:
    return KnowledgeRelation(
        id=relation_id,
        subject=KnowledgeRevisionRef(entry_id=subject, revision=1),
        object=KnowledgeRevisionRef(entry_id=object_, revision=1),
        kind=kind,
        created_by="reviewer",
        policy_id="lineage-v1",
        created_at=_NOW,
        metadata={} if metadata is None else metadata,
    )


def test_knowledge_relation_public_vocabulary_is_closed_and_exported() -> None:
    assert [kind.value for kind in KnowledgeRelationKind] == [
        "supersedes",
        "derived_from",
        "contradicts",
    ]
    assert [role.value for role in KnowledgeLineageRole] == [
        "supersedes",
        "superseded_by",
        "derived_from",
        "derivation_source_for",
        "contradicts",
    ]
    assert [value.value for value in KnowledgeLineageCurrentness] == ["current", "stale"]
    for name in (
        "KnowledgeRevisionRef",
        "KnowledgeRelationKind",
        "KnowledgeRelation",
        "KnowledgeRelationPublicationReceipt",
        "KnowledgeRelationQuery",
        "KnowledgeRelationResult",
        "KnowledgeLineageCurrentness",
        "KnowledgeLineageLink",
        "KnowledgeLineageQuery",
        "KnowledgeLineageResult",
        "KnowledgeLineageRole",
        "prepare_knowledge_relations",
    ):
        assert name in cayu.__all__
        assert getattr(cayu, name) is not None


def test_knowledge_relation_models_reject_ambiguous_or_unbounded_material() -> None:
    with pytest.raises(ValidationError, match="different logical entries"):
        _relation(subject="one-entry", object_="one-entry")
    with pytest.raises(ValidationError):
        KnowledgeRelation.model_validate(
            {
                **_relation().model_dump(),
                "kind": "application_defined_edge",
            }
        )

    lineage_query = KnowledgeLineageQuery(
        reference=KnowledgeRevisionRef(entry_id="subject-entry", revision=1)
    )
    with pytest.raises(ValidationError, match="currentness"):
        KnowledgeLineageResult(
            query=lineage_query,
            reference_current=lineage_query.reference,
            reference_status=KnowledgeStatus.ACTIVE,
            links=[
                KnowledgeLineageLink(
                    relation_id="relation-id",
                    kind=KnowledgeRelationKind.DERIVED_FROM,
                    role=KnowledgeLineageRole.DERIVED_FROM,
                    counterpart=KnowledgeRevisionRef(entry_id="object-entry", revision=1),
                    counterpart_current=KnowledgeRevisionRef(
                        entry_id="object-entry",
                        revision=1,
                    ),
                    counterpart_status=KnowledgeStatus.ACTIVE,
                    currentness=KnowledgeLineageCurrentness.STALE,
                    created_at=_NOW,
                )
            ],
        )
    with pytest.raises(ValidationError, match="metadata.*budget"):
        _relation(metadata={"payload": "x" * MAX_KNOWLEDGE_RELATION_BYTES})
    with pytest.raises(ValidationError, match="timezone-aware"):
        KnowledgeRelation.model_validate(
            {
                **_relation().model_dump(),
                "created_at": datetime(2026, 8, 25, 9, 0),
            }
        )


def test_knowledge_relation_models_defensively_copy_nested_inputs() -> None:
    metadata = {"review": {"reasons": ["source matched"]}}
    relation = _relation(metadata=metadata)
    metadata["review"]["reasons"].append("mutated")
    assert relation.metadata == {"review": {"reasons": ["source matched"]}}

    kinds = [KnowledgeRelationKind.SUPERSEDES]
    query = KnowledgeRelationQuery(
        reference=relation.subject,
        kinds=kinds,
    )
    kinds.append(KnowledgeRelationKind.CONTRADICTS)
    assert query.kinds == [KnowledgeRelationKind.SUPERSEDES]

    default_lineage_query = KnowledgeLineageQuery(reference=relation.subject)
    explicit_lineage_query = KnowledgeLineageQuery(
        reference=relation.subject,
        currentnesses=list(KnowledgeLineageCurrentness),
        counterpart_statuses=list(KnowledgeStatus),
    )
    assert default_lineage_query == explicit_lineage_query

    relation_ids = ["relation-a"]
    receipt = KnowledgeRelationPublicationReceipt(
        operation_id="operation-a",
        relation_ids=relation_ids,
        request_sha256="a" * 64,
        committed_at=_NOW,
    )
    relation_ids.append("relation-b")
    assert receipt.relation_ids == ["relation-a"]


def test_knowledge_relation_preparation_is_canonical_and_stably_fingerprinted() -> None:
    directional = _relation(
        "directional",
        subject="z-entry",
        object_="a-entry",
        kind=KnowledgeRelationKind.SUPERSEDES,
    )
    contradiction = _relation(
        "symmetric",
        subject="z-entry",
        object_="a-entry",
        kind=KnowledgeRelationKind.CONTRADICTS,
    )
    operation, prepared, fingerprint = prepare_knowledge_relations(
        [contradiction, directional],
        operation_id="operation-a",
    )
    _, reversed_prepared, reversed_fingerprint = prepare_knowledge_relations(
        [directional, contradiction],
        operation_id="operation-a",
    )

    assert operation == "operation-a"
    assert prepared == reversed_prepared
    assert fingerprint == reversed_fingerprint
    assert prepared[0].id == "directional"
    assert prepared[0].subject.entry_id == "z-entry"
    assert prepared[1].subject.entry_id == "a-entry"
    assert prepared[1].object.entry_id == "z-entry"
    assert KnowledgeRelation.model_validate_json(contradiction.model_dump_json()) == contradiction
    assert KnowledgeRelationQuery.model_validate_json(
        KnowledgeRelationQuery(reference=contradiction.subject).model_dump_json()
    ) == KnowledgeRelationQuery(reference=contradiction.subject)


def test_knowledge_relation_preparation_rejects_duplicate_batches() -> None:
    relation = _relation()
    with pytest.raises(ValueError, match="duplicate identities"):
        prepare_knowledge_relations(
            [relation, relation],
            operation_id="duplicate-id",
        )
    with pytest.raises(ValueError, match="semantic relation"):
        prepare_knowledge_relations(
            [relation, relation.model_copy(update={"id": "other-id"})],
            operation_id="duplicate-semantic",
        )
    with pytest.raises(ValueError, match="between 1"):
        prepare_knowledge_relations([], operation_id="empty")
    with pytest.raises(ValueError, match=str(MAX_KNOWLEDGE_RELATION_BATCH)):
        prepare_knowledge_relations(
            [
                _relation(
                    f"relation-{index}",
                    subject=f"subject-{index}",
                    object_=f"object-{index}",
                )
                for index in range(MAX_KNOWLEDGE_RELATION_BATCH + 1)
            ],
            operation_id="oversized",
        )


def test_relation_change_requires_relation_identity_and_never_carries_payload_text() -> None:
    base = {
        "id": "change-a",
        "sequence": 1,
        "entry_id": "subject-entry",
        "entry_revision": 1,
        "committed_at": _NOW,
    }
    with pytest.raises(ValidationError, match="require.*relation_id"):
        KnowledgeChange(kind=KnowledgeChangeKind.RELATION_PUBLISHED, **base)
    with pytest.raises(ValidationError, match="Only relation publication"):
        KnowledgeChange(
            kind=KnowledgeChangeKind.CREATED,
            relation_id="relation-a",
            **base,
        )
    change = KnowledgeChange(
        kind=KnowledgeChangeKind.RELATION_PUBLISHED,
        relation_id="relation-a",
        operation_id="operation-a",
        **base,
    )
    assert "text" not in change.model_dump()
    assert "metadata" not in change.model_dump()


def test_knowledge_relation_result_revalidates_page_truth() -> None:
    relation = _relation()
    with pytest.raises(ValidationError, match="match the result query"):
        KnowledgeRelationResult(
            query=KnowledgeRelationQuery(
                reference=relation.object,
                direction=KnowledgeRelationDirection.OUTGOING,
            ),
            relations=[relation],
        )
    with pytest.raises(ValidationError, match="identities must be unique"):
        KnowledgeRelationResult(
            query=KnowledgeRelationQuery(reference=relation.subject),
            relations=[relation, relation],
        )

    large_a = _relation("large-a", metadata={"payload": "x" * 3_900})
    large_b = _relation("large-b", metadata={"payload": "y" * 3_900})
    with pytest.raises(ValidationError, match="max_bytes"):
        KnowledgeRelationResult(
            query=KnowledgeRelationQuery(
                reference=large_a.subject,
                max_bytes=MAX_KNOWLEDGE_RELATION_BYTES,
            ),
            relations=[large_a, large_b],
        )


def test_in_memory_relation_publication_stages_the_complete_batch(monkeypatch) -> None:
    async def run() -> None:
        store = InMemoryKnowledgeStore(access_scope=cayu.KnowledgeAccessScope.privileged())
        for entry_id in ("stage-a", "stage-b", "stage-c"):
            await store.create_entry(KnowledgeEntry(id=entry_id, text=entry_id))
        relations = [
            _relation("stage-r1", subject="stage-a", object_="stage-b"),
            _relation("stage-r2", subject="stage-a", object_="stage-c"),
        ]
        next_sequence = store._next_change_sequence
        original = store._prepare_relation_change
        calls = 0

        def fail_second(relation, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected relation preparation failure")
            return original(relation, **kwargs)

        monkeypatch.setattr(store, "_prepare_relation_change", fail_second)
        with pytest.raises(RuntimeError, match="injected relation preparation failure"):
            await store.publish_relations(relations, operation_id="staged-operation")
        assert store._next_change_sequence == next_sequence
        assert await store.load_relation_publication_receipt("staged-operation") is None
        result = await store.read_relations(KnowledgeRelationQuery(reference=relations[0].subject))
        assert result is not None
        assert result.relations == []
        assert all(change.relation_id is None for change in (await store.read_changes()).changes)

    asyncio.run(run())


def test_in_memory_relation_endpoint_index_serves_reads_and_deletes() -> None:
    async def run() -> None:
        store = InMemoryKnowledgeStore(access_scope=cayu.KnowledgeAccessScope.privileged())
        for entry_id in ("anchor", "target", "unrelated-a", "unrelated-b"):
            await store.create_entry(KnowledgeEntry(id=entry_id, text=entry_id))
        anchor_relation = _relation("anchor-relation", subject="anchor", object_="target")
        unrelated_relation = _relation(
            "unrelated-relation",
            subject="unrelated-a",
            object_="unrelated-b",
        )
        await store.publish_relations(
            [anchor_relation, unrelated_relation],
            operation_id="indexed-relations",
        )

        assert store._relation_ids_by_endpoint == {
            ("anchor", 1): {"anchor-relation"},
            ("target", 1): {"anchor-relation"},
            ("unrelated-a", 1): {"unrelated-relation"},
            ("unrelated-b", 1): {"unrelated-relation"},
        }
        store._relations = _RejectFullRelationScan(store._relations)

        result = await store.read_relations(
            KnowledgeRelationQuery(reference=KnowledgeRevisionRef(entry_id="anchor", revision=1))
        )
        assert result is not None
        assert result.relations == [anchor_relation]

        deleted = await store.delete_entry("unrelated-a", expected_revision=1, hard=True)
        assert deleted is not None
        assert "unrelated-relation" not in store._relations
        assert ("unrelated-a", 1) not in store._relation_ids_by_endpoint
        assert ("unrelated-b", 1) not in store._relation_ids_by_endpoint
        assert store._relation_ids_by_endpoint == {
            ("anchor", 1): {"anchor-relation"},
            ("target", 1): {"anchor-relation"},
        }

    asyncio.run(run())


def test_embedding_worker_acknowledges_relation_changes_without_index_work() -> None:
    async def run() -> None:
        provider = _RecordingEmbeddingProvider()
        store = InMemoryEmbeddingKnowledgeStore(
            embedding_provider=provider,
            embedding_model="relation-test",
            embedding_dimensions=2,
            access_scope=cayu.KnowledgeAccessScope.privileged(),
        )
        for entry_id in ("embedding-subject", "embedding-object"):
            await store.create_entry(KnowledgeEntry(id=entry_id, text=entry_id))
        high_water = (await store.read_changes()).high_water_sequence
        await store.initialize_change_consumer(
            "relation-embedding-consumer",
            baseline_sequence=high_water,
        )
        relation = _relation(
            "embedding-relation",
            subject="embedding-subject",
            object_="embedding-object",
        )
        await store.publish_relations([relation], operation_id="embedding-relation-operation")

        result = await store.process_embedding_changes(
            "relation-embedding-consumer",
            "worker",
            limit=1,
        )

        assert result.claimed_changes == 1
        assert result.acknowledged_changes == 1
        assert result.processed_records == 0
        assert result.indexed_records == 0
        assert result.removed_records == 0
        assert result.failed_records == 0
        assert provider.calls == []

    asyncio.run(run())
