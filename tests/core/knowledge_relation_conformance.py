from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

from cayu.storage import (
    KnowledgeAccessDenied,
    KnowledgeAccessScope,
    KnowledgeChangeKind,
    KnowledgeEntry,
    KnowledgeRelation,
    KnowledgeRelationConflict,
    KnowledgeRelationDirection,
    KnowledgeRelationKind,
    KnowledgeRelationPublicationReceipt,
    KnowledgeRelationQuery,
    KnowledgeRevisionRef,
)

_NOW = datetime(2026, 8, 25, 9, 0, tzinfo=UTC)


def relation_entry(
    entry_id: str,
    *,
    namespace: str = "project:cayu",
    text: str | None = None,
    source_type: str | None = None,
    source_id: str | None = None,
) -> KnowledgeEntry:
    return KnowledgeEntry(
        id=entry_id,
        text=text or f"content for {entry_id}",
        namespace=namespace,
        labels={"project": "cayu"},
        source_type=source_type,
        source_id=source_id,
        created_at=_NOW,
        updated_at=_NOW,
    )


def relation_record(
    relation_id: str,
    *,
    subject: str,
    object_: str,
    kind: KnowledgeRelationKind,
    offset: int = 0,
) -> KnowledgeRelation:
    return KnowledgeRelation(
        id=relation_id,
        subject=KnowledgeRevisionRef(entry_id=subject, revision=1),
        object=KnowledgeRevisionRef(entry_id=object_, revision=1),
        kind=kind,
        created_by="reviewer",
        policy_id="knowledge-lineage-v1",
        created_at=_NOW + timedelta(seconds=offset),
        metadata={"safe_reason": relation_id},
    )


async def assert_knowledge_relation_conformance(store: Any) -> None:
    entries = [
        relation_entry("relation-anchor"),
        relation_entry("relation-old"),
        relation_entry("relation-source"),
        relation_entry("relation-conflict"),
        relation_entry("relation-other"),
    ]
    for entry in entries:
        await store.create_entry(entry)
    baseline = (await store.read_changes(after_sequence=0, limit=100)).high_water_sequence
    await store.initialize_change_consumer(
        "relation-change-consumer",
        baseline_sequence=baseline,
    )

    relations = [
        relation_record(
            "relation-a-supersedes",
            subject="relation-anchor",
            object_="relation-old",
            kind=KnowledgeRelationKind.SUPERSEDES,
        ),
        relation_record(
            "relation-b-derived",
            subject="relation-anchor",
            object_="relation-source",
            kind=KnowledgeRelationKind.DERIVED_FROM,
            offset=1,
        ),
    ]
    receipt = await store.publish_relations(
        list(reversed(relations)),
        operation_id="relation-batch-operation",
    )
    assert type(receipt) is KnowledgeRelationPublicationReceipt
    assert receipt.relation_ids == sorted(relation.id for relation in relations)
    assert receipt.replayed is False

    replay = await store.publish_relations(
        relations,
        operation_id="relation-batch-operation",
    )
    assert replay.replayed is True
    assert replay.committed_at == receipt.committed_at
    assert await store.load_relation_publication_receipt("relation-batch-operation") == receipt

    changed = relations[0].model_copy(update={"metadata": {"safe_reason": "changed"}})
    try:
        await store.publish_relations(
            [changed, relations[1]],
            operation_id="relation-batch-operation",
        )
    except KnowledgeRelationConflict as exc:
        assert exc.reason == "operation_reuse"
    else:  # pragma: no cover - assertion branch
        raise AssertionError("A relation operation identity accepted changed material.")

    for conflicting in (
        relations[0],
        relations[0].model_copy(update={"id": "relation-semantic-duplicate"}),
    ):
        try:
            await store.publish_relations(
                [conflicting],
                operation_id=f"collision-{conflicting.id}",
            )
        except KnowledgeRelationConflict as exc:
            assert exc.reason == "relation_exists"
        else:  # pragma: no cover - assertion branch
            raise AssertionError("Immutable relation occupancy was overwritten.")

    page_1_query = KnowledgeRelationQuery(
        reference=KnowledgeRevisionRef(entry_id="relation-anchor", revision=1),
        direction=KnowledgeRelationDirection.OUTGOING,
        limit=1,
    )
    page_1 = await store.read_relations(page_1_query)
    assert page_1 is not None
    assert [relation.id for relation in page_1.relations] == ["relation-a-supersedes"]
    assert page_1.truncated is True
    assert page_1.next_cursor is not None

    page_2 = await store.read_relations(
        page_1_query.model_copy(update={"cursor": page_1.next_cursor})
    )
    assert page_2 is not None
    assert [relation.id for relation in page_2.relations] == ["relation-b-derived"]
    assert page_2.truncated is False
    assert page_2.next_cursor is None

    old_incoming = await store.read_relations(
        KnowledgeRelationQuery(
            reference=KnowledgeRevisionRef(entry_id="relation-old", revision=1),
            direction=KnowledgeRelationDirection.INCOMING,
        )
    )
    assert old_incoming is not None
    assert [relation.id for relation in old_incoming.relations] == ["relation-a-supersedes"]
    assert (
        await store.read_relations(
            KnowledgeRelationQuery(
                reference=KnowledgeRevisionRef(entry_id="relation-old", revision=1),
                direction=KnowledgeRelationDirection.OUTGOING,
            )
        )
    ).relations == []

    try:
        await store.read_relations(
            page_1_query.model_copy(
                update={
                    "kinds": [KnowledgeRelationKind.DERIVED_FROM],
                    "cursor": page_1.next_cursor,
                }
            )
        )
    except ValueError:
        pass
    else:  # pragma: no cover - assertion branch
        raise AssertionError("A relation cursor crossed a changed kind filter.")

    advanced = entries[0].model_copy(
        update={
            "revision": 2,
            "text": "a later current revision",
            "updated_at": _NOW + timedelta(minutes=1),
        }
    )
    await store.append_entry_revision(advanced, expected_revision=1)
    advanced_object = entries[1].model_copy(
        update={
            "revision": 2,
            "text": "a later predecessor revision",
            "updated_at": _NOW + timedelta(minutes=1),
        }
    )
    await store.append_entry_revision(advanced_object, expected_revision=1)
    historical = await store.read_relations(
        KnowledgeRelationQuery(reference=KnowledgeRevisionRef(entry_id=entries[0].id, revision=1))
    )
    assert historical is not None
    assert {relation.subject.revision for relation in historical.relations} == {1}

    missing = relation_record(
        "relation-missing-endpoint",
        subject="relation-conflict",
        object_="relation-does-not-exist",
        kind=KnowledgeRelationKind.DERIVED_FROM,
    )
    valid = relation_record(
        "relation-would-have-been-written",
        subject="relation-conflict",
        object_="relation-other",
        kind=KnowledgeRelationKind.DERIVED_FROM,
    )
    try:
        await store.publish_relations(
            [valid, missing],
            operation_id="relation-atomic-failure",
        )
    except KnowledgeRelationConflict as exc:
        assert exc.reason == "endpoint_missing"
    else:  # pragma: no cover - assertion branch
        raise AssertionError("A missing endpoint did not reject the complete batch.")
    assert await store.load_relation_publication_receipt("relation-atomic-failure") is None
    untouched = await store.read_relations(
        KnowledgeRelationQuery(
            reference=KnowledgeRevisionRef(entry_id="relation-conflict", revision=1)
        )
    )
    assert untouched is not None and untouched.relations == []

    contradiction_a = relation_record(
        "relation-symmetric-conflict",
        subject="relation-conflict",
        object_="relation-other",
        kind=KnowledgeRelationKind.CONTRADICTS,
        offset=2,
    )
    contradiction_b = contradiction_a.model_copy(
        update={
            "subject": contradiction_a.object,
            "object": contradiction_a.subject,
        }
    )

    async def publish_contradiction(operation_id: str, relation: KnowledgeRelation):
        try:
            return await store.publish_relations([relation], operation_id=operation_id)
        except KnowledgeRelationConflict as exc:
            return exc

    outcomes = await asyncio.gather(
        publish_contradiction("relation-symmetric-a", contradiction_a),
        publish_contradiction("relation-symmetric-b", contradiction_b),
    )
    assert sum(type(outcome) is KnowledgeRelationPublicationReceipt for outcome in outcomes) == 1
    assert sum(type(outcome) is KnowledgeRelationConflict for outcome in outcomes) == 1
    symmetric = await store.read_relations(
        KnowledgeRelationQuery(
            reference=KnowledgeRevisionRef(entry_id="relation-other", revision=1),
            direction=KnowledgeRelationDirection.INCOMING,
            kinds=[KnowledgeRelationKind.CONTRADICTS],
        )
    )
    assert symmetric is not None
    assert [relation.id for relation in symmetric.relations] == ["relation-symmetric-conflict"]
    assert symmetric.relations[0].subject.entry_id < symmetric.relations[0].object.entry_id

    changes = await store.read_changes(after_sequence=0, limit=100)
    relation_changes = [
        change
        for change in changes.changes
        if change.kind is KnowledgeChangeKind.RELATION_PUBLISHED
    ]
    assert [change.relation_id for change in relation_changes] == [
        "relation-a-supersedes",
        "relation-b-derived",
        "relation-symmetric-conflict",
    ]
    assert all(change.operation_id is not None for change in relation_changes)

    first_claim = await store.claim_change(
        "relation-change-consumer",
        "relation-worker",
    )
    assert first_claim is not None
    assert first_claim.change.relation_id == "relation-a-supersedes"
    released = await store.release_change(first_claim)
    assert released.pending_change_sequence is None
    retried = await store.claim_change(
        "relation-change-consumer",
        "relation-worker",
    )
    assert retried is not None
    assert retried.change == first_claim.change
    assert retried.attempt == first_claim.attempt + 1
    acknowledged = await store.acknowledge_change(retried)
    assert acknowledged.cursor_sequence == retried.change.sequence

    deleted = await store.delete_entry(
        advanced.id,
        expected_revision=advanced.revision,
        hard=True,
    )
    assert deleted == advanced
    after_delete = await store.read_relations(
        KnowledgeRelationQuery(
            reference=KnowledgeRevisionRef(entry_id=advanced_object.id, revision=1)
        )
    )
    assert after_delete is not None
    assert after_delete.relations == []
    post_delete_replay = await store.publish_relations(
        relations,
        operation_id="relation-batch-operation",
    )
    assert post_delete_replay.replayed is True
    assert (
        await store.read_relations(
            KnowledgeRelationQuery(
                reference=KnowledgeRevisionRef(
                    entry_id=advanced_object.id,
                    revision=1,
                )
            )
        )
    ).relations == []

    recreated = relation_entry(advanced.id, text="recreated logical entry")
    await store.create_entry(recreated)
    reused_identity = relation_record(
        relations[0].id,
        subject=recreated.id,
        object_="relation-other",
        kind=KnowledgeRelationKind.DERIVED_FROM,
    )
    try:
        await store.publish_relations(
            [reused_identity],
            operation_id="relation-reused-after-delete",
        )
    except KnowledgeRelationConflict as exc:
        assert exc.reason == "relation_exists"
    else:  # pragma: no cover - assertion branch
        raise AssertionError("Hard deletion released an immutable relation identity.")


async def assert_knowledge_relation_scope_conformance(store: Any) -> None:
    privileged = KnowledgeAccessScope.privileged()
    alpha_scope = KnowledgeAccessScope.for_namespace("alpha")
    alpha_a = relation_entry("scope-alpha-a", namespace="alpha")
    alpha_b = relation_entry("scope-alpha-b", namespace="alpha")
    beta = relation_entry("scope-beta", namespace="beta")
    source_match = relation_entry(
        "scope-source-match",
        namespace="alpha",
        source_type="document",
        source_id="source-match",
    )
    for entry in (alpha_a, alpha_b, beta, source_match):
        await store.create_entry(entry, access_scope=privileged)
    baseline = (
        await store.read_changes(
            after_sequence=0,
            limit=100,
            access_scope=alpha_scope,
        )
    ).high_water_sequence
    await store.initialize_change_consumer(
        "scoped-relation-consumer",
        baseline_sequence=baseline,
        access_scope=alpha_scope,
    )

    visible = relation_record(
        "scope-visible-relation",
        subject=alpha_a.id,
        object_=alpha_b.id,
        kind=KnowledgeRelationKind.DERIVED_FROM,
    )
    hidden = relation_record(
        "scope-hidden-relation",
        subject=alpha_a.id,
        object_=beta.id,
        kind=KnowledgeRelationKind.CONTRADICTS,
        offset=1,
    )
    source_filtered = relation_record(
        "scope-source-filtered-relation",
        subject=source_match.id,
        object_=alpha_b.id,
        kind=KnowledgeRelationKind.DERIVED_FROM,
        offset=2,
    )
    await store.publish_relations(
        [visible],
        operation_id="scope-visible-operation",
        access_scope=privileged,
    )
    await store.publish_relations(
        [hidden],
        operation_id="scope-hidden-operation",
        access_scope=privileged,
    )
    await store.publish_relations(
        [source_filtered],
        operation_id="scope-source-filtered-operation",
        access_scope=privileged,
    )

    result = await store.read_relations(
        KnowledgeRelationQuery(reference=KnowledgeRevisionRef(entry_id=alpha_a.id, revision=1)),
        access_scope=alpha_scope,
    )
    assert result is not None
    assert [relation.id for relation in result.relations] == [visible.id]
    assert (
        await store.load_relation_publication_receipt(
            "scope-hidden-operation",
            access_scope=alpha_scope,
        )
        is None
    )
    try:
        await store.publish_relations(
            [hidden],
            operation_id="scope-hidden-operation",
            access_scope=alpha_scope,
        )
    except KnowledgeAccessDenied:
        pass
    else:  # pragma: no cover - assertion branch
        raise AssertionError("A relation receipt replay exposed an inaccessible endpoint.")

    changes = await store.read_changes(
        after_sequence=0,
        limit=100,
        access_scope=alpha_scope,
    )
    relation_change_ids = {
        change.relation_id
        for change in changes.changes
        if change.kind is KnowledgeChangeKind.RELATION_PUBLISHED
    }
    assert visible.id in relation_change_ids
    assert hidden.id not in relation_change_ids
    source_scope = KnowledgeAccessScope.for_namespace(
        "alpha",
        allowed_source_types=["document"],
    )
    source_changes = await store.read_changes(
        after_sequence=0,
        limit=100,
        access_scope=source_scope,
    )
    assert all(change.relation_id != source_filtered.id for change in source_changes.changes)
    claim = await store.claim_change(
        "scoped-relation-consumer",
        "scope-worker",
        access_scope=alpha_scope,
    )
    assert claim is not None
    assert claim.change.relation_id == visible.id
    await store.acknowledge_change(claim, access_scope=alpha_scope)
    source_claim = await store.claim_change(
        "scoped-relation-consumer",
        "scope-worker",
        access_scope=alpha_scope,
    )
    assert source_claim is not None
    assert source_claim.change.relation_id == source_filtered.id
    await store.acknowledge_change(source_claim, access_scope=alpha_scope)
    assert (
        await store.claim_change(
            "scoped-relation-consumer",
            "scope-worker",
            access_scope=alpha_scope,
        )
        is None
    )

    historical_subject = relation_entry(
        "scope-historical-subject",
        namespace="alpha",
    ).model_copy(update={"labels": {"era": "old"}})
    historical_object = relation_entry(
        "scope-historical-object",
        namespace="alpha",
    ).model_copy(update={"labels": {"era": "old"}})
    for entry in (historical_subject, historical_object):
        await store.create_entry(entry, access_scope=privileged)
        await store.append_entry_revision(
            entry.model_copy(
                update={
                    "revision": 2,
                    "labels": {"era": "new"},
                    "updated_at": _NOW + timedelta(minutes=1),
                }
            ),
            expected_revision=1,
            access_scope=privileged,
        )
    old_scope = KnowledgeAccessScope.for_namespace(
        "alpha",
        required_labels={"era": "old"},
    )
    historical_baseline = (
        await store.read_changes(
            after_sequence=0,
            limit=100,
            access_scope=old_scope,
        )
    ).high_water_sequence
    await store.initialize_change_consumer(
        "historical-relation-consumer",
        baseline_sequence=historical_baseline,
        access_scope=old_scope,
    )
    historical = relation_record(
        "scope-historical-relation",
        subject=historical_subject.id,
        object_=historical_object.id,
        kind=KnowledgeRelationKind.DERIVED_FROM,
        offset=3,
    )
    await store.publish_relations(
        [historical],
        operation_id="scope-historical-operation",
        access_scope=privileged,
    )

    privileged_result = await store.read_relations(
        KnowledgeRelationQuery(reference=historical.subject),
        access_scope=privileged,
    )
    assert privileged_result is not None
    assert [relation.id for relation in privileged_result.relations] == [historical.id]
    privileged_receipt = await store.load_relation_publication_receipt(
        "scope-historical-operation",
        access_scope=privileged,
    )
    assert privileged_receipt is not None
    assert privileged_receipt.relation_ids == [historical.id]
    assert (
        await store.read_relations(
            KnowledgeRelationQuery(reference=historical.subject),
            access_scope=old_scope,
        )
        is None
    )
    assert (
        await store.load_relation_publication_receipt(
            "scope-historical-operation",
            access_scope=old_scope,
        )
        is None
    )
    try:
        await store.publish_relations(
            [historical],
            operation_id="scope-historical-operation",
            access_scope=old_scope,
        )
    except KnowledgeAccessDenied:
        pass
    else:  # pragma: no cover - assertion branch
        raise AssertionError("A historical relation receipt ignored current authority.")
    historical_changes = await store.read_changes(
        after_sequence=historical_baseline,
        limit=100,
        access_scope=old_scope,
    )
    assert all(change.relation_id != historical.id for change in historical_changes.changes)
    assert (
        await store.claim_change(
            "historical-relation-consumer",
            "scope-worker",
            access_scope=old_scope,
        )
        is None
    )
