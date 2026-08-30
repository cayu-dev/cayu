from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from tests.core.knowledge_access_scope_conformance import (
    assert_knowledge_access_scope_conformance,
)
from tests.core.knowledge_index_readiness_conformance import (
    assert_index_readiness_conformance,
)
from tests.core.knowledge_maintenance_conformance import (
    _create_proposal_entries,
    maintenance_decision,
    maintenance_proposal,
)
from tests.core.knowledge_none_terms_conformance import (
    assert_entry_wide_none_terms_conformance,
    assert_entry_wide_none_terms_precede_chunk_pagination,
)
from tests.core.knowledge_phrase_conformance import (
    assert_token_exact_phrase_search_conformance,
)
from tests.core.knowledge_publication_conformance import (
    assert_concurrent_publication_conformance,
    assert_failed_publication_left_no_state,
    assert_owned_publication_conformance,
    assert_stale_operation_cannot_replace_newer_publication,
    publication_material,
)
from tests.core.knowledge_store_conformance import (
    verify_embedding_space_isolation,
    verify_projection_readiness,
)
from tests.core.test_knowledge_maintenance_persistence import (
    _REVIEW_SCOPE,
    _accepted,
    _assert_publication_conformance,
    _decision,
    _publisher,
)

from cayu.embeddings import (
    TextEmbedding,
    TextEmbeddingProvider,
    TextEmbeddingRequest,
    TextEmbeddingResult,
)
from cayu.evals.knowledge_maintenance import (
    KnowledgeMaintenanceEvaluationResult,
    load_knowledge_maintenance_evaluation_corpus,
    run_knowledge_maintenance_evaluation,
)
from cayu.knowledge_maintenance_persistence import (
    KnowledgeMaintenanceProposalPublicationConflict,
)
from cayu.storage import (
    MAX_KNOWLEDGE_CHUNK_ID_BYTES,
    MAX_KNOWLEDGE_CHUNK_INDEX,
    MAX_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT,
    KnowledgeAccessScope,
    KnowledgeChunk,
    KnowledgeEmbeddingProjection,
    KnowledgeEmbeddingProjectionConflict,
    KnowledgeEntry,
    KnowledgeEntryReadLimitExceeded,
    KnowledgeEvidence,
    KnowledgeIndexReadinessUpdate,
    KnowledgeIndexState,
    KnowledgeListGroup,
    KnowledgeListQuery,
    KnowledgeMaintenanceDecisionKind,
    KnowledgeQuery,
    KnowledgeRelation,
    KnowledgeRelationKind,
    KnowledgeRelationQuery,
    KnowledgeRevisionRef,
    KnowledgeRevisionResetRequired,
    KnowledgeSearchMode,
    KnowledgeStatus,
    KnowledgeVisibility,
    knowledge_chunk_embedding_identity,
)
from cayu.storage import migrations as schema_migrations
from cayu.storage.memory import (
    _knowledge_access_snapshot,
    _knowledge_access_snapshot_json,
    _knowledge_chunk_content_hash,
    _knowledge_publication_v1_request_sha256,
)
from cayu.storage.migrations import LATEST_REVISION, MIN_SUPPORTED_REVISION, SchemaMode
from cayu.work_context import agent_recall_facet_aspect

pytestmark = pytest.mark.usefixtures("postgres_dsn")

_ACCESS_SCOPE = KnowledgeAccessScope.privileged()
_MAINTENANCE_EVALUATION_CORPUS = (
    Path(__file__).resolve().parents[2] / "benchmarks/memory/knowledge-maintenance-corpus-v1.json"
)
_MAINTENANCE_EVALUATION_RESULTS = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/memory/knowledge-maintenance-evaluation-results-v1.json"
)


def _maintenance_result_without_backend_latency(
    result: KnowledgeMaintenanceEvaluationResult,
) -> dict:
    payload = result.model_dump(mode="json")
    payload.pop("backend")
    payload["metrics"].pop("latency_p50_ms")
    payload["metrics"].pop("latency_p95_ms")
    for case in payload["cases"]:
        case.pop("latency_ms")
    return payload


def test_postgres_knowledge_maintenance_evaluation_parity(postgres_dsn: str) -> None:
    async def run() -> None:
        from cayu import PostgresKnowledgeStore

        await _drop_all(postgres_dsn)
        store = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            result = await run_knowledge_maintenance_evaluation(
                load_knowledge_maintenance_evaluation_corpus(_MAINTENANCE_EVALUATION_CORPUS),
                store,
                backend="postgres",
            )
        finally:
            await store.close()
        assert result.metrics.routing_precision == 1.0
        assert result.metrics.routing_recall == 1.0
        assert result.metrics.information_retention == 1.0
        assert result.metrics.evidence_retention == 1.0
        assert result.metrics.unsafe_acceptance_rate == 0.0
        assert result.metrics.lifecycle_correctness == 1.0
        assert result.metrics.lineage_correctness == 1.0
        assert result.metrics.model_call_count == 0
        checked_payload = json.loads(_MAINTENANCE_EVALUATION_RESULTS.read_text(encoding="utf-8"))
        frozen = KnowledgeMaintenanceEvaluationResult.model_validate(checked_payload["results"][0])
        assert _maintenance_result_without_backend_latency(
            result
        ) == _maintenance_result_without_backend_latency(frozen)

    try:
        asyncio.run(run())
    finally:
        asyncio.run(_drop_all(postgres_dsn))


def test_postgres_accepted_plan_publication_and_review_handoff(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        from cayu import PostgresKnowledgeStore

        await _drop_all(postgres_dsn)
        store = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await _assert_publication_conformance(store, "postgres")
        finally:
            await store.close()

    try:
        asyncio.run(run())
    finally:
        asyncio.run(_drop_all(postgres_dsn))


def test_postgres_publication_load_validates_the_decision_record(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        from cayu import PostgresKnowledgeStore

        await _drop_all(postgres_dsn)
        store = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            request, routing, planning = await _accepted(store, "malformed-decision")
            publisher = _publisher(store)
            publication = await publisher.publish(request, routing, planning)
            decision = _decision(
                publication.proposal,
                kind=KnowledgeMaintenanceDecisionKind.REJECT,
                suffix="postgres-malformed-decision",
            )
            await store.apply_maintenance_decision(
                publication.proposal,
                decision,
                access_scope=_REVIEW_SCOPE,
            )
            async with store._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    "UPDATE cayu_knowledge_maintenance_decisions "
                    "SET decision = '{}'::jsonb WHERE operation_id = %s",
                    (decision.operation_id,),
                )
                await conn.commit()
            with pytest.raises(KnowledgeMaintenanceProposalPublicationConflict) as error:
                await publisher.load(publication.proposal.id)
            assert error.value.reason == "malformed_receipt"
        finally:
            await store.close()

    try:
        asyncio.run(run())
    finally:
        asyncio.run(_drop_all(postgres_dsn))


def test_postgres_bounded_entry_read_refuses_before_loading_content(
    postgres_dsn: str,
    monkeypatch,
) -> None:
    async def run() -> None:
        from cayu import PostgresKnowledgeStore

        await _drop_all(postgres_dsn)
        store = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
            access_scope=_ACCESS_SCOPE,
        )
        try:
            entry = KnowledgeEntry(id="oversized-read", text="x" * 1_000_000)
            await store.create_entry(entry)

            async def fail_load(*_args, **_kwargs):
                raise AssertionError("oversized entry content was loaded")

            monkeypatch.setattr(store, "_load_entry_in_scope", fail_load)
            with pytest.raises(KnowledgeEntryReadLimitExceeded):
                await store.get_entry(entry.id, max_bytes=256)
        finally:
            await store.close()

    try:
        asyncio.run(run())
    finally:
        asyncio.run(_drop_all(postgres_dsn))


def test_postgres_bounded_entry_read_reuses_one_authorization_time(
    postgres_dsn: str,
    monkeypatch,
) -> None:
    async def run() -> None:
        from cayu import PostgresKnowledgeStore

        await _drop_all(postgres_dsn)
        store = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
            access_scope=_ACCESS_SCOPE,
        )
        try:
            entry = KnowledgeEntry(id="bounded-read-time", text="bounded")
            await store.create_entry(entry)
            descriptor_times: list[datetime] = []
            hydration_times: list[datetime] = []
            load_descriptor = store._load_entry_payload_bytes_in_scope
            load_entry = store._load_entry_in_scope

            async def tracked_descriptor(*args, access_now: datetime, **kwargs):
                descriptor_times.append(access_now)
                return await load_descriptor(*args, access_now=access_now, **kwargs)

            async def tracked_entry(*args, access_now: datetime | None = None, **kwargs):
                assert access_now is not None
                hydration_times.append(access_now)
                return await load_entry(*args, access_now=access_now, **kwargs)

            monkeypatch.setattr(
                store,
                "_load_entry_payload_bytes_in_scope",
                tracked_descriptor,
            )
            monkeypatch.setattr(store, "_load_entry_in_scope", tracked_entry)

            loaded = await store.get_entry(entry.id, max_bytes=10_000)
            assert loaded == entry
            assert descriptor_times == hydration_times
            assert len(descriptor_times) == 1
        finally:
            await store.close()

    try:
        asyncio.run(run())
    finally:
        asyncio.run(_drop_all(postgres_dsn))


_TABLES = (
    "cayu_agent_recall_subscription_wake_states",
    "cayu_agent_recall_subscription_wake_releases",
    "cayu_agent_recall_subscription_wake_claims",
    "cayu_agent_recall_subscription_evaluations",
    "cayu_agent_recall_subscription_states",
    "cayu_agent_recall_subscription_releases",
    "cayu_agent_recall_subscription_claims",
    "cayu_agent_recall_subscription_publications",
    "cayu_agent_recall_subscription_heads",
    "cayu_agent_recall_subscription_revisions",
    "cayu_agent_recall_delivery_acknowledgements",
    "cayu_agent_recall_delivery_releases",
    "cayu_agent_recall_delivery_claims",
    "cayu_agent_recall_delivery_states",
    "cayu_agent_recall_deliveries",
    "cayu_agent_recall_checkpoint_heads",
    "cayu_agent_recall_checkpoints",
    "cayu_agent_work_context_publications",
    "cayu_agent_work_context_heads",
    "cayu_agent_work_context_revisions",
    "cayu_knowledge_embeddings",
    "cayu_knowledge_index_readiness_current",
    "cayu_knowledge_index_readiness_events",
    "cayu_task_terminalization_receipts",
    "cayu_completion_decision_application_receipts",
    "cayu_completion_decisions",
    "cayu_completion_verification_claims",
    "cayu_completion_verifier_profiles",
    "cayu_completion_proposals",
    "cayu_work_attempt_execution_claims",
    "cayu_work_attempt_admissions",
    "cayu_work_attempts",
    "cayu_task_session_execution_authority",
    "cayu_work_contracts",
    "cayu_recall_item_exposures",
    "cayu_context_exposures",
    "cayu_recall_receipts",
    "cayu_knowledge_maintenance_proposals",
    "cayu_knowledge_maintenance_decisions",
    "cayu_knowledge_relation_publication_receipts",
    "cayu_knowledge_relations",
    "cayu_knowledge_change_acknowledgements",
    "cayu_knowledge_change_consumers",
    "cayu_knowledge_change_labels",
    "cayu_knowledge_change_audiences",
    "cayu_knowledge_changes",
    "cayu_knowledge_evidence",
    "cayu_knowledge_publication_receipts",
    "cayu_knowledge_labels",
    "cayu_knowledge_aspects",
    "cayu_knowledge_impact_targets",
    "cayu_knowledge_chunks",
    "cayu_knowledge_revisions",
    "cayu_knowledge_entries",
    "cayu_event_watcher_state",
    "cayu_budget_reservation_identities",
    "cayu_events",
    "cayu_session_labels",
    "cayu_targeted_tool_grant_uses",
    "cayu_targeted_tool_grants",
    "cayu_public_authority_aliases",
    "cayu_public_authority_alias_keys",
    "cayu_transcript_search_configuration",
    "cayu_transcript_messages",
    "cayu_session_message_queue",
    "cayu_persisted_event_side_effects",
    "cayu_mcp_manifest_baselines",
    "cayu_checkpoints",
    "cayu_session_operations",
    "cayu_local_execution_attempts",
    "cayu_tasks",
    "cayu_sessions",
    "cayu_eval_baseline_mutations",
    "cayu_eval_baselines",
    "cayu_eval_result_records",
    "cayu_eval_results",
    "cayu_eval_runs",
    "cayu_eval_cases",
    "cayu_eval_suites",
    "cayu_eval_corpora",
    "cayu_schema_migrations",
)


async def _initialize_historical_schema(
    postgres_dsn: str,
    *,
    through_revision: int,
) -> None:
    """Create an intentionally old schema without relaxing current store startup."""
    from cayu import PostgresSessionStore

    revisions = schema_migrations.REVISIONS
    schema_migrations.REVISIONS = tuple(
        revision for revision in revisions if revision.revision <= through_revision
    )
    historical_store = PostgresSessionStore(
        postgres_dsn,
        min_size=1,
        max_size=2,
        schema_mode=SchemaMode.MIGRATE,
    )
    # These migration tests emulate an older binary. The current session store
    # correctly requires revision 46 because it advertises indexed transcript
    # search; only this throwaway historical instance may accept the old target.
    historical_store._min_required_revision = through_revision
    try:
        await historical_store.ensure_schema()
    finally:
        await historical_store.close()
        schema_migrations.REVISIONS = revisions


async def _insert_pre_revision_65_entry(
    postgres_dsn: str,
    entry: KnowledgeEntry,
) -> None:
    """Seed an old schema without teaching the production adapter legacy writes."""
    import psycopg
    from psycopg.types.json import Jsonb

    async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                """
                INSERT INTO cayu_knowledge_entries (
                    id, namespace, current_revision, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    entry.id,
                    entry.namespace,
                    entry.revision,
                    entry.created_at,
                    entry.updated_at,
                ),
            )
            await cursor.execute(
                """
                INSERT INTO cayu_knowledge_revisions (
                    entry_id, revision, text, kind, visibility, status,
                    created_by_type, created_by, created_at, updated_at,
                    source_type, source_uri, source_id, source_hash,
                    importance, importance_source, confidence, last_used_at,
                    expires_at, title, metadata
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    entry.id,
                    entry.revision,
                    entry.text,
                    entry.kind,
                    str(entry.visibility),
                    str(entry.status),
                    str(entry.created_by_type),
                    entry.created_by,
                    entry.created_at,
                    entry.updated_at,
                    entry.source_type,
                    entry.source_uri,
                    entry.source_id,
                    entry.source_hash,
                    entry.importance,
                    entry.importance_source,
                    entry.confidence,
                    entry.last_used_at,
                    entry.expires_at,
                    entry.title,
                    Jsonb(entry.metadata),
                ),
            )
        await connection.commit()


def test_postgres_knowledge_write_locks_are_batched_in_global_order(
    postgres_dsn: str,
) -> None:
    from cayu.storage.postgres import _lock_knowledge_write_identities

    class RecordingCursor:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[object, ...]]] = []

        async def execute(self, query: str, params: tuple[object, ...]) -> None:
            self.calls.append((query, params))

    async def run() -> tuple[
        list[tuple[str, tuple[object, ...]]],
        list[tuple[str, tuple[object, ...]]],
    ]:
        cursor = RecordingCursor()
        await _lock_knowledge_write_identities(cursor)
        await _lock_knowledge_write_identities(
            cursor,
            entry_ids=("entry-b", "entry-a"),
            chunk_ids=("chunk-b", "chunk-a", "chunk-b"),
            operation_ids=("operation-b", "operation-a"),
            relation_ids=("relation-b", "relation-a"),
            relation_semantics=("semantic-b", "semantic-a"),
        )
        bulk_cursor = RecordingCursor()
        await _lock_knowledge_write_identities(
            bulk_cursor,
            chunk_ids=tuple(f"chunk-{index:04d}" for index in range(1_000)),
        )
        return cursor.calls, bulk_cursor.calls

    calls, bulk_calls = asyncio.run(run())

    assert len(calls) == 1
    query, params = calls[0]
    assert "unnest(%s::text[])" in query
    assert "SELECT DISTINCT hashtextextended" in query
    assert "ORDER BY lock_key" in query
    assert params == (
        [
            "knowledge-chunk:chunk-a",
            "knowledge-chunk:chunk-b",
            "knowledge-entry:entry-a",
            "knowledge-entry:entry-b",
            "knowledge-operation:operation-a",
            "knowledge-operation:operation-b",
            "knowledge-relation-semantic:semantic-a",
            "knowledge-relation-semantic:semantic-b",
            "knowledge-relation:relation-a",
            "knowledge-relation:relation-b",
        ],
    )
    assert len(bulk_calls) == 1
    assert bulk_calls[0][1] == ([f"knowledge-chunk:chunk-{index:04d}" for index in range(1_000)],)


def test_postgres_relation_write_locks_follow_category_order(
    postgres_dsn: str,
) -> None:
    from cayu.storage.postgres import _lock_knowledge_relation_write_identities

    class RecordingCursor:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[object, ...]]] = []

        async def execute(self, query: str, params: tuple[object, ...]) -> None:
            self.calls.append((query, params))

    async def run() -> list[tuple[str, tuple[object, ...]]]:
        cursor = RecordingCursor()
        await _lock_knowledge_relation_write_identities(
            cursor,
            operation_id="operation-a",
            relations=[
                KnowledgeRelation(
                    id="relation-a",
                    subject=KnowledgeRevisionRef(entry_id="entry-b", revision=1),
                    object=KnowledgeRevisionRef(entry_id="entry-a", revision=1),
                    kind=KnowledgeRelationKind.DERIVED_FROM,
                )
            ],
        )
        return cursor.calls

    calls = asyncio.run(run())

    assert len(calls) == 3
    assert calls[0][1] == (["knowledge-operation:operation-a"],)
    assert calls[1][1] == (["knowledge-entry:entry-a", "knowledge-entry:entry-b"],)
    relation_identities = calls[2][1][0]
    assert isinstance(relation_identities, list)
    assert relation_identities[0].startswith("knowledge-relation-semantic:")
    assert relation_identities[1] == "knowledge-relation:relation-a"


def test_postgres_cancelled_relation_publication_rolls_back_atomically(
    postgres_dsn: str,
    monkeypatch,
) -> None:
    async def run() -> None:
        from cayu import PostgresKnowledgeStore

        await _drop_all(postgres_dsn)
        store = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
            access_scope=_ACCESS_SCOPE,
        )
        try:
            subject = KnowledgeEntry(id="cancelled-relation-subject", text="subject")
            object_ = KnowledgeEntry(id="cancelled-relation-object", text="object")
            await store.create_entry(subject)
            await store.create_entry(object_)
            relation = KnowledgeRelation(
                id="cancelled-relation",
                subject=KnowledgeRevisionRef(entry_id=subject.id, revision=1),
                object=KnowledgeRevisionRef(entry_id=object_.id, revision=1),
                kind=KnowledgeRelationKind.DERIVED_FROM,
            )
            original_insert_change = store._insert_relation_change
            entered = asyncio.Event()

            async def pause_after_relation_insert(*args, **kwargs):
                entered.set()
                await asyncio.Future()

            monkeypatch.setattr(
                store,
                "_insert_relation_change",
                pause_after_relation_insert,
            )
            publication = asyncio.create_task(
                store.publish_relations(
                    [relation],
                    operation_id="cancelled-relation-operation",
                )
            )
            await asyncio.wait_for(entered.wait(), timeout=5)
            publication.cancel()
            with pytest.raises(asyncio.CancelledError):
                await publication
            monkeypatch.setattr(
                store,
                "_insert_relation_change",
                original_insert_change,
            )

            assert (
                await store.load_relation_publication_receipt("cancelled-relation-operation")
                is None
            )
            empty = await store.read_relations(KnowledgeRelationQuery(reference=relation.subject))
            assert empty is not None
            assert empty.relations == []
            assert all(
                change.relation_id is None for change in (await store.read_changes()).changes
            )

            committed = await store.publish_relations(
                [relation],
                operation_id="cancelled-relation-operation",
            )
            assert committed.replayed is False
        finally:
            await store.close()
            await _drop_all(postgres_dsn)

    asyncio.run(run())


def test_postgres_maintenance_candidate_routing_matches_exact_relation_state(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        from cayu import (
            KnowledgeMaintenanceCandidateSignal,
            KnowledgeMaintenanceRouter,
            KnowledgeMaintenanceRoutingOmissionReason,
            KnowledgeMaintenanceRoutingRequest,
            KnowledgeMaintenanceSignalKind,
            PostgresKnowledgeStore,
        )

        await _drop_all(postgres_dsn)
        store = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
            access_scope=_ACCESS_SCOPE,
        )
        try:
            left = KnowledgeEntry(id="routing-postgres-left", text="left")
            right = KnowledgeEntry(id="routing-postgres-right", text="right")
            await store.create_entry(left)
            await store.create_entry(right)
            relation = KnowledgeRelation(
                id="routing-postgres-contradiction",
                subject=KnowledgeRevisionRef(entry_id=left.id, revision=1),
                object=KnowledgeRevisionRef(entry_id=right.id, revision=1),
                kind=KnowledgeRelationKind.CONTRADICTS,
                created_by="test",
                policy_id="routing-postgres-policy",
            )
            signal = KnowledgeMaintenanceCandidateSignal(
                id="routing-postgres-signal",
                kind=KnowledgeMaintenanceSignalKind.CONTRADICTION,
                references=(relation.subject, relation.object),
                producer_id="postgres-conformance",
                producer_version="1",
                reason_code="reviewed_contradiction",
                relation_id=relation.id,
                observed_at=datetime.now(UTC),
            )
            request = KnowledgeMaintenanceRoutingRequest(
                id="routing-postgres-request",
                policy_id="routing-postgres-policy",
                namespace="default",
                access_scope=_ACCESS_SCOPE,
                signals=(signal,),
                created_at=signal.observed_at,
            )
            before = await KnowledgeMaintenanceRouter(store).route(request)
            assert before.candidates == ()
            assert before.omissions[0].reason is (
                KnowledgeMaintenanceRoutingOmissionReason.CONDITION_NOT_MET
            )

            await store.publish_relations(
                [relation],
                operation_id="routing-postgres-relation-publication",
            )
            after = await KnowledgeMaintenanceRouter(store).route(request)
            assert {candidate.reference for candidate in after.candidates} == {
                relation.subject,
                relation.object,
            }
            assert after.omissions == ()
            assert not after.truncated
        finally:
            await store.close()
            await _drop_all(postgres_dsn)

    asyncio.run(run())


def test_postgres_cancelled_maintenance_rolls_back_atomically(
    postgres_dsn: str,
    monkeypatch,
) -> None:
    async def run() -> None:
        from cayu import PostgresKnowledgeStore

        await _drop_all(postgres_dsn)
        store = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
            access_scope=_ACCESS_SCOPE,
        )
        proposal = maintenance_proposal("cancelled-postgres-maintenance")
        decision = maintenance_decision(
            proposal,
            operation_id="cancelled-postgres-maintenance-operation",
            kind=KnowledgeMaintenanceDecisionKind.APPROVE,
        )
        try:
            await _create_proposal_entries(store, proposal)
            baseline = (await store.read_changes(after_sequence=0, limit=100)).high_water_sequence
            original_insert_relations = store._insert_relations
            entered = asyncio.Event()

            async def pause_after_relation_insert(cur, relations) -> None:
                await original_insert_relations(cur, relations)
                entered.set()
                await asyncio.Future()

            monkeypatch.setattr(store, "_insert_relations", pause_after_relation_insert)
            application = asyncio.create_task(store.apply_maintenance_decision(proposal, decision))
            await asyncio.wait_for(entered.wait(), timeout=5)
            application.cancel()
            with pytest.raises(asyncio.CancelledError):
                await application
            monkeypatch.setattr(store, "_insert_relations", original_insert_relations)

            replacement = await store.get_entry(proposal.replacement.entry_id)
            source = await store.get_entry(proposal.sources[0].entry_id)
            assert replacement is not None
            assert replacement.revision == 1
            assert replacement.status is KnowledgeStatus.PENDING
            assert source is not None
            assert source.revision == 1
            assert source.status is KnowledgeStatus.ACTIVE
            relations = await store.read_relations(
                KnowledgeRelationQuery(reference=proposal.sources[0])
            )
            assert relations is not None
            assert relations.relations == []
            assert await store.load_maintenance_decision_receipt(decision.operation_id) is None
            assert (await store.read_changes(after_sequence=0, limit=100)).high_water_sequence == (
                baseline
            )

            committed = await store.apply_maintenance_decision(proposal, decision)
            assert committed.replayed is False
        finally:
            await store.close()
            await _drop_all(postgres_dsn)

    asyncio.run(run())


def test_postgres_relation_change_access_fails_closed_for_malformed_audiences(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        from cayu import PostgresKnowledgeStore

        await _drop_all(postgres_dsn)
        store = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
            access_scope=_ACCESS_SCOPE,
        )
        try:
            for entry_id in ("audience-subject", "audience-object"):
                await store.create_entry(KnowledgeEntry(id=entry_id, text=entry_id))
            relation = KnowledgeRelation(
                id="audience-relation",
                subject=KnowledgeRevisionRef(entry_id="audience-subject", revision=1),
                object=KnowledgeRevisionRef(entry_id="audience-object", revision=1),
                kind=KnowledgeRelationKind.DERIVED_FROM,
            )
            await store.publish_relations([relation], operation_id="audience-operation")
            relation_change = next(
                change
                for change in (await store.read_changes()).changes
                if change.relation_id == relation.id
            )
            async with store._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE cayu_knowledge_change_audiences
                    SET audience_kind = 'after'
                    WHERE change_sequence = %s AND audience_kind = 'object_current'
                    """,
                    (relation_change.sequence,),
                )
                await conn.commit()

            visible = await store.read_changes(
                after_sequence=relation_change.sequence - 1,
            )
            assert all(change.relation_id != relation.id for change in visible.changes)
        finally:
            await store.close()
            await _drop_all(postgres_dsn)

    asyncio.run(run())


class KeywordEmbeddingProvider(TextEmbeddingProvider):
    name = "keyword-test"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
        self.calls.append(list(request.texts))
        return TextEmbeddingResult(
            model=request.model,
            embeddings=[
                TextEmbedding(index=index, vector=_test_embedding_vector(text))
                for index, text in enumerate(request.texts)
            ],
        )


def _test_embedding_vector(text: str) -> list[float]:
    folded = text.casefold()
    return [
        1.0
        if any(
            term in folded for term in ("auth", "broker", "credential", "github", "proxy", "token")
        )
        else 0.0,
        1.0 if any(term in folded for term in ("invoice", "payment", "refund")) else 0.0,
        1.0 if any(term in folded for term in ("sendgrid", "email")) else 0.0,
    ]


async def _drop_all(dsn: str) -> None:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            for table in _TABLES:
                await cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        await conn.commit()


async def _legacy_knowledge_snapshot(cursor) -> tuple[object, ...]:
    await cursor.execute(
        """
        SELECT table_name, column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name LIKE 'cayu_knowledge_%'
        ORDER BY table_name, ordinal_position
        """
    )
    schema_rows = tuple(await cursor.fetchall())
    await cursor.execute(
        "SELECT id, namespace, text, metadata::text FROM cayu_knowledge_entries ORDER BY id"
    )
    entries = tuple(await cursor.fetchall())
    await cursor.execute(
        "SELECT entry_id, key, value FROM cayu_knowledge_labels ORDER BY entry_id, key"
    )
    labels = tuple(await cursor.fetchall())
    await cursor.execute(
        "SELECT id, entry_id, chunk_index, text, metadata::text "
        "FROM cayu_knowledge_chunks ORDER BY id"
    )
    chunks = tuple(await cursor.fetchall())
    await cursor.execute(
        "SELECT revision, kind, compatible_from FROM cayu_schema_migrations ORDER BY revision"
    )
    ledger = tuple(await cursor.fetchall())
    return schema_rows, entries, labels, chunks, ledger


def _new_store(dsn: str):
    from cayu import PostgresKnowledgeStore

    return PostgresKnowledgeStore(
        dsn,
        access_scope=_ACCESS_SCOPE,
        min_size=1,
        max_size=4,
        schema_mode=SchemaMode.CREATE,
    )


def _new_embedding_store(
    dsn: str,
    provider: TextEmbeddingProvider,
    *,
    max_size: int = 4,
    access_scope: KnowledgeAccessScope | None = _ACCESS_SCOPE,
):
    from cayu import PostgresEmbeddingKnowledgeStore

    return PostgresEmbeddingKnowledgeStore(
        dsn,
        access_scope=access_scope,
        min_size=1,
        max_size=max_size,
        schema_mode=SchemaMode.CREATE,
        embedding_provider=provider,
        embedding_model="test-embedding",
        embedding_dimensions=3,
        semantic_min_score=0.70,
    )


def test_postgres_embedding_store_passes_projection_conformance(postgres_dsn: str) -> None:
    async def run() -> None:
        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        readiness_store = _new_embedding_store(postgres_dsn, KeywordEmbeddingProvider())
        try:
            await verify_projection_readiness(readiness_store, adapter="postgres-embedding")
        finally:
            await readiness_store.close()

        await _drop_all(postgres_dsn)
        space_store = _new_embedding_store(postgres_dsn, KeywordEmbeddingProvider())
        try:
            await verify_embedding_space_isolation(
                space_store,
                adapter="postgres-embedding",
            )
        finally:
            await space_store.close()
            await _drop_all(postgres_dsn)

    asyncio.run(run())


def test_postgres_embedding_exact_revision_search_restricts_semantic_candidates(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        provider = KeywordEmbeddingProvider()
        store = _new_embedding_store(postgres_dsn, provider)

        async def create(entry_id: str, *, importance: float) -> None:
            text = "GitHub credential proxy policy."
            await store.create_entry(
                KnowledgeEntry(id=entry_id, text=text, importance=importance),
                [
                    KnowledgeChunk(
                        id=f"{entry_id}-chunk",
                        entry_id=entry_id,
                        text=text,
                        chunk_index=0,
                    )
                ],
            )

        try:
            empty = await store.search_revisions(
                KnowledgeQuery(
                    text="GitHub credential proxy",
                    mode=KnowledgeSearchMode.SEMANTIC,
                    limit=1,
                ),
                (),
            )
            assert empty.hits == []
            assert empty.index_coverage[0].eligible_records == 0
            assert provider.calls == []

            await create("semantic-global-winner", importance=1.0)
            await create("semantic-selected", importance=0.1)
            worker = await store.process_embedding_changes("exact-semantic", "worker")
            assert worker.acknowledged_changes == 2
            query = KnowledgeQuery(
                text="GitHub credential proxy",
                mode=KnowledgeSearchMode.SEMANTIC,
                limit=1,
            )
            global_result = await store.search(query)
            restricted = await store.search_revisions(
                query,
                (KnowledgeRevisionRef(entry_id="semantic-selected", revision=1),),
            )

            assert [hit.entry.id for hit in global_result.hits] == ["semantic-global-winner"]
            assert [hit.entry.id for hit in restricted.hits] == ["semantic-selected"]
            assert restricted.index_coverage[0].eligible_records == 1
        finally:
            await store.close()
            await _drop_all(postgres_dsn)

    asyncio.run(run())


def test_postgres_embedding_exact_revision_search_skips_provider_without_ready_candidates(
    postgres_dsn: str,
) -> None:
    class FailIfCalledEmbeddingProvider(TextEmbeddingProvider):
        name = "fail-if-called-test"

        def __init__(self) -> None:
            self.call_count = 0

        async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
            self.call_count += 1
            raise AssertionError("query embedding must not run without a READY exact candidate")

    async def run() -> None:
        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        provider = FailIfCalledEmbeddingProvider()
        store = _new_embedding_store(postgres_dsn, provider)
        text = "GitHub credential proxy policy."
        try:
            await store.create_entry(
                KnowledgeEntry(id="semantic-pending", text=text),
                [
                    KnowledgeChunk(
                        id="semantic-pending-chunk",
                        entry_id="semantic-pending",
                        text=text,
                        chunk_index=0,
                    )
                ],
            )
            reference = (KnowledgeRevisionRef(entry_id="semantic-pending", revision=1),)
            semantic = await store.search_revisions(
                KnowledgeQuery(
                    text="GitHub credential proxy",
                    mode=KnowledgeSearchMode.SEMANTIC,
                    limit=1,
                ),
                reference,
            )
            hybrid = await store.search_revisions(
                KnowledgeQuery(
                    text="GitHub credential proxy",
                    mode=KnowledgeSearchMode.HYBRID,
                    limit=1,
                ),
                reference,
            )

            assert semantic.hits == []
            assert semantic.index_coverage[0].eligible_records == 1
            assert semantic.index_coverage[0].pending_records == 1
            assert [hit.entry.id for hit in hybrid.hits] == ["semantic-pending"]
            assert hybrid.index_coverage == semantic.index_coverage
            assert provider.call_count == 0
        finally:
            await store.close()
            await _drop_all(postgres_dsn)

    asyncio.run(run())


def test_postgres_embedding_frontier_search_excludes_later_readiness(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        provider = KeywordEmbeddingProvider()
        store = _new_embedding_store(postgres_dsn, provider)
        text = "GitHub credential proxy policy."
        try:
            await store.create_entry(
                KnowledgeEntry(id="semantic-frontier", text=text),
                [
                    KnowledgeChunk(
                        id="semantic-frontier-chunk",
                        entry_id="semantic-frontier",
                        text=text,
                        chunk_index=0,
                    )
                ],
            )
            knowledge_frontier = (await store.read_changes()).high_water_sequence
            worker = await store.process_embedding_changes("semantic-frontier", "worker")
            assert worker.acknowledged_changes == 1
            readiness_frontier = (await store.read_index_readiness()).high_water_sequence
            query = KnowledgeQuery(
                text="GitHub credential proxy",
                mode=KnowledgeSearchMode.SEMANTIC,
                limit=10,
            )
            calls_after_indexing = len(provider.calls)

            before_readiness = await store.search_revisions(
                query,
                (KnowledgeRevisionRef(entry_id="semantic-frontier", revision=1),),
                knowledge_sequence=knowledge_frontier,
                index_readiness_sequence=0,
            )
            assert len(provider.calls) == calls_after_indexing
            after_readiness = await store.search_at_frontier(
                query,
                knowledge_sequence=knowledge_frontier,
                index_readiness_sequence=readiness_frontier,
            )

            assert before_readiness.hits == []
            assert before_readiness.index_coverage[0].pending_records == 1
            assert [hit.entry.id for hit in after_readiness.hits] == ["semantic-frontier"]
        finally:
            await store.close()
            await _drop_all(postgres_dsn)

    asyncio.run(run())


def test_postgres_embedding_frontier_replays_the_captured_projection_attempt(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        store = _new_embedding_store(postgres_dsn, KeywordEmbeddingProvider())
        try:
            entry = await store.create_entry(
                KnowledgeEntry(
                    id="postgres-frontier-projection-attempt",
                    text="GitHub credential proxy",
                )
            )
            chunk = (await store.read_chunks(entry.id))[0]
            identity = knowledge_chunk_embedding_identity(
                chunk,
                embedding_model="test-embedding",
                dimensions=3,
            )
            first_pending = await store.publish_index_readiness(
                KnowledgeIndexReadinessUpdate(
                    identity=identity,
                    state=KnowledgeIndexState.PENDING,
                    attempt_id="postgres-frontier-projection-first",
                ),
                expected_sequence=None,
                operation_id="postgres-frontier-projection-first-pending",
            )
            await store.store_embedding_projections(
                [
                    KnowledgeEmbeddingProjection(
                        identity=identity,
                        readiness_sequence=first_pending.sequence,
                        attempt_id=first_pending.attempt_id,
                        vector=[1.0, 0.0, 0.0],
                    )
                ]
            )
            first_ready = await store.publish_index_readiness(
                KnowledgeIndexReadinessUpdate(
                    identity=identity,
                    state=KnowledgeIndexState.READY,
                    attempt_id=first_pending.attempt_id,
                ),
                expected_sequence=first_pending.sequence,
                operation_id="postgres-frontier-projection-first-ready",
            )
            knowledge_frontier = (await store.read_changes()).high_water_sequence
            query = KnowledgeQuery(
                text="GitHub credential",
                mode=KnowledgeSearchMode.SEMANTIC,
                min_score=0.75,
            )
            captured = await store.search_at_frontier(
                query,
                knowledge_sequence=knowledge_frontier,
                index_readiness_sequence=first_ready.sequence,
            )

            second_pending = await store.publish_index_readiness(
                KnowledgeIndexReadinessUpdate(
                    identity=identity,
                    state=KnowledgeIndexState.PENDING,
                    attempt_id="postgres-frontier-projection-second",
                ),
                expected_sequence=first_ready.sequence,
                operation_id="postgres-frontier-projection-second-pending",
            )
            await store.store_embedding_projections(
                [
                    KnowledgeEmbeddingProjection(
                        identity=identity,
                        readiness_sequence=second_pending.sequence,
                        attempt_id=second_pending.attempt_id,
                        vector=[0.0, 1.0, 0.0],
                    )
                ]
            )
            await store.publish_index_readiness(
                KnowledgeIndexReadinessUpdate(
                    identity=identity,
                    state=KnowledgeIndexState.READY,
                    attempt_id=second_pending.attempt_id,
                ),
                expected_sequence=second_pending.sequence,
                operation_id="postgres-frontier-projection-second-ready",
            )

            current = await store.search(query)
            replay = await store.search_at_frontier(
                query,
                knowledge_sequence=knowledge_frontier,
                index_readiness_sequence=first_ready.sequence,
            )
            assert [hit.entry.id for hit in captured.hits] == [entry.id]
            assert current.hits == []
            assert replay == captured
        finally:
            await store.close()
            await _drop_all(postgres_dsn)

    asyncio.run(run())


def test_postgres_embedding_frontier_search_retains_hnsw_plan(
    postgres_dsn: str,
) -> None:
    class ExplainCursor:
        def __init__(self, cursor) -> None:
            self._cursor = cursor
            self.plan: str | None = None

        async def execute(self, statement, params=None) -> None:
            rendered = str(statement)
            if "WITH nearest_chunks AS" in rendered:
                await self._cursor.execute("EXPLAIN (COSTS OFF) " + rendered, params)
                self.plan = "\n".join(str(row[0]) for row in await self._cursor.fetchall())
            await self._cursor.execute(statement, params)

        async def fetchall(self):
            return await self._cursor.fetchall()

    async def run() -> None:
        from cayu.storage.postgres import _begin_knowledge_read_snapshot

        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        store = _new_embedding_store(postgres_dsn, KeywordEmbeddingProvider())
        text = "GitHub credential proxy policy."
        try:
            await store.create_entry(
                KnowledgeEntry(id="semantic-frontier-plan", text=text),
                [
                    KnowledgeChunk(
                        id="semantic-frontier-plan-chunk",
                        entry_id="semantic-frontier-plan",
                        text=text,
                        chunk_index=0,
                    )
                ],
            )
            knowledge_frontier = (await store.read_changes()).high_water_sequence
            worker = await store.process_embedding_changes(
                "semantic-frontier-plan",
                "worker",
            )
            assert worker.acknowledged_changes == 1
            readiness_frontier = (await store.read_index_readiness()).high_water_sequence
            query = KnowledgeQuery(
                text="GitHub credential proxy",
                mode=KnowledgeSearchMode.SEMANTIC,
                limit=1,
            )

            async with (
                store._pool.connection() as connection,
                connection.transaction(),
                connection.cursor() as cursor,
            ):
                await _begin_knowledge_read_snapshot(cursor)
                await cursor.execute("ANALYZE cayu_knowledge_embeddings")
                await cursor.execute("SET LOCAL enable_seqscan = off")
                await cursor.execute("SET LOCAL enable_sort = off")
                coverage = await store._index_coverage_in_snapshot(
                    cursor,
                    query,
                    access_scope=_ACCESS_SCOPE,
                    through_change_sequence=knowledge_frontier,
                    through_index_readiness_sequence=readiness_frontier,
                )
                explain_cursor = ExplainCursor(cursor)
                rows, _, _ = await store._semantic_search_rows_in_snapshot(
                    explain_cursor,
                    query,
                    _test_embedding_vector(query.text or ""),
                    access_scope=_ACCESS_SCOPE,
                    ready_records=coverage.ready_records,
                    through_change_sequence=knowledge_frontier,
                    through_index_readiness_sequence=readiness_frontier,
                )

            assert [entry_id for entry_id, _, _ in rows] == ["semantic-frontier-plan"]
            assert explain_cursor.plan is not None
            assert store._embedding_history_hnsw_index_name() in explain_cursor.plan
        finally:
            await store.close()
            await _drop_all(postgres_dsn)

    asyncio.run(run())


def test_postgres_knowledge_store_owned_publication_conformance(postgres_dsn: str) -> None:
    async def run() -> None:
        await _drop_all(postgres_dsn)
        store = _new_store(postgres_dsn)
        try:
            await assert_owned_publication_conformance(store)
            await assert_concurrent_publication_conformance(store)
            await assert_stale_operation_cannot_replace_newer_publication(store)
        finally:
            await store.close()
            await _drop_all(postgres_dsn)

    asyncio.run(run())


def test_postgres_index_readiness_conformance(postgres_dsn: str) -> None:
    async def run() -> None:
        await _drop_all(postgres_dsn)
        store = _new_store(postgres_dsn)
        try:
            await assert_index_readiness_conformance(store)
        finally:
            await store.close()
            await _drop_all(postgres_dsn)

    asyncio.run(run())


def test_postgres_owned_publication_does_not_hold_sequence_lock_while_writing_payload(
    postgres_dsn: str,
) -> None:
    from cayu import PostgresKnowledgeStore

    class BlockingEvidenceStore(PostgresKnowledgeStore):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.evidence_inserted = asyncio.Event()
            self.release_evidence_insert = asyncio.Event()

        async def _insert_evidence(self, cur, evidence) -> None:
            await super()._insert_evidence(cur, evidence)
            if evidence and evidence[0].entry_id == "slow-publication":
                self.evidence_inserted.set()
                await self.release_evidence_insert.wait()

    async def run() -> None:
        await _drop_all(postgres_dsn)
        store = BlockingEvidenceStore(
            postgres_dsn,
            access_scope=_ACCESS_SCOPE,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
        )
        publication_task = None
        try:
            await store.ensure_schema()
            entry = KnowledgeEntry(id="slow-publication", text="Large owned payload.")
            publication_task = asyncio.create_task(
                store.publish_entry_revision(
                    entry,
                    [
                        KnowledgeChunk(
                            id="slow-publication:r1:0",
                            entry_id=entry.id,
                            chunk_index=0,
                            text=entry.text,
                        )
                    ],
                    evidence=[
                        KnowledgeEvidence(
                            id="slow-publication-evidence",
                            entry_id=entry.id,
                            source_type="document",
                            source_id="slow-source",
                            source_revision="1",
                        )
                    ],
                    operation_id="slow-publication-operation",
                )
            )
            await asyncio.wait_for(store.evidence_inserted.wait(), timeout=2)

            unrelated = await asyncio.wait_for(
                store.create_entry(
                    KnowledgeEntry(id="unrelated-publication", text="Independent write.")
                ),
                timeout=2,
            )
            assert unrelated.id == "unrelated-publication"
        finally:
            store.release_evidence_insert.set()
            if publication_task is not None:
                await publication_task
            await store.close()
            await _drop_all(postgres_dsn)

    asyncio.run(run())


def test_postgres_knowledge_access_scope_conformance(postgres_dsn: str) -> None:
    from cayu import PostgresKnowledgeStore

    async def run() -> None:
        await _drop_all(postgres_dsn)
        store = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await assert_knowledge_access_scope_conformance(store)
        finally:
            await store.close()
            await _drop_all(postgres_dsn)

    asyncio.run(run())


def test_postgres_exact_revision_search_filters_before_ranking(postgres_dsn: str) -> None:
    async def run() -> None:
        from cayu import PostgresKnowledgeStore

        await _drop_all(postgres_dsn)
        scope = KnowledgeAccessScope.for_namespace("project:exact-recall")
        store = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
            access_scope=scope,
        )
        query_text = "checkpoint aware delta target phrase memory"

        async def create(entry_id: str, text: str) -> None:
            await store.create_entry(
                KnowledgeEntry(
                    id=entry_id,
                    namespace="project:exact-recall",
                    text=text,
                ),
                [
                    KnowledgeChunk(
                        id=f"{entry_id}-chunk",
                        entry_id=entry_id,
                        text=text,
                        chunk_index=0,
                    )
                ],
                access_scope=scope,
            )

        try:
            await create("strong-unselected", " ".join([query_text] * 8))
            await create("selected", query_text)
            query = KnowledgeQuery(
                text=query_text,
                namespace="project:exact-recall",
                mode=KnowledgeSearchMode.KEYWORD,
                limit=1,
            )
            global_result = await store.search(query, access_scope=scope)
            restricted = await store.search_revisions(
                query,
                (KnowledgeRevisionRef(entry_id="selected", revision=1),),
                access_scope=scope,
            )
            current = await store.get_entry("selected", access_scope=scope)
            assert current is not None
            await store.append_entry_revision(
                current.model_copy(update={"revision": 2, "text": f"revised {query_text}"}),
                [
                    KnowledgeChunk(
                        id="selected-revision-2-chunk",
                        entry_id="selected",
                        entry_revision=2,
                        text=f"revised {query_text}",
                        chunk_index=0,
                    )
                ],
                expected_revision=1,
                access_scope=scope,
            )
            stale = await store.search_revisions(
                query,
                (KnowledgeRevisionRef(entry_id="selected", revision=1),),
                access_scope=scope,
            )

            assert [hit.entry.id for hit in global_result.hits] == ["strong-unselected"]
            assert [hit.entry.id for hit in restricted.hits] == ["selected"]
            assert stale.hits == []
        finally:
            await store.close()

    asyncio.run(run())


def test_postgres_checkpoint_recall_full_delta_and_no_work_parity(postgres_dsn: str) -> None:
    async def run() -> None:
        from cayu import (
            DEFAULT_AGENT_RECALL_CHECKPOINT_STREAM_ID,
            KNOWLEDGE_LEXICAL_CHANNEL,
            KNOWLEDGE_SEMANTIC_CHANNEL,
            AgentRecallProcessingMode,
            AgentRecallProcessingRequest,
            AgentRecallProcessor,
            AgentWorkContext,
            PostgresKnowledgeStore,
            RecallSituation,
            WeightedReciprocalRankFusionConfig,
        )

        await _drop_all(postgres_dsn)
        now = datetime(2026, 8, 28, 8, 0, tzinfo=UTC)
        namespace = "project:checkpoint-recall"
        scope = KnowledgeAccessScope.for_namespace(namespace)
        store = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
            access_scope=scope,
        )
        context = AgentWorkContext.create(
            task_id="postgres-checkpoint-recall",
            goal="Process PostgreSQL shared-memory changes",
            revision=1,
            operation_id="postgres-checkpoint-context",
            published_by="test-suite",
            published_at=now,
        )
        processor = AgentRecallProcessor(
            store,
            fusion_config=WeightedReciprocalRankFusionConfig(
                configuration_version="postgres-checkpoint-recall-v1",
                channel_weights={
                    KNOWLEDGE_LEXICAL_CHANNEL: 1.0,
                    KNOWLEDGE_SEMANTIC_CHANNEL: 1.0,
                },
                max_candidates_per_channel=20,
                fused_head_limit=20,
            ),
        )

        def request(operation_id: str, checkpoint=None) -> AgentRecallProcessingRequest:
            return AgentRecallProcessingRequest(
                agent_id="postgres-agent",
                work_context=context,
                situation=RecallSituation(
                    query="checkpoint aware delta target phrase memory",
                    knowledge_access_scope=scope,
                    knowledge_namespace=namespace,
                    current_time=now,
                ),
                checkpoint_stream_id=DEFAULT_AGENT_RECALL_CHECKPOINT_STREAM_ID,
                checkpoint=checkpoint,
                processing_id=f"processing-{operation_id}",
                operation_id=operation_id,
                updated_by="test-suite",
                updated_at=now,
            )

        async def create(entry_id: str) -> None:
            text = "checkpoint aware delta target phrase memory"
            await store.create_entry(
                KnowledgeEntry(id=entry_id, namespace=namespace, text=text),
                [
                    KnowledgeChunk(
                        id=f"{entry_id}-chunk",
                        entry_id=entry_id,
                        text=text,
                        chunk_index=0,
                    )
                ],
                access_scope=scope,
            )

        try:
            await create("postgres-initial")
            initial = await processor.process(request("postgres-initial-full"))
            assert initial.mode is AgentRecallProcessingMode.FULL_INDEX
            assert initial.proposed_checkpoint is not None
            unchanged = await processor.process(
                request("postgres-no-work", initial.proposed_checkpoint)
            )
            assert unchanged.mode is AgentRecallProcessingMode.NO_WORK
            await create("postgres-delta")
            delta = await processor.process(request("postgres-delta", initial.proposed_checkpoint))
            assert delta.mode is AgentRecallProcessingMode.DELTA
            assert [reference.entry_id for reference in delta.eligible_revisions] == [
                "postgres-delta"
            ]
            assert [
                candidate.record.locator["entry_id"] for candidate in delta.recall.candidates
            ] == ["postgres-delta"]
        finally:
            await store.close()

    asyncio.run(run())


def test_postgres_scoped_entry_hydration_uses_one_read_snapshot(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        from cayu import PostgresKnowledgeStore

        await _drop_all(postgres_dsn)
        privileged = _new_store(postgres_dsn)
        try:
            await privileged.create_entry(
                KnowledgeEntry(
                    id="entry",
                    text="snapshot protected",
                    labels={"project": "alpha"},
                )
            )
        finally:
            await privileged.close()

        class RacingStore(PostgresKnowledgeStore):
            changed = False

            async def _load_labels(
                self,
                cur,
                entry_id: str,
                revision: int,
            ) -> dict[str, str]:
                if not self.changed:
                    self.changed = True
                    peer = _new_store(postgres_dsn)
                    try:
                        current = await peer.get_entry(entry_id)
                        assert current is not None
                        await peer.append_entry_revision(
                            current.model_copy(
                                update={"revision": 2, "labels": {"project": "beta"}}
                            ),
                            expected_revision=1,
                        )
                    finally:
                        await peer.close()
                return await super()._load_labels(cur, entry_id, revision)

        scope = KnowledgeAccessScope.for_namespace(
            "default",
            required_labels={"project": "alpha"},
        )
        racing = RacingStore(
            postgres_dsn,
            access_scope=scope,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            loaded = await racing.get_entry("entry")
        finally:
            await racing.close()
        assert loaded is not None
        assert loaded.labels == {"project": "alpha"}

        current = _new_store(postgres_dsn)
        try:
            updated = await current.get_entry("entry")
        finally:
            await current.close()
            await _drop_all(postgres_dsn)
        assert updated is not None
        assert updated.labels == {"project": "beta"}

    asyncio.run(run())


def test_postgres_semantic_candidate_hydration_uses_one_read_snapshot(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        from cayu import PostgresEmbeddingKnowledgeStore

        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        original = KnowledgeEntry(
            id="tenant-a-entry",
            namespace="tenant-a",
            text="Tenant A credential policy.",
        )
        original_chunk = KnowledgeChunk(
            id="reusable-chunk",
            entry_id=original.id,
            chunk_index=0,
            text=original.text,
        )
        privileged = _new_embedding_store(postgres_dsn, KeywordEmbeddingProvider())
        try:
            await privileged.create_entry(original, [original_chunk])
            await privileged.process_embedding_changes(
                "snapshot-postgres-index",
                "worker",
            )
        finally:
            await privileged.close()

        class RacingStore(PostgresEmbeddingKnowledgeStore):
            changed = False

            async def _load_entry_in_scope(self, cur, entry_id, access_scope):
                if not self.changed:
                    self.changed = True
                    peer = _new_store(postgres_dsn)
                    try:
                        await peer.append_entry_revision(
                            original.model_copy(update={"revision": 2}),
                            [
                                KnowledgeChunk(
                                    id="tenant-a-replacement",
                                    entry_id=original.id,
                                    entry_revision=2,
                                    chunk_index=0,
                                    text="Tenant A replacement policy.",
                                )
                            ],
                            expected_revision=1,
                        )
                        tenant_b = KnowledgeEntry(
                            id="tenant-b-entry",
                            namespace="tenant-b",
                            text="Tenant B secret credential policy.",
                        )
                        await peer.create_entry(
                            tenant_b,
                            [
                                KnowledgeChunk(
                                    id="tenant-b:r1:0",
                                    entry_id=tenant_b.id,
                                    chunk_index=0,
                                    text=tenant_b.text,
                                )
                            ],
                        )
                    finally:
                        await peer.close()
                return await super()._load_entry_in_scope(cur, entry_id, access_scope)

        scope = KnowledgeAccessScope.for_namespace("tenant-a")
        racing = RacingStore(
            postgres_dsn,
            access_scope=scope,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
            embedding_provider=KeywordEmbeddingProvider(),
            embedding_model="test-embedding",
            embedding_dimensions=3,
            semantic_min_score=0.70,
        )
        try:
            result = await racing.search(
                KnowledgeQuery(
                    text="credential",
                    namespace="tenant-a",
                    mode=KnowledgeSearchMode.SEMANTIC,
                )
            )
        finally:
            await racing.close()

        assert racing.changed is True
        assert [hit.entry.id for hit in result.hits] == [original.id]
        assert result.hits[0].chunk == original_chunk

        current = _new_store(postgres_dsn)
        try:
            assert [chunk.id for chunk in await current.read_chunks(original.id)] == [
                "tenant-a-replacement"
            ]
            reused = await current.read_chunks("tenant-b-entry")
            assert reused[0].id == "tenant-b:r1:0"
            assert reused[0].text == "Tenant B secret credential policy."
        finally:
            await current.close()
            await _drop_all(postgres_dsn)

    asyncio.run(run())


def test_postgres_hybrid_lanes_share_one_read_snapshot(postgres_dsn: str) -> None:
    async def run() -> None:
        from cayu import PostgresEmbeddingKnowledgeStore

        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        original = KnowledgeEntry(
            id="hybrid-original",
            namespace="tenant-a",
            text="Original credential policy.",
        )
        original_chunk = KnowledgeChunk(
            id="hybrid-original-chunk",
            entry_id=original.id,
            chunk_index=0,
            text=original.text,
        )
        later = KnowledgeEntry(
            id="hybrid-later",
            namespace="tenant-a",
            text="Later credential policy.",
        )
        later_chunk = KnowledgeChunk(
            id="hybrid-later-chunk",
            entry_id=later.id,
            chunk_index=0,
            text=later.text,
        )
        privileged = _new_embedding_store(postgres_dsn, KeywordEmbeddingProvider())
        try:
            await privileged.create_entry(original, [original_chunk])
            await privileged.process_embedding_changes(
                "hybrid-snapshot-postgres-index",
                "worker",
            )
        finally:
            await privileged.close()

        class RacingStore(PostgresEmbeddingKnowledgeStore):
            changed = False

            async def _scored_semantic_rows(
                self,
                cur,
                rows,
                query,
                *,
                access_scope,
            ):
                scored = await super()._scored_semantic_rows(
                    cur,
                    rows,
                    query,
                    access_scope=access_scope,
                )
                if not self.changed:
                    self.changed = True
                    peer = _new_store(postgres_dsn)
                    try:
                        await peer.delete_entry(
                            original.id,
                            expected_revision=original.revision,
                            hard=True,
                        )
                        await peer.create_entry(later, [later_chunk])
                    finally:
                        await peer.close()
                return scored

        scope = KnowledgeAccessScope.for_namespace("tenant-a")
        racing = RacingStore(
            postgres_dsn,
            access_scope=scope,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
            embedding_provider=KeywordEmbeddingProvider(),
            embedding_model="test-embedding",
            embedding_dimensions=3,
            semantic_min_score=0.70,
        )
        try:
            result = await racing.search(
                KnowledgeQuery(
                    text="credential",
                    namespace="tenant-a",
                    mode=KnowledgeSearchMode.HYBRID,
                )
            )
        finally:
            await racing.close()

        assert racing.changed is True
        assert [hit.entry.id for hit in result.hits] == [original.id]
        assert result.hits[0].chunk == original_chunk

        current = _new_store(postgres_dsn)
        try:
            assert await current.get_entry(original.id) is None
            assert await current.get_entry(later.id) == later
        finally:
            await current.close()
            await _drop_all(postgres_dsn)

    asyncio.run(run())


def test_postgres_embedding_worker_does_not_apply_stale_derived_embeddings(
    postgres_dsn: str,
) -> None:
    class FirstCallBlockingEmbeddingProvider(TextEmbeddingProvider):
        name = "blocking-keyword-test"

        def __init__(self) -> None:
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()
            self.call_count = 0

        async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
            self.call_count += 1
            if self.call_count == 1:
                self.first_started.set()
                await self.release_first.wait()
            return TextEmbeddingResult(
                model=request.model,
                embeddings=[
                    TextEmbedding(index=index, vector=_test_embedding_vector(text))
                    for index, text in enumerate(request.texts)
                ],
            )

    async def run() -> tuple[str, list[tuple[str, str]]]:
        import psycopg

        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        provider = FirstCallBlockingEmbeddingProvider()
        store = _new_embedding_store(postgres_dsn, provider)
        old_task: asyncio.Task | None = None
        try:
            old_entry, old_chunks = publication_material(
                entry_id="reused_embedding_publication",
                text="GitHub credential proxy policy.",
            )
            old_chunks = [old_chunks[0].model_copy(update={"id": "old-publication-chunk"})]
            await store.publish_entry_revision(
                old_entry,
                old_chunks,
                operation_id="old-embedding-publication",
            )
            old_task = asyncio.create_task(
                store.process_embedding_changes(
                    "stale-postgres-index",
                    "worker",
                )
            )
            await asyncio.wait_for(provider.first_started.wait(), timeout=2)
            await store.delete_entry(
                old_entry.id,
                expected_revision=old_entry.revision,
                hard=True,
            )
            new_entry, new_chunks = publication_material(
                entry_id=old_entry.id,
                text="Invoice payment refund policy.",
                timestamp_offset=1,
            )
            new_chunks = [new_chunks[0].model_copy(update={"id": "new-publication-chunk"})]
            await store.publish_entry_revision(
                new_entry,
                new_chunks,
                operation_id="new-embedding-publication",
            )
            provider.release_first.set()
            await old_task
            async with (
                await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
                conn.cursor() as cur,
            ):
                await cur.execute(
                    """
                    SELECT chunk_id, projection_content_hash
                    FROM cayu_knowledge_embeddings
                    WHERE entry_id = %s
                    ORDER BY chunk_id
                    """,
                    (old_entry.id,),
                )
                rows = [(str(row[0]), str(row[1])) for row in await cur.fetchall()]
            return _knowledge_chunk_content_hash(new_chunks[0]), rows
        finally:
            provider.release_first.set()
            if old_task is not None and not old_task.done():
                await asyncio.gather(old_task, return_exceptions=True)
            await store.close()
            await _drop_all(postgres_dsn)

    expected_hash, rows = asyncio.run(run())

    assert rows == [("new-publication-chunk", expected_hash)]


def test_postgres_embedding_worker_fences_superseded_attempt_vector_write(
    postgres_dsn: str,
) -> None:
    class FirstCallBlockingEmbeddingProvider(TextEmbeddingProvider):
        name = "blocking-attempt-fence-test"

        def __init__(self) -> None:
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()
            self.call_count = 0

        async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
            self.call_count += 1
            call = self.call_count
            if call == 1:
                self.first_started.set()
                await self.release_first.wait()
            vector = [1.0, 0.0, 0.0] if call == 1 else [0.0, 1.0, 0.0]
            return TextEmbeddingResult(
                model=request.model,
                embeddings=[
                    TextEmbedding(index=index, vector=vector)
                    for index, _ in enumerate(request.texts)
                ],
            )

    async def run() -> tuple[str, int]:
        import psycopg

        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        provider = FirstCallBlockingEmbeddingProvider()
        store = _new_embedding_store(postgres_dsn, provider)
        slow: asyncio.Task | None = None
        try:
            entry, chunks = publication_material(
                entry_id="embedding-attempt-fence",
                text="GitHub credential proxy policy.",
            )
            await store.publish_entry_revision(
                entry,
                chunks,
                operation_id="embedding-attempt-fence-publication",
            )
            slow = asyncio.create_task(
                store.process_embedding_changes("slow-attempt-index", "worker-a")
            )
            await asyncio.wait_for(provider.first_started.wait(), timeout=2)
            fast = await store.process_embedding_changes("fast-attempt-index", "worker-b")
            provider.release_first.set()
            await slow
            assert fast.indexed_records == 1
            async with (
                await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
                conn.cursor() as cur,
            ):
                await cur.execute("SELECT embedding::text FROM cayu_knowledge_embeddings")
                row = await cur.fetchone()
            assert row is not None
            return str(row[0]), provider.call_count
        finally:
            provider.release_first.set()
            if slow is not None and not slow.done():
                await asyncio.gather(slow, return_exceptions=True)
            await store.close()
            await _drop_all(postgres_dsn)

    vector, call_count = asyncio.run(run())

    assert vector == "[0,1,0]"
    assert call_count == 2


def test_postgres_hard_delete_cannot_remove_same_id_republication_embeddings(
    postgres_dsn: str,
) -> None:
    from cayu import PostgresEmbeddingKnowledgeStore

    class DelayedDeleteCleanupStore(PostgresEmbeddingKnowledgeStore):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.cleanup_started = asyncio.Event()
            self.release_cleanup = asyncio.Event()

        async def _drop_entry_embeddings(
            self,
            entry_id: str,
            *,
            expected_deleted_revision: int | None,
            limit: int,
        ) -> tuple[int, bool]:
            if not self.cleanup_started.is_set():
                self.cleanup_started.set()
                await self.release_cleanup.wait()
            return await super()._drop_entry_embeddings(
                entry_id,
                expected_deleted_revision=expected_deleted_revision,
                limit=limit,
            )

    async def run() -> tuple[str, list[tuple[str, str]]]:
        import psycopg

        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        store = DelayedDeleteCleanupStore(
            postgres_dsn,
            access_scope=_ACCESS_SCOPE,
            schema_mode=SchemaMode.CREATE,
            embedding_provider=KeywordEmbeddingProvider(),
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        stale_cleanup: asyncio.Task | None = None
        try:
            old_entry, old_chunks = publication_material(
                entry_id="delete-republication",
                text="Old GitHub credential policy.",
            )
            old_chunks = [old_chunks[0].model_copy(update={"id": "old-delete-chunk"})]
            await store.publish_entry_revision(
                old_entry,
                old_chunks,
                operation_id="old-delete-operation",
            )
            await store.process_embedding_changes("delete-republication-index", "worker")
            await store.delete_entry(
                old_entry.id,
                expected_revision=old_entry.revision,
                hard=True,
            )
            stale_cleanup = asyncio.create_task(
                store.process_embedding_changes("delete-republication-index", "old-worker")
            )
            await asyncio.wait_for(store.cleanup_started.wait(), timeout=2)
            new_entry, new_chunks = publication_material(
                entry_id=old_entry.id,
                text="New invoice payment policy.",
                timestamp_offset=1,
            )
            new_chunks = [new_chunks[0].model_copy(update={"id": "new-delete-chunk"})]
            await store.publish_entry_revision(
                new_entry,
                new_chunks,
                operation_id="new-delete-operation",
            )
            await store.process_embedding_changes(
                "republication-index",
                "new-worker",
            )
            store.release_cleanup.set()
            await stale_cleanup

            async with (
                await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
                conn.cursor() as cur,
            ):
                await cur.execute(
                    """
                    SELECT chunk_id, projection_content_hash
                    FROM cayu_knowledge_embeddings
                    WHERE entry_id = %s
                    ORDER BY chunk_id
                    """,
                    (old_entry.id,),
                )
                rows = [(str(row[0]), str(row[1])) for row in await cur.fetchall()]
            return _knowledge_chunk_content_hash(new_chunks[0]), rows
        finally:
            store.release_cleanup.set()
            if stale_cleanup is not None and not stale_cleanup.done():
                await asyncio.gather(stale_cleanup, return_exceptions=True)
            await store.close()
            await _drop_all(postgres_dsn)

    expected_hash, rows = asyncio.run(run())

    assert rows == [("new-delete-chunk", expected_hash)]


def test_postgres_remember_knowledge_reconciles_ack_loss_and_restart(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        from cayu import PostgresKnowledgeStore, RememberKnowledgeTool, ToolContext

        class AcknowledgementLossPostgresStore(PostgresKnowledgeStore):
            async def publish_entry_revision(
                self,
                entry,
                chunks,
                *,
                operation_id,
                expected_revision=None,
            ):
                await super().publish_entry_revision(
                    entry,
                    chunks,
                    operation_id=operation_id,
                    expected_revision=expected_revision,
                )
                raise RuntimeError("secret canary acknowledgement failure")

        await _drop_all(postgres_dsn)
        context_options = {
            "session_id": "session_1",
            "idempotency_key": "postgres-durable-remember-operation",
        }
        store = AcknowledgementLossPostgresStore(
            postgres_dsn,
            access_scope=_ACCESS_SCOPE,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
        )
        first = await RememberKnowledgeTool().run(
            ToolContext(knowledge_store=store, **context_options),
            {"text": "PostgreSQL knowledge survives acknowledgement loss."},
        )
        await store.close()

        reopened = _new_store(postgres_dsn)
        try:
            replay = await RememberKnowledgeTool().run(
                ToolContext(knowledge_store=reopened, **context_options),
                {"text": "PostgreSQL knowledge survives acknowledgement loss."},
            )
            assert first.is_error is False
            assert first.structured["post_write_error"] == ("publication_acknowledgement_lost")
            assert "secret canary" not in first.content
            assert "secret canary" not in repr(first.structured)
            assert replay.is_error is False
            assert replay.structured["written"] is False
            assert replay.structured["already_known"] is None
            assert replay.structured["publication_replayed"] is True
            assert replay.structured["status"] is None
        finally:
            await reopened.close()
            await _drop_all(postgres_dsn)

    asyncio.run(run())


def test_postgres_remember_knowledge_reports_failed_embedding_without_repeating_it(
    postgres_dsn: str,
) -> None:
    class CountingFailingEmbeddingProvider(TextEmbeddingProvider):
        name = "counting-failing-test"

        def __init__(self) -> None:
            self.call_count = 0

        async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
            self.call_count += 1
            raise RuntimeError("secret canary embedding failure")

    async def run() -> tuple[object, object, object, int]:
        from cayu import RememberKnowledgeTool, ToolContext

        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        provider = CountingFailingEmbeddingProvider()
        store = _new_embedding_store(postgres_dsn, provider)
        try:
            context = ToolContext(
                session_id="session_1",
                idempotency_key="postgres-failed-derived-operation",
                knowledge_store=store,
            )
            arguments = {
                "text": "PostgreSQL source publication survives derived embedding failure."
            }
            first = await RememberKnowledgeTool().run(context, arguments)
            replay = await RememberKnowledgeTool().run(context, arguments)
            worker_result = await store.process_embedding_changes(
                "failed-postgres-index",
                "worker",
            )
            return first, replay, worker_result, provider.call_count
        finally:
            await store.close()
            await _drop_all(postgres_dsn)

    first, replay, worker_result, call_count = asyncio.run(run())

    assert first.is_error is False
    assert first.structured is not None
    assert "post_write_error" not in first.structured
    assert "secret canary" not in first.content
    assert "secret canary" not in repr(first.structured)
    assert replay.is_error is False
    assert replay.structured is not None
    assert replay.structured["written"] is False
    assert replay.structured["already_known"] is None
    assert replay.structured["publication_replayed"] is True
    assert replay.structured["status"] is None
    assert worker_result.failed_records == 1
    assert call_count == 1


def test_postgres_knowledge_publication_rolls_back_each_material_write(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        from cayu import PostgresKnowledgeStore

        class FailingPublicationStore(PostgresKnowledgeStore):
            failure_phase: str | None = None

            async def _insert_entry(self, cur, entry) -> None:
                await super()._insert_entry(cur, entry)
                self._fail_after("entry")

            async def _insert_chunks(self, cur, entry, chunks) -> None:
                await super()._insert_chunks(cur, entry, chunks)
                self._fail_after("chunks")

            async def _insert_publication_receipt(self, cur, receipt, entry) -> None:
                await super()._insert_publication_receipt(cur, receipt, entry)
                self._fail_after("receipt")

            def _fail_after(self, phase: str) -> None:
                if self.failure_phase == phase:
                    raise RuntimeError(f"injected {phase}-boundary failure")

        await _drop_all(postgres_dsn)
        store = FailingPublicationStore(
            postgres_dsn,
            access_scope=_ACCESS_SCOPE,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            for index, failure_phase in enumerate(("entry", "chunks", "receipt")):
                entry, chunks = publication_material(
                    entry_id=f"postgres-rollback-{failure_phase}",
                    timestamp_offset=index,
                )
                store.failure_phase = failure_phase
                with pytest.raises(RuntimeError, match=rf"{failure_phase}-boundary"):
                    await store.publish_entry_revision(
                        entry,
                        chunks,
                        operation_id=f"postgres-rollback-{failure_phase}",
                    )
                await assert_failed_publication_left_no_state(
                    store,
                    entry_id=entry.id,
                    operation_id=f"postgres-rollback-{failure_phase}",
                )
        finally:
            await store.close()
            await _drop_all(postgres_dsn)

    asyncio.run(run())


@pytest.mark.parametrize(
    "failure_phase",
    ["replacement", "predecessor", "relations", "relation_change", "decision"],
)
def test_postgres_maintenance_rolls_back_every_material_boundary(
    postgres_dsn: str,
    failure_phase: str,
) -> None:
    async def run() -> None:
        from cayu import PostgresKnowledgeStore

        class FailingMaintenanceStore(PostgresKnowledgeStore):
            lifecycle_writes = 0

            async def _append_revision(self, *args, **kwargs) -> None:
                await super()._append_revision(*args, **kwargs)
                self.lifecycle_writes += 1
                self._fail_after("replacement" if self.lifecycle_writes == 1 else "predecessor")

            async def _insert_relations(self, cur, relations) -> None:
                await super()._insert_relations(cur, relations)
                self._fail_after("relations")

            async def _insert_relation_change(self, *args, **kwargs):
                change = await super()._insert_relation_change(*args, **kwargs)
                self._fail_after("relation_change")
                return change

            async def _insert_maintenance_record(self, *args, **kwargs) -> None:
                await super()._insert_maintenance_record(*args, **kwargs)
                self._fail_after("decision")

            def _fail_after(self, phase: str) -> None:
                if phase == failure_phase:
                    raise RuntimeError(f"injected {phase}-boundary failure")

        await _drop_all(postgres_dsn)
        store = FailingMaintenanceStore(
            postgres_dsn,
            access_scope=_ACCESS_SCOPE,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
        )
        proposal = maintenance_proposal(f"postgres-rollback-{failure_phase}")
        decision = maintenance_decision(
            proposal,
            operation_id=f"postgres-rollback-{failure_phase}-operation",
            kind=KnowledgeMaintenanceDecisionKind.APPROVE,
        )
        try:
            await _create_proposal_entries(store, proposal)
            baseline = (await store.read_changes(after_sequence=0, limit=100)).high_water_sequence
            with pytest.raises(RuntimeError, match=rf"{failure_phase}-boundary"):
                await store.apply_maintenance_decision(proposal, decision)

            replacement = await store.get_entry(proposal.replacement.entry_id)
            source = await store.get_entry(proposal.sources[0].entry_id)
            assert replacement is not None
            assert replacement.revision == 1
            assert replacement.status is KnowledgeStatus.PENDING
            assert source is not None
            assert source.revision == 1
            assert source.status is KnowledgeStatus.ACTIVE
            relations = await store.read_relations(
                KnowledgeRelationQuery(reference=proposal.sources[0])
            )
            assert relations is not None
            assert relations.relations == []
            assert await store.load_maintenance_decision_receipt(decision.operation_id) is None
            assert (await store.read_changes(after_sequence=0, limit=100)).high_water_sequence == (
                baseline
            )
        finally:
            await store.close()
            await _drop_all(postgres_dsn)

    asyncio.run(run())


async def _skip_if_pgvector_unavailable(dsn: str) -> None:
    import psycopg

    try:
        async with await psycopg.AsyncConnection.connect(dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.commit()
    except Exception as exc:
        pytest.skip(f"pgvector extension is not available: {exc}")


def _run(dsn: str, coro_factory):
    async def runner():
        await _drop_all(dsn)
        store = _new_store(dsn)
        try:
            return await coro_factory(store)
        finally:
            await store.close()

    return asyncio.run(runner())


def test_postgres_knowledge_store_rejects_out_of_range_chunk_index_atomically(
    postgres_dsn: str,
) -> None:
    async def ops(store):
        entry = KnowledgeEntry(id="entry_chunk_index", text="memory")
        chunk = KnowledgeChunk(
            id="chunk_index",
            entry_id=entry.id,
            chunk_index=0,
            text="chunk",
        )
        object.__setattr__(chunk, "chunk_index", MAX_KNOWLEDGE_CHUNK_INDEX + 1)

        with pytest.raises(ValueError, match=str(MAX_KNOWLEDGE_CHUNK_INDEX)):
            await store.create_entry(entry, [chunk])
        assert await store.get_entry(entry.id) is None
        assert await store.read_chunks(entry.id) == []

    _run(postgres_dsn, ops)


def test_postgres_knowledge_store_persists_entries_chunks_and_filters(postgres_dsn: str) -> None:
    async def ops(store):
        await store.create_entry(
            KnowledgeEntry(
                id="invoice_warning",
                text="Do not send invoice reminders when the PO number is missing.",
                namespace="ops",
                labels={"project": "invoice_agent", "user": "alice"},
                kind="warning",
                visibility=KnowledgeVisibility.PROJECT,
                aspects=["finance"],
                impact_targets=["finance.reminders"],
                source_type="manual",
                source_id="invoice_rules",
                importance=0.8,
                metadata={"numbers": {"ordinary": 1.0, "zero": -0.0, "fractional": 1e-7}},
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                updated_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            [
                KnowledgeChunk(
                    id="invoice_warning:0",
                    entry_id="invoice_warning",
                    chunk_index=0,
                    text="Invoice reminders require a PO number.",
                    source_uri="manual://invoice_rules",
                    metadata={"numbers": {"ordinary": 1.0, "zero": -0.0, "fractional": 1e-7}},
                )
            ],
        )
        await store.create_entry(
            KnowledgeEntry(
                id="other_project_warning",
                text="Invoice reminders require a PO number.",
                namespace="ops",
                labels={"project": "other_agent", "user": "alice"},
                kind="warning",
                visibility=KnowledgeVisibility.PROJECT,
            )
        )

        loaded = await store.get_entry("invoice_warning")
        result = await store.search(
            KnowledgeQuery(
                text="invoice reminders",
                namespace="ops",
                labels={"project": "invoice_agent"},
                kinds=["warning"],
                visibilities=[KnowledgeVisibility.PROJECT],
                aspects=["finance"],
                impact_targets=["finance.reminders"],
                source_type="manual",
                source_id="invoice_rules",
            )
        )
        denied = await store.search(
            KnowledgeQuery(
                text="invoice reminders",
                namespace="ops",
                labels={"project": "missing"},
            )
        )
        return loaded, result, denied

    loaded, result, denied = _run(postgres_dsn, ops)

    assert loaded is not None
    assert loaded.labels == {"project": "invoice_agent", "user": "alice"}
    assert loaded.aspects == ["finance"]
    assert loaded.impact_targets == ["finance.reminders"]
    assert loaded.metadata["numbers"] == {"ordinary": 1, "zero": 0, "fractional": 1e-7}
    assert type(loaded.metadata["numbers"]["ordinary"]) is int
    assert type(loaded.metadata["numbers"]["zero"]) is int
    assert type(loaded.metadata["numbers"]["fractional"]) is float
    assert [hit.entry.id for hit in result.hits] == ["invoice_warning"]
    assert result.hits[0].chunk is not None
    assert result.hits[0].chunk.id == "invoice_warning:0"
    assert result.hits[0].chunk.metadata["numbers"] == {
        "ordinary": 1,
        "zero": 0,
        "fractional": 1e-7,
    }
    assert result.hits[0].score_kind == "postgres_full_text"
    assert result.total_hits_known == 1
    assert denied.hits == []


def test_postgres_embedding_knowledge_store_persists_semantic_vectors(postgres_dsn: str) -> None:
    async def ops():
        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        provider = KeywordEmbeddingProvider()
        store = _new_embedding_store(postgres_dsn, provider)
        try:
            await store.create_entry(
                KnowledgeEntry(
                    id="git_policy",
                    text="Use a credential broker for GitHub auth from remote sandboxes.",
                    namespace="ops",
                    labels={"project": "cayu"},
                    kind="procedure",
                    aspects=["credentials", "git"],
                )
            )
            await store.create_entry(
                KnowledgeEntry(
                    id="invoice_policy",
                    text="Invoice refunds require payment approval.",
                    namespace="ops",
                    labels={"project": "cayu"},
                    kind="procedure",
                    aspects=["invoices"],
                )
            )
            await store.process_embedding_changes("persistence-index", "worker")
            result = await store.search(
                KnowledgeQuery(
                    text="auth broker",
                    namespace="ops",
                    labels={"project": "cayu"},
                    mode=KnowledgeSearchMode.SEMANTIC,
                )
            )
        finally:
            await store.close()

        reopened_provider = KeywordEmbeddingProvider()
        reopened = _new_embedding_store(postgres_dsn, reopened_provider)
        try:
            reopened_result = await reopened.search(
                KnowledgeQuery(
                    text="github credential proxy",
                    namespace="ops",
                    labels={"project": "cayu"},
                    mode=KnowledgeSearchMode.SEMANTIC,
                )
            )
        finally:
            await reopened.close()
        return result, reopened_result, provider.calls, reopened_provider.calls

    result, reopened_result, calls, reopened_calls = asyncio.run(ops())

    assert [hit.entry.id for hit in result.hits] == ["git_policy"]
    assert result.hits[0].score_kind == "postgres_semantic"
    assert result.hits[0].chunk is not None
    assert [hit.entry.id for hit in reopened_result.hits] == ["git_policy"]
    assert reopened_calls == [["github credential proxy"]]
    assert calls[:2] == [
        ["Use a credential broker for GitHub auth from remote sandboxes."],
        ["Invoice refunds require payment approval."],
    ]


def test_postgres_embedding_worker_continues_one_change_within_record_budget(
    postgres_dsn: str,
) -> None:
    async def ops():
        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        provider = KeywordEmbeddingProvider()
        store = _new_embedding_store(postgres_dsn, provider)
        try:
            entry = KnowledgeEntry(
                id="bounded-postgres-embedding-change",
                text="Bounded embedding work.",
            )
            chunks = [
                KnowledgeChunk(
                    id=f"bounded-postgres-embedding-change:{index}",
                    entry_id=entry.id,
                    chunk_index=index,
                    text=text,
                )
                for index, text in enumerate(("github auth", "invoice payment", "refund approval"))
            ]
            await store.create_entry(entry, chunks)
            first = await store.process_embedding_changes(
                "bounded-postgres-embedding-index",
                "worker",
                limit=1,
                record_limit=2,
            )
            second = await store.process_embedding_changes(
                "bounded-postgres-embedding-index",
                "worker",
                limit=1,
                record_limit=2,
            )
            state = await store.load_change_consumer_state("bounded-postgres-embedding-index")
            return first, second, state, provider.calls
        finally:
            await store.close()
            await _drop_all(postgres_dsn)

    first, second, state, calls = asyncio.run(ops())

    assert first.processed_records == 2
    assert first.claimed_changes == 1
    assert first.acknowledged_changes == 0
    assert second.processed_records == 1
    assert second.claimed_changes == 1
    assert second.acknowledged_changes == 1
    assert state is not None
    assert state.cursor_sequence == 1
    assert calls == [["github auth", "invoice payment"], ["refund approval"]]


def test_postgres_embedding_worker_pages_stale_cleanup_within_record_budget(
    postgres_dsn: str,
) -> None:
    async def ops():
        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        store = _new_embedding_store(postgres_dsn, KeywordEmbeddingProvider())
        try:
            entry = KnowledgeEntry(id="bounded-postgres-cleanup", text="Old projection set.")
            await store.create_entry(
                entry,
                [
                    KnowledgeChunk(
                        id=f"bounded-postgres-cleanup:{index}",
                        entry_id=entry.id,
                        chunk_index=index,
                        text=f"old projection {index}",
                    )
                    for index in range(5)
                ],
            )
            await store.process_embedding_changes(
                "bounded-postgres-cleanup-index",
                "worker",
                record_limit=10,
            )
            await store.append_entry_revision(
                entry.model_copy(update={"revision": 2, "text": "Current projection."}),
                [
                    KnowledgeChunk(
                        id="bounded-postgres-cleanup:current",
                        entry_id=entry.id,
                        entry_revision=2,
                        chunk_index=0,
                        text="current projection",
                    )
                ],
                expected_revision=1,
            )
            results = []
            for _ in range(4):
                result = await store.process_embedding_changes(
                    "bounded-postgres-cleanup-index",
                    "worker",
                    limit=1,
                    record_limit=2,
                )
                results.append(result)
                if result.acknowledged_changes:
                    break
            async with store._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute("SELECT COUNT(*) FROM cayu_knowledge_embeddings")
                remaining = int((await cur.fetchone())[0])
            return results, remaining
        finally:
            await store.close()
            await _drop_all(postgres_dsn)

    results, remaining = asyncio.run(ops())

    assert all(result.processed_records <= 2 for result in results)
    assert sum(result.removed_records for result in results) == 5
    assert results[-1].acknowledged_changes == 1
    assert remaining == 1


def test_postgres_embedding_worker_repairs_committed_vector_after_restart(
    postgres_dsn: str,
) -> None:
    from cayu import PostgresEmbeddingKnowledgeStore

    class CrashAfterVectorStore(PostgresEmbeddingKnowledgeStore):
        fail_ready_once = True

        async def publish_index_readiness(self, update, **kwargs):
            if self.fail_ready_once and update.state is KnowledgeIndexState.READY:
                self.fail_ready_once = False
                raise RuntimeError("simulated crash after vector commit")
            return await super().publish_index_readiness(update, **kwargs)

    async def ops():
        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        provider = KeywordEmbeddingProvider()
        store = CrashAfterVectorStore(
            postgres_dsn,
            access_scope=_ACCESS_SCOPE,
            schema_mode=SchemaMode.CREATE,
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        try:
            await store.create_entry(
                KnowledgeEntry(id="crash-window", text="GitHub credential proxy.")
            )
            with pytest.raises(RuntimeError, match="simulated crash"):
                await store.process_embedding_changes("crash-postgres-index", "worker-a")
        finally:
            await store.close()

        reopened = _new_embedding_store(postgres_dsn, provider)
        try:
            retry = await reopened.process_embedding_changes(
                "crash-postgres-index",
                "worker-b",
            )
            result = await reopened.search(
                KnowledgeQuery(text="auth", mode=KnowledgeSearchMode.SEMANTIC)
            )
        finally:
            await reopened.close()
        return retry, result, provider.calls

    retry, result, calls = asyncio.run(ops())

    assert retry.indexed_records == 1
    assert retry.acknowledged_changes == 1
    assert [hit.entry.id for hit in result.hits] == ["crash-window"]
    assert result.index_coverage[0].complete is True
    assert calls == [["GitHub credential proxy."], ["auth"]]


def test_postgres_embedding_knowledge_store_query_min_score_overrides_store_default(
    postgres_dsn: str,
) -> None:
    async def ops():
        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        provider = KeywordEmbeddingProvider()
        store = _new_embedding_store(postgres_dsn, provider)
        store.semantic_min_score = 1.0
        try:
            await store.create_entry(KnowledgeEntry(id="matching", text="GitHub credential proxy."))
            await store.create_entry(
                KnowledgeEntry(id="orthogonal", text="Invoice payment policy.")
            )
            await store.process_embedding_changes("min-score-postgres-index", "worker")
            return await store.search(
                KnowledgeQuery(
                    text="auth broker",
                    mode=KnowledgeSearchMode.SEMANTIC,
                    min_score=0.0,
                )
            )
        finally:
            await store.close()

    result = asyncio.run(ops())

    assert [hit.entry.id for hit in result.hits] == ["matching", "orthogonal"]
    assert result.hits[0].score_normalized == 1.0
    assert result.hits[1].score_normalized == 0.5


def test_postgres_embedding_lifecycle_revisions_replace_stale_derived_rows(
    postgres_dsn: str,
) -> None:
    async def ops() -> tuple[KnowledgeEntry, KnowledgeEntry, list[tuple[str, int]]]:
        import psycopg

        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        store = _new_embedding_store(postgres_dsn, KeywordEmbeddingProvider())
        try:
            created = await store.create_entry(
                KnowledgeEntry(id="lifecycle-embedding", text="GitHub credential proxy.")
            )
            archived = await store.transition_entry_status(
                created.id,
                expected_revision=created.revision,
                from_status=KnowledgeStatus.ACTIVE,
                to_status=KnowledgeStatus.ARCHIVED,
            )
            deleted = await store.delete_entry(
                archived.id,
                expected_revision=archived.revision,
            )
            assert deleted is not None
            await store.process_embedding_changes("lifecycle-postgres-index", "worker")
        finally:
            await store.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute(
                """
                SELECT chunk.id, chunk.entry_revision
                FROM cayu_knowledge_embeddings AS embedding
                JOIN cayu_knowledge_chunks AS chunk
                  ON chunk.id = embedding.chunk_id
                 AND chunk.entry_id = embedding.entry_id
                WHERE embedding.entry_id = %s
                ORDER BY chunk.entry_revision, chunk.id
                """,
                (created.id,),
            )
            rows = [(str(row[0]), int(row[1])) for row in await cursor.fetchall()]
        return archived, deleted, rows

    archived, deleted, rows = asyncio.run(ops())

    assert archived.revision == 2
    assert deleted.revision == 3
    assert rows == []


def test_postgres_embedding_knowledge_store_skips_hnsw_for_large_dimensions(
    postgres_dsn: str,
) -> None:
    async def ops():
        from cayu import PostgresEmbeddingKnowledgeStore

        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        store = PostgresEmbeddingKnowledgeStore(
            postgres_dsn,
            access_scope=_ACCESS_SCOPE,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
            embedding_provider=KeywordEmbeddingProvider(),
            embedding_model="large-test-embedding",
            embedding_dimensions=3072,
        )
        try:
            await store._ensure_ready()
        finally:
            await store.close()

        import psycopg

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                """
                SELECT COUNT(*)
                FROM pg_catalog.pg_index AS index_state
                JOIN pg_catalog.pg_class AS index_record
                  ON index_record.oid = index_state.indexrelid
                JOIN pg_catalog.pg_class AS table_record
                  ON table_record.oid = index_state.indrelid
                JOIN pg_catalog.pg_am AS access_method
                  ON access_method.oid = index_record.relam
                WHERE table_record.relname = 'cayu_knowledge_embeddings'
                  AND access_method.amname = 'hnsw'
                """
            )
            row = await cur.fetchone()
        assert row is not None
        return int(row[0])

    index_count = asyncio.run(ops())

    assert index_count == 0


def test_postgres_embedding_knowledge_store_reports_dimension_mismatch_before_indexing(
    postgres_dsn: str,
) -> None:
    async def ops():
        from cayu import PostgresEmbeddingKnowledgeStore

        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        first = PostgresEmbeddingKnowledgeStore(
            postgres_dsn,
            access_scope=_ACCESS_SCOPE,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
            embedding_provider=KeywordEmbeddingProvider(),
            embedding_model="large-test-embedding",
            embedding_dimensions=3072,
        )
        try:
            await first._ensure_ready()
        finally:
            await first.close()

        second = PostgresEmbeddingKnowledgeStore(
            postgres_dsn,
            access_scope=_ACCESS_SCOPE,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
            embedding_provider=KeywordEmbeddingProvider(),
            embedding_model="small-test-embedding",
            embedding_dimensions=3,
        )
        try:
            with pytest.raises(
                RuntimeError,
                match=(
                    "Drop the derived cayu_knowledge_embeddings table, restart with "
                    "schema_mode=CREATE or MIGRATE"
                ),
            ):
                await second._ensure_ready()
        finally:
            await second.close()

    asyncio.run(ops())


@pytest.mark.parametrize(
    ("defect", "constraint_kind", "definition_fragment", "replace_not_valid"),
    (
        ("primary-key", "p", None, False),
        ("identity-check", "c", "identity_sha256", False),
        ("revision-check", "c", "entry_revision > 0", False),
        ("readiness-check", "c", "readiness_sequence > 0", False),
        ("embedding-hash-check", "c", "embedding_sha256", False),
        ("chunk-revision-foreign-key", "f", None, False),
        ("unvalidated-chunk-revision-foreign-key", "f", None, True),
    ),
)
def test_postgres_embedding_schema_rejects_missing_declared_constraints(
    postgres_dsn: str,
    defect: str,
    constraint_kind: str,
    definition_fragment: str | None,
    replace_not_valid: bool,
) -> None:
    async def ops() -> None:
        import psycopg
        from psycopg import sql

        from cayu import PostgresEmbeddingKnowledgeStore

        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        created = _new_embedding_store(postgres_dsn, KeywordEmbeddingProvider())
        try:
            await created._ensure_ready()
        finally:
            await created.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute(
                """
                SELECT constraint_record.conname,
                       pg_get_constraintdef(constraint_record.oid)
                FROM pg_catalog.pg_constraint AS constraint_record
                WHERE constraint_record.conrelid =
                      'cayu_knowledge_embeddings'::regclass
                  AND constraint_record.contype = %s
                """,
                (constraint_kind,),
            )
            candidates = [
                (str(name), str(definition).lower()) for name, definition in await cursor.fetchall()
            ]
            matching = [
                name
                for name, definition in candidates
                if definition_fragment is None or definition_fragment in definition
            ]
            assert len(matching) == 1, (defect, candidates)
            await cursor.execute(
                sql.SQL("ALTER TABLE cayu_knowledge_embeddings DROP CONSTRAINT {}").format(
                    sql.Identifier(matching[0])
                )
            )
            if replace_not_valid:
                await cursor.execute(
                    """
                    ALTER TABLE cayu_knowledge_embeddings
                    ADD CONSTRAINT seeded_unvalidated_embedding_foreign_key
                    FOREIGN KEY (chunk_id, entry_id, entry_revision)
                    REFERENCES cayu_knowledge_chunks(id, entry_id, entry_revision)
                    ON DELETE CASCADE NOT VALID
                    """
                )
            await connection.commit()

        validated = PostgresEmbeddingKnowledgeStore(
            postgres_dsn,
            access_scope=_ACCESS_SCOPE,
            schema_mode=SchemaMode.VALIDATE,
            embedding_provider=KeywordEmbeddingProvider(),
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        try:
            with pytest.raises(RuntimeError, match="revision-bound projection contract"):
                await validated._ensure_ready()
        finally:
            await validated.close()
            await _drop_all(postgres_dsn)

    asyncio.run(ops())


@pytest.mark.parametrize(
    ("index_name", "replacement_columns", "replacement_predicate"),
    (
        ("idx_cayu_knowledge_embeddings_entry", "entry_revision, entry_id", None),
        (
            "idx_cayu_knowledge_embeddings_model_dims",
            "dimensions, embedding_model",
            None,
        ),
        (
            "idx_cayu_knowledge_embeddings_entry",
            "entry_id, entry_revision",
            "entry_revision > 1",
        ),
        (
            "idx_cayu_knowledge_embeddings_current_identity",
            "identity_sha256",
            "current_projection",
        ),
    ),
)
def test_postgres_embedding_schema_rejects_conflicting_required_indexes(
    postgres_dsn: str,
    index_name: str,
    replacement_columns: str,
    replacement_predicate: str | None,
) -> None:
    async def ops() -> None:
        import psycopg
        from psycopg import sql

        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        created = _new_embedding_store(postgres_dsn, KeywordEmbeddingProvider())
        try:
            await created._ensure_ready()
        finally:
            await created.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute(sql.SQL("DROP INDEX {}").format(sql.Identifier(index_name)))
            statement = sql.SQL("CREATE INDEX {} ON cayu_knowledge_embeddings ({})").format(
                sql.Identifier(index_name),
                sql.SQL(replacement_columns),
            )
            if replacement_predicate is not None:
                statement += sql.SQL(" WHERE ") + sql.SQL(replacement_predicate)
            await cursor.execute(statement)
            await connection.commit()

        conflicting = _new_embedding_store(postgres_dsn, KeywordEmbeddingProvider())
        try:
            with pytest.raises(RuntimeError, match=index_name):
                await conflicting._ensure_ready()
        finally:
            await conflicting.close()
            await _drop_all(postgres_dsn)

    asyncio.run(ops())


def test_postgres_embedding_schema_rejects_cross_space_hnsw_index(
    postgres_dsn: str,
) -> None:
    async def ops() -> None:
        import psycopg

        from cayu import PostgresEmbeddingKnowledgeStore

        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        created = _new_embedding_store(postgres_dsn, KeywordEmbeddingProvider())
        try:
            await created._ensure_ready()
        finally:
            await created.close()
        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                """
                CREATE INDEX invalid_cross_space_hnsw
                ON cayu_knowledge_embeddings
                USING hnsw (embedding vector_cosine_ops)
                """
            )
            await conn.commit()

        validated = PostgresEmbeddingKnowledgeStore(
            postgres_dsn,
            access_scope=_ACCESS_SCOPE,
            schema_mode=SchemaMode.VALIDATE,
            embedding_provider=KeywordEmbeddingProvider(),
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        try:
            with pytest.raises(RuntimeError, match="must isolate one complete"):
                await validated._ensure_ready()
        finally:
            await validated.close()
            await _drop_all(postgres_dsn)

    asyncio.run(ops())


@pytest.mark.parametrize(
    "current_predicate",
    (
        "NOT current_projection",
        "current_projection AND entry_id = 'unexpected-scope'",
    ),
)
def test_postgres_embedding_schema_rejects_restricted_current_hnsw_index(
    postgres_dsn: str,
    current_predicate: str,
) -> None:
    async def ops() -> None:
        import psycopg
        from psycopg import sql

        from cayu import PostgresEmbeddingKnowledgeStore
        from cayu.storage.memory import (
            KNOWLEDGE_CHUNK_TEXT_GENERATOR,
            KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION,
            KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION,
            KNOWLEDGE_CHUNK_TEXT_PROJECTION,
            KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION,
        )

        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        created = _new_embedding_store(postgres_dsn, KeywordEmbeddingProvider())
        try:
            await created._ensure_ready()
            index_name = created._embedding_hnsw_index_name()
        finally:
            await created.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute(sql.SQL("DROP INDEX {}").format(sql.Identifier(index_name)))
            await cursor.execute(
                sql.SQL(
                    """
                    CREATE INDEX {index}
                    ON cayu_knowledge_embeddings USING hnsw (embedding vector_cosine_ops)
                    WHERE {current_predicate}
                      AND projection_type = {projection_type}
                      AND embedding_model = {embedding_model}
                      AND dimensions = {dimensions}
                      AND preprocessing_version = {preprocessing_version}
                      AND generator = {generator}
                      AND generator_version = {generator_version}
                      AND index_representation_version = {index_representation_version}
                    """
                ).format(
                    index=sql.Identifier(index_name),
                    current_predicate=sql.SQL(current_predicate),
                    projection_type=sql.Literal(KNOWLEDGE_CHUNK_TEXT_PROJECTION),
                    embedding_model=sql.Literal("test-embedding"),
                    dimensions=sql.Literal(3),
                    preprocessing_version=sql.Literal(KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION),
                    generator=sql.Literal(KNOWLEDGE_CHUNK_TEXT_GENERATOR),
                    generator_version=sql.Literal(KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION),
                    index_representation_version=sql.Literal(
                        KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION
                    ),
                )
            )
            await connection.commit()

        validated = PostgresEmbeddingKnowledgeStore(
            postgres_dsn,
            access_scope=_ACCESS_SCOPE,
            schema_mode=SchemaMode.VALIDATE,
            embedding_provider=KeywordEmbeddingProvider(),
            embedding_model="test-embedding",
            embedding_dimensions=3,
        )
        try:
            with pytest.raises(RuntimeError, match="embedding HNSW index"):
                await validated._ensure_ready()
        finally:
            await validated.close()
            await _drop_all(postgres_dsn)

    asyncio.run(ops())


def test_postgres_embedding_knowledge_store_backfills_existing_chunks(
    postgres_dsn: str,
) -> None:
    async def ops():
        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        base = _new_store(postgres_dsn)
        try:
            await base.create_entry(
                KnowledgeEntry(
                    id="git_policy",
                    text="Use a credential broker for GitHub auth from remote sandboxes.",
                    namespace="ops",
                    labels={"project": "cayu"},
                    kind="procedure",
                )
            )
            await base.create_entry(
                KnowledgeEntry(
                    id="invoice_policy",
                    text="GitHub token pushes should use the broker.",
                    namespace="ops",
                    labels={"project": "cayu"},
                    kind="procedure",
                )
            )
            await base.create_entry(
                KnowledgeEntry(
                    id="other_policy",
                    text="Invoice refunds require payment approval.",
                    namespace="ops",
                    labels={"project": "other"},
                    kind="procedure",
                )
            )
        finally:
            await base.close()

        provider = KeywordEmbeddingProvider()
        store = _new_embedding_store(postgres_dsn, provider)
        try:
            # Explicit bounded backfill embeds the missing chunks one page at a
            # time; semantic searches remain read-only and are exercised separately.
            first_backfill = await store.backfill_embeddings(
                KnowledgeListQuery(
                    namespace="ops",
                    labels={"project": "cayu"},
                ),
                limit=1,
            )
            second_backfill = await store.backfill_embeddings(
                KnowledgeListQuery(
                    namespace="ops",
                    labels={"project": "cayu"},
                ),
                limit=1,
            )
            third_backfill = await store.backfill_embeddings(
                KnowledgeListQuery(
                    namespace="ops",
                    labels={"project": "cayu"},
                ),
                limit=10,
            )
            first_refresh = await store.backfill_embeddings(
                KnowledgeListQuery(
                    namespace="ops",
                    labels={"project": "cayu"},
                ),
                limit=1,
                refresh_existing=True,
            )
            assert first_refresh.next_cursor is not None
            second_refresh = await store.backfill_embeddings(
                KnowledgeListQuery(
                    namespace="ops",
                    labels={"project": "cayu"},
                ),
                limit=1,
                refresh_existing=True,
                cursor=first_refresh.next_cursor,
            )
        finally:
            await store.close()
        return (
            first_backfill,
            second_backfill,
            third_backfill,
            first_refresh,
            second_refresh,
            provider.calls,
        )

    first_backfill, second_backfill, third_backfill, first_refresh, second_refresh, calls = (
        asyncio.run(ops())
    )

    assert first_backfill.scanned_records == 1
    assert first_backfill.indexed_records == 1
    assert first_backfill.failed_records == 0
    assert first_backfill.skipped_records == 0
    assert second_backfill.scanned_records == 1
    assert second_backfill.indexed_records == 1
    assert second_backfill.failed_records == 0
    assert second_backfill.skipped_records == 0
    assert third_backfill.scanned_records == 0
    assert third_backfill.indexed_records == 0
    assert third_backfill.failed_records == 0
    assert third_backfill.skipped_records == 0
    assert first_refresh.scanned_records == 1
    assert first_refresh.indexed_records == 1
    assert first_refresh.failed_records == 0
    assert first_refresh.next_cursor is not None
    assert second_refresh.scanned_records == 1
    assert second_refresh.indexed_records == 1
    assert second_refresh.failed_records == 0
    assert second_refresh.next_cursor is None
    cayu_texts = {
        "GitHub token pushes should use the broker.",
        "Use a credential broker for GitHub auth from remote sandboxes.",
    }
    single_calls = sorted(tuple(call) for call in calls if len(call) == 1)
    assert len(single_calls) == 4
    assert {call[0] for call in single_calls} == cayu_texts


class FlakyEmbeddingProvider(TextEmbeddingProvider):
    """Keyword provider that can be toggled to fail, simulating an outage."""

    name = "flaky-test"

    def __init__(self) -> None:
        self.fail = False
        self.calls: list[list[str]] = []

    async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
        if self.fail:
            raise RuntimeError("embedding provider is unavailable")
        self.calls.append(list(request.texts))
        return TextEmbeddingResult(
            model=request.model,
            embeddings=[
                TextEmbedding(index=index, vector=_test_embedding_vector(text))
                for index, text in enumerate(request.texts)
            ],
        )


def test_postgres_embedding_failure_is_visible_until_explicit_backfill_recovers(
    postgres_dsn: str,
) -> None:
    async def ops():
        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        provider = FlakyEmbeddingProvider()
        from cayu import PostgresEmbeddingKnowledgeStore

        store = PostgresEmbeddingKnowledgeStore(
            postgres_dsn,
            access_scope=_ACCESS_SCOPE,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
            embedding_provider=provider,
            embedding_model="test-embedding",
            embedding_dimensions=3,
            semantic_min_score=0.70,
        )
        try:
            # Canonical publication never invokes the embedding provider.
            provider.fail = True
            stored = await store.create_entry(
                KnowledgeEntry(
                    id="git_policy",
                    text="Use a credential broker for GitHub auth from remote sandboxes.",
                    namespace="ops",
                    labels={"project": "cayu"},
                    kind="procedure",
                )
            )
            loaded = await store.get_entry("git_policy")
            keyword_hit = await store.search(
                KnowledgeQuery(
                    text="broker",
                    namespace="ops",
                    labels={"project": "cayu"},
                    mode=KnowledgeSearchMode.KEYWORD,
                )
            )
            failed_worker = await store.process_embedding_changes(
                "flaky-postgres-index",
                "worker",
            )
            embedded_calls_during_outage = list(provider.calls)

            # Semantic reads report the failed projection but never mutate it.
            provider.fail = False
            before_recovery = await store.search(
                KnowledgeQuery(
                    text="auth broker",
                    namespace="ops",
                    labels={"project": "cayu"},
                    mode=KnowledgeSearchMode.SEMANTIC,
                )
            )
            backfill = await store.backfill_embeddings(
                KnowledgeListQuery(namespace="ops", labels={"project": "cayu"})
            )
            semantic_hit = await store.search(
                KnowledgeQuery(
                    text="auth broker",
                    namespace="ops",
                    labels={"project": "cayu"},
                    mode=KnowledgeSearchMode.SEMANTIC,
                )
            )
        finally:
            await store.close()
        return (
            stored,
            loaded,
            keyword_hit,
            failed_worker,
            embedded_calls_during_outage,
            before_recovery,
            backfill,
            semantic_hit,
        )

    (
        stored,
        loaded,
        keyword_hit,
        failed_worker,
        outage_calls,
        before_recovery,
        backfill,
        semantic_hit,
    ) = asyncio.run(ops())

    # The write succeeded and returned the entry despite the embedding failure.
    assert stored.id == "git_policy"
    assert loaded is not None
    # No embeddings were persisted during the outage.
    assert outage_calls == []
    assert failed_worker.failed_records == 1
    # Keyword search still surfaces the durable entry with no embeddings present.
    assert [hit.entry.id for hit in keyword_hit.hits] == ["git_policy"]
    assert before_recovery.hits == []
    assert before_recovery.index_coverage[0].failed_records == 1
    assert backfill.indexed_records == 1
    assert backfill.failed_records == 0
    # After explicit repair the semantic search finds it.
    assert [hit.entry.id for hit in semantic_hit.hits] == ["git_policy"]
    assert semantic_hit.hits[0].score_kind == "postgres_semantic"


def test_postgres_accepts_precomputed_projection_only_for_current_pending_attempt(
    postgres_dsn: str,
) -> None:
    async def ops():
        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        provider = KeywordEmbeddingProvider()
        store = _new_embedding_store(postgres_dsn, provider)
        try:
            await store.create_entry(
                KnowledgeEntry(id="external-postgres-projection", text="GitHub proxy.")
            )
            chunk = (await store.read_chunks("external-postgres-projection"))[0]
            from cayu import knowledge_chunk_embedding_identity

            identity = knowledge_chunk_embedding_identity(
                chunk,
                embedding_model="test-embedding",
                dimensions=3,
            )
            pending = await store.publish_index_readiness(
                KnowledgeIndexReadinessUpdate(
                    identity=identity,
                    state=KnowledgeIndexState.PENDING,
                    attempt_id="external-postgres-attempt",
                ),
                expected_sequence=None,
                operation_id="external-postgres-projection:pending",
            )
            projection = KnowledgeEmbeddingProjection(
                identity=identity,
                readiness_sequence=pending.sequence,
                attempt_id=pending.attempt_id,
                vector=[1.0, 0.0, 0.0],
            )
            stored = await store.store_embedding_projections([projection])
            replayed = await store.store_embedding_projections([projection])
            with pytest.raises(KnowledgeEmbeddingProjectionConflict) as raised:
                await store.store_embedding_projections(
                    [projection.model_copy(update={"vector": [0.0, 1.0, 0.0]})]
                )
            await store.publish_index_readiness(
                KnowledgeIndexReadinessUpdate(
                    identity=identity,
                    state=KnowledgeIndexState.READY,
                    attempt_id=pending.attempt_id,
                ),
                expected_sequence=pending.sequence,
                operation_id="external-postgres-projection:ready",
            )
            stale = await store.store_embedding_projections([projection])
            result = await store.search(
                KnowledgeQuery(
                    text="github",
                    mode=KnowledgeSearchMode.SEMANTIC,
                    min_score=0.0,
                )
            )
            with pytest.raises(ValueError, match="limit.*less than or equal"):
                await store.backfill_embeddings(limit=MAX_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT + 1)
            return stored, replayed, raised.value, stale, result, provider.calls
        finally:
            await store.close()
            await _drop_all(postgres_dsn)

    stored, replayed, conflict, stale, result, provider_calls = asyncio.run(ops())

    assert [identity.entry_id for identity in stored.stored_identities] == [
        "external-postgres-projection"
    ]
    assert replayed.stored_identities == stored.stored_identities
    assert conflict.reason == "attempt_vector_conflict"
    assert stale.stored_identities == []
    assert [hit.entry.id for hit in result.hits] == ["external-postgres-projection"]
    assert provider_calls == [["github"]]


def test_postgres_projection_write_result_reapplies_per_call_access_scope(
    postgres_dsn: str,
) -> None:
    async def ops():
        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        store = _new_embedding_store(
            postgres_dsn,
            KeywordEmbeddingProvider(),
            access_scope=None,
        )
        privileged = KnowledgeAccessScope.privileged()
        unauthorized = KnowledgeAccessScope.for_namespace("tenant-b")
        try:
            await store.create_entry(
                KnowledgeEntry(
                    id="scope-fenced-projection",
                    namespace="tenant-a",
                    text="GitHub proxy.",
                ),
                access_scope=privileged,
            )
            chunk = (
                await store.read_chunks(
                    "scope-fenced-projection",
                    access_scope=privileged,
                )
            )[0]
            from cayu import knowledge_chunk_embedding_identity

            identity = knowledge_chunk_embedding_identity(
                chunk,
                embedding_model="test-embedding",
                dimensions=3,
            )
            pending = await store.publish_index_readiness(
                KnowledgeIndexReadinessUpdate(
                    identity=identity,
                    state=KnowledgeIndexState.PENDING,
                    attempt_id="scope-fenced-attempt",
                ),
                expected_sequence=None,
                operation_id="scope-fenced-projection:pending",
                access_scope=privileged,
            )
            projection = KnowledgeEmbeddingProjection(
                identity=identity,
                readiness_sequence=pending.sequence,
                attempt_id=pending.attempt_id,
                vector=[1.0, 0.0, 0.0],
            )
            authorized = await store.store_embedding_projections(
                [projection],
                access_scope=privileged,
            )
            rejected = await store.store_embedding_projections(
                [projection],
                access_scope=unauthorized,
            )
            return authorized, rejected
        finally:
            await store.close()
            await _drop_all(postgres_dsn)

    authorized, rejected = asyncio.run(ops())

    assert [identity.entry_id for identity in authorized.stored_identities] == [
        "scope-fenced-projection"
    ]
    assert rejected.stored_identities == []


def test_postgres_concurrent_projection_writers_cannot_replace_one_attempt_vector(
    postgres_dsn: str,
) -> None:
    async def ops():
        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        store = _new_embedding_store(
            postgres_dsn,
            KeywordEmbeddingProvider(),
            max_size=2,
        )
        try:
            await store.create_entry(
                KnowledgeEntry(id="concurrent-projection", text="GitHub proxy.")
            )
            chunk = (await store.read_chunks("concurrent-projection"))[0]
            from cayu import knowledge_chunk_embedding_identity

            identity = knowledge_chunk_embedding_identity(
                chunk,
                embedding_model="test-embedding",
                dimensions=3,
            )
            pending = await store.publish_index_readiness(
                KnowledgeIndexReadinessUpdate(
                    identity=identity,
                    state=KnowledgeIndexState.PENDING,
                    attempt_id="concurrent-projection-attempt",
                ),
                expected_sequence=None,
                operation_id="concurrent-projection:pending",
            )
            first = KnowledgeEmbeddingProjection(
                identity=identity,
                readiness_sequence=pending.sequence,
                attempt_id=pending.attempt_id,
                vector=[1.0, 0.0, 0.0],
            )
            second = first.model_copy(update={"vector": [0.0, 1.0, 0.0]})
            return await asyncio.gather(
                store.store_embedding_projections([first]),
                store.store_embedding_projections([second]),
                return_exceptions=True,
            )
        finally:
            await store.close()
            await _drop_all(postgres_dsn)

    results = asyncio.run(ops())

    conflicts = [
        result for result in results if isinstance(result, KnowledgeEmbeddingProjectionConflict)
    ]
    accepted = [result for result in results if not isinstance(result, BaseException)]
    assert len(conflicts) == 1
    assert conflicts[0].reason == "attempt_vector_conflict"
    assert len(accepted) == 1
    assert [identity.entry_id for identity in accepted[0].stored_identities] == [
        "concurrent-projection"
    ]


def test_postgres_projection_store_serializes_readiness_and_keeps_one_current_attempt(
    postgres_dsn: str,
) -> None:
    from cayu import PostgresEmbeddingKnowledgeStore

    class BlockingProjectionStore(PostgresEmbeddingKnowledgeStore):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.projection_readiness_locked = asyncio.Event()
            self.release_projection_write = asyncio.Event()

        async def _lock_embedding_projection_readiness(
            self,
            cursor,
            identity_sha256s,
        ) -> None:
            await super()._lock_embedding_projection_readiness(cursor, identity_sha256s)
            self.projection_readiness_locked.set()
            await self.release_projection_write.wait()

    async def ops():
        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        store = BlockingProjectionStore(
            postgres_dsn,
            access_scope=_ACCESS_SCOPE,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
            embedding_provider=KeywordEmbeddingProvider(),
            embedding_model="test-embedding",
            embedding_dimensions=3,
            semantic_min_score=0.9,
        )
        projection_task = None
        ready_task = None
        try:
            await store.create_entry(
                KnowledgeEntry(id="serialized-projection", text="GitHub proxy.")
            )
            chunk = (await store.read_chunks("serialized-projection"))[0]
            identity = knowledge_chunk_embedding_identity(
                chunk,
                embedding_model="test-embedding",
                dimensions=3,
            )
            first_pending = await store.publish_index_readiness(
                KnowledgeIndexReadinessUpdate(
                    identity=identity,
                    state=KnowledgeIndexState.PENDING,
                    attempt_id="serialized-projection-first",
                ),
                expected_sequence=None,
                operation_id="serialized-projection:first-pending",
            )
            first_projection = KnowledgeEmbeddingProjection(
                identity=identity,
                readiness_sequence=first_pending.sequence,
                attempt_id=first_pending.attempt_id,
                vector=[1.0, 0.0, 0.0],
            )
            projection_task = asyncio.create_task(
                store.store_embedding_projections([first_projection])
            )
            await asyncio.wait_for(store.projection_readiness_locked.wait(), timeout=2)

            ready_task = asyncio.create_task(
                store.publish_index_readiness(
                    KnowledgeIndexReadinessUpdate(
                        identity=identity,
                        state=KnowledgeIndexState.READY,
                        attempt_id=first_pending.attempt_id,
                    ),
                    expected_sequence=first_pending.sequence,
                    operation_id="serialized-projection:first-ready",
                )
            )
            await asyncio.sleep(0.05)
            assert not ready_task.done()

            store.release_projection_write.set()
            first_write = await asyncio.wait_for(projection_task, timeout=2)
            first_ready = await asyncio.wait_for(ready_task, timeout=2)
            projection_task = None
            ready_task = None

            second_pending = await store.publish_index_readiness(
                KnowledgeIndexReadinessUpdate(
                    identity=identity,
                    state=KnowledgeIndexState.PENDING,
                    attempt_id="serialized-projection-second",
                ),
                expected_sequence=first_ready.sequence,
                operation_id="serialized-projection:second-pending",
            )
            second_write = await store.store_embedding_projections(
                [
                    KnowledgeEmbeddingProjection(
                        identity=identity,
                        readiness_sequence=second_pending.sequence,
                        attempt_id=second_pending.attempt_id,
                        vector=[0.0, 1.0, 0.0],
                    )
                ]
            )
            await store.publish_index_readiness(
                KnowledgeIndexReadinessUpdate(
                    identity=identity,
                    state=KnowledgeIndexState.READY,
                    attempt_id=second_pending.attempt_id,
                ),
                expected_sequence=second_pending.sequence,
                operation_id="serialized-projection:second-ready",
            )
            async with store._pool.connection() as connection, connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT readiness_sequence
                    FROM cayu_knowledge_embeddings
                    WHERE current_projection
                    """
                )
                current_sequences = [int(row[0]) for row in await cursor.fetchall()]
                await cursor.execute(
                    """
                    SELECT index_state.indisunique,
                           pg_get_expr(index_state.indpred, index_state.indrelid)
                    FROM pg_catalog.pg_index AS index_state
                    JOIN pg_catalog.pg_class AS index_record
                      ON index_record.oid = index_state.indexrelid
                    WHERE index_record.relname =
                          'idx_cayu_knowledge_embeddings_current_identity'
                    """
                )
                current_index = await cursor.fetchone()
            stale_vector_result = await store.search(
                KnowledgeQuery(
                    text="github",
                    mode=KnowledgeSearchMode.SEMANTIC,
                    min_score=0.9,
                )
            )
            return (
                first_write,
                second_write,
                second_pending.sequence,
                current_sequences,
                current_index,
                stale_vector_result,
            )
        finally:
            store.release_projection_write.set()
            for task in (projection_task, ready_task):
                if task is not None:
                    task.cancel()
            await asyncio.gather(
                *(task for task in (projection_task, ready_task) if task is not None),
                return_exceptions=True,
            )
            await store.close()
            await _drop_all(postgres_dsn)

    (
        first_write,
        second_write,
        second_sequence,
        current_sequences,
        current_index,
        stale_vector_result,
    ) = asyncio.run(ops())

    assert len(first_write.stored_identities) == 1
    assert len(second_write.stored_identities) == 1
    assert current_sequences == [second_sequence]
    assert current_index == (True, "current_projection")
    assert stale_vector_result.hits == []


def test_postgres_knowledge_store_defaults_hide_inactive_and_expired(
    postgres_dsn: str,
) -> None:
    async def ops(store):
        await store.create_entry(KnowledgeEntry(id="active", text="deployment warning"))
        await store.create_entry(
            KnowledgeEntry(
                id="pending",
                text="deployment warning",
                status=KnowledgeStatus.PENDING,
            )
        )
        await store.create_entry(
            KnowledgeEntry(
                id="expired",
                text="deployment warning",
                expires_at=datetime.now(UTC) - timedelta(seconds=1),
            )
        )
        active = await store.search(KnowledgeQuery(text="deployment"))
        pending = await store.search(
            KnowledgeQuery(text="deployment", statuses=[KnowledgeStatus.PENDING])
        )
        expired = await store.search(KnowledgeQuery(text="deployment", include_expired=True))
        return active, pending, expired

    active, pending, expired = _run(postgres_dsn, ops)

    assert [hit.entry.id for hit in active.hits] == ["active"]
    assert [hit.entry.id for hit in pending.hits] == ["pending"]
    assert [hit.entry.id for hit in expired.hits] == ["expired", "active"]


def test_postgres_knowledge_store_preserves_custom_chunks_on_entry_update(
    postgres_dsn: str,
) -> None:
    async def ops(store):
        current = await store.create_entry(
            KnowledgeEntry(id="doc", text="Document summary.", metadata={"version": 1}),
            [
                KnowledgeChunk(
                    id="doc:0",
                    entry_id="doc",
                    chunk_index=0,
                    text="Custom indexed body.",
                    metadata={"indexer": "custom"},
                )
            ],
        )
        await store.append_entry_revision(
            current.model_copy(update={"revision": 2, "metadata": {"version": 2}}),
            expected_revision=1,
        )
        chunks = await store.read_chunks("doc")
        result = await store.search(KnowledgeQuery(text="custom indexed"))
        return chunks, result

    chunks, result = _run(postgres_dsn, ops)

    assert len(chunks) == 1
    assert chunks[0].text == "Custom indexed body."
    assert chunks[0].metadata == {"indexer": "custom"}
    assert [hit.entry.id for hit in result.hits] == ["doc"]


def test_postgres_knowledge_store_empty_kind_filter_returns_no_matches(
    postgres_dsn: str,
) -> None:
    async def ops(store):
        await store.create_entry(KnowledgeEntry(id="entry", text="billing memory"))
        return await store.search(KnowledgeQuery(text="billing", kinds=[]))

    result = _run(postgres_dsn, ops)

    assert result.hits == []
    assert result.total_hits_known == 0


def test_postgres_knowledge_store_search_reports_preview_truncation(
    postgres_dsn: str,
) -> None:
    async def ops(store):
        await store.create_entry(KnowledgeEntry(id="entry", text="billing memory has more text"))
        return await store.search(KnowledgeQuery(text="billing", max_bytes=7))

    result = _run(postgres_dsn, ops)

    assert len(result.hits) == 1
    assert result.hits[0].text_preview == "billing"
    assert result.truncated is True


def test_postgres_knowledge_store_search_dedupes_across_large_chunk_matches(
    postgres_dsn: str,
) -> None:
    async def ops(store):
        await store.create_entry(
            KnowledgeEntry(id="large", text="invoice corpus"),
            [
                KnowledgeChunk(
                    id=f"large:{index}",
                    entry_id="large",
                    chunk_index=index,
                    text=f"invoice repeated chunk {index}",
                )
                for index in range(1200)
            ],
        )
        await store.create_entry(KnowledgeEntry(id="small", text="invoice policy"))
        return await store.search(KnowledgeQuery(text="invoice", limit=2))

    result = _run(postgres_dsn, ops)

    assert {hit.entry.id for hit in result.hits} == {"large", "small"}
    assert result.total_hits_known == 2
    assert result.truncated is False


def test_postgres_knowledge_store_structured_keyword_search(postgres_dsn: str) -> None:
    async def ops(store):
        await store.create_entry(
            KnowledgeEntry(id="github_secret", text="GitHub push requires a credential broker.")
        )
        await store.create_entry(
            KnowledgeEntry(id="sendgrid_secret", text="SendGrid email uses a secret proxy.")
        )
        await store.create_entry(
            KnowledgeEntry(id="github_test", text="GitHub test credentials are fixture-only.")
        )
        return await store.search(
            KnowledgeQuery(
                any_terms=["credential", "secret"],
                all_terms=["github push"],
                none_terms=["fixture only"],
            )
        )

    result = _run(postgres_dsn, ops)

    assert [hit.entry.id for hit in result.hits] == ["github_secret"]


def test_postgres_knowledge_store_filter_only_exact_aspect_groups(
    postgres_dsn: str,
) -> None:
    scope_a = agent_recall_facet_aspect("scope_ids", "scope:a")
    scope_b = agent_recall_facet_aspect("scope_ids", "scope:b")
    entity_target = agent_recall_facet_aspect("entity_ids", "entity:target")
    entity_other = agent_recall_facet_aspect("entity_ids", "entity:other")

    async def ops(store):
        await store.create_entry(
            KnowledgeEntry(
                id="facet_target",
                text="unrelated target body",
                aspects=[scope_b, entity_target],
            )
        )
        await store.create_entry(
            KnowledgeEntry(
                id="facet_wrong_entity",
                text="unrelated wrong entity body",
                aspects=[scope_b, entity_other],
            )
        )
        await store.create_entry(
            KnowledgeEntry(
                id="facet_wrong_scope",
                text="unrelated wrong scope body",
                aspects=[entity_target],
            )
        )
        return await store.search(
            KnowledgeQuery(
                text=None,
                aspect_groups=[[scope_a, scope_b], [entity_target]],
                mode=KnowledgeSearchMode.KEYWORD,
            )
        )

    result = _run(postgres_dsn, ops)

    assert [hit.entry.id for hit in result.hits] == ["facet_target"]
    assert result.hits[0].score_kind == "exact_metadata"
    assert result.hits[0].reason == "exact aspect filter"


def test_postgres_embedding_auto_facet_only_search_uses_exact_metadata(
    postgres_dsn: str,
) -> None:
    async def ops():
        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        provider = KeywordEmbeddingProvider()
        store = _new_embedding_store(postgres_dsn, provider)
        try:
            await store.create_entry(
                KnowledgeEntry(
                    id="facet-target",
                    text="Semantically unrelated target body.",
                    aspects=["scope:b", "entity:target"],
                )
            )
            await store.create_entry(
                KnowledgeEntry(
                    id="facet-other",
                    text="Semantically unrelated other body.",
                    aspects=["scope:b", "entity:other"],
                )
            )
            provider.calls.clear()
            result = await store.search(
                KnowledgeQuery(
                    aspect_groups=[["scope:a", "scope:b"], ["entity:target"]],
                )
            )
        finally:
            await store.close()
        return result, provider.calls

    try:
        result, calls = asyncio.run(ops())
    finally:
        asyncio.run(_drop_all(postgres_dsn))

    assert [hit.entry.id for hit in result.hits] == ["facet-target"]
    assert result.query.mode is KnowledgeSearchMode.AUTO
    assert result.hits[0].score_kind == "exact_metadata"
    assert result.hits[0].reason == "exact aspect filter"
    assert calls == []


def test_postgres_knowledge_store_phrase_search_conformance(postgres_dsn: str) -> None:
    async def ops(store) -> None:
        await assert_token_exact_phrase_search_conformance(store)

    _run(postgres_dsn, ops)


def test_postgres_knowledge_store_applies_none_terms_to_the_complete_entry(
    postgres_dsn: str,
) -> None:
    async def ops(store) -> None:
        await assert_entry_wide_none_terms_conformance(
            store,
            mode=KnowledgeSearchMode.KEYWORD,
        )

    _run(postgres_dsn, ops)


def test_postgres_knowledge_store_filters_none_terms_before_chunk_pagination(
    postgres_dsn: str,
) -> None:
    async def ops(store) -> None:
        await assert_entry_wide_none_terms_precede_chunk_pagination(store)

    _run(postgres_dsn, ops)


@pytest.mark.parametrize(
    "mode",
    [KnowledgeSearchMode.SEMANTIC, KnowledgeSearchMode.HYBRID],
)
def test_postgres_embedding_store_applies_none_terms_to_the_complete_entry(
    postgres_dsn: str,
    mode: KnowledgeSearchMode,
) -> None:
    async def ops() -> None:
        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        store = _new_embedding_store(postgres_dsn, KeywordEmbeddingProvider())
        try:
            await assert_entry_wide_none_terms_conformance(
                store,
                mode=mode,
            )
        finally:
            await store.close()

    asyncio.run(ops())


def test_postgres_embedding_none_terms_do_not_consume_semantic_candidate_limit(
    postgres_dsn: str,
) -> None:
    async def ops() -> tuple[list[str], int | None, bool]:
        from cayu.storage.memory import (
            KNOWLEDGE_CHUNK_TEXT_GENERATOR,
            KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION,
            KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION,
            KNOWLEDGE_CHUNK_TEXT_PROJECTION,
            KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION,
        )
        from cayu.storage.postgres import (
            _PGVECTOR_SEMANTIC_CANDIDATE_MULTIPLIER,
            _postgres_knowledge_filter_sql,
            _postgres_knowledge_none_filter_sql,
            _postgres_vector_literal,
        )

        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        store = _new_embedding_store(
            postgres_dsn,
            KeywordEmbeddingProvider(),
            # Keep the planner controls and the store search on one session.
            max_size=1,
        )
        try:
            # Exceed pgvector's default HNSW candidate list so the exact safe
            # vector cannot be reached after nearer candidates are excluded.
            for index in range(64):
                await store.create_entry(
                    KnowledgeEntry(
                        id=f"excluded_{index}",
                        title="Deprecated integration",
                        text=f"GitHub credential instructions {index}.",
                    )
                )
            await store.create_entry(
                KnowledgeEntry(
                    id="safe",
                    text="Invoice payment instructions for the valid lower-ranked candidate.",
                )
            )
            await store.process_embedding_changes(
                "none-candidate-postgres-index",
                "worker",
                limit=100,
            )
            query = KnowledgeQuery(
                text="github",
                none_terms=["deprecated"],
                mode=KnowledgeSearchMode.SEMANTIC,
                min_score=0.5,
                limit=1,
            )
            where_sql, params = _postgres_knowledge_filter_sql(query)
            none_sql, none_params = _postgres_knowledge_none_filter_sql(query)
            vector_literal = _postgres_vector_literal(_test_embedding_vector("github"))
            candidate_limit = max(
                query.limit,
                query.limit * _PGVECTOR_SEMANTIC_CANDIDATE_MULTIPLIER,
            )
            hnsw_query = f"""
                SELECT e.id
                FROM cayu_knowledge_embeddings AS emb
                JOIN cayu_knowledge_chunks AS c
                  ON c.id = emb.chunk_id
                 AND c.entry_id = emb.entry_id
                 AND c.entry_revision = emb.entry_revision
                JOIN cayu_knowledge_current_entries AS e
                  ON e.id = emb.entry_id AND e.revision = c.entry_revision
                JOIN cayu_knowledge_index_readiness_current AS readiness_current
                  ON readiness_current.identity_sha256 = emb.identity_sha256
                JOIN cayu_knowledge_index_readiness_events AS readiness
                  ON readiness.sequence = readiness_current.sequence
                 AND readiness.identity_sha256 = emb.identity_sha256
                 AND readiness.state = 'ready'
                WHERE emb.projection_type = %s
                  AND emb.current_projection
                  AND emb.embedding_model = %s
                  AND emb.dimensions = %s
                  AND emb.preprocessing_version = %s
                  AND emb.generator = %s
                  AND emb.generator_version = %s
                  AND emb.index_representation_version = %s
                {where_sql}
                {none_sql}
                ORDER BY emb.embedding <=> %s::vector
                LIMIT %s
            """
            hnsw_params = [
                KNOWLEDGE_CHUNK_TEXT_PROJECTION,
                store.embedding_model,
                store.embedding_dimensions,
                KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION,
                KNOWLEDGE_CHUNK_TEXT_GENERATOR,
                KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION,
                KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION,
                *params,
                *none_params,
                vector_literal,
                candidate_limit,
            ]
            async with store._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute("ANALYZE cayu_knowledge_embeddings")
                # Make HNSW authoritative and shrink its internal candidate
                # list below the number of nearer excluded entries.
                await cur.execute("SET enable_seqscan = off")
                await cur.execute("SET enable_sort = off")
                await cur.execute("SET hnsw.ef_search = 1")
                await cur.execute("EXPLAIN " + hnsw_query, hnsw_params)
                plan = "\n".join(str(row[0]) for row in await cur.fetchall())
                await cur.execute(hnsw_query, hnsw_params)
                raw_hnsw_entry_ids = [str(row[0]) for row in await cur.fetchall()]
                await cur.execute("SET enable_sort = on")
            assert any(
                index_name in plan
                for index_name in (
                    store._embedding_hnsw_index_name(),
                    store._embedding_history_hnsw_index_name(),
                )
            )
            assert raw_hnsw_entry_ids == []

            result = await store.search(query)
            async with store._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute("SHOW enable_seqscan")
                assert (await cur.fetchone())[0] == "off"
                await cur.execute("SHOW enable_indexscan")
                assert (await cur.fetchone())[0] == "on"
                await cur.execute("SHOW enable_sort")
                assert (await cur.fetchone())[0] == "on"
                await cur.execute("SHOW hnsw.ef_search")
                assert (await cur.fetchone())[0] == "1"
        finally:
            await store.close()
        return (
            [hit.entry.id for hit in result.hits],
            result.total_hits_known,
            result.truncated,
        )

    assert asyncio.run(ops()) == (["safe"], 1, False)


def test_postgres_embedding_access_filters_cannot_hide_ready_hnsw_candidates(
    postgres_dsn: str,
) -> None:
    async def ops():
        from cayu import PostgresEmbeddingKnowledgeStore

        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        privileged = _new_embedding_store(postgres_dsn, KeywordEmbeddingProvider())
        try:
            for index in range(64):
                await privileged.create_entry(
                    KnowledgeEntry(
                        id=f"tenant-b-nearer-{index}",
                        namespace="tenant-b",
                        text=f"GitHub credential instructions {index}.",
                    )
                )
            await privileged.create_entry(
                KnowledgeEntry(
                    id="tenant-a-authorized",
                    namespace="tenant-a",
                    text="Invoice payment instructions for the authorized tenant.",
                )
            )
            await privileged.process_embedding_changes(
                "filtered-hnsw-postgres-index",
                "worker",
                limit=100,
            )
        finally:
            await privileged.close()

        store = PostgresEmbeddingKnowledgeStore(
            postgres_dsn,
            access_scope=KnowledgeAccessScope.for_namespace("tenant-a"),
            min_size=1,
            max_size=1,
            schema_mode=SchemaMode.CREATE,
            embedding_provider=KeywordEmbeddingProvider(),
            embedding_model="test-embedding",
            embedding_dimensions=3,
            semantic_min_score=0.0,
        )
        try:
            await store._ensure_ready()
            async with store._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute("ANALYZE cayu_knowledge_embeddings")
                await cur.execute("SET enable_seqscan = off")
                await cur.execute("SET enable_sort = off")
                await cur.execute("SET hnsw.ef_search = 1")
            result = await store.search(
                KnowledgeQuery(
                    text="github",
                    namespace="tenant-a",
                    mode=KnowledgeSearchMode.SEMANTIC,
                    min_score=0.0,
                    limit=1,
                )
            )
            async with store._pool.connection() as conn, conn.cursor() as cur:
                await cur.execute("SHOW enable_seqscan")
                enable_seqscan = str((await cur.fetchone())[0])
                await cur.execute("SHOW enable_indexscan")
                enable_indexscan = str((await cur.fetchone())[0])
        finally:
            await store.close()
            await _drop_all(postgres_dsn)
        return result, enable_seqscan, enable_indexscan

    result, enable_seqscan, enable_indexscan = asyncio.run(ops())

    assert [hit.entry.id for hit in result.hits] == ["tenant-a-authorized"]
    assert result.total_hits_known == 1
    assert result.truncated is False
    assert result.index_coverage[0].ready_records == 1
    assert result.index_coverage[0].complete is True
    assert enable_seqscan == "off"
    assert enable_indexscan == "on"


def test_postgres_embedding_search_reports_filtered_pending_coverage_without_backfill(
    postgres_dsn: str,
) -> None:
    async def ops():
        await _drop_all(postgres_dsn)
        base_store = _new_store(postgres_dsn)
        try:
            await base_store.create_entry(
                KnowledgeEntry(
                    id="excluded",
                    text="Integration summary.",
                    importance=1.0,
                ),
                [
                    KnowledgeChunk(
                        id="excluded:0",
                        entry_id="excluded",
                        chunk_index=0,
                        text="GitHub excluded-marker instructions.",
                    ),
                    KnowledgeChunk(
                        id="excluded:1",
                        entry_id="excluded",
                        chunk_index=1,
                        text="Deprecated proxy guidance.",
                    ),
                ],
            )
            await base_store.create_entry(
                KnowledgeEntry(
                    id="safe",
                    text="GitHub safe-marker instructions.",
                    importance=0.0,
                )
            )
        finally:
            await base_store.close()

        await _skip_if_pgvector_unavailable(postgres_dsn)
        provider = KeywordEmbeddingProvider()
        store = _new_embedding_store(postgres_dsn, provider)
        try:
            result = await store.search(
                KnowledgeQuery(
                    text="github",
                    none_terms=["deprecated"],
                    mode=KnowledgeSearchMode.SEMANTIC,
                )
            )
        finally:
            await store.close()
        return result, provider.calls

    result, calls = asyncio.run(ops())

    assert result.hits == []
    assert result.index_coverage[0].eligible_records == 1
    assert result.index_coverage[0].pending_records == 1
    assert result.index_coverage[0].complete is False
    assert calls == [["github"]]


def test_postgres_knowledge_store_searches_entry_text_with_custom_chunks(
    postgres_dsn: str,
) -> None:
    async def ops(store):
        await store.create_entry(
            KnowledgeEntry(
                id="broker_summary",
                text="Remote sandbox Git operations need a brokered credential boundary.",
            ),
            [
                KnowledgeChunk(
                    id="broker_summary:0",
                    entry_id="broker_summary",
                    chunk_index=0,
                    text="Implementation details live in the separate chunk body.",
                )
            ],
        )
        return await store.search(KnowledgeQuery(text="brokered credential"))

    result = _run(postgres_dsn, ops)

    assert [hit.entry.id for hit in result.hits] == ["broker_summary"]
    assert result.hits[0].reason == "entry text match"
    assert "brokered credential" in result.hits[0].text_preview


def test_postgres_knowledge_store_matches_singular_plural_token_variants(
    postgres_dsn: str,
) -> None:
    async def ops(store):
        await store.create_entry(
            KnowledgeEntry(
                id="remote_git",
                title="Remote sandbox Git credential boundary",
                text=(
                    "GitHub clone or push from a remote sandbox should use a brokered "
                    "proxy. The trusted side injects the credential outside the sandbox."
                ),
            )
        )
        await store.create_entry(
            KnowledgeEntry(
                id="fixture",
                text="Fixture credentials in local tests are not production guidance.",
            )
        )
        return await store.search(
            KnowledgeQuery(
                all_terms=["GitHub", "credentials"],
                any_terms=["sandbox", "push", "token"],
            )
        )

    result = _run(postgres_dsn, ops)

    assert [hit.entry.id for hit in result.hits] == ["remote_git"]


def test_postgres_knowledge_store_matches_y_plural_token_variants(
    postgres_dsn: str,
) -> None:
    async def ops(store):
        await store.create_entry(KnowledgeEntry(id="keys", text="Store API keys securely."))
        await store.create_entry(KnowledgeEntry(id="policies", text="Security policies apply."))
        key_result = await store.search(KnowledgeQuery(text="key"))
        policy_result = await store.search(KnowledgeQuery(text="policy"))
        return key_result, policy_result

    key_result, policy_result = _run(postgres_dsn, ops)

    assert [hit.entry.id for hit in key_result.hits] == ["keys"]
    assert [hit.entry.id for hit in policy_result.hits] == ["policies"]


def test_postgres_knowledge_store_all_terms_match_across_entry_document(
    postgres_dsn: str,
) -> None:
    async def ops(store):
        await store.create_entry(
            KnowledgeEntry(
                id="split_match",
                title="GitHub credential policy",
                text="Remote sandbox operations use a trusted boundary.",
            ),
            [
                KnowledgeChunk(
                    id="split_match:0",
                    entry_id="split_match",
                    chunk_index=0,
                    text="Use a brokered proxy for push operations.",
                )
            ],
        )
        return await store.search(KnowledgeQuery(all_terms=["github", "proxy"]))

    result = _run(postgres_dsn, ops)

    assert [hit.entry.id for hit in result.hits] == ["split_match"]


def test_postgres_knowledge_store_all_terms_do_not_match_across_unrelated_chunks(
    postgres_dsn: str,
) -> None:
    async def ops(store):
        await store.create_entry(
            KnowledgeEntry(id="split_chunks", text="General operations note."),
            [
                KnowledgeChunk(
                    id="split_chunks:0",
                    entry_id="split_chunks",
                    chunk_index=0,
                    text="GitHub push requires special handling.",
                ),
                KnowledgeChunk(
                    id="split_chunks:1",
                    entry_id="split_chunks",
                    chunk_index=1,
                    text="Use a brokered proxy for remote credentials.",
                ),
            ],
        )
        return await store.search(KnowledgeQuery(all_terms=["github", "proxy"]))

    result = _run(postgres_dsn, ops)

    assert result.hits == []


def test_postgres_knowledge_store_lists_entries_and_facets(postgres_dsn: str) -> None:
    async def ops(store):
        await store.create_entry(
            KnowledgeEntry(
                id="runbook",
                namespace="ops",
                kind="procedure",
                labels={"project": "billing"},
                text="Payment reminder runbook.",
            )
        )
        await store.create_entry(
            KnowledgeEntry(
                id="warning",
                namespace="ops",
                kind="warning",
                labels={"project": "billing"},
                text="Do not send reminders without approval.",
            )
        )
        await store.create_entry(
            KnowledgeEntry(
                id="archived",
                namespace="ops",
                kind="warning",
                status=KnowledgeStatus.ARCHIVED,
                text="Old warning.",
            )
        )
        return await store.list_entries(
            KnowledgeListQuery(
                namespace="ops",
                labels={"project": "billing"},
                group_by=KnowledgeListGroup.KIND,
            )
        )

    result = _run(postgres_dsn, ops)

    assert result.total_entries_known == 2
    assert {item.entry.id for item in result.entries} == {"runbook", "warning"}
    assert [(facet.value, facet.count) for facet in result.facets] == [
        ("procedure", 1),
        ("warning", 1),
    ]


def test_postgres_knowledge_store_caps_facets(postgres_dsn: str) -> None:
    async def ops(store):
        for index in range(5):
            await store.create_entry(
                KnowledgeEntry(
                    id=f"entry_{index}",
                    labels={"area": f"area_{index}"},
                    text=f"Knowledge entry {index}.",
                )
            )
        return await store.list_entries(
            KnowledgeListQuery(
                group_by=KnowledgeListGroup.LABEL,
                limit=3,
            )
        )

    result = _run(postgres_dsn, ops)

    assert len(result.facets) == 3
    assert result.facets_truncated is True
    assert result.truncated is True


def test_postgres_knowledge_store_chunk_windows_and_truncation(postgres_dsn: str) -> None:
    async def ops(store):
        await store.create_entry(
            KnowledgeEntry(id="doc", text="summary"),
            [
                KnowledgeChunk(id="chunk_0", entry_id="doc", chunk_index=0, text="alpha beta"),
                KnowledgeChunk(
                    id="chunk_1",
                    entry_id="doc",
                    chunk_index=1,
                    text="gamma delta",
                    content_hash="full-hash",
                ),
                KnowledgeChunk(id="chunk_2", entry_id="doc", chunk_index=2, text="epsilon zeta"),
            ],
        )
        window = await store.read_chunks("doc", chunk_index=1, around=1, max_chunks=3)
        centered = await store.read_chunks("doc", chunk_index=2, around=10, max_chunks=1)
        truncated = await store.read_chunks("doc", chunk_index=1, around=0, max_bytes=5)
        return window, centered, truncated

    window, centered, truncated = _run(postgres_dsn, ops)

    assert [chunk.id for chunk in window] == ["chunk_0", "chunk_1", "chunk_2"]
    assert [chunk.id for chunk in centered] == ["chunk_2"]
    assert truncated[0].text == "gamma"
    assert truncated[0].content_hash is None


def test_postgres_knowledge_store_title_match_uses_title_preview(postgres_dsn: str) -> None:
    async def ops(store):
        await store.create_entry(
            KnowledgeEntry(
                id="title_match",
                title="Invoice approval warning",
                text="The body does not include the searched approval terms.",
            )
        )
        return await store.search(KnowledgeQuery(text="invoice approval"))

    result = _run(postgres_dsn, ops)

    assert [hit.entry.id for hit in result.hits] == ["title_match"]
    assert result.hits[0].reason == "title match"
    assert result.hits[0].text_preview == "Invoice approval warning"


def test_postgres_knowledge_store_updates_status_and_deletes_entries(
    postgres_dsn: str,
) -> None:
    async def ops(store):
        await store.create_entry(
            KnowledgeEntry(
                id="pending_runbook",
                text="deployment rollback procedure",
                namespace="project:cayu",
                labels={"project": "cayu"},
                status=KnowledgeStatus.PENDING,
            )
        )
        active = await store.transition_entry_status(
            "pending_runbook",
            expected_revision=1,
            from_status=KnowledgeStatus.PENDING,
            to_status=KnowledgeStatus.ACTIVE,
            expected_namespace="project:cayu",
            expected_labels={"project": "cayu"},
        )
        with pytest.raises(ValueError, match="not 'pending'"):
            await store.transition_entry_status(
                "pending_runbook",
                expected_revision=active.revision,
                from_status=KnowledgeStatus.PENDING,
                to_status=KnowledgeStatus.ARCHIVED,
                expected_namespace="project:cayu",
                expected_labels={"project": "cayu"},
            )
        await store.create_entry(
            KnowledgeEntry(
                id="pending_other",
                text="other project procedure",
                namespace="project:other",
                labels={"project": "other"},
                status=KnowledgeStatus.PENDING,
            )
        )
        with pytest.raises(ValueError, match="expected namespace"):
            await store.transition_entry_status(
                "pending_other",
                expected_revision=1,
                from_status=KnowledgeStatus.PENDING,
                to_status=KnowledgeStatus.ACTIVE,
                expected_namespace="project:cayu",
            )
        with pytest.raises(ValueError, match="expected labels"):
            await store.transition_entry_status(
                "pending_other",
                expected_revision=1,
                from_status=KnowledgeStatus.PENDING,
                to_status=KnowledgeStatus.ACTIVE,
                expected_labels={"project": "cayu"},
            )
        runbook = await store.create_entry(
            KnowledgeEntry(id="runbook", text="deployment rollback procedure")
        )
        archived = await store.transition_entry_status(
            "runbook",
            expected_revision=runbook.revision,
            from_status=KnowledgeStatus.ACTIVE,
            to_status=KnowledgeStatus.ARCHIVED,
        )
        archived_search = await store.search(
            KnowledgeQuery(text="deployment", statuses=[KnowledgeStatus.ARCHIVED])
        )
        soft_deleted = await store.delete_entry(
            "runbook",
            expected_revision=archived.revision,
        )
        deleted_search = await store.search(
            KnowledgeQuery(text="deployment", statuses=[KnowledgeStatus.DELETED])
        )
        assert soft_deleted is not None
        hard_deleted = await store.delete_entry(
            "runbook",
            expected_revision=soft_deleted.revision,
            hard=True,
        )
        missing = await store.get_entry("runbook")
        missing_delete = await store.delete_entry("runbook", expected_revision=1, hard=True)
        return (
            active,
            archived,
            archived_search,
            soft_deleted,
            deleted_search,
            hard_deleted,
            missing,
            missing_delete,
        )

    (
        active,
        archived,
        archived_search,
        soft_deleted,
        deleted_search,
        hard_deleted,
        missing,
        missing_delete,
    ) = _run(postgres_dsn, ops)

    assert active.status is KnowledgeStatus.ACTIVE
    assert archived.status is KnowledgeStatus.ARCHIVED
    assert [hit.entry.id for hit in archived_search.hits] == ["runbook"]
    assert soft_deleted is not None
    assert soft_deleted.status is KnowledgeStatus.DELETED
    assert [hit.entry.id for hit in deleted_search.hits] == ["runbook"]
    assert hard_deleted is not None
    assert hard_deleted.status is KnowledgeStatus.DELETED
    assert missing is None
    assert missing_delete is None


def test_postgres_knowledge_store_rejects_invalid_revision_chunks(
    postgres_dsn: str,
) -> None:
    async def ops(store):
        current = await store.create_entry(KnowledgeEntry(id="entry", text="text"))
        successor = current.model_copy(update={"revision": 2})
        with pytest.raises(ValueError, match="cannot be empty"):
            await store.append_entry_revision(successor, [], expected_revision=1)
        with pytest.raises(ValueError, match="belong"):
            await store.append_entry_revision(
                successor,
                [
                    KnowledgeChunk(
                        id="chunk",
                        entry_id="other",
                        entry_revision=2,
                        chunk_index=0,
                        text="text",
                    )
                ],
                expected_revision=1,
            )
        with pytest.raises(ValueError, match="ids"):
            await store.append_entry_revision(
                successor,
                [
                    KnowledgeChunk(
                        id="chunk",
                        entry_id="entry",
                        entry_revision=2,
                        chunk_index=0,
                        text="first",
                    ),
                    KnowledgeChunk(
                        id="chunk",
                        entry_id="entry",
                        entry_revision=2,
                        chunk_index=1,
                        text="second",
                    ),
                ],
                expected_revision=1,
            )

    _run(postgres_dsn, ops)


def test_postgres_knowledge_schema_rejects_a_dangling_current_revision(
    postgres_dsn: str,
) -> None:
    async def ops() -> int:
        import psycopg
        from psycopg.errors import ForeignKeyViolation

        await _drop_all(postgres_dsn)
        store = _new_store(postgres_dsn)
        try:
            await store.create_entry(KnowledgeEntry(id="entry", text="current"))
        finally:
            await store.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            with pytest.raises(ForeignKeyViolation):
                async with connection.transaction(), connection.cursor() as cursor:
                    await cursor.execute(
                        "UPDATE cayu_knowledge_entries SET current_revision = %s WHERE id = %s",
                        (2, "entry"),
                    )
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT current_revision FROM cayu_knowledge_entries WHERE id = %s",
                    ("entry",),
                )
                row = await cursor.fetchone()
                assert row is not None
                return int(row[0])

    assert asyncio.run(ops()) == 1


@pytest.mark.parametrize(
    ("object_name", "drop_sql"),
    (
        (
            "cayu_knowledge_current_entries",
            "DROP VIEW cayu_knowledge_current_entries",
        ),
        (
            "idx_cayu_knowledge_revisions_status",
            "DROP INDEX idx_cayu_knowledge_revisions_status",
        ),
        (
            "cayu_knowledge_entries",
            "ALTER TABLE cayu_knowledge_entries "
            "DROP CONSTRAINT cayu_knowledge_entries_current_revision_fk",
        ),
    ),
)
def test_postgres_revision_schema_validation_rejects_missing_structural_objects(
    postgres_dsn: str,
    object_name: str,
    drop_sql: str,
) -> None:
    async def ops() -> None:
        import psycopg

        from cayu import PostgresKnowledgeStore

        await _drop_all(postgres_dsn)
        create_store = _new_store(postgres_dsn)
        try:
            await create_store.ensure_schema()
        finally:
            await create_store.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute(drop_sql)
            await connection.commit()

        validate_store = PostgresKnowledgeStore(
            postgres_dsn,
            access_scope=_ACCESS_SCOPE,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
        )
        try:
            with pytest.raises(RuntimeError, match=object_name):
                await validate_store.ensure_schema()
        finally:
            await validate_store.close()
            await _drop_all(postgres_dsn)

    asyncio.run(ops())


def test_postgres_knowledge_store_rejects_unsupported_search_modes(
    postgres_dsn: str,
) -> None:
    async def ops(store):
        await store.create_entry(KnowledgeEntry(id="entry", text="billing memory"))
        with pytest.raises(ValueError, match="supports only auto and keyword"):
            await store.search(KnowledgeQuery(text="billing", mode=KnowledgeSearchMode.SEMANTIC))

    _run(postgres_dsn, ops)


def test_postgres_knowledge_schema_migrates_and_coexists_with_session_store(
    postgres_dsn: str,
) -> None:
    async def ops():
        import psycopg

        from cayu import PostgresKnowledgeStore, PostgresSessionStore
        from cayu.core import Message
        from cayu.runtime import RunRequest, SessionIdentity

        await _drop_all(postgres_dsn)
        session_store = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await session_store.create(
                RunRequest(agent_name="assistant", messages=[Message.text("user", "hi")]),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
        finally:
            await session_store.close()

        knowledge_store = PostgresKnowledgeStore(
            postgres_dsn,
            access_scope=_ACCESS_SCOPE,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await knowledge_store.create_entry(
                KnowledgeEntry(id="entry", text="shared database memory")
            )
            result = await knowledge_store.search(KnowledgeQuery(text="shared database"))
        finally:
            await knowledge_store.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT revision, compatible_from FROM cayu_schema_migrations ORDER BY revision"
            )
            revisions = [(int(row[0]), int(row[1])) for row in await cur.fetchall()]
            await cur.execute("SELECT to_regclass('cayu_knowledge_entries')")
            knowledge_row = await cur.fetchone()
            assert knowledge_row is not None
            knowledge_table = knowledge_row[0]
            await cur.execute("SELECT to_regclass('cayu_knowledge_chunks')")
            chunks_row = await cur.fetchone()
            assert chunks_row is not None
            chunks_table = chunks_row[0]
            await cur.execute("SELECT to_regclass('cayu_knowledge_publication_receipts')")
            receipts_row = await cur.fetchone()
            assert receipts_row is not None
            receipts_table = receipts_row[0]
        return result, revisions, knowledge_table, chunks_table, receipts_table

    result, revisions, knowledge_table, chunks_table, receipts_table = asyncio.run(ops())

    assert [hit.entry.id for hit in result.hits] == ["entry"]
    assert revisions[-1] == (LATEST_REVISION, MIN_SUPPORTED_REVISION)
    assert knowledge_table == "cayu_knowledge_entries"
    assert chunks_table == "cayu_knowledge_chunks"
    assert receipts_table == "cayu_knowledge_publication_receipts"


def test_postgres_knowledge_store_batches_multi_entry_hit_hydration(postgres_dsn: str) -> None:
    async def ops(store):
        for index in range(3):
            await store.create_entry(
                KnowledgeEntry(
                    id=f"entry_{index}",
                    text=f"Shared deployment warning number {index}.",
                    labels={"project": f"proj_{index}", "shared": "yes"},
                    aspects=[f"aspect_{index}"],
                    impact_targets=[f"target_{index}"],
                )
            )
        return await store.search(KnowledgeQuery(text="deployment warning", limit=10))

    result = _run(postgres_dsn, ops)

    # Batched hydration must keep per-entry label/aspect/impact lists grouped by
    # entry rather than cross-contaminating across hits.
    by_entry = {hit.entry.id: hit for hit in result.hits}
    assert set(by_entry) == {"entry_0", "entry_1", "entry_2"}
    for index in range(3):
        hit = by_entry[f"entry_{index}"]
        assert hit.entry.labels == {"project": f"proj_{index}", "shared": "yes"}
        assert hit.entry.aspects == [f"aspect_{index}"]
        assert hit.entry.impact_targets == [f"target_{index}"]
        assert hit.chunk is not None
        assert hit.chunk.entry_id == f"entry_{index}"


def test_postgres_knowledge_store_list_reports_multi_chunk_counts(postgres_dsn: str) -> None:
    async def ops(store):
        await store.create_entry(KnowledgeEntry(id="single", text="Single chunk entry."))
        await store.create_entry(
            KnowledgeEntry(id="multi", text="Multi chunk entry."),
            [
                KnowledgeChunk(
                    id=f"multi:{index}",
                    entry_id="multi",
                    chunk_index=index,
                    text=f"Body part {index}.",
                )
                for index in range(3)
            ],
        )
        return await store.list_entries(KnowledgeListQuery(limit=10))

    result = _run(postgres_dsn, ops)

    counts = {item.entry.id: item.chunk_count for item in result.entries}
    assert counts == {"single": 1, "multi": 3}


async def _count_embeddings(dsn: str) -> int:
    import psycopg

    async with (
        await psycopg.AsyncConnection.connect(dsn) as conn,
        conn.cursor() as cur,
    ):
        await cur.execute("SELECT COUNT(*) FROM cayu_knowledge_embeddings")
        row = await cur.fetchone()
    return 0 if row is None else int(row[0])


def test_postgres_knowledge_store_prune_expired_hard_deletes(postgres_dsn: str) -> None:
    # MEM-05: prune_expired hard-deletes expired entries; the read filter only hides them.
    async def ops(store):
        await store.create_entry(KnowledgeEntry(id="active", text="deployment warning"))
        await store.create_entry(
            KnowledgeEntry(
                id="expired",
                text="deployment warning",
                expires_at=datetime.now(UTC) - timedelta(seconds=1),
            )
        )
        pruned = await store.prune_expired()
        leftover = await store.search(KnowledgeQuery(text="deployment", include_expired=True))
        return pruned, [hit.entry.id for hit in leftover.hits], await store.get_entry("expired")

    pruned, leftover_ids, expired_entry = _run(postgres_dsn, ops)

    assert pruned == 1
    assert expired_entry is None
    assert leftover_ids == ["active"]


def test_postgres_embedding_store_prune_expired_cascades_to_embeddings(postgres_dsn: str) -> None:
    # MEM-05: the embedding subclass inherits prune_expired; the entries FK cascade must also drop
    # the vectors from cayu_knowledge_embeddings (no explicit override needed).
    async def ops():
        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        store = _new_embedding_store(postgres_dsn, KeywordEmbeddingProvider())
        try:
            await store.create_entry(
                KnowledgeEntry(
                    id="expired",
                    text="GitHub credential proxy runbook.",
                    expires_at=datetime.now(UTC) - timedelta(seconds=1),
                )
            )
            await store.process_embedding_changes("prune-postgres-index", "worker")
            before = await _count_embeddings(postgres_dsn)
            pruned = await store.prune_expired()
            after = await _count_embeddings(postgres_dsn)
        finally:
            await store.close()
        return before, pruned, after

    before, pruned, after = asyncio.run(ops())

    assert before == 1
    assert pruned == 1
    assert after == 0


def test_postgres_embedding_store_persists_complete_projection_identity(
    postgres_dsn: str,
) -> None:
    async def ops():
        import psycopg

        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        store = _new_embedding_store(postgres_dsn, KeywordEmbeddingProvider())
        try:
            await store.create_entry(
                KnowledgeEntry(id="doc", text="GitHub credential proxy runbook.")
            )
            await store.process_embedding_changes("identity-postgres-index", "worker")
            result = await store.search(
                KnowledgeQuery(text="auth broker", mode=KnowledgeSearchMode.SEMANTIC)
            )
        finally:
            await store.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                """
                SELECT DISTINCT projection_type, embedding_model, dimensions,
                       preprocessing_version, generator, generator_version,
                       index_representation_version, entry_revision
                FROM cayu_knowledge_embeddings
                """
            )
            identities = await cur.fetchall()
        return [hit.entry.id for hit in result.hits], identities

    hit_ids, identities = asyncio.run(ops())

    assert hit_ids == ["doc"]
    assert identities == [
        (
            "knowledge_chunk_text",
            "test-embedding",
            3,
            "cayu:knowledge-chunk-text:v1",
            "cayu:canonical-knowledge-chunk",
            "1",
            "float32-cosine-v1",
            1,
        )
    ]


async def _distinct_embedding_models(dsn: str) -> list[str]:
    import psycopg

    async with (
        await psycopg.AsyncConnection.connect(dsn) as conn,
        conn.cursor() as cur,
    ):
        await cur.execute("SELECT DISTINCT embedding_model FROM cayu_knowledge_embeddings")
        return sorted(str(row[0]) for row in await cur.fetchall())


async def _embedding_hnsw_predicates(dsn: str) -> list[tuple[str, str]]:
    import psycopg

    async with (
        await psycopg.AsyncConnection.connect(dsn) as conn,
        conn.cursor() as cur,
    ):
        await cur.execute(
            """
            SELECT index_record.relname,
                   pg_get_expr(index_state.indpred, index_state.indrelid)
            FROM pg_catalog.pg_index AS index_state
            JOIN pg_catalog.pg_class AS index_record
              ON index_record.oid = index_state.indexrelid
            JOIN pg_catalog.pg_class AS table_record
              ON table_record.oid = index_state.indrelid
            JOIN pg_catalog.pg_am AS access_method
              ON access_method.oid = index_record.relam
            WHERE table_record.relname = 'cayu_knowledge_embeddings'
              AND access_method.amname = 'hnsw'
            ORDER BY index_record.relname
            """
        )
        return [(str(row[0]), str(row[1])) for row in await cur.fetchall()]


def test_postgres_embedding_store_segregates_models_until_explicit_reindex(
    postgres_dsn: str,
) -> None:
    async def ops():
        from cayu import PostgresEmbeddingKnowledgeStore

        await _drop_all(postgres_dsn)
        await _skip_if_pgvector_unavailable(postgres_dsn)
        store = _new_embedding_store(postgres_dsn, KeywordEmbeddingProvider())
        try:
            await store.create_entry(
                KnowledgeEntry(id="doc", text="GitHub credential proxy runbook.")
            )
            await store.process_embedding_changes("model-v1-postgres-index", "worker")
        finally:
            await store.close()

        models_before = await _distinct_embedding_models(postgres_dsn)
        other = PostgresEmbeddingKnowledgeStore(
            postgres_dsn,
            access_scope=_ACCESS_SCOPE,
            schema_mode=SchemaMode.CREATE,
            embedding_provider=KeywordEmbeddingProvider(),
            embedding_model="other-embedding-model",
            embedding_dimensions=3,
        )
        try:
            query = KnowledgeQuery(text="auth broker", mode=KnowledgeSearchMode.SEMANTIC)
            before_reindex = await other.search(query)
            backfill = await other.backfill_embeddings()
            after_reindex = await other.search(query)
        finally:
            await other.close()
        return (
            models_before,
            before_reindex,
            backfill,
            after_reindex,
            await _distinct_embedding_models(postgres_dsn),
            await _embedding_hnsw_predicates(postgres_dsn),
        )

    (
        models_before,
        before_reindex,
        backfill,
        after_reindex,
        models_after,
        hnsw_indexes,
    ) = asyncio.run(ops())

    assert models_before == ["test-embedding"]
    assert before_reindex.hits == []
    assert before_reindex.index_coverage[0].pending_records == 1
    assert backfill.indexed_records == 1
    assert [hit.entry.id for hit in after_reindex.hits] == ["doc"]
    assert models_after == ["other-embedding-model", "test-embedding"]
    assert len(hnsw_indexes) == 4
    assert len({name for name, _ in hnsw_indexes}) == 4
    assert sum("_history_" in name for name, _ in hnsw_indexes) == 2
    assert sum("current_projection" in predicate for _, predicate in hnsw_indexes) == 2
    assert all(
        "embedding_model =" in predicate
        and "dimensions = 3" in predicate
        and "generator_version =" in predicate
        and "index_representation_version =" in predicate
        for _, predicate in hnsw_indexes
    )


def test_postgres_revision_60_refuses_populated_knowledge_without_backfill(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresKnowledgeStore

        await _drop_all(postgres_dsn)
        await _initialize_historical_schema(postgres_dsn, through_revision=59)
        entry = KnowledgeEntry(
            id="revision-59-entry",
            text="Revision 60 must not rewrite this populated store.",
        )
        await _insert_pre_revision_65_entry(postgres_dsn, entry)

        migrator = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.MIGRATE,
            access_scope=_ACCESS_SCOPE,
        )
        try:
            with pytest.raises(
                schema_migrations.SchemaTooOld,
                match="clean prerelease knowledge-lineage break",
            ):
                await migrator.ensure_schema()
        finally:
            await migrator.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute("SELECT MAX(revision) FROM cayu_schema_migrations")
            assert await cursor.fetchone() == (59,)
            await cursor.execute(
                "SELECT text FROM cayu_knowledge_revisions WHERE entry_id = %s AND revision = 1",
                (entry.id,),
            )
            assert await cursor.fetchone() == (entry.text,)
            await cursor.execute("SELECT to_regclass('cayu_knowledge_relations')")
            assert await cursor.fetchone() == (None,)

    try:
        asyncio.run(run())
    finally:
        asyncio.run(_drop_all(postgres_dsn))


def test_postgres_revision_60_initializes_empty_pre_relation_schema_directly(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresKnowledgeStore

        await _drop_all(postgres_dsn)
        await _initialize_historical_schema(postgres_dsn, through_revision=59)
        migrator = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.MIGRATE,
            access_scope=_ACCESS_SCOPE,
        )
        try:
            await migrator.ensure_schema()
        finally:
            await migrator.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute("SELECT MAX(revision) FROM cayu_schema_migrations")
            assert await cursor.fetchone() == (schema_migrations.LATEST_REVISION,)
            await cursor.execute(
                "SELECT to_regclass('cayu_knowledge_relations'), "
                "to_regclass('cayu_knowledge_relation_publication_receipts'), "
                "to_regclass('cayu_knowledge_maintenance_decisions'), "
                "to_regclass('cayu_knowledge_maintenance_proposals')"
            )
            assert await cursor.fetchone() == (
                "cayu_knowledge_relations",
                "cayu_knowledge_relation_publication_receipts",
                "cayu_knowledge_maintenance_decisions",
                "cayu_knowledge_maintenance_proposals",
            )

    try:
        asyncio.run(run())
    finally:
        asyncio.run(_drop_all(postgres_dsn))


def test_postgres_revision_63_refuses_populated_knowledge_without_interpretation(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresKnowledgeStore

        await _drop_all(postgres_dsn)
        await _initialize_historical_schema(postgres_dsn, through_revision=62)
        entry = KnowledgeEntry(
            id="revision-62-entry",
            text="Revision 63 must not interpret this populated store.",
        )
        await _insert_pre_revision_65_entry(postgres_dsn, entry)

        migrator = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.MIGRATE,
            access_scope=_ACCESS_SCOPE,
        )
        try:
            with pytest.raises(
                schema_migrations.SchemaTooOld,
                match="clean prerelease reviewed-maintenance break",
            ):
                await migrator.ensure_schema()
        finally:
            await migrator.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute("SELECT MAX(revision) FROM cayu_schema_migrations")
            assert await cursor.fetchone() == (62,)
            await cursor.execute(
                "SELECT text FROM cayu_knowledge_revisions WHERE entry_id = %s AND revision = 1",
                (entry.id,),
            )
            assert await cursor.fetchone() == (entry.text,)
            await cursor.execute("SELECT to_regclass('cayu_knowledge_maintenance_decisions')")
            assert await cursor.fetchone() == (None,)

    try:
        asyncio.run(run())
    finally:
        asyncio.run(_drop_all(postgres_dsn))


def test_postgres_revision_63_initializes_empty_knowledge_schema_directly(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresKnowledgeStore

        await _drop_all(postgres_dsn)
        await _initialize_historical_schema(postgres_dsn, through_revision=62)
        migrator = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.MIGRATE,
            access_scope=_ACCESS_SCOPE,
        )
        try:
            await migrator.ensure_schema()
        finally:
            await migrator.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute("SELECT MAX(revision) FROM cayu_schema_migrations")
            assert await cursor.fetchone() == (schema_migrations.LATEST_REVISION,)
            await cursor.execute(
                "SELECT to_regclass('cayu_knowledge_maintenance_decisions'), "
                "to_regclass('cayu_knowledge_maintenance_proposals')"
            )
            assert await cursor.fetchone() == (
                "cayu_knowledge_maintenance_decisions",
                "cayu_knowledge_maintenance_proposals",
            )

    try:
        asyncio.run(run())
    finally:
        asyncio.run(_drop_all(postgres_dsn))


def test_postgres_revision_65_refuses_populated_knowledge_without_backfill(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresKnowledgeStore

        await _drop_all(postgres_dsn)
        await _initialize_historical_schema(postgres_dsn, through_revision=64)
        entry = KnowledgeEntry(id="revision-64-entry", text="Must remain untouched.")
        await _insert_pre_revision_65_entry(postgres_dsn, entry)

        migrator = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.MIGRATE,
            access_scope=_ACCESS_SCOPE,
        )
        try:
            with pytest.raises(
                schema_migrations.SchemaTooOld,
                match="clean prerelease bounded-entry-read break",
            ):
                await migrator.ensure_schema()
        finally:
            await migrator.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute("SELECT MAX(revision) FROM cayu_schema_migrations")
            assert await cursor.fetchone() == (64,)
            await cursor.execute(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'cayu_knowledge_revisions'
                  AND column_name = 'payload_bytes'
                """
            )
            assert await cursor.fetchone() is None
            await cursor.execute(
                "SELECT text FROM cayu_knowledge_revisions WHERE entry_id = %s AND revision = 1",
                (entry.id,),
            )
            assert await cursor.fetchone() == (entry.text,)

    try:
        asyncio.run(run())
    finally:
        asyncio.run(_drop_all(postgres_dsn))


def test_postgres_revision_67_adds_empty_proposal_storage_without_backfill(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresKnowledgeStore

        await _drop_all(postgres_dsn)
        creator = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
            access_scope=_ACCESS_SCOPE,
        )
        try:
            await creator.ensure_schema()
            await creator.create_entry(
                KnowledgeEntry(
                    id="revision-66-entry",
                    text="Preserve this exact revision.",
                )
            )
        finally:
            await creator.close()
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute("DROP TABLE cayu_knowledge_maintenance_proposals")
                await cursor.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 67")
            await connection.commit()

        migrator = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.MIGRATE,
            access_scope=_ACCESS_SCOPE,
        )
        try:
            await migrator.ensure_schema()
        finally:
            await migrator.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute("SELECT MAX(revision) FROM cayu_schema_migrations")
            assert await cursor.fetchone() == (schema_migrations.LATEST_REVISION,)
            await cursor.execute(
                "SELECT text FROM cayu_knowledge_revisions "
                "WHERE entry_id = 'revision-66-entry' AND revision = 1"
            )
            assert await cursor.fetchone() == ("Preserve this exact revision.",)
            await cursor.execute("SELECT COUNT(*) FROM cayu_knowledge_maintenance_proposals")
            assert await cursor.fetchone() == (0,)

    try:
        asyncio.run(run())
    finally:
        asyncio.run(_drop_all(postgres_dsn))


def test_postgres_revision_67_rejects_malformed_proposal_storage(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresKnowledgeStore

        await _drop_all(postgres_dsn)
        creator = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
            access_scope=_ACCESS_SCOPE,
        )
        try:
            await creator.ensure_schema()
        finally:
            await creator.close()
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute("DROP TABLE cayu_knowledge_maintenance_proposals")
                await cursor.execute(
                    "CREATE TABLE cayu_knowledge_maintenance_proposals "
                    "(operation_id TEXT PRIMARY KEY)"
                )
            await connection.commit()

        validator = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
            access_scope=_ACCESS_SCOPE,
        )
        try:
            with pytest.raises(RuntimeError, match="pending maintenance proposal contract"):
                await validator.ensure_schema()
        finally:
            await validator.close()

    try:
        asyncio.run(run())
    finally:
        asyncio.run(_drop_all(postgres_dsn))


def test_postgres_revision_63_rejects_a_malformed_maintenance_table(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresKnowledgeStore

        await _drop_all(postgres_dsn)
        creator = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
            access_scope=_ACCESS_SCOPE,
        )
        try:
            await creator.ensure_schema()
        finally:
            await creator.close()
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute("DROP TABLE cayu_knowledge_maintenance_decisions")
                await cursor.execute(
                    "CREATE TABLE cayu_knowledge_maintenance_decisions "
                    "(operation_id TEXT PRIMARY KEY)"
                )
            await connection.commit()

        validator = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
            access_scope=_ACCESS_SCOPE,
        )
        try:
            with pytest.raises(RuntimeError, match="reviewed knowledge maintenance contract"):
                await validator.ensure_schema()
        finally:
            await validator.close()

    try:
        asyncio.run(run())
    finally:
        asyncio.run(_drop_all(postgres_dsn))


def test_postgres_revision_43_preserves_revision_42_knowledge_without_fabricated_changes(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresKnowledgeStore
        from cayu.storage import postgres as postgres_storage

        await _drop_all(postgres_dsn)
        await _initialize_historical_schema(postgres_dsn, through_revision=42)

        timestamp = datetime(2026, 8, 18, 9, 0, tzinfo=UTC)
        entry = KnowledgeEntry(
            id="preserved-entry",
            text="Revision 42 knowledge survives.",
            labels={"project": "cayu"},
            created_at=timestamp,
            updated_at=timestamp,
        )
        chunk = KnowledgeChunk(
            id="preserved-entry:r1:0",
            entry_id=entry.id,
            entry_revision=1,
            chunk_index=0,
            text=entry.text,
        )
        operation_id = "preserved-revision-42-publication"
        request_sha256 = _knowledge_publication_v1_request_sha256(
            entry,
            [chunk],
            expected_revision=None,
        )
        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute(
                """
                INSERT INTO cayu_knowledge_entries (
                    id, namespace, current_revision, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s)
                """,
                (entry.id, entry.namespace, 1, timestamp, timestamp),
            )
            await cursor.execute(
                """
                INSERT INTO cayu_knowledge_revisions (
                    entry_id, revision, text, kind, visibility, status,
                    created_by_type, created_by, created_at, updated_at,
                    source_type, source_uri, source_id, source_hash,
                    importance, importance_source, confidence, last_used_at,
                    expires_at, title, metadata
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb
                )
                """,
                postgres_storage._knowledge_entry_row_values(entry)[:-1],
            )
            await cursor.execute(
                """
                INSERT INTO cayu_knowledge_labels (
                    entry_id, entry_revision, key, value
                ) VALUES (%s, %s, %s, %s)
                """,
                (entry.id, 1, "project", "cayu"),
            )
            await cursor.execute(
                """
                INSERT INTO cayu_knowledge_chunks (
                    id, entry_id, entry_revision, chunk_index,
                    text, content_hash, source_uri, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                postgres_storage._knowledge_chunk_row_values(chunk),
            )
            await cursor.execute(
                """
                INSERT INTO cayu_knowledge_publication_receipts (
                    operation_id, entry_id, entry_revision, expected_revision,
                    request_sha256, entry_created_at, entry_updated_at,
                    committed_at, access_snapshot
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    operation_id,
                    entry.id,
                    entry.revision,
                    None,
                    request_sha256,
                    entry.created_at,
                    entry.updated_at,
                    timestamp,
                    _knowledge_access_snapshot_json(_knowledge_access_snapshot(entry)),
                ),
            )
            await connection.commit()

        await _initialize_historical_schema(postgres_dsn, through_revision=43)
        store = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
            access_scope=_ACCESS_SCOPE,
        )
        store._min_required_revision = 43
        try:
            replay = await store.publish_entry_revision(
                entry,
                [chunk],
                operation_id=operation_id,
            )
            assert replay.replayed is True
            assert replay.committed_at == timestamp
            assert await store.get_entry(entry.id) == entry
            assert await store.read_chunks(entry.id) == [chunk]
            evidence = await store.read_evidence(entry.id)
            assert evidence is not None
            assert evidence.evidence == []
            assert evidence.total_evidence_known == 0
            async with (
                await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
                connection.cursor() as cursor,
            ):
                await cursor.execute("SELECT COUNT(*) FROM cayu_knowledge_changes")
                assert await cursor.fetchone() == (0,)
        finally:
            await store.close()
            await _drop_all(postgres_dsn)

    asyncio.run(run())


def test_postgres_revision_43_preserves_migrated_expiration_cleanup_audiences(
    postgres_dsn: str,
    monkeypatch,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresKnowledgeStore
        from cayu.storage import postgres as postgres_storage

        await _drop_all(postgres_dsn)
        await _initialize_historical_schema(postgres_dsn, through_revision=42)

        baseline = datetime.now(UTC)
        future_expiry = baseline + timedelta(hours=1)
        entries = (
            KnowledgeEntry(
                id="migrated-future-expiry",
                text="Visible when the outbox baseline was established.",
                labels={"expiry": "future"},
                created_at=baseline,
                updated_at=baseline,
                expires_at=future_expiry,
            ),
            KnowledgeEntry(
                id="migrated-past-expiry",
                text="Already expired when the outbox baseline was established.",
                labels={"expiry": "past"},
                created_at=baseline - timedelta(hours=2),
                updated_at=baseline - timedelta(hours=2),
                expires_at=baseline - timedelta(hours=1),
            ),
        )
        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            for entry in entries:
                await cursor.execute(
                    """
                    INSERT INTO cayu_knowledge_entries (
                        id, namespace, current_revision, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s)
                    """,
                    (entry.id, entry.namespace, 1, entry.created_at, entry.updated_at),
                )
                await cursor.execute(
                    """
                    INSERT INTO cayu_knowledge_revisions (
                        entry_id, revision, text, kind, visibility, status,
                        created_by_type, created_by, created_at, updated_at,
                        source_type, source_uri, source_id, source_hash,
                        importance, importance_source, confidence, last_used_at,
                        expires_at, title, metadata
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb
                    )
                    """,
                    postgres_storage._knowledge_entry_row_values(entry)[:-1],
                )
                await cursor.execute(
                    """
                    INSERT INTO cayu_knowledge_labels (
                        entry_id, entry_revision, key, value
                    ) VALUES (%s, %s, %s, %s)
                    """,
                    (entry.id, 1, "expiry", entry.labels["expiry"]),
                )
            await connection.commit()

        await _initialize_historical_schema(postgres_dsn, through_revision=43)
        store = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
            access_scope=None,
        )
        store._min_required_revision = 43
        future_scope = KnowledgeAccessScope.for_namespace(
            "default",
            required_labels={"expiry": "future"},
        )
        past_scope = KnowledgeAccessScope.for_namespace(
            "default",
            required_labels={"expiry": "past"},
        )
        after_expiry = future_expiry + timedelta(hours=1)

        class PostExpiryDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return after_expiry if tz is not None else after_expiry.replace(tzinfo=None)

        try:
            assert (
                await store.get_entry(
                    entries[0].id,
                    access_scope=future_scope,
                )
                == entries[0]
            )
            assert await store.get_entry(entries[1].id, access_scope=past_scope) is None
            monkeypatch.setattr(postgres_storage, "datetime", PostExpiryDatetime)
            assert (
                await store.prune_expired(
                    access_scope=_ACCESS_SCOPE,
                    now=after_expiry,
                )
                == 2
            )
            async with (
                await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
                connection.cursor() as cursor,
            ):
                await cursor.execute(
                    "SELECT change_record.entry_id, audience.requires_include_expired "
                    "FROM cayu_knowledge_changes AS change_record "
                    "JOIN cayu_knowledge_change_audiences AS audience "
                    "ON audience.change_sequence = change_record.sequence "
                    "WHERE change_record.kind = 'expired' "
                    "AND audience.audience_kind = 'before' "
                    "ORDER BY change_record.entry_id"
                )
                assert await cursor.fetchall() == [
                    ("migrated-future-expiry", False),
                    ("migrated-past-expiry", True),
                ]
        finally:
            await store.close()
            await _drop_all(postgres_dsn)

    asyncio.run(run())


def test_postgres_revision_43_rejects_out_of_contract_revision_42_identities(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu.storage import postgres as postgres_storage

        await _drop_all(postgres_dsn)
        await _initialize_historical_schema(postgres_dsn, through_revision=42)

        entry = KnowledgeEntry(id="bounded-entry", text="Valid revision-42 entry.")
        oversized_chunk_id = "c" * (MAX_KNOWLEDGE_CHUNK_ID_BYTES + 1)
        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute(
                """
                INSERT INTO cayu_knowledge_entries (
                    id, namespace, current_revision, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s)
                """,
                (entry.id, entry.namespace, 1, entry.created_at, entry.updated_at),
            )
            await cursor.execute(
                """
                INSERT INTO cayu_knowledge_revisions (
                    entry_id, revision, text, kind, visibility, status,
                    created_by_type, created_by, created_at, updated_at,
                    source_type, source_uri, source_id, source_hash,
                    importance, importance_source, confidence, last_used_at,
                    expires_at, title, metadata
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb
                )
                """,
                postgres_storage._knowledge_entry_row_values(entry)[:-1],
            )
            await cursor.execute(
                """
                INSERT INTO cayu_knowledge_chunks (
                    id, entry_id, entry_revision, chunk_index,
                    text, content_hash, source_uri, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    oversized_chunk_id,
                    entry.id,
                    entry.revision,
                    0,
                    entry.text,
                    None,
                    None,
                    "{}",
                ),
            )
            await connection.commit()

        with pytest.raises(schema_migrations.SchemaTooOld, match="bounds knowledge"):
            await _initialize_historical_schema(postgres_dsn, through_revision=43)

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute("SELECT MAX(revision) FROM cayu_schema_migrations")
            assert (await cursor.fetchone())[0] == 42
            await cursor.execute("SELECT to_regclass('cayu_knowledge_evidence')")
            assert (await cursor.fetchone())[0] is None
        await _drop_all(postgres_dsn)

    asyncio.run(run())


def test_postgres_revision_migration_refuses_populated_legacy_knowledge_unchanged(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresKnowledgeStore

        await _drop_all(postgres_dsn)
        await _initialize_historical_schema(postgres_dsn, through_revision=41)

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute(
                """
                INSERT INTO cayu_knowledge_entries (
                    id, namespace, text, kind, visibility, status,
                    created_by_type, created_by, created_at, updated_at, metadata
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb
                )
                """,
                (
                    "legacy-entry",
                    "default",
                    "legacy text must survive",
                    "fact",
                    "global",
                    "active",
                    "system",
                    "legacy-test",
                    datetime(2026, 8, 18, 9, 0, tzinfo=UTC),
                    datetime(2026, 8, 18, 9, 0, tzinfo=UTC),
                    '{"proof":"unchanged"}',
                ),
            )
            await cursor.execute(
                "INSERT INTO cayu_knowledge_labels (entry_id, key, value) VALUES (%s, %s, %s)",
                ("legacy-entry", "project", "cayu"),
            )
            await cursor.execute(
                """
                INSERT INTO cayu_knowledge_chunks (
                    id, entry_id, chunk_index, text, metadata
                ) VALUES (%s, %s, %s, %s, %s::jsonb)
                """,
                ("legacy-entry:0", "legacy-entry", 0, "legacy chunk must survive", "{}"),
            )
            await connection.commit()

            before = await _legacy_knowledge_snapshot(cursor)

        migration = PostgresKnowledgeStore(
            postgres_dsn,
            access_scope=_ACCESS_SCOPE,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.MIGRATE,
        )
        try:
            with pytest.raises(KnowledgeRevisionResetRequired) as raised:
                await migration.ensure_schema()
        finally:
            await migration.close()

        assert raised.value.assessment.populated_tables == (
            "cayu_knowledge_chunks",
            "cayu_knowledge_entries",
            "cayu_knowledge_labels",
        )
        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            assert await _legacy_knowledge_snapshot(cursor) == before
        assert before[-1][-1] == (41, "breaking", 41)

    try:
        asyncio.run(run())
    finally:
        asyncio.run(_drop_all(postgres_dsn))


def test_postgres_revision_migration_refuses_unversioned_knowledge_before_ddl(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu import PostgresKnowledgeStore

        await _drop_all(postgres_dsn)
        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute(
                "CREATE TABLE cayu_knowledge_entries (id TEXT PRIMARY KEY, text TEXT NOT NULL)"
            )
            await cursor.execute(
                "INSERT INTO cayu_knowledge_entries (id, text) VALUES (%s, %s)",
                ("unversioned-entry", "must survive"),
            )
            await connection.commit()

        migration = PostgresKnowledgeStore(
            postgres_dsn,
            access_scope=_ACCESS_SCOPE,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.MIGRATE,
        )
        try:
            with pytest.raises(KnowledgeRevisionResetRequired):
                await migration.ensure_schema()
        finally:
            await migration.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute("SELECT to_regclass('cayu_schema_migrations')")
            assert await cursor.fetchone() == (None,)
            await cursor.execute("SELECT id, text FROM cayu_knowledge_entries")
            assert await cursor.fetchall() == [("unversioned-entry", "must survive")]

    try:
        asyncio.run(run())
    finally:
        asyncio.run(_drop_all(postgres_dsn))
