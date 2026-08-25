from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from itertools import permutations
from typing import Any

import pytest
from pydantic import ValidationError

from cayu import (
    MAX_KNOWLEDGE_RELATION_BYTES,
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeActorType,
    KnowledgeEntry,
    KnowledgeEntryReadLimitExceeded,
    KnowledgeMaintenanceCandidateSignal,
    KnowledgeMaintenanceRouter,
    KnowledgeMaintenanceRouterConfig,
    KnowledgeMaintenanceRoutingLimitExceeded,
    KnowledgeMaintenanceRoutingOmissionReason,
    KnowledgeMaintenanceRoutingRequest,
    KnowledgeMaintenanceRoutingTimeout,
    KnowledgeMaintenanceSignalKind,
    KnowledgeRelation,
    KnowledgeRelationKind,
    KnowledgeRelationResult,
    KnowledgeRevisionRef,
    KnowledgeStatus,
    KnowledgeVisibility,
    SQLiteKnowledgeStore,
)
from cayu._validation import canonical_durable_json_bytes
from cayu.storage.memory import knowledge_entry_payload_bytes

_NOW = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)
_OLD = _NOW - timedelta(days=90)
_PRIVILEGED = KnowledgeAccessScope.privileged()


def _entry(
    entry_id: str,
    *,
    namespace: str = "project:cayu",
    text: str | None = None,
    revision: int = 1,
    status: KnowledgeStatus = KnowledgeStatus.ACTIVE,
    created_at: datetime = _OLD,
    updated_at: datetime = _OLD,
    last_used_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> KnowledgeEntry:
    return KnowledgeEntry(
        id=entry_id,
        revision=revision,
        text=text or f"Knowledge for {entry_id}",
        namespace=namespace,
        labels={"project": "cayu"},
        visibility=KnowledgeVisibility.PROJECT,
        status=status,
        created_by_type=KnowledgeActorType.APP,
        created_by="test",
        created_at=created_at,
        updated_at=updated_at,
        last_used_at=last_used_at,
        expires_at=expires_at,
    )


def _ref(entry_id: str, revision: int = 1) -> KnowledgeRevisionRef:
    return KnowledgeRevisionRef(entry_id=entry_id, revision=revision)


def _signal(
    signal_id: str,
    kind: KnowledgeMaintenanceSignalKind,
    *references: KnowledgeRevisionRef,
    observed_at: datetime = _NOW,
    threshold_at: datetime | None = None,
    raw_score: float | None = None,
    relation_id: str | None = None,
) -> KnowledgeMaintenanceCandidateSignal:
    return KnowledgeMaintenanceCandidateSignal(
        id=signal_id,
        kind=kind,
        references=references,
        producer_id="test-router-signal-source",
        producer_version="1",
        reason_code=f"reason:{signal_id}",
        observed_at=observed_at,
        threshold_at=threshold_at,
        relation_id=(
            relation_id or f"relation:{signal_id}"
            if kind is KnowledgeMaintenanceSignalKind.CONTRADICTION
            else relation_id
        ),
        raw_score=raw_score,
        score_kind="test-similarity-v1" if raw_score is not None else None,
    )


def _request(
    *signals: KnowledgeMaintenanceCandidateSignal,
    access_scope: KnowledgeAccessScope = _PRIVILEGED,
    namespace: str = "project:cayu",
) -> KnowledgeMaintenanceRoutingRequest:
    return KnowledgeMaintenanceRoutingRequest(
        id="route-1",
        policy_id="reviewed-consolidation-v1",
        namespace=namespace,
        labels={"project": "cayu"},
        access_scope=access_scope,
        signals=signals,
        created_at=_NOW,
    )


async def _create(store, *entries: KnowledgeEntry) -> None:
    for entry in entries:
        await store.create_entry(entry, access_scope=_PRIVILEGED)


def test_signal_contract_is_strict_canonical_and_score_neutral() -> None:
    left = _ref("left")
    right = _ref("right")
    signal = _signal(
        "duplicate",
        KnowledgeMaintenanceSignalKind.DUPLICATE_HINT,
        right,
        left,
        raw_score=-47.25,
    )

    assert signal.references == (left, right)
    assert len(signal.fingerprint) == 64
    assert signal.raw_score == -47.25

    with pytest.raises(ValidationError, match="requires two exact revisions"):
        _signal("bad-pair", KnowledgeMaintenanceSignalKind.DUPLICATE_HINT, left)
    with pytest.raises(ValidationError, match="requires `threshold_at`"):
        _signal("bad-expiry", KnowledgeMaintenanceSignalKind.EXPIRY, left)
    with pytest.raises(ValidationError, match="Only duplicate hints"):
        _signal(
            "bad-score",
            KnowledgeMaintenanceSignalKind.EXACT_REFERENCE,
            left,
            raw_score=1.0,
        )
    with pytest.raises(ValidationError, match="requires `relation_id`"):
        KnowledgeMaintenanceCandidateSignal(
            id="unbound-contradiction",
            kind=KnowledgeMaintenanceSignalKind.CONTRADICTION,
            references=(left, right),
            producer_id="test",
            producer_version="1",
            reason_code="reviewed_contradiction",
            observed_at=_NOW,
        )
    with pytest.raises(ValidationError, match="machine-readable"):
        KnowledgeMaintenanceCandidateSignal(
            id="unsafe-reason",
            kind=KnowledgeMaintenanceSignalKind.EXACT_REFERENCE,
            references=(left,),
            producer_id="test",
            producer_version="1",
            reason_code="contains secret words",
            observed_at=_NOW,
        )
    with pytest.raises(ValidationError, match="must be set together"):
        KnowledgeMaintenanceCandidateSignal(
            id="bad-score-kind",
            kind=KnowledgeMaintenanceSignalKind.DUPLICATE_HINT,
            references=(left, right),
            producer_id="test",
            producer_version="1",
            reason_code="test",
            observed_at=_NOW,
            raw_score=1.0,
        )


def test_request_canonicalizes_signal_order_and_rejects_revision_ambiguity() -> None:
    first = _signal(
        "first",
        KnowledgeMaintenanceSignalKind.EXACT_REFERENCE,
        _ref("a"),
        observed_at=_OLD,
    )
    second = _signal(
        "second",
        KnowledgeMaintenanceSignalKind.EXACT_REFERENCE,
        _ref("b"),
    )
    forward = _request(first, second)
    reverse = _request(second, first)

    assert forward.signals == reverse.signals
    assert forward.fingerprint == reverse.fingerprint

    with pytest.raises(ValidationError, match="conflicting revisions"):
        _request(
            _signal("old", KnowledgeMaintenanceSignalKind.EXACT_REFERENCE, _ref("a", 1)),
            _signal("new", KnowledgeMaintenanceSignalKind.EXACT_REFERENCE, _ref("a", 2)),
        )
    with pytest.raises(ValidationError, match="semantic observation"):
        _request(
            first,
            first.model_copy(update={"id": "same-observation-new-id"}),
        )
    duplicate = _signal(
        "duplicate",
        KnowledgeMaintenanceSignalKind.DUPLICATE_HINT,
        _ref("a"),
        _ref("b"),
        raw_score=0.1,
    )
    with pytest.raises(ValidationError, match="semantic observation"):
        _request(
            duplicate,
            duplicate.model_copy(update={"id": "rescored", "raw_score": 0.9}),
        )
    with pytest.raises(ValidationError, match="after its routing request"):
        _request(
            _signal(
                "future",
                KnowledgeMaintenanceSignalKind.EXACT_REFERENCE,
                _ref("a"),
                observed_at=_NOW + timedelta(seconds=1),
            )
        )


def test_router_output_is_deterministic_and_defensively_copied() -> None:
    async def run(order: tuple[KnowledgeMaintenanceCandidateSignal, ...]):
        store = InMemoryKnowledgeStore()
        left = _entry("left")
        right = _entry("right")
        await _create(store, left, right)
        result = await KnowledgeMaintenanceRouter(store).route(_request(*order))
        return store, result

    exact = _signal(
        "exact",
        KnowledgeMaintenanceSignalKind.EXACT_REFERENCE,
        _ref("right"),
    )
    duplicate = _signal(
        "duplicate",
        KnowledgeMaintenanceSignalKind.DUPLICATE_HINT,
        _ref("right"),
        _ref("left"),
        raw_score=0.99,
    )
    outputs = [asyncio.run(run(order))[1] for order in permutations((exact, duplicate))]

    assert outputs[0].model_dump(mode="json") == outputs[1].model_dump(mode="json")
    result = outputs[0]
    assert tuple(signal.id for signal in result.routed_signals) == ("exact", "duplicate")
    assert tuple(candidate.reference.entry_id for candidate in result.candidates) == (
        "right",
        "left",
    )
    assert result.candidates[0].signal_ids == ("exact", "duplicate")
    assert result.schema_version == 1
    assert result.signal_count == 2
    assert result.loaded_reference_count == 2
    assert result.relation_payload_bytes == 0
    assert not result.truncated
    assert len(result.fingerprint) == 64

    tampered = result.model_dump(mode="python")
    tampered["candidates"][0]["signal_kinds"] = (KnowledgeMaintenanceSignalKind.DUPLICATE_HINT,)
    with pytest.raises(ValidationError, match="signal kinds"):
        type(result).model_validate(tampered)

    duplicated = result.model_dump(mode="python")
    duplicated["candidates"] = (
        *duplicated["candidates"],
        duplicated["candidates"][0],
    )
    with pytest.raises(ValidationError, match="cannot repeat"):
        type(result).model_validate(duplicated)

    oversized_relation_payload = result.model_dump(mode="python")
    oversized_relation_payload["relation_payload_bytes"] = (
        oversized_relation_payload["max_relation_load_bytes"] + 1
    )
    with pytest.raises(ValidationError, match="relation_payload_bytes"):
        type(result).model_validate(oversized_relation_payload)

    result.candidates[0].entry.labels["mutated"] = "outside"
    store, fresh = asyncio.run(run((exact, duplicate)))
    persisted = asyncio.run(store.get_entry("right", access_scope=_PRIVILEGED))
    assert persisted is not None and "mutated" not in persisted.labels
    assert "mutated" not in fresh.candidates[0].entry.labels


def test_raw_similarity_score_never_overrides_configured_priority() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await _create(
            store,
            _entry("a"),
            _entry("b"),
            _entry("c"),
            _entry("d"),
        )
        low = _signal(
            "a-low-score",
            KnowledgeMaintenanceSignalKind.DUPLICATE_HINT,
            _ref("a"),
            _ref("b"),
            raw_score=-1_000.0,
        )
        high = _signal(
            "z-high-score",
            KnowledgeMaintenanceSignalKind.DUPLICATE_HINT,
            _ref("c"),
            _ref("d"),
            raw_score=1_000.0,
        )
        router = KnowledgeMaintenanceRouter(
            store,
            config=KnowledgeMaintenanceRouterConfig(
                max_candidates=2,
                max_candidate_reads=4,
                max_concurrency=4,
            ),
        )
        return await router.route(_request(high, low))

    result = asyncio.run(run())

    assert tuple(signal.id for signal in result.routed_signals) == ("a-low-score",)
    assert result.omissions[0].signal_id == "z-high-score"
    assert result.omissions[0].reason is KnowledgeMaintenanceRoutingOmissionReason.CANDIDATE_LIMIT
    assert result.truncated


def test_router_fails_stale_and_scope_mismatched_signals_closed() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await _create(
            store,
            _entry("stale"),
            _entry("other", namespace="project:other"),
        )
        await store.append_entry_revision(
            _entry("stale", revision=2, text="new current revision", updated_at=_NOW),
            expected_revision=1,
            access_scope=_PRIVILEGED,
        )
        result = await KnowledgeMaintenanceRouter(store).route(
            _request(
                _signal(
                    "stale-signal",
                    KnowledgeMaintenanceSignalKind.EXACT_REFERENCE,
                    _ref("stale", 1),
                ),
                _signal(
                    "other-scope",
                    KnowledgeMaintenanceSignalKind.EXACT_REFERENCE,
                    _ref("other"),
                ),
            )
        )
        isolated = await KnowledgeMaintenanceRouter(store).route(
            _request(
                _signal(
                    "hidden",
                    KnowledgeMaintenanceSignalKind.EXACT_REFERENCE,
                    _ref("other"),
                ),
                access_scope=KnowledgeAccessScope.for_namespace(
                    "project:cayu",
                    required_labels={"project": "cayu"},
                    allowed_visibilities=[KnowledgeVisibility.PROJECT],
                ),
            )
        )
        return result, isolated

    result, isolated = asyncio.run(run())

    assert result.candidates == ()
    assert {item.signal_id: item.reason for item in result.omissions} == {
        "stale-signal": KnowledgeMaintenanceRoutingOmissionReason.STALE_REVISION,
        "other-scope": KnowledgeMaintenanceRoutingOmissionReason.SCOPE_MISMATCH,
    }
    assert isolated.omissions[0].reason is KnowledgeMaintenanceRoutingOmissionReason.UNAVAILABLE
    assert isolated.loaded_reference_count == 0
    assert not isolated.truncated


def test_router_verifies_expiry_inactivity_and_active_lifecycle() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await _create(
            store,
            _entry("expired", expires_at=_NOW - timedelta(days=1)),
            _entry("fresh", expires_at=_NOW + timedelta(days=1)),
            _entry("unused", last_used_at=_OLD),
            _entry("recent", updated_at=_NOW - timedelta(hours=1)),
            _entry("archived", status=KnowledgeStatus.ARCHIVED),
        )
        signals = (
            _signal(
                "expired",
                KnowledgeMaintenanceSignalKind.EXPIRY,
                _ref("expired"),
                threshold_at=_NOW,
            ),
            _signal(
                "fresh",
                KnowledgeMaintenanceSignalKind.EXPIRY,
                _ref("fresh"),
                threshold_at=_NOW,
            ),
            _signal(
                "unused",
                KnowledgeMaintenanceSignalKind.LOW_USAGE,
                _ref("unused"),
                threshold_at=_NOW - timedelta(days=30),
            ),
            _signal(
                "recent",
                KnowledgeMaintenanceSignalKind.LOW_USAGE,
                _ref("recent"),
                threshold_at=_NOW - timedelta(days=30),
            ),
            _signal(
                "archived",
                KnowledgeMaintenanceSignalKind.EXACT_REFERENCE,
                _ref("archived"),
            ),
        )
        return await KnowledgeMaintenanceRouter(store).route(_request(*signals))

    result = asyncio.run(run())

    assert {candidate.reference.entry_id for candidate in result.candidates} == {
        "expired",
        "unused",
    }
    assert {item.signal_id: item.reason for item in result.omissions} == {
        "fresh": KnowledgeMaintenanceRoutingOmissionReason.CONDITION_NOT_MET,
        "recent": KnowledgeMaintenanceRoutingOmissionReason.CONDITION_NOT_MET,
        "archived": KnowledgeMaintenanceRoutingOmissionReason.LIFECYCLE_MISMATCH,
    }


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_router_requires_a_durable_exact_contradiction(backend: str, tmp_path) -> None:
    async def run():
        store = (
            InMemoryKnowledgeStore()
            if backend == "memory"
            else SQLiteKnowledgeStore(tmp_path / "routing.sqlite")
        )
        try:
            await _create(store, _entry("left"), _entry("right"))
            signal = _signal(
                "contradiction",
                KnowledgeMaintenanceSignalKind.CONTRADICTION,
                _ref("left"),
                _ref("right"),
            )
            before = await KnowledgeMaintenanceRouter(store).route(_request(signal))
            await store.publish_relations(
                [
                    KnowledgeRelation(
                        id="relation:contradiction",
                        subject=_ref("left"),
                        object=_ref("right"),
                        kind=KnowledgeRelationKind.CONTRADICTS,
                        created_by="test",
                        policy_id="reviewed-consolidation-v1",
                        created_at=_NOW,
                    )
                ],
                operation_id="publish:contradiction",
                access_scope=_PRIVILEGED,
            )
            after = await KnowledgeMaintenanceRouter(store).route(_request(signal))
            wrong_identity = await KnowledgeMaintenanceRouter(store).route(
                _request(signal.model_copy(update={"relation_id": "relation:other"}))
            )
            return before, after, wrong_identity
        finally:
            close = getattr(store, "close", None)
            if callable(close):
                await close()

    before, after, wrong_identity = asyncio.run(run())

    assert before.candidates == ()
    assert before.omissions[0].reason is KnowledgeMaintenanceRoutingOmissionReason.CONDITION_NOT_MET
    assert {candidate.reference.entry_id for candidate in after.candidates} == {
        "left",
        "right",
    }
    assert after.omissions == ()
    assert wrong_identity.candidates == ()
    assert wrong_identity.omissions[0].reason is (
        KnowledgeMaintenanceRoutingOmissionReason.CONDITION_NOT_MET
    )


def test_contradiction_lookup_paginates_or_reports_incomplete_coverage() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await _create(store, _entry("anchor"), _entry("earlier"), _entry("target"))
        await store.publish_relations(
            [
                KnowledgeRelation(
                    id="relation:earlier",
                    subject=_ref("anchor"),
                    object=_ref("earlier"),
                    kind=KnowledgeRelationKind.CONTRADICTS,
                    created_by="test",
                    created_at=_OLD,
                ),
                KnowledgeRelation(
                    id="relation:target",
                    subject=_ref("anchor"),
                    object=_ref("target"),
                    kind=KnowledgeRelationKind.CONTRADICTS,
                    created_by="test",
                    created_at=_NOW,
                ),
            ],
            operation_id="publish:paged-contradictions",
            access_scope=_PRIVILEGED,
        )
        signal = _signal(
            "target",
            KnowledgeMaintenanceSignalKind.CONTRADICTION,
            _ref("anchor"),
            _ref("target"),
            relation_id="relation:target",
        )
        complete = await KnowledgeMaintenanceRouter(
            store,
            config=KnowledgeMaintenanceRouterConfig(
                relation_page_limit=1,
                max_relation_records_per_signal=2,
            ),
        ).route(_request(signal))
        incomplete = await KnowledgeMaintenanceRouter(
            store,
            config=KnowledgeMaintenanceRouterConfig(
                relation_page_limit=1,
                max_relation_records_per_signal=1,
            ),
        ).route(_request(signal))
        return complete, incomplete

    complete, incomplete = asyncio.run(run())

    assert {candidate.reference.entry_id for candidate in complete.candidates} == {
        "anchor",
        "target",
    }
    assert complete.omissions == ()
    assert incomplete.candidates == ()
    assert incomplete.omissions[0].reason is (
        KnowledgeMaintenanceRoutingOmissionReason.RELATION_COVERAGE_INCOMPLETE
    )
    assert incomplete.truncated


def test_contradiction_relation_must_precede_its_claimed_observation() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await _create(store, _entry("left"), _entry("right"))
        await store.publish_relations(
            [
                KnowledgeRelation(
                    id="relation:future",
                    subject=_ref("left"),
                    object=_ref("right"),
                    kind=KnowledgeRelationKind.CONTRADICTS,
                    created_by="test",
                    created_at=_NOW + timedelta(seconds=1),
                )
            ],
            operation_id="publish:future-contradiction",
            access_scope=_PRIVILEGED,
        )
        signal = _signal(
            "future",
            KnowledgeMaintenanceSignalKind.CONTRADICTION,
            _ref("left"),
            _ref("right"),
        )
        return await KnowledgeMaintenanceRouter(store).route(_request(signal))

    result = asyncio.run(run())

    assert result.candidates == ()
    assert result.omissions[0].reason is (
        KnowledgeMaintenanceRoutingOmissionReason.CONDITION_NOT_MET
    )


class _RelationBudgetStore:
    def __init__(self, relations: list[KnowledgeRelation]) -> None:
        self.relations = {relation.subject.entry_id: relation for relation in relations}
        self.requested_max_bytes: list[int] = []
        self.requested_references: list[str] = []
        self.in_flight_bytes = 0
        self.max_in_flight_bytes = 0

    async def get_entry(self, entry_id: str, **_: Any) -> KnowledgeEntry:
        return _entry(entry_id)

    async def read_relations(self, query, **_: Any) -> KnowledgeRelationResult:
        self.requested_max_bytes.append(query.max_bytes)
        self.requested_references.append(query.reference.entry_id)
        self.in_flight_bytes += query.max_bytes
        self.max_in_flight_bytes = max(self.max_in_flight_bytes, self.in_flight_bytes)
        try:
            await asyncio.sleep(0)
            return KnowledgeRelationResult(
                query=query,
                relations=[self.relations[query.reference.entry_id]],
            )
        finally:
            self.in_flight_bytes -= query.max_bytes


def _relation_budget_fixture(
    *prefixes: str,
) -> tuple[list[KnowledgeRelation], tuple[KnowledgeMaintenanceCandidateSignal, ...]]:
    relations: list[KnowledgeRelation] = []
    signals: list[KnowledgeMaintenanceCandidateSignal] = []
    for prefix in prefixes:
        relation = KnowledgeRelation(
            id=f"relation:{prefix}",
            subject=_ref(f"{prefix}-anchor"),
            object=_ref(f"{prefix}-target"),
            kind=KnowledgeRelationKind.CONTRADICTS,
            created_by="test",
            created_at=_OLD,
        )
        relations.append(relation)
        signals.append(
            _signal(
                f"signal:{prefix}",
                KnowledgeMaintenanceSignalKind.CONTRADICTION,
                relation.subject,
                relation.object,
                relation_id=relation.id,
            )
        )
    return relations, tuple(signals)


def test_relation_pages_reserve_one_request_wide_byte_budget_before_launch() -> None:
    async def run():
        relations, signals = _relation_budget_fixture("a", "b", "c")
        store = _RelationBudgetStore(relations)
        max_bytes = 4 * 1024 * 1024
        result = await KnowledgeMaintenanceRouter(
            store,
            config=KnowledgeMaintenanceRouterConfig(
                max_signals=3,
                max_candidate_reads=6,
                max_candidates=6,
                max_concurrency=512,
                relation_page_limit=1000,
                max_relation_records_per_signal=1000,
                relation_page_max_bytes=max_bytes,
                max_relation_load_bytes=max_bytes,
            ),
        ).route(_request(*reversed(signals)))
        return relations, store, result

    relations, store, result = asyncio.run(run())
    max_bytes = 4 * 1024 * 1024
    expected_payload_bytes = sum(
        len(
            canonical_durable_json_bytes(
                relation.model_dump(mode="json"),
                "knowledge relation",
            )
        )
        for relation in relations
    )

    assert tuple(signal.id for signal in result.routed_signals) == (
        "signal:a",
        "signal:b",
        "signal:c",
    )
    assert result.omissions == ()
    assert result.relation_payload_bytes == expected_payload_bytes
    assert result.max_relation_load_bytes == max_bytes
    assert store.requested_references == ["a-anchor", "b-anchor", "c-anchor"]
    assert store.requested_max_bytes[0] == max_bytes
    assert store.requested_max_bytes == sorted(store.requested_max_bytes, reverse=True)
    assert store.max_in_flight_bytes <= max_bytes


def test_exhausted_relation_byte_budget_fails_remaining_signals_closed() -> None:
    async def run():
        relations, signals = _relation_budget_fixture("a", "b")
        store = _RelationBudgetStore(relations)
        result = await KnowledgeMaintenanceRouter(
            store,
            config=KnowledgeMaintenanceRouterConfig(
                max_candidate_reads=4,
                max_candidates=4,
                max_concurrency=512,
                relation_page_max_bytes=MAX_KNOWLEDGE_RELATION_BYTES,
                max_relation_load_bytes=MAX_KNOWLEDGE_RELATION_BYTES,
            ),
        ).route(_request(*reversed(signals)))
        return store, result

    store, result = asyncio.run(run())

    assert tuple(signal.id for signal in result.routed_signals) == ("signal:a",)
    assert tuple(omission.signal_id for omission in result.omissions) == ("signal:b",)
    assert result.omissions[0].reason is (
        KnowledgeMaintenanceRoutingOmissionReason.RELATION_COVERAGE_INCOMPLETE
    )
    assert 0 < result.relation_payload_bytes <= MAX_KNOWLEDGE_RELATION_BYTES
    assert store.requested_references == ["a-anchor"]
    assert store.max_in_flight_bytes == MAX_KNOWLEDGE_RELATION_BYTES
    assert result.truncated


def test_router_rejects_a_relation_page_bound_to_another_query() -> None:
    class MismatchedQueryStore:
        async def get_entry(self, entry_id: str, **_: Any) -> KnowledgeEntry:
            return _entry(entry_id)

        async def read_relations(self, query, **_: Any):
            from cayu import KnowledgeRelationQuery, KnowledgeRelationResult

            mismatched = KnowledgeRelationQuery(
                reference=query.reference,
                kinds=query.kinds,
                limit=query.limit,
                max_bytes=query.max_bytes + 1,
                cursor=query.cursor,
            )
            return KnowledgeRelationResult(query=mismatched)

    async def run() -> None:
        signal = _signal(
            "contradiction",
            KnowledgeMaintenanceSignalKind.CONTRADICTION,
            _ref("left"),
            _ref("right"),
        )
        router = KnowledgeMaintenanceRouter(
            MismatchedQueryStore(),
            config=KnowledgeMaintenanceRouterConfig(
                relation_page_limit=1,
                max_relation_records_per_signal=2,
            ),
        )
        with pytest.raises(TypeError, match="exact submitted query"):
            await router.route(_request(signal))

    asyncio.run(run())


def test_router_rejects_an_entry_returned_for_another_identity() -> None:
    class MismatchedEntryStore:
        def __init__(self) -> None:
            self.blocked = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def get_entry(self, entry_id: str, **_: Any) -> KnowledgeEntry:
            if entry_id == "blocked":
                self.blocked.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.cancelled.set()
            await self.blocked.wait()
            return _entry("different")

        async def read_relations(self, query, **_: Any):
            return None

    async def run() -> None:
        store = MismatchedEntryStore()
        router = KnowledgeMaintenanceRouter(
            store,
            config=KnowledgeMaintenanceRouterConfig(max_concurrency=2),
        )
        with pytest.raises(TypeError, match="exact requested logical entry"):
            await router.route(
                _request(
                    _signal(
                        "exact",
                        KnowledgeMaintenanceSignalKind.EXACT_REFERENCE,
                        _ref("expected"),
                    ),
                    _signal(
                        "blocked",
                        KnowledgeMaintenanceSignalKind.EXACT_REFERENCE,
                        _ref("blocked"),
                    ),
                )
            )
        await asyncio.wait_for(store.cancelled.wait(), timeout=1.0)

    asyncio.run(run())


def test_pair_signal_is_omitted_atomically_when_count_or_byte_budget_cannot_fit() -> None:
    async def run():
        store = InMemoryKnowledgeStore()
        await _create(
            store,
            _entry("left", text="L" * 2_000),
            _entry("right", text="R" * 2_000),
        )
        signal = _signal(
            "duplicate",
            KnowledgeMaintenanceSignalKind.DUPLICATE_HINT,
            _ref("left"),
            _ref("right"),
        )
        count_limited = await KnowledgeMaintenanceRouter(
            store,
            config=KnowledgeMaintenanceRouterConfig(
                max_candidates=1,
                max_candidate_reads=2,
                max_concurrency=2,
            ),
        ).route(_request(signal))
        byte_limited = await KnowledgeMaintenanceRouter(
            store,
            config=KnowledgeMaintenanceRouterConfig(
                max_candidate_reads=2,
                max_candidates=2,
                max_concurrency=2,
                max_candidate_bytes=512,
            ),
        ).route(_request(signal))
        return count_limited, byte_limited

    count_limited, byte_limited = asyncio.run(run())

    assert count_limited.candidates == ()
    assert count_limited.routed_signals == ()
    assert count_limited.omissions[0].reason is (
        KnowledgeMaintenanceRoutingOmissionReason.CANDIDATE_LIMIT
    )
    assert byte_limited.candidates == ()
    assert byte_limited.routed_signals == ()
    assert byte_limited.omissions[0].reason is (
        KnowledgeMaintenanceRoutingOmissionReason.CANDIDATE_BYTES
    )
    assert byte_limited.candidate_payload_bytes <= byte_limited.max_candidate_bytes


def test_oversized_candidates_are_refused_by_the_store_before_materialization() -> None:
    class RefusingStore:
        def __init__(self) -> None:
            self.max_bytes: list[int] = []

        async def get_entry(self, entry_id: str, *, max_bytes: int, **_: Any):
            self.max_bytes.append(max_bytes)
            raise KnowledgeEntryReadLimitExceeded(
                entry_id,
                revision=1,
                payload_bytes=1024 * 1024,
                max_bytes=max_bytes,
            )

        async def read_relations(self, query, **_: Any):
            return None

    async def run():
        store = RefusingStore()
        signals = tuple(
            _signal(
                f"oversized-{index:02d}",
                KnowledgeMaintenanceSignalKind.EXACT_REFERENCE,
                _ref(f"oversized-{index:02d}"),
            )
            for index in range(20)
        )
        result = await KnowledgeMaintenanceRouter(
            store,
            config=KnowledgeMaintenanceRouterConfig(
                max_signals=20,
                max_candidate_reads=20,
                max_candidates=20,
                max_candidate_bytes=256,
                max_concurrency=8,
            ),
        ).route(_request(*signals))
        return store, result

    store, result = asyncio.run(run())

    assert store.max_bytes == [256] * 20
    assert result.candidates == ()
    assert result.loaded_reference_count == 0
    assert {omission.reason for omission in result.omissions} == {
        KnowledgeMaintenanceRoutingOmissionReason.CANDIDATE_BYTES
    }


def test_aggregate_candidate_load_budget_is_deterministic() -> None:
    async def run():
        entries = [_entry(f"load-budget-{index:02d}") for index in range(5)]
        store = InMemoryKnowledgeStore(entries, access_scope=_PRIVILEGED)
        load_bytes = sum(knowledge_entry_payload_bytes(entry) for entry in entries[:2])
        signals = tuple(
            _signal(
                f"load-budget-{index:02d}",
                KnowledgeMaintenanceSignalKind.EXACT_REFERENCE,
                _ref(entry.id),
            )
            for index, entry in enumerate(entries)
        )
        result = await KnowledgeMaintenanceRouter(
            store,
            config=KnowledgeMaintenanceRouterConfig(
                max_signals=5,
                max_candidate_reads=5,
                max_candidates=5,
                max_candidate_bytes=128 * 1024,
                max_candidate_load_bytes=load_bytes,
                max_concurrency=5,
            ),
        ).route(_request(*signals))
        return result

    result = asyncio.run(run())

    assert result.loaded_reference_count == 2
    assert [candidate.reference.entry_id for candidate in result.candidates] == [
        "load-budget-00",
        "load-budget-01",
    ]
    assert [omission.reason for omission in result.omissions] == [
        KnowledgeMaintenanceRoutingOmissionReason.CANDIDATE_BYTES,
        KnowledgeMaintenanceRoutingOmissionReason.CANDIDATE_BYTES,
        KnowledgeMaintenanceRoutingOmissionReason.CANDIDATE_BYTES,
    ]


class _CountingStore:
    def __init__(self) -> None:
        self.get_calls = 0
        self.relation_calls = 0

    async def get_entry(self, entry_id: str, **_: Any) -> KnowledgeEntry | None:
        self.get_calls += 1
        return None

    async def read_relations(self, query, **_: Any):
        self.relation_calls += 1
        return None


def test_zero_signal_path_performs_no_store_work() -> None:
    async def run():
        store = _CountingStore()
        result = await KnowledgeMaintenanceRouter(store).route(_request())
        return store, result

    store, result = asyncio.run(run())

    assert result.candidates == ()
    assert result.omissions == ()
    assert result.signal_count == 0
    assert result.loaded_reference_count == 0
    assert store.get_calls == 0
    assert store.relation_calls == 0


def test_pre_read_ceiling_fails_before_store_access() -> None:
    async def run():
        store = _CountingStore()
        router = KnowledgeMaintenanceRouter(
            store,
            config=KnowledgeMaintenanceRouterConfig(
                max_candidate_reads=1,
                max_candidates=1,
                max_concurrency=1,
            ),
        )
        request = _request(
            _signal("one", KnowledgeMaintenanceSignalKind.EXACT_REFERENCE, _ref("one")),
            _signal("two", KnowledgeMaintenanceSignalKind.EXACT_REFERENCE, _ref("two")),
        )
        with pytest.raises(KnowledgeMaintenanceRoutingLimitExceeded) as raised:
            await router.route(request)
        return store, raised.value

    store, error = asyncio.run(run())

    assert error.limit == "max_candidate_reads"
    assert store.get_calls == 0
    assert store.relation_calls == 0


class _BlockingStore(_CountingStore):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def get_entry(self, entry_id: str, **_: Any) -> KnowledgeEntry | None:
        self.get_calls += 1
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cancelled.set()
        return None


def test_timeout_is_typed_and_cancels_in_flight_reads() -> None:
    async def run():
        store = _BlockingStore()
        router = KnowledgeMaintenanceRouter(
            store,
            config=KnowledgeMaintenanceRouterConfig(timeout_seconds=0.01),
        )
        with pytest.raises(KnowledgeMaintenanceRoutingTimeout):
            await router.route(
                _request(
                    _signal(
                        "blocked",
                        KnowledgeMaintenanceSignalKind.EXACT_REFERENCE,
                        _ref("blocked"),
                    )
                )
            )
        await asyncio.wait_for(store.cancelled.wait(), timeout=1.0)

    asyncio.run(run())


def test_timeout_covers_synchronous_payload_preparation() -> None:
    class SlowBudgetRouter(KnowledgeMaintenanceRouter):
        def _apply_payload_budgets(self, eligible, entry_payload_bytes):
            time.sleep(0.02)
            return super()._apply_payload_budgets(eligible, entry_payload_bytes)

    async def run() -> None:
        store = InMemoryKnowledgeStore()
        await _create(store, _entry("slow"))
        router = SlowBudgetRouter(
            store,
            config=KnowledgeMaintenanceRouterConfig(timeout_seconds=0.005),
        )
        with pytest.raises(KnowledgeMaintenanceRoutingTimeout):
            await router.route(
                _request(
                    _signal(
                        "slow",
                        KnowledgeMaintenanceSignalKind.EXACT_REFERENCE,
                        _ref("slow"),
                    )
                )
            )

    asyncio.run(run())


def test_caller_cancellation_propagates_and_cancels_in_flight_reads() -> None:
    async def run():
        store = _BlockingStore()
        task = asyncio.create_task(
            KnowledgeMaintenanceRouter(store).route(
                _request(
                    _signal(
                        "blocked",
                        KnowledgeMaintenanceSignalKind.EXACT_REFERENCE,
                        _ref("blocked"),
                    )
                )
            )
        )
        await asyncio.wait_for(store.started.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(store.cancelled.wait(), timeout=1.0)

    asyncio.run(run())


def test_caller_cancellation_cancels_in_flight_relation_reads() -> None:
    class BlockingRelationStore:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def get_entry(self, entry_id: str, **_: Any) -> KnowledgeEntry:
            return _entry(entry_id)

        async def read_relations(self, query, **_: Any) -> KnowledgeRelationResult:
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()
            raise AssertionError("cancelled relation read resumed")

    async def run() -> None:
        store = BlockingRelationStore()
        task = asyncio.create_task(
            KnowledgeMaintenanceRouter(store).route(
                _request(
                    _signal(
                        "blocked-relation",
                        KnowledgeMaintenanceSignalKind.CONTRADICTION,
                        _ref("left"),
                        _ref("right"),
                    )
                )
            )
        )
        await asyncio.wait_for(store.started.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(store.cancelled.wait(), timeout=1.0)

    asyncio.run(run())


def test_config_is_copied_and_fingerprinted() -> None:
    config = KnowledgeMaintenanceRouterConfig()
    store = _CountingStore()
    router = KnowledgeMaintenanceRouter(store, config=config)
    first = router.config
    first_priority = first.signal_priority

    assert router.config is not first
    assert router.config.signal_priority == first_priority
    assert len(config.fingerprint) == 64
    bounded_scan = KnowledgeMaintenanceRouterConfig(
        relation_page_limit=50,
        max_relation_records_per_signal=1,
    )
    assert bounded_scan.max_relation_records_per_signal == 1
    with pytest.raises(ValidationError, match="every signal kind exactly once"):
        KnowledgeMaintenanceRouterConfig(
            signal_priority=(KnowledgeMaintenanceSignalKind.EXACT_REFERENCE,)
        )
    with pytest.raises(ValidationError, match="max_candidate_load_bytes"):
        KnowledgeMaintenanceRouterConfig(max_candidate_load_bytes=4 * 1024 * 1024 + 1)
    with pytest.raises(ValidationError, match="max_relation_load_bytes"):
        KnowledgeMaintenanceRouterConfig(max_relation_load_bytes=MAX_KNOWLEDGE_RELATION_BYTES - 1)
    with pytest.raises(ValidationError, match="max_relation_load_bytes"):
        KnowledgeMaintenanceRouterConfig(max_relation_load_bytes=4 * 1024 * 1024 + 1)
