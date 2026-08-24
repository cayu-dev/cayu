from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from cayu import (
    CandidatePolicyDisposition,
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeCandidatePolicyDecision,
    KnowledgeCurator,
    KnowledgeCuratorConfig,
    KnowledgeListQuery,
    KnowledgeQuery,
    KnowledgeReviewWorkflow,
    KnowledgeStatus,
    LearningBatch,
    LearningBatchOutcome,
    LearningBatchResult,
    LearningCandidate,
    LearningCandidateOutcome,
    LearningCandidateResult,
    LearningDecision,
    LearningSignal,
    LearningSignalOutcome,
    LearningSignalResult,
    LearningSourceReference,
    LearningVerdict,
    SQLiteKnowledgeStore,
    group_learning_signals,
)

_ACCESS_SCOPE = KnowledgeAccessScope.privileged()
_NOW = datetime(2026, 8, 24, 2, 0, tzinfo=UTC)


def _source(
    source_id: str = "session-1",
    *,
    source_hash: str = "sha256:source-1",
) -> LearningSourceReference:
    return LearningSourceReference(
        source_type="session",
        source_id=source_id,
        source_hash=source_hash,
        locator={"event_id": f"event:{source_id}"},
    )


def _signal(
    signal_id: str = "signal-1",
    *,
    scope: str = "project:cayu",
    summary: str = "A deploy failed because migrations ran after the service started.",
    occurred_at: datetime = _NOW,
) -> LearningSignal:
    return LearningSignal(
        id=signal_id,
        deduplication_key=f"dedupe:{signal_id}",
        kind="deployment_failure",
        scope=scope,
        summary=summary,
        source_references=(_source(signal_id, source_hash=f"sha256:{signal_id}"),),
        occurred_at=occurred_at,
        metadata={"environment": "staging"},
    )


def _batch(*signals: LearningSignal) -> LearningBatch:
    return LearningBatch(id="batch-1", signals=signals or (_signal(),))


def _candidate(
    proposal_key: str = "deploy-migrations-first",
    *,
    text: str = "Run database migrations before starting the new service revision.",
    signal_ids: tuple[str, ...] = ("signal-1",),
    kind: str = "procedure",
) -> LearningCandidate:
    return LearningCandidate(
        proposal_key=proposal_key,
        text=text,
        title="Deploy migrations before service startup",
        kind=kind,
        aspects=("deployment", "migrations"),
        signal_ids=signal_ids,
        confidence_hint=0.8,
        metadata={"generator_category": "deployment"},
    )


def _accepted(code: str = "supported") -> LearningDecision:
    return LearningDecision(
        verdict=LearningVerdict.ACCEPTED,
        code=code,
        notes="The exact failure evidence supports the proposed procedure.",
        confidence=0.95,
        metadata={"rubric": "support-v1"},
    )


class _Generator:
    def __init__(self, candidates: list[LearningCandidate] | None = None) -> None:
        self.candidates = [_candidate()] if candidates is None else candidates
        self.calls = 0

    async def generate_candidates(self, batch: LearningBatch) -> list[LearningCandidate]:
        self.calls += 1
        return list(self.candidates)


class _Evaluator:
    def __init__(
        self, decisions: dict[str, LearningDecision | BaseException] | None = None
    ) -> None:
        self.decisions = decisions or {}
        self.calls: list[str] = []

    async def evaluate_candidate(
        self,
        candidate: LearningCandidate,
        signals: tuple[LearningSignal, ...],
    ) -> LearningDecision:
        self.calls.append(candidate.proposal_key)
        assert tuple(signal.id for signal in signals) == candidate.signal_ids
        decision = self.decisions.get(candidate.proposal_key, _accepted())
        if isinstance(decision, BaseException):
            raise decision
        return decision


class _Policy:
    async def apply_candidate_policy(
        self,
        candidate: LearningCandidate,
        signals: tuple[LearningSignal, ...],
    ) -> KnowledgeCandidatePolicyDecision:
        if candidate.proposal_key == "reject-me":
            return KnowledgeCandidatePolicyDecision(
                disposition=CandidatePolicyDisposition.REJECTED,
                code="contains_domain_secret",
            )
        return KnowledgeCandidatePolicyDecision(
            disposition=CandidatePolicyDisposition.ACCEPTED,
            code="redacted",
            candidate=candidate.model_copy(
                update={"text": candidate.text.replace("internal", "approved")}
            ),
        )


def _config(**updates: Any) -> KnowledgeCuratorConfig:
    return KnowledgeCuratorConfig(
        candidate_generator_identity="test.generator.v1",
        evaluator_identity="test.evaluator.v1",
        namespace="project:cayu",
        labels={"project": "cayu"},
        **updates,
    )


def _entry_id(result: LearningCandidateResult) -> str:
    assert result.entry_id is not None
    return result.entry_id


def _curator(
    store,
    *,
    generator: _Generator | None = None,
    evaluator: _Evaluator | None = None,
    config: KnowledgeCuratorConfig | None = None,
    candidate_policy=None,
) -> KnowledgeCurator:
    return KnowledgeCurator(
        store,
        candidate_generator=generator or _Generator(),
        evaluator=evaluator or _Evaluator(),
        candidate_policy=candidate_policy,
        config=config or _config(),
        clock=lambda: _NOW,
    )


def test_curator_persists_pending_revision_with_exact_evidence_and_review_path() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        curator = _curator(store)
        result = await curator.curate(_batch(_signal()))
        candidate_result = result.candidates[0]
        entry_id = _entry_id(candidate_result)
        entry = await store.get_entry(entry_id)
        evidence = await store.read_evidence(entry_id)
        hidden = await store.search(
            KnowledgeQuery(text="migrations service", namespace="project:cayu")
        )
        review = KnowledgeReviewWorkflow(store, namespace="project:cayu")
        approved = await review.approve(entry_id)
        recalled = await store.search(
            KnowledgeQuery(text="migrations service", namespace="project:cayu")
        )
        return result, entry, evidence, hidden, approved, recalled

    result, entry, evidence, hidden, approved, recalled = asyncio.run(run())

    assert result.outcome is LearningBatchOutcome.COMPLETED
    assert result.code == "completed"
    assert result.signal_count == 1
    assert result.candidate_count == 1
    assert result.signals[0].outcome is LearningSignalOutcome.CANDIDATE_GENERATED
    assert result.signals[0].candidate_proposal_keys == ("deploy-migrations-first",)
    candidate_result = result.candidates[0]
    assert candidate_result.outcome is LearningCandidateOutcome.PENDING_PERSISTED
    assert candidate_result.decision is not None
    assert candidate_result.decision.verdict is LearningVerdict.ACCEPTED
    assert entry is not None
    assert entry.status is KnowledgeStatus.PENDING
    assert entry.namespace == "project:cayu"
    assert entry.labels == {"project": "cayu"}
    assert entry.created_by == "knowledge_curator"
    assert entry.metadata["cayu_curator"]["proposal_key"] == "deploy-migrations-first"
    assert evidence is not None
    assert evidence.total_evidence_known == 1
    assert evidence.evidence[0].source_id == "signal-1"
    assert evidence.evidence[0].metadata["learning_signal_id"] == "signal-1"
    assert hidden.hits == []
    assert approved.status is KnowledgeStatus.ACTIVE
    assert [hit.entry.id for hit in recalled.hits] == [entry.id]


def test_curator_uses_the_same_publication_path_with_sqlite(tmp_path) -> None:
    async def run():
        store = SQLiteKnowledgeStore(tmp_path / "curator.sqlite", access_scope=_ACCESS_SCOPE)
        try:
            result = await _curator(store).curate(_batch(_signal()))
            candidate_result = result.candidates[0]
            entry_id = _entry_id(candidate_result)
            entry = await store.get_entry(entry_id)
            evidence = await store.read_evidence(entry_id)
            approved = await KnowledgeReviewWorkflow(store, namespace="project:cayu").approve(
                entry_id
            )
            recalled = await store.search(
                KnowledgeQuery(text="migrations service", namespace="project:cayu")
            )
            return result, entry, evidence, approved, recalled
        finally:
            await store.close()

    result, entry, evidence, approved, recalled = asyncio.run(run())

    assert result.candidates[0].outcome is LearningCandidateOutcome.PENDING_PERSISTED
    assert entry is not None and entry.status is KnowledgeStatus.PENDING
    assert evidence is not None and evidence.total_evidence_known == 1
    assert approved.status is KnowledgeStatus.ACTIVE
    assert [hit.entry.id for hit in recalled.hits] == [entry.id]


def test_curator_persists_signal_and_candidate_sources_without_false_signal_metadata() -> None:
    candidate = _candidate().model_copy(
        update={"source_references": (_source("artifact-1", source_hash="sha256:artifact"),)}
    )

    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        result = await _curator(store, generator=_Generator([candidate])).curate(_batch(_signal()))
        evidence = await store.read_evidence(_entry_id(result.candidates[0]))
        return result, evidence

    result, evidence = asyncio.run(run())

    assert result.candidates[0].outcome is LearningCandidateOutcome.PENDING_PERSISTED
    assert evidence is not None and evidence.total_evidence_known == 2
    signal_source, candidate_source = evidence.evidence
    assert signal_source.metadata["learning_source_origin"] == "signal"
    assert signal_source.metadata["learning_signal_id"] == "signal-1"
    assert candidate_source.metadata["learning_source_origin"] == "candidate"
    assert "learning_signal_id" not in candidate_source.metadata


def test_curator_exact_retry_returns_existing_status_without_duplicate_revision() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        generator = _Generator()
        evaluator = _Evaluator()
        curator = _curator(store, generator=generator, evaluator=evaluator)
        first = await curator.curate(_batch(_signal()))
        second = await curator.curate(_batch(_signal()))
        entry_id = _entry_id(first.candidates[0])
        entry = await store.get_entry(entry_id)
        evidence = await store.read_evidence(entry_id)
        return first, second, entry, evidence, generator.calls, evaluator.calls

    first, second, entry, evidence, generator_calls, evaluator_calls = asyncio.run(run())

    assert first.candidates[0].outcome is LearningCandidateOutcome.PENDING_PERSISTED
    assert second.candidates[0].outcome is LearningCandidateOutcome.EXISTING_PENDING
    assert first.candidates[0].entry_id == second.candidates[0].entry_id
    assert entry is not None and entry.revision == 1
    assert evidence is not None and evidence.total_evidence_known == 1
    assert generator_calls == 2
    assert evaluator_calls == ["deploy-migrations-first"]


def test_curator_reports_zero_candidates_without_writing_knowledge() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        result = await _curator(store, generator=_Generator([])).curate(_batch(_signal()))
        listed = await store.list_entries(KnowledgeListQuery(statuses=[KnowledgeStatus.PENDING]))
        return result, listed

    result, listed = asyncio.run(run())

    assert result.outcome is LearningBatchOutcome.COMPLETED
    assert result.candidate_count == 0
    assert result.candidates == ()
    assert result.signals[0].outcome is LearningSignalOutcome.NO_CANDIDATE_GENERATED
    assert result.signals[0].candidate_proposal_keys == ()
    assert listed.entries == []


def test_curator_reprocessing_preserves_active_and_archived_status() -> None:
    async def run():
        active_store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        active_curator = _curator(active_store)
        first = await active_curator.curate(_batch(_signal()))
        active_id = _entry_id(first.candidates[0])
        await KnowledgeReviewWorkflow(active_store, namespace="project:cayu").approve(active_id)
        active = await active_curator.curate(_batch(_signal()))

        archived_store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        archived_curator = _curator(archived_store)
        second = await archived_curator.curate(_batch(_signal()))
        archived_id = _entry_id(second.candidates[0])
        await KnowledgeReviewWorkflow(archived_store, namespace="project:cayu").reject(archived_id)
        archived = await archived_curator.curate(_batch(_signal()))
        return active, archived

    active, archived = asyncio.run(run())

    assert active.candidates[0].outcome is LearningCandidateOutcome.EXISTING_ACTIVE
    assert active.candidates[0].entry_revision == 2
    assert archived.candidates[0].outcome is LearningCandidateOutcome.EXISTING_ARCHIVED
    assert archived.candidates[0].entry_revision == 2


def test_curator_concurrent_exact_processing_converges_on_one_revision() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        first, second = await asyncio.gather(
            _curator(store).curate(_batch(_signal())),
            _curator(store).curate(_batch(_signal())),
        )
        entry_id = _entry_id(first.candidates[0])
        entry = await store.get_entry(entry_id)
        evidence = await store.read_evidence(entry_id)
        return first, second, entry, evidence

    first, second, entry, evidence = asyncio.run(run())

    outcomes = {first.candidates[0].outcome, second.candidates[0].outcome}
    assert outcomes == {
        LearningCandidateOutcome.PENDING_PERSISTED,
        LearningCandidateOutcome.EXISTING_PENDING,
    }
    assert entry is not None and entry.revision == 1
    assert evidence is not None and evidence.total_evidence_known == 1


def test_curator_enforces_evaluator_concurrency_across_concurrent_batches() -> None:
    class BatchGenerator:
        async def generate_candidates(self, batch):
            return [_candidate(f"{batch.id}:{index}") for index in range(3)]

    class TrackingEvaluator:
        def __init__(self) -> None:
            self.active = 0
            self.maximum_active = 0

        async def evaluate_candidate(self, candidate, signals):
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            try:
                await asyncio.sleep(0.01)
                return _accepted()
            finally:
                self.active -= 1

    async def run():
        evaluator = TrackingEvaluator()
        curator = KnowledgeCurator(
            InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE),
            candidate_generator=BatchGenerator(),
            evaluator=evaluator,
            config=_config(max_evaluator_concurrency=2),
            clock=lambda: _NOW,
        )
        first_batch = _batch(_signal())
        second_batch = first_batch.model_copy(update={"id": "batch-2"})
        await asyncio.gather(curator.curate(first_batch), curator.curate(second_batch))
        return evaluator.maximum_active

    assert asyncio.run(run()) == 2


def test_curator_same_proposal_key_with_changed_material_fails_closed() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        first_curator = _curator(store, generator=_Generator([_candidate(text="Original text.")]))
        changed_curator = _curator(store, generator=_Generator([_candidate(text="Changed text.")]))
        first = await first_curator.curate(_batch(_signal()))
        changed = await changed_curator.curate(_batch(_signal()))
        entry = await store.get_entry(_entry_id(first.candidates[0]))
        return first, changed, entry

    first, changed, entry = asyncio.run(run())

    assert first.candidates[0].outcome is LearningCandidateOutcome.PENDING_PERSISTED
    assert changed.candidates[0].outcome is LearningCandidateOutcome.CONFLICT
    assert changed.candidates[0].code == "published_proposal_conflict"
    assert entry is not None and entry.text == "Original text."


def test_curator_generator_failure_and_invalid_output_create_no_entries() -> None:
    class FailingGenerator:
        async def generate_candidates(self, batch):
            raise RuntimeError("private generator diagnostic")

    class InvalidGenerator:
        async def generate_candidates(self, batch):
            return "not typed candidates"

    async def run(generator):
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        curator = KnowledgeCurator(
            store,
            candidate_generator=generator,
            evaluator=_Evaluator(),
            config=_config(),
            clock=lambda: _NOW,
        )
        result = await curator.curate(_batch(_signal()))
        listed = await store.list_entries(KnowledgeListQuery())
        return result, listed

    failed, failed_entries = asyncio.run(run(FailingGenerator()))
    invalid, invalid_entries = asyncio.run(run(InvalidGenerator()))

    assert failed.outcome is LearningBatchOutcome.GENERATOR_FAILED
    assert failed.code == "candidate_generator_failed"
    assert failed.candidates == ()
    assert failed.signals[0].outcome is LearningSignalOutcome.BATCH_FAILED
    assert failed_entries.entries == []
    assert invalid.outcome is LearningBatchOutcome.GENERATOR_INVALID
    assert invalid.code == "candidate_generator_output_invalid"
    assert invalid_entries.entries == []
    assert "private generator diagnostic" not in failed.model_dump_json()


def test_curator_rejects_whole_generator_result_before_any_write() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        candidates = [
            _candidate("valid"),
            _candidate("invalid-kind", kind="generator-invented-kind"),
        ]
        result = await _curator(store, generator=_Generator(candidates)).curate(_batch(_signal()))
        listed = await store.list_entries(KnowledgeListQuery())
        return result, listed

    result, listed = asyncio.run(run())

    assert result.outcome is LearningBatchOutcome.GENERATOR_INVALID
    assert listed.entries == []


def test_curator_rejects_duplicate_proposal_keys_before_evaluation_or_write() -> None:
    evaluator = _Evaluator()

    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        result = await _curator(
            store,
            generator=_Generator([_candidate(), _candidate()]),
            evaluator=evaluator,
        ).curate(_batch(_signal()))
        listed = await store.list_entries(KnowledgeListQuery(statuses=[KnowledgeStatus.PENDING]))
        return result, listed

    result, listed = asyncio.run(run())

    assert result.outcome is LearningBatchOutcome.GENERATOR_INVALID
    assert evaluator.calls == []
    assert listed.entries == []


def test_curator_isolates_evaluator_rejection_and_failure() -> None:
    candidates = [
        _candidate("accepted"),
        _candidate("rejected"),
        _candidate("failed"),
    ]
    evaluator = _Evaluator(
        {
            "accepted": _accepted(),
            "rejected": LearningDecision(
                verdict=LearningVerdict.REJECTED,
                code="unsupported_synthesis",
                notes="The evidence does not support this candidate.",
            ),
            "failed": RuntimeError("private evaluator failure"),
        }
    )

    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        result = await _curator(
            store,
            generator=_Generator(candidates),
            evaluator=evaluator,
        ).curate(_batch(_signal()))
        listed = await store.list_entries(KnowledgeListQuery(statuses=[KnowledgeStatus.PENDING]))
        return result, listed

    result, listed = asyncio.run(run())

    assert [candidate.outcome for candidate in result.candidates] == [
        LearningCandidateOutcome.PENDING_PERSISTED,
        LearningCandidateOutcome.EVALUATOR_REJECTED,
        LearningCandidateOutcome.FAILED,
    ]
    assert result.candidates[2].code == "evaluator_failed"
    assert len(listed.entries) == 1
    assert "private evaluator failure" not in result.model_dump_json()


def test_curator_policy_can_transform_or_reject_without_changing_identity() -> None:
    candidates = [
        _candidate("transform", text="Use the internal deployment checklist."),
        _candidate("reject-me", text="Persist a domain secret."),
    ]

    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        result = await _curator(
            store,
            generator=_Generator(candidates),
            candidate_policy=_Policy(),
            config=_config(policy_identity="test.policy.v1"),
        ).curate(_batch(_signal()))
        entry = await store.get_entry(_entry_id(result.candidates[0]))
        return result, entry

    result, entry = asyncio.run(run())

    assert result.candidates[0].outcome is LearningCandidateOutcome.PENDING_PERSISTED
    assert entry is not None and entry.text == "Use the approved deployment checklist."
    assert result.candidates[1].outcome is LearningCandidateOutcome.POLICY_REJECTED
    assert result.candidates[1].code == "contains_domain_secret"


def test_curator_enforces_input_bounds_before_generator_call() -> None:
    generator = _Generator()
    curator = _curator(
        InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE),
        generator=generator,
        config=_config(max_signal_bytes=128),
    )

    with pytest.raises(ValueError, match="signal exceeds"):
        asyncio.run(curator.curate(_batch(_signal(summary="x" * 200))))

    assert generator.calls == 0


@pytest.mark.parametrize(
    ("config_update", "candidate"),
    [
        ({"max_candidate_text_bytes": 8}, _candidate(text="candidate text is too long")),
        (
            {"max_candidate_title_bytes": 8},
            _candidate().model_copy(update={"title": "candidate title is too long"}),
        ),
    ],
)
def test_curator_rejects_component_specific_candidate_bounds_before_evaluation(
    config_update: dict[str, Any],
    candidate: LearningCandidate,
) -> None:
    evaluator = _Evaluator()

    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        result = await _curator(
            store,
            generator=_Generator([candidate]),
            evaluator=evaluator,
            config=_config(**config_update),
        ).curate(_batch(_signal()))
        listed = await store.list_entries(KnowledgeListQuery(statuses=[KnowledgeStatus.PENDING]))
        return result, listed

    result, listed = asyncio.run(run())

    assert result.outcome is LearningBatchOutcome.GENERATOR_INVALID
    assert evaluator.calls == []
    assert listed.entries == []


def test_curator_enforces_source_reference_byte_bound_before_generator_call() -> None:
    generator = _Generator()
    curator = _curator(
        InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE),
        generator=generator,
        config=_config(max_source_reference_bytes=64),
    )

    with pytest.raises(ValueError, match="source reference exceeds"):
        asyncio.run(curator.curate(_batch(_signal())))

    assert generator.calls == 0


def test_curator_preflights_enriched_evidence_metadata_before_evaluation() -> None:
    signal_id = "i" * 256
    signal = LearningSignal(
        id=signal_id,
        deduplication_key="d" * 256,
        kind="deployment_failure",
        scope="project:cayu",
        summary="A bounded observation with metadata near the evidence ceiling.",
        source_references=(
            LearningSourceReference(
                source_type="session",
                source_id="session-1",
                source_hash="sha256:source-1",
                metadata={"x": "a" * 15_800},
            ),
        ),
        occurred_at=_NOW,
    )
    evaluator = _Evaluator()

    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        result = await _curator(
            store,
            generator=_Generator([_candidate(signal_ids=(signal_id,))]),
            evaluator=evaluator,
            config=_config(
                max_signal_bytes=64 * 1024,
                max_source_reference_bytes=64 * 1024,
            ),
        ).curate(_batch(signal))
        listed = await store.list_entries(KnowledgeListQuery(statuses=[KnowledgeStatus.PENDING]))
        return result, listed

    result, listed = asyncio.run(run())

    assert result.outcome is LearningBatchOutcome.GENERATOR_INVALID
    assert result.code == "candidate_generator_output_invalid"
    assert evaluator.calls == []
    assert listed.entries == []


def test_curator_reconciles_publication_acknowledgement_loss() -> None:
    class AcknowledgementLossStore(InMemoryKnowledgeStore):
        async def publish_entry_revision(
            self,
            entry,
            chunks,
            *,
            evidence=None,
            access_scope=None,
            operation_id,
            expected_revision=None,
        ):
            await super().publish_entry_revision(
                entry,
                chunks,
                evidence=evidence,
                access_scope=access_scope,
                operation_id=operation_id,
                expected_revision=expected_revision,
            )
            raise RuntimeError("private acknowledgement failure")

    async def run():
        store = AcknowledgementLossStore(access_scope=_ACCESS_SCOPE)
        result = await _curator(store).curate(_batch(_signal()))
        entry = await store.get_entry(_entry_id(result.candidates[0]))
        return result, entry

    result, entry = asyncio.run(run())

    candidate = result.candidates[0]
    assert candidate.outcome is LearningCandidateOutcome.PENDING_PERSISTED
    assert candidate.warning_code == "publication_acknowledgement_lost"
    assert entry is not None and entry.status is KnowledgeStatus.PENDING
    assert "private acknowledgement failure" not in result.model_dump_json()


def test_curator_recovers_an_exact_receipt_after_a_mismatched_acknowledgement() -> None:
    class MismatchedAcknowledgementStore(InMemoryKnowledgeStore):
        async def publish_entry_revision(
            self,
            entry,
            chunks,
            *,
            evidence=None,
            access_scope=None,
            operation_id,
            expected_revision=None,
        ):
            receipt = await super().publish_entry_revision(
                entry,
                chunks,
                evidence=evidence,
                access_scope=access_scope,
                operation_id=operation_id,
                expected_revision=expected_revision,
            )
            return receipt.model_copy(update={"request_sha256": "0" * 64})

    async def run():
        store = MismatchedAcknowledgementStore(access_scope=_ACCESS_SCOPE)
        result = await _curator(store).curate(_batch(_signal()))
        entry = await store.get_entry(_entry_id(result.candidates[0]))
        return result, entry

    result, entry = asyncio.run(run())

    candidate = result.candidates[0]
    assert candidate.outcome is LearningCandidateOutcome.PENDING_PERSISTED
    assert candidate.warning_code == "publication_acknowledgement_lost"
    assert entry is not None and entry.revision == 1


def test_curator_does_not_claim_a_competing_payload_as_its_own_publication() -> None:
    class CompetingPayloadStore(InMemoryKnowledgeStore):
        async def publish_entry_revision(
            self,
            entry,
            chunks,
            *,
            evidence=None,
            access_scope=None,
            operation_id,
            expected_revision=None,
        ):
            audit = {
                **entry.metadata["cayu_curator"],
                "batch_id": "competing-batch",
            }
            competing_entry = entry.model_copy(
                update={
                    "metadata": {
                        **entry.metadata,
                        "cayu_curator": audit,
                    }
                }
            )
            await super().publish_entry_revision(
                competing_entry,
                chunks,
                evidence=evidence,
                access_scope=access_scope,
                operation_id=operation_id,
                expected_revision=expected_revision,
            )
            raise RuntimeError("private ambiguous publication failure")

    async def run():
        store = CompetingPayloadStore(access_scope=_ACCESS_SCOPE)
        result = await _curator(store).curate(_batch(_signal()))
        entry = await store.get_entry(_entry_id(result.candidates[0]))
        return result, entry

    result, entry = asyncio.run(run())

    candidate = result.candidates[0]
    assert candidate.outcome is LearningCandidateOutcome.EXISTING_PENDING
    assert candidate.warning_code is None
    assert entry is not None
    assert entry.metadata["cayu_curator"]["batch_id"] == "competing-batch"
    assert "private ambiguous publication failure" not in result.model_dump_json()


def test_curator_rejects_a_receipt_returned_for_another_operation() -> None:
    class MismatchedReceiptStore(InMemoryKnowledgeStore):
        def __init__(self) -> None:
            super().__init__(access_scope=_ACCESS_SCOPE)
            self.return_mismatched_receipt = False

        async def load_entry_publication_receipt(self, operation_id, *, access_scope=None):
            receipt = await super().load_entry_publication_receipt(
                operation_id,
                access_scope=access_scope,
            )
            if receipt is not None and self.return_mismatched_receipt:
                return receipt.model_copy(update={"operation_id": "another-operation"})
            return receipt

    async def run():
        store = MismatchedReceiptStore()
        first = await _curator(store).curate(_batch(_signal()))
        store.return_mismatched_receipt = True
        evaluator = _Evaluator()
        retry = await _curator(store, evaluator=evaluator).curate(_batch(_signal()))
        entry = await store.get_entry(_entry_id(first.candidates[0]))
        return retry, entry, evaluator.calls

    retry, entry, evaluator_calls = asyncio.run(run())

    assert retry.candidates[0].outcome is LearningCandidateOutcome.CONFLICT
    assert retry.candidates[0].code == "publication_receipt_identity_conflict"
    assert evaluator_calls == []
    assert entry is not None and entry.revision == 1


def test_curator_rejects_a_receipt_for_a_non_creation_transition() -> None:
    class NonCreationReceiptStore(InMemoryKnowledgeStore):
        def __init__(self) -> None:
            super().__init__(access_scope=_ACCESS_SCOPE)
            self.return_non_creation_receipt = False

        async def load_entry_publication_receipt(self, operation_id, *, access_scope=None):
            receipt = await super().load_entry_publication_receipt(
                operation_id,
                access_scope=access_scope,
            )
            if receipt is not None and self.return_non_creation_receipt:
                return receipt.model_copy(
                    update={
                        "expected_revision": 1,
                        "entry_revision": 2,
                    }
                )
            return receipt

    async def run():
        store = NonCreationReceiptStore()
        first = await _curator(store).curate(_batch(_signal()))
        store.return_non_creation_receipt = True
        evaluator = _Evaluator()
        retry = await _curator(store, evaluator=evaluator).curate(_batch(_signal()))
        return first, retry, evaluator.calls

    first, retry, evaluator_calls = asyncio.run(run())

    assert first.candidates[0].outcome is LearningCandidateOutcome.PENDING_PERSISTED
    assert retry.candidates[0].outcome is LearningCandidateOutcome.CONFLICT
    assert retry.candidates[0].code == "publication_receipt_identity_conflict"
    assert evaluator_calls == []


def test_curator_keeps_an_owned_publication_alive_after_caller_cancellation() -> None:
    class PausingStore(InMemoryKnowledgeStore):
        def __init__(self) -> None:
            super().__init__(access_scope=_ACCESS_SCOPE)
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.publish_calls = 0

        async def publish_entry_revision(
            self,
            entry,
            chunks,
            *,
            evidence=None,
            access_scope=None,
            operation_id,
            expected_revision=None,
        ):
            self.publish_calls += 1
            self.started.set()
            await self.release.wait()
            return await super().publish_entry_revision(
                entry,
                chunks,
                evidence=evidence,
                access_scope=access_scope,
                operation_id=operation_id,
                expected_revision=expected_revision,
            )

    async def run():
        store = PausingStore()
        curator = _curator(store)
        cancelled = asyncio.create_task(curator.curate(_batch(_signal())))
        await store.started.wait()
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        store.release.set()
        retry = await curator.curate(_batch(_signal()))
        entry = await store.get_entry(_entry_id(retry.candidates[0]))
        return retry, entry, store.publish_calls

    retry, entry, publish_calls = asyncio.run(run())

    assert retry.candidates[0].outcome is LearningCandidateOutcome.EXISTING_PENDING
    assert entry is not None and entry.revision == 1
    assert publish_calls == 1


def test_curator_bounds_publications_that_outlive_their_callers() -> None:
    class CandidatePerBatchGenerator:
        async def generate_candidates(self, batch):
            signal = batch.signals[0]
            return [
                _candidate(
                    f"proposal:{batch.id}",
                    signal_ids=(signal.id,),
                )
            ]

    class PausingStore(InMemoryKnowledgeStore):
        def __init__(self) -> None:
            super().__init__(access_scope=_ACCESS_SCOPE)
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def publish_entry_revision(
            self,
            entry,
            chunks,
            *,
            evidence=None,
            access_scope=None,
            operation_id,
            expected_revision=None,
        ):
            self.started.set()
            await self.release.wait()
            return await super().publish_entry_revision(
                entry,
                chunks,
                evidence=evidence,
                access_scope=access_scope,
                operation_id=operation_id,
                expected_revision=expected_revision,
            )

    async def run():
        store = PausingStore()
        curator = KnowledgeCurator(
            store,
            candidate_generator=CandidatePerBatchGenerator(),
            evaluator=_Evaluator(),
            config=_config(max_in_flight_publications=1),
            clock=lambda: _NOW,
        )
        first = asyncio.create_task(
            curator.curate(LearningBatch(id="first", signals=(_signal("first"),)))
        )
        await store.started.wait()
        second = await curator.curate(LearningBatch(id="second", signals=(_signal("second"),)))
        store.release.set()
        first_result = await first
        return first_result, second

    first, second = asyncio.run(run())

    assert first.candidates[0].outcome is LearningCandidateOutcome.PENDING_PERSISTED
    assert second.candidates[0].outcome is LearningCandidateOutcome.FAILED
    assert second.candidates[0].code == "publication_capacity_exhausted"


def test_curator_isolates_store_failure_between_candidates() -> None:
    class OneCandidateFailsStore(InMemoryKnowledgeStore):
        async def publish_entry_revision(
            self,
            entry,
            chunks,
            *,
            evidence=None,
            access_scope=None,
            operation_id,
            expected_revision=None,
        ):
            if entry.metadata["cayu_curator"]["proposal_key"] == "store-fails":
                raise RuntimeError("private store failure")
            return await super().publish_entry_revision(
                entry,
                chunks,
                evidence=evidence,
                access_scope=access_scope,
                operation_id=operation_id,
                expected_revision=expected_revision,
            )

    async def run():
        store = OneCandidateFailsStore(access_scope=_ACCESS_SCOPE)
        result = await _curator(
            store,
            generator=_Generator([_candidate("store-fails"), _candidate("store-succeeds")]),
        ).curate(_batch(_signal()))
        listed = await store.list_entries(KnowledgeListQuery(statuses=[KnowledgeStatus.PENDING]))
        return result, listed

    result, listed = asyncio.run(run())

    assert [candidate.outcome for candidate in result.candidates] == [
        LearningCandidateOutcome.FAILED,
        LearningCandidateOutcome.PENDING_PERSISTED,
    ]
    assert result.candidates[0].code == "publication_outcome_ambiguous"
    assert len(listed.entries) == 1
    assert "private store failure" not in result.model_dump_json()


def test_curator_fails_closed_on_component_timeouts_and_malformed_evaluation() -> None:
    class SlowGenerator:
        async def generate_candidates(self, batch):
            await asyncio.Event().wait()

    class SlowEvaluator:
        async def evaluate_candidate(self, candidate, signals):
            await asyncio.Event().wait()

    class MalformedEvaluator:
        async def evaluate_candidate(self, candidate, signals):
            return {"verdict": "accepted"}

    async def run():
        generator_store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        generator_timeout = await KnowledgeCurator(
            generator_store,
            candidate_generator=SlowGenerator(),
            evaluator=_Evaluator(),
            config=_config(candidate_generator_timeout_seconds=0.001),
            clock=lambda: _NOW,
        ).curate(_batch(_signal()))

        results = []
        for evaluator in (SlowEvaluator(), MalformedEvaluator()):
            store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
            result = await KnowledgeCurator(
                store,
                candidate_generator=_Generator(),
                evaluator=evaluator,
                config=_config(evaluator_timeout_seconds=0.001),
                clock=lambda: _NOW,
            ).curate(_batch(_signal()))
            listed = await store.list_entries(KnowledgeListQuery())
            results.append((result, listed))
        return generator_timeout, results

    generator_timeout, results = asyncio.run(run())

    assert generator_timeout.outcome is LearningBatchOutcome.GENERATOR_TIMED_OUT
    assert [result.candidates[0].code for result, _listed in results] == [
        "evaluator_timed_out",
        "evaluator_failed",
    ]
    assert all(listed.entries == [] for _result, listed in results)


def test_candidate_policy_cannot_expand_complete_provenance_past_the_bound() -> None:
    class ExpandingPolicy:
        async def apply_candidate_policy(self, candidate, signals):
            return KnowledgeCandidatePolicyDecision(
                disposition=CandidatePolicyDisposition.ACCEPTED,
                code="add-source",
                candidate=candidate.model_copy(
                    update={"source_references": (_source("extra-source"),)}
                ),
            )

    async def run():
        store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        result = await _curator(
            store,
            candidate_policy=ExpandingPolicy(),
            config=_config(
                policy_identity="test.expanding-policy.v1",
                max_source_references_per_candidate=1,
            ),
        ).curate(_batch(_signal()))
        listed = await store.list_entries(KnowledgeListQuery())
        return result, listed

    result, listed = asyncio.run(run())

    assert result.candidates[0].outcome is LearningCandidateOutcome.FAILED
    assert result.candidates[0].code == "candidate_policy_output_invalid"
    assert listed.entries == []


def test_group_learning_signals_is_scope_partitioned_bounded_and_deterministic() -> None:
    signals = [
        _signal("b", scope="scope:b", occurred_at=_NOW + timedelta(seconds=2)),
        _signal("a2", scope="scope:a", occurred_at=_NOW + timedelta(seconds=1)),
        _signal("a1", scope="scope:a", occurred_at=_NOW),
    ]

    first = group_learning_signals(signals, max_signals_per_batch=1)
    second = group_learning_signals(reversed(signals), max_signals_per_batch=1)

    assert [(batch.scope, batch.signals[0].id) for batch in first] == [
        ("scope:a", "a1"),
        ("scope:a", "a2"),
        ("scope:b", "b"),
    ]
    assert [batch.id for batch in first] == [batch.id for batch in second]

    with pytest.raises(ValueError, match="repeat an id"):
        group_learning_signals(
            [_signal("duplicate"), _signal("duplicate")],
            max_signals_per_batch=1,
        )


def test_curator_contracts_are_strict_and_copy_nested_input() -> None:
    locator = {"event": {"ordinal": 1}}
    reference = LearningSourceReference(
        source_type="session",
        source_id="session-1",
        source_hash="sha256:source",
        locator=locator,
    )
    locator["event"]["ordinal"] = 2

    assert reference.locator == {"event": {"ordinal": 1}}
    with pytest.raises(ValidationError):
        LearningSignal.model_validate(
            {
                **_signal().model_dump(mode="python"),
                "unexpected": "rejected",
            }
        )


def test_curator_result_contracts_reject_impossible_success_and_broken_linkage() -> None:
    fingerprint = "0" * 64
    accepted = _accepted()

    with pytest.raises(ValidationError, match="requires an entry projection"):
        LearningCandidateResult(
            proposal_key="persisted-without-entry",
            candidate_fingerprint=fingerprint,
            outcome=LearningCandidateOutcome.PENDING_PERSISTED,
            code="pending_persisted",
            decision=accepted,
        )

    with pytest.raises(ValidationError, match="does not match.*entry status"):
        LearningCandidateResult(
            proposal_key="persisted-with-wrong-status",
            candidate_fingerprint=fingerprint,
            outcome=LearningCandidateOutcome.PENDING_PERSISTED,
            code="pending_persisted",
            decision=accepted,
            entry_id="entry-1",
            entry_revision=1,
            entry_status=KnowledgeStatus.ACTIVE,
        )

    with pytest.raises(ValidationError, match="requires a rejected decision"):
        LearningCandidateResult(
            proposal_key="rejected-with-accepted-decision",
            candidate_fingerprint=fingerprint,
            outcome=LearningCandidateOutcome.EVALUATOR_REJECTED,
            code="rejected",
            decision=accepted,
        )

    persisted = LearningCandidateResult(
        proposal_key="persisted",
        candidate_fingerprint=fingerprint,
        outcome=LearningCandidateOutcome.PENDING_PERSISTED,
        code="pending_persisted",
        decision=accepted,
        entry_id="entry-1",
        entry_revision=1,
        entry_status=KnowledgeStatus.PENDING,
    )
    unrelated_signal = LearningSignalResult(
        signal_id="signal-1",
        signal_fingerprint=fingerprint,
        outcome=LearningSignalOutcome.CANDIDATE_GENERATED,
        code="candidate_generated",
        candidate_proposal_keys=("not-returned",),
    )
    with pytest.raises(ValidationError, match="reference exactly the returned candidates"):
        LearningBatchResult(
            batch_id="batch-1",
            batch_fingerprint=fingerprint,
            configuration_fingerprint=fingerprint,
            scope="project:cayu",
            outcome=LearningBatchOutcome.COMPLETED,
            code="completed",
            signal_count=1,
            candidate_count=1,
            signals=(unrelated_signal,),
            candidates=(persisted,),
            processed_at=_NOW,
        )


def test_curator_requires_separate_generator_and_evaluator() -> None:
    class SelfApprovingComponent:
        async def generate_candidates(self, batch):
            return [_candidate()]

        async def evaluate_candidate(self, candidate, signals):
            return _accepted()

    component = SelfApprovingComponent()
    with pytest.raises(ValueError, match="separate components"):
        KnowledgeCurator(
            InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE),
            candidate_generator=component,
            evaluator=component,
            config=_config(),
        )


def test_curator_requires_truthful_optional_policy_identity() -> None:
    store = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)

    with pytest.raises(ValueError, match="requires an explicit"):
        KnowledgeCurator(
            store,
            candidate_generator=_Generator(),
            evaluator=_Evaluator(),
            candidate_policy=_Policy(),
            config=_config(),
        )
    with pytest.raises(ValueError, match="requires a candidate policy"):
        KnowledgeCurator(
            store,
            candidate_generator=_Generator(),
            evaluator=_Evaluator(),
            config=_config(policy_identity="unused.policy.v1"),
        )


def test_curator_defensively_copies_configuration() -> None:
    labels = {"project": "cayu"}
    config = KnowledgeCuratorConfig(
        candidate_generator_identity="test.generator.v1",
        evaluator_identity="test.evaluator.v1",
        namespace="project:cayu",
        labels=labels,
    )
    curator = _curator(
        InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE),
        config=config,
    )

    labels["project"] = "caller-mutated"
    exposed = curator.config
    exposed.labels["project"] = "result-mutated"

    assert curator.config.labels == {"project": "cayu"}


def test_curator_rejects_non_durable_chunk_configuration_at_construction() -> None:
    with pytest.raises(ValidationError, match="chunk_target_bytes.*at most"):
        _config(
            chunk_target_bytes=2**70,
            chunk_overlap_bytes=2**69,
        )


def test_curator_rejects_fixed_configuration_outside_store_scope() -> None:
    scope = KnowledgeAccessScope.for_namespace(
        "project:other",
        required_labels={"tenant": "other"},
        allowed_statuses=[KnowledgeStatus.PENDING],
    )

    with pytest.raises(ValueError, match="namespace is outside"):
        _curator(InMemoryKnowledgeStore(access_scope=scope))
