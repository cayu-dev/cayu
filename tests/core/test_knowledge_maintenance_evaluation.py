from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from pathlib import Path

import pytest

import cayu.evals as evals
from cayu.evals.knowledge_maintenance import (
    KnowledgeMaintenanceEvaluationCorpus,
    KnowledgeMaintenanceEvaluationResult,
    KnowledgeMaintenanceEvaluationScenario,
    load_knowledge_maintenance_evaluation_corpus,
    run_knowledge_maintenance_evaluation,
)
from cayu.knowledge_maintenance_persistence import (
    KnowledgeMaintenanceProposalPublicationConflict,
    KnowledgeMaintenanceProposalPublicationOutcome,
)
from cayu.recall import RECALL_MAX_QUERY_BYTES
from cayu.storage import (
    MAX_KNOWLEDGE_MAINTENANCE_SOURCES,
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeEvidenceResult,
    KnowledgeLineageCurrentness,
    KnowledgeLineageQuery,
    KnowledgeLineageRole,
    KnowledgeMaintenanceDecisionKind,
    KnowledgeMaintenanceStale,
    KnowledgeRelation,
    KnowledgeRelationKind,
    KnowledgeRevisionRef,
    KnowledgeStatus,
    SQLiteKnowledgeStore,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CORPUS_PATH = _REPOSITORY_ROOT / "benchmarks/memory/knowledge-maintenance-corpus-v1.json"
_RESULTS_PATH = (
    _REPOSITORY_ROOT / "benchmarks/memory/knowledge-maintenance-evaluation-results-v1.json"
)


class _DecisionAwareEvidenceStore(InMemoryKnowledgeStore):
    def __init__(self) -> None:
        super().__init__()
        self._review_attempted: set[str] = set()

    async def apply_maintenance_decision(self, proposal, decision, *, access_scope=None):
        self._review_attempted.add(proposal.replacement.entry_id)
        return await super().apply_maintenance_decision(
            proposal,
            decision,
            access_scope=access_scope,
        )

    async def read_evidence(self, entry_id, **kwargs):
        if entry_id not in self._review_attempted:
            raise AssertionError("Evaluation evidence was read before the review attempt.")
        return await super().read_evidence(entry_id, **kwargs)


class _CorruptEvidenceRevisionStore(_DecisionAwareEvidenceStore):
    async def read_evidence(self, entry_id, **kwargs):
        result = await super().read_evidence(entry_id, **kwargs)
        if result is None:
            return None
        return result.model_copy(
            update={
                "evidence": [
                    item.model_copy(update={"source_revision": "wrong-revision"})
                    for item in result.evidence
                ]
            }
        )


class _CorruptEvidenceBindingStore(_DecisionAwareEvidenceStore):
    async def read_evidence(self, entry_id, **kwargs):
        result = await super().read_evidence(entry_id, **kwargs)
        if result is None:
            return None
        return result.model_copy(
            update={
                "evidence": [
                    item.model_copy(
                        update={
                            "source_hash": "0" * 64,
                            "locator": {
                                "entry_id": item.source_id,
                                "revision": 999,
                            },
                        }
                    )
                    for item in result.evidence
                ]
            }
        )


class _MissingActiveSuccessorEvidenceStore(_DecisionAwareEvidenceStore):
    async def read_evidence(self, entry_id, **kwargs):
        result = await super().read_evidence(entry_id, **kwargs)
        if result is None or result.entry_revision == 1:
            return result
        return result.model_copy(update={"evidence": [], "total_evidence_known": 0})


class _CorruptRecallMaterialStore(_DecisionAwareEvidenceStore):
    async def search(self, query, *, access_scope=None):
        result = await super().search(query, access_scope=access_scope)
        hits = []
        for hit in result.hits:
            if hit.entry.source_type != "knowledge_maintenance":
                hits.append(hit)
                continue
            update: dict[str, object] = {
                "entry": hit.entry.model_copy(update={"text": "CORRUPTED RECALL CONTENT"})
            }
            if hit.chunk is not None:
                update["chunk"] = hit.chunk.model_copy(update={"text": "CORRUPTED RECALL CONTENT"})
            hits.append(hit.model_copy(update=update))
        return result.model_copy(update={"hits": hits})


class _UnexpectedApprovalEntryStore(_DecisionAwareEvidenceStore):
    async def apply_maintenance_decision(self, proposal, decision, *, access_scope=None):
        receipt = await super().apply_maintenance_decision(
            proposal,
            decision,
            access_scope=access_scope,
        )
        if decision.kind is KnowledgeMaintenanceDecisionKind.APPROVE:
            source = await super().get_entry(
                proposal.sources[0].entry_id,
                access_scope=KnowledgeAccessScope.privileged(),
            )
            assert source is not None
            await super().create_entry(
                KnowledgeEntry(
                    id="unexpected-maintenance-write",
                    text="Unrelated hidden material written during approval.",
                    namespace=source.namespace,
                    labels=dict(source.labels),
                    visibility=source.visibility,
                    status=KnowledgeStatus.ACTIVE,
                    created_by_type=source.created_by_type,
                    created_by="adversarial-test",
                    created_at=receipt.committed_at,
                    updated_at=receipt.committed_at,
                    source_type="adversarial_test",
                    source_id="unexpected-maintenance-write",
                ),
                access_scope=KnowledgeAccessScope.privileged(),
            )
        return receipt


class _UnknownListTotalStore(_DecisionAwareEvidenceStore):
    async def list_entries(self, query, *, access_scope=None):
        result = await super().list_entries(query, access_scope=access_scope)
        return result.model_copy(update={"total_entries_known": None})


class _WrongPublicationReceiptStore(_DecisionAwareEvidenceStore):
    async def publish_maintenance_proposal(self, *args, **kwargs):
        receipt = await super().publish_maintenance_proposal(*args, **kwargs)
        return receipt.model_copy(update={"request_sha256": "0" * 64})


class _WrongDurablePublicationStore(_DecisionAwareEvidenceStore):
    async def load_maintenance_proposal_publication(self, *args, **kwargs):
        publication = await super().load_maintenance_proposal_publication(*args, **kwargs)
        if publication is None:
            return None
        return publication.model_copy(
            update={"outcome": KnowledgeMaintenanceProposalPublicationOutcome.EXISTING_DECIDED}
        )


class _MutatingRejectStore(_DecisionAwareEvidenceStore):
    async def apply_maintenance_decision(self, proposal, decision, *, access_scope=None):
        receipt = await super().apply_maintenance_decision(
            proposal,
            decision,
            access_scope=access_scope,
        )
        if decision.kind is KnowledgeMaintenanceDecisionKind.REJECT:
            source = await super().get_entry(
                proposal.sources[0].entry_id,
                access_scope=KnowledgeAccessScope.privileged(),
            )
            assert source is not None
            await super().append_entry_revision(
                source.model_copy(
                    update={
                        "revision": source.revision + 1,
                        "text": source.text + " Unexpected rejection mutation.",
                        "updated_at": source.updated_at + timedelta(seconds=1),
                    }
                ),
                expected_revision=source.revision,
                access_scope=KnowledgeAccessScope.privileged(),
            )
        return receipt


class _WrongStaleReasonStore(_DecisionAwareEvidenceStore):
    async def apply_maintenance_decision(self, proposal, decision, *, access_scope=None):
        try:
            return await super().apply_maintenance_decision(
                proposal,
                decision,
                access_scope=access_scope,
            )
        except KnowledgeMaintenanceStale as exc:
            raise KnowledgeMaintenanceStale("relation_endpoint") from exc


class _MissingArchivedHistoryStore(_DecisionAwareEvidenceStore):
    async def get_entry(
        self,
        entry_id,
        *,
        revision=None,
        max_bytes=None,
        access_scope=None,
    ):
        entry = await super().get_entry(
            entry_id,
            revision=revision,
            max_bytes=max_bytes,
            access_scope=access_scope,
        )
        if entry_id != "historical-lineage:source" or revision != 1:
            return entry
        current = await super().get_entry(
            entry_id,
            access_scope=KnowledgeAccessScope.privileged(),
        )
        return None if current is not None and current.status is KnowledgeStatus.ARCHIVED else entry


class _WrongAppliedReceiptStore(_DecisionAwareEvidenceStore):
    def __init__(self) -> None:
        super().__init__()
        self._wrong_receipts = {}

    async def apply_maintenance_decision(self, proposal, decision, *, access_scope=None):
        receipt = await super().apply_maintenance_decision(
            proposal,
            decision,
            access_scope=access_scope,
        )
        if receipt.replacement is None:
            return receipt
        wrong = receipt.model_copy(
            update={
                "replacement": receipt.replacement.model_copy(
                    update={"revision": receipt.replacement.revision - 1}
                )
            }
        )
        self._wrong_receipts[receipt.operation_id] = wrong
        return wrong

    async def load_maintenance_decision_receipt(
        self,
        operation_id,
        *,
        access_scope=None,
    ):
        receipt = await super().load_maintenance_decision_receipt(
            operation_id,
            access_scope=access_scope,
        )
        return self._wrong_receipts.get(operation_id, receipt)


class _WrongRejectedReceiptStore(_DecisionAwareEvidenceStore):
    def __init__(self) -> None:
        super().__init__()
        self._wrong_receipts = {}

    async def apply_maintenance_decision(self, proposal, decision, *, access_scope=None):
        receipt = await super().apply_maintenance_decision(
            proposal,
            decision,
            access_scope=access_scope,
        )
        if receipt.replacement is not None:
            return receipt
        wrong = receipt.model_copy(update={"proposal_id": "wrong-review-proposal"})
        self._wrong_receipts[receipt.operation_id] = wrong
        return wrong

    async def load_maintenance_decision_receipt(
        self,
        operation_id,
        *,
        access_scope=None,
    ):
        receipt = await super().load_maintenance_decision_receipt(
            operation_id,
            access_scope=access_scope,
        )
        return self._wrong_receipts.get(operation_id, receipt)


class _MutatingAppliedSourceStore(_DecisionAwareEvidenceStore):
    async def apply_maintenance_decision(self, proposal, decision, *, access_scope=None):
        receipt = await super().apply_maintenance_decision(
            proposal,
            decision,
            access_scope=access_scope,
        )
        if receipt.archived_revisions:
            reference = receipt.archived_revisions[0]
            source = await super().get_entry(
                reference.entry_id,
                access_scope=KnowledgeAccessScope.privileged(),
            )
            assert source is not None
            await super().append_entry_revision(
                source.model_copy(
                    update={
                        "revision": source.revision + 1,
                        "text": source.text + " Unexpected post-approval mutation.",
                        "updated_at": source.updated_at + timedelta(seconds=1),
                    }
                ),
                expected_revision=source.revision,
                access_scope=KnowledgeAccessScope.privileged(),
            )
        return receipt


class _WrongEvidenceTargetStore(_DecisionAwareEvidenceStore):
    async def read_evidence(self, entry_id, **kwargs):
        result = await super().read_evidence(entry_id, **kwargs)
        if result is None:
            return None
        foreign_entry_id = "foreign-valid-evidence-target"
        return KnowledgeEvidenceResult(
            entry_id=foreign_entry_id,
            entry_revision=1,
            evidence=[
                item.model_copy(
                    update={
                        "entry_id": foreign_entry_id,
                        "entry_revision": 1,
                    }
                )
                for item in result.evidence
            ],
            truncated=result.truncated,
            limit=result.limit,
            max_bytes=result.max_bytes,
            total_evidence_known=result.total_evidence_known,
        )


class _WrongLineageRevisionStore(_DecisionAwareEvidenceStore):
    async def inspect_lineage(self, query, *, access_scope=None):
        result = await super().inspect_lineage(query, access_scope=access_scope)
        if result is None or not any(
            link.role is KnowledgeLineageRole.SUPERSEDES for link in result.links
        ):
            return result
        return result.model_copy(
            update={
                "links": [
                    link.model_copy(
                        update={
                            "counterpart": link.counterpart_current,
                            "currentness": KnowledgeLineageCurrentness.CURRENT,
                        }
                    )
                    if link.role is KnowledgeLineageRole.SUPERSEDES
                    else link
                    for link in result.links
                ]
            }
        )


class _WrongLineageQueryStore(_DecisionAwareEvidenceStore):
    async def inspect_lineage(self, query, *, access_scope=None):
        if (
            query.reference.entry_id == "reviewer-rejection:a"
            and access_scope == KnowledgeAccessScope.privileged()
        ):
            query = KnowledgeLineageQuery(
                reference=KnowledgeRevisionRef(
                    entry_id="reviewer-rejection:b",
                    revision=1,
                ),
                limit=query.limit,
                max_bytes=query.max_bytes,
            )
        return await super().inspect_lineage(query, access_scope=access_scope)


class _CrossCaseMutationStore(_DecisionAwareEvidenceStore):
    async def apply_maintenance_decision(self, proposal, decision, *, access_scope=None):
        receipt = await super().apply_maintenance_decision(
            proposal,
            decision,
            access_scope=access_scope,
        )
        if decision.kind is KnowledgeMaintenanceDecisionKind.APPROVE and any(
            source.entry_id == "historical-lineage:source" for source in proposal.sources
        ):
            earlier = await super().get_entry(
                "duplicate-merge:distractor",
                access_scope=KnowledgeAccessScope.privileged(),
            )
            assert earlier is not None
            await super().append_entry_revision(
                earlier.model_copy(
                    update={
                        "revision": earlier.revision + 1,
                        "text": earlier.text + " Corrupted by a later evaluation case.",
                        "updated_at": earlier.updated_at + timedelta(seconds=1),
                    }
                ),
                expected_revision=earlier.revision,
                access_scope=KnowledgeAccessScope.privileged(),
            )
        return receipt


class _ForeignTerminalWriteStore(_DecisionAwareEvidenceStore):
    async def apply_maintenance_decision(self, proposal, decision, *, access_scope=None):
        receipt = await super().apply_maintenance_decision(
            proposal,
            decision,
            access_scope=access_scope,
        )
        if decision.kind is KnowledgeMaintenanceDecisionKind.APPROVE and any(
            source.entry_id == "historical-lineage:source" for source in proposal.sources
        ):
            source = await super().get_entry(
                proposal.sources[0].entry_id,
                access_scope=KnowledgeAccessScope.privileged(),
            )
            assert source is not None
            await super().create_entry(
                KnowledgeEntry(
                    id="foreign-terminal-write",
                    text="Unexpected write outside every evaluation case namespace.",
                    namespace="outside:evaluation:cases",
                    labels=dict(source.labels),
                    visibility=source.visibility,
                    status=KnowledgeStatus.ACTIVE,
                    created_by_type=source.created_by_type,
                    created_by="adversarial-test",
                    created_at=receipt.committed_at,
                    updated_at=receipt.committed_at,
                    source_type="adversarial_test",
                    source_id="foreign-terminal-write",
                ),
                access_scope=KnowledgeAccessScope.privileged(),
            )
        return receipt


class _CrossCaseEvidenceMutationStore(_DecisionAwareEvidenceStore):
    def __init__(self) -> None:
        super().__init__()
        self._earlier_replacement_id: str | None = None
        self._final_case_finished = False

    async def apply_maintenance_decision(self, proposal, decision, *, access_scope=None):
        receipt = await super().apply_maintenance_decision(
            proposal,
            decision,
            access_scope=access_scope,
        )
        if any(source.entry_id == "duplicate-merge:a" for source in proposal.sources):
            self._earlier_replacement_id = proposal.replacement.entry_id
        if any(source.entry_id == "historical-lineage:source" for source in proposal.sources):
            self._final_case_finished = True
        return receipt

    async def read_evidence(self, entry_id, **kwargs):
        result = await super().read_evidence(entry_id, **kwargs)
        if (
            result is not None
            and self._final_case_finished
            and entry_id == self._earlier_replacement_id
        ):
            return result.model_copy(update={"evidence": [], "total_evidence_known": 0})
        return result


class _CrossCaseReceiptMutationStore(_DecisionAwareEvidenceStore):
    def __init__(self) -> None:
        super().__init__()
        self._earlier_operation_id: str | None = None
        self._final_case_finished = False

    async def apply_maintenance_decision(self, proposal, decision, *, access_scope=None):
        receipt = await super().apply_maintenance_decision(
            proposal,
            decision,
            access_scope=access_scope,
        )
        if any(source.entry_id == "duplicate-merge:a" for source in proposal.sources):
            self._earlier_operation_id = decision.operation_id
        if any(source.entry_id == "historical-lineage:source" for source in proposal.sources):
            self._final_case_finished = True
        return receipt

    async def load_maintenance_decision_receipt(
        self,
        operation_id,
        *,
        access_scope=None,
    ):
        receipt = await super().load_maintenance_decision_receipt(
            operation_id,
            access_scope=access_scope,
        )
        if (
            receipt is not None
            and self._final_case_finished
            and operation_id == self._earlier_operation_id
        ):
            return receipt.model_copy(update={"proposal_id": "corrupted-terminal-proposal"})
        return receipt


class _CrossCasePublicationMutationStore(_DecisionAwareEvidenceStore):
    def __init__(self) -> None:
        super().__init__()
        self._earlier_proposal_id: str | None = None
        self._final_case_finished = False

    async def apply_maintenance_decision(self, proposal, decision, *, access_scope=None):
        receipt = await super().apply_maintenance_decision(
            proposal,
            decision,
            access_scope=access_scope,
        )
        if any(source.entry_id == "duplicate-merge:a" for source in proposal.sources):
            self._earlier_proposal_id = proposal.id
        if any(source.entry_id == "historical-lineage:source" for source in proposal.sources):
            self._final_case_finished = True
        return receipt

    async def load_maintenance_proposal_publication(
        self,
        proposal_id,
        *,
        access_scope=None,
    ):
        publication = await super().load_maintenance_proposal_publication(
            proposal_id,
            access_scope=access_scope,
        )
        if (
            publication is not None
            and self._final_case_finished
            and proposal_id == self._earlier_proposal_id
        ):
            return publication.model_copy(
                update={"outcome": KnowledgeMaintenanceProposalPublicationOutcome.EXISTING_PENDING}
            )
        return publication


class _CrossCaseRecallLossStore(_DecisionAwareEvidenceStore):
    def __init__(self) -> None:
        super().__init__()
        self._final_case_finished = False

    async def apply_maintenance_decision(self, proposal, decision, *, access_scope=None):
        receipt = await super().apply_maintenance_decision(
            proposal,
            decision,
            access_scope=access_scope,
        )
        if any(source.entry_id == "historical-lineage:source" for source in proposal.sources):
            self._final_case_finished = True
        return receipt

    async def search(self, query, *, access_scope=None):
        result = await super().search(query, access_scope=access_scope)
        if (
            self._final_case_finished
            and query.namespace == "evaluation:knowledge-maintenance:duplicate-merge"
        ):
            return result.model_copy(update={"hits": [], "total_hits_known": 0, "truncated": False})
        return result


class _CrossCaseHistoricalRevisionLossStore(_DecisionAwareEvidenceStore):
    def __init__(self) -> None:
        super().__init__()
        self._final_case_finished = False

    async def apply_maintenance_decision(self, proposal, decision, *, access_scope=None):
        receipt = await super().apply_maintenance_decision(
            proposal,
            decision,
            access_scope=access_scope,
        )
        if any(source.entry_id == "historical-lineage:source" for source in proposal.sources):
            self._final_case_finished = True
        return receipt

    async def get_entry(
        self,
        entry_id,
        *,
        revision=None,
        max_bytes=None,
        access_scope=None,
    ):
        entry = await super().get_entry(
            entry_id,
            revision=revision,
            max_bytes=max_bytes,
            access_scope=access_scope,
        )
        if self._final_case_finished and entry_id == "duplicate-merge:a" and revision == 1:
            return None
        return entry


class _CrossCaseChunkLossStore(_DecisionAwareEvidenceStore):
    def __init__(self) -> None:
        super().__init__()
        self._duplicate_replacement_id: str | None = None
        self._final_case_finished = False

    async def apply_maintenance_decision(self, proposal, decision, *, access_scope=None):
        receipt = await super().apply_maintenance_decision(
            proposal,
            decision,
            access_scope=access_scope,
        )
        if any(source.entry_id == "duplicate-merge:a" for source in proposal.sources):
            self._duplicate_replacement_id = proposal.replacement.entry_id
        if any(source.entry_id == "historical-lineage:source" for source in proposal.sources):
            self._final_case_finished = True
        return receipt

    async def read_chunks(self, entry_id, **kwargs):
        chunks = await super().read_chunks(entry_id, **kwargs)
        if self._final_case_finished and entry_id == self._duplicate_replacement_id:
            return []
        return chunks


class _SlowGlobalTerminalAuditStore(InMemoryKnowledgeStore):
    terminal_audit_delay_seconds = 0.2

    async def list_entries(self, query, *, access_scope=None):
        if query.namespace is None and query.limit > 1:
            await asyncio.sleep(self.terminal_audit_delay_seconds)
        return await super().list_entries(query, access_scope=access_scope)


class _RejectedRelationMutationStore(_DecisionAwareEvidenceStore):
    async def apply_maintenance_decision(self, proposal, decision, *, access_scope=None):
        receipt = await super().apply_maintenance_decision(
            proposal,
            decision,
            access_scope=access_scope,
        )
        if decision.kind is KnowledgeMaintenanceDecisionKind.REJECT:
            relation = KnowledgeRelation(
                id="adversarial-rejected-replacement-distractor",
                subject=proposal.replacement,
                object=KnowledgeRevisionRef(
                    entry_id="reviewer-rejection:distractor",
                    revision=1,
                ),
                kind=KnowledgeRelationKind.DERIVED_FROM,
                created_by="adversarial-test",
                created_at=receipt.committed_at + timedelta(microseconds=1),
            )
            await super().publish_relations(
                [relation],
                operation_id="adversarial-rejected-relation-publication",
                access_scope=KnowledgeAccessScope.privileged(),
            )
        return receipt


class _TruncatedApprovedLineageStore(_DecisionAwareEvidenceStore):
    async def apply_maintenance_decision(self, proposal, decision, *, access_scope=None):
        receipt = await super().apply_maintenance_decision(
            proposal,
            decision,
            access_scope=access_scope,
        )
        if (
            decision.kind is KnowledgeMaintenanceDecisionKind.APPROVE
            and len(proposal.relations) == MAX_KNOWLEDGE_MAINTENANCE_SOURCES
        ):
            first = proposal.relations[0]
            relation = KnowledgeRelation(
                id="adversarial-hidden-fifty-first-relation",
                subject=first.subject,
                object=first.object,
                kind=KnowledgeRelationKind.DERIVED_FROM,
                created_by="adversarial-test",
                created_at=receipt.committed_at + timedelta(microseconds=1),
            )
            await super().publish_relations(
                [relation],
                operation_id="adversarial-hidden-lineage-publication",
                access_scope=KnowledgeAccessScope.privileged(),
            )
        return receipt


def test_reference_maintenance_evaluation_is_reproducible_across_builtin_backends(
    tmp_path: Path,
) -> None:
    async def run() -> list[KnowledgeMaintenanceEvaluationResult]:
        corpus = load_knowledge_maintenance_evaluation_corpus(_CORPUS_PATH)
        memory = await run_knowledge_maintenance_evaluation(
            corpus,
            InMemoryKnowledgeStore(),
            backend="memory",
        )
        sqlite_store = SQLiteKnowledgeStore(tmp_path / "maintenance-evaluation.sqlite")
        try:
            sqlite = await run_knowledge_maintenance_evaluation(
                corpus,
                sqlite_store,
                backend="sqlite",
            )
        finally:
            await sqlite_store.close()
        return [memory, sqlite]

    actual = asyncio.run(run())
    checked_payload = json.loads(_RESULTS_PATH.read_text(encoding="utf-8"))
    checked = [
        KnowledgeMaintenanceEvaluationResult.model_validate(item)
        for item in checked_payload["results"]
    ]

    assert checked_payload["schema_version"] == ("cayu.knowledge_maintenance_evaluation_matrix.v1")
    assert checked_payload["corpus_revision"] == "knowledge-maintenance-reference-v1"
    assert [result.backend for result in actual] == ["memory", "sqlite"]
    for result, frozen in zip(actual, checked, strict=True):
        assert _without_latency(result) == _without_latency(frozen)
        assert result.metrics.routing_precision == 1.0
        assert result.metrics.routing_recall == 1.0
        assert result.metrics.information_retention == 1.0
        assert result.metrics.evidence_retention == 1.0
        assert result.metrics.unsafe_acceptance_rate == 0.0
        assert result.metrics.lifecycle_correctness == 1.0
        assert result.metrics.lineage_correctness == 1.0
        assert result.metrics.model_call_count == 0
        assert result.metrics.latency_p50_ms >= 0.0
        assert result.metrics.latency_p95_ms >= result.metrics.latency_p50_ms
        assert {case.scenario for case in result.cases} == set(
            KnowledgeMaintenanceEvaluationScenario
        )
    assert [_without_latency_case(case) for case in actual[0].cases] == [
        _without_latency_case(case) for case in actual[1].cases
    ]


def test_maintenance_evaluation_requires_empty_store_and_runner_configuration() -> None:
    async def run() -> None:
        corpus = load_knowledge_maintenance_evaluation_corpus(_CORPUS_PATH)
        store = InMemoryKnowledgeStore()
        await store.create_entry(
            KnowledgeEntry(id="existing", text="existing knowledge"),
            access_scope=KnowledgeAccessScope.privileged(),
        )
        with pytest.raises(ValueError, match="requires an empty store"):
            await run_knowledge_maintenance_evaluation(
                corpus,
                store,
                backend="memory",
            )
        with pytest.raises(ValueError, match="runner-owned field 'provider_calls'"):
            await run_knowledge_maintenance_evaluation(
                corpus,
                InMemoryKnowledgeStore(),
                backend="memory",
                configuration={"provider_calls": True},
            )
        with pytest.raises(ValueError, match="configuration exceeds"):
            await run_knowledge_maintenance_evaluation(
                corpus,
                InMemoryKnowledgeStore(),
                backend="memory",
                configuration={"oversized": "x" * (64 * 1024)},
            )

    asyncio.run(run())


def test_private_maintenance_corpus_uses_the_same_bounded_contract(tmp_path: Path) -> None:
    payload = json.loads(_CORPUS_PATH.read_text(encoding="utf-8"))
    payload["origin"] = "external_private"
    private_path = tmp_path / "private-maintenance-corpus.json"
    private_path.write_text(json.dumps(payload), encoding="utf-8")

    corpus = load_knowledge_maintenance_evaluation_corpus(private_path)

    assert corpus.origin == "external_private"
    assert len(corpus.cases) == 6


def test_maintenance_corpus_rejects_a_scenario_with_weakened_safety(
    tmp_path: Path,
) -> None:
    payload = json.loads(_CORPUS_PATH.read_text(encoding="utf-8"))
    unresolved = next(
        case for case in payload["cases"] if case["scenario"] == "unresolved_contradiction"
    )
    unresolved["evaluator_verdict"] = "accepted"
    unresolved["review_decision"] = "approve"
    malformed_path = tmp_path / "unsafe-maintenance-corpus.json"
    malformed_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="evaluator_verdict conflicts"):
        load_knowledge_maintenance_evaluation_corpus(malformed_path)


def test_maintenance_corpus_and_lineage_share_the_executable_source_bound() -> None:
    with pytest.raises(ValueError, match="executable maintenance source bound"):
        _wide_maintenance_corpus(MAX_KNOWLEDGE_MAINTENANCE_SOURCES + 1)

    result = asyncio.run(
        run_knowledge_maintenance_evaluation(
            _wide_maintenance_corpus(MAX_KNOWLEDGE_MAINTENANCE_SOURCES),
            InMemoryKnowledgeStore(),
            backend="memory",
        )
    )

    assert result.cases[0].storage_outcome == "applied"
    assert result.cases[0].lineage_correct is True
    assert result.metrics.lineage_correctness == 1.0


def test_maintenance_corpus_rejects_non_executable_fixture_identities() -> None:
    payload = _wide_maintenance_corpus(1).model_dump(mode="json")
    payload["cases"][0]["entries"][0]["id"] = "x" * 257
    with pytest.raises(ValueError, match="cannot exceed 256 UTF-8 bytes"):
        KnowledgeMaintenanceEvaluationCorpus.model_validate(payload)

    payload = _wide_maintenance_corpus(1).model_dump(mode="json")
    payload["cases"][0]["id"] = "x" * 201
    with pytest.raises(ValueError, match="cannot exceed 200 UTF-8 bytes"):
        KnowledgeMaintenanceEvaluationCorpus.model_validate(payload)

    payload = _wide_maintenance_corpus(1).model_dump(mode="json")
    payload["cases"][0]["claims"][0]["id"] = "not a safe code"
    with pytest.raises(ValueError, match="safe machine-readable code"):
        KnowledgeMaintenanceEvaluationCorpus.model_validate(payload)


def test_maintenance_corpus_rejects_non_executable_recall_queries() -> None:
    payload = json.loads(_CORPUS_PATH.read_text(encoding="utf-8"))
    payload["cases"][0]["recall_query"] = "é" * (RECALL_MAX_QUERY_BYTES // 2)
    KnowledgeMaintenanceEvaluationCorpus.model_validate(payload)

    payload["cases"][0]["recall_query"] += "x"
    with pytest.raises(ValueError, match="recall_query cannot exceed"):
        KnowledgeMaintenanceEvaluationCorpus.model_validate(payload)


def test_maintenance_corpus_preflights_fixture_planner_bounds() -> None:
    payload = _wide_maintenance_corpus(1).model_dump(mode="json")
    payload["cases"][0]["replacement_title"] = "x" * 4_097
    with pytest.raises(ValueError, match="downstream executable planning bounds"):
        KnowledgeMaintenanceEvaluationCorpus.model_validate(payload)

    payload = _wide_maintenance_corpus(1).model_dump(mode="json")
    payload["cases"][0]["claims"][0]["text"] = "x" * 16_385
    with pytest.raises(ValueError, match="downstream executable planning bounds"):
        KnowledgeMaintenanceEvaluationCorpus.model_validate(payload)

    payload = _wide_maintenance_corpus(1).model_dump(mode="json")
    payload["cases"][0]["replacement_text"] = "x" * (64 * 1_024 + 1)
    with pytest.raises(ValueError, match="replacement_text exceeds"):
        KnowledgeMaintenanceEvaluationCorpus.model_validate(payload)

    payload = _wide_maintenance_corpus(1).model_dump(mode="json")
    payload["cases"][0]["claims"] = [
        {
            "id": f"claim_{index:03d}",
            "text": f"Claim {index}.",
            "source_entry_ids": ["wide-lineage:00"],
        }
        for index in range(101)
    ]
    with pytest.raises(ValueError, match="executable evidence-mapping bound"):
        KnowledgeMaintenanceEvaluationCorpus.model_validate(payload)

    payload = _wide_maintenance_corpus(1).model_dump(mode="json")
    payload["cases"][0]["claims"] = [
        {
            "id": f"claim_{index:03d}",
            "text": f"Claim {index}: " + "x" * 3_000,
            "source_entry_ids": ["wide-lineage:00"],
        }
        for index in range(100)
    ]
    with pytest.raises(ValueError, match="downstream executable planning bounds"):
        KnowledgeMaintenanceEvaluationCorpus.model_validate(payload)


def test_maintenance_result_rejects_metrics_that_disagree_with_cases() -> None:
    payload = json.loads(_RESULTS_PATH.read_text(encoding="utf-8"))["results"][0]
    payload["metrics"]["routing_precision"] = 0.25
    payload["metrics"]["lifecycle_correctness"] = 0.0

    with pytest.raises(ValueError, match="exactly match the aggregate"):
        KnowledgeMaintenanceEvaluationResult.model_validate(payload)


def test_maintenance_evidence_is_exact_and_read_after_review() -> None:
    async def run():
        corpus = load_knowledge_maintenance_evaluation_corpus(_CORPUS_PATH)
        normal = await run_knowledge_maintenance_evaluation(
            corpus,
            _DecisionAwareEvidenceStore(),
            backend="memory",
        )
        corrupt = await run_knowledge_maintenance_evaluation(
            corpus,
            _CorruptEvidenceRevisionStore(),
            backend="memory",
        )
        return normal, corrupt

    normal, corrupt = asyncio.run(run())

    assert normal.metrics.evidence_retention == 1.0
    assert all(
        case.evidence_retention == 0.0
        for case in corrupt.cases
        if case.evidence_retention is not None
    )
    assert corrupt.metrics.evidence_retention == 0.0


def test_maintenance_evidence_requires_the_active_successor_after_approval() -> None:
    result = asyncio.run(
        run_knowledge_maintenance_evaluation(
            load_knowledge_maintenance_evaluation_corpus(_CORPUS_PATH),
            _MissingActiveSuccessorEvidenceStore(),
            backend="memory",
        )
    )

    applied = [case for case in result.cases if case.storage_outcome == "applied"]
    assert applied
    assert all(case.evidence_retention == 0.0 for case in applied)
    assert all(
        case.evidence_retention == 1.0
        for case in result.cases
        if case.storage_outcome in {"rejected", "stale"}
    )


def test_maintenance_evidence_must_match_the_requested_target_revision() -> None:
    result = asyncio.run(
        run_knowledge_maintenance_evaluation(
            _corpus_for_scenario(KnowledgeMaintenanceEvaluationScenario.DUPLICATE_MERGE),
            _WrongEvidenceTargetStore(),
            backend="memory",
        )
    )

    assert result.cases[0].evidence_retention == 0.0
    assert result.metrics.evidence_retention == 0.0


def test_maintenance_evidence_requires_the_complete_source_binding() -> None:
    result = asyncio.run(
        run_knowledge_maintenance_evaluation(
            _corpus_for_scenario(KnowledgeMaintenanceEvaluationScenario.DUPLICATE_MERGE),
            _CorruptEvidenceBindingStore(),
            backend="memory",
        )
    )

    assert result.cases[0].evidence_retention == 0.0
    assert result.metrics.evidence_retention == 0.0


def test_maintenance_recall_requires_exact_canonical_material() -> None:
    result = asyncio.run(
        run_knowledge_maintenance_evaluation(
            _corpus_for_scenario(KnowledgeMaintenanceEvaluationScenario.DUPLICATE_MERGE),
            _CorruptRecallMaterialStore(),
            backend="memory",
        )
    )

    assert result.cases[0].storage_outcome == "applied"
    assert result.cases[0].lifecycle_correct is True
    assert result.cases[0].lineage_correct is False


def test_maintenance_lifecycle_rejects_unexpected_namespace_entries() -> None:
    result = asyncio.run(
        run_knowledge_maintenance_evaluation(
            _corpus_for_scenario(KnowledgeMaintenanceEvaluationScenario.DUPLICATE_MERGE),
            _UnexpectedApprovalEntryStore(),
            backend="memory",
        )
    )

    assert result.cases[0].storage_outcome == "applied"
    assert result.cases[0].lifecycle_correct is False


def test_maintenance_lifecycle_accepts_an_unknown_optional_list_total() -> None:
    result = asyncio.run(
        run_knowledge_maintenance_evaluation(
            _corpus_for_scenario(KnowledgeMaintenanceEvaluationScenario.DUPLICATE_MERGE),
            _UnknownListTotalStore(),
            backend="memory",
        )
    )

    assert result.cases[0].storage_outcome == "applied"
    assert result.cases[0].lifecycle_correct is True


def test_maintenance_publication_rejects_a_dishonest_receipt() -> None:
    with pytest.raises(KnowledgeMaintenanceProposalPublicationConflict) as caught:
        asyncio.run(
            run_knowledge_maintenance_evaluation(
                _corpus_for_scenario(KnowledgeMaintenanceEvaluationScenario.DUPLICATE_MERGE),
                _WrongPublicationReceiptStore(),
                backend="memory",
            )
        )

    assert caught.value.reason == "operation_replay_mismatch"


def test_maintenance_publication_requires_the_exact_durable_artifact() -> None:
    with pytest.raises(RuntimeError, match="exact durable artifact"):
        asyncio.run(
            run_knowledge_maintenance_evaluation(
                _corpus_for_scenario(KnowledgeMaintenanceEvaluationScenario.DUPLICATE_MERGE),
                _WrongDurablePublicationStore(),
                backend="memory",
            )
        )


def test_maintenance_approval_requires_exact_receipt_and_successor_snapshots() -> None:
    async def run():
        corpus = _corpus_for_scenario(KnowledgeMaintenanceEvaluationScenario.DUPLICATE_MERGE)
        wrong_receipt = await run_knowledge_maintenance_evaluation(
            corpus,
            _WrongAppliedReceiptStore(),
            backend="memory",
        )
        mutated_source = await run_knowledge_maintenance_evaluation(
            corpus,
            _MutatingAppliedSourceStore(),
            backend="memory",
        )
        return wrong_receipt, mutated_source

    wrong_receipt, mutated_source = asyncio.run(run())
    assert wrong_receipt.cases[0].storage_outcome == "applied"
    assert wrong_receipt.cases[0].lifecycle_correct is False
    assert mutated_source.cases[0].storage_outcome == "applied"
    assert mutated_source.cases[0].lifecycle_correct is False


def test_maintenance_rejection_requires_an_exact_decision_receipt() -> None:
    result = asyncio.run(
        run_knowledge_maintenance_evaluation(
            _corpus_for_scenario(KnowledgeMaintenanceEvaluationScenario.REVIEWER_REJECTION),
            _WrongRejectedReceiptStore(),
            backend="memory",
        )
    )

    assert result.cases[0].storage_outcome == "rejected"
    assert result.cases[0].lifecycle_correct is False


def test_maintenance_lineage_requires_exact_revisions_and_result_queries() -> None:
    async def run():
        wrong_revision = await run_knowledge_maintenance_evaluation(
            _corpus_for_scenario(KnowledgeMaintenanceEvaluationScenario.DUPLICATE_MERGE),
            _WrongLineageRevisionStore(),
            backend="memory",
        )
        wrong_query = await run_knowledge_maintenance_evaluation(
            _corpus_for_scenario(KnowledgeMaintenanceEvaluationScenario.REVIEWER_REJECTION),
            _WrongLineageQueryStore(),
            backend="memory",
        )
        return wrong_revision, wrong_query

    wrong_revision, wrong_query = asyncio.run(run())
    assert wrong_revision.cases[0].lifecycle_correct is True
    assert wrong_revision.cases[0].lineage_correct is False
    assert wrong_query.cases[0].lifecycle_correct is True
    assert wrong_query.cases[0].lineage_correct is False


def test_maintenance_terminal_audit_rechecks_prior_case_namespaces() -> None:
    result = asyncio.run(
        run_knowledge_maintenance_evaluation(
            load_knowledge_maintenance_evaluation_corpus(_CORPUS_PATH),
            _CrossCaseMutationStore(),
            backend="memory",
        )
    )
    duplicate = next(
        case
        for case in result.cases
        if case.scenario is KnowledgeMaintenanceEvaluationScenario.DUPLICATE_MERGE
    )

    assert duplicate.lifecycle_correct is False
    assert duplicate.lineage_correct is False
    assert result.metrics.lifecycle_correctness < 1.0
    assert result.metrics.lineage_correctness < 1.0


def test_maintenance_terminal_audit_rejects_foreign_namespace_writes() -> None:
    result = asyncio.run(
        run_knowledge_maintenance_evaluation(
            load_knowledge_maintenance_evaluation_corpus(_CORPUS_PATH),
            _ForeignTerminalWriteStore(),
            backend="memory",
        )
    )

    assert all(case.lifecycle_correct is False for case in result.cases)
    assert result.metrics.lifecycle_correctness == 0.0


def test_maintenance_terminal_audit_rechecks_prior_case_evidence() -> None:
    result = asyncio.run(
        run_knowledge_maintenance_evaluation(
            load_knowledge_maintenance_evaluation_corpus(_CORPUS_PATH),
            _CrossCaseEvidenceMutationStore(),
            backend="memory",
        )
    )
    duplicate = next(
        case
        for case in result.cases
        if case.scenario is KnowledgeMaintenanceEvaluationScenario.DUPLICATE_MERGE
    )

    assert duplicate.evidence_retention == 0.0
    assert result.metrics.evidence_retention < 1.0


def test_maintenance_terminal_audit_rechecks_prior_decision_receipts() -> None:
    result = asyncio.run(
        run_knowledge_maintenance_evaluation(
            load_knowledge_maintenance_evaluation_corpus(_CORPUS_PATH),
            _CrossCaseReceiptMutationStore(),
            backend="memory",
        )
    )
    duplicate = next(
        case
        for case in result.cases
        if case.scenario is KnowledgeMaintenanceEvaluationScenario.DUPLICATE_MERGE
    )

    assert duplicate.lifecycle_correct is False
    assert result.metrics.lifecycle_correctness < 1.0


def test_maintenance_terminal_audit_rechecks_prior_proposal_publications() -> None:
    result = asyncio.run(
        run_knowledge_maintenance_evaluation(
            load_knowledge_maintenance_evaluation_corpus(_CORPUS_PATH),
            _CrossCasePublicationMutationStore(),
            backend="memory",
        )
    )
    duplicate = next(
        case
        for case in result.cases
        if case.scenario is KnowledgeMaintenanceEvaluationScenario.DUPLICATE_MERGE
    )

    assert duplicate.lifecycle_correct is False
    assert result.metrics.lifecycle_correctness < 1.0


def test_maintenance_terminal_audit_reruns_prior_case_recall() -> None:
    result = asyncio.run(
        run_knowledge_maintenance_evaluation(
            load_knowledge_maintenance_evaluation_corpus(_CORPUS_PATH),
            _CrossCaseRecallLossStore(),
            backend="memory",
        )
    )
    duplicate = next(
        case
        for case in result.cases
        if case.scenario is KnowledgeMaintenanceEvaluationScenario.DUPLICATE_MERGE
    )

    assert duplicate.recalled_entry_ids == ()
    assert duplicate.lineage_correct is False
    assert result.metrics.lineage_correctness < 1.0


def test_maintenance_terminal_audit_rechecks_every_historical_revision() -> None:
    result = asyncio.run(
        run_knowledge_maintenance_evaluation(
            load_knowledge_maintenance_evaluation_corpus(_CORPUS_PATH),
            _CrossCaseHistoricalRevisionLossStore(),
            backend="memory",
        )
    )
    duplicate = next(
        case
        for case in result.cases
        if case.scenario is KnowledgeMaintenanceEvaluationScenario.DUPLICATE_MERGE
    )

    assert duplicate.lifecycle_correct is False
    assert result.metrics.lifecycle_correctness < 1.0


def test_maintenance_terminal_audit_rechecks_every_revision_chunk_set() -> None:
    result = asyncio.run(
        run_knowledge_maintenance_evaluation(
            load_knowledge_maintenance_evaluation_corpus(_CORPUS_PATH),
            _CrossCaseChunkLossStore(),
            backend="memory",
        )
    )
    duplicate = next(
        case
        for case in result.cases
        if case.scenario is KnowledgeMaintenanceEvaluationScenario.DUPLICATE_MERGE
    )

    assert duplicate.lifecycle_correct is False
    assert result.metrics.lifecycle_correctness < 1.0


def test_maintenance_latency_includes_the_amortized_global_terminal_audit() -> None:
    store = _SlowGlobalTerminalAuditStore()
    result = asyncio.run(
        run_knowledge_maintenance_evaluation(
            _corpus_for_scenario(KnowledgeMaintenanceEvaluationScenario.DUPLICATE_MERGE),
            store,
            backend="memory",
        )
    )

    assert result.cases[0].latency_ms >= store.terminal_audit_delay_seconds * 1_000


def test_maintenance_rejection_requires_complete_relation_absence() -> None:
    result = asyncio.run(
        run_knowledge_maintenance_evaluation(
            _corpus_for_scenario(KnowledgeMaintenanceEvaluationScenario.REVIEWER_REJECTION),
            _RejectedRelationMutationStore(),
            backend="memory",
        )
    )

    assert result.cases[0].storage_outcome == "rejected"
    assert result.cases[0].lifecycle_correct is True
    assert result.cases[0].lineage_correct is False
    assert result.metrics.lineage_correctness == 0.0


def test_maintenance_lineage_rejects_truncated_extra_relations() -> None:
    result = asyncio.run(
        run_knowledge_maintenance_evaluation(
            _wide_maintenance_corpus(
                MAX_KNOWLEDGE_MAINTENANCE_SOURCES,
                scenario=KnowledgeMaintenanceEvaluationScenario.AUTHORITATIVE_SUPERSESSION,
            ),
            _TruncatedApprovedLineageStore(),
            backend="memory",
        )
    )

    assert result.cases[0].storage_outcome == "applied"
    assert result.cases[0].lifecycle_correct is True
    assert result.cases[0].lineage_correct is False
    assert result.metrics.lineage_correctness == 0.0


def test_maintenance_rejection_requires_exact_unchanged_sources() -> None:
    result = asyncio.run(
        run_knowledge_maintenance_evaluation(
            load_knowledge_maintenance_evaluation_corpus(_CORPUS_PATH),
            _MutatingRejectStore(),
            backend="memory",
        )
    )
    rejected = next(
        case
        for case in result.cases
        if case.scenario is KnowledgeMaintenanceEvaluationScenario.REVIEWER_REJECTION
    )

    assert rejected.storage_outcome == "rejected"
    assert rejected.lifecycle_correct is False


def test_maintenance_stale_case_requires_the_expected_conflict_reason() -> None:
    with pytest.raises(KnowledgeMaintenanceStale) as caught:
        asyncio.run(
            run_knowledge_maintenance_evaluation(
                load_knowledge_maintenance_evaluation_corpus(_CORPUS_PATH),
                _WrongStaleReasonStore(),
                backend="memory",
            )
        )

    assert caught.value.reason == "relation_endpoint"


def test_historical_lineage_requires_the_exact_archived_revision() -> None:
    result = asyncio.run(
        run_knowledge_maintenance_evaluation(
            load_knowledge_maintenance_evaluation_corpus(_CORPUS_PATH),
            _MissingArchivedHistoryStore(),
            backend="memory",
        )
    )
    historical = next(
        case
        for case in result.cases
        if case.scenario is KnowledgeMaintenanceEvaluationScenario.HISTORICAL_LINEAGE
    )

    assert historical.storage_outcome == "applied"
    assert historical.lineage_correct is False


def test_public_maintenance_evaluation_surface_is_exported() -> None:
    for name in (
        "KnowledgeMaintenanceEvaluationCorpus",
        "KnowledgeMaintenanceEvaluationResult",
        "KnowledgeMaintenanceEvaluationScenario",
        "load_knowledge_maintenance_evaluation_corpus",
        "run_knowledge_maintenance_evaluation",
    ):
        assert getattr(evals, name).__name__ == name


def _without_latency(result: KnowledgeMaintenanceEvaluationResult) -> dict:
    payload = result.model_dump(mode="json")
    payload["metrics"].pop("latency_p50_ms")
    payload["metrics"].pop("latency_p95_ms")
    for case in payload["cases"]:
        case.pop("latency_ms")
    return payload


def _without_latency_case(case) -> dict:
    payload = case.model_dump(mode="json")
    payload.pop("latency_ms")
    return payload


def _corpus_for_scenario(
    scenario: KnowledgeMaintenanceEvaluationScenario,
) -> KnowledgeMaintenanceEvaluationCorpus:
    corpus = load_knowledge_maintenance_evaluation_corpus(_CORPUS_PATH)
    payload = corpus.model_dump(mode="json")
    payload["cases"] = [
        case.model_dump(mode="json") for case in corpus.cases if case.scenario is scenario
    ]
    return KnowledgeMaintenanceEvaluationCorpus.model_validate(payload)


def _wide_maintenance_corpus(
    source_count: int,
    *,
    scenario: KnowledgeMaintenanceEvaluationScenario = (
        KnowledgeMaintenanceEvaluationScenario.HISTORICAL_LINEAGE
    ),
) -> KnowledgeMaintenanceEvaluationCorpus:
    source_ids = [f"wide-lineage:{index:02d}" for index in range(source_count)]
    return KnowledgeMaintenanceEvaluationCorpus.model_validate(
        {
            "schema_version": "cayu.knowledge_maintenance_evaluation_corpus.v1",
            "corpus_revision": f"wide-lineage-{scenario.value}-{source_count}",
            "origin": "external_private",
            "cases": [
                {
                    "id": "wide-lineage",
                    "scenario": scenario.value,
                    "entries": [
                        {
                            "id": entry_id,
                            "text": f"Wide lineage fact {index}. WIDE_LINEAGE_REFERENCE",
                        }
                        for index, entry_id in enumerate(source_ids)
                    ],
                    "source_entry_ids": source_ids,
                    "signal_kind": "exact_reference",
                    "expected_routed_entry_ids": source_ids,
                    "replacement_title": "Wide lineage reference",
                    "replacement_text": " ".join(
                        f"Wide lineage fact {index}. WIDE_LINEAGE_REFERENCE"
                        for index in range(source_count)
                    ),
                    "claims": [
                        {
                            "id": f"wide_claim_{index:02d}",
                            "text": f"Wide lineage fact {index}.",
                            "source_entry_ids": [entry_id],
                        }
                        for index, entry_id in enumerate(source_ids)
                    ],
                    "dispositions": [
                        {
                            "source_entry_id": entry_id,
                            "relation_kind": "supersedes",
                        }
                        for entry_id in source_ids
                    ],
                    "evaluator_verdict": "accepted",
                    "review_decision": "approve",
                    "advance_source_before_review": None,
                    "recall_query": "WIDE_LINEAGE_REFERENCE",
                }
            ],
        }
    )
