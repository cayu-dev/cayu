from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
from tests.core.knowledge_governance_conformance import (
    assert_inaccessible_malformed_activation_receipt_is_hidden,
)

from cayu import (
    MAX_KNOWLEDGE_ACTIVATION_CHUNKS,
    MAX_KNOWLEDGE_ACTIVATION_EVALUATOR_RESULT_BYTES,
    MAX_KNOWLEDGE_ACTIVATION_EVIDENCE_RECORDS,
    MAX_KNOWLEDGE_ACTIVATION_IDENTITY_BYTES,
    InMemoryKnowledgeStore,
    KnowledgeAccessDenied,
    KnowledgeAccessScope,
    KnowledgeActivationConflict,
    KnowledgeActivationDecision,
    KnowledgeActivationDisposition,
    KnowledgeActivationPolicyError,
    KnowledgeActivationReceipt,
    KnowledgeActivationRequest,
    KnowledgeActivationSource,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeEvidence,
    KnowledgeGovernanceConfig,
    KnowledgeGovernanceMode,
    KnowledgePublicationConflict,
    KnowledgeReviewWorkflow,
    KnowledgeStatus,
    SQLiteKnowledgeStore,
    decide_knowledge_activation,
    prepare_knowledge_activation_request,
)

_SCOPE = KnowledgeAccessScope.privileged()
_NOW = datetime(2026, 8, 30, 12, tzinfo=UTC)


class _StaticPolicy:
    def __init__(
        self,
        disposition: KnowledgeActivationDisposition,
        *,
        identity: str = "app.activation-policy",
        version: str = "7",
    ) -> None:
        self.disposition = disposition
        self.identity = identity
        self.version = version
        self.calls = 0

    async def decide_activation(self, request):
        self.calls += 1
        return KnowledgeActivationDecision(
            request_sha256=request.fingerprint,
            disposition=self.disposition,
            policy_identity=self.identity,
            policy_version=self.version,
            code=f"policy_{self.disposition.value}",
            annotations={"rule": "trusted-source"},
        )


def _candidate() -> tuple[KnowledgeEntry, list[KnowledgeChunk]]:
    entry = KnowledgeEntry(
        id="governed-entry",
        text="Production migrations run before service deployment.",
        status=KnowledgeStatus.PENDING,
        created_at=_NOW,
        updated_at=_NOW,
        source_type="test",
        source_id="signal-1",
        source_hash="source-sha256",
    )
    return entry, [
        KnowledgeChunk(
            id="governed-entry:r1:0",
            entry_id=entry.id,
            text=entry.text,
            chunk_index=0,
        )
    ]


def _store(kind: str, tmp_path):
    if kind == "memory":
        return InMemoryKnowledgeStore(access_scope=_SCOPE)
    return SQLiteKnowledgeStore(tmp_path / "governance.sqlite", access_scope=_SCOPE)


def test_activation_authority_binds_entry_and_evidence_timestamps() -> None:
    async def run() -> None:
        store = InMemoryKnowledgeStore(access_scope=_SCOPE)
        entry, chunks = _candidate()
        evidence = [
            KnowledgeEvidence(
                id="governed-entry-origin",
                entry_id=entry.id,
                source_type="test",
                source_id="signal-1",
                source_revision="1",
                created_at=_NOW - timedelta(minutes=5),
            )
        ]
        request = prepare_knowledge_activation_request(
            entry,
            chunks,
            evidence=evidence,
            access_scope=_SCOPE,
            operation_id="timestamp-bound-publication",
            governance_mode=KnowledgeGovernanceMode.REVIEWED,
            source=KnowledgeActivationSource.CURATOR,
        )
        authority = await decide_knowledge_activation(
            request,
            config=KnowledgeGovernanceConfig(),
        )
        assert request.candidate_entry.created_at == entry.created_at
        assert request.candidate_entry.updated_at == entry.updated_at
        assert request.evidence[0].created_at == evidence[0].created_at

        shifted_entry = entry.model_copy(
            update={
                "created_at": entry.created_at + timedelta(days=1),
                "updated_at": entry.updated_at + timedelta(days=1),
            }
        )
        with pytest.raises(ValueError, match="does not bind the publication entry material"):
            await store.publish_entry_revision(
                shifted_entry,
                chunks,
                evidence=evidence,
                operation_id=request.operation_id,
                activation_authority=authority,
            )

        shifted_evidence = [
            evidence[0].model_copy(
                update={"created_at": evidence[0].created_at + timedelta(days=1)}
            )
        ]
        with pytest.raises(ValueError, match="does not bind publication chunks and evidence"):
            await store.publish_entry_revision(
                entry,
                chunks,
                evidence=shifted_evidence,
                operation_id=request.operation_id,
                activation_authority=authority,
            )

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_governed_publication_atomically_records_exact_activation(kind, tmp_path) -> None:
    async def run():
        store = _store(kind, tmp_path)
        entry, chunks = _candidate()
        config = KnowledgeGovernanceConfig(
            mode=KnowledgeGovernanceMode.POLICY_AUTOMATIC,
            policy_identity="app.activation-policy",
            policy_version="7",
        )
        policy = _StaticPolicy(KnowledgeActivationDisposition.ACTIVATE)
        request = prepare_knowledge_activation_request(
            entry,
            chunks,
            access_scope=_SCOPE,
            operation_id="governed-publication",
            governance_mode=config.mode,
            source=KnowledgeActivationSource.CURATOR,
            forbidden_authority_identities=("candidate-generator", "evaluator"),
        )
        authority = await decide_knowledge_activation(request, config=config, policy=policy)
        active = entry.model_copy(update={"status": KnowledgeStatus.ACTIVE})
        first = await store.publish_entry_revision(
            active,
            chunks,
            operation_id="governed-publication",
            activation_authority=authority,
        )
        activation = await store.load_activation_receipt("governed-publication")
        replay = await store.publish_entry_revision(
            active,
            chunks,
            operation_id="governed-publication",
            activation_authority=authority,
        )
        stored = await store.get_entry(entry.id)
        if isinstance(store, SQLiteKnowledgeStore):
            await store.close()
        return first, activation, replay, stored, policy.calls, authority.request.access_scope

    first, activation, replay, stored, calls, policy_scope = asyncio.run(run())

    assert first.replayed is False
    assert replay.replayed is True
    assert activation is not None
    assert activation.entry_id == "governed-entry"
    assert activation.publication_request_sha256 == first.request_sha256
    assert activation.authority.decision.policy_identity == "app.activation-policy"
    assert activation.authority.decision.annotations == {"rule": "trusted-source"}
    assert policy_scope == _SCOPE
    assert stored is not None and stored.status is KnowledgeStatus.ACTIVE
    assert calls == 1


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_governed_publication_replay_rejects_receipt_commit_boundary_mismatch(
    kind,
    tmp_path,
) -> None:
    async def run():
        store = _store(kind, tmp_path)
        entry, chunks = _candidate()
        request = prepare_knowledge_activation_request(
            entry,
            chunks,
            access_scope=_SCOPE,
            operation_id="mismatched-commit-boundary",
            governance_mode=KnowledgeGovernanceMode.REVIEWED,
            source=KnowledgeActivationSource.CURATOR,
        )
        authority = await decide_knowledge_activation(
            request,
            config=KnowledgeGovernanceConfig(),
        )
        await store.publish_entry_revision(
            entry,
            chunks,
            operation_id=request.operation_id,
            activation_authority=authority,
        )
        activation = await store.load_activation_receipt(request.operation_id)
        assert activation is not None
        shifted = activation.model_copy(
            update={"committed_at": activation.committed_at + timedelta(seconds=1)}
        )
        if isinstance(store, InMemoryKnowledgeStore):
            store._activation_receipts[request.operation_id] = shifted
        else:
            store._connection.execute(
                "UPDATE cayu_knowledge_activation_receipts "
                "SET committed_at = ?, receipt_json = ? WHERE operation_id = ?",
                (
                    shifted.committed_at.isoformat(),
                    shifted.model_dump_json(),
                    request.operation_id,
                ),
            )
            store._connection.commit()
        try:
            with pytest.raises(KnowledgePublicationConflict) as conflict:
                await store.publish_entry_revision(
                    entry,
                    chunks,
                    operation_id=request.operation_id,
                    activation_authority=authority,
                )
            return conflict.value.reason
        finally:
            if isinstance(store, SQLiteKnowledgeStore):
                await store.close()

    assert asyncio.run(run()) == "activation_mismatch"


def test_activation_policy_fails_closed_for_missing_malformed_and_self_authority() -> None:
    async def run():
        entry, chunks = _candidate()
        request = prepare_knowledge_activation_request(
            entry,
            chunks,
            access_scope=_SCOPE,
            operation_id="invalid-policy",
            governance_mode=KnowledgeGovernanceMode.AUTONOMOUS,
            source=KnowledgeActivationSource.CURATOR,
            forbidden_authority_identities=("evaluator",),
        )
        config = KnowledgeGovernanceConfig(
            mode=KnowledgeGovernanceMode.AUTONOMOUS,
            policy_identity="app.activation-policy",
            policy_version="7",
        )
        with pytest.raises(KnowledgeActivationPolicyError) as missing:
            await decide_knowledge_activation(request, config=config)
        assert missing.value.code == "activation_policy_missing"

        class MalformedPolicy:
            async def decide_activation(self, request):
                return {"disposition": "activate"}

        with pytest.raises(KnowledgeActivationPolicyError) as malformed:
            await decide_knowledge_activation(
                request,
                config=config,
                policy=MalformedPolicy(),
            )
        assert malformed.value.code == "activation_policy_output_invalid"

        class ValidationBypassPolicy:
            async def decide_activation(self, request):
                valid = KnowledgeActivationDecision(
                    request_sha256=request.fingerprint,
                    disposition=KnowledgeActivationDisposition.ACTIVATE,
                    policy_identity="app.activation-policy",
                    policy_version="7",
                    code="forged_oversized_output",
                )
                return valid.model_copy(update={"annotations": {"diagnostic": "x" * 17_000}})

        with pytest.raises(KnowledgeActivationPolicyError) as bypassed:
            await decide_knowledge_activation(
                request,
                config=config,
                policy=ValidationBypassPolicy(),
            )
        assert bypassed.value.code == "activation_policy_output_invalid"

        class WrongRequestPolicy:
            async def decide_activation(self, request):
                return KnowledgeActivationDecision(
                    request_sha256="0" * 64,
                    disposition=KnowledgeActivationDisposition.ACTIVATE,
                    policy_identity="app.activation-policy",
                    policy_version="7",
                    code="wrong_request",
                )

        with pytest.raises(KnowledgeActivationPolicyError) as wrong_request:
            await decide_knowledge_activation(
                request,
                config=config,
                policy=WrongRequestPolicy(),
            )
        assert wrong_request.value.code == "activation_policy_output_invalid"

        class FailingPolicy:
            async def decide_activation(self, request):
                raise RuntimeError("private policy diagnostic")

        with pytest.raises(KnowledgeActivationPolicyError) as failed:
            await decide_knowledge_activation(
                request,
                config=config,
                policy=FailingPolicy(),
            )
        assert failed.value.code == "activation_policy_failed"

        class SlowPolicy:
            async def decide_activation(self, request):
                await asyncio.sleep(1)

        timeout_config = config.model_copy(update={"policy_timeout_seconds": 0.001})
        with pytest.raises(KnowledgeActivationPolicyError) as timed_out:
            await decide_knowledge_activation(
                request,
                config=timeout_config,
                policy=SlowPolicy(),
            )
        assert timed_out.value.code == "activation_policy_timed_out"

        timeout_cancelled = asyncio.Event()
        timeout_release = asyncio.Event()
        timeout_finished = asyncio.Event()

        class TimeoutSuppressingPolicy:
            async def decide_activation(self, request):
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    timeout_cancelled.set()
                    await timeout_release.wait()
                    return KnowledgeActivationDecision(
                        request_sha256=request.fingerprint,
                        disposition=KnowledgeActivationDisposition.ACTIVATE,
                        policy_identity="app.activation-policy",
                        policy_version="7",
                        code="late_activate",
                    )
                finally:
                    timeout_finished.set()

        with pytest.raises(KnowledgeActivationPolicyError) as suppressed_timeout:
            await decide_knowledge_activation(
                request,
                config=timeout_config,
                policy=TimeoutSuppressingPolicy(),
            )
        assert suppressed_timeout.value.code == "activation_policy_timed_out"
        await asyncio.wait_for(timeout_cancelled.wait(), timeout=1)
        timeout_release.set()
        await asyncio.wait_for(timeout_finished.wait(), timeout=1)

        wrong = _StaticPolicy(
            KnowledgeActivationDisposition.ACTIVATE,
            identity="evaluator",
        )
        self_authorizing_config = config.model_copy(update={"policy_identity": "evaluator"})
        with pytest.raises(KnowledgeActivationPolicyError) as self_authorized:
            await decide_knowledge_activation(
                request,
                config=self_authorizing_config,
                policy=wrong,
            )
        assert self_authorized.value.code == "activation_policy_output_invalid"

    asyncio.run(run())


def test_activation_policy_cannot_suppress_caller_cancellation() -> None:
    async def run() -> None:
        entry, chunks = _candidate()
        request = prepare_knowledge_activation_request(
            entry,
            chunks,
            access_scope=_SCOPE,
            operation_id="cancelled-policy",
            governance_mode=KnowledgeGovernanceMode.AUTONOMOUS,
            source=KnowledgeActivationSource.CURATOR,
        )
        config = KnowledgeGovernanceConfig(
            mode=KnowledgeGovernanceMode.AUTONOMOUS,
            policy_identity="app.activation-policy",
            policy_version="7",
        )
        entered = asyncio.Event()
        suppressed = asyncio.Event()

        class CancellationSuppressingPolicy:
            async def decide_activation(self, request):
                entered.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    suppressed.set()
                    return KnowledgeActivationDecision(
                        request_sha256=request.fingerprint,
                        disposition=KnowledgeActivationDisposition.ACTIVATE,
                        policy_identity="app.activation-policy",
                        policy_version="7",
                        code="cancelled_activate",
                    )

        invocation = asyncio.create_task(
            decide_knowledge_activation(
                request,
                config=config,
                policy=CancellationSuppressingPolicy(),
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=1)
        invocation.cancel("test caller cancellation")
        with pytest.raises(asyncio.CancelledError, match="test caller cancellation"):
            await invocation
        await asyncio.wait_for(suppressed.wait(), timeout=1)

    asyncio.run(run())


def test_reviewed_governance_never_invokes_an_automatic_policy() -> None:
    async def run():
        entry, chunks = _candidate()
        request = prepare_knowledge_activation_request(
            entry,
            chunks,
            access_scope=_SCOPE,
            operation_id="reviewed-routing",
            governance_mode=KnowledgeGovernanceMode.REVIEWED,
            source=KnowledgeActivationSource.MODEL_TOOL,
        )
        policy = _StaticPolicy(KnowledgeActivationDisposition.ACTIVATE)
        with pytest.raises(KnowledgeActivationPolicyError) as configured:
            await decide_knowledge_activation(
                request,
                config=KnowledgeGovernanceConfig(),
                policy=policy,
            )
        assert configured.value.code == "reviewed_mode_policy_configured"
        assert policy.calls == 0

    asyncio.run(run())


def test_sqlite_activation_receipt_failure_rolls_back_the_whole_publication(
    tmp_path,
) -> None:
    class FailingActivationReceiptStore(SQLiteKnowledgeStore):
        def _insert_activation_receipt_unlocked(self, receipt, *, access_entry):
            raise RuntimeError("injected activation receipt failure")

    async def run():
        store = FailingActivationReceiptStore(
            tmp_path / "governance-failure.sqlite",
            access_scope=_SCOPE,
        )
        entry, chunks = _candidate()
        request = prepare_knowledge_activation_request(
            entry,
            chunks,
            access_scope=_SCOPE,
            operation_id="failed-governed-publication",
            governance_mode=KnowledgeGovernanceMode.REVIEWED,
            source=KnowledgeActivationSource.CURATOR,
        )
        authority = await decide_knowledge_activation(
            request,
            config=KnowledgeGovernanceConfig(),
        )
        try:
            with pytest.raises(RuntimeError, match="injected activation receipt failure"):
                await store.publish_entry_revision(
                    entry,
                    chunks,
                    operation_id="failed-governed-publication",
                    activation_authority=authority,
                )
            stored = await store.get_entry(entry.id)
            publication = await store.load_entry_publication_receipt("failed-governed-publication")
            activation = await store.load_activation_receipt("failed-governed-publication")
            return stored, publication, activation
        finally:
            await store.close()

    stored, publication, activation = asyncio.run(run())

    assert stored is None
    assert publication is None
    assert activation is None


def test_sqlite_review_activation_receipt_failure_rolls_back_the_successor(
    tmp_path,
) -> None:
    class FailingActivationReceiptStore(SQLiteKnowledgeStore):
        def _insert_activation_receipt_unlocked(self, receipt, *, access_entry):
            raise RuntimeError("injected review activation receipt failure")

    async def run():
        store = FailingActivationReceiptStore(
            tmp_path / "review-governance-failure.sqlite",
            access_scope=_SCOPE,
        )
        entry, chunks = _candidate()
        await store.create_entry(entry, chunks)
        workflow = KnowledgeReviewWorkflow(store)
        try:
            with pytest.raises(
                RuntimeError,
                match="injected review activation receipt failure",
            ):
                await workflow.approve(
                    entry.id,
                    operation_id="failed-review-approval",
                    reviewer_identity="human.release-reviewer",
                    reviewer_version="2026-08",
                )
            current = await store.get_entry(entry.id)
            publication = await store.load_entry_publication_receipt("failed-review-approval")
            activation = await store.load_activation_receipt("failed-review-approval")
            return current, publication, activation
        finally:
            await store.close()

    current, publication, activation = asyncio.run(run())

    assert current is not None
    assert current.status is KnowledgeStatus.PENDING
    assert current.revision == 1
    assert publication is None
    assert activation is None


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_review_approval_is_attributed_atomic_and_exactly_replayable(kind, tmp_path) -> None:
    async def run():
        store = _store(kind, tmp_path)
        entry, chunks = _candidate()
        await store.create_entry(entry, chunks)
        workflow = KnowledgeReviewWorkflow(store)
        first = await workflow.approve(
            entry.id,
            operation_id="review-governed-entry",
            reviewer_identity="human.release-reviewer",
            reviewer_version="2026-08",
            annotations={"ticket": "release-42"},
        )
        replay = await workflow.approve(
            entry.id,
            operation_id="review-governed-entry",
            reviewer_identity="human.release-reviewer",
            reviewer_version="2026-08",
            annotations={"ticket": "release-42"},
        )
        publication = await store.load_entry_publication_receipt("review-governed-entry")
        if isinstance(store, SQLiteKnowledgeStore):
            await store.close()
        return first, replay, publication

    first, replay, publication = asyncio.run(run())

    assert first.entry.status is KnowledgeStatus.ACTIVE
    assert first.entry.revision == 2
    assert first.receipt.replayed is False
    assert first.receipt.authority.request.candidate_entry.revision == 1
    assert first.receipt.authority.decision.policy_identity == "human.release-reviewer"
    assert publication is not None
    assert publication.request_sha256 == first.receipt.publication_request_sha256
    assert publication.committed_at == first.receipt.committed_at
    assert replay.entry == first.entry
    assert replay.receipt.replayed is True


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_review_activation_receipt_uses_the_active_revision_scope(kind, tmp_path) -> None:
    async def run():
        store = (
            InMemoryKnowledgeStore()
            if kind == "memory"
            else SQLiteKnowledgeStore(tmp_path / "review-audit-scope.sqlite")
        )
        entry, chunks = _candidate()
        await store.create_entry(entry, chunks, access_scope=_SCOPE)
        workflow = KnowledgeReviewWorkflow(store, access_scope=_SCOPE)
        approval = await workflow.approve(
            entry.id,
            operation_id="review-audit-scope",
            reviewer_identity="human.release-reviewer",
            reviewer_version="2026-08",
        )
        active_audit = await store.load_activation_receipt(
            "review-audit-scope",
            access_scope=KnowledgeAccessScope.for_namespace("default"),
        )
        pending_audit = await store.load_activation_receipt(
            "review-audit-scope",
            access_scope=KnowledgeAccessScope.for_namespace(
                "default",
                allowed_statuses=[KnowledgeStatus.PENDING],
            ),
        )
        if isinstance(store, SQLiteKnowledgeStore):
            await store.close()
        return approval, active_audit, pending_audit

    approval, active_audit, pending_audit = asyncio.run(run())

    assert active_audit is not None
    assert active_audit.entry_revision == approval.entry.revision
    assert pending_audit is None


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_review_approval_requires_scope_for_the_active_successor(kind, tmp_path) -> None:
    async def run():
        pending_only = KnowledgeAccessScope.for_namespace(
            "default",
            allowed_statuses=[KnowledgeStatus.PENDING],
        )
        store = (
            InMemoryKnowledgeStore(access_scope=pending_only)
            if kind == "memory"
            else SQLiteKnowledgeStore(
                tmp_path / "pending-only-review.sqlite",
                access_scope=pending_only,
            )
        )
        entry, chunks = _candidate()
        await store.create_entry(entry, chunks)
        workflow = KnowledgeReviewWorkflow(store)
        try:
            with pytest.raises(KnowledgeAccessDenied):
                await workflow.approve(
                    entry.id,
                    operation_id="pending-only-review",
                    reviewer_identity="human.release-reviewer",
                    reviewer_version="2026-08",
                )
            return await store.get_entry(entry.id)
        finally:
            if isinstance(store, SQLiteKnowledgeStore):
                await store.close()

    current = asyncio.run(run())

    assert current is not None
    assert current.status is KnowledgeStatus.PENDING
    assert current.revision == 1


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "reviewer_identity",
    ["knowledge_curator", "curator.generator", "curator.evaluator"],
)
def test_review_approval_rejects_candidate_and_evaluator_self_authority(
    kind,
    reviewer_identity,
    tmp_path,
) -> None:
    async def run():
        store = _store(kind, tmp_path)
        entry, chunks = _candidate()
        entry = entry.model_copy(
            update={
                "created_by": "knowledge_curator",
                "metadata": {
                    "cayu_curator": {
                        "candidate_generator_identity": "curator.generator",
                        "evaluator_identity": "curator.evaluator",
                        "policy_identity": None,
                    }
                },
            }
        )
        await store.create_entry(entry, chunks)
        workflow = KnowledgeReviewWorkflow(store)
        try:
            with pytest.raises(ValueError, match="cannot authorize activation"):
                await workflow.approve(
                    entry.id,
                    operation_id=f"self-review-{reviewer_identity}",
                    reviewer_identity=reviewer_identity,
                    reviewer_version="1",
                )
            current = await store.get_entry(entry.id)
            publication = await store.load_entry_publication_receipt(
                f"self-review-{reviewer_identity}"
            )
            activation = await store.load_activation_receipt(f"self-review-{reviewer_identity}")
            return current, publication, activation
        finally:
            if isinstance(store, SQLiteKnowledgeStore):
                await store.close()

    current, publication, activation = asyncio.run(run())

    assert current is not None
    assert current.status is KnowledgeStatus.PENDING
    assert publication is None
    assert activation is None


def test_activation_decision_rejects_oversized_annotations() -> None:
    with pytest.raises(ValueError, match="annotations"):
        KnowledgeActivationDecision(
            request_sha256="a" * 64,
            disposition=KnowledgeActivationDisposition.ACTIVATE,
            policy_identity="app.activation-policy",
            policy_version="7",
            code="oversized_annotations",
            annotations={"diagnostic": "x" * 17_000},
        )


def test_activation_request_rejects_oversized_record_counts_before_copying() -> None:
    entry, chunks = _candidate()
    request = prepare_knowledge_activation_request(
        entry,
        chunks,
        access_scope=_SCOPE,
        operation_id="bounded-activation-request",
        governance_mode=KnowledgeGovernanceMode.REVIEWED,
        source=KnowledgeActivationSource.CURATOR,
    )
    material = request.model_dump(mode="python")

    with pytest.raises(ValueError, match="chunks.*cannot contain more"):
        KnowledgeActivationRequest.model_validate(
            {
                **material,
                "chunks": [material["chunks"][0]] * (MAX_KNOWLEDGE_ACTIVATION_CHUNKS + 1),
            }
        )
    with pytest.raises(ValueError, match="evidence.*cannot contain more"):
        KnowledgeActivationRequest.model_validate(
            {
                **material,
                "evidence": [None] * (MAX_KNOWLEDGE_ACTIVATION_EVIDENCE_RECORDS + 1),
            }
        )


def test_activation_request_binds_one_bounded_evaluator_result() -> None:
    entry, chunks = _candidate()
    common = {
        "entry": entry,
        "chunks": chunks,
        "access_scope": _SCOPE,
        "operation_id": "evaluator-bound-activation-request",
        "governance_mode": KnowledgeGovernanceMode.POLICY_AUTOMATIC,
        "source": KnowledgeActivationSource.CURATOR,
        "evaluator_identity": "independent.evaluator",
    }

    with pytest.raises(ValueError, match="does not match its decision fingerprint"):
        prepare_knowledge_activation_request(
            **common,
            evaluator_result={"verdict": "accepted"},
            evaluator_decision_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="evaluator_result.*byte limit"):
        prepare_knowledge_activation_request(
            **common,
            evaluator_result={"diagnostic": "x" * MAX_KNOWLEDGE_ACTIVATION_EVALUATOR_RESULT_BYTES},
            evaluator_decision_sha256="a" * 64,
        )


def test_governance_schema_version_rejects_boolean_alias() -> None:
    with pytest.raises(ValueError, match="schema_version.*integer 1"):
        KnowledgeGovernanceConfig(schema_version=True)


@pytest.mark.parametrize(
    ("revision_alias", "boolean_alias"),
    [(True, 1), ("1", "true")],
)
def test_activation_contracts_reject_coerced_scalar_aliases(
    revision_alias: object,
    boolean_alias: object,
) -> None:
    entry, chunks = _candidate()
    request = prepare_knowledge_activation_request(
        entry,
        chunks,
        access_scope=_SCOPE,
        operation_id="strict-activation-scalars",
        governance_mode=KnowledgeGovernanceMode.REVIEWED,
        source=KnowledgeActivationSource.CURATOR,
    )
    authority = asyncio.run(
        decide_knowledge_activation(request, config=KnowledgeGovernanceConfig())
    )
    receipt = KnowledgeActivationReceipt(
        operation_id=request.operation_id,
        entry_id=entry.id,
        entry_revision=1,
        expected_revision=None,
        publication_request_sha256="a" * 64,
        authority=authority,
        committed_at=_NOW,
    )

    request_payload = request.model_dump(mode="json")
    request_payload["target_revision"] = revision_alias
    with pytest.raises(ValueError, match="target_revision.*integer"):
        KnowledgeActivationRequest.model_validate_json(json.dumps(request_payload))

    receipt_payload = receipt.model_dump(mode="json")
    receipt_payload["entry_revision"] = revision_alias
    with pytest.raises(ValueError, match="entry_revision.*integer"):
        KnowledgeActivationReceipt.model_validate_json(json.dumps(receipt_payload))

    receipt_payload = receipt.model_dump(mode="json")
    receipt_payload["replayed"] = boolean_alias
    with pytest.raises(ValueError, match="replayed.*boolean"):
        KnowledgeActivationReceipt.model_validate_json(json.dumps(receipt_payload))

    with pytest.raises(ValueError, match="policy_timeout_seconds.*number"):
        KnowledgeGovernanceConfig(policy_timeout_seconds=revision_alias)


def test_knowledge_author_identity_obeys_activation_authority_bound() -> None:
    boundary_identity = "é" * (MAX_KNOWLEDGE_ACTIVATION_IDENTITY_BYTES // 2)
    entry = KnowledgeEntry(
        id="bounded-author", text="Bounded author.", created_by=boundary_identity
    )

    assert entry.created_by == boundary_identity
    with pytest.raises(ValueError, match="created_by.*at most"):
        KnowledgeEntry(
            id="oversized-author",
            text="Oversized author.",
            created_by=f"{boundary_identity}a",
        )


def test_conflicting_activation_decision_cannot_reuse_publication_operation() -> None:
    async def run():
        store = InMemoryKnowledgeStore(access_scope=_SCOPE)
        entry, chunks = _candidate()
        config = KnowledgeGovernanceConfig(
            mode=KnowledgeGovernanceMode.POLICY_AUTOMATIC,
            policy_identity="app.activation-policy",
            policy_version="7",
        )
        request = prepare_knowledge_activation_request(
            entry,
            chunks,
            access_scope=_SCOPE,
            operation_id="conflicting-decision",
            governance_mode=config.mode,
            source=KnowledgeActivationSource.CURATOR,
        )
        activate = await decide_knowledge_activation(
            request,
            config=config,
            policy=_StaticPolicy(KnowledgeActivationDisposition.ACTIVATE),
        )
        await store.publish_entry_revision(
            entry.model_copy(update={"status": KnowledgeStatus.ACTIVE}),
            chunks,
            operation_id="conflicting-decision",
            activation_authority=activate,
        )
        route = await decide_knowledge_activation(
            request,
            config=config,
            policy=_StaticPolicy(KnowledgeActivationDisposition.ROUTE_TO_REVIEW),
        )
        with pytest.raises(KnowledgePublicationConflict):
            await store.publish_entry_revision(
                entry,
                chunks,
                operation_id="conflicting-decision",
                activation_authority=route,
            )

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_review_operation_cannot_be_reused_and_audit_survives_pruning(kind, tmp_path) -> None:
    async def run():
        store = _store(kind, tmp_path)
        entry, chunks = _candidate()
        expired = entry.model_copy(update={"expires_at": _NOW})
        await store.create_entry(expired, chunks)
        workflow = KnowledgeReviewWorkflow(store)
        approval = await workflow.approve(
            expired.id,
            operation_id="review-operation",
            reviewer_identity="human.release-reviewer",
            reviewer_version="2026-08",
        )
        other = KnowledgeEntry(
            id="other-entry",
            text="A different publication cannot occupy a review operation.",
            status=KnowledgeStatus.PENDING,
            created_at=_NOW,
            updated_at=_NOW,
        )
        other_chunks = [
            KnowledgeChunk(
                id="other-entry:r1:0",
                entry_id=other.id,
                text=other.text,
                chunk_index=0,
            )
        ]
        with pytest.raises(KnowledgePublicationConflict) as occupied:
            await store.publish_entry_revision(
                other,
                other_chunks,
                operation_id="review-operation",
            )
        assert occupied.value.reason == "operation_occupied"
        pruned = await store.prune_expired(now=_NOW)
        audit = await store.load_activation_receipt("review-operation")
        if isinstance(store, SQLiteKnowledgeStore):
            await store.close()
        return approval, pruned, audit

    approval, pruned, audit = asyncio.run(run())

    assert approval.entry.status is KnowledgeStatus.ACTIVE
    assert pruned == 1
    assert audit is not None
    assert audit.authority.decision.policy_identity == "human.release-reviewer"


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_review_approval_cannot_reuse_a_publication_operation(kind, tmp_path) -> None:
    async def run():
        store = _store(kind, tmp_path)
        occupied = KnowledgeEntry(id="occupied-entry", text="Owns the publication operation.")
        await store.publish_entry_revision(
            occupied,
            [
                KnowledgeChunk(
                    id="occupied-entry:r1:0",
                    entry_id=occupied.id,
                    text=occupied.text,
                    chunk_index=0,
                )
            ],
            operation_id="shared-operation",
        )
        pending, chunks = _candidate()
        await store.create_entry(pending, chunks)
        workflow = KnowledgeReviewWorkflow(store)
        with pytest.raises(KnowledgeActivationConflict) as conflict:
            await workflow.approve(
                pending.id,
                operation_id="shared-operation",
                reviewer_identity="human.release-reviewer",
                reviewer_version="2026-08",
            )
        current = await store.get_entry(pending.id)
        activation = await store.load_activation_receipt("shared-operation")
        if isinstance(store, SQLiteKnowledgeStore):
            await store.close()
        return conflict.value.reason, current, activation

    reason, current, activation = asyncio.run(run())

    assert reason == "operation_occupied"
    assert current is not None
    assert current.status is KnowledgeStatus.PENDING
    assert current.revision == 1
    assert activation is None


def test_sqlite_activation_receipt_rejects_inconsistent_indexed_columns(tmp_path) -> None:
    async def run():
        store = SQLiteKnowledgeStore(
            tmp_path / "inconsistent-activation-receipt.sqlite",
            access_scope=_SCOPE,
        )
        entry, chunks = _candidate()
        request = prepare_knowledge_activation_request(
            entry,
            chunks,
            access_scope=_SCOPE,
            operation_id="inconsistent-activation-receipt",
            governance_mode=KnowledgeGovernanceMode.REVIEWED,
            source=KnowledgeActivationSource.CURATOR,
        )
        authority = await decide_knowledge_activation(
            request,
            config=KnowledgeGovernanceConfig(),
        )
        await store.publish_entry_revision(
            entry,
            chunks,
            operation_id=request.operation_id,
            activation_authority=authority,
        )
        store._connection.execute(
            "UPDATE cayu_knowledge_activation_receipts "
            "SET publication_request_sha256 = ? WHERE operation_id = ?",
            ("b" * 64, request.operation_id),
        )
        store._connection.commit()
        try:
            with pytest.raises(KnowledgeActivationConflict) as conflict:
                await store.load_activation_receipt(request.operation_id)
            return conflict.value.reason
        finally:
            await store.close()

    assert asyncio.run(run()) == "malformed_receipt"


def test_sqlite_activation_receipt_authorization_uses_one_read_snapshot(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteKnowledgeStore(
            tmp_path / "activation-receipt-read-snapshot.sqlite",
            access_scope=_SCOPE,
        )
        entry, chunks = _candidate()
        request = prepare_knowledge_activation_request(
            entry,
            chunks,
            access_scope=_SCOPE,
            operation_id="activation-receipt-read-snapshot",
            governance_mode=KnowledgeGovernanceMode.REVIEWED,
            source=KnowledgeActivationSource.CURATOR,
        )
        authority = await decide_knowledge_activation(
            request,
            config=KnowledgeGovernanceConfig(),
        )
        await store.publish_entry_revision(
            entry,
            chunks,
            operation_id=request.operation_id,
            activation_authority=authority,
        )
        authorization_query_transaction_states: list[bool] = []

        def record_authorization_query(statement: str) -> None:
            normalized = " ".join(statement.lower().split())
            if (
                "from cayu_knowledge_current_entries" in normalized
                or "from cayu_knowledge_activation_retirements" in normalized
            ):
                authorization_query_transaction_states.append(store._connection.in_transaction)

        store._connection.set_trace_callback(record_authorization_query)
        try:
            receipt = await store.load_activation_receipt(request.operation_id)
        finally:
            store._connection.set_trace_callback(None)
            await store.close()
        assert receipt is not None
        assert len(authorization_query_transaction_states) >= 2
        assert all(authorization_query_transaction_states)

    asyncio.run(run())


def test_sqlite_inaccessible_malformed_activation_receipt_is_hidden(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteKnowledgeStore(tmp_path / "inaccessible-malformed-activation.sqlite")

        async def corrupt_receipt(operation_id: str) -> None:
            store._connection.execute(
                "UPDATE cayu_knowledge_activation_receipts "
                "SET receipt_json = '{}' WHERE operation_id = ?",
                (operation_id,),
            )
            store._connection.commit()

        try:
            await assert_inaccessible_malformed_activation_receipt_is_hidden(
                store,
                corrupt_receipt=corrupt_receipt,
            )
        finally:
            await store.close()

    asyncio.run(run())
