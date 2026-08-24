from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from tests.evals.eval_store_conformance import (
    _scenario,
    assert_eval_store_conformance,
    assert_scenario_progress_conformance,
)
from tests.evals.test_corpus_execution import (
    _corpus,
    _model_judge_corpus,
    _model_judge_target,
    _provider,
    _target,
)

from cayu.evals.corpus import EvalCaseSpec, EvalCorpusDocument, EvalSuiteSpec
from cayu.evals.execution import run_corpus_suite
from cayu.evals.store import (
    EvalCaseCatalogQuery,
    EvalCatalogQuery,
    EvalRunAdmissionConflict,
    EvalRunClaimLost,
    EvalRunFailureCode,
    EvalRunQuery,
    EvalRunRequest,
    EvalRunStateConflict,
    EvalRunStatus,
    EvalStoreResultTooLarge,
    EvalSuiteCatalogQuery,
    InMemoryEvalStore,
)
from cayu.vaults.redaction import SecretRedactor

_NO_SECRETS = SecretRedactor()


async def _save_corpus(store, corpus):
    return await store.save_corpus(
        corpus,
        redact_json=_NO_SECRETS.redact_json,
    )


async def _admit_run(store, request):
    return await store.admit_run(
        request,
        redact_json=_NO_SECRETS.redact_json,
    )


async def _publish_result(store, claim, result):
    return await store.publish_result(
        claim,
        result,
        redact_json=_NO_SECRETS.redact_json,
    )


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


def test_memory_eval_store_shared_conformance() -> None:
    async def exercise() -> None:
        corpus = _corpus(trials=1)
        result = await run_corpus_suite(
            _target(_provider(trials=1)),
            corpus,
            corpus.suites[0].id,
            max_concurrency=1,
        )
        await assert_eval_store_conformance(
            InMemoryEvalStore(),
            corpus=corpus,
            result=result,
        )
        await assert_scenario_progress_conformance(
            InMemoryEvalStore(),
            corpus=corpus,
        )

    asyncio.run(exercise())


def test_memory_eval_store_is_explicitly_process_local() -> None:
    async def exercise() -> None:
        corpus = _corpus()
        first = InMemoryEvalStore()
        await _save_corpus(first, corpus)
        scenario = _scenario(corpus, text="Process-local scenario.")
        await first.save_scenario(scenario, redact_json=_NO_SECRETS.redact_json)
        await _admit_run(first, _request(corpus))

        restarted = InMemoryEvalStore()
        assert restarted.durable is False
        assert await restarted.load_corpus(corpus.revision) is None
        assert await restarted.load_scenario(scenario.revision) is None
        assert await restarted.load_run("run-1") is None

    asyncio.run(exercise())


def _request(corpus, *, run_id: str = "run-1", key: str = "request-1") -> EvalRunRequest:
    suite = corpus.suites[0]
    return EvalRunRequest(
        run_id=run_id,
        idempotency_key="sha256:" + key.encode().hex().ljust(64, "0")[:64],
        corpus_revision=corpus.revision,
        target_key=corpus.target_key,
        suite_id=suite.id,
        suite_revision=suite.revision,
        max_concurrency=2,
    )


def test_memory_store_catalog_is_immutable_bounded_and_keyset_paginated() -> None:
    async def exercise() -> None:
        clock = _Clock()
        store = InMemoryEvalStore(clock=clock)
        first = _corpus(input_text="First request")
        second = _corpus(input_text="Second request")

        saved_first = await _save_corpus(store, first)
        assert await _save_corpus(store, first) == saved_first
        clock.advance(1)
        saved_second = await _save_corpus(store, second)

        page = await store.list_corpora(EvalCatalogQuery(limit=1))
        assert [item.revision for item in page.items] == [saved_second.revision]
        assert page.has_more is True
        following = await store.list_corpora(EvalCatalogQuery(limit=1, cursor=page.next_cursor))
        assert [item.revision for item in following.items] == [saved_first.revision]
        assert following.has_more is False
        with pytest.raises(ValueError, match="cursor does not match this query"):
            await store.list_corpora(
                EvalCatalogQuery(
                    target_key="another-agent",
                    limit=1,
                    cursor=page.next_cursor,
                )
            )

        restored = await store.load_corpus(first.revision)
        assert restored == first
        with pytest.raises(EvalStoreResultTooLarge):
            await store.load_corpus(first.revision, max_bytes=1)

        suites = await store.list_suites(EvalSuiteCatalogQuery(corpus_revision=first.revision))
        assert [(suite.id, suite.revision) for suite in suites.items] == [
            (first.suites[0].id, first.suites[0].revision)
        ]
        cases = await store.list_cases(
            EvalCaseCatalogQuery(
                corpus_revision=first.revision,
                suite_id=first.suites[0].id,
            )
        )
        assert [(case.id, case.revision) for case in cases.items] == [
            (first.cases[0].id, first.cases[0].revision)
        ]

    asyncio.run(exercise())


def test_suite_catalog_is_byte_bounded_and_uses_portable_keyset_order() -> None:
    async def exercise() -> None:
        base = _corpus()
        second_suite = EvalSuiteSpec.create(
            id="a-second-suite",
            name="Second suite",
            description="x" * 2_000,
        )
        original = base.cases[0]
        second_case = EvalCaseSpec.create(
            id="a-second-case",
            suite_id=second_suite.id,
            name="Second case",
            source=original.source,
            input=original.input,
            assertions=original.assertions,
        )
        corpus = EvalCorpusDocument.create(
            target_key=base.target_key,
            evidence_policy=base.evidence_policy,
            pricing_profile=base.pricing_profile,
            suites=(base.suites[0], second_suite),
            cases=(base.cases[0], second_case),
        )
        store = InMemoryEvalStore()
        await _save_corpus(store, corpus)

        first = await store.list_suites(
            EvalSuiteCatalogQuery(corpus_revision=corpus.revision, limit=1)
        )
        assert [item.id for item in first.items] == ["a-second-suite"]
        assert first.has_more is True
        second = await store.list_suites(
            EvalSuiteCatalogQuery(
                corpus_revision=corpus.revision,
                cursor=first.next_cursor,
                limit=1,
            )
        )
        assert [item.id for item in second.items] == ["refund-regressions"]
        assert second.has_more is False

        with pytest.raises(EvalStoreResultTooLarge):
            await store.list_suites(
                EvalSuiteCatalogQuery(
                    corpus_revision=corpus.revision,
                    limit=1,
                    max_result_bytes=1_024,
                )
            )

    asyncio.run(exercise())


def test_memory_store_admission_is_atomic_idempotent_and_contract_bound() -> None:
    async def exercise() -> None:
        corpus = _corpus()
        store = InMemoryEvalStore()
        await _save_corpus(store, corpus)
        request = _request(corpus)

        admitted, replay = await asyncio.gather(
            _admit_run(store, request),
            _admit_run(store, request.model_copy(update={"run_id": "retry-id"})),
        )
        assert admitted == replay
        assert admitted.id == "run-1"
        assert admitted.status is EvalRunStatus.QUEUED

        with pytest.raises(EvalRunAdmissionConflict, match="idempotency key"):
            await _admit_run(
                store,
                request.model_copy(
                    update={"run_id": "run-2", "max_concurrency": 1},
                ),
            )
        with pytest.raises(EvalRunAdmissionConflict, match="run id"):
            await _admit_run(store, _request(corpus, run_id="run-1", key="request-2"))

        await _admit_run(store, _request(corpus, run_id="run-3", key="request-3"))
        page = await store.list_runs(EvalRunQuery(limit=1))
        assert page.has_more is True
        with pytest.raises(ValueError, match="cursor does not match this query"):
            await store.list_runs(
                EvalRunQuery(
                    status=EvalRunStatus.RUNNING,
                    limit=1,
                    cursor=page.next_cursor,
                )
            )

    asyncio.run(exercise())


def test_memory_store_claims_are_fenced_reclaimable_and_cancellable() -> None:
    async def exercise() -> None:
        clock = _Clock()
        claim_ids = iter(("claim-1", "claim-2", "claim-3"))
        store = InMemoryEvalStore(clock=clock, claim_id_factory=lambda: next(claim_ids))
        corpus = _corpus()
        await _save_corpus(store, corpus)
        await _admit_run(store, _request(corpus))

        claimed = await store.claim_run(lease_seconds=10)
        assert claimed is not None
        first_claim = claimed.claim
        assert first_claim.epoch == 1
        assert claimed.run.attempt_count == 1

        clock.advance(11)
        reclaimed = await store.claim_run(lease_seconds=10)
        assert reclaimed is not None
        second_claim = reclaimed.claim
        assert second_claim.epoch == 2
        assert reclaimed.run.attempt_count == 2
        with pytest.raises(EvalRunClaimLost):
            await store.heartbeat_run(first_claim)

        cancelling = await store.request_cancel(reclaimed.run.id)
        assert cancelling.status is EvalRunStatus.CANCELLING
        cancelled = await store.finish_cancel(second_claim)
        assert cancelled.status is EvalRunStatus.CANCELLED
        assert cancelled.attempt_count == 2
        assert cancelled.finished_at is not None
        assert await store.finish_cancel(second_claim) == cancelled

        await _admit_run(store, _request(corpus, run_id="run-2", key="request-2"))
        queued_cancelled = await store.request_cancel("run-2")
        assert queued_cancelled.status is EvalRunStatus.CANCELLED
        assert queued_cancelled.attempt_count == 0
        assert await store.claim_run() is None

        await _admit_run(store, _request(corpus, run_id="run-3", key="request-3"))
        expiring = await store.claim_run(lease_seconds=10)
        assert expiring is not None
        clock.advance(11)
        expired_cancelled = await store.request_cancel("run-3")
        assert expired_cancelled.status is EvalRunStatus.CANCELLED
        assert expired_cancelled.attempt_count == 1
        assert expired_cancelled.ownership is None
        with pytest.raises(EvalRunClaimLost):
            await store.heartbeat_run(expiring.claim)
        assert await store.claim_run() is None

    asyncio.run(exercise())


def test_memory_store_publishes_only_matching_immutable_safe_results() -> None:
    async def exercise() -> None:
        corpus = _corpus(trials=1)
        result = await run_corpus_suite(
            _target(_provider(trials=1)),
            corpus,
            corpus.suites[0].id,
            max_concurrency=1,
        )
        store = InMemoryEvalStore()
        await _save_corpus(store, corpus)
        await _admit_run(store, _request(corpus))
        claimed = await store.claim_run()
        assert claimed is not None
        claim = claimed.claim

        completed = await _publish_result(store, claim, result)
        assert completed.status is EvalRunStatus.COMPLETED
        assert completed.attempt_count == 1
        assert completed.result is not None
        assert completed.result.revision == result.revision
        assert await _publish_result(store, claim, result) == completed
        assert await store.load_result(completed.id) == result

        runs = await store.list_runs(EvalRunQuery(status=EvalRunStatus.COMPLETED))
        assert runs.items == (completed,)

        changed = result.model_copy(
            update={"revision": "sha256:" + "0" * 64},
        )
        with pytest.raises(Exception):
            await _publish_result(store, claim, changed)

    asyncio.run(exercise())


def test_repeated_model_judge_attempts_are_fenced_and_remain_attributable() -> None:
    async def exercise() -> None:
        clock = _Clock()
        store = InMemoryEvalStore(clock=clock)
        judge, judge_provider = _model_judge_target(requests=2)
        candidate_provider = _provider(trials=2)
        target = _target(candidate_provider, model_judges=(judge,))
        corpus = _model_judge_corpus(judge)
        await _save_corpus(store, corpus)
        await _admit_run(store, _request(corpus))

        first_lease = await store.claim_run(lease_seconds=10)
        assert first_lease is not None
        first_result = await run_corpus_suite(target, corpus, corpus.suites[0].id)

        clock.advance(11)
        second_lease = await store.claim_run(lease_seconds=10)
        assert second_lease is not None
        second_result = await run_corpus_suite(target, corpus, corpus.suites[0].id)

        with pytest.raises(EvalRunClaimLost):
            await _publish_result(store, first_lease.claim, first_result)
        completed = await _publish_result(store, second_lease.claim, second_result)

        assert completed.status is EvalRunStatus.COMPLETED
        assert completed.attempt_count == 2
        assert second_lease.claim.epoch == 2
        assert len(candidate_provider.requests) == 2
        assert len(judge_provider.requests) == 2
        assert await store.load_result(completed.id) == second_result

    asyncio.run(exercise())


def test_memory_store_failure_codes_replace_exception_text_and_invalid_transitions_reject() -> None:
    async def exercise() -> None:
        corpus = _corpus()
        store = InMemoryEvalStore()
        await _save_corpus(store, corpus)
        await _admit_run(store, _request(corpus))
        claimed = await store.claim_run()
        assert claimed is not None
        claim = claimed.claim

        failed = await store.fail_run(claim, EvalRunFailureCode.EXECUTION_FAILED)
        assert failed.status is EvalRunStatus.FAILED
        assert failed.failure_code is EvalRunFailureCode.EXECUTION_FAILED
        assert "exception" not in failed.model_dump_json().lower()
        assert await store.fail_run(claim, EvalRunFailureCode.EXECUTION_FAILED) == failed
        with pytest.raises(EvalRunStateConflict):
            await store.fail_run(claim, EvalRunFailureCode.WORKER_INTERRUPTED)

    asyncio.run(exercise())
