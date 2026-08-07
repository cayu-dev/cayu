from __future__ import annotations

import json

import pytest

from cayu.evals.corpus import (
    CorpusUserMessageSpec,
    EvalCaseSpec,
    EvalCorpusDocument,
    RunInputSpec,
)
from cayu.evals.execution import CorpusExecutionResult
from cayu.evals.store import (
    EvalCaseCatalogQuery,
    EvalCatalogQuery,
    EvalRunClaimLost,
    EvalRunFailureCode,
    EvalRunQuery,
    EvalRunRequest,
    EvalRunStateConflict,
    EvalRunStatus,
    EvalStore,
    EvalStorePublicationRejected,
    EvalStoreResultTooLarge,
    EvalSuiteCatalogQuery,
)
from cayu.vaults.redaction import SecretRedactor

_NO_SECRETS = SecretRedactor()


def _corpus_with_input(corpus: EvalCorpusDocument, text: str) -> EvalCorpusDocument:
    original = corpus.cases[0]
    case = EvalCaseSpec.create(
        id=original.id,
        suite_id=original.suite_id,
        name=original.name,
        description=original.description,
        source=original.source,
        input=RunInputSpec(messages=(CorpusUserMessageSpec(text=text),)),
        assertions=original.assertions,
    )
    return EvalCorpusDocument.create(
        target_key=corpus.target_key,
        evidence_policy=corpus.evidence_policy,
        pricing_profile=corpus.pricing_profile,
        suites=corpus.suites,
        cases=(case,),
    )


def _broken_redaction_boundary(_value):
    raise RuntimeError("must not cross the store boundary")


def _request(corpus, *, suffix: str, concurrency: int = 1) -> EvalRunRequest:
    suite = corpus.suites[0]
    digest_character = {"a": "a", "b": "b", "c": "c"}[suffix]
    return EvalRunRequest(
        run_id=f"conformance-{suffix}",
        idempotency_key="sha256:" + digest_character * 64,
        corpus_revision=corpus.revision,
        target_key=corpus.target_key,
        suite_id=suite.id,
        suite_revision=suite.revision,
        max_concurrency=concurrency,
    )


async def assert_eval_store_conformance(
    store: EvalStore,
    *,
    corpus,
    result: CorpusExecutionResult,
) -> None:
    """Pin backend-neutral catalog, lifecycle, fencing, and result semantics."""

    saved = await store.save_corpus(
        corpus,
        redact_json_values=_NO_SECRETS.redact_json_values,
    )
    assert (
        await store.save_corpus(
            corpus,
            redact_json_values=_NO_SECRETS.redact_json_values,
        )
        == saved
    )
    corpus_bytes = len(
        json.dumps(
            corpus.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    assert saved.document_bytes == corpus_bytes
    assert await store.load_corpus(corpus.revision, max_bytes=corpus_bytes) == corpus
    with pytest.raises(EvalStoreResultTooLarge):
        await store.load_corpus(corpus.revision, max_bytes=corpus_bytes - 1)

    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"
    unsafe = _corpus_with_input(corpus, secret)
    with pytest.raises(EvalStorePublicationRejected, match="configured workload secret"):
        await store.save_corpus(
            unsafe,
            redact_json_values=SecretRedactor(secret).redact_json_values,
        )
    assert await store.load_corpus(unsafe.revision) is None
    with pytest.raises(EvalStorePublicationRejected, match="could not cross"):
        await store.save_corpus(
            corpus,
            redact_json_values=_broken_redaction_boundary,
        )

    corpora = await store.list_corpora(EvalCatalogQuery(limit=1))
    assert corpora.items == (saved,)
    assert corpora.has_more is False
    suites = await store.list_suites(EvalSuiteCatalogQuery(corpus_revision=corpus.revision))
    assert [(item.id, item.revision) for item in suites.items] == [
        (corpus.suites[0].id, corpus.suites[0].revision)
    ]
    cases = await store.list_cases(
        EvalCaseCatalogQuery(
            corpus_revision=corpus.revision,
            suite_id=corpus.suites[0].id,
        )
    )
    assert [(item.id, item.revision) for item in cases.items] == [
        (corpus.cases[0].id, corpus.cases[0].revision)
    ]

    cancel_request = _request(corpus, suffix="a")
    with pytest.raises(EvalStorePublicationRejected, match="configured workload secret"):
        await store.admit_run(
            cancel_request,
            redact_json_values=SecretRedactor(cancel_request.run_id).redact_json_values,
        )
    assert await store.load_run(cancel_request.run_id) is None
    admitted = await store.admit_run(
        cancel_request,
        redact_json_values=_NO_SECRETS.redact_json_values,
    )
    assert (
        await store.admit_run(
            cancel_request.model_copy(update={"run_id": "conformance-a-retry"}),
            redact_json_values=_NO_SECRETS.redact_json_values,
        )
        == admitted
    )
    claimed = await store.claim_run()
    assert claimed is not None
    assert claimed.run.id == admitted.id
    active_public_record = await store.load_run(admitted.id)
    assert active_public_record == claimed.run
    active_public_json = active_public_record.model_dump_json()
    assert "claim_id" not in active_public_json
    assert "idempotency_key" not in active_public_json
    assert "owner_id" not in active_public_json
    claim = claimed.claim
    renewed = await store.heartbeat_run(claim)
    assert renewed.ownership is not None
    cancelling = await store.request_cancel(admitted.id)
    assert cancelling.status is EvalRunStatus.CANCELLING
    with pytest.raises(EvalRunStateConflict):
        await store.publish_result(
            claim,
            result,
            redact_json_values=_NO_SECRETS.redact_json_values,
        )
    cancelled = await store.finish_cancel(claim)
    assert cancelled.status is EvalRunStatus.CANCELLED
    assert await store.finish_cancel(claim) == cancelled

    result_request = _request(corpus, suffix="b")
    await store.admit_run(
        result_request,
        redact_json_values=_NO_SECRETS.redact_json_values,
    )
    result_claimed = await store.claim_run()
    assert result_claimed is not None
    result_claim = result_claimed.claim
    with pytest.raises(EvalStorePublicationRejected, match="configured workload secret"):
        await store.publish_result(
            result_claim,
            result,
            redact_json_values=SecretRedactor(
                result.target.application_release_id
            ).redact_json_values,
        )
    still_running = await store.load_run(result_claim.run_id)
    assert still_running is not None
    assert still_running.status is EvalRunStatus.RUNNING
    completed = await store.publish_result(
        result_claim,
        result,
        redact_json_values=_NO_SECRETS.redact_json_values,
    )
    assert completed.status is EvalRunStatus.COMPLETED
    assert (
        await store.publish_result(
            result_claim,
            result,
            redact_json_values=_NO_SECRETS.redact_json_values,
        )
        == completed
    )
    result_bytes = len(
        json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    assert await store.load_result(completed.id, max_bytes=result_bytes) == result
    with pytest.raises(EvalStoreResultTooLarge):
        await store.load_result(completed.id, max_bytes=result_bytes - 1)

    failure_request = _request(corpus, suffix="c")
    await store.admit_run(
        failure_request,
        redact_json_values=_NO_SECRETS.redact_json_values,
    )
    failure_claimed = await store.claim_run()
    assert failure_claimed is not None
    stale_failure_claim = failure_claimed.claim
    released = await store.release_run(stale_failure_claim)
    assert released.status is EvalRunStatus.QUEUED
    failure_reclaimed = await store.claim_run()
    assert failure_reclaimed is not None
    with pytest.raises(EvalRunClaimLost):
        await store.heartbeat_run(stale_failure_claim)
    failure_claim = failure_reclaimed.claim
    failed = await store.fail_run(failure_claim, EvalRunFailureCode.EXECUTION_FAILED)
    assert failed.status is EvalRunStatus.FAILED
    assert await store.fail_run(failure_claim, EvalRunFailureCode.EXECUTION_FAILED) == failed

    terminal = await store.list_runs(EvalRunQuery(limit=3))
    assert {item.status for item in terminal.items} == {
        EvalRunStatus.CANCELLED,
        EvalRunStatus.COMPLETED,
        EvalRunStatus.FAILED,
    }
    public_records = terminal.model_dump_json()
    assert "trajectory" not in public_records
    assert "exception" not in public_records
    assert "credential" not in public_records
    assert "request-" not in public_records
    assert "idempotency_key" not in public_records
    assert "claim_id" not in public_records
