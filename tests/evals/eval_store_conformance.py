from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from typing import Literal

import pytest

from cayu.evals.corpus import (
    CorpusUserMessageSpec,
    EvalCaseSpec,
    EvalCorpusDocument,
    RunInputSpec,
    _content_revision,
)
from cayu.evals.execution import (
    CorpusExecutionResult,
    EvaluationTargetIdentity,
)
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
from cayu.runtime.manifest import AppManifest, ToolManifest
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


def _corpus_with_target(corpus: EvalCorpusDocument, target_key: str) -> EvalCorpusDocument:
    return EvalCorpusDocument.create(
        target_key=target_key,
        evidence_policy=corpus.evidence_policy,
        pricing_profile=corpus.pricing_profile,
        suites=corpus.suites,
        cases=corpus.cases,
    )


def _broken_redaction_boundary(_value):
    raise RuntimeError("must not cross the store boundary")


def _result_with_run_update(result: CorpusExecutionResult, **updates) -> CorpusExecutionResult:
    changed = result.run.model_copy(update=updates)
    revision_document = changed.model_dump(mode="json", exclude={"revision"})
    changed = changed.model_copy(
        update={"revision": _content_revision(revision_document, "published eval run")}
    )
    return CorpusExecutionResult.create(target=result.target, run=changed)


def _result_with_secret_manifest_key(
    result: CorpusExecutionResult,
    secret: str,
) -> CorpusExecutionResult:
    manifest = result.target.app_manifest
    agent = manifest.agents[0]
    tool = ToolManifest(
        name="credential-key-probe",
        description="Credential-key publication regression.",
        effect="read",
        parallel_safe=True,
        input_schema={secret: {"type": "string"}},
        policy_coverage="allowed",
        registration_provenance=agent.registration_provenance,
        implementation_provenance=agent.implementation_provenance,
    )
    document = manifest.model_dump(mode="json")
    document["agents"][0]["tools"] = [tool.model_dump(mode="json")]
    document.pop("fingerprint")
    canonical = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    document["fingerprint"] = hashlib.sha256(canonical).hexdigest()
    target = EvaluationTargetIdentity(
        target_key=result.target.target_key,
        application_release_id=result.target.application_release_id,
        app_manifest=AppManifest.model_validate(document),
    )
    return CorpusExecutionResult.create(target=target, run=result.run)


def _results_with_conflicting_corpus_contract(
    result: CorpusExecutionResult,
) -> tuple[CorpusExecutionResult, ...]:
    wrong_revision = "sha256:" + "f" * 64
    case = result.run.cases[0]
    trial = case.trials[0]
    assertion = trial.assertions[0]
    contradictory_detail = assertion.detail.model_copy(
        update={"expected": "failed", "actual": "failed"}
    )
    contradictory_assertion = assertion.model_copy(update={"detail": contradictory_detail})
    contradictory_trial = trial.model_copy(
        update={"assertions": (contradictory_assertion, *trial.assertions[1:])}
    )
    contradictory_case = case.model_copy(update={"trials": (contradictory_trial, *case.trials[1:])})
    return (
        _result_with_run_update(result, evidence_policy_revision=wrong_revision),
        _result_with_run_update(result, pricing_profile_fingerprint=wrong_revision),
        _result_with_run_update(
            result,
            cases=(
                case.model_copy(update={"case_revision": wrong_revision}),
                *result.run.cases[1:],
            ),
        ),
        _result_with_run_update(
            result,
            cases=(contradictory_case, *result.run.cases[1:]),
        ),
    )


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


async def assert_eval_store_reconstruction_releases_heartbeat_capacity(
    store: EvalStore,
    *,
    corpus: EvalCorpusDocument,
    result: CorpusExecutionResult,
    read_kind: Literal["corpus", "result"],
    parser_owner: object,
    parser_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove CPU reconstruction cannot occupy durable lease-operation capacity."""

    if read_kind not in {"corpus", "result"}:
        raise ValueError("read_kind must be corpus or result.")
    suite = corpus.suites[0]

    def request(run_id: str, idempotency_character: str) -> EvalRunRequest:
        return EvalRunRequest(
            run_id=run_id,
            idempotency_key="sha256:" + idempotency_character * 64,
            corpus_revision=corpus.revision,
            target_key=corpus.target_key,
            suite_id=suite.id,
            suite_revision=suite.revision,
            max_concurrency=1,
        )

    await store.save_corpus(corpus, redact_json=_NO_SECRETS.redact_json)
    await store.admit_run(
        request("capacity-completed-run", "7"),
        redact_json=_NO_SECRETS.redact_json,
    )
    completed_lease = await store.claim_run(target_key=corpus.target_key)
    assert completed_lease is not None
    await store.publish_result(
        completed_lease.claim,
        result,
        redact_json=_NO_SECRETS.redact_json,
    )

    await store.admit_run(
        request("capacity-active-run", "8"),
        redact_json=_NO_SECRETS.redact_json,
    )
    active_lease = await store.claim_run(
        target_key=corpus.target_key,
        lease_seconds=5,
    )
    assert active_lease is not None

    parser_started = threading.Event()
    release_parser = threading.Event()
    original_parser = getattr(parser_owner, parser_name)

    def blocking_parser(document):
        parser_started.set()
        if not release_parser.wait(timeout=5):
            raise AssertionError("Timed out releasing eval document reconstruction.")
        return original_parser(document)

    monkeypatch.setattr(parser_owner, parser_name, blocking_parser)
    read_task = asyncio.create_task(
        store.load_corpus(corpus.revision)
        if read_kind == "corpus"
        else store.load_result("capacity-completed-run")
    )
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 2
        while not parser_started.is_set() and loop.time() < deadline:
            await asyncio.sleep(0.01)
        assert parser_started.is_set()

        heartbeat = await asyncio.wait_for(
            store.heartbeat_run(active_lease.claim, extend_seconds=5),
            timeout=2,
        )
        assert heartbeat.ownership is not None
        assert heartbeat.ownership.epoch == active_lease.claim.epoch

        release_parser.set()
        assert await read_task == (corpus if read_kind == "corpus" else result)
        await store.release_run(active_lease.claim)
    finally:
        release_parser.set()
        await asyncio.gather(read_task, return_exceptions=True)


async def assert_eval_store_conformance(
    store: EvalStore,
    *,
    corpus,
    result: CorpusExecutionResult,
) -> None:
    """Pin backend-neutral catalog, lifecycle, fencing, and result semantics."""

    saved = await store.save_corpus(
        corpus,
        redact_json=_NO_SECRETS.redact_json,
    )
    assert (
        await store.save_corpus(
            corpus,
            redact_json=_NO_SECRETS.redact_json,
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
            redact_json=SecretRedactor(secret).redact_json,
        )
    assert await store.load_corpus(unsafe.revision) is None
    with pytest.raises(EvalStorePublicationRejected, match="could not cross"):
        await store.save_corpus(
            corpus,
            redact_json=_broken_redaction_boundary,
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
            redact_json=SecretRedactor(cancel_request.run_id).redact_json,
        )
    assert await store.load_run(cancel_request.run_id) is None
    admitted = await store.admit_run(
        cancel_request,
        redact_json=_NO_SECRETS.redact_json,
    )
    assert (
        await store.admit_run(
            cancel_request.model_copy(update={"run_id": "conformance-a-retry"}),
            redact_json=_NO_SECRETS.redact_json,
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
            redact_json=_NO_SECRETS.redact_json,
        )
    cancelled = await store.finish_cancel(claim)
    assert cancelled.status is EvalRunStatus.CANCELLED
    assert await store.finish_cancel(claim) == cancelled

    result_request = _request(corpus, suffix="b")
    await store.admit_run(
        result_request,
        redact_json=_NO_SECRETS.redact_json,
    )
    result_claimed = await store.claim_run()
    assert result_claimed is not None
    result_claim = result_claimed.claim
    with pytest.raises(EvalStorePublicationRejected, match="configured workload secret"):
        await store.publish_result(
            result_claim,
            result,
            redact_json=SecretRedactor(result.target.application_release_id).redact_json,
        )
    secret_key = "workload-secret-key-canary-ABCDEFGHIJKLMNOP"
    with pytest.raises(EvalStorePublicationRejected, match="configured workload secret"):
        await store.publish_result(
            result_claim,
            _result_with_secret_manifest_key(result, secret_key),
            redact_json=SecretRedactor(secret_key).redact_json,
        )
    for conflicting_result in _results_with_conflicting_corpus_contract(result):
        with pytest.raises(EvalRunStateConflict, match="immutable corpus suite contract"):
            await store.publish_result(
                result_claim,
                conflicting_result,
                redact_json=_NO_SECRETS.redact_json,
            )
    still_running = await store.load_run(result_claim.run_id)
    assert still_running is not None
    assert still_running.status is EvalRunStatus.RUNNING
    completed = await store.publish_result(
        result_claim,
        result,
        redact_json=_NO_SECRETS.redact_json,
    )
    assert completed.status is EvalRunStatus.COMPLETED
    assert (
        await store.publish_result(
            result_claim,
            result,
            redact_json=_NO_SECRETS.redact_json,
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
        redact_json=_NO_SECRETS.redact_json,
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

    other_corpus = _corpus_with_target(corpus, "other-target")
    await store.save_corpus(other_corpus, redact_json=_NO_SECRETS.redact_json)
    other_suite = other_corpus.suites[0]
    other_request = EvalRunRequest(
        run_id="target-scope-other",
        idempotency_key="sha256:" + "d" * 64,
        corpus_revision=other_corpus.revision,
        target_key=other_corpus.target_key,
        suite_id=other_suite.id,
        suite_revision=other_suite.revision,
        max_concurrency=1,
    )
    main_suite = corpus.suites[0]
    main_request = EvalRunRequest(
        run_id="target-scope-main",
        idempotency_key="sha256:" + "e" * 64,
        corpus_revision=corpus.revision,
        target_key=corpus.target_key,
        suite_id=main_suite.id,
        suite_revision=main_suite.revision,
        max_concurrency=1,
    )
    await store.admit_run(other_request, redact_json=_NO_SECRETS.redact_json)
    await store.admit_run(main_request, redact_json=_NO_SECRETS.redact_json)

    other_claimed = await store.claim_run_for_targets((corpus.target_key, other_corpus.target_key))
    assert other_claimed is not None
    assert other_claimed.run.id == other_request.run_id
    await store.release_run(other_claimed.claim)
    await store.request_cancel(other_request.run_id)
    main_claimed = await store.claim_run(target_key=corpus.target_key)
    assert main_claimed is not None
    assert main_claimed.run.id == main_request.run_id
    await store.release_run(main_claimed.claim)
    await store.request_cancel(main_request.run_id)

    with pytest.raises(ValueError, match="cannot be empty"):
        await store.claim_run_for_targets(())
    with pytest.raises(ValueError, match="must be unique"):
        await store.claim_run_for_targets((corpus.target_key, corpus.target_key))

    target_page = await store.list_runs(EvalRunQuery(target_key=corpus.target_key, limit=1))
    assert target_page.items
    assert all(item.spec.target_key == corpus.target_key for item in target_page.items)
    assert target_page.next_cursor is not None
    with pytest.raises(ValueError, match="cursor does not match this query"):
        await store.list_runs(
            EvalRunQuery(
                target_key=other_corpus.target_key,
                limit=1,
                cursor=target_page.next_cursor,
            )
        )
