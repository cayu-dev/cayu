from __future__ import annotations

import asyncio

import pytest
from tests.evals.eval_store_conformance import captured_result_for_corpus
from tests.evals.test_corpus_execution import _corpus, _provider, _target

from cayu.evals.corpus import EvalCaseSpec, EvalCorpusDocument
from cayu.evals.execution import run_corpus_suite
from cayu.evals.execution_comparison import (
    compare_corpus_execution_results,
    compare_eval_results,
    eval_result_compatibility,
)
from cayu.evals.results import (
    CapturedEvaluationResultV1,
    EvalResultOrigin,
    EvalResultTargetIdentityV1,
    captured_evaluation_result_from_json,
    eval_result_projection,
)
from cayu.evals.store import (
    EvalBaselineConflict,
    EvalBaselineKey,
    EvalBaselineUpdate,
    EvalStorePublicationRejected,
    EvalStoreResultTooLarge,
    InMemoryEvalStore,
)
from cayu.vaults.redaction import SecretRedactor

_NO_SECRETS = SecretRedactor()


def test_captured_and_fresh_results_share_one_comparison_projection() -> None:
    corpus = _corpus(trials=1)
    fresh = asyncio.run(
        run_corpus_suite(
            _target(_provider(trials=1)),
            corpus,
            corpus.suites[0].id,
            max_concurrency=1,
        )
    )
    captured = captured_result_for_corpus(corpus, fresh)

    captured_projection = eval_result_projection(captured)
    fresh_projection = eval_result_projection(fresh)
    assert captured_projection.origin is EvalResultOrigin.CAPTURED_SESSION
    assert fresh_projection.origin is EvalResultOrigin.FRESH_EXECUTION
    assert captured_projection.target.application_release_id == "captured-release"
    assert fresh_projection.target.application_release_id == "release-2026-08-06"
    assert eval_result_compatibility(captured, fresh).comparable is True

    comparison = compare_eval_results(captured, fresh)
    assert comparison.compatibility.comparable is True
    assert comparison.regressions == ()
    with pytest.raises(TypeError, match="baseline must be an exact CorpusExecutionResult"):
        compare_corpus_execution_results(captured, fresh)  # type: ignore[arg-type]


def test_captured_result_rejects_corpus_and_source_drift() -> None:
    corpus = _corpus(trials=1)
    fresh = asyncio.run(
        run_corpus_suite(
            _target(_provider(trials=1)),
            corpus,
            corpus.suites[0].id,
            max_concurrency=1,
        )
    )
    captured = captured_result_for_corpus(corpus, fresh)
    forged = captured.model_dump(mode="json")
    forged["target"]["application_release_id"] = "another-release"
    with pytest.raises(ValueError, match="source identity"):
        CapturedEvaluationResultV1.create(
            corpus=corpus,
            target=EvalResultTargetIdentityV1.model_validate(forged["target"]),
            score=captured.score,
        )

    original = corpus.cases[0]
    second = EvalCaseSpec.create(
        id="second-captured-case",
        suite_id=original.suite_id,
        name="Second captured case",
        description=None,
        source=original.source,
        input=original.input,
        assertions=original.assertions,
    )
    multi_case_corpus = EvalCorpusDocument.create(
        target_key=corpus.target_key,
        evidence_policy=corpus.evidence_policy,
        pricing_profile=corpus.pricing_profile,
        suites=corpus.suites,
        cases=(*corpus.cases, second),
    )
    with pytest.raises(ValueError, match="exactly its scored case"):
        CapturedEvaluationResultV1.create(
            corpus=multi_case_corpus,
            target=captured.target,
            score=captured.score,
        )


def test_captured_result_json_rejects_duplicate_keys_and_unknown_versions() -> None:
    corpus = _corpus(trials=1)
    fresh = asyncio.run(
        run_corpus_suite(
            _target(_provider(trials=1)),
            corpus,
            corpus.suites[0].id,
            max_concurrency=1,
        )
    )
    captured = captured_result_for_corpus(corpus, fresh)
    source = captured.model_dump_json()
    duplicated = source.replace(
        '"schema_version":1',
        '"schema_version":1,"schema_version":1',
        1,
    )
    with pytest.raises(ValueError, match="duplicate.*key"):
        captured_evaluation_result_from_json(duplicated)
    with pytest.raises(ValueError, match="unsupported schema_version"):
        captured_evaluation_result_from_json(
            source.replace('"schema_version":1', '"schema_version":2', 1)
        )


def test_memory_store_saves_captured_result_and_audited_baseline_atomically() -> None:
    async def exercise() -> None:
        corpus = _corpus(trials=1)
        fresh = await run_corpus_suite(
            _target(_provider(trials=1)),
            corpus,
            corpus.suites[0].id,
            max_concurrency=1,
        )
        captured = captured_result_for_corpus(corpus, fresh)
        store = InMemoryEvalStore()
        saved = await store.save_captured_result(
            corpus,
            captured,
            redact_json=_NO_SECRETS.redact_json,
        )
        assert saved.revision == captured.revision
        assert saved.origin is EvalResultOrigin.CAPTURED_SESSION
        assert await store.load_corpus(corpus.revision) == corpus
        assert await store.load_result_by_revision(captured.revision) == captured
        assert await store.load_result_record(captured.revision) == saved
        with pytest.raises(EvalStoreResultTooLarge):
            await store.load_result_by_revision(
                captured.revision,
                max_bytes=saved.document_bytes - 1,
            )

        key = EvalBaselineKey(
            target_key=corpus.target_key,
            corpus_revision=corpus.revision,
            suite_id=corpus.suites[0].id,
        )
        update = EvalBaselineUpdate(
            key=key,
            result_revision=captured.revision,
            expected_generation=0,
            operation_id="sha256:" + "1" * 64,
            actor_id="operator-123",
        )
        mutation = await store.set_baseline(
            update,
            redact_json=_NO_SECRETS.redact_json,
        )
        assert mutation.resulting_generation == 1
        assert mutation.previous_result_revision is None
        assert (
            await store.set_baseline(
                update,
                redact_json=_NO_SECRETS.redact_json,
            )
            == mutation
        )
        assert await store.load_baseline_mutation(update.operation_id) == mutation
        baseline = await store.load_baseline(key)
        assert baseline is not None
        assert baseline.result_revision == captured.revision
        assert baseline.updated_by == "operator-123"

        with pytest.raises(EvalBaselineConflict, match="generation changed"):
            await store.set_baseline(
                update.model_copy(update={"operation_id": "sha256:" + "2" * 64}),
                redact_json=_NO_SECRETS.redact_json,
            )
        with pytest.raises(EvalBaselineConflict, match="another mutation"):
            await store.set_baseline(
                update.model_copy(update={"actor_id": "operator-456"}),
                redact_json=_NO_SECRETS.redact_json,
            )

        secret = "private-actor-canary-ABCDEFGHIJKLMNOP"
        with pytest.raises(EvalStorePublicationRejected, match="configured workload secret"):
            await store.set_baseline(
                EvalBaselineUpdate(
                    key=key,
                    result_revision=captured.revision,
                    expected_generation=1,
                    operation_id="sha256:" + "3" * 64,
                    actor_id=secret,
                ),
                redact_json=SecretRedactor(secret).redact_json,
            )
        await store.close()

    asyncio.run(exercise())
