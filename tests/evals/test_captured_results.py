from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError
from tests.evals.eval_store_conformance import captured_result_for_corpus
from tests.evals.test_corpus_execution import _corpus, _provider, _target

from cayu.evals.corpus import EvalCaseSpec, EvalCorpusDocument
from cayu.evals.execution import run_corpus_suite
from cayu.evals.execution_comparison import (
    CorpusComparisonReason,
    compare_corpus_execution_results,
    compare_eval_results,
    eval_result_compatibility,
)
from cayu.evals.execution_reporting import render_captured_evaluation_html
from cayu.evals.results import (
    EVAL_RESULT_PROJECTION_SCHEMA_VERSION,
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
    assert EVAL_RESULT_PROJECTION_SCHEMA_VERSION == 2
    assert captured_projection.schema_version == 2
    assert fresh_projection.schema_version == 2
    assert fresh_projection.trial_policy_revision == fresh.run.trial_policy.revision
    assert captured_projection.target.application_release_id == "captured-release"
    assert fresh_projection.target.application_release_id == "release-2026-08-06"
    assert captured.score.memory_attribution == (fresh.run.cases[0].trials[0].memory_attribution)
    assert captured_projection.memory_attribution_support == "unsupported"
    assert fresh_projection.memory_attribution_support == "unsupported"
    assert eval_result_compatibility(captured, fresh).comparable is True

    comparison = compare_eval_results(captured, fresh)
    assert comparison.compatibility.comparable is True
    assert comparison.baseline.memory_attribution_support == "unsupported"
    assert comparison.current.memory_attribution_support == "unsupported"
    rendered = render_captured_evaluation_html(captured)
    assert "Memory attribution complete" in rendered
    assert "limitations none" in rendered
    assert "lifecycle none" in rendered
    assert "record inspection is unsupported in HTML" in rendered
    assert captured.score.memory_attribution.revision in rendered
    assert comparison.regressions == ()
    with pytest.raises(TypeError, match="baseline must be an exact CorpusExecutionResult"):
        compare_corpus_execution_results(captured, fresh)  # type: ignore[arg-type]


def test_comparison_distinguishes_trial_policy_and_accepted_exposure_contract() -> None:
    corpus = _corpus(trials=1)
    fresh = asyncio.run(
        run_corpus_suite(
            _target(_provider(trials=1)),
            corpus,
            corpus.suites[0].id,
            max_concurrency=1,
        )
    )
    projection = eval_result_projection(fresh)

    policy_changed = projection.model_copy(update={"trial_policy_revision": "sha256:" + "d" * 64})
    policy_compatibility = eval_result_compatibility(projection, policy_changed)
    assert policy_compatibility.comparable is False
    assert policy_compatibility.reasons == (CorpusComparisonReason.TRIAL_POLICY_REVISION_MISMATCH,)

    baseline_exposure = projection.model_copy(
        update={
            "accepted_exposure_revision": "sha256:" + "e" * 64,
            "accepted_exposure_comparison_revision": "sha256:" + "a" * 64,
        }
    )
    release_changed_exposure = projection.model_copy(
        update={
            "accepted_exposure_revision": "sha256:" + "f" * 64,
            "accepted_exposure_comparison_revision": "sha256:" + "a" * 64,
        }
    )
    release_compatibility = eval_result_compatibility(
        baseline_exposure,
        release_changed_exposure,
    )
    assert release_compatibility.comparable is True
    assert release_compatibility.reasons == ()
    release_comparison = compare_eval_results(baseline_exposure, release_changed_exposure)
    assert release_comparison.baseline.accepted_exposure_revision != (
        release_comparison.current.accepted_exposure_revision
    )
    assert release_comparison.baseline.accepted_exposure_comparison_revision == (
        release_comparison.current.accepted_exposure_comparison_revision
    )

    contract_changed_exposure = release_changed_exposure.model_copy(
        update={"accepted_exposure_comparison_revision": "sha256:" + "b" * 64}
    )
    exposure_compatibility = eval_result_compatibility(
        baseline_exposure,
        contract_changed_exposure,
    )
    assert exposure_compatibility.comparable is False
    assert exposure_compatibility.reasons == (
        CorpusComparisonReason.ACCEPTED_EXPOSURE_CONTRACT_MISMATCH,
    )

    missing_exposure_compatibility = eval_result_compatibility(
        baseline_exposure,
        projection,
    )
    assert missing_exposure_compatibility.comparable is False
    assert missing_exposure_compatibility.reasons == (
        CorpusComparisonReason.ACCEPTED_EXPOSURE_CONTRACT_MISMATCH,
    )

    captured = captured_result_for_corpus(corpus, fresh)
    captured_projection = eval_result_projection(captured)
    cross_origin_compatibility = eval_result_compatibility(
        captured_projection,
        baseline_exposure,
    )
    assert CorpusComparisonReason.ACCEPTED_EXPOSURE_CONTRACT_MISMATCH not in (
        cross_origin_compatibility.reasons
    )
    forged_captured_projection = captured_projection.model_dump(mode="python")
    forged_captured_projection["accepted_exposure_revision"] = "sha256:" + "a" * 64
    forged_captured_projection["accepted_exposure_comparison_revision"] = "sha256:" + "b" * 64
    with pytest.raises(ValidationError, match="cannot carry accepted work exposure"):
        type(captured_projection).model_validate(forged_captured_projection)


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
