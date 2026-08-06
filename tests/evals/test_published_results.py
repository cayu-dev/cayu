from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from cayu.evals.corpus import (
    ChildStatusAssertionSpec,
    CorpusUserMessageSpec,
    EvalCaseSpec,
    EvalCorpusDocument,
    EvalSuiteSpec,
    EvaluationEvidencePolicySpec,
    EvaluationSourceIdentityV1,
    FinalOutputContainsAssertionSpec,
    FinalOutputEqualsAssertionSpec,
    MaxEstimatedCostAssertionSpec,
    MaxModelStepsAssertionSpec,
    MaxToolCallsAssertionSpec,
    MaxTotalTokensAssertionSpec,
    PricingProfileIdentityV1,
    RootStatusAssertionSpec,
    RunInputSpec,
    ToolCalledAssertionSpec,
    ToolsCalledInOrderAssertionSpec,
    TrialRequestSpec,
    UsageRecordedAssertionSpec,
    assertion_spec_revision,
    eval_run_contract_for_corpus,
)
from cayu.evals.models import (
    EvalAssertionResult,
    EvalCaseResult,
    EvalOutcome,
    EvalRun,
    EvalStatus,
    EvalTrialResult,
)
from cayu.evals.published import (
    PUBLISHED_EVAL_SCHEMA_VERSION,
    PublishedAssertionResult,
    PublishedEvalRun,
    PublishedEvalTrialResult,
    PublishedFinalOutputEqualsDetail,
    PublishedMaxEstimatedCostDetail,
    PublishedToolsCalledInOrderDetail,
    PublishedUsageSummaryV1,
    publish_eval_run,
)
from cayu.evals.reporting import eval_run_to_json, load_eval_run
from cayu.evals.result_contract import (
    EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES,
    PUBLISHED_EVAL_OUTPUT_PREVIEW_BUDGET_BYTES,
    EvalTrialOutputPreviewV1,
)
from cayu.runtime.costs import SessionCostSummary
from cayu.runtime.usage import (
    SessionUsageSummary,
    UsageMetrics,
    build_aggregate_usage_metrics,
    session_usage_summary_payload,
)


def _specs():
    return (
        RootStatusAssertionSpec(id="root", expected="completed"),
        ChildStatusAssertionSpec(id="child", expected="failed", min_count=1, max_count=1),
        FinalOutputEqualsAssertionSpec(id="equals", expected="Approved"),
        FinalOutputContainsAssertionSpec(id="contains", expected="prove"),
        ToolCalledAssertionSpec(id="called", tool_name="lookup", min_count=1, max_count=1),
        ToolsCalledInOrderAssertionSpec(id="order", tool_names=("lookup",)),
        MaxToolCallsAssertionSpec(id="tools", maximum=1),
        MaxModelStepsAssertionSpec(id="steps", maximum=1),
        UsageRecordedAssertionSpec(id="usage", min_total_tokens=1),
        MaxTotalTokensAssertionSpec(id="tokens", maximum=15),
        MaxEstimatedCostAssertionSpec(id="cost", maximum="1", currency="USD"),
    )


def _corpus() -> EvalCorpusDocument:
    suite = EvalSuiteSpec.create(
        id="suite",
        name="Suite",
        trial_request=TrialRequestSpec(trials=2, timeout_seconds=60),
    )
    case = EvalCaseSpec.create(
        id="case",
        suite_id=suite.id,
        name="Case",
        source=EvaluationSourceIdentityV1(
            application_release_id="release",
            app_manifest_schema_version="7",
            app_manifest_fingerprint="a" * 64,
            evidence_revision="sha256:" + "b" * 64,
        ),
        input=RunInputSpec(messages=(CorpusUserMessageSpec(text="Run the case."),)),
        assertions=_specs(),
    )
    return EvalCorpusDocument.create(
        target_key="target",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        pricing_profile=PricingProfileIdentityV1(
            pricing_semantics_version=1,
            fingerprint="sha256:" + "c" * 64,
            price_book_version="v1",
            generated_at="2026-08-05T00:00:00Z",
            currencies=("USD",),
        ),
        suites=(suite,),
        cases=(case,),
    )


def _metadata(kind: str) -> dict:
    common = {"secret_field": "secret-token", "session_id": "private-session"}
    return {
        "root_status": {**common, "actual": "completed"},
        "child_status": {**common, "count": 1},
        "final_output_equals": {**common, "matched": True},
        "final_output_contains": {**common, "matched": True},
        "tool_called": {**common, "count": 1},
        "tools_called_in_order": {**common, "actual": ["lookup"]},
        "max_tool_calls": {**common, "actual": 1},
        "max_model_steps": {**common, "actual": 1},
        "usage_recorded": {**common, "total_tokens": 15},
        "max_total_tokens": {**common, "actual": 15},
        "max_estimated_cost": {
            **common,
            "estimated_cost": "0.000025",
            "priced_model_steps": 1,
            "unpriced_model_steps": 0,
            "currency": "USD",
        },
    }[kind]


def _assertion_results(*, cost_unavailable: bool):
    results = []
    for spec in _specs():
        unavailable = cost_unavailable and spec.kind == "max_estimated_cost"
        metadata = _metadata(spec.kind)
        if unavailable:
            metadata = {
                **metadata,
                "priced_model_steps": 0,
                "unpriced_model_steps": 1,
            }
        results.append(
            EvalAssertionResult(
                name=spec.id,
                assertion_revision=assertion_spec_revision(spec),
                outcome=EvalOutcome.UNAVAILABLE if unavailable else EvalOutcome.PASSED,
                score=None if unavailable else 1.0,
                message="secret-token raw diagnostic",
                metadata=metadata,
            )
        )
    return tuple(results)


def _trial(
    number: int,
    *,
    started_at: datetime,
    cost_unavailable: bool,
) -> EvalTrialResult:
    session_id = f"private-session-{number}"
    assertions = _assertion_results(cost_unavailable=cost_unavailable)
    return EvalTrialResult(
        trial_number=number,
        status=EvalStatus.UNAVAILABLE if cost_unavailable else EvalStatus.PASSED,
        session_id=session_id,
        score=None if cost_unavailable else 1.0,
        final_output="Approved",
        assertions=assertions,
        unavailable_reason=("secret-token raw unavailable" if cost_unavailable else None),
        evidence_complete=True,
        events_count=3,
        usage_summary=session_usage_summary_payload(
            SessionUsageSummary(
                session_id=session_id,
                model_steps=1,
                tool_calls=1,
                provider_names=["private-provider"],
                models=["private-model"],
                usage=UsageMetrics(total_tokens=15),
            )
        ),
        started_at=started_at,
        completed_at=started_at + timedelta(seconds=1),
        duration_ms=1_000,
    )


def _run(*, corpus: EvalCorpusDocument | None = None) -> EvalRun:
    contract_corpus = _corpus() if corpus is None else corpus
    started_at = datetime(2026, 8, 5, tzinfo=UTC)
    trials = (
        _trial(1, started_at=started_at, cost_unavailable=False),
        _trial(2, started_at=started_at + timedelta(seconds=1), cost_unavailable=True),
    )
    case = EvalCaseResult.from_trials(
        case_id="case",
        trials=trials,
        authored_session_id="private-authored-session",
        started_at=started_at,
        completed_at=started_at + timedelta(seconds=2),
        metadata={"secret": "secret-token"},
    )
    return EvalRun(
        run_id="private-run-id",
        suite_id="suite",
        status=case.status,
        score=case.score,
        cases=(case,),
        started_at=started_at,
        completed_at=started_at + timedelta(seconds=2),
        duration_ms=2_000,
        metadata={"secret": "secret-token"},
        run_contract=eval_run_contract_for_corpus(contract_corpus, "suite"),
    )


def test_published_graph_preserves_trials_and_reproducible_aggregates_only():
    corpus = _corpus()
    run = _run()
    published = publish_eval_run(corpus, run)

    assert PUBLISHED_EVAL_SCHEMA_VERSION == 2
    assert published.schema_version == 2
    assert published.corpus_revision == corpus.revision
    assert published.status == "unavailable"
    assert published.score is None
    assert published.duration_ms == 2_000
    assert [trial.status for trial in published.cases[0].trials] == [
        "passed",
        "unavailable",
    ]
    assert published.cases[0].trials[0].usage is not None
    assert published.cases[0].trials[0].usage.total_tokens == 15
    assert published.cases[0].trials[1].assertions[-1].score is None
    assert all(trial.output.evidence_state == "unavailable" for trial in published.cases[0].trials)
    assert run.cases[0].assertions[0].assertion_revision == assertion_spec_revision(
        corpus.cases[0].assertions[0]
    )

    encoded = published.model_dump_json()
    for forbidden in (
        "secret-token",
        "private-session",
        "private-run-id",
        "private-provider",
        "private-model",
        '"final_output":',
        '"unavailable_reason":',
        '"error"',
        '"metadata"',
        '"trajectory"',
    ):
        assert forbidden not in encoded
    assert PublishedEvalRun.model_validate_json(encoded) == published


def test_published_eval_run_rejects_v1_before_validating_its_obsolete_shape():
    published = publish_eval_run(_corpus(), _run())
    document = published.model_dump(mode="json")
    document["schema_version"] = 1
    for case in document["cases"]:
        for trial in case["trials"]:
            trial.pop("output")
            trial["code"] = trial["status"]

    with pytest.raises(ValidationError, match="other versions are unsupported"):
        PublishedEvalRun.model_validate(document)

    with pytest.raises(ValidationError, match="schema_version must be the integer 2"):
        PublishedEvalRun.model_validate(
            {**published.model_dump(mode="json"), "schema_version": "2"}
        )

    versionless = published.model_dump(mode="json")
    versionless.pop("schema_version")
    with pytest.raises(ValidationError, match="schema_version is required"):
        PublishedEvalRun.model_validate(versionless)


def test_output_preview_rejects_forged_size_digest_and_truncation_metadata():
    with pytest.raises(ValidationError, match="output preview exceeds"):
        EvalTrialOutputPreviewV1.model_validate(
            {
                "text": "x" * (EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES + 1),
                "evidence_state": "complete",
                "retained_chars": EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES + 1,
                "retained_bytes": EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES + 1,
                "retained_sha256": "0" * 64,
            }
        )
    preview = EvalTrialOutputPreviewV1.from_retained_evidence(
        "Approved",
        "complete",
        max_preview_bytes=EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES,
    )
    document = preview.model_dump(mode="python")

    for update, match in (
        ({"retained_chars": 1}, "character and UTF-8 byte counts"),
        ({"retained_sha256": "0" * 64}, "digest does not match"),
        ({"preview_truncated": True}, "must omit"),
    ):
        with pytest.raises(ValidationError, match=match):
            EvalTrialOutputPreviewV1.model_validate({**document, **update})

    truncated = EvalTrialOutputPreviewV1.from_retained_evidence(
        "x" * (EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES + 1),
        "complete",
        max_preview_bytes=EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES,
    ).model_dump(mode="python")
    for update, match in (
        ({"retained_chars": len(truncated["text"])}, "omit retained characters"),
        (
            {"retained_bytes": truncated["retained_chars"] * 4 + 1},
            "character and UTF-8 byte counts",
        ),
    ):
        with pytest.raises(ValidationError, match=match):
            EvalTrialOutputPreviewV1.model_validate({**truncated, **update})

    with pytest.raises(ValueError, match="max_preview_bytes"):
        EvalTrialOutputPreviewV1.from_retained_evidence(
            "",
            "unavailable",
            max_preview_bytes=0,
        )


def test_published_run_preflights_the_aggregate_output_preview_budget():
    document = publish_eval_run(_corpus(), _run()).model_dump(mode="python")
    trial = document["cases"][0]["trials"][0]
    text = "x" * EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES
    output = EvalTrialOutputPreviewV1.from_retained_evidence(
        text,
        "complete",
        max_preview_bytes=EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES,
    ).model_dump(mode="python")
    trials = [
        {
            **deepcopy(trial),
            "trial_number": number,
            "output": output,
        }
        for number in range(1, 101)
    ]
    # Two case branches would retain 3.125 MiB of previews. The 2 MiB public
    # budget must reject the raw graph before nested result construction.
    document["cases"] = [
        {**deepcopy(document["cases"][0]), "case_id": case_id, "trials": trials}
        for case_id in ("case-a", "case-b")
    ]

    with pytest.raises(ValidationError, match="aggregate output-preview limit"):
        PublishedEvalRun.model_validate(document)

    assert len(text) * 200 > PUBLISHED_EVAL_OUTPUT_PREVIEW_BUDGET_BYTES


def test_published_assertion_details_are_closed_and_allowlisted():
    published = publish_eval_run(_corpus(), _run())
    first_trial = published.cases[0].trials[0]

    assert [result.detail.kind for result in first_trial.assertions] == [
        spec.kind for spec in _specs()
    ]
    assert first_trial.assertions[0].detail.actual == "completed"
    assert first_trial.assertions[4].detail.matching_count == 1
    assert first_trial.assertions[-1].detail.estimated_cost == "0.000025"
    assert first_trial.assertions[-1].message == "Assertion passed."


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("status",), "passed", "aggregates"),
        (("duration_ms",), 1, "duration"),
        (("revision",), "sha256:" + "0" * 64, "revision"),
        (("cases", 0, "trials", 0, "message"), "raw error", "diagnostics"),
    ],
)
def test_published_models_reject_forged_derived_fields(path, value, match):
    document = publish_eval_run(_corpus(), _run()).model_dump(mode="python")
    target = document
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value

    with pytest.raises(ValidationError, match=match):
        PublishedEvalRun.model_validate(document)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda document: document["cases"][0]["trials"][0]["assertions"][0]["detail"].update(
                actual=None
            ),
            "observed evidence",
        ),
        (
            lambda document: document["cases"][0]["trials"][0]["assertions"][0]["detail"].update(
                actual="failed"
            ),
            "contradicts",
        ),
        (
            lambda document: document["cases"][0]["trials"][0]["assertions"][-1]["detail"].update(
                unpriced_model_steps=1
            ),
            "fully priced",
        ),
        (
            lambda document: document["cases"][0]["trials"][0]["assertions"][2]["detail"].update(
                matched=False
            ),
            "contradicts",
        ),
        (
            lambda document: document["cases"][0]["trials"][0]["assertions"][5]["detail"].update(
                matched=False
            ),
            "contradicts",
        ),
        (
            lambda document: document["cases"][0]["trials"][0]["usage"].update(total_tokens=16),
            "must match trial usage",
        ),
        (
            lambda document: document.update(pricing_profile_fingerprint=None),
            "pricing profile fingerprint",
        ),
    ],
)
def test_published_models_reject_impossible_observation_contracts(mutate, match):
    document = publish_eval_run(_corpus(), _run()).model_dump(mode="python")
    mutate(document)

    with pytest.raises(ValidationError, match=match):
        PublishedEvalRun.model_validate(document)


def test_published_usage_rejects_nonportable_integer_magnitude():
    with pytest.raises(ValidationError, match="less than or equal"):
        PublishedUsageSummaryV1(
            model_steps=2**63,
            tool_calls=0,
            total_tokens=0,
        )
    with pytest.raises(ValidationError, match="greater than or equal"):
        PublishedUsageSummaryV1(
            model_steps=0,
            tool_calls=0,
            total_tokens=-1,
        )


def test_published_usage_preserves_large_exact_aggregate_counts():
    base = _corpus()
    source_case = base.cases[0]
    suite = EvalSuiteSpec.create(
        id="large-suite",
        name="Large usage suite",
        trial_request=TrialRequestSpec(trials=1, timeout_seconds=60),
    )
    spec = RootStatusAssertionSpec(id="root", expected="completed")
    case_spec = EvalCaseSpec.create(
        id="large-case",
        suite_id=suite.id,
        name="Large usage case",
        source=source_case.source,
        input=source_case.input,
        assertions=(spec,),
    )
    corpus = EvalCorpusDocument.create(
        target_key=base.target_key,
        evidence_policy=base.evidence_policy,
        suites=(suite,),
        cases=(case_spec,),
    )
    started_at = datetime(2026, 8, 5, tzinfo=UTC)
    session_id = "large-usage-session"
    trial = EvalTrialResult(
        trial_number=1,
        status=EvalStatus.PASSED,
        session_id=session_id,
        score=1.0,
        assertions=(
            EvalAssertionResult(
                name=spec.id,
                assertion_revision=assertion_spec_revision(spec),
                outcome=EvalOutcome.PASSED,
                score=1.0,
                metadata={"actual": "completed"},
            ),
        ),
        evidence_complete=True,
        usage_summary=session_usage_summary_payload(
            SessionUsageSummary(
                session_id=session_id,
                model_steps=2,
                usage=build_aggregate_usage_metrics(total_tokens=2**63),
            )
        ),
        started_at=started_at,
        completed_at=started_at + timedelta(seconds=1),
        duration_ms=1_000,
    )
    case = EvalCaseResult.from_trials(case_id=case_spec.id, trials=(trial,))
    run = EvalRun(
        suite_id=suite.id,
        status=case.status,
        score=case.score,
        cases=(case,),
        started_at=started_at,
        completed_at=started_at + timedelta(seconds=1),
        duration_ms=1_000,
        run_contract=eval_run_contract_for_corpus(corpus, suite.id),
    )

    published = publish_eval_run(corpus, run)

    assert published.cases[0].trials[0].usage is not None
    assert published.cases[0].trials[0].usage.total_tokens == 2**63
    encoded = published.model_dump_json()
    assert f'"total_tokens":"{2**63}"' in encoded
    assert PublishedEvalRun.model_validate_json(encoded) == published


def test_published_tool_order_rejects_false_empty_sequence_match():
    with pytest.raises(ValidationError, match="Empty expected and actual tool orders must match"):
        PublishedToolsCalledInOrderDetail(
            expected_count=0,
            actual_count=0,
            matched=False,
        )


@pytest.mark.parametrize("outcome", ["unavailable", "error"])
def test_unscored_published_assertions_reject_conclusive_observations(outcome):
    with pytest.raises(ValidationError, match="observed|conclusive"):
        PublishedAssertionResult(
            assertion_id="output",
            assertion_revision="sha256:" + "a" * 64,
            outcome=outcome,
            score=None,
            code=outcome,
            message=(
                "Required evidence was unavailable."
                if outcome == "unavailable"
                else "Assertion evaluation failed."
            ),
            detail=PublishedFinalOutputEqualsDetail(matched=True),
        )


def test_unavailable_cost_observation_requires_unpriced_steps():
    common = {
        "assertion_id": "cost",
        "assertion_revision": "sha256:" + "a" * 64,
        "outcome": "unavailable",
        "score": None,
        "code": "unavailable",
        "message": "Required evidence was unavailable.",
    }
    valid = PublishedAssertionResult(
        **common,
        detail=PublishedMaxEstimatedCostDetail(
            maximum="1",
            currency="USD",
            estimated_cost="0",
            priced_model_steps=0,
            unpriced_model_steps=1,
        ),
    )
    assert valid.outcome == "unavailable"

    with pytest.raises(ValidationError, match="unpriced model steps"):
        PublishedAssertionResult(
            **common,
            detail=PublishedMaxEstimatedCostDetail(
                maximum="1",
                currency="USD",
                estimated_cost="0.1",
                priced_model_steps=1,
                unpriced_model_steps=0,
            ),
        )


def test_published_run_rejects_variable_suite_trial_counts():
    published = publish_eval_run(_corpus(), _run())
    case = published.cases[0]
    second_case = case.model_copy(
        update={
            "case_id": "case-two",
            "trials": case.trials[:1],
            "status": case.trials[0].status,
            "score": case.trials[0].score,
            "duration_ms": case.trials[0].duration_ms,
        }
    )
    document = published.model_dump(mode="python")
    document["cases"] = (case, second_case)
    document["duration_ms"] = case.duration_ms + second_case.duration_ms

    with pytest.raises(ValidationError, match="suite-wide trial count"):
        PublishedEvalRun.model_validate(document)


def test_published_run_rejects_more_than_corpus_expanded_result_limit():
    published = publish_eval_run(_corpus(), _run())
    source_trial = published.cases[0].trials[0]
    # 100 cases x 100 trials x 1 assertion reaches the 10,000-result limit;
    # the final 101st case proves the public graph enforces the same corpus bound.
    cases = []
    for case_index in range(101):
        trials = tuple(
            source_trial.model_copy(
                update={
                    "trial_number": trial_index + 1,
                    "assertions": source_trial.assertions[:1],
                    "score": 1.0,
                    "status": "passed",
                    "evidence_complete": True,
                    "code": "passed",
                    "message": "Trial passed.",
                }
            )
            for trial_index in range(100)
        )
        cases.append(
            published.cases[0].model_copy(
                update={
                    "case_id": f"case-{case_index:03d}",
                    "status": "passed",
                    "score": 1.0,
                    "trials": trials,
                    "duration_ms": sum(trial.duration_ms for trial in trials),
                }
            )
        )
    document = published.model_dump(mode="python")
    document.update(
        status="passed",
        score=1.0,
        cases=tuple(cases),
        duration_ms=sum(case.duration_ms for case in cases),
    )

    with pytest.raises(ValidationError, match="expanded assertion-result limit"):
        PublishedEvalRun.model_validate(document)


def test_published_graph_preflight_fails_closed_on_a_malformed_leading_branch():
    document = {
        "schema_version": PUBLISHED_EVAL_SCHEMA_VERSION,
        "cases": [
            {"trials": "malformed"},
            {"trials": [{"assertions": [{}] * 10_001}]},
        ],
    }

    with pytest.raises(ValidationError, match=r"cases\[0\]\.trials must be an array"):
        PublishedEvalRun.model_validate(document)


def test_projection_rejects_incomplete_or_misaligned_internal_contracts():
    corpus = _corpus()
    run = _run()
    wrong_name = run.cases[0].trials[0].assertions[0].model_copy(update={"name": "wrong"})
    assertions = (wrong_name, *run.cases[0].trials[0].assertions[1:])
    forged_trial = run.cases[0].trials[0].model_copy(update={"assertions": assertions})
    forged_case = run.cases[0].model_copy(update={"trials": (forged_trial, run.cases[0].trials[1])})
    forged_run = run.model_copy(update={"cases": (forged_case,)})

    with pytest.raises(ValidationError):
        publish_eval_run(corpus, forged_run)

    one_trial_case = EvalCaseResult.from_trials(
        case_id="case",
        trials=(run.cases[0].trials[0],),
    )
    with pytest.raises(ValidationError, match="trial counts must match its run contract"):
        EvalRun(
            suite_id="suite",
            status=one_trial_case.status,
            score=one_trial_case.score,
            cases=(one_trial_case,),
            started_at=one_trial_case.started_at,
            completed_at=one_trial_case.completed_at,
            duration_ms=one_trial_case.duration_ms,
            run_contract=run.run_contract,
        )


def test_lossless_run_rejects_contracted_trial_mismatch_during_validation_and_load(tmp_path):
    document = json.loads(eval_run_to_json(_run()))
    document["run_contract"]["trials"] = 1

    with pytest.raises(ValidationError, match="trial counts must match its run contract"):
        EvalRun.model_validate(document)

    path = tmp_path / "mismatched-contract.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError, match="trial counts must match its run contract"):
        load_eval_run(path)


def test_complete_lossless_and_published_trials_require_exact_usage():
    corpus = _corpus()
    run = _run(corpus=corpus)
    lossless = run.cases[0].trials[0].model_dump(mode="python")
    lossless["usage_summary"] = None

    with pytest.raises(ValidationError, match="Complete trial evidence requires an exact usage"):
        EvalTrialResult.model_validate(lossless)

    document = publish_eval_run(corpus, run).model_dump(mode="python")
    document["cases"][0]["trials"][0]["usage"] = None
    with pytest.raises(ValidationError, match="Complete published trials require exact usage"):
        PublishedEvalRun.model_validate(document)


def test_published_trial_rejects_unordered_assertion_input():
    trial = publish_eval_run(_corpus(), _run()).cases[0].trials[0]
    document = trial.model_dump(mode="python")
    document["assertions"] = set(trial.assertions)

    with pytest.raises(ValidationError, match="assertions.*ordered array"):
        PublishedEvalTrialResult.model_validate(document)


def test_eval_run_contract_rejects_unordered_case_input():
    contract = eval_run_contract_for_corpus(_corpus(), "suite")
    document = contract.model_dump(mode="python")
    document["cases"] = set(contract.cases)

    with pytest.raises(ValidationError, match="contract cases must be an ordered array"):
        type(contract).model_validate(document)


def test_published_trial_rejects_contradictory_root_status_observations():
    document = publish_eval_run(_corpus(), _run()).cases[0].trials[0].model_dump(mode="python")
    root = next(
        assertion
        for assertion in document["assertions"]
        if assertion["detail"]["kind"] == "root_status"
    )
    contradictory = deepcopy(root)
    contradictory["assertion_id"] = "root-failed"
    contradictory["assertion_revision"] = "sha256:" + "f" * 64
    contradictory["detail"].update(expected="failed", actual="failed")
    document["assertions"] = (*document["assertions"], contradictory)

    with pytest.raises(ValidationError, match="root-status observations must agree"):
        PublishedEvalTrialResult.model_validate(document)


@pytest.mark.parametrize(
    ("observed_kind", "unavailable_kind", "observation_fields"),
    [
        ("root_status", "root_status", ("actual",)),
        ("child_status", "child_status", ("matching_count",)),
        ("final_output_equals", "final_output_contains", ("matched",)),
        ("tool_called", "max_tool_calls", ("actual",)),
        ("max_model_steps", "max_model_steps", ("actual",)),
        ("usage_recorded", "max_total_tokens", ("actual",)),
        (
            "max_estimated_cost",
            "max_estimated_cost",
            ("estimated_cost", "priced_model_steps", "unpriced_model_steps"),
        ),
    ],
)
def test_published_trial_rejects_mixed_availability_within_one_evidence_area(
    observed_kind,
    unavailable_kind,
    observation_fields,
):
    document = publish_eval_run(_corpus(), _run()).cases[0].trials[0].model_dump(mode="python")
    observed = next(
        assertion
        for assertion in document["assertions"]
        if assertion["detail"]["kind"] == observed_kind
    )
    if observed_kind == unavailable_kind:
        unavailable = deepcopy(observed)
        unavailable["assertion_id"] = f"second-{unavailable_kind.replace('_', '-')}"
        unavailable["assertion_revision"] = "sha256:" + "9" * 64
        document["assertions"] = (*document["assertions"], unavailable)
    else:
        unavailable = next(
            assertion
            for assertion in document["assertions"]
            if assertion["detail"]["kind"] == unavailable_kind
        )
    unavailable.update(
        outcome="unavailable",
        score=None,
        code="unavailable",
        message="Required evidence was unavailable.",
    )
    unavailable["detail"].update({field: None for field in observation_fields})
    document.update(
        status="unavailable",
        score=None,
        code="assertion_evidence_unavailable",
        message="Required assertion evidence was unavailable.",
    )

    with pytest.raises(ValidationError, match="evidence availability.*trial area"):
        PublishedEvalTrialResult.model_validate(document)


def test_complete_published_trial_requires_root_status_observation():
    document = publish_eval_run(_corpus(), _run()).cases[0].trials[0].model_dump(mode="python")
    root = next(
        assertion
        for assertion in document["assertions"]
        if assertion["detail"]["kind"] == "root_status"
    )
    root.update(
        outcome="unavailable",
        score=None,
        code="unavailable",
        message="Required evidence was unavailable.",
    )
    root["detail"]["actual"] = None
    document.update(
        status="unavailable",
        score=None,
        code="assertion_evidence_unavailable",
        message="Required assertion evidence was unavailable.",
    )

    with pytest.raises(ValidationError, match="requires a root-status observation"):
        PublishedEvalTrialResult.model_validate(document)


def test_published_trial_allows_assertion_error_without_erasing_shared_observations():
    document = publish_eval_run(_corpus(), _run()).cases[0].trials[0].model_dump(mode="python")
    output = next(
        assertion
        for assertion in document["assertions"]
        if assertion["detail"]["kind"] == "final_output_contains"
    )
    output.update(
        outcome="error",
        score=None,
        code="error",
        message="Assertion evaluation failed.",
    )
    output["detail"]["matched"] = None
    document.update(
        status="error",
        score=None,
        code="assertion_evaluation_failed",
        message="Assertion evaluation failed.",
    )

    assert PublishedEvalTrialResult.model_validate(document).status == "error"


def test_published_trial_rejects_tool_observations_above_exact_usage():
    document = publish_eval_run(_corpus(), _run()).cases[0].trials[0].model_dump(mode="python")
    called = next(
        assertion
        for assertion in document["assertions"]
        if assertion["detail"]["kind"] == "tool_called"
    )
    called["detail"].update(matching_count=2, max_count=2)

    with pytest.raises(ValidationError, match="cannot exceed trial tool calls"):
        PublishedEvalTrialResult.model_validate(document)


def test_published_trial_rejects_distinct_tool_counts_above_exact_usage():
    document = publish_eval_run(_corpus(), _run()).cases[0].trials[0].model_dump(mode="python")
    called = next(
        assertion
        for assertion in document["assertions"]
        if assertion["detail"]["kind"] == "tool_called"
    )
    second_tool = deepcopy(called)
    second_tool["assertion_id"] = "second-tool"
    second_tool["assertion_revision"] = "sha256:" + "d" * 64
    second_tool["detail"]["tool_name"] = "search"
    document["assertions"] = (*document["assertions"], second_tool)

    with pytest.raises(ValidationError, match="cannot exceed trial tool calls"):
        PublishedEvalTrialResult.model_validate(document)


@pytest.mark.parametrize(
    ("completed_count", "failed_count", "accepted"),
    [
        (499, 1, True),
        (500, 500, False),
    ],
)
def test_published_trial_enforces_child_status_partition(
    completed_count,
    failed_count,
    accepted,
):
    document = publish_eval_run(_corpus(), _run()).cases[0].trials[0].model_dump(mode="python")
    failed = next(
        assertion
        for assertion in document["assertions"]
        if assertion["detail"]["kind"] == "child_status"
    )
    failed["detail"].update(
        min_count=failed_count,
        max_count=failed_count,
        matching_count=failed_count,
    )
    completed = deepcopy(failed)
    completed["assertion_id"] = "completed-children"
    completed["assertion_revision"] = "sha256:" + "c" * 64
    completed["detail"].update(
        expected="completed",
        min_count=completed_count,
        max_count=completed_count,
        matching_count=completed_count,
    )
    document["assertions"] = (*document["assertions"], completed)

    if accepted:
        PublishedEvalTrialResult.model_validate(document)
    else:
        with pytest.raises(ValidationError, match="cannot exceed retained child evidence"):
            PublishedEvalTrialResult.model_validate(document)


@pytest.mark.parametrize(
    ("kind", "change", "match"),
    [
        (
            "child_status",
            {"matching_count": 2, "max_count": 2},
            "child-status observations for the same status must agree",
        ),
        (
            "tool_called",
            {"matching_count": 0, "min_count": 0, "max_count": 0},
            "tool-call observations for the same tool must agree",
        ),
        (
            "tools_called_in_order",
            {"expected_count": 2, "actual_count": 2, "matched": True},
            "tool-order observations must agree",
        ),
        (
            "max_estimated_cost",
            {"estimated_cost": "0.000026"},
            "cost observations for the same currency must agree",
        ),
    ],
)
def test_published_trial_rejects_conflicting_repeated_observations(kind, change, match):
    document = publish_eval_run(_corpus(), _run()).cases[0].trials[0].model_dump(mode="python")
    original = next(
        assertion for assertion in document["assertions"] if assertion["detail"]["kind"] == kind
    )
    contradictory = deepcopy(original)
    contradictory["assertion_id"] = f"second-{kind.replace('_', '-')}"
    contradictory["assertion_revision"] = "sha256:" + "e" * 64
    contradictory["detail"].update(change)
    document["assertions"] = (*document["assertions"], contradictory)

    with pytest.raises(ValidationError, match=match):
        PublishedEvalTrialResult.model_validate(document)


def test_incomplete_published_trial_metrics_still_require_usage():
    document = publish_eval_run(_corpus(), _run()).model_dump(mode="python")
    trial = document["cases"][0]["trials"][1]
    trial["evidence_complete"] = False
    trial["usage"] = None

    with pytest.raises(ValidationError, match="observations require trial usage"):
        PublishedEvalRun.model_validate(document)


def test_projection_rejects_results_from_a_different_assertion_contract():
    corpus = _corpus()
    case = corpus.cases[0]
    changed_assertions = list(case.assertions)
    changed_assertions[2] = FinalOutputEqualsAssertionSpec(
        id="equals",
        expected="Definitely not approved",
    )
    changed_case = EvalCaseSpec.create(
        id=case.id,
        suite_id=case.suite_id,
        name=case.name,
        description=case.description,
        source=case.source,
        input=case.input,
        assertions=changed_assertions,
    )
    changed_corpus = EvalCorpusDocument.create(
        target_key=corpus.target_key,
        evidence_policy=corpus.evidence_policy,
        pricing_profile=corpus.pricing_profile,
        suites=corpus.suites,
        cases=(changed_case,),
    )

    with pytest.raises(ValueError, match="result revision does not match"):
        publish_eval_run(changed_corpus, _run(corpus=changed_corpus))


def test_projection_requires_the_exact_contract_fixed_before_execution():
    corpus = _corpus()
    run = _run(corpus=corpus)
    unbound = EvalRun.model_validate({**run.model_dump(mode="python"), "run_contract": None})

    with pytest.raises(ValueError, match="execution-time run contract"):
        publish_eval_run(corpus, unbound)

    case = corpus.cases[0]
    changed_case = EvalCaseSpec.create(
        id=case.id,
        suite_id=case.suite_id,
        name=case.name,
        description=case.description,
        source=case.source,
        input=RunInputSpec(messages=(CorpusUserMessageSpec(text="A different request."),)),
        assertions=case.assertions,
    )
    changed_input_corpus = EvalCorpusDocument.create(
        target_key=corpus.target_key,
        evidence_policy=corpus.evidence_policy,
        pricing_profile=corpus.pricing_profile,
        suites=corpus.suites,
        cases=(changed_case,),
    )
    changed_pricing_corpus = EvalCorpusDocument.create(
        target_key=corpus.target_key,
        evidence_policy=corpus.evidence_policy,
        pricing_profile=PricingProfileIdentityV1(
            pricing_semantics_version=1,
            fingerprint="sha256:" + "d" * 64,
            price_book_version="v2",
            generated_at="2026-08-05T00:00:00Z",
            currencies=("USD",),
        ),
        suites=corpus.suites,
        cases=corpus.cases,
    )

    for changed_corpus in (changed_input_corpus, changed_pricing_corpus):
        with pytest.raises(ValueError, match="does not match the corpus contract"):
            publish_eval_run(changed_corpus, run)


def test_projection_checks_safe_output_and_tool_order_decisions():
    corpus = _corpus()
    run = _run(corpus=corpus)
    first_trial = run.cases[0].trials[0]

    output_result = first_trial.assertions[2].model_copy(update={"metadata": {"matched": False}})
    wrong_output_trial = first_trial.model_copy(
        update={
            "assertions": (
                *first_trial.assertions[:2],
                output_result,
                *first_trial.assertions[3:],
            )
        }
    )
    wrong_output_case = EvalCaseResult.from_trials(
        case_id="case",
        trials=(wrong_output_trial, run.cases[0].trials[1]),
        started_at=run.cases[0].started_at,
        completed_at=run.cases[0].completed_at,
    )
    wrong_output_run = EvalRun(
        suite_id="suite",
        status=wrong_output_case.status,
        score=wrong_output_case.score,
        cases=(wrong_output_case,),
        started_at=run.started_at,
        completed_at=run.completed_at,
        duration_ms=run.duration_ms,
        run_contract=run.run_contract,
    )
    with pytest.raises(ValidationError, match="contradicts"):
        publish_eval_run(corpus, wrong_output_run)

    order_result = first_trial.assertions[5].model_copy(
        update={"metadata": {"actual": ["different-tool"]}}
    )
    wrong_order_trial = first_trial.model_copy(
        update={
            "assertions": (
                *first_trial.assertions[:5],
                order_result,
                *first_trial.assertions[6:],
            )
        }
    )
    wrong_order_case = EvalCaseResult.from_trials(
        case_id="case",
        trials=(wrong_order_trial, run.cases[0].trials[1]),
        started_at=run.cases[0].started_at,
        completed_at=run.cases[0].completed_at,
    )
    wrong_order_run = EvalRun(
        suite_id="suite",
        status=wrong_order_case.status,
        score=wrong_order_case.score,
        cases=(wrong_order_case,),
        started_at=run.started_at,
        completed_at=run.completed_at,
        duration_ms=run.duration_ms,
        run_contract=run.run_contract,
    )
    with pytest.raises(ValidationError, match="contradicts"):
        publish_eval_run(corpus, wrong_order_run)


def test_projection_uses_redacted_final_output_decision_without_copying_raw_output():
    base_corpus = _corpus()
    base_case = base_corpus.cases[0]
    redacted_equals = FinalOutputEqualsAssertionSpec(
        id="equals",
        expected="Approved [REDACTED_SECRET]",
    )
    changed_assertions = (
        *base_case.assertions[:2],
        redacted_equals,
        *base_case.assertions[3:],
    )
    changed_case = EvalCaseSpec.create(
        id=base_case.id,
        suite_id=base_case.suite_id,
        name=base_case.name,
        description=base_case.description,
        source=base_case.source,
        input=base_case.input,
        assertions=changed_assertions,
    )
    corpus = EvalCorpusDocument.create(
        target_key=base_corpus.target_key,
        evidence_policy=base_corpus.evidence_policy,
        pricing_profile=base_corpus.pricing_profile,
        suites=base_corpus.suites,
        cases=(changed_case,),
    )
    base_run = _run(corpus=corpus)
    trials = []
    for trial in base_run.cases[0].trials:
        redacted_result = trial.assertions[2].model_copy(
            update={"assertion_revision": assertion_spec_revision(redacted_equals)}
        )
        trials.append(
            trial.model_copy(
                update={
                    "final_output": "Approved secret-token",
                    "assertions": (
                        *trial.assertions[:2],
                        redacted_result,
                        *trial.assertions[3:],
                    ),
                }
            )
        )
    case = EvalCaseResult.from_trials(
        case_id="case",
        trials=trials,
        started_at=base_run.cases[0].started_at,
        completed_at=base_run.cases[0].completed_at,
    )
    run = EvalRun(
        suite_id="suite",
        status=case.status,
        score=case.score,
        cases=(case,),
        started_at=base_run.started_at,
        completed_at=base_run.completed_at,
        duration_ms=base_run.duration_ms,
        run_contract=base_run.run_contract,
    )

    published = publish_eval_run(corpus, run)

    assert published.cases[0].trials[0].assertions[2].detail.matched is True
    assert "secret-token" not in published.model_dump_json()


def test_public_eval_trial_contract_has_no_publication_side_channel():
    trial = _run().cases[0].trials[0]
    document = trial.model_dump(mode="python")
    document["public_data"] = {"output": "caller-secret-token"}

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EvalTrialResult.model_validate(document)

    assert "public_data" not in EvalTrialResult.model_json_schema(mode="validation")["properties"]
    assert (
        "public_data" not in EvalTrialResult.model_json_schema(mode="serialization")["properties"]
    )


@pytest.mark.parametrize("estimated_cost", ["secret-token", "1e999999999"])
def test_scored_cost_detail_rejects_incomplete_or_noncanonical_metadata(estimated_cost):
    run = _run()
    cost = (
        run.cases[0]
        .trials[0]
        .assertions[-1]
        .model_copy(
            update={
                "metadata": {
                    "estimated_cost": estimated_cost,
                    "priced_model_steps": 1,
                }
            }
        )
    )
    trial = (
        run.cases[0]
        .trials[0]
        .model_copy(update={"assertions": (*run.cases[0].trials[0].assertions[:-1], cost)})
    )
    case = EvalCaseResult.from_trials(
        case_id="case",
        trials=(trial, run.cases[0].trials[1]),
        started_at=run.cases[0].started_at,
        completed_at=run.cases[0].completed_at,
    )
    with pytest.raises(ValidationError, match="observed evidence"):
        publish_eval_run(
            _corpus(),
            EvalRun(
                suite_id="suite",
                status=case.status,
                score=case.score,
                cases=(case,),
                started_at=run.started_at,
                completed_at=run.completed_at,
                duration_ms=run.duration_ms,
                run_contract=run.run_contract,
            ),
        )


def test_projection_rejects_cost_metadata_that_contradicts_its_exact_summary():
    corpus = _corpus()
    run = _run(corpus=corpus)
    source_case = run.cases[0]
    source_trial = source_case.trials[0]
    source_cost = source_trial.assertions[-1]
    forged_cost = source_cost.model_copy(
        update={
            "cost_summary": SessionCostSummary(
                session_id=source_trial.session_id,
                currency="USD",
                model_steps=1,
                priced_model_steps=1,
                unpriced_model_steps=0,
                total_cost=Decimal("99"),
            )
        }
    )
    forged_trial = source_trial.model_copy(
        update={"assertions": (*source_trial.assertions[:-1], forged_cost)}
    )
    forged_case = EvalCaseResult.from_trials(
        case_id=source_case.case_id,
        trials=(forged_trial, source_case.trials[1]),
        started_at=source_case.started_at,
        completed_at=source_case.completed_at,
    )
    forged_run = EvalRun(
        suite_id=run.suite_id,
        status=forged_case.status,
        score=forged_case.score,
        cases=(forged_case,),
        started_at=run.started_at,
        completed_at=run.completed_at,
        duration_ms=run.duration_ms,
        run_contract=run.run_contract,
    )

    with pytest.raises(ValueError, match="cost metadata does not match"):
        publish_eval_run(corpus, forged_run)


@pytest.mark.parametrize("metadata_currency", [None, "EUR"])
def test_projection_requires_exact_cost_metadata_currency_without_summary(
    metadata_currency,
):
    corpus = _corpus()
    run = _run(corpus=corpus)
    source_case = run.cases[0]
    source_trial = source_case.trials[0]
    source_cost = source_trial.assertions[-1]
    metadata = dict(source_cost.metadata)
    if metadata_currency is None:
        metadata.pop("currency")
    else:
        metadata["currency"] = metadata_currency
    forged_cost = source_cost.model_copy(update={"cost_summary": None, "metadata": metadata})
    forged_trial = source_trial.model_copy(
        update={"assertions": (*source_trial.assertions[:-1], forged_cost)}
    )
    forged_case = EvalCaseResult.from_trials(
        case_id=source_case.case_id,
        trials=(forged_trial, source_case.trials[1]),
        started_at=source_case.started_at,
        completed_at=source_case.completed_at,
    )
    forged_run = EvalRun(
        suite_id=run.suite_id,
        status=forged_case.status,
        score=forged_case.score,
        cases=(forged_case,),
        started_at=run.started_at,
        completed_at=run.completed_at,
        duration_ms=run.duration_ms,
        run_contract=run.run_contract,
    )

    with pytest.raises(ValueError, match="cost metadata currency does not match"):
        publish_eval_run(corpus, forged_run)


def test_projection_handles_non_scalar_allowlisted_metadata_without_copying_it():
    run = _run()
    root = (
        run.cases[0]
        .trials[0]
        .assertions[0]
        .model_copy(update={"metadata": {"actual": ["completed", "secret-token"]}})
    )
    trial = (
        run.cases[0]
        .trials[0]
        .model_copy(update={"assertions": (root, *run.cases[0].trials[0].assertions[1:])})
    )
    case = EvalCaseResult.from_trials(
        case_id="case",
        trials=(trial, run.cases[0].trials[1]),
        started_at=run.cases[0].started_at,
        completed_at=run.cases[0].completed_at,
    )

    with pytest.raises(ValidationError, match="observed evidence"):
        publish_eval_run(
            _corpus(),
            EvalRun(
                suite_id="suite",
                status=case.status,
                score=case.score,
                cases=(case,),
                started_at=run.started_at,
                completed_at=run.completed_at,
                duration_ms=run.duration_ms,
                run_contract=run.run_contract,
            ),
        )
