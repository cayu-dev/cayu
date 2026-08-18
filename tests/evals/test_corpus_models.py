from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from cayu.evals.corpus import (
    EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE,
    EVAL_CORPUS_MAX_MERGE_INPUTS,
    EVAL_CORPUS_MAX_MESSAGE_CHARS,
    EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS,
    EVAL_CORPUS_SCHEMA_VERSION,
    EVIDENCE_MAX_TOTAL_TOKENS,
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
    ModelJudgeAssertionSpec,
    PricingProfileIdentityV1,
    RootStatusAssertionSpec,
    RunInputSpec,
    ToolCalledAssertionSpec,
    ToolsCalledInOrderAssertionSpec,
    TrialRequestSpec,
    UsageRecordedAssertionSpec,
    assertion_spec_revision,
    eval_corpus_from_json,
    eval_corpus_inspection_to_json,
    eval_corpus_to_json,
    eval_run_contract_for_corpus,
    inspect_eval_corpus,
    load_eval_corpus,
    merge_eval_corpora,
    merge_eval_corpus_files,
)


def _model_judge_assertion(
    *,
    evaluator_key: str = "quality-judge",
    rubric: str = "Score whether the answer is correct and useful.",
    rubric_version: str = "quality-v1",
    threshold: float = 0.7,
    include_transcript: bool = False,
) -> ModelJudgeAssertionSpec:
    return ModelJudgeAssertionSpec(
        id="answer-quality",
        evaluator_key=evaluator_key,
        rubric=rubric,
        rubric_version=rubric_version,
        threshold=threshold,
        include_transcript=include_transcript,
    )


def _source(*, evidence_revision: str = "sha256:" + "e" * 64):
    return EvaluationSourceIdentityV1(
        application_release_id="release-2026-08-05",
        app_manifest_schema_version="7",
        app_manifest_fingerprint="a" * 64,
        evidence_revision=evidence_revision,
    )


def _pricing() -> PricingProfileIdentityV1:
    return PricingProfileIdentityV1(
        fingerprint="sha256:" + "b" * 64,
        pricing_semantics_version=1,
        price_book_version="2026-08-05",
        generated_at="2026-08-05T00:00:00Z",
        currencies=("EUR", "USD"),
    )


def _assertions(*, include_cost: bool = False):
    assertions = (
        RootStatusAssertionSpec(id="root-completed", expected="completed"),
        ChildStatusAssertionSpec(
            id="children-completed",
            expected="completed",
            min_count=1,
            max_count=3,
        ),
        FinalOutputEqualsAssertionSpec(id="exact-answer", expected="Approved"),
        FinalOutputContainsAssertionSpec(id="mentions-approval", expected="Approve"),
        ToolCalledAssertionSpec(
            id="lookup-called",
            tool_name="lookup_invoice",
            min_count=1,
            max_count=2,
        ),
        ToolsCalledInOrderAssertionSpec(
            id="tool-order",
            tool_names=("lookup_invoice", "request_approval"),
        ),
        MaxToolCallsAssertionSpec(id="tool-budget", maximum=3),
        MaxModelStepsAssertionSpec(id="step-budget", maximum=4),
        UsageRecordedAssertionSpec(id="usage-recorded", min_total_tokens=1),
        MaxTotalTokensAssertionSpec(id="token-budget", maximum=4_000),
    )
    if include_cost:
        return (
            *assertions,
            MaxEstimatedCostAssertionSpec(
                id="cost-budget",
                maximum="0.05",
                currency="USD",
            ),
        )
    return assertions


def _suite(*, suite_id: str = "refund-regressions", name: str = "Refund regressions"):
    return EvalSuiteSpec.create(
        id=suite_id,
        name=name,
        description="Production-derived refund behavior.",
        trial_request=TrialRequestSpec(trials=3, timeout_seconds=120),
    )


def _case(
    *,
    case_id: str = "refund-approval",
    suite_id: str = "refund-regressions",
    name: str = "Refund approval",
    include_cost: bool = False,
):
    return EvalCaseSpec.create(
        id=case_id,
        suite_id=suite_id,
        name=name,
        description="A refund requiring approval.",
        source=_source(),
        input=RunInputSpec(
            messages=(
                CorpusUserMessageSpec(text="Please refund invoice 123."),
                CorpusUserMessageSpec(text="Use the standard approval path."),
            )
        ),
        assertions=_assertions(include_cost=include_cost),
    )


def _corpus(*, include_cost: bool = False) -> EvalCorpusDocument:
    return EvalCorpusDocument.create(
        target_key="refund-agent",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        pricing_profile=_pricing() if include_cost else None,
        suites=(_suite(),),
        cases=(_case(include_cost=include_cost),),
    )


def test_corpus_v1_round_trips_as_deterministic_json():
    corpus = _corpus(include_cost=True)

    first = eval_corpus_to_json(corpus)
    restored = eval_corpus_from_json(first)
    second = eval_corpus_to_json(restored)

    assert restored == corpus
    assert first == second
    assert first.endswith("\n")
    document = json.loads(first)
    assert document["schema_version"] == EVAL_CORPUS_SCHEMA_VERSION
    assert document["target_key"] == "refund-agent"
    assert document["cases"][0]["source"] == {
        "schema_version": 1,
        "application_release_id": "release-2026-08-05",
        "app_manifest_schema_version": "7",
        "app_manifest_fingerprint": "a" * 64,
        "evidence_revision": "sha256:" + "e" * 64,
    }
    assert '"session_id":' not in first
    assert '"agent_name":' not in first
    assert '"provider":' not in first
    assert '"model":' not in first


def test_every_portable_assertion_kind_round_trips_through_a_case():
    case = _case(include_cost=True)
    restored = EvalCaseSpec.model_validate(case.model_dump(mode="json"))

    assert tuple(assertion.kind for assertion in restored.assertions) == (
        "root_status",
        "child_status",
        "final_output_equals",
        "final_output_contains",
        "tool_called",
        "tools_called_in_order",
        "max_tool_calls",
        "max_model_steps",
        "usage_recorded",
        "max_total_tokens",
        "max_estimated_cost",
    )


def test_portable_model_judge_spec_round_trips_as_bounded_authority_free_data():
    spec = _model_judge_assertion()

    restored = ModelJudgeAssertionSpec.model_validate(spec.model_dump(mode="json"))

    assert restored == spec
    assert restored.kind == "model_judge"
    document = restored.model_dump(mode="json")
    assert document == {
        "id": "answer-quality",
        "description": None,
        "kind": "model_judge",
        "evaluator_key": "quality-judge",
        "rubric": "Score whether the answer is correct and useful.",
        "rubric_version": "quality-v1",
        "threshold": 0.7,
        "include_transcript": False,
    }
    assert "app" not in document
    assert "provider" not in document
    assert "credential" not in document


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("evaluator_key", "UNKNOWN KEY"),
        ("rubric", ""),
        ("rubric_version", ""),
        ("threshold", float("nan")),
        ("threshold", -0.1),
        ("threshold", 1.1),
    ),
)
def test_portable_model_judge_spec_rejects_invalid_contract_fields(field, value):
    document = _model_judge_assertion().model_dump(mode="python")
    document[field] = value

    with pytest.raises(ValidationError):
        ModelJudgeAssertionSpec.model_validate(document)


def test_portable_model_judge_revision_covers_every_evaluation_input():
    original = _model_judge_assertion()
    changes = (
        _model_judge_assertion(evaluator_key="safety-judge"),
        _model_judge_assertion(rubric="Score only factual correctness."),
        _model_judge_assertion(rubric_version="quality-v2"),
        _model_judge_assertion(threshold=0.8),
        _model_judge_assertion(include_transcript=True),
    )

    assert (
        len(
            {
                assertion_spec_revision(original),
                *(assertion_spec_revision(item) for item in changes),
            }
        )
        == 6
    )


def test_portable_token_bound_is_exact_across_ieee_754_json_boundaries():
    policy = EvaluationEvidencePolicySpec.standard()

    assert EVIDENCE_MAX_TOTAL_TOKENS == 2**53 - 1
    assert policy.max_total_tokens == EVIDENCE_MAX_TOTAL_TOKENS
    assert int(float(policy.max_total_tokens)) == policy.max_total_tokens
    with pytest.raises(ValidationError):
        MaxTotalTokensAssertionSpec(
            id="lossy-token-budget",
            maximum=EVIDENCE_MAX_TOTAL_TOKENS + 1,
        )


def test_ordered_portable_inputs_reject_unordered_python_containers():
    with pytest.raises(ValidationError, match="tool_names.*ordered array"):
        ToolsCalledInOrderAssertionSpec(
            id="tool-order",
            tool_names={"lookup_invoice", "request_approval"},
        )
    with pytest.raises(ValidationError, match="currencies.*ordered array"):
        PricingProfileIdentityV1(
            fingerprint="sha256:" + "b" * 64,
            pricing_semantics_version=1,
            price_book_version="2026-08-05",
            generated_at="2026-08-05T00:00:00Z",
            currencies={"EUR", "USD"},
        )
    with pytest.raises(ValidationError, match="messages.*ordered array"):
        RunInputSpec(messages={CorpusUserMessageSpec(text="First")})

    case = _case()
    with pytest.raises(TypeError, match="assertions must be an ordered sequence"):
        EvalCaseSpec.create(
            id=case.id,
            suite_id=case.suite_id,
            name=case.name,
            description=case.description,
            source=case.source,
            input=case.input,
            assertions=set(case.assertions),
        )

    corpus = _corpus()
    with pytest.raises(TypeError, match="suites must be an ordered sequence"):
        EvalCorpusDocument.create(
            target_key=corpus.target_key,
            evidence_policy=corpus.evidence_policy,
            suites=set(corpus.suites),
            cases=corpus.cases,
        )


def test_content_revisions_cover_semantic_and_descriptive_content():
    original_case = _case()
    renamed_case = _case(name="Renamed refund approval")
    changed_evidence_case = EvalCaseSpec.create(
        id=original_case.id,
        suite_id=original_case.suite_id,
        name=original_case.name,
        description=original_case.description,
        source=_source(evidence_revision="sha256:" + "f" * 64),
        input=original_case.input,
        assertions=original_case.assertions,
    )
    original_suite = _suite()
    renamed_suite = _suite(name="Renamed suite")

    assert original_case.revision != renamed_case.revision
    assert original_case.revision != changed_evidence_case.revision
    assert original_suite.revision != renamed_suite.revision
    assert (
        _corpus().revision
        != EvalCorpusDocument.create(
            target_key="refund-agent-v2",
            evidence_policy=EvaluationEvidencePolicySpec.standard(),
            suites=(original_suite,),
            cases=(original_case,),
        ).revision
    )


def test_run_contract_freezes_every_execution_relevant_corpus_identity():
    corpus = _corpus(include_cost=True)
    contract = eval_run_contract_for_corpus(corpus, "refund-regressions")

    assert contract.corpus_revision == corpus.revision
    assert contract.target_key == corpus.target_key
    assert contract.suite_revision == corpus.suites[0].revision
    assert contract.evidence_policy_revision == corpus.evidence_policy.revision
    assert contract.pricing_profile_fingerprint == corpus.pricing_profile.fingerprint
    assert contract.trials == corpus.suites[0].trial_request.trials
    assert contract.timeout_seconds == corpus.suites[0].trial_request.timeout_seconds
    assert [(case.case_id, case.case_revision) for case in contract.cases] == [
        (corpus.cases[0].id, corpus.cases[0].revision)
    ]
    assert eval_run_contract_for_corpus(corpus, "refund-regressions") == contract

    with pytest.raises(ValueError, match="does not contain suite"):
        eval_run_contract_for_corpus(corpus, "missing-suite")


def test_assertion_revision_covers_description_and_expectation():
    base = FinalOutputContainsAssertionSpec(id="answer", expected="approved")
    described = FinalOutputContainsAssertionSpec(
        id="answer",
        expected="approved",
        description="Must mention approval.",
    )
    changed = FinalOutputContainsAssertionSpec(id="answer", expected="declined")

    assert assertion_spec_revision(base) != assertion_spec_revision(described)
    assert assertion_spec_revision(base) != assertion_spec_revision(changed)


def test_public_factories_reject_assertion_subclasses():
    class CustomAssertion(RootStatusAssertionSpec):
        pass

    custom = CustomAssertion(id="custom", expected="completed")

    with pytest.raises(TypeError, match="exact built-in"):
        assertion_spec_revision(custom)
    with pytest.raises(TypeError, match="exact built-in"):
        EvalCaseSpec.create(
            id="case",
            suite_id="suite",
            name="Case",
            source=_source(),
            input=RunInputSpec(messages=(CorpusUserMessageSpec(text="Run."),)),
            assertions=(custom,),
        )


@pytest.mark.parametrize("model_factory", [_case, _suite, _corpus])
def test_revisioned_models_reject_forged_or_stale_revisions(model_factory):
    model = model_factory()
    document = model.model_dump(mode="python")
    document["name" if "name" in document else "target_key"] = "changed"

    with pytest.raises(ValidationError, match="revision does not match"):
        type(model).model_validate(document)

    forged = model.model_copy(update={"revision": "sha256:" + "0" * 64})
    with pytest.raises(ValidationError, match="revision does not match"):
        type(model).model_validate(forged)


def test_corpus_creation_sorts_suites_and_cases_canonically():
    suite_a = _suite(suite_id="a-suite", name="A")
    suite_b = _suite(suite_id="b-suite", name="B")
    case_a = _case(case_id="a-case", suite_id="a-suite", name="A")
    case_b = _case(case_id="b-case", suite_id="b-suite", name="B")

    corpus = EvalCorpusDocument.create(
        target_key="refund-agent",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        suites=(suite_b, suite_a),
        cases=(case_b, case_a),
    )

    assert [suite.id for suite in corpus.suites] == ["a-suite", "b-suite"]
    assert [case.id for case in corpus.cases] == ["a-case", "b-case"]


def test_direct_corpus_validation_rejects_noncanonical_order():
    suite_a = _suite(suite_id="a-suite", name="A")
    suite_b = _suite(suite_id="b-suite", name="B")
    case_a = _case(case_id="a-case", suite_id="a-suite", name="A")
    case_b = _case(case_id="b-case", suite_id="b-suite", name="B")
    canonical = EvalCorpusDocument.create(
        target_key="refund-agent",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        suites=(suite_a, suite_b),
        cases=(case_a, case_b),
    )
    document = canonical.model_dump(mode="python")
    document["suites"] = list(reversed(document["suites"]))

    with pytest.raises(ValidationError, match="suites must be sorted"):
        EvalCorpusDocument.model_validate(document)


def test_corpus_rejects_duplicate_ids_and_unknown_suite_references():
    suite = _suite()
    canonical = _corpus().model_dump(mode="python")

    duplicated_suite = {**canonical, "suites": [suite, suite]}
    duplicated_suite["revision"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError, match="suite IDs must be unique"):
        EvalCorpusDocument.model_validate(duplicated_suite)

    unknown_case = _case(suite_id="missing-suite")
    with pytest.raises(ValidationError, match="unknown suites"):
        EvalCorpusDocument.create(
            target_key="refund-agent",
            evidence_policy=EvaluationEvidencePolicySpec.standard(),
            suites=(suite,),
            cases=(unknown_case,),
        )


def test_corpus_rejects_suites_without_cases():
    populated = _suite(suite_id="populated", name="Populated")
    empty = _suite(suite_id="empty", name="Empty")

    with pytest.raises(ValidationError, match="require at least one case: empty"):
        EvalCorpusDocument.create(
            target_key="refund-agent",
            evidence_policy=EvaluationEvidencePolicySpec.standard(),
            suites=(populated, empty),
            cases=(_case(suite_id="populated"),),
        )


def test_case_rejects_empty_or_duplicate_assertions():
    common = {
        "id": "refund-approval",
        "suite_id": "refund-regressions",
        "name": "Refund approval",
        "source": _source(),
        "input": RunInputSpec(messages=(CorpusUserMessageSpec(text="Refund this."),)),
    }
    with pytest.raises(ValidationError):
        EvalCaseSpec.create(assertions=(), **common)

    duplicate = RootStatusAssertionSpec(id="same", expected="completed")
    with pytest.raises(ValidationError, match="assertion IDs must be unique"):
        EvalCaseSpec.create(assertions=(duplicate, duplicate), **common)


def test_cost_assertions_require_pricing_identity():
    with pytest.raises(ValidationError, match="Cost assertions require"):
        EvalCorpusDocument.create(
            target_key="refund-agent",
            evidence_policy=EvaluationEvidencePolicySpec.standard(),
            suites=(_suite(),),
            cases=(_case(include_cost=True),),
        )


def test_cost_assertion_currencies_must_exist_in_pricing_profile():
    unsupported_case = EvalCaseSpec.create(
        id="unsupported-currency",
        suite_id="refund-regressions",
        name="Unsupported currency",
        source=_source(),
        input=RunInputSpec(messages=(CorpusUserMessageSpec(text="Run."),)),
        assertions=(MaxEstimatedCostAssertionSpec(id="cost", maximum="1", currency="GBP"),),
    )

    with pytest.raises(ValidationError, match="absent from the pricing profile: GBP"):
        EvalCorpusDocument.create(
            target_key="refund-agent",
            evidence_policy=EvaluationEvidencePolicySpec.standard(),
            pricing_profile=_pricing(),
            suites=(_suite(),),
            cases=(unsupported_case,),
        )

    document = {
        "schema_version": EVAL_CORPUS_SCHEMA_VERSION,
        "revision": "sha256:" + "0" * 64,
        "target_key": "refund-agent",
        "evidence_policy": EvaluationEvidencePolicySpec.standard().model_dump(mode="json"),
        "pricing_profile": _pricing().model_dump(mode="json"),
        "suites": [_suite().model_dump(mode="json")],
        "cases": [unsupported_case.model_dump(mode="json")],
    }
    with pytest.raises(ValidationError, match="absent from the pricing profile: GBP"):
        EvalCorpusDocument.model_validate(document)
    with pytest.raises(ValidationError, match="absent from the pricing profile: GBP"):
        eval_corpus_from_json(json.dumps(document))


def test_pricing_profile_currency_bound_limits_distinct_cost_currencies():
    supported_currencies = tuple(f"C{index:02d}" for index in range(32))
    pricing_document = _pricing().model_dump(mode="python")
    pricing_document["currencies"] = supported_currencies
    pricing = PricingProfileIdentityV1.model_validate(pricing_document)
    assertions = tuple(
        MaxEstimatedCostAssertionSpec(id=f"cost-{index}", maximum="1", currency=f"C{index:02d}")
        for index in range(33)
    )
    case = EvalCaseSpec.create(
        id="too-many-currencies",
        suite_id="refund-regressions",
        name="Too many currencies",
        source=_source(),
        input=RunInputSpec(messages=(CorpusUserMessageSpec(text="Run."),)),
        assertions=assertions,
    )

    with pytest.raises(ValidationError, match="absent from the pricing profile: C32"):
        EvalCorpusDocument.create(
            target_key="refund-agent",
            evidence_policy=EvaluationEvidencePolicySpec.standard(),
            pricing_profile=pricing,
            suites=(_suite(),),
            cases=(case,),
        )


def test_corpus_rejects_suites_that_cannot_fit_the_complete_published_graph():
    suite = EvalSuiteSpec.create(
        id="refund-regressions",
        name="Refund regressions",
        trial_request=TrialRequestSpec(trials=100, timeout_seconds=120),
    )

    def cases(count: int):
        return tuple(
            _case(
                case_id=f"case-{index:03d}",
                name=f"Case {index}",
            )
            for index in range(count)
        )

    accepted = EvalCorpusDocument.create(
        target_key="refund-agent",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        suites=(suite,),
        cases=cases(10),
    )

    assert (
        sum(len(case.assertions) for case in accepted.cases) * suite.trial_request.trials
        == EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS
    )
    with pytest.raises(ValidationError, match="expands to 11000 published assertion results"):
        EvalCorpusDocument.create(
            target_key="refund-agent",
            evidence_policy=EvaluationEvidencePolicySpec.standard(),
            suites=(suite,),
            cases=cases(11),
        )


def test_corpus_expanded_result_limit_applies_independently_to_each_suite():
    suites = tuple(
        EvalSuiteSpec.create(
            id=f"suite-{index}",
            name=f"Suite {index}",
            trial_request=TrialRequestSpec(trials=100, timeout_seconds=120),
        )
        for index in range(2)
    )
    cases = tuple(
        EvalCaseSpec.create(
            id=f"case-{index}",
            suite_id=suite.id,
            name=f"Case {index}",
            source=_source(),
            input=RunInputSpec(messages=(CorpusUserMessageSpec(text="Run."),)),
            assertions=tuple(
                RootStatusAssertionSpec(id=f"root-{item}", expected="completed")
                for item in range(60)
            ),
        )
        for index, suite in enumerate(suites)
    )

    corpus = EvalCorpusDocument.create(
        target_key="refund-agent",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        suites=suites,
        cases=cases,
    )

    assert inspect_eval_corpus(corpus).expanded_assertion_result_count == 12_000


def test_corpus_inspection_is_stable_bounded_and_defensively_revalidated():
    corpus = _corpus(include_cost=True)

    inspection = inspect_eval_corpus(corpus)

    assert inspection.revision == corpus.revision
    assert inspection.target_key == corpus.target_key
    assert inspection.suite_count == 1
    assert inspection.case_count == 1
    assert inspection.assertion_count == 11
    assert inspection.expanded_assertion_result_count == 33

    with pytest.raises(ValidationError, match="assertion_count is impossible"):
        type(inspection.suites[0])(
            **{
                **inspection.suites[0].model_dump(mode="python"),
                "assertion_count": (
                    inspection.suites[0].case_count * EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE + 1
                ),
            }
        )
    with pytest.raises(ValidationError, match="expanded result count is impossible"):
        type(inspection.suites[0])(
            **{
                **inspection.suites[0].model_dump(mode="python"),
                "case_count": 100,
                "assertion_count": 101,
                "trials": 100,
            }
        )
    assert inspection.suites[0].trials == 3
    assert json.loads(eval_corpus_inspection_to_json(inspection))["revision"] == corpus.revision

    forged = corpus.model_copy(update={"target_key": "forged-target"})
    with pytest.raises(ValidationError, match="revision does not match"):
        inspect_eval_corpus(forged)


def test_merge_corpora_deduplicates_equal_content_and_sorts_new_suites():
    first = _corpus()
    second_suite = _suite(suite_id="account-regressions", name="Account regressions")
    second_case = _case(
        case_id="account-review",
        suite_id=second_suite.id,
        name="Account review",
    )
    second = EvalCorpusDocument.create(
        target_key=first.target_key,
        evidence_policy=first.evidence_policy,
        suites=(second_suite,),
        cases=(second_case,),
    )

    merged = merge_eval_corpora((first, first, second))

    assert tuple(suite.id for suite in merged.suites) == (
        "account-regressions",
        "refund-regressions",
    )
    assert tuple(case.id for case in merged.cases) == (
        "account-review",
        "refund-approval",
    )
    assert merge_eval_corpora((merged, first)) == merged


def test_merge_corpora_rejects_identity_and_revision_conflicts_unless_replaced():
    original = _corpus()
    replacement_suite = _suite(name="Updated refund regressions")
    replacement = EvalCorpusDocument.create(
        target_key=original.target_key,
        evidence_policy=original.evidence_policy,
        suites=(replacement_suite,),
        cases=original.cases,
    )

    with pytest.raises(ValueError, match="suite 'refund-regressions' has conflicting"):
        merge_eval_corpora((original, replacement))
    merged = merge_eval_corpora((original, replacement), replace_conflicts=True)
    assert merged.suites == (replacement_suite,)

    wrong_target = EvalCorpusDocument.create(
        target_key="other-target",
        evidence_policy=original.evidence_policy,
        suites=original.suites,
        cases=original.cases,
    )
    with pytest.raises(ValueError, match="different target keys"):
        merge_eval_corpora((original, wrong_target))


def test_merge_corpora_rejects_unbounded_iterables_without_consuming_them():
    consumed = 0

    def corpora():
        nonlocal consumed
        while True:
            consumed += 1
            yield _corpus()

    with pytest.raises(TypeError, match="ordered sequence"):
        merge_eval_corpora(corpora())

    assert consumed == 0

    class AdversarialList(list):
        def __iter__(self):
            nonlocal consumed
            consumed += 1
            return super().__iter__()

    with pytest.raises(TypeError, match="ordered sequence"):
        merge_eval_corpora(AdversarialList([_corpus()]))

    assert consumed == 0


def test_merge_corpora_rejects_the_retained_graph_before_reading_another_input(monkeypatch):
    import cayu.evals.corpus as corpus_module

    first = _corpus()
    second_suite = _suite(suite_id="account-regressions", name="Account regressions")
    second = EvalCorpusDocument.create(
        target_key=first.target_key,
        evidence_policy=first.evidence_policy,
        suites=(second_suite,),
        cases=(
            _case(
                case_id="account-review",
                suite_id=second_suite.id,
                name="Account review",
            ),
        ),
    )
    individual_limit = max(
        len(eval_corpus_to_json(first).encode("utf-8")),
        len(eval_corpus_to_json(second).encode("utf-8")),
    )
    monkeypatch.setattr(corpus_module, "EVAL_CORPUS_MAX_BYTES", individual_limit)

    with pytest.raises(ValueError, match="Merged eval corpus exceeds .* canonical JSON bytes"):
        merge_eval_corpora((first, second, object()))


def test_merge_corpora_rejects_the_retained_count_before_reading_another_input(monkeypatch):
    import cayu.evals.corpus as corpus_module

    first = _corpus()
    second = EvalCorpusDocument.create(
        target_key=first.target_key,
        evidence_policy=first.evidence_policy,
        suites=first.suites,
        cases=(_case(case_id="account-review", name="Account review"),),
    )
    monkeypatch.setattr(corpus_module, "EVAL_CORPUS_MAX_CASES", 1)

    with pytest.raises(ValueError, match="cannot contain more than 1 case"):
        merge_eval_corpora((first, second, object()))


def test_merge_corpora_reclaims_the_retained_byte_budget_on_replacement(monkeypatch):
    import cayu.evals.corpus as corpus_module

    original = _corpus()
    replacement_case = _case(name="R")
    replacement = EvalCorpusDocument.create(
        target_key=original.target_key,
        evidence_policy=original.evidence_policy,
        suites=original.suites,
        cases=(replacement_case,),
    )
    limit = max(
        len(eval_corpus_to_json(original).encode("utf-8")),
        len(eval_corpus_to_json(replacement).encode("utf-8")),
    )
    monkeypatch.setattr(corpus_module, "EVAL_CORPUS_MAX_BYTES", limit)

    assert merge_eval_corpora(
        (original, replacement),
        replace_conflicts=True,
    ).cases == (replacement_case,)


def test_file_merge_is_atomic_and_preserves_destination_on_replace_failure(tmp_path, monkeypatch):
    import cayu.evals.corpus as corpus_module

    destination = tmp_path / "evals.json"
    incoming = tmp_path / "incoming.json"
    original = _corpus()
    destination.write_text(eval_corpus_to_json(original), encoding="utf-8")
    second_suite = _suite(suite_id="account-regressions", name="Account regressions")
    second = EvalCorpusDocument.create(
        target_key=original.target_key,
        evidence_policy=original.evidence_policy,
        suites=(second_suite,),
        cases=(
            _case(
                case_id="account-review",
                suite_id=second_suite.id,
                name="Account review",
            ),
        ),
    )
    incoming.write_text(eval_corpus_to_json(second), encoding="utf-8")
    before = destination.read_bytes()

    def fail_replace(source, target):
        raise OSError("replace failed")

    monkeypatch.setattr(corpus_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        merge_eval_corpus_files(destination, (incoming,))

    assert destination.read_bytes() == before
    assert not tuple(tmp_path.glob(".evals.json.*.tmp"))


def test_file_merge_validates_before_atomically_replacing_destination(tmp_path):
    destination = tmp_path / "evals.json"
    incoming = tmp_path / "incoming.json"
    original = _corpus()
    incoming.write_text(eval_corpus_to_json(original), encoding="utf-8")

    merged = merge_eval_corpus_files(destination, (incoming,))

    assert load_eval_corpus(destination) == merged == original


def test_file_merge_rejects_symbolic_link_destination(tmp_path):
    target = tmp_path / "target.json"
    destination = tmp_path / "evals.json"
    incoming = tmp_path / "incoming.json"
    corpus = _corpus()
    target.write_text(eval_corpus_to_json(corpus), encoding="utf-8")
    destination.symlink_to(target)
    incoming.write_text(eval_corpus_to_json(corpus), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot be a symbolic link"):
        merge_eval_corpus_files(destination, (incoming,))

    assert destination.is_symlink()
    assert load_eval_corpus(target) == corpus


def test_file_merge_does_not_report_failure_after_atomic_replace_succeeds(
    tmp_path,
    monkeypatch,
):
    import cayu.evals.corpus as corpus_module

    destination = tmp_path / "evals.json"
    incoming = tmp_path / "incoming.json"
    corpus = _corpus()
    incoming.write_text(eval_corpus_to_json(corpus), encoding="utf-8")
    original_open = corpus_module.os.open

    def reject_directory_fsync_open(path, flags, *args, **kwargs):
        if Path(path) == tmp_path and flags == corpus_module.os.O_RDONLY:
            raise OSError("directory fsync unsupported")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(corpus_module.os, "open", reject_directory_fsync_open)

    merged = merge_eval_corpus_files(destination, (incoming,))

    assert merged == corpus
    assert load_eval_corpus(destination) == corpus


def test_file_merge_counts_existing_destination_before_reading_inputs(tmp_path):
    destination = tmp_path / "evals.json"
    destination.write_text(eval_corpus_to_json(_corpus()), encoding="utf-8")
    missing = tmp_path / "missing.json"

    with pytest.raises(ValueError, match="cannot total more than"):
        merge_eval_corpus_files(
            destination,
            (missing,) * EVAL_CORPUS_MAX_MERGE_INPUTS,
        )


def test_file_merge_rejects_unbounded_iterables_without_consuming_them(tmp_path):
    consumed = 0

    def inputs():
        nonlocal consumed
        while True:
            consumed += 1
            yield tmp_path / "missing.json"

    with pytest.raises(TypeError, match="ordered sequence"):
        merge_eval_corpus_files(tmp_path / "evals.json", inputs())

    assert consumed == 0

    class AdversarialList(list):
        def __iter__(self):
            nonlocal consumed
            consumed += 1
            return super().__iter__()

    with pytest.raises(TypeError, match="ordered sequence"):
        merge_eval_corpus_files(
            tmp_path / "evals.json",
            AdversarialList([tmp_path / "missing.json"]),
        )

    assert consumed == 0


def test_merge_entry_points_require_an_exact_conflict_policy(tmp_path):
    corpus = _corpus()
    source = tmp_path / "source.json"
    source.write_text(eval_corpus_to_json(corpus), encoding="utf-8")

    with pytest.raises(TypeError, match="replace_conflicts must be a bool"):
        merge_eval_corpora((corpus,), replace_conflicts=1)
    with pytest.raises(TypeError, match="replace_conflicts must be a bool"):
        merge_eval_corpus_files(
            tmp_path / "destination.json",
            (source,),
            replace_conflicts=1,
        )

    assert not (tmp_path / "destination.json").exists()


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: TrialRequestSpec(trials=True), "valid integer"),
        (lambda: TrialRequestSpec(timeout_seconds=0), "greater than or equal"),
        (
            lambda: MaxEstimatedCostAssertionSpec(
                id="cost",
                maximum="5e-2",
                currency="USD",
            ),
            "canonical decimal",
        ),
        (
            lambda: MaxEstimatedCostAssertionSpec(
                id="cost",
                maximum="1e999999999",
                currency="USD",
            ),
            "canonical decimal",
        ),
        (
            lambda: PricingProfileIdentityV1(
                fingerprint="sha256:" + "b" * 64,
                pricing_semantics_version=1,
                price_book_version="v1",
                generated_at="now",
                currencies=("USD", "EUR"),
            ),
            "unique and sorted",
        ),
        (
            lambda: RunInputSpec(
                messages=(CorpusUserMessageSpec(text="x" * (EVAL_CORPUS_MAX_MESSAGE_CHARS + 1)),)
            ),
            "at most",
        ),
    ],
)
def test_portable_models_reject_noncanonical_or_oversized_values(factory, match):
    with pytest.raises(ValidationError, match=match):
        factory()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document["cases"][0].update({"session_id": "sess_private"}),
        lambda document: document["cases"][0].update({"agent_name": "refund-agent"}),
        lambda document: document["cases"][0].update({"callback": "module:function"}),
        lambda document: document["cases"][0]["input"]["messages"][0].update({"role": "system"}),
        lambda document: document["cases"][0]["assertions"][0].update({"kind": "python_callback"}),
    ],
)
def test_corpus_rejects_runtime_authority_unknown_roles_and_assertion_kinds(mutation):
    document = _corpus().model_dump(mode="python")
    mutation(document)

    with pytest.raises(ValidationError):
        EvalCorpusDocument.model_validate(document)


def test_json_loader_rejects_duplicate_keys_unknown_versions_and_nonportable_numbers():
    with pytest.raises(ValueError, match="duplicate"):
        eval_corpus_from_json('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="unsupported schema_version"):
        eval_corpus_from_json('{"schema_version":2}')
    with pytest.raises(ValueError, match="finite"):
        eval_corpus_from_json('{"schema_version":1,"value":NaN}')
    with pytest.raises(ValueError, match="signed 64-bit"):
        eval_corpus_from_json('{"schema_version":1,"value":9223372036854775808}')


@pytest.mark.parametrize("schema_version", [True, 1.0, "1"])
def test_nested_schema_versions_do_not_coerce_from_other_json_types(schema_version):
    document = _corpus().model_dump(mode="python")
    document["cases"][0]["source"]["schema_version"] = schema_version

    with pytest.raises(ValidationError, match="schema_version must be the integer 1"):
        EvalCorpusDocument.model_validate(document)


def test_evidence_omission_flags_do_not_coerce_from_numbers():
    document = EvaluationEvidencePolicySpec.standard().model_dump(mode="python")
    document["include_event_payloads"] = 0

    with pytest.raises(ValidationError, match="include_event_payloads must be false"):
        EvaluationEvidencePolicySpec.model_validate(document)


def test_file_loader_preflights_bytes_and_utf8(monkeypatch, tmp_path):
    import cayu.evals.corpus as corpus_module

    monkeypatch.setattr(corpus_module, "EVAL_CORPUS_MAX_BYTES", 32)
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"x" * 33)
    invalid_utf8 = tmp_path / "invalid.json"
    invalid_utf8.write_bytes(b"{\xff}")

    with pytest.raises(ValueError, match="exceeds 32 bytes"):
        load_eval_corpus(oversized)
    with pytest.raises(ValueError, match="must be UTF-8"):
        load_eval_corpus(invalid_utf8)


def test_text_loader_rejects_character_overflow_before_utf8_encoding(monkeypatch):
    import cayu.evals.corpus as corpus_module

    monkeypatch.setattr(corpus_module, "EVAL_CORPUS_MAX_BYTES", 1)

    # A surrogate would fail UTF-8 encoding. The declared size boundary must
    # win first so oversized direct text never reaches that allocation/scan.
    with pytest.raises(ValueError, match="exceeds 1 bytes"):
        eval_corpus_from_json("\ud800\ud800")


def test_bounded_text_rejects_overflow_before_durable_text_scan(monkeypatch):
    import cayu.evals.corpus as corpus_module

    def unexpected_scan(*args, **kwargs):
        raise AssertionError("oversized text reached durable validation")

    monkeypatch.setattr(corpus_module, "require_durable_text", unexpected_scan)

    with pytest.raises(ValueError, match="at most 65536 characters"):
        FinalOutputEqualsAssertionSpec(id="output", expected="x" * 65_537)


def test_export_rejects_size_before_building_an_oversized_json_string(monkeypatch):
    import cayu.evals.corpus as corpus_module

    corpus = _corpus()
    monkeypatch.setattr(corpus_module, "EVAL_CORPUS_MAX_BYTES", 32)

    with pytest.raises(ValidationError, match="exceeds 32 canonical JSON bytes"):
        eval_corpus_to_json(corpus)


def test_model_limit_covers_the_pretty_serialized_form(monkeypatch):
    import cayu.evals.corpus as corpus_module

    corpus = _corpus()
    compact_bytes = len(
        json.dumps(
            corpus.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    monkeypatch.setattr(corpus_module, "EVAL_CORPUS_MAX_BYTES", compact_bytes)

    with pytest.raises(ValidationError, match="serialized JSON bytes"):
        eval_corpus_to_json(corpus)


def test_json_loader_rejects_extreme_nesting_with_a_bounded_error():
    source = '{"schema_version":1,"value":' + "[" * 2_000 + "]" * 2_000 + "}"

    with pytest.raises(ValueError, match="JSON nesting"):
        eval_corpus_from_json(source)


def test_models_are_immutable():
    corpus = _corpus()

    with pytest.raises(ValidationError, match="frozen"):
        corpus.target_key = "changed"
    with pytest.raises(ValidationError, match="frozen"):
        corpus.cases[0].name = "changed"
