from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from cayu.evals.corpus import (
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
    PricingProfileIdentityV1,
    RootStatusAssertionSpec,
    RunInputSpec,
    ToolCalledAssertionSpec,
    ToolsCalledInOrderAssertionSpec,
    TrialRequestSpec,
    UsageRecordedAssertionSpec,
    assertion_spec_revision,
    eval_corpus_from_json,
    eval_corpus_to_json,
    eval_run_contract_for_corpus,
    load_eval_corpus,
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
