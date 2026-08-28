from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from cayu.evals.corpus import (
    CorpusUserMessageSpec,
    FinalOutputContainsAssertionSpec,
    RootStatusAssertionSpec,
    RunInputSpec,
    StructuredRubricCriterionV1,
    TrialRequestSpec,
)
from cayu.evals.suite_authoring import (
    EvalCaseDraftV1,
    EvalCaseDraftV2,
    EvalScenarioStimulusV1,
    EvalSimpleInputStimulusV1,
    EvalSuiteDocumentV2,
    EvalSuiteDraftV1,
    EvalSuiteDraftV2,
    PublicJudgeReferenceDraftV1,
    StructuredModelJudgeAssertionDraftV1,
    StructuredRubricDraftV1,
    add_eval_case,
    compile_eval_suite_draft,
    compile_eval_suite_draft_v2,
    duplicate_eval_case,
    eval_suite_document_from_json,
    eval_suite_document_to_json,
    eval_suite_selection,
    revise_eval_case,
    validate_eval_suite_selection,
    validate_expected_eval_suite_revision,
)


def _revision(character: str) -> str:
    return "sha256:" + character * 64


def _case(
    case_id: str = "refund-request",
    *,
    prompt: str = "Refund invoice 123.",
) -> EvalCaseDraftV1:
    return EvalCaseDraftV1(
        id=case_id,
        name="Refund request",
        stimulus=EvalSimpleInputStimulusV1(
            input=RunInputSpec(messages=(CorpusUserMessageSpec(text=prompt),))
        ),
        assertions=(
            RootStatusAssertionSpec(id="completed", expected="completed"),
            FinalOutputContainsAssertionSpec(id="mentions-refund", expected="refund"),
        ),
    )


def _draft(*cases: EvalCaseDraftV1) -> EvalSuiteDraftV1:
    return EvalSuiteDraftV1(
        id="refund-regressions",
        target_key="assistant.default",
        name="Refund regressions",
        description="Reusable authored behavior.",
        trial_request=TrialRequestSpec(trials=1, timeout_seconds=120),
        cases=cases or (_case(),),
    )


def test_suite_draft_compiles_to_canonical_immutable_revisions() -> None:
    draft = _draft(_case("second-case"), _case("first-case"))

    first = compile_eval_suite_draft(draft)
    second = compile_eval_suite_draft(draft)

    assert first == second
    assert [case.id for case in first.cases] == ["first-case", "second-case"]
    assert first.revision.startswith("sha256:")
    assert all(case.revision.startswith("sha256:") for case in first.cases)
    assert first.suite.revision.startswith("sha256:")
    assert first.cases[0].source is None

    restored = eval_suite_document_from_json(eval_suite_document_to_json(first))
    assert restored == first


def test_v2_structured_judge_draft_compiles_server_owned_nested_revisions_and_round_trips() -> None:
    assertion = StructuredModelJudgeAssertionDraftV1(
        id="quality",
        judge_profile_key="quality-judge",
        judge_profile_revision=_revision("a"),
        rubric=StructuredRubricDraftV1(
            id="answer-quality",
            criteria=(
                StructuredRubricCriterionV1(
                    id="correctness",
                    name="Correctness",
                    description="The answer is correct.",
                    weight="0.4",
                ),
                StructuredRubricCriterionV1(
                    id="usefulness",
                    name="Usefulness",
                    description="The answer helps the user.",
                    weight="0.6",
                ),
            ),
        ),
        reference=PublicJudgeReferenceDraftV1(
            id="refund-policy",
            expected_answer="Refunds are available within 30 days.",
            expected_facts=("The refund window is 30 days.",),
        ),
        threshold="0.7",
    )
    draft = EvalSuiteDraftV2(
        id="refund-quality",
        target_key="assistant.default",
        name="Refund quality",
        cases=(
            EvalCaseDraftV2(
                id="refund-request",
                name="Refund request",
                stimulus=EvalSimpleInputStimulusV1(
                    input=RunInputSpec(
                        messages=(CorpusUserMessageSpec(text="Can I get a refund?"),)
                    )
                ),
                assertions=(assertion,),
            ),
        ),
    )

    document = compile_eval_suite_draft_v2(draft)
    compiled = document.cases[0].assertions[0]

    assert document.schema_version == 2
    assert compiled.kind == "structured_model_judge"
    assert compiled.rubric.revision.startswith("sha256:")
    assert compiled.reference is not None
    assert compiled.reference.revision.startswith("sha256:")
    assert eval_suite_document_from_json(eval_suite_document_to_json(document)) == document

    editable = EvalSuiteDraftV2.from_document(document)
    editable_assertion = editable.cases[0].assertions[0]
    assert type(editable_assertion) is StructuredModelJudgeAssertionDraftV1
    assert "revision" not in editable_assertion.rubric.model_dump(mode="json")
    assert editable_assertion.reference is not None
    assert "revision" not in editable_assertion.reference.model_dump(mode="json")
    assert compile_eval_suite_draft_v2(editable) == EvalSuiteDocumentV2.model_validate(document)


def test_simple_and_exact_scenario_stimuli_are_distinct() -> None:
    scenario_case = EvalCaseDraftV1(
        id="approval-flow",
        name="Approval flow",
        stimulus=EvalScenarioStimulusV1(
            scenario_id="approval-flow",
            scenario_revision="sha256:" + "a" * 64,
        ),
        assertions=(RootStatusAssertionSpec(id="completed", expected="completed"),),
    )

    document = compile_eval_suite_draft(_draft(_case(), scenario_case))

    assert document.cases[0].stimulus.kind == "scenario"
    assert document.cases[1].stimulus.kind == "simple_input"


def test_add_duplicate_and_revise_create_new_history_without_mutating_prior_revisions() -> None:
    original = compile_eval_suite_draft(_draft())
    added = add_eval_case(original, _case("refund-follow-up"))
    duplicated = duplicate_eval_case(
        added,
        "refund-request",
        new_case_id="refund-copy",
        new_name="Refund request copy",
    )
    current = next(case for case in duplicated.cases if case.id == "refund-request")
    replacement = _case("refund-request", prompt="Refund invoice 456.")
    revised = revise_eval_case(
        duplicated,
        replacement,
        expected_case_revision=current.revision,
    )

    assert len(original.cases) == 1
    assert len(added.cases) == 2
    assert len(duplicated.cases) == 3
    assert len(revised.cases) == 3
    assert len({original.revision, added.revision, duplicated.revision, revised.revision}) == 4
    revised_case = next(case for case in revised.cases if case.id == "refund-request")
    assert revised_case.revision != current.revision
    assert next(case for case in original.cases if case.id == "refund-request") == original.cases[0]


def test_mutations_reject_identity_collisions_missing_cases_and_stale_revisions() -> None:
    document = compile_eval_suite_draft(_draft())

    with pytest.raises(ValueError, match="already exists"):
        add_eval_case(document, _case())
    with pytest.raises(KeyError, match="not found"):
        duplicate_eval_case(document, "missing", new_case_id="copy")
    with pytest.raises(ValueError, match="changed after"):
        revise_eval_case(
            document,
            _case(prompt="Changed"),
            expected_case_revision="sha256:" + "0" * 64,
        )
    with pytest.raises(ValueError, match="changed after"):
        validate_expected_eval_suite_revision(document, "sha256:" + "0" * 64)


def test_full_and_subset_selections_freeze_exact_case_revisions() -> None:
    document = compile_eval_suite_draft(_draft(_case("case-a"), _case("case-b"), _case("case-c")))

    full = eval_suite_selection(document)
    subset = eval_suite_selection(document, ("case-c", "case-a"))

    assert full.mode == "full_suite"
    assert [case.id for case in full.cases] == ["case-a", "case-b", "case-c"]
    assert subset.mode == "subset"
    assert [case.id for case in subset.cases] == ["case-a", "case-c"]
    assert subset.suite_document_revision == document.revision
    assert subset.suite_revision == document.suite.revision
    assert subset.revision != full.revision
    assert validate_eval_suite_selection(full, document) == full
    assert validate_eval_suite_selection(subset, document) == subset

    with pytest.raises(ValueError, match="at least one"):
        eval_suite_selection(document, ())
    with pytest.raises(ValueError, match="unique"):
        eval_suite_selection(document, ("case-a", "case-a"))
    with pytest.raises(KeyError, match="not found"):
        eval_suite_selection(document, ("missing",))

    changed = revise_eval_case(
        document,
        _case("case-a", prompt="Changed"),
        expected_case_revision=document.cases[0].revision,
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_eval_suite_selection(subset, changed)


def test_authoring_models_reject_unknown_fields_duplicate_ids_and_forged_revisions() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        EvalCaseDraftV1.model_validate({**_case().model_dump(), "authority": "browser"})
    with pytest.raises(ValidationError, match="unique"):
        _draft(_case(), _case())

    document = compile_eval_suite_draft(_draft())
    forged = document.model_dump(mode="json")
    forged["revision"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError, match="does not match"):
        eval_suite_document_from_json(json.dumps(forged))


def test_suite_compilation_rejects_excessive_trial_assertion_expansion() -> None:
    assertions = tuple(
        RootStatusAssertionSpec(id=f"completed-{index}", expected="completed")
        for index in range(51)
    )
    cases = tuple(
        EvalCaseDraftV1(
            id=f"case-{index}",
            name=f"Case {index}",
            stimulus=EvalSimpleInputStimulusV1(
                input=RunInputSpec(messages=(CorpusUserMessageSpec(text=f"Prompt {index}"),))
            ),
            assertions=assertions,
        )
        for index in range(2)
    )

    with pytest.raises(ValidationError, match="10200 published assertion results"):
        compile_eval_suite_draft(
            EvalSuiteDraftV1(
                id="oversized-expansion",
                target_key="assistant.default",
                name="Oversized expansion",
                trial_request=TrialRequestSpec(trials=100),
                cases=cases,
            )
        )


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_authoring_json_rejects_nonportable_numbers(constant: str) -> None:
    with pytest.raises(ValueError, match="invalid"):
        eval_suite_document_from_json('{"value":' + constant + "}")
