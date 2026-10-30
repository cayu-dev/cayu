from __future__ import annotations

import pytest
from pydantic import ValidationError

from cayu import AgentSpec, CayuApp, RunRequest
from cayu.evals import CorpusTarget, ScriptedModelProvider
from cayu.evals.corpus import CorpusUserMessageSpec, RootStatusAssertionSpec, RunInputSpec
from cayu.evals.execution import compile_corpus_suite, evaluation_target_identity
from cayu.evals.store import EvalRunInvocation, EvalScenarioRunInvocation
from cayu.evals.suite_authoring import (
    EvalCaseDraftV1,
    EvalCaseDraftV2,
    EvalScenarioStimulusV1,
    EvalSimpleInputStimulusV1,
    EvalSuiteDraftV1,
    compile_eval_suite_draft,
    duplicate_eval_case,
    eval_suite_selection,
)
from cayu.evals.suite_execution import corpus_for_authored_simple_selection
from cayu.evals.workspace_fixtures import WorkspaceFixtureSpec

_REVISION_A = "sha256:" + "a" * 64
_REVISION_B = "sha256:" + "b" * 64
_REVISION_C = "sha256:" + "c" * 64


def _target(*, release: str, system_prompt: str, extra_agent: bool = False) -> CorpusTarget:
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="scripted-model",
            system_prompt=system_prompt,
        )
    )
    if extra_agent:
        app.register_agent(AgentSpec(name="secondary", model="scripted-model"))
    return CorpusTarget(
        key="assistant.default",
        app=app,
        request_base=RunRequest(agent_name="assistant", messages=[]),
        application_release_id=release,
    )


def test_authored_corpus_identity_is_stable_across_application_releases() -> None:
    document = compile_eval_suite_draft(
        EvalSuiteDraftV1(
            id="regressions",
            target_key="assistant.default",
            name="Regressions",
            cases=(
                EvalCaseDraftV1(
                    id="case-one",
                    name="Case one",
                    stimulus=EvalSimpleInputStimulusV1(
                        input=RunInputSpec(
                            messages=(CorpusUserMessageSpec(text="Evaluate this behavior."),)
                        )
                    ),
                    assertions=(RootStatusAssertionSpec(id="completed", expected="completed"),),
                ),
            ),
        )
    )
    selection = eval_suite_selection(document)
    first_target = _target(release="release-one", system_prompt="First release.")
    second_target = _target(
        release="release-two",
        system_prompt="Second release.",
        extra_agent=True,
    )

    first_identity = evaluation_target_identity(first_target)
    second_identity = evaluation_target_identity(second_target)
    assert first_identity.application_release_id != second_identity.application_release_id
    assert first_identity.app_manifest_fingerprint != second_identity.app_manifest_fingerprint

    first = corpus_for_authored_simple_selection(document, selection, first_target)
    second = corpus_for_authored_simple_selection(document, selection, second_target)

    assert first == second
    assert first.cases[0].source.application_release_id == "cayu-authored-case-v1"
    assert first.cases[0].source.app_manifest_schema_version == "authored-eval-case-v1"
    assert first.cases[0].source.evidence_revision == document.cases[0].revision
    assert compile_corpus_suite(first, first_target, document.suite.id).corpus == first
    assert compile_corpus_suite(second, second_target, document.suite.id).corpus == second


def test_an_authored_case_fixture_reaches_its_corpus_case_and_an_absent_one_is_omitted() -> None:
    def draft(**fields) -> EvalCaseDraftV1:
        return EvalCaseDraftV1(
            id="case-one",
            name="Case one",
            stimulus=EvalSimpleInputStimulusV1(
                input=RunInputSpec(messages=(CorpusUserMessageSpec(text="Check the claim."),))
            ),
            assertions=(RootStatusAssertionSpec(id="completed", expected="completed"),),
            **fields,
        )

    def suite(case: EvalCaseDraftV1):
        return compile_eval_suite_draft(
            EvalSuiteDraftV1(
                id="claims", target_key="assistant.default", name="Claims", cases=(case,)
            )
        )

    fixture = WorkspaceFixtureSpec(repository="payments", revision="a" * 40)
    plain = suite(draft())
    fixed = suite(draft(workspace=fixture))
    # A case without a fixture serializes, and so revises, as it did before fixtures.
    assert "workspace" not in plain.cases[0].model_dump(mode="json")
    assert fixed.cases[0].revision != plain.cases[0].revision
    assert EvalCaseDraftV2.from_case(fixed.cases[0]).workspace == fixture
    target = _target(release="release-one", system_prompt="First release.")
    corpus = corpus_for_authored_simple_selection(fixed, eval_suite_selection(fixed), target)
    assert corpus.cases[0].workspace == fixture
    plain_corpus = corpus_for_authored_simple_selection(plain, eval_suite_selection(plain), target)
    assert plain_corpus.cases[0].workspace is None
    # A duplicated case starts from the same fixture.
    duplicated = duplicate_eval_case(fixed, "case-one", new_case_id="case-two")
    assert [case.workspace for case in duplicated.cases] == [fixture, fixture]


def test_a_scenario_case_takes_no_workspace_fixture() -> None:
    # A resumed scenario trial would get a fresh checkout without its earlier changes.
    scenario = EvalScenarioStimulusV1(scenario_id="refund", scenario_revision=_REVISION_A)
    fixture = WorkspaceFixtureSpec(repository="payments", revision="a" * 40)
    for draft_type in (EvalCaseDraftV1, EvalCaseDraftV2):
        with pytest.raises(ValidationError, match="scenario case can't have a workspace"):
            draft_type(
                id="case-one",
                name="Case one",
                stimulus=scenario,
                assertions=(RootStatusAssertionSpec(id="completed", expected="completed"),),
                workspace=fixture,
            )


def test_authored_invocation_provenance_is_paired_and_legacy_shape_stays_stable() -> None:
    legacy_scenario = EvalScenarioRunInvocation(
        scenario_revision=_REVISION_A,
        binding_revision=_REVISION_B,
        trials=1,
        timeout_seconds=30,
    )
    assert "authored_suite_revision" not in legacy_scenario.model_dump(mode="json")
    assert "authored_case_revision" not in legacy_scenario.model_dump(mode="json")

    with pytest.raises(ValueError, match="both suite and selection"):
        EvalRunInvocation(authored_suite_revision=_REVISION_A)
    with pytest.raises(ValueError, match="launch revision and lane must be paired"):
        EvalRunInvocation(
            authored_suite_revision=_REVISION_A,
            authored_suite_selection_revision=_REVISION_B,
            authored_suite_launch_revision=_REVISION_C,
        )
    invocation = EvalRunInvocation(
        authored_suite_revision=_REVISION_A,
        authored_suite_selection_revision=_REVISION_B,
        authored_suite_launch_revision=_REVISION_C,
        authored_suite_launch_lane=0,
    )
    assert invocation.authored_suite_launch_lane == 0
    with pytest.raises(ValueError, match="both suite and case"):
        EvalScenarioRunInvocation(
            scenario_revision=_REVISION_A,
            binding_revision=_REVISION_B,
            authored_suite_revision=_REVISION_C,
            trials=1,
            timeout_seconds=30,
        )
    with pytest.raises(ValueError, match="must match its parent"):
        EvalRunInvocation(
            authored_suite_revision=_REVISION_A,
            authored_suite_selection_revision=_REVISION_B,
            scenario=EvalScenarioRunInvocation(
                scenario_revision=_REVISION_A,
                binding_revision=_REVISION_B,
                authored_suite_revision=_REVISION_C,
                authored_case_revision=_REVISION_A,
                trials=1,
                timeout_seconds=30,
            ),
        )
