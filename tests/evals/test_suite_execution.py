from __future__ import annotations

import pytest

from cayu import AgentSpec, CayuApp, CorpusTarget, RunRequest, ScriptedModelProvider
from cayu.evals.corpus import CorpusUserMessageSpec, RootStatusAssertionSpec, RunInputSpec
from cayu.evals.execution import compile_corpus_suite, evaluation_target_identity
from cayu.evals.store import EvalRunInvocation, EvalScenarioRunInvocation
from cayu.evals.suite_authoring import (
    EvalCaseDraftV1,
    EvalSimpleInputStimulusV1,
    EvalSuiteDraftV1,
    compile_eval_suite_draft,
    eval_suite_selection,
)
from cayu.evals.suite_execution import corpus_for_authored_simple_selection

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
