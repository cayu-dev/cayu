from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from tests.evals.test_corpus_execution import _corpus

from cayu.evals.scenario import (
    EVAL_SCENARIO_MAX_BYTES,
    EVAL_SCENARIO_SCHEMA_VERSION,
    EvalScenarioDocumentV2,
    EvalScenarioInspectionV2,
    ScenarioApprovalCheckpointEventV2,
    ScenarioArtifactRequirementV2,
    ScenarioFilePartV2,
    ScenarioInitialInputEventV2,
    ScenarioInputV2,
    ScenarioJsonPartV2,
    ScenarioQueuedInputEventV2,
    ScenarioResumedInputEventV2,
    ScenarioSecretRequirementV2,
    ScenarioTextPartV2,
    ScenarioUserMessageV2,
    compile_eval_scenario,
    eval_scenario_from_json,
    eval_scenario_to_json,
    inspect_eval_scenario,
    load_eval_scenario,
    scenario_from_corpus_case,
)


def _message(*parts) -> ScenarioUserMessageV2:
    return ScenarioUserMessageV2.create(parts)


def _input(*messages) -> ScenarioInputV2:
    return ScenarioInputV2.create(messages)


def _artifact() -> ScenarioArtifactRequirementV2:
    return ScenarioArtifactRequirementV2(
        id="invoice",
        source="artifact_reference",
        reference="artifact-public-invoice",
        content_sha256="a" * 64,
        filename="invoice.pdf",
        content_type="application/pdf",
        size_bytes=1_024,
    )


def _scenario() -> EvalScenarioDocumentV2:
    return EvalScenarioDocumentV2.create(
        id="refund-follow-up",
        target_key="refund-agent",
        name="Refund follow-up",
        description="Exercise a retained multi-stage customer request.",
        artifact_requirements=(_artifact(),),
        secret_requirements=(
            ScenarioSecretRequirementV2(
                id="refund-test-account",
                usage="tool",
                purpose="Resolve the current test account at launch.",
            ),
        ),
        events=(
            ScenarioInitialInputEventV2(
                sequence=0,
                id="initial",
                input=_input(
                    _message(
                        ScenarioTextPartV2(text="Review this refund request."),
                        ScenarioJsonPartV2(value={"order_id": "order-42"}),
                        ScenarioFilePartV2(artifact_requirement_id="invoice"),
                    )
                ),
            ),
            ScenarioQueuedInputEventV2(
                sequence=1,
                id="customer-update",
                delivery_mode="next_turn",
                input=_input(_message(ScenarioTextPartV2(text="The package arrived damaged."))),
            ),
            ScenarioApprovalCheckpointEventV2(
                sequence=2,
                id="refund-approval",
                tool_name="issue_refund",
                occurrence=1,
            ),
            ScenarioResumedInputEventV2(
                sequence=3,
                id="requested-detail",
                resume_kind="user_input",
                input=_input(_message(ScenarioTextPartV2(text="Use the original payment method."))),
            ),
        ),
    )


def test_scenario_v2_round_trips_every_portable_stimulus_kind(tmp_path) -> None:
    scenario = _scenario()
    encoded = eval_scenario_to_json(scenario)
    restored = eval_scenario_from_json(encoded)

    assert restored == scenario
    assert restored.schema_version == EVAL_SCENARIO_SCHEMA_VERSION
    assert restored.revision.startswith("sha256:")
    assert [event.kind for event in restored.events] == [
        "initial",
        "queued",
        "approval_checkpoint",
        "resumed",
    ]
    assert "approve" not in encoded
    assert "approval_id" not in encoded
    assert "session_id" not in encoded

    path = tmp_path / "scenario.json"
    path.write_text(encoded, encoding="utf-8")
    assert load_eval_scenario(path) == scenario


def test_scenario_compiler_preserves_order_without_resolving_authority() -> None:
    compiled = compile_eval_scenario(_scenario())

    assert compiled.revision == compiled.document.revision
    assert compiled.target_key == "refund-agent"
    assert compiled.initial.id == "initial"
    assert [step.id for step in compiled.steps] == [
        "customer-update",
        "refund-approval",
        "requested-detail",
    ]
    assert compiled.artifact_requirement("invoice") == _artifact()
    assert compiled.secret_requirement("refund-test-account").usage == "tool"
    with pytest.raises(KeyError, match="not found"):
        compiled.artifact_requirement("missing")


def test_scenario_inspection_is_bounded_metadata_only() -> None:
    inspection = inspect_eval_scenario(_scenario())

    assert inspection.event_count == 4
    assert inspection.input_event_count == 3
    assert inspection.approval_checkpoint_count == 1
    assert inspection.message_count == 3
    assert inspection.part_count == 5
    assert inspection.artifact_requirement_count == 1
    assert inspection.secret_requirement_count == 1


@pytest.mark.parametrize(
    "updates",
    [
        {"approval_checkpoint_count": 2},
        {"message_count": 2},
        {"part_count": 2},
    ],
)
def test_scenario_inspection_rejects_impossible_counts(updates) -> None:
    document = inspect_eval_scenario(_scenario()).model_dump(mode="python")
    document.update(updates)

    with pytest.raises(ValidationError, match="counts are inconsistent|count is impossible"):
        EvalScenarioInspectionV2.model_validate(document)


def test_requirement_order_is_canonical_but_event_order_is_identity() -> None:
    scenario = _scenario()
    second_artifact = ScenarioArtifactRequirementV2(
        id="customer-photo",
        source="fixture_digest",
        content_sha256="b" * 64,
        filename="damage.png",
        content_type="image/png",
        size_bytes=2_048,
    )
    values = {
        "id": scenario.id,
        "target_key": scenario.target_key,
        "name": scenario.name,
        "description": scenario.description,
        "source": scenario.source,
        "events": scenario.events,
        "secret_requirements": scenario.secret_requirements,
    }
    left = EvalScenarioDocumentV2.create(
        **values,
        artifact_requirements=(second_artifact, _artifact()),
    )
    right = EvalScenarioDocumentV2.create(
        **values,
        artifact_requirements=(_artifact(), second_artifact),
    )
    assert left == right
    assert [item.id for item in left.artifact_requirements] == ["customer-photo", "invoice"]

    events = list(scenario.events)
    queued = events[1].model_copy(update={"sequence": 3})
    resumed = events[3].model_copy(update={"sequence": 1})
    reordered = EvalScenarioDocumentV2.create(
        id=scenario.id,
        target_key=scenario.target_key,
        name=scenario.name,
        description=scenario.description,
        source=scenario.source,
        events=(events[0], resumed, events[2], queued),
        artifact_requirements=scenario.artifact_requirements,
        secret_requirements=scenario.secret_requirements,
    )
    assert reordered.revision != scenario.revision


@pytest.mark.parametrize(
    ("events", "message"),
    [
        (
            (
                ScenarioQueuedInputEventV2(
                    sequence=0,
                    id="queued",
                    delivery_mode="on_idle",
                    input=_input(_message(ScenarioTextPartV2(text="later"))),
                ),
            ),
            "exactly one initial",
        ),
        (
            (
                ScenarioInitialInputEventV2(
                    sequence=1,
                    id="initial",
                    input=_input(_message(ScenarioTextPartV2(text="start"))),
                ),
            ),
            "contiguous",
        ),
        (
            (
                ScenarioInitialInputEventV2(
                    sequence=0,
                    id="same",
                    input=_input(_message(ScenarioTextPartV2(text="start"))),
                ),
                ScenarioResumedInputEventV2(
                    sequence=1,
                    id="same",
                    input=_input(_message(ScenarioTextPartV2(text="resume"))),
                ),
            ),
            "IDs must be unique",
        ),
    ],
)
def test_scenario_event_order_and_identity_fail_closed(events, message) -> None:
    with pytest.raises(ValidationError, match=message):
        EvalScenarioDocumentV2.create(
            id="invalid",
            target_key="agent",
            name="Invalid",
            events=events,
        )


def test_scenario_rejects_missing_artifact_requirements() -> None:
    with pytest.raises(ValidationError, match="undeclared artifact"):
        EvalScenarioDocumentV2.create(
            id="missing-artifact",
            target_key="agent",
            name="Missing artifact",
            events=(
                ScenarioInitialInputEventV2(
                    sequence=0,
                    id="initial",
                    input=_input(_message(ScenarioFilePartV2(artifact_requirement_id="missing"))),
                ),
            ),
        )


@pytest.mark.parametrize(
    "requirement",
    [
        ScenarioArtifactRequirementV2.model_construct(
            id="fixture",
            source="fixture_digest",
            reference="must-not-exist",
            content_sha256="a" * 64,
            filename="fixture.txt",
            content_type="text/plain",
            size_bytes=1,
        ),
        ScenarioArtifactRequirementV2.model_construct(
            id="artifact",
            source="artifact_reference",
            reference=None,
            content_sha256="a" * 64,
            filename="artifact.txt",
            content_type="text/plain",
            size_bytes=1,
        ),
    ],
)
def test_scenario_revalidates_artifact_source_contract(requirement) -> None:
    with pytest.raises(ValidationError, match="need `reference` exactly"):
        EvalScenarioDocumentV2.create(
            id="bad-artifact",
            target_key="agent",
            name="Bad artifact",
            artifact_requirements=(requirement,),
            events=(
                ScenarioInitialInputEventV2(
                    sequence=0,
                    id="initial",
                    input=_input(_message(ScenarioTextPartV2(text="start"))),
                ),
            ),
        )


def test_approval_checkpoints_require_unique_ascending_occurrences() -> None:
    with pytest.raises(ValidationError, match="unique ascending"):
        EvalScenarioDocumentV2.create(
            id="bad-approvals",
            target_key="agent",
            name="Bad approvals",
            events=(
                ScenarioInitialInputEventV2(
                    sequence=0,
                    id="initial",
                    input=_input(_message(ScenarioTextPartV2(text="start"))),
                ),
                ScenarioApprovalCheckpointEventV2(
                    sequence=1,
                    id="second",
                    tool_name="send_email",
                    occurrence=2,
                ),
                ScenarioApprovalCheckpointEventV2(
                    sequence=2,
                    id="first",
                    tool_name="send_email",
                    occurrence=1,
                ),
            ),
        )


def test_scenario_revision_covers_nested_structured_input() -> None:
    scenario = _scenario()
    document = scenario.model_dump(mode="json")
    document["events"][0]["input"]["messages"][0]["content"][1]["value"]["order_id"] = "changed"
    with pytest.raises(ValidationError, match="revision does not match"):
        EvalScenarioDocumentV2.model_validate(document)


def test_scenario_nested_structured_input_is_immutable_and_serializable() -> None:
    source = {"nested": [{"value": "original"}]}
    scenario = EvalScenarioDocumentV2.create(
        id="immutable-json",
        target_key="agent",
        name="Immutable JSON",
        events=(
            ScenarioInitialInputEventV2(
                sequence=0,
                id="initial",
                input=_input(_message(ScenarioJsonPartV2(value=source))),
            ),
        ),
    )
    part = scenario.events[0].input.messages[0].content[0]
    assert isinstance(part, ScenarioJsonPartV2)

    source["nested"][0]["value"] = "caller mutation"
    assert part.value["nested"][0]["value"] == "original"
    with pytest.raises(TypeError, match="Frozen JSON values cannot be mutated"):
        part.value["nested"][0]["value"] = "document mutation"

    document = scenario.model_dump(mode="json")
    assert type(document["events"][0]["input"]["messages"][0]["content"][0]["value"]) is dict
    assert eval_scenario_from_json(eval_scenario_to_json(scenario)) == scenario
    assert compile_eval_scenario(scenario).document == scenario


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 2**63])
def test_scenario_json_parts_reject_nonportable_values(value) -> None:
    with pytest.raises(ValidationError):
        ScenarioJsonPartV2(value={"bad": value})


def test_scenario_json_loader_rejects_duplicate_keys_and_unknown_versions() -> None:
    encoded = eval_scenario_to_json(_scenario())
    with pytest.raises(ValueError, match="duplicate JSON object keys"):
        eval_scenario_from_json('{"schema_version":2,"schema_version":2}')

    document = json.loads(encoded)
    document["schema_version"] = 3
    with pytest.raises(ValueError, match="unsupported schema_version 3"):
        eval_scenario_from_json(json.dumps(document))


def test_scenario_loader_applies_byte_limit_before_json_decode(tmp_path) -> None:
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * (EVAL_SCENARIO_MAX_BYTES + 1))
    with pytest.raises(ValueError, match="exceeds"):
        load_eval_scenario(path)


def test_corpus_v2_runnable_case_compiles_to_scenario_v2() -> None:
    corpus = _corpus(input_text="Evaluate this request")
    case = corpus.cases[0]

    scenario = scenario_from_corpus_case(corpus, case.id)

    assert scenario.target_key == corpus.target_key
    assert scenario.id == case.id
    assert scenario.source == case.source
    initial = scenario.events[0]
    assert isinstance(initial, ScenarioInitialInputEventV2)
    parts = initial.input.messages[0].content
    assert len(parts) == 1
    assert isinstance(parts[0], ScenarioTextPartV2)
    assert parts[0].text == "Evaluate this request"
    assert scenario.artifact_requirements == ()
    assert scenario.secret_requirements == ()


def test_captured_only_corpus_case_needs_authored_scenario_stimuli() -> None:
    corpus = _corpus()
    case = corpus.cases[0].model_copy(update={"input": None})
    case_document = case.model_dump(mode="json", exclude={"revision"})
    from cayu.evals.corpus import EvalCaseSpec, _content_revision

    captured_case = EvalCaseSpec(
        revision=_content_revision(case_document, "eval case spec"),
        **case_document,
    )
    from cayu.evals.corpus import EvalCorpusDocument

    captured = EvalCorpusDocument.create(
        target_key=corpus.target_key,
        evidence_policy=corpus.evidence_policy,
        pricing_profile=corpus.pricing_profile,
        suites=corpus.suites,
        cases=(captured_case,),
    )

    with pytest.raises(ValueError, match="need authored scenario stimuli"):
        scenario_from_corpus_case(captured, captured_case.id)


def test_scenario_models_reject_authority_fields() -> None:
    document = _scenario().model_dump(mode="json")
    document["session_id"] = "private-session"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EvalScenarioDocumentV2.model_validate(document)
