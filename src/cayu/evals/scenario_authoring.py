from __future__ import annotations

from pydantic import Field, StrictStr, field_validator

from cayu.evals.corpus import (
    EvaluationSourceIdentityV1,
    _bounded_durable_text,
    _model_python_input,
    _ordered_sequence_argument,
    _ordered_sequence_input,
    _portable_id,
    _PortableModel,
    _sha256_revision,
)
from cayu.evals.scenario import (
    EVAL_SCENARIO_MAX_ARTIFACT_REQUIREMENTS,
    EVAL_SCENARIO_MAX_EVENTS,
    EVAL_SCENARIO_MAX_SECRET_REQUIREMENTS,
    EvalScenarioDocumentV2,
    ScenarioArtifactRequirementV2,
    ScenarioEventV2,
    ScenarioSecretRequirementV2,
    _validated_event,
)


class EvalScenarioDraftV2(_PortableModel):
    """Revision-free scenario material edited by a human or Control Plane.

    A draft is never executable and is not accepted by the durable store. The
    framework canonicalizes it into :class:`EvalScenarioDocumentV2`, computes
    the content revision, and then runs launch preflight against current
    server-owned authority.
    """

    id: StrictStr
    target_key: StrictStr
    name: StrictStr
    description: StrictStr | None = None
    source: EvaluationSourceIdentityV1 | None = None
    events: tuple[ScenarioEventV2, ...] = Field(
        min_length=1,
        max_length=EVAL_SCENARIO_MAX_EVENTS,
    )
    artifact_requirements: tuple[ScenarioArtifactRequirementV2, ...] = Field(
        default_factory=tuple,
        max_length=EVAL_SCENARIO_MAX_ARTIFACT_REQUIREMENTS,
    )
    secret_requirements: tuple[ScenarioSecretRequirementV2, ...] = Field(
        default_factory=tuple,
        max_length=EVAL_SCENARIO_MAX_SECRET_REQUIREMENTS,
    )

    @field_validator("events", "artifact_requirements", "secret_requirements", mode="before")
    @classmethod
    def validate_collections_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("id", "target_key")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=2_048,
            nonblank=True,
            clean=True,
        )

    @classmethod
    def from_scenario(cls, scenario: EvalScenarioDocumentV2) -> EvalScenarioDraftV2:
        if type(scenario) is not EvalScenarioDocumentV2:
            raise TypeError("scenario must be an exact EvalScenarioDocumentV2.")
        validated = EvalScenarioDocumentV2.model_validate(_model_python_input(scenario))
        return cls(
            id=validated.id,
            target_key=validated.target_key,
            name=validated.name,
            description=validated.description,
            source=validated.source,
            events=validated.events,
            artifact_requirements=validated.artifact_requirements,
            secret_requirements=validated.secret_requirements,
        )


def compile_eval_scenario_draft(draft: EvalScenarioDraftV2) -> EvalScenarioDocumentV2:
    """Canonicalize one authority-free draft and compute its immutable revision."""

    if type(draft) is not EvalScenarioDraftV2:
        raise TypeError("draft must be an exact EvalScenarioDraftV2.")
    validated = EvalScenarioDraftV2.model_validate(_model_python_input(draft))
    events = _ordered_sequence_argument(validated.events, "events")
    artifacts = _ordered_sequence_argument(
        validated.artifact_requirements,
        "artifact_requirements",
    )
    secrets = _ordered_sequence_argument(validated.secret_requirements, "secret_requirements")
    return EvalScenarioDocumentV2.create(
        id=validated.id,
        target_key=validated.target_key,
        name=validated.name,
        description=validated.description,
        source=validated.source,
        events=tuple(_validated_event(event) for event in events),
        artifact_requirements=tuple(
            ScenarioArtifactRequirementV2.model_validate(_model_python_input(requirement))
            for requirement in artifacts
        ),
        secret_requirements=tuple(
            ScenarioSecretRequirementV2.model_validate(_model_python_input(requirement))
            for requirement in secrets
        ),
    )


def replace_eval_scenario_artifact_requirement(
    scenario: EvalScenarioDocumentV2,
    replacement: ScenarioArtifactRequirementV2,
) -> EvalScenarioDocumentV2:
    """Return a new immutable scenario revision with one exact requirement replaced."""

    if type(scenario) is not EvalScenarioDocumentV2:
        raise TypeError("scenario must be an exact EvalScenarioDocumentV2.")
    if type(replacement) is not ScenarioArtifactRequirementV2:
        raise TypeError("replacement must be an exact ScenarioArtifactRequirementV2.")
    validated = EvalScenarioDocumentV2.model_validate(_model_python_input(scenario))
    copied_replacement = ScenarioArtifactRequirementV2.model_validate(
        _model_python_input(replacement)
    )
    if all(item.id != copied_replacement.id for item in validated.artifact_requirements):
        raise KeyError(f"Scenario artifact requirement not found: {copied_replacement.id}")
    requirements = tuple(
        copied_replacement if item.id == copied_replacement.id else item
        for item in validated.artifact_requirements
    )
    return EvalScenarioDocumentV2.create(
        id=validated.id,
        target_key=validated.target_key,
        name=validated.name,
        description=validated.description,
        source=validated.source,
        events=validated.events,
        artifact_requirements=requirements,
        secret_requirements=validated.secret_requirements,
    )


def validate_expected_scenario_revision(
    scenario: EvalScenarioDocumentV2,
    expected_revision: str,
) -> EvalScenarioDocumentV2:
    """Reject stale mutation requests before any artifact or store side effect."""

    if type(scenario) is not EvalScenarioDocumentV2:
        raise TypeError("scenario must be an exact EvalScenarioDocumentV2.")
    expected_revision = _sha256_revision(expected_revision, "expected_revision")
    validated = EvalScenarioDocumentV2.model_validate(_model_python_input(scenario))
    if validated.revision != expected_revision:
        raise ValueError("Scenario changed after the reviewed revision.")
    return validated


__all__ = [
    "EvalScenarioDraftV2",
    "compile_eval_scenario_draft",
    "replace_eval_scenario_artifact_requirement",
    "validate_expected_scenario_revision",
]
