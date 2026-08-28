from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from decimal import Decimal
from functools import partial
from typing import Annotated, Any, Literal, Protocol, TypeAlias, cast, overload

from pydantic import Field, StrictStr, field_validator, model_validator

from cayu._validation import (
    durable_json_object_from_pairs,
    json_utf8_size_within_limit,
    parse_durable_json_integer_literal,
    reject_nonportable_json_constant,
)
from cayu.evals.corpus import (
    EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE,
    EVAL_CORPUS_MAX_BYTES,
    EVAL_CORPUS_MAX_CASES,
    EVAL_CORPUS_MAX_JUDGE_CRITERIA,
    EVAL_CORPUS_MAX_JUDGE_EXPLANATION_CHARS,
    EVAL_CORPUS_MAX_JUDGE_REFERENCE_FACTS,
    EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS,
    EVAL_CORPUS_MAX_PUBLISHED_JUDGE_EXPLANATION_CHARS,
    ChildStatusAssertionSpec,
    EvalJudgeEvidenceSelectionV1,
    EvalSuiteSpec,
    EvaluationSourceIdentityV1,
    FinalOutputContainsAssertionSpec,
    FinalOutputEqualsAssertionSpec,
    MaxEstimatedCostAssertionSpec,
    MaxModelStepsAssertionSpec,
    MaxToolCallsAssertionSpec,
    MaxTotalTokensAssertionSpec,
    ModelJudgeAssertionSpec,
    PrivateJudgeReferenceV1,
    PublicJudgeReferenceV1,
    RootStatusAssertionSpec,
    RunInputSpec,
    StructuredModelJudgeAssertionSpec,
    StructuredRubricCriterionV1,
    StructuredRubricV1,
    ToolCalledAssertionSpec,
    ToolsCalledInOrderAssertionSpec,
    TrialRequestSpec,
    UsageRecordedAssertionSpec,
    _bounded_durable_text,
    _content_revision,
    _exact_decimal_sum,
    _model_content_revision,
    _model_python_input,
    _ordered_sequence_argument,
    _ordered_sequence_input,
    _portable_id,
    _PortableModel,
    _pretty_json_size_within_limit,
    _SchemaV1PortableModel,
    _sha256_revision,
    _unit_interval_decimal_text,
    _validated_assertion_spec,
)

EVAL_SUITE_AUTHORING_SCHEMA_VERSION = 1
EVAL_SUITE_AUTHORING_V2_SCHEMA_VERSION = 2
EVAL_SUITE_SELECTION_SCHEMA_VERSION = 1
EVAL_SUITE_AUTHORING_MAX_BYTES = EVAL_CORPUS_MAX_BYTES


# Suite authoring V1 has an already-published wire contract. Structured judge
# authoring belongs to V2 (PR 05); keeping a separate union prevents the corpus
# V2 addition from silently widening what V1 clients are expected to round-trip.
EvalSuiteAuthoringAssertionSpecV1: TypeAlias = Annotated[
    RootStatusAssertionSpec
    | ChildStatusAssertionSpec
    | FinalOutputEqualsAssertionSpec
    | FinalOutputContainsAssertionSpec
    | ToolCalledAssertionSpec
    | ToolsCalledInOrderAssertionSpec
    | MaxToolCallsAssertionSpec
    | MaxModelStepsAssertionSpec
    | UsageRecordedAssertionSpec
    | MaxTotalTokensAssertionSpec
    | MaxEstimatedCostAssertionSpec
    | ModelJudgeAssertionSpec,
    Field(discriminator="kind"),
]

EvalSuiteAuthoringAssertionSpecV2: TypeAlias = Annotated[
    RootStatusAssertionSpec
    | ChildStatusAssertionSpec
    | FinalOutputEqualsAssertionSpec
    | FinalOutputContainsAssertionSpec
    | ToolCalledAssertionSpec
    | ToolsCalledInOrderAssertionSpec
    | MaxToolCallsAssertionSpec
    | MaxModelStepsAssertionSpec
    | UsageRecordedAssertionSpec
    | MaxTotalTokensAssertionSpec
    | MaxEstimatedCostAssertionSpec
    | ModelJudgeAssertionSpec
    | StructuredModelJudgeAssertionSpec,
    Field(discriminator="kind"),
]


class StructuredRubricDraftV1(_PortableModel):
    """Revision-free structured rubric material accepted from an editor."""

    schema_version: Literal[1] = 1
    id: StrictStr
    criteria: tuple[StructuredRubricCriterionV1, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_JUDGE_CRITERIA,
    )

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("criteria", mode="before")
    @classmethod
    def validate_criteria_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("criteria")
    @classmethod
    def validate_criteria(
        cls,
        value: tuple[StructuredRubricCriterionV1, ...],
    ) -> tuple[StructuredRubricCriterionV1, ...]:
        if any(type(criterion) is not StructuredRubricCriterionV1 for criterion in value):
            raise TypeError("criteria must contain exact StructuredRubricCriterionV1 values.")
        return tuple(
            StructuredRubricCriterionV1.model_validate(_model_python_input(criterion))
            for criterion in value
        )

    @model_validator(mode="after")
    def validate_contract(self) -> StructuredRubricDraftV1:
        criterion_ids = tuple(criterion.id for criterion in self.criteria)
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("Structured rubric criterion IDs must be unique.")
        if _exact_decimal_sum(Decimal(criterion.weight) for criterion in self.criteria) != 1:
            raise ValueError("Structured rubric criterion weights must sum exactly to 1.")
        return self

    @classmethod
    def from_rubric(cls, rubric: StructuredRubricV1) -> StructuredRubricDraftV1:
        if type(rubric) is not StructuredRubricV1:
            raise TypeError("rubric must be an exact StructuredRubricV1.")
        validated = StructuredRubricV1.model_validate(_model_python_input(rubric))
        return cls(id=validated.id, criteria=validated.criteria)

    def compile(self) -> StructuredRubricV1:
        return StructuredRubricV1.create(id=self.id, criteria=self.criteria)


class PublicJudgeReferenceDraftV1(_PortableModel):
    """Revision-free public reference truth accepted from an editor."""

    schema_version: Literal[1] = 1
    kind: Literal["public_reference"] = "public_reference"
    id: StrictStr
    expected_answer: StrictStr | None = None
    expected_facts: tuple[StrictStr, ...] = Field(
        default=(),
        max_length=EVAL_CORPUS_MAX_JUDGE_REFERENCE_FACTS,
    )

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("expected_facts", mode="before")
    @classmethod
    def validate_expected_facts_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @model_validator(mode="after")
    def validate_contract(self) -> PublicJudgeReferenceDraftV1:
        # The immutable model remains the single source of all text bounds.
        PublicJudgeReferenceV1.create(
            id=self.id,
            expected_answer=self.expected_answer,
            expected_facts=self.expected_facts,
        )
        return self

    @classmethod
    def from_reference(cls, reference: PublicJudgeReferenceV1) -> PublicJudgeReferenceDraftV1:
        if type(reference) is not PublicJudgeReferenceV1:
            raise TypeError("reference must be an exact PublicJudgeReferenceV1.")
        validated = PublicJudgeReferenceV1.model_validate(_model_python_input(reference))
        return cls(
            id=validated.id,
            expected_answer=validated.expected_answer,
            expected_facts=validated.expected_facts,
        )

    def compile(self) -> PublicJudgeReferenceV1:
        return PublicJudgeReferenceV1.create(
            id=self.id,
            expected_answer=self.expected_answer,
            expected_facts=self.expected_facts,
        )


JudgeReferenceDraftV1: TypeAlias = Annotated[
    PublicJudgeReferenceDraftV1 | PrivateJudgeReferenceV1,
    Field(discriminator="kind"),
]


class StructuredModelJudgeAssertionDraftV1(_PortableModel):
    """Revision-free rubric assertion material compiled only by the server."""

    id: StrictStr
    kind: Literal["structured_model_judge"] = "structured_model_judge"
    description: StrictStr | None = None
    judge_profile_key: StrictStr
    judge_profile_revision: StrictStr
    rubric: StructuredRubricDraftV1
    reference: JudgeReferenceDraftV1 | None = None
    threshold: StrictStr = "0.5"
    evidence: EvalJudgeEvidenceSelectionV1 = Field(default_factory=EvalJudgeEvidenceSelectionV1)

    @field_validator("id", "judge_profile_key")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

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

    @field_validator("judge_profile_revision")
    @classmethod
    def validate_judge_profile_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, value: str, info) -> str:
        return _unit_interval_decimal_text(value, info.field_name)

    @classmethod
    def from_assertion(
        cls,
        assertion: StructuredModelJudgeAssertionSpec,
    ) -> StructuredModelJudgeAssertionDraftV1:
        if type(assertion) is not StructuredModelJudgeAssertionSpec:
            raise TypeError("assertion must be an exact StructuredModelJudgeAssertionSpec.")
        validated = StructuredModelJudgeAssertionSpec.model_validate(_model_python_input(assertion))
        reference = validated.reference
        draft_reference: JudgeReferenceDraftV1 | None
        if type(reference) is PublicJudgeReferenceV1:
            draft_reference = PublicJudgeReferenceDraftV1.from_reference(reference)
        elif type(reference) is PrivateJudgeReferenceV1:
            draft_reference = reference
        elif reference is None:
            draft_reference = None
        else:
            raise TypeError("reference must use an exact built-in judge reference type.")
        return cls(
            id=validated.id,
            description=validated.description,
            judge_profile_key=validated.judge_profile_key,
            judge_profile_revision=validated.judge_profile_revision,
            rubric=StructuredRubricDraftV1.from_rubric(validated.rubric),
            reference=draft_reference,
            threshold=validated.threshold,
            evidence=validated.evidence,
        )

    def compile(self) -> StructuredModelJudgeAssertionSpec:
        reference = self.reference
        compiled_reference: PublicJudgeReferenceV1 | PrivateJudgeReferenceV1 | None
        if type(reference) is PublicJudgeReferenceDraftV1:
            compiled_reference = reference.compile()
        elif type(reference) is PrivateJudgeReferenceV1:
            compiled_reference = reference
        elif reference is None:
            compiled_reference = None
        else:
            raise TypeError("reference must use an exact built-in judge reference type.")
        return StructuredModelJudgeAssertionSpec(
            id=self.id,
            description=self.description,
            judge_profile_key=self.judge_profile_key,
            judge_profile_revision=self.judge_profile_revision,
            rubric=self.rubric.compile(),
            reference=compiled_reference,
            threshold=self.threshold,
            evidence=self.evidence,
        )


EvalSuiteAuthoringAssertionDraftSpecV2: TypeAlias = Annotated[
    RootStatusAssertionSpec
    | ChildStatusAssertionSpec
    | FinalOutputEqualsAssertionSpec
    | FinalOutputContainsAssertionSpec
    | ToolCalledAssertionSpec
    | ToolsCalledInOrderAssertionSpec
    | MaxToolCallsAssertionSpec
    | MaxModelStepsAssertionSpec
    | UsageRecordedAssertionSpec
    | MaxTotalTokensAssertionSpec
    | MaxEstimatedCostAssertionSpec
    | ModelJudgeAssertionSpec
    | StructuredModelJudgeAssertionDraftV1,
    Field(discriminator="kind"),
]


class EvalSimpleInputStimulusV1(_PortableModel):
    """One ordinary fresh invocation authored without captured provenance."""

    kind: Literal["simple_input"] = "simple_input"
    input: RunInputSpec

    @field_validator("input")
    @classmethod
    def validate_input(cls, value: RunInputSpec) -> RunInputSpec:
        if type(value) is not RunInputSpec:
            raise TypeError("input must be an exact RunInputSpec.")
        return RunInputSpec.model_validate(_model_python_input(value))


class EvalScenarioStimulusV1(_PortableModel):
    """A content-addressed reference to one separately persisted scenario."""

    kind: Literal["scenario"] = "scenario"
    scenario_id: StrictStr
    scenario_revision: StrictStr

    @field_validator("scenario_id")
    @classmethod
    def validate_scenario_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("scenario_revision")
    @classmethod
    def validate_scenario_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)


EvalCaseStimulusV1: TypeAlias = Annotated[
    EvalSimpleInputStimulusV1 | EvalScenarioStimulusV1,
    Field(discriminator="kind"),
]

_STIMULUS_TYPES = (EvalSimpleInputStimulusV1, EvalScenarioStimulusV1)


def _validated_stimulus(stimulus: EvalCaseStimulusV1) -> EvalCaseStimulusV1:
    stimulus_type = type(stimulus)
    if stimulus_type not in _STIMULUS_TYPES:
        raise TypeError("stimulus must use an exact built-in EvalCaseStimulusV1 type.")
    return stimulus_type.model_validate(_model_python_input(stimulus))


class _AssertionWithId(Protocol):
    id: str


def _validate_unique_assertion_ids(assertions: Sequence[_AssertionWithId]) -> None:
    ids = tuple(assertion.id for assertion in assertions)
    duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(
            "Eval case assertion IDs must be unique; duplicated: " + ", ".join(duplicates)
        )


class _EvalCaseAuthoringBase(_PortableModel):
    id: StrictStr
    name: StrictStr
    description: StrictStr | None = None
    source: EvaluationSourceIdentityV1 | None = None
    stimulus: EvalCaseStimulusV1

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
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

    @field_validator("stimulus")
    @classmethod
    def validate_stimulus(cls, value: EvalCaseStimulusV1) -> EvalCaseStimulusV1:
        return _validated_stimulus(value)

    @field_validator("source")
    @classmethod
    def validate_source(
        cls,
        value: EvaluationSourceIdentityV1 | None,
    ) -> EvaluationSourceIdentityV1 | None:
        if value is None:
            return None
        if type(value) is not EvaluationSourceIdentityV1:
            raise TypeError("source must be an exact EvaluationSourceIdentityV1 or None.")
        return EvaluationSourceIdentityV1.model_validate(_model_python_input(value))


class _EvalCaseAuthoringMaterial(_EvalCaseAuthoringBase):
    assertions: tuple[EvalSuiteAuthoringAssertionSpecV1, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE,
    )

    @field_validator("assertions", mode="before")
    @classmethod
    def validate_assertions_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("assertions")
    @classmethod
    def validate_assertions(
        cls,
        value: tuple[EvalSuiteAuthoringAssertionSpecV1, ...],
    ) -> tuple[EvalSuiteAuthoringAssertionSpecV1, ...]:
        return tuple(
            cast("EvalSuiteAuthoringAssertionSpecV1", _validated_assertion_spec(assertion))
            for assertion in value
        )

    @model_validator(mode="after")
    def validate_assertion_ids(self) -> _EvalCaseAuthoringMaterial:
        _validate_unique_assertion_ids(self.assertions)
        return self


class EvalCaseDraftV1(_EvalCaseAuthoringMaterial):
    """Bounded, revision-free case material accepted by authoring previews."""

    @classmethod
    def from_case(cls, case: EvalCaseDefinitionV1) -> EvalCaseDraftV1:
        if type(case) is not EvalCaseDefinitionV1:
            raise TypeError("case must be an exact EvalCaseDefinitionV1.")
        validated = EvalCaseDefinitionV1.model_validate(_model_python_input(case))
        return cls.model_validate(validated.model_dump(mode="python", exclude={"revision"}))


class EvalCaseDefinitionV1(_EvalCaseAuthoringMaterial):
    """One immutable authored case revision with truthful optional provenance."""

    revision: StrictStr

    @field_validator("revision")
    @classmethod
    def validate_revision_shape(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @model_validator(mode="after")
    def validate_revision(self) -> EvalCaseDefinitionV1:
        expected = _model_content_revision(self, "authored eval case")
        if self.revision != expected:
            raise ValueError("Authored eval case revision does not match its content.")
        return self

    @classmethod
    def create(cls, draft: EvalCaseDraftV1) -> EvalCaseDefinitionV1:
        if type(draft) is not EvalCaseDraftV1:
            raise TypeError("draft must be an exact EvalCaseDraftV1.")
        validated = EvalCaseDraftV1.model_validate(_model_python_input(draft))
        document = validated.model_dump(mode="json")
        return cls(
            revision=_content_revision(document, "authored eval case"),
            **document,
        )


class _EvalCaseAuthoringMaterialV2(_EvalCaseAuthoringBase):
    """V2 case material that can represent trusted structured judging."""

    assertions: tuple[EvalSuiteAuthoringAssertionSpecV2, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE,
    )

    @field_validator("assertions", mode="before")
    @classmethod
    def validate_assertions_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("assertions")
    @classmethod
    def validate_assertions(
        cls,
        value: tuple[EvalSuiteAuthoringAssertionSpecV2, ...],
    ) -> tuple[EvalSuiteAuthoringAssertionSpecV2, ...]:
        return tuple(_validated_assertion_spec(assertion) for assertion in value)

    @model_validator(mode="after")
    def validate_assertion_ids(self) -> _EvalCaseAuthoringMaterialV2:
        _validate_unique_assertion_ids(self.assertions)
        return self


class _EvalCaseAuthoringDraftMaterialV2(_EvalCaseAuthoringBase):
    assertions: tuple[EvalSuiteAuthoringAssertionDraftSpecV2, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE,
    )

    @field_validator("assertions", mode="before")
    @classmethod
    def validate_assertions_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("assertions")
    @classmethod
    def validate_assertions(
        cls,
        value: tuple[EvalSuiteAuthoringAssertionDraftSpecV2, ...],
    ) -> tuple[EvalSuiteAuthoringAssertionDraftSpecV2, ...]:
        validated: list[EvalSuiteAuthoringAssertionDraftSpecV2] = []
        for assertion in value:
            if type(assertion) is StructuredModelJudgeAssertionDraftV1:
                validated.append(
                    StructuredModelJudgeAssertionDraftV1.model_validate(
                        _model_python_input(assertion)
                    )
                )
            else:
                validated.append(
                    cast(
                        "EvalSuiteAuthoringAssertionDraftSpecV2",
                        _validated_assertion_spec(
                            cast("EvalSuiteAuthoringAssertionSpecV2", assertion)
                        ),
                    )
                )
        return tuple(validated)

    @model_validator(mode="after")
    def validate_assertion_ids(self) -> _EvalCaseAuthoringDraftMaterialV2:
        _validate_unique_assertion_ids(self.assertions)
        return self


class EvalCaseDraftV2(_EvalCaseAuthoringDraftMaterialV2):
    """Revision-free V2 case material accepted by authoring previews."""

    @classmethod
    def from_case(cls, case: EvalCaseDefinitionV1 | EvalCaseDefinitionV2) -> EvalCaseDraftV2:
        if type(case) not in {EvalCaseDefinitionV1, EvalCaseDefinitionV2}:
            raise TypeError("case must be an exact authored eval case definition.")
        if type(case) is EvalCaseDefinitionV1:
            validated: EvalCaseDefinition = EvalCaseDefinitionV1.model_validate(
                _model_python_input(case)
            )
        else:
            validated = EvalCaseDefinitionV2.model_validate(_model_python_input(case))
        assertions: list[EvalSuiteAuthoringAssertionDraftSpecV2] = []
        for assertion in validated.assertions:
            if type(assertion) is StructuredModelJudgeAssertionSpec:
                assertions.append(StructuredModelJudgeAssertionDraftV1.from_assertion(assertion))
            else:
                assertions.append(cast("EvalSuiteAuthoringAssertionDraftSpecV2", assertion))
        material = validated.model_dump(mode="python", exclude={"revision", "assertions"})
        return cls(assertions=tuple(assertions), **material)


class EvalCaseDefinitionV2(_EvalCaseAuthoringMaterialV2):
    """One immutable V2 case revision with structured-judge fidelity."""

    revision: StrictStr

    @field_validator("revision")
    @classmethod
    def validate_revision_shape(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @model_validator(mode="after")
    def validate_revision(self) -> EvalCaseDefinitionV2:
        expected = _model_content_revision(self, "authored eval case")
        if self.revision != expected:
            raise ValueError("Authored eval case revision does not match its content.")
        return self

    @classmethod
    def create(cls, draft: EvalCaseDraftV2) -> EvalCaseDefinitionV2:
        if type(draft) is not EvalCaseDraftV2:
            raise TypeError("draft must be an exact EvalCaseDraftV2.")
        validated = EvalCaseDraftV2.model_validate(_model_python_input(draft))
        compiled_assertions = tuple(
            assertion.compile()
            if type(assertion) is StructuredModelJudgeAssertionDraftV1
            else assertion
            for assertion in validated.assertions
        )
        document = validated.model_dump(mode="json", exclude={"assertions"})
        document["assertions"] = [
            assertion.model_dump(mode="json") for assertion in compiled_assertions
        ]
        return cls(
            revision=_content_revision(document, "authored eval case"),
            **document,
        )


class EvalSuiteDraftV1(_PortableModel):
    """Revision-free suite material edited by an SDK or Control Plane client."""

    id: StrictStr
    target_key: StrictStr
    name: StrictStr
    description: StrictStr | None = None
    trial_request: TrialRequestSpec = Field(default_factory=TrialRequestSpec)
    cases: tuple[EvalCaseDraftV1, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_CASES,
    )

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

    @field_validator("cases", mode="before")
    @classmethod
    def validate_cases_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("trial_request")
    @classmethod
    def validate_trial_request(cls, value: TrialRequestSpec) -> TrialRequestSpec:
        if type(value) is not TrialRequestSpec:
            raise TypeError("trial_request must be an exact TrialRequestSpec.")
        return TrialRequestSpec.model_validate(_model_python_input(value))

    @field_validator("cases")
    @classmethod
    def validate_cases(
        cls,
        value: tuple[EvalCaseDraftV1, ...],
    ) -> tuple[EvalCaseDraftV1, ...]:
        if any(type(case) is not EvalCaseDraftV1 for case in value):
            raise TypeError("cases must contain exact EvalCaseDraftV1 values.")
        return tuple(EvalCaseDraftV1.model_validate(_model_python_input(case)) for case in value)

    @model_validator(mode="after")
    def validate_case_ids(self) -> EvalSuiteDraftV1:
        ids = tuple(case.id for case in self.cases)
        duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
        if duplicates:
            raise ValueError(
                "Authored eval case IDs must be unique; duplicated: " + ", ".join(duplicates)
            )
        return self

    @classmethod
    def from_document(cls, document: EvalSuiteDocumentV1) -> EvalSuiteDraftV1:
        if type(document) is not EvalSuiteDocumentV1:
            raise TypeError("document must be an exact EvalSuiteDocumentV1.")
        validated = EvalSuiteDocumentV1.model_validate(_model_python_input(document))
        return cls(
            id=validated.suite.id,
            target_key=validated.target_key,
            name=validated.suite.name,
            description=validated.suite.description,
            trial_request=validated.suite.trial_request,
            cases=tuple(EvalCaseDraftV1.from_case(case) for case in validated.cases),
        )


class EvalSuiteDraftV2(_PortableModel):
    """Explicit V2 editor material with structured-judge authoring support."""

    schema_version: Literal[2] = EVAL_SUITE_AUTHORING_V2_SCHEMA_VERSION
    id: StrictStr
    target_key: StrictStr
    name: StrictStr
    description: StrictStr | None = None
    trial_request: TrialRequestSpec = Field(default_factory=TrialRequestSpec)
    cases: tuple[EvalCaseDraftV2, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_CASES,
    )

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

    @field_validator("cases", mode="before")
    @classmethod
    def validate_cases_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("trial_request")
    @classmethod
    def validate_trial_request(cls, value: TrialRequestSpec) -> TrialRequestSpec:
        if type(value) is not TrialRequestSpec:
            raise TypeError("trial_request must be an exact TrialRequestSpec.")
        return TrialRequestSpec.model_validate(_model_python_input(value))

    @field_validator("cases")
    @classmethod
    def validate_cases(
        cls,
        value: tuple[EvalCaseDraftV2, ...],
    ) -> tuple[EvalCaseDraftV2, ...]:
        if any(type(case) is not EvalCaseDraftV2 for case in value):
            raise TypeError("cases must contain exact EvalCaseDraftV2 values.")
        return tuple(EvalCaseDraftV2.model_validate(_model_python_input(case)) for case in value)

    @model_validator(mode="after")
    def validate_case_ids(self) -> EvalSuiteDraftV2:
        ids = tuple(case.id for case in self.cases)
        duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
        if duplicates:
            raise ValueError(
                "Authored eval case IDs must be unique; duplicated: " + ", ".join(duplicates)
            )
        return self

    @classmethod
    def from_document(cls, document: EvalSuiteDocumentV1 | EvalSuiteDocumentV2) -> EvalSuiteDraftV2:
        validated = _validated_eval_suite_document(document)
        return cls(
            id=validated.suite.id,
            target_key=validated.target_key,
            name=validated.suite.name,
            description=validated.suite.description,
            trial_request=validated.suite.trial_request,
            cases=tuple(EvalCaseDraftV2.from_case(case) for case in validated.cases),
        )


class EvalSuiteDocumentV1(_SchemaV1PortableModel):
    """One immutable, authority-free authored suite and its exact case revisions."""

    schema_version: Literal[1] = EVAL_SUITE_AUTHORING_SCHEMA_VERSION
    revision: StrictStr
    target_key: StrictStr
    suite: EvalSuiteSpec
    cases: tuple[EvalCaseDefinitionV1, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_CASES,
    )

    @field_validator("revision")
    @classmethod
    def validate_revision_shape(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("target_key")
    @classmethod
    def validate_target_key(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("cases", mode="before")
    @classmethod
    def validate_cases_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("suite")
    @classmethod
    def validate_suite(cls, value: EvalSuiteSpec) -> EvalSuiteSpec:
        if type(value) is not EvalSuiteSpec:
            raise TypeError("suite must be an exact EvalSuiteSpec.")
        return EvalSuiteSpec.model_validate(_model_python_input(value))

    @field_validator("cases")
    @classmethod
    def validate_cases(
        cls,
        value: tuple[EvalCaseDefinitionV1, ...],
    ) -> tuple[EvalCaseDefinitionV1, ...]:
        if any(type(case) is not EvalCaseDefinitionV1 for case in value):
            raise TypeError("cases must contain exact EvalCaseDefinitionV1 values.")
        return tuple(
            EvalCaseDefinitionV1.model_validate(_model_python_input(case)) for case in value
        )

    @model_validator(mode="after")
    def validate_contract(self) -> EvalSuiteDocumentV1:
        case_ids = tuple(case.id for case in self.cases)
        if case_ids != tuple(sorted(case_ids)):
            raise ValueError("Authored eval cases must be sorted by id.")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Authored eval case IDs must be unique.")
        published_results = (
            sum(len(case.assertions) for case in self.cases) * self.suite.trial_request.trials
        )
        if published_results > EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS:
            raise ValueError(
                "Authored eval suite expands to "
                f"{published_results} published assertion results; the maximum is "
                f"{EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS}."
            )
        if not json_utf8_size_within_limit(self, EVAL_SUITE_AUTHORING_MAX_BYTES):
            raise ValueError(
                f"Authored eval suite exceeds {EVAL_SUITE_AUTHORING_MAX_BYTES} "
                "canonical JSON bytes."
            )
        if not _pretty_json_size_within_limit(self, EVAL_SUITE_AUTHORING_MAX_BYTES):
            raise ValueError(
                f"Authored eval suite exceeds {EVAL_SUITE_AUTHORING_MAX_BYTES} "
                "serialized JSON bytes."
            )
        expected = _model_content_revision(self, "authored eval suite")
        if self.revision != expected:
            raise ValueError("Authored eval suite revision does not match its content.")
        return self


class EvalSuiteDocumentV2(_PortableModel):
    """Immutable V2 suite preserving structured rubric and judge contracts."""

    schema_version: Literal[2] = EVAL_SUITE_AUTHORING_V2_SCHEMA_VERSION
    revision: StrictStr
    target_key: StrictStr
    suite: EvalSuiteSpec
    cases: tuple[EvalCaseDefinitionV2, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_CASES,
    )

    @field_validator("revision")
    @classmethod
    def validate_revision_shape(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("target_key")
    @classmethod
    def validate_target_key(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("cases", mode="before")
    @classmethod
    def validate_cases_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("suite")
    @classmethod
    def validate_suite(cls, value: EvalSuiteSpec) -> EvalSuiteSpec:
        if type(value) is not EvalSuiteSpec:
            raise TypeError("suite must be an exact EvalSuiteSpec.")
        return EvalSuiteSpec.model_validate(_model_python_input(value))

    @field_validator("cases")
    @classmethod
    def validate_cases(
        cls,
        value: tuple[EvalCaseDefinitionV2, ...],
    ) -> tuple[EvalCaseDefinitionV2, ...]:
        if any(type(case) is not EvalCaseDefinitionV2 for case in value):
            raise TypeError("cases must contain exact EvalCaseDefinitionV2 values.")
        return tuple(
            EvalCaseDefinitionV2.model_validate(_model_python_input(case)) for case in value
        )

    @model_validator(mode="after")
    def validate_contract(self) -> EvalSuiteDocumentV2:
        _validate_eval_suite_document_contract(self)
        return self


EvalSuiteDocument: TypeAlias = EvalSuiteDocumentV1 | EvalSuiteDocumentV2
EvalSuiteDraft: TypeAlias = EvalSuiteDraftV1 | EvalSuiteDraftV2
EvalCaseDefinition: TypeAlias = EvalCaseDefinitionV1 | EvalCaseDefinitionV2


def _validate_eval_suite_document_contract(document: EvalSuiteDocument) -> None:
    case_ids = tuple(case.id for case in document.cases)
    if case_ids != tuple(sorted(case_ids)):
        raise ValueError("Authored eval cases must be sorted by id.")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Authored eval case IDs must be unique.")
    published_results = (
        sum(len(case.assertions) for case in document.cases) * document.suite.trial_request.trials
    )
    if published_results > EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS:
        raise ValueError(
            "Authored eval suite expands to "
            f"{published_results} published assertion results; the maximum is "
            f"{EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS}."
        )
    published_explanation_slots = sum(
        len(assertion.rubric.criteria)
        for case in document.cases
        for assertion in case.assertions
        if type(assertion) is StructuredModelJudgeAssertionSpec
        and type(assertion.reference) is not PrivateJudgeReferenceV1
    )
    explanation_chars = (
        published_explanation_slots
        * document.suite.trial_request.trials
        * EVAL_CORPUS_MAX_JUDGE_EXPLANATION_CHARS
    )
    if explanation_chars > EVAL_CORPUS_MAX_PUBLISHED_JUDGE_EXPLANATION_CHARS:
        raise ValueError(
            "Authored eval suite permits "
            f"{explanation_chars} published judge explanation characters; the maximum is "
            f"{EVAL_CORPUS_MAX_PUBLISHED_JUDGE_EXPLANATION_CHARS}."
        )
    if not json_utf8_size_within_limit(document, EVAL_SUITE_AUTHORING_MAX_BYTES):
        raise ValueError(
            f"Authored eval suite exceeds {EVAL_SUITE_AUTHORING_MAX_BYTES} canonical JSON bytes."
        )
    if not _pretty_json_size_within_limit(document, EVAL_SUITE_AUTHORING_MAX_BYTES):
        raise ValueError(
            f"Authored eval suite exceeds {EVAL_SUITE_AUTHORING_MAX_BYTES} serialized JSON bytes."
        )
    expected = _model_content_revision(document, "authored eval suite")
    if document.revision != expected:
        raise ValueError("Authored eval suite revision does not match its content.")


def _validated_eval_suite_document(document: EvalSuiteDocument) -> EvalSuiteDocument:
    if type(document) is EvalSuiteDocumentV1:
        return EvalSuiteDocumentV1.model_validate(_model_python_input(document))
    if type(document) is EvalSuiteDocumentV2:
        return EvalSuiteDocumentV2.model_validate(_model_python_input(document))
    raise TypeError("document must be an exact authored eval suite document.")


def compile_eval_suite_draft(draft: EvalSuiteDraftV1) -> EvalSuiteDocumentV1:
    """Canonicalize bounded editor material into one immutable suite revision."""

    if type(draft) is not EvalSuiteDraftV1:
        raise TypeError("draft must be an exact EvalSuiteDraftV1.")
    validated = EvalSuiteDraftV1.model_validate(_model_python_input(draft))
    suite = EvalSuiteSpec.create(
        id=validated.id,
        name=validated.name,
        description=validated.description,
        trial_request=validated.trial_request,
    )
    cases = tuple(
        sorted(
            (EvalCaseDefinitionV1.create(case) for case in validated.cases),
            key=lambda case: case.id,
        )
    )
    document: dict[str, Any] = {
        "schema_version": EVAL_SUITE_AUTHORING_SCHEMA_VERSION,
        "target_key": validated.target_key,
        "suite": suite.model_dump(mode="json"),
        "cases": [case.model_dump(mode="json") for case in cases],
    }
    return EvalSuiteDocumentV1(
        revision=_content_revision(document, "authored eval suite"),
        **document,
    )


def compile_eval_suite_draft_v2(draft: EvalSuiteDraftV2) -> EvalSuiteDocumentV2:
    """Canonicalize V2 editor material without losing structured judge fields."""

    if type(draft) is not EvalSuiteDraftV2:
        raise TypeError("draft must be an exact EvalSuiteDraftV2.")
    validated = EvalSuiteDraftV2.model_validate(_model_python_input(draft))
    suite = EvalSuiteSpec.create(
        id=validated.id,
        name=validated.name,
        description=validated.description,
        trial_request=validated.trial_request,
    )
    cases = tuple(
        sorted(
            (EvalCaseDefinitionV2.create(case) for case in validated.cases),
            key=lambda case: case.id,
        )
    )
    document: dict[str, Any] = {
        "schema_version": EVAL_SUITE_AUTHORING_V2_SCHEMA_VERSION,
        "target_key": validated.target_key,
        "suite": suite.model_dump(mode="json"),
        "cases": [case.model_dump(mode="json") for case in cases],
    }
    return EvalSuiteDocumentV2(
        revision=_content_revision(document, "authored eval suite"),
        **document,
    )


def compile_eval_suite_authoring_draft(draft: EvalSuiteDraft) -> EvalSuiteDocument:
    """Compile the exact draft wire version selected by the caller."""

    if type(draft) is EvalSuiteDraftV1:
        return compile_eval_suite_draft(draft)
    if type(draft) is EvalSuiteDraftV2:
        return compile_eval_suite_draft_v2(draft)
    raise TypeError("draft must be an exact authored eval suite draft.")


@overload
def validate_expected_eval_suite_revision(
    document: EvalSuiteDocumentV1,
    expected_revision: str,
) -> EvalSuiteDocumentV1: ...


@overload
def validate_expected_eval_suite_revision(
    document: EvalSuiteDocumentV2,
    expected_revision: str,
) -> EvalSuiteDocumentV2: ...


def validate_expected_eval_suite_revision(
    document: EvalSuiteDocument,
    expected_revision: str,
) -> EvalSuiteDocument:
    """Reject stale save or mutation requests before durable side effects."""

    expected_revision = _sha256_revision(expected_revision, "expected_revision")
    validated = _validated_eval_suite_document(document)
    if validated.revision != expected_revision:
        raise ValueError("Authored eval suite changed after the reviewed revision.")
    return validated


def _document_from_parts(
    original: EvalSuiteDocumentV1,
    cases: Sequence[EvalCaseDefinitionV1],
) -> EvalSuiteDocumentV1:
    draft = EvalSuiteDraftV1(
        id=original.suite.id,
        target_key=original.target_key,
        name=original.suite.name,
        description=original.suite.description,
        trial_request=original.suite.trial_request,
        cases=tuple(EvalCaseDraftV1.from_case(case) for case in cases),
    )
    return compile_eval_suite_draft(draft)


def add_eval_case(
    document: EvalSuiteDocumentV1,
    case: EvalCaseDraftV1,
) -> EvalSuiteDocumentV1:
    """Return a new suite revision containing one new stable case ID."""

    if type(document) is not EvalSuiteDocumentV1:
        raise TypeError("document must be an exact EvalSuiteDocumentV1.")
    validated = validate_expected_eval_suite_revision(document, document.revision)
    if type(case) is not EvalCaseDraftV1:
        raise TypeError("case must be an exact EvalCaseDraftV1.")
    added = EvalCaseDefinitionV1.create(case)
    if any(current.id == added.id for current in validated.cases):
        raise ValueError(f"Authored eval case already exists: {added.id}")
    return _document_from_parts(validated, (*validated.cases, added))


def duplicate_eval_case(
    document: EvalSuiteDocumentV1,
    case_id: str,
    *,
    new_case_id: str,
    new_name: str | None = None,
) -> EvalSuiteDocumentV1:
    """Copy one case under an explicit new identity without aliasing history."""

    if type(document) is not EvalSuiteDocumentV1:
        raise TypeError("document must be an exact EvalSuiteDocumentV1.")
    validated = validate_expected_eval_suite_revision(document, document.revision)
    case_id = _portable_id(case_id, "case_id")
    new_case_id = _portable_id(new_case_id, "new_case_id")
    source = next((case for case in validated.cases if case.id == case_id), None)
    if source is None:
        raise KeyError(f"Authored eval case not found: {case_id}")
    duplicate = EvalCaseDraftV1(
        id=new_case_id,
        name=source.name if new_name is None else new_name,
        description=source.description,
        source=source.source,
        stimulus=source.stimulus,
        assertions=source.assertions,
    )
    return add_eval_case(validated, duplicate)


def revise_eval_case(
    document: EvalSuiteDocumentV1,
    replacement: EvalCaseDraftV1,
    *,
    expected_case_revision: str,
) -> EvalSuiteDocumentV1:
    """Replace one case only when the caller reviewed its current revision."""

    if type(document) is not EvalSuiteDocumentV1:
        raise TypeError("document must be an exact EvalSuiteDocumentV1.")
    validated = validate_expected_eval_suite_revision(document, document.revision)
    if type(replacement) is not EvalCaseDraftV1:
        raise TypeError("replacement must be an exact EvalCaseDraftV1.")
    expected_case_revision = _sha256_revision(
        expected_case_revision,
        "expected_case_revision",
    )
    current = next(
        (case for case in validated.cases if case.id == replacement.id),
        None,
    )
    if current is None:
        raise KeyError(f"Authored eval case not found: {replacement.id}")
    if current.revision != expected_case_revision:
        raise ValueError("Authored eval case changed after the reviewed revision.")
    revised = EvalCaseDefinitionV1.create(replacement)
    return _document_from_parts(
        validated,
        tuple(revised if case.id == revised.id else case for case in validated.cases),
    )


class EvalSelectedCaseV1(_PortableModel):
    id: StrictStr
    revision: StrictStr

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)


class EvalSuiteSelectionV1(_SchemaV1PortableModel):
    """Content identity for a full-suite or explicit-subset launch request."""

    schema_version: Literal[1] = EVAL_SUITE_SELECTION_SCHEMA_VERSION
    revision: StrictStr
    suite_document_revision: StrictStr
    suite_id: StrictStr
    suite_revision: StrictStr
    mode: Literal["full_suite", "subset"]
    cases: tuple[EvalSelectedCaseV1, ...] = Field(min_length=1, max_length=EVAL_CORPUS_MAX_CASES)

    @field_validator("revision", "suite_document_revision", "suite_revision")
    @classmethod
    def validate_revisions(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("suite_id")
    @classmethod
    def validate_suite_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("cases", mode="before")
    @classmethod
    def validate_cases_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("cases")
    @classmethod
    def validate_cases(
        cls,
        value: tuple[EvalSelectedCaseV1, ...],
    ) -> tuple[EvalSelectedCaseV1, ...]:
        if any(type(case) is not EvalSelectedCaseV1 for case in value):
            raise TypeError("cases must contain exact EvalSelectedCaseV1 values.")
        return tuple(EvalSelectedCaseV1.model_validate(_model_python_input(case)) for case in value)

    @model_validator(mode="after")
    def validate_contract(self) -> EvalSuiteSelectionV1:
        ids = tuple(case.id for case in self.cases)
        if ids != tuple(sorted(ids)):
            raise ValueError("Selected eval cases must be sorted by id.")
        if len(ids) != len(set(ids)):
            raise ValueError("Selected eval case IDs must be unique.")
        expected = _model_content_revision(self, "eval suite selection")
        if self.revision != expected:
            raise ValueError("Eval suite selection revision does not match its content.")
        return self


def eval_suite_selection(
    document: EvalSuiteDocument,
    case_ids: Sequence[str] | None = None,
) -> EvalSuiteSelectionV1:
    """Freeze either every case or an explicit non-empty subset by revision."""

    validated = validate_expected_eval_suite_revision(document, document.revision)
    cases_by_id = {case.id: case for case in validated.cases}
    if case_ids is None:
        mode: Literal["full_suite", "subset"] = "full_suite"
        selected = validated.cases
    else:
        raw_ids = _ordered_sequence_argument(case_ids, "case_ids")
        normalized = tuple(_portable_id(case_id, "case_ids") for case_id in raw_ids)
        if not normalized:
            raise ValueError("case_ids must contain at least one case.")
        if len(normalized) != len(set(normalized)):
            raise ValueError("case_ids must be unique.")
        unknown = sorted(set(normalized) - set(cases_by_id))
        if unknown:
            raise KeyError("Authored eval cases not found: " + ", ".join(unknown))
        mode = "subset"
        selected = tuple(cases_by_id[case_id] for case_id in sorted(normalized))
    selected_cases = tuple(
        EvalSelectedCaseV1(id=case.id, revision=case.revision) for case in selected
    )
    material: dict[str, Any] = {
        "schema_version": EVAL_SUITE_SELECTION_SCHEMA_VERSION,
        "suite_document_revision": validated.revision,
        "suite_id": validated.suite.id,
        "suite_revision": validated.suite.revision,
        "mode": mode,
        "cases": [case.model_dump(mode="json") for case in selected_cases],
    }
    selection = EvalSuiteSelectionV1(
        revision=_content_revision(material, "eval suite selection"),
        **material,
    )
    return validate_eval_suite_selection(selection, validated)


def validate_eval_suite_selection(
    selection: EvalSuiteSelectionV1,
    document: EvalSuiteDocument,
) -> EvalSuiteSelectionV1:
    """Match one selection to the exact immutable suite and case revisions."""

    if type(selection) is not EvalSuiteSelectionV1:
        raise TypeError("selection must be an exact EvalSuiteSelectionV1.")
    validated_selection = EvalSuiteSelectionV1.model_validate(_model_python_input(selection))
    validated_document = validate_expected_eval_suite_revision(
        document,
        document.revision,
    )
    if (
        validated_selection.suite_document_revision != validated_document.revision
        or validated_selection.suite_id != validated_document.suite.id
        or validated_selection.suite_revision != validated_document.suite.revision
    ):
        raise ValueError("Eval suite selection does not match its immutable suite.")
    document_cases = {case.id: case.revision for case in validated_document.cases}
    selected_cases = {case.id: case.revision for case in validated_selection.cases}
    if any(document_cases.get(case_id) != revision for case_id, revision in selected_cases.items()):
        raise ValueError("Eval suite selection contains an unknown or changed case revision.")
    if validated_selection.mode == "full_suite" and selected_cases != document_cases:
        raise ValueError("Full-suite selection must contain every immutable case revision.")
    return validated_selection


def eval_suite_document_to_json(document: EvalSuiteDocument) -> str:
    """Return deterministic, human-readable authored-suite JSON."""

    validated = validate_expected_eval_suite_revision(document, document.revision)
    rendered = (
        json.dumps(
            validated.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if len(rendered.encode("utf-8")) > EVAL_SUITE_AUTHORING_MAX_BYTES:
        raise ValueError(
            f"Authored eval suite JSON exceeds {EVAL_SUITE_AUTHORING_MAX_BYTES} bytes."
        )
    return rendered


def eval_suite_document_from_json(source: str) -> EvalSuiteDocument:
    """Load one bounded authored-suite document from strict JSON text."""

    if type(source) is not str:
        raise TypeError("eval_suite_document_from_json requires text.")
    if len(source) > EVAL_SUITE_AUTHORING_MAX_BYTES:
        raise ValueError(
            f"Authored eval suite JSON exceeds {EVAL_SUITE_AUTHORING_MAX_BYTES} bytes."
        )
    try:
        raw = source.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "Authored eval suite JSON must contain valid Unicode scalar text."
        ) from exc
    if len(raw) > EVAL_SUITE_AUTHORING_MAX_BYTES:
        raise ValueError(
            f"Authored eval suite JSON exceeds {EVAL_SUITE_AUTHORING_MAX_BYTES} bytes."
        )
    try:
        decoded = json.loads(
            source,
            parse_int=partial(
                parse_durable_json_integer_literal,
                field_name="authored eval suite JSON",
            ),
            parse_constant=partial(
                reject_nonportable_json_constant,
                field_name="authored eval suite JSON",
            ),
            object_pairs_hook=partial(
                durable_json_object_from_pairs,
                field_name="authored eval suite JSON",
            ),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Authored eval suite JSON is invalid.") from exc
    if type(decoded) is not dict:
        raise ValueError("Authored eval suite JSON must be an object.")
    schema_version = decoded.get("schema_version")
    if schema_version == 1:
        return EvalSuiteDocumentV1.model_validate(decoded)
    if schema_version == 2:
        return EvalSuiteDocumentV2.model_validate(decoded)
    raise ValueError("Unsupported authored eval suite schema version.")


__all__ = [
    "EVAL_SUITE_AUTHORING_MAX_BYTES",
    "EVAL_SUITE_AUTHORING_SCHEMA_VERSION",
    "EVAL_SUITE_AUTHORING_V2_SCHEMA_VERSION",
    "EVAL_SUITE_SELECTION_SCHEMA_VERSION",
    "EvalCaseDefinitionV1",
    "EvalCaseDefinitionV2",
    "EvalCaseDraftV1",
    "EvalCaseDraftV2",
    "EvalCaseStimulusV1",
    "EvalScenarioStimulusV1",
    "EvalSelectedCaseV1",
    "EvalSimpleInputStimulusV1",
    "EvalSuiteDocumentV1",
    "EvalSuiteDocumentV2",
    "EvalSuiteDraftV1",
    "EvalSuiteDraftV2",
    "EvalSuiteSelectionV1",
    "PublicJudgeReferenceDraftV1",
    "StructuredModelJudgeAssertionDraftV1",
    "StructuredRubricDraftV1",
    "add_eval_case",
    "compile_eval_suite_authoring_draft",
    "compile_eval_suite_draft",
    "compile_eval_suite_draft_v2",
    "duplicate_eval_case",
    "eval_suite_document_from_json",
    "eval_suite_document_to_json",
    "eval_suite_selection",
    "revise_eval_case",
    "validate_eval_suite_selection",
    "validate_expected_eval_suite_revision",
]
