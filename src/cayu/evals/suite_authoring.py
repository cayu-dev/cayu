from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from functools import partial
from typing import Annotated, Any, Literal, TypeAlias

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
    EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS,
    AssertionSpec,
    EvalSuiteSpec,
    EvaluationSourceIdentityV1,
    RunInputSpec,
    TrialRequestSpec,
    _bounded_durable_text,
    _content_revision,
    _model_content_revision,
    _model_python_input,
    _ordered_sequence_argument,
    _ordered_sequence_input,
    _portable_id,
    _PortableModel,
    _pretty_json_size_within_limit,
    _SchemaV1PortableModel,
    _sha256_revision,
    _validated_assertion_spec,
)

EVAL_SUITE_AUTHORING_SCHEMA_VERSION = 1
EVAL_SUITE_SELECTION_SCHEMA_VERSION = 1
EVAL_SUITE_AUTHORING_MAX_BYTES = EVAL_CORPUS_MAX_BYTES


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


class _EvalCaseAuthoringMaterial(_PortableModel):
    id: StrictStr
    name: StrictStr
    description: StrictStr | None = None
    source: EvaluationSourceIdentityV1 | None = None
    stimulus: EvalCaseStimulusV1
    assertions: tuple[AssertionSpec, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE,
    )

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

    @field_validator("assertions", mode="before")
    @classmethod
    def validate_assertions_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

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

    @field_validator("assertions")
    @classmethod
    def validate_assertions(
        cls,
        value: tuple[AssertionSpec, ...],
    ) -> tuple[AssertionSpec, ...]:
        return tuple(_validated_assertion_spec(assertion) for assertion in value)

    @model_validator(mode="after")
    def validate_assertion_ids(self):
        ids = tuple(assertion.id for assertion in self.assertions)
        duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
        if duplicates:
            raise ValueError(
                "Eval case assertion IDs must be unique; duplicated: " + ", ".join(duplicates)
            )
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


def validate_expected_eval_suite_revision(
    document: EvalSuiteDocumentV1,
    expected_revision: str,
) -> EvalSuiteDocumentV1:
    """Reject stale save or mutation requests before durable side effects."""

    if type(document) is not EvalSuiteDocumentV1:
        raise TypeError("document must be an exact EvalSuiteDocumentV1.")
    expected_revision = _sha256_revision(expected_revision, "expected_revision")
    validated = EvalSuiteDocumentV1.model_validate(_model_python_input(document))
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
    document: EvalSuiteDocumentV1,
    case_ids: Sequence[str] | None = None,
) -> EvalSuiteSelectionV1:
    """Freeze either every case or an explicit non-empty subset by revision."""

    if type(document) is not EvalSuiteDocumentV1:
        raise TypeError("document must be an exact EvalSuiteDocumentV1.")
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
    document: EvalSuiteDocumentV1,
) -> EvalSuiteSelectionV1:
    """Match one selection to the exact immutable suite and case revisions."""

    if type(selection) is not EvalSuiteSelectionV1:
        raise TypeError("selection must be an exact EvalSuiteSelectionV1.")
    if type(document) is not EvalSuiteDocumentV1:
        raise TypeError("document must be an exact EvalSuiteDocumentV1.")
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


def eval_suite_document_to_json(document: EvalSuiteDocumentV1) -> str:
    """Return deterministic, human-readable authored-suite JSON."""

    if type(document) is not EvalSuiteDocumentV1:
        raise TypeError("document must be an exact EvalSuiteDocumentV1.")
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


def eval_suite_document_from_json(source: str) -> EvalSuiteDocumentV1:
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
    return EvalSuiteDocumentV1.model_validate(decoded)


__all__ = [
    "EVAL_SUITE_AUTHORING_MAX_BYTES",
    "EVAL_SUITE_AUTHORING_SCHEMA_VERSION",
    "EVAL_SUITE_SELECTION_SCHEMA_VERSION",
    "EvalCaseDefinitionV1",
    "EvalCaseDraftV1",
    "EvalCaseStimulusV1",
    "EvalScenarioStimulusV1",
    "EvalSelectedCaseV1",
    "EvalSimpleInputStimulusV1",
    "EvalSuiteDocumentV1",
    "EvalSuiteDraftV1",
    "EvalSuiteSelectionV1",
    "add_eval_case",
    "compile_eval_suite_draft",
    "duplicate_eval_case",
    "eval_suite_document_from_json",
    "eval_suite_document_to_json",
    "eval_suite_selection",
    "revise_eval_case",
    "validate_eval_suite_selection",
    "validate_expected_eval_suite_revision",
]
