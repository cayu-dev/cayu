from __future__ import annotations

import json
from collections.abc import Sequence
from enum import StrEnum
from functools import partial
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    StrictBool,
    StrictFloat,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._validation import (
    copy_durable_json_object,
    durable_json_object_from_pairs,
    json_utf8_size_within_limit,
    parse_durable_json_integer_literal,
    reject_nonportable_json_constant,
)
from cayu.evals.corpus import (
    EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE,
    EVAL_CORPUS_MAX_CASES,
    EvalCorpusDocument,
    _bounded_durable_text,
    _content_revision,
    _model_python_input,
    _ordered_sequence_argument,
    _ordered_sequence_input,
    _portable_id,
    _SchemaV1PortableModel,
    _sha256_hex,
    _sha256_revision,
    assertion_spec_revision,
)
from cayu.evals.execution import CorpusExecutionResult, EvaluationTargetIdentity
from cayu.evals.promotion import CapturedRunScoreV1
from cayu.evals.published import (
    PublishedAssertionResult,
    PublishedStatus,
    _assertion_contract,
    _published_score,
    _published_status_from_statuses,
)

CAPTURED_EVALUATION_RESULT_SCHEMA_VERSION = 1
CAPTURED_EVALUATION_RESULT_MAX_BYTES = 4 << 20
EVAL_RESULT_PROJECTION_SCHEMA_VERSION = 1
EVAL_RESULT_PROJECTION_MAX_BYTES = 4 << 20


class EvalResultOrigin(StrEnum):
    """How one immutable published result was produced."""

    CAPTURED_SESSION = "captured_session"
    FRESH_EXECUTION = "fresh_execution"


class EvalResultTargetIdentityV1(_SchemaV1PortableModel):
    """Comparable public target identity without executable application authority."""

    schema_version: Literal[1] = 1
    target_key: StrictStr
    application_release_id: StrictStr
    app_manifest_schema_version: StrictStr
    app_manifest_fingerprint: StrictStr

    @field_validator("target_key")
    @classmethod
    def validate_target_key(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("application_release_id")
    @classmethod
    def validate_release_id(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @field_validator("app_manifest_schema_version")
    @classmethod
    def validate_manifest_schema_version(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=32,
            nonblank=True,
            clean=True,
        )

    @field_validator("app_manifest_fingerprint")
    @classmethod
    def validate_manifest_fingerprint(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @classmethod
    def from_fresh_target(
        cls,
        target: EvaluationTargetIdentity,
    ) -> EvalResultTargetIdentityV1:
        if type(target) is not EvaluationTargetIdentity:
            raise TypeError("target must be an exact EvaluationTargetIdentity.")
        validated = EvaluationTargetIdentity.model_validate(_model_python_input(target))
        return cls(
            target_key=validated.target_key,
            application_release_id=validated.application_release_id,
            app_manifest_schema_version=validated.app_manifest_schema_version,
            app_manifest_fingerprint=validated.app_manifest_fingerprint,
        )


class CapturedEvaluationResultV1(_SchemaV1PortableModel):
    """Immutable public result scored only from retained session evidence.

    The document deliberately contains no session id, store handle, application
    object, credential, or historical execution authority. Its corpus revision
    supplies the separately persisted expectation and input contract.
    """

    schema_version: Literal[1] = CAPTURED_EVALUATION_RESULT_SCHEMA_VERSION
    revision: StrictStr
    origin: Literal[EvalResultOrigin.CAPTURED_SESSION] = EvalResultOrigin.CAPTURED_SESSION
    target: EvalResultTargetIdentityV1
    corpus_revision: StrictStr
    suite_id: StrictStr
    suite_revision: StrictStr
    score: CapturedRunScoreV1

    @field_validator("revision", "corpus_revision", "suite_revision")
    @classmethod
    def validate_revisions(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("suite_id")
    @classmethod
    def validate_suite_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("target", mode="before")
    @classmethod
    def copy_target(cls, value: object) -> object:
        if type(value) is EvalResultTargetIdentityV1:
            return EvalResultTargetIdentityV1.model_validate(_model_python_input(value))
        if isinstance(value, BaseModel):
            raise TypeError("target must be an exact EvalResultTargetIdentityV1 or JSON object.")
        return value

    @field_validator("score", mode="before")
    @classmethod
    def copy_score(cls, value: object) -> object:
        if type(value) is CapturedRunScoreV1:
            return CapturedRunScoreV1.model_validate(_model_python_input(value))
        if isinstance(value, BaseModel):
            raise TypeError("score must be an exact CapturedRunScoreV1 or JSON object.")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> CapturedEvaluationResultV1:
        if not json_utf8_size_within_limit(self, CAPTURED_EVALUATION_RESULT_MAX_BYTES):
            raise ValueError(
                "Captured evaluation result exceeds "
                f"{CAPTURED_EVALUATION_RESULT_MAX_BYTES} canonical JSON bytes."
            )
        expected = _content_revision(
            self.model_dump(mode="json"),
            "captured evaluation result",
        )
        if self.revision != expected:
            raise ValueError("Captured evaluation result revision does not match its content.")
        return self

    @classmethod
    def create(
        cls,
        *,
        corpus: EvalCorpusDocument,
        target: EvalResultTargetIdentityV1,
        score: CapturedRunScoreV1,
    ) -> CapturedEvaluationResultV1:
        if type(corpus) is not EvalCorpusDocument:
            raise TypeError("corpus must be an exact EvalCorpusDocument.")
        if type(target) is not EvalResultTargetIdentityV1:
            raise TypeError("target must be an exact EvalResultTargetIdentityV1.")
        if type(score) is not CapturedRunScoreV1:
            raise TypeError("score must be an exact CapturedRunScoreV1.")
        validated_corpus = EvalCorpusDocument.model_validate(_model_python_input(corpus))
        validated_target = EvalResultTargetIdentityV1.model_validate(_model_python_input(target))
        validated_score = CapturedRunScoreV1.model_validate(_model_python_input(score))
        suite = next(
            (
                item
                for item in validated_corpus.suites
                if any(
                    case.id == validated_score.case_id and case.suite_id == item.id
                    for case in validated_corpus.cases
                )
            ),
            None,
        )
        if suite is None:
            raise ValueError("Captured score case is absent from the immutable corpus.")
        document = {
            "schema_version": CAPTURED_EVALUATION_RESULT_SCHEMA_VERSION,
            "origin": EvalResultOrigin.CAPTURED_SESSION.value,
            "target": validated_target.model_dump(mode="json"),
            "corpus_revision": validated_corpus.revision,
            "suite_id": suite.id,
            "suite_revision": suite.revision,
            "score": validated_score.model_dump(mode="json"),
        }
        result = cls(
            revision=_content_revision(document, "captured evaluation result"),
            target=validated_target,
            corpus_revision=validated_corpus.revision,
            suite_id=suite.id,
            suite_revision=suite.revision,
            score=validated_score,
        )
        validate_captured_result_for_corpus(result, validated_corpus)
        return result


class EvalResultAssertionIdentityV1(_SchemaV1PortableModel):
    schema_version: Literal[1] = 1
    assertion_id: StrictStr
    assertion_revision: StrictStr
    comparison_revision: StrictStr

    @field_validator("assertion_id")
    @classmethod
    def validate_assertion_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("assertion_revision", "comparison_revision")
    @classmethod
    def validate_assertion_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)


class EvalResultCaseProjectionV1(_SchemaV1PortableModel):
    schema_version: Literal[1] = 1
    case_id: StrictStr
    case_revision: StrictStr
    status: PublishedStatus
    score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    assertions: tuple[EvalResultAssertionIdentityV1, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE,
    )

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("case_revision")
    @classmethod
    def validate_case_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("assertions", mode="before")
    @classmethod
    def validate_assertion_order(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @model_validator(mode="after")
    def validate_case_contract(self) -> EvalResultCaseProjectionV1:
        if (self.status in {"passed", "failed"}) != (self.score is not None):
            raise ValueError("Eval result case status contradicts its score.")
        assertion_ids = tuple(item.assertion_id for item in self.assertions)
        if len(assertion_ids) != len(set(assertion_ids)):
            raise ValueError("Eval result case assertion identities must be unique.")
        return self


class EvalResultProjectionV1(_SchemaV1PortableModel):
    """The single comparison projection shared by captured and fresh results."""

    schema_version: Literal[1] = EVAL_RESULT_PROJECTION_SCHEMA_VERSION
    result_revision: StrictStr
    origin: EvalResultOrigin
    target: EvalResultTargetIdentityV1
    corpus_revision: StrictStr
    suite_id: StrictStr
    suite_revision: StrictStr
    evidence_policy_revision: StrictStr
    pricing_profile_fingerprint: StrictStr | None = None
    uses_pricing: StrictBool
    status: PublishedStatus
    score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    cases: tuple[EvalResultCaseProjectionV1, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_CASES,
    )

    @field_validator(
        "result_revision",
        "corpus_revision",
        "suite_revision",
        "evidence_policy_revision",
        "pricing_profile_fingerprint",
    )
    @classmethod
    def validate_revisions(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_revision(value, info.field_name)

    @field_validator("suite_id")
    @classmethod
    def validate_suite_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("cases", mode="before")
    @classmethod
    def validate_case_order(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @model_validator(mode="after")
    def validate_projection(self) -> EvalResultProjectionV1:
        case_ids = tuple(case.case_id for case in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Eval result projection case identities must be unique.")
        expected_status = _published_status_from_statuses(case.status for case in self.cases)
        expected_score = _published_score(case.score for case in self.cases)
        if self.status != expected_status or self.score != expected_score:
            raise ValueError("Eval result projection aggregates do not match its cases.")
        if not json_utf8_size_within_limit(self, EVAL_RESULT_PROJECTION_MAX_BYTES):
            raise ValueError(
                "Eval result projection exceeds "
                f"{EVAL_RESULT_PROJECTION_MAX_BYTES} canonical JSON bytes."
            )
        return self


def _assertion_identities(
    assertions: Sequence[PublishedAssertionResult],
) -> tuple[EvalResultAssertionIdentityV1, ...]:
    ordered = _ordered_sequence_argument(assertions, "assertions")
    return tuple(
        EvalResultAssertionIdentityV1(
            assertion_id=assertion.assertion_id,
            assertion_revision=assertion.assertion_revision,
            comparison_revision=_content_revision(
                {"contract": list(_assertion_contract(assertion))},
                "published assertion comparison contract",
            ),
        )
        for assertion in ordered
    )


def validate_captured_result_for_corpus(
    result: CapturedEvaluationResultV1,
    corpus: EvalCorpusDocument,
) -> CapturedEvaluationResultV1:
    """Bind one captured score to the exact immutable corpus it was saved with."""

    if type(result) is not CapturedEvaluationResultV1:
        raise TypeError("result must be an exact CapturedEvaluationResultV1.")
    if type(corpus) is not EvalCorpusDocument:
        raise TypeError("corpus must be an exact EvalCorpusDocument.")
    validated = CapturedEvaluationResultV1.model_validate(_model_python_input(result))
    validated_corpus = EvalCorpusDocument.model_validate(_model_python_input(corpus))
    if validated.corpus_revision != validated_corpus.revision:
        raise ValueError("Captured result corpus revision does not match stored content.")
    if validated.target.target_key != validated_corpus.target_key:
        raise ValueError("Captured result target key does not match its corpus.")
    suite = next(
        (item for item in validated_corpus.suites if item.id == validated.suite_id),
        None,
    )
    if suite is None or suite.revision != validated.suite_revision:
        raise ValueError("Captured result suite does not match its corpus.")
    suite_cases = tuple(item for item in validated_corpus.cases if item.suite_id == suite.id)
    if len(suite_cases) != 1 or suite_cases[0].id != validated.score.case_id:
        raise ValueError("Captured result suite must contain exactly its scored case.")
    case = suite_cases[0]
    if case.revision != validated.score.case_revision:
        raise ValueError("Captured result case does not match its corpus.")
    expected_assertions = tuple(
        (assertion.id, assertion_spec_revision(assertion)) for assertion in case.assertions
    )
    actual_assertions = tuple(
        (assertion.assertion_id, assertion.assertion_revision)
        for assertion in validated.score.assertions
    )
    if actual_assertions != expected_assertions:
        raise ValueError("Captured result assertions do not match its immutable corpus case.")
    if validated.score.evidence_policy_revision != validated_corpus.evidence_policy.revision:
        raise ValueError("Captured result evidence policy does not match its corpus.")
    expected_pricing = (
        None
        if validated_corpus.pricing_profile is None
        else validated_corpus.pricing_profile.fingerprint
    )
    if validated.score.pricing_profile_fingerprint not in {None, expected_pricing}:
        raise ValueError("Captured result pricing identity does not match its corpus.")
    if case.source is None:
        raise ValueError("Captured result corpus case requires captured source identity.")
    if (
        validated.target.application_release_id != case.source.application_release_id
        or validated.target.app_manifest_schema_version != case.source.app_manifest_schema_version
        or validated.target.app_manifest_fingerprint != case.source.app_manifest_fingerprint
        or validated.score.evidence_revision != case.source.evidence_revision
    ):
        raise ValueError("Captured result source identity does not match its corpus case.")
    return validated


def eval_result_projection(
    result: CorpusExecutionResult | CapturedEvaluationResultV1,
) -> EvalResultProjectionV1:
    """Project either immutable result origin into one comparison contract."""

    if type(result) is CorpusExecutionResult:
        validated_fresh = CorpusExecutionResult.model_validate(_model_python_input(result))
        run = validated_fresh.run
        cases = tuple(
            EvalResultCaseProjectionV1(
                case_id=case.case_id,
                case_revision=case.case_revision,
                status=case.status,
                score=case.score,
                assertions=_assertion_identities(case.trials[0].assertions),
            )
            for case in run.cases
        )
        return EvalResultProjectionV1(
            result_revision=validated_fresh.revision,
            origin=EvalResultOrigin.FRESH_EXECUTION,
            target=EvalResultTargetIdentityV1.from_fresh_target(validated_fresh.target),
            corpus_revision=run.corpus_revision,
            suite_id=run.suite_id,
            suite_revision=run.suite_revision,
            evidence_policy_revision=run.evidence_policy_revision,
            pricing_profile_fingerprint=run.pricing_profile_fingerprint,
            uses_pricing=any(
                assertion.detail.kind == "max_estimated_cost"
                for case in run.cases
                for trial in case.trials
                for assertion in trial.assertions
            ),
            status=run.status,
            score=run.score,
            cases=cases,
        )
    if type(result) is CapturedEvaluationResultV1:
        validated_captured = CapturedEvaluationResultV1.model_validate(_model_python_input(result))
        score = validated_captured.score
        case = EvalResultCaseProjectionV1(
            case_id=score.case_id,
            case_revision=score.case_revision,
            status=score.status,
            score=score.score,
            assertions=_assertion_identities(score.assertions),
        )
        return EvalResultProjectionV1(
            result_revision=validated_captured.revision,
            origin=EvalResultOrigin.CAPTURED_SESSION,
            target=validated_captured.target,
            corpus_revision=validated_captured.corpus_revision,
            suite_id=validated_captured.suite_id,
            suite_revision=validated_captured.suite_revision,
            evidence_policy_revision=score.evidence_policy_revision,
            pricing_profile_fingerprint=score.pricing_profile_fingerprint,
            uses_pricing=any(
                assertion.detail.kind == "max_estimated_cost" for assertion in score.assertions
            ),
            status=score.status,
            score=score.score,
            cases=(case,),
        )
    raise TypeError("result must be an exact CorpusExecutionResult or CapturedEvaluationResultV1.")


def captured_evaluation_result_from_json(source: str) -> CapturedEvaluationResultV1:
    """Parse a bounded captured-result document without format guessing."""

    if type(source) is not str:
        raise TypeError("captured_evaluation_result_from_json requires text.")
    if len(source) > CAPTURED_EVALUATION_RESULT_MAX_BYTES:
        raise ValueError(
            f"Captured evaluation result JSON exceeds {CAPTURED_EVALUATION_RESULT_MAX_BYTES} bytes."
        )
    try:
        raw = source.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("Captured evaluation result JSON must contain valid Unicode.") from exc
    if len(raw) > CAPTURED_EVALUATION_RESULT_MAX_BYTES:
        raise ValueError(
            f"Captured evaluation result JSON exceeds {CAPTURED_EVALUATION_RESULT_MAX_BYTES} bytes."
        )
    try:
        decoded = json.loads(
            source,
            parse_int=partial(
                parse_durable_json_integer_literal,
                field_name="captured evaluation result JSON",
            ),
            parse_constant=partial(
                reject_nonportable_json_constant,
                field_name="captured evaluation result JSON",
            ),
            object_pairs_hook=partial(
                durable_json_object_from_pairs,
                field_name="captured evaluation result JSON",
            ),
        )
    except RecursionError as exc:
        raise ValueError(
            "Captured evaluation result JSON nesting exceeds the supported depth."
        ) from exc
    document = copy_durable_json_object(decoded, "captured evaluation result JSON")
    raw_version = document.get("schema_version")
    if type(raw_version) is not int or raw_version != CAPTURED_EVALUATION_RESULT_SCHEMA_VERSION:
        raise ValueError(
            "Captured evaluation result has unsupported schema_version "
            f"{raw_version!r}; this Cayu version supports only "
            f"{CAPTURED_EVALUATION_RESULT_SCHEMA_VERSION}."
        )
    return CapturedEvaluationResultV1.model_validate(document)
