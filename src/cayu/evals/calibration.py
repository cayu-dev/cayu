"""Trusted structured-judge calibration over immutable retained evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from pydantic import Field, StrictBool, StrictInt, StrictStr, field_validator, model_validator

from cayu.evals.corpus import (
    EVAL_CORPUS_MAX_JUDGE_CRITERIA,
    EVAL_CORPUS_MAX_TOTAL_MESSAGE_CHARS,
    JudgeProfileIdentityV1,
    StructuredModelJudgeAssertionSpec,
    _bounded_durable_text,
    _content_revision,
    _exact_weighted_decimal,
    _model_content_revision,
    _model_python_input,
    _ordered_sequence_input,
    _portable_id,
    _PortableModel,
    _sha256_revision,
    _unit_interval_decimal_text,
    assertion_spec_revision,
)
from cayu.evals.execution import (
    CorpusTarget,
    _candidate_judge_route_relation,
    model_judge_profile,
)
from cayu.evals.judges import _canonical_unit_decimal
from cayu.evals.portable_assertions import (
    _CompiledStructuredModelJudgeAssertion,
    _trusted_model_judge_binding,
    _TrustedPrivateJudgeReferenceBinding,
)
from cayu.evals.published import (
    PublishedAssertionResult,
    PublishedStructuredModelJudgeDetail,
    _published_assertion,
    _published_reference_contract,
    _spec_reference_contract,
)

EVAL_JUDGE_CALIBRATION_SCHEMA_VERSION = 1
EVAL_JUDGE_CALIBRATION_MAX_TRIALS = 10
EVAL_JUDGE_CALIBRATION_MAX_BYTES = 2 << 20


class EvalJudgeCalibrationCriterionLabelV1(_PortableModel):
    """One human score for an exact rubric criterion."""

    criterion_id: StrictStr
    score: StrictStr

    @field_validator("criterion_id")
    @classmethod
    def validate_criterion_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: str, info) -> str:
        return _unit_interval_decimal_text(value, info.field_name)


class EvalJudgeCalibrationEvidenceProvenanceV1(_PortableModel):
    """Operator-declared origin for fixed evidence, without implied verification."""

    schema_version: Literal[1] = EVAL_JUDGE_CALIBRATION_SCHEMA_VERSION
    kind: Literal["operator_supplied"] = "operator_supplied"
    source_id: StrictStr

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)


class EvalJudgeCalibrationEvidenceV1(_PortableModel):
    """Content-addressed, public-safe candidate evidence that is never executed."""

    schema_version: Literal[1] = EVAL_JUDGE_CALIBRATION_SCHEMA_VERSION
    revision: StrictStr
    provenance: EvalJudgeCalibrationEvidenceProvenanceV1
    task: StrictStr
    final_output: StrictStr
    transcript: StrictStr | None = None

    @field_validator("revision")
    @classmethod
    def validate_revision_shape(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("task")
    @classmethod
    def validate_task(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=EVAL_CORPUS_MAX_TOTAL_MESSAGE_CHARS,
            nonblank=True,
            clean=False,
        )

    @field_validator("final_output")
    @classmethod
    def validate_final_output(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=EVAL_CORPUS_MAX_TOTAL_MESSAGE_CHARS,
            nonblank=False,
            clean=False,
        )

    @field_validator("transcript")
    @classmethod
    def validate_transcript(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=EVAL_CORPUS_MAX_TOTAL_MESSAGE_CHARS,
            nonblank=True,
            clean=False,
        )

    @model_validator(mode="after")
    def validate_revision(self) -> EvalJudgeCalibrationEvidenceV1:
        if self.revision != _model_content_revision(self, "judge calibration evidence"):
            raise ValueError("Judge calibration evidence revision does not match its content.")
        return self

    @classmethod
    def create(
        cls,
        *,
        source_id: str,
        task: str,
        final_output: str,
        transcript: str | None,
    ) -> EvalJudgeCalibrationEvidenceV1:
        material = {
            "schema_version": EVAL_JUDGE_CALIBRATION_SCHEMA_VERSION,
            "provenance": EvalJudgeCalibrationEvidenceProvenanceV1(source_id=source_id).model_dump(
                mode="json"
            ),
            "task": task,
            "final_output": final_output,
            "transcript": transcript,
        }
        return cls(
            revision=_content_revision(material, "judge calibration evidence"),
            **material,
        )


class EvalJudgeCalibrationHumanLabelV1(_PortableModel):
    """Human ground truth bound to one exact structured rubric."""

    schema_version: Literal[1] = EVAL_JUDGE_CALIBRATION_SCHEMA_VERSION
    rubric_revision: StrictStr
    criteria: tuple[EvalJudgeCalibrationCriterionLabelV1, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_JUDGE_CRITERIA,
    )
    aggregate_score: StrictStr

    @field_validator("rubric_revision")
    @classmethod
    def validate_rubric_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("criteria", mode="before")
    @classmethod
    def validate_criteria_order(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("aggregate_score")
    @classmethod
    def validate_aggregate_score(cls, value: str, info) -> str:
        return _unit_interval_decimal_text(value, info.field_name)


class EvalJudgeCalibrationDraftV1(_PortableModel):
    """Revision-free browser material for one calibration run."""

    schema_version: Literal[1] = EVAL_JUDGE_CALIBRATION_SCHEMA_VERSION
    id: StrictStr
    target_key: StrictStr
    assertion: StructuredModelJudgeAssertionSpec
    evidence_source_id: StrictStr
    task: StrictStr
    final_output: StrictStr
    transcript: StrictStr | None = None
    human_criteria: tuple[EvalJudgeCalibrationCriterionLabelV1, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_JUDGE_CRITERIA,
    )
    trials: StrictInt = Field(default=1, ge=1, le=EVAL_JUDGE_CALIBRATION_MAX_TRIALS)

    @field_validator("id", "target_key")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("evidence_source_id")
    @classmethod
    def validate_evidence_source_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("task")
    @classmethod
    def validate_task(cls, value: str, info) -> str:
        return EvalJudgeCalibrationEvidenceV1.validate_task(value, info)

    @field_validator("final_output")
    @classmethod
    def validate_final_output(cls, value: str, info) -> str:
        return EvalJudgeCalibrationEvidenceV1.validate_final_output(value, info)

    @field_validator("transcript")
    @classmethod
    def validate_transcript(cls, value: str | None, info) -> str | None:
        return EvalJudgeCalibrationEvidenceV1.validate_transcript(value, info)

    @field_validator("human_criteria", mode="before")
    @classmethod
    def validate_human_criteria_order(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)


class EvalJudgeCalibrationDefinitionV1(_PortableModel):
    """Immutable calibration inputs; invoking it can only execute the judge."""

    schema_version: Literal[1] = EVAL_JUDGE_CALIBRATION_SCHEMA_VERSION
    revision: StrictStr
    id: StrictStr
    target_key: StrictStr
    assertion: StructuredModelJudgeAssertionSpec
    evidence: EvalJudgeCalibrationEvidenceV1
    human_label: EvalJudgeCalibrationHumanLabelV1
    trials: StrictInt = Field(ge=1, le=EVAL_JUDGE_CALIBRATION_MAX_TRIALS)

    @field_validator("revision")
    @classmethod
    def validate_revision_shape(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("id", "target_key")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @model_validator(mode="after")
    def validate_contract(self) -> EvalJudgeCalibrationDefinitionV1:
        rubric = self.assertion.rubric
        if self.human_label.rubric_revision != rubric.revision:
            raise ValueError("Calibration human label does not match the rubric revision.")
        human_ids = tuple(item.criterion_id for item in self.human_label.criteria)
        rubric_ids = tuple(item.id for item in rubric.criteria)
        if human_ids != rubric_ids:
            raise ValueError("Calibration human labels must match rubric criterion order.")
        expected_human_aggregate = _exact_weighted_decimal(
            (criterion.weight, Decimal(label.score))
            for criterion, label in zip(rubric.criteria, self.human_label.criteria, strict=True)
        )
        if Decimal(self.human_label.aggregate_score) != expected_human_aggregate:
            raise ValueError("Calibration human aggregate does not match criterion labels.")
        expects_transcript = self.assertion.evidence.include_transcript
        if expects_transcript != (self.evidence.transcript is not None):
            raise ValueError("Calibration transcript does not match the judge evidence selection.")
        if self.revision != _model_content_revision(self, "judge calibration definition"):
            raise ValueError("Judge calibration definition revision does not match its content.")
        return self


def compile_eval_judge_calibration_draft(
    draft: EvalJudgeCalibrationDraftV1,
) -> EvalJudgeCalibrationDefinitionV1:
    """Canonicalize editor material into one immutable fixed-evidence contract."""

    if type(draft) is not EvalJudgeCalibrationDraftV1:
        raise TypeError("draft must be an exact EvalJudgeCalibrationDraftV1.")
    validated = EvalJudgeCalibrationDraftV1.model_validate(_model_python_input(draft))
    rubric = validated.assertion.rubric
    human_ids = tuple(item.criterion_id for item in validated.human_criteria)
    rubric_ids = tuple(item.id for item in rubric.criteria)
    if human_ids != rubric_ids:
        raise ValueError("Calibration human labels must match rubric criterion order.")
    human_aggregate = _exact_weighted_decimal(
        (criterion.weight, Decimal(label.score))
        for criterion, label in zip(rubric.criteria, validated.human_criteria, strict=True)
    )
    evidence = EvalJudgeCalibrationEvidenceV1.create(
        source_id=validated.evidence_source_id,
        task=validated.task,
        final_output=validated.final_output,
        transcript=validated.transcript,
    )
    human_label = EvalJudgeCalibrationHumanLabelV1(
        rubric_revision=rubric.revision,
        criteria=validated.human_criteria,
        aggregate_score=_canonical_unit_decimal(human_aggregate),
    )
    material = {
        "schema_version": EVAL_JUDGE_CALIBRATION_SCHEMA_VERSION,
        "id": validated.id,
        "target_key": validated.target_key,
        "assertion": validated.assertion.model_dump(mode="json"),
        "evidence": evidence.model_dump(mode="json"),
        "human_label": human_label.model_dump(mode="json"),
        "trials": validated.trials,
    }
    return EvalJudgeCalibrationDefinitionV1(
        revision=_content_revision(material, "judge calibration definition"),
        **material,
    )


@dataclass(frozen=True, slots=True)
class PreparedEvalJudgeCalibration:
    """Process-local judge authority paired with immutable browser-reviewed input."""

    definition: EvalJudgeCalibrationDefinitionV1
    judge_profile: JudgeProfileIdentityV1
    candidate_route_relation: Literal["independent_model", "same_model"]
    evaluator: _CompiledStructuredModelJudgeAssertion


def prepare_eval_judge_calibration(
    definition: EvalJudgeCalibrationDefinitionV1,
    target: CorpusTarget,
) -> PreparedEvalJudgeCalibration:
    """Resolve exact current authority without compiling any candidate request."""

    if type(definition) is not EvalJudgeCalibrationDefinitionV1:
        raise TypeError("definition must be an exact EvalJudgeCalibrationDefinitionV1.")
    if type(target) is not CorpusTarget:
        raise TypeError("target must be an exact CorpusTarget.")
    validated = EvalJudgeCalibrationDefinitionV1.model_validate(_model_python_input(definition))
    if validated.target_key != target.key:
        raise ValueError("Calibration target does not match current server authority.")
    judge = next(
        (item for item in target.model_judges if item.key == validated.assertion.judge_profile_key),
        None,
    )
    if judge is None:
        raise ValueError("The selected judge profile is not currently published.")
    profile = model_judge_profile(judge)
    if profile.revision != validated.assertion.judge_profile_revision:
        raise ValueError("The selected judge profile changed after calibration preview.")
    relation = _candidate_judge_route_relation(target, profile)
    if relation == "same_model" and profile.same_model_use != "allowed_and_labeled":
        raise ValueError("The selected judge profile forbids same-model judging.")
    binding = _trusted_model_judge_binding(
        key=judge.key,
        app=judge.app,
        agent_name=judge.agent_name,
        profile=profile,
        privacy_policy=judge.privacy_policy,
        private_references=tuple(
            _TrustedPrivateJudgeReferenceBinding(
                key=reference.key,
                revision=reference.revision,
                content=reference.content,
                privacy_policy_key=reference.privacy_policy_key,
                privacy_policy_revision=reference.privacy_policy_revision,
            )
            for reference in judge.private_references
        ),
        price_book=judge.price_book,
        candidate_route_relation=relation,
    )
    evaluator = _CompiledStructuredModelJudgeAssertion(
        validated.assertion,
        binding=binding,
        app=target.app,
        evidence_policy=target.evidence_policy,
    )
    return PreparedEvalJudgeCalibration(
        definition=validated,
        judge_profile=profile,
        candidate_route_relation=relation,
        evaluator=evaluator,
    )


class EvalJudgeCalibrationTrialV1(_PortableModel):
    """One lossless judge observation compared with the fixed human label."""

    schema_version: Literal[1] = EVAL_JUDGE_CALIBRATION_SCHEMA_VERSION
    revision: StrictStr
    sequence: StrictInt = Field(ge=1, le=EVAL_JUDGE_CALIBRATION_MAX_TRIALS)
    judgment: PublishedAssertionResult
    aggregate_absolute_error: StrictStr | None = None
    pass_agreement: StrictBool | None = None

    @field_validator("revision")
    @classmethod
    def validate_revision_shape(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("aggregate_absolute_error")
    @classmethod
    def validate_error(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _unit_interval_decimal_text(value, info.field_name)

    @model_validator(mode="after")
    def validate_contract(self) -> EvalJudgeCalibrationTrialV1:
        if type(self.judgment.detail) is not PublishedStructuredModelJudgeDetail:
            raise ValueError("Calibration trials must retain a structured judge result.")
        scored = self.judgment.outcome in {"passed", "failed"}
        if scored != (self.aggregate_absolute_error is not None):
            raise ValueError("Only scored calibration trials carry aggregate error.")
        if scored != (self.pass_agreement is not None):
            raise ValueError("Only scored calibration trials carry pass agreement.")
        if self.revision != _model_content_revision(self, "judge calibration trial"):
            raise ValueError("Judge calibration trial revision does not match its content.")
        return self

    @classmethod
    def create(
        cls,
        *,
        sequence: int,
        judgment: PublishedAssertionResult,
        human_aggregate_score: str,
        threshold: str,
    ) -> EvalJudgeCalibrationTrialV1:
        detail = judgment.detail
        if type(detail) is not PublishedStructuredModelJudgeDetail:
            raise ValueError("Calibration trials require a structured judge result.")
        scored = detail.aggregate_score is not None
        error = None
        agreement = None
        if scored:
            judge_score = Decimal(detail.aggregate_score)
            human_score = Decimal(human_aggregate_score)
            error = _canonical_unit_decimal(abs(judge_score - human_score))
            agreement = (judge_score >= Decimal(threshold)) == (human_score >= Decimal(threshold))
        material = {
            "schema_version": EVAL_JUDGE_CALIBRATION_SCHEMA_VERSION,
            "sequence": sequence,
            "judgment": judgment.model_dump(mode="json"),
            "aggregate_absolute_error": error,
            "pass_agreement": agreement,
        }
        return cls(
            revision=_content_revision(material, "judge calibration trial"),
            sequence=sequence,
            judgment=judgment,
            aggregate_absolute_error=error,
            pass_agreement=agreement,
        )


async def run_eval_judge_calibration_trial(
    prepared: PreparedEvalJudgeCalibration,
    *,
    sequence: int,
) -> EvalJudgeCalibrationTrialV1:
    """Execute exactly one judge call over retained text; no candidate path exists."""

    if type(prepared) is not PreparedEvalJudgeCalibration:
        raise TypeError("prepared must be an exact PreparedEvalJudgeCalibration.")
    evidence = prepared.definition.evidence
    internal = await prepared.evaluator.evaluate_retained_material(
        task=evidence.task,
        final_output=evidence.final_output,
        transcript_text=evidence.transcript,
    )
    published = _published_assertion(prepared.definition.assertion, internal)
    return EvalJudgeCalibrationTrialV1.create(
        sequence=sequence,
        judgment=published,
        human_aggregate_score=prepared.definition.human_label.aggregate_score,
        threshold=prepared.definition.assertion.threshold,
    )


class EvalJudgeCalibrationReportV1(_PortableModel):
    """Immutable completed calibration with every fixed-evidence judge trial."""

    schema_version: Literal[1] = EVAL_JUDGE_CALIBRATION_SCHEMA_VERSION
    revision: StrictStr
    run_id: StrictStr
    definition: EvalJudgeCalibrationDefinitionV1
    judge_profile: JudgeProfileIdentityV1
    candidate_route_relation: Literal["independent_model", "same_model"]
    trials: tuple[EvalJudgeCalibrationTrialV1, ...] = Field(
        min_length=1,
        max_length=EVAL_JUDGE_CALIBRATION_MAX_TRIALS,
    )

    @field_validator("revision")
    @classmethod
    def validate_revision_shape(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("trials", mode="before")
    @classmethod
    def validate_trial_order(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @model_validator(mode="after")
    def validate_contract(self) -> EvalJudgeCalibrationReportV1:
        if (self.judge_profile.key, self.judge_profile.revision) != (
            self.definition.assertion.judge_profile_key,
            self.definition.assertion.judge_profile_revision,
        ):
            raise ValueError("Calibration report judge profile does not match its definition.")
        if len(self.trials) != self.definition.trials:
            raise ValueError("Calibration report must retain every requested trial.")
        if tuple(item.sequence for item in self.trials) != tuple(range(1, len(self.trials) + 1)):
            raise ValueError("Calibration trial sequence must be contiguous and ordered.")
        if (
            self.candidate_route_relation == "same_model"
            and self.judge_profile.same_model_use != "allowed_and_labeled"
        ):
            raise ValueError("Calibration report used a forbidden same-model judge route.")
        assertion = self.definition.assertion
        for trial in self.trials:
            if (trial.judgment.assertion_id, trial.judgment.assertion_revision) != (
                assertion.id,
                assertion_spec_revision(assertion),
            ):
                raise ValueError("Calibration trial assertion does not match the definition.")
            detail = trial.judgment.detail
            if type(detail) is not PublishedStructuredModelJudgeDetail:
                raise ValueError("Calibration reports require structured judge trials.")
            if detail.judge_profile != self.judge_profile:
                raise ValueError("Calibration trial judge profile does not match the report.")
            if detail.candidate_route_relation != self.candidate_route_relation:
                raise ValueError("Calibration trial route relation does not match the report.")
            if (detail.rubric_id, detail.rubric_revision) != (
                assertion.rubric.id,
                assertion.rubric.revision,
            ):
                raise ValueError("Calibration trial rubric does not match the definition.")
            if (detail.threshold, detail.evidence) != (
                assertion.threshold,
                assertion.evidence,
            ):
                raise ValueError("Calibration trial policy does not match the definition.")
            if _published_reference_contract(detail.reference) != _spec_reference_contract(
                assertion.reference
            ):
                raise ValueError("Calibration trial reference does not match the definition.")
            if detail.aggregate_score is None:
                continue
            if tuple((item.criterion_id, item.weight) for item in detail.criteria) != tuple(
                (item.id, item.weight) for item in assertion.rubric.criteria
            ):
                raise ValueError("Calibration trial criteria do not match the definition.")
            judge_score = Decimal(detail.aggregate_score)
            human_score = Decimal(self.definition.human_label.aggregate_score)
            expected_error = _canonical_unit_decimal(abs(judge_score - human_score))
            expected_agreement = (judge_score >= Decimal(assertion.threshold)) == (
                human_score >= Decimal(assertion.threshold)
            )
            if (
                trial.aggregate_absolute_error != expected_error
                or trial.pass_agreement is not expected_agreement
            ):
                raise ValueError("Calibration trial comparison does not match the human label.")
        if self.revision != _model_content_revision(self, "judge calibration report"):
            raise ValueError("Judge calibration report revision does not match its content.")
        return self

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        prepared: PreparedEvalJudgeCalibration,
        trials: tuple[EvalJudgeCalibrationTrialV1, ...],
    ) -> EvalJudgeCalibrationReportV1:
        material = {
            "schema_version": EVAL_JUDGE_CALIBRATION_SCHEMA_VERSION,
            "run_id": run_id,
            "definition": prepared.definition.model_dump(mode="json"),
            "judge_profile": prepared.judge_profile.model_dump(mode="json"),
            "candidate_route_relation": prepared.candidate_route_relation,
            "trials": [trial.model_dump(mode="json") for trial in trials],
        }
        return cls(
            revision=_content_revision(material, "judge calibration report"),
            run_id=run_id,
            definition=prepared.definition,
            judge_profile=prepared.judge_profile,
            candidate_route_relation=prepared.candidate_route_relation,
            trials=trials,
        )


def eval_judge_calibration_report_to_json(report: EvalJudgeCalibrationReportV1) -> str:
    """Serialize one validated report into deterministic bounded storage JSON."""

    if type(report) is not EvalJudgeCalibrationReportV1:
        raise TypeError("report must be an exact EvalJudgeCalibrationReportV1.")
    validated = EvalJudgeCalibrationReportV1.model_validate(_model_python_input(report))
    rendered = json.dumps(
        validated.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(rendered.encode("utf-8")) > EVAL_JUDGE_CALIBRATION_MAX_BYTES:
        raise ValueError(
            f"Judge calibration report exceeds {EVAL_JUDGE_CALIBRATION_MAX_BYTES} bytes."
        )
    return rendered


def eval_judge_calibration_report_from_json(source: str) -> EvalJudgeCalibrationReportV1:
    """Load one bounded report without accepting an unknown wire shape."""

    if type(source) is not str:
        raise TypeError("eval_judge_calibration_report_from_json requires text.")
    try:
        encoded = source.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "Judge calibration report must contain valid Unicode scalar text."
        ) from exc
    if len(encoded) > EVAL_JUDGE_CALIBRATION_MAX_BYTES:
        raise ValueError(
            f"Judge calibration report exceeds {EVAL_JUDGE_CALIBRATION_MAX_BYTES} bytes."
        )
    return EvalJudgeCalibrationReportV1.model_validate_json(encoded)


__all__ = [
    "EVAL_JUDGE_CALIBRATION_MAX_BYTES",
    "EVAL_JUDGE_CALIBRATION_MAX_TRIALS",
    "EVAL_JUDGE_CALIBRATION_SCHEMA_VERSION",
    "EvalJudgeCalibrationCriterionLabelV1",
    "EvalJudgeCalibrationDefinitionV1",
    "EvalJudgeCalibrationDraftV1",
    "EvalJudgeCalibrationEvidenceProvenanceV1",
    "EvalJudgeCalibrationEvidenceV1",
    "EvalJudgeCalibrationHumanLabelV1",
    "EvalJudgeCalibrationReportV1",
    "EvalJudgeCalibrationTrialV1",
    "PreparedEvalJudgeCalibration",
    "compile_eval_judge_calibration_draft",
    "eval_judge_calibration_report_from_json",
    "eval_judge_calibration_report_to_json",
    "prepare_eval_judge_calibration",
    "run_eval_judge_calibration_trial",
]
