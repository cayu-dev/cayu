"""External CLI evidence: publication identity never implies native execution authority."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._validation import (
    DurableJsonLimits,
    copy_durable_record,
    require_durable_clean_nonblank,
    require_durable_text,
)
from cayu.evals.execution import CORPUS_EXECUTION_RESULT_MAX_BYTES, CorpusExecutionResult
from cayu.evals.models import (
    EvalAssertionResult,
    EvalOutcome,
    EvalStatus,
    aggregate_eval_score,
    aggregate_eval_status,
)
from cayu.evals.trial_policy import EvalSuiteTrialPolicyV1

EVAL_REPORT_MAX_BYTES = CORPUS_EXECUTION_RESULT_MAX_BYTES + (64 << 10)
# Every JSON node consumes at least one encoded byte; a byte-sized node ceiling
# preserves the full admitted report domain while bounding validation work.
EVAL_REPORT_JSON_LIMITS = DurableJsonLimits(EVAL_REPORT_MAX_BYTES, EVAL_REPORT_MAX_BYTES)
EVAL_REPORT_MAX_CASES = 1_000
EVAL_REPORT_MAX_TRIALS = 100


def _identifier(value: str) -> str:
    return require_durable_clean_nonblank(value, "report identity")


def _bounded_text(value: str) -> str:
    if len(value.encode("utf-8")) > 64 << 10:
        raise ValueError("Report text exceeds its encoded byte limit.")
    return value


_Identifier = Annotated[StrictStr, Field(min_length=1, max_length=256), AfterValidator(_identifier)]
_Revision = Annotated[StrictStr, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
_Text = Annotated[StrictStr, Field(max_length=64 << 10), AfterValidator(_bounded_text)]
_Score = Annotated[StrictFloat, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class _ReportModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, revalidate_instances="always", hide_input_in_errors=True
    )

    @field_validator("*", mode="after")
    @classmethod
    def validate_text(cls, value):
        if type(value) is str:
            require_durable_text(value, "report field")
        return value


class EvalReportConflict(ValueError):
    """An immutable report operation or requested placement disagrees with stored evidence."""


class EvalReportOrigin(_ReportModel):
    """Publisher-supplied attribution, separate from authenticated publisher identity."""

    kind: Literal["cli", "ci"] = "cli"
    provider: _Identifier | None = None
    repository: _Identifier | None = None
    branch: _Identifier | None = None
    commit: Annotated[StrictStr, Field(pattern=r"^[0-9a-fA-F]{40,64}$")] | None = None
    pr_head_commit: Annotated[StrictStr, Field(pattern=r"^[0-9a-fA-F]{40,64}$")] | None = None
    pull_request: StrictInt | None = Field(default=None, ge=1, le=2**53 - 1)
    run_url: StrictStr | None = Field(default=None, max_length=2048)
    run_id: _Identifier | None = None
    run_attempt: StrictInt | None = Field(default=None, ge=1, le=2**31 - 1)
    job: _Identifier | None = None
    sources: dict[StrictStr, Literal["flag", "environment"]] = Field(default_factory=dict)

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, value):
        if len(value) > 12 or not set(value) <= cls.model_fields.keys() - {"sources"}:
            raise ValueError("Invalid report attribution sources.")
        return value

    @field_validator("run_url")
    @classmethod
    def validate_url(cls, value):
        if value is not None:
            parsed = urlsplit(value)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "CI run URLs require HTTPS without credentials, query or fragment."
                )
        return value


class EvalReportedCheck(_ReportModel):
    name: _Identifier
    outcome: EvalOutcome
    score: _Score | None = None
    threshold: _Score | None = None
    message: _Text = ""

    @model_validator(mode="after")
    def validate_outcome(self):
        # Preserve the existing assertion owner, including unavailable/error score rules.
        EvalAssertionResult.model_validate(self.model_dump(mode="python"))
        return self


class EvalReportedTrial(_ReportModel):
    trial_number: StrictInt = Field(ge=1, le=EVAL_REPORT_MAX_TRIALS)
    status: EvalStatus
    score: _Score | None = None
    final_output: _Text = ""
    structured_output: JsonValue = None
    error: _Text | None = None
    unavailable_reason: _Text | None = None
    assertions: tuple[EvalReportedCheck, ...] = Field(max_length=256)

    @field_validator("structured_output", mode="before")
    @classmethod
    def validate_structured_output(cls, value):
        return copy_durable_record(
            value, "reported structured output", limits=EVAL_REPORT_JSON_LIMITS
        )

    @model_validator(mode="after")
    def validate_assertion_summary(self):
        # Match the retained portion of EvalTrialResult's result contract. The
        # private workflow failure-capture marker is intentionally not exported;
        # its conservative ERROR + all-UNAVAILABLE projection remains valid.
        expected = aggregate_eval_status(
            EvalStatus(check.outcome.value) for check in self.assertions
        )
        if (
            self.status == EvalStatus.ERROR
            and self.assertions
            and all(check.outcome == EvalOutcome.UNAVAILABLE for check in self.assertions)
        ):
            expected = EvalStatus.ERROR
        if (self.assertions and self.status != expected) or (
            not self.assertions
            and self.status not in {EvalStatus.ERROR, EvalStatus.UNAVAILABLE, EvalStatus.SKIPPED}
        ):
            raise ValueError("Reported trial status disagrees with its retained assertions.")
        if self.status in {EvalStatus.ERROR, EvalStatus.UNAVAILABLE}:
            if self.score is not None:
                raise ValueError("Unscored reported trial states cannot carry a score.")
        else:
            expected_score = (
                aggregate_eval_score(check.score for check in self.assertions)
                if self.assertions
                else 0.0
            )
            if self.score != expected_score:
                raise ValueError("Reported trial score disagrees with its retained assertions.")
        if self.status == EvalStatus.ERROR:
            if not self.error or not self.error.strip():
                raise ValueError("Reported error trials require an error diagnostic.")
        elif self.error is not None:
            raise ValueError("Only reported error trials can carry an error diagnostic.")
        if self.status == EvalStatus.UNAVAILABLE:
            if not self.unavailable_reason or not self.unavailable_reason.strip():
                raise ValueError("Reported unavailable trials require an unavailable reason.")
        elif self.unavailable_reason is not None:
            raise ValueError("Only reported unavailable trials can carry an unavailable reason.")
        return self


def _reported_assertion_contract_conflict(trials):
    # The native case owner also fails closed on incompatible retained assertion
    # layouts. Report projections retain names/thresholds, not private metadata.
    if len(trials) <= 1:
        return False
    if any(len(trial.assertions) != len(trials[0].assertions) for trial in trials[1:]):
        return True
    for group in zip(*(trial.assertions for trial in trials), strict=True):
        if (
            len({check.name for check in group}) != 1
            or len({check.threshold for check in group if check.score is not None}) > 1
        ):
            return True
    return False


class EvalReportedCase(_ReportModel):
    case_id: _Identifier
    error: _Text | None = None
    unavailable_reason: _Text | None = None
    status: EvalStatus
    score: _Score | None = None
    trial_policy: EvalSuiteTrialPolicyV1
    trials: tuple[EvalReportedTrial, ...] = Field(min_length=1, max_length=EVAL_REPORT_MAX_TRIALS)

    @model_validator(mode="after")
    def validate_trials(self):
        if tuple(trial.trial_number for trial in self.trials) != tuple(
            range(1, len(self.trials) + 1)
        ) or self.trial_policy.trial_count != len(self.trials):
            raise ValueError("Report trial identities disagree with the selected trial policy.")
        statuses = tuple(trial.status for trial in self.trials)
        if self.status == EvalStatus.ERROR:
            if self.score is not None:
                raise ValueError("Error cases cannot carry an aggregate score.")
            # Native aggregation may discover a contract error not represented by
            # trial statuses (including revisions omitted from the safe projection).
            if not self.error or not self.error.strip():
                raise ValueError("Reported error cases require an error diagnostic.")
        else:
            if _reported_assertion_contract_conflict(self.trials):
                raise ValueError("Reported case assertions have inconsistent retained contracts.")
            expected = next(
                (
                    status
                    for status in (EvalStatus.ERROR, EvalStatus.UNAVAILABLE, EvalStatus.SKIPPED)
                    if status in statuses
                ),
                None,
            )
            if expected is None:
                expected = (
                    EvalStatus.PASSED
                    if statuses.count(EvalStatus.PASSED) >= self.trial_policy.minimum_passed_trials
                    else EvalStatus.FAILED
                )
            if self.status != expected:
                raise ValueError("Report case status disagrees with its retained trial policy.")
        if self.status != EvalStatus.ERROR and self.score != aggregate_eval_score(
            trial.score for trial in self.trials
        ):
            raise ValueError("Report case score disagrees with its retained trials.")
        return self


class EvalReportedRun(_ReportModel):
    """Allowlisted direct evidence, without arbitrary metadata, trajectories or local handles."""

    suite_id: _Identifier
    status: EvalStatus
    score: _Score | None = None
    started_at: datetime
    completed_at: datetime
    duration_ms: StrictInt = Field(ge=0, le=2**63 - 1)
    cases: tuple[EvalReportedCase, ...] = Field(min_length=1, max_length=EVAL_REPORT_MAX_CASES)

    @model_validator(mode="after")
    def validate_run(self):
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Report case identities must be unique.")
        if self.started_at.utcoffset() is None or self.completed_at.utcoffset() is None:
            raise ValueError("Report execution timestamps require a timezone.")
        if self.completed_at < self.started_at:
            raise ValueError("Report completion cannot precede its start.")
        if self.status != aggregate_eval_status(case.status for case in self.cases):
            raise ValueError("Report status disagrees with its cases.")
        if self.score != aggregate_eval_score(case.score for case in self.cases):
            raise ValueError("Report score disagrees with its cases.")
        return self


class EvalDirectReportEvidence(_ReportModel):
    kind: Literal["direct"] = "direct"
    run: EvalReportedRun


class EvalCorpusReportEvidence(_ReportModel):
    kind: Literal["corpus"] = "corpus"
    result: CorpusExecutionResult


class EvalReportCorpusAssociation(_ReportModel):
    corpus_revision: _Revision
    suite_id: _Identifier


class EvalReport(_ReportModel):
    schema_version: Literal[1] = 1
    report_id: _Identifier
    target_key: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    program_key: _Identifier
    declared_suite_revision: _Revision | None = None
    origin: EvalReportOrigin = Field(default_factory=EvalReportOrigin)
    evidence: Annotated[
        EvalDirectReportEvidence | EvalCorpusReportEvidence, Field(discriminator="kind")
    ]
    association: EvalReportCorpusAssociation | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError("Report schema_version must be integer1.")
        return value

    @model_validator(mode="after")
    def validate_association(self):
        if self.association is not None:
            if type(self.evidence) is not EvalCorpusReportEvidence:
                raise ValueError("Only exact corpus evidence can request a native association.")
            result = self.evidence.result
            if (
                result.target.target_key != self.target_key
                or result.run.corpus_revision != self.association.corpus_revision
                or result.run.suite_id != self.association.suite_id
            ):
                raise ValueError("Report association disagrees with its execution contract.")
        return self

    @property
    def suite_id(self) -> str:
        return (
            self.evidence.run.suite_id
            if isinstance(self.evidence, EvalDirectReportEvidence)
            else self.evidence.result.run.suite_id
        )


class EvalReportedSeriesSelector(_ReportModel):
    source: Literal["reported"] = "reported"
    series_id: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def id(self) -> str:
        from cayu.evals.read_views import encode_view_token

        return encode_view_token(["reported-suite", self.model_dump(mode="json")])


class EvalReportedSeriesIdentity(_ReportModel):
    """An external history grouping; deliberately has no invented immutable revision."""

    source: Literal["reported"] = "reported"
    target_key: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    publisher_id: _Identifier
    program_key: _Identifier
    suite_id: _Identifier

    @property
    def id(self) -> str:
        return self.selector.id

    @property
    def selector(self) -> EvalReportedSeriesSelector:
        digest = hashlib.sha256(
            json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return EvalReportedSeriesSelector(series_id=digest)


def report_operation_id(publisher_id: str, report_id: str) -> str:
    return hashlib.sha256(
        json.dumps(["cayu.eval-report.v1", publisher_id, report_id], separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


class EvalReportReceipt(_ReportModel):
    id: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    report_id: _Identifier
    publisher_id: _Identifier
    target_key: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    accepted_at: datetime
    run_key: StrictStr

    @model_validator(mode="after")
    def validate_identity(self):
        from cayu.evals.read_views import run_view_id

        if (
            self.id != report_operation_id(self.publisher_id, self.report_id)
            or self.run_key != run_view_id("report", self.id)
            or self.accepted_at.utcoffset() is None
        ):
            raise ValueError("Invalid report receipt identity.")
        return self


class EvalReportRecord(_ReportModel):
    receipt: EvalReportReceipt
    report: EvalReport

    @property
    def series(self) -> EvalReportedSeriesIdentity:
        return EvalReportedSeriesIdentity(
            target_key=self.report.target_key,
            publisher_id=self.receipt.publisher_id,
            program_key=self.report.program_key,
            suite_id=self.report.suite_id,
        )


def report_canonical_bytes(report: EvalReport) -> bytes:
    return json.dumps(
        report.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def report_fingerprint(report: EvalReport) -> str:
    return hashlib.sha256(report_canonical_bytes(report)).hexdigest()
