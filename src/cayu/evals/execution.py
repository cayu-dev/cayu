from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._validation import copy_durable_json_object, json_utf8_size_within_limit
from cayu.core.messages import Message, MessageRole, TextPart, detach_message
from cayu.evals.corpus import (
    EVAL_CORPUS_MAX_CASES,
    EVAL_CORPUS_MAX_MESSAGE_CHARS,
    EVAL_CORPUS_MAX_MESSAGES_PER_CASE,
    EVAL_CORPUS_MAX_TIMEOUT_SECONDS,
    EVAL_CORPUS_MAX_TOTAL_MESSAGE_CHARS,
    EVAL_CORPUS_MAX_TRIALS,
    EvalCorpusDocument,
    EvaluationEvidencePolicySpec,
    _bounded_durable_text,
    _content_revision,
    _model_content_revision,
    _model_python_input,
    _portable_id,
    _sha256_revision,
    eval_run_contract_for_corpus,
    pricing_profile_identity,
)
from cayu.evals.models import EvalRun, EvalRunContractV1, _model_instance_python_input
from cayu.evals.portable_assertions import _compile_corpus_assertion_specs
from cayu.evals.published import PublishedEvalRun, _publish_eval_run_with_trial_public_data
from cayu.evals.result_contract import (
    EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES,
    PUBLISHED_EVAL_OUTPUT_PREVIEW_BUDGET_BYTES,
)
from cayu.evals.runner import EvalCase, EvalSuite, _run_eval_suite_with_public_projection
from cayu.runtime.app import CayuApp
from cayu.runtime.costs import PriceBook, copy_price_book
from cayu.runtime.manifest import AppManifest
from cayu.runtime.sessions import RunRequest, copy_run_request

CORPUS_EXECUTION_MAX_BOOTSTRAP_MESSAGES = EVAL_CORPUS_MAX_MESSAGES_PER_CASE
CORPUS_EXECUTION_MAX_TOTAL_INPUT_CHARS = EVAL_CORPUS_MAX_TOTAL_MESSAGE_CHARS * 2
CORPUS_EXECUTION_MAX_COMPILED_INPUT_CHARS = 8 << 20
CORPUS_EXECUTION_MAX_CONCURRENCY = 32
CORPUS_EXECUTION_MAX_APP_MANIFEST_BYTES = 1 << 20
CORPUS_EXECUTION_MAX_REQUEST_BASE_BYTES = 64 << 10
CORPUS_EXECUTION_RESULT_MAX_BYTES = 40 << 20
CORPUS_EXECUTION_RESULT_SCHEMA_VERSION = 1


def _bootstrap_message_text(message: Message) -> str:
    """Return text from a bootstrap message after the target validator has run."""

    if len(message.content) != 1 or type(message.content[0]) is not TextPart:
        raise RuntimeError("Validated CorpusTarget bootstrap message lost its text contract.")
    return message.content[0].text


def _require_public_target_text(app: CayuApp, value: str, field_name: str) -> str:
    """Reject target identity text that the app's publication boundary would redact."""

    try:
        redacted = app.redact_json(value)
    except Exception as exc:
        raise ValueError(
            f"CorpusTarget {field_name} could not cross the application redaction boundary."
        ) from exc
    if type(redacted) is not str or redacted != value:
        raise ValueError(f"CorpusTarget {field_name} contains a workload secret.")
    return value


class CorpusExecutionLimits(BaseModel):
    """Trusted ceilings that may narrow, but never expand, portable corpus limits."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    max_cases: StrictInt = Field(default=EVAL_CORPUS_MAX_CASES, ge=1, le=EVAL_CORPUS_MAX_CASES)
    max_trials: StrictInt = Field(default=EVAL_CORPUS_MAX_TRIALS, ge=1, le=EVAL_CORPUS_MAX_TRIALS)
    max_timeout_seconds: StrictInt = Field(
        default=EVAL_CORPUS_MAX_TIMEOUT_SECONDS,
        ge=1,
        le=EVAL_CORPUS_MAX_TIMEOUT_SECONDS,
    )
    max_concurrency: StrictInt = Field(
        default=CORPUS_EXECUTION_MAX_CONCURRENCY,
        ge=1,
        le=CORPUS_EXECUTION_MAX_CONCURRENCY,
    )
    max_bootstrap_messages: StrictInt = Field(
        default=CORPUS_EXECUTION_MAX_BOOTSTRAP_MESSAGES,
        ge=0,
        le=CORPUS_EXECUTION_MAX_BOOTSTRAP_MESSAGES,
    )
    max_total_input_chars: StrictInt = Field(
        default=CORPUS_EXECUTION_MAX_TOTAL_INPUT_CHARS,
        ge=1,
        le=CORPUS_EXECUTION_MAX_TOTAL_INPUT_CHARS,
    )
    max_compiled_input_chars: StrictInt = Field(
        default=CORPUS_EXECUTION_MAX_COMPILED_INPUT_CHARS,
        ge=1,
        le=CORPUS_EXECUTION_MAX_COMPILED_INPUT_CHARS,
    )


class EvaluationTargetIdentity(BaseModel):
    """Public diagnostic identity of the fresh application used for execution."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    schema_version: Literal[1] = 1
    target_key: StrictStr
    application_release_id: StrictStr
    app_manifest: AppManifest

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

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

    @field_validator("app_manifest", mode="before")
    @classmethod
    def copy_app_manifest(cls, value: object) -> object:
        if type(value) is AppManifest:
            return AppManifest.model_validate(
                value.model_dump(mode="python", round_trip=True, warnings="none")
            )
        if isinstance(value, BaseModel):
            raise TypeError("app_manifest must be an exact AppManifest or JSON object.")
        return value

    @model_validator(mode="after")
    def validate_app_manifest(self) -> EvaluationTargetIdentity:
        fingerprint = self.app_manifest.fingerprint
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise ValueError("app_manifest_fingerprint must be a lowercase SHA-256 hex digest.")
        manifest_document = copy_durable_json_object(
            self.app_manifest.model_dump(mode="json"),
            "AppManifest",
        )
        manifest_document.pop("fingerprint")
        expected_fingerprint = hashlib.sha256(
            json.dumps(
                manifest_document,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if fingerprint != expected_fingerprint:
            raise ValueError("AppManifest fingerprint does not match its content.")
        if not json_utf8_size_within_limit(
            self.app_manifest,
            CORPUS_EXECUTION_MAX_APP_MANIFEST_BYTES,
        ):
            raise ValueError(
                "AppManifest exceeds the corpus execution limit of "
                f"{CORPUS_EXECUTION_MAX_APP_MANIFEST_BYTES} canonical JSON bytes."
            )
        return self

    @property
    def app_manifest_schema_version(self) -> str:
        return self.app_manifest.schema_version

    @property
    def app_manifest_fingerprint(self) -> str:
        return self.app_manifest.fingerprint


class CorpusTarget(BaseModel):
    """Trusted local execution authority selected by an eval project target."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    key: StrictStr
    app: CayuApp
    request_base: RunRequest
    bootstrap_messages: tuple[Message, ...] = ()
    application_release_id: StrictStr
    evidence_policy: EvaluationEvidencePolicySpec = Field(
        default_factory=EvaluationEvidencePolicySpec.standard
    )
    price_book: PriceBook | None = None
    limits: CorpusExecutionLimits = Field(default_factory=CorpusExecutionLimits)

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        return _portable_id(value, "key")

    @field_validator("app")
    @classmethod
    def validate_app(cls, value: CayuApp) -> CayuApp:
        if not isinstance(value, CayuApp):
            raise TypeError("CorpusTarget app must be a CayuApp.")
        return value

    @field_validator("request_base", mode="before")
    @classmethod
    def validate_request_base(cls, value: object) -> RunRequest:
        if type(value) is not RunRequest:
            raise TypeError("CorpusTarget request_base must be an exact RunRequest.")
        messages = getattr(value, "messages", None)
        if type(messages) is not list or messages:
            raise ValueError("CorpusTarget request_base must have no messages.")
        return copy_run_request(value)

    @field_validator("bootstrap_messages", mode="before")
    @classmethod
    def validate_bootstrap_messages(cls, value: object) -> tuple[Message, ...]:
        if not isinstance(value, list | tuple):
            raise TypeError("CorpusTarget bootstrap_messages must be an ordered list or tuple.")
        if len(value) > CORPUS_EXECUTION_MAX_BOOTSTRAP_MESSAGES:
            raise ValueError(
                "CorpusTarget bootstrap_messages cannot contain more than "
                f"{CORPUS_EXECUTION_MAX_BOOTSTRAP_MESSAGES} messages."
            )
        copied: list[Message] = []
        for index, message in enumerate(value):
            if type(message) is not Message:
                raise TypeError("CorpusTarget bootstrap_messages must contain exact Messages.")
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
            if role not in {MessageRole.SYSTEM, MessageRole.USER}:
                raise ValueError(
                    "CorpusTarget bootstrap_messages support only system and user roles."
                )
            if (
                not isinstance(content, tuple)
                or len(content) != 1
                or type(content[0]) is not TextPart
            ):
                raise ValueError(
                    "CorpusTarget bootstrap_messages must each contain exactly one TextPart."
                )
            text = getattr(content[0], "text", None)
            if type(text) is not str or len(text) > EVAL_CORPUS_MAX_MESSAGE_CHARS:
                raise ValueError(
                    f"bootstrap_messages[{index}] text must be at most "
                    f"{EVAL_CORPUS_MAX_MESSAGE_CHARS} characters."
                )
            copied.append(detach_message(message))
        return tuple(copied)

    @field_validator("application_release_id")
    @classmethod
    def validate_application_release_id(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @field_validator("evidence_policy", mode="before")
    @classmethod
    def copy_evidence_policy(cls, value: object) -> EvaluationEvidencePolicySpec:
        if type(value) is not EvaluationEvidencePolicySpec:
            raise TypeError("CorpusTarget evidence_policy must be an exact policy.")
        return EvaluationEvidencePolicySpec.model_validate(_model_python_input(value))

    @field_validator("price_book", mode="before")
    @classmethod
    def copy_trusted_price_book(cls, value: object) -> PriceBook | None:
        if value is None:
            return None
        if type(value) is not PriceBook:
            raise TypeError("CorpusTarget price_book must be an exact PriceBook or None.")
        return copy_price_book(value)

    @field_validator("limits", mode="before")
    @classmethod
    def copy_limits(cls, value: object) -> CorpusExecutionLimits:
        if type(value) is not CorpusExecutionLimits:
            raise TypeError("CorpusTarget limits must be exact CorpusExecutionLimits.")
        return CorpusExecutionLimits.model_validate(value.model_dump(mode="python"))

    @model_validator(mode="after")
    def validate_authority_boundary(self) -> CorpusTarget:
        _require_public_target_text(self.app, self.key, "key")
        _require_public_target_text(
            self.app,
            self.application_release_id,
            "application_release_id",
        )
        identity_fields = (
            "session_id",
            "parent_session_id",
            "causal_budget_id",
            "task_id",
            "task_worker_id",
        )
        populated = tuple(
            field_name for field_name in identity_fields if getattr(self.request_base, field_name)
        )
        if populated:
            raise ValueError(
                "CorpusTarget request_base cannot carry runtime identity fields: "
                + ", ".join(populated)
                + "."
            )
        if self.request_base.structured_output is not None:
            raise ValueError("CorpusTarget request_base cannot request structured output.")
        if self.request_base._runtime_generated_authority:
            raise ValueError("CorpusTarget request_base cannot carry runtime-generated authority.")
        if self.request_base._input_redactions_applied:
            raise ValueError("CorpusTarget request_base cannot carry prior input-redaction state.")
        if not json_utf8_size_within_limit(
            self.request_base.model_dump(mode="json"),
            CORPUS_EXECUTION_MAX_REQUEST_BASE_BYTES,
        ):
            raise ValueError(
                "CorpusTarget request_base exceeds "
                f"{CORPUS_EXECUTION_MAX_REQUEST_BASE_BYTES} canonical JSON bytes."
            )
        if self.price_book is not None:
            pricing_profile_identity(self.price_book)
        if len(self.bootstrap_messages) > self.limits.max_bootstrap_messages:
            raise ValueError("CorpusTarget bootstrap messages exceed its configured limit.")
        bootstrap_chars = sum(
            len(_bootstrap_message_text(message)) for message in self.bootstrap_messages
        )
        if bootstrap_chars > self.limits.max_total_input_chars:
            raise ValueError("CorpusTarget bootstrap text exceeds its configured input limit.")
        if bootstrap_chars > self.limits.max_compiled_input_chars:
            raise ValueError("CorpusTarget bootstrap text exceeds its compiled-suite input limit.")
        return self


@dataclass(frozen=True)
class CompiledCorpusSuite:
    """One validated corpus suite ready for the existing evaluator."""

    corpus: EvalCorpusDocument
    suite: EvalSuite
    run_contract: EvalRunContractV1
    trials: int
    timeout_seconds: int


def _copy_corpus_target(target: CorpusTarget) -> CorpusTarget:
    if type(target) is not CorpusTarget:
        raise TypeError("target must be an exact CorpusTarget.")
    return CorpusTarget(
        key=target.key,
        app=target.app,
        request_base=target.request_base,
        bootstrap_messages=target.bootstrap_messages,
        application_release_id=target.application_release_id,
        evidence_policy=target.evidence_policy,
        price_book=target.price_book,
        limits=target.limits,
    )


def evaluation_target_identity(target: CorpusTarget) -> EvaluationTargetIdentity:
    """Describe a validated target without invoking application dependencies."""

    validated = _copy_corpus_target(target)
    manifest = validated.app.describe()
    if type(manifest) is not AppManifest:
        raise TypeError("CayuApp.describe() must return an AppManifest.")
    return EvaluationTargetIdentity(
        target_key=validated.key,
        application_release_id=validated.application_release_id,
        app_manifest=manifest,
    )


def compile_corpus_suite(
    corpus: EvalCorpusDocument,
    target: CorpusTarget,
    suite_id: str,
) -> CompiledCorpusSuite:
    """Compile authority-free corpus data against one trusted local target."""

    if type(corpus) is not EvalCorpusDocument:
        raise TypeError("corpus must be an exact EvalCorpusDocument.")
    validated_corpus = EvalCorpusDocument.model_validate(_model_python_input(corpus))
    validated_target = _copy_corpus_target(target)
    validated_suite_id = _portable_id(suite_id, "suite_id")
    if validated_corpus.target_key != validated_target.key:
        raise ValueError("Eval corpus target key does not match the trusted CorpusTarget.")
    if validated_corpus.evidence_policy != validated_target.evidence_policy:
        raise ValueError("Eval corpus evidence policy does not match the trusted CorpusTarget.")

    suite_spec = next(
        (suite for suite in validated_corpus.suites if suite.id == validated_suite_id),
        None,
    )
    if suite_spec is None:
        raise ValueError(f"Eval corpus does not contain suite {validated_suite_id!r}.")
    case_specs = tuple(
        case for case in validated_corpus.cases if case.suite_id == validated_suite_id
    )
    if len(case_specs) > validated_target.limits.max_cases:
        raise ValueError("Eval corpus suite exceeds the trusted target case limit.")
    if suite_spec.trial_request.trials > validated_target.limits.max_trials:
        raise ValueError("Eval corpus suite exceeds the trusted target trial limit.")
    if suite_spec.trial_request.timeout_seconds > validated_target.limits.max_timeout_seconds:
        raise ValueError("Eval corpus suite exceeds the trusted target timeout limit.")

    bootstrap_chars = sum(
        len(_bootstrap_message_text(message)) for message in validated_target.bootstrap_messages
    )
    assertion_counts = tuple(len(case.assertions) for case in case_specs)
    compiled_assertions = iter(
        _compile_corpus_assertion_specs(
            tuple(assertion for case in case_specs for assertion in case.assertions),
            app=validated_target.app,
            evidence_policy=validated_target.evidence_policy,
            trusted_pricing=validated_target.price_book,
            expected_pricing_profile=validated_corpus.pricing_profile,
        )
    )
    compiled_input_chars = 0
    compiled_cases: list[EvalCase] = []
    for case_spec, assertion_count in zip(case_specs, assertion_counts, strict=True):
        corpus_messages = tuple(
            Message.text(MessageRole.USER, message.text) for message in case_spec.input.messages
        )
        total_input_chars = bootstrap_chars + sum(
            len(message.text) for message in case_spec.input.messages
        )
        if total_input_chars > validated_target.limits.max_total_input_chars:
            raise ValueError(
                f"Eval corpus case {case_spec.id!r} exceeds the trusted target input limit."
            )
        compiled_input_chars += total_input_chars
        if compiled_input_chars > validated_target.limits.max_compiled_input_chars:
            raise ValueError("Eval corpus suite exceeds the trusted target compiled-input limit.")
        request = validated_target.request_base.model_copy(
            update={
                "messages": [
                    *validated_target.bootstrap_messages,
                    *corpus_messages,
                ]
            }
        )
        request = copy_run_request(request)
        assertions = [next(compiled_assertions) for _ in range(assertion_count)]
        compiled_cases.append(
            EvalCase(
                id=case_spec.id,
                request=request,
                assertions=assertions,
                metadata={
                    "corpus_revision": validated_corpus.revision,
                    "case_revision": case_spec.revision,
                },
            )
        )
    suite = EvalSuite(
        id=suite_spec.id,
        cases=compiled_cases,
        metadata={
            "corpus_revision": validated_corpus.revision,
            "suite_revision": suite_spec.revision,
        },
    )
    return CompiledCorpusSuite(
        corpus=validated_corpus,
        suite=suite,
        run_contract=eval_run_contract_for_corpus(validated_corpus, suite_spec.id),
        trials=suite_spec.trial_request.trials,
        timeout_seconds=suite_spec.trial_request.timeout_seconds,
    )


class CorpusExecutionResult(BaseModel):
    """Safe published run plus the fresh target identity used to produce it."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    schema_version: Literal[1] = CORPUS_EXECUTION_RESULT_SCHEMA_VERSION
    revision: StrictStr
    target: EvaluationTargetIdentity
    run: PublishedEvalRun

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator("revision")
    @classmethod
    def validate_revision_shape(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("target", mode="before")
    @classmethod
    def copy_target(cls, value: object) -> object:
        if type(value) is EvaluationTargetIdentity:
            return EvaluationTargetIdentity(
                target_key=value.target_key,
                application_release_id=value.application_release_id,
                app_manifest=value.app_manifest,
            )
        if isinstance(value, BaseModel):
            raise TypeError("target must be an exact EvaluationTargetIdentity or JSON object.")
        return value

    @field_validator("run", mode="before")
    @classmethod
    def copy_run(cls, value: object) -> object:
        if type(value) is PublishedEvalRun:
            return PublishedEvalRun.model_validate(
                value.model_dump(mode="python", round_trip=True, warnings="none")
            )
        if isinstance(value, BaseModel):
            raise TypeError("run must be an exact PublishedEvalRun or JSON object.")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> CorpusExecutionResult:
        if self.target.target_key != self.run.target_key:
            raise ValueError("Execution target key does not match the published eval run.")
        if any(
            trial.status in {"passed", "failed"} and trial.output.evidence_state == "unavailable"
            for case in self.run.cases
            for trial in case.trials
        ):
            raise ValueError(
                "Scored corpus execution trials require retained redacted output evidence."
            )
        if not json_utf8_size_within_limit(self, CORPUS_EXECUTION_RESULT_MAX_BYTES):
            raise ValueError(
                "Corpus execution result exceeds "
                f"{CORPUS_EXECUTION_RESULT_MAX_BYTES} canonical JSON bytes."
            )
        if self.revision != _model_content_revision(self, "corpus execution result"):
            raise ValueError("Corpus execution result revision does not match its content.")
        return self

    @classmethod
    def create(
        cls,
        *,
        target: EvaluationTargetIdentity,
        run: PublishedEvalRun,
    ) -> CorpusExecutionResult:
        if type(target) is not EvaluationTargetIdentity:
            raise TypeError("target must be an exact EvaluationTargetIdentity.")
        if type(run) is not PublishedEvalRun:
            raise TypeError("run must be an exact PublishedEvalRun.")
        document = {
            "schema_version": CORPUS_EXECUTION_RESULT_SCHEMA_VERSION,
            "target": target.model_dump(mode="json"),
            "run": run.model_dump(mode="json"),
        }
        return cls(
            revision=_content_revision(document, "corpus execution result"),
            target=target,
            run=run,
        )


async def run_corpus_suite(
    target: CorpusTarget,
    corpus: EvalCorpusDocument,
    suite_id: str,
    *,
    max_concurrency: int = 1,
) -> CorpusExecutionResult:
    """Execute one corpus suite through Cayu's existing runner and publish it safely."""

    validated_target = _copy_corpus_target(target)
    if type(max_concurrency) is not int:
        raise TypeError("max_concurrency must be an int.")
    if not 1 <= max_concurrency <= validated_target.limits.max_concurrency:
        raise ValueError(
            "max_concurrency must be between 1 and the trusted target concurrency limit."
        )
    compiled = compile_corpus_suite(corpus, validated_target, suite_id)
    target_before = evaluation_target_identity(validated_target)
    trial_count = len(compiled.suite.cases) * compiled.trials
    output_preview_bytes = min(
        EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES,
        PUBLISHED_EVAL_OUTPUT_PREVIEW_BUDGET_BYTES // trial_count,
    )
    internal_run, trial_public_data_by_case = await _run_eval_suite_with_public_projection(
        validated_target.app,
        compiled.suite,
        max_concurrency=max_concurrency,
        case_timeout_seconds=compiled.timeout_seconds,
        trials=compiled.trials,
        output_preview_bytes=output_preview_bytes,
    )
    target_after = evaluation_target_identity(validated_target)
    if target_after != target_before:
        raise RuntimeError("CorpusTarget application manifest changed during eval execution.")
    run_document: dict[str, Any] = _model_instance_python_input(internal_run)
    run_document["run_contract"] = compiled.run_contract
    bound_run = EvalRun.model_validate(run_document)
    return CorpusExecutionResult.create(
        target=target_before,
        run=_publish_eval_run_with_trial_public_data(
            compiled.corpus,
            bound_run,
            trial_public_data_by_case=trial_public_data_by_case,
        ),
    )
