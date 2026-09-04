from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._validation import copy_durable_json_object, json_utf8_size_within_limit
from cayu.core.messages import Message, MessageRole, TextPart, detach_message
from cayu.core.workflows import WorkflowSpec, copy_workflow_spec
from cayu.evals._execution_profile_errors import EvalExecutionProfileChangedError
from cayu.evals.capacity import (
    DEFAULT_EVAL_MAX_ACTIVE_TRIALS,
    EVAL_MAX_CONCURRENCY,
    EvalExecutionCapacity,
)
from cayu.evals.corpus import (
    EVAL_CORPUS_MAX_CASES,
    EVAL_CORPUS_MAX_MESSAGE_CHARS,
    EVAL_CORPUS_MAX_MESSAGES_PER_CASE,
    EVAL_CORPUS_MAX_TIMEOUT_SECONDS,
    EVAL_CORPUS_MAX_TOTAL_MESSAGE_CHARS,
    EVAL_CORPUS_MAX_TRIALS,
    EVIDENCE_MAX_TOTAL_TOKENS,
    EvalCorpusDocument,
    EvaluationEvidencePolicySpec,
    JudgePrivacyPolicyV1,
    JudgeProfileIdentityV1,
    MaxEstimatedCostAssertionSpec,
    ModelJudgeAssertionSpec,
    PricingProfileIdentityV1,
    PrivateJudgeReferenceV1,
    StructuredModelJudgeAssertionSpec,
    _bounded_durable_text,
    _canonical_decimal_text,
    _content_revision,
    _model_content_revision,
    _model_python_input,
    _portable_id,
    _sha256_revision,
    eval_run_contract_for_corpus,
    pricing_profile_identity,
)
from cayu.evals.external import (
    ExternalProcessTargetIdentityV1,
    ExternalTrialEnvelopeV1,
    ExternalTrialIdentityV1,
    with_external_trial_envelope,
)
from cayu.evals.models import (
    EvalRun,
    EvalRunContractV2,
    EvalTrialResult,
    _model_instance_python_input,
)
from cayu.evals.portable_assertions import (
    _compile_corpus_assertion_specs,
    _model_judge_implementation_revision,
    _trusted_model_judge_binding,
    _TrustedPrivateJudgeReferenceBinding,
)
from cayu.evals.published import PublishedEvalRun, _publish_eval_run_with_trial_public_data
from cayu.evals.result_contract import (
    EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES,
    PUBLISHED_EVAL_OUTPUT_PREVIEW_BUDGET_BYTES,
    _EvalTrialPublicData,
)
from cayu.evals.runner import EvalCase, EvalSuite, _run_eval_suite_with_public_projection
from cayu.evals.trial_policy import EvalSuiteRunExposureV1
from cayu.evals.workflow_target import (
    WORKFLOW_EVAL_DEFAULT_CLOSE_TIMEOUT_SECONDS,
    WORKFLOW_EVAL_MAX_APPLICATION_CONTEXT_BYTES,
    WORKFLOW_EVAL_MAX_CLOSE_TIMEOUT_SECONDS,
    WorkflowEvalFactory,
    WorkflowEvalInstanceScope,
    WorkflowEvalResultProjector,
    WorkflowEvalTargetIdentityV1,
)
from cayu.runtime.app import CayuApp
from cayu.runtime.costs import PriceBook, copy_price_book
from cayu.runtime.execution_profiles import ExecutionProfileIdentity
from cayu.runtime.manifest import AppManifest, _app_manifest_fingerprint
from cayu.runtime.sessions import RunRequest, copy_run_request

CORPUS_EXECUTION_MAX_BOOTSTRAP_MESSAGES = EVAL_CORPUS_MAX_MESSAGES_PER_CASE
CORPUS_EXECUTION_MAX_TOTAL_INPUT_CHARS = EVAL_CORPUS_MAX_TOTAL_MESSAGE_CHARS * 2
CORPUS_EXECUTION_MAX_COMPILED_INPUT_CHARS = 8 << 20
CORPUS_EXECUTION_DEFAULT_MAX_CONCURRENCY = DEFAULT_EVAL_MAX_ACTIVE_TRIALS
# Backwards-compatible import name. This value is the default target authority,
# not a Runtime-wide ceiling; callers may configure a larger finite value.
CORPUS_EXECUTION_MAX_CONCURRENCY = CORPUS_EXECUTION_DEFAULT_MAX_CONCURRENCY
CORPUS_EXECUTION_MAX_APP_MANIFEST_BYTES = 1 << 20
CORPUS_EXECUTION_MAX_REQUEST_BASE_BYTES = 64 << 10
CORPUS_EXECUTION_RESULT_MAX_BYTES = 40 << 20
CORPUS_EXECUTION_RESULT_SCHEMA_VERSION = 3
CORPUS_EXECUTION_MAX_MODEL_JUDGES = 32
MODEL_JUDGE_MAX_PRIVATE_REFERENCES = 256
MODEL_JUDGE_MAX_PRIVATE_REFERENCE_CHARS = 65_536


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
        default=CORPUS_EXECUTION_DEFAULT_MAX_CONCURRENCY,
        ge=1,
        le=EVAL_MAX_CONCURRENCY,
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
    external_process: ExternalProcessTargetIdentityV1 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    workflow: WorkflowEvalTargetIdentityV1 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

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

    @field_validator("external_process", mode="before")
    @classmethod
    def copy_external_process(cls, value: object) -> object:
        if type(value) is ExternalProcessTargetIdentityV1:
            return value.model_dump(mode="json")
        return value

    @field_validator("workflow", mode="before")
    @classmethod
    def copy_workflow(cls, value: object) -> object:
        if type(value) is WorkflowEvalTargetIdentityV1:
            return value.model_dump(mode="json")
        return value

    @model_validator(mode="after")
    def validate_app_manifest(self) -> EvaluationTargetIdentity:
        if self.external_process is not None and self.workflow is not None:
            raise ValueError("An eval target cannot be both external-process and workflow-root.")
        fingerprint = self.app_manifest.fingerprint
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise ValueError("app_manifest_fingerprint must be a lowercase SHA-256 hex digest.")
        expected_fingerprint = _app_manifest_fingerprint(self.app_manifest.model_dump(mode="json"))
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


class PrivateJudgeReferenceTarget(BaseModel):
    """Trusted evaluator-only reference content that never enters a corpus or result."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    key: StrictStr
    revision: StrictStr
    content: StrictStr = Field(repr=False)
    privacy_policy_key: StrictStr
    privacy_policy_revision: StrictStr

    @field_validator("key", "privacy_policy_key")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("revision", "privacy_policy_revision")
    @classmethod
    def validate_revisions(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=MODEL_JUDGE_MAX_PRIVATE_REFERENCE_CHARS,
            nonblank=True,
            clean=False,
        )

    @model_validator(mode="after")
    def validate_content_revision(self) -> PrivateJudgeReferenceTarget:
        expected = _content_revision(
            {
                "key": self.key,
                "content": self.content,
                "privacy_policy_key": self.privacy_policy_key,
                "privacy_policy_revision": self.privacy_policy_revision,
            },
            "private judge reference",
        )
        if self.revision != expected:
            raise ValueError("Private judge reference revision does not match its content.")
        return self

    @classmethod
    def create(
        cls,
        *,
        key: str,
        content: str,
        privacy_policy: JudgePrivacyPolicyV1,
    ) -> PrivateJudgeReferenceTarget:
        if type(privacy_policy) is not JudgePrivacyPolicyV1:
            raise TypeError("privacy_policy must be an exact JudgePrivacyPolicyV1.")
        validated_policy = JudgePrivacyPolicyV1.model_validate(_model_python_input(privacy_policy))
        document = {
            "key": key,
            "content": content,
            "privacy_policy_key": validated_policy.key,
            "privacy_policy_revision": validated_policy.revision,
        }
        return cls(
            revision=_content_revision(document, "private judge reference"),
            **document,
        )

    def portable_identity(self) -> PrivateJudgeReferenceV1:
        return PrivateJudgeReferenceV1(
            key=self.key,
            revision=self.revision,
            privacy_policy_key=self.privacy_policy_key,
            privacy_policy_revision=self.privacy_policy_revision,
        )


class ModelJudgeTarget(BaseModel):
    """Trusted local execution authority for one portable evaluator key."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    key: StrictStr
    app: CayuApp
    agent_name: StrictStr
    label: StrictStr | None = None
    privacy_policy: JudgePrivacyPolicyV1 = Field(default_factory=JudgePrivacyPolicyV1.public_only)
    private_references: tuple[PrivateJudgeReferenceTarget, ...] = Field(
        default=(),
        max_length=MODEL_JUDGE_MAX_PRIVATE_REFERENCES,
    )
    timeout_seconds: StrictInt = Field(default=120, ge=1, le=EVAL_CORPUS_MAX_TIMEOUT_SECONDS)
    max_input_tokens: StrictInt = Field(
        default=32_768,
        ge=1,
        le=EVIDENCE_MAX_TOTAL_TOKENS,
    )
    max_output_tokens: StrictInt = Field(
        default=4_096,
        ge=1,
        le=EVIDENCE_MAX_TOTAL_TOKENS,
    )
    max_total_tokens: StrictInt = Field(
        default=36_864,
        ge=1,
        le=EVIDENCE_MAX_TOTAL_TOKENS,
    )
    max_estimated_cost: StrictStr | None = None
    cost_currency: StrictStr = "USD"
    price_book: PriceBook | None = None
    allow_same_model: StrictBool = False

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        return _portable_id(value, "key")

    @field_validator("app")
    @classmethod
    def validate_app(cls, value: CayuApp) -> CayuApp:
        if not isinstance(value, CayuApp):
            raise TypeError("ModelJudgeTarget app must be a CayuApp.")
        return value

    @field_validator("agent_name")
    @classmethod
    def validate_agent_name(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @field_validator("privacy_policy", mode="before")
    @classmethod
    def copy_privacy_policy(cls, value: object) -> JudgePrivacyPolicyV1:
        if type(value) is not JudgePrivacyPolicyV1:
            raise TypeError("privacy_policy must be an exact JudgePrivacyPolicyV1.")
        return JudgePrivacyPolicyV1.model_validate(_model_python_input(value))

    @field_validator("private_references", mode="before")
    @classmethod
    def copy_private_references(cls, value: object) -> tuple[PrivateJudgeReferenceTarget, ...]:
        if not isinstance(value, list | tuple):
            raise TypeError("private_references must be an ordered list or tuple.")
        copied: list[PrivateJudgeReferenceTarget] = []
        for reference in value:
            if type(reference) is not PrivateJudgeReferenceTarget:
                raise TypeError(
                    "private_references must contain exact PrivateJudgeReferenceTarget values."
                )
            copied.append(PrivateJudgeReferenceTarget.model_validate(reference.model_dump()))
        keys = tuple(reference.key for reference in copied)
        if len(keys) != len(set(keys)):
            raise ValueError("Private judge reference keys must be unique.")
        return tuple(copied)

    @field_validator("max_estimated_cost")
    @classmethod
    def validate_max_estimated_cost(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        value = _canonical_decimal_text(value, info.field_name)
        if Decimal(value) <= 0:
            raise ValueError("max_estimated_cost must be greater than zero.")
        return value

    @field_validator("cost_currency")
    @classmethod
    def validate_cost_currency(cls, value: str, info) -> str:
        value = _bounded_durable_text(
            value,
            info.field_name,
            max_chars=16,
            nonblank=True,
            clean=True,
        )
        if (
            not value[0].isalpha()
            or not value.isascii()
            or not all(
                character.isupper() or character.isdigit() or character in "._-"
                for character in value
            )
        ):
            raise ValueError("cost_currency must be a portable uppercase identifier.")
        return value

    @field_validator("price_book", mode="before")
    @classmethod
    def copy_price_book(cls, value: object) -> PriceBook | None:
        if value is None:
            return None
        if type(value) is not PriceBook:
            raise TypeError("price_book must be an exact PriceBook or None.")
        return copy_price_book(value)

    @model_validator(mode="after")
    def validate_authority_boundary(self) -> ModelJudgeTarget:
        _require_public_target_text(self.app, self.key, "model_judges.key")
        _require_public_target_text(
            self.app,
            self.label or self.key,
            "model_judges.label",
        )
        _require_public_target_text(
            self.app,
            self.privacy_policy.key,
            "model_judges.privacy_policy.key",
        )
        if self.max_total_tokens < max(self.max_input_tokens, self.max_output_tokens):
            raise ValueError("max_total_tokens cannot be below an individual token ceiling.")
        if (self.max_estimated_cost is None) != (self.price_book is None):
            raise ValueError(
                "A model-judge cost ceiling requires both max_estimated_cost and price_book."
            )
        if self.price_book is not None:
            identity = pricing_profile_identity(self.price_book)
            if self.cost_currency not in identity.currencies:
                raise ValueError("Model-judge cost currency is absent from its price book.")
        if self.private_references and not self.privacy_policy.allow_private_reference:
            raise ValueError("Private references require a policy that permits them.")
        for reference in self.private_references:
            _require_public_target_text(
                self.app,
                reference.key,
                "model_judges.private_references.key",
            )
            if (
                reference.privacy_policy_key,
                reference.privacy_policy_revision,
            ) != (self.privacy_policy.key, self.privacy_policy.revision):
                raise ValueError("Private reference privacy policy does not match the judge.")
        _trusted_model_judge_binding(
            key=self.key,
            app=self.app,
            agent_name=self.agent_name,
        )
        return self


def _copy_model_judge_target(target: ModelJudgeTarget) -> ModelJudgeTarget:
    if type(target) is not ModelJudgeTarget:
        raise TypeError("target must be an exact ModelJudgeTarget.")
    return ModelJudgeTarget(
        key=target.key,
        app=target.app,
        agent_name=target.agent_name,
        label=target.label,
        privacy_policy=target.privacy_policy,
        private_references=target.private_references,
        timeout_seconds=target.timeout_seconds,
        max_input_tokens=target.max_input_tokens,
        max_output_tokens=target.max_output_tokens,
        max_total_tokens=target.max_total_tokens,
        max_estimated_cost=target.max_estimated_cost,
        cost_currency=target.cost_currency,
        price_book=target.price_book,
        allow_same_model=target.allow_same_model,
    )


def model_judge_implementation_revision(target: ModelJudgeTarget) -> str:
    """Return the implementation identity a trusted target will publish."""

    validated = _copy_model_judge_target(target)
    return _model_judge_implementation_revision(
        key=validated.key,
        app=validated.app,
        agent_name=validated.agent_name,
    )


def model_judge_profile(target: ModelJudgeTarget) -> JudgeProfileIdentityV1:
    """Publish the bounded identity and ceilings of one explicit judge route."""

    validated = _copy_model_judge_target(target)
    manifest = validated.app.describe()
    agent = next(item for item in manifest.agents if item.name == validated.agent_name)
    if agent.resolved_provider is None:
        raise ValueError("Trusted model judge must resolve exactly one provider.")
    provider_name = agent.resolved_provider
    model = agent.model
    allowed_evidence = tuple(
        item
        for item, allowed in (
            ("final_output", True),
            ("transcript", validated.privacy_policy.allow_transcript),
            ("public_reference", validated.privacy_policy.allow_public_reference),
            ("private_reference", validated.privacy_policy.allow_private_reference),
        )
        if allowed
    )
    pricing = (
        None
        if validated.price_book is None
        else pricing_profile_identity(validated.price_book).fingerprint
    )
    implementation_revision = model_judge_implementation_revision(validated)
    document = {
        "schema_version": 1,
        "key": validated.key,
        "label": validated.label or validated.key,
        "provider_name": provider_name,
        "model": model,
        "implementation_revision": implementation_revision,
        "allowed_evidence": list(allowed_evidence),
        "timeout_seconds": validated.timeout_seconds,
        "max_input_tokens": validated.max_input_tokens,
        "max_output_tokens": validated.max_output_tokens,
        "max_total_tokens": validated.max_total_tokens,
        "max_estimated_cost": validated.max_estimated_cost,
        "cost_currency": (
            validated.cost_currency if validated.max_estimated_cost is not None else None
        ),
        "pricing_profile_fingerprint": pricing,
        "privacy_policy_key": validated.privacy_policy.key,
        "privacy_policy_revision": validated.privacy_policy.revision,
        "same_model_use": ("allowed_and_labeled" if validated.allow_same_model else "forbidden"),
    }
    _require_public_target_text(validated.app, provider_name, "model_judges.provider_name")
    _require_public_target_text(validated.app, model, "model_judges.model")
    return JudgeProfileIdentityV1(
        revision=_content_revision(document, "judge profile identity"),
        key=validated.key,
        label=validated.label or validated.key,
        provider_name=provider_name,
        model=model,
        implementation_revision=implementation_revision,
        allowed_evidence=allowed_evidence,
        timeout_seconds=validated.timeout_seconds,
        max_input_tokens=validated.max_input_tokens,
        max_output_tokens=validated.max_output_tokens,
        max_total_tokens=validated.max_total_tokens,
        max_estimated_cost=validated.max_estimated_cost,
        cost_currency=(
            validated.cost_currency if validated.max_estimated_cost is not None else None
        ),
        pricing_profile_fingerprint=pricing,
        privacy_policy_key=validated.privacy_policy.key,
        privacy_policy_revision=validated.privacy_policy.revision,
        same_model_use=("allowed_and_labeled" if validated.allow_same_model else "forbidden"),
    )


def _candidate_judge_route_relation(
    target: CorpusTarget,
    judge_profile: JudgeProfileIdentityV1,
) -> Literal["independent_model", "same_model"]:
    request_target = target.request_base.target
    if request_target is None:
        manifest = target.app.describe()
        candidate = next(
            (agent for agent in manifest.agents if agent.name == target.request_base.agent_name),
            None,
        )
        if candidate is None or candidate.resolved_provider is None:
            raise ValueError("CorpusTarget candidate agent must resolve exactly one provider.")
        candidate_route = (candidate.resolved_provider, candidate.model)
    else:
        candidate_route = (request_target.provider_name, request_target.model)
    return (
        "same_model"
        if candidate_route == (judge_profile.provider_name, judge_profile.model)
        else "independent_model"
    )


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
    model_judges: tuple[ModelJudgeTarget, ...] = Field(
        default=(),
        max_length=CORPUS_EXECUTION_MAX_MODEL_JUDGES,
    )
    limits: CorpusExecutionLimits = Field(default_factory=CorpusExecutionLimits)
    external_process: ExternalProcessTargetIdentityV1 | None = Field(
        default=None,
        exclude=True,
    )

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

    @field_validator("model_judges", mode="before")
    @classmethod
    def copy_model_judges(cls, value: object) -> tuple[ModelJudgeTarget, ...]:
        if not isinstance(value, list | tuple):
            raise TypeError("CorpusTarget model_judges must be an ordered list or tuple.")
        copied: list[ModelJudgeTarget] = []
        for item in value:
            if type(item) is not ModelJudgeTarget:
                raise TypeError(
                    "CorpusTarget model_judges must contain exact ModelJudgeTarget values."
                )
            copied.append(_copy_model_judge_target(item))
        keys = tuple(item.key for item in copied)
        if len(keys) != len(set(keys)):
            raise ValueError("CorpusTarget model judge keys must be unique.")
        return tuple(copied)

    @field_validator("limits", mode="before")
    @classmethod
    def copy_limits(cls, value: object) -> CorpusExecutionLimits:
        if type(value) is not CorpusExecutionLimits:
            raise TypeError("CorpusTarget limits must be exact CorpusExecutionLimits.")
        return CorpusExecutionLimits.model_validate(value.model_dump(mode="python"))

    @field_validator("external_process", mode="before")
    @classmethod
    def copy_external_process(cls, value: object) -> ExternalProcessTargetIdentityV1 | None:
        if value is None:
            return None
        if type(value) is not ExternalProcessTargetIdentityV1:
            raise TypeError(
                "CorpusTarget external_process must be an exact "
                "ExternalProcessTargetIdentityV1 or None."
            )
        return ExternalProcessTargetIdentityV1.model_validate(value.model_dump(mode="json"))

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
        if (
            self.external_process is not None
            and self.external_process.evidence_policy_revision != self.evidence_policy.revision
        ):
            raise ValueError(
                "External process target evidence policy does not match its CorpusTarget."
            )
        return self


class WorkflowEvalTarget(CorpusTarget):
    """Trusted application-owned workflow-root execution target.

    ``request_base`` and ``bootstrap_messages`` remain the shared corpus input
    compiler contract. The request's agent name is only a profile-probe identity;
    candidate execution is performed exclusively by ``workflow_factory``.
    """

    workflow_spec: WorkflowSpec
    implementation_revision: StrictStr
    result_projector_revision: StrictStr
    execution_scope_revision: StrictStr
    instance_scope: WorkflowEvalInstanceScope = WorkflowEvalInstanceScope.PER_TRIAL
    workflow_factory: WorkflowEvalFactory = Field(exclude=True, repr=False)
    result_projector: WorkflowEvalResultProjector = Field(exclude=True, repr=False)
    application_context: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)
    close_timeout_seconds: float = WORKFLOW_EVAL_DEFAULT_CLOSE_TIMEOUT_SECONDS

    @field_validator("workflow_spec", mode="before")
    @classmethod
    def copy_workflow_spec(cls, value: object) -> WorkflowSpec:
        if not isinstance(value, WorkflowSpec):
            raise TypeError("WorkflowEvalTarget workflow_spec must be a WorkflowSpec.")
        return copy_workflow_spec(value)

    @field_validator(
        "implementation_revision",
        "result_projector_revision",
        "execution_scope_revision",
    )
    @classmethod
    def validate_workflow_revisions(cls, value: str, info) -> str:
        if (
            len(value) != 71
            or not value.startswith("sha256:")
            or any(character not in "0123456789abcdef" for character in value[7:])
        ):
            raise ValueError(f"{info.field_name} must be a sha256 revision.")
        return value

    @field_validator("workflow_factory", "result_projector")
    @classmethod
    def validate_callbacks(cls, value: object, info) -> object:
        if not callable(value):
            raise TypeError(f"WorkflowEvalTarget {info.field_name} must be callable.")
        return value

    @field_validator("application_context", mode="before")
    @classmethod
    def copy_application_context(cls, value: object) -> dict[str, Any]:
        copied = copy_durable_json_object(value, "application_context")
        if not json_utf8_size_within_limit(
            copied,
            WORKFLOW_EVAL_MAX_APPLICATION_CONTEXT_BYTES,
        ):
            raise ValueError("application_context exceeds its canonical JSON byte limit.")
        return copied

    @field_validator("close_timeout_seconds", mode="before")
    @classmethod
    def validate_close_timeout(cls, value: object) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= WORKFLOW_EVAL_MAX_CLOSE_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "close_timeout_seconds must be a finite positive number no greater than "
                f"{WORKFLOW_EVAL_MAX_CLOSE_TIMEOUT_SECONDS}."
            )
        return float(value)

    @model_validator(mode="after")
    def validate_workflow_contract(self) -> WorkflowEvalTarget:
        if self.external_process is not None:
            raise ValueError("WorkflowEvalTarget cannot also configure an external process.")
        # Constructing the identity here catches changed/malformed revisions before
        # corpus compilation or candidate/provider dispatch.
        self.identity()
        return self

    def identity(self) -> WorkflowEvalTargetIdentityV1:
        return WorkflowEvalTargetIdentityV1.create(
            workflow_spec=self.workflow_spec,
            implementation_revision=self.implementation_revision,
            result_projector_revision=self.result_projector_revision,
            execution_scope_revision=self.execution_scope_revision,
            application_context=self.application_context,
            evidence_policy_revision=self.evidence_policy.revision,
            instance_scope=self.instance_scope,
            close_timeout_seconds=self.close_timeout_seconds,
        )


@dataclass(frozen=True)
class CompiledCorpusSuite:
    """One validated corpus suite ready for the existing evaluator."""

    corpus: EvalCorpusDocument
    suite: EvalSuite
    run_contract: EvalRunContractV2
    trials: int
    timeout_seconds: int


@dataclass(frozen=True)
class _CorpusCompilationContext:
    corpus: EvalCorpusDocument
    target: CorpusTarget


def _copy_corpus_target(target: CorpusTarget) -> CorpusTarget:
    if type(target) is WorkflowEvalTarget:
        return WorkflowEvalTarget(
            key=target.key,
            app=target.app,
            request_base=target.request_base,
            bootstrap_messages=target.bootstrap_messages,
            application_release_id=target.application_release_id,
            evidence_policy=target.evidence_policy,
            price_book=target.price_book,
            model_judges=target.model_judges,
            limits=target.limits,
            workflow_spec=target.workflow_spec,
            implementation_revision=target.implementation_revision,
            result_projector_revision=target.result_projector_revision,
            execution_scope_revision=target.execution_scope_revision,
            instance_scope=target.instance_scope,
            workflow_factory=target.workflow_factory,
            result_projector=target.result_projector,
            application_context=target.application_context,
            close_timeout_seconds=target.close_timeout_seconds,
        )
    if type(target) is not CorpusTarget:
        raise TypeError("target must be an exact CorpusTarget or WorkflowEvalTarget.")
    return CorpusTarget(
        key=target.key,
        app=target.app,
        request_base=target.request_base,
        bootstrap_messages=target.bootstrap_messages,
        application_release_id=target.application_release_id,
        evidence_policy=target.evidence_policy,
        price_book=target.price_book,
        model_judges=target.model_judges,
        limits=target.limits,
        external_process=target.external_process,
    )


def evaluation_target_identity(
    target: CorpusTarget,
    *,
    project_root: str | Path | None = None,
) -> EvaluationTargetIdentity:
    """Describe a validated target without invoking application dependencies."""

    validated = _copy_corpus_target(target)
    return _evaluation_target_identity_from_validated_target(
        validated,
        project_root=project_root,
    )


def _evaluation_target_identity_from_validated_target(
    target: CorpusTarget,
    *,
    project_root: str | Path | None = None,
) -> EvaluationTargetIdentity:
    """Describe an internally validated target without another potentially large copy."""

    if type(target) not in {CorpusTarget, WorkflowEvalTarget}:
        raise TypeError("target must be an exact CorpusTarget or WorkflowEvalTarget.")
    manifest = (
        target.app.describe()
        if project_root is None
        else target.app.describe(project_root=project_root)
    )
    if type(manifest) is not AppManifest:
        raise TypeError("CayuApp.describe() must return an AppManifest.")
    return EvaluationTargetIdentity(
        target_key=target.key,
        application_release_id=target.application_release_id,
        app_manifest=manifest,
        external_process=target.external_process,
        workflow=(target.identity() if type(target) is WorkflowEvalTarget else None),
    )


def _prepare_corpus_compilation(
    corpus: EvalCorpusDocument,
    target: CorpusTarget,
) -> _CorpusCompilationContext:
    return _prepare_corpus_with_validated_target(corpus, _copy_corpus_target(target))


def _prepare_corpus_with_validated_target(
    corpus: EvalCorpusDocument,
    target: CorpusTarget,
) -> _CorpusCompilationContext:
    if type(corpus) is not EvalCorpusDocument:
        raise TypeError("corpus must be an exact EvalCorpusDocument.")
    if type(target) not in {CorpusTarget, WorkflowEvalTarget}:
        raise TypeError("target must be an exact CorpusTarget or WorkflowEvalTarget.")
    validated_corpus = EvalCorpusDocument.model_validate(_model_python_input(corpus))
    if validated_corpus.target_key != target.key:
        raise ValueError("Eval corpus target key does not match the trusted CorpusTarget.")
    if validated_corpus.evidence_policy != target.evidence_policy:
        raise ValueError("Eval corpus evidence policy does not match the trusted CorpusTarget.")
    return _CorpusCompilationContext(corpus=validated_corpus, target=target)


def _compile_prepared_corpus_suite(
    context: _CorpusCompilationContext,
    suite_id: str,
    *,
    trusted_pricing_identity: PricingProfileIdentityV1 | None = None,
) -> CompiledCorpusSuite:
    validated_corpus = context.corpus
    validated_target = context.target
    validated_suite_id = _portable_id(suite_id, "suite_id")

    suite_spec = next(
        (suite for suite in validated_corpus.suites if suite.id == validated_suite_id),
        None,
    )
    if suite_spec is None:
        raise ValueError(f"Eval corpus does not contain suite {validated_suite_id!r}.")
    case_specs = tuple(
        case for case in validated_corpus.cases if case.suite_id == validated_suite_id
    )
    if any(case.input is None for case in case_specs):
        raise ValueError(
            "Captured-only eval cases cannot execute until runnable input is authored."
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
    profiled_judge_keys: set[str] = set()
    for case in case_specs:
        for assertion in case.assertions:
            if type(assertion) is ModelJudgeAssertionSpec:
                profiled_judge_keys.add(assertion.evaluator_key)
            elif type(assertion) is StructuredModelJudgeAssertionSpec:
                profiled_judge_keys.add(assertion.judge_profile_key)
    trusted_model_judges = []
    for judge in validated_target.model_judges:
        profile = model_judge_profile(judge) if judge.key in profiled_judge_keys else None
        trusted_model_judges.append(
            _trusted_model_judge_binding(
                key=judge.key,
                app=judge.app,
                agent_name=judge.agent_name,
                profile=profile,
                privacy_policy=(judge.privacy_policy if profile is not None else None),
                private_references=(
                    tuple(
                        _TrustedPrivateJudgeReferenceBinding(
                            key=reference.key,
                            revision=reference.revision,
                            content=reference.content,
                            privacy_policy_key=reference.privacy_policy_key,
                            privacy_policy_revision=reference.privacy_policy_revision,
                        )
                        for reference in judge.private_references
                    )
                    if profile is not None
                    else ()
                ),
                price_book=judge.price_book if profile is not None else None,
                candidate_route_relation=(
                    _candidate_judge_route_relation(validated_target, profile)
                    if profile is not None
                    else "independent_model"
                ),
            )
        )
    compiled_assertions = iter(
        _compile_corpus_assertion_specs(
            tuple(assertion for case in case_specs for assertion in case.assertions),
            app=validated_target.app,
            evidence_policy=validated_target.evidence_policy,
            trusted_pricing=validated_target.price_book,
            expected_pricing_profile=validated_corpus.pricing_profile,
            trusted_pricing_identity=trusted_pricing_identity,
            trusted_model_judges=tuple(trusted_model_judges),
        )
    )
    compiled_input_chars = 0
    compiled_cases: list[EvalCase] = []
    for case_spec, assertion_count in zip(case_specs, assertion_counts, strict=True):
        case_input = case_spec.input
        if case_input is None:  # Narrowed by the suite-level admission check above.
            raise RuntimeError("Captured-only eval input passed fresh-execution admission.")
        if (
            case_input.opaque_external_case_ref is not None
            and validated_target.external_process is None
        ):
            raise ValueError("Opaque external case references require an external process target.")
        corpus_messages = tuple(
            Message.text(MessageRole.USER, message.text) for message in case_input.messages
        )
        total_input_chars = bootstrap_chars + sum(
            len(message.text) for message in case_input.messages
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


def compile_corpus_suite(
    corpus: EvalCorpusDocument,
    target: CorpusTarget,
    suite_id: str,
) -> CompiledCorpusSuite:
    """Compile authority-free corpus data against one trusted local target."""

    return _compile_prepared_corpus_suite(
        _prepare_corpus_compilation(corpus, target),
        suite_id,
    )


def _validate_corpus_target_compatibility(
    corpus: EvalCorpusDocument,
    target: CorpusTarget,
) -> None:
    """Validate every suite against one trusted target without repeated full copies."""

    context = _prepare_corpus_compilation(corpus, target)
    uses_pricing = any(
        type(assertion) is MaxEstimatedCostAssertionSpec
        for case in context.corpus.cases
        for assertion in case.assertions
    )
    trusted_pricing_identity = None
    if uses_pricing:
        if context.target.price_book is None:
            raise ValueError("Eval corpus pricing profile does not match the trusted CorpusTarget.")
        trusted_pricing_identity = pricing_profile_identity(context.target.price_book)
    for suite in context.corpus.suites:
        suite_cases = tuple(case for case in context.corpus.cases if case.suite_id == suite.id)
        if all(case.input is None for case in suite_cases):
            continue
        if any(case.input is None for case in suite_cases):
            raise ValueError("An eval suite cannot mix captured-only and runnable cases.")
        _compile_prepared_corpus_suite(
            context,
            suite.id,
            trusted_pricing_identity=trusted_pricing_identity,
        )


class CorpusExecutionResult(BaseModel):
    """Safe published run plus the fresh target identity used to produce it."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    schema_version: Literal[3] = CORPUS_EXECUTION_RESULT_SCHEMA_VERSION
    revision: StrictStr
    target: EvaluationTargetIdentity
    run: PublishedEvalRun
    external_trials: tuple[ExternalTrialIdentityV1, ...] = Field(
        default=(),
        max_length=EVAL_CORPUS_MAX_CASES * EVAL_CORPUS_MAX_TRIALS,
        exclude_if=lambda value: not value,
    )

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 3.")
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
                external_process=value.external_process,
                workflow=value.workflow,
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

    @field_validator("external_trials", mode="before")
    @classmethod
    def copy_external_trials(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise ValueError("external_trials must be an ordered array.")
        return [
            item.model_dump(mode="json") if type(item) is ExternalTrialIdentityV1 else item
            for item in value
        ]

    @model_validator(mode="after")
    def validate_contract(self) -> CorpusExecutionResult:
        if self.target.target_key != self.run.target_key:
            raise ValueError("Execution target key does not match the published eval run.")
        if self.target.external_process is None:
            if self.external_trials:
                raise ValueError("Ordinary corpus results cannot carry external trial identities.")
        else:
            if (
                self.target.external_process.evidence_policy_revision
                != self.run.evidence_policy_revision
            ):
                raise ValueError(
                    "External target evidence policy does not match the published run."
                )
            expected = tuple(
                (
                    case.case_id,
                    case.case_revision,
                    trial.trial_number,
                )
                for case in self.run.cases
                for trial in case.trials
            )
            observed = tuple(
                (trial.case_id, trial.case_revision, trial.trial_number)
                for trial in self.external_trials
            )
            if observed != expected:
                raise ValueError(
                    "External trial identities do not match the published case/trial order."
                )
            if any(
                trial.target_key != self.target.target_key
                or trial.target_revision != self.target.external_process.revision
                or trial.corpus_revision != self.run.corpus_revision
                or trial.suite_id != self.run.suite_id
                or trial.suite_revision != self.run.suite_revision
                for trial in self.external_trials
            ):
                raise ValueError(
                    "External trial identities do not match the published execution contract."
                )
            if len({trial.native_run_id for trial in self.external_trials}) != 1:
                raise ValueError("External trial identities require one exact native run ID.")
        if (
            self.target.workflow is not None
            and self.target.workflow.evidence_policy_revision != self.run.evidence_policy_revision
        ):
            raise ValueError("Workflow target evidence policy does not match the published run.")
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
        external_trials: tuple[ExternalTrialIdentityV1, ...] = (),
    ) -> CorpusExecutionResult:
        if type(target) is not EvaluationTargetIdentity:
            raise TypeError("target must be an exact EvaluationTargetIdentity.")
        if type(run) is not PublishedEvalRun:
            raise TypeError("run must be an exact PublishedEvalRun.")
        if type(external_trials) is not tuple or any(
            type(trial) is not ExternalTrialIdentityV1 for trial in external_trials
        ):
            raise TypeError("external_trials must be a tuple of ExternalTrialIdentityV1 values.")
        document = {
            "schema_version": CORPUS_EXECUTION_RESULT_SCHEMA_VERSION,
            "target": target.model_dump(mode="json"),
            "run": run.model_dump(mode="json"),
        }
        if external_trials:
            document["external_trials"] = [
                trial.model_dump(mode="json") for trial in external_trials
            ]
        return cls(
            revision=_content_revision(document, "corpus execution result"),
            target=target,
            run=run,
            external_trials=external_trials,
        )


async def run_corpus_suite(
    target: CorpusTarget,
    corpus: EvalCorpusDocument,
    suite_id: str,
    *,
    max_concurrency: int | None = None,
    execution_capacity: EvalExecutionCapacity | None = None,
) -> CorpusExecutionResult:
    """Execute one corpus suite through Cayu's existing runner and publish it safely."""

    validated_target = _copy_corpus_target(target)
    compiled = _compile_prepared_corpus_suite(
        _prepare_corpus_with_validated_target(corpus, validated_target),
        suite_id,
    )
    if max_concurrency is None:
        # Portable suites already bind a policy, including the legacy serial
        # fallback. Never reinterpret that identity using process defaults.
        max_concurrency = compiled.run_contract.trial_policy.max_concurrency
    _validate_corpus_concurrency(validated_target, max_concurrency)
    return await _run_compiled_corpus_suite(
        validated_target,
        compiled,
        max_concurrency=max_concurrency,
        execution_capacity=execution_capacity,
    )


async def _run_compiled_corpus_suite(
    target: CorpusTarget,
    compiled: CompiledCorpusSuite,
    *,
    max_concurrency: int,
    manifest_project_root: Path | None = None,
    expected_app_manifest_fingerprint: str | None = None,
    expected_execution_profile: ExecutionProfileIdentity | None = None,
    native_run_id: str | None = None,
    execution_capacity: EvalExecutionCapacity | None = None,
    accepted_exposure: EvalSuiteRunExposureV1 | None = None,
    completed_trials: Mapping[tuple[str, int], tuple[EvalTrialResult, _EvalTrialPublicData]]
    | None = None,
    trial_completed: Callable[[str, EvalTrialResult, _EvalTrialPublicData], Awaitable[None]]
    | None = None,
) -> CorpusExecutionResult:
    """Execute one internally compiled suite without repeating corpus compilation."""

    validated_target = _copy_corpus_target(target)
    if type(compiled) is not CompiledCorpusSuite:
        raise TypeError("compiled must be an exact CompiledCorpusSuite.")
    if compiled.corpus.target_key != validated_target.key:
        raise ValueError("Compiled corpus target key does not match the trusted target.")
    if compiled.corpus.evidence_policy != validated_target.evidence_policy:
        raise ValueError("Compiled corpus evidence policy does not match the trusted target.")
    if (
        expected_execution_profile is not None
        and type(expected_execution_profile) is not ExecutionProfileIdentity
    ):
        raise TypeError(
            "expected_execution_profile must be an exact ExecutionProfileIdentity or None."
        )
    if accepted_exposure is not None and type(accepted_exposure) is not EvalSuiteRunExposureV1:
        raise TypeError("accepted_exposure must be an exact EvalSuiteRunExposureV1 or None.")
    _validate_corpus_concurrency(validated_target, max_concurrency)
    if max_concurrency > compiled.run_contract.trial_policy.max_concurrency:
        raise ValueError("max_concurrency cannot exceed the immutable suite trial policy.")

    target_before = _evaluation_target_identity_from_validated_target(
        validated_target,
        project_root=manifest_project_root,
    )
    if (
        expected_app_manifest_fingerprint is not None
        and target_before.app_manifest_fingerprint != expected_app_manifest_fingerprint
    ):
        raise EvalExecutionProfileChangedError(
            "CorpusTarget application manifest does not match its registered identity."
        )
    workflow_execution_profile_fingerprint: str | None = None
    if type(validated_target) is WorkflowEvalTarget:
        try:
            workflow_execution_profile_fingerprint = (
                await validated_target.app.inspect_run_execution_profile(
                    copy_run_request(validated_target.request_base)
                )
            )
        except Exception as exc:
            raise EvalExecutionProfileChangedError(
                "Workflow target execution profile could not be established "
                f"({type(exc).__name__})."
            ) from None
        if (
            expected_execution_profile is not None
            and workflow_execution_profile_fingerprint != expected_execution_profile.fingerprint
        ):
            raise EvalExecutionProfileChangedError(
                "Workflow target execution profile does not match its registered identity."
            )
    trial_count = len(compiled.suite.cases) * compiled.trials
    output_preview_bytes = min(
        EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES,
        PUBLISHED_EVAL_OUTPUT_PREVIEW_BUDGET_BYTES // trial_count,
    )

    def run_stream(request: RunRequest):
        if expected_app_manifest_fingerprint is not None:
            current_target = _evaluation_target_identity_from_validated_target(
                validated_target,
                project_root=manifest_project_root,
            )
            if current_target.app_manifest_fingerprint != expected_app_manifest_fingerprint:
                raise EvalExecutionProfileChangedError(
                    "CorpusTarget application manifest changed before fresh eval execution."
                )
        return validated_target.app._run_with_public_projection(
            request,
            expected_execution_profile=expected_execution_profile,
        )

    selected_run_id = str(uuid4()) if native_run_id is None else native_run_id
    case_revisions = {case.id: case.revision for case in compiled.corpus.cases}
    external_case_refs = {
        case.id: None if case.input is None else case.input.opaque_external_case_ref
        for case in compiled.corpus.cases
    }
    external_trials: tuple[ExternalTrialIdentityV1, ...] = ()
    trial_request_transform = None
    if validated_target.external_process is not None:
        external_process = validated_target.external_process
        external_trials = tuple(
            ExternalTrialIdentityV1.create(
                native_run_id=selected_run_id,
                target_key=validated_target.key,
                target_revision=external_process.revision,
                corpus_revision=compiled.corpus.revision,
                suite_id=compiled.suite.id,
                suite_revision=compiled.run_contract.suite_revision,
                case_id=case.id,
                case_revision=case_revisions[case.id],
                trial_number=trial_number,
            )
            for case in compiled.suite.cases
            for trial_number in range(1, compiled.trials + 1)
        )
        external_trial_by_slot = {
            (trial.case_id, trial.trial_number): trial for trial in external_trials
        }

        def external_trial_request(
            suite_id: str,
            case_id: str,
            trial_number: int,
            request: RunRequest,
        ) -> RunRequest:
            if suite_id != compiled.suite.id:
                raise ValueError("External trial suite identity changed before dispatch.")
            trial = external_trial_by_slot[(case_id, trial_number)]
            messages = with_external_trial_envelope(
                request.messages,
                ExternalTrialEnvelopeV1(
                    trial=trial,
                    opaque_case_ref=external_case_refs[case_id],
                ),
            )
            return request.model_copy(update={"messages": messages})

        trial_request_transform = external_trial_request

    internal_run, trial_public_data_by_case = await _run_eval_suite_with_public_projection(
        validated_target.app,
        compiled.suite,
        max_concurrency=max_concurrency,
        case_timeout_seconds=compiled.timeout_seconds,
        trials=compiled.trials,
        trial_policy=compiled.run_contract.trial_policy,
        output_preview_bytes=output_preview_bytes,
        run_stream=(
            run_stream
            if type(validated_target) is not WorkflowEvalTarget
            and (
                expected_app_manifest_fingerprint is not None
                or expected_execution_profile is not None
            )
            else None
        ),
        run_id=selected_run_id,
        trial_request_transform=trial_request_transform,
        execution_capacity=execution_capacity,
        completed_trials=completed_trials,
        trial_completed=trial_completed,
        workflow_target=(
            validated_target if type(validated_target) is WorkflowEvalTarget else None
        ),
        workflow_execution_profile_fingerprint=workflow_execution_profile_fingerprint,
    )
    if workflow_execution_profile_fingerprint is not None:
        try:
            final_workflow_execution_profile_fingerprint = (
                await validated_target.app.inspect_run_execution_profile(
                    copy_run_request(validated_target.request_base)
                )
            )
        except Exception as exc:
            raise EvalExecutionProfileChangedError(
                "Workflow target execution profile could not be revalidated "
                f"({type(exc).__name__})."
            ) from None
        if final_workflow_execution_profile_fingerprint != workflow_execution_profile_fingerprint:
            raise EvalExecutionProfileChangedError(
                "Workflow target execution profile changed during eval execution."
            )
    return await asyncio.to_thread(
        _finalize_compiled_corpus_result,
        validated_target,
        compiled,
        target_before,
        internal_run,
        trial_public_data_by_case,
        manifest_project_root,
        external_trials,
        accepted_exposure,
    )


def _finalize_compiled_corpus_result(
    target: CorpusTarget,
    compiled: CompiledCorpusSuite,
    target_before: EvaluationTargetIdentity,
    internal_run: EvalRun,
    trial_public_data_by_case: dict[str, tuple[_EvalTrialPublicData, ...]],
    manifest_project_root: Path | None,
    external_trials: tuple[ExternalTrialIdentityV1, ...] = (),
    accepted_exposure: EvalSuiteRunExposureV1 | None = None,
) -> CorpusExecutionResult:
    """Construct and validate the complete published result off the event loop."""

    target_after = _evaluation_target_identity_from_validated_target(
        target,
        project_root=manifest_project_root,
    )
    if (
        target_after.target_key != target_before.target_key
        or target_after.application_release_id != target_before.application_release_id
        or target_after.app_manifest_fingerprint != target_before.app_manifest_fingerprint
        or target_after.external_process != target_before.external_process
        or target_after.workflow != target_before.workflow
    ):
        raise EvalExecutionProfileChangedError(
            "CorpusTarget application manifest changed or execution target identity changed "
            "during eval execution."
        )
    run_document: dict[str, Any] = _model_instance_python_input(internal_run)
    run_document["run_contract"] = compiled.run_contract
    bound_run = EvalRun.model_validate(run_document)
    return CorpusExecutionResult.create(
        target=target_before,
        run=_publish_eval_run_with_trial_public_data(
            compiled.corpus,
            bound_run,
            trial_public_data_by_case=trial_public_data_by_case,
            accepted_exposure=accepted_exposure,
        ),
        external_trials=external_trials,
    )


def _validate_corpus_concurrency(target: CorpusTarget, max_concurrency: int) -> None:
    if type(max_concurrency) is not int:
        raise TypeError("max_concurrency must be an int.")
    if not 1 <= max_concurrency <= target.limits.max_concurrency:
        raise ValueError(
            "max_concurrency must be between 1 and the trusted target concurrency limit."
        )
