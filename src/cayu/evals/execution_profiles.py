"""Server-published execution identity and policy for durable Evals."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._validation import (
    canonical_durable_json_bytes,
    require_durable_clean_nonblank,
    require_unicode_scalar_text,
)
from cayu.evals.capacity import EVAL_MAX_CONCURRENCY
from cayu.evals.corpus import (
    EVAL_CORPUS_MAX_CASES,
    EVAL_CORPUS_MAX_TIMEOUT_SECONDS,
    EVAL_CORPUS_MAX_TRIALS,
    EvaluationEvidencePolicySpec,
)
from cayu.evals.execution import (
    CORPUS_EXECUTION_MAX_BOOTSTRAP_MESSAGES,
    CORPUS_EXECUTION_MAX_COMPILED_INPUT_CHARS,
    CORPUS_EXECUTION_MAX_TOTAL_INPUT_CHARS,
    CorpusTarget,
    WorkflowEvalTarget,
)
from cayu.runtime.config import MAX_STEPS
from cayu.runtime.execution_profiles import ExecutionProfileIdentity
from cayu.runtime.sessions import copy_run_request
from cayu.runtime.stop_policy import RunLimits, copy_run_limits

EVAL_EXECUTION_PROFILE_MAX_TEXT_CHARS = 256
_EVAL_EXECUTION_PROFILE_REVISION_DOMAIN = "eval_execution_profile_v1"
_EVAL_EXECUTION_PROFILE_COMPARISON_DOMAIN = "eval_execution_profile_comparison_v1"
_EVAL_EXECUTION_TARGET_MATERIAL_DOMAIN = b"cayu-eval-execution-target-material-v1\0"
_EVAL_EXECUTION_TARGET_PROCESS_SCOPE_DOMAIN = b"cayu-eval-execution-target-process-scope-v1\0"


def _revision(value: object, field_name: str) -> str:
    digest = sha256(canonical_durable_json_bytes(value, field_name)).hexdigest()
    return f"sha256:{digest}"


def _validate_revision(value: str, field_name: str) -> str:
    if (
        len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{field_name} must be a sha256 revision.")
    return value


class EvalExecutionProfilePolicyV1(BaseModel):
    """Trusted application declaration layered around runtime execution identity.

    The policy contains no callback or browser-provided authority. A generated or
    otherwise undeclared profile uses the conservative one-by-one defaults and
    claims only fresh Cayu session state, not reset external application state.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    fixture_strategy: Literal["none", "application_managed"] = "none"
    reset_strategy: Literal["fresh_session_only", "application_managed"] = "fresh_session_only"
    effect_posture: Literal[
        "ordinary_application_authority",
        "isolated_application_authority",
    ] = "ordinary_application_authority"
    isolation_revision: StrictStr | None = Field(default=None, max_length=71)
    max_trials: StrictInt = Field(default=1, ge=1, le=EVAL_CORPUS_MAX_TRIALS)
    max_concurrency: StrictInt = Field(
        default=1,
        ge=1,
        le=EVAL_MAX_CONCURRENCY,
    )

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator("isolation_revision")
    @classmethod
    def validate_isolation_revision(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_revision(value, "isolation_revision")

    @model_validator(mode="after")
    def validate_scale_contract(self) -> EvalExecutionProfilePolicyV1:
        application_managed = (
            self.fixture_strategy == "application_managed"
            or self.reset_strategy == "application_managed"
            or self.effect_posture == "isolated_application_authority"
        )
        if application_managed != (self.isolation_revision is not None):
            raise ValueError(
                "Application-managed fixture, reset, or isolated effects require exactly one "
                "isolation_revision."
            )
        if (self.max_trials > 1 or self.max_concurrency > 1) and (
            self.reset_strategy != "application_managed" or self.isolation_revision is None
        ):
            raise ValueError(
                "Repeated or concurrent eval execution requires an application-managed reset "
                "contract with a stable isolation revision."
            )
        return self

    @classmethod
    def safe_default(cls) -> EvalExecutionProfilePolicyV1:
        return cls()


class EvalExecutionCandidateIdentityV1(BaseModel):
    """Safe current candidate identity selected by runtime preparation."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    agent_name: StrictStr = Field(min_length=1, max_length=EVAL_EXECUTION_PROFILE_MAX_TEXT_CHARS)
    provider_name: StrictStr = Field(
        min_length=1,
        max_length=EVAL_EXECUTION_PROFILE_MAX_TEXT_CHARS,
    )
    model: StrictStr = Field(min_length=1, max_length=EVAL_EXECUTION_PROFILE_MAX_TEXT_CHARS)
    environment_name: StrictStr | None = Field(
        default=None,
        min_length=1,
        max_length=EVAL_EXECUTION_PROFILE_MAX_TEXT_CHARS,
    )
    runtime_execution_profile_schema_version: StrictInt = Field(ge=1, le=64)
    runtime_execution_profile_fingerprint: StrictStr = Field(min_length=64, max_length=64)

    @field_validator("agent_name", "provider_name", "model", "environment_name")
    @classmethod
    def validate_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        value = require_durable_clean_nonblank(value, info.field_name)
        return require_unicode_scalar_text(value, info.field_name)

    @field_validator("runtime_execution_profile_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("runtime_execution_profile_fingerprint must be lowercase SHA-256.")
        return value


class EvalExecutionTargetMaterialIdentityV1(BaseModel):
    """Opaque commitment to candidate inputs owned by one trusted target.

    Runtime execution authority is represented separately by
    ``ExecutionProfileIdentity``. This identity covers the remaining target
    material consumed while compiling each fresh request: the request base,
    bootstrap messages, and complete corpus-execution limit object.

    Public-safe material has a restart-portable structural digest. Material
    crossing the application's workload-secret boundary uses a process-keyed
    HMAC instead, so it remains exact within the serving process without
    exposing an offline digest of private values. A restart then changes the
    process scope and safely rejects previously admitted work.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    kind: Literal["structural_sha256", "process_local_hmac_sha256"]
    fingerprint: StrictStr = Field(min_length=64, max_length=64)
    process_scope: StrictStr | None = Field(default=None, min_length=64, max_length=64)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator("fingerprint", "process_scope")
    @classmethod
    def validate_fingerprints(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{info.field_name} must be lowercase SHA-256 hex.")
        return value

    @model_validator(mode="after")
    def validate_process_scope(self) -> EvalExecutionTargetMaterialIdentityV1:
        process_local = self.kind == "process_local_hmac_sha256"
        if process_local != (self.process_scope is not None):
            raise ValueError("Only process-local target material requires a process scope.")
        return self


class EvalExecutionResourceCeilingsV1(BaseModel):
    """Complete server-owned limits that HTTP may only narrow."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    max_cases: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_CASES)
    max_trials: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_TRIALS)
    max_concurrency: StrictInt = Field(ge=1, le=EVAL_MAX_CONCURRENCY)
    max_timeout_seconds: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_TIMEOUT_SECONDS)
    max_bootstrap_messages: StrictInt = Field(
        ge=0,
        le=CORPUS_EXECUTION_MAX_BOOTSTRAP_MESSAGES,
    )
    max_total_input_chars: StrictInt = Field(
        ge=1,
        le=CORPUS_EXECUTION_MAX_TOTAL_INPUT_CHARS,
    )
    max_compiled_input_chars: StrictInt = Field(
        ge=1,
        le=CORPUS_EXECUTION_MAX_COMPILED_INPUT_CHARS,
    )
    max_steps: StrictInt = Field(ge=1, le=MAX_STEPS)
    run_limits: RunLimits

    @field_validator("run_limits", mode="before")
    @classmethod
    def copy_limits(cls, value: object) -> object:
        if type(value) is RunLimits:
            return copy_run_limits(value)
        return value


class EvalExecutionProfileV1(BaseModel):
    """Bounded public snapshot of one currently executable server profile."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    revision: StrictStr = Field(min_length=71, max_length=71)
    profile_id: StrictStr = Field(min_length=1, max_length=128)
    label: StrictStr = Field(min_length=1, max_length=EVAL_EXECUTION_PROFILE_MAX_TEXT_CHARS * 2)
    source: Literal["generated", "explicit"]
    target_key: StrictStr = Field(min_length=1, max_length=128)
    application_release_id: StrictStr = Field(
        min_length=1,
        max_length=EVAL_EXECUTION_PROFILE_MAX_TEXT_CHARS,
    )
    app_manifest_fingerprint: StrictStr = Field(min_length=64, max_length=64)
    candidate: EvalExecutionCandidateIdentityV1
    target_material: EvalExecutionTargetMaterialIdentityV1
    fixture_strategy: Literal["none", "application_managed"]
    reset_strategy: Literal["fresh_session_only", "application_managed"]
    effect_posture: Literal[
        "ordinary_application_authority",
        "isolated_application_authority",
    ]
    isolation_revision: StrictStr | None = Field(default=None, max_length=71)
    evidence_policy: EvaluationEvidencePolicySpec
    ceilings: EvalExecutionResourceCeilingsV1

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator("revision", "isolation_revision")
    @classmethod
    def validate_revisions(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _validate_revision(value, info.field_name)

    @field_validator("app_manifest_fingerprint")
    @classmethod
    def validate_manifest_fingerprint(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("app_manifest_fingerprint must be lowercase SHA-256.")
        return value

    @field_validator("evidence_policy", mode="before")
    @classmethod
    def copy_evidence_policy(cls, value: object) -> object:
        if type(value) is EvaluationEvidencePolicySpec:
            return value.model_dump(mode="json")
        return value

    @model_validator(mode="after")
    def validate_revision(self) -> EvalExecutionProfileV1:
        material = self.model_dump(mode="json", exclude={"revision"})
        expected = _revision(material, _EVAL_EXECUTION_PROFILE_REVISION_DOMAIN)
        if self.revision != expected:
            raise ValueError("Eval execution profile revision does not match its content.")
        return self

    @property
    def comparison_revision(self) -> str:
        """Return the execution-semantic identity used between application releases.

        Admission must use ``revision``, which binds every published byte. Result
        comparison deliberately excludes release and presentation metadata while
        retaining the complete candidate, runtime, target-material, isolation,
        evidence, and resource contract.
        """

        validated = EvalExecutionProfileV1.model_validate(
            self.model_dump(mode="python", round_trip=True, warnings="none")
        )
        material = validated.model_dump(
            mode="json",
            exclude={
                "revision",
                "profile_id",
                "label",
                "source",
                "application_release_id",
                "app_manifest_fingerprint",
            },
        )
        return _revision(material, _EVAL_EXECUTION_PROFILE_COMPARISON_DOMAIN)

    @classmethod
    def create(
        cls,
        *,
        profile_id: str,
        label: str,
        source: Literal["generated", "explicit"],
        target_key: str,
        application_release_id: str,
        app_manifest_fingerprint: str,
        candidate: EvalExecutionCandidateIdentityV1,
        target_material: EvalExecutionTargetMaterialIdentityV1,
        fixture_strategy: Literal["none", "application_managed"],
        reset_strategy: Literal["fresh_session_only", "application_managed"],
        effect_posture: Literal[
            "ordinary_application_authority",
            "isolated_application_authority",
        ],
        isolation_revision: str | None,
        evidence_policy: EvaluationEvidencePolicySpec,
        ceilings: EvalExecutionResourceCeilingsV1,
    ) -> EvalExecutionProfileV1:
        draft = cls.model_construct(
            schema_version=1,
            revision="sha256:" + "0" * 64,
            profile_id=profile_id,
            label=label,
            source=source,
            target_key=target_key,
            application_release_id=application_release_id,
            app_manifest_fingerprint=app_manifest_fingerprint,
            candidate=candidate,
            target_material=target_material,
            fixture_strategy=fixture_strategy,
            reset_strategy=reset_strategy,
            effect_posture=effect_posture,
            isolation_revision=isolation_revision,
            evidence_policy=evidence_policy,
            ceilings=ceilings,
        )
        material = draft.model_dump(mode="json", exclude={"revision"})
        revision = _revision(material, _EVAL_EXECUTION_PROFILE_REVISION_DOMAIN)
        return cls.model_validate({"revision": revision, **material})


class EvalExecutionProfileBindingV1(BaseModel):
    """Durable server-prepared runtime identity for one admitted eval run."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    profile_revision: StrictStr = Field(min_length=71, max_length=71)
    runtime_execution_profile: ExecutionProfileIdentity

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator("profile_revision")
    @classmethod
    def validate_profile_revision(cls, value: str) -> str:
        return _validate_revision(value, "profile_revision")

    @field_validator("runtime_execution_profile", mode="before")
    @classmethod
    def copy_runtime_profile(cls, value: object) -> object:
        if type(value) is ExecutionProfileIdentity:
            return value.model_dump(mode="json")
        return value


@dataclass(frozen=True, slots=True)
class PreparedEvalExecutionProfile:
    """Current public snapshot paired with exact process-local execution authority."""

    target: CorpusTarget
    snapshot: EvalExecutionProfileV1
    binding: EvalExecutionProfileBindingV1


def _eval_execution_target_material_identity(
    target: CorpusTarget,
) -> EvalExecutionTargetMaterialIdentityV1:
    """Commit every non-runtime target input without publishing private values."""

    material = {
        "schema_version": 1,
        "request_base": target.request_base.model_dump(
            mode="json",
            round_trip=True,
            warnings="none",
        ),
        "bootstrap_messages": [
            message.model_dump(mode="json", round_trip=True, warnings="none")
            for message in target.bootstrap_messages
        ],
        # Keep the entire object in the commitment as a forward-safety fence.
        # Individual current ceilings are also published for operator decisions.
        "execution_limits": target.limits.model_dump(mode="json"),
        "external_process": (
            None
            if target.external_process is None
            else target.external_process.model_dump(mode="json")
        ),
        "workflow": (
            target.identity().model_dump(mode="json")
            if type(target) is WorkflowEvalTarget
            else None
        ),
    }
    canonical = canonical_durable_json_bytes(material, "eval execution target material")
    try:
        public_material = target.app.redact_json(material)
    except Exception as exc:
        raise ValueError(
            "Eval execution target material could not cross the application redaction boundary."
        ) from exc
    if public_material == material:
        return EvalExecutionTargetMaterialIdentityV1(
            kind="structural_sha256",
            fingerprint=sha256(_EVAL_EXECUTION_TARGET_MATERIAL_DOMAIN + canonical).hexdigest(),
        )

    process_identity = require_durable_clean_nonblank(
        target.app._execution_profile_process_identity,
        "execution_profile_process_identity",
    )
    process_key = process_identity.encode("utf-8")
    return EvalExecutionTargetMaterialIdentityV1(
        kind="process_local_hmac_sha256",
        fingerprint=hmac.new(
            process_key,
            _EVAL_EXECUTION_TARGET_MATERIAL_DOMAIN + canonical,
            sha256,
        ).hexdigest(),
        process_scope=sha256(_EVAL_EXECUTION_TARGET_PROCESS_SCOPE_DOMAIN + process_key).hexdigest(),
    )


async def prepare_eval_execution_profile(
    target: CorpusTarget,
    *,
    profile_id: str,
    label: str,
    source: Literal["generated", "explicit"],
    app_manifest_fingerprint: str,
    policy: EvalExecutionProfilePolicyV1,
) -> PreparedEvalExecutionProfile:
    """Resolve current runtime identity without admitting a session or doing governed work."""

    if type(target) not in {CorpusTarget, WorkflowEvalTarget}:
        raise TypeError("target must be an exact CorpusTarget or WorkflowEvalTarget.")
    if type(policy) is not EvalExecutionProfilePolicyV1:
        raise TypeError("policy must be an exact EvalExecutionProfilePolicyV1.")
    if policy.max_trials > target.limits.max_trials:
        raise ValueError("Eval execution profile trial ceiling exceeds its target authority.")
    if policy.max_concurrency > target.limits.max_concurrency:
        raise ValueError("Eval execution profile concurrency ceiling exceeds its target authority.")

    request = target.app._with_application_run_defaults(
        copy_run_request(target.request_base),
    )
    prepared = await target.app._session_engine._prepare_initial_run(
        request,
        admit_session=False,
    )
    if prepared is None:
        raise RuntimeError("Eval execution profile conflicts with contracted task authority.")
    runtime_profile = ExecutionProfileIdentity.model_validate(
        prepared.execution_profile.model_dump(mode="json")
    )
    snapshot = EvalExecutionProfileV1.create(
        profile_id=profile_id,
        label=label,
        source=source,
        target_key=target.key,
        application_release_id=target.application_release_id,
        app_manifest_fingerprint=app_manifest_fingerprint,
        candidate=EvalExecutionCandidateIdentityV1(
            agent_name=prepared.registered_agent.spec.name,
            provider_name=prepared.registered_provider.name,
            model=prepared.session_identity.model,
            environment_name=prepared.request.environment_name,
            runtime_execution_profile_schema_version=runtime_profile.schema_version,
            runtime_execution_profile_fingerprint=runtime_profile.fingerprint,
        ),
        target_material=_eval_execution_target_material_identity(target),
        fixture_strategy=policy.fixture_strategy,
        reset_strategy=policy.reset_strategy,
        effect_posture=policy.effect_posture,
        isolation_revision=policy.isolation_revision,
        evidence_policy=target.evidence_policy,
        ceilings=EvalExecutionResourceCeilingsV1(
            max_cases=target.limits.max_cases,
            max_trials=policy.max_trials,
            max_concurrency=policy.max_concurrency,
            max_timeout_seconds=target.limits.max_timeout_seconds,
            max_bootstrap_messages=target.limits.max_bootstrap_messages,
            max_total_input_chars=target.limits.max_total_input_chars,
            max_compiled_input_chars=target.limits.max_compiled_input_chars,
            max_steps=prepared.request.max_steps,
            run_limits=prepared.request.limits,
        ),
    )
    public = snapshot.model_dump(mode="json")
    if target.app.redact_json(public) != public:
        raise ValueError("Eval execution profile contains workload-secret material.")
    binding = EvalExecutionProfileBindingV1(
        profile_revision=snapshot.revision,
        runtime_execution_profile=runtime_profile,
    )
    return PreparedEvalExecutionProfile(
        target=target,
        snapshot=snapshot,
        binding=binding,
    )


__all__ = [
    "EvalExecutionCandidateIdentityV1",
    "EvalExecutionProfileBindingV1",
    "EvalExecutionProfilePolicyV1",
    "EvalExecutionProfileV1",
    "EvalExecutionResourceCeilingsV1",
    "EvalExecutionTargetMaterialIdentityV1",
    "PreparedEvalExecutionProfile",
    "prepare_eval_execution_profile",
]
