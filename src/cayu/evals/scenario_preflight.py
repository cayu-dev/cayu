from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from cayu._validation import canonical_durable_json_bytes, require_durable_clean_nonblank
from cayu.artifacts import (
    ArtifactScope,
    ArtifactStore,
    ArtifactStoreUnavailableError,
    InvalidArtifactIdError,
    copy_artifact_read_result,
)
from cayu.evals.corpus import (
    EVAL_CORPUS_MAX_TIMEOUT_SECONDS,
    EVAL_CORPUS_MAX_TRIALS,
    _model_python_input,
    _portable_id,
    _sha256_hex,
    _sha256_revision,
)
from cayu.evals.execution import (
    CORPUS_EXECUTION_MAX_CONCURRENCY,
    CorpusTarget,
    evaluation_target_identity,
)
from cayu.evals.scenario import (
    EVAL_SCENARIO_MAX_ARTIFACT_REQUIREMENTS,
    EVAL_SCENARIO_MAX_SECRET_REQUIREMENTS,
    EvalScenarioDocumentV2,
    ScenarioApprovalCheckpointEventV2,
    ScenarioArtifactRequirementV2,
    ScenarioFilePartV2,
    ScenarioInitialInputEventV2,
    ScenarioQueuedInputEventV2,
    ScenarioResumedInputEventV2,
    ScenarioSecretRequirementV2,
)
from cayu.evals.scenario_authoring import replace_eval_scenario_artifact_requirement
from cayu.evals.store import EvalRunCostBudget
from cayu.runtime.budgets import budget_pricing_preflight_error
from cayu.runtime.stop_policy import RunLimits, copy_run_limits
from cayu.vaults import Vault, VaultError, copy_secret_ref

SCENARIO_PREFLIGHT_MAX_DIAGNOSTICS = 1_024
SCENARIO_PREFLIGHT_ARTIFACT_READ_CONCURRENCY = 8


class _ScenarioPreflightModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class ScenarioLaunchDiagnosticCode(StrEnum):
    TARGET_MISMATCH = "target_mismatch"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    ENVIRONMENT_UNAVAILABLE = "environment_unavailable"
    EXECUTION_BINDING_REQUIRED = "execution_binding_required"
    ARTIFACT_BINDING_REQUIRED = "artifact_binding_required"
    ARTIFACT_NOT_RETAINED = "artifact_not_retained"
    ARTIFACT_ACCESS_DENIED = "artifact_access_denied"
    ARTIFACT_STORE_UNAVAILABLE = "artifact_store_unavailable"
    ARTIFACT_CONTENT_INCONSISTENT = "artifact_content_inconsistent"
    SECRET_REFERENCE_UNAVAILABLE = "secret_reference_unavailable"
    APPROVAL_TOOL_UNAVAILABLE = "approval_tool_unavailable"
    APPROVAL_POLICY_SELECTION_REQUIRED = "approval_policy_selection_required"
    EXECUTION_LIMIT_EXCEEDED = "execution_limit_exceeded"
    PRICING_UNAVAILABLE = "pricing_unavailable"
    ACTOR_AUTHORITY_UNAVAILABLE = "actor_authority_unavailable"
    SCENARIO_CONTENT_UNSAFE = "scenario_content_unsafe"


_DIAGNOSTIC_COPY: dict[ScenarioLaunchDiagnosticCode, tuple[str, str]] = {
    ScenarioLaunchDiagnosticCode.TARGET_MISMATCH: (
        "The scenario is bound to a different published target.",
        "Select the scenario's target or create a reviewed revision for the intended target.",
    ),
    ScenarioLaunchDiagnosticCode.PROVIDER_UNAVAILABLE: (
        "The current target cannot resolve its model provider.",
        "Restore the target's provider registration or select another published target.",
    ),
    ScenarioLaunchDiagnosticCode.ENVIRONMENT_UNAVAILABLE: (
        "The selected execution environment is not currently available.",
        "Select a published environment or restore the target's declared environment.",
    ),
    ScenarioLaunchDiagnosticCode.EXECUTION_BINDING_REQUIRED: (
        "A current execution binding cannot be proven before admission.",
        "Select a concrete server-published binding that matches the target.",
    ),
    ScenarioLaunchDiagnosticCode.ARTIFACT_BINDING_REQUIRED: (
        "A scenario file is not bound to a reusable environment fixture.",
        "Prepare an environment fixture or select a retained artifact with the required digest.",
    ),
    ScenarioLaunchDiagnosticCode.ARTIFACT_NOT_RETAINED: (
        "A bound scenario artifact is no longer retained.",
        "Restore the artifact or bind a retained fixture with the required digest.",
    ),
    ScenarioLaunchDiagnosticCode.ARTIFACT_ACCESS_DENIED: (
        "The current target cannot read a bound scenario artifact.",
        "Use an authorized environment fixture or restore artifact access.",
    ),
    ScenarioLaunchDiagnosticCode.ARTIFACT_STORE_UNAVAILABLE: (
        "The current artifact store cannot complete scenario preflight.",
        "Restore the artifact store and retry preflight.",
    ),
    ScenarioLaunchDiagnosticCode.ARTIFACT_CONTENT_INCONSISTENT: (
        "A bound artifact does not match the scenario's immutable file requirement.",
        "Bind the exact retained bytes or prepare a new fixture from a trusted source.",
    ),
    ScenarioLaunchDiagnosticCode.SECRET_REFERENCE_UNAVAILABLE: (
        "A named scenario secret has no current server-published binding.",
        "Select an approved target profile that publishes this secret requirement.",
    ),
    ScenarioLaunchDiagnosticCode.APPROVAL_TOOL_UNAVAILABLE: (
        "An approval checkpoint names a tool unavailable to the current target.",
        "Select a target exposing the tool or remove the obsolete checkpoint.",
    ),
    ScenarioLaunchDiagnosticCode.APPROVAL_POLICY_SELECTION_REQUIRED: (
        "The current target does not prove a fresh approval pause for this checkpoint.",
        "Select a server-published policy that requires a fresh decision for the tool.",
    ),
    ScenarioLaunchDiagnosticCode.EXECUTION_LIMIT_EXCEEDED: (
        "The scenario or requested run bounds exceed the current target's limits.",
        "Reduce the scenario, trials, concurrency, timeout, or per-run limits.",
    ),
    ScenarioLaunchDiagnosticCode.PRICING_UNAVAILABLE: (
        "The requested cost bound cannot be priced for the current target.",
        "Remove the cost bound or restore compatible server-owned pricing.",
    ),
    ScenarioLaunchDiagnosticCode.ACTOR_AUTHORITY_UNAVAILABLE: (
        "Current operator authority cannot admit scenario execution.",
        "Authenticate with normal session-execution and Evals mutation authority.",
    ),
    ScenarioLaunchDiagnosticCode.SCENARIO_CONTENT_UNSAFE: (
        "The scenario cannot cross the application's publication boundary unchanged.",
        "Remove or replace redacted workload data before saving or launching.",
    ),
}


class ScenarioLaunchDiagnosticV2(_ScenarioPreflightModel):
    code: ScenarioLaunchDiagnosticCode
    message: StrictStr = Field(min_length=1, max_length=512)
    remediation: StrictStr = Field(min_length=1, max_length=512)
    event_id: StrictStr | None = Field(default=None, max_length=128)
    requirement_id: StrictStr | None = Field(default=None, max_length=128)

    @field_validator("event_id", "requirement_id")
    @classmethod
    def validate_optional_id(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _portable_id(value, info.field_name)

    @model_validator(mode="after")
    def validate_copy(self) -> ScenarioLaunchDiagnosticV2:
        if (self.message, self.remediation) != _DIAGNOSTIC_COPY[self.code]:
            raise ValueError("Scenario launch diagnostic copy does not match its code.")
        return self


class ScenarioLaunchSettingsV2(_ScenarioPreflightModel):
    """Authority-free operator selections narrowed by current server policy."""

    environment_name: StrictStr | None = Field(default=None, max_length=256)
    approval_behavior: Literal["fresh_decision"] = "fresh_decision"
    trials: StrictInt = Field(default=1, ge=1, le=EVAL_CORPUS_MAX_TRIALS)
    max_concurrency: StrictInt = Field(
        default=1,
        ge=1,
        le=CORPUS_EXECUTION_MAX_CONCURRENCY,
    )
    timeout_seconds: StrictInt = Field(
        default=300,
        ge=1,
        le=EVAL_CORPUS_MAX_TIMEOUT_SECONDS,
    )
    max_steps: StrictInt | None = Field(default=None, ge=1, le=256)
    limits: RunLimits | None = None
    cost_budget: EvalRunCostBudget | None = None
    artifact_references: dict[StrictStr, StrictStr] = Field(
        default_factory=dict,
        max_length=EVAL_SCENARIO_MAX_ARTIFACT_REQUIREMENTS,
    )

    @field_validator("environment_name")
    @classmethod
    def validate_environment_name(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("limits", mode="before")
    @classmethod
    def require_run_scoped_limits(cls, value: object) -> object:
        if isinstance(value, RunLimits) and value.scope != "run":
            raise ValueError("Scenario execution limits must use run scope.")
        if isinstance(value, Mapping):
            scope = cast("Mapping[str, object]", value).get("scope", "run")
            if scope != "run":
                raise ValueError("Scenario execution limits must use run scope.")
        return value

    @field_validator("artifact_references")
    @classmethod
    def validate_artifact_references(cls, value: dict[str, str]) -> dict[str, str]:
        validated: dict[str, str] = {}
        for requirement_id, artifact_id in value.items():
            requirement_id = _portable_id(
                requirement_id,
                "artifact_references requirement id",
            )
            artifact_id = require_durable_clean_nonblank(
                artifact_id,
                "artifact_references artifact id",
            )
            if len(artifact_id) > 512:
                raise ValueError("artifact_references artifact ids cannot exceed 512 characters.")
            validated[requirement_id] = artifact_id
        return validated


class ScenarioArtifactLaunchBindingV2(_ScenarioPreflightModel):
    requirement_id: StrictStr = Field(max_length=128)
    artifact_id: StrictStr = Field(max_length=512)
    content_sha256: StrictStr

    @field_validator("requirement_id")
    @classmethod
    def validate_requirement_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("content_sha256")
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)


class ScenarioSecretLaunchBindingV2(_ScenarioPreflightModel):
    """Public proof that a named requirement has a current vault binding."""

    requirement_id: StrictStr = Field(max_length=128)
    usage: Literal["provider", "tool", "environment", "artifact", "other"]

    @field_validator("requirement_id")
    @classmethod
    def validate_requirement_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)


class ScenarioLaunchBindingV2(_ScenarioPreflightModel):
    """Public-safe facts frozen by one successful current-authority preflight."""

    revision: StrictStr
    scenario_revision: StrictStr
    target_key: StrictStr
    application_release_id: StrictStr = Field(max_length=256)
    app_manifest_fingerprint: StrictStr
    agent_name: StrictStr = Field(max_length=256)
    environment_name: StrictStr | None = Field(default=None, max_length=256)
    approval_behavior: Literal["fresh_decision"] = "fresh_decision"
    trials: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_TRIALS)
    max_concurrency: StrictInt = Field(ge=1, le=CORPUS_EXECUTION_MAX_CONCURRENCY)
    timeout_seconds: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_TIMEOUT_SECONDS)
    max_steps: StrictInt = Field(ge=1, le=256)
    target_limits: RunLimits
    operator_run_limits: RunLimits | None = None
    cost_budget: EvalRunCostBudget | None = None
    artifacts: tuple[ScenarioArtifactLaunchBindingV2, ...] = Field(
        default_factory=tuple,
        max_length=EVAL_SCENARIO_MAX_ARTIFACT_REQUIREMENTS,
    )
    secrets: tuple[ScenarioSecretLaunchBindingV2, ...] = Field(
        default_factory=tuple,
        max_length=EVAL_SCENARIO_MAX_SECRET_REQUIREMENTS,
    )

    @field_validator("revision", "scenario_revision")
    @classmethod
    def validate_revisions(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("target_key")
    @classmethod
    def validate_target_key(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("application_release_id", "agent_name", "environment_name")
    @classmethod
    def validate_identity_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("app_manifest_fingerprint")
    @classmethod
    def validate_manifest_fingerprint(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("target_limits", "operator_run_limits", mode="before")
    @classmethod
    def copy_limits(cls, value: object) -> object:
        if value is None:
            return None
        if type(value) is RunLimits:
            return copy_run_limits(value)
        if isinstance(value, BaseModel):
            raise TypeError("Scenario launch limits must be exact RunLimits instances or JSON.")
        return value

    @model_validator(mode="after")
    def validate_revision(self, info: ValidationInfo) -> ScenarioLaunchBindingV2:
        artifact_ids = tuple(item.requirement_id for item in self.artifacts)
        secret_ids = tuple(item.requirement_id for item in self.secrets)
        if artifact_ids != tuple(sorted(set(artifact_ids))):
            raise ValueError("Scenario artifact launch bindings must be unique and sorted.")
        if secret_ids != tuple(sorted(set(secret_ids))):
            raise ValueError("Scenario secret launch bindings must be unique and sorted.")
        if self.operator_run_limits is not None:
            if self.operator_run_limits.scope != "run":
                raise ValueError("Operator scenario limits must use run scope.")
            if _requested_limits_broaden(self.target_limits, self.operator_run_limits):
                raise ValueError("Operator scenario limits cannot broaden target limits.")
        if info.context is not None and info.context.get("skip_revision") is True:
            return self
        material = self.model_dump(mode="json", exclude={"revision"})
        expected = (
            "sha256:"
            + hashlib.sha256(
                canonical_durable_json_bytes(material, "scenario launch binding")
            ).hexdigest()
        )
        if self.revision != expected:
            raise ValueError("Scenario launch binding revision does not match its content.")
        return self

    @classmethod
    def create(cls, **values: Any) -> ScenarioLaunchBindingV2:
        material = cls.model_validate(
            {"revision": "sha256:" + "0" * 64, **values},
            context={"skip_revision": True},
        ).model_dump(mode="json", exclude={"revision"})
        revision = (
            "sha256:"
            + hashlib.sha256(
                canonical_durable_json_bytes(material, "scenario launch binding")
            ).hexdigest()
        )
        return cls(revision=revision, **values)


class ScenarioLaunchPreflightResultV2(_ScenarioPreflightModel):
    ready: StrictBool
    scenario_revision: StrictStr
    binding: ScenarioLaunchBindingV2 | None = None
    diagnostics: tuple[ScenarioLaunchDiagnosticV2, ...] = Field(
        default_factory=tuple,
        max_length=SCENARIO_PREFLIGHT_MAX_DIAGNOSTICS,
    )

    @field_validator("scenario_revision")
    @classmethod
    def validate_scenario_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @model_validator(mode="after")
    def validate_readiness(self) -> ScenarioLaunchPreflightResultV2:
        if self.ready != (self.binding is not None and not self.diagnostics):
            raise ValueError("Scenario launch readiness contradicts its result material.")
        if not self.ready and not self.diagnostics:
            raise ValueError("Unavailable scenario launch requires at least one diagnostic.")
        return self


class ScenarioArtifactMaterializationV2(_ScenarioPreflightModel):
    scenario: EvalScenarioDocumentV2
    requirement_id: StrictStr = Field(max_length=128)
    artifact_id: StrictStr = Field(max_length=512)

    @field_validator("requirement_id")
    @classmethod
    def validate_requirement_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)


class ScenarioArtifactMaterializationError(RuntimeError):
    def __init__(self, code: ScenarioLaunchDiagnosticCode) -> None:
        self.code = ScenarioLaunchDiagnosticCode(code)
        super().__init__(_DIAGNOSTIC_COPY[self.code][0])


def _diagnostic(
    code: ScenarioLaunchDiagnosticCode,
    *,
    event_id: str | None = None,
    requirement_id: str | None = None,
) -> ScenarioLaunchDiagnosticV2:
    message, remediation = _DIAGNOSTIC_COPY[code]
    return ScenarioLaunchDiagnosticV2(
        code=code,
        message=message,
        remediation=remediation,
        event_id=event_id,
        requirement_id=requirement_id,
    )


def _requested_limits_broaden(current: RunLimits, requested: RunLimits | None) -> bool:
    if requested is None:
        return False
    for field_name in (
        "max_input_tokens",
        "max_output_tokens",
        "max_total_tokens",
        "max_tool_calls",
        "max_elapsed_seconds",
    ):
        current_value = getattr(current, field_name)
        requested_value = getattr(requested, field_name)
        if (
            current_value is not None
            and requested_value is not None
            and requested_value > current_value
        ):
            return True
    return False


def _environment_material(
    target: CorpusTarget,
    settings: ScenarioLaunchSettingsV2,
    *,
    project_root: str | Path | None = None,
):
    identity = evaluation_target_identity(target, project_root=project_root)
    manifest = identity.app_manifest
    pinned = target.request_base.environment_name
    requested = settings.environment_name
    diagnostics: list[ScenarioLaunchDiagnosticV2] = []
    if pinned is not None and requested is not None and requested != pinned:
        diagnostics.append(_diagnostic(ScenarioLaunchDiagnosticCode.EXECUTION_BINDING_REQUIRED))
    environment_name = requested or pinned or manifest.defaults.environment
    environment_manifest = next(
        (item for item in manifest.environments if item.name == environment_name),
        None,
    )
    registration = next(
        (
            item
            for item in target.app.list_environment_registrations()
            if item.spec.name == environment_name
        ),
        None,
    )
    if environment_name is not None and (environment_manifest is None or registration is None):
        diagnostics.append(_diagnostic(ScenarioLaunchDiagnosticCode.ENVIRONMENT_UNAVAILABLE))
    return identity, environment_name, environment_manifest, registration, diagnostics


async def preflight_eval_scenario(
    scenario: EvalScenarioDocumentV2,
    target: CorpusTarget,
    settings: ScenarioLaunchSettingsV2 | None = None,
    *,
    actor_authorized: bool,
    project_root: str | Path | None = None,
) -> ScenarioLaunchPreflightResultV2:
    """Resolve current launch facts without invoking providers or application work."""

    if type(scenario) is not EvalScenarioDocumentV2:
        raise TypeError("scenario must be an exact EvalScenarioDocumentV2.")
    if type(target) is not CorpusTarget:
        raise TypeError("target must be an exact CorpusTarget.")
    if type(actor_authorized) is not bool:
        raise TypeError("actor_authorized must be a bool.")
    validated = EvalScenarioDocumentV2.model_validate(_model_python_input(scenario))
    selected = (
        ScenarioLaunchSettingsV2()
        if settings is None
        else ScenarioLaunchSettingsV2.model_validate(_model_python_input(settings))
    )
    diagnostics: list[ScenarioLaunchDiagnosticV2] = []
    if validated.target_key != target.key:
        diagnostics.append(_diagnostic(ScenarioLaunchDiagnosticCode.TARGET_MISMATCH))
    if not actor_authorized:
        diagnostics.append(_diagnostic(ScenarioLaunchDiagnosticCode.ACTOR_AUTHORITY_UNAVAILABLE))

    try:
        public = validated.model_dump(mode="json")
        if target.app.redact_json(public) != public:
            diagnostics.append(_diagnostic(ScenarioLaunchDiagnosticCode.SCENARIO_CONTENT_UNSAFE))
    except Exception:
        diagnostics.append(_diagnostic(ScenarioLaunchDiagnosticCode.SCENARIO_CONTENT_UNSAFE))

    identity, environment_name, environment_manifest, registration, env_diagnostics = (
        _environment_material(target, selected, project_root=project_root)
    )
    diagnostics.extend(env_diagnostics)
    manifest = identity.app_manifest
    agent = next(
        (item for item in manifest.agents if item.name == target.request_base.agent_name),
        None,
    )
    if agent is None or agent.resolved_provider is None:
        diagnostics.append(_diagnostic(ScenarioLaunchDiagnosticCode.PROVIDER_UNAVAILABLE))

    if (
        selected.trials > target.limits.max_trials
        or selected.max_concurrency > target.limits.max_concurrency
        or selected.timeout_seconds > target.limits.max_timeout_seconds
        or (selected.max_steps is not None and selected.max_steps > target.request_base.max_steps)
        or _requested_limits_broaden(target.request_base.limits, selected.limits)
    ):
        diagnostics.append(_diagnostic(ScenarioLaunchDiagnosticCode.EXECUTION_LIMIT_EXCEEDED))

    if selected.cost_budget is not None:
        pricing_error: str | None = None
        if target.price_book is None:
            pricing_error = "pricing unavailable"
        else:
            try:
                model_target = target.app.resolve_run_model_target(target.request_base)
                provider = target.app.get_provider(model_target.provider_name)
                pricing_error = budget_pricing_preflight_error(
                    target.price_book,
                    provider_name=(provider.billing_provider_name or model_target.provider_name),
                    model=model_target.model,
                    currency=selected.cost_budget.currency,
                )
            except (KeyError, RuntimeError, TypeError, ValueError):
                pricing_error = "pricing unavailable"
        if pricing_error is not None:
            diagnostics.append(_diagnostic(ScenarioLaunchDiagnosticCode.PRICING_UNAVAILABLE))

    tools = {} if agent is None else {item.name: item for item in agent.tools}
    for event in validated.events:
        if not isinstance(event, ScenarioApprovalCheckpointEventV2):
            continue
        tool = tools.get(event.tool_name)
        if tool is None:
            diagnostics.append(
                _diagnostic(
                    ScenarioLaunchDiagnosticCode.APPROVAL_TOOL_UNAVAILABLE,
                    event_id=event.id,
                )
            )
        elif tool.policy_coverage != "approval_required":
            diagnostics.append(
                _diagnostic(
                    ScenarioLaunchDiagnosticCode.APPROVAL_POLICY_SELECTION_REQUIRED,
                    event_id=event.id,
                )
            )

    secret_bindings: list[ScenarioSecretLaunchBindingV2] = []
    secret_requirements = validated.secret_requirements
    vault = None if registration is None else registration.environment.vault
    if secret_requirements and vault is None:
        diagnostics.extend(
            _diagnostic(
                ScenarioLaunchDiagnosticCode.SECRET_REFERENCE_UNAVAILABLE,
                requirement_id=requirement.id,
            )
            for requirement in secret_requirements
        )
    elif secret_requirements:
        selected_vault = cast("Vault", vault)
        semaphore = asyncio.Semaphore(SCENARIO_PREFLIGHT_ARTIFACT_READ_CONCURRENCY)

        async def inspect_secret(requirement: ScenarioSecretRequirementV2):
            try:
                async with semaphore:
                    reference = copy_secret_ref(
                        await selected_vault.get(
                            requirement.id,
                            scope={
                                "target_key": target.key,
                                "environment_name": environment_name,
                                "usage": requirement.usage,
                            },
                        )
                    )
            except (VaultError, OSError, TypeError, ValueError):
                return None, _diagnostic(
                    ScenarioLaunchDiagnosticCode.SECRET_REFERENCE_UNAVAILABLE,
                    requirement_id=requirement.id,
                )
            except Exception:
                return None, _diagnostic(
                    ScenarioLaunchDiagnosticCode.SECRET_REFERENCE_UNAVAILABLE,
                    requirement_id=requirement.id,
                )
            if reference.name != requirement.id:
                return None, _diagnostic(
                    ScenarioLaunchDiagnosticCode.SECRET_REFERENCE_UNAVAILABLE,
                    requirement_id=requirement.id,
                )
            return (
                ScenarioSecretLaunchBindingV2(
                    requirement_id=requirement.id,
                    usage=requirement.usage,
                ),
                None,
            )

        inspected_secrets = await asyncio.gather(
            *(inspect_secret(requirement) for requirement in secret_requirements)
        )
        for binding, diagnostic in inspected_secrets:
            if binding is not None:
                secret_bindings.append(binding)
            if diagnostic is not None:
                diagnostics.append(diagnostic)

    artifact_bindings: list[ScenarioArtifactLaunchBindingV2] = []
    requirements = validated.artifact_requirements
    requirement_ids = {item.id for item in requirements}
    if any(key not in requirement_ids for key in selected.artifact_references):
        raise ValueError("artifact_references contains an unknown scenario requirement.")
    runtime = manifest.runtime
    if requirements and (environment_name is None or registration is None):
        diagnostics.extend(
            _diagnostic(
                ScenarioLaunchDiagnosticCode.ARTIFACT_BINDING_REQUIRED,
                requirement_id=requirement.id,
            )
            for requirement in requirements
        )
    elif requirements and (
        environment_manifest is None
        or environment_manifest.factory_backed
        or registration.factory is not None
    ):
        diagnostics.extend(
            _diagnostic(
                ScenarioLaunchDiagnosticCode.EXECUTION_BINDING_REQUIRED,
                requirement_id=requirement.id,
            )
            for requirement in requirements
        )
    elif requirements:
        requirement_by_id = {item.id: item for item in requirements}
        oversized_event_ids: list[str] = []
        for event in validated.events:
            if not isinstance(
                event,
                ScenarioInitialInputEventV2
                | ScenarioQueuedInputEventV2
                | ScenarioResumedInputEventV2,
            ):
                continue
            occurrences = tuple(
                requirement_by_id[part.artifact_requirement_id]
                for message in event.input.messages
                for part in message.content
                if isinstance(part, ScenarioFilePartV2)
            )
            if (
                len(occurrences) > runtime.max_file_attachments_per_request
                or sum(item.size_bytes for item in occurrences)
                > runtime.max_total_file_attachment_bytes
                or any(item.size_bytes > runtime.max_file_attachment_bytes for item in occurrences)
            ):
                oversized_event_ids.append(event.id)
        if oversized_event_ids:
            diagnostics.extend(
                _diagnostic(
                    ScenarioLaunchDiagnosticCode.EXECUTION_LIMIT_EXCEEDED,
                    event_id=event_id,
                )
                for event_id in oversized_event_ids
            )
        else:
            store = registration.environment.artifact_store
            if not isinstance(store, ArtifactStore):
                diagnostics.extend(
                    _diagnostic(
                        ScenarioLaunchDiagnosticCode.ARTIFACT_BINDING_REQUIRED,
                        requirement_id=requirement.id,
                    )
                    for requirement in requirements
                )
            else:
                semaphore = asyncio.Semaphore(SCENARIO_PREFLIGHT_ARTIFACT_READ_CONCURRENCY)

                async def inspect(requirement: ScenarioArtifactRequirementV2):
                    artifact_id = selected.artifact_references.get(
                        requirement.id,
                        requirement.reference,
                    )
                    if artifact_id is None:
                        return None, _diagnostic(
                            ScenarioLaunchDiagnosticCode.ARTIFACT_BINDING_REQUIRED,
                            requirement_id=requirement.id,
                        )
                    try:
                        async with semaphore:
                            read = copy_artifact_read_result(
                                await store.read_bytes(
                                    artifact_id,
                                    max_bytes=requirement.size_bytes,
                                ),
                                expected_artifact_id=artifact_id,
                                max_content_bytes=requirement.size_bytes,
                            )
                    except (FileNotFoundError, InvalidArtifactIdError):
                        return None, _diagnostic(
                            ScenarioLaunchDiagnosticCode.ARTIFACT_NOT_RETAINED,
                            requirement_id=requirement.id,
                        )
                    except PermissionError:
                        return None, _diagnostic(
                            ScenarioLaunchDiagnosticCode.ARTIFACT_ACCESS_DENIED,
                            requirement_id=requirement.id,
                        )
                    except (ArtifactStoreUnavailableError, OSError):
                        return None, _diagnostic(
                            ScenarioLaunchDiagnosticCode.ARTIFACT_STORE_UNAVAILABLE,
                            requirement_id=requirement.id,
                        )
                    except Exception:
                        return None, _diagnostic(
                            ScenarioLaunchDiagnosticCode.ARTIFACT_CONTENT_INCONSISTENT,
                            requirement_id=requirement.id,
                        )
                    metadata = read.metadata
                    digest = hashlib.sha256(read.content).hexdigest()
                    if metadata.scope is ArtifactScope.SESSION:
                        return None, _diagnostic(
                            ScenarioLaunchDiagnosticCode.ARTIFACT_BINDING_REQUIRED,
                            requirement_id=requirement.id,
                        )
                    if (
                        read.truncated
                        or read.total_bytes != requirement.size_bytes
                        or len(read.content) != requirement.size_bytes
                        or metadata.scope is not ArtifactScope.ENVIRONMENT
                        or metadata.environment_name != environment_name
                        or metadata.agent_name not in {None, target.request_base.agent_name}
                        or metadata.filename != requirement.filename
                        or metadata.content_type != requirement.content_type
                        or metadata.size_bytes != requirement.size_bytes
                        or digest != requirement.content_sha256
                    ):
                        return None, _diagnostic(
                            ScenarioLaunchDiagnosticCode.ARTIFACT_CONTENT_INCONSISTENT,
                            requirement_id=requirement.id,
                        )
                    return (
                        ScenarioArtifactLaunchBindingV2(
                            requirement_id=requirement.id,
                            artifact_id=artifact_id,
                            content_sha256=digest,
                        ),
                        None,
                    )

                inspected = await asyncio.gather(*(inspect(item) for item in requirements))
                for binding, diagnostic in inspected:
                    if binding is not None:
                        artifact_bindings.append(binding)
                    if diagnostic is not None:
                        diagnostics.append(diagnostic)

    diagnostics = diagnostics[:SCENARIO_PREFLIGHT_MAX_DIAGNOSTICS]
    if diagnostics:
        return ScenarioLaunchPreflightResultV2(
            ready=False,
            scenario_revision=validated.revision,
            diagnostics=tuple(diagnostics),
        )
    binding = ScenarioLaunchBindingV2.create(
        scenario_revision=validated.revision,
        target_key=target.key,
        application_release_id=target.application_release_id,
        app_manifest_fingerprint=identity.app_manifest_fingerprint,
        agent_name=target.request_base.agent_name,
        environment_name=environment_name,
        approval_behavior=selected.approval_behavior,
        trials=selected.trials,
        max_concurrency=selected.max_concurrency,
        timeout_seconds=selected.timeout_seconds,
        max_steps=(
            target.request_base.max_steps if selected.max_steps is None else selected.max_steps
        ),
        target_limits=target.request_base.limits,
        operator_run_limits=selected.limits,
        cost_budget=selected.cost_budget,
        artifacts=tuple(sorted(artifact_bindings, key=lambda item: item.requirement_id)),
        secrets=tuple(sorted(secret_bindings, key=lambda item: item.requirement_id)),
    )
    return ScenarioLaunchPreflightResultV2(
        ready=True,
        scenario_revision=validated.revision,
        binding=binding,
    )


async def materialize_eval_scenario_artifact_fixture(
    scenario: EvalScenarioDocumentV2,
    target: CorpusTarget,
    requirement_id: str,
    *,
    environment_name: str | None = None,
    source_artifact_id: str | None = None,
    project_root: str | Path | None = None,
) -> ScenarioArtifactMaterializationV2:
    """Idempotently copy exact retained bytes into a reusable environment fixture."""

    if type(scenario) is not EvalScenarioDocumentV2:
        raise TypeError("scenario must be an exact EvalScenarioDocumentV2.")
    if type(target) is not CorpusTarget:
        raise TypeError("target must be an exact CorpusTarget.")
    validated = EvalScenarioDocumentV2.model_validate(_model_python_input(scenario))
    requirement_id = _portable_id(requirement_id, "requirement_id")
    if validated.target_key != target.key:
        raise ScenarioArtifactMaterializationError(ScenarioLaunchDiagnosticCode.TARGET_MISMATCH)
    requirement = next(
        (item for item in validated.artifact_requirements if item.id == requirement_id),
        None,
    )
    if requirement is None:
        raise KeyError(f"Scenario artifact requirement not found: {requirement_id}")
    if source_artifact_id is not None:
        source_artifact_id = require_durable_clean_nonblank(
            source_artifact_id,
            "source_artifact_id",
        )
    source_artifact_id = source_artifact_id or requirement.reference
    if source_artifact_id is None:
        raise ScenarioArtifactMaterializationError(
            ScenarioLaunchDiagnosticCode.ARTIFACT_BINDING_REQUIRED
        )
    settings = ScenarioLaunchSettingsV2(environment_name=environment_name)
    identity, selected_environment, environment_manifest, destination, diagnostics = (
        _environment_material(target, settings, project_root=project_root)
    )
    if diagnostics or selected_environment is None or destination is None:
        raise ScenarioArtifactMaterializationError(
            ScenarioLaunchDiagnosticCode.ENVIRONMENT_UNAVAILABLE
        )
    if environment_manifest is None or environment_manifest.factory_backed:
        raise ScenarioArtifactMaterializationError(
            ScenarioLaunchDiagnosticCode.EXECUTION_BINDING_REQUIRED
        )
    destination_store = destination.environment.artifact_store
    if not isinstance(destination_store, ArtifactStore):
        raise ScenarioArtifactMaterializationError(
            ScenarioLaunchDiagnosticCode.ARTIFACT_BINDING_REQUIRED
        )
    if requirement.size_bytes > identity.app_manifest.runtime.max_file_attachment_bytes:
        raise ScenarioArtifactMaterializationError(
            ScenarioLaunchDiagnosticCode.EXECUTION_LIMIT_EXCEEDED
        )

    try:
        read = copy_artifact_read_result(
            await destination_store.read_bytes(
                source_artifact_id,
                max_bytes=requirement.size_bytes,
            ),
            expected_artifact_id=source_artifact_id,
            max_content_bytes=requirement.size_bytes,
        )
    except (FileNotFoundError, InvalidArtifactIdError) as exc:
        raise ScenarioArtifactMaterializationError(
            ScenarioLaunchDiagnosticCode.ARTIFACT_NOT_RETAINED
        ) from exc
    except PermissionError as exc:
        raise ScenarioArtifactMaterializationError(
            ScenarioLaunchDiagnosticCode.ARTIFACT_ACCESS_DENIED
        ) from exc
    except (ArtifactStoreUnavailableError, OSError) as exc:
        raise ScenarioArtifactMaterializationError(
            ScenarioLaunchDiagnosticCode.ARTIFACT_STORE_UNAVAILABLE
        ) from exc
    except Exception as exc:
        raise ScenarioArtifactMaterializationError(
            ScenarioLaunchDiagnosticCode.ARTIFACT_CONTENT_INCONSISTENT
        ) from exc
    if (
        read.truncated
        or read.total_bytes != requirement.size_bytes
        or len(read.content) != requirement.size_bytes
        or read.metadata.environment_name != selected_environment
        or read.metadata.agent_name not in {None, target.request_base.agent_name}
        or read.metadata.filename != requirement.filename
        or read.metadata.content_type != requirement.content_type
        or hashlib.sha256(read.content).hexdigest() != requirement.content_sha256
    ):
        raise ScenarioArtifactMaterializationError(
            ScenarioLaunchDiagnosticCode.ARTIFACT_CONTENT_INCONSISTENT
        )
    content = read.content
    fixture_identity = canonical_durable_json_bytes(
        {
            "schema_version": 1,
            "target_key": target.key,
            "environment_name": selected_environment,
            "requirement": {
                "id": requirement.id,
                "content_sha256": requirement.content_sha256,
                "filename": requirement.filename,
                "content_type": requirement.content_type,
                "size_bytes": requirement.size_bytes,
            },
        },
        "scenario artifact fixture identity",
    )
    fixture_id = f"art_{hashlib.sha256(fixture_identity).hexdigest()[:32]}"
    replacement = requirement.model_copy(
        update={
            "source": "artifact_reference",
            "reference": fixture_id,
        }
    )
    updated = replace_eval_scenario_artifact_requirement(validated, replacement)
    try:
        public = updated.model_dump(mode="json")
        if target.app.redact_json(public) != public:
            raise ValueError("fixture scenario redacted")
    except Exception as exc:
        raise ScenarioArtifactMaterializationError(
            ScenarioLaunchDiagnosticCode.SCENARIO_CONTENT_UNSAFE
        ) from exc
    try:
        metadata = await destination_store.put_bytes(
            content,
            artifact_id=fixture_id,
            filename=requirement.filename,
            content_type=requirement.content_type,
            scope=ArtifactScope.ENVIRONMENT,
            agent_name=target.request_base.agent_name,
            environment_name=selected_environment,
            metadata={
                "cayu_eval_fixture": {
                    "schema_version": 1,
                    "content_sha256": requirement.content_sha256,
                    "requirement_id": requirement.id,
                }
            },
        )
    except PermissionError as exc:
        raise ScenarioArtifactMaterializationError(
            ScenarioLaunchDiagnosticCode.ARTIFACT_ACCESS_DENIED
        ) from exc
    except (ArtifactStoreUnavailableError, OSError) as exc:
        raise ScenarioArtifactMaterializationError(
            ScenarioLaunchDiagnosticCode.ARTIFACT_STORE_UNAVAILABLE
        ) from exc
    except Exception as exc:
        raise ScenarioArtifactMaterializationError(
            ScenarioLaunchDiagnosticCode.ARTIFACT_CONTENT_INCONSISTENT
        ) from exc
    if (
        metadata.id != fixture_id
        or metadata.scope is not ArtifactScope.ENVIRONMENT
        or metadata.environment_name != selected_environment
        or metadata.filename != requirement.filename
        or metadata.content_type != requirement.content_type
        or metadata.size_bytes != requirement.size_bytes
    ):
        raise ScenarioArtifactMaterializationError(
            ScenarioLaunchDiagnosticCode.ARTIFACT_CONTENT_INCONSISTENT
        )
    return ScenarioArtifactMaterializationV2(
        scenario=updated,
        requirement_id=requirement.id,
        artifact_id=fixture_id,
    )


__all__ = [
    "SCENARIO_PREFLIGHT_ARTIFACT_READ_CONCURRENCY",
    "SCENARIO_PREFLIGHT_MAX_DIAGNOSTICS",
    "ScenarioArtifactLaunchBindingV2",
    "ScenarioArtifactMaterializationError",
    "ScenarioArtifactMaterializationV2",
    "ScenarioLaunchBindingV2",
    "ScenarioLaunchDiagnosticCode",
    "ScenarioLaunchDiagnosticV2",
    "ScenarioLaunchPreflightResultV2",
    "ScenarioLaunchSettingsV2",
    "ScenarioSecretLaunchBindingV2",
    "materialize_eval_scenario_artifact_fixture",
    "preflight_eval_scenario",
]
