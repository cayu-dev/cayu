"""Server-owned executable target registry for project Evals."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

from cayu._validation import require_durable_clean_nonblank, require_unicode_scalar_text
from cayu.core.agents import AgentSpec
from cayu.evals._execution_profile_errors import EvalExecutionProfileChangedError
from cayu.evals.capacity import EvalExecutionCapacity
from cayu.evals.corpus import JudgePrivacyPolicyV1
from cayu.evals.execution import (
    CorpusExecutionLimits,
    CorpusTarget,
    ModelJudgeTarget,
    _candidate_judge_route_relation,
    evaluation_target_identity,
    model_judge_profile,
)
from cayu.evals.execution_profiles import (
    EvalExecutionProfilePolicyV1,
    PreparedEvalExecutionProfile,
    prepare_eval_execution_profile,
)
from cayu.evals.store import EvalRunInvocation, EvalStore
from cayu.project_control_plane import ProjectEvalJudgeConfiguration
from cayu.runtime.app import CayuApp
from cayu.runtime.budgets import BudgetLimit, budget_pricing_preflight_error
from cayu.runtime.costs import PriceBook
from cayu.runtime.invocation import SessionExecutionSource
from cayu.runtime.sessions import (
    RunRequest,
    copy_run_request,
    run_request_with_runtime_invocation,
)
from cayu.runtime.stop_policy import RunLimits
from cayu.server.config import (
    DEFAULT_EVAL_LEASE_SECONDS,
    DEFAULT_EVAL_POLL_INTERVAL_SECONDS,
    DEFAULT_EVAL_SHUTDOWN_GRACE_SECONDS,
    EvalsConfig,
)
from cayu.server.contracts import (
    MAX_EVAL_TARGET_COMPONENT_CHARS,
    MAX_EVAL_TARGETS,
    EvalExecutionProfileDiagnostic,
    EvalJudgePrivateReferenceCatalogEntry,
    EvalJudgeProfileRouteCatalogEntry,
    EvalTargetCatalogEntry,
    EvalTargetCatalogResponse,
)

DEFAULT_EVAL_PROFILE_ID = "default"
_EXPLICIT_EVAL_PROFILE_ID = "explicit"
_TARGET_KEY_DOMAIN = b"cayu-generated-eval-target-v1\0"
_EVAL_PROFILE_RESOLUTION_CONCURRENCY = 16
_GENERATED_JUDGE_AGENT_NAME = "cayu-evals-default-judge"
_GENERATED_JUDGE_KEY = "project-default-judge"


def _narrow_optional_limit(current: int | None, requested: int | None) -> int | None:
    if current is None:
        return requested
    if requested is None:
        return current
    return min(current, requested)


def _narrow_run_limits(current: RunLimits, requested: RunLimits | None) -> RunLimits:
    if requested is None:
        return RunLimits.model_validate(current.model_dump(mode="python"))
    return RunLimits(
        max_input_tokens=_narrow_optional_limit(
            current.max_input_tokens,
            requested.max_input_tokens,
        ),
        max_output_tokens=_narrow_optional_limit(
            current.max_output_tokens,
            requested.max_output_tokens,
        ),
        max_total_tokens=_narrow_optional_limit(
            current.max_total_tokens,
            requested.max_total_tokens,
        ),
        max_tool_calls=_narrow_optional_limit(
            current.max_tool_calls,
            requested.max_tool_calls,
        ),
        max_elapsed_seconds=_narrow_optional_limit(
            current.max_elapsed_seconds,
            requested.max_elapsed_seconds,
        ),
        # Every eval trial creates a fresh session, so run and session scope
        # measure the same initial invocation. The persisted operator ceilings
        # deliberately use run scope and cannot broaden a target base limit.
        scope="run",
    )


def cost_budget_currencies_for_target(target: CorpusTarget) -> tuple[str, ...]:
    """Return currencies currently compatible with one target's resolved model route."""

    if type(target) is not CorpusTarget:
        raise TypeError("target must be an exact CorpusTarget.")
    pricing = target.price_book
    if pricing is None:
        return ()
    try:
        model_target = target.app.resolve_run_model_target(target.request_base)
        provider = target.app.get_provider(model_target.provider_name)
    except (KeyError, RuntimeError, ValueError):
        # Provider readiness is independent from catalog availability. A server
        # with an unroutable target must remain inspectable, but it cannot safely
        # advertise or admit a model-priced cost ceiling.
        return ()
    pricing_provider_name = provider.billing_provider_name or model_target.provider_name
    effective_at = datetime.now(UTC)
    candidate_currencies = tuple(
        sorted(
            {
                schedule.pricing.currency.upper()
                for price in pricing.prices
                for schedule in price.schedules
            }
        )
    )
    return tuple(
        currency
        for currency in candidate_currencies
        if budget_pricing_preflight_error(
            pricing,
            provider_name=pricing_provider_name,
            model=model_target.model,
            currency=currency,
            effective_at=effective_at,
        )
        is None
    )


def target_for_eval_invocation(
    target: CorpusTarget,
    invocation: EvalRunInvocation,
) -> CorpusTarget:
    """Apply one durable run's contractions and trusted provenance to a target."""

    if type(target) is not CorpusTarget:
        raise TypeError("target must be an exact CorpusTarget.")
    if type(invocation) is not EvalRunInvocation:
        raise TypeError("invocation must be an exact EvalRunInvocation.")
    request = copy_run_request(target.request_base)
    if (
        invocation.source is SessionExecutionSource.HTTP_RUN
        and request.invocation_origin is not None
    ):
        raise ValueError("HTTP eval execution cannot inherit a host-asserted SDK origin.")
    scenario_environment = (
        None if invocation.scenario is None else invocation.scenario.environment_name
    )
    if (
        scenario_environment is not None
        and request.environment_name is not None
        and scenario_environment != request.environment_name
    ):
        raise ValueError("Eval scenario environment conflicts with the published target.")
    budget_limits = request.budget_limits
    if invocation.cost_budget is not None:
        if target.price_book is None:
            raise ValueError("A cost-bounded eval run requires server-owned target pricing.")
        compatible_currencies = cost_budget_currencies_for_target(target)
        if invocation.cost_budget.currency not in compatible_currencies:
            raise ValueError(
                "A cost-bounded eval run requires pricing compatible with the target model "
                "and requested currency."
            )
        budget_limits = (
            *budget_limits,
            BudgetLimit(
                scope="run",
                max_estimated_cost=invocation.cost_budget.max_estimated_cost,
                pricing=target.price_book,
                currency=invocation.cost_budget.currency,
                allow_unpriced=False,
                action="interrupt",
            ),
        )
    request = request.model_copy(
        update={
            "environment_name": scenario_environment or request.environment_name,
            "max_steps": (
                request.max_steps
                if invocation.max_steps is None
                else min(request.max_steps, invocation.max_steps)
            ),
            "limits": _narrow_run_limits(request.limits, invocation.limits),
            "budget_limits": budget_limits,
        }
    )
    request = run_request_with_runtime_invocation(
        copy_run_request(request),
        source=invocation.source,
        verified_origin=invocation.origin,
    )
    return CorpusTarget(
        key=target.key,
        app=target.app,
        request_base=request,
        bootstrap_messages=target.bootstrap_messages,
        application_release_id=target.application_release_id,
        evidence_policy=target.evidence_policy,
        price_book=target.price_book,
        model_judges=target.model_judges,
        limits=target.limits,
        external_process=target.external_process,
    )


def _target_identity_component(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    require_unicode_scalar_text(value, field_name)
    if len(value) > MAX_EVAL_TARGET_COMPONENT_CHARS:
        raise ValueError(
            f"{field_name} cannot exceed {MAX_EVAL_TARGET_COMPONENT_CHARS} characters."
        )
    return value


def derive_eval_target_key(
    *,
    project_id: str,
    agent_name: str,
    profile_id: str = DEFAULT_EVAL_PROFILE_ID,
) -> str:
    """Derive a release-independent key from unambiguous logical identity bytes."""

    components = (
        _target_identity_component(project_id, "project_id"),
        _target_identity_component(agent_name, "agent_name"),
        _target_identity_component(profile_id, "profile_id"),
    )
    digest = hashlib.sha256()
    digest.update(_TARGET_KEY_DOMAIN)
    for component in components:
        encoded = component.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return f"eval.{digest.hexdigest()}"


@dataclass(frozen=True, slots=True, repr=False)
class EvalTargetRegistration:
    """One public identity paired with process-local execution authority."""

    catalog_entry: EvalTargetCatalogEntry
    target: CorpusTarget
    execution_profile_policy: EvalExecutionProfilePolicyV1
    manifest_project_root: Path | None = None

    def __post_init__(self) -> None:
        if type(self.catalog_entry) is not EvalTargetCatalogEntry:
            raise TypeError("catalog_entry must be an exact EvalTargetCatalogEntry.")
        if type(self.target) is not CorpusTarget:
            raise TypeError("target must be an exact CorpusTarget.")
        if type(self.execution_profile_policy) is not EvalExecutionProfilePolicyV1:
            raise TypeError(
                "execution_profile_policy must be an exact EvalExecutionProfilePolicyV1."
            )
        if self.catalog_entry.target_key != self.target.key:
            raise ValueError("Eval target catalog key does not match its runtime target.")
        if self.catalog_entry.agent_name != self.target.request_base.agent_name:
            raise ValueError("Eval target catalog agent does not match its request authority.")
        if self.catalog_entry.application_release_id != self.target.application_release_id:
            raise ValueError("Eval target catalog release does not match its runtime target.")
        policy = self.execution_profile_policy
        if (
            policy.max_trials > self.target.limits.max_trials
            or policy.max_concurrency > self.target.limits.max_concurrency
        ):
            raise ValueError("Eval execution profile exceeds its runtime target authority.")
        if (
            self.catalog_entry.max_trials != policy.max_trials
            or self.catalog_entry.max_concurrency != policy.max_concurrency
            or self.catalog_entry.max_timeout_seconds != self.target.limits.max_timeout_seconds
            or self.catalog_entry.max_steps != self.target.request_base.max_steps
        ):
            raise ValueError("Eval target catalog limits do not match its execution authority.")
        if self.manifest_project_root is not None and (
            not isinstance(self.manifest_project_root, Path)
            or not self.manifest_project_root.is_absolute()
        ):
            raise TypeError("manifest_project_root must be an absolute Path or None.")

    def execution_target(self) -> CorpusTarget:
        """Return this target narrowed to its published Evals execution policy."""

        policy = self.execution_profile_policy
        limits = self.target.limits.model_copy(
            update={
                "max_trials": policy.max_trials,
                "max_concurrency": policy.max_concurrency,
            }
        )
        return CorpusTarget(
            key=self.target.key,
            app=self.target.app,
            request_base=self.target.request_base,
            bootstrap_messages=self.target.bootstrap_messages,
            application_release_id=self.target.application_release_id,
            evidence_policy=self.target.evidence_policy,
            price_book=self.target.price_book,
            model_judges=self.target.model_judges,
            limits=limits,
            external_process=self.target.external_process,
        )


class EvalTargetRegistry:
    """Immutable bounded map from published keys to local runtime authority."""

    __slots__ = ("_catalog", "_registrations")

    def __init__(self, registrations: Iterable[EvalTargetRegistration]) -> None:
        if isinstance(registrations, str | bytes):
            raise TypeError("registrations must be an iterable of target registrations.")
        items = tuple(registrations)
        if not items:
            raise ValueError("An eval target registry cannot be empty.")
        if len(items) > MAX_EVAL_TARGETS:
            raise ValueError(f"An eval target registry cannot exceed {MAX_EVAL_TARGETS} targets.")
        if any(type(item) is not EvalTargetRegistration for item in items):
            raise TypeError("registrations must contain exact EvalTargetRegistration values.")

        ordered = tuple(
            sorted(
                items,
                key=lambda item: (
                    item.catalog_entry.agent_name,
                    item.catalog_entry.profile_id,
                    item.catalog_entry.target_key,
                ),
            )
        )
        keys = tuple(item.target.key for item in ordered)
        if len(keys) != len(set(keys)):
            raise ValueError("Eval target key collision detected while building the registry.")
        logical_identities = tuple(
            (
                item.catalog_entry.project_id,
                item.catalog_entry.agent_name,
                item.catalog_entry.profile_id,
            )
            for item in ordered
        )
        if len(logical_identities) != len(set(logical_identities)):
            raise ValueError("Eval target logical identities must be unique.")
        app = ordered[0].target.app
        if any(item.target.app is not app for item in ordered):
            raise ValueError("All eval registry targets must belong to one CayuApp instance.")
        manifest_project_root = ordered[0].manifest_project_root
        if any(item.manifest_project_root != manifest_project_root for item in ordered):
            raise ValueError("All eval registry targets must use one application-manifest root.")
        manifest_identity = evaluation_target_identity(
            ordered[0].target,
            project_root=manifest_project_root,
        )
        if any(
            item.catalog_entry.app_manifest_fingerprint
            != manifest_identity.app_manifest_fingerprint
            for item in ordered
        ):
            raise ValueError(
                "Eval target catalog manifest does not match its runtime provenance policy."
            )

        registrations_by_key = {item.target.key: item for item in ordered}
        self._registrations = MappingProxyType(registrations_by_key)
        self._catalog = EvalTargetCatalogResponse(
            items=tuple(item.catalog_entry for item in ordered),
            default_target_key=ordered[0].target.key,
        )

    def __repr__(self) -> str:
        return f"EvalTargetRegistry(target_count={len(self._registrations)})"

    @property
    def target_keys(self) -> tuple[str, ...]:
        return tuple(item.target_key for item in self._catalog.items)

    @property
    def default_target_key(self) -> str:
        return self._catalog.default_target_key

    def catalog(self) -> EvalTargetCatalogResponse:
        """Return the immutable registry projection without runtime preparation."""

        return self._catalog.model_copy(deep=True)

    async def resolved_catalog(self) -> EvalTargetCatalogResponse:
        """Resolve current profile identity for every bounded published target."""

        resolution_limit = asyncio.Semaphore(_EVAL_PROFILE_RESOLUTION_CONCURRENCY)

        async def resolve(registration: EvalTargetRegistration) -> EvalTargetCatalogEntry:
            async with resolution_limit:
                try:
                    prepared = await self.prepare_execution_profile(registration.target.key)
                except asyncio.CancelledError:
                    raise
                except EvalExecutionProfileChangedError:
                    diagnostic = EvalExecutionProfileDiagnostic.for_code(
                        "application_identity_changed"
                    )
                except Exception:
                    diagnostic = EvalExecutionProfileDiagnostic.for_code(
                        "runtime_authority_unavailable"
                    )
                else:
                    return registration.catalog_entry.model_copy(
                        update={
                            "execution_profile_ready": True,
                            "execution_profile": prepared.snapshot,
                            "execution_profile_diagnostics": (),
                        },
                        deep=True,
                    )
                return registration.catalog_entry.model_copy(
                    update={
                        "execution_profile_ready": False,
                        "execution_profile": None,
                        "execution_profile_diagnostics": (diagnostic,),
                    },
                    deep=True,
                )

        entries = await asyncio.gather(
            *(resolve(registration) for registration in self._registrations.values())
        )
        return EvalTargetCatalogResponse(
            items=tuple(entries),
            default_target_key=self.default_target_key,
        )

    async def prepare_execution_profile(
        self,
        target_key: str,
        *,
        effective_target: CorpusTarget | None = None,
    ) -> PreparedEvalExecutionProfile:
        """Resolve one exact current profile without admitting runtime state."""

        registration = self.registration(target_key)
        if registration is None:
            raise KeyError(f"Eval target not found: {target_key}")
        target = registration.execution_target() if effective_target is None else effective_target
        if (
            type(target) is not CorpusTarget
            or target.key != target_key
            or target.app is not registration.target.app
        ):
            raise ValueError("Effective eval target does not match its published registration.")
        target = CorpusTarget(
            key=target.key,
            app=target.app,
            request_base=target.request_base,
            bootstrap_messages=target.bootstrap_messages,
            application_release_id=target.application_release_id,
            evidence_policy=target.evidence_policy,
            price_book=target.price_book,
            model_judges=target.model_judges,
            limits=target.limits.model_copy(
                update={
                    "max_trials": registration.execution_profile_policy.max_trials,
                    "max_concurrency": registration.execution_profile_policy.max_concurrency,
                }
            ),
            external_process=target.external_process,
        )
        identity = evaluation_target_identity(
            target,
            project_root=registration.manifest_project_root,
        )
        if identity.app_manifest_fingerprint != registration.catalog_entry.app_manifest_fingerprint:
            raise EvalExecutionProfileChangedError(
                "Current eval target manifest does not match its published registration."
            )
        prepared = await prepare_eval_execution_profile(
            target,
            profile_id=registration.catalog_entry.profile_id,
            label=registration.catalog_entry.label,
            source=registration.catalog_entry.source,
            app_manifest_fingerprint=registration.catalog_entry.app_manifest_fingerprint,
            policy=registration.execution_profile_policy,
        )
        durable_invocation = EvalRunInvocation(
            execution_profile=prepared.binding,
            execution_profile_snapshot=prepared.snapshot,
        )
        durable_binding = durable_invocation.execution_profile
        if durable_binding != prepared.binding:
            raise RuntimeError("Eval execution profile binding did not round-trip durably.")
        if durable_invocation.execution_profile_snapshot != prepared.snapshot:
            raise RuntimeError("Eval execution profile snapshot did not round-trip durably.")
        identity_after = evaluation_target_identity(
            target,
            project_root=registration.manifest_project_root,
        )
        if (
            identity_after.app_manifest_fingerprint
            != registration.catalog_entry.app_manifest_fingerprint
        ):
            raise EvalExecutionProfileChangedError(
                "Current eval target manifest changed during profile preparation."
            )
        return prepared

    def get(self, target_key: str) -> CorpusTarget | None:
        registration = self._registrations.get(target_key)
        return None if registration is None else registration.target

    def registration(self, target_key: str) -> EvalTargetRegistration | None:
        """Resolve executable authority and its manifest policy as one immutable value."""

        return self._registrations.get(target_key)

    def registration_for_agent(
        self,
        agent_name: str,
    ) -> EvalTargetRegistration | None:
        """Resolve the unambiguous published target for one session agent."""

        agent_name = _target_identity_component(agent_name, "agent_name")
        matches = tuple(
            registration
            for registration in self._registrations.values()
            if registration.catalog_entry.agent_name == agent_name
        )
        if len(matches) == 1:
            return matches[0]
        defaults = tuple(
            registration
            for registration in matches
            if registration.catalog_entry.profile_id == DEFAULT_EVAL_PROFILE_ID
        )
        return defaults[0] if len(defaults) == 1 else None


@dataclass(frozen=True, slots=True, repr=False)
class ResolvedEvalsRuntime:
    """Validated store, registry, and worker policy used by server routes."""

    registry: EvalTargetRegistry
    store: EvalStore
    execution_capacity: EvalExecutionCapacity
    lease_seconds: int
    poll_interval_seconds: float
    shutdown_grace_seconds: float

    def __post_init__(self) -> None:
        if type(self.registry) is not EvalTargetRegistry:
            raise TypeError("registry must be an exact EvalTargetRegistry.")
        if not isinstance(self.store, EvalStore) or not self.store.durable:
            raise TypeError("store must be a durable EvalStore.")
        if type(self.execution_capacity) is not EvalExecutionCapacity:
            raise TypeError("execution_capacity must be an exact EvalExecutionCapacity.")


def generated_eval_target_registry(
    app: CayuApp,
    *,
    project_id: str,
    application_release_id: str,
    app_manifest_fingerprint: str,
    app_manifest_project_root: Path | None = None,
    price_book: PriceBook | None = None,
    judge_configuration: ProjectEvalJudgeConfiguration | None = None,
) -> EvalTargetRegistry | None:
    """Build one normal-authority target per currently registered agent."""

    if not isinstance(app, CayuApp):
        raise TypeError("app must be a CayuApp.")
    project_id = _target_identity_component(project_id, "project_id")
    application_release_id = _target_identity_component(
        application_release_id,
        "application_release_id",
    )
    app_manifest_fingerprint = require_durable_clean_nonblank(
        app_manifest_fingerprint,
        "app_manifest_fingerprint",
    )
    agent_names = app.list_agents()
    if not agent_names:
        return None
    if len(agent_names) > MAX_EVAL_TARGETS:
        raise ValueError(f"Automatic Evals supports at most {MAX_EVAL_TARGETS} registered agents.")
    if (
        judge_configuration is not None
        and type(judge_configuration) is not ProjectEvalJudgeConfiguration
    ):
        raise TypeError(
            "judge_configuration must be an exact ProjectEvalJudgeConfiguration or None."
        )

    policy = EvalExecutionProfilePolicyV1.safe_default()
    model_judges = _generated_project_model_judges(
        app,
        judge_configuration,
        price_book=price_book,
    )
    registrations: list[EvalTargetRegistration] = []
    for agent_name in agent_names:
        agent_name = _target_identity_component(agent_name, "agent_name")
        target_key = derive_eval_target_key(
            project_id=project_id,
            agent_name=agent_name,
            profile_id=DEFAULT_EVAL_PROFILE_ID,
        )
        target = CorpusTarget(
            key=target_key,
            app=app,
            request_base=RunRequest(agent_name=agent_name, messages=[]),
            application_release_id=application_release_id,
            price_book=price_book,
            model_judges=model_judges,
            limits=CorpusExecutionLimits(max_trials=1, max_concurrency=1),
        )
        cost_budget_currencies = cost_budget_currencies_for_target(target)
        judge_profiles = tuple(model_judge_profile(judge) for judge in model_judges)
        entry = EvalTargetCatalogEntry(
            target_key=target_key,
            project_id=project_id,
            agent_name=agent_name,
            profile_id=DEFAULT_EVAL_PROFILE_ID,
            label=f"{agent_name} · Default",
            source="generated",
            application_release_id=application_release_id,
            app_manifest_fingerprint=app_manifest_fingerprint,
            max_trials=target.limits.max_trials,
            max_concurrency=target.limits.max_concurrency,
            max_timeout_seconds=target.limits.max_timeout_seconds,
            max_steps=target.request_base.max_steps,
            cost_budget_available=bool(cost_budget_currencies),
            cost_budget_currencies=cost_budget_currencies,
            judge_profiles=judge_profiles,
            judge_profile_routes=tuple(
                EvalJudgeProfileRouteCatalogEntry(
                    judge_profile_key=profile.key,
                    judge_profile_revision=profile.revision,
                    candidate_route_relation=_candidate_judge_route_relation(target, profile),
                )
                for profile in judge_profiles
            ),
            judge_private_references=(),
            execution_profile_ready=False,
            execution_profile=None,
            execution_profile_diagnostics=(
                EvalExecutionProfileDiagnostic.for_code("not_resolved"),
            ),
        )
        _require_public_catalog_entry(app, entry)
        registrations.append(
            EvalTargetRegistration(
                catalog_entry=entry,
                target=target,
                execution_profile_policy=policy,
                manifest_project_root=app_manifest_project_root,
            )
        )
    return EvalTargetRegistry(registrations)


def _generated_project_model_judges(
    app: CayuApp,
    configuration: ProjectEvalJudgeConfiguration | None,
    *,
    price_book: PriceBook | None,
) -> tuple[ModelJudgeTarget, ...]:
    if configuration is None:
        return ()
    if configuration.max_estimated_cost is not None and price_book is None:
        raise ValueError("Project default judge cost ceiling requires project Evals pricing.")
    public_route = {
        "provider_name": configuration.provider_name,
        "model": configuration.model,
    }
    if app.redact_json(public_route) != public_route:
        raise ValueError("Project default judge route contains a workload secret.")
    provider = app.get_provider(configuration.provider_name)
    judge_app = CayuApp(enable_logging=False)
    judge_app.register_provider(provider, default=True)
    judge_app.register_agent(
        AgentSpec(
            name=_GENERATED_JUDGE_AGENT_NAME,
            provider_name=configuration.provider_name,
            model=configuration.model,
        )
    )
    privacy_policy = JudgePrivacyPolicyV1.create(
        key=configuration.privacy_policy,
        allow_transcript=configuration.privacy_policy == "public-and-transcript",
        allow_public_reference=True,
        allow_private_reference=False,
    )
    return (
        ModelJudgeTarget(
            key=_GENERATED_JUDGE_KEY,
            label="Project default judge",
            app=judge_app,
            agent_name=_GENERATED_JUDGE_AGENT_NAME,
            privacy_policy=privacy_policy,
            timeout_seconds=configuration.timeout_seconds,
            max_input_tokens=configuration.max_input_tokens,
            max_output_tokens=configuration.max_output_tokens,
            max_total_tokens=configuration.max_total_tokens,
            max_estimated_cost=configuration.max_estimated_cost,
            cost_currency=configuration.cost_currency or "USD",
            price_book=(price_book if configuration.max_estimated_cost is not None else None),
            allow_same_model=configuration.allow_same_model,
        ),
    )


def explicit_eval_target_registry(
    target: CorpusTarget,
    *,
    policy: EvalExecutionProfilePolicyV1 | None = None,
) -> EvalTargetRegistry:
    """Adapt the V1 singleton target into the common registry contract."""

    if type(target) is not CorpusTarget:
        raise TypeError("target must be an exact CorpusTarget.")
    policy = EvalExecutionProfilePolicyV1.safe_default() if policy is None else policy
    if type(policy) is not EvalExecutionProfilePolicyV1:
        raise TypeError("policy must be an exact EvalExecutionProfilePolicyV1 or None.")
    identity = evaluation_target_identity(target)
    agent_name = _target_identity_component(target.request_base.agent_name, "agent_name")
    cost_budget_currencies = cost_budget_currencies_for_target(target)
    judge_profiles = tuple(
        sorted(
            (model_judge_profile(judge) for judge in target.model_judges),
            key=lambda profile: profile.key,
        )
    )
    profiles_by_key = {profile.key: profile for profile in judge_profiles}
    judge_profile_routes = tuple(
        EvalJudgeProfileRouteCatalogEntry(
            judge_profile_key=profile.key,
            judge_profile_revision=profile.revision,
            candidate_route_relation=_candidate_judge_route_relation(target, profile),
        )
        for profile in judge_profiles
    )
    judge_private_references = tuple(
        sorted(
            (
                EvalJudgePrivateReferenceCatalogEntry(
                    judge_profile_key=judge.key,
                    judge_profile_revision=profiles_by_key[judge.key].revision,
                    reference=reference.portable_identity(),
                )
                for judge in target.model_judges
                for reference in judge.private_references
            ),
            key=lambda item: (
                item.judge_profile_key,
                item.reference.key,
                item.reference.revision,
            ),
        )
    )
    entry = EvalTargetCatalogEntry(
        target_key=target.key,
        project_id=None,
        agent_name=agent_name,
        profile_id=_EXPLICIT_EVAL_PROFILE_ID,
        label=f"{agent_name} · Explicit",
        source="explicit",
        application_release_id=identity.application_release_id,
        app_manifest_fingerprint=identity.app_manifest.fingerprint,
        max_trials=policy.max_trials,
        max_concurrency=policy.max_concurrency,
        max_timeout_seconds=target.limits.max_timeout_seconds,
        max_steps=target.request_base.max_steps,
        cost_budget_available=bool(cost_budget_currencies),
        cost_budget_currencies=cost_budget_currencies,
        judge_profiles=judge_profiles,
        judge_profile_routes=judge_profile_routes,
        judge_private_references=judge_private_references,
        execution_profile_ready=False,
        execution_profile=None,
        execution_profile_diagnostics=(EvalExecutionProfileDiagnostic.for_code("not_resolved"),),
    )
    _require_public_catalog_entry(target.app, entry)
    return EvalTargetRegistry(
        (
            EvalTargetRegistration(
                catalog_entry=entry,
                target=target,
                execution_profile_policy=policy,
            ),
        )
    )


def resolved_evals_runtime(
    *,
    explicit: EvalsConfig | None,
    registry: EvalTargetRegistry | None,
    automatic_store: EvalStore | None,
) -> ResolvedEvalsRuntime | None:
    """Apply indivisible explicit-V1 precedence to the generated project plan."""

    if explicit is not None:
        if type(explicit) is not EvalsConfig:
            raise TypeError("explicit must be an exact EvalsConfig or None.")
        return ResolvedEvalsRuntime(
            registry=explicit_eval_target_registry(
                explicit.target,
                policy=explicit.execution_profile_policy,
            ),
            store=explicit.store,
            execution_capacity=explicit.execution_capacity,
            lease_seconds=explicit.lease_seconds,
            poll_interval_seconds=explicit.poll_interval_seconds,
            shutdown_grace_seconds=explicit.shutdown_grace_seconds,
        )
    if registry is None or automatic_store is None:
        return None
    return ResolvedEvalsRuntime(
        registry=registry,
        store=automatic_store,
        execution_capacity=EvalExecutionCapacity(),
        lease_seconds=DEFAULT_EVAL_LEASE_SECONDS,
        poll_interval_seconds=DEFAULT_EVAL_POLL_INTERVAL_SECONDS,
        shutdown_grace_seconds=DEFAULT_EVAL_SHUTDOWN_GRACE_SECONDS,
    )


def _require_public_catalog_entry(app: CayuApp, entry: EvalTargetCatalogEntry) -> None:
    public = entry.model_dump(mode="json")
    try:
        redacted = app.redact_json(public)
    except Exception as exc:
        raise ValueError(
            "Eval target identity could not cross the application redaction boundary."
        ) from exc
    if redacted != public:
        raise ValueError("Eval target identity contains a workload secret.")


__all__ = [
    "DEFAULT_EVAL_PROFILE_ID",
    "EvalTargetRegistry",
    "ResolvedEvalsRuntime",
    "derive_eval_target_key",
    "explicit_eval_target_registry",
    "generated_eval_target_registry",
    "resolved_evals_runtime",
    "target_for_eval_invocation",
]
