"""CayuApp's managed-settings declarations and the schema they assemble into.

``CayuApp`` inherits these; they live here so the application module stays
within its size limit. Parts declare settings where they are registered; this
module only collects the declarations, adds the framework's own (chosen
``CayuConfig`` fields and keyed application budget limits) and assembles the
Release's schema.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ValidationError

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.budgets.base import BudgetPolicy
from cayu.configuration import MAX_STEPS, CayuConfig, CayuConfigSource
from cayu.context.thinking import ThinkingConfig
from cayu.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfileIdentity,
    _available_component,
)
from cayu.managed_settings import (
    KEY_PATTERN,
    MAX_KEY_LENGTH,
    MAX_REASON_LENGTH,
    ManagedSetting,
    ManagedSettings,
    agent_setting_defaults,
    managed_settings_schema_document,
    namespace_segment,
    namespaced_settings,
    thinking_effort_choices,
)
from cayu.runtime._application_run_defaults import (
    _RunConfigurationRequest,
    with_application_run_defaults,
)
from cayu.runtime._execution_profile_admission import model_finalization_material
from cayu.runtime.application_lifecycle import _admitted_entrance
from cayu.sessions.requests import RunRequest

if TYPE_CHECKING:
    from cayu.agents import AgentSpec
    from cayu.applications import CayuApp
    from cayu.budgets.base import BudgetLimit
    from cayu.sessions.records import Session


@dataclass(frozen=True, slots=True)
class ConfigSetting:
    """One ``CayuConfig`` field promoted to a managed setting."""

    setting: ManagedSetting
    # The field's path in CAYU_CONFIG_FIELD_OWNERS, whose source becomes "managed".
    path: str
    read: Callable[[CayuConfig], Any]
    # Put a value into CayuConfig.model_dump(mode="python").
    write: Callable[[dict[str, Any], Any], None]


def _write_run(field: str) -> Callable[[dict[str, Any], Any], None]:
    def write(data: dict[str, Any], value: Any) -> None:
        data["run"][field] = value

    return write


def _write_nested(field: str, name: str) -> Callable[[dict[str, Any], Any], None]:
    def write(data: dict[str, Any], value: Any) -> None:
        data["run"][field] = {**(data["run"][field] or {}), name: value}

    return write


def _limit(name: str, label: str, description: str) -> ConfigSetting:
    return ConfigSetting(
        ManagedSetting(
            key=f"config.run.limits.{name}",
            owner="config",
            type="int",
            minimum=1,
            maximum=MAX_DURABLE_JSON_INTEGER,
            label=label,
            description=description,
        ),
        "run.limits",
        lambda config: getattr(config.run.limits, name),
        _write_nested("limits", name),
    )


# The CayuConfig fields Cloud can manage. Run limits, max_steps and retries apply
# to new sessions; thinking is folded into each agent when it is registered, so
# it applies at restart.
CONFIG_SETTINGS: tuple[ConfigSetting, ...] = (
    ConfigSetting(
        ManagedSetting(
            key="config.run.max_steps",
            owner="config",
            type="int",
            minimum=1,
            maximum=MAX_STEPS,
            label="Maximum steps",
            description="Model steps a run may take before it stops.",
        ),
        "run.max_steps",
        lambda config: config.run.max_steps,
        _write_run("max_steps"),
    ),
    _limit("max_input_tokens", "Input token limit", "Input tokens a run may use."),
    _limit("max_output_tokens", "Output token limit", "Output tokens a run may use."),
    _limit("max_total_tokens", "Total token limit", "Input and output tokens a run may use."),
    _limit("max_tool_calls", "Tool call limit", "Tool calls a run may make."),
    _limit("max_elapsed_seconds", "Time limit (seconds)", "Seconds a run may take."),
    ConfigSetting(
        ManagedSetting(
            key="config.run.retry_policy.max_attempts",
            owner="config",
            type="int",
            minimum=1,
            maximum=50,
            label="Model call attempts",
            description="Attempts at one model call, counting the first, before it fails.",
        ),
        "run.retry_policy",
        lambda config: config.run.retry_policy.max_attempts,
        _write_nested("retry_policy", "max_attempts"),
    ),
    ConfigSetting(
        ManagedSetting(
            key="config.run.thinking.effort",
            owner="config",
            type="enum",
            choices=thinking_effort_choices(),
            applies="restart",
            label="Thinking effort",
            description="Default reasoning effort for agents that don't set their own.",
        ),
        "run.thinking",
        lambda config: None if config.run.thinking is None else config.run.thinking.effort,
        _write_nested("thinking", "effort"),
    ),
)


def config_settings(config: CayuConfig) -> tuple[ManagedSetting, ...]:
    """The promoted config settings, with the code's values as defaults."""

    return tuple(
        item.setting.model_copy(update={"default": item.read(config)}) for item in CONFIG_SETTINGS
    )


def budget_limit_name(limit: BudgetLimit) -> str:
    """A budget limit's name in setting keys: its scope, key and window (``app-month``)."""

    parts: list[str] = [limit.scope] if limit.key is None else [limit.scope, limit.key]
    window = limit.window
    if window.kind == "calendar":
        parts.append(str(window.period))
    elif window.kind == "rolling":
        parts.append(f"{window.duration_seconds}s")
    return "-".join(parts)


def budget_settings(policy: BudgetPolicy | None) -> tuple[ManagedSetting, ...]:
    """One amount setting per application budget limit that has a unique, key-shaped name."""

    if policy is None:
        return ()
    names = [budget_limit_name(limit) for limit in policy.limits]
    settings = []
    for name, limit in zip(names, policy.limits, strict=True):
        key = f"budget.{name}.max_estimated_cost"
        if names.count(name) != 1 or not KEY_PATTERN.fullmatch(key) or len(key) > MAX_KEY_LENGTH:
            continue
        amount = _amount(limit.max_estimated_cost)
        settings.append(
            ManagedSetting(
                key=key,
                owner=f"budget:{name}",
                type="number",
                # BudgetLimit needs an amount above 0; the contract's minimum is inclusive.
                # The code's own amount may be smaller still; it stays a valid default.
                minimum=min(_MIN_BUDGET_AMOUNT, amount),
                default=amount,
                applies="restart",
                label=f"Budget {name} ({limit.currency})"[:80],
                description=f"Most the {name} budget may spend, estimated from usage.",
            )
        )
    return tuple(settings)


def _agent_model_key(key: str) -> str | None:
    """The agent whose ``model`` setting ``key`` is, if it is one."""

    if key.startswith("agent.") and key.endswith(".model") and key.count(".") == 2:
        return key.split(".")[1]
    return None


_MIN_BUDGET_AMOUNT = 0.000001
_CONFIG_HISTORY = 32


def _amount(value: Decimal) -> int | float:
    return int(value) if value == value.to_integral_value() else float(value)


def _reason(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        errors = exc.errors(include_url=False, include_input=False)
        text = str(errors[0]["msg"]) if errors else "invalid value"
    else:
        text = str(exc) or "invalid value"
    return text[:MAX_REASON_LENGTH]


def managed_config(
    code: CayuConfig, value_for: Callable[[ConfigSetting], Any]
) -> tuple[CayuConfig, frozenset[str], dict[str, str]]:
    """The code's config with Cloud's values in it, the managed paths, and refusals.

    Each value is applied on its own, so one the config refuses (a limit the
    runtime can't hold) leaves the others in place.
    """

    data = code.model_dump(mode="python", warnings=False)
    config, paths, refused = code, set(), {}
    for item in CONFIG_SETTINGS:
        value = value_for(item)
        if value is None:
            continue
        candidate = deepcopy(data)
        item.write(candidate, value)
        try:
            config = CayuConfig.model_validate(candidate)
        except ValidationError as exc:
            refused[item.setting.key] = _reason(exc)
            continue
        data = candidate
        paths.add(item.path)
    return config, frozenset(paths), refused


def managed_budget_policy(
    policy: BudgetPolicy | None, values: Mapping[str, Any]
) -> tuple[BudgetPolicy | None, dict[str, str]]:
    """The budget policy with Cloud's amounts, and the amounts it refused."""

    if policy is None:
        return None, {}
    settings = {setting.key: setting for setting in budget_settings(policy)}
    limits: list[BudgetLimit] = []
    refused: dict[str, str] = {}
    for limit in policy.limits:
        key = f"budget.{budget_limit_name(limit)}.max_estimated_cost"
        setting = settings.get(key)
        value = None if setting is None else values.get(key)
        if setting is not None and value is not None:
            try:
                setting.validate_value(value)
            except ValueError as exc:
                refused[key] = _reason(exc)
                limits.append(limit)
                continue
            amount = Decimal(str(value))
            if amount > 0:
                limit = limit.model_copy(update={"max_estimated_cost": amount})
            else:
                refused[key] = "must be greater than 0"
        limits.append(limit)
    return BudgetPolicy(limits=tuple(limits)), refused


class ManagedSettingsEntrances:
    """Managed-settings declarations for ``CayuApp`` and how Cloud's values apply."""

    _managed_setting_declarations: tuple[ManagedSetting, ...] = ()
    _managed_settings_controller: ManagedSettings | None = None
    # The code's config and its sources, before any value from Cloud.
    _managed_base: tuple[CayuConfig, Mapping[str, CayuConfigSource]] | None = None
    _managed_code_budget: tuple[BudgetPolicy | None] | None = None
    _managed_refusals: Mapping[str, str] = MappingProxyType({})
    _managed_model_checks: Mapping[tuple[str, str], str | None] = MappingProxyType({})
    # Effective configs this process has used, newest first, so a session admitted
    # under an earlier one can still be matched to its execution profile.
    _managed_config_history: tuple[CayuConfig, ...] = ()

    @property
    def managed_settings_controller(self) -> ManagedSettings | None:
        """The controller reading this Agent's settings from Cayu Cloud, if any."""

        return self._managed_settings_controller

    def _bind_managed_settings(self: CayuApp, controller: ManagedSettings | None) -> None:
        """Read Cloud's values once while the app is built, then follow the controller."""

        if controller is None:
            release = self.release_identity
            controller = ManagedSettings.from_environment(
                release_id=None if release is None else release.id
            )
            if controller is None:
                return
        if type(controller) is not ManagedSettings:
            raise TypeError("managed_settings must be ManagedSettings.")
        self._managed_settings_controller = controller
        self._managed_base = (self._config, self._config_sources)
        self._managed_config_history = (self._config,)
        controller.bind(outcomes=self._managed_outcomes, on_snapshot=self._apply_managed_snapshot)
        controller.read_at_build()
        self._budget_policy = self._managed_budget(self._budget_policy)
        self._apply_managed_snapshot()

    def _managed_values(self: CayuApp, *, restart: bool) -> Mapping[str, Any]:
        controller = self._managed_settings_controller
        if controller is None:
            return {}
        snapshot = controller.build_snapshot if restart else controller.snapshot
        return {} if snapshot is None else snapshot.values

    def _managed_value(self: CayuApp, setting: ManagedSetting) -> Any:
        """Cloud's value for a setting, when it is valid for its descriptor."""

        value = self._managed_values(restart=setting.applies == "restart").get(setting.key)
        try:
            return setting.validate_value(value)
        except ValueError:
            return None

    def _apply_managed_snapshot(self: CayuApp) -> None:
        """Swap in the config new sessions use; running sessions keep what they resolved."""

        base = self._managed_base
        if base is None:
            return
        code, sources = base
        config, paths, refused = managed_config(
            code, lambda item: self._managed_value(item.setting)
        )
        self._config_sources = MappingProxyType(
            {path: ("managed" if path in paths else source) for path, source in sources.items()}
        )
        self._config = config
        self._managed_config_history = (
            config,
            *(item for item in self._managed_config_history if item != config),
        )[:_CONFIG_HISTORY]
        self._managed_refusals = MappingProxyType(
            {
                **{k: v for k, v in self._managed_refusals.items() if not k.startswith("config.")},
                **refused,
            }
        )
        self._managed_model_checks = MappingProxyType({})

    def _with_application_run_defaults(
        self: CayuApp,
        request: _RunConfigurationRequest,
        session: Session | None = None,
        *,
        seeded: bool = False,
        expected: ExecutionProfileIdentity | None = None,
    ) -> _RunConfigurationRequest:
        """Resolve omitted run controls from the application's config.

        A seeded durable subagent request is already committed, so its retry policy
        stays as it is. With an ``expected`` profile (a participant session or a
        queued child admitted earlier), the run controls of the config it was admitted
        under are used when Cloud has changed them since.
        """

        def apply(config: CayuConfig) -> _RunConfigurationRequest:
            return with_application_run_defaults(
                request,
                config,
                self._config_sources["run.max_steps"],
                session,
                freeze_retry_policy=self._managed_settings_controller is not None and not seeded,
            )

        if expected is not None and len(self._managed_config_history) > 1:
            target = next(
                (
                    item
                    for item in expected.components
                    if item.component_class is ExecutionProfileComponentClass.FINALIZATION
                ),
                None,
            )
            for config in self._managed_config_history:
                if target is None:
                    break
                candidate = apply(config)
                if type(candidate) is not RunRequest:
                    break
                material = model_finalization_material(
                    max_steps=candidate.max_steps,
                    limits=candidate.limits,
                    retry_policy=candidate.retry_policy or config.run.retry_policy,
                    tool_completion=candidate.tool_completion,
                )
                if _available_component(target.component_class, target.strength, material) == (
                    target
                ):
                    return candidate
        return apply(self._config)

    def _managed_budget(self: CayuApp, policy: BudgetPolicy | None) -> BudgetPolicy | None:
        """Record the code's budget policy and apply Cloud's restart amounts to it."""

        if self._managed_settings_controller is None:
            return policy
        self._managed_code_budget = (policy,)
        managed, refused = managed_budget_policy(policy, self._managed_values(restart=True))
        self._managed_refusals = MappingProxyType(
            {
                **{k: v for k, v in self._managed_refusals.items() if not k.startswith("budget.")},
                **refused,
            }
        )
        return managed

    def managed_setting_value(self: CayuApp, key: str) -> Any:
        """A declared setting's value: Cloud's when it is set and valid, else the default.

        ``restart`` settings return the value read when the app was built;
        ``next_session`` and ``live`` ones the latest snapshot's. Read a
        ``next_session`` setting when a session starts. A ``secret_ref`` value is
        the name of an environment variable, never its content.
        """

        setting = next((item for item in self.managed_settings() if item.key == key), None)
        if setting is None:
            raise KeyError(f"No managed setting is declared as {key!r}.")
        value = self._managed_value(setting)
        if value is None or self._managed_refusal(setting, value) is not None:
            return setting.default
        return value

    def _managed_refusal(self: CayuApp, setting: ManagedSetting, value: Any) -> str | None:
        """Why this process does not apply a value, or None when it does."""

        try:
            setting.validate_value(value)
        except ValueError as exc:
            return _reason(exc)
        if setting.key in self._managed_refusals:
            return self._managed_refusals[setting.key]
        if setting.type == "secret_ref" and value not in os.environ:
            return "the environment variable is not set on the Agent"
        agent = _agent_model_key(setting.key)
        if agent is not None:
            return self._managed_model_refusal(agent, value)
        return None

    def _managed_model_refusal(self: CayuApp, agent_name: str, model: str) -> str | None:
        if (agent_name, model) in self._managed_model_checks:
            return self._managed_model_checks[(agent_name, model)]
        try:
            spec = self._get_registered_agent(agent_name).spec
        except Exception:
            return "the agent is not registered"
        try:
            # The same model-identity rules as the agent's own model (no blank or padded ids).
            type(spec).model_validate({**spec.model_dump(mode="python"), "model": model})
        except ValidationError as exc:
            return _reason(exc)
        unregistered = "the agent's provider is not registered"
        reason = self._managed_provider_refusal(spec, model, unregistered=unregistered)
        if reason == unregistered:
            return reason
        self._managed_model_checks = MappingProxyType(
            {**self._managed_model_checks, (agent_name, model): reason}
        )
        return reason

    def _managed_provider_refusal(
        self: CayuApp, spec: AgentSpec, model: str, *, unregistered: str | None = None
    ) -> str | None:
        """Why the agent's provider would refuse ``model``, as session preparation routes it.

        ``unregistered`` is the answer when no provider is registered for it yet.
        """

        try:
            provider = (
                self._get_registered_provider(spec.provider_name)
                if spec.provider_name is not None
                else self._route_registered_provider_for_model(model=model)
                or self._get_registered_provider(None)
            )
        except (KeyError, RuntimeError):
            # Not registered yet (KeyError) or no provider at all (RuntimeError).
            return unregistered
        except Exception:
            # Such as a model that matches several providers' patterns.
            return "the agent's provider does not accept this model"
        try:
            provider.provider.preflight_model_target(model=model)
        except Exception:
            return "the agent's provider does not accept this model"
        return None

    def _managed_outcomes(self: CayuApp) -> list[dict[str, Any]]:
        """What this process does with each declared setting, for Cloud's report."""

        current = self._managed_values(restart=False)
        built = self._managed_values(restart=True)
        outcomes = []
        for setting in self.managed_settings():
            value = current.get(setting.key)
            in_effect = built.get(setting.key) if setting.applies == "restart" else value
            if value is None and in_effect is None:
                outcomes.append({"key": setting.key, "outcome": "default"})
                continue
            reason = None if value is None else self._managed_refusal(setting, value)
            if reason is None and in_effect != value:
                reason = "takes effect when the service restarts"
            if reason is None:
                outcomes.append({"key": setting.key, "outcome": "adopted"})
            else:
                outcomes.append({"key": setting.key, "outcome": "refused", "reason": reason})
        return outcomes

    def _managed_agent_model(self: CayuApp, agent_name: str) -> tuple[bool, str | None]:
        """Whether Cloud manages an agent's model for new sessions, and its value now."""

        key = f"agent.{agent_name}.model"
        setting = next(
            (
                item
                for item in self._managed_setting_declarations
                # restart reads the value from the build, next_session the latest.
                if item.key == key
            ),
            None,
        )
        if (
            setting is None
            or self._managed_settings_controller is None
            or (
                self.model_policy is not None
                and any(item.agent_name == agent_name for item in self.model_policy.controllers)
            )
        ):
            return False, None
        value = self._managed_value(setting)
        if value is None or self._managed_refusal(setting, value) is not None:
            return True, None
        return True, value

    def _managed_session_evidence(
        self: CayuApp, agent_name: str, actual: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """The Cloud values a new session starts with, recorded on ``session.started``.

        ``actual`` holds what the session resolved for the settings it applies; a Cloud
        value that differs from it (an explicit request field, a retained model, a
        queued child's earlier value) is left out.
        """

        controller = self._managed_settings_controller
        snapshot = None if controller is None else controller.snapshot
        if snapshot is None:
            return None
        settings = {setting.key: setting for setting in self.managed_settings()}
        values = {}
        for item in self._managed_outcomes():
            key = item["key"]
            if item["outcome"] != "adopted" or not (
                key.startswith(("config.", "budget.", f"agent.{agent_name}."))
            ):
                continue
            value = self._managed_value(settings[key])
            if key in actual and actual[key] != value:
                continue
            values[key] = value
        return {"revision": snapshot.revision, "values": values}

    @_admitted_entrance
    async def start_managed_settings(
        self: CayuApp, *, role: Literal["web", "worker"] = "web"
    ) -> None:
        """Poll Cayu Cloud for this Agent's settings; a no-op outside Cloud.

        ``create_server``, ``mount_cayu`` and ``cayu worker`` call it; ``aclose``
        stops it.
        """

        if self._managed_settings_controller is not None:
            await self._managed_settings_controller.start(role=role)

    async def stop_managed_settings(self: CayuApp) -> None:
        if self._managed_settings_controller is not None:
            await self._managed_settings_controller.close()

    def declare_settings(self: CayuApp, settings: Iterable[ManagedSetting]) -> None:
        """Declare the application's own settings, under the ``app.`` namespace.

        The application reads their values with ``managed_setting_value``.
        """

        self._add_managed_settings(namespaced_settings(settings, prefix="app", owner="app"))

    def managed_settings(self: CayuApp) -> tuple[ManagedSetting, ...]:
        """Every setting this application declares, sorted by key.

        An agent whose model the Gateway's model policy manages leaves its
        ``model`` setting out: the policy owns it.
        """

        policy_model_keys = (
            set()
            if self.model_policy is None
            else {f"agent.{item.agent_name}.model" for item in self.model_policy.controllers}
        )
        declared = tuple(
            setting
            for setting in self._managed_setting_declarations
            if setting.key not in policy_model_keys
        )
        # A budget limit's name comes from its key, which may be a secret value.
        budgets = tuple(
            setting
            for setting in budget_settings(self._managed_code_budget_policy())
            if self.redact_json(setting.descriptor()) == setting.descriptor()
        )
        return tuple(
            sorted(
                (*config_settings(self._managed_code_config()), *budgets, *declared),
                key=lambda item: item.key,
            )
        )

    def managed_settings_schema(self: CayuApp) -> dict[str, Any]:
        """The Release's ``cayu.managed-settings-schema/v1`` document."""

        return managed_settings_schema_document(self.managed_settings())

    def _managed_code_config(self: CayuApp) -> CayuConfig:
        return self._config if self._managed_base is None else self._managed_base[0]

    def _managed_code_budget_policy(self: CayuApp) -> BudgetPolicy | None:
        if self._managed_code_budget is None:
            return self._budget_policy
        return self._managed_code_budget[0]

    def _check_managed_settings(self: CayuApp, declared: tuple[ManagedSetting, ...]) -> None:
        """Refuse declarations that leak a secret or push the schema over its limits."""

        for setting in declared:
            descriptor = setting.descriptor()
            if self.redact_json(descriptor) != descriptor:
                raise ValueError(f"Managed setting {setting.key!r} contains a secret value.")
        managed_settings_schema_document((*self.managed_settings(), *declared))

    def _drop_managed_settings(self: CayuApp, *, owner: str) -> None:
        """Forget the settings of a part that was unregistered."""

        self._managed_setting_declarations = tuple(
            item for item in self._managed_setting_declarations if item.owner != owner
        )

    def _add_managed_settings(self: CayuApp, declared: tuple[ManagedSetting, ...]) -> None:
        if declared:
            self._check_managed_settings(declared)
            self._managed_setting_declarations = (*self._managed_setting_declarations, *declared)

    def _agent_managed_settings(
        self: CayuApp,
        spec: AgentSpec,
        settings: Iterable[ManagedSetting] | None,
        *,
        default_thinking: ThinkingConfig | None,
    ) -> tuple[AgentSpec, tuple[ManagedSetting, ...]]:
        """Check an agent's settings before it registers, with Cloud's restart values
        applied to its spec; add the settings once it has registered."""

        if settings is None:
            return spec, ()
        if isinstance(settings, ManagedSetting | str | bytes):
            raise TypeError("settings must be an iterable of ManagedSetting.")
        items = tuple(settings)
        name = namespace_segment(spec.name, part="Agent")
        # Defaults are the code's: the application's thinking before Cloud's value.
        code_thinking = self._managed_code_config().run.thinking
        thinking = spec.thinking if spec.thinking is not None else code_thinking
        declared = namespaced_settings(
            items,
            prefix=f"agent.{name}",
            owner=f"agent:{name}",
            defaults=agent_setting_defaults(
                [item for item in items if type(item) is ManagedSetting],
                model=spec.model,
                thinking_effort=None if thinking is None else thinking.effort,
                provider_options=spec.provider_options,
            ),
        )
        self._check_managed_settings(declared)
        return self._agent_spec_at_restart(spec, declared, default_thinking), declared

    def _agent_spec_at_restart(
        self: CayuApp,
        spec: AgentSpec,
        declared: tuple[ManagedSetting, ...],
        default_thinking: ThinkingConfig | None,
    ) -> AgentSpec:
        refused: dict[str, str] = {}
        for setting in declared:
            relative = setting.key.removeprefix(f"agent.{spec.name}.")
            value = self._managed_value(setting) if setting.applies == "restart" else None
            # The model, restart or not, is chosen in session preparation (see
            # _managed_agent_model), where its provider is registered and preflights it.
            if value is None or not (
                relative == "thinking.effort" or relative.startswith("provider_options.")
            ):
                continue
            update: dict[str, Any]
            if relative == "thinking.effort":
                thinking = spec.thinking or default_thinking or ThinkingConfig()
                update = {"thinking": thinking.model_copy(update={"effort": value})}
            else:
                name = relative.removeprefix("provider_options.")
                update = {"provider_options": {**spec.provider_options, name: value}}
            try:
                spec = type(spec).model_validate({**spec.model_dump(mode="python"), **update})
            except ValidationError as exc:
                refused[setting.key] = _reason(exc)
        if refused:
            self._managed_refusals = MappingProxyType({**self._managed_refusals, **refused})
        return spec

    def _part_managed_settings(
        self: CayuApp, settings: Iterable[ManagedSetting] | None, *, part: str, name: str
    ) -> tuple[ManagedSetting, ...]:
        """Check a provider's or environment factory's settings before it is registered."""

        if settings is None:
            return ()
        segment = namespace_segment(name, part=part.capitalize())
        declared = namespaced_settings(settings, prefix=f"{part}.{segment}", owner=f"{part}:{name}")
        self._check_managed_settings(declared)
        return declared
