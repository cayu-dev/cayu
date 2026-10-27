"""CayuApp's managed-settings declarations and the schema they assemble into.

``CayuApp`` inherits these; they live here so the application module stays
within its size limit. Parts declare settings where they are registered; this
module only collects the declarations, adds the framework's own (chosen
``CayuConfig`` fields and keyed application budget limits) and assembles the
Release's schema.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.configuration import MAX_STEPS, CayuConfig
from cayu.managed_settings import (
    KEY_PATTERN,
    MAX_KEY_LENGTH,
    ManagedSetting,
    agent_setting_defaults,
    managed_settings_schema_document,
    namespace_segment,
    namespaced_settings,
    thinking_effort_choices,
)

if TYPE_CHECKING:
    from cayu.agents import AgentSpec
    from cayu.applications import CayuApp
    from cayu.budgets.base import BudgetLimit, BudgetPolicy
    from cayu.context.thinking import ThinkingConfig


@dataclass(frozen=True, slots=True)
class ConfigSetting:
    """One ``CayuConfig`` field promoted to a managed setting."""

    setting: ManagedSetting
    # The field's path in CAYU_CONFIG_FIELD_OWNERS, whose source becomes "managed".
    path: str
    read: Callable[[CayuConfig], Any]


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


_MIN_BUDGET_AMOUNT = 0.000001


def _amount(value: Decimal) -> int | float:
    return int(value) if value == value.to_integral_value() else float(value)


class ManagedSettingsEntrances:
    """Managed-settings declarations for ``CayuApp``."""

    _managed_setting_declarations: tuple[ManagedSetting, ...] = ()

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
        return self._config

    def _managed_code_budget_policy(self: CayuApp) -> BudgetPolicy | None:
        return self._budget_policy

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
    ) -> tuple[ManagedSetting, ...]:
        """Check an agent's settings before it is registered; add them afterwards."""

        if settings is None:
            return ()
        if isinstance(settings, ManagedSetting | str | bytes):
            raise TypeError("settings must be an iterable of ManagedSetting.")
        items = tuple(settings)
        name = namespace_segment(spec.name, part="Agent")
        thinking = spec.thinking if spec.thinking is not None else default_thinking
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
        return declared

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
