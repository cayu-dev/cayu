"""Settings an application's parts declare so Cayu Cloud can manage them.

A part declares its settings where it is registered: ``register_agent(...,
settings=...)``, ``register_provider(..., settings=...)``,
``register_environment_factory(..., settings=...)`` and
``CayuApp.declare_settings`` for the application itself. Chosen ``CayuConfig``
fields and keyed application budget limits are declared by the framework. The
code's value is always the default; a value set in Cloud overrides it.

The assembled list is the Release's ``cayu.managed-settings-schema/v1``
document (see ``docs/managed-settings.md``). It is part of ``describe_app`` and
is published to Cloud in the agent registry registration when the Release
starts.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from math import isfinite
from typing import Annotated, Any, Literal, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    WithJsonSchema,
    model_validator,
)

from cayu.context.thinking import ThinkingEffort

MANAGED_SETTINGS_SCHEMA_CONTRACT = "cayu.managed-settings-schema/v1"
# Limits of the published schema (contract section 2). Assembly refuses to exceed them.
MAX_MANAGED_SETTINGS = 500
MAX_MANAGED_SETTINGS_SCHEMA_BYTES = 256 * 1024
MAX_KEY_LENGTH = 128
MAX_OWNER_LENGTH = 128
MAX_LABEL_LENGTH = 80
MAX_DESCRIPTION_LENGTH = 500
MAX_STRING_LENGTH = 4096
MAX_CHOICES = 100

ManagedSettingType = Literal["bool", "int", "number", "string", "enum", "secret_ref"]
ManagedSettingApplies = Literal["restart", "next_session", "live"]
ManagedSettingValue = Annotated[
    StrictBool | StrictInt | StrictFloat | StrictStr | None,
    # Spelled out: the generated order of union members varies across Python versions.
    WithJsonSchema({"type": ["boolean", "integer", "number", "string", "null"]}),
]
_Bound = Annotated[StrictInt | StrictFloat | None, WithJsonSchema({"type": ["number", "null"]})]

KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_-]+)+$")
# A part's own key, before the part's namespace is put in front of it.
_RELATIVE_KEY_PATTERN = re.compile(r"^[a-z0-9_-]+(\.[a-z0-9_-]+)*$")
# One namespace segment: an agent, provider or environment name, or a budget key.
_SEGMENT_PATTERN = re.compile(r"^[a-z0-9_-]+$")
SECRET_REF_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")
_THINKING_EFFORTS: tuple[str, ...] = get_args(ThinkingEffort)
_THINKING_EFFORT_LABELS = {"xhigh": "Extra high"}


class ManagedSettingChoice(BaseModel):
    """One allowed value of an ``enum`` setting."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    value: StrictStr = Field(min_length=1, max_length=256)
    label: StrictStr = Field(min_length=1, max_length=MAX_LABEL_LENGTH)


def _absent(value: object) -> bool:
    return value is None


class ManagedSetting(BaseModel):
    """One setting descriptor (``docs/managed-settings.md``, section 1).

    Parts declare settings with a key relative to themselves (``"max_steps"``);
    registration puts the part's namespace in front (``"agent.reviewer.max_steps"``)
    and sets ``owner``. ``default`` is the code's value, or ``None`` when unset.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    key: StrictStr = Field(min_length=1, max_length=MAX_KEY_LENGTH)
    type: ManagedSettingType
    label: StrictStr = Field(min_length=1, max_length=MAX_LABEL_LENGTH)
    applies: ManagedSettingApplies = "next_session"
    default: ManagedSettingValue = None
    owner: StrictStr | None = Field(
        default=None, min_length=1, max_length=MAX_OWNER_LENGTH, exclude_if=_absent
    )
    description: StrictStr | None = Field(
        default=None, max_length=MAX_DESCRIPTION_LENGTH, exclude_if=_absent
    )
    minimum: _Bound = Field(default=None, exclude_if=_absent)
    maximum: _Bound = Field(default=None, exclude_if=_absent)
    max_length: StrictInt | None = Field(
        default=None, ge=1, le=MAX_STRING_LENGTH, exclude_if=_absent
    )
    choices: tuple[ManagedSettingChoice, ...] | None = Field(default=None, exclude_if=_absent)

    @model_validator(mode="after")
    def _validate_shape(self) -> ManagedSetting:
        if not (KEY_PATTERN.fullmatch(self.key) or _RELATIVE_KEY_PATTERN.fullmatch(self.key)):
            raise ValueError(
                "key must be dotted lowercase segments of letters, digits, '_' and '-'."
            )
        numeric = self.type in {"int", "number"}
        if not numeric and (self.minimum is not None or self.maximum is not None):
            raise ValueError("minimum and maximum apply only to int and number settings.")
        if self.type == "int" and any(
            type(bound) is not int for bound in (self.minimum, self.maximum) if bound is not None
        ):
            raise ValueError("int bounds must be integers.")
        for bound in (self.minimum, self.maximum):
            if type(bound) is float and not isfinite(bound):
                raise ValueError("bounds must be finite.")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum must not be greater than maximum.")
        if (self.type == "string") != (self.max_length is not None):
            raise ValueError("max_length is required for string settings and only for them.")
        if (self.type == "enum") != (self.choices is not None):
            raise ValueError("choices are required for enum settings and only for them.")
        if self.choices is not None:
            if not 1 <= len(self.choices) <= MAX_CHOICES:
                raise ValueError(f"enum settings need 1 to {MAX_CHOICES} choices.")
            values = [choice.value for choice in self.choices]
            if len(set(values)) != len(values):
                raise ValueError("enum choices must have distinct values.")
        if self.default is not None:
            self.validate_value(self.default)
        return self

    def validate_value(self, value: object) -> ManagedSettingValue:
        """Return ``value`` if it is valid for this setting, else raise ``ValueError``."""

        if value is None:
            return None
        if self.type == "bool":
            if type(value) is not bool:
                raise ValueError("expected true or false")
            return value
        if self.type == "int":
            if type(value) is not int:
                raise ValueError("expected an integer")
            return self._within_bounds(value)
        if self.type == "number":
            if not isinstance(value, int | float) or type(value) is bool or not isfinite(value):
                raise ValueError("expected a finite number")
            return self._within_bounds(value)
        if type(value) is not str:
            raise ValueError("expected a string")
        if self.type == "string":
            assert self.max_length is not None
            if len(value) > self.max_length:
                raise ValueError(f"longer than {self.max_length} characters")
            return value
        if self.type == "enum":
            assert self.choices is not None
            if value not in {choice.value for choice in self.choices}:
                raise ValueError("not one of the allowed choices")
            return value
        if not SECRET_REF_PATTERN.fullmatch(value):
            raise ValueError("expected the name of an environment variable")
        return value

    def _within_bounds(self, value: int | float) -> int | float:
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"less than the minimum {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"greater than the maximum {self.maximum}")
        return value

    def descriptor(self) -> dict[str, Any]:
        """Return the contract's JSON descriptor."""

        return self.model_dump(mode="json")


def _choices(values: Iterable[str | tuple[str, str] | ManagedSettingChoice]) -> tuple:
    result = []
    for item in values:
        if isinstance(item, ManagedSettingChoice):
            result.append(item)
        elif isinstance(item, tuple):
            result.append(ManagedSettingChoice(value=item[0], label=item[1]))
        else:
            result.append(ManagedSettingChoice(value=item, label=item))
    return tuple(result)


def model_setting(
    *,
    label: str = "Model",
    description: str | None = "The model new sessions use.",
    choices: Iterable[str | tuple[str, str] | ManagedSettingChoice] | None = None,
) -> ManagedSetting:
    """Declare an agent's model; pass it to ``register_agent(settings=...)``.

    New sessions use the value set in Cloud; running and resumed sessions keep
    the model they started with. The default is the agent's own model. Agents
    whose model the Gateway's model policy manages leave it out of the schema.
    """

    if choices is None:
        return ManagedSetting(
            key="model", type="string", max_length=256, label=label, description=description
        )
    return ManagedSetting(
        key="model", type="enum", choices=_choices(choices), label=label, description=description
    )


def thinking_effort_choices(
    efforts: Iterable[str] = _THINKING_EFFORTS,
) -> tuple[ManagedSettingChoice, ...]:
    """Choices for a thinking-effort setting, labelled for people."""

    return tuple(
        ManagedSettingChoice(
            value=effort, label=_THINKING_EFFORT_LABELS.get(effort, effort.capitalize())
        )
        for effort in efforts
    )


def thinking_effort_setting(
    *,
    label: str = "Thinking effort",
    description: str | None = "How much the model reasons before it answers.",
    choices: Iterable[str] = _THINKING_EFFORTS,
) -> ManagedSetting:
    """Declare an agent's thinking effort; it applies when the service restarts."""

    return ManagedSetting(
        key="thinking.effort",
        type="enum",
        choices=thinking_effort_choices(choices),
        applies="restart",
        label=label,
        description=description,
    )


def provider_option_setting(
    name: str,
    *,
    type: Literal["bool", "int", "number", "string", "enum"],
    label: str,
    description: str | None = None,
    minimum: int | float | None = None,
    maximum: int | float | None = None,
    max_length: int | None = None,
    choices: Iterable[str | tuple[str, str] | ManagedSettingChoice] | None = None,
) -> ManagedSetting:
    """Declare one of an agent's provider options; it applies when the service restarts."""

    return ManagedSetting(
        key=f"provider_options.{name}",
        type=type,
        applies="restart",
        label=label,
        description=description,
        minimum=minimum,
        maximum=maximum,
        max_length=max_length,
        choices=None if choices is None else _choices(choices),
    )


def namespace_segment(name: str, *, part: str) -> str:
    """Return ``name`` if it can be one segment of a setting key."""

    if not _SEGMENT_PATTERN.fullmatch(name):
        raise ValueError(
            f"{part} {name!r} cannot declare managed settings: its name must use only "
            "lowercase letters, digits, '_' and '-'."
        )
    return name


def namespaced_settings(
    settings: Iterable[ManagedSetting] | None,
    *,
    prefix: str,
    owner: str,
    defaults: Mapping[str, Any] | None = None,
) -> tuple[ManagedSetting, ...]:
    """Return a part's settings with full keys, its owner and its code defaults."""

    if settings is None:
        return ()
    if isinstance(settings, ManagedSetting | str | bytes):
        raise TypeError("settings must be an iterable of ManagedSetting.")
    result: list[ManagedSetting] = []
    for setting in settings:
        if type(setting) is not ManagedSetting:
            raise TypeError("settings must contain only ManagedSetting values.")
        if not _RELATIVE_KEY_PATTERN.fullmatch(setting.key):
            raise ValueError(f"Setting key {setting.key!r} must be relative to its part.")
        if setting.owner is not None and setting.owner != owner:
            raise ValueError(
                f"Setting {setting.key!r} belongs to {owner!r}, not {setting.owner!r}."
            )
        update: dict[str, Any] = {"key": f"{prefix}.{setting.key}", "owner": owner}
        if defaults is not None and setting.key in defaults:
            update["default"] = defaults[setting.key]
        data = {**setting.model_dump(mode="python"), **update}
        try:
            full = ManagedSetting.model_validate(data)
        except ValueError as exc:
            raise ValueError(f"Setting {update['key']!r} is invalid: {exc}") from None
        if not KEY_PATTERN.fullmatch(full.key):
            raise ValueError(f"Setting key {full.key!r} is not a valid managed setting key.")
        result.append(full)
    keys = [item.key for item in result]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{owner} declares the same setting more than once.")
    return tuple(result)


def agent_setting_defaults(
    settings: Sequence[ManagedSetting],
    *,
    model: str,
    thinking_effort: str | None,
    provider_options: Mapping[str, Any],
) -> dict[str, Any]:
    """Check an agent's settings that Cayu applies, and return their code defaults."""

    defaults: dict[str, Any] = {}
    for setting in settings:
        key = setting.key
        if key == "model":
            if setting.type not in {"string", "enum"} or setting.applies == "live":
                raise ValueError(
                    "An agent's 'model' setting is a string or enum that applies at "
                    "next_session or restart."
                )
            defaults[key] = model
        elif key.startswith("thinking."):
            if key != "thinking.effort" or setting.type != "enum":
                raise ValueError("The only thinking setting is the enum 'thinking.effort'.")
            assert setting.choices is not None
            if not {choice.value for choice in setting.choices} <= set(_THINKING_EFFORTS):
                raise ValueError(
                    "thinking.effort choices must be among: " + ", ".join(_THINKING_EFFORTS)
                )
            defaults[key] = thinking_effort
        elif key.startswith("provider_options."):
            name = key.removeprefix("provider_options.")
            if "." in name or setting.type == "secret_ref":
                raise ValueError(f"{key!r} must name one scalar provider option.")
            value = provider_options.get(name)
            defaults[key] = value if type(value) in {bool, int, float, str} else None
        else:
            continue
        if key != "model" and setting.applies != "restart":
            # Execution profiles fingerprint them, so a resumed session would no
            # longer match its profile if they changed between sessions.
            raise ValueError(f"An agent's {key!r} setting applies only at restart.")
    return defaults


def canonical_settings_json(settings: Sequence[Mapping[str, Any]]) -> bytes:
    """Canonical JSON of a ``settings`` list: sorted keys, no spaces, UTF-8."""

    return json.dumps(
        list(settings), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def settings_digest(settings: Sequence[Mapping[str, Any]]) -> str:
    """The schema digest: SHA-256 of the canonical JSON of ``settings``."""

    return "sha256:" + hashlib.sha256(canonical_settings_json(settings)).hexdigest()


class ManagedSettingsLimitExceeded(ValueError):
    """The assembled schema would exceed the contract's limits."""


def managed_settings_schema_document(settings: Iterable[ManagedSetting]) -> dict[str, Any]:
    """Assemble the sorted ``cayu.managed-settings-schema/v1`` document, within its limits."""

    ordered = sorted(settings, key=lambda item: item.key)
    counts = Counter(item.key for item in ordered)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError("Managed settings are declared more than once: " + ", ".join(duplicates))
    missing_owner = [item.key for item in ordered if item.owner is None]
    if missing_owner:
        raise ValueError("Managed settings have no owner: " + ", ".join(missing_owner))
    if len(ordered) > MAX_MANAGED_SETTINGS:
        raise ManagedSettingsLimitExceeded(
            f"The application declares {len(ordered)} managed settings; "
            f"the limit is {MAX_MANAGED_SETTINGS}."
        )
    descriptors = [item.descriptor() for item in ordered]
    document = {
        "contract": MANAGED_SETTINGS_SCHEMA_CONTRACT,
        "digest": settings_digest(descriptors),
        "settings": descriptors,
    }
    size = len(json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    if size > MAX_MANAGED_SETTINGS_SCHEMA_BYTES:
        raise ManagedSettingsLimitExceeded(
            f"The managed settings schema is {size} bytes; "
            f"the limit is {MAX_MANAGED_SETTINGS_SCHEMA_BYTES}."
        )
    return document


__all__ = [
    "MANAGED_SETTINGS_SCHEMA_CONTRACT",
    "MAX_MANAGED_SETTINGS",
    "MAX_MANAGED_SETTINGS_SCHEMA_BYTES",
    "ManagedSetting",
    "ManagedSettingApplies",
    "ManagedSettingChoice",
    "ManagedSettingType",
    "ManagedSettingValue",
    "ManagedSettingsLimitExceeded",
    "managed_settings_schema_document",
    "model_setting",
    "provider_option_setting",
    "settings_digest",
    "thinking_effort_choices",
    "thinking_effort_setting",
]
