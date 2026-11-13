"""Decide a tool call from a managed setting's value.

``SettingToolPolicy`` reads one declared managed setting and maps its value to a
decision::

    SettingToolPolicy(
        app,                    # a ManagedSettingReader; CayuApp is one
        "app.merge_modes",
        {"auto": ToolPolicyDecision.ALLOW,
         "approve": ToolPolicyDecision.REQUIRE_APPROVAL,
         "report": ToolPolicyDecision.DENY},
        key_argument="repository",
    )

The setting is an ``enum``, or a map setting (``map_setting``) whose values are
choices; for a map, ``key_argument`` names the call argument whose value selects
the entry. The value is read as the setting's ``applies`` says: a ``live``
setting when each call is authorized, a ``next_session`` one when the session
starts. A session keeps the ``next_session`` values it started with in its
metadata (``cayu:session_settings``), as it keeps its model, so its decisions
don't change while it runs or when it resumes; a later value applies to new
sessions. A session with no recorded value reads it when the call is
authorized.

``otherwise`` (DENY or REQUIRE_APPROVAL, never ALLOW) decides when there is
nothing to map: the setting isn't declared or isn't one the policy can map (see
``problem``), Cloud or the environment gave a value the application refused,
the argument is missing or isn't a string, or the map has no entry for it. A
refused value never falls back to the declared default; an unset one does.
Lookup is exact. The result's metadata records the reading under
``cayu:setting``. Route the policy to the tools it decides with
``ToolPolicyRouter``.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

from cayu._validation import require_clean_nonblank
from cayu.session_metadata import SESSION_SETTINGS_METADATA_KEY
from cayu.tools.policy import (
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)

if TYPE_CHECKING:
    from cayu.managed_settings import ManagedSetting, ManagedSettingReading

SETTING_POLICY_METADATA_KEY = "cayu:setting"


class ManagedSettingReader(Protocol):
    """Declares managed settings and reads one's current value and where it came from."""

    def managed_settings(self) -> tuple[ManagedSetting, ...]: ...

    def read_managed_setting(self, key: str) -> ManagedSettingReading: ...


def _decision(value: object, field_name: str) -> ToolPolicyDecision:
    try:
        return ToolPolicyDecision(value)
    except ValueError:
        raise ValueError(f"{field_name} must be a ToolPolicyDecision.") from None


class SettingToolPolicy(ToolPolicy):
    """Map a managed setting's value (or one entry of a map setting) to a decision."""

    def __init__(
        self,
        settings: ManagedSettingReader,
        setting: str,
        decisions: Mapping[str, ToolPolicyDecision | str],
        *,
        key_argument: str | None = None,
        otherwise: ToolPolicyDecision | str = ToolPolicyDecision.DENY,
    ) -> None:
        if not callable(getattr(settings, "read_managed_setting", None)) or not callable(
            getattr(settings, "managed_settings", None)
        ):
            raise TypeError(
                "settings must have managed_settings() and read_managed_setting(key), as "
                "CayuApp does."
            )
        if not isinstance(decisions, Mapping) or not decisions:
            raise ValueError("decisions must map at least one setting value to a decision.")
        copied: dict[str, ToolPolicyDecision] = {}
        for value, decision in decisions.items():
            if type(value) is not str:
                raise TypeError("decisions must be keyed by the setting's string values.")
            copied[value] = _decision(decision, f"decisions[{value!r}]")
        fallback = _decision(otherwise, "otherwise")
        if fallback is ToolPolicyDecision.ALLOW:
            raise ValueError("otherwise must be DENY or REQUIRE_APPROVAL, never ALLOW.")
        self._settings = settings
        self._setting = require_clean_nonblank(setting, "setting")
        self._decisions = MappingProxyType(copied)
        self._key_argument = (
            None if key_argument is None else require_clean_nonblank(key_argument, "key_argument")
        )
        self._otherwise = fallback

    @property
    def setting(self) -> str:
        return self._setting

    @property
    def decisions(self) -> Mapping[str, ToolPolicyDecision]:
        return self._decisions

    @property
    def key_argument(self) -> str | None:
        return self._key_argument

    @property
    def otherwise(self) -> ToolPolicyDecision:
        return self._otherwise

    def _execution_profile_material(self) -> dict[str, object]:
        # The live value is data, not configuration: it changes without a new profile.
        return {
            "setting": self._setting,
            "decisions": {
                value: decision.value for value, decision in sorted(self._decisions.items())
            },
            "key_argument": self._key_argument,
            "otherwise": self._otherwise.value,
        }

    def possible_decisions(self) -> frozenset[ToolPolicyDecision]:
        return frozenset((*self._decisions.values(), self._otherwise))

    @property
    def settings(self) -> ManagedSettingReader:
        return self._settings

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        record: dict[str, Any] = {"key": self._setting}
        declared = next(
            (item for item in self._settings.managed_settings() if item.key == self._setting),
            None,
        )
        if declared is None:
            return self._fallback(record, f"{self._setting} is not a declared setting")
        if self._declaration_problem(declared) is not None:
            # cayu check reports it as TOOL_POLICY_SETTING_INVALID; never allow meanwhile.
            return self._fallback(record, f"the policy can't map {self._setting} as declared")
        reading = (
            session_setting_reading(request.session.metadata, self._setting)
            if declared.applies == "next_session"
            else None
        )
        if reading is None:
            try:
                reading = self._settings.read_managed_setting(self._setting)
            except KeyError:
                return self._fallback(record, f"{self._setting} is not a declared setting")
        record["source"] = reading.source
        if reading.refused is not None:
            # A refused value never falls back to the declared default.
            return self._fallback(record, f"{self._setting} has a value that was refused")
        value = reading.value
        where = ""
        if self._key_argument is not None:
            entry = request.arguments.get(self._key_argument)
            if type(entry) is not str:
                return self._fallback(
                    record, f"the call has no string {self._key_argument} to look up"
                )
            record["entry"] = entry
            where = f" for {entry}"
            value = value.get(entry) if type(value) is dict else None
            if value is None:
                return self._fallback(record, f"{self._setting} has no entry for {entry}")
        if type(value) is not str or value not in self._decisions:
            return self._fallback(record, f"{self._setting} has no value this policy maps")
        record["value"] = value
        return ToolPolicyResult(
            decision=self._decisions[value],
            reason=f"{self._setting} is {value}{where}.",
            metadata={SETTING_POLICY_METADATA_KEY: record},
        )

    def _fallback(self, record: dict[str, Any], why: str) -> ToolPolicyResult:
        return ToolPolicyResult(
            decision=self._otherwise,
            reason=f"{why[0].upper()}{why[1:]}, so the policy's fallback applies.",
            metadata={SETTING_POLICY_METADATA_KEY: {**record, "value": None}},
        )

    def problem(
        self, declared: ManagedSetting | None, input_schema: Mapping[str, Any]
    ) -> str | None:
        """Why this policy can't decide calls to a tool with ``input_schema``, or None.

        ``declared`` is the application's declaration of the setting, or None when
        it declares none. ``cayu check`` reports the answer as
        ``TOOL_POLICY_SETTING_INVALID``.
        """

        if declared is None:
            return f"{self._setting} is not a declared managed setting."
        problem = self._declaration_problem(declared)
        if problem is not None:
            return problem
        if self._key_argument is not None:
            properties = input_schema.get("properties")
            argument = (
                properties.get(self._key_argument) if isinstance(properties, Mapping) else None
            )
            if not isinstance(argument, Mapping) or argument.get("type") != "string":
                return f"key_argument {self._key_argument} is not a string property of the tool."
        return None

    def _declaration_problem(self, declared: ManagedSetting) -> str | None:
        """Why the declared setting doesn't fit ``decisions`` and ``key_argument``, or None."""

        choices = setting_choices(declared)
        if choices is None:
            return (
                f"{self._setting} must be an enum setting, or a map setting whose values are "
                "choices."
            )
        if (declared.format == "map") != (self._key_argument is not None):
            return (
                f"{self._setting} is a map setting, so key_argument must name the argument "
                "that selects its entry."
                if declared.format == "map"
                else f"{self._setting} is an enum setting, so it takes no key_argument."
            )
        unmapped = sorted(choices - set(self._decisions))
        extra = sorted(set(self._decisions) - choices)
        if unmapped or extra:
            parts = []
            if unmapped:
                parts.append(f"unmapped choices: {', '.join(unmapped)}")
            if extra:
                parts.append(f"mapped values that aren't choices: {', '.join(extra)}")
            return f"decisions must map {self._setting}'s choices exactly ({'; '.join(parts)})."
        return None

    def default_allows(self, declared: ManagedSetting) -> bool:
        """Whether, with nothing configured, the declared default lets a call run unapproved."""

        default = declared.read_value(declared.default)
        values = default.values() if type(default) is dict else (default,)
        return any(self._decisions.get(value) is ToolPolicyDecision.ALLOW for value in values)

    def allowing_values(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                value
                for value, decision in self._decisions.items()
                if decision is ToolPolicyDecision.ALLOW
            )
        )


def session_setting_record(reading: ManagedSettingReading) -> dict[str, Any]:
    """How a session records a ``next_session`` setting it started with."""

    return {"value": reading.value, "source": reading.source, "refused": reading.refused}


def session_setting_reading(metadata: Mapping[str, Any], key: str) -> ManagedSettingReading | None:
    """The reading of ``key`` a session recorded when it started; None when it has none."""

    from cayu.managed_settings import ManagedSettingReading

    recorded = metadata.get(SESSION_SETTINGS_METADATA_KEY)
    item = recorded.get(key) if isinstance(recorded, Mapping) else None
    if not isinstance(item, Mapping) or item.get("source") not in {
        "cloud",
        "environment",
        "default",
    }:
        return None
    refused = item.get("refused")
    return ManagedSettingReading(
        key=key,
        value=item.get("value"),
        source=item["source"],
        refused=refused if type(refused) is str else None,
    )


def setting_choices(declared: ManagedSetting) -> frozenset[str] | None:
    """The values a policy maps: an enum's choices, or a map setting's value choices."""

    if declared.type == "enum" and declared.choices is not None:
        return frozenset(choice.value for choice in declared.choices)
    if declared.format == "map" and declared.value_choices is not None:
        return frozenset(choice.value for choice in declared.value_choices)
    return None


__all__ = ["ManagedSettingReader", "SettingToolPolicy"]
