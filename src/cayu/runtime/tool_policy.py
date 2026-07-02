from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank
from cayu.core.agents import AgentSpec
from cayu.runtime.sessions import Session


class ToolPolicyDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


ANY_TAINT_LABEL = "*"
TAINT_LABELS_METADATA_KEY = "cayu:taint_labels"
# Set to True in a ToolPolicyRequest's metadata when a call is being re-authorized because a
# before_tool_call hook modified the arguments. Stateful policies (rate limiters, counters, audit
# sinks) can re-check the limit but skip re-incrementing so a hook-modified call counts once.
TOOL_POLICY_REAUTHORIZATION_METADATA_KEY = "cayu:tool_policy_reauthorization"


class ToolPolicyRequest(BaseModel):
    """Input passed to a tool policy before a registered tool executes."""

    model_config = ConfigDict(extra="forbid")

    session: Session
    agent: AgentSpec
    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    environment_name: str | None = None
    workspace_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tool_name", "tool_call_id")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("environment_name", "workspace_id")
    @classmethod
    def validate_optional_names(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("arguments", "metadata", mode="before")
    @classmethod
    def copy_json_fields(cls, value: dict[str, Any], info) -> dict[str, Any]:
        return copy_json_value(value, info.field_name)


class ToolPolicyResult(BaseModel):
    """Authorization decision for one tool call."""

    model_config = ConfigDict(extra="forbid")

    decision: ToolPolicyDecision
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reason")
    @classmethod
    def validate_optional_reason(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, info.field_name)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")


class ToolPolicy(ABC):
    """Authorizes registered tool calls before execution."""

    @abstractmethod
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        """Return whether this tool call may execute."""


class AllowAllToolPolicy(ToolPolicy):
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


class StaticToolPolicy(ToolPolicy):
    """Simple allow/deny policy for static tool scopes.

    Deny rules win over allow rules. When `allow` is omitted, all tools are
    allowed except explicitly denied tools. When `allow` is provided, only those
    tools are allowed.
    """

    def __init__(
        self,
        *,
        allow: Iterable[str] | None = None,
        deny: Iterable[str] | None = None,
    ) -> None:
        self.allow = _copy_tool_name_set(allow, "allow") if allow is not None else None
        self.deny = _copy_tool_name_set(deny, "deny") if deny is not None else frozenset()

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        if request.tool_name in self.deny:
            return ToolPolicyResult(
                decision=ToolPolicyDecision.DENY,
                reason=f"Tool denied by policy: {request.tool_name}",
            )

        if self.allow is not None and request.tool_name not in self.allow:
            return ToolPolicyResult(
                decision=ToolPolicyDecision.DENY,
                reason=f"Tool not allowed by policy: {request.tool_name}",
            )

        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


class ParameterRule(ABC):
    """Validates one argument path in a tool call.

    A path can be a top-level argument name such as ``"to"`` or a dotted path
    such as ``"request.url"`` for nested JSON objects.
    """

    @property
    @abstractmethod
    def parameter(self) -> str:
        """Human-readable argument path this rule validates."""

    @abstractmethod
    def check(self, arguments: dict[str, Any]) -> str | None:
        """Return a violation reason, or ``None`` when the arguments satisfy the rule."""


class RequiredFieldRule(ParameterRule):
    """Require an argument path to be present and non-empty."""

    def __init__(self, parameter: str) -> None:
        self._path = _copy_parameter_path(parameter)
        self._parameter = ".".join(self._path)

    @property
    def parameter(self) -> str:
        return self._parameter

    def check(self, arguments: dict[str, Any]) -> str | None:
        value = _get_argument_path(arguments, self._path)
        if value is _MISSING:
            return f"Required parameter '{self.parameter}' is missing."
        if _is_empty_parameter_value(value):
            return f"Required parameter '{self.parameter}' is empty."
        return None


class AllowlistRule(ParameterRule):
    """Require an argument value to match one of the explicitly allowed strings."""

    def __init__(self, parameter: str, *, values: Iterable[str]) -> None:
        self._path = _copy_parameter_path(parameter)
        self._parameter = ".".join(self._path)
        self.values = _copy_nonempty_string_set(values, "values")
        if not self.values:
            raise ValueError("AllowlistRule values cannot be empty.")

    @property
    def parameter(self) -> str:
        return self._parameter

    def check(self, arguments: dict[str, Any]) -> str | None:
        value = _get_argument_path(arguments, self._path)
        if value is _MISSING:
            return None
        if type(value) is not str:
            return f"Parameter '{self.parameter}' must be a string for allowlist validation."
        if value not in self.values:
            return f"Parameter '{self.parameter}' value is not allowed."
        return None


class DenyPatternRule(ParameterRule):
    """Reject a string argument when it matches one of the denied regex patterns."""

    def __init__(self, parameter: str, *, patterns: Iterable[str]) -> None:
        self._path = _copy_parameter_path(parameter)
        self._parameter = ".".join(self._path)
        pattern_list = _copy_nonempty_string_list(patterns, "patterns")
        if not pattern_list:
            raise ValueError("DenyPatternRule patterns cannot be empty.")
        self.patterns = tuple(pattern_list)
        self._compiled_patterns = tuple(re.compile(pattern) for pattern in pattern_list)

    @property
    def parameter(self) -> str:
        return self._parameter

    def check(self, arguments: dict[str, Any]) -> str | None:
        value = _get_argument_path(arguments, self._path)
        if value is _MISSING:
            return None
        if type(value) is not str:
            return f"Parameter '{self.parameter}' must be a string for pattern validation."
        for pattern in self._compiled_patterns:
            if pattern.search(value):
                return f"Parameter '{self.parameter}' matches a denied pattern."
        return None


class ParameterConstrainedToolPolicy(ToolPolicy):
    """Validate tool-call arguments with per-tool parameter rules.

    Tools not present in ``rules`` are allowed. For constrained tools, the first
    violated rule returns ``decision`` with a structured metadata payload. Deny
    is the default because argument constraints are most often used as a hard
    safety boundary, but requiring approval is also supported.
    """

    def __init__(
        self,
        rules: dict[str, Iterable[ParameterRule]],
        *,
        decision: ToolPolicyDecision = ToolPolicyDecision.DENY,
    ) -> None:
        if type(rules) is not dict:
            raise TypeError("ParameterConstrainedToolPolicy rules must be a dict.")
        decision = ToolPolicyDecision(decision)
        if decision == ToolPolicyDecision.ALLOW:
            raise ValueError("ParameterConstrainedToolPolicy decision cannot be ALLOW.")
        self.decision = decision
        self.rules: dict[str, tuple[ParameterRule, ...]] = {}
        for tool_name, tool_rules in rules.items():
            name = require_clean_nonblank(tool_name, "tool_name")
            self.rules[name] = _copy_parameter_rules(tool_rules, f"rules[{name!r}]")

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        rules = self.rules.get(request.tool_name)
        if rules is None:
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)
        for index, rule in enumerate(rules):
            violation = rule.check(request.arguments)
            if violation is not None:
                return ToolPolicyResult(
                    decision=self.decision,
                    reason=violation,
                    metadata={
                        "policy": "parameter_constrained",
                        "tool_name": request.tool_name,
                        "parameter": rule.parameter,
                        "rule": type(rule).__name__,
                        "rule_index": index,
                    },
                )
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


class TaintAwareToolPolicy(ToolPolicy):
    """Protect sensitive tools after untrusted tool output enters a session.

    This policy does not scan content for prompt-injection strings. Instead,
    apps classify tool boundaries:

    - ``taint_sources`` maps inbound/untrusted tool names to taint labels.
    - ``protected_tools`` maps sensitive outbound tools to labels that should
      deny or require approval once present in the session.

    Cayu's runtime supplies durable prior labels through
    ``request.metadata["cayu:taint_labels"]`` and adds same-round labels while
    planning a batch of tool calls.
    """

    def __init__(
        self,
        *,
        taint_sources: Mapping[str, Iterable[str]],
        protected_tools: Mapping[str, Iterable[str]],
        decision: ToolPolicyDecision = ToolPolicyDecision.REQUIRE_APPROVAL,
    ) -> None:
        if type(taint_sources) is not dict:
            raise TypeError("TaintAwareToolPolicy taint_sources must be a dict.")
        if type(protected_tools) is not dict:
            raise TypeError("TaintAwareToolPolicy protected_tools must be a dict.")
        copied_sources: dict[str, frozenset[str]] = {}
        for tool_name, labels in taint_sources.items():
            copied_sources[require_clean_nonblank(tool_name, "tool_name")] = _copy_taint_labels(
                labels,
                f"taint_sources[{tool_name!r}]",
                allow_any=False,
            )
        copied_protected: dict[str, frozenset[str]] = {}
        for tool_name, labels in protected_tools.items():
            copied_protected[require_clean_nonblank(tool_name, "tool_name")] = _copy_taint_labels(
                labels,
                f"protected_tools[{tool_name!r}]",
                allow_any=True,
            )
        if not copied_sources:
            raise ValueError("TaintAwareToolPolicy taint_sources cannot be empty.")
        if not copied_protected:
            raise ValueError("TaintAwareToolPolicy protected_tools cannot be empty.")
        decision = ToolPolicyDecision(decision)
        if decision == ToolPolicyDecision.ALLOW:
            raise ValueError("TaintAwareToolPolicy decision cannot be ALLOW.")
        self.taint_sources = MappingProxyType(copied_sources)
        self.protected_tools = MappingProxyType(copied_protected)
        self.decision = decision

    def labels_for_source_tool(self, tool_name: str) -> frozenset[str]:
        """Return labels added when ``tool_name`` produces a terminal result."""
        return self.taint_sources.get(require_clean_nonblank(tool_name, "tool_name"), frozenset())

    def protected_labels_for_tool(self, tool_name: str) -> frozenset[str]:
        """Return taint labels that protect ``tool_name``."""
        return self.protected_tools.get(
            require_clean_nonblank(tool_name, "tool_name"),
            frozenset(),
        )

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        active_labels = set(taint_labels_from_metadata(request.metadata))
        protected_labels = self.protected_labels_for_tool(request.tool_name)
        if not protected_labels:
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)

        matched_labels = _matched_taint_labels(active_labels, protected_labels)
        if not matched_labels:
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)

        return ToolPolicyResult(
            decision=self.decision,
            reason="Tool call is protected after untrusted data entered the session.",
            metadata={
                "policy": "taint_aware",
                "tool_name": request.tool_name,
                "taint_labels": sorted(active_labels),
                "matched_taint_labels": sorted(matched_labels),
                "protected_taint_labels": sorted(protected_labels),
            },
        )


def taint_labels_from_metadata(metadata: Mapping[str, Any]) -> frozenset[str]:
    labels = metadata.get(TAINT_LABELS_METADATA_KEY)
    if labels is None:
        return frozenset()
    return _copy_taint_labels(labels, TAINT_LABELS_METADATA_KEY, allow_any=False)


def metadata_with_taint_labels(
    metadata: Mapping[str, Any],
    labels: Iterable[str],
) -> dict[str, Any]:
    copied = copy_json_value(dict(metadata), "metadata")
    copied[TAINT_LABELS_METADATA_KEY] = sorted(
        _copy_taint_labels(labels, "labels", allow_any=False)
    )
    return copied


def _copy_tool_name_set(value: Iterable[str], field_name: str) -> frozenset[str]:
    if isinstance(value, str | bytes):
        raise TypeError(f"{field_name} must be an iterable of tool names.")
    try:
        names = list(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of tool names.") from exc

    copied: set[str] = set()
    for index, name in enumerate(names):
        copied.add(require_clean_nonblank(name, f"{field_name}[{index}]"))
    return frozenset(copied)


def _copy_nonempty_string_list(value: Iterable[str], field_name: str) -> list[str]:
    if isinstance(value, str | bytes):
        raise TypeError(f"{field_name} must be an iterable of strings.")
    try:
        items = list(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of strings.") from exc
    copied: list[str] = []
    for index, item in enumerate(items):
        copied.append(require_nonblank(item, f"{field_name}[{index}]"))
    return copied


def _copy_nonempty_string_set(value: Iterable[str], field_name: str) -> frozenset[str]:
    return frozenset(_copy_nonempty_string_list(value, field_name))


def _copy_taint_labels(
    value: Iterable[str],
    field_name: str,
    *,
    allow_any: bool,
) -> frozenset[str]:
    labels = _copy_nonempty_string_set(value, field_name)
    if not labels:
        raise ValueError(f"{field_name} cannot be empty.")
    if not allow_any and ANY_TAINT_LABEL in labels:
        raise ValueError(f"{field_name} cannot contain {ANY_TAINT_LABEL!r}.")
    return labels


def _matched_taint_labels(
    active_labels: set[str],
    protected_labels: frozenset[str],
) -> set[str]:
    if ANY_TAINT_LABEL in protected_labels:
        return set(active_labels)
    return active_labels.intersection(protected_labels)


def _copy_parameter_rules(
    value: Iterable[ParameterRule],
    field_name: str,
) -> tuple[ParameterRule, ...]:
    if isinstance(value, ParameterRule):
        raise TypeError(f"{field_name} must be an iterable of ParameterRule instances.")
    try:
        rules = list(value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of ParameterRule instances.") from exc
    copied: list[ParameterRule] = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, ParameterRule):
            raise TypeError(f"{field_name}[{index}] must be a ParameterRule.")
        copied.append(rule)
    if not copied:
        raise ValueError(f"{field_name} cannot be empty.")
    return tuple(copied)


def _copy_parameter_path(parameter: str) -> tuple[str, ...]:
    cleaned = require_clean_nonblank(parameter, "parameter")
    parts = tuple(require_clean_nonblank(part, "parameter") for part in cleaned.split("."))
    if not parts:
        raise ValueError("parameter must contain at least one path part.")
    return parts


class _MissingArgument:
    pass


_MISSING = _MissingArgument()


def _get_argument_path(arguments: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = arguments
    for part in path:
        if type(current) is not dict or part not in current:
            return _MISSING
        current = current[part]
    return current


def _is_empty_parameter_value(value: Any) -> bool:
    if value is None:
        return True
    if type(value) is str:
        return not value.strip()
    if type(value) in {list, dict}:
        return len(value) == 0
    return False
