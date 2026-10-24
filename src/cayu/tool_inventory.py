"""Shared declaration and event semantics for local and Cloud tool inventories.

Transport limits belong to each consumer. Classification never invokes a tool or
its policy, and usage contributions contain no arguments or results.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cached_property
from importlib import metadata
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from cayu.events import Event, EventType

if TYPE_CHECKING:
    from cayu.runtime.manifest import ToolManifest

SourceKind = Literal["agent", "package", "mcp", "cayu", "provider", "unknown"]
ToolEffect = Literal["none", "idempotent", "external"]
ToolApproval = Literal["never", "always", "sometimes", "blocked", "unknown"]
_SOURCE_KINDS: dict[str, SourceKind] = {"project": "agent", "built_in": "cayu"}
APPROVALS: dict[str, ToolApproval] = {
    "allowed": "never",
    "approval_required": "always",
    "conditional": "sometimes",
    "denied": "blocked",
}
EFFECTS: dict[str, ToolEffect] = {
    "none": "none",
    "idempotent": "idempotent",
    "external": "external",
}
TOOL_USAGE_EVENT_TYPES = (
    EventType.TOOL_CALL_COMPLETED,
    EventType.TOOL_CALL_FAILED,
    EventType.TOOL_CALL_APPROVED,
    EventType.TOOL_CALL_APPROVAL_DENIED,
    EventType.MODEL_HOSTED_TOOL_CALL,
)
_HOSTED_TERMINAL_STATUSES = frozenset({"completed", "failed", "incomplete", "outcome_unknown"})
_LINE_BREAKS = re.compile(r"[\x00-\x1f\x7f]")


class ToolSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    kind: SourceKind
    package: str | None = None
    version: str | None = None
    server: str | None = None
    provider: str | None = None


def _line(value: str | None, limit: int | None) -> str | None:
    if value is None:
        return None
    cleaned = _LINE_BREAKS.sub(" ", value).strip()
    return cleaned if cleaned and (limit is None or len(cleaned) <= limit) else None


class ToolSourceMetadata:
    """Lazy metadata snapshot owned by one inventory build, never process-global."""

    def __init__(self) -> None:
        self._versions: dict[str, str | None] = {}

    @cached_property
    def packages(self) -> Mapping[str, list[str]]:
        return metadata.packages_distributions()

    def version(self, package: str) -> str | None:
        if package not in self._versions:
            try:
                self._versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                self._versions[package] = None
        return self._versions[package]


def classify_tool_source(
    tool: ToolManifest,
    registered: object,
    sanitize: Callable[[str | None], str | None],
    *,
    source_metadata: ToolSourceMetadata,
    name_limit: int | None = None,
    version_limit: int | None = None,
) -> ToolSource:
    from cayu.mcp.tools import McpToolAdapter

    if isinstance(registered, McpToolAdapter):
        server = _line(sanitize(registered.server.name), name_limit)
        return (
            ToolSource(kind="unknown") if server is None else ToolSource(kind="mcp", server=server)
        )
    provenance = tool.implementation_provenance
    if provenance.kind == "external":
        module = provenance.symbol.partition(":")[0]
        for package in sorted(source_metadata.packages.get(module.split(".")[0], ())):
            version = source_metadata.version(package)
            if version is None:
                continue
            name, version = (
                _line(sanitize(package), name_limit),
                _line(sanitize(version), version_limit),
            )
            if name is not None and version is not None:
                return ToolSource(kind="package", package=name, version=version)
        return ToolSource(kind="unknown")
    return ToolSource(kind=_SOURCE_KINDS.get(provenance.kind, "unknown"))


@dataclass(frozen=True)
class ToolUsageContribution:
    name: str
    calls: int = 0
    failures: int = 0
    approved: int = 0
    rejected: int = 0
    outcome: Literal["completed", "failed", "incomplete", "outcome_unknown"] | None = None


def tool_usage_contribution(event: Event) -> ToolUsageContribution | None:
    """One retained terminal call or approval decision; never infer missing outcomes."""
    if event.type == EventType.MODEL_HOSTED_TOOL_CALL:
        status = event.payload.get("status")
        if (
            event.payload.get("tool_type") != "web_search"
            or type(status) is not str
            or status not in _HOSTED_TERMINAL_STATUSES
        ):
            return None
        # Explicit branches retain a closed typed outcome vocabulary.
        outcome: Literal["completed", "failed", "incomplete", "outcome_unknown"]
        if status == "completed":
            outcome = "completed"
        elif status == "failed":
            outcome = "failed"
        elif status == "incomplete":
            outcome = "incomplete"
        else:
            outcome = "outcome_unknown"
        return ToolUsageContribution(
            "openai_web_search", calls=1, failures=int(status == "failed"), outcome=outcome
        )
    if event.tool_name is None:
        return None
    if event.type == EventType.TOOL_CALL_COMPLETED:
        return ToolUsageContribution(event.tool_name, calls=1, outcome="completed")
    if event.type == EventType.TOOL_CALL_FAILED:
        return ToolUsageContribution(event.tool_name, calls=1, failures=1, outcome="failed")
    if event.type == EventType.TOOL_CALL_APPROVED:
        return ToolUsageContribution(event.tool_name, approved=1)
    if event.type == EventType.TOOL_CALL_APPROVAL_DENIED:
        return ToolUsageContribution(event.tool_name, rejected=1)
    return None
