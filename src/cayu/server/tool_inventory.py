"""Public tool declarations, with shared Cloud classification and local cardinality."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cayu.runtime.manifest import describe_inventory_tools
from cayu.tool_inventory import (
    APPROVALS,
    EFFECTS,
    ToolSource,
    ToolSourceMetadata,
    classify_tool_source,
)

if TYPE_CHECKING:
    from pathlib import Path

    from cayu.applications import CayuApp


def tool_declarations(
    app: CayuApp, *, project_root: Path | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Preserve each specialist's declarations, including colliding public names.

    The returned mapping is internal and keyed by registered agent identity. Only
    its values cross the HTTP boundary. Names and metadata in those values are
    public projections, never raw-name lookup capabilities.
    """
    declarations = describe_inventory_tools(app, project_root=project_root)

    def sanitize(value: str | None) -> str | None:
        if value is None:
            return None
        redacted = app.redact_json(value)
        assert isinstance(redacted, str)
        return redacted

    source_metadata = ToolSourceMetadata()
    result: dict[str, list[dict[str, Any]]] = {}
    for agent_name, registration in sorted(app._agent_registry.registrations.items()):
        rows: list[dict[str, Any]] = []
        for tool, (_, registered) in zip(
            declarations[agent_name], sorted(registration.tools.items()), strict=True
        ):
            rows.append(
                {
                    "name": sanitize(registered.name),
                    "description": sanitize(registered.description),
                    "display": (
                        None
                        if registered.display is None
                        else {
                            "verb": sanitize(registered.display.verb),
                            "noun": sanitize(registered.display.noun),
                            "plural": sanitize(registered.display.plural),
                        }
                    ),
                    "input_schema": app.redact_json(registered.schema),
                    "parallel_safe": registered.parallel_safe,
                    "effect": EFFECTS.get(tool.effect, "external"),
                    "workspace_mutation": registered.workspace_mutation,
                    "source": classify_tool_source(
                        tool, registered.tool, sanitize, source_metadata=source_metadata
                    ).model_dump(mode="json"),
                    "approval": APPROVALS.get(tool.policy_coverage, "unknown"),
                }
            )
        for hosted in registration.hosted_tools:
            rows.append(
                {
                    "name": sanitize(hosted.type),
                    "description": "Provider-hosted tool",
                    "display": None,
                    "input_schema": {},
                    "parallel_safe": False,
                    "effect": "none",
                    "workspace_mutation": False,
                    "source": ToolSource(
                        kind="provider", provider=sanitize(hosted.type.partition("_")[0])
                    ).model_dump(mode="json"),
                    "approval": "never",
                }
            )
        result[agent_name] = rows
    return result


def public_tool_names(app: CayuApp) -> dict[str, str]:
    """Private registered names mapped to their public grouping labels."""
    names: set[str] = set()
    for registration in app._agent_registry.registrations.values():
        names.update(registration.tools)
        names.update(tool.type for tool in registration.hosted_tools)
    result = {}
    for name in names:
        public = app.redact_json(name)
        assert isinstance(public, str)
        result[name] = public
    return result
