"""`cayu cloud connectors tools` and `cayu cloud connections run-tool`: connector package tools.

A connector package defines Agent tools by toolset (`ramp`), which Cayu Cloud runs on a slot's
binding (`cayu.connections.tools`). `connectors tools TOOLSET` shows a toolset's catalog as
Cloud offers it, refreshes the copy in the user's cache, and with `--save` keeps it in the
project (`.cayu/connector-tools/TOOLSET.json`) so CI can load the tools offline.
`connections run-tool --application AGENT SLOT TOOL --arguments-file F` runs one tool on the
Agent's binding of SLOT, as the signed-in developer, with the arguments in F (a JSON object).
"""

from __future__ import annotations

import argparse
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote

from cayu.cli._cloud_api import CloudApiClient
from cayu.cli._cloud_reference import APPLICATION_HELP
from cayu.cli._cloud_settings import _cell, _table
from cayu.cli._json_inputs import json_object_file
from cayu.connections import _tool_catalog
from cayu.connections._tool_catalog import Catalog, ConnectorToolsError


def add_connector_tools_parser(connectors: argparse.ArgumentParser) -> None:
    actions = connectors.add_subparsers(dest="connectors_command")
    tools = actions.add_parser(
        "tools",
        help="Show a connector package's Agent tools.",
        description=(
            "Show the tools of TOOLSET as this Cayu Cloud offers them: each tool's effect, the "
            "scopes it needs and the approval it asks for. Refreshes the copy in your cache; "
            "--save also keeps it in the project for offline CI."
        ),
    )
    tools.add_argument("toolset", help="The toolset, such as ramp.")
    tools.add_argument(
        "--save",
        action="store_true",
        help="Save the catalog as .cayu/connector-tools/TOOLSET.json in the project.",
    )
    tools.add_argument(
        "--json", action="store_true", help="Print the JSON envelope instead of a table."
    )
    tools.set_defaults(_cloud_render=render_tools)


def add_run_tool_parser(actions: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    run = actions.add_parser(
        "run-tool",
        help="Run a connector package's tool on an Agent's slot.",
        description=(
            "Run TOOL of a connector package on the Agent's binding of SLOT, as you: the same "
            "binding, scopes and rules the Agent's own calls get, recorded as a local run."
        ),
    )
    run.add_argument("slot")
    run.add_argument("tool", help="The tool's name in its package, such as ramp_list_bills.")
    run.add_argument("--application", required=True, help=APPLICATION_HELP)
    run.add_argument(
        "--arguments-file",
        required=True,
        type=Path,
        help="A JSON object of the tool's arguments.",
    )


def connector_tools_command(
    arguments: argparse.Namespace, client: CloudApiClient
) -> dict[str, Any]:
    catalog = _catalog(client, arguments.toolset)
    _tool_catalog.save(_tool_catalog.cache_path(catalog.toolset), catalog.document)
    outcome: dict[str, Any] = {"operation": "connectors.tools", "result": dict(catalog.document)}
    if arguments.save:
        path = _tool_catalog.snapshot_path(catalog.toolset)
        if not _tool_catalog.save(path, catalog.document):
            raise OSError(f"Could not write {path}.")
        outcome["saved"] = str(path)
    return outcome


def run_tool_command(
    arguments: argparse.Namespace, client: CloudApiClient, application_id: str
) -> dict[str, Any]:
    given = json_object_file(arguments.arguments_file)
    toolset = arguments.tool.partition("_")[0]
    catalog = _catalog(client, toolset)
    descriptor = next((tool for tool in catalog.tools if tool.name == arguments.tool), None)
    if descriptor is None:
        raise ValueError(f"The toolset {toolset} has no tool {arguments.tool}.")
    answer = client.request(
        "POST",
        f"/v1/applications/{quote(application_id, safe='')}/connections/"
        f"{quote(arguments.slot, safe='')}/tools/{quote(descriptor.name, safe='')}",
        payload={"arguments": given, "digest": descriptor.digest, "call_id": str(uuid.uuid4())},
    )
    return {"operation": "connections.run-tool", "result": answer}


def _catalog(client: CloudApiClient, toolset: str) -> Catalog:
    if not _tool_catalog.TOOLSET_PATTERN.fullmatch(toolset):
        raise ValueError(f"{toolset!r} isn't a toolset id.")
    document = client.request("GET", f"/v1/connector-tools/{toolset}")
    try:
        return _tool_catalog.catalog_from(document, toolset)
    except ConnectorToolsError as exc:
        raise ValueError(str(exc)) from None


def render_tools(outcome: Mapping[str, Any]) -> str:
    """A toolset's catalog: what serves it, then its tools as a table."""

    result = outcome["result"]
    lines = [
        f"{result['toolset']} {result['version']} ({result['package']}): connectors "
        f"{', '.join(result['connectors'])}; default slot {result['default_slot']}",
        "",
    ]
    tools = result.get("tools") or []
    lines.extend(
        _table(
            ("TOOL", "EFFECT", "SCOPES", "APPROVAL", "DESCRIPTION"),
            [
                (
                    tool["name"],
                    tool["effect"],
                    _cell(", ".join(tool.get("scopes") or []) or "-"),
                    _cell(tool.get("approval")),
                    _cell(tool["description"].splitlines()[0], limit=60),
                )
                for tool in tools
            ],
        )
        if tools
        else ["No tools."]
    )
    if "saved" in outcome:
        lines.extend(["", f"Saved to {outcome['saved']}."])
    return "\n".join(lines)
