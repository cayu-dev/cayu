"""Declared environments and their retained-session activity via the agent API."""

from __future__ import annotations

import argparse
import json
from typing import Any
from urllib.parse import quote

from cayu.cli._output import add_output_options, print_table
from cayu.cli._read_targets import ReadTarget, add_read_target_options, run_read_command


def add_environment_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "environment",
        help="Inspect an agent's environments.",
        description="Read environment configuration and retained-session activity from a running agent. Use show for one environment's details.",
    )
    commands = parser.add_subparsers(dest="environment_command", required=True)
    for name in ("list", "show"):
        command = commands.add_parser(
            name,
            help="List environments." if name == "list" else "Show one environment.",
            description="Read declared MCP servers, file storage and session activity. Inspect sessions filtered by this environment to investigate its recent work.",
        )
        if name == "show":
            command.add_argument("name", help="Registered environment name.")
        add_read_target_options(command, allow_store=False)
        add_output_options(command, formats=("json", "table"), default="table")


def run_environment(args: argparse.Namespace) -> int:
    return run_read_command(
        args, _run_environment, error_code="ENVIRONMENT_INSPECTION_FAILED", schema_version="1"
    )


async def _run_environment(args: argparse.Namespace, target: ReadTarget) -> int:
    from cayu.cli._cloud_agent_api import CayuApi, open_cloud_agent_api
    from cayu.cli.diagnostics import read_authorization

    if target.store is not None:
        raise ValueError(
            "Environment inspection requires --server-url or --application; a session database does not declare environments."
        )
    if target.server_url is not None:
        api = CayuApi(
            target.server_url,
            authorization=read_authorization(target.authorization_env),
            timeout_seconds=target.timeout_seconds,
        )
    else:
        assert target.application is not None
        api = open_cloud_agent_api(
            target.application, context=target.context, timeout_seconds=target.timeout_seconds
        )
    try:
        path = "/environments"
        if args.environment_command == "show":
            encoded_name = quote(args.name, safe="")
            # urllib keeps dots unescaped; HTTP clients otherwise normalize dot segments.
            if encoded_name in {".", ".."}:
                encoded_name = encoded_name.replace(".", "%2E")
            path += "/" + encoded_name
        result = api.get(path, resource="environments")
        if not isinstance(result, dict) or not isinstance(result.get("environments"), list):
            raise ValueError("The server returned an invalid environment catalog.")
        required = {
            "name",
            "mcp_server_names",
            "artifact_store_kind",
            "session_count",
            "last_activity_at",
            "activity_as_of",
        }
        if any(
            not isinstance(row, dict) or not required.issubset(row)
            for row in result["environments"]
        ):
            raise ValueError("The server does not publish environment details.")
        if args.environment_command == "show" and (
            len(result["environments"]) != 1 or result["environments"][0]["name"] != args.name
        ):
            raise ValueError("The server did not return the requested environment.")
        if args.output_format == "json":
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        else:
            rows = [
                {
                    "Environment": row["name"],
                    "Files": row.get("artifact_store_kind") or "Not configured",
                    "MCP servers": ", ".join(row.get("mcp_server_names", [])) or "None",
                    "Sessions": row.get("session_count")
                    if row.get("session_count") is not None
                    else "Unavailable",
                    "Last activity": row.get("last_activity_at")
                    or ("None" if row.get("session_count") == "0" else "Unavailable"),
                }
                for row in result["environments"]
            ]
            print_table(("Environment", "Files", "MCP servers", "Sessions", "Last activity"), rows)
        return 0
    finally:
        api.close()
