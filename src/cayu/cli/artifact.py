"""Saved-file inventory queries through the agent's protected API."""

from __future__ import annotations

import argparse
import json
from typing import Any

from cayu.cli._output import add_output_options, print_table, terminal_safe
from cayu.cli._read_targets import ReadTarget, add_read_target_options, run_read_command

_ORDERS = (
    "created_at_desc",
    "created_at_asc",
    "filename_asc",
    "filename_desc",
    "size_bytes_desc",
    "size_bytes_asc",
)


def add_artifact_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "artifact",
        help="Inspect an agent's saved files.",
        description="List and search saved files. Open the dashboard preview to inspect a file's contents.",
    )
    commands = parser.add_subparsers(dest="artifact_command", required=True)
    listing = commands.add_parser(
        "list",
        help="List or search saved files.",
        description="Search public file metadata and producing-session titles before paging. Use --artifact-store-id to select one store and --offset for the next page.",
    )
    listing.add_argument(
        "--query", "--q", dest="query", help="Literal public-text search; at most 256 characters."
    )
    listing.add_argument("--order-by", choices=_ORDERS, default="created_at_desc")
    listing.add_argument("--artifact-store-id")
    listing.add_argument("--scope", choices=("session", "environment"))
    listing.add_argument("--session", dest="session_id")
    listing.add_argument("--agent", dest="agent_name")
    listing.add_argument("--environment", dest="environment_name")
    listing.add_argument("--limit", type=int, default=100)
    listing.add_argument("--offset", type=int, default=0)
    add_read_target_options(listing, allow_store=False)
    add_output_options(listing, formats=("json", "table"), default="table")


def run_artifact(args: argparse.Namespace) -> int:
    return run_read_command(
        args, _run_artifact, error_code="ARTIFACT_INSPECTION_FAILED", schema_version="1"
    )


async def _run_artifact(args: argparse.Namespace, target: ReadTarget) -> int:
    from cayu.cli._cloud_agent_api import CayuApi, open_cloud_agent_api
    from cayu.cli.diagnostics import read_authorization

    if not 1 <= args.limit <= 500 or not 0 <= args.offset <= 10_000:
        raise ValueError("Artifact limit must be 1..500 and offset 0..10000.")
    if args.query is not None and len(args.query) > 256:
        raise ValueError("Artifact query must be at most 256 characters.")
    if target.server_url is not None:
        api = CayuApi(
            target.server_url,
            authorization=read_authorization(target.authorization_env),
            timeout_seconds=target.timeout_seconds,
        )
    else:
        assert target.application is not None
        api = open_cloud_agent_api(
            target.application,
            context=target.context,
            timeout_seconds=target.timeout_seconds,
            wake_timeout_seconds=target.wake_timeout_seconds,
            offers_wake_option=True,
        )
    try:
        if (
            args.query is not None or args.order_by != "created_at_desc"
        ) and args.artifact_store_id is None:
            capability = api.get("/contract", resource="server capabilities")
            for key in ("capabilities", "surfaces", "artifact_search", "read", "enabled"):
                capability = capability.get(key) if isinstance(capability, dict) else None
            if capability is not True:
                raise ValueError(
                    "The server does not advertise global artifact search and sorting support."
                )
        query = {
            key: str(value)
            for key, value in (
                ("q", args.query),
                ("order_by", args.order_by),
                ("limit", args.limit),
                ("offset", args.offset),
                *(
                    (key, getattr(args, key))
                    for key in (
                        "artifact_store_id",
                        "scope",
                        "session_id",
                        "agent_name",
                        "environment_name",
                    )
                ),
            )
            if value is not None
        }
        result = api.get("/artifacts", query=query, resource="artifacts")
        if not isinstance(result, dict) or not isinstance(result.get("artifacts"), list):
            raise ValueError("The server returned an invalid artifact inventory.")
        if any(
            not isinstance(row, dict) or not {"filename", "artifact_store_id", "id"}.issubset(row)
            for row in result["artifacts"]
        ):
            raise ValueError("The server returned invalid artifact rows.")
        if args.output_format == "json":
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        else:
            rows = [
                {
                    "File": row["filename"],
                    "Session": row.get("session_title") or "Untitled session"
                    if row.get("session_id")
                    else "—",
                    "Specialist": row.get("agent_name") or "—",
                    "Bytes": row.get("size_bytes"),
                    "Store": row["artifact_store_id"],
                    "ID": row["id"],
                }
                for row in result["artifacts"]
            ]
            print_table(("File", "Session", "Specialist", "Bytes", "Store", "ID"), rows)
            total = result.get("total_count")
            print(
                f"{len(rows)} files loaded; total {'unavailable' if total is None else terminal_safe(str(total))}."
            )
            if result.get("next_offset") is not None:
                print(f"Next page: --offset {terminal_safe(str(result['next_offset']))}")
        return 0
    finally:
        api.close()
