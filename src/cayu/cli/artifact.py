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
    listing.add_argument(
        "--cursor", help="Continue from the previous page's next cursor (keyset paging)."
    )
    add_read_target_options(listing, allow_store=False)
    add_output_options(listing, formats=("json", "table"), default="table")
    reindex = commands.add_parser(
        "reindex",
        help="Check or repair the artifact index against the artifact stores.",
        description=(
            "Walk each artifact store once and compare it with the artifact index that "
            "lists and counts read: an artifact with no row, a row with no artifact, or "
            "differing metadata. --check only reports; without it the index is repaired. "
            "Exits 1 when --check finds drift. Large stores take a while; raise "
            "--timeout-seconds."
        ),
    )
    reindex.add_argument("--check", action="store_true", help="Report drift; change nothing.")
    reindex.add_argument("--artifact-store-id", help="Only this artifact store.")
    add_read_target_options(reindex, allow_store=False)
    add_output_options(reindex, formats=("json", "table"), default="table")


def run_artifact(args: argparse.Namespace) -> int:
    return run_read_command(
        args, _run_artifact, error_code="ARTIFACT_INSPECTION_FAILED", schema_version="1"
    )


def _open_api(target: ReadTarget):
    from cayu.cli._cloud_agent_api import CayuApi, open_cloud_agent_api
    from cayu.cli.diagnostics import read_authorization

    if target.server_url is not None:
        return CayuApi(
            target.server_url,
            authorization=read_authorization(target.authorization_env),
            timeout_seconds=target.timeout_seconds,
        )
    assert target.application is not None
    return open_cloud_agent_api(
        target.application,
        context=target.context,
        timeout_seconds=target.timeout_seconds,
        wake_timeout_seconds=target.wake_timeout_seconds,
        offers_wake_option=True,
    )


async def _run_reindex(args: argparse.Namespace, target: ReadTarget) -> int:
    api = _open_api(target)
    try:
        body: dict[str, Any] = {"check": bool(args.check)}
        if args.artifact_store_id is not None:
            body["artifact_store_id"] = args.artifact_store_id
        result = api.post("/artifacts/index/reindex", body, resource="artifact index")
        stores = result.get("stores") if isinstance(result, dict) else None
        if not isinstance(stores, list) or not all(isinstance(row, dict) for row in stores):
            raise ValueError("The server returned an invalid artifact index report.")
        if args.output_format == "json":
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        else:
            print_table(
                ("Store", "Artifacts", "Rows", "Missing", "Orphaned", "Changed", "Result"),
                [
                    {
                        "Store": row.get("artifact_store_id"),
                        "Artifacts": row.get("store_count"),
                        "Rows": row.get("index_count"),
                        "Missing": row.get("missing_count"),
                        "Orphaned": row.get("orphaned_count"),
                        "Changed": row.get("changed_count"),
                        "Result": ("repaired" if row.get("repaired") else "drift")
                        if row.get("drift")
                        else "in step",
                    }
                    for row in stores
                ],
            )
            for row in stores:
                for kind in ("missing", "orphaned", "changed"):
                    for artifact_id in row.get(kind) or ():
                        print(
                            f"{kind}: {terminal_safe(str(row.get('artifact_store_id')))} "
                            f"{terminal_safe(str(artifact_id))}"
                        )
        return 1 if args.check and any(row.get("drift") for row in stores) else 0
    finally:
        api.close()


async def _run_artifact(args: argparse.Namespace, target: ReadTarget) -> int:
    if args.artifact_command == "reindex":
        return await _run_reindex(args, target)
    if not 1 <= args.limit <= 500 or not 0 <= args.offset <= 10_000:
        raise ValueError("Artifact limit must be 1..500 and offset 0..10000.")
    if args.query is not None and len(args.query) > 256:
        raise ValueError("Artifact query must be at most 256 characters.")
    api = _open_api(target)
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
                ("cursor", args.cursor),
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
            if result.get("next_cursor") is not None:
                print(f"Next page: --cursor {terminal_safe(str(result['next_cursor']))}")
            elif result.get("next_offset") is not None:
                print(f"Next page: --offset {terminal_safe(str(result['next_offset']))}")
        return 0
    finally:
        api.close()
