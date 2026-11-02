"""Declared environments, their retained-session activity, and workflow runs' shared allocations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

from cayu.cli._output import add_output_options, print_table
from cayu.cli._read_targets import (
    ReadTarget,
    _positive_seconds,
    add_read_target_options,
    run_read_command,
)

# The fields `cayu environment allocations` reads from each allocation.
_ALLOCATION_FIELDS = frozenset(
    {
        "allocation_id",
        "workflow_run_id",
        "environment_name",
        "state",
        "reconnectable",
        "run_state",
        "created_at",
    }
)


def add_environment_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "environment",
        help="Inspect an agent's environments.",
        description="Read environment configuration and retained-session activity from a running agent. Use show for one environment's details, and allocations for the sandboxes workflow runs share.",
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
    _add_allocations_parser(commands)


def _add_allocations_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "allocations",
        help="List or release the sandboxes workflow runs share.",
        description=(
            "A workflow run whose steps share one sandbox (allocation_scope parent or "
            "causal_budget) records it in the run's checkpoint. A run that crashed and is "
            "never resumed keeps it until someone releases it; these commands list those "
            "records and release one, copying its work back or discarding it. With "
            "--application NAME they reach a Cayu Cloud Agent, and with --server-url a running "
            "server (both need developer or admin access). Without either they open the "
            "project's application in this process, as `cayu session close` does, because a "
            "release reaches the sandbox through the environment factory it registers."
        ),
    )
    actions = parser.add_subparsers(dest="allocations_command", required=True)
    listing = actions.add_parser(
        "list",
        help="List recorded shared allocations, newest workflow run first.",
        description=(
            "Read the shared allocations workflow runs recorded. Each row says whether its run "
            "looks like it is running (this process or a step session runs it, or it journaled "
            "something within --inactive-for), has stopped, or has ended. Reconnect metadata "
            "and binding state are never shown."
        ),
    )
    listing.add_argument("--workflow-run", metavar="RUN_ID", help="Read only this workflow run.")
    listing.add_argument("--environment", metavar="NAME", help="Only this environment's.")
    listing.add_argument(
        "--limit",
        type=_runs_per_page,
        default=100,
        metavar="RUNS",
        help="How many workflow runs one page reads (1-200, default 100).",
    )
    listing.add_argument("--cursor", help="Continue from a previous page's next_cursor.")
    listing.add_argument(
        "--all", action="store_true", help="Follow next_cursor until every run has been read."
    )
    _add_inactive_for(listing)
    _add_allocation_target_options(listing)
    add_output_options(listing, formats=("json", "table"), default="table")

    release = actions.add_parser(
        "release",
        help="Copy back or discard one recorded shared allocation.",
        description=(
            "Release a shared allocation whose workflow run won't finish it. --copy-back "
            "reconnects, copies the work back to the source workspace as a completed run "
            "would, and disposes of the sandbox; --discard disposes of it without copying "
            "anything back, the usual choice for a crashed run. Refused while the run looks "
            "like it is running, unless --force; never while the server process itself runs "
            "it. The record is fenced first, so a run resumed later starts a fresh sandbox "
            "instead of reattaching this one."
        ),
    )
    release.add_argument("allocation_id", help="The allocation ID `allocations list` shows.")
    mode = release.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--discard",
        dest="mode",
        action="store_const",
        const="discard",
        help="Dispose of the sandbox without copying anything back.",
    )
    mode.add_argument(
        "--copy-back",
        dest="mode",
        action="store_const",
        const="copy_back",
        help="Copy the sandbox's work back to the source workspace, then dispose of it.",
    )
    release.add_argument(
        "--force",
        action="store_true",
        help=(
            "Release even though the run looks like it is running elsewhere. Use it only when "
            "you know that process is gone: a step still running in the sandbox would break."
        ),
    )
    _add_inactive_for(release)
    _add_allocation_target_options(release)
    add_output_options(release, formats=("json", "table"), default="table")


def _add_inactive_for(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--inactive-for",
        dest="inactive_for_seconds",
        type=_nonnegative_seconds,
        default=None,
        metavar="SECONDS",
        help=(
            "How long a run must have journaled nothing to count as stopped (default 900). "
            "A running step session counts as running whatever this is."
        ),
    )


def _add_allocation_target_options(parser: argparse.ArgumentParser) -> None:
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--server-url", help="The running Cayu server at this mount URL.")
    target.add_argument(
        "--application",
        metavar="NAME",
        help="A Cayu Cloud Agent's id, name or slug, signed in like `cayu cloud` commands.",
    )
    target.add_argument(
        "--target",
        metavar="MODULE:FACTORY",
        help="Override project discovery with a module:factory target (local only).",
    )
    parser.add_argument(
        "--authorization-env",
        default="CAYU_API_AUTHORIZATION",
        metavar="NAME",
        help="Authorization header environment variable for --server-url.",
    )
    parser.add_argument(
        "--context",
        type=Path,
        metavar="PATH",
        help="With --application: the private Cayu Cloud context JSON (`cayu cloud --context`).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_seconds,
        default=30.0,
        metavar="SECONDS",
        help="With --application or --server-url: how long each request may take (default: 30).",
    )


def run_environment(args: argparse.Namespace) -> int:
    if args.environment_command == "allocations":
        return _run_allocations(args)
    return run_read_command(
        args, _run_environment, error_code="ENVIRONMENT_INSPECTION_FAILED", schema_version="1"
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
        target.application, context=target.context, timeout_seconds=target.timeout_seconds
    )


async def _run_environment(args: argparse.Namespace, target: ReadTarget) -> int:
    if target.store is not None:
        raise ValueError(
            "Environment inspection requires --server-url or --application; a session database does not declare environments."
        )
    api = _open_api(target)
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


# Shared allocations -------------------------------------------------------


def _run_allocations(args: argparse.Namespace) -> int:
    import asyncio

    from cayu.cli._cloud_agent_api import CayuApiError
    from cayu.cli._output import output_destination, terminal_safe
    from cayu.cli._read_targets import cloud_login_output_conflict

    code = (
        "SHARED_ALLOCATION_LIST_FAILED"
        if args.allocations_command == "list"
        else "SHARED_ALLOCATION_RELEASE_FAILED"
    )
    if args.output is not None and args.application is not None:
        conflict = cloud_login_output_conflict(args.output, args.context)
        if conflict is not None:
            print(f"error: {conflict}", file=sys.stderr)
            return 1
    try:
        with output_destination(args.output):
            try:
                result = asyncio.run(_allocations_result(args))
            except Exception as exc:
                message = str(exc)
                if args.output_format != "json":
                    print(f"error: {terminal_safe(message)}", file=sys.stderr)
                    return 1
                error: dict[str, str] = {"code": code, "message": message}
                if isinstance(exc, CayuApiError):
                    error["category"] = exc.category
                print(
                    json.dumps(
                        {"schema_version": "1", "error": error}, ensure_ascii=False, sort_keys=True
                    )
                )
                return 1
            _render_allocations(args, result)
            return 0
    except OSError as exc:
        print(f"error: could not write output: {exc}", file=sys.stderr)
        return 1


async def _allocations_result(args: argparse.Namespace) -> dict[str, Any]:
    # argparse already refuses --target with --server-url or --application.
    if args.context is not None and args.application is None:
        raise ValueError("--context applies only with --application.")
    if args.server_url is not None or args.application is not None:
        return _remote_allocations(args)
    return await _local_allocations(args)


def _target(args: argparse.Namespace) -> ReadTarget:
    if args.server_url is not None:
        return ReadTarget(
            server_url=args.server_url,
            authorization_env=args.authorization_env,
            timeout_seconds=args.timeout_seconds,
        )
    application = args.application.strip()
    if not application:
        raise ValueError("--application must name an Agent.")
    return ReadTarget(
        application=application, context=args.context, timeout_seconds=args.timeout_seconds
    )


def _remote_allocations(args: argparse.Namespace) -> dict[str, Any]:
    from cayu.cli._cloud_agent_api import CayuApiError, CayuApiNotFound

    api = _open_api(_target(args))
    where = "server" if args.server_url is not None else "cloud"
    try:
        if args.allocations_command == "list":
            allocations: list[dict[str, Any]] = []
            cursor = args.cursor
            while True:
                query = {"limit": str(args.limit)}
                for key, value in (
                    ("workflow_run_id", args.workflow_run),
                    ("environment_name", args.environment),
                    ("cursor", cursor),
                ):
                    if value is not None:
                        query[key] = value
                if args.inactive_for_seconds is not None:
                    query["inactive_for_seconds"] = str(args.inactive_for_seconds)
                page = api.get("/shared-allocations", query=query, resource="shared allocations")
                rows = page.get("allocations") if isinstance(page, dict) else None
                if not isinstance(rows, list) or any(
                    not isinstance(row, dict) or not _ALLOCATION_FIELDS.issubset(row)
                    for row in rows
                ):
                    raise ValueError("The server returned an invalid shared allocation listing.")
                allocations.extend(rows)
                cursor = page.get("next_cursor")
                if not args.all or not isinstance(cursor, str) or not cursor:
                    break
            return {"where": where, "allocations": allocations, "next_cursor": cursor or None}
        body: dict[str, Any] = {
            "allocation_id": args.allocation_id,
            "mode": args.mode,
            "force": args.force,
        }
        if args.inactive_for_seconds is not None:
            body["inactive_for_seconds"] = args.inactive_for_seconds
        try:
            answer = api.mutate("/shared-allocations/release", body, resource="shared allocation")
        except CayuApiNotFound:
            raise ValueError(f"Shared allocation not found: {args.allocation_id}") from None
        except CayuApiError as exc:
            if exc.category == "request_rejected" and exc.detail:
                raise ValueError(exc.detail) from None
            if exc.category == "outcome_unknown":
                raise RuntimeError(
                    f"{exc} `cayu environment allocations list` shows whether it is still "
                    "recorded; releasing it again is safe."
                ) from None
            raise
        if not isinstance(answer, dict) or not isinstance(answer.get("allocation"), dict):
            raise RuntimeError("The server answered the release without a receipt.")
        return {"where": where, **answer}
    finally:
        api.close()


async def _local_allocations(args: argparse.Namespace) -> dict[str, Any]:
    from cayu.cli.project import (
        ProjectError,
        build_project_app,
        close_project_app,
        project_context,
        resolve_project,
    )

    command = f"cayu environment allocations {args.allocations_command}"
    try:
        project = resolve_project(args.target, command=command)
    except ProjectError as exc:
        raise ValueError(str(exc)) from None
    with project_context(project.root):
        try:
            app = build_project_app(project.target, command="Shared allocations")
        except ProjectError as exc:
            raise ValueError(str(exc)) from None
        try:
            if args.allocations_command == "list":
                allocations = []
                cursor = args.cursor
                while True:
                    page = await app.list_shared_allocations(
                        workflow_run_id=args.workflow_run,
                        environment_name=args.environment,
                        limit=args.limit,
                        cursor=cursor,
                        inactive_for_seconds=args.inactive_for_seconds,
                    )
                    allocations.extend(_allocation_payload(row) for row in page.allocations)
                    cursor = page.next_cursor
                    if not args.all or cursor is None:
                        break
                return {"where": "local", "allocations": allocations, "next_cursor": cursor}
            try:
                release = await app.release_shared_allocation(
                    args.allocation_id,
                    mode=args.mode,
                    force=args.force,
                    inactive_for_seconds=args.inactive_for_seconds,
                )
            except KeyError:
                raise ValueError(f"Shared allocation not found: {args.allocation_id}") from None
            return {
                "where": "local",
                "allocation": _allocation_payload(release.allocation),
                "mode": release.mode.value,
                "forced": release.forced,
                "message": release.message,
            }
        finally:
            await close_project_app(app)


def _allocation_payload(allocation: Any) -> dict[str, Any]:
    from dataclasses import asdict

    payload = asdict(allocation)
    created_at = payload.get("created_at")
    payload["created_at"] = None if created_at is None else created_at.isoformat()
    return payload


def _render_allocations(args: argparse.Namespace, result: dict[str, Any]) -> None:
    from cayu.cli._output import terminal_safe

    if args.output_format == "json":
        print(json.dumps({"schema_version": "1", **result}, ensure_ascii=False, sort_keys=True))
        return
    if args.allocations_command == "release":
        print(terminal_safe(str(result.get("message") or "Released.")))
        return
    rows = [
        {
            "Allocation": row["allocation_id"],
            "Workflow run": row["workflow_run_id"],
            "Environment": row["environment_name"],
            "State": row["state"],
            "Run": row["run_state"],
            "Created": row.get("created_at"),
        }
        for row in result["allocations"]
    ]
    if rows:
        print_table(("Allocation", "Workflow run", "Environment", "State", "Run", "Created"), rows)
    else:
        print("No shared allocations are recorded.")
    if result.get("next_cursor"):
        print(
            "More workflow runs to read: pass --cursor "
            f"{terminal_safe(str(result['next_cursor']))}, or --all."
        )


def _runs_per_page(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a whole number from 1 to 200") from None
    if not 1 <= number <= 200:
        raise argparse.ArgumentTypeError("must be a whole number from 1 to 200")
    return number


def _nonnegative_seconds(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a whole number of seconds") from None
    if number < 0:
        raise argparse.ArgumentTypeError("must be a whole number of seconds")
    return number
