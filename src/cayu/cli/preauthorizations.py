"""`cayu preauthorizations`: authorize one exact tool call for a task before it runs.

A preauthorization lets a task make one call (one tool, exactly these arguments) without
asking for approval, once. These commands issue one, list a task's, show one and revoke
one, through the routes ``app.host_tool_call_preauthorizations(...)`` serves: against the
project's own application in this process, recorded as ``--reviewer`` (your local user by
default), or with ``--application NAME`` against a Cayu Cloud Agent, which Cloud asks as the
person signed in with ``cayu cloud login``, so the Agent records them. See
``docs/tool-authorization-grants.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

from cayu.approvals.authorization_grants import TOOL_AUTHORIZATION_GRANT_MAX_LIFETIME_SECONDS
from cayu.cli._output import add_output_options, output_destination
from cayu.cli.inbox import _local_reviewer, _plain, _print_json, _print_table
from cayu.cli.project import (
    ProjectError,
    build_project_app,
    close_project_app,
    project_context,
    resolve_project,
)

_ROUTES = "/api/tool-call-preauthorizations"


class PreauthorizationError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def add_preauthorizations_parser(subparsers: Any) -> None:
    """Register ``cayu preauthorizations``."""

    parser = subparsers.add_parser(
        "preauthorizations",
        help="Authorize one exact tool call for a task before it runs.",
        description=(
            "A preauthorization lets a task make one exact tool call without asking for "
            "approval, once; it never overrides a denial or a mandatory approval. Issue one, "
            "list a task's, show one and revoke one. Without --application the commands open "
            "the project's application in this process; with --application NAME they ask that "
            "Cayu Cloud Agent as you, with your `cayu cloud login`. The application must host "
            "them (app.host_tool_call_preauthorizations) and decides who may do what."
        ),
    )
    commands = parser.add_subparsers(dest="preauthorizations_command", required=True)

    issue = commands.add_parser(
        "issue",
        help="Preauthorize one exact call for a task.",
        description=(
            "Preauthorize one call for a task: this tool with exactly these arguments. The "
            "arguments are hashed, not stored. The record names you as who authorized it. "
            "Giving --id makes a retry return the same record."
        ),
    )
    issue.add_argument("--task", required=True, metavar="TASK_ID", help="The task it is for.")
    issue.add_argument("--tool", required=True, metavar="NAME", help="The tool it may call.")
    arguments = issue.add_mutually_exclusive_group(required=True)
    arguments.add_argument(
        "--arguments", metavar="JSON", help="The call's exact arguments as a JSON object."
    )
    arguments.add_argument(
        "--arguments-file", type=Path, metavar="PATH", help="A file holding that JSON object."
    )
    issue.add_argument("--agent", metavar="NAME", help="Only this agent may make the call.")
    issue.add_argument("--id", dest="preauthorization_id", metavar="ID", help="The record's id.")
    issue.add_argument(
        "--expires-in-seconds",
        type=int,
        metavar="SECONDS",
        help=f"Unused after this long (1 to {TOOL_AUTHORIZATION_GRANT_MAX_LIFETIME_SECONDS}).",
    )
    issue.add_argument("--justification", help="Why, recorded on the record.")
    _add_common(issue)

    listing = commands.add_parser(
        "list",
        help="List a task's preauthorizations.",
        description=(
            "List a task's preauthorizations you may see, oldest first, with what each can do "
            "now. Show one with `cayu preauthorizations show ID`."
        ),
    )
    listing.add_argument("--task", required=True, metavar="TASK_ID", help="The task.")
    _add_common(listing)

    show = commands.add_parser(
        "show",
        help="Show one preauthorization.",
        description=(
            "Show one preauthorization: its task, tool, arguments digest, who authorized it and "
            "whether it was used or revoked. Revoke an unused one with "
            "`cayu preauthorizations revoke ID`."
        ),
    )
    show.add_argument("preauthorization_id", metavar="ID")
    _add_common(show)

    revoke = commands.add_parser(
        "revoke",
        help="Revoke an unused preauthorization.",
        description="Revoke an unused preauthorization. A used one can't be revoked.",
    )
    revoke.add_argument("preauthorization_id", metavar="ID")
    _add_common(revoke)


def _add_common(parser: argparse.ArgumentParser) -> None:
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--application",
        metavar="NAME",
        help="A Cayu Cloud Agent's name or slug, asked as you with your `cayu cloud login`.",
    )
    target.add_argument(
        "--target",
        metavar="MODULE:FACTORY",
        help="Override project discovery with a module:factory target (local only).",
    )
    parser.add_argument(
        "--context",
        type=Path,
        help="Private Cayu Cloud context JSON (with --application; default: the selected one).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="How long one Cayu Cloud request may take (default 60).",
    )
    parser.add_argument(
        "--reviewer",
        metavar="SUBJECT",
        help=(
            "Who a local record names (default: local:<your user name>). A Cloud Agent "
            "records your login."
        ),
    )
    add_output_options(parser, formats=("json", "table"), default="table")


def run_preauthorizations(args: argparse.Namespace) -> int:
    try:
        with output_destination(args.output):
            return asyncio.run(_run(args))
    except OSError as exc:
        print(f"error: could not write output: {exc}", file=sys.stderr)
        return 1


async def _run(args: argparse.Namespace) -> int:
    from cayu.cli._cloud_api import CloudApiError
    from cayu.cli._cloud_auth import CloudAuthError
    from cayu.cli.cloud import CloudCommandError

    try:
        request = _request(args)
        if args.application is not None:
            if args.reviewer is not None:
                raise ValueError("--reviewer applies locally; a Cloud Agent records your login.")
            answer = await asyncio.to_thread(_cloud_call, args, *request)
        else:
            project = resolve_project(args.target, command="cayu preauthorizations")
            with project_context(project.root):
                app = build_project_app(project.target, command="Preauthorizations")
                try:
                    answer = await _call(app, args.reviewer or _local_reviewer(), *request)
                finally:
                    await close_project_app(app)
    except (CloudAuthError, CloudCommandError) as exc:
        return _error(args, exc.category, str(exc))
    except CloudApiError as exc:
        return _error(args, exc.code or exc.category, str(exc))
    except PreauthorizationError as exc:
        return _error(args, exc.code, str(exc))
    except ProjectError as exc:
        return _error(args, "project_unavailable", str(exc))
    except ValueError as exc:
        return _error(args, "invalid_input", str(exc))
    if args.output_format == "json":
        _print_json(answer)
        return 0
    if args.preauthorizations_command == "list":
        views = answer.get("preauthorizations", [])
        if not views:
            print("No preauthorizations.")
        else:
            _print_table(
                ("ID", "STATUS", "TOOL", "AGENT", "AUTHORIZED BY", "EXPIRES"),
                [_row(view) for view in views],
            )
        return 0
    _print_view(answer)
    return 0


def _request(args: argparse.Namespace) -> tuple[str, str, dict[str, Any] | None, dict | None]:
    command = args.preauthorizations_command
    if command == "issue":
        if args.arguments is not None:
            text = args.arguments
        else:
            try:
                text = args.arguments_file.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(f"Can't read --arguments-file: {exc}.") from None
        try:
            arguments = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"The arguments are not JSON: {exc.msg}.") from None
        if not isinstance(arguments, dict):
            raise ValueError("The arguments must be a JSON object.")
        body: dict[str, Any] = {
            "task_id": args.task,
            "tool_name": args.tool,
            "arguments": arguments,
        }
        for key, value in (
            ("agent_name", args.agent),
            ("preauthorization_id", args.preauthorization_id),
            ("lifetime_seconds", args.expires_in_seconds),
            ("justification", args.justification),
        ):
            if value is not None:
                body[key] = value
        return "POST", _ROUTES, None, body
    if command == "list":
        return "GET", _ROUTES, {"task_id": args.task}, None
    path = f"{_ROUTES}/{quote(args.preauthorization_id, safe='')}"
    if command == "show":
        return "GET", path, None, None
    return "POST", path + "/revoke", None, None


async def _call(
    app: Any,
    reviewer: str,
    method: str,
    path: str,
    params: dict[str, Any] | None,
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    hosting = app.tool_call_preauthorization_hosting
    if hosting is None:
        raise PreauthorizationError(
            "preauthorizations_not_served",
            "This application doesn't serve tool-call preauthorizations; its factory "
            "calls app.host_tool_call_preauthorizations(...) to serve them.",
        )
    import httpx
    from fastapi import FastAPI

    from cayu.server.auth import AuthContext
    from cayu.server.tool_call_preauthorizations import ToolCallPreauthorizationServer

    identity = AuthContext(subject=reviewer)
    api = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    api.include_router(
        ToolCallPreauthorizationServer(hosting.store, authorize=hosting.authorize).router(
            auth=lambda _request: identity
        ),
        prefix="/api",
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api), base_url="http://cayu.local"
    ) as client:
        response = await client.request(method, path, params=params, json=payload)
    try:
        body = response.json()
    except ValueError:
        body = None
    if not 200 <= response.status_code < 300:
        detail = body.get("detail") if isinstance(body, dict) else None
        if isinstance(detail, list):
            # A request FastAPI refused: each entry names a field and why.
            detail = (
                "; ".join(
                    f"{'.'.join(str(part) for part in entry.get('loc', ())[1:])}: {entry.get('msg')}"
                    for entry in detail
                    if isinstance(entry, dict)
                )
                or None
            )
        code = {403: "forbidden", 404: "not_found", 409: "conflict"}.get(
            response.status_code, "invalid_input"
        )
        raise PreauthorizationError(
            code, detail if isinstance(detail, str) else f"HTTP {response.status_code}"
        )
    return body if isinstance(body, dict) else {}


def _cloud_call(
    args: argparse.Namespace,
    method: str,
    path: str,
    params: dict[str, Any] | None,
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """The same route on a Cayu Cloud Agent, which Cloud asks as the signed-in person."""

    from cayu.cli._cloud_api import CloudApiError
    from cayu.cli.cloud import cloud_client, resolve_application_id

    client = cloud_client(context=args.context, timeout_seconds=args.timeout_seconds)
    application_id = resolve_application_id(client, args.application)
    cloud_path = f"/v1/applications/{quote(application_id, safe='')}" + path.removeprefix("/api")
    try:
        answer = client.request(method, cloud_path, payload=payload, query=params)
    except CloudApiError as exc:
        if exc.status_code == 404 and exc.code is None:
            # No such route: this Cayu Cloud doesn't forward these to an Agent yet.
            raise PreauthorizationError(
                "preauthorizations_unsupported",
                "This Cayu Cloud can't reach an Agent's tool-call preauthorizations yet.",
            ) from None
        raise
    if answer.get("served") is False:
        raise PreauthorizationError(
            "preauthorizations_not_served",
            "The Agent doesn't serve tool-call preauthorizations; its factory calls "
            "app.host_tool_call_preauthorizations(...) to serve them.",
        )
    answer.pop("served", None)
    return answer


def _error(args: argparse.Namespace, code: str, message: str) -> int:
    if args.output_format == "json":
        _print_json({"error": {"code": code, "message": message}})
    else:
        print(f"error: {_plain(message)}", file=sys.stderr)
    return 1


def _row(view: dict[str, Any]) -> dict[str, Any]:
    record = view.get("preauthorization") or {}
    return {
        "ID": record.get("preauthorization_id"),
        "STATUS": view.get("status"),
        "TOOL": record.get("tool_name"),
        "AGENT": record.get("agent_name") or "any",
        "AUTHORIZED BY": (record.get("authorized_by") or {}).get("subject"),
        "EXPIRES": record.get("expires_at") or "never",
    }


def _print_view(view: dict[str, Any]) -> None:
    record = view.get("preauthorization") or {}
    use = record.get("use") or {}
    lines: list[tuple[str, object]] = [
        ("Preauthorization", record.get("preauthorization_id")),
        ("Status", view.get("status")),
        ("Task", record.get("task_id")),
        ("Tool", record.get("tool_name")),
        ("Agent", record.get("agent_name") or "any"),
        ("Arguments digest", record.get("arguments_digest")),
        (
            "Authorized",
            f"{record.get('created_at')} by {(record.get('authorized_by') or {}).get('subject')}",
        ),
        ("Expires", record.get("expires_at") or "never"),
        ("Justification", record.get("justification")),
    ]
    if use:
        lines.append(
            (
                "Used",
                f"{use.get('used_at')} by call {use.get('tool_call_id')} "
                f"in session {use.get('session_id')}",
            )
        )
    if record.get("revoked_at") is not None:
        lines.append(
            (
                "Revoked",
                f"{record.get('revoked_at')} by {(record.get('revoked_by') or {}).get('subject')}",
            )
        )
    width = max(len(label) for label, _ in lines)
    for label, value in lines:
        print(f"{label + ':':<{width + 1}} {_plain(value)}")
