"""`cayu tools propose`: raise one exact tool call from the terminal, as application code does.

It opens the project's application in this process, as `cayu inbox` does, and calls
``app.propose_tool_call``: the call goes through the agent's exposure and tool policy
like a model's. A proposal-only tool is admitted through the runtime's own proposal
record; any other tool must be exposed. Nothing runs behind the policy's back: the
result is a pending approval in the inbox, a block, or, when the policy allows it,
the tool runs here, in this process.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import sys
from pathlib import Path
from typing import Any

from cayu.cli._json_inputs import json_object_file
from cayu.cli._output import add_output_options, output_destination, terminal_safe
from cayu.cli.project import (
    ProjectError,
    build_project_app,
    build_project_service,
    close_project_app,
    project_context,
    resolve_project,
)

CLI_SCHEMA_VERSION = "1"
_SUCCESS_STATUSES = frozenset({"completed", "pending_approval"})
_WOULD = {
    "allow": "would run without approval",
    "require_approval": "would wait for approval",
    "deny": "would be blocked",
}


def add_tools_parser(subparsers: Any) -> None:
    """Register ``cayu tools``."""

    tools = subparsers.add_parser(
        "tools",
        help="Work with an agent's tools from the terminal.",
        description="Work with an agent's tools from the terminal.",
    )
    commands = tools.add_subparsers(dest="tools_command", required=True)
    propose = commands.add_parser(
        "propose",
        help="Raise one exact tool call for an agent, as application code does.",
        description=(
            "Raise one exact call of TOOL for AGENT without a model turn, through "
            "app.propose_tool_call in the project's application in this process. The call "
            "goes through the agent's exposure and tool policy like a model's: it waits in "
            "the inbox when the policy requires approval, is blocked when the policy or "
            "exposure refuses it, and runs here when the policy allows it. A proposal-only "
            "tool is admitted through the proposal's runtime record. Exit status is 0 when "
            "the call completed or waits for approval, and 1 otherwise."
        ),
    )
    propose.add_argument("agent", metavar="AGENT", help="The agent's registered name.")
    propose.add_argument("tool", metavar="TOOL", help="The tool to call.")
    propose.add_argument(
        "--args",
        dest="arguments",
        metavar="JSON|@FILE",
        default="{}",
        help="The call's arguments: a JSON object, or @FILE naming a JSON file (default {}).",
    )
    propose.add_argument(
        "--session-id",
        metavar="ID",
        help=(
            "The proposal session's id. If that session already proposes the same call, "
            "the command prints where it stands and proposes nothing."
        ),
    )
    propose.add_argument(
        "--as",
        dest="as_subject",
        metavar="SUBJECT",
        help="Who the proposal is recorded as (default: local:<your user name>).",
    )
    propose.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show what the policy would decide, recording nothing. The policy sees "
            "metadata cayu:dry_run = true, which only a preview can set."
        ),
    )
    propose.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Propose a call of an external-effect tool without asking. The policy decides "
            "when the call is proposed and may let it run here, so this is required when "
            "stdin isn't a terminal."
        ),
    )
    propose.add_argument(
        "--target",
        metavar="MODULE:FACTORY",
        help="Override project discovery with a module:factory target.",
    )
    add_output_options(propose, formats=("json", "table"), default="table")


def run_tools(args: argparse.Namespace) -> int:
    try:
        with output_destination(args.output):
            return asyncio.run(_run(args))
    except OSError as exc:
        print(f"error: could not write output: {exc}", file=sys.stderr)
        return 1


async def _run(args: argparse.Namespace) -> int:
    try:
        arguments = _arguments(args.arguments)
        subject = args.as_subject or _local_subject()
        return await _propose(args, arguments, subject)
    except (ProjectError, ValueError, KeyError, RuntimeError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        _error(args, str(message) or type(exc).__name__)
        return 1


def _arguments(text: str) -> dict[str, Any]:
    if text.startswith("@"):
        return json_object_file(Path(text[1:]))
    try:
        value = json.loads(text)
    except ValueError:
        raise ValueError("--args must be a JSON object or @FILE.") from None
    if type(value) is not dict:
        raise ValueError("--args must be a JSON object.")
    return value


def _local_subject() -> str:
    try:
        return f"local:{getpass.getuser()}"
    except (KeyError, OSError):
        raise ValueError("Can't tell your user name; name yourself with --as.") from None


async def _propose(args: argparse.Namespace, arguments: dict[str, Any], subject: str) -> int:
    from cayu.sessions.invocation import InvocationOriginClaim

    origin = InvocationOriginClaim(subject=subject)
    project = resolve_project(args.target, command="cayu tools propose")
    with project_context(project.root):
        service = (
            None
            if project.service_target is None
            else build_project_service(
                project.service_target, mode="production", command="Tools propose"
            )
        )
        app = (
            build_project_app(project.target, command="Tools propose")
            if service is None
            else service.cayu_app
        )
        try:
            return await _propose_in(app, args, arguments, origin)
        finally:
            await close_project_app(app)


async def _propose_in(app, args, arguments, origin) -> int:
    from cayu.tools.policy import ToolPolicyDecision

    if args.session_id is not None and not args.dry_run:
        existing = await _existing(app, args, arguments)
        if existing is not None:
            _print_outcome(args, existing, preview=None, existing=True)
            return 0 if existing.status.value in _SUCCESS_STATUSES else 1
    preview = None
    if args.dry_run or not args.yes:
        preview = await app.preview_tool_call(
            args.agent,
            args.tool,
            arguments,
            session_id=args.session_id,
            invocation_origin=origin,
        )
    if preview is not None and args.dry_run:
        _print_preview(args, preview)
        return 1 if preview.decision is ToolPolicyDecision.DENY else 0
    # The preview is advisory: the policy decides again when the call is proposed and can
    # allow what the preview didn't, so consent hangs on the tool's effect, not the preview.
    if preview is not None and preview.effect == "external" and not _confirmed(args, preview):
        return 1
    await app.start_model_policy()
    await app.start_model_slots()
    outcome = await app.propose_tool_call(
        args.agent,
        args.tool,
        arguments,
        session_id=args.session_id,
        invocation_origin=origin,
    )
    _print_outcome(args, outcome, preview=preview, existing=False)
    return 0 if outcome.status.value in _SUCCESS_STATUSES else 1


async def _existing(app, args, arguments):
    """The outcome of the proposal ``--session-id`` names, when it proposes this call."""

    from cayu.sessions._tool_call_proposal_record import (
        proposal_arguments_sha256,
        session_tool_call_proposal,
    )

    outcome = await app.read_tool_call_proposal(args.session_id)
    if outcome is None:
        return None
    session = await app.session_store.load(args.session_id)
    record = None if session is None else session_tool_call_proposal(session.metadata)
    if (
        session is None
        or session.agent_name != args.agent
        or record is None
        or record["tool_name"] != args.tool
        or record["arguments_sha256"] != proposal_arguments_sha256(arguments)
    ):
        raise ValueError(
            f"Session {args.session_id} already proposes a different call; "
            "choose another --session-id."
        )
    return outcome


def _confirmed(args: argparse.Namespace, preview) -> bool:
    """Ask before proposing an external-effect call; refuse without a terminal."""

    question = (
        f"{args.tool} is an external-effect tool: if the tool policy allows the call, it "
        f"runs now, in this process (the preview says it {_WOULD[preview.decision.value]})."
    )
    if not sys.stdin.isatty():
        _error(args, f"{question} Pass --yes to propose it.")
        return False
    print(f"{terminal_safe(question)} Propose it? [y/N] ", end="", file=sys.stderr, flush=True)
    answer = sys.stdin.readline().strip().lower()
    if answer in {"y", "yes"}:
        return True
    _error(args, "Not run.")
    return False


def _print_outcome(args, outcome, *, preview, existing: bool) -> None:
    if args.output_format == "json":
        payload: dict[str, Any] = {
            "schema_version": CLI_SCHEMA_VERSION,
            "existing": existing,
            "outcome": outcome.model_dump(mode="json"),
        }
        if preview is not None:
            payload["preview"] = preview.model_dump(mode="json")
        print(json.dumps(payload, sort_keys=True))
        return
    session = terminal_safe(outcome.session_id)
    status = outcome.status.value.replace("_", " ")
    lead = "Session" if existing else "Proposed in session"
    line = f"{lead} {session}: {terminal_safe(args.tool)} is {status}."
    if outcome.reason:
        line += f" {terminal_safe(outcome.reason)}"
    print(line)
    if outcome.status.value == "pending_approval":
        print(f"It waits for approval: `cayu inbox list --session {session}`.")
    elif outcome.result is not None:
        print(terminal_safe(outcome.result.content))


def _print_preview(args, preview) -> None:
    if args.output_format == "json":
        print(
            json.dumps(
                {
                    "schema_version": CLI_SCHEMA_VERSION,
                    "dry_run": True,
                    "preview": preview.model_dump(mode="json"),
                },
                sort_keys=True,
            )
        )
        return
    would = _WOULD[preview.decision.value]
    where = "" if preview.blocked_by is None else f" ({preview.blocked_by.replace('_', ' ')})"
    line = f"{terminal_safe(preview.tool_name)} {would}{where}."
    if preview.reason:
        line += f" {terminal_safe(preview.reason)}"
    print(line)
    if preview.metadata:
        print(f"Policy metadata: {terminal_safe(json.dumps(preview.metadata, sort_keys=True))}")
    print("Nothing was recorded.")


def _error(args: argparse.Namespace, message: str) -> None:
    if getattr(args, "output_format", None) == "json":
        print(
            json.dumps(
                {"schema_version": CLI_SCHEMA_VERSION, "error": {"message": message}},
                sort_keys=True,
            )
        )
    print(f"error: {terminal_safe(message)}", file=sys.stderr)
