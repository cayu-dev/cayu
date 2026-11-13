"""`cayu session start`: start a conversation as a person, from the terminal.

It opens the project's application in this process, as `cayu inbox` does, and runs
one ``RunRequest``: as ``--subject`` through ``app.access(subject)``, the path the
product routes take, so the application's resource-access policy decides whether the
person may start it with those labels; without ``--subject`` as an operator.

``--scripted-reply`` and ``--scripted-tool-call`` turn it into demonstration data: the
session runs under ``DemoModelProvider`` with those turns, in order, as what the agent
says first (a reply, an ``ask_user`` question, a call its tool policy sends for
approval). Its provider is then ``demo``, which marks it as scripted, and
`cayu serve --dev` or `cayu inbox` can answer and continue it with canned replies.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, cast
from uuid import uuid4

from cayu._resource_access_errors import ResourceAccessDenied
from cayu.cli._output import add_output_options, terminal_safe

CLI_SCHEMA_VERSION = "1"
#: Session metadata that says a session was started by this command, and how.
START_METADATA_KEY = "cayu_session_start"


class _ScriptTurn(argparse.Action):
    """Collect scripted turns of both kinds in the order they were given."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        turns = list(getattr(namespace, self.dest, None) or [])
        if turns and _ends_the_run(turns[-1]):
            # Nothing carries the rest of a script to the process that continues the
            # session, so a turn after one that always ends the run would never play.
            raise argparse.ArgumentError(
                self,
                "a scripted reply ends the agent's turn and an ask_user question waits for "
                "an answer, so nothing can follow either in one command.",
            )
        turns.append(_reply(values) if self.const == "reply" else _call(values))
        setattr(namespace, self.dest, turns)


def add_session_start_parser(commands: Any) -> None:
    start = commands.add_parser(
        "start",
        help="Start a conversation as a person, optionally with scripted turns.",
        description=(
            "Start a session with a first message, in the project's application in this "
            "process. With --subject it runs as that person through app.access(subject), "
            "as the product routes do, so the resource-access policy decides whether they "
            "may start it with those labels; without it, as an operator. With "
            "--scripted-reply and --scripted-tool-call the agent's first turns are those, in "
            "the order given, from the demonstration provider (provider `demo`), so no "
            "model is called: use it for demonstration data and smoke tests. A scripted "
            "ask_user question or a call needing approval leaves the session waiting in "
            "the inbox; `cayu inbox answer`, `cayu inbox approve` or `cayu serve --dev` "
            "continue it with canned, marked replies. It prints the session id, and with "
            "--format json the session."
        ),
    )
    start.add_argument("--agent", required=True, help="The agent to start (its registered name).")
    start.add_argument("--message", required=True, help="The person's first message.")
    start.add_argument(
        "--subject",
        help=(
            "Start it as this person (an authenticated subject the application's "
            "resource-access policy knows). Without it, it starts as an operator."
        ),
    )
    start.add_argument(
        "--label",
        action="append",
        type=_label,
        default=[],
        metavar="KEY=VALUE",
        help="A session label, such as organization=acme. Repeat it for several.",
    )
    start.add_argument("--title", help="The session's title.")
    start.add_argument("--session-id", help="The new session's id (default: a generated one).")
    start.add_argument(
        "--scripted-reply",
        dest="script",
        action=_ScriptTurn,
        const="reply",
        metavar="TEXT",
        help="A scripted turn: the agent replies TEXT. Repeat and mix with tool calls.",
    )
    start.add_argument(
        "--scripted-tool-call",
        dest="script",
        action=_ScriptTurn,
        const="tool_call",
        metavar="TOOL=JSON",
        help=(
            "A scripted turn: the agent calls TOOL with the JSON object's arguments, for "
            'example ask_user=\'{"question": "Which week?", "options": ["17 Nov", "24 Nov"]}\'. '
            "The tool runs, or waits for approval, as the agent's tool policy decides."
        ),
    )
    start.add_argument(
        "--target",
        metavar="MODULE:FACTORY",
        help="Override project discovery with a module:factory target.",
    )
    add_output_options(start, formats=("json", "table"), default="table")


def run_session_start(args: argparse.Namespace) -> int:
    import asyncio

    from cayu.cli.project import ProjectError

    try:
        payload = asyncio.run(_start(args))
    except (ProjectError, ValueError, RuntimeError, NotImplementedError, LookupError) as exc:
        _render_error(args, str(exc) or type(exc).__name__)
        return 1
    if args.output_format == "json":
        print(json.dumps({"schema_version": CLI_SCHEMA_VERSION, **payload}, sort_keys=True))
    else:
        session = payload["session"]
        print(f"Started session {terminal_safe(session['id'])}; it is {session['status']}.")
        if payload["failure"] is not None:
            print(f"It failed: {terminal_safe(payload['failure'])}", file=sys.stderr)
        elif payload["waiting_on"]:
            what = ", ".join(terminal_safe(action["title"]) for action in payload["waiting_on"])
            print(
                f"It waits for a person ({what}): `cayu inbox list --session "
                f"{terminal_safe(session['id'])}`."
            )
    return 1 if payload["failure"] is not None else 0


async def _start(args: argparse.Namespace) -> dict[str, Any]:
    from cayu.cli.project import (
        build_project_app,
        build_project_service,
        close_project_app,
        project_context,
        resolve_project,
    )

    script = args.script or []
    project = resolve_project(args.target, command="cayu session start")
    with project_context(project.root):
        service = (
            None
            if project.service_target is None
            else build_project_service(
                project.service_target, mode="production", command="Session start"
            )
        )
        app = (
            build_project_app(project.target, command="Session start")
            if service is None
            else service.cayu_app
        )
        try:
            # The run happens here, so start what a server starts first.
            await app.start_model_policy()
            await app.start_model_slots()
            request, demo = _request(args, script, app)
            run = app.run if args.subject is None else (await _access(app, args.subject)).run
            failure: str | None = None
            try:
                async for event in run(request):
                    if event.type == "session.failed":
                        failure = _failure_text(event.payload.get("error"))
            except ResourceAccessDenied:
                raise ValueError(
                    f"The resource-access policy doesn't let {args.subject} start a session "
                    "of this agent with these labels."
                ) from None
            assert request.session_id is not None
            session = await app.session_store.load(request.session_id)
            waiting = await _waiting_on(app, request.session_id)
            played = 0 if demo is None else demo.turns_played
            if failure is None and played < len(script):
                # A call the tool policy sent for approval stops the run early.
                failure = (
                    f"The session stopped after {played} of {len(script)} scripted turns, "
                    "and the rest weren't played: nothing carries them to the process that "
                    "continues it. End the script at the first turn that waits for a person."
                )
            session_id = app.project_session_id_for_exposure(request.session_id)
        finally:
            await close_project_app(app)
    if session is None:
        raise RuntimeError(f"Session {session_id} wasn't stored; nothing was started.")
    return {
        "session": {
            "id": session_id,
            "status": session.status.value,
            "agent_name": session.agent_name,
            "provider_name": session.provider_name,
            "model": session.model,
            "labels": dict(session.labels),
            "created_at": session.created_at.isoformat(),
        },
        "subject": args.subject,
        "scripted": bool(script),
        "waiting_on": waiting,
        "failure": failure,
    }


async def _waiting_on(app: Any, session_id: str) -> list[dict[str, Any]]:
    """What the session waits on a person for: its approvals and questions."""

    from cayu.sessions.pending_action_contracts import PendingActionQuery

    listed = await app.session_store.query_pending_actions(
        PendingActionQuery(session_id=session_id, limit=20)
    )
    return [
        {"kind": action.kind.value, "title": action.title, "tool_name": action.tool_name}
        for action in listed.actions
    ]


def _request(args: argparse.Namespace, script: list[Any], app: Any):
    from cayu import Message, RunRequest
    from cayu.sessions import ModelTarget

    target = None
    demo = None
    metadata: dict[str, Any] = {}
    if script:
        from cayu.providers.demo import DEMO_MODEL, DEMO_PROVIDER_NAME, DemoModelProvider

        if DEMO_PROVIDER_NAME in app.list_providers():
            raise ValueError(
                f"The project registers its own `{DEMO_PROVIDER_NAME}` provider, so scripted "
                "turns can't be added to it; start the session without --scripted-reply and "
                "--scripted-tool-call."
            )
        # Only this session names it; it never becomes the application's default.
        demo = DemoModelProvider(script)
        app._provider_registry.register(demo, claim_default=False)
        target = ModelTarget(provider_name=DEMO_PROVIDER_NAME, model=DEMO_MODEL)
        metadata[START_METADATA_KEY] = {"scripted_turns": len(script)}
    request = RunRequest(
        agent_name=args.agent,
        session_id=args.session_id or f"session-{uuid4().hex}",
        title=args.title,
        labels=dict(args.label),
        messages=[Message.text("user", args.message)],
        target=target,
        metadata=metadata,
    )
    return request, demo


async def _access(app: Any, subject: str):
    if app.resource_access_policy is None:
        raise ValueError(
            "This application has no resource-access policy, so it can't start a session as "
            f"{subject}; leave out --subject to start it as an operator."
        )
    try:
        return await app.access(subject)
    except ResourceAccessDenied:
        raise ValueError(f"The resource-access policy refuses {subject}.") from None


def _failure_text(error: object) -> str:
    if isinstance(error, dict):
        details = cast("dict[str, Any]", error)
        return str(details.get("message") or details.get("type") or "the run failed")
    return str(error or "the run failed")


def _render_error(args: argparse.Namespace, message: str) -> None:
    if getattr(args, "output_format", "table") == "json":
        print(
            json.dumps(
                {
                    "schema_version": CLI_SCHEMA_VERSION,
                    "error": {"code": "SESSION_START_FAILED", "message": message},
                },
                sort_keys=True,
            )
        )
        return
    print(f"error: {terminal_safe(message)}", file=sys.stderr)


def _label(value: str) -> tuple[str, str]:
    key, separator, label = value.partition("=")
    if not separator or not key.strip():
        raise argparse.ArgumentTypeError("a label is KEY=VALUE.")
    return key.strip(), label


def _ends_the_run(turn: Any) -> bool:
    return isinstance(turn, str) or turn.payload.get("name") == "ask_user"


def _reply(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentError(None, "--scripted-reply must not be blank.")
    return value


def _call(value: str):
    from cayu.providers import ModelStreamEvent

    name, separator, raw = value.partition("=")
    if not separator or not name.strip():
        raise argparse.ArgumentError(None, "--scripted-tool-call is TOOL=JSON.")
    try:
        arguments = json.loads(raw)
    except ValueError as exc:
        raise argparse.ArgumentError(
            None, f"--scripted-tool-call {name.strip()}: the arguments aren't JSON ({exc})."
        ) from None
    if not isinstance(arguments, dict):
        raise argparse.ArgumentError(
            None, f"--scripted-tool-call {name.strip()}: the arguments must be a JSON object."
        )
    return ModelStreamEvent.tool_call(name=name.strip(), arguments=arguments)
