"""`cayu inbox`: review and decide what an agent is waiting on a person for.

The commands work the same against the project's own application in this process and
against a Cloud Agent (``--application``), through ``cayu.cli._inbox_client``. Arguments
show only through the application's human-review policy (``docs/human-review.md``), and
approving or denying settles the item through the inbox's settle route.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import html
import json
import sys
import unicodedata
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from cayu.cli._inbox_client import (
    CloudInbox,
    InboxClient,
    InboxClientError,
    LocalInbox,
)
from cayu.cli._output import add_output_options, output_destination
from cayu.cli.project import (
    ProjectError,
    build_project_app,
    build_project_service,
    close_project_app,
    project_context,
    resolve_project,
)

CLI_SCHEMA_VERSION = "1"
_KINDS = ("approval", "question", "recovery", "knowledge")
#: The purpose `ToolArgumentReview` accepts unless it is configured otherwise.
_DEFAULT_PURPOSE = "approvals"
_REASON_MAX_CHARS = 4096
_CELL_MAX_CHARS = 60


def add_inbox_parser(subparsers: Any) -> None:
    """Register ``cayu inbox``."""

    inbox = subparsers.add_parser(
        "inbox",
        help="Review and decide what an agent is waiting on a person for.",
        description=(
            "List an agent's inbox (tool approvals, questions, recovery items and knowledge "
            "proposals), show one item with the arguments the application's human-review "
            "policy lets you see, and approve or deny tool approvals. Without --application "
            "it opens the project's application and its store in this process; with "
            "--application NAME it reads that Cayu Cloud Agent with your `cayu cloud login`."
        ),
    )
    commands = inbox.add_subparsers(dest="inbox_command", required=True)

    listing = commands.add_parser(
        "list",
        help="List what is waiting, oldest first.",
        description="List what is waiting, oldest first. Use an item's id with `cayu inbox show`.",
    )
    listing.add_argument("--kind", choices=_KINDS, help="Only items of this kind.")
    listing.add_argument("--agent", help="Only this agent's items (exact name).")
    listing.add_argument("--session", help="Only this session's items (public session id).")
    _add_target_options(listing)

    show = commands.add_parser(
        "show",
        help="Show one item and the arguments you may see.",
        description=(
            "Show one item's state. For a pending approval or question it also shows the "
            "application's human-review view for you: the arguments its policy shows, the "
            "calls the decision covers, and guidance. Nothing is shown that the policy "
            "withholds; without a policy the inbox shows no arguments."
        ),
    )
    show.add_argument("item_id", metavar="ITEM_ID")
    show.add_argument(
        "--full",
        action="store_true",
        help="Also read each value the review shows only in part, whole.",
    )
    _add_review_options(show)
    _add_target_options(show)

    for decision, verb, what in (
        ("approve", "Approve", "runs the call with the arguments the agent proposed"),
        ("deny", "Deny", "refuses the call; the agent is told it was denied"),
    ):
        command = commands.add_parser(
            decision,
            help=f"{verb} a pending tool approval.",
            description=(
                f"{verb} a pending tool approval: it {what}. When the application has a "
                "human-review policy, the decision is bound to the review you are issued "
                "now, so run `cayu inbox show` first. Approving needs a review that shows "
                "the call; a withheld review can only be denied. Locally the session then "
                "continues in this process until it pauses or ends."
            ),
        )
        command.add_argument("item_id", metavar="ITEM_ID")
        command.add_argument(
            "--reason",
            help=f"Why, recorded with the decision (at most {_REASON_MAX_CHARS} characters).",
        )
        command.add_argument(
            "--wait-seconds",
            type=_wait_seconds,
            default=3600.0,
            help=(
                "Locally, how long to wait for the work the decision continues in this process "
                "before stopping it; a stopped session can be resumed (default 3600)."
            ),
        )
        command.add_argument(
            "--claim-task",
            action="store_true",
            help=(
                "The session belongs to a task whose worker let it go while it waits: let the "
                "runtime claim the task's continuation, as the dashboard's option does. A task "
                "another worker holds is refused."
            ),
        )
        _add_review_options(command)
        _add_target_options(command)


def _add_review_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--purpose",
        help=(
            "The human-review purpose to ask for (default: the application's own, or "
            f'"{_DEFAULT_PURPOSE}" for a Cloud Agent).'
        ),
    )


def _add_target_options(parser: argparse.ArgumentParser) -> None:
    where = parser.add_argument_group("where the inbox is")
    target = where.add_mutually_exclusive_group()
    target.add_argument(
        "--application",
        metavar="NAME",
        help="A Cayu Cloud Agent's name or slug, read with your `cayu cloud login`.",
    )
    target.add_argument(
        "--target",
        metavar="MODULE:FACTORY",
        help="Override project discovery with a module:factory target (local only).",
    )
    where.add_argument(
        "--context",
        type=Path,
        help="Private Cayu Cloud context JSON (with --application; default: the selected one).",
    )
    where.add_argument(
        "--timeout-seconds",
        type=_positive_seconds,
        default=60.0,
        help="How long one Cayu Cloud request may take (default 60).",
    )
    where.add_argument(
        "--reviewer",
        metavar="SUBJECT",
        help=(
            "Who local decisions are recorded as, and who the human-review policy sees "
            "(default: local:<your user name>). A Cloud Agent records your login."
        ),
    )
    add_output_options(parser, formats=("json", "table"), default="table")


def run_inbox(args: argparse.Namespace) -> int:
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
        reason = getattr(args, "reason", None)
        if reason is not None:
            _check_reason(reason)
        if args.application is not None and args.reviewer is not None:
            raise ValueError("--reviewer applies to the local inbox; a Cloud Agent records you.")
        async with _inbox(args) as inbox:
            if args.inbox_command == "list":
                return await _list(args, inbox)
            if args.inbox_command == "show":
                return await _show(args, inbox)
            return await _decide(args, inbox)
    except InboxClientError as exc:
        _render_error(args, exc.code, str(exc), item=exc.item)
    except (CloudApiError, CloudAuthError, CloudCommandError) as exc:
        _render_error(args, exc.category, str(exc))
    except ProjectError as exc:
        _render_error(args, "project_unavailable", str(exc))
    except ValueError as exc:
        _render_error(args, "invalid_input", str(exc))
    return 1


@asynccontextmanager
async def _inbox(args: argparse.Namespace) -> AsyncIterator[InboxClient]:
    if args.application is not None:
        from cayu.cli.cloud import cloud_client, resolve_application_id

        client = cloud_client(context=args.context, timeout_seconds=args.timeout_seconds)
        application_id = await asyncio.to_thread(resolve_application_id, client, args.application)
        yield CloudInbox(client, application_id)
        return
    project = resolve_project(args.target, command="cayu inbox")
    with project_context(project.root):
        # A project's public service gives the sessions it started their continuation
        # policies, so its sessions are continued as `cayu serve` continues them.
        service = (
            None
            if project.service_target is None
            else build_project_service(project.service_target, mode="production", command="Inbox")
        )
        app = (
            build_project_app(project.target, command="Inbox")
            if service is None
            else service.cayu_app
        )
        deciding = args.inbox_command in {"approve", "deny"}
        local: LocalInbox | None = None
        try:
            reviewer = args.reviewer or _local_reviewer()
            if deciding:
                # A decision runs the agent here, so start what a server starts first;
                # closing the app stops it.
                await app.start_model_policy()
            local = LocalInbox(
                app,
                reviewer=reviewer,
                continuation_loop_policies=None
                if service is None
                else service._continuation_loop_policies,
                wait_seconds=args.wait_seconds if deciding else None,
            )
            yield local
        finally:
            if local is not None:
                await local.aclose()
            # Closing waits for the work a decision continued in this process, such as a
            # parent the approved child resumes, before stopping what is left.
            if deciding:
                remaining = None if local is None else local.remaining_wait_seconds()
                # Shutdown gets what the continuation wait left, and a moment at least.
                timeout = args.wait_seconds if remaining is None else max(remaining, 0.05)
                await close_project_app(app, timeout_s=timeout)
            else:
                await close_project_app(app)


async def _list(args: argparse.Namespace, inbox: InboxClient) -> int:
    listing = await inbox.list_items(kind=args.kind, session_id=args.session, agent_name=args.agent)
    if args.output_format == "json":
        _print_json(
            {
                "where": inbox.where,
                "items": listing.items,
                "complete": listing.complete,
                "notes": listing.notes,
            }
        )
        return 0
    if not listing.items:
        print("Nothing is waiting.")
    else:
        rows = [
            {
                "kind": item.get("kind"),
                "agent": item.get("agent_name"),
                "session": item.get("session_id"),
                "what": _summary(item),
                "created": item.get("created_at"),
                "id": item.get("id"),
            }
            for item in listing.items
        ]
        _print_table(("kind", "agent", "session", "what", "created", "id"), rows)
    for note in listing.notes:
        print(f"note: {_plain(note)}", file=sys.stderr)
    if not listing.complete:
        print("note: this listing may not hold every item.", file=sys.stderr)
    return 0


async def _show(args: argparse.Namespace, inbox: InboxClient) -> int:
    state = await inbox.item(args.item_id)
    item = state.get("item") if state.get("status") == "pending" else None
    review: dict[str, Any] | None = None
    review_error: InboxClientError | None = None
    long_text: dict[str, dict[str, Any]] = {}
    if isinstance(item, dict) and _has_review(item):
        try:
            review = await inbox.review(
                args.item_id, str(item["session_id"]), purpose=_purpose(args, inbox)
            )
        except InboxClientError as exc:
            review_error = exc
        if review is not None and not _review_matches(review, item):
            review, review_error = None, _review_mismatch()
        if review is not None and (await inbox.item(args.item_id)).get("status") != "pending":
            # Settled while the review was read: the session's review may be another one's.
            review, review_error = None, _review_mismatch()
        if review is not None and args.full and review.get("reference") is not None:
            for long_field in review.get("long_fields") or ():
                key = str(long_field.get("key"))
                long_text[key] = await inbox.long_field(
                    args.item_id,
                    str(item["session_id"]),
                    reference=review["reference"],
                    key=key,
                )
    if args.output_format == "json":
        _print_json(
            {
                "where": inbox.where,
                "item": state,
                "review": None if review is None else _public_review(review),
                "review_error": None
                if review_error is None
                else {"code": review_error.code, "message": str(review_error)},
                "long_fields": {key: _public_review(value) for key, value in long_text.items()},
            }
        )
        return 0
    _print_state(state)
    if isinstance(item, dict) and (
        item.get("kind") in {"approval", "recovery"}
        or review is not None
        or review_error is not None
    ):
        print()
        _print_arguments(item, review, review_error, long_text)
    return 0


async def _decide(args: argparse.Namespace, inbox: InboxClient) -> int:
    decision = args.inbox_command
    state = await inbox.item(args.item_id)
    item = state.get("item")
    if state.get("status") != "pending" or not isinstance(item, dict):
        raise InboxClientError(
            "not_pending",
            f"This item is {state.get('status')}, so there is nothing to {decision}.",
            item=state,
        )
    if item.get("kind") != "approval":
        raise InboxClientError(
            "not_an_approval",
            f"`cayu inbox {decision}` settles tool approvals; this item is a {item.get('kind')}.",
        )
    body: dict[str, Any] = {"decision": decision}
    if args.reason is not None:
        body["reason"] = args.reason
    if args.claim_task:
        body["task_continuation"] = "claim"
    if _has_review(item):
        reference = await _decision_reference(args, inbox, item, decision)
        if reference is not None:
            body["review_reference"] = reference
    answer = await inbox.settle(args.item_id, body)
    settled: dict[str, Any] = answer["item"] if isinstance(answer.get("item"), dict) else {}
    if inbox.where == "local" and args.output_format != "json":
        print(
            f"Recorded. Session {_plain(item.get('session_id'))} continues in this process...",
            file=sys.stderr,
        )
    session_status = await inbox.after_settle(item.get("session_id"))
    if args.output_format == "json":
        _print_json(
            {
                "where": inbox.where,
                "decision": decision,
                "replayed": answer.get("replayed") is True,
                "item": settled,
                "session_status": session_status,
            }
        )
        return 0
    settlement = settled.get("settlement") or {}
    who = (settlement.get("decided_by") or {}).get("subject")
    done = "Approved" if decision == "approve" else "Denied"
    again = " (already recorded)" if answer.get("replayed") is True else ""
    print(f"{done} {args.item_id}{again}" + (f" as {_plain(who)}." if who else "."))
    if session_status is not None:
        print(f"Session {_plain(item.get('session_id'))} is {session_status}.")
    return 0


async def _decision_reference(
    args: argparse.Namespace, inbox: InboxClient, item: dict[str, Any], decision: str
) -> dict[str, Any] | None:
    """The review reference the decision is bound to, or None to settle without one."""

    try:
        review = await inbox.review(
            args.item_id, str(item["session_id"]), purpose=_purpose(args, inbox)
        )
    except InboxClientError as exc:
        # A Cloud or an Agent that serves no reviews, for an item that doesn't say its
        # application has a policy: send the decision without a reference and let the
        # application answer.
        if (
            exc.code in {"review_unsupported", "review_not_served"}
            and item.get("arguments_unavailable_reason") != "quarantined"
        ):
            return None
        raise
    if not _review_matches(review, item):
        raise _review_mismatch()
    reference = review.get("reference")
    if not isinstance(reference, dict):
        raise InboxClientError(
            "review_unavailable",
            f"No current review can authorize this decision: {review.get('guidance')}",
        )
    if decision == "approve" and review.get("status") != "permitted":
        raise InboxClientError(
            "review_withheld",
            "The review withholds this call's content, so it can't be approved: "
            f"{review.get('guidance')} You can deny it.",
        )
    return reference


def _has_review(item: dict[str, Any]) -> bool:
    """Whether the item's application may show it through a human-review policy."""

    if item.get("kind") not in {"approval", "question"} or not item.get("session_id"):
        return False
    return item.get("arguments_unavailable_reason") != "no_review_policy"


def _review_matches(review: dict[str, Any], item: dict[str, Any]) -> bool:
    expected = "tool_approval" if item.get("kind") == "approval" else "user_input"
    return review.get("kind") in {None, expected}


def _review_mismatch() -> InboxClientError:
    return InboxClientError(
        "review_mismatch",
        "The session's current review is for another interaction. List the inbox again.",
    )


def _purpose(args: argparse.Namespace, inbox: InboxClient) -> str:
    if args.purpose is not None:
        return args.purpose
    return inbox.purpose or _DEFAULT_PURPOSE


def _public_review(review: dict[str, Any]) -> dict[str, Any]:
    """The review as printed: its opaque reference stays inside the CLI."""

    return {key: value for key, value in review.items() if key != "reference"}


def _check_reason(reason: str) -> None:
    if not reason.strip() or len(reason) > _REASON_MAX_CHARS:
        raise ValueError(f"--reason must be 1 to {_REASON_MAX_CHARS} characters and not blank.")


def _print_state(state: dict[str, Any]) -> None:
    item = state.get("item") if isinstance(state.get("item"), dict) else {}
    rows: list[tuple[str, object]] = [
        ("item", state.get("id")),
        ("kind", state.get("kind")),
        ("status", state.get("status")),
        ("agent", state.get("agent_name")),
        ("session", state.get("session_id")),
    ]
    if item:
        rows += [
            ("title", item.get("title")),
            ("tool", item.get("tool_name")),
            ("detail", item.get("detail")),
            ("question", item.get("question")),
            ("options", ", ".join(item.get("options") or ()) or None),
            ("created", item.get("created_at")),
            ("decisions", ", ".join(item.get("decisions") or ()) or None),
        ]
        deadline = item.get("execution_deadline")
        if isinstance(deadline, dict):
            rows.append(("deadline", deadline.get("expires_at")))
    settlement = state.get("settlement")
    if isinstance(settlement, dict):
        decided_by = settlement.get("decided_by") or {}
        rows += [
            ("decision", settlement.get("decision")),
            ("decided by", decided_by.get("subject")),
            ("decided at", settlement.get("decided_at")),
            ("reason", settlement.get("reason")),
        ]
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        if value is not None:
            print(f"{name.ljust(width)}  {_plain(value)}")


def _print_arguments(
    item: dict[str, Any],
    review: dict[str, Any] | None,
    review_error: InboxClientError | None,
    long_text: dict[str, dict[str, Any]],
) -> None:
    if review is None:
        if item.get("arguments_unavailable_reason") == "no_review_policy":
            print(
                "Arguments are not shown: the application has no human-review policy. "
                "Configure one, such as ToolArgumentReview, to show selected arguments."
            )
        elif review_error is not None:
            print(f"Arguments are not shown: {_plain(str(review_error))}")
        else:
            print("Arguments are not shown.")
        return
    print(f"Review ({review.get('status')}): {_plain(review.get('guidance'))}")
    fields = review.get("fields") or ()
    if fields:
        print("Arguments the policy shows:")
        for display in fields:
            print(f"  {_plain(display.get('label'))}: {_continued(display.get('text'))}")
    elif review.get("status") == "permitted":
        print("The policy shows no arguments for this call.")
    calls = review.get("calls") or ()
    if calls:
        print("Calls this decision covers:")
        for call in calls:
            print(
                f"  {_plain(call.get('tool_name'))} ({_plain(call.get('tool_call_id'))}): "
                f"{_plain(call.get('on_grant'))}"
            )
    for long_field in review.get("long_fields") or ():
        key = str(long_field.get("key"))
        read = long_text.get(key)
        label = _plain(long_field.get("label"))
        if read is None:
            print(
                f"{label} is shown only in part ({long_field.get('characters')} characters); "
                "run with --full to read it whole."
            )
        elif read.get("status") == "permitted":
            print(f"{label}, whole:")
            print(f"    | {_continued(read.get('text'))}")
        else:
            print(f"{label} can't be read whole ({read.get('status')}).")


def _summary(item: dict[str, Any]) -> str:
    kind = item.get("kind")
    if kind in {"approval", "recovery"} and item.get("tool_name"):
        text = f"{item['tool_name']}: {item.get('detail') or item.get('title') or ''}"
    elif kind == "question" and item.get("question"):
        text = str(item["question"])
    else:
        text = str(item.get("title") or "")
    text = _plain(text).replace("\n", " ").replace("\t", " ").strip(" :")
    return text if len(text) <= _CELL_MAX_CHARS else text[: _CELL_MAX_CHARS - 3] + "..."


def _continued(value: object) -> str:
    """A review value whose later lines are marked as its own, so a value can't print lines
    that look like other arguments the policy shows."""

    return _review_text(value).replace("\n", "\n    | ")


def _local_reviewer() -> str:
    try:
        return f"local:{getpass.getuser()}"
    except (KeyError, OSError):
        raise ValueError("Can't tell your user name; name yourself with --reviewer.") from None


def _review_text(value: object) -> str:
    """A review field: HTML-escaped text from the policy, shown as plain text."""

    return _plain(html.unescape(value) if isinstance(value, str) else value)


def _plain(value: object) -> str:
    """Untrusted text made safe for a terminal: control and format characters are spaces.

    Tabs and line breaks stay. Everything else in Unicode's Cc and Cf categories (escape
    sequences, 8-bit CSI, bidi overrides) could change what the terminal shows.
    """

    if value is None:
        return "-"
    return "".join(
        " "
        if character not in "\t\n" and unicodedata.category(character) in {"Cc", "Cf"}
        else character
        for character in str(value)
    )


def _print_table(headers: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    rendered = [[_plain(row.get(header)).replace("\n", " ") for header in headers] for row in rows]
    widths = [
        max(len(header), *(len(row[index]) for row in rendered))
        for index, header in enumerate(headers)
    ]
    print("  ".join(header.upper().ljust(widths[index]) for index, header in enumerate(headers)))
    for row in rendered:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)).rstrip())


def _print_json(payload: dict[str, Any]) -> None:
    print(
        json.dumps(
            {"schema_version": CLI_SCHEMA_VERSION, **payload}, ensure_ascii=False, sort_keys=True
        )
    )


def _render_error(
    args: argparse.Namespace, code: str, message: str, *, item: dict[str, Any] | None = None
) -> None:
    if getattr(args, "output_format", "table") == "json":
        _print_json({"error": {"code": code, "message": message}, "item": item})
        return
    print(f"error: {_plain(message)}", file=sys.stderr)
    settlement = (item or {}).get("settlement")
    if isinstance(settlement, dict):
        who = (settlement.get("decided_by") or {}).get("subject") or "someone"
        print(
            f"It was settled: {_plain(settlement.get('decision'))} by {_plain(who)} at "
            f"{_plain(settlement.get('decided_at'))}.",
            file=sys.stderr,
        )


def _positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("seconds must be a number.") from None
    if not 0 < seconds <= 3600:
        raise argparse.ArgumentTypeError("seconds must be more than 0 and at most 3600.")
    return seconds


def _wait_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("seconds must be a number.") from None
    if not 0 < seconds <= 86_400:
        raise argparse.ArgumentTypeError("seconds must be more than 0 and at most 86400.")
    return seconds
