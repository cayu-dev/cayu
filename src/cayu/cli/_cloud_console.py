"""`cayu cloud console`: drive Cayu Cloud's Console projects without the web app.

Console is Cloud's workspace where a person builds an Agent with a coding agent. Each command
calls Cloud's Console routes (`/v1/console/...`) with the `cayu cloud login` (or a selected
context's API key) and prints one JSON document; `events --follow` prints each step as one
JSON line instead. `pull` and `push` move a project's source between Cloud and a local
directory (`_cloud_console_source`), so a local coding agent can change it. `deploy` and
`deploy-merged` take `--idempotency-key` (`_cloud_writes`).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_console_source import (
    CHECKOUT_METADATA,
    ConsoleCheckout,
    ConsoleError,
    console_source_problem,
    source_archive,
    source_digest,
    source_files,
    source_paths,
    source_tree,
    write_tree,
)
from cayu.cli._cloud_project import SOURCE_BUNDLE_MAX_BYTES
from cayu.cli._cloud_writes import (
    CloudWriteError,
    add_idempotency_key_argument,
    add_message_arguments,
    message_text,
    send_write,
    write_key,
)
from cayu.cli._output import terminal_safe

#: The detail of every Console route's 404 where Console is not available.
CONSOLE_UNAVAILABLE = "Console is not available in this environment."
#: The commands that change a Console project, which belongs to one Organization.
PROJECT_WRITES = (
    "rename",
    "delete",
    "stop",
    "retry",
    "wake",
    "instruct",
    "answer",
    "push",
    "restore",
    "pull-request",
    "deploy-merged",
)
EVENTS_FOLLOW_POLL_SECONDS = 1.0
EVENTS_MAX_TRANSIENT_FAILURES = 6
# A retry's backoff starts from at least this, however short the polling interval.
_EVENTS_MIN_BACKOFF_SECONDS = 1.0
_EVENTS_MAX_BACKOFF_SECONDS = 30.0
# A push through a workspace installs dependencies and restarts the preview.
PUSH_WAIT_SECONDS = 900.0
PUSH_POLL_SECONDS = 2.0
_PENDING_REVISION_JOB = frozenset({"queued", "running"})
_COMMENT = re.compile(r"([1-9][0-9]*):(.+?):([1-9][0-9]*):(.+)", re.DOTALL)


def add_console_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    application: Callable[[str], str],
    seconds: Callable[[str], float],
) -> None:
    """Add `console` to `cayu cloud`; ``application`` parses an Agent slug, ``seconds`` the
    wait arguments."""

    console = commands.add_parser(
        "console",
        help="Build Agents in Cayu Cloud's Console, and move their source to and from here.",
        description=(
            "Drive Console projects, where a coding agent builds an Agent in Cayu Cloud: "
            "create and instruct them, read their steps and revisions, pull a project's source "
            "into a directory and push it back, and deploy it. Every command prints one JSON "
            "document."
        ),
    )
    actions = console.add_subparsers(dest="console_command", required=True)

    actions.add_parser(
        "list",
        help="Your Organization's Console projects.",
        description="Your Organization's Console projects.",
    )
    actions.add_parser(
        "insights",
        help="Console's numbers for your Organization.",
        description="How much your Organization has built and published with Console, by month.",
    ).add_argument(
        "--tz",
        help="Count months in this IANA time zone, such as America/Los_Angeles (default UTC).",
    )
    activity = actions.add_parser(
        "activity",
        help="What happened in your Organization's projects, newest first.",
        description="What happened in your Organization's Console projects, newest first.",
    )
    activity.add_argument("--limit", type=_positive, default=40)
    activity.add_argument("--cursor", help="The `next_cursor` of the page before.")

    create = actions.add_parser(
        "create",
        help="Create a project, or open an Agent's Release in one.",
        description=(
            "Create a project from Console's starting source, or with --from-agent open an "
            "existing Agent's selected Release (or --release) as the project's first revision."
        ),
    )
    create.add_argument("--name", help="The project's name.")
    create.add_argument(
        "--from-agent",
        dest="application",
        metavar="AGENT",
        help="The Agent to open: its name, slug or id.",
    )
    create.add_argument("--release", help="With --from-agent: the Release to open.")
    rename = actions.add_parser("rename", help="Rename a project.", description="Rename a project.")
    rename.add_argument("project_id")
    rename.add_argument("--name", required=True)
    actions.add_parser(
        "delete",
        help="Delete a project.",
        description="Delete a project. Its Agent and the Agent's Releases stay.",
    ).add_argument("project_id")
    actions.add_parser(
        "show",
        help="One project.",
        description="One project: its conversation, revisions, files and workspace.",
    ).add_argument("project_id")
    for name, summary in (
        ("stop", "Ask Console to stop the project's running turn."),
        ("retry", "Run the project's latest instruction again."),
        ("wake", "Wake or create the project's workspace."),
        ("preview", "Issue a short-lived URL that shows the project's preview."),
    ):
        actions.add_parser(name, help=summary, description=summary).add_argument("project_id")

    instruct = actions.add_parser(
        "instruct",
        help="Tell the project's coding agent what to do next.",
        description="Send the project's coding agent an instruction, which starts a turn.",
    )
    instruct.add_argument("project_id")
    add_message_arguments(instruct)
    instruct.add_argument(
        "--comment",
        action="append",
        type=_review_comment,
        default=[],
        metavar="REVISION:PATH:LINE:TEXT",
        help="A review comment on one line of a revision's file; repeat for each.",
    )
    answer = actions.add_parser(
        "answer",
        help="Answer the question the coding agent paused on.",
        description="Answer the question the project's coding agent paused its turn on.",
    )
    answer.add_argument("project_id")
    add_message_arguments(answer)

    source = actions.add_parser(
        "source",
        help="Download the project's latest source as a .tar.gz.",
        description=(
            "Download the latest revision's source, a gzip tar under source/, into a file."
        ),
    )
    source.add_argument("project_id")
    source.add_argument("--output", required=True, type=Path, help="The file to write.")
    pull = actions.add_parser(
        "pull",
        help="Write the project's latest revision into a directory.",
        description=(
            "Write the project's latest revision into a directory, with a record of it in "
            f"{CHECKOUT_METADATA.as_posix()}, for a local coding agent to change and `push` "
            "to record."
        ),
    )
    pull.add_argument("project_id")
    pull.add_argument("--dir", type=Path, default=Path("."), dest="directory")
    pull.add_argument(
        "--force",
        action="store_true",
        help=(
            "Pull into a non-empty directory holding an earlier pull of this project, "
            "replacing its source; what a push leaves out (.git, .venv, .env, ...) stays."
        ),
    )
    push = actions.add_parser(
        "push",
        help="Record a pulled directory's source as the project's next revision.",
        description=(
            "Record a pulled directory's source as the project's next revision, refused when "
            "the project changed since the pull. When the project's workspace is running, "
            "Console writes the source into it first, and push waits for that."
        ),
    )
    push.add_argument("project_id")
    push.add_argument("--dir", type=Path, default=Path("."), dest="directory")
    push.add_argument("--summary", required=True, help="What the change does.")
    push.add_argument("--poll-seconds", type=seconds, default=PUSH_POLL_SECONDS)
    push.add_argument("--wait-seconds", type=seconds, default=PUSH_WAIT_SECONDS)

    events = actions.add_parser(
        "events",
        help="The coding agent's steps.",
        description=(
            "A page of the project's coding-agent steps after --after, with the `next_after` "
            "to read on from. With --follow, each new step as one JSON line until the "
            "project's turn ends."
        ),
    )
    events.add_argument("project_id")
    events.add_argument("--after", type=_non_negative, default=0)
    events.add_argument("--limit", type=_positive, default=200)
    events.add_argument(
        "--follow",
        action="store_true",
        help="Keep printing new steps, one JSON object per line, until the turn ends.",
    )
    events.add_argument(
        "--poll-seconds",
        type=seconds,
        help=f"With --follow: seconds between reads (default {EVENTS_FOLLOW_POLL_SECONDS:g}).",
    )
    events.set_defaults(_cloud_render=_render_events)
    diff = actions.add_parser(
        "diff",
        help="What a revision changed.",
        description="The per-file changes a revision made to the one before it.",
    )
    diff.add_argument("project_id")
    diff.add_argument("revision", type=_positive)
    restore = actions.add_parser(
        "restore",
        help="Make a revision's source the latest again.",
        description="Make a revision's source the latest again, as a new revision.",
    )
    restore.add_argument("project_id")
    restore.add_argument("revision", type=_positive)

    deploy = actions.add_parser(
        "deploy",
        help="Publish the project's latest revision as a Release of its Agent.",
        description=(
            "Publish the project's latest revision as a Release of its Agent (created on the "
            "first deploy). Once built and smoke-tested, Cayu Cloud promotes it, so the Agent "
            "serves it."
        ),
    )
    deploy.add_argument("project_id")
    deploy.add_argument(
        "--application",
        type=application,
        help=(
            "The slug of the Agent to deploy to, for a project not yet linked to one; Cayu "
            "Cloud creates the Agent when none has it yet."
        ),
    )
    deploy.add_argument(
        "--no-promote",
        dest="promote",
        action="store_false",
        help=(
            "Stage the Release instead: it stops once smoke-tested, for someone to promote "
            "later with `cayu cloud deployment promote`."
        ),
    )
    add_idempotency_key_argument(deploy)
    pull_request = actions.add_parser(
        "pull-request",
        help="Deliver a project opened from a repository as a pull request.",
        description=(
            "For a project opened from an Agent built from a repository: deliver its latest "
            "revision (or --revision) as a commit on the project's branch and its pull "
            "request, found or opened. The same revision again answers the same pull request."
        ),
    )
    pull_request.add_argument("project_id")
    pull_request.add_argument("--revision", type=_positive)
    pull_request.add_argument(
        "--new-branch",
        action="store_true",
        help="Start a new branch from the same base, for a branch someone else changed.",
    )
    pull_request.add_argument(
        "--status",
        action="store_true",
        help="Deliver nothing: show the pull request's state as GitHub reports it now.",
    )
    deploy_merged = actions.add_parser(
        "deploy-merged",
        help="Deploy the project's merged pull request.",
        description=(
            "Once the project's pull request is merged, deploy its merge commit as a new "
            "Release of the same Agent, built from the repository."
        ),
    )
    deploy_merged.add_argument("project_id")
    deploy_merged.add_argument(
        "--idempotency-key",
        help=(
            "Send the deploy under this key. Without one Cayu Cloud keys it by the pull "
            "request, so deploying the same merge again answers the same Release."
        ),
    )


def console_command(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    is_transient: Callable[[CloudApiError], bool],
    transient_limit: int,
    validate_wait: Callable[[float, float], None],
) -> dict[str, Any]:
    """Run one `cayu cloud console` command and return its JSON answer.

    A changing command's `--application` is already the Agent's id.
    """

    try:
        return _command(
            arguments,
            client=client,
            is_transient=is_transient,
            transient_limit=transient_limit,
            validate_wait=validate_wait,
        )
    except CloudApiError as exc:
        raise _refusal(exc) from None


def _command(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    is_transient: Callable[[CloudApiError], bool],
    transient_limit: int,
    validate_wait: Callable[[float, float], None],
) -> dict[str, Any]:
    action = arguments.console_command
    if action == "list":
        return _answer(action, client.request("GET", "/v1/console/projects"))
    if action == "insights":
        query = None if not arguments.tz else {"tz": arguments.tz}
        return _answer(action, client.request("GET", "/v1/console/insights", query=query))
    if action == "activity":
        query = {"limit": str(arguments.limit)}
        if arguments.cursor is not None:
            query["cursor"] = arguments.cursor
        return _answer(action, client.request("GET", "/v1/console/activity", query=query))
    if action == "create":
        fields = {
            "name": arguments.name,
            "application_id": arguments.application,
            "release_id": arguments.release,
        }
        payload = {name: value for name, value in fields.items() if value is not None}
        return _answer(action, client.request("POST", "/v1/console/projects", payload=payload))
    project = _project(arguments.project_id)
    if action == "rename":
        payload = {"name": arguments.name}
        return _answer(action, client.request("PATCH", project, payload=payload))
    if action == "delete":
        client.request("DELETE", project)
        return _answer(action, {"deleted": arguments.project_id})
    if action == "show":
        return _answer(action, client.request("GET", project))
    if action in {"stop", "retry", "wake", "preview"}:
        route = {"wake": "workspace", "preview": "preview-session"}.get(action, action)
        return _answer(action, client.request("POST", f"{project}/{route}"))
    if action in {"instruct", "answer"}:
        payload: dict[str, object] = {"content": message_text(arguments)}
        if action == "instruct" and arguments.comment:
            payload["comments"] = arguments.comment
        route = "messages" if action == "instruct" else "answer"
        return _answer(action, client.request("POST", f"{project}/{route}", payload=payload))
    if action == "source":
        return _answer(action, _source(client, project, arguments.output))
    if action == "pull":
        return _answer(action, _pull(arguments, client=client, project=project))
    if action == "push":
        validate_wait(arguments.poll_seconds, arguments.wait_seconds)
        return _answer(
            action,
            _push(
                arguments,
                client=client,
                project=project,
                is_transient=is_transient,
                transient_limit=transient_limit,
            ),
        )
    if action == "events":
        if not arguments.follow:
            if arguments.poll_seconds is not None:
                raise CloudApiError("invalid_input", "--poll-seconds applies only with --follow.")
            query = {"after": str(arguments.after), "limit": str(arguments.limit)}
            return _answer(action, client.request("GET", f"{project}/events", query=query))
        return _follow_events(arguments, client=client, project=project, is_transient=is_transient)
    if action in {"diff", "restore"}:
        revision = f"{project}/revisions/{arguments.revision}/{action}"
        return _answer(action, client.request("GET" if action == "diff" else "POST", revision))
    if action == "deploy":
        key = write_key(arguments, "console-deploy")
        payload = {"application_id": arguments.application, "promote": arguments.promote}
        result = send_write(client, "POST", f"{project}/deployments", key=key, payload=payload)
        if not arguments.promote:
            _require_staged(result, key)
        return {**_answer(action, result), "idempotency_key": key}
    if action == "pull-request":
        path = f"{project}/pull-request"
        if arguments.status:
            return _answer(action, client.request("GET", path))
        delivery: dict[str, object] = {"new_branch": arguments.new_branch}
        if arguments.revision is not None:
            delivery["revision"] = arguments.revision
        return _answer(action, client.request("POST", path, payload=delivery))
    path = f"{project}/deploy-merged"
    if arguments.idempotency_key is None:
        # Cloud keys it by the pull request, so the same merge deploys once.
        return _answer(action, client.request("POST", path))
    result = send_write(client, "POST", path, key=arguments.idempotency_key)
    return {**_answer(action, result), "idempotency_key": arguments.idempotency_key}


def _source(client: CloudApiClient, project: str, output: Path) -> dict[str, Any]:
    """Stream the latest revision's archive into ``output``, replaced only once it is whole."""

    target = output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, partial = tempfile.mkstemp(prefix=".cayu-console-source-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as written:
            downloaded = client.download(
                f"{project}/source", written, max_bytes=SOURCE_BUNDLE_MAX_BYTES
            )
        os.replace(partial, target)
    except BaseException:
        Path(partial).unlink(missing_ok=True)
        raise
    return {
        "output": str(target),
        "size_bytes": downloaded.size,
        "source_digest": f"sha256:{downloaded.sha256}",
    }


def _pull(arguments: argparse.Namespace, *, client: CloudApiClient, project: str) -> dict[str, Any]:
    """Write the latest revision's source into the directory and record what was pulled."""

    project_id = arguments.project_id
    buffer = io.BytesIO()
    downloaded = client.download(f"{project}/source", buffer, max_bytes=SOURCE_BUNDLE_MAX_BYTES)
    archive = buffer.getvalue()
    digest = f"sha256:{downloaded.sha256}"
    # Read after the download, the history lists the downloaded revision whatever was
    # recorded since.
    # A restore records an earlier revision's digest again; the latest one with it is pulled.
    revision = max(
        (
            item["number"]
            for item in _revisions(client.request("GET", project))
            if item.get("content_digest") == digest
        ),
        default=None,
    )
    if not isinstance(revision, int):
        raise ConsoleError(
            "console_pull_unavailable",
            f"Console project {project_id}'s latest revision was stored by Console's previous "
            "builder, so it can't be pulled; continue it in Console.",
        )
    files = source_files(archive)
    directory = arguments.directory.expanduser().resolve()
    try:
        skipped, kept = write_tree(directory, files, project_id=project_id, force=arguments.force)
        ConsoleCheckout(project_id=project_id, revision=revision, source_digest=digest).write(
            directory
        )
    except OSError as exc:
        raise ConsoleError(
            "console_pull_failed",
            f"Could not write the source into {directory}: {type(exc).__name__}",
        ) from None
    return {
        "directory": str(directory),
        "file_count": len(files),
        # Paths of the revision a forced pull left as they were here, since a push leaves
        # them out (a local `.env.local`).
        "kept": kept,
        # Paths of the revision written that a push from this directory would leave out.
        "not_pushed": sorted(set(files) - set(skipped) - set(kept) - set(source_paths(directory))),
        "project_id": project_id,
        "revision": revision,
        # Paths of the revision a pull never writes, which a push records deleted.
        "skipped": skipped,
        "source_digest": digest,
    }


def _push(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    project: str,
    is_transient: Callable[[CloudApiError], bool],
    transient_limit: int,
) -> dict[str, Any]:
    """Record the pulled directory's source as the project's next revision."""

    project_id = arguments.project_id
    directory = arguments.directory.expanduser().resolve()
    checkout = ConsoleCheckout.read(directory)
    if checkout.project_id != project_id:
        raise ConsoleError(
            "console_checkout_mismatch",
            f"{directory} holds Console project {checkout.project_id}'s source, not "
            f"{project_id}'s.",
        )
    pulled = {"pulled_revision": checkout.revision}
    files = source_tree(directory)
    archive = source_archive(files)
    problem = console_source_problem(files)
    if problem is not None:
        raise ConsoleError("source_invalid", f"The source can't be a Console revision: {problem}")
    digest = source_digest(archive)
    # Refuse what Cloud would refuse anyway before uploading a large bundle.
    _preflight_push(
        client.request("GET", project), project_id=project_id, checkout=checkout, digest=digest
    )
    upload = client.request(
        "POST",
        "/v1/source-bundles/uploads",
        payload={"content_digest": digest, "size_bytes": len(archive)},
    )
    upload_url = upload.get("upload_url")
    if upload.get("content_digest") != digest or not isinstance(upload_url, str):
        raise CloudApiError(
            "api_response_invalid", "Cayu Cloud returned an invalid source bundle upload."
        )
    client.upload_bytes(upload_url, archive)
    try:
        answer = client.request(
            "POST",
            f"{project}/revisions",
            payload={
                "expected_latest_digest": checkout.source_digest,
                "source_digest": digest,
                "summary": arguments.summary,
            },
        )
    except CloudApiError as exc:
        if exc.status_code not in {409, 422}:
            raise
        raise ConsoleError(
            "console_push_conflict" if exc.status_code == 409 else "console_push_rejected",
            exc.detail or str(exc),
            status_code=exc.status_code,
            code=exc.code,
            details=pulled,
        ) from None
    job_id = answer.get("job_id")
    if answer.get("status") in _PENDING_REVISION_JOB and isinstance(job_id, str):
        answer = _await_revision_job(
            client,
            arguments,
            path=f"{project}/revision-jobs/{quote(job_id, safe='')}",
            job_id=job_id,
            pulled=pulled,
            is_transient=is_transient,
            transient_limit=transient_limit,
        )
    job = {} if not isinstance(job_id, str) else {"job_id": job_id}
    if answer.get("status") == "refused":
        raise ConsoleError(
            "console_push_refused",
            _text(answer.get("error")) or "Console refused the source.",
            details={**pulled, **job},
        )
    number = answer.get("number")
    if not isinstance(answer.get("recorded"), bool):
        raise ConsoleError(
            "console_push_unsupported",
            "This Cayu Cloud predates console push, so it recorded the source without checking "
            "that the project hadn't changed since the pull; check the project's revisions "
            "with `cayu cloud console show`.",
            details={**pulled, "revision": number if isinstance(number, int) else None},
        )
    if not isinstance(number, int) or answer.get("content_digest") != digest:
        raise CloudApiError(
            "api_response_invalid", "Cayu Cloud returned an invalid Console revision."
        )
    ConsoleCheckout(project_id=project_id, revision=number, source_digest=digest).write(directory)
    return {
        "directory": str(directory),
        "file_count": len(files),
        "project_id": project_id,
        "pulled_revision": checkout.revision,
        "revision": number,
        "source_digest": digest,
        "status": "saved" if answer["recorded"] else "unchanged",
        **job,
    }


def _preflight_push(
    project: dict[str, Any], *, project_id: str, checkout: ConsoleCheckout, digest: str
) -> None:
    """Refuse a push Cloud would refuse whatever its source: a turn running, or a latest
    revision that is neither the pulled one nor the pushed source."""

    pulled = {"pulled_revision": checkout.revision}
    if project.get("turn_active") is True:
        raise ConsoleError(
            "console_push_conflict",
            f"Console is working on project {project_id}; push after its turn ends.",
            details=pulled,
        )
    latest = max(_revisions(project), key=lambda item: item["number"], default=None)
    if latest is not None and latest.get("content_digest") not in {checkout.source_digest, digest}:
        raise ConsoleError(
            "console_push_conflict",
            f"Console project {project_id} changed since you pulled revision "
            f"{checkout.revision}: its latest revision is {latest['number']}. Pull again and "
            "reapply your change.",
            details=pulled,
        )


def _await_revision_job(
    client: CloudApiClient,
    arguments: argparse.Namespace,
    *,
    path: str,
    job_id: str,
    pulled: Mapping[str, object],
    is_transient: Callable[[CloudApiError], bool],
    transient_limit: int,
) -> dict[str, Any]:
    """Read the job that writes the source into the project's workspace until it settles,
    riding out brief API outages and saying on standard error what it waits for."""

    deadline = _monotonic() + arguments.wait_seconds
    failures = 0
    said: object = None
    while True:
        try:
            answer = client.request("GET", path)
        except CloudApiError as exc:
            if not is_transient(exc) or failures >= transient_limit:
                raise
            failures += 1
        else:
            failures = 0
            status = answer.get("status")
            if status not in _PENDING_REVISION_JOB:
                return answer
            if status != said:
                print(
                    "cayu cloud: waiting for Console to write the source into project "
                    f"{terminal_safe(arguments.project_id)}'s workspace; the job is "
                    f"{terminal_safe(str(status))}.",
                    file=sys.stderr,
                )
                said = status
        remaining = deadline - _monotonic()
        if remaining <= 0:
            raise ConsoleError(
                "console_push_pending",
                f"Console is still writing the source into project {arguments.project_id}'s "
                f"workspace after {arguments.wait_seconds:g} seconds; run `cayu cloud console "
                "show` to see whether it recorded the revision.",
                details={**pulled, "job_id": job_id},
            )
        _sleep(min(arguments.poll_seconds, remaining))


def _follow_events(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    project: str,
    is_transient: Callable[[CloudApiError], bool],
) -> dict[str, Any]:
    """Print each new step as one JSON line until the project's turn ends, or Ctrl-C.

    Every poll reads the project first and then every step after the cursor, so the steps a
    turn wrote before it ended are printed before the command returns.
    """

    interval = (
        EVENTS_FOLLOW_POLL_SECONDS if arguments.poll_seconds is None else arguments.poll_seconds
    )
    after: int = arguments.after
    failures = 0
    try:
        while True:
            reading_project = True
            try:
                detail = client.request("GET", project)
                reading_project = False
                while True:
                    query = {"after": str(after), "limit": str(arguments.limit)}
                    page = client.request("GET", f"{project}/events", query=query)
                    events = page.get("events")
                    if not isinstance(events, list):
                        raise CloudApiError(
                            "api_response_invalid", "Cayu Cloud returned an invalid events page."
                        )
                    for event in events:
                        print(json.dumps(event, sort_keys=True), flush=True)
                    following = page.get("next_after", after)
                    after = following if isinstance(following, int) else after
                    if len(events) < arguments.limit:
                        break
            except CloudApiError as exc:
                failures += 1
                if not is_transient(exc) or failures > EVENTS_MAX_TRANSIENT_FAILURES:
                    raise _follow_failure(
                        exc, arguments.project_id, reading_project=reading_project
                    ) from None
                backoff = max(interval, _EVENTS_MIN_BACKOFF_SECONDS) * 2**failures
                _sleep(min(backoff, _EVENTS_MAX_BACKOFF_SECONDS))
                continue
            failures = 0
            if not detail.get("turn_active"):
                break
            _sleep(interval)
    except KeyboardInterrupt:
        pass
    return {**_answer("events", {"next_after": after}), "followed": True}


def _follow_failure(
    error: CloudApiError, project_id: str, *, reading_project: bool
) -> CloudApiError:
    """A project read's own 404 means the project is missing, unless Console is off, where
    every Console route answers 404 with ``CONSOLE_UNAVAILABLE``."""

    if error.status_code == 404 and reading_project and error.detail != CONSOLE_UNAVAILABLE:
        return CloudApiError(
            "console_project_not_found",
            f"Console project {project_id} was not found in this Organization.",
            status_code=404,
        )
    return error


def _render_events(result: Mapping[str, Any]) -> str | None:
    """`events --follow` printed its steps as it went; a page prints as JSON."""

    return "" if result.get("followed") else None


def _refusal(exc: CloudApiError) -> CloudApiError:
    """Cloud's refusal as the command reports it: Console's plain-text reason in the message,
    and where Console is off, `console_unavailable`. A write's refusal keeps its key."""

    if exc.category != "api_request_rejected" or exc.status_code is None:
        return exc
    if exc.status_code == 404 and exc.detail == CONSOLE_UNAVAILABLE:
        category, message = "console_unavailable", "Console is not available on this Cayu Cloud."
    elif exc.code is None and exc.detail and exc.detail not in str(exc):
        category = exc.category
        message = f"Cayu Cloud API returned HTTP {exc.status_code}: {exc.detail}"
    else:
        return exc
    rebuilt = CloudApiError(
        category,
        message,
        status_code=exc.status_code,
        detail=exc.detail,
        code=exc.code,
        rejected_fields=exc.rejected_fields,
        body=exc.body,
    )
    if isinstance(exc, CloudWriteError):
        return CloudWriteError(rebuilt, idempotency_key=exc.idempotency_key)
    return rebuilt


def _require_staged(result: dict[str, Any], key: str) -> None:
    """Fail when a Cloud from before staging ignored ``--no-promote`` and will promote."""

    deployment = _object(result.get("deployment"))
    if deployment.get("stage_only") is True:
        return
    deployment_id = _text(deployment.get("id"))
    application_id = _text(deployment.get("application_id"))
    raise ConsoleError(
        "staging_unsupported",
        f"This Cayu Cloud doesn't support --no-promote, so Release {deployment_id} of "
        f"{application_id} will be promoted once smoke-tested.",
        details={
            "application_id": application_id,
            "deployment_id": deployment_id,
            "idempotency_key": key,
        },
    )


def _revisions(project: Mapping[str, Any]) -> list[dict[str, Any]]:
    revisions = project.get("revisions")
    return [
        cast("dict[str, Any]", item)
        for item in (revisions if isinstance(revisions, list) else [])
        if isinstance(item, dict)
        and isinstance(item.get("number"), int)
        and not isinstance(item.get("number"), bool)
    ]


def _object(value: object) -> dict[str, Any]:
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _review_comment(value: str) -> dict[str, object]:
    match = _COMMENT.fullmatch(value)
    if match is None:
        raise argparse.ArgumentTypeError("a comment looks like REVISION:PATH:LINE:TEXT")
    revision, path, line, body = match.groups()
    return {"revision": int(revision), "path": path, "line": int(line), "body": body}


def _project(project_id: str) -> str:
    return f"/v1/console/projects/{quote(project_id, safe='')}"


def _answer(operation: str, result: dict[str, Any]) -> dict[str, Any]:
    return {"operation": f"console.{operation}", "result": result}


def _positive(value: str) -> int:
    number = _integer(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive whole number")
    return number


def _non_negative(value: str) -> int:
    number = _integer(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or a positive whole number")
    return number


def _integer(value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a whole number") from exc


def _sleep(seconds: float) -> None:
    """Wait; tests replace this name, never the `time` module other code shares."""

    time.sleep(seconds)


def _monotonic() -> float:
    return time.monotonic()
