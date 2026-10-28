"""`cayu cloud compass`: enroll an Agent in Compass and review what it finds.

The commands mirror Cayu Cloud's operator `compass` commands and call the same
customer API routes (`/v1/compass/agents/...`) with the `cayu cloud login` (or a
selected context's API key). Compass routes need a developer or an organization
administrator. Writes take `--idempotency-key`; without one a fresh key is chosen
and printed, on an error too, so a write whose answer was lost can be repeated
with it.
"""

from __future__ import annotations

import argparse
import secrets
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

from cayu.cli._cloud_api import CloudApiClient, CloudApiError

WAIT_DEFAULT_POLL_SECONDS = 5.0
WAIT_DEFAULT_SECONDS = 30 * 60.0
# A job's statuses once Compass's worker has finished with it.
_JOB_ENDED = frozenset({"succeeded", "failed"})
# `enroll` options and the request fields they set; an option left out takes Cloud's default.
_ENROLL_FIELDS = (
    ("agents", "agents"),
    ("max_forks", "max_forks"),
    ("fork_budget_usd", "fork_budget_usd"),
    ("min_runs", "min_runs"),
    ("lookback_days", "lookback_days"),
    ("mission_budget_usd", "mission_budget_usd"),
    ("monthly_budget_usd", "monthly_budget_usd"),
    ("release", "release_id"),
    ("api_path", "api_path"),
)
# `findings accept` options, which set the request fields of the same names.
_ACCEPT_TEXT = ("note", "requirement", "hypothesis", "expected_effect", "risk")
_MISSION_STATUSES = ("ready", "starting", "started", "failed", "unknown")
_FINDING_STATUSES = ("proposed", "accepted", "dismissed")


class CompassWriteError(CloudApiError):
    """A refused or unanswered Compass write, with the key to repeat it under, and the jobs
    it queued when only waiting for them failed."""

    def __init__(
        self, error: CloudApiError, *, idempotency_key: str, job_ids: list[str] | None = None
    ) -> None:
        super().__init__(
            error.category,
            str(error),
            status_code=error.status_code,
            detail=error.detail,
            code=error.code,
            rejected_fields=error.rejected_fields,
            transport_error=error.transport_error,
            request=error.request,
        )
        self.idempotency_key = idempotency_key
        self.details: dict[str, object] = {"idempotency_key": idempotency_key}
        if job_ids:
            self.details["job_ids"] = job_ids


def add_compass_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    application: Callable[[str], str],
    seconds: Callable[[str], float],
) -> None:
    """Add `compass` to `cayu cloud`; ``application`` and ``seconds`` parse those arguments."""

    compass = commands.add_parser(
        "compass",
        help="Enroll Agents in Compass and review its findings.",
        description=(
            "Enroll an Agent in Compass, which watches how it runs and proposes changes a "
            "person accepts or dismisses. Needs a developer or organization administrator."
        ),
    )
    actions = compass.add_subparsers(dest="compass_command", required=True)
    actions.add_parser("agents", description="Every Agent, with its Compass enrollment.")
    actions.add_parser(
        "show", description="One Agent's enrollment, schedule, recent passes and counts."
    ).add_argument("application", type=application)

    enroll = actions.add_parser(
        "enroll",
        description=(
            "Enroll an Agent, or change its enrollment. Compass judges findings by the "
            "objectives the Agent's Release declares under [tool.cayu.improvement] "
            "(`cayu guide evals-production#objectives-for-compass`). Each enroll sets the "
            "whole enrollment: options left out take Cloud's defaults."
        ),
    )
    enroll.add_argument("application", type=application)
    enroll.add_argument(
        "--agent",
        action="append",
        dest="agents",
        required=True,
        metavar="NAME",
        help=(
            "A Cayu agent inside the Agent that Compass watches; repeat for each (1-16). Each "
            "must be one the Agent registers: an Agent whose fork controller names another "
            "doesn't start. Name every agent to keep: one left out is no longer watched."
        ),
    )
    enroll.add_argument(
        "--allow-model-runs",
        action="store_true",
        help="Let Compass run its forks, each capped at --fork-budget-usd.",
    )
    enroll.add_argument("--fork-budget-usd", help='The cap on each fork\'s spend, such as "2.50".')
    enroll.add_argument("--max-forks", type=int, help="Forks Compass may open (default 12).")
    enroll.add_argument(
        "--min-runs", type=int, help="Runs a step needs before a finding rests on it (default 5)."
    )
    enroll.add_argument(
        "--lookback-days", type=int, help="The window each pass reads (default 14)."
    )
    enroll.add_argument(
        "--mission-budget-usd", help="What one mission Compass makes ready may spend (default 20)."
    )
    enroll.add_argument(
        "--monthly-budget-usd", help="What a month's missions may spend together (default 100)."
    )
    enroll.add_argument(
        "--release", help="The Release whose objectives judge findings (default: the selected one)."
    )
    enroll.add_argument(
        "--api-path",
        choices=("/api", "/cayu/api"),
        help=(
            "Where the Agent serves the Cayu API: /api under cayu serve (the default), "
            "/cayu/api when it mounts Cayu with mount_cayu."
        ),
    )
    enroll.add_argument("--idempotency-key")

    actions.add_parser(
        "disable",
        description="Disable the Agent's enrollment and remove Compass's fork controller from it.",
    ).add_argument("application", type=application)

    for name, what in (
        ("verify", "Ask Compass to check the Agent's consent again."),
        ("pass", "Ask Compass for a pass over the schedule's window now."),
    ):
        job = actions.add_parser(
            name,
            description=(
                f"{what} Prints the first job Compass queued, with one per watched agent in jobs."
            ),
        )
        job.add_argument("application", type=application)
        job.add_argument("--agent", help="Only this watched agent (default: every one).")
        job.add_argument("--idempotency-key")
        job.add_argument(
            "--wait",
            action="store_true",
            help="Wait until the job succeeds or fails, and print it then.",
        )
        job.add_argument("--poll-seconds", type=seconds, default=WAIT_DEFAULT_POLL_SECONDS)
        job.add_argument("--wait-seconds", type=seconds, default=WAIT_DEFAULT_SECONDS)

    show_job = actions.add_parser("job", description="One verify or pass job of the Agent's.")
    show_job.add_argument("application", type=application)
    show_job.add_argument("job_id")

    missions = actions.add_parser(
        "missions",
        description=(
            "The Compound missions Compass made ready, newest first, each with Cloud's start "
            "of it, and why Cloud starts none of the Agent's now, if it doesn't."
        ),
    )
    missions.add_argument("application", type=application)
    missions.add_argument("--status", choices=_MISSION_STATUSES)

    findings = actions.add_parser("findings", description="What Compass found, and its review.")
    finding_actions = findings.add_subparsers(dest="findings_command", required=True)
    listing = finding_actions.add_parser("list", description="Findings, strongest first.")
    listing.add_argument("application", type=application)
    listing.add_argument("--status", choices=_FINDING_STATUSES)
    listing.add_argument("--agent", help="Only this watched agent's findings.")
    for name, what in (
        ("show", "One finding."),
        ("accept", "Accept a finding, optionally sharpening it."),
        ("dismiss", "Dismiss a finding, saying why."),
    ):
        finding = finding_actions.add_parser(name, description=what)
        finding.add_argument("application", type=application)
        finding.add_argument("finding_id")
        if name == "accept":
            for option in _ACCEPT_TEXT:
                finding.add_argument(f"--{option.replace('_', '-')}")
            finding.add_argument(
                "--anatomy",
                action="append",
                dest="affected_anatomy",
                metavar="PATH",
                help="A part of the Agent the finding affects; repeat for each.",
            )
            finding.add_argument(
                "--case",
                action="append",
                dest="development_cases",
                metavar="CASE_ID",
                help="An Arena development case the finding cites; repeat for each.",
            )
        if name == "dismiss":
            finding.add_argument("--note", required=True)
        if name != "show":
            finding.add_argument("--idempotency-key")


def _agent_path(application: str) -> str:
    return f"/v1/compass/agents/{quote(application, safe='')}"


def compass_command(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    is_transient: Callable[[CloudApiError], bool],
    transient_limit: int,
    validate_wait: Callable[[float, float], None],
) -> dict[str, Any]:
    """Run one `cayu cloud compass` command and return its JSON answer."""

    action = arguments.compass_command
    if action == "agents":
        return _answer("agents", client.request("GET", "/v1/compass/agents"))
    path = _agent_path(arguments.application)
    if action == "show":
        return _answer(action, client.request("GET", path))
    if action == "disable":
        return _answer(action, client.request("DELETE", path))
    if action == "job":
        return _answer(action, client.request("GET", _job_path(path, arguments.job_id)))
    if action == "missions":
        query = _query(status=arguments.status)
        return _answer(action, client.request("GET", f"{path}/missions", query=query))
    if action == "enroll":
        payload: dict[str, object] = {
            field: getattr(arguments, option)
            for option, field in _ENROLL_FIELDS
            if getattr(arguments, option) is not None
        }
        if arguments.allow_model_runs:
            payload["allow_model_runs"] = True
        return _enroll(client, arguments, path, payload)
    if action in {"verify", "pass"}:
        if arguments.wait:
            validate_wait(arguments.poll_seconds, arguments.wait_seconds)
        route = "verify" if action == "verify" else "passes"
        answer = _write(
            client,
            action,
            arguments,
            "POST",
            f"{path}/{route}",
            None,
            query=_query(agent=arguments.agent),
        )
        if arguments.wait:
            try:
                answer["result"] = _wait_for_jobs(
                    client,
                    arguments,
                    answer["result"],
                    path=path,
                    is_transient=is_transient,
                    transient_limit=transient_limit,
                )
            except CloudApiError as exc:
                # The jobs are queued: repeating the command with this key answers them again
                # instead of queueing more, and `compass job` reads each.
                raise CompassWriteError(
                    exc,
                    idempotency_key=answer["idempotency_key"],
                    job_ids=_job_ids(answer["result"]),
                ) from None
        return answer
    finding_action = arguments.findings_command
    operation = f"findings.{finding_action}"
    if finding_action == "list":
        query = _query(status=arguments.status, agent=arguments.agent)
        return _answer(operation, client.request("GET", f"{path}/findings", query=query))
    finding = f"{path}/findings/{quote(arguments.finding_id, safe='')}"
    if finding_action == "show":
        return _answer(operation, client.request("GET", finding))
    if finding_action == "accept":
        review: dict[str, object] = {
            option: getattr(arguments, option)
            for option in (*_ACCEPT_TEXT, "affected_anatomy", "development_cases")
            if getattr(arguments, option) is not None
        }
    else:
        review = {"note": arguments.note}
    return _write(client, operation, arguments, "POST", f"{finding}/{finding_action}", review)


def _answer(operation: str, result: dict[str, Any]) -> dict[str, Any]:
    return {"operation": f"compass.{operation}", "result": result}


def _query(**parameters: str | None) -> dict[str, str] | None:
    given = {name: value for name, value in parameters.items() if value is not None}
    return given or None


def _job_path(path: str, job_id: str) -> str:
    return f"{path}/jobs/{quote(job_id, safe='')}"


def _write(
    client: CloudApiClient,
    operation: str,
    arguments: argparse.Namespace,
    method: str,
    path: str,
    payload: dict[str, object] | None,
    *,
    query: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Send one write under the given or a fresh key, and print the key either way."""

    key = arguments.idempotency_key or f"compass-{operation}-{secrets.token_hex(12)}"
    try:
        result = client.request(method, path, payload=payload, idempotency_key=key, query=query)
    except CloudApiError as exc:
        raise CompassWriteError(exc, idempotency_key=key) from None
    return {**_answer(operation, result), "idempotency_key": key}


def _enroll(
    client: CloudApiClient, arguments: argparse.Namespace, path: str, payload: dict[str, object]
) -> dict[str, Any]:
    """Enroll with `agents`, falling back to `agent` on a Cloud that predates several agents.

    Such a Cloud refuses the `agents` field while validating the request, before it records
    anything under the key, so the same key is safe to send again with `agent`.
    """

    try:
        return _write(client, "enroll", arguments, "PUT", path, payload)
    except CompassWriteError as exc:
        if ("body.agents", "extra_forbidden") not in exc.rejected_fields:
            raise
        agents = payload["agents"]
        assert isinstance(agents, list)
        if len(agents) != 1:
            raise CompassWriteError(
                CloudApiError(
                    "api_request_rejected",
                    "This Cayu Cloud lets Compass watch one agent per Agent; pass --agent once.",
                    status_code=exc.status_code,
                ),
                idempotency_key=exc.idempotency_key,
            ) from None
        single = {key: value for key, value in payload.items() if key != "agents"}
        single["agent"] = agents[0]
        arguments.idempotency_key = exc.idempotency_key
        return _write(client, "enroll", arguments, "PUT", path, single)


def _job_ids(queued: dict[str, Any]) -> list[str]:
    listed = queued.get("jobs")
    jobs = [item for item in listed if isinstance(item, dict)] if isinstance(listed, list) else []
    ids = [job.get("job_id") for job in jobs or [queued]]
    return [job_id for job_id in ids if isinstance(job_id, str) and job_id]


def _wait_for_jobs(
    client: CloudApiClient,
    arguments: argparse.Namespace,
    queued: dict[str, Any],
    *,
    path: str,
    is_transient: Callable[[CloudApiError], bool],
    transient_limit: int,
) -> dict[str, Any]:
    """The queued jobs once Compass's worker has finished each, riding out brief API outages.

    Cloud queues one job per watched agent and answers the first job's fields with every job
    in ``jobs``; a Cloud that predates several agents answers the one job alone. The answer
    keeps that shape: the first finished job's fields, and ``jobs`` when Cloud sent it. Each
    round reads every unfinished job and then sleeps once, so watching more agents doesn't
    use up the wait.
    """

    listed = queued.get("jobs")
    sent = [item for item in listed if isinstance(item, dict)] if isinstance(listed, list) else []
    jobs = [{key: value for key, value in job.items() if key != "jobs"} for job in sent or [queued]]
    job_ids = []
    for job in jobs:
        job_id = job.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise CloudApiError("api_response_invalid", "Compass job is missing its job_id.")
        job_ids.append(job_id)
    deadline = time.monotonic() + arguments.wait_seconds
    failures = 0
    while unfinished := [
        index for index, job in enumerate(jobs) if job.get("status") not in _JOB_ENDED
    ]:
        if time.monotonic() >= deadline:
            waiting = ", ".join(job_ids[index] for index in unfinished)
            jobs_word, pronoun = ("Jobs", "each") if len(unfinished) > 1 else ("Job", "it")
            raise CloudApiError(
                "wait_timeout",
                f"{jobs_word} {waiting} didn't finish within {arguments.wait_seconds:g} seconds; "
                f"read {pronoun} later with `cayu cloud compass job {arguments.application} "
                "JOB_ID`, or repeat the command with its idempotency key.",
            )
        time.sleep(arguments.poll_seconds)
        try:
            for index in unfinished:
                jobs[index] = client.request("GET", _job_path(path, job_ids[index]))
            failures = 0
        except CloudApiError as exc:
            if not is_transient(exc) or failures >= transient_limit:
                raise
            failures += 1
    if not sent:
        return jobs[0]
    return {**jobs[0], "jobs": jobs}
