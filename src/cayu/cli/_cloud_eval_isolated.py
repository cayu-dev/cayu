"""`cayu cloud eval run --on isolated`: run a corpus suite in Cayu Cloud's evaluation service.

An isolated run never touches the deployed Agent. Cloud runs each trial in its own allocation
of one of the Agent's Releases (its selected Release unless `--release` names another), with
the Release's own code and migrations, no live secret, and, where the install gives one, a
database of its own that is dropped when the trial or run ends. Nothing it does lands in the
live Agent's sessions, tasks, inbox, knowledge or usage.

The command loads the corpus locally, checks the suite and states the run's bound, submits the
corpus with one evaluation under an idempotency key (`--operation-id`, or a fresh one it prints
in a continuation command: the same key and request answer with the evaluation it already
started), waits for it to finish and reads its receipt. `runs list|show|cancel` read and cancel
these evaluations by id; ids starting with `evl_` are Cloud's, others the Agent's own runs.
"""

from __future__ import annotations

import argparse
import json
import secrets
import shlex
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from cayu.cli._cloud_api import CloudApiError
from cayu.cli._output import note, terminal_safe

if TYPE_CHECKING:
    from cayu.cli._cloud_api import CloudApiClient
    from cayu.evals.corpus import EvalCorpusDocument, EvalCorpusSuiteInspectionV1

SCHEMA_VERSION = 1
EVALUATION_PREFIX = "evl_"
TERMINAL = frozenset({"completed", "failed", "cancelled"})
# Cloud's bounds on one evaluation (its evaluation service contract).
MAX_TRIALS = 10
MAX_CONCURRENCY = 32
# Cloud's largest page of evaluations; `runs list` reads as many pages as its --limit needs.
PAGE_LIMIT = 100
# How many pages one `runs list` reads at most, looking for runs of one status.
MAX_PAGES = 20
# A Cayu Cloud whose evaluation service predates corpus suites, or that runs none, refuses one so.
_NOT_OFFERED = frozenset({"suite_kind_unsupported", "evaluations_unavailable"})
# A read that may succeed later: Cloud's API was out of reach, deploying or overloaded.
_TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})


def is_isolated_run(run_id: str) -> bool:
    return run_id.startswith(EVALUATION_PREFIX)


def run_isolated(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    application_id: str,
    corpus: EvalCorpusDocument,
    suite: EvalCorpusSuiteInspectionV1,
    recovery_arguments: Sequence[str],
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
    error: Callable[..., Exception],
) -> dict[str, Any]:
    """Submit the suite as one evaluation of the Agent's Release, wait for it and read its
    receipt. ``error(category, message, details=...)`` makes the command's refusals."""

    release_id = arguments.release or _selected_release(client, application_id, error=error)
    operation_id = arguments.operation_id or f"cayu-cli-eval-{secrets.token_hex(16)}"
    suite_request: dict[str, Any] = {
        "corpus": corpus.model_dump(mode="json"),
        "suite_id": suite.id,
    }
    if arguments.target is not None:
        suite_request["eval_target"] = arguments.target
    body: dict[str, Any] = {
        "release_id": release_id,
        "suite": suite_request,
        "trials": arguments.trials,
        "purpose": "development",
        "idempotency_key": operation_id,
        "max_concurrency": arguments.max_concurrency,
    }
    if arguments.max_parallel_trials is not None:
        body["max_parallel_trials"] = arguments.max_parallel_trials
    environment = {
        name: value
        for name, value in (("database", arguments.database), ("isolation", arguments.isolation))
        if value is not None
    }
    if environment:
        body["environment"] = environment
    bound = {
        "release_id": release_id,
        "suite_id": suite.id,
        "cases": suite.case_count,
        "trials_per_case": suite.trials,
        "allocations": arguments.trials,
        # A run sharing one database runs its allocations one at a time, whatever it asks.
        "parallel_allocations": 1
        if arguments.isolation == "run"
        else min(arguments.max_parallel_trials or 1, arguments.trials),
        "total_trials": suite.case_count * suite.trials * arguments.trials,
        "max_concurrency": arguments.max_concurrency,
        "timeout_seconds_per_trial": suite.timeout_seconds,
        "environment": environment or None,
    }
    note(_bound_text(bound))
    commands = _continuation(
        arguments,
        application_id=application_id,
        release_id=release_id,
        operation_id=operation_id,
        recovery_arguments=recovery_arguments,
    )
    note(
        f"Starting the evaluation with operation ID {operation_id}. If this command stops before "
        f"it names the evaluation, `{commands['continue']}` starts nothing new and continues."
    )
    try:
        evaluation = client.request(
            "POST", f"/v1/applications/{quote(application_id, safe='')}/evaluations", payload=body
        )
    except CloudApiError as exc:
        if exc.code in _NOT_OFFERED:
            raise error(
                "isolated_runs_unavailable",
                "This Cayu Cloud doesn't run corpus suites in an isolated environment yet "
                f"({exc.code}); run the suite on the live Agent with --on live.",
                details={"code": exc.code},
            ) from None
        if exc.code == "database_unsupported":
            raise error(
                "database_unsupported",
                "This Cayu Cloud gives eval environments no database of their own; run without "
                "--database postgres, or with --database none.",
                details={"code": exc.code},
            ) from None
        raise
    evaluation_id = _evaluation_id(evaluation, error=error)
    commands |= run_commands(application_id, evaluation_id, recovery_arguments)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "application_id": application_id,
        "on": "isolated",
        "operation_id": operation_id,
        "corpus_revision": corpus.revision,
        "bound": bound,
        "run": evaluation,
        "summary": None,
        "receipt": None,
        "output": None,
        "commands": commands,
    }
    if arguments.no_wait:
        return {"operation": "eval.run", "result": result}
    try:
        evaluation = _wait(
            client,
            application_id,
            evaluation,
            wait_seconds=arguments.wait_seconds,
            poll_seconds=arguments.poll_seconds,
            sleep=sleep,
            monotonic=monotonic,
        )
    except CloudApiError as exc:
        # The evaluation goes on in Cloud whatever this read did; name it.
        raise error(
            exc.category,
            f"{exc} Evaluation {evaluation_id} goes on in Cayu Cloud; `{commands['show']}` "
            "shows it.",
            details={"run_id": evaluation_id, "commands": commands},
        ) from None
    result["run"] = evaluation
    if evaluation.get("status") not in TERMINAL:
        raise error(
            "eval_run_still_running",
            f"Evaluation {evaluation_id} hasn't finished; `{commands['show']}` shows its progress.",
            details={"run_id": evaluation_id, "commands": commands},
        )
    if evaluation["status"] != "completed":
        failure = evaluation.get("failure") or {}
        raise error(
            "eval_run_failed" if evaluation["status"] == "failed" else "eval_run_cancelled",
            f"Evaluation {evaluation_id} {evaluation['status']}"
            + (
                f" ({failure.get('code')}): {failure.get('message')}"
                if isinstance(failure, dict) and failure.get("code")
                else "."
            ),
            details={"run_id": evaluation_id, "failure": failure or None, "commands": commands},
        )
    receipt = client.request("GET", _evaluation_path(application_id, evaluation_id) + "/receipt")
    result["receipt"] = receipt
    result["summary"] = receipt_summary(receipt)
    if arguments.output is not None:
        Path(arguments.output).write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        result["output"] = str(arguments.output)
    return {"operation": "eval.run", "result": result}


def list_isolated(
    client: CloudApiClient, application_id: str, *, limit: int, status: str | None
) -> dict[str, Any]:
    """The Agent's newest ``limit`` evaluations, of ``status`` when given, in the shape `runs
    list` prints. Cloud filters by nothing but Release, so a status filter reads further pages
    until it has ``limit`` runs or the list ends (at most ``MAX_PAGES`` pages)."""

    path = f"/v1/applications/{quote(application_id, safe='')}/evaluations"
    items: list[dict[str, Any]] = []
    cursor: str | None = None
    for _ in range(MAX_PAGES):
        query = {"limit": str(PAGE_LIMIT if status is not None else min(limit, PAGE_LIMIT))}
        if cursor is not None:
            query["cursor"] = cursor
        page = client.request("GET", path, query=query)
        for item in page.get("items") or []:
            if isinstance(item, dict) and (status is None or item.get("status") == status):
                items.append(item)
        next_cursor = page.get("next_cursor")
        cursor = next_cursor if isinstance(next_cursor, str) and next_cursor else None
        if cursor is None or len(items) >= limit:
            break
    return {
        "on": "isolated",
        "items": [_listed(item) for item in items[:limit]],
        "has_more": len(items) > limit or cursor is not None,
    }


def show_isolated(client: CloudApiClient, application_id: str, run_id: str) -> dict[str, Any]:
    evaluation = client.request("GET", _evaluation_path(application_id, run_id))
    receipt = None
    if evaluation.get("status") in TERMINAL:
        receipt = client.request("GET", _evaluation_path(application_id, run_id) + "/receipt")
    return {
        "on": "isolated",
        "run": evaluation,
        "summary": None if receipt is None else receipt_summary(receipt),
        "receipt": receipt,
    }


def cancel_isolated(client: CloudApiClient, application_id: str, run_id: str) -> dict[str, Any]:
    return {
        "on": "isolated",
        "run": client.request("POST", _evaluation_path(application_id, run_id) + "/cancel"),
    }


def receipt_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """What a receipt says of the run: its cases' outcomes and its accuracy, never content."""

    outcomes: dict[str, int] = {}
    for fact in receipt.get("case_facts") or []:
        if isinstance(fact, dict) and isinstance(fact.get("outcome"), str):
            outcomes[fact["outcome"]] = outcomes.get(fact["outcome"], 0) + 1
    metrics = receipt.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        "status": receipt.get("status"),
        "accuracy": metrics.get("accuracy"),
        "cases": outcomes,
        "fingerprint": receipt.get("fingerprint"),
    }


def run_commands(
    application_id: str, run_id: str, recovery_arguments: Sequence[str]
) -> dict[str, str]:
    prefix = ["cayu", "cloud", *recovery_arguments, "eval", "runs"]
    suffix = ["--application", application_id, run_id]
    return {action: shlex.join([*prefix, action, *suffix]) for action in ("show", "cancel")}


def render_isolated_run(result: Mapping[str, Any]) -> str:
    run = result["run"]
    lines = [f"Evaluation {_text(run.get('id'))}: {_text(run.get('status'))}"]
    summary = result.get("summary")
    if summary is not None:
        cases = summary["cases"]
        accuracy = summary.get("accuracy")
        lines.append(
            "Cases: "
            + (
                ", ".join(f"{count} {outcome}" for outcome, count in sorted(cases.items()))
                or "none"
            )
            + ("" if accuracy is None else f" (accuracy {accuracy:g})")
        )
    databases = (run.get("environment") or {}).get("databases") or []
    if databases:
        states = ", ".join(sorted({str(item.get("state")) for item in databases}))
        lines.append(f"Eval databases: {len(databases)} ({states})")
    if result.get("output") is not None:
        lines.append(f"Receipt: {_text(result['output'])}")
    elif summary is None:
        lines.append(f"Progress: {result['commands']['show']}")
    return "\n".join(lines)


def _listed(item: Mapping[str, Any]) -> dict[str, Any]:
    suite = item.get("suite")
    if not isinstance(suite, dict):
        suite = {}
    return {
        "id": item.get("id"),
        "on": "isolated",
        "status": item.get("status"),
        "suite_id": suite.get("suite_id") or suite.get("eval_target"),
        "release_id": item.get("release_id"),
        "created_at": item.get("created_at"),
        "failure": item.get("failure"),
    }


def _selected_release(
    client: CloudApiClient, application_id: str, *, error: Callable[..., Exception]
) -> str:
    application = client.request("GET", f"/v1/applications/{quote(application_id, safe='')}")
    selected = application.get("current_deployment_id")
    if not isinstance(selected, str) or not selected:
        raise error(
            "release_not_found",
            f"Agent {application_id} has no selected Release to evaluate; pass --release.",
        )
    return selected


def _evaluation_id(evaluation: Mapping[str, Any], *, error: Callable[..., Exception]) -> str:
    evaluation_id = evaluation.get("id")
    if not isinstance(evaluation_id, str) or not is_isolated_run(evaluation_id):
        raise error("api_response_invalid", "Cayu Cloud did not name the evaluation it started.")
    return evaluation_id


def _wait(
    client: CloudApiClient,
    application_id: str,
    evaluation: dict[str, Any],
    *,
    wait_seconds: float,
    poll_seconds: float,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> dict[str, Any]:
    """Read the evaluation until it ends or the wait does; a read Cloud can't answer for a
    while (its API deploying, say) is read again, since the evaluation goes on regardless."""

    deadline = monotonic() + wait_seconds
    path = _evaluation_path(application_id, str(evaluation["id"]))
    unreachable = False
    while evaluation.get("status") not in TERMINAL and monotonic() < deadline:
        sleep(min(poll_seconds, max(0.0, deadline - monotonic())))
        try:
            evaluation = client.request("GET", path)
        except CloudApiError as exc:
            if not _transient(exc):
                raise
            if not unreachable:
                note(
                    f"{exc} Evaluation {evaluation['id']} goes on in Cayu Cloud; reading it again."
                )
            unreachable = True
        else:
            unreachable = False
    return evaluation


def _transient(exc: CloudApiError) -> bool:
    return exc.category == "api_unavailable" or exc.status_code in _TRANSIENT_STATUSES


def _evaluation_path(application_id: str, evaluation_id: str) -> str:
    return (
        f"/v1/applications/{quote(application_id, safe='')}/evaluations/"
        f"{quote(evaluation_id, safe='')}"
    )


def _continuation(
    arguments: argparse.Namespace,
    *,
    application_id: str,
    release_id: str,
    operation_id: str,
    recovery_arguments: Sequence[str],
) -> dict[str, str]:
    """The `eval run` that repeats this submission exactly (the same Release, so a deploy in
    between changes nothing), and the `runs list` that shows it."""

    command = [
        *("cayu", "cloud", *recovery_arguments, "eval", "run"),
        *("--application", application_id, "--corpus", str(arguments.corpus)),
        *("--suite", arguments.suite, "--release", release_id),
        *("--trials", str(arguments.trials), "--max-concurrency", str(arguments.max_concurrency)),
    ]
    for option, value in (
        ("--max-parallel-trials", arguments.max_parallel_trials),
        ("--target", arguments.target),
        ("--database", arguments.database),
        ("--isolation", arguments.isolation),
        ("--output", arguments.output),
    ):
        if value is not None:
            command += [option, str(value)]
    command += ["--operation-id", operation_id]
    listing = [
        *("cayu", "cloud", *recovery_arguments, "eval", "runs", "list"),
        *("--application", application_id),
    ]
    return {"continue": shlex.join(command), "list": shlex.join(listing)}


def _bound_text(bound: Mapping[str, Any]) -> str:
    environment = bound["environment"] or {}
    database = environment.get("database")
    where = (
        "its own eval database"
        if database == "postgres"
        else "no database of its own"
        if database == "none"
        else "an eval database where the install gives one"
    )
    isolation = environment.get("isolation") or "trial"
    parallel = bound.get("parallel_allocations") or 1
    at_once = (
        ""
        if parallel == 1
        else f" Up to {parallel} allocations run at once, as far as the install allows."
    )
    return "\n".join(
        [
            f"Isolated evaluation of Release {bound['release_id']}, suite {bound['suite_id']}: "
            f"{count(bound['cases'], 'case')} x {count(bound['trials_per_case'], 'trial')}"
            f" x {count(bound['allocations'], 'allocation')} = "
            f"{count(bound['total_trials'], 'trial')}, at most {bound['max_concurrency']} "
            "running at a time in each allocation.",
            f"Each trial: {bound['timeout_seconds_per_trial']}s. Each allocation gets {where} "
            f"(isolation {isolation}); nothing reaches the live Agent's stores.{at_once}",
        ]
    )


def environment_show(
    client: CloudApiClient, application_id: str, *, release_id: str | None
) -> dict[str, Any]:
    """What an isolated evaluation of the Release (default: the Agent's selected one) would
    get, from Cloud's evaluation catalog: names and limits, never a value."""

    query = {} if release_id is None else {"release_id": release_id}
    catalog = client.request(
        "GET",
        f"/v1/applications/{quote(application_id, safe='')}/evaluation-catalog",
        query=query,
    )
    execution = catalog.get("execution")
    return {
        "release_id": catalog.get("release_id"),
        "version": catalog.get("version"),
        "selected": catalog.get("selected"),
        "eval_targets": list(catalog.get("eval_targets") or []),
        # None from a Cloud that doesn't report it yet.
        "environment": catalog.get("environment"),
        "execution": execution if isinstance(execution, dict) else None,
        "model_access_problem": catalog.get("model_access_problem"),
    }


def _names(items: Sequence[object]) -> str:
    return ", ".join(_text(item) for item in items) or "none"


def render_environment(result: Mapping[str, Any]) -> str:
    lines = [
        f"Release {_text(result['release_id'])} (version {_text(result['version'])}"
        + (", selected)" if result.get("selected") else ")")
    ]
    execution = result.get("execution") or {}
    if execution.get("available") is False:
        lines.append(f"Isolated runs unavailable: {_text(execution.get('reason'))}")
    targets = _names(result["eval_targets"])
    environment = result.get("environment")
    if not isinstance(environment, dict):
        lines.append(
            f"This Cayu Cloud doesn't describe its eval environments yet. Eval targets: {targets}."
        )
        return "\n".join(lines)
    seed = environment.get("seed")
    variables = [
        str(item.get("name")) + (" (secret)" if item.get("secret") else "")
        for item in environment.get("eval_only_variables") or []
        if isinstance(item, dict)
    ]
    limits = environment.get("limits") or {}
    lines += [
        f"Eval targets: {targets}"
        + (
            f" (default {_text(environment['default_eval_target'])})"
            if environment.get("default_eval_target")
            else ""
        ),
        "Database: "
        + (
            "its own PostgreSQL database for each allocation by default"
            if environment.get("database") == "postgres"
            else "none (this install gives no eval databases)"
            if not environment.get("databases_available")
            else "none by default"
        ),
        "Seed: "
        + (
            f"{_text(seed.get('target'))}, up to {_text(seed.get('timeout_seconds'))}s"
            if isinstance(seed, dict)
            else _text(environment.get("seed_problem") or "none declared")
        ),
        "Fixtures: " + _names(environment.get("fixtures") or []),
        "Eval-only variables: " + _names(variables),
        "Model keys brokered to trials: " + _names(environment.get("brokered_credentials") or []),
        f"Limits: {_text(limits.get('max_trials'))} trials, "
        f"{_text(limits.get('max_parallel_trials'))} at once, "
        f"concurrency {_text(limits.get('max_concurrency'))} in each.",
    ]
    if result.get("model_access_problem"):
        lines.append(_text(result["model_access_problem"]))
    return "\n".join(lines)


def count(number: object, noun: str) -> str:
    return f"{number} {noun}" + ("" if number == 1 else "s")


def _text(value: object) -> str:
    return terminal_safe(str(value))
