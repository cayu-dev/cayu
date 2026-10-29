"""`cayu cloud eval`: run and read a deployed Agent's evals from the terminal.

The commands call the eval API the Agent's own Cayu serves (`/api/evals/...`), the way `cayu
task list --application` reads its tasks: Cayu Cloud names the Agent's Cayu address and its
Developer dashboard login, and the CLI sends that login to the Agent (`_cloud_agent_api`). A
run executes on the Agent's eval worker, in its current release, with its secrets, sandboxes
and stores; the CLI imports the corpus, admits the run and reads it.

`eval run` loads a local corpus, checks the suite against the eval target the Agent publishes
(`GET /evals/targets`), states the run's bound, imports the corpus and admits one run under an
idempotency key, then waits for it and downloads its report. The key is `--operation-id`, or a
fresh one the command prints, in a continuation command that also pins the eval target's
execution-profile revision: the server binds the key to the whole request, so repeating it
exactly admits nothing new and goes on with the run it admitted, also after a deploy changed
the target's current profile. A continuation sends the admission before any import, which
the target's current release checks. `runs list|show|cancel`, `compare` and `baseline` read
and act on runs by id; a baseline approval's continuation likewise repeats its operation ID
and the generation it was decided over.
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import shlex
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from cayu.cli._cloud_agent_api import CayuApi, CayuApiError, CayuApiNotFound, cloud_agent_api
from cayu.cli._cloud_api import CloudApiError
from cayu.cli._cloud_releases import add_output_argument
from cayu.cli._cloud_settings import _cell, _table
from cayu.cli._output import terminal_safe

if TYPE_CHECKING:
    from cayu.cli._cloud_api import CloudApiClient
    from cayu.evals.corpus import EvalCorpusDocument, EvalCorpusSuiteInspectionV1

SCHEMA_VERSION = 1
DEFAULT_POLL_SECONDS = 5.0
DEFAULT_WAIT_SECONDS = 6 * 60 * 60.0
DEFAULT_LIST_LIMIT = 20
# The Agent's bound on one page of eval runs (`EVAL_STORE_MAX_PAGE_SIZE`).
MAX_LIST_LIMIT = 200
# `EvalRunStatus`, in lifecycle order; the first three are still running.
RUN_STATUSES = ("queued", "running", "cancelling", "completed", "failed", "cancelled")
TERMINAL_RUN_STATUSES = frozenset(RUN_STATUSES[3:])
# The Agent accepts up to 512 characters; the CLI's other operation IDs stop at 256.
_OPERATION_ID_MAX_CHARS = 256
_DEFAULT_CURRENCY = "USD"
_CURRENCY = re.compile(r"[A-Z0-9._-]{1,16}")
# Reading an asleep Agent wakes its web process, which can take a minute or more.
_WAKE_TIMEOUT_SECONDS = 150.0
# Reads that fail this way say nothing about the run, which goes on in the Agent; a deploy
# that replaces the Agent's web process mid-run answers like this for a while.
_TRANSIENT_CATEGORIES = frozenset({"starting", "timeout", "unreachable"})
_REPORT_RECORD_TYPE = "cayu.eval-result-report"
# Eval revisions, such as execution-profile revisions and baseline operation IDs.
_EVAL_REVISION = re.compile(r"sha256:[0-9a-f]{64}")
# The Agent's own refusals of a run admission (`cayu.server._eval_run_admission` and
# `_corpus_routes`), told apart by their detail: the corpus isn't stored; the key already
# admitted another request; and, answered only after the key turned out to have admitted
# nothing, the target's current execution profile refusing the request.
_CORPUS_NOT_FOUND = "Eval corpus not found."
_IDEMPOTENCY_CONFLICT = "Idempotency-Key is already bound to another eval run request."
_PROFILE_REFUSALS = frozenset(
    {
        "Eval run exceeds the published execution-profile trial limit.",
        "Eval run exceeds the published execution-profile concurrency limit.",
        "The selected eval execution profile changed after it was reviewed. Refresh readiness "
        "before launching.",
        "The selected eval execution profile changed during launch preparation. Refresh "
        "readiness before launching.",
        "The application identity for this eval execution profile changed after the target "
        "was published. Refresh the deployment before launching.",
    }
)


class CloudEvalError(CloudApiError):
    """A refused, failed or unfinished eval command, with what the caller needs to go on."""

    def __init__(
        self,
        category: str,
        message: str,
        *,
        status_code: int | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(category, message, status_code=status_code)
        self.details: dict[str, object] = dict(details or {})


def add_eval_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    seconds: Callable[[str], float],
) -> None:
    """Add `eval` to `cayu cloud`; ``seconds`` parses wait and poll durations."""

    evals = commands.add_parser(
        "eval",
        help="Run and read a deployed Agent's evals.",
        description=(
            "Run a corpus suite on a deployed Agent's eval target, on its own worker and in "
            "its current release, and read, cancel, compare and approve its runs."
        ),
    )
    actions = evals.add_subparsers(dest="eval_command", required=True)

    run = actions.add_parser(
        "run",
        description=(
            "Import a corpus into the Agent, admit one run of one of its suites on the eval "
            "target the corpus names, and wait for the result. The command states the run's "
            "bound before admitting it. The trial count is part of the corpus suite: to run a "
            "different count, add a second suite with that count to the corpus."
        ),
    )
    _add_application_argument(run)
    run.add_argument("--corpus", type=Path, required=True, metavar="FILE", help="Corpus JSON.")
    run.add_argument("--suite", required=True, metavar="ID", help="The corpus suite to run.")
    run.add_argument(
        "--max-concurrency",
        type=_positive_count,
        default=1,
        metavar="N",
        help=(
            "Trials running at once (default 1); at most what the target and the suite's "
            "trial policy allow."
        ),
    )
    run.add_argument(
        "--max-cost",
        type=_cost_amount,
        metavar="AMOUNT",
        help=(
            "Interrupt each trial's agent once its observed cost reaches AMOUNT, such as "
            "0.50. AI judges keep their own profiles' limits. Needs a target that publishes "
            "pricing."
        ),
    )
    run.add_argument(
        "--currency",
        type=_currency,
        metavar="CODE",
        help=f"The currency of --max-cost (default {_DEFAULT_CURRENCY}).",
    )
    run.add_argument(
        "--operation-id",
        type=_operation_id,
        metavar="ID",
        help=(
            "The admission's idempotency key. Without one a fresh key is printed in a "
            "continuation command that admits nothing new and continues with the same run."
        ),
    )
    run.add_argument(
        "--execution-profile-revision",
        type=_eval_revision,
        metavar="REV",
        help=(
            "With --operation-id, repeat an earlier admission exactly: the eval target's "
            "execution-profile revision it was made under, from its continuation command. A "
            "deploy since then changes the target's current revision, not this one."
        ),
    )
    run.add_argument(
        "--output",
        type=Path,
        metavar="FILE",
        help="Write the run's JSON report here, the file `cayu eval compare` reads.",
    )
    run.add_argument("--no-wait", action="store_true", help="Return once the run is admitted.")
    run.add_argument(
        "--wait-seconds",
        type=seconds,
        default=DEFAULT_WAIT_SECONDS,
        help=f"How long to wait for the run (default {DEFAULT_WAIT_SECONDS:.0f}).",
    )
    run.add_argument(
        "--poll-seconds",
        type=seconds,
        default=DEFAULT_POLL_SECONDS,
        help=f"How often to read its progress (default {DEFAULT_POLL_SECONDS:.0f}).",
    )
    add_output_argument(run, render_run, failure=render_failure)

    runs = actions.add_parser("runs", description="List, show or cancel an Agent's eval runs.")
    run_actions = runs.add_subparsers(dest="eval_runs_command", required=True)
    listing = run_actions.add_parser(
        "list",
        description=(
            "List an eval target's runs, newest first; the Agent's default target unless "
            "--target-key names another."
        ),
    )
    _add_application_argument(listing)
    listing.add_argument(
        "--target-key", metavar="KEY", help="The eval target (`GET /evals/targets`)."
    )
    listing.add_argument("--status", choices=RUN_STATUSES)
    listing.add_argument(
        "--limit",
        type=_list_limit,
        default=DEFAULT_LIST_LIMIT,
        metavar="N",
        help=f"The newest N runs (default {DEFAULT_LIST_LIMIT}, at most {MAX_LIST_LIMIT}).",
    )
    add_output_argument(listing, render_runs, failure=render_failure)
    show = run_actions.add_parser(
        "show", description="Show one eval run: its status, progress and result."
    )
    _add_application_argument(show)
    show.add_argument("run_id", metavar="RUN_ID")
    add_output_argument(show, render_run_detail, failure=render_failure)
    cancel = run_actions.add_parser(
        "cancel",
        description=(
            "Ask the Agent to cancel an eval run. Trials already running stop; the run ends "
            "cancelled."
        ),
    )
    _add_application_argument(cancel)
    cancel.add_argument("run_id", metavar="RUN_ID")
    add_output_argument(cancel, render_cancel, failure=render_failure)

    compare = actions.add_parser(
        "compare",
        description=(
            "Compare two completed runs of the same suite on the Agent: whether they are "
            "comparable, and the current run's regressions."
        ),
    )
    _add_application_argument(compare)
    compare.add_argument("baseline_run_id", metavar="BASELINE_RUN_ID")
    compare.add_argument("current_run_id", metavar="CURRENT_RUN_ID")
    add_output_argument(compare, render_comparison, failure=render_failure)

    baseline = actions.add_parser(
        "baseline",
        description=(
            "Approve a completed run's result as the baseline of its suite, the result later "
            "runs are compared with."
        ),
    )
    _add_application_argument(baseline)
    baseline.add_argument("run_id", metavar="RUN_ID")
    baseline.add_argument(
        "--expected-generation",
        type=_generation,
        metavar="N",
        help=(
            "Approve only over the baseline at generation N (0 when the suite has none); "
            "without it, over the current baseline. From an approval's continuation command."
        ),
    )
    baseline.add_argument(
        "--operation-id",
        type=_eval_revision,
        metavar="ID",
        help=(
            "With --expected-generation, repeat an earlier approval: its operation ID, from its "
            "continuation command."
        ),
    )
    add_output_argument(baseline, render_baseline, failure=render_failure)


def eval_command(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    validate_wait: Callable[[float, float], None],
    recovery_arguments: Sequence[str] = (),
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run one `cayu cloud eval` command and return its JSON answer."""

    action = arguments.eval_command
    if action == "run":
        return _run(
            arguments,
            client=client,
            validate_wait=validate_wait,
            recovery_arguments=recovery_arguments,
            sleep=sleep,
            monotonic=monotonic,
        )
    application_id = _application_id(client, arguments.application)
    with _agent_api(client, application_id, arguments.timeout_seconds) as api:
        if action == "runs":
            if arguments.eval_runs_command == "list":
                result = _list_runs(api, arguments)
            elif arguments.eval_runs_command == "show":
                result = _show_run(api, arguments.run_id)
            else:
                result = _cancel_run(api, arguments.run_id)
            operation = f"eval.runs.{arguments.eval_runs_command}"
        elif action == "compare":
            result = _compare(api, arguments.baseline_run_id, arguments.current_run_id)
            operation = "eval.compare"
        else:
            result = _approve_baseline(
                api,
                arguments,
                application_id=application_id,
                recovery_arguments=recovery_arguments,
            )
            operation = "eval.baseline"
    return {
        "operation": operation,
        "result": {"schema_version": SCHEMA_VERSION, "application_id": application_id, **result},
    }


def _run(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    validate_wait: Callable[[float, float], None],
    recovery_arguments: Sequence[str],
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> dict[str, Any]:
    if not arguments.no_wait:
        validate_wait(arguments.poll_seconds, arguments.wait_seconds)
    if arguments.currency is not None and arguments.max_cost is None:
        raise CloudEvalError("invalid_input", "--currency applies only with --max-cost.")
    continuing = arguments.execution_profile_revision is not None
    if continuing and arguments.operation_id is None:
        raise CloudEvalError(
            "invalid_input",
            "--execution-profile-revision repeats an earlier admission; pass it with that "
            "admission's --operation-id.",
        )
    if arguments.output is not None:
        from cayu.cli.evals import _reject_output_path_aliases

        _reject_output_path_aliases(
            outputs=(("--output", str(arguments.output)),),
            protected=(("the eval corpus", str(arguments.corpus)),),
        )
    corpus = _load_corpus(arguments.corpus)
    suite = _corpus_suite(corpus, arguments.suite)
    cost_cap = (
        None
        if arguments.max_cost is None
        else {
            "max_estimated_cost": arguments.max_cost,
            "currency": arguments.currency or _DEFAULT_CURRENCY,
        }
    )
    operation_id = arguments.operation_id or f"cayu-cli-eval-{secrets.token_hex(16)}"
    application_id = _application_id(client, arguments.application)
    with _agent_api(client, application_id, arguments.timeout_seconds) as api:
        # Repeating an admission must reach the Agent's replay unchanged; its own checks
        # against a newer release would refuse a run that was admitted under the older one.
        target = _published_target(api, corpus.target_key, require_ready=not continuing)
        current_profile = target.get("execution_profile")
        current_revision = (
            current_profile.get("revision") if isinstance(current_profile, dict) else None
        )
        if not continuing:
            _check_bounds(
                target, suite, max_concurrency=arguments.max_concurrency, cost_cap=cost_cap
            )
        profile_revision = arguments.execution_profile_revision or current_revision
        if not isinstance(profile_revision, str):
            raise CloudEvalError(
                "api_response_invalid", "The Agent's eval target names no execution profile."
            )
        commands = _run_commands(
            arguments,
            application_id=application_id,
            operation_id=operation_id,
            profile_revision=profile_revision,
            target_key=corpus.target_key,
            recovery_arguments=recovery_arguments,
        )
        bound = _bound(
            target, corpus, suite, max_concurrency=arguments.max_concurrency, cost_cap=cost_cap
        )
        _say(_bound_text(bound))

        def import_corpus() -> None:
            imported = api.post(
                "/evals/corpora", corpus.model_dump(mode="json"), resource="eval corpus import"
            )
            if not isinstance(imported, dict) or imported.get("revision") != corpus.revision:
                raise CloudEvalError(
                    "api_response_invalid", "The Agent did not confirm the eval corpus it imported."
                )

        # An import is checked against the target's current release, which a continuation may
        # no longer satisfy; its admission replays without one while the corpus is stored.
        if not continuing:
            import_corpus()
        _say(
            f"Admitting the run with operation ID {operation_id}. If this command stops before "
            f"it names the run, `{commands['continue']}` admits nothing new and continues."
        )
        run = _admit(
            api,
            corpus_revision=corpus.revision,
            suite_id=suite.id,
            profile_revision=profile_revision,
            profile_changed=profile_revision != current_revision,
            max_concurrency=arguments.max_concurrency,
            cost_cap=cost_cap,
            operation_id=operation_id,
            commands=commands,
            import_corpus=import_corpus if continuing else None,
        )
        run_id = run["id"]
        commands = {**commands, **_run_id_commands(application_id, run_id, recovery_arguments)}
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "application_id": application_id,
            "operation_id": operation_id,
            "execution_profile_revision": profile_revision,
            "corpus_revision": corpus.revision,
            "bound": bound,
            "run": run,
            "progress": None,
            "summary": None,
            "output": None,
            "commands": commands,
        }
        if arguments.no_wait:
            return {"operation": "eval.run", "result": result}
        run, progress = _wait_for_run(
            api,
            run,
            wait_seconds=arguments.wait_seconds,
            poll_seconds=arguments.poll_seconds,
            operation_id=operation_id,
            commands=commands,
            sleep=sleep,
            monotonic=monotonic,
        )
        result.update(run=run, progress=progress)
        if run["status"] != "completed":
            raise _unfinished_run_error(run, operation_id=operation_id, commands=commands)
        report = _report(api, run_id, commands=commands)
    result["summary"] = _report_summary(report, run_id=run_id)
    if arguments.output is not None:
        _write_report(arguments.output, report, run_id=run_id, commands=commands)
        result["output"] = str(arguments.output)
    return {"operation": "eval.run", "result": result}


def _load_corpus(path: Path) -> EvalCorpusDocument:
    from cayu.evals.corpus import load_eval_corpus

    try:
        return load_eval_corpus(path)
    except OSError as exc:
        raise CloudEvalError(
            "corpus_unreadable",
            f"Could not read the eval corpus {path}: {exc.strerror or type(exc).__name__}.",
        ) from None
    except ValueError as exc:
        raise CloudEvalError(
            "invalid_corpus", f"{path} is not a valid eval corpus: {exc}"
        ) from None


def _corpus_suite(corpus: EvalCorpusDocument, suite_id: str) -> EvalCorpusSuiteInspectionV1:
    from cayu.evals.corpus import inspect_eval_corpus

    suites = inspect_eval_corpus(corpus).suites
    suite = next((item for item in suites if item.id == suite_id), None)
    if suite is None:
        raise CloudEvalError(
            "suite_not_found",
            f"The corpus has no suite {suite_id}; its suites are "
            f"{', '.join(item.id for item in suites)}.",
        )
    return suite


def _published_target(api: CayuApi, target_key: str, *, require_ready: bool) -> dict[str, Any]:
    """The Agent's catalog entry for ``target_key``, ready to run when ``require_ready``."""

    catalog = api.get("/evals/targets", resource="eval targets")
    items = catalog.get("items") if isinstance(catalog, dict) else None
    if not isinstance(items, list):
        raise CloudEvalError("api_response_invalid", "The Agent's eval target list is invalid.")
    entries = [item for item in items if isinstance(item, dict)]
    keys = [str(item.get("target_key")) for item in entries]
    target = next((item for item in entries if item.get("target_key") == target_key), None)
    if target is None:
        raise CloudEvalError(
            "eval_target_not_found",
            f"This release has no eval target named {target_key}; its targets are "
            f"{', '.join(keys) or 'none'}.",
            details={"target_key": target_key, "targets": keys},
        )
    profile = target.get("execution_profile")
    ready = target.get("execution_profile_ready") is True and isinstance(profile, dict)
    if require_ready and not ready:
        diagnostics = [
            item
            for item in target.get("execution_profile_diagnostics") or []
            if isinstance(item, dict)
        ]
        reasons = " ".join(
            f"{item.get('message')} {item.get('remediation')}" for item in diagnostics
        )
        raise CloudEvalError(
            "eval_target_not_ready",
            f"The eval target {target_key} can't run yet." + (f" {reasons}" if reasons else ""),
            details={
                "target_key": target_key,
                "diagnostics": [item.get("code") for item in diagnostics],
            },
        )
    return target


def _check_bounds(
    target: Mapping[str, Any],
    suite: EvalCorpusSuiteInspectionV1,
    *,
    max_concurrency: int,
    cost_cap: Mapping[str, Any] | None,
) -> None:
    """Refuse, before anything is admitted, a run the target's published limits refuse."""

    key = target["target_key"]
    refusals: list[str] = []
    if suite.trials > target["max_trials"]:
        refusals.append(
            f"Suite {suite.id} runs {suite.trials} trials per case, and the eval target {key} "
            f"allows at most {target['max_trials']}. The trial count is part of the corpus: "
            "add a suite with fewer trials to run fewer."
        )
    max_cases = target["execution_profile"].get("ceilings", {}).get("max_cases")
    if isinstance(max_cases, int) and suite.case_count > max_cases:
        refusals.append(
            f"Suite {suite.id} has {suite.case_count} cases, and the eval target {key} runs at "
            f"most {max_cases}."
        )
    if suite.timeout_seconds > target["max_timeout_seconds"]:
        refusals.append(
            f"Suite {suite.id} gives each trial {suite.timeout_seconds}s, and the eval target "
            f"{key} allows at most {target['max_timeout_seconds']}s."
        )
    if max_concurrency > target["max_concurrency"]:
        refusals.append(
            f"--max-concurrency {max_concurrency} is more than the eval target {key} allows "
            f"({target['max_concurrency']})."
        )
    if max_concurrency > suite.max_concurrency:
        refusals.append(
            f"--max-concurrency {max_concurrency} is more than suite {suite.id}'s trial policy "
            f"allows ({suite.max_concurrency})."
        )
    if cost_cap is not None:
        currencies = [str(item) for item in target.get("cost_budget_currencies") or []]
        if target.get("cost_budget_available") is not True or not currencies:
            refusals.append(
                f"The eval target {key} publishes no pricing, so it can't interrupt a trial at "
                "a cost; run it without --max-cost."
            )
        elif cost_cap["currency"] not in currencies:
            refusals.append(
                f"The eval target {key} prices runs in {', '.join(currencies)}, not "
                f"{cost_cap['currency']}; pass --currency with one of them."
            )
    if refusals:
        raise CloudEvalError(
            "eval_run_out_of_bounds",
            " ".join(refusals),
            details={"target_key": key, "suite_id": suite.id},
        )


def _bound(
    target: Mapping[str, Any],
    corpus: EvalCorpusDocument,
    suite: EvalCorpusSuiteInspectionV1,
    *,
    max_concurrency: int,
    cost_cap: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The most work the run may take, and what bounds its cost, stated before it is admitted.

    `--max-cost` becomes the candidate's per-trial cost threshold: it interrupts the agent
    under evaluation, not the AI judges, whose evaluations run under their own judge profiles.
    """

    threshold = None
    if cost_cap is not None:
        threshold = {
            "per_trial": str(cost_cap["max_estimated_cost"]),
            "currency": cost_cap["currency"],
        }
    profiles = {
        item.get("key"): item
        for item in target.get("judge_profiles") or []
        if isinstance(item, dict)
    }
    judges = []
    for key, assertions in sorted(_suite_judge_assertions(corpus, suite.id).items()):
        profile = profiles.get(key)
        judges.append(
            {
                "key": key,
                "evaluations": assertions * suite.trials,
                "limits": None
                if profile is None
                else {
                    name: profile.get(name)
                    for name in (
                        "max_estimated_cost",
                        "cost_currency",
                        "max_total_tokens",
                        "timeout_seconds",
                    )
                },
            }
        )
    return {
        "target_key": target["target_key"],
        "suite_id": suite.id,
        "cases": suite.case_count,
        "trials_per_case": suite.trials,
        "total_trials": suite.case_count * suite.trials,
        "max_concurrency": max_concurrency,
        "max_steps_per_trial": target.get("max_steps"),
        "timeout_seconds_per_trial": suite.timeout_seconds,
        "candidate_cost_threshold": threshold,
        "judges": judges,
    }


def _suite_judge_assertions(corpus: EvalCorpusDocument, suite_id: str) -> dict[str, int]:
    """The suite's AI-judge assertions per case trial, by judge profile key."""

    from cayu.evals.corpus import ModelJudgeAssertionSpec, StructuredModelJudgeAssertionSpec

    counts: dict[str, int] = {}
    for case in corpus.cases:
        if case.suite_id != suite_id:
            continue
        for assertion in case.assertions:
            if type(assertion) is ModelJudgeAssertionSpec:
                key = assertion.evaluator_key
            elif type(assertion) is StructuredModelJudgeAssertionSpec:
                key = assertion.judge_profile_key
            else:
                continue
            counts[key] = counts.get(key, 0) + 1
    return counts


def _bound_text(bound: Mapping[str, Any]) -> str:
    steps = bound["max_steps_per_trial"]
    lines = [
        f"Suite {bound['suite_id']} on eval target {bound['target_key']}: "
        f"{_count(bound['cases'], 'case')} x "
        f"{_count(bound['trials_per_case'], 'trial')} = {_count(bound['total_trials'], 'trial')}, "
        f"at most {bound['max_concurrency']} running at a time.",
        "Each trial: "
        + ("no published step limit" if steps is None else f"up to {_count(steps, 'model step')}")
        + f" and {bound['timeout_seconds_per_trial']}s.",
    ]
    threshold = bound["candidate_cost_threshold"]
    if threshold is None:
        lines.append(
            "No candidate cost threshold: pass --max-cost to interrupt each trial's agent at "
            "an amount."
        )
    else:
        lines.append(
            "Candidate cost threshold: each trial's agent is interrupted once its observed "
            f"cost reaches {threshold['per_trial']} {threshold['currency']}; one model call "
            "can cross it before its cost is known."
        )
    if bound["judges"]:
        lines.append(
            "AI judges run under their own judge profiles, not --max-cost: "
            + "; ".join(_judge_text(judge) for judge in bound["judges"])
            + "."
        )
    return "\n".join(lines)


def _judge_text(judge: Mapping[str, Any]) -> str:
    text = f"{judge['key']} ({_count(judge['evaluations'], 'evaluation')})"
    limits = judge["limits"]
    if limits is None:
        return text + " with no published limits"
    cost = limits["max_estimated_cost"]
    threshold = (
        "no cost threshold"
        if cost is None
        else f"a cost threshold of {cost} {limits['cost_currency']} "
        "(one model call can exceed it before its cost is known)"
    )
    return (
        f"{text}, each with {threshold}, {_count(limits['max_total_tokens'], 'token')} and "
        f"{limits['timeout_seconds']}s"
    )


def _admit(
    api: CayuApi,
    *,
    corpus_revision: str,
    suite_id: str,
    profile_revision: str,
    profile_changed: bool,
    max_concurrency: int,
    cost_cap: Mapping[str, Any] | None,
    operation_id: str,
    commands: Mapping[str, str],
    import_corpus: Callable[[], None] | None,
) -> dict[str, Any]:
    """Admit the run, or replay the admission ``operation_id`` already made.

    ``import_corpus`` (a continuation) runs only when the Agent answers that it doesn't store
    the corpus; the same request then goes again under the same key.
    """

    body: dict[str, Any] = {
        "corpus_revision": corpus_revision,
        "suite_id": suite_id,
        "expected_execution_profile_revision": profile_revision,
        "max_concurrency": max_concurrency,
    }
    if cost_cap is not None:
        body["cost_budget"] = {
            "max_estimated_cost": str(cost_cap["max_estimated_cost"]),
            "currency": cost_cap["currency"],
        }

    def send() -> Any:
        try:
            return api.mutate("/evals/runs", body, resource="eval run", operation_id=operation_id)
        except CayuApiError as exc:
            refusal = _admission_refusal(
                exc, operation_id=operation_id, profile_changed=profile_changed, commands=commands
            )
            if refusal is None:
                raise
            raise refusal from None

    try:
        record = send()
    except CayuApiNotFound as exc:
        if import_corpus is None or exc.detail != _CORPUS_NOT_FOUND:
            raise
        # A definite refusal that admitted nothing, so sending the request again is safe.
        import_corpus()
        record = send()
    return _run_summary(record)


def _admission_refusal(
    exc: CayuApiError,
    *,
    operation_id: str,
    profile_changed: bool,
    commands: Mapping[str, str],
) -> CloudEvalError | None:
    """What an admission's failure means for its operation ID, when that is known."""

    if exc.category == "outcome_unknown":
        return CloudEvalError(
            "outcome_unknown",
            "The Agent did not confirm the eval run's admission; it may have admitted it. "
            f"`{commands['continue']}` admits nothing new if it did.",
            status_code=exc.status_code,
            details={"operation_id": operation_id, "commands": dict(commands)},
        )
    if exc.detail == _IDEMPOTENCY_CONFLICT:
        return CloudEvalError(
            "operation_id_conflict",
            f"Operation ID {operation_id} already admitted a different eval run request (its "
            "corpus file or options have changed since), so this one admitted nothing. The run "
            f"it admitted still exists: `{commands['list']}` lists the eval target's runs.",
            status_code=exc.status_code,
            details={"operation_id": operation_id, "commands": {"list": commands["list"]}},
        )
    if profile_changed and exc.detail in _PROFILE_REFUSALS:
        # The Agent replays an admitted request before checking it against its current
        # release, so its release refusing the exact request means this key admitted nothing.
        return CloudEvalError(
            "eval_release_changed",
            "The Agent's release changed before this run was admitted, so there is no run "
            f"to continue ({exc}). Start a new run without --operation-id and "
            "--execution-profile-revision.",
            status_code=exc.status_code,
            details={"operation_id": operation_id},
        )
    return None


def _wait_for_run(
    api: CayuApi,
    run: dict[str, Any],
    *,
    wait_seconds: float,
    poll_seconds: float,
    operation_id: str,
    commands: Mapping[str, str],
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Read the run until it ends, printing its progress whenever it changes."""

    deadline = monotonic() + wait_seconds
    run_id = run["id"]
    progress: dict[str, Any] | None = None
    shown: str | None = None
    unreachable = False
    while True:
        try:
            current = _read_run(api, run_id)
            progress = _read_progress(api, run_id)
        except CayuApiError as exc:
            if not _transient(exc):
                raise
            if not unreachable:
                _say(f"{exc} Run {run_id} goes on in the Agent; reading it again.")
            unreachable = True
        else:
            unreachable = False
            run = current
            line = _progress_line(run, progress)
            if line != shown:
                _say(line)
                shown = line
            if run["status"] in TERMINAL_RUN_STATUSES:
                return run, progress
        if monotonic() >= deadline:
            raise CloudEvalError(
                "eval_run_still_running",
                f"The eval run {run_id} is still {run['status']} after waiting "
                f"{wait_seconds:.0f}s; the Agent keeps running it.",
                details={
                    "run_id": run_id,
                    "status": run["status"],
                    "operation_id": operation_id,
                    "progress": progress,
                    "commands": dict(commands),
                },
            )
        sleep(poll_seconds)


def _unfinished_run_error(
    run: Mapping[str, Any], *, operation_id: str, commands: Mapping[str, str]
) -> CloudEvalError:
    run_id, status = run["id"], run["status"]
    details: dict[str, object] = {
        "run_id": run_id,
        "status": status,
        "operation_id": operation_id,
        "commands": {"show": commands["show"]},
    }
    if status == "cancelled":
        return CloudEvalError(
            "eval_run_cancelled", f"The eval run {run_id} was cancelled.", details=details
        )
    diagnostic = run.get("failure_diagnostic")
    reason = diagnostic.get("reason") if isinstance(diagnostic, dict) else None
    details.update(failure_code=run.get("failure_code"), failure_reason=reason)
    return CloudEvalError(
        "eval_run_failed",
        f"The eval run {run_id} failed: {run.get('failure_code')}"
        + (f" ({reason})." if reason else "."),
        details=details,
    )


def _report(api: CayuApi, run_id: str, *, commands: Mapping[str, str]) -> dict[str, Any]:
    try:
        report = api.get(f"{_run_path(run_id)}/report.json", resource="eval report")
    except CayuApiError as exc:
        raise CloudEvalError(
            exc.category,
            f"The eval run {run_id} completed, but its report could not be read: {exc}",
            status_code=exc.status_code,
            details={"run_id": run_id, "commands": {"continue": commands["continue"]}},
        ) from None
    if not isinstance(report, dict) or report.get("record_type") != _REPORT_RECORD_TYPE:
        raise CloudEvalError(
            "api_response_invalid", f"The Agent's report for eval run {run_id} is invalid."
        )
    return report


def _report_summary(report: Mapping[str, Any], *, run_id: str) -> dict[str, Any]:
    """The result's status, score and trial outcomes, from the report's presentation."""

    presentation = report.get("presentation")
    if not isinstance(presentation, dict):
        raise CloudEvalError(
            "api_response_invalid", f"The Agent's report for eval run {run_id} is invalid."
        )
    trials: dict[str, int] = {"passed": 0, "failed": 0, "error": 0, "unavailable": 0}
    for case in presentation.get("cases") or []:
        for trial in (case.get("trials") or []) if isinstance(case, dict) else []:
            status = trial.get("status") if isinstance(trial, dict) else None
            if isinstance(status, str):
                trials[status] = trials.get(status, 0) + 1
    return {
        "result_revision": presentation.get("result_revision"),
        "status": presentation.get("status"),
        "score": presentation.get("score"),
        "trials": trials,
    }


def _write_report(
    path: Path, report: Mapping[str, Any], *, run_id: str, commands: Mapping[str, str]
) -> None:
    # Written as the Agent renders it: sorted keys, two-space indent, a final newline.
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise CloudEvalError(
            "output_unwritable",
            f"Could not write the report of eval run {run_id} to {path}: "
            f"{exc.strerror or type(exc).__name__}. The run completed; `{commands['continue']}` "
            "downloads its report again.",
            details={"run_id": run_id, "commands": {"continue": commands["continue"]}},
        ) from None


def _list_runs(api: CayuApi, arguments: argparse.Namespace) -> dict[str, Any]:
    query = {"limit": str(arguments.limit)}
    if arguments.target_key is not None:
        query["target_key"] = arguments.target_key
    if arguments.status is not None:
        query["status"] = arguments.status
    page = api.get("/evals/runs", query=query, resource="eval runs")
    items = page.get("items") if isinstance(page, dict) else None
    if not isinstance(items, list):
        raise CloudEvalError("api_response_invalid", "The Agent's eval run list is invalid.")
    return {
        "items": [_run_summary(item) for item in items],
        "has_more": page.get("has_more") is True,
    }


def _show_run(api: CayuApi, run_id: str) -> dict[str, Any]:
    return {"run": _read_run(api, run_id), "progress": _read_progress(api, run_id)}


def _cancel_run(api: CayuApi, run_id: str) -> dict[str, Any]:
    try:
        record = api.mutate(f"{_run_path(run_id)}/cancel", {}, resource="eval run cancellation")
    except CayuApiError as exc:
        if exc.category != "outcome_unknown":
            raise
        raise CloudEvalError(
            "outcome_unknown",
            f"The Agent did not confirm cancelling eval run {run_id}; it may have. Run the "
            "command again: cancelling a run twice is harmless.",
            status_code=exc.status_code,
            details={"run_id": run_id},
        ) from None
    return {"run": _run_summary(record)}


def _compare(api: CayuApi, baseline_run_id: str, current_run_id: str) -> dict[str, Any]:
    answer = api.post(
        "/evals/comparisons",
        {"baseline_run_id": baseline_run_id, "current_run_id": current_run_id},
        resource="eval comparison",
    )
    comparison = answer.get("comparison") if isinstance(answer, dict) else None
    if not isinstance(comparison, dict):
        raise CloudEvalError("api_response_invalid", "The Agent's eval comparison is invalid.")
    return {
        "baseline": _run_summary(answer.get("baseline")),
        "current": _run_summary(answer.get("current")),
        "comparison": comparison,
    }


def _approve_baseline(
    api: CayuApi,
    arguments: argparse.Namespace,
    *,
    application_id: str,
    recovery_arguments: Sequence[str],
) -> dict[str, Any]:
    """Approve the run's result as its suite's baseline, CAS-bound to one generation.

    Without ``--expected-generation`` the decision is made over the current baseline under a
    fresh operation ID. With it (a continuation), both are sent as given, so the Agent replays
    a decision it already applied and refuses one the baseline has moved past.
    """

    run_id = arguments.run_id
    if arguments.operation_id is not None and arguments.expected_generation is None:
        raise CloudEvalError(
            "invalid_input",
            "--operation-id repeats an earlier approval; pass it with that approval's "
            "--expected-generation.",
        )
    run, revision, current = _run_result(api, run_id)
    if arguments.expected_generation is None:
        if current is not None and current.get("result_revision") == revision:
            return {"run": run, "baseline": current, "changed": False}
        generation = _baseline_generation(current)
    else:
        generation = arguments.expected_generation
    operation_id = arguments.operation_id or "sha256:" + secrets.token_hex(32)
    decision = {"run_id": run_id, "operation_id": operation_id, "expected_generation": generation}
    command = [
        *("cayu", "cloud", *recovery_arguments, "eval", "baseline"),
        *("--application", application_id, run_id),
        *("--operation-id", operation_id, "--expected-generation", str(generation)),
    ]
    commands = {"continue": shlex.join(command)}
    try:
        answer = api.mutate(
            f"/evals/results/{quote(revision, safe='')}/baseline",
            {
                "result_revision": revision,
                "expected_generation": generation,
                "operation_id": operation_id,
            },
            resource="eval baseline",
        )
    except CayuApiError as exc:
        if exc.category == "outcome_unknown":
            raise CloudEvalError(
                "outcome_unknown",
                f"The Agent did not confirm approving eval run {run_id} as the baseline; it "
                f"may have. `{commands['continue']}` repeats this decision rather than making "
                "another.",
                status_code=exc.status_code,
                details={**decision, "commands": commands},
            ) from None
        if exc.status_code == 409:
            moved = _run_result(api, run_id)[2]
            if _baseline_generation(moved) != generation:
                raise CloudEvalError(
                    "eval_baseline_conflict",
                    f"The baseline of suite {run['suite_id']} moved to generation "
                    f"{_baseline_generation(moved)} after this approval was decided over "
                    f"generation {generation}, so it was not applied. Approve run {run_id} "
                    "again without --expected-generation to decide over the current baseline.",
                    status_code=exc.status_code,
                    details={**decision, "baseline": moved},
                ) from None
        raise
    mutation = answer.get("mutation") if isinstance(answer, dict) else None
    baseline = answer.get("baseline") if isinstance(answer, dict) else None
    if (
        not isinstance(mutation, dict)
        or not isinstance(baseline, dict)
        or mutation.get("selected_result_revision") != revision
        or mutation.get("expected_generation") != generation
    ):
        raise CloudEvalError("api_response_invalid", "The Agent did not confirm the baseline.")
    # A replayed decision answers the mutation it applied and the baseline as it is now.
    return {**decision, "run": run, "baseline": baseline, "mutation": mutation, "changed": True}


def _run_result(api: CayuApi, run_id: str) -> tuple[dict[str, Any], str, dict[str, Any] | None]:
    """The completed run, its result revision and its suite's current baseline."""

    detail = api.get(f"{_run_path(run_id)}/result", resource="eval run result")
    run = _run_summary(detail.get("run") if isinstance(detail, dict) else None)
    result = run["result"]
    if not isinstance(result, dict) or not isinstance(result.get("revision"), str):
        raise CloudEvalError("api_response_invalid", "The Agent's eval run result is invalid.")
    baseline = detail.get("baseline")
    if baseline is not None and (
        not isinstance(baseline, dict) or type(baseline.get("generation")) is not int
    ):
        raise CloudEvalError("api_response_invalid", "The Agent's eval baseline is invalid.")
    return run, result["revision"], baseline


def _baseline_generation(baseline: Mapping[str, Any] | None) -> int:
    return 0 if baseline is None else baseline["generation"]


def _read_run(api: CayuApi, run_id: str) -> dict[str, Any]:
    return _run_summary(api.get(_run_path(run_id), resource="eval run"))


def _read_progress(api: CayuApi, run_id: str) -> dict[str, Any] | None:
    """The run's trial progress, or ``None`` from an Agent whose store has no run views."""

    from cayu.evals.read_views import run_view_id

    try:
        view = api.get(
            f"/evals/run-views/{run_view_id('run', run_id)}",
            query={"cases_limit": "1"},
            resource="eval run progress",
        )
    except CayuApiError as exc:
        if exc.category in {"request_rejected", "unsupported"}:
            return None
        raise
    progress = view.get("progress") if isinstance(view, dict) else None
    return progress if isinstance(progress, dict) else None


def _run_summary(record: Any) -> dict[str, Any]:
    """What these commands show of an `EvalRunRecord`, without its execution profile."""

    if not isinstance(record, dict):
        raise CloudEvalError("api_response_invalid", "The Agent's eval run record is invalid.")
    spec = record.get("spec")
    status = record.get("status")
    if (
        not isinstance(spec, dict)
        or not isinstance(spec.get("run_id"), str)
        or status not in RUN_STATUSES
    ):
        raise CloudEvalError("api_response_invalid", "The Agent's eval run record is invalid.")
    invocation = spec.get("invocation")
    cost_budget = invocation.get("cost_budget") if isinstance(invocation, dict) else None
    return {
        "id": spec["run_id"],
        "status": status,
        "target_key": spec.get("target_key"),
        "corpus_revision": spec.get("corpus_revision"),
        "suite_id": spec.get("suite_id"),
        "max_concurrency": spec.get("max_concurrency"),
        "cost_budget": cost_budget if isinstance(cost_budget, dict) else None,
        "attempt_count": record.get("attempt_count"),
        "created_at": record.get("created_at"),
        "started_at": record.get("started_at"),
        "finished_at": record.get("finished_at"),
        "cancel_requested_at": record.get("cancel_requested_at"),
        "result": record.get("result"),
        "failure_code": record.get("failure_code"),
        "failure_diagnostic": record.get("failure_diagnostic"),
    }


def _progress_line(run: Mapping[str, Any], progress: Mapping[str, Any] | None) -> str:
    line = f"Run {run['id']}: {run['status']}"
    if progress is not None and isinstance(progress.get("completed_trials"), int):
        line += f", {progress['completed_trials']}/{progress.get('total_trials')} trials finished"
    return line


def _run_commands(
    arguments: argparse.Namespace,
    *,
    application_id: str,
    operation_id: str,
    profile_revision: str,
    target_key: str,
    recovery_arguments: Sequence[str],
) -> dict[str, str]:
    """The `eval run` that repeats this admission exactly, so it continues with its run, and
    the `runs list` of its eval target."""

    listing = [
        *("cayu", "cloud", *recovery_arguments, "eval", "runs", "list"),
        *("--application", application_id, "--target-key", target_key),
    ]
    command = [
        "cayu",
        "cloud",
        *recovery_arguments,
        "eval",
        "run",
        "--application",
        application_id,
        "--corpus",
        str(arguments.corpus),
        "--suite",
        arguments.suite,
        "--max-concurrency",
        str(arguments.max_concurrency),
    ]
    if arguments.max_cost is not None:
        command += ["--max-cost", str(arguments.max_cost)]
        command += ["--currency", arguments.currency or _DEFAULT_CURRENCY]
    command += ["--operation-id", operation_id]
    command += ["--execution-profile-revision", profile_revision]
    if arguments.output is not None:
        command += ["--output", str(arguments.output)]
    return {"continue": shlex.join(command), "list": shlex.join(listing)}


def _run_id_commands(
    application_id: str, run_id: str, recovery_arguments: Sequence[str]
) -> dict[str, str]:
    prefix = ["cayu", "cloud", *recovery_arguments, "eval", "runs"]
    suffix = ["--application", application_id, run_id]
    return {action: shlex.join([*prefix, action, *suffix]) for action in ("show", "cancel")}


@contextmanager
def _agent_api(
    client: CloudApiClient, application_id: str, timeout_seconds: float
) -> Iterator[CayuApi]:
    """The Agent's Cayu API; its failures become `cayu cloud` errors."""

    try:
        api = cloud_agent_api(
            client,
            application_id,
            timeout_seconds=timeout_seconds,
            wake_timeout_seconds=_WAKE_TIMEOUT_SECONDS,
        )
    except CayuApiError as exc:
        raise CloudEvalError(exc.category, str(exc), status_code=exc.status_code) from None
    try:
        # Find the Agent's Cayu API first, so a Cayu without eval routes is told apart from an
        # Agent that serves no Cayu at all.
        api.get("/app", resource="agent")
        yield api
    except CayuApiError as exc:
        if exc.category == "unsupported" and exc.status_code == 404:
            raise CloudEvalError(
                "evals_unavailable",
                f"{api.description} serves no evals: its Cayu predates them, or it runs "
                "without a durable eval store.",
                status_code=exc.status_code,
            ) from None
        raise CloudEvalError(exc.category, str(exc), status_code=exc.status_code) from None
    finally:
        api.close()


def _application_id(client: CloudApiClient, reference: str) -> str:
    from cayu.cli.cloud import resolve_application_id

    return resolve_application_id(client, reference)


def _run_path(run_id: str) -> str:
    return f"/evals/runs/{quote(run_id, safe='')}"


def _transient(exc: CayuApiError) -> bool:
    if exc.category in _TRANSIENT_CATEGORIES:
        return True
    return exc.category == "server_error" and (exc.status_code or 0) >= 500


def _say(text: str) -> None:
    print(terminal_safe(text, keep_lines=True), file=sys.stderr, flush=True)


def _count(number: object, noun: str) -> str:
    return f"{number} {noun}" + ("" if number == 1 else "s")


def _add_application_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--application", required=True, help="The Agent's slug or name.")


def _positive_count(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number") from None
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _list_limit(value: str) -> int:
    number = _positive_count(value)
    if number > MAX_LIST_LIMIT:
        raise argparse.ArgumentTypeError(f"must be at most {MAX_LIST_LIMIT}")
    return number


def _cost_amount(value: str) -> Decimal:
    try:
        amount = Decimal(value)
    except InvalidOperation:
        raise argparse.ArgumentTypeError(f"{value!r} is not an amount such as 0.50") from None
    if not amount.is_finite() or amount <= 0:
        raise argparse.ArgumentTypeError("must be a positive amount")
    return amount


def _currency(value: str) -> str:
    code = value.upper()
    if _CURRENCY.fullmatch(code) is None:
        raise argparse.ArgumentTypeError(f"{value!r} is not a currency code such as USD")
    return code


def _eval_revision(value: str) -> str:
    if _EVAL_REVISION.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be sha256: and 64 lowercase hex digits")
    return value


def _generation(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a whole number") from None
    if number < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return number


def _operation_id(value: str) -> str:
    if (
        not value
        or len(value) > _OPERATION_ID_MAX_CHARS
        or value != value.strip()
        or any(not " " <= character <= "~" for character in value)
    ):
        raise argparse.ArgumentTypeError(
            f"must be 1-{_OPERATION_ID_MAX_CHARS} printable ASCII characters"
        )
    return value


def render_run(answer: Mapping[str, Any]) -> str:
    result = answer["result"]
    run = result["run"]
    lines = [f"Run {_text(run['id'])}: {_text(run['status'])}"]
    summary = result["summary"]
    if summary is not None:
        score = summary["score"]
        lines[0] += (
            f" ({_text(summary['status'])}" + ("" if score is None else f", score {score:g}") + ")"
        )
        trials = summary["trials"]
        lines.append(
            f"Trials: {trials.get('passed', 0)} passed, {trials.get('failed', 0)} failed, "
            f"{trials.get('error', 0)} error, {trials.get('unavailable', 0)} unavailable"
        )
    if result["output"] is not None:
        lines.append(f"Report: {_text(result['output'])}")
    elif summary is None:
        lines.append(f"Progress: {result['commands']['show']}")
    return "\n".join(lines)


def render_runs(answer: Mapping[str, Any]) -> str:
    listed = answer["result"]
    items = listed["items"]
    if not items:
        return f"No eval runs for {_cell(listed['application_id'])}."
    rows = [
        (
            _cell(item["id"]),
            _cell(item["status"]),
            _cell(item["suite_id"]),
            _cell(item["created_at"]),
            _result_cell(item),
        )
        for item in items
    ]
    lines = _table(("RUN", "STATUS", "SUITE", "CREATED", "RESULT"), rows)
    if listed["has_more"]:
        lines.append(f"Showing the newest {len(items)}; pass a larger --limit for more.")
    return "\n".join(lines)


def render_run_detail(answer: Mapping[str, Any]) -> str:
    result = answer["result"]
    run, progress = result["run"], result["progress"]
    lines = [
        _text(_progress_line(run, progress)),
        f"Suite: {_text(run['suite_id'])} on eval target {_text(run['target_key'])}",
        f"Corpus: {_text(run['corpus_revision'])}",
        f"Created: {_text(run['created_at'])}",
    ]
    if run["finished_at"] is not None:
        lines.append(f"Finished: {_text(run['finished_at'])}")
    if run["result"] is not None or run["failure_code"] is not None:
        lines.append(f"Result: {_result_cell(run)}")
    return "\n".join(lines)


def render_cancel(answer: Mapping[str, Any]) -> str:
    run = answer["result"]["run"]
    run_id, status = _text(run["id"]), run["status"]
    if status == "cancelling":
        return f"Cancelling run {run_id}: its running trials stop and it ends cancelled."
    if status in TERMINAL_RUN_STATUSES and status != "cancelled":
        return f"Run {run_id} already ended {_text(status)}; cancelling it changed nothing."
    return f"Run {run_id} is {_text(status)}."


def render_comparison(answer: Mapping[str, Any]) -> str:
    result = answer["result"]
    comparison = result["comparison"]
    lines = [
        f"Baseline {_text(result['baseline']['id'])}: {_result_cell(result['baseline'])}",
        f"Current  {_text(result['current']['id'])}: {_result_cell(result['current'])}",
    ]
    compatibility = comparison.get("compatibility") or {}
    if compatibility.get("comparable") is not True:
        reasons = ", ".join(str(reason) for reason in compatibility.get("reasons") or [])
        lines.append(f"Not comparable: {_text(reasons or 'unknown reason')}.")
        return "\n".join(lines)
    regressions = comparison.get("regressions") or []
    lines.append(f"Comparable; {_count(len(regressions), 'regression')}.")
    for regression in regressions:
        scope = regression.get("case_id") or regression.get("scope")
        lines.append(f"  {_text(scope)}: {_text(regression.get('kind'))}")
    return "\n".join(lines)


def render_baseline(answer: Mapping[str, Any]) -> str:
    result = answer["result"]
    run, baseline = result["run"], result["baseline"]
    suite = _text(run["suite_id"])
    if not result["changed"]:
        return (
            f"Run {_text(run['id'])} was already the baseline of suite {suite} "
            f"(generation {_text(baseline.get('generation'))})."
        )
    applied = result["mutation"].get("resulting_generation")
    line = f"Run {_text(run['id'])} is now the baseline of suite {suite} (generation {applied})."
    if baseline.get("generation") != applied:
        line = (
            f"Run {_text(run['id'])} was approved as the baseline of suite {suite} (generation "
            f"{_text(applied)}); it has since moved to generation "
            f"{_text(baseline.get('generation'))}."
        )
    return line


def render_failure(error: Mapping[str, Any]) -> str:
    lines = [f"Error: {_text(error['message'])}"]
    for label, key in (("Run", "run_id"), ("Operation ID", "operation_id")):
        if isinstance(error.get(key), str):
            lines.append(f"{label}: {_text(error[key])}")
    commands = error.get("commands")
    if isinstance(commands, dict):
        lines += [f"  {_text(name)}: {_text(command)}" for name, command in commands.items()]
    return "\n".join(lines)


def _result_cell(run: Mapping[str, Any]) -> str:
    result = run.get("result")
    if isinstance(result, dict):
        score = result.get("score")
        return _text(result.get("status")) + ("" if score is None else f" {score:g}")
    if run.get("failure_code") is not None:
        return f"failed: {_text(run['failure_code'])}"
    return "-"


def _text(value: object) -> str:
    return "-" if value is None else terminal_safe(str(value))
