"""`cayu eval targets` and `cayu eval suites draft|check|run`: author and run suites from the
terminal.

These are the Developer dashboard's suite editor and launch actions (**New suite**, **Edit
suite**, **Check suite**, **Save revision**, **Run saved revision** and a corpus suite's **Run
suite**) and its Settings readiness, through the same server API:

- `targets list` reads the eval targets (`GET /evals/targets`): each one's readiness, execution
  profile, limits, pricing and judge profiles.
- `suites draft` writes an editable suite draft: a saved authored revision's (`GET
  /evals/suites/{revision}`), made a draft as **Edit suite** makes it, or a new suite with
  **New suite**'s sample case.
- `suites check` is **Check suite** (`POST /evals/suites/preview`); `--save` is **Save
  revision** (`POST /evals/suites`). A revision is its content, so saving it again changes
  nothing.
- `suites run` is **Run saved revision** of an authored suite (`POST
  /evals/suites/{revision}/runs/preview`, then `POST .../runs`) and **Run suite** of a corpus
  suite (`POST /evals/runs`), under an idempotency key. `--preview` is **Check launch**: it
  states the work and runs nothing. The printed run command carries the key and the reviewed
  execution profiles (and an authored suite's exposure) and sends that admission as given, so
  repeating it answers with the runs the key admitted.
"""

from __future__ import annotations

import argparse
import json
import secrets
import shlex
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

from cayu.cli._cloud_agent_api import CayuApi, CayuApiError
from cayu.cli._cloud_evals import (
    _DEFAULT_CURRENCY,
    _EVAL_REVISION,
    _IDEMPOTENCY_CONFLICT,
    CloudEvalError,
    _admit,
    _bound,
    _check_bounds,
    _corpus_suite,
    _cost_amount,
    _currency,
    _eval_revision,
    _operation_id,
    _positive_count,
    _published_target,
    _run_summary,
    _text,
)
from cayu.cli._cloud_settings import _cell, _table
from cayu.cli._eval_calibration import _json_object
from cayu.cli._eval_curation import _diagnosed_refusal

# The dashboard's new suite (`newEvalSuiteDraft`): one simple case with sample input and the
# completed-session check.
_NEW_SUITE_NAME = "Evaluation suite"
_NEW_CASE = {
    "id": "case-1",
    "name": "Case 1",
    "description": None,
    "source": None,
    "stimulus": {
        "kind": "simple_input",
        "input": {"messages": [{"role": "user", "text": "Describe the behavior to evaluate."}]},
    },
    "assertions": [{"id": "root_status", "kind": "root_status", "expected": "completed"}],
}
_DEFAULT_CORPUS_CONCURRENCY = 1
# The field of a launch group's reviewed execution profile.
_PROFILE = "execution_profile_revision"
# The server's refusals of an authored launch (`cayu.server._suite_launch_routes`): the
# selection can't run now, and the reviewed exposure or execution profiles changed since.
_LAUNCH_NOT_READY = "Authored eval suite launch requirements are not currently ready."
_LAUNCH_CHANGED = frozenset(
    {
        "The authored-suite maximum work or cost exposure changed after readiness. Check "
        "launch readiness again.",
        "The authored-suite execution profile changed after readiness. Check launch "
        "readiness again.",
    }
)
# The server's refusals of a save (`cayu.server._suite_authoring_routes`): the revision is
# stored under another parent, or the parent is another suite's; and a scenario a case runs
# changed after the check.
_SAVE_CONFLICT = "Authored eval suite revision conflicts with stored content."
_SCENARIOS_CHANGED = "Authored eval suite scenario references changed before save."


def add_target_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> list[tuple[argparse.ArgumentParser, Callable[[Mapping[str, Any]], str]]]:
    """Add `targets` to `cayu eval`; return its parsers with their text renderers."""

    targets = commands.add_parser(
        "targets",
        help="List the eval targets and their readiness.",
        description=(
            "List the eval targets a running server or Cloud Agent publishes, as Evals → "
            "Settings shows them: readiness and execution profile, limits, pricing and judge "
            "profiles."
        ),
    )
    actions = targets.add_subparsers(dest="eval_targets_command", required=True)
    listing = actions.add_parser(
        "list",
        help="List the eval targets.",
        description=(
            "List the eval targets, the default one first named by `default_target_key`. A "
            "target that can't run says why in `execution_profile_diagnostics`."
        ),
    )
    return [(listing, render_targets)]


def add_suite_authoring_parsers(
    suite_actions: argparse._SubParsersAction[argparse.ArgumentParser],
) -> list[tuple[argparse.ArgumentParser, Callable[[Mapping[str, Any]], str]]]:
    """Add `draft`, `check` and `run` to `cayu eval suites`; return their parsers with their
    text renderers."""

    draft = suite_actions.add_parser(
        "draft",
        help="Write an editable suite draft.",
        description=(
            "Write an editable draft of a saved authored suite, as Edit suite opens it, or of a "
            "new suite with one sample case, as New suite starts it. Edit the file, then check "
            "and save it with `suites check --file FILE --save`."
        ),
    )
    source = draft.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "suite", nargs="?", metavar="SUITE", help="An authored suite's id from `suites list`."
    )
    source.add_argument("--new", metavar="SUITE_ID", help="Start a new suite with this id.")
    draft.add_argument(
        "--target-key",
        metavar="KEY",
        help="With --new: the eval target it runs on (default: the server's default).",
    )
    draft.add_argument(
        "--name", metavar="NAME", help=f"With --new: its name (default: {_NEW_SUITE_NAME})."
    )
    draft.add_argument(
        "--file",
        type=Path,
        required=True,
        metavar="FILE",
        help="Where to write the draft; the file must not exist yet.",
    )

    check = suite_actions.add_parser(
        "check",
        help="Check a suite draft, and save it.",
        description=(
            "Compile a suite draft into its immutable revision and check whether it is ready "
            "to save, as Check suite does; --save then saves that revision, as Save revision "
            "does. Saving runs nothing, and saving a saved revision again changes nothing."
        ),
    )
    check.add_argument(
        "--file",
        type=Path,
        required=True,
        metavar="FILE",
        help="A suite draft, such as `suites draft` writes, or a saved revision's JSON.",
    )
    check.add_argument("--save", action="store_true", help="Save the checked revision.")
    check.add_argument(
        "--parent",
        type=_eval_revision,
        metavar="REVISION",
        help=(
            "The saved revision the draft edits, from `suites draft SUITE`; saved as its "
            "parent, so cases left unchanged keep their link to the session they came from."
        ),
    )

    run = suite_actions.add_parser(
        "run",
        help="Run a saved suite.",
        description=(
            "Admit runs of a saved suite under an idempotency key, as Run saved revision does "
            "for an authored suite and Run suite for a corpus suite. --preview states the work "
            "and runs nothing, as Check launch does, and prints the command that runs it. The "
            "command returns once the runs are admitted; `cayu eval runs show RUN_ID` follows "
            "each."
        ),
    )
    run.add_argument("suite", metavar="SUITE", help="The suite's id from `suites list`.")
    run.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        metavar="CASE_ID",
        help="Run only this case of an authored suite; repeat for more (default: every case).",
    )
    run.add_argument(
        "--max-concurrency",
        type=_positive_count,
        metavar="N",
        help=(
            f"A corpus suite's trials running at once (default {_DEFAULT_CORPUS_CONCURRENCY}). "
            "An authored suite's trial policy sets its own."
        ),
    )
    run.add_argument(
        "--max-cost",
        type=_cost_amount,
        metavar="AMOUNT",
        help="Interrupt each trial's agent once its observed cost reaches AMOUNT, such as 0.50.",
    )
    run.add_argument(
        "--currency",
        type=_currency,
        metavar="CODE",
        help=f"The currency of --max-cost (default {_DEFAULT_CURRENCY}).",
    )
    run.add_argument(
        "--preview", action="store_true", help="State the work and print the run command."
    )
    run.add_argument(
        "--operation-id",
        type=_operation_id,
        metavar="ID",
        help=(
            "The launch's idempotency key. Without one a fresh key is printed in the command "
            "that continues the launch."
        ),
    )
    run.add_argument(
        "--expected-exposure-revision",
        type=_eval_revision,
        metavar="REV",
        help="An authored suite's reviewed work exposure, from a printed run command.",
    )
    run.add_argument(
        "--expected-execution-profile",
        action="append",
        type=_profile_expectation,
        metavar="CASES=REV",
        help=(
            "One reviewed launch group of an authored suite, its comma-separated case ids and "
            "execution profile revision, from a printed run command; repeated per group."
        ),
    )
    run.add_argument(
        "--expected-execution-profile-revision",
        type=_eval_revision,
        metavar="REV",
        help="A corpus suite's reviewed execution profile, from a printed run command.",
    )
    return [(draft, render_draft), (check, render_suite_check), (run, render_suite_run)]


def check_suite_options(args: argparse.Namespace) -> dict[str, Any] | None:
    """Check the options and read a draft file, before any request; return the draft."""

    action = args.eval_suites_command
    if action == "draft":
        if args.new is None and (args.target_key is not None or args.name is not None):
            raise ValueError("--target-key and --name describe a new suite; pass them with --new.")
        if args.file.exists():
            raise ValueError(
                f"--file {args.file} exists; name a new file, so no edits are overwritten."
            )
        return None
    if action == "check":
        return _draft_from_file(args.file)
    if action != "run":
        return None
    if args.currency is not None and args.max_cost is None:
        raise ValueError("--currency is the currency of --max-cost; pass it with --max-cost.")
    authored = (args.expected_exposure_revision, args.expected_execution_profile)
    if (authored[0] is None) != (authored[1] is None):
        raise ValueError(
            "--expected-exposure-revision and --expected-execution-profile name a reviewed "
            "launch together; pass both, from a printed run command."
        )
    pinned = authored[0] is not None or args.expected_execution_profile_revision is not None
    if authored[0] is not None and args.expected_execution_profile_revision is not None:
        raise ValueError(
            "--expected-execution-profile-revision pins a corpus suite's run, and "
            "--expected-execution-profile an authored suite's; pass one."
        )
    if pinned and args.operation_id is None:
        raise ValueError(
            "A reviewed launch is repeated under its key; pass --operation-id with it, from "
            "the printed run command."
        )
    if args.preview and (pinned or args.operation_id is not None):
        raise ValueError(
            "--preview reviews a new launch; leave out --operation-id and the expected revisions."
        )
    if args.case_ids is not None and len(set(args.case_ids)) != len(args.case_ids):
        raise ValueError("Each --case names a different case.")
    return None


def run_target_command(api: CayuApi) -> dict[str, Any]:
    catalog = api.get("/evals/targets", resource="eval targets")
    items = catalog.get("items") if isinstance(catalog, dict) else None
    if not isinstance(items, list) or not isinstance(catalog.get("default_target_key"), str):
        raise CloudEvalError("api_response_invalid", "The server's eval target list is invalid.")
    return {"items": items, "default_target_key": catalog["default_target_key"]}


def run_suite_command(
    api: CayuApi,
    args: argparse.Namespace,
    file_draft: dict[str, Any] | None,
    *,
    target: Sequence[str],
) -> dict[str, Any]:
    action = args.eval_suites_command
    if action == "draft":
        return _write_draft(api, args, target=target)
    if action == "check":
        assert file_draft is not None
        return _check(api, args, file_draft, target=target)
    identity = _suite_identity(args.suite)
    if identity["source"] == "authored":
        return _run_authored(api, args, identity, target=target)
    return _run_corpus(api, args, identity, target=target)


def _draft_from_file(path: Path) -> dict[str, Any]:
    """The draft a file holds; a saved revision's JSON becomes its draft, as Edit suite makes
    it."""

    document = _json_object(path, "--file")
    if "revision" not in document:
        return document
    try:
        return _draft_of(document)
    except ValueError as exc:
        raise ValueError(f"--file {path} is not a saved suite revision: {exc}") from None


def _draft_of(document: Mapping[str, Any]) -> dict[str, Any]:
    """A saved suite revision's draft, made as Edit suite makes it; `ValueError` when
    ``document`` is not a saved revision."""

    from cayu.evals.suite_authoring import EvalSuiteDraftV3, eval_suite_document_from_json

    saved = eval_suite_document_from_json(json.dumps(document, ensure_ascii=False))
    return EvalSuiteDraftV3.from_document(saved).model_dump(mode="json")


def _write_draft(
    api: CayuApi, args: argparse.Namespace, *, target: Sequence[str]
) -> dict[str, Any]:
    parent = None
    if args.new is None:
        identity = _suite_identity(args.suite)
        if identity["source"] != "authored":
            raise CloudEvalError(
                "invalid_input",
                f"Suite {args.suite} is a suite of a stored corpus, edited as its corpus file; "
                "only an authored suite has a draft.",
            )
        parent = identity["revision"]
        document = api.get(f"/evals/suites/{quote(parent, safe='')}", resource="eval suite")
        try:
            draft = _draft_of(document)
        except ValueError:
            raise CloudEvalError(
                "api_response_invalid", "The server's eval suite is invalid."
            ) from None
    else:
        target_key = args.target_key or run_target_command(api)["default_target_key"]
        draft = {
            "schema_version": 3,
            "id": args.new,
            "target_key": target_key,
            "name": args.name or _NEW_SUITE_NAME,
            "description": None,
            "trial_request": {
                "trials": 1,
                "timeout_seconds": 300,
                "minimum_passed_trials": 1,
                "max_concurrency": 1,
            },
            "cases": [_NEW_CASE],
        }
    text = json.dumps(draft, ensure_ascii=False, indent=2) + "\n"
    try:
        with args.file.open("x", encoding="utf-8") as stream:
            stream.write(text)
    except FileExistsError:
        raise ValueError(
            f"--file {args.file} exists; name a new file, so no edits are overwritten."
        ) from None
    except OSError as exc:
        raise CloudEvalError(
            "output_unwritable",
            f"Could not write {args.file}: {exc.strerror or type(exc).__name__}.",
        ) from None
    check = ["cayu", "eval", "suites", "check", "--file", str(args.file), "--save"]
    if parent is not None:
        check += ["--parent", parent]
    return {
        "file": str(args.file),
        "parent_suite_revision": parent,
        "draft": draft,
        "commands": {"check": shlex.join([*check, *target])},
    }


def _check(
    api: CayuApi, args: argparse.Namespace, draft: dict[str, Any], *, target: Sequence[str]
) -> dict[str, Any]:
    from cayu.evals.read_views import EvalSuiteViewIdentity

    checked = api.post("/evals/suites/preview", {"draft": draft}, resource="eval suite check")
    suite = checked.get("suite") if isinstance(checked, dict) else None
    if (
        not isinstance(suite, dict)
        or not isinstance(suite.get("revision"), str)
        or not isinstance(suite.get("target_key"), str)
        or type(checked.get("ready")) is not bool
    ):
        raise CloudEvalError("api_response_invalid", "The server's eval suite check is invalid.")
    suite_id = EvalSuiteViewIdentity(
        source="authored", target_key=suite["target_key"], revision=suite["revision"]
    ).id
    entry = None
    commands: dict[str, str] = {}
    if args.save:
        if not checked["ready"]:
            raise _diagnosed_refusal(
                "suite_not_ready",
                "The suite can't be saved until these are fixed, so nothing was saved",
                checked.get("diagnostics"),
            )
        entry = _save(api, suite, parent=args.parent)
        commands["check_launch"] = shlex.join(
            ["cayu", "eval", "suites", "run", suite_id, "--preview", *target]
        )
    elif checked["ready"]:
        save = ["cayu", "eval", "suites", "check", "--file", str(args.file), "--save"]
        if args.parent is not None:
            save += ["--parent", args.parent]
        commands["save"] = shlex.join([*save, *target])
    return {
        "suite": suite,
        "suite_id": suite_id,
        "selection": checked.get("full_selection"),
        "ready": checked["ready"],
        "diagnostics": checked.get("diagnostics") or [],
        "entry": entry,
        "commands": commands,
    }


def _save(api: CayuApi, suite: Mapping[str, Any], *, parent: str | None) -> dict[str, Any]:
    try:
        saved = api.post(
            "/evals/suites",
            {
                "expected_suite_revision": suite["revision"],
                "suite": suite,
                "parent_suite_revision": parent,
            },
            resource="eval suite",
        )
    except CayuApiError as exc:
        details = {"suite_revision": suite["revision"], "parent_suite_revision": parent}
        if exc.detail == _SCENARIOS_CHANGED:
            raise CloudEvalError(
                "suite_scenarios_changed",
                "A scenario a case of the suite runs changed after the check, so nothing was "
                "saved. Check the suite again.",
                status_code=exc.status_code,
                details=details,
            ) from None
        if exc.detail != _SAVE_CONFLICT:
            raise
        reasons = "it is already saved under another parent (a saved revision keeps its first)"
        if parent is not None:
            reasons += f", or --parent {parent} is not a saved revision of this suite here"
        raise CloudEvalError(
            "suite_conflict",
            f"The server didn't save revision {suite['revision']}: {reasons}.",
            status_code=exc.status_code,
            details=details,
        ) from None
    entry = saved.get("entry") if isinstance(saved, dict) else None
    confirmed = saved.get("suite") if isinstance(saved, dict) else None
    if (
        not isinstance(entry, dict)
        or not isinstance(confirmed, dict)
        or confirmed.get("revision") != suite["revision"]
    ):
        raise CloudEvalError("api_response_invalid", "The server did not confirm the suite.")
    return entry


def _suite_identity(suite: str) -> dict[str, Any]:
    """The suite an id from `suites list` names: its source, eval target and revision.

    The id encodes them (`EvalSuiteViewIdentity`), so it is decoded as the server decodes it,
    with no request: the suite's view would carry its cases, however large they are.
    """

    from cayu.evals.read_views import EvalSuiteViewIdentity, decode_suite_view

    try:
        identity = decode_suite_view(suite)
    except ValueError:
        raise CloudEvalError(
            "invalid_input", f"{suite} is not a suite id from `cayu eval suites list`."
        ) from None
    if not isinstance(identity, EvalSuiteViewIdentity):
        raise CloudEvalError(
            "invalid_input",
            f"Suite {suite} is a reported series of external eval reports, which runs nowhere.",
        )
    return identity.model_dump(mode="json")


def _run_authored(
    api: CayuApi,
    args: argparse.Namespace,
    identity: Mapping[str, Any],
    *,
    target: Sequence[str],
) -> dict[str, Any]:
    """Admit an authored suite's runs, one per launch group, under one key.

    Without the reviewed exposure and profiles, the launch is checked first and its work
    reviewed. With them (a printed run command) the admission is sent as given, so the server
    answers with the runs the key admitted, also after a deploy changed the current profiles.
    """

    if args.max_concurrency is not None:
        raise CloudEvalError(
            "invalid_input",
            f"Suite {args.suite} is an authored suite, whose trial policy sets its "
            "concurrency; leave out --max-concurrency.",
        )
    if args.expected_execution_profile_revision is not None:
        raise CloudEvalError(
            "invalid_input",
            "--expected-execution-profile-revision pins a corpus suite's run; an authored "
            "suite's run command pins --expected-execution-profile per launch group.",
        )
    path = f"/evals/suites/{quote(identity['revision'], safe='')}/runs"
    selection: dict[str, Any] = {}
    if args.case_ids is not None:
        selection["case_ids"] = list(args.case_ids)
    budget = _cost_budget(args)
    if budget is not None:
        selection["cost_budget"] = budget
    preview = None
    if args.expected_exposure_revision is None:
        preview = _launch_preview(api, f"{path}/preview", selection)
        exposure = preview["exposure"]["revision"]
        profiles = [
            {"case_ids": item["case_ids"], _PROFILE: item[_PROFILE]} for item in preview["launches"]
        ]
    else:
        exposure, profiles = args.expected_exposure_revision, args.expected_execution_profile
    operation_id = args.operation_id or f"cayu-cli-eval-suite-{secrets.token_hex(16)}"
    pins = ["--operation-id", operation_id, "--expected-exposure-revision", exposure]
    for profile in profiles:
        pins += [
            "--expected-execution-profile",
            f"{','.join(profile['case_ids'])}={profile[_PROFILE]}",
        ]
    commands = _launch_commands(args, identity, pins, target=target)
    if args.preview:
        return _launched(args, identity, preview=preview, commands={"run": commands["continue"]})
    body = {
        **selection,
        "expected_execution_profiles": profiles,
        "expected_exposure_revision": exposure,
    }
    try:
        launched = api.mutate(path, body, resource="eval suite run", operation_id=operation_id)
    except CayuApiError as exc:
        raise _launch_refusal(exc, operation_id=operation_id, commands=commands) from None
    groups = launched.get("runs") if isinstance(launched, dict) else None
    if not isinstance(groups, list) or not groups or not all(isinstance(g, dict) for g in groups):
        raise CloudEvalError("api_response_invalid", "The server did not confirm the suite run.")
    runs = [
        {
            "kind": group.get("kind"),
            "case_ids": group.get("case_ids"),
            "run": _run_summary(group.get("run")),
        }
        for group in groups
    ]
    return _launched(
        args,
        identity,
        preview=preview,
        operation_id=operation_id,
        runs=runs,
        commands=commands,
    )


def _launch_preview(api: CayuApi, path: str, selection: Mapping[str, Any]) -> dict[str, Any]:
    preview = api.post(path, selection, resource="eval suite launch readiness")
    launches = preview.get("launches") if isinstance(preview, dict) else None
    if not isinstance(launches, list) or type(preview.get("ready")) is not bool:
        raise CloudEvalError(
            "api_response_invalid", "The server's eval suite launch readiness is invalid."
        )
    if not preview["ready"]:
        raise _diagnosed_refusal(
            "suite_launch_not_ready",
            "The server can't run this selection now, so nothing was run",
            preview.get("diagnostics"),
        )
    exposure = preview.get("exposure")
    if (
        not isinstance(exposure, dict)
        or not isinstance(exposure.get("revision"), str)
        or not all(
            isinstance(item, dict)
            and isinstance(item.get("case_ids"), list)
            and isinstance(item.get(_PROFILE), str)
            for item in launches
        )
    ):
        raise CloudEvalError(
            "api_response_invalid", "The server's eval suite launch readiness is invalid."
        )
    return preview


def _run_corpus(
    api: CayuApi,
    args: argparse.Namespace,
    identity: Mapping[str, Any],
    *,
    target: Sequence[str],
) -> dict[str, Any]:
    """Admit one run of a corpus suite, as the suite page's Run suite does.

    A printed run command repeats the admission with the execution profile it was reviewed
    under, so the server answers with the run the key admitted.
    """

    from cayu.evals.corpus import eval_corpus_from_json

    if args.case_ids is not None:
        raise CloudEvalError(
            "invalid_input",
            f"Suite {args.suite} is a corpus suite, which runs whole; --case selects cases of "
            "an authored suite.",
        )
    if args.expected_exposure_revision is not None:
        raise CloudEvalError(
            "invalid_input",
            "--expected-exposure-revision and --expected-execution-profile pin an authored "
            "suite's run; a corpus suite's run command pins "
            "--expected-execution-profile-revision.",
        )
    suite_id = identity.get("suite_id")
    if not isinstance(suite_id, str):
        raise CloudEvalError("api_response_invalid", "The server's eval suite is invalid.")
    corpus_revision, target_key = identity["revision"], identity["target_key"]
    continuing = args.expected_execution_profile_revision is not None
    entry = _published_target(api, target_key, require_ready=not continuing)
    current = (entry.get("execution_profile") or {}).get("revision")
    profile = args.expected_execution_profile_revision or current
    if not isinstance(profile, str):
        raise CloudEvalError(
            "api_response_invalid", "The server's eval target names no execution profile."
        )
    max_concurrency = args.max_concurrency or _DEFAULT_CORPUS_CONCURRENCY
    budget = _cost_budget(args)
    cost_cap = (
        None
        if budget is None
        else {"max_estimated_cost": args.max_cost, "currency": budget["currency"]}
    )
    bound = None
    if not continuing:
        document = api.get(
            f"/evals/corpora/{quote(corpus_revision, safe='')}", resource="eval corpus"
        )
        try:
            corpus = eval_corpus_from_json(json.dumps(document, ensure_ascii=False))
        except (TypeError, ValueError):
            raise CloudEvalError(
                "api_response_invalid", "The server's eval corpus is invalid."
            ) from None
        suite = _corpus_suite(corpus, suite_id)
        _check_bounds(entry, suite, max_concurrency=max_concurrency, cost_cap=cost_cap)
        bound = _bound(entry, corpus, suite, max_concurrency=max_concurrency, cost_cap=cost_cap)
    operation_id = args.operation_id or f"cayu-cli-eval-suite-{secrets.token_hex(16)}"
    pins = ["--max-concurrency", str(max_concurrency)]
    pins += ["--operation-id", operation_id, "--expected-execution-profile-revision", profile]
    commands = _launch_commands(args, identity, pins, target=target)
    if args.preview:
        return _launched(args, identity, bound=bound, commands={"run": commands["continue"]})
    run = _admit(
        api,
        corpus_revision=corpus_revision,
        suite_id=suite_id,
        profile_revision=profile,
        profile_changed=profile != current,
        max_concurrency=max_concurrency,
        cost_cap=cost_cap,
        operation_id=operation_id,
        commands=commands,
        import_corpus=None,
        profile_option="--expected-execution-profile-revision",
    )
    return _launched(
        args,
        identity,
        bound=bound,
        operation_id=operation_id,
        runs=[{"kind": "corpus", "case_ids": None, "run": run}],
        commands=commands,
    )


def _launched(
    args: argparse.Namespace,
    identity: Mapping[str, Any],
    *,
    commands: Mapping[str, str],
    preview: Mapping[str, Any] | None = None,
    bound: Mapping[str, Any] | None = None,
    operation_id: str | None = None,
    runs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "suite": args.suite,
        "source": identity["source"],
        "suite_revision": identity["revision"],
        "preview": preview,
        "bound": bound,
        "operation_id": operation_id,
        "runs": list(runs),
        "commands": dict(commands),
    }


def _launch_commands(
    args: argparse.Namespace,
    identity: Mapping[str, Any],
    pins: Sequence[str],
    *,
    target: Sequence[str],
) -> dict[str, str]:
    """The `suites run` that sends this admission again, and the `runs list` that shows the
    runs a launch admitted."""

    command = ["cayu", "eval", "suites", "run", args.suite]
    for case_id in args.case_ids or ():
        command += ["--case", case_id]
    if args.max_cost is not None:
        command += ["--max-cost", str(args.max_cost)]
        command += ["--currency", args.currency or _DEFAULT_CURRENCY]
    listing = ["cayu", "eval", "runs", "list", "--target-key", identity["target_key"]]
    return {
        "continue": shlex.join([*command, *pins, *target]),
        "list": shlex.join([*listing, *target]),
    }


def _launch_refusal(
    exc: CayuApiError, *, operation_id: str, commands: Mapping[str, str]
) -> Exception:
    if exc.category == "outcome_unknown":
        return CloudEvalError(
            "outcome_unknown",
            "The server did not confirm the suite's runs; it may have admitted some or all of "
            f"them. `{commands['continue']}` admits nothing it already admitted.",
            status_code=exc.status_code,
            details={"operation_id": operation_id, "commands": dict(commands)},
        )
    if exc.detail == _IDEMPOTENCY_CONFLICT:
        return CloudEvalError(
            "operation_id_conflict",
            f"Operation ID {operation_id} already admitted a different suite launch, so this "
            f"one admitted nothing. `{commands['list']}` lists the runs it admitted.",
            status_code=exc.status_code,
            details={"operation_id": operation_id, "commands": {"list": commands["list"]}},
        )
    if exc.detail in _LAUNCH_CHANGED:
        return CloudEvalError(
            "suite_launch_changed",
            "The work or execution profiles the launch check reviewed changed since, so "
            "nothing was admitted. Check the launch again.",
            status_code=exc.status_code,
            details={"operation_id": operation_id},
        )
    if exc.detail == _LAUNCH_NOT_READY:
        return CloudEvalError(
            "suite_launch_not_ready",
            "The server can't run this selection now, so nothing was admitted. Check the "
            "launch again for the reasons.",
            status_code=exc.status_code,
            details={"operation_id": operation_id},
        )
    return exc


def _cost_budget(args: argparse.Namespace) -> dict[str, str] | None:
    if args.max_cost is None:
        return None
    return {
        "max_estimated_cost": str(args.max_cost),
        "currency": args.currency or _DEFAULT_CURRENCY,
    }


def _profile_expectation(value: str) -> dict[str, Any]:
    cases, separator, revision = value.rpartition("=")
    case_ids = cases.split(",") if cases else []
    if not separator or not case_ids or not all(case_ids) or not _EVAL_REVISION.fullmatch(revision):
        raise argparse.ArgumentTypeError("must be CASE_ID[,CASE_ID...]=sha256:REVISION")
    return {"case_ids": case_ids, _PROFILE: revision}


def render_targets(answer: Mapping[str, Any]) -> str:
    result = answer["result"]
    rows = []
    reasons = []
    for item in result["items"]:
        ready = item.get("execution_profile_ready") is True
        judges = ", ".join(str(profile.get("key")) for profile in item.get("judge_profiles") or ())
        rows.append(
            (
                _cell(item.get("target_key"))
                + (" (default)" if item.get("target_key") == result["default_target_key"] else ""),
                _cell(item.get("label")),
                "ready" if ready else "not ready",
                _cell(judges or None),
            )
        )
        for diagnostic in item.get("execution_profile_diagnostics") or ():
            reasons.append(
                f"{_text(item.get('target_key'))}: {_text(diagnostic.get('message'))} "
                f"{_text(diagnostic.get('remediation'))}"
            )
    return "\n".join([*_table(("TARGET", "LABEL", "RUNS", "JUDGES"), rows), *reasons])


def render_draft(answer: Mapping[str, Any]) -> str:
    result = answer["result"]
    origin = (
        f"revision {_text(result['parent_suite_revision'])}"
        if result["parent_suite_revision"] is not None
        else "a new suite"
    )
    return (
        f"Wrote a draft of {origin} to {_text(result['file'])}.\n"
        f"Check and save it: {_text(result['commands']['check'])}"
    )


def render_suite_check(answer: Mapping[str, Any]) -> str:
    result = answer["result"]
    suite = result["suite"]
    name = (suite.get("suite") or {}).get("name")
    lines = [f"Suite {_text(name)} ({_text(suite['revision'])})"]
    if result["ready"]:
        lines[0] += " is ready to save."
    else:
        lines[0] += " is not ready to save:"
        for item in result["diagnostics"]:
            lines.append(f"  {_text(item.get('case_id') or '-')}: {_text(item.get('message'))}")
    if result["entry"] is not None:
        lines.append(f"Saved. Suite: {_text(result['suite_id'])}")
    for name, label in (("save", "Save it"), ("check_launch", "Check its launch")):
        if name in result["commands"]:
            lines.append(f"{label}: {_text(result['commands'][name])}")
    return "\n".join(lines)


def render_suite_run(answer: Mapping[str, Any]) -> str:
    result = answer["result"]
    lines = []
    exposure = (result["preview"] or {}).get("exposure")
    if isinstance(exposure, dict):
        lines.append(
            f"At most {_text(exposure.get('candidate_trials'))} candidate trials and "
            f"{_text(exposure.get('judge_evaluations'))} judge evaluations, "
            f"{_text(exposure.get('max_concurrency'))} at a time."
        )
    if result["bound"] is not None:
        bound = result["bound"]
        lines.append(
            f"At most {_text(bound['total_trials'])} trials, {_text(bound['max_concurrency'])} "
            "at a time."
        )
    for item in result["runs"]:
        run = item["run"]
        cases = ", ".join(item["case_ids"] or ()) or "every case"
        lines.append(f"Run {_text(run['id'])}: {_text(run['status'])} ({_text(cases)})")
    if "run" in result["commands"]:
        lines.append(f"Run it: {_text(result['commands']['run'])}")
    elif result["runs"]:
        lines.append(f"Follow them: {_text(result['commands']['list'])}")
    return "\n".join(lines)
