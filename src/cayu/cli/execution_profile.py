"""`cayu execution-profile`: predict profile admission for a release before deploying it.

Both subcommands build the project's application from its factory target, as
`cayu serve`, `cayu worker` and `cayu inspect` do, resolve candidate profiles
through the ordinary read-only initial-run preflight, and close the application.
Candidate inspection never loads or writes a session, admits a run, or
dispatches provider, tool, hook or environment-factory work. Whatever the
factory itself does while constructing the application still happens.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cayu._version import package_version
from cayu.build_provenance import current_runtime_build_provenance
from cayu.cli._output import add_output_options, output_destination
from cayu.cli.project import (
    ProjectError,
    build_project_app,
    close_project_app,
    project_context,
    resolve_project,
)
from cayu.execution_profiles import (
    ExecutionProfileAdmissionBoundary,
    ExecutionProfileIdentity,
    predict_execution_profile_admission,
)
from cayu.sessions._model_failover import ModelTarget

SCHEMA_VERSION = "1"

#: Exit codes. 1 is a general error (the project could not be built or the
#: input is invalid) and 2 an argparse usage error.
EXIT_OK = 0
#: ``predict``: at least one stored profile is not known to be admitted.
EXIT_MAY_NOT_RESUME = 3
#: ``candidates``: at least one requested candidate could not be resolved.
EXIT_CANDIDATE_UNAVAILABLE = 4

MAX_PREDICTION_ENTRIES = 10_000
_MAX_ERROR_MESSAGE_CHARS = 1_000


class _CandidateFailure:
    def __init__(self, message: str) -> None:
        self.message = message


# Agent, environment (None: no environment), provider and model (None: the
# agent's default target), and causal budget id (None: a fresh session's).
_CandidateKey = tuple[str, str | None, str | None, str | None, str | None]


class _PredictionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    id: str = Field(min_length=1, max_length=512)
    boundary: ExecutionProfileAdmissionBoundary
    agent_name: str = Field(min_length=1)
    # The session's recorded values, used exactly: null means no environment.
    environment_name: str | None
    provider_name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    causal_budget_id: str = Field(min_length=1)
    expected_profile: ExecutionProfileIdentity


class _PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    sessions: list[_PredictionEntry] = Field(max_length=MAX_PREDICTION_ENTRIES)


def add_execution_profile_parser(subparsers: Any) -> None:
    """Register the ``execution-profile`` command group."""

    group = subparsers.add_parser(
        "execution-profile",
        help="Predict whether stored sessions resume on this release.",
        description=(
            "Resolve this release's candidate execution profiles and predict whether "
            "stored sessions resume on it. Read-only: nothing is started or admitted."
        ),
    )
    commands = group.add_subparsers(dest="execution_profile_command", required=True)

    candidates = commands.add_parser(
        "candidates",
        help="Print each agent's candidate execution profile.",
        description=(
            "Print the execution profile a fresh run of each agent would freeze on this "
            "release. Exits 0 when every candidate resolved and 4 when one could not."
        ),
    )
    _add_target_argument(candidates)
    candidates.add_argument(
        "--agent",
        action="append",
        default=[],
        help="Inspect this agent (repeatable; default: every registered agent).",
    )
    environment = candidates.add_mutually_exclusive_group()
    environment.add_argument(
        "--environment",
        help="Resolve candidates for this environment (default: the application default).",
    )
    environment.add_argument(
        "--no-environment",
        action="store_true",
        help="Resolve candidates for a session that has no environment.",
    )
    candidates.add_argument("--provider", help="Resolve candidates for this provider.")
    candidates.add_argument("--model", help="Resolve candidates for this model.")
    candidates.add_argument(
        "--causal-budget-id",
        help="Resolve candidates for a session with this causal budget id.",
    )
    add_output_options(candidates)

    predict = commands.add_parser(
        "predict",
        help="Predict profile admission for stored session profiles.",
        description=(
            "Compare stored session profiles with this release's candidates for the same "
            "agent, environment, provider, model and causal budget id. Exits 0 when every "
            "entry is predicted to resume and 3 when at least one may not."
        ),
    )
    _add_target_argument(predict)
    predict.add_argument(
        "--input",
        required=True,
        metavar="FILE",
        help='JSON file with {"sessions": [...]}, or - for stdin.',
    )
    add_output_options(predict)


def run_execution_profile(args: argparse.Namespace) -> int:
    """Dispatch a parsed ``execution-profile`` invocation; return its exit code."""

    try:
        with output_destination(args.output):
            if args.execution_profile_command == "candidates":
                return _run_candidates(args)
            return _run_predict(args)
    except OSError as exc:
        print(f"error: could not write output: {exc}", file=sys.stderr)
        return 1


def _add_target_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "target",
        nargs="?",
        help="Override project discovery with a module:factory target.",
    )


def _run_candidates(args: argparse.Namespace) -> int:
    if (args.provider is None) != (args.model is None):
        return _fail(args, "INVALID_ARGUMENTS", "--provider and --model must be given together.")
    try:
        project = resolve_project(args.target, command="cayu execution-profile candidates")
        with project_context(project.root):
            app = build_project_app(project.target, command="Execution-profile")
            agents = tuple(args.agent) or app.list_agents()
            environment = (
                None
                if args.no_environment
                else app.default_environment_name
                if args.environment is None
                else args.environment
            )
            candidates, errors = asyncio.run(
                _resolve_candidates(
                    app,
                    [
                        (agent, environment, args.provider, args.model, args.causal_budget_id)
                        for agent in agents
                    ],
                )
            )
    except Exception as exc:
        return _fail(args, "PROJECT_BOOT_FAILED", _project_error_message(exc))
    payload = {
        **_header(app),
        "candidates": [
            item.model_dump(mode="json")
            for item in candidates.values()
            if not isinstance(item, _CandidateFailure)
        ],
        "errors": errors,
    }
    if args.output_format == "json":
        print(json.dumps(payload, sort_keys=True))
    else:
        _print_candidates(payload)
    return EXIT_CANDIDATE_UNAVAILABLE if errors else EXIT_OK


def _run_predict(args: argparse.Namespace) -> int:
    try:
        raw = sys.stdin.read() if args.input == "-" else Path(args.input).read_text("utf-8")
        entries = _PredictionInput.model_validate_json(raw).sessions
    except (OSError, ValueError) as exc:
        message = (
            "Prediction input does not match the documented shape."
            if isinstance(exc, ValidationError)
            else f"Could not read prediction input: {exc}"
        )
        return _fail(args, "INVALID_INPUT", message)
    try:
        project = resolve_project(args.target, command="cayu execution-profile predict")
        with project_context(project.root):
            app = build_project_app(project.target, command="Execution-profile")
            requests: list[_CandidateKey] = [
                (
                    entry.agent_name,
                    entry.environment_name,
                    entry.provider_name,
                    entry.model,
                    entry.causal_budget_id,
                )
                for entry in entries
            ]
            candidates, _errors = asyncio.run(_resolve_candidates(app, requests))
    except Exception as exc:
        return _fail(args, "PROJECT_BOOT_FAILED", _project_error_message(exc))

    policy_configured = app.execution_profile_policy_identity is not None
    predictions = []
    summary = {"total": len(entries), "admits": 0, "does_not_admit": 0, "undetermined": 0}
    for entry, key in zip(entries, requests, strict=True):
        candidate = candidates.get(key)
        item: dict[str, Any] = {
            "id": entry.id,
            "boundary": entry.boundary.value,
            "agent_name": entry.agent_name,
            "environment_name": entry.environment_name,
            "provider_name": entry.provider_name,
            "model": entry.model,
            "causal_budget_id": entry.causal_budget_id,
            "candidate_fingerprint": None,
            "prediction": None,
            "error": None,
        }
        if isinstance(candidate, _CandidateFailure):
            item["error"] = {"code": "CANDIDATE_UNAVAILABLE", "message": candidate.message}
            summary["undetermined"] += 1
        else:
            assert candidate is not None
            prediction = predict_execution_profile_admission(
                entry.expected_profile,
                candidate.execution_profile,
                boundary=entry.boundary,
                application_policy_configured=policy_configured,
            )
            item["candidate_fingerprint"] = candidate.execution_profile.fingerprint
            item["prediction"] = prediction.model_dump(mode="json")
            if prediction.admits is True:
                summary["admits"] += 1
            elif prediction.admits is False:
                summary["does_not_admit"] += 1
            else:
                summary["undetermined"] += 1
        predictions.append(item)
    payload = {**_header(app), "predictions": predictions, "summary": summary}
    if args.output_format == "json":
        print(json.dumps(payload, sort_keys=True))
    else:
        _print_predictions(payload)
    return EXIT_OK if summary["admits"] == summary["total"] else EXIT_MAY_NOT_RESUME


async def _resolve_candidates(
    app: Any,
    requests: list[_CandidateKey],
) -> tuple[dict[_CandidateKey, Any], list[dict[str, Any]]]:
    """Resolve each distinct request once, then close the application.

    Candidate failures are reported, not raised. A failure to close is reported
    on stderr so it never replaces resolved candidates.
    """

    try:
        return await _resolve_each_candidate(app, requests)
    finally:
        try:
            await close_project_app(app)
        except Exception as exc:
            message = app._secret_redactor.redact_text(f"{type(exc).__name__}: {exc}")
            print(
                f"warning: closing the application failed: {message[:_MAX_ERROR_MESSAGE_CHARS]}",
                file=sys.stderr,
            )


async def _resolve_each_candidate(
    app: Any,
    requests: list[_CandidateKey],
) -> tuple[dict[_CandidateKey, Any], list[dict[str, Any]]]:
    resolved: dict[_CandidateKey, Any] = {}
    errors: list[dict[str, Any]] = []
    for key in requests:
        if key in resolved:
            continue
        agent_name, environment_name, provider_name, model, causal_budget_id = key
        target = (
            None
            if provider_name is None or model is None
            else ModelTarget(provider_name=provider_name, model=model)
        )
        try:
            resolved[key] = await app.inspect_candidate_execution_profile(
                agent_name,
                environment_name=environment_name,
                target=target,
                causal_budget_id=causal_budget_id,
            )
        except Exception as exc:
            message = app._secret_redactor.redact_text(f"{type(exc).__name__}: {exc}")
            message = message[:_MAX_ERROR_MESSAGE_CHARS]
            resolved[key] = _CandidateFailure(message)
            errors.append(
                {
                    "agent_name": agent_name,
                    "environment_name": environment_name,
                    "provider_name": provider_name,
                    "model": model,
                    "causal_budget_id": causal_budget_id,
                    "code": "CANDIDATE_UNAVAILABLE",
                    "message": message,
                }
            )
    return resolved, errors


def _header(app: Any) -> dict[str, Any]:
    identity = app.execution_profile_policy_identity
    return {
        "schema_version": SCHEMA_VERSION,
        "cayu_version": package_version(),
        "runtime_build_provenance": current_runtime_build_provenance().model_dump(mode="json"),
        "execution_profile_policy": {"configured": identity is not None, "identity": identity},
    }


def _fail(args: argparse.Namespace, code: str, message: str) -> int:
    if args.output_format == "json":
        print(
            json.dumps(
                {"schema_version": SCHEMA_VERSION, "error": {"code": code, "message": message}},
                sort_keys=True,
            )
        )
    else:
        print(f"error: {message}", file=sys.stderr)
    return 1


def _project_error_message(exc: Exception) -> str:
    if isinstance(exc, ProjectError):
        return str(exc)
    return f"Application factory failed ({type(exc).__name__}): {exc}"


def _print_candidates(payload: dict[str, Any]) -> None:
    policy = payload["execution_profile_policy"]
    print(f"cayu {payload['cayu_version']}; execution-profile policy: {policy['identity']}")
    for item in payload["candidates"]:
        environment = item["environment_name"] or "-"
        print(
            f"{item['agent_name']}  {item['provider_name']}/{item['model']}  "
            f"environment={environment}  {item['execution_profile']['fingerprint']}"
        )
    for error in payload["errors"]:
        print(f"{error['agent_name']}  unavailable: {error['message']}")


def _print_predictions(payload: dict[str, Any]) -> None:
    for item in payload["predictions"]:
        prediction = item["prediction"]
        if prediction is None:
            print(f"{item['id']}  unknown: {item['error']['message']}")
            continue
        changed = ", ".join(prediction["changed_component_classes"]) or "nothing"
        print(
            f"{item['id']}  {item['boundary']}  {prediction['outcome']}  "
            f"admits={prediction['admits']}  changed: {changed}"
        )
    summary = payload["summary"]
    print(
        f"{summary['admits']} of {summary['total']} admitted; "
        f"{summary['does_not_admit']} not admitted; {summary['undetermined']} undetermined"
    )
