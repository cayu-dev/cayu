"""Stable command entrypoint for scaffolded Cayu projects."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

from cayu.applications import CayuApp
from cayu.configuration import DEFAULT_MAX_STEPS, MAX_STEPS
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.diagnostics import capture_provider_errors
from cayu.sessions.outcomes import run_to_completion
from cayu.sessions.requests import RunRequest


def run_project_entrypoint(
    app_factory: Callable[[], CayuApp],
    argv: list[str] | None = None,
    *,
    validate_run: Callable[[CayuApp, str], None] | None = None,
) -> int:
    """Run one registered agent from a generated project's command line."""

    args = _parser().parse_args(argv)
    if not args.message.strip():
        print("setup error: --message must not be blank", file=sys.stderr)
        return 2
    if args.max_steps is not None and args.max_steps < 1:
        print("setup error: --max-steps must be at least 1", file=sys.stderr)
        return 2
    if args.max_steps is not None and args.max_steps > MAX_STEPS:
        print(f"setup error: --max-steps must be at most {MAX_STEPS}", file=sys.stderr)
        return 2

    try:
        app = app_factory()
        if not isinstance(app, CayuApp):
            raise TypeError("the project factory must return a CayuApp")
        agent_name = _select_agent(app, args.agent)
        if validate_run is not None:
            validate_run(app, agent_name)
    except Exception as error:
        print(f"setup error: {error}", file=sys.stderr)
        return 2

    run_request = RunRequest(
        agent_name=agent_name,
        messages=[Message.text("user", args.message)],
    )
    if args.max_steps is not None:
        run_request = run_request.model_copy(update={"max_steps": args.max_steps})
    provider_errors: list[dict[str, Any]] = []
    # Opt-in only: provider error bodies can echo request content, so they are
    # printed to the developer's own terminal, never stored with the session.
    capture = (
        capture_provider_errors(provider_errors.append, redactor=app._secret_redactor)
        if args.show_provider_errors
        else nullcontext()
    )
    with capture:

        async def execute():
            try:
                # aclose() owns stopping it, within the shutdown deadline.
                await app.start_model_policy()
                return await run_to_completion(app, run_request)
            finally:
                shutdown = await app.aclose()
                if not shutdown.settled:
                    print(f"warning: {shutdown.summary()}", file=sys.stderr, flush=True)

        outcome = asyncio.run(execute())
    if outcome.ok:
        print(outcome.final_text)
        return 0

    detail = outcome.error or outcome.status.value
    print(f"run failed: {detail} (session {outcome.session_id})", file=sys.stderr)
    for record in provider_errors:
        print(f"provider error: {_describe_provider_error(record)}", file=sys.stderr)
    if not args.show_provider_errors and any(
        event.type == EventType.MODEL_ERROR for event in outcome.events
    ):
        print(
            "Rerun with --show-provider-errors for the full provider error record.",
            file=sys.stderr,
        )
    return 1


def _describe_provider_error(record: dict[str, Any]) -> str:
    error = record.get("error", {})
    status = record.get("http_status_code")
    parts = [f"HTTP {status}"] if status is not None else []
    parts.extend(f"{name}={error[name]}" for name in ("type", "code", "param") if name in error)
    parts.append(error.get("message", "no message in the provider response"))
    return "; ".join(parts)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a registered Cayu agent.")
    parser.add_argument(
        "--agent",
        help="Registered agent name; optional when the project has only one.",
    )
    parser.add_argument("--message", required=True, help="User message text.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=f"Model-step ceiling; defaults to the app setting ({DEFAULT_MAX_STEPS} normally).",
    )
    parser.add_argument(
        "--show-provider-errors",
        action="store_true",
        help=(
            "Print provider error details, such as why a model was rejected, to stderr. "
            "They can echo request content, so use this for local debugging."
        ),
    )
    return parser


def _select_agent(app: CayuApp, requested: str | None) -> str:
    available = tuple(sorted(app.list_agents()))
    rendered = ", ".join(available) or "none"
    if requested is not None:
        if requested not in available:
            raise ValueError(f"unknown agent {requested!r}; available agents: {rendered}")
        return requested
    if len(available) == 1:
        return available[0]
    if not available:
        raise ValueError("the project has no registered agents")
    raise ValueError(f"multiple agents are registered; pass --agent NAME (available: {rendered})")
