"""Stable command entrypoint for scaffolded Cayu projects."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable

from cayu.core import Message
from cayu.runtime import CayuApp, RunRequest, run_to_completion

_MAX_STEPS = 256


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
    if args.max_steps < 1:
        print("setup error: --max-steps must be at least 1", file=sys.stderr)
        return 2
    if args.max_steps > _MAX_STEPS:
        print(f"setup error: --max-steps must be at most {_MAX_STEPS}", file=sys.stderr)
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

    outcome = asyncio.run(
        run_to_completion(
            app,
            RunRequest(
                agent_name=agent_name,
                messages=[Message.text("user", args.message)],
                max_steps=args.max_steps,
            ),
        )
    )
    if outcome.ok:
        print(outcome.final_text)
        return 0

    detail = outcome.error or outcome.status.value
    print(f"run failed: {detail} (session {outcome.session_id})", file=sys.stderr)
    return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a registered Cayu agent.")
    parser.add_argument(
        "--agent",
        help="Registered agent name; optional when the project has only one.",
    )
    parser.add_argument("--message", required=True, help="User message text.")
    parser.add_argument("--max-steps", type=int, default=12)
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
