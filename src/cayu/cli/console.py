from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable
from typing import Any

import cayu
from cayu.cli.project import (
    CayuProject as ConsoleProject,
)
from cayu.cli.project import (
    ProjectError as ConsoleError,
)
from cayu.cli.project import (
    build_project_app,
    project_context,
    resolve_project,
)
from cayu.runtime.app import CayuApp


def add_console_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "console",
        help="Open an IPython console with a booted Cayu application.",
        description=(
            "Open an IPython console with a live, writable Cayu application. "
            "Use `cayu inspect` first for read-only structural inspection."
        ),
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Override project discovery with a module:factory target.",
    )


def run_console(args: argparse.Namespace) -> int:
    try:
        project = resolve_project(args.target, command="cayu console")
        start_ipython = _load_ipython()
        with project_context(project.root):
            app = build_project_app(project.target, command="Console")
            namespace = {
                "cayu": cayu,
                "app": app,
                "sessions": app.session_store,
                "tasks": app.task_store,
                "knowledge": app.knowledge_store,
            }
            print(_banner(app=app, project=project))
            try:
                start_ipython(argv=[], user_ns=namespace)
            except EOFError:
                return 0
            except KeyboardInterrupt:
                return 130
    except ConsoleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _load_ipython() -> Callable[..., Any]:
    try:
        ipython = importlib.import_module("IPython")
    except ImportError as exc:
        raise ConsoleError(
            'Cayu console requires IPython. Install it with: pip install "cayu[console]"'
        ) from exc
    start_ipython = getattr(ipython, "start_ipython", None)
    if not callable(start_ipython):
        raise ConsoleError("Installed IPython does not expose start_ipython().")
    return start_ipython


def _banner(*, app: CayuApp, project: ConsoleProject) -> str:
    from cayu.cli import _version

    agents = ", ".join(app.list_agents()) or "none"
    providers = ", ".join(app.list_providers()) or "none"
    environments = ", ".join(app.list_environments()) or "none"
    task_store = type(app.task_store).__name__ if app.task_store is not None else "none"
    knowledge_store = (
        type(app.knowledge_store).__name__ if app.knowledge_store is not None else "none"
    )
    return "\n".join(
        (
            f"Cayu {_version()} console",
            f"Project: {project.root}",
            f"Factory: {project.target}",
            f"Agents: {agents}",
            f"Providers: {providers}",
            f"Environments: {environments}",
            f"Session store: {type(app.session_store).__name__}",
            f"Task store: {task_store}",
            f"Knowledge store: {knowledge_store}",
            "Warning: this is a live, writable application console.",
        )
    )
