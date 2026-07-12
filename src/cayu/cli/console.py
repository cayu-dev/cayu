from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys
import tomllib
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cayu
from cayu.cli._targets import TargetResolutionError, load_target
from cayu.runtime.app import CayuApp


@dataclass(frozen=True)
class ConsoleProject:
    root: Path
    target: str


class ConsoleError(Exception):
    """An actionable console configuration or contract error."""


def add_console_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "console",
        help="Open an IPython console with a booted Cayu application.",
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Override project discovery with a module:factory target.",
    )


def run_console(args: argparse.Namespace) -> int:
    try:
        project = _resolve_project(args.target)
        start_ipython = _load_ipython()
        with _project_context(project.root):
            app = _build_app(project.target)
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


def _resolve_project(explicit_target: str | None) -> ConsoleProject:
    cwd = Path.cwd().resolve()
    if explicit_target is not None:
        return ConsoleProject(root=cwd, target=explicit_target)

    for directory in (cwd, *cwd.parents):
        pyproject = directory / "pyproject.toml"
        if not pyproject.is_file():
            continue
        try:
            config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConsoleError(f"Could not read {pyproject}: {exc}") from exc
        tool_config = config.get("tool", {})
        cayu_config = tool_config.get("cayu", {}) if isinstance(tool_config, dict) else {}
        target = cayu_config.get("factory") if isinstance(cayu_config, dict) else None
        if target is None:
            continue
        if not isinstance(target, str) or not target.strip():
            raise ConsoleError(f"{pyproject}: [tool.cayu].factory must be a non-empty string.")
        return ConsoleProject(root=directory, target=target)

    raise ConsoleError(
        'No Cayu project found. Add [tool.cayu] factory = "module:build_app" '
        "to pyproject.toml, or pass cayu console module:build_app."
    )


@contextmanager
def _project_context(root: Path) -> Iterator[None]:
    original_cwd = Path.cwd()
    root_text = os.fspath(root)
    sys.path.insert(0, root_text)
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(original_cwd)
        with suppress(ValueError):
            sys.path.remove(root_text)


def _build_app(target: str) -> CayuApp:
    try:
        factory = load_target(
            target,
            label="Console factory target",
            normalize_errors=True,
        )
    except TargetResolutionError as exc:
        raise ConsoleError(str(exc)) from exc
    if not callable(factory):
        raise ConsoleError("Console factory target must be a callable, not an application object.")
    if inspect.iscoroutinefunction(factory):
        raise ConsoleError(
            "Console factory must be synchronous; async factories are not supported."
        )

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError) as exc:
        raise ConsoleError("Console factory must expose a zero-argument signature.") from exc
    required = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if required:
        names = ", ".join(parameter.name for parameter in required)
        raise ConsoleError(f"Console factory must not require arguments (required: {names}).")

    app = factory()
    if inspect.iscoroutine(app):
        # Discarding a rejected native coroutine without closing it emits a RuntimeWarning.
        app.close()
    if inspect.isawaitable(app):
        raise ConsoleError(
            "Console factory returned an awaitable; async factories are not supported."
        )
    if not isinstance(app, CayuApp):
        raise ConsoleError("Console factory must return a CayuApp.")
    return app


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
