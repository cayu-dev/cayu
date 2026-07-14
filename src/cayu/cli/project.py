from __future__ import annotations

import inspect
import os
import sys
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from cayu.cli._targets import TargetResolutionError, load_target
from cayu.runtime.app import CayuApp


@dataclass(frozen=True)
class CayuProject:
    root: Path
    target: str


class ProjectError(Exception):
    """An actionable project discovery or application-factory contract error."""


def resolve_project(
    explicit_target: str | None = None,
    *,
    command: str = "cayu",
) -> CayuProject:
    cwd = Path.cwd().resolve()
    if explicit_target is not None:
        return CayuProject(root=cwd, target=explicit_target)

    for directory in (cwd, *cwd.parents):
        pyproject = directory / "pyproject.toml"
        if not pyproject.is_file():
            continue
        try:
            config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ProjectError(f"Could not read {pyproject}: {exc}") from exc
        tool_config = config.get("tool", {})
        cayu_config = tool_config.get("cayu", {}) if isinstance(tool_config, dict) else {}
        target = cayu_config.get("factory") if isinstance(cayu_config, dict) else None
        if target is None:
            continue
        if not isinstance(target, str) or not target.strip():
            raise ProjectError(f"{pyproject}: [tool.cayu].factory must be a non-empty string.")
        return CayuProject(root=directory, target=target)

    raise ProjectError(
        'No Cayu project found. Add [tool.cayu] factory = "module:build_app" '
        f"to pyproject.toml, or pass {command} module:build_app."
    )


@contextmanager
def project_context(root: Path) -> Iterator[None]:
    original_cwd = Path.cwd()
    root_text = os.fspath(root)
    original_path = list(sys.path)
    sys.path[:] = [root_text, *(entry for entry in original_path if entry != root_text)]
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original_path


def build_project_app(target: str, *, command: str = "Project") -> CayuApp:
    try:
        factory = load_target(
            target,
            label=f"{command} factory target",
            normalize_errors=True,
        )
    except TargetResolutionError as exc:
        raise ProjectError(str(exc)) from exc
    if not callable(factory):
        raise ProjectError(
            f"{command} factory target must be a callable, not an application object."
        )
    if inspect.iscoroutinefunction(factory):
        raise ProjectError(
            f"{command} factory must be synchronous; async factories are not supported."
        )

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError) as exc:
        raise ProjectError(f"{command} factory must expose a zero-argument signature.") from exc
    required = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if required:
        names = ", ".join(parameter.name for parameter in required)
        raise ProjectError(f"{command} factory must not require arguments (required: {names}).")

    app = factory()
    if inspect.iscoroutine(app):
        # The coroutine is rejected below, but closing it preserves the invariant
        # that invalid async factories do not emit a "never awaited" warning.
        app.close()
    if inspect.isawaitable(app):
        raise ProjectError(
            f"{command} factory returned an awaitable; async factories are not supported."
        )
    if not isinstance(app, CayuApp):
        raise ProjectError(f"{command} factory must return a CayuApp.")
    return app
