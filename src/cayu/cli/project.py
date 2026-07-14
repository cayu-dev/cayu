from __future__ import annotations

import importlib
import inspect
import os
import sys
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder
from pathlib import Path
from types import ModuleType

from cayu.cli._targets import TargetResolutionError, load_target
from cayu.runtime.app import CayuApp


@dataclass(frozen=True)
class CayuProject:
    root: Path
    target: str


class ProjectError(Exception):
    """An actionable project discovery or application-factory contract error."""


def _project_import_roots(root: Path) -> set[str]:
    roots: set[str] = set()
    for child in root.iterdir():
        name = child.stem if child.is_file() and child.suffix == ".py" else child.name
        if name.isidentifier() and (child.is_dir() or child.suffix == ".py"):
            roots.add(name)
    return roots


def _remove_modules(import_roots: set[str]) -> dict[str, ModuleType]:
    removed = {
        name: module
        for name, module in sys.modules.items()
        if name.partition(".")[0] in import_roots
    }
    for name in removed:
        sys.modules.pop(name, None)
    return removed


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
    except (OSError, ValueError):
        return False
    return True


def _safe_import_path(
    entries: list[str],
    *,
    root: Path,
    original_cwd: Path,
) -> tuple[str, ...]:
    safe_entries: list[str] = []
    for entry in entries:
        candidate = Path(entry) if entry else original_cwd
        if not candidate.is_absolute():
            candidate = original_cwd / candidate
        if not _is_within(candidate, root):
            safe_entries.append(os.fspath(candidate.resolve()))
    return tuple(safe_entries)


def _module_is_from_root(module: ModuleType, root: Path) -> bool:
    locations: list[str] = []
    module_file = getattr(module, "__file__", None)
    if isinstance(module_file, str):
        locations.append(module_file)
    spec = getattr(module, "__spec__", None)
    search_locations = getattr(spec, "submodule_search_locations", None)
    if search_locations is not None:
        locations.extend(location for location in search_locations if isinstance(location, str))
    return any(_is_within(Path(location), root) for location in locations)


class _StdlibShadowGuard(MetaPathFinder):
    def __init__(self, roots: set[str], search_path: tuple[str, ...]) -> None:
        self._roots = roots
        self._search_path = search_path

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if path is not None or fullname.partition(".")[0] not in self._roots:
            return None
        spec = PathFinder.find_spec(fullname, self._search_path, target)
        if spec is None:
            raise ModuleNotFoundError(
                f"Could not resolve standard-library module {fullname!r} outside the project.",
                name=fullname,
            )
        return spec


def _install_before_path_finder(finder: MetaPathFinder) -> None:
    for index, existing in enumerate(sys.meta_path):
        if existing is PathFinder:
            sys.meta_path.insert(index, finder)
            return
    sys.meta_path.append(finder)


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
    root = root.resolve()
    root_text = os.fspath(root)
    original_path = list(sys.path)
    project_roots = _project_import_roots(root)
    stdlib_roots = project_roots.intersection(sys.stdlib_module_names)
    import_roots = project_roots.difference(stdlib_roots)
    original_modules = _remove_modules(import_roots)
    original_stdlib_modules = {
        name: module
        for name, module in sys.modules.items()
        if name.partition(".")[0] in stdlib_roots
    }
    shadowed_roots = {
        name.partition(".")[0]
        for name, module in original_stdlib_modules.items()
        if _module_is_from_root(module, root)
    }
    for name in original_stdlib_modules:
        if name.partition(".")[0] in shadowed_roots:
            sys.modules.pop(name, None)
    stdlib_guard = _StdlibShadowGuard(
        stdlib_roots,
        _safe_import_path(original_path, root=root, original_cwd=original_cwd),
    )
    _install_before_path_finder(stdlib_guard)
    try:
        sys.path[:] = [root_text, *(entry for entry in original_path if entry != root_text)]
        os.chdir(root)
        importlib.invalidate_caches()
        yield
    finally:
        # Remove both replacements for modules present at entry and project modules first
        # imported inside the context. Then restore the exact entry snapshot.
        if stdlib_guard in sys.meta_path:
            sys.meta_path.remove(stdlib_guard)
        _remove_modules(import_roots)
        _remove_modules(stdlib_roots)
        sys.modules.update(original_modules)
        sys.modules.update(original_stdlib_modules)
        os.chdir(original_cwd)
        sys.path[:] = original_path
        importlib.invalidate_caches()


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
