"""Check, without starting it, that a `cayu worker` command can start where it is deployed.

`cayu worker NAME` starts one process per CPU unless its command, its environment or
`[tool.cayu.worker_processes]` says otherwise, and a worker that passes
``reconcile_abandoned_after_s`` to ``run_task_worker`` refuses to run in more than one.
:func:`worker_command_findings` resolves the process count with the command's own
parser and rules, reads the worker's entrypoint source without importing it, and
reports a worker that would refuse to start. `cayu check --deploy` runs it on the
`[worker]` command of `cayu-cloud.toml`.
"""

from __future__ import annotations

import ast
import shlex
from collections.abc import Mapping
from pathlib import Path

from cayu.cli.project import ProjectError
from cayu.cli.worker import (
    WORKER_PROCESSES_ENV,
    WorkerError,
    parse_worker_arguments,
    resolve_worker_project,
    worker_process_plan,
)
from cayu.runtime._process_workers import single_process_worker_requirement
from cayu.runtime.checks import DiagnosticSeverity, ProjectDiagnostic

# The run_task_worker option that declares the only worker for its tasks; the worker
# then refuses to run in a multi-process group (require_single_process_worker).
_SINGLE_PROCESS_OPTION = "reconcile_abandoned_after_s"


def is_cayu_worker_command(command: str | None) -> bool:
    return _words(command)[:2] == ("cayu", "worker")


def worker_command_findings(
    root: Path,
    command: str,
    environment: Mapping[str, str],
    *,
    path: str = "cayu-cloud.toml:[worker].command",
) -> tuple[ProjectDiagnostic, ...]:
    """What would stop ``command``, a `cayu worker` command, starting in ``environment``.

    ``environment`` is the one the worker runs with, such as `cayu-cloud.toml`'s
    `[env]`, not this shell's. A process count left to one per CPU counts as several,
    since the CPUs of the machine it runs on aren't known here. Only the worker's own
    entrypoint module is read: the entrypoint function and the functions of that
    module it calls by name, or the whole module when the function isn't found there.
    """

    if not is_cayu_worker_command(command):
        return ()
    try:
        args = parse_worker_arguments(_words(command)[2:])
        project = resolve_worker_project(args.name, start=root)
        plan = worker_process_plan(args, project, environment)
    except (ProjectError, WorkerError) as exc:
        return (
            _finding(
                "WORKER_COMMAND_INVALID",
                path=path,
                message=f"`{command}` would refuse to start: {exc}",
                hint="Fix the command or the worker's configuration in pyproject.toml.",
                anchor="worker-command-invalid",
            ),
        )
    if plan.count == 1:
        return ()
    module = _passes_single_process_option(project.root, project.worker_target)
    if module is None:
        return ()
    started = (
        "one process per CPU (the default; the CPUs where it runs aren't known here)"
        if plan.count is None
        else f"{plan.count} processes (from {plan.requested.source})"
    )
    return (
        _finding(
            "WORKER_SINGLE_PROCESS_REQUIRED",
            path=path,
            message=(
                f"Worker {args.name!r} ({project.worker_target}) passes "
                f"{_SINGLE_PROCESS_OPTION} in {module}, but `{command}` would start "
                f"{started}, so it would exit at startup. "
                f"{single_process_worker_requirement(_SINGLE_PROCESS_OPTION)}"
            ),
            hint=_single_process_fix(args.name, plan.requested.source),
            anchor="worker-single-process-required",
        ),
    )


def _single_process_fix(name: str, source: str) -> str:
    """The fix that takes effect where the count came from: the flag wins over the
    environment, which wins over pyproject."""

    command = f"Set --processes 1 in the command (`cayu worker {name} --processes 1`)"
    if source == "--processes":
        return f"{command}."
    if source == WORKER_PROCESSES_ENV:
        return f"{command}, or set {WORKER_PROCESSES_ENV} to 1."
    return f"{command}, or set {name} = 1 in [tool.cayu.worker_processes] in pyproject.toml."


def _words(command: str | None) -> tuple[str, ...]:
    try:
        return tuple(shlex.split(command or ""))
    except ValueError:
        return tuple((command or "").split())


def _finding(code: str, *, path: str, message: str, hint: str, anchor: str) -> ProjectDiagnostic:
    return ProjectDiagnostic(
        code=code,
        severity=DiagnosticSeverity.ERROR,
        subject="cayu worker",
        path=path,
        message=message,
        hint=hint,
        tags=("deploy",),
        documentation_anchor=f"cayu guide diagnostics#{anchor}",
        verification_command="cayu check --deploy",
    )


def _passes_single_process_option(root: Path, target: str) -> str | None:
    """The project module, relative to ``root``, where ``target`` passes the option."""

    module, _, function = target.partition(":")
    source = _module_source(root, module.strip())
    if source is None:
        return None
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError, ValueError):
        return None
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    constants = _module_constants(tree)
    entry = functions.get(function.strip())
    pending: list[ast.AST] = [tree if entry is None else entry]
    visited = {function.strip()}
    while pending:
        for node in ast.walk(pending.pop()):
            if _sets_option(node, constants):
                return source.relative_to(root).as_posix()
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in functions
                and node.func.id not in visited
            ):
                visited.add(node.func.id)
                pending.append(functions[node.func.id])
    return None


def _sets_option(node: ast.AST, constants: dict[str, ast.expr]) -> bool:
    """A keyword or dictionary entry giving the option a value that can't be ``None``."""

    if isinstance(node, ast.keyword):
        return node.arg == _SINGLE_PROCESS_OPTION and _never_none(node.value, constants)
    if isinstance(node, ast.Dict):
        return any(
            isinstance(key, ast.Constant)
            and key.value == _SINGLE_PROCESS_OPTION
            and _never_none(value, constants)
            for key, value in zip(node.keys, node.values, strict=True)
        )
    return False


def _never_none(
    node: ast.expr, constants: dict[str, ast.expr], seen: frozenset[str] = frozenset()
) -> bool:
    """Whether ``node`` is a value known here that isn't ``None``.

    A literal, arithmetic on such values, ``int(...)`` or ``float(...)``, or a name the
    module assigns once at top level to one of those. Anything else, such as a
    parameter or a setting, may be ``None`` at run time and doesn't count, so the
    check never blocks a worker for a value it can't see.
    """

    if isinstance(node, ast.Constant):
        return node.value is not None
    if isinstance(node, ast.UnaryOp):
        return _never_none(node.operand, constants, seen)
    if isinstance(node, ast.BinOp):
        return _never_none(node.left, constants, seen) and _never_none(node.right, constants, seen)
    if isinstance(node, ast.Call):
        return isinstance(node.func, ast.Name) and node.func.id in {"float", "int"}
    if isinstance(node, ast.Name) and node.id in constants and node.id not in seen:
        return _never_none(constants[node.id], constants, seen | {node.id})
    return False


def _module_constants(tree: ast.Module) -> dict[str, ast.expr]:
    """Names the module binds exactly once, by a top-level assignment, with their values."""

    values: dict[str, ast.expr] = {}
    bound: dict[str, int] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            targets, value = statement.targets, statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            targets, value = [statement.target], statement.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                values[target.id] = value
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound[node.id] = bound.get(node.id, 0) + 1
    return {name: value for name, value in values.items() if bound.get(name) == 1}


def _module_source(root: Path, module: str) -> Path | None:
    """The project's own source file for ``module``, without importing it."""

    parts = module.split(".")
    if not all(part.isidentifier() for part in parts):
        return None
    base = root.joinpath(*parts)
    resolved_root = root.resolve()
    for candidate in (base / "__init__.py", base.with_suffix(".py")):
        if candidate.is_file() and candidate.resolve().is_relative_to(resolved_root):
            return candidate
    return None
